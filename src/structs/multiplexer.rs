/// Implements a multiplexer that reads and writes blocks to a stream from multiple threads.
///
/// The write implementation identifies the blocks by thread_id and tries to write in 64K blocks. The file
/// ends up with an interleaved stream of blocks from each thread.
///
/// The read implementation reads the blocks from the file and sends them to the appropriate worker thread.
use crate::{helpers::*, ExitCode};
use anyhow::{Context, Result};
use byteorder::{ReadBytesExt, WriteBytesExt};
use std::{
    cmp,
    io::{Cursor, Read, Write},
    mem::swap,
    sync::mpsc::{channel, Receiver, SendError, Sender},
};

/// The message that is sent between the threads
enum Message {
    Eof,
    WriteBlock(u8, Vec<u8>),
}

pub struct MultiplexWriter {
    thread_id: u8,
    sender: Sender<Message>,
    buffer: Vec<u8>,
}

const WRITE_BUFFER_SIZE: usize = 65536;

impl Write for MultiplexWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut copy_start = 0;
        while copy_start < buf.len() {
            let amount_to_copy = cmp::min(
                WRITE_BUFFER_SIZE - self.buffer.len(),
                buf.len() - copy_start,
            );
            self.buffer
                .extend_from_slice(&buf[copy_start..copy_start + amount_to_copy]);

            if self.buffer.len() == WRITE_BUFFER_SIZE {
                self.flush()?;
            }

            copy_start += amount_to_copy;
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.buffer.len() > 0 {
            let mut new_buffer = Vec::with_capacity(WRITE_BUFFER_SIZE);
            swap(&mut new_buffer, &mut self.buffer);

            self.sender
                .send(Message::WriteBlock(self.thread_id, new_buffer))
                .unwrap();
        }
        Ok(())
    }
}

// if we are using Rayon, these are the primatives to use to spawn thread pool work items
#[cfg(feature = "use_rayon")]
fn my_scope<'scope, OP, R>(op: OP) -> R
where
    OP: FnOnce(&rayon_core::Scope<'scope>) -> R,
{
    rayon_core::in_place_scope(op)
}

#[cfg(feature = "use_rayon")]
fn my_spawn<'scope, BODY>(s: &rayon_core::Scope<'scope>, body: BODY)
where
    BODY: FnOnce() + Send + 'scope,
{
    s.spawn(|_| body())
}

// if we are not using Rayon, just spawn regular threads
#[cfg(not(feature = "use_rayon"))]
fn my_scope<'env, F, T>(f: F) -> T
where
    F: for<'scope> FnOnce(&'scope std::thread::Scope<'scope, 'env>) -> T,
{
    std::thread::scope::<'env, F, T>(f)
}

#[cfg(not(feature = "use_rayon"))]
fn my_spawn<'scope, F, T>(s: &'scope std::thread::Scope<'scope, '_>, f: F)
where
    F: FnOnce() -> T + Send + 'scope,
    T: Send + 'scope,
{
    s.spawn::<F, T>(f);
}

/// Given an arbitrary writer, this function will launch the given number of threads and call the processor function
/// on each of them, and collect the output written by each thread to the writer in blocks identified by the thread_id.
///
/// This output stream can be processed by multiple_read to get the data back, using the same number of threads.
pub fn multiplex_write<WRITE, FN, RESULT>(
    writer: &mut WRITE,
    num_threads: usize,
    processor: FN,
) -> Result<Vec<RESULT>>
where
    WRITE: Write,
    FN: Fn(&mut MultiplexWriter, usize) -> Result<RESULT> + Send + Copy,
    RESULT: Send,
{
    let mut thread_results = Vec::new();
    for _i in 0..num_threads {
        thread_results.push(None);
    }

    my_scope(|s| -> Result<()> {
        let (tx, rx) = channel();

        for (thread_id, result) in thread_results.iter_mut().enumerate() {
            let cloned_sender = tx.clone();

            let mut thread_writer = MultiplexWriter {
                thread_id: thread_id as u8,
                sender: cloned_sender,
                buffer: Vec::with_capacity(WRITE_BUFFER_SIZE),
            };

            let mut f = move || -> Result<RESULT> {
                let r = processor(&mut thread_writer, thread_id)?;

                thread_writer.flush().context(here!())?;

                thread_writer.sender.send(Message::Eof).context(here!())?;
                Ok(r)
            };

            my_spawn(s, move || {
                *result = Some(f());
            });
        }

        // drop the sender so that the channel breaks when all the threads exit
        drop(tx);

        // wait to collect work and done messages from all the threads
        let mut threads_left = num_threads;

        while threads_left > 0 {
            let value = rx.recv().context(here!());
            match value {
                Ok(Message::Eof) => {
                    threads_left -= 1;
                }
                Ok(Message::WriteBlock(thread_id, b)) => {
                    let l = b.len() - 1;

                    writer.write_u8(thread_id).context(here!())?;
                    writer.write_u8((l & 0xff) as u8).context(here!())?;
                    writer.write_u8(((l >> 8) & 0xff) as u8).context(here!())?;
                    writer.write_all(&b[..]).context(here!())?;
                }
                Err(_) => {
                    // if we get a receiving error here, this means that one of the threads broke
                    // with an error, and this error will be collected when we join the threads
                    break;
                }
            }
        }

        // in place scope will join all the threads before it exits
        return Ok(());
    })
    .context(here!())?;

    let mut thread_not_run = false;
    let mut results = Vec::new();

    for result in thread_results.drain(..) {
        match result {
            None => thread_not_run = true,
            Some(Ok(r)) => results.push(r),
            // if there was an error processing anything, return it
            Some(Err(e)) => return Err(e),
        }
    }

    if thread_not_run {
        return err_exit_code(ExitCode::GeneralFailure, "thread did not run");
    }

    Ok(results)
}

/// Used by the processor thread to read data in a blocking way.
/// The thread_id is used only to assert that we are only
/// getting the data that we are expecting.
pub struct MultiplexReader {
    /// the multiplexed thread stream we are processing
    thread_id: u8,

    /// the receiver part of the channel to get more buffers
    receiver: Receiver<Message>,

    /// what we are reading. When this returns zero, we try to
    /// refill the buffer if we haven't reached the end of the stream
    current_buffer: Cursor<Vec<u8>>,

    /// once we get told we are at the end of the stream, we just
    /// always return 0 bytes
    end_of_file: bool,
}

impl Read for MultiplexReader {
    /// fast path for reads. If we run out of data, take the slow path
    #[inline(always)]
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let amount_read = self.current_buffer.read(buf)?;
        if amount_read > 0 {
            return Ok(amount_read);
        }

        self.read_slow(buf)
    }
}

impl MultiplexReader {
    /// slow path for reads, try to get a new buffer or
    /// return zero if at the end of the stream
    #[cold]
    #[inline(never)]
    fn read_slow(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        while !self.end_of_file {
            let amount_read = self.current_buffer.read(buf)?;
            if amount_read > 0 {
                return Ok(amount_read);
            }

            match self.receiver.recv() {
                Ok(r) => match r {
                    Message::Eof => {
                        self.end_of_file = true;
                    }
                    Message::WriteBlock(tid, block) => {
                        debug_assert_eq!(
                            tid, self.thread_id,
                            "incoming thread must be equal to processing thread"
                        );
                        self.current_buffer = Cursor::new(block);
                    }
                },
                Err(e) => {
                    return Result::Err(std::io::Error::new(std::io::ErrorKind::Other, e));
                }
            }
        }

        // nothing if we reached the end of file
        return Ok(0);
    }
}

/// Reads data in multiplexed format and sends it to the appropriate processor, each
/// running on its own thread. The processor function is called with the thread_id and
/// a blocking reader that it can use to read its own data.
///
/// Once the multiplexed data is finished reading, we break the channel to the worker threads
/// causing processor that is trying to read from the channel to error out and exit. After all
/// the readers have exited, we collect the results/errors from all the processors and return a vector
/// of the results back to the caller.
pub fn multiplex_read<READ, FN, RESULT>(
    reader: &mut READ,
    num_threads: usize,
    processor: FN,
) -> Result<Vec<RESULT>>
where
    READ: Read,
    FN: Fn(usize, &mut MultiplexReader) -> Result<RESULT> + Send + Copy,
    RESULT: Send,
{
    // track if we got an error while trying to send to a thread
    let mut error_sending: Option<SendError<Message>> = None;

    let mut thread_results = Vec::<Option<Result<RESULT>>>::new();
    for _i in 0..num_threads {
        thread_results.push(None);
    }

    my_scope(|s| -> Result<()> {
        let mut channel_to_sender = Vec::new();

        // create a channel for each stream and spawn a work item to read from it
        // the return value from each work item is stored in thread_results, which
        // is collected at the end
        for (thread_id, result) in thread_results.iter_mut().enumerate() {
            let (tx, rx) = channel();
            channel_to_sender.push(tx);

            my_spawn(s, move || {
                // get the appropriate receiver so we can read out data from it
                let mut proc_reader = MultiplexReader {
                    thread_id: thread_id as u8,
                    current_buffer: Cursor::new(Vec::new()),
                    receiver: rx,
                    end_of_file: false,
                };
                *result = Some(processor(thread_id, &mut proc_reader));
            });
        }

        // now that the channels are waiting for input, read the stream and send all the buffers to their respective readers
        loop {
            let mut thread_marker_a = [0; 1];
            if reader.read(&mut thread_marker_a)? == 0 {
                break;
            }

            let thread_marker = thread_marker_a[0];

            let thread_id = (thread_marker & 0xf) as u8;

            if thread_id >= channel_to_sender.len() as u8 {
                return err_exit_code(
                    ExitCode::BadLeptonFile,
                    format!("invalid thread_id {0}", thread_id).as_str(),
                );
            }

            let data_length = if thread_marker < 16 {
                let b0 = reader.read_u8().context(here!())?;
                let b1 = reader.read_u8().context(here!())?;

                ((b1 as usize) << 8) + b0 as usize + 1
            } else {
                // This format is used by Lepton C++ to write encoded chunks with length of 4096, 16384 or 65536 bytes
                let flags = (thread_marker >> 4) & 3;

                1024 << (2 * flags)
            };

            //info!("offset {0} len {1}", reader.stream_position()?-2, data_length);

            let mut buffer = vec![0; data_length as usize];

            reader
                .read_exact(&mut buffer)
                .with_context(|| format!("reading {0} bytes", buffer.len()))?;

            let e =
                channel_to_sender[thread_id as usize].send(Message::WriteBlock(thread_id, buffer));

            if let Err(e) = e {
                error_sending = Some(e);
                break;
            }
        }
        //info!("done sending!");

        for c in channel_to_sender {
            // ignore the result of send, since a thread may have already blown up with an error and we will get it when we join (rather than exiting with a useless channel broken message)
            let _ = c.send(Message::Eof);
        }

        Ok(())
    })?;

    let mut result = Vec::new();
    let mut thread_not_run = false;
    for i in thread_results.drain(..) {
        match i {
            None => thread_not_run = true,
            Some(Err(e)) => {
                return Err(e).context(here!());
            }
            Some(Ok(r)) => {
                result.push(r);
            }
        }
    }

    if thread_not_run {
        return err_exit_code(ExitCode::GeneralFailure, "thread did not run").context(here!());
    }

    // if there was an error during send, it should have resulted in an error from one of the threads above and
    // we wouldn't get here, but as an extra precaution, we check here to make sure we didn't miss anything
    if let Some(e) = error_sending {
        return Err(e).context(here!());
    }

    Ok(result)
}

/// simple end to end test that write the thread id and reads it back
#[test]
fn test_multiplex_end_to_end() {
    let mut output = Vec::new();

    let w = multiplex_write(&mut output, 10, |writer, thread_id| -> Result<usize> {
        writer.write_u32::<byteorder::LittleEndian>(thread_id as u32)?;

        Ok(thread_id)
    })
    .unwrap();

    assert_eq!(w[..], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    let mut reader = Cursor::new(output);

    let r = multiplex_read(&mut reader, 10, |thread_id, reader| -> Result<usize> {
        let read_thread_id = reader.read_u32::<byteorder::LittleEndian>()?;
        assert_eq!(read_thread_id, thread_id as u32);
        Ok(thread_id)
    })
    .unwrap();

    assert_eq!(r[..], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
