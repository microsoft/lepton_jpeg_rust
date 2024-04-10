use crate::{helpers::*, ExitCode};
use anyhow::{Context, Result};
use byteorder::{ReadBytesExt, WriteBytesExt};
use std::{
    cmp,
    io::{Cursor, Read, Seek, Write},
    mem::swap,
    sync::mpsc::{channel, Receiver, SendError, Sender},
};

/// Implements a multiplexer that reads and writes blocks to a stream from multiple threads.
///
/// The write implementation identifies the blocks by thread_id and tries to write in 64K blocks. The file
/// ends up with an interleaved stream of blocks from each thread.
///
/// The read implementation reads the blocks from the file and sends them to the appropriate worker thread.

enum Message {
    Eof,
    WriteBlock(u8, Vec<u8>),
}

pub struct MessageSender {
    thread_id: u8,
    sender: Sender<Message>,
    buffer: Vec<u8>,
}

const WRITE_BUFFER_SIZE: usize = 65536;

impl Write for MessageSender {
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

/// used by the worker thread to read data for the given thread from the
/// receiver. The thread_id is used only to assert that we are only
/// getting the data that we are expecting
pub struct MessageReceiver {
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

impl Read for MessageReceiver {
    /// fast path for reads. If we get zero bytes, take the slow path
    #[inline(always)]
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let amount_read = self.current_buffer.read(buf)?;
        if amount_read > 0 {
            return Ok(amount_read);
        }

        self.read_slow(buf)
    }
}

impl MessageReceiver {
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

pub fn multiplex_read<READ, FN, RESULT>(
    reader: &mut READ,
    num_threads: usize,
    last_data_position: u64,
    processor: FN,
) -> Result<Vec<RESULT>>
where
    READ: Read + Seek,
    FN: Fn(usize, &mut MessageReceiver) -> Result<RESULT> + Send + Copy,
    RESULT: Send,
{
    // track if we got an error while trying to send to a thread
    let mut error_sending: Option<SendError<Message>> = None;

    let mut thread_results = Vec::<Option<Result<RESULT>>>::new();
    for _i in 0..num_threads {
        thread_results.push(None);
    }

    rayon::in_place_scope(|s| -> Result<()> {
        let mut channel_to_sender = Vec::new();

        // create a channel for each stream and spawn a work item to read from it
        // the return value from each work item is stored in thread_results, which
        // is collected at the end
        for (thread_id, result) in thread_results.iter_mut().enumerate() {
            let (tx, rx) = channel();
            channel_to_sender.push(tx);

            s.spawn(move |_| {
                // get the appropriate receiver so we can read out data from it
                let mut proc_reader = MessageReceiver {
                    thread_id: thread_id as u8,
                    current_buffer: Cursor::new(Vec::new()),
                    receiver: rx,
                    end_of_file: false,
                };
                *result = Some(processor(thread_id, &mut proc_reader));
            });
        }

        // now that the channels are waiting for input, read the stream and send all the buffers to their respective readers
        while reader.stream_position().context(here!())? < last_data_position - 4 {
            let thread_marker = reader.read_u8().context(here!())?;
            let thread_id = (thread_marker & 0xf) as u8;

            if thread_id >= channel_to_sender.len() as u8 {
                return err_exit_code(
                    ExitCode::BadLeptonFile,
                    format!(
                        "invalid thread_id at {0} of {1} at {2}",
                        reader.stream_position().unwrap(),
                        last_data_position,
                        here!()
                    )
                    .as_str(),
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

            let mut buffer = Vec::<u8>::new();
            buffer.resize(data_length as usize, 0);
            reader.read_exact(&mut buffer).with_context(|| {
                format!(
                    "reading {0} bytes at {1} of {2} at {3}",
                    buffer.len(),
                    reader.stream_position().unwrap(),
                    last_data_position,
                    here!()
                )
            })?;

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

pub fn multiplex_write<WRITE, FN, RESULT>(
    writer: &mut WRITE,
    num_threads: usize,
    processor: FN,
) -> Result<Vec<RESULT>>
where
    WRITE: Write,
    FN: Fn(&mut MessageSender, usize) -> Result<RESULT> + Send + Copy,
    RESULT: Send,
{
    let mut thread_results = Vec::<Option<Result<RESULT>>>::new();

    for _i in 0..num_threads {
        thread_results.push(None);
    }

    rayon::in_place_scope(|s| -> Result<()> {
        let (tx, rx) = channel();

        for (thread_id, result) in thread_results.iter_mut().enumerate() {
            let cloned_sender = tx.clone();

            let mut thread_writer = MessageSender {
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

            s.spawn(move |_| {
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
                    break;
                }
            }
        }

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
