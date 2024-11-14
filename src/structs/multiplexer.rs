use std::cmp;
use std::collections::VecDeque;
use std::io::{Cursor, Read, Write};
use std::mem::swap;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

use byteorder::WriteBytesExt;

use crate::lepton_error::{AddContext, ExitCode, LeptonError, Result};
/// Implements a multiplexer that reads and writes blocks to a stream from multiple threads.
///
/// The write implementation identifies the blocks by thread_id and tries to write in 64K blocks. The file
/// ends up with an interleaved stream of blocks from each thread.
///
/// The read implementation reads the blocks from the file and sends them to the appropriate worker thread.
use crate::{helpers::*, lepton_error::err_exit_code, structs::partial_buffer::PartialBuffer};

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

// if we are not using Rayon, just spawn regular threads
#[cfg(not(feature = "use_rayon"))]
fn my_spawn_simple<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(f);
}

#[cfg(feature = "use_rayon")]
fn my_spawn_simple<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    rayon_core::spawn(f);
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
    FN: Fn(&mut MultiplexWriter, usize) -> Result<RESULT> + Send + Sync + 'static,
    RESULT: Send,
{
    let mut thread_results = Vec::new();
    for _i in 0..num_threads {
        thread_results.push(None);
    }

    my_scope(|s| -> Result<()> {
        let (tx, rx) = channel();
        let arc_processor = Arc::new(Box::new(processor));

        for (thread_id, result) in thread_results.iter_mut().enumerate() {
            let cloned_sender = tx.clone();

            let mut thread_writer = MultiplexWriter {
                thread_id: thread_id as u8,
                sender: cloned_sender,
                buffer: Vec::with_capacity(WRITE_BUFFER_SIZE),
            };

            let processor_clone = arc_processor.clone();

            let f = move || -> Result<RESULT> {
                let r = processor_clone(&mut thread_writer, thread_id)?;

                thread_writer.flush().context()?;

                thread_writer.sender.send(Message::Eof).context()?;
                Ok(r)
            };

            my_spawn(s, move || {
                *result = Some(catch_unwind_result(f));
            });
        }

        // drop the sender so that the channel breaks when all the threads exit
        drop(tx);

        // wait to collect work and done messages from all the threads
        let mut threads_left = num_threads;

        while threads_left > 0 {
            let value = rx.recv().context();
            match value {
                Ok(Message::Eof) => {
                    threads_left -= 1;
                }
                Ok(Message::WriteBlock(thread_id, b)) => {
                    // block length and thread header
                    let l = b.len() - 1;
                    if l == 4095 || l == 16383 || l == 65535 {
                        // length is a special power of 2 - standard block length is 2^16
                        writer
                            .write_u8(thread_id | ((b.len().ilog2() as u8 >> 1) - 5) << 4)
                            .context()?;
                    } else {
                        writer.write_u8(thread_id).context()?;
                        writer.write_u8((l & 0xff) as u8).context()?;
                        writer.write_u8(((l >> 8) & 0xff) as u8).context()?;
                    }
                    writer.write_all(&b[..]).context()?;
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
    .context()?;

    let mut thread_not_run = false;
    let mut results = Vec::new();

    for result in thread_results.drain(..) {
        match result {
            None => thread_not_run = true,
            Some(Ok(r)) => results.push(r),
            // if there was an error processing anything, return it
            Some(Err(e)) => return Err(e.into()),
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
                    return std::io::Result::Err(std::io::Error::new(std::io::ErrorKind::Other, e));
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
pub struct MultiplexReaderState<RESULT> {
    sender_channels: Vec<Sender<Message>>,
    result_receiver: Receiver<(usize, core::result::Result<RESULT, LeptonError>)>,
    retention_bytes: usize,
    current_state: State,
}

enum State {
    StartBlock,
    U16Length(u8),
    Block(u8, usize),
}

impl<RESULT> MultiplexReaderState<RESULT> {
    pub fn new<FN>(
        num_threads: usize,
        retention_bytes: usize,
        max_processor_threads: usize,
        processor: FN,
    ) -> MultiplexReaderState<RESULT>
    where
        FN: Fn(usize, &mut MultiplexReader) -> Result<RESULT> + Send + Sync + 'static,
        RESULT: Send + 'static,
    {
        let (result_sender, result_receiver) =
            channel::<(usize, core::result::Result<RESULT, LeptonError>)>();

        let arc_processor = Arc::new(Box::new(processor));

        let mut channel_to_sender = Vec::new();

        // collect the worker threads in a queue so we can spawn them
        let mut work = VecDeque::new();

        for thread_id in 0..num_threads {
            // create a channel for each stream and spawn a work item to read from it
            // the return value from each work item is stored in thread_results, which
            // is collected at the end
            let (tx, rx) = channel::<Message>();
            channel_to_sender.push(tx);

            let cloned_processor = arc_processor.clone();
            let cloned_result_sender = result_sender.clone();

            work.push_back((thread_id, rx, cloned_processor, cloned_result_sender));
        }

        let shared_queue = Arc::new(Mutex::new(work));

        // spawn the worker threads to process all the items
        // (there may be less processor threads than the number of threads in the image)
        for _i in 0..num_threads.min(max_processor_threads) {
            let q = shared_queue.clone();

            my_spawn_simple(move || {
                loop {
                    // do this to make sure the lock gets
                    let w = q.lock().unwrap().pop_front();

                    if let Some((thread_id, rx, cloned_processor, cloned_result_sender)) = w {
                        // get the appropriate receiver so we can read out data from it
                        let mut proc_reader = MultiplexReader {
                            thread_id: thread_id as u8,
                            current_buffer: Cursor::new(Vec::new()),
                            receiver: rx,
                            end_of_file: false,
                        };

                        let result = catch_unwind_result(move || {
                            cloned_processor(thread_id, &mut proc_reader)
                        });

                        // nothing to do if we fail to return the results (since the caller
                        // died and we can't return the results)
                        _ = cloned_result_sender.send((thread_id, result));
                    } else {
                        break;
                    }
                }
            });
        }

        MultiplexReaderState {
            sender_channels: channel_to_sender,
            result_receiver: result_receiver,
            current_state: State::StartBlock,
            retention_bytes,
        }
    }

    /// process as much incoming data as we can and send it to the appropriate thread
    pub fn process_buffer(&mut self, source: &mut PartialBuffer<'_>) -> Result<()> {
        while source.continue_processing() {
            match self.current_state {
                State::StartBlock => {
                    if let Some(a) = source.take_n::<1>(self.retention_bytes) {
                        let thread_marker = a[0];

                        let thread_id = thread_marker & 0xf;

                        if usize::from(thread_id) >= self.sender_channels.len() {
                            return err_exit_code(
                                ExitCode::BadLeptonFile,
                                format!("invalid thread_id {0}", thread_id).as_str(),
                            );
                        }

                        if thread_marker < 16 {
                            self.current_state = State::U16Length(thread_id);
                        } else {
                            let flags = (thread_marker >> 4) & 3;
                            self.current_state = State::Block(thread_id, 1024 << (2 * flags));
                        }
                    } else {
                        break;
                    }
                }
                State::U16Length(thread_marker) => {
                    if let Some(a) = source.take_n::<2>(self.retention_bytes) {
                        let b0 = usize::from(a[0]);
                        let b1 = usize::from(a[1]);

                        self.current_state = State::Block(thread_marker, (b1 << 8) + b0 + 1);
                    } else {
                        break;
                    }
                }
                State::Block(thread_id, data_length) => {
                    if let Some(a) = source.take(data_length, self.retention_bytes) {
                        // ignore if we get error sending because channel died since we will collect
                        // the error later. We don't want to interrupt the other threads that are processing
                        // so we only get the error from the thread that actually errored out.
                        let _ = self.sender_channels[usize::from(thread_id)]
                            .send(Message::WriteBlock(thread_id, a));
                        self.current_state = State::StartBlock;
                    } else {
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Called once all the incoming buffers are passed to process buffers,
    /// waits for all the threads to finish processing and returns the results.
    pub fn complete(&mut self) -> Result<Vec<RESULT>> {
        let mut results = Vec::new();
        for thread_id in 0..self.sender_channels.len() {
            // send eof to all threads (ignore results since they might be dead already)
            _ = self.sender_channels[thread_id].send(Message::Eof);
            results.push(None);
        }

        let mut error = None;
        for _i in 0..self.sender_channels.len() {
            match self.result_receiver.recv().context()? {
                (thread_id, Ok(r)) => {
                    results[thread_id] = Some(r);
                }
                (_thread_id, Err(e)) => {
                    error = Some(e);
                }
            }
        }

        if let Some(e) = error {
            Err(e).context()
        } else {
            let results: Vec<RESULT> = results.into_iter().map(|x| x.unwrap()).collect();
            Ok(results)
        }
    }
}

/// simple end to end test that write the thread id and reads it back
#[test]
fn test_multiplex_end_to_end() {
    use byteorder::ReadBytesExt;

    let mut output = Vec::new();

    let w = multiplex_write(&mut output, 10, |writer, thread_id| -> Result<usize> {
        for i in thread_id as u32..10000 {
            writer.write_u32::<byteorder::LittleEndian>(i)?;
        }

        Ok(thread_id)
    })
    .unwrap();

    assert_eq!(w[..], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    let mut extra = Vec::new();

    let mut multiplex_state =
        MultiplexReaderState::new(10, 0, 8, |thread_id, reader| -> Result<usize> {
            for i in thread_id as u32..10000 {
                let read_thread_id = reader.read_u32::<byteorder::LittleEndian>()?;
                assert_eq!(read_thread_id, i);
            }
            Ok(thread_id)
        });

    // do worst case, we are just given byte at a time
    for i in 0..output.len() {
        let mut i = PartialBuffer::new(&output[i..=i], &mut extra);
        multiplex_state.process_buffer(&mut i).unwrap();
    }

    let r = multiplex_state.complete().unwrap();

    assert_eq!(r[..], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn test_multiplex_read_error() {
    let mut multiplex_state = MultiplexReaderState::new(10, 0, 8, |_, _| -> Result<usize> {
        Err(LeptonError::new(ExitCode::FileNotFound, "test error"))?
    });

    let e: LeptonError = multiplex_state.complete().unwrap_err().into();
    assert_eq!(e.exit_code(), ExitCode::FileNotFound);
    assert!(e.message().starts_with("test error"));
}

#[test]
fn test_multiplex_read_panic() {
    let mut multiplex_state = MultiplexReaderState::new(10, 0, 8, |_, _| -> Result<usize> {
        panic!();
    });

    let e: LeptonError = multiplex_state.complete().unwrap_err().into();
    assert_eq!(e.exit_code(), ExitCode::AssertionFailure);
}

// test catching errors in the multiplex_write function
#[test]
fn test_multiplex_write_error() {
    let mut output = Vec::new();

    let e: LeptonError = multiplex_write(&mut output, 10, |_, thread_id| -> Result<usize> {
        if thread_id == 3 {
            // have one thread fail
            Err(LeptonError::new(ExitCode::FileNotFound, "test error"))?
        } else {
            Ok(0)
        }
    })
    .unwrap_err()
    .into();

    assert_eq!(e.exit_code(), ExitCode::FileNotFound);
    assert!(e.message().starts_with("test error"));
}

// test catching errors in the multiplex_write function
#[test]
fn test_multiplex_write_panic() {
    let mut output = Vec::new();

    let e: LeptonError = multiplex_write(&mut output, 10, |_, thread_id| -> Result<usize> {
        if thread_id == 5 {
            panic!();
        }
        Ok(0)
    })
    .unwrap_err()
    .into();

    assert_eq!(e.exit_code(), ExitCode::AssertionFailure);
}
