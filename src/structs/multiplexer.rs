use std::cmp;
use std::collections::VecDeque;
use std::io::{Cursor, Read, Write};
use std::mem::swap;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

use byteorder::WriteBytesExt;

use crate::lepton_error::{AddContext, ExitCode, Result};
/// Implements a multiplexer that reads and writes blocks to a stream from multiple threads.
///
/// The write implementation identifies the blocks by thread_id and tries to write in 64K blocks. The file
/// ends up with an interleaved stream of blocks from each thread.
///
/// The read implementation reads the blocks from the file and sends them to the appropriate worker thread.
use crate::{helpers::*, lepton_error::err_exit_code, structs::partial_buffer::PartialBuffer};

/// The message that is sent between the threads
enum Message {
    Eof(usize),
    WriteBlock(usize, Vec<u8>),
}

pub struct MultiplexWriter {
    thread_id: usize,
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

// if we are not using Rayon, just spawn regular threads
#[cfg(not(feature = "use_rayon"))]
fn my_spawn_simple<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    super::simple_threadpool::execute(f);
}

#[cfg(feature = "use_rayon")]
fn my_spawn_simple<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    rayon_core::spawn(f);
}

/// Collects the thread results and errors and returns them as a vector
struct ThreadResults<RESULT> {
    results: Vec<Receiver<Result<RESULT>>>,
}

impl<RESULT> ThreadResults<RESULT> {
    fn new() -> Self {
        ThreadResults {
            results: Vec::new(),
        }
    }
    /// creates a closure that wraps the passed in closure, catches any panics,
    /// collects the return result and send it to the receiver to collect.
    fn send_results(
        &mut self,
        f: impl FnOnce() -> Result<RESULT> + Send + 'static,
    ) -> impl FnOnce() {
        let (tx, rx) = channel();

        self.results.push(rx);

        move || {
            let r = catch_unwind_result(f);
            let _ = tx.send(r);
        }
    }

    /// extracts the results from all the receivers and returns them as a vector, or returns an
    /// error if any of the threads errored out.
    fn receive_results(&mut self) -> Result<Vec<RESULT>> {
        let mut final_results = Vec::new();

        let mut error_found = None;
        for r in self.results.drain(..) {
            match r.recv() {
                Ok(Ok(r)) => final_results.push(r),
                Ok(Err(e)) => {
                    error_found = Some(e);
                }
                Err(e) => {
                    // prefer real errors over broken channel errors
                    if error_found.is_none() {
                        error_found = Some(e.into());
                    }
                }
            }
        }

        if let Some(error) = error_found {
            Err(error)
        } else {
            Ok(final_results)
        }
    }
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
    RESULT: Send + 'static,
{
    let mut thread_results = ThreadResults::new();

    // receives packets from threads as they are generated
    let mut packet_receivers = Vec::new();

    let arc_processor = Arc::new(Box::new(processor));

    for thread_id in 0..num_threads {
        let (tx, rx) = channel();

        let mut thread_writer = MultiplexWriter {
            thread_id: thread_id,
            sender: tx,
            buffer: Vec::with_capacity(WRITE_BUFFER_SIZE),
        };

        let processor_clone = arc_processor.clone();

        let f = thread_results.send_results(move || {
            let r = processor_clone(&mut thread_writer, thread_id)?;

            thread_writer.flush().context()?;

            thread_writer
                .sender
                .send(Message::Eof(thread_id))
                .context()?;
            Ok(r)
        });

        my_spawn_simple(f);

        packet_receivers.push(rx);
    }

    // now we have all the threads running, we can write the data to the writer
    // carusel through the threads and write the data to the writer so that they
    // get written in a deterministic order.
    let mut current_thread_writer = 0;
    loop {
        match packet_receivers[current_thread_writer].recv() {
            Ok(Message::WriteBlock(thread_id, b)) => {
                // block length and thread header
                let tid = thread_id as u8;
                let l = b.len() - 1;
                if l == 4095 || l == 16383 || l == 65535 {
                    // length is a special power of 2 - standard block length is 2^16
                    writer.write_u8(tid | ((l.ilog2() as u8 >> 1) - 4) << 4)?;
                } else {
                    writer.write_u8(tid)?;
                    writer.write_u8((l & 0xff) as u8)?;
                    writer.write_u8(((l >> 8) & 0xff) as u8)?;
                }
                // block itself
                writer.write_all(&b[..])?;

                // go to next thread
                current_thread_writer = (current_thread_writer + 1) % packet_receivers.len();
            }
            Ok(Message::Eof(_)) | Err(_) => {
                packet_receivers.remove(current_thread_writer);
                if packet_receivers.len() == 0 {
                    break;
                }

                current_thread_writer = current_thread_writer % packet_receivers.len();
            }
        }
    }
    thread_results.receive_results()
}

/// Used by the processor thread to read data in a blocking way.
/// The thread_id is used only to assert that we are only
/// getting the data that we are expecting.
pub struct MultiplexReader {
    /// the multiplexed thread stream we are processing
    thread_id: usize,

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
                    Message::Eof(_tid) => {
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
    result_receiver: ThreadResults<RESULT>,
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
        let arc_processor = Arc::new(Box::new(processor));

        let mut channel_to_sender = Vec::new();

        // collect the worker threads in a queue so we can spawn them
        let mut work = VecDeque::new();
        let mut result_receiver = ThreadResults::new();

        for thread_id in 0..num_threads {
            let (tx, rx) = channel::<Message>();
            channel_to_sender.push(tx);

            let cloned_processor = arc_processor.clone();

            let f = result_receiver.send_results(move || {
                // get the appropriate receiver so we can read out data from it
                let mut proc_reader = MultiplexReader {
                    thread_id: thread_id,
                    current_buffer: Cursor::new(Vec::new()),
                    receiver: rx,
                    end_of_file: false,
                };

                cloned_processor(thread_id, &mut proc_reader)
            });

            work.push_back(f);
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

                    if let Some(f) = w {
                        f();
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
                        let tid = usize::from(thread_id);
                        let _ = self.sender_channels[tid].send(Message::WriteBlock(tid, a));
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
        for thread_id in 0..self.sender_channels.len() {
            // send eof to all threads (ignore results since they might be dead already)
            let _ = self.sender_channels[thread_id].send(Message::Eof(thread_id));
        }

        self.result_receiver.receive_results()
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

#[cfg(test)]
use crate::lepton_error::LeptonError;

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
