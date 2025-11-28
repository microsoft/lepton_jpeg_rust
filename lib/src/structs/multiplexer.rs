//! Implements a multiplexer that reads and writes blocks to a stream from multiple partitions. Each
//! partition can run on it own thread to allow for increased parallelism when processing large images.
//!
//! The write implementation identifies the blocks by partition_id and tries to write in 64K blocks. The file
//! ends up with an interleaved stream of blocks from each partition.
//!
//! The read implementation reads the blocks from the file and sends them to the appropriate worker thread
//! for the partition.

use std::cmp;
use std::collections::VecDeque;
use std::io::{Cursor, Read, Write};
use std::mem::swap;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex};

use byteorder::WriteBytesExt;

use super::simple_threadpool::LeptonThreadPool;

use crate::lepton_error::{AddContext, ExitCode, Result};
use crate::{LeptonError, Metrics};
use crate::{helpers::*, lepton_error::err_exit_code, structs::partial_buffer::PartialBuffer};

/// The message that is sent between the threads
enum Message {
    Eof(usize),
    WriteBlock(usize, Vec<u8>),
}

pub struct MultiplexWriter {
    partition_id: usize,
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
                .send(Message::WriteBlock(self.partition_id, new_buffer))
                .unwrap();
        }
        Ok(())
    }
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
    fn send_results<T: FnOnce() -> Result<RESULT> + Send + 'static>(
        &mut self,
        f: T,
    ) -> impl FnOnce() + use<RESULT, T> {
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

/// Given an arbitrary writer, this function will launch the given number of partitions and call the processor function
/// on each of them, and collect the output written by each partition to the writer in blocks identified by the partition_id.
///
/// This output stream can be processed by multiple_read to get the data back, using the same number of threads.
pub fn multiplex_write<WRITE, FN, RESULT>(
    writer: &mut WRITE,
    num_partitions: usize,
    max_processor_threads: usize,
    thread_pool: &dyn LeptonThreadPool,
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

    let mut work: VecDeque<Box<dyn FnOnce() + Send>> = VecDeque::new();

    for partition_id in 0..num_partitions {
        let (tx, rx) = channel();

        let mut thread_writer = MultiplexWriter {
            partition_id,
            sender: tx,
            buffer: Vec::with_capacity(WRITE_BUFFER_SIZE),
        };

        let processor_clone = arc_processor.clone();

        let f = Box::new(thread_results.send_results(move || {
            let r = processor_clone(&mut thread_writer, partition_id)?;

            thread_writer.flush().context()?;

            thread_writer
                .sender
                .send(Message::Eof(partition_id))
                .context()?;
            Ok(r)
        }));
        work.push_back(f);

        packet_receivers.push(rx);
    }

    drop(arc_processor);

    if thread_pool.max_parallelism() > 1 {
        spawn_processor_threads(thread_pool, max_processor_threads, work);
    } else {
        // single threaded, just run all the work inline, which will
        // fill build up the receiver queue to write the image
        for f in work.drain(..) {
            f();
        }
    }

    // now we have all the threads running, we can write the data to the writer
    // carusel through the threads and write the data to the writer so that they
    // get written in a deterministic order.
    let mut current_thread_writer = 0;
    loop {
        match packet_receivers[current_thread_writer].recv() {
            Ok(Message::WriteBlock(partition_id, b)) => {
                // block length and partition header
                let tid = partition_id as u8;
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
/// The partition_id is used only to assert that we are only
/// getting the data that we are expecting.
pub struct MultiplexReader {
    /// the multiplexed thread stream we are processing
    partition_id: usize,

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
                            tid, self.partition_id,
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
/// running on its own thread. The processor function is called with the partition_id and
/// a blocking reader that it can use to read its own data.
///
/// Once the multiplexed data is finished reading, we break the channel to the worker threads
/// causing processor that is trying to read from the channel to error out and exit. After all
/// the readers have exited, we collect the results/errors from all the processors and return a vector
/// of the results back to the caller.
pub struct MultiplexReaderState<RESULT> {
    sender_channels: Vec<Sender<Message>>,
    receiver_channels: Vec<Receiver<MultiplexReadResult<RESULT>>>,
    retention_bytes: usize,
    current_state: State,
    single_thread_work: Option<VecDeque<Box<dyn FnOnce() + Send>>>,
    merged_metrics: Metrics,
}

enum State {
    StartBlock,
    U16Length(u8),
    Block(u8, usize),
}

pub enum MultiplexReadResult<RESULT> {
    Result(RESULT),
    Error(LeptonError),
    Complete(Metrics),
}

/// Given a number of threads, this function will create a multiplexed reader state that
/// can be used to process incoming multiplexed data. The processor function is called
/// on each thread with the partition_id and a blocking reader that it can use to read its own data.
///
/// Each processor is also given a sender channel that it can use to send back results or errors.
/// Partial results can be sent back by sending multiple results before the end of file is reached.
///
/// The state object returned can be used to process incoming data and retrieve results/errors
/// from the threads.
pub fn multiplex_read<FN, RESULT>(
    num_partitions: usize,
    max_processor_threads: usize,
    thread_pool: &dyn LeptonThreadPool,
    retention_bytes: usize,
    processor: FN,
) -> MultiplexReaderState<RESULT>
where
    FN: Fn(usize, &mut MultiplexReader, &Sender<MultiplexReadResult<RESULT>>) -> Result<()>
        + Send
        + Sync
        + 'static,
    RESULT: Send + 'static,
{
    let arc_processor = Arc::new(Box::new(processor));

    let mut channel_to_sender = Vec::new();

    // collect the worker threads in a queue so we can spawn them
    let mut work = VecDeque::new();
    let mut result_receiver = Vec::new();

    for partition_id in 0..num_partitions {
        let (tx, rx) = channel::<Message>();
        channel_to_sender.push(tx);

        let cloned_processor = arc_processor.clone();

        let (result_tx, result_rx) = channel::<MultiplexReadResult<RESULT>>();
        result_receiver.push(result_rx);

        let f: Box<dyn FnOnce() + Send> = Box::new(move || {
            // get the appropriate receiver so we can read out data from it
            let mut proc_reader = MultiplexReader {
                partition_id,
                current_buffer: Cursor::new(Vec::new()),
                receiver: rx,
                end_of_file: false,
            };

            if let Err(e) =
                catch_unwind_result(|| cloned_processor(partition_id, &mut proc_reader, &result_tx))
            {
                let _ = result_tx.send(MultiplexReadResult::Error(e));
            }
        });

        work.push_back(f);
    }

    let single_thread_work = if thread_pool.max_parallelism() > 1 {
        spawn_processor_threads(thread_pool, max_processor_threads, work);
        None
    } else {
        Some(work)
    };

    MultiplexReaderState {
        sender_channels: channel_to_sender,
        receiver_channels: result_receiver,
        current_state: State::StartBlock,
        retention_bytes,
        single_thread_work,
        merged_metrics: Metrics::default(),
    }
}

/// spawns the processor threads to handle the work items in the queue. There may be fewer workers
/// than work items.
fn spawn_processor_threads(
    thread_pool: &dyn LeptonThreadPool,
    max_processor_threads: usize,
    work: VecDeque<Box<dyn FnOnce() + Send>>,
) {
    let work_threads = work.len().min(max_processor_threads);
    let shared_queue = Arc::new(Mutex::new(work));

    // spawn the worker threads to process all the items
    // (there may be less processor threads than the number of threads in the image)
    for _i in 0..work_threads {
        let q = shared_queue.clone();

        thread_pool.run(Box::new(move || {
            loop {
                // do this to make sure the lock gets
                let w = q.lock().unwrap().pop_front();

                if let Some(f) = w {
                    f();
                } else {
                    break;
                }
            }
        }));
    }
}

impl<RESULT> MultiplexReaderState<RESULT> {
    /// process as much incoming data as we can and send it to the appropriate thread
    pub fn process_buffer(&mut self, source: &mut PartialBuffer<'_>) -> Result<()> {
        while source.continue_processing() {
            match self.current_state {
                State::StartBlock => {
                    if let Some(a) = source.take_n::<1>(self.retention_bytes) {
                        let thread_marker = a[0];

                        let partition_id = thread_marker & 0xf;

                        if usize::from(partition_id) >= self.sender_channels.len() {
                            return err_exit_code(
                                ExitCode::BadLeptonFile,
                                format!("invalid partition_id {0}", partition_id),
                            );
                        }

                        if thread_marker < 16 {
                            self.current_state = State::U16Length(partition_id);
                        } else {
                            let flags = (thread_marker >> 4) & 3;
                            self.current_state = State::Block(partition_id, 1024 << (2 * flags));
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
                State::Block(partition_id, data_length) => {
                    if let Some(a) = source.take(data_length, self.retention_bytes) {
                        // ignore if we get error sending because channel died since we will collect
                        // the error later. We don't want to interrupt the other threads that are processing
                        // so we only get the error from the thread that actually errored out.
                        let tid = usize::from(partition_id);
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

    /// retrieves the next available result from the threads. If complete is true, this function
    /// will block until all threads are complete and return the first result or error it finds.
    /// If complete is false, this function will return immediately if no results are available.
    pub fn retrieve_result(&mut self, complete: bool) -> Result<Option<RESULT>> {
        if let Some(value) =
            Self::try_get_result(&mut self.receiver_channels, &mut self.merged_metrics)?
        {
            return Ok(Some(value));
        }

        if complete {
            // if we are complete, send eof to all threads
            for partition_id in 0..self.sender_channels.len() {
                // send eof to all threads (ignore results since they might be dead already)
                let _ = self.sender_channels[partition_id].send(Message::Eof(partition_id));
            }
            self.sender_channels.clear();

            // if we are running single threaded, now do all the work since we've buffered up everything
            // and broken the sender channels, so there's no danger of deadlock
            if let Some(single_thread_work) = &mut self.single_thread_work {
                while let Some(f) = single_thread_work.pop_front() {
                    f();

                    if let Some(value) =
                        Self::try_get_result(&mut self.receiver_channels, &mut self.merged_metrics)?
                    {
                        return Ok(Some(value));
                    }
                }
            }

            // if we are complete, then walk through all the channels to get the first result by blocking
            while let Some(r) = self.receiver_channels.get_mut(0) {
                match r.recv() {
                    Ok(v) => match v {
                        MultiplexReadResult::Result(v) => return Ok(Some(v)),
                        MultiplexReadResult::Error(e) => return Err(e),
                        MultiplexReadResult::Complete(m) => {
                            // finished, so remove it and try the next one
                            self.merged_metrics.merge_from(m);
                            self.receiver_channels.remove(0);
                        }
                    },
                    Err(e) => {
                        // channel is closed unexpectedly, clear out all channels and return error
                        self.receiver_channels.clear();
                        return Err(e.into());
                    }
                }
            }
        }
        // nothing left to read
        Ok(None)
    }

    /// tries to get a result from the receiver channels without blocking
    fn try_get_result(
        receiver_channels: &mut Vec<Receiver<MultiplexReadResult<RESULT>>>,
        metrics: &mut Metrics,
    ) -> Result<Option<RESULT>> {
        // if we aren't complete, use non-blocking to try to get some results
        // from the first thread
        while let Some(r) = receiver_channels.get_mut(0) {
            match r.try_recv() {
                Ok(v) => match v {
                    MultiplexReadResult::Result(v) => return Ok(Some(v)),
                    MultiplexReadResult::Error(e) => return Err(e),
                    MultiplexReadResult::Complete(m) => {
                        // finished, so remove it and try the next one
                        metrics.merge_from(m);
                        receiver_channels.remove(0);
                    }
                },
                Err(TryRecvError::Disconnected) => {
                    // finished, so remove it and try the next one
                    return Err(LeptonError::new(
                        ExitCode::AssertionFailure,
                        "multiplexed reader channel disconnected unexpectedly",
                    ));
                }
                Err(TryRecvError::Empty) => {
                    // no result yet, exit loop without result
                    break;
                }
            }
        }
        Ok(None)
    }

    /// takes the merged metrics from all the threads
    pub fn take_metrics(&mut self) -> Metrics {
        std::mem::take(&mut self.merged_metrics)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use byteorder::ReadBytesExt;

    use super::*;
    use crate::lepton_error::{ExitCode, LeptonError};
    use crate::{DEFAULT_THREAD_POOL, SingleThreadPool};

    /// simple end to end test that write the thread id and reads it back
    #[test]
    fn test_multiplex_end_to_end() {
        let mut output = Vec::new();

        let w = multiplex_write(
            &mut output,
            10,
            10,
            &DEFAULT_THREAD_POOL,
            |writer, partition_id| -> Result<usize> {
                for i in partition_id as u32..10000 {
                    writer.write_u32::<byteorder::LittleEndian>(i)?;
                }

                Ok(partition_id)
            },
        )
        .unwrap();

        assert_eq!(w[..], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

        for max_processor_threads in 1..=10 {
            test_read(&output, &w, max_processor_threads);
        }
    }

    fn test_read(output: &[u8], w: &[usize], max_processor_threads: usize) {
        let mut extra = Vec::new();
        let single = SingleThreadPool::default();

        let mut multiplex_state = multiplex_read(
            10,
            max_processor_threads,
            if max_processor_threads == 1 {
                // for a single thread we shouldn't spawn any threads
                &single
            } else {
                &DEFAULT_THREAD_POOL
            },
            0,
            |partition_id, reader, result_tx: &Sender<MultiplexReadResult<usize>>| {
                for i in partition_id as u32..10000 {
                    let read_partition_id = reader.read_u32::<byteorder::LittleEndian>()?;
                    assert_eq!(read_partition_id, i);
                }
                result_tx.send(MultiplexReadResult::Result(partition_id))?;

                let mut metrics = Metrics::default();
                metrics.record_cpu_worker_time(Duration::new(1, 0));

                result_tx.send(MultiplexReadResult::Complete(metrics))?;
                Ok(())
            },
        );

        // do worst case, we are just given byte at a time
        let mut r = Vec::new();

        for i in 0..output.len() {
            let mut i = PartialBuffer::new(&output[i..=i], &mut extra);
            multiplex_state.process_buffer(&mut i).unwrap();

            if let Some(res) = multiplex_state.retrieve_result(false).unwrap() {
                r.push(res);
            }
        }

        while let Some(res) = multiplex_state.retrieve_result(true).unwrap() {
            r.push(res);
        }

        let metrics = multiplex_state.take_metrics();
        assert_eq!(metrics.get_cpu_time_worker_time(), Duration::new(10, 0));

        assert_eq!(r[..], w[..]);
    }

    #[test]
    fn test_multiplex_read_error() {
        let mut multiplex_state = multiplex_read(
            10,
            10,
            &DEFAULT_THREAD_POOL,
            0,
            |_, _, _: &Sender<MultiplexReadResult<()>>| -> Result<()> {
                Err(LeptonError::new(ExitCode::FileNotFound, "test error"))?
            },
        );

        let e: LeptonError = multiplex_state.retrieve_result(true).unwrap_err().into();
        assert_eq!(e.exit_code(), ExitCode::FileNotFound);
        assert!(e.message().starts_with("test error"));
    }

    #[test]
    fn test_multiplex_read_panic() {
        let mut multiplex_state = multiplex_read(
            10,
            10,
            &DEFAULT_THREAD_POOL,
            0,
            |_, _, _: &Sender<MultiplexReadResult<()>>| -> Result<()> {
                panic!();
            },
        );

        let e: LeptonError = multiplex_state.retrieve_result(true).unwrap_err().into();
        assert_eq!(e.exit_code(), ExitCode::AssertionFailure);
    }

    // test catching errors in the multiplex_write function
    #[test]
    fn test_multiplex_write_error() {
        let mut output = Vec::new();

        let e: LeptonError = multiplex_write(
            &mut output,
            10,
            10,
            &DEFAULT_THREAD_POOL,
            |_, partition_id| -> Result<usize> {
                if partition_id == 3 {
                    // have one partition fail
                    Err(LeptonError::new(ExitCode::FileNotFound, "test error"))?
                } else {
                    Ok(0)
                }
            },
        )
        .unwrap_err()
        .into();

        assert_eq!(e.exit_code(), ExitCode::FileNotFound);
        assert!(e.message().starts_with("test error"));
    }

    // test catching errors in the multiplex_write function
    #[test]
    fn test_multiplex_write_panic() {
        let mut output = Vec::new();

        let e: LeptonError = multiplex_write(
            &mut output,
            10,
            10,
            &DEFAULT_THREAD_POOL,
            |_, partition_id| -> Result<usize> {
                if partition_id == 5 {
                    panic!();
                }
                Ok(0)
            },
        )
        .unwrap_err()
        .into();

        assert_eq!(e.exit_code(), ExitCode::AssertionFailure);
    }
}
