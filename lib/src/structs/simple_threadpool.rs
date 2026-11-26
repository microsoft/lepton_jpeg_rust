/// A simple thread pool implementation that can be used to evaluate closures on separate threads.
///
/// The pool creates at most one worker per available CPU and applies backpressure when its bounded
/// queue is full.
///
/// No unsafe code is used.
use std::{
    collections::VecDeque,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Condvar, LazyLock, Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    thread::{self, spawn},
};

use crate::consts::MAX_THREADS_SUPPORTED_BY_LEPTON_FORMAT;
use crate::lepton_error::{ExitCode, Result, err_exit_code};

type WorkItem = Box<dyn FnOnce() + Send + 'static>;

struct WorkQueue {
    work: VecDeque<WorkItem>,
    shutdown: bool,
}

struct PoolState {
    queue: Mutex<WorkQueue>,
    work_available: Condvar,
    idle_threads: AtomicUsize,
    capacity: usize,
}

struct PoolRuntime {
    state: Arc<PoolState>,
}

impl Drop for PoolRuntime {
    fn drop(&mut self) {
        self.state.queue.lock().unwrap().shutdown = true;
        self.state.work_available.notify_all();
    }
}

/// A trait that defines the interface for a Lepton thread pool.
/// It has a simple fire-and-forget interface, which is sufficient for the current use cases,
/// but also requires the thread pool to be static, since we don't require the thread
/// to return within a specific lifetime.
pub trait LeptonThreadPool {
    /// Returns the maximum parallelism supported by the thread pool.
    fn max_parallelism(&self) -> usize;
    /// Runs a closure on a thread from the thread pool. The thread
    /// thread lifetime is not specified, so it can must be static.
    fn run(&self, f: Box<dyn FnOnce() + Send + 'static>);

    /// Attempts to schedule a closure, returning an error instead of blocking when the pool is saturated.
    fn try_run(&self, f: Box<dyn FnOnce() + Send + 'static>) -> Result<()> {
        self.run(f);
        Ok(())
    }
}

/// Holds either a reference to a LeptonThreadPool or an owned Box<dyn LeptonThreadPool>.
///
/// This is useful for APIs that want to accept either a reference to a static or global thread pool
/// or an owned thread pool.
pub enum ThreadPoolHolder<'a> {
    /// Reference to a LeptonThreadPool
    Dyn(&'a dyn LeptonThreadPool),
    /// Owned Box<dyn LeptonThreadPool>
    Owned(Box<dyn LeptonThreadPool>),
}

impl LeptonThreadPool for ThreadPoolHolder<'_> {
    fn max_parallelism(&self) -> usize {
        match self {
            ThreadPoolHolder::Dyn(p) => p.max_parallelism(),
            ThreadPoolHolder::Owned(p) => p.max_parallelism(),
        }
    }
    fn run(&self, f: Box<dyn FnOnce() + Send + 'static>) {
        match self {
            ThreadPoolHolder::Dyn(p) => p.run(f),
            ThreadPoolHolder::Owned(p) => p.run(f),
        }
    }
    fn try_run(&self, f: Box<dyn FnOnce() + Send + 'static>) -> Result<()> {
        match self {
            ThreadPoolHolder::Dyn(p) => p.try_run(f),
            ThreadPoolHolder::Owned(p) => p.try_run(f),
        }
    }
}

/// Priority levels for threads in the thread pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LeptonThreadPriority {
    /// Low priority thread
    Low,
    /// Normal priority thread, we don't touch the priority of these threads.
    #[default]
    Normal,
    /// High priority thread
    High,
}

/// A fixed-size thread pool with a bounded work queue.
#[derive(Default)]
pub struct SimpleThreadPool {
    priority: LeptonThreadPriority,
    runtime: OnceLock<PoolRuntime>,
}

impl SimpleThreadPool {
    /// Creates a new thread pool with the specified priority.
    pub const fn new(priority: LeptonThreadPriority) -> Self {
        SimpleThreadPool {
            priority,
            runtime: OnceLock::new(),
        }
    }

    /// Returns the number of idle threads in the thread pool.
    #[allow(dead_code)]
    pub fn get_idle_threads(&self) -> usize {
        self.runtime.get().map_or(0, |runtime| {
            runtime.state.idle_threads.load(Ordering::Acquire)
        })
    }

    fn create_runtime(&self) -> PoolRuntime {
        let worker_count = *NUM_CPUS;
        let state = Arc::new(PoolState {
            queue: Mutex::new(WorkQueue {
                work: VecDeque::with_capacity(worker_count),
                shutdown: false,
            }),
            work_available: Condvar::new(),
            idle_threads: AtomicUsize::new(0),
            capacity: worker_count.saturating_mul(MAX_THREADS_SUPPORTED_BY_LEPTON_FORMAT),
        });

        for _ in 0..worker_count {
            let state = state.clone();
            let priority = self.priority;

            spawn(move || Self::worker_loop(priority, state));
        }

        PoolRuntime { state }
    }

    fn worker_loop(priority: LeptonThreadPriority, state: Arc<PoolState>) {
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        match priority {
            LeptonThreadPriority::Low => {
                let _ = thread_priority::set_current_thread_priority(
                    thread_priority::ThreadPriority::Min,
                );
            }
            LeptonThreadPriority::Normal => {}
            LeptonThreadPriority::High => {
                let _ = thread_priority::set_current_thread_priority(
                    thread_priority::ThreadPriority::Max,
                );
            }
        }

        loop {
            state.idle_threads.fetch_add(1, Ordering::Release);
            let work = {
                let mut queue = state.queue.lock().unwrap();
                while queue.work.is_empty() && !queue.shutdown {
                    queue = state.work_available.wait(queue).unwrap();
                }
                state.idle_threads.fetch_sub(1, Ordering::Release);

                if queue.shutdown && queue.work.is_empty() {
                    return;
                }

                queue.work.pop_front().unwrap()
            };

            let _ = catch_unwind(AssertUnwindSafe(work));
        }
    }

    fn try_enqueue(&self, work: WorkItem) -> Result<()> {
        let runtime = self.runtime.get_or_init(|| self.create_runtime());
        let mut queue = runtime.state.queue.lock().unwrap();

        if queue.work.len() >= runtime.state.capacity {
            return err_exit_code(ExitCode::ChannelFailure, "thread pool work queue is full");
        }

        queue.work.push_back(work);
        runtime.state.work_available.notify_one();
        Ok(())
    }

    /// Executes a closure on the bounded pool.
    fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.try_enqueue(Box::new(f)).unwrap();
    }
}

/// A default instance of the `SimpleThreadPool` that can be used for encoding and decoding operations.
pub static DEFAULT_THREAD_POOL: SimpleThreadPool =
    SimpleThreadPool::new(LeptonThreadPriority::Normal);

impl LeptonThreadPool for SimpleThreadPool {
    fn max_parallelism(&self) -> usize {
        *NUM_CPUS
    }
    fn run(&self, f: Box<dyn FnOnce() + Send + 'static>) {
        self.execute(f);
    }
    fn try_run(&self, f: Box<dyn FnOnce() + Send + 'static>) -> Result<()> {
        self.try_enqueue(f)
    }
}

static NUM_CPUS: LazyLock<usize> = LazyLock::new(|| thread::available_parallelism().unwrap().get());

#[test]
fn test_threadpool() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    let pool = SimpleThreadPool::new(LeptonThreadPriority::Normal);
    let a: Arc<AtomicU32> = Arc::new(AtomicU32::new(0));

    for _ in 0..*NUM_CPUS {
        let aref = a.clone();
        pool.try_run(Box::new(move || {
            aref.fetch_add(1, Ordering::AcqRel);
        }))
        .unwrap();
    }

    while a.load(std::sync::atomic::Ordering::Acquire) < *NUM_CPUS as u32 {
        thread::yield_now();
    }

    println!("Idle threads: {}", pool.get_idle_threads());
}

#[test]
fn test_threadpool_rejects_saturated_queue() {
    use std::sync::Condvar;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let pool = SimpleThreadPool::new(LeptonThreadPriority::Normal);
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let started = Arc::new(AtomicUsize::new(0));

    for _ in 0..*NUM_CPUS {
        let gate = gate.clone();
        let started = started.clone();
        pool.try_run(Box::new(move || {
            started.fetch_add(1, Ordering::Release);
            let (lock, condvar) = &*gate;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = condvar.wait(released).unwrap();
            }
        }))
        .unwrap();
    }

    while started.load(Ordering::Acquire) < *NUM_CPUS {
        thread::yield_now();
    }

    let capacity = pool.runtime.get().unwrap().state.capacity;
    for _ in 0..capacity {
        pool.try_run(Box::new(|| {})).unwrap();
    }

    let error = pool.try_run(Box::new(|| {})).unwrap_err();
    assert_eq!(error.exit_code(), ExitCode::ChannelFailure);

    let (lock, condvar) = &*gate;
    *lock.lock().unwrap() = true;
    condvar.notify_all();
}

/// single thread pool that creates that doesn't create any threads
#[derive(Default)]
pub struct SingleThreadPool {}

impl LeptonThreadPool for SingleThreadPool {
    fn max_parallelism(&self) -> usize {
        1
    }
    fn run(&self, _f: Box<dyn FnOnce() + Send + 'static>) {
        panic!("SingleThreadPool does not support run; execute directly instead");
    }
}
