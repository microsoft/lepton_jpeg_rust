/// A simple thread pool implementation that can be used to evaluate closures on separate threads.
///
/// The pool will keep a number of threads equal to the number of CPUs available on the system, and
/// will reuse threads that are idle.
///
/// If more tasks are submitted than there are threads, the pool will spawn new threads to handle
/// the extra tasks.
///
/// Why write yet another threadpool? There wasn't one that was that supported dynamically growing
/// the threadpool (rayon and tokio are all fixed), which is important since otherwise there is
/// unpredicable latency when the number of tasks submitted is greater than the number of threads.
///
/// No unsafe code is used.
use std::{
    sync::{
        Arc, LazyLock, Mutex,
        mpsc::{Sender, channel},
    },
    thread::{self, spawn},
};

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

/// A simple thread pool that spawns threads on demand and reuses them for executing closures.
/// There is no limit on the number of threads, but the number of idle threads is limited to the number of CPUs available.
#[derive(Default)]
pub struct SimpleThreadPool {
    priority: LeptonThreadPriority,
    idle_threads: LazyLock<Arc<Mutex<Vec<Sender<Box<dyn FnOnce() + Send + 'static>>>>>>,
}

impl SimpleThreadPool {
    /// Creates a new thread pool with the specified priority.
    pub const fn new(priority: LeptonThreadPriority) -> Self {
        SimpleThreadPool {
            priority,
            idle_threads: LazyLock::new(|| Arc::new(Mutex::new(Vec::new()))),
        }
    }

    /// Returns the number of idle threads in the thread pool.
    #[allow(dead_code)]
    pub fn get_idle_threads(&self) -> usize {
        self.idle_threads.lock().unwrap().len()
    }

    /// Executes a closure on a thread from the thread pool. Does not block or return any result.
    fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        if let Some(sender) = self.idle_threads.lock().unwrap().pop() {
            sender.send(Box::new(f)).unwrap();
        } else {
            // channel for receiving future work on this thread
            let (tx_schedule, rx_schedule) = channel();

            let priority = self.priority;
            let idle_threads = self.idle_threads.clone();

            spawn(move || {
                #[cfg(any(target_os = "windows", target_os = "linux"))]
                match priority {
                    LeptonThreadPriority::Low => thread_priority::set_current_thread_priority(
                        thread_priority::ThreadPriority::Min,
                    )
                    .unwrap(),
                    LeptonThreadPriority::Normal => {}
                    LeptonThreadPriority::High => thread_priority::set_current_thread_priority(
                        thread_priority::ThreadPriority::Max,
                    )
                    .unwrap(),
                }

                f();

                loop {
                    if let Ok(mut i) = idle_threads.lock() {
                        // stick back into list of idle threads if there aren't more than
                        // the number of cpus already there.
                        if i.len() > *NUM_CPUS {
                            // just exits the thread
                            break;
                        }
                        i.push(tx_schedule.clone());
                    } else {
                        break;
                    }

                    if let Ok(f) = rx_schedule.recv() {
                        f();
                    } else {
                        // channel broken, exit thread
                        break;
                    }
                }
            });
        }
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
}

static NUM_CPUS: LazyLock<usize> = LazyLock::new(|| thread::available_parallelism().unwrap().get());

#[test]
fn test_threadpool() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    let a: Arc<AtomicU32> = Arc::new(AtomicU32::new(0));

    for _i in 0usize..100 {
        let aref = a.clone();
        DEFAULT_THREAD_POOL.execute(move || {
            aref.fetch_add(1, Ordering::AcqRel);
        });
    }

    while a.load(std::sync::atomic::Ordering::Acquire) < 100 {
        thread::yield_now();
    }

    println!("Idle threads: {}", DEFAULT_THREAD_POOL.get_idle_threads());
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
