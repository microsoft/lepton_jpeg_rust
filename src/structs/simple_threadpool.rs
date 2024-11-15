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
        Arc, Condvar, LazyLock, Mutex,
    },
    thread::{self, spawn},
};

type BoxedTrait = Box<dyn FnOnce() + Send + 'static>;

struct ThreadInPool {
    m: Mutex<Option<BoxedTrait>>,
    cond: Condvar,
}

impl ThreadInPool {
    fn new() -> Self {
        ThreadInPool {
            m: Mutex::new(None),
            cond: Condvar::new(),
        }
    }

    fn put(&self, f: BoxedTrait) {
        if let Ok(mut m) = self.m.lock() {
            assert!(m.is_none());
            *m = Some(f);
            self.cond.notify_one();
        }
    }

    fn take(&self) -> Option<BoxedTrait> {
        if let Ok(m) = self.m.lock() {
            if let Ok(mut c) = self.cond.wait(m) {
                return c.take();
            }
        }
        None
    }
}

static IDLE_THREADS: LazyLock<Mutex<Vec<Arc<ThreadInPool>>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));
static NUM_CPUS: LazyLock<usize> = LazyLock::new(|| thread::available_parallelism().unwrap().get());

#[allow(dead_code)]
pub fn get_idle_threads() -> usize {
    IDLE_THREADS.lock().unwrap().len()
}

/// Executes a closure on a thread from the thread pool. Does not block or return any result.
pub fn execute<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    let i = IDLE_THREADS.lock().unwrap().pop();

    if let Some(t) = i {
        t.put(Box::new(f));
        return;
    }

    let t = Arc::new(ThreadInPool::new());

    spawn(move || {
        f();

        loop {
            if let Ok(mut i) = IDLE_THREADS.lock() {
                // if we don't have more idle threads than CPUs, we can add this thread to the pool
                if i.len() < *NUM_CPUS {
                    i.push(t.clone());
                    drop(i);

                    if let Some(w) = t.take() {
                        w();
                        continue;
                    }
                }
            }

            break;
        }
    });
}

#[test]
fn test_threadpool() {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    let a: Arc<AtomicU32> = Arc::new(AtomicU32::new(0));

    for _i in 0usize..100 {
        let aref = a.clone();
        execute(move || {
            aref.fetch_add(1, Ordering::AcqRel);
        });
    }

    while a.load(std::sync::atomic::Ordering::Acquire) < 100 {
        thread::yield_now();
    }

    println!("Idle threads: {}", get_idle_threads());
}
