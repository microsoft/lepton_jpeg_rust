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
        mpsc::{channel, Sender},
        LazyLock, Mutex,
    },
    thread::{self, spawn},
};

static IDLE_THREADS: LazyLock<Mutex<Vec<Sender<Box<dyn FnOnce() + Send + 'static>>>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));
static NUM_CPUS: LazyLock<usize> = LazyLock::new(|| thread::available_parallelism().unwrap().get());

#[cfg(any(target_os = "windows", target_os = "linux"))]
static THREAD_PRIORITY: Mutex<Option<thread_priority::ThreadPriority>> = Mutex::new(None);

#[allow(dead_code)]
pub fn get_idle_threads() -> usize {
    IDLE_THREADS.lock().unwrap().len()
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
#[allow(dead_code)]
pub fn set_thread_priority(priority: thread_priority::ThreadPriority) {
    *THREAD_PRIORITY.lock().unwrap() = Some(priority);
}

/// Executes a closure on a thread from the thread pool. Does not block or return any result.
pub fn execute<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    if let Some(sender) = IDLE_THREADS.lock().unwrap().pop() {
        sender.send(Box::new(f)).unwrap();
    } else {
        // channel for receiving future work on this thread
        let (tx_schedule, rx_schedule) = channel();

        #[cfg(any(target_os = "windows", target_os = "linux"))]
        let thread_priority = THREAD_PRIORITY.lock().unwrap().clone();

        spawn(move || {
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            if let Some(priority) = thread_priority {
                thread_priority::set_current_thread_priority(priority).unwrap();
            }

            f();

            loop {
                if let Ok(mut i) = IDLE_THREADS.lock() {
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
