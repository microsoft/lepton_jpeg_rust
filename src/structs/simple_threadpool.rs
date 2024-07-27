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
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        mpsc::{channel, Sender},
        LazyLock, Mutex,
    },
    thread::{self, spawn},
};

use anyhow::Result;

static IDLE_THREADS: LazyLock<Mutex<Vec<Sender<Box<dyn FnOnce() + Send + 'static>>>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));
static NUM_CPUS: LazyLock<usize> = LazyLock::new(|| thread::available_parallelism().unwrap().get());

/// Evaluates a closure on a thread from the thread pool. Returns a closure that will block until the
/// result is available, and return that result.
pub fn evaluate<F, R>(f: F) -> Box<dyn Fn() -> Result<R>>
where
    R: Send + 'static,
    F: FnOnce() -> Result<R> + Send + 'static,
{
    // channel used to send result back to executor
    let (tx, rx) = channel();

    execute(move || {
        let r = catch_unwind(AssertUnwindSafe(f));

        let r = match r {
            Ok(r) => r,
            Err(panic_info) => {
                let error_message = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    format!("Panic occurred: {}", s)
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    format!("Panic occurred: {}", s)
                } else {
                    "Panic occurred: Unknown panic payload".to_string()
                };

                Err(anyhow::Error::msg(error_message))
            }
        };
        let _ = tx.send(r);
    });

    // return a closure that will block until the result is available
    Box::new(move || match rx.recv() {
        Ok(r) => r,
        Err(e) => Err(anyhow::Error::new(e)),
    })
}

#[allow(dead_code)]
pub fn get_idle_threads() -> usize {
    IDLE_THREADS.lock().unwrap().len()
}

/// Executes a closure on a thread from the thread pool. Does not block or return any result.
fn execute<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    if let Some(sender) = IDLE_THREADS.lock().unwrap().pop() {
        sender.send(Box::new(f)).unwrap();
    } else {
        let (tx, rx) = channel();

        spawn(move || {
            f();

            loop {
                if let Ok(mut i) = IDLE_THREADS.lock() {
                    // stick back into list of idle threads if there aren't more than
                    // the number of cpus already there.
                    if i.len() > *NUM_CPUS {
                        // just exits the thread
                        break;
                    }
                    i.push(tx.clone());
                } else {
                    break;
                }

                if let Ok(f) = rx.recv() {
                    f();
                } else {
                    break;
                }
            }
        });
    }
}

#[test]
fn test_threadpool() {
    let mut results = Vec::new();
    for i in 0usize..100 {
        results.push(evaluate(move || Ok(i)));
    }

    for i in 0..100 {
        let r = results[i]();
        match r {
            Ok(r) => assert_eq!(r, i),
            Err(_) => panic!(),
        }
    }

    println!("Idle threads: {}", get_idle_threads());
}
