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

pub fn evaluate<F, R>(f: F) -> Box<dyn Fn() -> R>
where
    R: Send + 'static,
    F: FnOnce() -> R + Send + 'static,
{
    let (tx, rx) = channel();

    execute(move || {
        let r = f();
        tx.send(r).unwrap();
    });

    Box::new(move || rx.recv().unwrap())
}

pub fn execute<F>(f: F)
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
                    if i.len() > *NUM_CPUS {
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
    for i in 0..100 {
        results.push(evaluate(move || i));
    }

    for i in 0..100 {
        assert_eq!(results[i](), i);
    }
}
