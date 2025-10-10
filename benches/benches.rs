use std::{io::Cursor, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use lepton_jpeg::{EnabledFeatures, LeptonThreadPool};

fn read_file(filename: &str, ext: &str) -> Vec<u8> {
    let filename = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("images")
        .join(filename.to_owned() + ext);
    //println!("reading {0}", filename.to_str().unwrap());
    let mut f = std::fs::File::open(filename).unwrap();

    let mut content = Vec::new();
    std::io::Read::read_to_end(&mut f, &mut content).unwrap();

    content
}

/// single thread pool that creates just one threadpool thread
/// useful for benchmarks to measure total end-to-end runtime
struct SingleThreadPool {
    sender: std::sync::mpsc::Sender<Box<dyn FnOnce() + Send + 'static>>,
}

impl SingleThreadPool {
    pub fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<Box<dyn FnOnce() + Send + 'static>>();

        std::thread::spawn(move || {
            while let Ok(f) = rx.recv() {
                f();
            }
        });

        SingleThreadPool { sender: tx }
    }
}

impl LeptonThreadPool for SingleThreadPool {
    fn run(&self, f: Box<dyn FnOnce() + Send + 'static>) {
        self.sender.send(f).unwrap();
    }
}

fn end_to_end_benches(c: &mut Criterion) {
    let thread_pool = SingleThreadPool::new();
    let jpeg = read_file("iphone", ".jpg");
    let lep = read_file("iphone", ".lep");

    c.bench_function("Lepton encode", |b| {
        b.iter(|| {
            let mut output = Vec::with_capacity(jpeg.len());
            lepton_jpeg::encode_lepton(
                &mut Cursor::new(&jpeg),
                &mut Cursor::new(&mut output),
                &EnabledFeatures::compat_lepton_vector_write(),
                &thread_pool,
            )
        })
    });

    c.bench_function("Lepton decode", |b| {
        b.iter(|| {
            let mut output = Vec::with_capacity(lep.len());
            lepton_jpeg::decode_lepton(
                &mut Cursor::new(&lep),
                &mut Cursor::new(&mut output),
                &EnabledFeatures::compat_lepton_vector_write(),
                &thread_pool,
            )
        })
    });
}

criterion_group! {
   name = group1;
   config = Criterion::default().warm_up_time(Duration::from_secs(5));
   targets = end_to_end_benches
}

fn micro_benchmarks(c: &mut Criterion) {
    use lepton_jpeg::micro_benchmark::{
        benchmark_idct, benchmark_read_jpeg, benchmark_roundtrip_coefficient, benchmark_write_jpeg,
    };

    c.bench_function("jpeg read", |b| b.iter(benchmark_read_jpeg()));

    c.bench_function("jpeg write", |b| b.iter(benchmark_write_jpeg()));

    c.bench_function("roundtrip coefficient write", |b| {
        b.iter(benchmark_roundtrip_coefficient())
    });

    c.bench_function("idct benchmark", |b| b.iter(benchmark_idct()));
}

criterion_group!(group2, micro_benchmarks);

criterion_main!(group1, group2);
