use std::collections::HashMap;
use std::time::Duration;

#[cfg(windows)]
use cpu_time::ThreadTime;

/// platform independent threadtime measurement
pub struct CpuTimeMeasure {
    #[cfg(windows)]
    start: ThreadTime,
    #[cfg(not(windows))]
    start: std::time::SystemTime,
}

impl CpuTimeMeasure {
    pub fn new() -> Self {
        Self {
            #[cfg(windows)]
            start: ThreadTime::now(),
            #[cfg(not(windows))]
            start: std::time::SystemTime::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        #[cfg(windows)]
        {
            self.start.elapsed()
        }
        #[cfg(not(windows))]
        {
            self.start.elapsed().unwrap()
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone, Hash, Eq)]
pub enum ModelSubComponent {
    Exp,
    Sign,
    Residual,
    Noise,
}

#[derive(Debug, PartialEq, Copy, Clone, Hash, Eq)]
#[repr(u8)]
pub enum ModelComponent {
    Dummy,
    Coef(ModelSubComponent),
    DC(ModelSubComponent),
    Edge(ModelSubComponent),
    NonZero7x7Count,
    NonZeroEdgeCount,
}

#[derive(Default, Debug)]
pub struct ModelComponentStatistics {
    pub total_bits: i64,
    pub total_compressed: i64,
}

#[derive(Default, Debug)]
pub struct Metrics {
    map: HashMap<ModelComponent, ModelComponentStatistics>,
    cpu_time_worker_time: Duration,
}

impl Metrics {
    #[allow(dead_code)]
    pub fn record_compression_stats(
        &mut self,
        cmp: ModelComponent,
        total_bits: i64,
        total_compressed: i64,
    ) {
        let e = self
            .map
            .entry(cmp)
            .or_insert(ModelComponentStatistics::default());
        e.total_bits += total_bits;
        e.total_compressed += total_compressed;
    }

    pub fn record_cpu_worker_time(&mut self, duration: Duration) {
        self.cpu_time_worker_time += duration;
    }

    #[allow(dead_code)]
    pub fn print_metrics(&self) {
        let mut sort_vec = Vec::new();
        for x in &self.map {
            sort_vec.push((x.0, x.1));
        }

        sort_vec.sort_by(|a, b| a.1.total_compressed.cmp(&b.1.total_compressed).reverse());

        let total_compressed: i64 = sort_vec.iter().map(|x| x.1.total_compressed).sum();

        for x in &sort_vec {
            let name = format!("{0:?}", x.0);

            println!(
                "{0:16} total_bits={1:9} compressed_bits={2:9} ratio={3:4} comp_delta={4:10}k storage={5:0.1}%, comp={6:0.2}%)",
                name,
                x.1.total_bits,
                x.1.total_compressed,
                x.1.total_compressed * 100 / x.1.total_bits,
                (x.1.total_bits - x.1.total_compressed)/(8*1024),
                (x.1.total_compressed as f64) * 100f64 / (total_compressed as f64),
                ((x.1.total_bits - x.1.total_compressed) as f64)/(total_compressed as f64)*100f64
            );
        }

        println!(
            "total_compressed = {0} bits, {1} bytes",
            total_compressed,
            total_compressed / 8
        );
        println!("worker_cpu={0}ms", self.cpu_time_worker_time.as_millis());
    }

    pub fn drain(&mut self) -> Metrics {
        Metrics {
            map: self.map.drain().collect(),
            cpu_time_worker_time: self.cpu_time_worker_time,
        }
    }

    pub fn get_cpu_time_worker_time(&self) -> Duration {
        self.cpu_time_worker_time
    }

    pub fn merge_from(&mut self, mut source_metrics: Metrics) {
        for x in source_metrics.map.drain() {
            let e = self
                .map
                .entry(x.0)
                .or_insert(ModelComponentStatistics::default());
            e.total_bits += x.1.total_bits;
            e.total_compressed += x.1.total_compressed;
        }

        self.cpu_time_worker_time += source_metrics.cpu_time_worker_time;
    }
}
