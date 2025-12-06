use std::io::{self, Read, Write};
use std::time::Instant;

fn main() -> io::Result<()> {
    // only print every TTFB_THRESHOLD_MS milliseconds (default = 10ms)
    let mut threshold = 10u128;
    if let Ok(ms_string) = std::env::var("TTFB_THRESHOLD_MS") {
        if let Ok(ms) = ms_string.parse::<u128>() {
            if ms > 0 {
                threshold = ms;
            }
        }
    }

    let start = Instant::now();
    let mut total = 0usize;
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    let mut report = io::stderr().lock();
    let mut buffer = [0u8; 64 << 10];

    let mut read: usize = 0;
    let mut then = start.elapsed();
    loop {
        let read_now = input.read(&mut buffer)?;
        total += read_now;
        read += read_now;

        let now = start.elapsed();
        if read_now == 0 || total == read_now || (now - then).as_millis() >= threshold {
            report.write_all(
                format!(
                    "ttfb: {:10.3} {:10} {:10}\n",
                    now.as_secs_f32(),
                    read,
                    total
                )
                .as_bytes(),
            )?;
            read = 0;
            then = now;
        }

        if read_now == 0 {
            break Ok(()); // EOF
        }
        output.write_all(&buffer[..read_now])?
    }
}
