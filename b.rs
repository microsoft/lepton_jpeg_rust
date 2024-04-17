use std::process::Command;
use std::env;

fn main() {
    let is_release = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string()) == "release";

    let args: Vec<String> = env::args().skip(1).collect();

    let mut rustflags = "-Ccontrol-flow-guard -Ctarget-feature=+crt-static -Clink-args=/DYNAMICBASE /CETCOMPAT".to_string();

    // Run cargo build with the normal arguments
    let mut cargo_build = Command::new("cargo");
    cargo_build.arg("build");
    cargo_build.args(&args);
    cargo_build.env("RUSTFLAGS", &rustflags);

    let build_output = cargo_build.output().unwrap();
    if !build_output.status.success() {
        panic!("Cargo build failed: {}", String::from_utf8_lossy(&build_output.stderr));
    }

    println!("Build successful!");

    if is_release
    {
        let mut cargo_build = Command::new("cargo");
        cargo_build.arg("build");
        cargo_build.args(&args);
        cargo_build.env("RUSTFLAGS", &rustflags);
        rustflags.push_str(" -C target-feature=+avx2,+lzcnt");
    
        let cargo_build = cargo_build.output().unwrap();
        if !cargo_build.status.success() {
            panic!("Cargo build for avx2 failed: {}", String::from_utf8_lossy(&build_output.stderr));
        }
    
        println!("Build for avx2 successful!");
    }
}
