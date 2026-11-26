# Repository instructions

## Project and layout

This repository is a Rust port of Dropbox Lepton for lossless compression of
baseline and progressive JPEGs. Read `README.md` for usage, `DESIGN.md` for the
codec design, and `CONTRIBUTING.md` before contributing.

- `lib`: the `lepton_jpeg` library. Public APIs are exported from `lib/src/lib.rs`;
  JPEG parsing and writing live in `lib/src/jpeg`, and Lepton coding, probability
  models, and threading live in `lib/src/structs`.
- `util`: the `lepton_jpeg_util` command-line application.
- `dll`: the C-compatible dynamic library.
- `python`: Python bindings built with PyO3 and maturin, with pytest tests.
- `tests/end_to_end.rs`: round-trip and existing-format compatibility tests using
  the fixtures in `images`.
- `benches`: Criterion benchmarks; library internals needed for benchmarking are
  exposed through the `micro_benchmark` feature.
- `fuzz`: a separate cargo-fuzz workspace, not a member of the root workspace.

## Correctness and implementation

- Preserve byte-for-byte JPEG reconstruction, not just decoded pixel equality.
  Metadata, trailing data, and supported malformed or truncated input behavior
  matter. Preserve decoding compatibility with existing `.lep` fixtures.
- Encoder and decoder probability-model updates must remain identical and
  deterministic. Changes to coefficient ordering, arithmetic coding, prediction,
  or partitioning must preserve that invariant.
- The core library forbids unsafe code and denies missing public documentation.
  Keep those guarantees; follow the existing `LeptonError` and `ExitCode` error
  handling patterns when processing invalid input.
- Use stable Rust features. Follow the existing `wide`-based SIMD approach rather
  than introducing architecture-specific intrinsics.
- Keep changes focused and follow nearby code and workspace lint settings.
  Preserve existing license headers when editing source files.
- Add focused regression tests for changed behavior. For codec changes, use
  byte-for-byte round trips and existing-file decode tests, including relevant
  baseline, progressive, malformed, and multithreaded cases.
- Keep wrapper APIs and tests consistent when changing shared library behavior.

## Toolchain and validation

Use the Rust toolchain required by `lib/Cargo.toml` and pinned in
`.github/workflows/rust.yml`; the README's older minimum may lag behind.
The workspace uses Rust edition 2024. Python requirements are defined in
`python/pyproject.toml`.

Run Cargo commands from the repository root. `.cargo/config.toml` sets
`WORKSPACE_ROOT` for fixture lookup and Windows hardening flags; preserve those
settings.

Choose tests appropriate to the change, then use the applicable CI checks:

```text
cargo fmt --check --all
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo build --locked --workspace
cargo nextest run --workspace --profile ci
```

Standard Cargo tests can also be run with `cargo test --locked --workspace`.
For focused library tests use `cargo test --locked -p lepton_jpeg <test-filter>`;
for fixture-based integration tests use
`cargo test --locked -p lepton_jpeg_root --test end_to_end <test-filter>`.

For Python binding changes, run from the `python` directory:

```text
uv sync --locked
uv run --no-sync pytest
```

CI also cross-builds the core library for WebAssembly, Linux ARM64, and Windows.
Keep platform-specific code appropriately gated and consult the workflow for
the current target matrix and exact build commands.
