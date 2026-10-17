# SortRust

Rust port of **both** sorting projects in `SortPrim`:

- `sorting/src/SortingBenchmarks.java` — parallel LSD `FastRadixSort` (+9 opts)
- `sorting_numba.py` — insertion, bucket, quick, merge, radix (serial base-10),
  parallel radix (base-10), parallel radix optimized (base-256),
  timsort (serial + parallel), plus the benchmark engine

Prime work (`PrimeRust`) is intentionally out of scope — sorting only, per request.

## Layout (workspace)

```text
SortRust/
  Cargo.toml                  # workspace; release: opt-3 + LTO + codegen-units=1
  crates/
    sort-core/                # library: all algorithms, no I/O
      src/{lib,insertion,bucket,quick,merge,radix,parallel_radix,
            parallel_radix_opt,fast_radix,timsort,utils}.rs
      benches/sorting.rs      # criterion (small algos @10k, fast family @1M)
    sort-cli/                 # `sort-bench` binary (clap + stats + hash)
```

## Rules (from `learned.md`, enforced)

- **Measure first, one change at a time** — release builds only, `-a N` gives
  mean / 95% CI / stddev / skewness (same t-table as fibonacci/prime).
- **Rayon at exactly one level** — outer parallel loops only; inner sorts,
  counts, and merges are serial. No nested joins.
- **Correctness is differential** — every algo is tested against
  `slice::sort_unstable`; the CLI re-verifies against a std-sorted oracle
  outside the timed section and prints `SHA256(le_bytes)[..16]`.
- **`target-cpu=native` + mimalloc + PGO** — see below.
- **Heavy runs are explicit** — defaults are 1 M CLI / 10 k–1 M criterion.
  10 M/100 M only via explicit `--size`; keep the box quiet (`Get-Process`
  after kills) and interleave A/B for close calls.

## Quick start

```powershell
# Build + test (gate: clippy + fmt + test)
cargo test --workspace --release
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check

# Bench CLI (fast subset by default; --all adds insertion/bucket/quick/merge)
cargo run --release -p sort-cli -- --size 1000000 -a 3
cargo run --release -p sort-cli -- --all --size 100000 -a 2
cargo run --release -p sort-cli -- --algo fast_radix --size 1000000 -a 5

# Criterion (HTML reports under target/criterion)
cargo bench -p sort-core
```

## Performance recipe (stable Windows)

```powershell
# 1. Native CPU (biggest cheap win, ~1.7x in fib/prime work; binary is machine-specific)
$env:RUSTFLAGS="-C target-cpu=native"
cargo build --release -p sort-cli

# 2. PGO: instrument → train on representative sizes → merge → use
rustup component add llvm-tools-preview
$env:RUSTFLAGS="-Cprofile-generate=C:/temp/pgo-data -C target-cpu=native"
cargo build --release -p sort-cli
.\target\release\sort-bench --size 1000000 -a 1
.\target\release\sort-bench --size 5000000 -a 1
& "${env:USERPROFILE}\.rustup\toolchains\*\lib\rustlib\x86_64-pc-windows-msvc\bin\llvm-profdata.exe" merge -o C:/temp/pgo-data/merged.profdata C:/temp/pgo-data
$env:RUSTFLAGS="-Cprofile-use=C:/temp/pgo-data/merged.profdata -C target-cpu=native"
cargo build --release -p sort-cli
```

mimalloc is the global allocator in `sort-cli` (measured ~10% on
allocation-heavy parallel phases in fib/prime; needs VS `cl.exe` at build —
no MinGW needed).

## Algorithm notes

| Module | Source | Notes |
|---|---|---|
| `insertion` | numba `insertion_sort` | in-place; also backs bucket runs |
| `bucket` | numba `bucket_sort` | `sqrt(n)` buckets, int range math, rayon over buckets |
| `quick` | numba `quicksort` | functional middle-pivot, serial, faithful alloc pattern |
| `merge` | numba `merge_sort` | functional, serial |
| `radix` | numba `radix_sort` | serial base-10; bias `^ MIN` extends to full `i32` |
| `parallel_radix` | numba `parallel_radix_sort` | base-10, parallel count + scatter |
| `parallel_radix_opt` | numba `parallel_radix_sort_optimized` | base-256 bitwise, passes from `max`, ping-pong |
| `fast_radix` | Java `FastRadixSort` | all 9 opts (see module docs); buffered 512 scatter |
| `timsort` | numba `*_timsort` | min_run recipe, serial + parallel-merge variants |

References (`std_unstable`, `std_stable`, `rayon_par_sort`) mirror the
Python `np.sort` / `np.sort(stable)` / `x.sort()` baselines.

## Decision log

- Keep ALL algos (per answer), but CLI defaults to the fast subset so a bare
  run finishes; slow ones behind `--all` / `--algo`.
- Bias-once `^ 0x8000_0000` replaces the Java per-pass MSB XOR — same order,
  stated in `fast_radix.rs`.
- Parallel scatters use disjoint-range raw writes under `rayon` joins
  (spans proven disjoint by construction); serial fallback below 500 K.
- Bucket is fastest only at small n: `√n` buckets × insertion sort wins on
  L1-resident buckets but scales as O(n√n) (10 K → 100 K measured 11×);
  serial distribute/concat + skew risk cap it. Not a rayon-beater; left as-is.
- `fast_radix` chunk rule fixed (2026-10-11): literal Java clamp
  (`min(chunks, n/500K)`) idled threads at mid sizes (2 chunks at 1 M / 4
  threads). Now `chunks = threads×2`, serial only below 500 K — justified:
  rayon `par_iter` chunks are single tasks, no ForkJoin split overhead.
  1 M: 9.9 → 6.5 ms, beats rayon 9.1 ms (non-overlapping 95% CIs).
- 100 M validation (2026-10-11, 12 threads, `-a 1 -w 1`, seed 42,
  value range ±1 M; orders share one multiset so radix pass counts match).
  All YES, identical oracle hash `c23e60bb40ba81a7` every run (s):


  | algo | random | ascending | descending |
  |---|---|---|---|
  | radix serial base-10 | 15.98 | 15.84 | 15.62 |
  | radix par base-10 | 2.54 | 2.33 | 2.38 |
  | radix par base-256 | 0.59 | 0.65 | 0.49 |
  | radix16 | 0.93 | 0.44 | 0.42 |
  | timsort serial | 11.30 | 2.85 | 3.23 |
  | timsort parallel | 2.70 | 1.34 | 1.48 |
  | fast_radix | 0.43 | 0.50 | 0.41 |
  | std_unstable | 2.83 | 0.06 | 2.97 |
  | std_stable | 4.96 | 0.04 | 3.62 |
  | rayon par_sort | 0.58 | 0.06 | 0.41 |

  Verdict (250 M random, `-a 10`, 12 threads): fast_radix 1.079 s
  [1.00, 1.16] vs rayon 1.312 s [1.30, 1.33] — separated CIs, ~18%
  faster. Two things made it solid: (a) the CLI now reuses the scratch
  buffer across runs via `fast_radix_sort_with_buf` ([3] as intended —
  fresh 1 GB zeroing inside every timed run had inflated mean and
  variance; skew fell 0.71 → 0.03), (b) n=10 reps. Scope of the claim:
  i32 only (rayon is generic over any `Ord`), random/descending inputs;
  ascending still belongs to run-detecting comparison sorts.
  radix16 beats base-256 on ordered inputs (histogram locality) but
  loses on random.
  Our timsort has no natural-run detection, so sorted input only halves
  it (cheap comparisons, same merge traffic) — known simplification.
- Slow algos (insertion, bucket, quicksort, merge) are capped
  (`--order` runs at 10 K prove all YES on all three orders with the same
  oracle hash; insertion shows the textbook 0.02 / 8.0 / 14.8 ms
  asc/rand/desc split). Above 100 K (20 K for insertion) the CLI refuses
  instead of OOMing/hanging: functional quick/merge allocate multiples
  of n, insertion is quadratic worst-case.
