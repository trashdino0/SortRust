//! `sort-bench` — benchmark harness for `sort-core`.
//!
//! Conventions inherited from `fibonacci-rust` / `prime-rust` (`learned.md`):
//! warmed release runs, `-a N` mean / 95% CI / stddev / skewness, verification
//! outside the timed section, `anyhow` context at the binary edge, rayon at
//! exactly one level.

use anyhow::Result;
use clap::{Parser, ValueEnum};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};
use std::time::Instant;

use sort_core::{
    bucket_sort, fast_radix_sort, fast_radix_sort_with_buf, insertion_sort, merge_sort,
    non_parallel_timsort, parallel_radix_sort, parallel_radix_sort_optimized, parallel_timsort,
    quicksort, radix16_sort, radix_sort,
};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq)]
enum Order {
    /// Uniform random in [lo, hi).
    Random,
    /// Same multiset as random, sorted ascending.
    Ascending,
    /// Same multiset as random, sorted descending.
    Descending,
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq)]
enum Algo {
    Insertion,
    Bucket,
    Quick,
    Merge,
    Radix,
    ParRadix,
    ParRadixOpt,
    Radix16,
    TimsortSerial,
    TimsortParallel,
    FastRadix,
    StdUnstable,
    StdStable,
    RayonParSort,
}

impl Algo {
    fn name(&self) -> &'static str {
        match self {
            Algo::Insertion => "insertion",
            Algo::Bucket => "bucket",
            Algo::Quick => "quicksort",
            Algo::Merge => "merge",
            Algo::Radix => "radix(serial-base10)",
            Algo::ParRadix => "radix(par-base10)",
            Algo::ParRadixOpt => "radix(par-base256)",
            Algo::Radix16 => "radix16(2x16bit)",
            Algo::TimsortSerial => "timsort(serial)",
            Algo::TimsortParallel => "timsort(parallel)",
            Algo::FastRadix => "fast_radix(java-port)",
            Algo::StdUnstable => "std_unstable(ref)",
            Algo::StdStable => "std_stable(ref)",
            Algo::RayonParSort => "rayon_par_sort(ref)",
        }
    }

    fn is_slow(&self) -> bool {
        matches!(
            self,
            Algo::Insertion | Algo::Quick | Algo::Merge | Algo::Bucket
        )
    }
}

const ALL: [Algo; 14] = [
    Algo::Insertion,
    Algo::Bucket,
    Algo::Quick,
    Algo::Merge,
    Algo::Radix,
    Algo::ParRadix,
    Algo::ParRadixOpt,
    Algo::Radix16,
    Algo::TimsortSerial,
    Algo::TimsortParallel,
    Algo::FastRadix,
    Algo::StdUnstable,
    Algo::StdStable,
    Algo::RayonParSort,
];

const FAST_DEFAULT: [Algo; 10] = [
    Algo::Radix,
    Algo::ParRadix,
    Algo::ParRadixOpt,
    Algo::Radix16,
    Algo::TimsortSerial,
    Algo::TimsortParallel,
    Algo::FastRadix,
    Algo::StdUnstable,
    Algo::StdStable,
    Algo::RayonParSort,
];

#[derive(Parser, Debug)]
#[command(
    name = "sort-bench",
    version,
    about = "Sorting benchmark (Rust port of Java + numba suite)"
)]
struct Args {
    /// Array size to sort.
    #[arg(long, default_value_t = 1_000_000)]
    size: usize,

    /// Timed runs per algorithm (stats need >= 2).
    #[arg(short = 'a', long = "average", default_value_t = 1)]
    runs: u32,

    /// Warm-up runs per algorithm (untimed).
    #[arg(short = 'w', long = "warmup", default_value_t = 1)]
    warmup: u32,

    /// Run every algorithm, including slow O(n^2)/alloc-heavy ones.
    #[arg(long, default_value_t = false)]
    all: bool,

    /// Single algorithm to run (default: fast subset; with --all: everything).
    #[arg(long, value_enum)]
    algo: Option<Algo>,

    /// Inclusive lower bound of uniform input.
    #[arg(long, default_value_t = -1_000_000)]
    lo: i32,

    /// Exclusive upper bound of uniform input.
    #[arg(long, default_value_t = 1_000_000)]
    hi: i32,

    /// RNG seed for reproducible inputs.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Input distribution. Ascending/descending reuse the random multiset
    /// (sorted / reverse-sorted), so value range — and radix pass count —
    /// is identical across orders; only element order varies.
    #[arg(long, value_enum, default_value_t = Order::Random)]
    order: Order,

    /// Rayon worker threads (0 = default).
    #[arg(long, default_value_t = 0)]
    jobs: usize,
}

fn t_crit_95(df: usize) -> f64 {
    const T: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    if df == 0 {
        f64::NAN
    } else if df <= 30 {
        T[df - 1]
    } else {
        1.96
    }
}

fn short_hash(data: &[i32]) -> String {
    let mut h = Sha256::new();
    for &x in data {
        h.update(x.to_le_bytes());
    }
    let d = h.finalize();
    d[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// Sort `v` in place / functionally depending on algo. Returns owned sorted
/// vec for uniform hashing (functional algos already return one).
fn run_once(algo: Algo, v: Vec<i32>) -> Vec<i32> {
    match algo {
        Algo::Insertion => {
            let mut w = v;
            insertion_sort(&mut w);
            w
        }
        Algo::Bucket => {
            let mut w = v;
            bucket_sort(&mut w);
            w
        }
        Algo::Quick => quicksort(&v),
        Algo::Merge => merge_sort(&v),
        Algo::Radix => {
            let mut w = v;
            radix_sort(&mut w);
            w
        }
        Algo::ParRadix => {
            let mut w = v;
            parallel_radix_sort(&mut w);
            w
        }
        Algo::ParRadixOpt => {
            let mut w = v;
            parallel_radix_sort_optimized(&mut w);
            w
        }
        Algo::Radix16 => {
            let mut w = v;
            radix16_sort(&mut w);
            w
        }
        Algo::TimsortSerial => {
            let mut w = v;
            non_parallel_timsort(&mut w);
            w
        }
        Algo::TimsortParallel => {
            let mut w = v;
            parallel_timsort(&mut w);
            w
        }
        Algo::FastRadix => {
            let mut w = v;
            fast_radix_sort(&mut w);
            w
        }
        Algo::StdUnstable => {
            let mut w = v;
            w.sort_unstable();
            w
        }
        Algo::StdStable => {
            let mut w = v;
            w.sort();
            w
        }
        Algo::RayonParSort => {
            use rayon::slice::ParallelSliceMut;
            let mut w = v;
            w.par_sort_unstable();
            w
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.size == 0 {
        anyhow::bail!("--size must be > 0");
    }
    if args.hi <= args.lo {
        anyhow::bail!("--hi ({}) must exceed --lo ({})", args.hi, args.lo);
    }
    if args.jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global()
            .ok(); // already built (tests) -> keep existing
    }

    let mut algos: Vec<Algo> = if let Some(a) = args.algo {
        vec![a]
    } else if args.all {
        ALL.to_vec()
    } else {
        FAST_DEFAULT.to_vec()
    };
    // Guardrail: functional / quadratic algos blow memory or time at huge
    // n (quicksort/merge allocate multiples of n per level; insertion and
    // bucket scale as O(n^2)/O(n*sqrt(n))). Refuse them above their caps so
    // a 100 M run can't OOM or hang; cover them via small-size --all runs.
    const SLOW_MAX_N: usize = 100_000; // bucket, quicksort, merge
    const INSERTION_MAX_N: usize = 20_000; // quadratic worst case
    if args.size > SLOW_MAX_N && algos.iter().any(|a| a.is_slow() && *a != Algo::Insertion) {
        if matches!(args.algo, Some(a) if a.is_slow() && a != Algo::Insertion) {
            anyhow::bail!(
                "{} at n={} refused: capped at n<={} (use a small --size)",
                args.algo.unwrap().name(),
                args.size,
                SLOW_MAX_N
            );
        }
        println!(
            "Skipping slow algos at n={} (cap n<={}); cover via small-size --all",
            args.size, SLOW_MAX_N
        );
        algos.retain(|a| !a.is_slow() || *a == Algo::Insertion);
    }
    if args.size > INSERTION_MAX_N && algos.contains(&Algo::Insertion) {
        if args.algo == Some(Algo::Insertion) {
            anyhow::bail!(
                "insertion at n={} refused: capped at n<={} (quadratic worst case)",
                args.size,
                INSERTION_MAX_N
            );
        }
        println!(
            "Skipping insertion at n={} (cap n<={})",
            args.size, INSERTION_MAX_N
        );
        algos.retain(|a| *a != Algo::Insertion);
    }
    if !args.all {
        let skipped: Vec<_> = ALL
            .iter()
            .filter(|a| a.is_slow() && !algos.contains(a))
            .map(|a| a.name())
            .collect();
        if !skipped.is_empty() {
            println!("Skipping slow algos (use --all): {}", skipped.join(", "));
        }
    }

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut raw: Vec<i32> = (0..args.size)
        .map(|_| rng.gen_range(args.lo..args.hi))
        .collect();
    let order_name = match args.order {
        Order::Random => "random",
        Order::Ascending => {
            raw.sort_unstable();
            "ascending"
        }
        Order::Descending => {
            raw.sort_unstable();
            raw.reverse();
            "descending"
        }
    };
    let mut oracle = raw.clone();
    oracle.sort_unstable();
    let oracle_hash = short_hash(&oracle);
    println!(
        "n = {} order = {} | lo = {} hi = {} seed = {} | threads = {} | oracle sha[..16] = {oracle_hash}",
        args.size,
        order_name,
        args.lo,
        args.hi,
        args.seed,
        rayon::current_num_threads(),
    );

    println!(
        "\n{:<22} | {:>12} | {:>10} | Sorted?",
        "Algorithm", "Mean", "Last hash"
    );
    println!("{}", "-".repeat(70));

    for algo in algos {
        // Reusable scratch for fast_radix ([3]): allocating 1 GB of zeros
        // inside every timed run at 250 M inflates both mean and variance.
        // The buffer is fully overwritten each pass, so reuse is sound.
        let mut scratch: Vec<u32> = if algo == Algo::FastRadix {
            vec![0u32; args.size]
        } else {
            Vec::new()
        };
        let mut run = |v: Vec<i32>| -> Vec<i32> {
            if algo == Algo::FastRadix {
                let mut w = v;
                fast_radix_sort_with_buf(&mut w, &mut scratch);
                w
            } else {
                run_once(algo, v)
            }
        };
        // Warm-up outside the clock (also JIT/pool warm).
        for _ in 0..args.warmup {
            let _ = run(raw.clone());
        }
        let mut times = Vec::with_capacity(args.runs.max(1) as usize);
        let mut last: Vec<i32> = Vec::new();
        for _ in 0..args.runs.max(1) {
            let input = raw.clone();
            let t0 = Instant::now();
            last = run(input);
            times.push(t0.elapsed().as_secs_f64());
        }
        // Verification outside the timed section.
        let sorted_ok = last.windows(2).all(|w| w[0] <= w[1]);
        let correct = last == oracle;
        let hash = short_hash(&last);
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        println!(
            "{:<22} | {:>10.5}s | {hash} | {}",
            algo.name(),
            mean,
            if correct && sorted_ok { "YES" } else { "NO" }
        );
        if times.len() > 1 {
            let n = times.len() as f64;
            let var = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0);
            let sd = var.sqrt();
            let moe = t_crit_95(times.len() - 1) * (sd / n.sqrt());
            let skew = if sd > 0.0 {
                times.iter().map(|t| ((t - mean) / sd).powi(3)).sum::<f64>() / n
            } else {
                0.0
            };
            println!(
                "  runs={} mean={mean:.5}s 95%CI=[{:.5}, {:.5}] sd={sd:.5} skew={skew:.3}",
                times.len(),
                mean - moe,
                mean + moe
            );
        }
        if !(correct && sorted_ok) {
            eprintln!("  MISMATCH vs oracle for {}", algo.name());
        }
    }
    println!("\nDone. Heavy sizes (10M+/100M) by explicit --size only; box-quiet checks apply (see README).");
    Ok(())
}
