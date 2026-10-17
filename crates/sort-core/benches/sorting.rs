use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sort_core::{
    bucket_sort, fast_radix_sort, insertion_sort, merge_sort, non_parallel_timsort,
    parallel_radix_sort, parallel_radix_sort_optimized, parallel_timsort, quicksort, radix_sort,
};

fn data(n: usize, seed: u64) -> Vec<i32> {
    // Deterministic xorshift (no rand dependency in benches beyond dev-deps use).
    let mut st = seed.max(1);
    let mut next = move || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        st
    };
    (0..n).map(|_| (next() % 20_000) as i32 - 10_000).collect()
}

fn bench_all(c: &mut Criterion) {
    // Small enough for O(n^2)/alloc-heavy algos; scaling benches for the fast
    // radix family live in `bench_fast`.
    let small = data(10_000, 42);
    c.bench_function("insertion/10k", |b| {
        b.iter(|| {
            let mut v = black_box(small.clone());
            insertion_sort(&mut v);
            black_box(v);
        })
    });
    c.bench_function("bucket/10k", |b| {
        b.iter(|| {
            let mut v = black_box(small.clone());
            bucket_sort(&mut v);
            black_box(v);
        })
    });
    c.bench_function("quicksort/10k", |b| {
        b.iter(|| black_box(quicksort(black_box(&small))))
    });
    c.bench_function("merge/10k", |b| {
        b.iter(|| black_box(merge_sort(black_box(&small))))
    });
    c.bench_function("radix_serial/10k", |b| {
        b.iter(|| {
            let mut v = black_box(small.clone());
            radix_sort(&mut v);
            black_box(v);
        })
    });
    c.bench_function("timsort_serial/10k", |b| {
        b.iter(|| {
            let mut v = black_box(small.clone());
            non_parallel_timsort(&mut v);
            black_box(v);
        })
    });
    c.bench_function("timsort_parallel/10k", |b| {
        b.iter(|| {
            let mut v = black_box(small.clone());
            parallel_timsort(&mut v);
            black_box(v);
        })
    });
}

fn bench_fast(c: &mut Criterion) {
    // 1M-element scaling for the algorithms that can take it.
    let big = data(1_000_000, 7);
    for (name, f) in [
        ("radix_par_base10", parallel_radix_sort as fn(&mut [i32])),
        (
            "radix_par_base256",
            parallel_radix_sort_optimized as fn(&mut [i32]),
        ),
        ("fast_radix", fast_radix_sort as fn(&mut [i32])),
    ] {
        let id = format!("{name}/1M");
        c.bench_function(&id, |b| {
            b.iter(|| {
                let mut v = black_box(big.clone());
                f(&mut v);
                black_box(v);
            })
        });
    }
    c.bench_function("std_unstable/1M", |b| {
        b.iter(|| {
            let mut v = black_box(big.clone());
            v.sort_unstable();
            black_box(v);
        })
    });
}

criterion_group!(benches, bench_all, bench_fast);
criterion_main!(benches);
