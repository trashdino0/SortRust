//! `FastRadixSort` — Rust port of `SortingBenchmarks.java`.
//!
//! The nine documented optimizations are preserved (numbers match the Java
//! header):
//!
//! - [1] Sign-bit fix: the Java XORs `0x80000000` on the MSB pass. Here we
//!   bias once (`x ^ 0x8000_0000`, order-preserving `i32 → u32`) so all four
//!   passes run on plain bytes — provably the same permutation.
//! - [2] Uniform-pass skip: any pass whose global histogram has a single
//!   full bucket is skipped (MSB included).
//! - [3] No internal clone: `sort` borrows `&mut [i32]` and allocates one
//!   scratch buffer; [`fast_radix_sort_with_buf`] lets callers reuse it.
//! - [4] 512-int scatter buffers per bucket (2 KiB) amortise random writes.
//! - [5] Chunk count `parallelism × 2`, clamped so chunks stay ≥ THRESHOLD.
//! - [6] Single pre-scan builds all four global histograms + pass-0
//!   per-chunk rows; globals are order-independent so they stay valid.
//! - [7] Per-chunk histogram rows (no false sharing by construction).
//! - [8] Coarse 500 K threshold decides the chunk count (serial below it).
//! - [9] Per-thread offset rows are cloned locally before the hot loop.

use rayon::prelude::*;

const BITS: u32 = 8;
const BUCKETS: usize = 1 << BITS; // 256
const MASK: u32 = (BUCKETS as u32) - 1;
const BUFFER_SIZE: usize = 512; // [4]
const THRESHOLD: usize = 500_000; // [8]

/// Sorts `arr` in place with the Java `FastRadixSort` algorithm.
pub fn fast_radix_sort(arr: &mut [i32]) {
    let n = arr.len();
    if n <= 1 {
        return;
    }
    let mut buf = vec![0u32; n];
    fast_radix_sort_with_buf(arr, &mut buf);
}

/// Same as [`fast_radix_sort`] but reuses caller-provided scratch space
/// (`buf.len() >= arr.len()`). Saves the 400 MB-class alloc at 100 M ([3]).
pub fn fast_radix_sort_with_buf(arr: &mut [i32], buf: &mut [u32]) {
    let n = arr.len();
    assert!(buf.len() >= n, "scratch buffer too small");
    if n <= 1 {
        return;
    }

    // [1] bias once; equivalent to XOR on the MSB pass only.
    let mut a: Vec<u32> = arr.iter().map(|&x| x as u32 ^ 0x8000_0000).collect();
    let b_slot: &mut [u32] = &mut buf[..n];

    let t = rayon::current_num_threads().max(1);
    // [5][8]: parallelism*2 chunks. Deviation from Java noted: Java clamps
    // chunk *count* by the 500 K task-split threshold (ForkJoin tree
    // overhead); with rayon's flat `par_iter` a chunk is one task, so
    // oversplitting costs nothing and clamping only idles threads
    // (at 1 M / 4 threads the literal rule yields 2 chunks!). Serial
    // fallback below THRESHOLD is kept.
    let mut chunks = (t * 2).max(1).min(n);
    if n <= THRESHOLD {
        chunks = 1;
    }
    let chunk = n.div_ceil(chunks);

    // [6] one parallel pre-scan: pass-0 rows + all four globals.
    let (mut h0, g) = prescan(&a, n, chunks, chunk);

    // Live buffer flag: true = `a`, false = `b`.
    let mut live_in_a = true;

    for (pass, global) in g.iter().enumerate() {
        let shift = pass as u32 * BITS;

        // [2] uniform skip.
        if global.contains(&n) {
            continue;
        }

        let mut global_offsets = [0usize; BUCKETS];
        for i in 1..BUCKETS {
            global_offsets[i] = global_offsets[i - 1] + global[i - 1];
        }

        // Per-chunk histogram of the CURRENT array: reuse pass-0 rows,
        // rebuild for passes 1-3 (array was reordered).
        if pass > 0 {
            let live: &[u32] = if live_in_a { &a } else { b_slot };
            h0 = count_pass(live, n, chunks, chunk, shift);
        }

        // Scatter offsets per chunk.
        let mut thread_offsets = vec![[0usize; BUCKETS]; chunks];
        for j in 0..BUCKETS {
            let mut cur = global_offsets[j];
            for c in 0..chunks {
                thread_offsets[c][j] = cur;
                cur += h0[c][j];
            }
        }

        if live_in_a {
            scatter(&a, b_slot, n, chunks, chunk, shift, &thread_offsets);
        } else {
            scatter(b_slot, &mut a, n, chunks, chunk, shift, &thread_offsets);
        }
        live_in_a = !live_in_a;
    }

    let sorted: &[u32] = if live_in_a { &a } else { b_slot };
    for (dst, &k) in arr.iter_mut().zip(sorted.iter()) {
        *dst = (k ^ 0x8000_0000) as i32;
    }
}

/// [6] Pre-scan returning pass-0 per-chunk rows + four global histograms.
fn prescan(
    a: &[u32],
    n: usize,
    chunks: usize,
    chunk: usize,
) -> (Vec<[usize; BUCKETS]>, [[usize; BUCKETS]; 4]) {
    if chunks == 1 {
        let mut h0 = [0usize; BUCKETS];
        let mut l1 = [0usize; BUCKETS];
        let mut l2 = [0usize; BUCKETS];
        let mut l3 = [0usize; BUCKETS];
        for &v in a {
            h0[(v & MASK) as usize] += 1;
            l1[((v >> 8) & MASK) as usize] += 1;
            l2[((v >> 16) & MASK) as usize] += 1;
            l3[((v >> 24) & MASK) as usize] += 1;
        }
        return (vec![h0], [h0, l1, l2, l3]);
    }
    let rows: Vec<[[usize; BUCKETS]; 4]> = (0..chunks)
        .into_par_iter()
        .map(|c| {
            let s = (c * chunk).min(n);
            let e = (s + chunk).min(n);
            let mut h = [[0usize; BUCKETS]; 4];
            for &v in &a[s..e] {
                h[0][(v & MASK) as usize] += 1;
                h[1][((v >> 8) & MASK) as usize] += 1;
                h[2][((v >> 16) & MASK) as usize] += 1;
                h[3][((v >> 24) & MASK) as usize] += 1;
            }
            h
        })
        .collect();
    let mut h0 = vec![[0usize; BUCKETS]; chunks];
    let mut globals = [[0usize; BUCKETS]; 4];
    for (c, h) in rows.into_iter().enumerate() {
        h0[c] = h[0];
        for p in 0..4 {
            for i in 0..BUCKETS {
                globals[p][i] += h[p][i];
            }
        }
    }
    (h0, globals)
}

/// Per-pass per-chunk histogram of the live array (passes 1-3).
fn count_pass(
    live: &[u32],
    n: usize,
    chunks: usize,
    chunk: usize,
    shift: u32,
) -> Vec<[usize; BUCKETS]> {
    if chunks == 1 {
        let mut h = [0usize; BUCKETS];
        for &v in live {
            h[((v >> shift) & MASK) as usize] += 1;
        }
        return vec![h];
    }
    (0..chunks)
        .into_par_iter()
        .map(|c| {
            let s = (c * chunk).min(n);
            let e = (s + chunk).min(n);
            let mut h = [0usize; BUCKETS];
            for &v in &live[s..e] {
                h[((v >> shift) & MASK) as usize] += 1;
            }
            h
        })
        .collect()
}

/// Parallel buffered scatter `src -> dst` ([4][9]).
fn scatter(
    src: &[u32],
    dst: &mut [u32],
    n: usize,
    chunks: usize,
    chunk: usize,
    shift: u32,
    thread_offsets: &[[usize; BUCKETS]],
) {
    if chunks == 1 {
        // Serial buffered path (also the THRESHOLD fallback [8]).
        let mut my = thread_offsets[0]; // [9] local copy
        let mut buffers = vec![0u32; BUCKETS * BUFFER_SIZE];
        let mut counts = [0usize; BUCKETS];
        for &v in src {
            let bk = ((v >> shift) & MASK) as usize;
            buffers[bk * BUFFER_SIZE + counts[bk]] = v;
            counts[bk] += 1;
            if counts[bk] == BUFFER_SIZE {
                dst[my[bk]..my[bk] + BUFFER_SIZE]
                    .copy_from_slice(&buffers[bk * BUFFER_SIZE..bk * BUFFER_SIZE + BUFFER_SIZE]);
                my[bk] += BUFFER_SIZE;
                counts[bk] = 0;
            }
        }
        for bk in 0..BUCKETS {
            if counts[bk] > 0 {
                dst[my[bk]..my[bk] + counts[bk]]
                    .copy_from_slice(&buffers[bk * BUFFER_SIZE..bk * BUFFER_SIZE + counts[bk]]);
            }
        }
        return;
    }
    #[derive(Clone, Copy)]
    struct DstPtr(*mut u32);
    // SAFETY: thread_offsets partitions dst into disjoint per-chunk ranges;
    // par_iter joins before return.
    unsafe impl Send for DstPtr {}
    unsafe impl Sync for DstPtr {}
    let dst_ptr = DstPtr(dst.as_mut_ptr());
    let _ = n;
    (0..chunks).into_par_iter().for_each(|c| {
        // Copy the whole wrapper by value (new name avoids the
        // redundant-locals lint): guarantees the closure captures `DstPtr`
        // (which is Sync) rather than its raw-pointer field.
        let my_dst = dst_ptr;
        let s = (c * chunk).min(n);
        let e = (s + chunk).min(src.len());
        let mut my = thread_offsets[c]; // [9] local copy, no 2-D traffic
        let mut buffers = vec![0u32; BUCKETS * BUFFER_SIZE]; // [4]
        let mut counts = [0usize; BUCKETS];
        for &v in &src[s..e] {
            let bk = ((v >> shift) & MASK) as usize;
            buffers[bk * BUFFER_SIZE + counts[bk]] = v;
            counts[bk] += 1;
            if counts[bk] == BUFFER_SIZE {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buffers.as_ptr().add(bk * BUFFER_SIZE),
                        my_dst.0.add(my[bk]),
                        BUFFER_SIZE,
                    );
                }
                my[bk] += BUFFER_SIZE;
                counts[bk] = 0;
            }
        }
        for bk in 0..BUCKETS {
            let cnt = counts[bk];
            if cnt > 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buffers.as_ptr().add(bk * BUFFER_SIZE),
                        my_dst.0.add(my[bk]),
                        cnt,
                    );
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{assert_sorted_vs_std, random_vec};

    #[test]
    fn matches_std_random_and_negatives() {
        for seed in [101, 202] {
            let src = random_vec(20_000, -1_000_000, 1_000_000, seed);
            let mut v = src.clone();
            fast_radix_sort(&mut v);
            assert_sorted_vs_std(&src, &v);
        }
        // Full-range incl. extremes (sign-bit path [1]).
        let src = vec![i32::MIN, i32::MAX, 0, -1, 1, i32::MIN + 1, i32::MAX - 1];
        let mut v = src.clone();
        fast_radix_sort(&mut v);
        assert_sorted_vs_std(&src, &v);
    }

    #[test]
    fn uniform_and_sorted_inputs_skip_cleanly() {
        let src = vec![42; 5000];
        let mut v = src.clone();
        fast_radix_sort(&mut v);
        assert_sorted_vs_std(&src, &v);
        let mut s: Vec<i32> = (0..5000).collect();
        let src = s.clone();
        fast_radix_sort(&mut s);
        assert_sorted_vs_std(&src, &s);
    }
}
