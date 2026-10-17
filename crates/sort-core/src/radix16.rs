//! Parallel LSD radix, 2-pass 16-bit — custom contender aimed at beating
//! `rayon_par_sort`.
//!
//! Rationale: at large n sorting is memory-bound and each 8-bit pass costs
//! one counting sweep + one scatter sweep (read + write) of the whole array.
//! Two 16-bit passes move the same 32 bits in **4 sweeps instead of 8**.
//! Price: per-thread histograms grow 256 → 65 536 entries (512 KiB, L2-sized
//! on Zen 2, L3 otherwise) with random-access increments, plus ~2.5 MiB of
//! transient offset tables per pass. Standard 8/16-bit-LSD tradeoff; wins
//! once the array dwarfs the tables (≥ ~1 M elements).
//!
//! Signed handling, ping-pong buffers, disjoint-scatter discipline, and the
//! chunk-clamp fix are shared with [`crate::parallel_radix_opt`]. A
//! uniform-pass skip (cf. Java [2]) is kept: if one 16-bit bucket holds all
//! of n, the pass is a no-op copy and is skipped.

use rayon::prelude::*;

const RADIX: usize = 1 << 16;
const MASK: u32 = 0xFFFF;

/// Sorts `arr` in place (parallel, 2-pass 16-bit LSD).
pub fn radix16_sort(arr: &mut [i32]) {
    let n = arr.len();
    if n <= 1 {
        return;
    }
    let mut keys: Vec<u32> = arr.iter().map(|&x| x as u32 ^ 0x8000_0000).collect();
    radix16_u32(&mut keys);
    for (dst, k) in arr.iter_mut().zip(keys) {
        *dst = (k ^ 0x8000_0000) as i32;
    }
}

fn passes_for(mx: u32) -> usize {
    if mx == 0 {
        return 1;
    }
    let bits = 32 - mx.leading_zeros() as usize;
    bits.div_ceil(16).max(1)
}

fn radix16_u32(a: &mut [u32]) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    let t = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(t);

    let mx = *a.iter().max().unwrap();
    let passes = passes_for(mx).min(2);

    let mut buf_a: Vec<u32> = a.to_vec();
    let mut buf_b: Vec<u32> = vec![0; n];
    let mut src_is_a = true;

    for p in 0..passes {
        let shift = (p * 16) as u32;
        let (src, dst) = if src_is_a {
            (buf_a.as_slice(), buf_b.as_mut_slice())
        } else {
            (buf_b.as_slice(), buf_a.as_mut_slice())
        };

        let local: Vec<Vec<usize>> = (0..t)
            .into_par_iter()
            .map(|ti| {
                let s = (ti * chunk).min(n);
                let e = (s + chunk).min(n);
                let mut c = vec![0usize; RADIX];
                for &v in &src[s..e] {
                    c[((v >> shift) & MASK) as usize] += 1;
                }
                c
            })
            .collect();

        let mut total = vec![0usize; RADIX];
        for row in &local {
            for (d, &c) in row.iter().enumerate() {
                total[d] += c;
            }
        }
        // Uniform-pass skip: whole array already in one bucket.
        if total.contains(&n) {
            continue;
        }

        // global_off[ti][d] + starts[d] computed in one serial pass.
        let mut starts = vec![0usize; RADIX];
        for d in 1..RADIX {
            starts[d] = starts[d - 1] + total[d - 1];
        }
        // Per-thread base = starts[d] + sum of earlier threads' counts.
        // Folded into the scatter tasks to avoid a t*64K offset table:
        // prefix over threads per digit is O(t) setup each, done once here.
        let mut thread_base = vec![vec![0usize; RADIX]; t];
        for d in 0..RADIX {
            let mut cur = starts[d];
            for ti in 0..t {
                thread_base[ti][d] = cur;
                cur += local[ti][d];
            }
        }

        #[derive(Clone, Copy)]
        struct DstPtr(*mut u32);
        // SAFETY: thread_base partitions dst into disjoint per-(thread,
        // digit) ranges; par_iter joins before return.
        unsafe impl Send for DstPtr {}
        unsafe impl Sync for DstPtr {}
        let dst_ptr = DstPtr(dst.as_mut_ptr());
        let dst_len = dst.len();
        (0..t).into_par_iter().for_each(|ti| {
            let my_dst = dst_ptr;
            let my_base = &thread_base[ti];
            let s = (ti * chunk).min(n);
            let e = (s + chunk).min(n);
            // 512 KiB position table per task, heap-allocated.
            let mut pos = my_base.to_vec();
            for &v in &src[s..e] {
                let d = ((v >> shift) & MASK) as usize;
                unsafe {
                    debug_assert!(pos[d] < dst_len);
                    *my_dst.0.add(pos[d]) = v;
                }
                pos[d] += 1;
            }
        });

        src_is_a = !src_is_a;
    }

    if !src_is_a {
        a.copy_from_slice(&buf_b);
    } else {
        a.copy_from_slice(&buf_a);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{assert_sorted_vs_std, random_vec};

    #[test]
    fn matches_std() {
        for seed in [61, 77] {
            let src = random_vec(6000, -50_000, 50_000, seed);
            let mut v = src.clone();
            radix16_sort(&mut v);
            assert_sorted_vs_std(&src, &v);
        }
    }

    #[test]
    fn extremes_uniform_sorted_and_single_pass() {
        let src = vec![i32::MIN, i32::MAX, 0, -1, 1];
        let mut v = src.clone();
        radix16_sort(&mut v);
        assert_sorted_vs_std(&src, &v);
        // Uniform input exercises the pass-skip path.
        let mut u = vec![7; 1000];
        radix16_sort(&mut u);
        assert_eq!(u, vec![7; 1000]);
        // Small max -> single 16-bit pass.
        let src = random_vec(2000, 0, 1000, 9);
        let mut w = src.clone();
        radix16_sort(&mut w);
        assert_sorted_vs_std(&src, &w);
    }
}
