//! Parallel LSD radix, base-256 optimized — port of
//! `sorting_numba.py::parallel_radix_sort_optimized`.
//!
//! Differences vs [`crate::parallel_radix`]: 8-bit digits with bitwise
//! extract (`(v >> shift) & 0xFF`), pass count derived from the max value
//! (`passes = ceil(bitlen(mx)/8)`), ping-pong buffers pre-allocated once.
//! Signed handling: bias once (`^ 0x8000_0000`), same order-preserving trick
//! as the Java sign-bit fix [1] but applied upfront.

use rayon::prelude::*;

const RADIX: usize = 256;
const MASK: u32 = 0xFF;

/// Sorts `arr` in place (parallel, base-256).
pub fn parallel_radix_sort_optimized(arr: &mut [i32]) {
    let n = arr.len();
    if n <= 1 {
        return;
    }
    let mut keys: Vec<u32> = arr.iter().map(|&x| x as u32 ^ 0x8000_0000).collect();
    parallel_radix_u256(&mut keys);
    for (dst, k) in arr.iter_mut().zip(keys) {
        *dst = (k ^ 0x8000_0000) as i32;
    }
}

fn passes_for(mx: u32) -> usize {
    if mx == 0 {
        return 1;
    }
    let bits = 32 - mx.leading_zeros() as usize;
    bits.div_ceil(8).max(1)
}

fn parallel_radix_u256(a: &mut [u32]) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    let t = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(t);

    let mx = *a.iter().max().unwrap();
    let passes = passes_for(mx);

    let mut buf_a: Vec<u32> = a.to_vec();
    let mut buf_b: Vec<u32> = vec![0; n];
    let mut src_is_a = true;

    for p in 0..passes {
        let shift = (p * 8) as u32;
        let (src, dst) = if src_is_a {
            (buf_a.as_slice(), buf_b.as_mut_slice())
        } else {
            (buf_b.as_slice(), buf_a.as_mut_slice())
        };

        let local: Vec<[usize; RADIX]> = (0..t)
            .into_par_iter()
            .map(|ti| {
                let s = (ti * chunk).min(n);
                let e = (s + chunk).min(n);
                let mut c = [0usize; RADIX];
                for &v in &src[s..e] {
                    c[((v >> shift) & MASK) as usize] += 1;
                }
                c
            })
            .collect();

        let mut global_off = vec![[0usize; RADIX]; t];
        let mut total = [0usize; RADIX];
        for d in 0..RADIX {
            for ti in 0..t {
                global_off[ti][d] = total[d];
                total[d] += local[ti][d];
            }
        }
        let mut starts = [0usize; RADIX];
        for d in 1..RADIX {
            starts[d] = starts[d - 1] + total[d - 1];
        }

        #[derive(Clone, Copy)]
        struct DstPtr(*mut u32);
        // SAFETY: offset math partitions dst into disjoint per-(thread, digit)
        // ranges; par_iter joins before return.
        unsafe impl Send for DstPtr {}
        unsafe impl Sync for DstPtr {}
        let dst_ptr = DstPtr(dst.as_mut_ptr());
        let dst_len = dst.len();
        (0..t).into_par_iter().for_each(|ti| {
            // Copy the whole wrapper by value (new name avoids the
            // redundant-locals lint): guarantees the closure captures `DstPtr`
            // (which is Sync) rather than its raw-pointer field.
            let my_dst = dst_ptr;
            let s = (ti * chunk).min(n);
            let e = (s + chunk).min(n);
            let mut pos = [0usize; RADIX];
            for d in 0..RADIX {
                pos[d] = starts[d] + global_off[ti][d];
            }
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
        for seed in [41, 55] {
            let src = random_vec(6000, -50_000, 50_000, seed);
            let mut v = src.clone();
            parallel_radix_sort_optimized(&mut v);
            assert_sorted_vs_std(&src, &v);
        }
    }

    #[test]
    fn extremes_and_uniform() {
        let src = vec![i32::MIN, i32::MAX, 0, -1, 1];
        let mut v = src.clone();
        parallel_radix_sort_optimized(&mut v);
        assert_sorted_vs_std(&src, &v);
        let mut u = vec![7; 1000];
        parallel_radix_sort_optimized(&mut u);
        assert_eq!(u, vec![7; 1000]);
    }
}
