//! Parallel LSD radix, base-10 — port of `parallel_radix_sort`.
//!
//! Per-pass shape matches the Python: parallel local counting, tiny serial
//! prefix, parallel scatter, ping-pong buffers. Signed inputs are handled by
//! biasing once (`x ^ 0x8000_0000`), which is order-preserving, so the digit
//! passes all run on `u32`.

use rayon::prelude::*;

/// Sorts `arr` in place with parallel base-10 LSD radix.
pub fn parallel_radix_sort(arr: &mut [i32]) {
    let n = arr.len();
    if n <= 1 {
        return;
    }
    let mut keys: Vec<u32> = arr.iter().map(|&x| x as u32 ^ 0x8000_0000).collect();
    parallel_radix_u32(&mut keys);
    for (dst, k) in arr.iter_mut().zip(keys) {
        *dst = (k ^ 0x8000_0000) as i32;
    }
}

fn parallel_radix_u32(a: &mut [u32]) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    let t = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(t);

    let mut buf_a: Vec<u32> = a.to_vec();
    let mut buf_b: Vec<u32> = vec![0; n];
    // `src`/`dst` as indices into the two buffers; 0 = buf_a, 1 = buf_b.
    let (mut src_is_a, mut exp): (bool, u64) = (true, 1);
    let mx = *buf_a.iter().max().unwrap() as u64;

    while mx / exp > 0 {
        let (src, dst) = if src_is_a {
            (buf_a.as_slice(), buf_b.as_mut_slice())
        } else {
            (buf_b.as_slice(), buf_a.as_mut_slice())
        };

        // 1. Parallel local counts: one [10] row per thread.
        let local: Vec<[usize; 10]> = (0..t)
            .into_par_iter()
            .map(|ti| {
                let s = (ti * chunk).min(n);
                let e = (s + chunk).min(n);
                let mut c = [0usize; 10];
                for &v in &src[s..e] {
                    c[((v as u64 / exp) % 10) as usize] += 1;
                }
                c
            })
            .collect();

        // 2. Serial prefix: global offsets + digit starts.
        let mut global_off = vec![[0usize; 10]; t];
        let mut total = [0usize; 10];
        for d in 0..10 {
            for ti in 0..t {
                global_off[ti][d] = total[d];
                total[d] += local[ti][d];
            }
        }
        let mut starts = [0usize; 10];
        for d in 1..10 {
            starts[d] = starts[d - 1] + total[d - 1];
        }

        // 3. Parallel scatter; disjoint output ranges per (thread, digit).
        // Raw pointer shared across rayon tasks: sound because offset math
        // partitions `dst` into disjoint per-(thread, digit) ranges and the
        // par_iter joins before return.
        #[derive(Clone, Copy)]
        struct DstPtr(*mut u32);
        // SAFETY: justified above; tasks never touch the same index.
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
            let mut pos = [0usize; 10];
            for d in 0..10 {
                pos[d] = starts[d] + global_off[ti][d];
            }
            for &v in &src[s..e] {
                let d = ((v as u64 / exp) % 10) as usize;
                unsafe {
                    debug_assert!(pos[d] < dst_len);
                    *my_dst.0.add(pos[d]) = v;
                }
                pos[d] += 1;
            }
        });

        src_is_a = !src_is_a;
        exp *= 10;
    }

    if !src_is_a {
        // Live data ended in buf_b; copy back into `a`.
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
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build_global()
            .ok();
        for seed in [21, 33] {
            let src = random_vec(5000, -10_000, 10_000, seed);
            let mut v = src.clone();
            parallel_radix_sort(&mut v);
            assert_sorted_vs_std(&src, &v);
        }
    }

    #[test]
    fn edge_cases() {
        let mut v: Vec<i32> = vec![];
        parallel_radix_sort(&mut v);
        let src = vec![i32::MIN, i32::MAX, 0, -1, 1, i32::MIN];
        let mut v = src.clone();
        parallel_radix_sort(&mut v);
        assert_sorted_vs_std(&src, &v);
    }
}
