//! Bucket sort — port of `sorting_numba.py::bucket_sort` (`@njit(parallel=True)`).
//!
//! Python: `num_buckets = sqrt(n)`, float range split, serial distribute,
//! `prange` insertion-sort of buckets, serial concat. Same shape here with
//! rayon at exactly one level (the per-bucket sort).

use rayon::prelude::*;

use crate::insertion::insertion_sort;

/// Sorts `arr` in place.
pub fn bucket_sort(arr: &mut [i32]) {
    let n = arr.len();
    if n <= 1 {
        return;
    }
    let min = *arr.iter().min().unwrap();
    let max = *arr.iter().max().unwrap();
    if max == min {
        return;
    }
    let num_buckets = (n as f64).sqrt() as usize + 1;
    let range = max as i64 - min as i64; // > 0

    let mut buckets: Vec<Vec<i32>> = vec![Vec::new(); num_buckets];
    for &x in arr.iter() {
        // Integer version of Python's `int((x - min) / rng)` where
        // `rng = range / num_buckets`: idx = (x-min)*num_buckets/range.
        let idx = ((x as i64 - min as i64) as u64 * num_buckets as u64 / range as u64) as usize;
        buckets[idx.min(num_buckets - 1)].push(x);
    }

    // Single parallel level: serial insertion inside each bucket.
    buckets.par_iter_mut().for_each(|b| {
        if b.len() > 1 {
            insertion_sort(b);
        }
    });

    let mut pos = 0;
    for b in &buckets {
        arr[pos..pos + b.len()].copy_from_slice(b);
        pos += b.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{assert_sorted_vs_std, random_vec};

    #[test]
    fn matches_std() {
        for seed in [3, 8] {
            let src = random_vec(4000, 0, 10_000, seed);
            let mut v = src.clone();
            bucket_sort(&mut v);
            assert_sorted_vs_std(&src, &v);
        }
    }

    #[test]
    fn edges() {
        let mut v = vec![4, 4, 4, 4];
        bucket_sort(&mut v);
        assert_eq!(v, vec![4, 4, 4, 4]);
        let src = vec![9, -3, 0, -3, 9, 2];
        let mut v = src.clone();
        bucket_sort(&mut v);
        assert_sorted_vs_std(&src, &v);
    }
}
