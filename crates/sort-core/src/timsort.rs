//! Timsort (simplified) — ports of `parallel_timsort` / `non_parallel_timsort`.
//!
//! Both share: min_run from the standard bit-recipe, insertion runs, then
//! doubling merge passes. The parallel variant parallelizes each merge pass
//! with rayon (one level only); merges themselves stay serial.

use rayon::prelude::*;

use crate::insertion::insertion_sort_range;

const MIN_MERGE: usize = 32;

fn min_run(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut r = 0;
    let mut tmp = n;
    while tmp >= MIN_MERGE {
        r |= tmp & 1;
        tmp >>= 1;
    }
    tmp + r
}

fn merge_in_place(arr: &mut [i32], left: usize, mid: usize, right: usize) {
    if mid >= right || left > mid {
        return;
    }
    let l = arr[left..=mid].to_vec();
    let r = arr[mid + 1..=right].to_vec();
    let (mut i, mut j, mut k) = (0, 0, left);
    while i < l.len() && j < r.len() {
        if l[i] <= r[j] {
            arr[k] = l[i];
            i += 1;
        } else {
            arr[k] = r[j];
            j += 1;
        }
        k += 1;
    }
    while i < l.len() {
        arr[k] = l[i];
        i += 1;
        k += 1;
    }
    while j < r.len() {
        arr[k] = r[j];
        j += 1;
        k += 1;
    }
}

/// Serial timsort (port of `non_parallel_timsort`).
pub fn non_parallel_timsort(arr: &mut [i32]) {
    let n = arr.len();
    if n <= 1 {
        return;
    }
    let run = min_run(n).max(1);
    let mut start = 0;
    while start < n {
        let end = (start + run - 1).min(n - 1);
        insertion_sort_range(arr, start, end);
        start += run;
    }
    let mut sz = run;
    while sz < n {
        let mut left = 0;
        while left < n {
            let mid = (left + sz - 1).min(n - 1);
            let right = (left + 2 * sz - 1).min(n - 1);
            if mid < right {
                merge_in_place(arr, left, mid, right);
            }
            left += 2 * sz;
        }
        sz *= 2;
    }
}

/// Parallel timsort (port of `parallel_timsort`): merge passes run in
/// parallel; run formation stays serial, like the Python pre-loop.
pub fn parallel_timsort(arr: &mut [i32]) {
    let n = arr.len();
    if n <= 1 {
        return;
    }
    let run = min_run(n).max(1);
    let mut start = 0;
    while start < n {
        let end = (start + run - 1).min(n - 1);
        insertion_sort_range(arr, start, end);
        start += run;
    }
    let mut sz = run;
    while sz < n {
        // Collect independent merge triples, then merge in parallel.
        // Each triple writes a disjoint [left..=right] span, but the
        // borrow checker can't see that, so scatter via raw pointers
        // inside a scoped, join-guarded section.
        let triples: Vec<(usize, usize, usize)> = {
            let mut t = Vec::new();
            let mut left = 0;
            while left < n {
                let mid = (left + sz - 1).min(n - 1);
                let right = (left + 2 * sz - 1).min(n - 1);
                if mid < right {
                    t.push((left, mid, right));
                }
                left += 2 * sz;
            }
            t
        };
        if triples.is_empty() {
            break;
        }
        let ptr = arr.as_mut_ptr();
        let len = arr.len();
        // SAFETY: triples are pairwise-disjoint spans (stride 2*sz), all
        // inside `arr`; rayon joins before return, so no aliasing across
        // threads and no use-after-scope. Read+write stays within one
        // span per task; temp copies are per-task owned Vecs.
        #[derive(Clone, Copy)]
        struct RawPtr(*mut i32);
        unsafe impl Send for RawPtr {}
        unsafe impl Sync for RawPtr {}
        let raw = RawPtr(ptr);
        triples
            .into_par_iter()
            .for_each(|(left, mid, right)| unsafe {
                // Copy the whole wrapper by value (new name avoids the
                // redundant-locals lint): guarantees the closure captures
                // `RawPtr` (which is Sync) rather than its raw-pointer field.
                let my_raw = raw;
                debug_assert!(right < len);
                let slice = std::slice::from_raw_parts_mut(my_raw.0.add(left), right - left + 1);
                let mid_rel = mid - left;
                let l = slice[..=mid_rel].to_vec();
                let r = slice[mid_rel + 1..].to_vec();
                let (mut i, mut j, mut k) = (0usize, 0usize, 0usize);
                while i < l.len() && j < r.len() {
                    if l[i] <= r[j] {
                        slice[k] = l[i];
                        i += 1;
                    } else {
                        slice[k] = r[j];
                        j += 1;
                    }
                    k += 1;
                }
                while i < l.len() {
                    slice[k] = l[i];
                    i += 1;
                    k += 1;
                }
                while j < r.len() {
                    slice[k] = r[j];
                    j += 1;
                    k += 1;
                }
            });
        sz *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{assert_sorted_vs_std, random_vec};

    #[test]
    fn both_match_std() {
        for seed in [13, 29] {
            let src = random_vec(5000, -8000, 8000, seed);
            let mut a = src.clone();
            non_parallel_timsort(&mut a);
            assert_sorted_vs_std(&src, &a);
            let mut b = src.clone();
            parallel_timsort(&mut b);
            assert_sorted_vs_std(&src, &b);
        }
    }
}
