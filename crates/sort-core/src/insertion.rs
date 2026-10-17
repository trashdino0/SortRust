//! Serial insertion sort — direct port of `sorting_numba.py::insertion_sort`.
//!
//! Python (numba `@njit`) is in-place; this is too. Small slices only in
//! practice (bucket/timsort runs); kept as a standalone algo per request to
//! keep ALL algorithms.

/// Sorts `arr` in place with insertion sort.
pub fn insertion_sort(arr: &mut [i32]) {
    for i in 1..arr.len() {
        let key = arr[i];
        let mut j = i;
        while j > 0 && arr[j - 1] > key {
            arr[j] = arr[j - 1];
            j -= 1;
        }
        arr[j] = key;
    }
}

/// Insertion sort over a sub-range `[start, end]` inclusive, used by timsort
/// runs. Mirrors the Python run loops exactly.
pub(crate) fn insertion_sort_range(arr: &mut [i32], start: usize, end_inclusive: usize) {
    if arr.is_empty() || start >= arr.len() {
        return;
    }
    let end = end_inclusive.min(arr.len() - 1);
    for i in (start + 1)..=end {
        let key = arr[i];
        let mut j = i;
        while j > start && arr[j - 1] > key {
            arr[j] = arr[j - 1];
            j -= 1;
        }
        arr[j] = key;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{assert_sorted_vs_std, random_vec};

    #[test]
    fn empty_and_single() {
        let mut v: Vec<i32> = vec![];
        insertion_sort(&mut v);
        assert!(v.is_empty());
        let mut v = vec![5];
        insertion_sort(&mut v);
        assert_eq!(v, vec![5]);
    }

    #[test]
    fn matches_std() {
        for seed in [1, 7, 42] {
            let src = random_vec(500, 0, 10_000, seed);
            let mut v = src.clone();
            insertion_sort(&mut v);
            assert_sorted_vs_std(&src, &v);
        }
    }

    #[test]
    fn negatives_and_reversed() {
        let src = vec![5, -1, 3, -10, 0, 5, -1];
        let mut v = src.clone();
        insertion_sort(&mut v);
        assert_sorted_vs_std(&src, &v);
    }
}
