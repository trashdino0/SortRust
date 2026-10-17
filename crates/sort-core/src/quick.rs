//! Functional quicksort — port of `sorting_numba.py::quicksort` (`@njit` serial).
//!
//! Python picks the middle pivot and builds `left / middle / right` with
//! boolean masks, then concatenates recursion results. This port does the
//! same allocation pattern (faithful, not optimal) so benchmark comparisons
//! stay apples-to-apples. No rayon here: the Python source is serial, and
//! `learned.md` §3 forbids nested parallelism under an outer parallel driver.

/// Returns a new sorted `Vec` (functional style, like the Python source).
pub fn quicksort(arr: &[i32]) -> Vec<i32> {
    if arr.len() <= 1 {
        return arr.to_vec();
    }
    let pivot = arr[arr.len() / 2];
    let mut left = Vec::new();
    let mut middle = Vec::new();
    let mut right = Vec::new();
    for &v in arr {
        if v < pivot {
            left.push(v);
        } else if v == pivot {
            middle.push(v);
        } else {
            right.push(v);
        }
    }
    let mut out = quicksort(&left);
    out.extend_from_slice(&middle);
    out.extend_from_slice(&quicksort(&right));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{assert_sorted_vs_std, random_vec};

    #[test]
    fn matches_std() {
        for seed in [1, 2, 3] {
            let src = random_vec(1000, -5000, 5000, seed);
            let out = quicksort(&src);
            assert_sorted_vs_std(&src, &out);
        }
    }

    #[test]
    fn edges() {
        assert_eq!(quicksort(&[]), Vec::<i32>::new());
        assert_eq!(quicksort(&[7]), vec![7]);
        let dup = vec![5; 100];
        assert_eq!(quicksort(&dup), dup);
    }
}
