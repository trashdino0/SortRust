//! Functional merge sort — port of `sorting_numba.py::merge_sort`.
//!
//! Serial recursion returning a fresh `Vec`, like the Python source
//! (`np.empty_like` + two recursive calls + linear merge).

/// Returns a new sorted `Vec`.
pub fn merge_sort(arr: &[i32]) -> Vec<i32> {
    if arr.len() <= 1 {
        return arr.to_vec();
    }
    let mid = arr.len() / 2;
    let left = merge_sort(&arr[..mid]);
    let right = merge_sort(&arr[mid..]);
    merge(&left, &right)
}

fn merge(left: &[i32], right: &[i32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        if left[i] < right[j] {
            out.push(left[i]);
            i += 1;
        } else {
            out.push(right[j]);
            j += 1;
        }
    }
    out.extend_from_slice(&left[i..]);
    out.extend_from_slice(&right[j..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{assert_sorted_vs_std, random_vec};

    #[test]
    fn matches_std() {
        for seed in [11, 22] {
            let src = random_vec(2000, -10_000, 10_000, seed);
            let out = merge_sort(&src);
            assert_sorted_vs_std(&src, &out);
        }
    }
}
