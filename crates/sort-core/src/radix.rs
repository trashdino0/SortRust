//! Serial LSD radix sort, base-10 — port of `sorting_numba.py::radix_sort`.
//!
//! The Python code assumes non-negative inputs (`(val // exp) % 10` with
//! `@njit` semantics). This port works on the full `i32` range by sorting
//! the order-preserving biased keys `x as u32 ^ 0x8000_0000` (same idea as
//! the Java sign-bit fix [1]) with base-10 digits, then unbiasing.

/// Sorts `arr` in place (serial, base-10 LSD).
pub fn radix_sort(arr: &mut [i32]) {
    if arr.is_empty() {
        return;
    }
    // Bias once; LSD passes keep the bijection, unbias at the end.
    let mut keys: Vec<u32> = arr.iter().map(|&x| x as u32 ^ 0x8000_0000).collect();
    radix_sort_u32(&mut keys);
    for (dst, k) in arr.iter_mut().zip(keys) {
        *dst = (k ^ 0x8000_0000) as i32;
    }
}

fn radix_sort_u32(a: &mut [u32]) {
    if a.is_empty() {
        return;
    }
    let mx = *a.iter().max().unwrap();
    let mut exp: u64 = 1;
    let mut buf = vec![0u32; a.len()];
    let mut src: &mut [u32] = a;
    let mut dst: &mut [u32] = &mut buf;
    // Ping-pong; track where the live data is.
    let mut live_in_a = true;
    while mx as u64 / exp > 0 {
        let mut count = [0usize; 10];
        for &v in src.iter() {
            count[((v as u64 / exp) % 10) as usize] += 1;
        }
        for i in 1..10 {
            count[i] += count[i - 1];
        }
        for &v in src.iter().rev() {
            let d = ((v as u64 / exp) % 10) as usize;
            count[d] -= 1;
            dst[count[d]] = v;
        }
        std::mem::swap(&mut src, &mut dst);
        live_in_a = !live_in_a;
        exp *= 10;
    }
    if !live_in_a {
        // Live data sits in `buf` (== `dst` after last swap); copy back.
        // `src` now points at buf, `dst` at a.
        dst.copy_from_slice(src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{assert_sorted_vs_std, random_vec};

    #[test]
    fn matches_std_incl_negatives() {
        for seed in [5, 9] {
            let src = random_vec(3000, -10_000, 10_000, seed);
            let mut v = src.clone();
            radix_sort(&mut v);
            assert_sorted_vs_std(&src, &v);
        }
        let src = random_vec(2000, 0, 10_000, 1);
        let mut v = src.clone();
        radix_sort(&mut v);
        assert_sorted_vs_std(&src, &v);
    }

    #[test]
    fn extremes() {
        let src = vec![i32::MAX, i32::MIN, 0, -1, 1];
        let mut v = src.clone();
        radix_sort(&mut v);
        assert_sorted_vs_std(&src, &v);
    }
}
