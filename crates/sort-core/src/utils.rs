//! Utilities: sortedness, differential oracle, deterministic RNG for tests.
//!
//! `learned.md` §6: differential tests with constructed oracles + property
//! checks. For sorting the oracle is `slice::sort` (independent algorithm)
//! plus an exact multiset-equality check via comparison of sorted clones.

/// Returns true if `arr` is non-decreasing.
pub fn is_sorted(arr: &[i32]) -> bool {
    arr.windows(2).all(|w| w[0] <= w[1])
}

/// Asserts `candidate` is sorted and is a permutation of `original`
/// (by comparing against `original` sorted with the std lib).
pub fn assert_sorted_vs_std(original: &[i32], candidate: &[i32]) {
    assert!(
        is_sorted(candidate),
        "candidate is not sorted: {:?}...",
        &candidate[..candidate.len().min(16)]
    );
    let mut expected = original.to_vec();
    expected.sort_unstable();
    assert_eq!(
        candidate,
        &expected[..],
        "candidate differs from std-sorted oracle"
    );
}

/// Tiny deterministic xorshift64 RNG (no dev-dependency needed in lib).
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Deterministic pseudo-random vec in `[lo, hi)` for tests/benches.
pub fn random_vec(n: usize, lo: i32, hi: i32, seed: u64) -> Vec<i32> {
    assert!(hi > lo, "hi must exceed lo");
    let span = (hi as i64 - lo as i64) as u64;
    let mut st = seed.max(1);
    (0..n)
        .map(|_| (xorshift64(&mut st) % span) as i32 + lo)
        .collect()
}
