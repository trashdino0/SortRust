//! `sort-core` — faithful Rust ports of the Java + Python sorting work.
//!
//! Sources:
//! - `sorting/src/SortingBenchmarks.java` — `FastRadixSort` (+9 opts, see
//!   [`fast_radix`]).
//! - `sorting_numba.py` — insertion, bucket, quick, merge, radix (base-10),
//!   parallel radix (base-10), parallel radix optimized (base-256),
//!   timsort (serial + parallel).
//!
//! Rules inherited from `learned.md`:
//! - rayon at exactly one level (no nested parallel joins);
//! - warmed release runs, `-a N` stats live in the CLI crate;
//! - correctness via differential tests vs `slice::sort` (see `utils`).

pub mod bucket;
pub mod fast_radix;
pub mod insertion;
pub mod merge;
pub mod parallel_radix;
pub mod parallel_radix_opt;
pub mod quick;
pub mod radix;
pub mod radix16;
pub mod timsort;
pub mod utils;

pub use bucket::bucket_sort;
pub use fast_radix::{fast_radix_sort, fast_radix_sort_with_buf};
pub use insertion::insertion_sort;
pub use merge::merge_sort;
pub use parallel_radix::parallel_radix_sort;
pub use parallel_radix_opt::parallel_radix_sort_optimized;
pub use quick::quicksort;
pub use radix::radix_sort;
pub use radix16::radix16_sort;
pub use timsort::{non_parallel_timsort, parallel_timsort};
