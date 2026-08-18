//! Selection of the active Boolean-factorization backend.
//!
//! The whole engine names the backend through the [`Factorizer`] and
//! [`BooleanFunction`] aliases here rather than any concrete library type.
//! Exactly one `backend-*` cargo feature is enabled per build; switching
//! backends for benchmarking is `--no-default-features --features backend-<x>`
//! plus a recompile.

use super::boolean_factorization::BooleanFactorization;

// Ensure exactly one backend is active
const N_ACTIVE_BACKENDS: usize = cfg!(feature = "backend-rsdd") as usize
    + cfg!(feature = "backend-oxidd") as usize
    + cfg!(feature = "backend-cudd") as usize
    + cfg!(feature = "backend-sylvan") as usize;

const _: () = assert!(
    N_ACTIVE_BACKENDS == 1,
    "select exactly one Boolean-factorization backend: enable exactly one of \
     backend-rsdd, backend-oxidd, backend-cudd, backend-sylvan \
     (use --no-default-features to drop the default rsdd)"
);

// Active Factorizer
#[cfg(feature = "backend-cudd")]
pub use super::cudd_impl::CuddFactorizer as Factorizer;
#[cfg(feature = "backend-oxidd")]
pub use super::oxidd_impl::OxiddFactorizer as Factorizer;
#[cfg(feature = "backend-rsdd")]
pub use super::rsdd_impl::RsddFactorizer as Factorizer;
#[cfg(feature = "backend-sylvan")]
pub use super::sylvan_impl::SylvanFactorizer as Factorizer;

/// The active Boolean-function handle type.
pub type BooleanFunction = <Factorizer as BooleanFactorization>::Ptr;
