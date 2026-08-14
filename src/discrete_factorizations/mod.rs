//! Abstraction over compact Boolean-function / knowledge-compilation backends.
//!
//! [`boolean_factorization`] defines the backend-agnostic interface; each
//! concrete backend lives in its own `*_impl` module, feature-gated so exactly
//! one is compiled per build. [`active`] resolves the enabled feature to the
//! `Factorizer` / `BooleanFunction` aliases the rest of the engine names.

pub mod active;
pub mod boolean_factorization;
pub mod wmc;

#[cfg(feature = "backend-cudd")]
pub mod cudd_impl;
#[cfg(feature = "backend-oxidd")]
pub mod oxidd_impl;
#[cfg(feature = "backend-rsdd")]
pub mod rsdd_impl;
#[cfg(feature = "backend-sylvan")]
pub mod sylvan_impl;

pub use active::{BooleanFunction, Factorizer};
pub use boolean_factorization::{BddNode, BooleanFactorization, BooleanFunctionOps, VarId};
pub use wmc::{Semiring, WeightMap, Wmc};
