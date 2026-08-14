//! Abstract interface for compact Boolean-function representations (BDDs,
//! tensor trains, …) used by the exact-inference engine to hold BDD-guarded
//! worlds and run weighted model counting (WMC).
//!
//! The interface is deliberately split, mirroring the natural division in a
//! hash-consed BDD package (and in `rsdd` specifically):
//!
//! - An impl of [`BooleanFunctionOps`] is the cheap *handle* to a Boolean function: not `Copy`,
//!   but a ref-counted `Clone`/`Drop` handle backends use to keep nodes alive. Constants
//!   and constant-tests are answerable without a manager, so pure value code
//!   (formatting, the monad's `pure`/`fail`) can name and inspect handles with
//!   no factorizer in scope.
//! - [`BooleanFactorization`] is the *manager* (arena/builder). All connectives,
//!   variable allocation, and structural traversal go through it, because in a
//!   hash-consed representation those operations need the shared node store.
//!
//! Weighted model counting is a separate capability, [`Wmc`], parameterized by
//! the weight semiring. Keeping it out of the core trait lets a backend choose
//! its own weight types (and bounds) without forcing every backend to satisfy
//! one library's semiring API.

use std::fmt::Debug;
use std::hash::Hash;

/// A handle to a Boolean function: an *owning* reference, answerable about
/// constants without a manager.
///
/// Deliberately **not** `Copy`. A handle keeps its function alive for as long
/// as it exists — `Clone` takes a reference on the underlying node and `Drop`
/// releases it — which is what lets the garbage-collected backends reclaim
/// intermediate nodes as a computation proceeds. Backends whose native pointer
/// is already `Copy + 'static` (rsdd) satisfy this trivially: clone is a copy
/// and there is nothing to drop.
///
/// Constants stay manager-free: each backend answers `true_ptr`/`false_ptr`
/// from its thread-local manager (see each `*_impl`), so pure value code can
/// still name them with no factorizer in scope.
pub trait BooleanFunctionOps: Clone + Eq + Hash + Debug {
    /// The `true` (⊤) constant.
    fn true_ptr() -> Self;
    /// The `false` (⊥) constant.
    fn false_ptr() -> Self;
    /// Whether this handle is exactly the `true` constant.
    fn is_true(&self) -> bool;
    /// Whether this handle is exactly the `false` constant.
    fn is_false(&self) -> bool;
}

/// A representation-agnostic variable identifier. Replaces `rsdd`'s `VarLabel`
/// at the trait boundary; the `u64` is an opaque dense index the manager
/// assigns and orders (see [`BooleanFactorization::new_var_at_position`]).
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct VarId(pub u64);

/// One level of decomposition of a Boolean function.
///
/// Complement/sign is **already pushed into** `low`/`high`, so `Inner`'s
/// children are honest sub-functions and walking them needs no sign
/// bookkeeping. This is what lets the engine's DAG walks (support, marginal,
/// path sampling) be written generically against any factorizer.
pub enum BddNode<P> {
    True,
    False,
    Inner { var: VarId, low: P, high: P },
}

/// Backend-agnostic diagnostics. Fields are `Option` because not every
/// representation has a meaningful notion of each metric (e.g. a backend with
/// no apply cache has no recursive-call count).
pub struct FactorizationStats {
    /// Apply/ITE recursive-call count, if the backend tracks one.
    pub num_recursive_calls: Option<u64>,
}

/// The Boolean-function manager: connectives, variable allocation, structural
/// traversal, and diagnostics.
pub trait BooleanFactorization {
    /// The handle type this factorizer produces and consumes.
    type Ptr: BooleanFunctionOps;

    // ---- logical connectives ----
    // Operands are borrowed and results are owned, mirroring `&str -> String`:
    // an operand the caller still needs is not consumed, and the result is a
    // fresh reference the caller is responsible for.
    fn and(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr;
    fn or(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr;
    fn xor(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr;
    /// If and only if (Boolean equality).
    fn iff(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr;
    /// Ternary if-then-else: `if f then g else h`.
    fn ite(&self, f: &Self::Ptr, g: &Self::Ptr, h: &Self::Ptr) -> Self::Ptr;
    fn negate(&self, a: &Self::Ptr) -> Self::Ptr;

    // ---- variables / allocation ----
    /// A literal for an already-allocated variable at the given polarity.
    fn var(&self, v: VarId, polarity: bool) -> Self::Ptr;
    /// Allocate a fresh variable at order `position` (this is how the engine
    /// controls variable ordering by callstack) and return its id plus a
    /// literal at the given polarity.
    fn new_var_at_position(&self, position: usize, polarity: bool) -> (VarId, Self::Ptr);

    // ---- structural traversal ----
    /// Decompose `p` one level, with complement already pushed into the
    /// children (so `Inner` children are honest sub-functions).
    ///
    /// The children are owned handles, so a walk holds its own references and
    /// stays valid even if the caller drops the root mid-traversal.
    fn node(&self, p: &Self::Ptr) -> BddNode<Self::Ptr>;
    /// A stable identity that **distinguishes** `f` from `¬f`. Used as a memo
    /// key by computations whose value differs by sign (WMC, marginals).
    fn node_id(&self, p: &Self::Ptr) -> u64;
    /// A stable identity that **ignores** sign, so `f` and `¬f` share a key.
    /// Used by sign-insensitive walks (e.g. variable support). The default is
    /// correct but may visit both polarities of a shared node once each;
    /// sign-bit representations should override it to drop the sign bit.
    fn node_ref_id(&self, p: &Self::Ptr) -> u64 {
        self.node_id(p)
    }

    // ---- diagnostics ----
    /// Number of variables the manager has allocated.
    fn num_vars(&self) -> usize;
    /// Node count of the function rooted at `p`.
    fn count_nodes(&self, p: &Self::Ptr) -> usize;
    /// Backend diagnostics; see [`FactorizationStats`].
    fn stats(&self) -> FactorizationStats;
}
