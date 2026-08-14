//! The Sylvan-backed implementation of [`BooleanFactorization`], via raw
//! `sylvan-sys` FFI.
//!
//! Sylvan is a multi-core, garbage-collected BDD library: a `BDD` is a bare
//! `u64` (an index into a global node table, with the complement bit in the top
//! bit), and its manager is **process-global** and runs on the Lace work-stealing
//! runtime. Two consequences shape this backend:
//!
//! * **Global init once.** `Lace_start` + `sylvan_init_*` set up global state, so
//!   they run exactly once (via a [`Once`]) even though `new()` is called per
//!   inference run. Each factorizer keeps its own interner and variable counter;
//!   sharing the global table across runs is harmless (WMC weights are per-run).
//! * **Tracing GC + refcounted roots.** Sylvan's collector marks from a root
//!   set it can enumerate: the `sylvan_ref`'d nodes, protected pointers, and
//!   in-flight Lace frames. A bare `u64` sitting in a Rust struct is invisible
//!   to that marker, so every live handle must hold a reference. [`SylvanBdd`]
//!   therefore `Sylvan_ref`s on clone and `Sylvan_deref`s on drop.
//!
//! Because the manager is global, run the alternate-backend tests single-threaded
//! (`--test-threads=1`). All `unsafe`/FFI is quarantined in this module.

use std::cell::RefCell;
use std::sync::Once;

use sylvan_sys::bdd::{Sylvan_and, Sylvan_equiv, Sylvan_ite, Sylvan_not, Sylvan_or, Sylvan_xor};
use sylvan_sys::common::{Sylvan_init_package, Sylvan_set_limits};
use sylvan_sys::lace::Lace_start;
use sylvan_sys::mtbdd::{
    Sylvan_count_refs, Sylvan_deref, Sylvan_high, Sylvan_init_bdd, Sylvan_ithvar, Sylvan_low,
    Sylvan_nodecount, Sylvan_ref, Sylvan_var,
};
use sylvan_sys::{BDD, MTBDD_FALSE, MTBDD_TRUE};

use super::boolean_factorization::{
    BddNode, BooleanFactorization, BooleanFunctionOps, FactorizationStats, VarId,
};
use super::wmc::{wmc_fold, Semiring, WeightMap, Wmc};

/// One-time global Lace + Sylvan initialization. `Sylvan_set_limits` sets a
/// memory *cap* (grown lazily); ample for the inference workloads here. GC
/// stays enabled (the library default); live nodes are protected via
/// ref-counting rather than by disabling collection.
static SYLVAN_INIT: Once = Once::new();

fn ensure_sylvan_initialized() {
    SYLVAN_INIT.call_once(|| unsafe {
        Lace_start(1, 0); // single worker
        Sylvan_set_limits(1 << 30, 1, 5);
        Sylvan_init_package();
        Sylvan_init_bdd();
        // GC stays enabled (default) so the table can resize for large BDDs; we
        // keep our nodes alive by ref-counting them on intern instead.
    });
}

/// Sylvan's complement mark lives in the top bit, so the two terminals are the
/// marked and unmarked forms of node 0.
#[inline]
fn is_terminal(b: BDD) -> bool {
    b == MTBDD_TRUE || b == MTBDD_FALSE
}

/// An owning handle to a Sylvan BDD: one `Sylvan_ref` per live handle,
/// released on drop.
///
/// Terminals are exempt from ref/deref. They are never collected, and
/// `Sylvan_ref` is a hash insert into the global roots table rather than a
/// cheap counter bump — `pure_monad` alone mints two constants per call, so
/// keeping them out of that table is worth the branch.
#[derive(Debug)]
pub struct SylvanBdd(BDD);

impl SylvanBdd {
    #[inline]
    fn adopt(b: BDD) -> Self {
        if !is_terminal(b) {
            unsafe { Sylvan_ref(b) };
        }
        SylvanBdd(b)
    }
}

impl Clone for SylvanBdd {
    #[inline]
    fn clone(&self) -> Self {
        SylvanBdd::adopt(self.0)
    }
}

impl Drop for SylvanBdd {
    #[inline]
    fn drop(&mut self) {
        if !is_terminal(self.0) {
            unsafe { Sylvan_deref(self.0) };
        }
    }
}

impl PartialEq for SylvanBdd {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for SylvanBdd {}
impl std::hash::Hash for SylvanBdd {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl BooleanFunctionOps for SylvanBdd {
    #[inline]
    fn true_ptr() -> Self {
        // Sylvan's constants are compile-time values, not manager-owned, so
        // this backend needs no thread-local to answer them.
        SylvanBdd(MTBDD_TRUE)
    }
    #[inline]
    fn false_ptr() -> Self {
        SylvanBdd(MTBDD_FALSE)
    }
    #[inline]
    fn is_true(&self) -> bool {
        self.0 == MTBDD_TRUE
    }
    #[inline]
    fn is_false(&self) -> bool {
        self.0 == MTBDD_FALSE
    }
}

struct SylvanInner {
    num_vars: u32,
}

/// A Sylvan ROBDD factorizer (a per-run view onto the global Sylvan manager).
pub struct SylvanFactorizer {
    inner: RefCell<SylvanInner>,
}

impl SylvanFactorizer {
    /// Create a fresh factorizer, initializing global Sylvan state on first use.
    pub fn new() -> Self {
        ensure_sylvan_initialized();
        SylvanFactorizer {
            inner: RefCell::new(SylvanInner { num_vars: 0 }),
        }
    }

    /// Take a reference on a native result and wrap it. Refs immediately,
    /// before anything else can allocate and trigger a collection.
    #[inline]
    fn wrap(&self, b: BDD) -> SylvanBdd {
        SylvanBdd::adopt(b)
    }
}

impl Default for SylvanFactorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply a binary Sylvan op by id (the manager is global — no manager arg),
/// interning the result.
macro_rules! binop {
    ($self:ident, $a:ident, $b:ident, $op:ident) => {{
        let r = unsafe { $op($a.0, $b.0) };
        $self.wrap(r)
    }};
}

impl BooleanFactorization for SylvanFactorizer {
    type Ptr = SylvanBdd;

    #[inline]
    fn and(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr {
        binop!(self, a, b, Sylvan_and)
    }
    #[inline]
    fn or(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr {
        binop!(self, a, b, Sylvan_or)
    }
    #[inline]
    fn xor(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr {
        binop!(self, a, b, Sylvan_xor)
    }
    #[inline]
    fn iff(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr {
        binop!(self, a, b, Sylvan_equiv)
    }
    #[inline]
    fn ite(&self, f: &Self::Ptr, g: &Self::Ptr, h: &Self::Ptr) -> Self::Ptr {
        let r = unsafe { Sylvan_ite(f.0, g.0, h.0) };
        self.wrap(r)
    }
    #[inline]
    fn negate(&self, a: &Self::Ptr) -> Self::Ptr {
        self.wrap(unsafe { Sylvan_not(a.0) })
    }

    #[inline]
    fn var(&self, v: VarId, polarity: bool) -> Self::Ptr {
        let lit = unsafe { Sylvan_ithvar(v.0 as u32) };
        let node = if polarity {
            lit
        } else {
            unsafe { Sylvan_not(lit) }
        };
        self.wrap(node)
    }

    fn new_var_at_position(&self, _position: usize, polarity: bool) -> (VarId, Self::Ptr) {
        // Sylvan variable index == order level with no cheap mid-order insert, so
        // `position` is a no-op hint and variables are allocated sequentially.
        // WMC/posteriors are order-independent, so only node counts are affected.
        let vn = {
            let mut inner = self.inner.borrow_mut();
            let v = inner.num_vars;
            inner.num_vars += 1;
            v
        };
        let lit = unsafe { Sylvan_ithvar(vn) };
        let node = if polarity {
            lit
        } else {
            unsafe { Sylvan_not(lit) }
        };
        (VarId(vn as u64), self.wrap(node))
    }

    fn node(&self, p: &Self::Ptr) -> BddNode<Self::Ptr> {
        if p.is_false() {
            return BddNode::False;
        }
        if p.is_true() {
            return BddNode::True;
        }
        let b = p.0;
        // `Sylvan_low`/`Sylvan_high` already fold this node's complement bit into
        // the returned children (verified: the honest low/high of `¬v` are the
        // negated children), so they are honest sub-functions as-is — the
        // `BddNode` contract. Interning still distinguishes `f`/`¬f` because the
        // underlying `u64`s differ in the complement bit.
        let var = unsafe { Sylvan_var(b) } as u64;
        let low = unsafe { Sylvan_low(b) };
        let high = unsafe { Sylvan_high(b) };
        BddNode::Inner {
            var: VarId(var),
            low: self.wrap(low),
            high: self.wrap(high),
        }
    }

    #[inline]
    fn node_id(&self, p: &Self::Ptr) -> u64 {
        // The handle *is* the identity: `f` and `¬f` differ in the complement
        // bit of the underlying `u64`.
        p.0
    }

    #[inline]
    fn node_ref_id(&self, p: &Self::Ptr) -> u64 {
        // Sign-insensitive: strip Sylvan's complement mark (the top bit).
        p.0 & !MTBDD_TRUE
    }

    #[inline]
    fn num_vars(&self) -> usize {
        self.inner.borrow().num_vars as usize
    }
    #[inline]
    fn count_nodes(&self, p: &Self::Ptr) -> usize {
        unsafe { Sylvan_nodecount(p.0) }
    }
    fn stats(&self) -> FactorizationStats {
        if std::env::var_os("PLUCK_SYLVAN_STATS").is_some() {
            // The live root set: should rise and fall with the engine's working set.
            eprintln!("sylvan: refs={}", unsafe { Sylvan_count_refs() });
        }
        FactorizationStats {
            num_recursive_calls: None,
        }
    }
}

impl<T: Semiring> Wmc<T> for SylvanFactorizer {
    fn wmc(&self, p: &Self::Ptr, w: &WeightMap<T>) -> T {
        wmc_fold(self, p, w)
    }
}
