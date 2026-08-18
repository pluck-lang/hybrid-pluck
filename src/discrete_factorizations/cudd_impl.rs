//! The CUDD-backed implementation of [`BooleanFactorization`], via raw
//! `cudd-sys` FFI.
//!
//! CUDD hands out `*mut DdNode` pointers with the complement bit in the
//! pointer's low bit and manual reference counting; a stored pointer is only
//! valid while it is `Cudd_Ref`'d and not garbage-collected.
//!
//! [`CuddBdd`] is therefore an *owning* handle: it `Cudd_Ref`s on clone and
//! `Cudd_RecursiveDeref`s on drop, which is the discipline CUDD's own manual
//! is written around. That is what lets intermediate nodes die — a result of
//! `Cudd_bddAnd` comes back with `ref == 0` and is protected only by whoever
//! references it, so once the last handle to a transient sub-result drops,
//! CUDD counts it dead and sweeps it at the next table growth. This can be 
//! useful in cases where intermediate BDDs are very large and cause memory
//! blowups.
//!
//! Dynamic reordering is disabled so our `new_var_at_position` order is
//! authoritative.
//!
//! All `unsafe`/FFI is quarantined in this module.

use std::cell::Cell;
use std::os::raw::c_int;

use cudd_sys::cudd::{
    Cudd_AutodynDisable, Cudd_DagSize, Cudd_E, Cudd_Init, Cudd_IsComplement, Cudd_NodeReadIndex,
    Cudd_Not, Cudd_ReadCacheHits, Cudd_ReadCacheLookUps, Cudd_ReadDead,
    Cudd_ReadGarbageCollectionTime, Cudd_ReadGarbageCollections, Cudd_ReadLogicZero,
    Cudd_ReadMemoryInUse, Cudd_ReadNodeCount, Cudd_ReadOne, Cudd_ReadPeakLiveNodeCount,
    Cudd_ReadSize, Cudd_RecursiveDeref, Cudd_Ref, Cudd_Regular, Cudd_SetLooseUpTo, Cudd_T,
    Cudd_bddAnd, Cudd_bddIte, Cudd_bddIthVar, Cudd_bddNewVarAtLevel, Cudd_bddOr, Cudd_bddXnor,
    Cudd_bddXor, CUDD_CACHE_SLOTS, CUDD_UNIQUE_SLOTS,
};
use cudd_sys::{DdManager, DdNode};

use super::boolean_factorization::{
    BddNode, BooleanFactorization, BooleanFunctionOps, FactorizationStats, VarId,
};
use super::wmc::{wmc_fold, Semiring, WeightMap, Wmc};

thread_local! {
    /// The manager `true_ptr`/`false_ptr` answer from.
    ///
    /// `BooleanFunctionOps` mints constants without a factorizer in scope, but
    /// CUDD's constants belong to a manager, so the manager has to be reachable
    /// from somewhere. It is thread-local rather than global because tests run
    /// one manager per thread; a process-wide static would hand one test
    /// another test's constants (observed: an immediate SIGSEGV).
    ///
    /// Only constant *construction* reads this. `Drop`, `is_true` and
    /// `is_false` use the manager stored in the handle itself, so a handle
    /// created by an earlier run stays valid even after a later
    /// `CuddFactorizer::new` re-points this cell.
    static CUDD_MGR: Cell<*mut DdManager> = const { Cell::new(std::ptr::null_mut()) };
}

/// Unique-table slots to allow before a collection is considered; override with
/// `PLUCK_CUDD_LOOSE` (a larger value collects less often).
const DEFAULT_LOOSE_UP_TO: u32 = 100_000;

fn loose_up_to() -> u32 {
    std::env::var("PLUCK_CUDD_LOOSE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_LOOSE_UP_TO)
}

fn current_mgr() -> *mut DdManager {
    let m = CUDD_MGR.with(|c| c.get());
    assert!(
        !m.is_null(),
        "CuddBdd constant requested before any CuddFactorizer was created on this thread"
    );
    m
}

/// An owning handle to a CUDD node: one outstanding `Cudd_Ref` per live
/// handle, released on drop.
///
/// Carries its own manager so `Drop` never has to consult thread-local state —
/// handles may be dropped long after (and in a different order than) the
/// factorizer that made them.
#[derive(Debug)]
pub struct CuddBdd {
    node: *mut DdNode,
    mgr: *mut DdManager,
}

impl CuddBdd {
    /// Take a reference on `node` and wrap it. `node` must be live at the call
    /// (either freshly returned by CUDD, or reachable from a handle we hold);
    /// the ref is taken before anything else can allocate and trigger a
    /// collection.
    #[inline]
    fn adopt(mgr: *mut DdManager, node: *mut DdNode) -> Self {
        assert!(!node.is_null(), "CUDD returned a null node (out of memory)");
        unsafe { Cudd_Ref(node) };
        CuddBdd { node, mgr }
    }
}

impl Clone for CuddBdd {
    #[inline]
    fn clone(&self) -> Self {
        unsafe { Cudd_Ref(self.node) };
        CuddBdd {
            node: self.node,
            mgr: self.mgr,
        }
    }
}

impl Drop for CuddBdd {
    #[inline]
    fn drop(&mut self) {
        // Balanced against the ref taken in `adopt`/`clone`. Constants are
        // dereffed like anything else: the manager holds its own reference to
        // `one`, so a balanced pair can never drive it to zero.
        unsafe { Cudd_RecursiveDeref(self.mgr, self.node) };
    }
}

// Identity is the node pointer; the manager is incidental. (Handles are
// `!Send`/`!Sync` via the raw pointers, so two managers' handles cannot meet.)
impl PartialEq for CuddBdd {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}
impl Eq for CuddBdd {}
impl std::hash::Hash for CuddBdd {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.node.hash(state);
    }
}

impl BooleanFunctionOps for CuddBdd {
    #[inline]
    fn true_ptr() -> Self {
        let mgr = current_mgr();
        CuddBdd::adopt(mgr, unsafe { Cudd_ReadOne(mgr) })
    }
    #[inline]
    fn false_ptr() -> Self {
        let mgr = current_mgr();
        CuddBdd::adopt(mgr, unsafe { Cudd_ReadLogicZero(mgr) })
    }
    #[inline]
    fn is_true(&self) -> bool {
        self.node == unsafe { Cudd_ReadOne(self.mgr) }
    }
    #[inline]
    fn is_false(&self) -> bool {
        self.node == unsafe { Cudd_ReadLogicZero(self.mgr) }
    }
}

/// A CUDD ROBDD factorizer.
pub struct CuddFactorizer {
    mgr: *mut DdManager,
}

impl CuddFactorizer {
    /// Create a fresh factorizer and make it this thread's constant source.
    /// Disables dynamic reordering so our variable order is authoritative.
    pub fn new() -> Self {
        let mgr = unsafe { Cudd_Init(0, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0) };
        assert!(!mgr.is_null(), "Cudd_Init failed");
        unsafe { Cudd_AutodynDisable(mgr) };
        unsafe { Cudd_SetLooseUpTo(mgr, loose_up_to()) };
        CUDD_MGR.with(|c| c.set(mgr));
        CuddFactorizer { mgr }
    }

    /// Wrap a native result as an owned handle.
    #[inline]
    fn wrap(&self, node: *mut DdNode) -> CuddBdd {
        CuddBdd::adopt(self.mgr, node)
    }
}

impl Default for CuddFactorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply a binary CUDD op, taking a reference on the result.
macro_rules! binop {
    ($self:ident, $a:ident, $b:ident, $op:ident) => {{
        let r = unsafe { $op($self.mgr, $a.node, $b.node) };
        $self.wrap(r)
    }};
}

impl BooleanFactorization for CuddFactorizer {
    type Ptr = CuddBdd;

    #[inline]
    fn and(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr {
        binop!(self, a, b, Cudd_bddAnd)
    }
    #[inline]
    fn or(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr {
        binop!(self, a, b, Cudd_bddOr)
    }
    #[inline]
    fn xor(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr {
        binop!(self, a, b, Cudd_bddXor)
    }
    #[inline]
    fn iff(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr {
        binop!(self, a, b, Cudd_bddXnor)
    }
    #[inline]
    fn ite(&self, f: &Self::Ptr, g: &Self::Ptr, h: &Self::Ptr) -> Self::Ptr {
        let r = unsafe { Cudd_bddIte(self.mgr, f.node, g.node, h.node) };
        self.wrap(r)
    }
    #[inline]
    fn negate(&self, a: &Self::Ptr) -> Self::Ptr {
        // Complement is a pointer-bit flip onto the same underlying node, so
        // the new handle takes its own reference on that shared node.
        self.wrap(unsafe { Cudd_Not(a.node) })
    }

    #[inline]
    fn var(&self, v: VarId, polarity: bool) -> Self::Ptr {
        let lit = unsafe { Cudd_bddIthVar(self.mgr, v.0 as c_int) };
        let node = if polarity {
            lit
        } else {
            unsafe { Cudd_Not(lit) }
        };
        self.wrap(node)
    }

    fn new_var_at_position(&self, position: usize, polarity: bool) -> (VarId, Self::Ptr) {
        // CUDD's NewVarAtLevel inserts at the given order level (shifting the
        // rest), matching rsdd's insert-at-position semantics — so our variable
        // order is honored. `VarId` is the stable variable *index* (invariant
        // under later level shifts), which `var()`/`node()` also use.
        let lit = unsafe { Cudd_bddNewVarAtLevel(self.mgr, position as c_int) };
        assert!(!lit.is_null(), "Cudd_bddNewVarAtLevel failed");
        let index = unsafe { Cudd_NodeReadIndex(lit) } as u64;
        let node = if polarity {
            lit
        } else {
            unsafe { Cudd_Not(lit) }
        };
        (VarId(index), self.wrap(node))
    }

    fn node(&self, p: &Self::Ptr) -> BddNode<Self::Ptr> {
        if p.is_false() {
            return BddNode::False;
        }
        if p.is_true() {
            return BddNode::True;
        }
        let np = p.node;
        // Push the incoming complement bit into both children so they are honest
        // sub-functions (`Cudd_T` = high/var-true, `Cudd_E` = low/var-false).
        let (var, high, low) = unsafe {
            let comp = Cudd_IsComplement(np) != 0;
            let reg = Cudd_Regular(np);
            let var = Cudd_NodeReadIndex(reg) as u64;
            let t = Cudd_T(reg);
            let e = Cudd_E(reg);
            let high = if comp { Cudd_Not(t) } else { t };
            let low = if comp { Cudd_Not(e) } else { e };
            (var, high, low)
        };
        // The children are handed out as owned handles, so each takes its own
        // reference: a walk stays valid even if the caller drops `p`.
        BddNode::Inner {
            var: VarId(var),
            low: self.wrap(low),
            high: self.wrap(high),
        }
    }

    #[inline]
    fn node_id(&self, p: &Self::Ptr) -> u64 {
        // Sign-distinguishing: a complemented pointer differs from its regular
        // form in the low bit.
        p.node as u64
    }

    #[inline]
    fn node_ref_id(&self, p: &Self::Ptr) -> u64 {
        // Sign-insensitive: mask off CUDD's complement bit so `f` and `¬f`
        // share a key (halves the work in `support_vars`).
        unsafe { Cudd_Regular(p.node) as u64 }
    }

    #[inline]
    fn num_vars(&self) -> usize {
        unsafe { Cudd_ReadSize(self.mgr) as usize }
    }
    #[inline]
    fn count_nodes(&self, p: &Self::Ptr) -> usize {
        unsafe { Cudd_DagSize(p.node) as usize }
    }
    fn stats(&self) -> FactorizationStats {
        if std::env::var_os("PLUCK_CUDD_STATS").is_some() {
            self.report_cudd_counters();
        }
        FactorizationStats {
            num_recursive_calls: None,
        }
    }
}

impl CuddFactorizer {
    fn report_cudd_counters(&self) {
        unsafe {
            eprintln!(
                "cudd: live={} dead={} peak_live={} gc_runs={} gc_ms={} mem_mb={:.1} \
                 cache_lookups={} cache_hits={}",
                Cudd_ReadNodeCount(self.mgr),
                Cudd_ReadDead(self.mgr),
                Cudd_ReadPeakLiveNodeCount(self.mgr),
                Cudd_ReadGarbageCollections(self.mgr),
                Cudd_ReadGarbageCollectionTime(self.mgr),
                Cudd_ReadMemoryInUse(self.mgr) as f64 / 1.0e6,
                Cudd_ReadCacheLookUps(self.mgr),
                Cudd_ReadCacheHits(self.mgr),
            );
        }
    }
}

impl<T: Semiring> Wmc<T> for CuddFactorizer {
    fn wmc(&self, p: &Self::Ptr, w: &WeightMap<T>) -> T {
        wmc_fold(self, p, w)
    }
}
