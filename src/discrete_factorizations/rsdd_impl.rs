//! The `rsdd`-backed implementation of [`BooleanFactorization`].
//!
//! This is the one module that knows about `rsdd`. It wraps a single
//! `RobddBuilder` whose arena is leaked to `'static` (so pointers can be stored
//! freely in the value graph, caches, and maps without a borrow tying them to
//! the manager). All the `unsafe` needed for that leak is quarantined here.

use std::collections::HashMap;
use std::fmt;

use rsdd::builder::bdd::RobddBuilder;
use rsdd::builder::cache::AllIteTable;
use rsdd::builder::BottomUpBuilder;
use rsdd::repr::{BddPtr, DDNNFPtr, PartialVariableOrder, VarLabel, VarOrder, WmcParams};
use rsdd::util::semirings::Semiring as RsddSemiring;

use super::boolean_factorization::{
    BddNode, BooleanFactorization, BooleanFunctionOps, FactorizationStats, VarId,
};
use super::wmc::{Semiring, WeightMap, Wmc};

/// The concrete BDD builder type, with the arena leaked to `'static`.
type Builder = RobddBuilder<'static, AllIteTable<BddPtr<'static>>>;

#[inline]
fn to_label(v: VarId) -> VarLabel {
    VarLabel::new(v.0)
}

#[inline]
fn from_label(l: VarLabel) -> VarId {
    VarId(l.value())
}

// ---------------------------------------------------------------------------
// The handle
// ---------------------------------------------------------------------------

/// An owning handle to an `rsdd` BDD.
///
/// `rsdd` never reclaims — its arena is leaked to `'static` — so "owning" is
/// free here: `Clone` copies the inner pointer and there is nothing to release
/// on drop. The newtype exists so this backend presents the same non-`Copy`
/// interface as the garbage-collected ones, which is what lets the engine be
/// written once against handles that really do own their nodes.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct RsddBdd(BddPtr<'static>);

impl RsddBdd {
    #[inline]
    fn raw(&self) -> BddPtr<'static> {
        self.0
    }
}

impl BooleanFunctionOps for RsddBdd {
    #[inline]
    fn true_ptr() -> Self {
        RsddBdd(<BddPtr as DDNNFPtr>::true_ptr())
    }
    #[inline]
    fn false_ptr() -> Self {
        RsddBdd(<BddPtr as DDNNFPtr>::false_ptr())
    }
    #[inline]
    fn is_true(&self) -> bool {
        DDNNFPtr::is_true(&self.0)
    }
    #[inline]
    fn is_false(&self) -> bool {
        DDNNFPtr::is_false(&self.0)
    }
}

// ---------------------------------------------------------------------------
// The factorizer
// ---------------------------------------------------------------------------

/// An `rsdd` ROBDD factorizer. Cheap to copy — it is a `'static` reference to
/// the shared, leaked builder.
#[derive(Copy, Clone)]
pub struct RsddFactorizer {
    builder: &'static Builder,
}

impl RsddFactorizer {
    /// Create a fresh factorizer with an empty variable order. The underlying
    /// builder is leaked to `'static`; there is one such leak per call (a small
    /// permanent allocation per inference run).
    pub fn new() -> Self {
        let order = VarOrder::linear_order(0);
        let builder: &'static mut Builder =
            Box::leak(Box::new(RobddBuilder::<AllIteTable<BddPtr>>::new(order)));
        RsddFactorizer { builder }
    }
}

impl Default for RsddFactorizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BooleanFactorization for RsddFactorizer {
    type Ptr = RsddBdd;

    #[inline]
    fn and(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr {
        RsddBdd(self.builder.and(a.raw(), b.raw()))
    }
    #[inline]
    fn or(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr {
        RsddBdd(self.builder.or(a.raw(), b.raw()))
    }
    #[inline]
    fn xor(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr {
        RsddBdd(self.builder.xor(a.raw(), b.raw()))
    }
    #[inline]
    fn iff(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr {
        RsddBdd(self.builder.iff(a.raw(), b.raw()))
    }
    #[inline]
    fn ite(&self, f: &Self::Ptr, g: &Self::Ptr, h: &Self::Ptr) -> Self::Ptr {
        RsddBdd(self.builder.ite(f.raw(), g.raw(), h.raw()))
    }
    #[inline]
    fn negate(&self, a: &Self::Ptr) -> Self::Ptr {
        RsddBdd(self.builder.negate(a.raw()))
    }

    #[inline]
    fn var(&self, v: VarId, polarity: bool) -> Self::Ptr {
        RsddBdd(self.builder.var(to_label(v), polarity))
    }
    #[inline]
    fn new_var_at_position(&self, position: usize, polarity: bool) -> (VarId, Self::Ptr) {
        let (label, ptr) = self.builder.new_var_at_position(position, polarity);
        (from_label(label), RsddBdd(ptr))
    }

    fn node(&self, p: &Self::Ptr) -> BddNode<Self::Ptr> {
        match p.raw() {
            BddPtr::PtrTrue => BddNode::True,
            BddPtr::PtrFalse => BddNode::False,
            BddPtr::Reg(_) | BddPtr::Compl(_) => {
                // Push the complement bit into the children so callers see
                // honest sub-functions (the same idiom the engine's DAG walks
                // used to hand-roll).
                let raw = p.raw();
                let (low, high) = if raw.is_neg() {
                    (raw.low_raw().neg(), raw.high_raw().neg())
                } else {
                    (raw.low_raw(), raw.high_raw())
                };
                let var = from_label(p.raw().var().unwrap());
                BddNode::Inner {
                    var,
                    low: RsddBdd(low),
                    high: RsddBdd(high),
                }
            }
        }
    }

    #[inline]
    fn node_id(&self, p: &Self::Ptr) -> u64 {
        // Sign-distinguishing key: node address in the high bits, complement
        // bit in bit 0.
        match p.raw() {
            BddPtr::PtrTrue => 0,
            BddPtr::PtrFalse => 1,
            BddPtr::Reg(node) => (node as *const _ as u64) << 1,
            BddPtr::Compl(node) => ((node as *const _ as u64) << 1) | 1,
        }
    }

    #[inline]
    fn node_ref_id(&self, p: &Self::Ptr) -> u64 {
        // Drop the complement bit: `f` and `¬f` share a raw node address.
        self.node_id(p) >> 1
    }

    #[inline]
    fn num_vars(&self) -> usize {
        self.builder.num_vars()
    }
    #[inline]
    fn count_nodes(&self, p: &Self::Ptr) -> usize {
        p.raw().count_nodes()
    }
    fn stats(&self) -> FactorizationStats {
        FactorizationStats {
            num_recursive_calls: Some(self.builder.stats().num_recursive_calls as u64),
        }
    }
}

// ---------------------------------------------------------------------------
// WMC: delegate to rsdd's native, memoized `unsmoothed_wmc`.
// ---------------------------------------------------------------------------

/// A generic bridge from our [`Semiring`] to `rsdd`'s, so `rsdd`'s native
/// `unsmoothed_wmc` (which is generic over `rsdd`'s semiring) can count over any
/// of our weight types. Because the bridge is generic, the weight type itself
/// (e.g. `Spn<EvidenceLeaf>`) needs only the one shared [`Semiring`] impl — no
/// second, `rsdd`-specific impl. Unwrap the `.0` to recover the weight.
#[derive(Debug, Clone)]
struct RsddWeight<T>(T);

impl<T: std::ops::Add<Output = T>> std::ops::Add for RsddWeight<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        RsddWeight(self.0 + rhs.0)
    }
}
impl<T: std::ops::Mul<Output = T>> std::ops::Mul for RsddWeight<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        RsddWeight(self.0 * rhs.0)
    }
}
impl<T: fmt::Display> fmt::Display for RsddWeight<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
// `rsdd::Semiring` requires `Debug + Clone + Display + Add + Mul` (all supplied
// above) plus these two identities.
impl<T> RsddSemiring for RsddWeight<T>
where
    T: Semiring + fmt::Debug + fmt::Display,
{
    fn zero() -> Self {
        RsddWeight(T::zero())
    }
    fn one() -> Self {
        RsddWeight(T::one())
    }
}

/// Build an `rsdd` `WmcParams` over the bridge weight from a backend-agnostic
/// [`WeightMap`].
fn to_rsdd_params<T>(w: &WeightMap<T>) -> WmcParams<RsddWeight<T>>
where
    T: Semiring + fmt::Debug + fmt::Display + 'static,
{
    let mut params = WmcParams::new(HashMap::new());
    for (i, entry) in w.var_to_val.iter().enumerate() {
        if let Some((low, high)) = entry {
            params.set_weight(
                VarLabel::new(i as u64),
                RsddWeight(low.clone()),
                RsddWeight(high.clone()),
            );
        }
    }
    params
}

// `rsdd` keeps its native, memoized `unsmoothed_wmc`; we just bridge the weight
// semiring. The extra `Debug + Display` bounds are `rsdd::Semiring`'s and live
// on this impl (satisfied by every real weight type, e.g. `Spn<EvidenceLeaf>`).
impl<T> Wmc<T> for RsddFactorizer
where
    T: Semiring + fmt::Debug + fmt::Display + 'static,
{
    fn wmc(&self, p: &Self::Ptr, w: &WeightMap<T>) -> T {
        p.raw().unsmoothed_wmc(&to_rsdd_params(w)).0
    }
}
