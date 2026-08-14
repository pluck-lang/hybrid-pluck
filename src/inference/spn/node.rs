//! Generic SPNs: `Spn<L: SpnLeaf>`
//!
//! Core functionality shared across all SPN types
//! Sum nodes and product nodes always have a log_coeff and children
//! However, different SPN leaf types give different semantics to the
//! overall SPN.

use itertools::Itertools;

use super::coeff::{SpnCoeff, SpnLeaf};

use crate::inference::conjugate_pairs::suffstat::ContVarName;
use std::collections::{BTreeSet, HashMap};

use std::hash::{Hash, Hasher};

use std::rc::Rc;

/// The kinds of SPN nodes.
#[derive(Debug, Clone)]
pub enum SpnKind<L: SpnLeaf> {
    /// A pure scalar; the value lives in the outer `Spn::log_coeff`.
    Scalar,
    /// A typed leaf.
    Leaf(L),
    /// A sum of children (addition in the semiring). After
    /// `Spn::sum` normalisation, each child's `log_coeff` is its
    /// branch weight divided by `log_Z`, and the outer node carries
    /// `log_Z = logsumexp_i(wᵢ)`. Same scale-out-the-outer convention
    /// as `Product`.
    Sum(Vec<Spn<L>>),
    /// A product of children with DISJOINT scopes. Children carry
    /// `log_coeff = one()` after normalisation; the outer node holds
    /// the product of their original coefficients.
    Product(Vec<Spn<L>>),
}

#[derive(Debug, Clone)]
pub struct Spn<L: SpnLeaf> {
    pub(crate) log_coeff: L::Coeff,
    pub(crate) inner: Rc<SpnInner<L>>,
}

#[derive(Debug, Clone)]
pub struct SpnInner<L: SpnLeaf> {
    kind: SpnKind<L>,
    scope: BTreeSet<ContVarName>,
}

// ---------------------------------------------------------------------------
// Conditional structural equality + hashing.
//
// Only needed by leaf types that opt into hash-consing (currently
// only EvidenceLeaf). PosteriorLeaf doesn't derive Eq/Hash, and these
// impls are simply absent there.
// ---------------------------------------------------------------------------

impl<L> PartialEq for SpnInner<L>
where
    L: SpnLeaf + PartialEq,
    L::Coeff: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

impl<L> Eq for SpnInner<L>
where
    L: SpnLeaf + Eq,
    L::Coeff: Eq,
{
}

impl<L> Hash for SpnInner<L>
where
    L: SpnLeaf + Hash,
    L::Coeff: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
    }
}

impl<L> PartialEq for SpnKind<L>
where
    L: SpnLeaf + PartialEq,
    L::Coeff: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SpnKind::Scalar, SpnKind::Scalar) => true,
            (SpnKind::Leaf(a), SpnKind::Leaf(b)) => a == b,
            (SpnKind::Sum(c1), SpnKind::Sum(c2)) | (SpnKind::Product(c1), SpnKind::Product(c2)) => {
                c1.len() == c2.len() && c1.iter().zip(c2).all(|(a, b)| spn_structural_eq(a, b))
            }
            _ => false,
        }
    }
}

impl<L> Eq for SpnKind<L>
where
    L: SpnLeaf + Eq,
    L::Coeff: Eq,
{
}

impl<L> Hash for SpnKind<L>
where
    L: SpnLeaf + Hash,
    L::Coeff: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            SpnKind::Scalar => {}
            SpnKind::Leaf(lk) => lk.hash(state),
            SpnKind::Sum(children) | SpnKind::Product(children) => {
                children.len().hash(state);
                for c in children {
                    hash_spn_child(c, state);
                }
            }
        }
    }
}

/// Two children are structurally equal iff their interned inner Rcs
/// point to the same allocation AND their log_coeffs are equal.
///
/// Pointer equality on the inner Rc is a complete structural equality
/// test for interned leaf types, so this is a constant-time check.
/// For non-interned types (e.g. `PosteriorLeaf`), this is still
/// correct but never returns true across separately-constructed nodes.
fn spn_structural_eq<L: SpnLeaf>(a: &Spn<L>, b: &Spn<L>) -> bool
where
    L::Coeff: PartialEq,
{
    a.log_coeff == b.log_coeff && Rc::ptr_eq(&a.inner, &b.inner)
}

fn hash_spn_child<L: SpnLeaf, H: Hasher>(spn: &Spn<L>, state: &mut H)
where
    L::Coeff: Hash,
{
    spn.log_coeff.hash(state);
    (Rc::as_ptr(&spn.inner) as usize).hash(state);
}

// ---------------------------------------------------------------------------
// Generic Spn<L> constructors and accessors.
// ---------------------------------------------------------------------------

impl<L: SpnLeaf> Spn<L> {
    fn new(log_coeff: L::Coeff, kind: SpnKind<L>, scope: BTreeSet<ContVarName>) -> Self {
        Spn {
            log_coeff,
            inner: L::intern(SpnInner { kind, scope }),
        }
    }

    /// A pure scalar carrying `exp(log_coeff)` and an empty scope.
    /// For `RealEps` this builds a scalar with `power = 0`; use
    /// `Spn::from_coeff` to build a scalar that already carries
    /// an epsilon power.
    pub fn scalar(log_coeff: f64) -> Self {
        Spn::new(
            L::Coeff::one().scale_log(log_coeff),
            SpnKind::Scalar,
            BTreeSet::new(),
        )
    }

    /// A pure scalar from a typed coefficient. Equivalent to
    /// `scalar(log_coeff)` for `NotNan<f64>`; for `RealEps` this
    /// preserves the epsilon power.
    pub fn from_coeff(c: L::Coeff) -> Self {
        Spn::new(c, SpnKind::Scalar, BTreeSet::new())
    }

    pub fn zero() -> Self {
        Spn::new(L::Coeff::zero(), SpnKind::Scalar, BTreeSet::new())
    }

    pub fn one() -> Self {
        Spn::new(L::Coeff::one(), SpnKind::Scalar, BTreeSet::new())
    }

    /// Generic leaf constructor: scope is read from the leaf and must
    /// be non-empty (debug-asserted).
    pub fn leaf(l: L) -> Self {
        let scope = l.scope();
        debug_assert!(!scope.is_empty());
        Spn::new(L::Coeff::one(), SpnKind::Leaf(l), scope)
    }

    pub fn is_zero(&self) -> bool {
        self.log_coeff.is_zero()
    }

    pub(crate) fn is_one(&self) -> bool {
        self.log_coeff.is_one() && matches!(self.kind(), SpnKind::Scalar)
    }

    /// Is this a pure scalar (empty scope)?
    pub(crate) fn is_scalar(&self) -> bool {
        matches!(self.kind(), SpnKind::Scalar)
    }

    pub fn log_coeff(&self) -> L::Coeff {
        self.log_coeff
    }

    pub fn kind(&self) -> &SpnKind<L> {
        &self.inner.kind
    }

    pub fn scope(&self) -> &BTreeSet<ContVarName> {
        &self.inner.scope
    }

    /// Raw pointer to the inner node, usable as a cache key.
    /// Two SPNs sharing the same `Rc<SpnInner>` (via hash-consing)
    /// return the same pointer.
    pub fn inner_ptr(&self) -> *const SpnInner<L> {
        Rc::as_ptr(&self.inner)
    }

    /// Create a scaled copy by a plain log-domain scalar. Shares the
    /// inner Rc, so `O(1)`. Epsilon power preserved (for RealEps).
    pub(crate) fn scalar_mul(&self, log_s: f64) -> Self {
        Spn {
            log_coeff: self.log_coeff.scale_log(log_s),
            inner: self.inner.clone(),
        }
    }

    /// Create a scaled copy by a typed coefficient. Shares the inner Rc.
    /// For RealEps this combines log_coeff AND epsilon power.
    pub(crate) fn coeff_mul(&self, factor: L::Coeff) -> Self {
        Spn {
            log_coeff: self.log_coeff.product(factor),
            inner: self.inner.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Generic sum / product normalisation.
//
// `sum` and `product` are fully generic over `L`. They use
// `L::Coeff`'s logsumexp / product / divide / scale_log
// ---------------------------------------------------------------------------

impl<L: SpnLeaf> Spn<L> {
    /// Create a sum node. Normalisation steps:
    ///
    /// 1. **Flatten** nested sums. A grand-child's `log_coeff` is
    ///    multiplied by the inner sum's outer coefficient before being
    ///    spliced in (this preserves linear value).
    /// 2. **Drop zero children.**
    /// 3. **Dedup** children sharing the same interned `inner_ptr`
    ///    (only happens for interned leaf types, e.g. `EvidenceLeaf`),
    ///    by `logsumexp`-ing their log_coeffs.
    /// 4. **Degenerate collapses**: empty → `zero()`; single → that child.
    /// 5. **Normalise**: factor out `log_Z = logsumexp_i(child.log_coeff)`.
    ///    Each child's log_coeff becomes `child.log_coeff / log_Z`;
    ///    the outer node's log_coeff is `log_Z`.
    /// 6. **Canonical order**: sort children by inner pointer so two
    ///    structurally-equal sums hit the same intern entry.
    pub fn sum(children: Vec<Spn<L>>) -> Self {
        let mut flat: Vec<Spn<L>> = Vec::new();
        for child in children {
            if child.is_zero() {
                continue;
            }
            match child.kind() {
                SpnKind::Sum(grandchildren) => {
                    // Distribute child's outer log_coeff into grandchildren.
                    let outer = child.log_coeff;
                    for gc in grandchildren {
                        let scaled = gc.coeff_mul(outer);
                        if !scaled.is_zero() {
                            flat.push(scaled);
                        }
                    }
                }
                _ => flat.push(child),
            }
        }

        // Dedup pointer-equal children (interned types only).
        let mut deduped: Vec<Spn<L>> = Vec::new();
        let mut ptr_to_idx: HashMap<*const SpnInner<L>, usize> = HashMap::new();
        for child in flat {
            let ptr = Rc::as_ptr(&child.inner);
            if let Some(&idx) = ptr_to_idx.get(&ptr) {
                deduped[idx].log_coeff = deduped[idx].log_coeff.logsumexp(child.log_coeff);
            } else {
                ptr_to_idx.insert(ptr, deduped.len());
                deduped.push(child);
            }
        }
        deduped.retain(|c| !c.is_zero());

        if deduped.is_empty() {
            return Self::zero();
        }
        if deduped.len() == 1 {
            return deduped.pop().unwrap();
        }

        // Normalise: factor out log_Z.
        let log_z = deduped
            .iter()
            .fold(L::Coeff::zero(), |acc, c| acc.logsumexp(c.log_coeff));
        // Defensive: with ≥2 non-zero survivors this should be
        // unreachable, but guard against it anyway.
        if log_z.is_zero() {
            return Self::zero();
        }
        for c in deduped.iter_mut() {
            c.log_coeff = c.log_coeff.divide(log_z);
        }
        // Defensive re-check of the degenerate-collapse cases; should
        // be a no-op given the invariants established above.
        deduped.retain(|c| !c.is_zero());
        if deduped.is_empty() {
            return Self::zero();
        }
        if deduped.len() == 1 {
            let mut only = deduped.pop().unwrap();
            only.log_coeff = log_z.product(only.log_coeff);
            return only;
        }

        // Canonical order: pointer-sort. After dedup no two
        // children share a pointer, so no tie-break needed. For
        // non-interned types the order is allocation-dependent but
        // harmless (no interning means no canonicalisation benefit).
        deduped.sort_unstable_by_key(|c| Rc::as_ptr(&c.inner) as usize);

        let scope = deduped.iter().fold(BTreeSet::new(), |mut acc, c| {
            acc.extend(c.scope());
            acc
        });
        Spn::new(log_z, SpnKind::Sum(deduped), scope)
    }

    /// Create a product node. Children must have disjoint scopes
    /// (debug-asserted).
    ///
    /// Normalisation pulls every surviving child's `log_coeff` into
    /// the outer coefficient so each child stays at
    /// `log_coeff = one()`. This maximises hash-consing hits — two
    /// products whose leaves are pointer-equal but have different
    /// per-leaf weights collapse to the same interned node.
    pub fn product(children: Vec<Spn<L>>) -> Self {
        let mut flat: Vec<Spn<L>> = Vec::new();
        let mut scope: BTreeSet<ContVarName> = BTreeSet::new();
        let mut accum: L::Coeff = L::Coeff::one();

        fn pull_out<L: SpnLeaf>(mut child: Spn<L>, flat: &mut Vec<Spn<L>>, accum: &mut L::Coeff) {
            *accum = accum.product(child.log_coeff);
            child.log_coeff = L::Coeff::one();
            if !child.is_scalar() {
                flat.push(child);
            }
        }

        for child in children {
            if child.is_zero() {
                return Self::zero();
            }
            debug_assert!(scope.intersection(child.scope()).collect_vec().is_empty());
            scope.extend(child.scope().iter().copied());
            match child.kind() {
                SpnKind::Product(grandchildren) => {
                    accum = accum.product(child.log_coeff);
                    for gc in grandchildren {
                        pull_out::<L>(gc.clone(), &mut flat, &mut accum);
                    }
                }
                _ => pull_out::<L>(child, &mut flat, &mut accum),
            }
        }

        if flat.is_empty() {
            // No children → pure scalar carrying the accumulated coeff.
            return Self::from_coeff(accum);
        }
        if flat.len() == 1 {
            // One child → return it scaled by the accumulator.
            return flat.pop().unwrap().coeff_mul(accum);
        }

        // Canonical order: pointer-sort (no log_coeff tie-break needed —
        // every surviving child has log_coeff = one()).
        flat.sort_unstable_by_key(|c| Rc::as_ptr(&c.inner) as usize);
        Spn::new(accum, SpnKind::Product(flat), scope)
    }
}
