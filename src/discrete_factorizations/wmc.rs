//! Weighted model counting: the shared weight semiring and the generic fold.
//!
//! There is one weight abstraction — [`Semiring`] — with a single implementation
//! per weight type (see the `Spn<EvidenceLeaf>` impl beside the SPN arithmetic
//! in `spn::evidence`). [`Wmc`] is the capability the engine calls, implemented
//! per backend: the arena backends (OxiDD/CUDD/Sylvan) count with the generic
//! [`wmc_fold`] over the trait's own traversal, while `rsdd` specializes to its
//! native `unsmoothed_wmc`.

use super::boolean_factorization::{BooleanFactorization, VarId};

/// Per-variable `(low, high)` weights for WMC. Mirrors the variable-weight
/// table of a BDD library's WMC params, plus the default weight for variables
/// absent from the table.
pub struct WeightMap<T> {
    pub one: T,
    /// Dense, indexed by `VarId.0`. `None` means the variable carries the
    /// multiplicative identity weight on both edges.
    pub var_to_val: Vec<Option<(T, T)>>,
}

impl<T: Clone> WeightMap<T> {
    /// A weight map whose absent-variable default weight is `one`, with no
    /// per-variable weights set yet.
    pub fn new(one: T) -> Self {
        WeightMap {
            one,
            var_to_val: Vec::new(),
        }
    }

    /// Set the `(low, high)` weights for `v`, growing the dense table as needed.
    pub fn set_weight(&mut self, v: VarId, low: T, high: T) {
        let i = v.0 as usize;
        if i >= self.var_to_val.len() {
            self.var_to_val.resize_with(i + 1, || None);
        }
        self.var_to_val[i] = Some((low, high));
    }

    /// The `(low, high)` weights for `v`. Absent variables return
    /// `(one, one)` — the unsmoothed identity. In practice every allocated
    /// variable is given a weight immediately, so this default is a safety
    /// net rather than a live semantic path.
    pub fn var_weight(&self, v: VarId) -> (T, T) {
        match self.var_to_val.get(v.0 as usize) {
            Some(Some((l, h))) => (l.clone(), h.clone()),
            _ => (self.one.clone(), self.one.clone()),
        }
    }
}

/// The weight semiring for weighted model counting: a commutative semiring with
/// the `+`/`*` operators plus its two identities. Deliberately tiny and
/// dependency-free so every backend shares one trait and each weight type needs
/// exactly one implementation.
pub trait Semiring: Clone + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self> {
    /// Additive identity (weight of the `false` terminal / an empty sum).
    fn zero() -> Self;
    /// Multiplicative identity (weight of the `true` terminal).
    fn one() -> Self;
}

/// Weighted model counting capability, parameterized by the weight semiring.
/// Implemented once per backend (generic fold or native).
pub trait Wmc<T>: BooleanFactorization {
    /// The weighted model count of `p` under the weights `w`.
    fn wmc(&self, p: &Self::Ptr, w: &WeightMap<T>) -> T;
}

/// A generic, memoized weighted model count over any [`BooleanFactorization`].
/// The path used by every backend without a library-native count.
///
/// Walks the sign-normalized decomposition from [`BooleanFactorization::node`]
/// and reproduces the standard *unsmoothed* recurrence — `true ↦ one`,
/// `false ↦ zero`, and an inner node on `var` with weights `(low_w, high_w)`
/// maps to `low_w · wmc(low) + high_w · wmc(high)`. Memoized on
/// [`BooleanFactorization::node_id`] (sign-distinguishing), since the value
/// differs by sign.
#[cfg(any(
    feature = "backend-oxidd",
    feature = "backend-cudd",
    feature = "backend-sylvan"
))]
pub fn wmc_fold<M, T>(mgr: &M, root: &M::Ptr, w: &WeightMap<T>) -> T
where
    M: BooleanFactorization,
    T: Semiring,
{
    use super::boolean_factorization::{BddNode, BooleanFunctionOps};
    use std::collections::HashMap;

    fn go<M, T>(mgr: &M, p: &M::Ptr, w: &WeightMap<T>, cache: &mut HashMap<u64, T>) -> T
    where
        M: BooleanFactorization,
        T: Semiring,
    {
        // Terminals first (no memo traffic), then the memo, and only then
        // decompose. `node()` materializes owned children, so checking the
        // cache after decomposing — as this used to — took and released two
        // references per hit for nothing. DAG sharing makes hits the common
        // case, so the order matters.
        if p.is_true() {
            return T::one();
        }
        if p.is_false() {
            return T::zero();
        }
        let key = mgr.node_id(p);
        if let Some(v) = cache.get(&key) {
            return v.clone();
        }
        let res = match mgr.node(p) {
            BddNode::True => T::one(),
            BddNode::False => T::zero(),
            BddNode::Inner { var, low, high } => {
                let (low_w, high_w) = w.var_weight(var);
                low_w * go(mgr, &low, w, cache) + high_w * go(mgr, &high, w, cache)
            }
        };
        cache.insert(key, res.clone());
        res
    }
    go(mgr, root, w, &mut HashMap::new())
}
