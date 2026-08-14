//! `SpnCoeff` and `SpnLeaf`: the two trait knobs that let `Spn` be
//! generic over (a) the scalar coefficient type carried on each node and
//! (b) the leaf type.
//!
//! Two `Spn` instantiations live in this crate:
//!
//! - `Spn<EvidenceLeaf>` (a.k.a. `EvidenceSpn`) — `Coeff = NotNan<f64>`;
//!   hash-consed via a thread-local intern table.
//! - `Spn<PosteriorLeaf>` (a.k.a. `PosteriorSpn`) — `Coeff = RealEps`,
//!   so per-mixture-branch weights can carry an epsilon power; **not**
//!   interned (posterior trees are produced once and consumed once).
//!
//! The generic skeleton (constructors, sum/product normalisation,
//! scope tracking) lives in `spn.rs` and is fully sharable. Operations
//! that touch suff stats / priors (`spn_mul`, `log_likelihood`,
//! `log_marginal_likelihood`, `posterior`, `Semiring`) live in the
//! evidence-only `impl Spn<EvidenceLeaf>` block.

use std::collections::BTreeSet;
use std::fmt::Debug;
use std::rc::Rc;

use ordered_float::NotNan;

use crate::utils::epsilon::RealEps;
use crate::utils::math::logsumexp;

use crate::inference::conjugate_pairs::ContVarName;

/// Log-domain scalar coefficient carried on every `Spn` node.
///
/// Two implementations:
/// - `NotNan<f64>` for evidence SPNs: plain log-domain numbers.
/// - `RealEps` for posterior SPNs: log-coefficient plus an epsilon
///   power, so disintegrated (Dirac-restricted) branches can be
///   distinguished from density branches.
pub trait SpnCoeff: Copy + Debug {
    /// The semiring zero (linear `0`, log `−∞`).
    fn zero() -> Self;

    /// The semiring one (linear `1`, log `0`, power `0` for RealEps).
    fn one() -> Self;

    fn is_zero(self) -> bool;
    fn is_one(self) -> bool;

    /// Linear-domain addition (`logsumexp`).
    fn logsumexp(self, rhs: Self) -> Self;

    /// Linear-domain multiplication
    fn product(self, rhs: Self) -> Self;

    /// Multiply by a plain log-domain scalar (no epsilon contribution).
    fn scale_log(self, log_s: f64) -> Self;

    /// Linear-domain division (log-domain subtraction).
    ///
    /// Called by `Spn::sum` after `logsumexp` has computed `log_Z`
    /// over all surviving children.
    fn divide(self, rhs: Self) -> Self;
}

impl SpnCoeff for NotNan<f64> {
    fn zero() -> Self {
        NotNan::new(f64::NEG_INFINITY).unwrap()
    }
    fn one() -> Self {
        NotNan::new(0.0).unwrap()
    }
    fn is_zero(self) -> bool {
        self.into_inner() == f64::NEG_INFINITY
    }
    fn is_one(self) -> bool {
        self.into_inner().abs() < 1e-15
    }
    fn logsumexp(self, rhs: Self) -> Self {
        NotNan::new(logsumexp(self.into_inner(), rhs.into_inner())).unwrap()
    }
    fn product(self, rhs: Self) -> Self {
        self + rhs
    }
    fn scale_log(self, log_s: f64) -> Self {
        self + NotNan::new(log_s).unwrap()
    }
    fn divide(self, rhs: Self) -> Self {
        NotNan::new(self.into_inner() - rhs.into_inner()).unwrap()
    }
}

impl SpnCoeff for RealEps {
    fn zero() -> Self {
        RealEps::zero()
    }
    fn one() -> Self {
        RealEps::from_log(0.0, 0)
    }
    fn is_zero(self) -> bool {
        RealEps::is_zero(&self)
    }
    fn is_one(self) -> bool {
        self.power == 0 && self.log_coeff.abs() < 1e-15
    }
    fn logsumexp(self, rhs: Self) -> Self {
        // RealEps::Add IS logsumexp with min-power filtering.
        self + rhs
    }
    fn product(self, rhs: Self) -> Self {
        // RealEps::Mul adds log_coeffs and powers.
        self * rhs
    }
    fn scale_log(self, log_s: f64) -> Self {
        RealEps::scale_log(&self, log_s)
    }
    fn divide(self, rhs: Self) -> Self {
        if self.power > rhs.power {
            // `self / rhs` would have positive epsilon power —
            // infinitesimal relative to rhs. In Spn::sum's normalised
            // weight space (where `rhs = log_Z` is the dominant sum),
            // that's exactly zero.
            Self::zero()
        } else {
            // self.power <= rhs.power. Under `Spn::sum`'s contract
            // (rhs = logsumexp over a set including self), this is
            // strict equality. Lesser would mean self dominates rhs;
            // that's unreachable under the invariant — if it were
            // reached, `power: self.power - rhs.power` (u32) would
            // underflow.
            self / rhs
        }
    }
}

/// Leaf type carried in `SpnKind::Leaf(_)`.
pub trait SpnLeaf: Clone + Debug + Sized {
    /// The scalar coefficient type carried on every `Spn<Self>` node.
    type Coeff: SpnCoeff;

    fn scope(&self) -> BTreeSet<ContVarName>;

    /// Wrap an `SpnInner` in an `Rc`, optionally deduplicating against
    /// a per-type intern table.
    ///
    /// Default: `Rc::new(inner)` — no interning. `EvidenceLeaf` overrides
    /// this to look up the thread-local intern table; on a hit, the
    /// existing `Rc` is cloned and returned, so structurally-equal
    /// sub-DAGs share a single allocation and pointer equality
    /// (`Rc::ptr_eq`) becomes a complete structural equality test.
    fn intern(inner: super::node::SpnInner<Self>) -> Rc<super::node::SpnInner<Self>> {
        Rc::new(inner)
    }
}
