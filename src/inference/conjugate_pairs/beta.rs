use std::collections::BTreeSet;
use std::fmt::{Display, Formatter, Result};
use std::hash::{Hash, Hasher};

use itertools::Itertools;
use ordered_float::NotNan;

use super::suffstat::{ContVarName, Prior, SuffStat, PRIOR_QUANTIZATION_LEVEL};
use crate::utils::epsilon::RealEps;
use crate::utils::math::{log_beta_pdf, log_beta_ratio, quant};

/// A Beta prior over a single variable.
///
/// Two shapes:
/// - `Beta(a, b)`: a standard Beta prior.
/// - `Pinned(r)`: a Dirac prior at `r` (from disintegration).
///
/// The `N` parameter is the variable name type. Internal use carries
/// `N = ContVarName` (the default); display / serialization use can
/// instantiate with `N = String` to attach human-readable names.
/// Inference-related impls (`Prior`, `SuffStat::ConjugatePrior`,
/// `.name()`) only exist for `N = ContVarName`; for other `N` the
/// type is purely a data container.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum BetaPrior<N = ContVarName> {
    Beta { name: N, a: f64, b: f64 },
    Pinned { name: N, value: f64 },
}

impl<N: Clone> BetaPrior<N> {
    pub fn name(&self) -> N {
        match self {
            BetaPrior::Beta { name, .. } => name.clone(),
            BetaPrior::Pinned { name, .. } => name.clone(),
        }
    }
}

impl BetaPrior<ContVarName> {
    /// Re-tag this prior with a display name. Used by the lib layer to
    /// produce client-facing `BetaPrior<String>` values without
    /// duplicating the variant data.
    pub fn with_display_name(self, name: String) -> BetaPrior<String> {
        match self {
            BetaPrior::Beta { a, b, .. } => BetaPrior::Beta { name, a, b },
            BetaPrior::Pinned { value, .. } => BetaPrior::Pinned { name, value },
        }
    }
}

impl<N: Hash> Hash for BetaPrior<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            BetaPrior::Beta { name, a, b } => {
                "Beta".hash(state);
                name.hash(state);
                quant(*a, PRIOR_QUANTIZATION_LEVEL).hash(state);
                quant(*b, PRIOR_QUANTIZATION_LEVEL).hash(state);
            }
            BetaPrior::Pinned { name, value } => {
                "Pinned".hash(state);
                name.hash(state);
                quant(*value, PRIOR_QUANTIZATION_LEVEL).hash(state);
            }
        }
    }
}

impl<N: PartialEq> PartialEq for BetaPrior<N> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                BetaPrior::Beta {
                    name: n1,
                    a: a1,
                    b: b1,
                },
                BetaPrior::Beta {
                    name: n2,
                    a: a2,
                    b: b2,
                },
            ) => {
                n1 == n2
                    && quant(*a1, PRIOR_QUANTIZATION_LEVEL) == quant(*a2, PRIOR_QUANTIZATION_LEVEL)
                    && quant(*b1, PRIOR_QUANTIZATION_LEVEL) == quant(*b2, PRIOR_QUANTIZATION_LEVEL)
            }
            (
                BetaPrior::Pinned {
                    name: n1,
                    value: v1,
                },
                BetaPrior::Pinned {
                    name: n2,
                    value: v2,
                },
            ) => {
                n1 == n2
                    && quant(*v1, PRIOR_QUANTIZATION_LEVEL) == quant(*v2, PRIOR_QUANTIZATION_LEVEL)
            }
            _ => false,
        }
    }
}

impl<N: Eq> Eq for BetaPrior<N> {}

impl<N: Display + Hash> Display for BetaPrior<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            BetaPrior::Beta { name, a, b } => {
                write!(
                    f,
                    "{} ~ Beta({:.p$}, {:.p$})",
                    name,
                    a,
                    b,
                    p = crate::utils::display_precision::param_precision()
                )
            }
            BetaPrior::Pinned { name, value } => {
                write!(
                    f,
                    "{} = {:.p$}",
                    name,
                    value,
                    p = crate::utils::display_precision::param_precision()
                )
            }
        }
    }
}

impl Prior for BetaPrior<ContVarName> {
    type Realization = NotNan<f64>;

    fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_ {
        std::iter::once(match self {
            BetaPrior::Beta { name, .. } => name,
            BetaPrior::Pinned { name, .. } => name,
        })
    }

    /// Draw a realisation from this Beta prior.
    /// - `Beta(a, b)`: standard Beta draw.
    /// - `Pinned(v)`: returns `v`.
    fn sample<R: rand::Rng>(&self, rng: &mut R) -> NotNan<f64> {
        use crate::utils::sampling::sample_beta;
        let value: f64 = match self {
            BetaPrior::Beta { a, b, .. } => sample_beta(rng, *a, *b),
            BetaPrior::Pinned { value, .. } => *value,
        };
        NotNan::new(value).expect("Beta sample is NaN")
    }

    fn push_into(&self, sampled: NotNan<f64>, out: &mut super::AssignmentBuilder) {
        out.push_beta(self.name(), sampled);
    }
}

impl From<BetaPrior> for super::PosteriorLeaf {
    fn from(p: BetaPrior) -> Self {
        super::PosteriorLeaf::Beta(p)
    }
}

// ---------------------------------------------------------------------------
// BetaFlip: per-family BDD-flip variants. Lives here (not in
// lazy_kc/state.rs) so adding a new Beta-side flip variant — and the
// EvidenceLeaf weights it produces — happens in one place.
// ---------------------------------------------------------------------------

use crate::inference::conjugate_pairs::EvidenceLeaf;
use crate::inference::spn::node::Spn;

/// Beta-family BDD-flip variants.
pub enum BetaFlip {
    /// Flip of a symbolic Beta variable — weight tracks sufficient statistics.
    Var(ContVarName),
    /// Probability pinning for disintegration: `prob_eq(name, r)`.
    /// Treated as a pinned observation rather than a flip target.
    ProbPin(ContVarName, f64),
}

impl BetaFlip {
    /// SPN-weight pair `(low_edge, high_edge)` carried on the BDD variable.
    pub fn weights(&self) -> (Spn<EvidenceLeaf>, Spn<EvidenceLeaf>) {
        match self {
            BetaFlip::Var(name) => (
                Spn::leaf(EvidenceLeaf::Beta(BetaSuffStat::counts(*name, 0, 1))),
                Spn::leaf(EvidenceLeaf::Beta(BetaSuffStat::counts(*name, 1, 0))),
            ),
            BetaFlip::ProbPin(name, r) => (
                Spn::leaf(EvidenceLeaf::Beta(BetaSuffStat::not_eq(*name, *r))),
                Spn::leaf(EvidenceLeaf::Beta(BetaSuffStat::real_eq(*name, *r))),
            ),
        }
    }

    /// Per-variant hash discriminator used as part of the BDD-variable
    /// callstack key.
    pub fn discriminator(&self) -> u64 {
        match self {
            BetaFlip::Var(name) => name.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1),
            BetaFlip::ProbPin(name, r) => {
                let r_bits = r.to_bits();
                name.wrapping_mul(0x517CC1B727220A95).wrapping_add(r_bits)
            }
        }
    }

    /// `ProbPin` is a measure-zero observation — looked up by callstack
    /// hash alone in `observation_vars`. `Var` is a regular flip.
    pub fn is_pinned_observation(&self) -> bool {
        matches!(self, BetaFlip::ProbPin(..))
    }

    /// Probability used by the sample-mode flip resolver. Returns `None`
    /// for variants that aren't flip-target probabilities (`ProbPin`
    /// is an observation, not a target).
    pub fn sample_probability(&self, a: &super::Assignment) -> Option<f64> {
        match self {
            BetaFlip::Var(name) => Some(a.beta_value(*name).into_inner()),
            BetaFlip::ProbPin(..) => None,
        }
    }
}

/// Sufficient statistics for a single Beta variable.
///
/// `Counts { h, t, excluded }`: h heads, t tails, plus a set of explicitly
///     excluded values. Excluded values have Lebesgue measure 0 under
///     Beta so they do not change the density; they are kept to reject
///     a later `RealEq(r)` observation with `r ∈ excluded`.
/// `RealEq { value }`: the variable was pinned (Dirac observation).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BetaSuffStat {
    Counts {
        name: ContVarName,
        h: u64,
        t: u64,
        excluded: BTreeSet<NotNan<f64>>,
    },
    RealEq {
        name: ContVarName,
        value: NotNan<f64>,
    },
}

impl BetaSuffStat {
    pub fn counts(name: ContVarName, h: u64, t: u64) -> Self {
        Self::Counts {
            name,
            h,
            t,
            excluded: BTreeSet::new(),
        }
    }

    pub fn not_eq(name: ContVarName, r: f64) -> Self {
        let mut excluded = BTreeSet::new();
        excluded.insert(NotNan::new(r).unwrap());
        Self::Counts {
            name,
            h: 0,
            t: 0,
            excluded,
        }
    }

    pub fn real_eq(name: ContVarName, r: f64) -> Self {
        Self::RealEq {
            name,
            value: NotNan::new(r).unwrap(),
        }
    }

    pub fn name(&self) -> ContVarName {
        match self {
            BetaSuffStat::Counts { name, .. } => *name,
            BetaSuffStat::RealEq { name, .. } => *name,
        }
    }
}

// Counts × RealEq correction factor: r^h * (1-r)^t in log domain.
fn counts_real_factor(h: u64, t: u64, r: f64) -> f64 {
    let mut lf = 0.0;
    if h > 0 {
        if r <= 0.0 {
            return f64::NEG_INFINITY;
        }
        lf += h as f64 * r.ln();
    }
    if t > 0 {
        if r >= 1.0 {
            return f64::NEG_INFINITY;
        }
        lf += t as f64 * (1.0 - r).ln();
    }
    lf
}

fn beta_log_marginal_likelihood(stat: &BetaSuffStat, prior: &BetaPrior<ContVarName>) -> RealEps {
    debug_assert_eq!(vec![&stat.name()], prior.scope().collect_vec());
    match (stat, prior) {
        (BetaSuffStat::Counts { h, t, .. }, BetaPrior::Beta { a, b, .. }) => {
            // Exclusion set has Lebesgue measure 0 under Beta: no density effect.
            RealEps::from_log(log_beta_ratio(*a + *h as f64, *b + *t as f64, *a, *b), 0)
        }
        (BetaSuffStat::Counts { h, t, excluded, .. }, BetaPrior::Pinned { value: r, .. }) => {
            if excluded.contains(&NotNan::new(*r).unwrap()) {
                return RealEps::zero();
            }
            let mut log_val = 0.0;
            if *h > 0 {
                if *r <= 0.0 {
                    return RealEps::zero();
                }
                log_val += *h as f64 * r.ln();
            }
            if *t > 0 {
                if *r >= 1.0 {
                    return RealEps::zero();
                }
                log_val += *t as f64 * (1.0 - r).ln();
            }
            RealEps::from_log(log_val, 0)
        }
        (BetaSuffStat::RealEq { value: r, .. }, BetaPrior::Beta { a, b, .. }) => {
            RealEps::from_log(log_beta_pdf(**r, *a, *b), 1)
        }
        (BetaSuffStat::RealEq { value: r, .. }, BetaPrior::Pinned { value: s, .. }) => {
            if r.into_inner() == *s {
                RealEps::scalar(1.0)
            } else {
                RealEps::zero()
            }
        }
    }
}

impl SuffStat for BetaSuffStat {
    type ConjugatePrior = BetaPrior;

    fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_ {
        std::iter::once(match self {
            BetaSuffStat::Counts { name, .. } => name,
            BetaSuffStat::RealEq { name, .. } => name,
        })
    }

    fn merge(&self, other: &Self) -> (Self, f64) {
        debug_assert_eq!(
            self.name(),
            other.name(),
            "BetaSuffStat::merge requires overlapping scopes"
        );
        match (self, other) {
            (
                BetaSuffStat::Counts {
                    name,
                    h: h1,
                    t: t1,
                    excluded: e1,
                },
                BetaSuffStat::Counts {
                    h: h2,
                    t: t2,
                    excluded: e2,
                    ..
                },
            ) => {
                let mut excluded = e1.clone();
                excluded.extend(e2.iter().copied());
                (
                    BetaSuffStat::Counts {
                        name: *name,
                        h: h1 + h2,
                        t: t1 + t2,
                        excluded,
                    },
                    0.0,
                )
            }
            (
                BetaSuffStat::Counts { h, t, excluded, .. },
                BetaSuffStat::RealEq { name, value: r },
            )
            | (
                BetaSuffStat::RealEq { name, value: r },
                BetaSuffStat::Counts { h, t, excluded, .. },
            ) => {
                if excluded.contains(r) {
                    (
                        BetaSuffStat::RealEq {
                            name: *name,
                            value: *r,
                        },
                        f64::NEG_INFINITY,
                    )
                } else {
                    let log_factor = counts_real_factor(*h, *t, **r);
                    (
                        BetaSuffStat::RealEq {
                            name: *name,
                            value: *r,
                        },
                        log_factor,
                    )
                }
            }
            (BetaSuffStat::RealEq { name, value: r1 }, BetaSuffStat::RealEq { value: r2, .. }) => {
                if r1 != r2 {
                    (
                        BetaSuffStat::RealEq {
                            name: *name,
                            value: *r1,
                        },
                        f64::NEG_INFINITY,
                    )
                } else {
                    (
                        BetaSuffStat::RealEq {
                            name: *name,
                            value: *r1,
                        },
                        0.0,
                    )
                }
            }
        }
    }

    fn log_likelihood(&self, value: &NotNan<f64>) -> f64 {
        debug_assert!(
            matches!(self, BetaSuffStat::RealEq { .. })
                || (value.into_inner() > 0.0 && value.into_inner() < 1.0),
            "BetaSuffStat::log_likelihood: Counts requires value ∈ (0, 1); got {}",
            value.into_inner()
        );
        match self {
            BetaSuffStat::Counts { h, t, excluded, .. } => {
                if excluded.contains(value) {
                    f64::NEG_INFINITY
                } else {
                    (**value).ln() * (*h as f64) + (1.0 - **value).ln() * (*t as f64)
                }
            }
            BetaSuffStat::RealEq {
                value: pinned_value,
                ..
            } => {
                if value == pinned_value {
                    0.0
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
    }

    fn posterior(
        &self,
        prior: &BetaPrior,
        query_vars: &BTreeSet<ContVarName>,
    ) -> (Option<BetaPrior>, RealEps) {
        debug_assert_eq!(self.scope().collect_vec(), prior.scope().collect_vec());
        let log_marginal_likelihood = beta_log_marginal_likelihood(self, prior);
        if !query_vars.contains(&self.name()) {
            return (None, log_marginal_likelihood);
        }
        if log_marginal_likelihood.is_zero() {
            return (None, log_marginal_likelihood);
        }
        let (a, b) = match prior {
            BetaPrior::Beta { a, b, .. } => (*a, *b),
            _ => panic!("BetaSuffStat::posterior: only Beta priors are supported"),
        };
        let name = self.name();
        (
            Some(match self {
                BetaSuffStat::Counts { h, t, .. } => BetaPrior::Beta {
                    name,
                    a: a + *h as f64,
                    b: b + *t as f64,
                },
                BetaSuffStat::RealEq { value: r, .. } => BetaPrior::Pinned {
                    name,
                    value: r.into_inner(),
                },
            }),
            log_marginal_likelihood,
        )
    }

    fn read_realization(&self, a: &super::Assignment) -> NotNan<f64> {
        a.beta_value(self.name())
    }

    fn lookup_prior_in(&self, reg: &super::PriorRegistry) -> BetaPrior {
        reg.beta_prior(self.name()).clone()
    }

    // Override `posterior_in` to borrow the prior directly from the
    // registry rather than cloning it through `lookup_prior_in`.
    fn posterior_in(
        &self,
        reg: &super::PriorRegistry,
        query_vars: &BTreeSet<ContVarName>,
    ) -> (Option<BetaPrior>, RealEps) {
        self.posterior(reg.beta_prior(self.name()), query_vars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_same_variable_counts_counts() {
        let a = BetaSuffStat::counts(1, 3, 0);
        let b = BetaSuffStat::counts(1, 2, 1);
        let (merged, factor) = a.merge(&b);
        assert_eq!(factor, 0.0);
        match merged {
            BetaSuffStat::Counts {
                name,
                h,
                t,
                excluded,
            } => {
                assert_eq!(name, 1);
                assert_eq!(h, 5);
                assert_eq!(t, 1);
                assert!(excluded.is_empty());
            }
            _ => panic!("expected Counts"),
        }
    }

    #[test]
    fn merge_same_variable_counts_realeq() {
        // Counts {h=2, t=1} × RealEq(0.4) → RealEq(0.4) with factor 2*ln(0.4) + 1*ln(0.6).
        let a = BetaSuffStat::counts(1, 2, 1);
        let b = BetaSuffStat::real_eq(1, 0.4);
        let (merged, factor) = a.merge(&b);
        let expected = 2.0 * 0.4_f64.ln() + 0.6_f64.ln();
        assert!((factor - expected).abs() < 1e-12);
        match merged {
            BetaSuffStat::RealEq { name, value } => {
                assert_eq!(name, 1);
                assert!((value.into_inner() - 0.4).abs() < 1e-12);
            }
            _ => panic!("expected RealEq"),
        }
    }

    #[test]
    fn merge_same_variable_contradiction() {
        let a = BetaSuffStat::real_eq(1, 0.4);
        let b = BetaSuffStat::real_eq(1, 0.5);
        let (_, factor) = a.merge(&b);
        assert_eq!(factor, f64::NEG_INFINITY);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "overlapping scopes")]
    fn merge_disjoint_panics_in_debug() {
        let a = BetaSuffStat::counts(1, 1, 0);
        let b = BetaSuffStat::counts(2, 1, 0);
        let _ = a.merge(&b);
    }

    #[test]
    fn posterior_updates_matching_variable() {
        let stat = BetaSuffStat::counts(1, 3, 2);
        let prior = BetaPrior::Beta {
            name: 1,
            a: 1.0,
            b: 1.0,
        };
        let q: BTreeSet<ContVarName> = [1u64].iter().copied().collect();
        let post = stat
            .posterior(&prior, &q)
            .0
            .expect("posterior should be Some");
        match post {
            BetaPrior::Beta { name, a, b } => {
                assert_eq!(name, 1);
                assert!((a - 4.0).abs() < 1e-12);
                assert!((b - 3.0).abs() < 1e-12);
            }
            _ => panic!("expected Beta posterior"),
        }
    }

    #[test]
    fn posterior_empty_query_returns_none() {
        let stat = BetaSuffStat::counts(1, 3, 2);
        let prior = BetaPrior::Beta {
            name: 1,
            a: 1.0,
            b: 1.0,
        };
        let q: BTreeSet<ContVarName> = BTreeSet::new();
        assert!(stat.posterior(&prior, &q).0.is_none());
    }

    #[test]
    fn posterior_disjoint_query_returns_none() {
        let stat = BetaSuffStat::counts(1, 3, 2);
        let prior = BetaPrior::Beta {
            name: 1,
            a: 1.0,
            b: 1.0,
        };
        let q: BTreeSet<ContVarName> = [99u64].iter().copied().collect();
        assert!(stat.posterior(&prior, &q).0.is_none());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn posterior_debug_asserts_name_mismatch() {
        let stat = BetaSuffStat::counts(1, 1, 0);
        let prior = BetaPrior::Beta {
            name: 2,
            a: 1.0,
            b: 1.0,
        };
        let q: BTreeSet<ContVarName> = [1u64].iter().copied().collect();
        let _ = stat.posterior(&prior, &q);
    }

    #[test]
    fn log_likelihood_counts_matches_formula() {
        // Counts{h=2, t=1} at x=0.4 → 2 ln 0.4 + ln 0.6.
        let stat = BetaSuffStat::counts(1, 2, 1);
        let ll = stat.log_likelihood(&NotNan::new(0.4).unwrap());
        let expected = 2.0 * 0.4_f64.ln() + 0.6_f64.ln();
        assert!((ll - expected).abs() < 1e-12);
    }

    #[test]
    fn log_likelihood_counts_excluded_value_is_neg_infinity() {
        let mut excluded = BTreeSet::new();
        excluded.insert(NotNan::new(0.5).unwrap());
        let stat = BetaSuffStat::Counts {
            name: 1,
            h: 1,
            t: 1,
            excluded,
        };
        let ll = stat.log_likelihood(&NotNan::new(0.5).unwrap());
        assert_eq!(ll, f64::NEG_INFINITY);
    }

    #[test]
    fn log_likelihood_realeq_matches_value() {
        let stat = BetaSuffStat::real_eq(1, 0.4);
        let ll_eq = stat.log_likelihood(&NotNan::new(0.4).unwrap());
        assert_eq!(ll_eq, 0.0);
        let ll_neq = stat.log_likelihood(&NotNan::new(0.5).unwrap());
        assert_eq!(ll_neq, f64::NEG_INFINITY);
    }

    #[test]
    fn realeq_against_beta_prior_is_pinned() {
        let stat = BetaSuffStat::real_eq(1, 0.7);
        let prior = BetaPrior::Beta {
            name: 1,
            a: 2.0,
            b: 3.0,
        };
        let q: BTreeSet<ContVarName> = [1u64].iter().copied().collect();
        let post = stat
            .posterior(&prior, &q)
            .0
            .expect("posterior should be Some");
        match post {
            BetaPrior::Pinned { name, value } => {
                assert_eq!(name, 1);
                assert!((value - 0.7).abs() < 1e-12);
            }
            _ => panic!("expected Pinned posterior"),
        }
    }

    // -------------------- BetaPrior::sample --------------------

    use rand::SeedableRng;

    fn rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn sample_beta_in_unit_interval() {
        let p = BetaPrior::Beta {
            name: 1,
            a: 2.0,
            b: 5.0,
        };
        let mut r = rng();
        for _ in 0..200 {
            let v = p.sample(&mut r).into_inner();
            assert!((0.0..=1.0).contains(&v), "Beta sample out of range: {}", v);
        }
    }

    #[test]
    fn sample_beta_empirical_mean_matches_prior() {
        // Beta(2, 5) has mean 2/7 ≈ 0.2857.
        let p = BetaPrior::Beta {
            name: 1,
            a: 2.0,
            b: 5.0,
        };
        let mut r = rng();
        let n = 5000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += p.sample(&mut r).into_inner();
        }
        let mean = sum / n as f64;
        let expected = 2.0 / 7.0;
        assert!(
            (mean - expected).abs() < 0.02,
            "empirical mean {:.4} differs from {:.4}",
            mean,
            expected
        );
    }

    #[test]
    fn sample_pinned_returns_value() {
        let p = BetaPrior::Pinned {
            name: 4,
            value: 0.42,
        };
        let mut r = rng();
        for _ in 0..10 {
            assert_eq!(p.sample(&mut r).into_inner(), 0.42);
        }
    }

    #[test]
    fn test_beta_exact_equality() {
        let b1 = BetaPrior::Beta {
            name: "p",
            a: 2.0,
            b: 1.0,
        };
        let b2 = b1.clone();
        assert_eq!(b1, b2);
    }

    #[test]
    fn test_beta_approx_equality() {
        let b1 = BetaPrior::Beta {
            name: "p",
            a: 2.0,
            b: 1.0,
        };
        let b2 = BetaPrior::Beta {
            name: "p",
            a: 2.0 + 1e-10,
            b: 1.0 + 1e-10,
        };
        assert_eq!(b1, b2);
    }

    #[test]
    fn test_beta_disequality() {
        let b1 = BetaPrior::Beta {
            name: "p",
            a: 2.0,
            b: 1.0,
        };
        let b2 = BetaPrior::Beta {
            name: "q",
            a: 2.0,
            b: 1.0,
        };
        let b3 = BetaPrior::Beta {
            name: "p",
            a: 3.0,
            b: 1.0,
        };
        let b4 = BetaPrior::Beta {
            name: "p",
            a: 2.0,
            b: 4.0,
        };
        assert_ne!(b1, b2);
        assert_ne!(b1, b3);
        assert_ne!(b1, b4);
    }
}
