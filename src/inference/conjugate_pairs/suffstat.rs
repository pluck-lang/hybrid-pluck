use std::collections::BTreeSet;
use std::fmt::Display;
use std::hash::Hash;

use crate::inference::conjugate_pairs::{Assignment, AssignmentBuilder, PriorRegistry};
use crate::utils::epsilon::RealEps;

pub type ContVarName = u64;

// Shared quantization level for float hashing
/// Hash + Eq dedupe on quantized floats (1e-6) so distinct BDD/SPN paths
/// whose posteriors agree up to float noise bucket together
pub const PRIOR_QUANTIZATION_LEVEL: f64 = 1e6;

pub trait Prior: Display + Hash {
    type Realization;

    fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_;

    fn sample<R: rand::Rng>(&self, rng: &mut R) -> Self::Realization;

    /// Append a sampled realisation into the family-specific vec of an
    /// `AssignmentBuilder`. Beta/Dirichlet push one entry; Gaussian
    /// splays one entry per variable across its `var_order`.
    fn push_into(&self, sampled: Self::Realization, out: &mut AssignmentBuilder);
}

pub trait SuffStat: PartialEq + Clone + Hash {
    type ConjugatePrior: Prior;

    // The random variables referenced by this SuffStat
    fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_;

    /// Merge two sufficient statistics over OVERLAPPING scopes.
    ///
    /// Returns the merged stat and a log-correction factor: the
    /// log-likelihood mass discharged by the merge (`0.0` when the stats are
    /// simply combined, `-inf` if they contradict, e.g. two distinct pinned
    /// values). Disjoint scopes are not a valid input — the caller should
    /// build an `Spn::product` of the two leaves instead. Impls debug-assert
    /// this.
    fn merge(&self, other: &Self) -> (Self, f64);

    // Likelihood of SuffStat observations given values of random variables
    fn log_likelihood(&self, value: &<Self::ConjugatePrior as Prior>::Realization) -> f64;

    /// Conjugate update of the (single-leaf) prior, projected to the
    /// variables in `query_vars`. The prior and the suff stat must agree
    /// on scope; impls debug-assert.
    ///
    /// Returns:
    /// - `(Some(p), Z)`: the conjugate posterior projected to
    ///   `scope ∩ query_vars` and the marginal likelihood
    /// - (`None`, Z): the leaf's scope is disjoint from `query_vars`; nothing
    ///   to report. We still get a marginal likelihood.
    fn posterior(
        &self,
        prior: &Self::ConjugatePrior,
        query_vars: &BTreeSet<ContVarName>,
    ) -> (Option<Self::ConjugatePrior>, RealEps);

    /// Look up this leaf's realisation in an `Assignment`. Returns
    /// owned data so the trait signature stays uniform; families with
    /// naturally-borrowable realisations (e.g. Dirichlet's
    /// `Box<[NotNan<f64>]>`) can override `log_likelihood_in` to skip
    /// the implied clone.
    fn read_realization(&self, a: &Assignment) -> <Self::ConjugatePrior as Prior>::Realization;

    /// Look up this leaf's conjugate prior in a `PriorRegistry`.
    /// Returns owned; families that natively store priors in the
    /// registry (Beta, Dirichlet) can override `posterior_in` to skip
    /// the implied clone.
    fn lookup_prior_in(&self, reg: &PriorRegistry) -> Self::ConjugatePrior;

    /// Combined assignment-lookup + log-likelihood. The default impl
    /// composes `read_realization` and `log_likelihood`; families
    /// override to skip a clone when their natural realisation form
    /// is borrowable.
    fn log_likelihood_in(&self, a: &Assignment) -> f64 {
        self.log_likelihood(&self.read_realization(a))
    }

    /// Combined prior-lookup + posterior. The default impl composes
    /// `lookup_prior_in` and `posterior`; families override to skip a
    /// clone when the prior can be borrowed directly from the
    /// registry.
    fn posterior_in(
        &self,
        reg: &PriorRegistry,
        query_vars: &BTreeSet<ContVarName>,
    ) -> (Option<Self::ConjugatePrior>, RealEps) {
        self.posterior(&self.lookup_prior_in(reg), query_vars)
    }
}
