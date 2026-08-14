use ordered_float::NotNan;

use super::suffstat::{ContVarName, Prior, SuffStat, PRIOR_QUANTIZATION_LEVEL};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter, Result};
use std::hash::{Hash, Hasher};

use crate::utils::epsilon::RealEps;
use crate::utils::math::{log_dirichlet_pdf, log_dirichlet_ratio, quant};

/// Dirichlet prior over a single categorical variable.
///
/// Two shapes (mirrors `BetaPrior`):
/// - `Dirichlet(alphas)`: a standard Dirichlet prior; `alphas[i]` is the
///   prior pseudo-count for category `i`.
/// - `Pinned(value)`: a Dirac prior at `value`. Produced when a Dirichlet
///   vector is observed equal to a specific probability vector (the
///   `RealEq` sufficient stat, used by `DirichletFlip::VecPin`).
///
/// The `N` parameter is the variable name type. Internal inference
/// carries `N = ContVarName` (the default); display / serialization
/// can instantiate with `N = String`. The `Prior` impl exists only for
/// `N = ContVarName`.
/// `#[serde(untagged)]` keeps the JSON shape flat — the `Dirichlet` variant
/// serializes as `{"name": …, "alphas": [...]}` and `Pinned` as
/// `{"name": …, "value": [...]}`, with no wrapper tag naming the variant.
/// The two variants are unambiguous because their field sets differ.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum DirichletPrior<N = ContVarName> {
    Dirichlet {
        name: N,
        alphas: Vec<f64>,
    },
    /// Dirac prior at a fixed probability vector. Stored as raw `f64` for
    /// serde compatibility (the sample boundary re-wraps in `NotNan`,
    /// which is safe because every entry came from a finite probability
    /// draw or `RealEq` observation).
    Pinned {
        name: N,
        value: Vec<f64>,
    },
}

impl<N: Clone> DirichletPrior<N> {
    pub fn name(&self) -> N {
        match self {
            DirichletPrior::Dirichlet { name, .. } => name.clone(),
            DirichletPrior::Pinned { name, .. } => name.clone(),
        }
    }
}

impl DirichletPrior<ContVarName> {
    /// Re-tag this prior with a display name. Used by the lib layer to
    /// produce client-facing `DirichletPrior<String>` values.
    pub fn with_display_name(self, name: String) -> DirichletPrior<String> {
        match self {
            DirichletPrior::Dirichlet { alphas, .. } => DirichletPrior::Dirichlet { name, alphas },
            DirichletPrior::Pinned { value, .. } => DirichletPrior::Pinned { name, value },
        }
    }
}

impl<N: Hash> Hash for DirichletPrior<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            DirichletPrior::Dirichlet { name, alphas } => {
                "Dirichlet".hash(state);
                name.hash(state);
                for a in alphas {
                    quant(*a, PRIOR_QUANTIZATION_LEVEL).hash(state);
                }
            }
            DirichletPrior::Pinned { name, value } => {
                "Pinned".hash(state);
                name.hash(state);
                for v in value {
                    quant(*v, PRIOR_QUANTIZATION_LEVEL).hash(state);
                }
            }
        }
    }
}

impl<N: PartialEq> PartialEq for DirichletPrior<N> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                DirichletPrior::Dirichlet {
                    name: n1,
                    alphas: a1,
                },
                DirichletPrior::Dirichlet {
                    name: n2,
                    alphas: a2,
                },
            ) => {
                if n1 != n2 || a1.len() != a2.len() {
                    return false;
                }
                for (a, b) in a1.iter().zip(a2.iter()) {
                    if quant(*a, PRIOR_QUANTIZATION_LEVEL) != quant(*b, PRIOR_QUANTIZATION_LEVEL) {
                        return false;
                    }
                }
                true
            }
            (
                DirichletPrior::Pinned {
                    name: n1,
                    value: v1,
                },
                DirichletPrior::Pinned {
                    name: n2,
                    value: v2,
                },
            ) => {
                if n1 != n2 || v1.len() != v2.len() {
                    return false;
                }
                for (a, b) in v1.iter().zip(v2.iter()) {
                    if quant(*a, PRIOR_QUANTIZATION_LEVEL) != quant(*b, PRIOR_QUANTIZATION_LEVEL) {
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }
}

impl<N: Eq> Eq for DirichletPrior<N> {}

impl<N: Display> Display for DirichletPrior<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            DirichletPrior::Dirichlet { name, alphas } => {
                let p = crate::utils::display_precision::param_precision();
                let alphas = alphas
                    .iter()
                    .map(|x| format!("{:.p$}", x, p = p))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "{} ~ Dirichlet({})", name, alphas)
            }
            DirichletPrior::Pinned { name, value } => {
                let p = crate::utils::display_precision::param_precision();
                let value = value
                    .iter()
                    .map(|x| format!("{:.p$}", x, p = p))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "{} = ({})", name, value)
            }
        }
    }
}

impl Prior for DirichletPrior<ContVarName> {
    type Realization = Box<[NotNan<f64>]>;

    fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_ {
        std::iter::once(match self {
            DirichletPrior::Dirichlet { name, .. } => name,
            DirichletPrior::Pinned { name, .. } => name,
        })
    }

    /// Draw a realisation:
    /// - `Dirichlet(alphas)`: standard Dirichlet draw via per-component
    ///   Gamma draws plus normalisation.
    /// - `Pinned(value)`: returns the stored vector unchanged.
    fn sample<R: rand::Rng>(&self, rng: &mut R) -> Box<[NotNan<f64>]> {
        match self {
            DirichletPrior::Dirichlet { alphas, .. } => {
                crate::utils::sampling::sample_dirichlet(rng, alphas)
                    .into_iter()
                    .map(|x| NotNan::new(x).expect("Dirichlet sample component is NaN"))
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            }
            DirichletPrior::Pinned { value, .. } => value
                .iter()
                .map(|v| NotNan::new(*v).expect("Pinned Dirichlet value contains NaN"))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    fn push_into(&self, sampled: Box<[NotNan<f64>]>, out: &mut super::AssignmentBuilder) {
        out.push_dirichlet(self.name(), sampled);
    }
}

impl From<DirichletPrior> for super::PosteriorLeaf {
    fn from(p: DirichletPrior) -> Self {
        super::PosteriorLeaf::Dirichlet(p)
    }
}

// ---------------------------------------------------------------------------
// DirichletFlip: per-family BDD-flip variants.
// ---------------------------------------------------------------------------

use crate::inference::conjugate_pairs::EvidenceLeaf;
use crate::inference::spn::node::Spn;

/// Stick-breaking Beta counts produced by observing categorical outcome `index`
/// in a Dirichlet with `categories` total categories.
///
/// For `index < categories - 1`: all sticks before `index` missed (+1 tails),
/// and stick `index` hit (+1 heads).
/// For `index == categories - 1`: all N-1 sticks missed (the "leftover" category).
pub fn stick_counts_for_category(index: usize, categories: usize) -> BTreeMap<usize, (u64, u64)> {
    let mut map = BTreeMap::new();
    let n_sticks = categories.saturating_sub(1);
    for j in 0..n_sticks.min(index) {
        map.insert(j, (0, 1));
    }
    if index < n_sticks {
        map.insert(index, (1, 0));
    }
    map
}

/// Dirichlet-family BDD-flip variants.
pub enum DirichletFlip {
    /// Flip of a symbolic Dirichlet probability — weight tracks sufficient
    /// statistics for the indexed category.
    VarElement {
        name: ContVarName,
        index: usize,
        categories: usize,
    },
    /// A single stick of a Dirichlet represented in stick-breaking form.
    /// Used by `categorical` to build the per-category Bernoullis.
    /// High edge: this stick was hit — +1 on its Beta's hit count.
    /// Low edge: this stick was missed — +1 on its Beta's miss count.
    StickElement { name: ContVarName, index: usize },
    /// Dirichlet vector pinning for disintegration: observe the whole
    /// probability vector equal to `value`. Treated as a pinned
    /// measure-zero observation (parallel to `BetaFlip::ProbPin`).
    VecPin {
        name: ContVarName,
        value: Box<[NotNan<f64>]>,
    },
}

impl DirichletFlip {
    pub fn weights(&self) -> (Spn<EvidenceLeaf>, Spn<EvidenceLeaf>) {
        match self {
            DirichletFlip::VarElement {
                name,
                index,
                categories,
            } => {
                // `(flip (vector_index @i p))` semantics: marginalize over which
                // categorical outcome occurred. High edge = "observed category `index`";
                // low edge = Sum over j ≠ index of "observed category j".
                let high_weight = Spn::leaf(EvidenceLeaf::Dirichlet(DirichletSuffStat::counts(
                    *name,
                    stick_counts_for_category(*index, *categories),
                )));
                let mut low_weight_components = Vec::new();
                for j in 0..*categories {
                    if j != *index {
                        low_weight_components.push(Spn::leaf(EvidenceLeaf::Dirichlet(
                            DirichletSuffStat::counts(
                                *name,
                                stick_counts_for_category(j, *categories),
                            ),
                        )));
                    }
                }
                (Spn::sum(low_weight_components), high_weight)
            }
            DirichletFlip::StickElement { name, index } => (
                Spn::leaf(EvidenceLeaf::Dirichlet(DirichletSuffStat::counts(
                    *name,
                    BTreeMap::from([(*index, (0u64, 1u64))]),
                ))),
                Spn::leaf(EvidenceLeaf::Dirichlet(DirichletSuffStat::counts(
                    *name,
                    BTreeMap::from([(*index, (1u64, 0u64))]),
                ))),
            ),
            DirichletFlip::VecPin { name, value } => {
                // High edge = "observed vector equals `value`" → Dirac
                //   (`DirichletSuffStat::RealEq`).
                // Low edge = "vector is anything else" → measure-1 event
                //   under any continuous Dirichlet prior, but we record
                //   `value` in the exclusion set so a later `RealEq(value)`
                //   merge detects the contradiction (parallels
                //   `BetaFlip::ProbPin`'s use of `BetaSuffStat::not_eq`).
                let low = Spn::leaf(EvidenceLeaf::Dirichlet(DirichletSuffStat::not_eq(
                    *name,
                    value.clone(),
                )));
                let high = Spn::leaf(EvidenceLeaf::Dirichlet(DirichletSuffStat::real_eq(
                    *name,
                    value.clone(),
                )));
                (low, high)
            }
        }
    }

    pub fn discriminator(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        match self {
            DirichletFlip::VarElement { name, index, .. } => {
                let mut hasher = DefaultHasher::new();
                name.hash(&mut hasher);
                index.hash(&mut hasher);
                hasher.finish().wrapping_add(0xD1E2F30415263748)
            }
            DirichletFlip::StickElement { name, index } => {
                let mut hasher = DefaultHasher::new();
                name.hash(&mut hasher);
                index.hash(&mut hasher);
                hasher.finish().wrapping_add(0x5A6B7C8D9EAFBE29)
            }
            DirichletFlip::VecPin { name, value } => {
                let mut hasher = DefaultHasher::new();
                name.hash(&mut hasher);
                for v in value.iter() {
                    v.into_inner().to_bits().hash(&mut hasher);
                }
                hasher.finish().wrapping_add(0xE3A4B5C6D7E8F901)
            }
        }
    }

    /// `VecPin` is a measure-zero observation looked up by callstack hash
    /// alone (parallel to `BetaFlip::ProbPin`). Other variants are regular
    /// flips with non-trivial low/high weights.
    pub fn is_pinned_observation(&self) -> bool {
        matches!(self, DirichletFlip::VecPin { .. })
    }

    pub fn sample_probability(&self, a: &super::Assignment) -> Option<f64> {
        match self {
            DirichletFlip::VarElement { name, index, .. } => {
                Some(a.dirichlet_value(*name)[*index].into_inner())
            }
            // StickElement isn't reached via `sample_compile_flip` today
            // (categorical uses a different sample-mode path).
            DirichletFlip::StickElement { .. } => None,
            // VecPin is an observation, not a flip target.
            DirichletFlip::VecPin { .. } => None,
        }
    }
}

/// Sufficient statistics for a single Dirichlet variable.
///
/// Two shapes (mirrors `BetaSuffStat`):
/// - `Counts { counts, excluded }`: stick-breaking counts. The Dirichlet is
///   factored into N-1 independent sticks, each accumulating `(hits, misses)`
///   as Dirichlet-conjugate counts; the leftover category's alpha-update comes
///   from the last stick's miss count. `excluded` records vector values that
///   the BDD path is asserting the Dirichlet is NOT equal to (the low edge of
///   `DirichletFlip::VecPin`). Exclusions have Lebesgue measure 0 under a
///   continuous Dirichlet so they don't change the density, but they're kept
///   so a later `RealEq(v)` with `v ∈ excluded` is detected as a contradiction.
/// - `RealEq { value }`: the vector was pinned (Dirac observation) — produced
///   by `DirichletFlip::VecPin` when a Dirichlet vector is observed equal to
///   a specific probability vector.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DirichletSuffStat {
    Counts {
        name: ContVarName,
        counts: BTreeMap<usize, (u64, u64)>,
        excluded: BTreeSet<Box<[NotNan<f64>]>>,
    },
    RealEq {
        name: ContVarName,
        value: Box<[NotNan<f64>]>,
    },
}

impl DirichletSuffStat {
    /// Counts-style constructor (no exclusions).
    pub fn counts(name: ContVarName, counts: BTreeMap<usize, (u64, u64)>) -> Self {
        DirichletSuffStat::Counts {
            name,
            counts,
            excluded: BTreeSet::new(),
        }
    }

    /// "Not-equal-to" constructor: empty counts plus a single excluded vector.
    /// Used as the low edge of `DirichletFlip::VecPin` so contradiction with
    /// a later `RealEq(value)` is detected by `merge`.
    pub fn not_eq(name: ContVarName, value: Box<[NotNan<f64>]>) -> Self {
        let mut excluded = BTreeSet::new();
        excluded.insert(value);
        DirichletSuffStat::Counts {
            name,
            counts: BTreeMap::new(),
            excluded,
        }
    }

    /// Real-equality (Dirac) constructor.
    pub fn real_eq(name: ContVarName, value: Box<[NotNan<f64>]>) -> Self {
        DirichletSuffStat::RealEq { name, value }
    }

    pub fn name(&self) -> ContVarName {
        match self {
            DirichletSuffStat::Counts { name, .. } => *name,
            DirichletSuffStat::RealEq { name, .. } => *name,
        }
    }
}

/// Stick-breaking log-likelihood of `counts` evaluated at `value`.
///
/// Extracted as a helper because `merge` (in the new `Counts × RealEq` case)
/// and the `SuffStat::log_likelihood` impl both need this computation.
fn counts_log_likelihood_at(counts: &BTreeMap<usize, (u64, u64)>, value: &[NotNan<f64>]) -> f64 {
    let n = value.len();
    // Suffix sums r_i = Σ_{j ≥ i} θ_j.
    let mut suffix = vec![0.0_f64; n + 1];
    for i in (0..n).rev() {
        suffix[i] = suffix[i + 1] + value[i].into_inner();
    }
    debug_assert!(
        (suffix[0] - 1.0).abs() < 1e-9,
        "counts_log_likelihood_at: probability vector must sum to 1; got {}",
        suffix[0]
    );
    let mut total = 0.0_f64;
    for (&i, &(h, t)) in counts {
        if h == 0 && t == 0 {
            continue;
        }
        let theta_i = value[i].into_inner();
        let r_i = suffix[i];
        if r_i <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if h > 0 {
            if theta_i <= 0.0 {
                return f64::NEG_INFINITY;
            }
            total += (h as f64) * (theta_i / r_i).ln();
        }
        if t > 0 {
            let rest = r_i - theta_i;
            if rest <= 0.0 {
                return f64::NEG_INFINITY;
            }
            total += (t as f64) * (rest / r_i).ln();
        }
    }
    total
}

fn vectors_match(a: &[NotNan<f64>], b: &[NotNan<f64>]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

impl SuffStat for DirichletSuffStat {
    type ConjugatePrior = DirichletPrior;

    fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_ {
        std::iter::once(match self {
            DirichletSuffStat::Counts { name, .. } => name,
            DirichletSuffStat::RealEq { name, .. } => name,
        })
    }

    fn merge(&self, other: &Self) -> (Self, f64) {
        debug_assert_eq!(
            self.name(),
            other.name(),
            "DirichletSuffStat::merge requires overlapping scopes"
        );
        match (self, other) {
            (
                DirichletSuffStat::Counts {
                    name,
                    counts: c1,
                    excluded: e1,
                },
                DirichletSuffStat::Counts {
                    counts: c2,
                    excluded: e2,
                    ..
                },
            ) => {
                let mut counts = c1.clone();
                for (index, (h2, t2)) in c2 {
                    let entry = counts.entry(*index).or_insert((0, 0));
                    entry.0 += *h2;
                    entry.1 += *t2;
                }
                let mut excluded = e1.clone();
                excluded.extend(e2.iter().cloned());
                (
                    DirichletSuffStat::Counts {
                        name: *name,
                        counts,
                        excluded,
                    },
                    0.0,
                )
            }
            (
                DirichletSuffStat::Counts {
                    counts, excluded, ..
                },
                DirichletSuffStat::RealEq { name, value },
            )
            | (
                DirichletSuffStat::RealEq { name, value },
                DirichletSuffStat::Counts {
                    counts, excluded, ..
                },
            ) => {
                if excluded.contains(value) {
                    return (
                        DirichletSuffStat::RealEq {
                            name: *name,
                            value: value.clone(),
                        },
                        f64::NEG_INFINITY,
                    );
                }
                let factor = counts_log_likelihood_at(counts, value);
                (
                    DirichletSuffStat::RealEq {
                        name: *name,
                        value: value.clone(),
                    },
                    factor,
                )
            }
            (
                DirichletSuffStat::RealEq { name, value: v1 },
                DirichletSuffStat::RealEq { value: v2, .. },
            ) => {
                if vectors_match(v1, v2) {
                    (
                        DirichletSuffStat::RealEq {
                            name: *name,
                            value: v1.clone(),
                        },
                        0.0,
                    )
                } else {
                    (
                        DirichletSuffStat::RealEq {
                            name: *name,
                            value: v1.clone(),
                        },
                        f64::NEG_INFINITY,
                    )
                }
            }
        }
    }

    fn log_likelihood(&self, value: &Box<[NotNan<f64>]>) -> f64 {
        match self {
            DirichletSuffStat::Counts { counts, .. } => counts_log_likelihood_at(counts, value),
            DirichletSuffStat::RealEq { value: pinned, .. } => {
                if vectors_match(pinned, value) {
                    0.0
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
    }

    fn posterior(
        &self,
        prior: &DirichletPrior,
        query_vars: &BTreeSet<ContVarName>,
    ) -> (Option<DirichletPrior>, RealEps) {
        debug_assert_eq!(self.name(), prior.name());
        let name = self.name();
        let marginal_likelihood = match (self, prior) {
            (
                DirichletSuffStat::Counts { counts, .. },
                DirichletPrior::Dirichlet { alphas, .. },
            ) => RealEps::from_log(log_dirichlet_ratio(alphas, counts), 0),
            (DirichletSuffStat::RealEq { value, .. }, DirichletPrior::Dirichlet { alphas, .. }) => {
                // Measure-zero observation: RealEps order 1 marks the
                // Dirac contribution (parallels `BetaSuffStat::RealEq`
                // against `BetaPrior::Beta`).
                RealEps::from_log(log_dirichlet_pdf(value, alphas), 1)
            }
            (
                DirichletSuffStat::Counts {
                    counts, excluded, ..
                },
                DirichletPrior::Pinned { value, .. },
            ) => {
                // Convert raw f64 → NotNan once for the likelihood eval.
                let nn: Box<[NotNan<f64>]> = value
                    .iter()
                    .map(|v| NotNan::new(*v).expect("Pinned Dirichlet value contains NaN"))
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                if excluded.contains(&nn) {
                    RealEps::zero()
                } else {
                    RealEps::from_log(counts_log_likelihood_at(counts, &nn), 0)
                }
            }
            (
                DirichletSuffStat::RealEq { value: stat_v, .. },
                DirichletPrior::Pinned { value: prior_v, .. },
            ) => {
                let matches = stat_v.len() == prior_v.len()
                    && stat_v
                        .iter()
                        .zip(prior_v.iter())
                        .all(|(s, p)| s.into_inner() == *p);
                if matches {
                    RealEps::scalar(1.0)
                } else {
                    RealEps::zero()
                }
            }
        };
        if marginal_likelihood.is_zero() {
            return (None, marginal_likelihood);
        }
        if !query_vars.contains(&self.name()) {
            return (None, marginal_likelihood);
        }
        let posterior_prior = match (self, prior) {
            (
                DirichletSuffStat::Counts { counts, .. },
                DirichletPrior::Dirichlet { alphas, .. },
            ) => {
                // Stick-breaking → Dirichlet alpha conversion:
                //   α_i' = α_i + h_i     for stick indices i ∈ 0..N-2
                //   α_{N-1}' = α_{N-1} + t_{N-2}   (last stick's miss feeds the leftover)
                let mut alphas = alphas.clone();
                let n = alphas.len();
                let last_stick = n.saturating_sub(2);
                for (&i, &(h, t)) in counts {
                    alphas[i] += h as f64;
                    if i == last_stick && n >= 2 {
                        alphas[n - 1] += t as f64;
                    }
                }
                DirichletPrior::Dirichlet { name, alphas }
            }
            (DirichletSuffStat::RealEq { value, .. }, _) => DirichletPrior::Pinned {
                name,
                value: value.iter().map(|v| v.into_inner()).collect(),
            },
            (DirichletSuffStat::Counts { .. }, DirichletPrior::Pinned { value, .. }) => {
                // Counts against a Pinned prior leave the prior unchanged
                // (the density factor is already in marginal_likelihood).
                DirichletPrior::Pinned {
                    name,
                    value: value.clone(),
                }
            }
        };
        (Some(posterior_prior), marginal_likelihood)
    }

    fn read_realization(&self, a: &super::Assignment) -> Box<[NotNan<f64>]> {
        a.dirichlet_value(self.name()).clone()
    }

    fn lookup_prior_in(&self, reg: &super::PriorRegistry) -> DirichletPrior {
        reg.dirichlet_prior(self.name()).clone()
    }

    // Override `log_likelihood_in` to borrow the realisation directly
    // from the assignment, avoiding the `Box<[NotNan<f64>]>` clone
    // that the default impl would do via `read_realization`.
    fn log_likelihood_in(&self, a: &super::Assignment) -> f64 {
        self.log_likelihood(a.dirichlet_value(self.name()))
    }

    // Override `posterior_in` to borrow the prior directly, skipping
    // the clone in `lookup_prior_in`.
    fn posterior_in(
        &self,
        reg: &super::PriorRegistry,
        query_vars: &BTreeSet<ContVarName>,
    ) -> (Option<DirichletPrior>, RealEps) {
        self.posterior(reg.dirichlet_prior(self.name()), query_vars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stat(name: ContVarName, entries: &[(usize, u64, u64)]) -> DirichletSuffStat {
        let mut counts = BTreeMap::new();
        for &(i, h, t) in entries {
            counts.insert(i, (h, t));
        }
        DirichletSuffStat::counts(name, counts)
    }

    #[test]
    fn merge_same_variable_adds_per_stick() {
        let a = stat(1, &[(0, 2, 1), (1, 0, 3)]);
        let b = stat(1, &[(0, 1, 0), (2, 5, 0)]);
        let (merged, factor) = a.merge(&b);
        assert_eq!(factor, 0.0);
        match merged {
            DirichletSuffStat::Counts {
                name,
                counts,
                excluded,
            } => {
                assert_eq!(name, 1);
                assert_eq!(counts.get(&0), Some(&(3, 1)));
                assert_eq!(counts.get(&1), Some(&(0, 3)));
                assert_eq!(counts.get(&2), Some(&(5, 0)));
                assert!(excluded.is_empty());
            }
            _ => panic!("expected Counts"),
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "overlapping scopes")]
    fn merge_disjoint_panics_in_debug() {
        let a = stat(1, &[(0, 1, 0)]);
        let b = stat(2, &[(0, 1, 0)]);
        let _ = a.merge(&b);
    }

    #[test]
    fn posterior_updates_alphas_with_last_stick_leftover() {
        // 3-category Dirichlet ⇒ 2 sticks (indices 0, 1). Last stick is index 1.
        // A hit on stick 0 → α_0 += h. A miss on stick 1 → α_2 += t (leftover).
        let prior = DirichletPrior::Dirichlet {
            name: 1,
            alphas: vec![1.0, 1.0, 1.0],
        };
        let s = stat(1, &[(0, 2, 0), (1, 1, 4)]);
        let q: BTreeSet<ContVarName> = [1u64].iter().copied().collect();
        let post = s.posterior(&prior, &q).0.expect("posterior should be Some");
        match post {
            DirichletPrior::Dirichlet { name, alphas } => {
                assert_eq!(name, 1);
                assert!((alphas[0] - 3.0).abs() < 1e-12); // 1 + 2 hits
                assert!((alphas[1] - 2.0).abs() < 1e-12); // 1 + 1 hit
                assert!((alphas[2] - 5.0).abs() < 1e-12); // 1 + 4 leftover
            }
            _ => panic!("expected Dirichlet posterior"),
        }
    }

    #[test]
    fn posterior_empty_query_returns_none() {
        let prior = DirichletPrior::Dirichlet {
            name: 1,
            alphas: vec![1.0, 1.0, 1.0],
        };
        let s = stat(1, &[(0, 2, 0)]);
        let q: BTreeSet<ContVarName> = BTreeSet::new();
        assert!(s.posterior(&prior, &q).0.is_none());
    }

    #[test]
    fn posterior_disjoint_query_returns_none() {
        let prior = DirichletPrior::Dirichlet {
            name: 1,
            alphas: vec![1.0, 1.0, 1.0],
        };
        let s = stat(1, &[(0, 2, 0)]);
        let q: BTreeSet<ContVarName> = [99u64].iter().copied().collect();
        assert!(s.posterior(&prior, &q).0.is_none());
    }

    fn theta(values: &[f64]) -> Box<[NotNan<f64>]> {
        values
            .iter()
            .map(|x| NotNan::new(*x).unwrap())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    #[test]
    fn log_likelihood_single_hit_first_stick() {
        // 2 categories, θ = [0.5, 0.5]. Counts: (stick 0, (h=1, t=0)).
        // r_0 = 1.0, stick value = θ_0 / r_0 = 0.5. ln(0.5).
        let s = stat(1, &[(0, 1, 0)]);
        let ll = s.log_likelihood(&theta(&[0.5, 0.5]));
        assert!((ll - 0.5_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn log_likelihood_mixed_hits_and_misses() {
        // 3 categories, θ = [0.2, 0.3, 0.5].
        // r_0 = 1.0, r_1 = 0.8.
        // Counts: stick 0 → (2, 1); stick 1 → (1, 2).
        // Contribution from stick 0: 2 ln(0.2/1.0) + 1 ln(0.8/1.0)
        //                                = 2 ln 0.2 + ln 0.8.
        // Contribution from stick 1: 1 ln(0.3/0.8) + 2 ln(0.5/0.8).
        let s = stat(1, &[(0, 2, 1), (1, 1, 2)]);
        let ll = s.log_likelihood(&theta(&[0.2, 0.3, 0.5]));
        let expected =
            2.0 * 0.2_f64.ln() + 0.8_f64.ln() + (0.3_f64 / 0.8).ln() + 2.0 * (0.5_f64 / 0.8).ln();
        assert!(
            (ll - expected).abs() < 1e-12,
            "ll={} expected={}",
            ll,
            expected
        );
    }

    #[test]
    fn log_likelihood_zero_mass_category_with_hit_is_neg_infinity() {
        // θ = [0.0, 1.0] but a hit on stick 0 → -inf.
        let s = stat(1, &[(0, 1, 0)]);
        let ll = s.log_likelihood(&theta(&[0.0, 1.0]));
        assert_eq!(ll, f64::NEG_INFINITY);
    }

    #[test]
    fn log_likelihood_against_dirichlet_ratio_reference() {
        // For a flat α=(1,1,1) Dirichlet, the marginal-likelihood ratio is
        // independent of θ; for a Dirichlet(α) draw with stick representation,
        // the log-likelihood at θ should depend only on θ. Cross-check by
        // independently re-deriving the formula on a fresh θ.
        let s = stat(1, &[(0, 3, 1), (1, 0, 2)]);
        let val = theta(&[0.4, 0.4, 0.2]);
        let ll = s.log_likelihood(&val);

        // Hand re-implementation.
        let r0: f64 = 0.4 + 0.4 + 0.2;
        let r1: f64 = 0.4 + 0.2;
        let manual =
            3.0 * (0.4_f64 / r0).ln() + ((r0 - 0.4) / r0).ln() + 2.0 * ((r1 - 0.4) / r1).ln();
        assert!((ll - manual).abs() < 1e-12);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn posterior_debug_asserts_name_mismatch() {
        let prior = DirichletPrior::Dirichlet {
            name: 1,
            alphas: vec![1.0, 1.0],
        };
        let s = stat(2, &[(0, 1, 0)]);
        let q: BTreeSet<ContVarName> = [2u64].iter().copied().collect();
        let _ = s.posterior(&prior, &q);
    }

    // -------------------- DirichletPrior::sample --------------------

    use rand::SeedableRng;

    #[test]
    fn sample_dirichlet_components_sum_to_one() {
        let p = DirichletPrior::Dirichlet {
            name: 9,
            alphas: vec![2.0, 3.0, 5.0],
        };
        let mut r = rand::rngs::StdRng::seed_from_u64(42);
        let v = <DirichletPrior as Prior>::sample(&p, &mut r);
        assert_eq!(v.len(), 3);
        let total: f64 = v.iter().map(|x| x.into_inner()).sum();
        assert!((total - 1.0).abs() < 1e-12, "dirichlet sum = {}", total);
    }

    // -------------------- RealEq pinning (Phase A) --------------------

    fn vec_nn(xs: &[f64]) -> Box<[NotNan<f64>]> {
        xs.iter()
            .map(|x| NotNan::new(*x).unwrap())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    #[test]
    fn merge_same_variable_counts_realeq_match_factor() {
        // Counts={stick 0: (2, 1)} × RealEq(θ=[0.2, 0.3, 0.5]).
        // counts_log_likelihood_at: r_0=1.0, stick_0_val=θ_0/r_0=0.2.
        //   contribution = 2 ln(0.2) + 1 ln(0.8/1.0) = 2 ln(0.2) + ln(0.8).
        let c = stat(1, &[(0, 2, 1)]);
        let r = DirichletSuffStat::real_eq(1, vec_nn(&[0.2, 0.3, 0.5]));
        let (merged, factor) = c.merge(&r);
        let expected = 2.0 * 0.2_f64.ln() + 0.8_f64.ln();
        assert!(
            (factor - expected).abs() < 1e-12,
            "got {} expected {}",
            factor,
            expected
        );
        match merged {
            DirichletSuffStat::RealEq { name, value } => {
                assert_eq!(name, 1);
                assert_eq!(value.len(), 3);
                assert!((value[0].into_inner() - 0.2).abs() < 1e-12);
            }
            _ => panic!("expected RealEq"),
        }
    }

    #[test]
    fn merge_realeq_counts_symmetric() {
        // RealEq × Counts: same factor, regardless of order.
        let c = stat(1, &[(0, 2, 1)]);
        let r = DirichletSuffStat::real_eq(1, vec_nn(&[0.2, 0.3, 0.5]));
        let (_, f1) = c.merge(&r);
        let (_, f2) = r.merge(&c);
        assert!((f1 - f2).abs() < 1e-12);
    }

    #[test]
    fn merge_realeq_realeq_match() {
        let a = DirichletSuffStat::real_eq(1, vec_nn(&[0.4, 0.6]));
        let b = DirichletSuffStat::real_eq(1, vec_nn(&[0.4, 0.6]));
        let (_, factor) = a.merge(&b);
        assert_eq!(factor, 0.0);
    }

    #[test]
    fn merge_realeq_realeq_mismatch() {
        let a = DirichletSuffStat::real_eq(1, vec_nn(&[0.4, 0.6]));
        let b = DirichletSuffStat::real_eq(1, vec_nn(&[0.5, 0.5]));
        let (_, factor) = a.merge(&b);
        assert_eq!(factor, f64::NEG_INFINITY);
    }

    #[test]
    fn posterior_realeq_with_dirichlet_prior_returns_pinned() {
        let prior = DirichletPrior::Dirichlet {
            name: 1,
            alphas: vec![2.0, 3.0],
        };
        let stat = DirichletSuffStat::real_eq(1, vec_nn(&[0.4, 0.6]));
        let q: BTreeSet<ContVarName> = [1u64].iter().copied().collect();
        let (post, marginal) = stat.posterior(&prior, &q);
        // Marginal likelihood = log Dirichlet PDF at value.
        let expected_log = log_dirichlet_pdf(
            &[NotNan::new(0.4).unwrap(), NotNan::new(0.6).unwrap()],
            &[2.0, 3.0],
        );
        assert!(
            (marginal.log_coeff - expected_log).abs() < 1e-12,
            "marginal log = {} expected {}",
            marginal.log_coeff,
            expected_log
        );
        match post.expect("posterior should be Some") {
            DirichletPrior::Pinned { name, value } => {
                assert_eq!(name, 1);
                assert_eq!(value, vec![0.4, 0.6]);
            }
            _ => panic!("expected Pinned posterior"),
        }
    }

    #[test]
    fn posterior_counts_with_pinned_prior_is_density_factor() {
        let prior = DirichletPrior::Pinned {
            name: 1,
            value: vec![0.2, 0.3, 0.5],
        };
        // Counts on stick 0: (h=2, t=0). At θ=[0.2,0.3,0.5]: 2 ln(0.2).
        let stat = stat(1, &[(0, 2, 0)]);
        let q: BTreeSet<ContVarName> = [1u64].iter().copied().collect();
        let (post, marginal) = stat.posterior(&prior, &q);
        let expected = 2.0 * 0.2_f64.ln();
        assert!(
            (marginal.log_coeff - expected).abs() < 1e-12,
            "marginal log = {} expected {}",
            marginal.log_coeff,
            expected
        );
        // Posterior prior remains Pinned at the same value.
        match post.expect("posterior should be Some") {
            DirichletPrior::Pinned { value, .. } => {
                assert_eq!(value, vec![0.2, 0.3, 0.5]);
            }
            _ => panic!("expected Pinned posterior to stay Pinned"),
        }
    }

    #[test]
    fn log_likelihood_realeq_match_returns_zero() {
        let stat = DirichletSuffStat::real_eq(1, vec_nn(&[0.4, 0.6]));
        let v = vec_nn(&[0.4, 0.6]);
        assert_eq!(stat.log_likelihood(&v), 0.0);
    }

    #[test]
    fn log_likelihood_realeq_mismatch_is_neg_infinity() {
        let stat = DirichletSuffStat::real_eq(1, vec_nn(&[0.4, 0.6]));
        let v = vec_nn(&[0.5, 0.5]);
        assert_eq!(stat.log_likelihood(&v), f64::NEG_INFINITY);
    }

    #[test]
    fn sample_from_pinned_returns_pinned_value() {
        let p = DirichletPrior::Pinned {
            name: 9,
            value: vec![0.1, 0.2, 0.7],
        };
        let mut r = rand::rngs::StdRng::seed_from_u64(0);
        let v = <DirichletPrior as Prior>::sample(&p, &mut r);
        assert_eq!(v.len(), 3);
        assert!((v[0].into_inner() - 0.1).abs() < 1e-15);
        assert!((v[1].into_inner() - 0.2).abs() < 1e-15);
        assert!((v[2].into_inner() - 0.7).abs() < 1e-15);
    }

    #[test]
    fn vec_pin_flip_weights_are_low_counts_high_real_eq() {
        let flip = DirichletFlip::VecPin {
            name: 1,
            value: vec_nn(&[0.3, 0.7]),
        };
        let (low, high) = flip.weights();
        // We just check the EvidenceLeaf shape — full SPN evaluation lives
        // in the inference-end tests.
        use crate::inference::conjugate_pairs::EvidenceLeaf;
        use crate::inference::spn::node::SpnKind;
        match low.kind() {
            SpnKind::Leaf(EvidenceLeaf::Dirichlet(DirichletSuffStat::Counts {
                name,
                counts,
                excluded,
            })) => {
                assert_eq!(*name, 1);
                assert!(counts.is_empty());
                // Low edge of VecPin records `value` in `excluded` so a
                // later RealEq(value) merge detects contradiction.
                assert_eq!(excluded.len(), 1);
                let v: &[NotNan<f64>] = excluded.iter().next().unwrap();
                assert_eq!(v.len(), 2);
            }
            _ => panic!("expected low edge = empty Counts leaf with one exclusion"),
        }
        match high.kind() {
            SpnKind::Leaf(EvidenceLeaf::Dirichlet(DirichletSuffStat::RealEq { name, value })) => {
                assert_eq!(*name, 1);
                assert_eq!(value.len(), 2);
            }
            _ => panic!("expected high edge = RealEq leaf"),
        }
    }

    #[test]
    fn vec_pin_is_pinned_observation() {
        let flip = DirichletFlip::VecPin {
            name: 1,
            value: vec_nn(&[0.5, 0.5]),
        };
        assert!(flip.is_pinned_observation());
        let other = DirichletFlip::VarElement {
            name: 2,
            index: 0,
            categories: 2,
        };
        assert!(!other.is_pinned_observation());
    }
}
