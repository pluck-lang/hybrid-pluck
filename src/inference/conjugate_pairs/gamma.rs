use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter, Result};
use std::hash::{Hash, Hasher};

use itertools::Itertools;
use ordered_float::NotNan;

use super::gamma_mixture::GammaMixture;
use super::suffstat::{ContVarName, Prior, SuffStat, PRIOR_QUANTIZATION_LEVEL};
use crate::inference::conjugate_pairs::exponential::{ExponentialObs, ExponentialPrior};
use crate::inference::conjugate_pairs::poisson::{PoissonObs, PoissonPrior};
use crate::utils::epsilon::RealEps;
use crate::utils::math::{log_gamma_marginal, log_gamma_pdf, quant, signed_logsumexp};

/// A Gamma prior over a single variable.
///
/// Three shapes:
/// - `Gamma(shape, rate)`: a standard Gamma prior.
/// - `Mixture`: a finite *signed* weighted combination of Gammas — see
///   [`GammaMixture`] for how these posteriors arise.
/// - `Pinned(r)`: a Dirac prior at `r` (from disintegration).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum GammaPrior<N = ContVarName> {
    Gamma { name: N, shape: f64, rate: f64 },
    Mixture { name: N, mixture: GammaMixture },
    Pinned { name: N, value: f64 },
}

impl<N: Clone> GammaPrior<N> {
    pub fn name(&self) -> N {
        match self {
            GammaPrior::Gamma { name, .. } => name.clone(),
            GammaPrior::Mixture { name, .. } => name.clone(),
            GammaPrior::Pinned { name, .. } => name.clone(),
        }
    }
}

impl<N> GammaPrior<N> {
    pub fn name_ref(&self) -> &N {
        match self {
            GammaPrior::Gamma { name, .. } => name,
            GammaPrior::Mixture { name, .. } => name,
            GammaPrior::Pinned { name, .. } => name,
        }
    }

    /// The distribution of `s · X` when `self` is the distribution of `X`
    /// (`s > 0`). Scaling a gamma multiplies its mean by `s`:
    /// `Gamma(shape, rate)` ⇒ `Gamma(shape, rate/s)`, a `Pinned` point `v` ⇒
    /// `v·s`, and a mixture's component rates each divide by `s` (the mixing
    /// weights are preserved — every component's mass scales by the same
    /// factor). Used to report a queried *scaled* gamma `s·g`.
    pub fn scaled(self, s: f64) -> GammaPrior<N> {
        debug_assert!(
            s > 0.0,
            "GammaPrior::scaled: scale must be positive, got {s}"
        );
        match self {
            GammaPrior::Gamma { name, shape, rate } => GammaPrior::Gamma {
                name,
                shape,
                rate: rate / s,
            },
            GammaPrior::Pinned { name, value } => GammaPrior::Pinned {
                name,
                value: value * s,
            },
            // Dividing every component rate by `s` is the substitution
            // `c → c/s`, i.e. `scale_rate(1/s)`. That scales every component's
            // *mass* by the same factor `s` (the `s^{-n}` weight shift and the
            // `s^{n+1}` from `c → c/s` combine to `s¹`, independent of `n`), so
            // the mixture is no longer normalized — re-normalize to restore the
            // original mixing weights (which `s·X` shares with `X`).
            GammaPrior::Mixture { name, mixture } => {
                let mut m = mixture.scale_rate(1.0 / s);
                m.normalize();
                GammaPrior::Mixture { name, mixture: m }
            }
        }
    }

    /// Draw a value from this prior.
    /// - `Gamma(shape, rate)`: standard Gamma draw.
    /// - `Mixture`: rejection sampling against the positive sub-mixture.
    /// - `Pinned(v)`: returns `v`.
    pub fn sample_value<R: rand::Rng>(&self, rng: &mut R) -> NotNan<f64> {
        use crate::utils::sampling::sample_gamma;
        let value = match self {
            GammaPrior::Gamma { shape, rate, .. } => 1.0 / rate * sample_gamma(rng, *shape),
            GammaPrior::Mixture { mixture, .. } => mixture.sample(rng),
            GammaPrior::Pinned { value, .. } => *value,
        };
        NotNan::new(value).expect("Gamma sample is NaN")
    }
}

impl GammaPrior<ContVarName> {
    /// Re-tag this prior with a display name. Used by the lib layer to
    /// produce client-facing `GammaPrior<String>` values without
    /// duplicating the variant data.
    pub fn with_display_name(self, name: String) -> GammaPrior<String> {
        match self {
            GammaPrior::Gamma { shape, rate, .. } => GammaPrior::Gamma { name, shape, rate },
            GammaPrior::Mixture { mixture, .. } => GammaPrior::Mixture { name, mixture },
            GammaPrior::Pinned { value, .. } => GammaPrior::Pinned { name, value },
        }
    }
}

impl<N: Hash> Hash for GammaPrior<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            GammaPrior::Gamma { name, shape, rate } => {
                "Gamma".hash(state);
                name.hash(state);
                quant(*shape, PRIOR_QUANTIZATION_LEVEL).hash(state);
                quant(*rate, PRIOR_QUANTIZATION_LEVEL).hash(state);
            }
            GammaPrior::Mixture { name, mixture } => {
                "Mixture".hash(state);
                name.hash(state);
                mixture.hash(state);
            }
            GammaPrior::Pinned { name, value } => {
                "Pinned".hash(state);
                name.hash(state);
                quant(*value, PRIOR_QUANTIZATION_LEVEL).hash(state);
            }
        }
    }
}

impl<N: PartialEq> PartialEq for GammaPrior<N> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                GammaPrior::Gamma {
                    name: n1,
                    shape: shape1,
                    rate: rate1,
                },
                GammaPrior::Gamma {
                    name: n2,
                    shape: shape2,
                    rate: rate2,
                },
            ) => {
                n1 == n2
                    && quant(*shape1, PRIOR_QUANTIZATION_LEVEL)
                        == quant(*shape2, PRIOR_QUANTIZATION_LEVEL)
                    && quant(*rate1, PRIOR_QUANTIZATION_LEVEL)
                        == quant(*rate2, PRIOR_QUANTIZATION_LEVEL)
            }
            (
                GammaPrior::Mixture {
                    name: n1,
                    mixture: m1,
                },
                GammaPrior::Mixture {
                    name: n2,
                    mixture: m2,
                },
            ) => n1 == n2 && m1 == m2,
            (
                GammaPrior::Pinned {
                    name: n1,
                    value: v1,
                },
                GammaPrior::Pinned {
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

impl<N: Eq> Eq for GammaPrior<N> {}

impl<N: Display + Hash> Display for GammaPrior<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            GammaPrior::Gamma { name, shape, rate } => {
                write!(
                    f,
                    "{} ~ Gamma({:.p$}, {:.p$})",
                    name,
                    shape,
                    rate,
                    p = crate::utils::display_precision::param_precision()
                )
            }
            GammaPrior::Mixture { name, mixture } => {
                write!(f, "{} ~ Mix[{}]", name, mixture)
            }
            GammaPrior::Pinned { name, value } => {
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

/// The two λ-indexed gamma-consumer draw families. (The symbolic-value
/// constructor `mk_value` lives as an inherent impl in
/// `inference::sampling`, next to the values it builds.)
#[derive(Debug, Clone, Copy)]
pub enum GammaDrawFamily {
    Poisson,
    Exponential,
}

/// A joint distribution over a latent gamma rate variable *and* the
/// Poisson/Exponential draw variables that reference it.
///
/// The draw families are λ-indexed conditionals (`TruncatedPoisson(λ, ·)`,
/// `TruncatedExp(λ, ·)`): λ is bound by the `gamma` component at sample
/// time. A prior joint has unconstrained draw families; a posterior joint
/// carries the observation truncations, which is what makes the draws'
/// posteriors queryable.
#[derive(Debug, Clone, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "N: serde::Serialize + Ord",
    deserialize = "N: serde::Deserialize<'de> + Ord"
))]
pub struct JointGammaPrior<N = ContVarName> {
    pub gamma: GammaPrior<N>,
    pub poisson: PoissonPrior<N>,
    pub exponential: ExponentialPrior<N>,
}

impl<N: Clone> JointGammaPrior<N> {
    /// The rate variable's name — the registry sort/lookup key for the block.
    pub fn name(&self) -> N {
        self.gamma.name()
    }
}

impl JointGammaPrior<ContVarName> {
    /// Wrap a bare rate prior with no attached draws (the registration
    /// form: draw structure attaches via evidence suff stats).
    pub fn from_gamma(gamma: GammaPrior<ContVarName>) -> Self {
        JointGammaPrior {
            gamma,
            poisson: PoissonPrior::unconstrained(std::iter::empty()),
            exponential: ExponentialPrior::unconstrained(std::iter::empty()),
        }
    }

    /// Re-tag every variable (rate + all draws) with a display name.
    pub fn with_display_names(&self, f: impl Fn(ContVarName) -> String) -> JointGammaPrior<String> {
        JointGammaPrior {
            gamma: self.gamma.clone().with_display_name(f(self.gamma.name())),
            poisson: self.poisson.map_names(&f),
            exponential: self.exponential.map_names(&f),
        }
    }

    /// Project to `query_vars`: draws outside the query are marginalized
    /// out — for the factorized joint that is simply dropping them. The
    /// rate component is always kept: it is the mixing variable the
    /// remaining truncated draws are indexed by.
    fn project(self, query_vars: &BTreeSet<ContVarName>) -> Self {
        JointGammaPrior {
            gamma: self.gamma,
            poisson: self.poisson.restricted_to(query_vars),
            exponential: self.exponential.restricted_to(query_vars),
        }
    }
}

impl<N: Display + Hash + Clone> Display for JointGammaPrior<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        writeln!(f, "{}", self.gamma)?;
        self.poisson.fmt(self.gamma.name(), f)?;
        self.exponential.fmt(self.gamma.name(), f)?;
        Ok(())
    }
}

impl Prior for JointGammaPrior<ContVarName> {
    type Realization = BTreeMap<ContVarName, NotNan<f64>>;

    fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_ {
        std::iter::once(self.gamma.name_ref())
            .chain(self.poisson.scope())
            .chain(self.exponential.scope())
    }

    /// Draw the rate from the gamma component, then every draw from its
    /// truncated conditional given that rate.
    fn sample<R: rand::Rng>(&self, rng: &mut R) -> BTreeMap<ContVarName, NotNan<f64>> {
        let gamma_draw = self.gamma.sample_value(rng);
        let mut realization = BTreeMap::new();
        realization.insert(self.gamma.name(), gamma_draw);
        realization.extend(self.poisson.sample(gamma_draw, rng));
        realization.extend(self.exponential.sample(gamma_draw, rng));
        realization
    }

    fn push_into(
        &self,
        sampled: BTreeMap<ContVarName, NotNan<f64>>,
        out: &mut super::AssignmentBuilder,
    ) {
        // The whole block lives in the assignment's gamma family, the way
        // a Gaussian block splays across its var_order.
        for (name, value) in sampled {
            out.push_gamma(name, value);
        }
    }
}

impl From<JointGammaPrior> for super::PosteriorLeaf {
    fn from(p: JointGammaPrior) -> Self {
        super::PosteriorLeaf::Gamma(p)
    }
}

/// Sufficient statistics for a single Gamma variable.
///
/// Accumulates every kind of observation that references the variable:
/// - `poisson`: per-draw integer ranges on `k ~ Poisson(λ)`
/// - `exponential`: per-draw real ranges / points on `x ~ Exp(λ)`
/// - `pin`: a direct Dirac observation `λ = r` (from disintegration)
/// - `excluded`: explicitly excluded values `λ ≠ r`. Excluded values
///   have Lebesgue measure 0 under Gamma so they do not change the
///   density; they are kept to reject a later `pin = r` observation.
///
/// Dirac state (`pin`, exponential `Eq`) is scored in `posterior`, not
/// folded into `merge` factors — see `merge` for why.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GammaSuffStat {
    name: ContVarName,
    poisson: PoissonObs,
    exponential: ExponentialObs,
    /// `λ = r` (disintegration). Scored ε¹ in `posterior` — see `merge`
    /// for why it isn't folded into a merge factor instead.
    pin: Option<NotNan<f64>>,
    /// Extra log-coefficient carried by the pin: the change-of-variables
    /// Jacobian from a *scaled* disintegration (see
    /// [`GammaSuffStat::real_eq_scaled`]). Zero for an unscaled pin. Added
    /// to the pinned-density `z` in `posterior`.
    pin_log_coeff: NotNan<f64>,
    /// `λ ≠ r`. Measure-zero under Gamma so it does not change the
    /// density; kept to reject a later `pin = r`.
    excluded: BTreeSet<NotNan<f64>>,
}

impl GammaSuffStat {
    fn empty(name: ContVarName) -> Self {
        GammaSuffStat {
            name,
            poisson: PoissonObs::default(),
            exponential: ExponentialObs::default(),
            pin: None,
            pin_log_coeff: NotNan::new(0.0).unwrap(),
            excluded: BTreeSet::new(),
        }
    }

    /// Observation of one or more Poisson draws `var ~ Poisson(name)`.
    pub fn poisson(name: ContVarName, poisson: PoissonObs) -> Self {
        GammaSuffStat {
            poisson,
            ..Self::empty(name)
        }
    }

    /// Observation of one or more Exponential draws `var ~ Exp(name)`.
    pub fn exponential(name: ContVarName, exponential: ExponentialObs) -> Self {
        GammaSuffStat {
            exponential,
            ..Self::empty(name)
        }
    }

    /// Dirac observation `λ = value` (disintegration). Test-only fixture;
    /// production code builds rate pins via [`GammaSuffStat::real_eq_scaled`].
    #[cfg(test)]
    pub fn real_eq(name: ContVarName, value: f64) -> Self {
        GammaSuffStat {
            pin: Some(NotNan::new(value).unwrap()),
            ..Self::empty(name)
        }
    }

    /// Disintegration of a *scaled* rate pin `scale·λ = value`: pins
    /// `λ = value/scale` and carries the change-of-variables Jacobian
    /// `−ln(scale)` in `pin_log_coeff` (density of `scale·λ` at `value`
    /// equals density of `λ` at `value/scale` times `1/scale`).
    pub fn real_eq_scaled(name: ContVarName, value: f64, scale: f64) -> Self {
        debug_assert!(
            scale > 0.0,
            "real_eq_scaled: scale must be positive, got {scale}"
        );
        GammaSuffStat {
            pin: Some(NotNan::new(value / scale).unwrap()),
            pin_log_coeff: NotNan::new(-scale.ln()).unwrap(),
            ..Self::empty(name)
        }
    }

    /// Measure-zero exclusion `λ ≠ r`.
    pub fn not_eq(name: ContVarName, r: f64) -> Self {
        GammaSuffStat {
            excluded: BTreeSet::from([NotNan::new(r).unwrap()]),
            ..Self::empty(name)
        }
    }

    pub fn name(&self) -> ContVarName {
        self.name
    }

    /// Marginal likelihood of all draw observations at a fixed rate `λ`
    /// (draws integrated out), i.e. P(obs | λ)
    fn marginal_log_likelihood_contribution(&self, lambda: &NotNan<f64>) -> RealEps {
        self.poisson.marginal_log_likelihood_contribution(lambda)
            * self
                .exponential
                .marginal_log_likelihood_contribution(lambda)
    }
}

impl SuffStat for GammaSuffStat {
    type ConjugatePrior = JointGammaPrior;

    fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_ {
        std::iter::once(&self.name)
            .chain(self.poisson.scope())
            .chain(self.exponential.scope())
    }

    /// Merging a pin with draw observations emits no Beta-style
    /// discharge factor: Beta evaluates its counts at the pin because it
    /// drops them from the merged stat; here the observations are kept —
    /// (a) exponential `Eq` factors carry ε-powers a plain `f64` cannot,
    /// and (b) the draws' truncations must survive into the joint
    /// posterior to stay queryable. `posterior`'s pinned branches score
    /// `P(obs | pin)` exactly once instead.
    fn merge(&self, other: &Self) -> (Self, f64) {
        debug_assert_eq!(
            self.name, other.name,
            "GammaSuffStat::merge requires overlapping scopes"
        );
        let (poisson, poisson_logp) = self.poisson.merge(&other.poisson);
        let (exponential, exp_logp) = self.exponential.merge(&other.exponential);
        let excluded: BTreeSet<NotNan<f64>> =
            self.excluded.union(&other.excluded).copied().collect();
        // Two distinct pins contradict; otherwise keep the present one.
        // Agreeing pins sum their Jacobians (`pin_log_coeff`) — each scaled
        // disintegration contributed its own `−ln s` change-of-variables
        // factor. (Only reachable when the same gamma is pinned by two
        // scaled `real_eq`s; the common case has a single pin.)
        let (pin, pin_log_coeff, pin_logp) = match (self.pin, other.pin) {
            (Some(a), Some(b)) if a != b => (Some(a), self.pin_log_coeff, f64::NEG_INFINITY),
            (Some(a), Some(_)) => (Some(a), self.pin_log_coeff + other.pin_log_coeff, 0.0),
            (Some(a), None) => (Some(a), self.pin_log_coeff, 0.0),
            (None, b) => (b, other.pin_log_coeff, 0.0),
        };
        // A pin that is also excluded is a contradiction.
        let excl_logp = if pin.is_some_and(|p| excluded.contains(&p)) {
            f64::NEG_INFINITY
        } else {
            0.0
        };
        (
            GammaSuffStat {
                name: self.name,
                poisson,
                exponential,
                pin,
                pin_log_coeff,
                excluded,
            },
            poisson_logp + exp_logp + pin_logp + excl_logp,
        )
    }

    /// Indicator log-likelihood of all observations at a full realization
    /// of the block (rate + draws). With the draw values in the
    /// realization every observed event is determined, so each factor is
    /// `0.0` / `−∞`.
    fn log_likelihood(&self, value: &BTreeMap<ContVarName, NotNan<f64>>) -> f64 {
        let lambda = value.get(&self.name).unwrap();
        if let Some(pin) = self.pin {
            if *lambda != pin {
                return f64::NEG_INFINITY;
            }
        }
        if self.excluded.contains(lambda) {
            return f64::NEG_INFINITY;
        }
        self.poisson.log_likelihood(value) + self.exponential.log_likelihood(value)
    }

    fn posterior(
        &self,
        prior: &JointGammaPrior,
        query_vars: &BTreeSet<ContVarName>,
    ) -> (Option<JointGammaPrior>, RealEps) {
        debug_assert_eq!(self.scope().collect_vec(), prior.scope().collect_vec());
        let name = self.name;

        // The λ component of the posterior, paired with the marginal `z`.
        // Observation marginals are RealEps: each exponential Dirac
        // contributes one ε-power; interval events contribute power 0.
        let (candidate, z): (Option<GammaPrior>, RealEps) = match (self.pin, &prior.gamma) {
            (_, GammaPrior::Mixture { .. }) => {
                panic!("GammaSuffStat::posterior: Mixture priors are not supported")
            }

            // Pinned stat λ = r, continuous prior: density at r (ε¹) times
            // the observation marginal at r (which carries its own ε's).
            // `pin_log_coeff` adds the scaled-disintegration Jacobian to the
            // density term (see `GammaSuffStat::real_eq_scaled`).
            (Some(r), GammaPrior::Gamma { shape, rate, .. }) => {
                if self.excluded.contains(&r) {
                    (None, RealEps::zero())
                } else {
                    let z = RealEps::from_log(
                        log_gamma_pdf(r.into_inner(), *shape, *rate)
                            + self.pin_log_coeff.into_inner(),
                        1,
                    ) * self.marginal_log_likelihood_contribution(&r);
                    (
                        Some(GammaPrior::Pinned {
                            name,
                            value: r.into_inner(),
                        }),
                        z,
                    )
                }
            }
            // Pinned stat against a pinned prior: agree (no extra density
            // power — the prior is already a point mass) or vanish. (A scaled
            // pin's `pin_log_coeff` is carried through for uniformity, but
            // this branch is unreachable for scaled pins: a scale only arises
            // when the gamma is a genuine variable, i.e. a `Gamma` prior.)
            (Some(r), GammaPrior::Pinned { value: s, .. }) => {
                if r.into_inner() == *s && !self.excluded.contains(&r) {
                    let z = self.marginal_log_likelihood_contribution(&r)
                        * RealEps::from_log(self.pin_log_coeff.into_inner(), 0);
                    (Some(GammaPrior::Pinned { name, value: *s }), z)
                } else {
                    (None, RealEps::zero())
                }
            }

            // Unpinned stat against a pinned prior: score observations at
            // the pinned value.
            (None, GammaPrior::Pinned { value: s, .. }) => {
                let s_nn = NotNan::new(*s).unwrap();
                if self.excluded.contains(&s_nn) {
                    (None, RealEps::zero())
                } else {
                    let z = self.marginal_log_likelihood_contribution(&s_nn);
                    (Some(GammaPrior::Pinned { name, value: *s }), z)
                }
            }

            // Unpinned stat against a Gamma prior: the conjugate update is
            // a finite signed mixture of Gammas. `excluded` is
            // measure-zero under Gamma and is ignored.
            (None, GammaPrior::Gamma { shape, rate, .. }) => {
                let mixture = self.poisson.mixture_representation()
                    * self.exponential.mixture_representation();
                // Z = Σⱼ sⱼ wⱼ · Γ(a+nⱼ)β^a / (Γ(a)(β+cⱼ)^{a+nⱼ}).
                let (z_log, z_sign) = signed_logsumexp(mixture.0.iter().map(|(t, w)| {
                    (
                        w.log_w
                            + log_gamma_marginal(*shape, *rate, t.n.into_inner(), t.c.into_inner()),
                        w.sign,
                    )
                }));
                if z_sign <= 0 {
                    // Exact cancellation / numerically non-positive mass.
                    (None, RealEps::zero())
                } else {
                    // Mixture weights are plain reals, so the ε-powers of
                    // exponential Diracs are accounted explicitly here.
                    let z = RealEps::from_log(z_log, self.exponential.n_exact() as u32);
                    // Fold the Gamma(a, β) prior into the likelihood
                    // mixture: each term λ^{a−1+nⱼ} e^{−(β+cⱼ)λ} is an
                    // unnormalised Gamma(a+nⱼ, β+cⱼ) component.
                    let mut post = GammaMixture::gamma(*shape, *rate) * mixture;
                    post.normalize();
                    let posterior = match post.as_single_gamma() {
                        Some((shape, rate)) => GammaPrior::Gamma { name, shape, rate },
                        None => GammaPrior::Mixture {
                            name,
                            mixture: post,
                        },
                    };
                    (Some(posterior), z)
                }
            }
        };

        if z.is_zero() || !self.scope().any(|n| query_vars.contains(n)) {
            return (None, z);
        }
        // Couple the λ component with the draws' conditional families
        // (the observation truncations) — identical across branches —
        // then project to the queried variables.
        let joint = candidate.map(|gamma| {
            JointGammaPrior {
                gamma,
                poisson: PoissonPrior::conditioned_on(&self.poisson),
                exponential: ExponentialPrior::conditioned_on(&self.exponential),
            }
            .project(query_vars)
        });
        (joint, z)
    }

    fn read_realization(&self, a: &super::Assignment) -> BTreeMap<ContVarName, NotNan<f64>> {
        self.scope().map(|n| (*n, a.gamma_value(*n))).collect()
    }

    /// Build the block's joint prior: the rate prior from the registry
    /// plus unconstrained families over this stat's draws.
    ///
    /// Registry entries carry empty draw maps — draws are deliberately
    /// not registry-registered, so any SPN that references a draw must
    /// contain its joint gamma leaf (the missing-variable prior fallbacks
    /// can never cover a draw).
    fn lookup_prior_in(&self, reg: &super::PriorRegistry) -> JointGammaPrior {
        JointGammaPrior {
            gamma: reg.gamma_prior(self.name()).gamma.clone(),
            poisson: PoissonPrior::unconstrained(self.poisson.scope().copied()),
            exponential: ExponentialPrior::unconstrained(self.exponential.scope().copied()),
        }
    }
}

use crate::inference::conjugate_pairs::EvidenceLeaf;
use crate::inference::spn::node::Spn;
use crate::utils::intervals::{Interval, IntervalOrEq};

/// Gamma-family BDD-flip variants: Dirac pinning of the rate variable,
/// and observations of Poisson / Exponential draws.
pub enum GammaFlip {
    /// Rate pinning for disintegration: `real_eq(scale·rate, value)`.
    /// High edge: pin `λ = value/scale` (carrying the `−ln scale` Jacobian);
    /// low edge: exclusion `λ ≠ value/scale`. `scale` is the rate multiplier
    /// (`1.0` for a bare `real_eq(rate, value)`).
    RatePin {
        name: ContVarName,
        value: f64,
        scale: f64,
    },
    /// Observation event on a Poisson draw `draw ~ Poisson(scale·rate)`: a
    /// point `Eq(k)` or a half-open interval. A regular flip — both
    /// edges carry real probability mass; the low edge is the exact
    /// complement (`[0, k) ∪ [k+1, ∞)` for a point, `[0, geq) ∪ [lt, ∞)`
    /// for an interval). `scale` is the rate multiplier (a scaled gamma).
    PoissonEvent {
        gamma: ContVarName,
        draw: ContVarName,
        event: IntervalOrEq<u64>,
        scale: f64,
    },
    /// Dirac observation on an Exponential draw `draw ~ Exp(scale·rate)`:
    /// `real_eq(draw, v)`. High edge: `draw = v` (density, ε¹); low edge:
    /// exclusion `draw ≠ v` (probability 1, keeps the draw in SPN scope).
    /// Treated as a pinned observation. `scale` is the rate multiplier.
    ExpEq {
        gamma: ContVarName,
        draw: ContVarName,
        value: f64,
        scale: f64,
    },
    /// Interval event on an Exponential draw `draw ~ Exp(scale·rate)`:
    /// `draw ∈ [geq, lt)`. A regular flip; the low edge is the complement
    /// `[0, geq) ∪ [lt, ∞)` — exact, each boundary point belongs to
    /// exactly one side under the half-open convention. `scale` is the
    /// rate multiplier.
    ExpInterval {
        gamma: ContVarName,
        draw: ContVarName,
        interval: Interval<NotNan<f64>>,
        scale: f64,
    },
}

impl GammaFlip {
    /// SPN-weight pair `(low_edge, high_edge)` carried on the BDD variable.
    pub fn weights(&self) -> (Spn<EvidenceLeaf>, Spn<EvidenceLeaf>) {
        let leaf = |stat: GammaSuffStat| Spn::leaf(EvidenceLeaf::Gamma(stat));
        match self {
            GammaFlip::RatePin { name, value, scale } => (
                // Both edges are on `λ = value/scale` (the disintegrated pin).
                leaf(GammaSuffStat::not_eq(*name, *value / *scale)),
                leaf(GammaSuffStat::real_eq_scaled(*name, *value, *scale)),
            ),
            GammaFlip::PoissonEvent {
                gamma,
                draw,
                event,
                scale,
            } => {
                let s = NotNan::new(*scale).expect("gamma scale must be finite");
                let piece = |c: IntervalOrEq<u64>| {
                    leaf(GammaSuffStat::poisson(
                        *gamma,
                        PoissonObs::constraint(*draw, c).with_scale(s),
                    ))
                };
                // The complement's lower/upper pieces. `Spn::sum`
                // collapses a single piece to its leaf; an empty
                // complement (the full support — callers shortcut that
                // case to a constant) collapses to zero.
                let (below, from) = match event {
                    IntervalOrEq::Eq(k) => (*k, Some(k + 1)),
                    IntervalOrEq::Interval(interval) => (interval.geq, interval.lt),
                };
                let mut complement = Vec::new();
                if below > 0 {
                    complement.push(piece(IntervalOrEq::lt(below)));
                }
                if let Some(from) = from {
                    complement.push(piece(IntervalOrEq::geq(from)));
                }
                (Spn::sum(complement), piece(event.clone()))
            }
            GammaFlip::ExpEq {
                gamma,
                draw,
                value,
                scale,
            } => {
                let s = NotNan::new(*scale).expect("gamma scale must be finite");
                (
                    leaf(GammaSuffStat::exponential(
                        *gamma,
                        ExponentialObs::not_eq(*draw, *value).with_scale(s),
                    )),
                    leaf(GammaSuffStat::exponential(
                        *gamma,
                        ExponentialObs::real_eq(*draw, *value).with_scale(s),
                    )),
                )
            }
            GammaFlip::ExpInterval {
                gamma,
                draw,
                interval,
                scale,
            } => {
                let s = NotNan::new(*scale).expect("gamma scale must be finite");
                let piece = |i: Interval<NotNan<f64>>| {
                    leaf(GammaSuffStat::exponential(
                        *gamma,
                        ExponentialObs::interval(*draw, i).with_scale(s),
                    ))
                };
                let zero = NotNan::new(0.0).unwrap();
                let mut complement = Vec::new();
                if interval.geq > zero {
                    complement.push(piece(Interval {
                        geq: zero,
                        lt: Some(interval.geq),
                    }));
                }
                if let Some(lt) = interval.lt {
                    complement.push(piece(Interval::geq(lt)));
                }
                (Spn::sum(complement), piece(interval.clone()))
            }
        }
    }

    /// Per-variant hash discriminator used as part of the BDD-variable
    /// callstack key.
    pub fn discriminator(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        match self {
            GammaFlip::RatePin { name, value, scale } => {
                (0u8, name, value.to_bits(), scale.to_bits()).hash(&mut hasher);
            }
            GammaFlip::PoissonEvent {
                gamma,
                draw,
                event,
                scale,
            } => {
                (1u8, gamma, draw, event, scale.to_bits()).hash(&mut hasher);
            }
            GammaFlip::ExpEq {
                gamma,
                draw,
                value,
                scale,
            } => {
                (2u8, gamma, draw, value.to_bits(), scale.to_bits()).hash(&mut hasher);
            }
            GammaFlip::ExpInterval {
                gamma,
                draw,
                interval,
                scale,
            } => {
                (3u8, gamma, draw, interval, scale.to_bits()).hash(&mut hasher);
            }
        }
        hasher.finish().wrapping_add(0x6A09E667F3BCC909)
    }

    /// `RatePin` and `ExpEq` are measure-zero observations — looked up by
    /// callstack hash alone in `observation_vars`. The event/interval
    /// variants are regular flips with real probability mass on both
    /// edges.
    pub fn is_pinned_observation(&self) -> bool {
        matches!(self, GammaFlip::RatePin { .. } | GammaFlip::ExpEq { .. })
    }

    /// Probability used by the sample-mode flip resolver: the indicator
    /// of the observed event at the assignment's draw value. `None` for
    /// the observation variants. The draw must already be covered by the
    /// assignment (`SampleState::ensure_gamma_draw`).
    ///
    /// `scale` is irrelevant here: the draw was *sampled* at the scaled
    /// rate `scale·λ` (in `ensure_gamma_draw`), so its realised value is
    /// already in draw units and the indicator checks the constraint
    /// directly.
    pub fn sample_probability(&self, a: &super::Assignment) -> Option<f64> {
        match self {
            GammaFlip::RatePin { .. } | GammaFlip::ExpEq { .. } => None,
            GammaFlip::PoissonEvent { draw, event, .. } => {
                let v = a.gamma_value(*draw).into_inner() as u64;
                Some(if event.contains(v) { 1.0 } else { 0.0 })
            }
            GammaFlip::ExpInterval { draw, interval, .. } => {
                let v = a.gamma_value(*draw);
                Some(if interval.contains(v) { 1.0 } else { 0.0 })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::conjugate_pairs::poisson::log_prob as poisson_log_prob;
    use crate::utils::math::ln_factorial;

    const LAMBDA: ContVarName = 1; // the Gamma variable
    const K1: ContVarName = 10; // Poisson draw names
    const K2: ContVarName = 11;
    const X1: ContVarName = 12; // Exponential draw names

    fn nn(x: f64) -> NotNan<f64> {
        NotNan::new(x).unwrap()
    }

    /// Gamma prior parametrised by shape and **rate**.
    fn gamma_prior(shape: f64, rate: f64) -> GammaPrior {
        GammaPrior::Gamma {
            name: LAMBDA,
            shape,
            rate,
        }
    }

    fn query() -> BTreeSet<ContVarName> {
        [LAMBDA].iter().copied().collect()
    }

    /// Joint prior matching `stat`'s scope: `gamma` on the rate plus
    /// unconstrained families over the stat's draws — what
    /// `lookup_prior_in` builds from a registry.
    fn joint_with(gamma: GammaPrior, stat: &GammaSuffStat) -> JointGammaPrior {
        JointGammaPrior {
            gamma,
            poisson: PoissonPrior::unconstrained(stat.poisson.scope().copied()),
            exponential: ExponentialPrior::unconstrained(stat.exponential.scope().copied()),
        }
    }

    fn joint_prior(stat: &GammaSuffStat, shape: f64, rate: f64) -> JointGammaPrior {
        joint_with(gamma_prior(shape, rate), stat)
    }

    /// Full realization of a block, for indicator `log_likelihood` tests.
    fn realization(entries: &[(ContVarName, f64)]) -> BTreeMap<ContVarName, NotNan<f64>> {
        entries.iter().map(|&(n, v)| (n, nn(v))).collect()
    }

    /// Mean of a posterior: Σⱼ sⱼ wⱼ · shapeⱼ/rateⱼ (a single Gamma is the
    /// one-component case).
    fn posterior_mean(p: &GammaPrior) -> f64 {
        match p {
            GammaPrior::Gamma { shape, rate, .. } => shape / rate,
            GammaPrior::Mixture { mixture, .. } => mixture
                .components()
                .map(|(shape, rate, log_w, sign)| sign as f64 * log_w.exp() * shape / rate)
                .sum(),
            GammaPrior::Pinned { value, .. } => *value,
        }
    }

    /// Brute-force `∫ likelihood(λ)·Gamma(λ; shape, rate) dλ` (and the
    /// posterior mean) by trapezoid quadrature, using the marginal
    /// observation likelihood at each rate.
    fn brute_force_z_and_mean(stat: &GammaSuffStat, shape: f64, rate: f64) -> (f64, f64) {
        let steps = 400_000;
        let hi = 50.0;
        let dx = hi / steps as f64;
        let mut z = 0.0;
        let mut mean = 0.0;
        for i in 1..steps {
            let x = i as f64 * dx;
            let obs_ll = stat.marginal_log_likelihood_contribution(&nn(x)).log_coeff;
            let f = (log_gamma_pdf(x, shape, rate) + obs_ll).exp() * dx;
            z += f;
            mean += x * f;
        }
        (z, mean / z)
    }

    /// `[lo, hi]` Poisson observation, built by intersecting `geq`/`leq`.
    fn poisson_between(var: ContVarName, lo: u64, hi: u64) -> GammaSuffStat {
        GammaSuffStat::poisson(LAMBDA, PoissonObs::geq(var, lo))
            .merge(&GammaSuffStat::poisson(LAMBDA, PoissonObs::leq(var, hi)))
            .0
    }

    // -------------------- merge --------------------

    #[test]
    fn merge_same_draw_ranges_intersect() {
        // k ≤ 2 AND k ≥ 2 ⇒ k ∈ [2, 3) = {2}. (The interval form stays
        // an interval — `Eq` is reserved for syntactic point
        // observations; both score identically for integer draws.)
        let a = GammaSuffStat::poisson(LAMBDA, PoissonObs::leq(K1, 2));
        let b = GammaSuffStat::poisson(LAMBDA, PoissonObs::geq(K1, 2));
        let (m, f) = a.merge(&b);
        assert_eq!(f, 0.0);
        let singleton = IntervalOrEq::Interval(Interval {
            geq: 2,
            lt: Some(3),
        });
        assert_eq!(
            m,
            GammaSuffStat::poisson(LAMBDA, PoissonObs::constraint(K1, singleton))
        );
    }

    #[test]
    fn merge_distinct_draws_accumulate() {
        let a = GammaSuffStat::poisson(LAMBDA, PoissonObs::eq(K1, 2));
        let b = GammaSuffStat::poisson(LAMBDA, PoissonObs::eq(K2, 0));
        let (m, f) = a.merge(&b);
        assert_eq!(f, 0.0);
        assert_eq!(m.poisson.0.len(), 2);
    }

    #[test]
    fn merge_empty_intersection_contradicts() {
        let a = GammaSuffStat::poisson(LAMBDA, PoissonObs::leq(K1, 1));
        let b = GammaSuffStat::poisson(LAMBDA, PoissonObs::geq(K1, 3));
        let (_, f) = a.merge(&b);
        assert_eq!(f, f64::NEG_INFINITY);
    }

    #[test]
    fn merge_mixed_families_accumulate() {
        let a = GammaSuffStat::poisson(LAMBDA, PoissonObs::eq(K1, 1));
        let b = GammaSuffStat::exponential(LAMBDA, ExponentialObs::geq(X1, 0.5));
        let (m, f) = a.merge(&b);
        assert_eq!(f, 0.0);
        assert_eq!(m.poisson.0.len(), 1);
        assert_eq!(m.exponential.ranges().len(), 1);
    }

    #[test]
    fn merge_pins_agree_and_disagree() {
        let a = GammaSuffStat::real_eq(LAMBDA, 1.5);
        let (_, f_ok) = a.merge(&GammaSuffStat::real_eq(LAMBDA, 1.5));
        assert_eq!(f_ok, 0.0);
        let (_, f_bad) = a.merge(&GammaSuffStat::real_eq(LAMBDA, 2.5));
        assert_eq!(f_bad, f64::NEG_INFINITY);
    }

    #[test]
    fn merge_pin_with_exclusion_contradicts() {
        let a = GammaSuffStat::real_eq(LAMBDA, 1.5);
        let b = GammaSuffStat::not_eq(LAMBDA, 1.5);
        let (_, f) = a.merge(&b);
        assert_eq!(f, f64::NEG_INFINITY);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "overlapping scopes")]
    fn merge_disjoint_panics_in_debug() {
        let a = GammaSuffStat::poisson(1, PoissonObs::eq(K1, 1));
        let b = GammaSuffStat::poisson(2, PoissonObs::eq(K1, 1));
        let _ = a.merge(&b);
    }

    // -------------------- posterior: conjugate updates --------------------

    #[test]
    fn poisson_exact_obs_is_standard_conjugate_update() {
        // λ ~ Gamma(a, rate β), observe k = m exactly:
        // posterior Gamma(a + m, β + 1), Z = Γ(a+m)β^a / (Γ(a)(β+1)^{a+m} m!).
        let (a, beta, m) = (2.0, 2.0, 3.0);
        let stat = GammaSuffStat::poisson(LAMBDA, PoissonObs::eq(K1, m as u64));
        let (post, z) = stat.posterior(&joint_prior(&stat, a, beta), &query());
        assert_eq!(z.power, 0);
        let expected_z = log_gamma_marginal(a, beta, m, 1.0) - ln_factorial(m as u64);
        assert!((z.log_coeff - expected_z).abs() < 1e-10);
        match post.expect("posterior should be Some").gamma {
            GammaPrior::Gamma { shape, rate, .. } => {
                assert!((shape - (a + m)).abs() < 1e-12);
                assert!((rate - (beta + 1.0)).abs() < 1e-12);
            }
            other => panic!("expected single Gamma posterior, got {}", other),
        }
    }

    #[test]
    fn exponential_dirac_obs_is_standard_conjugate_update() {
        // λ ~ Gamma(a, rate β), observe x = v exactly (density λe^{−λv}):
        // posterior Gamma(a + 1, β + v), Z carries ε¹.
        let (a, beta, v) = (2.0, 2.0, 0.7);
        let stat = GammaSuffStat::exponential(LAMBDA, ExponentialObs::real_eq(X1, v));
        let (post, z) = stat.posterior(&joint_prior(&stat, a, beta), &query());
        assert_eq!(z.power, 1);
        let expected_z = log_gamma_marginal(a, beta, 1.0, v);
        assert!((z.log_coeff - expected_z).abs() < 1e-10);
        match post.expect("posterior should be Some").gamma {
            GammaPrior::Gamma { shape, rate, .. } => {
                assert!((shape - (a + 1.0)).abs() < 1e-12);
                assert!((rate - (beta + v)).abs() < 1e-12);
            }
            other => panic!("expected single Gamma posterior, got {}", other),
        }
    }

    // -------------------- posterior: SCALED conjugate updates --------------------

    #[test]
    fn scaled_exponential_dirac_obs_conjugate_update() {
        // x ~ Exp(s·λ), observe x = v exactly: density (s·λ)e^{−s·λ·v}.
        // Posterior Gamma(a + 1, β + s·v); Z carries ε¹ with the ln(s)
        // Jacobian: log_coeff = ln s + log_gamma_marginal(a, β, 1, s·v).
        let (a, beta, v, s) = (2.0, 2.0, 0.7, 5.0);
        let obs = ExponentialObs::real_eq(X1, v).with_scale(nn(s));
        let stat = GammaSuffStat::exponential(LAMBDA, obs);
        let (post, z) = stat.posterior(&joint_prior(&stat, a, beta), &query());
        assert_eq!(z.power, 1);
        let expected_z = s.ln() + log_gamma_marginal(a, beta, 1.0, s * v);
        assert!((z.log_coeff - expected_z).abs() < 1e-10);
        match post.expect("posterior should be Some").gamma {
            GammaPrior::Gamma { shape, rate, .. } => {
                assert!((shape - (a + 1.0)).abs() < 1e-12);
                assert!((rate - (beta + s * v)).abs() < 1e-12);
            }
            other => panic!("expected single Gamma posterior, got {}", other),
        }
    }

    #[test]
    fn scaled_exponential_survival_carries_no_jacobian() {
        // P(x ≥ b | s·λ) = e^{−s·λ·b}: a probability (power 0), so no ln(s)
        // term — the survival mixture term has n = 0.
        let (a, beta, b, s) = (2.0, 1.0, 0.5, 3.0);
        let obs = ExponentialObs::geq(X1, b).with_scale(nn(s));
        let stat = GammaSuffStat::exponential(LAMBDA, obs);
        let (_, z) = stat.posterior(&joint_prior(&stat, a, beta), &query());
        assert_eq!(z.power, 0);
        let expected_z = log_gamma_marginal(a, beta, 0.0, s * b);
        assert!((z.log_coeff - expected_z).abs() < 1e-10);
    }

    #[test]
    fn scaled_poisson_exact_obs_conjugate_update() {
        // k ~ Poisson(s·λ), observe k = m: pmf (s·λ)^m e^{−s·λ}/m!.
        // Posterior Gamma(a + m, β + s); Z (power 0) includes m·ln(s).
        let (a, beta, m, s) = (2.0, 2.0, 3u64, 1.5);
        let obs = PoissonObs::eq(K1, m).with_scale(nn(s));
        let stat = GammaSuffStat::poisson(LAMBDA, obs);
        let (post, z) = stat.posterior(&joint_prior(&stat, a, beta), &query());
        assert_eq!(z.power, 0);
        let expected_z =
            (m as f64) * s.ln() + log_gamma_marginal(a, beta, m as f64, s) - ln_factorial(m);
        assert!((z.log_coeff - expected_z).abs() < 1e-10);
        match post.expect("posterior should be Some").gamma {
            GammaPrior::Gamma { shape, rate, .. } => {
                assert!((shape - (a + m as f64)).abs() < 1e-12);
                assert!((rate - (beta + s)).abs() < 1e-12);
            }
            other => panic!("expected single Gamma posterior, got {}", other),
        }
    }

    // -------------------- posterior: SCALED disintegration --------------------

    #[test]
    fn scaled_rate_pin_pins_quotient_and_carries_jacobian() {
        // `s·g = v` ⇒ pin g = v/s; density picks up the `1/s` Jacobian, so
        // z.log_coeff = log_gamma_pdf(v/s; a, β) − ln s, power 1 (a Dirac).
        let (a, beta, v, s) = (2.0, 1.0, 2.0, 4.0);
        let stat = GammaSuffStat::real_eq_scaled(LAMBDA, v, s);
        let (post, z) = stat.posterior(&joint_prior(&stat, a, beta), &query());
        assert_eq!(z.power, 1);
        let expected = log_gamma_pdf(v / s, a, beta) - s.ln();
        assert!((z.log_coeff - expected).abs() < 1e-10);
        match post.expect("posterior should be Some").gamma {
            GammaPrior::Pinned { value, .. } => assert!((value - v / s).abs() < 1e-12),
            other => panic!("expected Pinned posterior, got {}", other),
        }
    }

    #[test]
    fn gamma_prior_scaled_transforms_each_variant() {
        // Gamma: rate divides by s, shape unchanged.
        match (GammaPrior::Gamma {
            name: LAMBDA,
            shape: 3.0,
            rate: 2.0,
        })
        .scaled(4.0)
        {
            GammaPrior::Gamma { shape, rate, .. } => {
                assert!((shape - 3.0).abs() < 1e-12);
                assert!((rate - 0.5).abs() < 1e-12);
            }
            other => panic!("expected Gamma, got {other}"),
        }
        // Pinned: value multiplies by s.
        match (GammaPrior::Pinned {
            name: LAMBDA,
            value: 1.5,
        })
        .scaled(4.0)
        {
            GammaPrior::Pinned { value, .. } => assert!((value - 6.0).abs() < 1e-12),
            other => panic!("expected Pinned, got {other}"),
        }
        // Mixture: component rates divide by s; mixing weights preserved
        // (still normalized to 1).
        let stat = GammaSuffStat::poisson(LAMBDA, PoissonObs::geq(K1, 2));
        let (post, _) = stat.posterior(&joint_prior(&stat, 2.0, 1.0), &query());
        let mix = match post.unwrap().gamma {
            GammaPrior::Mixture { mixture, .. } => mixture,
            other => panic!("expected Mixture, got {other}"),
        };
        let s = 4.0;
        let scaled = match (GammaPrior::Mixture {
            name: LAMBDA,
            mixture: mix.clone(),
        })
        .scaled(s)
        {
            GammaPrior::Mixture { mixture, .. } => mixture,
            other => panic!("expected Mixture, got {other}"),
        };
        let total: f64 = scaled
            .components()
            .map(|(_, _, lw, sign)| sign as f64 * lw.exp())
            .sum();
        assert!((total - 1.0).abs() < 1e-9, "weights sum to {total}");
        let before: Vec<_> = mix.components().collect();
        let after: Vec<_> = scaled.components().collect();
        assert_eq!(before.len(), after.len());
        for ((sh0, r0, lw0, sg0), (sh1, r1, lw1, sg1)) in before.iter().zip(after.iter()) {
            assert!((sh0 - sh1).abs() < 1e-12, "shape changed");
            assert!((r1 - r0 / s).abs() < 1e-12, "rate not divided by s");
            assert_eq!(sg0, sg1, "sign changed");
            assert!(
                (lw0.exp() - lw1.exp()).abs() < 1e-9,
                "mixing weight changed"
            );
        }
    }

    #[test]
    fn merge_of_agreeing_scaled_pins_sums_jacobians() {
        // `2·g = 4` and `3·g = 6` both pin g = 2; the merged pin carries both
        // Jacobians, −ln 2 − ln 3.
        let a = GammaSuffStat::real_eq_scaled(LAMBDA, 4.0, 2.0); // pin 2, −ln2
        let b = GammaSuffStat::real_eq_scaled(LAMBDA, 6.0, 3.0); // pin 2, −ln3
        let (m, f) = a.merge(&b);
        assert_eq!(f, 0.0);
        assert!((m.pin_log_coeff.into_inner() - (-(2.0_f64.ln()) - 3.0_f64.ln())).abs() < 1e-12);
        assert_eq!(m.pin, Some(nn(2.0)));
    }

    #[test]
    fn scaled_exponential_z_matches_quadrature() {
        // Cross-check the analytic Z (ε¹ coefficient) against trapezoid
        // quadrature of the scaled density.
        let (a, beta, v, s) = (2.5, 1.3, 0.6, 4.0);
        let obs = ExponentialObs::real_eq(X1, v).with_scale(nn(s));
        let stat = GammaSuffStat::exponential(LAMBDA, obs);
        let (_, z) = stat.posterior(&joint_prior(&stat, a, beta), &query());
        let (z_num, _) = brute_force_z_and_mean(&stat, a, beta);
        let z_ana = z.log_coeff.exp();
        assert!(
            (z_ana - z_num).abs() < 1e-4 * z_num.max(1.0),
            "analytic {z_ana} vs numeric {z_num}"
        );
    }

    #[test]
    fn scaled_poisson_z_matches_quadrature() {
        let (a, beta, m, s) = (3.0, 1.5, 2u64, 1.7);
        let obs = PoissonObs::eq(K1, m).with_scale(nn(s));
        let stat = GammaSuffStat::poisson(LAMBDA, obs);
        let (_, z) = stat.posterior(&joint_prior(&stat, a, beta), &query());
        let (z_num, _) = brute_force_z_and_mean(&stat, a, beta);
        let z_ana = z.log_coeff.exp();
        assert!(
            (z_ana - z_num).abs() < 1e-4 * z_num.max(1.0),
            "analytic {z_ana} vs numeric {z_num}"
        );
    }

    #[test]
    fn unbounded_range_gives_signed_mixture() {
        // Observe k ≥ 2: likelihood 1 − pois(0|λ) − pois(1|λ) ⇒ three
        // components, two negative; signed weights sum to 1.
        let stat = GammaSuffStat::poisson(LAMBDA, PoissonObs::geq(K1, 2));
        let (post, z) = stat.posterior(&joint_prior(&stat, 2.0, 1.0), &query());
        assert_eq!(z.power, 0);
        match post.expect("posterior should be Some").gamma {
            GammaPrior::Mixture { mixture, .. } => {
                assert_eq!(mixture.len(), 3);
                let negatives = mixture
                    .components()
                    .filter(|&(_, _, _, sign)| sign < 0)
                    .count();
                assert_eq!(negatives, 2);
                let total: f64 = mixture
                    .components()
                    .map(|(_, _, log_w, sign)| sign as f64 * log_w.exp())
                    .sum();
                assert!((total - 1.0).abs() < 1e-10, "weights sum to {}", total);
            }
            other => panic!("expected Mixture posterior, got {}", other),
        }
    }

    #[test]
    fn bounded_range_gives_positive_mixture() {
        // Observe k ∈ [1, 2]: two positive components Gamma(a+1), Gamma(a+2).
        let stat = poisson_between(K1, 1, 2);
        let (post, _) = stat.posterior(&joint_prior(&stat, 2.0, 1.0), &query());
        match post.expect("posterior should be Some").gamma {
            GammaPrior::Mixture { mixture, .. } => {
                assert_eq!(mixture.len(), 2);
                assert!(mixture.components().all(|(_, _, _, sign)| sign > 0));
                let total: f64 = mixture
                    .components()
                    .map(|(_, _, log_w, sign)| sign as f64 * log_w.exp())
                    .sum();
                assert!((total - 1.0).abs() < 1e-10);
            }
            other => panic!("expected Mixture posterior, got {}", other),
        }
    }

    #[test]
    fn exclusion_only_stat_leaves_prior_untouched() {
        // λ ≠ r is measure-zero under Gamma: Z = 1, posterior = prior.
        let stat = GammaSuffStat::not_eq(LAMBDA, 1.5);
        let prior = joint_prior(&stat, 2.0, 0.5);
        let (post, z) = stat.posterior(&prior, &query());
        assert_eq!(z.power, 0);
        assert!(z.log_coeff.abs() < 1e-12);
        assert_eq!(post.expect("posterior should be Some"), prior);
    }

    #[test]
    fn pin_posterior_is_pinned_with_density_marginal() {
        let (a, beta, r) = (2.0, 2.0, 1.3);
        let stat = GammaSuffStat::real_eq(LAMBDA, r);
        let (post, z) = stat.posterior(&joint_prior(&stat, a, beta), &query());
        assert_eq!(z.power, 1);
        assert!((z.log_coeff - log_gamma_pdf(r, a, beta)).abs() < 1e-12);
        match post.expect("posterior should be Some").gamma {
            GammaPrior::Pinned { value, .. } => assert!((value - r).abs() < 1e-12),
            other => panic!("expected Pinned posterior, got {}", other),
        }
    }

    #[test]
    fn pin_with_obs_scores_obs_at_pin() {
        // λ = r together with k = 2 on a draw: Z = gamma_pdf(r)·pois(2|r)·ε.
        let (a, beta, r) = (2.0, 2.0, 1.3);
        let stat = {
            let (m, f) = GammaSuffStat::real_eq(LAMBDA, r)
                .merge(&GammaSuffStat::poisson(LAMBDA, PoissonObs::eq(K1, 2)));
            assert_eq!(f, 0.0);
            m
        };
        let (post, z) = stat.posterior(&joint_prior(&stat, a, beta), &query());
        assert_eq!(z.power, 1);
        let expected = log_gamma_pdf(r, a, beta) + poisson_log_prob(&IntervalOrEq::eq(2), &nn(r));
        assert!((z.log_coeff - expected).abs() < 1e-12);
        assert!(matches!(
            post.expect("posterior should be Some").gamma,
            GammaPrior::Pinned { .. }
        ));
    }

    #[test]
    fn pin_with_exponential_dirac_gives_eps_squared() {
        // λ = r together with x = v: Z = gamma_pdf(r)·(r e^{−rv})·ε².
        // The ε² companion to `pin_with_obs_scores_obs_at_pin`: merge
        // defers the observation to `posterior` (factor 0, no Beta-style
        // discharge), which then scores it exactly once at the pin.
        let (a, beta, r, v) = (2.0, 2.0, 1.3, 0.7);
        let stat = {
            let (m, f) = GammaSuffStat::real_eq(LAMBDA, r).merge(&GammaSuffStat::exponential(
                LAMBDA,
                ExponentialObs::real_eq(X1, v),
            ));
            assert_eq!(f, 0.0);
            m
        };
        let (post, z) = stat.posterior(&joint_prior(&stat, a, beta), &query());
        assert_eq!(z.power, 2);
        let expected = log_gamma_pdf(r, a, beta) + (r.ln() - r * v);
        assert!((z.log_coeff - expected).abs() < 1e-12);
        assert!(matches!(
            post.expect("posterior should be Some").gamma,
            GammaPrior::Pinned { .. }
        ));
    }

    #[test]
    fn eps_power_counts_exponential_diracs() {
        let stat = GammaSuffStat::exponential(LAMBDA, ExponentialObs::real_eq(X1, 0.5))
            .merge(&GammaSuffStat::poisson(LAMBDA, PoissonObs::geq(K1, 1)))
            .0;
        let (_, z) = stat.posterior(&joint_prior(&stat, 2.0, 1.0), &query());
        assert_eq!(z.power, 1);
    }

    #[test]
    fn posterior_empty_and_disjoint_query_returns_none() {
        let stat = GammaSuffStat::poisson(LAMBDA, PoissonObs::eq(K1, 2));
        let prior = joint_prior(&stat, 2.0, 1.0);
        assert!(stat.posterior(&prior, &BTreeSet::new()).0.is_none());
        let other: BTreeSet<ContVarName> = [99u64].iter().copied().collect();
        assert!(stat.posterior(&prior, &other).0.is_none());
    }

    #[test]
    fn posterior_projects_to_queried_draws_keeping_rate() {
        // Querying only k1: the joint keeps λ (the mixing variable the
        // truncated draws are indexed by) plus k1's truncation; k2 and
        // x1 marginalize out by dropping. The marginal z is unchanged.
        let stat = GammaSuffStat::poisson(LAMBDA, PoissonObs::geq(K1, 2))
            .merge(&GammaSuffStat::poisson(LAMBDA, PoissonObs::eq(K2, 1)))
            .0
            .merge(&GammaSuffStat::exponential(
                LAMBDA,
                ExponentialObs::geq(X1, 0.5),
            ))
            .0;
        let prior = joint_prior(&stat, 2.0, 1.0);

        let full_q: BTreeSet<ContVarName> = [LAMBDA, K1, K2, X1].iter().copied().collect();
        let (full, z_full) = stat.posterior(&prior, &full_q);
        let full = full.expect("posterior should be Some");
        assert_eq!(full.poisson.scope().copied().collect_vec(), vec![K1, K2]);
        assert_eq!(full.exponential.scope().copied().collect_vec(), vec![X1]);

        let k1_only: BTreeSet<ContVarName> = [K1].iter().copied().collect();
        let (proj, z_proj) = stat.posterior(&prior, &k1_only);
        let proj = proj.expect("posterior should be Some");
        assert_eq!(proj.gamma, full.gamma);
        assert_eq!(z_proj.power, z_full.power);
        assert!((z_proj.log_coeff - z_full.log_coeff).abs() < 1e-12);
        assert_eq!(
            proj.poisson,
            PoissonPrior::conditioned_on(&PoissonObs::geq(K1, 2))
        );
        assert_eq!(proj.exponential.scope().count(), 0);
    }

    #[test]
    fn contradictory_merge_reports_neg_infinity_factor() {
        // The contradiction lives in the merge factor — the merged stat
        // keeps a valid constraint per draw. The SPN product consumes
        // the factor (`mul_leaf_into_leaf` collapses −∞ to Spn::zero()),
        // so the contradiction zeroes the weighted leaf, not the stat.
        let a = GammaSuffStat::poisson(LAMBDA, PoissonObs::leq(K1, 2));
        let b = GammaSuffStat::poisson(LAMBDA, PoissonObs::geq(K1, 3));
        let (_, f) = a.merge(&b);
        assert_eq!(f, f64::NEG_INFINITY);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn posterior_debug_asserts_name_mismatch() {
        let stat = GammaSuffStat::poisson(LAMBDA, PoissonObs::eq(K1, 2));
        let prior = joint_with(
            GammaPrior::Gamma {
                name: 99,
                shape: 2.0,
                rate: 1.0,
            },
            &stat,
        );
        let _ = stat.posterior(&prior, &query());
    }

    // -------------------- Z and posterior vs brute force --------------------

    #[test]
    fn marginal_and_mean_match_quadrature_mixed_observations() {
        // k1 ≥ 2, k2 = 1, x ∈ [0.5, 2.0]: exercises the complement trick,
        // factorials, exponential intervals and the (n, c) collapse at once.
        let (a, beta) = (2.0, 2.0);
        let stat = {
            let (m, _) = GammaSuffStat::poisson(LAMBDA, PoissonObs::geq(K1, 2))
                .merge(&GammaSuffStat::poisson(LAMBDA, PoissonObs::eq(K2, 1)));
            let (m, _) = m.merge(&GammaSuffStat::exponential(
                LAMBDA,
                ExponentialObs::geq(X1, 0.5),
            ));
            let (m, _) = m.merge(&GammaSuffStat::exponential(
                LAMBDA,
                ExponentialObs::lt(X1, 2.0),
            ));
            m
        };
        let (post, z) = stat.posterior(&joint_prior(&stat, a, beta), &query());
        let (z_num, mean_num) = brute_force_z_and_mean(&stat, a, beta);
        assert!(
            (z.log_coeff - z_num.ln()).abs() < 1e-4,
            "Z: closed form {} vs quadrature {}",
            z.log_coeff,
            z_num.ln()
        );
        let mean = posterior_mean(&post.expect("posterior should be Some").gamma);
        assert!(
            (mean - mean_num).abs() < 1e-3,
            "mean: closed form {} vs quadrature {}",
            mean,
            mean_num
        );
    }

    #[test]
    fn marginal_matches_quadrature_with_dirac() {
        // Exponential Dirac alongside a Poisson range observation.
        let (a, beta) = (3.0, 3.0);
        let stat = GammaSuffStat::exponential(LAMBDA, ExponentialObs::real_eq(X1, 0.8))
            .merge(&GammaSuffStat::poisson(LAMBDA, PoissonObs::leq(K1, 2)))
            .0;
        let (post, z) = stat.posterior(&joint_prior(&stat, a, beta), &query());
        assert_eq!(z.power, 1);
        let (z_num, mean_num) = brute_force_z_and_mean(&stat, a, beta);
        assert!((z.log_coeff - z_num.ln()).abs() < 1e-4);
        let mean = posterior_mean(&post.expect("posterior should be Some").gamma);
        assert!((mean - mean_num).abs() < 1e-3);
    }

    // -------------------- marginal likelihood at a fixed rate --------------------

    #[test]
    fn marginal_factorises_over_draws() {
        let stat = GammaSuffStat::poisson(LAMBDA, PoissonObs::eq(K1, 2))
            .merge(&GammaSuffStat::exponential(
                LAMBDA,
                ExponentialObs::geq(X1, 0.5),
            ))
            .0;
        let lambda = nn(1.7);
        let expected = poisson_log_prob(&IntervalOrEq::eq(2), &lambda)
            + ExponentialObs::geq(X1, 0.5)
                .marginal_log_likelihood_contribution(&lambda)
                .log_coeff;
        let m = stat.marginal_log_likelihood_contribution(&lambda);
        assert_eq!(m.power, 0);
        assert!((m.log_coeff - expected).abs() < 1e-12);
    }

    #[test]
    fn marginal_multiplies_powers_across_families() {
        // Poisson (power 0) × exponential Dirac (power 1): RealEps `*`
        // adds powers and log-coeffs — regression for the `+` (logsumexp)
        // bug, which would silently drop the higher-power factor.
        let stat = GammaSuffStat::poisson(LAMBDA, PoissonObs::eq(K1, 2))
            .merge(&GammaSuffStat::exponential(
                LAMBDA,
                ExponentialObs::real_eq(X1, 0.7),
            ))
            .0;
        let lambda = nn(1.3);
        let m = stat.marginal_log_likelihood_contribution(&lambda);
        assert_eq!(m.power, 1);
        let expected = poisson_log_prob(&IntervalOrEq::eq(2), &lambda) + (1.3_f64.ln() - 1.3 * 0.7);
        assert!((m.log_coeff - expected).abs() < 1e-12);
    }

    // -------------------- log_likelihood (indicators) --------------------

    #[test]
    fn log_likelihood_is_indicator_over_draws() {
        // The realization carries the draw values, so every observed
        // event is determined: 0 when all hold, −∞ on any violation.
        let stat = GammaSuffStat::poisson(LAMBDA, PoissonObs::eq(K1, 2))
            .merge(&GammaSuffStat::exponential(
                LAMBDA,
                ExponentialObs::geq(X1, 0.5),
            ))
            .0;
        let ok = realization(&[(LAMBDA, 1.7), (K1, 2.0), (X1, 0.9)]);
        assert_eq!(stat.log_likelihood(&ok), 0.0);
        let bad_poisson = realization(&[(LAMBDA, 1.7), (K1, 3.0), (X1, 0.9)]);
        assert_eq!(stat.log_likelihood(&bad_poisson), f64::NEG_INFINITY);
        let bad_exp = realization(&[(LAMBDA, 1.7), (K1, 2.0), (X1, 0.2)]);
        assert_eq!(stat.log_likelihood(&bad_exp), f64::NEG_INFINITY);
    }

    #[test]
    fn log_likelihood_pin_and_exclusion() {
        let pin = GammaSuffStat::real_eq(LAMBDA, 1.5);
        assert_eq!(pin.log_likelihood(&realization(&[(LAMBDA, 1.5)])), 0.0);
        assert_eq!(
            pin.log_likelihood(&realization(&[(LAMBDA, 1.6)])),
            f64::NEG_INFINITY
        );
        let excl = GammaSuffStat::not_eq(LAMBDA, 1.5);
        assert_eq!(
            excl.log_likelihood(&realization(&[(LAMBDA, 1.5)])),
            f64::NEG_INFINITY
        );
        assert_eq!(excl.log_likelihood(&realization(&[(LAMBDA, 1.6)])), 0.0);
    }

    // -------------------- registry round trip --------------------

    #[test]
    fn posterior_in_uses_registry_prior() {
        let mut reg = super::super::PriorRegistry::new();
        reg.add_gamma(gamma_prior(2.0, 0.5));
        let stat = GammaSuffStat::poisson(LAMBDA, PoissonObs::eq(K1, 3));
        let direct = stat.posterior(&joint_prior(&stat, 2.0, 0.5), &query());
        let via_reg = stat.posterior_in(&reg, &query());
        assert_eq!(direct.0, via_reg.0);
        assert!((direct.1.log_coeff - via_reg.1.log_coeff).abs() < 1e-12);
    }

    // -------------------- sampling --------------------

    use rand::SeedableRng;

    fn rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn sample_gamma_empirical_mean_matches_prior() {
        // Gamma(shape 2, rate 2) has mean shape/rate = 1.
        let p = gamma_prior(2.0, 2.0);
        let mut r = rng();
        let n = 5000;
        let mean = (0..n)
            .map(|_| p.sample_value(&mut r).into_inner())
            .sum::<f64>()
            / n as f64;
        assert!((mean - 1.0).abs() < 0.05, "empirical mean {:.4}", mean);
    }

    #[test]
    fn sample_pinned_returns_value() {
        let p = GammaPrior::Pinned {
            name: LAMBDA,
            value: 0.42,
        };
        let mut r = rng();
        for _ in 0..10 {
            assert_eq!(p.sample_value(&mut r).into_inner(), 0.42);
        }
    }

    #[test]
    fn sample_signed_mixture_empirical_mean_matches_analytic() {
        // Posterior of Gamma(2, rate 1) given k ≥ 2 — a genuinely signed mixture.
        let stat = GammaSuffStat::poisson(LAMBDA, PoissonObs::geq(K1, 2));
        let (post, _) = stat.posterior(&joint_prior(&stat, 2.0, 1.0), &query());
        let post = post.expect("posterior should be Some").gamma;
        assert!(matches!(post, GammaPrior::Mixture { .. }));
        let analytic = posterior_mean(&post);
        let mut r = rng();
        let n = 20_000;
        let mean = (0..n)
            .map(|_| post.sample_value(&mut r).into_inner())
            .sum::<f64>()
            / n as f64;
        assert!(
            (mean - analytic).abs() < 0.05,
            "empirical mean {:.4} vs analytic {:.4}",
            mean,
            analytic
        );
    }

    #[test]
    fn joint_sample_pushes_whole_block_into_assignment() {
        // Posterior sampling covers λ and every draw, respecting the
        // truncations; push_into splays the block into Assignment.gamma.
        let stat = GammaSuffStat::poisson(LAMBDA, PoissonObs::geq(K1, 2))
            .merge(&GammaSuffStat::exponential(
                LAMBDA,
                ExponentialObs::real_eq(X1, 0.7),
            ))
            .0;
        let q: BTreeSet<ContVarName> = [LAMBDA, K1, X1].iter().copied().collect();
        let (post, _) = stat.posterior(&joint_prior(&stat, 2.0, 1.0), &q);
        let post = post.expect("posterior should be Some");
        let mut r = rng();
        let sampled = post.sample(&mut r);
        assert_eq!(sampled.keys().copied().collect_vec(), vec![LAMBDA, K1, X1]);
        assert!(sampled[&K1].into_inner() >= 2.0);
        assert_eq!(sampled[&X1].into_inner(), 0.7);
        let mut out = super::super::AssignmentBuilder::new();
        post.push_into(sampled, &mut out);
        let a = out.finalize();
        assert_eq!(a.gamma.len(), 3);
        assert_eq!(a.gamma_value(X1).into_inner(), 0.7);
    }

    #[test]
    fn posterior_predictive_mean_matches_negative_binomial() {
        // Observe k1 = m; query the unconstrained draw k2 (vacuous
        // interval). Marginally k2 | data is the Gamma-Poisson (negative
        // binomial) predictive with mean E[λ | data] = (a+m)/(β+1).
        let (a, beta, m) = (2.0, 2.0, 3u64);
        let stat = GammaSuffStat::poisson(LAMBDA, PoissonObs::eq(K1, m))
            .merge(&GammaSuffStat::poisson(LAMBDA, PoissonObs::geq(K2, 0)))
            .0;
        let q: BTreeSet<ContVarName> = [K2].iter().copied().collect();
        let (post, _) = stat.posterior(&joint_prior(&stat, a, beta), &q);
        let post = post.expect("posterior should be Some");
        let mut r = rng();
        let n = 20_000;
        let mean = (0..n)
            .map(|_| post.sample(&mut r)[&K2].into_inner())
            .sum::<f64>()
            / n as f64;
        let expected = (a + m as f64) / (beta + 1.0);
        assert!(
            (mean - expected).abs() < 0.05,
            "empirical mean {:.4} vs analytic {:.4}",
            mean,
            expected
        );
    }

    // -------------------- prior equality / hashing --------------------

    #[test]
    fn prior_quantised_equality() {
        let g1 = gamma_prior(2.0, 0.5);
        let g2 = GammaPrior::Gamma {
            name: LAMBDA,
            shape: 2.0 + 1e-10,
            rate: 0.5 + 1e-10,
        };
        assert_eq!(g1, g2);
        let g3 = GammaPrior::Gamma {
            name: LAMBDA,
            shape: 3.0,
            rate: 0.5,
        };
        assert_ne!(g1, g3);
        let g4 = GammaPrior::Gamma {
            name: 99,
            shape: 2.0,
            rate: 0.5,
        };
        assert_ne!(g1, g4);
    }

    #[test]
    fn mixture_equality_is_componentwise() {
        let stat = GammaSuffStat::poisson(LAMBDA, PoissonObs::geq(K1, 2));
        let p1 = stat
            .posterior(&joint_prior(&stat, 2.0, 1.0), &query())
            .0
            .unwrap();
        let p2 = stat
            .posterior(&joint_prior(&stat, 2.0, 1.0), &query())
            .0
            .unwrap();
        assert_eq!(p1, p2);
        let p3 = stat
            .posterior(&joint_prior(&stat, 2.5, 1.0), &query())
            .0
            .unwrap();
        assert_ne!(p1, p3);
    }

    // -------------------- GammaFlip --------------------

    fn flip_assignment(entries: &[(ContVarName, f64)]) -> super::super::Assignment {
        super::super::Assignment::from_sorted(
            vec![],
            vec![],
            entries.iter().map(|(n, v)| (*n, nn(*v))).collect(),
            vec![],
        )
    }

    #[test]
    fn poisson_event_flip_edges_are_indicators() {
        let flip = GammaFlip::PoissonEvent {
            gamma: LAMBDA,
            draw: K1,
            event: IntervalOrEq::eq(2),
            scale: 1.0,
        };
        let (low, high) = flip.weights();
        let at = |k: f64| flip_assignment(&[(LAMBDA, 1.3), (K1, k)]);
        // draw = 2: the eq edge holds, the complement edge is impossible.
        assert_eq!(high.log_likelihood(&at(2.0)), 0.0);
        assert_eq!(low.log_likelihood(&at(2.0)), f64::NEG_INFINITY);
        // draw = 5: lands in the complement's upper piece.
        assert_eq!(high.log_likelihood(&at(5.0)), f64::NEG_INFINITY);
        assert_eq!(low.log_likelihood(&at(5.0)), 0.0);
        // Boundary checks pin the complement pieces to exactly
        // [0, 2) ∪ [3, ∞): no gap, no overlap with the point.
        assert_eq!(low.log_likelihood(&at(0.0)), 0.0);
        assert_eq!(low.log_likelihood(&at(1.0)), 0.0);
        assert_eq!(high.log_likelihood(&at(1.0)), f64::NEG_INFINITY);
        assert_eq!(low.log_likelihood(&at(3.0)), 0.0);
        assert_eq!(high.log_likelihood(&at(3.0)), f64::NEG_INFINITY);
        assert_eq!(flip.sample_probability(&at(2.0)), Some(1.0));
        assert_eq!(flip.sample_probability(&at(5.0)), Some(0.0));
    }

    #[test]
    fn poisson_range_flip_complement_partitions_at_boundary() {
        // k ≥ 2: complement is exactly [0, 2).
        let flip = GammaFlip::PoissonEvent {
            gamma: LAMBDA,
            draw: K1,
            event: IntervalOrEq::geq(2),
            scale: 1.0,
        };
        let (low, high) = flip.weights();
        let at = |k: f64| flip_assignment(&[(LAMBDA, 1.3), (K1, k)]);
        assert_eq!(high.log_likelihood(&at(2.0)), 0.0);
        assert_eq!(low.log_likelihood(&at(2.0)), f64::NEG_INFINITY);
        assert_eq!(high.log_likelihood(&at(1.0)), f64::NEG_INFINITY);
        assert_eq!(low.log_likelihood(&at(1.0)), 0.0);
    }

    #[test]
    fn exp_interval_flip_edges_are_indicators() {
        let flip = GammaFlip::ExpInterval {
            gamma: LAMBDA,
            draw: X1,
            interval: Interval::geq(nn(0.5)),
            scale: 1.0,
        };
        let (low, high) = flip.weights();
        let at = |x: f64| flip_assignment(&[(LAMBDA, 1.3), (X1, x)]);
        assert_eq!(high.log_likelihood(&at(0.9)), 0.0);
        assert_eq!(low.log_likelihood(&at(0.9)), f64::NEG_INFINITY);
        assert_eq!(high.log_likelihood(&at(0.2)), f64::NEG_INFINITY);
        assert_eq!(low.log_likelihood(&at(0.2)), 0.0);
        // The boundary belongs to the high side only ([0.5, ∞) vs [0, 0.5)).
        assert_eq!(high.log_likelihood(&at(0.5)), 0.0);
        assert_eq!(low.log_likelihood(&at(0.5)), f64::NEG_INFINITY);
        assert_eq!(flip.sample_probability(&at(0.9)), Some(1.0));
        assert_eq!(flip.sample_probability(&at(0.2)), Some(0.0));
    }

    #[test]
    fn exp_eq_flip_low_edge_excludes_the_point() {
        let flip = GammaFlip::ExpEq {
            gamma: LAMBDA,
            draw: X1,
            value: 0.5,
            scale: 1.0,
        };
        assert!(flip.is_pinned_observation());
        let (low, high) = flip.weights();
        let at = |x: f64| flip_assignment(&[(LAMBDA, 1.3), (X1, x)]);
        assert_eq!(high.log_likelihood(&at(0.5)), 0.0);
        assert_eq!(low.log_likelihood(&at(0.5)), f64::NEG_INFINITY);
        assert_eq!(low.log_likelihood(&at(0.7)), 0.0);
        assert_eq!(flip.sample_probability(&at(0.5)), None);
    }

    #[test]
    fn rate_pin_flip_is_pinned_observation() {
        let flip = GammaFlip::RatePin {
            name: LAMBDA,
            value: 1.5,
            scale: 1.0,
        };
        assert!(flip.is_pinned_observation());
        let (low, high) = flip.weights();
        let at = |l: f64| flip_assignment(&[(LAMBDA, l)]);
        assert_eq!(high.log_likelihood(&at(1.5)), 0.0);
        assert_eq!(low.log_likelihood(&at(1.5)), f64::NEG_INFINITY);
        assert_eq!(low.log_likelihood(&at(2.0)), 0.0);
    }

    #[test]
    fn flip_discriminators_distinguish_variants_and_params() {
        let d = |f: &GammaFlip| f.discriminator();
        let pin = GammaFlip::RatePin {
            name: LAMBDA,
            value: 1.5,
            scale: 1.0,
        };
        let pin2 = GammaFlip::RatePin {
            name: LAMBDA,
            value: 2.5,
            scale: 1.0,
        };
        let pois = GammaFlip::PoissonEvent {
            gamma: LAMBDA,
            draw: K1,
            event: IntervalOrEq::eq(2),
            scale: 1.0,
        };
        let pois2 = GammaFlip::PoissonEvent {
            gamma: LAMBDA,
            draw: K1,
            event: IntervalOrEq::lt(3),
            scale: 1.0,
        };
        // The same event at a different scale is a distinct flip.
        let pois_scaled = GammaFlip::PoissonEvent {
            gamma: LAMBDA,
            draw: K1,
            event: IntervalOrEq::eq(2),
            scale: 1.5,
        };
        assert_ne!(d(&pin), d(&pin2));
        assert_ne!(d(&pin), d(&pois));
        assert_ne!(d(&pois), d(&pois2));
        assert_ne!(d(&pois), d(&pois_scaled));
    }

    #[test]
    fn scaled_flip_weights_carry_scale_into_obs() {
        // An ExpEq flip at scale s builds high/low edges whose
        // ExponentialObs carry scale s, so the conjugate posterior over the
        // rate uses the scaled likelihood.
        let s = 5.0;
        let flip = GammaFlip::ExpEq {
            gamma: LAMBDA,
            draw: X1,
            value: 0.7,
            scale: s,
        };
        let (_, high) = flip.weights();
        // The high edge's leaf posterior against Gamma(a, β) must match the
        // scaled-Dirac update Gamma(a+1, β + s·v) (see
        // `scaled_exponential_dirac_obs_conjugate_update`).
        let (a, beta, v) = (2.0, 2.0, 0.7);
        let direct =
            GammaSuffStat::exponential(LAMBDA, ExponentialObs::real_eq(X1, v).with_scale(nn(s)));
        let (post, _) = direct.posterior(&joint_prior(&direct, a, beta), &query());
        match post.expect("Some").gamma {
            GammaPrior::Gamma { shape, rate, .. } => {
                assert!((shape - (a + 1.0)).abs() < 1e-12);
                assert!((rate - (beta + s * v)).abs() < 1e-12);
            }
            other => panic!("expected Gamma, got {other}"),
        }
        // Sanity: the flip's high edge is a non-empty SPN leaf.
        let _ = high;
    }
}
