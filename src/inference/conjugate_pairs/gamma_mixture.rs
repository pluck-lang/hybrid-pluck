/// Semiring of (possibly infinite) Gamma mixture densities
/// represented as collections of terms `sign · exp(log_w) · λ^n e^{−cλ}`.
///
/// Each term `λ^n e^{−cλ}` is an *unnormalised* Gamma kernel with shape
/// `n + 1` and rate `c`. A signed mixture of these is exactly what a Gamma
/// posterior under partial (range) observations looks like: complement
/// tricks such as `P(k ≥ lo | λ) = 1 − Σ_{k<lo} pois(k|λ)` subtract
/// probability mass, so some terms carry negative weights. The overall
/// density is non-negative and the (component) weights sum to 1.
use std::{
    collections::BTreeMap,
    fmt::{self, Display, Formatter},
    hash::{Hash, Hasher},
    ops::{Add, Mul},
};

use itertools::Itertools;
use ordered_float::NotNan;

use super::suffstat::PRIOR_QUANTIZATION_LEVEL;
use crate::utils::math::{ln_gamma, log_gamma_pdf, quant, signed_logsumexp};

/// Collections of terms `sign · exp(log_w) · λ^n e^{−cλ}`
/// with at most *one* occurence of each `λ^n e^{−cλ}` term.
// JSON has no notion of struct map keys, so serialize the mixture as a
// flat sequence of `(term, weight)` pairs rather than as its `BTreeMap`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(
    into = "Vec<(GammaTerm, SignedWeight)>",
    from = "Vec<(GammaTerm, SignedWeight)>"
)]
pub struct GammaMixture(pub BTreeMap<GammaTerm, SignedWeight>);

impl GammaMixture {
    /// The multiplicative identity: the single term `λ^0 e^{−0·λ}` with
    /// weight `+1` (likelihood ≡ 1).
    pub fn one() -> Self {
        GammaMixture(BTreeMap::from([(
            GammaTerm {
                n: NotNan::new(0.0).unwrap(),
                c: NotNan::new(0.0).unwrap(),
            },
            SignedWeight {
                log_w: 0.0,
                sign: 1,
            },
        )]))
    }

    /// The single (unnormalised) Gamma kernel `λ^{shape−1} e^{−rate·λ}`
    /// with weight `+1` — i.e. a `Gamma(shape, rate)` prior as a one-term
    /// mixture. Multiply by a likelihood mixture to obtain the posterior.
    pub fn gamma(shape: f64, rate: f64) -> Self {
        GammaMixture(BTreeMap::from([(
            GammaTerm {
                n: NotNan::new(shape - 1.0).unwrap(),
                c: NotNan::new(rate).unwrap(),
            },
            SignedWeight {
                log_w: 0.0,
                sign: 1,
            },
        )]))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Each term viewed as a *normalised* Gamma component
    /// `(shape = n+1, rate = c, log_weight, sign)`, where `log_weight` is
    /// the log-magnitude of the mixing weight:
    /// `log_w + lnΓ(shape) − shape·ln(rate)` (the term weight times the
    /// kernel's mass `Γ(n+1)/c^{n+1}`).
    pub(super) fn components(&self) -> impl Iterator<Item = (f64, f64, f64, i8)> + '_ {
        self.0.iter().map(|(t, w)| {
            let shape = t.n + 1.0;
            let rate = t.c.into_inner();
            let log_weight = w.log_w + ln_gamma(shape) - (shape) * rate.ln();
            (shape, rate, log_weight, w.sign)
        })
    }

    /// Rescale the term weights so the signed component weights sum to `+1`.
    pub fn normalize(&mut self) {
        let (log_norm, _) =
            signed_logsumexp(self.components().map(|(_, _, log_w, sign)| (log_w, sign)));
        for w in self.0.values_mut() {
            w.log_w -= log_norm;
        }
    }

    /// Re-scale the *rate* variable by a positive constant `s`: substitute
    /// `λ → s·λ` in every term. A kernel `w·λⁿ e^{−cλ}` becomes
    /// `(w·sⁿ)·λⁿ e^{−(s·c)λ}`, i.e. `(n, c) → (n, s·c)` and
    /// `log_w += n·ln s`. This turns a unit-rate likelihood mixture into the
    /// likelihood for a draw whose effective rate is `s·λ` (a scaled gamma).
    ///
    /// Scaling `c` by `s > 0` is injective and leaves `n` fixed, so the term
    /// keys stay distinct — no merging is needed. `n = 0` terms (constants,
    /// Poisson `≥` complement) are invariant, as expected.
    pub fn scale_rate(self, s: f64) -> GammaMixture {
        debug_assert!(s > 0.0, "scale_rate: scale must be positive, got {s}");
        let ln_s = s.ln();
        let scaled = self.0.into_iter().map(|(t, w)| {
            let n = t.n.into_inner();
            (
                GammaTerm {
                    n: t.n,
                    c: NotNan::new(t.c.into_inner() * s).unwrap(),
                },
                SignedWeight {
                    log_w: w.log_w + n * ln_s,
                    sign: w.sign,
                },
            )
        });
        GammaMixture(scaled.collect())
    }

    /// If the mixture is a single positive Gamma kernel, return its
    /// `(shape, rate)` — the posterior collapses to a plain `Gamma`.
    pub fn as_single_gamma(&self) -> Option<(f64, f64)> {
        let mut it = self.0.iter();
        let (t, w) = it.next()?;
        if it.next().is_none() && w.sign > 0 {
            Some((t.n + 1.0, t.c.into_inner()))
        } else {
            None
        }
    }

    /// Rejection-sample from the signed mixture `f = Σ sⱼ wⱼ gⱼ`.
    ///
    /// Proposal: the positive sub-mixture `g⁺ ∝ Σ_{sⱼ>0} wⱼ gⱼ`. Negative
    /// components only remove mass, so `f(x) ≤ Σ₊ wⱼ gⱼ(x)` pointwise and
    /// accepting with probability `f(x) / Σ₊ wⱼ gⱼ(x)` is exact.
    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> f64 {
        use crate::utils::sampling::{pick_categorical_log, sample_gamma};
        let comps: Vec<(f64, f64, f64, i8)> = self.components().collect();
        let positive: Vec<(f64, f64, f64, i8)> =
            comps.iter().copied().filter(|c| c.3 > 0).collect();
        assert!(
            !positive.is_empty(),
            "GammaMixture::sample: mixture has no positive components"
        );
        let log_weights: Vec<f64> = positive.iter().map(|c| c.2).collect();
        let all_positive = positive.len() == comps.len();
        const MAX_REJECTS: usize = 100_000;
        // `log_at(comp) = log[ |weight| · Gamma(shape, rate)(x) ]`.
        for _ in 0..MAX_REJECTS {
            let (shape, rate, _, _) = positive[pick_categorical_log(rng, &log_weights)];
            let x = sample_gamma(rng, shape) / rate;
            if all_positive {
                return x;
            }
            let log_at = |c: &(f64, f64, f64, i8)| c.2 + log_gamma_pdf(x, c.0, c.1);
            let (log_f, sign) = signed_logsumexp(comps.iter().map(|c| (log_at(c), c.3)));
            if sign <= 0 {
                continue;
            }
            let (log_envelope, _) = signed_logsumexp(positive.iter().map(|c| (log_at(c), 1)));
            if rng.gen::<f64>() < (log_f - log_envelope).exp() {
                return x;
            }
        }
        panic!("GammaMixture::sample: no acceptance after {MAX_REJECTS} proposals");
    }
}

impl Display for GammaMixture {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let body = self
            .components()
            .map(|(shape, rate, log_w, sign)| {
                format!(
                    "{:+.4}·Gamma({:.4}, {:.4})",
                    sign as f64 * log_w.exp(),
                    shape,
                    rate
                )
            })
            .join(" ");
        write!(f, "{}", body)
    }
}

impl Hash for GammaMixture {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (t, w) in &self.0 {
            t.n.hash(state);
            quant(t.c.into_inner(), PRIOR_QUANTIZATION_LEVEL).hash(state);
            quant(w.log_w, PRIOR_QUANTIZATION_LEVEL).hash(state);
            w.sign.hash(state);
        }
    }
}

impl PartialEq for GammaMixture {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(other.0.iter())
                .all(|((t1, w1), (t2, w2))| {
                    t1.n == t2.n
                        && quant(t1.c.into_inner(), PRIOR_QUANTIZATION_LEVEL)
                            == quant(t2.c.into_inner(), PRIOR_QUANTIZATION_LEVEL)
                        && quant(w1.log_w, PRIOR_QUANTIZATION_LEVEL)
                            == quant(w2.log_w, PRIOR_QUANTIZATION_LEVEL)
                        && w1.sign == w2.sign
                })
    }
}

impl Eq for GammaMixture {}

/// One term `λ^n e^{−cλ}` (an unnormalised `Gamma(n+1, c)` kernel). The
/// power `n` is real-valued: likelihood expansions contribute integer
/// powers, but folding a `Gamma(shape, rate)` prior shifts them by
/// `shape − 1`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct GammaTerm {
    pub(super) n: NotNan<f64>,
    pub(super) c: NotNan<f64>,
}

impl GammaTerm {
    /// Shape exponent.
    pub fn n(&self) -> f64 {
        self.n.into_inner()
    }
    pub fn c(&self) -> f64 {
        self.c.into_inner()
    }
}

impl Add for GammaTerm {
    type Output = GammaTerm;
    fn add(self, rhs: Self) -> Self::Output {
        GammaTerm {
            n: self.n + rhs.n,
            c: self.c + rhs.c,
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct SignedWeight {
    pub(super) log_w: f64,
    pub(super) sign: i8,
}

impl SignedWeight {
    pub fn log_w(&self) -> f64 {
        self.log_w
    }
    pub fn sign(&self) -> i8 {
        self.sign
    }
}

impl Mul for SignedWeight {
    type Output = SignedWeight;
    fn mul(self, rhs: Self) -> Self::Output {
        SignedWeight {
            log_w: self.log_w + rhs.log_w,
            sign: self.sign * rhs.sign,
        }
    }
}

impl Add for SignedWeight {
    type Output = SignedWeight;
    fn add(self, rhs: Self) -> Self::Output {
        let (log_w, sign) = signed_logsumexp([(self.log_w, self.sign), (rhs.log_w, rhs.sign)]);
        SignedWeight { log_w, sign }
    }
}

impl Mul for GammaMixture {
    type Output = GammaMixture;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut new: GammaMixture = GammaMixture(BTreeMap::new());
        for (t_term, t_weight) in self.0 {
            for (s_term, s_weight) in &rhs.0 {
                let term = t_term + *s_term;
                let weight = match new.0.get(&term) {
                    Some(&existing) => t_weight * *s_weight + existing,
                    None => t_weight * *s_weight,
                };
                new.0.insert(term, weight);
            }
        }
        new
    }
}

impl From<GammaMixture> for Vec<(GammaTerm, SignedWeight)> {
    fn from(m: GammaMixture) -> Self {
        m.0.into_iter().collect()
    }
}

impl From<Vec<(GammaTerm, SignedWeight)>> for GammaMixture {
    fn from(terms: Vec<(GammaTerm, SignedWeight)>) -> Self {
        GammaMixture(terms.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(n: f64, c: f64, log_w: f64, sign: i8) -> (GammaTerm, SignedWeight) {
        (
            GammaTerm {
                n: NotNan::new(n).unwrap(),
                c: NotNan::new(c).unwrap(),
            },
            SignedWeight { log_w, sign },
        )
    }

    #[test]
    fn scale_rate_maps_term_correctly() {
        // (n, c, log_w) → (n, s·c, log_w + n·ln s); sign preserved.
        let s = 5.0;
        let m = GammaMixture(BTreeMap::from([term(3.0, 2.0, 0.5, 1)]));
        let scaled = m.scale_rate(s);
        let (t, w) = scaled.0.iter().next().unwrap();
        assert_eq!(t.n.into_inner(), 3.0);
        assert!((t.c.into_inner() - 2.0 * s).abs() < 1e-12);
        assert!((w.log_w - (0.5 + 3.0 * s.ln())).abs() < 1e-12);
        assert_eq!(w.sign, 1);
    }

    #[test]
    fn scale_rate_leaves_constant_term_invariant() {
        // An n = 0 term (constant / Poisson `≥` complement) is invariant
        // under λ → s·λ: c stays 0 and log_w is unchanged (0·ln s = 0).
        let scaled = GammaMixture::one().scale_rate(7.0);
        let (t, w) = scaled.0.iter().next().unwrap();
        assert_eq!(t.n.into_inner(), 0.0);
        assert_eq!(t.c.into_inner(), 0.0);
        assert_eq!(w.log_w, 0.0);
    }

    #[test]
    fn scale_rate_by_one_is_identity() {
        let m = GammaMixture(BTreeMap::from([term(2.0, 1.5, 0.3, -1)]));
        assert_eq!(m.clone().scale_rate(1.0), m);
    }
}
