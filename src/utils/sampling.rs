//! Stateless sampling primitives used by `Prior::sample` impls and by SPN
//! traversal code.
//!
//! Closed-form numeric utilities (logsumexp, log-Beta PDF, etc.) live in
//! `crate::utils::math`; this module holds the RNG-driven ones.

use rand::Rng;

/// Standard-normal draw via Box–Muller. `u1` is floored at `MIN_POSITIVE`
/// to avoid `ln(0)`.
pub fn sample_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Gamma(shape, 1) via Marsaglia–Tsang (for `shape >= 1`) plus the
/// Ahrens–Dieter boost `Gamma(a) = Gamma(a + 1) · U^(1/a)` for `shape < 1`.
pub fn sample_gamma<R: Rng>(rng: &mut R, shape: f64) -> f64 {
    if shape < 1.0 {
        let u = rng.gen::<f64>().max(f64::MIN_POSITIVE);
        return sample_gamma(rng, shape + 1.0) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let (x, v) = loop {
            let x = sample_normal(rng);
            let v = 1.0 + c * x;
            if v > 0.0 {
                break (x, v * v * v);
            }
        };
        let u = rng.gen::<f64>();
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Beta(a, b) via the `X / (X + Y)` trick on two Gamma draws.
pub fn sample_beta<R: Rng>(rng: &mut R, a: f64, b: f64) -> f64 {
    let x = sample_gamma(rng, a);
    let y = sample_gamma(rng, b);
    if x + y == 0.0 {
        0.5
    } else {
        x / (x + y)
    }
}

/// Dirichlet(alpha) via per-component Gamma draws followed by
/// normalisation. If all draws underflow to zero, fall back to a uniform
/// distribution over the components.
pub fn sample_dirichlet<R: Rng>(rng: &mut R, alpha: &[f64]) -> Vec<f64> {
    let mut samples: Vec<f64> = Vec::with_capacity(alpha.len());
    let mut sum = 0.0;
    for &a in alpha {
        let v = sample_gamma(rng, a);
        samples.push(v);
        sum += v;
    }
    if sum == 0.0 {
        let uniform = 1.0 / alpha.len() as f64;
        samples.iter_mut().for_each(|x| *x = uniform);
    } else {
        samples.iter_mut().for_each(|x| *x /= sum);
    }
    samples
}

/// Pick an index from `0..log_weights.len()` proportional to
/// `exp(log_weights[i])`. Numerically stable via the standard
/// `log_w - max_log_w` shift.
pub fn pick_categorical_log<R: Rng>(rng: &mut R, log_weights: &[f64]) -> usize {
    debug_assert!(!log_weights.is_empty());
    let max_lw = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let lin: Vec<f64> = log_weights.iter().map(|w| (w - max_lw).exp()).collect();
    let total: f64 = lin.iter().sum();
    let u = rng.gen::<f64>() * total;
    let mut cum = 0.0;
    for (i, &w) in lin.iter().enumerate() {
        cum += w;
        if u < cum {
            return i;
        }
    }
    log_weights.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn sample_normal_empirical_moments() {
        let mut r = rng();
        let n = 5000;
        let xs: Vec<f64> = (0..n).map(|_| sample_normal(&mut r)).collect();
        let mean: f64 = xs.iter().sum::<f64>() / n as f64;
        let var: f64 = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.05, "empirical mean {:.4}", mean);
        assert!((var - 1.0).abs() < 0.05, "empirical var {:.4}", var);
    }

    #[test]
    fn sample_gamma_shape_one_empirical_mean() {
        // Gamma(1, 1) has mean 1, var 1.
        let mut r = rng();
        let n = 5000;
        let total: f64 = (0..n).map(|_| sample_gamma(&mut r, 1.0)).sum();
        let mean = total / n as f64;
        assert!((mean - 1.0).abs() < 0.05, "empirical mean {:.4}", mean);
    }

    #[test]
    fn sample_beta_in_unit_interval() {
        let mut r = rng();
        for _ in 0..200 {
            let v = sample_beta(&mut r, 2.0, 5.0);
            assert!((0.0..=1.0).contains(&v), "Beta sample out of range: {}", v);
        }
    }

    #[test]
    fn sample_dirichlet_components_sum_to_one() {
        let mut r = rng();
        let v = sample_dirichlet(&mut r, &[2.0, 3.0, 5.0]);
        assert_eq!(v.len(), 3);
        let total: f64 = v.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "dirichlet sum = {}", total);
    }

    #[test]
    fn pick_categorical_log_matches_target_distribution() {
        // log_weights = [ln 0.7, ln 0.2, ln 0.1]; index 0 should win
        // about 70% of the time.
        let log_weights = [0.7_f64.ln(), 0.2_f64.ln(), 0.1_f64.ln()];
        let mut r = rng();
        let n = 10_000;
        let mut counts = [0u32; 3];
        for _ in 0..n {
            counts[pick_categorical_log(&mut r, &log_weights)] += 1;
        }
        let freq = |i: usize| counts[i] as f64 / n as f64;
        assert!((freq(0) - 0.7).abs() < 0.02, "freq[0] = {:.4}", freq(0));
        assert!((freq(1) - 0.2).abs() < 0.02, "freq[1] = {:.4}", freq(1));
        assert!((freq(2) - 0.1).abs() < 0.02, "freq[2] = {:.4}", freq(2));
    }
}
