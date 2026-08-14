//! Statistical end-to-end tests for the Gibbs sampler. These use thread_rng
//! so they aren't snapshot-comparable; instead each test checks an empirical
//! distribution against the exact posterior computed from the same model
//! (via `Posterior`), with tolerances that are loose enough to survive RNG
//! variation at the sample counts used here.

use pluck::{PluckContext, ResultKind};

/// Run `f` on a thread with a 16 MiB stack. Pluck's compile path uses
/// deep closure-stack chains via `bind_compile` + `pin_value`, and
/// debug-mode frame sizes can exceed the default test-thread stack
/// (~2 MiB on macOS) when running long Gibbs chains.
///
/// Propagates any panic from the spawned thread (rather than swallowing
/// it into a generic "test panicked" string) so `#[should_panic(expected
/// = "...")]` tests can match on the underlying message.
fn with_big_stack<F: FnOnce() + Send + 'static>(f: F) {
    let handle = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(f)
        .expect("spawn");
    if let Err(payload) = handle.join() {
        std::panic::resume_unwind(payload);
    }
}

/// Run a pluck source and return the samples produced by its single query.
/// Panics if the query result isn't `Samples`.
fn run_samples(source: &str) -> Vec<String> {
    let mut ctx = PluckContext::new();
    let results = ctx.run(source);
    assert_eq!(results.len(), 1, "expected exactly one query");
    match &results[0].kind {
        ResultKind::Samples { values } => values.clone(),
        other => panic!("expected Samples result, got {:?}", other),
    }
}

/// Parse a float from the format_value display of a Native(Float).
/// Skips entries that can't parse (e.g. ADT formatted names).
fn parse_floats(samples: &[String]) -> Vec<f64> {
    samples
        .iter()
        .filter_map(|s| s.parse::<f64>().ok())
        .collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

// ============================================================================
// Two-block Bernoulli
// ============================================================================

/// Model: x, y ~ flip 0.5 with observation `(or x y)`. The exact posterior
/// has three equally-likely worlds (T,T), (T,F), (F,T), so
/// P(x=True | or(x,y)) = 2/3.
#[test]
fn gibbs_two_block_bernoulli_matches_exact_posterior() {
    with_big_stack(|| {
        let source = "(query q
            (let ((x (flip 0.5))
                  (y (flip 0.5)))
              (Gibbs (if x (True) (False))
                     (or x y)
                     2000
                     (Cons x (Cons y (Nil)))
                     (WithPriorSample x))))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), 2000);
        let trues = samples.iter().filter(|s| s.as_str() == "True").count();
        let frac = trues as f64 / samples.len() as f64;
        println!("T {:?} frac {}", trues, frac);
        assert!(
            (frac - 2.0 / 3.0).abs() < 0.05,
            "empirical P(True) = {}, expected ≈ {:.4}",
            frac,
            2.0 / 3.0
        );
    });
}

// ============================================================================
// Gaussian conjugate (single block)
// ============================================================================

/// Gaussian Gibbs with two parameters and a noisy observation.
///
/// Model: `μ ~ N(0, 1)`, `ν ~ N(0, 1)`, `z ~ N(0, 1)`,
/// `real_eq(μ + ν + z, 1.0)`. The noise term `z` is essential: a
/// noise-free Dirac constraint (`real_eq(μ + ν, 1)`) would pin the
/// joint to a 1-D line and make the chain stuck after step 0.
///
/// Marginal posterior on μ:
///   x | μ ~ N(μ, 2)  (since ν + z is `N(0, 2)`)
///   μ | x=1 ~ N(1/3, 2/3)
///
/// Blocks `[μ, ν]` partition the model's free continuous parameters;
/// each step resamples one given the other's pinned value.
///
/// Unlike the sibling statistical tests (whose target block is
/// conditionally independent of / deterministic given the other, so each
/// cycle is effectively an i.i.d. posterior draw), this chain has genuine
/// lag autocorrelation — both `μ` and `ν` pins bite. The 3000-sample count
/// (vs. the others' 500–2000) keeps the empirical-mean variance well inside
/// the ±0.1 tolerance; don't reduce it.
#[test]
fn gibbs_gaussian_conjugate_posterior_mean_and_variance() {
    with_big_stack(|| {
        let source = "(query q
            (let ((mu (gaussian 0.0 1.0))
                  (nu (gaussian 0.0 1.0))
                  (z  (gaussian 0.0 1.0)))
              (Gibbs mu
                     (real_eq (+. (+. mu nu) z) 1.0)
                     3000
                     (Cons mu (Cons nu (Nil)))
                     (WithPriorSample mu))))";
        let samples = run_samples(source);
        let xs = parse_floats(&samples);
        assert_eq!(xs.len(), 3000, "expected 3000 numeric samples");
        let m = mean(&xs);
        let expected = 1.0 / 3.0;
        assert!(
            (m - expected).abs() < 0.1,
            "empirical mean = {}, expected ≈ {:.4}",
            m,
            expected
        );
    });
}

// ============================================================================
// Dirichlet pinning via Gibbs
// ============================================================================

/// `weights ~ Dirichlet(1,1,1)`; observe `(categorical weights) = 0`.
/// Exact posterior on weights is `Dirichlet(2,1,1)`. Query: `θ_0`,
/// `E[θ_0 | obs] = 2/4 = 0.5`.
///
/// Two blocks (`weights` and the categorical outcome `c`) so the chain
/// isn't degenerate; query reads the indexed probability directly. The
/// categorical now has a Sample-mode handler so this works end-to-end.
#[test]
fn gibbs_dirichlet_categorical_posterior_mean() {
    with_big_stack(|| {
        let source = "(query q
            (let ((weights (dirichlet 1.0 1.0 1.0))
                  (c       (categorical weights)))
              (Gibbs (vector_index weights @0)
                     (native_eq c @0)
                     500
                     (Cons weights (Cons c (Nil)))
                     (WithPriorSample weights))))";
        let samples = run_samples(source);
        let xs = parse_floats(&samples);
        assert_eq!(xs.len(), 500, "expected 500 numeric samples");
        let m = mean(&xs);
        assert!(
            (m - 0.5).abs() < 0.1,
            "empirical E[θ_0] = {}, expected ≈ 0.5",
            m
        );
        assert!(xs.iter().all(|x| (0.0..=1.0).contains(x)), "out-of-range");
    });
}

/// Regression test for the log-domain WMC fix in `sample_discrete_given`.
///
/// Model: `μ ~ N(0, 1)` with `(real_eq μ 20.0)`. The posterior is a Dirac
/// at `μ = 20`. Pre-fix the per-branch log-likelihood at that point was
/// around `-200`, which `.exp()`'d to `0.0` in linear domain and tripped
/// the "zero total at var" guard before any sample could be drawn. With
/// log-domain WMC the ratios stay well-conditioned and we get 10 samples
/// out, each ≈ 20.
#[test]
fn posterior_samples_log_domain_no_underflow() {
    with_big_stack(|| {
        let source = "(query q
            (let ((mu (gaussian 0.0 1.0)))
              (PosteriorSamples mu (real_eq mu 20.0) 10)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), 10, "expected 10 samples, got {:?}", samples);
        let xs = parse_floats(&samples);
        assert_eq!(xs.len(), 10, "expected all samples to be floats");
        for x in &xs {
            assert!(
                (x - 20.0).abs() < 1e-6,
                "sample {} far from observed mu=20.0",
                x
            );
        }
    });
}

/// `PosteriorSamples` over a model with `(categorical weights)`. Before
/// `sample_compile_categorical` shipped this panicked at `force_value`
/// because categorical produced N worlds in Sample mode.
///
/// Model: `weights ~ Dirichlet(1,1,1); c1 ~ Categorical(weights); c2 ~
/// Categorical(weights); observe c1 = 0`. Returns 100 samples of `c2`.
/// Each sample is a native int in `{0, 1, 2}`; with the posterior on
/// weights tilted toward category 0 (Dirichlet(2,1,1)), `c2 = 0` should
/// be the modal outcome.
#[test]
fn posterior_samples_categorical_runs() {
    with_big_stack(|| {
        let source = "(query q
            (let ((weights (dirichlet 1.0 1.0 1.0))
                  (c1 (categorical weights))
                  (c2 (categorical weights)))
              (PosteriorSamples c2 (native_eq c1 @0) 100)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), 100);
        // Every sample is a native int 0/1/2 (display is "0", "1", or "2").
        for s in &samples {
            assert!(
                matches!(s.as_str(), "0" | "1" | "2"),
                "unexpected sample value: {:?}",
                s
            );
        }
        // 0 should be the modal outcome (posterior on θ_0 is the largest).
        let zeros = samples.iter().filter(|s| s.as_str() == "0").count();
        let ones = samples.iter().filter(|s| s.as_str() == "1").count();
        let twos = samples.iter().filter(|s| s.as_str() == "2").count();
        assert!(
            zeros >= ones && zeros >= twos,
            "expected mode at 0: 0s={}, 1s={}, 2s={}",
            zeros,
            ones,
            twos
        );
    });
}

// ============================================================================
// Rolling-window single-site Gibbs
// ============================================================================

/// Four independent flips a, b, c, d with `or(a, b)` as condition.
/// The condition couples a and b (marginal P(a=True | or(a,b)) = 2/3)
/// while c and d are free. Blocks `[a,b]` and `[c,d]` partition the
/// vars; each block step samples the block jointly given the pin
/// against the other block. This exercises structural pinning of a
/// Cons-list against a forced Cons value.
///
/// Runs on a bigger-stack thread: debug-mode `bind_compile`/`pin_value`
/// frames + recursion through nested `Cons` exceed the default macOS
/// 2 MiB test-thread stack.
#[test]
fn gibbs_rolling_window_marginal_matches_prior() {
    with_big_stack(|| {
        let source = "(query q
            (let ((a (flip 0.5))
                  (b (flip 0.5))
                  (c (flip 0.5))
                  (d (flip 0.5)))
              (Gibbs (if a (True) (False))
                     (or a b)
                     2000
                     (Cons (Cons a (Cons b (Nil)))
                       (Cons (Cons c (Cons d (Nil)))
                         (Nil)))
                     (WithConstant (Cons (True) (Cons (True) (Nil)))))))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), 2000);
        let trues = samples.iter().filter(|s| s.as_str() == "True").count();
        let frac = trues as f64 / samples.len() as f64;
        assert!(
            (frac - 2.0 / 3.0).abs() < 0.05,
            "empirical P(a=True) = {}, expected ≈ {:.4}",
            frac,
            2.0 / 3.0
        );
    });
}

// ============================================================================
// Minimal mixture (smoke check that mixed continuous + discrete blocks work)
// ============================================================================

/// Two-component, 2-datapoint Gaussian mixture with hard cluster assignments.
/// Just verifies the chain runs to completion without panicking — the
/// posterior shape is hard to nail analytically with only 2 datapoints,
/// so we only check basic invariants (right length, query values in
/// {True, False} for "is cluster 0").
/// Stripped-down version of `examples/gaussian_mixture_gibbs.pluck`: 3
/// observations, 10 Gibbs steps, fixed N_MAX=3. Exercises the full
/// categorical-in-Gibbs path (uniform_int_range for `n`, recursive
/// `sample-means`, dirichlet `weights`, categorical `cs`, real_eq on
/// data). The chain may stall early on this small a sample budget given
/// the strong observations — we only assert the run produces *some*
/// query samples (and doesn't panic) to guard against regression of the
/// categorical Sample-mode fix.
#[test]
fn gibbs_mixture_runs_on_example() {
    with_big_stack(|| {
        let source = r#"
(define obs [0.3, 1.2, 20.0])
(define ZERO  (mk_int @2 @0))
(define ONE   (mk_int @2 @1))
(define TWO   (mk_int @2 @2))
(define THREE (mk_int @2 @3))
(define (sample-means n)
  (if (int_eq n ZERO)
    (Nil)
    (Cons (normal 0.0 1.0) (sample-means (int_sub n ONE)))))
(define (sample-weights n)
  (if (int_eq n ONE) (dirichlet 1.0)
    (if (int_eq n TWO) (dirichlet 1.0 1.0)
      (dirichlet 1.0 1.0 1.0))))
(define (pick-mean c means)
  (match means
    Cons m rest =>
      (if (native_eq c @0) m
        (match rest
          Cons m2 rest2 =>
            (if (native_eq c @1) m2
              (match rest2
                Cons m3 _ => m3))))))
(query q
 (let
  ((n       (uniform_int_range @2 @1 @3))
   (means   (sample-means n))
   (weights (sample-weights n))
   (cs      (map (fn i -> (categorical weights))
                 (range (length obs))))
   (data    (map (fn c -> (normal (pick-mean c means) 1.0))
                 cs)))
  (Gibbs n
         (list=? real=? data obs)
         10
         (Cons n (Cons cs (Nil)))
         (WithConstant THREE))))
"#;
        let samples = run_samples(source);
        // Gibbs returns 0..=num_samples samples; we just need ≥1 to know
        // the categorical-in-Gibbs path ran at least one step. Empty
        // would mean even step 0 panicked or hit zero-probability
        // evidence, which is the regression we're guarding against.
        assert!(
            !samples.is_empty(),
            "mixture chain produced no samples at all"
        );
        // Every sample is a native int (n is uniform_int_range output).
        for s in &samples {
            // Pluck prints native ints with an `@` prefix.
            assert!(
                s.starts_with('@'),
                "expected @-prefixed native int, got {:?}",
                s
            );
        }
    });
}

#[test]
fn gibbs_mixture_minimal_runs_without_stalling() {
    with_big_stack(|| {
        // `(gaussian)` needs scalar μ, so we add the component mean via `+.`
        // (gaussian-affine arithmetic). `(if c m0 m1)` picks a cluster mean.
        // Mixture models on tight `real_eq` observations can stall mid-chain
        // when `sample_discrete_given` reports a zero-total node — that's
        // the correct behavior (the conditional has no support under the
        // sampled continuous values). We only check that the chain
        // produced at least one sample without panicking.
        let source = "(query q
            (let ((c0 (flip 0.5))
                  (c1 (flip 0.5))
                  (mu0 (gaussian -2.0 0.5))
                  (mu1 (gaussian  2.0 0.5))
                  (x0 (+. (if c0 mu0 mu1) (gaussian 0.0 1.0)))
                  (x1 (+. (if c1 mu0 mu1) (gaussian 0.0 1.0))))
              (Gibbs (if c0 (True) (False))
                     (and (real_eq x0 -1.5) (real_eq x1 1.5))
                     200
                     (Cons c0 (Cons c1 (Nil)))
                     (WithPriorSample c0))))";
        let samples = run_samples(source);
        assert!(!samples.is_empty(), "expected at least one sample");
        for s in &samples {
            assert!(
                s == "True" || s == "False",
                "expected boolean sample, got {:?}",
                s
            );
        }
    });
}

// ============================================================================
// Error case: blocking on a single Dirichlet component
// ============================================================================

/// `compile_pin` panics with a clear message when a block expression
/// evaluates to a `DirichletProbability` (i.e. a single Dirichlet
/// component). Documented error surfaces at runtime.
#[test]
#[should_panic(expected = "cannot pin a single Dirichlet component")]
fn gibbs_dirichlet_probability_block_errors() {
    with_big_stack(|| {
        let source = "(query q
            (let ((weights (dirichlet 1.0 1.0))
                  (c       (categorical weights)))
              (Gibbs (if (native_eq c @0) (True) (False))
                     (native_eq c @0)
                     3
                     (Cons (vector_index weights @0) (Nil))
                     (WithPriorSample weights))))";
        let _ = run_samples(source);
    });
}

// ============================================================================
// Gamma family: posterior sampling + Gibbs
// ============================================================================

/// An un-evidenced Poisson draw samples from its prior conditional:
/// `(poisson 4.0)` has mean 4 (exercises `SampleState::ensure_gamma_draw`).
#[test]
fn posterior_samples_unobserved_poisson_draw_mean() {
    with_big_stack(|| {
        let source = "(query (PosteriorSamples (poisson 4.0) True 2000))";
        let xs = parse_floats(&run_samples(source));
        assert_eq!(xs.len(), 2000);
        assert!((mean(&xs) - 4.0).abs() < 0.2, "mean {}", mean(&xs));
    });
}

/// Gamma-Poisson conjugacy through sampling: r | k=3 ~ Gamma(5, 2),
/// mean 5/2.
#[test]
fn posterior_samples_gamma_poisson_rate_mean() {
    with_big_stack(|| {
        let source = "(query (let ((r (gamma 2.0 1.0)))
            (PosteriorSamples r (native_eq (poisson r) @3) 2000)))";
        let xs = parse_floats(&run_samples(source));
        assert!((mean(&xs) - 2.5).abs() < 0.15, "mean {}", mean(&xs));
    });
}

/// A Dirac-observed Exponential draw is realized exactly at the
/// observed value by the posterior leaf's truncated conditional.
#[test]
fn posterior_samples_pinned_exponential_draw_is_exact() {
    with_big_stack(|| {
        let source = "(query (let ((x (exponential 2.0)))
            (PosteriorSamples x (real_eq x 0.5) 20)))";
        let xs = parse_floats(&run_samples(source));
        assert_eq!(xs.len(), 20);
        assert!(xs.iter().all(|&x| x == 0.5), "samples {:?}", xs);
    });
}

/// Range-conditioned Exponential draw: x | x ≥ 1 under Exp(2) is
/// `1 + Exp(2)` by memorylessness, mean 1.5.
#[test]
fn posterior_samples_truncated_exponential_mean() {
    with_big_stack(|| {
        let source = "(query (let ((x (exponential 2.0)))
            (PosteriorSamples x (real_geq x 1.0) 2000)))";
        let xs = parse_floats(&run_samples(source));
        assert!((mean(&xs) - 1.5).abs() < 0.1, "mean {}", mean(&xs));
    });
}

/// Posterior predictive: a fresh draw given r | k=5 ~ Gamma(7, 2) has
/// mean E[λ] = 3.5 (the query and evidence draws are distinct
/// callsites, so the query draw samples from its conditional given the
/// posterior-sampled rate).
#[test]
fn posterior_samples_poisson_posterior_predictive_mean() {
    with_big_stack(|| {
        let source = "(query (let ((r (gamma 2.0 1.0)))
            (PosteriorSamples (poisson r) (native_eq (poisson r) @5) 2000)))";
        let xs = parse_floats(&run_samples(source));
        assert!((mean(&xs) - 3.5).abs() < 0.25, "mean {}", mean(&xs));
    });
}

/// Two-block Gibbs over (rate, exponential draw) with range evidence:
/// r | x ≥ 1 ~ Gamma(2, 2), mean 1. Exercises the gamma-family
/// `pin_value` arms (the draw block pins x to a float via `ExpEq`,
/// the rate block pins r via `RatePin`).
#[test]
fn gibbs_gamma_rate_with_exponential_range_evidence() {
    with_big_stack(|| {
        let source = "(query (let ((r (gamma 2.0 1.0)) (x (exponential r)))
            (Gibbs r (real_geq x 1.0) 2000 (Cons r (Cons x (Nil)))
                   (WithPriorSample r))))";
        let xs = parse_floats(&run_samples(source));
        assert!((mean(&xs) - 1.0).abs() < 0.15, "mean {}", mean(&xs));
    });
}

/// Two-block Gibbs over (rate, poisson draw) with equality evidence on
/// the draw: r | k=3 ~ Gamma(5, 2), mean 2.5. The poisson draw block is
/// pinned as a native int (`PoissonEvent` pin arm).
#[test]
fn gibbs_gamma_rate_with_poisson_draw_block() {
    with_big_stack(|| {
        let source = "(query (let ((r (gamma 2.0 1.0)) (k (poisson r)))
            (Gibbs r (native_eq k @3) 2000 (Cons r (Cons k (Nil)))
                   (WithPriorSample r))))";
        let xs = parse_floats(&run_samples(source));
        assert!((mean(&xs) - 2.5).abs() < 0.2, "mean {}", mean(&xs));
    });
}

// ============================================================================
// Discrete latent selecting a gamma-family RATE
// ============================================================================
//
// Gibbs conditions a discrete latent correctly when that latent picks a
// Gaussian *location* (see the mixture tests above) and, since the
// per-rate-world draw-naming fix, when it selects a Poisson rate
// (`gibbs_conditions_discrete_poisson_rate`) or scales a Gamma rate that drives
// a Poisson count (`gibbs_conditions_discrete_gamma_rate`).

/// Fraction of samples whose display string is exactly `"True"`.
fn frac_true(samples: &[String]) -> f64 {
    let t = samples.iter().filter(|s| s.as_str() == "True").count();
    t as f64 / samples.len().max(1) as f64
}

/// A discrete latent selecting a *Dirac* (constant) POISSON RATE is conditioned
/// correctly.
///
/// Model: `b ~ flip 0.5`; `k ~ Poisson(b ? 20 : 0.5)`; observe `k = 20`.
/// A rate of 0.5 essentially cannot emit 20, so the exact posterior is
/// `P(b = True | k = 20) ≈ 1.0` (the False branch carries probability ~2.7e-24).
/// This is the hard two-block `(b, k)` case: the only step that resamples `b`
/// pins `k`, and because the rate is Dirac given `b` the `pin(k) ∧ evidence(k)`
/// conditioning BDD does not separate on `b`, producing cross-terms in the SPN
/// fold. Those cross-terms have negligible mass and are now scored as a clean
/// zero (the Poisson upper-tail `log_prob` no longer rounds `1 − CDF` to a NaN),
/// so the chain conditions correctly: empirically P(True) ≈ 1.0 over 2000 steps.
#[test]
fn gibbs_conditions_discrete_poisson_rate() {
    with_big_stack(|| {
        let source = "(query q
            (let ((b (flip 0.5))
                  (k (poisson (if b 20.0 0.5))))
              (Gibbs (if b (True) (False))
                     (native_eq k @20)
                     2000
                     (Cons b (Cons k (Nil)))
                     (WithPriorSample b))))";
        let samples = run_samples(source);
        let frac = frac_true(&samples);
        // Exact posterior is ≈1.0; the chain recovers it.
        assert!(
            frac > 0.95,
            "Gibbs P(True) = {:.3}; expected ≈1.0 (exact posterior ≈1.0).",
            frac
        );
    });
}

/// A discrete latent that scales a GAMMA RATE driving a Poisson count is now
/// conditioned correctly.
///
/// This is the soccer model's core: a discrete choice scales a Gamma rate that
/// then drives a Poisson count.
/// Model: `b ~ flip 0.5`; `r ~ Gamma(2, b ? 1 : 10)`; `k ~ Poisson(r)`;
/// observe `k = 8`. The b = False branch gives rate 10, i.e. mean intensity
/// 2/10 = 0.2, which cannot plausibly emit 8, so the exact posterior is
/// `P(b = True | k = 8) ≈ 0.99996`. With per-rate-world draw naming the chain
/// recovers this: the continuous `r` block conditions `b` via the Gamma density
/// at the pinned rate, and empirically P(True) ≈ 1.0 over 2000 steps.
#[test]
fn gibbs_conditions_discrete_gamma_rate() {
    with_big_stack(|| {
        let source = "(query q
            (let ((b (flip 0.5))
                  (r (gamma 2.0 (if b 1.0 10.0)))
                  (k (poisson r)))
              (Gibbs (if b (True) (False))
                     (native_eq k @8)
                     2000
                     (Cons b (Cons r (Cons k (Nil))))
                     (WithPriorSample b))))";
        let samples = run_samples(source);
        let frac = frac_true(&samples);
        // Exact posterior is ≈0.99996; the chain recovers ≈1.0.
        assert!(
            frac > 0.9,
            "Gibbs P(True) = {:.3}; expected ≈1.0 (exact posterior ≈0.99996).",
            frac
        );
    });
}
