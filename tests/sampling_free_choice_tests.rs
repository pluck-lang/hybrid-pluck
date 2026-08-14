//! Regression tests for free-choice resolution in Sample mode
//! (`PosteriorSamples` and `Gibbs`, which share `force_value`).
//!
//! Ensures that free choices encountered during sampling are correctly handled
//!
//! Every test below encodes the CORRECT posterior expectation. Tolerances
//! follow the `gibbs_tests.rs` convention: ±0.05 absolute frequency at
//! n = 2000 (≥ 4.5σ for every probability asserted here), loose enough to
//! survive RNG variation, far tighter than the buggy outcomes (which sit at
//! frequency 0.0 or 1.0, or spread mass onto impossible values).

use pluck::{PluckContext, ResultKind};

const N_SAMPLES: usize = 2000;
const TOL: f64 = 0.05;

/// Run `f` on a thread with a 16 MiB stack. Pluck's compile path uses deep
/// closure-stack chains via `bind_compile`, and debug-mode frame sizes can
/// exceed the default test-thread stack (~2 MiB on macOS).
///
/// Propagates any panic from the spawned thread so `#[should_panic]` tests
/// can match on the underlying message.
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

/// Empirical frequency of `target` among `samples`.
fn frac(samples: &[String], target: &str) -> f64 {
    samples.iter().filter(|s| s.as_str() == target).count() as f64 / samples.len() as f64
}

/// Parse floats from `format_value` displays, skipping non-numeric entries.
fn parse_floats(samples: &[String]) -> Vec<f64> {
    samples
        .iter()
        .filter_map(|s| s.parse::<f64>().ok())
        .collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Assert `frac(samples, target)` is within `TOL` of `expected`.
fn assert_frac(samples: &[String], target: &str, expected: f64, label: &str) {
    let f = frac(samples, target);
    assert!(
        (f - expected).abs() < TOL,
        "{}: empirical P({}) = {:.4}, expected ≈ {:.4} (±{})",
        label,
        target,
        f,
        expected,
        TOL
    );
}

// ============================================================================
// free flip in a case BRANCH (`compile_case_of` KC-reset position)
// ============================================================================

/// Evidence pins `obs = True`, so the query always takes the `if` branch.
/// The inner `flip 0.2` does not appear in the evidence BDD, so its
/// posterior equals its prior: P(True) = 0.2.
///
/// Pre-fix: the branch was compiled in KC mode, yielding two consistent
/// worlds; first-world selection returned `True` on 2000/2000 samples.
#[test]
fn free_flip_in_case_branch_follows_prior() {
    with_big_stack(|| {
        let source = "(query free-flip-in-branch
            (let ((obs (flip 0.5)))
              (PosteriorSamples (if obs (flip 0.2) (False)) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "True", 0.2, "free flip in case branch");
    });
}

// ============================================================================
// free flip in a FUNCTION BODY (`compile_app` KC-reset position)
// ============================================================================

/// Same posterior as free_flip_in_case_branch_follows_prior (P(True) = 0.2),
/// but the free flip sits inside a `define`d function body, exercising `compile_app`'s
/// KC reset independently of `compile_case_of`'s.
#[test]
fn free_flip_in_function_body_follows_prior() {
    with_big_stack(|| {
        let source = "(define (draw u) (flip 0.2))
            (query free-flip-in-function
              (let ((obs (flip 0.5)))
                (PosteriorSamples (draw obs) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "True", 0.2, "free flip in function body");
    });
}

// ============================================================================
// free categorical in a branch (distribution-shape signal)
// ============================================================================

/// Evidence pins `obs = True`; the categorical is evidence-independent, so
/// its posterior equals its prior: P(0) = 0.2, P(1) = 0.3, P(2) = 0.5.
/// Categorical outcomes are `Native(Int)`s and display as bare integers.
///
/// Pre-fix: first-world selection returned category `0` — the LEAST likely
/// outcome — on 2000/2000 samples.
#[test]
fn free_categorical_in_branch_follows_prior() {
    with_big_stack(|| {
        let source = "(query free-categorical-in-branch
            (let ((obs (flip 0.5)))
              (PosteriorSamples (if obs (categorical 0.2 0.3 0.5) @9) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "0", 0.2, "free categorical");
        assert_frac(&samples, "1", 0.3, "free categorical");
        assert_frac(&samples, "2", 0.5, "free categorical");
    });
}

// ============================================================================
// free flip at the query ROOT (genuine Sample-mode position)
// ============================================================================

/// If the posterior sample query is just a sampling statement completely
/// unconstrained/independent from the evidence, we should just get prior samples
#[test]
fn free_flip_at_query_root_follows_prior() {
    with_big_stack(|| {
        let source = "(query free-flip-at-query-root
            (let ((obs (flip 0.5)))
              (PosteriorSamples (flip 0.2) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "True", 0.2, "control flip at root");
    });
}

// ============================================================================
// evidence-BDD variable skipped by the sampled path
// ============================================================================

/// `x, y ~ flip 0.5`, evidence `(or x y)`, query `(if x y (False))`.
/// Exact posterior: P(True) = P(x ∧ y | x ∨ y) = (1/4)/(3/4) = 1/3.
#[test]
fn path_skipped_evidence_var_follows_prior() {
    with_big_stack(|| {
        let source = "(query path-skipped-evidence-var-follows-prior
            (let ((x (flip 0.5)) (y (flip 0.5)))
              (PosteriorSamples (if x y (False)) (or x y) 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "True", 1.0 / 3.0, "path-skipped evidence var");
    });
}

// ============================================================================
// free `uniform_int_range`: IntDist bit resolution
// ============================================================================

/// `(uniform_int_range @2 @1 @3)` is evidence-independent, so its posterior
/// equals its prior: uniform over {1, 2, 3} — and NEVER 0, which is outside
/// the range.
///
/// Pre-fix: `force_value`'s IntDist arm resolved each unpinned bit BDD with
/// an independent 0.5 coin flip, ignoring the correlated range encoding:
/// ≈25% each over {@0, @1, @2, @3}, including the impossible `@0`.
#[test]
fn free_uniform_int_range_uniform_and_in_range() {
    with_big_stack(|| {
        let source = "(query free-uniform-int-range
            (let ((obs (flip 0.5)))
              (PosteriorSamples (uniform_int_range @2 @1 @3) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        let zeros = samples.iter().filter(|s| s.as_str() == "@0").count();
        assert_eq!(
            zeros, 0,
            "{} out-of-range @0 samples (range is [1, 3])",
            zeros
        );
        assert_frac(&samples, "@1", 1.0 / 3.0, "uniform_int_range");
        assert_frac(&samples, "@2", 1.0 / 3.0, "uniform_int_range");
        assert_frac(&samples, "@3", 1.0 / 3.0, "uniform_int_range");
    });
}

// ============================================================================
// free flip in a branch, Gibbs path (`force_step` → `force_value`)
// ============================================================================

/// Gibbs over blocks {x, y} with evidence `(or x y)`; the query's inner
/// `flip 0.2` is free. P(x=T | x∨y) = 2/3 (exact, matches the existing
/// two-block Bernoulli Gibbs test), and the free flip multiplies in its
/// prior: P(True) = (2/3) · 0.2 = 2/15 ≈ 0.1333.
///
/// Pre-fix: the free flip collapsed to `True`, giving P(True) ≈ 2/3.
/// Confirms the bug surface is shared by the Gibbs driver, not specific to
/// `PosteriorSamples`.
#[test]
fn gibbs_free_flip_in_branch_follows_prior() {
    with_big_stack(|| {
        let source = "(query gibbs-free-flip-in-branch
            (let ((x (flip 0.5)) (y (flip 0.5)))
              (Gibbs (if x (flip 0.2) (False))
                     (or x y)
                     2000
                     (Cons x (Cons y (Nil)))
                     (WithPriorSample x))))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(
            &samples,
            "True",
            (2.0 / 3.0) * 0.2,
            "Gibbs free flip in branch",
        );
    });
}

// ============================================================================
// Recursive generative functions (forward single-world execution)
// ============================================================================

/// Geometric via Peano constructors: `(if (flip 0.5) (O) (S (geomp)))`.
/// Posterior = prior = Geometric(0.5): P(k) = 0.5^(k+1). Peano nats
/// display as bare integers.
///
/// Pre-fix: the branch flip collapsed to `True` → 100% zeros (lazy KC kept
/// it terminating because the recursion is constructor-guarded).
#[test]
fn geometric_peano_follows_prior() {
    with_big_stack(|| {
        let source = "(define (geomp) (if (flip 0.5) (O) (S (geomp))))
            (query geom-peano (PosteriorSamples (geomp) (True) 2000))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "0", 0.5, "geom-peano");
        assert_frac(&samples, "1", 0.25, "geom-peano");
        assert_frac(&samples, "2", 0.125, "geom-peano");
    });
}

/// Geometric via native ints: the recursion sits in a STRICT position
/// (`int_add`'s argument), so before Sample-mode propagation this
/// knowledge-compiled the unbounded recursion symbolically and
/// stack-overflowed in seconds for 5 samples. With propagation the
/// per-sample cost is O(executed path) and the distribution is
/// Geometric(0.5). This test doubles as the termination/architecture
/// regression guard.
#[test]
fn geometric_native_terminates_and_follows_prior() {
    with_big_stack(|| {
        let source = "(define (geomn)
              (if (flip 0.5) (mk_int @8 @0) (int_add (mk_int @8 @1) (geomn))))
            (query geom-native (PosteriorSamples (geomn) (True) 2000))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "@0", 0.5, "geom-native");
        assert_frac(&samples, "@1", 0.25, "geom-native");
        assert_frac(&samples, "@2", 0.125, "geom-native");
    });
}

// ============================================================================
// Within-sample consistency (free draws recorded in the constraint)
// ============================================================================

/// `z ~ flip 0.3` is met twice per sample: as a forced thunk root (first
/// Pair component) and inside the taken `if` branch. Both meetings must
/// resolve to the same drawn value — free draws are recorded in the
/// per-sample trace AND constraint, so key-level and guard-level
/// revisits agree. Pairs display as `(a, b)`.
#[test]
fn within_sample_flip_consistency() {
    with_big_stack(|| {
        let source = "(query flip-consistency
            (let ((obs (flip 0.5)) (z (flip 0.3)))
              (PosteriorSamples (Pair z (if obs z (True))) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        let agree_true = frac(&samples, "(True, True)");
        let agree_false = frac(&samples, "(False, False)");
        assert!(
            (agree_true + agree_false - 1.0).abs() < 1e-9,
            "flip consistency: {} mismatched pairs",
            samples
                .iter()
                .filter(|s| s.as_str() != "(True, True)" && s.as_str() != "(False, False)")
                .count()
        );
        assert!(
            (agree_true - 0.3).abs() < TOL,
            "flip consistency: empirical P(True pair) = {:.4}, expected ≈ 0.3",
            agree_true
        );
    });
}

/// The same shared `uniform_int_range` draw forced twice must land on the
/// same value (the range walk re-resolves through the per-sample trace),
/// with the correct uniform marginal over {1, 2, 3}.
#[test]
fn within_sample_int_dist_consistency() {
    with_big_stack(|| {
        let source = "(query int-consistency
            (let ((n (uniform_int_range @2 @1 @3)))
              (PosteriorSamples (Pair n n) (True) 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        for target in ["(@1, @1)", "(@2, @2)", "(@3, @3)"] {
            assert_frac(&samples, target, 1.0 / 3.0, "IntDist consistency");
        }
        let mismatched = samples
            .iter()
            .filter(|s| !["(@1, @1)", "(@2, @2)", "(@3, @3)"].contains(&s.as_str()))
            .count();
        assert_eq!(mismatched, 0, "IntDist consistency: mismatched pairs");
    });
}

// ============================================================================
// Independent applications: each call site draws its own free flip
// ============================================================================

/// Two applications of the same function must compile the body's free flip
/// at distinct callstack addresses — fresh randomness per application.
/// `(Pair (draw obs) (draw obs))` therefore has independent Bernoulli(0.5)
/// components: all four pair shapes at probability 1/4.
///
/// Failure signature if the applications shared one variable (or one trace
/// entry): mixed pairs vanish — only `(True, True)` / `(False, False)`
/// appear, at 1/2 each. Note both call sites use the SAME argument, so only
/// the application position can distinguish them.
///
/// Complements `within_sample_flip_consistency`, which checks the opposite
/// contract: ONE application bound by `let` and referenced twice must agree
/// with itself.
#[test]
fn separate_applications_draw_independent_flips() {
    with_big_stack(|| {
        let source = "(define (draw u) (flip 0.5))
            (query independent-applications
              (let ((obs (flip 0.5)))
                (PosteriorSamples (Pair (draw obs) (draw obs)) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        for target in [
            "(True, True)",
            "(True, False)",
            "(False, True)",
            "(False, False)",
        ] {
            assert_frac(&samples, target, 0.25, "independent applications");
        }
    });
}

// ============================================================================
// ThunkUnion selection — the one live multi-world resolution point
// ============================================================================
//
// In Sample mode every compile returns exactly one world (asserted at the
// bind boundary, `sample_compile_one`, and `force_value`). The single
// remaining multi-world structure is a ThunkUnion: a KC-built merge that
// reaches Sample-mode evaluation through a shared env value (e.g. a
// top-level `match` whose scrutinee was knowledge-compiled before the
// PosteriorSamples node). `evaluate_thunk_union` resolves it via
// `select_world`, prior-drawing any free guard variable. These tests pin
// that path — and double as canaries for the exactly-one-world asserts.

/// The top-level `match` scrutinee is KC-evaluated (it sits outside the
/// PosteriorSamples node), so `join_monad` merges the two `Pair` worlds and
/// binds `p` to a ThunkUnion with guards {g, ¬g} over the free `flip 0.3`
/// variable. Per sample, the TU selector must prior-draw g and pick the
/// matching component: P(True) = 0.3·0.2 + 0.7·0.8 = 0.62.
#[test]
fn thunk_union_selection_follows_mixture() {
    with_big_stack(|| {
        let source = "(query tu-prob
            (match (if (flip 0.3) (Pair 0.2 (True)) (Pair 0.8 (True)))
              Pair p c => (PosteriorSamples (flip p) c 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "True", 0.62, "ThunkUnion mixture");
    });
}

/// A `let`-bound `if` over two `mk_int`s is bit-level-merged by KC into ONE
/// symbolic IntDist — but only inside the thunk's cache, which Sample mode
/// bypasses: forcing the shared thunk re-executes it forward (one branch,
/// constant bits). Guards the no-KC-leak property that justifies the
/// exactly-one-world asserts. P(True) = 0.5.
#[test]
fn shared_thunk_reevaluates_forward() {
    with_big_stack(|| {
        let source = "(query smuggle-let
            (let ((n (if (flip 0.5) (mk_int @2 @1) (mk_int @2 @2))))
              (PosteriorSamples (int_eq n (mk_int @2 @1)) (True) 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "True", 0.5, "shared thunk forward re-eval");
    });
}

/// A recursive generative process used as the QUERY TARGET (not just the
/// evidence): `force_value` re-evaluates `gen` forward in Sample mode. Each
/// level binds `c = (nxt prev)` and reads `c` three times — `int_dist_eq c
/// EOS`, the `Cons` head, and the recursive `(gen c)` — while `prev` is read
/// inside `nxt`. Sample mode bypasses the KC thunk cache, so WITHOUT
/// per-sample memoization every read re-derives the whole prefix from
/// scratch: E[work] = Σ_D P(depth ≥ D)·3^D = Σ_D 0.5^D·3^D diverges, and the
/// query never returns (the original `rotation_cipher` hang). Memoizing a
/// thunk by identity within a sample makes it linear. Length is geometric:
/// P(`[]`) = 0.5, P(`"B"`) = 0.25.
#[test]
fn recursive_query_target_memoizes() {
    with_big_stack(|| {
        let source = "(define EOS2 (mk_int @8 @36))
            (define (nxt prev)
              (if (int_dist_eq prev (mk_int @8 @65))
                (discrete ((mk_int @8 @66) 0.5) (EOS2 0.5))
                (discrete ((mk_int @8 @65) 0.5) (EOS2 0.5))))
            (define (gen prev)
              (let ((c (nxt prev)))
                (if (int_dist_eq c EOS2)
                  (Nil)
                  (Cons c (gen c)))))
            (query g (PosteriorSamples (gen (mk_int @8 @65)) (True) 2000))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "[]", 0.5, "recursive query target: P(empty)");
        assert_frac(&samples, "\"B\"", 0.25, "recursive query target: P(\"B\")");
    });
}

/// Same ThunkUnion-through-`match` shape as the mixture test, but the
/// components are IntDist thunks consumed by `int_eq`: selection happens at
/// the TU layer, so the comparison sees constant bits (a symbolic operand
/// would trip the exactly-one-world assert via `int_dist_eq`'s two-world
/// result). P(True) = 0.5.
#[test]
fn thunk_union_int_dist_selection() {
    with_big_stack(|| {
        let source = "(query tu-int
            (match (if (flip 0.5) (Pair (mk_int @2 @1) (True)) (Pair (mk_int @2 @2) (True)))
              Pair n c => (PosteriorSamples (int_eq n (mk_int @2 @1)) (True) 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "True", 0.5, "ThunkUnion IntDist selection");
    });
}

// ============================================================================
// Mid-sample continuous-variable registration
// ============================================================================

/// A `dirichlet` + `categorical` reached inside a FUNCTION BODY in Sample
/// mode: the Dirichlet's continuous variable is first *registered*
/// mid-sample (the evidence never compiles that body), after the sample's
/// continuous assignment was drawn. `SampleState::ensure_continuous` must
/// extend the assignment with a fresh prior draw — exact, because a
/// variable the evidence never compiled cannot appear in the evidence SPN.
/// Marginal over categories = E[Dirichlet(1,1)] = (1/2, 1/2).
#[test]
fn mid_sample_continuous_registration_dirichlet() {
    with_big_stack(|| {
        let source = "(define (draw-cat) (categorical (dirichlet 1.0 1.0)))
            (query cat-dirichlet
              (let ((obs (flip 0.5)))
                (PosteriorSamples (draw-cat) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "0", 0.5, "mid-sample dirichlet registration");
        assert_frac(&samples, "1", 0.5, "mid-sample dirichlet registration");
    });
}

/// Same shape for a Gaussian: `(normal 0 1)` inside a function body is
/// registered mid-sample and forced via `force_symbolic` against the
/// extended assignment. Sample mean of N(0,1) over 2000 draws ≈ 0
/// (tolerance 0.1 ≈ 4.5σ of the mean).
#[test]
fn mid_sample_continuous_registration_gaussian() {
    with_big_stack(|| {
        let source = "(define (draw-g u) (normal 0.0 1.0))
            (query g-body
              (let ((obs (flip 0.5)))
                (PosteriorSamples (draw-g obs) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        let vals = parse_floats(&samples);
        assert_eq!(vals.len(), N_SAMPLES, "all samples should be floats");
        let m = mean(&vals);
        assert!(
            m.abs() < 0.1,
            "mid-sample gaussian registration: sample mean = {:.4}, expected ≈ 0",
            m
        );
    });
}

// ============================================================================
// Free discrete choice routes WHICH gamma draw the evidence observes
// ============================================================================
//
// Each syntactic `(exponential ...)` call site allocates its own draw
// variable, so when a free flip selects between two call sites the evidence
// SPN is a Sum whose branches scope DIFFERENT draws. The posterior sampler
// realizes only the sampled branch's draws, and draws are (by design) not
// registry-registered, so the unsampled branch's draw is absent from the
// assignment when `sample_discrete_given` evaluates that branch's flip
// weights.
//
// Exact posterior on lam ~ Gamma(2,1) with likelihood
// 0.5·(1−e^{−λ}) + 0.5·(1−e^{−2λ}):
//   Z = 1 − 0.5/(1+1)² − 0.5/(1+2)² = 0.819444…
//   E[λ] = (2 − 0.5·2/2³ − 0.5·2/3³)/Z ≈ 2.2429
// Posterior sd ≈ 1.42 ⇒ SE of the mean at n=2000 ≈ 0.032; tol 0.15 ≈ 4.7σ.
//
// Pre-fix: panics with `Assignment::gamma_value: variable N not found`.

/// PosteriorSamples form: the router flip `w` is free in the evidence BDD.
#[test]
fn posterior_samples_free_choice_routes_gamma_draw() {
    with_big_stack(|| {
        let source = "(query free-choice-gamma-draw
            (let ((lam (gamma 2.0 1.0))
                  (w (flip 0.5)))
              (PosteriorSamples lam
                (if w
                    (real_lt (exponential lam) 1.0)
                    (real_lt (exponential lam) 2.0))
                2000)))";
        let samples = run_samples(source);
        let xs = parse_floats(&samples);
        assert_eq!(xs.len(), N_SAMPLES, "all samples should be floats");
        let m = mean(&xs);
        assert!(
            (m - 2.2429).abs() < 0.15,
            "free-choice gamma draw: empirical E[lam] = {:.4}, expected ≈ 2.2429",
            m
        );
    });
}

/// Gibbs form: block `z` is pinned each step, but the router `w` stays FREE
/// in every conditioning BDD (the ant_colonies shape: pinning one block
/// leaves the other block's routing choices free). `z` is independent of
/// `lam`, so the posterior on `lam` is the same as above.
#[test]
fn gibbs_free_choice_routes_gamma_draw() {
    with_big_stack(|| {
        let source = "(query gibbs-free-choice-gamma-draw
            (let ((lam (gamma 2.0 1.0))
                  (w (flip 0.5))
                  (z (flip 0.5)))
              (Gibbs lam
                     (and z
                          (if w
                              (real_lt (exponential lam) 1.0)
                              (real_lt (exponential lam) 2.0)))
                     2000
                     (Cons z (Nil))
                     (WithConstant (True)))))";
        let samples = run_samples(source);
        let xs = parse_floats(&samples);
        assert_eq!(xs.len(), N_SAMPLES, "all samples should be floats");
        let m = mean(&xs);
        assert!(
            (m - 2.2429).abs() < 0.15,
            "Gibbs free-choice gamma draw: empirical E[lam] = {:.4}, expected ≈ 2.2429",
            m
        );
    });
}

// ============================================================================
// CORNER CASE: shared draw across rate-worlds — fill must use the routing
// variable's POSTERIOR weights, not a uniform pick over the rates
// ============================================================================
//
// `(exponential (if w 1.0 10.0))` is ONE call site whose rate depends on the
// world, so it allocates ONE draw `d` registered under TWO rates (1.0 under
// w, 10.0 under ¬w). When the evidence `(or z (real_lt d 1.0))` is satisfied
// via `z` (the d-omitting Sum branch), `d` is skipped by the posterior sample
// and filled by `realize_missing_gamma_draws`.
//
// The CORRECT fill law is the rate-mixture weighted by P(w | path). On the
// z=T path the evidence is independent of w, so P(w=A | z=T) = prior = 0.9.
// pluck's own EXACT inference (`Posterior [d] (or z (real_lt d 1.0))`) gives
// the ground-truth posterior of d as a 4-component mixture:
//   53.93%  Exp(1)               (w=A, z=T, d unobserved)   mean 1.0
//   34.09%  TruncExp(1, [0,1))   (w=A, z=F)                 mean 0.41802
//    5.99%  Exp(10)              (w=B, z=T, d unobserved)   mean 0.1
//    5.99%  TruncExp(10, [0,1))  (w=B, z=F)                 mean 0.09995
//   => E[d | E] = 0.6938.
// Note the z=T mass splits 53.93 : 5.99 = 0.9 : 0.1 — exactly the posterior
// weights the fill must reproduce.
//
// Previously a single draw was shared across both rate-worlds and filled from a
// uniformly-chosen rate (mean 0.55 → estimate ≈ 0.473, ~0.22 below the exact
// 0.6938). Since the per-rate-world draw-naming fix, each rate-world has its own
// draw, so the routing variable's posterior weights are respected and the
// estimate matches the exact value.
#[test]
fn corner_case_shared_draw_uses_routing_posterior_weights() {
    with_big_stack(|| {
        let source = "(query corner-shared-draw-weights
            (let ((w (flip 0.9))
                  (z (flip 0.5))
                  (d (exponential (if w 1.0 10.0))))
              (PosteriorSamples d (or z (real_lt d 1.0)) 6000)))";
        let samples = run_samples(source);
        let xs = parse_floats(&samples);
        assert_eq!(xs.len(), 6000, "all samples should be floats");
        let m = mean(&xs);
        // Exact (pluck `Posterior`): E[d|E] = 0.6938. SE at n=6000 ≈ 0.011,
        // so 0.05 ≈ 4.5σ; the uniform-fill value ≈ 0.473 misses by ~0.22.
        assert!(
            (m - 0.6938).abs() < 0.05,
            "corner-case shared draw: empirical E[d] = {:.4}, expected ≈ 0.6938 \
             (uniform-rate fill gives ≈ 0.473)",
            m
        );
    });
}

// Poisson sibling of the exponential corner case above, exercising the
// Poisson arm of `realize_missing_gamma_draws`. `(poisson (if w 1.0 10.0))`
// is ONE call site → ONE draw `k` registered under two rates (1.0 under w,
// 10.0 under ¬w). Evidence `(or z (native_eq k @0))` skips `k` on the z=T
// branch, so it is filled rather than observed.
//
// pluck's EXACT inference (`Posterior [k] (or z (native_eq k @0))`) gives the
// ground-truth posterior of k as a 4-component mixture:
//   67.61%  Poisson(1)              (w=A, z=T, k unobserved)   mean 1.0
//   24.87%  TruncPoisson(1, {0})    (w=A, z=F)                 mean 0
//    7.51%  Poisson(10)             (w=B, z=T, k unobserved)   mean 10.0
//    0.00%  TruncPoisson(10, {0})   (w=B, z=F, ~e^-10 mass)    mean 0
//   => E[k | E] = 0.6761·1 + 0.0751·10 = 1.427.
// As before the z=T mass splits 67.61 : 7.51 = 0.9 : 0.1 — the posterior
// weights the fill must reproduce.
//
// Previously a single shared draw was filled from a uniformly-chosen rate
// (mean 5.5 → estimate ≈ 4.0). With per-rate-world draw naming each rate-world
// has its own draw, so the routing posterior weights are respected and the
// estimate matches the exact value. (Fixed alongside its sibling.)
#[test]
fn corner_case_shared_poisson_draw_uses_routing_posterior_weights() {
    with_big_stack(|| {
        let source = "(query corner-shared-poisson-weights
            (let ((w (flip 0.9))
                  (z (flip 0.5))
                  (k (poisson (if w 1.0 10.0))))
              (PosteriorSamples k (or z (native_eq k @0)) 6000)))";
        let samples = run_samples(source);
        let xs = parse_floats(&samples);
        assert_eq!(xs.len(), 6000, "all samples should be floats");
        let m = mean(&xs);
        // Exact (pluck `Posterior`): E[k|E] = 1.427. SD ≈ 2.75 ⇒ SE at
        // n=6000 ≈ 0.036, so 0.2 ≈ 5.6σ; the uniform-fill value ≈ 4.0
        // misses by ~2.6.
        assert!(
            (m - 1.427).abs() < 0.2,
            "corner-case shared poisson draw: empirical E[k] = {:.4}, expected ≈ 1.427 \
             (uniform-rate fill gives ≈ 4.0)",
            m
        );
    });
}

// ============================================================================
// Error-mass policy: program error along the sampled path panics
// ============================================================================

/// A partial match over a free flip in a branch position: the sampled path
/// reaches shaved error mass with probability 1 − 0.3¹⁰⁰ ≈ 1.
#[test]
#[should_panic(expected = "program error along the sampled path")]
fn error_mass_along_sampled_path_panics() {
    with_big_stack(|| {
        let source = "(query err-branch
            (let ((obs (flip 0.5)))
              (PosteriorSamples (if obs (match (flip 0.3) True => (True)) (False)) obs 100)))";
        let _ = run_samples(source);
    });
}
