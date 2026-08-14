//! Integration tests for *scaled gammas*: using `(*. g s)` / `(/. g s)` as
//! the rate of an `exponential` / `poisson`. Organized by implementation
//! phase (see the plan): Phase 1 covers the parse/compile frontend (legal
//! scaling compiles; illegal gamma arithmetic panics with a clear message).
//! Later phases add posterior-equivalence, pooling, disintegration, and
//! query/display tests.

use pluck::{GammaPrior, PluckContext, ResultKind};

/// Run a pluck source on a 16 MiB stack (compile uses deep `bind_compile`
/// closure chains) and propagate any panic so `#[should_panic(expected =
/// "...")]` can match the underlying message.
fn run_on_big_stack(source: &'static str) -> Vec<pluck::QueryResult> {
    let handle = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let mut ctx = PluckContext::new();
            ctx.run(source)
        })
        .expect("spawn");
    match handle.join() {
        Ok(results) => results,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

// ============================================================================
// Phase 1 — frontend: legal scaling compiles, illegal arithmetic panics.
// ============================================================================

#[test]
fn scaled_exponential_rate_compiles_and_runs() {
    // `(*. r 5.0)` as a rate must parse and compile end-to-end. (The posterior
    // numbers only become *correct* once Phase 3 threads the scale into the
    // conjugate update; here we only assert it runs and yields a distribution.)
    let results = run_on_big_stack(
        "(query (let ((r (gamma 2.0 1.0))) \
           (Posterior r (real_eq (exponential (*. r 5.0)) 0.3))))",
    );
    assert_eq!(results.len(), 1);
    assert!(matches!(results[0].kind, ResultKind::Distribution { .. }));
}

#[test]
fn scaled_poisson_rate_compiles_and_runs() {
    let results = run_on_big_stack(
        "(query (let ((r (gamma 2.0 1.0))) \
           (Posterior r (native_eq (poisson (*. r 1.5)) @2))))",
    );
    assert_eq!(results.len(), 1);
    assert!(matches!(results[0].kind, ResultKind::Distribution { .. }));
}

#[test]
fn dividing_gamma_rate_compiles_and_runs() {
    // `(/. r 2.0)` is a scale of 0.5.
    let results = run_on_big_stack(
        "(query (let ((r (gamma 2.0 1.0))) \
           (Posterior r (real_eq (exponential (/. r 2.0)) 0.3))))",
    );
    assert_eq!(results.len(), 1);
    assert!(matches!(results[0].kind, ResultKind::Distribution { .. }));
}

#[test]
#[should_panic(expected = "addition/subtraction of a Gamma")]
fn adding_to_a_gamma_panics() {
    run_on_big_stack(
        "(query (let ((r (gamma 2.0 1.0))) \
           (Posterior r (real_eq (exponential (+. r 1.0)) 0.3))))",
    );
}

#[test]
#[should_panic(expected = "can only be scaled by a numeric constant")]
fn multiplying_two_gammas_panics() {
    run_on_big_stack(
        "(query (let ((r (gamma 2.0 1.0))) \
           (Posterior r (real_eq (exponential (*. r r)) 0.3))))",
    );
}

#[test]
#[should_panic(expected = "Exponential/Poisson draw")]
fn scaling_a_draw_panics() {
    run_on_big_stack(
        "(query (let ((r (gamma 2.0 1.0))) \
           (Posterior r (real_eq (*. (exponential r) 5.0) 0.3))))",
    );
}

#[test]
#[should_panic(expected = "positive scale")]
fn scaling_a_gamma_by_zero_panics() {
    run_on_big_stack(
        "(query (let ((r (gamma 2.0 1.0))) \
           (Posterior r (real_eq (exponential (*. r 0.0)) 0.3))))",
    );
}

// ============================================================================
// Phase 3 — end-to-end posterior correctness (scale threaded into the
// conjugate update). The queried gamma's posterior is read straight out of
// the result packet and checked against the hand-computed conjugate update.
// ============================================================================

/// Extract the single queried gamma's `(shape, rate)` from a one-item,
/// one-gamma `Posterior` result.
fn single_gamma_posterior(source: &'static str) -> (f64, f64) {
    let results = run_on_big_stack(source);
    assert_eq!(results.len(), 1, "expected one query");
    let items = match &results[0].kind {
        ResultKind::Distribution { items } => items,
        other => panic!("expected Distribution, got {other:?}"),
    };
    assert_eq!(items.len(), 1, "expected a single posterior world");
    let gammas = &items[0].posteriors.gamma;
    assert_eq!(gammas.len(), 1, "expected exactly one gamma in the packet");
    match &gammas[0].gamma {
        GammaPrior::Gamma { shape, rate, .. } => (*shape, *rate),
        other => panic!("expected a Gamma posterior, got {other}"),
    }
}

#[test]
fn scaled_exponential_single_use_posterior() {
    // g ~ Gamma(2, 1); observe Exp(3·g) = 0.5.
    // Scaled Dirac update: Gamma(2 + 1, 1 + 3·0.5) = Gamma(3, 2.5).
    let (shape, rate) = single_gamma_posterior(
        "(query (let ((g (gamma 2.0 1.0))) \
           (Posterior g (real_eq (exponential (*. g 3.0)) 0.5))))",
    );
    assert!((shape - 3.0).abs() < 1e-9, "shape {shape}");
    assert!((rate - 2.5).abs() < 1e-9, "rate {rate}");
}

#[test]
fn scaled_exponential_reparameterization_equivalence() {
    // Scaling the rate by m is equivalent to dividing the gamma's rate by m
    // *for the scaled variable*: m·g with g~Gamma(k,1) is distributed
    // Gamma(k, 1/m). Querying the underlying g instead, the scaled-rate model
    // g~Gamma(2,1) with Exp(4·g)=v gives posterior Gamma(3, 1 + 4v); check
    // that equals the direct Dirac math for s=4, v=0.25 ⇒ Gamma(3, 2).
    let (shape, rate) = single_gamma_posterior(
        "(query (let ((g (gamma 2.0 1.0))) \
           (Posterior g (real_eq (exponential (*. g 4.0)) 0.25))))",
    );
    assert!((shape - 3.0).abs() < 1e-9, "shape {shape}");
    assert!((rate - 2.0).abs() < 1e-9, "rate {rate}");
}

#[test]
fn pooled_scaled_exponentials_share_one_posterior() {
    // One shared g ~ Gamma(2, 1) feeds two exponentials at DIFFERENT scales,
    // both observed. The two scaled-Dirac updates pool into one posterior:
    //   Gamma(2 + 2, 1 + 2·0.5 + 4·0.25) = Gamma(4, 3).
    let (shape, rate) = single_gamma_posterior(
        "(query (let ((g (gamma 2.0 1.0))) \
           (Posterior g (and (real_eq (exponential (*. g 2.0)) 0.5) \
                             (real_eq (exponential (*. g 4.0)) 0.25)))))",
    );
    assert!((shape - 4.0).abs() < 1e-9, "shape {shape}");
    assert!((rate - 3.0).abs() < 1e-9, "rate {rate}");
}

#[test]
fn scaled_exponential_draw_is_sampled_at_scaled_rate() {
    // g ~ Gamma(3, 2), so E[1/g] = rate/(shape−1) = 2/2 = 1. A queried,
    // unobserved draw x ~ Exp(4·g) has mean E[1/(4·g)] = (1/4)·E[1/g] = 0.25.
    // If the scale were dropped, the mean would be ≈ 1.0 (4× larger) — so
    // this directly checks the draw is realized at the SCALED rate.
    let results = run_on_big_stack(
        "(query x (let ((g (gamma 3.0 2.0))) \
           (PosteriorSamples (exponential (*. g 4.0)) (True) 6000)))",
    );
    let values = match &results[0].kind {
        ResultKind::Samples { values } => values,
        other => panic!("expected Samples, got {other:?}"),
    };
    let xs: Vec<f64> = values
        .iter()
        .filter_map(|s| s.parse::<f64>().ok())
        .collect();
    assert!(
        xs.len() > 5000,
        "expected mostly numeric samples, got {}",
        xs.len()
    );
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    assert!(
        (mean - 0.25).abs() < 0.05,
        "empirical mean {mean:.4}, expected ≈ 0.25 (drawn at 4·g, not g)"
    );
}

#[test]
fn pooled_scaled_poisson_share_one_posterior() {
    // Shared g ~ Gamma(2, 1) feeds two Poissons at scales 2 and 3:
    //   observe Poisson(2·g)=@1 and Poisson(3·g)=@4.
    // Pooled update: shape += 1 + 4 = 5; rate += 2 + 3 = 5.
    //   Gamma(2 + 5, 1 + 5) = Gamma(7, 6).
    let (shape, rate) = single_gamma_posterior(
        "(query (let ((g (gamma 2.0 1.0))) \
           (Posterior g (and (native_eq (poisson (*. g 2.0)) @1) \
                             (native_eq (poisson (*. g 3.0)) @4)))))",
    );
    assert!((shape - 7.0).abs() < 1e-9, "shape {shape}");
    assert!((rate - 6.0).abs() < 1e-9, "rate {rate}");
}

// ============================================================================
// Phase 4 — direct disintegration of a scaled gamma:
//   (real_eq (*. g s) v)  ⇒  pin g = v/s, with the 1/s Jacobian on the
//   evidence. Pinning the scaled rate must (a) pin g to v/s and (b) discount
//   the evidence by ln(s) relative to the unscaled pin g = v/s.
// ============================================================================

/// The pinned value of a one-item, one-gamma `Posterior` result whose gamma
/// collapsed to a point mass.
fn pinned_gamma_value(source: &'static str) -> f64 {
    let results = run_on_big_stack(source);
    let items = match &results[0].kind {
        ResultKind::Distribution { items } => items,
        other => panic!("expected Distribution, got {other:?}"),
    };
    assert_eq!(items.len(), 1, "expected a single posterior world");
    let gammas = &items[0].posteriors.gamma;
    assert_eq!(gammas.len(), 1, "expected exactly one gamma in the packet");
    match &gammas[0].gamma {
        GammaPrior::Pinned { value, .. } => *value,
        other => panic!("expected a Pinned posterior, got {other}"),
    }
}

/// Posterior probability of the item displayed as `target`.
fn posterior_prob(source: &'static str, target: &str) -> f64 {
    let results = run_on_big_stack(source);
    let items = match &results[0].kind {
        ResultKind::Distribution { items } => items,
        other => panic!("expected Distribution, got {other:?}"),
    };
    items
        .iter()
        .filter(|i| i.display == target)
        .map(|i| i.log_probability.exp())
        .sum()
}

#[test]
fn scaled_rate_pin_pins_the_quotient() {
    // (real_eq (*. g 4.0) 2.0) and (real_eq g 0.5) both pin g = 0.5.
    let scaled = pinned_gamma_value(
        "(query g (let ((g (gamma 2.0 1.0))) (Posterior g (real_eq (*. g 4.0) 2.0))))",
    );
    let direct =
        pinned_gamma_value("(query g (let ((g (gamma 2.0 1.0))) (Posterior g (real_eq g 0.5))))");
    assert!((scaled - 0.5).abs() < 1e-9, "scaled pin value {scaled}");
    assert!((direct - 0.5).abs() < 1e-9, "direct pin value {direct}");
}

// ============================================================================
// Phase 5 — query + display of a scaled gamma. Querying `(*. g s)` reports the
// distribution of `s·g` (Gamma(shape, rate/s)) and shows the scale in the label.
// ============================================================================

#[test]
fn querying_scaled_gamma_reports_scaled_distribution() {
    // g ~ Gamma(2, 1); observe Exp(g) = 0.5 ⇒ posterior g ~ Gamma(3, 1.5).
    // Querying (*. g 2.0) reports the distribution of 2·g:
    //   Gamma(3, 1.5/2) = Gamma(3, 0.75).
    let (shape, rate) = single_gamma_posterior(
        "(query scaled-g (let ((g (gamma 2.0 1.0))) \
           (Posterior (*. g 2.0) (real_eq (exponential g) 0.5))))",
    );
    assert!((shape - 3.0).abs() < 1e-9, "shape {shape}");
    assert!((rate - 0.75).abs() < 1e-9, "rate {rate}");

    // Sanity: querying the bare g reports the unscaled posterior Gamma(3, 1.5).
    let (bshape, brate) = single_gamma_posterior(
        "(query g (let ((g (gamma 2.0 1.0))) \
           (Posterior g (real_eq (exponential g) 0.5))))",
    );
    assert!((bshape - 3.0).abs() < 1e-9, "bare shape {bshape}");
    assert!((brate - 1.5).abs() < 1e-9, "bare rate {brate}");
}

#[test]
fn scaled_gamma_query_label_shows_the_scale() {
    // The queried value's display renders the scale factor (`2 * …`).
    let results = run_on_big_stack(
        "(query scaled-g (let ((g (gamma 2.0 1.0))) \
           (Posterior (*. g 2.0) (real_eq (exponential g) 0.5))))",
    );
    let items = match &results[0].kind {
        ResultKind::Distribution { items } => items,
        other => panic!("expected Distribution, got {other:?}"),
    };
    assert_eq!(items.len(), 1);
    let display = &items[0].display;
    assert!(
        display.contains("2") && display.contains('*'),
        "display {display:?} should show the scale factor `2 * …`"
    );
}

// ============================================================================
// Phase 5b — the SAME gamma queried at MULTIPLE scales reports one entry per
// scale (keyed by the queried expression `s·g`, mirroring the Gaussian family),
// and the underlying variable's posterior is never mutated.
// ============================================================================

/// The `GammaPrior` of every gamma entry in a one-item `Posterior` result.
fn gamma_entries(source: &'static str) -> Vec<GammaPrior<String>> {
    let results = run_on_big_stack(source);
    let items = match &results[0].kind {
        ResultKind::Distribution { items } => items,
        other => panic!("expected Distribution, got {other:?}"),
    };
    assert_eq!(items.len(), 1, "expected a single posterior world");
    items[0]
        .posteriors
        .gamma
        .iter()
        .map(|jg| jg.gamma.clone())
        .collect()
}

#[test]
fn pair_of_two_scales_reports_two_entries() {
    // g ~ Gamma(2,1), observe Exp(g)=0.5 ⇒ g ~ Gamma(3, 1.5).
    // Querying (Pair (*. g 2) (*. g 3)) must report BOTH 2·g ~ Gamma(3, 0.75)
    // and 3·g ~ Gamma(3, 0.5) as separate, correctly-labelled entries.
    let entries = gamma_entries(
        "(query both (let ((g (gamma 2.0 1.0))) \
           (Posterior (Pair (*. g 2.0) (*. g 3.0)) (real_eq (exponential g) 0.5))))",
    );
    assert_eq!(
        entries.len(),
        2,
        "expected two scaled entries, got {entries:?}"
    );
    let mut by_rate: Vec<(String, f64, f64)> = entries
        .iter()
        .map(|g| match g {
            GammaPrior::Gamma { name, shape, rate } => (name.clone(), *shape, *rate),
            other => panic!("expected Gamma, got {other}"),
        })
        .collect();
    // Sort by descending rate: 2·g has rate 0.75, 3·g has rate 0.5.
    by_rate.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    let (n2, sh2, r2) = &by_rate[0];
    let (n3, sh3, r3) = &by_rate[1];
    assert!(
        (sh2 - 3.0).abs() < 1e-9 && (r2 - 0.75).abs() < 1e-9,
        "2·g entry {by_rate:?}"
    );
    assert!(
        (sh3 - 3.0).abs() < 1e-9 && (r3 - 0.5).abs() < 1e-9,
        "3·g entry {by_rate:?}"
    );
    assert!(n2.contains("2 *"), "2·g label {n2:?}");
    assert!(n3.contains("3 *"), "3·g label {n3:?}");
}

#[test]
fn scaled_rate_with_queried_draw_keeps_conditioning_rate_unscaled() {
    // Querying (Pair (*. g 2) (exponential g)) with the g ~ Gamma(2,1) prior:
    // the queried draw needs g's UNSCALED posterior as its conditioning rate,
    // so we report BOTH the bare g (Gamma(2,1)) and the scaled 2·g
    // (Gamma(2,0.5)) — not a single in-place-mutated entry (the old bug).
    let entries = gamma_entries(
        "(query both (let ((g (gamma 2.0 1.0))) \
           (Posterior (Pair (*. g 2.0) (exponential g)) (True))))",
    );
    assert_eq!(
        entries.len(),
        2,
        "expected bare g + scaled 2·g, got {entries:?}"
    );
    let rates: Vec<f64> = entries
        .iter()
        .map(|g| match g {
            GammaPrior::Gamma { shape, rate, .. } => {
                assert!((shape - 2.0).abs() < 1e-9, "shape {shape}");
                *rate
            }
            other => panic!("expected Gamma, got {other}"),
        })
        .collect();
    assert!(
        rates.iter().any(|r| (r - 1.0).abs() < 1e-9),
        "expected an UNSCALED g (rate 1.0) for the draw to condition on: {rates:?}"
    );
    assert!(
        rates.iter().any(|r| (r - 0.5).abs() < 1e-9),
        "expected a scaled 2·g (rate 0.5): {rates:?}"
    );
}

#[test]
fn scaled_rate_pin_jacobian_reweights_competing_branches() {
    // Two branches of a fair flip pin the SAME g to 0.5:
    //   c=True : (real_eq (*. g 4.0) 2.0) — evidence carries the 1/4 Jacobian
    //   c=False: (real_eq g 0.5)          — no Jacobian
    // A single-world posterior would normalize the Jacobian away, but here it
    // competes: P(True) ∝ 0.5·d·(1/4), P(False) ∝ 0.5·d, so
    //   P(True) = (1/4)/(1 + 1/4) = 0.2.
    let p_true = posterior_prob(
        "(query c (let ((c (flip 0.5)) (g (gamma 2.0 1.0))) \
           (Posterior c (if c (real_eq (*. g 4.0) 2.0) (real_eq g 0.5)))))",
        "True",
    );
    assert!(
        (p_true - 0.2).abs() < 1e-9,
        "P(True) = {p_true}, expected 0.2 (scaled branch downweighted by 1/4)"
    );
}
