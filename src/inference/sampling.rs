//! Posterior-sampling driver.
//! One `Assignment` flows through, `SampleState` is
//! consulted by name, and BDD-level sampling lives on `LazyKCState`.

use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use rand::Rng;

use super::conjugate_pairs::exponential::ExponentialPrior;
// Re-export at the historical path; the enum lives with the gamma family.
pub use super::conjugate_pairs::gamma::GammaDrawFamily;
use super::conjugate_pairs::poisson::PoissonPrior;
use super::conjugate_pairs::{Assignment, ContVarName, PriorRegistry};
use super::lazy_kc::compile::CompileMode;
use super::lazy_kc::ctx::CompilerCtx;
use super::lazy_kc::state::{support_vars, CallstackKey, LazyKCState, World};
use crate::discrete_factorizations::{
    BooleanFactorization, BooleanFunction, BooleanFunctionOps, VarId,
};
use crate::inference::lazy_kc::state::GuardedWorlds;
use crate::language::types::Symbol;
use crate::language::values::*;

impl GammaDrawFamily {
    /// The symbolic `PluckValue` for a draw of this family at rate
    /// `scale * gamma`.
    pub fn mk_value(self, gamma: ContVarName, draw: ContVarName, scale: f64) -> PluckVal {
        match self {
            GammaDrawFamily::Poisson => mk_poisson(gamma, draw, scale),
            GammaDrawFamily::Exponential => mk_exponential(gamma, draw, scale),
        }
    }
}

/// Per-sample state for BDD-constrained forward evaluation.
///
/// Holds the sampled-evidence BDD as the constraint plus the pre-sampled
/// continuous values that flips / observations consult by name. The
/// `LazyKCState` is passed alongside as a separate parameter throughout,
/// rather than folded into this struct, to avoid conflicting with other
/// call sites' existing `&mut LazyKCState` argument.
pub struct SampleState {
    /// Per-sample discrete constraint: a **growing conjunction of
    /// literals**. Starts as the evidence-path literals from
    /// `sample_discrete_given` and accumulates every on-the-fly draw
    /// (`resolve_flip` free draws, `resolve_observation` false-defaults,
    /// `extend_constraint_over` extensions), so every later meeting of the
    /// same variable — by key or through a guard's support — reads off the
    /// same decision. Invariant: only literals are ever AND-ed in; a
    /// non-literal guard would bias later prior-draws of its variables.
    pub constraint: BooleanFunction,
    /// Pre-sampled continuous values (Beta + Gaussian + Dirichlet).
    pub assignment: Assignment,
    /// Per-sample trace cache: `(callstack_hash, discriminator) -> bool`.
    /// Lets a callstack key revisit return the same answer within a
    /// single sample.
    pub trace: HashMap<CallstackKey, bool>,
    /// Per-sample thunk memo: thunk Rc-identity (pointer, as `usize`) ->
    /// `(keepalive, result)`. Sample mode bypasses the KC thunk cache
    /// (its results are single-world collapses valid only for one
    /// constraint/assignment), but within a *single* sample a thunk always
    /// collapses to the same value. Memoizing by identity keeps a
    /// thunk referenced N times from being re-evaluated N times; without it
    /// a generative recursion whose body references its argument `b` times
    /// costs `O(b^depth)` and never finishes. `keepalive` holds the key thunk's
    /// `Rc` so its allocation can't be freed and its pointer reused by a
    /// different thunk within the sample.
    pub memo: HashMap<usize, (PluckVal, PluckVal)>,
    /// True constructor symbol.
    pub true_sym: Symbol,
}

impl SampleState {
    /// Make sure `names` are covered by this sample's continuous
    /// assignment, drawing any missing one from its prior in `registry`.
    ///
    /// The per-sample assignment is drawn up front over every prior
    /// registered *so far*, but forward (Sample-mode) execution can reach a
    /// `beta` / `gaussian` / `dirichlet` whose first registration happens
    /// mid-sample (e.g. inside a function body never compiled before).
    /// Such a variable cannot be referenced by the evidence — the evidence
    /// was compiled earlier and callstack-keyed dedup would have returned
    /// the already-registered name — so its posterior equals its prior and
    /// a fresh prior draw is exact, mirroring the on-the-fly rule for
    /// discrete choices. Later samples find the variable already registered
    /// and cover it in the up-front draw.
    pub fn ensure_continuous(
        &mut self,
        names: impl IntoIterator<Item = ContVarName>,
        registry: &PriorRegistry,
    ) {
        let scope = self.assignment.scope();
        let missing: BTreeSet<ContVarName> =
            names.into_iter().filter(|n| !scope.contains(n)).collect();
        if missing.is_empty() {
            return;
        }
        let drawn = registry.slice(&missing).sample(&mut rand::thread_rng());
        self.assignment = self.assignment.merge(&drawn);
    }

    /// Make sure `draw` is covered by this sample's assignment, drawing
    /// it from its conditional given the assignment's rate if missing.
    ///
    /// Draws are deliberately not registry-registered (see
    /// `GammaSuffStat::lookup_prior_in`), so `ensure_continuous` can
    /// never cover them. Evidence-observed draws are realized up front
    /// by the posterior leaf's truncated conditionals
    /// (`JointGammaPrior::sample`); a draw missing here was never
    /// observed, so its posterior conditional equals the prior
    /// conditional given λ and a fresh draw is exact. Callers must
    /// invoke this before anything reads the draw from the assignment
    /// (`force_symbolic`, interval-flip indicators).
    pub fn ensure_gamma_draw(
        &mut self,
        rate: ContVarName,
        draw: ContVarName,
        family: GammaDrawFamily,
        scale: f64,
    ) {
        if self.assignment.scope().contains(&draw) {
            return;
        }
        // The draw's effective rate is `scale * g`.
        let lambda = self.assignment.gamma_value(rate)
            * ordered_float::NotNan::new(scale).expect("gamma scale must be finite");
        let rng = &mut rand::thread_rng();
        let sampled = match family {
            GammaDrawFamily::Poisson => {
                PoissonPrior::unconstrained(std::iter::once(draw)).sample(lambda, rng)
            }
            GammaDrawFamily::Exponential => {
                ExponentialPrior::unconstrained(std::iter::once(draw)).sample(lambda, rng)
            }
        };
        let single = Assignment::from_sorted(vec![], vec![], sampled.into_iter().collect(), vec![]);
        self.assignment = self.assignment.merge(&single);
    }

    pub fn check_cache(&self, val: &PluckVal) -> Option<GuardedWorlds> {
        if let Some((_, result)) = self.memo.get(&(Rc::as_ptr(val) as usize)) {
            Some((
                vec![(result.clone(), BooleanFunction::true_ptr())],
                BooleanFunction::true_ptr(),
            ))
        } else {
            None
        }
    }

    pub fn set_cache(&mut self, val: &PluckVal, worlds: &GuardedWorlds) {
        // Sample mode is single-world; cache that value keyed by the
        // thunk's `Rc` identity, holding the `Rc` alive (`val.clone()`)
        // so its pointer can't be reused by a later thunk this sample.
        if let Some((world_val, _)) = worlds.0.first() {
            self.memo
                .insert(Rc::as_ptr(val) as usize, (val.clone(), world_val.clone()));
        }
    }
}

/// Top-level posterior sampling for a `PosteriorSamples` query.
///
/// Args: `[query_thunk, evidence_thunk, n_thunk]`. Returns `n`
/// fully-forced posterior samples.
pub fn posterior_sample(
    args: &[PluckVal],
    ctx: &mut CompilerCtx,
    o_sym: Symbol,
    s_sym: Symbol,
) -> Vec<PluckVal> {
    assert!(
        args.len() >= 3,
        "PosteriorSamples requires 3 args: query, evidence, num_samples"
    );
    let query_thunk = &args[0];
    let evidence_thunk = &args[1];
    let n_thunk = &args[2];

    // KC-compile the evidence thunk to get its worlds.
    let evidence_results = {
        let mut kc_mode = CompileMode::KC {
            path_condition: ctx.state.true_bf.clone(),
        };
        ctx.evaluate_thunk(evidence_thunk, &mut kc_mode).0
    };

    let n_samples = force_n_samples(n_thunk, o_sym, s_sym, ctx);

    // Find the True-guarded evidence BDD.
    let true_sym = ctx.true_sym;
    let evidence_bf = evidence_results
        .iter()
        .find_map(|(result, bf)| match result.as_ref() {
            PluckValue::Value { constructor, args }
                if *constructor == true_sym && args.is_empty() && !bf.is_false() =>
            {
                Some(bf.clone())
            }
            _ => None,
        });
    let evidence_bf = match evidence_bf {
        Some(bf) => bf,
        None => {
            eprintln!("PosteriorSamples: evidence has zero probability; cannot sample.");
            return Vec::new();
        }
    };

    // Sample loop. Pre-compute the evidence SPN once.
    let evidence_spn = ctx.state.wmc_spn(evidence_bf.clone());
    let mut samples = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..n_samples {
        let mut assignment = ctx
            .state
            .sample_continuous_posterior(&evidence_spn, &mut rng);
        // Draws observed only in unsampled mixture branches are absent from
        // the posterior sample; cover them before weight evaluation.
        ctx.state
            .cover_support_gamma_draws(evidence_bf.clone(), &mut assignment, &mut rng);
        let sampled_bf =
            ctx.state
                .sample_discrete_given(evidence_bf.clone(), &assignment, &mut rng);
        if sampled_bf.is_false() {
            eprintln!("PosteriorSamples: sampled BDD is False; skipping sample.");
            continue;
        }

        let mut state = SampleState {
            constraint: sampled_bf,
            assignment,
            trace: HashMap::new(),
            memo: HashMap::new(),
            true_sym: ctx.true_sym,
        };
        let forced = force_value(query_thunk, &mut state, ctx);
        samples.push(forced);
    }

    samples
}

/// Resolve a flip outcome under the constraint.
///
/// Order of resolution:
///   1. Per-sample trace cache (so revisits return the same answer).
///   2. Constraint BDD via `LazyKCState::constraint_forces`.
///   3. Otherwise sample freely with probability `p` — and record the drawn
///      literal in `state.constraint`, so guard-level meetings of the same
///      variable (`select_world` / `extend_constraint_over`) see the draw.
///      Trace and constraint always agree: a free draw writes both.
///
/// Callers must register the flip variable (`current_address_weighted`)
/// under `key` before calling, so the drawn literal can be looked up in
/// `var_of_callstack`.
pub fn resolve_flip(
    p: f64,
    key: CallstackKey,
    state: &mut SampleState,
    kc_state: &LazyKCState,
) -> bool {
    if let Some(&hit) = state.trace.get(&key) {
        return hit;
    }
    if let Some(forced) = kc_state.constraint_forces(state.constraint.clone(), key.0, key.1) {
        state.trace.insert(key, forced);
        return forced;
    }
    let outcome = rand::thread_rng().gen::<f64>() < p;
    state.trace.insert(key, outcome);
    let (_, var_bf) = kc_state
        .var_of_callstack
        .get(&key)
        .expect("resolve_flip: flip variable not registered before resolution");
    let builder = kc_state.fac();
    let lit = if outcome {
        var_bf.clone()
    } else {
        builder.negate(var_bf)
    };
    state.constraint = builder.and(&state.constraint, &lit);
    outcome
}

/// Resolve a measure-zero observation (`ProbEq` / `RealEq`) under the
/// constraint. Looks up the variable in `observation_vars` keyed by
/// callstack hash alone. Observations default to false if the constraint
/// doesn't pin them; the `¬var` default is recorded in the constraint so
/// guard-level meetings of the same variable agree (the same rule
/// `extend_constraint_over` applies to observation vars found in guard
/// supports).
pub fn resolve_observation(state: &mut SampleState, kc_state: &LazyKCState) -> bool {
    let hash = kc_state.callstack.hash();
    if let Some(forced) = kc_state.constraint_forces_observation(state.constraint.clone(), hash) {
        return forced;
    }
    if let Some((_, var_bf)) = kc_state.observation_vars.get(&hash) {
        let builder = kc_state.fac();
        state.constraint = builder.and(&state.constraint, &builder.negate(var_bf));
    }
    false
}

/// Pin every support variable of `bfs` in `state.constraint`:
/// already-pinned variables are kept; observation variables are resolved
/// false (the same default as `resolve_observation`); everything else is
/// drawn from its prior weight evaluated at the continuous assignment
/// (`var_sample_prob`). Every newly resolved literal is AND-ed into the
/// constraint, so later meetings of the same variable read off the same
/// decision.
///
/// Invariant: only literals ever enter the constraint — AND-ing a
/// non-literal guard (e.g. `a ∨ b`) would bias a later prior-draw of `a`.
fn extend_constraint_over(
    bfs: &[BooleanFunction],
    state: &mut SampleState,
    kc_state: &LazyKCState,
) {
    let mut support: BTreeSet<usize> = BTreeSet::new();
    for bf in bfs {
        support_vars(kc_state.fac(), bf, &mut support);
    }
    // Guard supports can reference gamma-event flips whose draws neither
    // the evidence scope nor forward execution has realized yet; cover
    // them before `var_sample_prob` evaluates the weights.
    kc_state.realize_missing_gamma_draws(
        support.iter().copied(),
        &mut state.assignment,
        &mut rand::thread_rng(),
    );
    let builder = kc_state.fac();
    for label in support {
        let var_bf = builder.var(VarId(label as u64), true);
        if kc_state
            .implies(state.constraint.clone(), var_bf.clone())
            .is_some()
        {
            continue; // already pinned by the constraint
        }
        let polarity = if kc_state.observation_var_labels.contains(&label) {
            false
        } else {
            let p = kc_state.var_sample_prob(label, &state.assignment);
            rand::thread_rng().gen::<f64>() < p
        };
        let lit = if polarity {
            var_bf
        } else {
            builder.negate(&var_bf)
        };
        state.constraint = builder.and(&state.constraint, &lit);
    }
}

/// Resolve guarded `worlds` to the single world consistent with the
/// constraint, prior-sampling any free choice the guards depend on.
///
/// 0 consistent worlds → panic (the sampled path met shaved error mass);
/// 1 → take it; >1 → extend the constraint over the support of the
/// *consistent* guards only (AND-ing literals can never revive an
/// inconsistent world), re-filter, and take the unique survivor.
///
/// Sole consumer: ThunkUnion selection in `evaluate_thunk_union` — the one
/// place Sample-mode evaluation still meets multiple guarded alternatives
/// (KC-built unions reached through shared env values). Everything else in
/// Sample mode is single-world by construction (asserted at the bind
/// boundary, `sample_compile_one`, and `force_value`).
pub(crate) fn select_world(
    worlds: Vec<World>,
    state: &mut SampleState,
    kc_state: &LazyKCState,
) -> PluckVal {
    let builder = kc_state.fac();
    let mut consistent: Vec<World> = worlds
        .into_iter()
        .filter(|(_, guard)| !builder.and(&state.constraint, guard).is_false())
        .collect();
    if consistent.len() > 1 {
        let guards: Vec<BooleanFunction> = consistent.iter().map(|(_, g)| g.clone()).collect();
        extend_constraint_over(&guards, state, kc_state);
        consistent.retain(|(_, guard)| !builder.and(&state.constraint, guard).is_false());
    }
    match consistent.len() {
        1 => consistent.pop().unwrap().0,
        0 => panic!(
            "Sample mode: no world consistent with the sampled constraint \
             (program error along the sampled path)"
        ),
        n => panic!(
            "select_world: {} worlds remain consistent after pinning every \
             guard variable — world guards are not disjoint",
            n
        ),
    }
}

/// Force every thunk in `val` and resolve every symbolic leaf against the
/// sampled world, returning a value with no `Thunk` / `ThunkUnion` /
/// non-concrete `IntDist` leaves and no symbolic `Probability` /
/// `GaussianExpr` / `DirichletProbability` / `DirichletVector` nodes.
///
/// With Sample-mode propagation in `bind_compile`, thunk evaluation
/// returns exactly one world (asserted): multi-world KC artifacts are
/// resolved before they surface here — ThunkUnions inside
/// `evaluate_thunk_union`, everything else at the bind boundary — and
/// IntDists arrive with constant bits.
pub fn force_value(val: &PluckVal, state: &mut SampleState, ctx: &mut CompilerCtx) -> PluckVal {
    match val.as_ref() {
        PluckValue::Thunk(_) | PluckValue::ThunkUnion(_) => {
            // Evaluate in Sample mode: exactly one world. Zero worlds means
            // the sampled path hit shaved error mass (failed pattern match,
            // `(error)`, applying a non-function) — a program error per the
            // error-mass policy — or an inference limit if `hit_limit`.
            let evaluated = {
                let mut mode = CompileMode::Sample(state);
                let (worlds, _) = ctx.evaluate_thunk(val, &mut mode);
                if worlds.is_empty() {
                    if ctx.state.hit_limit {
                        panic!("Sample mode: inference limit (max_depth / time_limit) hit");
                    }
                    panic!(
                        "Sample mode: thunk evaluation produced no worlds \
                         (program error along the sampled path)"
                    );
                }
                assert_eq!(
                    worlds.len(),
                    1,
                    "Sample mode: thunk evaluation returned {} worlds; \
                     expected exactly one",
                    worlds.len()
                );
                worlds.into_iter().next().unwrap().0
            };
            force_value(&evaluated, state, ctx)
        }
        PluckValue::Value { constructor, args } => {
            let forced_args: Vec<PluckVal> = args
                .iter()
                .map(|arg| force_value(arg, state, ctx))
                .collect();
            mk_value(*constructor, forced_args)
        }
        PluckValue::FloatMatrix { entries, shape } => {
            let forced_entries: Vec<PluckVal> =
                entries.iter().map(|e| force_value(e, state, ctx)).collect();
            crate::language::values::mk_float_matrix(forced_entries, shape.clone())
        }
        PluckValue::IntDist { bits } => {
            // Sample-mode compilation only produces constant-bit IntDists
            // (mk_int, the uniform_int{,_range} sample walkers, add/sub
            // over constants); symbolic bits would mean a KC-built value
            // leaked into Sample mode.
            assert!(
                bits.iter().all(|b| b.is_true() || b.is_false()),
                "force_value: IntDist with symbolic bits in Sample mode"
            );
            val.clone()
        }
        // The symbolic-continuous variants route through one method.
        v if v.is_symbolic() => {
            // Gamma-family draws may be uncovered when a KC-built value
            // reaches Sample mode through ThunkUnion selection without
            // `compile_gamma_consumer` re-running this sample; cover
            // them before reading the assignment.
            match v {
                PluckValue::Poisson { name, gamma, scale } => {
                    state.ensure_gamma_draw(*gamma, *name, GammaDrawFamily::Poisson, *scale)
                }
                PluckValue::Exponential { name, gamma, scale } => {
                    state.ensure_gamma_draw(*gamma, *name, GammaDrawFamily::Exponential, *scale)
                }
                _ => {}
            }
            v.force_symbolic(&state.assignment, state.true_sym)
                .expect("is_symbolic / force_symbolic disagreement")
        }
        _ => val.clone(),
    }
}

// Extract `n`. The thunk should evaluate
// to a native int or a Peano-encoded nat; running it in Sample mode
// with true_bf constraint and an empty assignment is safe because
// such an expression has no flips / observations to resolve.
pub fn force_n_samples(
    n_thunk: &Rc<PluckValue>,
    o_sym: Symbol,
    s_sym: Symbol,
    ctx: &mut CompilerCtx,
) -> usize {
    let mut state = SampleState {
        constraint: ctx.state.true_bf.clone(),
        assignment: Assignment::from_sorted(vec![], vec![], vec![], vec![]),
        trace: HashMap::new(),
        memo: HashMap::new(),
        true_sym: ctx.true_sym,
    };
    let n_val = force_value(n_thunk, &mut state, ctx);
    extract_nat(&n_val, o_sym, s_sym).unwrap_or_else(|| {
        panic!(
            "PosteriorSamples: third argument must be an integer, got {:?}",
            n_val
        )
    }) as usize
}
