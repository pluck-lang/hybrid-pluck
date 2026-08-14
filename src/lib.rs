//! Pluck library entry point.
//!
//! The shape: a `PluckContext` parses + runs `.pluck` source, returning
//! `Vec<QueryResult>`. Distribution-style queries (Marginal, Posterior)
//! flatten to a `Vec<MixtureItem>` carrying display strings and
//! conjugate posteriors per component; sample-style queries
//! (PosteriorSamples) return a flat `Vec<String>`.
//!
//! Marginal vs Posterior differ only in which expression we compile —
//! Posterior wraps the model in `(given evidence model)`. After
//! compilation the pipelines are identical.

pub(crate) mod discrete_factorizations;
pub(crate) mod inference;
pub(crate) mod language;
pub(crate) mod utils;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// The crate's public API. The module tree above is crate-internal
// (`pub(crate)`); everything external consumers (the `pluck` binary and the
// integration tests) need is re-exported here at the crate root.
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub use inference::conjugate_pairs::{
    BetaPrior, DirichletPrior, GammaPrior, GaussianPrior, JointGammaPrior,
};
use inference::conjugate_pairs::{
    ContVarName, EvidenceLeaf, ExponentialObs, GammaSuffStat, GaussianAffineExpr, PoissonObs,
    PriorRegistry,
};
use inference::lazy_kc::ctx::CompilerCtx;
use inference::lazy_kc::state::{LazyKCConfig, LazyKCState};
use inference::sampling::posterior_sample;
use inference::spn::Spn;
use itertools::Itertools;
use language::define::DefinitionRegistry;
use language::pexpr::PExpr;
use language::toplevel::parse_toplevel;
use language::types::{StringInterner, TypeRegistry};
use language::values::{empty_env, env_cons, PluckVal, PluckValue};
use nalgebra as na;
pub use utils::display_precision::weight_precision;
use utils::epsilon::RealEps;
use utils::formatting::{format_value, NamingScheme};
use utils::intervals::IntervalOrEq;
pub use utils::math::logsumexp;
use utils::stats::{KcStats, StageTimes};
use utils::union_find::UnionFind;

use crate::discrete_factorizations::BooleanFunction;
use crate::language::values::EnvInner;

// ════════════════════════════════════════════════════════════════════
// 1. PluckContext
// ════════════════════════════════════════════════════════════════════

fn build_resolver(file_registry: &HashMap<String, String>) -> impl Fn(&str) -> Option<String> + '_ {
    |path: &str| {
        if let Some(content) = file_registry.get(path) {
            return Some(content.clone());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Ok(content) = std::fs::read_to_string(path) {
                return Some(content);
            }
        }
        None
    }
}

/// Top-level Pluck session. Owns interner, types, defs, and the
/// include resolver's file registry.
pub struct PluckContext {
    pub interner: StringInterner,
    pub types: TypeRegistry,
    pub defs: DefinitionRegistry,
    pub file_registry: HashMap<String, String>,
    syms: Syms,
    /// Wall-clock time to parse the last program run through
    /// [`run_with_progress`](PluckContext::run_with_progress). Excludes the
    /// stdlib parse (that happens once in [`load_stdlib`](PluckContext::load_stdlib)
    /// and is deliberately not timed). Surfaced by the `PLUCK_STATS`
    /// instrumentation and folded into the reported inference total.
    parse_time: Duration,
}

// TODO - have all the different compilers/ctx reference this symbols struct
// instead of having their own symbols
struct Syms {
    true_sym: u32,
    false_sym: u32,
    marginal_sym: u32,
    posterior_sym: u32,
    posterior_samples_sym: u32,
    gibbs_sym: u32,
    o_sym: u32,
    s_sym: u32,
    nil_sym: u32,
    cons_sym: u32,
    with_prior_sample_sym: u32,
    with_constant_sym: u32,
}

impl PluckContext {
    pub fn new() -> Self {
        let mut interner = StringInterner::new();
        let mut types = TypeRegistry::new();
        types.register_builtins(&mut interner);

        let syms = Syms {
            true_sym: interner.intern("True"),
            false_sym: interner.intern("False"),
            marginal_sym: interner.intern("Marginal"),
            posterior_sym: interner.intern("Posterior"),
            posterior_samples_sym: interner.intern("PosteriorSamples"),
            gibbs_sym: interner.intern("Gibbs"),
            o_sym: interner.intern("O"),
            s_sym: interner.intern("S"),
            nil_sym: interner.intern("Nil"),
            cons_sym: interner.intern("Cons"),
            with_prior_sample_sym: interner.intern("WithPriorSample"),
            with_constant_sym: interner.intern("WithConstant"),
        };

        let mut ctx = PluckContext {
            interner,
            types,
            defs: DefinitionRegistry::new(),
            file_registry: HashMap::new(),
            syms,
            parse_time: Duration::ZERO,
        };
        ctx.load_stdlib();
        ctx
    }

    fn load_stdlib(&mut self) {
        let stdlib_source = include_str!("../stdlib.pluck");
        let resolver = build_resolver(&self.file_registry);
        parse_toplevel(
            stdlib_source,
            &mut self.interner,
            &mut self.types,
            &mut self.defs,
            &resolver,
        );
    }

    pub fn register_file(&mut self, name: String, content: String) {
        self.file_registry.insert(name, content);
    }

    pub fn run(&mut self, source: &str) -> Vec<QueryResult> {
        self.run_with_progress(source, None)
    }

    /// Like [`run`](Self::run), but a [`GibbsProgress`] handle can be supplied
    /// so a caller can poll a live sample count while a Gibbs chain runs. The
    /// sampler does no rendering; it only bumps the handle's counter.
    pub fn run_with_progress(
        &mut self,
        source: &str,
        progress: Option<&GibbsProgress>,
    ) -> Vec<QueryResult> {
        let queries = {
            let resolver = build_resolver(&self.file_registry);
            // Time parsing of the actual program only. The stdlib is parsed
            // separately in `load_stdlib` and is intentionally excluded.
            let parse_start = std::time::Instant::now();
            let queries = parse_toplevel(
                source,
                &mut self.interner,
                &mut self.types,
                &mut self.defs,
                &resolver,
            );
            self.parse_time = parse_start.elapsed();
            queries
        };

        let mut results = Vec::new();
        for query in queries {
            results.push(self.process_query(&query.expr, &query.display_str, progress));
        }
        results
    }

    fn get_query_val(&self, expr: &PExpr, state: &mut LazyKCState) -> Option<Rc<PluckValue>> {
        let env = empty_env();
        let mut ctx = CompilerCtx {
            types: &self.types,
            defs: &self.defs,
            true_sym: self.syms.true_sym,
            false_sym: self.syms.false_sym,
            state,
        };
        let mut mode = inference::lazy_kc::compile::CompileMode::KC {
            path_condition: ctx.state.true_bf.clone(),
        };
        let (worlds, _) = ctx.traced_compile(expr, &env, &mut mode, 0);

        if worlds.len() != 1 {
            eprintln!(
                    "Query must evaluate to a single (Marginal …) / (Posterior …) / (PosteriorSamples …) value, got {} worlds",
                    worlds.len()
                );
            return None;
        }

        Some(worlds.into_iter().next().unwrap().0)
    }

    fn process_query(
        &mut self,
        expr: &PExpr,
        display_str: &str,
        progress: Option<&GibbsProgress>,
    ) -> QueryResult {
        inference::spn::evidence::clear_intern_table();
        inference::spn::evidence::clear_mul_cache();
        let mut state = LazyKCState::new(LazyKCConfig::default());
        #[cfg(not(target_arch = "wasm32"))]
        {
            state.start_time = Some(std::time::Instant::now());
        }

        // Symbolic execution (program → BDD) starts here. `get_query_val`
        // compiles the query wrapper; the distribution handlers compile the
        // model/evidence and add their own compile time to this prefix.
        let sym_start = std::time::Instant::now();
        let query_val = match self.get_query_val(expr, &mut state) {
            Some(val) => val,
            None => {
                return self.empty_distribution(display_str);
            }
        };
        let sym_pre = sym_start.elapsed();

        let kind = match query_val.as_ref() {
            PluckValue::Value { constructor, args } if *constructor == self.syms.marginal_sym => {
                self.process_marginal_query(args, &mut state, sym_pre)
            }
            PluckValue::Value { constructor, args } if *constructor == self.syms.posterior_sym => {
                self.process_posterior_query(args, &mut state, sym_pre)
            }
            PluckValue::Value { constructor, args }
                if *constructor == self.syms.posterior_samples_sym =>
            {
                self.process_posterior_samples_query(args, &mut state)
            }
            PluckValue::Value { constructor, args } if *constructor == self.syms.gibbs_sym => {
                self.process_gibbs_query(args, &mut state, progress)
            }
            _ => {
                let val_str = format_value(&query_val, &self.interner, None);
                eprintln!(
                    "Expected Marginal / Posterior / PosteriorSamples query, got: {}",
                    val_str
                );
                ResultKind::Distribution { items: Vec::new() }
            }
        };

        QueryResult {
            query: display_str.to_string(),
            kind,
        }
    }

    fn process_marginal_query(
        &self,
        args: &[PluckVal],
        state: &mut LazyKCState,
        sym_pre: Duration,
    ) -> ResultKind {
        if args.is_empty() {
            return ResultKind::Distribution { items: Vec::new() };
        }
        let thunk = args[0].clone();
        let compile_start = std::time::Instant::now();
        let full = {
            let mut ctx = CompilerCtx {
                types: &self.types,
                defs: &self.defs,
                true_sym: self.syms.true_sym,
                false_sym: self.syms.false_sym,
                state,
            };
            let mut mode = inference::lazy_kc::compile::CompileMode::KC {
                path_condition: ctx.state.true_bf.clone(),
            };
            let (thunk_worlds, _) = ctx.evaluate_thunk(&thunk, &mut mode);
            ctx.infer_full_distribution(thunk_worlds)
        };
        let symbolic = sym_pre + compile_start.elapsed();
        self.run_distribution(full, state, symbolic)
    }

    // A Posterior model evidence query arrives as a PluckValue::Value whose args
    // are already-evaluated PluckVals representing model thunk and evidence thunk
    // We want to apply given (Bayesian conditioning defined in stdlib.pluck) to these two values.
    // The problem: given is a top-level definition that the compiler invokes via a PExpr::Defined
    // node — i.e., it needs a PExpr tree, not raw PluckVals. So stuff the two PluckVals into the
    // environment under fresh names, then write a PExpr that references them via PExpr::Var
    fn build_given_expr_and_env(&mut self, args: &[PluckVal]) -> Option<(PExpr, Rc<EnvInner>)> {
        let given_sym = self.interner.intern("given");
        self.defs.lookup(given_sym)?;
        let a_sym = self.interner.intern("__posterior_a");
        let b_sym = self.interner.intern("__posterior_b");
        let env = env_cons(
            a_sym,
            args[0].clone(),
            env_cons(b_sym, args[1].clone(), empty_env()),
        );
        Some((
            PExpr::App {
                func: Box::new(PExpr::App {
                    func: Box::new(PExpr::Defined { name: given_sym }),
                    arg: Box::new(PExpr::Var { name: b_sym }),
                }),
                arg: Box::new(PExpr::Var { name: a_sym }),
            },
            env,
        ))
    }

    fn process_posterior_query(
        &mut self,
        args: &[PluckVal],
        state: &mut LazyKCState,
        sym_pre: Duration,
    ) -> ResultKind {
        if args.len() < 2 {
            return ResultKind::Distribution { items: Vec::new() };
        }
        let (given_expr, env) = match self.build_given_expr_and_env(args) {
            None => {
                eprintln!("'given' not defined in stdlib — cannot process Posterior query");
                return ResultKind::Distribution { items: Vec::new() };
            }
            Some((expr, env)) => (expr, env),
        };
        let compile_start = std::time::Instant::now();
        let full = {
            let mut ctx = CompilerCtx {
                types: &self.types,
                defs: &self.defs,
                true_sym: self.syms.true_sym,
                false_sym: self.syms.false_sym,
                state,
            };
            let mut mode = inference::lazy_kc::compile::CompileMode::KC {
                path_condition: ctx.state.true_bf.clone(),
            };
            let (ret_worlds, _) = ctx.traced_compile(&given_expr, &env, &mut mode, 0);
            ctx.infer_full_distribution(ret_worlds)
        };
        let symbolic = sym_pre + compile_start.elapsed();
        self.run_distribution(full, state, symbolic)
    }

    fn process_posterior_samples_query(
        &self,
        args: &[PluckVal],
        state: &mut LazyKCState,
    ) -> ResultKind {
        if args.len() < 3 {
            return ResultKind::Distribution { items: Vec::new() };
        }
        self.run_samples(args, state)
    }

    fn process_gibbs_query(
        &mut self,
        args: &[PluckVal],
        state: &mut LazyKCState,
        progress: Option<&GibbsProgress>,
    ) -> ResultKind {
        if args.len() < 5 {
            eprintln!(
                "Gibbs requires 5 args (query, cond, num_samples, blocks_list, initialization); got {}",
                args.len()
            );
            return ResultKind::Distribution { items: Vec::new() };
        }
        self.run_gibbs(args, state, progress)
    }

    fn empty_distribution(&self, display_str: &str) -> QueryResult {
        QueryResult {
            query: display_str.to_string(),
            kind: ResultKind::Distribution { items: Vec::new() },
        }
    }

    fn run_distribution(
        &self,
        worlds: Vec<(PluckVal, BooleanFunction)>,
        state: &LazyKCState,
        symbolic: Duration,
    ) -> ResultKind {
        let items = distribution_items(
            &worlds,
            state,
            &self.interner,
            &self.types,
            symbolic,
            self.parse_time,
        );
        ResultKind::Distribution { items }
    }

    fn run_samples(&self, args: &[PluckVal], state: &mut LazyKCState) -> ResultKind {
        let mut ctx = CompilerCtx {
            types: &self.types,
            defs: &self.defs,
            true_sym: self.syms.true_sym,
            false_sym: self.syms.false_sym,
            state,
        };
        let samples = posterior_sample(args, &mut ctx, self.syms.o_sym, self.syms.s_sym);
        let values = samples
            .iter()
            .map(|s| format_value(s, &self.interner, None))
            .collect();
        ResultKind::Samples { values }
    }

    fn run_gibbs(
        &mut self,
        args: &[PluckVal],
        state: &mut LazyKCState,
        progress: Option<&GibbsProgress>,
    ) -> ResultKind {
        let mut ctx = CompilerCtx {
            types: &self.types,
            defs: &self.defs,
            true_sym: self.syms.true_sym,
            false_sym: self.syms.false_sym,
            state,
        };
        let symbols =
            inference::gibbs::GibbsCtx::generate_gibbs_symbols(&self.syms, &mut self.interner);
        let outcome = match inference::gibbs::GibbsCtx::new(args, &mut ctx, symbols) {
            Ok(runner) => {
                // Publish the total and reset the counter so the CLI's monitor
                // renders this query's progress from zero.
                let runner = match progress {
                    Some(p) => {
                        p.total.store(runner.num_samples(), Ordering::Relaxed);
                        p.current.store(0, Ordering::Relaxed);
                        runner.with_progress(p.current.clone())
                    }
                    None => runner,
                };
                runner.run(&mut ctx)
            }
            Err(msg) => {
                eprintln!("Gibbs: {msg}");
                inference::gibbs::GibbsOutcome::Stuck {
                    samples: Vec::new(),
                }
            }
        };
        let samples = match outcome {
            inference::gibbs::GibbsOutcome::Complete { samples } => samples,
            inference::gibbs::GibbsOutcome::Stuck { samples } => {
                eprintln!(
                    "Gibbs: chain stalled after {} samples (cond ∧ pin unsatisfiable)",
                    samples.len()
                );
                samples
            }
        };
        let values = samples
            .iter()
            .map(|s| format_value(s, &self.interner, None))
            .collect();
        ResultKind::Samples { values }
    }
}

impl Default for PluckContext {
    fn default() -> Self {
        Self::new()
    }
}

// ════════════════════════════════════════════════════════════════════
// 2. Client-facing result types
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueryResult {
    pub query: String,
    pub kind: ResultKind,
}

/// Shared progress handle for a running Gibbs chain. Pass via
/// [`PluckContext::run_with_progress`]; the sampler bumps `current` to the
/// running sample count and `run_gibbs` publishes `total`. A caller (e.g. the
/// CLI) polls these to render progress — the sampler itself never renders.
#[derive(Debug, Clone)]
pub struct GibbsProgress {
    pub current: Arc<AtomicUsize>,
    pub total: Arc<AtomicUsize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "result_kind", rename_all = "snake_case")]
pub enum ResultKind {
    Distribution { items: Vec<MixtureItem> },
    Samples { values: Vec<String> },
}

/// One mixture component. The library returns the flat mixture and
/// lets clients group as needed. Two items can share the same `display`
/// when the discrete value is reachable along distinct BDD paths whose
/// posteriors over continuous variables differ.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MixtureItem {
    pub display: String,
    pub log_probability: f64,
    pub posteriors: PosteriorPacket,
}

/// Bundled posteriors for one mixture component.
///
/// Posteriors are the same `BetaPrior` / `DirichletPrior` /
/// `GaussianPrior` types used internally by the SPN, instantiated with
/// `String` names. One `GaussianPrior` per independent block of affine
/// expressions: independent expressions land in separate blocks;
/// expressions sharing underlying Gaussian variables share a block
/// whose `var_order` is their display names.
pub type PosteriorPacket = PriorRegistry<String>;

// ════════════════════════════════════════════════════════════════════
// 3. Worlds → MixtureItems
// ════════════════════════════════════════════════════════════════════

pub type Path = Vec<(u32, usize)>;

#[derive(Default, Debug)]
pub struct ResultPositions {
    // BTreeMap (not HashMap) so downstream naming/display order is deterministic across runs
    gauss_path_to_expr: BTreeMap<Path, GaussianAffineExpr>,
    beta_path_to_var: BTreeMap<Path, ContVarName>,
    dir_path_to_var: BTreeMap<Path, ContVarName>,
    /// Each gamma-family query position → `(variable, scale)`. `scale` is the
    /// gamma scale factor (`1.0` for an unscaled gamma or any draw); a queried
    /// scaled gamma `s·g` reports its *scaled* distribution `Gamma(shape, rate/s)`.
    gamma_path_to_var: BTreeMap<Path, (ContVarName, ordered_float::NotNan<f64>)>,
}

fn collect_positions(val: &PluckVal, path: &mut Path, positions: &mut ResultPositions) {
    use language::values::SymbolicPositionKind;
    if let Some(kind) = val.position_kind() {
        match kind {
            SymbolicPositionKind::Beta(name) => {
                positions.beta_path_to_var.insert(path.clone(), name);
            }
            SymbolicPositionKind::Dirichlet(name) => {
                positions.dir_path_to_var.insert(path.clone(), name);
            }
            SymbolicPositionKind::Gamma { name, scale } => {
                let s = ordered_float::NotNan::new(scale).expect("gamma scale must be finite");
                positions.gamma_path_to_var.insert(path.clone(), (name, s));
            }
            SymbolicPositionKind::Gaussian(expr) => {
                positions.gauss_path_to_expr.insert(path.clone(), expr);
            }
        }
        return;
    }
    if let PluckValue::Value { constructor, args } = val.as_ref() {
        for (i, arg) in args.iter().enumerate() {
            path.push((*constructor, i));
            collect_positions(arg, path, positions);
            path.pop();
        }
    }
    if let PluckValue::FloatMatrix { entries, .. } = val.as_ref() {
        // Treat each matrix entry like a list element so the existing
        // path-based naming machinery can produce x[0], x[1], … for
        // FloatMatrix contents the same way it does for Cons-list contents.
        // We can't reach the interner here, so synthesise the "Cons" path
        // step with a sentinel constructor symbol that NamingScheme treats
        // as a list-spine entry (see `cons_sym` lookup in NamingScheme).
        // u32::MAX is the same sentinel `format_value` uses when "Cons" is
        // missing from the interner.
        for (i, e) in entries.iter().enumerate() {
            path.push((u32::MAX, i));
            collect_positions(e, path, positions);
            path.pop();
        }
    }
}

// Collect per-position data across all worlds. Preserve
// first-encountered order for deterministic naming.
fn collect_variable_positions(worlds: &[(PluckVal, BooleanFunction)]) -> Vec<ResultPositions> {
    let mut positions_by_world = Vec::new();
    for (val, _) in worlds {
        let mut positions = ResultPositions::default();
        let mut path = Vec::new();
        collect_positions(val, &mut path, &mut positions);
        positions_by_world.push(positions);
    }
    positions_by_world
}

/// Restore priors for any variable in `scope` that the partial
/// posterior didn't touch. Variables outside `scope` are *not*
/// re-introduced — the global prior registry can hold variables that
/// were used during inference but never appear in the value tree, and
/// those have no place in the user-facing posterior.
fn fill_in_missing_variables_with_prior(
    priors: &PriorRegistry,
    partial_posterior: &PriorRegistry,
    scope: &BTreeSet<ContVarName>,
) -> PriorRegistry {
    let posterior_variables: BTreeSet<ContVarName> = partial_posterior.scope().copied().collect();
    let unchanged_variables: BTreeSet<ContVarName> = scope
        .iter()
        .filter(|n| !posterior_variables.contains(n))
        .copied()
        .collect();
    partial_posterior.merge(&priors.slice(&unchanged_variables))
}

fn distribution_items(
    worlds: &[(PluckVal, BooleanFunction)],
    state: &LazyKCState,
    interner: &StringInterner,
    types: &TypeRegistry,
    symbolic: Duration,
    parse: Duration,
) -> Vec<MixtureItem> {
    let position_data = collect_variable_positions(worlds);
    let priors = &state.prior_registry;

    // Opt-in internal BDD/SPN size + timing instrumentation (PLUCK_STATS env var).
    let mut stats = KcStats::new();
    stats.set_symbolic(symbolic);
    stats.set_parse(parse);

    let mut world_likelihoods: HashMap<(String, PosteriorPacket), RealEps> = HashMap::new();
    let mut order: Vec<(String, PosteriorPacket)> = Vec::new();

    for ((val, bf), positions) in worlds.iter().zip(position_data.iter()) {
        // Naming is per-world: a world's value tree is the universe of
        // display-named variables we can resolve. Building naming
        // globally (flattened across worlds) loses entries on same-path
        // collisions and leaks expressions from other worlds into the
        // per-world registry, both of which cause downstream panics.
        let naming_t0 = std::time::Instant::now();
        let naming = NamingScheme::derive(positions, interner, types);
        // Every scale each gamma RATE variable is queried at, as a set: a
        // gamma queried as both `2·g` and `3·g` reports BOTH (one entry per
        // scale), mirroring how the Gaussian family reports per queried
        // affine expression rather than per underlying variable.
        let mut gamma_query_scales: BTreeMap<ContVarName, BTreeSet<ordered_float::NotNan<f64>>> =
            BTreeMap::new();
        for (var, scale) in positions.gamma_path_to_var.values() {
            gamma_query_scales.entry(*var).or_default().insert(*scale);
        }
        let naming_setup = naming_t0.elapsed();
        let scope = scope_of(val);
        let t0 = std::time::Instant::now();
        let evidence = state.wmc_spn(bf.clone());
        // Cover queried Poisson/Exponential draws with vacuous
        // (probability-1) joint leaves. Registry priors deliberately
        // carry no draw families — "any SPN that references a draw must
        // contain its joint gamma leaf" (`GammaSuffStat::lookup_prior_in`)
        // — so a queried draw the evidence never observed would
        // otherwise vanish from the packet, leaving only the rate. The
        // full-support observation is the multiplicative identity
        // against any existing constraint and leaves z unchanged.
        let evidence = gamma_draw_refs(val).into_iter().fold(evidence, |spn, r| {
            spn * Spn::leaf(EvidenceLeaf::Gamma(r.vacuous_stat()))
        });
        let t1 = std::time::Instant::now();
        let posterior = evidence.posterior(priors, &scope);
        let t2 = std::time::Instant::now();
        // `.components()` is eager (allocates the whole flattened mixture up
        // front), so binding it here isolates the flatten cost from the loop.
        let comps = posterior.components();
        let t3 = std::time::Instant::now();

        let mut n_components = 0usize;
        for (weight, partial_factored_component) in comps {
            if weight.is_zero() {
                continue;
            }
            n_components += 1;
            let full_factored_component =
                fill_in_missing_variables_with_prior(priors, &partial_factored_component, &scope);
            let packet = rename_to_affine_posterior_packet(
                &full_factored_component,
                &naming,
                &gamma_query_scales,
            );
            let display = format_value(val, interner, Some(&naming));
            let key = (display, packet);
            if !world_likelihoods.contains_key(&key) {
                order.push(key.clone());
            }
            let log_likelihood = world_likelihoods.entry(key).or_insert(RealEps::zero());
            *log_likelihood = *log_likelihood + weight;
        }
        // Capture the consume-loop end explicitly (rather than `t3.elapsed()`)
        // so the recorded `display` window unambiguously excludes the size
        // walks `record_world` runs after this point.
        let t4 = std::time::Instant::now();
        stats.record_world(
            state.fac(),
            bf.clone(),
            &evidence,
            StageTimes {
                wmc: t1 - t0,
                posterior: t2 - t1,
                flatten: t3 - t2,
                display: naming_setup + (t4 - t3),
            },
            n_components,
        );
    }

    stats.report(state);

    let z: RealEps = world_likelihoods
        .values()
        .copied()
        .fold(RealEps::zero(), |a, b| a + b);

    if z.is_zero() {
        return Vec::new();
    }

    let mut items: Vec<MixtureItem> = order
        .into_iter()
        .filter_map(|key| {
            let w = *world_likelihoods
                .get(&key)
                .unwrap_or_else(|| panic!("Could not find {:?}", key));
            let normed = w / z;
            if normed.is_zero() || normed.power != 0 {
                return None;
            }
            let (display, posteriors) = key;
            Some(MixtureItem {
                display,
                log_probability: normed.log_coeff,
                posteriors,
            })
        })
        .collect();

    items.sort_by(|a, b| {
        b.log_probability
            .partial_cmp(&a.log_probability)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.display.cmp(&b.display))
            // Used to make ordering canonical
            .then_with(|| a.posteriors.to_string().cmp(&b.posteriors.to_string()))
    });

    items
}

fn scope_of(val: &PluckVal) -> BTreeSet<ContVarName> {
    let mut scope = BTreeSet::new();
    collect_scope(val, &mut scope);
    scope
}

fn collect_scope(val: &PluckVal, scope: &mut BTreeSet<ContVarName>) {
    val.scope_into(scope);
    if let PluckValue::Value { args, .. } = val.as_ref() {
        for arg in args {
            collect_scope(arg, scope);
        }
    }
    if let PluckValue::FloatMatrix { entries, .. } = val.as_ref() {
        for e in entries {
            collect_scope(e, scope);
        }
    }
}

/// A Poisson/Exponential draw referenced by a world's value tree:
/// `(rate variable, draw variable, scale, family)`. The draw's effective
/// rate is `scale · rate` (a scaled gamma), carried so the vacuous leaf
/// samples a *queried but unobserved* draw at the right rate. Ordered so
/// the vacuous leaves fold into the evidence SPN deterministically.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum GammaDrawRef {
    Poisson(ContVarName, ContVarName, ordered_float::NotNan<f64>),
    Exponential(ContVarName, ContVarName, ordered_float::NotNan<f64>),
}

impl GammaDrawRef {
    /// The vacuous (full-support, probability-1) joint observation that
    /// keeps this draw in the posterior packet, tagged with its rate scale
    /// so a sampled draw is drawn at `scale · λ`.
    fn vacuous_stat(&self) -> GammaSuffStat {
        match self {
            GammaDrawRef::Poisson(gamma, draw, scale) => GammaSuffStat::poisson(
                *gamma,
                PoissonObs::constraint(*draw, IntervalOrEq::geq(0)).with_scale(*scale),
            ),
            GammaDrawRef::Exponential(gamma, draw, scale) => GammaSuffStat::exponential(
                *gamma,
                ExponentialObs::geq(*draw, 0.0).with_scale(*scale),
            ),
        }
    }
}

/// The deduped Poisson/Exponential draws referenced by the value tree,
/// used by `distribution_items` to keep queried draws in the posterior
/// packet.
fn gamma_draw_refs(val: &PluckVal) -> BTreeSet<GammaDrawRef> {
    let mut out = BTreeSet::new();
    collect_gamma_draw_refs(val, &mut out);
    out
}

fn collect_gamma_draw_refs(val: &PluckVal, out: &mut BTreeSet<GammaDrawRef>) {
    match val.as_ref() {
        PluckValue::Poisson { name, gamma, scale } => {
            let s = ordered_float::NotNan::new(*scale).expect("gamma scale must be finite");
            out.insert(GammaDrawRef::Poisson(*gamma, *name, s));
        }
        PluckValue::Exponential { name, gamma, scale } => {
            let s = ordered_float::NotNan::new(*scale).expect("gamma scale must be finite");
            out.insert(GammaDrawRef::Exponential(*gamma, *name, s));
        }
        PluckValue::Value { args, .. } => {
            for arg in args {
                collect_gamma_draw_refs(arg, out);
            }
        }
        PluckValue::FloatMatrix { entries, .. } => {
            for e in entries {
                collect_gamma_draw_refs(e, out);
            }
        }
        _ => {}
    }
}

// ════════════════════════════════════════════════════════════════════
// 4. Build a PosteriorPacket from a posterior component
// ════════════════════════════════════════════════════════════════════

/// Re-tag a posterior `PriorRegistry<ContVarName>` as a client-facing
/// `PosteriorPacket = PriorRegistry<String>` using `naming`, and rewrite
/// its Gaussian section from the underlying-variable basis into the
/// basis of the affine expressions the program queried.
fn rename_to_affine_posterior_packet(
    registry: &PriorRegistry,
    naming: &NamingScheme,
    gamma_query_scales: &BTreeMap<ContVarName, BTreeSet<ordered_float::NotNan<f64>>>,
) -> PosteriorPacket {
    let beta = registry
        .beta
        .iter()
        .map(|b| {
            let name = b.name();
            let display = naming.var_names.get(&name).unwrap_or_else(|| {
                panic!(
                    "rename_to_affine_posterior_packet: per-world naming \
                     missing Beta variable {name}; this should have been \
                     populated by collect_positions for any var in this \
                     world's value tree"
                )
            });
            b.clone().with_display_name(display.clone())
        })
        .sorted_by_key(|b| b.name().clone())
        .collect();
    let dirichlet = registry
        .dirichlet
        .iter()
        .map(|d| {
            let display = naming.var_names.get(&d.name()).unwrap_or_else(|| {
                panic!(
                    "rename_to_affine_posterior_packet: per-world naming \
                     missing Dirichlet variable {}; this should have been \
                     populated by collect_positions for any var in this \
                     world's value tree",
                    d.name()
                )
            });
            d.clone().with_display_name(display.clone())
        })
        .sorted_by_key(|d| d.name().clone())
        .collect();
    // A queried scaled gamma `s·g` is reported as its OWN entry — the
    // distribution of `s·g` = `posterior(g).scaled(s)` — rather than by
    // mutating `g`'s entry in place (which would corrupt the conditioning
    // rate that draws reference). This mirrors the Gaussian family, which
    // reports per queried affine expression, not per underlying variable.
    // Each rate variable therefore emits ONE entry per distinct queried
    // scale (plus the unscaled entry when draws need the bare conditioning
    // rate, or as a fallback when no query position was recorded).
    let gamma = registry
        .gamma
        .iter()
        .flat_map(|g| {
            let rate_var = g.name();
            // Bare-rate display name (var name, or `rate_{id}` fallback for a
            // rate that only appears as a draw's conditioning variable). A
            // missing naming entry is not a bug here (unlike beta/dirichlet),
            // and the fallback must be a pure function of the id since the
            // packet is a HashMap/sort key.
            let bare_name = |name: ContVarName| {
                naming
                    .var_names
                    .get(&name)
                    .cloned()
                    .unwrap_or_else(|| format!("rate_{name}"))
            };
            // Draws conditioned on `g` need the bare (unscaled) entry to show
            // as their rate, so include scale 1.0 whenever there are draws (or
            // as a fallback for a rate with no recorded query position, e.g.
            // the unit-test callers passing an empty map).
            let has_draws =
                !g.poisson.constraints().is_empty() || !g.exponential.constraints().is_empty();
            let mut scales: BTreeSet<ordered_float::NotNan<f64>> = gamma_query_scales
                .get(&rate_var)
                .cloned()
                .unwrap_or_default();
            if has_draws || scales.is_empty() {
                scales.insert(ordered_float::NotNan::new(1.0).unwrap());
            }
            scales.into_iter().map(move |s| {
                if s.into_inner() == 1.0 {
                    // Bare entry: the UNSCALED rate posterior plus its draws.
                    g.with_display_names(bare_name)
                } else {
                    // Scaled rate `s·g`: its own quantity, labelled exactly as
                    // the value renders it (`render_symbolic` ⇒ "s * <name>"),
                    // with no draws (those live on the bare entry).
                    let label = language::values::mk_gamma_scaled(rate_var, s.into_inner())
                        .render_symbolic(Some(naming))
                        .expect("scaled gamma renders to a name");
                    JointGammaPrior::from_gamma(g.gamma.clone().scaled(s.into_inner()))
                        .with_display_names(|_| label.clone())
                }
            })
        })
        .sorted_by_key(|g| g.name().clone())
        .collect();
    let gaussian = build_affine_gaussian_blocks(registry, naming);
    PriorRegistry {
        beta,
        dirichlet,
        gamma,
        gaussian,
    }
}

fn build_affine_gaussian_blocks(
    registry: &PriorRegistry,
    naming: &NamingScheme,
) -> Vec<GaussianPrior<String>> {
    let unique_exprs = dedupe_and_order_exprs(naming);
    let touched = compute_touched_blocks(&unique_exprs, registry);
    let components = group_exprs_by_shared_blocks(&touched);
    components
        .into_iter()
        .map(|comp| build_block_for_component(&comp, &unique_exprs, registry, naming))
        .collect()
}

/// Deduplicated queried affine expressions ordered by their minimum
/// underlying `ContVarName`. `naming.expr_names` is already first-seen
/// deduped (one entry per unique `GaussianAffineExpr`).
///
/// `naming.expr_names` is a `HashMap`, so its iteration order is
/// non-deterministic. We must therefore sort by a *total* key — sorting
/// just on `min ContVarName` leaves ties (e.g. `2*G0 + G1` vs `G0 - G1`)
/// to be resolved by whichever order HashMap happened to return, which
/// produces flaky var_order snapshots. Compare on the full
/// (coefficients, constant) tuple to break ties.
fn dedupe_and_order_exprs(naming: &NamingScheme) -> Vec<&GaussianAffineExpr> {
    let mut exprs: Vec<&GaussianAffineExpr> = naming.expr_names.keys().collect();
    exprs.sort_by(|a, b| {
        let a_keys: Vec<_> = a.coefficients.keys().collect();
        let b_keys: Vec<_> = b.coefficients.keys().collect();
        a_keys
            .cmp(&b_keys)
            .then_with(|| {
                let a_vals: Vec<_> = a.coefficients.values().collect();
                let b_vals: Vec<_> = b.coefficients.values().collect();
                a_vals.cmp(&b_vals)
            })
            .then_with(|| a.constant.cmp(&b.constant))
    });
    exprs
}

/// For each queried expression, the set of `registry.gaussian` block
/// indices whose `var_order` intersects the expression's scope.
fn compute_touched_blocks(
    unique_exprs: &[&GaussianAffineExpr],
    registry: &PriorRegistry,
) -> Vec<BTreeSet<usize>> {
    unique_exprs
        .iter()
        .map(|expr| {
            let touched: BTreeSet<usize> = (0..registry.gaussian.len())
                .filter(|&bi| {
                    expr.scope()
                        .any(|v| registry.gaussian[bi].var_order.contains(v))
                })
                .collect();
            // Every var the expression mentions should live in some
            // posterior block. Guaranteed because per-world naming
            // restricts `unique_exprs` to expressions found in this
            // world's value tree, and `fill_in_missing_variables_with_prior`
            // adds back any prior-only var in that scope. Asserted (not
            // debug-only) because a failure here corrupts the posterior
            // packet and would otherwise resurface as a much-less-specific
            // panic deeper in `gaussian_prior_for`.
            assert!(
                expr.scope().all(|v| {
                    touched
                        .iter()
                        .any(|&bi| registry.gaussian[bi].var_order.contains(v))
                }),
                "queried affine expression touches a variable not in any registry Gaussian block",
            );
            touched
        })
        .collect()
}

/// Two expressions are connected iff they share at least one posterior
/// Gaussian block. Returns one `Vec<usize>` per connected component,
/// ordered by the component's minimum touched block index. Within each
/// component, expression indices preserve the input ordering.
fn group_exprs_by_shared_blocks(touched: &[BTreeSet<usize>]) -> Vec<Vec<usize>> {
    let n = touched.len();
    let mut uf = UnionFind::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            if !touched[i].is_disjoint(&touched[j]) {
                uf.union(i, j);
            }
        }
    }
    let mut components = uf.components();
    for c in components.iter_mut() {
        c.sort();
    }
    components.sort_by_key(|c| {
        c.iter()
            .flat_map(|&i| touched[i].iter().copied())
            .min()
            .unwrap_or(usize::MAX)
    });
    components
}

/// Build the `GaussianPrior<String>` for one connected component of
/// expressions. Assembles the joint posterior over all underlying vars
/// the component touches via `gaussian_prior_for`, then computes the
/// k×k joint over the expressions themselves with `affine_moments` /
/// `affine_covariance`.
fn build_block_for_component(
    component: &[usize],
    unique_exprs: &[&GaussianAffineExpr],
    registry: &PriorRegistry,
    naming: &NamingScheme,
) -> GaussianPrior<String> {
    let scope: BTreeSet<ContVarName> = component
        .iter()
        .flat_map(|&i| unique_exprs[i].scope().copied())
        .collect();
    let block = registry.gaussian_prior_for(&scope);

    let k = component.len();
    let mut mean = na::DVector::<f64>::zeros(k);
    let mut cov = na::DMatrix::<f64>::zeros(k, k);
    let mut var_order: Vec<String> = Vec::with_capacity(k);

    let coeffs_f64: Vec<BTreeMap<ContVarName, f64>> = component
        .iter()
        .map(|&i| {
            unique_exprs[i]
                .coefficients
                .iter()
                .map(|(name, val)| (*name, val.into_inner()))
                .collect()
        })
        .collect();

    for (row, &expr_idx) in component.iter().enumerate() {
        let expr = unique_exprs[expr_idx];
        let constant = expr.constant.into_inner();
        let (m, v) = block.affine_moments(&coeffs_f64[row], constant);
        mean[row] = m;
        cov[(row, row)] = v;
        for col in 0..row {
            let c = block.affine_covariance(&coeffs_f64[row], &coeffs_f64[col]);
            cov[(row, col)] = c;
            cov[(col, row)] = c;
        }
        let display = naming.expr_names.get(expr).unwrap_or_else(|| {
            panic!(
                "build_block_for_component: per-world naming missing \
                 Gaussian affine expression; should have been populated \
                 by collect_positions for any expr in this world's value tree"
            )
        });
        var_order.push(display.clone());
    }

    GaussianPrior {
        var_order,
        mean,
        cov,
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::NotNan;

    use super::*;

    fn affine(coeffs: &[(ContVarName, f64)], constant: f64) -> GaussianAffineExpr {
        GaussianAffineExpr {
            coefficients: coeffs
                .iter()
                .map(|&(k, v)| (k, NotNan::new(v).unwrap()))
                .collect(),
            constant: NotNan::new(constant).unwrap(),
        }
    }

    fn gauss_block(var_order: Vec<ContVarName>, mean: &[f64], cov: &[&[f64]]) -> GaussianPrior {
        let n = var_order.len();
        let m = na::DVector::from_iterator(n, mean.iter().copied());
        let c =
            na::DMatrix::from_row_iterator(n, n, cov.iter().flat_map(|row| row.iter().copied()));
        GaussianPrior {
            var_order,
            mean: m,
            cov: c,
        }
    }

    fn naming_with(
        vars: &[(ContVarName, &str)],
        exprs: &[(GaussianAffineExpr, &str)],
    ) -> NamingScheme {
        NamingScheme {
            var_names: vars.iter().map(|(n, s)| (*n, s.to_string())).collect(),
            expr_names: exprs
                .iter()
                .map(|(e, s)| (e.clone(), s.to_string()))
                .collect(),
        }
    }

    #[test]
    fn identity_affine_single_var() {
        // expr = 1·x + 0 over a 1×1 block. One output row, mean & var copied.
        let expr = affine(&[(1, 1.0)], 0.0);
        let registry = PriorRegistry::from_sorted(
            vec![],
            vec![],
            vec![],
            vec![gauss_block(vec![1], &[1.5], &[&[2.0]])],
        );
        let naming = naming_with(&[], &[(expr.clone(), "x")]);

        let packet = rename_to_affine_posterior_packet(&registry, &naming, &BTreeMap::new());
        assert_eq!(packet.gaussian.len(), 1);
        let g = &packet.gaussian[0];
        assert_eq!(g.var_order, vec!["x".to_string()]);
        assert!((g.mean[0] - 1.5).abs() < 1e-12);
        assert!((g.cov[(0, 0)] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn two_independent_exprs_two_blocks() {
        // exprs `1·x` and `1·y` over independent blocks {1}, {2}.
        let ex_x = affine(&[(1, 1.0)], 0.0);
        let ex_y = affine(&[(2, 1.0)], 0.0);
        let registry = PriorRegistry::from_sorted(
            vec![],
            vec![],
            vec![],
            vec![
                gauss_block(vec![1], &[1.0], &[&[1.0]]),
                gauss_block(vec![2], &[2.0], &[&[3.0]]),
            ],
        );
        let naming = naming_with(&[], &[(ex_x.clone(), "x"), (ex_y.clone(), "y")]);

        let packet = rename_to_affine_posterior_packet(&registry, &naming, &BTreeMap::new());
        assert_eq!(packet.gaussian.len(), 2);
        // Ordered by min-touched-block. ex_x touches block 0, ex_y touches block 1.
        assert_eq!(packet.gaussian[0].var_order, vec!["x".to_string()]);
        assert!((packet.gaussian[0].mean[0] - 1.0).abs() < 1e-12);
        assert!((packet.gaussian[0].cov[(0, 0)] - 1.0).abs() < 1e-12);
        assert_eq!(packet.gaussian[1].var_order, vec!["y".to_string()]);
        assert!((packet.gaussian[1].mean[0] - 2.0).abs() < 1e-12);
        assert!((packet.gaussian[1].cov[(0, 0)] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn expr_spans_two_independent_blocks() {
        // expr `1·x + 1·y` where x ∈ block 0 and y ∈ block 1 are independent.
        // One output block; mean = μ_x + μ_y; var = σ²_x + σ²_y.
        let ex = affine(&[(1, 1.0), (2, 1.0)], 0.0);
        let registry = PriorRegistry::from_sorted(
            vec![],
            vec![],
            vec![],
            vec![
                gauss_block(vec![1], &[5.0], &[&[2.0]]),
                gauss_block(vec![2], &[7.0], &[&[3.0]]),
            ],
        );
        let naming = naming_with(&[], &[(ex.clone(), "z")]);

        let packet = rename_to_affine_posterior_packet(&registry, &naming, &BTreeMap::new());
        assert_eq!(packet.gaussian.len(), 1);
        let g = &packet.gaussian[0];
        assert_eq!(g.var_order, vec!["z".to_string()]);
        assert!((g.mean[0] - 12.0).abs() < 1e-12);
        assert!((g.cov[(0, 0)] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn dedupe_same_expr_at_two_paths() {
        // Same GaussianAffineExpr at two distinct paths → one output row.
        let expr = affine(&[(1, 1.0)], 0.0);
        let registry = PriorRegistry::from_sorted(
            vec![],
            vec![],
            vec![],
            vec![gauss_block(vec![1], &[0.0], &[&[1.0]])],
        );
        let naming = naming_with(&[], &[(expr.clone(), "x")]);

        let packet = rename_to_affine_posterior_packet(&registry, &naming, &BTreeMap::new());
        assert_eq!(packet.gaussian.len(), 1);
        assert_eq!(packet.gaussian[0].var_order.len(), 1);
        assert_eq!(packet.gaussian[0].var_order[0], "x");
    }
}
