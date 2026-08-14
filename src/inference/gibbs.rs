//! Gibbs sampler driver.
//!
//! Surface syntax:
//! `(Gibbs query cond num_samples (Cons e1 (Cons e2 (Nil))) initialization)`,
//! where `initialization` is `(WithConstant expr)` or `(WithPriorSample expr)`.
//!
//! The chain's block-1 value is seeded once up front from `initialization`
//! (see `seed_initial_block`); the seed contributes no collected sample. Each
//! step then samples one block expression conditional on the running pin
//! against the previously-sampled block's value, accumulating one query sample
//! per step. After `num_samples` steps, returns either `GibbsOutcome::Complete`
//! or `GibbsOutcome::Stuck { samples }` when the chain hits a
//! zero-probability conditioning event; `samples.len()` is then the number
//! of steps completed before the stall.

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rand::rngs::ThreadRng;

use super::conjugate_pairs::Assignment;
use super::lazy_kc::compile::CompileMode;
use super::lazy_kc::ctx::CompilerCtx;
use super::sampling::{force_value, SampleState};
use crate::discrete_factorizations::{BooleanFactorization, BooleanFunction, BooleanFunctionOps};
use crate::inference::sampling::force_n_samples;
use crate::language::pexpr::{CaseOfGuard, PExpr};
use crate::language::types::{StringInterner, Symbol};
use crate::language::values::{empty_env, env_cons, EnvInner, PluckVal, PluckValue};
use crate::{inference, Syms};

/// Max prior-sample redraws before a `WithPriorSample` init is declared
/// off-support (no overlap with the conditioning evidence). Tunable.
const N_INIT_RETRIES: usize = 10;

/// Result of a Gibbs run, exposed to callers embedding pluck in Rust.
/// The Pluck-language surface flattens this into a samples Cons-list and
/// surfaces stalls via stderr — see `lib.rs::run_gibbs`.
pub enum GibbsOutcome {
    /// Chain ran the full `num_samples`; `samples.len() == num_samples`.
    Complete { samples: Vec<PluckVal> },
    /// Chain halted because evidence ∧ pin was unsatisfiable. `samples`
    /// holds the query values accumulated before the stall, so
    /// `samples.len()` is the number of steps completed.
    Stuck { samples: Vec<PluckVal> },
}

pub struct GibbsSymbols {
    o_sym: Symbol,
    s_sym: Symbol,
    nil_sym: Symbol,
    cons_sym: Symbol,
    with_prior_sample_sym: Symbol,
    with_constant_sym: Symbol,
    block_sym: Symbol,
    value_sym: Symbol,
    evidence_sym: Symbol,
}

pub struct GibbsCtx {
    query_thunk: PluckVal,
    evidence_thunk: PluckVal,
    num_samples: usize,
    blocks: Vec<PluckVal>,
    num_blocks: usize,
    out_samples: Vec<PluckVal>,
    prev_block_value: PluckVal,
    rng: ThreadRng,
    progress: Option<Arc<AtomicUsize>>,
    symbols: GibbsSymbols,
}

impl GibbsCtx {
    pub(crate) fn generate_gibbs_symbols(
        symbols: &Syms,
        interner: &mut StringInterner,
    ) -> GibbsSymbols {
        // Fresh symbols not already in `symbols`, used to build each step's conditioning expression.
        let block_sym = interner.intern("__gibbs_block");
        let value_sym = interner.intern("__gibbs_value");
        let evidence_sym = interner.intern("__gibbs_evidence");

        GibbsSymbols {
            o_sym: symbols.o_sym,
            s_sym: symbols.s_sym,
            nil_sym: symbols.nil_sym,
            cons_sym: symbols.cons_sym,
            with_prior_sample_sym: symbols.with_prior_sample_sym,
            with_constant_sym: symbols.with_constant_sym,
            block_sym,
            value_sym,
            evidence_sym,
        }
    }

    /// Build the sampler state. Returns
    /// `Err` (rather than a stalled outcome) when the inputs are malformed
    /// or the condition has zero probability.
    pub fn new(
        args: &[PluckVal],
        ctx: &mut CompilerCtx,
        symbols: GibbsSymbols,
    ) -> Result<Self, String> {
        assert!(
            args.len() >= 5,
            "Gibbs requires 5 args: query, cond, num_samples, blocks_list, initialization"
        );
        let query_thunk = args[0].clone();
        let evidence_thunk = args[1].clone();
        let n_thunk = &args[2];
        let blocks_thunk = &args[3];
        let init_arg = &args[4];

        // Resolve num_samples (same shape as posterior_sample).
        let num_samples = force_n_samples(n_thunk, symbols.o_sym, symbols.s_sym, ctx);

        // Force the blocks Cons-list eagerly into a Vec<PluckVal>.
        let blocks = match force_cons_list(blocks_thunk, ctx, symbols.nil_sym, symbols.cons_sym) {
            Ok(b) if b.is_empty() => {
                return Err("requires at least one block expression".to_string());
            }
            Ok(b) => b,
            Err(msg) => {
                return Err(msg);
            }
        };
        let num_blocks = blocks.len();

        // Seed block 1 per the initialization BEFORE constructing, so
        // `prev_block_value` is never absent. Init errors (malformed,
        // off-support, inconsistent constant) surface here alongside the
        // condition-zero-probability error above.
        let mut rng = rand::thread_rng();
        let prev_block_value = seed_initial_block(
            init_arg,
            &symbols,
            &blocks[0],
            &evidence_thunk,
            &mut rng,
            ctx,
        )?;

        Ok(GibbsCtx {
            query_thunk,
            evidence_thunk,
            num_samples,
            blocks,
            num_blocks,
            out_samples: Vec::with_capacity(num_samples),
            prev_block_value,
            rng,
            progress: None,
            symbols,
        })
    }

    /// Attach a shared counter that `run` bumps to the running sample count
    /// each iteration, so a caller (e.g. the CLI) can poll progress.
    pub fn with_progress(mut self, counter: Arc<AtomicUsize>) -> Self {
        self.progress = Some(counter);
        self
    }

    /// Total samples this chain will collect (resolved during `new`).
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Drive the chain. Consumes `self` so the accumulated `out_samples` can
    /// be moved straight into the returned `GibbsOutcome`.
    ///
    /// `blocks[0]` is already seeded into `prev_block_value` (see
    /// `seed_initial_block`), so each iteration pins the most-recently-sampled
    /// block (`cur`) and samples the next one. The seed itself contributes no
    /// collected query; all `num_samples` samples come from the loop.
    pub fn run(mut self, ctx: &mut CompilerCtx) -> GibbsOutcome {
        let mut cur = 0usize;
        for _ in 0..self.num_samples {
            let next = (cur + 1) % self.num_blocks;

            // Pin the current block to its running value.
            let cur_thunk = self.blocks[cur].clone();
            let prev_val = self.prev_block_value.clone();

            let conditioning_bf = build_conditioning_bf(
                &cur_thunk,
                &prev_val,
                &self.evidence_thunk,
                &self.symbols,
                ctx,
            );
            if conditioning_bf.is_false() {
                return GibbsOutcome::Stuck {
                    samples: self.out_samples,
                };
            }

            let (sampled_bf, assignment) = match sample_world(conditioning_bf, ctx, &mut self.rng) {
                Some(pair) => pair,
                None => {
                    return GibbsOutcome::Stuck {
                        samples: self.out_samples,
                    };
                }
            };

            // Force the next block (for the next step's pin) and query (for
            // the output) under this step's sampled constraint. Cheap Rc
            // clone of the block thunk releases the borrow on `self.blocks`.
            let next_thunk = self.blocks[next].clone();
            let (block_val, query_val) = self.force_step(&next_thunk, sampled_bf, assignment, ctx);

            self.out_samples.push(query_val);
            if let Some(p) = &self.progress {
                p.store(self.out_samples.len(), Ordering::Relaxed);
            }
            self.prev_block_value = block_val;
            cur = next;
        }

        GibbsOutcome::Complete {
            samples: self.out_samples,
        }
    }

    /// Force the block expression (for the next step's pin) and the query
    /// expression (for the output) under this step's sampled constraint. Both
    /// `force_value` calls share one `SampleState` so the assignment and trace
    /// stay consistent. Returns `(block_val, query_val)`.
    fn force_step(
        &self,
        block_thunk: &PluckVal,
        sampled_bf: BooleanFunction,
        assignment: Assignment,
        ctx: &mut CompilerCtx,
    ) -> (PluckVal, PluckVal) {
        let mut state = SampleState {
            constraint: sampled_bf,
            assignment,
            trace: HashMap::new(),
            memo: HashMap::new(),
            true_sym: ctx.true_sym,
        };
        let block_val = force_value(block_thunk, &mut state, ctx);
        let query_val = force_value(&self.query_thunk, &mut state, ctx);
        (block_val, query_val)
    }
}

/// Similar to `build_given_expr_and_env` in lib.rs - we have *values* for the block expression,
/// pinned value and evidence, but compiler expects PExprs. So stuff the PluckVals into the
/// environment under fresh names, then write a PExpr that references them via PExpr::Var
fn build_conditioning_expr(
    block_thunk: &PluckVal,
    block_value: &PluckVal,
    evidence_thunk: &PluckVal,
    symbols: &GibbsSymbols,
    ctx: &mut CompilerCtx,
) -> (PExpr, Rc<EnvInner>) {
    let mut env = empty_env();
    env = env_cons(symbols.block_sym, block_thunk.clone(), env);
    env = env_cons(symbols.value_sym, block_value.clone(), env);
    env = env_cons(symbols.evidence_sym, evidence_thunk.clone(), env);

    // Pin expr == val
    let pin_expr = PExpr::Pin {
        expr: Box::new(PExpr::Var {
            name: symbols.block_sym,
        }),
        val: Box::new(PExpr::Var {
            name: symbols.value_sym,
        }),
    };

    // Match on evidence -> True
    let evidence_expr = PExpr::CaseOf {
        guards: vec![CaseOfGuard {
            constructor: ctx.true_sym,
            args: vec![],
        }],
        scrutinee: Box::new(PExpr::Var {
            name: symbols.evidence_sym,
        }),
        branches: vec![PExpr::Construct {
            constructor: ctx.true_sym,
            args: vec![],
        }],
    };

    // We match on pin_expr -> True AND evidence -> True
    (
        PExpr::CaseOf {
            guards: vec![CaseOfGuard {
                constructor: ctx.true_sym,
                args: vec![],
            }],
            scrutinee: Box::new(pin_expr),
            branches: vec![evidence_expr],
        },
        env,
    )
}

/// Pin `block_thunk`'s symbolic worlds to `value`, OR-ing together the worlds
/// where the equality holds into the single pin BDD used as a step constraint.
fn build_conditioning_bf(
    block_thunk: &PluckVal,
    block_value: &PluckVal,
    evidence_thunk: &PluckVal,
    symbols: &GibbsSymbols,
    ctx: &mut CompilerCtx,
) -> BooleanFunction {
    let (conditioning_expr, env) =
        build_conditioning_expr(block_thunk, block_value, evidence_thunk, symbols, ctx);
    let mut mode = inference::lazy_kc::compile::CompileMode::KC {
        path_condition: ctx.state.true_bf.clone(),
    };
    let (worlds, _) = ctx.traced_compile(&conditioning_expr, &env, &mut mode, 0);
    or_true_worlds(worlds, ctx)
}

/// OR together the BDDs of every world whose result is the nullary `True`
/// constructor (the worlds in which a pin equality holds).
fn or_true_worlds(
    worlds: Vec<(PluckVal, BooleanFunction)>,
    ctx: &mut CompilerCtx,
) -> BooleanFunction {
    let builder = ctx.state.fac();
    let mut acc = BooleanFunction::false_ptr();
    for (result, bf) in worlds {
        // Since the conditioning_expr only matches on true, all the result values should be true
        assert!(
            if let PluckValue::Value { constructor, args } = result.as_ref() {
                *constructor == ctx.true_sym && args.is_empty()
            } else {
                false
            },
            "Found non-true result value in condition_expr result"
        );
        acc = builder.or(&acc, &bf);
    }
    acc
}

/// Sample a single world under the evidence. Continuous and discrete sampling
/// are done in two stages; they can disagree when the continuous draw lands in
/// a region where the chosen discrete path has zero (log)-marginal mass — e.g.
/// when the posterior is multi-modal and the SPN posterior picks one mode while
/// the BDD requires another. Returns `None` in that case.
fn sample_world(
    evidence_bf: BooleanFunction,
    ctx: &mut CompilerCtx,
    rng: &mut ThreadRng,
) -> Option<(BooleanFunction, Assignment)> {
    let evidence_spn = ctx.state.wmc_spn(evidence_bf.clone());
    let mut assignment = ctx.state.sample_continuous_posterior(&evidence_spn, rng);
    // Draws observed only in unsampled mixture branches are absent from
    // the posterior sample; cover them before weight evaluation.
    ctx.state
        .cover_support_gamma_draws(evidence_bf.clone(), &mut assignment, rng);
    let candidate = ctx
        .state
        .sample_discrete_given(evidence_bf, &assignment, rng);
    if !candidate.is_false() {
        return Some((candidate, assignment));
    }
    None
}

/// Force a single thunk to a concrete value under a sampled world. (`force_step`
/// is this run twice, sharing one `SampleState` so block and query stay jointly
/// consistent.)
fn force_one(
    thunk: &PluckVal,
    constraint: BooleanFunction,
    assignment: Assignment,
    ctx: &mut CompilerCtx,
) -> PluckVal {
    let mut state = SampleState {
        constraint,
        assignment,
        trace: HashMap::new(),
        memo: HashMap::new(),
        true_sym: ctx.true_sym,
    };
    force_value(thunk, &mut state, ctx)
}

/// Draw `expr` from its prior (no evidence) and force it to a concrete value.
/// Continuous variables are sampled from their prior via
/// `sample_continuous_posterior` over an empty-scope SPN; discrete flips are
/// free-sampled at their prior probability inside `force_value`. A deterministic
/// `expr` simply evaluates to its value.
fn sample_expr_from_prior(expr: &PluckVal, rng: &mut ThreadRng, ctx: &mut CompilerCtx) -> PluckVal {
    // KC-compile `expr` so its continuous priors are registered before we draw.
    {
        let mut kc_mode = CompileMode::KC {
            path_condition: ctx.state.true_bf.clone(),
        };
        let _ = ctx.evaluate_thunk(expr, &mut kc_mode);
    }
    let true_bf = ctx.state.true_bf.clone();
    let prior_spn = ctx.state.wmc_spn(true_bf.clone());
    let assignment = ctx.state.sample_continuous_posterior(&prior_spn, rng);
    // Back to Sample mode for the force. Sample mode bypasses the KC thunk
    // cache, so the KC entries registered above are left in place (and reused).
    force_one(expr, true_bf, assignment, ctx)
}

/// Seed the chain's initial block-1 value per the `Initialization` argument.
/// Pure seed: returns the value `blocks[0]` starts pinned to; collects no query
/// sample. Errors are user-interpretable and surface from `GibbsCtx::new`.
fn seed_initial_block(
    init_arg: &PluckVal,
    symbols: &GibbsSymbols,
    block_thunk: &PluckVal,
    evidence_thunk: &PluckVal,
    rng: &mut ThreadRng,
    ctx: &mut CompilerCtx,
) -> Result<PluckVal, String> {
    let (ctor, inner) = force_constructor(init_arg, ctx)?;
    if ctor == symbols.with_constant_sym && inner.len() == 1 {
        seed_via_constant(&inner[0], block_thunk, evidence_thunk, symbols, rng, ctx)
    } else if ctor == symbols.with_prior_sample_sym && inner.len() == 1 {
        seed_via_prior_sample(&inner[0], block_thunk, evidence_thunk, symbols, rng, ctx)
    } else {
        Err("initialization must be (WithPriorSample expr) or (WithConstant expr).".to_string())
    }
}

/// `WithConstant` seeding: we are given an `init_expr` which evaluates to the
/// starting value to which `block_expr` is pinned. Note the expression may have
/// randomness (which gets sampled). It's the user's responsibility to ensure that the
/// value of `init_expr` is valid for `block_expr` under the evidence
fn seed_via_constant(
    init_expr: &PluckVal,
    block_expr: &PluckVal,
    evidence_thunk: &PluckVal,
    symbols: &GibbsSymbols,
    rng: &mut ThreadRng,
    ctx: &mut CompilerCtx,
) -> Result<PluckVal, String> {
    // Force the constant once (any randomness inside it is sampled once).
    let const_val = sample_expr_from_prior(init_expr, rng, ctx);
    // Eager consistency check: `block_expr = const` must be compatible with the
    // evidence, else the chain would silently stall at step 0.
    let condition_bf = build_conditioning_bf(block_expr, &const_val, evidence_thunk, symbols, ctx);
    if condition_bf.is_false() {
        Err("WithConstant value is inconsistent with the evidence \
                    (zero probability)."
            .to_string())
    } else {
        Ok(const_val)
    }
}

/// `WithPriorSample` seeding: prior-sample `init_expr`, pin the expression to
/// the draw, and sample `blocks[0]` given that pin plus the evidence.
/// Off-support draws (no evidence overlap) are retried up to `N_INIT_RETRIES`
/// times before erroring.
fn seed_via_prior_sample(
    init_expr: &PluckVal,
    block_thunk: &PluckVal,
    evidence_thunk: &PluckVal,
    symbols: &GibbsSymbols,
    rng: &mut ThreadRng,
    ctx: &mut CompilerCtx,
) -> Result<PluckVal, String> {
    for _ in 0..N_INIT_RETRIES {
        let init_val = sample_expr_from_prior(init_expr, rng, ctx);
        let condition_bf =
            build_conditioning_bf(init_expr, &init_val, evidence_thunk, symbols, ctx);
        if condition_bf.is_false() {
            continue; // shape mismatch (shouldn't vary across draws, but cheap)
        }
        if let Some((sampled_bf, assignment)) = sample_world(condition_bf, ctx, rng) {
            return Ok(force_one(block_thunk, sampled_bf, assignment, ctx));
        }
        // sample_world == None: continuous/discrete disagreement → redraw.
    }
    Err(format!(
        "WithPriorSample drew {} off-support init values; the initialization \
         expression has no overlap with the conditioning evidence. Choose an \
         init expression with full support under the evidence (e.g. `x` when \
         the constraint is `x + y = 1`).",
        N_INIT_RETRIES
    ))
}

/// KC-compile `init_arg` and return the `(constructor, args)` of its single
/// live world. Errors if it produces no live world or is not a constructor
/// value (e.g. the user passed a non-`Initialization` expression).
fn force_constructor(
    init_arg: &PluckVal,
    ctx: &mut CompilerCtx,
) -> Result<(Symbol, Vec<PluckVal>), String> {
    let mut kc_mode = CompileMode::KC {
        path_condition: ctx.state.true_bf.clone(),
    };
    let (worlds, _) = ctx.evaluate_thunk(init_arg, &mut kc_mode);
    let live = worlds.into_iter().find(|(_, bf)| !bf.is_false());
    match live {
        Some((val, _)) => match val.as_ref() {
            PluckValue::Value { constructor, args } => Ok((*constructor, args.clone())),
            _ => Err("initialization must be a constructor value \
                      (WithPriorSample or WithConstant)."
                .to_string()),
        },
        None => Err("initialization evaluated to no live worlds.".to_string()),
    }
}

/// Fully force a Cons-list of thunks into a `Vec<PluckVal>` of element
/// thunks. Returns an error if the list is not Nil-terminated under the
/// trivial path condition.
fn force_cons_list(
    list_thunk: &PluckVal,
    ctx: &mut CompilerCtx,
    nil_sym: Symbol,
    cons_sym: Symbol,
) -> Result<Vec<PluckVal>, String> {
    let mut out: Vec<PluckVal> = Vec::new();
    let mut current = list_thunk.clone();
    loop {
        let mut mode = CompileMode::KC {
            path_condition: ctx.state.true_bf.clone(),
        };
        let (worlds, _) = ctx.evaluate_thunk(&current, &mut mode);
        // TODO - should we assert that there must be a *single* true-guarded world here?
        let live = worlds.into_iter().find(|(_, bf)| !bf.is_false());
        let val = match live {
            Some((v, _)) => v,
            None => {
                return Err("blocks list evaluated to no live worlds".to_string());
            }
        };
        match val.as_ref() {
            PluckValue::Value { constructor, args } if *constructor == nil_sym => {
                return Ok(out);
            }
            PluckValue::Value { constructor, args } if *constructor == cons_sym => {
                if args.len() != 2 {
                    return Err(format!(
                        "Cons in blocks list has {} args, expected 2",
                        args.len()
                    ));
                }
                out.push(args[0].clone());
                current = args[1].clone();
            }
            _ => {
                return Err(
                    "blocks list must be a Cons/Nil chain, got constructor not Nil/Cons".into(),
                );
            }
        }
    }
}
