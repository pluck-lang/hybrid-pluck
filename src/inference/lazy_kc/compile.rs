use std::rc::Rc;

use ordered_float::NotNan;

use super::ctx::CompilerCtx;
use super::monad::*;
use super::state::{FlipWeight, GuardedWorlds, LazyKCState, World};
use crate::discrete_factorizations::{BooleanFactorization, BooleanFunction, BooleanFunctionOps};
use crate::inference::conjugate_pairs::{
    BetaFlip, BetaPrior, DirichletFlip, DirichletPrior, GammaFlip, GammaPrior, GaussianFlip,
    GaussianPrior,
};
use crate::inference::int_dists::{
    enumerate_int_dist, int_dist_add, int_dist_eq, int_dist_lt, int_dist_sub, IntDist,
};
use crate::inference::sampling::{GammaDrawFamily, SampleState};
use crate::language::pexpr::{CaseOfGuard, NativeVal, PExpr};
use crate::language::types::Symbol;
use crate::language::values::*;
use crate::utils::intervals::{Interval, IntervalOrEq};

/// Mode under which compilation evaluates an expression.
///
/// - `KC { path_condition }` — knowledge compilation: every flip / observation
///   forks worlds and the path_condition narrows down each branch. Used for
///   exact inference and for evidence / pin (conditioning) compilation.
/// - `Sample(state)` — forward sampling threading a single sampled world:
///   stochastic choices resolve against the per-sample constraint
///   (reading off pinned variables, prior-sampling and recording free
///   ones), and every `bind_compile` runs its continuation once, with this
///   mode propagated. Function bodies, case branches, and strict arguments
///   execute along the sampled path only; every `traced_compile` /
///   `evaluate_thunk` returns exactly one world (asserted at the bind
///   boundary, `sample_compile_one`, and `force_value`; ThunkUnion
///   selection in `evaluate_thunk_union` — see `select_world` — is the one
///   multi-world resolution point). The `path_condition()` accessor
///   returns `true_bf` so non-branching code paths work uniformly.
pub enum CompileMode<'a> {
    KC { path_condition: BooleanFunction },
    Sample(&'a mut SampleState),
}

impl<'a> CompileMode<'a> {
    /// Outer path condition for the current compile invocation.
    /// In Sample mode the constraint lives on `SampleState`, not on the
    /// mode itself, so this returns `true_bf`.
    pub fn path_condition(&self) -> BooleanFunction {
        match self {
            CompileMode::KC { path_condition } => path_condition.clone(),
            CompileMode::Sample(_) => BooleanFunction::true_ptr(),
        }
    }
}

/// The four float operations, extended for GaussianExpr and scaled Gamma values.
#[derive(Clone, Copy)]
enum FloatOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Real range comparisons. `Lt`/`Geq` are exactly closed under negation
/// over half-open events, so each compiles to one BDD flip whose false
/// branch is the complementary event.
#[derive(Clone, Copy)]
enum RealCmp {
    /// `expr < val` — the event `[0, val)`.
    Lt,
    /// `expr ≥ val` — the event `[val, ∞)`.
    Geq,
}

/// Integer range comparisons (native ints / Peano nats / Poisson draws).
#[derive(Clone, Copy)]
enum NativeCmp {
    /// `a ≤ b`.
    Leq,
    /// `a ≥ b`.
    Geq,
}

impl NativeCmp {
    /// The comparison seen from the right operand's side:
    /// `a ≤ b ⇔ b ≥ a`.
    fn mirror(self) -> Self {
        match self {
            NativeCmp::Leq => NativeCmp::Geq,
            NativeCmp::Geq => NativeCmp::Leq,
        }
    }
}

/// Classification of the `Categorical` arg list:
/// either a single DirichletVector, or N constant floats summing to 1.
enum CatKind {
    Dirichlet { name: u64, categories: usize },
    Constants(Vec<f64>),
}

impl<'a> CompilerCtx<'a> {
    fn bool_val(&self, b: bool) -> PluckVal {
        mk_value(if b { self.true_sym } else { self.false_sym }, vec![])
    }

    /// Returns the appropriate constant value when a flip probability is
    /// (essentially) 0.0 or 1.0; otherwise None. Used by both KC and
    /// sample-mode flip paths.
    fn constant_flip_shortcut(&self, p: f64) -> Option<PluckVal> {
        if (p - 0.0).abs() < 1e-15 {
            return Some(self.bool_val(false));
        }
        if (p - 1.0).abs() < 1e-15 {
            return Some(self.bool_val(true));
        }
        None
    }

    /// Top-level compile: handles limit checks, callstack push/pop, then
    /// dispatches via [`Self::compile`].
    pub fn traced_compile(
        &mut self,
        expr: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
        strict_order_index: i32,
    ) -> GuardedWorlds {
        if mode.path_condition().is_false() {
            self.state.stats.num_false_pc_exits += 1;
            return false_path_condition_worlds();
        }

        if self.state.hit_limit {
            return inference_error_worlds();
        }

        if let Some(max_depth) = self.state.cfg.max_depth {
            if self.state.depth > max_depth {
                self.state.hit_limit = true;
                return inference_error_worlds();
            }
        }

        if self.state.check_time_limit() {
            self.state.hit_limit = true;
            return inference_error_worlds();
        }

        self.state.depth += 1;
        self.state.stats.num_compile_calls += 1;
        self.state.callstack.push(strict_order_index);

        let result = self.compile(expr, env, mode);

        self.state.callstack.pop();
        self.state.depth -= 1;

        if self.state.hit_limit {
            return inference_error_worlds();
        }
        result
    }

    /// Dispatches on the PExpr variant. Each arm forwards to a dedicated
    /// `compile_*` method (or, for one-liners, returns inline).
    fn compile(&mut self, expr: &PExpr, env: &Env, mode: &mut CompileMode<'_>) -> GuardedWorlds {
        let pc = mode.path_condition();
        match expr {
            PExpr::Abs { var, body } => self.compile_abs(*var, body, env, pc),
            PExpr::Var { name } => self.compile_var(*name, env, mode),
            PExpr::Defined { name } => self.compile_defined(*name, mode),
            PExpr::App { func, arg } => self.compile_app(func, arg, env, mode),
            PExpr::Construct { constructor, args } => {
                self.compile_construct(*constructor, args, env, pc)
            }
            PExpr::CaseOf {
                guards,
                scrutinee,
                branches,
            } => self.compile_case_of(scrutinee, guards, branches, env, mode),
            PExpr::YComb { func } => self.compile_ycomb(func, env, mode),
            PExpr::Flip { prob } => self.compile_flip(prob, env, mode),
            PExpr::ConstNative { val } => self.compile_const_native(val, pc),
            PExpr::NativeEq { a, b } => self.compile_native_eq(a, b, env, mode),
            PExpr::NativeLeq { a, b } => self.compile_native_cmp(NativeCmp::Leq, a, b, env, mode),
            PExpr::NativeGeq { a, b } => self.compile_native_cmp(NativeCmp::Geq, a, b, env, mode),
            PExpr::GetArgs { expr: inner } => self.compile_get_args(inner, env, mode),
            PExpr::GetConstructor { expr: inner } => self.compile_get_constructor(inner, env, mode),
            PExpr::Print { expr: inner } => {
                // Print is a no-op in KC mode (side effects aren't meaningful).
                self.traced_compile(inner, env, mode, 0)
            }
            PExpr::Error { .. } => program_error_worlds(),

            PExpr::FAdd { a, b } => {
                self.compile_variable_aware_binop(FloatOp::Add, a, b, env, mode)
            }
            PExpr::FSub { a, b } => {
                self.compile_variable_aware_binop(FloatOp::Sub, a, b, env, mode)
            }
            PExpr::FMul { a, b } => {
                self.compile_variable_aware_binop(FloatOp::Mul, a, b, env, mode)
            }
            PExpr::FDiv { a, b } => {
                self.compile_variable_aware_binop(FloatOp::Div, a, b, env, mode)
            }

            PExpr::MkInt { bitwidth, value } => self.compile_mk_int(bitwidth, value, env, mode),
            PExpr::UniformInt { bitwidth } => self.compile_uniform_int(bitwidth, env, mode),
            PExpr::UniformIntRange { bitwidth, lo, hi } => {
                self.compile_uniform_int_range(bitwidth, lo, hi, env, mode)
            }
            PExpr::IntDistEq { a, b } => self.compile_int_dist_eq(a, b, env, mode),
            PExpr::IntDistAdd { a, b } => self.compile_int_dist_add(a, b, env, mode),
            PExpr::IntDistSub { a, b } => self.compile_int_dist_sub(a, b, env, mode),
            PExpr::IntDistLt { a, b } => self.compile_int_dist_lt(a, b, env, mode),

            PExpr::GSymbol { name } => pure_monad(mk_native_symbol(*name), pc),
            PExpr::GVarSymbol { type_name } => pure_monad(mk_native_symbol(*type_name), pc),

            PExpr::Beta { a, b } => self.compile_beta(a, b, env, mode),
            PExpr::ProbEq { prob, val } => self.compile_prob_eq(prob, val, env, mode),
            PExpr::Gaussian { mu, sigma } => self.compile_gaussian(mu, sigma, env, mode),
            PExpr::RealEq { expr, val } => self.compile_real_eq(expr, val, env, mode),
            PExpr::RealLt { expr, val } => self.compile_real_cmp(RealCmp::Lt, expr, val, env, mode),
            PExpr::RealGeq { expr, val } => {
                self.compile_real_cmp(RealCmp::Geq, expr, val, env, mode)
            }
            PExpr::Dirichlet { alphas } => self.compile_dirichlet(alphas, env, mode),
            PExpr::DirichletEq { expr: dexpr, val } => {
                self.compile_dirichlet_eq(dexpr, val, env, mode)
            }
            PExpr::Index { v, idx } => self.compile_index(v, idx, env, mode),
            PExpr::Categorical { probs } => self.compile_categorical(probs, env, mode),
            PExpr::FloatMatrix { entries, shape } => {
                self.compile_float_matrix(entries, shape, env, mode)
            }
            PExpr::MatMul { a, b } => self.compile_matmul(a, b, env, mode),
            PExpr::Sum { expr: inner } => self.compile_sum(inner, env, mode),
            PExpr::Pin { expr, val } => self.compile_pin(expr, val, env, mode),
            PExpr::Gamma { shape, rate } => self.compile_gamma(shape, rate, env, mode),
            PExpr::Poisson { rate } => {
                self.compile_gamma_consumer(rate, env, mode, GammaDrawFamily::Poisson)
            }
            PExpr::Exponential { rate } => {
                self.compile_gamma_consumer(rate, env, mode, GammaDrawFamily::Exponential)
            }
        }
    }

    fn compile_abs(
        &self,
        var: Symbol,
        body: &PExpr,
        env: &Env,
        pc: BooleanFunction,
    ) -> GuardedWorlds {
        let closure = mk_closure(var, body.clone().into(), env.clone());
        pure_monad(closure, pc)
    }

    fn compile_var(
        &mut self,
        name: Symbol,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        match env_lookup(env, name) {
            Some(val) => match val.as_ref() {
                PluckValue::Thunk(_) | PluckValue::ThunkUnion(_) => self.evaluate_thunk(&val, mode),
                _ => pure_monad(val, mode.path_condition()),
            },
            None => {
                eprintln!("variable not found: {:?}", name);
                program_error_worlds()
            }
        }
    }

    fn compile_defined(&mut self, name: Symbol, mode: &mut CompileMode<'_>) -> GuardedWorlds {
        match self.defs.lookup(name) {
            Some(def) => {
                let expr = def.expr.clone();
                let env = empty_env();
                self.traced_compile(&expr, &env, mode, 0)
            }
            None => {
                eprintln!("undefined: {:?}", name);
                program_error_worlds()
            }
        }
    }

    fn compile_construct(
        &mut self,
        constructor: Symbol,
        args: &[PExpr],
        env: &Env,
        pc: BooleanFunction,
    ) -> GuardedWorlds {
        let thunked_args: Vec<PluckVal> = args
            .iter()
            .enumerate()
            .map(|(i, arg)| make_thunk(arg, env.clone(), i as i32, self.state))
            .collect();

        let val = mk_value(constructor, thunked_args);
        pure_monad(val, pc)
    }

    fn compile_app(
        &mut self,
        func: &PExpr,
        arg: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let arg_thunk = make_thunk(arg, env.clone(), 1, self.state);
        let func_worlds = self.traced_compile(func, env, mode, 0);

        bind_compile(
            func_worlds,
            pc,
            self,
            mode,
            |func_val, _pc, ctx, mode| match func_val.as_ref() {
                PluckValue::Closure(closure_data) => {
                    let mut new_env = env_cons(
                        closure_data.var,
                        arg_thunk.clone(),
                        closure_data.env.clone(),
                    );
                    if let Some(rec_name) = closure_data.rec_name {
                        if let Some(self_ref) = closure_data.self_ref.borrow().as_ref() {
                            new_env = env_cons(rec_name, self_ref.clone(), new_env);
                        }
                    }
                    ctx.traced_compile(&closure_data.body, &new_env, mode, 2)
                }
                _ => {
                    eprintln!("tried to apply a non-function value");
                    program_error_worlds()
                }
            },
        )
    }

    fn compile_case_of(
        &mut self,
        scrutinee: &PExpr,
        guards: &[CaseOfGuard],
        branches: &[PExpr],
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        // Captured (not read off the cont's mode) so the diagnostic still
        // fires while the continuation runs under a bind-constructed KC mode.
        let in_sample_mode = matches!(mode, CompileMode::Sample(_));
        let scrutinee_worlds = self.traced_compile(scrutinee, env, mode, 0);

        bind_compile(
            scrutinee_worlds,
            pc,
            self,
            mode,
            |scrutinee_val, _pc, ctx, mode| {
                match scrutinee_val.as_ref() {
                    PluckValue::Value { constructor, args } => {
                        for (i, guard) in guards.iter().enumerate() {
                            if guard.constructor == *constructor {
                                let mut new_env = env.clone();
                                for (j, arg_name) in guard.args.iter().enumerate() {
                                    if j < args.len() {
                                        new_env = env_cons(*arg_name, args[j].clone(), new_env);
                                    }
                                }
                                return ctx.traced_compile(
                                    &branches[i],
                                    &new_env,
                                    mode,
                                    (i + 1) as i32,
                                );
                            }
                        }
                        // No matching guard — pattern match failure.
                        // Expected behavior in probabilistic programs (e.g., `given` matches True only).
                        if in_sample_mode {
                            eprintln!("incomplete case/match encountered in sampling mode")
                        }
                        program_error_worlds()
                    }
                    _ => {
                        eprintln!("case/match on non-value: {:?}", scrutinee_val);
                        program_error_worlds()
                    }
                }
            },
        )
    }

    fn compile_ycomb(
        &mut self,
        func: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let func_worlds = self.traced_compile(func, env, mode, 0);

        bind_compile(
            func_worlds,
            pc,
            self,
            mode,
            |func_val, pc, _ctx, _mode| match func_val.as_ref() {
                PluckValue::Closure(outer_closure) => {
                    let rec_name = outer_closure.var;
                    match outer_closure.body.as_ref() {
                        PExpr::Abs {
                            var: arg_name,
                            body,
                        } => {
                            let closure = mk_y_closure(
                                body.clone().into(),
                                outer_closure.env.clone(),
                                rec_name,
                                *arg_name,
                            );
                            pure_monad(closure, pc)
                        }
                        _ => {
                            eprintln!("Y combinator expects a function returning a function");
                            program_error_worlds()
                        }
                    }
                }
                _ => {
                    eprintln!("Y combinator applied to non-function");
                    program_error_worlds()
                }
            },
        )
    }

    fn compile_flip(
        &mut self,
        prob: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        if matches!(mode, CompileMode::Sample(_)) {
            return self.sample_compile_flip(prob, env, mode);
        }

        let pc = mode.path_condition();
        let prob_worlds = self.traced_compile(prob, env, mode, 0);

        bind_compile(prob_worlds, pc, self, mode, |prob_val, pc, ctx, _mode| {
            let flip_weight = flip_weight_from(prob_val.as_ref());

            if let FlipWeight::Constant(p) = &flip_weight {
                if let Some(v) = ctx.constant_flip_shortcut(*p) {
                    return pure_monad(v, pc);
                }
            }

            let flip_var = ctx
                .state
                .with_callstack_index(1, |s| s.current_address_weighted(flip_weight));

            let true_val = ctx.bool_val(true);
            let false_val = ctx.bool_val(false);
            if_then_else_monad(true_val, false_val, flip_var, pc, ctx)
        })
    }

    fn compile_const_native(&self, val: &NativeVal, pc: BooleanFunction) -> GuardedWorlds {
        let pluck_val = match val {
            NativeVal::Float(f) => mk_native_float(*f),
            NativeVal::Int(i) => mk_native_int(*i),
            NativeVal::Symbol(s) => mk_native_symbol(*s),
            NativeVal::Bool(b) => mk_native_bool(*b),
        };
        pure_monad(pluck_val, pc)
    }

    fn compile_native_eq(
        &mut self,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);

        bind_compile(a_worlds, pc, self, mode, |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, |b_val, pc, ctx, mode| {
                // A Poisson draw compared to a native int / Peano nat is
                // an observation event on the draw.
                let poisson = match (a_val.as_ref(), b_val.as_ref()) {
                    (PluckValue::Poisson { name, gamma, scale }, _) => {
                        Some((*name, *gamma, *scale, b_val.clone()))
                    }
                    (_, PluckValue::Poisson { name, gamma, scale }) => {
                        Some((*name, *gamma, *scale, a_val.clone()))
                    }
                    _ => None,
                };
                if let Some((draw, gamma, scale, other)) = poisson {
                    let k = ctx.force_int(&other, mode).unwrap_or_else(|| {
                        panic!(
                            "native_eq: a Poisson draw must be compared to a \
                             native int or nat literal, got {:?}",
                            other
                        )
                    });
                    if k < 0 {
                        // Poisson support is the non-negative integers.
                        return pure_monad(ctx.bool_val(false), pc);
                    }
                    let flip = GammaFlip::PoissonEvent {
                        gamma,
                        draw,
                        event: IntervalOrEq::eq(k as u64),
                        scale,
                    };
                    if matches!(mode, CompileMode::Sample(_)) {
                        return ctx.sample_gamma_flip_outcome(
                            flip,
                            GammaDrawFamily::Poisson,
                            gamma,
                            draw,
                            scale,
                            mode,
                        );
                    }
                    return ctx.gamma_flip_worlds(flip, pc);
                }
                let eq = match (a_val.as_ref(), b_val.as_ref()) {
                    (PluckValue::Native(a), PluckValue::Native(b)) => a == b,
                    _ => false,
                };
                pure_monad(ctx.bool_val(eq), pc)
            })
        })
    }

    /// Integer range comparison: `native_leq` / `native_geq` over native
    /// ints and Peano nats. When one side is a Poisson draw, compiles an
    /// observation event on the draw (the comparison is mirrored when
    /// the draw is on the right).
    fn compile_native_cmp(
        &mut self,
        cmp: NativeCmp,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);

        bind_compile(a_worlds, pc, self, mode, move |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, move |b_val, pc, ctx, mode| {
                let op_name = match cmp {
                    NativeCmp::Leq => "native_leq",
                    NativeCmp::Geq => "native_geq",
                };
                let poisson = match (a_val.as_ref(), b_val.as_ref()) {
                    (PluckValue::Poisson { name, gamma, scale }, _) => {
                        Some((*name, *gamma, *scale, b_val.clone(), cmp))
                    }
                    (_, PluckValue::Poisson { name, gamma, scale }) => {
                        Some((*name, *gamma, *scale, a_val.clone(), cmp.mirror()))
                    }
                    _ => None,
                };
                if let Some((draw, gamma, scale, other, cmp)) = poisson {
                    let n = ctx.force_int(&other, mode).unwrap_or_else(|| {
                        panic!(
                            "{op_name}: a Poisson draw must be compared to a \
                             native int or nat literal, got {:?}",
                            other
                        )
                    });
                    // Trivial events given the draw's support [0, ∞).
                    match cmp {
                        NativeCmp::Leq if n < 0 => return pure_monad(ctx.bool_val(false), pc),
                        NativeCmp::Geq if n <= 0 => return pure_monad(ctx.bool_val(true), pc),
                        _ => {}
                    }
                    let event = match cmp {
                        // k ≤ n ⇔ k ∈ [0, n+1).
                        NativeCmp::Leq => IntervalOrEq::lt(n as u64 + 1),
                        NativeCmp::Geq => IntervalOrEq::geq(n as u64),
                    };
                    let flip = GammaFlip::PoissonEvent {
                        gamma,
                        draw,
                        event,
                        scale,
                    };
                    if matches!(mode, CompileMode::Sample(_)) {
                        return ctx.sample_gamma_flip_outcome(
                            flip,
                            GammaDrawFamily::Poisson,
                            gamma,
                            draw,
                            scale,
                            mode,
                        );
                    }
                    return ctx.gamma_flip_worlds(flip, pc);
                }
                // Deterministic comparison over ints / Peano nats.
                let ka = ctx.force_int(&a_val, mode);
                let kb = ctx.force_int(&b_val, mode);
                match (ka, kb) {
                    (Some(x), Some(y)) => {
                        let holds = match cmp {
                            NativeCmp::Leq => x <= y,
                            NativeCmp::Geq => x >= y,
                        };
                        pure_monad(ctx.bool_val(holds), pc)
                    }
                    _ => panic!(
                        "{op_name}: expected native ints / nat literals or a \
                         Poisson draw, got {:?} and {:?}",
                        a_val, b_val
                    ),
                }
            })
        })
    }

    /// Force `val` to a concrete integer: a native int directly, or a
    /// Peano `S`/`O` chain whose (possibly thunked) spine is forced
    /// level by level. Returns `None` for non-integer shapes. Panics if
    /// a spine level is stochastic — integer comparands must be
    /// deterministic.
    fn force_int(&mut self, val: &PluckVal, mode: &mut CompileMode<'_>) -> Option<i64> {
        let (o_sym, s_sym) = self
            .types
            .find_nat_constructors()
            .expect("force_int: nat type not registered");
        let mut count: i64 = 0;
        let mut current = val.clone();
        loop {
            match current.as_ref() {
                PluckValue::Native(NativeVal::Int(i)) => return Some(i + count),
                PluckValue::Value { constructor, args }
                    if *constructor == o_sym && args.is_empty() =>
                {
                    return Some(count)
                }
                PluckValue::Value { constructor, args }
                    if *constructor == s_sym && args.len() == 1 =>
                {
                    count += 1;
                    let (mut worlds, _) = self.evaluate_thunk(&args[0], mode);
                    assert_eq!(
                        worlds.len(),
                        1,
                        "force_int: stochastic nat spine — integer comparands \
                         must be deterministic"
                    );
                    current = worlds.pop().unwrap().0;
                }
                _ => return None,
            }
        }
    }

    /// Create the BDD variable for a gamma-family observation flip and
    /// return True/False worlds split on it.
    fn gamma_flip_worlds(&mut self, flip: GammaFlip, pc: BooleanFunction) -> GuardedWorlds {
        let obs_var = self
            .state
            .with_callstack_index(2, |s| s.current_address_weighted(FlipWeight::Gamma(flip)));
        let true_val = self.bool_val(true);
        let false_val = self.bool_val(false);
        if_then_else_monad(true_val, false_val, obs_var, pc, self)
    }

    /// Sample-mode resolution of a gamma-family interval flip: cover the
    /// draw, register the flip variable (under the same callstack index
    /// as `gamma_flip_worlds`, so evidence-compiled and sample-compiled
    /// keys agree), and resolve it with the indicator probability at the
    /// assignment's draw value — mirroring `sample_compile_flip`.
    fn sample_gamma_flip_outcome(
        &mut self,
        flip: GammaFlip,
        family: GammaDrawFamily,
        rate: u64,
        draw: u64,
        scale: f64,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let CompileMode::Sample(sample_state) = mode else {
            unreachable!("sample_gamma_flip_outcome called outside Sample mode")
        };
        sample_state.ensure_gamma_draw(rate, draw, family, scale);
        let weight = FlipWeight::Gamma(flip);
        let p = weight
            .sample_probability(&sample_state.assignment)
            .expect("gamma interval flips have an indicator probability");
        let discriminator = weight.discriminator();
        let key: super::state::CallstackKey = self.state.with_callstack_index(2, |s| {
            let k = (s.callstack.hash(), discriminator);
            s.current_address_weighted(weight);
            k
        });
        let outcome = crate::inference::sampling::resolve_flip(p, key, sample_state, &*self.state);
        let val = self.bool_val(outcome);
        pure_monad(val, BooleanFunction::true_ptr())
    }

    fn compile_get_args(
        &mut self,
        inner: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let worlds = self.traced_compile(inner, env, mode, 0);
        let (nil_s, cons_s) = self
            .types
            .find_list_constructors()
            .expect("list type not registered");

        bind_compile(
            worlds,
            pc,
            self,
            mode,
            move |val, pc, _ctx, _mode| match val.as_ref() {
                PluckValue::Value { args, .. } => {
                    let mut list = mk_value(nil_s, vec![]);
                    for arg in args.iter().rev() {
                        list = mk_value(cons_s, vec![arg.clone(), list]);
                    }
                    pure_monad(list, pc)
                }
                _ => pure_monad(val, pc),
            },
        )
    }

    fn compile_get_constructor(
        &mut self,
        inner: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let worlds = self.traced_compile(inner, env, mode, 0);

        bind_compile(worlds, pc, self, mode, |val, pc, _ctx, _mode| {
            match val.as_ref() {
                PluckValue::Value { constructor, .. } => {
                    let sym_val = mk_native_symbol(*constructor);
                    pure_monad(sym_val, pc)
                }
                _ => pure_monad(val, pc),
            }
        })
    }

    fn compile_mk_int(
        &mut self,
        bitwidth: &PExpr,
        value: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let bw_worlds = self.traced_compile(bitwidth, env, mode, 0);

        bind_compile(bw_worlds, pc, self, mode, |bw_val, pc, ctx, mode| {
            let val_worlds = ctx.traced_compile(value, env, mode, 1);
            bind_compile(val_worlds, pc, ctx, mode, |v_val, pc, _ctx, _mode| {
                let width = as_int_or_panic(bw_val.as_ref(), "mk_int bitwidth") as usize;
                let v = as_int_or_panic(v_val.as_ref(), "mk_int value") as u64;

                let bits: Vec<BooleanFunction> = (0..width)
                    .map(|i| {
                        if (v >> i) & 1 == 1 {
                            BooleanFunction::true_ptr()
                        } else {
                            BooleanFunction::false_ptr()
                        }
                    })
                    .collect();

                pure_monad(mk_int_dist(bits), pc)
            })
        })
    }

    fn compile_uniform_int(
        &mut self,
        bitwidth: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        if matches!(mode, CompileMode::Sample(_)) {
            return self.sample_compile_uniform_int(bitwidth, env, mode);
        }

        let pc = mode.path_condition();
        let bw_worlds = self.traced_compile(bitwidth, env, mode, 0);

        bind_compile(bw_worlds, pc, self, mode, |bw_val, pc, ctx, _mode| {
            let width = as_int_or_panic(bw_val.as_ref(), "uniform_int bitwidth") as usize;

            let bits: Vec<BooleanFunction> = (0..width)
                .map(|i| {
                    ctx.state
                        .with_callstack_index(i as i32, |s| s.current_address(0.5))
                })
                .collect();

            pure_monad(mk_int_dist(bits), pc)
        })
    }

    /// Sample-mode counterpart of `compile_uniform_int`: resolve each bit
    /// via `resolve_flip` at p = 0.5 instead of allocating symbolic bit
    /// BDDs. Each bit registers its variable with the same
    /// `with_callstack_index(i)` + `current_address(0.5)` addressing as the
    /// KC path, so evidence/pin constraints over these variables read off
    /// correctly. One concrete world, no symbolic bits.
    fn sample_compile_uniform_int(
        &mut self,
        bitwidth: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let bw_val = self.sample_compile_one(bitwidth, env, mode, 0, "uniform_int: bitwidth");
        let width = as_int_or_panic(bw_val.as_ref(), "uniform_int bitwidth") as usize;

        let CompileMode::Sample(sample_state) = mode else {
            unreachable!()
        };
        let mut bits: Vec<BooleanFunction> = Vec::with_capacity(width);
        for i in 0..width {
            let key = self.state.with_callstack_index(i as i32, |s| {
                let k = (s.callstack.hash(), 0.5_f64.to_bits());
                s.current_address(0.5);
                k
            });
            let bit =
                crate::inference::sampling::resolve_flip(0.5, key, sample_state, &*self.state);
            bits.push(if bit {
                BooleanFunction::true_ptr()
            } else {
                BooleanFunction::false_ptr()
            });
        }
        pure_monad(mk_int_dist(bits), BooleanFunction::true_ptr())
    }

    fn compile_uniform_int_range(
        &mut self,
        bitwidth: &PExpr,
        lo: &PExpr,
        hi: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        if matches!(mode, CompileMode::Sample(_)) {
            return self.sample_compile_uniform_int_range(bitwidth, lo, hi, env, mode);
        }

        let pc = mode.path_condition();
        let bw_worlds = self.traced_compile(bitwidth, env, mode, 0);

        bind_compile(bw_worlds, pc, self, mode, |bw_val, pc, ctx, mode| {
            let lo_worlds = ctx.traced_compile(lo, env, mode, 1);
            bind_compile(lo_worlds, pc, ctx, mode, |lo_val, pc, ctx, mode| {
                let hi_worlds = ctx.traced_compile(hi, env, mode, 2);
                bind_compile(hi_worlds, pc, ctx, mode, |hi_val, pc, ctx, _mode| {
                    let width =
                        as_int_or_panic(bw_val.as_ref(), "uniform_int_range bitwidth") as usize;
                    let start = as_int_or_panic(lo_val.as_ref(), "uniform_int_range lo");
                    let stop = as_int_or_panic(hi_val.as_ref(), "uniform_int_range hi");

                    assert!(stop >= start, "uniform_int_range: hi must be >= lo");

                    let mut bits: Vec<BooleanFunction> = vec![BooleanFunction::false_ptr(); width];

                    encode_range(
                        start,
                        stop,
                        BooleanFunction::true_ptr(),
                        &mut bits,
                        width,
                        ctx.state,
                        0,
                    );

                    pure_monad(mk_int_dist(bits), pc)
                })
            })
        })
    }

    /// Sample-mode counterpart of `compile_uniform_int_range`. Walks **one
    /// root-to-leaf path** of the same binary-split tree as `encode_range`
    /// instead of allocating the whole encoding (O(log range) draws and
    /// variable registrations instead of O(range) variables + bit BDDs,
    /// then emits the landed value's concrete bits — one world, in-range by
    /// construction.
    ///
    /// Variable-identity discipline: each split registers its variable with
    /// the **identical** addressing to `encode_range` —
    /// `with_callstack_index(depth_idx)` + `current_address(p)`, key
    /// `(callstack_hash, p.to_bits())` — so evidence/pin constraints over
    /// these variables read off correctly via `resolve_flip`. Sibling
    /// subtrees at equal depth with equal `p` share a variable in KC's
    /// encoding; the path walk visits one node per depth, and the
    /// conditional probabilities along the path are KC's, so the
    /// distributions agree (uniform over `[lo, hi]`).
    fn sample_compile_uniform_int_range(
        &mut self,
        bitwidth: &PExpr,
        lo: &PExpr,
        hi: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let bw_val = self.sample_compile_one(bitwidth, env, mode, 0, "uniform_int_range: bitwidth");
        let lo_val = self.sample_compile_one(lo, env, mode, 1, "uniform_int_range: lo");
        let hi_val = self.sample_compile_one(hi, env, mode, 2, "uniform_int_range: hi");

        let width = as_int_or_panic(bw_val.as_ref(), "uniform_int_range bitwidth") as usize;
        let start = as_int_or_panic(lo_val.as_ref(), "uniform_int_range lo");
        let stop = as_int_or_panic(hi_val.as_ref(), "uniform_int_range hi");
        assert!(stop >= start, "uniform_int_range: hi must be >= lo");

        let CompileMode::Sample(sample_state) = mode else {
            unreachable!()
        };
        let mut lo_cur = start;
        let mut hi_cur = stop;
        let mut depth_idx: i32 = 0;
        while lo_cur != hi_cur {
            // Same split arithmetic as encode_range.
            let mid = lo_cur + (hi_cur - lo_cur) / 2;
            let lower_size = (mid - lo_cur + 1) as f64;
            let upper_size = (hi_cur - mid) as f64;
            let p = lower_size / (lower_size + upper_size);

            let key = self.state.with_callstack_index(depth_idx, |s| {
                let k = (s.callstack.hash(), p.to_bits());
                s.current_address(p);
                k
            });
            let take_lower =
                crate::inference::sampling::resolve_flip(p, key, sample_state, &*self.state);
            if take_lower {
                hi_cur = mid;
            } else {
                lo_cur = mid + 1;
            }
            depth_idx += 1;
        }

        let bits: Vec<BooleanFunction> = (0..width)
            .map(|i| {
                if ((lo_cur as u64) >> i) & 1 == 1 {
                    BooleanFunction::true_ptr()
                } else {
                    BooleanFunction::false_ptr()
                }
            })
            .collect();
        pure_monad(mk_int_dist(bits), BooleanFunction::true_ptr())
    }

    fn compile_int_dist_eq(
        &mut self,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);

        bind_compile(a_worlds, pc, self, mode, |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, |b_val, pc, ctx, _mode| {
                let a_bits = as_int_dist_bits(a_val.as_ref(), "int_dist_eq: first argument");
                let b_bits = as_int_dist_bits(b_val.as_ref(), "int_dist_eq: second argument");

                let a_int = IntDist::new(a_bits.to_vec());
                let b_int = IntDist::new(b_bits.to_vec());

                let eq_bf = int_dist_eq(&a_int, &b_int, ctx.state.fac());

                let true_val = ctx.bool_val(true);
                let false_val = ctx.bool_val(false);
                if_then_else_monad(true_val, false_val, eq_bf, pc, ctx)
            })
        })
    }

    fn compile_int_dist_add(
        &mut self,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);

        bind_compile(a_worlds, pc, self, mode, |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, |b_val, pc, ctx, _mode| {
                let a_bits = as_int_dist_bits(a_val.as_ref(), "int_add: first argument");
                let b_bits = as_int_dist_bits(b_val.as_ref(), "int_add: second argument");

                let a_int = IntDist::new(a_bits.to_vec());
                let b_int = IntDist::new(b_bits.to_vec());
                let result = int_dist_add(&a_int, &b_int, ctx.state.fac());
                pure_monad(mk_int_dist(result.bits), pc)
            })
        })
    }

    fn compile_int_dist_sub(
        &mut self,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);

        bind_compile(a_worlds, pc, self, mode, |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, |b_val, pc, ctx, _mode| {
                let a_bits = as_int_dist_bits(a_val.as_ref(), "int_sub: first argument");
                let b_bits = as_int_dist_bits(b_val.as_ref(), "int_sub: second argument");

                let a_int = IntDist::new(a_bits.to_vec());
                let b_int = IntDist::new(b_bits.to_vec());
                let result = int_dist_sub(&a_int, &b_int, ctx.state.fac());
                pure_monad(mk_int_dist(result.bits), pc)
            })
        })
    }

    fn compile_int_dist_lt(
        &mut self,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);

        bind_compile(a_worlds, pc, self, mode, |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, |b_val, pc, ctx, _mode| {
                let a_bits = as_int_dist_bits(a_val.as_ref(), "int_lt: first argument");
                let b_bits = as_int_dist_bits(b_val.as_ref(), "int_lt: second argument");

                let a_int = IntDist::new(a_bits.to_vec());
                let b_int = IntDist::new(b_bits.to_vec());
                let lt_bf = int_dist_lt(&a_int, &b_int, ctx.state.fac());

                let true_val = ctx.bool_val(true);
                let false_val = ctx.bool_val(false);
                if_then_else_monad(true_val, false_val, lt_bf, pc, ctx)
            })
        })
    }

    fn compile_beta(
        &mut self,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);

        bind_compile(a_worlds, pc, self, mode, |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, |b_val, pc, ctx, mode| {
                let a_f = as_float_or_panic(
                    a_val.as_ref(),
                    "beta: first argument must be a number (use 1.0 not 1)",
                );
                let b_f =
                    as_float_or_panic(b_val.as_ref(), "beta: second argument must be a number");

                let name = ctx.state.current_continuous_name(&[a_f, b_f]);
                ctx.state.register_prior(BetaPrior::Beta {
                    name,
                    a: a_f,
                    b: b_f,
                });
                // First registration can happen mid-sample (forward
                // execution reaching a body the evidence never compiled);
                // cover the variable in this sample's assignment.
                if let CompileMode::Sample(state) = mode {
                    state.ensure_continuous([name], &ctx.state.prior_registry);
                }
                pure_monad(mk_probability(name), pc)
            })
        })
    }

    fn compile_prob_eq(
        &mut self,
        prob: &PExpr,
        val: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        if matches!(mode, CompileMode::Sample(_)) {
            return self.sample_compile_prob_eq(prob, val, env, mode);
        }

        let pc = mode.path_condition();
        let prob_worlds = self.traced_compile(prob, env, mode, 0);

        bind_compile(prob_worlds, pc, self, mode, |prob_val, pc, ctx, mode| {
            let val_worlds = ctx.traced_compile(val, env, mode, 1);
            bind_compile(val_worlds, pc, ctx, mode, |val_val, pc, ctx, _mode| {
                let r = as_float_or_panic(
                    val_val.as_ref(),
                    "prob_eq: second argument must be a number",
                );
                match prob_val.as_ref() {
                    PluckValue::Probability(name) => {
                        let pin_var = ctx.state.with_callstack_index(2, |s| {
                            s.current_address_weighted(FlipWeight::Beta(BetaFlip::ProbPin(
                                *name, r,
                            )))
                        });
                        let true_val = ctx.bool_val(true);
                        let false_val = ctx.bool_val(false);
                        if_then_else_monad(true_val, false_val, pin_var, pc, ctx)
                    }
                    PluckValue::Native(NativeVal::Float(c)) => {
                        let result = ctx.bool_val((*c - r).abs() < 1e-15);
                        pure_monad(result, pc)
                    }
                    _ => panic!("prob_eq: first argument must be a Probability or float"),
                }
            })
        })
    }

    fn compile_gaussian(
        &mut self,
        mu: &PExpr,
        sigma: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let mu_worlds = self.traced_compile(mu, env, mode, 0);

        bind_compile(mu_worlds, pc, self, mode, |mu_val, pc, ctx, mode| {
            let sigma_worlds = ctx.traced_compile(sigma, env, mode, 1);
            bind_compile(sigma_worlds, pc, ctx, mode, |sigma_val, pc, ctx, mode| {
                // Univariate path: both args are scalars. sigma here is the
                // standard deviation (existing contract).
                if matches!(
                    mu_val.as_ref(),
                    PluckValue::Native(NativeVal::Float(_)) | PluckValue::Native(NativeVal::Int(_))
                ) && matches!(
                    sigma_val.as_ref(),
                    PluckValue::Native(NativeVal::Float(_)) | PluckValue::Native(NativeVal::Int(_))
                ) {
                    let mu_f = as_float_or_panic(mu_val.as_ref(), "gaussian: mu must be a number");
                    let sigma_f =
                        as_float_or_panic(sigma_val.as_ref(), "gaussian: sigma must be a number");
                    let name = ctx.state.current_continuous_name(&[mu_f, sigma_f]);
                    ctx.state
                        .register_prior(GaussianPrior::from_single(name, mu_f, sigma_f));
                    // Mid-sample first registration: cover the variable in
                    // this sample's assignment (see SampleState::ensure_continuous).
                    if let CompileMode::Sample(state) = mode {
                        state.ensure_continuous([name], &ctx.state.prior_registry);
                    }
                    let mut coefficients = std::collections::BTreeMap::new();
                    coefficients.insert(name, ordered_float::NotNan::new(1.0).unwrap());
                    return pure_monad(mk_gaussian_expr(0.0, coefficients), pc);
                }

                // Multivariate path: mu is a 1D FloatMatrix of length N, sigma
                // is a 2D NxN FloatMatrix (treated as a *covariance* matrix).
                let (mu_entries, mu_shape) = match mu_val.as_ref() {
                    PluckValue::FloatMatrix { entries, shape } => (entries.clone(), shape.clone()),
                    _ => panic!(
                        "gaussian: mu must be a scalar or a 1D FloatMatrix, got {:?}",
                        mu_val
                    ),
                };
                let (cov_entries, cov_shape) = match sigma_val.as_ref() {
                    PluckValue::FloatMatrix { entries, shape } => (entries.clone(), shape.clone()),
                    _ => panic!(
                        "gaussian: sigma must be a scalar or a 2D FloatMatrix, got {:?}",
                        sigma_val
                    ),
                };
                assert_eq!(
                    mu_shape.len(),
                    1,
                    "gaussian: mu matrix must be 1D, got shape {:?}",
                    mu_shape
                );
                let n = mu_shape[0];
                assert_eq!(
                    cov_shape,
                    vec![n, n],
                    "gaussian: cov must have shape [{}, {}], got {:?}",
                    n,
                    n,
                    cov_shape
                );

                // Force every entry of mu and cov to a concrete f64.
                let mut mu_vec = nalgebra::DVector::<f64>::zeros(n);
                for (i, e) in mu_entries.iter().enumerate() {
                    let (c, m) = force_entry_to_affine(ctx, e, pc.clone());
                    assert!(
                        m.is_empty(),
                        "gaussian: mu[{}] depends on a symbolic Gaussian; mu must be deterministic",
                        i
                    );
                    mu_vec[i] = c;
                }
                let mut cov_mat = nalgebra::DMatrix::<f64>::zeros(n, n);
                for r in 0..n {
                    for c_idx in 0..n {
                        let (cval, m) =
                            force_entry_to_affine(ctx, &cov_entries[r * n + c_idx], pc.clone());
                        assert!(
                            m.is_empty(),
                            "gaussian: cov[{},{}] depends on a symbolic Gaussian; cov must be deterministic",
                            r,
                            c_idx
                        );
                        cov_mat[(r, c_idx)] = cval;
                    }
                }
                // Symmetry check.
                for r in 0..n {
                    for c_idx in (r + 1)..n {
                        let asym = (cov_mat[(r, c_idx)] - cov_mat[(c_idx, r)]).abs();
                        assert!(
                            asym < 1e-9,
                            "gaussian: cov is not symmetric at [{},{}] (|diff|={})",
                            r,
                            c_idx,
                            asym
                        );
                    }
                }
                // PSD check via Cholesky. nalgebra's `cholesky` returns Option.
                assert!(
                    cov_mat.clone().cholesky().is_some(),
                    "gaussian: cov is not positive semi-definite"
                );

                // Dedup key: callstack + flattened (mu, cov).
                let mut key_params: Vec<f64> = Vec::with_capacity(n + n * n);
                key_params.extend(mu_vec.iter().copied());
                for r in 0..n {
                    for c in 0..n {
                        key_params.push(cov_mat[(r, c)]);
                    }
                }
                let names = ctx.state.current_continuous_names(&key_params, n);

                ctx.state.register_prior(GaussianPrior::from_multivariate(
                    names.clone(),
                    mu_vec,
                    cov_mat,
                ));
                // Mid-sample first registration: cover the block in this
                // sample's assignment (see SampleState::ensure_continuous).
                if let CompileMode::Sample(state) = mode {
                    state.ensure_continuous(names.iter().copied(), &ctx.state.prior_registry);
                }

                // Build the 1D FloatMatrix whose i-th entry is the symbolic
                // affine "0 + 1*names[i]".
                let one = ordered_float::NotNan::new(1.0).unwrap();
                let entries: Vec<PluckVal> = names
                    .iter()
                    .map(|&nm| {
                        let mut coeffs = std::collections::BTreeMap::new();
                        coeffs.insert(nm, one);
                        mk_gaussian_expr(0.0, coeffs)
                    })
                    .collect();
                pure_monad(mk_float_matrix(entries, vec![n]), pc)
            })
        })
    }

    fn compile_real_eq(
        &mut self,
        expr: &PExpr,
        val: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        if matches!(mode, CompileMode::Sample(_)) {
            return self.sample_compile_real_eq(expr, val, env, mode);
        }

        let pc = mode.path_condition();
        let expr_worlds = self.traced_compile(expr, env, mode, 0);

        bind_compile(expr_worlds, pc, self, mode, |expr_val, pc, ctx, mode| {
            let val_worlds = ctx.traced_compile(val, env, mode, 1);
            bind_compile(val_worlds, pc, ctx, mode, |val_v, pc, ctx, _mode| {
                let observed = as_float_or_panic(val_v.as_ref(), "real_eq: val must be a number");
                match expr_val.as_ref() {
                    PluckValue::GaussianExpr {
                        constant,
                        coefficients,
                    } => {
                        if coefficients.is_empty() {
                            // Degenerate: no Gaussian variables, just compare floats.
                            let result = ctx.bool_val((*constant - observed).abs() < 1e-15);
                            pure_monad(result, pc)
                        } else {
                            let coeffs_f64 = coefficients
                                .iter()
                                .map(|(k, v)| (*k, v.into_inner()))
                                .collect();
                            let weight = FlipWeight::Gaussian(GaussianFlip::Obs {
                                coefficients: coeffs_f64,
                                constant: *constant,
                                observed,
                            });
                            let obs_var = ctx
                                .state
                                .with_callstack_index(2, |s| s.current_address_weighted(weight));
                            let true_val = ctx.bool_val(true);
                            let false_val = ctx.bool_val(false);
                            if_then_else_monad(true_val, false_val, obs_var, pc, ctx)
                        }
                    }
                    PluckValue::Native(NativeVal::Float(f)) => {
                        let result = ctx.bool_val((*f - observed).abs() < 1e-15);
                        pure_monad(result, pc)
                    }
                    PluckValue::Gamma { name, scale } => {
                        // `(real_eq (*. g s) v)` pins `g = v/s`; the unsat
                        // check is on `v` since `v > 0 ⇔ v/s > 0` for `s > 0`.
                        if observed <= 0.0 {
                            // Gamma support is (0, ∞): the pin is unsatisfiable.
                            pure_monad(ctx.bool_val(false), pc)
                        } else {
                            ctx.gamma_flip_worlds(
                                GammaFlip::RatePin {
                                    name: *name,
                                    value: observed,
                                    scale: *scale,
                                },
                                pc,
                            )
                        }
                    }
                    PluckValue::Exponential { name, gamma, scale } => {
                        if observed < 0.0 {
                            // Exp support is [0, ∞).
                            pure_monad(ctx.bool_val(false), pc)
                        } else {
                            ctx.gamma_flip_worlds(
                                GammaFlip::ExpEq {
                                    gamma: *gamma,
                                    draw: *name,
                                    value: observed,
                                    scale: *scale,
                                },
                                pc,
                            )
                        }
                    }
                    PluckValue::Poisson { .. } => {
                        panic!("real_eq: Poisson draws are integer-valued; use native_eq")
                    }
                    _ => panic!(
                        "real_eq: expr must be a GaussianExpr, Gamma, Exponential \
                         or float, got {:?}",
                        expr_val
                    ),
                }
            })
        })
    }

    /// Real range comparison: `real_lt` / `real_geq` over Exponential
    /// draws (or plain floats). Each compiles to one BDD flip whose
    /// false branch is the complementary half-open event.
    fn compile_real_cmp(
        &mut self,
        cmp: RealCmp,
        expr: &PExpr,
        val: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let expr_worlds = self.traced_compile(expr, env, mode, 0);

        bind_compile(
            expr_worlds,
            pc,
            self,
            mode,
            move |expr_val, pc, ctx, mode| {
                let val_worlds = ctx.traced_compile(val, env, mode, 1);
                bind_compile(val_worlds, pc, ctx, mode, move |val_v, pc, ctx, mode| {
                    let op_name = match cmp {
                        RealCmp::Lt => "real_lt",
                        RealCmp::Geq => "real_geq",
                    };
                    let bound = as_float_or_panic(
                        val_v.as_ref(),
                        &format!("{op_name}: bound must be a number"),
                    );
                    match expr_val.as_ref() {
                        PluckValue::Native(NativeVal::Float(f)) => {
                            let holds = match cmp {
                                RealCmp::Lt => *f < bound,
                                RealCmp::Geq => *f >= bound,
                            };
                            pure_monad(ctx.bool_val(holds), pc)
                        }
                        PluckValue::Exponential { name, gamma, scale } => {
                            // Exp support is [0, ∞): a non-positive bound
                            // makes the event trivial ([0, b) is empty,
                            // [b, ∞) is the full support).
                            if bound <= 0.0 {
                                let holds = matches!(cmp, RealCmp::Geq);
                                return pure_monad(ctx.bool_val(holds), pc);
                            }
                            let b = NotNan::new(bound).unwrap();
                            let interval = match cmp {
                                RealCmp::Lt => Interval::lt(b),
                                RealCmp::Geq => Interval::geq(b),
                            };
                            let flip = GammaFlip::ExpInterval {
                                gamma: *gamma,
                                draw: *name,
                                interval,
                                scale: *scale,
                            };
                            if matches!(mode, CompileMode::Sample(_)) {
                                return ctx.sample_gamma_flip_outcome(
                                    flip,
                                    GammaDrawFamily::Exponential,
                                    *gamma,
                                    *name,
                                    *scale,
                                    mode,
                                );
                            }
                            ctx.gamma_flip_worlds(flip, pc)
                        }
                        v => panic!(
                            "{op_name}: range observations are only supported on \
                         Exponential draws (or plain floats), got {:?}",
                            v
                        ),
                    }
                })
            },
        )
    }

    /// Pin a symbolic expression against a fully-forced concrete value.
    ///
    /// Used internally by the Gibbs sampler: at each step, the previous
    /// block's expression is compiled and pinned against its sampled value
    /// to form the conditioning BDD for the next step. Not a user-facing
    /// primitive.
    ///
    /// Returns `True`-valued `GuardedWorlds` (or an empty world list). The
    /// driver uses the True-guarded BDD as the additional conditioning
    /// constraint for the next sample step. A pin is a conjunctive constraint
    /// and only its True region is consumed, so every arm of [`Self::pin_value`]
    /// yields "True-or-nothing" — see [`Self::pin_holds_when`].
    pub fn compile_pin(
        &mut self,
        expr: &PExpr,
        value: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        debug_assert!(
            matches!(mode, CompileMode::KC { .. }),
            "compile_pin is conditioning machinery and must run in KC mode \
             (a Sample-mode bind would collapse the pinned expression's \
             worlds to one)"
        );
        let pc = mode.path_condition();
        let expr_worlds = self.traced_compile(expr, env, mode, 0);
        // Deliberate KC reset (NOT the continuation's mode): Pin is
        // conditioning machinery and needs *all* worlds of its value
        // expression. It appears in Gibbs conditioning (always compiled in
        // KC mode), never in sampled queries.
        let result = bind_compile(expr_worlds, pc, self, mode, |sym_val, pc, ctx, _mode| {
            let mut inner_mode = CompileMode::KC {
                path_condition: pc.clone(),
            };
            let val_worlds = ctx.traced_compile(value, env, &mut inner_mode, 1);
            bind_compile(
                val_worlds,
                pc,
                ctx,
                &mut inner_mode,
                |concrete_val, pc, ctx, _mode| ctx.pin_value(&sym_val, &concrete_val, pc),
            )
        });
        // Enforce the "True-or-nothing" contract at the pin boundary:
        // pin_value only ever returns True worlds, any 'false' worlds
        // (i.e. those where the pin constraint is not satisfied) should
        // just return an empty worlds list through program_error_worlds
        let true_sym = self.true_sym;
        debug_assert!(
            result.0.iter().all(|(v, _)| matches!(
                v.as_ref(),
                PluckValue::Value { constructor, args }
                    if *constructor == true_sym && args.is_empty()
            )),
            "compile_pin returned a non-True world; pin_value must yield \
             True-valued worlds or none"
        );
        result
    }

    /// Inner recursion for `compile_pin`. Both `sym` and `concrete` are
    /// already top-level forced (no surface thunks); nested args may still
    /// be thunked and are forced via `evaluate_thunk` as we descend.
    pub fn pin_value(
        &mut self,
        sym: &PluckVal,
        concrete: &PluckVal,
        pc: BooleanFunction,
    ) -> GuardedWorlds {
        match (sym.as_ref(), concrete.as_ref()) {
            // --- ADT × ADT: structural recursion ---
            (
                PluckValue::Value {
                    constructor: cs,
                    args: args_s,
                },
                PluckValue::Value {
                    constructor: cc,
                    args: args_c,
                },
            ) => {
                if cs != cc || args_s.len() != args_c.len() {
                    return program_error_worlds();
                }
                // Special case for Dirichlet: the concrete shape after
                // `force_symbolic` of a DirichletVector is
                // `Value { true_sym, [Native(Float)...] }`. The symbolic
                // side is `DirichletVector`, not `Value`, so this arm
                // doesn't apply — see the DirichletVector arm below.
                let pairs: Vec<(PluckVal, PluckVal)> =
                    args_s.iter().cloned().zip(args_c.iter().cloned()).collect();
                self.pin_args_and(pairs, pc)
            }

            // --- IntDist (symbolic) × IntDist (literal bits after forcing) ---
            (PluckValue::IntDist { bits: bs }, PluckValue::IntDist { bits: bc }) => {
                if bs.len() != bc.len() {
                    return program_error_worlds();
                }
                let builder = self.state.fac();
                let mut eq_bf = BooleanFunction::true_ptr();
                for (b_s, b_c) in bs.iter().zip(bc.iter()) {
                    let bit_iff = builder.iff(b_s, b_c);
                    eq_bf = builder.and(&eq_bf, &bit_iff);
                }
                // The pin holds exactly in the `eq_bf` region (the bits agree
                // with the concrete value); guard the True world by it.
                self.pin_holds_when(eq_bf)
            }

            // --- Native × Native: exact equality ---
            (PluckValue::Native(a), PluckValue::Native(b)) => {
                if a == b {
                    pure_monad(self.bool_val(true), pc)
                } else {
                    program_error_worlds()
                }
            }

            // --- Probability (Beta) × Native(Float): emit ProbPin observation ---
            (PluckValue::Probability(name), PluckValue::Native(NativeVal::Float(r))) => {
                let pin_var = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(FlipWeight::Beta(BetaFlip::ProbPin(*name, *r)))
                });
                self.pin_holds_when(pin_var)
            }

            // --- Gamma rate × Native(Float): emit RatePin observation ---
            (PluckValue::Gamma { name, scale }, PluckValue::Native(NativeVal::Float(r))) => {
                if *r <= 0.0 {
                    // Gamma support is (0, ∞): the pin is unsatisfiable
                    // (`r > 0 ⇔ r/scale > 0` for `scale > 0`).
                    return program_error_worlds();
                }
                let pin_var = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(FlipWeight::Gamma(GammaFlip::RatePin {
                        name: *name,
                        value: *r,
                        scale: *scale,
                    }))
                });
                self.pin_holds_when(pin_var)
            }

            // --- Exponential draw × Native(Float): emit ExpEq observation ---
            (
                PluckValue::Exponential { name, gamma, scale },
                PluckValue::Native(NativeVal::Float(r)),
            ) => {
                if *r < 0.0 {
                    // Exp support is [0, ∞).
                    return program_error_worlds();
                }
                let pin_var = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(FlipWeight::Gamma(GammaFlip::ExpEq {
                        gamma: *gamma,
                        draw: *name,
                        value: *r,
                        scale: *scale,
                    }))
                });
                self.pin_holds_when(pin_var)
            }

            // --- Poisson draw × Native(Int): emit PoissonEvent observation ---
            // `force_symbolic` realizes Poisson draws as native ints, so
            // this is the shape Gibbs pins arrive in.
            (PluckValue::Poisson { name, gamma, scale }, PluckValue::Native(NativeVal::Int(k))) => {
                if *k < 0 {
                    // Poisson support is the non-negative integers.
                    return program_error_worlds();
                }
                let pin_var = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(FlipWeight::Gamma(GammaFlip::PoissonEvent {
                        gamma: *gamma,
                        draw: *name,
                        event: IntervalOrEq::eq(*k as u64),
                        scale: *scale,
                    }))
                });
                self.pin_holds_when(pin_var)
            }

            // --- GaussianExpr × Native(Float): emit Gaussian observation ---
            (
                PluckValue::GaussianExpr {
                    constant,
                    coefficients,
                },
                PluckValue::Native(NativeVal::Float(r)),
            ) => {
                if coefficients.is_empty() {
                    // Degenerate: no Gaussian variables, just compare floats.
                    if (*constant - *r).abs() < 1e-15 {
                        pure_monad(self.bool_val(true), pc)
                    } else {
                        program_error_worlds()
                    }
                } else {
                    let coeffs_f64 = coefficients
                        .iter()
                        .map(|(k, v)| (*k, v.into_inner()))
                        .collect();
                    let weight = FlipWeight::Gaussian(GaussianFlip::Obs {
                        coefficients: coeffs_f64,
                        constant: *constant,
                        observed: *r,
                    });
                    let obs_var = self
                        .state
                        .with_callstack_index(2, |s| s.current_address_weighted(weight));
                    self.pin_holds_when(obs_var)
                }
            }

            // --- DirichletVector × float-vector-shaped concrete ---
            (
                PluckValue::DirichletVector { name, categories },
                PluckValue::Value { .. } | PluckValue::FloatMatrix { .. },
            ) => {
                let observed = extract_dirichlet_value(concrete);
                if observed.len() != *categories {
                    return program_error_worlds();
                }
                let weight = FlipWeight::Dirichlet(DirichletFlip::VecPin {
                    name: *name,
                    value: observed,
                });
                let obs_var = self
                    .state
                    .with_callstack_index(2, |s| s.current_address_weighted(weight));
                self.pin_holds_when(obs_var)
            }

            // --- DirichletProbability × Float: unsupported ---
            (PluckValue::DirichletProbability { .. }, PluckValue::Native(NativeVal::Float(_))) => {
                panic!(
                    "compile_pin: cannot pin a single Dirichlet component \
                     (e.g. (vector_index weights i)); block on the parent \
                     vector instead so correlations between components are \
                     preserved."
                );
            }

            // --- FloatMatrix × FloatMatrix: element-wise recursion ---
            (
                PluckValue::FloatMatrix {
                    entries: es,
                    shape: ss,
                },
                PluckValue::FloatMatrix {
                    entries: ec,
                    shape: sc,
                },
            ) => {
                if ss != sc || es.len() != ec.len() {
                    return program_error_worlds();
                }
                let pairs: Vec<(PluckVal, PluckVal)> =
                    es.iter().cloned().zip(ec.iter().cloned()).collect();
                self.pin_args_and(pairs, pc)
            }

            // --- Any other shape mismatch: not equal ---
            _ => program_error_worlds(),
        }
    }

    /// Build the worlds for a pin element whose equality reduces to a single
    /// BDD `g` (the region in which the pin holds): one `True` world guarded
    /// by `g`, or no world when `g` is unsatisfiable.
    ///
    /// Unlike `if_then_else_monad`, this emits **no** `False` world. A pin is
    /// a conjunctive constraint and only its True region is ever consumed (by
    /// `or_true_worlds` / the enclosing `case … of True`). This avoids doing
    /// unnecessary work by building the False region BDD
    fn pin_holds_when(&self, g: BooleanFunction) -> GuardedWorlds {
        if g.is_false() {
            program_error_worlds()
        } else {
            (vec![(self.bool_val(true), g)], BooleanFunction::true_ptr())
        }
    }

    /// AND of per-argument pin results.
    ///
    /// Every arm of `pin_value` yields "True-or-nothing" (see
    /// [`Self::pin_holds_when`]), so the conjunction is just: force each
    /// element and bind. `bind_monad` ANDs each element's guard into the
    /// running path condition, so the surviving True world's guard is the
    /// conjunction of the per-element equality regions. An element that
    /// cannot hold contributes no world and prunes that branch — no
    /// short-circuit on a running `False` value is needed (none is produced).
    /// Each pair's symbolic side is `evaluate_thunk`'d before recursing so
    /// nested thunks are forced under the current path condition.
    fn pin_args_and(
        &mut self,
        pairs: Vec<(PluckVal, PluckVal)>,
        pc: BooleanFunction,
    ) -> GuardedWorlds {
        let mut acc = pure_monad(self.bool_val(true), pc.clone());
        for (sym_arg, concrete_arg) in pairs {
            // Deliberate KC modes throughout (pin is KC-only conditioning
            // machinery; see compile_pin).
            let mut outer_mode = CompileMode::KC {
                path_condition: pc.clone(),
            };
            acc = bind_compile(
                acc,
                pc.clone(),
                self,
                &mut outer_mode,
                move |_, pc, ctx, _mode| {
                    // Force the symbolic arg (may be a Thunk / ThunkUnion).
                    let mut inner_mode = CompileMode::KC {
                        path_condition: pc.clone(),
                    };
                    let forced = ctx.evaluate_thunk(&sym_arg, &mut inner_mode);
                    let concrete_inner = concrete_arg.clone();
                    bind_compile(
                        forced,
                        pc,
                        ctx,
                        &mut inner_mode,
                        move |forced_sym, pc, ctx, _mode| {
                            ctx.pin_value(&forced_sym, &concrete_inner, pc)
                        },
                    )
                },
            );
        }
        acc
    }

    /// `dexpr` must evaluate to a `DirichletVector`; `val` must evaluate
    /// to a 1-D `FloatMatrix` of `Native(Float)` entries with length
    /// equal to the Dirichlet's category count. Emits a
    /// `DirichletFlip::VecPin` measure-zero observation that the
    /// disintegration machinery resolves via `DirichletSuffStat::RealEq`.
    fn compile_dirichlet_eq(
        &mut self,
        dexpr: &PExpr,
        val: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        if matches!(mode, CompileMode::Sample(_)) {
            return self.sample_compile_dirichlet_eq(dexpr, val, env, mode);
        }

        let pc = mode.path_condition();
        let expr_worlds = self.traced_compile(dexpr, env, mode, 0);

        bind_compile(expr_worlds, pc, self, mode, |expr_val, pc, ctx, mode| {
            let val_worlds = ctx.traced_compile(val, env, mode, 1);
            bind_compile(val_worlds, pc, ctx, mode, |val_v, pc, ctx, _mode| {
                let observed = extract_dirichlet_value(val_v.as_ref());
                match expr_val.as_ref() {
                    PluckValue::DirichletVector { name, categories } => {
                        assert_eq!(
                            observed.len(),
                            *categories,
                            "dirichlet_eq: value length {} does not match Dirichlet's {} categories",
                            observed.len(),
                            *categories
                        );
                        let weight = FlipWeight::Dirichlet(DirichletFlip::VecPin {
                            name: *name,
                            value: observed,
                        });
                        let obs_var = ctx
                            .state
                            .with_callstack_index(2, |s| s.current_address_weighted(weight));
                        let true_val = ctx.bool_val(true);
                        let false_val = ctx.bool_val(false);
                        if_then_else_monad(true_val, false_val, obs_var, pc, ctx)
                    }
                    PluckValue::FloatMatrix { .. } => {
                        // Both sides already concrete vectors: structural equality.
                        let lhs = extract_dirichlet_value(expr_val.as_ref());
                        let matches = lhs.len() == observed.len()
                            && lhs.iter().zip(observed.iter()).all(|(a, b)| a == b);
                        let result = ctx.bool_val(matches);
                        pure_monad(result, pc)
                    }
                    _ => panic!(
                        "dirichlet_eq: expr must be a DirichletVector or 1-D FloatMatrix, got {:?}",
                        expr_val
                    ),
                }
            })
        })
    }

    fn sample_compile_dirichlet_eq(
        &mut self,
        dexpr: &PExpr,
        val: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let expr_val = self.sample_compile_one(dexpr, env, mode, 0, "DirichletEq: expr");
        let val_v = self.sample_compile_one(val, env, mode, 1, "DirichletEq: val");

        let observed = extract_dirichlet_value(val_v.as_ref());
        match expr_val.as_ref() {
            PluckValue::DirichletVector { name, categories } => {
                assert_eq!(
                    observed.len(),
                    *categories,
                    "dirichlet_eq: value length {} does not match Dirichlet's {} categories",
                    observed.len(),
                    *categories
                );
                let weight = FlipWeight::Dirichlet(DirichletFlip::VecPin {
                    name: *name,
                    value: observed,
                });
                let CompileMode::Sample(sample_state) = mode else {
                    unreachable!()
                };
                let outcome = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(weight);
                    crate::inference::sampling::resolve_observation(sample_state, s)
                });
                pure_monad(self.bool_val(outcome), BooleanFunction::true_ptr())
            }
            PluckValue::FloatMatrix { .. } => {
                let lhs = extract_dirichlet_value(expr_val.as_ref());
                let matches = lhs.len() == observed.len()
                    && lhs.iter().zip(observed.iter()).all(|(a, b)| a == b);
                pure_monad(self.bool_val(matches), BooleanFunction::true_ptr())
            }
            _ => panic!(
                "dirichlet_eq: expr must be a DirichletVector or 1-D FloatMatrix, got {:?}",
                expr_val
            ),
        }
    }

    fn compile_dirichlet(
        &mut self,
        alphas: &[PExpr],
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        self.compile_dirichlet_step(alphas, env, mode, 0, vec![])
    }

    fn compile_dirichlet_step(
        &mut self,
        remaining: &[PExpr],
        env: &Env,
        mode: &mut CompileMode<'_>,
        idx: i32,
        acc: Vec<f64>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        if remaining.is_empty() {
            let name = self.state.current_continuous_name(&acc);
            self.state.register_prior(DirichletPrior::Dirichlet {
                name,
                alphas: acc.clone(),
            });
            // Mid-sample first registration: cover the variable in this
            // sample's assignment (see SampleState::ensure_continuous).
            if let CompileMode::Sample(state) = mode {
                state.ensure_continuous([name], &self.state.prior_registry);
            }
            let dirichlet = mk_dirichlet(name, acc.len());
            return pure_monad(dirichlet, pc);
        }

        let expr = &remaining[0];
        let worlds = self.traced_compile(expr, env, mode, idx);

        bind_compile(worlds, pc, self, mode, move |val, _new_pc, ctx, mode| {
            let f = as_float_or_panic(
                val.as_ref(),
                &format!("dirichlet: argument {} must be a number", idx),
            );
            let mut new_acc = acc.clone();
            new_acc.push(f);
            ctx.compile_dirichlet_step(&remaining[1..], env, mode, idx + 1, new_acc)
        })
    }

    fn compile_index(
        &mut self,
        v: &PExpr,
        idx: &[PExpr],
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        assert!(
            !idx.is_empty(),
            "vector_index requires at least one index argument"
        );
        let pc = mode.path_condition();
        let v_worlds = self.traced_compile(v, env, mode, 0);
        let idx_owned: Vec<PExpr> = idx.to_vec();
        bind_compile(v_worlds, pc, self, mode, move |v_val, _pc, ctx, mode| {
            ctx.compile_index_resolve(v_val, &idx_owned, env, mode, Vec::new())
        })
    }

    /// Resolve the remaining index expressions one at a time via
    /// `bind_compile`, accumulating concrete i64s. When `remaining` is empty,
    /// dispatch to the per-variant helper. Each multi-world index expression
    /// spawns its own continuation; `join_monad` merges the resulting worlds.
    fn compile_index_resolve(
        &mut self,
        v_val: PluckVal,
        remaining: &[PExpr],
        env: &Env,
        mode: &mut CompileMode<'_>,
        acc: Vec<i64>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        if remaining.is_empty() {
            return match v_val.as_ref() {
                PluckValue::DirichletVector { name, categories } => {
                    self.compile_index_dirichlet(*name, *categories, &acc, pc)
                }
                PluckValue::FloatMatrix { entries, shape } => {
                    self.compile_index_float_matrix(entries, shape, &acc, pc)
                }
                _ => panic!(
                    "vector_index: first argument must be a DirichletVector or FloatMatrix, got {:?}",
                    v_val
                ),
            };
        }
        let worlds = self.traced_compile(&remaining[0], env, mode, 0);
        let rest: Vec<PExpr> = remaining[1..].to_vec();
        bind_compile(
            worlds,
            pc,
            self,
            mode,
            move |idx_val, _new_pc, ctx, mode| {
                let i = as_int_or_panic(idx_val.as_ref(), "vector_index: index");
                let mut new_acc = acc.clone();
                new_acc.push(i);
                ctx.compile_index_resolve(v_val.clone(), &rest, env, mode, new_acc)
            },
        )
    }

    fn compile_index_dirichlet(
        &self,
        name: u64,
        categories: usize,
        idx_vals: &[i64],
        pc: BooleanFunction,
    ) -> GuardedWorlds {
        assert_eq!(
            idx_vals.len(),
            1,
            "vector_index on DirichletVector takes exactly one index, got {}",
            idx_vals.len()
        );
        let i = idx_vals[0];
        assert!(
            i >= 0 && (i as usize) < categories,
            "vector_index: index {} out of bounds for DirichletVector of {} categories",
            i,
            categories
        );
        pure_monad(mk_dirichlet_probability(name, i as usize, categories), pc)
    }

    /// 1D + 1 idx → scalar entry; 2D + 1 idx → row as 1D FloatMatrix;
    /// 2D + 2 idx → scalar entry. Other shape/arity combos panic.
    fn compile_index_float_matrix(
        &self,
        entries: &[PluckVal],
        shape: &[usize],
        idx_vals: &[i64],
        pc: BooleanFunction,
    ) -> GuardedWorlds {
        let bounds_check = |dim_idx: usize, val: i64| {
            let dim = shape[dim_idx];
            assert!(
                val >= 0 && (val as usize) < dim,
                "vector_index: index {} out of bounds for axis {} (size {})",
                val,
                dim_idx,
                dim
            );
        };

        let result = match (shape.len(), idx_vals.len()) {
            (1, 1) => {
                bounds_check(0, idx_vals[0]);
                entries[idx_vals[0] as usize].clone()
            }
            (2, 1) => {
                bounds_check(0, idx_vals[0]);
                let cols = shape[1];
                let row_start = idx_vals[0] as usize * cols;
                let row: Vec<PluckVal> = entries[row_start..row_start + cols].to_vec();
                mk_float_matrix(row, vec![cols])
            }
            (2, 2) => {
                bounds_check(0, idx_vals[0]);
                bounds_check(1, idx_vals[1]);
                let cols = shape[1];
                entries[idx_vals[0] as usize * cols + idx_vals[1] as usize].clone()
            }
            (r, k) => panic!(
                "vector_index: unsupported (rank, n_indices) = ({}, {}) for FloatMatrix of shape {:?}",
                r, k, shape
            ),
        };
        pure_monad(result, pc)
    }

    fn compile_categorical(
        &mut self,
        probs: &[PExpr],
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        if matches!(mode, CompileMode::Sample(_)) {
            return self.sample_compile_categorical(probs, env, mode);
        }
        let pc = mode.path_condition();
        let total = probs.len();
        self.compile_categorical_step(probs, env, pc, 0, total, vec![], None)
    }

    /// Sample-mode counterpart of `compile_categorical`. Walks the N-1
    /// stick-breaking BDD variables one at a time, calls `resolve_flip`
    /// per stick (consulting `state.constraint` then sampling), and
    /// returns the first hit as a `Native(Int)` outcome — exactly one
    /// world, satisfying `force_value`'s invariant.
    ///
    /// Each stick allocates its BDD variable at the same callstack-index
    /// plus FlipWeight key as KC mode's `allocate_one_hot_vars`, so any
    /// upstream pin or evidence BDD referring to these vars stays in
    /// agreement.
    fn sample_compile_categorical(
        &mut self,
        probs: &[PExpr],
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        // Resolve each prob arg to a single value (Sample mode).
        let mut float_acc: Vec<f64> = Vec::new();
        let mut vector: Option<(u64, usize)> = None;
        for (idx, prob) in probs.iter().enumerate() {
            let val = self.sample_compile_one(prob, env, mode, idx as i32, "Categorical: prob");
            match val.as_ref() {
                PluckValue::Native(NativeVal::Float(f)) => float_acc.push(*f),
                PluckValue::DirichletVector { name, categories } => {
                    if probs.len() != 1 {
                        panic!(
                            "Categorical: DirichletVector must be the single argument, got {} total args",
                            probs.len()
                        );
                    }
                    vector = Some((*name, *categories));
                }
                _ => panic!(
                    "Categorical: argument {} must be a float or a DirichletVector, got {:?}",
                    idx, val
                ),
            }
        }

        let kind = if let Some((name, categories)) = vector {
            if !float_acc.is_empty() {
                eprintln!("Categorical: cannot mix DirichletVector with constant probabilities");
                return program_error_worlds();
            }
            CatKind::Dirichlet { name, categories }
        } else {
            let sum_diff = (float_acc.iter().sum::<f64>() - 1.0).abs();
            if sum_diff > 1e-9 {
                eprintln!("Categorical: constant probabilities do not sum to one");
                return program_error_worlds();
            }
            CatKind::Constants(float_acc)
        };

        let n = match &kind {
            CatKind::Dirichlet { categories, .. } => *categories,
            CatKind::Constants(fs) => fs.len(),
        };
        if n == 0 {
            return program_error_worlds();
        }
        // Degenerate single-category: one outcome, no BDD vars allocated
        // (mirrors `finalize_categorical`'s `n == 1` fast path).
        if n == 1 {
            return pure_monad(mk_native_int(0), BooleanFunction::true_ptr());
        }

        // Stick-breaking conditional probability per stick.
        // - Constants: `β_i = p_i / (1 − Σ_{j<i} p_j)` (pre-computed).
        // - Dirichlet: `β_i = θ_i / r_i` where `r_i = Σ_{j≥i} θ_j`
        //   (suffix sums over the sampled θ).
        let betas: Vec<f64> = match &kind {
            CatKind::Constants(fs) => stick_betas(fs),
            CatKind::Dirichlet { name, .. } => {
                let CompileMode::Sample(sample_state) = mode else {
                    unreachable!()
                };
                let theta = sample_state.assignment.dirichlet_value(*name);
                let mut suffix = vec![0.0_f64; n + 1];
                for i in (0..n).rev() {
                    suffix[i] = suffix[i + 1] + theta[i].into_inner();
                }
                (0..n - 1)
                    .map(|i| {
                        let r_i = suffix[i];
                        if r_i <= 0.0 {
                            0.0
                        } else {
                            theta[i].into_inner() / r_i
                        }
                    })
                    .collect()
            }
        };

        // Walk sticks. The first one to hit determines the outcome; if
        // none hit, the outcome is the leftover category `n - 1`. The
        // index `i` drives both `betas[i]` (constants path), the per-
        // stick `with_callstack_index`, and `StickElement::index`, so a
        // range loop is clearer than `.iter().enumerate()`.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n - 1 {
            let weight = match &kind {
                CatKind::Dirichlet { name, .. } => {
                    FlipWeight::Dirichlet(DirichletFlip::StickElement {
                        name: *name,
                        index: i,
                    })
                }
                CatKind::Constants(_) => FlipWeight::Constant(betas[i]),
            };
            let discriminator = weight.discriminator();
            // Match `allocate_one_hot_vars`'s per-stick callstack index so
            // the (callstack_hash, discriminator) key matches KC mode.
            let key = self.state.with_callstack_index(i as i32, |s| {
                let k = (s.callstack.hash(), discriminator);
                s.current_address_weighted(weight);
                k
            });
            let CompileMode::Sample(sample_state) = mode else {
                unreachable!()
            };
            let hit =
                crate::inference::sampling::resolve_flip(betas[i], key, sample_state, &*self.state);
            if hit {
                return pure_monad(mk_native_int(i as i64), BooleanFunction::true_ptr());
            }
        }

        pure_monad(mk_native_int((n - 1) as i64), BooleanFunction::true_ptr())
    }

    #[allow(clippy::too_many_arguments)] // recursive accumulator
    fn compile_categorical_step(
        &mut self,
        remaining: &[PExpr],
        env: &Env,
        pc: BooleanFunction,
        idx: i32,
        total_args: usize,
        float_acc: Vec<f64>,
        vector: Option<(u64, usize)>,
    ) -> GuardedWorlds {
        if remaining.is_empty() {
            let kind = if let Some((name, categories)) = vector {
                if !float_acc.is_empty() {
                    eprintln!(
                        "Categorical: cannot mix DirichletVector with constant probabilities"
                    );
                    return program_error_worlds();
                }
                CatKind::Dirichlet { name, categories }
            } else {
                let sum_diff = (float_acc.iter().sum::<f64>() - 1.0).abs();
                if sum_diff > 1e-9 {
                    eprintln!("Categorical: constant probabilities do not sum to one");
                    return program_error_worlds();
                }
                CatKind::Constants(float_acc)
            };
            return self.finalize_categorical(pc, kind);
        }

        let expr = &remaining[0];
        // Deliberate KC reset: this helper is only reachable in KC mode
        // (`compile_categorical` intercepts Sample mode with
        // `sample_compile_categorical` before stepping).
        let mut inner_mode = CompileMode::KC {
            path_condition: pc.clone(),
        };
        let worlds = self.traced_compile(expr, env, &mut inner_mode, idx);

        bind_compile(
            worlds,
            pc,
            self,
            &mut inner_mode,
            move |val, new_pc, ctx, _mode| {
                let mut new_float_acc = float_acc.clone();
                let mut new_vector = vector;

                match val.as_ref() {
                    PluckValue::Native(NativeVal::Float(f)) => new_float_acc.push(*f),
                    PluckValue::DirichletVector { name, categories } => {
                        if total_args != 1 {
                            panic!(
                            "Categorical: DirichletVector must be the single argument, got {} total args",
                            total_args
                        );
                        }
                        new_vector = Some((*name, *categories));
                    }
                    _ => panic!(
                        "Categorical: argument {} must be a float or a DirichletVector, got {:?}",
                        idx, val
                    ),
                };

                ctx.compile_categorical_step(
                    &remaining[1..],
                    env,
                    new_pc,
                    idx + 1,
                    total_args,
                    new_float_acc,
                    new_vector,
                )
            },
        )
    }

    fn finalize_categorical(&mut self, pc: BooleanFunction, kind: CatKind) -> GuardedWorlds {
        let n = match &kind {
            CatKind::Dirichlet { categories, .. } => *categories,
            CatKind::Constants(fs) => fs.len(),
        };
        if n == 0 {
            return program_error_worlds();
        }

        let vals: Vec<PluckVal> = (0..n).map(|i| mk_native_int(i as i64)).collect();

        // Single-category degenerate case: one "world" with guard `true`.
        if n == 1 {
            return categorical_monad(vals, vec![BooleanFunction::true_ptr()], pc, self);
        }

        // Stick-breaking conditionals for constants (see `stick_betas`).
        let betas: Option<Vec<f64>> = match &kind {
            CatKind::Constants(fs) => Some(stick_betas(fs)),
            CatKind::Dirichlet { .. } => None,
        };

        let vars = allocate_one_hot_vars(
            n - 1,
            |i| match &kind {
                CatKind::Dirichlet { name, .. } => {
                    FlipWeight::Dirichlet(DirichletFlip::StickElement {
                        name: *name,
                        index: i,
                    })
                }
                CatKind::Constants(_) => FlipWeight::Constant(betas.as_ref().unwrap()[i]),
            },
            self.state,
        );

        let conditions = build_stick_breaking_guards(&vars, self.state);

        categorical_monad(vals, conditions, pc, self)
    }

    /// Compile a float binary op whose operands may be symbolic continuous
    /// values, not just plain floats: `GaussianExpr` (affine combinations of
    /// Gaussians) and scaled `Gamma` values both flow through here. A
    /// `Gamma` scaled/divided by a constant stays a (re-scaled) `Gamma` so a
    /// shared gamma keeps pooling; anything the symbolic paths don't handle
    /// falls through to the numeric/affine `apply_float_op`.
    fn compile_variable_aware_binop(
        &mut self,
        op: FloatOp,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);

        bind_compile(a_worlds, pc, self, mode, move |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, move |b_val, pc, ctx, _mode| {
                // Scaling a gamma by a constant is handled here, *before*
                // the Gaussian-affine path, so a scaled `Gamma` keeps
                // referencing its underlying variable (preserving pooling).
                let result = match try_scale_gamma(op, &a_val, &b_val) {
                    Some(scaled) => scaled,
                    None => apply_float_op(&op, &a_val, &b_val, ctx, pc.clone()),
                };
                pure_monad(result, pc)
            })
        })
    }

    fn compile_gamma(
        &mut self,
        shape: &PExpr,
        rate: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let shape_worlds = self.traced_compile(shape, env, mode, 0);

        bind_compile(shape_worlds, pc, self, mode, |shape_val, pc, ctx, mode| {
            let rate_worlds = ctx.traced_compile(rate, env, mode, 1);
            bind_compile(rate_worlds, pc, ctx, mode, |rate_val, pc, ctx, mode| {
                let shape_f = as_float_or_panic(
                    shape_val.as_ref(),
                    "Gamma: first argument must be a number (use 1.0 not 1)",
                );
                let rate_f = as_float_or_panic(
                    rate_val.as_ref(),
                    "Gamma: second argument must be a number (use 1.0 not 1)",
                );
                assert!(
                    shape_f > 0.0 && rate_f > 0.0,
                    "Gamma: shape and rate must be positive, got Gamma({shape_f}, {rate_f})"
                );

                let name = ctx.state.current_continuous_name(&[shape_f, rate_f]);
                ctx.state.register_prior(GammaPrior::Gamma {
                    name,
                    shape: shape_f,
                    rate: rate_f,
                });
                // First registration can happen mid-sample (forward
                // execution reaching a body the evidence never compiled);
                // cover the variable in this sample's assignment.
                if let CompileMode::Sample(state) = mode {
                    state.ensure_continuous([name], &ctx.state.prior_registry);
                }
                pure_monad(mk_gamma(name), pc)
            })
        })
    }

    /// Compile a PExpr that takes a `rate` parameter which can be either a float or a Gamma variable
    /// An *constant* rate parameter gets compiled to a Dirac prior on the rate, so that everything
    /// gets an associated gamma parameter
    fn compile_gamma_consumer(
        &mut self,
        rate: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
        family: GammaDrawFamily,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();

        let rate_worlds = self.traced_compile(rate, env, mode, 0);
        bind_compile(
            rate_worlds,
            pc,
            self,
            mode,
            move |rate_val, pc, ctx, mode| {
                // A `Gamma` rate carries a scale (`scale * g`); a constant
                // rate becomes a Dirac prior with scale 1.
                let (gamma_name, scale) = match rate_val.as_ref() {
                    PluckValue::Gamma { name, scale } => (*name, *scale),
                    value => {
                        let value_f = as_float_or_panic(
                            value,
                            "Exponential: rate parameter must be Gamma or a number (use 1.0 not 1)",
                        );
                        assert!(
                            value_f > 0.0,
                            "Poisson/Exponential: rate must be positive, got {value_f}"
                        );

                        let name = ctx.state.current_continuous_name(&[value_f]);
                        ctx.state.register_prior(GammaPrior::Pinned {
                            name,
                            value: value_f,
                        });
                        // First registration can happen mid-sample (forward
                        // execution reaching a body the evidence never compiled);
                        // cover the variable in this sample's assignment.
                        if let CompileMode::Sample(state) = mode {
                            state.ensure_continuous([name], &ctx.state.prior_registry);
                        }

                        (name, 1.0)
                    }
                };
                // Each rate-world gets its OWN draw variable: keying the draw
                // name on the rate's identity (`gamma_name`, `scale`) keeps a
                // world-dependent rate expression like `(poisson (if b 20.0 0.5))`
                // from aliasing one draw across two rate-worlds.
                // Per-rate-world draws ensure likelihood leaves are disjoint and
                // are also the case where the two-stage sampler's indicator
                // weights are exact. (`gamma_name` is distinct per rate-world —
                // a distinct Pinned name per constant rate, or the gamma's own
                // name — and `scale` separates same-rate/different-scale worlds.)
                let scale_nn = NotNan::new(scale).expect("gamma scale must be finite");
                let draw_name = ctx
                    .state
                    .current_continuous_name(&[gamma_name as f64, scale]);
                // Record draw → (family, rate) so weight evaluation can realize a
                // draw the posterior sample skipped (see
                // `LazyKCState::realize_missing_gamma_draws`).
                ctx.state
                    .gamma_draws
                    .insert(draw_name, (family, (gamma_name, scale_nn)));
                // Sample mode realizes the draw itself: a fresh draw from
                // the conditional given the assignment's rate, unless the
                // evidence already covered it (see `ensure_gamma_draw`).
                if let CompileMode::Sample(state) = mode {
                    state.ensure_gamma_draw(gamma_name, draw_name, family, scale);
                }
                pure_monad(family.mk_value(gamma_name, draw_name, scale), pc)
            },
        )
    }

    // ===== FloatMatrix helpers =====

    /// Compile a `FloatMatrix` literal. Literal float entries materialize as
    /// `Native(Float)` directly (enabling the 0/1 short-circuit downstream).
    /// All other entries are wrapped as thunks and forced on demand by the
    /// matrix ops that touch them.
    fn compile_float_matrix(
        &mut self,
        entries: &[Vec<PExpr>],
        shape: &[usize],
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let mut flat: Vec<PluckVal> = Vec::with_capacity(shape.iter().product());
        for row in entries {
            for cell in row {
                let v: PluckVal = match cell {
                    // Literal float — materialize directly; no thunk.
                    PExpr::ConstNative {
                        val: NativeVal::Float(f),
                    } => mk_native_float(*f),
                    PExpr::ConstNative {
                        val: NativeVal::Int(i),
                    } => mk_native_float(*i as f64),
                    // Distinct flat index per entry so structurally-identical
                    // inline draws (e.g. two `(normal 0 1)` in one literal) land
                    // at distinct callstack positions and stay independent RVs.
                    other => make_thunk(other, env.clone(), flat.len() as i32, self.state),
                };
                flat.push(v);
            }
        }
        pure_monad(mk_float_matrix(flat, shape.to_vec()), pc)
    }

    /// Compile `(sum m)` — sum every entry of a FloatMatrix. Returns a
    /// `Native(Float)` when no Gaussian terms survive, otherwise a
    /// `GaussianExpr`. Entries that peek as `Native(Float(0.0))` are skipped
    /// without forcing.
    fn compile_sum(
        &mut self,
        expr: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let m_worlds = self.traced_compile(expr, env, mode, 0);
        bind_compile(m_worlds, pc, self, mode, |m_val, pc, ctx, _mode| {
            let entries = match m_val.as_ref() {
                PluckValue::FloatMatrix { entries, .. } => entries.clone(),
                _ => panic!("sum: argument must be a FloatMatrix, got {:?}", m_val),
            };
            let mut acc_const = 0.0_f64;
            let mut acc_coeffs: std::collections::BTreeMap<u64, ordered_float::NotNan<f64>> =
                std::collections::BTreeMap::new();
            for cell in &entries {
                if let Some(f) = entry_constant(cell.as_ref()) {
                    if f == 0.0 {
                        continue;
                    }
                    acc_const += f;
                    continue;
                }
                let (c, m) = force_entry_to_affine(ctx, cell, pc.clone());
                acc_const += c;
                acc_coeffs = merge_coefficients(&acc_coeffs, &m, |a, b| a + b);
            }
            let out = if acc_coeffs.is_empty() {
                mk_native_float(acc_const)
            } else {
                mk_gaussian_expr(acc_const, acc_coeffs)
            };
            pure_monad(out, pc)
        })
    }

    // Compile `(@ A B)` — matrix product. Supports:
    //   * 2D × 2D, 2D × 1D, 1D × 2D, 1D × 1D
    //   * any mix of pure-float and symbolic-Gaussian entries, subject to the
    //     linearity guard (a single output cell cannot mix two non-trivial
    //     Gaussian terms via multiplication)
    // Per-term short-circuit: if `A[i,k]` peeks as 0, `B[k,j]` is not forced.
    fn compile_matmul(
        &mut self,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);
        bind_compile(a_worlds, pc, self, mode, |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, |b_val, pc, ctx, _mode| {
                let (a_entries, a_shape) = expect_float_matrix(&a_val, "@: left operand");
                let (b_entries, b_shape) = expect_float_matrix(&b_val, "@: right operand");
                let (a_rows, a_cols, a_was_1d) = lift_for_matmul(a_shape, true);
                let (b_rows, b_cols, b_was_1d) = lift_for_matmul(b_shape, false);
                assert_eq!(
                    a_cols, b_rows,
                    "@: inner dimensions disagree: lhs cols = {}, rhs rows = {}",
                    a_cols, b_rows
                );

                let mut out_entries: Vec<PluckVal> = Vec::with_capacity(a_rows * b_cols);
                for i in 0..a_rows {
                    for j in 0..b_cols {
                        let mut acc_const = 0.0_f64;
                        let mut acc_coeffs: std::collections::BTreeMap<
                            u64,
                            ordered_float::NotNan<f64>,
                        > = std::collections::BTreeMap::new();
                        for k in 0..a_cols {
                            let (c, m) = cell_op_affine(
                                &FloatOp::Mul,
                                &a_entries[i * a_cols + k],
                                &b_entries[k * b_cols + j],
                                ctx,
                                pc.clone(),
                            );
                            acc_const += c;
                            if !m.is_empty() {
                                acc_coeffs = merge_coefficients(&acc_coeffs, &m, |a, b| a + b);
                            }
                        }
                        out_entries.push(pack_affine(acc_const, acc_coeffs));
                    }
                }

                let result = squeeze_matmul_result(out_entries, a_rows, b_cols, a_was_1d, b_was_1d);
                pure_monad(result, pc)
            })
        })
    }

    // ===== Sample-mode helpers =====

    /// Compile `expr` once in Sample mode and unwrap its single world's
    /// value.
    ///
    /// Exactly one world is expected: every Sample-mode primitive either
    /// has a single-world sample handler or resolves multi-world KC
    /// artifacts before they can reach here (`bind_compile`'s Sample
    /// short-circuit, ThunkUnion selection in `evaluate_thunk_union`).
    /// Zero worlds means the sampled path hit a program error (error-mass
    /// policy: panic), or an inference limit if `hit_limit` is set.
    fn sample_compile_one(
        &mut self,
        expr: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
        strict_order_index: i32,
        label: &'static str,
    ) -> PluckVal {
        let (worlds, _) = self.traced_compile(expr, env, mode, strict_order_index);
        if worlds.is_empty() {
            if self.state.hit_limit {
                panic!(
                    "Sample-mode {}: inference limit (max_depth / time_limit) hit",
                    label
                );
            }
            panic!(
                "Sample-mode {}: produced no worlds \
                 (program error along the sampled path)",
                label
            );
        }
        assert_eq!(
            worlds.len(),
            1,
            "Sample-mode {}: expected exactly one world, got {}",
            label,
            worlds.len()
        );
        worlds.into_iter().next().unwrap().0
    }

    fn sample_compile_flip(
        &mut self,
        prob: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let prob_val = self.sample_compile_one(prob, env, mode, 0, "Flip: prob");
        let flip_weight = flip_weight_from(prob_val.as_ref());

        if let FlipWeight::Constant(p) = &flip_weight {
            if let Some(v) = self.constant_flip_shortcut(*p) {
                return pure_monad(v, BooleanFunction::true_ptr());
            }
        }

        let CompileMode::Sample(sample_state) = mode else {
            unreachable!();
        };
        let p = flip_weight
            .sample_probability(&sample_state.assignment)
            .expect("sample_compile_flip: flip weight has no scalar probability");
        let discriminator = flip_weight.discriminator();
        let key: super::state::CallstackKey = self.state.with_callstack_index(1, |s| {
            let k = (s.callstack.hash(), discriminator);
            s.current_address_weighted(flip_weight);
            k
        });
        let outcome = crate::inference::sampling::resolve_flip(p, key, sample_state, &*self.state);
        let val = self.bool_val(outcome);

        pure_monad(val, BooleanFunction::true_ptr())
    }

    fn sample_compile_prob_eq(
        &mut self,
        prob: &PExpr,
        val: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let prob_val = self.sample_compile_one(prob, env, mode, 0, "ProbEq: prob");
        let val_val = self.sample_compile_one(val, env, mode, 1, "ProbEq: val");

        let r = as_float_or_panic(
            val_val.as_ref(),
            "prob_eq: second argument must be a number",
        );
        match prob_val.as_ref() {
            PluckValue::Probability(name) => {
                let CompileMode::Sample(sample_state) = mode else {
                    unreachable!()
                };
                let outcome = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(FlipWeight::Beta(BetaFlip::ProbPin(*name, r)));
                    crate::inference::sampling::resolve_observation(sample_state, s)
                });
                pure_monad(self.bool_val(outcome), BooleanFunction::true_ptr())
            }
            PluckValue::Native(NativeVal::Float(c)) => {
                let result = self.bool_val((*c - r).abs() < 1e-15);
                pure_monad(result, BooleanFunction::true_ptr())
            }
            _ => panic!("prob_eq: first argument must be a Probability or float"),
        }
    }

    fn sample_compile_real_eq(
        &mut self,
        expr: &PExpr,
        val: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let expr_val = self.sample_compile_one(expr, env, mode, 0, "RealEq: expr");
        let val_v = self.sample_compile_one(val, env, mode, 1, "RealEq: val");

        let observed = as_float_or_panic(val_v.as_ref(), "real_eq: val must be a number");
        match expr_val.as_ref() {
            PluckValue::GaussianExpr {
                constant,
                coefficients,
            } => {
                if coefficients.is_empty() {
                    let result = self.bool_val((*constant - observed).abs() < 1e-15);
                    return pure_monad(result, BooleanFunction::true_ptr());
                }
                let coeffs_f64 = coefficients
                    .iter()
                    .map(|(k, v)| (*k, v.into_inner()))
                    .collect();
                let weight = FlipWeight::Gaussian(GaussianFlip::Obs {
                    coefficients: coeffs_f64,
                    constant: *constant,
                    observed,
                });
                let CompileMode::Sample(sample_state) = mode else {
                    unreachable!()
                };
                let outcome = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(weight);
                    crate::inference::sampling::resolve_observation(sample_state, s)
                });
                pure_monad(self.bool_val(outcome), BooleanFunction::true_ptr())
            }
            PluckValue::Native(NativeVal::Float(f)) => {
                let result = self.bool_val((*f - observed).abs() < 1e-15);
                pure_monad(result, BooleanFunction::true_ptr())
            }
            PluckValue::Gamma { name, scale } => {
                // `(real_eq (*. g s) v)` pins `g = v/s`; unsat check on `v`.
                if observed <= 0.0 {
                    // Gamma support is (0, ∞): the pin is unsatisfiable.
                    return pure_monad(self.bool_val(false), BooleanFunction::true_ptr());
                }
                let weight = FlipWeight::Gamma(GammaFlip::RatePin {
                    name: *name,
                    value: observed,
                    scale: *scale,
                });
                let CompileMode::Sample(sample_state) = mode else {
                    unreachable!()
                };
                let outcome = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(weight);
                    crate::inference::sampling::resolve_observation(sample_state, s)
                });
                pure_monad(self.bool_val(outcome), BooleanFunction::true_ptr())
            }
            PluckValue::Exponential { name, gamma, scale } => {
                if observed < 0.0 {
                    // Exp support is [0, ∞).
                    return pure_monad(self.bool_val(false), BooleanFunction::true_ptr());
                }
                let weight = FlipWeight::Gamma(GammaFlip::ExpEq {
                    gamma: *gamma,
                    draw: *name,
                    value: observed,
                    scale: *scale,
                });
                let CompileMode::Sample(sample_state) = mode else {
                    unreachable!()
                };
                let outcome = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(weight);
                    crate::inference::sampling::resolve_observation(sample_state, s)
                });
                pure_monad(self.bool_val(outcome), BooleanFunction::true_ptr())
            }
            PluckValue::Poisson { .. } => {
                panic!("real_eq: Poisson draws are integer-valued; use native_eq")
            }
            _ => panic!(
                "real_eq: expr must be a GaussianExpr, Gamma, Exponential \
                 or float, got {:?}",
                expr_val
            ),
        }
    }

    // ===== Thunk evaluation =====

    /// Evaluate a thunk or thunk union, returning GuardedWorlds.
    ///
    /// For a simple thunk: evaluate the expression with the captured environment.
    /// Uses caching: checks if any cached result is still valid under the current path condition.
    ///
    /// For a thunk union: evaluate each component thunk under its guard.
    pub fn evaluate_thunk(&mut self, val: &PluckVal, mode: &mut CompileMode<'_>) -> GuardedWorlds {
        let path_condition = mode.path_condition();
        if path_condition.is_false() {
            return false_path_condition_worlds();
        }

        // Per-sample thunk memoization (see `SampleState::memo`).
        if let (CompileMode::Sample(state), PluckValue::Thunk(_)) = (&*mode, val.as_ref()) {
            if let Some(worlds) = state.check_cache(val) {
                return worlds;
            }
        }

        let worlds = match val.as_ref() {
            PluckValue::Thunk(thunk_data) => self.evaluate_single_thunk(thunk_data, mode),
            PluckValue::ThunkUnion(union_data) => self.evaluate_thunk_union(union_data, mode),
            _ => return pure_monad(val.clone(), path_condition),
        };

        if let CompileMode::Sample(state) = &mut *mode {
            state.set_cache(val, &worlds);
        }
        worlds
    }

    /// Evaluate a thunk (no cache) — the raw evaluation path.
    fn evaluate_single_thunk_no_cache(
        &mut self,
        thunk_data: &ThunkData,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let old_callstack = std::mem::replace(
            &mut self.state.callstack,
            super::state::Callstack::from_vec(thunk_data.callstack.clone()),
        );

        let result = self.traced_compile(
            &thunk_data.expr,
            &thunk_data.env,
            mode,
            thunk_data.strict_order_index,
        );

        self.state.callstack = old_callstack;
        result
    }

    /// Evaluate a single thunk with singleton cache widening.
    fn evaluate_single_thunk(
        &mut self,
        thunk_data: &ThunkData,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        // Sample mode must not share the KC thunk cache: its results are
        // single-world collapses valid only for one (constraint, assignment).
        // Within-sample consistency is guaranteed by SampleState
        // (trace/assignment/constraint), not this cache.
        if matches!(mode, CompileMode::Sample(_)) {
            return self.evaluate_single_thunk_no_cache(thunk_data, mode);
        }

        let path_condition = mode.path_condition();
        // Check cache for a hit
        {
            let cache = thunk_data.cache.borrow();
            if let Some((cached_worlds, cached_used_info)) = cache.first() {
                let builder = self.state.fac();
                let neg_pc = builder.negate(&path_condition);
                let implies = builder.or(&neg_pc, cached_used_info);
                if implies.is_true() {
                    self.state.stats.num_cache_hits += 1;
                    return (cached_worlds.clone(), cached_used_info.clone());
                }
            }
        }

        self.state.stats.num_cache_misses += 1;

        let is_empty = thunk_data.cache.borrow().is_empty();
        if is_empty {
            self.state.stats.num_thunk_first_eval += 1;
            let result = self.evaluate_single_thunk_no_cache(thunk_data, mode);
            thunk_data.cache.borrow_mut().push(result.clone());
            return result;
        }

        // Singleton cache widening: merge cached result with fresh evaluation
        // of the uncovered region.
        self.state.stats.num_thunk_widen += 1;
        let (cached_worlds, cache_guard) = {
            let cache = thunk_data.cache.borrow();
            cache[0].clone()
        };

        let builder = self.state.fac();
        let neg_cache_guard = builder.negate(&cache_guard);
        let inner_path_condition = builder.and(&path_condition, &neg_cache_guard);

        if inner_path_condition.is_false() {
            self.state.stats.num_cache_hits += 1;
            return (cached_worlds, cache_guard);
        }

        let mut inner_mode = CompileMode::KC {
            path_condition: inner_path_condition,
        };
        let (miss_worlds, miss_used) =
            self.evaluate_single_thunk_no_cache(thunk_data, &mut inner_mode);

        let nested_worlds = vec![
            ((cached_worlds, cache_guard.clone()), cache_guard),
            ((miss_worlds, miss_used), neg_cache_guard),
        ];
        let result = join_monad(nested_worlds, BooleanFunction::true_ptr(), self);

        thunk_data.cache.borrow_mut()[0] = result.clone();

        result
    }

    /// Evaluate a thunk union without caching: bind over the component thunks.
    fn evaluate_thunk_union_no_cache(
        &mut self,
        union_data: &ThunkUnionData,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let path_condition = mode.path_condition();
        let nested_worlds_list: Vec<(PluckVal, BooleanFunction)> = union_data.thunks.clone();
        let pre_worlds: GuardedWorlds = (nested_worlds_list, BooleanFunction::true_ptr());

        bind_monad(
            pre_worlds,
            path_condition,
            self,
            |thunk_val, _pc, ctx, mode| {
                // `mode` is the bind-constructed per-world KC mode (identical to
                // the `KC { path_condition: pc }` this cont used to hand-roll).
                ctx.evaluate_thunk(&thunk_val, mode)
            },
        )
    }

    /// Evaluate a thunk union by evaluating each component thunk.
    ///
    /// No singleton cache here. Individual component
    /// thunks (LazyKCThunk) have their own singleton caches, so work is still
    /// cached at the thunk level.
    ///
    /// In Sample mode, ThunkUnions are KC artifacts (evidence-side
    /// `join_monad` merges) that forward evaluation can still meet through
    /// shared env values: select the single component consistent with the
    /// per-sample constraint — guards over already-drawn variables read off
    /// the constraint, undecided guard variables are prior-drawn by
    /// `select_world`'s extension (the same rule as everywhere else) — and
    /// evaluate only that component.
    fn evaluate_thunk_union(
        &mut self,
        union_data: &ThunkUnionData,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        self.state.stats.num_tu_eval += 1;
        if matches!(mode, CompileMode::Sample(_)) {
            let chosen = {
                let CompileMode::Sample(state) = &mut *mode else {
                    unreachable!()
                };
                // (thunk, guard) pairs have the same shape as worlds.
                crate::inference::sampling::select_world(
                    union_data.thunks.clone(),
                    state,
                    &*self.state,
                )
            };
            return self.evaluate_thunk(&chosen, mode);
        }
        self.evaluate_thunk_union_no_cache(union_data, mode)
    }

    /// Iteratively forces all thunks and IntDists in the value tree until none remain.
    pub fn infer_full_distribution(&mut self, initial_results: Vec<World>) -> Vec<World> {
        let mut queue: Vec<World> = initial_results;
        let mut resolved: Vec<World> = Vec::new();

        while let Some((current_val, current_bf)) = queue.pop() {
            if current_bf.is_false() {
                continue;
            }

            if let Some(thunk_path) = find_first_thunk(&current_val) {
                let thunk = get_value_at_path(&current_val, &thunk_path);

                let mut inner_mode = CompileMode::KC {
                    path_condition: current_bf.clone(),
                };
                let (sub_results, _used_info) = self.evaluate_thunk(&thunk, &mut inner_mode);

                for (sub_val, sub_bf) in sub_results {
                    let builder = self.state.fac();
                    let new_bf = builder.and(&current_bf, &sub_bf);
                    if !new_bf.is_false() {
                        let new_val = replace_at_path(&current_val, &thunk_path, &sub_val);
                        queue.push((new_val, new_bf));
                    }
                }
                continue;
            }

            if let Some(int_dist_path) = find_first_non_const_int_dist(&current_val) {
                let int_dist_val = get_value_at_path(&current_val, &int_dist_path);
                if let PluckValue::IntDist { bits } = int_dist_val.as_ref() {
                    let int_dist = IntDist::new(bits.clone());
                    let expanded = enumerate_int_dist(&int_dist, &current_bf, self.state.fac());
                    for (int_value, world_bf) in expanded {
                        if !world_bf.is_false() {
                            let const_bits: Vec<BooleanFunction> = (0..bits.len())
                                .map(|i| {
                                    if (int_value >> i) & 1 == 1 {
                                        BooleanFunction::true_ptr()
                                    } else {
                                        BooleanFunction::false_ptr()
                                    }
                                })
                                .collect();
                            let const_int_dist = mk_int_dist(const_bits);
                            let new_val =
                                replace_at_path(&current_val, &int_dist_path, &const_int_dist);
                            queue.push((new_val, world_bf));
                        }
                    }
                    continue;
                }
            }

            resolved.push((current_val, current_bf));
        }

        resolved.reverse();
        resolved
    }
}

/// Recursive binary splitting to encode a uniform distribution over [lo_val..=hi_val].
fn encode_range(
    lo_val: i64,
    hi_val: i64,
    guard: BooleanFunction,
    bits: &mut Vec<BooleanFunction>,
    width: usize,
    state: &mut LazyKCState,
    depth_idx: i32,
) {
    if guard.is_false() {
        return;
    }
    if lo_val == hi_val {
        let builder = state.fac();
        for (i, bit) in bits.iter_mut().enumerate().take(width) {
            let bit_set = ((lo_val as u64) >> i) & 1 == 1;
            let bit_bf = if bit_set {
                BooleanFunction::true_ptr()
            } else {
                BooleanFunction::false_ptr()
            };
            let contribution = builder.and(&bit_bf, &guard);
            *bit = builder.or(bit, &contribution);
        }
        return;
    }
    let mid = lo_val + (hi_val - lo_val) / 2;
    let lower_size = (mid - lo_val + 1) as f64;
    let upper_size = (hi_val - mid) as f64;
    let p = lower_size / (lower_size + upper_size);

    let addr = state.with_callstack_index(depth_idx, |s| s.current_address(p));

    let builder = state.fac();
    let lo_guard = builder.and(&guard, &addr);
    let neg_addr = builder.negate(&addr);
    let hi_guard = builder.and(&guard, &neg_addr);

    encode_range(lo_val, mid, lo_guard, bits, width, state, depth_idx + 1);
    encode_range(mid + 1, hi_val, hi_guard, bits, width, state, depth_idx + 1);
}

/// Convert a constant probability vector into the stick-breaking
/// conditional probabilities used by categorical's one-hot BDD encoding:
/// `β_i = p_i / (1 − Σ_{j<i} p_j)`. Produces `fs.len() − 1` betas
/// (one per stick); the leftover category is implicit.
fn stick_betas(fs: &[f64]) -> Vec<f64> {
    let n = fs.len();
    let mut remaining = 1.0;
    let mut out = Vec::with_capacity(n.saturating_sub(1));
    for &f in fs.iter().take(n.saturating_sub(1)) {
        if remaining <= 0.0 {
            out.push(0.0);
        } else {
            out.push((f / remaining).clamp(0.0, 1.0));
        }
        remaining -= f;
    }
    out
}

/// Allocate `n` boolean BDD variables, one per index, with weights produced by `weight_fn`.
/// Each variable is placed under a distinct callstack sentinel (`i as i32`) so they
/// occupy separate positions in the sorted callstack ordering.
fn allocate_one_hot_vars<F>(
    n: usize,
    mut weight_fn: F,
    state: &mut LazyKCState,
) -> Vec<BooleanFunction>
where
    F: FnMut(usize) -> FlipWeight,
{
    (0..n)
        .map(|i| state.with_callstack_index(i as i32, |s| s.current_address_weighted(weight_fn(i))))
        .collect()
}

/// Build stick-breaking guards over the given N-1 boolean BDD variables,
/// producing N guards (one per category).
///
/// For k ∈ 0..N-1:  guard_k = (¬vars[0] ∧ … ∧ ¬vars[k-1]) ∧ vars[k]
/// For k = N-1:     guard_k =  ¬vars[0] ∧ … ∧ ¬vars[N-2]     (broke all sticks)
fn build_stick_breaking_guards(
    vars: &[BooleanFunction],
    state: &LazyKCState,
) -> Vec<BooleanFunction> {
    let builder = state.fac();
    let n_sticks = vars.len();
    let n_categories = n_sticks + 1;

    let mut guards = Vec::with_capacity(n_categories);
    let mut prefix = BooleanFunction::true_ptr();
    for var in vars.iter() {
        guards.push(builder.and(&prefix, var));
        prefix = builder.and(&prefix, &builder.negate(var));
    }
    guards.push(prefix);
    guards
}

/// Create a thunk (lazy value) for an expression.
///
/// If the expression is a Var that points to an existing Thunk in the env,
/// return it directly. Otherwise, wrap the expression in a new thunk.
fn make_thunk(expr: &PExpr, env: Env, strict_order_index: i32, state: &LazyKCState) -> PluckVal {
    if let PExpr::Var { name } = expr {
        if let Some(val) = env_lookup(&env, *name) {
            // Only return the env value directly if it's a Thunk.
            // For ThunkUnions and other values, fall through to create a wrapper thunk.
            // The wrapper thunk's cache prevents re-evaluating the inner value.
            if matches!(val.as_ref(), PluckValue::Thunk(_)) {
                return val;
            }
        }
    }
    mk_thunk(
        Rc::new(expr.clone()),
        env,
        strict_order_index,
        state.callstack.to_vec(),
    )
}

/// Force a FloatMatrix entry (possibly a thunk) into its canonical affine form
/// `(constant, coefficients)`. Pure-float entries return `(f, empty)`;
/// `GaussianExpr` entries return their `(constant, coefficients)` directly.
/// Thunks are evaluated via the standard pipeline; we assume single-world
/// evaluation (panic otherwise — the typical case is a leaf gaussian or a
/// simple arithmetic expression with no branching).
fn force_entry_to_affine(
    ctx: &mut CompilerCtx,
    entry: &PluckVal,
    pc: BooleanFunction,
) -> (
    f64,
    std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>,
) {
    let resolved: PluckVal = match entry.as_ref() {
        PluckValue::Thunk(_) | PluckValue::ThunkUnion(_) => {
            let mut mode = CompileMode::KC { path_condition: pc };
            let (worlds, _) = ctx.evaluate_thunk(entry, &mut mode);
            assert_eq!(
                worlds.len(),
                1,
                "matrix entry produced {} worlds when forced; multi-world matrix entries are not supported",
                worlds.len()
            );
            worlds.into_iter().next().unwrap().0
        }
        _ => entry.clone(),
    };
    match resolved.as_ref() {
        PluckValue::Native(NativeVal::Float(f)) => (*f, std::collections::BTreeMap::new()),
        PluckValue::Native(NativeVal::Int(i)) => (*i as f64, std::collections::BTreeMap::new()),
        PluckValue::GaussianExpr {
            constant,
            coefficients,
        } => (*constant, coefficients.clone()),
        _ => panic!(
            "matrix entry forced to non-numeric / non-Gaussian value: {:?}",
            resolved
        ),
    }
}

/// Apply a binary `FloatOp` to two cell values in their forced affine form,
/// applying the same algebra as `compile_variable_aware_binop` does for
/// scalars. Used by both elementwise matrix ops and the existing scalar path
/// (after refactor). The linearity guard panics when `*.` / `/.` would
/// multiply two non-trivial Gaussian terms.
fn affine_binop(
    op: &FloatOp,
    (ca, ma): &(
        f64,
        std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>,
    ),
    (cb, mb): &(
        f64,
        std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>,
    ),
) -> (
    f64,
    std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>,
) {
    match op {
        FloatOp::Add => (ca + cb, merge_coefficients(ma, mb, |x, y| x + y)),
        FloatOp::Sub => (
            ca - cb,
            merge_coefficients(ma, &scale_coefficients(mb, -1.0), |x, y| x + y),
        ),
        FloatOp::Mul => {
            if !ma.is_empty() && !mb.is_empty() {
                panic!("*.: cannot multiply two non-trivial Gaussian terms (would be nonlinear)");
            }
            if ma.is_empty() {
                (ca * cb, scale_coefficients(mb, *ca))
            } else {
                (ca * cb, scale_coefficients(ma, *cb))
            }
        }
        FloatOp::Div => {
            if !mb.is_empty() {
                panic!("/.: divisor must be a constant (cannot divide by a Gaussian term)");
            }
            assert!(*cb != 0.0, "/.: division by zero");
            (ca / cb, scale_coefficients(ma, 1.0 / cb))
        }
    }
}

/// Wrap a `(constant, coefficients)` pair as a PluckVal. Empty coefficient
/// map collapses to a `Native(Float)` so downstream peeks (for the 0/1 fast
/// path) still work.
fn pack_affine(c: f64, m: std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>) -> PluckVal {
    if m.is_empty() {
        mk_native_float(c)
    } else {
        mk_gaussian_expr(c, m)
    }
}

/// Apply a binary FloatOp to two top-level values. Handles all 4 combinations
/// of (scalar, FloatMatrix). Matrix×matrix requires shape equality; scalar
/// broadcasts against a matrix of any shape. Falls back to `cell_op` for the
/// scalar×scalar case (which is itself just an entry in the matrix grid).
/// Intercept gamma-scaling arithmetic before the Gaussian-affine path.
///
/// A `Gamma` value (`scale * g`) multiplied or divided by a positive numeric
/// constant stays a `Gamma` with an updated scale, so a shared gamma keeps
/// pooling its conjugate posterior across its differently-scaled uses (rather
/// than being rewritten into a fresh, unshared gamma). Returns:
/// - `Some(scaled)` for a valid `Gamma`-by-constant scaling,
/// - `None` when no gamma-family value is involved (the caller falls back to
///   the numeric / Gaussian-affine path),
///
/// and panics for unsupported gamma-family arithmetic: addition/subtraction of
/// a gamma, gamma-by-gamma, constant/gamma, scaling by a non-positive constant,
/// or *any* arithmetic on an `Exponential` / `Poisson` draw (a draw is a
/// sample, not a rate, so it can't be algebraically scaled).
fn try_scale_gamma(op: FloatOp, a: &PluckVal, b: &PluckVal) -> Option<PluckVal> {
    // A draw is a sample, not a rate — it can't be algebraically scaled.
    let is_draw = |v: &PluckVal| {
        matches!(
            v.as_ref(),
            PluckValue::Exponential { .. } | PluckValue::Poisson { .. }
        )
    };
    if is_draw(a) || is_draw(b) {
        panic!(
            "float arithmetic on an Exponential/Poisson draw is not supported; \
             scale the gamma rate instead, e.g. (exponential (*. g 5.0))"
        );
    }

    let gamma_parts = |v: &PluckVal| match v.as_ref() {
        PluckValue::Gamma { name, scale } => Some((*name, *scale)),
        _ => None,
    };
    let native_const = |v: &PluckVal| match v.as_ref() {
        PluckValue::Native(NativeVal::Float(f)) => Some(*f),
        _ => None,
    };

    // Only intervene when a gamma is actually involved.
    if gamma_parts(a).is_none() && gamma_parts(b).is_none() {
        return None;
    }

    match op {
        FloatOp::Mul => {
            // Gamma * const  or  const * Gamma.
            let (name, scale, c) = match (gamma_parts(a), native_const(b)) {
                (Some((n, s)), Some(c)) => (n, s, c),
                _ => match (native_const(a), gamma_parts(b)) {
                    (Some(c), Some((n, s))) => (n, s, c),
                    _ => panic!(
                        "a Gamma variable can only be scaled by a numeric constant \
                         (use a float literal, e.g. (*. g 5.0))"
                    ),
                },
            };
            let new_scale = scale * c;
            assert!(
                new_scale > 0.0,
                "scaling a Gamma must yield a positive scale, got {new_scale} \
                 (scale {scale} * {c})"
            );
            Some(mk_gamma_scaled(name, new_scale))
        }
        FloatOp::Div => match (gamma_parts(a), native_const(b)) {
            // Only Gamma / const is a gamma; const / Gamma is not.
            (Some((name, scale)), Some(c)) => {
                let new_scale = scale / c;
                assert!(
                    new_scale > 0.0,
                    "dividing a Gamma must yield a positive scale, got {new_scale} \
                     (scale {scale} / {c})"
                );
                Some(mk_gamma_scaled(name, new_scale))
            }
            _ => panic!(
                "a Gamma variable can only be divided by a numeric constant \
                 (constant / Gamma is not supported)"
            ),
        },
        FloatOp::Add | FloatOp::Sub => panic!(
            "a Gamma variable supports only scaling (`*.` / `/.`); \
             addition/subtraction of a Gamma is not supported"
        ),
    }
}

fn apply_float_op(
    op: &FloatOp,
    a: &PluckVal,
    b: &PluckVal,
    ctx: &mut CompilerCtx,
    pc: BooleanFunction,
) -> PluckVal {
    let a_is_mat = matches!(a.as_ref(), PluckValue::FloatMatrix { .. });
    let b_is_mat = matches!(b.as_ref(), PluckValue::FloatMatrix { .. });

    match (a_is_mat, b_is_mat) {
        (false, false) => cell_op(op, a, b, ctx, pc),
        (true, true) => {
            let (ae, ash) = match a.as_ref() {
                PluckValue::FloatMatrix { entries, shape } => (entries.clone(), shape.clone()),
                _ => unreachable!(),
            };
            let (be, bsh) = match b.as_ref() {
                PluckValue::FloatMatrix { entries, shape } => (entries.clone(), shape.clone()),
                _ => unreachable!(),
            };
            assert_eq!(
                ash, bsh,
                "elementwise float op: shape mismatch {:?} vs {:?}",
                ash, bsh
            );
            let out: Vec<PluckVal> = ae
                .iter()
                .zip(be.iter())
                .map(|(ca, cb)| cell_op(op, ca, cb, ctx, pc.clone()))
                .collect();
            mk_float_matrix(out, ash)
        }
        (true, false) => {
            let (ae, ash) = match a.as_ref() {
                PluckValue::FloatMatrix { entries, shape } => (entries.clone(), shape.clone()),
                _ => unreachable!(),
            };
            let out: Vec<PluckVal> = ae
                .iter()
                .map(|ca| cell_op(op, ca, b, ctx, pc.clone()))
                .collect();
            mk_float_matrix(out, ash)
        }
        (false, true) => {
            let (be, bsh) = match b.as_ref() {
                PluckValue::FloatMatrix { entries, shape } => (entries.clone(), shape.clone()),
                _ => unreachable!(),
            };
            let out: Vec<PluckVal> = be
                .iter()
                .map(|cb| cell_op(op, a, cb, ctx, pc.clone()))
                .collect();
            mk_float_matrix(out, bsh)
        }
    }
}

/// Apply a binary FloatOp to two PluckVal cells with the 0/1 short-circuit.
/// Used by elementwise matrix ops AND the scalar dispatch. Forces operands
/// only when the short-circuit does not apply.
/// Lazy wrapper around `affine_binop`. Peeks each operand for `Native(Float)`
/// 0/1 and applies the per-op short-circuit without forcing the other side
/// (saving thunk evaluation). Returns the result in **affine** form so the
/// matmul accumulator can fold it directly.
///
/// Note: this forces the propagated side in the 1-case (and the constant-only
/// side in mixed scalar × Gaussian cases) because the return type is the
/// affine pair, not a `PluckVal`. Callers wanting to preserve a thunk
/// unforced in a result matrix can't use this — they'd have to layer their
/// own propagation on top.
fn cell_op_affine(
    op: &FloatOp,
    a: &PluckVal,
    b: &PluckVal,
    ctx: &mut CompilerCtx,
    pc: BooleanFunction,
) -> (
    f64,
    std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>,
) {
    let a_known = entry_constant(a.as_ref());
    let b_known = entry_constant(b.as_ref());
    let empty = std::collections::BTreeMap::new();
    match (op, a_known, b_known) {
        // Add: 0 + x = x; x + 0 = x.
        (FloatOp::Add, Some(0.0), _) => return force_entry_to_affine(ctx, b, pc),
        (FloatOp::Add, _, Some(0.0)) => return force_entry_to_affine(ctx, a, pc),
        // Sub: x - 0 = x. (0 - x still requires forcing x to negate it.)
        (FloatOp::Sub, _, Some(0.0)) => return force_entry_to_affine(ctx, a, pc),
        // Mul: 0 * x = x * 0 = 0 (skip forcing other side).
        (FloatOp::Mul, Some(0.0), _) | (FloatOp::Mul, _, Some(0.0)) => return (0.0, empty),
        // Mul: 1 * x = x * 1 = x.
        (FloatOp::Mul, Some(1.0), _) => return force_entry_to_affine(ctx, b, pc),
        (FloatOp::Mul, _, Some(1.0)) => return force_entry_to_affine(ctx, a, pc),
        // Div: 0 / x = 0 (skip forcing divisor); x / 1 = x.
        (FloatOp::Div, Some(0.0), _) => return (0.0, empty),
        (FloatOp::Div, _, Some(1.0)) => return force_entry_to_affine(ctx, a, pc),
        _ => {}
    }
    let aa = force_entry_to_affine(ctx, a, pc.clone());
    let bb = force_entry_to_affine(ctx, b, pc);
    affine_binop(op, &aa, &bb)
}

fn cell_op(
    op: &FloatOp,
    a: &PluckVal,
    b: &PluckVal,
    ctx: &mut CompilerCtx,
    pc: BooleanFunction,
) -> PluckVal {
    let (c, m) = cell_op_affine(op, a, b, ctx, pc);
    pack_affine(c, m)
}

/// Borrow the `entries` / `shape` of a `FloatMatrix` `PluckVal`, panicking
/// with `context` in the message when the value isn't a FloatMatrix.
fn expect_float_matrix<'a>(val: &'a PluckVal, context: &str) -> (&'a [PluckVal], &'a [usize]) {
    match val.as_ref() {
        PluckValue::FloatMatrix { entries, shape } => (entries, shape),
        _ => panic!("{}: expected a FloatMatrix, got {:?}", context, val),
    }
}

/// Lift a rank-1 or rank-2 shape into canonical `(rows, cols, was_1d)` for
/// matmul. The left operand's 1D is treated as a row (1 × n); the right
/// operand's 1D is treated as a column (n × 1). Panics on rank ≥ 3.
fn lift_for_matmul(shape: &[usize], is_left: bool) -> (usize, usize, bool) {
    match shape {
        [n] => {
            if is_left {
                (1, *n, true)
            } else {
                (*n, 1, true)
            }
        }
        [r, c] => (*r, *c, false),
        _ => panic!("@: rank > 2 not supported, got shape {:?}", shape),
    }
}

/// Wrap the row-major `out_entries` buffer into the right `PluckVal` for the
/// operand-shape combination. 1D × 1D yields a bare scalar (no wrapping).
fn squeeze_matmul_result(
    out_entries: Vec<PluckVal>,
    a_rows: usize,
    b_cols: usize,
    a_was_1d: bool,
    b_was_1d: bool,
) -> PluckVal {
    match (a_was_1d, b_was_1d) {
        (true, true) => {
            debug_assert_eq!(out_entries.len(), 1);
            out_entries.into_iter().next().unwrap()
        }
        (true, false) => mk_float_matrix(out_entries, vec![b_cols]),
        (false, true) => mk_float_matrix(out_entries, vec![a_rows]),
        (false, false) => mk_float_matrix(out_entries, vec![a_rows, b_cols]),
    }
}

fn merge_coefficients(
    a: &std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>,
    b: &std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>,
    op: fn(f64, f64) -> f64,
) -> std::collections::BTreeMap<u64, ordered_float::NotNan<f64>> {
    let mut result = a.clone();
    for (&k, v) in b {
        let entry = result
            .entry(k)
            .or_insert_with(|| ordered_float::NotNan::new(0.0).unwrap());
        let combined = op(entry.into_inner(), v.into_inner());
        *entry = ordered_float::NotNan::new(combined).expect("merge_coefficients produced NaN");
        if entry.into_inner().abs() < 1e-30 {
            result.remove(&k);
        }
    }
    result
}

fn scale_coefficients(
    m: &std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>,
    s: f64,
) -> std::collections::BTreeMap<u64, ordered_float::NotNan<f64>> {
    m.iter()
        .map(|(&k, v)| {
            (
                k,
                ordered_float::NotNan::new(v.into_inner() * s).expect("scale_coefficients NaN"),
            )
        })
        .collect()
}

fn as_float_or_panic(val: &PluckValue, context: &str) -> f64 {
    match val {
        PluckValue::Native(NativeVal::Float(f)) => *f,
        PluckValue::Native(NativeVal::Int(i)) => *i as f64,
        _ => panic!("{}, got {:?}", context, val),
    }
}

/// Extract a 1-D Dirichlet probability vector from a `PluckValue`.
///
/// Accepts:
/// - `FloatMatrix` with 1-D shape `[n]` whose entries are all `Native(Float)`.
/// - `Value { constructor: _, args }` where every arg is `Native(Float)` —
///   matches the shape produced by `force_symbolic` for a sampled
///   `DirichletVector`.
///
/// Panics on any other shape; this is a user-error path that should be
/// rare in practice.
fn extract_dirichlet_value(val: &PluckValue) -> Box<[ordered_float::NotNan<f64>]> {
    let push = |f: f64, out: &mut Vec<ordered_float::NotNan<f64>>| {
        out.push(
            ordered_float::NotNan::new(f).expect("dirichlet_eq: probability component is NaN"),
        );
    };
    let mut out: Vec<ordered_float::NotNan<f64>> = Vec::new();
    match val {
        PluckValue::FloatMatrix { entries, shape } => {
            assert!(
                shape.len() == 1,
                "dirichlet_eq: value must be a 1-D FloatMatrix, got shape {:?}",
                shape
            );
            for e in entries {
                match e.as_ref() {
                    PluckValue::Native(NativeVal::Float(f)) => push(*f, &mut out),
                    PluckValue::Native(NativeVal::Int(i)) => push(*i as f64, &mut out),
                    _ => panic!(
                        "dirichlet_eq: FloatMatrix entry must be a Native float, got {:?}",
                        e
                    ),
                }
            }
        }
        PluckValue::Value { args, .. } => {
            for a in args {
                match a.as_ref() {
                    PluckValue::Native(NativeVal::Float(f)) => push(*f, &mut out),
                    PluckValue::Native(NativeVal::Int(i)) => push(*i as f64, &mut out),
                    _ => panic!(
                        "dirichlet_eq: constructor arg must be a Native float, got {:?}",
                        a
                    ),
                }
            }
        }
        _ => panic!(
            "dirichlet_eq: value must be a FloatMatrix or constructor of floats, got {:?}",
            val
        ),
    }
    out.into_boxed_slice()
}

fn as_int_or_panic(val: &PluckValue, context: &str) -> i64 {
    match val {
        PluckValue::Native(NativeVal::Int(i)) => *i,
        _ => panic!("{} must be an int, got {:?}", context, val),
    }
}

fn as_int_dist_bits<'a>(val: &'a PluckValue, context: &str) -> &'a [BooleanFunction] {
    match val {
        PluckValue::IntDist { bits } => bits,
        _ => panic!("{} must be an IntDist, got {:?}", context, val),
    }
}

/// Derive a `FlipWeight` from a probability value. Used by both KC and sample flip paths.
fn flip_weight_from(prob_val: &PluckValue) -> FlipWeight {
    prob_val.flip_weight().unwrap_or_else(|| {
        panic!(
            "flip probability must be a float or Probability, got {:?}",
            prob_val
        )
    })
}

/// Find the path to the first thunk in a value tree (DFS).
fn find_first_thunk(val: &PluckVal) -> Option<Vec<usize>> {
    fn find_inner(val: &PluckVal, path: &mut Vec<usize>) -> bool {
        match val.as_ref() {
            PluckValue::Thunk(_) | PluckValue::ThunkUnion(_) => true,
            PluckValue::Value { args, .. } => {
                for (i, arg) in args.iter().enumerate() {
                    path.push(i);
                    if find_inner(arg, path) {
                        return true;
                    }
                    path.pop();
                }
                false
            }
            _ => false,
        }
    }
    let mut path = Vec::new();
    if find_inner(val, &mut path) {
        Some(path)
    } else {
        None
    }
}

/// Find the path to the first non-constant IntDist in a value tree (DFS).
/// A non-constant IntDist has at least one bit that is neither true_ptr nor false_ptr.
fn find_first_non_const_int_dist(val: &PluckVal) -> Option<Vec<usize>> {
    fn find_inner(val: &PluckVal, path: &mut Vec<usize>) -> bool {
        match val.as_ref() {
            PluckValue::IntDist { bits } => !bits.iter().all(|b| b.is_true() || b.is_false()),
            PluckValue::Value { args, .. } => {
                for (i, arg) in args.iter().enumerate() {
                    path.push(i);
                    if find_inner(arg, path) {
                        return true;
                    }
                    path.pop();
                }
                false
            }
            _ => false,
        }
    }
    let mut path = Vec::new();
    if find_inner(val, &mut path) {
        Some(path)
    } else {
        None
    }
}

fn get_value_at_path(val: &PluckVal, path: &[usize]) -> PluckVal {
    if path.is_empty() {
        return val.clone();
    }
    match val.as_ref() {
        PluckValue::Value { args, .. } => get_value_at_path(&args[path[0]], &path[1..]),
        _ => val.clone(),
    }
}

fn replace_at_path(val: &PluckVal, path: &[usize], new_val: &PluckVal) -> PluckVal {
    if path.is_empty() {
        return new_val.clone();
    }
    match val.as_ref() {
        PluckValue::Value { constructor, args } => {
            let mut new_args = args.clone();
            if path.len() == 1 {
                new_args[path[0]] = new_val.clone();
            } else {
                new_args[path[0]] = replace_at_path(&args[path[0]], &path[1..], new_val);
            }
            mk_value(*constructor, new_args)
        }
        _ => val.clone(),
    }
}

#[cfg(test)]
mod try_scale_gamma_tests {
    use super::*;

    fn assert_scaled_gamma(v: &PluckVal, expect_name: u64, expect_scale: f64) {
        match v.as_ref() {
            PluckValue::Gamma { name, scale } => {
                assert_eq!(*name, expect_name);
                assert_eq!(*scale, expect_scale);
            }
            other => panic!("expected scaled Gamma, got {other:?}"),
        }
    }

    #[test]
    fn mul_by_constant_scales_either_order() {
        let g = mk_gamma(7); // scale 1
                             // g * 5.0
        let r = try_scale_gamma(FloatOp::Mul, &g, &mk_native_float(5.0)).unwrap();
        assert_scaled_gamma(&r, 7, 5.0);
        // 5.0 * g  (constant on the left)
        let r = try_scale_gamma(FloatOp::Mul, &mk_native_float(5.0), &g).unwrap();
        assert_scaled_gamma(&r, 7, 5.0);
    }

    #[test]
    fn scale_composes() {
        // (g * 2) * 3  ==  g * 6
        let g2 = mk_gamma_scaled(7, 2.0);
        let r = try_scale_gamma(FloatOp::Mul, &g2, &mk_native_float(3.0)).unwrap();
        assert_scaled_gamma(&r, 7, 6.0);
    }

    #[test]
    fn div_by_constant_scales() {
        let g = mk_gamma(7);
        let r = try_scale_gamma(FloatOp::Div, &g, &mk_native_float(2.0)).unwrap();
        assert_scaled_gamma(&r, 7, 0.5);
    }

    #[test]
    fn no_gamma_falls_through() {
        // Two plain floats: not our business — return None so the numeric
        // path handles it.
        let r = try_scale_gamma(FloatOp::Mul, &mk_native_float(2.0), &mk_native_float(3.0));
        assert!(r.is_none());
    }

    #[test]
    #[should_panic(expected = "addition/subtraction of a Gamma")]
    fn add_of_gamma_panics() {
        try_scale_gamma(FloatOp::Add, &mk_gamma(7), &mk_native_float(1.0));
    }

    #[test]
    #[should_panic(expected = "can only be scaled by a numeric constant")]
    fn gamma_times_gamma_panics() {
        try_scale_gamma(FloatOp::Mul, &mk_gamma(7), &mk_gamma(8));
    }

    #[test]
    #[should_panic(expected = "Exponential/Poisson draw")]
    fn scaling_a_draw_panics() {
        let draw = mk_exponential(7, 99, 1.0);
        try_scale_gamma(FloatOp::Mul, &draw, &mk_native_float(5.0));
    }

    #[test]
    #[should_panic(expected = "positive scale")]
    fn scaling_by_zero_panics() {
        try_scale_gamma(FloatOp::Mul, &mk_gamma(7), &mk_native_float(0.0));
    }

    #[test]
    #[should_panic(expected = "constant / Gamma is not supported")]
    fn constant_over_gamma_panics() {
        try_scale_gamma(FloatOp::Div, &mk_native_float(5.0), &mk_gamma(7));
    }
}
