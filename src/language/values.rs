use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::rc::Rc;

use super::pexpr::{NativeVal, PExpr};
use super::types::Symbol;
use crate::discrete_factorizations::BooleanFunction;
use crate::inference::conjugate_pairs::{
    Assignment, BetaFlip, ContVarName, DirichletFlip, GaussianAffineExpr,
};
use crate::inference::lazy_kc::state::FlipWeight;
use crate::utils::formatting::{subscript_str, NamingScheme};

/// A reference-counted Pluck value. Used everywhere values are stored
/// (environments, constructor args, thunk results).
pub type PluckVal = Rc<PluckValue>;

/// Per-thunk evaluation cache: list of `(worlds, used_information_bf)`
/// pairs. A cached entry is valid when `path_condition => used_information`.
pub type ThunkCache = RefCell<Vec<(Vec<(PluckVal, BooleanFunction)>, BooleanFunction)>>;

/// The runtime value type for Pluck.
///
/// In Julia this was split across Value, NativeValue, Closure, and Thunk.
/// Here we unify into a single enum for simplicity.
#[derive(Debug, Clone)]
pub enum PluckValue {
    /// An algebraic data type value: constructor + arguments.
    /// Arguments may be thunks (lazy) or fully evaluated.
    Value {
        constructor: Symbol,
        args: Vec<PluckVal>,
    },
    /// A native (Julia/Rust) value: float, int, or symbol.
    Native(NativeVal),
    /// A closure: captured environment + body expression.
    Closure(ClosureData),
    /// An IntDist: a fixed-width integer represented as a vector of BDD bits.
    /// Each bit is a BDD pointer representing the probability distribution over that bit.
    /// LSB-first ordering: bits[0] is the least significant bit.
    IntDist { bits: Vec<BooleanFunction> },
    /// A lazy thunk: unevaluated expression with captured environment.
    /// Used by the lazy KC backend for lazy argument passing.
    Thunk(ThunkData),
    /// A union of thunks from different branches, each guarded by a BDD.
    /// Created when join_monad merges values with the same constructor.
    ThunkUnion(ThunkUnionData),
    /// A symbolic Beta variable: references `prior_registry` in LazyKCState.
    /// Known-constant probabilities use Native(Float) instead.
    Probability(u64),
    /// A symbolic Dirichlet variable: references `prior_registry` in LazyKCState.
    DirichletVector { name: u64, categories: usize },
    /// A symbolic Dirichlet probability, corresponding to a single element of a DirichletVector
    DirichletProbability {
        name: u64,
        index: usize,
        categories: usize,
    },
    /// An affine combination of Gaussian variables: constant + sum(coeff_i * X_i).
    /// Used during compilation; the coefficients track linear combinations.
    /// TODO: can this be a HashMap?
    GaussianExpr {
        constant: f64,
        coefficients: BTreeMap<u64, ordered_float::NotNan<f64>>,
    },
    /// A symbolic Gamma variable, optionally scaled by a positive constant.
    /// `scale` is the multiplier applied to the underlying gamma variable
    /// `name`: the value represents `scale * g`. A bare gamma has `scale == 1.0`.
    /// Scaling is kept lazily here (rather than rewritten into a fresh gamma)
    /// so a single shared `g` can be scaled differently at different sites and
    /// still update one pooled conjugate posterior.
    Gamma { name: u64, scale: f64 },
    /// A symbolic Exponential variable which references *both* the underling
    /// gamma rate parameter and itself. `scale` is the multiplier on the rate:
    /// the draw is distributed `Exp(scale * g)`. The draw stored under `name`
    /// is already sampled at the scaled rate, so forcing reads it directly;
    /// `scale` is retained for building observation flips at the use site.
    Exponential { name: u64, gamma: u64, scale: f64 },
    /// A symbolic Poisson variable which references *both* the underling
    /// gamma rate parameter and itself. `scale` is the multiplier on the rate:
    /// the draw is distributed `Poisson(scale * g)`. As with `Exponential`,
    /// the stored draw is already at the scaled rate.
    Poisson { name: u64, gamma: u64, scale: f64 },
    /// A float matrix. Row-major flat storage; `shape` is `[n]` for 1D or
    /// `[rows, cols]` for 2D. Entries are heterogeneous: any entry may be a
    /// `Native(Float)`, a `GaussianExpr`, or a `Thunk` waiting to be forced.
    /// `entries.len()` always equals `shape.iter().product()`.
    FloatMatrix {
        entries: Vec<PluckVal>,
        shape: Vec<usize>,
    },
}

/// Data for a thunk value (lazy KC backend).
#[derive(Clone)]
pub struct ThunkData {
    /// The expression to evaluate when forced.
    pub expr: Rc<PExpr>,
    /// The captured environment.
    pub env: Env,
    /// The strict order index for variable ordering.
    pub strict_order_index: i32,
    /// The callstack at thunk creation time.
    pub callstack: Vec<i32>,
    /// Evaluation cache; see `ThunkCache`.
    pub cache: ThunkCache,
}

impl fmt::Debug for ThunkData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Thunk(callstack={:?}, cache_len={})",
            self.callstack,
            self.cache.borrow().len()
        )
    }
}

/// Data for a thunk union value (lazy KC backend).
#[derive(Clone)]
pub struct ThunkUnionData {
    /// List of (thunk_or_value, guard_bf) pairs.
    pub thunks: Vec<(PluckVal, BooleanFunction)>,
}

impl fmt::Debug for ThunkUnionData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ThunkUnion(len={})", self.thunks.len())
    }
}

/// Data for a closure value.
#[derive(Clone)]
pub struct ClosureData {
    /// The variable name bound by the lambda
    pub var: Symbol,
    /// The body expression
    pub body: Rc<PExpr>,
    /// The captured environment
    pub env: Env,
    /// For self-referential closures (Y combinator), the self-reference
    /// is stored as an Option that gets filled in after construction.
    pub self_ref: RefCell<Option<PluckVal>>,
    /// Name of the recursive function (for Y combinator closures).
    pub rec_name: Option<Symbol>,
}

impl fmt::Debug for ClosureData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Closure(var={:?}, body=..., env_len={})",
            self.var,
            env_len(&self.env)
        )
    }
}

/// An environment is a persistent linked list of bindings.
/// This matches Julia's EnvCons/EnvNil.
pub type Env = Rc<EnvInner>;

#[derive(Debug, Clone)]
pub enum EnvInner {
    Nil,
    Cons {
        name: Symbol,
        val: PluckVal,
        tail: Env,
    },
}

pub fn empty_env() -> Env {
    Rc::new(EnvInner::Nil)
}

pub fn env_cons(name: Symbol, val: PluckVal, tail: Env) -> Env {
    Rc::new(EnvInner::Cons { name, val, tail })
}

/// Look up a variable by name in the environment.
/// Returns the first match (innermost binding).
pub fn env_lookup(env: &Env, name: Symbol) -> Option<PluckVal> {
    let mut current = env;
    loop {
        match current.as_ref() {
            EnvInner::Nil => return None,
            EnvInner::Cons { name: n, val, tail } => {
                if *n == name {
                    return Some(val.clone());
                }
                current = tail;
            }
        }
    }
}

pub fn env_len(env: &Env) -> usize {
    let mut count = 0;
    let mut current = env;
    loop {
        match current.as_ref() {
            EnvInner::Nil => return count,
            EnvInner::Cons { tail, .. } => {
                count += 1;
                current = tail;
            }
        }
    }
}

/// Decode a natural number from `val`. Accepts either a `Native(Int)`
/// or a strict Peano-encoded `S(S(...(O)))` chain where every cons is
/// `s_sym` (arity 1) and the terminator is `o_sym` (arity 0). Returns
/// `None` for anything else — including unary chains with the wrong
/// constructors (e.g. `Cons` instead of `S`).
pub fn extract_nat(val: &PluckVal, o_sym: Symbol, s_sym: Symbol) -> Option<u64> {
    if let PluckValue::Native(NativeVal::Int(i)) = val.as_ref() {
        return Some(*i as u64);
    }
    let mut count: u64 = 0;
    let mut current = val;
    loop {
        match current.as_ref() {
            PluckValue::Value { constructor, args } if *constructor == o_sym && args.is_empty() => {
                return Some(count);
            }
            PluckValue::Value { constructor, args } if *constructor == s_sym && args.len() == 1 => {
                count += 1;
                current = &args[0];
            }
            _ => return None,
        }
    }
}

// --- Convenience constructors ---

pub fn mk_value(constructor: Symbol, args: Vec<PluckVal>) -> PluckVal {
    Rc::new(PluckValue::Value { constructor, args })
}

pub fn mk_native_float(v: f64) -> PluckVal {
    Rc::new(PluckValue::Native(NativeVal::Float(v)))
}

pub fn mk_native_int(v: i64) -> PluckVal {
    Rc::new(PluckValue::Native(NativeVal::Int(v)))
}

pub fn mk_native_symbol(s: Symbol) -> PluckVal {
    Rc::new(PluckValue::Native(NativeVal::Symbol(s)))
}

pub fn mk_native_bool(b: bool) -> PluckVal {
    Rc::new(PluckValue::Native(NativeVal::Bool(b)))
}

pub fn mk_closure(var: Symbol, body: Rc<PExpr>, env: Env) -> PluckVal {
    Rc::new(PluckValue::Closure(ClosureData {
        var,
        body,
        env,
        self_ref: RefCell::new(None),
        rec_name: None,
    }))
}

pub fn mk_int_dist(bits: Vec<BooleanFunction>) -> PluckVal {
    Rc::new(PluckValue::IntDist { bits })
}

/// Create a self-referential closure for the Y combinator.
/// This is `make_self_loop` from the Julia code.
///
/// Y takes `(λ rec_name -> (λ arg_name -> body))` and creates a closure
/// where `rec_name` is bound to the closure itself.
pub fn mk_y_closure(body: Rc<PExpr>, env: Env, rec_name: Symbol, arg_name: Symbol) -> PluckVal {
    let closure = Rc::new(PluckValue::Closure(ClosureData {
        var: arg_name,
        body,
        env,
        self_ref: RefCell::new(None),
        rec_name: Some(rec_name),
    }));
    // Set up the self-reference
    if let PluckValue::Closure(ref data) = *closure {
        *data.self_ref.borrow_mut() = Some(closure.clone());
    }
    closure
}

pub fn mk_thunk(
    expr: Rc<PExpr>,
    env: Env,
    strict_order_index: i32,
    callstack: Vec<i32>,
) -> PluckVal {
    Rc::new(PluckValue::Thunk(ThunkData {
        expr,
        env,
        strict_order_index,
        callstack,
        cache: RefCell::new(Vec::new()),
    }))
}

pub fn mk_thunk_union(thunks: Vec<(PluckVal, BooleanFunction)>) -> PluckVal {
    Rc::new(PluckValue::ThunkUnion(ThunkUnionData { thunks }))
}

pub fn mk_probability(beta_name: u64) -> PluckVal {
    Rc::new(PluckValue::Probability(beta_name))
}

pub fn mk_gamma(gamma_name: u64) -> PluckVal {
    Rc::new(PluckValue::Gamma {
        name: gamma_name,
        scale: 1.0,
    })
}

/// Create a scaled gamma value representing `scale * g`.
pub fn mk_gamma_scaled(gamma_name: u64, scale: f64) -> PluckVal {
    debug_assert!(scale > 0.0, "gamma scale must be positive, got {scale}");
    Rc::new(PluckValue::Gamma {
        name: gamma_name,
        scale,
    })
}

pub fn mk_exponential(gamma_name: u64, exponential_name: u64, scale: f64) -> PluckVal {
    debug_assert!(
        scale > 0.0,
        "exponential scale must be positive, got {scale}"
    );
    Rc::new(PluckValue::Exponential {
        gamma: gamma_name,
        name: exponential_name,
        scale,
    })
}

pub fn mk_poisson(gamma_name: u64, poisson_name: u64, scale: f64) -> PluckVal {
    debug_assert!(scale > 0.0, "poisson scale must be positive, got {scale}");
    Rc::new(PluckValue::Poisson {
        gamma: gamma_name,
        name: poisson_name,
        scale,
    })
}

pub fn mk_dirichlet_probability(dirichlet_name: u64, index: usize, categories: usize) -> PluckVal {
    Rc::new(PluckValue::DirichletProbability {
        name: dirichlet_name,
        index,
        categories,
    })
}

pub fn mk_dirichlet(dirichlet_name: u64, categories: usize) -> PluckVal {
    Rc::new(PluckValue::DirichletVector {
        name: dirichlet_name,
        categories,
    })
}

pub fn mk_gaussian_expr(
    constant: f64,
    coefficients: BTreeMap<u64, ordered_float::NotNan<f64>>,
) -> PluckVal {
    Rc::new(PluckValue::GaussianExpr {
        constant,
        coefficients,
    })
}

pub fn mk_float_matrix(entries: Vec<PluckVal>, shape: Vec<usize>) -> PluckVal {
    let expected: usize = shape.iter().product();
    debug_assert_eq!(
        entries.len(),
        expected,
        "FloatMatrix entry count {} does not match shape {:?} (= {})",
        entries.len(),
        shape,
        expected
    );
    Rc::new(PluckValue::FloatMatrix { entries, shape })
}

/// Peek a matrix entry without forcing. Returns `Some(f)` iff the entry is
/// exactly a `Native(Float)`. Used by the 0/1 fast paths in matrix ops to
/// short-circuit without evaluating thunks.
pub fn entry_constant(v: &PluckValue) -> Option<f64> {
    match v {
        PluckValue::Native(NativeVal::Float(f)) => Some(*f),
        _ => None,
    }
}

// ============================================================================
// Symbolic-value dispatch
//
// The "symbolic" `PluckValue` variants (`Probability`,
// `DirichletProbability`, `DirichletVector`, `GaussianExpr`, `Gamma`,
// `Exponential`, `Poisson`) reference continuous random variables
// registered in the inference layer. Methods that dispatch per-family
// on these variants live here; adding a new symbolic variant means one
// new arm in each method.
// ============================================================================

/// Classification of a symbolic value for `ResultPositions` collection.
/// Variants carry whatever key the corresponding `ResultPositions` map
/// is indexed by — `ContVarName` for Beta/Dirichlet/Gamma, a
/// freshly-built `GaussianAffineExpr` for Gaussian. The Gamma variant
/// covers the whole gamma family: the rate name for `Gamma` values, the
/// draw name for `Exponential`/`Poisson` values.
#[derive(Debug)]
pub enum SymbolicPositionKind {
    Beta(ContVarName),
    Dirichlet(ContVarName),
    /// The gamma family. `name` is the rate name for `Gamma` values, the
    /// draw name for `Exponential`/`Poisson` values. `scale` is the gamma
    /// scale factor (`1.0` for an unscaled gamma or any draw); a queried
    /// scaled gamma reports its *scaled* distribution.
    Gamma {
        name: ContVarName,
        scale: f64,
    },
    Gaussian(GaussianAffineExpr),
}

impl PluckValue {
    /// True iff this is one of the symbolic-continuous variants.
    pub fn is_symbolic(&self) -> bool {
        matches!(
            self,
            PluckValue::Probability(_)
                | PluckValue::DirichletProbability { .. }
                | PluckValue::DirichletVector { .. }
                | PluckValue::GaussianExpr { .. }
                | PluckValue::Gamma { .. }
                | PluckValue::Exponential { .. }
                | PluckValue::Poisson { .. }
        )
    }

    /// Structural equality on symbolic variants. Returns `false` for
    /// any non-symbolic value, and for the two Dirichlet variants
    /// (preserving the pre-refactor `values_equal` behaviour, which
    /// fell through to `false` for those variants).
    pub fn symbolic_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PluckValue::Probability(a), PluckValue::Probability(b)) => a == b,
            (
                PluckValue::GaussianExpr {
                    constant: ca,
                    coefficients: ma,
                },
                PluckValue::GaussianExpr {
                    constant: cb,
                    coefficients: mb,
                },
            ) => ca.to_bits() == cb.to_bits() && ma == mb,
            // Gamma-family names are globally unique, so name equality
            // is exact value equality — but a bare `g` and a scaled `g*s`
            // share a name yet are different values, so the scale must match.
            (
                PluckValue::Gamma { name: a, scale: sa },
                PluckValue::Gamma { name: b, scale: sb },
            ) => a == b && sa.to_bits() == sb.to_bits(),
            (
                PluckValue::Exponential {
                    name: na,
                    gamma: ga,
                    scale: sa,
                },
                PluckValue::Exponential {
                    name: nb,
                    gamma: gb,
                    scale: sb,
                },
            ) => na == nb && ga == gb && sa.to_bits() == sb.to_bits(),
            (
                PluckValue::Poisson {
                    name: na,
                    gamma: ga,
                    scale: sa,
                },
                PluckValue::Poisson {
                    name: nb,
                    gamma: gb,
                    scale: sb,
                },
            ) => na == nb && ga == gb && sa.to_bits() == sb.to_bits(),
            _ => false,
        }
    }

    /// Force this symbolic value against a posterior-sampled
    /// assignment, returning the concrete `PluckVal` to substitute,
    /// or `None` if not symbolic. The `true_sym` argument is used by
    /// the `DirichletVector` case to wrap the realised probability
    /// vector as a flat constructor-style value matching the
    /// pre-refactor behaviour.
    pub fn force_symbolic(&self, a: &Assignment, true_sym: Symbol) -> Option<PluckVal> {
        match self {
            PluckValue::Probability(name) => {
                let p = a.beta_value(*name).into_inner();
                Some(mk_native_float(p))
            }
            PluckValue::DirichletProbability { name, index, .. } => {
                let p = a.dirichlet_value(*name)[*index].into_inner();
                Some(mk_native_float(p))
            }
            PluckValue::DirichletVector { name, categories } => {
                let probs = a.dirichlet_value(*name);
                assert_eq!(probs.len(), *categories);
                let args: Vec<PluckVal> = probs
                    .iter()
                    .map(|p| mk_native_float(p.into_inner()))
                    .collect();
                // Reconstitute as a flat vector value; downstream callers
                // already treat the categories as positional args.
                Some(mk_value(true_sym, args))
            }
            PluckValue::GaussianExpr {
                constant,
                coefficients,
            } => {
                let mut total = *constant;
                for (name, coef) in coefficients {
                    let g_val = a
                        .gaussian
                        .iter()
                        .find(|(n, _)| *n == *name)
                        .map(|(_, v)| v.into_inner())
                        .expect("force_symbolic: GaussianExpr references unassigned variable");
                    total += coef.into_inner() * g_val;
                }
                Some(mk_native_float(total))
            }
            // The whole gamma family (rate + draws) lives in the
            // assignment's gamma section. Poisson draws are
            // integer-valued and force to native ints (so e.g. Gibbs
            // pins compare them with `Native(Int)`).
            //
            // Asymmetry: a `Gamma` value *is* `scale * g`, so we multiply
            // the realised rate by the scale. An `Exponential`/`Poisson`
            // draw, however, is stored already sampled at the scaled rate
            // `scale * g`, so its realised value needs no transform — the
            // scale is irrelevant when reading a draw back.
            PluckValue::Gamma { name, scale } => {
                Some(mk_native_float(scale * a.gamma_value(*name).into_inner()))
            }
            PluckValue::Exponential { name, .. } => {
                Some(mk_native_float(a.gamma_value(*name).into_inner()))
            }
            PluckValue::Poisson { name, .. } => {
                let v = a.gamma_value(*name).into_inner();
                debug_assert_eq!(
                    v.fract(),
                    0.0,
                    "force_symbolic: non-integer Poisson draw {v}"
                );
                Some(mk_native_int(v as i64))
            }
            _ => None,
        }
    }

    /// Map a flip-target value (symbolic OR `Native(Float)`) to its
    /// `FlipWeight`. Returns `None` for non-flip-targets.
    pub fn flip_weight(&self) -> Option<FlipWeight> {
        match self {
            PluckValue::Native(NativeVal::Float(f)) => Some(FlipWeight::Constant(*f)),
            PluckValue::Probability(name) => Some(FlipWeight::Beta(BetaFlip::Var(*name))),
            PluckValue::DirichletProbability {
                name,
                index,
                categories,
            } => Some(FlipWeight::Dirichlet(DirichletFlip::VarElement {
                name: *name,
                index: *index,
                categories: *categories,
            })),
            _ => None,
        }
    }

    /// Render a symbolic value with the given naming scheme. Returns
    /// `None` for non-symbolic values; for `GaussianExpr` with empty
    /// coefficients (a pure constant) returns the constant as a string.
    pub fn render_symbolic(&self, naming: Option<&NamingScheme>) -> Option<String> {
        match self {
            PluckValue::Probability(name) => {
                Some(match naming.and_then(|n| n.var_names.get(name)) {
                    Some(n) => n.clone(),
                    None => format!("Beta(#{}) [prior]", name),
                })
            }
            PluckValue::DirichletProbability { name, index, .. } => {
                Some(match naming.and_then(|n| n.var_names.get(name)) {
                    Some(n) => format!("{}{}", n, subscript_str(*index)),
                    None => format!("Dirichlet(#{})[{}] [prior]", name, index),
                })
            }
            PluckValue::DirichletVector { name, .. } => {
                Some(match naming.and_then(|n| n.var_names.get(name)) {
                    Some(n) => n.clone(),
                    None => format!("Dirichlet(#{}) [prior]", name),
                })
            }
            PluckValue::GaussianExpr {
                constant,
                coefficients,
            } => {
                if coefficients.is_empty() {
                    return Some(format!("{}", constant));
                }
                if let Some(n) = naming.and_then(|n| {
                    n.expr_names
                        .get(&GaussianAffineExpr::new(coefficients, constant))
                }) {
                    return Some(n.clone());
                }
                // Fallback: raw expression form.
                let mut parts = Vec::new();
                if *constant != 0.0 {
                    parts.push(format!("{}", constant));
                }
                for (name, coeff) in coefficients {
                    if coeff.into_inner() == 1.0 {
                        parts.push(format!("G#{}", name));
                    } else {
                        parts.push(format!("{}*G#{}", coeff, name));
                    }
                }
                Some(format!("GaussianExpr({})", parts.join(" + ")))
            }
            PluckValue::Gamma { name, scale } => {
                let base = match naming.and_then(|n| n.var_names.get(name)) {
                    Some(n) => n.clone(),
                    None => format!("Gamma(#{}) [prior]", name),
                };
                // Show the scale factor when the gamma is scaled (`scale * g`).
                Some(if *scale == 1.0 {
                    base
                } else {
                    format!("{} * {}", scale, base)
                })
            }
            PluckValue::Exponential { name, .. } => {
                Some(match naming.and_then(|n| n.var_names.get(name)) {
                    Some(n) => n.clone(),
                    None => format!("Exponential(#{}) [prior]", name),
                })
            }
            PluckValue::Poisson { name, .. } => {
                Some(match naming.and_then(|n| n.var_names.get(name)) {
                    Some(n) => n.clone(),
                    None => format!("Poisson(#{}) [prior]", name),
                })
            }
            _ => None,
        }
    }

    /// Insert this value's underlying `ContVarName`s into `out`.
    /// Handles self only — callers walk into `Value::args` themselves.
    pub fn scope_into(&self, out: &mut BTreeSet<ContVarName>) {
        match self {
            PluckValue::Probability(name) => {
                out.insert(*name);
            }
            PluckValue::GaussianExpr { coefficients, .. } => {
                for &n in coefficients.keys() {
                    out.insert(n);
                }
            }
            PluckValue::DirichletProbability { name, .. }
            | PluckValue::DirichletVector { name, .. } => {
                out.insert(*name);
            }
            PluckValue::Gamma { name, .. } => {
                out.insert(*name);
            }
            // A draw co-travels with its rate: the rate is the mixing
            // variable the draw's conditional is indexed by, so both
            // belong to the value's continuous scope.
            PluckValue::Exponential { name, gamma, .. }
            | PluckValue::Poisson { name, gamma, .. } => {
                out.insert(*name);
                out.insert(*gamma);
            }
            _ => {}
        }
    }

    /// Classify this symbolic value for `ResultPositions` collection.
    /// Returns `None` for non-symbolic values and for `GaussianExpr`
    /// with empty coefficients (a pure constant carries no position).
    pub fn position_kind(&self) -> Option<SymbolicPositionKind> {
        match self {
            PluckValue::Probability(name) => Some(SymbolicPositionKind::Beta(*name)),
            PluckValue::DirichletProbability { name, .. }
            | PluckValue::DirichletVector { name, .. } => {
                Some(SymbolicPositionKind::Dirichlet(*name))
            }
            PluckValue::GaussianExpr {
                coefficients,
                constant,
            } if !coefficients.is_empty() => Some(SymbolicPositionKind::Gaussian(
                GaussianAffineExpr::new(coefficients, constant),
            )),
            // The position is the variable the value *is*: the rate for
            // `Gamma`, the draw for `Exponential`/`Poisson` (a queried
            // draw's rate gets the `rate_{name}` fallback in
            // `rename_to_affine_posterior_packet` when not itself
            // queried). A scaled `Gamma` carries its scale so the reported
            // distribution can be scaled; draws are in their own units, so
            // their reported scale is always `1.0`.
            PluckValue::Gamma { name, scale } => Some(SymbolicPositionKind::Gamma {
                name: *name,
                scale: *scale,
            }),
            PluckValue::Exponential { name, .. } | PluckValue::Poisson { name, .. } => {
                Some(SymbolicPositionKind::Gamma {
                    name: *name,
                    scale: 1.0,
                })
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::NotNan;

    use super::*;

    fn nn(v: f64) -> NotNan<f64> {
        NotNan::new(v).unwrap()
    }

    #[test]
    fn symbolic_eq_distinguishes_gamma_scale() {
        // A bare gamma and a scaled gamma share a name but are different values.
        let bare = mk_gamma(7);
        let scaled5 = mk_gamma_scaled(7, 5.0);
        let scaled5_again = mk_gamma_scaled(7, 5.0);
        let scaled3 = mk_gamma_scaled(7, 3.0);

        assert!(bare.symbolic_eq(&mk_gamma(7)));
        assert!(scaled5.symbolic_eq(&scaled5_again));
        assert!(!bare.symbolic_eq(&scaled5)); // scale 1 vs 5
        assert!(!scaled5.symbolic_eq(&scaled3)); // scale 5 vs 3
    }

    #[test]
    fn force_symbolic_scales_gamma_but_not_draws() {
        // gamma #7 realised at 2.0.
        let a = Assignment::from_sorted(vec![], vec![], vec![(7, nn(2.0))], vec![]);

        // A scaled gamma forces to `scale * value`.
        let scaled = mk_gamma_scaled(7, 5.0);
        match scaled.force_symbolic(&a, 0).unwrap().as_ref() {
            PluckValue::Native(NativeVal::Float(f)) => assert_eq!(*f, 10.0),
            other => panic!("expected Native(Float), got {other:?}"),
        }

        // A bare gamma forces to its value unchanged.
        match mk_gamma(7).force_symbolic(&a, 0).unwrap().as_ref() {
            PluckValue::Native(NativeVal::Float(f)) => assert_eq!(*f, 2.0),
            other => panic!("expected Native(Float), got {other:?}"),
        }

        // A draw stores its value already at the scaled rate, so forcing it
        // reads the assignment directly — the scale must NOT be applied again.
        let draw = mk_exponential(7, 99, 5.0);
        let a_draw = Assignment::from_sorted(vec![], vec![], vec![(99, nn(2.0))], vec![]);
        match draw.force_symbolic(&a_draw, 0).unwrap().as_ref() {
            PluckValue::Native(NativeVal::Float(f)) => assert_eq!(*f, 2.0),
            other => panic!("expected Native(Float), got {other:?}"),
        }
    }
}
