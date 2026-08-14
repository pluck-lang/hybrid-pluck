use std::fmt;

use super::types::Symbol;

/// A guard in a case/match expression: constructor name + bound variable names.
#[derive(Debug, Clone)]
pub struct CaseOfGuard {
    pub constructor: Symbol,
    pub args: Vec<Symbol>,
}

/// The core AST for Pluck expressions.
///
/// Corresponds to Julia's `PExpr{H}` but as a flat enum rather than a generic struct.
#[derive(Debug, Clone)]
pub enum PExpr {
    /// Function application: (f x)
    App {
        func: Box<PExpr>,
        arg: Box<PExpr>,
    },
    /// Lambda abstraction: (λ var -> body)
    Abs {
        var: Symbol,
        body: Box<PExpr>,
    },
    /// Variable reference (De Bruijn-like: we store the name but resolve to env index)
    Var {
        name: Symbol,
    },
    /// Reference to a top-level definition
    Defined {
        name: Symbol,
    },
    /// Algebraic data type constructor: (Constructor arg1 arg2 ...)
    Construct {
        constructor: Symbol,
        args: Vec<PExpr>,
    },
    /// Pattern matching: (case scrutinee of ...)
    CaseOf {
        guards: Vec<CaseOfGuard>,
        scrutinee: Box<PExpr>,
        branches: Vec<PExpr>,
    },
    /// Bernoulli flip: (flip p)
    Flip {
        prob: Box<PExpr>,
    },
    /// Y-combinator for recursion: (Y f)
    YComb {
        func: Box<PExpr>,
    },
    /// Native constant value (float, int, symbol)
    ConstNative {
        val: NativeVal,
    },

    // --- Primitive operations ---
    /// Native equality: (native_eq a b)
    NativeEq {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    /// Make integer distribution: (mk_int bitwidth value)
    MkInt {
        bitwidth: Box<PExpr>,
        value: Box<PExpr>,
    },
    /// Uniform random integer: (uniform_int bitwidth)
    UniformInt {
        bitwidth: Box<PExpr>,
    },
    /// Uniform random integer in range: (uniform_int_range bitwidth lo hi)
    UniformIntRange {
        bitwidth: Box<PExpr>,
        lo: Box<PExpr>,
        hi: Box<PExpr>,
    },
    /// Integer distribution equality: (int_eq a b)
    IntDistEq {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    /// Integer distribution addition (wrapping): (int_add a b)
    IntDistAdd {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    /// Integer distribution subtraction (wrapping): (int_sub a b)
    IntDistSub {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    /// Integer distribution less-than: (int_lt a b)
    IntDistLt {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    /// Get constructor arguments as a list: (get_args expr)
    GetArgs {
        expr: Box<PExpr>,
    },
    /// Get constructor name as a symbol: (get_constructor expr)
    GetConstructor {
        expr: Box<PExpr>,
    },
    /// Print and return: (print expr)
    Print {
        expr: Box<PExpr>,
    },
    /// Raise an error
    Error {
        msg: Box<PExpr>,
    },
    /// Index into a DirichletVector or FloatMatrix: (vector_index v i [j])
    /// Vector-first, variadic indices.
    Index {
        v: Box<PExpr>,
        idx: Vec<PExpr>,
    },

    // --- Float operations ---
    /// Float division: (/. a b)
    FDiv {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    /// Float multiplication: (*. a b)
    FMul {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    /// Float addition: (+. a b)
    FAdd {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    /// Float subtraction: (-. a b)
    FSub {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },

    // --- Grammar/metaprogramming symbols (for program synthesis) ---
    GSymbol {
        name: Symbol,
    },
    GVarSymbol {
        type_name: Symbol,
    },

    // --- Conjugate prior primitives ---
    /// Beta distribution: (beta a b) — creates symbolic probability variable.
    Beta {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    /// Probability equality test: (prob_eq prob_expr real_const)
    /// Returns boolean; used for disintegration/conditioning on continuous values.
    ProbEq {
        prob: Box<PExpr>,
        val: Box<PExpr>,
    },

    /// Gaussian distribution: (gaussian mu sigma) — creates symbolic Gaussian variable.
    Gaussian {
        mu: Box<PExpr>,
        sigma: Box<PExpr>,
    },
    /// Real equality test: (real_eq expr real_const)
    /// Returns boolean; used for observing Gaussian, Gamma, and Exponential expressions.
    RealEq {
        expr: Box<PExpr>,
        val: Box<PExpr>,
    },
    /// Real range test: (real_lt expr real_const) — `expr < val`,
    /// i.e. the half-open event `[0, val)` for Exponential draws.
    /// `real_lt` / `real_geq` are exactly closed under negation.
    RealLt {
        expr: Box<PExpr>,
        val: Box<PExpr>,
    },
    /// Real range test: (real_geq expr real_const) — `expr ≥ val`.
    RealGeq {
        expr: Box<PExpr>,
        val: Box<PExpr>,
    },

    /// Dirichlet distribution: (dirichlet alpha1 alpha2 ...) — creates symbolic Dirichlet variable.
    Dirichlet {
        alphas: Vec<PExpr>,
    },
    /// Dirichlet vector equality test: (dirichlet_eq dirichlet_expr value_matrix).
    /// Returns boolean; observes the Dirichlet vector equal to a specific
    /// probability vector. Parallels `RealEq` for Gaussian variables.
    DirichletEq {
        expr: Box<PExpr>,
        val: Box<PExpr>,
    },

    /// Categorical distribution: (categorical p1 p2 ...) — creates symbolic categorical variable.
    Categorical {
        probs: Vec<PExpr>,
    },

    /// Integer range test: (native_leq a b) — `a ≤ b` over native ints /
    /// Peano nats; observes Poisson draws when one side is a draw.
    NativeLeq {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    /// Integer range test: (native_geq a b) — `a ≥ b`.
    NativeGeq {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },

    // Poisson distribution: (poisson r) - creates symbolic poisson variable
    Poisson {
        rate: Box<PExpr>,
    },

    // Exponential distribution: (exponential r) - creates symbolic exponential variable
    Exponential {
        rate: Box<PExpr>,
    },

    // Gamma distribution: (gamma shape rate) - creates symbolic gamma variable
    Gamma {
        shape: Box<PExpr>,
        rate: Box<PExpr>,
    },

    // --- Matrix/Vector support ---
    /// Float matrix literal: `{1, 2, 3}` or `{{1, 2, 3,}, {4, 5, 6}}`.
    /// Entries are arbitrary expressions; shape is `[n]` for 1D, `[rows, cols]` for 2D.
    /// Row-major storage: `entries.len() == shape[0]` (or 1 for 1D),
    /// and every row has `shape.last().unwrap()` columns.
    FloatMatrix {
        entries: Vec<Vec<PExpr>>,
        shape: Vec<usize>,
    },

    /// Matrix multiplication: (@ a b)
    MatMul {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },

    /// Sum all entries of a FloatMatrix: (sum m)
    Sum {
        expr: Box<PExpr>,
    },

    /// Pin an expr to a value - used internally by the Gibbs Sampler, no corresponding language construct
    Pin {
        expr: Box<PExpr>,
        val: Box<PExpr>,
    },
}

/// A native (non-ADT) value that can appear in a ConstNative.
#[derive(Debug, Clone, PartialEq)]
pub enum NativeVal {
    Float(f64),
    Int(i64),
    Symbol(Symbol),
    Bool(bool),
}

impl fmt::Display for NativeVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NativeVal::Float(v) => write!(f, "{}", v),
            NativeVal::Int(v) => write!(f, "{}", v),
            NativeVal::Symbol(s) => write!(f, "'{}", s),
            NativeVal::Bool(b) => write!(f, "{}", b),
        }
    }
}
