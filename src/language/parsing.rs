use std::str::FromStr;

use super::define::DefinitionRegistry;
use super::pexpr::{CaseOfGuard, NativeVal, PExpr};
use super::types::{StringInterner, Symbol, TypeRegistry};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Primitive {
    Flip,
    NativeEq,
    MkInt,
    UniformInt,
    UniformIntRange,
    IntEq,
    IntDistEq,
    IntAdd,
    IntSub,
    IntLt,
    GetArgs,
    GetConstructor,
    ConstructorsEqual,
    Print,
    Error,
    FDiv,
    FMul,
    FAdd,
    FSub,
    Beta,
    ProbEq,
    Gaussian,
    RealEq,
    RealLt,
    RealGeq,
    NativeLeq,
    NativeGeq,
    Dirichlet,
    DirichletEq,
    Categorical,
    Index,
    MatMul,
    Sum,
    Poisson,
    Gamma,
    Exponential,
}

impl FromStr for Primitive {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "flip" => Ok(Primitive::Flip),
            "native_eq" => Ok(Primitive::NativeEq),
            "mk_int" => Ok(Primitive::MkInt),
            "uniform_int" => Ok(Primitive::UniformInt),
            "uniform_int_range" => Ok(Primitive::UniformIntRange),
            "int_eq" => Ok(Primitive::IntEq),
            "int_dist_eq" => Ok(Primitive::IntDistEq),
            "int_add" => Ok(Primitive::IntAdd),
            "int_sub" => Ok(Primitive::IntSub),
            "int_lt" => Ok(Primitive::IntLt),
            "get_args" => Ok(Primitive::GetArgs),
            "get_constructor" => Ok(Primitive::GetConstructor),
            "constructors_equal" => Ok(Primitive::ConstructorsEqual),
            "print" => Ok(Primitive::Print),
            "error" => Ok(Primitive::Error),
            "/." => Ok(Primitive::FDiv),
            "*." => Ok(Primitive::FMul),
            "+." => Ok(Primitive::FAdd),
            "-." => Ok(Primitive::FSub),
            "beta" => Ok(Primitive::Beta),
            "prob_eq" => Ok(Primitive::ProbEq),
            "gaussian" => Ok(Primitive::Gaussian),
            "real_eq" => Ok(Primitive::RealEq),
            "real_lt" => Ok(Primitive::RealLt),
            "real_geq" => Ok(Primitive::RealGeq),
            "native_leq" => Ok(Primitive::NativeLeq),
            "native_geq" => Ok(Primitive::NativeGeq),
            "dirichlet" => Ok(Primitive::Dirichlet),
            "dirichlet_eq" => Ok(Primitive::DirichletEq),
            "categorical" => Ok(Primitive::Categorical),
            "vector_index" => Ok(Primitive::Index),
            "@" => Ok(Primitive::MatMul),
            "sum" => Ok(Primitive::Sum),
            "poisson" => Ok(Primitive::Poisson),
            "gamma" => Ok(Primitive::Gamma),
            "exponential" => Ok(Primitive::Exponential),
            _ => Err(format!("Unknown primitive: '{}'", s)),
        }
    }
}

/// Tokenize a Pluck source string into a list of tokens.
/// Comments start with ";" and run to end of line.
pub fn tokenize(s: &str) -> Vec<String> {
    // Remove line comments: everything from the first ';' to end of line
    let processed: String = s
        .lines()
        .map(|line| {
            if let Some(pos) = line.find(';') {
                &line[..pos]
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut tokens = Vec::new();
    let bytes = processed.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        } else if c == '"' {
            let start = i;
            i += 1;
            while i < bytes.len() && bytes[i] as char != '"' {
                i += 1;
            }
            assert!(i < bytes.len(), "unterminated string literal");
            let token = &processed[start..=i];
            tokens.push(token.to_string());
            i += 1;
        } else if c == '-' && i + 1 < bytes.len() && bytes[i + 1] as char == '>' {
            tokens.push("->".to_string());
            i += 2;
        } else if "(){}[],~`".contains(c) {
            tokens.push(c.to_string());
            i += 1;
        } else {
            let start = i;
            while i < bytes.len() {
                let ch = bytes[i] as char;
                if ch.is_whitespace() || "(){}[],~`\"".contains(ch) {
                    break;
                }
                if ch == '-' && i + 1 < bytes.len() && bytes[i + 1] as char == '>' {
                    break;
                }
                i += 1;
            }
            tokens.push(processed[start..i].to_string());
        }
    }
    tokens
}

/// Parser context holding all mutable state needed during parsing.
pub struct ParseContext<'a> {
    pub interner: &'a mut StringInterner,
    pub types: &'a TypeRegistry,
    pub defs: &'a DefinitionRegistry,
}

/// Recursive descent parser for Pluck expressions.
/// Returns the parsed expression and the remaining unconsumed tokens.
pub fn parse_expr_inner<'a>(
    tokens: &'a [&str],
    ctx: &mut ParseContext,
    env: &[String],
) -> (PExpr, &'a [&'a str]) {
    assert!(!tokens.is_empty(), "unexpected end of input");

    let token = tokens[0];

    if token == "(" {
        let tokens = &tokens[1..];
        let keyword = tokens[0];

        if keyword == "lam" || keyword == "lambda" || keyword == "λ" || keyword == "fn" {
            parse_lambda(&tokens[1..], ctx, env)
        } else if keyword == "if" {
            parse_if(&tokens[1..], ctx, env)
        } else if keyword == "Y" {
            parse_y(&tokens[1..], ctx, env)
        } else if keyword == "case" || keyword == "match" {
            parse_case(&tokens[1..], ctx, env)
        } else if keyword == "let" {
            parse_let(&tokens[1..], ctx, env)
        } else if is_constructor(keyword, ctx) {
            parse_constructor_expr(keyword, &tokens[1..], ctx, env)
        } else if is_prim(keyword) && !ctx.defs.is_defined(ctx.interner.intern(keyword)) {
            parse_prim(keyword, &tokens[1..], ctx, env)
        } else if keyword == "discrete" {
            parse_discrete(&tokens[1..], ctx, env)
        } else if keyword == "uniform" {
            parse_uniform(&tokens[1..], ctx, env)
        } else {
            // Function application: (f x1 x2 ...)
            parse_application(tokens, ctx, env)
        }
    } else if let Some(name) = token.strip_prefix('\'') {
        let sym = ctx.interner.intern(name);
        (
            PExpr::ConstNative {
                val: NativeVal::Symbol(sym),
            },
            &tokens[1..],
        )
    } else if token.starts_with("0c") && token.len() == 3 {
        // Byte literal: 0cX
        let byte = token.as_bytes()[2] as i64;
        let bitwidth = Box::new(PExpr::ConstNative {
            val: NativeVal::Int(8),
        });
        let val = Box::new(PExpr::ConstNative {
            val: NativeVal::Int(byte),
        });
        (
            PExpr::MkInt {
                bitwidth,
                value: val,
            },
            &tokens[1..],
        )
    } else if token.starts_with('"') && token.ends_with('"') {
        // String literal -> list of 8-bit ints
        let inner = &token[1..token.len() - 1];
        let bytes: Vec<u8> = inner.bytes().collect();
        let nil_sym = ctx.interner.intern("Nil");
        let cons_sym = ctx.interner.intern("Cons");
        let mut expr = PExpr::Construct {
            constructor: nil_sym,
            args: vec![],
        };
        for &b in bytes.iter().rev() {
            let mk_int = PExpr::MkInt {
                bitwidth: Box::new(PExpr::ConstNative {
                    val: NativeVal::Int(8),
                }),
                value: Box::new(PExpr::ConstNative {
                    val: NativeVal::Int(b as i64),
                }),
            };
            expr = PExpr::Construct {
                constructor: cons_sym,
                args: vec![mk_int, expr],
            };
        }
        (expr, &tokens[1..])
    } else if token == "[" {
        // List literal: [a, b, c]
        parse_list_literal(&tokens[1..], ctx, env)
    } else if token == "{" {
        // Float-matrix / vector literal: {a, b, c} or {{1, 2}, {3, 4}}
        parse_matrix_literal(&tokens[1..], ctx, env)
    } else if let Some(stripped) = token.strip_prefix('?') {
        let name = ctx.interner.intern(stripped);
        (PExpr::GSymbol { name }, &tokens[1..])
    } else if let Some(stripped) = token.strip_prefix('#') {
        let type_name = ctx.interner.intern(stripped);
        (PExpr::GVarSymbol { type_name }, &tokens[1..])
    } else if token.starts_with('@')
        && token.len() > 1
        && token[1..].chars().all(|c| c.is_ascii_digit())
    {
        // `@N` (digits only) is an integer literal. Bare `@` is the matmul
        // primitive name and is handled by `is_prim` in keyword position.
        let idx: i64 = token[1..].parse().unwrap();
        (
            PExpr::ConstNative {
                val: NativeVal::Int(idx),
            },
            &tokens[1..],
        )
    } else if token.chars().all(|c| c.is_ascii_digit()) {
        // Natural number literal -> unary nat encoding
        let val: u64 = token.parse().unwrap();
        (nat_to_expr(val, ctx), &tokens[1..])
    } else if token.chars().all(|c| c.is_ascii_digit() || c == '.') && token.contains('.') {
        // Float literal (e.g. 3.14)
        let val: f64 = token.parse().unwrap();
        (
            PExpr::ConstNative {
                val: NativeVal::Float(val),
            },
            &tokens[1..],
        )
    } else if token.starts_with('-')
        && token.len() > 1
        && token[1..].chars().all(|c| c.is_ascii_digit() || c == '.')
    {
        // Negative numeric literal (e.g. -2.0, -3)
        let val: f64 = token.parse().unwrap();
        (
            PExpr::ConstNative {
                val: NativeVal::Float(val),
            },
            &tokens[1..],
        )
    } else if token == "true" {
        let true_sym = ctx.interner.intern("True");
        (
            PExpr::Construct {
                constructor: true_sym,
                args: vec![],
            },
            &tokens[1..],
        )
    } else if token == "false" {
        let false_sym = ctx.interner.intern("False");
        (
            PExpr::Construct {
                constructor: false_sym,
                args: vec![],
            },
            &tokens[1..],
        )
    } else if token == "nothing" {
        let unit_sym = ctx.interner.intern("Unit");
        (
            PExpr::Construct {
                constructor: unit_sym,
                args: vec![],
            },
            &tokens[1..],
        )
    } else if env.contains(&token.to_string()) || token.starts_with('$') {
        let name_str = if let Some(stripped) = token.strip_prefix('$') {
            stripped
        } else {
            token
        };
        let name = ctx.interner.intern(name_str);
        (PExpr::Var { name }, &tokens[1..])
    } else if ctx.defs.is_defined(ctx.interner.intern(token)) {
        let name = ctx.interner.intern(token);
        (PExpr::Defined { name }, &tokens[1..])
    } else {
        // Check if it's an argument-less constructor
        let sym = ctx.interner.intern(token);
        if ctx.types.is_constructor(sym) {
            let arity = ctx.types.constructor_arity(sym);
            assert_eq!(
                arity, 0,
                "constructor {} used without arguments but has arity {}",
                token, arity
            );
            (
                PExpr::Construct {
                    constructor: sym,
                    args: vec![],
                },
                &tokens[1..],
            )
        } else {
            panic!("unknown token: '{}' with env {:?}", token, env);
        }
    }
}

fn parse_lambda<'a>(
    tokens: &'a [&str],
    ctx: &mut ParseContext,
    env: &[String],
) -> (PExpr, &'a [&'a str]) {
    let mut tokens = tokens;

    // Handle zero-argument lambda: (λ -> body)
    if tokens[0] == "->" {
        tokens = &tokens[1..];
        let dummy = "_".to_string();
        let mut new_env = vec![dummy];
        new_env.extend_from_slice(env);
        let (body, tokens) = parse_expr_inner(tokens, ctx, &new_env);
        assert_eq!(tokens[0], ")", "expected closing paren in lambda");
        let underscore = ctx.interner.intern("_");
        return (
            PExpr::Abs {
                var: underscore,
                body: Box::new(body),
            },
            &tokens[1..],
        );
    }

    let mut arg_names = Vec::new();
    loop {
        let name = tokens[0].to_string();
        arg_names.push(name);
        tokens = &tokens[1..];
        // Skip optional comma
        if tokens[0] == "," {
            tokens = &tokens[1..];
        }
        if tokens[0] == "->" {
            tokens = &tokens[1..];
            break;
        }
    }

    let mut new_env: Vec<String> = arg_names.iter().rev().cloned().collect();
    new_env.extend_from_slice(env);

    let (body, tokens) = parse_expr_inner(tokens, ctx, &new_env);
    assert_eq!(tokens[0], ")", "expected closing paren in lambda");

    // Wrap body in nested Abs (outermost first)
    let mut expr = body;
    for name in arg_names.iter().rev() {
        let sym = ctx.interner.intern(name);
        expr = PExpr::Abs {
            var: sym,
            body: Box::new(expr),
        };
    }

    (expr, &tokens[1..])
}

fn parse_if<'a>(
    tokens: &'a [&str],
    ctx: &mut ParseContext,
    env: &[String],
) -> (PExpr, &'a [&'a str]) {
    let (cond, tokens) = parse_expr_inner(tokens, ctx, env);
    let (then_expr, tokens) = parse_expr_inner(tokens, ctx, env);
    let (else_expr, tokens) = parse_expr_inner(tokens, ctx, env);
    assert_eq!(tokens[0], ")", "expected closing paren in if");
    let true_sym = ctx.interner.intern("True");
    let false_sym = ctx.interner.intern("False");
    (
        PExpr::CaseOf {
            guards: vec![
                CaseOfGuard {
                    constructor: true_sym,
                    args: vec![],
                },
                CaseOfGuard {
                    constructor: false_sym,
                    args: vec![],
                },
            ],
            scrutinee: Box::new(cond),
            branches: vec![then_expr, else_expr],
        },
        &tokens[1..],
    )
}

fn parse_y<'a>(
    tokens: &'a [&str],
    ctx: &mut ParseContext,
    env: &[String],
) -> (PExpr, &'a [&'a str]) {
    let (f, tokens) = parse_expr_inner(tokens, ctx, env);
    let mut expr = PExpr::YComb { func: Box::new(f) };
    if tokens[0] != ")" {
        // (Y f x) -> App(Y(f), x)
        let (x, tokens) = parse_expr_inner(tokens, ctx, env);
        expr = PExpr::App {
            func: Box::new(expr),
            arg: Box::new(x),
        };
        assert_eq!(tokens[0], ")", "expected closing paren in Y");
        return (expr, &tokens[1..]);
    }
    (expr, &tokens[1..])
}

fn parse_case<'a>(
    tokens: &'a [&str],
    ctx: &mut ParseContext,
    env: &[String],
) -> (PExpr, &'a [&'a str]) {
    let (scrutinee, mut tokens) = parse_expr_inner(tokens, ctx, env);

    // Skip "of" if present (case uses "of", match doesn't)
    if tokens[0] == "of" {
        tokens = &tokens[1..];
    }

    let mut guards = Vec::new();
    let mut branches = Vec::new();

    while tokens[0] != ")" {
        assert_ne!(
            tokens[0], "(",
            "unnecessary parens around pattern match guard"
        );
        let constructor_str = tokens[0];
        let constructor = ctx.interner.intern(constructor_str);
        tokens = &tokens[1..];

        let mut args = Vec::new();

        if tokens[0] == "=>" {
            // Constructor => body (possibly with lambdas for arg binding)
            tokens = &tokens[1..];
            let (mut body, rest) = parse_expr_inner(tokens, ctx, env);
            tokens = rest;
            // Unwrap any leading Abs nodes to extract bound variable names
            while let PExpr::Abs { var, body: inner } = body {
                args.push(var);
                body = *inner;
            }
            guards.push(CaseOfGuard { constructor, args });
            branches.push(body);
        } else {
            // Constructor arg1 arg2 ... => body
            let mut new_env: Vec<String> = Vec::new();
            while tokens[0] != "=>" {
                let arg_name = tokens[0].to_string();
                args.push(ctx.interner.intern(&arg_name));
                new_env.push(arg_name);
                tokens = &tokens[1..];
            }
            tokens = &tokens[1..]; // skip =>
            new_env.reverse();
            let mut combined_env = new_env;
            combined_env.extend_from_slice(env);
            let (body, rest) = parse_expr_inner(tokens, ctx, &combined_env);
            tokens = rest;
            guards.push(CaseOfGuard { constructor, args });
            branches.push(body);
        }

        // Skip optional "|"
        if !tokens.is_empty() && tokens[0] == "|" {
            tokens = &tokens[1..];
        }
    }

    (
        PExpr::CaseOf {
            guards,
            scrutinee: Box::new(scrutinee),
            branches,
        },
        &tokens[1..],
    )
}

fn parse_let<'a>(
    tokens: &'a [&str],
    ctx: &mut ParseContext,
    env: &[String],
) -> (PExpr, &'a [&'a str]) {
    let mut tokens = tokens;
    let close_token = if tokens[0] == "(" { ")" } else { "]" };
    tokens = &tokens[1..]; // skip opening ( or [

    let mut bindings: Vec<(String, PExpr)> = Vec::new();
    let mut extended_env: Vec<String> = Vec::new();
    extended_env.extend_from_slice(env);

    while tokens[0] != close_token {
        if tokens[0] == "(" {
            // Nested pair format: (var val)
            tokens = &tokens[1..];
            let var = tokens[0].to_string();
            tokens = &tokens[1..];
            let (val, rest) = parse_expr_inner(tokens, ctx, &extended_env);
            tokens = rest;
            assert_eq!(tokens[0], ")", "expected closing paren in let binding");
            tokens = &tokens[1..];
            extended_env.insert(0, var.clone());
            bindings.push((var, val));
        } else {
            // Flat format: var val
            let var = tokens[0].to_string();
            tokens = &tokens[1..];
            let (val, rest) = parse_expr_inner(tokens, ctx, &extended_env);
            tokens = rest;
            extended_env.insert(0, var.clone());
            bindings.push((var, val));
        }
    }
    tokens = &tokens[1..];

    let (body, tokens) = parse_expr_inner(tokens, ctx, &extended_env);
    assert_eq!(tokens[0], ")", "expected closing paren at end of let");

    // Desugar to nested (App (Abs var body) val)
    let mut expr = body;
    for (var, val) in bindings.into_iter().rev() {
        let sym = ctx.interner.intern(&var);
        expr = PExpr::App {
            func: Box::new(PExpr::Abs {
                var: sym,
                body: Box::new(expr),
            }),
            arg: Box::new(val),
        };
    }

    (expr, &tokens[1..])
}

fn parse_constructor_expr<'a>(
    name: &str,
    tokens: &'a [&str],
    ctx: &mut ParseContext,
    env: &[String],
) -> (PExpr, &'a [&'a str]) {
    let constructor = ctx.interner.intern(name);
    let mut tokens = tokens;
    let mut args = Vec::new();
    while tokens[0] != ")" {
        let (arg, rest) = parse_expr_inner(tokens, ctx, env);
        args.push(arg);
        tokens = rest;
    }
    let expected_arity = ctx.types.constructor_arity(constructor);
    assert_eq!(
        args.len(),
        expected_arity,
        "wrong number of arguments for constructor {}. Expected {}, got {}",
        name,
        expected_arity,
        args.len()
    );
    (PExpr::Construct { constructor, args }, &tokens[1..])
}

fn is_constructor(name: &str, ctx: &ParseContext) -> bool {
    // intern() requires &mut, so use lookup() instead to check without mutating.
    if let Some(&sym) = ctx.interner_lookup(name) {
        ctx.types.is_constructor(sym)
    } else {
        false
    }
}

impl ParseContext<'_> {
    fn interner_lookup(&self, s: &str) -> Option<&Symbol> {
        self.interner.lookup(s)
    }
}

fn is_prim(name: &str) -> bool {
    name.parse::<Primitive>().is_ok()
}

pub enum Arity {
    Fixed(usize),
    AtLeast(usize),
}

fn prim_arity(prim: &Primitive) -> Arity {
    match prim {
        Primitive::Flip => Arity::Fixed(1),
        Primitive::NativeEq
        | Primitive::ConstructorsEqual
        | Primitive::IntEq
        | Primitive::IntDistEq
        | Primitive::IntAdd
        | Primitive::IntSub
        | Primitive::IntLt
        | Primitive::FDiv
        | Primitive::FMul
        | Primitive::FAdd
        | Primitive::FSub => Arity::Fixed(2),
        Primitive::Index => Arity::AtLeast(2),
        Primitive::MkInt
        | Primitive::Beta
        | Primitive::ProbEq
        | Primitive::Gaussian
        | Primitive::RealEq
        | Primitive::RealLt
        | Primitive::RealGeq
        | Primitive::NativeLeq
        | Primitive::NativeGeq
        | Primitive::DirichletEq => Arity::Fixed(2),
        Primitive::UniformInt => Arity::Fixed(1),
        Primitive::UniformIntRange => Arity::Fixed(3),
        Primitive::GetArgs | Primitive::GetConstructor | Primitive::Print | Primitive::Error => {
            Arity::Fixed(1)
        }
        Primitive::Dirichlet => Arity::AtLeast(1),
        Primitive::Categorical => Arity::AtLeast(1),
        Primitive::MatMul => Arity::Fixed(2),
        Primitive::Sum => Arity::Fixed(1),
        Primitive::Poisson => Arity::Fixed(1),
        Primitive::Gamma => Arity::Fixed(2),
        Primitive::Exponential => Arity::Fixed(1),
    }
}

fn validate_prim_args(prim: &Primitive, args: &[PExpr]) {
    let arity = prim_arity(prim);
    match arity {
        Arity::Fixed(n) => assert_eq!(
            args.len(),
            n,
            "{:?} expected {} arguments, got {}",
            prim,
            n,
            args.len()
        ),
        Arity::AtLeast(n) => assert!(
            args.len() >= n,
            "{:?} expected at least {} arguments, got {}",
            prim,
            n,
            args.len()
        ),
    }
}

fn parse_prim<'a>(
    name: &str,
    tokens: &'a [&str],
    ctx: &mut ParseContext,
    env: &[String],
) -> (PExpr, &'a [&'a str]) {
    let prim = name.parse::<Primitive>().unwrap();
    let mut tokens = tokens;
    let mut args = Vec::new();

    while !tokens.is_empty() && tokens[0] != ")" {
        let (arg, rest) = parse_expr_inner(tokens, ctx, env);
        args.push(arg);
        tokens = rest;
    }

    // Validate arity and paren presence
    assert!(
        !tokens.is_empty() && tokens[0] == ")",
        "expected closing paren after primitive {}",
        name
    );
    validate_prim_args(&prim, &args);

    let expr = match prim {
        Primitive::Flip => PExpr::Flip {
            prob: Box::new(args.remove(0)),
        },
        Primitive::NativeEq | Primitive::ConstructorsEqual => PExpr::NativeEq {
            a: Box::new(args.remove(0)),
            b: Box::new(args.remove(0)),
        },
        Primitive::MkInt => PExpr::MkInt {
            bitwidth: Box::new(args.remove(0)),
            value: Box::new(args.remove(0)),
        },
        Primitive::UniformInt => PExpr::UniformInt {
            bitwidth: Box::new(args.remove(0)),
        },
        Primitive::UniformIntRange => PExpr::UniformIntRange {
            bitwidth: Box::new(args.remove(0)),
            lo: Box::new(args.remove(0)),
            hi: Box::new(args.remove(0)),
        },
        Primitive::IntEq | Primitive::IntDistEq => PExpr::IntDistEq {
            a: Box::new(args.remove(0)),
            b: Box::new(args.remove(0)),
        },
        Primitive::IntAdd => PExpr::IntDistAdd {
            a: Box::new(args.remove(0)),
            b: Box::new(args.remove(0)),
        },
        Primitive::IntSub => PExpr::IntDistSub {
            a: Box::new(args.remove(0)),
            b: Box::new(args.remove(0)),
        },
        Primitive::IntLt => PExpr::IntDistLt {
            a: Box::new(args.remove(0)),
            b: Box::new(args.remove(0)),
        },
        Primitive::GetArgs => PExpr::GetArgs {
            expr: Box::new(args.remove(0)),
        },
        Primitive::GetConstructor => PExpr::GetConstructor {
            expr: Box::new(args.remove(0)),
        },
        Primitive::Print => PExpr::Print {
            expr: Box::new(args.remove(0)),
        },
        Primitive::Error => PExpr::Error {
            msg: Box::new(args.remove(0)),
        },
        Primitive::FDiv => PExpr::FDiv {
            a: Box::new(args.remove(0)),
            b: Box::new(args.remove(0)),
        },
        Primitive::FMul => PExpr::FMul {
            a: Box::new(args.remove(0)),
            b: Box::new(args.remove(0)),
        },
        Primitive::FAdd => PExpr::FAdd {
            a: Box::new(args.remove(0)),
            b: Box::new(args.remove(0)),
        },
        Primitive::FSub => PExpr::FSub {
            a: Box::new(args.remove(0)),
            b: Box::new(args.remove(0)),
        },
        Primitive::Beta => PExpr::Beta {
            a: Box::new(args.remove(0)),
            b: Box::new(args.remove(0)),
        },
        Primitive::ProbEq => PExpr::ProbEq {
            prob: Box::new(args.remove(0)),
            val: Box::new(args.remove(0)),
        },
        Primitive::Gaussian => PExpr::Gaussian {
            mu: Box::new(args.remove(0)),
            sigma: Box::new(args.remove(0)),
        },
        Primitive::RealEq => PExpr::RealEq {
            expr: Box::new(args.remove(0)),
            val: Box::new(args.remove(0)),
        },
        Primitive::RealLt => PExpr::RealLt {
            expr: Box::new(args.remove(0)),
            val: Box::new(args.remove(0)),
        },
        Primitive::RealGeq => PExpr::RealGeq {
            expr: Box::new(args.remove(0)),
            val: Box::new(args.remove(0)),
        },
        Primitive::NativeLeq => PExpr::NativeLeq {
            a: Box::new(args.remove(0)),
            b: Box::new(args.remove(0)),
        },
        Primitive::NativeGeq => PExpr::NativeGeq {
            a: Box::new(args.remove(0)),
            b: Box::new(args.remove(0)),
        },
        Primitive::Dirichlet => PExpr::Dirichlet { alphas: args },
        Primitive::DirichletEq => PExpr::DirichletEq {
            expr: Box::new(args.remove(0)),
            val: Box::new(args.remove(0)),
        },
        Primitive::Categorical => PExpr::Categorical { probs: args },
        Primitive::Index => {
            let v = Box::new(args.remove(0));
            PExpr::Index { v, idx: args }
        }
        Primitive::MatMul => PExpr::MatMul {
            a: Box::new(args.remove(0)),
            b: Box::new(args.remove(0)),
        },
        Primitive::Sum => PExpr::Sum {
            expr: Box::new(args.remove(0)),
        },
        Primitive::Poisson => PExpr::Poisson {
            rate: Box::new(args.remove(0)),
        },
        Primitive::Gamma => PExpr::Gamma {
            shape: Box::new(args.remove(0)),
            rate: Box::new(args.remove(0)),
        },
        Primitive::Exponential => PExpr::Exponential {
            rate: Box::new(args.remove(0)),
        },
    };

    (expr, &tokens[1..])
}

fn parse_discrete<'a>(
    tokens: &'a [&str],
    ctx: &mut ParseContext,
    env: &[String],
) -> (PExpr, &'a [&'a str]) {
    let mut tokens = tokens;
    let mut options = Vec::new();
    let mut probs = Vec::new();

    while tokens[0] != ")" {
        assert_eq!(
            tokens[0], "(",
            "expected opening paren in discrete distribution pair"
        );
        tokens = &tokens[1..];
        let (expr, rest) = parse_expr_inner(tokens, ctx, env);
        tokens = rest;
        let prob: f64 = tokens[0]
            .parse()
            .expect("probability must be a literal number");
        probs.push(prob);
        tokens = &tokens[1..];
        assert_eq!(tokens[0], ")", "expected closing paren in discrete pair");
        tokens = &tokens[1..];
        options.push(expr);
    }

    let expr = build_discrete_tree(&options, &probs, ctx);
    (expr, &tokens[1..])
}

fn parse_uniform<'a>(
    tokens: &'a [&str],
    ctx: &mut ParseContext,
    env: &[String],
) -> (PExpr, &'a [&'a str]) {
    let mut tokens = tokens;
    let mut options = Vec::new();

    while tokens[0] != ")" {
        let (expr, rest) = parse_expr_inner(tokens, ctx, env);
        options.push(expr);
        tokens = rest;
    }

    let n = options.len();
    let probs: Vec<f64> = vec![1.0 / n as f64; n];
    let expr = build_discrete_tree(&options, &probs, ctx);
    (expr, &tokens[1..])
}

/// Build a nested if-expression tree for a discrete distribution.
/// This matches the Julia `discrete()` function behavior.
fn build_discrete_tree(options: &[PExpr], probs: &[f64], ctx: &mut ParseContext) -> PExpr {
    assert!(!options.is_empty());
    if options.len() == 1 {
        return options[0].clone();
    }

    // Conditional probability: P(first | first or rest)
    let total: f64 = probs.iter().sum();
    let p = probs[0] / total;

    let true_sym = ctx.interner.intern("True");
    let false_sym = ctx.interner.intern("False");

    let rest = build_discrete_tree(&options[1..], &probs[1..], ctx);

    PExpr::CaseOf {
        guards: vec![
            CaseOfGuard {
                constructor: true_sym,
                args: vec![],
            },
            CaseOfGuard {
                constructor: false_sym,
                args: vec![],
            },
        ],
        scrutinee: Box::new(PExpr::Flip {
            prob: Box::new(PExpr::ConstNative {
                val: NativeVal::Float(p),
            }),
        }),
        branches: vec![options[0].clone(), rest],
    }
}

fn parse_application<'a>(
    tokens: &'a [&str],
    ctx: &mut ParseContext,
    env: &[String],
) -> (PExpr, &'a [&'a str]) {
    let (f, mut tokens) = parse_expr_inner(tokens, ctx, env);
    let mut args = Vec::new();
    while tokens[0] != ")" {
        let (arg, rest) = parse_expr_inner(tokens, ctx, env);
        args.push(arg);
        tokens = rest;
    }

    // If no arguments, insert Unit
    if args.is_empty() {
        let unit_sym = ctx.interner.intern("Unit");
        args.push(PExpr::Construct {
            constructor: unit_sym,
            args: vec![],
        });
    }

    // Build curried application
    let mut expr = f;
    for arg in args {
        expr = PExpr::App {
            func: Box::new(expr),
            arg: Box::new(arg),
        };
    }

    (expr, &tokens[1..])
}

fn parse_list_literal<'a>(
    tokens: &'a [&str],
    ctx: &mut ParseContext,
    env: &[String],
) -> (PExpr, &'a [&'a str]) {
    let mut tokens = tokens;
    let mut vals = Vec::new();

    while tokens[0] != "]" {
        let (val, rest) = parse_expr_inner(tokens, ctx, env);
        vals.push(val);
        tokens = rest;
        if !tokens.is_empty() && tokens[0] == "," {
            tokens = &tokens[1..];
        }
    }
    tokens = &tokens[1..]; // skip ']'

    let nil_sym = ctx.interner.intern("Nil");
    let cons_sym = ctx.interner.intern("Cons");

    let mut expr = PExpr::Construct {
        constructor: nil_sym,
        args: vec![],
    };
    for val in vals.into_iter().rev() {
        expr = PExpr::Construct {
            constructor: cons_sym,
            args: vec![val, expr],
        };
    }

    (expr, tokens)
}

//  Parse the body of a matrix literal. `tokens` starts after the opening `{`.
//
//  A matrix literal is either:
//    * 2D: every top-level element is a nested `{row}`; all rows must have the
//  same length; shape is `[n_rows, n_cols]`.
//    * 1D: no top-level element is a nested `{...}`; shape is `[n_cols]`.
//  Mixing nested rows and scalar entries at the top level is a parse error.
//  Commas between entries are optional (a trailing comma is allowed).
fn parse_matrix_literal<'a>(
    tokens: &'a [&str],
    ctx: &mut ParseContext,
    env: &[String],
) -> (PExpr, &'a [&'a str]) {
    enum Entry {
        Row(Vec<PExpr>),
        Scalar(PExpr),
    }

    let mut tokens = tokens;
    let mut entries: Vec<Entry> = Vec::new();

    while !tokens.is_empty() && tokens[0] != "}" {
        if tokens[0] == "{" {
            let (row, rest) = parse_matrix_row_body(&tokens[1..], ctx, env);
            entries.push(Entry::Row(row));
            tokens = rest;
        } else {
            let (expr, rest) = parse_expr_inner(tokens, ctx, env);
            entries.push(Entry::Scalar(expr));
            tokens = rest;
        }
        if !tokens.is_empty() && tokens[0] == "," {
            tokens = &tokens[1..];
        }
    }
    assert!(!tokens.is_empty(), "unterminated matrix literal");
    tokens = &tokens[1..]; // skip outer `}`

    let all_rows = entries.iter().all(|e| matches!(e, Entry::Row(_)));
    let all_scalars = entries.iter().all(|e| matches!(e, Entry::Scalar(_)));

    if all_rows {
        let rows: Vec<Vec<PExpr>> = entries
            .into_iter()
            .map(|e| match e {
                Entry::Row(r) => r,
                _ => unreachable!(),
            })
            .collect();
        let cols = rows[0].len();
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(
                r.len(),
                cols,
                "matrix literal row {} has length {}, expected {}",
                i,
                r.len(),
                cols
            );
        }
        let shape = vec![rows.len(), cols];
        (
            PExpr::FloatMatrix {
                entries: rows,
                shape,
            },
            tokens,
        )
    } else if all_scalars {
        let row: Vec<PExpr> = entries
            .into_iter()
            .map(|e| match e {
                Entry::Scalar(s) => s,
                _ => unreachable!(),
            })
            .collect();
        let shape = vec![row.len()];
        (
            PExpr::FloatMatrix {
                entries: vec![row],
                shape,
            },
            tokens,
        )
    } else {
        panic!("matrix literal cannot mix nested rows and scalar entries at the top level");
    }
}

/// Parse the body of a nested matrix row `{...}`. `tokens` starts after the
/// row's opening `{`. Each entry is a scalar PExpr (a nested `{...}` row is not
/// allowed). Commas between entries are optional. Consumes the closing `}`.
fn parse_matrix_row_body<'a>(
    tokens: &'a [&str],
    ctx: &mut ParseContext,
    env: &[String],
) -> (Vec<PExpr>, &'a [&'a str]) {
    let mut tokens = tokens;
    let mut row: Vec<PExpr> = Vec::new();
    while !tokens.is_empty() && tokens[0] != "}" {
        if tokens[0] == "{" {
            panic!("nested matrix rows are not allowed inside a matrix row");
        }
        let (expr, rest) = parse_expr_inner(tokens, ctx, env);
        row.push(expr);
        tokens = rest;
        if !tokens.is_empty() && tokens[0] == "," {
            tokens = &tokens[1..];
        }
    }
    assert!(!tokens.is_empty(), "unterminated matrix row");
    (row, &tokens[1..])
}

/// Convert a natural number to Pluck's unary nat representation: 0 -> O, 3 -> S(S(S(O)))
fn nat_to_expr(n: u64, ctx: &mut ParseContext) -> PExpr {
    let o_sym = ctx.interner.intern("O");
    let s_sym = ctx.interner.intern("S");
    let mut expr = PExpr::Construct {
        constructor: o_sym,
        args: vec![],
    };
    for _ in 0..n {
        expr = PExpr::Construct {
            constructor: s_sym,
            args: vec![expr],
        };
    }
    expr
}
