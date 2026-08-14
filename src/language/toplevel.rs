use std::rc::Rc;

use super::define::DefinitionRegistry;
use super::parsing::{parse_expr_inner, tokenize, ParseContext};
use super::pexpr::PExpr;
use super::types::{StringInterner, TypeRegistry};

/// A resolver closure that maps a module name/path to its source content.
/// Returns `Some(content)` if the module is found, or `None` if not.
pub type IncludeResolver<'a> = &'a dyn Fn(&str) -> Option<String>;

/// Parse a Pluck source string, applying every top-level form and returning
/// the inference queries.
///
/// Top-level forms:
/// - (define (f args...) body) — function definition
/// - (define x expr) — value definition
/// - (define-type name (Ctor1 args...) (Ctor2 args...) ...) — ADT type definition
/// - (query expr) — inference query
/// - (include "path") — file include
///
/// All forms except `query` are applied purely as **side effects during
/// parsing**: definitions land in `defs`, types in `types`, and includes are
/// resolved (their queries spliced into the result). Only `query` forms carry
/// information the caller needs afterward, so those are the only ones returned.
pub fn parse_toplevel(
    s: &str,
    interner: &mut StringInterner,
    types: &mut TypeRegistry,
    defs: &mut DefinitionRegistry,
    resolver: IncludeResolver,
) -> Vec<Query> {
    let tokens = tokenize(s);
    let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
    let mut rest: &[&str] = &token_refs;
    let mut queries = Vec::new();

    while !rest.is_empty() {
        let (new_queries, remaining) = process_toplevel_form(rest, interner, types, defs, resolver);
        queries.extend(new_queries);
        rest = remaining;
    }

    queries
}

/// A parsed `(query …)` form: the query expression plus its display string.
#[derive(Debug)]
pub struct Query {
    pub expr: Rc<PExpr>,
    pub display_str: String,
}

fn process_toplevel_form<'a>(
    tokens: &'a [&str],
    interner: &mut StringInterner,
    types: &mut TypeRegistry,
    defs: &mut DefinitionRegistry,
    resolver: IncludeResolver,
) -> (Vec<Query>, &'a [&'a str]) {
    assert!(!tokens.is_empty(), "unexpected end of input");

    if tokens[0] != "(" {
        // Bare expression: parsed for validation / token consumption; no query.
        let mut ctx = ParseContext {
            interner,
            types,
            defs,
        };
        let (_expr, rest) = parse_expr_inner(tokens, &mut ctx, &[]);
        return (Vec::new(), rest);
    }

    let keyword = tokens[1];

    if keyword == "query" {
        let (query, rest) = parse_and_process_query(tokens, interner, types, defs);
        (vec![query], rest)
    } else if keyword == "define-type" {
        let rest = parse_and_process_define_type(tokens, interner, types, defs);
        (Vec::new(), rest)
    } else if keyword == "define" {
        let inner = &tokens[2..];
        let rest = if inner[0] == "(" {
            // Function definition: (define (f args...) body)
            parse_and_process_define_function(inner, interner, types, defs)
        } else {
            // Value definition: (define x expr)
            parse_and_process_define_value(inner, interner, types, defs)
        };
        (Vec::new(), rest)
    } else if keyword == "include" {
        parse_and_process_include(tokens, interner, types, defs, resolver)
    } else {
        let mut ctx = ParseContext {
            interner,
            types,
            defs,
        };
        let (_expr, rest) = parse_expr_inner(tokens, &mut ctx, &[]);
        (Vec::new(), rest)
    }
}

fn parse_and_process_query<'a>(
    tokens: &'a [&str],
    interner: &mut StringInterner,
    types: &mut TypeRegistry,
    defs: &mut DefinitionRegistry,
) -> (Query, &'a [&'a str]) {
    let tokens = &tokens[2..];

    let end_idx = find_ending_paren(tokens);
    let query_tokens = &tokens[..end_idx];

    // Determine if the first token is a query name or an expression
    // Julia logic: if first token is "(", it's a single expression;
    // otherwise, the first token is the query name and the rest is the expression.
    let first_paren = query_tokens.iter().position(|t| *t == "(");

    let (display_str, expr_tokens) = if first_paren == Some(0) {
        (detokenize(query_tokens), query_tokens)
    } else if first_paren.is_none() && query_tokens.len() == 1 {
        (query_tokens[0].to_string(), query_tokens)
    } else if first_paren.is_some() || query_tokens.len() > 1 {
        let name = query_tokens[0].to_string();
        (name, &query_tokens[1..])
    } else {
        (detokenize(query_tokens), query_tokens)
    };

    let mut ctx = ParseContext {
        interner,
        types,
        defs,
    };
    let (expr, _) = parse_expr_inner(expr_tokens, &mut ctx, &[]);

    assert_eq!(tokens[end_idx], ")", "expected closing paren for query");

    (
        Query {
            expr: Rc::new(expr),
            display_str,
        },
        &tokens[end_idx + 1..],
    )
}

fn parse_and_process_define_type<'a>(
    tokens: &'a [&str],
    interner: &mut StringInterner,
    types: &mut TypeRegistry,
    _defs: &mut DefinitionRegistry,
) -> &'a [&'a str] {
    let mut tokens = &tokens[2..];

    let type_name = interner.intern(tokens[0]);
    tokens = &tokens[1..];

    let mut constructors = Vec::new();
    while tokens[0] != ")" {
        assert_eq!(
            tokens[0], "(",
            "expected opening paren in constructor definition"
        );
        let end_idx = tokens.iter().position(|t| *t == ")").unwrap();
        let ctor_name = interner.intern(tokens[1]);
        let args: Vec<_> = tokens[2..end_idx]
            .iter()
            .map(|s| interner.intern(s))
            .collect();
        constructors.push((ctor_name, args));
        tokens = &tokens[end_idx + 1..];
    }

    types.define_type(type_name, constructors);

    &tokens[1..]
}

fn parse_and_process_define_function<'a>(
    tokens: &'a [&str],
    interner: &mut StringInterner,
    types: &mut TypeRegistry,
    defs: &mut DefinitionRegistry,
) -> &'a [&'a str] {
    let mut tokens = &tokens[1..];
    let fname_str = tokens[0].to_string();
    let fname = interner.intern(&fname_str);

    // Pre-register a dummy definition so the function can reference itself
    defs.define(
        fname,
        Rc::new(PExpr::Construct {
            constructor: interner.intern("Unit"),
            args: vec![],
        }),
    );

    tokens = &tokens[1..];

    let mut arg_names = Vec::new();
    while tokens[0] != ")" {
        arg_names.push(tokens[0].to_string());
        tokens = &tokens[1..];
    }
    tokens = &tokens[1..];

    // Build environment for body parsing (args in scope)
    let mut env: Vec<String> = arg_names.iter().rev().cloned().collect();

    // For zero-arg functions, add a dummy "_" parameter
    if arg_names.is_empty() {
        env.insert(0, "_".to_string());
    }

    let mut ctx = ParseContext {
        interner,
        types,
        defs,
    };
    let (body, tokens) = parse_expr_inner(tokens, &mut ctx, &env);
    assert_eq!(
        tokens[0], ")",
        "expected closing paren when defining {}",
        fname_str
    );

    // Wrap body in lambda abstractions
    let mut expr = body;
    if arg_names.is_empty() {
        let underscore = ctx.interner.intern("_");
        expr = PExpr::Abs {
            var: underscore,
            body: Box::new(expr),
        };
    } else {
        for name in arg_names.iter().rev() {
            let sym = ctx.interner.intern(name);
            expr = PExpr::Abs {
                var: sym,
                body: Box::new(expr),
            };
        }
    }

    defs.define(fname, Rc::new(expr));

    &tokens[1..]
}

fn parse_and_process_define_value<'a>(
    tokens: &'a [&str],
    interner: &mut StringInterner,
    types: &mut TypeRegistry,
    defs: &mut DefinitionRegistry,
) -> &'a [&'a str] {
    let name_str = tokens[0];
    let name = interner.intern(name_str);

    defs.define(
        name,
        Rc::new(PExpr::Construct {
            constructor: interner.intern("Unit"),
            args: vec![],
        }),
    );

    let mut ctx = ParseContext {
        interner,
        types,
        defs,
    };
    let (expr, tokens) = parse_expr_inner(&tokens[1..], &mut ctx, &[]);
    assert_eq!(tokens[0], ")", "expected closing paren in define");

    defs.define(name, Rc::new(expr));

    &tokens[1..]
}

fn parse_and_process_include<'a>(
    tokens: &'a [&str],
    interner: &mut StringInterner,
    types: &mut TypeRegistry,
    defs: &mut DefinitionRegistry,
    resolver: IncludeResolver,
) -> (Vec<Query>, &'a [&'a str]) {
    let path_token = tokens[2];
    assert!(
        path_token.starts_with('"') && path_token.ends_with('"'),
        "include expects a string literal path"
    );
    let path = path_token[1..path_token.len() - 1].to_string();
    assert_eq!(tokens[3], ")", "expected closing paren in include");

    let rest = &tokens[4..];

    match resolver(&path) {
        Some(content) => {
            // Splice the included file's queries in; its defines/types are
            // applied to the shared registries as a side effect.
            let queries = parse_toplevel(&content, interner, types, defs, resolver);
            (queries, rest)
        }
        None => {
            panic!("failed to include \"{}\": module not found", path);
        }
    }
}

fn find_ending_paren(tokens: &[&str]) -> usize {
    let mut depth = 1i32;
    let mut i = 0;
    // Depth starts at 1 to account for the opening paren the caller already consumed.
    while depth > 0 && i < tokens.len() {
        if tokens[i] == "(" {
            depth += 1;
        } else if tokens[i] == ")" {
            depth -= 1;
        }
        if depth > 0 {
            i += 1;
        }
    }
    i
}

fn detokenize(tokens: &[&str]) -> String {
    let mut result = String::new();
    for (i, token) in tokens.iter().enumerate() {
        if *token == "(" || *token == ")" {
            result.push_str(token);
            if *token == ")" && i + 1 < tokens.len() && tokens[i + 1] == "(" {
                result.push(' ');
            }
        } else {
            if i > 0 && tokens[i - 1] != "(" {
                result.push(' ');
            }
            result.push_str(token);
        }
    }
    result
}
