# Language Reference

This document describes the surface syntax of Hybrid Pluck and every primitive the
language provides. For more details of the continuous distributions supported and how
they can be used, see [Continuous Variables](continuous_variables.md); for the query forms 
and how to read results see [Queries](queries.md).

## Contents

1. [Programs and top-level forms](#programs-and-top-level-forms)
2. [Expressions](#expressions)
3. [Numeric Representations](#numeric-representations)
4. [Algebraic datatypes](#algebraic-datatypes)
5. [Functions](#functions)
6. [Primitives](#primitives)
7. [The standard library](#the-standard-library)

## Programs and top-level forms

A **program** is a sequence of **top-level forms**. A top-level form is one of:

* a **value definition**, `(define <var> <expr>)`
* a **function definition**, `(define (<func-name> <arg-1> ... <arg-n>) <expr>)`
* a **type definition**,
  `(define-type <type-name> (<constructor-1> <arg> ...) ... (<constructor-m> <arg> ...))`
* a **query**, `(query <query-name> <expr>)`
* an **include**, `(include "<path>")`

Definitions and type definitions take effect as side effects of parsing, in order, so a
form may refer to anything defined above it. Only queries produce output.

`(include "path")` splices another file into the current one: the included file's
definitions and types are added to the same environment, and its queries run as if they
had been written at the point of inclusion. Paths are resolved against the embedding
application's file registry first, then the filesystem.

A bare expression at top level is parsed and discarded. This is occasionally useful for
checking that an expression parses, but it computes nothing.

### Query names

If the first token after `query` is `(`, the whole expression text becomes the query's
display name. Otherwise the first token is taken as the name and the rest is the query
expression. So

```scheme
(query my-query (Marginal (flip 0.3)))
```

prints under `my-query:`, while

```scheme
(query (Marginal (flip 0.3)))
```

prints under `(Marginal(flip 0.3)):`.

## Expressions

An **expression** is one of:

* a **variable**, `<var>`
* a **let-expression**. Three equivalent binding-list forms are accepted:
  ```scheme
  (let ((x <expr>) (y <expr>)) <body>)   ; parenthesized pairs
  (let (x <expr> y <expr>) <body>)       ; flat
  (let [x <expr> y <expr>] <body>)       ; bracketed
  ```
  Bindings are sequential: each one may refer to those before it.
* an **if-then-else expression**, `(if <expr> <expr> <expr>)`
* a **match expression**:
  ```scheme
  (match <expr>
    <constructor-1> <var-1-1> ... <var-1-n> => <expr-1>
    ...
    <constructor-m> <var-m-1> ... <var-m-k> => <expr-m>)
  ```
  `case` is a synonym for `match`, and an optional `of` may follow the scrutinee
  (`(case e of C => ...)`).
* an **anonymous function**, `(lambda <arg-1> ... <arg-n> -> <expr>)`. The keywords
  `lambda`, `lam`, `fn`, and `λ` are all equivalent; the standard library uses `fn`.
* a **constructor expression**, `(<constructor> <expr-1> ... <expr-n>)`
* an **application**, `(<expr> <expr-1> ... <expr-n>)`
* a **fixpoint**, `(Y <expr>)` or `(Y <expr> <arg>)`
* a **constant** (see [below](#numeric-representations))
* a **primitive** application (see [Primitives](#primitives))
* a **random primitive**:
  - `(flip <expr>)`
  - `(uniform <expr-1> ... <expr-n>)`
  - `(discrete (<expr-1> <const-1>) ... (<expr-n> <const-n>))`, where the constants are
    floating-point *literals* summing to 1

Comments start with `;` and run to the end of the line.

Pattern guards are written bare — `Cons x xs => ...`. Wrapping a guard in parentheses is
an error.

Expressions used inside a `(query ...)` form must evaluate to a value of type `query`;
see [Queries](queries.md).

## Numeric Representations

Hybrid Pluck has four distinct numeric representations. Mixing them up is a
common source of confusing errors.

| Syntax | Produces | Operations |
|---|---|---|
| `2` | Peano natural, i.e. `(S (S (O)))` | `+ - * == < > >= <=`, `nat=?` (stdlib) |
| `2.0`, `-2.0` | float | `+. -. *. /.` |
| `@2` | native int | `native_eq`, `native_leq`, `native_geq` |
| `(mk_int @8 @2)` | 8-bit integer distribution | `int_eq`, `int_add`, `int_sub`, `int_lt` |

Naturals are *unary*, so `100` expands to a hundred nested `S` constructors — cheap to
pattern-match, expensive to count with. Native ints are what `categorical` returns.
Integer distributions are bit-vectors of BDD variables and are what `uniform_int`,
`uniform_int_range`, and string characters produce.

A frequent first error is passing a natural where a float is wanted:
`(gamma 1 1)` fails with *"Gamma: first argument must be a number (use 1.0 not 1)"*.

Other literals:

| Syntax | Produces |
|---|---|
| `[a, b, c]` | a `Cons`/`Nil` list |
| `{a, b, c}` | a 1-D `FloatMatrix` (a vector) |
| `{{a, b}, {c, d}}` | a 2-D `FloatMatrix` |
| `"cat"` | a list of 8-bit integer distributions — a string *is* a list of characters |
| `0cA` | a single 8-bit character literal |
| `'name` | a symbol |
| `true`, `false` | `(True)`, `(False)` |
| `nothing` | `(Unit)` |
| `$x` | `x`, forced to be read as a variable |

Matrix literals may not mix nested rows and scalar entries at the top level, and rank is
limited to 2. Commas are optional everywhere they appear above.

The symbols `?name` and `#type` are reserved for program synthesis and have no meaning in
ordinary programs.

## Algebraic datatypes

New (recursive) algebraic data types are defined with
`(define-type <type-name> (<constructor> <arg-type>...) ...)`. Because the implementation
is untyped, the `<arg-type>`s may be any symbol and serve to convey intent — but they are
not entirely cosmetic: a field's name is used when auto-naming continuous variables in
query output (see [Continuous Variables](continuous_variables.md#naming)), so short,
distinct field names produce more readable posteriors. The *arity* of each constructor is
determined by the number of listed arguments and is checked at parse time.

Four types are built into the implementation:

```scheme
;; Boolean type
(define-type bool (True) (False))
;; Natural numbers -- either zero (O) or successor (S nat)
(define-type nat (O) (S nat))
;; List type
(define-type list (Nil) (Cons head tail))
;; Unit type
(define-type unit (Unit))
```

and four more are defined in `stdlib.pluck`:

```scheme
;; Tuple type
(define-type pair (Pair any any))

;; The type of queries -- see docs/queries.md
(define-type query
  (Marginal any)
  (Posterior any bool)
  (PosteriorSamples any bool int)
  (Gibbs any bool int any Initialization))

;; How a Gibbs chain's first block is initialized
(define-type Initialization
  (WithPriorSample any)
  (WithConstant any))

;; Fixed-width booleans-as-integers
(define-type int1 (Int1 bool))
;; ... also int2, int3, int4
```

To construct a value, use `(<constructor> <arg-1> ... <arg-n>)`. A tuple is `(Pair e1 e2)`
and a list is `(Cons 3 (Cons 2 (Nil)))` or, equivalently, `[3, 2]`. Natural-number
literals are parsed as repeated applications of the `nat` constructors.

Pattern matching uses

```scheme
(match <scrutinee>
  <constructor1> <var1-1> ... <var1-n> => <expr1>
  <constructor2> <var2-1> ... <var2-m> => <expr2>
  ...)
```

For example, the standard library implements several functions over the built-in types:

```scheme
;; Extract first element from a pair
(define (fst p)
  (match p
    Pair x _ => x))

;; Map a function over a list
(define (map f xs)
  (match xs
    Nil => (Nil)
    Cons x xs => (Cons (f x) (map f xs))))
```

Every pattern is a single constructor followed by one variable per argument. Nested
patterns are **not** supported — you cannot match against `(Cons x (Nil))` — and there is
no `else` pattern.

> **A match that has no branch for the scrutinee's constructor does not raise an error.**
> It silently contributes zero probability. This is deliberate — it is how conditioning is
> implemented — but it means an accidentally non-exhaustive match quietly deletes
> probability mass. If a query's numbers look wrong, suspect a missing branch. The `error`
> primitive behaves the same way.

Several built-in functions compare values for equality:

| Function | Compares |
|---|---|
| `constructor=?` | whether two ADT values have the same constructor |
| `=?` | two ADT values structurally (recursive) |
| `==`, `nat=?` | two naturals |
| `real=?` | a real-valued expression against a constant |
| `char=?` | two characters |
| `string=?` | two strings |
| `list=?` | two lists, given an element predicate — e.g. `(list=? ==)` for lists of naturals |

## Functions

`(define (<func-name> <arg-1> ... <arg-n>) <body>)` defines a function. Anonymous
functions use `lambda`/`fn`: `(fn <arg-1> ... <arg-n> -> <body>)`. All multi-argument
functions are automatically curried, so `(f x y z)` is sugar for `(((f x) y) z)`.

As further sugar, `(define (<fname>) <body>)`, `(fn -> <body>)`, and `(<expr>)` are
shorthand for `(define (<fname> _) <body>)`, `(fn _ -> <body>)`, and `(<expr> (Unit))`
respectively.

Functions defined with `define` may use any previously defined name in their body,
including the name being defined, so recursion works directly. Anonymous recursive
functions can be written with the built-in Y-combinator,
`(Y (fn <fname> <arg-1> ... <arg-n> -> <body>))`. There is no special syntax for mutual
recursion, but it can be encoded with `Pair`:

```scheme
(define even-and-odd
  (let ((even (fst even-and-odd))
        (odd  (snd even-and-odd)))
    (Pair
      (fn n ->
        (match n
          O => (True)
          S m => (odd m)))
      (fn n ->
        (match n
          O => (False)
          S m => (even m))))))
(define even (fst even-and-odd))
(define odd (snd even-and-odd))
```

In general `define` is for *function* definitions and deterministic constants. Evaluating
a definition should not require making random choices — do **not** write
`(define b (flip 0.5))`. Random choices are tracked when they happen during the
evaluation of a query, either because the query expression itself makes them or because
it calls functions that do.

### Shadowing

A `define` of a primitive's name shadows the primitive. The standard library relies on
this to expose primitives as ordinary functions:

```scheme
(define (real=? x y) (real_eq x y))
(define (constructor=? a b) (constructors_equal a b))
```

The same applies to constructors: a `define-type` can silently redefine `S`, `Y`, `Cons`,
or even `Marginal`. See [Sharp Edges](sharp_edges.md#shadowing-built-in-names).

## Primitives

Primitives are applied like functions but are recognised by the parser and checked for
arity. Names marked *n+* take at least that many arguments.

### Discrete sampling

| Primitive | Arity | Meaning |
|---|---|---|
| `(flip p)` | 1 | `(True)` with probability `p`, `(False)` otherwise. `p` may be a float or a Beta variable. |
| `(categorical p1 ... pn)` | 1+ | A native int in `0..n-1` with the given probabilities; or, with a single argument, a Dirichlet vector. |
| `(uniform_int w)` | 1 | A uniform `w`-bit integer distribution. |
| `(uniform_int_range w lo hi)` | 3 | A uniform `w`-bit integer distribution on `[lo, hi]`. |
| `(mk_int w v)` | 2 | The constant `v` as a `w`-bit integer distribution. |

`(uniform e1 ... en)` and `(discrete (e1 p1) ... (en pn))` are macros the parser expands
into decision trees of `flip`s. In `flip` the probability may be computed at runtime, but
in `discrete` the probabilities must be literal numbers; `(discrete (x p) ...)` with a
variable `p` fails with *"probability must be a literal number"*. Write
`(if b (discrete (x 0.2) (y 0.8)) (discrete (x 0.7) (y 0.3)))` rather than trying to make
the weights dynamic.

The geometric distribution is defined in the standard library in terms of `flip`, and so
does accept a dynamic argument:

```scheme
(define (geom p)
  (if (flip p)
    (O)
    (S (geom p))))
```

### Continuous sampling

| Primitive | Arity | Meaning |
|---|---|---|
| `(beta a b)` | 2 | A Beta-distributed probability. `(beta 1.0 1.0)` is uniform on [0,1]. |
| `(dirichlet a1 ... an)` | 1+ | A Dirichlet-distributed probability vector. |
| `(gaussian mu sigma)` | 2 | A Gaussian variable; also multivariate — see below. |
| `(gamma shape rate)` | 2 | A Gamma-distributed rate. |
| `(poisson rate)` | 1 | A Poisson draw; `rate` may be a float or a Gamma variable. |
| `(exponential rate)` | 1 | An Exponential draw; `rate` may be a float or a Gamma variable. |

The stdlib function `(normal mu sigma)` is defined as `(+. (gaussian 0.0 sigma) mu)` and
is what you normally want. Note the second argument is a **standard deviation**.

All parameters must be floats (`1.0`, not `1`). See
[Continuous Variables](continuous_variables.md) for what may be done with the resulting
values.

### Observation and comparison

| Primitive | Arity | Meaning |
|---|---|---|
| `(real_eq e v)` | 2 | `e = v` for a Gaussian expression, Gamma, Exponential draw, or float |
| `(real_lt e v)` | 2 | `e < v`, for Exponential draws or floats |
| `(real_geq e v)` | 2 | `e >= v`, for Exponential draws or floats |
| `(prob_eq p v)` | 2 | `p = v` for a Beta probability or float |
| `(dirichlet_eq d v)` | 2 | A Dirichlet vector equals the 1-D matrix `v` |
| `(native_eq a b)` | 2 | Equality of native ints, symbols, or constructors |
| `(native_leq a b)`, `(native_geq a b)` | 2 | Comparison of native ints / Poisson draws |
| `(int_eq a b)` / `(int_dist_eq a b)` | 2 | Equality of integer distributions |
| `(int_lt a b)` | 2 | Less-than for integer distributions |
| `(constructors_equal a b)` | 2 | Whether two values share a constructor |

### Arithmetic

| Primitive | Arity | Meaning |
|---|---|---|
| `(+. a b)`, `(-. a b)`, `(*. a b)`, `(/. a b)` | 2 | Float arithmetic; also elementwise on matrices |
| `(int_add a b)`, `(int_sub a b)` | 2 | Wrapping arithmetic on integer distributions |

Float arithmetic also applies to symbolic continuous values, subject to the linearity
restrictions described in [Continuous Variables](continuous_variables.md#what-arithmetic-is-allowed).

### Vectors and matrices

| Primitive | Arity | Meaning |
|---|---|---|
| `(vector_index v i [j])` | 2+ | Index a `FloatMatrix` or Dirichlet vector with native ints |
| `(@ a b)` | 2 | Matrix multiplication (rank ≤ 2) |
| `(sum m)` | 1 | Sum of all entries of a `FloatMatrix` |

> **`vector_index` is not `index`.** `vector_index` is this primitive, indexing matrices
> and Dirichlet vectors with native ints (`@0`, `@1`, ...). `index` is the standard-library
> accessor for `Cons` lists, taking a natural.

### Reflection and other

| Primitive | Arity | Meaning |
|---|---|---|
| `(get_constructor e)` | 1 | The constructor of a value, as a symbol |
| `(get_args e)` | 1 | A value's constructor arguments, as a list |
| `(print e)` | 1 | Returns `e`; a no-op during inference |
| `(error msg)` | 1 | Contributes zero probability |

## The standard library

`stdlib.pluck`, at the repository root, is loaded into every program. Its contents:

**Booleans** — `and`, `or`, `not`, `iff`

**Natural-number arithmetic** — `inc`, `dec`, `+`, `-`, `*`, `mod`, `iseven`, `==`,
`nat=?`, `<`, `>`, `<=`, `>=`, `float-of-nat`

**Lists** — `car`, `cdr`, `cdr_safe`, `isempty`, `length`, `index`, `range`, `append`,
`append_one`, `take`, `map`, `mapi`, `filter`, `filteri`, `fold`, `all`, `any`,
`zip_with`, `list=?`, `head-or`

**Pairs** — `Pair`, `fst`, `snd`

**Equality** — `=?`, `constructor=?`, `constructors_equal`, `char=?`, `string=?`, `real=?`,
and `int1=?` through `int4=?` for the fixed-width boolean-vector types

**Distributions** — `geom`, `randnat`, `randnatlist`, `normal`, `make-uniform`,
`normalize-seq`, `sample-seq`

**Matrices** — `dot` (defined as `(sum (*. a b))`)

**Gibbs helpers** — `block-gibbs`, `subsets-n-minus-1` (see [Queries](queries.md#gibbs))
