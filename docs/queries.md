# Queries

A query is a top-level form, `(query <name> <expr>)`, whose expression evaluates to a
value of the `query` type. There are four kinds.

## Contents

1. [The four query forms](#the-four-query-forms)
2. [Termination](#termination)
3. [Gibbs](#gibbs)
4. [Failure modes](#failure-modes)
5. [Reading the results](#reading-the-results)

## The four query forms

The `query` type is an ordinary algebraic datatype, defined in `stdlib.pluck`:

```scheme
(define-type query
  (Marginal any)
  (Posterior any bool)
  (PosteriorSamples any bool int)
  (Gibbs any bool int any Initialization))
```

| Form | Computes |
|---|---|
| `(Marginal e)` | the full marginal distribution over the values `e` can take |
| `(Posterior e cond)` | the distribution over `e` given that `cond` is true |
| `(PosteriorSamples e cond n)` | `n` exact samples from that posterior |
| `(Gibbs e cond n blocks init)` | `n` samples from a Gibbs chain |

`Marginal` and `Posterior` return exact distributions; `PosteriorSamples` and `Gibbs`
return a list of sampled values.

A query typically looks like this, binding the model's variables in a `let` so both the
queried expression and the evidence can refer to them:

```scheme
(query my-query
  (let ((x <expr>)
        (y <expr>))
    (Posterior
      <expression-involving-x-and-y>
      <condition-involving-x-and-y>)))
```

## Termination

Running a query may not terminate. The requirements differ by query kind, but all rest on
*lazy sure termination*.

Intuitively, an expression `e` lazily surely terminates if fully forcing its value — as
`print` would in a lazy language — halts for **every** possible random seed. Lazy
evaluation is what makes this weaker than it sounds: `(Cons (flip 0.5) (Nil))` is already
in weak-head normal form and needs no coin flip at all to reach it.

`(geom 0.5)` is *not* lazily surely terminating: there is a seed on which every flip comes
up tails and it never halts. But `(< (geom 0.5) 10)` is, because lazy evaluation examines
at most the first 11 flips before deciding.

| Query | Terminates if |
|---|---|
| `(Marginal e)` | `e` lazily surely terminates |
| `(Posterior e c)` | `(if c e (Unit))` lazily surely terminates |
| `(PosteriorSamples e c n)` | `c` lazily surely terminates and `(if c e (Unit))` terminates with probability 1 |
| `(Gibbs e c n bs init)` | as `PosteriorSamples`, for the query, evidence, and every block |

The `Posterior` rule is what makes conditioning useful:
`(let (n (geom 0.5)) (Posterior n (< n 10)))` terminates even though
`(Posterior (geom 0.5) (True))` would not — the condition bounds how much of `n` is ever
forced.

The `PosteriorSamples` rule is weaker still: `(PosteriorSamples (geom 0.5) (True) 20)`
does terminate, because `(geom 0.5)` halts for all but a probability-zero set of seeds.

Two practical notes:

* Ctrl-C interrupts a running query.
* Deeply recursive programs can exhaust the stack: the compiler recurses in Rust, and the
  test harness explicitly runs examples on a 1 GB stack. See
  [Sharp Edges](sharp_edges.md#termination-and-non-termination).

## Gibbs

`(Gibbs query cond num_samples blocks initialization)` runs a Gibbs chain. It exists for
models whose *discrete* structure is too large to enumerate exactly.

**Arguments.**

* `query` — the expression sampled and returned at each step.
* `cond` — the evidence, held fixed for the whole chain.
* `num_samples` — the number of steps; each step emits exactly one sample.
* `blocks` — a list of expressions, cycled one per step. Each step pins the
  previously-sampled block and resamples the next.
* `initialization` — either `(WithConstant e)` or `(WithPriorSample e)`.

**Initialization.** `WithConstant` pins the first block to the value of `e` (any
randomness inside `e` is sampled once); the value must be consistent with the evidence, or
the chain reports

```
Gibbs: WithConstant value is inconsistent with the evidence (zero probability).
```

`WithPriorSample` draws `e` from its prior and samples the first block given that draw,
retrying up to 10 times before concluding the initialization expression has no overlap
with the evidence. The seed contributes no sample of its own; all `num_samples` come from
the loop.

**Block helpers.** The standard library provides `block-gibbs`, which turns a list of
variables into leave-one-out blocks — the usual "resample one variable given all the
others" scheme.

**Stalling.** If the evidence conjoined with the current pin becomes unsatisfiable, the
chain stops early and returns what it has:

```
Gibbs: chain stalled after N samples (cond ∧ pin unsatisfiable)
```

If this happens in a model with continuous variables and no genuine zero-probability
event, the cause may be numerical — see `PLUCK_EQ_TOL` in
[Sharp Edges](sharp_edges.md#measure-zero-observations).

**Progress.** While a chain runs, the CLI prints `gibbs: count / total` to stderr.

### Block on discrete variables, not continuous ones

Each Gibbs step pins the current block to its previous value and compiles that pin into a
conditioning BDD. Pinning a **discrete** variable genuinely constrains the problem: it
rules out execution paths, so what remains to be solved is strictly smaller. Pinning a
**continuous** variable does not. Continuous variables are already carried symbolically
and integrated analytically, so fixing one to a number removes no discrete search space —
it buys nothing over leaving it symbolic, and costs a sampling step.

**So: put discrete latents in the blocks, and marginalize continuous ones or make them the
query.** Gibbs is not a way to handle continuous variables; it is a way to avoid
enumerating too many discrete paths.

`examples/gaussian_mixture_gibbs.pluck` is exactly this shape. The blocks are the discrete
cluster assignments; the Gaussian means and Dirichlet weights are never blocked and stay
symbolic:

```scheme
(Gibbs
  n                              ; query — return cluster count per step
  (list=? real=? data obs)       ; condition on observed values
  1000                           ; num MCMC steps (each emits one query sample)
  (block-gibbs cs)               ; Do block gibbs on cluster assignments
  (WithPriorSample cs))          ; seed the chain with random cluster assignments
```

## Failure modes

**The query must resolve to a single query constructor.** Randomness *inside* the
arguments is fine — the compiler merges same-constructor worlds, so

```scheme
(query c (if (flip 0.5) (Marginal (flip 0.2)) (Marginal (flip 0.4))))
```

runs and correctly reports `True` with probability `0.5·0.2 + 0.5·0.4 = 0.3`. But making
the *choice of query kind* random does not work:

```scheme
(query b (if (flip 0.5) (Marginal (flip 0.3)) (Posterior (flip 0.3) (True))))
```

```
Query must evaluate to a single (Marginal …) / (Posterior …) / (PosteriorSamples …) value, got 2 worlds
```

**The expression must be a query value.** `(query bad2 (flip 0.5))` fails the same way,
since a `flip` yields two worlds; an expression that yields one non-query value instead
reports *"Expected Marginal / Posterior / PosteriorSamples query, got: ..."*.

**Constructor arity is checked at parse time.** `(Gibbs x (True) 5)` fails with
*"wrong number of arguments for constructor Gibbs. Expected 5, got 3"*.

**An impossible condition yields nothing.** If the evidence has probability zero, the
normalising constant is zero and the query prints an empty result.

## Reading the results

### Distributions

`Marginal` and `Posterior` print one line per distinct value, sorted by probability
descending:

```
burglary-given-called:
  False  0.9970065507586945
  True  0.002993449241305545
```

Values that are, or contain, continuous variables also print their posteriors indented
beneath:

```
coinBias:
  p  1
        p ~ Beta(5.0000, 7.0000)
```

When the *same* displayed value is reachable along several discrete paths whose continuous
posteriors differ, the components are shown as a tree, each with its share of that value's
probability:

```
posterior-given-equal-to-one:
  p  0.9999999999999999
    ├ 57.89%
    │       p ~ Beta(1.0000, 2.0000)
    └ 42.11%
            p ~ Beta(2.0000, 1.0000)
```

Read this as: the answer is a two-component mixture, `Beta(1,2)` with weight 57.89% and
`Beta(2,1)` with weight 42.11%.

Only the top 10 values are printed in full. Beyond that, components with probability below
`1e-6` are collapsed into a summary line:

```
  ... and N more components (combined prob: ...)
```

### Samples

`PosteriorSamples` and `Gibbs` print one value per line, with no probabilities:

```
what-was-original:
  "bat"
  "bat"
  ...
```

These vary from run to run.

### Values

Values print in a readable form rather than as raw constructors:

| Value | Prints as |
|---|---|
| list | `[1, 2, 3]` |
| pair | `(a, b)` |
| natural, native int, float | `3`, `0`, `2.5` — plain decimals |
| string (list of 8-bit characters) | `"cat"` |
| printable 8-bit character | `'A'` |
| other constant integer distribution | `@200` |
| non-constant integer distribution | `IntDist{width=8}` |
| matrix | `{1, 2, 3}` or `{{1, 2}, {3, 4}}` |
| nullary constructor | `True`, `Nil` |
| other constructor | `(Cons 1 xs)` |

For what the probability next to a continuous-valued display actually means, see
[Sharp Edges](sharp_edges.md#measure-zero-observations).
