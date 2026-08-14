# Sharp Edges

Things that surprise people. Roughly in order of how likely you are to hit them.

## Contents

1. [Termination and non-termination](#termination-and-non-termination)
2. [Measure-zero observations](#measure-zero-observations)
3. [Performance considerations](#performance-considerations)
4. [Shadowing built-in names](#shadowing-built-in-names)

## Termination and non-termination

Executing a query may not terminate. The per-query requirements are given in
[Queries](queries.md#termination); this section is about what it looks like when things go
wrong and how to fix them.

Because the requirement is stated in terms of *lazy* evaluation, a non-terminating query
can often be rewritten into a terminating one without changing the distribution it
denotes. The classic case is the order of a conjunction:

```scheme
(define (generate-list)
  (if (flip 0.5) (Nil) (Cons (flip 0.3) (generate-list))))
(define (all-true l)
  (match l
    Nil => (True)
    Cons x xs => (and x (all-true xs))))

;; Terminates:
(query good-query
  (let ((xs (generate-list)))
    (Marginal (and (< (length xs) 10) (all-true xs)))))

;; Does NOT terminate:
(query bad-query
  (let ((xs (generate-list)))
    (Marginal (and (all-true xs) (< (length xs) 10)))))
```

`good-query` returns

```
good-query:
  True  0.5882352907255859
  False  0.41176470927441405
```

`bad-query` dies with

```
thread 'main' has overflowed its stack
fatal runtime error: stack overflow, aborting
```

The two expressions are logically equivalent. The difference is that `<` short-circuits:
`(< (length xs) 10)` returns `(False)` as soon as ten elements exist, no matter how long
the list would have become, so it is lazily surely terminating. `(all-true xs)` is not —
there is a seed on which `generate-list` never stops — so evaluating it first forces an
unbounded list.

**Rule of thumb:** put the condition that bounds the data *first*.

### Stack overflows

As above, non-termination usually surfaces as a stack overflow rather than a hang, because
the compiler recurses in Rust. Deeply recursive but perfectly terminating programs can hit
the same limit: the integration-test harness runs every example on a 1 GB stack for
exactly this reason. If a legitimate program overflows, run it under a thread with a
larger stack, or restructure the recursion.

Ctrl-C interrupts a running query.

## Measure-zero observations

Conditioning on an equality between continuous quantities — `(real=? (normal a 1.0) 1.0)`
— conditions on an event of probability zero. Hybrid Pluck handles this with
*epsilon-indexed* semantics: an expression such as `(real=? x v)` is interpreted as the
limit of the likelihood of the positive-probability event `|x - v| < ε/2` as the tolerance
`ε` goes to 0.

### A surprising example

Running the following program
```scheme
;; x and y are independent standard normals.
;; Observe x = 1. Then a coin picks one of them, and we observe that it is 1 too.
(query measure-zero
  (let ((x (normal 0.0 1.0))
        (y (normal 0.0 1.0))
        (b (flip 0.5))
        (z (if b x y)))
    (Posterior b (and (real=? x 1.0) (real=? z 1.0)))))
```
gives output:
```
measure-zero:
  True  1
```
i.e. the coin **must** have picked `x`.

There are two possible execution paths we can take through the program - one where the coin
`b` picks `x`, and the other where it picks `y`. In the first case, we observe `x = 1` twice,
which equivalent to observing it once. In the second, we observe *both* `x = 1` and `y = 1`. 

Intuitively, in the second case we are making *two* measure-zero observations, which is
infinitely less likely than making *one*, so the coin must have picked `x`. More precisely,
after conditioning on `x = 1` we are evaluating events under the law `P( · | x = 1)`. Under
that law the event `y = 1` still has probability zero, while the event `x = 1` has
probability one — giving the same answer.

### Tolerances

* `prob_eq` compares floats within a fixed `1e-15` window.
* `dirichlet_eq` pins the entire vector, and checks its length against the Dirichlet's
  category count.
* `PLUCK_EQ_TOL` controls the tolerance of the equality gate applied to *sampled*
  continuous assignments. In ill-conditioned models — mixtures with a wide range of
  variances, say — a reconstructed assignment can land just off the constraint manifold
  and zero out the only live discrete path, which surfaces as a Gibbs chain stalling.
  Raising the tolerance is the documented remedy.

## Performance considerations

Beyond outright non-termination, it is easy to write queries that are merely very slow.
Different parts of inference may be slow for different models.
`PLUCK_STATS=1` gives a breakdown by time: it reports `symbolic-exec`, `wmc`, and `posterior` times
separately, along with BDD and mixture sizes, which can be helpful in debugging slow queries.

### 1. Laziness

Pluck uses laziness to prune the space of executions it must explore. Write query
expressions and conditions so that a lazy interpreter could answer them after generating
only a small part of the data structure involved, and write the models so they generate
data lazily — computing the first element of a list before the second. This is the same
property that governs [termination](#termination-and-non-termination).

### 2. Variable order

Inference builds binary decision diagrams over the program's random choices, and BDD size
is extremely sensitive to the order of those variables. The order is the order in which a
*strict* interpreter would encounter the choices. So in
`(let ((x (flip 0.3)) (y (flip 0.4))) (or y x))`, `x` precedes `y` even though lazy
evaluation looks at `y` first — and reordering `let` bindings changes the order.

The rule of thumb: **generate and condition in an interleaved fashion**, so that each new
variable depends only on recently-created ones and never has to "reach back" past the
whole history.

`examples/hmm_first_order.pluck` conditions each emission immediately after the transition
that produced it. Rewriting it to build the whole chain and compare at the end
(`(list=? real=? (chain ...) obs)`) changes nothing about the model — same 77 random
choices, same 1023 posterior components — but it changes the BDD dramatically:

| | BDD nodes | total time |
|---|---|---|
| interleaved | 77 | ~0.11 s |
| compared at the end | 49,205 | ~0.26 s |

For a chain of length `n` the two forms cost `9n - 4` and `(5/2)(3^n - 1)` nodes
respectively — linear against exponential. The batched form forces every observation
variable to sort after every transition flip, so the BDD has to remember the entire state
path instead of just the current state.

### 3. Repeated sub-expressions

`(+ (f x) (f x))` is not equivalent to `(let ((y (f x))) (+ y y))`: if `f` makes random
choices, the first performs them twice, independently. But when a shared sub-expression
appears in *both branches* of a conditional — `(if (flip 0.4) (+ 3 (f x)) (* 2 (f x)))` —
binding it once is worthwhile: `(let ((y (f x))) (if (flip 0.4) (+ 3 y) (* 2 y)))` denotes
the same distribution but creates a single thunk instead of two.

## Shadowing built-in names

A `define-type` may silently redefine an existing constructor. Consider a grammar:

```scheme
(define-type non-terminal (S) (V) (N) (Adj) (NP) (VP))
```

`S` is already the successor constructor for naturals. Pluck accepts the redefinition, and
every built-in function that relies on naturals then breaks at runtime. Names to watch:
`S`, `O`, `Y`, `Cons`, `Nil`, `Pair`, `True`, `False`, `Unit`, and the query constructors
`Marginal`, `Posterior`, `PosteriorSamples`, `Gibbs`, `WithConstant`, `WithPriorSample` —
all of these are ordinary definitions, not reserved words.

A `define` of a *primitive's* name shadows the primitive too. This is used deliberately by
the standard library (`real=?`, `constructor=?`), but it also means defining a function
called `sum` or `index` silently replaces the built-in one.
