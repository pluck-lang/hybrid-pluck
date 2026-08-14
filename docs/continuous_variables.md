# Continuous Variables

Hybrid Pluck supports a number of continuous conjugate priors and continuous observations.
This document outlines which distributions are supported, and how they can be used 
in Hybrid Pluck.

For the syntax of each primitive see the
[Language Reference](language_reference.md#primitives).

## Contents

1. [The conjugate families](#the-conjugate-families)
2. [Observing continuous values](#observing-continuous-values)
3. [What arithmetic is allowed](#what-arithmetic-is-allowed)
4. [Vectors and matrices](#vectors-and-matrices)
5. [How posteriors are named and printed](#how-posteriors-are-named-and-printed)
6. [Worked examples](#worked-examples)

## The conjugate families

Five prior families are supported, each paired with the consumers it is conjugate to.

| Prior | Written | Consumed by | Observed with |
|---|---|---|---|
| Beta | `(beta a b)` | `(flip p)` | `prob_eq` |
| Dirichlet | `(dirichlet a1 ... an)` | `(categorical v)` | `dirichlet_eq` |
| Gaussian | `(gaussian mu sigma)`, `(normal mu sigma)` | affine expressions | `real_eq` |
| Gamma | `(gamma shape rate)` | `(poisson r)`, `(exponential r)` | `real_eq`, or via its draws |
| — | `(poisson r)` draw | — | `native_eq`, `native_leq`, `native_geq` |
| — | `(exponential r)` draw | — | `real_eq`, `real_lt`, `real_geq` |

**Beta.** A Beta value is a symbolic *probability*, and its natural consumer is `flip`.
`(beta 1.0 1.0)` is the uniform prior on `[0,1]`. Conditioning on the outcomes of flips
performs the usual Beta-Bernoulli update: `Beta(a, b)` with `h` heads and `t` tails
becomes `Beta(a + h, b + t)`.

**Dirichlet.** A Dirichlet value is a symbolic probability *vector*, consumed by
`categorical`. Individual components are extracted with `vector_index`, and a component
can be used anywhere a probability can — including as the argument to `flip`.

**Gaussian.** `(gaussian mu sigma)` takes a **standard deviation**, not a variance. The
stdlib `normal` is `(+. (gaussian 0.0 sigma) mu)` — an affine shift of a zero-mean
Gaussian — and is normally what you want. Gaussians are closed under affine combination,
so sums, differences, and constant multiples of Gaussians remain exactly representable.

**Gamma.** A Gamma value is a symbolic *rate*, and its consumers are `poisson` and
`exponential`, both of which accept either a float or a Gamma in the rate position.
Observing the resulting draws updates the Gamma.

All parameters to these primitives must be floats. `(gamma 1 1)` fails with

```
Gamma: first argument must be a number (use 1.0 not 1)
```

because `1` is a Peano natural — see
[numeric representations](language_reference.md#numeric-representations).

## Observing continuous values

Continuous variables are constrained by *observation primitives*, which return booleans
and are normally used as the evidence argument of a `Posterior` query.

| Primitive | Applies to | Meaning |
|---|---|---|
| `real_eq` | Gaussian expressions, Gamma, Exponential draws, floats | exact equality |
| `real_lt`, `real_geq` | Exponential draws, floats | half-open range |
| `prob_eq` | Beta probabilities, floats | exact equality |
| `dirichlet_eq` | Dirichlet vectors, 1-D matrices | the whole vector equals a value |
| `native_eq`, `native_leq`, `native_geq` | native ints, Poisson draws | integer (in)equality |

`real=?` in the standard library is just `real_eq` exposed as an ordinary function, and is
what most example programs use.

Two restrictions worth memorising:

* **Poisson draws are integers.** `(real_eq p 1.0)` on a Poisson draw fails with
  *"real_eq: Poisson draws are integer-valued; use native_eq"*. Use `native_eq`,
  `native_leq`, `native_geq` instead.
* **Range observations are Exponential-only.** `real_lt` and `real_geq` support Exponential
  draws (and plain floats). They are exactly closed under negation, so `not (real_lt e v)`
  is `real_geq e v`.

Observing an equality between continuous quantities conditions on an event of probability
zero. Hybrid Pluck handles this correctly by disintegration, but the semantics have consequences
that are easy to get wrong; see
[Sharp Edges](sharp_edges.md#measure-zero-observations) for a worked example where the
answer is *not* what naive density-cancelling reasoning predicts.

Note that because primitives like `real_eq` return booleans, we can also use them in 
control flow `if (real_eq (normal 1.0 1.0) 1.0) ...` which will be correctly handled by
Hybrid Pluck. In practice this is rarely used, since `real_eq (normal 1.0 1.0) 1.0` represents
an event of measure zero, so it mostly used for *conditioning* in a `Posterior` query.

## What arithmetic is allowed

The engine keeps continuous values in closed form, which means arithmetic on them is
restricted to operations that preserve the family. Unsupported operations will give
an error.

### Gaussians: affine only

Gaussians are closed under affine combination. You may:

* add and subtract Gaussian expressions freely — `(+. a b)`, `(-. a b)`
* scale by a constant — `(*. a 2.0)`, `(/. a 2.0)`
* add constants — `(+. a 1.0)`

You may not multiply two Gaussians, or divide by one, as this is a non-conjugate operation:

```
*.: cannot multiply two non-trivial Gaussian terms (would be nonlinear)
```

### Gammas: scaling only

A Gamma may be multiplied or divided by a positive constant, which yields a scaled Gamma.
It may not be added to or subtracted from anything:

```
a Gamma variable supports only scaling (`*.` / `/.`); addition/subtraction of a Gamma is not supported
a Gamma variable can only be scaled by a numeric constant (use a float literal, e.g. (*. g 5.0))
scaling a Gamma must yield a positive scale, got -2 (scale 1 * -2)
```

### Draws: no arithmetic at all

Poisson and Exponential *draws* are samples, not rates, so they cannot be scaled
algebraically:

```
float arithmetic on an Exponential/Poisson draw is not supported;
scale the gamma rate instead, e.g. (exponential (*. g 5.0))
```

The error message states the fix: transform the rate, not the draw.

### Categorical: one Dirichlet, alone

`categorical` takes either a list of float weights or a *single* Dirichlet vector.

### `discrete` and `uniform` cannot take latent probabilities

`discrete` and `uniform` are parse-time macros over literal probabilities, so a Beta or
Dirichlet cannot be used with them (*"probability must be a literal number"*). A latent
probability must go through `flip` or `categorical`.

## Vectors and matrices

`FloatMatrix` literals — `{1.0, 2.0}` for a vector, `{{1.0, 2.0}, {3.0, 4.0}}` for a
matrix — hold arbitrary expressions, including symbolic Gaussians, and support:

* `(vector_index m i)` and `(vector_index m i j)`, indexing with native ints
* `(@ a b)` for matrix multiplication (rank ≤ 2)
* `(sum m)` for the sum of all entries
* elementwise `+. -. *. /.` between same-shaped matrices
* `(dot a b)`, defined in the standard library as `(sum (*. a b))`

The linearity rules above still apply elementwise, so `@` and `dot` are allowed between a
constant matrix and a Gaussian one but not between two Gaussian ones.

### Multivariate Gaussians

`(gaussian mu cov)` also accepts a 1-D `FloatMatrix` mean and a 2-D **covariance** matrix
(not standard deviations, and not a Cholesky factor). The covariance is checked at compile
time for symmetry and positive semi-definiteness, and the mean must be deterministic.

```scheme
(query joint
 (let ((x (gaussian {0.0, 0.0}
                    {{1.0, 0.5},
                     {0.5, 1.0}})))
   (Posterior (vector_index x @1)
              (real_eq (vector_index x @0) 1.0))))
```

```
joint:
  x  1
        x ~ N(0.5000, 0.8660)
```

Conditioning a correlated pair on one component shifts the other, exactly as the analytic
answer `N(0.5, 0.75)` predicts — note the printed `0.8660` is the standard deviation,
`sqrt(0.75)`.

## How posteriors are named and printed

### Naming

Continuous variables have no user-visible names, so the engine generates them positionally
from where each variable appears in the queried value:

| Family | Names |
|---|---|
| Gaussian expressions | `x, y, z, w, v, u, t, s, r`, then `x10`, ... |
| Beta and Dirichlet | `p, q, r, s, t, u, v, w`, then `p9`, ... |
| Gamma family (rates and draws) | `λ, μ, ν, κ, θ, ρ, σ, τ`, then `λ_9`, ... |

Two refinements make output more readable:

* Values in a list get **subscripts** — `x₀, x₁, x₂` — and components of a Dirichlet
  vector `p` print as `p₀, p₁, p₂`.
* If a variable sits in a constructor field whose name is short (5 characters or fewer)
  and unambiguous, that field name is used instead of a generated one. Naming your
  `define-type` fields well therefore pays off in readable output.

Names are per-query: the same underlying variable may print differently in two queries.

### Print formats

| Form | Meaning |
|---|---|
| `p ~ Beta(5.0000, 7.0000)` | Beta posterior |
| `p ~ Dirichlet(1.0000, 1.0000, 1.0000)` | Dirichlet posterior |
| `x ~ N(1.0000, 0.5774)` | Gaussian posterior — **mean and standard deviation** |
| `λ ~ Gamma(2.0000, 1.5000)` | Gamma posterior (shape, rate) |
| `λ ~ Poisson(rate_0)` | a Poisson draw, conditional on its rate variable |
| `p = 0.5000` | a variable pinned to a constant by an exact observation |
| `p = (0.2000, 0.3000, 0.5000)` | a pinned Dirichlet vector |
| `λ ~ Mix[...]` | a mixture posterior over a Gamma rate |

Several variables that ended up correlated are reported as one joint block:

```
x, y ~ Joint Gaussian:
  mean = [1.0000, 1.0000]
  cov  = [0.2500, -0.2500]
         [-0.2500, 0.2500]
```

Independent variables appear as separate entries. Parameters print to 4 decimal places by
default; set `PLUCK_DISPLAY_PRECISION` to widen them.

## Worked examples

Each of these is a real file; the output shown is what it prints.

### Beta-Bernoulli — `examples/coinBias.pluck`

A coin bias with a `Beta(2,5)` prior, observed through five flips with outcomes
`[1,1,0,1,0]`:

```scheme
(define (obs-flip p y) (iff (flip p) y))
(define (obs-all p ys)
  (match ys
    Nil => (True)
    Cons y rest => (and (obs-flip p y) (obs-all p rest))))

(query coinBias
  (let ((a (beta 2.0 5.0)))
    (Posterior a (obs-all a [true, true, false, true, false]))))
```

```
coinBias:
  p  1
        p ~ Beta(5.0000, 7.0000)
```

The textbook conjugate update: `Beta(2 + 3, 5 + 2)`.

### Gaussian-Gaussian — `examples/conjugate_gaussians.pluck`

```scheme
(query conjugate_gaussians
  (let ((a (normal 0.0 1.0)))
    (Posterior a
      (and (real=? (normal a 1.0) 1.0)
           (real=? (normal a 1.0) 2.0)))))
```

```
conjugate_gaussians:
  x  1
        x ~ N(1.0000, 0.5774)
```

Precision adds: `1 + 1 + 1 = 3`, so the posterior is `N((0+1+2)/3, sqrt(1/3))`.

### Dirichlet-Categorical — `tests/programs/dirichlet_categorical.pluck`

```scheme
(define (dirichlet-categorical)
    (let ((p (dirichlet 1.0 1.0 1.0))
          (first (categorical p))
          (second (categorical p)))
    (Posterior second (native_eq first @1))))

(query second-given-that-first-is-one (dirichlet-categorical))
```

```
second-given-that-first-is-one:
  1  0.5000000000000001
  0  0.25
  2  0.25
```

Seeing category 1 once makes it more likely to recur — the Dirichlet has been updated even
though it was never queried directly.

### Gamma-Exponential — `tests/programs/exponential_geq.pluck`

```scheme
(query (let ((r (gamma 2.0 1.0))) (Posterior r (real_geq (exponential r) 0.5))))
```

```
(let((r(gamma 2.0 1.0))) (Posterior r(real_geq(exponential r) 0.5))):
  λ  1
        λ ~ Gamma(2.0000, 1.5000)
```

A *range* observation on the draw updates the rate: `P(x >= 0.5 | λ) = e^(-0.5λ)`, so the
`Gamma(2, 1)` prior becomes `Gamma(2, 1.5)`.

### A queried but unobserved draw — `tests/programs/poisson_marginal.pluck`

```scheme
(query (Marginal (poisson 5.0)))
```

```
(Marginal(poisson 5.0)):
  λ  1
        rate_0 = 5.0000
        λ ~ Poisson(rate_0)
```

The draw is reported as its conditional law given the (pinned) rate, rather than
disappearing from the output.