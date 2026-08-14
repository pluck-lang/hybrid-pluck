# Guided Tour: A Hidden Markov Model

This is a walk through one complete program, `examples/run_walk_hrm.pluck`. Run it with:

```
cargo run --release examples/run_walk_hrm.pluck
```

## The scenario

A person has just bought a new running watch that tracks their heart rate while exercising.
They decide to test it, and alternate between running and walking. The watch records a
measurement every 10 seconds as follows:

```
82, 81, 80, 83, 120, 124, 120, 85
```

Given this data, we might like to infer a few things:

1. What is this person's baseline (walking) heart rate?
2. When were they running?
3. How likely are they to start running, and to stop once they have?
4. How much does running raise their heart rate?

Some of these queries are about a **discrete** state - was the person running or walking?
Others are about **continuous** parameters - their baseline heart rate, the probability
of going from running to walking. Hybrid Pluck is designed to handle exactly these types
of hybrid models.

## The probabilistic model

We can answer the questions above with a probabilistic model implemented in Hybrid Pluck.
We use a hidden Markov model with unknown transition probabilities and continuous emissions.

Our model is as follows:

* At each step the person is either running or not. This is the **hidden state**.
* The state evolves step to step: if walking, they start running with probability
  `start-rate`; if running, they stop with probability `stop-rate`.
* At each step the monitor emits a heart-rate reading, drawn around a mean that depends on
  the state — a baseline when walking, and the baseline plus an increment when running.

Four quantities are unknown and shared across all eight steps: `start-rate`, `stop-rate`,
the baseline heart rate, and the increment. We put priors on them and infer the posterior
using Hybrid Pluck's exact inference.

### The parameters

```scheme
(define (gen-params)
  (let
   ((start-rate (beta 1.0 1.0))
    (stop-rate (beta 1.0 1.0))
    (base-hr (normal 70.0 20.0))
    (incr (normal 40.0 20.0)))
    (Pair (Pair start-rate stop-rate)
          (Pair base-hr (+. base-hr incr)))))
```

Both `start-rate` and `stop-rate` have *uniform* priors on the probability, written as a
`(beta 1.0 1.0)` distribution. This is an example of Beta/Bernoulli conjugacy.

The running heart rate is `(+. base-hr incr)`, not a fresh variable. This makes
the walking mean and the running mean correlated.

The function returns the four parameters nested in `Pair`s, so `(fst (snd params))` is the
baseline heart rate and `(snd (snd params))` the running one.

### The chain

```scheme
(define (hmm params state past hrs t)
  (match t
    O => (Pair (reverse hrs) (reverse past))
    S t => (let
            ((running state)
             (hr (normal (if running
                           (snd (snd params))
                           (fst (snd params))) 5.0))
             (new-running (if running
                            (not (flip (snd (fst params))))
                            (flip (fst (fst params))))))
             (hmm params
                  new-running
                  (Cons state past)
                  (Cons hr hrs)
                  t))))
```

We represent the time `t` as a Peano natural, which has a base `O` representing zero, 
and a successor `S n` representing the natural number `n + 1`. 

We recurse over the Peano natural `t`, accumulating the emitted heart rates
and the visited states, and reverse both at the base case (so that we emit lists which increase
in time from left to right instead of being backwards). 

Each step does two things:

* **Emit.** `(normal <mean> 5.0)` where the mean is chosen by an ordinary `if` on the
  boolean state. The *discrete* latent selects which *continuous* expression is emitted. 
  Every such choice doubles the number of mixture components in the final Gaussian posterior.
* **Transition.** `(flip start-rate)` when walking, `(not (flip stop-rate))` when running.
  The flip probabilities are the symbolic Beta variables, so transitioning from running to
  walking or vice versa (or remaining in the same state) updates the respective beta posterior.

### The query

```scheme
(query
 (let ((params (gen-params))
       (res (hmm params false [] [] (length obs)))
       (hrs (fst res))
       (sts (snd res)))
   (Posterior (fst (snd params)) (list=? real=? hrs obs))))
```

The chain starts in the walking state (`false`). The evidence is
`(list=? real=? hrs obs)` — the generated heart rates equal the observed ones.
Note that this is a *measure zero* event - we condition on the generated heart rates
being *exactly* equal to the observed ones. This different from the 'soft' conditioning
often used in other PPLs with statements such as `score` or `observe`. For more
discussion, see [measure-zero observations](sharp_edges.md#measure-zero-observations).

The queried expression is `(fst (snd params))`, the baseline heart rate, 
giving the following output:

```
(let((params(gen-params)) … (Posterior(fst(snd params)) (list=? real=? hrs obs))):
  x  1.0000000000000024
    ├ 100.00%
    │       x ~ N(82.0408, 2.2089)
    ├ 0.00%
    │       x ~ N(81.1877, 2.4621)
    ├ 0.00%
    │       x ~ N(81.6652, 2.4621)
    …
```

Unpacking the output:

* The line `x  1.0000000000000024` represents the *queried value*. In this case, we 
  queried a symbolic Gaussian, so its "value" is the symbolic value `x` with probability `1`.
* The lines underneath represent the mixture components of the distribution of `x`, each
  with an associated symbolic Gaussian posterior. We see that almost all the probability
  mass is on the *first* branch, but each branch corresponds to some possible run/walk history.

To answer the other questions, change what you query: `(fst (fst params))` for the
start-rate, `sts` for the state sequence, `(snd (snd params))` for the running heart rate.

## The cost of hybrid exact inference

Hidden Markov models are often used as example models for exact discrete inference
because they can be represented with a *linearly* sized structure. Each state transition
depends only on the previous one, so we can represent the full chain with a boolean 
decision diagram (BDD) of size `4n - 4` for `n` observations, even though the number
of possible run/walk state histories is `2^(n-1)`.

However, once we add Gaussian observations, we can no longer avoid the exponential. This
is because each state history contributes a *different* component to the Gaussian posterior
mixture*. Therefore, we inevitably have to construct exponentially many Gaussian mixture
components. 

This is a common issue with hybrid exact inference. Even when the discrete structure
can be represented with polynomial space, adding continuous variables often makes 
exact inference have worse polynomial degree or go exponential. For this reason
Hybrid Pluck can do exact inference for many small and moderately sized datasets, 
but will quickly time out for larger models and data.

There are two options to handle this:

* **Use [`Gibbs`](queries.md#gibbs)** over the discrete state sequence. This is precisely
  the case Gibbs exists for — too many discrete paths to enumerate, while the continuous
  parameters stay symbolic and are handled analytically.
* **Keep the chain short** enough to enumerate, as this example does.

## Things to try

Each of these is a small edit to `examples/run_walk_hrm.pluck`:

* Name the query, so the output header is readable.
* Query `(fst (fst params))` instead — the start-rate — and get a Beta posterior rather
  than a Gaussian one. That answers question 3.
* Query `sts` to answer question 2: when were they running?
* Query `(snd (snd params))` for the running heart rate, and compare it against the
  baseline to answer question 4.
* Shorten `obs` by one reading and re-run with `PLUCK_STATS=1`; watch the component count
  halve. Lengthen it and watch the blowup.
* Replace the Gaussian emission `(normal ... 5.0)` with a state-dependent coin flip,
  `(flip (if running 0.9 0.1))`, condition with `(list=? iff hrs obs)` against a list of
  booleans, and confirm under `PLUCK_STATS=1` that the BDD goes linear.
* Switch the query to `Gibbs` over `sts`, and compare the answer to the exact one.