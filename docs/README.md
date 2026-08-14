# Hybrid Pluck Documentation

Hybrid Pluck is a probabilistic programming language that performs *exact* inference by
knowledge compilation. Its discrete core compiles programs to binary decision diagrams;
the hybrid extension adds continuous variables — as observations or as conjugate latents —
and reports exact conjugate posteriors rather than samples.

## Quick start

```
cargo build --release
./target/release/pluck examples/coinBias.pluck
```

```
coinBias:
  p  1
        p ~ Beta(5.0000, 7.0000)
```

Then read [A Simple Example](a_simple_example.md).

## Contents

| Document | Description |
|---|---|
| [A Simple Example](a_simple_example.md) | Write and run your first program, and understand what it prints |
| [Guided Tour: A Hidden Markov Model](guided_tour_hmm.md) | An end-to-end walkthrough of a realistic model |
| [Language Reference](language_reference.md) | Syntax, primitives, and standard-library functions |
| [Continuous Variables](continuous_variables.md) | Which continuous distributions exist, how to observe them, and what arithmetic is allowed |
| [Queries](queries.md) | Differences between `Marginal`, `Posterior`, `PosteriorSamples`, and `Gibbs`, and their output |
| [Sharp Edges](sharp_edges.md) | Non-termination, measure-zero observations, and performance considerations |
| [Examples](examples.md) | An index of all the existing example programs |

## Where things live

* `examples/` — 45 runnable example programs
* `tests/programs/` — 71 small single-feature programs with expected outputs
* `stdlib.pluck` — the standard library, loaded into every program
* `src/` — the implementation; see the [root README](../README.md) for a module map
