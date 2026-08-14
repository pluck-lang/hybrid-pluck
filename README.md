# Hybrid Pluck

Hybrid Pluck is an extension of the Pluck probabilistic programming language to include support for continuous variables (as observations or conjugate latents).

This implementation is based on a Rust port of the original, Julia-based Pluck library. This supplement also contains a copy of `rsdd`, a Rust library for decision diagrams, which Pluck uses during knowledge compilation.

AI Disclosure: Claude Code was used to assist in porting `Pluck.jl` to Rust, and in our implementation of Hybrid Pluck. The inference algorithm that Hybrid Pluck uses and the code's architecture were primarily designed by the human authors of this repo.

## Prerequisites 

Hybrid Pluck requires `cargo` and `rustc` to build. If you do not have them installed locally, we recommend using [rustup](https://rustup.rs) to install the rust toolchain.

## Building

Once `cargo` is available on your system, Hybrid Pluck can be built by calling the following command at the root of the repo:
```
cargo build --release 
```
This will create a Hybrid Pluck binary at `./target/release/pluck`. To build with debug flags enabled, use
```
cargo build
```

## Running Hybrid Pluck Programs

Programs can be run either using `cargo run` or directly using the binary from the previous step:
```
cargo run --release examples/coinBias.pluck
./target/release/pluck examples/coinBias.pluck
```
The `PLUCK_STATS` environment variable can be set to show additional output about the program execution, such as times, BDD size, and SPN size.
```
PLUCK_STATS=1 cargo run --release examples/coinBias.pluck
```

## Documentation

Full documentation lives in [`docs/`](docs/README.md):

- [A Simple Example](docs/a_simple_example.md) — write and run a simple program with discrete and continuous variables, and interpret the output
- [Guided Tour: A Hidden Markov Model](docs/guided_tour_hmm.md) — an end to end walkthrough of a realistic model
- [Language Reference](docs/language_reference.md) — syntax, primitives, and the standard library
- [Continuous Variables](docs/continuous_variables.md) — conjugate families and continuous observations
- [Queries](docs/queries.md) — `Marginal`, `Posterior`, `PosteriorSamples`, and `Gibbs`
- [Sharp Edges](docs/sharp_edges.md) — non-termination, measure-zero observations, and performance
- [Examples](docs/examples.md) — a catalogue of the programs in `examples/`

## Example Programs

The `examples/` folder contains many examples of Hybrid Pluck programs, including those used to benchmark the performance of Hybrid Pluck against other systems. Note that certain larger variants may take a few minutes to run or have a larger memory footprint, but all example programs have been verified to run in under 10 minutes on a single CPU with 64GB of memory.

## Source Code

The source code is in `src`, organized into 4 modules:

- `discrete_factorizations` - trait for a knowledge compilation/BDD backend, plus implementations for 4 different BDD libraries.
- `inference` - exact inference machinery, including knowledge compilation, weighted model counting, and conjugate pairs.
- `language` - language frontend for syntax and parsing.
- `utils` - other shared utils.

The `inference` module contains most of the core implementation and contains the following submodules:
- `conjugate_pairs` - defines the interface for conjugate priors and likelihoods, and contains implementations for Beta/Bernoulli, Dirichlet/Categorical, Normal/Normal, Gamma/Gamma, and Gamma/Poisson pairs.
- `lazy_kc` - lazy knowledge compilation machinery for symbolic execution
- `spn` - interface and semiring operations for sum product networks. Defines *2* types of sum product networks - those representing *likelihoods* and those representing *posterior distributions*.