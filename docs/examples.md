# Examples

The `examples/` directory holds 45 programs. Every one runs with

```
cargo run --release examples/<name>.pluck
```

Most finish in well under a second. A few of the larger models take minutes and a
substantial amount of memory; all of them have been verified to run in under ten minutes
on a single Intel Emerald Rapids CPU with 64 GB of RAM.

## Start here

| Example | description |
|---|---|
| `simple_flip` | The smallest possible program: `(Marginal (flip 0.3))` |
| `coin_and` | The AND of two independent flips |
| `if_then_else` | A conditional flip, where both branches give the same marginal |
| `burglary_alarm` | The classic Bayesian network: was there a burglary, given that Mary called? |
| `geometric_distribution` | Sum of two geometric variables; a marginal and a posterior query |
| `simple_example` | The program from [A Simple Example](a_simple_example.md): a flip choosing between two normals, with a Beta prior |

## Discrete models

| Example | description |
|---|---|
| `diamond_network` | A 100-node diamond network — will a message get through? Exact inference at scale |
| `sorted_list` | Generating random sorted lists of naturals and querying their properties |
| `bst` | Binary search trees; recursive algebraic datatypes |
| `strings` | Two random strings concatenated with a space — recover the originals |
| `typos` | A probabilistic string-edit model: substitutions, deletions, insertions |
| `posterior_sampling` | `PosteriorSamples` over strings and over a Bayes net |

## Conjugate basics

Good next stops after the tutorial — each has an analytically known answer stated in its
header comment.

| Example | description |
|---|---|
| `beta_bernoulli` | The canonical conjugate update |
| `coinBias` | Beta(2,5) prior, five observed flips, posterior Beta(5,7) |
| `conjugate_gaussians` | Two Gaussian observations of one latent; posterior N(1, 0.5774) |
| `bivariate_gaussian` | A correlated 2-D prior; observe one component, query the other |
| `linear_transform` | Pushing a 2-D Gaussian through a constant matrix with `@` |
| `addFun_sum` | Three standard normals summed; a small continuous benchmark |

## Regression

| Example | description |
|---|---|
| `linear_regression` | Textbook Bayesian linear regression; pure linear-Gaussian, solved exactly |
| `polyreg` | Polynomial regression with the degree drawn from a truncated geometric |
| `mv_dot_regression` | Two-feature regression using `FloatMatrix` and the stdlib `dot` |
| `outliers` | Robust regression: per-point Bernoulli outlier indicators over a planted line |
| `spike_and_slab` | George & McCulloch variable selection on the Hald cement data; latent inclusion indicators, solved exactly |

## Mixtures and clustering

| Example | description |
|---|---|
| `gaussian_mixture` | Two-component Gaussian mixture with a Beta mixing weight |
| `binary_mixture` | Currently byte-identical to `gaussian_mixture` |
| `normal_mixture` | Two-component mixture with shared means; a PSI/hybit benchmark port |
| `gaussian_mixture_gibbs` | Mixture where the *number* of components is itself latent; block Gibbs over cluster assignments |

## Sequential models

| Example | description |
|---|---|
| `run_walk_hrm` | Run/walk heart-rate HMM — the subject of the [guided tour](guided_tour_hmm.md) |
| `hmm_first_order` | K = 3 states via `categorical` with Gaussian emissions; a lesson in variable order |
| `hmm_second_order` | State depends on the previous two |
| `hmm_factorial` | Two independent hidden chains driving a single observation |
| `sir_model` | SIR epidemic HMM with four Beta parameters over 12 steps |
| `ssm` | State-space model over 2-D points |
| `changepoint_coal` | British coal-mining disaster counts, 1851–1961; Gamma-Poisson with a latent changepoint |

## Structured and applied models

| Example | description |
|---|---|
| `pcfg` | Probabilistic context-free grammar with Dirichlet priors on production probabilities |
| `f81_joint` | F81 molecular-evolution likelihood on a fixed 4-tip phylogenetic tree |
| `cube_slam` | SLAM: a robot observing an 8-vertex cube from up to 5 poses |
| `soccer` | Tournament model with Dirichlet-categorical team identities and Gamma-Poisson goals |
| `dawid_skene` | Crowd annotation: recover true labels and annotator reliability jointly |
| `king_ei` | King's beta-binomial ecological inference from aggregate margins |
| `clickGraph` | A click-graph similarity model over five trials |
| `clinicalTrial1`, `clinicalTrial2` | Is a treatment effective? Two variants of the model comparison |
| `GPA` | Two grading systems with mixed discrete/continuous GPA distributions |
| `secret_ballot` | Per-member Beta vote probabilities recovered from observed ballot majorities alone |

## Which example uses which feature

| Feature | Examples |
|---|---|
| `Gibbs` | `gaussian_mixture_gibbs` — the only example using it |
| `PosteriorSamples` | `posterior_sampling`, `bst` |
| `beta` | `coinBias`, `beta_bernoulli`, `sir_model`, `clickGraph`, `dawid_skene`, `king_ei`, and 14 more |
| `gaussian` / `normal` | `conjugate_gaussians`, `linear_regression`, `run_walk_hrm`, `hmm_first_order`, and 16 more |
| `dirichlet` | `pcfg`, `soccer`, `f81_joint`, `gaussian_mixture_gibbs` |
| `categorical` | `hmm_first_order`, `hmm_second_order`, `hmm_factorial`, `pcfg`, `soccer`, `f81_joint`, `gaussian_mixture_gibbs` |
| `gamma`, `poisson` | `changepoint_coal`, `soccer` |
| `prob_eq` | `GPA` |
| `native_leq` | `hmm_second_order` |
| `@` (matmul) | `linear_transform`, `cube_slam` |
| `dot` | `mv_dot_regression`, `spike_and_slab` |
| `uniform_int_range` | `posterior_sampling`, `strings`, `typos`, `gaussian_mixture_gibbs` |
| `mk_int` | `gaussian_mixture_gibbs` |
| multivariate `gaussian` | `bivariate_gaussian`, `linear_transform`, `cube_slam`, `mv_dot_regression` |

No example currently uses `exponential`, `dirichlet_eq`, `native_geq`, `sum`, or bare
`uniform_int`. For those, see `tests/programs/` — for instance `exponential_geq.pluck`,
`dirichlet_observation.pluck`, `poisson_geq.pluck`, and `sum_and_dot.pluck`.
