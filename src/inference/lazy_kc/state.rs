use std::collections::{BTreeSet, HashMap, HashSet};

use ordered_float::NotNan;

use crate::discrete_factorizations::{
    BddNode, BooleanFactorization, BooleanFunction, BooleanFunctionOps, Factorizer, VarId,
    WeightMap, Wmc,
};
use crate::inference::conjugate_pairs::exponential::ExponentialPrior;
use crate::inference::conjugate_pairs::poisson::PoissonPrior;
use crate::inference::conjugate_pairs::{
    Assignment, BetaFlip, ContVarName, DirichletFlip, GammaDrawFamily, GammaFlip, GaussianFlip,
    PriorRegistry, RegistryInsert,
};
use crate::inference::spn::evidence::EvidenceLeaf;
use crate::inference::spn::Spn;
use crate::language::values::PluckVal;
use crate::utils::math::logsumexp;

/// Type alias for a "world": a (value, BDD guard) pair.
pub type World = (PluckVal, BooleanFunction);

/// A list of worlds together with a "used information" BDD.
/// The used_information tracks what information was used to compute these worlds,
/// enabling cache validity checking for thunks.
pub type GuardedWorlds = (Vec<World>, BooleanFunction);

/// Configuration for the lazy knowledge compilation engine.
#[derive(Clone, Default)]
pub struct LazyKCConfig {
    pub max_depth: Option<usize>,
    pub time_limit: Option<f64>,
}

/// Statistics collected during compilation.
#[derive(Default)]
pub struct LazyKCStats {
    pub num_compile_calls: usize,
    pub num_cache_hits: usize,
    pub num_cache_misses: usize,
    pub num_thunk_first_eval: usize,
    pub num_thunk_widen: usize,
    pub num_tu_eval: usize,
    pub num_false_pc_exits: usize,
}

/// Position-dependent random value for Zobrist hashing.
/// Deterministic function of (depth, value) — no stored table needed.
fn zobrist(depth: usize, val: i32) -> u64 {
    let mut h = (depth as u64).wrapping_mul(0x9E3779B97F4A7C15);
    h ^= (val as u64).wrapping_mul(0x517CC1B727220A95);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51AFD7ED558CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CEB9FE1A85EC53);
    h ^= h >> 33;
    h
}

/// Incrementally-hashed callstack. The hash is maintained in O(1) per push/pop
/// via Zobrist hashing, eliminating the need to clone the Vec for HashMap lookups.
pub struct Callstack {
    stack: Vec<i32>,
    hash: u64,
}

impl Default for Callstack {
    fn default() -> Self {
        Self::new()
    }
}

impl Callstack {
    pub fn new() -> Self {
        Callstack {
            stack: Vec::new(),
            hash: 0,
        }
    }

    pub fn push(&mut self, val: i32) {
        let depth = self.stack.len();
        self.hash ^= zobrist(depth, val);
        self.stack.push(val);
    }

    pub fn pop(&mut self) -> i32 {
        let val = self.stack.pop().expect("pop from empty callstack");
        self.hash ^= zobrist(self.stack.len(), val); // XOR is self-inverse
        val
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }
    pub fn as_slice(&self) -> &[i32] {
        &self.stack
    }
    pub fn to_vec(&self) -> Vec<i32> {
        self.stack.clone()
    }

    /// Compute hash as if extra i32 values were appended (without modifying stack).
    /// Used for continuous_var_key (beta/gaussian param dedup).
    pub fn hash_with_suffix(&self, extra: &[i32]) -> u64 {
        let mut h = self.hash;
        for (i, &val) in extra.iter().enumerate() {
            h ^= zobrist(self.stack.len() + i, val);
        }
        h
    }

    /// Reconstruct from a saved Vec<i32> (recomputes hash from scratch).
    pub fn from_vec(v: Vec<i32>) -> Self {
        let mut hash = 0u64;
        for (depth, &val) in v.iter().enumerate() {
            hash ^= zobrist(depth, val);
        }
        Callstack { stack: v, hash }
    }
}

/// A callstack key used for HashMap lookups. Uses Zobrist hash instead of full Vec.
pub type CallstackKey = (u64, u64); // (callstack_hash, discriminator)

/// A full callstack key used for BDD variable ordering (needs lexicographic sort).
pub type SortedCallstackKey = (Vec<i32>, u64);

/// Describes what kind of weight a BDD variable should have.
///
/// Family-specific variants live in
/// `conjugate_pairs/{beta,dirichlet,gamma,gaussian}.rs`;
/// `FlipWeight` is a thin wrapper that dispatches `weights` /
/// `discriminator` / `is_pinned_observation` / `sample_probability` to
/// the per-family enums.
pub enum FlipWeight {
    /// Standard flip with known probability p.
    Constant(f64),
    Beta(BetaFlip),
    Dirichlet(DirichletFlip),
    Gamma(GammaFlip),
    Gaussian(GaussianFlip),
}

impl FlipWeight {
    /// SPN-weight pair `(low_edge, high_edge)` for this flip.
    pub fn weights(&self) -> (Spn<EvidenceLeaf>, Spn<EvidenceLeaf>) {
        match self {
            FlipWeight::Constant(p) => (Spn::scalar((1.0 - p).ln()), Spn::scalar(p.ln())),
            FlipWeight::Beta(f) => f.weights(),
            FlipWeight::Dirichlet(f) => f.weights(),
            FlipWeight::Gamma(f) => f.weights(),
            FlipWeight::Gaussian(f) => f.weights(),
        }
    }

    /// Per-variant hash used to key BDD variables alongside the
    /// callstack hash. Exposed so the sampling evaluator can rebuild
    /// the same `CallstackKey` for an existing flip.
    pub fn discriminator(&self) -> u64 {
        match self {
            FlipWeight::Constant(p) => p.to_bits(),
            FlipWeight::Beta(f) => f.discriminator(),
            FlipWeight::Dirichlet(f) => f.discriminator(),
            FlipWeight::Gamma(f) => f.discriminator(),
            FlipWeight::Gaussian(f) => f.discriminator(),
        }
    }

    /// True for measure-zero observations (`ProbPin`, `GaussianObs`,
    /// `RatePin`, `ExpEq`) — looked up by callstack hash alone in
    /// `observation_vars`.
    pub fn is_pinned_observation(&self) -> bool {
        match self {
            FlipWeight::Constant(_) => false,
            FlipWeight::Beta(f) => f.is_pinned_observation(),
            FlipWeight::Dirichlet(f) => f.is_pinned_observation(),
            FlipWeight::Gamma(f) => f.is_pinned_observation(),
            FlipWeight::Gaussian(f) => f.is_pinned_observation(),
        }
    }

    /// Probability used by the sample-mode flip resolver, or `None`
    /// for variants without a meaningful scalar probability
    /// (observations).
    pub fn sample_probability(&self, a: &Assignment) -> Option<f64> {
        match self {
            FlipWeight::Constant(p) => Some(*p),
            FlipWeight::Beta(f) => f.sample_probability(a),
            FlipWeight::Dirichlet(f) => f.sample_probability(a),
            FlipWeight::Gamma(f) => f.sample_probability(a),
            FlipWeight::Gaussian(f) => f.sample_probability(a),
        }
    }
}

/// A gamma-consumer draw's registry entry: its family plus the single
/// `(rate variable, scale)` it was registered under. Draws are named
/// per-rate-world (see `compile_gamma_consumer`), so each draw maps to
/// exactly one rate (see `LazyKCState::gamma_draws`).
type GammaDrawEntry = (GammaDrawFamily, (ContVarName, NotNan<f64>));

/// The main state for lazy knowledge compilation.
pub struct LazyKCState {
    /// The active Boolean-factorization backend (owns the arena).
    pub factorizer: Factorizer,
    /// WMC weight map using SPN semiring.
    pub weight_map: WeightMap<Spn<EvidenceLeaf>>,
    /// The `true` boolean-function constant (the always-satisfied guard).
    pub true_bf: BooleanFunction,
    /// Current callstack for variable ordering.
    pub callstack: Callstack,
    /// Sorted callstacks for ordering (full Vecs needed for lexicographic BDD ordering).
    pub sorted_callstacks: Vec<SortedCallstackKey>,
    /// Map from callstack hash key to (full callstack Vec, BDD variable).
    /// The full Vec is stored for collision validation (debug_assert).
    pub var_of_callstack: HashMap<CallstackKey, (Vec<i32>, BooleanFunction)>,
    /// Variable labels in `sorted_callstacks` order (parallel array, indexed by position).
    pub sorted_var_labels: Vec<usize>,
    /// Compilation depth.
    pub depth: usize,
    /// Statistics.
    pub stats: LazyKCStats,
    /// Configuration.
    pub cfg: LazyKCConfig,
    /// Start time for time limiting.
    #[cfg(not(target_arch = "wasm32"))]
    pub start_time: Option<std::time::Instant>,
    #[cfg(target_arch = "wasm32")]
    pub start_time: Option<()>,
    /// Whether we hit a limit.
    pub hit_limit: bool,
    /// Registry for all priors: maps variable name to prior distribution.
    pub prior_registry: PriorRegistry,
    /// Shared counter for generating unique continuous variable names (Beta + Gaussian).
    pub continuous_var_counter: u64,
    /// Map from callstack hash to (full callstack+params Vec, continuous variable name).
    /// When `(beta a b)` or `(gaussian mu sigma)` is compiled twice at the same callstack position
    /// (e.g., from a `define` referenced multiple times), the same beta/gaussian name is returned.
    pub continuous_of_callstack: HashMap<u64, (Vec<i32>, ContVarName)>,
    /// Like `continuous_of_callstack` but for multivariate Gaussians, where one
    /// (callstack, params) key allocates N correlated names atomically.
    pub continuous_blocks_of_callstack: HashMap<u64, (Vec<i32>, Vec<ContVarName>)>,
    /// Map from callstack hash to (full callstack Vec, BDD variable) for observation-type
    /// variables (ProbPin, GaussianObs). Used by the sampling evaluator to look up the BDD
    /// variable without needing to reconstruct the full weight-type-specific key.
    pub observation_vars: HashMap<u64, (Vec<i32>, BooleanFunction)>,
    /// Variable labels of observation-type variables (the same vars as
    /// `observation_vars`, keyed by label instead of callstack hash). Lets
    /// the sampling evaluator classify a variable found in a guard's
    /// support as a measure-zero observation (resolved false unless pinned)
    /// rather than a flip (drawn from its prior).
    pub observation_var_labels: HashSet<usize>,
    /// Every gamma-consumer draw, keyed by draw name → (family, all
    /// `(rate variable, scale)` pairs it was registered under). The draw's
    /// effective rate is `scale · rate` (a scaled gamma). A draw normally has
    /// ONE rate; a single call site whose rate expression is world-dependent
    /// (e.g. `(exponential (if w lamA lamB))`) is compiled once per rate-world
    /// at the same callstack, sharing one draw name across SEVERAL rates — the
    /// Vec keeps that ambiguity explicit (see `realize_missing_gamma_draws`).
    pub gamma_draws: HashMap<ContVarName, GammaDrawEntry>,
}

impl LazyKCState {
    /// Create a new LazyKCState.
    pub fn new(cfg: LazyKCConfig) -> Self {
        let factorizer = Factorizer::new();

        let true_bf = BooleanFunction::true_ptr();

        let weight_map = WeightMap::new(Spn::one());

        LazyKCState {
            factorizer,
            weight_map,
            true_bf,
            callstack: Callstack::new(),
            sorted_callstacks: Vec::new(),
            var_of_callstack: HashMap::new(),
            sorted_var_labels: Vec::new(),
            depth: 0,
            stats: LazyKCStats::default(),
            cfg,
            start_time: None,
            hit_limit: false,
            prior_registry: PriorRegistry::new(),
            continuous_var_counter: 0,
            continuous_of_callstack: HashMap::new(),
            continuous_blocks_of_callstack: HashMap::new(),
            observation_vars: HashMap::new(),
            observation_var_labels: HashSet::new(),
            gamma_draws: HashMap::new(),
        }
    }

    /// Create or look up a BDD variable for a constant-probability flip.
    pub fn current_address(&mut self, p: f64) -> BooleanFunction {
        self.current_address_weighted(FlipWeight::Constant(p))
    }

    /// Insert a prior into this state's `prior_registry`, routing through
    /// the appropriate family-specific `add_*`. Lets per-family compile
    /// bodies avoid knowing which named insert method to call.
    pub fn register_prior<P: RegistryInsert>(&mut self, prior: P) {
        prior.insert_into(&mut self.prior_registry);
    }

    /// Push `idx` onto the callstack, run `f`, then pop. Used for scoping
    /// BDD-variable allocation under a positional sentinel.
    pub fn with_callstack_index<R>(&mut self, idx: i32, f: impl FnOnce(&mut Self) -> R) -> R {
        self.callstack.push(idx);
        let r = f(self);
        self.callstack.pop();
        r
    }

    /// Create or look up a BDD variable with the given weight type.
    pub fn current_address_weighted(&mut self, weight: FlipWeight) -> BooleanFunction {
        let ch = self.callstack.hash();
        let discriminator = weight.discriminator();
        let key: CallstackKey = (ch, discriminator);

        if let Some((stored_cs, bf)) = self.var_of_callstack.get(&key) {
            debug_assert_eq!(
                stored_cs.as_slice(),
                self.callstack.as_slice(),
                "Zobrist hash collision in var_of_callstack! key={:?}",
                key
            );
            return bf.clone();
        }

        // Find insertion position using binary search on full callstack Vecs
        let cs_vec = self.callstack.to_vec();
        let sort_key: SortedCallstackKey = (cs_vec.clone(), discriminator);
        let pos = self
            .sorted_callstacks
            .binary_search_by(|probe| probe.0.cmp(&sort_key.0).then(probe.1.cmp(&sort_key.1)))
            .unwrap_or_else(|pos| pos);

        // Create new variable at the correct position
        let (var_id, new_var) = self.factorizer.new_var_at_position(pos, true);

        // SPN weights — per-family construction lives on `FlipWeight`.
        let (low_weight, high_weight) = weight.weights();
        self.weight_map.set_weight(var_id, low_weight, high_weight);

        self.sorted_callstacks.insert(pos, sort_key);
        self.sorted_var_labels.insert(pos, var_id.0 as usize);
        self.var_of_callstack
            .insert(key, (cs_vec.clone(), new_var.clone()));

        // For measure-zero observations (ProbPin, GaussianObs), also store a
        // callstack-hash → BddPtr mapping so the sampling evaluator can find
        // the variable without reconstructing the weight-type-specific key,
        // plus the label so guard-support classification can identify it.
        if weight.is_pinned_observation() {
            self.observation_vars.insert(ch, (cs_vec, new_var.clone()));
            self.observation_var_labels.insert(var_id.0 as usize);
        }

        new_var
    }

    /// Build the dedup hash for a continuous variable at the given callstack position.
    /// Hashes the callstack with the f64 parameters appended as i32 pairs.
    /// to ensure consistent key construction.
    pub fn continuous_var_hash(callstack: &Callstack, params: &[f64]) -> u64 {
        let bits: Vec<i32> = params
            .iter()
            .flat_map(|&p| {
                let bits = p.to_bits();
                [bits as i32, (bits >> 32) as i32]
            })
            .collect();
        callstack.hash_with_suffix(&bits)
    }

    /// Build the full Vec key for a continuous variable (used for collision validation).
    fn continuous_var_key_vec(callstack: &Callstack, params: &[f64]) -> Vec<i32> {
        let mut key = callstack.to_vec();
        for &p in params {
            let bits = p.to_bits() as i64;
            key.push(bits as i32);
            key.push((bits >> 32) as i32);
        }
        key
    }

    /// Get or create a continuous variable name for the current callstack position and prior.
    /// If `(beta a b)` or `(gaussian mu sigma)` has already been compiled at this callstack position with the
    /// same parameters, returns the same continuous name (deduplication for `define` references).
    /// Different parameters at the same call site create distinct variables.
    pub fn current_continuous_name(&mut self, params: &[f64]) -> ContVarName {
        let hash_key = Self::continuous_var_hash(&self.callstack, params);
        if let Some((stored_key, existing)) = self.continuous_of_callstack.get(&hash_key) {
            let full_key = Self::continuous_var_key_vec(&self.callstack, params);
            debug_assert_eq!(
                stored_key, &full_key,
                "Zobrist hash collision in continuous_of_callstack!"
            );
            return *existing;
        }
        let name = self.continuous_var_counter;
        self.continuous_var_counter += 1;
        let full_key = Self::continuous_var_key_vec(&self.callstack, params);
        self.continuous_of_callstack
            .insert(hash_key, (full_key, name));
        name
    }

    /// Get or create a *block* of `n` correlated continuous variable names for
    /// the current callstack position and prior. Mirrors
    /// `current_continuous_name`'s dedup contract: repeated calls with the same
    /// `(callstack, params)` return the same block (same N names, same order).
    /// Used by multivariate Gaussian priors.
    pub fn current_continuous_names(&mut self, params: &[f64], n: usize) -> Vec<ContVarName> {
        let hash_key = Self::continuous_var_hash(&self.callstack, params);
        if let Some((stored_key, existing)) = self.continuous_blocks_of_callstack.get(&hash_key) {
            let full_key = Self::continuous_var_key_vec(&self.callstack, params);
            debug_assert_eq!(
                stored_key, &full_key,
                "Zobrist hash collision in continuous_blocks_of_callstack!"
            );
            debug_assert_eq!(
                existing.len(),
                n,
                "continuous_blocks_of_callstack: re-entry with mismatched block size"
            );
            return existing.clone();
        }
        let names: Vec<ContVarName> = (0..n)
            .map(|_| {
                let name = self.continuous_var_counter;
                self.continuous_var_counter += 1;
                name
            })
            .collect();
        let full_key = Self::continuous_var_key_vec(&self.callstack, params);
        self.continuous_blocks_of_callstack
            .insert(hash_key, (full_key, names.clone()));
        names
    }

    /// Compute weighted model count as an SPN.
    pub fn wmc_spn(&self, bf: BooleanFunction) -> Spn<EvidenceLeaf> {
        self.factorizer.wmc(&bf, &self.weight_map)
    }

    /// The active Boolean-factorization backend.
    pub fn fac(&self) -> &Factorizer {
        &self.factorizer
    }

    /// If `constraint` entails `var_bf` (true) or its negation (false),
    /// return `Some(bool)`; otherwise `None`.
    pub(crate) fn implies(
        &self,
        constraint: BooleanFunction,
        var_bf: BooleanFunction,
    ) -> Option<bool> {
        let fac = self.fac();
        let neg_constraint = fac.negate(&constraint);
        if fac.or(&neg_constraint, &var_bf).is_true() {
            return Some(true);
        }
        let neg_var = fac.negate(&var_bf);
        if fac.or(&neg_constraint, &neg_var).is_true() {
            return Some(false);
        }
        None
    }

    /// Check whether `constraint` forces the flip variable identified by
    /// `(callstack_hash, key)` (a key in `var_of_callstack`). Returns
    /// `Some(true)` if forced true, `Some(false)` if forced false,
    /// `None` if the variable is missing or the constraint doesn't pin it.
    pub fn constraint_forces(
        &self,
        constraint: BooleanFunction,
        callstack_hash: u64,
        key: u64,
    ) -> Option<bool> {
        let (_, var_bf) = self.var_of_callstack.get(&(callstack_hash, key))?;
        self.implies(constraint, var_bf.clone())
    }

    /// Check whether `constraint` forces the observation variable at
    /// `callstack_hash` (a key in `observation_vars`). Returns
    /// `Some(true)`/`Some(false)` if forced, `None` otherwise.
    pub fn constraint_forces_observation(
        &self,
        constraint: BooleanFunction,
        callstack_hash: u64,
    ) -> Option<bool> {
        let (_, var_bf) = self.observation_vars.get(&callstack_hash)?;
        self.implies(constraint, var_bf.clone())
    }

    /// Draw a full continuous assignment from `P(continuous | evidence)`.
    ///
    /// Variables in `evidence_spn.scope()` are sampled from the conjugate
    /// posterior; variables registered in `prior_registry` but absent
    /// from that scope are drawn from their prior.
    pub fn sample_continuous_posterior<R: rand::Rng>(
        &self,
        evidence_spn: &Spn<EvidenceLeaf>,
        rng: &mut R,
    ) -> Assignment {
        let evidence_scope: BTreeSet<ContVarName> = evidence_spn.scope().iter().copied().collect();
        let touched = evidence_spn
            .posterior(&self.prior_registry, &evidence_scope)
            .sample(rng);

        let full_scope: BTreeSet<ContVarName> = self.prior_registry.scope().copied().collect();
        let missing: BTreeSet<ContVarName> =
            full_scope.difference(&touched.scope()).copied().collect();
        let untouched = self.prior_registry.slice(&missing).sample(rng);

        touched.merge(&untouched)
    }

    /// Cover gamma draws referenced by the weight SPNs of `labels` that are
    /// missing from `assignment`, drawing each from its family conditional
    /// given the assignment's rate (the prior-conditional law of an
    /// unobserved draw — the weight-driven generalization of
    /// `SampleState::ensure_gamma_draw`).
    ///
    /// A draw can be missing when the posterior sample realized a Sum
    /// branch that doesn't observe it (a free discrete choice routed the
    /// evidence to a different consumer call site); its flips can still sit
    /// in the BDD support whose weights `sample_discrete_given` /
    /// `var_sample_prob` evaluate. Rates are always covered already: every
    /// rate is registry-registered, so the up-front draw spans it.
    ///
    /// Each draw maps to exactly one rate: a world-dependent rate
    /// (`(poisson (if b 20.0 0.5))`) gets a distinct draw per rate-world
    /// (`compile_gamma_consumer`), so there is no rate-mixture to resolve
    /// here — the single registered rate is the draw's true unobserved law,
    /// and the per-branch-distinct draws make the two-stage sampler's
    /// indicator weights exact.
    pub fn realize_missing_gamma_draws<R: rand::Rng>(
        &self,
        labels: impl IntoIterator<Item = usize>,
        assignment: &mut Assignment,
        rng: &mut R,
    ) {
        let mut covered = assignment.scope();
        for label in labels {
            let (low, high) = self.weight_map.var_weight(VarId(label as u64));
            for var in low.scope().iter().chain(high.scope().iter()) {
                if covered.contains(var) {
                    continue;
                }
                // Missing non-draw variables are left alone: reading one
                // later panics in `read_realization`, flagging a genuine
                // coverage bug rather than papering over it.
                let Some((family, (rate, scale))) = self.gamma_draws.get(var).copied() else {
                    continue;
                };
                // The draw's effective rate is `scale · λ` (a scaled gamma).
                let lambda = assignment.gamma_value(rate) * scale;
                let sampled = match family {
                    GammaDrawFamily::Poisson => {
                        PoissonPrior::unconstrained(std::iter::once(*var)).sample(lambda, rng)
                    }
                    GammaDrawFamily::Exponential => {
                        ExponentialPrior::unconstrained(std::iter::once(*var)).sample(lambda, rng)
                    }
                };
                let single =
                    Assignment::from_sorted(vec![], vec![], sampled.into_iter().collect(), vec![]);
                *assignment = assignment.merge(&single);
                covered.insert(*var);
            }
        }
    }

    /// `realize_missing_gamma_draws` over the support of `bf` — the
    /// pre-`sample_discrete_given` step shared by `posterior_sample` and
    /// the Gibbs `sample_world`.
    pub fn cover_support_gamma_draws<R: rand::Rng>(
        &self,
        bf: BooleanFunction,
        assignment: &mut Assignment,
        rng: &mut R,
    ) {
        let mut support: BTreeSet<usize> = BTreeSet::new();
        support_vars(self.fac(), &bf, &mut support);
        self.realize_missing_gamma_draws(support, assignment, rng);
    }

    /// Sample one root-to-True **path** through `evidence_bf` from
    /// `P(discrete | continuous = assignment, evidence_bf)`. With every
    /// continuous variable fixed, the BDD weights reduce to plain scalars
    /// and the standard bottom-up-marginal + top-down sample is exact.
    ///
    /// Returns the conjunction of the literals on the sampled path — a
    /// **partial** assignment, not a full one: variables the path
    /// short-circuits past (the evidence is satisfied either way given the
    /// literals so far) are absent. Such variables are deliberately left
    /// unconstrained; downstream sampling draws them from their prior on
    /// demand (`P(v | path, evidence) = prior(v)` since flips are a-priori
    /// independent and every extension of the path satisfies the evidence).
    pub fn sample_discrete_given<R: rand::Rng>(
        &self,
        evidence_bf: BooleanFunction,
        assignment: &Assignment,
        rng: &mut R,
    ) -> BooleanFunction {
        // Log-domain edge weights, restricted to the evidence BDD's
        // support: the marginal walk and the top-down sample only ever
        // consult support variables (everything else falls back to the
        // default weight below). Restricting also keeps this from
        // evaluating query-side gamma-draw flips registered by a
        // previous sample's forward run, whose draws this assignment
        // does not cover (those are realized on demand — see
        // `SampleState::ensure_gamma_draw`). Support-side gamma draws,
        // by contrast, must already be covered: callers run
        // `cover_support_gamma_draws` first, which fills draws the
        // posterior sample skipped (unsampled Sum branches).
        let mut support: BTreeSet<usize> = BTreeSet::new();
        support_vars(self.fac(), &evidence_bf, &mut support);
        let mut log_weights: HashMap<usize, (f64, f64)> = HashMap::new();
        for &label_val in &support {
            let (low_spn, high_spn) = self.weight_map.var_weight(VarId(label_val as u64));
            let log_low = low_spn.log_likelihood(assignment);
            let log_high = high_spn.log_likelihood(assignment);
            log_weights.insert(label_val, (log_low, log_high));
        }

        // Bottom-up log-marginal at every BDD node:
        //   log_m[node] = logsumexp(log_w_lo + log_m[lo], log_w_hi + log_m[hi]).
        let mut log_cache: HashMap<u64, f64> = HashMap::new();
        bottomup_log_marginal(self.fac(), &evidence_bf, &log_weights, &mut log_cache);

        // Default per-variable log-weight for vars not in the registry —
        // the linear-domain version used `(0.5, 0.5)`, which translates to
        // `(ln 0.5, ln 0.5)`.
        let default_log_weight = (-std::f64::consts::LN_2, -std::f64::consts::LN_2);

        // Top-down sample, AND-ing chosen literals into the result.
        // If the traversal aborts before reaching `PtrTrue` (because some
        // node's children both have zero log-marginal under this continuous
        // assignment), we return `false_ptr` — *not* the partial result.
        // A partial conjunction would leave downstream `force_value` to
        // resolve unconstrained variables independently, which can produce
        // joint-constraint-violating assignments (e.g. an `IntDist`
        // resolving to a value outside its `uniform_int_range`).
        let fac = self.fac();
        let mut result = BooleanFunction::true_ptr();
        let mut current = evidence_bf;
        let mut aborted = false;
        loop {
            match fac.node(&current) {
                BddNode::True => break,
                BddNode::False => {
                    eprintln!("sample_discrete_given: reached False during sampling");
                    aborted = true;
                    break;
                }
                BddNode::Inner {
                    var,
                    low: l,
                    high: h,
                } => {
                    let choose_high = if l.is_false() {
                        true
                    } else if h.is_false() {
                        false
                    } else {
                        let (log_w_lo, log_w_hi) = log_weights
                            .get(&(var.0 as usize))
                            .copied()
                            .unwrap_or(default_log_weight);
                        let log_m_lo = *log_cache
                            .get(&fac.node_id(&l))
                            .unwrap_or(&f64::NEG_INFINITY);
                        let log_m_hi = *log_cache
                            .get(&fac.node_id(&h))
                            .unwrap_or(&f64::NEG_INFINITY);
                        let log_prob_low = log_w_lo + log_m_lo;
                        let log_prob_high = log_w_hi + log_m_hi;
                        let log_total = logsumexp(log_prob_low, log_prob_high);
                        if log_total == f64::NEG_INFINITY {
                            eprintln!("sample_discrete_given: zero total at var {:?}", var);
                            aborted = true;
                            break;
                        }
                        // Sample `high` with probability
                        // `exp(log_prob_high - log_total)`. Comparing
                        // `u.ln()` against the difference avoids
                        // exponentiating a possibly very negative value;
                        // both sides stay in log domain. `u == 0.0` →
                        // `u.ln() == -inf`, which always picks high
                        // whenever log_prob_high is finite — that's
                        // correct (zero-probability low edge).
                        let u: f64 = rng.gen();
                        u.ln() < log_prob_high - log_total
                    };

                    let lit = fac.var(var, choose_high);
                    result = fac.and(&result, &lit);
                    current = if choose_high { h } else { l };
                }
            }
        }
        if aborted {
            BooleanFunction::false_ptr()
        } else {
            result
        }
    }

    /// Probability that BDD variable `label` is true under its flip prior,
    /// evaluated at the sampled continuous `assignment`.
    ///
    /// Reads the `(low, high)` SPN weights off `weight_map` and normalizes:
    /// `p = w_hi / (w_lo + w_hi)`, computed in log domain as
    /// `1 / (1 + exp(ll - lh))`. Per family:
    /// - `Constant(p)` weights are `((1-p).ln(), p.ln())` → returns `p`;
    /// - Beta flips are `(t·ln(1-θ), h·ln θ)` counts → returns the sampled θ;
    /// - Dirichlet sticks return the stick-breaking conditional
    ///   `β_i = θ_i / Σ_{j≥i} θ_j` (see `counts_log_likelihood_at`).
    ///
    /// Not meaningful for measure-zero observation variables (ProbPin,
    /// GaussianObs, VecPin) — callers must filter those out first via
    /// `observation_var_labels`.
    pub fn var_sample_prob(&self, label: usize, assignment: &Assignment) -> f64 {
        let (low_spn, high_spn) = self.weight_map.var_weight(VarId(label as u64));
        let ll = low_spn.log_likelihood(assignment);
        let lh = high_spn.log_likelihood(assignment);
        debug_assert!(
            ll != f64::NEG_INFINITY || lh != f64::NEG_INFINITY,
            "var_sample_prob: both edge weights are zero for var {}",
            label
        );
        1.0 / (1.0 + (ll - lh).exp())
    }

    /// Check if we've exceeded the time limit.
    pub fn check_time_limit(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let (Some(start), Some(limit)) = (self.start_time, self.cfg.time_limit) {
                return start.elapsed().as_secs_f64() > limit;
            }
        }
        false
    }
}

/// Collect the support (variable labels) of `root` into `out`.
///
/// DFS over the decision DAG via [`BooleanFactorization::node`] with a visited
/// set so shared subgraphs are walked once (a plain fold would not dedupe
/// them — the same reason `bottomup_log_marginal` hand-rolls its walk).
/// Support is sign-insensitive, so nodes are keyed by [`node_ref_id`], which
/// ignores complement.
///
/// [`node_ref_id`]: BooleanFactorization::node_ref_id
pub fn support_vars<M: BooleanFactorization>(mgr: &M, root: &M::Ptr, out: &mut BTreeSet<usize>) {
    fn walk<M: BooleanFactorization>(
        mgr: &M,
        p: &M::Ptr,
        out: &mut BTreeSet<usize>,
        visited: &mut HashSet<u64>,
    ) {
        // Terminals and the visited check come before `node()`: decomposing
        // materializes owned children, so testing membership afterwards took
        // and released two references per already-seen node for nothing.
        if p.is_true() || p.is_false() {
            return;
        }
        if !visited.insert(mgr.node_ref_id(p)) {
            return;
        }
        if let BddNode::Inner { var, low, high } = mgr.node(p) {
            out.insert(var.0 as usize);
            walk(mgr, &low, out, visited);
            walk(mgr, &high, out, visited);
        }
    }
    let mut visited = HashSet::new();
    walk(mgr, root, out, &mut visited);
}

/// Log-domain bottom-up marginal for `sample_discrete_given`. Returns
/// `log(sum over assignments of product of edge weights)` for the
/// subtree rooted at `p`. `True → 0.0` (log 1), `False → NEG_INFINITY`
/// (log 0). Caching keyed by the sign-distinguishing [`node_id`] ensures
/// DAG-shared subgraphs are visited once (and `f`/`¬f` stay distinct).
///
/// [`node_id`]: BooleanFactorization::node_id
fn bottomup_log_marginal<M: BooleanFactorization>(
    mgr: &M,
    p: &M::Ptr,
    log_weights: &HashMap<usize, (f64, f64)>,
    cache: &mut HashMap<u64, f64>,
) -> f64 {
    let key = mgr.node_id(p);
    if let Some(&val) = cache.get(&key) {
        return val;
    }
    let result = match mgr.node(p) {
        BddNode::True => 0.0,
        BddNode::False => f64::NEG_INFINITY,
        BddNode::Inner { var, low, high } => {
            let log_low_m = bottomup_log_marginal(mgr, &low, log_weights, cache);
            let log_high_m = bottomup_log_marginal(mgr, &high, log_weights, cache);
            let (log_w_lo, log_w_hi) = log_weights
                .get(&(var.0 as usize))
                .copied()
                .unwrap_or((-std::f64::consts::LN_2, -std::f64::consts::LN_2));
            logsumexp(log_w_lo + log_low_m, log_w_hi + log_high_m)
        }
    };
    cache.insert(key, result);
    result
}
