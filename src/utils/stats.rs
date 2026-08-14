//! Opt-in engine instrumentation, enabled by the `PLUCK_STATS` env var.
//!
//! Reports the sizes of the internal representations a distribution query
//! builds — the discrete boolean-function guards and the continuous-likelihood
//! SPN, counted both as an unshared TREE and as the hash-consed DAG — plus a
//! coarse per-stage timing breakdown, all to stderr.
//!
//! This is a measurement aid only. When `PLUCK_STATS` is unset the collector is
//! disabled: every method is a cheap no-op and nothing is printed.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::time::Duration;

use crate::discrete_factorizations::{BooleanFactorization, BooleanFunction, Factorizer};
use crate::inference::lazy_kc::state::LazyKCState;
use crate::inference::spn::evidence::intern_table_size;
use crate::inference::spn::{Spn, SpnKind, SpnLeaf};

/// Per-stage wall-clock for processing one world.
#[derive(Clone, Copy, Default)]
pub struct StageTimes {
    /// Building the evidence SPN from the boolean-function guard (`wmc_spn`).
    pub wmc: Duration,
    /// Converting the evidence SPN to a posterior SPN.
    pub posterior: Duration,
    /// `.components()` — flattening the posterior DAG into an explicit mixture.
    pub flatten: Duration,
    /// Naming derivation + component consume loop: packets, display strings,
    /// accumulation.
    pub display: Duration,
}

/// Accumulates boolean-function/SPN size and timing measurements across the
/// worlds of a single distribution query. Construct with [`KcStats::new`]; when
/// disabled (the `PLUCK_STATS` env var is unset) all methods no-op.
#[derive(Default)]
pub struct KcStats {
    enabled: bool,
    worlds: usize,
    // Boolean-function guard sizes (per world).
    bf_sum: usize,
    bf_max: usize,
    // Evidence-SPN sizes, counted as an unshared TREE...
    spn_tree_sum: usize,
    spn_tree_max: usize,
    spn_leaf_sum: usize,
    sum_arity_max: usize,
    prod_arity_max: usize,
    // ...and as the interned DAG (shared sub-SPNs counted once).
    spn_dag_max: usize,
    spn_dag_all: HashSet<*const ()>,
    // Flattened posterior + timings.
    components: usize,
    /// See [`KcStats::set_parse`].
    t_parse: f64,
    /// See [`KcStats::set_symbolic`].
    t_symbolic: f64,
    t_wmc: f64,
    t_post: f64,
    t_flatten: f64,
    t_display: f64,
}

impl KcStats {
    /// Enabled iff the `PLUCK_STATS` env var is set.
    pub fn new() -> Self {
        KcStats {
            enabled: std::env::var_os("PLUCK_STATS").is_some(),
            ..Default::default()
        }
    }

    /// Record the program-parsing (source → AST) time for this query. Covers
    /// only the program itself; the stdlib parse is excluded. Called once
    /// before the per-world loop; no-op when disabled.
    pub fn set_parse(&mut self, d: Duration) {
        if !self.enabled {
            return;
        }
        self.t_parse += d.as_secs_f64();
    }

    /// Record the symbolic-execution (program → boolean-function compilation)
    /// time for this query. Called once before the per-world loop; no-op when
    /// disabled.
    pub fn set_symbolic(&mut self, d: Duration) {
        if !self.enabled {
            return;
        }
        self.t_symbolic += d.as_secs_f64();
    }

    /// Record one world: its boolean-function guard, its evidence SPN, the
    /// per-stage timings, and how many (non-zero) mixture components it
    /// flattened to.
    pub fn record_world<L: SpnLeaf>(
        &mut self,
        fac: &Factorizer,
        bf: BooleanFunction,
        evidence: &Spn<L>,
        times: StageTimes,
        components: usize,
    ) {
        if !self.enabled {
            return;
        }
        self.worlds += 1;

        let gn = fac.count_nodes(&bf);
        self.bf_sum += gn;
        self.bf_max = self.bf_max.max(gn);

        let tree = spn_tree_size(evidence);
        self.spn_tree_sum += tree.nodes;
        self.spn_tree_max = self.spn_tree_max.max(tree.nodes);
        self.spn_leaf_sum += tree.leaves;
        self.sum_arity_max = self.sum_arity_max.max(tree.sums);
        self.prod_arity_max = self.prod_arity_max.max(tree.products);

        let mut world_dag = HashSet::new();
        collect_dag_nodes(evidence, &mut world_dag);
        collect_dag_nodes(evidence, &mut self.spn_dag_all);
        self.spn_dag_max = self.spn_dag_max.max(world_dag.len());

        self.components += components;
        self.t_wmc += times.wmc.as_secs_f64();
        self.t_post += times.posterior.as_secs_f64();
        self.t_flatten += times.flatten.as_secs_f64();
        self.t_display += times.display.as_secs_f64();
    }

    /// Print the gathered stats to stderr. No-op when disabled.
    pub fn report(&self, state: &LazyKCState) {
        if !self.enabled {
            return;
        }
        let nw = self.worlds.max(1);
        eprintln!("=== PLUCK_STATS ===");
        eprintln!("worlds (value,guard pairs)            : {}", self.worlds);
        eprintln!(
            "continuous latents (Beta/Gaussian/…) : {}",
            state.continuous_var_counter
        );
        eprintln!(
            "guard vars allocated (discrete bits)  : {}",
            state.fac().num_vars()
        );
        eprintln!(
            "factorizer recursive calls            : {}",
            state.fac().stats().num_recursive_calls.unwrap_or(0)
        );
        eprintln!(
            "guard nodes      : sum={}  max={}  mean={}",
            self.bf_sum,
            self.bf_max,
            self.bf_sum / nw
        );
        eprintln!(
            "SPN nodes (TREE) : sum={}  max={}  mean={}",
            self.spn_tree_sum,
            self.spn_tree_max,
            self.spn_tree_sum / nw
        );
        eprintln!(
            "SPN nodes (DAG)  : per-world-max={}  all-worlds-unique={}  intern-table={}",
            self.spn_dag_max,
            self.spn_dag_all.len(),
            intern_table_size()
        );
        eprintln!(
            "SPN leaf nodes   : sum={}  (max sum-node-count={}, max product-node-count={})",
            self.spn_leaf_sum, self.sum_arity_max, self.prod_arity_max
        );
        eprintln!(
            "posterior mixture components (total)  : {}",
            self.components
        );
        // `total` is the sum of the six stages, so it excludes the size-walk
        // instrumentation above (`record_world` runs those walks outside every
        // timed window). Parsing (program only, not stdlib) is folded in so the
        // reported inference total accounts for it. The `inference_seconds:`
        // line repeats it under the label the benchmark harness greps for.
        let total = self.t_parse
            + self.t_symbolic
            + self.t_wmc
            + self.t_post
            + self.t_flatten
            + self.t_display;
        eprintln!(
            "time (s): parse={:.4}  symbolic-exec={:.4}  wmc={:.4}  posterior={:.4}  flatten={:.4}  display/naming={:.4}  total={:.4}",
            self.t_parse, self.t_symbolic, self.t_wmc, self.t_post, self.t_flatten, self.t_display, total
        );
        eprintln!("===================");
        eprintln!("inference_seconds: {:.8}", total);
    }
}

/// Node tallies from a TREE walk of an SPN (shared sub-DAGs counted once per
/// reference — i.e. the unrolled size).
#[derive(Default, Clone, Copy)]
struct TreeSize {
    nodes: usize,
    leaves: usize,
    sums: usize,
    products: usize,
}

/// Size an SPN as a TREE: shared sub-DAGs are counted once per reference, so
/// this is the unrolled node count (contrast [`collect_dag_nodes`]).
///
/// Memoised on `inner_ptr` and saturating: the tree count can be exponentially
/// larger than the DAG, so a naive recursion would re-walk shared children and
/// be exponential-TIME itself (and overflow). Memoisation still ADDS a shared
/// child's count once per reference (so the reported total is the true unrolled
/// size) while visiting each DAG node only once — O(DAG) time.
fn spn_tree_size<L: SpnLeaf>(spn: &Spn<L>) -> TreeSize {
    fn go<L: SpnLeaf>(spn: &Spn<L>, memo: &mut HashMap<*const (), TreeSize>) -> TreeSize {
        let ptr = spn.inner_ptr() as *const ();
        if let Some(&v) = memo.get(&ptr) {
            return v;
        }
        let v = match spn.kind() {
            SpnKind::Scalar => TreeSize {
                nodes: 1,
                leaves: 0,
                sums: 0,
                products: 0,
            },
            SpnKind::Leaf(_) => TreeSize {
                nodes: 1,
                leaves: 1,
                sums: 0,
                products: 0,
            },
            SpnKind::Sum(cs) | SpnKind::Product(cs) => {
                let is_sum = matches!(spn.kind(), SpnKind::Sum(_));
                let mut acc = TreeSize {
                    nodes: 1,
                    leaves: 0,
                    sums: is_sum as usize,
                    products: !is_sum as usize,
                };
                for c in cs {
                    let child = go(c, memo);
                    acc.nodes = acc.nodes.saturating_add(child.nodes);
                    acc.leaves = acc.leaves.saturating_add(child.leaves);
                    acc.sums = acc.sums.saturating_add(child.sums);
                    acc.products = acc.products.saturating_add(child.products);
                }
                acc
            }
        };
        memo.insert(ptr, v);
        v
    }
    go(spn, &mut HashMap::new())
}

/// Collect DAG-unique `SpnInner` allocations by `Rc` pointer identity. For
/// interned leaf types (`EvidenceLeaf`) pointer identity == structural
/// identity, so `seen.len()` is the true shared-DAG node count.
fn collect_dag_nodes<L: SpnLeaf>(spn: &Spn<L>, seen: &mut HashSet<*const ()>) {
    let ptr = Rc::as_ptr(&spn.inner) as *const ();
    if !seen.insert(ptr) {
        return;
    }
    if let SpnKind::Sum(cs) | SpnKind::Product(cs) = spn.kind() {
        for c in cs {
            collect_dag_nodes(c, seen);
        }
    }
}
