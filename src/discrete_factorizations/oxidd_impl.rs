//! The OxiDD-backed implementation of [`BooleanFactorization`].
//!
//! OxiDD (pure Rust) is used via its **complement-edge** BDD variant
//! (`oxidd::bcdd`)
//! 
//! OxiDD *does not* support inserting variables at specific positions
//! so the resulting variable ordering (and program performance) may
//! differ meaningfully from other backends.

use std::cell::RefCell;

use oxidd::bcdd::{BCDDFunction, BCDDManagerRef};
use oxidd::{BooleanFunction, Edge, Function, HasLevel, InnerNode, Manager, ManagerRef, Node};

use super::boolean_factorization::{
    BddNode, BooleanFactorization, BooleanFunctionOps, FactorizationStats, VarId,
};
use super::wmc::{wmc_fold, Semiring, WeightMap, Wmc};

/// Node and apply-cache capacities for the manager. OxiDD's index-based manager
/// takes a fixed capacity up front (unlike rsdd's unbounded arena or CUDD's
/// growable table) and reports `OutOfMemory` — surfaced as a panic below — once
/// exceeded. It is preallocated, so the capacity is paid in resident memory
/// whether or not it is used. Overridable via `PLUCK_OXIDD_CAPACITY`.
const DEFAULT_NODE_CAPACITY: usize = 1 << 23; // ~8.4M nodes

fn node_capacity() -> usize {
    std::env::var("PLUCK_OXIDD_CAPACITY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NODE_CAPACITY)
}

thread_local! {
    /// This thread's manager plus its two constant handles.
    ///
    /// `BooleanFunctionOps` mints constants with no factorizer in scope, and
    /// oxidd's constants belong to a manager. The handles are cached rather
    /// than rebuilt because `BCDDFunction::t(m)` needs a manager read-lock and
    /// `pure_monad` — one of the hottest paths in the engine — asks for two
    /// constants per call; cloning a cached handle is just a refcount bump.
    static OXIDD_CONSTS: RefCell<Option<(BCDDFunction, BCDDFunction)>> =
        const { RefCell::new(None) };
}

fn constant(want_true: bool) -> BCDDFunction {
    OXIDD_CONSTS.with(|c| {
        let b = c.borrow();
        let (f, t) = b
            .as_ref()
            .expect("OxiddBdd constant requested before any OxiddFactorizer on this thread");
        if want_true {
            t.clone()
        } else {
            f.clone()
        }
    })
}

/// An owning handle to an OxiDD BCDD function.
///
/// The newtype exists only to add `Debug` (which `BCDDFunction` does not
/// derive) and to keep oxidd's types out of the engine-facing signature
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct OxiddBdd(BCDDFunction);

impl std::fmt::Debug for OxiddBdd {
    fn fmt(&self, fm: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            fm,
            "OxiddBdd({:?})",
            self.0.with_manager_shared(|_m, e| e.node_id())
        )
    }
}

impl BooleanFunctionOps for OxiddBdd {
    #[inline]
    fn true_ptr() -> Self {
        OxiddBdd(constant(true))
    }
    #[inline]
    fn false_ptr() -> Self {
        OxiddBdd(constant(false))
    }
    #[inline]
    fn is_true(&self) -> bool {
        self.0 == constant(true)
    }
    #[inline]
    fn is_false(&self) -> bool {
        self.0 == constant(false)
    }
}

struct OxiddInner {
    num_vars: u32,
}

pub struct OxiddFactorizer {
    mref: BCDDManagerRef,
    inner: RefCell<OxiddInner>,
}

impl OxiddFactorizer {
    /// Create a fresh factorizer with an empty variable order, and make it this
    /// thread's constant source.
    pub fn new() -> Self {
        let cap = node_capacity();
        let mref = oxidd::bcdd::new_manager(cap, cap, 1);
        let (f_false, f_true) =
            mref.with_manager_shared(|m| (BCDDFunction::f(m), BCDDFunction::t(m)));
        OXIDD_CONSTS.with(|c| *c.borrow_mut() = Some((f_false, f_true)));
        OxiddFactorizer {
            mref,
            inner: RefCell::new(OxiddInner { num_vars: 0 }),
        }
    }

    fn literal(&self, vn: u32, polarity: bool) -> BCDDFunction {
        self.mref
            .with_manager_shared(|m| {
                if polarity {
                    BCDDFunction::var(m, vn)
                } else {
                    BCDDFunction::not_var(m, vn)
                }
            })
            .expect("oxidd: out of memory building a literal")
    }
}

impl Default for OxiddFactorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply a binary `BooleanFunction` op, yielding an owned handle.
macro_rules! binop {
    ($self:ident, $a:ident, $b:ident, $op:ident) => {{
        OxiddBdd(
            $a.0.$op(&$b.0)
                .expect(concat!("oxidd: out of memory in ", stringify!($op))),
        )
    }};
}

impl BooleanFactorization for OxiddFactorizer {
    type Ptr = OxiddBdd;

    #[inline]
    fn and(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr {
        binop!(self, a, b, and)
    }
    #[inline]
    fn or(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr {
        binop!(self, a, b, or)
    }
    #[inline]
    fn xor(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr {
        binop!(self, a, b, xor)
    }
    #[inline]
    fn iff(&self, a: &Self::Ptr, b: &Self::Ptr) -> Self::Ptr {
        binop!(self, a, b, equiv)
    }
    #[inline]
    fn ite(&self, f: &Self::Ptr, g: &Self::Ptr, h: &Self::Ptr) -> Self::Ptr {
        OxiddBdd(f.0.ite(&g.0, &h.0).expect("oxidd: out of memory in ite"))
    }
    #[inline]
    fn negate(&self, a: &Self::Ptr) -> Self::Ptr {
        OxiddBdd(a.0.not().expect("oxidd: out of memory in not"))
    }

    #[inline]
    fn var(&self, v: VarId, polarity: bool) -> Self::Ptr {
        OxiddBdd(self.literal(v.0 as u32, polarity))
    }

    fn new_var_at_position(&self, _position: usize, polarity: bool) -> (VarId, Self::Ptr) {
        // OxiDD 0.12 appends variables at the bottom of the order (no cheap
        // mid-order insert), so `position` is a no-op hint. WMC/posteriors are
        // order-independent, so this only affects node counts, not results.
        let vn = self.mref.with_manager_exclusive(|m| m.add_vars(1).start);
        self.inner.borrow_mut().num_vars = vn + 1;
        (VarId(vn as u64), OxiddBdd(self.literal(vn, polarity)))
    }

    fn node(&self, p: &Self::Ptr) -> BddNode<Self::Ptr> {
        if p.is_false() {
            return BddNode::False;
        }
        if p.is_true() {
            return BddNode::True;
        }
        let f = &p.0;
        // Decompose one level, folding the incoming complement tag into the
        // children so they are honest sub-functions. `child(0)` = high (then,
        // var = true), `child(1)` = low (else, var = false); no reordering ever
        // runs, so level == variable number.
        let decomp = f.with_manager_shared(|m, edge| match m.get_node(edge) {
            Node::Terminal(_) => None,
            Node::Inner(nd) => {
                let level = nd.level();
                let ptag = edge.tag();
                let high_b = nd.child(0);
                let low_b = nd.child(1);
                let high_e = m.clone_edge(&high_b).with_tag_owned(high_b.tag() ^ ptag);
                let low_e = m.clone_edge(&low_b).with_tag_owned(low_b.tag() ^ ptag);
                let high = BCDDFunction::from_edge(m, high_e);
                let low = BCDDFunction::from_edge(m, low_e);
                Some((level, high, low))
            }
        });
        match decomp {
            None => BddNode::True, // unreachable: constants handled above
            Some((level, high, low)) => BddNode::Inner {
                var: VarId(level as u64),
                low: OxiddBdd(low),
                high: OxiddBdd(high),
            },
        }
    }

    #[inline]
    fn node_id(&self, p: &Self::Ptr) -> u64 {
        // Node index in the high bits, complement tag in bit 0. `Tag: Eq +
        // Default` and the default tag is the uncomplemented one, so the sign
        // is readable without naming oxidd's `EdgeTag` (which the `oxidd`
        // facade does not re-export).
        p.0.with_manager_shared(|_m, edge| {
            let complemented = edge.tag() != Default::default();
            ((edge.node_id() as u64) << 1) | complemented as u64
        })
    }

    #[inline]
    fn node_ref_id(&self, p: &Self::Ptr) -> u64 {
        // Sign-insensitive: the node index alone, tag dropped.
        p.0.with_manager_shared(|_m, edge| edge.node_id() as u64)
    }

    #[inline]
    fn num_vars(&self) -> usize {
        self.inner.borrow().num_vars as usize
    }
    #[inline]
    fn count_nodes(&self, p: &Self::Ptr) -> usize {
        p.0.node_count()
    }
    fn stats(&self) -> FactorizationStats {
        FactorizationStats {
            num_recursive_calls: None,
        }
    }
}

impl<T: Semiring> Wmc<T> for OxiddFactorizer {
    fn wmc(&self, p: &Self::Ptr, w: &WeightMap<T>) -> T {
        wmc_fold(self, p, w)
    }
}
