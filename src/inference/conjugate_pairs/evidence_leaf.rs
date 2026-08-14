//! `EvidenceLeaf`: per-family discriminated union of sufficient statistics.
//!
//! Lives in `conjugate_pairs/` (alongside the per-family files) so that
//! adding a new family means adding one enum variant here plus one file
//! under `conjugate_pairs/`. The hash-consing intern table follows the
//! enum because it keys on `Rc<SpnInner<EvidenceLeaf>>`.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashSet};
use std::rc::Rc;

use ordered_float::NotNan;

use crate::inference::conjugate_pairs::{
    BetaSuffStat, ContVarName, DirichletSuffStat, GammaSuffStat, GaussianObs, SuffStat,
};
use crate::inference::spn::coeff::SpnLeaf;
use crate::inference::spn::node::SpnInner;

/// A typed sufficient statistic carried by an SPN leaf. Each variable is
/// globally typed (Beta xor Dirichlet xor Gamma xor Gaussian), so two
/// leaves with overlapping scopes share a single `EvidenceLeaf` variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EvidenceLeaf {
    Beta(BetaSuffStat),
    Dirichlet(DirichletSuffStat),
    Gamma(GammaSuffStat),
    Gaussian(GaussianObs),
}

// Hash-consing intern table (EvidenceLeaf only; other `SpnLeaf` impls use `Rc::new(...)` directly).
// TODO: Current hash-consing does not interact well with sum normalization
// two sum nodes with the same weight ratios but different log coeff might end up
// having slightly different weights after normalization so they get hashed
// separately.

thread_local! {
    static INTERN_TABLE: RefCell<HashSet<Rc<SpnInner<EvidenceLeaf>>>> = RefCell::new(HashSet::new());
}

/// Clear the EvidenceLeaf intern table. Call between top-level queries to
/// free memory.
pub fn clear_intern_table() {
    INTERN_TABLE.with(|table| table.borrow_mut().clear());
}

/// Current size of the EvidenceLeaf intern table (diagnostics).
pub fn intern_table_size() -> usize {
    INTERN_TABLE.with(|table| table.borrow().len())
}

impl SpnLeaf for EvidenceLeaf {
    type Coeff = NotNan<f64>;

    fn scope(&self) -> BTreeSet<ContVarName> {
        crate::for_each_evidence_family!(self, |s| s.scope().copied().collect())
    }

    fn intern(inner: SpnInner<Self>) -> Rc<SpnInner<Self>> {
        INTERN_TABLE.with(|table| {
            let mut table = table.borrow_mut();
            if let Some(existing) = table.get(&inner) {
                existing.clone()
            } else {
                let rc = Rc::new(inner);
                table.insert(rc.clone());
                rc
            }
        })
    }
}

impl EvidenceLeaf {
    /// Same-family merge of two leaves with overlapping scopes. Returns
    /// `Some((merged_leaf, log_factor))` if both leaves belong to the
    /// same family; `None` otherwise. The cross-family arm is
    /// unreachable under the SuffStat invariant that each `ContVarName`
    /// is globally typed.
    pub fn try_merge(&self, other: &EvidenceLeaf) -> Option<(EvidenceLeaf, f64)> {
        match (self, other) {
            (EvidenceLeaf::Beta(a), EvidenceLeaf::Beta(b)) => {
                let (m, f) = a.merge(b);
                Some((EvidenceLeaf::Beta(m), f))
            }
            (EvidenceLeaf::Dirichlet(a), EvidenceLeaf::Dirichlet(b)) => {
                let (m, f) = a.merge(b);
                Some((EvidenceLeaf::Dirichlet(m), f))
            }
            (EvidenceLeaf::Gamma(a), EvidenceLeaf::Gamma(b)) => {
                let (m, f) = a.merge(b);
                Some((EvidenceLeaf::Gamma(m), f))
            }
            (EvidenceLeaf::Gaussian(a), EvidenceLeaf::Gaussian(b)) => {
                let (m, f) = a.merge(b);
                Some((EvidenceLeaf::Gaussian(m), f))
            }
            _ => None,
        }
    }
}
