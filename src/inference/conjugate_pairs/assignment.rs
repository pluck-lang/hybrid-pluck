//! A full assignment to every continuous variable in an SPN's scope.
//!
//! Variables are globally typed (Beta xor Dirichlet xor Gamma xor
//! Gaussian), so one sorted list is kept per family. Lookup is binary
//! search by name; each accessor panics on a missing entry.

use itertools::Itertools;
use ordered_float::NotNan;
use std::collections::BTreeSet;

use crate::inference::conjugate_pairs::ContVarName;
use crate::utils::sorted_vec_by_name::lookup_sorted_by_name;

/// A full assignment over continuous variables. Partial assignments are
/// not supported. The contract is that the assignment covers every variable in the SPN's scope;
/// the accessors `panic!` otherwise.
#[derive(Debug, Clone)]
pub struct Assignment {
    /// Sorted by `ContVarName`.
    pub beta: Vec<(ContVarName, NotNan<f64>)>,
    /// Each entry is a fixed-size probability vector (length matches the
    /// variable's Dirichlet category count). Sorted by `ContVarName`.
    pub dirichlet: Vec<(ContVarName, Box<[NotNan<f64>]>)>,
    /// Sorted by `ContVarName`.
    pub gamma: Vec<(ContVarName, NotNan<f64>)>,
    /// One scalar per Gaussian variable. The SPN reassembles these into a
    /// `Vec<NotNan<f64>>` in `BTreeSet` order at each Gaussian leaf via
    /// `gaussian_vector`. Sorted by `ContVarName`.
    pub gaussian: Vec<(ContVarName, NotNan<f64>)>,
}

impl Assignment {
    /// Construct from per-family sorted lists. Debug-asserts strict
    /// monotonicity by name in each list.
    pub fn from_sorted(
        beta: Vec<(ContVarName, NotNan<f64>)>,
        dirichlet: Vec<(ContVarName, Box<[NotNan<f64>]>)>,
        gamma: Vec<(ContVarName, NotNan<f64>)>,
        gaussian: Vec<(ContVarName, NotNan<f64>)>,
    ) -> Self {
        debug_assert!(
            beta.windows(2).all(|w| w[0].0 < w[1].0),
            "Assignment::beta must be strictly increasing by name"
        );
        debug_assert!(
            dirichlet.windows(2).all(|w| w[0].0 < w[1].0),
            "Assignment::dirichlet must be strictly increasing by name"
        );
        debug_assert!(
            gamma.windows(2).all(|w| w[0].0 < w[1].0),
            "Assignment::gamma must be strictly increasing by name"
        );
        debug_assert!(
            gaussian.windows(2).all(|w| w[0].0 < w[1].0),
            "Assignment::gaussian must be strictly increasing by name"
        );
        Self {
            beta,
            dirichlet,
            gamma,
            gaussian,
        }
    }

    /// Look up the Beta value for `name`. Panics if missing.
    pub fn beta_value(&self, name: ContVarName) -> NotNan<f64> {
        lookup_sorted_by_name(&self.beta, name, |(n, _)| *n, "Assignment::beta_value").1
    }

    /// Look up the Gamma value for `name`. Panics if missing.
    pub fn gamma_value(&self, name: ContVarName) -> NotNan<f64> {
        lookup_sorted_by_name(&self.gamma, name, |(n, _)| *n, "Assignment::gamma_value").1
    }

    /// Look up the Dirichlet probability vector for `name`. Panics if missing.
    #[allow(clippy::borrowed_box)] // returned `&Box<…>` matches the
                                   // `Dirichlet::Realization = Box<[…]>` associated type used by
                                   // `SuffStat::log_likelihood`.
    pub fn dirichlet_value(&self, name: ContVarName) -> &Box<[NotNan<f64>]> {
        &lookup_sorted_by_name(
            &self.dirichlet,
            name,
            |(n, _)| *n,
            "Assignment::dirichlet_value",
        )
        .1
    }

    /// The set of every continuous-variable name covered by this assignment.
    pub fn scope(&self) -> BTreeSet<ContVarName> {
        let mut s = BTreeSet::new();
        s.extend(self.beta.iter().map(|(n, _)| *n));
        s.extend(self.dirichlet.iter().map(|(n, _)| *n));
        s.extend(self.gamma.iter().map(|(n, _)| *n));
        s.extend(self.gaussian.iter().map(|(n, _)| *n));
        s
    }

    /// Combine two disjoint assignments into one. Caller must ensure the
    /// scopes are disjoint; this is debug-asserted.
    pub fn merge(&self, other: &Self) -> Self {
        debug_assert!(
            self.scope().is_disjoint(&other.scope()),
            "Assignment::merge: scopes overlap"
        );
        let beta = self
            .beta
            .iter()
            .merge_by(other.beta.iter(), |a, b| a.0 <= b.0)
            .cloned()
            .collect();
        let dirichlet = self
            .dirichlet
            .iter()
            .merge_by(other.dirichlet.iter(), |a, b| a.0 <= b.0)
            .cloned()
            .collect();
        let gamma = self
            .gamma
            .iter()
            .merge_by(other.gamma.iter(), |a, b| a.0 <= b.0)
            .cloned()
            .collect();
        let gaussian = self
            .gaussian
            .iter()
            .merge_by(other.gaussian.iter(), |a, b| a.0 <= b.0)
            .cloned()
            .collect();
        Assignment::from_sorted(beta, dirichlet, gamma, gaussian)
    }

    /// Collect the Gaussian realisation for the variables in `scope`,
    /// returned in `BTreeSet` ascending order (matching
    /// `EvidenceLeaf::scope` iteration order). Panics on any missing
    /// variable.
    pub fn gaussian_vector(&self, scope: &BTreeSet<ContVarName>) -> Vec<NotNan<f64>> {
        scope
            .iter()
            .map(
                |name| match self.gaussian.binary_search_by_key(name, |(n, _)| *n) {
                    Ok(i) => self.gaussian[i].1,
                    Err(_) => panic!(
                        "Assignment::gaussian_vector: variable {} not assigned",
                        name
                    ),
                },
            )
            .collect()
    }
}

/// `Assignment` builder used by sampling. Pushes are
/// per-family and unordered; `finalize` sorts and constructs the
/// canonical `Assignment`. Disjoint scopes across pushes are a caller
/// invariant — `Assignment::from_sorted` debug-asserts strict
/// monotonicity at finalize.
#[derive(Debug, Default)]
pub struct AssignmentBuilder {
    beta: Vec<(ContVarName, NotNan<f64>)>,
    dirichlet: Vec<(ContVarName, Box<[NotNan<f64>]>)>,
    gamma: Vec<(ContVarName, NotNan<f64>)>,
    gaussian: Vec<(ContVarName, NotNan<f64>)>,
}

impl AssignmentBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_beta(&mut self, name: ContVarName, value: NotNan<f64>) {
        self.beta.push((name, value));
    }

    pub fn push_dirichlet(&mut self, name: ContVarName, value: Box<[NotNan<f64>]>) {
        self.dirichlet.push((name, value));
    }

    pub fn push_gamma(&mut self, name: ContVarName, value: NotNan<f64>) {
        self.gamma.push((name, value));
    }

    pub fn push_gaussian(&mut self, name: ContVarName, value: NotNan<f64>) {
        self.gaussian.push((name, value));
    }

    pub fn finalize(mut self) -> Assignment {
        self.beta.sort_by_key(|(n, _)| *n);
        self.dirichlet.sort_by_key(|(n, _)| *n);
        self.gamma.sort_by_key(|(n, _)| *n);
        self.gaussian.sort_by_key(|(n, _)| *n);
        Assignment::from_sorted(self.beta, self.dirichlet, self.gamma, self.gaussian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nn(x: f64) -> NotNan<f64> {
        NotNan::new(x).unwrap()
    }

    #[test]
    fn beta_lookup_roundtrips() {
        let a = Assignment::from_sorted(vec![(1, nn(0.2)), (3, nn(0.7))], vec![], vec![], vec![]);
        assert_eq!(a.beta_value(1), nn(0.2));
        assert_eq!(a.beta_value(3), nn(0.7));
    }

    #[test]
    #[should_panic(expected = "variable 2 not found")]
    fn beta_missing_panics() {
        let a = Assignment::from_sorted(vec![(1, nn(0.2))], vec![], vec![], vec![]);
        let _ = a.beta_value(2);
    }

    #[test]
    fn dirichlet_lookup_roundtrips() {
        let v: Box<[NotNan<f64>]> = vec![nn(0.3), nn(0.7)].into_boxed_slice();
        let a = Assignment::from_sorted(vec![], vec![(5, v.clone())], vec![], vec![]);
        assert_eq!(a.dirichlet_value(5), &v);
    }

    #[test]
    fn gamma_lookup_roundtrips() {
        let a = Assignment::from_sorted(vec![], vec![], vec![(2, nn(1.5))], vec![]);
        assert_eq!(a.gamma_value(2), nn(1.5));
    }

    #[test]
    #[should_panic(expected = "variable 6 not found")]
    fn dirichlet_missing_panics() {
        let a = Assignment::from_sorted(
            vec![],
            vec![(5, vec![nn(1.0)].into_boxed_slice())],
            vec![],
            vec![],
        );
        let _ = a.dirichlet_value(6);
    }

    #[test]
    fn gaussian_vector_in_scope_order() {
        let a = Assignment::from_sorted(
            vec![],
            vec![],
            vec![],
            vec![(1, nn(10.0)), (3, nn(30.0)), (7, nn(70.0))],
        );
        // BTreeSet iterates in ascending order regardless of how the
        // set is constructed.
        let scope: BTreeSet<ContVarName> = [7u64, 1, 3].iter().copied().collect();
        let v = a.gaussian_vector(&scope);
        assert_eq!(v, vec![nn(10.0), nn(30.0), nn(70.0)]);
    }

    #[test]
    #[should_panic(expected = "variable 2 not assigned")]
    fn gaussian_missing_panics() {
        let a = Assignment::from_sorted(vec![], vec![], vec![], vec![(1, nn(10.0))]);
        let scope: BTreeSet<ContVarName> = [1u64, 2].iter().copied().collect();
        let _ = a.gaussian_vector(&scope);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "strictly increasing")]
    fn from_sorted_rejects_unsorted_input() {
        let _ = Assignment::from_sorted(vec![(3, nn(0.2)), (1, nn(0.5))], vec![], vec![], vec![]);
    }
}
