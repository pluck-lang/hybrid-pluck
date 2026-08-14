//! Generic helpers for the strict-increasing-by-name sorted-vec
//! invariant shared by the Beta and Dirichlet sections of
//! `PriorRegistry` (and the equivalent fields on `Assignment`).
//!
//! Gaussian deliberately does *not* use these helpers — it owns
//! multi-variable blocks and looks up by scope, not by a single name.

use std::collections::BTreeSet;

/// Append `item` to a sorted vec, preserving the
/// strict-increasing-by-name invariant with distinct names.
///
/// Idempotent on same-content re-registration: if a prior equal to
/// `item` is already present (matching name AND params), this is a
/// no-op. The callstack-keyed naming guarantees same callstack ⇒ same
/// params, so when Gibbs re-compiles a deterministic prior-creating
/// expression for `compile_pin` it always hits this idempotent path.
pub(crate) fn add_to_sorted_by_name<P, K, F>(vec: &mut Vec<P>, item: P, key_of: F, label: &str)
where
    K: Ord + Copy + std::fmt::Display,
    P: PartialEq,
    F: Fn(&P) -> K,
{
    let new_key = key_of(&item);
    if let Some(existing) = vec.iter().find(|p| key_of(p) == new_key) {
        debug_assert!(
            existing == &item,
            "{}: re-registering name {} with different params",
            label,
            new_key
        );
        return;
    }
    debug_assert!(
        vec.iter().map(&key_of).is_sorted(),
        "{}: existing vec is not sorted by name",
        label
    );
    debug_assert!(
        vec.iter().map(&key_of).max().is_none_or(|m| m < new_key),
        "{}: new name {} is not larger than every existing name",
        label,
        new_key
    );
    vec.push(item);
}

/// Binary-search a sorted vec by name. Panics with `label` on miss.
pub(crate) fn lookup_sorted_by_name<'a, P, K, F>(
    vec: &'a [P],
    name: K,
    key_of: F,
    label: &str,
) -> &'a P
where
    K: Ord + Copy + std::fmt::Display,
    F: Fn(&P) -> K,
{
    match vec.binary_search_by_key(&name, &key_of) {
        Ok(i) => &vec[i],
        Err(_) => panic!("{}: variable {} not found", label, name),
    }
}

/// Filter a sorted vec to entries whose key is in `scope`. Preserves
/// order.
pub(crate) fn slice_sorted_by_name<P, K, F>(vec: &[P], scope: &BTreeSet<K>, key_of: F) -> Vec<P>
where
    K: Ord + Copy,
    P: Clone,
    F: Fn(&P) -> K,
{
    vec.iter()
        .filter(|p| scope.contains(&key_of(p)))
        .cloned()
        .collect()
}
