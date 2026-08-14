use num_traits::Num;

/// **Half-open** interval `[geq, lt)` over some numbers T; `lt = None`
/// means `[geq, ∞)`. The lower bound is inclusive, the upper bound
/// exclusive.
///
/// Half-open intervals tile the half-line exactly, so an observation
/// event and its complement *partition* the support: a boundary point
/// belongs to exactly one side, and the complement of `[a, b)` is
/// `[0, a) ∪ [b, ∞)` for integer and real T alike. Exact points are
/// deliberately not representable here — they live in
/// [`IntervalOrEq::Eq`], whose likelihood semantics are family-specific
/// (probability mass for integer draws, an ε¹ density for continuous
/// draws).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Interval<T: Ord + Copy + Num> {
    pub geq: T,
    pub lt: Option<T>,
}

impl<T: Ord + Copy + Num> Interval<T> {
    /// `[0, b)`.
    pub fn lt(b: T) -> Self {
        Interval {
            geq: T::zero(),
            lt: Some(b),
        }
    }

    /// `[b, ∞)`.
    pub fn geq(b: T) -> Self {
        Interval { geq: b, lt: None }
    }

    /// Empty iff `lt ≤ geq` (the half-open `[a, a)` is empty).
    pub fn is_empty(&self) -> bool {
        self.lt.is_some_and(|lt| lt <= self.geq)
    }

    /// `v ∈ [geq, lt)`.
    pub fn contains(&self, v: T) -> bool {
        v >= self.geq && self.lt.is_none_or(|lt| v < lt)
    }

    /// `[max(geq), min(lt))`, treating `None` as ∞.
    pub fn intersect(&self, other: &Self) -> Interval<T> {
        let lt = match (self.lt, other.lt) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };
        Interval {
            geq: self.geq.max(other.geq),
            lt,
        }
    }
}

/// A single constraint on a draw: a half-open interval or an exact
/// point. Shared by the Poisson (integer) and Exponential (real)
/// observation families.
///
/// `Eq` is kept distinct from a degenerate interval because its
/// likelihood semantics differ per family: for integer draws it is the
/// probability mass at the point, for continuous draws it is a Dirac
/// (density, ε¹). The data shape and conjunction logic are identical,
/// which is what this type captures.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum IntervalOrEq<T: Ord + Copy + Num> {
    /// `x ∈ [geq, lt)`.
    Interval(Interval<T>),
    /// `x = v` exactly.
    Eq(T),
}

impl<T: Ord + Copy + Num> IntervalOrEq<T> {
    /// `x = v`.
    pub fn eq(v: T) -> Self {
        IntervalOrEq::Eq(v)
    }

    /// `x ∈ [0, b)`.
    pub fn lt(b: T) -> Self {
        IntervalOrEq::Interval(Interval::lt(b))
    }

    /// `x ∈ [b, ∞)`.
    pub fn geq(b: T) -> Self {
        IntervalOrEq::Interval(Interval::geq(b))
    }

    pub fn is_empty(&self) -> bool {
        match self {
            IntervalOrEq::Interval(interval) => interval.is_empty(),
            IntervalOrEq::Eq(_) => false,
        }
    }

    /// `v` satisfies the constraint.
    pub fn contains(&self, v: T) -> bool {
        match self {
            IntervalOrEq::Interval(interval) => interval.contains(v),
            IntervalOrEq::Eq(e) => v == *e,
        }
    }

    /// Conjunction of two constraints on the same draw, or `None` when
    /// they contradict (empty intersection, a point outside the
    /// interval, two distinct points)
    pub fn merge(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (IntervalOrEq::Interval(i1), IntervalOrEq::Interval(i2)) => {
                let merged = i1.intersect(i2);
                (!merged.is_empty()).then_some(IntervalOrEq::Interval(merged))
            }
            (IntervalOrEq::Eq(v), IntervalOrEq::Interval(i))
            | (IntervalOrEq::Interval(i), IntervalOrEq::Eq(v)) => {
                // The interval condition is implied by x = v when v lies
                // inside it, and contradicted otherwise.
                i.contains(*v).then_some(IntervalOrEq::Eq(*v))
            }
            (IntervalOrEq::Eq(v1), IntervalOrEq::Eq(v2)) => {
                (v1 == v2).then_some(IntervalOrEq::Eq(*v1))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersect_narrows_to_singleton() {
        // k < 3 AND k ≥ 2 ⇒ k ∈ [2, 3) = {2} over the integers.
        let r = Interval::lt(3).intersect(&Interval::geq(2));
        assert_eq!(
            r,
            Interval {
                geq: 2,
                lt: Some(3)
            }
        );
        assert!(!r.is_empty());
        assert!(r.contains(2));
        assert!(!r.contains(3));
    }

    #[test]
    fn intersect_empty_detected() {
        // k < 3 AND k ≥ 3 is the empty [3, 3).
        let r = Interval::lt(3).intersect(&Interval::geq(3));
        assert!(r.is_empty());
    }

    #[test]
    fn intersect_unbounded_pair_stays_unbounded() {
        let r = Interval::geq(2).intersect(&Interval::geq(5));
        assert_eq!(r, Interval::geq(5));
    }

    #[test]
    fn boundary_belongs_to_exactly_one_side() {
        // [0, b) and [b, ∞) partition the support at b.
        let below = Interval::lt(3);
        let above = Interval::geq(3);
        assert!(!below.contains(3));
        assert!(above.contains(3));
    }

    #[test]
    fn merge_eq_inside_and_outside_interval() {
        let m = IntervalOrEq::eq(2).merge(&IntervalOrEq::geq(1));
        assert_eq!(m, Some(IntervalOrEq::eq(2)));
        // The boundary of [0, 2) is outside it.
        assert_eq!(IntervalOrEq::eq(2).merge(&IntervalOrEq::lt(2)), None);
    }

    #[test]
    fn merge_eq_eq_match_and_mismatch() {
        assert_eq!(
            IntervalOrEq::eq(5).merge(&IntervalOrEq::eq(5)),
            Some(IntervalOrEq::eq(5))
        );
        assert_eq!(IntervalOrEq::eq(5).merge(&IntervalOrEq::eq(6)), None);
    }

    #[test]
    fn merge_empty_intersection_is_none() {
        assert_eq!(IntervalOrEq::lt(2).merge(&IntervalOrEq::geq(3)), None);
    }
}
