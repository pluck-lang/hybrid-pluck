use ordered_float::NotNan;

/// A draw constraint paired with the positive scale `s` on its draw's rate:
/// the draw is distributed with effective rate `s · λ` (a *scaled gamma*).
///
/// The scale travels *with* the constraint through every map transform
/// (`merge`, `conditioned_on`, `map_names`, `restricted_to`, …), so the
/// Exponential/Poisson families never have to keep a parallel `scale` map in
/// sync with their constraint maps. An unscaled draw has `scale == 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Scaled<C> {
    pub constraint: C,
    pub scale: NotNan<f64>,
}

impl<C> Scaled<C> {
    pub fn new(constraint: C, scale: NotNan<f64>) -> Self {
        Scaled { constraint, scale }
    }

    /// Wrap a constraint at unit scale (an unscaled draw).
    pub fn unit(constraint: C) -> Self {
        Scaled {
            constraint,
            scale: NotNan::new(1.0).unwrap(),
        }
    }

    /// Apply `f` to the inner constraint, preserving the scale.
    pub fn map<D>(self, f: impl FnOnce(C) -> D) -> Scaled<D> {
        Scaled {
            constraint: f(self.constraint),
            scale: self.scale,
        }
    }
}
