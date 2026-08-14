// ---------------------------------------------------------------------------
// RealEps: real numbers with infinitesimal tracking
// ---------------------------------------------------------------------------
use std::ops;

use super::math::logsumexp;

/// A value of the form `exp(log_coeff) * eps^power`, used for marginal probability
/// computation when disintegration (prob=?) is involved.
/// Stored in log-domain for numerical stability.
#[derive(Debug, Clone, Copy)]
pub struct RealEps {
    pub log_coeff: f64,
    pub power: u32,
}

impl RealEps {
    pub fn zero() -> Self {
        RealEps {
            log_coeff: f64::NEG_INFINITY,
            power: 0,
        }
    }

    pub fn scalar(v: f64) -> Self {
        RealEps {
            log_coeff: if v <= 0.0 { f64::NEG_INFINITY } else { v.ln() },
            power: 0,
        }
    }

    /// TODO: enforce not-NaN SPN weights here (e.g. `debug_assert!(!log_coeff
    /// .is_nan())`). A NaN coefficient silently poisons every downstream
    /// `logsumexp`/product (`NaN + x = NaN`) and corrupts the whole marginal —
    /// it should be caught at construction. (The `NotNan<f64>` coefficient used
    /// by evidence-side SPNs already panics on NaN; `RealEps` posterior weights
    /// have no such guard yet.)
    pub fn from_log(log_coeff: f64, power: u32) -> Self {
        RealEps { log_coeff, power }
    }

    pub fn is_zero(&self) -> bool {
        self.log_coeff == f64::NEG_INFINITY
    }

    /// Multiply by a log-domain scalar (no change to epsilon power).
    pub fn scale_log(&self, log_s: f64) -> Self {
        RealEps {
            log_coeff: self.log_coeff + log_s,
            power: self.power,
        }
    }
}

impl ops::Add for RealEps {
    type Output = RealEps;
    fn add(self, rhs: RealEps) -> RealEps {
        if self.is_zero() {
            return rhs;
        }
        if rhs.is_zero() {
            return self;
        }
        if self.power == rhs.power {
            RealEps {
                log_coeff: logsumexp(self.log_coeff, rhs.log_coeff),
                power: self.power,
            }
        } else if self.power < rhs.power {
            self
        } else {
            rhs
        }
    }
}

impl ops::Mul for RealEps {
    type Output = RealEps;
    fn mul(self, rhs: RealEps) -> RealEps {
        RealEps {
            log_coeff: self.log_coeff + rhs.log_coeff,
            power: self.power + rhs.power,
        }
    }
}

impl ops::Div for RealEps {
    type Output = RealEps;
    fn div(self, rhs: RealEps) -> RealEps {
        RealEps {
            log_coeff: self.log_coeff - rhs.log_coeff,
            power: self.power - rhs.power,
        }
    }
}
