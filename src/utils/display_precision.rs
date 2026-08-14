//! Numeric precision for posterior pretty-printing.
//!
//! By default posterior parameters print at 4 decimals and mixture-weight
//! percentages at 2 — the historical output. Setting the `PLUCK_DISPLAY_PRECISION`
//! environment variable to an integer N widens BOTH to N decimals. This is opt-in:
//! it exists to let external tooling read higher-precision output without
//! changing pluck's default human-readable format. The value is read from the
//! environment once and cached.

use std::sync::OnceLock;

/// Parsed `PLUCK_DISPLAY_PRECISION` (`None` if unset or unparsable), cached.
fn precision_override() -> Option<usize> {
    static P: OnceLock<Option<usize>> = OnceLock::new();
    *P.get_or_init(|| {
        std::env::var("PLUCK_DISPLAY_PRECISION")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
    })
}

/// Decimal places for conjugate-family parameters (Beta / Gamma / Normal /
/// Dirichlet). Default 4; overridden by `PLUCK_DISPLAY_PRECISION`.
pub fn param_precision() -> usize {
    precision_override().unwrap_or(4)
}

/// Decimal places for within-group mixture-weight percentages. Default 2;
/// overridden by `PLUCK_DISPLAY_PRECISION`.
pub fn weight_precision() -> usize {
    precision_override().unwrap_or(2)
}
