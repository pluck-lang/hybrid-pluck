pub mod coeff;
pub mod evidence;
pub mod node;
pub mod posterior;
pub mod sampling;

pub use coeff::SpnLeaf;
pub use node::{Spn, SpnKind};
pub use posterior::{posterior_leaf, PosteriorSpn};
