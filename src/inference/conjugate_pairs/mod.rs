pub mod assignment;
pub mod beta;
pub mod dirichlet;
pub mod evidence_leaf;
pub mod exponential;
pub mod gamma;
pub mod gamma_mixture;
pub mod gaussian;
pub mod poisson;
pub mod posterior_leaf;
pub mod registry;
pub mod scaled;
pub mod suffstat;

pub use assignment::{Assignment, AssignmentBuilder};
pub use beta::{BetaFlip, BetaPrior, BetaSuffStat};
pub use dirichlet::{DirichletFlip, DirichletPrior, DirichletSuffStat};
pub use evidence_leaf::EvidenceLeaf;
pub use exponential::ExponentialObs;
pub use gamma::{GammaDrawFamily, GammaFlip, GammaPrior, GammaSuffStat, JointGammaPrior};
pub use gaussian::{GaussianAffineExpr, GaussianFlip, GaussianObs, GaussianPrior};
pub use poisson::PoissonObs;
pub use posterior_leaf::PosteriorLeaf;
pub use registry::{PriorRegistry, RegistryBuilder, RegistryInsert};
pub use scaled::Scaled;
pub use suffstat::{ContVarName, Prior, SuffStat};

/// Dispatch `$body` over each variant of `EvidenceLeaf`, binding the
/// inner per-family `SuffStat` to `$s`. Adding a new family means one
/// new arm here.
#[macro_export]
macro_rules! for_each_evidence_family {
    ($leaf:expr, |$s:ident| $body:expr) => {
        match $leaf {
            $crate::inference::conjugate_pairs::EvidenceLeaf::Beta($s) => $body,
            $crate::inference::conjugate_pairs::EvidenceLeaf::Dirichlet($s) => $body,
            $crate::inference::conjugate_pairs::EvidenceLeaf::Gamma($s) => $body,
            $crate::inference::conjugate_pairs::EvidenceLeaf::Gaussian($s) => $body,
        }
    };
}

/// Dispatch `$body` over each variant of `PosteriorLeaf`, binding the
/// inner per-family prior to `$p`. Adding a new family means one new
/// arm here.
#[macro_export]
macro_rules! for_each_posterior_family {
    ($leaf:expr, |$p:ident| $body:expr) => {
        match $leaf {
            $crate::inference::conjugate_pairs::PosteriorLeaf::Beta($p) => $body,
            $crate::inference::conjugate_pairs::PosteriorLeaf::Dirichlet($p) => $body,
            $crate::inference::conjugate_pairs::PosteriorLeaf::Gamma($p) => $body,
            $crate::inference::conjugate_pairs::PosteriorLeaf::Gaussian($p) => $body,
        }
    };
}
