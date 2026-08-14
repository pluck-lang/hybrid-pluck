//! Top-down sampling from a `PosteriorSpn`.
//!
//! At each node:
//! - `Sum(children)`: pick one child proportional to its normalised
//!   `log_coeff`. Sum normalisation (see `Spn::sum`) ensures surviving
//!   children share the same epsilon power, so a plain log-weight
//!   categorical draw suffices.
//! - `Product(children)`: recurse independently into each child. Children
//!   have disjoint scopes by the Product invariant; the per-child
//!   assignments concatenate without collision.
//! - `Leaf(L)`: dispatch to `Prior::sample` on the underlying prior.
//!   Per-family logic lives in `beta.rs`, `dirichlet.rs`, `gaussian.rs`;
//!   numeric primitives live in `crate::utils::sampling`.

use rand::Rng;

use super::node::SpnKind;
use super::posterior::{PosteriorLeaf, PosteriorSpn};
use crate::inference::conjugate_pairs::{Assignment, AssignmentBuilder, Prior};

impl PosteriorSpn {
    /// Draw a posterior sample by top-down traversal. The returned
    /// `Assignment` covers the leaves along the sampled branches — a
    /// proper subset of `self.scope()` when Sum children have unequal
    /// scopes (mixture branches observing different variables). Callers
    /// needing full coverage fill the rest themselves: registered
    /// variables from the registry fallback
    /// (`sample_continuous_posterior`), gamma draws from
    /// `LazyKCState::realize_missing_gamma_draws`.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Assignment {
        // Top-down traversal visits leaves in pointer-canonical order, not
        // name order; `AssignmentBuilder::finalize` sorts before constructing
        // the `Assignment` (which debug-asserts strict monotonicity).
        let mut builder = AssignmentBuilder::new();
        sample_into(self, rng, &mut builder);
        builder.finalize()
    }
}

fn sample_into<R: Rng>(node: &PosteriorSpn, rng: &mut R, out: &mut AssignmentBuilder) {
    match node.kind() {
        SpnKind::Scalar => {}
        SpnKind::Leaf(l) => sample_leaf(l, rng, out),
        SpnKind::Sum(children) => {
            let log_weights: Vec<f64> = children.iter().map(|c| c.log_coeff().log_coeff).collect();
            let idx = crate::utils::sampling::pick_categorical_log(rng, &log_weights);
            sample_into(&children[idx], rng, out);
        }
        SpnKind::Product(children) => {
            for c in children {
                sample_into(c, rng, out);
            }
        }
    }
}

fn sample_leaf<R: Rng>(leaf: &PosteriorLeaf, rng: &mut R, out: &mut AssignmentBuilder) {
    crate::for_each_posterior_family!(leaf, |p| {
        let sampled = p.sample(rng);
        p.push_into(sampled, out);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::conjugate_pairs::beta::BetaPrior;
    use crate::inference::conjugate_pairs::ContVarName;
    use crate::inference::spn::posterior::posterior_leaf;
    use crate::utils::epsilon::RealEps;

    use rand::SeedableRng;

    fn rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    fn re(log_coeff: f64, power: u32) -> RealEps {
        RealEps::from_log(log_coeff, power)
    }

    fn beta_leaf_at(name: ContVarName, a: f64, b: f64, coeff: RealEps) -> PosteriorSpn {
        posterior_leaf(coeff, PosteriorLeaf::Beta(BetaPrior::Beta { name, a, b }))
    }

    #[test]
    fn sample_product_concatenates() {
        let l1 = beta_leaf_at(1, 1.0, 1.0, re(0.0, 0));
        let l2 = beta_leaf_at(2, 1.0, 1.0, re(0.0, 0));
        let p = PosteriorSpn::product(vec![l1, l2]);
        let mut r = rng();
        let a = p.sample(&mut r);
        assert_eq!(a.beta.len(), 2);
        assert_eq!(a.beta[0].0, 1);
        assert_eq!(a.beta[1].0, 2);
    }

    #[test]
    fn sample_sum_chooses_one_branch() {
        // Disjoint-scope branches so picking one yields a single Beta in
        // the assignment.
        let l1 = beta_leaf_at(1, 1.0, 1.0, re(0.0, 0));
        let l2 = beta_leaf_at(2, 1.0, 1.0, re(0.0, 0));
        let s = PosteriorSpn::sum(vec![l1, l2]);
        let mut r = rng();
        let a = s.sample(&mut r);
        assert_eq!(a.beta.len(), 1);
        let chosen = a.beta[0].0;
        assert!(chosen == 1 || chosen == 2);
    }
}
