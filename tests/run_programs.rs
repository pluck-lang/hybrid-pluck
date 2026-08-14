//! One test per `tests/programs/*.pluck`, each compared against its
//! `.expected.json` snapshot. Refresh snapshots with `UPDATE_SNAPSHOTS=1`.

mod common;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::{env, fs};

use libtest_mimic::{Arguments, Failed};
use pluck::{
    BetaPrior, DirichletPrior, GammaPrior, GaussianPrior, JointGammaPrior, MixtureItem,
    PluckContext, PosteriorPacket, QueryResult, ResultKind,
};

fn main() -> ExitCode {
    let args = Arguments::from_args();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/programs");
    let tests = common::pluck_trials(&dir, run_program_snapshot);

    libtest_mimic::run(&args, tests).exit_code()
}

const ABS_TOL: f64 = 1e-9;
const REL_TOL: f64 = 1e-9;

fn approx_eq(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if !a.is_finite() || !b.is_finite() {
        return a == b;
    }
    let diff = (a - b).abs();
    diff <= ABS_TOL || diff <= REL_TOL * a.abs().max(b.abs())
}

fn fmt_mismatch(
    path: &str,
    field: &str,
    expected: impl std::fmt::Display,
    actual: impl std::fmt::Display,
) -> String {
    format!("{}/{}: expected {}, got {}", path, field, expected, actual)
}

fn cmp_f64(path: &str, field: &str, e: f64, a: f64) -> Result<(), String> {
    if approx_eq(e, a) {
        Ok(())
    } else {
        let diff = (e - a).abs();
        Err(format!(
            "{}/{}: expected {}, got {} (diff {:e})",
            path, field, e, a, diff
        ))
    }
}

fn cmp_vec_f64(path: &str, field: &str, e: &[f64], a: &[f64]) -> Result<(), String> {
    if e.len() != a.len() {
        return Err(fmt_mismatch(
            path,
            &format!("{}.len", field),
            e.len(),
            a.len(),
        ));
    }
    for (i, (ev, av)) in e.iter().zip(a.iter()).enumerate() {
        cmp_f64(path, &format!("{}[{}]", field, i), *ev, *av)?;
    }
    Ok(())
}

fn cmp_beta(path: &str, e: &BetaPrior<String>, a: &BetaPrior<String>) -> Result<(), String> {
    match (e, a) {
        (
            BetaPrior::Beta {
                name: en,
                a: ea,
                b: eb,
            },
            BetaPrior::Beta {
                name: an,
                a: aa,
                b: ab,
            },
        ) => {
            if en != an {
                return Err(fmt_mismatch(path, "beta.name", en, an));
            }
            cmp_f64(path, "beta.a", *ea, *aa)?;
            cmp_f64(path, "beta.b", *eb, *ab)
        }
        (
            BetaPrior::Pinned {
                name: en,
                value: ev,
            },
            BetaPrior::Pinned {
                name: an,
                value: av,
            },
        ) => {
            if en != an {
                return Err(fmt_mismatch(path, "beta.name", en, an));
            }
            cmp_f64(path, "beta.value", *ev, *av)
        }
        (e, a) => Err(fmt_mismatch(
            path,
            "beta.variant",
            beta_variant(e),
            beta_variant(a),
        )),
    }
}

fn beta_variant(p: &BetaPrior<String>) -> &'static str {
    match p {
        BetaPrior::Beta { .. } => "Beta",
        BetaPrior::Pinned { .. } => "Pinned",
    }
}

fn cmp_dirichlet(
    path: &str,
    e: &DirichletPrior<String>,
    a: &DirichletPrior<String>,
) -> Result<(), String> {
    if e.name() != a.name() {
        return Err(fmt_mismatch(path, "dirichlet.name", e.name(), a.name()));
    }
    match (e, a) {
        (
            DirichletPrior::Dirichlet { alphas: ea, .. },
            DirichletPrior::Dirichlet { alphas: aa, .. },
        ) => cmp_vec_f64(path, "dirichlet.alphas", ea, aa),
        (DirichletPrior::Pinned { value: ev, .. }, DirichletPrior::Pinned { value: av, .. }) => {
            cmp_vec_f64(path, "dirichlet.value", ev, av)
        }
        _ => Err(fmt_mismatch(
            path,
            "dirichlet.kind",
            format!("{:?}", std::mem::discriminant(e)),
            format!("{:?}", std::mem::discriminant(a)),
        )),
    }
}

fn cmp_gaussian(
    path: &str,
    e: &GaussianPrior<String>,
    a: &GaussianPrior<String>,
) -> Result<(), String> {
    if e.var_order != a.var_order {
        return Err(fmt_mismatch(
            path,
            "gaussian.var_order",
            format!("{:?}", e.var_order),
            format!("{:?}", a.var_order),
        ));
    }
    let e_mean: Vec<f64> = e.mean.iter().copied().collect();
    let a_mean: Vec<f64> = a.mean.iter().copied().collect();
    cmp_vec_f64(path, "gaussian.mean", &e_mean, &a_mean)?;
    if e.cov.shape() != a.cov.shape() {
        return Err(fmt_mismatch(
            path,
            "gaussian.cov.shape",
            format!("{:?}", e.cov.shape()),
            format!("{:?}", a.cov.shape()),
        ));
    }
    let (rows, cols) = e.cov.shape();
    for i in 0..rows {
        for j in 0..cols {
            cmp_f64(
                path,
                &format!("gaussian.cov[{},{}]", i, j),
                e.cov[(i, j)],
                a.cov[(i, j)],
            )?;
        }
    }
    Ok(())
}

fn cmp_gamma_prior(
    path: &str,
    e: &GammaPrior<String>,
    a: &GammaPrior<String>,
) -> Result<(), String> {
    match (e, a) {
        (
            GammaPrior::Gamma {
                name: en,
                shape: es,
                rate: er,
            },
            GammaPrior::Gamma {
                name: an,
                shape: as_,
                rate: ar,
            },
        ) => {
            if en != an {
                return Err(fmt_mismatch(path, "gamma.name", en, an));
            }
            cmp_f64(path, "gamma.shape", *es, *as_)?;
            cmp_f64(path, "gamma.rate", *er, *ar)
        }
        (
            GammaPrior::Pinned {
                name: en,
                value: ev,
            },
            GammaPrior::Pinned {
                name: an,
                value: av,
            },
        ) => {
            if en != an {
                return Err(fmt_mismatch(path, "gamma.name", en, an));
            }
            cmp_f64(path, "gamma.value", *ev, *av)
        }
        (
            GammaPrior::Mixture {
                name: en,
                mixture: em,
            },
            GammaPrior::Mixture {
                name: an,
                mixture: am,
            },
        ) => {
            if en != an {
                return Err(fmt_mismatch(path, "gamma.name", en, an));
            }
            if em.0.len() != am.0.len() {
                return Err(fmt_mismatch(path, "mixture.len", em.0.len(), am.0.len()));
            }
            for (i, ((et, ew), (at, aw))) in em.0.iter().zip(am.0.iter()).enumerate() {
                cmp_f64(path, &format!("mixture[{i}].n"), et.n(), at.n())?;
                cmp_f64(path, &format!("mixture[{i}].c"), et.c(), at.c())?;
                cmp_f64(path, &format!("mixture[{i}].log_w"), ew.log_w(), aw.log_w())?;
                if ew.sign() != aw.sign() {
                    return Err(fmt_mismatch(
                        path,
                        &format!("mixture[{i}].sign"),
                        ew.sign(),
                        aw.sign(),
                    ));
                }
            }
            Ok(())
        }
        _ => Err(fmt_mismatch(
            path,
            "gamma.variant",
            format!("{:?}", e),
            format!("{:?}", a),
        )),
    }
}

fn cmp_joint_gamma(
    path: &str,
    e: &JointGammaPrior<String>,
    a: &JointGammaPrior<String>,
) -> Result<(), String> {
    cmp_gamma_prior(path, &e.gamma, &a.gamma)?;
    // Draw families: names + truncation constraints. The constraints are
    // exact data (integer intervals / program-literal real bounds that
    // roundtrip through JSON losslessly), so structural equality is the
    // right comparison.
    if e.poisson.constraints() != a.poisson.constraints() {
        return Err(fmt_mismatch(
            path,
            "poisson draws",
            format!("{:?}", e.poisson.constraints()),
            format!("{:?}", a.poisson.constraints()),
        ));
    }
    if e.exponential.constraints() != a.exponential.constraints() {
        return Err(fmt_mismatch(
            path,
            "exponential draws",
            format!("{:?}", e.exponential.constraints()),
            format!("{:?}", a.exponential.constraints()),
        ));
    }
    Ok(())
}

fn cmp_packet(path: &str, e: &PosteriorPacket, a: &PosteriorPacket) -> Result<(), String> {
    if e.beta.len() != a.beta.len() {
        return Err(fmt_mismatch(path, "betas.len", e.beta.len(), a.beta.len()));
    }
    for (i, (eb, ab)) in e.beta.iter().zip(a.beta.iter()).enumerate() {
        cmp_beta(&format!("{}/betas[{}]", path, i), eb, ab)?;
    }
    if e.dirichlet.len() != a.dirichlet.len() {
        return Err(fmt_mismatch(
            path,
            "dirichlets.len",
            e.dirichlet.len(),
            a.dirichlet.len(),
        ));
    }
    for (i, (ed, ad)) in e.dirichlet.iter().zip(a.dirichlet.iter()).enumerate() {
        cmp_dirichlet(&format!("{}/dirichlets[{}]", path, i), ed, ad)?;
    }
    if e.gaussian.len() != a.gaussian.len() {
        return Err(fmt_mismatch(
            path,
            "gaussians.len",
            e.gaussian.len(),
            a.gaussian.len(),
        ));
    }
    for (i, (eg, ag)) in e.gaussian.iter().zip(a.gaussian.iter()).enumerate() {
        cmp_gaussian(&format!("{}/gaussians[{}]", path, i), eg, ag)?;
    }
    if e.gamma.len() != a.gamma.len() {
        return Err(fmt_mismatch(
            path,
            "gammas.len",
            e.gamma.len(),
            a.gamma.len(),
        ));
    }
    for (i, (eg, ag)) in e.gamma.iter().zip(a.gamma.iter()).enumerate() {
        cmp_joint_gamma(&format!("{}/gammas[{}]", path, i), eg, ag)?;
    }
    Ok(())
}

fn cmp_item(path: &str, e: &MixtureItem, a: &MixtureItem) -> Result<(), String> {
    if e.display != a.display {
        return Err(fmt_mismatch(path, "display", &e.display, &a.display));
    }
    cmp_f64(
        path,
        "log_probability",
        e.log_probability,
        a.log_probability,
    )?;
    cmp_packet(
        &format!("{}/posteriors", path),
        &e.posteriors,
        &a.posteriors,
    )
}

fn cmp_kind(path: &str, e: &ResultKind, a: &ResultKind) -> Result<(), String> {
    match (e, a) {
        (ResultKind::Distribution { items: ei }, ResultKind::Distribution { items: ai }) => {
            if ei.len() != ai.len() {
                return Err(fmt_mismatch(path, "items.len", ei.len(), ai.len()));
            }
            for (i, (e_item, a_item)) in ei.iter().zip(ai.iter()).enumerate() {
                cmp_item(&format!("{}/items[{}]", path, i), e_item, a_item)?;
            }
            Ok(())
        }
        (ResultKind::Samples { values: ev }, ResultKind::Samples { values: av }) => {
            if ev.len() != av.len() {
                return Err(fmt_mismatch(path, "samples.len", ev.len(), av.len()));
            }
            for (i, (e_v, a_v)) in ev.iter().zip(av.iter()).enumerate() {
                if e_v != a_v {
                    return Err(fmt_mismatch(
                        &format!("{}/samples[{}]", path, i),
                        "value",
                        e_v,
                        a_v,
                    ));
                }
            }
            Ok(())
        }
        _ => Err(fmt_mismatch(path, "kind", kind_variant(e), kind_variant(a))),
    }
}

fn kind_variant(k: &ResultKind) -> &'static str {
    match k {
        ResultKind::Distribution { .. } => "Distribution",
        ResultKind::Samples { .. } => "Samples",
    }
}

fn cmp_query_result(path: &str, e: &QueryResult, a: &QueryResult) -> Result<(), String> {
    if e.query != a.query {
        return Err(fmt_mismatch(path, "query", &e.query, &a.query));
    }
    cmp_kind(path, &e.kind, &a.kind)
}

fn cmp_query_results(
    name: &str,
    expected: &[QueryResult],
    actual: &[QueryResult],
) -> Result<(), String> {
    if expected.len() != actual.len() {
        return Err(format!(
            "{}: expected {} results, got {}",
            name,
            expected.len(),
            actual.len()
        ));
    }
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        cmp_query_result(&format!("{}[{}]", name, i), e, a)?;
    }
    Ok(())
}

fn run_program(path: &Path) -> Result<Vec<QueryResult>, Failed> {
    let source = fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    let mut ctx = PluckContext::new();
    Ok(ctx.run(&source))
}

fn run_program_snapshot(pluck_path: PathBuf, name: String) -> Result<(), Failed> {
    let snap_path = pluck_path.with_extension("expected.json");
    let update = env::var("UPDATE_SNAPSHOTS").ok().as_deref() == Some("1");

    let actual = run_program(&pluck_path)?;

    // Every failure below — missing, malformed, or stale snapshot — is fixed by
    // rewriting the file, so under UPDATE_SNAPSHOTS they share one path.
    let mismatch = match fs::read_to_string(&snap_path) {
        Err(_) => format!("{name}: missing .expected.json (refresh with UPDATE_SNAPSHOTS=1)"),
        Ok(text) => match serde_json::from_str::<Vec<QueryResult>>(&text) {
            Err(e) => format!("malformed snapshot {}: {}", snap_path.display(), e),
            Ok(expected) => match cmp_query_results(&name, &expected, &actual) {
                Ok(()) => return Ok(()),
                // Skip formatting the full dump when it is only going to be discarded.
                Err(diff) if update => diff,
                Err(diff) => format!(
                    "{diff}\n\nRefresh with UPDATE_SNAPSHOTS=1 cargo test\n\nExpected:\n{expected:?}\n\nGot:\n{actual:?}"
                ),
            },
        },
    };

    if update {
        write_snapshot(&snap_path, &actual)
    } else {
        Err(mismatch.into())
    }
}

fn write_snapshot(path: &Path, results: &[QueryResult]) -> Result<(), Failed> {
    let json = serde_json::to_string_pretty(results)
        .map_err(|e| format!("failed to serialize results for {}: {}", path.display(), e))?;
    fs::write(path, json).map_err(|e| format!("failed to write {}: {}", path.display(), e))?;
    Ok(())
}
