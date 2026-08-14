//! One test per `examples/*.pluck`: the example must run without panicking and
//! produce a non-empty result for every query.

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use libtest_mimic::{Arguments, Failed};
use pluck::{PluckContext, ResultKind};

/// Examples recurse deeply; the default test-thread stack is not enough.
const STACK_SIZE: usize = 1024 * 1024 * 1024;

fn main() -> ExitCode {
    let args = Arguments::from_args();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples");
    let tests = common::pluck_trials(&dir, run_example);

    libtest_mimic::run(&args, tests).exit_code()
}

fn result_is_nonempty(kind: &ResultKind) -> bool {
    match kind {
        ResultKind::Distribution { items } => !items.is_empty(),
        ResultKind::Samples { values } => !values.is_empty(),
    }
}

fn run_example(path: PathBuf, name: String) -> Result<(), Failed> {
    // The failure is returned through the join rather than panicking, so the
    // message survives as the trial's failure text.
    std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(move || check_example(&path, &name))
        .expect("spawn worker thread")
        .join()
        .map_err(|_| Failed::from("worker thread panicked"))?
}

fn check_example(path: &Path, name: &str) -> Result<(), Failed> {
    let source = fs::read_to_string(path)
        .map_err(|e| format!("{}: failed to read {}: {}", name, path.display(), e))?;

    let mut ctx = PluckContext::new();
    let results = ctx.run(&source);

    if results.is_empty() {
        return Err(format!("{name}: produced no query results").into());
    }
    for r in &results {
        if !result_is_nonempty(&r.kind) {
            return Err(format!("{name}: query `{}` produced an empty result", r.query).into());
        }
    }
    Ok(())
}
