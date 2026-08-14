//! Shared file discovery for the `.pluck` test harnesses.
//!
//! Both harnesses set `harness = false` and enumerate their inputs here at
//! runtime, so no `.pluck` directory is a build-time input and editing one
//! recompiles nothing.

use std::fs;
use std::path::{Path, PathBuf};

use libtest_mimic::{Failed, Trial};

/// One [`Trial`] per `*.pluck` file directly inside `dir`, named after the file
/// stem. Paths are sorted so the run order is stable.
pub fn pluck_trials<F>(dir: &Path, runner: F) -> Vec<Trial>
where
    F: Fn(PathBuf, String) -> Result<(), Failed> + Copy + Send + 'static,
{
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", dir.display(), e))
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("pluck"))
        .collect();
    paths.sort();

    paths
        .into_iter()
        .map(|path| {
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_else(|| panic!("no stem for {}", path.display()))
                .to_string();
            Trial::test(name.clone(), move || runner(path, name))
        })
        .collect()
}
