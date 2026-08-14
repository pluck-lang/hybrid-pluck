//! Display name generation for values returned in a query

use std::collections::HashMap;
use std::rc::Rc;

use crate::discrete_factorizations::{BooleanFunction, BooleanFunctionOps};
use crate::inference::conjugate_pairs::{ContVarName, GaussianAffineExpr};
use crate::language::types::{StringInterner, TypeRegistry};
use crate::language::values::{extract_nat, PluckVal, PluckValue};
use crate::{Path, ResultPositions};

/// Display names for continuous-variable leaves in a query distribution.
///
/// `var_names`: one entry per underlying `ContVarName`. Beta,
/// Dirichlet, and Gamma-family leaves key here, so two references to
/// the same variable resolve to the same name.
///
/// `expr_names`: one entry per Gaussian affine-expression position.
/// Same expression at distinct positions resolves to distinct names.
#[derive(Debug)]
pub struct NamingScheme {
    pub(crate) var_names: HashMap<ContVarName, String>,
    pub(crate) expr_names: HashMap<GaussianAffineExpr, String>,
}

impl NamingScheme {
    pub fn derive(
        positions: &ResultPositions,
        interner: &StringInterner,
        types: &TypeRegistry,
    ) -> Self {
        let cons_sym = interner.lookup("Cons").copied().unwrap_or(u32::MAX);

        // Gaussian: assign one display name per affine expression.
        let mut gauss_idx = 0usize;
        let mut gauss_gen = || {
            let r = gaussian_var_name(gauss_idx);
            gauss_idx += 1;
            r
        };
        let gauss_paths_only: Vec<Path> = positions.gauss_path_to_expr.keys().cloned().collect();
        let gauss_names =
            assign_position_names(&gauss_paths_only, interner, types, cons_sym, &mut gauss_gen);

        // Beta + Dirichlet share a name generator (p, q, r, …).
        // For each family, derive a per-path name; then collapse to
        // per-ContVarName by picking the first name seen for each var.
        let mut prob_gen = ProbVarNameGenerator::new();

        let beta_paths_only: Vec<Path> = positions.beta_path_to_var.keys().cloned().collect();
        let beta_names =
            assign_position_names(&beta_paths_only, interner, types, cons_sym, &mut || {
                prob_gen.next_name()
            });

        let dir_paths_only: Vec<Path> = positions.dir_path_to_var.keys().cloned().collect();
        let dir_names =
            assign_position_names(&dir_paths_only, interner, types, cons_sym, &mut || {
                prob_gen.next_name()
            });

        let mut greek_gen = GreekVarNameGenerator::new();
        let gamma_paths_only: Vec<Path> = positions.gamma_path_to_var.keys().cloned().collect();
        let gamma_names =
            assign_position_names(&gamma_paths_only, interner, types, cons_sym, &mut || {
                greek_gen.next_name()
            });

        // Collapse to per-variable: first-seen name wins.
        let mut var_names = HashMap::new();
        for ((_, var), name) in positions.beta_path_to_var.iter().zip(beta_names.iter()) {
            var_names.entry(*var).or_insert_with(|| name.clone());
        }
        for ((_, var), name) in positions.dir_path_to_var.iter().zip(dir_names.iter()) {
            var_names.entry(*var).or_insert_with(|| name.clone());
        }
        for ((_, (var, _scale)), name) in positions.gamma_path_to_var.iter().zip(gamma_names.iter())
        {
            var_names.entry(*var).or_insert_with(|| name.clone());
        }

        let mut expr_names = HashMap::new();
        for ((_, expr), name) in positions.gauss_path_to_expr.iter().zip(gauss_names.iter()) {
            expr_names
                .entry(expr.clone())
                .or_insert_with(|| name.clone());
        }

        NamingScheme {
            var_names,
            expr_names,
        }
    }
}

fn gaussian_var_name(idx: usize) -> String {
    let names = ["x", "y", "z", "w", "v", "u", "t", "s", "r"];
    if idx < names.len() {
        names[idx].to_string()
    } else {
        format!("x{}", idx + 1)
    }
}

struct ProbVarNameGenerator {
    index: usize,
}

impl ProbVarNameGenerator {
    fn new() -> Self {
        Self { index: 0 }
    }
    fn next_name(&mut self) -> String {
        let names = ["p", "q", "r", "s", "t", "u", "v", "w"];
        let idx = self.index;
        self.index += 1;
        if idx < names.len() {
            names[idx].to_string()
        } else {
            format!("p{}", idx + 1)
        }
    }
}

/// Greek-letter names for gamma-family variables (rates and
/// Poisson/Exponential draws), falling back to `λ_{i}` once exhausted.
struct GreekVarNameGenerator {
    index: usize,
}

impl GreekVarNameGenerator {
    fn new() -> Self {
        Self { index: 0 }
    }
    fn next_name(&mut self) -> String {
        let names = ["λ", "μ", "ν", "κ", "θ", "ρ", "σ", "τ"];
        let idx = self.index;
        self.index += 1;
        if idx < names.len() {
            names[idx].to_string()
        } else {
            format!("λ_{}", idx + 1)
        }
    }
}

fn detect_list_groups(positions: &[Path], cons_sym: u32) -> Option<Vec<(Path, Vec<usize>)>> {
    let mut groups: Vec<(Path, Vec<(usize, usize)>)> = Vec::new();
    for (pos_idx, path) in positions.iter().enumerate() {
        if path.is_empty() {
            return None;
        }
        let last = path.last().unwrap();
        if last.0 != cons_sym || last.1 != 0 {
            return None;
        }
        let mut list_idx = 0;
        let mut prefix_end = path.len() - 1;
        while prefix_end > 0 && path[prefix_end - 1] == (cons_sym, 1) {
            list_idx += 1;
            prefix_end -= 1;
        }
        let prefix = path[..prefix_end].to_vec();
        if let Some(group) = groups.iter_mut().find(|(p, _)| *p == prefix) {
            group.1.push((list_idx, pos_idx));
        } else {
            groups.push((prefix, vec![(list_idx, pos_idx)]));
        }
    }
    let mut result = Vec::new();
    for (prefix, mut entries) in groups {
        entries.sort_by_key(|(idx, _)| *idx);
        let indices: Vec<usize> = entries.into_iter().map(|(_, pi)| pi).collect();
        result.push((prefix, indices));
    }
    Some(result)
}

fn field_name_for_path(
    path: &[(u32, usize)],
    interner: &StringInterner,
    types: &TypeRegistry,
    cons_sym: u32,
) -> Option<String> {
    let step = path.iter().rev().find(|(ctor, _)| *ctor != cons_sym)?;
    let (ctor, arg_idx) = *step;
    let fields = types.args_of_constructor.get(&ctor)?;
    if arg_idx >= fields.len() {
        return None;
    }
    let field_sym = fields[arg_idx];
    let field_name = interner.resolve(field_sym);
    let is_distinct = fields
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != arg_idx)
        .all(|(_, f)| *f != field_sym);
    if is_distinct && field_name.len() <= 5 && !field_name.is_empty() {
        Some(field_name.to_string())
    } else {
        None
    }
}

fn assign_position_names(
    positions: &[Path],
    interner: &StringInterner,
    types: &TypeRegistry,
    cons_sym: u32,
    name_gen: &mut impl FnMut() -> String,
) -> Vec<String> {
    let mut names = vec![String::new(); positions.len()];
    let mut assigned = vec![false; positions.len()];

    if let Some(groups) = detect_list_groups(positions, cons_sym) {
        for (prefix, indices) in groups.iter() {
            let base = if !prefix.is_empty() {
                field_name_for_path(prefix, interner, types, cons_sym)
            } else {
                None
            };
            if indices.len() == 1 {
                if let Some(name) = &base {
                    names[indices[0]] = name.clone();
                    assigned[indices[0]] = true;
                }
            } else {
                let base_name = base.unwrap_or_else(&mut *name_gen);
                for (i, &pos_idx) in indices.iter().enumerate() {
                    names[pos_idx] = format!("{}{}", base_name, subscript_str(i));
                    assigned[pos_idx] = true;
                }
            }
        }
    }

    let mut tentative_names: Vec<Option<String>> = vec![None; positions.len()];
    for (i, path) in positions.iter().enumerate() {
        if assigned[i] {
            continue;
        }
        if let Some(name) = field_name_for_path(path, interner, types, cons_sym) {
            tentative_names[i] = Some(name);
        }
    }

    let mut all_name_counts: HashMap<String, usize> = HashMap::new();
    for name in &names {
        if !name.is_empty() {
            *all_name_counts.entry(name.clone()).or_insert(0) += 1;
        }
    }
    for n in tentative_names.iter().flatten() {
        *all_name_counts.entry(n.clone()).or_insert(0) += 1;
    }

    for (i, tentative) in tentative_names.iter().enumerate() {
        if assigned[i] {
            continue;
        }
        if let Some(n) = tentative {
            if all_name_counts[n] == 1 {
                names[i] = n.clone();
                assigned[i] = true;
                continue;
            }
        }
        names[i] = name_gen();
        assigned[i] = true;
    }

    names
}

pub fn format_value(
    val: &PluckVal,
    interner: &StringInterner,
    naming: Option<&NamingScheme>,
) -> String {
    let nil_sym = interner.lookup("Nil").copied().unwrap_or(u32::MAX);
    let cons_sym = interner.lookup("Cons").copied().unwrap_or(u32::MAX);
    let mut path = Vec::new();
    format_value_inner(val, interner, naming, nil_sym, cons_sym, &mut path)
}

fn format_value_inner(
    val: &PluckVal,
    interner: &StringInterner,
    naming: Option<&NamingScheme>,
    nil_sym: u32,
    cons_sym: u32,
    path: &mut Path,
) -> String {
    // String detection (Cons/Nil chain of 8-bit constant IntDists).
    if let Some(s) = try_extract_string(val, nil_sym, cons_sym) {
        return format!("\"{}\"", s);
    }

    // List detection — recurse with the path properly extended.
    if let Some(elements) = try_format_list(val, interner, naming, nil_sym, cons_sym, path) {
        return format!("[{}]", elements.join(", "));
    }

    // Natural number (strict O/S chain, or a Native(Int)).
    if let (Some(&o), Some(&s)) = (interner.lookup("O"), interner.lookup("S")) {
        if let Some(n) = extract_nat(val, o, s) {
            return format!("{}", n);
        }
    }

    match val.as_ref() {
        PluckValue::Value { constructor, args } => {
            let name = interner.resolve(*constructor);
            if name == "Pair" && args.len() == 2 {
                path.push((*constructor, 0));
                let a = format_value_inner(&args[0], interner, naming, nil_sym, cons_sym, path);
                path.pop();
                path.push((*constructor, 1));
                let b = format_value_inner(&args[1], interner, naming, nil_sym, cons_sym, path);
                path.pop();
                return format!("({}, {})", a, b);
            }
            if args.is_empty() {
                name.to_string()
            } else {
                let mut parts = Vec::with_capacity(args.len());
                for (i, arg) in args.iter().enumerate() {
                    path.push((*constructor, i));
                    parts.push(format_value_inner(
                        arg, interner, naming, nil_sym, cons_sym, path,
                    ));
                    path.pop();
                }
                format!("({} {})", name, parts.join(" "))
            }
        }
        PluckValue::Native(v) => match v {
            crate::language::pexpr::NativeVal::Float(f) => format!("{}", f),
            crate::language::pexpr::NativeVal::Int(i) => format!("{}", i),
            crate::language::pexpr::NativeVal::Symbol(s) => format!("'{}", interner.resolve(*s)),
            crate::language::pexpr::NativeVal::Bool(b) => format!("{}", b),
        },
        PluckValue::Closure(_) => "<closure>".to_string(),
        PluckValue::IntDist { bits } => format_int_dist(bits),
        PluckValue::Thunk(_) => "<thunk>".to_string(),
        PluckValue::ThunkUnion(_) => "<thunk-union>".to_string(),
        PluckValue::FloatMatrix { entries, shape } => {
            format_matrix(entries, shape, interner, naming, nil_sym, cons_sym, path)
        }
        // Symbolic-continuous variants route through `render_symbolic`,
        // which knows how to consult the naming scheme per family.
        v if v.is_symbolic() => v
            .render_symbolic(naming)
            .expect("is_symbolic / render_symbolic disagreement"),
        _ => unreachable!("format_value_inner: every PluckValue arm was handled above"),
    }
}

fn format_matrix(
    entries: &[Rc<PluckValue>],
    shape: &Vec<usize>,
    interner: &StringInterner,
    naming: Option<&NamingScheme>,
    nil_sym: u32,
    cons_sym: u32,
    path: &mut Path,
) -> String {
    let rank = shape.len();
    if rank == 1 {
        let parts: Vec<String> = entries
            .iter()
            .map(|e| format_value_inner(e, interner, naming, nil_sym, cons_sym, path))
            .collect();
        format!("{{{}}}", parts.join(", "))
    } else if rank == 2 {
        let cols = shape[1];
        let mut rows = Vec::with_capacity(shape[0]);
        for r in 0..shape[0] {
            let row_parts: Vec<String> = (0..cols)
                .map(|c| {
                    format_value_inner(
                        &entries[r * cols + c],
                        interner,
                        naming,
                        nil_sym,
                        cons_sym,
                        path,
                    )
                })
                .collect();
            rows.push(format!("{{{}}}", row_parts.join(", ")));
        }
        format!("{{{}}}", rows.join(", "))
    } else {
        format!("<FloatMatrix shape={:?}>", shape)
    }
}

fn try_format_list(
    val: &PluckVal,
    interner: &StringInterner,
    naming: Option<&NamingScheme>,
    nil_sym: u32,
    cons_sym: u32,
    path: &mut Path,
) -> Option<Vec<String>> {
    // First check the shape (no formatting): walk the spine to confirm
    // it's a proper Cons/Nil chain.
    {
        let mut cur = val;
        loop {
            match cur.as_ref() {
                PluckValue::Value { constructor, args }
                    if *constructor == nil_sym && args.is_empty() =>
                {
                    break;
                }
                PluckValue::Value { constructor, args }
                    if *constructor == cons_sym && args.len() == 2 =>
                {
                    cur = &args[1];
                }
                _ => return None,
            }
        }
    }

    // Shape OK — render elements, extending the path for each one.
    // Track how many (Cons, 1) spine steps we push so we can pop them
    // all at the end and leave `path` unchanged for the caller.
    let mut elements = Vec::new();
    let mut spine_pushes = 0usize;
    let mut current = val;
    loop {
        match current.as_ref() {
            PluckValue::Value { constructor, args }
                if *constructor == nil_sym && args.is_empty() =>
            {
                for _ in 0..spine_pushes {
                    path.pop();
                }
                return Some(elements);
            }
            PluckValue::Value { constructor, args }
                if *constructor == cons_sym && args.len() == 2 =>
            {
                path.push((cons_sym, 0));
                elements.push(format_value_inner(
                    &args[0], interner, naming, nil_sym, cons_sym, path,
                ));
                path.pop();
                path.push((cons_sym, 1));
                spine_pushes += 1;
                current = &args[1];
            }
            _ => unreachable!("shape was validated above"),
        }
    }
}

fn try_extract_string(val: &PluckVal, nil_sym: u32, cons_sym: u32) -> Option<String> {
    let mut chars = Vec::new();
    let mut current = val;
    loop {
        match current.as_ref() {
            PluckValue::Value { constructor, args }
                if *constructor == nil_sym && args.is_empty() =>
            {
                if chars.is_empty() {
                    return None;
                }
                return Some(String::from_utf8_lossy(&chars).to_string());
            }
            PluckValue::Value { constructor, args }
                if *constructor == cons_sym && args.len() == 2 =>
            {
                if let PluckValue::IntDist { bits } = args[0].as_ref() {
                    if bits.len() == 8 {
                        let mut byte_val: u8 = 0;
                        let mut is_const = true;
                        for (i, bit) in bits.iter().enumerate() {
                            if bit.is_true() {
                                byte_val |= 1 << i;
                            } else if !bit.is_false() {
                                is_const = false;
                                break;
                            }
                        }
                        if is_const {
                            chars.push(byte_val);
                            current = &args[1];
                            continue;
                        }
                    }
                }
                return None;
            }
            _ => return None,
        }
    }
}

fn format_int_dist(bits: &[BooleanFunction]) -> String {
    let mut is_const = true;
    let mut int_val: u64 = 0;
    for (i, bit) in bits.iter().enumerate() {
        if bit.is_true() {
            int_val |= 1 << i;
        } else if bit.is_false() {
        } else {
            is_const = false;
            break;
        }
    }
    if is_const && bits.len() == 8 && (32..=126).contains(&int_val) {
        format!("'{}'", int_val as u8 as char)
    } else if is_const {
        format!("@{}", int_val)
    } else {
        format!("IntDist{{width={}}}", bits.len())
    }
}

/// Render a non-negative integer as a Unicode subscript string
/// (`12` → `"₁₂"`).
pub fn subscript_str(n: usize) -> String {
    n.to_string()
        .chars()
        .map(|c| char::from_u32(0x2080 + (c as u32 - '0' as u32)).unwrap_or(c))
        .collect()
}
