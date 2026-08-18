use std::rc::Rc;

use super::compile::CompileMode;
use super::ctx::CompilerCtx;
use super::state::{GuardedWorlds, World};
use crate::discrete_factorizations::{
    BooleanFactorization, BooleanFunction, BooleanFunctionOps, Factorizer,
};
use crate::language::types::Symbol;
use crate::language::values::*;

/// Wrap a value into a single-element world list guarded by `true` (the pc
/// parameter is accepted for signature uniformity but unused).
/// This is the `pure` operation of the monad.
pub fn pure_monad(val: PluckVal, _path_condition: BooleanFunction) -> GuardedWorlds {
    (
        vec![(val, BooleanFunction::true_ptr())],
        BooleanFunction::true_ptr(),
    )
}

/// An empty worlds list (impossible/error state — program error, shaves off probability).
pub fn program_error_worlds() -> GuardedWorlds {
    (Vec::new(), BooleanFunction::true_ptr())
}

/// An empty worlds list for inference errors (depth limit, time limit).
pub fn inference_error_worlds() -> GuardedWorlds {
    (Vec::new(), BooleanFunction::true_ptr())
}

/// An empty worlds list for false path conditions.
/// The used_information is BDD_FALSE, meaning this can be reused if you can prove false.
pub fn false_path_condition_worlds() -> GuardedWorlds {
    (Vec::new(), BooleanFunction::false_ptr())
}

/// Monadic bind — the **KC engine**: for each world in `pre_worlds`, apply
/// `cont` and collect+join results.
///
/// The continuation receives
/// `(value, inner_path_condition, ctx, mode)` and returns GuardedWorlds.
/// The mode handed to the continuation is a per-world
/// `KC { path_condition: inner_path_condition }` — exactly the mode every
/// continuation used to construct by hand. Sample-mode dispatch lives in
/// `bind_compile`; this function is KC-only.
pub fn bind_monad<F>(
    pre_worlds: GuardedWorlds,
    path_condition: BooleanFunction,
    ctx: &mut CompilerCtx,
    mut cont: F,
) -> GuardedWorlds
where
    F: FnMut(PluckVal, BooleanFunction, &mut CompilerCtx, &mut CompileMode<'_>) -> GuardedWorlds,
{
    let (pre_world_list, pre_used_info) = pre_worlds;

    let mut nested_worlds: Vec<(GuardedWorlds, BooleanFunction)> = Vec::new();

    for (pre_val, pre_guard) in pre_world_list {
        if ctx.state.hit_limit {
            return inference_error_worlds();
        }

        let builder = ctx.state.fac();
        let inner_path_condition = builder.and(&path_condition, &pre_guard);

        if inner_path_condition.is_false() {
            ctx.state.stats.num_false_pc_exits += 1;
            nested_worlds.push((false_path_condition_worlds(), pre_guard));
            continue;
        }

        let mut inner_mode = CompileMode::KC {
            path_condition: inner_path_condition.clone(),
        };
        let post_worlds = cont(pre_val, inner_path_condition, ctx, &mut inner_mode);
        nested_worlds.push((post_worlds, pre_guard));
    }

    join_monad(nested_worlds, pre_used_info, ctx)
}

/// Bind wrapper that dispatches on the caller's `mode`; callers typically
/// call `self.compile(...)` to produce `pre_worlds`, then pass them here.
///
/// In **Sample mode**, this is the forward single-world threading point:
/// pre-worlds contain **exactly one** world (asserted — every Sample-mode
/// producer is single-world; the only remaining multi-world resolution
/// point is ThunkUnion selection in `evaluate_thunk_union`), and the
/// continuation runs exactly **once**, receiving the *Sample* mode — so
/// function bodies, case branches, and strict second arguments execute
/// forward along the sampled path instead of being knowledge-compiled.
/// Running the continuation once is essential: a per-speculative-world run
/// would draw samples for branches that don't exist in the sampled world.
///
/// In **KC mode**, delegates to `bind_monad`, the per-world fork/join
/// engine. The path_condition is threaded through to bind_monad so that
/// ThunkUnion evaluations triggered from continuations can prune branches
/// whose guards are inconsistent with the current path condition.
pub fn bind_compile<F>(
    pre_worlds: GuardedWorlds,
    path_condition: BooleanFunction,
    ctx: &mut CompilerCtx,
    mode: &mut CompileMode<'_>,
    mut cont: F,
) -> GuardedWorlds
where
    F: FnMut(PluckVal, BooleanFunction, &mut CompilerCtx, &mut CompileMode<'_>) -> GuardedWorlds,
{
    if matches!(mode, CompileMode::Sample(_)) {
        let (worlds, _) = pre_worlds;
        if worlds.is_empty() {
            if ctx.state.hit_limit {
                // Inference limit (max_depth / time_limit), not a program
                // error: propagate the empty result.
                return inference_error_worlds();
            }
            // Shaved probability mass: failed pattern match, `(error)`, or
            // applying a non-function.
            // (compile_case_of has already eprintln'd context for the
            // incomplete-match case).
            panic!(
                "Sample mode: expression produced no worlds \
                 (program error along the sampled path)"
            );
        }
        assert_eq!(
            worlds.len(),
            1,
            "Sample mode: bind received {} pre-worlds; every Sample-mode \
             producer must yield exactly one world",
            worlds.len()
        );
        let val = worlds.into_iter().next().unwrap().0;
        // `pure_monad` ignores its pc argument and sample handlers emit
        // `true` guards, so `true_ptr` is the right pc for the cont.
        return cont(val, BooleanFunction::true_ptr(), ctx, mode);
    }
    bind_monad(pre_worlds, path_condition, ctx, cont)
}

/// Create a ThunkUnion that flattens nested ThunkUnions and deduplicates by Rc identity.
///
/// Flattens nested unions and merges guards of identical thunks (by object identity). 
/// Without flattening, nested ThunkUnion trees cause exponential traversal in evaluate_thunk_union.
fn mk_thunk_union_flat(thunks: Vec<(PluckVal, BooleanFunction)>, builder: &Factorizer) -> PluckVal {
    let mut flat: Vec<(PluckVal, BooleanFunction)> = Vec::new();

    for (val, outer_guard) in thunks {
        if let PluckValue::ThunkUnion(union_data) = val.as_ref() {
            // Flatten: push each inner entry with guard = outer_guard & inner_guard
            for (inner_val, inner_guard) in &union_data.thunks {
                let combined_guard = builder.and(&outer_guard, inner_guard);
                if combined_guard.is_false() {
                    continue;
                }
                // Deduplicate by Rc pointer identity
                let mut found = false;
                for (existing_val, existing_guard) in flat.iter_mut() {
                    if Rc::ptr_eq(existing_val, inner_val) {
                        *existing_guard = builder.or(existing_guard, &combined_guard);
                        found = true;
                        break;
                    }
                }
                if !found {
                    flat.push((inner_val.clone(), combined_guard));
                }
            }
        } else {
            // Not a ThunkUnion — add directly, dedup by Rc identity
            let mut found = false;
            for (existing_val, existing_guard) in flat.iter_mut() {
                if Rc::ptr_eq(existing_val, &val) {
                    *existing_guard = builder.or(existing_guard, &outer_guard);
                    found = true;
                    break;
                }
            }
            if !found {
                flat.push((val, outer_guard));
            }
        }
    }

    if flat.len() == 1 {
        flat.into_iter().next().unwrap().0
    } else {
        mk_thunk_union(flat)
    }
}

/// Join/merge worlds that have the same constructor.
///
/// This is the key optimization in lazy KC: when two worlds have the same
/// constructor but different arguments, we merge them into a single world
/// with ThunkUnion arguments. This avoids exponential blowup.
pub fn join_monad(
    nested_worlds: Vec<(GuardedWorlds, BooleanFunction)>,
    pre_used_info: BooleanFunction,
    ctx: &mut CompilerCtx,
) -> GuardedWorlds {
    let builder = ctx.state.fac();

    // Compute used_information
    let mut used_information = pre_used_info;
    for ((_, used_info), pre_guard) in &nested_worlds {
        // used_information &= implies(pre_guard, used_info)
        // implies(a, b) = !a | b
        let neg_guard = builder.negate(pre_guard);
        let implies = builder.or(&neg_guard, used_info);
        used_information = builder.and(&used_information, &implies);
    }

    // Collect and merge the resulting worlds
    let mut join_results: Vec<World> = Vec::new();
    // Group Value worlds by constructor for ThunkUnion merging
    let mut results_for_constructor: Vec<(Symbol, Vec<(PluckVal, BooleanFunction)>)> = Vec::new();

    for ((post_worlds, _), pre_guard) in nested_worlds {
        for (post_val, post_guard) in post_worlds {
            let pre_and_post = builder.and(&post_guard, &pre_guard);
            if pre_and_post.is_false() {
                continue;
            }

            match post_val.as_ref() {
                PluckValue::Value { constructor, .. } => {
                    let ctor = *constructor;
                    if let Some(group) =
                        results_for_constructor.iter_mut().find(|(c, _)| *c == ctor)
                    {
                        group.1.push((post_val, pre_and_post));
                    } else {
                        results_for_constructor.push((ctor, vec![(post_val, pre_and_post)]));
                    }
                }
                PluckValue::Closure(_) | PluckValue::Native(_) => {
                    // For non-Value results, merge identical values
                    let mut found = false;
                    for (existing_val, existing_guard) in join_results.iter_mut() {
                        if std::rc::Rc::ptr_eq(existing_val, &post_val)
                            || values_equal(existing_val, &post_val)
                        {
                            *existing_guard = builder.or(existing_guard, &pre_and_post);
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        join_results.push((post_val, pre_and_post));
                    }
                }
                PluckValue::IntDist { bits } => {
                    // Combine IntDists: OR each bit under its respective guard
                    let mut found = false;
                    for (existing_val, existing_guard) in join_results.iter_mut() {
                        if let PluckValue::IntDist {
                            bits: existing_bits,
                        } = existing_val.as_ref()
                        {
                            if existing_bits.len() == bits.len() {
                                // Same width — merge by OR-ing bits under guards
                                let mut new_bits = Vec::with_capacity(bits.len());
                                for i in 0..bits.len() {
                                    let new_contrib = builder.and(&bits[i], &pre_and_post);
                                    new_bits.push(builder.or(&existing_bits[i], &new_contrib));
                                }
                                let new_guard = builder.or(existing_guard, &pre_and_post);
                                *existing_val = mk_int_dist(new_bits);
                                *existing_guard = new_guard;
                                found = true;
                                break;
                            }
                        }
                    }
                    if !found {
                        // Guard the bits before pushing — needed so that when
                        // a second IntDist is merged in later, OR-ing is correct.
                        let guarded_bits: Vec<BooleanFunction> =
                            bits.iter().map(|b| builder.and(b, &pre_and_post)).collect();
                        join_results.push((mk_int_dist(guarded_bits), pre_and_post));
                    }
                }
                PluckValue::Thunk(_) | PluckValue::ThunkUnion(_) => {
                    // Thunks shouldn't appear here directly as post_val
                    // TODO this should be an assert!
                    join_results.push((post_val, pre_and_post));
                }
                PluckValue::FloatMatrix { .. } => {
                    // FloatMatrices merge by Rc-identity only — peeking into
                    // entries to compare would force thunks and defeat lazy
                    // evaluation. Distinct matrices stay as separate worlds.
                    let mut found = false;
                    for (existing_val, existing_guard) in join_results.iter_mut() {
                        if std::rc::Rc::ptr_eq(existing_val, &post_val) {
                            *existing_guard = builder.or(existing_guard, &pre_and_post);
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        join_results.push((post_val, pre_and_post));
                    }
                }
                v if v.is_symbolic() => {
                    // Symbolic continuous values: merge identical ones.
                    let mut found = false;
                    for (existing_val, existing_guard) in join_results.iter_mut() {
                        if values_equal(existing_val, &post_val) {
                            *existing_guard = builder.or(existing_guard, &pre_and_post);
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        join_results.push((post_val, pre_and_post));
                    }
                }
                // The guarded `is_symbolic` arm above covers every
                // symbolic variant; this is unreachable in practice
                // but required for Rust exhaustiveness checking.
                _ => unreachable!("join_monad: unhandled PluckValue variant"),
            }
        }
    }

    // Now process constructor groups
    for (ctor, entries) in results_for_constructor {
        if entries.len() == 1 {
            let (val, guard) = entries.into_iter().next().unwrap();
            join_results.push((val, guard));
            continue;
        }

        // Deduplicate: group by value identity
        let mut deduped: Vec<(PluckVal, BooleanFunction)> = Vec::new();
        for (val, guard) in &entries {
            let mut found = false;
            for (existing_val, existing_guard) in deduped.iter_mut() {
                if std::rc::Rc::ptr_eq(existing_val, val) || values_equal(existing_val, val) {
                    *existing_guard = builder.or(existing_guard, guard);
                    found = true;
                    break;
                }
            }
            if !found {
                deduped.push((val.clone(), guard.clone()));
            }
        }

        if deduped.len() == 1 {
            let (val, guard) = deduped.into_iter().next().unwrap();
            join_results.push((val, guard));
            continue;
        }

        // Multiple distinct worlds for same constructor -> merge into ThunkUnion args
        let mut overall_guard = BooleanFunction::false_ptr();
        let num_args = match deduped[0].0.as_ref() {
            PluckValue::Value { args, .. } => args.len(),
            _ => unreachable!(),
        };

        let mut thunks_of_arg: Vec<Vec<(PluckVal, BooleanFunction)>> = vec![Vec::new(); num_args];

        for (post_val, pre_and_post) in &deduped {
            overall_guard = builder.or(&overall_guard, pre_and_post);
            if let PluckValue::Value { args, .. } = post_val.as_ref() {
                for (i, arg) in args.iter().enumerate() {
                    thunks_of_arg[i].push((arg.clone(), pre_and_post.clone()));
                }
            }
        }

        let merged_args: Vec<PluckVal> = thunks_of_arg
            .into_iter()
            .map(|thunks| {
                if thunks.len() == 1 {
                    thunks.into_iter().next().unwrap().0
                } else {
                    mk_thunk_union_flat(thunks, builder)
                }
            })
            .collect();

        join_results.push((mk_value(ctor, merged_args), overall_guard));
    }

    (join_results, used_information)
}

/// If-then-else in the BDD world:
/// Creates two worlds: (val_true, condition) and (val_false, NOT condition)
pub fn if_then_else_monad(
    val_true: PluckVal,
    val_false: PluckVal,
    condition: BooleanFunction,
    _path_condition: BooleanFunction,
    ctx: &mut CompilerCtx,
) -> GuardedWorlds {
    let builder = ctx.state.fac();
    let neg_condition = builder.negate(&condition);

    let mut worlds = Vec::new();
    if !condition.is_false() {
        worlds.push((val_true, condition));
    }
    if !neg_condition.is_false() {
        worlds.push((val_false, neg_condition));
    }
    (worlds, BooleanFunction::true_ptr())
}

/// Build one world per (value, condition) pair, dropping any pair whose
/// condition is unsatisfiable (false). Guards are the given conditions
/// directly; this function does not require them to be mutually exclusive
/// or exhaustive.
pub fn categorical_monad(
    vals: Vec<PluckVal>,
    conditions: Vec<BooleanFunction>,
    _path_condition: BooleanFunction,
    _ctx: &mut CompilerCtx,
) -> GuardedWorlds {
    let mut worlds = Vec::new();
    for (val, condition) in vals.iter().zip(conditions) {
        if !condition.is_false() {
            worlds.push((val.clone(), condition));
        }
    }
    (worlds, BooleanFunction::true_ptr())
}

/// Simple structural equality check for PluckValue.
/// Used in join_monad for deduplication.
fn values_equal(a: &PluckVal, b: &PluckVal) -> bool {
    match (a.as_ref(), b.as_ref()) {
        (PluckValue::Native(a), PluckValue::Native(b)) => a == b,
        (
            PluckValue::Value {
                constructor: ca,
                args: aa,
            },
            PluckValue::Value {
                constructor: cb,
                args: ab,
            },
        ) => {
            ca == cb
                && aa.len() == ab.len()
                && aa.iter().zip(ab).all(|(x, y)| std::rc::Rc::ptr_eq(x, y))
        }
        (PluckValue::IntDist { bits: ba }, PluckValue::IntDist { bits: bb }) => {
            // IntDist equality: same width and all BDD pointers identical
            ba.len() == bb.len() && ba.iter().zip(bb).all(|(x, y)| *x == *y)
        }
        _ => a.symbolic_eq(b),
    }
}
