use crate::language::define::DefinitionRegistry;
use crate::language::types::{Symbol, TypeRegistry};

use super::state::LazyKCState;

/// Bundles the immutable references and mutable state that every compilation
/// helper needs. `mode` stays as a method parameter (never stored here):
/// the bind machinery threads it — `bind_monad` constructs a per-world KC
/// mode with a tighter `path_condition`, while Sample mode is propagated
/// through `bind_compile`'s short-circuit.
pub struct CompilerCtx<'a> {
    pub types: &'a TypeRegistry,
    pub defs: &'a DefinitionRegistry,
    pub true_sym: Symbol,
    pub false_sym: Symbol,
    pub state: &'a mut LazyKCState,
}
