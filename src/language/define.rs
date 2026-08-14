use std::collections::HashMap;
use std::rc::Rc;

use super::pexpr::PExpr;
use super::types::Symbol;

/// A named definition in Pluck (function or value).
#[derive(Debug, Clone)]
pub struct Definition {
    pub name: Symbol,
    pub expr: Rc<PExpr>,
}

/// Registry of top-level definitions.
/// In Julia this was the global `DEFINITIONS` dict.
/// Here it lives inside PluckContext.
pub struct DefinitionRegistry {
    definitions: HashMap<Symbol, Definition>,
}

impl Default for DefinitionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DefinitionRegistry {
    pub fn new() -> Self {
        DefinitionRegistry {
            definitions: HashMap::new(),
        }
    }

    pub fn define(&mut self, name: Symbol, expr: Rc<PExpr>) {
        self.definitions.insert(name, Definition { name, expr });
    }

    pub fn lookup(&self, name: Symbol) -> Option<&Definition> {
        self.definitions.get(&name)
    }

    pub fn is_defined(&self, name: Symbol) -> bool {
        self.definitions.contains_key(&name)
    }

    pub fn clear(&mut self) {
        self.definitions.clear();
    }
}
