use std::collections::HashMap;

/// Interned symbol — index into the interner.
pub type Symbol = u32;

/// A string interner for fast symbol comparison.
pub struct StringInterner {
    map: HashMap<String, Symbol>,
    strings: Vec<String>,
}

impl StringInterner {
    /// Look up a string without interning it. Returns None if not yet interned.
    pub fn lookup(&self, s: &str) -> Option<&Symbol> {
        self.map.get(s)
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

impl StringInterner {
    pub fn new() -> Self {
        StringInterner {
            map: HashMap::new(),
            strings: Vec::new(),
        }
    }

    pub fn intern(&mut self, s: &str) -> Symbol {
        if let Some(&sym) = self.map.get(s) {
            return sym;
        }
        let sym = self.strings.len() as Symbol;
        self.strings.push(s.to_string());
        self.map.insert(s.to_string(), sym);
        sym
    }

    pub fn resolve(&self, sym: Symbol) -> &str {
        &self.strings[sym as usize]
    }
}

/// Registry for algebraic data types (sum-of-product types).
///
/// Tracks constructor -> type mappings, constructor arity, and field names.
pub struct TypeRegistry {
    /// constructor symbol -> type symbol
    pub type_of_constructor: HashMap<Symbol, Symbol>,
    /// type symbol -> list of constructor symbols
    pub constructors_of_type: HashMap<Symbol, Vec<Symbol>>,
    /// constructor symbol -> list of field name symbols
    pub args_of_constructor: HashMap<Symbol, Vec<Symbol>>,
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeRegistry {
    pub fn new() -> Self {
        TypeRegistry {
            type_of_constructor: HashMap::new(),
            constructors_of_type: HashMap::new(),
            args_of_constructor: HashMap::new(),
        }
    }

    /// Register a new algebraic datatype.
    /// `constructors` maps constructor name -> list of field names.
    pub fn define_type(&mut self, type_name: Symbol, constructors: Vec<(Symbol, Vec<Symbol>)>) {
        let mut ctor_list = Vec::new();
        for (ctor, args) in constructors {
            self.type_of_constructor.insert(ctor, type_name);
            self.args_of_constructor.insert(ctor, args);
            ctor_list.push(ctor);
        }
        self.constructors_of_type.insert(type_name, ctor_list);
    }

    pub fn constructor_arity(&self, ctor: Symbol) -> usize {
        self.args_of_constructor
            .get(&ctor)
            .map_or(0, |args| args.len())
    }

    pub fn is_constructor(&self, sym: Symbol) -> bool {
        self.type_of_constructor.contains_key(&sym)
    }

    /// Find the list-like type: a type with exactly two constructors,
    /// one nullary (Nil) and one binary (Cons). Returns (nil, cons).
    pub fn find_list_constructors(&self) -> Option<(Symbol, Symbol)> {
        self.constructors_of_type.iter().find_map(|(_, ctors)| {
            if ctors.len() == 2 {
                let a0 = self.constructor_arity(ctors[0]);
                let a1 = self.constructor_arity(ctors[1]);
                if a0 == 0 && a1 == 2 {
                    Some((ctors[0], ctors[1]))
                } else if a1 == 0 && a0 == 2 {
                    Some((ctors[1], ctors[0]))
                } else {
                    None
                }
            } else {
                None
            }
        })
    }

    /// Find the Peano-nat type: a type with exactly two constructors,
    /// one nullary (O) and one unary (S). Returns (o, s).
    pub fn find_nat_constructors(&self) -> Option<(Symbol, Symbol)> {
        self.constructors_of_type.iter().find_map(|(_, ctors)| {
            if ctors.len() == 2 {
                let a0 = self.constructor_arity(ctors[0]);
                let a1 = self.constructor_arity(ctors[1]);
                if a0 == 0 && a1 == 1 {
                    Some((ctors[0], ctors[1]))
                } else if a1 == 0 && a0 == 1 {
                    Some((ctors[1], ctors[0]))
                } else {
                    None
                }
            } else {
                None
            }
        })
    }

    /// Register built-in types: bool, nat, list, unit.
    pub fn register_builtins(&mut self, interner: &mut StringInterner) {
        let bool_sym = interner.intern("bool");
        let true_sym = interner.intern("True");
        let false_sym = interner.intern("False");
        self.define_type(bool_sym, vec![(true_sym, vec![]), (false_sym, vec![])]);

        let nat_sym = interner.intern("nat");
        let o_sym = interner.intern("O");
        let s_sym = interner.intern("S");
        let n_sym = interner.intern("n");
        self.define_type(nat_sym, vec![(o_sym, vec![]), (s_sym, vec![n_sym])]);

        let list_sym = interner.intern("list");
        let nil_sym = interner.intern("Nil");
        let cons_sym = interner.intern("Cons");
        let head_sym = interner.intern("head");
        let tail_sym = interner.intern("tail");
        self.define_type(
            list_sym,
            vec![(nil_sym, vec![]), (cons_sym, vec![head_sym, tail_sym])],
        );

        let unit_sym = interner.intern("unit");
        let unit_ctor = interner.intern("Unit");
        self.define_type(unit_sym, vec![(unit_ctor, vec![])]);
    }
}
