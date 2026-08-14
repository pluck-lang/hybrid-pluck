//! WASM bindings.
//!
//! `run` serializes the library's `Vec<QueryResult>` to JS with
//! `serde-wasm-bindgen`. Consumers parse that wire format directly.

use wasm_bindgen::prelude::*;

use crate::PluckContext;

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

fn serialize_results(query_results: Vec<crate::QueryResult>) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&query_results).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub struct WasmPluckContext {
    inner: PluckContext,
}

#[wasm_bindgen]
impl WasmPluckContext {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        WasmPluckContext {
            inner: PluckContext::new(),
        }
    }

    pub fn register_file(&mut self, name: &str, content: &str) {
        self.inner
            .register_file(name.to_string(), content.to_string());
    }

    pub fn run(&mut self, source: &str) -> Result<JsValue, JsValue> {
        let query_results = self.inner.run(source);
        serialize_results(query_results)
    }
}

#[wasm_bindgen]
pub fn run_pluck(source: &str) -> Result<JsValue, JsValue> {
    let mut ctx = WasmPluckContext::new();
    ctx.run(source)
}
