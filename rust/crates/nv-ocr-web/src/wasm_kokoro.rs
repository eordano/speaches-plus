use js_sys::Function;
use wasm_bindgen::prelude::*;

use crate::kokoro_web;

fn js_err(e: impl core::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn emit(cb: &Function, json: String) {
    if let Ok(v) = js_sys::JSON::parse(&json) {
        let _ = cb.call1(&JsValue::NULL, &v);
    }
}

#[wasm_bindgen]
pub fn kokoro_stage_clear() {
    kokoro_web::kokoro_stage_clear()
}

#[wasm_bindgen]
pub fn kokoro_stage_reserve(name: &str, size: u32) {
    kokoro_web::kokoro_stage_reserve(name, size)
}

#[wasm_bindgen]
pub fn kokoro_stage_push(name: &str, chunk: &[u8]) {
    kokoro_web::kokoro_stage_push(name, chunk)
}

#[wasm_bindgen]
pub fn kokoro_stage_ref(name: &str, pcm24k: &[f32]) {
    kokoro_web::kokoro_stage_ref(name, pcm24k)
}

#[wasm_bindgen]
pub fn kokoro_staged() -> String {
    kokoro_web::kokoro_staged()
}

#[wasm_bindgen]
pub fn kokoro_init() -> Result<String, JsValue> {
    kokoro_web::init_core().map_err(js_err)
}

#[wasm_bindgen]
pub fn kokoro_synth(
    ids: &[u32],
    style: &[f32],
    speed: f64,
    seed: u64,
    with_adapter: bool,
) -> Result<Vec<f32>, JsValue> {
    let adapter = if with_adapter {
        kokoro_web::adapter_map()
    } else {
        None
    };
    kokoro_web::synth_core(ids, style, speed, seed, adapter.as_ref()).map_err(js_err)
}

#[wasm_bindgen]
pub fn kokoro_fit_style(opts_json: String, on_event: Function) -> Result<String, JsValue> {
    kokoro_web::fit_style_core(&opts_json, &mut |j| emit(&on_event, j)).map_err(js_err)
}

#[wasm_bindgen]
pub fn kokoro_fit_adapter(
    opts_json: String,
    style: &[f32],
    on_event: Function,
) -> Result<String, JsValue> {
    kokoro_web::fit_adapter_core(&opts_json, style, &mut |j| emit(&on_event, j)).map_err(js_err)
}

#[wasm_bindgen]
pub fn kokoro_adapter_safetensors() -> Result<Option<Vec<u8>>, JsValue> {
    kokoro_web::adapter_safetensors().map_err(js_err)
}
