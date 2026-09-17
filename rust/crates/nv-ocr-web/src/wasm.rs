use std::cell::RefCell;
use std::collections::BTreeMap;

use candle_core::Device;
use js_sys::Function;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::train::e4b::{e4b_train_options, run_e4b_window_qlora_pumped, E4bWindowSample};
use crate::train::{run_qlora_pumped, TrainOptions, TrainSample};
use crate::{
    ensure_browser_gpu, init_with_adapter, AdapterBytes, OcrWebOptions, OcrWebSession, PageSample,
};
use nv_models::dots_ocr::DotsMode;
use nv_weights::WeightLoader;

pub const ASYNC_READBACK_PUMP_LANDED: bool = true;

pub const EVERY_BROWSER_READBACK_RIDES_THE_PUMP_SO_NOTHING_HERE_BLOCKS_THE_EVENT_LOOP: &str =
    "engine_ocr_page awaits the graph decoder's pumped token/logit fetches and the resident \
     vision splice, and engine_train awaits the pumped twins of every QLoRA kernel readback; \
     the blocking map_async-plus-poll(Wait) arm still serves native callers untouched, and \
     ASYNC_READBACK_PUMP_LANDED records that the upfront refusals this surface used to carry \
     are gone";

thread_local! {
    static STAGE: RefCell<BTreeMap<String, Vec<u8>>> = const { RefCell::new(BTreeMap::new()) };
    static ADAPTER: RefCell<Option<AdapterBytes>> = const { RefCell::new(None) };
    static SESSION: RefCell<Option<OcrWebSession>> = const { RefCell::new(None) };
    static TRAIN_STAGE: RefCell<BTreeMap<String, Vec<u8>>> =
        const { RefCell::new(BTreeMap::new()) };
    static TRAIN_SAMPLES: RefCell<Vec<StagedTrainSample>> = const { RefCell::new(Vec::new()) };
    static TRAIN_ADAPTER: RefCell<Option<(Vec<u8>, Vec<u8>)>> = const { RefCell::new(None) };
    static E4B_SAMPLES: RefCell<Vec<E4bWindowSample>> = const { RefCell::new(Vec::new()) };
}

pub const E4B_TRAIN_SHARES_THE_TRAINER_STAGE_AND_ADAPTER_SLOTS_WITH_THE_DOTS_ARM: &str =
    "config.json plus the two window shards go through train_weights_push like the dots \
     snapshot, and the finished adapter lands where train_adapter_safetensors reads it; only \
     the samples differ, which is why e4b_sample_stage takes a window dump file whole";

enum StagedTrainSample {
    Rows(TrainSample),
    Page { spec: PageSample, mode: DotsMode },
}

fn js_err(e: impl core::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn emit(cb: &Function, json: String) {
    if let Ok(v) = js_sys::JSON::parse(&json) {
        let _ = cb.call1(&JsValue::NULL, &v);
    }
}

fn session_info_json(s: &OcrWebSession) -> String {
    let adapter = match s.adapter() {
        Some(a) => serde_json::json!({"matched": a.matched, "rank": a.rank}),
        None => serde_json::Value::Null,
    };
    serde_json::json!({"kv_span": s.kv_span(), "adapter": adapter}).to_string()
}

#[wasm_bindgen]
pub fn weights_clear() {
    STAGE.with(|s| s.borrow_mut().clear());
}

#[wasm_bindgen]
pub fn weights_push(name: &str, chunk: &[u8]) {
    STAGE.with(|s| {
        s.borrow_mut()
            .entry(name.to_string())
            .or_default()
            .extend_from_slice(chunk)
    });
}

#[wasm_bindgen]
pub fn weights_staged() -> String {
    STAGE.with(|s| {
        let listing: Vec<serde_json::Value> = s
            .borrow()
            .iter()
            .map(|(k, v)| serde_json::json!({"name": k, "size": v.len()}))
            .collect();
        serde_json::Value::Array(listing).to_string()
    })
}

#[wasm_bindgen]
pub fn adapter_stage(config_json: String, safetensors: &[u8]) {
    ADAPTER.with(|a| {
        *a.borrow_mut() = Some(AdapterBytes {
            config_json,
            safetensors: safetensors.to_vec(),
        })
    });
}

#[wasm_bindgen]
pub fn adapter_clear() {
    ADAPTER.with(|a| *a.borrow_mut() = None);
}

#[wasm_bindgen]
pub fn adapter_staged() -> bool {
    ADAPTER.with(|a| a.borrow().is_some())
}

fn webgpu_available() -> bool {
    js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("navigator"))
        .ok()
        .filter(|n| !n.is_undefined())
        .and_then(|n| js_sys::Reflect::get(&n, &JsValue::from_str("gpu")).ok())
        .map(|g| !g.is_undefined())
        .unwrap_or(false)
}

#[wasm_bindgen]
pub async fn engine_init(opts_json: String) -> Result<String, JsValue> {
    if !webgpu_available() {
        return Err(js_err(
            "navigator.gpu is missing: WebGPU is required (wgpu's web backend aborts without it)",
        ));
    }
    let v: serde_json::Value = serde_json::from_str(&opts_json).map_err(js_err)?;
    let mut opts = OcrWebOptions::default();
    if let Some(n) = v.get("kv_span").and_then(|x| x.as_u64()) {
        opts.kv_span = n as usize;
    }
    let stage = STAGE.with(|s| std::mem::take(&mut *s.borrow_mut()));
    if stage.is_empty() {
        return Err(js_err(
            "no weights staged: push ocr-model bytes through weights_push first",
        ));
    }
    let adapter = ADAPTER.with(|a| a.borrow().clone());
    let session = init_with_adapter(stage, opts, adapter)
        .await
        .map_err(js_err)?;
    let info = session_info_json(&session);
    SESSION.with(|s| *s.borrow_mut() = Some(session));
    Ok(info)
}

#[wasm_bindgen]
pub fn engine_ready() -> bool {
    SESSION.with(|s| s.borrow().is_some())
}

#[wasm_bindgen]
pub fn engine_info() -> Option<String> {
    SESSION.with(|s| s.borrow().as_ref().map(session_info_json))
}

#[wasm_bindgen]
pub async fn engine_ocr_page(image: Vec<u8>, on_frame: Function) -> Result<String, JsValue> {
    let _ = EVERY_BROWSER_READBACK_RIDES_THE_PUMP_SO_NOTHING_HERE_BLOCKS_THE_EVENT_LOOP;
    let mut session = SESSION
        .with(|s| s.borrow_mut().take())
        .ok_or_else(|| js_err("engine not initialized: call engine_init first"))?;
    let mut on_region = |e: nv_models::dots_ocr::LayoutElement| {
        emit(
            &on_frame,
            serde_json::json!({
                "type": "region",
                "bbox": e.bbox,
                "category": e.category,
                "text": e.text,
            })
            .to_string(),
        );
    };
    let res = session.ocr_page_pumped(&image, &mut on_region).await;
    SESSION.with(|s| *s.borrow_mut() = Some(session));
    let report = res.map_err(js_err)?;
    Ok(serde_json::json!({
        "regions": report.regions,
        "prompt_tokens": report.prompt_tokens,
        "generated_tokens": report.generated_tokens,
        "looped": report.looped,
        "hit_eos": report.hit_eos,
    })
    .to_string())
}

thread_local! {
    static Q38_STAGE: RefCell<BTreeMap<String, Vec<u8>>> =
        const { RefCell::new(BTreeMap::new()) };
    static Q38_MODEL: RefCell<Option<crate::q38::Q38ParityModel>> = const { RefCell::new(None) };
}

#[wasm_bindgen]
pub fn q38_weights_clear() {
    Q38_STAGE.with(|s| s.borrow_mut().clear());
    Q38_MODEL.with(|m| *m.borrow_mut() = None);
}

#[wasm_bindgen]
pub fn q38_install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let msg = info.to_string();
        if let Ok(console) = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("console"))
        {
            if let Ok(error) = js_sys::Reflect::get(&console, &JsValue::from_str("error")) {
                if let Some(f) = error.dyn_ref::<Function>() {
                    let _ = f.call1(&console, &JsValue::from_str(&format!("RUST_PANIC|{msg}")));
                }
            }
        }
    }));
}

#[wasm_bindgen]
pub async fn q38_tint_probe() -> Result<String, JsValue> {
    let ctx = ensure_browser_gpu().await.map_err(js_err)?;
    let sg = ctx.caps.subgroup;
    let mut sources = nv_models::qwen3_5_dense_wgpu::browser_module_sources(sg);
    sources.push((
        "wk:gemv_w4a16_affine".to_string(),
        nv_kernels::wgpu_backend::kernels::gemv_w4a16_affine::source(),
    ));
    sources.push((
        "wk:gemm_w4a16_affine".to_string(),
        nv_kernels::wgpu_backend::kernels::gemm_w4a16_affine::source(),
    ));
    let mut out: Vec<serde_json::Value> = Vec::new();
    for (tag, src) in sources {
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = nv_kernels::wgpu_backend::dispatch::create_module_honoring_unchecked_optin(
            ctx, &tag, &src,
        );
        if let Some(e) = scope.pop().await {
            out.push(serde_json::json!({"tag": tag, "err": e.to_string()}));
            continue;
        }
        let mut entries: Vec<String> = Vec::new();
        let mut rest = src.as_str();
        while let Some(i) = rest.find("@compute") {
            rest = &rest[i..];
            if let Some(f) = rest.find("fn ") {
                let name: String = rest[f + 3..]
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                entries.push(name);
                rest = &rest[f + 3..];
            } else {
                break;
            }
        }
        for entry in entries {
            let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let _p = ctx
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&tag),
                    layout: None,
                    module: &module,
                    entry_point: Some(&entry),
                    compilation_options: Default::default(),
                    cache: None,
                });
            if let Some(e) = scope.pop().await {
                out.push(serde_json::json!({
                    "tag": format!("{tag}::{entry}"),
                    "err": e.to_string(),
                }));
            }
            if out.len() > 24 {
                break;
            }
        }
        if out.len() > 24 {
            break;
        }
    }
    let l = ctx.device.limits();
    let (limited, out): (Vec<serde_json::Value>, Vec<serde_json::Value>) =
        out.into_iter().partition(|f| {
            let e = f["err"].as_str().unwrap_or("");
            e.contains("exceeds the maximum") || e.contains("larger than the maximum allowed")
        });
    Ok(serde_json::json!({
        "sg": sg,
        "adapter": ctx.summary(),
        "limited": limited,
        "limits": {
            "max_compute_workgroup_size_x": l.max_compute_workgroup_size_x,
            "max_compute_invocations_per_workgroup": l.max_compute_invocations_per_workgroup,
            "max_compute_workgroup_storage_size": l.max_compute_workgroup_storage_size,
            "max_storage_buffers_per_shader_stage": l.max_storage_buffers_per_shader_stage,
            "max_storage_buffer_binding_size": l.max_storage_buffer_binding_size,
            "max_buffer_size": l.max_buffer_size,
        },
        "failures": out,
    })
    .to_string())
}

#[wasm_bindgen]
pub fn q38_weights_reserve(name: &str, size: u32) {
    Q38_STAGE.with(|s| {
        s.borrow_mut()
            .entry(name.to_string())
            .or_default()
            .reserve_exact(size as usize)
    });
}

#[wasm_bindgen]
pub fn q38_weights_push(name: &str, chunk: &[u8]) {
    Q38_STAGE.with(|s| {
        s.borrow_mut()
            .entry(name.to_string())
            .or_default()
            .extend_from_slice(chunk)
    });
}

#[wasm_bindgen]
pub fn q38_weights_staged() -> String {
    Q38_STAGE.with(|s| {
        let listing: Vec<serde_json::Value> = s
            .borrow()
            .iter()
            .map(|(k, v)| serde_json::json!({"name": k, "size": v.len()}))
            .collect();
        serde_json::Value::Array(listing).to_string()
    })
}

thread_local! {
    static Q38_EXTERN: RefCell<BTreeMap<String, js_sys::Uint8Array>> =
        const { RefCell::new(BTreeMap::new()) };
    static Q38_EXTERN_CONFIG: RefCell<Option<String>> = const { RefCell::new(None) };
}

pub const EXTERN_SHARDS_STAY_IN_JS_HEAP_SO_16_GB_NEVER_TRANSITS_WASM32_LINEAR_MEMORY: &str =
    "q38_extern_set holds each fetched shard as a JS-side Uint8Array handle; boot opens an \
     extern WeightLoader whose read closures copy only the requested tensor ranges into wasm \
     transiently (subarray + copy_to), so the wasm heap peak is one packing transient, not the \
     model";

#[wasm_bindgen]
pub fn q38_extern_clear() {
    Q38_EXTERN.with(|s| s.borrow_mut().clear());
    Q38_EXTERN_CONFIG.with(|c| *c.borrow_mut() = None);
    Q38_MODEL.with(|m| *m.borrow_mut() = None);
}

#[wasm_bindgen]
pub fn q38_extern_config(config_json: String) {
    Q38_EXTERN_CONFIG.with(|c| *c.borrow_mut() = Some(config_json));
}

#[wasm_bindgen]
pub fn q38_extern_set(name: &str, bytes: js_sys::Uint8Array) {
    Q38_EXTERN.with(|s| s.borrow_mut().insert(name.to_string(), bytes));
}

#[wasm_bindgen]
pub fn q38_extern_staged() -> String {
    Q38_EXTERN.with(|s| {
        let listing: Vec<serde_json::Value> = s
            .borrow()
            .iter()
            .map(|(k, v)| serde_json::json!({"name": k, "size": v.length()}))
            .collect();
        serde_json::Value::Array(listing).to_string()
    })
}

#[wasm_bindgen]
pub async fn q38_extern_boot(opts_json: String) -> Result<String, JsValue> {
    let _ = EXTERN_SHARDS_STAY_IN_JS_HEAP_SO_16_GB_NEVER_TRANSITS_WASM32_LINEAR_MEMORY;
    if !webgpu_available() {
        return Err(js_err(
            "navigator.gpu is missing: WebGPU is required (wgpu's web backend aborts without it)",
        ));
    }
    let v: serde_json::Value = serde_json::from_str(&opts_json).map_err(js_err)?;
    let kv_span = v
        .get("kv_span")
        .and_then(|x| x.as_u64())
        .ok_or_else(|| js_err("q38 extern opts need kv_span"))? as usize;
    let config_json = Q38_EXTERN_CONFIG
        .with(|c| c.borrow().clone())
        .ok_or_else(|| js_err("no q38 config staged: call q38_extern_config first"))?;
    let handles: Vec<(String, js_sys::Uint8Array)> =
        Q38_EXTERN.with(|s| s.borrow().clone().into_iter().collect());
    if handles.is_empty() {
        return Err(js_err(
            "no q38 extern shards staged: hand each fetched Uint8Array to q38_extern_set first",
        ));
    }
    let named: Vec<(String, nv_weights::ExternBytes)> = handles
        .into_iter()
        .filter(|(name, _)| name.ends_with(".safetensors"))
        .map(|(name, arr)| {
            let len = arr.byte_length() as u64;
            let read: nv_weights::ExternReadFn = Box::new(move |off: u64, dst: &mut [u8]| {
                let end = off
                    .checked_add(dst.len() as u64)
                    .filter(|e| *e <= arr.byte_length() as u64)
                    .ok_or_else(|| anyhow::anyhow!("extern read past the end of a staged shard"))?;
                arr.subarray(off as u32, end as u32).copy_to(dst);
                Ok(())
            });
            (name, nv_weights::ExternBytes { len, read })
        })
        .collect();
    let loader = WeightLoader::open_extern_shards(named, &Device::Cpu).map_err(js_err)?;
    let (model, boot) = crate::q38::q38_boot_from_loader(&config_json, &loader, kv_span)
        .await
        .map_err(|e| js_err(format!("{e:#}")))?;
    Q38_MODEL.with(|m| *m.borrow_mut() = Some(model));
    Ok(serde_json::json!({
        "caps_summary": boot.caps_summary,
        "vocab": boot.vocab,
        "layer_types": boot.layer_types,
    })
    .to_string())
}

#[wasm_bindgen]
pub async fn q38_parity_boot(opts_json: String) -> Result<String, JsValue> {
    if !webgpu_available() {
        return Err(js_err(
            "navigator.gpu is missing: WebGPU is required (wgpu's web backend aborts without it)",
        ));
    }
    let v: serde_json::Value = serde_json::from_str(&opts_json).map_err(js_err)?;
    if v.get("subgroups_ungranted").and_then(|x| x.as_bool()) == Some(true) {
        nv_kernels::wgpu_backend::device::browser_force_subgroups_ungranted();
    }
    let prompt_len = v
        .get("prompt_len")
        .and_then(|x| x.as_u64())
        .ok_or_else(|| js_err("q38 parity opts need prompt_len"))? as usize;
    let steps = v
        .get("steps")
        .and_then(|x| x.as_u64())
        .ok_or_else(|| js_err("q38 parity opts need steps"))? as usize;
    let max_seq = match v.get("kv_span").and_then(|x| x.as_u64()) {
        Some(n) => n as usize,
        None => {
            prompt_len + steps + crate::q38::Q38_PARITY_MAX_SEQ_SLACK_KEEPS_THE_KV_AND_SCRATCH_SMALL
        }
    };
    let stage = Q38_STAGE.with(|s| std::mem::take(&mut *s.borrow_mut()));
    if stage.is_empty() {
        return Err(js_err(
            "no q38 weights staged: push the slice shards through q38_weights_push first",
        ));
    }
    let (model, boot) = crate::q38::q38_parity_boot(stage, max_seq)
        .await
        .map_err(|e| js_err(format!("{e:#}")))?;
    Q38_MODEL.with(|m| *m.borrow_mut() = Some(model));
    Ok(serde_json::json!({
        "caps_summary": boot.caps_summary,
        "vocab": boot.vocab,
        "layer_types": boot.layer_types,
    })
    .to_string())
}

async fn q38_with_model<F, Fut>(f: F) -> Result<u32, JsValue>
where
    F: FnOnce(crate::q38::Q38ParityModel) -> Fut,
    Fut: core::future::Future<Output = (crate::q38::Q38ParityModel, anyhow::Result<u32>)>,
{
    let model = Q38_MODEL
        .with(|m| m.borrow_mut().take())
        .ok_or_else(|| js_err("q38 model not booted: call q38_parity_boot first"))?;
    let (model, res) = f(model).await;
    Q38_MODEL.with(|m| *m.borrow_mut() = Some(model));
    res.map_err(js_err)
}

#[wasm_bindgen]
pub async fn q38_parity_prefill(prompt: Vec<u32>) -> Result<u32, JsValue> {
    q38_with_model(|mut model| async move {
        let r = model.prefill(&prompt).await;
        (model, r)
    })
    .await
}

#[wasm_bindgen]
pub async fn q38_parity_step() -> Result<u32, JsValue> {
    q38_with_model(|mut model| async move {
        let r = model.decode_step().await;
        (model, r)
    })
    .await
}

#[wasm_bindgen]
pub async fn q38_parity_step_quiet() -> Result<u32, JsValue> {
    q38_with_model(|mut model| async move {
        let r = model.decode_step_quiet().await;
        (model, r)
    })
    .await
}

#[wasm_bindgen]
pub fn q38_parity_reset() -> Result<(), JsValue> {
    Q38_MODEL.with(|m| {
        m.borrow_mut()
            .as_mut()
            .ok_or_else(|| js_err("q38 model not booted"))?
            .reset_for_census_determinism_replay()
            .map_err(|e| js_err(format!("{e:#}")))
    })
}

#[wasm_bindgen]
pub fn q38_parity_labels() -> Result<String, JsValue> {
    Q38_MODEL.with(|m| {
        m.borrow()
            .as_ref()
            .map(|model| {
                serde_json::Value::Array(
                    model
                        .step_pass_labels()
                        .into_iter()
                        .map(serde_json::Value::String)
                        .collect(),
                )
                .to_string()
            })
            .ok_or_else(|| js_err("q38 model not booted"))
    })
}

#[wasm_bindgen]
pub fn q38_logits_bin() -> Result<Vec<u8>, JsValue> {
    Q38_MODEL.with(|m| {
        m.borrow()
            .as_ref()
            .map(|model| model.logits_le().to_vec())
            .ok_or_else(|| js_err("q38 model not booted"))
    })
}

#[wasm_bindgen]
pub fn train_weights_clear() {
    TRAIN_STAGE.with(|s| s.borrow_mut().clear());
}

#[wasm_bindgen]
pub fn train_weights_reserve(name: &str, size: u32) {
    TRAIN_STAGE.with(|s| {
        s.borrow_mut()
            .entry(name.to_string())
            .or_default()
            .reserve_exact(size as usize)
    });
}

#[wasm_bindgen]
pub fn train_weights_push(name: &str, chunk: &[u8]) {
    TRAIN_STAGE.with(|s| {
        s.borrow_mut()
            .entry(name.to_string())
            .or_default()
            .extend_from_slice(chunk)
    });
}

#[wasm_bindgen]
pub fn train_weights_staged() -> String {
    TRAIN_STAGE.with(|s| {
        let listing: Vec<serde_json::Value> = s
            .borrow()
            .iter()
            .map(|(k, v)| serde_json::json!({"name": k, "size": v.len()}))
            .collect();
        serde_json::Value::Array(listing).to_string()
    })
}

#[wasm_bindgen]
pub fn train_samples_clear() {
    TRAIN_SAMPLES.with(|s| s.borrow_mut().clear());
}

#[wasm_bindgen]
pub fn train_samples_staged() -> usize {
    TRAIN_SAMPLES.with(|s| s.borrow().len())
}

fn parse_mode(v: Option<&str>) -> Result<DotsMode, JsValue> {
    match v.unwrap_or("plain") {
        "plain" => Ok(DotsMode::PlainOcr),
        "layout-all" => Ok(DotsMode::LayoutAll),
        "layout-only" => Ok(DotsMode::LayoutOnly),
        other => Err(js_err(format!(
            "unknown sample mode {other} (plain|layout-all|layout-only)"
        ))),
    }
}

#[wasm_bindgen]
pub fn train_sample_stage_rows(meta_json: String, rows: &[u8]) -> Result<(), JsValue> {
    let v: serde_json::Value = serde_json::from_str(&meta_json).map_err(js_err)?;
    let ids: Vec<u32> = v
        .get("ids")
        .and_then(|x| x.as_array())
        .ok_or_else(|| js_err("sample meta needs an ids array"))?
        .iter()
        .map(|x| x.as_u64().map(|n| n as u32))
        .collect::<Option<_>>()
        .ok_or_else(|| js_err("ids must be unsigned integers"))?;
    let get = |k: &str| -> Result<usize, JsValue> {
        v.get(k)
            .and_then(|x| x.as_u64())
            .map(|n| n as usize)
            .ok_or_else(|| js_err(format!("sample meta needs {k}")))
    };
    let completion_start = get("completion_start")?;
    let vision_start = get("vision_start")?;
    let n_vis = get("n_vis")?;
    if !rows.len().is_multiple_of(4) {
        return Err(js_err("rows bytes must be little-endian f32"));
    }
    let rows: Vec<f32> = rows
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    TRAIN_SAMPLES.with(|s| {
        s.borrow_mut().push(StagedTrainSample::Rows(TrainSample {
            ids,
            completion_start,
            vision_start,
            n_vis,
            rows,
        }))
    });
    Ok(())
}

#[wasm_bindgen]
pub fn train_sample_stage_page(meta_json: String, image: &[u8]) -> Result<(), JsValue> {
    let v: serde_json::Value = serde_json::from_str(&meta_json).map_err(js_err)?;
    let text = v
        .get("text")
        .and_then(|x| x.as_str())
        .ok_or_else(|| js_err("sample meta needs the corrected text"))?
        .to_string();
    let mode = parse_mode(v.get("mode").and_then(|x| x.as_str()))?;
    let bbox = match v.get("bbox").and_then(|x| x.as_array()) {
        None => None,
        Some(a) => {
            let b: Vec<f64> = a.iter().filter_map(|x| x.as_f64()).collect();
            if b.len() != 4 {
                return Err(js_err("bbox must be [x1, y1, x2, y2]"));
            }
            Some([b[0], b[1], b[2], b[3]])
        }
    };
    TRAIN_SAMPLES.with(|s| {
        s.borrow_mut().push(StagedTrainSample::Page {
            spec: PageSample {
                image: image.to_vec(),
                text,
                bbox,
            },
            mode,
        })
    });
    Ok(())
}

fn parse_train_options(opts_json: &str) -> Result<TrainOptions, JsValue> {
    parse_train_options_over(opts_json, TrainOptions::default())
}

fn parse_train_options_over(
    opts_json: &str,
    mut opts: TrainOptions,
) -> Result<TrainOptions, JsValue> {
    let v: serde_json::Value = serde_json::from_str(opts_json).map_err(js_err)?;
    if let Some(n) = v.get("steps").and_then(|x| x.as_u64()) {
        opts.steps = n as usize;
    }
    if let Some(n) = v.get("rank").and_then(|x| x.as_u64()) {
        opts.rank = n as usize;
    }
    if let Some(n) = v.get("grad_layers").and_then(|x| x.as_u64()) {
        opts.grad_layers = n as usize;
    }
    if let Some(n) = v.get("seed").and_then(|x| x.as_u64()) {
        opts.seed = n;
    }
    if let Some(x) = v.get("alpha").and_then(|x| x.as_f64()) {
        opts.alpha = x;
    }
    if let Some(x) = v.get("lr").and_then(|x| x.as_f64()) {
        opts.lr = x;
    }
    if let Some(x) = v.get("b_lr_ratio").and_then(|x| x.as_f64()) {
        opts.b_lr_ratio = x;
    }
    Ok(opts)
}

#[wasm_bindgen]
pub async fn engine_train(opts_json: String, on_step: Function) -> Result<String, JsValue> {
    let _ =
        crate::train::TRAINER_IS_AN_EXPLICIT_SCHEDULE_OVER_THE_QLORA_WGSL_KERNELS_NOT_AN_AUTOGRAD;
    let opts = parse_train_options(&opts_json)?;
    let (config_json, loader) =
        take_trainer_stage("the mlx dots.ocr-4bit snapshot (config.json + safetensors)")?;
    let staged = TRAIN_SAMPLES.with(|s| std::mem::take(&mut *s.borrow_mut()));
    if staged.is_empty() {
        return Err(js_err(
            "no training samples staged: stage dump rows or corrected pages first",
        ));
    }
    let ctx = ensure_browser_gpu().await.map_err(js_err)?;
    let mut samples: Vec<TrainSample> = Vec::with_capacity(staged.len());
    for st in staged {
        match st {
            StagedTrainSample::Rows(s) => samples.push(s),
            StagedTrainSample::Page { spec, mode } => {
                let session = SESSION.with(|s| s.borrow_mut().take()).ok_or_else(|| {
                    js_err(
                        "page samples ride the serving session's vision tower: call engine_init \
                         before engine_train, or stage precomputed rows instead",
                    )
                })?;
                let built = session.build_train_sample_pumped(&spec, mode).await;
                SESSION.with(|s| *s.borrow_mut() = Some(session));
                samples.push(built.map_err(js_err)?);
            }
        }
    }
    let report = run_qlora_pumped(
        ctx,
        &loader,
        &config_json,
        &samples,
        &opts,
        &mut |step, loss| {
            emit(
                &on_step,
                serde_json::json!({"type": "train_step", "step": step, "loss": loss}).to_string(),
            );
        },
    )
    .await
    .map_err(js_err)?;
    TRAIN_ADAPTER.with(|a| {
        *a.borrow_mut() = Some((
            report.adapter_config_json.clone(),
            report.adapter_safetensors.clone(),
        ))
    });
    Ok(serde_json::json!({
        "losses": report.losses,
        "all_grads_finite": report.all_grads_finite,
        "window_start": report.window_start,
        "modules": report.modules,
        "samples": samples.len(),
        "adapter_safetensors_bytes": report.adapter_safetensors.len(),
    })
    .to_string())
}

fn take_trainer_stage(what: &str) -> Result<(String, WeightLoader), JsValue> {
    let stage = TRAIN_STAGE.with(|s| std::mem::take(&mut *s.borrow_mut()));
    if stage.is_empty() {
        return Err(js_err(format!(
            "no trainer weights staged: push {what} through train_weights_push first"
        )));
    }
    let config_json = stage
        .get(crate::CONFIG_JSON_NAME)
        .and_then(|b| String::from_utf8(b.clone()).ok())
        .ok_or_else(|| js_err("trainer stage carries no config.json"))?;
    let shards: Vec<(String, Vec<u8>)> = stage
        .into_iter()
        .filter(|(name, _)| name.ends_with(".safetensors"))
        .collect();
    if shards.is_empty() {
        return Err(js_err("trainer stage carries no .safetensors shards"));
    }
    let loader = WeightLoader::open_from_named_bytes(shards, &Device::Cpu).map_err(js_err)?;
    Ok((config_json, loader))
}

#[wasm_bindgen]
pub fn e4b_samples_clear() {
    E4B_SAMPLES.with(|s| s.borrow_mut().clear());
}

#[wasm_bindgen]
pub fn e4b_samples_staged() -> usize {
    E4B_SAMPLES.with(|s| s.borrow().len())
}

#[wasm_bindgen]
pub fn e4b_sample_stage(safetensors: &[u8]) -> Result<(), JsValue> {
    let sample = E4bWindowSample::from_safetensors_bytes(safetensors).map_err(js_err)?;
    E4B_SAMPLES.with(|s| s.borrow_mut().push(sample));
    Ok(())
}

#[wasm_bindgen]
pub async fn e4b_train(opts_json: String, on_step: Function) -> Result<String, JsValue> {
    let _ = E4B_TRAIN_SHARES_THE_TRAINER_STAGE_AND_ADAPTER_SLOTS_WITH_THE_DOTS_ARM;
    let opts = parse_train_options_over(&opts_json, e4b_train_options())?;
    let (config_json, loader) =
        take_trainer_stage("the E4B config.json and the two window shards")?;
    let samples = E4B_SAMPLES.with(|s| std::mem::take(&mut *s.borrow_mut()));
    if samples.is_empty() {
        return Err(js_err(
            "no E4B window samples staged: hand each dump file to e4b_sample_stage first",
        ));
    }
    let ctx = ensure_browser_gpu().await.map_err(js_err)?;
    let report = run_e4b_window_qlora_pumped(
        ctx,
        &loader,
        &config_json,
        &samples,
        &opts,
        &mut |step, loss| {
            emit(
                &on_step,
                serde_json::json!({"type": "train_step", "step": step, "loss": loss}).to_string(),
            );
        },
    )
    .await
    .map_err(js_err)?;
    TRAIN_ADAPTER.with(|a| {
        *a.borrow_mut() = Some((
            report.adapter_config_json.clone(),
            report.adapter_safetensors.clone(),
        ))
    });
    Ok(serde_json::json!({
        "losses": report.losses,
        "all_grads_finite": report.all_grads_finite,
        "window_start": report.window_start,
        "modules": report.modules,
        "samples": samples.len(),
        "adapter_safetensors_bytes": report.adapter_safetensors.len(),
    })
    .to_string())
}

#[wasm_bindgen]
pub fn train_adapter_config_json() -> Option<String> {
    TRAIN_ADAPTER.with(|a| {
        a.borrow()
            .as_ref()
            .and_then(|(cfg, _)| String::from_utf8(cfg.clone()).ok())
    })
}

#[wasm_bindgen]
pub fn train_adapter_safetensors() -> Option<Vec<u8>> {
    TRAIN_ADAPTER.with(|a| a.borrow().as_ref().map(|(_, st)| st.clone()))
}
