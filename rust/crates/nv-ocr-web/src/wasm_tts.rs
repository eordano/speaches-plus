use js_sys::Function;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::voxtral_tts_web as web;
use nv_kernels::wgpu_backend::dispatch::ReadbackArm;

fn js_err(e: impl core::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn ctx_err(e: anyhow::Error) -> JsValue {
    JsValue::from_str(&format!("{e:#}"))
}

#[wasm_bindgen]
pub fn tts_clear() {
    web::tts_clear()
}

#[wasm_bindgen]
pub fn tts_stage_text(name: &str, text: String) {
    web::tts_stage_text(name, text)
}

fn range_reader(reader: Function) -> nv_weights::ExternReadFn {
    Box::new(move |off: u64, dst: &mut [u8]| {
        let got = reader
            .call2(&JsValue::NULL, &JsValue::from_f64(off as f64), &JsValue::from_f64(dst.len() as f64))
            .map_err(|e| anyhow::anyhow!("range reader threw: {:?}", e.as_string().unwrap_or_default()))?;
        let arr: js_sys::Uint8Array = got
            .dyn_into()
            .map_err(|_| anyhow::anyhow!("range reader returned something other than a Uint8Array"))?;
        anyhow::ensure!(
            arr.length() as usize == dst.len(),
            "range reader returned {} bytes for a {}-byte read at {off}",
            arr.length(),
            dst.len()
        );
        arr.copy_to(dst);
        Ok(())
    })
}

#[wasm_bindgen]
pub async fn tts_open_checkpoint(shards: js_sys::Array) -> Result<(), JsValue> {
    crate::ensure_browser_gpu().await.map_err(ctx_err)?;
    let field = |o: &JsValue, k: &str| js_sys::Reflect::get(o, &JsValue::from_str(k)).map_err(|_| js_err(format!("shard lacks {k}")));
    let mut out = Vec::new();
    for o in shards.iter() {
        let name = field(&o, "name")?.as_string().ok_or_else(|| js_err("shard name is not a string"))?;
        let len = field(&o, "len")?.as_f64().ok_or_else(|| js_err("shard len is not a number"))? as u64;
        let read: Function = field(&o, "read")?.dyn_into().map_err(|_| js_err("shard read is not a function"))?;
        out.push((name, len, range_reader(read)));
    }
    web::tts_open_checkpoint(out).map_err(ctx_err)
}

#[wasm_bindgen]
pub fn tts_codec_boot() -> Result<String, JsValue> {
    web::codec_boot().map_err(ctx_err)
}

#[wasm_bindgen]
pub async fn tts_codec_decode(codes: Vec<i32>) -> Result<Vec<f32>, JsValue> {
    web::codec_decode(codes, ReadbackArm::Pumped).await.map_err(ctx_err)
}

#[wasm_bindgen]
pub fn tts_codec_stream_reset() -> Result<(), JsValue> {
    web::codec_stream_reset().map_err(js_err)
}

#[wasm_bindgen]
pub async fn tts_codec_decode_frames(frames: Vec<i32>) -> Result<Vec<f32>, JsValue> {
    web::codec_decode_frames(frames, ReadbackArm::Pumped).await.map_err(ctx_err)
}

#[wasm_bindgen]
pub fn tts_boot(with_backbone: bool) -> Result<String, JsValue> {
    web::tts_boot(with_backbone).map_err(ctx_err)
}

#[wasm_bindgen]
pub fn tts_stage_voice_torch(voice: &str, pt_bytes: &[u8]) -> Result<(), JsValue> {
    web::tts_stage_voice_torch(voice, pt_bytes).map_err(ctx_err)
}

#[wasm_bindgen]
pub async fn tts_frame_from_hidden(hidden: Vec<f32>, x0: Vec<f32>, trace: bool) -> Result<String, JsValue> {
    web::tts_frame_from_hidden(hidden, x0, trace, ReadbackArm::Pumped).await.map_err(ctx_err)
}

fn request(text: String, voice: String, noise_rows: Vec<f32>, seed: u64, max_frames: u32, teacher_frames: Vec<u32>) -> web::SynthRequest {
    web::SynthRequest { text, voice, noise_rows, seed, max_frames: max_frames as usize, teacher_frames }
}

#[wasm_bindgen]
pub async fn tts_synthesize(text: String, voice: String, noise_rows: Vec<f32>, seed: u64, max_frames: u32, teacher_frames: Vec<u32>) -> Result<String, JsValue> {
    web::tts_synthesize(request(text, voice, noise_rows, seed, max_frames, teacher_frames), ReadbackArm::Pumped)
        .await
        .map(|v| v.to_string())
        .map_err(ctx_err)
}

#[wasm_bindgen]
pub async fn tts_speak(text: String, voice: String, seed: u64, max_frames: u32) -> Result<Vec<f32>, JsValue> {
    web::tts_speak(request(text, voice, Vec::new(), seed, max_frames, Vec::new()), ReadbackArm::Pumped).await.map_err(ctx_err)
}
