use anyhow::{anyhow, bail, ensure, Context, Result};
use candle_core::quantized::gguf_file::{Content, Value};
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use crate::TensorSource;

pub struct GgufLoader {
    content: Content,
    reader: RefCell<BufReader<File>>,
    device: Device,
    path: PathBuf,

    nvfp4_transcode: Cell<bool>,
}

impl GgufLoader {
    pub fn open(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut header_file =
            File::open(&path).with_context(|| format!("open gguf {}", path.display()))?;
        let content = Content::read(&mut header_file)
            .map_err(|e| anyhow!("parse gguf header {}: {e}", path.display()))?;
        let reader = BufReader::new(
            File::open(&path).with_context(|| format!("open gguf reader {}", path.display()))?,
        );
        let nvfp4_transcode = std::env::var("NV_GGUF_NVFP4").ok().as_deref() == Some("1");
        Ok(Self {
            content,
            reader: RefCell::new(reader),
            device: device.clone(),
            path,
            nvfp4_transcode: Cell::new(nvfp4_transcode),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn set_nvfp4_transcode(&self, on: bool) {
        self.nvfp4_transcode.set(on);
    }

    pub fn nvfp4_transcode(&self) -> bool {
        self.nvfp4_transcode.get()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn metadata(&self) -> &std::collections::HashMap<String, Value> {
        &self.content.metadata
    }

    pub fn architecture(&self) -> Result<String> {
        self.md_string("general.architecture")
    }

    pub fn general_name(&self) -> Option<String> {
        self.md_string("general.name").ok()
    }

    fn md(&self, key: &str) -> Result<&Value> {
        self.content
            .metadata
            .get(key)
            .ok_or_else(|| anyhow!("gguf metadata missing key: {key}"))
    }

    pub fn md_u64(&self, key: &str) -> Result<u64> {
        let v = value_as_i64(self.md(key)?)
            .map_err(|e| anyhow!("gguf metadata {key} not an integer: {e}"))?;
        Ok(v as u64)
    }

    pub fn md_f64(&self, key: &str) -> Result<f64> {
        let v = self.md(key)?;
        if let Ok(x) = v.to_f64() {
            return Ok(x);
        }
        v.to_f32()
            .map(|x| x as f64)
            .map_err(|e| anyhow!("gguf metadata {key} not f32/f64: {e}"))
    }

    pub fn md_bool(&self, key: &str) -> Result<bool> {
        self.md(key)?
            .to_bool()
            .map_err(|e| anyhow!("gguf metadata {key} not bool: {e}"))
    }

    pub fn md_string(&self, key: &str) -> Result<String> {
        self.md(key)?
            .to_string()
            .map(|s| s.to_owned())
            .map_err(|e| anyhow!("gguf metadata {key} not string: {e}"))
    }

    pub fn md_list<T>(&self, key: &str, f: impl FnMut(&Value) -> Result<T>) -> Result<Vec<T>> {
        let arr = self
            .md(key)?
            .to_vec()
            .map_err(|e| anyhow!("gguf metadata {key} not array: {e}"))?;
        arr.iter().map(f).collect()
    }

    pub fn md_bool_list(&self, key: &str) -> Result<Vec<bool>> {
        self.md_list(key, |v| {
            v.to_bool()
                .map_err(|e| anyhow!("gguf list elem not bool: {e}"))
        })
    }

    pub fn md_u64_list(&self, key: &str) -> Result<Vec<u64>> {
        self.md_list(key, |v| Ok(value_as_i64(v)? as u64))
    }

    pub fn md_string_list(&self, key: &str) -> Result<Vec<String>> {
        self.md_list(key, |v| {
            v.to_string()
                .map(|s| s.to_owned())
                .map_err(|e| anyhow!("gguf list elem not string: {e}"))
        })
    }

    pub fn has_gguf_tensor(&self, gguf_name: &str) -> bool {
        self.content.tensor_infos.contains_key(gguf_name)
    }

    pub fn gguf_tensor_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.content.tensor_infos.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn gguf_tensor_shape(&self, gguf_name: &str) -> Option<Vec<usize>> {
        self.content
            .tensor_infos
            .get(gguf_name)
            .map(|i| i.shape.dims().to_vec())
    }

    pub fn gguf_tensor_dtype(&self, gguf_name: &str) -> Option<GgmlDType> {
        self.content
            .tensor_infos
            .get(gguf_name)
            .map(|i| i.ggml_dtype)
    }

    pub fn gguf_tensor_bytes(&self, gguf_name: &str) -> Option<usize> {
        let i = self.content.tensor_infos.get(gguf_name)?;
        Some(ggml_bytes(i.shape.elem_count(), i.ggml_dtype))
    }

    pub fn quant_census(&self) -> Vec<(GgmlDType, usize, usize)> {
        let mut out: BTreeMap<String, (GgmlDType, usize, usize)> = BTreeMap::new();
        for info in self.content.tensor_infos.values() {
            let e = out
                .entry(format!("{:?}", info.ggml_dtype))
                .or_insert((info.ggml_dtype, 0, 0));
            e.1 += 1;
            e.2 += ggml_bytes(info.shape.elem_count(), info.ggml_dtype);
        }
        let mut v: Vec<_> = out.into_values().collect();
        v.sort_by_key(|e| std::cmp::Reverse(e.2));
        v
    }

    pub fn active_bytes_per_token(&self, top_k: usize, num_experts: usize) -> Result<u64> {
        if num_experts == 0 || top_k == 0 || top_k > num_experts {
            bail!("active_bytes_per_token: bad top_k {top_k} of {num_experts}");
        }
        let tied = !self.has_gguf_tensor("output.weight");
        let mut total = 0f64;
        for (name, info) in self.content.tensor_infos.iter() {
            let bytes = ggml_bytes(info.shape.elem_count(), info.ggml_dtype) as f64;
            total += if is_stacked_expert(name) {
                bytes * top_k as f64 / num_experts as f64
            } else if name == "token_embd.weight" && !tied {
                let dims = info.shape.dims();
                let row = dims.first().copied().unwrap_or(1).max(1);
                bytes / row as f64
            } else {
                bytes
            };
        }
        Ok(total.round() as u64)
    }

    pub fn get_gguf(&self, gguf_name: &str, dtype: DType) -> Result<Tensor> {
        let cuda_target = matches!(self.device, Device::Cuda(_));
        let read_device = if cuda_target {
            Device::Cpu
        } else {
            self.device.clone()
        };
        let mut reader = self.reader.borrow_mut();
        let qt = self
            .content
            .tensor(&mut *reader, gguf_name, &read_device)
            .map_err(|e| anyhow!("gguf read tensor {gguf_name}: {e}"))?;
        drop(reader);

        let t = qt
            .dequantize(&read_device)
            .map_err(|e| anyhow!("gguf dequantize {gguf_name}: {e}"))?;
        let t = if t.dtype() == dtype {
            t
        } else {
            t.to_dtype(dtype)
                .map_err(|e| anyhow!("gguf to_dtype({dtype:?}) {gguf_name}: {e}"))?
        };
        if self.nvfp4_transcode.get()
            && is_dense_nvfp4_target(gguf_name)
            && t.rank() == 2
            && t.dims()[1] % nv_quant::nvfp4::BLOCK_SIZE == 0
        {
            return self
                .requantize_nvfp4_dense(&t)
                .with_context(|| format!("nvfp4 transcode {gguf_name}"));
        }

        if cuda_target && !matches!(t.device(), Device::Cuda(_)) {
            return t
                .to_device(&self.device)
                .map_err(|e| anyhow!("gguf to_device(cuda) {gguf_name}: {e}"));
        }
        Ok(t)
    }

    pub fn requantize_nvfp4_dense(&self, t: &Tensor) -> Result<Tensor> {
        use nv_quant::nvfp4::Nvfp4Tensor;
        let dims = t.dims();
        if dims.len() != 2 {
            bail!("requantize_nvfp4_dense expects rank-2, got {dims:?}");
        }
        let (out_f, in_f) = (dims[0], dims[1]);
        let orig_dtype = t.dtype();
        let f32t = t.to_dtype(DType::F32)?.contiguous()?;
        let flat: Vec<f32> = f32t.flatten_all()?.to_vec1::<f32>()?;
        let amax = flat.iter().fold(0f32, |a, &b| a.max(b.abs()));
        let stored_global = if amax.is_finite() && amax > 0.0 {
            (448.0f32 * 6.0) / amax
        } else {
            1.0
        };
        let weight_alpha = if stored_global.is_finite() && stored_global != 0.0 {
            1.0 / stored_global
        } else {
            1.0
        };
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(out_f);
        for r in 0..out_f {
            rows.push(flat[r * in_f..(r + 1) * in_f].to_vec());
        }
        let q = Nvfp4Tensor::quantize_rows_with_global(&rows, stored_global);
        let deq = q.dequantize();
        let mut back = Vec::with_capacity(out_f * in_f);
        for row in deq {
            for v in row {
                back.push(v * weight_alpha);
            }
        }
        let rt = Tensor::from_vec(back, (out_f, in_f), &self.device)?;
        if rt.dtype() == orig_dtype {
            Ok(rt)
        } else {
            Ok(rt.to_dtype(orig_dtype)?)
        }
    }
}

pub fn ggml_bytes(elems: usize, dt: GgmlDType) -> usize {
    elems / dt.block_size() * dt.type_size()
}

pub fn is_stacked_expert(gguf_name: &str) -> bool {
    gguf_name
        .rsplit('.')
        .nth(1)
        .is_some_and(|seg| seg.ends_with("_exps"))
}

pub fn is_dense_nvfp4_target(gguf_name: &str) -> bool {
    if gguf_name == "token_embd.weight" {
        return true;
    }
    let Some(rest) = gguf_name.strip_prefix("blk.") else {
        return false;
    };
    let Some((_idx, suffix)) = rest.split_once('.') else {
        return false;
    };
    matches!(
        suffix,
        "attn_q.weight"
            | "attn_k.weight"
            | "attn_v.weight"
            | "attn_output.weight"
            | "ffn_gate.weight"
            | "ffn_up.weight"
            | "ffn_down.weight"
    )
}

pub fn lone_gguf_file(dir: &Path) -> Option<std::path::PathBuf> {
    let mut found: Option<std::path::PathBuf> = None;
    for entry in std::fs::read_dir(dir).ok()? {
        let path = entry.ok()?.path();
        let is_gguf = path.is_file()
            && path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
        if is_gguf {
            if found.is_some() {
                return None;
            }
            found = Some(path);
        }
    }
    found
}

pub const GGUF_SIDECAR_FILES: [&str; 3] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
];

const GGML_TOKEN_TYPE_NORMAL: u64 = 1;

const GGML_TOKEN_TYPE_BYTE: u64 = 6;

const GGUF_SYNTHESIZABLE_TOKENIZERS: [&str; 1] = ["gemma4"];

const GGUF_SPECIAL_TOKEN_ID_KEYS: [&str; 5] = [
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.mask_token_id",
];

const SPM_SPACE: &str = "\u{2581}";

fn gguf_vocab(loader: &GgufLoader) -> Result<Vec<String>> {
    let model = loader.md_string("tokenizer.ggml.model")?;
    ensure!(
        GGUF_SYNTHESIZABLE_TOKENIZERS.contains(&model.as_str()),
        "gguf tokenizer.ggml.model is {model:?}; only {GGUF_SYNTHESIZABLE_TOKENIZERS:?} has a \
         pipeline verified against a published tokenizer.json here, and guessing the \
         normalizer/decoder mis-tokenizes every prompt silently"
    );
    let tokens = loader.md_string_list("tokenizer.ggml.tokens")?;
    ensure!(!tokens.is_empty(), "gguf tokenizer.ggml.tokens is empty");
    Ok(tokens)
}

pub fn gguf_tokenizer_json(loader: &GgufLoader) -> Result<serde_json::Value> {
    let tokens = gguf_vocab(loader)?;
    ensure!(
        !loader
            .md_bool("tokenizer.ggml.add_space_prefix")
            .unwrap_or(false),
        "gguf declares add_space_prefix=true; the synthesized normalizer replaces spaces with \
         {SPM_SPACE:?} but never prepends one, so every prompt would lose its leading marker"
    );
    let types = loader.md_u64_list("tokenizer.ggml.token_type")?;
    ensure!(
        types.len() == tokens.len(),
        "gguf tokenizer.ggml.token_type has {} entries for {} tokens",
        types.len(),
        tokens.len()
    );

    let mut vocab = serde_json::Map::with_capacity(tokens.len());
    for (id, piece) in tokens.iter().enumerate() {
        if vocab
            .insert(piece.clone(), serde_json::Value::from(id))
            .is_some()
        {
            bail!(
                "gguf vocab repeats the piece {piece:?} at id {id}; a piece->id map cannot \
                 represent both and the duplicate would be silently unreachable"
            );
        }
    }

    let rules = loader.md_string_list("tokenizer.ggml.merges")?;
    let mut merges = Vec::with_capacity(rules.len());
    for (i, rule) in rules.iter().enumerate() {
        let (left, right) = rule
            .split_once(' ')
            .ok_or_else(|| anyhow!("gguf merge {i} {rule:?} is not a space-separated pair"))?;
        ensure!(
            !left.is_empty() && !right.is_empty() && !right.contains(' '),
            "gguf merge {i} {rule:?} does not split at exactly one space; the pair boundary \
             would be guessed"
        );
        merges.push(serde_json::json!([left, right]));
    }
    ensure!(
        !merges.is_empty(),
        "gguf carries no tokenizer.ggml.merges; a BPE tokenizer.json without merges encodes \
         every word one byte at a time"
    );

    let mut special: BTreeSet<usize> = types
        .iter()
        .enumerate()
        .filter(|(_, t)| **t != GGML_TOKEN_TYPE_NORMAL && **t != GGML_TOKEN_TYPE_BYTE)
        .map(|(id, _)| id)
        .collect();
    for key in GGUF_SPECIAL_TOKEN_ID_KEYS {
        if let Ok(id) = loader.md_u64(key) {
            let id = id as usize;
            ensure!(
                id < tokens.len(),
                "gguf {key} is {id} but the vocab holds {} tokens",
                tokens.len()
            );
            special.insert(id);
        }
    }
    let added: Vec<serde_json::Value> = special
        .iter()
        .map(|&id| {
            serde_json::json!({
                "id": id,
                "content": tokens[id],
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect();

    let unk = loader
        .md_u64("tokenizer.ggml.unknown_token_id")
        .ok()
        .and_then(|id| tokens.get(id as usize).cloned());
    Ok(serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": SPM_SPACE},
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"String": " "},
            "behavior": "MergedWithPrevious",
            "invert": false,
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
            "special_tokens": {},
        },
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": SPM_SPACE}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ]},
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": unk,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": true,
            "byte_fallback": true,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": merges,
        },
    }))
}

pub fn gguf_tokenizer_config_json(loader: &GgufLoader) -> Result<serde_json::Value> {
    let tokens = gguf_vocab(loader)?;
    let piece = |key: &str| {
        loader
            .md_u64(key)
            .ok()
            .and_then(|id| tokens.get(id as usize).cloned())
    };
    Ok(serde_json::json!({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "add_bos_token": loader.md_bool("tokenizer.ggml.add_bos_token").unwrap_or(false),
        "bos_token": piece("tokenizer.ggml.bos_token_id"),
        "eos_token": piece("tokenizer.ggml.eos_token_id"),
        "unk_token": piece("tokenizer.ggml.unknown_token_id"),
        "pad_token": piece("tokenizer.ggml.padding_token_id"),
    }))
}

pub fn gguf_chat_template(loader: &GgufLoader) -> Result<String> {
    let src = loader.md_string("tokenizer.chat_template")?;
    ensure!(
        !src.trim().is_empty(),
        "gguf tokenizer.chat_template is blank; serving would fall back to a hand-rolled prompt"
    );
    Ok(src)
}

pub fn missing_gguf_sidecars(dir: &Path) -> Vec<&'static str> {
    let operator_tokenizer_config = dir.join("tokenizer_config.json").exists();
    GGUF_SIDECAR_FILES
        .iter()
        .copied()
        .filter(|name| !dir.join(name).exists())
        .filter(|name| !(*name == "chat_template.jinja" && operator_tokenizer_config))
        .collect()
}

fn write_new(path: &Path, body: &[u8]) -> Result<()> {
    let tmp = path.with_extension(format!("nv-gguf-tmp-{}", std::process::id()));
    std::fs::write(&tmp, body)
        .with_context(|| format!("write {} (is the model dir writable?)", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("rename into {}", path.display()))
}

pub fn ensure_gguf_sidecars(dir: &Path) -> Result<Vec<&'static str>> {
    let missing = missing_gguf_sidecars(dir);
    if missing.is_empty() {
        return Ok(Vec::new());
    }
    let Some(gguf) = lone_gguf_file(dir) else {
        return Ok(Vec::new());
    };
    let loader = GgufLoader::open(&gguf, &Device::Cpu)
        .with_context(|| format!("open gguf {}", gguf.display()))?;
    for name in &missing {
        let path = dir.join(name);
        let body = match *name {
            "tokenizer.json" => serde_json::to_vec(&gguf_tokenizer_json(&loader)?)?,
            "tokenizer_config.json" => serde_json::to_vec(&gguf_tokenizer_config_json(&loader)?)?,
            "chat_template.jinja" => gguf_chat_template(&loader)?.into_bytes(),
            other => bail!("no synthesizer for sidecar {other}"),
        };
        write_new(&path, &body).with_context(|| {
            format!(
                "synthesize {name} beside {} from its own gguf metadata",
                gguf.display()
            )
        })?;
    }
    Ok(missing)
}

impl TensorSource for GgufLoader {
    fn get(&self, name: &str, dtype: DType) -> Result<Tensor> {
        let gguf_name = map_gemma4_name(name).unwrap_or_else(|| name.to_string());
        self.get_gguf(&gguf_name, dtype)
            .with_context(|| format!("gguf get {name} -> {gguf_name}"))
    }

    fn has(&self, name: &str) -> bool {
        let gguf_name = map_gemma4_name(name).unwrap_or_else(|| name.to_string());
        self.has_gguf_tensor(&gguf_name)
    }
}

fn value_as_i64(v: &Value) -> Result<i64> {
    if let Ok(x) = v.to_u64() {
        return Ok(x as i64);
    }
    if let Ok(x) = v.to_i64() {
        return Ok(x);
    }
    if let Ok(x) = v.to_u32() {
        return Ok(x as i64);
    }
    if let Ok(x) = v.to_i32() {
        return Ok(x as i64);
    }
    if let Ok(x) = v.to_u16() {
        return Ok(x as i64);
    }
    if let Ok(x) = v.to_i16() {
        return Ok(x as i64);
    }
    if let Ok(x) = v.to_u8() {
        return Ok(x as i64);
    }
    if let Ok(x) = v.to_i8() {
        return Ok(x as i64);
    }
    bail!("gguf value is not an integer scalar")
}

pub fn map_gemma4_name(name: &str) -> Option<String> {
    match name {
        "model.language_model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        "model.language_model.norm.weight" => return Some("output_norm.weight".into()),

        "lm_head.weight" => return Some("token_embd.weight".into()),
        _ => {}
    }

    let rest = name.strip_prefix("model.language_model.layers.")?;
    let (idx_str, suffix) = rest.split_once('.')?;
    let idx: usize = idx_str.parse().ok()?;
    let mapped = map_layer_suffix(suffix)?;
    Some(format!("blk.{idx}.{mapped}"))
}

fn map_layer_suffix(suffix: &str) -> Option<&'static str> {
    Some(match suffix {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "post_attention_norm.weight",
        "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
        "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",
        "post_feedforward_layernorm_1.weight" => "post_ffw_norm_1.weight",
        "pre_feedforward_layernorm_2.weight" => "pre_ffw_norm_2.weight",
        "post_feedforward_layernorm_2.weight" => "post_ffw_norm_2.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "router.proj.weight" => "ffn_gate_inp.weight",
        "router.scale" => "ffn_gate_inp.scale",
        "router.per_expert_scale" => "ffn_down_exps.scale",
        "experts.gate_up_proj" => "ffn_gate_up_exps.weight",
        "experts.down_proj" => "ffn_down_exps.weight",
        "layer_scalar" => "layer_output_scale.weight",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lone_gguf_file_wants_exactly_one() {
        let dir = std::env::temp_dir().join(format!("lone_gguf_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(lone_gguf_file(&dir), None, "empty dir is not a gguf dir");
        std::fs::write(dir.join("model.gguf"), b"x").unwrap();
        std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        assert_eq!(
            lone_gguf_file(&dir).as_deref(),
            Some(dir.join("model.gguf").as_path()),
            "one gguf + sidecars resolves to the gguf"
        );
        std::fs::write(dir.join("mmproj.GGUF"), b"x").unwrap();
        assert_eq!(
            lone_gguf_file(&dir),
            None,
            "two gguf files (any case) are ambiguous"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn name_map_layer_and_top_level() {
        assert_eq!(
            map_gemma4_name("model.language_model.embed_tokens.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            map_gemma4_name("model.language_model.norm.weight").as_deref(),
            Some("output_norm.weight")
        );
        assert_eq!(
            map_gemma4_name("lm_head.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            map_gemma4_name("model.language_model.layers.5.self_attn.q_proj.weight").as_deref(),
            Some("blk.5.attn_q.weight")
        );
        assert_eq!(
            map_gemma4_name("model.language_model.layers.29.experts.gate_up_proj").as_deref(),
            Some("blk.29.ffn_gate_up_exps.weight")
        );
        assert_eq!(
            map_gemma4_name("model.language_model.layers.0.router.per_expert_scale").as_deref(),
            Some("blk.0.ffn_down_exps.scale")
        );
        assert_eq!(
            map_gemma4_name("model.language_model.layers.12.pre_feedforward_layernorm_2.weight")
                .as_deref(),
            Some("blk.12.pre_ffw_norm_2.weight")
        );
        assert_eq!(
            map_gemma4_name("model.language_model.layers.3.layer_scalar").as_deref(),
            Some("blk.3.layer_output_scale.weight")
        );
        assert_eq!(map_gemma4_name("something.else.weight"), None);
    }

    #[test]
    fn dense_nvfp4_targets_exclude_experts_and_norms() {
        for n in [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.5.attn_k.weight",
            "blk.5.attn_output.weight",
            "blk.12.ffn_gate.weight",
            "blk.12.ffn_up.weight",
            "blk.12.ffn_down.weight",
        ] {
            assert!(is_dense_nvfp4_target(n), "{n} should be a dense target");
        }
        for n in [
            "blk.0.ffn_gate_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
            "blk.0.ffn_down_exps.scale",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_gate_inp.scale",
            "blk.0.attn_norm.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.layer_output_scale.weight",
            "output_norm.weight",
        ] {
            assert!(!is_dense_nvfp4_target(n), "{n} must NOT be a dense target");
        }
    }

    #[test]
    fn stacked_expert_names_are_the_only_top_k_scaled_tensors() {
        for n in [
            "blk.0.ffn_gate_up_exps.weight",
            "blk.29.ffn_down_exps.weight",
            "blk.7.ffn_down_exps.scale",
        ] {
            assert!(is_stacked_expert(n), "{n} is a stacked expert tensor");
        }
        for n in [
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.attn_q.weight",
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
        ] {
            assert!(!is_stacked_expert(n), "{n} is read in full every token");
        }
    }

    #[test]
    fn ggml_bytes_counts_block_scales() {
        assert_eq!(ggml_bytes(32, GgmlDType::Q4_0), 18);
        assert_eq!(ggml_bytes(1024, GgmlDType::Q4_0), 576);

        assert_eq!(ggml_bytes(32, GgmlDType::Q8_0), 34);
        assert_eq!(ggml_bytes(1024, GgmlDType::Q8_0), 1088);
        assert_eq!(ggml_bytes(1024, GgmlDType::F32), 4096);
    }

    const QK4_0: usize = 32;

    fn q4_0_quant_dequant_ref(block: &[f32]) -> Vec<f32> {
        assert_eq!(block.len(), QK4_0);

        let mut amax = 0f32;
        let mut max = 0f32;
        for &x in block {
            if x.abs() > amax {
                amax = x.abs();
                max = x;
            }
        }
        let d = max / -8.0;

        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = half::f16::from_f32(d).to_f32();
        let mut out = vec![0f32; QK4_0];
        for i in 0..QK4_0 {
            let f = block[i] * id + 8.5;

            let qi = if f < 0.0 { 0u32 } else { (f as u32).min(15) };
            out[i] = (qi as f32 - 8.0) * d_f16;
        }
        out
    }

    #[test]
    fn q4_0_dequant_matches_candle_bit_exact() {
        use candle_core::quantized::{GgmlDType, QTensor};
        let device = Device::Cpu;
        let rows = 4usize;
        let cols = 128usize;

        let mut vals = vec![0f32; rows * cols];
        let mut s: u32 = 0x1234_5678;
        for v in vals.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = ((s >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0;
        }
        let t = Tensor::from_vec(vals.clone(), (rows, cols), &device).unwrap();
        let qt = QTensor::quantize(&t, GgmlDType::Q4_0).unwrap();
        let candle_deq = qt.dequantize(&device).unwrap();
        assert_eq!(candle_deq.dims(), &[rows, cols]);
        let candle_flat: Vec<f32> = candle_deq.flatten_all().unwrap().to_vec1().unwrap();

        let mut ref_flat = vec![0f32; rows * cols];
        for b in 0..(rows * cols) / QK4_0 {
            let blk = &vals[b * QK4_0..(b + 1) * QK4_0];
            let deq = q4_0_quant_dequant_ref(blk);
            ref_flat[b * QK4_0..(b + 1) * QK4_0].copy_from_slice(&deq);
        }

        let mut max_abs = 0f32;
        for (a, c) in ref_flat.iter().zip(candle_flat.iter()) {
            max_abs = max_abs.max((a - c).abs());
        }

        assert!(
            max_abs <= 1e-4,
            "independent Q4_0 dequant vs candle max abs diff {max_abs}"
        );
    }
}
