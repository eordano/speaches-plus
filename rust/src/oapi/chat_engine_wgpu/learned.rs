use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};

use nv_kernels::wgpu_backend::kernels::kv_fp8::decode_e4m3;
use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_assistant_wgpu::{
    AssistantLayerWeights, AssistantWgpuSpec, AssistantWgpuWeights, BackboneKvBinding,
    Gemma4AssistantWgpu, PackedHiddenSrc,
};
use nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu;
use nv_specdecode::gemma4_assistant::{AssistantLayerType, FixedSharedKv, Gemma4AssistantDrafter};

use super::spec::{SpecKnobs, SPEC_ASSISTANT_DIR_ENV, SPEC_DRAFTER_ENV, SPEC_K_MAX};

const EMBED_NAME: &str = "model.language_model.embed_tokens.weight";
const ASSISTANT_REPO_CACHE: &str =
    ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-q4_0-unquantized-assistant/snapshots";
pub const SPEC_ASSISTANT_HOST_ENV: &str = "NV_WGPU_SPEC_ASSISTANT_HOST";

pub fn dequant_kv_rows(
    fp8: &[u32],
    scales: &[f32],
    n_kv: usize,
    head_dim: usize,
    start: usize,
    len: usize,
) -> Result<Tensor> {
    anyhow::ensure!(n_kv > 0 && head_dim > 0 && len > 0, "empty kv view");
    let end = start + len;
    anyhow::ensure!(
        fp8.len() * 4 >= end * n_kv * head_dim,
        "fp8 kv buffer holds {} bytes, need {}",
        fp8.len() * 4,
        end * n_kv * head_dim
    );
    anyhow::ensure!(
        scales.len() >= end * n_kv,
        "kv scales hold {}, need {}",
        scales.len(),
        end * n_kv
    );
    let mut lut = [0f32; 256];
    for (b, slot) in lut.iter_mut().enumerate() {
        *slot = decode_e4m3(b as u8);
    }
    let mut out = vec![0f32; n_kv * len * head_dim];
    for h in 0..n_kv {
        for i in 0..len {
            let slot = start + i;
            let scale = scales[slot * n_kv + h];
            let base = (slot * n_kv + h) * head_dim;
            let obase = (h * len + i) * head_dim;
            for d in 0..head_dim {
                let idx = base + d;
                let byte = (fp8[idx / 4] >> (8 * (idx % 4))) & 0xff;
                out[obase + d] = lut[byte as usize] * scale;
            }
        }
    }
    Ok(Tensor::from_vec(out, (n_kv, len, head_dim), &Device::Cpu)?)
}

fn e4m3_lut() -> [f32; 256] {
    let mut lut = [0f32; 256];
    for (b, slot) in lut.iter_mut().enumerate() {
        *slot = decode_e4m3(b as u8);
    }
    lut
}

pub struct KvMirror {
    n_kv: usize,
    hd: usize,
    len: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl KvMirror {
    pub fn new(n_kv: usize, hd: usize) -> Self {
        Self {
            n_kv,
            hd,
            len: 0,
            k: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.len = 0;
        self.k.clear();
        self.v.clear();
    }

    pub fn append(
        &mut self,
        k_fp8: &[u32],
        v_fp8: &[u32],
        k_scales: &[f32],
        v_scales: &[f32],
        added: usize,
    ) -> Result<()> {
        if added == 0 {
            return Ok(());
        }
        let vals = added * self.n_kv * self.hd;
        anyhow::ensure!(
            k_fp8.len() * 4 >= vals && v_fp8.len() * 4 >= vals,
            "kv mirror append: fp8 buffers hold {}/{} bytes, need {vals}",
            k_fp8.len() * 4,
            v_fp8.len() * 4
        );
        anyhow::ensure!(
            k_scales.len() >= added * self.n_kv && v_scales.len() >= added * self.n_kv,
            "kv mirror append: scales hold {}/{}, need {}",
            k_scales.len(),
            v_scales.len(),
            added * self.n_kv
        );
        let lut = e4m3_lut();
        let unpack = |dst: &mut Vec<f32>, fp8: &[u32], scales: &[f32]| {
            dst.reserve(vals);
            for idx in 0..vals {
                let byte = (fp8[idx / 4] >> (8 * (idx % 4))) & 0xff;
                dst.push(lut[byte as usize] * scales[idx / self.hd]);
            }
        };
        unpack(&mut self.k, k_fp8, k_scales);
        unpack(&mut self.v, v_fp8, v_scales);
        self.len += added;
        Ok(())
    }

    pub fn tensors(&self, start: usize, len: usize) -> Result<(Tensor, Tensor)> {
        anyhow::ensure!(len > 0, "empty kv mirror view");
        anyhow::ensure!(
            start + len <= self.len,
            "kv mirror view {start}+{len} outside mirrored {}",
            self.len
        );
        let (n_kv, hd) = (self.n_kv, self.hd);
        let pick = |src: &[f32]| -> Result<Tensor> {
            let mut out = vec![0f32; n_kv * len * hd];
            for h in 0..n_kv {
                for i in 0..len {
                    let sbase = ((start + i) * n_kv + h) * hd;
                    let obase = (h * len + i) * hd;
                    out[obase..obase + hd].copy_from_slice(&src[sbase..sbase + hd]);
                }
            }
            Ok(Tensor::from_vec(out, (n_kv, len, hd), &Device::Cpu)?)
        };
        Ok((pick(&self.k)?, pick(&self.v)?))
    }
}

fn default_assistant_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let root = Path::new(&home).join(ASSISTANT_REPO_CACHE);
    for entry in std::fs::read_dir(&root).ok()? {
        let p = entry.ok()?.path();
        if p.is_dir() && p.join("model.safetensors").is_file() {
            return Some(p);
        }
    }
    None
}

fn kv_writing_layers(cfg: &Gemma4Config) -> Result<(usize, usize)> {
    let mut sliding = None;
    let mut full = None;
    for (i, kind) in cfg.layer_types.iter().enumerate() {
        if cfg.kv_source_layer(i).is_none() {
            match kind {
                LayerType::SlidingAttention => sliding = Some(i),
                LayerType::FullAttention => full = Some(i),
            }
        }
    }
    Ok((
        sliding.context("no kv-writing sliding layer")?,
        full.context("no kv-writing full-attention layer")?,
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HiddenLoc {
    Decode,
    Verify(usize),
}

pub struct AssistantSpecDrafter {
    drafter: Gemma4AssistantDrafter,
    weights: nv_weights::WeightLoader,
    hidden: usize,
    normalizer: f32,
    sliding_layer: usize,
    full_layer: usize,
    sliding_nkv: usize,
    sliding_hd: usize,
    full_nkv: usize,
    full_hd: usize,
    sliding_mirror: KvMirror,
    full_mirror: KvMirror,
    mirror_epoch: u64,
    assistant_dir: PathBuf,
    gpu: Option<Gemma4AssistantWgpu>,
}

impl AssistantSpecDrafter {
    pub fn load(model_dir: &Path, assistant_dir: &Path, cfg: &Gemma4Config) -> Result<Self> {
        let drafter = Gemma4AssistantDrafter::try_load(assistant_dir, &Device::Cpu)
            .with_context(|| format!("load assistant drafter from {}", assistant_dir.display()))?;
        let acfg = drafter.config();
        anyhow::ensure!(
            acfg.backbone_hidden_size == cfg.hidden_size,
            "assistant backbone_hidden_size {} != target hidden_size {}",
            acfg.backbone_hidden_size,
            cfg.hidden_size
        );
        anyhow::ensure!(
            acfg.head_dim == cfg.head_dim_for(LayerType::SlidingAttention)
                && acfg.global_head_dim == cfg.head_dim_for(LayerType::FullAttention),
            "assistant head dims ({}, {}) do not match target ({}, {})",
            acfg.head_dim,
            acfg.global_head_dim,
            cfg.head_dim_for(LayerType::SlidingAttention),
            cfg.head_dim_for(LayerType::FullAttention)
        );
        let (sliding_layer, full_layer) = kv_writing_layers(cfg)?;
        let weights = nv_weights::WeightLoader::open_dir(model_dir, &Device::Cpu)
            .with_context(|| format!("open target weights in {}", model_dir.display()))?;
        anyhow::ensure!(
            weights.st_dtype_of(EMBED_NAME) == Some(nv_weights::StDtype::BF16),
            "assistant drafter needs a bf16 {EMBED_NAME} in the target checkpoint"
        );
        anyhow::ensure!(
            weights.shape_of(EMBED_NAME).as_deref() == Some(&[cfg.vocab_size, cfg.hidden_size][..]),
            "unexpected embed table shape {:?}",
            weights.shape_of(EMBED_NAME)
        );
        let sliding_nkv = cfg.num_kv_heads_for(LayerType::SlidingAttention);
        let sliding_hd = cfg.head_dim_for(LayerType::SlidingAttention);
        let full_nkv = cfg.num_kv_heads_for(LayerType::FullAttention);
        let full_hd = cfg.head_dim_for(LayerType::FullAttention);
        Ok(Self {
            drafter,
            weights,
            hidden: cfg.hidden_size,
            normalizer: (cfg.hidden_size as f32).sqrt(),
            sliding_layer,
            full_layer,
            sliding_nkv,
            sliding_hd,
            full_nkv,
            full_hd,
            sliding_mirror: KvMirror::new(sliding_nkv, sliding_hd),
            full_mirror: KvMirror::new(full_nkv, full_hd),
            mirror_epoch: 0,
            assistant_dir: assistant_dir.to_path_buf(),
            gpu: None,
        })
    }

    pub fn from_env(model_dir: &Path, model: &Gemma4E4bWgpu) -> Option<Self> {
        if !SpecKnobs::from_env().enabled {
            return None;
        }
        let sel = std::env::var(SPEC_DRAFTER_ENV).ok();
        let sel = sel.as_deref().map(str::trim);
        let explicit = sel == Some("assistant");
        if !explicit && sel.is_some() {
            return None;
        }
        let assistant_dir = match std::env::var(SPEC_ASSISTANT_DIR_ENV) {
            Ok(d) => PathBuf::from(d),
            Err(_) => match default_assistant_dir() {
                Some(d) => d,
                None => {
                    if explicit {
                        tracing::warn!(
                            "{SPEC_DRAFTER_ENV}=assistant but no checkpoint in the HF cache and \
                             {SPEC_ASSISTANT_DIR_ENV} unset; falling back to suffix drafting"
                        );
                    } else {
                        tracing::info!(
                            "no assistant drafter checkpoint in the HF cache and \
                             {SPEC_ASSISTANT_DIR_ENV} unset; spec decode drafts from the suffix \
                             automaton"
                        );
                    }
                    return None;
                }
            },
        };
        match Self::load(model_dir, &assistant_dir, model.config()) {
            Ok(mut d) => {
                tracing::info!(
                    assistant_dir = %assistant_dir.display(),
                    sliding_layer = d.sliding_layer,
                    full_layer = d.full_layer,
                    "wgpu spec decode uses the learned assistant drafter"
                );
                if std::env::var(SPEC_ASSISTANT_HOST_ENV).ok().as_deref() != Some("1") {
                    d.attach_gpu(model);
                }
                Some(d)
            }
            Err(e) => {
                tracing::warn!(
                    error = %format!("{e:#}"),
                    "assistant drafter failed to load; falling back to suffix drafting"
                );
                None
            }
        }
    }

    pub fn attach_gpu(&mut self, model: &Gemma4E4bWgpu) {
        match self.build_gpu(model) {
            Ok(mut g) => {
                let (decode_hid, _) = model.decode_hid_gpu();
                let verify_hid = model.verify_hid_gpu().map(|(b, _, _)| b);
                match g.bind_hidden_sources(decode_hid, verify_hid) {
                    Ok(()) => tracing::info!(
                        "assistant drafter forward runs on wgpu (gpu-resident hidden)"
                    ),
                    Err(e) => tracing::warn!(
                        error = %format!("{e:#}"),
                        "assistant drafter hidden stays host-read this session"
                    ),
                }
                self.gpu = Some(g);
            }
            Err(e) => {
                tracing::warn!(
                    error = %format!("{e:#}"),
                    "assistant wgpu forward unavailable; drafting on candle CPU"
                );
            }
        }
    }

    pub fn gpu_active(&self) -> bool {
        self.gpu.is_some()
    }

    pub fn propose_loc(
        &mut self,
        model: &Gemma4E4bWgpu,
        last_token: u32,
        loc: HiddenLoc,
        k: usize,
    ) -> Result<Vec<u32>> {
        let committed = model.current_pos();
        if k == 0 || committed == 0 {
            return Ok(Vec::new());
        }
        let gpu = self
            .gpu
            .as_mut()
            .context("propose_loc without an active gpu drafter")?;
        let src = match loc {
            HiddenLoc::Decode => PackedHiddenSrc::Decode,
            HiddenLoc::Verify(row) => {
                let (_, row_words, rows) = model
                    .verify_hid_gpu()
                    .context("verify hidden buffer missing")?;
                anyhow::ensure!(row < rows, "verify hidden row {row} out of 0..{rows}");
                PackedHiddenSrc::Verify {
                    word_off: row * row_words,
                }
            }
        };
        gpu.propose_packed(last_token, committed, k, src)
    }

    fn build_gpu(&self, model: &Gemma4E4bWgpu) -> Result<Gemma4AssistantWgpu> {
        let acfg = self.drafter.config();
        anyhow::ensure!(
            acfg.use_ordered_embeddings,
            "wgpu drafter forward needs ordered embeddings"
        );
        let (embed_chunks, embed_rows_per_chunk) = model.embed_table_gpu();
        let (sk, sv, sks, svs) = model
            .kv_cache_gpu(self.sliding_layer)
            .context("sliding kv cache missing")?;
        let (fk, fv, fks, fvs) = model
            .kv_cache_gpu(self.full_layer)
            .context("full-attention kv cache missing")?;
        let weights = load_gpu_weights(&self.assistant_dir, acfg)?;
        let spec = AssistantWgpuSpec {
            backbone_hidden: acfg.backbone_hidden_size,
            hidden: acfg.hidden_size,
            intermediate: acfg.intermediate_size,
            n_heads: acfg.num_attention_heads,
            vocab: acfg.vocab_size,
            n_centroids: acfg.num_centroids,
            top_k: acfg.centroid_top_k,
            eps: acfg.rms_norm_eps as f32,
            sliding_window: acfg.sliding_window,
            sliding_theta: acfg.sliding_rope_theta,
            full_theta: acfg.full_rope_theta,
            full_partial: acfg.full_partial_rotary_factor,
            sliding_hd: self.sliding_hd,
            full_hd: self.full_hd,
            sliding_nkv: self.sliding_nkv,
            full_nkv: self.full_nkv,
            layers_sliding: acfg
                .layer_types
                .iter()
                .map(|t| *t == AssistantLayerType::Sliding)
                .collect(),
            eos: acfg.eos_token_ids.clone(),
            embed_normalizer: self.normalizer,
        };
        Gemma4AssistantWgpu::new(
            spec,
            &weights,
            embed_chunks,
            embed_rows_per_chunk,
            BackboneKvBinding {
                k_fp8: sk,
                v_fp8: sv,
                k_scales: sks,
                v_scales: svs,
            },
            BackboneKvBinding {
                k_fp8: fk,
                v_fp8: fv,
                k_scales: fks,
                v_scales: fvs,
            },
            model.max_seq(),
            SPEC_K_MAX,
        )
    }

    pub fn sliding_layer(&self) -> usize {
        self.sliding_layer
    }

    pub fn full_layer(&self) -> usize {
        self.full_layer
    }

    pub fn embed_scaled(&self, token: u32) -> Result<Tensor> {
        let bytes = self.weights.raw_bytes(EMBED_NAME)?;
        let off = token as usize * self.hidden * 2;
        anyhow::ensure!(
            off + self.hidden * 2 <= bytes.len(),
            "token {token} outside the embed table"
        );
        let mut row = Vec::with_capacity(self.hidden);
        for i in 0..self.hidden {
            let bits = u16::from_le_bytes([bytes[off + 2 * i], bytes[off + 2 * i + 1]]);
            row.push(f32::from_bits((bits as u32) << 16) * self.normalizer);
        }
        Ok(Tensor::from_vec(row, self.hidden, &Device::Cpu)?)
    }

    pub fn propose_from_parts(
        &self,
        last_token: u32,
        last_hidden: &[f32],
        committed: usize,
        k: usize,
        kv: &FixedSharedKv,
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(
            last_hidden.len() == self.hidden,
            "last_hidden holds {} values, want {}",
            last_hidden.len(),
            self.hidden
        );
        let hidden = Tensor::from_vec(last_hidden.to_vec(), self.hidden, &Device::Cpu)?;
        let embedder = |tok: u32| self.embed_scaled(tok);
        self.drafter
            .propose(last_token, &hidden, committed, k, &embedder, kv)
    }

    fn sync_mirrors(&mut self, model: &Gemma4E4bWgpu, committed: usize) -> Result<()> {
        let epoch = model.kv_epoch();
        if epoch != self.mirror_epoch
            || committed < self.sliding_mirror.len()
            || committed < self.full_mirror.len()
        {
            self.sliding_mirror.clear();
            self.full_mirror.clear();
            self.mirror_epoch = epoch;
        }
        for (li, mirror) in [
            (self.sliding_layer, &mut self.sliding_mirror),
            (self.full_layer, &mut self.full_mirror),
        ] {
            let start = mirror.len();
            let added = committed - start;
            if added == 0 {
                continue;
            }
            let (kf, vf, ks, vs) = model
                .kv_cache_snapshot_range(li, start, added)?
                .with_context(|| format!("kv cache missing for layer {li}"))?;
            mirror.append(&kf, &vf, &ks, &vs, added)?;
        }
        Ok(())
    }

    pub fn propose(
        &mut self,
        model: &Gemma4E4bWgpu,
        last_token: u32,
        last_hidden: &[f32],
        k: usize,
    ) -> Result<Vec<u32>> {
        let committed = model.current_pos();
        if k == 0 || committed == 0 {
            return Ok(Vec::new());
        }
        if let Some(gpu) = self.gpu.as_mut() {
            match gpu.propose(last_token, last_hidden, committed, k) {
                Ok(toks) => return Ok(toks),
                Err(e) => tracing::warn!(
                    error = %format!("{e:#}"),
                    "assistant wgpu forward failed; host forward this round"
                ),
            }
        }
        self.sync_mirrors(model, committed)?;
        let win = self.drafter.config().sliding_window.max(1);
        let s_start = committed.saturating_sub(win);
        let (sk, sv) = self.sliding_mirror.tensors(s_start, committed - s_start)?;
        let (fk, fv) = self.full_mirror.tensors(0, committed)?;
        let kv = FixedSharedKv {
            sliding: (sk, sv),
            full: (fk, fv),
        };
        self.propose_from_parts(last_token, last_hidden, committed, k, &kv)
    }
}

fn bf16_bits(w: &nv_weights::WeightLoader, name: &str) -> Result<Vec<u16>> {
    if w.st_dtype_of(name) == Some(nv_weights::StDtype::BF16) {
        let raw = w
            .raw_bytes(name)
            .with_context(|| format!("raw_bytes {name}"))?;
        anyhow::ensure!(raw.len().is_multiple_of(2), "{name}: odd byte length");
        let mut out = vec![0u16; raw.len() / 2];
        for (i, o) in out.iter_mut().enumerate() {
            *o = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
        }
        return Ok(out);
    }
    let t = w
        .get(name, candle_core::DType::F32)
        .with_context(|| format!("load {name}"))?;
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(v.into_iter().map(f32_to_bf16_bits).collect())
}

fn f32_to_bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    let round = ((u >> 16) & 1) + 0x7fff;
    ((u.wrapping_add(round)) >> 16) as u16
}

fn load_gpu_weights(
    assistant_dir: &Path,
    acfg: &nv_specdecode::gemma4_assistant::Gemma4AssistantConfig,
) -> Result<AssistantWgpuWeights> {
    let st = assistant_dir.join("model.safetensors");
    let w = nv_weights::WeightLoader::open_file(&st, &Device::Cpu)
        .with_context(|| format!("open {}", st.display()))?;
    let ordering = {
        let t = w.get("masked_embedding.token_ordering", candle_core::DType::I64)?;
        let vals: Vec<i64> = t.flatten_all()?.to_vec1()?;
        anyhow::ensure!(
            vals.len() == acfg.vocab_size,
            "token_ordering holds {}, want {}",
            vals.len(),
            acfg.vocab_size
        );
        let mut out = Vec::with_capacity(vals.len());
        for v in vals {
            anyhow::ensure!(
                v >= 0 && (v as usize) < acfg.vocab_size,
                "token_ordering value {v} out of range"
            );
            out.push(v as u32);
        }
        out
    };
    let mut layers = Vec::with_capacity(acfg.num_hidden_layers);
    for i in 0..acfg.num_hidden_layers {
        let p = format!("model.layers.{i}");
        let scalar = {
            let t = w.get(&format!("{p}.layer_scalar"), candle_core::DType::F32)?;
            let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
            anyhow::ensure!(v.len() == 1, "{p}.layer_scalar holds {} values", v.len());
            v[0]
        };
        layers.push(AssistantLayerWeights {
            q: bf16_bits(&w, &format!("{p}.self_attn.q_proj.weight"))?,
            q_norm: bf16_bits(&w, &format!("{p}.self_attn.q_norm.weight"))?,
            o: bf16_bits(&w, &format!("{p}.self_attn.o_proj.weight"))?,
            gate: bf16_bits(&w, &format!("{p}.mlp.gate_proj.weight"))?,
            up: bf16_bits(&w, &format!("{p}.mlp.up_proj.weight"))?,
            down: bf16_bits(&w, &format!("{p}.mlp.down_proj.weight"))?,
            ln_in: bf16_bits(&w, &format!("{p}.input_layernorm.weight"))?,
            ln_post_attn: bf16_bits(&w, &format!("{p}.post_attention_layernorm.weight"))?,
            ln_pre_ff: bf16_bits(&w, &format!("{p}.pre_feedforward_layernorm.weight"))?,
            ln_post_ff: bf16_bits(&w, &format!("{p}.post_feedforward_layernorm.weight"))?,
            scalar,
        });
    }
    Ok(AssistantWgpuWeights {
        pre: bf16_bits(&w, "pre_projection.weight")?,
        post: bf16_bits(&w, "post_projection.weight")?,
        norm: bf16_bits(&w, "model.norm.weight")?,
        lm_head: bf16_bits(&w, "model.embed_tokens.weight")?,
        centroids: bf16_bits(&w, "masked_embedding.centroids.weight")?,
        ordering,
        layers,
    })
}
