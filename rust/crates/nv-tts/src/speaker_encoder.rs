use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::ops::{sigmoid, softmax};
use candle_nn::{Conv1d, Conv1dConfig};
use nv_weights::WeightLoader;

use crate::spk_mel::{log_mel_24k, SPK_MEL_N_MELS, SPK_MEL_SAMPLE_RATE};

pub const CHECKPOINT_SPEAKER_PREFIX: &str = "speaker_encoder.";

pub fn qwen3_tts_base_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("NV_TTS_BASE_TALKER_DIR") {
        let p = PathBuf::from(d);
        if p.join("model.safetensors").is_file() {
            return Some(p);
        }
    }
    let home = std::env::var_os("HOME")?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots");
    for e in std::fs::read_dir(&base).ok()?.flatten() {
        let p = e.path();
        if p.is_dir() && p.join("model.safetensors").is_file() {
            return Some(p);
        }
    }
    None
}

#[derive(Clone, Debug)]
pub struct SpeakerEncoderConfig {
    pub mel_dim: usize,
    pub enc_dim: usize,
    pub enc_channels: Vec<usize>,
    pub enc_kernel_sizes: Vec<usize>,
    pub enc_dilations: Vec<usize>,
    pub enc_attention_channels: usize,
    pub enc_res2net_scale: usize,
    pub enc_se_channels: usize,
    pub sample_rate: usize,
}

impl Default for SpeakerEncoderConfig {
    fn default() -> Self {
        Self {
            mel_dim: 128,
            enc_dim: 1024,
            enc_channels: vec![512, 512, 512, 512, 1536],
            enc_kernel_sizes: vec![5, 3, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 4, 1],
            enc_attention_channels: 128,
            enc_res2net_scale: 8,
            enc_se_channels: 128,
            sample_rate: 24_000,
        }
    }
}

impl SpeakerEncoderConfig {
    pub fn from_hf_config_file(path: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&raw)?;
        let mut cfg = Self::default();
        let Some(sec) = v.get("speaker_encoder_config") else {
            return Ok(cfg);
        };
        if let Some(x) = sec.get("mel_dim").and_then(|x| x.as_u64()) {
            cfg.mel_dim = x as usize;
        }
        if let Some(x) = sec.get("enc_dim").and_then(|x| x.as_u64()) {
            cfg.enc_dim = x as usize;
        }
        if let Some(x) = sec.get("enc_attention_channels").and_then(|x| x.as_u64()) {
            cfg.enc_attention_channels = x as usize;
        }
        if let Some(x) = sec.get("enc_res2net_scale").and_then(|x| x.as_u64()) {
            cfg.enc_res2net_scale = x as usize;
        }
        if let Some(x) = sec.get("enc_se_channels").and_then(|x| x.as_u64()) {
            cfg.enc_se_channels = x as usize;
        }
        if let Some(x) = sec.get("sample_rate").and_then(|x| x.as_u64()) {
            cfg.sample_rate = x as usize;
        }
        for (key, slot) in [
            ("enc_channels", &mut cfg.enc_channels),
            ("enc_kernel_sizes", &mut cfg.enc_kernel_sizes),
            ("enc_dilations", &mut cfg.enc_dilations),
        ] {
            if let Some(arr) = sec.get(key).and_then(|x| x.as_array()) {
                let parsed: Option<Vec<usize>> =
                    arr.iter().map(|x| x.as_u64().map(|u| u as usize)).collect();
                if let Some(p) = parsed {
                    *slot = p;
                }
            }
        }
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.enc_channels.len() != self.enc_kernel_sizes.len()
            || self.enc_channels.len() != self.enc_dilations.len()
        {
            anyhow::bail!(
                "enc_channels ({}), enc_kernel_sizes ({}), enc_dilations ({}) must have equal length",
                self.enc_channels.len(),
                self.enc_kernel_sizes.len(),
                self.enc_dilations.len()
            );
        }
        if self.enc_channels.len() < 3 {
            anyhow::bail!(
                "enc_channels needs >= 3 entries, got {}",
                self.enc_channels.len()
            );
        }
        for (i, &c) in self.enc_channels[..self.enc_channels.len() - 1]
            .iter()
            .enumerate()
        {
            if c % self.enc_res2net_scale != 0 {
                anyhow::bail!(
                    "enc_channels[{i}] = {c} not divisible by res2net scale {}",
                    self.enc_res2net_scale
                );
            }
        }
        Ok(())
    }
}

pub type TensorGetter<'a> = &'a dyn Fn(&str) -> Result<Tensor>;

fn reflect_pad_last(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let rank = x.rank();
    let t = x.dim(rank - 1)?;
    if t <= left.max(right) {
        anyhow::bail!("reflect pad ({left},{right}) requires time dim > pad, got {t}");
    }
    let mut idx: Vec<u32> = Vec::with_capacity(t + left + right);
    for i in (1..=left).rev() {
        idx.push(i as u32);
    }
    idx.extend(0..t as u32);
    for i in 1..=right {
        idx.push((t - 1 - i) as u32);
    }
    let idx = Tensor::from_vec(idx, left + t + right, x.device())?;
    Ok(x.index_select(&idx, rank - 1)?)
}

struct SameConv1d {
    conv: Conv1d,
    pad_left: usize,
    pad_right: usize,
}

impl SameConv1d {
    fn load(
        get: TensorGetter,
        name: &str,
        in_c: usize,
        out_c: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Self> {
        if kernel.is_multiple_of(2) {
            anyhow::bail!("Conv kernel must be odd for SAME padding, got {kernel}");
        }
        let w = get(&format!("{name}.weight"))?;
        if w.dims() != [out_c, in_c, kernel] {
            anyhow::bail!(
                "{name}.weight: expected [{out_c}, {in_c}, {kernel}], got {:?}",
                w.dims()
            );
        }
        let b = get(&format!("{name}.bias"))?;
        if b.dims() != [out_c] {
            anyhow::bail!("{name}.bias: expected [{out_c}], got {:?}", b.dims());
        }
        let cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let total = dilation * (kernel - 1);
        Ok(Self {
            conv: Conv1d::new(w, Some(b), cfg),
            pad_left: total / 2,
            pad_right: total - total / 2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = reflect_pad_last(x, self.pad_left, self.pad_right)?;
        Ok(self.conv.forward(&x)?)
    }
}

struct TdnnBlock {
    conv: SameConv1d,
}

impl TdnnBlock {
    fn load(
        get: TensorGetter,
        name: &str,
        in_c: usize,
        out_c: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Self> {
        Ok(Self {
            conv: SameConv1d::load(get, &format!("{name}.conv"), in_c, out_c, kernel, dilation)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.conv.forward(x)?.relu()?)
    }
}

struct Res2NetBlock {
    blocks: Vec<TdnnBlock>,
    scale: usize,
}

impl Res2NetBlock {
    fn load(
        get: TensorGetter,
        name: &str,
        in_c: usize,
        out_c: usize,
        scale: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Self> {
        let in_chunk = in_c / scale;
        let hidden_chunk = out_c / scale;
        let mut blocks = Vec::with_capacity(scale - 1);
        for i in 0..scale - 1 {
            blocks.push(TdnnBlock::load(
                get,
                &format!("{name}.blocks.{i}"),
                in_chunk,
                hidden_chunk,
                kernel,
                dilation,
            )?);
        }
        Ok(Self { blocks, scale })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let c = x.dim(1)?;
        let chunk = c / self.scale;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(self.scale);
        let mut processed: Option<Tensor> = None;
        for i in 0..self.scale {
            let part = x.narrow(1, i * chunk, chunk)?;
            let cur = match i {
                0 => part,
                1 => self.blocks[0].forward(&part)?,
                _ => {
                    let prev = processed.as_ref().expect("processed set after i==1");
                    self.blocks[i - 1].forward(&(part.add(prev))?)?
                }
            };
            if i >= 1 {
                processed = Some(cur.clone());
            }
            outputs.push(cur);
        }
        let refs: Vec<&Tensor> = outputs.iter().collect();
        Ok(Tensor::cat(&refs, 1)?)
    }
}

struct SeBlock {
    conv1: SameConv1d,
    conv2: SameConv1d,
}

impl SeBlock {
    fn load(get: TensorGetter, name: &str, in_c: usize, se_c: usize, out_c: usize) -> Result<Self> {
        Ok(Self {
            conv1: SameConv1d::load(get, &format!("{name}.conv1"), in_c, se_c, 1, 1)?,
            conv2: SameConv1d::load(get, &format!("{name}.conv2"), se_c, out_c, 1, 1)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = x.mean_keepdim(2)?;
        let gate = self.conv1.forward(&gate)?.relu()?;
        let gate = sigmoid(&self.conv2.forward(&gate)?)?;
        Ok(x.broadcast_mul(&gate)?)
    }
}

struct SeRes2NetBlock {
    tdnn1: TdnnBlock,
    res2net: Res2NetBlock,
    tdnn2: TdnnBlock,
    se: SeBlock,
}

impl SeRes2NetBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        get: TensorGetter,
        name: &str,
        in_c: usize,
        out_c: usize,
        scale: usize,
        se_c: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Self> {
        Ok(Self {
            tdnn1: TdnnBlock::load(get, &format!("{name}.tdnn1"), in_c, out_c, 1, 1)?,
            res2net: Res2NetBlock::load(
                get,
                &format!("{name}.res2net_block"),
                out_c,
                out_c,
                scale,
                kernel,
                dilation,
            )?,
            tdnn2: TdnnBlock::load(get, &format!("{name}.tdnn2"), out_c, out_c, 1, 1)?,
            se: SeBlock::load(get, &format!("{name}.se_block"), out_c, se_c, out_c)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let h = self.tdnn1.forward(x)?;
        let h = self.res2net.forward(&h)?;
        let h = self.tdnn2.forward(&h)?;
        let h = self.se.forward(&h)?;
        Ok(h.add(residual)?)
    }
}

struct AttentiveStatsPooling {
    tdnn: TdnnBlock,
    conv: SameConv1d,
    eps: f64,
}

impl AttentiveStatsPooling {
    fn load(get: TensorGetter, name: &str, channels: usize, attn_c: usize) -> Result<Self> {
        Ok(Self {
            tdnn: TdnnBlock::load(get, &format!("{name}.tdnn"), channels * 3, attn_c, 1, 1)?,
            conv: SameConv1d::load(get, &format!("{name}.conv"), attn_c, channels, 1, 1)?,
            eps: 1e-12,
        })
    }

    fn weighted_stats(&self, x: &Tensor, w: &Tensor) -> Result<(Tensor, Tensor)> {
        let mean = x.broadcast_mul(w)?.sum(2)?;
        let diff = x.broadcast_sub(&mean.unsqueeze(2)?)?;
        let var = diff.sqr()?.broadcast_mul(w)?.sum(2)?;
        let eps = Tensor::new(self.eps as f32, x.device())?.to_dtype(var.dtype())?;
        let std = var.broadcast_maximum(&eps)?.sqrt()?;
        Ok((mean, std))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dim(2)?;
        let uniform = Tensor::full(1.0f32 / t as f32, (1usize, 1usize, t), x.device())?
            .to_dtype(x.dtype())?;
        let (mean_u, std_u) = self.weighted_stats(x, &uniform)?;
        let mean_rep = mean_u.unsqueeze(2)?.broadcast_as(x.dims())?;
        let std_rep = std_u.unsqueeze(2)?.broadcast_as(x.dims())?;
        let attn_in = Tensor::cat(&[x, &mean_rep, &std_rep], 1)?;
        let a = self.tdnn.forward(&attn_in)?;
        let a = a.tanh()?;
        let a = self.conv.forward(&a)?;
        let a = softmax(&a, 2)?;
        let (mean, std) = self.weighted_stats(x, &a)?;
        Ok(Tensor::cat(&[&mean, &std], 1)?.unsqueeze(2)?)
    }
}

pub struct SpeakerEncoder {
    cfg: SpeakerEncoderConfig,
    device: Device,
    first: TdnnBlock,
    se_blocks: Vec<SeRes2NetBlock>,
    mfa: TdnnBlock,
    asp: AttentiveStatsPooling,
    fc: SameConv1d,
}

impl SpeakerEncoder {
    pub fn from_getter(
        cfg: SpeakerEncoderConfig,
        get: TensorGetter,
        device: &Device,
    ) -> Result<Self> {
        cfg.validate()?;
        let ch = &cfg.enc_channels;
        let ks = &cfg.enc_kernel_sizes;
        let dil = &cfg.enc_dilations;
        let first = TdnnBlock::load(get, "blocks.0", cfg.mel_dim, ch[0], ks[0], dil[0])?;
        let mut se_blocks = Vec::with_capacity(ch.len() - 2);
        for i in 1..ch.len() - 1 {
            se_blocks.push(SeRes2NetBlock::load(
                get,
                &format!("blocks.{i}"),
                ch[i - 1],
                ch[i],
                cfg.enc_res2net_scale,
                cfg.enc_se_channels,
                ks[i],
                dil[i],
            )?);
        }
        let cat_c: usize = ch[1..ch.len() - 1].iter().sum();
        let last = *ch.last().expect("validated non-empty");
        if cat_c != last {
            anyhow::bail!(
                "multi-layer aggregation width {cat_c} != enc_channels[-1] {last}; \
                 checkpoint layout not supported"
            );
        }
        let mfa = TdnnBlock::load(get, "mfa", last, last, ks[ch.len() - 1], dil[ch.len() - 1])?;
        let asp = AttentiveStatsPooling::load(get, "asp", last, cfg.enc_attention_channels)?;
        let fc = SameConv1d::load(get, "fc", last * 2, cfg.enc_dim, 1, 1)?;
        Ok(Self {
            cfg,
            device: device.clone(),
            first,
            se_blocks,
            mfa,
            asp,
            fc,
        })
    }

    pub fn from_weights(
        cfg: SpeakerEncoderConfig,
        weights: &WeightLoader,
        prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        let get = move |name: &str| -> Result<Tensor> {
            weights
                .get(&format!("{prefix}{name}"), DType::F32)
                .with_context(|| format!("load {prefix}{name}"))
        };
        Self::from_getter(cfg, &get, device)
    }

    pub fn from_tensor_map(
        cfg: SpeakerEncoderConfig,
        map: &HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<Self> {
        let get = move |name: &str| -> Result<Tensor> {
            map.get(name)
                .cloned()
                .ok_or_else(|| anyhow!("missing tensor {name}"))
        };
        Self::from_getter(cfg, &get, device)
    }

    pub fn checkpoint_has_speaker_encoder(dir: &Path) -> bool {
        let shard = dir.join("model.safetensors");
        let Ok(weights) = WeightLoader::open_file(&shard, &Device::Cpu) else {
            return false;
        };
        weights.has("speaker_encoder.blocks.0.conv.weight")
    }

    pub fn from_qwen3_checkpoint(dir: &Path, device: &Device) -> Result<Self> {
        let shard = dir.join("model.safetensors");
        let weights = WeightLoader::open_file(&shard, device)
            .with_context(|| format!("open {}", shard.display()))?;
        if !weights.has("speaker_encoder.blocks.0.conv.weight") {
            anyhow::bail!(
                "no speaker_encoder.* tensors in {}; this checkpoint (CustomVoice/VoiceDesign) \
                 has no speaker encoder -- voice-profile enrollment needs a Base checkpoint",
                shard.display()
            );
        }
        let cfg =
            SpeakerEncoderConfig::from_hf_config_file(&dir.join("config.json")).unwrap_or_default();
        Self::from_weights(cfg, &weights, CHECKPOINT_SPEAKER_PREFIX, device)
    }

    pub fn config(&self) -> &SpeakerEncoderConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let dims = mel.dims();
        if dims.len() != 3 {
            return Err(anyhow!(
                "SpeakerEncoder.forward: expected rank-3 mel (B, mel, T), got {:?}",
                dims
            ));
        }
        if dims[1] != self.cfg.mel_dim {
            return Err(anyhow!(
                "SpeakerEncoder.forward: mel_dim mismatch (got {}, configured {})",
                dims[1],
                self.cfg.mel_dim
            ));
        }
        if dims[2] == 0 {
            return Err(anyhow!("SpeakerEncoder.forward: empty time dimension"));
        }
        let x = mel.to_dtype(DType::F32)?.to_device(&self.device)?;
        let mut h = self.first.forward(&x)?;
        let mut taps: Vec<Tensor> = Vec::with_capacity(self.se_blocks.len());
        for block in &self.se_blocks {
            h = block.forward(&h)?;
            taps.push(h.clone());
        }
        let refs: Vec<&Tensor> = taps.iter().collect();
        let agg = Tensor::cat(&refs, 1)?;
        let agg = self.mfa.forward(&agg)?;
        let pooled = self.asp.forward(&agg)?;
        let out = self.fc.forward(&pooled)?;
        Ok(out.squeeze(2)?)
    }

    pub fn encode(&self, mel: &Tensor) -> Result<Vec<f32>> {
        if mel.dim(0)? != 1 {
            anyhow::bail!("encode() requires batch == 1, got {}", mel.dim(0)?);
        }
        let out = self.forward(mel)?;
        Ok(out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
    }

    pub fn encode_pcm_24k(&self, samples_24k: &[f32]) -> Result<Vec<f32>> {
        if self.cfg.mel_dim != SPK_MEL_N_MELS || self.cfg.sample_rate != SPK_MEL_SAMPLE_RATE {
            anyhow::bail!(
                "encode_pcm_24k supports mel_dim {} at {} Hz, configured {} at {}",
                SPK_MEL_N_MELS,
                SPK_MEL_SAMPLE_RATE,
                self.cfg.mel_dim,
                self.cfg.sample_rate
            );
        }
        let (mel_flat, frames) = log_mel_24k(samples_24k)?;
        let mel = Tensor::from_vec(mel_flat, (1usize, SPK_MEL_N_MELS, frames), &self.device)?;
        self.encode(&mel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn tiny_cfg() -> SpeakerEncoderConfig {
        SpeakerEncoderConfig {
            mel_dim: 8,
            enc_dim: 12,
            enc_channels: vec![16, 16, 16, 32],
            enc_kernel_sizes: vec![5, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 1],
            enc_attention_channels: 6,
            enc_res2net_scale: 4,
            enc_se_channels: 5,
            sample_rate: 24_000,
        }
    }

    fn det_tensor(shape: &[usize], seed: u32, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let x = ((i as u64 + 1) * (seed as u64 * 2654435761 + 1)) % 10_007;
                (x as f32 / 10_007.0 - 0.5) * 0.2
            })
            .collect();
        Tensor::from_vec(data, shape, dev).unwrap()
    }

    fn tiny_weight_map(cfg: &SpeakerEncoderConfig, dev: &Device) -> HashMap<String, Tensor> {
        let mut m = HashMap::new();
        let mut seed = 1u32;
        let mut put = |m: &mut HashMap<String, Tensor>, name: &str, shape: Vec<usize>| {
            let s = seed;
            seed += 1;
            m.insert(format!("{name}.weight"), det_tensor(&shape, s, dev));
            m.insert(
                format!("{name}.bias"),
                det_tensor(&[shape[0]], s + 1000, dev),
            );
        };
        let ch = &cfg.enc_channels;
        put(
            &mut m,
            "blocks.0.conv",
            vec![ch[0], cfg.mel_dim, cfg.enc_kernel_sizes[0]],
        );
        for i in 1..ch.len() - 1 {
            let base = format!("blocks.{i}");
            put(
                &mut m,
                &format!("{base}.tdnn1.conv"),
                vec![ch[i], ch[i - 1], 1],
            );
            let chunk = ch[i] / cfg.enc_res2net_scale;
            for j in 0..cfg.enc_res2net_scale - 1 {
                put(
                    &mut m,
                    &format!("{base}.res2net_block.blocks.{j}.conv"),
                    vec![chunk, chunk, cfg.enc_kernel_sizes[i]],
                );
            }
            put(&mut m, &format!("{base}.tdnn2.conv"), vec![ch[i], ch[i], 1]);
            put(
                &mut m,
                &format!("{base}.se_block.conv1"),
                vec![cfg.enc_se_channels, ch[i], 1],
            );
            put(
                &mut m,
                &format!("{base}.se_block.conv2"),
                vec![ch[i], cfg.enc_se_channels, 1],
            );
        }
        let last = *ch.last().unwrap();
        put(&mut m, "mfa.conv", vec![last, last, 1]);
        put(
            &mut m,
            "asp.tdnn.conv",
            vec![cfg.enc_attention_channels, last * 3, 1],
        );
        put(
            &mut m,
            "asp.conv",
            vec![last, cfg.enc_attention_channels, 1],
        );
        put(&mut m, "fc", vec![cfg.enc_dim, last * 2, 1]);
        m
    }

    fn tiny_encoder() -> SpeakerEncoder {
        let cfg = tiny_cfg();
        let map = tiny_weight_map(&cfg, &cpu());
        SpeakerEncoder::from_tensor_map(cfg, &map, &cpu()).expect("build tiny encoder")
    }

    fn det_mel(cfg: &SpeakerEncoderConfig, t: usize, phase: f32) -> Tensor {
        let mut data = vec![0.0f32; cfg.mel_dim * t];
        for c in 0..cfg.mel_dim {
            for i in 0..t {
                data[c * t + i] =
                    ((c as f32 + phase).sin() * (i as f32 * 0.13 + phase).cos()) * 0.5;
            }
        }
        Tensor::from_vec(data, (1usize, cfg.mel_dim, t), &cpu()).unwrap()
    }

    #[test]
    fn reflect_pad_matches_manual_reference() {
        let x = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0],
            (1usize, 1usize, 5usize),
            &cpu(),
        )
        .unwrap();
        let padded = reflect_pad_last(&x, 2, 2).unwrap();
        let v = padded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn reflect_pad_rejects_pad_ge_len() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0], (1usize, 1usize, 2usize), &cpu()).unwrap();
        assert!(reflect_pad_last(&x, 2, 0).is_err());
    }

    #[test]
    fn missing_weight_key_is_a_hard_error() {
        let cfg = tiny_cfg();
        let mut map = tiny_weight_map(&cfg, &cpu());
        map.remove("asp.conv.weight");
        let err = SpeakerEncoder::from_tensor_map(cfg, &map, &cpu())
            .err()
            .expect("must fail");
        assert!(format!("{err:#}").contains("asp.conv.weight"), "{err:#}");
    }

    #[test]
    fn wrong_weight_shape_is_a_hard_error() {
        let cfg = tiny_cfg();
        let mut map = tiny_weight_map(&cfg, &cpu());
        map.insert(
            "blocks.0.conv.weight".to_string(),
            det_tensor(&[3, 3, 3], 7, &cpu()),
        );
        assert!(SpeakerEncoder::from_tensor_map(cfg, &map, &cpu()).is_err());
    }

    #[test]
    fn forward_shape_deterministic_and_input_dependent() {
        let enc = tiny_encoder();
        let mel_a = det_mel(enc.config(), 40, 0.0);
        let mel_b = det_mel(enc.config(), 40, 1.7);
        let a1 = enc.encode(&mel_a).unwrap();
        let a2 = enc.encode(&mel_a).unwrap();
        let b = enc.encode(&mel_b).unwrap();
        assert_eq!(a1.len(), enc.config().enc_dim);
        assert_eq!(a1, a2, "same input must be deterministic");
        assert!(
            a1.iter().any(|x| x.abs() > 1e-6),
            "embedding must be non-zero"
        );
        let diff: f32 = a1.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
        assert!(
            diff > 1e-4,
            "different mels must produce different embeddings"
        );
    }

    #[test]
    fn forward_rejects_wrong_mel_bins_and_empty_time() {
        let enc = tiny_encoder();
        let wrong = Tensor::zeros((1usize, 9usize, 20usize), DType::F32, &cpu()).unwrap();
        assert!(enc.forward(&wrong).is_err());
        let empty = Tensor::zeros((1usize, 8usize, 0usize), DType::F32, &cpu()).unwrap();
        assert!(enc.forward(&empty).is_err());
    }

    #[test]
    fn encode_rejects_batch_gt_1() {
        let enc = tiny_encoder();
        let mel = Tensor::zeros((2usize, 8usize, 16usize), DType::F32, &cpu()).unwrap();
        assert!(enc.encode(&mel).is_err());
    }

    #[test]
    fn time_shift_invariance_is_approximate_not_degenerate() {
        let enc = tiny_encoder();
        let mel = det_mel(enc.config(), 64, 0.3);
        let long = enc.encode(&mel).unwrap();
        let mel_half = det_mel(enc.config(), 32, 0.3);
        let short = enc.encode(&mel_half).unwrap();
        let dot: f32 = long.iter().zip(&short).map(|(a, b)| a * b).sum();
        let na: f32 = long.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = short.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(na > 0.0 && nb > 0.0);
        let cos = dot / (na * nb);
        assert!(
            cos > 0.5,
            "same synthetic content at two lengths should stay similar, cos = {cos}"
        );
    }

    #[test]
    fn config_validation_rejects_bad_shapes() {
        let mut cfg = tiny_cfg();
        cfg.enc_kernel_sizes.pop();
        assert!(cfg.validate().is_err());
        let mut cfg = tiny_cfg();
        cfg.enc_channels = vec![15, 15, 15, 30];
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn default_config_matches_qwen3_tts_base() {
        let cfg = SpeakerEncoderConfig::default();
        assert_eq!(cfg.mel_dim, 128);
        assert_eq!(cfg.enc_dim, 1024);
        assert_eq!(cfg.enc_channels, vec![512, 512, 512, 512, 1536]);
        assert_eq!(cfg.enc_kernel_sizes, vec![5, 3, 3, 3, 1]);
        assert_eq!(cfg.enc_dilations, vec![1, 2, 3, 4, 1]);
        assert_eq!(cfg.enc_res2net_scale, 8);
        assert_eq!(cfg.sample_rate, 24_000);
    }

    #[test]
    fn encode_pcm_24k_runs_on_tiny_config_only_with_matching_mel() {
        let enc = tiny_encoder();
        let pcm = vec![0.1f32; 24_000];
        assert!(
            enc.encode_pcm_24k(&pcm).is_err(),
            "tiny config has mel_dim 8, pcm path must refuse"
        );
    }
}
