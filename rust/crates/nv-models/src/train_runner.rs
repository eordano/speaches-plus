use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use nv_layers::linear::{Linear, LoraDeltaHook};
use nv_train::{
    lora_delta, max_abs_diff, save_peft_raw, LoraConfig, LoraTrainable, RawLora, TrainingLoraHook,
};
use nv_weights::{TensorSource, WeightLoader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::dense_train::DenseTrainModel;
use crate::gemma4::{Gemma4Attention, Gemma4Config, Gemma4Mlp};
use crate::gemma4_moe::{Gemma4Moe, Gemma4MoeConfig};

pub enum BaseModel {
    Moe(Gemma4Moe),
    Dense(DenseTrainModel),
}

impl BaseModel {
    pub fn vocab(&self) -> usize {
        match self {
            BaseModel::Moe(m) => m.config().base.vocab_size,
            BaseModel::Dense(d) => d.vocab_size(),
        }
    }
    pub fn hidden(&self) -> usize {
        match self {
            BaseModel::Moe(m) => m.config().base.hidden_size,
            BaseModel::Dense(d) => d.config().hidden_size,
        }
    }
    pub fn device(&self) -> Device {
        match self {
            BaseModel::Moe(m) => m.device().clone(),
            BaseModel::Dense(d) => d.device().clone(),
        }
    }

    fn attn_mlp_layers(&self) -> Vec<(&Gemma4Attention, &Gemma4Mlp)> {
        match self {
            BaseModel::Moe(m) => m.layers().iter().map(|l| (&l.self_attn, &l.mlp)).collect(),
            BaseModel::Dense(d) => d.layers().iter().map(|l| (&l.self_attn, &l.mlp)).collect(),
        }
    }

    pub fn forward_logits(&self, tokens: &Tensor, positions: &Tensor) -> Result<Tensor> {
        match self {
            BaseModel::Moe(m) => {
                let seq = tokens.dims().get(1).copied().unwrap_or(0);
                let mut cache = m.new_kv_cache(seq.max(2))?;
                m.forward_with_cache(tokens, positions, &mut cache)
            }
            BaseModel::Dense(d) => d.forward_logits(tokens, positions),
        }
    }

    pub fn is_dense(&self) -> bool {
        matches!(self, BaseModel::Dense(_))
    }

    pub fn layer_counts(&self) -> (usize, usize) {
        match self {
            BaseModel::Moe(m) => {
                let n = m.layers().len();
                (n, n)
            }
            BaseModel::Dense(d) => (d.layers().len(), d.full_num_layers()),
        }
    }
}

struct Nvfp4DequantSource<'a> {
    inner: &'a WeightLoader,
    device: Device,
}

impl<'a> Nvfp4DequantSource<'a> {
    fn dense_nvfp4_module(&self, name: &str) -> Option<String> {
        let module = name.strip_suffix(".weight")?;
        let has_scale = self.inner.has(&format!("{module}.weight_scale"))
            && self.inner.has(&format!("{module}.weight_scale_2"));
        if has_scale {
            Some(module.to_string())
        } else {
            None
        }
    }
}

fn scalar_f32(inner: &WeightLoader, name: &str) -> Result<f32> {
    let v: Vec<f32> = inner.get(name, DType::F32)?.flatten_all()?.to_vec1()?;
    Ok(*v.first().unwrap_or(&1.0))
}

impl<'a> TensorSource for Nvfp4DequantSource<'a> {
    fn has(&self, name: &str) -> bool {
        self.inner.has(name)
    }

    fn get(&self, name: &str, dtype: DType) -> Result<Tensor> {
        if let Some(module) = self.dense_nvfp4_module(name) {
            let shape = self
                .inner
                .shape_of(name)
                .with_context(|| format!("missing shape for packed nvfp4 {name}"))?;
            if shape.len() != 2 {
                bail!("packed nvfp4 {name}: expected rank-2 [out, in/2], got {shape:?}");
            }
            let out_f = shape[0];
            let in_f = shape[1] * 2;
            let packed = self.inner.raw_bytes(name)?.to_vec();
            let scales = self
                .inner
                .raw_bytes(&format!("{module}.weight_scale"))?
                .to_vec();
            let weight_mult = scalar_f32(self.inner, &format!("{module}.weight_scale_2"))?;
            let vals = nv_quant::nvfp4::dequantize_packed_linear(
                &packed,
                &scales,
                out_f,
                in_f,
                weight_mult,
            );

            let t = Tensor::from_vec(vals, (out_f, in_f), &Device::Cpu)?.to_dtype(dtype)?;
            return Ok(t.to_device(&self.device)?);
        }
        Ok(self.inner.get(name, dtype)?.to_device(&self.device)?)
    }
}

fn is_packed_nvfp4(weights: &WeightLoader) -> bool {
    weights
        .names()
        .iter()
        .any(|n| n.ends_with(".weight_scale_2"))
}

fn open_base_weights(path: &Path, device: &Device) -> Result<WeightLoader> {
    let st = path.join("model.safetensors");
    if st.is_file() {
        WeightLoader::open_file(&st, device)
            .with_context(|| format!("open weights {}", st.display()))
    } else {
        WeightLoader::open_dir(path, device)
            .with_context(|| format!("open sharded weights {}", path.display()))
    }
}

pub fn base_is_packed_nvfp4(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    open_base_weights(path, &Device::Cpu)
        .map(|w| is_packed_nvfp4(&w))
        .unwrap_or(false)
}

#[derive(Clone, Debug)]
pub struct TrainArgs {
    pub base: PathBuf,
    pub data: PathBuf,
    pub out: PathBuf,
    pub rank: usize,
    pub alpha: f64,
    pub targets: Vec<String>,
    pub steps: usize,
    pub lr: f64,
    pub seed: u64,
}

impl TrainArgs {
    pub fn default_targets() -> Vec<String> {
        ["q", "k", "v", "o", "gate", "up", "down"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct TrainSummary {
    pub losses: Vec<f32>,
    pub serving_equiv_maxabs: f32,
    pub trainable_vars: usize,
    pub num_examples: usize,
    pub base_dtype: String,
    pub nvfp4_base: bool,
    pub dense_base: bool,
    pub layers_built: usize,
    pub config_layers: usize,
    pub deterministic: bool,
    pub device: String,
    pub checkpointed: bool,
    pub lmhead_chunk: usize,
    pub modules: Vec<String>,
    pub adapter_path: PathBuf,
    pub config_path: PathBuf,
}

pub fn select_device() -> Result<Device> {
    match std::env::var("NV_TRAIN_DEVICE").ok().as_deref() {
        Some("cuda") | Some("gpu") => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).map_err(|e| anyhow::anyhow!("open CUDA device 0: {e}"))
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("NV_TRAIN_DEVICE=cuda requires building nvk-train with --features cuda")
            }
        }
        _ => Ok(Device::Cpu),
    }
}

fn checkpointing_enabled() -> bool {
    std::env::var("NV_TRAIN_CKPT").ok().as_deref() == Some("1")
}

fn grad_store_from(
    vars: &[Var],
    grads: &[Option<Tensor>],
) -> Result<candle_core::backprop::GradStore> {
    let mut gs = vars[0].as_tensor().sum_all()?.backward()?;
    for (v, g) in vars.iter().zip(grads.iter()) {
        if let Some(g) = g {
            gs.insert(v.as_tensor(), g.clone());
        }
    }
    Ok(gs)
}

pub fn load_base(path: &Path, device: &Device) -> Result<(BaseModel, DType)> {
    if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
        let model = Gemma4Moe::from_gguf(path, device, DType::BF16)
            .with_context(|| format!("load gguf base {}", path.display()))?;
        return Ok((BaseModel::Moe(model), DType::BF16));
    }
    if path.is_dir() {
        let cfg_path = path.join("config.json");

        let base_cfg = Gemma4Config::from_hf_json_file(&cfg_path)
            .with_context(|| format!("read config {}", cfg_path.display()))?;

        let load_device = if device.is_cuda() {
            Device::Cpu
        } else {
            device.clone()
        };
        let weights = open_base_weights(path, &load_device)?;
        let packed = is_packed_nvfp4(&weights);

        if base_cfg.enable_moe_block {
            let cfg = Gemma4MoeConfig::from_hf_json_file(&cfg_path)
                .with_context(|| format!("read moe config {}", cfg_path.display()))?;
            let model = if packed {
                let src = Nvfp4DequantSource {
                    inner: &weights,
                    device: device.clone(),
                };
                Gemma4Moe::from_loader_dtype(cfg, &src, device, DType::F32)
                    .with_context(|| format!("load packed-nvfp4 moe base {}", path.display()))?
            } else {
                Gemma4Moe::from_loader_dtype(cfg, &weights, device, DType::F32)?
            };
            return Ok((BaseModel::Moe(model), DType::F32));
        }

        let base_dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let plain_weights;
        let weights = if !packed && device.is_cuda() {
            plain_weights = open_base_weights(path, device)?;
            &plain_weights
        } else {
            &weights
        };
        let model = if packed {
            let src = Nvfp4DequantSource {
                inner: &weights,
                device: device.clone(),
            };
            DenseTrainModel::from_loader_dtype(base_cfg, &src, device, base_dtype)
                .with_context(|| format!("load packed-nvfp4 dense base {}", path.display()))?
        } else {
            DenseTrainModel::from_loader_dtype(base_cfg, weights, device, base_dtype)?
        };
        return Ok((BaseModel::Dense(model), base_dtype));
    }
    bail!(
        "--base {} is neither a .gguf file nor a directory with config.json + model.safetensors",
        path.display()
    )
}

struct FusedTarget<'a> {
    lin: &'a Linear,
    in_features: usize,

    components: Vec<(String, usize, usize, bool)>,

    seed: u64,
}

fn want(targets: &[String], comp: &str) -> bool {
    targets.iter().any(|t| t == comp)
}

fn build_targets<'a>(
    layers: &[(&'a Gemma4Attention, &'a Gemma4Mlp)],
    targets: &[String],
    seed0: u64,
) -> Result<Vec<FusedTarget<'a>>> {
    let mut out = Vec::new();
    let mut si = seed0;
    for (li, (attn, mlp)) in layers.iter().enumerate() {
        let p = format!("model.language_model.layers.{li}");
        let attn = *attn;
        let q_dim = attn.q_dim;
        let kv_dim = attn.kv_dim;
        let has_v = attn.has_v;

        if want(targets, "q") || want(targets, "k") || want(targets, "v") {
            let mut comps = vec![
                (
                    format!("{p}.self_attn.q_proj"),
                    0,
                    q_dim,
                    want(targets, "q"),
                ),
                (
                    format!("{p}.self_attn.k_proj"),
                    q_dim,
                    kv_dim,
                    want(targets, "k"),
                ),
            ];
            if has_v {
                comps.push((
                    format!("{p}.self_attn.v_proj"),
                    q_dim + kv_dim,
                    kv_dim,
                    want(targets, "v"),
                ));
            }
            out.push(FusedTarget {
                lin: &attn.qkv_proj,
                in_features: attn.qkv_proj.in_features(),
                components: comps,
                seed: si,
            });
            si += 1;
        }

        if want(targets, "o") {
            out.push(FusedTarget {
                lin: &attn.o_proj,
                in_features: attn.o_proj.in_features(),
                components: vec![(
                    format!("{p}.self_attn.o_proj"),
                    0,
                    attn.o_proj.out_features(),
                    true,
                )],
                seed: si,
            });
            si += 1;
        }

        if want(targets, "gate") || want(targets, "up") {
            let gu = &mlp.gate_up_proj;
            let out_f = gu.out_features();
            let inter = out_f / 2;
            out.push(FusedTarget {
                lin: gu,
                in_features: gu.in_features(),
                components: vec![
                    (
                        format!("{p}.mlp.gate_proj"),
                        0,
                        inter,
                        want(targets, "gate"),
                    ),
                    (
                        format!("{p}.mlp.up_proj"),
                        inter,
                        inter,
                        want(targets, "up"),
                    ),
                ],
                seed: si,
            });
            si += 1;
        }

        if want(targets, "down") {
            let dp = &mlp.down_proj;
            out.push(FusedTarget {
                lin: dp,
                in_features: dp.in_features(),
                components: vec![(format!("{p}.mlp.down_proj"), 0, dp.out_features(), true)],
                seed: si,
            });
            si += 1;
        }
    }
    if out.is_empty() {
        bail!("no trainable targets selected (--target was {:?})", targets);
    }
    Ok(out)
}

fn byte_tokenize(s: &str, vocab: usize) -> Vec<u32> {
    s.bytes().map(|b| (b as u32) % vocab as u32).collect()
}

fn load_dataset(path: &Path, vocab: usize, max_seq: usize) -> Result<Vec<Vec<u32>>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read dataset {}", path.display()))?;
    let mut examples = Vec::new();
    for (lineno, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("{}:{}: invalid json", path.display(), lineno + 1))?;
        let mut ids: Vec<u32> = if let Some(arr) = v.get("ids").and_then(|x| x.as_array()) {
            arr.iter()
                .filter_map(|x| x.as_u64())
                .map(|x| (x as u32) % vocab as u32)
                .collect()
        } else if let Some(t) = v.get("text").and_then(|x| x.as_str()) {
            byte_tokenize(t, vocab)
        } else if v.get("prompt").is_some() || v.get("completion").is_some() {
            let prompt = v.get("prompt").and_then(|x| x.as_str()).unwrap_or("");
            let completion = v.get("completion").and_then(|x| x.as_str()).unwrap_or("");
            byte_tokenize(&format!("{prompt}{completion}"), vocab)
        } else {
            bail!(
                "{}:{}: line has none of the supported keys (ids | text | prompt/completion)",
                path.display(),
                lineno + 1
            );
        };
        if ids.len() > max_seq {
            ids.truncate(max_seq);
        }
        if ids.len() >= 2 {
            examples.push(ids);
        }
    }
    if examples.is_empty() {
        bail!(
            "dataset {} produced no usable (>=2 token) examples",
            path.display()
        );
    }
    Ok(examples)
}

fn accumulate(total: Option<Tensor>, loss: Tensor) -> Result<Option<Tensor>> {
    Ok(Some(match total {
        Some(t) => (t + loss)?,
        None => loss,
    }))
}

fn solo_example_loss(model: &BaseModel, ids: &[u32], dev: &Device) -> Result<Tensor> {
    let seq = ids.len();
    let tokens = Tensor::from_vec(ids.to_vec(), (1usize, seq), dev)?;
    let positions = Tensor::from_vec((0..seq as u32).collect::<Vec<_>>(), seq, dev)?;
    let logits = model.forward_logits(&tokens, &positions)?;
    let logits2 = logits.squeeze(0)?;
    let inp = logits2.narrow(0, 0, seq - 1)?;
    let tgt = Tensor::from_vec(ids[1..].to_vec(), seq - 1, dev)?;
    Ok(candle_nn::loss::cross_entropy(&inp, &tgt)?)
}

fn same_length_bucket_loss_is_count_times_the_stacked_ce_mean(
    model: &BaseModel,
    rows: &[&Vec<u32>],
    seq: usize,
    dev: &Device,
) -> Result<Tensor> {
    let count = rows.len();
    let mut flat: Vec<u32> = Vec::with_capacity(count * seq);
    for ids in rows {
        flat.extend_from_slice(ids);
    }
    let tokens = Tensor::from_vec(flat, (count, seq), dev)?;
    let positions = Tensor::from_vec((0..seq as u32).collect::<Vec<_>>(), seq, dev)?;
    let logits = model.forward_logits(&tokens, &positions)?;
    let vocab = logits.dims()[2];
    let inp = logits
        .narrow(1, 0, seq - 1)?
        .contiguous()?
        .reshape((count * (seq - 1), vocab))?;
    let mut tgt_flat: Vec<u32> = Vec::with_capacity(count * (seq - 1));
    for ids in rows {
        tgt_flat.extend_from_slice(&ids[1..]);
    }
    let tgt = Tensor::from_vec(tgt_flat, count * (seq - 1), dev)?;
    let stacked_mean = candle_nn::loss::cross_entropy(&inp, &tgt)?;
    Ok(stacked_mean.affine(count as f64, 0.0)?)
}

fn batch_loss(model: &BaseModel, batch: &[Vec<u32>]) -> Result<Tensor> {
    let dev = model.device();
    let mut total: Option<Tensor> = None;
    if !model.is_dense() {
        for ids in batch {
            total = accumulate(total, solo_example_loss(model, ids, &dev)?)?;
        }
        return total.context("empty batch");
    }
    let mut buckets: std::collections::BTreeMap<usize, Vec<&Vec<u32>>> =
        std::collections::BTreeMap::new();
    for ids in batch {
        buckets.entry(ids.len()).or_default().push(ids);
    }
    let solo_forced = std::env::var("NV_TRAIN_SOLO").ok().as_deref() == Some("1");
    for (seq, rows) in &buckets {
        if solo_forced {
            for ids in rows {
                total = accumulate(total, solo_example_loss(model, ids, &dev)?)?;
            }
            continue;
        }
        let loss =
            same_length_bucket_loss_is_count_times_the_stacked_ce_mean(model, rows, *seq, &dev)?;
        total = accumulate(total, loss)?;
    }
    total.context("empty batch")
}

struct Lcg(u64);
impl Lcg {
    fn f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as u32 as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn serving_equivalence(
    targets: &[FusedTarget],
    loras: &[Arc<LoraTrainable>],
    device: &Device,
) -> Result<f32> {
    let mut worst = 0f32;
    for (t, lora) in targets.iter().zip(loras.iter()) {
        let mut rng = Lcg(0xC0FFEE ^ t.seed);
        let n = 4usize;
        let data: Vec<f32> = (0..n * t.in_features).map(|_| rng.f32()).collect();
        let x = Tensor::from_vec(data, (n, t.in_features), device)?;
        let full = lora.delta_forward(&x, false)?;
        let mut pieces = Vec::new();
        for (_name, off, rows, _emit) in &t.components {
            let b_slice = lora.b_tensor().narrow(0, *off, *rows)?;
            let piece = lora_delta(lora.a_tensor(), &b_slice, lora.scaling(), &x, None)?;
            pieces.push(piece);
        }
        let cat = Tensor::cat(&pieces, 1)?;
        worst = worst.max(max_abs_diff(&full, &cat)?);
    }
    Ok(worst)
}

pub fn run(args: &TrainArgs) -> Result<TrainSummary> {
    let device = select_device()?;
    let device_label = if device.is_cuda() { "cuda" } else { "cpu" };
    let (model, base_dtype) = load_base(&args.base, &device)?;
    let vocab = model.vocab();
    if model.is_dense() {
        let (built, full) = model.layer_counts();
        eprintln!(
            "DENSE_BASE 1  layers_built {built}  config_layers {full}  hidden {}  vocab {vocab}  \
             device {device_label}  base_dtype {base_dtype:?}",
            model.hidden()
        );
    }

    let attn_mlp = model.attn_mlp_layers();
    let targets = build_targets(&attn_mlp, &args.targets, args.seed)?;

    let cfg = LoraConfig {
        r: args.rank,
        alpha: args.alpha,
        dropout: 0.0,
    };
    let mut loras: Vec<Arc<LoraTrainable>> = Vec::new();
    let mut vars: Vec<Var> = Vec::new();
    for t in &targets {
        let base = t
            .lin
            .weight()
            .with_context(|| {
                "target Linear has no materialised weight (packed nvfp4/fp8 base needs host \
                 dequant before LoRA training -- see T3 log)"
            })?
            .clone();
        let lora = Arc::new(LoraTrainable::new(&base, cfg, t.seed, &device)?);
        vars.extend(lora.trainable_vars());
        let hook: Arc<dyn LoraDeltaHook> = Arc::new(TrainingLoraHook::new(lora.clone(), true));
        t.lin.attach_lora(hook)?;
        loras.push(lora);
    }

    let dataset = load_dataset(&args.data, vocab, 32)?;

    let checkpointed = checkpointing_enabled() && model.is_dense();
    let lmhead_chunk = crate::dense_train::lmhead_chunk_from_env();
    if checkpointing_enabled() && !model.is_dense() {
        bail!("NV_TRAIN_CKPT=1 is only supported for a dense base");
    }

    let mut losses = Vec::with_capacity(args.steps);
    if args.steps > 0 {
        let mut opt = AdamW::new(
            vars.clone(),
            ParamsAdamW {
                lr: args.lr,
                weight_decay: 0.0,
                ..Default::default()
            },
        )?;
        for step in 0..args.steps {
            let lv = if checkpointed {
                let dense = match &model {
                    BaseModel::Dense(d) => d,
                    _ => unreachable!("checkpointed guarded to dense"),
                };
                let sg = dense.train_step_checkpointed(&dataset, &vars, lmhead_chunk)?;
                let gs = grad_store_from(&vars, &sg.grads)?;
                opt.step(&gs)?;
                sg.loss
            } else {
                let loss = batch_loss(&model, &dataset)?;
                let lv = loss.to_scalar::<f32>()?;
                opt.backward_step(&loss)?;
                lv
            };
            losses.push(lv);
            if step < 3 || step % 10 == 0 || step == args.steps - 1 {
                eprintln!("step {step:4}  loss = {lv:.6e}");
            }
        }
    }

    let serving_equiv_maxabs = serving_equivalence(&targets, &loras, &device)?;

    let mut entries: Vec<RawLora> = Vec::new();
    for (t, lora) in targets.iter().zip(loras.iter()) {
        for (name, off, rows, emit) in &t.components {
            if !*emit {
                continue;
            }
            let a = lora.a_tensor().clone();
            let b = lora.b_tensor().narrow(0, *off, *rows)?;
            entries.push(RawLora {
                module: name.clone(),
                a,
                b,
            });
        }
    }
    let base_name = args.base.to_string_lossy().to_string();
    save_peft_raw(&args.out, &entries, args.rank, args.alpha, 0.0, &base_name)?;

    let (layers_built, config_layers) = model.layer_counts();
    let dense_base = model.is_dense();
    let modules: Vec<String> = entries.iter().map(|e| e.module.clone()).collect();
    Ok(TrainSummary {
        losses,
        serving_equiv_maxabs,
        trainable_vars: vars.len(),
        num_examples: dataset.len(),
        base_dtype: format!("{base_dtype:?}"),
        nvfp4_base: base_is_packed_nvfp4(&args.base),
        dense_base,
        layers_built,
        config_layers,
        deterministic: std::env::var("NV_DETERMINISTIC").ok().as_deref() == Some("1"),
        device: device_label.to_string(),
        checkpointed,
        lmhead_chunk,
        modules,
        adapter_path: args.out.join("adapter_model.safetensors"),
        config_path: args.out.join("adapter_config.json"),
    })
}
