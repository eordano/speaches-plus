use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor, Var};
use nv_layers::linear::LoraDeltaHook;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub mod dots;
pub mod dots_fixture;
#[cfg(feature = "metal")]
pub mod dots_rope;
#[cfg(feature = "metal")]
pub mod dots_w4head;

pub fn w4head_requested_via_env() -> bool {
    std::env::var("NV_DOTS_QLORA_W4HEAD").as_deref() == Ok("1")
}
pub mod qwen38;
pub mod qwen38_fixture;
pub mod qwen38_gdn_chunked;
pub mod qwen38_gdn_head;
pub mod qwen38_ce;
pub mod train_prof;

struct DetRng(u64);

impl DetRng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as f32;
        (bits / (1u64 << 24) as f32).max(f32::MIN_POSITIVE)
    }

    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32();
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LoraConfig {
    pub r: usize,
    pub alpha: f64,
    pub dropout: f32,
}

impl LoraConfig {
    pub fn unit_scaling(r: usize) -> Self {
        Self {
            r,
            alpha: r as f64,
            dropout: 0.0,
        }
    }

    pub fn scaling(&self) -> f64 {
        self.alpha / self.r as f64
    }
}

pub struct LoraTrainable {
    base: Option<Tensor>,
    a: Var,
    b: Var,
    r: usize,
    alpha: f64,
    scaling: f64,
    dropout: f32,
    in_features: usize,
    out_features: usize,
    compute_dtype: DType,
    det_seeded: bool,
}

impl LoraTrainable {
    pub fn new(base: &Tensor, cfg: LoraConfig, seed: u64, device: &Device) -> Result<Self> {
        let dims = base.dims();
        if dims.len() != 2 {
            bail!(
                "LoraTrainable base must be 2-D [out, in], got rank {}",
                dims.len()
            );
        }
        let mut built = Self::from_features(dims[0], dims[1], cfg, seed, device)?;
        built.base = Some(base.detach().to_device(device)?);
        Ok(built)
    }

    pub fn from_features(
        out_features: usize,
        in_features: usize,
        cfg: LoraConfig,
        seed: u64,
        device: &Device,
    ) -> Result<Self> {
        if cfg.r == 0 {
            bail!("LoRA rank r must be positive");
        }

        let mut rng = DetRng::new(seed);
        let std = 1.0f32 / (in_features as f32).sqrt();
        let a_vals: Vec<f32> = (0..cfg.r * in_features)
            .map(|_| rng.next_normal() * std)
            .collect();
        let a_t = Tensor::from_vec(a_vals, (cfg.r, in_features), device)?;
        let a = Var::from_tensor(&a_t)?;

        let b = Var::zeros((out_features, cfg.r), DType::F32, device)?;

        let det_seeded = nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_DETERMINISTIC");

        Ok(Self {
            base: None,
            a,
            b,
            r: cfg.r,
            alpha: cfg.alpha,
            scaling: cfg.scaling(),
            dropout: cfg.dropout,
            in_features,
            out_features,
            compute_dtype: DType::F32,
            det_seeded,
        })
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn rank(&self) -> usize {
        self.r
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn scaling(&self) -> f64 {
        self.scaling
    }

    pub fn trainable_vars(&self) -> Vec<Var> {
        vec![self.a.clone(), self.b.clone()]
    }

    pub fn a_tensor(&self) -> &Tensor {
        self.a.as_tensor()
    }

    pub fn b_tensor(&self) -> &Tensor {
        self.b.as_tensor()
    }

    pub fn base_tensor(&self) -> Option<&Tensor> {
        self.base.as_ref()
    }

    pub fn base_forward(&self, x: &Tensor) -> Result<Tensor> {
        let base = self.base.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "base_forward on a dims-only LoraTrainable: from_features holds no dense base \
                 (the quantized-resident Linear owns the base GEMM); use delta_forward"
            )
        })?;
        let (x2, mut out_dims, n) = self.flatten_input(x)?;
        let base_f = base.to_dtype(self.compute_dtype)?;
        let y2 = x2.matmul(&base_f.t()?.contiguous()?)?;
        out_dims.push(self.out_features);
        let _ = n;
        Ok(y2.reshape(out_dims)?)
    }

    pub fn delta_forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let (x2, mut out_dims, _n) = self.flatten_input(x)?;
        let xd = if train && self.dropout > 0.0 {
            self.apply_dropout(&x2)?
        } else {
            x2
        };
        let a = self.a.as_tensor();
        let b = self.b.as_tensor();
        let xr = xd.matmul(&a.t()?.contiguous()?)?;
        let delta = xr.matmul(&b.t()?.contiguous()?)?;
        let delta = (delta * self.scaling)?;
        out_dims.push(self.out_features);
        Ok(delta.reshape(out_dims)?)
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let base = self.base_forward(x)?;
        let delta = self.delta_forward(x, train)?;
        Ok((base + delta)?)
    }

    fn flatten_input(&self, x: &Tensor) -> Result<(Tensor, Vec<usize>, usize)> {
        let dims = x.dims().to_vec();
        let last = *dims.last().unwrap_or(&0);
        if last != self.in_features {
            bail!(
                "LoraTrainable: input last dim {} != in_features {}",
                last,
                self.in_features
            );
        }
        let leading: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x
            .reshape((leading, self.in_features))?
            .to_dtype(self.compute_dtype)?;
        let out_dims = dims[..dims.len() - 1].to_vec();
        Ok((x2, out_dims, leading))
    }

    fn apply_dropout(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.elem_count();
        let mut rng = DetRng::new(0xD0_9F ^ n as u64);
        let keep = 1.0f32 - self.dropout;
        let scale = 1.0f32 / keep.max(f32::MIN_POSITIVE);
        let mask: Vec<f32> = (0..n)
            .map(|_| if rng.next_f32() < keep { scale } else { 0.0 })
            .collect();
        let mask = Tensor::from_vec(mask, x.shape(), x.device())?.to_dtype(x.dtype())?;
        Ok((x * mask)?)
    }

    pub fn deterministic(&self) -> bool {
        self.det_seeded
    }
}

pub struct PeftEntry<'a> {
    pub module: String,
    pub lora: &'a LoraTrainable,
}

pub fn save_peft(
    dir: impl AsRef<Path>,
    entries: &[PeftEntry<'_>],
    base_model_name_or_path: &str,
) -> Result<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;

    if entries.is_empty() {
        bail!("save_peft: no entries");
    }

    let r = entries[0].lora.r;
    let alpha = entries[0].lora.alpha;
    for e in entries {
        if e.lora.r != r || e.lora.alpha != alpha {
            bail!("save_peft: all modules must share the same r and alpha");
        }
    }

    let mut targets: Vec<String> = Vec::new();
    for e in entries {
        let leaf = e.module.rsplit('.').next().unwrap_or(&e.module).to_string();
        if !targets.contains(&leaf) {
            targets.push(leaf);
        }
    }
    targets.sort();

    let cfg = serde_json::json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": entries[0].lora.dropout,
        "bias": "none",
        "use_rslora": false,
        "use_dora": false,
        "modules_to_save": serde_json::Value::Null,
        "fan_in_fan_out": false,
        "inference_mode": true,
        "target_modules": targets,
        "base_model_name_or_path": base_model_name_or_path,
    });
    std::fs::write(
        dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&cfg)?,
    )?;

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for e in entries {
        let a = e.lora.a.as_tensor().to_dtype(DType::BF16)?.contiguous()?;
        let b = e.lora.b.as_tensor().to_dtype(DType::BF16)?.contiguous()?;
        tensors.insert(format!("base_model.model.{}.lora_A.weight", e.module), a);
        tensors.insert(format!("base_model.model.{}.lora_B.weight", e.module), b);
    }
    candle_core::safetensors::save(&tensors, dir.join("adapter_model.safetensors"))?;
    Ok(())
}

pub const LORA_TVIEW_DROPS_TWO_FACTOR_TRANSPOSE_COPIES_PER_APPLICATION: &str =
    "lora_delta_raw materialized a.t().contiguous() and b.t().contiguous() on every hook \
     application -- two copy kernels forward plus their copy backwards through autograd, paid \
     once per projection per layer per bucket (336 applications per qlora step before the \
     backward). NV_DOTS_LORA_TVIEW=1 feeds the transposed views straight to matmul; the mlx \
     steel GEMM walks the identical k-order through transposed addressing, verified bitwise \
     against the canonical 20-step trajectory before any flip";

fn lora_tview_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NV_DOTS_LORA_TVIEW").as_deref() == Ok("1"))
}

fn lora_delta_raw(
    a: &Tensor,
    b: &Tensor,
    scaling: f64,
    x2: &Tensor,
    win: Option<(usize, usize)>,
) -> Result<Tensor> {
    let x = x2.to_dtype(DType::F32)?;
    let a = a.to_dtype(DType::F32)?;
    let b = b.to_dtype(DType::F32)?;
    let b = match win {
        Some((off, rows)) => b.narrow(0, off, rows)?,
        None => b,
    };
    if lora_tview_enabled() {
        let _ = LORA_TVIEW_DROPS_TWO_FACTOR_TRANSPOSE_COPIES_PER_APPLICATION;
        let xr = x.matmul(&a.t()?)?;
        let delta = xr.matmul(&b.t()?)?;
        return Ok((delta * scaling)?);
    }
    let xr = x.matmul(&a.t()?.contiguous()?)?;
    let delta = xr.matmul(&b.t()?.contiguous()?)?;
    Ok((delta * scaling)?)
}

pub fn lora_delta(
    a: &Tensor,
    b: &Tensor,
    scaling: f64,
    x2: &Tensor,
    win: Option<(usize, usize)>,
) -> Result<Tensor> {
    lora_delta_raw(a, b, scaling, x2, win)
}

pub struct RawLora {
    pub module: String,
    pub a: Tensor,
    pub b: Tensor,
}

fn peft_config_json(
    entries_modules: impl Iterator<Item = String>,
    r: usize,
    alpha: f64,
    lora_dropout: f32,
    base_model_name_or_path: &str,
) -> serde_json::Value {
    let mut targets: Vec<String> = Vec::new();
    for m in entries_modules {
        let leaf = m.rsplit('.').next().unwrap_or(&m).to_string();
        if !targets.contains(&leaf) {
            targets.push(leaf);
        }
    }
    targets.sort();
    serde_json::json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": lora_dropout,
        "bias": "none",
        "use_rslora": false,
        "use_dora": false,
        "modules_to_save": serde_json::Value::Null,
        "fan_in_fan_out": false,
        "inference_mode": true,
        "target_modules": targets,
        "base_model_name_or_path": base_model_name_or_path,
    })
}

pub fn save_peft_raw(
    dir: impl AsRef<Path>,
    entries: &[RawLora],
    r: usize,
    alpha: f64,
    lora_dropout: f32,
    base_model_name_or_path: &str,
) -> Result<()> {
    save_peft_raw_with_speculator(
        dir,
        entries,
        r,
        alpha,
        lora_dropout,
        base_model_name_or_path,
        None,
    )
}

pub const SPECULATOR_SUBDIR: &str = "speculator";

pub const SPECULATOR_KIND_KEY: &str = "nv_speculator_kind";

pub const THE_SPECULATOR_PAIR_IS_WRITTEN_BY_THE_SAVE_PATH_ITSELF_BECAUSE_A_POST_HOC_PATCH_WOULD_BE_CLOBBERED_BY_ANY_RESAVE: &str =
    "save_peft_raw rewrites adapter_config.json from scratch on every call, so a speculator-kind \
     field patched in afterwards would vanish the next time the trainer saves. The speculator \
     deltas therefore ship through save_peft_raw_with_speculator in one call: the main config \
     gains nv_speculator_kind and the speculator/ subdir gains its own peft pair, inside the one \
     out_dir that stamp_adapter_id, roll_current_adapter_symlink and hot_load all operate on, so \
     target adapter and speculator swap atomically. probe_adapter requires exactly \
     adapter_config.json + adapter_model.safetensors and ignores extra entries, so the subdir is \
     inert to today's discovery and hot-load.";

pub struct SpeculatorArtifact {
    pub kind: String,
    pub entries: Vec<RawLora>,
    pub r: usize,
    pub alpha: f64,
}

fn write_adapter_pair_tensors(dir: &Path, entries: &[RawLora]) -> Result<()> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for e in entries {
        let a = e.a.to_dtype(DType::BF16)?.contiguous()?;
        let b = e.b.to_dtype(DType::BF16)?.contiguous()?;
        tensors.insert(format!("base_model.model.{}.lora_A.weight", e.module), a);
        tensors.insert(format!("base_model.model.{}.lora_B.weight", e.module), b);
    }
    candle_core::safetensors::save(&tensors, dir.join("adapter_model.safetensors"))?;
    Ok(())
}

pub fn save_peft_raw_with_speculator(
    dir: impl AsRef<Path>,
    entries: &[RawLora],
    r: usize,
    alpha: f64,
    lora_dropout: f32,
    base_model_name_or_path: &str,
    speculator: Option<&SpeculatorArtifact>,
) -> Result<()> {
    let _ = THE_SPECULATOR_PAIR_IS_WRITTEN_BY_THE_SAVE_PATH_ITSELF_BECAUSE_A_POST_HOC_PATCH_WOULD_BE_CLOBBERED_BY_ANY_RESAVE;
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;
    if entries.is_empty() {
        bail!("save_peft_raw: no entries");
    }
    let mut cfg = peft_config_json(
        entries.iter().map(|e| e.module.clone()),
        r,
        alpha,
        lora_dropout,
        base_model_name_or_path,
    );
    if let Some(spec) = speculator {
        cfg.as_object_mut()
            .expect("peft_config_json builds an object")
            .insert(
                SPECULATOR_KIND_KEY.to_string(),
                serde_json::Value::String(spec.kind.clone()),
            );
    }
    std::fs::write(
        dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&cfg)?,
    )?;
    write_adapter_pair_tensors(dir, entries)?;

    if let Some(spec) = speculator {
        if spec.entries.is_empty() {
            bail!(
                "speculator artifact of kind {} has no delta entries; shipping an empty \
                 speculator/ pair would advertise a co-trained drafter that was never trained",
                spec.kind
            );
        }
        let sdir = dir.join(SPECULATOR_SUBDIR);
        std::fs::create_dir_all(&sdir)?;
        let mut scfg = peft_config_json(
            spec.entries.iter().map(|e| e.module.clone()),
            spec.r,
            spec.alpha,
            0.0,
            base_model_name_or_path,
        );
        scfg.as_object_mut()
            .expect("peft_config_json builds an object")
            .insert(
                SPECULATOR_KIND_KEY.to_string(),
                serde_json::Value::String(spec.kind.clone()),
            );
        std::fs::write(
            sdir.join("adapter_config.json"),
            serde_json::to_string_pretty(&scfg)?,
        )?;
        write_adapter_pair_tensors(&sdir, &spec.entries)?;
    }
    Ok(())
}

pub struct FusedSplitHook {
    parts: Vec<(Tensor, Tensor, usize, usize)>,
    scaling: f64,
    in_features: usize,
    out_features: usize,
}

impl FusedSplitHook {
    pub fn new(
        parts: Vec<(Tensor, Tensor, usize, usize)>,
        scaling: f64,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        for (a, b, off, rows) in &parts {
            if a.dims().len() != 2 || b.dims().len() != 2 {
                bail!("FusedSplitHook: A and B must be 2-D");
            }
            if a.dims()[1] != in_features {
                bail!("FusedSplitHook: A in_features mismatch");
            }
            if b.dims()[0] != *rows || a.dims()[0] != b.dims()[1] {
                bail!("FusedSplitHook: B [rows,r] mismatch");
            }
            if off + rows > out_features {
                bail!("FusedSplitHook: window past fused out_features");
            }
        }
        Ok(Self {
            parts,
            scaling,
            in_features,
            out_features,
        })
    }
}

impl LoraDeltaHook for FusedSplitHook {
    fn in_features(&self) -> usize {
        self.in_features
    }
    fn out_features(&self) -> usize {
        self.out_features
    }
    fn apply(
        &self,
        x2: &Tensor,
        y2: &Tensor,
        win: Option<(usize, usize)>,
    ) -> Result<Option<Tensor>> {
        if win.is_some() {
            bail!(
                "FusedSplitHook: windowed forward_rows not supported (CPU serve path is win=None)"
            );
        }
        let x = x2.to_dtype(DType::F32)?;
        let n = x.dims()[0];
        let dev = x.device();
        let mut total = Tensor::zeros((n, self.out_features), DType::F32, dev)?;
        for (a, b, off, rows) in &self.parts {
            let d = lora_delta_raw(a, b, self.scaling, &x, None)?;
            let left = *off;
            let right = self.out_features - off - rows;
            let mut pieces: Vec<Tensor> = Vec::new();
            if left > 0 {
                pieces.push(Tensor::zeros((n, left), DType::F32, dev)?);
            }
            pieces.push(d);
            if right > 0 {
                pieces.push(Tensor::zeros((n, right), DType::F32, dev)?);
            }
            let padded = Tensor::cat(&pieces, 1)?;
            total = (total + padded)?;
        }
        let total = total.to_dtype(y2.dtype())?;
        Ok(Some((y2 + total)?))
    }
}

pub struct TrainingLoraHook {
    lora: Arc<LoraTrainable>,
    train: bool,
}

impl TrainingLoraHook {
    pub fn new(lora: Arc<LoraTrainable>, train: bool) -> Self {
        Self { lora, train }
    }
    pub fn lora(&self) -> &Arc<LoraTrainable> {
        &self.lora
    }
}

impl LoraDeltaHook for TrainingLoraHook {
    fn in_features(&self) -> usize {
        self.lora.in_features()
    }
    fn out_features(&self) -> usize {
        self.lora.out_features()
    }
    fn apply(
        &self,
        x2: &Tensor,
        y2: &Tensor,
        win: Option<(usize, usize)>,
    ) -> Result<Option<Tensor>> {
        let xd = if self.train && self.lora.dropout > 0.0 {
            self.lora.apply_dropout(&x2.to_dtype(DType::F32)?)?
        } else {
            x2.to_dtype(DType::F32)?
        };
        let delta = lora_delta_raw(
            self.lora.a_tensor(),
            self.lora.b_tensor(),
            self.lora.scaling(),
            &xd,
            win,
        )?;
        let delta = delta.to_dtype(y2.dtype())?;
        Ok(Some((y2 + delta)?))
    }
}

pub struct StaticLoraHook {
    a: Tensor,
    b: Tensor,
    scaling: f64,
    in_features: usize,
    out_features: usize,
}

impl StaticLoraHook {
    pub fn new(a: Tensor, b: Tensor, scaling: f64) -> Result<Self> {
        if a.dims().len() != 2 || b.dims().len() != 2 {
            bail!("StaticLoraHook: A and B must be 2-D");
        }
        let r = a.dims()[0];
        let in_features = a.dims()[1];
        let out_features = b.dims()[0];
        if b.dims()[1] != r {
            bail!(
                "StaticLoraHook: A [r,in]=[{},{}] and B [out,r]=[{},{}] rank mismatch",
                r,
                in_features,
                out_features,
                b.dims()[1]
            );
        }
        Ok(Self {
            a: a.detach(),
            b: b.detach(),
            scaling,
            in_features,
            out_features,
        })
    }
}

impl LoraDeltaHook for StaticLoraHook {
    fn in_features(&self) -> usize {
        self.in_features
    }
    fn out_features(&self) -> usize {
        self.out_features
    }
    fn apply(
        &self,
        x2: &Tensor,
        y2: &Tensor,
        win: Option<(usize, usize)>,
    ) -> Result<Option<Tensor>> {
        let delta = lora_delta_raw(&self.a, &self.b, self.scaling, x2, win)?;
        let delta = delta.to_dtype(y2.dtype())?;
        Ok(Some((y2 + delta)?))
    }
}

pub fn grad_is_finite_by_one_scalar_readback_because_nan_and_inf_survive_a_sum(
    g: &Tensor,
) -> Result<bool> {
    Ok(g.to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?
        .is_finite())
}

pub fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?;
    let b = b.to_dtype(DType::F32)?;
    let d = (a - b)?.abs()?.flatten_all()?.max(0)?;
    Ok(d.to_scalar::<f32>()?)
}

fn byte_tokenize(s: &str, vocab: usize) -> Vec<u32> {
    s.bytes().map(|b| (b as u32) % vocab as u32).collect()
}

pub const JSONL_MAX_SEQ_IS_512_SHARED_BY_BOTH_ENDPOINT_ARMS_BECAUSE_THE_DEFAULT_QWEN_WINDOWS_GDN_GRAPH_AT_512_IS_ABOUT_14_GB:
    usize = 512;

pub const BYTE_TOKENIZATION_EXISTS_ONLY_FOR_BASES_THAT_SHIP_NO_TOKENIZER_JSON: &str =
    "qwen38_fixture, the fine-tuning e2e tiny base and examples/lora/example-base ship \
     config.json + model.safetensors without a tokenizer.json, so their jsonl rows fall back to \
     bytes mod vocab. Every real base directory carries tokenizer.json and its rows tokenize \
     through it; byte ids against a real vocabulary would train the adapter on template noise, \
     not language. for_base enforces this: a directory that looks real (a sibling \
     tokenizer.model or tokenizer_config.json, or a real-sized config vocab) is an error, not a \
     fallback, and the fixture fallback announces itself on stderr.";

pub const BYTE_FALLBACK_REFUSES_VOCABS_OF_4096_PLUS_BECAUSE_FIXTURE_VOCABS_ARE_UNDER_512_AND_REAL_ONES_START_NEAR_32K:
    usize = 4096;

pub enum DatasetTokenizer {
    Real(Box<tokenizers::Tokenizer>),
    ByteFallbackForTokenizerlessFixtureBases,
}

impl DatasetTokenizer {
    pub fn for_base(base: &Path, vocab: usize) -> Result<Self> {
        let tokenizer_json = base.join("tokenizer.json");
        if tokenizer_json.is_file() {
            return Ok(Self::Real(Box::new(nv_tokenizer::load_tokenizer(
                &tokenizer_json,
            )?)));
        }
        for leaf in ["tokenizer.model", "tokenizer_config.json"] {
            if base.join(leaf).is_file() {
                bail!(
                    "base {} ships {leaf} but no tokenizer.json, so this is a real checkpoint \
                     with an unusable tokenizer snapshot; refusing the byte fallback because \
                     {}",
                    base.display(),
                    BYTE_TOKENIZATION_EXISTS_ONLY_FOR_BASES_THAT_SHIP_NO_TOKENIZER_JSON
                );
            }
        }
        if vocab >= BYTE_FALLBACK_REFUSES_VOCABS_OF_4096_PLUS_BECAUSE_FIXTURE_VOCABS_ARE_UNDER_512_AND_REAL_ONES_START_NEAR_32K
        {
            bail!(
                "base {} has no tokenizer.json yet a vocab of {vocab}, so this is a real \
                 checkpoint with an incomplete snapshot; refusing the byte fallback because {}",
                base.display(),
                BYTE_TOKENIZATION_EXISTS_ONLY_FOR_BASES_THAT_SHIP_NO_TOKENIZER_JSON
            );
        }
        eprintln!(
            "nv-train: base {} ships no tokenizer.json; jsonl rows tokenize as bytes mod \
             {vocab} (fixture-only path)",
            base.display()
        );
        Ok(Self::ByteFallbackForTokenizerlessFixtureBases)
    }

    pub fn is_real(&self) -> bool {
        matches!(self, Self::Real(_))
    }

    pub fn encode(&self, s: &str, vocab: usize) -> Result<Vec<u32>> {
        match self {
            Self::Real(tok) => {
                let enc = tok
                    .encode(s, false)
                    .map_err(|e| anyhow::anyhow!("tokenize dataset row: {e}"))?;
                let ids: Vec<u32> = enc.get_ids().to_vec();
                if let Some((pos, id)) = ids
                    .iter()
                    .enumerate()
                    .find(|(_, i)| (**i as usize) >= vocab)
                {
                    bail!(
                        "tokenizer produced id {id} at token position {pos} but the model config \
                         says vocab_size {vocab}; dropping or wrapping it would silently corrupt \
                         the token stream and the prompt boundary, so this base/tokenizer pair \
                         cannot be trained on"
                    );
                }
                Ok(ids)
            }
            Self::ByteFallbackForTokenizerlessFixtureBases => Ok(byte_tokenize(s, vocab)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenRow {
    pub ids: Vec<u32>,
    pub completion_start: usize,
}

impl TokenRow {
    pub fn over_all_positions(ids: Vec<u32>) -> Self {
        Self {
            ids,
            completion_start: 0,
        }
    }

    pub fn target_mask_where_predicting_a_prompt_token_weighs_zero(&self) -> Vec<f32> {
        (1..self.ids.len())
            .map(|predicted| {
                if predicted >= self.completion_start {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    pub fn unmasked_targets(&self) -> usize {
        self.ids.len().saturating_sub(self.completion_start.max(1))
    }
}

pub fn eval_loss_scalar_is_the_sum_of_the_per_row_ce_means_in_dataset_row_order(
    per_row_ce: &[f32],
) -> Option<f32> {
    if per_row_ce.is_empty() {
        None
    } else {
        Some(per_row_ce.iter().sum())
    }
}

pub fn ce_mean_over_unmasked_positions(
    logits_t_vocab: &Tensor,
    targets: &Tensor,
    mask: &[f32],
) -> Result<Tensor> {
    let t = targets.dims1()?;
    ensure_mask_shape(mask, t)?;
    if mask.iter().all(|w| *w == 1.0) {
        return Ok(candle_nn::loss::cross_entropy(logits_t_vocab, targets)?);
    }
    let unmasked: f32 = mask.iter().sum();
    let logp = candle_nn::ops::log_softmax(logits_t_vocab, candle_core::D::Minus1)?;
    let picked = logp.gather(&targets.unsqueeze(1)?, 1)?.squeeze(1)?;
    let m = Tensor::from_vec(mask.to_vec(), t, logits_t_vocab.device())?;
    Ok(picked
        .mul(&m)?
        .sum_all()?
        .affine(-1.0 / unmasked as f64, 0.0)?)
}

pub fn ce_weighted_over_positions(
    logits_t_vocab: &Tensor,
    targets: &Tensor,
    weights: &[f32],
) -> Result<Tensor> {
    let t = targets.dims1()?;
    ensure_mask_shape(weights, t)?;
    let logp = candle_nn::ops::log_softmax(logits_t_vocab, candle_core::D::Minus1)?;
    let picked = logp.gather(&targets.unsqueeze(1)?, 1)?.squeeze(1)?;
    let w = Tensor::from_vec(weights.to_vec(), t, logits_t_vocab.device())?;
    Ok(picked.mul(&w)?.sum_all()?.neg()?)
}

fn ensure_mask_shape(mask: &[f32], targets: usize) -> Result<()> {
    if mask.len() != targets {
        bail!(
            "loss mask has {} weights for {} target positions",
            mask.len(),
            targets
        );
    }
    if mask.iter().all(|w| *w == 0.0) {
        bail!("every target position is masked, so this row has no loss to train on");
    }
    Ok(())
}

pub fn load_jsonl_token_dataset(
    path: &Path,
    base: &Path,
    vocab: usize,
    max_seq: usize,
) -> Result<Vec<TokenRow>> {
    use anyhow::Context as _;
    let tokenizer = DatasetTokenizer::for_base(base, vocab)?;
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read dataset {}", path.display()))?;
    let mut examples = Vec::new();
    let mut parsed_rows = 0usize;
    for (lineno, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        parsed_rows += 1;
        let v: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("{}:{}: invalid json", path.display(), lineno + 1))?;
        let mut row: TokenRow = if let Some(arr) = v.get("ids").and_then(|x| x.as_array()) {
            TokenRow::over_all_positions(
                arr.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| (x as u32) % vocab as u32)
                    .collect(),
            )
        } else if let Some(t) = v.get("text").and_then(|x| x.as_str()) {
            TokenRow::over_all_positions(tokenizer.encode(t, vocab)?)
        } else if v.get("prompt").is_some() || v.get("completion").is_some() {
            let prompt = v.get("prompt").and_then(|x| x.as_str()).unwrap_or("");
            let completion = v.get("completion").and_then(|x| x.as_str()).unwrap_or("");
            let mut ids = tokenizer.encode(prompt, vocab)?;
            let completion_start = ids.len();
            ids.extend(tokenizer.encode(completion, vocab)?);
            TokenRow {
                ids,
                completion_start,
            }
        } else {
            bail!(
                "{}:{}: line has none of the supported keys (ids | text | prompt/completion)",
                path.display(),
                lineno + 1
            );
        };
        if row.ids.len() > max_seq {
            row.ids.truncate(max_seq);
        }
        if row.ids.len() >= 2 && row.completion_start < row.ids.len() {
            examples.push(row);
        }
    }
    if examples.is_empty() {
        bail!(
            "dataset {} produced no usable examples (>=2 tokens with at least one completion \
             token to predict)",
            path.display()
        );
    }
    if examples.len() < parsed_rows {
        eprintln!(
            "[jsonl-dataset] {}: kept {} of {} parsed rows (dropped rows had <2 tokens or a \
             completion entirely past max_seq {}); per-row outputs index the loaded dataset, \
             not file lines",
            path.display(),
            examples.len(),
            parsed_rows,
            max_seq
        );
    }
    Ok(examples)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_lora(out: usize, inp: usize, cfg: LoraConfig) -> LoraTrainable {
        let base = Tensor::zeros((out, inp), DType::F32, &Device::Cpu).unwrap();
        LoraTrainable::new(&base, cfg, 7, &Device::Cpu).unwrap()
    }

    #[test]
    fn scaling_is_alpha_over_rank_and_unit_scaling_means_exactly_one() {
        assert_eq!(LoraConfig::unit_scaling(8).scaling(), 1.0);
        assert_eq!(LoraConfig::unit_scaling(64).scaling(), 1.0);
        let cfg = LoraConfig {
            r: 8,
            alpha: 16.0,
            dropout: 0.0,
        };
        assert_eq!(cfg.scaling(), 2.0, "alpha 16 over rank 8");

        let cfg = LoraConfig {
            r: 32,
            alpha: 8.0,
            dropout: 0.0,
        };
        assert_eq!(cfg.scaling(), 0.25);
    }

    #[test]
    fn a_rank_zero_or_non_matrix_base_is_refused_rather_than_producing_an_adapter() {
        let base = Tensor::zeros((4, 4), DType::F32, &Device::Cpu).unwrap();
        assert!(
            LoraTrainable::new(
                &base,
                LoraConfig {
                    r: 0,
                    alpha: 1.0,
                    dropout: 0.0
                },
                1,
                &Device::Cpu
            )
            .is_err(),
            "rank 0 has no low-rank factors at all"
        );
        let rank3 = Tensor::zeros((2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        assert!(LoraTrainable::new(&rank3, LoraConfig::unit_scaling(4), 1, &Device::Cpu).is_err());
    }

    #[test]
    fn b_starts_at_zero_so_an_untrained_adapter_is_the_identity() {
        let lora = cpu_lora(6, 4, LoraConfig::unit_scaling(2));
        let x = Tensor::ones((3, 4), DType::F32, &Device::Cpu).unwrap();
        let delta = lora_delta(lora.a.as_tensor(), lora.b.as_tensor(), 1.0, &x, None).unwrap();
        assert_eq!(
            max_abs_diff(
                &delta,
                &Tensor::zeros((3, 6), DType::F32, &Device::Cpu).unwrap()
            )
            .unwrap(),
            0.0
        );

        let a_mag = max_abs_diff(
            lora.a.as_tensor(),
            &Tensor::zeros(lora.a.as_tensor().shape(), DType::F32, &Device::Cpu).unwrap(),
        )
        .unwrap();
        assert!(a_mag > 0.0, "A must be randomly initialised, got all zeros");
    }

    #[test]
    fn the_same_seed_gives_the_same_a_and_a_different_seed_does_not() {
        let base = Tensor::zeros((6, 4), DType::F32, &Device::Cpu).unwrap();
        let cfg = LoraConfig::unit_scaling(2);
        let one = LoraTrainable::new(&base, cfg, 42, &Device::Cpu).unwrap();
        let same = LoraTrainable::new(&base, cfg, 42, &Device::Cpu).unwrap();
        let other = LoraTrainable::new(&base, cfg, 43, &Device::Cpu).unwrap();
        assert_eq!(
            max_abs_diff(one.a.as_tensor(), same.a.as_tensor()).unwrap(),
            0.0,
            "a seeded init that is not reproducible cannot be compared across runs"
        );
        assert!(max_abs_diff(one.a.as_tensor(), other.a.as_tensor()).unwrap() > 0.0);
    }

    #[test]
    fn the_window_narrows_b_from_the_offset_so_a_fused_split_writes_its_own_rows() {
        let dev = Device::Cpu;
        let a = Tensor::ones((2, 3), DType::F32, &dev).unwrap();
        let b = Tensor::from_vec(
            vec![1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            (4, 2),
            &dev,
        )
        .unwrap();
        let x = Tensor::ones((1, 3), DType::F32, &dev).unwrap();

        let full = lora_delta(&a, &b, 1.0, &x, None).unwrap();
        assert_eq!(full.dims(), &[1, 4]);
        assert_eq!(
            full.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![6.0, 12.0, 18.0, 24.0]
        );

        let win = lora_delta(&a, &b, 1.0, &x, Some((1, 2))).unwrap();
        assert_eq!(win.dims(), &[1, 2], "the window sets the output width");
        assert_eq!(
            win.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![12.0, 18.0],
            "rows 1..3 of B, not 0..2"
        );
    }

    #[test]
    fn a_saved_adapter_carries_the_tensor_names_a_peft_loader_looks_for() {
        let dir = std::env::temp_dir().join(format!("nvtrain-peft-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let q = cpu_lora(
            4,
            4,
            LoraConfig {
                r: 2,
                alpha: 4.0,
                dropout: 0.0,
            },
        );
        let k = cpu_lora(
            4,
            4,
            LoraConfig {
                r: 2,
                alpha: 4.0,
                dropout: 0.0,
            },
        );
        save_peft(
            &dir,
            &[
                PeftEntry {
                    module: "layers.0.self_attn.q_proj".into(),
                    lora: &q,
                },
                PeftEntry {
                    module: "layers.1.self_attn.k_proj".into(),
                    lora: &k,
                },
            ],
            "google/gemma-4-E4B-it",
        )
        .unwrap();

        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("adapter_config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(cfg["peft_type"], "LORA");
        assert_eq!(cfg["r"], 2);
        assert_eq!(cfg["lora_alpha"], 4.0);
        assert_eq!(cfg["base_model_name_or_path"], "google/gemma-4-E4B-it");

        assert_eq!(
            cfg["target_modules"],
            serde_json::json!(["k_proj", "q_proj"])
        );

        let st =
            candle_core::safetensors::load(dir.join("adapter_model.safetensors"), &Device::Cpu)
                .unwrap();
        let mut keys: Vec<&String> = st.keys().collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
                "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
                "base_model.model.layers.1.self_attn.k_proj.lora_A.weight",
                "base_model.model.layers.1.self_attn.k_proj.lora_B.weight",
            ]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_adapter_whose_modules_disagree_on_r_or_alpha_is_refused() {
        let dir = std::env::temp_dir().join(format!("nvtrain-mixed-{}", std::process::id()));
        let a = cpu_lora(
            4,
            4,
            LoraConfig {
                r: 2,
                alpha: 4.0,
                dropout: 0.0,
            },
        );
        let b_diff_r = cpu_lora(
            4,
            4,
            LoraConfig {
                r: 4,
                alpha: 4.0,
                dropout: 0.0,
            },
        );
        let b_diff_alpha = cpu_lora(
            4,
            4,
            LoraConfig {
                r: 2,
                alpha: 8.0,
                dropout: 0.0,
            },
        );

        assert!(
            save_peft(&dir, &[], "m").is_err(),
            "no entries is not an adapter"
        );
        assert!(save_peft(
            &dir,
            &[
                PeftEntry {
                    module: "a".into(),
                    lora: &a
                },
                PeftEntry {
                    module: "b".into(),
                    lora: &b_diff_r
                },
            ],
            "m"
        )
        .is_err());
        assert!(save_peft(
            &dir,
            &[
                PeftEntry {
                    module: "a".into(),
                    lora: &a
                },
                PeftEntry {
                    module: "b".into(),
                    lora: &b_diff_alpha
                },
            ],
            "m"
        )
        .is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
