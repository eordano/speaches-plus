use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor, Var};
use nv_layers::linear::LoraDeltaHook;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

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
    base: Tensor,
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
        let out_features = dims[0];
        let in_features = dims[1];
        if cfg.r == 0 {
            bail!("LoRA rank r must be positive");
        }

        let base = base.detach().to_device(device)?;

        let mut rng = DetRng::new(seed);
        let std = 1.0f32 / (in_features as f32).sqrt();
        let a_vals: Vec<f32> = (0..cfg.r * in_features)
            .map(|_| rng.next_normal() * std)
            .collect();
        let a_t = Tensor::from_vec(a_vals, (cfg.r, in_features), device)?;
        let a = Var::from_tensor(&a_t)?;

        let b = Var::zeros((out_features, cfg.r), DType::F32, device)?;

        let det_seeded = std::env::var("NV_DETERMINISTIC").ok().as_deref() == Some("1");

        Ok(Self {
            base,
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

    pub fn base_tensor(&self) -> &Tensor {
        &self.base
    }

    pub fn base_forward(&self, x: &Tensor) -> Result<Tensor> {
        let (x2, mut out_dims, n) = self.flatten_input(x)?;
        let base_f = self.base.to_dtype(self.compute_dtype)?;
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
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;
    if entries.is_empty() {
        bail!("save_peft_raw: no entries");
    }
    let cfg = peft_config_json(
        entries.iter().map(|e| e.module.clone()),
        r,
        alpha,
        lora_dropout,
        base_model_name_or_path,
    );
    std::fs::write(
        dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&cfg)?,
    )?;

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

pub fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?;
    let b = b.to_dtype(DType::F32)?;
    let d = (a - b)?.abs()?.flatten_all()?.max(0)?;
    Ok(d.to_scalar::<f32>()?)
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
        let cfg = LoraConfig { r: 8, alpha: 16.0, dropout: 0.0 };
        assert_eq!(cfg.scaling(), 2.0, "alpha 16 over rank 8");

        let cfg = LoraConfig { r: 32, alpha: 8.0, dropout: 0.0 };
        assert_eq!(cfg.scaling(), 0.25);
    }

    #[test]
    fn a_rank_zero_or_non_matrix_base_is_refused_rather_than_producing_an_adapter() {
        let base = Tensor::zeros((4, 4), DType::F32, &Device::Cpu).unwrap();
        assert!(
            LoraTrainable::new(&base, LoraConfig { r: 0, alpha: 1.0, dropout: 0.0 }, 1, &Device::Cpu)
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
        assert_eq!(max_abs_diff(&delta, &Tensor::zeros((3, 6), DType::F32, &Device::Cpu).unwrap()).unwrap(), 0.0);

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
        assert_eq!(full.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![6.0, 12.0, 18.0, 24.0]);

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
        let q = cpu_lora(4, 4, LoraConfig { r: 2, alpha: 4.0, dropout: 0.0 });
        let k = cpu_lora(4, 4, LoraConfig { r: 2, alpha: 4.0, dropout: 0.0 });
        save_peft(
            &dir,
            &[
                PeftEntry { module: "layers.0.self_attn.q_proj".into(), lora: &q },
                PeftEntry { module: "layers.1.self_attn.k_proj".into(), lora: &k },
            ],
            "google/gemma-4-E4B-it",
        )
        .unwrap();

        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("adapter_config.json")).unwrap())
                .unwrap();
        assert_eq!(cfg["peft_type"], "LORA");
        assert_eq!(cfg["r"], 2);
        assert_eq!(cfg["lora_alpha"], 4.0);
        assert_eq!(cfg["base_model_name_or_path"], "google/gemma-4-E4B-it");

        assert_eq!(cfg["target_modules"], serde_json::json!(["k_proj", "q_proj"]));

        let st = candle_core::safetensors::load(dir.join("adapter_model.safetensors"), &Device::Cpu)
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
        let a = cpu_lora(4, 4, LoraConfig { r: 2, alpha: 4.0, dropout: 0.0 });
        let b_diff_r = cpu_lora(4, 4, LoraConfig { r: 4, alpha: 4.0, dropout: 0.0 });
        let b_diff_alpha = cpu_lora(4, 4, LoraConfig { r: 2, alpha: 8.0, dropout: 0.0 });

        assert!(save_peft(&dir, &[], "m").is_err(), "no entries is not an adapter");
        assert!(save_peft(
            &dir,
            &[
                PeftEntry { module: "a".into(), lora: &a },
                PeftEntry { module: "b".into(), lora: &b_diff_r },
            ],
            "m"
        )
        .is_err());
        assert!(save_peft(
            &dir,
            &[
                PeftEntry { module: "a".into(), lora: &a },
                PeftEntry { module: "b".into(), lora: &b_diff_alpha },
            ],
            "m"
        )
        .is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
