use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use nv_quant::LinearKind;

#[cfg(feature = "cuda")]
use candle_core::Device;
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::sync::Mutex;

pub trait LoraDeltaHook: Send + Sync {
    fn in_features(&self) -> usize;
    fn out_features(&self) -> usize;
    fn apply(
        &self,
        x2: &Tensor,
        y2: &Tensor,
        win: Option<(usize, usize)>,
    ) -> Result<Option<Tensor>>;
}

#[cfg(feature = "cuda")]
impl LoraDeltaHook for crate::lora_slots::LoraHook {
    fn in_features(&self) -> usize {
        crate::lora_slots::LoraHook::in_features(self)
    }

    fn out_features(&self) -> usize {
        crate::lora_slots::LoraHook::out_features(self)
    }

    fn apply(
        &self,
        x2: &Tensor,
        y2: &Tensor,
        win: Option<(usize, usize)>,
    ) -> Result<Option<Tensor>> {
        crate::lora_slots::LoraHook::apply(self, x2, y2, win)?;
        Ok(None)
    }
}

pub struct Linear {
    kind: LinearKind,
    storage: LinearStorage,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
    lora: std::sync::RwLock<Option<Arc<dyn LoraDeltaHook>>>,
}

#[cfg(feature = "cuda")]
pub struct FusedPreNorm<'a> {
    pub weight_bf16: &'a Tensor,
    pub eps: f32,
}

thread_local! {

    static FORCE_DENSE_BF16: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn with_dense_bf16<R>(f: impl FnOnce() -> R) -> R {
    struct Guard(bool);
    impl Drop for Guard {
        fn drop(&mut self) {
            FORCE_DENSE_BF16.with(|c| c.set(self.0));
        }
    }
    let prev = FORCE_DENSE_BF16.with(|c| {
        let p = c.get();
        c.set(true);
        p
    });
    let _g = Guard(prev);
    f()
}

#[cfg(feature = "cuda")]
fn force_dense_bf16_enabled() -> bool {
    FORCE_DENSE_BF16.with(|c| c.get())
}

pub fn nvfp4_true_m_from(raw: Option<&str>) -> bool {
    raw.is_some_and(|v| v != "0")
}

pub fn fp8_scale_mode_from(raw: Option<&str>) -> nv_quant::fp8::Fp8ScaleMode {
    use nv_quant::fp8::Fp8ScaleMode;
    match raw {
        Some(v)
            if v.eq_ignore_ascii_case("tensor")
                || v.eq_ignore_ascii_case("per_tensor")
                || v.eq_ignore_ascii_case("per-tensor")
                || v == "0" =>
        {
            Fp8ScaleMode::PerTensor
        }
        _ => Fp8ScaleMode::PerOuterRow,
    }
}

pub fn fp8_scale_mode() -> nv_quant::fp8::Fp8ScaleMode {
    fp8_scale_mode_from(std::env::var("NV_FP8_SCALE_MODE").ok().as_deref())
}

#[cfg(feature = "cuda")]
fn fp8_per_row_probe(
    runner: &Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    ordinal: usize,
) -> std::result::Result<(), String> {
    static PROBE: std::sync::Mutex<
        std::collections::BTreeMap<usize, std::result::Result<(), String>>,
    > = std::sync::Mutex::new(std::collections::BTreeMap::new());
    if let Some(v) = PROBE.lock().unwrap().get(&ordinal) {
        return v.clone();
    }
    let verdict = match runner.lock() {
        Err(e) => return Err(format!("fp8 runner mutex poisoned: {e}")),
        Ok(mut r) => r
            .probe_per_row_scale_support()
            .map_err(|e| format!("{e:#}")),
    };
    PROBE
        .lock()
        .unwrap()
        .entry(ordinal)
        .or_insert(verdict)
        .clone()
}

pub fn fp8_weight_payload(
    weight_host: &[half::bf16],
    out_features: usize,
    in_features: usize,
    checkpoint_rows: Option<&[f32]>,
    mode: nv_quant::fp8::Fp8ScaleMode,
) -> Result<(Vec<u8>, Vec<f32>)> {
    use nv_quant::fp8::{
        quantize_e4m3_per_row, quantize_e4m3_per_tensor, quantize_e4m3_with_row_scales,
        Fp8ScaleMode,
    };
    if weight_host.len() != out_features * in_features {
        anyhow::bail!(
            "fp8 weight payload: {} values for a [{out_features}, {in_features}] weight",
            weight_host.len()
        );
    }
    match (mode, checkpoint_rows) {
        (Fp8ScaleMode::PerOuterRow, Some(rows)) => {
            if rows.len() != out_features {
                anyhow::bail!(
                    "fp8 weight payload: {} checkpoint scales for {out_features} output rows",
                    rows.len()
                );
            }
            let bytes =
                quantize_e4m3_with_row_scales(weight_host, out_features, in_features, rows)?;
            Ok((bytes, rows.to_vec()))
        }
        (Fp8ScaleMode::PerOuterRow, None) => {
            quantize_e4m3_per_row(weight_host, out_features, in_features)
        }
        (Fp8ScaleMode::PerTensor, Some(rows)) => {
            let first = *rows
                .first()
                .ok_or_else(|| anyhow::anyhow!("empty checkpoint weight_scale"))?;
            if rows.iter().any(|s| *s != first) {
                anyhow::bail!(
                    "NV_FP8_SCALE_MODE=tensor cannot represent this checkpoint: its \
                     weight_scale varies across output rows ({} distinct values), and \
                     collapsing it to one scale would silently coarsen the checkpoint's own \
                     granularity. Unset NV_FP8_SCALE_MODE to use per-row scaling.",
                    {
                        let mut v = rows.to_vec();
                        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        v.dedup();
                        v.len()
                    }
                );
            }
            let bytes =
                quantize_e4m3_with_row_scales(weight_host, out_features, in_features, rows)?;
            Ok((bytes, vec![first; out_features]))
        }
        (Fp8ScaleMode::PerTensor, None) => {
            let (bytes, scale) = quantize_e4m3_per_tensor(weight_host);
            Ok((bytes, vec![scale; out_features]))
        }
    }
}

pub fn small_m_det_from(raw: Option<&str>) -> bool {
    raw != Some("0")
}

#[cfg(feature = "cuda")]
pub const NVFP4_M1_GEMV_STAYS_OPT_IN_UNTIL_A_SERVING_AB_LANDS: &str =
    "NV_NVFP4_M1_GEMV=1 routes m=1 NVFP4 linears through nvfp4_gemv_bf16act (bf16 activations, \
     no activation quantize, no m=16 padding) -- it outruns the padded-16 LT route at 31B \
     decode shapes (current numbers: perf/runs.jsonl) and is numerically TIGHTER than the \
     default (it skips the fp4 activation round-trip), but any argmax it flips vs the shipping \
     route is unmeasured, so the LT route stays the default until a serving A/B lands";

#[cfg(feature = "cuda")]
fn nvfp4_m1_gemv_opted_in() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NV_NVFP4_M1_GEMV").ok().as_deref() == Some("1"))
}

#[cfg(feature = "cuda")]
fn nvfp4_true_m_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| nvfp4_true_m_from(std::env::var("NV_NVFP4_TRUE_M").ok().as_deref()))
}

#[cfg(feature = "cuda")]
fn small_m_det_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| small_m_det_from(std::env::var("NV_BF16_SMALLM_DET").ok().as_deref()))
}

#[cfg(feature = "cuda")]
fn nvfp4_quant_fullpad_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NV_NVFP4_QUANT_FULLPAD").is_ok_and(|v| v != "0"))
}

#[cfg(feature = "cuda")]
thread_local! {
    static VERIFY_TC_FP8_LT_GEMM_SCOPE_SET_ONLY_AROUND_A_VERIFY_CHAIN_FORWARD:
        std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(feature = "cuda")]
pub fn verify_tc_fp8_lt_gemm_scope_active() -> bool {
    VERIFY_TC_FP8_LT_GEMM_SCOPE_SET_ONLY_AROUND_A_VERIFY_CHAIN_FORWARD.with(|c| c.get())
}

#[cfg(feature = "cuda")]
pub struct VerifyTcFp8LtGemmScopeGuard {
    was_off_so_drop_restores_off: bool,
}

#[cfg(feature = "cuda")]
impl VerifyTcFp8LtGemmScopeGuard {
    pub fn enter_if(on: bool) -> Self {
        let was_off = !verify_tc_fp8_lt_gemm_scope_active();
        if on && was_off {
            VERIFY_TC_FP8_LT_GEMM_SCOPE_SET_ONLY_AROUND_A_VERIFY_CHAIN_FORWARD
                .with(|c| c.set(true));
        }
        Self {
            was_off_so_drop_restores_off: on && was_off,
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for VerifyTcFp8LtGemmScopeGuard {
    fn drop(&mut self) {
        if self.was_off_so_drop_restores_off {
            let _ = VERIFY_TC_FP8_LT_GEMM_SCOPE_SET_ONLY_AROUND_A_VERIFY_CHAIN_FORWARD
                .try_with(|c| c.set(false));
        }
    }
}

#[cfg(feature = "cuda")]
fn stream_is_capturing(stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> bool {
    use cudarc::driver::sys as drv;
    let mut st = drv::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let rc = unsafe { drv::cuStreamIsCapturing(stream.cu_stream(), &mut st) };
    rc == drv::CUresult::CUDA_SUCCESS
        && st != drv::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
}

enum LinearStorage {
    Bf16 {
        weight: Tensor,

        weight_t: Option<Tensor>,
    },
    #[cfg(feature = "cuda")]
    Fp8E4m3(Fp8Storage),
    #[cfg(feature = "cuda")]
    Nvfp4(Nvfp4Storage),
}

#[cfg(feature = "cuda")]
struct Fp8Storage {
    weight_u8: cudarc::driver::CudaSlice<u8>,
    #[allow(dead_code)]
    a_scale_dev: cudarc::driver::CudaSlice<f32>,

    b_scale_dev: cudarc::driver::CudaSlice<f32>,

    b_scale_rows_dev: cudarc::driver::CudaSlice<f32>,

    weight_scale_rows: Vec<f32>,
    scale_mode: nv_quant::fp8::Fp8ScaleMode,
    cuda_device: candle_core::CudaDevice,
    runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
}

#[cfg(feature = "cuda")]
struct Nvfp4Storage {
    weight_u8: cudarc::driver::CudaSlice<u8>,
    weight_scales_cm: cudarc::driver::CudaSlice<u8>,
    cuda_device: candle_core::CudaDevice,
    runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    weight_alpha: f32,

    weight_alpha_dev: cudarc::driver::CudaSlice<f32>,
    input_stored_global: f32,
    a_staging: Mutex<std::collections::HashMap<(usize, usize), StagedActivations>>,
}

#[cfg(feature = "cuda")]
struct StagedActivations {
    epoch: u64,
    hwm_rows: usize,
    packed: cudarc::driver::CudaSlice<u8>,
    scales: cudarc::driver::CudaSlice<u8>,
}

pub(crate) fn this_call_is_building_an_autograd_graph(x: &Tensor) -> bool {
    if !x.track_op() {
        return false;
    }
    !leaf_gradient_loss_was_explicitly_allowed()
}

pub(crate) fn leaf_gradient_loss_was_explicitly_allowed() -> bool {
    std::env::var("NV_ALLOW_LEAF_GRADIENT_LOSS").ok().as_deref() == Some("1")
}

fn a_quantized_fast_path_cannot_be_differentiated(path: &str, x: &Tensor) -> Result<()> {
    if this_call_is_building_an_autograd_graph(x) {
        anyhow::bail!(
            "{path}: this quantized CUDA fast path has no backward and no dense weight to \
             fall back on, so its output would be a graph leaf and a training step through \
             it would contribute no gradient at all, silently -- including to every adapter \
             in an earlier layer. Train against bf16 weights, or give this path a backward."
        );
    }
    Ok(())
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            anyhow::bail!("Linear weight must be 2-D, got rank {}", dims.len());
        }
        let out_features = dims[0];
        let in_features = dims[1];
        let weight_t = Some(weight.t()?.contiguous()?);
        Ok(Self {
            kind: LinearKind::Bf16,
            storage: LinearStorage::Bf16 { weight, weight_t },
            bias,
            in_features,
            out_features,
            lora: std::sync::RwLock::new(None),
        })
    }

    pub fn new_no_pretranspose(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            anyhow::bail!("Linear weight must be 2-D, got rank {}", dims.len());
        }
        let out_features = dims[0];
        let in_features = dims[1];
        Ok(Self {
            kind: LinearKind::Bf16,
            storage: LinearStorage::Bf16 {
                weight,
                weight_t: None,
            },
            bias,
            in_features,
            out_features,
            lora: std::sync::RwLock::new(None),
        })
    }

    pub fn from_candle_vb(
        vb: VarBuilder,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Self> {
        let vb_bf = vb.clone().to_dtype(DType::BF16);
        let weight = vb_bf.get((out_features, in_features), "weight")?;
        let bias = if bias {
            Some(vb_bf.get(out_features, "bias")?)
        } else {
            None
        };
        Self::new(weight, bias)
    }

    pub fn kind(&self) -> &LinearKind {
        &self.kind
    }

    #[cfg(feature = "cuda")]
    pub fn nvfp4_parts(
        &self,
    ) -> Option<(
        &cudarc::driver::CudaSlice<u8>,
        &cudarc::driver::CudaSlice<u8>,
        f32,
        f32,
    )> {
        match &self.storage {
            LinearStorage::Nvfp4(s) => Some((
                &s.weight_u8,
                &s.weight_scales_cm,
                s.weight_alpha,
                s.input_stored_global,
            )),
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::type_complexity)]
    pub fn nvfp4_parts_full(
        &self,
    ) -> Option<(
        &cudarc::driver::CudaSlice<u8>,
        &cudarc::driver::CudaSlice<u8>,
        &cudarc::driver::CudaSlice<f32>,
        f32,
        f32,
    )> {
        match &self.storage {
            LinearStorage::Nvfp4(s) => Some((
                &s.weight_u8,
                &s.weight_scales_cm,
                &s.weight_alpha_dev,
                s.weight_alpha,
                s.input_stored_global,
            )),
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn fp8_scale_parts(&self) -> Option<(&[f32], nv_quant::fp8::Fp8ScaleMode)> {
        match &self.storage {
            LinearStorage::Fp8E4m3(s) => Some((&s.weight_scale_rows, s.scale_mode)),
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn fp8_e4m3_row_weight_and_scales_so_gdn_prenorm_folds_into_gemv_e4m3_mk_h(
        &self,
    ) -> Option<(
        &cudarc::driver::CudaSlice<u8>,
        &cudarc::driver::CudaSlice<f32>,
    )> {
        match &self.storage {
            LinearStorage::Fp8E4m3(s)
                if s.scale_mode == nv_quant::fp8::Fp8ScaleMode::PerOuterRow
                    && self.bias.is_none()
                    && self.in_features % 16 == 0 =>
            {
                Some((&s.weight_u8, &s.b_scale_rows_dev))
            }
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn nvfp4_runner(
        &self,
    ) -> Option<std::sync::Arc<std::sync::Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>> {
        match &self.storage {
            LinearStorage::Nvfp4(s) => Some(s.runner.clone()),
            _ => None,
        }
    }

    pub fn weight(&self) -> Option<&Tensor> {
        #[cfg(feature = "cuda")]
        {
            match &self.storage {
                LinearStorage::Bf16 { weight, .. } => Some(weight),
                LinearStorage::Fp8E4m3(_) => None,
                LinearStorage::Nvfp4(_) => None,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            match &self.storage {
                LinearStorage::Bf16 { weight, .. } => Some(weight),
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn dequant_weight(&self) -> Result<Option<Tensor>> {
        use half::bf16;
        match &self.storage {
            LinearStorage::Bf16 { weight, .. } => Ok(Some(weight.clone())),
            LinearStorage::Fp8E4m3(_) => Ok(None),
            LinearStorage::Nvfp4(s) => {
                let stream = crate::cuda_stream::current_stream(&s.cuda_device);
                #[allow(deprecated)]
                let packed = stream
                    .memcpy_dtov(&s.weight_u8)
                    .map_err(|e| anyhow::anyhow!("dtoh nvfp4 packed: {e}"))?;
                #[allow(deprecated)]
                let scales_sw = stream
                    .memcpy_dtov(&s.weight_scales_cm)
                    .map_err(|e| anyhow::anyhow!("dtoh nvfp4 scales: {e}"))?;
                let vals = nv_quant::nvfp4::dequantize_packed_swizzled(
                    &packed,
                    &scales_sw,
                    self.out_features,
                    self.in_features,
                    s.weight_alpha,
                );
                let bf: Vec<bf16> = vals.iter().map(|&v| bf16::from_f32(v)).collect();
                let t = Tensor::from_vec(
                    bf,
                    (self.out_features, self.in_features),
                    &Device::Cuda(s.cuda_device.clone()),
                )?;
                Ok(Some(t))
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn dequant_weight(&self) -> Result<Option<Tensor>> {
        Ok(self.weight().cloned())
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn attach_lora(&self, hook: Arc<dyn LoraDeltaHook>) -> Result<()> {
        if hook.in_features() != self.in_features {
            anyhow::bail!(
                "attach_lora: hook in_features {} != linear in_features {}",
                hook.in_features(),
                self.in_features
            );
        }
        if hook.out_features() != self.out_features {
            anyhow::bail!(
                "attach_lora: hook out_features {} != linear out_features {}",
                hook.out_features(),
                self.out_features
            );
        }
        let mut slot = self
            .lora
            .write()
            .map_err(|e| anyhow::anyhow!("lora hook lock poisoned: {e}"))?;
        *slot = Some(hook);
        Ok(())
    }

    pub fn detach_lora(&self) {
        if let Ok(mut slot) = self.lora.write() {
            *slot = None;
        }
    }

    pub fn has_lora(&self) -> bool {
        self.lora.read().map(|s| s.is_some()).unwrap_or(false)
    }

    fn lora_delta(&self, x2: &Tensor, y2: Tensor, win: Option<(usize, usize)>) -> Result<Tensor> {
        let hook = {
            let slot = self
                .lora
                .read()
                .map_err(|e| anyhow::anyhow!("lora hook lock poisoned: {e}"))?;
            match &*slot {
                Some(h) => h.clone(),
                None => return Ok(y2),
            }
        };
        match hook.apply(x2, &y2, win)? {
            Some(replaced) => Ok(replaced),
            None => Ok(y2),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        if *dims.last().unwrap() != self.in_features {
            anyhow::bail!(
                "Linear: input last dim {} != in_features {}",
                dims.last().unwrap(),
                self.in_features
            );
        }
        self.forward_inner(x, false)
    }

    pub fn forward_dense(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_inner(x, true)
    }

    fn forward_inner(&self, x: &Tensor, force_dense: bool) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        if *dims.last().unwrap() != self.in_features {
            anyhow::bail!(
                "Linear: input last dim {} != in_features {}",
                dims.last().unwrap(),
                self.in_features
            );
        }
        let leading: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((leading, self.in_features))?;

        let out2 = self.matmul_impl(&x2, leading, force_dense)?;
        let out2 = self.lora_delta(&x2, out2, None)?;

        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(self.out_features);
        let mut out = out2.reshape(out_dims)?;
        if let Some(b) = &self.bias {
            out = out.broadcast_add(&b.to_dtype(out.dtype())?)?;
        }
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    pub fn prenorm_nvfp4_eligible(&self) -> bool {
        matches!(self.storage, LinearStorage::Nvfp4(_)) && self.bias.is_none() && !self.has_lora()
    }

    #[cfg(feature = "cuda")]
    pub fn forward_prenorm_nvfp4(
        &self,
        x_pre: &Tensor,
        norm_weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        if !self.prenorm_nvfp4_eligible() {
            anyhow::bail!("forward_prenorm_nvfp4: linear is not an eligible NVFP4 projection");
        }
        let dims = x_pre.dims().to_vec();
        if *dims.last().unwrap() != self.in_features {
            anyhow::bail!(
                "forward_prenorm_nvfp4: input last dim {} != in_features {}",
                dims.last().unwrap(),
                self.in_features
            );
        }
        let leading: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x_pre.reshape((leading, self.in_features))?;
        let s = match &self.storage {
            LinearStorage::Nvfp4(s) => s,
            _ => unreachable!("checked eligible above"),
        };
        let out2 = self.matmul_nvfp4_impl(
            s,
            &x2,
            leading,
            Some(FusedPreNorm {
                weight_bf16: norm_weight,
                eps,
            }),
        )?;
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(self.out_features);
        Ok(out2.reshape(out_dims)?)
    }

    fn matmul_impl(&self, x2: &Tensor, leading: usize, force_dense: bool) -> Result<Tensor> {
        match &self.storage {
            LinearStorage::Bf16 { weight, weight_t } => {
                self.matmul_bf16(weight, weight_t.as_ref(), x2, leading, force_dense)
            }
            #[cfg(feature = "cuda")]
            LinearStorage::Fp8E4m3(s) => self.matmul_fp8(s, x2, leading),
            #[cfg(feature = "cuda")]
            LinearStorage::Nvfp4(s) => self.matmul_nvfp4(s, x2, leading),
        }
    }

    pub fn ensure_pretransposed(&mut self) -> Result<bool> {
        match &mut self.storage {
            LinearStorage::Bf16 { weight, weight_t } => {
                if weight_t.is_none() {
                    *weight_t = Some(weight.t()?.contiguous()?);
                }
                Ok(true)
            }
            #[allow(unreachable_patterns)]
            _ => Ok(false),
        }
    }

    #[cfg(feature = "cuda")]
    fn matmul_bf16(
        &self,
        weight: &Tensor,
        weight_t: Option<&Tensor>,
        x2: &Tensor,
        leading: usize,
        force_dense: bool,
    ) -> Result<Tensor> {
        use half::bf16;
        use nv_quant::matmul::TensorCoreGemm;

        if !matches!(x2.device(), Device::Cuda(_)) || x2.dtype() != DType::BF16 {
            return self.matmul_fallback(weight, x2);
        }
        let x_bf = x2.contiguous()?;
        let dev = match x_bf.device() {
            Device::Cuda(d) => d.clone(),
            _ => return self.matmul_fallback(weight, &x_bf),
        };
        let stream = crate::cuda_stream::current_stream(&dev);

        if leading == 1
            && !force_dense
            && !force_dense_bf16_enabled()
            && self.in_features % 2 == 0
            && std::env::var("NV_LINEAR_BF16_FORCE_CUBLAS").is_err()
        {
            return self.matmul_bf16_gemv(weight, &x_bf, &dev);
        }

        let w_bf = match weight_t {
            Some(wt) => wt.contiguous()?,
            None => weight.contiguous()?,
        };
        let m = leading as u64;
        let n = self.out_features as u64;
        let k = self.in_features as u64;

        let mut c_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(leading * self.out_features)
                .map_err(|e| anyhow::anyhow!(e))?
        };

        let gemm = TensorCoreGemm::new(stream.clone())?;
        {
            let (x_storage, xl) = x_bf.storage_and_layout();
            let (w_storage, wl) = w_bf.storage_and_layout();
            let x_cuda = match &*x_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for input"),
            };
            let w_cuda = match &*w_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for weight"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
            let x_off = xl.start_offset();
            let w_off = wl.start_offset();
            if weight_t.is_some() {
                gemm.bf16_matmul_row_major_offs(
                    &stream, x_slice, x_off, w_slice, w_off, &mut c_dev, m, n, k, 1.0, 0.0,
                )?;
            } else if (2..=16).contains(&leading) && small_m_det_enabled() {
                gemm.bf16_matmul_row_major_bt_det_offs(
                    &stream, x_slice, x_off, w_slice, w_off, &mut c_dev, m, n, k, 1.0, 0.0,
                )?;
            } else {
                gemm.bf16_matmul_row_major_bt_offs(
                    &stream, x_slice, x_off, w_slice, w_off, &mut c_dev, m, n, k, 1.0, 0.0,
                )?;
            }
        }

        let out_storage = candle_core::CudaStorage::wrap_cuda_slice(c_dev, dev);
        let storage = candle_core::Storage::Cuda(out_storage);
        if this_call_is_building_an_autograd_graph(x2) {
            return self.matmul_fallback(weight, x2);
        }
        let out = candle_core::Tensor::from_storage(
            storage,
            (leading, self.out_features),
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn matmul_bf16_gemv(
        &self,
        weight: &Tensor,
        x_bf: &Tensor,
        dev: &candle_core::CudaDevice,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_kernels::cuda as nvk;

        let w_bf = weight.contiguous()?;
        let stream = crate::cuda_stream::current_stream(&dev);
        let n = self.out_features;
        let k = self.in_features;

        let mut c_dev: cudarc::driver::CudaSlice<bf16> =
            unsafe { stream.alloc::<bf16>(n).map_err(|e| anyhow::anyhow!(e))? };

        let rc = {
            let (x_storage, xl) = x_bf.storage_and_layout();
            let (w_storage, wl) = w_bf.storage_and_layout();
            let x_cuda = match &*x_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for input"),
            };
            let w_cuda = match &*w_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for weight"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let x_view = x_slice.slice(xl.start_offset()..);
            let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
            let w_view = w_slice.slice(wl.start_offset()..);
            let (xp, _) = x_view.device_ptr(&stream);
            let (wp, _) = w_view.device_ptr(&stream);
            let (yp, _) = c_dev.device_ptr_mut(&stream);
            unsafe {
                nvk::gemv_bf16(
                    stream.cu_stream() as *mut _,
                    wp as *const u16,
                    xp as *const u16,
                    yp as *mut u16,
                    n as i32,
                    k as i32,
                )
            }
        };
        if rc != 0 {
            anyhow::bail!("gemv_bf16 kernel returned {rc}");
        }

        let storage = candle_core::CudaStorage::wrap_cuda_slice(c_dev, dev.clone());
        let storage = candle_core::Storage::Cuda(storage);
        if this_call_is_building_an_autograd_graph(x_bf) {
            return self.matmul_fallback(weight, x_bf);
        }
        Ok(candle_core::Tensor::from_storage(
            storage,
            (1usize, self.out_features),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    #[cfg(not(feature = "cuda"))]
    fn matmul_bf16(
        &self,
        weight: &Tensor,
        _weight_t: Option<&Tensor>,
        x2: &Tensor,
        _leading: usize,
        _force_dense: bool,
    ) -> Result<Tensor> {
        self.matmul_fallback(weight, x2)
    }

    pub fn forward_rows(&self, x: &Tensor, row_off: usize, rows: usize) -> Result<Tensor> {
        let weight = match &self.storage {
            LinearStorage::Bf16 { weight, .. } => weight,
            #[cfg(feature = "cuda")]
            _ => anyhow::bail!("forward_rows requires bf16 storage"),
        };
        if self.bias.is_some() {
            anyhow::bail!("forward_rows does not support bias");
        }
        if row_off + rows > self.out_features {
            anyhow::bail!(
                "forward_rows: rows {row_off}..{} out of range (out_features {})",
                row_off + rows,
                self.out_features
            );
        }
        let dims = x.dims().to_vec();
        if *dims.last().unwrap_or(&0) != self.in_features {
            anyhow::bail!(
                "forward_rows: input last dim {:?} != in_features {}",
                dims.last(),
                self.in_features
            );
        }
        let leading: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((leading, self.in_features))?;
        let out2 = self.matmul_bf16_rows(weight, &x2, leading, row_off, rows)?;
        let out2 = self.lora_delta(&x2, out2, Some((row_off, rows)))?;
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(rows);
        Ok(out2.reshape(out_dims)?)
    }

    #[cfg(feature = "cuda")]
    fn matmul_bf16_rows(
        &self,
        weight: &Tensor,
        x2: &Tensor,
        leading: usize,
        row_off: usize,
        rows: usize,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_kernels::cuda as nvk;
        use nv_quant::matmul::TensorCoreGemm;

        if !matches!(x2.device(), Device::Cuda(_)) || x2.dtype() != DType::BF16 {
            return self.matmul_fallback(&weight.narrow(0, row_off, rows)?, x2);
        }
        let x_bf = x2.contiguous()?;
        let dev = match x_bf.device() {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        let k = self.in_features;

        if leading == 1
            && !force_dense_bf16_enabled()
            && k % 2 == 0
            && std::env::var("NV_LINEAR_BF16_FORCE_CUBLAS").is_err()
        {
            let w_bf = weight.contiguous()?;
            let mut c_dev: cudarc::driver::CudaSlice<bf16> =
                unsafe { stream.alloc::<bf16>(rows).map_err(|e| anyhow::anyhow!(e))? };
            let rc = {
                let (x_storage, xl) = x_bf.storage_and_layout();
                let (w_storage, wl) = w_bf.storage_and_layout();
                let x_cuda = match &*x_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("expected cuda storage for input"),
                };
                let w_cuda = match &*w_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("expected cuda storage for weight"),
                };
                let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
                let x_view = x_slice.slice(xl.start_offset()..);
                let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
                let w_view = w_slice.slice(wl.start_offset() + row_off * k..);
                let (xp, _) = x_view.device_ptr(&stream);
                let (wp, _) = w_view.device_ptr(&stream);
                let (yp, _) = c_dev.device_ptr_mut(&stream);
                unsafe {
                    nvk::gemv_bf16(
                        stream.cu_stream() as *mut _,
                        wp as *const u16,
                        xp as *const u16,
                        yp as *mut u16,
                        rows as i32,
                        k as i32,
                    )
                }
            };
            if rc != 0 {
                anyhow::bail!("gemv_bf16 kernel returned {rc}");
            }
            let storage = candle_core::CudaStorage::wrap_cuda_slice(c_dev, dev);
            if this_call_is_building_an_autograd_graph(x2) {
                return self.matmul_fallback(&weight.narrow(0, row_off, rows)?, x2);
            }
            return Ok(candle_core::Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, rows),
                candle_core::op::BackpropOp::none(),
                false,
            ));
        }

        let w_bf = weight.contiguous()?;
        let m = leading as u64;

        let mut c_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(leading * rows)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let gemm = TensorCoreGemm::new(stream.clone())?;
        {
            let (x_storage, xl) = x_bf.storage_and_layout();
            let (w_storage, wl) = w_bf.storage_and_layout();
            let x_cuda = match &*x_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for input"),
            };
            let w_cuda = match &*w_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for weight"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
            gemm.bf16_matmul_row_major_bt_offs(
                &stream,
                x_slice,
                xl.start_offset(),
                w_slice,
                wl.start_offset() + row_off * k,
                &mut c_dev,
                m,
                rows as u64,
                k as u64,
                1.0,
                0.0,
            )?;
        }
        let out_storage = candle_core::CudaStorage::wrap_cuda_slice(c_dev, dev);
        if this_call_is_building_an_autograd_graph(x2) {
            return self.matmul_fallback(&weight.narrow(0, row_off, rows)?, x2);
        }
        Ok(candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(out_storage),
            (leading, rows),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    #[cfg(not(feature = "cuda"))]
    fn matmul_bf16_rows(
        &self,
        weight: &Tensor,
        x2: &Tensor,
        _leading: usize,
        row_off: usize,
        rows: usize,
    ) -> Result<Tensor> {
        self.matmul_fallback(&weight.narrow(0, row_off, rows)?, x2)
    }

    pub fn forward_dense_det(&self, x: &Tensor) -> Result<Tensor> {
        let weight = match &self.storage {
            LinearStorage::Bf16 { weight, .. } => weight,
            #[cfg(feature = "cuda")]
            _ => anyhow::bail!("forward_dense_det requires bf16 storage"),
        };
        if self.bias.is_some() {
            anyhow::bail!("forward_dense_det does not support bias");
        }
        let dims = x.dims().to_vec();
        if *dims.last().unwrap_or(&0) != self.in_features {
            anyhow::bail!(
                "forward_dense_det: input last dim {:?} != in_features {}",
                dims.last(),
                self.in_features
            );
        }
        let leading: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((leading, self.in_features))?;
        let out2 = self.matmul_bf16_det(weight, &x2, leading)?;
        let out2 = self.lora_delta(&x2, out2, None)?;
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(self.out_features);
        Ok(out2.reshape(out_dims)?)
    }

    #[cfg(feature = "cuda")]
    fn matmul_bf16_det(&self, weight: &Tensor, x2: &Tensor, leading: usize) -> Result<Tensor> {
        use half::bf16;
        use nv_quant::matmul::TensorCoreGemm;

        if !matches!(x2.device(), Device::Cuda(_)) || x2.dtype() != DType::BF16 {
            return self.matmul_fallback(weight, x2);
        }
        let x_bf = x2.contiguous()?;
        let dev = match x_bf.device() {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        let w_bf = weight.contiguous()?;
        let m = leading as u64;
        let n = self.out_features as u64;
        let k = self.in_features as u64;

        let mut c_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(leading * self.out_features)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let gemm = TensorCoreGemm::new(stream.clone())?;
        {
            let (x_storage, xl) = x_bf.storage_and_layout();
            let (w_storage, wl) = w_bf.storage_and_layout();
            let x_cuda = match &*x_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for input"),
            };
            let w_cuda = match &*w_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for weight"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
            gemm.bf16_matmul_row_major_bt_det_offs(
                &stream,
                x_slice,
                xl.start_offset(),
                w_slice,
                wl.start_offset(),
                &mut c_dev,
                m,
                n,
                k,
                1.0,
                0.0,
            )?;
        }
        let out_storage = candle_core::CudaStorage::wrap_cuda_slice(c_dev, dev);
        if this_call_is_building_an_autograd_graph(x2) {
            return self.matmul_fallback(weight, x2);
        }
        Ok(candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(out_storage),
            (leading, self.out_features),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    #[cfg(not(feature = "cuda"))]
    fn matmul_bf16_det(&self, weight: &Tensor, x2: &Tensor, _leading: usize) -> Result<Tensor> {
        self.matmul_fallback(weight, x2)
    }

    fn matmul_fallback(&self, weight: &Tensor, x2: &Tensor) -> Result<Tensor> {
        let on_cpu = matches!(x2.device(), candle_core::Device::Cpu);
        let needs_bf16_upcast =
            on_cpu && (weight.dtype() == DType::BF16 || x2.dtype() == DType::BF16);
        if needs_bf16_upcast {
            let orig_dtype = x2.dtype();
            let x_f32 = x2.to_dtype(DType::F32)?;
            let w_f32 = weight.to_dtype(DType::F32)?;
            let out = x_f32.matmul(&w_f32.t()?.contiguous()?)?;
            return Ok(out.to_dtype(orig_dtype)?);
        }
        let w = weight.to_dtype(x2.dtype())?;
        Ok(x2.matmul(&w.t()?.contiguous()?)?)
    }

    #[cfg(feature = "cuda")]
    fn matmul_fp8(&self, s: &Fp8Storage, x2: &Tensor, leading: usize) -> Result<Tensor> {
        use half::bf16;

        let dev = match x2.device() {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("Fp8 Linear requires CUDA input"),
        };
        if let Some(out) = self.matmul_fp8_small_m_gemv(s, x2, leading, &dev)? {
            return Ok(out);
        }
        if s.scale_mode == nv_quant::fp8::Fp8ScaleMode::PerOuterRow {
            return self.matmul_fp8_device_rowquant_pertensor_raw_gemm_rowcol_epilogue(
                s, x2, leading, &dev,
            );
        }
        let _ = &s.cuda_device;
        let stream = crate::cuda_stream::current_stream(&dev);
        let m = leading as u64;
        let n = self.out_features as u64;
        let k = self.in_features as u64;

        let x_bf = x2.to_dtype(DType::BF16)?.contiguous()?;
        let x_host: Vec<bf16> = x_bf.flatten_all()?.to_vec1::<bf16>()?;

        let (x_fp8_bytes, x_row_scales) = {
            let (bytes, scale) = nv_quant::fp8::quantize_e4m3_per_tensor(&x_host);
            (bytes, vec![scale])
        };
        let scale_factor = match self.kind {
            LinearKind::Fp8E4m3 { a_scale, .. } => a_scale,
            _ => 1.0,
        };
        let effective_a_scales: Vec<f32> = x_row_scales.iter().map(|v| v * scale_factor).collect();
        #[allow(deprecated)]
        let x_fp8_dev = stream
            .clone_htod(&x_fp8_bytes)
            .map_err(|e| anyhow::anyhow!(e))?;
        #[allow(deprecated)]
        let a_scale_dev_call = stream
            .clone_htod(&effective_a_scales)
            .map_err(|e| anyhow::anyhow!(e))?;
        let mut d_dev: cudarc::driver::CudaSlice<bf16> = stream
            .alloc_zeros::<bf16>(leading * self.out_features)
            .map_err(|e| anyhow::anyhow!(e))?;

        {
            let mut runner = s
                .runner
                .lock()
                .map_err(|e| anyhow::anyhow!("Fp8 runner mutex poisoned: {e}"))?;
            runner.matmul_e4m3_weight_row(
                &x_fp8_dev,
                &s.weight_u8,
                &mut d_dev,
                m,
                n,
                k,
                &a_scale_dev_call,
                &s.b_scale_dev,
            )?
        }

        let out_storage = candle_core::CudaStorage::wrap_cuda_slice(d_dev, dev);
        let storage = candle_core::Storage::Cuda(out_storage);
        a_quantized_fast_path_cannot_be_differentiated("matmul_fp8", x2)?;
        let out = candle_core::Tensor::from_storage(
            storage,
            (leading, self.out_features),
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn matmul_fp8_small_m_gemv(
        &self,
        s: &Fp8Storage,
        x2: &Tensor,
        leading: usize,
        dev: &candle_core::CudaDevice,
    ) -> Result<Option<Tensor>> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_kernels::cuda as nvk;

        if s.scale_mode != nv_quant::fp8::Fp8ScaleMode::PerOuterRow
            || !(1..=16).contains(&leading)
            || self.in_features % 16 != 0
            || x2.dtype() != DType::BF16
        {
            return Ok(None);
        }
        if leading >= 2 && verify_tc_fp8_lt_gemm_scope_active() {
            return Ok(None);
        }
        a_quantized_fast_path_cannot_be_differentiated("matmul_fp8_small_m_gemv", x2)?;
        let x_bf = x2.contiguous()?;
        let stream = crate::cuda_stream::current_stream(dev);
        let n = self.out_features;
        let k = self.in_features;
        let mut c_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(leading * n)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let rc = {
            let (x_storage, xl) = x_bf.storage_and_layout();
            let x_cuda = match &*x_storage {
                candle_core::Storage::Cuda(st) => st,
                _ => anyhow::bail!("expected cuda storage for fp8 gemv input"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let x_view = x_slice.slice(xl.start_offset()..);
            let (xp, _gx) = x_view.device_ptr(&stream);
            let (wp, _gw) = s.weight_u8.device_ptr(&stream);
            let (sp, _gs) = s.b_scale_rows_dev.device_ptr(&stream);
            let (yp, _gy) = c_dev.device_ptr_mut(&stream);
            unsafe {
                nvk::gemv_e4m3_mk(
                    stream.cu_stream() as *mut _,
                    wp as *const u8,
                    sp as *const f32,
                    xp as *const u16,
                    yp as *mut u16,
                    n as i32,
                    k as i32,
                    leading as i32,
                )
            }
        };
        if rc == -1 {
            return Ok(None);
        }
        if rc != 0 {
            anyhow::bail!("gemv_e4m3_mk kernel returned {rc}");
        }
        let storage = candle_core::CudaStorage::wrap_cuda_slice(c_dev, dev.clone());
        let storage = candle_core::Storage::Cuda(storage);
        Ok(Some(candle_core::Tensor::from_storage(
            storage,
            (leading, n),
            candle_core::op::BackpropOp::none(),
            false,
        )))
    }

    #[cfg(feature = "cuda")]
    fn matmul_fp8_device_rowquant_pertensor_raw_gemm_rowcol_epilogue(
        &self,
        s: &Fp8Storage,
        x2: &Tensor,
        leading: usize,
        dev: &candle_core::CudaDevice,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_kernels::cuda as nvk;

        let m = leading;
        let n = self.out_features;
        let k = self.in_features;
        anyhow::ensure!(
            k % 16 == 0,
            "fp8 row-scaled linear: in_features {k} % 16 != 0; cuBLASLt e4m3 needs 16-element \
             aligned K and every q38 fp8 module satisfies it, so a trip here is a new module"
        );
        a_quantized_fast_path_cannot_be_differentiated(
            "matmul_fp8_device_rowquant_pertensor_raw_gemm_rowcol_epilogue",
            x2,
        )?;
        let stream = crate::cuda_stream::current_stream(dev);
        let x_bf = x2.to_dtype(DType::BF16)?.contiguous()?;

        let mut x_q: cudarc::driver::CudaSlice<u8> = unsafe {
            stream.alloc::<u8>(m * k).map_err(|e| anyhow::anyhow!(e))?
        };
        let mut a_scale_rows: cudarc::driver::CudaSlice<f32> = unsafe {
            stream.alloc::<f32>(m).map_err(|e| anyhow::anyhow!(e))?
        };
        {
            let (x_storage, xl) = x_bf.storage_and_layout();
            let x_cuda = match &*x_storage {
                candle_core::Storage::Cuda(st) => st,
                _ => anyhow::bail!("expected cuda storage for fp8 rowquant input"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let x_view = x_slice.slice(xl.start_offset()..);
            let (xp, _gx) = x_view.device_ptr(&stream);
            let (qp, _gq) = x_q.device_ptr_mut(&stream);
            let (sp, _gs) = a_scale_rows.device_ptr_mut(&stream);
            let rc = unsafe {
                nvk::rowquant_e4m3(
                    stream.cu_stream() as *mut _,
                    xp as *const u16,
                    qp as *mut u8,
                    sp as *mut f32,
                    m as i32,
                    k as i32,
                )
            };
            anyhow::ensure!(rc == 0, "rowquant_e4m3 rc={rc}");
        }

        let mut d_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(m * n)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        {
            let mut runner = s
                .runner
                .lock()
                .map_err(|e| anyhow::anyhow!("Fp8 runner mutex poisoned: {e}"))?;
            runner.matmul_e4m3_weight_row(
                &x_q,
                &s.weight_u8,
                &mut d_dev,
                m as u64,
                n as u64,
                k as u64,
                &s.a_scale_dev,
                &s.a_scale_dev,
            )?
        }
        {
            let (dp, _gd) = d_dev.device_ptr_mut(&stream);
            let (rp, _gr) = a_scale_rows.device_ptr(&stream);
            let (cp, _gc) = s.b_scale_rows_dev.device_ptr(&stream);
            let rc = unsafe {
                nvk::scale_rowcol_bf16(
                    stream.cu_stream() as *mut _,
                    dp as *mut u16,
                    rp as *const f32,
                    cp as *const f32,
                    m as i32,
                    n as i32,
                )
            };
            anyhow::ensure!(rc == 0, "scale_rowcol_bf16 rc={rc}");
        }
        let storage = candle_core::CudaStorage::wrap_cuda_slice(d_dev, dev.clone());
        Ok(candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (m, n),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    #[cfg(feature = "cuda")]
    fn matmul_nvfp4(&self, s: &Nvfp4Storage, x2: &Tensor, leading: usize) -> Result<Tensor> {
        self.matmul_nvfp4_impl(s, x2, leading, None)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_nvfp4_gemv_m1_unconditionally_which_a_draft_only_lm_head_requires(
        &self,
        x: &Tensor,
    ) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let leading: usize = dims[..dims.len().saturating_sub(1)].iter().product();
        anyhow::ensure!(
            leading == 1,
            "forward_nvfp4_gemv_m1: draft lm_head serves one row per step, got leading {leading}"
        );
        anyhow::ensure!(
            dims.last() == Some(&self.in_features),
            "forward_nvfp4_gemv_m1: input last dim {:?} != in_features {}",
            dims.last(),
            self.in_features
        );
        let x2 = x.reshape((1usize, self.in_features))?;
        match &self.storage {
            LinearStorage::Nvfp4(s) => self.matmul_nvfp4_m1_gemv(s, &x2, None),
            _ => anyhow::bail!("forward_nvfp4_gemv_m1: linear is not NVFP4-resident"),
        }
    }

    #[cfg(feature = "cuda")]
    fn matmul_nvfp4_m1_gemv(
        &self,
        s: &Nvfp4Storage,
        x2: &Tensor,
        prenorm: Option<&FusedPreNorm<'_>>,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use std::ffi::c_void;

        let dev = match x2.device() {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("NVFP4 Linear requires CUDA input"),
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        let x_bf = x2.to_dtype(DType::BF16)?.contiguous()?;
        let mut d: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(self.out_features)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let mut xn: Option<cudarc::driver::CudaSlice<bf16>> = None;
        let rc = {
            let (x_storage, xl) = x_bf.storage_and_layout();
            let x_cuda = match &*x_storage {
                candle_core::Storage::Cuda(cs) => cs,
                _ => anyhow::bail!("expected cuda storage for input"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let x_view = x_slice.slice(xl.start_offset()..);
            let (x_ptr, _gx) = x_view.device_ptr(&stream);
            let gemv_x_ptr = if let Some(pn) = prenorm {
                let mut buf: cudarc::driver::CudaSlice<bf16> = unsafe {
                    stream
                        .alloc::<bf16>(self.in_features)
                        .map_err(|e| anyhow::anyhow!(e))?
                };
                let w_c = pn.weight_bf16.clone();
                let (w_storage, wl) = w_c.storage_and_layout();
                let w_cuda = match &*w_storage {
                    candle_core::Storage::Cuda(cs) => cs,
                    _ => anyhow::bail!("expected cuda storage for norm weight"),
                };
                let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
                let w_view = w_slice.slice(wl.start_offset()..);
                let (nw_ptr, _gn) = w_view.device_ptr(&stream);
                let p = {
                    let (xn_ptr, _gxn) = buf.device_ptr_mut(&stream);
                    let rc_norm = unsafe {
                        nv_kernels::cuda::rmsnorm_bf16(
                            stream.cu_stream() as *mut c_void,
                            x_ptr as *const u16,
                            nw_ptr as *const u16,
                            xn_ptr as *mut u16,
                            1,
                            self.in_features,
                            pn.eps,
                        )
                    };
                    if rc_norm != 0 {
                        anyhow::bail!("rmsnorm_bf16 rc={rc_norm}");
                    }
                    xn_ptr
                };
                xn = Some(buf);
                p
            } else {
                x_ptr
            };
            let (w_ptr, _gw) = s.weight_u8.device_ptr(&stream);
            let (s_ptr, _gs) = s.weight_scales_cm.device_ptr(&stream);
            let (d_ptr, _gd) = d.device_ptr_mut(&stream);
            unsafe {
                nv_kernels::cuda::nvfp4_gemv_bf16act(
                    stream.cu_stream() as *mut c_void,
                    w_ptr as *const u8,
                    s_ptr as *const u8,
                    gemv_x_ptr as *const u16,
                    d_ptr as *mut u16,
                    s.weight_alpha,
                    self.out_features as i32,
                    self.in_features as i32,
                )
            }
        };
        drop(xn);
        if rc != 0 {
            anyhow::bail!("nvfp4_gemv_bf16act rc={rc}");
        }
        let d_storage = candle_core::CudaStorage::wrap_cuda_slice(d, dev);
        let storage = candle_core::Storage::Cuda(d_storage);
        a_quantized_fast_path_cannot_be_differentiated("matmul_nvfp4_m1_gemv", x2)?;
        Ok(candle_core::Tensor::from_storage(
            storage,
            (1, self.out_features),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    #[cfg(feature = "cuda")]
    fn matmul_nvfp4_impl(
        &self,
        s: &Nvfp4Storage,
        x2: &Tensor,
        leading: usize,
        prenorm: Option<FusedPreNorm<'_>>,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_quant::nvfp4::{BLOCK_SIZE, MIN_TILE};
        use std::ffi::c_void;

        let dev = match x2.device() {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("NVFP4 Linear requires CUDA input"),
        };
        let _ = &s.cuda_device;
        let stream = crate::cuda_stream::current_stream(&dev);
        let n = self.out_features as u64;
        let k = self.in_features as u64;
        if (self.in_features % BLOCK_SIZE) != 0 {
            anyhow::bail!(
                "NVFP4 Linear: in_features {} must be a multiple of {}",
                self.in_features,
                BLOCK_SIZE
            );
        }
        if self.out_features < MIN_TILE || self.in_features < MIN_TILE {
            anyhow::bail!(
                "NVFP4 Linear: out_features and in_features must each be >= {} (got {} and {})",
                MIN_TILE,
                self.out_features,
                self.in_features
            );
        }

        let m_logical = leading;
        if m_logical == 0 {
            anyhow::bail!("NVFP4 Linear: empty input batch");
        }

        if m_logical == 1 && nvfp4_m1_gemv_opted_in() {
            return self.matmul_nvfp4_m1_gemv(s, x2, prenorm.as_ref());
        }

        let true_m = nvfp4_true_m_enabled() && {
            let runner = s
                .runner
                .lock()
                .map_err(|e| anyhow::anyhow!("NVFP4 runner mutex poisoned: {e}"))?;
            runner.supports_true_m(
                m_logical as u64,
                self.out_features as u64,
                self.in_features as u64,
            )
        };
        let m_padded = if true_m {
            m_logical
        } else {
            m_logical.max(MIN_TILE)
        };

        let x_bf = x2.to_dtype(DType::BF16)?.contiguous()?;

        let blocks_per_row = self.in_features / BLOCK_SIZE;
        let packed_bytes = m_padded * self.in_features / 2;
        let scales_bytes = ((m_padded + 127) / 128) * 128 * ((blocks_per_row + 3) / 4) * 4;

        let mut d_padded: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(m_padded * self.out_features)
                .map_err(|e| anyhow::anyhow!(e))?
        };

        let skip_pad = m_padded > m_logical && !nvfp4_quant_fullpad_enabled();
        let key = (nv_quant::stream_cache_key(&stream), m_padded);
        let mut staged_map = if skip_pad {
            let mut map = s
                .a_staging
                .lock()
                .map_err(|e| anyhow::anyhow!("NVFP4 staging mutex poisoned: {e}"))?;
            if !stream_is_capturing(&stream) {
                map.retain(|k, v| v.epoch == nv_quant::stream_epoch(k.0));
                if !map.contains_key(&key) {
                    let a = stream
                        .alloc_zeros::<u8>(packed_bytes)
                        .map_err(|e| anyhow::anyhow!(e))?;
                    let sc = stream
                        .alloc_zeros::<u8>(scales_bytes)
                        .map_err(|e| anyhow::anyhow!(e))?;
                    map.insert(
                        key,
                        StagedActivations {
                            epoch: nv_quant::stream_epoch(key.0),
                            hwm_rows: 0,
                            packed: a,
                            scales: sc,
                        },
                    );
                }
            }
            if map.contains_key(&key) {
                if let Some(entry) = map.get_mut(&key) {
                    if m_logical < entry.hwm_rows {
                        stream
                            .memset_zeros(&mut entry.packed)
                            .map_err(|e| anyhow::anyhow!(e))?;
                        stream
                            .memset_zeros(&mut entry.scales)
                            .map_err(|e| anyhow::anyhow!(e))?;
                    }
                    entry.hwm_rows = m_logical;
                }
                Some(map)
            } else {
                None
            }
        } else {
            None
        };

        let mut a_local: Option<(cudarc::driver::CudaSlice<u8>, cudarc::driver::CudaSlice<u8>)> =
            None;
        if staged_map.is_none() {
            let a = unsafe {
                stream
                    .alloc::<u8>(packed_bytes)
                    .map_err(|e| anyhow::anyhow!(e))?
            };
            let sc = unsafe {
                stream
                    .alloc::<u8>(scales_bytes)
                    .map_err(|e| anyhow::anyhow!(e))?
            };
            a_local = Some((a, sc));
        }

        {
            let rows_only = staged_map.is_some();
            let (a_data, a_scales) = match staged_map.as_mut() {
                Some(map) => {
                    let entry = map.get_mut(&key).expect("inserted above");
                    (&mut entry.packed, &mut entry.scales)
                }
                None => {
                    let (a, sc) = a_local.as_mut().expect("allocated above");
                    (a, sc)
                }
            };
            {
                let (x_storage, xl) = x_bf.storage_and_layout();
                let x_cuda = match &*x_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("expected cuda storage for input"),
                };
                let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
                let x_view = x_slice.slice(xl.start_offset()..);
                let (x_ptr, _gx) = x_view.device_ptr(&stream);
                let (a_ptr, _ga) = a_data.device_ptr_mut(&stream);
                let (s_ptr, _gs) = a_scales.device_ptr_mut(&stream);
                let rc = if let Some(pn) = prenorm.as_ref() {
                    let w_c = pn.weight_bf16.clone();
                    let (w_storage, wl) = w_c.storage_and_layout();
                    let w_cuda = match &*w_storage {
                        candle_core::Storage::Cuda(s) => s,
                        _ => anyhow::bail!("expected cuda storage for norm weight"),
                    };
                    let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
                    let w_view = w_slice.slice(wl.start_offset()..);
                    let (w_ptr, _gw) = w_view.device_ptr(&stream);
                    let m_padded_arg = if rows_only { m_logical } else { m_padded };
                    unsafe {
                        nv_kernels::cuda::rmsnorm_quantize_nvfp4_bf16(
                            stream.cu_stream() as *mut c_void,
                            x_ptr as *const u16,
                            w_ptr as *const u16,
                            a_ptr as *mut u8,
                            s_ptr as *mut u8,
                            s.input_stored_global,
                            pn.eps,
                            m_padded_arg as i32,
                            m_logical as i32,
                            self.in_features as i32,
                        )
                    }
                } else if rows_only {
                    unsafe {
                        nv_kernels::cuda::quantize_nvfp4_bf16_rows(
                            stream.cu_stream() as *mut c_void,
                            x_ptr as *const u16,
                            a_ptr as *mut u8,
                            s_ptr as *mut u8,
                            s.input_stored_global,
                            m_logical as i32,
                            self.in_features as i32,
                        )
                    }
                } else {
                    unsafe {
                        nv_kernels::cuda::quantize_nvfp4_bf16(
                            stream.cu_stream() as *mut c_void,
                            x_ptr as *const u16,
                            a_ptr as *mut u8,
                            s_ptr as *mut u8,
                            s.input_stored_global,
                            m_padded as i32,
                            m_logical as i32,
                            self.in_features as i32,
                        )
                    }
                };
                if rc != 0 {
                    anyhow::bail!("quantize_nvfp4_bf16 rc={rc}");
                }
            }

            let mut runner = s
                .runner
                .lock()
                .map_err(|e| anyhow::anyhow!("NVFP4 runner mutex poisoned: {e}"))?;

            runner.set_stream(stream.clone());
            runner.matmul_scaled_alpha_dev(
                a_data,
                a_scales,
                &s.weight_u8,
                &s.weight_scales_cm,
                &mut d_padded,
                m_padded as u64,
                n,
                k,
                &s.weight_alpha_dev,
                s.weight_alpha,
            )?;
        }

        let d_storage = candle_core::CudaStorage::wrap_cuda_slice(d_padded, dev);
        let storage = candle_core::Storage::Cuda(d_storage);
        a_quantized_fast_path_cannot_be_differentiated("matmul_nvfp4_impl", x2)?;
        let full = candle_core::Tensor::from_storage(
            storage,
            (m_padded, self.out_features),
            candle_core::op::BackpropOp::none(),
            false,
        );
        if m_padded == m_logical {
            Ok(full)
        } else {
            Ok(full.narrow(0, 0, m_logical)?.contiguous()?)
        }
    }
}

#[cfg(feature = "cuda")]
impl Linear {
    #[doc(hidden)]
    #[cfg(feature = "cuda")]
    pub fn nvfp4_staging_diag(&self, stream_key: usize, m_padded: usize) -> Option<(u64, usize)> {
        match &self.storage {
            LinearStorage::Nvfp4(s) => s.a_staging.lock().ok().and_then(|m| {
                m.get(&(stream_key, m_padded))
                    .map(|e| (e.epoch, e.hwm_rows))
            }),
            _ => None,
        }
    }

    pub fn new_fp8_e4m3(
        weight_u8: cudarc::driver::CudaSlice<u8>,
        weight_scale: f32,
        in_features: usize,
        out_features: usize,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    ) -> Result<Self> {
        Self::new_fp8_e4m3_row_scales(
            weight_u8,
            vec![weight_scale; out_features],
            in_features,
            out_features,
            bias,
            device,
            runner,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_fp8_e4m3_row_scales(
        weight_u8: cudarc::driver::CudaSlice<u8>,
        weight_scale_rows: Vec<f32>,
        in_features: usize,
        out_features: usize,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    ) -> Result<Self> {
        Self::new_fp8_e4m3_row_scales_in_mode(
            weight_u8,
            weight_scale_rows,
            in_features,
            out_features,
            bias,
            device,
            runner,
            fp8_scale_mode(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_fp8_e4m3_row_scales_in_mode(
        weight_u8: cudarc::driver::CudaSlice<u8>,
        weight_scale_rows: Vec<f32>,
        in_features: usize,
        out_features: usize,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
        scale_mode: nv_quant::fp8::Fp8ScaleMode,
    ) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("Fp8 Linear requires a CUDA device"),
        };
        if scale_mode == nv_quant::fp8::Fp8ScaleMode::PerOuterRow {
            let ordinal = dev.cuda_stream().context().ordinal();
            if let Err(detail) = fp8_per_row_probe(&runner, ordinal) {
                anyhow::bail!(
                    "fp8 per-row (OUTER_VEC_32F) scaling is not served by cuBLASLt on this \
                     device, so the default fp8 scale mode cannot run a single GEMM -- \
                     refusing at load instead of failing on the first request. Probe: \
                     {detail}. Set NV_FP8_SCALE_MODE=tensor to fall back to per-tensor fp8 \
                     scales, or drop the fp8 opt-in (e.g. unset NV_ATTN_PROJ_QUANT)."
                );
            }
        }
        Self::new_fp8_e4m3_row_scales_without_the_cublaslt_probe(
            weight_u8,
            weight_scale_rows,
            in_features,
            out_features,
            bias,
            device,
            runner,
            scale_mode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_fp8_e4m3_row_scales_without_the_cublaslt_probe(
        weight_u8: cudarc::driver::CudaSlice<u8>,
        weight_scale_rows: Vec<f32>,
        in_features: usize,
        out_features: usize,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
        scale_mode: nv_quant::fp8::Fp8ScaleMode,
    ) -> Result<Self> {
        if weight_scale_rows.len() != out_features {
            anyhow::bail!(
                "Fp8 Linear: {} weight scales for {out_features} output rows",
                weight_scale_rows.len()
            );
        }
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("Fp8 Linear requires a CUDA device"),
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        #[allow(deprecated)]
        let a_scale_dev = stream
            .clone_htod(&[1.0f32])
            .map_err(|e| anyhow::anyhow!(e))?;

        let b_scale_scalar = weight_scale_rows.iter().copied().fold(0f32, f32::max);
        let b_scale_scalar = if b_scale_scalar > 0.0 {
            b_scale_scalar
        } else {
            1.0
        };
        #[allow(deprecated)]
        let b_scale_dev = stream
            .clone_htod(&[b_scale_scalar])
            .map_err(|e| anyhow::anyhow!(e))?;
        #[allow(deprecated)]
        let b_scale_rows_dev = stream
            .clone_htod(&weight_scale_rows)
            .map_err(|e| anyhow::anyhow!(e))?;
        let storage = LinearStorage::Fp8E4m3(Fp8Storage {
            weight_u8,
            a_scale_dev,
            b_scale_dev,
            b_scale_rows_dev,
            weight_scale_rows,
            scale_mode,
            cuda_device: dev,
            runner,
        });
        Ok(Self {
            kind: LinearKind::Fp8E4m3 {
                a_scale: 1.0,
                b_scale: b_scale_scalar,
            },
            storage,
            bias,
            in_features,
            out_features,
            lora: std::sync::RwLock::new(None),
        })
    }

    pub fn from_bf16_quantized_fp8(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    ) -> Result<Self> {
        Self::from_bf16_quantized_fp8_with_scales(weight, None, bias, device, runner)
    }

    pub fn from_bf16_quantized_fp8_with_scales(
        weight: &Tensor,
        checkpoint_scale_rows: Option<&[f32]>,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    ) -> Result<Self> {
        Self::from_bf16_quantized_fp8_in_mode(
            weight,
            checkpoint_scale_rows,
            bias,
            device,
            runner,
            fp8_scale_mode(),
        )
    }

    pub fn from_bf16_quantized_fp8_in_mode(
        weight: &Tensor,
        checkpoint_scale_rows: Option<&[f32]>,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
        scale_mode: nv_quant::fp8::Fp8ScaleMode,
    ) -> Result<Self> {
        use half::bf16;

        let dims = weight.dims();
        if dims.len() != 2 {
            anyhow::bail!("Linear weight must be 2-D, got rank {}", dims.len());
        }
        let out_features = dims[0];
        let in_features = dims[1];
        let weight_bf = weight.to_dtype(DType::BF16)?.contiguous()?;
        let weight_host: Vec<bf16> = weight_bf.flatten_all()?.to_vec1::<bf16>()?;
        let (weight_bytes, weight_scale_rows) = fp8_weight_payload(
            &weight_host,
            out_features,
            in_features,
            checkpoint_scale_rows,
            scale_mode,
        )?;
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("Fp8 Linear requires a CUDA device"),
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        #[allow(deprecated)]
        let weight_u8 = stream
            .clone_htod(&weight_bytes)
            .map_err(|e| anyhow::anyhow!(e))?;
        Self::new_fp8_e4m3_row_scales_in_mode(
            weight_u8,
            weight_scale_rows,
            in_features,
            out_features,
            bias,
            device,
            runner,
            scale_mode,
        )
    }

    pub fn from_quantized_weight_fp8(
        qw: &nv_weights::QuantizedWeight,
        dequantized: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    ) -> Result<Self> {
        if qw.scheme != nv_weights::QuantScheme::Fp8E4m3 {
            anyhow::bail!(
                "from_quantized_weight_fp8: checkpoint scheme is {:?}, not Fp8E4m3",
                qw.scheme
            );
        }
        if qw.shape != dequantized.dims() {
            anyhow::bail!(
                "from_quantized_weight_fp8: checkpoint shape {:?} != dequantized {:?}",
                qw.shape,
                dequantized.dims()
            );
        }
        let rows = qw.fp8_weight_scale_rows()?;
        Self::from_bf16_quantized_fp8_with_scales(
            dequantized,
            rows.as_deref(),
            bias,
            device,
            runner,
        )
    }

    pub fn new_nvfp4(
        weight_u8: cudarc::driver::CudaSlice<u8>,
        weight_scales_cm: cudarc::driver::CudaSlice<u8>,
        in_features: usize,
        out_features: usize,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    ) -> Result<Self> {
        Self::new_nvfp4_scaled(
            weight_u8,
            weight_scales_cm,
            in_features,
            out_features,
            bias,
            device,
            runner,
            1.0,
            1.0,
        )
    }

    pub fn new_nvfp4_scaled(
        weight_u8: cudarc::driver::CudaSlice<u8>,
        weight_scales_cm: cudarc::driver::CudaSlice<u8>,
        in_features: usize,
        out_features: usize,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
        weight_alpha: f32,
        input_stored_global: f32,
    ) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("NVFP4 Linear requires a CUDA device"),
        };

        let stream = dev.cuda_stream();
        let mut weight_alpha_dev = stream
            .alloc_zeros::<f32>(1)
            .map_err(|e| anyhow::anyhow!("alloc weight_alpha_dev: {e:?}"))?;
        stream
            .memcpy_htod(&[weight_alpha], &mut weight_alpha_dev)
            .map_err(|e| anyhow::anyhow!("htod weight_alpha_dev: {e:?}"))?;
        let storage = LinearStorage::Nvfp4(Nvfp4Storage {
            weight_u8,
            weight_scales_cm,
            cuda_device: dev,
            runner,
            weight_alpha,
            weight_alpha_dev,
            input_stored_global,
            a_staging: Mutex::new(std::collections::HashMap::new()),
        });
        Ok(Self {
            kind: LinearKind::Nvfp4,
            storage,
            bias,
            in_features,
            out_features,
            lora: std::sync::RwLock::new(None),
        })
    }

    pub fn from_bf16_quantized_nvfp4_dev(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    ) -> Result<Self> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_quant::nvfp4::{BLOCK_SIZE, MIN_TILE};
        use std::ffi::c_void;

        let dims = weight.dims();
        if dims.len() != 2 {
            anyhow::bail!("Linear weight must be 2-D, got rank {}", dims.len());
        }
        let out_features = dims[0];
        let in_features = dims[1];
        if in_features % BLOCK_SIZE != 0 {
            anyhow::bail!(
                "nvfp4 dev quant: in_features {} must be a multiple of {}",
                in_features,
                BLOCK_SIZE
            );
        }
        if out_features < MIN_TILE || in_features < MIN_TILE {
            anyhow::bail!(
                "nvfp4 dev quant: dims [{out_features}, {in_features}] below min tile {MIN_TILE}"
            );
        }
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("NVFP4 Linear requires a CUDA device"),
        };
        let weight_bf = weight
            .to_device(device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let amax = weight_bf
            .abs()?
            .max_all()?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        let stored_weight_global = if amax.is_finite() && amax > 0.0 {
            (448.0f32 * 6.0) / amax
        } else {
            1.0
        };
        let weight_alpha = if stored_weight_global.is_finite() && stored_weight_global != 0.0 {
            1.0 / stored_weight_global
        } else {
            1.0
        };

        let stream = crate::cuda_stream::current_stream(&dev);
        let blocks_per_row = in_features / BLOCK_SIZE;
        let sf_rows = out_features.div_ceil(128) * 128;
        let scales_bytes = sf_rows * blocks_per_row.div_ceil(4) * 4;
        let mut w_packed: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(out_features * in_features / 2)
            .map_err(|e| anyhow::anyhow!(e))?;
        let mut w_scales: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(scales_bytes)
            .map_err(|e| anyhow::anyhow!(e))?;
        {
            let (w_storage, wl) = weight_bf.storage_and_layout();
            let w_cuda = match &*w_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for weight"),
            };
            let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
            let w_view = w_slice.slice(wl.start_offset()..);
            let (wp, _gw) = w_view.device_ptr(&stream);
            let (pp, _gp) = w_packed.device_ptr_mut(&stream);
            let (sp, _gs) = w_scales.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::quantize_nvfp4_bf16(
                    stream.cu_stream() as *mut c_void,
                    wp as *const u16,
                    pp as *mut u8,
                    sp as *mut u8,
                    stored_weight_global,
                    out_features as i32,
                    out_features as i32,
                    in_features as i32,
                )
            };
            if rc != 0 {
                anyhow::bail!("quantize_nvfp4_bf16 (weight) rc={rc}");
            }
        }
        stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
        Self::new_nvfp4_scaled(
            w_packed,
            w_scales,
            in_features,
            out_features,
            bias,
            device,
            runner,
            weight_alpha,
            1.0,
        )
    }

    pub fn from_bf16_quantized_nvfp4(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    ) -> Result<Self> {
        use half::bf16;
        use nv_quant::nvfp4::Nvfp4Tensor;

        let dims = weight.dims();
        if dims.len() != 2 {
            anyhow::bail!("Linear weight must be 2-D, got rank {}", dims.len());
        }
        let out_features = dims[0];
        let in_features = dims[1];
        let weight_bf = weight.to_dtype(DType::BF16)?.contiguous()?;
        let host_bf: Vec<bf16> = weight_bf.flatten_all()?.to_vec1::<bf16>()?;
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(out_features);
        for r in 0..out_features {
            let mut row = Vec::with_capacity(in_features);
            for j in 0..in_features {
                row.push(host_bf[r * in_features + j].to_f32());
            }
            rows.push(row);
        }
        let q = Nvfp4Tensor::quantize_rows(&rows);
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("NVFP4 Linear requires a CUDA device"),
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        #[allow(deprecated)]
        let weight_u8 = stream.clone_htod(&q.data).map_err(|e| anyhow::anyhow!(e))?;
        let weight_scales_sw = q.scales_swizzled();
        #[allow(deprecated)]
        let weight_scales = stream
            .clone_htod(&weight_scales_sw)
            .map_err(|e| anyhow::anyhow!(e))?;
        Self::new_nvfp4(
            weight_u8,
            weight_scales,
            in_features,
            out_features,
            bias,
            device,
            runner,
        )
    }

    pub fn nvfp4_draft_twin_rows_quantized_from_this_resident_copy_which_stays_untouched(
        &self,
        runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
        row0: usize,
        rows: usize,
    ) -> Result<Linear> {
        anyhow::ensure!(
            rows >= 1 && row0 + rows <= self.out_features,
            "nvfp4 draft twin: rows [{row0}, {row0}+{rows}) out of range for out_features {}",
            self.out_features
        );
        match &self.storage {
            LinearStorage::Bf16 { weight, .. } => {
                let device = weight.device().clone();
                let w = weight.narrow(0, row0, rows)?.contiguous()?;
                Self::from_bf16_quantized_nvfp4_dev(&w, None, &device, runner)
            }
            LinearStorage::Fp8E4m3(s) => {
                anyhow::ensure!(
                    s.scale_mode == nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
                    "nvfp4 draft twin: fp8 source must carry per-row scales, got {:?}",
                    s.scale_mode
                );
                let k = self.in_features;
                let stream = crate::cuda_stream::current_stream(&s.cuda_device);
                let bytes_view = s.weight_u8.slice(row0 * k..(row0 + rows) * k);
                #[allow(deprecated)]
                let bytes = stream
                    .memcpy_dtov(&bytes_view)
                    .map_err(|e| anyhow::anyhow!("dtoh fp8 rows for nvfp4 draft twin: {e}"))?;
                let host = fp8_rowscale_dequant_bf16_host(
                    &bytes,
                    &s.weight_scale_rows[row0..row0 + rows],
                    rows,
                    k,
                );
                drop(bytes);
                let device = Device::Cuda(s.cuda_device.clone());
                let w = Tensor::from_vec(host, (rows, k), &device)?;
                Self::from_bf16_quantized_nvfp4_dev(&w, None, &device, runner)
            }
            LinearStorage::Nvfp4(_) => anyhow::bail!(
                "nvfp4 draft twin: source linear is already NVFP4-resident; use it directly"
            ),
        }
    }
}

pub fn quantize_fp8_per_tensor(values: &[half::bf16]) -> (Vec<u8>, f32) {
    nv_quant::fp8::quantize_e4m3_per_tensor(values)
}

pub fn fp8_rowscale_dequant_bf16_host(
    bytes: &[u8],
    row_scales: &[f32],
    out_features: usize,
    in_features: usize,
) -> Vec<half::bf16> {
    let mut host = vec![half::bf16::ZERO; out_features * in_features];
    let jobs = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .clamp(1, out_features.max(1));
    let rows_per_job = out_features.div_ceil(jobs);
    std::thread::scope(|s| {
        for (ci, dst_chunk) in host.chunks_mut(rows_per_job * in_features).enumerate() {
            let r0 = ci * rows_per_job;
            let src_chunk = &bytes[r0 * in_features..r0 * in_features + dst_chunk.len()];
            let chunk_scales = &row_scales[r0..r0 + dst_chunk.len() / in_features];
            s.spawn(move || {
                for (r, dst_row) in dst_chunk.chunks_mut(in_features).enumerate() {
                    let scale = chunk_scales[r];
                    let src_row = &src_chunk[r * in_features..(r + 1) * in_features];
                    for (dst, b) in dst_row.iter_mut().zip(src_row) {
                        *dst = half::bf16::from_f32(
                            f32::from(float8::F8E4M3::from_bits(*b)) * scale,
                        );
                    }
                }
            });
        }
    });
    host
}

pub fn checkpoint_module_is_fp8_e4m3_weight_with_scale(
    weights: &nv_weights::WeightLoader,
    module: &str,
) -> bool {
    weights.has(&format!("{module}.weight_scale"))
        && weights.st_dtype_of(&format!("{module}.weight"))
            == Some(nv_weights::StDtype::F8_E4M3)
}

pub fn fp8_e4m3_rowscale_checkpoint_dequant_linear(
    weights: &nv_weights::WeightLoader,
    module: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<Linear> {
    let weight_name = format!("{module}.weight");
    let scale_name = format!("{module}.weight_scale");
    let st = weights.st_dtype_of(&weight_name);
    anyhow::ensure!(
        st == Some(nv_weights::StDtype::F8_E4M3),
        "{weight_name}: fp8 dequant arm expects F8_E4M3 storage next to {scale_name}, got {st:?}"
    );
    let shape = weights
        .shape_of(&weight_name)
        .ok_or_else(|| anyhow::anyhow!("missing shape for {weight_name}"))?;
    anyhow::ensure!(
        shape == [out_features, in_features],
        "{weight_name}: shape {shape:?} != [{out_features}, {in_features}]"
    );
    let bytes = weights
        .raw_bytes(&weight_name)
        .map_err(|e| anyhow::anyhow!("read {weight_name}: {e}"))?;
    anyhow::ensure!(
        bytes.len() == out_features * in_features,
        "{weight_name}: {} bytes != {}",
        bytes.len(),
        out_features * in_features
    );
    let scale_t = weights
        .get(&scale_name, DType::F32)
        .map_err(|e| anyhow::anyhow!("load {scale_name}: {e}"))?;
    let scale_dims = scale_t.dims().to_vec();
    let scale_vals: Vec<f32> = scale_t.flatten_all()?.to_vec1()?;
    let row_scales = nv_weights::fp8_row_scales_from(&scale_dims, &scale_vals, out_features)
        .map_err(|e| anyhow::anyhow!("{scale_name} shape {scale_dims:?}: {e}"))?;
    let host = fp8_rowscale_dequant_bf16_host(bytes, &row_scales, out_features, in_features);
    let w = Tensor::from_vec(host, (out_features, in_features), weights.device())?;
    let w = if dtype == DType::BF16 {
        w
    } else {
        w.to_dtype(dtype)?
    };
    Linear::new(w, None)
}

#[cfg(feature = "cuda")]
pub const FP8_RESIDENT_WEIGHT_BYTES_PER_PARAM_REPLACING_4_FROM_BF16_DEQUANT_PLUS_PRETRANSPOSE:
    usize = 1;

#[cfg(feature = "cuda")]
pub fn fp8_e4m3_rowscale_checkpoint_resident_linear(
    weights: &nv_weights::WeightLoader,
    module: &str,
    out_features: usize,
    in_features: usize,
    device: &Device,
    runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
) -> Result<Linear> {
    let weight_name = format!("{module}.weight");
    let scale_name = format!("{module}.weight_scale");
    let st = weights.st_dtype_of(&weight_name);
    anyhow::ensure!(
        st == Some(nv_weights::StDtype::F8_E4M3),
        "{weight_name}: fp8 resident arm expects F8_E4M3 storage next to {scale_name}, got {st:?}"
    );
    let shape = weights
        .shape_of(&weight_name)
        .ok_or_else(|| anyhow::anyhow!("missing shape for {weight_name}"))?;
    anyhow::ensure!(
        shape == [out_features, in_features],
        "{weight_name}: shape {shape:?} != [{out_features}, {in_features}]"
    );
    let bytes = weights
        .raw_bytes(&weight_name)
        .map_err(|e| anyhow::anyhow!("read {weight_name}: {e}"))?;
    anyhow::ensure!(
        bytes.len()
            == out_features
                * in_features
                * FP8_RESIDENT_WEIGHT_BYTES_PER_PARAM_REPLACING_4_FROM_BF16_DEQUANT_PLUS_PRETRANSPOSE,
        "{weight_name}: {} bytes != {}",
        bytes.len(),
        out_features * in_features
    );
    let scale_t = weights
        .get(&scale_name, DType::F32)
        .map_err(|e| anyhow::anyhow!("load {scale_name}: {e}"))?;
    let scale_dims = scale_t.dims().to_vec();
    let scale_vals: Vec<f32> = scale_t.flatten_all()?.to_vec1()?;
    let row_scales = nv_weights::fp8_row_scales_from(&scale_dims, &scale_vals, out_features)
        .map_err(|e| anyhow::anyhow!("{scale_name} shape {scale_dims:?}: {e}"))?;
    let dev = match device {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("fp8 resident linear requires a CUDA device"),
    };
    let stream = crate::cuda_stream::current_stream(&dev);
    #[allow(deprecated)]
    let weight_u8 = stream
        .clone_htod(bytes)
        .map_err(|e| anyhow::anyhow!(e))?;
    Linear::new_fp8_e4m3_row_scales_without_the_cublaslt_probe(
        weight_u8,
        row_scales,
        in_features,
        out_features,
        None,
        device,
        runner,
        nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use half::bf16;

    #[test]
    fn nvfp4_true_m_ships_default_off_and_zero_stays_off() {
        assert!(!nvfp4_true_m_from(None));
        assert!(!nvfp4_true_m_from(Some("0")));
        assert!(nvfp4_true_m_from(Some("1")));
        assert!(nvfp4_true_m_from(Some("")));
    }

    #[test]
    fn small_m_det_ships_default_on_and_zero_disables() {
        assert!(small_m_det_from(None));
        assert!(!small_m_det_from(Some("0")));
        assert!(small_m_det_from(Some("1")));
        assert!(small_m_det_from(Some("")));
    }

    #[test]
    fn fp8_ships_per_row_scales_and_only_an_explicit_env_coarsens_them() {
        use nv_quant::fp8::Fp8ScaleMode;
        assert_eq!(fp8_scale_mode_from(None), Fp8ScaleMode::PerOuterRow);
        assert_eq!(fp8_scale_mode_from(Some("")), Fp8ScaleMode::PerOuterRow);
        assert_eq!(fp8_scale_mode_from(Some("row")), Fp8ScaleMode::PerOuterRow);
        assert_eq!(fp8_scale_mode_from(Some("1")), Fp8ScaleMode::PerOuterRow);
        assert_eq!(fp8_scale_mode_from(Some("tensor")), Fp8ScaleMode::PerTensor);
        assert_eq!(fp8_scale_mode_from(Some("TENSOR")), Fp8ScaleMode::PerTensor);
        assert_eq!(
            fp8_scale_mode_from(Some("per_tensor")),
            Fp8ScaleMode::PerTensor
        );
        assert_eq!(
            fp8_scale_mode_from(Some("per-tensor")),
            Fp8ScaleMode::PerTensor
        );
        assert_eq!(fp8_scale_mode_from(Some("0")), Fp8ScaleMode::PerTensor);
    }

    #[test]
    fn fp8_weight_payload_defaults_to_one_scale_per_output_row() {
        use nv_quant::fp8::{dequantize_e4m3_per_row, Fp8ScaleMode};
        let (out_f, in_f) = (6usize, 32usize);
        let host: Vec<bf16> = (0..out_f * in_f)
            .map(|i| {
                let r = i / in_f;
                let c = i % in_f;
                bf16::from_f32(2f32.powi(-(4 * r as i32)) * ((c as f32) * 0.31).cos())
            })
            .collect();
        let want: Vec<f32> = host.iter().map(|v| v.to_f32()).collect();

        let (row_bytes, row_scales) =
            fp8_weight_payload(&host, out_f, in_f, None, Fp8ScaleMode::PerOuterRow).unwrap();
        assert_eq!(row_scales.len(), out_f);
        assert!(
            row_scales.windows(2).all(|w| w[0] > w[1]),
            "each quieter row must get its own smaller scale: {row_scales:?}"
        );

        let (tensor_bytes, tensor_scales) =
            fp8_weight_payload(&host, out_f, in_f, None, Fp8ScaleMode::PerTensor).unwrap();
        assert!(tensor_scales.windows(2).all(|w| w[0] == w[1]));

        let row_back = dequantize_e4m3_per_row(&row_bytes, out_f, in_f, &row_scales).unwrap();
        let tensor_back =
            dequantize_e4m3_per_row(&tensor_bytes, out_f, in_f, &tensor_scales).unwrap();
        let quiet = (out_f - 1) * in_f;
        let err = |got: &[f32]| -> f32 {
            got[quiet..]
                .iter()
                .zip(&want[quiet..])
                .map(|(g, w)| (g - w).abs() / w.abs().max(1e-12))
                .fold(0f32, f32::max)
        };
        let row_err = err(&row_back);
        let tensor_err = err(&tensor_back);
        assert!(row_err < 0.10, "per-row err {row_err} on the quiet row");
        assert!(
            tensor_err > 0.5,
            "per-tensor err {tensor_err} should destroy the quiet row"
        );
    }

    #[test]
    fn fp8_weight_payload_uses_checkpoint_scales_instead_of_recomputing() {
        use nv_quant::fp8::Fp8ScaleMode;
        let (out_f, in_f) = (3usize, 16usize);
        let host: Vec<bf16> = (0..out_f * in_f)
            .map(|i| bf16::from_f32(((i as f32) * 0.17 + 0.3).sin()))
            .collect();
        let (_auto_bytes, auto_scales) =
            fp8_weight_payload(&host, out_f, in_f, None, Fp8ScaleMode::PerOuterRow).unwrap();
        let ckpt: Vec<f32> = auto_scales.iter().map(|s| s * 1.5).collect();
        let (_bytes, used) =
            fp8_weight_payload(&host, out_f, in_f, Some(&ckpt), Fp8ScaleMode::PerOuterRow).unwrap();
        assert_eq!(used, ckpt, "checkpoint weight_scale must survive verbatim");

        assert!(fp8_weight_payload(
            &host,
            out_f,
            in_f,
            Some(&ckpt[..2]),
            Fp8ScaleMode::PerOuterRow
        )
        .is_err());

        assert!(
            fp8_weight_payload(&host, out_f, in_f, Some(&ckpt), Fp8ScaleMode::PerTensor).is_err()
        );
        let uniform = vec![auto_scales[0]; out_f];
        assert!(
            fp8_weight_payload(&host, out_f, in_f, Some(&uniform), Fp8ScaleMode::PerTensor).is_ok()
        );
    }

    #[test]
    fn fp8_weight_payload_rejects_a_length_mismatch() {
        use nv_quant::fp8::Fp8ScaleMode;
        let host: Vec<bf16> = vec![bf16::from_f32(1.0); 12];
        assert!(fp8_weight_payload(&host, 3, 5, None, Fp8ScaleMode::PerOuterRow).is_err());
        assert!(fp8_weight_payload(&host, 3, 4, None, Fp8ScaleMode::PerOuterRow).is_ok());
    }

    #[test]
    fn cpu_bf16_matmul_via_linear_forward() {
        let dev = Device::Cpu;

        let in_features = 4usize;
        let out_features = 3usize;
        let w_f32: Vec<f32> = vec![
            0.5, -1.0, 0.25, 0.75, -0.5, 1.5, -0.25, 0.0, 1.0, 0.0, -1.0, 0.5,
        ];
        let w_bf: Vec<bf16> = w_f32.iter().copied().map(bf16::from_f32).collect();
        let weight = Tensor::from_vec(w_bf, (out_features, in_features), &dev).unwrap();
        let linear = Linear::new(weight, None).expect("build bf16 Linear");

        let x_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 0.5, 0.5];
        let x_bf: Vec<bf16> = x_f32.iter().copied().map(bf16::from_f32).collect();
        let x = Tensor::from_vec(x_bf, (2, in_features), &dev).unwrap();

        let out = linear.forward(&x).expect("forward");

        assert_eq!(out.dims(), &[2, out_features]);

        assert_eq!(out.dtype(), DType::BF16);

        let mut expected = vec![0f32; 2 * out_features];
        for r in 0..2 {
            for o in 0..out_features {
                let mut s = 0f32;
                for k in 0..in_features {
                    s += x_f32[r * in_features + k] * w_f32[o * in_features + k];
                }
                expected[r * out_features + o] = s;
            }
        }
        let got_bf: Vec<bf16> = out
            .to_vec2::<bf16>()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        let got_f32: Vec<f32> = got_bf.iter().map(|x| x.to_f32()).collect();
        for (i, (g, e)) in got_f32.iter().zip(expected.iter()).enumerate() {
            let denom = e.abs().max(1e-3);
            let rel = (g - e).abs() / denom;
            assert!(
                rel < 1e-2,
                "element {i}: got {g}, expected {e}, rel err {rel}"
            );
        }
    }

    #[test]
    fn cpu_forward_honors_input_start_offset() {
        let dev = Device::Cpu;
        let in_features = 4usize;
        let out_features = 3usize;
        let w_bf: Vec<bf16> = (0..out_features * in_features)
            .map(|i| bf16::from_f32(((i as f32) * 0.13).sin()))
            .collect();
        let weight = Tensor::from_vec(w_bf, (out_features, in_features), &dev).unwrap();
        let linear = Linear::new(weight, None).unwrap();

        let rows: Vec<bf16> = (0..2 * in_features)
            .map(|i| bf16::from_f32(((i as f32) * 0.11).cos()))
            .collect();
        let mut full: Vec<bf16> = (0..in_features)
            .map(|i| bf16::from_f32(900.0 + i as f32))
            .collect();
        full.extend_from_slice(&rows);
        let full_t = Tensor::from_vec(full, (3, in_features), &dev).unwrap();
        let view = full_t.narrow(0, 1, 2).unwrap();
        let ref_t = Tensor::from_vec(rows, (2, in_features), &dev).unwrap();

        let y_view = linear.forward(&view).unwrap();
        let y_ref = linear.forward(&ref_t).unwrap();
        let a: Vec<bf16> = y_view
            .to_vec2::<bf16>()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        let b: Vec<bf16> = y_ref
            .to_vec2::<bf16>()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        );
    }
}
