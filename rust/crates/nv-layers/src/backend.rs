use std::fmt;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
#[cfg(feature = "wgpu")]
use nv_kernels::wgpu_backend::device::WgpuContext;
#[cfg(feature = "wgpu")]
use nv_kernels::wgpu_backend::kernels as wk;

#[cfg(feature = "wgpu")]
#[path = "wgpu_tensor.rs"]
pub mod wgpu_tensor;

pub const BACKEND_ENV: &str = "NV_KERNELS_BACKEND";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cuda,
    Wgpu,
    Cpu,
}

impl BackendKind {
    pub const ALL: [BackendKind; 3] = [BackendKind::Cuda, BackendKind::Wgpu, BackendKind::Cpu];

    pub fn name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
            Self::Cpu => "cpu",
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSel {
    Auto,
    Cuda,
    Wgpu,
    Cpu,
}

impl BackendSel {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Some(Self::Auto),
            "cuda" => Some(Self::Cuda),
            "wgpu" => Some(Self::Wgpu),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }

    pub fn from_env() -> Result<Self, BackendError> {
        match std::env::var(BACKEND_ENV) {
            Err(_) => Ok(Self::Auto),
            Ok(v) => Self::parse(&v).ok_or_else(|| {
                BackendError::Selection(format!(
                    "{BACKEND_ENV}={v:?} is not one of auto|cuda|wgpu|cpu"
                ))
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelId {
    GatherRowsBf16,
    Rmsnorm,
    RmsnormResidual,
    ResidualScale,
    Rope,
    RopeBf16,
    Silu,
    GeluTanhMul,
    GemvBf16,
    GemvW4a16,
    GemvNvfp4,
    GemmNvfp4,
    QuantizeNvfp4Bf16,
    KvFp8,
    KvFp8Paged,
    AttnDecode,
    FlashDecode,
    AttentionFp8Decode,
    GraphDecode,
    Sampler,
    MoePermute,
    MoeUnpermuteScatter,
    DepthwiseConv1dSiluBf16,
    GdnGating,
    GdnRecurrent,
    TreeVerifyAttn,
    TreeVerifyFp8,
    LoraGrouped,
    LoraFused,
    VerifyFusedNorms,
    MarlinGemmW4a16,
    MoeGroupedGemmNvfp4,
}

impl KernelId {
    pub const ALL: [KernelId; 32] = [
        KernelId::GatherRowsBf16,
        KernelId::Rmsnorm,
        KernelId::RmsnormResidual,
        KernelId::ResidualScale,
        KernelId::Rope,
        KernelId::RopeBf16,
        KernelId::Silu,
        KernelId::GeluTanhMul,
        KernelId::GemvBf16,
        KernelId::GemvW4a16,
        KernelId::GemvNvfp4,
        KernelId::GemmNvfp4,
        KernelId::QuantizeNvfp4Bf16,
        KernelId::KvFp8,
        KernelId::KvFp8Paged,
        KernelId::AttnDecode,
        KernelId::FlashDecode,
        KernelId::AttentionFp8Decode,
        KernelId::GraphDecode,
        KernelId::Sampler,
        KernelId::MoePermute,
        KernelId::MoeUnpermuteScatter,
        KernelId::DepthwiseConv1dSiluBf16,
        KernelId::GdnGating,
        KernelId::GdnRecurrent,
        KernelId::TreeVerifyAttn,
        KernelId::TreeVerifyFp8,
        KernelId::LoraGrouped,
        KernelId::LoraFused,
        KernelId::VerifyFusedNorms,
        KernelId::MarlinGemmW4a16,
        KernelId::MoeGroupedGemmNvfp4,
    ];

    pub const DENSE_DECODE_PATH: [KernelId; 17] = [
        KernelId::GatherRowsBf16,
        KernelId::Rmsnorm,
        KernelId::RmsnormResidual,
        KernelId::ResidualScale,
        KernelId::GraphDecode,
        KernelId::GemvBf16,
        KernelId::GemvW4a16,
        KernelId::GemvNvfp4,
        KernelId::GemmNvfp4,
        KernelId::QuantizeNvfp4Bf16,
        KernelId::Rope,
        KernelId::RopeBf16,
        KernelId::KvFp8,
        KernelId::FlashDecode,
        KernelId::GeluTanhMul,
        KernelId::Silu,
        KernelId::Sampler,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::GatherRowsBf16 => "gather_rows_bf16",
            Self::Rmsnorm => "rmsnorm",
            Self::RmsnormResidual => "rmsnorm_residual",
            Self::ResidualScale => "residual_scale",
            Self::Rope => "rope",
            Self::RopeBf16 => "rope_bf16",
            Self::Silu => "silu",
            Self::GeluTanhMul => "gelu_tanh_mul",
            Self::GemvBf16 => "gemv_bf16",
            Self::GemvW4a16 => "gemv_w4a16",
            Self::GemvNvfp4 => "gemv_nvfp4",
            Self::GemmNvfp4 => "gemm_nvfp4",
            Self::QuantizeNvfp4Bf16 => "quantize_nvfp4_bf16",
            Self::KvFp8 => "kv_fp8",
            Self::KvFp8Paged => "kv_fp8_paged",
            Self::AttnDecode => "attn_decode",
            Self::FlashDecode => "flash_decode",
            Self::AttentionFp8Decode => "attention_fp8_decode",
            Self::GraphDecode => "graph_decode",
            Self::Sampler => "sampler",
            Self::MoePermute => "moe_permute",
            Self::MoeUnpermuteScatter => "moe_unpermute_scatter",
            Self::DepthwiseConv1dSiluBf16 => "depthwise_conv1d_silu_bf16",
            Self::GdnGating => "gdn_gating",
            Self::GdnRecurrent => "gdn_recurrent",
            Self::TreeVerifyAttn => "tree_verify_attn",
            Self::TreeVerifyFp8 => "tree_verify_fp8",
            Self::LoraGrouped => "lora_grouped",
            Self::LoraFused => "lora_fused",
            Self::VerifyFusedNorms => "verify_fused_norms",
            Self::MarlinGemmW4a16 => "marlin_gemm_w4a16",
            Self::MoeGroupedGemmNvfp4 => "moe_grouped_fp4_gemm",
        }
    }
}

impl fmt::Display for KernelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

pub fn kind_supports(kind: BackendKind, kernel: KernelId) -> bool {
    match kind {
        BackendKind::Cuda => true,
        BackendKind::Wgpu => !matches!(kernel, KernelId::MarlinGemmW4a16),
        BackendKind::Cpu => matches!(
            kernel,
            KernelId::Rmsnorm | KernelId::Silu | KernelId::GemvBf16
        ),
    }
}

pub fn missing_on(kind: BackendKind, needed: &[KernelId]) -> Vec<KernelId> {
    needed
        .iter()
        .copied()
        .filter(|k| !kind_supports(kind, *k))
        .collect()
}

pub fn supporting_backends(kernel: KernelId) -> Vec<BackendKind> {
    BackendKind::ALL
        .iter()
        .copied()
        .filter(|b| kind_supports(*b, kernel))
        .collect()
}

#[derive(Debug)]
pub enum BackendError {
    Selection(String),
    Unavailable {
        kind: BackendKind,
        reason: String,
    },
    MissingKernel {
        kind: BackendKind,
        kernel: KernelId,
    },
    Shape(String),
    Bridge(String),
    Kernel {
        kind: BackendKind,
        kernel: KernelId,
        detail: String,
    },
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Selection(m) => write!(f, "backend selection: {m}"),
            Self::Unavailable { kind, reason } => {
                write!(f, "{kind} backend unavailable: {reason}")
            }
            Self::MissingKernel { kind, kernel } => {
                let have: Vec<&str> = supporting_backends(*kernel)
                    .into_iter()
                    .map(BackendKind::name)
                    .collect();
                write!(
                    f,
                    "kernel {kernel} is not implemented on the {kind} backend (implemented on: {}); no silent fallback is attempted",
                    have.join(", ")
                )
            }
            Self::Shape(m) => write!(f, "shape mismatch: {m}"),
            Self::Bridge(m) => write!(f, "candle-to-wgpu bridge: {m}"),
            Self::Kernel {
                kind,
                kernel,
                detail,
            } => {
                write!(f, "kernel {kernel} failed on the {kind} backend: {detail}")
            }
        }
    }
}

impl std::error::Error for BackendError {}

pub fn resolve_from(
    sel: BackendSel,
    probe_cuda: &dyn Fn() -> Result<(), String>,
    probe_wgpu: &dyn Fn() -> Result<(), String>,
) -> Result<BackendKind, BackendError> {
    match sel {
        BackendSel::Cpu => Ok(BackendKind::Cpu),
        BackendSel::Cuda => {
            probe_cuda()
                .map(|_| BackendKind::Cuda)
                .map_err(|reason| BackendError::Unavailable {
                    kind: BackendKind::Cuda,
                    reason,
                })
        }
        BackendSel::Wgpu => {
            probe_wgpu()
                .map(|_| BackendKind::Wgpu)
                .map_err(|reason| BackendError::Unavailable {
                    kind: BackendKind::Wgpu,
                    reason,
                })
        }
        BackendSel::Auto => {
            if probe_cuda().is_ok() {
                Ok(BackendKind::Cuda)
            } else if probe_wgpu().is_ok() {
                Ok(BackendKind::Wgpu)
            } else {
                Ok(BackendKind::Cpu)
            }
        }
    }
}

pub fn probe_cuda() -> Result<(), String> {
    #[cfg(feature = "cuda")]
    {
        match cudarc::driver::CudaContext::new(0) {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("cuda device 0: {e}")),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err("nv-layers compiled without the cuda feature".to_string())
    }
}

pub fn probe_wgpu() -> Result<(), String> {
    #[cfg(feature = "wgpu")]
    {
        match WgpuContext::shared() {
            Ok(ctx) => {
                let q = ctx.qualify();
                if q.qualified {
                    Ok(())
                } else {
                    Err(format!("wgpu adapter not qualified: {:?}", q.reason))
                }
            }
            Err(e) => Err(e.to_string()),
        }
    }
    #[cfg(not(feature = "wgpu"))]
    {
        Err("nv-layers compiled without the wgpu feature".to_string())
    }
}

pub fn availability() -> Vec<(BackendKind, Result<(), String>)> {
    vec![
        (BackendKind::Cuda, probe_cuda()),
        (BackendKind::Wgpu, probe_wgpu()),
        (BackendKind::Cpu, Ok(())),
    ]
}

#[cfg(feature = "cuda")]
pub struct CudaBackend {
    pub stream: std::sync::Arc<CudaStream>,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    fn stream_ptr(&self) -> *mut std::ffi::c_void {
        self.stream.cu_stream() as *mut std::ffi::c_void
    }
}

#[cfg(feature = "wgpu")]
pub struct WgpuBackend {
    pub ctx: &'static WgpuContext,
    pub residency: wgpu_tensor::ResidencyCache,
}

#[cfg(feature = "wgpu")]
impl WgpuBackend {
    pub fn upload_weight(
        &self,
        label: &str,
        t: &candle_core::Tensor,
    ) -> Result<std::sync::Arc<wgpu_tensor::WgpuTensor>, BackendError> {
        self.residency.get_or_upload(self.ctx, label, t)
    }
}

pub enum Backend {
    #[cfg(feature = "cuda")]
    Cuda(CudaBackend),
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuBackend),
    Cpu,
}

impl Backend {
    pub fn new(sel: BackendSel) -> Result<Self, BackendError> {
        let kind = resolve_from(sel, &probe_cuda, &probe_wgpu)?;
        Self::open(kind)
    }

    pub fn from_env() -> Result<Self, BackendError> {
        Self::new(BackendSel::from_env()?)
    }

    pub fn open(kind: BackendKind) -> Result<Self, BackendError> {
        match kind {
            BackendKind::Cpu => Ok(Self::Cpu),
            BackendKind::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    let ctx = cudarc::driver::CudaContext::new(0).map_err(|e| {
                        BackendError::Unavailable {
                            kind,
                            reason: e.to_string(),
                        }
                    })?;
                    Ok(Self::Cuda(CudaBackend {
                        stream: ctx.default_stream(),
                    }))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(BackendError::Unavailable {
                        kind,
                        reason: "nv-layers compiled without the cuda feature".to_string(),
                    })
                }
            }
            BackendKind::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    let ctx = WgpuContext::shared().map_err(|e| BackendError::Unavailable {
                        kind,
                        reason: e.to_string(),
                    })?;
                    Ok(Self::Wgpu(WgpuBackend {
                        ctx,
                        residency: wgpu_tensor::ResidencyCache::new(),
                    }))
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    Err(BackendError::Unavailable {
                        kind,
                        reason: "nv-layers compiled without the wgpu feature".to_string(),
                    })
                }
            }
        }
    }

    pub fn kind(&self) -> BackendKind {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => BackendKind::Cuda,
            #[cfg(feature = "wgpu")]
            Self::Wgpu(_) => BackendKind::Wgpu,
            Self::Cpu => BackendKind::Cpu,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => "cuda device 0".to_string(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(w) => w.ctx.summary(),
            Self::Cpu => "cpu reference".to_string(),
        }
    }

    pub fn supports(&self, kernel: KernelId) -> bool {
        kind_supports(self.kind(), kernel)
    }

    pub fn missing(&self, needed: &[KernelId]) -> Vec<KernelId> {
        missing_on(self.kind(), needed)
    }

    pub fn require(&self, kernel: KernelId) -> Result<(), BackendError> {
        if self.supports(kernel) {
            Ok(())
        } else {
            Err(BackendError::MissingKernel {
                kind: self.kind(),
                kernel,
            })
        }
    }

    pub fn rmsnorm_f32(
        &self,
        x: &[f32],
        weight: &[f32],
        batch: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<Vec<f32>, BackendError> {
        self.require(KernelId::Rmsnorm)?;
        check_len("rmsnorm x", x.len(), batch * hidden)?;
        check_len("rmsnorm weight", weight.len(), hidden)?;
        match self {
            Self::Cpu => Ok(cpu_rmsnorm_f32(x, weight, batch, hidden, eps)),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(w) => {
                let mut y = vec![0f32; batch * hidden];
                wk::rmsnorm::rmsnorm_f32(w.ctx, x, weight, &mut y, batch, hidden, eps)
                    .map_err(|e| wgpu_err(KernelId::Rmsnorm, e))?;
                Ok(y)
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(cu) => {
                #[allow(deprecated)]
                let dx: CudaSlice<f32> = cu.stream.clone_htod(x).map_err(cuda_err_rms)?;
                #[allow(deprecated)]
                let dw: CudaSlice<f32> = cu.stream.clone_htod(weight).map_err(cuda_err_rms)?;
                let mut dy: CudaSlice<f32> = cu
                    .stream
                    .alloc_zeros::<f32>(batch * hidden)
                    .map_err(cuda_err_rms)?;
                let rc = {
                    let (px, _a) = dx.device_ptr(&cu.stream);
                    let (pw, _b) = dw.device_ptr(&cu.stream);
                    let (py, _c) = dy.device_ptr_mut(&cu.stream);
                    unsafe {
                        nv_kernels::cuda::rmsnorm_f32(
                            cu.stream_ptr(),
                            px as *const f32,
                            pw as *const f32,
                            py as *mut f32,
                            batch,
                            hidden,
                            eps,
                        )
                    }
                };
                cuda_rc(KernelId::Rmsnorm, rc)?;
                cu.stream.synchronize().map_err(cuda_err_rms)?;
                #[allow(deprecated)]
                let out = cu.stream.memcpy_dtov(&dy).map_err(cuda_err_rms)?;
                Ok(out)
            }
        }
    }

    #[cfg_attr(not(any(feature = "wgpu", feature = "cuda")), allow(unused_variables))]
    pub fn silu_mul_f32(&self, x: &[f32], gate: &[f32]) -> Result<Vec<f32>, BackendError> {
        self.require(KernelId::Silu)?;
        check_len("silu_mul gate", gate.len(), x.len())?;
        let n = x.len();
        match self {
            Self::Cpu => Ok(cpu_silu_mul_f32(x, gate)),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(w) => {
                let mut y = vec![0f32; n];
                wk::silu::silu_mul_f32(w.ctx, x, gate, &mut y, n)
                    .map_err(|e| wgpu_err(KernelId::Silu, e))?;
                Ok(y)
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(cu) => {
                #[allow(deprecated)]
                let dx: CudaSlice<f32> = cu.stream.clone_htod(x).map_err(cuda_err_silu)?;
                #[allow(deprecated)]
                let dg: CudaSlice<f32> = cu.stream.clone_htod(gate).map_err(cuda_err_silu)?;
                let mut dy: CudaSlice<f32> =
                    cu.stream.alloc_zeros::<f32>(n).map_err(cuda_err_silu)?;
                let rc = {
                    let (px, _a) = dx.device_ptr(&cu.stream);
                    let (pg, _b) = dg.device_ptr(&cu.stream);
                    let (py, _c) = dy.device_ptr_mut(&cu.stream);
                    unsafe {
                        nv_kernels::cuda::silu_mul_f32(
                            cu.stream_ptr(),
                            px as *const f32,
                            pg as *const f32,
                            py as *mut f32,
                            n,
                        )
                    }
                };
                cuda_rc(KernelId::Silu, rc)?;
                cu.stream.synchronize().map_err(cuda_err_silu)?;
                #[allow(deprecated)]
                let out = cu.stream.memcpy_dtov(&dy).map_err(cuda_err_silu)?;
                Ok(out)
            }
        }
    }

    pub fn gemv_bf16(
        &self,
        w: &[u16],
        x: &[u16],
        n: usize,
        k: usize,
    ) -> Result<Vec<u16>, BackendError> {
        self.require(KernelId::GemvBf16)?;
        check_len("gemv_bf16 w", w.len(), n * k)?;
        check_len("gemv_bf16 x", x.len(), k)?;
        match self {
            Self::Cpu => Ok(cpu_gemv_bf16(w, x, n, k)),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(wb) => {
                let mut y = vec![0u16; n];
                wk::gemv_bf16::gemv_bf16(wb.ctx, w, x, &mut y, n, k)
                    .map_err(|e| wgpu_err(KernelId::GemvBf16, e))?;
                Ok(y)
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(cu) => {
                #[allow(deprecated)]
                let dw: CudaSlice<u16> = cu.stream.clone_htod(w).map_err(cuda_err_gemv)?;
                #[allow(deprecated)]
                let dx: CudaSlice<u16> = cu.stream.clone_htod(x).map_err(cuda_err_gemv)?;
                let mut dy: CudaSlice<u16> =
                    cu.stream.alloc_zeros::<u16>(n).map_err(cuda_err_gemv)?;
                let rc = {
                    let (pw, _a) = dw.device_ptr(&cu.stream);
                    let (px, _b) = dx.device_ptr(&cu.stream);
                    let (py, _c) = dy.device_ptr_mut(&cu.stream);
                    unsafe {
                        nv_kernels::cuda::gemv_bf16(
                            cu.stream_ptr(),
                            pw as *const u16,
                            px as *const u16,
                            py as *mut u16,
                            n as i32,
                            k as i32,
                        )
                    }
                };
                cuda_rc(KernelId::GemvBf16, rc)?;
                cu.stream.synchronize().map_err(cuda_err_gemv)?;
                #[allow(deprecated)]
                let out = cu.stream.memcpy_dtov(&dy).map_err(cuda_err_gemv)?;
                Ok(out)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(any(feature = "wgpu", feature = "cuda")), allow(unused_variables))]
    pub fn attn_decode_f32(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        start: usize,
        total: usize,
    ) -> Result<Vec<f32>, BackendError> {
        self.require(KernelId::AttnDecode)?;
        check_len("attn_decode q", q.len(), n_heads * head_dim)?;
        check_len("attn_decode k", k.len(), total * n_kv_heads * head_dim)?;
        check_len("attn_decode v", v.len(), total * n_kv_heads * head_dim)?;
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(w) => {
                let mut out = vec![0f32; n_heads * head_dim];
                wk::attn_decode::attn_decode_f32(
                    w.ctx, q, k, v, &mut out, n_heads, n_kv_heads, head_dim, start, total, 1.0,
                )
                .map_err(|e| wgpu_err(KernelId::AttnDecode, e))?;
                Ok(out)
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(cu) => {
                #[allow(deprecated)]
                let dq: CudaSlice<f32> = cu.stream.clone_htod(q).map_err(cuda_err_attn)?;
                #[allow(deprecated)]
                let dk: CudaSlice<f32> = cu.stream.clone_htod(k).map_err(cuda_err_attn)?;
                #[allow(deprecated)]
                let dv: CudaSlice<f32> = cu.stream.clone_htod(v).map_err(cuda_err_attn)?;
                let mut dout: CudaSlice<f32> = cu
                    .stream
                    .alloc_zeros::<f32>(n_heads * head_dim)
                    .map_err(cuda_err_attn)?;
                let rc = {
                    let (pq, _a) = dq.device_ptr(&cu.stream);
                    let (pk, _b) = dk.device_ptr(&cu.stream);
                    let (pv, _c) = dv.device_ptr(&cu.stream);
                    let (po, _d) = dout.device_ptr_mut(&cu.stream);
                    unsafe {
                        nv_kernels::cuda::attn_decode_f32(
                            cu.stream_ptr(),
                            pq as *const f32,
                            pk as *const f32,
                            pv as *const f32,
                            po as *mut f32,
                            n_heads as i32,
                            n_kv_heads as i32,
                            head_dim as i32,
                            total as i32,
                            start as i32,
                        )
                    }
                };
                cuda_rc(KernelId::AttnDecode, rc)?;
                cu.stream.synchronize().map_err(cuda_err_attn)?;
                #[allow(deprecated)]
                let out = cu.stream.memcpy_dtov(&dout).map_err(cuda_err_attn)?;
                Ok(out)
            }
            #[allow(unreachable_patterns)]
            _ => Err(BackendError::MissingKernel {
                kind: self.kind(),
                kernel: KernelId::AttnDecode,
            }),
        }
    }

    #[cfg(feature = "wgpu")]
    pub fn wgpu(&self) -> Option<&WgpuBackend> {
        match self {
            Self::Wgpu(w) => Some(w),
            _ => None,
        }
    }

    pub fn marlin_workspace_elems(&self) -> Result<usize, BackendError> {
        self.require(KernelId::MarlinGemmW4a16)?;
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => {
                let mut elems = 0i32;
                let rc = unsafe { nv_kernels::cuda::marlin_workspace_elems(&mut elems) };
                if rc != 0 {
                    return Err(BackendError::Kernel {
                        kind: BackendKind::Cuda,
                        kernel: KernelId::MarlinGemmW4a16,
                        detail: format!("rc={rc}"),
                    });
                }
                Ok(elems as usize)
            }
            #[allow(unreachable_patterns)]
            _ => Err(BackendError::MissingKernel {
                kind: self.kind(),
                kernel: KernelId::MarlinGemmW4a16,
            }),
        }
    }
}

fn check_len(what: &str, got: usize, want: usize) -> Result<(), BackendError> {
    if got == want {
        Ok(())
    } else {
        Err(BackendError::Shape(format!(
            "{what}: got {got}, want {want}"
        )))
    }
}

#[cfg(feature = "wgpu")]
fn wgpu_err(kernel: KernelId, e: nv_kernels::wgpu_backend::WgpuError) -> BackendError {
    BackendError::Kernel {
        kind: BackendKind::Wgpu,
        kernel,
        detail: e.to_string(),
    }
}

#[cfg(feature = "cuda")]
fn cuda_err(kernel: KernelId, e: impl fmt::Display) -> BackendError {
    BackendError::Kernel {
        kind: BackendKind::Cuda,
        kernel,
        detail: e.to_string(),
    }
}

#[cfg(feature = "cuda")]
fn cuda_err_rms(e: cudarc::driver::DriverError) -> BackendError {
    cuda_err(KernelId::Rmsnorm, e)
}

#[cfg(feature = "cuda")]
fn cuda_err_silu(e: cudarc::driver::DriverError) -> BackendError {
    cuda_err(KernelId::Silu, e)
}

#[cfg(feature = "cuda")]
fn cuda_err_gemv(e: cudarc::driver::DriverError) -> BackendError {
    cuda_err(KernelId::GemvBf16, e)
}

#[cfg(feature = "cuda")]
fn cuda_err_attn(e: cudarc::driver::DriverError) -> BackendError {
    cuda_err(KernelId::AttnDecode, e)
}

#[cfg(feature = "cuda")]
fn cuda_rc(kernel: KernelId, rc: i32) -> Result<(), BackendError> {
    if rc == 0 {
        Ok(())
    } else {
        Err(BackendError::Kernel {
            kind: BackendKind::Cuda,
            kernel,
            detail: format!("rc={rc}"),
        })
    }
}

fn cpu_rmsnorm_f32(x: &[f32], weight: &[f32], batch: usize, hidden: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; batch * hidden];
    for r in 0..batch {
        let row = &x[r * hidden..(r + 1) * hidden];
        let mut sum = 0f32;
        for v in row {
            sum += v * v;
        }
        let inv = 1.0 / (eps + sum / hidden as f32).sqrt();
        for (i, v) in row.iter().enumerate() {
            y[r * hidden + i] = v * inv * weight[i];
        }
    }
    y
}

fn cpu_silu_mul_f32(x: &[f32], gate: &[f32]) -> Vec<f32> {
    x.iter()
        .zip(gate.iter())
        .map(|(v, g)| {
            if *v < -88.0 {
                0.0
            } else {
                v / (1.0 + (-v).exp()) * g
            }
        })
        .collect()
}

fn cpu_gemv_bf16(w: &[u16], x: &[u16], n: usize, k: usize) -> Vec<u16> {
    use half::bf16;
    (0..n)
        .map(|row| {
            let mut acc = 0f32;
            for j in 0..k {
                acc += bf16::from_bits(w[row * k + j]).to_f32() * bf16::from_bits(x[j]).to_f32();
            }
            bf16::from_f32(acc).to_bits()
        })
        .collect()
}
