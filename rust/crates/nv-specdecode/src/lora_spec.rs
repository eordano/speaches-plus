use anyhow::{bail, Result};
use candle_core::Tensor;
use nv_layers::lora_slots::{LoraAdapter, LoraModuleSpec, LoraModuleWeights};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(any(feature = "cuda", feature = "wgpu"))]
use nv_models::gemma4::Gemma4;

pub fn qkv_module_names(layer: usize, has_v: bool) -> Vec<String> {
    let mut names = vec![format!("l{layer}.q"), format!("l{layer}.k")];
    if has_v {
        names.push(format!("l{layer}.v"));
    }
    names
}

pub fn det_rand_tensor(
    seed: u64,
    rows: usize,
    cols: usize,
    magnitude: f64,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut v = Vec::with_capacity(rows * cols);
    for i in 0..rows * cols {
        let mut z = seed
            .wrapping_add(0x9E3779B97F4A7C15u64.wrapping_mul(i as u64 + 1))
            .wrapping_mul(0xBF58476D1CE4E5B9);
        z ^= z >> 29;
        z = z.wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 32;
        let u = ((z & 0xFFFF) as f32 / 65535.0) - 0.5;
        v.push(half::bf16::from_f32(u * 2.0 * magnitude as f32));
    }
    Ok(Tensor::from_vec(v, (rows, cols), device)?)
}

pub fn synth_adapter(
    specs: &[LoraModuleSpec],
    rank: usize,
    magnitude: f64,
    seed: u64,
    device: &candle_core::Device,
) -> Result<LoraAdapter> {
    let mut modules = HashMap::new();
    for (i, spec) in specs.iter().enumerate() {
        let sa = seed ^ ((i as u64) << 8) ^ spec.name.len() as u64;
        let a = det_rand_tensor(sa ^ 0xA, rank, spec.in_features, magnitude, device)?;
        let b = det_rand_tensor(sa ^ 0xB, spec.out_features, rank, magnitude, device)?;
        modules.insert(spec.name.clone(), LoraModuleWeights { a, b });
    }
    Ok(LoraAdapter {
        scaling: 1.0,
        modules,
    })
}

struct TokenMapState {
    mapping: Vec<i32>,
    no_lora: bool,
    armed: bool,
}

pub struct LoraTokenMap {
    max_tokens: usize,
    max_loras: usize,
    state: Mutex<TokenMapState>,
}

impl LoraTokenMap {
    pub fn new(max_tokens: usize, max_loras: usize) -> Result<Arc<Self>> {
        if max_tokens == 0 || max_loras == 0 {
            bail!("LoraTokenMap dims must be non-zero");
        }
        Ok(Arc::new(Self {
            max_tokens,
            max_loras,
            state: Mutex::new(TokenMapState {
                mapping: Vec::new(),
                no_lora: true,
                armed: false,
            }),
        }))
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn max_loras(&self) -> usize {
        self.max_loras
    }

    pub fn set_mapping(&self, mapping: &[i32]) -> Result<()> {
        if mapping.is_empty() {
            bail!("LoraTokenMap.set_mapping: empty mapping");
        }
        if mapping.len() > self.max_tokens {
            bail!(
                "LoraTokenMap.set_mapping: {} tokens exceeds max_tokens {}",
                mapping.len(),
                self.max_tokens
            );
        }
        for &v in mapping {
            if v < -1 || v >= self.max_loras as i32 {
                bail!("LoraTokenMap.set_mapping: slot {v} out of range");
            }
        }
        let mut st = self
            .state
            .lock()
            .map_err(|e| anyhow::anyhow!("LoraTokenMap state poisoned: {e}"))?;
        st.no_lora = mapping.iter().all(|&v| v == -1);
        st.mapping = mapping.to_vec();
        st.armed = true;
        Ok(())
    }

    pub fn disarm(&self) {
        if let Ok(mut st) = self.state.lock() {
            st.armed = false;
        }
    }

    pub fn armed(&self) -> bool {
        self.state
            .lock()
            .map(|s| s.armed && !s.no_lora)
            .unwrap_or(false)
    }

    pub fn snapshot(&self) -> Option<Vec<i32>> {
        let st = self.state.lock().ok()?;
        if !st.armed || st.no_lora {
            return None;
        }
        Some(st.mapping.clone())
    }
}

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use nv_layers::linear::Linear;
use std::sync::atomic::AtomicU64;
#[cfg(feature = "wgpu")]
pub use wgpu_runtime::{WgpuLoraHook, WgpuLoraRuntime};

#[cfg(feature = "wgpu")]
mod wgpu_runtime {
    use super::LoraTokenMap;
    use anyhow::{anyhow, bail, Context, Result};
    use candle_core::{DType, Tensor};
    use half::bf16;
    use nv_kernels::wgpu_backend::kernels::lora as wgpu_lora;
    use nv_kernels::wgpu_backend::WgpuContext;
    use nv_layers::linear::{Linear, LoraDeltaHook};
    use nv_layers::lora_slots::{LoraAdapter, LoraModuleSpec, LoraSlotManager, LoraSlotStack};
    use nv_models::gemma4::Gemma4;
    use std::sync::Arc;

    pub const FUSED_MAX_M: usize = 64;

    pub struct WgpuLoraHook {
        ctx: &'static WgpuContext,
        map: Arc<LoraTokenMap>,
        a_slices: Vec<Vec<u16>>,
        b_slices: Vec<Vec<u16>>,
        widths: Vec<usize>,
        rank: usize,
        in_features: usize,
        out_features: usize,
    }

    fn host_bits(t: &Tensor) -> Result<Vec<u16>> {
        let v = t.flatten_all()?.to_dtype(DType::BF16)?.to_vec1::<bf16>()?;
        Ok(v.into_iter().map(|x| x.to_bits()).collect())
    }

    impl WgpuLoraHook {
        pub fn from_stacks(
            ctx: &'static WgpuContext,
            map: Arc<LoraTokenMap>,
            stacks: &[&LoraSlotStack],
        ) -> Result<Arc<Self>> {
            if stacks.is_empty() {
                bail!("WgpuLoraHook needs at least one slot stack");
            }
            let rank = stacks[0].max_rank();
            let in_features = stacks[0].in_features();
            if rank > wgpu_lora::FUSED_MAX_RANK {
                bail!(
                    "WgpuLoraHook: max_rank {rank} exceeds kernel limit {}",
                    wgpu_lora::FUSED_MAX_RANK
                );
            }
            let mut widths = Vec::with_capacity(stacks.len());
            let mut a_slices = Vec::with_capacity(stacks.len());
            let mut b_slices = Vec::with_capacity(stacks.len());
            for st in stacks {
                if st.max_rank() != rank || st.in_features() != in_features {
                    bail!("WgpuLoraHook: slot stacks must share max_rank and in_features");
                }
                if st.max_loras() != map.max_loras() {
                    bail!(
                        "WgpuLoraHook: stack max_loras {} != map max_loras {}",
                        st.max_loras(),
                        map.max_loras()
                    );
                }
                if st.lora_a_stacked().dtype() != DType::BF16 {
                    bail!("WgpuLoraHook: slot stacks must be bf16");
                }
                a_slices.push(host_bits(st.lora_a_stacked())?);
                b_slices.push(host_bits(st.lora_b_stacked())?);
                widths.push(st.out_features());
            }
            let out_features = widths.iter().sum();
            Ok(Arc::new(Self {
                ctx,
                map,
                a_slices,
                b_slices,
                widths,
                rank,
                in_features,
                out_features,
            }))
        }

        pub fn map(&self) -> &Arc<LoraTokenMap> {
            &self.map
        }
    }

    impl LoraDeltaHook for WgpuLoraHook {
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
            let Some(mapping) = self.map.snapshot() else {
                return Ok(None);
            };
            let m = mapping.len();
            let (xm, xk) = x2.dims2()?;
            if xm != m {
                bail!(
                    "WgpuLoraHook.apply: batch rows {xm} != armed mapping length {m}; \
                     call LoraTokenMap::set_mapping with the current token count"
                );
            }
            if xk != self.in_features {
                bail!(
                    "WgpuLoraHook.apply: x cols {xk} != in_features {}",
                    self.in_features
                );
            }
            let (win_off, win_len) = win.unwrap_or((0, self.out_features));
            if win_off + win_len > self.out_features {
                bail!("WgpuLoraHook.apply: window {win_off}+{win_len} exceeds out_features");
            }
            let (ym, yn) = y2.dims2()?;
            if ym != m || yn != win_len {
                bail!("WgpuLoraHook.apply: y dims [{ym},{yn}] != [{m},{win_len}]");
            }
            if y2.dtype() != DType::BF16 {
                bail!("WgpuLoraHook.apply: y must be bf16");
            }

            let x_host = host_bits(&x2.to_dtype(DType::BF16)?.contiguous()?)?;
            let mut y_host: Vec<u16> = y2
                .contiguous()?
                .flatten_all()?
                .to_vec1::<bf16>()?
                .into_iter()
                .map(|v| v.to_bits())
                .collect();

            let meta = wgpu_lora::LoraMeta::prepare(&mapping, self.map.max_loras());
            let a_refs: Vec<&[u16]> = self.a_slices.iter().map(|v| v.as_slice()).collect();
            let b_refs: Vec<&[u16]> = self.b_slices.iter().map(|v| v.as_slice()).collect();
            let full = win_off == 0 && win_len == self.out_features;

            if full && m > FUSED_MAX_M {
                wgpu_lora::lora_grouped(
                    self.ctx,
                    &x_host,
                    &a_refs,
                    &b_refs,
                    &mut y_host,
                    &meta,
                    &self.widths,
                    m,
                    self.rank,
                    self.in_features,
                    self.out_features,
                    1.0,
                    None,
                )
                .map_err(|e| anyhow!("wgpu lora_grouped: {e}"))?;
            } else {
                wgpu_lora::lora_fused(
                    self.ctx,
                    &x_host,
                    &a_refs,
                    &b_refs,
                    &mut y_host,
                    &meta,
                    &self.widths,
                    m,
                    self.rank,
                    self.in_features,
                    win_off,
                    win_len,
                    win_len,
                    1.0,
                )
                .map_err(|e| anyhow!("wgpu lora_fused: {e}"))?;
            }

            let y_new: Vec<bf16> = y_host.into_iter().map(bf16::from_bits).collect();
            let out = Tensor::from_vec(y_new, (m, win_len), y2.device())?;
            Ok(Some(out))
        }
    }

    pub struct WgpuLoraRuntime {
        map: Arc<LoraTokenMap>,
        manager: LoraSlotManager,
        slot: usize,
        hooked: usize,
    }

    impl WgpuLoraRuntime {
        pub fn install(
            ctx: &'static WgpuContext,
            targets: &[(&Linear, Vec<LoraModuleSpec>)],
            adapter: &LoraAdapter,
            max_rank: usize,
            max_tokens: usize,
            device: &candle_core::Device,
        ) -> Result<Self> {
            let map = LoraTokenMap::new(max_tokens, 1)?;
            let mut specs = Vec::new();
            for (_, group) in targets {
                for s in group {
                    specs.push(LoraModuleSpec::new(
                        s.name.clone(),
                        s.in_features,
                        s.out_features,
                    ));
                }
            }
            let mut manager = LoraSlotManager::new(1, max_rank, &specs, DType::BF16, device)
                .context("create LoraSlotManager")?;
            let slot = manager.activate(1, adapter).context("activate adapter")?;
            let mut hooked = 0usize;
            for (i, (linear, group)) in targets.iter().enumerate() {
                let stacks: Vec<&LoraSlotStack> = group
                    .iter()
                    .map(|s| manager.stack(&s.name).expect("stack registered above"))
                    .collect();
                let hook = WgpuLoraHook::from_stacks(ctx, map.clone(), &stacks)
                    .with_context(|| format!("build hook for target {i}"))?;
                linear
                    .attach_lora(hook)
                    .with_context(|| format!("attach hook to target {i}"))?;
                hooked += 1;
            }
            Ok(Self {
                map,
                manager,
                slot,
                hooked,
            })
        }

        pub fn install_gemma4_qkv(
            ctx: &'static WgpuContext,
            model: &Gemma4,
            adapter: &LoraAdapter,
            max_rank: usize,
            max_tokens: usize,
        ) -> Result<Self> {
            let device = model.device();
            let mut targets: Vec<(&Linear, Vec<LoraModuleSpec>)> = Vec::new();
            for (i, layer) in model.layers().iter().enumerate() {
                let attn = &layer.self_attn;
                let k_in = attn.qkv_proj.in_features();
                let widths: Vec<usize> = if attn.has_v {
                    vec![attn.q_dim, attn.kv_dim, attn.kv_dim]
                } else {
                    vec![attn.q_dim, attn.kv_dim]
                };
                let specs: Vec<LoraModuleSpec> = super::qkv_module_names(i, attn.has_v)
                    .into_iter()
                    .zip(widths)
                    .map(|(name, w)| LoraModuleSpec::new(name, k_in, w))
                    .collect();
                targets.push((&attn.qkv_proj, specs));
            }
            Self::install(ctx, &targets, adapter, max_rank, max_tokens, device)
        }

        pub fn detach_gemma4(model: &Gemma4) {
            for layer in model.layers() {
                layer.self_attn.qkv_proj.detach_lora();
            }
        }

        pub fn arm(&self, n_tokens: usize) -> Result<()> {
            self.map.set_mapping(&vec![self.slot as i32; n_tokens])
        }

        pub fn disarm(&self) {
            self.map.disarm();
        }

        pub fn armed(&self) -> bool {
            self.map.armed()
        }

        pub fn hooked_layers(&self) -> usize {
            self.hooked
        }

        pub fn manager(&self) -> &LoraSlotManager {
            &self.manager
        }

        pub fn map(&self) -> &Arc<LoraTokenMap> {
            &self.map
        }
    }
}

pub struct ServingDeltaSlice {
    pub offset: usize,
    pub width: usize,
    pub scaling: f64,
    pub a_t: Tensor,
    pub b_t: Tensor,
}

impl ServingDeltaSlice {
    pub fn from_peft_pair(
        module: &str,
        offset: usize,
        width: usize,
        in_features: usize,
        a: &Tensor,
        b: &Tensor,
        scaling: f64,
    ) -> Result<Self> {
        let (a_rank, a_in) = a.dims2()?;
        if a_in != in_features {
            bail!(
                "{module}: lora_A in dim {a_in} != linear in_features {in_features}; this \
                 adapter was trained against a different base"
            );
        }
        let (b_out, b_rank) = b.dims2()?;
        if b_out != width {
            bail!(
                "{module}: lora_B out dim {b_out} != projection slice width {width}; this \
                 adapter was trained against a different base"
            );
        }
        if a_rank != b_rank || a_rank == 0 {
            bail!("{module}: lora_A rank {a_rank} and lora_B rank {b_rank} must match and be non-zero");
        }
        Ok(Self {
            offset,
            width,
            scaling,
            a_t: a.t()?.contiguous()?.to_dtype(candle_core::DType::F32)?,
            b_t: b.t()?.contiguous()?.to_dtype(candle_core::DType::F32)?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServingHookEngagement {
    pub m1_kernel_applies: u64,
    pub m1_applies_inside_an_open_capture: u64,
    pub fallback_candle_applies: u64,
}

static SERVING_M1_KERNEL_APPLIES: AtomicU64 = AtomicU64::new(0);
static SERVING_M1_APPLIES_INSIDE_AN_OPEN_CAPTURE: AtomicU64 = AtomicU64::new(0);
static SERVING_FALLBACK_CANDLE_APPLIES: AtomicU64 = AtomicU64::new(0);

pub fn serving_hook_engagement_counted_host_side_so_replays_of_a_captured_graph_leave_it_flat(
) -> ServingHookEngagement {
    use std::sync::atomic::Ordering::Relaxed;
    ServingHookEngagement {
        m1_kernel_applies: SERVING_M1_KERNEL_APPLIES.load(Relaxed),
        m1_applies_inside_an_open_capture: SERVING_M1_APPLIES_INSIDE_AN_OPEN_CAPTURE.load(Relaxed),
        fallback_candle_applies: SERVING_FALLBACK_CANDLE_APPLIES.load(Relaxed),
    }
}

pub struct ServingLoraHook {
    slices: Vec<ServingDeltaSlice>,
    in_features: usize,
    out_features: usize,
    #[cfg(feature = "cuda")]
    cuda_m1_kernel_delta_that_survives_the_forked_stream_decode_capture:
        Option<Arc<CudaM1TwoPhaseDelta>>,
}

#[cfg(feature = "cuda")]
struct CudaM1TwoPhaseDelta {
    device: candle_core::CudaDevice,
    _a_tensors: Vec<Tensor>,
    _b_tensors: Vec<Tensor>,
    _a_ptrs_d: CudaSlice<u64>,
    _b_ptrs_d: CudaSlice<u64>,
    _slice_n_d: CudaSlice<i32>,
    _slice_start_d: CudaSlice<i32>,
    _partials: CudaSlice<f32>,
    a_ptrs_ptr: u64,
    b_ptrs_ptr: u64,
    slice_n_ptr: u64,
    slice_start_ptr: u64,
    partials_ptr: u64,
    rank: usize,
    in_features: usize,
    n_slices: usize,
    n_tiles: usize,
    k_tile: usize,
    max_n: usize,
}

#[cfg(feature = "cuda")]
fn cuda_bf16_ptr(t: &Tensor, stream: &Arc<cudarc::driver::CudaStream>) -> Result<u64> {
    use cudarc::driver::DevicePtr as _;
    let (storage, layout) = t.storage_and_layout();
    let s = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => bail!("expected cuda storage"),
    };
    let sl = s.as_cuda_slice::<half::bf16>()?;
    let view = sl.slice(layout.start_offset()..);
    let (p, g) = view.device_ptr(stream);
    drop(g);
    Ok(p)
}

#[cfg(feature = "cuda")]
impl CudaM1TwoPhaseDelta {
    fn launch(&self, x2: &Tensor, y2: &Tensor, win_off: usize, win_len: usize) -> Result<()> {
        use std::ffi::c_void;
        anyhow::ensure!(
            x2.dtype() == candle_core::DType::BF16 && x2.is_contiguous(),
            "CudaM1TwoPhaseDelta::launch requires a bf16 contiguous input: a candle dtype or \
             layout conversion here would launch on the legacy stream inside the open decode \
             capture, the exact failure this kernel path exists to avoid (got {:?}, \
             contiguous={})",
            x2.dtype(),
            x2.is_contiguous()
        );
        let stream = nv_layers::cuda_stream::current_stream(&self.device);
        let x_ptr = cuda_bf16_ptr(x2, &stream)?;
        let y_ptr = cuda_bf16_ptr(y2, &stream)?;
        let rc = unsafe {
            nv_kernels::lora::lora_m1_shrink_tiles(
                stream.cu_stream() as *mut c_void,
                x_ptr as *const u16,
                self.a_ptrs_ptr as *const u64,
                self.partials_ptr as *mut f32,
                self.n_slices as i32,
                self.rank as i32,
                self.in_features as i32,
                self.k_tile as i32,
                self.n_tiles as i32,
            )
        };
        if rc != 0 {
            bail!("lora_m1_shrink_tiles rc={rc}");
        }
        let rc = unsafe {
            nv_kernels::lora::lora_m1_expand_tiles(
                stream.cu_stream() as *mut c_void,
                self.partials_ptr as *const f32,
                self.b_ptrs_ptr as *const u64,
                y_ptr as *mut u16,
                self.slice_n_ptr as *const i32,
                self.slice_start_ptr as *const i32,
                self.n_slices as i32,
                self.rank as i32,
                self.n_tiles as i32,
                self.max_n as i32,
                win_off as i32,
                win_len as i32,
            )
        };
        if rc != 0 {
            bail!("lora_m1_expand_tiles rc={rc}");
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn build_cuda_m1_kernel_delta(
    in_features: usize,
    out_features: usize,
    slices: &[ServingDeltaSlice],
) -> Result<Option<Arc<CudaM1TwoPhaseDelta>>> {
    use candle_core::{DType, Device};
    use cudarc::driver::{DevicePtr as _, DevicePtrMut as _};
    let device = slices[0].a_t.device().clone();
    let Device::Cuda(dev) = device else {
        return Ok(None);
    };
    let mut sorted: Vec<&ServingDeltaSlice> = slices.iter().collect();
    sorted.sort_by_key(|s| s.offset);
    for w in sorted.windows(2) {
        if w[0].offset + w[0].width > w[1].offset {
            bail!(
                "ServingLoraHook: slices {}..{} and {}..{} overlap; the m=1 kernel slice \
                 table needs disjoint column ranges",
                w[0].offset,
                w[0].offset + w[0].width,
                w[1].offset,
                w[1].offset + w[1].width
            );
        }
    }
    let mut max_rank = 0usize;
    for s in &sorted {
        max_rank = max_rank.max(s.a_t.dims2()?.1);
    }
    if max_rank > nv_kernels::lora::M1_MAX_RANK {
        bail!(
            "ServingLoraHook: adapter rank {max_rank} exceeds the m=1 kernel limit {}; the \
             captured gemma4 decode runs on a forked stream where candle matmuls launch \
             uncaptured on the legacy stream and bake freed addresses, so serving on cuda \
             REQUIRES the nv-kernels m=1 path and an adapter it cannot express must refuse \
             to install",
            nv_kernels::lora::M1_MAX_RANK
        );
    }
    let cuda_dev = dev.clone();
    let stream = nv_layers::cuda_stream::current_stream(&cuda_dev);
    let mut a_tensors: Vec<Tensor> = Vec::with_capacity(sorted.len());
    let mut b_tensors: Vec<Tensor> = Vec::with_capacity(sorted.len());
    let mut a_addrs: Vec<u64> = Vec::with_capacity(sorted.len());
    let mut b_addrs: Vec<u64> = Vec::with_capacity(sorted.len());
    let mut slice_n: Vec<i32> = Vec::with_capacity(sorted.len());
    let mut slice_start: Vec<i32> = Vec::with_capacity(sorted.len());
    let mut max_n = 0usize;
    let cdev = candle_core::Device::Cuda(dev.clone());
    for s in sorted {
        let rank = s.a_t.dims2()?.1;
        let mut a = s.a_t.t()?.contiguous()?.to_dtype(DType::BF16)?;
        if rank < max_rank {
            let z = Tensor::zeros((max_rank - rank, in_features), DType::BF16, &cdev)?;
            a = Tensor::cat(&[a, z], 0)?.contiguous()?;
        }
        let b_t = if s.scaling != 1.0 {
            s.b_t.affine(s.scaling, 0.0)?
        } else {
            s.b_t.clone()
        };
        let mut b = b_t.t()?.contiguous()?.to_dtype(DType::BF16)?;
        if rank < max_rank {
            let z = Tensor::zeros((s.width, max_rank - rank), DType::BF16, &cdev)?;
            b = Tensor::cat(&[b, z], 1)?.contiguous()?;
        }
        a_addrs.push(cuda_bf16_ptr(&a, &stream)?);
        b_addrs.push(cuda_bf16_ptr(&b, &stream)?);
        a_tensors.push(a);
        b_tensors.push(b);
        slice_n.push(s.width as i32);
        slice_start.push(s.offset as i32);
        max_n = max_n.max(s.width);
    }
    let n_slices = a_tensors.len();
    let (n_tiles, k_tile) = nv_kernels::lora::m1_shrink_tiling(in_features);
    let a_ptrs_d = stream.clone_htod(&a_addrs)?;
    let b_ptrs_d = stream.clone_htod(&b_addrs)?;
    let slice_n_d = stream.clone_htod(&slice_n)?;
    let slice_start_d = stream.clone_htod(&slice_start)?;
    let mut partials = stream.alloc_zeros::<f32>(n_slices * n_tiles * max_rank)?;
    let ro = |s: &CudaSlice<u64>| -> u64 {
        let (p, g) = s.device_ptr(&stream);
        drop(g);
        p
    };
    let roi = |s: &CudaSlice<i32>| -> u64 {
        let (p, g) = s.device_ptr(&stream);
        drop(g);
        p
    };
    let a_ptrs_ptr = ro(&a_ptrs_d);
    let b_ptrs_ptr = ro(&b_ptrs_d);
    let slice_n_ptr = roi(&slice_n_d);
    let slice_start_ptr = roi(&slice_start_d);
    let partials_ptr = {
        let (p, g) = partials.device_ptr_mut(&stream);
        drop(g);
        p
    };
    let _ = out_features;
    Ok(Some(Arc::new(CudaM1TwoPhaseDelta {
        device: cuda_dev,
        _a_tensors: a_tensors,
        _b_tensors: b_tensors,
        _a_ptrs_d: a_ptrs_d,
        _b_ptrs_d: b_ptrs_d,
        _slice_n_d: slice_n_d,
        _slice_start_d: slice_start_d,
        _partials: partials,
        a_ptrs_ptr,
        b_ptrs_ptr,
        slice_n_ptr,
        slice_start_ptr,
        partials_ptr,
        rank: max_rank,
        in_features,
        n_slices,
        n_tiles,
        k_tile,
        max_n,
    })))
}

#[cfg(feature = "cuda")]
fn a_capture_is_open_so_candle_matmuls_would_launch_uncaptured_on_the_legacy_stream(
    x2: &Tensor,
) -> bool {
    use cudarc::driver::sys as drv;
    let candle_core::Device::Cuda(d) = x2.device() else {
        return false;
    };
    let stream = nv_layers::cuda_stream::current_stream(d);
    let mut st = drv::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let rc = unsafe { drv::cuStreamIsCapturing(stream.cu_stream(), &mut st) };
    rc == drv::CUresult::CUDA_SUCCESS
        && st != drv::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
}

impl ServingLoraHook {
    pub fn new(
        in_features: usize,
        out_features: usize,
        slices: Vec<ServingDeltaSlice>,
    ) -> Result<Arc<Self>> {
        if slices.is_empty() {
            bail!(
                "ServingLoraHook with zero slices would be an always-on no-op that still \
                 disables every has_lora fast path; skip attaching instead"
            );
        }
        for s in &slices {
            if s.offset + s.width > out_features {
                bail!(
                    "ServingLoraHook slice {}+{} exceeds out_features {out_features}",
                    s.offset,
                    s.width
                );
            }
        }
        #[cfg(feature = "cuda")]
        let cuda_m1 = build_cuda_m1_kernel_delta(in_features, out_features, &slices)?;
        Ok(Arc::new(Self {
            slices,
            in_features,
            out_features,
            #[cfg(feature = "cuda")]
            cuda_m1_kernel_delta_that_survives_the_forked_stream_decode_capture: cuda_m1,
        }))
    }
}

impl nv_layers::linear::LoraDeltaHook for ServingLoraHook {
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
        let (m, k) = x2.dims2()?;
        if k != self.in_features {
            bail!(
                "ServingLoraHook.apply: x cols {k} != in_features {}",
                self.in_features
            );
        }
        let (win_off, win_len) = win.unwrap_or((0, self.out_features));
        if win_off + win_len > self.out_features {
            bail!(
                "ServingLoraHook.apply: window {win_off}+{win_len} exceeds out_features {}",
                self.out_features
            );
        }
        let (ym, yn) = y2.dims2()?;
        if ym != m || yn != win_len {
            bail!("ServingLoraHook.apply: y dims [{ym},{yn}] != [{m},{win_len}]");
        }
        #[cfg(feature = "cuda")]
        {
            if let Some(hook) =
                &self.cuda_m1_kernel_delta_that_survives_the_forked_stream_decode_capture
            {
                if m == 1
                    && x2.device().is_cuda()
                    && y2.dtype() == candle_core::DType::BF16
                {
                    use std::sync::atomic::Ordering::Relaxed;
                    SERVING_M1_KERNEL_APPLIES.fetch_add(1, Relaxed);
                    if a_capture_is_open_so_candle_matmuls_would_launch_uncaptured_on_the_legacy_stream(x2) {
                        SERVING_M1_APPLIES_INSIDE_AN_OPEN_CAPTURE.fetch_add(1, Relaxed);
                    }
                    hook.launch(x2, y2, win_off, win_len)?;
                    return Ok(None);
                }
            }
            if x2.device().is_cuda()
                && a_capture_is_open_so_candle_matmuls_would_launch_uncaptured_on_the_legacy_stream(
                    x2,
                )
            {
                bail!(
                    "ServingLoraHook.apply: the current stream is capturing and this call \
                     (m={m}, y dtype {:?}) cannot take the nv-kernels m=1 path; refusing the \
                     candle fallback because its cublas launches go to the legacy stream, \
                     execute outside the capture, and bake freed buffer addresses into the \
                     replayed graph",
                    y2.dtype()
                );
            }
        }
        SERVING_FALLBACK_CANDLE_APPLIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let x_f32 = if x2.dtype() == candle_core::DType::F32 {
            x2.clone()
        } else {
            x2.to_dtype(candle_core::DType::F32)?
        };
        let mut out = y2.clone();
        let mut changed = false;
        for s in &self.slices {
            let lo = s.offset.max(win_off);
            let hi = (s.offset + s.width).min(win_off + win_len);
            if lo >= hi {
                continue;
            }
            let h = x_f32.matmul(&s.a_t)?;
            let b_t = s.b_t.narrow(1, lo - s.offset, hi - lo)?.contiguous()?;
            let mut d = h.matmul(&b_t)?;
            if s.scaling != 1.0 {
                d = d.affine(s.scaling, 0.0)?;
            }
            let d = d.to_dtype(out.dtype())?;
            let c0 = lo - win_off;
            let len = hi - lo;
            let seg = out.narrow(1, c0, len)?.add(&d)?;
            let mut parts: Vec<Tensor> = Vec::with_capacity(3);
            if c0 > 0 {
                parts.push(out.narrow(1, 0, c0)?);
            }
            parts.push(seg);
            if c0 + len < yn {
                parts.push(out.narrow(1, c0 + len, yn - c0 - len)?);
            }
            out = Tensor::cat(&parts, 1)?;
            changed = true;
        }
        if changed {
            Ok(Some(out))
        } else {
            Ok(None)
        }
    }
}

pub const GEMMA4_SERVING_LORA_PREFIX: &str = "model.language_model";

#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub struct ServingInstall {
    pub matched_modules: usize,
    pub hooked_linears: usize,
    pub skipped: Vec<String>,
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
fn plan_serving_hook<'m>(
    linear: &'m Linear,
    slices: &[(String, usize, usize)],
    adapter: &nv_weights::lora_adapter::LoraAdapter,
    planned: &mut Vec<(&'m Linear, Arc<ServingLoraHook>)>,
    matched: &mut usize,
) -> Result<()> {
    let mut hook_slices = Vec::new();
    for (module, offset, width) in slices {
        let Some(w) = adapter.loras.get(module) else {
            continue;
        };
        if w.is_embedding {
            bail!("{module}: embedding LoRA tensors have no linear projection to hook");
        }
        hook_slices.push(ServingDeltaSlice::from_peft_pair(
            module,
            *offset,
            *width,
            linear.in_features(),
            &w.lora_a,
            &w.lora_b,
            w.scaling,
        )?);
        *matched += 1;
    }
    if hook_slices.is_empty() {
        return Ok(());
    }
    let hook = ServingLoraHook::new(linear.in_features(), linear.out_features(), hook_slices)?;
    planned.push((linear, hook));
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub fn install_gemma4_serving_adapter(
    model: &Gemma4,
    adapter: &nv_weights::lora_adapter::LoraAdapter,
) -> Result<ServingInstall> {
    let mut planned: Vec<(&Linear, Arc<ServingLoraHook>)> = Vec::new();
    let mut matched = 0usize;
    let mut skipped: Vec<String> = Vec::new();
    for (li, layer) in model.layers().iter().enumerate() {
        let p = format!("{GEMMA4_SERVING_LORA_PREFIX}.layers.{li}");
        let attn = &layer.self_attn;
        let mut qkv_slices: Vec<(String, usize, usize)> = vec![
            (format!("{p}.self_attn.q_proj"), 0, attn.q_dim),
            (format!("{p}.self_attn.k_proj"), attn.q_dim, attn.kv_dim),
        ];
        let v_name = format!("{p}.self_attn.v_proj");
        if attn.has_v {
            qkv_slices.push((v_name, attn.q_dim + attn.kv_dim, attn.kv_dim));
        } else if adapter.loras.contains_key(&v_name) {
            skipped.push(format!(
                "{v_name}: this layer computes v as k (attention_k_eq_v), so the fused qkv \
                 projection has no v slice for the adapter to land on"
            ));
        }
        plan_serving_hook(
            &attn.qkv_proj,
            &qkv_slices,
            adapter,
            &mut planned,
            &mut matched,
        )?;
        plan_serving_hook(
            &attn.o_proj,
            &[(
                format!("{p}.self_attn.o_proj"),
                0,
                attn.o_proj.out_features(),
            )],
            adapter,
            &mut planned,
            &mut matched,
        )?;
        let inter = layer.mlp.gate_up_proj.out_features() / 2;
        plan_serving_hook(
            &layer.mlp.gate_up_proj,
            &[
                (format!("{p}.mlp.gate_proj"), 0, inter),
                (format!("{p}.mlp.up_proj"), inter, inter),
            ],
            adapter,
            &mut planned,
            &mut matched,
        )?;
        plan_serving_hook(
            &layer.mlp.down_proj,
            &[(
                format!("{p}.mlp.down_proj"),
                0,
                layer.mlp.down_proj.out_features(),
            )],
            adapter,
            &mut planned,
            &mut matched,
        )?;
    }
    if matched == 0 {
        bail!(
            "lora adapter matched no gemma4 projection; it carries {} modules, the first of \
             which is {:?}. Expected names like {GEMMA4_SERVING_LORA_PREFIX}.layers.0.self_attn.q_proj",
            adapter.loras.len(),
            adapter.loras.keys().next()
        );
    }
    let mut attached: Vec<&Linear> = Vec::new();
    for (linear, hook) in planned {
        if let Err(err) = linear.attach_lora(hook) {
            for l in attached {
                l.detach_lora();
            }
            return Err(err.context(
                "attach_lora failed after dim validation; every already-attached hook was \
                 detached so a refused adapter leaves no partial application",
            ));
        }
        attached.push(linear);
    }
    Ok(ServingInstall {
        matched_modules: matched,
        hooked_linears: attached.len(),
        skipped,
    })
}

pub const LAGUNA_SERVING_LORA_PREFIX_IS_THE_GEMMA4_NAMESPACE_BECAUSE_NV_WEIGHTS_NORMALIZE_MODULE_NAME_REWRITES_MODEL_LAYERS:
    &str = GEMMA4_SERVING_LORA_PREFIX;

#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub fn detach_laguna_serving_adapter(model: &nv_models::laguna::Laguna) {
    for layer in model.layers() {
        let attn = &layer.self_attn;
        attn.q_proj.detach_lora();
        attn.k_proj.detach_lora();
        attn.v_proj.detach_lora();
        attn.o_proj.detach_lora();
        if let Some(g) = &attn.g_proj {
            g.detach_lora();
        }
        if let nv_models::laguna::LagunaFfn::Dense(mlp) = &layer.ffn {
            mlp.gate_proj().detach_lora();
            mlp.up_proj().detach_lora();
            mlp.down_proj().detach_lora();
        }
    }
    model.set_attn_w8(true);
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub fn install_laguna_serving_adapter(
    model: &nv_models::laguna::Laguna,
    adapter: &nv_weights::lora_adapter::LoraAdapter,
) -> Result<ServingInstall> {
    use nv_models::laguna::LagunaFfn;
    let prefix =
        LAGUNA_SERVING_LORA_PREFIX_IS_THE_GEMMA4_NAMESPACE_BECAUSE_NV_WEIGHTS_NORMALIZE_MODULE_NAME_REWRITES_MODEL_LAYERS;
    let mut planned: Vec<(&Linear, Arc<ServingLoraHook>)> = Vec::new();
    let mut matched = 0usize;
    let mut matched_names: Vec<String> = Vec::new();
    let mut w8_bypassed_attention_matched = false;
    for (li, layer) in model.layers().iter().enumerate() {
        let p = format!("{prefix}.layers.{li}");
        let attn = &layer.self_attn;
        let mut targets: Vec<(&Linear, String, bool)> = vec![
            (&attn.q_proj, format!("{p}.self_attn.q_proj"), true),
            (&attn.k_proj, format!("{p}.self_attn.k_proj"), true),
            (&attn.v_proj, format!("{p}.self_attn.v_proj"), true),
            (&attn.o_proj, format!("{p}.self_attn.o_proj"), true),
        ];
        let g_name = format!("{p}.self_attn.g_proj");
        match &attn.g_proj {
            Some(g) => targets.push((g, g_name, false)),
            None => {
                if adapter.loras.contains_key(&g_name) {
                    bail!(
                        "{g_name}: this laguna config declares gating=none, so there is no \
                         g_proj linear for the adapter to land on"
                    );
                }
            }
        }
        match &layer.ffn {
            LagunaFfn::Dense(mlp) => {
                targets.push((mlp.gate_proj(), format!("{p}.mlp.gate_proj"), false));
                targets.push((mlp.up_proj(), format!("{p}.mlp.up_proj"), false));
                targets.push((mlp.down_proj(), format!("{p}.mlp.down_proj"), false));
            }
            LagunaFfn::Moe(_) => {
                let moe_ns = format!("{p}.mlp.");
                if let Some(name) = adapter.loras.keys().find(|k| k.starts_with(&moe_ns)) {
                    bail!(
                        "{name}: layer {li} is MoE, and laguna MoE ffn runs grouped expert \
                         kernels (with captured per-layer MoE graphs at m=1) that never route \
                         nv-layers Linear::forward, so a hook there would be silently ignored; \
                         refusing the whole adapter rather than serving a module subset"
                    );
                }
            }
        }
        for (linear, name, is_a_w8_bypassable_qkvo_site) in targets {
            if adapter.loras.contains_key(&name) {
                matched_names.push(name.clone());
                w8_bypassed_attention_matched |= is_a_w8_bypassable_qkvo_site;
            }
            plan_serving_hook(
                linear,
                &[(name, 0, linear.out_features())],
                adapter,
                &mut planned,
                &mut matched,
            )?;
        }
    }
    if matched == 0 {
        bail!(
            "lora adapter matched no laguna projection; it carries {} modules, the first of \
             which is {:?}. Laguna peft targets (model.layers.N.self_attn.q_proj, ...) arrive \
             here rewritten by nv_weights::lora_adapter::normalize_module_name to names like \
             {prefix}.layers.0.self_attn.q_proj",
            adapter.loras.len(),
            adapter.loras.keys().next()
        );
    }
    for key in adapter.loras.keys() {
        if !matched_names.iter().any(|m| m == key) {
            bail!(
                "{key}: matched no laguna projection while {matched} other modules did; \
                 serving a module subset would silently change the adapter's trained \
                 function. Hookable laguna targets are self_attn.{{q,k,v,o,g}}_proj and \
                 dense-layer mlp.{{gate,up,down}}_proj"
            );
        }
    }
    if w8_bypassed_attention_matched && model.attn_w8_active() && model.attn_w8_shape_policy() {
        bail!(
            "this laguna engine booted with the attn-w8 'shape' policy \
             (NV_LAGUNA_ATTN_W8/NV_LAGUNA_ATTN_FP8 contains 'shape'): apply_attn_w8_for \
             re-arms the quantized q/k/v/o gemv before every m=1 forward, and that gemv \
             bypasses the nv-layers Linear hook this adapter needs, so disabling it at \
             install cannot stick; boot the adapter engine without the shape policy"
        );
    }
    let mut attached: Vec<&Linear> = Vec::new();
    for (linear, hook) in planned {
        if let Err(err) = linear.attach_lora(hook) {
            for l in attached {
                l.detach_lora();
            }
            return Err(err.context(
                "attach_lora failed after dim validation; every already-attached hook was \
                 detached so a refused adapter leaves no partial application",
            ));
        }
        attached.push(linear);
    }
    let mut skipped: Vec<String> = Vec::new();
    if w8_bypassed_attention_matched && model.attn_w8_active() {
        model.set_attn_w8(false);
        let scope = model.attn_w8_scope();
        if scope != 0 {
            for l in attached {
                l.detach_lora();
            }
            model.set_attn_w8(true);
            bail!(
                "set_attn_w8(false) left w8 scope {scope}; the quantized q/k/v/o gemv would \
                 bypass the attached attention hooks, so every hook was detached and the \
                 adapter refused"
            );
        }
        skipped.push(
            "attn w8/fp8 m=1 projections disabled on this adapter engine: the quantized gemv \
             bypasses nv-layers Linear hooks, so hooked q/k/v/o must take Linear::forward"
                .into(),
        );
    }
    Ok(ServingInstall {
        matched_modules: matched,
        hooked_linears: attached.len(),
        skipped,
    })
}

#[cfg(feature = "cuda")]
pub struct Qwen4ExpServingAdapter {
    hooks: Vec<(String, Arc<ServingLoraHook>)>,
    experts: Vec<(usize, Arc<nv_models::qwen4_exp::lora::expert::ExpertLoraLayer>)>,
}

#[cfg(feature = "cuda")]
impl Qwen4ExpServingAdapter {
    pub fn prepare(
        model: &nv_models::qwen4_exp::Qwen4Exp,
        adapter: &nv_weights::lora_adapter::LoraAdapter,
    ) -> Result<Self> {
        Self::prepare_with_experts(model, adapter, Vec::new())
    }

    fn prepare_linears(
        model: &nv_models::qwen4_exp::Qwen4Exp,
        adapter: &nv_weights::lora_adapter::LoraAdapter,
    ) -> Result<Self> {
        let targets = model.lora_target_linears();
        for key in adapter.loras.keys() {
            if !targets.iter().any(|(name, ..)| name == key) {
                bail!(
                    "{key}: not a qwen4_exp runtime-LoRA target; serving the other modules would \
                     silently change the adapter's trained function. Targets are full-attention \
                     self_attn.{{q,k,v,o}}_proj and mlp.shared_expert.{{gate,up,down}}_proj: {}",
                    nv_models::qwen4_exp::lora::LINEAR_TARGETS_ARE_THE_BF16_CHECKPOINT_PROJECTIONS_AND_THE_ROUTED_EXPERTS_ADAPT_THROUGH_THE_MOE_LAYER_EXPERT_HOOK
                );
            }
        }
        let mut hooks = Vec::new();
        for (name, _, _, linear) in targets {
            let mut planned = Vec::new();
            let mut matched = 0usize;
            plan_serving_hook(
                linear,
                &[(name.clone(), 0, linear.out_features())],
                adapter,
                &mut planned,
                &mut matched,
            )?;
            if let Some((_, hook)) = planned.pop() {
                hooks.push((name, hook));
            }
        }
        Ok(Self { hooks, experts: Vec::new() })
    }

    pub fn prepare_with_experts(
        model: &nv_models::qwen4_exp::Qwen4Exp,
        adapter: &nv_weights::lora_adapter::LoraAdapter,
        experts: Vec<(usize, Arc<nv_models::qwen4_exp::lora::expert::ExpertLoraLayer>)>,
    ) -> Result<Self> {
        let mut s = Self::prepare_linears(model, adapter)?;
        s.experts = experts;
        if s.hooks.is_empty() && s.experts.is_empty() {
            bail!(
                "lora adapter matched no qwen4_exp projection and carries no expert lora; it has {} \
                 modules, the first of which is {:?}",
                adapter.loras.len(),
                adapter.loras.keys().next()
            );
        }
        Ok(s)
    }

    pub fn modules(&self) -> usize {
        self.hooks.len()
    }

    pub fn expert_layers(&self) -> usize {
        self.experts.len()
    }

    pub fn expert_device_bytes(&self) -> usize {
        self.experts.iter().map(|(_, l)| l.device_bytes()).sum()
    }

    pub fn attach(&self, model: &nv_models::qwen4_exp::Qwen4Exp) -> Result<()> {
        model.detach_all_lora();
        for (li, l) in &self.experts {
            if let Err(e) = model.attach_expert_lora(*li, l.clone()) {
                model.detach_all_lora();
                return Err(e);
            }
        }
        let targets = model.lora_target_linears();
        for (name, hook) in &self.hooks {
            let (_, _, _, linear) = targets
                .iter()
                .find(|(n, ..)| n == name)
                .ok_or_else(|| anyhow::anyhow!("{name}: not a projection of this qwen4_exp model"))?;
            if let Err(e) = linear.attach_lora(hook.clone()) {
                model.detach_all_lora();
                return Err(e);
            }
        }
        Ok(())
    }

    pub fn detach(model: &nv_models::qwen4_exp::Qwen4Exp) {
        model.detach_all_lora();
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub fn synth_qkv_adapter(
    model: &Gemma4,
    rank: usize,
    magnitude: f64,
    seed: u64,
) -> Result<LoraAdapter> {
    let device = model.device();
    let mut modules = HashMap::new();
    for (i, layer) in model.layers().iter().enumerate() {
        let attn = &layer.self_attn;
        let k_in = attn.qkv_proj.in_features();
        let widths: Vec<usize> = if attn.has_v {
            vec![attn.q_dim, attn.kv_dim, attn.kv_dim]
        } else {
            vec![attn.q_dim, attn.kv_dim]
        };
        for (name, w) in qkv_module_names(i, attn.has_v).into_iter().zip(widths) {
            let sa = seed ^ (i as u64) << 8 ^ name.len() as u64;
            let a = det_rand_tensor(sa ^ 0xA, rank, k_in, magnitude, device)?;
            let b = det_rand_tensor(sa ^ 0xB, w, rank, magnitude, device)?;
            modules.insert(name, LoraModuleWeights { a, b });
        }
    }
    Ok(LoraAdapter {
        scaling: 1.0,
        modules,
    })
}
