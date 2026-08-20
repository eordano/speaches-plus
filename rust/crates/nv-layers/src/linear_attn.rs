use anyhow::{Context, Result};
use candle_core::{DType, Tensor, D};
use nv_weights::WeightLoader;

#[cfg(feature = "cuda")]
use candle_core::Device;

use crate::linear::Linear;

#[derive(Clone, Copy, Debug)]
pub struct LinearAttentionConfig {
    pub hidden_size: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub mamba_ssm_dtype: DType,
    pub rms_eps: f64,
}

impl LinearAttentionConfig {
    pub fn key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    pub fn value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    pub fn v_per_k(&self) -> usize {
        self.linear_num_value_heads / self.linear_num_key_heads
    }
}

#[derive(Clone)]
pub struct LinAttnState {
    conv_state: Tensor,
    recurrent_state: Tensor,
    fused: bool,
}

impl LinAttnState {
    pub fn conv_state(&self) -> &Tensor {
        &self.conv_state
    }
    pub fn recurrent_state(&self) -> &Tensor {
        &self.recurrent_state
    }

    pub fn is_fused(&self) -> bool {
        self.fused
    }

    #[cfg(feature = "cuda")]
    pub fn deep_clone(&self) -> Result<Self> {
        let conv = deep_clone_tensor(&self.conv_state)?;
        let rec = deep_clone_tensor(&self.recurrent_state)?;
        Ok(Self {
            conv_state: conv,
            recurrent_state: rec,
            fused: self.fused,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn copy_data_from(&self, other: &LinAttnState) -> Result<()> {
        copy_tensor_data(&other.conv_state, &self.conv_state)?;
        copy_tensor_data(&other.recurrent_state, &self.recurrent_state)
    }

    #[cfg(feature = "cuda")]
    pub fn zero_data(&self) -> Result<()> {
        zero_tensor_data(&self.conv_state)?;
        zero_tensor_data(&self.recurrent_state)
    }

    #[cfg(feature = "cuda")]
    pub fn dump_primed_state_for_reuse(&self, out: &mut dyn std::io::Write) -> Result<()> {
        out.write_all(&[self.fused as u8])?;
        write_prime_state_tensor_dtype_dims_then_le_bits(&self.conv_state, out)?;
        write_prime_state_tensor_dtype_dims_then_le_bits(&self.recurrent_state, out)
    }

    #[cfg(feature = "cuda")]
    pub fn restore_primed_state_checked(
        input: &mut dyn std::io::Read,
        device: &Device,
    ) -> Result<Self> {
        let mut fused = [0u8; 1];
        input.read_exact(&mut fused)?;
        anyhow::ensure!(
            fused[0] <= 1,
            "LinAttnState restore: fused flag byte {} is not 0/1, the stream is misaligned",
            fused[0]
        );
        let conv_state = read_prime_state_tensor_dtype_dims_then_le_bits(input, device)?;
        let recurrent_state = read_prime_state_tensor_dtype_dims_then_le_bits(input, device)?;
        Ok(Self {
            conv_state,
            recurrent_state,
            fused: fused[0] == 1,
        })
    }
}

#[cfg(feature = "cuda")]
fn write_prime_state_tensor_dtype_dims_then_le_bits(
    t: &Tensor,
    out: &mut dyn std::io::Write,
) -> Result<()> {
    let dims = t.dims().to_vec();
    let dtype_tag: u8 = match t.dtype() {
        DType::BF16 => 0,
        DType::F32 => 1,
        other => anyhow::bail!(
            "LinAttnState dump handles only the BF16 conv / F32 recurrent state dtypes, got {other:?}"
        ),
    };
    out.write_all(&[dtype_tag, dims.len() as u8])?;
    for &d in &dims {
        out.write_all(&(d as u64).to_le_bytes())?;
    }
    let flat = t.contiguous()?.flatten_all()?;
    let bytes = match dtype_tag {
        0 => {
            let v = flat.to_vec1::<half::bf16>()?;
            let mut b = Vec::with_capacity(v.len() * 2);
            for x in v {
                b.extend_from_slice(&x.to_bits().to_le_bytes());
            }
            b
        }
        _ => {
            let v = flat.to_vec1::<f32>()?;
            let mut b = Vec::with_capacity(v.len() * 4);
            for x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
            b
        }
    };
    out.write_all(&(bytes.len() as u64).to_le_bytes())?;
    out.write_all(&bytes)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn read_prime_state_tensor_dtype_dims_then_le_bits(
    input: &mut dyn std::io::Read,
    device: &Device,
) -> Result<Tensor> {
    let mut hdr = [0u8; 2];
    input.read_exact(&mut hdr)?;
    let ndims = hdr[1] as usize;
    anyhow::ensure!(
        ndims <= 8,
        "LinAttnState restore: {ndims} dims is not a state tensor, the stream is misaligned"
    );
    let mut dims = Vec::with_capacity(ndims);
    let mut b8 = [0u8; 8];
    for _ in 0..ndims {
        input.read_exact(&mut b8)?;
        dims.push(u64::from_le_bytes(b8) as usize);
    }
    input.read_exact(&mut b8)?;
    let byte_len = u64::from_le_bytes(b8) as usize;
    let elems: usize = dims.iter().product();
    let mut bytes = vec![0u8; byte_len];
    input.read_exact(&mut bytes)?;
    match hdr[0] {
        0 => {
            anyhow::ensure!(
                byte_len == elems * 2,
                "LinAttnState restore: BF16 payload {byte_len} B != {elems} elems * 2"
            );
            let v: Vec<half::bf16> = bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            Ok(Tensor::from_vec(v, dims, device)?)
        }
        1 => {
            anyhow::ensure!(
                byte_len == elems * 4,
                "LinAttnState restore: F32 payload {byte_len} B != {elems} elems * 4"
            );
            let v: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(Tensor::from_vec(v, dims, device)?)
        }
        tag => anyhow::bail!(
            "LinAttnState restore: dtype tag {tag} unknown to this code version, refuse rather than misread"
        ),
    }
}

pub const MROW_VERIFY_ROWS_MAX_16_THE_GDN_CHUNK_KERNEL_T_CAP: usize = 16;
#[cfg(feature = "cuda")]
const GDN_L2_EPS_1E6_MATCHES_L2_NORMALIZE_LAST: f32 = 1e-6;
pub const MROW_VERIFY_CONV_KERNEL_MAX_8_THE_GDN_CONV_CHUNK_KERNEL_K_CAP: usize = 8;

pub fn chunk_prefill_env_read_per_call_so_one_process_can_ab_both_scan_paths() -> bool {
    const CHUNK_IS_THE_DEFAULT_SCAN_THE_PUBLISHED_PP512_5500_AND_WIKITEXT_PPL_7_0851_WERE_MEASURED_ON_IT_AND_ZERO_RESTORES_THE_TOKEN_SEQUENTIAL_CANDLE_SCAN_AT_ITS_6X_FORMATION_COST: &str =
        "NV_Q38_GDN_CHUNK_PREFILL";
    std::env::var(
        CHUNK_IS_THE_DEFAULT_SCAN_THE_PUBLISHED_PP512_5500_AND_WIKITEXT_PPL_7_0851_WERE_MEASURED_ON_IT_AND_ZERO_RESTORES_THE_TOKEN_SEQUENTIAL_CANDLE_SCAN_AT_ITS_6X_FORMATION_COST,
    )
    .ok()
    .as_deref()
        != Some("0")
}

pub fn mrow_verify_env_read_per_call_so_one_process_can_ab_both_verify_paths() -> bool {
    std::env::var("NV_Q38_MROW_VERIFY").ok().as_deref() == Some("1")
}

pub fn verify_tc_env_read_per_call_nv_q38_verify_tc_1_selects_projections_once_plus_lt_gemm_verify_arms(
) -> bool {
    std::env::var("NV_Q38_VERIFY_TC").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
pub mod gdn_step_prof {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    static ARMED: AtomicBool = AtomicBool::new(false);
    static ACC: Mutex<BTreeMap<&'static str, (f64, u32)>> = Mutex::new(BTreeMap::new());

    pub fn arm_only_while_an_eager_decode_profiler_is_active_because_every_lap_syncs(on: bool) {
        ARMED.store(
            on && std::env::var("NV_PROF_GDN").ok().as_deref() == Some("1"),
            Ordering::Relaxed,
        );
    }

    pub struct SyncLaps {
        dev: candle_core::CudaDevice,
        last: std::time::Instant,
    }

    impl SyncLaps {
        pub fn begin_if_armed(dev: &candle_core::CudaDevice) -> Option<Self> {
            if !ARMED.load(Ordering::Relaxed) {
                return None;
            }
            let _ = crate::cuda_stream::current_stream(dev).synchronize();
            Some(Self {
                dev: dev.clone(),
                last: std::time::Instant::now(),
            })
        }

        pub fn lap(&mut self, label: &'static str) {
            let _ = crate::cuda_stream::current_stream(&self.dev).synchronize();
            let now = std::time::Instant::now();
            let mut acc = ACC.lock().unwrap();
            let e = acc.entry(label).or_insert((0.0, 0));
            e.0 += (now - self.last).as_secs_f64() * 1000.0;
            e.1 += 1;
            self.last = now;
        }
    }

    pub fn report_and_reset(pos: usize) {
        let mut acc = ACC.lock().unwrap();
        if acc.is_empty() {
            return;
        }
        let total: f64 = acc.values().map(|v| v.0).sum();
        let mut entries: Vec<(&'static str, f64, u32)> =
            acc.iter().map(|(k, v)| (*k, v.0, v.1)).collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (label, ms, laps) in &entries {
            eprintln!(
                "[prof-gdn] pos={pos} {label:>22} {ms:8.3} ms ({:4.1}%) laps={laps}",
                100.0 * ms / total.max(1e-12)
            );
        }
        eprintln!("[prof-gdn] pos={pos} {:>22} {total:8.3} ms", "TOTAL");
        acc.clear();
    }
}

#[cfg(feature = "cuda")]
struct GdnDecodeScratch {
    conv_dim: usize,
    value_dim: usize,
    nk_dk: usize,
    n_v: usize,
    mixed: cudarc::driver::CudaSlice<half::bf16>,
    gated_tensor_reused_across_layers_because_the_stream_serializes: Tensor,
    qn: cudarc::driver::CudaSlice<f32>,
    kn: cudarc::driver::CudaSlice<f32>,
    g_exp: cudarc::driver::CudaSlice<f32>,
    beta: cudarc::driver::CudaSlice<f32>,
    core: cudarc::driver::CudaSlice<half::bf16>,
}

#[cfg(feature = "cuda")]
thread_local! {
    static GDN_DECODE_SCRATCH_POOL_KEYED_BY_DIMS_AND_NEVER_FREED_BECAUSE_CAPTURED_GRAPHS_BAKE_ITS_POINTERS:
        std::cell::RefCell<Vec<GdnDecodeScratch>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(feature = "cuda")]
struct GdnScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers(
    Option<GdnDecodeScratch>,
);

#[cfg(feature = "cuda")]
impl std::ops::Deref for GdnScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers {
    type Target = GdnDecodeScratch;
    fn deref(&self) -> &GdnDecodeScratch {
        self.0
            .as_ref()
            .expect("lease holds its scratch until drop")
    }
}

#[cfg(feature = "cuda")]
impl std::ops::DerefMut for GdnScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers {
    fn deref_mut(&mut self) -> &mut GdnDecodeScratch {
        self.0
            .as_mut()
            .expect("lease holds its scratch until drop")
    }
}

#[cfg(feature = "cuda")]
impl Drop for GdnScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers {
    fn drop(&mut self) {
        if let Some(s) = self.0.take() {
            let _ = GDN_DECODE_SCRATCH_POOL_KEYED_BY_DIMS_AND_NEVER_FREED_BECAUSE_CAPTURED_GRAPHS_BAKE_ITS_POINTERS
                .try_with(|c| c.borrow_mut().push(s));
        }
    }
}

#[cfg(feature = "cuda")]
fn gdn_decode_scratch_take_or_build(
    dev: &candle_core::CudaDevice,
    conv_dim: usize,
    value_dim: usize,
    nk_dk: usize,
    n_v: usize,
) -> Result<GdnScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers> {
    let pooled = GDN_DECODE_SCRATCH_POOL_KEYED_BY_DIMS_AND_NEVER_FREED_BECAUSE_CAPTURED_GRAPHS_BAKE_ITS_POINTERS
        .with(|c| {
            let mut v = c.borrow_mut();
            v.iter()
                .position(|s| {
                    s.conv_dim == conv_dim
                        && s.value_dim == value_dim
                        && s.nk_dk == nk_dk
                        && s.n_v == n_v
                })
                .map(|i| v.swap_remove(i))
        });
    if let Some(s) = pooled {
        return Ok(GdnScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers(Some(s)));
    }
    let stream = crate::cuda_stream::current_stream(dev);
    let mixed = unsafe {
        stream
            .alloc::<half::bf16>(conv_dim)
            .map_err(|e| anyhow::anyhow!("scratch mixed: {e:?}"))?
    };
    let gated = unsafe {
        stream
            .alloc::<half::bf16>(value_dim)
            .map_err(|e| anyhow::anyhow!("scratch gated: {e:?}"))?
    };
    let gated_tensor = {
        let storage = candle_core::CudaStorage::wrap_cuda_slice(gated, dev.clone());
        Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (1usize, 1usize, value_dim),
            candle_core::op::BackpropOp::none(),
            false,
        )
    };
    let qn = unsafe {
        stream
            .alloc::<f32>(nk_dk)
            .map_err(|e| anyhow::anyhow!("scratch qn: {e:?}"))?
    };
    let kn = unsafe {
        stream
            .alloc::<f32>(nk_dk)
            .map_err(|e| anyhow::anyhow!("scratch kn: {e:?}"))?
    };
    let g_exp = unsafe {
        stream
            .alloc::<f32>(n_v)
            .map_err(|e| anyhow::anyhow!("scratch g_exp: {e:?}"))?
    };
    let beta = unsafe {
        stream
            .alloc::<f32>(n_v)
            .map_err(|e| anyhow::anyhow!("scratch beta: {e:?}"))?
    };
    let core = unsafe {
        stream
            .alloc::<half::bf16>(value_dim)
            .map_err(|e| anyhow::anyhow!("scratch core: {e:?}"))?
    };
    Ok(GdnScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers(Some(
        GdnDecodeScratch {
            conv_dim,
            value_dim,
            nk_dk,
            n_v,
            mixed,
            gated_tensor_reused_across_layers_because_the_stream_serializes: gated_tensor,
            qn,
            kn,
            g_exp,
            beta,
            core,
        },
    )))
}

#[cfg(feature = "cuda")]
pub fn fused_decode_env() -> Option<bool> {
    static V: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *V.get_or_init(
        || match std::env::var("NV_GDN_FUSED_DECODE").ok().as_deref() {
            Some("0") => Some(false),
            Some("1") => Some(true),
            _ => None,
        },
    )
}

#[cfg(feature = "cuda")]
fn cuda_tensor_raw(t: &Tensor) -> Result<(candle_core::CudaDevice, u64, usize)> {
    let dev = match t.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("expected CUDA tensor"),
    };
    let stream = crate::cuda_stream::current_stream(&dev);
    let (st, l) = t.storage_and_layout();
    anyhow::ensure!(l.is_contiguous(), "expected contiguous tensor");
    let cuda = match &*st {
        candle_core::Storage::Cuda(s) => s,
        _ => anyhow::bail!("expected CUDA storage"),
    };
    let elem_bytes = t.dtype().size_in_bytes();
    let (ptr, len) = match t.dtype() {
        DType::BF16 => {
            let s = cuda.as_cuda_slice::<half::bf16>()?;
            let view = s.slice(l.start_offset()..l.start_offset() + t.elem_count());
            let (p, _g) = cudarc::driver::DevicePtr::device_ptr(&view, &stream);
            (p, t.elem_count() * elem_bytes)
        }
        DType::F32 => {
            let s = cuda.as_cuda_slice::<f32>()?;
            let view = s.slice(l.start_offset()..l.start_offset() + t.elem_count());
            let (p, _g) = cudarc::driver::DevicePtr::device_ptr(&view, &stream);
            (p, t.elem_count() * elem_bytes)
        }
        other => anyhow::bail!("unsupported dtype {other:?}"),
    };
    Ok((dev, ptr, len))
}

#[cfg(feature = "cuda")]
fn copy_tensor_data(src: &Tensor, dst: &Tensor) -> Result<()> {
    anyhow::ensure!(
        src.dtype() == dst.dtype() && src.elem_count() == dst.elem_count(),
        "state copy mismatch: src {:?} {:?} vs dst {:?} {:?}",
        src.dims(),
        src.dtype(),
        dst.dims(),
        dst.dtype()
    );
    let src_c = src.contiguous()?;
    let (dev, src_ptr, bytes) = cuda_tensor_raw(&src_c)?;
    let (_dev2, dst_ptr, _b2) = cuda_tensor_raw(dst)?;
    let stream = crate::cuda_stream::current_stream(&dev);
    unsafe {
        cudarc::driver::result::memcpy_dtod_async(dst_ptr, src_ptr, bytes, stream.cu_stream())
            .map_err(|e| anyhow::anyhow!("state dtod: {e:?}"))?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn zero_tensor_data(t: &Tensor) -> Result<()> {
    let (dev, ptr, bytes) = cuda_tensor_raw(t)?;
    let stream = crate::cuda_stream::current_stream(&dev);
    unsafe {
        cudarc::driver::result::memset_d8_async(ptr, 0u8, bytes, stream.cu_stream())
            .map_err(|e| anyhow::anyhow!("state memset: {e:?}"))?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn deep_clone_tensor(t: &Tensor) -> Result<Tensor> {
    let dev = match t.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("expected CUDA tensor"),
    };
    let stream = crate::cuda_stream::current_stream(&dev);
    let t_c = t.contiguous()?;
    let n = t_c.elem_count();
    let out = match t_c.dtype() {
        DType::BF16 => {
            let buf = stream
                .alloc_zeros::<half::bf16>(n)
                .map_err(|e| anyhow::anyhow!("alloc clone: {e:?}"))?;
            let storage = candle_core::CudaStorage::wrap_cuda_slice(buf, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                t_c.dims().to_vec(),
                candle_core::op::BackpropOp::none(),
                false,
            )
        }
        DType::F32 => {
            let buf = stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| anyhow::anyhow!("alloc clone: {e:?}"))?;
            let storage = candle_core::CudaStorage::wrap_cuda_slice(buf, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                t_c.dims().to_vec(),
                candle_core::op::BackpropOp::none(),
                false,
            )
        }
        other => anyhow::bail!("unsupported dtype {other:?}"),
    };
    copy_tensor_data(&t_c, &out)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
pub struct QkvzConcatFp8DecodeArm {
    weight_u8: cudarc::driver::CudaSlice<u8>,
    row_scales_dev: cudarc::driver::CudaSlice<f32>,
}

pub fn gdn_fuse_env_read_per_call_so_the_kill_switch_works_mid_process() -> bool {
    std::env::var("NV_Q38_GDN_FUSE").ok().as_deref() == Some("1")
}

pub fn gdn_split_env_read_per_call_so_the_kill_switch_works_mid_process() -> bool {
    std::env::var("NV_Q38_GDN_SPLIT").ok().as_deref() == Some("1")
}

pub fn gdn_prenorm_fold_env_read_per_call_so_the_kill_switch_works_mid_process() -> bool {
    std::env::var("NV_Q38_GDN_PRENORM_FOLD").ok().as_deref() == Some("1")
}

pub fn gdn_chunk_split_default_on_inside_mrow_kill_switch_nv_q38_gdn_chunk_split_0() -> bool {
    std::env::var("NV_Q38_GDN_CHUNK_SPLIT").ok().as_deref() != Some("0")
}

pub fn verify_gdn_chunk_env_opt_in_nv_q38_verify_gdn_chunk_1_writes_ckpts_in_place_off_pooled_scratch(
) -> bool {
    std::env::var("NV_Q38_VERIFY_GDN_CHUNK").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
struct MrowVerifyScratch {
    m: usize,
    conv_dim: usize,
    key_dim: usize,
    value_dim: usize,
    n_v: usize,
    mixed_seq: cudarc::driver::CudaSlice<half::bf16>,
    qn: cudarc::driver::CudaSlice<f32>,
    kn: cudarc::driver::CudaSlice<f32>,
    g_exp: cudarc::driver::CudaSlice<f32>,
    beta: cudarc::driver::CudaSlice<f32>,
    core: cudarc::driver::CudaSlice<half::bf16>,
}

#[cfg(feature = "cuda")]
thread_local! {
    static MROW_VERIFY_SCRATCH_POOL_SHARED_BY_EVERY_GDN_LAYER_BECAUSE_ONE_STREAM_SERIALIZES_THEM:
        std::cell::RefCell<Vec<MrowVerifyScratch>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(feature = "cuda")]
struct MrowScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers(
    Option<MrowVerifyScratch>,
);

#[cfg(feature = "cuda")]
impl std::ops::Deref for MrowScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers {
    type Target = MrowVerifyScratch;
    fn deref(&self) -> &MrowVerifyScratch {
        self.0.as_ref().expect("lease holds its scratch until drop")
    }
}

#[cfg(feature = "cuda")]
impl std::ops::DerefMut for MrowScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers {
    fn deref_mut(&mut self) -> &mut MrowVerifyScratch {
        self.0.as_mut().expect("lease holds its scratch until drop")
    }
}

#[cfg(feature = "cuda")]
impl Drop for MrowScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers {
    fn drop(&mut self) {
        if let Some(s) = self.0.take() {
            let _ = MROW_VERIFY_SCRATCH_POOL_SHARED_BY_EVERY_GDN_LAYER_BECAUSE_ONE_STREAM_SERIALIZES_THEM
                .try_with(|c| c.borrow_mut().push(s));
        }
    }
}

#[cfg(feature = "cuda")]
fn mrow_verify_scratch_take_or_build(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    m: usize,
    conv_dim: usize,
    key_dim: usize,
    value_dim: usize,
    n_v: usize,
) -> Result<MrowScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers> {
    let pooled = MROW_VERIFY_SCRATCH_POOL_SHARED_BY_EVERY_GDN_LAYER_BECAUSE_ONE_STREAM_SERIALIZES_THEM
        .with(|c| {
            let mut v = c.borrow_mut();
            v.iter()
                .position(|s| {
                    s.m == m
                        && s.conv_dim == conv_dim
                        && s.key_dim == key_dim
                        && s.value_dim == value_dim
                        && s.n_v == n_v
                })
                .map(|i| v.swap_remove(i))
        });
    if let Some(s) = pooled {
        return Ok(MrowScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers(Some(s)));
    }
    let mixed_seq = unsafe {
        stream
            .alloc::<half::bf16>(m * conv_dim)
            .map_err(|e| anyhow::anyhow!("alloc mrow scratch mixed: {e:?}"))?
    };
    let qn = unsafe {
        stream
            .alloc::<f32>(m * key_dim)
            .map_err(|e| anyhow::anyhow!("alloc mrow scratch qn: {e:?}"))?
    };
    let kn = unsafe {
        stream
            .alloc::<f32>(m * key_dim)
            .map_err(|e| anyhow::anyhow!("alloc mrow scratch kn: {e:?}"))?
    };
    let g_exp = unsafe {
        stream
            .alloc::<f32>(m * n_v)
            .map_err(|e| anyhow::anyhow!("alloc mrow scratch g_exp: {e:?}"))?
    };
    let beta = unsafe {
        stream
            .alloc::<f32>(m * n_v)
            .map_err(|e| anyhow::anyhow!("alloc mrow scratch beta: {e:?}"))?
    };
    let core = unsafe {
        stream
            .alloc::<half::bf16>(m * value_dim)
            .map_err(|e| anyhow::anyhow!("alloc mrow scratch core: {e:?}"))?
    };
    Ok(
        MrowScratchLeaseReturnsToThePoolOnDropSoNoPathFreesGraphBakedPointers(Some(
            MrowVerifyScratch {
                m,
                conv_dim,
                key_dim,
                value_dim,
                n_v,
                mixed_seq,
                qn,
                kn,
                g_exp,
                beta,
                core,
            },
        )),
    )
}

#[cfg(feature = "cuda")]
fn ckpt_rows_contiguous_base(
    ckpts: &[LinAttnState],
    conv_row_elems: usize,
    rec_row_elems: usize,
) -> Result<Option<(u64, u64)>> {
    let Some(first) = ckpts.first() else {
        return Ok(None);
    };
    let (_dc, conv_base, _lc) = cuda_tensor_raw(&first.conv_state)?;
    let (_dr, rec_base, _lr) = cuda_tensor_raw(&first.recurrent_state)?;
    let conv_row_bytes = (conv_row_elems * 2) as u64;
    let rec_row_bytes = (rec_row_elems * 4) as u64;
    for (j, ck) in ckpts.iter().enumerate() {
        let (_d1, cp, _l1) = cuda_tensor_raw(&ck.conv_state)?;
        let (_d2, rp, _l2) = cuda_tensor_raw(&ck.recurrent_state)?;
        if cp != conv_base + j as u64 * conv_row_bytes
            || rp != rec_base + j as u64 * rec_row_bytes
        {
            return Ok(None);
        }
    }
    Ok(Some((conv_base, rec_base)))
}

pub struct LinearAttention {
    cfg: LinearAttentionConfig,
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_a: Linear,
    in_proj_b: Linear,
    conv1d_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm_weight: Tensor,
    out_proj: Linear,
    ab_concat_one_gemv_when_both_bf16_because_two_48_row_launches_paid_double_latency:
        Option<Linear>,
    #[cfg(feature = "cuda")]
    qkvz_concat_fp8_built_only_under_nv_q38_gdn_fuse_costing_a_duplicate_resident_copy:
        Option<QkvzConcatFp8DecodeArm>,
}

impl LinearAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: LinearAttentionConfig,
        in_proj_qkv: Linear,
        in_proj_z: Linear,
        in_proj_a: Linear,
        in_proj_b: Linear,
        conv1d_weight: Tensor,
        a_log: Tensor,
        dt_bias: Tensor,
        norm_weight: Tensor,
        out_proj: Linear,
    ) -> Result<Self> {
        if !cfg
            .linear_num_value_heads
            .is_multiple_of(cfg.linear_num_key_heads)
        {
            anyhow::bail!(
                "LinearAttention: num_v_heads {} must be a multiple of num_k_heads {}",
                cfg.linear_num_value_heads,
                cfg.linear_num_key_heads,
            );
        }
        let expected_qkv = cfg.conv_dim();
        if in_proj_qkv.out_features() != expected_qkv {
            anyhow::bail!(
                "in_proj_qkv: expected out {}, got {}",
                expected_qkv,
                in_proj_qkv.out_features()
            );
        }
        if in_proj_z.out_features() != cfg.value_dim() {
            anyhow::bail!(
                "in_proj_z: expected out {}, got {}",
                cfg.value_dim(),
                in_proj_z.out_features()
            );
        }
        let n_v = cfg.linear_num_value_heads;
        if in_proj_a.out_features() != n_v || in_proj_b.out_features() != n_v {
            anyhow::bail!(
                "in_proj_a/b: expected out {}, got a={}, b={}",
                n_v,
                in_proj_a.out_features(),
                in_proj_b.out_features()
            );
        }
        let cdims = conv1d_weight.dims();
        if cdims.len() != 3
            || cdims[0] != cfg.conv_dim()
            || cdims[1] != 1
            || cdims[2] != cfg.linear_conv_kernel_dim
        {
            anyhow::bail!(
                "conv1d_weight: expected [{}, 1, {}], got {:?}",
                cfg.conv_dim(),
                cfg.linear_conv_kernel_dim,
                cdims
            );
        }
        let a_dims = a_log.dims();
        if a_dims.len() != 1 || a_dims[0] != n_v {
            anyhow::bail!("a_log: expected [{}], got {:?}", n_v, a_dims);
        }
        let dt_dims = dt_bias.dims();
        if dt_dims.len() != 1 || dt_dims[0] != n_v {
            anyhow::bail!("dt_bias: expected [{}], got {:?}", n_v, dt_dims);
        }
        let nw = norm_weight.dims();
        if nw.len() != 1 || nw[0] != cfg.linear_value_head_dim {
            anyhow::bail!(
                "norm_weight: expected [{}], got {:?}",
                cfg.linear_value_head_dim,
                nw
            );
        }
        if out_proj.in_features() != cfg.value_dim() || out_proj.out_features() != cfg.hidden_size {
            anyhow::bail!(
                "out_proj: expected [{}, {}], got [{}, {}]",
                cfg.hidden_size,
                cfg.value_dim(),
                out_proj.out_features(),
                out_proj.in_features()
            );
        }
        let ab_concat = match (in_proj_a.weight(), in_proj_b.weight()) {
            (Some(wa), Some(wb))
                if wa.dtype() == DType::BF16
                    && wb.dtype() == DType::BF16
                    && in_proj_a.bias().is_none()
                    && in_proj_b.bias().is_none() =>
            {
                let cat = Tensor::cat(&[wa, wb], 0)?.contiguous()?;
                Some(Linear::new(cat, None)?)
            }
            _ => None,
        };
        Ok(Self {
            cfg,
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            conv1d_weight,
            a_log,
            dt_bias,
            norm_weight,
            out_proj,
            ab_concat_one_gemv_when_both_bf16_because_two_48_row_launches_paid_double_latency:
                ab_concat,
            #[cfg(feature = "cuda")]
            qkvz_concat_fp8_built_only_under_nv_q38_gdn_fuse_costing_a_duplicate_resident_copy:
                None,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn install_qkvz_concat_fp8_decode_arm(
        &mut self,
        weight_bytes: &[u8],
        row_scales: &[f32],
        device: &Device,
    ) -> Result<()> {
        let rows = self.cfg.conv_dim() + self.cfg.value_dim();
        let k = self.cfg.hidden_size;
        anyhow::ensure!(
            weight_bytes.len() == rows * k,
            "qkvz concat: {} weight bytes != rows {rows} x k {k}",
            weight_bytes.len()
        );
        anyhow::ensure!(
            row_scales.len() == rows,
            "qkvz concat: {} row scales != rows {rows}",
            row_scales.len()
        );
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("qkvz concat requires a CUDA device"),
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        #[allow(deprecated)]
        let weight_u8 = stream
            .clone_htod(weight_bytes)
            .map_err(|e| anyhow::anyhow!("qkvz concat weight htod: {e:?}"))?;
        #[allow(deprecated)]
        let row_scales_dev = stream
            .clone_htod(row_scales)
            .map_err(|e| anyhow::anyhow!("qkvz concat scales htod: {e:?}"))?;
        self.qkvz_concat_fp8_built_only_under_nv_q38_gdn_fuse_costing_a_duplicate_resident_copy =
            Some(QkvzConcatFp8DecodeArm {
                weight_u8,
                row_scales_dev,
            });
        Ok(())
    }

    pub fn from_loader(
        cfg: LinearAttentionConfig,
        prefix: &str,
        weights: &WeightLoader,
        dtype: DType,
    ) -> Result<Self> {
        Self::from_loader_with_projection_loader(cfg, prefix, weights, dtype, &|w, m, o, i| {
            load_linear_else_fp8_rowscale_dequant_because_q38_mixed_checkpoints_store_gdn_qkv_z_out_as_f8e4m3(
                w, m, o, i, dtype,
            )
        })
    }

    #[cfg(feature = "cuda")]
    pub fn from_loader_fp8_projections_resident_1_byte_per_param(
        cfg: LinearAttentionConfig,
        prefix: &str,
        weights: &WeightLoader,
        dtype: DType,
        device: &Device,
        fp8_runner: &std::sync::Arc<std::sync::Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    ) -> Result<Self> {
        let mut la =
            Self::from_loader_with_projection_loader(cfg, prefix, weights, dtype, &|w, m, o, i| {
                load_linear_else_fp8_rowscale_resident_because_gdn_decode_bandwidth_wants_1_byte_per_param(
                    w, m, o, i, dtype, device, fp8_runner,
                )
            })?;
        if gdn_fuse_env_read_per_call_so_the_kill_switch_works_mid_process() {
            let conv_dim = la.cfg.conv_dim();
            let value_dim = la.cfg.value_dim();
            let hidden = la.cfg.hidden_size;
            if let Some((bytes, scales)) = qkvz_concat_raw_fp8_if_both_modules_are_fp8(
                weights, prefix, conv_dim, value_dim, hidden,
            )? {
                la.install_qkvz_concat_fp8_decode_arm(&bytes, &scales, device)?;
            }
        }
        Ok(la)
    }

    #[cfg(feature = "cuda")]
    pub fn from_loader_bf16_checkpoint_projections_quantized_to_fp8_resident_halving_gdn_decode_weight_traffic(
        cfg: LinearAttentionConfig,
        prefix: &str,
        weights: &WeightLoader,
        dtype: DType,
        device: &Device,
        fp8_runner: &std::sync::Arc<std::sync::Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    ) -> Result<Self> {
        Self::from_loader_with_projection_loader(cfg, prefix, weights, dtype, &|w, m, o, i| {
            load_linear_fp8_resident_quantizing_bf16_checkpoint_modules_because_qwen36_ships_gdn_qkv_z_out_as_bf16(
                w, m, o, i, device, fp8_runner,
            )
        })
    }

    fn from_loader_with_projection_loader(
        cfg: LinearAttentionConfig,
        prefix: &str,
        weights: &WeightLoader,
        dtype: DType,
        load_proj: &dyn Fn(&WeightLoader, &str, usize, usize) -> Result<Linear>,
    ) -> Result<Self> {
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();
        let conv_dim = cfg.conv_dim();
        let n_v = cfg.linear_num_value_heads;

        let in_proj_qkv = load_proj(
            weights,
            &format!("{prefix}.in_proj_qkv"),
            conv_dim,
            cfg.hidden_size,
        )?;
        let in_proj_z = load_proj(
            weights,
            &format!("{prefix}.in_proj_z"),
            value_dim,
            cfg.hidden_size,
        )?;
        let in_proj_a = load_linear(
            weights,
            &format!("{prefix}.in_proj_a.weight"),
            n_v,
            cfg.hidden_size,
            dtype,
        )?;
        let in_proj_b = load_linear(
            weights,
            &format!("{prefix}.in_proj_b.weight"),
            n_v,
            cfg.hidden_size,
            dtype,
        )?;
        let conv1d_weight = load_tensor(
            weights,
            &format!("{prefix}.conv1d.weight"),
            &[conv_dim, 1, cfg.linear_conv_kernel_dim],
            dtype,
        )?;
        let a_log = load_tensor(weights, &format!("{prefix}.A_log"), &[n_v], dtype)?;
        let dt_bias = load_tensor(weights, &format!("{prefix}.dt_bias"), &[n_v], dtype)?;
        let norm_weight = load_tensor(
            weights,
            &format!("{prefix}.norm.weight"),
            &[cfg.linear_value_head_dim],
            dtype,
        )?;
        let out_proj = load_proj(
            weights,
            &format!("{prefix}.out_proj"),
            cfg.hidden_size,
            value_dim,
        )?;
        let _ = key_dim;

        Self::new(
            cfg,
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            conv1d_weight,
            a_log,
            dt_bias,
            norm_weight,
            out_proj,
        )
    }

    pub fn config(&self) -> &LinearAttentionConfig {
        &self.cfg
    }

    pub fn forward_with_state(
        &self,
        x: &Tensor,
        state: &mut Option<LinAttnState>,
    ) -> Result<Tensor> {
        let (out, _) = self.forward_with_state_capture(x, state, false)?;
        Ok(out)
    }

    pub fn forward_with_state_capture(
        &self,
        x: &Tensor,
        state: &mut Option<LinAttnState>,
        capture: bool,
    ) -> Result<(Tensor, Vec<LinAttnState>)> {
        let input_dtype = x.dtype();
        let dims = x.dims().to_vec();
        if dims.len() != 3 {
            anyhow::bail!(
                "LinearAttention::forward_with_state: expected [B, T, H], got {:?}",
                dims
            );
        }
        let (b, t, h) = (dims[0], dims[1], dims[2]);
        if h != self.cfg.hidden_size {
            anyhow::bail!(
                "LinearAttention: hidden {} != cfg {}",
                h,
                self.cfg.hidden_size
            );
        }
        let n_k = self.cfg.linear_num_key_heads;
        let n_v = self.cfg.linear_num_value_heads;
        let d_k = self.cfg.linear_key_head_dim;
        let d_v = self.cfg.linear_value_head_dim;
        let key_dim = self.cfg.key_dim();
        let value_dim = self.cfg.value_dim();
        let conv_dim = self.cfg.conv_dim();
        let kernel = self.cfg.linear_conv_kernel_dim;
        let acc_dtype = self.cfg.mamba_ssm_dtype;
        let pad_left = kernel - 1;

        let prof_enabled = std::env::var("NV_PROF_LINATTN").is_ok();
        #[cfg(feature = "cuda")]
        let prof_dev: Option<candle_core::CudaDevice> = if prof_enabled {
            match x.device() {
                Device::Cuda(d) => Some(d.clone()),
                _ => None,
            }
        } else {
            None
        };
        let mut prof: Vec<(&'static str, f64)> = Vec::new();
        let prof_sync = || {
            #[cfg(feature = "cuda")]
            if let Some(dev) = &prof_dev {
                let _ = dev.cuda_stream().synchronize();
            }
        };
        macro_rules! tic {
            () => {{
                prof_sync();
                std::time::Instant::now()
            }};
        }
        macro_rules! toc {
            ($label:expr, $t0:expr) => {
                if prof_enabled {
                    prof_sync();
                    prof.push(($label, $t0.elapsed().as_secs_f64() * 1000.0));
                }
            };
        }

        let t0 = tic!();
        let qkv = self.in_proj_qkv.forward(x)?;
        let z = self.in_proj_z.forward(x)?;
        let a = self.in_proj_a.forward(x)?;
        let b_proj = self.in_proj_b.forward(x)?;
        toc!("in_proj_qkv_z_a_b", t0);

        let t0 = tic!();
        let qkv_bt = qkv.reshape((b, t, conv_dim))?;

        let prev_conv = state.as_ref().map(|s| s.conv_state.clone());
        let prev_recurrent = state.as_ref().map(|s| s.recurrent_state.clone());

        #[cfg(feature = "cuda")]
        let conv_bt_direct = if capture {
            None
        } else {
            self.run_conv_bt_silu_cuda_if_chunk_prefill(&qkv_bt, prev_conv.as_ref(), b, t)?
        };
        #[cfg(not(feature = "cuda"))]
        let conv_bt_direct: Option<(Tensor, Option<Tensor>)> = None;
        toc!("conv_bt_direct", t0);

        let t0 = tic!();
        let (mixed, conv_in_for_capture, new_conv_state) = match conv_bt_direct {
            Some((mx, cs)) => (mx, None, cs),
            None => {
                let qkv_btc = qkv_bt.transpose(1, 2)?.contiguous()?;
                let left = match &prev_conv {
                    Some(prev) => prev.to_dtype(qkv_btc.dtype())?,
                    None => Tensor::zeros(
                        (b, conv_dim, pad_left),
                        qkv_btc.dtype(),
                        qkv_btc.device(),
                    )?,
                };
                let conv_in = if pad_left > 0 {
                    Tensor::cat(&[&left, &qkv_btc], 2)?.contiguous()?
                } else {
                    qkv_btc.clone()
                };
                let new_conv_state = if pad_left > 0 {
                    let total_w = conv_in.dim(2)?;
                    Some(
                        conv_in
                            .narrow(2, total_w - pad_left, pad_left)?
                            .contiguous()?,
                    )
                } else {
                    None
                };
                let mixed = self.run_conv_transposed_layout(&conv_in, b, t)?;
                (mixed, Some(conv_in), new_conv_state)
            }
        };
        toc!("conv1d_silu", t0);

        let t0 = tic!();
        let (g_exp, beta) = compute_gdn_gating(&a, &b_proj, &self.a_log, &self.dt_bias, acc_dtype)?;
        toc!("gdn_gating", t0);

        #[cfg(feature = "cuda")]
        let fused_flat_state = if capture {
            None
        } else {
            let t0 = tic!();
            let r = self.run_fused_chunk_prefill_scan_norm_gate_if_enabled(
                &mixed,
                &z,
                &g_exp,
                &beta,
                b,
                t,
                prev_recurrent.as_ref(),
            )?;
            toc!("fused_qknorm_scan_rmsgate", t0);
            r
        };
        #[cfg(not(feature = "cuda"))]
        let fused_flat_state: Option<(Tensor, Tensor)> = None;

        let mut rec_ckpts: Vec<Tensor> = Vec::new();
        let (flat, final_state) = match fused_flat_state {
            Some(pair) => pair,
            None => {
                let t0 = tic!();
                let q_flat = mixed.narrow(2, 0, key_dim)?;
                let k_flat = mixed.narrow(2, key_dim, key_dim)?;
                let v_flat = mixed.narrow(2, 2 * key_dim, value_dim)?;

                let q = q_flat.reshape((b, t, n_k, d_k))?;
                let k = k_flat.reshape((b, t, n_k, d_k))?;
                let v = v_flat.reshape((b, t, n_v, d_v))?;

                let v_per_k = self.cfg.v_per_k();
                let q_grp = if v_per_k == 1 {
                    q
                } else {
                    q.unsqueeze(3)?
                        .expand((b, t, n_k, v_per_k, d_k))?
                        .contiguous()?
                        .reshape((b, t, n_v, d_k))?
                };
                let k_grp = if v_per_k == 1 {
                    k
                } else {
                    k.unsqueeze(3)?
                        .expand((b, t, n_k, v_per_k, d_k))?
                        .contiguous()?
                        .reshape((b, t, n_v, d_k))?
                };

                let q_grp = l2_normalize_last(&q_grp.to_dtype(acc_dtype)?)?;
                let k_grp = l2_normalize_last(&k_grp.to_dtype(acc_dtype)?)?;

                let scale = 1.0 / (d_k as f64).sqrt();
                let q_scaled = q_grp.affine(scale, 0.0)?;
                let k_f = k_grp;
                let v_f = v.to_dtype(acc_dtype)?;
                toc!("glue_pre_split_group_l2norm", t0);

                let t0 = tic!();
                let (core, final_state) = run_gdn_scan_candle_stateful(
                    &q_scaled,
                    &k_f,
                    &v_f,
                    &g_exp,
                    &beta,
                    b,
                    t,
                    n_v,
                    d_k,
                    d_v,
                    acc_dtype,
                    prev_recurrent.as_ref(),
                    if capture { Some(&mut rec_ckpts) } else { None },
                )?;
                toc!("scan_candle_token_sequential", t0);

                let t0 = tic!();
                let core_bf = core.to_dtype(input_dtype)?;
                let z_re = z.reshape((b, t, n_v, d_v))?;
                let normed = rmsnorm_last(&core_bf, &self.norm_weight, self.cfg.rms_eps)?;
                let gate =
                    candle_nn::ops::silu(&z_re.to_dtype(DType::F32)?)?.to_dtype(input_dtype)?;
                let gated = normed.mul(&gate)?;
                let flat = gated.reshape((b, t, value_dim))?;
                toc!("glue_post_rmsnorm_gate", t0);
                (flat, final_state)
            }
        };

        let t0 = tic!();
        let out = self.out_proj.forward(&flat)?;
        toc!("out_proj", t0);
        if prof_enabled {
            let total: f64 = prof.iter().map(|(_, v)| v).sum();
            let parts: Vec<String> = prof
                .iter()
                .map(|(l, v)| format!("{l}={v:.3}"))
                .collect();
            eprintln!(
                "[la_prof_prefill] b={b} t={t} total_ms={total:.3} {}",
                parts.join(" ")
            );
        }

        let conv_state = match new_conv_state {
            Some(cs) => cs,
            None => Tensor::zeros((b, conv_dim, 0), qkv_bt.dtype(), qkv_bt.device())?,
        };
        *state = Some(LinAttnState {
            conv_state,
            recurrent_state: final_state,
            fused: false,
        });

        let checkpoints = if capture {
            let conv_in = conv_in_for_capture
                .as_ref()
                .expect("capture always takes the padded conv_in path");
            let mut cks = Vec::with_capacity(t);
            for (ti, rec) in rec_ckpts.into_iter().enumerate() {
                let consumed = ti + 1;
                let conv_ck = if pad_left > 0 {
                    conv_in.narrow(2, consumed, pad_left)?.contiguous()?
                } else {
                    Tensor::zeros((b, conv_dim, 0), qkv_bt.dtype(), qkv_bt.device())?
                };
                cks.push(LinAttnState {
                    conv_state: conv_ck,
                    recurrent_state: rec,
                    fused: false,
                });
            }
            cks
        } else {
            Vec::new()
        };

        Ok((out, checkpoints))
    }

    fn run_conv_transposed_layout(&self, conv_in: &Tensor, b: usize, t: usize) -> Result<Tensor> {
        let conv_dim = self.cfg.conv_dim();
        let kernel = self.cfg.linear_conv_kernel_dim;
        let pad_left = kernel - 1;
        let _ = (b, t, conv_dim, kernel, pad_left);
        #[cfg(feature = "cuda")]
        if matches!(conv_in.device(), Device::Cuda(_))
            && conv_in.dtype() == DType::BF16
            && self.conv1d_weight.dtype() == DType::BF16
        {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let dev = match conv_in.device() {
                Device::Cuda(d) => d.clone(),
                _ => unreachable!(),
            };
            let stream = crate::cuda_stream::current_stream(&dev);
            let t_in = conv_in.dim(2)?;
            let mut y_dev: cudarc::driver::CudaSlice<half::bf16> =
                unsafe { stream.alloc::<half::bf16>(b * conv_dim * t_in)? };
            let conv_in_c = conv_in.contiguous()?;
            let w_c = self.conv1d_weight.contiguous()?;
            {
                let (xs, _xl) = conv_in_c.storage_and_layout();
                let (ws, _wl) = w_c.storage_and_layout();
                let x_cuda = match &*xs {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("expected cuda storage"),
                };
                let w_cuda = match &*ws {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("expected cuda storage"),
                };
                let x_slice = x_cuda.as_cuda_slice::<half::bf16>()?;
                let w_slice = w_cuda.as_cuda_slice::<half::bf16>()?;
                let (xp, _g1) = x_slice.device_ptr(&stream);
                let (wp, _g2) = w_slice.device_ptr(&stream);
                let (yp, _g3) = y_dev.device_ptr_mut(&stream);
                let rc = unsafe {
                    nv_kernels::cuda::depthwise_conv1d_silu_bf16(
                        stream.cu_stream() as *mut std::ffi::c_void,
                        xp as *const u16,
                        wp as *const u16,
                        yp as *mut u16,
                        b as i32,
                        conv_dim as i32,
                        t_in as i32,
                        kernel as i32,
                    )
                };
                anyhow::ensure!(rc == 0, "depthwise_conv1d_silu_bf16 rc={rc}");
            }
            let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
            let y_bct = Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (b, conv_dim, t_in),
                candle_core::op::BackpropOp::none(),
                false,
            );
            return Ok(y_bct.narrow(2, pad_left, t)?.transpose(1, 2)?.contiguous()?);
        }
        let conv_out = conv_in.conv1d(&self.conv1d_weight, 0, 1, 1, conv_dim)?;
        let conv_out = candle_nn::ops::silu(&conv_out)?;
        Ok(conv_out.transpose(1, 2)?.contiguous()?)
    }

    #[cfg(feature = "cuda")]
    fn run_conv_bt_silu_cuda_if_chunk_prefill(
        &self,
        qkv_bt: &Tensor,
        prev_conv: Option<&Tensor>,
        b: usize,
        t: usize,
    ) -> Result<Option<(Tensor, Option<Tensor>)>> {
        use cudarc::driver::DevicePtrMut;
        use nv_kernels::cuda as nvk;

        if !chunk_prefill_env_read_per_call_so_one_process_can_ab_both_scan_paths() {
            return Ok(None);
        }
        let conv_dim = self.cfg.conv_dim();
        let kernel = self.cfg.linear_conv_kernel_dim;
        let pad_left = kernel - 1;
        if !(2..=8).contains(&kernel) || b * t == 0 {
            return Ok(None);
        }
        let dev = match qkv_bt.device() {
            Device::Cuda(d) => d.clone(),
            _ => return Ok(None),
        };
        if qkv_bt.dtype() != DType::BF16 || self.conv1d_weight.dtype() != DType::BF16 {
            return Ok(None);
        }
        if let Some(prev) = prev_conv {
            if prev.dtype() != DType::BF16
                || prev.dims() != [b, conv_dim, pad_left]
            {
                return Ok(None);
            }
        }
        let stream = crate::cuda_stream::current_stream(&dev);
        let x_c = qkv_bt.contiguous()?;
        let w_c = self.conv1d_weight.contiguous()?;
        let (_d1, x_ptr, _l1) = cuda_tensor_raw(&x_c)?;
        let (_d2, w_ptr, _l2) = cuda_tensor_raw(&w_c)?;
        let prev_c = match prev_conv {
            Some(p) => Some(p.contiguous()?),
            None => None,
        };
        let state_in_ptr: u64 = match &prev_c {
            Some(p) => cuda_tensor_raw(p)?.1,
            None => 0,
        };
        let mut y_dev: cudarc::driver::CudaSlice<half::bf16> = unsafe {
            stream
                .alloc::<half::bf16>(b * t * conv_dim)
                .map_err(|e| anyhow::anyhow!("alloc conv_bt mixed: {e:?}"))?
        };
        let mut state_out_dev: cudarc::driver::CudaSlice<half::bf16> = unsafe {
            stream
                .alloc::<half::bf16>(b * conv_dim * pad_left)
                .map_err(|e| anyhow::anyhow!("alloc conv_bt state_out: {e:?}"))?
        };
        {
            let (yp, _gy) = y_dev.device_ptr_mut(&stream);
            let (sp, _gs) = state_out_dev.device_ptr_mut(&stream);
            let rc = unsafe {
                nvk::gdn_conv1d_silu_bt_bf16(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    x_ptr as *const u16,
                    state_in_ptr as *const u16,
                    w_ptr as *const u16,
                    yp as *mut u16,
                    sp as *mut u16,
                    b as i32,
                    t as i32,
                    conv_dim as i32,
                    kernel as i32,
                )
            };
            anyhow::ensure!(rc == 0, "gdn_conv1d_silu_bt_bf16 rc={rc}");
        }
        let mixed = {
            let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (b, t, conv_dim),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        let new_conv_state = {
            let storage = candle_core::CudaStorage::wrap_cuda_slice(state_out_dev, dev);
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (b, conv_dim, pad_left),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        Ok(Some((mixed, Some(new_conv_state))))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn run_fused_chunk_prefill_scan_norm_gate_if_enabled(
        &self,
        mixed: &Tensor,
        z: &Tensor,
        g_exp: &Tensor,
        beta: &Tensor,
        b: usize,
        t: usize,
        prev_recurrent: Option<&Tensor>,
    ) -> Result<Option<(Tensor, Tensor)>> {
        use cudarc::driver::DevicePtrMut;
        use nv_kernels::cuda as nvk;

        if !chunk_prefill_env_read_per_call_so_one_process_can_ab_both_scan_paths() {
            return Ok(None);
        }
        let c = &self.cfg;
        if c.linear_key_head_dim != 128
            || c.linear_value_head_dim != 128
            || c.mamba_ssm_dtype != DType::F32
            || !c.linear_num_value_heads.is_multiple_of(c.linear_num_key_heads)
        {
            return Ok(None);
        }
        let dev = match mixed.device() {
            Device::Cuda(d) => d.clone(),
            _ => return Ok(None),
        };
        if mixed.dtype() != DType::BF16
            || z.dtype() != DType::BF16
            || self.norm_weight.dtype() != DType::BF16
        {
            return Ok(None);
        }
        let n_k = c.linear_num_key_heads;
        let n_v = c.linear_num_value_heads;
        let d_k = c.linear_key_head_dim;
        let d_v = c.linear_value_head_dim;
        let key_dim = c.key_dim();
        let value_dim = c.value_dim();
        let conv_dim = c.conv_dim();
        if b * t * n_v == 0 {
            return Ok(None);
        }
        let stream = crate::cuda_stream::current_stream(&dev);

        let mixed_c = mixed.reshape((b, t, conv_dim))?.contiguous()?;
        let z_c = z.reshape((b, t, value_dim))?.contiguous()?;
        let g_c = g_exp.to_dtype(DType::F32)?.contiguous()?;
        let beta_c = beta.to_dtype(DType::F32)?.contiguous()?;
        let nw_c = self.norm_weight.contiguous()?;

        let qk_elems = b * t * n_k * d_k;
        let mut qn_dev: cudarc::driver::CudaSlice<f32> = unsafe {
            stream
                .alloc::<f32>(qk_elems)
                .map_err(|e| anyhow::anyhow!("alloc fused qn: {e:?}"))?
        };
        let mut kn_dev: cudarc::driver::CudaSlice<f32> = unsafe {
            stream
                .alloc::<f32>(qk_elems)
                .map_err(|e| anyhow::anyhow!("alloc fused kn: {e:?}"))?
        };
        let state_elems = b * n_v * d_k * d_v;
        let mut state_dev: cudarc::driver::CudaSlice<f32> = stream
            .alloc_zeros::<f32>(state_elems)
            .map_err(|e| anyhow::anyhow!("alloc fused scan state: {e:?}"))?;
        if let Some(prev) = prev_recurrent {
            let prev_c = prev.to_dtype(DType::F32)?.contiguous()?;
            anyhow::ensure!(
                prev_c.elem_count() == state_elems,
                "fused chunk scan init state has {} elems, want {state_elems}",
                prev_c.elem_count()
            );
            let (_pd, prev_ptr, bytes) = cuda_tensor_raw(&prev_c)?;
            let (sp, _gs) = state_dev.device_ptr_mut(&stream);
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(sp, prev_ptr, bytes, stream.cu_stream())
                    .map_err(|e| anyhow::anyhow!("fused chunk scan seed state dtod: {e:?}"))?;
            }
        }
        let mut core_dev: cudarc::driver::CudaSlice<half::bf16> = unsafe {
            stream
                .alloc::<half::bf16>(b * t * value_dim)
                .map_err(|e| anyhow::anyhow!("alloc fused core: {e:?}"))?
        };

        let (_d1, mixed_ptr, _l1) = cuda_tensor_raw(&mixed_c)?;
        let (_d2, z_ptr, _l2) = cuda_tensor_raw(&z_c)?;
        let (_d3, g_ptr, _l3) = cuda_tensor_raw(&g_c)?;
        let (_d4, beta_ptr, _l4) = cuda_tensor_raw(&beta_c)?;
        let (_d5, nw_ptr, _l5) = cuda_tensor_raw(&nw_c)?;

        {
            let (qp, _gq) = qn_dev.device_ptr_mut(&stream);
            let (kp, _gk) = kn_dev.device_ptr_mut(&stream);
            let rc = unsafe {
                nvk::gdn_prefill_qk_l2norm_from_mixed(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    mixed_ptr as *const u16,
                    qp as *mut f32,
                    kp as *mut f32,
                    (b * t) as i32,
                    n_k as i32,
                    conv_dim as i32,
                    key_dim as i32,
                    (1.0 / (d_k as f64).sqrt()) as f32,
                    GDN_L2_EPS_1E6_MATCHES_L2_NORMALIZE_LAST,
                )
            };
            anyhow::ensure!(rc == 0, "gdn_prefill_qk_l2norm_from_mixed rc={rc}");
        }
        {
            let (qp, _gq) = qn_dev.device_ptr_mut(&stream);
            let (kp, _gk) = kn_dev.device_ptr_mut(&stream);
            let (sp, _gs) = state_dev.device_ptr_mut(&stream);
            let (cp, _gc) = core_dev.device_ptr_mut(&stream);
            let rc = unsafe {
                nvk::gdn_recurrent_stateful_gqa_bf16out(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    qp as *const f32,
                    kp as *const f32,
                    mixed_ptr as *const u16,
                    g_ptr as *const f32,
                    beta_ptr as *const f32,
                    sp as *mut f32,
                    cp as *mut u16,
                    b as i32,
                    t as i32,
                    n_v as i32,
                    n_k as i32,
                    d_k as i32,
                    d_v as i32,
                    conv_dim as i32,
                    (2 * key_dim) as i32,
                )
            };
            anyhow::ensure!(rc == 0, "gdn_recurrent_stateful_gqa_bf16out rc={rc}");
        }
        {
            let (cp, _gc) = core_dev.device_ptr_mut(&stream);
            let rc = unsafe {
                nvk::gdn_prefill_rmsnorm_gate_bf16(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    cp as *const u16,
                    z_ptr as *const u16,
                    nw_ptr as *const u16,
                    cp as *mut u16,
                    (b * t * n_v) as i32,
                    d_v as i32,
                    self.cfg.rms_eps as f32,
                )
            };
            anyhow::ensure!(rc == 0, "gdn_prefill_rmsnorm_gate_bf16 rc={rc}");
        }

        let flat = {
            let storage = candle_core::CudaStorage::wrap_cuda_slice(core_dev, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (b, t, value_dim),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        let final_state = {
            let storage = candle_core::CudaStorage::wrap_cuda_slice(state_dev, dev);
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (b, n_v, d_k, d_v),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        Ok(Some((flat, final_state)))
    }

    #[cfg(feature = "cuda")]
    pub fn fused_decode_supported(&self) -> bool {
        if fused_decode_env() == Some(false) {
            return false;
        }
        let c = &self.cfg;
        c.mamba_ssm_dtype == DType::F32
            && c.linear_conv_kernel_dim >= 2
            && c.linear_value_head_dim % 32 == 0
            && c.linear_value_head_dim >= 32
            && c.linear_value_head_dim <= 1024
            && c.linear_key_head_dim >= 1
            && (2 * c.linear_key_head_dim + 32) * 4 <= 96 * 1024
            && self.conv1d_weight.dtype() == DType::BF16
            && self.a_log.dtype() == DType::BF16
            && self.dt_bias.dtype() == DType::BF16
            && self.norm_weight.dtype() == DType::BF16
    }

    #[cfg(feature = "cuda")]
    pub fn new_fused_state(&self, device: &Device) -> Result<LinAttnState> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("new_fused_state requires a CUDA device"),
        };
        anyhow::ensure!(self.fused_decode_supported(), "fused decode unsupported");
        let stream = crate::cuda_stream::current_stream(&dev);
        let conv_dim = self.cfg.conv_dim();
        let k = self.cfg.linear_conv_kernel_dim;
        let n_v = self.cfg.linear_num_value_heads;
        let d_k = self.cfg.linear_key_head_dim;
        let d_v = self.cfg.linear_value_head_dim;
        let conv_buf = stream
            .alloc_zeros::<half::bf16>(conv_dim * (k - 1))
            .map_err(|e| anyhow::anyhow!("alloc conv state: {e:?}"))?;
        let rec_buf = stream
            .alloc_zeros::<f32>(n_v * d_k * d_v)
            .map_err(|e| anyhow::anyhow!("alloc recurrent state: {e:?}"))?;
        let conv_state = {
            let storage = candle_core::CudaStorage::wrap_cuda_slice(conv_buf, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, conv_dim, k - 1),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        let recurrent_state = {
            let storage = candle_core::CudaStorage::wrap_cuda_slice(rec_buf, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, n_v, d_k, d_v),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        Ok(LinAttnState {
            conv_state,
            recurrent_state,
            fused: true,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn new_fused_verify_ckpt_rows_off_one_slab_so_chunk_kernels_write_ckpts_in_place(
        &self,
        device: &Device,
        rows: usize,
    ) -> Result<Vec<LinAttnState>> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("new_fused_verify_ckpt_rows requires a CUDA device"),
        };
        anyhow::ensure!(self.fused_decode_supported(), "fused decode unsupported");
        anyhow::ensure!(rows >= 1, "new_fused_verify_ckpt_rows: rows must be >= 1");
        let stream = crate::cuda_stream::current_stream(&dev);
        let conv_dim = self.cfg.conv_dim();
        let k = self.cfg.linear_conv_kernel_dim;
        let n_v = self.cfg.linear_num_value_heads;
        let d_k = self.cfg.linear_key_head_dim;
        let d_v = self.cfg.linear_value_head_dim;
        let conv_buf = stream
            .alloc_zeros::<half::bf16>(rows * conv_dim * (k - 1))
            .map_err(|e| anyhow::anyhow!("alloc verify conv ckpt slab: {e:?}"))?;
        let rec_buf = stream
            .alloc_zeros::<f32>(rows * n_v * d_k * d_v)
            .map_err(|e| anyhow::anyhow!("alloc verify rec ckpt slab: {e:?}"))?;
        let conv_slab = {
            let storage = candle_core::CudaStorage::wrap_cuda_slice(conv_buf, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (rows, conv_dim, k - 1),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        let rec_slab = {
            let storage = candle_core::CudaStorage::wrap_cuda_slice(rec_buf, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (rows, n_v, d_k, d_v),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        let mut out = Vec::with_capacity(rows);
        for j in 0..rows {
            out.push(LinAttnState {
                conv_state: conv_slab.narrow(0, j, 1)?,
                recurrent_state: rec_slab.narrow(0, j, 1)?,
                fused: true,
            });
        }
        let contiguous = ckpt_rows_contiguous_base(&out, conv_dim * (k - 1), n_v * d_k * d_v)?;
        anyhow::ensure!(
            contiguous.is_some(),
            "new_fused_verify_ckpt_rows: the slab narrows did not land at row stride; the chunk \
             kernels would write ckpt rows over each other"
        );
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_decode_fused(&self, x: &Tensor, state: &LinAttnState) -> Result<Option<Tensor>> {
        if !state.fused || !self.fused_decode_supported() {
            return Ok(None);
        }
        let dims = x.dims();
        if dims != [1, 1, self.cfg.hidden_size] || x.dtype() != DType::BF16 {
            return Ok(None);
        }
        let dev = match x.device() {
            Device::Cuda(d) => d.clone(),
            _ => return Ok(None),
        };

        if gdn_fuse_env_read_per_call_so_the_kill_switch_works_mid_process() {
            if let Some(concat) = &self
                .qkvz_concat_fp8_built_only_under_nv_q38_gdn_fuse_costing_a_duplicate_resident_copy
            {
                if let Some(out) = self.forward_decode_fused_qkvz_concat(x, state, concat, &dev)? {
                    return Ok(Some(out));
                }
            }
        }

        let n_v = self.cfg.linear_num_value_heads;
        let mut prof = gdn_step_prof::SyncLaps::begin_if_armed(&dev);
        let qkv = self.in_proj_qkv.forward(x)?.contiguous()?;
        if let Some(p) = prof.as_mut() {
            p.lap("in_qkv");
        }
        let z = self.in_proj_z.forward(x)?.contiguous()?;
        if let Some(p) = prof.as_mut() {
            p.lap("in_z");
        }
        let (a, b_proj) = match &self
            .ab_concat_one_gemv_when_both_bf16_because_two_48_row_launches_paid_double_latency
        {
            Some(ab) => {
                let y = ab.forward(x)?.contiguous()?;
                (y.narrow(D::Minus1, 0, n_v)?, y.narrow(D::Minus1, n_v, n_v)?)
            }
            None => (
                self.in_proj_a.forward(x)?.contiguous()?,
                self.in_proj_b.forward(x)?.contiguous()?,
            ),
        };
        if let Some(p) = prof.as_mut() {
            p.lap("in_ab");
        }
        if qkv.dtype() != DType::BF16
            || z.dtype() != DType::BF16
            || a.dtype() != DType::BF16
            || b_proj.dtype() != DType::BF16
        {
            return Ok(None);
        }
        self.fused_decode_conv_step_out_shared_by_plain_and_prenorm_folded_arms(
            state, &dev, &qkv, &z, &a, &b_proj, &mut prof,
        )
        .map(Some)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn fused_decode_conv_step_out_shared_by_plain_and_prenorm_folded_arms(
        &self,
        state: &LinAttnState,
        dev: &candle_core::CudaDevice,
        qkv: &Tensor,
        z: &Tensor,
        a: &Tensor,
        b_proj: &Tensor,
        prof: &mut Option<gdn_step_prof::SyncLaps>,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};

        let stream = crate::cuda_stream::current_stream(dev);
        let conv_dim = self.cfg.conv_dim();
        let value_dim = self.cfg.value_dim();
        let n_k = self.cfg.linear_num_key_heads;
        let n_v = self.cfg.linear_num_value_heads;
        let d_k = self.cfg.linear_key_head_dim;
        let d_v = self.cfg.linear_value_head_dim;
        let kernel = self.cfg.linear_conv_kernel_dim;

        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "[linear_attn] fused persistent-state decode path active \
                     (n_k={n_k} n_v={n_v} d_k={d_k} d_v={d_v} k={kernel})"
                );
            });
        }
        let (_d1, qkv_ptr, _l1) = cuda_tensor_raw(qkv)?;
        let (_d2, z_ptr, _l2) = cuda_tensor_raw(z)?;
        let (_d3, a_ptr, _l3) = cuda_tensor_raw(a)?;
        let (_d4, b_ptr, _l4) = cuda_tensor_raw(b_proj)?;
        let w_c = self.conv1d_weight.contiguous()?;
        let (_d5, w_ptr, _l5) = cuda_tensor_raw(&w_c)?;
        let (_d6, conv_state_ptr, _l6) = cuda_tensor_raw(&state.conv_state)?;
        let (_d7, rec_state_ptr, _l7) = cuda_tensor_raw(&state.recurrent_state)?;
        let a_log_c = self.a_log.contiguous()?;
        let dt_c = self.dt_bias.contiguous()?;
        let nw_c = self.norm_weight.contiguous()?;
        let (_d8, a_log_ptr, _l8) = cuda_tensor_raw(&a_log_c)?;
        let (_d9, dt_ptr, _l9) = cuda_tensor_raw(&dt_c)?;
        let (_d10, nw_ptr, _l10) = cuda_tensor_raw(&nw_c)?;

        let mut scratch_lease =
            gdn_decode_scratch_take_or_build(dev, conv_dim, value_dim, n_k * d_k, n_v)?;
        let scratch = &mut *scratch_lease;
        if let Some(p) = prof.as_mut() {
            p.lap("glue_ptrs_allocs");
        }

        {
            let (mp, _gm) = scratch.mixed.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::gdn_conv_decode_silu_bf16(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    qkv_ptr as *const u16,
                    conv_state_ptr as *mut u16,
                    w_ptr as *const u16,
                    mp as *mut u16,
                    conv_dim as i32,
                    kernel as i32,
                )
            };
            anyhow::ensure!(rc == 0, "gdn_conv_decode_silu_bf16 rc={rc}");
        }
        if let Some(p) = prof.as_mut() {
            p.lap("conv");
        }
        let mut split_step_taken = false;
        let (_gdev, gated_ptr, _glen) =
            cuda_tensor_raw(&scratch.gated_tensor_reused_across_layers_because_the_stream_serializes)?;
        if gdn_split_env_read_per_call_so_the_kill_switch_works_mid_process() {
            if let Some(p) = prof.as_mut() {
                p.lap("step_allocs");
            }
            let (mp, _gm) = scratch.mixed.device_ptr(&stream);
            let gp = gated_ptr;
            let (qnp, _gqn) = scratch.qn.device_ptr_mut(&stream);
            let (knp, _gkn) = scratch.kn.device_ptr_mut(&stream);
            let (gep, _gge) = scratch.g_exp.device_ptr_mut(&stream);
            let (bep, _gbe) = scratch.beta.device_ptr_mut(&stream);
            let (cop, _gco) = scratch.core.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::gdn_decode_step_split_bf16(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    mp as *const u16,
                    z_ptr as *const u16,
                    a_ptr as *const u16,
                    b_ptr as *const u16,
                    a_log_ptr as *const u16,
                    dt_ptr as *const u16,
                    nw_ptr as *const u16,
                    rec_state_ptr as *mut f32,
                    gp as *mut u16,
                    qnp as *mut f32,
                    knp as *mut f32,
                    gep as *mut f32,
                    bep as *mut f32,
                    cop as *mut u16,
                    n_k as i32,
                    n_v as i32,
                    d_k as i32,
                    d_v as i32,
                    self.cfg.rms_eps as f32,
                )
            };
            if rc == 0 {
                split_step_taken = true;
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    eprintln!(
                        "[linear_attn] NV_Q38_GDN_SPLIT step path active \
                         (prep+regstate-vsplit+gate, n_v={n_v} d_k={d_k} d_v={d_v})"
                    );
                });
            } else {
                anyhow::ensure!(rc == -1, "gdn_decode_step_split_bf16 rc={rc}");
            }
        }
        if !split_step_taken {
            let (mp, _gm) = scratch.mixed.device_ptr(&stream);
            let gp = gated_ptr;
            let rc = unsafe {
                nv_kernels::cuda::gdn_decode_step_bf16(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    mp as *const u16,
                    z_ptr as *const u16,
                    a_ptr as *const u16,
                    b_ptr as *const u16,
                    a_log_ptr as *const u16,
                    dt_ptr as *const u16,
                    nw_ptr as *const u16,
                    rec_state_ptr as *mut f32,
                    gp as *mut u16,
                    n_k as i32,
                    n_v as i32,
                    d_k as i32,
                    d_v as i32,
                    self.cfg.rms_eps as f32,
                )
            };
            anyhow::ensure!(rc == 0, "gdn_decode_step_bf16 rc={rc}");
        }
        if let Some(p) = prof.as_mut() {
            p.lap("step");
        }

        let flat = scratch
            .gated_tensor_reused_across_layers_because_the_stream_serializes
            .clone();
        drop(scratch_lease);
        let out = self.out_proj.forward(&flat)?;
        if let Some(p) = prof.as_mut() {
            p.lap("out_proj");
        }
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_decode_fused_batch_lanes_projections_once_then_per_lane_step_kernels(
        &self,
        x: &Tensor,
        states: &[&LinAttnState],
    ) -> Result<Option<Tensor>> {
        self.forward_decode_fused_batch_lanes_projections_once_then_per_lane_step_kernels_prof(
            x, states, None, None,
        )
    }

    #[cfg(feature = "cuda")]
    pub fn forward_decode_fused_batch_lanes_projections_once_then_per_lane_step_kernels_prof(
        &self,
        x: &Tensor,
        states: &[&LinAttnState],
        mut prof: Option<&mut gdn_step_prof::SyncLaps>,
        lane_streams_index_0_is_the_main_stream_conc_arm_also_selects_the_mrow_ab_gemm: Option<
            &[std::sync::Arc<cudarc::driver::CudaStream>],
        >,
    ) -> Result<Option<Tensor>> {
        use cudarc::driver::DevicePtrMut;

        if !self.fused_decode_supported() {
            return Ok(None);
        }
        let m = states.len();
        let dims = x.dims();
        if m == 0
            || dims != [1, m, self.cfg.hidden_size]
            || x.dtype() != DType::BF16
            || states.iter().any(|st| !st.fused)
        {
            return Ok(None);
        }
        let dev = match x.device() {
            Device::Cuda(d) => d.clone(),
            _ => return Ok(None),
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        let conv_dim = self.cfg.conv_dim();
        let value_dim = self.cfg.value_dim();
        let n_k = self.cfg.linear_num_key_heads;
        let n_v = self.cfg.linear_num_value_heads;
        let d_k = self.cfg.linear_key_head_dim;
        let d_v = self.cfg.linear_value_head_dim;
        let kernel = self.cfg.linear_conv_kernel_dim;
        let key_dim = n_k * d_k;

        let x_c = x.contiguous()?;
        let qkv = self.in_proj_qkv.forward(&x_c)?.contiguous()?;
        let z = self.in_proj_z.forward(&x_c)?.contiguous()?;
        if let Some(p) = prof.as_deref_mut() {
            p.lap("b.gdn.qkvz_proj");
        }
        let ab_stride = 2 * n_v;
        let a_off = 0usize;
        let b_off = n_v;
        let conc_arm =
            lane_streams_index_0_is_the_main_stream_conc_arm_also_selects_the_mrow_ab_gemm
                .filter(|ls| m > 1 && ls.len() >= m);
        let mrow_ab_subarm_opt_in_nv_q38_batch_gemm_ab_1_because_its_gemm_deltas_cascade_through_the_gdn_state_and_flipped_argmax_at_m2_where_the_route_was_clean =
            std::env::var("NV_Q38_BATCH_GEMM_AB").ok().as_deref() == Some("1");
        let ab_holder = if conc_arm.is_some()
            && mrow_ab_subarm_opt_in_nv_q38_batch_gemm_ab_1_because_its_gemm_deltas_cascade_through_the_gdn_state_and_flipped_argmax_at_m2_where_the_route_was_clean
        {
            let one_mrow_gemm_not_a_gemv_row_twin_acceptable_under_the_argmax_class_serving_bar =
                match &self
                    .ab_concat_one_gemv_when_both_bf16_because_two_48_row_launches_paid_double_latency
                {
                    Some(ab) => ab.forward(&x_c)?,
                    None => {
                        let a_t = self.in_proj_a.forward(&x_c)?;
                        let b_t = self.in_proj_b.forward(&x_c)?;
                        Tensor::cat(&[&a_t, &b_t], candle_core::D::Minus1)?
                    }
                };
            one_mrow_gemm_not_a_gemv_row_twin_acceptable_under_the_argmax_class_serving_bar
                .contiguous()?
        } else {
            let mut rows: Vec<Tensor> = Vec::with_capacity(m);
            for lane in 0..m {
                let x_row = x_c.narrow(1, lane, 1)?;
                let row = match &self
                    .ab_concat_one_gemv_when_both_bf16_because_two_48_row_launches_paid_double_latency
                {
                    Some(ab) => ab.forward(&x_row)?,
                    None => {
                        let a_t = self.in_proj_a.forward(&x_row)?;
                        let b_t = self.in_proj_b.forward(&x_row)?;
                        Tensor::cat(&[&a_t, &b_t], candle_core::D::Minus1)?
                    }
                };
                rows.push(row);
            }
            let refs: Vec<&Tensor> = rows.iter().collect();
            let per_row_m1_gemv_because_bf16_ab_has_no_m_row_twin_and_96_rows_by_hidden_is_cheap =
                Tensor::cat(&refs, 1)?.contiguous()?;
            per_row_m1_gemv_because_bf16_ab_has_no_m_row_twin_and_96_rows_by_hidden_is_cheap
        };
        if let Some(p) = prof.as_deref_mut() {
            p.lap("b.gdn.ab_proj");
        }
        if qkv.dtype() != DType::BF16
            || z.dtype() != DType::BF16
            || ab_holder.dtype() != DType::BF16
        {
            return Ok(None);
        }
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "[linear_attn] batch-lanes fused decode active: one qkv/z/ab GEMM over m \
                     lanes then per-lane conv+step kernels (m={m} n_k={n_k} n_v={n_v} d_k={d_k} \
                     d_v={d_v} k={kernel})"
                );
            });
        }

        let (_d1, qkv_ptr, _l1) = cuda_tensor_raw(&qkv)?;
        let (_d2, z_ptr, _l2) = cuda_tensor_raw(&z)?;
        let (_d3, ab_ptr, _l3) = cuda_tensor_raw(&ab_holder)?;
        let w_c = self.conv1d_weight.contiguous()?;
        let (_d4, w_ptr, _l4) = cuda_tensor_raw(&w_c)?;
        let a_log_c = self.a_log.contiguous()?;
        let dt_c = self.dt_bias.contiguous()?;
        let nw_c = self.norm_weight.contiguous()?;
        let (_d5, a_log_ptr, _l5) = cuda_tensor_raw(&a_log_c)?;
        let (_d6, dt_ptr, _l6) = cuda_tensor_raw(&dt_c)?;
        let (_d7, nw_ptr, _l7) = cuda_tensor_raw(&nw_c)?;

        let mut mixed: cudarc::driver::CudaSlice<half::bf16> = unsafe {
            stream
                .alloc::<half::bf16>(m * conv_dim)
                .map_err(|e| anyhow::anyhow!("alloc batch mixed: {e:?}"))?
        };
        let mut gated: cudarc::driver::CudaSlice<half::bf16> = unsafe {
            stream
                .alloc::<half::bf16>(m * value_dim)
                .map_err(|e| anyhow::anyhow!("alloc batch gated: {e:?}"))?
        };
        let mut qn_s: cudarc::driver::CudaSlice<f32> = unsafe {
            stream
                .alloc::<f32>(key_dim)
                .map_err(|e| anyhow::anyhow!("alloc batch qn: {e:?}"))?
        };
        let mut kn_s: cudarc::driver::CudaSlice<f32> = unsafe {
            stream
                .alloc::<f32>(key_dim)
                .map_err(|e| anyhow::anyhow!("alloc batch kn: {e:?}"))?
        };
        let mut ge_s: cudarc::driver::CudaSlice<f32> = unsafe {
            stream
                .alloc::<f32>(n_v)
                .map_err(|e| anyhow::anyhow!("alloc batch g_exp: {e:?}"))?
        };
        let mut be_s: cudarc::driver::CudaSlice<f32> = unsafe {
            stream
                .alloc::<f32>(n_v)
                .map_err(|e| anyhow::anyhow!("alloc batch beta: {e:?}"))?
        };
        let mut core_s: cudarc::driver::CudaSlice<half::bf16> = unsafe {
            stream
                .alloc::<half::bf16>(value_dim)
                .map_err(|e| anyhow::anyhow!("alloc batch core: {e:?}"))?
        };
        let (mixed_base, _gm) = mixed.device_ptr_mut(&stream);
        let (gated_base, _gg) = gated.device_ptr_mut(&stream);
        let (qn_ptr, _gq) = qn_s.device_ptr_mut(&stream);
        let (kn_ptr, _gk) = kn_s.device_ptr_mut(&stream);
        let (ge_ptr, _gge) = ge_s.device_ptr_mut(&stream);
        let (be_ptr, _gbe) = be_s.device_ptr_mut(&stream);
        let (core_ptr, _gco) = core_s.device_ptr_mut(&stream);
        let split_env = gdn_split_env_read_per_call_so_the_kill_switch_works_mid_process();
        let conc_steps = conc_arm.filter(|_| {
            let scratch_free_non_split_step_writes_only_lane_private_buffers = !split_env;
            scratch_free_non_split_step_writes_only_lane_private_buffers
        });
        if let Some(ls) = conc_steps {
            let ev = stream
                .record_event(None)
                .map_err(|e| anyhow::anyhow!("gdn lane fork event: {e:?}"))?;
            for s_i in ls[1..m].iter() {
                s_i.wait(&ev)
                    .map_err(|e| anyhow::anyhow!("gdn lane fork wait: {e:?}"))?;
            }
        }
        for (lane, st) in states.iter().enumerate() {
            let lane_cu = match conc_steps {
                Some(ls) if lane > 0 => ls[lane].cu_stream(),
                _ => stream.cu_stream(),
            };
            let (_dc, conv_state_ptr, _lc) = cuda_tensor_raw(&st.conv_state)?;
            let (_dr, rec_state_ptr, _lr) = cuda_tensor_raw(&st.recurrent_state)?;
            let qkv_lane = qkv_ptr + (lane * conv_dim * 2) as u64;
            let mixed_lane = mixed_base + (lane * conv_dim * 2) as u64;
            let z_lane = z_ptr + (lane * value_dim * 2) as u64;
            let a_lane = ab_ptr + ((lane * ab_stride + a_off) * 2) as u64;
            let b_lane = ab_ptr + ((lane * ab_stride + b_off) * 2) as u64;
            let gated_lane = gated_base + (lane * value_dim * 2) as u64;
            let rc = unsafe {
                nv_kernels::cuda::gdn_conv_decode_silu_bf16(
                    lane_cu as *mut std::ffi::c_void,
                    qkv_lane as *const u16,
                    conv_state_ptr as *mut u16,
                    w_ptr as *const u16,
                    mixed_lane as *mut u16,
                    conv_dim as i32,
                    kernel as i32,
                )
            };
            anyhow::ensure!(rc == 0, "batch-lane {lane} gdn_conv_decode_silu_bf16 rc={rc}");
            let mut split_step_taken = false;
            if split_env {
                let rc = unsafe {
                    nv_kernels::cuda::gdn_decode_step_split_bf16(
                        lane_cu as *mut std::ffi::c_void,
                        mixed_lane as *const u16,
                        z_lane as *const u16,
                        a_lane as *const u16,
                        b_lane as *const u16,
                        a_log_ptr as *const u16,
                        dt_ptr as *const u16,
                        nw_ptr as *const u16,
                        rec_state_ptr as *mut f32,
                        gated_lane as *mut u16,
                        qn_ptr as *mut f32,
                        kn_ptr as *mut f32,
                        ge_ptr as *mut f32,
                        be_ptr as *mut f32,
                        core_ptr as *mut u16,
                        n_k as i32,
                        n_v as i32,
                        d_k as i32,
                        d_v as i32,
                        self.cfg.rms_eps as f32,
                    )
                };
                if rc == 0 {
                    split_step_taken = true;
                } else {
                    anyhow::ensure!(rc == -1, "batch-lane {lane} gdn_decode_step_split_bf16 rc={rc}");
                }
            }
            if !split_step_taken {
                let rc = unsafe {
                    nv_kernels::cuda::gdn_decode_step_bf16(
                        lane_cu as *mut std::ffi::c_void,
                        mixed_lane as *const u16,
                        z_lane as *const u16,
                        a_lane as *const u16,
                        b_lane as *const u16,
                        a_log_ptr as *const u16,
                        dt_ptr as *const u16,
                        nw_ptr as *const u16,
                        rec_state_ptr as *mut f32,
                        gated_lane as *mut u16,
                        n_k as i32,
                        n_v as i32,
                        d_k as i32,
                        d_v as i32,
                        self.cfg.rms_eps as f32,
                    )
                };
                anyhow::ensure!(rc == 0, "batch-lane {lane} gdn_decode_step_bf16 rc={rc}");
            }
        }
        if let Some(ls) = conc_steps {
            for s_i in ls[1..m].iter() {
                let ev = s_i
                    .record_event(None)
                    .map_err(|e| anyhow::anyhow!("gdn lane join event: {e:?}"))?;
                stream
                    .wait(&ev)
                    .map_err(|e| anyhow::anyhow!("gdn lane join wait: {e:?}"))?;
            }
        }
        drop(_gm);
        drop(_gg);
        drop(_gq);
        drop(_gk);
        drop(_gge);
        drop(_gbe);
        drop(_gco);
        if let Some(p) = prof.as_deref_mut() {
            p.lap("b.gdn.lanes");
        }
        let flat = {
            let storage = candle_core::CudaStorage::wrap_cuda_slice(gated, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, m, value_dim),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        let out = self.out_proj.forward(&flat)?;
        if let Some(p) = prof.as_deref_mut() {
            p.lap("b.gdn.o_proj");
        }
        Ok(Some(out))
    }

    #[cfg(feature = "cuda")]
    pub fn forward_decode_fused_prenorm_folded_reading_raw_x_because_the_layer_rmsnorm_kernel_is_gone(
        &self,
        x_raw: &Tensor,
        pre_norm_weight_bf16: &Tensor,
        rstd_pack_f32_first_elem_is_rstd: &Tensor,
        state: &LinAttnState,
    ) -> Result<Option<Tensor>> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;

        if !gdn_prenorm_fold_env_read_per_call_so_the_kill_switch_works_mid_process() {
            return Ok(None);
        }
        if !state.fused || !self.fused_decode_supported() {
            return Ok(None);
        }
        let hidden = self.cfg.hidden_size;
        if x_raw.dims() != [1, 1, hidden] || x_raw.dtype() != DType::BF16 {
            return Ok(None);
        }
        if pre_norm_weight_bf16.dtype() != DType::BF16
            || pre_norm_weight_bf16.elem_count() != hidden
            || rstd_pack_f32_first_elem_is_rstd.dtype() != DType::F32
        {
            return Ok(None);
        }
        let dev = match x_raw.device() {
            Device::Cuda(d) => d.clone(),
            _ => return Ok(None),
        };
        let Some((qkv_wq, qkv_scales)) = self
            .in_proj_qkv
            .fp8_e4m3_row_weight_and_scales_so_gdn_prenorm_folds_into_gemv_e4m3_mk_h()
        else {
            return Ok(None);
        };
        let Some((z_wq, z_scales)) = self
            .in_proj_z
            .fp8_e4m3_row_weight_and_scales_so_gdn_prenorm_folds_into_gemv_e4m3_mk_h()
        else {
            return Ok(None);
        };
        let Some(ab) = &self
            .ab_concat_one_gemv_when_both_bf16_because_two_48_row_launches_paid_double_latency
        else {
            return Ok(None);
        };
        let Some(ab_w) = ab.weight() else {
            return Ok(None);
        };
        if ab_w.dtype() != DType::BF16 {
            return Ok(None);
        }

        let stream = crate::cuda_stream::current_stream(&dev);
        let conv_dim = self.cfg.conv_dim();
        let value_dim = self.cfg.value_dim();
        let n_v = self.cfg.linear_num_value_heads;
        let mut prof = gdn_step_prof::SyncLaps::begin_if_armed(&dev);

        let x_c = x_raw.contiguous()?;
        let (_dx, x_ptr, _lx) = cuda_tensor_raw(&x_c)?;
        let nw_c = pre_norm_weight_bf16.contiguous()?;
        let (_dn, nw_ptr, _ln) = cuda_tensor_raw(&nw_c)?;
        let (_dr, rstd_ptr, _lr) = cuda_tensor_raw(rstd_pack_f32_first_elem_is_rstd)?;
        let ab_w_c = ab_w.contiguous()?;
        let (_dw, ab_w_ptr, _lw) = cuda_tensor_raw(&ab_w_c)?;

        let mut qkv_buf: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(conv_dim)
                .map_err(|e| anyhow::anyhow!("alloc folded qkv: {e:?}"))?
        };
        let mut z_buf: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(value_dim)
                .map_err(|e| anyhow::anyhow!("alloc folded z: {e:?}"))?
        };
        let mut ab_buf: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(2 * n_v)
                .map_err(|e| anyhow::anyhow!("alloc folded ab: {e:?}"))?
        };

        {
            let (wp, _gw) = qkv_wq.device_ptr(&stream);
            let (sp, _gs) = qkv_scales.device_ptr(&stream);
            let (yp, _gy) = qkv_buf.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::gemv_e4m3_mk_h(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    wp as *const u8,
                    sp as *const f32,
                    x_ptr as *const u16,
                    nw_ptr as *const u16,
                    rstd_ptr as *const f32,
                    yp as *mut u16,
                    conv_dim as i32,
                    hidden as i32,
                    1,
                )
            };
            if rc == -1 {
                return Ok(None);
            }
            anyhow::ensure!(rc == 0, "folded qkv gemv_e4m3_mk_h rc={rc}");
        }
        if let Some(p) = prof.as_mut() {
            p.lap("in_qkv");
        }
        {
            let (wp, _gw) = z_wq.device_ptr(&stream);
            let (sp, _gs) = z_scales.device_ptr(&stream);
            let (yp, _gy) = z_buf.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::gemv_e4m3_mk_h(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    wp as *const u8,
                    sp as *const f32,
                    x_ptr as *const u16,
                    nw_ptr as *const u16,
                    rstd_ptr as *const f32,
                    yp as *mut u16,
                    value_dim as i32,
                    hidden as i32,
                    1,
                )
            };
            if rc == -1 {
                return Ok(None);
            }
            anyhow::ensure!(rc == 0, "folded z gemv_e4m3_mk_h rc={rc}");
        }
        if let Some(p) = prof.as_mut() {
            p.lap("in_z");
        }
        {
            let (yp, _gy) = ab_buf.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::gemv_bf16_normed(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    ab_w_ptr as *const u16,
                    x_ptr as *const u16,
                    nw_ptr as *const u16,
                    rstd_ptr as *const f32,
                    yp as *mut u16,
                    (2 * n_v) as i32,
                    hidden as i32,
                )
            };
            if rc == -1 {
                return Ok(None);
            }
            anyhow::ensure!(rc == 0, "folded ab gemv_bf16_normed rc={rc}");
        }
        if let Some(p) = prof.as_mut() {
            p.lap("in_ab");
        }
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "[linear_attn] NV_Q38_GDN_PRENORM_FOLD decode arm active \
                     (qkv/z via gemv_e4m3_mk_h, ab via gemv_bf16_normed, hidden={hidden})"
                );
            });
        }

        let wrap = |slice: cudarc::driver::CudaSlice<bf16>, cols: usize| -> Tensor {
            let storage = candle_core::CudaStorage::wrap_cuda_slice(slice, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, 1usize, cols),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        let qkv = wrap(qkv_buf, conv_dim);
        let z = wrap(z_buf, value_dim);
        let ab_t = wrap(ab_buf, 2 * n_v);
        let a = ab_t.narrow(D::Minus1, 0, n_v)?;
        let b_proj = ab_t.narrow(D::Minus1, n_v, n_v)?;
        self.fused_decode_conv_step_out_shared_by_plain_and_prenorm_folded_arms(
            state, &dev, &qkv, &z, &a, &b_proj, &mut prof,
        )
        .map(Some)
    }

    #[cfg(feature = "cuda")]
    fn forward_decode_fused_qkvz_concat(
        &self,
        x: &Tensor,
        state: &LinAttnState,
        concat: &QkvzConcatFp8DecodeArm,
        dev: &candle_core::CudaDevice,
    ) -> Result<Option<Tensor>> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};

        let hidden = self.cfg.hidden_size;
        if hidden % 16 != 0 {
            return Ok(None);
        }
        let Some(a_w_t) = self.in_proj_a.weight() else {
            return Ok(None);
        };
        let Some(b_w_t) = self.in_proj_b.weight() else {
            return Ok(None);
        };
        if a_w_t.dtype() != DType::BF16 || b_w_t.dtype() != DType::BF16 {
            return Ok(None);
        }
        let stream = crate::cuda_stream::current_stream(dev);
        let conv_dim = self.cfg.conv_dim();
        let value_dim = self.cfg.value_dim();
        let n_k = self.cfg.linear_num_key_heads;
        let n_v = self.cfg.linear_num_value_heads;
        let d_k = self.cfg.linear_key_head_dim;
        let d_v = self.cfg.linear_value_head_dim;
        let kernel = self.cfg.linear_conv_kernel_dim;

        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "[linear_attn] NV_Q38_GDN_FUSE qkvz+conv gemv and ab+step fused decode arm \
                     active (3 kernels per gdn layer)"
                );
            });
        }

        let x_c = x.contiguous()?;
        let (_dx, x_ptr, _lx) = cuda_tensor_raw(&x_c)?;
        let a_w_c = a_w_t.contiguous()?;
        let b_w_c = b_w_t.contiguous()?;
        let (_da, a_w_ptr, _la) = cuda_tensor_raw(&a_w_c)?;
        let (_db, b_w_ptr, _lb) = cuda_tensor_raw(&b_w_c)?;
        let w_c = self.conv1d_weight.contiguous()?;
        let (_dw, conv_w_ptr, _lw) = cuda_tensor_raw(&w_c)?;
        let (_dc, conv_state_ptr, _lc) = cuda_tensor_raw(&state.conv_state)?;
        let (_dr, rec_state_ptr, _lr) = cuda_tensor_raw(&state.recurrent_state)?;
        let a_log_c = self.a_log.contiguous()?;
        let dt_c = self.dt_bias.contiguous()?;
        let nw_c = self.norm_weight.contiguous()?;
        let (_d8, a_log_ptr, _l8) = cuda_tensor_raw(&a_log_c)?;
        let (_d9, dt_ptr, _l9) = cuda_tensor_raw(&dt_c)?;
        let (_d10, nw_ptr, _l10) = cuda_tensor_raw(&nw_c)?;

        let mut mixed: cudarc::driver::CudaSlice<half::bf16> = unsafe {
            stream
                .alloc::<half::bf16>(conv_dim)
                .map_err(|e| anyhow::anyhow!("alloc mixed: {e:?}"))?
        };
        let mut zbuf: cudarc::driver::CudaSlice<half::bf16> = unsafe {
            stream
                .alloc::<half::bf16>(value_dim)
                .map_err(|e| anyhow::anyhow!("alloc z: {e:?}"))?
        };
        let mut gated: cudarc::driver::CudaSlice<half::bf16> = unsafe {
            stream
                .alloc::<half::bf16>(value_dim)
                .map_err(|e| anyhow::anyhow!("alloc gated: {e:?}"))?
        };

        {
            let (wp, _gw) = concat.weight_u8.device_ptr(&stream);
            let (sp, _gs) = concat.row_scales_dev.device_ptr(&stream);
            let (mp, _gm) = mixed.device_ptr_mut(&stream);
            let (zp, _gz) = zbuf.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::gemv_e4m3_qkvz_conv_m1(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    wp as *const u8,
                    sp as *const f32,
                    x_ptr as *const u16,
                    conv_w_ptr as *const u16,
                    conv_state_ptr as *mut u16,
                    mp as *mut u16,
                    zp as *mut u16,
                    (conv_dim + value_dim) as i32,
                    hidden as i32,
                    conv_dim as i32,
                    kernel as i32,
                )
            };
            if rc == -1 {
                return Ok(None);
            }
            anyhow::ensure!(rc == 0, "gemv_e4m3_qkvz_conv_m1 rc={rc}");
        }
        {
            let (mp, _gm) = mixed.device_ptr(&stream);
            let (zp, _gz) = zbuf.device_ptr(&stream);
            let (gp, _gg) = gated.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::gdn_decode_step_ab_fused_bf16(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    x_ptr as *const u16,
                    a_w_ptr as *const u16,
                    b_w_ptr as *const u16,
                    mp as *const u16,
                    zp as *const u16,
                    a_log_ptr as *const u16,
                    dt_ptr as *const u16,
                    nw_ptr as *const u16,
                    rec_state_ptr as *mut f32,
                    gp as *mut u16,
                    hidden as i32,
                    n_k as i32,
                    n_v as i32,
                    d_k as i32,
                    d_v as i32,
                    self.cfg.rms_eps as f32,
                )
            };
            anyhow::ensure!(rc == 0, "gdn_decode_step_ab_fused_bf16 rc={rc}");
        }

        let flat = {
            let storage = candle_core::CudaStorage::wrap_cuda_slice(gated, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, 1usize, value_dim),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        let out = self.out_proj.forward(&flat)?;
        Ok(Some(out))
    }

    #[cfg(feature = "cuda")]
    pub fn forward_verify_mrow_projections_once_because_per_row_fused_steps_reread_every_gdn_weight(
        &self,
        x: &Tensor,
        state: &LinAttnState,
        ckpts: &[LinAttnState],
    ) -> Result<Option<Tensor>> {
        use cudarc::driver::DevicePtrMut;

        if !state.fused || !self.fused_decode_supported() {
            return Ok(None);
        }
        let dims = x.dims();
        if dims.len() != 3 || dims[0] != 1 || x.dtype() != DType::BF16 {
            return Ok(None);
        }
        let m = dims[1];
        if m < 2
            || m > MROW_VERIFY_ROWS_MAX_16_THE_GDN_CHUNK_KERNEL_T_CAP
            || dims[2] != self.cfg.hidden_size
        {
            return Ok(None);
        }
        if ckpts.len() < m || ckpts.iter().take(m).any(|c| !c.fused) {
            return Ok(None);
        }
        let kernel = self.cfg.linear_conv_kernel_dim;
        if !(2..=MROW_VERIFY_CONV_KERNEL_MAX_8_THE_GDN_CONV_CHUNK_KERNEL_K_CAP).contains(&kernel) {
            return Ok(None);
        }
        let dev = match x.device() {
            Device::Cuda(d) => d.clone(),
            _ => return Ok(None),
        };
        let stream = crate::cuda_stream::current_stream(&dev);

        let conv_dim = self.cfg.conv_dim();
        let value_dim = self.cfg.value_dim();
        let n_k = self.cfg.linear_num_key_heads;
        let n_v = self.cfg.linear_num_value_heads;
        let d_k = self.cfg.linear_key_head_dim;
        let d_v = self.cfg.linear_value_head_dim;

        let mut prof = gdn_step_prof::SyncLaps::begin_if_armed(&dev);
        let x_c = x.contiguous()?;
        let qkv = self.in_proj_qkv.forward(&x_c)?.contiguous()?;
        if let Some(p) = prof.as_mut() {
            p.lap("mrow_in_qkv");
        }
        let z = self.in_proj_z.forward(&x_c)?.contiguous()?;
        let a = self.in_proj_a.forward(&x_c)?.contiguous()?;
        let b_proj = self.in_proj_b.forward(&x_c)?.contiguous()?;
        if let Some(p) = prof.as_mut() {
            p.lap("mrow_in_z_a_b");
        }
        if qkv.dtype() != DType::BF16
            || z.dtype() != DType::BF16
            || a.dtype() != DType::BF16
            || b_proj.dtype() != DType::BF16
        {
            return Ok(None);
        }

        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "[linear_attn] m-row verify path active: one qkv/z/a/b GEMM over m rows \
                     then chunked fused state updates (m={m} n_k={n_k} n_v={n_v} d_k={d_k} \
                     d_v={d_v} k={kernel})"
                );
            });
        }
        if verify_gdn_chunk_env_opt_in_nv_q38_verify_gdn_chunk_1_writes_ckpts_in_place_off_pooled_scratch()
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "[linear_attn] verify gdn-chunk in-place ckpt arm requested \
                     (NV_Q38_VERIFY_GDN_CHUNK=1): chunk kernels target the ckpt slab and the \
                     per-row dtod fan-out is gone when the slab rows are stride-contiguous"
                );
            });
        }

        let (_d1, qkv_ptr, _l1) = cuda_tensor_raw(&qkv)?;
        let (_d2, z_ptr, _l2) = cuda_tensor_raw(&z)?;
        let (_d3, a_ptr, _l3) = cuda_tensor_raw(&a)?;
        let (_d4, b_ptr, _l4) = cuda_tensor_raw(&b_proj)?;
        let w_c = self.conv1d_weight.contiguous()?;
        let (_d5, w_ptr, _l5) = cuda_tensor_raw(&w_c)?;
        let a_log_c = self.a_log.contiguous()?;
        let dt_c = self.dt_bias.contiguous()?;
        let nw_c = self.norm_weight.contiguous()?;
        let (_d6, a_log_ptr, _l6) = cuda_tensor_raw(&a_log_c)?;
        let (_d7, dt_ptr, _l7) = cuda_tensor_raw(&dt_c)?;
        let (_d8, nw_ptr, _l8) = cuda_tensor_raw(&nw_c)?;

        let conv_row_elems = conv_dim * (kernel - 1);
        let rec_row_elems = n_v * d_k * d_v;
        anyhow::ensure!(
            state.conv_state.elem_count() == conv_row_elems
                && state.recurrent_state.elem_count() == rec_row_elems,
            "m-row verify live state geometry mismatch: conv {} want {conv_row_elems}, rec {} \
             want {rec_row_elems}",
            state.conv_state.elem_count(),
            state.recurrent_state.elem_count()
        );
        let (_d9, conv_state_ptr, _l9) = cuda_tensor_raw(&state.conv_state)?;
        let (_d10, rec_state_ptr, _l10) = cuda_tensor_raw(&state.recurrent_state)?;

        let key_dim = n_k * d_k;
        let ckpt_in_place_bases = if verify_gdn_chunk_env_opt_in_nv_q38_verify_gdn_chunk_1_writes_ckpts_in_place_off_pooled_scratch()
        {
            ckpt_rows_contiguous_base(&ckpts[..m], conv_row_elems, rec_row_elems)?
        } else {
            None
        };
        let mut scratch_lease = match ckpt_in_place_bases {
            Some(_) => Some(mrow_verify_scratch_take_or_build(
                &stream, m, conv_dim, key_dim, value_dim, n_v,
            )?),
            None => None,
        };
        let mut pooled_ptrs: Option<(u64, u64, u64, u64, u64, u64)> = None;
        let _pooled_guards = match scratch_lease.as_mut() {
            Some(lease) => {
                let s: &mut MrowVerifyScratch = lease;
                let (mp, g0) = s.mixed_seq.device_ptr_mut(&stream);
                let (qp, g1) = s.qn.device_ptr_mut(&stream);
                let (kp, g2) = s.kn.device_ptr_mut(&stream);
                let (gep, g3) = s.g_exp.device_ptr_mut(&stream);
                let (bep, g4) = s.beta.device_ptr_mut(&stream);
                let (cp, g5) = s.core.device_ptr_mut(&stream);
                pooled_ptrs = Some((mp, qp, kp, gep, bep, cp));
                Some((g0, g1, g2, g3, g4, g5))
            }
            None => None,
        };
        let mut owned_mixed: Option<cudarc::driver::CudaSlice<half::bf16>> =
            match pooled_ptrs {
                Some(_) => None,
                None => Some(unsafe {
                    stream
                        .alloc::<half::bf16>(m * conv_dim)
                        .map_err(|e| anyhow::anyhow!("alloc mrow mixed: {e:?}"))?
                }),
            };
        let mut owned_ckpt_conv: Option<cudarc::driver::CudaSlice<half::bf16>> =
            match ckpt_in_place_bases {
                Some(_) => None,
                None => Some(unsafe {
                    stream
                        .alloc::<half::bf16>(m * conv_row_elems)
                        .map_err(|e| anyhow::anyhow!("alloc mrow conv ckpts: {e:?}"))?
                }),
            };
        let mut owned_ckpt_rec: Option<cudarc::driver::CudaSlice<f32>> = match ckpt_in_place_bases {
            Some(_) => None,
            None => Some(unsafe {
                stream
                    .alloc::<f32>(m * rec_row_elems)
                    .map_err(|e| anyhow::anyhow!("alloc mrow rec ckpts: {e:?}"))?
            }),
        };
        let mut gated: cudarc::driver::CudaSlice<half::bf16> = unsafe {
            stream
                .alloc::<half::bf16>(m * value_dim)
                .map_err(|e| anyhow::anyhow!("alloc mrow gated: {e:?}"))?
        };
        let (mixed_ptr, _g_owned_mixed) = match pooled_ptrs {
            Some((mp, ..)) => (mp, None),
            None => {
                let (p, g) = owned_mixed.as_mut().unwrap().device_ptr_mut(&stream);
                (p, Some(g))
            }
        };
        let (ckpt_conv_base, _g_owned_ckpt_conv) = match ckpt_in_place_bases {
            Some((cb, _)) => (cb, None),
            None => {
                let (p, g) = owned_ckpt_conv.as_mut().unwrap().device_ptr_mut(&stream);
                (p, Some(g))
            }
        };
        let (ckpt_rec_base, _g_owned_ckpt_rec) = match ckpt_in_place_bases {
            Some((_, rb)) => (rb, None),
            None => {
                let (p, g) = owned_ckpt_rec.as_mut().unwrap().device_ptr_mut(&stream);
                (p, Some(g))
            }
        };

        {
            let mp = mixed_ptr;
            let cp = ckpt_conv_base;
            let rc = unsafe {
                nv_kernels::cuda::gdn_conv_decode_chunk_silu_bf16(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    qkv_ptr as *const u16,
                    conv_state_ptr as *const u16,
                    w_ptr as *const u16,
                    mp as *mut u16,
                    cp as *mut u16,
                    conv_dim as i32,
                    kernel as i32,
                    m as i32,
                )
            };
            anyhow::ensure!(rc == 0, "gdn_conv_decode_chunk_silu_bf16 rc={rc}");
        }
        if let Some(p) = prof.as_mut() {
            p.lap("mrow_conv_chunk");
        }
        let mut chunk_split_wrote_live_state = false;
        if gdn_chunk_split_default_on_inside_mrow_kill_switch_nv_q38_gdn_chunk_split_0() {
            let mut owned_split: Option<(
                cudarc::driver::CudaSlice<f32>,
                cudarc::driver::CudaSlice<f32>,
                cudarc::driver::CudaSlice<f32>,
                cudarc::driver::CudaSlice<f32>,
                cudarc::driver::CudaSlice<half::bf16>,
            )> = match pooled_ptrs {
                Some(_) => None,
                None => Some(unsafe {
                    (
                        stream
                            .alloc::<f32>(m * key_dim)
                            .map_err(|e| anyhow::anyhow!("alloc chunk-split qn: {e:?}"))?,
                        stream
                            .alloc::<f32>(m * key_dim)
                            .map_err(|e| anyhow::anyhow!("alloc chunk-split kn: {e:?}"))?,
                        stream
                            .alloc::<f32>(m * n_v)
                            .map_err(|e| anyhow::anyhow!("alloc chunk-split g_exp: {e:?}"))?,
                        stream
                            .alloc::<f32>(m * n_v)
                            .map_err(|e| anyhow::anyhow!("alloc chunk-split beta: {e:?}"))?,
                        stream
                            .alloc::<half::bf16>(m * value_dim)
                            .map_err(|e| anyhow::anyhow!("alloc chunk-split core: {e:?}"))?,
                    )
                }),
            };
            let (qp, kp, gep, bep, cp, _g_owned_split) = match pooled_ptrs {
                Some((_, qp, kp, gep, bep, cp)) => (qp, kp, gep, bep, cp, None),
                None => {
                    let s = owned_split.as_mut().unwrap();
                    let (qp, g0) = s.0.device_ptr_mut(&stream);
                    let (kp, g1) = s.1.device_ptr_mut(&stream);
                    let (gep, g2) = s.2.device_ptr_mut(&stream);
                    let (bep, g3) = s.3.device_ptr_mut(&stream);
                    let (cp, g4) = s.4.device_ptr_mut(&stream);
                    (qp, kp, gep, bep, cp, Some((g0, g1, g2, g3, g4)))
                }
            };
            let mp = mixed_ptr;
            let rp = ckpt_rec_base;
            let (gp, _gg) = gated.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::gdn_decode_chunk_split_bf16(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    mp as *const u16,
                    z_ptr as *const u16,
                    a_ptr as *const u16,
                    b_ptr as *const u16,
                    a_log_ptr as *const u16,
                    dt_ptr as *const u16,
                    nw_ptr as *const u16,
                    rec_state_ptr as *const f32,
                    rp as *mut f32,
                    rec_state_ptr as *mut f32,
                    gp as *mut u16,
                    qp as *mut f32,
                    kp as *mut f32,
                    gep as *mut f32,
                    bep as *mut f32,
                    cp as *mut u16,
                    n_k as i32,
                    n_v as i32,
                    d_k as i32,
                    d_v as i32,
                    self.cfg.rms_eps as f32,
                    m as i32,
                )
            };
            if rc == 0 {
                chunk_split_wrote_live_state = true;
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    eprintln!(
                        "[linear_attn] mrow chunk-split state path active \
                         (prep+regstate-vsplit+gate over t rows, n_v={n_v} d_k={d_k} d_v={d_v})"
                    );
                });
            } else {
                anyhow::ensure!(rc == -1, "gdn_decode_chunk_split_bf16 rc={rc}");
            }
        }
        if !chunk_split_wrote_live_state {
            let mp = mixed_ptr;
            let rp = ckpt_rec_base;
            let (gp, _gg) = gated.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::gdn_decode_chunk_bf16(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    mp as *const u16,
                    z_ptr as *const u16,
                    a_ptr as *const u16,
                    b_ptr as *const u16,
                    a_log_ptr as *const u16,
                    dt_ptr as *const u16,
                    nw_ptr as *const u16,
                    rec_state_ptr as *const f32,
                    rp as *mut f32,
                    gp as *mut u16,
                    n_k as i32,
                    n_v as i32,
                    d_k as i32,
                    d_v as i32,
                    self.cfg.rms_eps as f32,
                    m as i32,
                )
            };
            anyhow::ensure!(rc == 0, "gdn_decode_chunk_bf16 rc={rc}");
        }
        if let Some(p) = prof.as_mut() {
            p.lap("mrow_state_chunk");
        }

        let conv_row_bytes = conv_row_elems * 2;
        let rec_row_bytes = rec_row_elems * 4;
        if ckpt_in_place_bases.is_none() {
            for (j, ck) in ckpts.iter().take(m).enumerate() {
                anyhow::ensure!(
                    ck.conv_state.elem_count() == conv_row_elems
                        && ck.recurrent_state.elem_count() == rec_row_elems,
                    "m-row verify ckpt {j} geometry mismatch"
                );
                let (_dc, ck_conv_ptr, _lc) = cuda_tensor_raw(&ck.conv_state)?;
                let (_dr, ck_rec_ptr, _lr) = cuda_tensor_raw(&ck.recurrent_state)?;
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        ck_conv_ptr,
                        ckpt_conv_base + (j * conv_row_bytes) as u64,
                        conv_row_bytes,
                        stream.cu_stream(),
                    )
                    .map_err(|e| anyhow::anyhow!("mrow ckpt conv dtod row {j}: {e:?}"))?;
                    cudarc::driver::result::memcpy_dtod_async(
                        ck_rec_ptr,
                        ckpt_rec_base + (j * rec_row_bytes) as u64,
                        rec_row_bytes,
                        stream.cu_stream(),
                    )
                    .map_err(|e| anyhow::anyhow!("mrow ckpt rec dtod row {j}: {e:?}"))?;
                }
            }
        }
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                conv_state_ptr,
                ckpt_conv_base + ((m - 1) * conv_row_bytes) as u64,
                conv_row_bytes,
                stream.cu_stream(),
            )
            .map_err(|e| anyhow::anyhow!("mrow live conv advance dtod: {e:?}"))?;
            if !chunk_split_wrote_live_state {
                cudarc::driver::result::memcpy_dtod_async(
                    rec_state_ptr,
                    ckpt_rec_base + ((m - 1) * rec_row_bytes) as u64,
                    rec_row_bytes,
                    stream.cu_stream(),
                )
                .map_err(|e| anyhow::anyhow!("mrow live rec advance dtod: {e:?}"))?;
            }
        }

        if let Some(p) = prof.as_mut() {
            p.lap("mrow_ckpt_dtod");
        }
        let flat = {
            let storage = candle_core::CudaStorage::wrap_cuda_slice(gated, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, m, value_dim),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        let out = self.out_proj.forward(&flat)?;
        if let Some(p) = prof.as_mut() {
            p.lap("mrow_out_proj");
        }
        Ok(Some(out))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input_dtype = x.dtype();
        let dims = x.dims().to_vec();
        if dims.len() != 3 {
            anyhow::bail!(
                "LinearAttention::forward: expected [B, T, H], got {:?}",
                dims
            );
        }
        let (b, t, h) = (dims[0], dims[1], dims[2]);
        if h != self.cfg.hidden_size {
            anyhow::bail!(
                "LinearAttention: hidden {} != cfg {}",
                h,
                self.cfg.hidden_size
            );
        }
        let n_k = self.cfg.linear_num_key_heads;
        let n_v = self.cfg.linear_num_value_heads;
        let d_k = self.cfg.linear_key_head_dim;
        let d_v = self.cfg.linear_value_head_dim;
        let key_dim = self.cfg.key_dim();
        let value_dim = self.cfg.value_dim();
        let conv_dim = self.cfg.conv_dim();
        let kernel = self.cfg.linear_conv_kernel_dim;
        let acc_dtype = self.cfg.mamba_ssm_dtype;

        let prof_enabled = std::env::var("NV_PROF_LINATTN").is_ok();
        #[cfg(feature = "cuda")]
        let prof_dev: Option<candle_core::CudaDevice> = if prof_enabled {
            match x.device() {
                Device::Cuda(d) => Some(d.clone()),
                _ => None,
            }
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let _prof_dev: Option<()> = None;
        let mut prof: std::collections::BTreeMap<&'static str, f64> =
            std::collections::BTreeMap::new();
        let prof_sync = || {
            #[cfg(feature = "cuda")]
            if let Some(dev) = &prof_dev {
                let _ = dev.cuda_stream().synchronize();
            }
        };
        macro_rules! tic {
            () => {{
                prof_sync();
                std::time::Instant::now()
            }};
        }
        macro_rules! toc {
            ($label:expr, $t0:expr) => {
                if prof_enabled {
                    prof_sync();
                    *prof.entry($label).or_default() += $t0.elapsed().as_secs_f64() * 1000.0;
                }
            };
        }

        let t0 = tic!();
        let qkv = self.in_proj_qkv.forward(x)?;
        let z = self.in_proj_z.forward(x)?;
        let a = self.in_proj_a.forward(x)?;
        let b_proj = self.in_proj_b.forward(x)?;
        toc!("in_proj (qkv+z+a+b)", t0);

        let t0 = tic!();
        let qkv_bt = qkv.reshape((b, t, conv_dim))?;
        let qkv_btc = qkv_bt.transpose(1, 2)?.contiguous()?;
        let pad_left = kernel - 1;

        let use_fused_conv = {
            #[cfg(feature = "cuda")]
            {
                matches!(qkv_btc.device(), Device::Cuda(_))
                    && qkv_btc.dtype() == DType::BF16
                    && self.conv1d_weight.dtype() == DType::BF16
                    && pad_left == kernel - 1
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        };

        let mixed = if use_fused_conv {
            #[cfg(feature = "cuda")]
            {
                use cudarc::driver::{DevicePtr, DevicePtrMut};
                let dev = match qkv_btc.device() {
                    Device::Cuda(d) => d.clone(),
                    _ => unreachable!(),
                };
                let stream = crate::cuda_stream::current_stream(&dev);
                let mut y_dev: cudarc::driver::CudaSlice<half::bf16> =
                    unsafe { stream.alloc::<half::bf16>(b * conv_dim * t)? };
                let qkv_btc_c = qkv_btc.contiguous()?;
                let w_c = self.conv1d_weight.contiguous()?;
                {
                    let (xs, _xl) = qkv_btc_c.storage_and_layout();
                    let (ws, _wl) = w_c.storage_and_layout();
                    let x_cuda = match &*xs {
                        candle_core::Storage::Cuda(s) => s,
                        _ => anyhow::bail!("expected cuda storage"),
                    };
                    let w_cuda = match &*ws {
                        candle_core::Storage::Cuda(s) => s,
                        _ => anyhow::bail!("expected cuda storage"),
                    };
                    let x_slice = x_cuda.as_cuda_slice::<half::bf16>()?;
                    let w_slice = w_cuda.as_cuda_slice::<half::bf16>()?;
                    let (xp, _g1) = x_slice.device_ptr(&stream);
                    let (wp, _g2) = w_slice.device_ptr(&stream);
                    let (yp, _g3) = y_dev.device_ptr_mut(&stream);
                    let rc = unsafe {
                        nv_kernels::cuda::depthwise_conv1d_silu_bf16(
                            stream.cu_stream() as *mut std::ffi::c_void,
                            xp as *const u16,
                            wp as *const u16,
                            yp as *mut u16,
                            b as i32,
                            conv_dim as i32,
                            t as i32,
                            kernel as i32,
                        )
                    };
                    anyhow::ensure!(rc == 0, "depthwise_conv1d_silu_bf16 rc={rc}");
                }

                let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
                let y_bct = Tensor::from_storage(
                    candle_core::Storage::Cuda(storage),
                    (b, conv_dim, t),
                    candle_core::op::BackpropOp::none(),
                    false,
                );
                y_bct.transpose(1, 2)?.contiguous()?
            }
            #[cfg(not(feature = "cuda"))]
            {
                unreachable!()
            }
        } else {
            let conv_in = if pad_left > 0 {
                let pad =
                    Tensor::zeros((b, conv_dim, pad_left), qkv_btc.dtype(), qkv_btc.device())?;
                Tensor::cat(&[&pad, &qkv_btc], 2)?.contiguous()?
            } else {
                qkv_btc
            };
            let conv_out = conv_in.conv1d(&self.conv1d_weight, 0, 1, 1, conv_dim)?;
            let conv_out = candle_nn::ops::silu(&conv_out)?;
            conv_out.transpose(1, 2)?.contiguous()?
        };
        toc!("conv1d + silu", t0);

        let t0 = tic!();
        let q_flat = mixed.narrow(2, 0, key_dim)?;
        let k_flat = mixed.narrow(2, key_dim, key_dim)?;
        let v_flat = mixed.narrow(2, 2 * key_dim, value_dim)?;

        let q = q_flat.reshape((b, t, n_k, d_k))?;
        let k = k_flat.reshape((b, t, n_k, d_k))?;
        let v = v_flat.reshape((b, t, n_v, d_v))?;

        let v_per_k = self.cfg.v_per_k();
        let q_grp = if v_per_k == 1 {
            q
        } else {
            q.unsqueeze(3)?
                .expand((b, t, n_k, v_per_k, d_k))?
                .contiguous()?
                .reshape((b, t, n_v, d_k))?
        };
        let k_grp = if v_per_k == 1 {
            k
        } else {
            k.unsqueeze(3)?
                .expand((b, t, n_k, v_per_k, d_k))?
                .contiguous()?
                .reshape((b, t, n_v, d_k))?
        };

        let q_grp = l2_normalize_last(&q_grp.to_dtype(acc_dtype)?)?;
        let k_grp = l2_normalize_last(&k_grp.to_dtype(acc_dtype)?)?;

        let scale = 1.0 / (d_k as f64).sqrt();
        let q_scaled = q_grp.affine(scale, 0.0)?;
        let k_f = k_grp;
        let v_f = v.to_dtype(acc_dtype)?;
        toc!("split+reshape+norm+grouping", t0);

        let t0 = tic!();
        let (g_exp, beta) = compute_gdn_gating(&a, &b_proj, &self.a_log, &self.dt_bias, acc_dtype)?;
        toc!("gdn_gating", t0);

        let t0 = tic!();
        let core = run_gdn_scan(
            &q_scaled, &k_f, &v_f, &g_exp, &beta, b, t, n_v, d_k, d_v, acc_dtype,
        )?;
        toc!("gdn_scan", t0);

        let t0 = tic!();
        let core_bf = core.to_dtype(input_dtype)?;

        let z_re = z.reshape((b, t, n_v, d_v))?;

        let normed = rmsnorm_last(&core_bf, &self.norm_weight, self.cfg.rms_eps)?;
        let gate = candle_nn::ops::silu(&z_re.to_dtype(DType::F32)?)?.to_dtype(input_dtype)?;
        let gated = normed.mul(&gate)?;

        let flat = gated.reshape((b, t, value_dim))?;
        toc!("post (cast+rmsnorm+silu+mul+reshape)", t0);

        let t0 = tic!();
        let out = self.out_proj.forward(&flat)?;
        toc!("out_proj", t0);

        if prof_enabled {
            let total: f64 = prof.values().sum();
            eprintln!(
                "[la_prof] linear-attn one-call breakdown (total {:.2} ms):",
                total
            );
            let mut entries: Vec<(&&'static str, &f64)> = prof.iter().collect();
            entries.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (lbl, ms) in entries {
                eprintln!(
                    "[la_prof]   {:>40}  {:8.3} ms  ({:5.1}%)",
                    lbl,
                    ms,
                    100.0 * ms / total.max(1e-12)
                );
            }
        }
        Ok(out)
    }
}

#[allow(clippy::too_many_arguments)]
fn run_gdn_scan(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g_exp: &Tensor,
    beta: &Tensor,
    b: usize,
    t: usize,
    n_v: usize,
    d_k: usize,
    d_v: usize,
    acc_dtype: DType,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(_) = q.device() {
            if d_k == 128 && d_v == 128 && acc_dtype == DType::F32 {
                return run_gdn_scan_cuda_f32(q, k, v, g_exp, beta, b, t, n_v, d_k, d_v);
            }
        }
    }
    run_gdn_scan_candle(q, k, v, g_exp, beta, b, t, n_v, d_k, d_v, acc_dtype)
}

#[allow(clippy::too_many_arguments)]
fn run_gdn_scan_candle(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g_exp: &Tensor,
    beta: &Tensor,
    b: usize,
    t: usize,
    n_v: usize,
    d_k: usize,
    d_v: usize,
    acc_dtype: DType,
) -> Result<Tensor> {
    let (out, _final) = run_gdn_scan_candle_stateful(
        q, k, v, g_exp, beta, b, t, n_v, d_k, d_v, acc_dtype, None, None,
    )?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn run_gdn_scan_candle_stateful(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g_exp: &Tensor,
    beta: &Tensor,
    b: usize,
    t: usize,
    n_v: usize,
    d_k: usize,
    d_v: usize,
    acc_dtype: DType,
    init_state: Option<&Tensor>,
    mut capture: Option<&mut Vec<Tensor>>,
) -> Result<(Tensor, Tensor)> {
    let device = q.device();
    let mut state = match init_state {
        Some(s) => s.to_dtype(acc_dtype)?,
        None => Tensor::zeros((b, n_v, d_k, d_v), acc_dtype, device)?,
    };
    let mut outs: Vec<Tensor> = Vec::with_capacity(t);
    for ti in 0..t {
        let q_t = q.narrow(1, ti, 1)?.squeeze(1)?;
        let k_t = k.narrow(1, ti, 1)?.squeeze(1)?;
        let v_t = v.narrow(1, ti, 1)?.squeeze(1)?;
        let g_t = g_exp.narrow(1, ti, 1)?.squeeze(1)?;
        let beta_t = beta.narrow(1, ti, 1)?.squeeze(1)?;

        let decay = g_t.unsqueeze(2)?.unsqueeze(3)?;
        state = state.broadcast_mul(&decay)?;
        let k_col = k_t.unsqueeze(3)?;
        let kv_mem = state.broadcast_mul(&k_col)?.sum(2)?;
        let beta_col = beta_t.unsqueeze(2)?;
        let delta = v_t.sub(&kv_mem)?.broadcast_mul(&beta_col)?;
        let k_outer = k_t.unsqueeze(3)?;
        let delta_outer = delta.unsqueeze(2)?;
        let update = k_outer.broadcast_mul(&delta_outer)?;
        state = state.add(&update)?;
        let q_col = q_t.unsqueeze(3)?;
        let out_t = state.broadcast_mul(&q_col)?.sum(2)?;
        outs.push(out_t.unsqueeze(1)?);
        if let Some(buf) = capture.as_mut() {
            buf.push(state.clone());
        }
    }
    let out = Tensor::cat(&outs, 1)?;
    Ok((out, state))
}

#[cfg(feature = "cuda")]
fn run_gdn_scan_cuda_f32(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g_exp: &Tensor,
    beta: &Tensor,
    b: usize,
    t: usize,
    n_v: usize,
    d_k: usize,
    d_v: usize,
) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use nv_kernels::cuda as nvk;

    let q_c = q.to_dtype(DType::F32)?.contiguous()?;
    let k_c = k.to_dtype(DType::F32)?.contiguous()?;
    let v_c = v.to_dtype(DType::F32)?.contiguous()?;
    let g_c = g_exp.to_dtype(DType::F32)?.contiguous()?;
    let beta_c = beta.to_dtype(DType::F32)?.contiguous()?;

    let dev = match q_c.device() {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = crate::cuda_stream::current_stream(&dev);

    let mut out_dev: cudarc::driver::CudaSlice<f32> = stream
        .alloc_zeros::<f32>(b * t * n_v * d_v)
        .map_err(|e| anyhow::anyhow!(e))?;

    let rc = {
        let (qs, _ql) = q_c.storage_and_layout();
        let (ks, _kl) = k_c.storage_and_layout();
        let (vs, _vl) = v_c.storage_and_layout();
        let (gs, _gl) = g_c.storage_and_layout();
        let (bes, _bl) = beta_c.storage_and_layout();
        let q_cuda = match &*qs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("expected cuda storage"),
        };
        let k_cuda = match &*ks {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("expected cuda storage"),
        };
        let v_cuda = match &*vs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("expected cuda storage"),
        };
        let g_cuda = match &*gs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("expected cuda storage"),
        };
        let be_cuda = match &*bes {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("expected cuda storage"),
        };
        let q_slice = q_cuda.as_cuda_slice::<f32>()?;
        let k_slice = k_cuda.as_cuda_slice::<f32>()?;
        let v_slice = v_cuda.as_cuda_slice::<f32>()?;
        let g_slice = g_cuda.as_cuda_slice::<f32>()?;
        let be_slice = be_cuda.as_cuda_slice::<f32>()?;

        let (pq, _gq) = q_slice.device_ptr(&stream);
        let (pk, _gk) = k_slice.device_ptr(&stream);
        let (pv, _gv) = v_slice.device_ptr(&stream);
        let (pg, _gg) = g_slice.device_ptr(&stream);
        let (pbe, _gbe) = be_slice.device_ptr(&stream);
        let (pout, _gout) = out_dev.device_ptr_mut(&stream);

        unsafe {
            nvk::gdn_recurrent_f32(
                stream.cu_stream() as *mut _,
                pq as *const f32,
                pk as *const f32,
                pv as *const f32,
                pg as *const f32,
                pbe as *const f32,
                pout as *mut f32,
                b as i32,
                t as i32,
                n_v as i32,
                d_k as i32,
                d_v as i32,
            )
        }
    };
    if rc != 0 {
        anyhow::bail!("gdn_recurrent_f32 kernel returned {rc}");
    }

    let storage = candle_core::CudaStorage::wrap_cuda_slice(out_dev, dev);
    Ok(candle_core::Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        candle_core::Shape::from((b, t, n_v, d_v)),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

fn compute_gdn_gating(
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    acc_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(_) = a.device() {
            if a.dtype() == DType::BF16 && b.dtype() == DType::BF16 {
                return compute_gdn_gating_cuda_bf16(a, b, a_log, dt_bias, acc_dtype);
            }
        }
    }
    compute_gdn_gating_candle(a, b, a_log, dt_bias, acc_dtype)
}

fn compute_gdn_gating_candle(
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    acc_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let b_f = b.to_dtype(acc_dtype)?;
    let a_f = a.to_dtype(acc_dtype)?;
    let dt_bias_f = dt_bias.to_dtype(acc_dtype)?;
    let a_log_f = a_log.to_dtype(acc_dtype)?;

    let beta = candle_nn::ops::sigmoid(&b_f)?;
    let a_plus_bias = a_f.broadcast_add(&dt_bias_f)?;
    let sp = softplus(&a_plus_bias)?;
    let neg_exp_alog = a_log_f.exp()?.affine(-1.0, 0.0)?;
    let g = sp.broadcast_mul(&neg_exp_alog)?;
    let g_exp = g.exp()?;
    Ok((g_exp, beta))
}

#[cfg(feature = "cuda")]
fn compute_gdn_gating_cuda_bf16(
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    acc_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    use nv_kernels::cuda as nvk;

    let a_c = a.contiguous()?;
    let b_c = b.contiguous()?;
    let a_log_c = a_log.to_dtype(DType::BF16)?.contiguous()?;
    let dt_bias_c = dt_bias.to_dtype(DType::BF16)?.contiguous()?;

    let dims = a_c.dims().to_vec();
    if dims.is_empty() {
        anyhow::bail!("gdn_gating: a is scalar");
    }
    let num_heads = *dims.last().unwrap();
    let tokens: usize = dims[..dims.len() - 1].iter().product();
    let total = tokens * num_heads;
    if total == 0 {
        let g = Tensor::zeros(dims.clone(), DType::F32, a_c.device())?.to_dtype(acc_dtype)?;
        let beta = Tensor::zeros(dims, DType::BF16, a_c.device())?.to_dtype(acc_dtype)?;
        return Ok((g.exp()?, beta));
    }

    let dev = match a_c.device() {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = crate::cuda_stream::current_stream(&dev);

    let mut g_dev: cudarc::driver::CudaSlice<f32> = stream
        .alloc_zeros::<f32>(total)
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut beta_dev: cudarc::driver::CudaSlice<bf16> = stream
        .alloc_zeros::<bf16>(total)
        .map_err(|e| anyhow::anyhow!(e))?;

    let rc = {
        let (a_s, _al) = a_c.storage_and_layout();
        let (b_s, _bl) = b_c.storage_and_layout();
        let (al_s, _all) = a_log_c.storage_and_layout();
        let (dt_s, _dtl) = dt_bias_c.storage_and_layout();
        let a_cuda = match &*a_s {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("expected cuda storage"),
        };
        let b_cuda = match &*b_s {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("expected cuda storage"),
        };
        let al_cuda = match &*al_s {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("expected cuda storage"),
        };
        let dt_cuda = match &*dt_s {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("expected cuda storage"),
        };
        let a_sl = a_cuda.as_cuda_slice::<bf16>()?;
        let b_sl = b_cuda.as_cuda_slice::<bf16>()?;
        let al_sl = al_cuda.as_cuda_slice::<bf16>()?;
        let dt_sl = dt_cuda.as_cuda_slice::<bf16>()?;
        let (pa, _ga) = a_sl.device_ptr(&stream);
        let (pb, _gb) = b_sl.device_ptr(&stream);
        let (pal, _gal) = al_sl.device_ptr(&stream);
        let (pdt, _gdt) = dt_sl.device_ptr(&stream);
        let (pg, _gg) = g_dev.device_ptr_mut(&stream);
        let (pbeta, _gbeta) = beta_dev.device_ptr_mut(&stream);
        unsafe {
            nvk::gdn_gating_bf16(
                stream.cu_stream() as *mut _,
                pa as *const u16,
                pb as *const u16,
                pal as *const u16,
                pdt as *const u16,
                pg as *mut f32,
                pbeta as *mut u16,
                tokens,
                num_heads,
            )
        }
    };
    if rc != 0 {
        anyhow::bail!("gdn_gating_bf16 kernel returned {rc}");
    }

    let g_storage = candle_core::CudaStorage::wrap_cuda_slice(g_dev, dev.clone());
    let g_storage = candle_core::Storage::Cuda(g_storage);
    let shape: candle_core::Shape = dims.clone().into();
    let g = candle_core::Tensor::from_storage(
        g_storage,
        shape.clone(),
        candle_core::op::BackpropOp::none(),
        false,
    );
    let beta_storage = candle_core::CudaStorage::wrap_cuda_slice(beta_dev, dev);
    let beta_storage = candle_core::Storage::Cuda(beta_storage);
    let beta = candle_core::Tensor::from_storage(
        beta_storage,
        shape,
        candle_core::op::BackpropOp::none(),
        false,
    );

    let g_acc = g.to_dtype(acc_dtype)?;
    let g_exp = g_acc.exp()?;
    let beta_acc = beta.to_dtype(acc_dtype)?;
    Ok((g_exp, beta_acc))
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    let zero = Tensor::zeros(x.shape(), x.dtype(), x.device())?;
    let max_part = x.maximum(&zero)?;
    let neg_abs = x.abs()?.affine(-1.0, 0.0)?;
    let log_part = neg_abs.exp()?.affine(1.0, 1.0)?.log()?;
    Ok(max_part.add(&log_part)?)
}

fn l2_normalize_last(x: &Tensor) -> Result<Tensor> {
    let in_dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let sum_sq = xf.sqr()?.sum_keepdim(D::Minus1)?;
    let eps_t = Tensor::new(1e-6f32, x.device())?;
    let denom = sum_sq.broadcast_add(&eps_t)?.sqrt()?;
    let normed = xf.broadcast_div(&denom)?;
    Ok(normed.to_dtype(in_dt)?)
}

fn rmsnorm_last(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let in_dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let eps_t = Tensor::new(eps as f32, x.device())?;
    let denom = var.broadcast_add(&eps_t)?.sqrt()?;
    let normed = xf.broadcast_div(&denom)?;
    let w = weight.to_dtype(DType::F32)?;
    let scaled = normed.broadcast_mul(&w)?;
    Ok(scaled.to_dtype(in_dt)?)
}

fn load_linear_else_fp8_rowscale_dequant_because_q38_mixed_checkpoints_store_gdn_qkv_z_out_as_f8e4m3(
    weights: &WeightLoader,
    module: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<Linear> {
    let weight_name = format!("{module}.weight");
    let resolved = resolve_name(weights, &weight_name)?;
    let resolved_module = resolved
        .strip_suffix(".weight")
        .unwrap_or(&resolved)
        .to_string();
    if crate::linear::checkpoint_module_is_fp8_e4m3_weight_with_scale(weights, &resolved_module) {
        return crate::linear::fp8_e4m3_rowscale_checkpoint_dequant_linear(
            weights,
            &resolved_module,
            out_features,
            in_features,
            dtype,
        );
    }
    load_linear(weights, &weight_name, out_features, in_features, dtype)
}

#[cfg(feature = "cuda")]
fn qkvz_concat_raw_fp8_if_both_modules_are_fp8(
    weights: &WeightLoader,
    prefix: &str,
    conv_dim: usize,
    value_dim: usize,
    hidden: usize,
) -> Result<Option<(Vec<u8>, Vec<f32>)>> {
    let mut bytes: Vec<u8> = Vec::with_capacity((conv_dim + value_dim) * hidden);
    let mut scales: Vec<f32> = Vec::with_capacity(conv_dim + value_dim);
    for (module, rows) in [
        (format!("{prefix}.in_proj_qkv"), conv_dim),
        (format!("{prefix}.in_proj_z"), value_dim),
    ] {
        let resolved_w = resolve_name(weights, &format!("{module}.weight"))?;
        let resolved_module = resolved_w
            .strip_suffix(".weight")
            .unwrap_or(&resolved_w)
            .to_string();
        if !crate::linear::checkpoint_module_is_fp8_e4m3_weight_with_scale(
            weights,
            &resolved_module,
        ) {
            return Ok(None);
        }
        let raw = weights
            .raw_bytes(&resolved_w)
            .map_err(|e| anyhow::anyhow!("read {resolved_w}: {e}"))?;
        anyhow::ensure!(
            raw.len() == rows * hidden,
            "{resolved_w}: {} bytes != rows {rows} x hidden {hidden}",
            raw.len()
        );
        bytes.extend_from_slice(raw);
        let scale_name = format!("{resolved_module}.weight_scale");
        let scale_t = weights
            .get(&scale_name, DType::F32)
            .map_err(|e| anyhow::anyhow!("load {scale_name}: {e}"))?;
        let scale_dims = scale_t.dims().to_vec();
        let scale_vals: Vec<f32> = scale_t.flatten_all()?.to_vec1()?;
        let row_scales = nv_weights::fp8_row_scales_from(&scale_dims, &scale_vals, rows)
            .map_err(|e| anyhow::anyhow!("{scale_name} shape {scale_dims:?}: {e}"))?;
        scales.extend_from_slice(&row_scales);
    }
    Ok(Some((bytes, scales)))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn load_linear_else_fp8_rowscale_resident_because_gdn_decode_bandwidth_wants_1_byte_per_param(
    weights: &WeightLoader,
    module: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
    device: &Device,
    fp8_runner: &std::sync::Arc<std::sync::Mutex<nv_quant::fp8::Fp8GemmRunner>>,
) -> Result<Linear> {
    let weight_name = format!("{module}.weight");
    let resolved = resolve_name(weights, &weight_name)?;
    let resolved_module = resolved
        .strip_suffix(".weight")
        .unwrap_or(&resolved)
        .to_string();
    if crate::linear::checkpoint_module_is_fp8_e4m3_weight_with_scale(weights, &resolved_module) {
        return crate::linear::fp8_e4m3_rowscale_checkpoint_resident_linear(
            weights,
            &resolved_module,
            out_features,
            in_features,
            device,
            fp8_runner.clone(),
        );
    }
    load_linear(weights, &weight_name, out_features, in_features, dtype)
}

#[cfg(feature = "cuda")]
fn load_linear_fp8_resident_quantizing_bf16_checkpoint_modules_because_qwen36_ships_gdn_qkv_z_out_as_bf16(
    weights: &WeightLoader,
    module: &str,
    out_features: usize,
    in_features: usize,
    device: &Device,
    fp8_runner: &std::sync::Arc<std::sync::Mutex<nv_quant::fp8::Fp8GemmRunner>>,
) -> Result<Linear> {
    let weight_name = format!("{module}.weight");
    let resolved = resolve_name(weights, &weight_name)?;
    let resolved_module = resolved
        .strip_suffix(".weight")
        .unwrap_or(&resolved)
        .to_string();
    if crate::linear::checkpoint_module_is_fp8_e4m3_weight_with_scale(weights, &resolved_module) {
        return crate::linear::fp8_e4m3_rowscale_checkpoint_resident_linear(
            weights,
            &resolved_module,
            out_features,
            in_features,
            device,
            fp8_runner.clone(),
        );
    }
    let w = weights
        .get(&resolved, DType::BF16)
        .with_context(|| format!("load {resolved}"))?;
    let d = w.dims();
    anyhow::ensure!(
        d.len() == 2 && d[0] == out_features && d[1] == in_features,
        "linear {resolved}: expected [{out_features}, {in_features}], got {d:?}"
    );
    let weight_host: Vec<half::bf16> = w.flatten_all()?.to_vec1()?;
    let (weight_bytes, row_scales) = crate::linear::fp8_weight_payload(
        &weight_host,
        out_features,
        in_features,
        None,
        nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
    )?;
    let dev = match device {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("fp8 resident linear requires a CUDA device"),
    };
    let stream = crate::cuda_stream::current_stream(&dev);
    #[allow(deprecated)]
    let weight_u8 = stream
        .clone_htod(&weight_bytes)
        .map_err(|e| anyhow::anyhow!(e))?;
    Linear::new_fp8_e4m3_row_scales_without_the_cublaslt_probe(
        weight_u8,
        row_scales,
        in_features,
        out_features,
        None,
        device,
        fp8_runner.clone(),
        nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
    )
}

fn load_linear(
    weights: &WeightLoader,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<Linear> {
    let resolved = resolve_name(weights, name)?;
    let w = weights
        .get(&resolved, dtype)
        .with_context(|| format!("load {resolved}"))?;
    let d = w.dims();
    if d.len() != 2 || d[0] != out_features || d[1] != in_features {
        anyhow::bail!(
            "linear {resolved}: expected [{}, {}], got {:?}",
            out_features,
            in_features,
            d
        );
    }
    Linear::new(w, None)
}

fn load_tensor(
    weights: &WeightLoader,
    name: &str,
    expected: &[usize],
    dtype: DType,
) -> Result<Tensor> {
    let resolved = resolve_name(weights, name)?;
    let t = weights
        .get(&resolved, dtype)
        .with_context(|| format!("load {resolved}"))?;
    if t.dims() != expected {
        anyhow::bail!(
            "tensor {resolved}: expected {:?}, got {:?}",
            expected,
            t.dims()
        );
    }
    Ok(t)
}

fn resolve_name(weights: &WeightLoader, name: &str) -> Result<String> {
    if weights.has(name) {
        return Ok(name.to_string());
    }
    if let Some(stripped) = name.strip_prefix("model.") {
        if weights.has(stripped) {
            return Ok(stripped.to_string());
        }
    }
    anyhow::bail!("tensor not found (tried {name})")
}

#[cfg(all(test, feature = "cuda"))]
mod gdn_decode_scratch_lifetime_tests {
    use super::*;
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    const DIMS_A_CONV64_VAL32_NKDK4096_NV4_QN_IS_16KIB_SO_POOL_REUSE_IS_BLOCK_GRANULAR:
        (usize, usize, usize, usize) = (64, 32, 4096, 4);
    const DIMS_B_CONV128_VAL64_NKDK8192_NV8: (usize, usize, usize, usize) = (128, 64, 8192, 8);
    const PATTERN_COPIED_THROUGH_THE_BAKED_QN_POINTER: f32 = 7.5;
    const CANARY_FILL_THAT_MUST_SURVIVE_THE_REPLAY: f32 = -3.25;
    const CANARY_BARRAGE_4_QN_SIZED_ALLOCS_SWEEP_THE_POOL_FREE_LIST_SO_A_FREED_QN_BLOCK_GETS_REUSED:
        usize = 4;
    const SHAPE_TOKEN_FOR_THE_DIMS_A_GRAPH: u64 = 0xA11CE;

    fn silu_host(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    #[test]
    fn graph_replay_through_dims_a_qn_pointer_survives_a_dims_b_take_on_the_same_thread() {
        let dev = match candle_core::Device::new_cuda(0) {
            Ok(candle_core::Device::Cuda(d)) => d,
            _ => panic!(
                "no CUDA device 0: this is the cross-engine scratch UAF gate and must not \
                 report success having executed nothing"
            ),
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        let (ca, va, ka, na) = DIMS_A_CONV64_VAL32_NKDK4096_NV4_QN_IS_16KIB_SO_POOL_REUSE_IS_BLOCK_GRANULAR;
        let mut a = gdn_decode_scratch_take_or_build(&dev, ca, va, ka, na).unwrap();
        let a_qn_ptr_at_capture = {
            let (p, _g) = a.qn.device_ptr_mut(&stream);
            p
        };
        let pattern_host = vec![PATTERN_COPIED_THROUGH_THE_BAKED_QN_POINTER; ka];
        let mut src = stream.alloc_zeros::<f32>(ka).unwrap();
        stream.memcpy_htod(&pattern_host, &mut src).unwrap();
        let mut out = stream.alloc_zeros::<f32>(ka).unwrap();
        let src_ptr = {
            let (p, _g) = src.device_ptr(&stream);
            p
        };
        let out_ptr = {
            let (p, _g) = out.device_ptr_mut(&stream);
            p
        };
        stream.synchronize().unwrap();

        let fork = stream.context().new_stream().unwrap();
        let mut runner = nv_kernels::graph::CudaGraphRunner::new(fork.clone());
        let expected = silu_host(silu_host(PATTERN_COPIED_THROUGH_THE_BAKED_QN_POINTER));
        runner
            .run(SHAPE_TOKEN_FOR_THE_DIMS_A_GRAPH, |s| {
                let rc = unsafe {
                    nv_kernels::cuda::silu_f32(
                        s.cu_stream() as *mut std::ffi::c_void,
                        src_ptr as *const f32,
                        a_qn_ptr_at_capture as *mut f32,
                        ka,
                    )
                };
                anyhow::ensure!(rc == 0, "silu into qn rc={rc}");
                let rc = unsafe {
                    nv_kernels::cuda::silu_f32(
                        s.cu_stream() as *mut std::ffi::c_void,
                        a_qn_ptr_at_capture as *const f32,
                        out_ptr as *mut f32,
                        ka,
                    )
                };
                anyhow::ensure!(rc == 0, "silu out of qn rc={rc}");
                Ok(())
            })
            .unwrap();
        fork.synchronize().unwrap();
        let first: Vec<f32> = fork.memcpy_dtov(&out).unwrap();
        assert!(
            first.iter().all(|v| (*v - expected).abs() < 1e-3),
            "first launch must route the pattern through the dims-A qn scratch: got {} expected \
             {expected}",
            first[0]
        );
        drop(a);

        let (cb, vb, kb, nb) = DIMS_B_CONV128_VAL64_NKDK8192_NV8;
        let b = gdn_decode_scratch_take_or_build(&dev, cb, vb, kb, nb).unwrap();
        let canary_host = vec![CANARY_FILL_THAT_MUST_SURVIVE_THE_REPLAY; ka];
        let canaries: Vec<cudarc::driver::CudaSlice<f32>> = (0
            ..CANARY_BARRAGE_4_QN_SIZED_ALLOCS_SWEEP_THE_POOL_FREE_LIST_SO_A_FREED_QN_BLOCK_GETS_REUSED)
            .map(|_| {
                let mut c = stream.alloc_zeros::<f32>(ka).unwrap();
                stream.memcpy_htod(&canary_host, &mut c).unwrap();
                c
            })
            .collect();
        stream.memset_zeros(&mut out).unwrap();
        stream.synchronize().unwrap();

        runner
            .run(SHAPE_TOKEN_FOR_THE_DIMS_A_GRAPH, |_s| {
                Err(anyhow::anyhow!(
                    "shape token must already be cached: this call is a replay, not a capture"
                ))
            })
            .unwrap();
        fork.synchronize().unwrap();
        let replay: Vec<f32> = fork.memcpy_dtov(&out).unwrap();
        assert!(
            replay.iter().all(|v| (*v - expected).abs() < 1e-3),
            "replaying the dims-A graph after a dims-B take on the same thread read freed or \
             recycled memory through its baked qn pointer: got {} expected {expected}",
            replay[0]
        );
        for (i, c) in canaries.iter().enumerate() {
            let canary_back: Vec<f32> = stream.memcpy_dtov(c).unwrap();
            assert!(
                canary_back
                    .iter()
                    .all(|v| *v == CANARY_FILL_THAT_MUST_SURVIVE_THE_REPLAY),
                "the replayed graph wrote through a freed scratch pointer into bystander \
                 canary {i}: {}",
                canary_back[0]
            );
        }
        drop(b);

        let a2 = gdn_decode_scratch_take_or_build(&dev, ca, va, ka, na).unwrap();
        let a2_qn_ptr = {
            let (p, _g) = a2.qn.device_ptr(&stream);
            p
        };
        assert_eq!(
            a2_qn_ptr, a_qn_ptr_at_capture,
            "a dims-A retake must hand back the allocation whose pointer the captured graph \
             baked; a fresh pointer means the old scratch was freed and every replay of that \
             graph is a use-after-free"
        );
        drop(a2);
    }
}
