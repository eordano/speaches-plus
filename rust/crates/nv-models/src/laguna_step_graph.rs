#![cfg(feature = "cuda")]

pub mod prefix;

use anyhow::{bail, ensure, Context, Result};
use candle_core::{CudaDevice, DType, Device, Storage, Tensor};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::{c_void, CString};
use std::sync::{Arc, Once, OnceLock};

use crate::gemma4::{Gemma4Cache, LayerType};
use crate::laguna::{Laguna, LagunaFfn, LagunaKvCache, LagunaProjBytes};
use crate::laguna_fa2::{varlen_fwd_bf16, VarlenArgs};
use crate::laguna_fp8::{LagunaFp8ScalePolicy, LagunaKvCacheFp8};
use crate::laguna_graph::moe_block_body;
use nv_kernels::graph::CudaGraphRunner;
use nv_layers::moe_grouped::{GroupedDecodeContext, MoeGroupedWeights};
use std::env;
use std::path::PathBuf;

pub(crate) fn verify_prof_enabled() -> bool {
    env::var("NV_LAGUNA_VERIFY_PROF")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("fine"))
        .unwrap_or(false)
}

pub(crate) fn fine_prof_enabled() -> bool {
    env::var("NV_LAGUNA_VERIFY_PROF")
        .map(|v| v.eq_ignore_ascii_case("fine"))
        .unwrap_or(false)
}

const M1_FUSED_TC_SPLITS_32_MUST_EQUAL_THE_VERIFY_GRAPHS_SO_ROW_J_REDUCES_LIKE_THE_M1_STEP: usize =
    32;

fn verify_dense_rows_run_as_t_single_row_forwards_so_each_row_is_the_m1_step_bitwise_opt_in_until_the_ledger_reads_bitwise(
) -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_LAGUNA_VERIFY_DENSE_ROWS_AS_M1")
}

fn m1_full_attention_through_the_fused_tc_verify_kernel_opt_in_until_petas_no_draft_bracket() -> bool
{
    nv_config::env_arm_opt_in_set_to_anything_but_0_reread_every_call("NV_LAGUNA_M1_FUSED_TC")
}

struct M1FlashBufs {
    scratch: CudaSlice<f32>,
    fan_in: CudaSlice<u32>,
}

pub(crate) fn w8_bytes_ptr(p: &crate::laguna::LagunaProjW8, stream: &Arc<CudaStream>) -> u64 {
    match p.bytes() {
        LagunaProjBytes::I8(b) => {
            let (q, _g) = b.device_ptr(stream);
            q
        }
        LagunaProjBytes::E4m3(b) => {
            let (q, _g) = b.device_ptr(stream);
            q
        }
    }
}

pub(crate) fn w8_scale_ptr(p: &crate::laguna::LagunaProjW8, stream: &Arc<CudaStream>) -> u64 {
    let (q, _g) = p.row_scale().device_ptr(stream);
    q
}

const FINE_PER_LAYER: usize = 16;

pub(crate) const FINE_STAGES: [&str; 21] = [
    "embed",
    "qkv_proj",
    "rope_norms",
    "kv_append",
    "flash_sliding",
    "flash_full",
    "attn_gate",
    "o_proj",
    "moe_norm_gate",
    "moe_route",
    "moe_gather_quant",
    "moe_gemm_gate_up",
    "moe_silu_quant",
    "moe_gemm_down",
    "moe_scatter",
    "moe_tail",
    "shared_expert",
    "dense_ffn",
    "lm_head",
    "logits_argmax",
    "total",
];

pub(crate) fn prof_timestamp_at(
    buf: &CudaSlice<u64>,
    slot: usize,
    s: &Arc<CudaStream>,
) -> Result<()> {
    let (p, _g) = buf.device_ptr(s);
    let rc = unsafe {
        nv_kernels::cuda::prof_timestamp(
            s.cu_stream() as *mut c_void,
            (p + (slot * 8) as u64) as *mut u64,
        )
    };
    ensure!(rc == 0, "prof timestamp rc={rc}");
    Ok(())
}

pub struct StepProfStamps {
    buf: CudaSlice<u64>,
    stream: Arc<CudaStream>,
    n_layers: usize,
    fine: bool,
    layers: Vec<(bool, bool)>,
    acc: std::cell::RefCell<(u64, [f64; 21])>,
}

impl StepProfStamps {
    fn new(stream: &Arc<CudaStream>, layers: Vec<(bool, bool)>) -> Result<Self> {
        let n_layers = layers.len();
        let fine = fine_prof_enabled();
        let slots = if fine {
            4 + FINE_PER_LAYER * n_layers
        } else {
            4 + 2 * n_layers
        };
        let buf = stream
            .alloc_zeros::<u64>(slots)
            .map_err(|e| anyhow::anyhow!("prof buf: {e:?}"))?;
        Ok(Self {
            buf,
            stream: stream.clone(),
            n_layers,
            fine,
            layers,
            acc: std::cell::RefCell::new((0, [0.0; 21])),
        })
    }

    pub(crate) fn is_fine(&self) -> bool {
        self.fine
    }

    fn slot(&self, which: ProfPoint) -> usize {
        if self.fine {
            let base = |li: usize| 2 + FINE_PER_LAYER * li;
            return match which {
                ProfPoint::Start => 0,
                ProfPoint::Embed => 1,
                ProfPoint::Qkv(li) => base(li),
                ProfPoint::Rope(li) => base(li) + 1,
                ProfPoint::Append(li) => base(li) + 2,
                ProfPoint::Flash(li) => base(li) + 3,
                ProfPoint::Gate(li) => base(li) + 4,
                ProfPoint::Attn(li) => base(li) + 5,
                ProfPoint::MoeNormGate(li) => base(li) + 6,
                ProfPoint::MoeRoute(li) => base(li) + 7,
                ProfPoint::MoeScatter(li) => base(li) + 12,
                ProfPoint::SharedStart(li) => base(li) + 13,
                ProfPoint::SharedEnd(li) => base(li) + 14,
                ProfPoint::Ffn(li) => base(li) + 15,
                ProfPoint::LmHead => 2 + FINE_PER_LAYER * self.n_layers,
                ProfPoint::End => 3 + FINE_PER_LAYER * self.n_layers,
            };
        }
        match which {
            ProfPoint::Start => 0,
            ProfPoint::Embed => 1,
            ProfPoint::Attn(li) => 2 + 2 * li,
            ProfPoint::Ffn(li) => 3 + 2 * li,
            ProfPoint::LmHead => 2 + 2 * self.n_layers,
            ProfPoint::End => 3 + 2 * self.n_layers,
            _ => usize::MAX,
        }
    }

    pub(crate) fn record(&self, which: ProfPoint, s: &Arc<CudaStream>) -> Result<()> {
        let slot = self.slot(which);
        if slot == usize::MAX {
            return Ok(());
        }
        prof_timestamp_at(&self.buf, slot, s)
    }

    pub(crate) fn moe_grouped_prof_base(&self, li: usize, s: &Arc<CudaStream>) -> Option<u64> {
        if !self.fine {
            return None;
        }
        let (p, _g) = self.buf.device_ptr(s);
        Some(p + (self.slot(ProfPoint::MoeRoute(li)) * 8) as u64)
    }

    fn report_fine(&self, tag: &str, ts: &[u64]) -> Result<()> {
        let el = |a: ProfPoint, b: ProfPoint| -> f64 {
            (ts[self.slot(b)].saturating_sub(ts[self.slot(a)])) as f64 / 1e6
        };
        use ProfPoint as P;
        let mut st = [0f64; 21];
        st[0] = el(P::Start, P::Embed);
        let mut prev = P::Embed;
        for (li, &(sliding, dense)) in self.layers.iter().enumerate() {
            st[1] += el(prev, P::Qkv(li));
            st[2] += el(P::Qkv(li), P::Rope(li));
            st[3] += el(P::Rope(li), P::Append(li));
            let fl = el(P::Append(li), P::Flash(li));
            if sliding {
                st[4] += fl;
            } else {
                st[5] += fl;
            }
            st[6] += el(P::Flash(li), P::Gate(li));
            st[7] += el(P::Gate(li), P::Attn(li));
            if dense {
                st[17] += el(P::Attn(li), P::Ffn(li));
            } else {
                st[8] += el(P::Attn(li), P::MoeNormGate(li));
                let gb = self.slot(P::MoeRoute(li));
                let gel = |a: usize, b: usize| (ts[b].saturating_sub(ts[a])) as f64 / 1e6;
                st[9] += gel(self.slot(P::MoeNormGate(li)), gb);
                for i in 0..5 {
                    st[10 + i] += gel(gb + i, gb + i + 1);
                }
                st[15] += el(P::MoeScatter(li), P::Ffn(li));
                st[16] += el(P::SharedStart(li), P::SharedEnd(li));
            }
            prev = P::Ffn(li);
        }
        st[18] = el(prev, P::LmHead);
        st[19] = el(P::LmHead, P::End);
        st[20] = el(P::Start, P::End);
        let serial_sum: f64 = st[..16].iter().sum::<f64>() + st[17] + st[18] + st[19];
        {
            let mut acc = self.acc.borrow_mut();
            acc.0 += 1;
            if acc.0 > 1 {
                for i in 0..21 {
                    acc.1[i] += st[i];
                }
            }
            let n_acc = acc.0.saturating_sub(1).max(1) as f64;
            let mut line = format!("[laguna_fine_prof] {tag} n={}", acc.0);
            for (i, name) in FINE_STAGES.iter().enumerate() {
                line.push_str(&format!(" {name}={:.4}", st[i]));
            }
            line.push_str(&format!(" serial_sum={serial_sum:.4}"));
            eprintln!("{line}");
            if acc.0 > 1 {
                let mut mline = format!("[laguna_fine_prof_mean] {tag} n={}", acc.0 - 1);
                let mut msum = 0f64;
                for (i, name) in FINE_STAGES.iter().enumerate() {
                    let m = acc.1[i] / n_acc;
                    if i < 16 || i == 17 || i == 18 || i == 19 {
                        msum += m;
                    }
                    mline.push_str(&format!(" {name}={m:.4}"));
                }
                mline.push_str(&format!(" serial_sum={msum:.4}"));
                eprintln!("{mline}");
            }
        }
        Ok(())
    }

    fn report(&self, tag: &str) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("prof sync: {e:?}"))?;
        let ts: Vec<u64> = self
            .stream
            .clone_dtoh(&self.buf)
            .map_err(|e| anyhow::anyhow!("prof dtoh: {e:?}"))?;
        if self.fine {
            return self.report_fine(tag, &ts);
        }
        let el = |a: usize, b: usize| -> f64 { (ts[b].saturating_sub(ts[a])) as f64 / 1e6 };
        let n = self.n_layers;
        let embed = el(self.slot(ProfPoint::Start), self.slot(ProfPoint::Embed));
        let mut attn_ms = vec![0f64; n];
        let mut ffn_ms = vec![0f64; n];
        let mut prev = self.slot(ProfPoint::Embed);
        for li in 0..n {
            attn_ms[li] = el(prev, self.slot(ProfPoint::Attn(li)));
            ffn_ms[li] = el(
                self.slot(ProfPoint::Attn(li)),
                self.slot(ProfPoint::Ffn(li)),
            );
            prev = self.slot(ProfPoint::Ffn(li));
        }
        let mut line = format!("[laguna_verify_prof] {tag} embed={embed:.3}");
        let mut attn_total = 0f64;
        let mut ffn_total = 0f64;
        let mut g0 = 0usize;
        while g0 < n {
            let g1 = (g0 + 10).min(n);
            let a: f64 = attn_ms[g0..g1].iter().sum();
            let f: f64 = ffn_ms[g0..g1].iter().sum();
            attn_total += a;
            ffn_total += f;
            line.push_str(&format!(
                " attn{g0:02}_{:02}={a:.3} ffn{g0:02}_{:02}={f:.3}",
                g1 - 1,
                g1 - 1
            ));
            g0 = g1;
        }
        let lm = el(prev, self.slot(ProfPoint::LmHead));
        let logits = el(self.slot(ProfPoint::LmHead), self.slot(ProfPoint::End));
        let total = el(self.slot(ProfPoint::Start), self.slot(ProfPoint::End));
        let sum = embed + attn_total + ffn_total + lm + logits;
        line.push_str(&format!(
            " attn_total={attn_total:.3} ffn_total={ffn_total:.3} lm_head={lm:.3} logits={logits:.3} sum={sum:.3} total={total:.3}"
        ));
        eprintln!("{line}");
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ProfPoint {
    Start,
    Embed,
    Qkv(usize),
    Rope(usize),
    Append(usize),
    Flash(usize),
    Gate(usize),
    Attn(usize),
    MoeNormGate(usize),
    MoeRoute(usize),
    MoeScatter(usize),
    SharedStart(usize),
    SharedEnd(usize),
    Ffn(usize),
    LmHead,
    End,
}

pub(crate) fn bf16_ptr(t: &Tensor, stream: &Arc<CudaStream>, len: usize) -> Result<u64> {
    let (st, l) = t.storage_and_layout();
    ensure!(l.is_contiguous(), "step graph: tensor not contiguous");
    let cuda = match &*st {
        Storage::Cuda(s) => s,
        _ => bail!("step graph: tensor not CUDA"),
    };
    let slice = cuda.as_cuda_slice::<bf16>()?;
    let view = slice.slice(l.start_offset()..l.start_offset() + len);
    let (p, _g) = view.device_ptr(stream);
    Ok(p)
}

pub(crate) fn wrap_bf16(
    slice: CudaSlice<bf16>,
    dims: (usize, usize, usize),
    dev: &CudaDevice,
) -> Tensor {
    let storage = candle_core::CudaStorage::wrap_cuda_slice(slice, dev.clone());
    Tensor::from_storage(
        Storage::Cuda(storage),
        dims,
        candle_core::op::BackpropOp::none(),
        false,
    )
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct NvLagunaXqaInfo {
    pub abi_version: u32,
    pub query_rows: u32,
    pub capacity: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub page_size: u32,
    pub splits: u32,
    pub dynamic_shared_bytes: u32,
    pub scratch_alignment: u32,
    pub scratch_bytes: u64,
    pub metadata_bytes: u64,
    pub status_device: u64,
}

unsafe extern "C" {
    fn dlopen(filename: *const std::os::raw::c_char, flag: std::os::raw::c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const std::os::raw::c_char) -> *mut c_void;
}
const RTLD_NOW_GLIBC: std::os::raw::c_int = 2;
const RTLD_GLOBAL_GLIBC: std::os::raw::c_int = 0x100;

type XqaCreateFn = unsafe extern "C" fn(
    *const std::os::raw::c_char,
    u32,
    u32,
    *mut *mut c_void,
    *mut NvLagunaXqaInfo,
) -> i32;
type XqaLaunchFn =
    unsafe extern "C" fn(*mut c_void, u64, u64, u64, u64, u64, f32, *mut c_void) -> i32;
type XqaDestroyFn = unsafe extern "C" fn(*mut c_void) -> i32;

struct XqaAdapter {
    create: XqaCreateFn,
    launch: XqaLaunchFn,
    destroy: XqaDestroyFn,
    dir: PathBuf,
}

unsafe impl Send for XqaAdapter {}
unsafe impl Sync for XqaAdapter {}

const XQA_ADAPTER_IS_ROOTS_PINNED_FLASHINFER_0_6_4_XQA_KERNEL_BEHIND_A_C_ABI_LOADED_AT_RUNTIME_ONLY_WHEN_NV_LAGUNA_XQA_IS_1: &str =
    "NV_LAGUNA_XQA=1 needs NV_LAGUNA_XQA_DIR pointing at liblaguna_xqa_adapter.so and laguna-xqa-bf16-m{1,2,3,5}.cubin (temp/q38-coord/xqa-native-adapter-candidate/build.sh output)";

fn xqa_adapter() -> Result<Option<&'static XqaAdapter>> {
    static ADAPTER: OnceLock<Option<XqaAdapter>> = OnceLock::new();
    if !nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_LAGUNA_XQA") {
        return Ok(None);
    }
    let loaded = ADAPTER.get_or_init(|| {
        let dir = PathBuf::from(env::var("NV_LAGUNA_XQA_DIR").unwrap_or_default());
        let lib = CString::new(
            dir.join("liblaguna_xqa_adapter.so")
                .to_string_lossy()
                .as_bytes(),
        )
        .ok()?;
        let handle = unsafe { dlopen(lib.as_ptr(), RTLD_NOW_GLIBC | RTLD_GLOBAL_GLIBC) };
        if handle.is_null() {
            eprintln!("[laguna_xqa] dlopen failed for {}", lib.to_string_lossy());
            return None;
        }
        let sym = |name: &str| -> *mut c_void {
            let c = CString::new(name).unwrap();
            unsafe { dlsym(handle, c.as_ptr()) }
        };
        let (c, l, d) = (
            sym("nv_laguna_xqa_create"),
            sym("nv_laguna_xqa_launch"),
            sym("nv_laguna_xqa_destroy"),
        );
        if c.is_null() || l.is_null() || d.is_null() {
            eprintln!(
                "[laguna_xqa] adapter symbols missing in {}",
                lib.to_string_lossy()
            );
            return None;
        }
        Some(XqaAdapter {
            create: unsafe { std::mem::transmute::<*mut c_void, XqaCreateFn>(c) },
            launch: unsafe { std::mem::transmute::<*mut c_void, XqaLaunchFn>(l) },
            destroy: unsafe { std::mem::transmute::<*mut c_void, XqaDestroyFn>(d) },
            dir,
        })
    });
    match loaded {
        Some(a) => Ok(Some(a)),
        None => bail!("{XQA_ADAPTER_IS_ROOTS_PINNED_FLASHINFER_0_6_4_XQA_KERNEL_BEHIND_A_C_ABI_LOADED_AT_RUNTIME_ONLY_WHEN_NV_LAGUNA_XQA_IS_1}"),
    }
}

pub struct XqaPlan {
    adapter: &'static XqaAdapter,
    plan: *mut c_void,
    info: NvLagunaXqaInfo,
    rows: usize,
}

unsafe impl Send for XqaPlan {}

impl XqaPlan {
    fn new_for(
        rows: usize,
        capacity: usize,
        kv_heads: usize,
        q_heads: usize,
        hd: usize,
        label: &str,
    ) -> Result<Option<Self>> {
        let Some(adapter) = xqa_adapter()? else {
            return Ok(None);
        };
        ensure!(matches!(rows, 1 | 2 | 3 | 5) && kv_heads == 8 && q_heads == 48 && hd == 128 && capacity.is_multiple_of(128) && (128..=262144).contains(&capacity),
            "NV_LAGUNA_XQA=1 supports rows 1, 2, 3 or 5 (m{rows}.cubin), 48/8 heads, hd 128, capacity a multiple of 128 up to 262144 (got rows={rows} capacity={capacity} q={q_heads} kv={kv_heads} hd={hd})");
        let cubin = CString::new(
            adapter
                .dir
                .join(format!("laguna-xqa-bf16-m{rows}.cubin"))
                .to_string_lossy()
                .as_bytes(),
        )?;
        let mut plan: *mut c_void = std::ptr::null_mut();
        let mut info = NvLagunaXqaInfo::default();
        let rc = unsafe {
            (adapter.create)(
                cubin.as_ptr(),
                rows as u32,
                capacity as u32,
                &mut plan,
                &mut info,
            )
        };
        ensure!(
            rc == 0 && !plan.is_null(),
            "nv_laguna_xqa_create rc={rc} rows={rows} capacity={capacity}"
        );
        eprintln!("[laguna_xqa] route=xqa-bf16 graph={label} rows={rows} capacity={capacity} splits={} dynamic_shared_bytes={} scratch_bytes={} status_device={:#x} (pinned FlashInfer XQA via root's C ABI; full-attention layers only; sliding layers stay native)",
            info.splits, info.dynamic_shared_bytes, info.scratch_bytes, info.status_device);
        Ok(Some(Self {
            adapter,
            plan,
            info,
            rows,
        }))
    }

    fn launch(
        &self,
        s: &Arc<CudaStream>,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        committed: u64,
    ) -> Result<()> {
        let rc = unsafe {
            (self.adapter.launch)(
                self.plan,
                q,
                k,
                v,
                out,
                committed,
                1.0,
                s.cu_stream() as *mut c_void,
            )
        };
        ensure!(rc == 0, "nv_laguna_xqa_launch rc={rc} rows={}", self.rows);
        Ok(())
    }

    fn status_after_sync(&self, s: &Arc<CudaStream>) -> Result<()> {
        let mut status = [0u32; 1];
        let view = unsafe { s.upgrade_device_ptr::<u32>(self.info.status_device, 1) };
        let copied = s.memcpy_dtoh(&view, &mut status[..]);
        std::mem::forget(view);
        copied.map_err(|e| anyhow::anyhow!("xqa status dtoh: {e:?}"))?;
        ensure!(
            status[0] == 0,
            "xqa span status {} (committed + rows exceeded capacity; output poisoned)",
            status[0]
        );
        Ok(())
    }
}

impl Drop for XqaPlan {
    fn drop(&mut self) {
        unsafe { (self.adapter.destroy)(self.plan) };
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct NvLagunaXqaFp8Info {
    pub abi_version: u32,
    pub query_rows: u32,
    pub capacity: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub page_size: u32,
    pub splits: u32,
    pub dynamic_shared_bytes: u32,
    pub scratch_alignment: u32,
    pub scratch_bytes: u64,
    pub metadata_bytes: u64,
    pub status_device: u64,
    pub cache_elem_enum: u32,
    pub scale_mode: u32,
    pub k_scale: f32,
    pub v_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NvLagunaXqaFp8CacheView {
    pub cache_elem_enum: u32,
    pub scale_mode: u32,
    pub k_scale: f32,
    pub v_scale: f32,
    pub k: u64,
    pub v: u64,
    pub k_row_scales: u64,
    pub v_row_scales: u64,
}

const XQA_FP8_CACHE_ELEM_ENUM_2_IS_E4M3: u32 = 2;
const XQA_FP8_SCALE_MODE_1_IS_FIXED_ONE: u32 = 1;

type XqaFp8CreateFn = unsafe extern "C" fn(
    *const std::os::raw::c_char,
    u32,
    u32,
    *mut *mut c_void,
    *mut NvLagunaXqaFp8Info,
) -> i32;
type XqaFp8LaunchFn = unsafe extern "C" fn(
    *mut c_void,
    u64,
    *const NvLagunaXqaFp8CacheView,
    u64,
    u64,
    f32,
    *mut c_void,
) -> i32;
type XqaFp8DestroyFn = unsafe extern "C" fn(*mut c_void) -> i32;

struct XqaFp8Adapter {
    create: XqaFp8CreateFn,
    launch: XqaFp8LaunchFn,
    destroy: XqaFp8DestroyFn,
    dir: PathBuf,
}

unsafe impl Send for XqaFp8Adapter {}
unsafe impl Sync for XqaFp8Adapter {}

const XQA_FP8_ADAPTER_IS_ROOTS_FIXED_ONE_E4M3_KV_SPECIALIZATION_OF_THE_PINNED_FLASHINFER_XQA_KERNEL: &str =
    "NV_LAGUNA_XQA_FP8=1 needs NV_LAGUNA_XQA_FP8_DIR pointing at liblaguna_xqa_fp8_adapter.so and laguna-xqa-fp8-m{1,2,3,5}.cubin (temp/q38-coord/xqa-fp8-fixed-scale-candidate/build.sh output)";

fn xqa_fp8_adapter() -> Result<Option<&'static XqaFp8Adapter>> {
    static ADAPTER: OnceLock<Option<XqaFp8Adapter>> = OnceLock::new();
    if !env::var("NV_LAGUNA_XQA_FP8")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return Ok(None);
    }
    let loaded = ADAPTER.get_or_init(|| {
        let dir = PathBuf::from(env::var("NV_LAGUNA_XQA_FP8_DIR").unwrap_or_default());
        let lib = CString::new(
            dir.join("liblaguna_xqa_fp8_adapter.so")
                .to_string_lossy()
                .as_bytes(),
        )
        .ok()?;
        let handle = unsafe { dlopen(lib.as_ptr(), RTLD_NOW_GLIBC | RTLD_GLOBAL_GLIBC) };
        if handle.is_null() {
            eprintln!(
                "[laguna_xqa_fp8] dlopen failed for {}",
                lib.to_string_lossy()
            );
            return None;
        }
        let sym = |name: &str| -> *mut c_void {
            let c = CString::new(name).unwrap();
            unsafe { dlsym(handle, c.as_ptr()) }
        };
        let (c, l, d) = (
            sym("nv_laguna_xqa_fp8_create"),
            sym("nv_laguna_xqa_fp8_launch"),
            sym("nv_laguna_xqa_fp8_destroy"),
        );
        if c.is_null() || l.is_null() || d.is_null() {
            eprintln!(
                "[laguna_xqa_fp8] adapter symbols missing in {}",
                lib.to_string_lossy()
            );
            return None;
        }
        Some(XqaFp8Adapter {
            create: unsafe { std::mem::transmute::<*mut c_void, XqaFp8CreateFn>(c) },
            launch: unsafe { std::mem::transmute::<*mut c_void, XqaFp8LaunchFn>(l) },
            destroy: unsafe { std::mem::transmute::<*mut c_void, XqaFp8DestroyFn>(d) },
            dir,
        })
    });
    match loaded {
        Some(a) => Ok(Some(a)),
        None => bail!("{XQA_FP8_ADAPTER_IS_ROOTS_FIXED_ONE_E4M3_KV_SPECIALIZATION_OF_THE_PINNED_FLASHINFER_XQA_KERNEL}"),
    }
}

pub struct XqaFp8Plan {
    adapter: &'static XqaFp8Adapter,
    plan: *mut c_void,
    info: NvLagunaXqaFp8Info,
    rows: usize,
}

unsafe impl Send for XqaFp8Plan {}

impl XqaFp8Plan {
    pub(crate) fn new_for(
        rows: usize,
        capacity: usize,
        kv_heads: usize,
        q_heads: usize,
        hd: usize,
        policy: LagunaFp8ScalePolicy,
        label: &str,
    ) -> Result<Option<Self>> {
        let Some(adapter) = xqa_fp8_adapter()? else {
            return Ok(None);
        };
        ensure!(policy == LagunaFp8ScalePolicy::FixedOne,
            "NV_LAGUNA_XQA_FP8=1 reads raw e4m3 KV with a fixed unit scale and needs NV_LAGUNA_FP8_UNIT_SCALE=1 at cache creation; the dynamic per-token/head-scaled fp8 cache is incompatible and is never relabeled (got {policy:?})");
        ensure!(matches!(rows, 1 | 2 | 3 | 5) && kv_heads == 8 && q_heads == 48 && hd == 128 && capacity.is_multiple_of(128) && (128..=262144).contains(&capacity),
            "NV_LAGUNA_XQA_FP8=1 supports rows 1, 2, 3 or 5 (m{rows}.cubin), 48/8 heads, hd 128, capacity a multiple of 128 up to 262144 (got rows={rows} capacity={capacity} q={q_heads} kv={kv_heads} hd={hd})");
        let cubin = CString::new(
            adapter
                .dir
                .join(format!("laguna-xqa-fp8-m{rows}.cubin"))
                .to_string_lossy()
                .as_bytes(),
        )?;
        let mut plan: *mut c_void = std::ptr::null_mut();
        let mut info = NvLagunaXqaFp8Info::default();
        let rc = unsafe {
            (adapter.create)(
                cubin.as_ptr(),
                rows as u32,
                capacity as u32,
                &mut plan,
                &mut info,
            )
        };
        ensure!(
            rc == 0 && !plan.is_null(),
            "nv_laguna_xqa_fp8_create rc={rc} rows={rows} capacity={capacity}"
        );
        ensure!(info.cache_elem_enum == XQA_FP8_CACHE_ELEM_ENUM_2_IS_E4M3 && info.scale_mode == XQA_FP8_SCALE_MODE_1_IS_FIXED_ONE && info.k_scale == 1.0 && info.v_scale == 1.0,
            "fp8 XQA plan metadata is not the fixed-one e4m3 contract: elem={} scale_mode={} k_scale={} v_scale={}", info.cache_elem_enum, info.scale_mode, info.k_scale, info.v_scale);
        eprintln!("[laguna_xqa_fp8] route=xqa-fp8-fixed-one graph={label} rows={rows} capacity={capacity} splits={} dynamic_shared_bytes={} scratch_bytes={} status_device={:#x} (root's fixed-unit-scale e4m3 specialization of the pinned FlashInfer XQA kernel; full-attention layers only; numerically unqualified)",
            info.splits, info.dynamic_shared_bytes, info.scratch_bytes, info.status_device);
        Ok(Some(Self {
            adapter,
            plan,
            info,
            rows,
        }))
    }

    pub(crate) fn launch(
        &self,
        s: &Arc<CudaStream>,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        committed: u64,
    ) -> Result<()> {
        let view = NvLagunaXqaFp8CacheView {
            cache_elem_enum: XQA_FP8_CACHE_ELEM_ENUM_2_IS_E4M3,
            scale_mode: XQA_FP8_SCALE_MODE_1_IS_FIXED_ONE,
            k_scale: 1.0,
            v_scale: 1.0,
            k,
            v,
            k_row_scales: 0,
            v_row_scales: 0,
        };
        let rc = unsafe {
            (self.adapter.launch)(
                self.plan,
                q,
                &view,
                out,
                committed,
                1.0,
                s.cu_stream() as *mut c_void,
            )
        };
        ensure!(
            rc == 0,
            "nv_laguna_xqa_fp8_launch rc={rc} rows={}",
            self.rows
        );
        Ok(())
    }

    pub(crate) fn status_after_sync(&self, s: &Arc<CudaStream>) -> Result<()> {
        let mut status = [0u32; 1];
        let view = unsafe { s.upgrade_device_ptr::<u32>(self.info.status_device, 1) };
        let copied = s.memcpy_dtoh(&view, &mut status[..]);
        std::mem::forget(view);
        copied.map_err(|e| anyhow::anyhow!("xqa fp8 status dtoh: {e:?}"))?;
        ensure!(
            status[0] == 0,
            "xqa fp8 span status {} (committed + rows exceeded capacity; output poisoned)",
            status[0]
        );
        Ok(())
    }
}

impl Drop for XqaFp8Plan {
    fn drop(&mut self) {
        unsafe { (self.adapter.destroy)(self.plan) };
    }
}

pub enum StepCache {
    Bf16(LagunaKvCache),
    Fp8(LagunaKvCacheFp8),
}

impl StepCache {
    pub fn current_len(&self) -> usize {
        match self {
            Self::Bf16(c) => c.current_len(),
            Self::Fp8(c) => c.current_len(),
        }
    }
    pub fn max_seq_len(&self) -> usize {
        match self {
            Self::Bf16(c) => c.max_seq_len(),
            Self::Fp8(c) => c.max_seq_len(),
        }
    }
    pub fn is_fp8(&self) -> bool {
        matches!(self, Self::Fp8(_))
    }
    pub fn reset(&mut self) {
        match self {
            Self::Bf16(c) => c.reset(),
            Self::Fp8(c) => c.reset(),
        }
    }
    pub fn rollback(&mut self, n: usize) -> Result<()> {
        match self {
            Self::Bf16(c) => c.rollback(n),
            Self::Fp8(c) => c.rollback(n),
        }
    }
    fn advance(&mut self, n: usize) {
        match self {
            Self::Bf16(c) => c.advance(n),
            Self::Fp8(c) => c.advance(n),
        }
    }
    fn note_graph_write(&mut self) {
        match self {
            Self::Bf16(c) => c.note_graph_write(),
            Self::Fp8(c) => c.note_graph_write(),
        }
    }
    fn prepare_for_decode(&mut self, write_pos: usize, n_total: usize) -> Result<()> {
        match self {
            Self::Bf16(c) => c.prepare_for_decode(write_pos, n_total),
            Self::Fp8(c) => c.prepare_for_decode_dev(write_pos, n_total),
        }
    }
}

const BF16_CACHE_ACCESSORS_STAY_FOR_THE_PREFILL_CACHE_TESTS_IN_LAGUNA_RS_AND_FP8_STEP_GRAPHS_USE_KV_MUT: &str =
    "LagunaStepGraph::cache/cache_mut are the bf16 accessors; this graph owns an fp8 KV cache, use kv()/kv_mut()";

pub const AUX_TAP_AFTER_ATTENTION_OFFSET_10000_ADDED_TO_THE_LAYER_INDEX_TAPS_THE_POST_ATTENTION_RESIDUAL: usize = 10_000;

pub struct LagunaStepGraph {
    model: Arc<Laguna>,
    cache: StepCache,
    verify_split: Option<VerifySplitBufs>,
    fp8: Option<Fp8VerifyState>,
    xqa: Option<XqaPlan>,
    xqa_fp8: Option<XqaFp8Plan>,
    sliding_rows: Option<SlidingVerifyBufs>,
    dev: CudaDevice,
    device: Device,
    forked: Arc<CudaStream>,
    aux_gemm: Arc<CudaStream>,
    aux_shared: Arc<CudaStream>,
    runner: CudaGraphRunner,
    tok_buf: CudaSlice<u32>,
    host_tok: Box<[u32; 1]>,
    cu_q: CudaSlice<i32>,
    cu_full: CudaSlice<i32>,
    cu_slide: CudaSlice<i32>,
    logits_out: CudaSlice<f32>,
    am_part_val: CudaSlice<f32>,
    am_part_idx: CudaSlice<i32>,
    token_out: CudaSlice<u32>,
    meta_ptr: u64,
    layer_kv: Vec<Option<(Tensor, Tensor)>>,
    grouped: Vec<Option<Arc<MoeGroupedWeights>>>,
    moe_ctx: GroupedDecodeContext,
    s_cap: usize,
    captured: bool,
    prof: Option<StepProfStamps>,
    m1_flash: Option<M1FlashBufs>,
    aux_layers: Vec<usize>,
    aux_ts: Vec<Tensor>,

    _err_drain: CtxErrDrain,
}

struct CtxErrDrain(Arc<CudaContext>);

impl Drop for CtxErrDrain {
    fn drop(&mut self) {
        let _ = self.0.check_err();
    }
}

unsafe impl Send for LagunaStepGraph {}

impl LagunaStepGraph {
    pub fn new(model: Arc<Laguna>, cache: LagunaKvCache) -> Result<Self> {
        Self::new_inner(model, StepCache::Bf16(cache), &[])
    }

    pub fn new_with_aux_taps_of_every_listed_layers_output_row(
        model: Arc<Laguna>,
        cache: LagunaKvCache,
        aux_layers: &[usize],
    ) -> Result<Self> {
        Self::new_inner(model, StepCache::Bf16(cache), aux_layers)
    }

    pub fn new_fp8(model: Arc<Laguna>, cache: LagunaKvCacheFp8) -> Result<Self> {
        Self::new_inner(model, StepCache::Fp8(cache), &[])
    }

    pub fn aux(&self) -> &[Tensor] {
        &self.aux_ts
    }

    fn new_inner(model: Arc<Laguna>, cache: StepCache, aux_layers: &[usize]) -> Result<Self> {
        let device = model.device().clone();
        let Device::Cuda(dev) = device.clone() else {
            bail!("LagunaStepGraph requires a CUDA device")
        };
        ensure!(
            model.dtype() == DType::BF16,
            "LagunaStepGraph requires BF16"
        );
        let (meta_ptr, s_cap) = match &cache {
            StepCache::Bf16(c) => {
                ensure!(c.has_ring(), "LagunaStepGraph requires the ring KV cache");
                (
                    c.ring_meta_ptr()
                        .ok_or_else(|| anyhow::anyhow!("ring meta missing"))?,
                    c.sliding_cap().unwrap_or(0),
                )
            }
            StepCache::Fp8(c) => (c.ring_meta_ptr(), c.sliding_cap()),
        };
        ensure!(s_cap > 0, "LagunaStepGraph: no sliding capacity");
        let max_seq_len = cache.max_seq_len();

        let raw_ctx: Arc<CudaContext> = dev.cuda_stream().context().clone();
        let mut ctor_guard = crate::gemma4_batch_graph::graph_teardown::CtorForkGuard::new();
        let forked = ctor_guard
            .fork(&raw_ctx)
            .map_err(|e| anyhow::anyhow!("step graph stream: {e:?}"))?;
        let aux_gemm = ctor_guard
            .fork(&raw_ctx)
            .map_err(|e| anyhow::anyhow!("step graph aux gemm stream: {e:?}"))?;
        let aux_shared = ctor_guard
            .fork(&raw_ctx)
            .map_err(|e| anyhow::anyhow!("step graph aux shared stream: {e:?}"))?;

        let cfg = model.config();
        let tok_buf = forked.alloc_zeros::<u32>(1)?;
        let cu_q = forked.clone_htod(&[0i32, 1i32])?;
        let cu_full = forked.alloc_zeros::<i32>(2)?;
        let cu_slide = forked.alloc_zeros::<i32>(2)?;
        let logits_out = forked.alloc_zeros::<f32>(cfg.vocab_size)?;
        let am_parts = nv_kernels::cuda::argmax_parts().max(1);
        let am_part_val = forked.alloc_zeros::<f32>(am_parts)?;
        let am_part_idx = forked.alloc_zeros::<i32>(am_parts)?;
        let token_out = forked.alloc_zeros::<u32>(1)?;

        let mut layer_kv = Vec::with_capacity(model.layers().len());
        let mut grouped = Vec::with_capacity(model.layers().len());
        let mut fp8_layers: Vec<Option<Fp8VerifyLayer>> = Vec::with_capacity(model.layers().len());
        for (li, layer) in model.layers().iter().enumerate() {
            match &cache {
                StepCache::Bf16(c) => {
                    layer_kv.push(Some(c.layer_kv_bufs(li)));
                    fp8_layers.push(None);
                }
                StepCache::Fp8(c) => {
                    layer_kv.push(c.layer_bf16_bufs(li));
                    fp8_layers.push(c.layer_fp8_ptrs(li).map(Fp8VerifyLayer::from_ptrs));
                    ensure!(
                        layer_kv[li].is_some() != fp8_layers[li].is_some(),
                        "fp8 cache layer {li}: expected exactly one of bf16/fp8 storage"
                    );
                }
            }
            grouped.push(match &layer.ffn {
                LagunaFfn::Moe(moe) => {
                    let w = model
                        .grouped_weights(moe)?
                        .ok_or_else(|| anyhow::anyhow!("grouped MoE weights unavailable"))?;
                    Some(w)
                }
                LagunaFfn::Dense(_) => None,
            });
        }
        let xqa = if cache.is_fp8() {
            None
        } else {
            XqaPlan::new_for(
                1,
                max_seq_len,
                cfg.num_key_value_heads,
                cfg.num_attention_heads,
                cfg.head_dim,
                "m1-step",
            )?
        };
        let xqa_fp8 = match &cache {
            StepCache::Fp8(c) => XqaFp8Plan::new_for(
                1,
                max_seq_len,
                cfg.num_key_value_heads,
                cfg.num_attention_heads,
                cfg.head_dim,
                c.scale_policy(),
                "m1-step",
            )?,
            StepCache::Bf16(_) => None,
        };
        let sliding_rows: Option<SlidingVerifyBufs> = None;
        let (verify_split, fp8) = if let StepCache::Fp8(c) = &cache {
            let fp8_count = fp8_layers.iter().filter(|l| l.is_some()).count();
            let bufs = VerifySplitBufs::new(&model, 1, max_seq_len, &forked, fp8_count, 64)?;
            eprintln!("[laguna_step_graph] route=m1-captured kv=fp8-full-layers-bf16-sliding fp8_layers={fp8_count} attention=split-gqa-forced-by-fp8-kv splits={}", bufs.splits);
            (
                Some(bufs),
                Some(Fp8VerifyState {
                    scale_policy: c.scale_policy(),
                    layers: fp8_layers,
                    n_committed_ptr: meta_ptr,
                }),
            )
        } else if m1_full_attention_through_the_fused_tc_verify_kernel_opt_in_until_petas_no_draft_bracket() {
            let bufs = VerifySplitBufs::new(
                &model,
                1,
                max_seq_len,
                &forked,
                0,
                M1_FUSED_TC_SPLITS_32_MUST_EQUAL_THE_VERIFY_GRAPHS_SO_ROW_J_REDUCES_LIKE_THE_M1_STEP,
            )?;
            ensure!(
                bufs.fused_tc_stages_each_kv_tile_once_for_all_rows_on_tensor_cores,
                "NV_LAGUNA_M1_FUSED_TC=1 needs the fused-tc verify kernel (GQA group 6)"
            );
            eprintln!("[laguna_step_graph] route=m1-captured attention=fused-tc-shared-with-verify splits={}", bufs.splits);
            (Some(bufs), None)
        } else {
            (None, None)
        };
        let weights_fold_shared = grouped
            .iter()
            .flatten()
            .any(|w: &Arc<MoeGroupedWeights>| w.folded_shared);
        let moe_ctx = if weights_fold_shared {
            GroupedDecodeContext::new_folded_shared(
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                cfg.num_experts_per_tok,
                cfg.num_experts,
                1,
                &forked,
            )?
        } else {
            GroupedDecodeContext::new(
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                cfg.num_experts_per_tok,
                cfg.num_experts,
                &forked,
            )?
        };
        forked.synchronize()?;

        let prof = if verify_prof_enabled() {
            let layer_info: Vec<(bool, bool)> = model
                .layers()
                .iter()
                .map(|l| {
                    (
                        matches!(l.self_attn.kind, LayerType::SlidingAttention),
                        matches!(l.ffn, LagunaFfn::Dense(_)),
                    )
                })
                .collect();
            Some(StepProfStamps::new(&forked, layer_info)?)
        } else {
            None
        };
        let m1_flash = if cfg.head_dim == 128
            && cfg
                .num_attention_heads
                .is_multiple_of(cfg.num_key_value_heads)
            && matches!(cfg.num_attention_heads / cfg.num_key_value_heads, 6 | 8)
        {
            let elems = nv_kernels::cuda::laguna_flash_decode_gqa_scratch_elems(
                cfg.num_key_value_heads as i32,
            );
            let rows_sharing_the_m1_kernel = 1usize;
            Some(M1FlashBufs {
                scratch: forked
                    .alloc_zeros::<f32>(elems * rows_sharing_the_m1_kernel)
                    .map_err(|e| anyhow::anyhow!("m1 flash scratch: {e:?}"))?,
                fan_in: forked
                    .alloc_zeros::<u32>(cfg.num_key_value_heads * rows_sharing_the_m1_kernel)
                    .map_err(|e| anyhow::anyhow!("m1 flash fan_in: {e:?}"))?,
            })
        } else {
            None
        };
        let runner = CudaGraphRunner::new(forked.clone());
        ctor_guard.the_built_engine_owns_teardown_now();
        let mut aux_ts = Vec::with_capacity(aux_layers.len());
        for _ in aux_layers {
            aux_ts.push(Tensor::zeros(
                (1usize, 1usize, cfg.hidden_size),
                DType::BF16,
                &device,
            )?);
        }
        Ok(Self {
            model,
            cache,
            verify_split,
            fp8,
            xqa,
            xqa_fp8,
            sliding_rows,
            dev,
            device,
            forked,
            aux_gemm,
            aux_shared,
            runner,
            tok_buf,
            host_tok: Box::new([0u32; 1]),
            cu_q,
            cu_full,
            cu_slide,
            logits_out,
            am_part_val,
            am_part_idx,
            token_out,
            meta_ptr,
            layer_kv,
            grouped,
            moe_ctx,
            s_cap,
            captured: false,
            prof,
            m1_flash,
            aux_layers: aux_layers.to_vec(),
            aux_ts,
            _err_drain: CtxErrDrain(raw_ctx),
        })
    }

    pub fn cache(&self) -> &LagunaKvCache {
        match &self.cache {
            StepCache::Bf16(c) => c,
            StepCache::Fp8(_) => panic!("{BF16_CACHE_ACCESSORS_STAY_FOR_THE_PREFILL_CACHE_TESTS_IN_LAGUNA_RS_AND_FP8_STEP_GRAPHS_USE_KV_MUT}"),
        }
    }

    pub fn cache_mut(&mut self) -> &mut LagunaKvCache {
        match &mut self.cache {
            StepCache::Bf16(c) => c,
            StepCache::Fp8(_) => panic!("{BF16_CACHE_ACCESSORS_STAY_FOR_THE_PREFILL_CACHE_TESTS_IN_LAGUNA_RS_AND_FP8_STEP_GRAPHS_USE_KV_MUT}"),
        }
    }

    pub fn kv(&self) -> &StepCache {
        &self.cache
    }

    pub fn kv_mut(&mut self) -> &mut StepCache {
        &mut self.cache
    }

    pub fn current_len(&self) -> usize {
        self.cache.current_len()
    }

    pub fn captured_node_count(&self) -> usize {
        if self.captured {
            self.runner.cached_node_count()
        } else {
            0
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        nv_layers::cuda_stream::sync_legacy_then_forked(&self.dev, &self.forked)
    }

    pub fn step(&mut self, token: u32) -> Result<()> {
        self.model.apply_attn_w8_for(1);
        let write_pos = self.cache.current_len();
        self.cache
            .prepare_for_decode(write_pos, write_pos + 1)
            .context("step graph prepare_for_decode")?;
        self.host_tok[0] = token;
        let legacy = self.dev.cuda_stream();
        legacy
            .memcpy_htod(&self.host_tok[..], &mut self.tok_buf)
            .map_err(|e| anyhow::anyhow!("htod token: {e:?}"))?;

        let raw_ctx = legacy.context().clone();
        if raw_ctx.is_event_tracking() {
            unsafe { raw_ctx.disable_event_tracking() };
            legacy
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-capture legacy sync: {e:?}"))?;
        }

        let was_captured = self.captured;
        let forked = self.forked.clone();
        let LagunaStepGraph {
            model,
            dev,
            device,
            aux_gemm,
            aux_shared,
            runner,
            tok_buf,
            cu_q,
            cu_full,
            cu_slide,
            logits_out,
            am_part_val,
            am_part_idx,
            token_out,
            meta_ptr,
            layer_kv,
            grouped,
            moe_ctx,
            s_cap,
            cache,
            prof,
            m1_flash,
            verify_split,
            fp8,
            xqa,
            xqa_fp8,
            sliding_rows,
            aux_layers,
            aux_ts,
            ..
        } = self;
        let meta_ptr = *meta_ptr;
        let s_cap = *s_cap;
        let max_seq_len = cache.max_seq_len();
        let hidden = model.config().hidden_size;
        let aux_dst = aux_ts
            .iter()
            .map(|a| bf16_ptr(a, &forked, hidden))
            .collect::<Result<Vec<u64>>>()?;
        let prof = prof.as_ref();
        let m1_flash = m1_flash.as_ref();
        let verify_split = verify_split.as_ref();
        let fp8 = fp8.as_ref();
        let xqa = xqa.as_ref();
        let xqa_fp8 = xqa_fp8.as_ref();
        let sliding_rows = sliding_rows.as_ref();

        let logits_ptr = {
            let (lp, _g) = logits_out.device_ptr(&forked);
            lp
        };
        let mut body = |s: &Arc<CudaStream>, moe_ctx: &mut GroupedDecodeContext| -> Result<()> {
            step_body(
                model,
                dev,
                device,
                s,
                aux_gemm,
                aux_shared,
                tok_buf,
                cu_q,
                cu_full,
                cu_slide,
                logits_ptr,
                Some((&mut *am_part_val, &mut *am_part_idx, &mut *token_out)),
                meta_ptr,
                layer_kv,
                grouped,
                moe_ctx,
                s_cap,
                max_seq_len,
                1,
                aux_layers,
                &aux_dst,
                prof,
                m1_flash,
                verify_split,
                fp8,
                xqa,
                xqa_fp8,
                sliding_rows,
                false,
            )
        };

        if !was_captured {
            legacy
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-warm legacy sync: {e:?}"))?;
            nv_layers::cuda_stream::with_stream(forked.clone(), || body(&forked, moe_ctx))
                .context("step graph warm pass")?;
            forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("warm sync: {e:?}"))?;
        }

        runner
            .run_on(1u64, Some(&legacy), |s| {
                nv_layers::cuda_stream::with_stream(s.clone(), || body(s, moe_ctx))
            })
            .context("step graph capture/replay")?;
        if !was_captured {
            forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("post-capture sync: {e:?}"))?;
            self.captured = true;
        }
        if let Some(p) = self.prof.as_ref() {
            self.dev
                .cuda_stream()
                .synchronize()
                .map_err(|e| anyhow::anyhow!("prof legacy sync: {e:?}"))?;
            p.report("m1_step")?;
        }

        self.cache.advance(1);
        self.cache.note_graph_write();
        Ok(())
    }

    pub fn argmax_device(&self) -> Result<u32> {
        self.synchronize()?;
        if let Some(x) = self.xqa.as_ref() {
            x.status_after_sync(&self.forked)?;
        }
        if let Some(x) = self.xqa_fp8.as_ref() {
            x.status_after_sync(&self.forked)?;
        }
        let out = self
            .forked
            .clone_dtoh(&self.token_out)
            .map_err(|e| anyhow::anyhow!("dtoh token: {e:?}"))?;
        Ok(out[0])
    }

    pub fn logits_host(&self) -> Result<Vec<f32>> {
        self.synchronize()?;
        self.forked
            .clone_dtoh(&self.logits_out)
            .map_err(|e| anyhow::anyhow!("dtoh logits: {e:?}"))
    }
}

impl Drop for LagunaStepGraph {
    fn drop(&mut self) {
        crate::gemma4_batch_graph::graph_teardown::GraphTeardown::new(&self.forked)
            .with_stream(&self.aux_gemm)
            .with_stream(&self.aux_shared)
            .run(|| self.runner.invalidate());
    }
}

pub(crate) struct Fp8VerifyLayer {
    k_fp8: u64,
    v_fp8: u64,
    k_scales: u64,
    v_scales: u64,
}

impl Fp8VerifyLayer {
    fn from_ptrs((k_fp8, v_fp8, k_scales, v_scales): (u64, u64, u64, u64)) -> Self {
        Self {
            k_fp8,
            v_fp8,
            k_scales,
            v_scales,
        }
    }
}

pub(crate) struct Fp8VerifyState {
    scale_policy: LagunaFp8ScalePolicy,
    layers: Vec<Option<Fp8VerifyLayer>>,
    n_committed_ptr: u64,
}

struct VerifySplitBufs {
    scratch: CudaSlice<f32>,
    fan_in: CudaSlice<u32>,
    splits: usize,
    fused_tc_stages_each_kv_tile_once_for_all_rows_on_tensor_cores: bool,
}

#[allow(non_upper_case_globals)]
const FUSED_TC_ROWS_MAX_5_MUST_EQUAL_kMaxRows5_IN_LAGUNA_FLASH_FUSED_TC_CU: usize = 5;
#[allow(non_upper_case_globals)]
const FUSED_TC_GQA_GROUP_6_MUST_EQUAL_kGroup6_IN_LAGUNA_FLASH_FUSED_TC_CU: usize = 6;

impl VerifySplitBufs {
    fn new(
        model: &Laguna,
        t: usize,
        max_seq_len: usize,
        forked: &Arc<CudaStream>,
        fp8_layers: usize,
        splits: usize,
    ) -> Result<Self> {
        let cfg = model.config();
        ensure!(
            cfg.head_dim == 128
                && (1..=32).contains(&t)
                && max_seq_len <= i32::MAX as usize
                && t <= max_seq_len,
            "Laguna split verify requires head_dim=128, 1<=M<=32 and an i32 context capacity"
        );
        let mut max_heads = 0usize;
        for (li, layer) in model.layers().iter().enumerate() {
            if !matches!(layer.self_attn.kind, LayerType::FullAttention) {
                continue;
            }
            let heads = layer.self_attn.num_heads;
            ensure!(
                cfg.num_key_value_heads > 0
                    && heads % cfg.num_key_value_heads == 0
                    && matches!(heads / cfg.num_key_value_heads, 6 | 8),
                "Laguna split verify requires GQA group 6 or 8, layer {li} has {heads}/{}",
                cfg.num_key_value_heads
            );
            max_heads = max_heads.max(heads);
        }
        ensure!(
            max_heads > 0,
            "Laguna split verify requested but no layer has full attention"
        );
        let scratch_elems = t
            .checked_mul(max_heads)
            .and_then(|n| n.checked_mul(splits * (cfg.head_dim + 2)))
            .ok_or_else(|| anyhow::anyhow!("split scratch size overflow"))?;
        let scratch = forked
            .alloc_zeros::<f32>(scratch_elems)
            .map_err(|e| anyhow::anyhow!("split scratch: {e:?}"))?;
        let fan_in = forked
            .alloc_zeros::<u32>(t * cfg.num_key_value_heads)
            .map_err(|e| anyhow::anyhow!("split fan-in: {e:?}"))?;
        let fused_tc = t <= FUSED_TC_ROWS_MAX_5_MUST_EQUAL_kMaxRows5_IN_LAGUNA_FLASH_FUSED_TC_CU
            && max_heads / cfg.num_key_value_heads
                == FUSED_TC_GQA_GROUP_6_MUST_EQUAL_kGroup6_IN_LAGUNA_FLASH_FUSED_TC_CU;
        eprintln!("[laguna_verify_attention] route=split-gqa opt_in=1 splits={splits} rows={t} fused_tc={} max_seq_allocated={max_seq_len} fp8_layers={fp8_layers} kv_rounding=bf16 intermediate_kv_bytes=0 scratch_bytes={}", u8::from(fused_tc), scratch_elems * 4);
        Ok(VerifySplitBufs {
            scratch,
            fan_in,
            splits,
            fused_tc_stages_each_kv_tile_once_for_all_rows_on_tensor_cores: fused_tc,
        })
    }

    fn forward(
        &self,
        stream: &Arc<CudaStream>,
        q: u64,
        kv: (u64, u64, u64, u64),
        output: u64,
        committed: u64,
        heads: usize,
        kv_heads: usize,
        rows: usize,
        capacity: usize,
        kvq: i32,
    ) -> Result<()> {
        let (scratch, _gs) = self.scratch.device_ptr(stream);
        let (fan_in, _gf) = self.fan_in.device_ptr(stream);
        let fp8 = kvq != 0;
        if self.fused_tc_stages_each_kv_tile_once_for_all_rows_on_tensor_cores && fp8 {
            let rc = unsafe {
                nv_kernels::cuda::laguna_flash_decode_verify_fused_tc_fp8(
                    stream.cu_stream() as *mut c_void,
                    q as *const u16,
                    kv.0 as *const u8,
                    kv.1 as *const u8,
                    kv.2 as *const f32,
                    kv.3 as *const f32,
                    output as *mut u16,
                    committed as *const i32,
                    scratch as *mut f32,
                    fan_in as *mut u32,
                    heads as i32,
                    kv_heads as i32,
                    128,
                    rows as i32,
                    capacity as i32,
                    self.splits as i32,
                    128.0f32.powf(-0.5),
                    kvq,
                )
            };
            if rc == 0 {
                return Ok(());
            }
            ensure!(rc == -1, "Laguna fused-tc fp8 verify attention rc={rc}");
        }
        if self.fused_tc_stages_each_kv_tile_once_for_all_rows_on_tensor_cores && !fp8 {
            let rc = unsafe {
                nv_kernels::cuda::laguna_flash_decode_verify_fused_tc(
                    stream.cu_stream() as *mut c_void,
                    q as *const u16,
                    kv.0 as *const u16,
                    kv.1 as *const u16,
                    output as *mut u16,
                    committed as *const i32,
                    scratch as *mut f32,
                    fan_in as *mut u32,
                    heads as i32,
                    kv_heads as i32,
                    128,
                    rows as i32,
                    capacity as i32,
                    self.splits as i32,
                    128.0f32.powf(-0.5),
                )
            };
            ensure!(rc == 0, "Laguna fused-tc split verify attention rc={rc}");
            return Ok(());
        }
        let rc = unsafe {
            nv_kernels::cuda::laguna_flash_decode_verify(
                stream.cu_stream() as *mut c_void,
                q as *const u16,
                kv.0 as *const u8,
                kv.1 as *const u8,
                kv.2 as *const f32,
                kv.3 as *const f32,
                output as *mut u16,
                committed as *const i32,
                scratch as *mut f32,
                fan_in as *mut u32,
                heads as i32,
                kv_heads as i32,
                128,
                rows as i32,
                capacity as i32,
                self.splits as i32,
                128.0f32.powf(-0.5),
                kvq,
            )
        };
        ensure!(rc == 0, "Laguna split verify attention rc={rc} fp8={fp8}");
        Ok(())
    }
}

#[allow(non_upper_case_globals)]
const SLIDING_VERIFY_SPLITS_8_MUST_EQUAL_kLagSlidingVerifySplits8_IN_LAGUNA_STEP_CU: usize = 8;
const SLIDING_VERIFY_MAX_ROWS_8: usize = 8;

struct SlidingVerifyBufs {
    scratch: CudaSlice<f32>,
    fan_in: CudaSlice<u32>,
    rows_per_block: usize,
}

impl SlidingVerifyBufs {
    fn new_opt_in(model: &Laguna, t: usize, forked: &Arc<CudaStream>) -> Result<Option<Self>> {
        let rows_per_block = match env::var("NV_LAGUNA_VERIFY_SLIDING_ROWS") {
            Err(std::env::VarError::NotPresent) => return Ok(None),
            Ok(v) if v == "0" => return Ok(None),
            Ok(v) if v == "1" => 1usize,
            Ok(v) if v == "2" => 2usize,
            Ok(v) if v == "3" => 3usize,
            other => bail!("NV_LAGUNA_VERIFY_SLIDING_ROWS must be unset, 0, 1, 2 or 3 (rows per block; 1 measured fastest at 20 us vs 49 us for 2-3), got {other:?}"),
        };
        if !(2..=SLIDING_VERIFY_MAX_ROWS_8).contains(&t) {
            return Ok(None);
        }
        let cfg = model.config();
        let mut max_heads = 0usize;
        for layer in model.layers() {
            if matches!(layer.self_attn.kind, LayerType::FullAttention) {
                continue;
            }
            let heads = layer.self_attn.num_heads;
            ensure!(cfg.num_key_value_heads > 0 && heads % cfg.num_key_value_heads == 0 && matches!(heads / cfg.num_key_value_heads, 6 | 8) && cfg.head_dim == 128,
                "Laguna sliding rows verify requires GQA group 6 or 8 with head_dim 128 on the sliding layers");
            max_heads = max_heads.max(heads);
        }
        ensure!(
            max_heads > 0 && cfg.sliding_window > 0,
            "Laguna sliding rows verify requested but no sliding layer exists"
        );
        let scratch = forked
            .alloc_zeros::<f32>(
                t * max_heads
                    * SLIDING_VERIFY_SPLITS_8_MUST_EQUAL_kLagSlidingVerifySplits8_IN_LAGUNA_STEP_CU
                    * (cfg.head_dim + 2),
            )
            .map_err(|e| anyhow::anyhow!("sliding rows scratch: {e:?}"))?;
        let fan_in = forked
            .alloc_zeros::<u32>(t * cfg.num_key_value_heads)
            .map_err(|e| anyhow::anyhow!("sliding rows fan-in: {e:?}"))?;
        eprintln!("[laguna_verify_sliding_rows] route=share-kv-window-rows rows={t} rows_per_block={rows_per_block} splits={} window={} (sliding layers at t>=2 through the M-row share-kv flash body with a per-row window instead of FA2 varlen; fp32 online softmax, numerically a candidate)",
            SLIDING_VERIFY_SPLITS_8_MUST_EQUAL_kLagSlidingVerifySplits8_IN_LAGUNA_STEP_CU, cfg.sliding_window);
        Ok(Some(Self {
            scratch,
            fan_in,
            rows_per_block,
        }))
    }
}
#[allow(non_upper_case_globals)]
const VERIFY_ROWS_GEMV_MAX_ROWS_8_MUST_EQUAL_kLagRowsGemvMaxRows8_IN_LAGUNA_STEP_CU: usize = 8;

fn verify_rows_gemv_opt_in() -> bool {
    env::var("NV_LAGUNA_VERIFY_ROWS_GEMV")
        .map(|v| v == "1")
        .unwrap_or(false)
}

const Q8_ROWS_MAX_DYNAMIC_SMEM_98304: usize = 98304;

fn sliding_rows_on_the_m1_flash_kernel_opt_in_1_since_the_2026_09_14_served_identity_ruling_because_its_kernel_vs_kernel_bitwise_gate_did_not_hold_on_the_served_stream_unset_is_fa2_rows(
) -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_LAGUNA_VERIFY_SLIDING_M1ROWS")
}

const RSTD_FOLD_ROWS_MAX_1_THE_IN_KERNEL_REDUCTION_COSTS_30_VS_24_US_AT_5_ROWS: usize = 1;

fn qkvo_rstd_fuse_opt_in() -> bool {
    env::var("NV_LAGUNA_QKVO_RSTD_FUSE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn step_body(
    model: &Laguna,
    dev: &CudaDevice,
    device: &Device,
    s: &Arc<CudaStream>,
    aux_gemm: &Arc<CudaStream>,
    aux_shared: &Arc<CudaStream>,
    tok_buf: &CudaSlice<u32>,
    cu_q: &CudaSlice<i32>,
    cu_full: &mut CudaSlice<i32>,
    cu_slide: &mut CudaSlice<i32>,
    logits_ptr: u64,
    argmax_slots: Option<(
        &mut CudaSlice<f32>,
        &mut CudaSlice<i32>,
        &mut CudaSlice<u32>,
    )>,
    meta_ptr: u64,
    layer_kv: &[Option<(Tensor, Tensor)>],
    grouped: &[Option<Arc<MoeGroupedWeights>>],
    moe_ctx: &mut GroupedDecodeContext,
    s_cap: usize,
    max_seq_len: usize,
    t: usize,
    aux_layers: &[usize],
    aux_dst: &[u64],
    prof: Option<&StepProfStamps>,
    m1_flash: Option<&M1FlashBufs>,
    verify_split: Option<&VerifySplitBufs>,
    fp8: Option<&Fp8VerifyState>,
    xqa: Option<&XqaPlan>,
    xqa_fp8: Option<&XqaFp8Plan>,
    sliding_rows: Option<&SlidingVerifyBufs>,
    spec_head: bool,
) -> Result<()> {
    let cfg = model.config();
    let hidden = cfg.hidden_size;
    let n_kv = cfg.num_key_value_heads;
    let hd = cfg.head_dim;
    ensure!(moe_ctx.n_tokens() == t, "moe ctx n_tokens != t");
    ensure!(aux_layers.len() == aux_dst.len(), "aux slot mismatch");
    if let Some(p) = prof {
        ensure!(
            p.n_layers == model.layers().len(),
            "prof events sized wrong"
        );
        p.record(ProfPoint::Start, s)?;
    }

    {
        let (fp, _g1) = cu_full.device_ptr_mut(s);
        let (sp, _g2) = cu_slide.device_ptr_mut(s);
        let rc = unsafe {
            nv_kernels::cuda::laguna_seqlens_prep(
                s.cu_stream() as *mut c_void,
                meta_ptr as *const i32,
                fp as *mut i32,
                sp as *mut i32,
                t as i32,
            )
        };
        ensure!(rc == 0, "seqlens prep rc={rc}");
    }

    let tok_clone = tok_buf
        .try_clone()
        .map_err(|e| anyhow::anyhow!("clone tok_buf: {e:?}"))?;
    let tokens_t = {
        let storage = candle_core::CudaStorage::wrap_cuda_slice(tok_clone, dev.clone());
        Tensor::from_storage(
            Storage::Cuda(storage),
            (t,),
            candle_core::op::BackpropOp::none(),
            false,
        )
    };
    let mut x = crate::laguna::LagunaEmbedding::lookup_captured(model, &tokens_t, device)?
        .reshape((1usize, t, hidden))?;
    if let Some(p) = prof {
        p.record(ProfPoint::Embed, s)?;
    }

    let (cu_q_ptr, _gq) = cu_q.device_ptr(s);
    let (cu_full_ptr, _gf) = cu_full.device_ptr(s);
    let (cu_slide_ptr, _gs) = cu_slide.device_ptr(s);

    for (li, layer) in model.layers().iter().enumerate() {
        let attn = &layer.self_attn;
        let n_q = attn.num_heads;

        let g_lin = attn.g_proj.as_ref();
        let qkv_q8 = attn.m1_qkv_w8_fused(hidden, n_q, n_kv, hd);
        let fp8_qkvg = match (
            attn.q_proj
                .fp8_e4m3_row_weight_and_scales_so_gdn_prenorm_folds_into_gemv_e4m3_mk_h(),
            attn.k_proj
                .fp8_e4m3_row_weight_and_scales_so_gdn_prenorm_folds_into_gemv_e4m3_mk_h(),
            attn.v_proj
                .fp8_e4m3_row_weight_and_scales_so_gdn_prenorm_folds_into_gemv_e4m3_mk_h(),
            attn.fp8_gate_weight_and_scales(hidden, n_q),
        ) {
            (Some(q), Some(k), Some(v), Some(g))
                if attn.q_proj.in_features() == hidden
                    && attn.q_proj.out_features() == n_q * hd
                    && attn.k_proj.in_features() == hidden
                    && attn.k_proj.out_features() == n_kv * hd
                    && attn.v_proj.in_features() == hidden
                    && attn.v_proj.out_features() == n_kv * hd =>
            {
                Some((q, k, v, g))
            }
            _ => None,
        };
        let bf16_fused_norms = !layer.input_layernorm.fp8_e4m3_active()
            && !attn.q_norm.fp8_e4m3_active()
            && !attn.k_norm.fp8_e4m3_active();
        let m1_fused = bf16_fused_norms
            && t == 1
            && hd.is_multiple_of(32)
            && hd <= 512
            && hidden.is_multiple_of(8)
            && hidden <= 4096
            && (!attn.m1_w8_active_qkv() || qkv_q8.is_some())
            && attn.q_proj.weight().is_some()
            && attn.k_proj.weight().is_some()
            && attn.v_proj.weight().is_some()
            && g_lin.map(|g| g.weight().is_some() && g.bias().is_none()) == Some(true)
            && attn.q_proj.bias().is_none()
            && attn.k_proj.bias().is_none()
            && attn.v_proj.bias().is_none();
        if t >= 2 && verify_rows_gemv_opt_in() {
            static ROWS_GEMV_PRECONDITIONS_ONCE: Once = Once::new();
            ROWS_GEMV_PRECONDITIONS_ONCE.call_once(|| eprintln!(
                "[laguna_verify_projections] rows-gemv preconditions layer {li} t={t}: hd={hd} hidden={hidden} w8_qkv={} q_bf16={} k_bf16={} v_bf16={} g_present={} g_bf16={} g_bias_none={} qkv_bias_none={} o_bf16={} w8_o={}",
                attn.m1_w8_active_qkv(), attn.q_proj.weight().is_some(), attn.k_proj.weight().is_some(), attn.v_proj.weight().is_some(),
                g_lin.is_some(), g_lin.map(|g| g.weight().is_some()) == Some(true), g_lin.map(|g| g.bias().is_none()) == Some(true),
                attn.q_proj.bias().is_none() && attn.k_proj.bias().is_none() && attn.v_proj.bias().is_none(),
                attn.o_proj.weight().is_some(), attn.m1_w8_active_o()));
        }
        let rows_shape_ok = bf16_fused_norms
            && (2..=VERIFY_ROWS_GEMV_MAX_ROWS_8_MUST_EQUAL_kLagRowsGemvMaxRows8_IN_LAGUNA_STEP_CU)
                .contains(&t)
            && hd.is_multiple_of(32)
            && hd <= 512
            && hidden.is_multiple_of(16)
            && hidden <= 4096
            && g_lin.map(|g| g.weight().is_some() && g.bias().is_none()) == Some(true)
            && attn.q_proj.bias().is_none()
            && attn.k_proj.bias().is_none()
            && attn.v_proj.bias().is_none();
        let rows_common = rows_shape_ok && verify_rows_gemv_opt_in();
        let rows_w8 = rows_shape_ok
            && attn
                .m1_qkv_w8_fused(hidden, n_q, n_kv, hd)
                .map(|(_, _, _, is_fp8)| is_fp8)
                == Some(true);
        let rows_gemv = rows_common
            && !rows_w8
            && !attn.m1_w8_active_qkv()
            && attn.q_proj.weight().is_some()
            && attn.k_proj.weight().is_some()
            && attn.v_proj.weight().is_some();

        let fp8_prepared_rows = (1
            ..=VERIFY_ROWS_GEMV_MAX_ROWS_8_MUST_EQUAL_kLagRowsGemvMaxRows8_IN_LAGUNA_STEP_CU)
            .contains(&t)
            && t * hidden * 2 <= Q8_ROWS_MAX_DYNAMIC_SMEM_98304
            && hd.is_multiple_of(32)
            && hd <= 256
            && hidden.is_multiple_of(16)
            && hidden <= 4096
            && fp8_qkvg.is_some()
            && attn.q_proj.bias().is_none()
            && attn.k_proj.bias().is_none()
            && attn.v_proj.bias().is_none()
            && g_lin.is_some_and(|g| g.bias().is_none())
            && layer.input_layernorm.fp8_e4m3_active()
            && attn.q_norm.fp8_e4m3_active()
            && attn.k_norm.fp8_e4m3_active();

        let (rope, rotary_dim, rot_scale) = model.rope_for_kind(attn.kind);
        let cos_c = rope.cos().contiguous()?;
        let sin_c = rope.sin().contiguous()?;
        let cos_ptr = {
            let (cs, cl) = cos_c.storage_and_layout();
            let c_cuda = match &*cs {
                Storage::Cuda(st) => st,
                _ => bail!("rope cos not CUDA"),
            };
            let c_slice = c_cuda.as_cuda_slice::<f32>()?;
            let view = c_slice.slice(cl.start_offset()..);
            let (p, _g) = view.device_ptr(s);
            p
        };
        let sin_ptr = {
            let (ss_, sl) = sin_c.storage_and_layout();
            let s_cuda = match &*ss_ {
                Storage::Cuda(st) => st,
                _ => bail!("rope sin not CUDA"),
            };
            let s_slice = s_cuda.as_cuda_slice::<f32>()?;
            let view = s_slice.slice(sl.start_offset()..);
            let (p, _g) = view.device_ptr(s);
            p
        };
        let mut q_rot_dev: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * n_q * hd)? };
        let mut k_rot_dev: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * n_kv * hd)? };
        let mut v_slice: Option<CudaSlice<bf16>> = None;
        let mut g_pre: Option<CudaSlice<bf16>> = None;
        let mut v_tensor: Option<Tensor> = None;
        let mut normed_tensor: Option<Tensor> = None;
        let mut gate_from_the_fused_qkv_twin: Option<Tensor> = None;

        if fp8_prepared_rows {
            static FP8_PREPARED_ROWS_FIRED_ONCE: Once = Once::new();
            FP8_PREPARED_ROWS_FIRED_ONCE.call_once(|| eprintln!("[laguna_fp8_prepared_rows] rows={t} qkvg_fp8_norm_fused=true qk_normrope_rows=true matrix_weight_bits=8 norm_weight_bits=8"));
            let x_ptr = bf16_ptr(&x, s, t * hidden)?;
            let mut q_raw: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * n_q * hd)? };
            let mut k_raw: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * n_kv * hd)? };
            let mut v_dev: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * n_kv * hd)? };
            let mut g_dev: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * n_q)? };
            {
                let (q, k, v, g) = fp8_qkvg.expect("FP8 prepared rows predicate");
                let (bq, _qg) = q.0.device_ptr(s);
                let (sq, _qsg) = q.1.device_ptr(s);
                let (bk, _kg) = k.0.device_ptr(s);
                let (sk, _ksg) = k.1.device_ptr(s);
                let (bv, _vg) = v.0.device_ptr(s);
                let (sv, _vsg) = v.1.device_ptr(s);
                let (bg, _gg) = g.0.device_ptr(s);
                let (sg, _gsg) = g.1.device_ptr(s);
                let (norm_weight, norm_scale, width) = layer
                    .input_layernorm
                    .fp8_e4m3_weight_and_scale()
                    .expect("FP8 prepared input norm predicate");
                ensure!(
                    width == hidden && norm_weight.len() == hidden && norm_scale.len() == 1,
                    "FP8 prepared input norm storage does not match hidden={hidden}"
                );
                let (wn, _ng) = norm_weight.device_ptr(s);
                let (sn, _nsg) = norm_scale.device_ptr(s);
                let (yq, _yqg) = q_raw.device_ptr_mut(s);
                let (yk, _ykg) = k_raw.device_ptr_mut(s);
                let (yv, _yvg) = v_dev.device_ptr_mut(s);
                let (yg, _ygg) = g_dev.device_ptr_mut(s);
                let mut rstd: Option<CudaSlice<f32>> = if t > 1 {
                    Some(unsafe { s.alloc::<f32>(t)? })
                } else {
                    None
                };
                let rc = if let Some(rstd) = rstd.as_mut() {
                    let (rp, _rpg) = rstd.device_ptr_mut(s);
                    let rc = unsafe {
                        nv_kernels::cuda::laguna_rstd256_bf16_rows(
                            s.cu_stream() as *mut c_void,
                            x_ptr as *const u16,
                            rp as *mut f32,
                            hidden as i32,
                            layer.input_layernorm.eps() as f32,
                            t as i32,
                        )
                    };
                    ensure!(rc == 0, "FP8 prepared rstd rc={rc} layer {li} rows={t}");
                    unsafe {
                        nv_kernels::cuda::laguna_gemv_fp8_qkvg_rows_rstd(
                            s.cu_stream() as *mut c_void,
                            bq as *const u8,
                            sq as *const f32,
                            bk as *const u8,
                            sk as *const f32,
                            bv as *const u8,
                            sv as *const f32,
                            bg as *const u8,
                            sg as *const f32,
                            x_ptr as *const u16,
                            wn as *const c_void,
                            sn as *const f32,
                            rp as *const f32,
                            yq as *mut u16,
                            yk as *mut u16,
                            yv as *mut u16,
                            yg as *mut u16,
                            (n_q * hd) as i32,
                            (n_kv * hd) as i32,
                            (n_kv * hd) as i32,
                            n_q as i32,
                            hidden as i32,
                            t as i32,
                        )
                    }
                } else {
                    unsafe {
                        nv_kernels::cuda::laguna_gemv_fp8_qkvg_fp8_norm_rows(
                            s.cu_stream() as *mut c_void,
                            bq as *const u8,
                            sq as *const f32,
                            bk as *const u8,
                            sk as *const f32,
                            bv as *const u8,
                            sv as *const f32,
                            bg as *const u8,
                            sg as *const f32,
                            x_ptr as *const u16,
                            wn as *const u8,
                            sn as *const f32,
                            layer.input_layernorm.eps() as f32,
                            yq as *mut u16,
                            yk as *mut u16,
                            yv as *mut u16,
                            yg as *mut u16,
                            (n_q * hd) as i32,
                            (n_kv * hd) as i32,
                            (n_kv * hd) as i32,
                            n_q as i32,
                            hidden as i32,
                            t as i32,
                        )
                    }
                };
                ensure!(rc == 0, "FP8 prepared qkvg rc={rc} layer {li} rows={t}");
            }
            if let Some(p) = prof {
                p.record(ProfPoint::Qkv(li), s)?;
            }
            {
                let (qw, qs, qh) = attn
                    .q_norm
                    .fp8_e4m3_weight_and_scale()
                    .expect("FP8 prepared Q norm predicate");
                let (kw, ks, kh) = attn
                    .k_norm
                    .fp8_e4m3_weight_and_scale()
                    .expect("FP8 prepared K norm predicate");
                ensure!(
                    qh == hd
                        && kh == hd
                        && qw.len() == hd
                        && kw.len() == hd
                        && qs.len() == 1
                        && ks.len() == 1,
                    "FP8 prepared Q/K norm storage does not match head_dim={hd}"
                );
                let (qi, _qig) = q_raw.device_ptr(s);
                let (ki, _kig) = k_raw.device_ptr(s);
                let (qo, _qog) = q_rot_dev.device_ptr_mut(s);
                let (ko, _kog) = k_rot_dev.device_ptr_mut(s);
                let (qw, _qwg) = qw.device_ptr(s);
                let (qs, _qsg) = qs.device_ptr(s);
                let (kw, _kwg) = kw.device_ptr(s);
                let (ks, _ksg) = ks.device_ptr(s);
                let normrope = if t == 1 {
                    nv_kernels::cuda::laguna_qk_normrope_fp8_weight_rows
                } else {
                    nv_kernels::cuda::laguna_qk_normrope_fp8_weight_base_rows
                };
                let rc = unsafe {
                    normrope(
                        s.cu_stream() as *mut c_void,
                        qi as *const u16,
                        ki as *const u16,
                        qo as *mut u16,
                        ko as *mut u16,
                        qw as *const u8,
                        qs as *const f32,
                        kw as *const u8,
                        ks as *const f32,
                        cos_ptr as *const f32,
                        sin_ptr as *const f32,
                        meta_ptr as *const i32,
                        n_q as i32,
                        n_kv as i32,
                        hd as i32,
                        rotary_dim as i32,
                        rot_scale,
                        attn.q_norm.eps() as f32,
                        attn.k_norm.eps() as f32,
                        t as i32,
                    )
                };
                ensure!(rc == 0, "FP8 prepared normrope rc={rc} layer {li} rows={t}");
            }
            if let Some(p) = prof {
                p.record(ProfPoint::Rope(li), s)?;
            }
            v_slice = Some(v_dev);
            g_pre = Some(g_dev);
        } else if m1_fused {
            let x_ptr = bf16_ptr(&x, s, t * hidden)?;
            let mut rstd_dev: CudaSlice<f32> = unsafe { s.alloc::<f32>(1)? };
            let rstd_in_gemv = qkvo_rstd_fuse_opt_in() && qkv_q8.map(|q| q.3) == Some(true);
            if !rstd_in_gemv {
                let (rp, _g) = rstd_dev.device_ptr_mut(s);
                let rc = unsafe {
                    nv_kernels::cuda::laguna_rstd256_bf16(
                        s.cu_stream() as *mut c_void,
                        x_ptr as *const u16,
                        rp as *mut f32,
                        hidden as i32,
                        layer.input_layernorm.eps() as f32,
                    )
                };
                ensure!(rc == 0, "m1 rstd rc={rc} layer {li}");
            }
            let mut q_raw: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(n_q * hd)? };
            let mut k_raw: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(n_kv * hd)? };
            let mut v_dev: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(n_kv * hd)? };
            let mut g_dev: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(n_q)? };
            {
                let wq = bf16_ptr(attn.q_proj.weight().unwrap(), s, n_q * hd * hidden)?;
                let wk = bf16_ptr(attn.k_proj.weight().unwrap(), s, n_kv * hd * hidden)?;
                let wv = bf16_ptr(attn.v_proj.weight().unwrap(), s, n_kv * hd * hidden)?;
                let wg = bf16_ptr(g_lin.unwrap().weight().unwrap(), s, n_q * hidden)?;
                let wn = bf16_ptr(layer.input_layernorm.weight_bf16(), s, hidden)?;
                let (rp, _g0) = rstd_dev.device_ptr(s);
                let (yq, _g1) = q_raw.device_ptr_mut(s);
                let (yk, _g2) = k_raw.device_ptr_mut(s);
                let (yv, _g3) = v_dev.device_ptr_mut(s);
                let (yg, _g4) = g_dev.device_ptr_mut(s);
                let rc = if let Some((pq, pk, pv, is_fp8)) = qkv_q8 {
                    let (bq, sq) = (w8_bytes_ptr(pq, s), w8_scale_ptr(pq, s));
                    let (bk, sk) = (w8_bytes_ptr(pk, s), w8_scale_ptr(pk, s));
                    let (bv, sv) = (w8_bytes_ptr(pv, s), w8_scale_ptr(pv, s));
                    if is_fp8 && rstd_in_gemv {
                        static QKVO_RSTD_M1_FIRED_ONCE: Once = Once::new();
                        QKVO_RSTD_M1_FIRED_ONCE.call_once(|| eprintln!("[laguna_qkvo_rows] m1 rstd folded into laguna_gemv_q8_qkvg_normed_rows_rstd_fused (no rstd launch; bitwise to laguna_rstd256_bf16 + the fused rows GEMV, LAGUNA-Q8-ROWS gate)"));
                        unsafe {
                            nv_kernels::cuda::laguna_gemv_q8_qkvg_normed_rows_rstd_fused(
                                s.cu_stream() as *mut c_void,
                                1,
                                bq as *const c_void,
                                sq as *const f32,
                                bk as *const c_void,
                                sk as *const f32,
                                bv as *const c_void,
                                sv as *const f32,
                                wg as *const u16,
                                x_ptr as *const u16,
                                wn as *const u16,
                                layer.input_layernorm.eps() as f32,
                                yq as *mut u16,
                                yk as *mut u16,
                                yv as *mut u16,
                                yg as *mut u16,
                                (n_q * hd) as i32,
                                (n_kv * hd) as i32,
                                (n_kv * hd) as i32,
                                n_q as i32,
                                hidden as i32,
                                1,
                            )
                        }
                    } else if is_fp8 {
                        static QKVO_ROWS_M1_FIRED_ONCE: Once = Once::new();
                        QKVO_ROWS_M1_FIRED_ONCE.call_once(|| eprintln!("[laguna_qkvo_rows] m1 qkv+gate through laguna_gemv_q8_qkvg_normed_rows rows=1 (bitwise to the shared fused q8 kernel, LAGUNA-Q8-ROWS gate)"));
                        unsafe {
                            nv_kernels::cuda::laguna_gemv_q8_qkvg_normed_rows(
                                s.cu_stream() as *mut c_void,
                                1,
                                bq as *const c_void,
                                sq as *const f32,
                                bk as *const c_void,
                                sk as *const f32,
                                bv as *const c_void,
                                sv as *const f32,
                                wg as *const u16,
                                x_ptr as *const u16,
                                wn as *const u16,
                                rp as *const f32,
                                yq as *mut u16,
                                yk as *mut u16,
                                yv as *mut u16,
                                yg as *mut u16,
                                (n_q * hd) as i32,
                                (n_kv * hd) as i32,
                                (n_kv * hd) as i32,
                                n_q as i32,
                                hidden as i32,
                                1,
                            )
                        }
                    } else {
                        unsafe {
                            nv_kernels::cuda::gemv_q8_qkvg_normed(
                                s.cu_stream() as *mut c_void,
                                is_fp8 as i32,
                                bq as *const c_void,
                                sq as *const f32,
                                bk as *const c_void,
                                sk as *const f32,
                                bv as *const c_void,
                                sv as *const f32,
                                wg as *const u16,
                                x_ptr as *const u16,
                                wn as *const u16,
                                rp as *const f32,
                                yq as *mut u16,
                                yk as *mut u16,
                                yv as *mut u16,
                                yg as *mut u16,
                                (n_q * hd) as i32,
                                (n_kv * hd) as i32,
                                (n_kv * hd) as i32,
                                n_q as i32,
                                hidden as i32,
                            )
                        }
                    }
                } else {
                    unsafe {
                        nv_kernels::cuda::gemv_bf16_qkvg_normed(
                            s.cu_stream() as *mut c_void,
                            wq as *const u16,
                            wk as *const u16,
                            wv as *const u16,
                            wg as *const u16,
                            x_ptr as *const u16,
                            wn as *const u16,
                            rp as *const f32,
                            yq as *mut u16,
                            yk as *mut u16,
                            yv as *mut u16,
                            yg as *mut u16,
                            (n_q * hd) as i32,
                            (n_kv * hd) as i32,
                            (n_kv * hd) as i32,
                            n_q as i32,
                            hidden as i32,
                        )
                    }
                };
                ensure!(rc == 0, "m1 qkvg gemv rc={rc} layer {li}");
            }
            if let Some(p) = prof {
                p.record(ProfPoint::Qkv(li), s)?;
            }
            {
                let qw = bf16_ptr(attn.q_norm.weight_bf16(), s, hd)?;
                let kw = bf16_ptr(attn.k_norm.weight_bf16(), s, hd)?;
                let (qi, _g1) = q_raw.device_ptr(s);
                let (ki, _g2) = k_raw.device_ptr(s);
                let (qo, _g3) = q_rot_dev.device_ptr_mut(s);
                let (ko, _g4) = k_rot_dev.device_ptr_mut(s);
                let rc = unsafe {
                    nv_kernels::cuda::laguna_qk_normrope_bf16(
                        s.cu_stream() as *mut c_void,
                        qi as *const u16,
                        ki as *const u16,
                        qo as *mut u16,
                        ko as *mut u16,
                        qw as *const u16,
                        kw as *const u16,
                        cos_ptr as *const f32,
                        sin_ptr as *const f32,
                        meta_ptr as *const i32,
                        n_q as i32,
                        n_kv as i32,
                        hd as i32,
                        rotary_dim as i32,
                        rot_scale,
                        attn.q_norm.eps() as f32,
                        attn.k_norm.eps() as f32,
                    )
                };
                ensure!(rc == 0, "m1 normrope rc={rc} layer {li}");
            }
            if let Some(p) = prof {
                p.record(ProfPoint::Rope(li), s)?;
            }
            v_slice = Some(v_dev);
            g_pre = Some(g_dev);
        } else if rows_w8 {
            static ROWS_W8_FIRED_ONCE: Once = Once::new();
            ROWS_W8_FIRED_ONCE.call_once(|| eprintln!("[laguna_verify_projections] rows-w8 path fired: layer {li} t={t} (q/k/v through gemv_e4m3_mk_h with the M=1 rstd kernel and raw x, gate through the M-row bf16 GEMV; the M=1 fused q8 kernel's normalization and accumulation order)"));
            let (pq, pk, pv, _) = attn
                .m1_qkv_w8_fused(hidden, n_q, n_kv, hd)
                .expect("rows_w8 checked the fused w8 projections");
            let x_ptr = bf16_ptr(&x, s, t * hidden)?;
            let mut rstd_dev: CudaSlice<f32> = unsafe { s.alloc::<f32>(t)? };
            let rstd_in_gemv = qkvo_rstd_fuse_opt_in()
                && t <= RSTD_FOLD_ROWS_MAX_1_THE_IN_KERNEL_REDUCTION_COSTS_30_VS_24_US_AT_5_ROWS;
            if rstd_in_gemv {
                static QKVO_RSTD_VERIFY_FIRED_ONCE: Once = Once::new();
                QKVO_RSTD_VERIFY_FIRED_ONCE.call_once(|| eprintln!("[laguna_qkvo_rows] verify t={t} rstd folded into laguna_gemv_q8_qkvg_normed_rows_rstd_fused (no rstd launch)"));
            } else {
                static QKVO_ROWS_VERIFY_FIRED_ONCE: Once = Once::new();
                QKVO_ROWS_VERIFY_FIRED_ONCE.call_once(|| eprintln!("[laguna_qkvo_rows] verify t={t} qkv+gate through laguna_rstd256_bf16_rows + laguna_gemv_q8_qkvg_normed_rows (one launch each, bitwise to the rows-w8 five-launch form, LAGUNA-Q8-ROWS gate)"));
                let (rp, _g) = rstd_dev.device_ptr_mut(s);
                let rc = unsafe {
                    nv_kernels::cuda::laguna_rstd256_bf16_rows(
                        s.cu_stream() as *mut c_void,
                        x_ptr as *const u16,
                        rp as *mut f32,
                        hidden as i32,
                        layer.input_layernorm.eps() as f32,
                        t as i32,
                    )
                };
                ensure!(rc == 0, "rows-w8 rstd rows rc={rc} layer {li} t={t}");
            }
            let mut q_raw: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * n_q * hd)? };
            let mut k_raw: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * n_kv * hd)? };
            let mut v_dev: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * n_kv * hd)? };
            let mut g_dev: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * n_q)? };
            {
                let wn = bf16_ptr(layer.input_layernorm.weight_bf16(), s, hidden)?;
                let (rp, _g0) = rstd_dev.device_ptr(s);
                let wg = bf16_ptr(g_lin.unwrap().weight().unwrap(), s, n_q * hidden)?;
                let (bq, sq) = (w8_bytes_ptr(pq, s), w8_scale_ptr(pq, s));
                let (bk, sk) = (w8_bytes_ptr(pk, s), w8_scale_ptr(pk, s));
                let (bv, sv) = (w8_bytes_ptr(pv, s), w8_scale_ptr(pv, s));
                let (yq, _g1) = q_raw.device_ptr_mut(s);
                let (yk, _g2) = k_raw.device_ptr_mut(s);
                let (yv, _g3) = v_dev.device_ptr_mut(s);
                let (yg, _g4) = g_dev.device_ptr_mut(s);
                let rc = if rstd_in_gemv {
                    unsafe {
                        nv_kernels::cuda::laguna_gemv_q8_qkvg_normed_rows_rstd_fused(
                            s.cu_stream() as *mut c_void,
                            1,
                            bq as *const c_void,
                            sq as *const f32,
                            bk as *const c_void,
                            sk as *const f32,
                            bv as *const c_void,
                            sv as *const f32,
                            wg as *const u16,
                            x_ptr as *const u16,
                            wn as *const u16,
                            layer.input_layernorm.eps() as f32,
                            yq as *mut u16,
                            yk as *mut u16,
                            yv as *mut u16,
                            yg as *mut u16,
                            (n_q * hd) as i32,
                            (n_kv * hd) as i32,
                            (n_kv * hd) as i32,
                            n_q as i32,
                            hidden as i32,
                            t as i32,
                        )
                    }
                } else {
                    unsafe {
                        nv_kernels::cuda::laguna_gemv_q8_qkvg_normed_rows(
                            s.cu_stream() as *mut c_void,
                            1,
                            bq as *const c_void,
                            sq as *const f32,
                            bk as *const c_void,
                            sk as *const f32,
                            bv as *const c_void,
                            sv as *const f32,
                            wg as *const u16,
                            x_ptr as *const u16,
                            wn as *const u16,
                            rp as *const f32,
                            yq as *mut u16,
                            yk as *mut u16,
                            yv as *mut u16,
                            yg as *mut u16,
                            (n_q * hd) as i32,
                            (n_kv * hd) as i32,
                            (n_kv * hd) as i32,
                            n_q as i32,
                            hidden as i32,
                            t as i32,
                        )
                    }
                };
                ensure!(
                    rc == 0,
                    "rows-w8 fused q8 rows gemv rc={rc} layer {li} t={t}"
                );
            }
            if let Some(p) = prof {
                p.record(ProfPoint::Qkv(li), s)?;
            }
            let qw = bf16_ptr(attn.q_norm.weight_bf16(), s, hd)?;
            let kw = bf16_ptr(attn.k_norm.weight_bf16(), s, hd)?;
            let (qi, _g1) = q_raw.device_ptr(s);
            let (ki, _g2) = k_raw.device_ptr(s);
            let (qo, _g3) = q_rot_dev.device_ptr_mut(s);
            let (ko, _g4) = k_rot_dev.device_ptr_mut(s);
            let rc = unsafe {
                nv_kernels::cuda::laguna_qk_normrope_bf16_base_rows(
                    s.cu_stream() as *mut c_void,
                    qi as *const u16,
                    ki as *const u16,
                    qo as *mut u16,
                    ko as *mut u16,
                    qw as *const u16,
                    kw as *const u16,
                    cos_ptr as *const f32,
                    sin_ptr as *const f32,
                    meta_ptr as *const i32,
                    n_q as i32,
                    n_kv as i32,
                    hd as i32,
                    rotary_dim as i32,
                    rot_scale,
                    attn.q_norm.eps() as f32,
                    attn.k_norm.eps() as f32,
                    t as i32,
                )
            };
            ensure!(rc == 0, "rows-w8 normrope rows rc={rc} layer {li} t={t}");
            if let Some(p) = prof {
                p.record(ProfPoint::Rope(li), s)?;
            }
            v_slice = Some(v_dev);
            g_pre = Some(g_dev);
        } else if rows_gemv {
            static ROWS_GEMV_FIRED_ONCE: Once = Once::new();
            ROWS_GEMV_FIRED_ONCE.call_once(|| eprintln!("[laguna_verify_projections] rows-gemv path fired: layer {li} t={t} (qkv+gate and o_proj through the M-row fused GEMVs)"));
            let x_ptr = bf16_ptr(&x, s, t * hidden)?;
            let mut rstd_dev: CudaSlice<f32> = unsafe { s.alloc::<f32>(t)? };
            for r in 0..t {
                let (rp, _g) = rstd_dev.device_ptr_mut(s);
                let rc = unsafe {
                    nv_kernels::cuda::laguna_rstd256_bf16(
                        s.cu_stream() as *mut c_void,
                        (x_ptr + (r * hidden * 2) as u64) as *const u16,
                        (rp + (r * 4) as u64) as *mut f32,
                        hidden as i32,
                        layer.input_layernorm.eps() as f32,
                    )
                };
                ensure!(rc == 0, "rows rstd rc={rc} layer {li} row {r}");
            }
            let mut q_raw: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * n_q * hd)? };
            let mut k_raw: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * n_kv * hd)? };
            let mut v_dev: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * n_kv * hd)? };
            let mut g_dev: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * n_q)? };
            {
                let wq = bf16_ptr(attn.q_proj.weight().unwrap(), s, n_q * hd * hidden)?;
                let wk = bf16_ptr(attn.k_proj.weight().unwrap(), s, n_kv * hd * hidden)?;
                let wv = bf16_ptr(attn.v_proj.weight().unwrap(), s, n_kv * hd * hidden)?;
                let wg = bf16_ptr(g_lin.unwrap().weight().unwrap(), s, n_q * hidden)?;
                let wn = bf16_ptr(layer.input_layernorm.weight_bf16(), s, hidden)?;
                let (rp, _g0) = rstd_dev.device_ptr(s);
                let (yq, _g1) = q_raw.device_ptr_mut(s);
                let (yk, _g2) = k_raw.device_ptr_mut(s);
                let (yv, _g3) = v_dev.device_ptr_mut(s);
                let (yg, _g4) = g_dev.device_ptr_mut(s);
                let rc = unsafe {
                    nv_kernels::cuda::laguna_gemv_bf16_qkvg_normed_rows(
                        s.cu_stream() as *mut c_void,
                        wq as *const u16,
                        wk as *const u16,
                        wv as *const u16,
                        wg as *const u16,
                        x_ptr as *const u16,
                        wn as *const u16,
                        rp as *const f32,
                        yq as *mut u16,
                        yk as *mut u16,
                        yv as *mut u16,
                        yg as *mut u16,
                        (n_q * hd) as i32,
                        (n_kv * hd) as i32,
                        (n_kv * hd) as i32,
                        n_q as i32,
                        hidden as i32,
                        t as i32,
                    )
                };
                ensure!(rc == 0, "rows qkvg gemv rc={rc} layer {li}");
            }
            if let Some(p) = prof {
                p.record(ProfPoint::Qkv(li), s)?;
            }
            let q = wrap_bf16(q_raw, (1usize, t, n_q * hd), dev).reshape((1usize, t, n_q, hd))?;
            let q = attn.q_norm.forward(&q)?.contiguous()?;
            let k = wrap_bf16(k_raw, (1usize, t, n_kv * hd), dev).reshape((1usize, t, n_kv, hd))?;
            let k = attn.k_norm.forward(&k)?.contiguous()?;
            let v = wrap_bf16(v_dev, (1usize, t, n_kv * hd), dev).reshape((1usize, t, n_kv, hd))?;
            {
                let q_ptr = bf16_ptr(&q, s, t * n_q * hd)?;
                let k_ptr = bf16_ptr(&k, s, t * n_kv * hd)?;
                let (qo, _g1) = q_rot_dev.device_ptr_mut(s);
                let (ko, _g2) = k_rot_dev.device_ptr_mut(s);
                let rc = unsafe {
                    nv_kernels::cuda::laguna_rope_scale_bf16(
                        s.cu_stream() as *mut c_void,
                        q_ptr as *const u16,
                        k_ptr as *const u16,
                        qo as *mut u16,
                        ko as *mut u16,
                        cos_ptr as *const f32,
                        sin_ptr as *const f32,
                        meta_ptr as *const i32,
                        t as i32,
                        n_q as i32,
                        n_kv as i32,
                        hd as i32,
                        rotary_dim as i32,
                        rot_scale,
                    )
                };
                ensure!(rc == 0, "rows rope scale rc={rc} layer {li}");
            }
            if let Some(p) = prof {
                p.record(ProfPoint::Rope(li), s)?;
            }
            v_tensor = Some(v);
            g_pre = Some(g_dev);
        } else {
            let normed = layer.input_layernorm.forward(&x)?;
            let (q_raw, k_raw, v_raw) = match attn.proj_qkv_fused_rows(&normed, t)? {
                Some((q, k, v, g)) => {
                    gate_from_the_fused_qkv_twin = g;
                    (q, k, v)
                }
                None => (
                    attn.proj_q(&normed, t)?,
                    attn.proj_k(&normed, t)?,
                    attn.proj_v(&normed, t)?,
                ),
            };
            let q = q_raw.reshape((1usize, t, n_q, hd))?;
            let q = attn.q_norm.forward(&q)?.contiguous()?;
            let k = k_raw.reshape((1usize, t, n_kv, hd))?;
            let k = attn.k_norm.forward(&k)?.contiguous()?;
            let v = v_raw.reshape((1usize, t, n_kv, hd))?.contiguous()?;
            if let Some(p) = prof {
                p.record(ProfPoint::Qkv(li), s)?;
            }

            {
                let q_ptr = bf16_ptr(&q, s, t * n_q * hd)?;
                let k_ptr = bf16_ptr(&k, s, t * n_kv * hd)?;
                let (qo, _g1) = q_rot_dev.device_ptr_mut(s);
                let (ko, _g2) = k_rot_dev.device_ptr_mut(s);
                let rc = unsafe {
                    nv_kernels::cuda::laguna_rope_scale_bf16(
                        s.cu_stream() as *mut c_void,
                        q_ptr as *const u16,
                        k_ptr as *const u16,
                        qo as *mut u16,
                        ko as *mut u16,
                        cos_ptr as *const f32,
                        sin_ptr as *const f32,
                        meta_ptr as *const i32,
                        t as i32,
                        n_q as i32,
                        n_kv as i32,
                        hd as i32,
                        rotary_dim as i32,
                        rot_scale,
                    )
                };
                ensure!(rc == 0, "rope scale rc={rc} layer {li}");
            }
            if let Some(p) = prof {
                p.record(ProfPoint::Rope(li), s)?;
            }
            v_tensor = Some(v);
            normed_tensor = Some(normed);
        }

        let v_src_ptr = if let Some(vs) = &v_slice {
            let (p, _g) = vs.device_ptr(s);
            p
        } else {
            bf16_ptr(v_tensor.as_ref().unwrap(), s, t * n_kv * hd)?
        };
        let mut o_dev: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * n_q * hd)? };
        let fp8_layer = fp8.and_then(|st| st.layers[li].as_ref());
        if let Some(fpl) = fp8_layer {
            let st = fp8.unwrap();
            ensure!(
                matches!(attn.kind, LayerType::FullAttention),
                "fp8 kv slot on non-full layer {li}"
            );
            {
                let (k_src, _g) = k_rot_dev.device_ptr(s);
                let append = match st.scale_policy {
                    LagunaFp8ScalePolicy::PerTokenHead => nv_kernels::cuda::kv_append_fp8,
                    LagunaFp8ScalePolicy::FixedOne => nv_kernels::cuda::kv_append_fp8_unit_scale,
                    LagunaFp8ScalePolicy::PerTokenHeadInt8 => nv_kernels::cuda::kv_append_int8,
                };
                let rc = unsafe {
                    append(
                        s.cu_stream() as *mut c_void,
                        k_src as *const u16,
                        v_src_ptr as *const u16,
                        fpl.k_fp8 as *mut u8,
                        fpl.v_fp8 as *mut u8,
                        fpl.k_scales as *mut f32,
                        fpl.v_scales as *mut f32,
                        st.n_committed_ptr as *const i32,
                        t as i32,
                        n_kv as i32,
                        hd as i32,
                        0,
                    )
                };
                ensure!(rc == 0, "fp8 kv append rc={rc} layer {li}");
            }
            if let Some(p) = prof {
                p.record(ProfPoint::Append(li), s)?;
            }
            if let Some(xp) = xqa_fp8.filter(|x| x.rows == t && n_q == 48 && n_kv == 8 && hd == 128)
            {
                let (qp, _gq) = q_rot_dev.device_ptr(s);
                let (op, _go) = o_dev.device_ptr_mut(s);
                xp.launch(s, qp, fpl.k_fp8, fpl.v_fp8, op, st.n_committed_ptr)?;
            } else {
                let buffers = verify_split
                    .ok_or_else(|| anyhow::anyhow!("missing FP8 verify split buffers"))?;
                let (qp, _gq) = q_rot_dev.device_ptr(s);
                let (op, _go) = o_dev.device_ptr_mut(s);
                buffers.forward(
                    s,
                    qp,
                    (fpl.k_fp8, fpl.v_fp8, fpl.k_scales, fpl.v_scales),
                    op,
                    st.n_committed_ptr,
                    n_q,
                    n_kv,
                    t,
                    max_seq_len,
                    st.scale_policy
                        .kv_elem_code_matching_the_kernels_0_bf16_1_e4m3_2_int8(),
                )?;
            }
        } else {
            let (window, cap, meta_idx, cu_k_ptr, max_k) = match attn.kind {
                LayerType::SlidingAttention => {
                    (Some(cfg.sliding_window), s_cap, 1usize, cu_slide_ptr, s_cap)
                }
                LayerType::FullAttention => (None, max_seq_len, 0usize, cu_full_ptr, max_seq_len),
            };
            let (k_buf, v_buf) = layer_kv[li]
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing bf16 kv buffers, layer {li}"))?;
            let k_dst_ptr = bf16_ptr(k_buf, s, cap * n_kv * hd)?;
            let v_dst_ptr = bf16_ptr(v_buf, s, cap * n_kv * hd)?;
            {
                let (k_src, _g) = k_rot_dev.device_ptr(s);
                for (src, dst) in [(k_src, k_dst_ptr), (v_src_ptr, v_dst_ptr)] {
                    let rc = unsafe {
                        nv_kernels::cuda::kv_ring_append_bf16(
                            s.cu_stream() as *mut c_void,
                            src as *const u16,
                            dst as *mut u16,
                            (meta_ptr + (meta_idx * 4) as u64) as *const i32,
                            t as i32,
                            cap as i32,
                            n_kv as i32,
                            hd as i32,
                        )
                    };
                    ensure!(rc == 0, "ring append rc={rc} layer {li}");
                }
            }
            if let Some(p) = prof {
                p.record(ProfPoint::Append(li), s)?;
            }

            let m1_gqa = m1_flash
                .filter(|_| t == 1 && hd == 128 && n_q % n_kv == 0 && matches!(n_q / n_kv, 6 | 8));
            if let Some(xp) = xqa.filter(|x| {
                x.rows == t
                    && n_q == 48
                    && n_kv == 8
                    && hd == 128
                    && matches!(attn.kind, LayerType::FullAttention)
            }) {
                let (qp, _gq) = q_rot_dev.device_ptr(s);
                let (op, _go) = o_dev.device_ptr_mut(s);
                xp.launch(s, qp, k_dst_ptr, v_dst_ptr, op, meta_ptr)?;
            } else if let Some(buffers) =
                verify_split.filter(|_| matches!(attn.kind, LayerType::FullAttention))
            {
                let (qp, _gq) = q_rot_dev.device_ptr(s);
                let (op, _go) = o_dev.device_ptr_mut(s);
                buffers.forward(
                    s,
                    qp,
                    (k_dst_ptr, v_dst_ptr, 0, 0),
                    op,
                    meta_ptr,
                    n_q,
                    n_kv,
                    t,
                    max_seq_len,
                    0,
                )?;
            } else if let Some(bufs) = m1_flash.filter(|_| {
                (2..=SLIDING_VERIFY_MAX_ROWS_8).contains(&t)
                    && window.is_some()
                    && hd == 128
                    && n_q == 8 * n_kv
                    && sliding_rows_on_the_m1_flash_kernel_opt_in_1_since_the_2026_09_14_served_identity_ruling_because_its_kernel_vs_kernel_bitwise_gate_did_not_hold_on_the_served_stream_unset_is_fa2_rows()
            }) {
                static SLIDING_M1ROWS_FIRED_ONCE: Once = Once::new();
                SLIDING_M1ROWS_FIRED_ONCE.call_once(|| eprintln!("[laguna_verify_sliding_m1rows] fired: layer {li} t={t} window={} (grid.z rows of the m=1 flash-decode kernel, bitwise per row)", window.unwrap()));
                let (qp, _g1) = q_rot_dev.device_ptr(s);
                let (op, _g2) = o_dev.device_ptr_mut(s);
                let (scp, _g3) = bufs.scratch.device_ptr(s);
                let (fip, _g4) = bufs.fan_in.device_ptr(s);
                let scale = (hd as f32).powf(-0.5);
                let rc = unsafe {
                    nv_kernels::cuda::laguna_flash_decode_gqa_rows(
                        s.cu_stream() as *mut c_void,
                        qp as *const u16,
                        k_dst_ptr as *const u16,
                        v_dst_ptr as *const u16,
                        op as *mut u16,
                        (cu_k_ptr + 4) as *const i32,
                        1 - t as i32,
                        scp as *mut f32,
                        fip as *mut u32,
                        n_q as i32,
                        n_kv as i32,
                        hd as i32,
                        t as i32,
                        window.unwrap() as i32,
                        scale,
                    )
                };
                ensure!(rc == 0, "sliding m1-rows verify attention rc={rc} layer {li} t={t}");
            } else if let Some(sb) = sliding_rows.filter(|_| {
                (2..=SLIDING_VERIFY_MAX_ROWS_8).contains(&t)
                    && hd == 128
                    && window.is_some()
                    && n_q % n_kv == 0
                    && matches!(n_q / n_kv, 6 | 8)
            }) {
                static SLIDING_ROWS_FIRED_ONCE: Once = Once::new();
                SLIDING_ROWS_FIRED_ONCE.call_once(|| eprintln!("[laguna_verify_sliding_rows] fired: layer {li} t={t} window={} rows_per_block={}", window.unwrap(), sb.rows_per_block));
                let (qp, _g1) = q_rot_dev.device_ptr(s);
                let (op, _g2) = o_dev.device_ptr_mut(s);
                let (scp, _g3) = sb.scratch.device_ptr(s);
                let (fip, _g4) = sb.fan_in.device_ptr(s);
                let scale = (hd as f32).powf(-0.5);
                let rc = unsafe {
                    nv_kernels::cuda::laguna_flash_decode_verify_window(
                        s.cu_stream() as *mut c_void,
                        qp as *const u16,
                        k_dst_ptr as *const u16,
                        v_dst_ptr as *const u16,
                        op as *mut u16,
                        (cu_k_ptr + 4) as *const i32,
                        scp as *mut f32,
                        fip as *mut u32,
                        n_q as i32,
                        n_kv as i32,
                        hd as i32,
                        t as i32,
                        cap as i32,
                        window.unwrap() as i32,
                        sb.rows_per_block as i32,
                        scale,
                    )
                };
                ensure!(
                    rc == 0,
                    "sliding rows verify attention rc={rc} layer {li} t={t}"
                );
            } else if let Some(bufs) = m1_gqa {
                let (qp, _g1) = q_rot_dev.device_ptr(s);
                let (op, _g2) = o_dev.device_ptr_mut(s);
                let (scp, _g3) = bufs.scratch.device_ptr(s);
                let (fip, _g4) = bufs.fan_in.device_ptr(s);
                let scale = (hd as f32).powf(-0.5);
                let rc = unsafe {
                    nv_kernels::cuda::laguna_flash_decode_gqa(
                        s.cu_stream() as *mut c_void,
                        qp as *const u16,
                        k_dst_ptr as *const u16,
                        v_dst_ptr as *const u16,
                        op as *mut u16,
                        (cu_k_ptr + 4) as *const i32,
                        0,
                        scp as *mut f32,
                        fip as *mut u32,
                        n_q as i32,
                        n_kv as i32,
                        hd as i32,
                        window.map(|w| w as i32).unwrap_or(0),
                        scale,
                    )
                };
                ensure!(rc == 0, "m1 flash decode rc={rc} layer {li}");
            } else {
                let lse_dev: CudaSlice<f32> = s
                    .alloc_zeros::<f32>(n_q * t)?;
                let (qp, _g1) = q_rot_dev.device_ptr(s);
                let (op, _g2) = o_dev.device_ptr_mut(s);
                let (lp, _g3) = lse_dev.device_ptr(s);
                let scale = (hd as f32).powf(-0.5);
                unsafe {
                    varlen_fwd_bf16(
                        s.cu_stream() as *mut c_void,
                        &VarlenArgs {
                            q_ptr: qp,
                            k_ptr: k_dst_ptr,
                            v_ptr: v_dst_ptr,
                            o_ptr: op,
                            lse_ptr: lp,
                            cu_seqlens_q: cu_q_ptr,
                            cu_seqlens_k: cu_k_ptr,
                            max_seqlen_q: t,
                            max_seqlen_k: max_k,
                            h: n_q,
                            h_k: n_kv,
                            d: hd,
                            softmax_scale: scale,
                            window_size_left: window.map(|w| w.saturating_sub(1)),
                            window_size_right: Some(0),
                        },
                    )?;
                }
            }
        }
        if let Some(p) = prof {
            p.record(ProfPoint::Flash(li), s)?;
        }
        let o_t = wrap_bf16(o_dev, (1usize, t, n_q * hd), dev);

        let g = if g_pre.is_some() {
            None
        } else if let Some(gf) = gate_from_the_fused_qkv_twin.take() {
            Some(gf)
        } else {
            let g_proj = g_lin
                .ok_or_else(|| anyhow::anyhow!("step graph: expected g_proj on layer {li}"))?;
            Some(
                g_proj
                    .forward(normed_tensor.as_ref().unwrap())?
                    .contiguous()?,
            )
        };
        let a_ptr = bf16_ptr(&o_t, s, t * n_q * hd)?;
        let g_ptr = if let Some(gd) = &g_pre {
            let (p, _g) = gd.device_ptr(s);
            p
        } else {
            bf16_ptr(g.as_ref().unwrap(), s, t * n_q)?
        };
        let qkvo_o_ok = t <= VERIFY_ROWS_GEMV_MAX_ROWS_8_MUST_EQUAL_kLagRowsGemvMaxRows8_IN_LAGUNA_STEP_CU
            && t * n_q * hd * 9 / 4 <= Q8_ROWS_MAX_DYNAMIC_SMEM_98304
            && attn.m1_w8_active_o()
            && attn.o_proj.bias().is_none()
            && attn.w8.as_ref().is_some_and(|w| w.is_fp8() && matches!(w.o.bytes(), LagunaProjBytes::E4m3(b) if b.len() == hidden * n_q * hd));
        let attn_out_fused: Option<Tensor> = if qkvo_o_ok && n_q <= 64 && hd.is_multiple_of(8) {
            static O_GATE_FUSED_FIRED_ONCE: Once = Once::new();
            O_GATE_FUSED_FIRED_ONCE.call_once(|| eprintln!("[laguna_qkvo_rows] o_proj gated t={t} through laguna_gemv_q8_rows_gated (softplus gate folded into the x staging, bitwise to softplus_gate_exact_bf16 + laguna_gemv_q8_rows, LAGUNA-Q8-ROWS gate)"));
            let o8 = &attn.w8.as_ref().unwrap().o;
            let (wo, so) = (w8_bytes_ptr(o8, s), w8_scale_ptr(o8, s));
            let mut o_out: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * hidden)? };
            {
                let (yo, _g) = o_out.device_ptr_mut(s);
                let rc = unsafe {
                    nv_kernels::cuda::laguna_gemv_q8_rows_gated(
                        s.cu_stream() as *mut c_void,
                        1,
                        wo as *const c_void,
                        so as *const f32,
                        a_ptr as *const u16,
                        g_ptr as *const u16,
                        n_q as i32,
                        hd as i32,
                        yo as *mut u16,
                        hidden as i32,
                        (n_q * hd) as i32,
                        t as i32,
                    )
                };
                ensure!(
                    rc == 0,
                    "qkvo rows gated o_proj gemv rc={rc} layer {li} t={t}"
                );
            }
            Some(wrap_bf16(o_out, (1usize, t, hidden), dev))
        } else {
            None
        };
        let gated = if attn_out_fused.is_some() {
            None
        } else {
            Some({
                let mut out_dev: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * n_q * hd)? };
                {
                    let (op, _g) = out_dev.device_ptr_mut(s);
                    let rc = unsafe {
                        nv_kernels::cuda::softplus_gate_exact_bf16(
                            s.cu_stream() as *mut c_void,
                            a_ptr as *const u16,
                            g_ptr as *const u16,
                            op as *mut u16,
                            (t * n_q) as i32,
                            hd as i32,
                        )
                    };
                    ensure!(rc == 0, "softplus gate rc={rc} layer {li}");
                }
                wrap_bf16(out_dev, (1usize, t, n_q * hd), dev)
            })
        };
        if let Some(p) = prof {
            p.record(ProfPoint::Gate(li), s)?;
        }
        let attn_out = if let Some(o) = attn_out_fused {
            o
        } else {
            let gated = gated.expect("gate tensor when the fused gated o_proj is off");
            if rows_gemv
                && !attn.m1_w8_active_o()
                && attn.o_proj.weight().is_some()
                && attn.o_proj.bias().is_none()
            {
                let wo = bf16_ptr(attn.o_proj.weight().unwrap(), s, hidden * n_q * hd)?;
                let xg = bf16_ptr(&gated, s, t * n_q * hd)?;
                let mut o_out: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * hidden)? };
                {
                    let (yo, _g) = o_out.device_ptr_mut(s);
                    let rc = unsafe {
                        nv_kernels::cuda::laguna_gemv_bf16_rows(
                            s.cu_stream() as *mut c_void,
                            wo as *const u16,
                            xg as *const u16,
                            yo as *mut u16,
                            hidden as i32,
                            (n_q * hd) as i32,
                            t as i32,
                        )
                    };
                    ensure!(rc == 0, "rows o_proj gemv rc={rc} layer {li}");
                }
                wrap_bf16(o_out, (1usize, t, hidden), dev)
            } else if qkvo_o_ok {
                static QKVO_ROWS_O_FIRED_ONCE: Once = Once::new();
                QKVO_ROWS_O_FIRED_ONCE.call_once(|| eprintln!("[laguna_qkvo_rows] o_proj t={t} through laguna_gemv_q8_rows (bitwise to gemv_e4m3_mk_h with unit norm, LAGUNA-Q8-ROWS gate)"));
                let o8 = &attn.w8.as_ref().unwrap().o;
                let (wo, so) = (w8_bytes_ptr(o8, s), w8_scale_ptr(o8, s));
                let xg = bf16_ptr(&gated, s, t * n_q * hd)?;
                let mut o_out: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * hidden)? };
                {
                    let (yo, _g) = o_out.device_ptr_mut(s);
                    let rc = unsafe {
                        nv_kernels::cuda::laguna_gemv_q8_rows(
                            s.cu_stream() as *mut c_void,
                            1,
                            wo as *const c_void,
                            so as *const f32,
                            xg as *const u16,
                            yo as *mut u16,
                            hidden as i32,
                            (n_q * hd) as i32,
                            t as i32,
                        )
                    };
                    ensure!(rc == 0, "qkvo rows o_proj gemv rc={rc} layer {li} t={t}");
                }
                wrap_bf16(o_out, (1usize, t, hidden), dev)
            } else {
                attn.proj_o(&gated, t)?
            }
        };
        let after_attn = crate::gemma4::residual_add_scale_bf16_op(&x, &attn_out, 1.0, device)?;
        if let Some(slot) = aux_layers
            .iter()
            .position(|&l| l == li + AUX_TAP_AFTER_ATTENTION_OFFSET_10000_ADDED_TO_THE_LAYER_INDEX_TAPS_THE_POST_ATTENTION_RESIDUAL)
        {
            let src_ptr = bf16_ptr(&after_attn, s, t * hidden)?;
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(aux_dst[slot], src_ptr, t * hidden * 2, s.cu_stream())
                    .map_err(|e| anyhow::anyhow!("aux attention stage dtod: {e:?}"))?;
            }
        }
        if let Some(p) = prof {
            p.record(ProfPoint::Attn(li), s)?;
        }

        x = match &layer.ffn {
            LagunaFfn::Dense(mlp) => {
                let ffn = if t >= 2 && verify_dense_rows_run_as_t_single_row_forwards_so_each_row_is_the_m1_step_bitwise_opt_in_until_the_ledger_reads_bitwise() {
                    let mut rows_out = Vec::with_capacity(t);
                    for r in 0..t {
                        let row = after_attn.narrow(1, r, 1)?.contiguous()?;
                        let normed_row = layer.post_attention_layernorm.forward(&row)?;
                        rows_out.push(mlp.forward_fused_cuda(&normed_row)?);
                    }
                    Tensor::cat(&rows_out, 1)?.contiguous()?
                } else {
                    let normed_mlp = layer.post_attention_layernorm.forward(&after_attn)?;
                    mlp.forward_fused_cuda(&normed_mlp)?
                };
                crate::gemma4::residual_add_scale_bf16_op(&after_attn, &ffn, 1.0, device)?
            }
            LagunaFfn::Moe(moe) => {
                let w = grouped[li]
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing grouped weights, layer {li}"))?;

                {
                    let (rs, rl) = after_attn.storage_and_layout();
                    ensure!(rl.is_contiguous(), "resid not contiguous");
                    let r_cuda = match &*rs {
                        Storage::Cuda(st) => st,
                        _ => bail!("resid not CUDA"),
                    };
                    let r_slice = r_cuda.as_cuda_slice::<bf16>()?;
                    let r_view = r_slice.slice(rl.start_offset()..rl.start_offset() + t * hidden);
                    s.memcpy_dtod(&r_view, &mut moe_ctx.resid_in)
                        .map_err(|e| anyhow::anyhow!("resid stage dtod: {e:?}"))?;
                }
                moe_block_body(
                    s,
                    aux_gemm,
                    aux_shared,
                    moe,
                    &layer.post_attention_layernorm,
                    w,
                    moe_ctx,
                    dev,
                    prof.filter(|p| p.is_fine()).map(|p| (p, li)),
                )?;
                let mut out_dev: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(t * hidden)? };
                s.memcpy_dtod(&moe_ctx.out_bf16, &mut out_dev)
                    .map_err(|e| anyhow::anyhow!("moe out dtod: {e:?}"))?;
                wrap_bf16(out_dev, (1usize, t, hidden), dev)
            }
        };

        if let Some(slot) = aux_layers.iter().position(|&l| l == li) {
            let src_ptr = bf16_ptr(&x, s, t * hidden)?;
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    aux_dst[slot],
                    src_ptr,
                    t * hidden * 2,
                    s.cu_stream(),
                )
                .map_err(|e| anyhow::anyhow!("aux stage dtod: {e:?}"))?;
            }
        }
        if let Some(p) = prof {
            p.record(ProfPoint::Ffn(li), s)?;
        }
    }

    let raw = model.lm_head_from_prenorm_scoped(&x, spec_head)?;
    if let Some(p) = prof {
        p.record(ProfPoint::LmHead, s)?;
    }
    let vocab = cfg.vocab_size;
    {
        let raw_ptr = bf16_ptr(&raw, s, t * vocab)?;
        let rc = unsafe {
            nv_kernels::cuda::tanh_softcap_bf16_to_f32(
                s.cu_stream() as *mut c_void,
                raw_ptr as *const u16,
                logits_ptr as *mut f32,
                0.0,
                t * vocab,
            )
        };
        ensure!(rc == 0, "logits cast rc={rc}");
    }

    if let Some((am_part_val, am_part_idx, token_out)) = argmax_slots {
        let raw_ptr = bf16_ptr(&raw, s, t * vocab)? + ((t - 1) * vocab * 2) as u64;
        let (vp, _g1) = am_part_val.device_ptr_mut(s);
        let (ip, _g2) = am_part_idx.device_ptr_mut(s);
        let (tp, _g3) = token_out.device_ptr_mut(s);
        let rc = unsafe {
            nv_kernels::cuda::argmax_bf16(
                s.cu_stream() as *mut c_void,
                raw_ptr as *const u16,
                vocab as i32,
                vp as *mut f32,
                ip as *mut i32,
                std::ptr::null(),
                tp as *mut u32,
                std::ptr::null_mut(),
                0,
            )
        };
        ensure!(rc == 0, "device argmax rc={rc}");
    }
    if let Some(p) = prof {
        p.record(ProfPoint::End, s)?;
    }
    Ok(())
}

pub struct LagunaVerifyGraph<'m> {
    model: &'m Laguna,
    dev: CudaDevice,
    device: Device,
    forked: Arc<CudaStream>,
    aux_gemm: Arc<CudaStream>,
    aux_shared: Arc<CudaStream>,
    runner: CudaGraphRunner,
    t: usize,
    aux_layers: Vec<usize>,
    tok_buf: CudaSlice<u32>,
    host_toks: Vec<u32>,
    cu_q: CudaSlice<i32>,
    cu_full: CudaSlice<i32>,
    cu_slide: CudaSlice<i32>,
    logits_t: Tensor,
    aux_ts: Vec<Tensor>,
    meta_ptr: u64,
    layer_kv: Vec<Option<(Tensor, Tensor)>>,
    grouped: Vec<Option<Arc<MoeGroupedWeights>>>,
    moe_ctx: GroupedDecodeContext,
    s_cap: usize,
    max_seq_len: usize,
    captured: bool,
    prof: Option<StepProfStamps>,
    fp8: Option<Fp8VerifyState>,
    verify_split: Option<VerifySplitBufs>,
    m1_flash: Option<M1FlashBufs>,
    xqa: Option<XqaPlan>,
    xqa_fp8: Option<XqaFp8Plan>,
    sliding_rows: Option<SlidingVerifyBufs>,
    _err_drain: CtxErrDrain,
}

unsafe impl Send for LagunaVerifyGraph<'_> {}

enum VerifyCacheRef<'a> {
    Bf16(&'a LagunaKvCache),
    Fp8(&'a LagunaKvCacheFp8),
}

impl<'m> LagunaVerifyGraph<'m> {
    pub fn new(
        model: &'m Laguna,
        cache: &LagunaKvCache,
        t: usize,
        aux_layers: &[usize],
    ) -> Result<Self> {
        Self::new_inner(model, VerifyCacheRef::Bf16(cache), t, aux_layers)
    }

    pub fn new_fp8(
        model: &'m Laguna,
        cache: &LagunaKvCacheFp8,
        t: usize,
        aux_layers: &[usize],
    ) -> Result<Self> {
        Self::new_inner(model, VerifyCacheRef::Fp8(cache), t, aux_layers)
    }

    fn new_inner(
        model: &'m Laguna,
        cache: VerifyCacheRef<'_>,
        t: usize,
        aux_layers: &[usize],
    ) -> Result<Self> {
        let device = model.device().clone();
        let Device::Cuda(dev) = device.clone() else {
            bail!("LagunaVerifyGraph requires a CUDA device")
        };
        ensure!(
            model.dtype() == DType::BF16,
            "LagunaVerifyGraph requires BF16"
        );
        ensure!(t >= 1, "LagunaVerifyGraph: t must be >= 1");
        let (meta_ptr, s_cap, max_seq_len) = match &cache {
            VerifyCacheRef::Bf16(c) => {
                ensure!(c.has_ring(), "LagunaVerifyGraph requires the ring KV cache");
                let meta_ptr = c
                    .ring_meta_ptr()
                    .ok_or_else(|| anyhow::anyhow!("ring meta missing"))?;
                (meta_ptr, c.sliding_cap().unwrap_or(0), c.max_seq_len())
            }
            VerifyCacheRef::Fp8(c) => (c.ring_meta_ptr(), c.sliding_cap(), c.max_seq_len()),
        };
        ensure!(s_cap > 0, "LagunaVerifyGraph: no sliding capacity");

        let raw_ctx: Arc<CudaContext> = dev.cuda_stream().context().clone();
        let mut ctor_guard = crate::gemma4_batch_graph::graph_teardown::CtorForkGuard::new();
        let forked = ctor_guard
            .fork(&raw_ctx)
            .map_err(|e| anyhow::anyhow!("verify graph stream: {e:?}"))?;
        let aux_gemm = ctor_guard
            .fork(&raw_ctx)
            .map_err(|e| anyhow::anyhow!("verify aux gemm stream: {e:?}"))?;
        let aux_shared = ctor_guard
            .fork(&raw_ctx)
            .map_err(|e| anyhow::anyhow!("verify aux shared stream: {e:?}"))?;

        let cfg = model.config();
        let tok_buf = forked.alloc_zeros::<u32>(t)?;
        let cu_q = forked.clone_htod(&[0i32, t as i32])?;
        let cu_full = forked.alloc_zeros::<i32>(2)?;
        let cu_slide = forked.alloc_zeros::<i32>(2)?;

        let logits_t = Tensor::zeros((1usize, t, cfg.vocab_size), DType::F32, &device)?;
        let mut aux_ts = Vec::with_capacity(aux_layers.len());
        for _ in aux_layers {
            aux_ts.push(Tensor::zeros(
                (1usize, t, cfg.hidden_size),
                DType::BF16,
                &device,
            )?);
        }

        let mut layer_kv = Vec::with_capacity(model.layers().len());
        let mut grouped = Vec::with_capacity(model.layers().len());
        let mut fp8_layers: Vec<Option<Fp8VerifyLayer>> = Vec::with_capacity(model.layers().len());
        for (li, layer) in model.layers().iter().enumerate() {
            match &cache {
                VerifyCacheRef::Bf16(c) => {
                    layer_kv.push(Some(c.layer_kv_bufs(li)));
                    fp8_layers.push(None);
                }
                VerifyCacheRef::Fp8(c) => {
                    layer_kv.push(c.layer_bf16_bufs(li));
                    fp8_layers.push(c.layer_fp8_ptrs(li).map(
                        |(k_fp8, v_fp8, k_scales, v_scales)| Fp8VerifyLayer {
                            k_fp8,
                            v_fp8,
                            k_scales,
                            v_scales,
                        },
                    ));
                    ensure!(
                        layer_kv[li].is_some() != fp8_layers[li].is_some(),
                        "fp8 cache layer {li}: expected exactly one of bf16/fp8 storage"
                    );
                }
            }
            grouped.push(match &layer.ffn {
                LagunaFfn::Moe(moe) => Some(
                    model
                        .grouped_weights(moe)?
                        .ok_or_else(|| anyhow::anyhow!("grouped MoE weights unavailable"))?,
                ),
                LagunaFfn::Dense(_) => None,
            });
        }
        eprintln!("[laguna_verify_projections] rows_gemv={} rows={t} (opt-in: q/k/v/gate through M-row GEMVs with the M=1 kernels' normalization and accumulation order; the fired marker names the path taken)", if (2..=VERIFY_ROWS_GEMV_MAX_ROWS_8_MUST_EQUAL_kLagRowsGemvMaxRows8_IN_LAGUNA_STEP_CU).contains(&t) && verify_rows_gemv_opt_in() { "on" } else { "off" });
        let fp8_count = fp8_layers.iter().filter(|layer| layer.is_some()).count();
        let verify_split = Some(VerifySplitBufs::new(
            model,
            t,
            max_seq_len,
            &forked,
            fp8_count,
            M1_FUSED_TC_SPLITS_32_MUST_EQUAL_THE_VERIFY_GRAPHS_SO_ROW_J_REDUCES_LIKE_THE_M1_STEP,
        )?);
        let xqa = match &cache {
            VerifyCacheRef::Bf16(_) if matches!(t, 1 | 2 | 3 | 5) => XqaPlan::new_for(
                t,
                max_seq_len,
                cfg.num_key_value_heads,
                model
                    .layers()
                    .iter()
                    .filter(|l| matches!(l.self_attn.kind, LayerType::FullAttention))
                    .map(|l| l.self_attn.num_heads)
                    .max()
                    .unwrap_or(0),
                cfg.head_dim,
                &format!("verify-t{t}"),
            )?,
            _ => None,
        };
        let xqa_fp8 = match &cache {
            VerifyCacheRef::Fp8(c) if matches!(t, 1 | 2 | 3 | 5) => XqaFp8Plan::new_for(
                t,
                max_seq_len,
                cfg.num_key_value_heads,
                model
                    .layers()
                    .iter()
                    .filter(|l| matches!(l.self_attn.kind, LayerType::FullAttention))
                    .map(|l| l.self_attn.num_heads)
                    .max()
                    .unwrap_or(0),
                cfg.head_dim,
                c.scale_policy(),
                &format!("verify-t{t}"),
            )?,
            _ => None,
        };
        let sliding_rows = SlidingVerifyBufs::new_opt_in(model, t, &forked)?;
        let m1_flash = if (1..=SLIDING_VERIFY_MAX_ROWS_8).contains(&t)
            && cfg.head_dim == 128
            && cfg
                .num_attention_heads
                .is_multiple_of(cfg.num_key_value_heads)
            && matches!(cfg.num_attention_heads / cfg.num_key_value_heads, 6 | 8)
        {
            let elems = nv_kernels::cuda::laguna_flash_decode_gqa_scratch_elems(
                cfg.num_key_value_heads as i32,
            );
            let rows_sharing_the_m1_kernel = t.clamp(1, SLIDING_VERIFY_MAX_ROWS_8);
            Some(M1FlashBufs {
                scratch: forked
                    .alloc_zeros::<f32>(elems * rows_sharing_the_m1_kernel)
                    .map_err(|e| anyhow::anyhow!("verify m1 flash scratch: {e:?}"))?,
                fan_in: forked
                    .alloc_zeros::<u32>(cfg.num_key_value_heads * rows_sharing_the_m1_kernel)
                    .map_err(|e| anyhow::anyhow!("verify m1 flash fan_in: {e:?}"))?,
            })
        } else {
            None
        };
        let fp8 = match &cache {
            VerifyCacheRef::Bf16(_) => None,
            VerifyCacheRef::Fp8(c) => Some(Fp8VerifyState {
                scale_policy: c.scale_policy(),
                layers: fp8_layers,
                n_committed_ptr: meta_ptr,
            }),
        };
        let verify_folds_shared_like_the_m1_graph = grouped
            .iter()
            .flatten()
            .any(|w: &Arc<MoeGroupedWeights>| w.folded_shared);
        let moe_ctx = if verify_folds_shared_like_the_m1_graph {
            GroupedDecodeContext::new_folded_shared(
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                cfg.num_experts_per_tok,
                cfg.num_experts,
                t,
                &forked,
            )?
        } else {
            GroupedDecodeContext::new_multi(
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                cfg.num_experts_per_tok,
                cfg.num_experts,
                t,
                &forked,
            )?
        };
        forked.synchronize()?;

        let prof = if verify_prof_enabled() {
            let layer_info: Vec<(bool, bool)> = model
                .layers()
                .iter()
                .map(|l| {
                    (
                        matches!(l.self_attn.kind, LayerType::SlidingAttention),
                        matches!(l.ffn, LagunaFfn::Dense(_)),
                    )
                })
                .collect();
            Some(StepProfStamps::new(&forked, layer_info)?)
        } else {
            None
        };
        let runner = CudaGraphRunner::new(forked.clone());
        ctor_guard.the_built_engine_owns_teardown_now();
        Ok(Self {
            model,
            dev,
            device,
            forked,
            aux_gemm,
            aux_shared,
            runner,
            t,
            aux_layers: aux_layers.to_vec(),
            tok_buf,
            host_toks: vec![0u32; t],
            cu_q,
            cu_full,
            cu_slide,
            logits_t,
            aux_ts,
            meta_ptr,
            layer_kv,
            grouped,
            moe_ctx,
            s_cap,
            max_seq_len,
            captured: false,
            prof,
            fp8,
            m1_flash,
            verify_split,
            xqa,
            xqa_fp8,
            sliding_rows,
            _err_drain: CtxErrDrain(raw_ctx),
        })
    }

    pub(crate) fn accept_status_device(&self) -> Option<u64> {
        self.xqa
            .as_ref()
            .map(|x| x.info.status_device)
            .or_else(|| self.xqa_fp8.as_ref().map(|x| x.info.status_device))
    }

    pub(crate) fn drafts_device_ptr(&self, stream: &Arc<CudaStream>) -> (u64, usize) {
        let (p, _g) = self.tok_buf.device_ptr(stream);
        (
            p + std::mem::size_of::<u32>() as u64,
            self.t.saturating_sub(1),
        )
    }

    pub fn verify(
        &mut self,
        cache: &mut LagunaKvCache,
        tokens: &[u32],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        ensure!(
            self.fp8.is_none(),
            "verify: this graph was captured for the fp8 KV cache"
        );
        self.model.apply_attn_w8_for(self.t);
        ensure!(tokens.len() == self.t, "verify: expected {} tokens", self.t);
        ensure!(
            cache.ring_meta_ptr() == Some(self.meta_ptr),
            "verify: cache does not match the captured buffers"
        );
        let write_pos = cache.current_len();
        ensure!(
            write_pos + self.t <= self.max_seq_len,
            "verify: KV capacity exceeded"
        );
        cache
            .prepare_for_decode(write_pos, write_pos + self.t)
            .context("verify graph prepare_for_decode")?;
        let out = self.run_captured(tokens)?;
        if let Some(x) = self.xqa.as_ref() {
            nv_layers::cuda_stream::sync_legacy_then_forked(&self.dev, &self.forked)?;
            x.status_after_sync(&self.forked)?;
        }
        if let Some(x) = self.xqa_fp8.as_ref() {
            nv_layers::cuda_stream::sync_legacy_then_forked(&self.dev, &self.forked)?;
            x.status_after_sync(&self.forked)?;
        }
        cache.advance(self.t);
        cache.note_graph_write();
        Ok(out)
    }

    pub fn verify_fp8(
        &mut self,
        cache: &mut LagunaKvCacheFp8,
        tokens: &[u32],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        ensure!(
            self.fp8.is_some(),
            "verify_fp8: this graph was captured for the bf16 ring cache"
        );
        self.model.apply_attn_w8_for(self.t);
        ensure!(tokens.len() == self.t, "verify: expected {} tokens", self.t);
        ensure!(
            cache.ring_meta_ptr() == self.meta_ptr,
            "verify_fp8: cache does not match the captured buffers"
        );
        let write_pos = LagunaKvCacheFp8::current_len(cache);
        ensure!(
            write_pos + self.t <= self.max_seq_len,
            "verify: KV capacity exceeded"
        );
        cache
            .prepare_for_decode_dev(write_pos, write_pos + self.t)
            .context("fp8 verify graph prepare_for_decode")?;
        let out = self.run_captured(tokens)?;
        LagunaKvCacheFp8::advance(cache, self.t);
        cache.note_graph_write();
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn verify_device(
        &mut self,
        cache: &mut LagunaKvCache,
        anchor: u32,
        drafts: &CudaSlice<u32>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        ensure!(
            self.fp8.is_none(),
            "verify_device: this graph was captured for the fp8 KV cache"
        );
        self.model.apply_attn_w8_for(self.t);
        ensure!(
            cache.ring_meta_ptr() == Some(self.meta_ptr),
            "verify_device: cache does not match the captured buffers"
        );
        let write_pos = cache.current_len();
        ensure!(
            write_pos + self.t <= self.max_seq_len,
            "verify_device: KV capacity exceeded"
        );
        cache
            .prepare_for_decode(write_pos, write_pos + self.t)
            .context("verify_device prepare_for_decode")?;
        self.fill_tok_buf_device(anchor, drafts)?;
        let out = self.run_captured_inner()?;
        cache.advance(self.t);
        cache.note_graph_write();
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn verify_fp8_device(
        &mut self,
        cache: &mut LagunaKvCacheFp8,
        anchor: u32,
        drafts: &CudaSlice<u32>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        ensure!(
            self.fp8.is_some(),
            "verify_fp8_device: this graph was captured for the bf16 ring cache"
        );
        self.model.apply_attn_w8_for(self.t);
        ensure!(
            cache.ring_meta_ptr() == self.meta_ptr,
            "verify_fp8_device: cache does not match the captured buffers"
        );
        let write_pos = LagunaKvCacheFp8::current_len(cache);
        ensure!(
            write_pos + self.t <= self.max_seq_len,
            "verify_fp8_device: KV capacity exceeded"
        );
        cache
            .prepare_for_decode_dev(write_pos, write_pos + self.t)
            .context("verify_fp8_device prepare_for_decode")?;
        self.fill_tok_buf_device(anchor, drafts)?;
        let out = self.run_captured_inner()?;
        LagunaKvCacheFp8::advance(cache, self.t);
        cache.note_graph_write();
        Ok(out)
    }

    fn run_captured(&mut self, tokens: &[u32]) -> Result<(Tensor, Vec<Tensor>)> {
        self.host_toks.copy_from_slice(tokens);
        let legacy = self.dev.cuda_stream();
        legacy
            .memcpy_htod(&self.host_toks[..], &mut self.tok_buf)
            .map_err(|e| anyhow::anyhow!("htod verify tokens: {e:?}"))?;
        self.run_captured_inner()
    }

    #[cfg(feature = "cuda")]
    fn fill_tok_buf_device(&mut self, anchor: u32, drafts: &CudaSlice<u32>) -> Result<()> {
        ensure!(
            drafts.len() >= self.t.saturating_sub(1),
            "fill_tok_buf_device: drafts len {} < t-1 {}",
            drafts.len(),
            self.t.saturating_sub(1)
        );
        let legacy = self.dev.cuda_stream();
        {
            let mut head = self.tok_buf.slice_mut(0..1);
            let anchor_host = [anchor];
            legacy
                .memcpy_htod(&anchor_host[..], &mut head)
                .map_err(|e| anyhow::anyhow!("htod verify anchor: {e:?}"))?;
        }
        let src = drafts.slice(0..self.t - 1);
        let mut dst = self.tok_buf.slice_mut(1..self.t);
        legacy
            .memcpy_dtod(&src, &mut dst)
            .map_err(|e| anyhow::anyhow!("dtod verify drafts: {e:?}"))?;
        Ok(())
    }

    fn run_captured_inner(&mut self) -> Result<(Tensor, Vec<Tensor>)> {
        let legacy = self.dev.cuda_stream();
        let raw_ctx = legacy.context().clone();
        if raw_ctx.is_event_tracking() {
            unsafe { raw_ctx.disable_event_tracking() };
            legacy
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-capture legacy sync: {e:?}"))?;
        }

        let was_captured = self.captured;
        let forked = self.forked.clone();
        let logits_ptr = {
            let (ls, ll) = self.logits_t.storage_and_layout();
            let l_cuda = match &*ls {
                Storage::Cuda(st) => st,
                _ => bail!("verify logits not CUDA"),
            };
            let l_slice = l_cuda.as_cuda_slice::<f32>()?;
            let (p, _g) = l_slice.device_ptr(&forked);
            p + (ll.start_offset() * 4) as u64
        };
        let mut aux_dst = Vec::with_capacity(self.aux_ts.len());
        for a in &self.aux_ts {
            aux_dst.push(bf16_ptr(
                a,
                &forked,
                self.t * self.model.config().hidden_size,
            )?);
        }

        let LagunaVerifyGraph {
            model,
            dev,
            device,
            aux_gemm,
            aux_shared,
            runner,
            tok_buf,
            cu_q,
            cu_full,
            cu_slide,
            meta_ptr,
            layer_kv,
            grouped,
            moe_ctx,
            s_cap,
            max_seq_len,
            t,
            aux_layers,
            prof,
            fp8,
            m1_flash,
            verify_split,
            xqa,
            xqa_fp8,
            sliding_rows,
            ..
        } = self;
        let meta_ptr = *meta_ptr;
        let s_cap = *s_cap;
        let max_seq_len = *max_seq_len;
        let t = *t;
        let prof = prof.as_ref();
        let fp8 = fp8.as_ref();
        let xqa = xqa.as_ref();
        let xqa_fp8 = xqa_fp8.as_ref();
        let sliding_rows = sliding_rows.as_ref();
        let verify_split = verify_split.as_ref();

        let mut body = |s: &Arc<CudaStream>, moe_ctx: &mut GroupedDecodeContext| -> Result<()> {
            step_body(
                model,
                dev,
                device,
                s,
                aux_gemm,
                aux_shared,
                tok_buf,
                cu_q,
                cu_full,
                cu_slide,
                logits_ptr,
                None,
                meta_ptr,
                layer_kv,
                grouped,
                moe_ctx,
                s_cap,
                max_seq_len,
                t,
                aux_layers,
                &aux_dst,
                prof,
                m1_flash.as_ref(),
                verify_split,
                fp8,
                xqa,
                xqa_fp8,
                sliding_rows,
                true,
            )
        };

        if !was_captured {
            legacy
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-warm legacy sync: {e:?}"))?;
            nv_layers::cuda_stream::with_stream(forked.clone(), || body(&forked, moe_ctx))
                .context("verify graph warm pass")?;
            forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("warm sync: {e:?}"))?;
        }

        runner
            .run_on(t as u64, Some(&legacy), |s| {
                nv_layers::cuda_stream::with_stream(s.clone(), || body(s, moe_ctx))
            })
            .context("verify graph capture/replay")?;
        if !was_captured {
            forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("post-capture sync: {e:?}"))?;
            self.captured = true;
            if let Some(buffers) = self.verify_split.as_ref() {
                let fp8_layers = self
                    .fp8
                    .as_ref()
                    .map(|state| state.layers.iter().filter(|layer| layer.is_some()).count())
                    .unwrap_or(0);
                eprintln!("[laguna_verify_attention_capture] route=split-gqa captured=1 splits={} rows={} max_seq_allocated={} fp8_layers={fp8_layers}", buffers.splits, self.t, self.max_seq_len);
            }
        }
        if let Some(p) = self.prof.as_ref() {
            self.dev
                .cuda_stream()
                .synchronize()
                .map_err(|e| anyhow::anyhow!("prof legacy sync: {e:?}"))?;
            p.report(&format!("verify_t{}", self.t))?;
        }

        Ok((self.logits_t.clone(), self.aux_ts.clone()))
    }
}

impl Drop for LagunaVerifyGraph<'_> {
    fn drop(&mut self) {
        crate::gemma4_batch_graph::graph_teardown::GraphTeardown::new(&self.forked)
            .with_stream(&self.aux_gemm)
            .with_stream(&self.aux_shared)
            .run(|| self.runner.invalidate());
    }
}
