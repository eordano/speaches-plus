#[cfg(feature = "cuda")]
use anyhow::bail;
use anyhow::{ensure, Context, Result};
#[cfg(feature = "cuda")]
use candle_core::Storage;
use candle_core::{DType, Device, Tensor, D};
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_weights::WeightLoader;

use crate::gemma4::{Gemma4Config, LayerType};
use crate::CausalLm;
#[cfg(feature = "cuda")]
use candle_core::op::BackpropOp;
#[cfg(feature = "cuda")]
use candle_core::CudaDevice;
#[cfg(feature = "cuda")]
use candle_core::CudaStorage;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
#[cfg(feature = "cuda")]
use std::collections::BTreeMap;

const PREFIX: &str = "model.language_model";

pub const NV_E4B_BOOT_WARM_PREFILL_ENV_OPT_IN_1_RUNS_ONE_THROWAWAY_CHUNK_AT_MODEL_BUILD_BECAUSE_THE_FIRST_REAL_CHUNK_OF_A_FRESH_PROCESS_PAYS_ONE_TIME_MODULE_AND_WORKSPACE_INIT_LANDING_IN_TTFT:
    &str = "NV_E4B_BOOT_WARM_PREFILL";

const E4B_BOOT_WARM_CHUNK_512_MATCHES_THE_SERVING_PREFILL_CHUNK_SO_THE_THROWAWAY_PASS_COMPILES_EXACTLY_THE_MODULES_THE_FIRST_REAL_CHUNK_NEEDS:
    usize = 512;

fn boot_warm_prefill_env_opted_in() -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call(
        NV_E4B_BOOT_WARM_PREFILL_ENV_OPT_IN_1_RUNS_ONE_THROWAWAY_CHUNK_AT_MODEL_BUILD_BECAUSE_THE_FIRST_REAL_CHUNK_OF_A_FRESH_PROCESS_PAYS_ONE_TIME_MODULE_AND_WORKSPACE_INIT_LANDING_IN_TTFT,
    )
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrefillLogitsRows {
    LastRow,
    Discarded,
}

pub const NV_E4B_MLP_W4A8FP8_ENV_OPT_IN_1_RUNS_MLP_MARLIN_GEMMS_WITH_PER_ROW_FP8_ACTIVATIONS_AND_X512_COMPENSATED_SCALES_BECAUSE_CHUNK_512_PREFILL_IS_THE_LARGE_M_REGIME_WHERE_FP8_TENSOR_CORES_PAY_WHILE_DECODE_M1_KEEPS_THE_GEMV_PATH_UNTOUCHED_AND_ATTN_PROJECTIONS_STAY_A16_PER_THE_QWEN_W4A8_MLP_ONLY_PRECEDENT:
    &str = "NV_E4B_MLP_W4A8FP8";

pub fn mlp_w4a8fp8_env_opted_in() -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call(
        NV_E4B_MLP_W4A8FP8_ENV_OPT_IN_1_RUNS_MLP_MARLIN_GEMMS_WITH_PER_ROW_FP8_ACTIVATIONS_AND_X512_COMPENSATED_SCALES_BECAUSE_CHUNK_512_PREFILL_IS_THE_LARGE_M_REGIME_WHERE_FP8_TENSOR_CORES_PAY_WHILE_DECODE_M1_KEEPS_THE_GEMV_PATH_UNTOUCHED_AND_ATTN_PROJECTIONS_STAY_A16_PER_THE_QWEN_W4A8_MLP_ONLY_PRECEDENT,
    )
}

pub const E4B_FULL_GEMM_ATTN_FROM_THE_SECOND_KEY_TILE_4096_BECAUSE_INSIDE_ONE_TILE_FA2_WINS_THE_2K_CELL_BY_15_MS: usize =
    nv_layers::attn::E4B_FULL_ATTN_KEY_TILE_4096_KEEPS_THE_F32_SCORE_TILE_INSIDE_L2_AND_MATCHES_THE_SOFTMAX_KERNELS_REGISTER_BUDGET;

fn name(suffix: &str) -> String {
    format!("{PREFIX}.{suffix}")
}

fn load_tensor(weights: &WeightLoader, suffix: &str, dtype: DType) -> Result<Tensor> {
    let n = name(suffix);
    weights.get(&n, dtype).with_context(|| format!("load {n}"))
}

fn load_linear(
    weights: &WeightLoader,
    suffix: &str,
    dtype: DType,
    device: &Device,
) -> Result<QLinear> {
    let w = format!("{suffix}.weight");
    if weights.has(&name(&w)) {
        return Ok(QLinear::Dense(Linear::new(
            load_tensor(weights, &w, dtype)?,
            None,
        )?));
    }
    let packed = format!("{suffix}.weight_packed");
    ensure!(
        weights.has(&name(&packed)),
        "no .weight or .weight_packed for {suffix}"
    );
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        let (pbytes, scale, out, inp) = read_packed_parts(weights, suffix)?;
        return pick_quant_backend(&pbytes, scale, out, inp, device, suffix);
    }
    Ok(QLinear::Dense(Linear::new(
        dequant_pack_quantized(weights, suffix, dtype, device)?,
        None,
    )?))
}

fn load_linear_merged(
    weights: &WeightLoader,
    suffixes: &[&str],
    dtype: DType,
    device: &Device,
) -> Result<QLinear> {
    ensure!(suffixes.len() >= 2, "merge needs >= 2 linears");
    let wa = format!("{}.weight", suffixes[0]);
    if weights.has(&name(&wa)) {
        let parts = suffixes
            .iter()
            .map(|sfx| load_tensor(weights, &format!("{sfx}.weight"), dtype))
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<&Tensor> = parts.iter().collect();
        return Ok(QLinear::Dense(Linear::new(Tensor::cat(&refs, 0)?, None)?));
    }
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        let mut p: Vec<u8> = Vec::new();
        let mut scales: Vec<Tensor> = Vec::new();
        let mut out = 0usize;
        let mut inp = 0usize;
        for sfx in suffixes {
            let (pb, sb, ob, ib) = read_packed_parts(weights, sfx)?;
            if inp == 0 {
                inp = ib;
            }
            ensure!(ib == inp, "merged linears must share in_features");
            p.extend_from_slice(&pb);
            scales.push(sb);
            out += ob;
        }
        let refs: Vec<&Tensor> = scales.iter().collect();
        let scale = Tensor::cat(&refs, 0)?;
        return pick_quant_backend(&p, scale, out, inp, device, suffixes[0]);
    }
    let parts = suffixes
        .iter()
        .map(|sfx| dequant_pack_quantized(weights, sfx, dtype, device))
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&Tensor> = parts.iter().collect();
    Ok(QLinear::Dense(Linear::new(Tensor::cat(&refs, 0)?, None)?))
}

#[cfg(feature = "cuda")]
fn read_packed_parts(
    weights: &WeightLoader,
    suffix: &str,
) -> Result<(Vec<u8>, Tensor, usize, usize)> {
    let shape =
        load_tensor(weights, &format!("{suffix}.weight_shape"), DType::I64)?.to_vec1::<i64>()?;
    let out = shape[0] as usize;
    let inp = shape[1] as usize;
    let scale = load_tensor(weights, &format!("{suffix}.weight_scale"), DType::BF16)?;
    let pbytes = weights.raw_bytes(&name(&format!("{suffix}.weight_packed")))?;
    ensure!(
        pbytes.len() == out * (inp / 8) * 4,
        "{suffix} packed size mismatch"
    );
    Ok((pbytes.to_vec(), scale, out, inp))
}

#[cfg(feature = "cuda")]
fn pick_quant_backend(
    pbytes: &[u8],
    scale: Tensor,
    out: usize,
    inp: usize,
    device: &Device,
    suffix: &str,
) -> Result<QLinear> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CHOICE: OnceLock<Mutex<HashMap<(usize, usize, usize), bool>>> = OnceLock::new();

    let gs = inp / scale.dim(1)?;
    let marlin_supported =
        inp.is_multiple_of(16) && out.is_multiple_of(64) && inp.is_multiple_of(32);
    let a8_mlp = mlp_w4a8fp8_env_opted_in() && suffix.contains(".mlp.") && marlin_supported;
    if nv_config::env_arm_present_at_all_reread_every_call("NV_E4B_FORCE_GEMV") || !marlin_supported
    {
        return Ok(QLinear::Quant(W4a16Linear::from_raw(
            pbytes, scale, out, inp, device,
        )?));
    }
    if gs == 32 {
        let variant = if inp == 2560 && out == 20480 {
            Some(0)
        } else if inp == 10240 && out == 2560 {
            Some(7)
        } else {
            None
        };
        if let Some(variant) = variant {
            return Ok(QLinear::GemvM1(GemvM1Hybrid {
                gemv: W4a16Linear::from_raw(pbytes, scale.clone(), out, inp, device)?,
                marlin: MarlinLinear::from_raw_backend(pbytes, scale, out, inp, device, a8_mlp)?,
                variant,
                y_persist: std::sync::OnceLock::new(),
                geglu_persist: std::sync::OnceLock::new(),
                post_norm_last_block_counter: std::sync::OnceLock::new(),
            }));
        }
    }

    if nv_config::env_arm_present_at_all_reread_every_call("NV_E4B_GEMV_SMALL")
        && inp <= 3072
        && out <= 4096
    {
        return Ok(QLinear::Quant(W4a16Linear::from_raw(
            pbytes, scale, out, inp, device,
        )?));
    }

    if nv_config::env_arm_present_at_all_reread_every_call("NV_E4B_TINY_GEMV")
        && (inp <= 512 || out <= 512)
    {
        return Ok(QLinear::Quant(W4a16Linear::from_raw(
            pbytes, scale, out, inp, device,
        )?));
    }

    if !nv_config::env_arm_present_at_all_reread_every_call("NV_E4B_AUTOTUNE") {
        return Ok(QLinear::Marlin(MarlinLinear::from_raw_backend(
            pbytes, scale, out, inp, device, a8_mlp,
        )?));
    }

    let map = CHOICE.get_or_init(|| Mutex::new(HashMap::new()));
    let cached = map.lock().unwrap().get(&(out, inp, gs)).copied();
    match cached {
        Some(true) => Ok(QLinear::Marlin(MarlinLinear::from_raw(
            pbytes, scale, out, inp, device,
        )?)),
        Some(false) => Ok(QLinear::Quant(W4a16Linear::from_raw(
            pbytes, scale, out, inp, device,
        )?)),
        None => {
            let marlin = MarlinLinear::from_raw(pbytes, scale.clone(), out, inp, device)?;
            let gemv = W4a16Linear::from_raw(pbytes, scale, out, inp, device)?;
            let x = Tensor::zeros((1, inp), DType::BF16, device)?;
            let tm = bench_forward(|| marlin.forward(&x), device)?;
            let tg = bench_forward(|| gemv.forward(&x), device)?;
            let use_marlin = tm <= tg;
            map.lock().unwrap().insert((out, inp, gs), use_marlin);
            Ok(if use_marlin {
                QLinear::Marlin(marlin)
            } else {
                QLinear::Quant(gemv)
            })
        }
    }
}

#[cfg(feature = "cuda")]
fn bench_forward<F: FnMut() -> Result<Tensor>>(mut f: F, device: &Device) -> Result<f64> {
    let Device::Cuda(dev) = device.clone() else {
        bail!("bench_forward requires cuda")
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    for _ in 0..3 {
        let _ = f()?;
    }
    stream.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..20 {
        let _ = f()?;
    }
    stream.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() / 20.0)
}

fn dequant_pack_quantized(
    weights: &WeightLoader,
    suffix: &str,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let scale =
        load_tensor(weights, &format!("{suffix}.weight_scale"), DType::F32)?.to_vec2::<f32>()?;
    let shape =
        load_tensor(weights, &format!("{suffix}.weight_shape"), DType::I64)?.to_vec1::<i64>()?;
    let out = shape[0] as usize;
    let inp = shape[1] as usize;
    let cols = inp / 8;
    let gs = inp / scale[0].len();

    let pbytes = weights.raw_bytes(&name(&format!("{suffix}.weight_packed")))?;
    ensure!(
        pbytes.len() == out * cols * 4,
        "{suffix}.weight_packed: {} bytes != {}",
        pbytes.len(),
        out * cols * 4
    );
    let i32_at = |idx: usize| -> i32 {
        let b = 4 * idx;
        i32::from_le_bytes([pbytes[b], pbytes[b + 1], pbytes[b + 2], pbytes[b + 3]])
    };
    let mut w = vec![0f32; out * inp];
    for o in 0..out {
        let srow = &scale[o];
        let wrow = &mut w[o * inp..(o + 1) * inp];
        let base = o * cols;
        for j in 0..inp {
            let q = ((i32_at(base + j / 8) >> (4 * (j % 8))) & 0xF) - 8;
            wrow[j] = q as f32 * srow[j / gs];
        }
    }
    Tensor::from_vec(w, (out, inp), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

enum QLinear {
    Dense(Linear),
    #[cfg(feature = "cuda")]
    Quant(W4a16Linear),
    #[cfg(feature = "cuda")]
    Marlin(MarlinLinear),
    #[cfg(feature = "cuda")]
    GemvM1(GemvM1Hybrid),
}

#[cfg(feature = "cuda")]
struct GemvM1Hybrid {
    gemv: W4a16Linear,
    marlin: MarlinLinear,
    variant: i32,
    y_persist: std::sync::OnceLock<(Tensor, u64)>,
    geglu_persist: std::sync::OnceLock<(Tensor, u64)>,
    post_norm_last_block_counter: std::sync::OnceLock<CudaSlice<u32>>,
}

#[cfg(feature = "cuda")]
pub fn e4b_m1_gemv_folds_opt_in_until_the_served_stream_identity_proves_them() -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_E4B_M1_FOLD")
}

#[cfg(feature = "cuda")]
fn persist_bf16_row(
    lock: &std::sync::OnceLock<(Tensor, u64)>,
    elems: usize,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    dev: &CudaDevice,
) -> Result<(Tensor, u64)> {
    use cudarc::driver::DevicePtrMut;
    use half::bf16;
    if let Some((t, p)) = lock.get() {
        return Ok((t.clone(), *p));
    }
    let mut buf: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(elems)? };
    let ptr = {
        let (p, _g) = buf.device_ptr_mut(stream);
        p
    };
    let storage = CudaStorage::wrap_cuda_slice(buf, dev.clone());
    let storage = Storage::Cuda(storage);
    let t = Tensor::from_storage(storage, (1usize, elems), BackpropOp::none(), false);
    let _ = lock.set((t, ptr));
    let (t, p) = lock.get().unwrap();
    Ok((t.clone(), *p))
}

#[cfg(feature = "cuda")]
impl GemvM1Hybrid {
    fn persist_y(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        dev: &CudaDevice,
    ) -> Result<(Tensor, u64)> {
        persist_bf16_row(&self.y_persist, self.gemv.out_features, stream, dev)
    }

    fn m1_operands(
        &self,
        x: &Tensor,
    ) -> Result<(
        CudaDevice,
        std::sync::Arc<cudarc::driver::CudaStream>,
        Tensor,
    )> {
        let kin = *x.dims().last().unwrap();
        ensure!(kin == self.gemv.in_features, "gemv_m1 in mismatch");
        let Device::Cuda(dev) = self.gemv.device.clone() else {
            bail!("gemv_m1 requires cuda")
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let x2 = x.reshape((1, kin))?.to_dtype(DType::BF16)?.contiguous()?;
        Ok((dev, stream, x2))
    }

    fn forward_geglu_pairing_gate_row_n_with_up_row_n_plus_inter(
        &self,
        x: &Tensor,
    ) -> Result<Tensor> {
        use cudarc::driver::DevicePtr;
        use half::bf16;
        let m: usize = x.dims()[..x.dims().len() - 1].iter().product();
        ensure!(m == 1, "the geglu fold is M=1 only");
        ensure!(
            self.gemv.out_features % 2 == 0,
            "geglu needs gate rows then up rows"
        );
        let inter = self.gemv.out_features / 2;
        let (dev, stream, x2) = self.m1_operands(x)?;
        let (yt, yp) = persist_bf16_row(&self.geglu_persist, inter, &stream, &dev)?;
        let rc = {
            let (xs, _) = x2.storage_and_layout();
            let (ps, _) = self.gemv.packed.storage_and_layout();
            let (ss, _) = self.gemv.scale.storage_and_layout();
            let (Storage::Cuda(xc), Storage::Cuda(pc), Storage::Cuda(sc)) = (&*xs, &*ps, &*ss)
            else {
                bail!("gemv_m1: operands not cuda")
            };
            let (xp, _gx) = xc.as_cuda_slice::<bf16>()?.device_ptr(&stream);
            let (pp, _gp) = pc.as_cuda_slice::<u32>()?.device_ptr(&stream);
            let (sp, _gs) = sc.as_cuda_slice::<bf16>()?.device_ptr(&stream);
            unsafe {
                nv_kernels::cuda::gemv_w4a16_m1_proto_geglu(
                    stream.cu_stream() as *mut _,
                    pp as *const u32,
                    sp as *const u16,
                    xp as *const u16,
                    yp as *mut u16,
                    self.gemv.out_features as i32,
                    self.gemv.in_features as i32,
                    self.gemv.group_size as i32,
                    self.variant,
                )
            }
        };
        ensure!(rc == 0, "gemv_w4a16_m1_proto_geglu returned {rc}");
        yt.reshape((1usize, inter)).map_err(Into::into)
    }

    fn forward_then_rmsnorm_add_scale_in_the_last_block(
        &self,
        x: &Tensor,
        norm: &nv_layers::norm::RmsNorm,
        res: &Tensor,
        scale: f32,
        chain_eps: Option<f32>,
        next_w: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
        use cudarc::driver::DevicePtr;
        use half::bf16;
        let m: usize = x.dims()[..x.dims().len() - 1].iter().product();
        ensure!(m == 1, "the post-norm fold is M=1 only");
        let n = self.gemv.out_features;
        let (dev, stream, x2) = self.m1_operands(x)?;
        let (_yt, yp) = self.persist_y(&stream, &dev)?;
        let res_c = res.contiguous()?;
        ensure!(
            res_c.dtype() == DType::BF16 && res_c.elem_count() == n,
            "post-norm fold residual mismatch"
        );
        let w = norm.weight_bf16();
        ensure!(
            w.dtype() == DType::BF16 && w.elem_count() == n,
            "norm weight mismatch"
        );
        if next_w.is_some() {
            ensure!(chain_eps.is_some(), "next_w requires chain_eps");
        }
        let counter = match self.post_norm_last_block_counter.get() {
            Some(c) => c,
            None => {
                let z = stream.alloc_zeros::<u32>(1)?;
                let _ = self.post_norm_last_block_counter.set(z);
                self.post_norm_last_block_counter.get().unwrap()
            }
        };
        let cdev = Device::Cuda(dev.clone());
        let h_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
            &cdev,
            DType::BF16,
            &[n],
        )?;
        let rstd_dev = match chain_eps {
            Some(_) => Some(
                nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
                    &cdev,
                    DType::F32,
                    &[1],
                )?,
            ),
            None => None,
        };
        let normed_dev = match next_w {
            Some(nw) => {
                ensure!(
                    nw.dtype() == DType::BF16 && nw.elem_count() == n,
                    "next_w mismatch"
                );
                Some(
                    nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
                        &cdev,
                        DType::BF16,
                        &[n],
                    )?,
                )
            }
            None => None,
        };
        let rc = {
            let (xs, _) = x2.storage_and_layout();
            let (ps, _) = self.gemv.packed.storage_and_layout();
            let (ss, _) = self.gemv.scale.storage_and_layout();
            let (rs, _) = res_c.storage_and_layout();
            let (ws, _) = w.storage_and_layout();
            let (
                Storage::Cuda(xc),
                Storage::Cuda(pc),
                Storage::Cuda(sc),
                Storage::Cuda(rcu),
                Storage::Cuda(wc),
            ) = (&*xs, &*ps, &*ss, &*rs, &*ws)
            else {
                bail!("gemv_m1: operands not cuda")
            };
            let (xp, _gx) = xc.as_cuda_slice::<bf16>()?.device_ptr(&stream);
            let (pp, _gp) = pc.as_cuda_slice::<u32>()?.device_ptr(&stream);
            let (sp, _gs) = sc.as_cuda_slice::<bf16>()?.device_ptr(&stream);
            let (rp, _gr) = rcu.as_cuda_slice::<bf16>()?.device_ptr(&stream);
            let (wp, _gw) = wc.as_cuda_slice::<bf16>()?.device_ptr(&stream);
            let (cp, _gc) = counter.device_ptr(&stream);
            let hp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&h_dev)?;
            let rstd_ptr = match rstd_dev.as_ref() {
                Some(b) => nv_layers::step_arena::device_ptr_for_a_kernel_to_write(b)? as *mut f32,
                None => std::ptr::null_mut(),
            };
            let next_w_ptr = match next_w {
                Some(nw) => {
                    let (ns, _nl) = nw.storage_and_layout();
                    let Storage::Cuda(nc) = &*ns else {
                        bail!("next_w not cuda")
                    };
                    let (np, _gn) = nc.as_cuda_slice::<bf16>()?.device_ptr(&stream);
                    np as *const u16
                }
                None => std::ptr::null(),
            };
            let normed_ptr = match normed_dev.as_ref() {
                Some(b) => nv_layers::step_arena::device_ptr_for_a_kernel_to_write(b)? as *mut u16,
                None => std::ptr::null_mut(),
            };
            unsafe {
                nv_kernels::cuda::gemv_w4a16_m1_proto_post_norm(
                    stream.cu_stream() as *mut _,
                    pp as *const u32,
                    sp as *const u16,
                    xp as *const u16,
                    yp as *mut u16,
                    n as i32,
                    self.gemv.in_features as i32,
                    self.gemv.group_size as i32,
                    self.variant,
                    rp as *const u16,
                    wp as *const u16,
                    hp as *mut u16,
                    rstd_ptr,
                    next_w_ptr,
                    normed_ptr,
                    norm.eps() as f32,
                    scale,
                    chain_eps.unwrap_or(0.0),
                    cp as *mut u32,
                )
            }
        };
        ensure!(rc == 0, "gemv_w4a16_m1_proto_post_norm returned {rc}");
        let h = h_dev.reshape((1usize, n))?;
        let normed = normed_dev.map(|b| b.reshape((1usize, n))).transpose()?;
        Ok((h, rstd_dev, normed))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use cudarc::driver::DevicePtr;
        use half::bf16;
        let dims = x.dims().to_vec();
        let kin = *dims.last().unwrap();
        let m: usize = dims[..dims.len() - 1].iter().product();
        if m != 1 {
            return self.marlin.forward(x);
        }
        ensure!(kin == self.gemv.in_features, "gemv_m1 in mismatch");
        let Device::Cuda(dev) = self.gemv.device.clone() else {
            bail!("gemv_m1 requires cuda")
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let (yt, yp) = self.persist_y(&stream, &dev)?;
        let x2 = x.reshape((1, kin))?.to_dtype(DType::BF16)?.contiguous()?;
        let rc = {
            let (xs, _) = x2.storage_and_layout();
            let (ps, _) = self.gemv.packed.storage_and_layout();
            let (ss, _) = self.gemv.scale.storage_and_layout();
            let xc = match &*xs {
                Storage::Cuda(s) => s,
                _ => bail!("gemv_m1: x not cuda"),
            };
            let pc = match &*ps {
                Storage::Cuda(s) => s,
                _ => bail!("gemv_m1: packed not cuda"),
            };
            let sc = match &*ss {
                Storage::Cuda(s) => s,
                _ => bail!("gemv_m1: scale not cuda"),
            };
            let xsl = xc.as_cuda_slice::<bf16>()?;
            let psl = pc.as_cuda_slice::<u32>()?;
            let ssl = sc.as_cuda_slice::<bf16>()?;
            let (xp, _gx) = xsl.device_ptr(&stream);
            let (pp, _gp) = psl.device_ptr(&stream);
            let (sp, _gs) = ssl.device_ptr(&stream);
            unsafe {
                nv_kernels::cuda::gemv_w4a16_m1_proto(
                    stream.cu_stream() as *mut _,
                    pp as *const u32,
                    sp as *const u16,
                    xp as *const u16,
                    yp as *mut u16,
                    self.gemv.out_features as i32,
                    self.gemv.in_features as i32,
                    self.gemv.group_size as i32,
                    self.variant,
                )
            }
        };
        ensure!(rc == 0, "gemv_w4a16_m1 returned {rc}");
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(self.gemv.out_features);
        yt.reshape(out_dims).map_err(Into::into)
    }
}

impl QLinear {
    #[cfg(feature = "cuda")]
    fn forward_gelu_pli(&self, x: &Tensor, pli: &Tensor) -> Option<Result<Tensor>> {
        match self {
            QLinear::Quant(q) if q.in_features <= 3072 && q.group_size >= 32 => {
                Some(q.forward_gelu_pli(x, pli))
            }
            _ => None,
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            QLinear::Dense(l) => l.forward(x),
            #[cfg(feature = "cuda")]
            QLinear::Quant(q) => q.forward(x),
            #[cfg(feature = "cuda")]
            QLinear::Marlin(m) => m.forward(x),
            #[cfg(feature = "cuda")]
            QLinear::GemvM1(h) => h.forward(x),
        }
    }
}

#[cfg(feature = "cuda")]
struct W4a16Linear {
    packed: Tensor,
    scale: Tensor,
    out_features: usize,
    in_features: usize,
    group_size: usize,
    device: Device,
}

#[cfg(feature = "cuda")]
impl W4a16Linear {
    fn from_raw(
        pbytes: &[u8],
        scale: Tensor,
        out_features: usize,
        in_features: usize,
        device: &Device,
    ) -> Result<Self> {
        let group_size = in_features / scale.dim(1)?;
        let n = pbytes.len() / 4;
        let mut vals = Vec::with_capacity(n);
        for i in 0..n {
            let b = 4 * i;
            vals.push(u32::from_le_bytes([
                pbytes[b],
                pbytes[b + 1],
                pbytes[b + 2],
                pbytes[b + 3],
            ]));
        }
        let packed = Tensor::from_vec(vals, (out_features, in_features / 8), device)?;
        Ok(Self {
            packed,
            scale,
            out_features,
            in_features,
            group_size,
            device: device.clone(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use cudarc::driver::DevicePtr;
        use half::bf16;
        use nv_kernels::cuda as nvk;

        let dims = x.dims().to_vec();
        let kin = *dims.last().unwrap();
        ensure!(kin == self.in_features, "w4a16 in mismatch");
        let m: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((m, kin))?.to_dtype(DType::BF16)?.contiguous()?;
        let Device::Cuda(dev) = self.device.clone() else {
            bail!("w4a16 requires cuda")
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let y_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
            &Device::Cuda(dev.clone()),
            DType::BF16,
            &[m * self.out_features],
        )?;
        {
            let (xs, _) = x2.storage_and_layout();
            let (ps, _) = self.packed.storage_and_layout();
            let (ss, _) = self.scale.storage_and_layout();
            let xc = match &*xs {
                Storage::Cuda(s) => s,
                _ => bail!("w4a16: x not cuda"),
            };
            let pc = match &*ps {
                Storage::Cuda(s) => s,
                _ => bail!("w4a16: packed not cuda"),
            };
            let sc = match &*ss {
                Storage::Cuda(s) => s,
                _ => bail!("w4a16: scale not cuda"),
            };
            let xsl = xc.as_cuda_slice::<bf16>()?;
            let psl = pc.as_cuda_slice::<u32>()?;
            let ssl = sc.as_cuda_slice::<bf16>()?;
            let (xp, _gx) = xsl.device_ptr(&stream);
            let (pp, _gp) = psl.device_ptr(&stream);
            let (sp, _gs2) = ssl.device_ptr(&stream);
            let yp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&y_dev)?;
            let xp = xp as *const u16;
            let pp = pp as *const u32;
            let sp = sp as *const u16;
            let yp = yp as *mut u16;
            for mi in 0..m {
                let rc = unsafe {
                    nvk::gemv_w4a16(
                        stream.cu_stream() as *mut _,
                        pp,
                        sp,
                        xp.add(mi * kin),
                        yp.add(mi * self.out_features),
                        self.out_features as i32,
                        kin as i32,
                        self.group_size as i32,
                    )
                };
                if rc != 0 {
                    bail!("gemv_w4a16 returned {rc}");
                }
            }
        }
        let out = y_dev.reshape((m, self.out_features))?;
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(self.out_features);
        out.reshape(out_dims).map_err(Into::into)
    }
}

#[cfg(feature = "cuda")]
impl W4a16Linear {
    fn forward_gelu_pli(&self, x: &Tensor, pli: &Tensor) -> Result<Tensor> {
        use cudarc::driver::DevicePtr;
        use half::bf16;
        ensure!(
            self.in_features <= 3072 && self.group_size >= 32,
            "fused pli shape"
        );
        let dims = x.dims().to_vec();
        ensure!(
            *dims.last().unwrap() == self.in_features,
            "fused pli in mismatch"
        );
        let m: usize = dims[..dims.len() - 1].iter().product();
        ensure!(m == 1, "fused pli is M=1 only");
        ensure!(pli.elem_count() == self.out_features, "pli numel mismatch");
        let Device::Cuda(dev) = self.device.clone() else {
            bail!("fused pli requires cuda")
        };
        let x2 = x
            .reshape((1, self.in_features))?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let y_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
            &Device::Cuda(dev.clone()),
            DType::BF16,
            &[self.out_features],
        )?;
        let rc = {
            let (xs, _) = x2.storage_and_layout();
            let (ps, _) = self.packed.storage_and_layout();
            let (ss, _) = self.scale.storage_and_layout();
            let (pls, pll) = pli.storage_and_layout();
            let (p0, p1) = pll
                .contiguous_offsets()
                .ok_or_else(|| anyhow::anyhow!("pli not dense"))?;
            let xc = match &*xs {
                Storage::Cuda(s) => s,
                _ => bail!("x not cuda"),
            };
            let pc = match &*ps {
                Storage::Cuda(s) => s,
                _ => bail!("packed not cuda"),
            };
            let sc = match &*ss {
                Storage::Cuda(s) => s,
                _ => bail!("scale not cuda"),
            };
            let plc = match &*pls {
                Storage::Cuda(s) => s,
                _ => bail!("pli not cuda"),
            };
            let xsl = xc.as_cuda_slice::<bf16>()?;
            let psl = pc.as_cuda_slice::<u32>()?;
            let ssl = sc.as_cuda_slice::<bf16>()?;
            let plsl = plc.as_cuda_slice::<f32>()?;
            let plview = plsl.slice(p0..p1);
            let (xp, _g1) = xsl.device_ptr(&stream);
            let (pp, _g2) = psl.device_ptr(&stream);
            let (sp, _g3) = ssl.device_ptr(&stream);
            let (plp, _g4) = plview.device_ptr(&stream);
            let yp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&y_dev)?;
            unsafe {
                nv_kernels::cuda::gemv_w4a16_gelu_pli(
                    stream.cu_stream() as *mut _,
                    pp as *const u32,
                    sp as *const u16,
                    xp as *const u16,
                    plp as *const f32,
                    yp as *mut u16,
                    self.out_features as i32,
                    self.in_features as i32,
                    self.group_size as i32,
                )
            }
        };
        ensure!(rc == 0, "gemv_w4a16_gelu_pli returned {rc}");
        y_dev
            .reshape((1usize, self.out_features))
            .map_err(Into::into)
    }
}

#[cfg(feature = "cuda")]
pub(crate) enum MarlinM1Tail<'a> {
    RmsnormAddScale {
        norm: &'a nv_layers::norm::RmsNorm,
        res: &'a Tensor,
        scale: f32,
        chain_eps: Option<f32>,
        next_w: Option<&'a Tensor>,
    },
    GeluPli {
        pli: &'a Tensor,
    },
}

#[cfg(feature = "cuda")]
pub(crate) struct MarlinLinear {
    b_q_marlin: CudaSlice<i32>,
    b_scales: Tensor,
    workspace: CudaSlice<i32>,

    c_tmp: Option<std::sync::Arc<CudaSlice<f32>>>,
    c_persist: std::sync::OnceLock<(Tensor, u64)>,
    m1_tail_last_block_counter: std::sync::OnceLock<CudaSlice<u32>>,
    a8_fp8: bool,
    out_features: usize,
    in_features: usize,
    group_size: usize,
    device: CudaDevice,
}

#[cfg(feature = "cuda")]
struct MarlinZeroRegistry {
    entries: Vec<u64>,
    dev: Option<CudaSlice<u64>>,
    dev_ptr: u64,
    version: u64,
    uploaded_version: u64,
    uploaded_count: i32,
}

#[cfg(feature = "cuda")]
const MARLIN_ZERO_CAP: usize = 1024;

#[cfg(feature = "cuda")]
fn marlin_zero_registry() -> &'static std::sync::Mutex<MarlinZeroRegistry> {
    use std::sync::{Mutex, OnceLock};
    static REG: OnceLock<Mutex<MarlinZeroRegistry>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(MarlinZeroRegistry {
            entries: Vec::new(),
            dev: None,
            dev_ptr: 0,
            version: 0,
            uploaded_version: 0,
            uploaded_count: 0,
        })
    })
}

#[cfg(feature = "cuda")]
fn marlin_zero_register(ptr: u64, elems: u64) -> bool {
    let mut reg = marlin_zero_registry().lock().unwrap();
    if reg.entries.len() / 2 >= MARLIN_ZERO_CAP {
        return false;
    }
    reg.entries.push(ptr);
    reg.entries.push(elems);
    reg.version += 1;
    true
}

#[cfg(feature = "cuda")]
pub(crate) fn marlin_zero_upload(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<()> {
    use cudarc::driver::DevicePtrMut;
    let mut reg = marlin_zero_registry().lock().unwrap();
    if reg.version == reg.uploaded_version {
        return Ok(());
    }
    if reg.dev.is_none() {
        let mut dev: CudaSlice<u64> = stream.alloc_zeros::<u64>(2 * MARLIN_ZERO_CAP)?;
        let ptr = {
            let (p, _g) = dev.device_ptr_mut(stream);
            p
        };
        reg.dev = Some(dev);
        reg.dev_ptr = ptr;
    }
    let mut host = reg.entries.clone();
    host.resize(2 * MARLIN_ZERO_CAP, 0);
    let count = (reg.entries.len() / 2) as i32;
    let version = reg.version;
    stream.memcpy_htod(&host, reg.dev.as_mut().unwrap())?;
    stream.synchronize()?;
    reg.uploaded_version = version;
    reg.uploaded_count = count;
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn marlin_zero_launch(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<()> {
    let (ptr, count) = {
        let reg = marlin_zero_registry().lock().unwrap();
        (reg.dev_ptr, reg.uploaded_count)
    };
    if count == 0 {
        return Ok(());
    }
    let rc = unsafe {
        nv_kernels::cuda::multi_zero_bf16(
            stream.cu_stream() as *mut _,
            ptr as *const std::ffi::c_void,
            count,
        )
    };
    ensure!(rc == 0, "multi_zero_bf16 returned {rc}");
    Ok(())
}

#[cfg(feature = "cuda")]
fn marlin_zero_deregister_then_reupload_before_the_buffer_frees_so_no_zero_launch_scribbles_the_reused_address(
    ptr: u64,
    device: &CudaDevice,
) {
    let reupload = {
        let mut reg = marlin_zero_registry().lock().unwrap();
        let before = reg.entries.len();
        let mut i = 0;
        while i < reg.entries.len() {
            if reg.entries[i] == ptr {
                reg.entries.drain(i..i + 2);
            } else {
                i += 2;
            }
        }
        if reg.entries.len() == before {
            false
        } else {
            reg.version += 1;
            reg.dev.is_some()
        }
    };
    if reupload {
        let stream = nv_layers::cuda_stream::current_stream(device);
        let _ = marlin_zero_upload(&stream);
    }
}

#[cfg(feature = "cuda")]
impl Drop for MarlinLinear {
    fn drop(&mut self) {
        if let Some((_t, ptr)) = self.c_persist.get() {
            marlin_zero_deregister_then_reupload_before_the_buffer_frees_so_no_zero_launch_scribbles_the_reused_address(*ptr, &self.device);
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) fn marlin_zero_snapshot_host_entries_uploaded_device_list_and_count(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<(Vec<u64>, Vec<u64>, i32)> {
    let reg = marlin_zero_registry().lock().unwrap();
    let dev_list = match &reg.dev {
        Some(d) => stream.clone_dtoh(d)?,
        None => Vec::new(),
    };
    Ok((reg.entries.clone(), dev_list, reg.uploaded_count))
}

#[cfg(feature = "cuda")]
fn marlin_c_tmp_scratch(
    n: usize,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<std::sync::Arc<CudaSlice<f32>>> {
    use std::sync::{Arc, Mutex, OnceLock};
    static SCRATCH: OnceLock<Mutex<Option<(usize, Arc<CudaSlice<f32>>)>>> = OnceLock::new();
    let elems = 64 * n;
    let cell = SCRATCH.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap();
    if let Some((cap, buf)) = guard.as_ref() {
        if *cap >= elems {
            return Ok(buf.clone());
        }
    }
    let buf: Arc<CudaSlice<f32>> = Arc::new(stream.alloc_zeros::<f32>(elems)?);
    *guard = Some((elems, buf.clone()));
    Ok(buf)
}

#[cfg(feature = "cuda")]
fn marlin_a8_input_scratch(
    q_elems: usize,
    s_elems: usize,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<(
    std::sync::Arc<CudaSlice<u8>>,
    std::sync::Arc<CudaSlice<f32>>,
)> {
    use std::sync::{Arc, Mutex, OnceLock};
    type Bufs = (usize, Arc<CudaSlice<u8>>, usize, Arc<CudaSlice<f32>>);
    static SCRATCH: OnceLock<Mutex<Option<Bufs>>> = OnceLock::new();
    let cell = SCRATCH.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap();
    if let Some((qcap, q, scap, s)) = guard.as_ref() {
        if *qcap >= q_elems && *scap >= s_elems {
            return Ok((q.clone(), s.clone()));
        }
    }
    let qcap = q_elems.max(guard.as_ref().map(|g| g.0).unwrap_or(0));
    let scap = s_elems.max(guard.as_ref().map(|g| g.2).unwrap_or(0));
    let q: Arc<CudaSlice<u8>> = Arc::new(unsafe { stream.alloc::<u8>(qcap)? });
    let s: Arc<CudaSlice<f32>> = Arc::new(unsafe { stream.alloc::<f32>(scap)? });
    *guard = Some((qcap, q.clone(), scap, s.clone()));
    Ok((q, s))
}

#[cfg(feature = "cuda")]
impl MarlinLinear {
    pub(crate) fn from_raw(
        pbytes: &[u8],
        scale: Tensor,
        out: usize,
        inp: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::from_raw_backend(pbytes, scale, out, inp, device, false)
    }

    fn from_raw_backend(
        pbytes: &[u8],
        scale: Tensor,
        out: usize,
        inp: usize,
        device: &Device,
        a8_fp8: bool,
    ) -> Result<Self> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let Device::Cuda(dev) = device.clone() else {
            bail!("marlin requires cuda")
        };
        let group_size = inp / scale.dim(1)?;
        ensure!(
            !a8_fp8 || group_size >= 32,
            "marlin w4a8-fp8 needs group_size >= 32, got {group_size}"
        );

        let s = scale.t()?.contiguous()?;
        let ng = s.dim(0)?;
        let n = s.dim(1)?;
        ensure!((ng * n) % 64 == 0, "marlin scale reshape: {ng}x{n} not %64");
        let b_scales = if a8_fp8 {
            let mut perm = Vec::with_capacity(32);
            for i in 0..4u32 {
                for j in [0u32, 1, 8, 9, 16, 17, 24, 25] {
                    perm.push(2 * i + j);
                }
            }
            let perm_t = Tensor::from_vec(perm, 32, device)?;
            let s512 = (s.to_dtype(DType::F32)? * 512.0)?.to_dtype(DType::BF16)?;
            s512.reshape((ng * n / 32, 32))?
                .index_select(&perm_t, 1)?
                .reshape((ng, n))?
                .contiguous()?
        } else {
            let mut perm = Vec::with_capacity(64);
            for i in 0..8u32 {
                for j in 0..8u32 {
                    perm.push(i + 8 * j);
                }
            }
            let perm_t = Tensor::from_vec(perm, 64, device)?;
            s.reshape((ng * n / 64, 64))?
                .index_select(&perm_t, 1)?
                .reshape((ng, n))?
                .contiguous()?
        };

        let cols = inp / 8;
        ensure!(pbytes.len() == out * cols * 4, "packed size mismatch");
        let mut tvec = vec![0i32; cols * out];
        for o in 0..out {
            for k8 in 0..cols {
                let b = 4 * (o * cols + k8);
                let v =
                    i32::from_le_bytes([pbytes[b], pbytes[b + 1], pbytes[b + 2], pbytes[b + 3]]);
                tvec[k8 * out + o] = v;
            }
        }
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let b_q_packed: CudaSlice<i32> = stream.clone_htod(&tvec)?;
        let mut b_q_marlin: CudaSlice<i32> = unsafe { stream.alloc::<i32>(inp * out / 8)? };
        let rc = {
            let (pp, _gp) = b_q_packed.device_ptr(&stream);
            let (mp, _gm) = b_q_marlin.device_ptr_mut(&stream);
            if a8_fp8 {
                let pre = unsafe {
                    nv_kernels::cuda::marlin_int4_fp8_preprocess(
                        stream.cu_stream() as *mut _,
                        pp as *const std::ffi::c_void,
                        pp as *mut std::ffi::c_void,
                        (inp * out / 8) as i64,
                    )
                };
                ensure!(pre == 0, "marlin_int4_fp8_preprocess returned {pre}");
                unsafe {
                    nv_kernels::cuda::marlin_repack_w4a8(
                        stream.cu_stream() as *mut _,
                        pp as *const std::ffi::c_void,
                        mp as *mut std::ffi::c_void,
                        inp as i32,
                        out as i32,
                        4,
                    )
                }
            } else {
                unsafe {
                    nv_kernels::cuda::marlin_repack_w4a16(
                        stream.cu_stream() as *mut _,
                        pp as *const std::ffi::c_void,
                        mp as *mut std::ffi::c_void,
                        inp as i32,
                        out as i32,
                        4,
                    )
                }
            }
        };
        ensure!(rc == 0, "marlin_repack returned {rc}");
        drop(b_q_packed);

        let mut elems: i32 = 0;
        unsafe {
            nv_kernels::cuda::marlin_workspace_elems(&mut elems as *mut i32);
        }
        ensure!(elems > 0, "marlin workspace elems = {elems}");
        let workspace: CudaSlice<i32> = stream.clone_htod(&vec![0i32; elems as usize])?;

        let c_tmp = if nv_config::env_arm_present_at_all_reread_every_call("NV_E4B_MARLIN_FP32RED")
        {
            Some(marlin_c_tmp_scratch(out, &stream)?)
        } else {
            None
        };

        Ok(Self {
            b_q_marlin,
            b_scales,
            workspace,
            c_tmp,
            c_persist: std::sync::OnceLock::new(),
            m1_tail_last_block_counter: std::sync::OnceLock::new(),
            a8_fp8,
            out_features: out,
            in_features: inp,
            group_size,
            device: dev,
        })
    }

    fn persist_out(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<Option<(Tensor, u64)>> {
        use cudarc::driver::DevicePtrMut;
        use half::bf16;
        if let Some((t, p)) = self.c_persist.get() {
            return Ok(Some((t.clone(), *p)));
        }
        let mut buf: CudaSlice<bf16> = stream.alloc_zeros::<bf16>(self.out_features)?;
        let ptr = {
            let (p, _g) = buf.device_ptr_mut(stream);
            p
        };
        if !marlin_zero_register(ptr, self.out_features as u64) {
            return Ok(None);
        }
        let storage = CudaStorage::wrap_cuda_slice(buf, self.device.clone());
        let storage = Storage::Cuda(storage);
        let t = Tensor::from_storage(
            storage,
            (1usize, self.out_features),
            BackpropOp::none(),
            false,
        );
        let _ = self.c_persist.set((t, ptr));
        let (t, p) = self.c_persist.get().unwrap();
        Ok(Some((t.clone(), *p)))
    }

    fn forward_m1_then_tail_in_the_last_block(
        &self,
        x: &Tensor,
        tail: MarlinM1Tail<'_>,
    ) -> Result<Option<(Tensor, Option<Tensor>, Option<Tensor>)>> {
        use cudarc::driver::DevicePtr;
        use half::bf16;
        let dims = x.dims().to_vec();
        let kin = *dims.last().unwrap();
        ensure!(kin == self.in_features, "marlin in mismatch");
        let m: usize = dims[..dims.len() - 1].iter().product();
        if m != 1 || self.a8_fp8 {
            return Ok(None);
        }
        let stream = nv_layers::cuda_stream::current_stream(&self.device);
        let Some((_ct, cptr)) = self.persist_out(&stream)? else {
            return Ok(None);
        };
        let n = self.out_features;
        let x2 = x.reshape((1, kin))?.to_dtype(DType::BF16)?.contiguous()?;
        let counter = match self.m1_tail_last_block_counter.get() {
            Some(c) => c,
            None => {
                let z = stream.alloc_zeros::<u32>(1)?;
                let _ = self.m1_tail_last_block_counter.set(z);
                self.m1_tail_last_block_counter.get().unwrap()
            }
        };
        let cdev = Device::Cuda(self.device.clone());
        let arena = |dtype: DType, len: usize| {
            nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
                &cdev,
                dtype,
                &[len],
            )
        };
        let cuda_ptr = |t: &Tensor| -> Result<u64> {
            let (st, layout) = t.storage_and_layout();
            let Storage::Cuda(c) = &*st else {
                bail!("marlin tail operand not cuda")
            };
            let (p0, _p1) = layout
                .contiguous_offsets()
                .ok_or_else(|| anyhow::anyhow!("marlin tail operand not dense"))?;
            Ok(match t.dtype() {
                DType::BF16 => {
                    let (p, _g) = c.as_cuda_slice::<bf16>()?.device_ptr(&stream);
                    p + (p0 * 2) as u64
                }
                DType::F32 => {
                    let (p, _g) = c.as_cuda_slice::<f32>()?.device_ptr(&stream);
                    p + (p0 * 4) as u64
                }
                other => bail!("marlin tail operand dtype {other:?}"),
            })
        };
        let (
            kind,
            res_p,
            w_p,
            h_out,
            rstd_out,
            next_w_p,
            normed_out,
            pli_p,
            gelu_out,
            eps,
            scale,
            eps_next,
        ) = match tail {
            MarlinM1Tail::RmsnormAddScale {
                norm,
                res,
                scale,
                chain_eps,
                next_w,
            } => {
                ensure!(
                    res.dtype() == DType::BF16 && res.elem_count() == n,
                    "marlin tail residual mismatch"
                );
                let w = norm.weight_bf16();
                ensure!(w.elem_count() == n, "marlin tail norm weight mismatch");
                if let Some(nw) = next_w {
                    ensure!(
                        chain_eps.is_some() && nw.elem_count() == n,
                        "marlin tail next_w mismatch"
                    );
                }
                let h = arena(DType::BF16, n)?;
                let rstd = chain_eps.map(|_| arena(DType::F32, 1)).transpose()?;
                let normed = next_w.map(|_| arena(DType::BF16, n)).transpose()?;
                (
                    1i32,
                    cuda_ptr(res)?,
                    cuda_ptr(w)?,
                    Some(h),
                    rstd,
                    next_w.map(cuda_ptr).transpose()?.unwrap_or(0),
                    normed,
                    0u64,
                    None,
                    norm.eps() as f32,
                    scale,
                    chain_eps.unwrap_or(0.0),
                )
            }
            MarlinM1Tail::GeluPli { pli } => {
                ensure!(
                    pli.dtype() == DType::F32 && pli.elem_count() == n,
                    "marlin tail pli mismatch"
                );
                (
                    2i32,
                    0u64,
                    0u64,
                    None,
                    None,
                    0u64,
                    None,
                    cuda_ptr(pli)?,
                    Some(arena(DType::BF16, n)?),
                    0.0,
                    0.0,
                    0.0,
                )
            }
        };
        let out_ptr = |t: &Option<Tensor>| -> Result<u64> {
            match t {
                Some(t) => nv_layers::step_arena::device_ptr_for_a_kernel_to_write(t),
                None => Ok(0),
            }
        };
        let (hp, rsp, nop, gp) = (
            out_ptr(&h_out)?,
            out_ptr(&rstd_out)?,
            out_ptr(&normed_out)?,
            out_ptr(&gelu_out)?,
        );
        let rc = {
            let (xs, _xl) = x2.storage_and_layout();
            let (ss, _sl) = self.b_scales.storage_and_layout();
            let (Storage::Cuda(xc), Storage::Cuda(sc)) = (&*xs, &*ss) else {
                bail!("marlin: operands not cuda")
            };
            let (xp, _gx) = xc.as_cuda_slice::<bf16>()?.device_ptr(&stream);
            let (sp, _gs) = sc.as_cuda_slice::<bf16>()?.device_ptr(&stream);
            let (bp, _gb) = self.b_q_marlin.device_ptr(&stream);
            let (wp, _gw) = self.workspace.device_ptr(&stream);
            let (cp, _gc) = counter.device_ptr(&stream);
            unsafe {
                nv_kernels::cuda::marlin_gemm_w4a16_m1_tail(
                    stream.cu_stream() as *mut _,
                    xp as *const std::ffi::c_void,
                    bp as *const std::ffi::c_void,
                    sp as *const std::ffi::c_void,
                    cptr as *mut std::ffi::c_void,
                    wp as *mut std::ffi::c_void,
                    n as i32,
                    self.in_features as i32,
                    self.group_size as i32,
                    kind,
                    res_p as *const u16,
                    w_p as *const u16,
                    hp as *mut u16,
                    rsp as *mut f32,
                    next_w_p as *const u16,
                    nop as *mut u16,
                    pli_p as *const f32,
                    gp as *mut u16,
                    eps,
                    scale,
                    eps_next,
                    cp as *mut u32,
                )
            }
        };
        ensure!(rc == 0, "marlin_gemm_w4a16_m1_tail returned {rc}");
        let first = match kind {
            1 => h_out.unwrap(),
            _ => gelu_out.unwrap(),
        }
        .reshape((1usize, n))?;
        Ok(Some((
            first,
            rstd_out,
            normed_out.map(|t| t.reshape((1usize, n))).transpose()?,
        )))
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use cudarc::driver::DevicePtr;
        use half::bf16;
        let dims = x.dims().to_vec();
        let kin = *dims.last().unwrap();
        ensure!(kin == self.in_features, "marlin in mismatch");
        let m: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((m, kin))?.to_dtype(DType::BF16)?.contiguous()?;
        let stream = nv_layers::cuda_stream::current_stream(&self.device);
        if self.a8_fp8 {
            return self.forward_a8(&x2, m, &dims, &stream);
        }
        if m == 1 && self.c_tmp.is_none() {
            if let Some((t, cptr)) = self.persist_out(&stream)? {
                let rc = {
                    let (xs, _xl) = x2.storage_and_layout();
                    let (ss, _sl) = self.b_scales.storage_and_layout();
                    let xc = match &*xs {
                        Storage::Cuda(s) => s,
                        _ => bail!("marlin: x not cuda"),
                    };
                    let sc = match &*ss {
                        Storage::Cuda(s) => s,
                        _ => bail!("marlin: scales not cuda"),
                    };
                    let xsl = xc.as_cuda_slice::<bf16>()?;
                    let ssl = sc.as_cuda_slice::<bf16>()?;
                    let (xp, _gx) = xsl.device_ptr(&stream);
                    let (sp, _gs) = ssl.device_ptr(&stream);
                    let (bp, _gb) = self.b_q_marlin.device_ptr(&stream);
                    let (wp, _gw) = self.workspace.device_ptr(&stream);
                    unsafe {
                        nv_kernels::cuda::marlin_gemm_w4a16_prezeroed(
                            stream.cu_stream() as *mut _,
                            xp as *const std::ffi::c_void,
                            bp as *const std::ffi::c_void,
                            sp as *const std::ffi::c_void,
                            cptr as *mut std::ffi::c_void,
                            wp as *mut std::ffi::c_void,
                            1,
                            self.out_features as i32,
                            self.in_features as i32,
                            self.group_size as i32,
                            1,
                        )
                    }
                };
                ensure!(rc == 0, "marlin_gemm returned {rc}");
                let mut out_dims = dims[..dims.len() - 1].to_vec();
                out_dims.push(self.out_features);
                return t.reshape(out_dims).map_err(Into::into);
            }
        }
        let c_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
            &Device::Cuda(self.device.clone()),
            DType::BF16,
            &[m * self.out_features],
        )?;
        let rc = {
            let (xs, _xl) = x2.storage_and_layout();
            let (ss, _sl) = self.b_scales.storage_and_layout();
            let xc = match &*xs {
                Storage::Cuda(s) => s,
                _ => bail!("marlin: x not cuda"),
            };
            let sc = match &*ss {
                Storage::Cuda(s) => s,
                _ => bail!("marlin: scales not cuda"),
            };
            let xsl = xc.as_cuda_slice::<bf16>()?;
            let ssl = sc.as_cuda_slice::<bf16>()?;
            let (xp, _gx) = xsl.device_ptr(&stream);
            let (sp, _gs) = ssl.device_ptr(&stream);
            let (bp, _gb) = self.b_q_marlin.device_ptr(&stream);
            let (wp, _gw) = self.workspace.device_ptr(&stream);
            let cp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&c_dev)?;
            match &self.c_tmp {
                Some(tmp) => {
                    let (tp, _gt) = tmp.device_ptr(&stream);
                    unsafe {
                        nv_kernels::cuda::marlin_gemm_w4a16_ex(
                            stream.cu_stream() as *mut _,
                            xp as *const std::ffi::c_void,
                            bp as *const std::ffi::c_void,
                            sp as *const std::ffi::c_void,
                            cp as *mut std::ffi::c_void,
                            tp as *mut std::ffi::c_void,
                            wp as *mut std::ffi::c_void,
                            m as i32,
                            self.out_features as i32,
                            self.in_features as i32,
                            self.group_size as i32,
                            1,
                            0,
                            1,
                        )
                    }
                }
                None => unsafe {
                    nv_kernels::cuda::marlin_gemm_w4a16(
                        stream.cu_stream() as *mut _,
                        xp as *const std::ffi::c_void,
                        bp as *const std::ffi::c_void,
                        sp as *const std::ffi::c_void,
                        cp as *mut std::ffi::c_void,
                        wp as *mut std::ffi::c_void,
                        m as i32,
                        self.out_features as i32,
                        self.in_features as i32,
                        self.group_size as i32,
                        1,
                    )
                },
            }
        };
        ensure!(rc == 0, "marlin_gemm returned {rc}");
        let out = c_dev.reshape((m, self.out_features))?;
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(self.out_features);
        out.reshape(out_dims).map_err(Into::into)
    }

    fn forward_a8(
        &self,
        x2: &Tensor,
        m: usize,
        dims: &[usize],
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<Tensor> {
        use cudarc::driver::DevicePtr;
        use half::bf16;
        let (a_q, a_s) = marlin_a8_input_scratch(m * self.in_features, m, stream)?;
        let c_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
            &Device::Cuda(self.device.clone()),
            DType::BF16,
            &[m * self.out_features],
        )?;
        let rc = {
            let (xs, _xl) = x2.storage_and_layout();
            let (ss, _sl) = self.b_scales.storage_and_layout();
            let xc = match &*xs {
                Storage::Cuda(s) => s,
                _ => bail!("marlin a8: x not cuda"),
            };
            let sc = match &*ss {
                Storage::Cuda(s) => s,
                _ => bail!("marlin a8: scales not cuda"),
            };
            let xsl = xc.as_cuda_slice::<bf16>()?;
            let ssl = sc.as_cuda_slice::<bf16>()?;
            let (xp, _gx) = xsl.device_ptr(stream);
            let (sp, _gs) = ssl.device_ptr(stream);
            let (bp, _gb) = self.b_q_marlin.device_ptr(stream);
            let (wp, _gw) = self.workspace.device_ptr(stream);
            let (qp, _gq) = a_q.device_ptr(stream);
            let (ap, _ga) = a_s.device_ptr(stream);
            let cp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&c_dev)?;
            let rq = unsafe {
                nv_kernels::cuda::rowquant_e4m3(
                    stream.cu_stream() as *mut _,
                    xp as *const u16,
                    qp as *mut u8,
                    ap as *mut f32,
                    m as i32,
                    self.in_features as i32,
                )
            };
            ensure!(rq == 0, "rowquant_e4m3 returned {rq}");
            unsafe {
                nv_kernels::cuda::marlin_gemm_w4a8_fp8(
                    stream.cu_stream() as *mut _,
                    qp as *const std::ffi::c_void,
                    ap as *const std::ffi::c_void,
                    bp as *const std::ffi::c_void,
                    sp as *const std::ffi::c_void,
                    cp as *mut std::ffi::c_void,
                    wp as *mut std::ffi::c_void,
                    m as i32,
                    self.out_features as i32,
                    self.in_features as i32,
                    self.group_size as i32,
                    0,
                )
            }
        };
        ensure!(rc == 0, "marlin_gemm_w4a8_fp8 returned {rc}");
        let out = c_dev.reshape((m, self.out_features))?;
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(self.out_features);
        out.reshape(out_dims).map_err(Into::into)
    }
}

fn load_norm(weights: &WeightLoader, suffix: &str, eps: f64) -> Result<RmsNorm> {
    Ok(RmsNorm::new(
        load_tensor(weights, &format!("{suffix}.weight"), DType::F32)?,
        eps,
    ))
}

fn build_rope(
    head_dim: usize,
    base: f32,
    partial: f32,
    max_seq: usize,
    device: &Device,
) -> Result<Rope> {
    let half = head_dim / 2;
    let rope_angles = ((partial * head_dim as f32 / 2.0) as usize).min(half);
    let mut inv_freq = vec![0f32; half];
    for (i, f) in inv_freq[..rope_angles].iter_mut().enumerate() {
        *f = 1.0 / base.powf((i as f32 * 2.0) / (head_dim as f32));
    }
    Rope::from_inv_freq(
        RopeConfig {
            head_dim,
            max_seq_len: max_seq,
            base,
            kind: RopeKind::Standard,
        },
        &inv_freq,
        device,
    )
}

struct E4bLayer {
    kind: LayerType,
    head_dim: usize,
    n_kv_heads: usize,
    kv_source: Option<usize>,

    input_ln: RmsNorm,
    post_attn_ln: RmsNorm,
    pre_ff_ln: RmsNorm,
    post_ff_ln: RmsNorm,

    qkv_proj: QLinear,
    o_proj: QLinear,
    q_norm: RmsNorm,
    k_norm: Option<RmsNorm>,

    gate_up_proj: QLinear,
    down_proj: QLinear,

    per_layer_input_gate: QLinear,
    per_layer_projection: QLinear,
    post_per_layer_input_norm: RmsNorm,
    layer_scalar: f32,
}

pub struct Gemma4E4b {
    config: Gemma4Config,
    dtype: DType,
    device: Device,

    embed_tokens: Tensor,
    #[cfg(feature = "cuda")]
    lm_head_i8: Option<(CudaSlice<i8>, CudaSlice<f32>)>,
    embed_tokens_per_layer: Tensor,
    per_layer_model_projection: QLinear,
    per_layer_projection_norm: RmsNorm,
    layers: Vec<E4bLayer>,
    norm: RmsNorm,
    lm_head: Linear,

    sliding_rope: Rope,
    full_rope: Rope,

    normalizer: f64,
    embed_scale_per_layer: f64,
    per_layer_proj_scale: f64,
    per_layer_input_scale: f64,
    #[cfg(feature = "cuda")]
    prefill_prof: Option<E4bPrefillProf>,
}

pub const NV_E4B_PREFILL_PROF_ENV_OPT_IN_1_STAMPS_DEVICE_TIMESTAMPS_AT_EVERY_PREFILL_PHASE_BOUNDARY_SO_THE_CHUNK_WALL_ATTRIBUTES_TO_KERNEL_FAMILIES: &str = "NV_E4B_PREFILL_PROF";
#[cfg(feature = "cuda")]
const E4B_PREFILL_PROF_SLOTS_PER_LAYER_4_QKV_ATTN_MLP_PLI: usize = 4;
const E4B_PREFILL_PROF_FAMILIES_7: usize = 7;
const E4B_PREFILL_PROF_FAMILY_NAMES: [&str; E4B_PREFILL_PROF_FAMILIES_7] = [
    "embed_pli",
    "qkv_rope_cache",
    "attn_sliding",
    "attn_full",
    "mlp",
    "pli_contrib",
    "lm_head",
];

#[cfg(feature = "cuda")]
pub struct E4bPrefillProf {
    buf: CudaSlice<u64>,
    dev: CudaDevice,
    n_layers: usize,
    kinds: Vec<LayerType>,
    acc: std::sync::Mutex<E4bPrefillProfAcc>,
}

#[derive(Default, Clone)]
pub struct E4bPrefillProfAcc {
    pub family_ms: [f64; E4B_PREFILL_PROF_FAMILIES_7],
    pub chunks: usize,
    pub per_chunk_attn_ms: Vec<(f64, f64)>,
}

#[cfg(feature = "cuda")]
impl E4bPrefillProf {
    fn from_env(device: &Device, kinds: Vec<LayerType>) -> Result<Option<Self>> {
        if !nv_config::env_arm_opt_in_eq_some_1_reread_every_call(
            NV_E4B_PREFILL_PROF_ENV_OPT_IN_1_STAMPS_DEVICE_TIMESTAMPS_AT_EVERY_PREFILL_PHASE_BOUNDARY_SO_THE_CHUNK_WALL_ATTRIBUTES_TO_KERNEL_FAMILIES,
        ) {
            return Ok(None);
        }
        let Device::Cuda(dev) = device else {
            return Ok(None);
        };
        let n_layers = kinds.len();
        let slots = 3 + E4B_PREFILL_PROF_SLOTS_PER_LAYER_4_QKV_ATTN_MLP_PLI * n_layers;
        let stream = nv_layers::cuda_stream::current_stream(dev);
        let buf = stream
            .alloc_zeros::<u64>(slots)
            .map_err(|e| anyhow::anyhow!("e4b prefill prof buf: {e:?}"))?;
        Ok(Some(Self {
            buf,
            dev: dev.clone(),
            n_layers,
            kinds,
            acc: std::sync::Mutex::new(E4bPrefillProfAcc::default()),
        }))
    }

    fn stamp(&self, slot: usize) -> Result<()> {
        let stream = nv_layers::cuda_stream::current_stream(&self.dev);
        crate::laguna_step_graph::prof_timestamp_at(&self.buf, slot, &stream)
    }

    fn slot_qkv(&self, li: usize) -> usize {
        2 + E4B_PREFILL_PROF_SLOTS_PER_LAYER_4_QKV_ATTN_MLP_PLI * li
    }

    fn slot_end(&self) -> usize {
        2 + E4B_PREFILL_PROF_SLOTS_PER_LAYER_4_QKV_ATTN_MLP_PLI * self.n_layers
    }

    fn end_chunk(&self) -> Result<()> {
        let stream = nv_layers::cuda_stream::current_stream(&self.dev);
        let end = self.slot_end();
        self.stamp(end)?;
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("e4b prefill prof sync: {e:?}"))?;
        let ts: Vec<u64> = stream
            .clone_dtoh(&self.buf)
            .map_err(|e| anyhow::anyhow!("e4b prefill prof dtoh: {e:?}"))?;
        let ms = |a: usize, b: usize| (ts[b].saturating_sub(ts[a])) as f64 / 1e6;
        let mut acc = self.acc.lock().unwrap_or_else(|e| e.into_inner());
        acc.family_ms[0] += ms(0, 1);
        let mut prev = 1usize;
        let (mut sl, mut fu) = (0f64, 0f64);
        for li in 0..self.n_layers {
            let q = self.slot_qkv(li);
            acc.family_ms[1] += ms(prev, q);
            let a = ms(q, q + 1);
            match self.kinds[li] {
                LayerType::SlidingAttention => {
                    acc.family_ms[2] += a;
                    sl += a;
                }
                LayerType::FullAttention => {
                    acc.family_ms[3] += a;
                    fu += a;
                }
            }
            acc.family_ms[4] += ms(q + 1, q + 2);
            acc.family_ms[5] += ms(q + 2, q + 3);
            prev = q + 3;
        }
        acc.family_ms[6] += ms(prev, end);
        acc.chunks += 1;
        acc.per_chunk_attn_ms.push((sl, fu));
        Ok(())
    }

    pub fn take(&self) -> E4bPrefillProfAcc {
        let mut acc = self.acc.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *acc)
    }
}

impl E4bPrefillProfAcc {
    pub fn report_line(&self, tag: &str) -> String {
        let total: f64 = self.family_ms.iter().sum();
        let mut out = format!(
            "E4B-PREFILL-ATTRIB {tag} chunks={} stamped_ms={total:.1}",
            self.chunks
        );
        for (name, ms) in E4B_PREFILL_PROF_FAMILY_NAMES
            .iter()
            .zip(self.family_ms.iter())
        {
            out.push_str(&format!(
                " {name}={ms:.1}({:.1}%)",
                if total > 0.0 { 100.0 * ms / total } else { 0.0 }
            ));
        }
        if let (Some(first), Some(last)) = (
            self.per_chunk_attn_ms.first(),
            self.per_chunk_attn_ms.last(),
        ) {
            out.push_str(&format!(
                " attn_sliding_first_chunk_ms={:.2} attn_sliding_last_chunk_ms={:.2} attn_full_first_chunk_ms={:.2} attn_full_last_chunk_ms={:.2}",
                first.0, last.0, first.1, last.1
            ));
        }
        out
    }
}

pub struct E4bTrace {
    pub inputs_embeds: Tensor,
    pub per_layer_inputs: Tensor,
    pub hidden_after_l0: Tensor,
    pub hidden_after_l1: Tensor,
    pub hidden_last_layer: Tensor,
    pub logits_last: Vec<f32>,
}

impl Gemma4E4b {
    pub fn config(&self) -> &Gemma4Config {
        &self.config
    }

    pub fn from_loader(
        config: Gemma4Config,
        weights: &WeightLoader,
        device: &Device,
    ) -> Result<Self> {
        ensure!(
            config.has_per_layer_embeddings(),
            "gemma4_e4b requires hidden_size_per_layer_input > 0 (got {})",
            config.hidden_size_per_layer_input
        );

        let dtype = match device {
            Device::Cpu => DType::F32,
            _ => DType::BF16,
        };
        let eps = config.rms_norm_eps;
        let hpl = config.hidden_size_per_layer_input;

        let embed_tokens = load_tensor(weights, "embed_tokens.weight", dtype)?;

        #[cfg(feature = "cuda")]
        let lm_head_i8 =
            if nv_config::env_arm_present_at_all_reread_every_call("NV_E4B_LMHEAD_INT8") && device.is_cuda() {
                Some(nv_layers::lm_head_resident_quant::rowquant_i8_dev(
                    &embed_tokens,
                )?)
            } else {
                None
            };
        let embed_tokens_per_layer = load_tensor(weights, "embed_tokens_per_layer.weight", dtype)?;
        let per_layer_model_projection =
            load_linear(weights, "per_layer_model_projection", dtype, device)?;
        let per_layer_projection_norm = load_norm(weights, "per_layer_projection_norm", eps)?;
        let norm = load_norm(weights, "norm", eps)?;
        let lm_head = Linear::new(embed_tokens.clone(), None)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let kind = config.layer_kind(i);
            let head_dim = config.head_dim_for(kind);
            let n_kv_heads = config.num_kv_heads_for(kind);
            let kv_source = config.kv_source_layer(i);
            let p = format!("layers.{i}");

            let (qkv_proj, k_norm) = if kv_source.is_some() {
                (
                    load_linear(weights, &format!("{p}.self_attn.q_proj"), dtype, device)?,
                    None,
                )
            } else {
                (
                    load_linear_merged(
                        weights,
                        &[
                            &format!("{p}.self_attn.q_proj"),
                            &format!("{p}.self_attn.k_proj"),
                            &format!("{p}.self_attn.v_proj"),
                        ],
                        dtype,
                        device,
                    )?,
                    Some(load_norm(weights, &format!("{p}.self_attn.k_norm"), eps)?),
                )
            };

            let scalar = load_tensor(weights, &format!("{p}.layer_scalar"), DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let layer_scalar = *scalar.first().context("empty layer_scalar")?;

            layers.push(E4bLayer {
                kind,
                head_dim,
                n_kv_heads,
                kv_source,
                input_ln: load_norm(weights, &format!("{p}.input_layernorm"), eps)?,
                post_attn_ln: load_norm(weights, &format!("{p}.post_attention_layernorm"), eps)?,
                pre_ff_ln: load_norm(weights, &format!("{p}.pre_feedforward_layernorm"), eps)?,
                post_ff_ln: load_norm(weights, &format!("{p}.post_feedforward_layernorm"), eps)?,
                qkv_proj,
                o_proj: load_linear(weights, &format!("{p}.self_attn.o_proj"), dtype, device)?,
                q_norm: load_norm(weights, &format!("{p}.self_attn.q_norm"), eps)?,
                k_norm,
                gate_up_proj: load_linear_merged(
                    weights,
                    &[&format!("{p}.mlp.gate_proj"), &format!("{p}.mlp.up_proj")],
                    dtype,
                    device,
                )?,
                down_proj: load_linear(weights, &format!("{p}.mlp.down_proj"), dtype, device)?,
                per_layer_input_gate: load_linear(
                    weights,
                    &format!("{p}.per_layer_input_gate"),
                    dtype,
                    device,
                )?,
                per_layer_projection: load_linear(
                    weights,
                    &format!("{p}.per_layer_projection"),
                    dtype,
                    device,
                )?,
                post_per_layer_input_norm: load_norm(
                    weights,
                    &format!("{p}.post_per_layer_input_norm"),
                    eps,
                )?,
                layer_scalar,
            });
        }

        let sliding_rope = build_rope(
            config.head_dim,
            config.rope_theta_for(LayerType::SlidingAttention),
            config.rope_partial_factor_for(LayerType::SlidingAttention),
            config.max_position_embeddings,
            device,
        )?;
        let full_rope = build_rope(
            config.global_head_dim,
            config.rope_theta_for(LayerType::FullAttention),
            config.rope_partial_factor_for(LayerType::FullAttention),
            config.max_position_embeddings,
            device,
        )?;

        let hidden = config.hidden_size as f64;
        let model = Self {
            config,
            dtype,
            device: device.clone(),
            embed_tokens,
            #[cfg(feature = "cuda")]
            lm_head_i8,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            norm,
            lm_head,
            sliding_rope,
            full_rope,
            normalizer: hidden.sqrt(),
            embed_scale_per_layer: (hpl as f64).sqrt(),
            per_layer_proj_scale: hidden.powf(-0.5),
            per_layer_input_scale: 2f64.powf(-0.5),
            #[cfg(feature = "cuda")]
            prefill_prof: None,
        };
        #[cfg(feature = "cuda")]
        let model = {
            let kinds: Vec<LayerType> = model.layers.iter().map(|l| l.kind).collect();
            let mut model = model;
            model.prefill_prof = E4bPrefillProf::from_env(device, kinds)?;
            model
        };
        if device.is_cuda() && boot_warm_prefill_env_opted_in() {
            let warm =
                vec![0u32; E4B_BOOT_WARM_CHUNK_512_MATCHES_THE_SERVING_PREFILL_CHUNK_SO_THE_THROWAWAY_PASS_COMPILES_EXACTLY_THE_MODULES_THE_FIRST_REAL_CHUNK_NEEDS];
            let mut scratch: Vec<Option<(Tensor, Tensor)>> =
                vec![None; model.config.num_hidden_layers];
            let _ = model.forward_step(&warm, 0, &mut scratch)?;
            drop(scratch);
            model.device.synchronize()?;
        }
        Ok(model)
    }

    fn per_layer_inputs(&self, ids: &Tensor, inputs_embeds: &Tensor) -> Result<Tensor> {
        let seq = ids.dims1()?;
        let n_layers = self.config.num_hidden_layers;
        let hpl = self.config.hidden_size_per_layer_input;

        let ple = self
            .embed_tokens_per_layer
            .index_select(ids, 0)?
            .to_dtype(DType::F32)?
            .reshape((seq, n_layers, hpl))?;
        let ple = (ple * self.embed_scale_per_layer)?;

        let proj = self
            .per_layer_model_projection
            .forward(inputs_embeds)?
            .to_dtype(DType::F32)?;
        let proj = (proj * self.per_layer_proj_scale)?.reshape((seq, n_layers, hpl))?;
        let proj = self.per_layer_projection_norm.forward(&proj)?;

        ((proj + ple)? * self.per_layer_input_scale).map_err(Into::into)
    }

    fn attention(
        &self,
        layer: &E4bLayer,
        normed: &Tensor,
        positions: &Tensor,
        stash: &mut [Option<(Tensor, Tensor)>],
        idx: usize,
    ) -> Result<Tensor> {
        let seq = normed.dim(0)?;
        let n_heads = self.config.num_attention_heads;
        let hd = layer.head_dim;
        let rope = match layer.kind {
            LayerType::SlidingAttention => &self.sliding_rope,
            LayerType::FullAttention => &self.full_rope,
        };

        let nkv = layer.n_kv_heads;
        let qkv = layer.qkv_proj.forward(normed)?;
        let q = qkv
            .narrow(D::Minus1, 0, n_heads * hd)?
            .contiguous()?
            .reshape((1, seq, n_heads, hd))?;
        let q = layer.q_norm.forward(&q)?.to_dtype(DType::F32)?;

        let (k_rot, v) = match &layer.k_norm {
            Some(kn) => {
                let k = qkv
                    .narrow(D::Minus1, n_heads * hd, nkv * hd)?
                    .contiguous()?
                    .reshape((1, seq, nkv, hd))?;
                let k = kn.forward(&k)?.to_dtype(DType::F32)?;
                let v = qkv
                    .narrow(D::Minus1, (n_heads + nkv) * hd, nkv * hd)?
                    .contiguous()?
                    .reshape((1, seq, nkv, hd))?;

                let v = rms_no_weight(&v.to_dtype(DType::F32)?, self.config.rms_norm_eps)?;
                let (_q, k_rot) = rope.apply(&q, &k, positions)?;
                stash[idx] = Some((k_rot.clone(), v.clone()));
                (k_rot, v)
            }
            None => {
                let src = layer.kv_source.context("kv-shared layer without source")?;
                stash[src].clone().context("kv source not computed yet")?
            }
        };
        let (q_rot, _k) = rope.apply(&q, &k_rot, positions)?;

        let q = q_rot.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        let group = n_heads / layer.n_kv_heads;
        let k = repeat_kv(&k_rot.squeeze(0)?.transpose(0, 1)?, group)?;
        let v = repeat_kv(&v.squeeze(0)?.transpose(0, 1)?, group)?;

        let scores = q.matmul(&k.transpose(1, 2)?.contiguous()?)?;
        let scores = scores.broadcast_add(&causal_mask(seq, &self.device)?)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = probs.matmul(&v.contiguous()?)?;
        let out = out
            .transpose(0, 1)?
            .reshape((1, seq, n_heads * hd))?
            .to_dtype(self.dtype)?;
        layer.o_proj.forward(&out)
    }

    pub fn trace(&self, input_ids: &[u32]) -> Result<E4bTrace> {
        let seq = input_ids.len();
        let ids = Tensor::from_vec(input_ids.to_vec(), seq, &self.device)?;
        let positions =
            Tensor::from_vec((0..seq as u32).collect::<Vec<_>>(), (1, seq), &self.device)?;

        let inputs_embeds = self.embed_tokens.index_select(&ids, 0)?;
        let inputs_embeds = (inputs_embeds.to_dtype(DType::F32)? * self.normalizer)?
            .to_dtype(self.dtype)?
            .reshape((1, seq, self.config.hidden_size))?;

        let per_layer_inputs = self.per_layer_inputs(&ids, &inputs_embeds)?;

        let mut x = inputs_embeds.clone();
        let mut stash: Vec<Option<(Tensor, Tensor)>> = vec![None; self.layers.len()];
        let mut after_l0 = None;
        let mut after_l1 = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let pli = per_layer_inputs
                .narrow(1, i, 1)?
                .reshape((seq, self.config.hidden_size_per_layer_input))?;
            x = self.layer_forward(layer, &x, &positions, &pli, &mut stash, i)?;
            if i == 0 {
                after_l0 = Some(x.clone());
            } else if i == 1 {
                after_l1 = Some(x.clone());
            }
        }
        let hidden_last = x.clone();
        let normed = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&normed)?.to_dtype(DType::F32)?;
        let logits = softcap(&logits, self.config.final_logit_softcapping)?;
        let last = logits
            .narrow(1, seq - 1, 1)?
            .reshape(self.config.vocab_size)?;

        Ok(E4bTrace {
            inputs_embeds: inputs_embeds.squeeze(0)?,
            per_layer_inputs,
            hidden_after_l0: after_l0.unwrap().squeeze(0)?,
            hidden_after_l1: after_l1.unwrap().squeeze(0)?,
            hidden_last_layer: hidden_last.squeeze(0)?,
            logits_last: last.to_vec1::<f32>()?,
        })
    }

    fn layer_forward(
        &self,
        layer: &E4bLayer,
        x: &Tensor,
        positions: &Tensor,
        per_layer_input: &Tensor,
        stash: &mut [Option<(Tensor, Tensor)>],
        idx: usize,
    ) -> Result<Tensor> {
        let seq = x.dim(1)?;
        let normed = layer
            .input_ln
            .forward(&x.reshape((seq, self.config.hidden_size))?)?;
        let attn = self.attention(layer, &normed, positions, stash, idx)?;
        let attn = layer
            .post_attn_ln
            .forward(&attn.reshape((seq, self.config.hidden_size))?)?;
        let h = (x.reshape((seq, self.config.hidden_size))? + attn)?;

        let normed2 = layer.pre_ff_ln.forward(&h)?;
        let gu = layer.gate_up_proj.forward(&normed2)?;
        let inter = gu.dim(D::Minus1)? / 2;
        let gate = gu.narrow(D::Minus1, 0, inter)?.gelu()?;
        let up = gu.narrow(D::Minus1, inter, inter)?;
        let mlp = layer.down_proj.forward(&(gate * up)?)?;
        let mlp = layer.post_ff_ln.forward(&mlp)?;
        let h = (h + mlp)?;

        let gate = layer.per_layer_input_gate.forward(&h)?.gelu()?;
        let gated = (gate.to_dtype(DType::F32)? * per_layer_input)?.to_dtype(self.dtype)?;
        let contrib = layer.per_layer_projection.forward(&gated)?;
        let contrib = layer.post_per_layer_input_norm.forward(&contrib)?;
        let h = (h + contrib)?;

        let h = (h.to_dtype(DType::F32)? * layer.layer_scalar as f64)?.to_dtype(self.dtype)?;
        h.reshape((1, seq, self.config.hidden_size))
            .map_err(Into::into)
    }

    pub fn generate(&self, prompt: &[u32], max_new: usize, eos: &[u32]) -> Result<Vec<u32>> {
        ensure!(!prompt.is_empty(), "empty prompt");
        #[cfg(feature = "cuda")]
        if self.device.is_cuda() && e4b_kv_fp8_enabled() {
            return self.generate_fp8(prompt, max_new, eos);
        }
        let mut cache: Vec<Option<(Tensor, Tensor)>> = vec![None; self.layers.len()];
        let mut logits = self.forward_step(prompt, 0, &mut cache)?;
        let mut out = Vec::new();
        let start = prompt.len();
        for total in start..start + max_new {
            let next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            out.push(next);
            if eos.contains(&next) {
                break;
            }
            logits = self.forward_step(&[next], total, &mut cache)?;
        }
        Ok(out)
    }

    pub fn forward_step(
        &self,
        new_ids: &[u32],
        past_len: usize,
        cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(p) = &self.prefill_prof {
            p.stamp(0)?;
        }
        let mut kv = E4bKvMut::Cat(cache);
        let (logits, _hidden) =
            self.forward_step_inner(new_ids, None, past_len, &mut kv, PrefillLogitsRows::LastRow)?;
        let logits = logits.context("forward_step asked for logits")?;
        logits
            .narrow(0, logits.dim(0)? - 1, 1)?
            .reshape(self.config.vocab_size)
            .map_err(Into::into)
    }

    pub fn forward_step_embeds(
        &self,
        new_ids: &[u32],
        embeds: &Tensor,
        past_len: usize,
        cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        let mut kv = E4bKvMut::Cat(cache);
        let (logits, _hidden) = self.forward_step_inner(
            new_ids,
            Some(embeds),
            past_len,
            &mut kv,
            PrefillLogitsRows::LastRow,
        )?;
        let logits = logits.context("forward_step_embeds asked for logits")?;
        logits
            .narrow(0, logits.dim(0)? - 1, 1)?
            .reshape(self.config.vocab_size)
            .map_err(Into::into)
    }

    pub fn prefill_fixed_cache(&self, capacity: usize) -> Result<E4bPrefillFixedCache> {
        ensure!(capacity > 0, "prefill_fixed_cache: zero capacity");
        let mut k = Vec::with_capacity(self.layers.len());
        let mut v = Vec::with_capacity(self.layers.len());
        let mut k16 = Vec::with_capacity(self.layers.len());
        let mut v16 = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            if layer.k_norm.is_some() {
                let shape = (1usize, capacity, layer.n_kv_heads, layer.head_dim);
                k.push(Some(Tensor::zeros(shape, DType::F32, &self.device)?));
                v.push(Some(Tensor::zeros(shape, DType::F32, &self.device)?));
                if matches!(layer.kind, LayerType::FullAttention) {
                    let twin = (layer.n_kv_heads, capacity, layer.head_dim);
                    k16.push(Some(Tensor::zeros(twin, DType::BF16, &self.device)?));
                    v16.push(Some(Tensor::zeros(twin, DType::BF16, &self.device)?));
                } else {
                    k16.push(None);
                    v16.push(None);
                }
            } else {
                k.push(None);
                v.push(None);
                k16.push(None);
                v16.push(None);
            }
        }
        Ok(E4bPrefillFixedCache {
            k,
            v,
            k_bf16_twin_kvh_major_for_the_gemm_arm: k16,
            v_bf16_twin_kvh_major_for_the_gemm_arm: v16,
            capacity,
            len: 0,
        })
    }

    pub fn forward_step_fixed(
        &self,
        new_ids: &[u32],
        embeds_override: Option<&Tensor>,
        past_len: usize,
        cache: &mut E4bPrefillFixedCache,
    ) -> Result<Tensor> {
        self.forward_step_fixed_rows(
            new_ids,
            embeds_override,
            past_len,
            cache,
            PrefillLogitsRows::LastRow,
        )
        .and_then(|l| l.context("forward_step_fixed asked for logits"))
    }

    pub fn forward_step_fixed_rows(
        &self,
        new_ids: &[u32],
        embeds_override: Option<&Tensor>,
        past_len: usize,
        cache: &mut E4bPrefillFixedCache,
        rows: PrefillLogitsRows,
    ) -> Result<Option<Tensor>> {
        let n = new_ids.len();
        cache.begin_step(past_len, n)?;
        #[cfg(feature = "cuda")]
        if let Some(p) = &self.prefill_prof {
            p.stamp(0)?;
        }
        let logits = {
            let mut kv = E4bKvMut::FixedPrefill(cache);
            let (logits, _hidden) =
                self.forward_step_inner(new_ids, embeds_override, past_len, &mut kv, rows)?;
            logits
        };
        cache.len = past_len + n;
        logits
            .map(|l| {
                let last = l.dim(0)? - 1;
                l.narrow(0, last, 1)?
                    .reshape(self.config.vocab_size)
                    .map_err(Into::into)
            })
            .transpose()
    }

    #[cfg(feature = "cuda")]
    pub fn forward_step_fp8(
        &self,
        new_ids: &[u32],
        past_len: usize,
        cache: &mut E4bFp8Cache,
    ) -> Result<Tensor> {
        let n = new_ids.len();
        cache.begin_step(past_len, n)?;
        let mut kv = E4bKvMut::Fp8(cache);
        let (logits, _hidden) =
            self.forward_step_inner(new_ids, None, past_len, &mut kv, PrefillLogitsRows::LastRow)?;
        let logits = logits.context("forward_step_fp8 asked for logits")?;
        logits
            .narrow(0, logits.dim(0)? - 1, 1)?
            .reshape(self.config.vocab_size)
            .map_err(Into::into)
    }

    #[cfg(feature = "cuda")]
    pub fn generate_fp8(&self, prompt: &[u32], max_new: usize, eos: &[u32]) -> Result<Vec<u32>> {
        ensure!(!prompt.is_empty(), "empty prompt");
        let cap = (prompt.len() + max_new + 8).next_power_of_two().max(64);
        let mut cache = E4bFp8Cache::new(&self.config, cap, &self.device)?;
        let mut logits = self.forward_step_fp8(prompt, 0, &mut cache)?;
        let mut out = Vec::new();
        let start = prompt.len();
        for total in start..start + max_new {
            let next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            out.push(next);
            if eos.contains(&next) {
                break;
            }
            logits = self.forward_step_fp8(&[next], total, &mut cache)?;
        }
        Ok(out)
    }

    fn forward_step_inner(
        &self,
        new_ids: &[u32],
        embeds_override: Option<&Tensor>,
        past_len: usize,
        kv: &mut E4bKvMut,
        rows: PrefillLogitsRows,
    ) -> Result<(Option<Tensor>, Tensor)> {
        let n = new_ids.len();
        let hidden = self.config.hidden_size;
        #[cfg(feature = "cuda")]
        if let Device::Cuda(d) = &self.device {
            let stream = nv_layers::cuda_stream::current_stream(d);
            marlin_zero_upload(&stream)?;
            marlin_zero_launch(&stream)?;
        }
        let ids = Tensor::from_vec(new_ids.to_vec(), n, &self.device)?;
        let embeds = match embeds_override {
            Some(e) => {
                ensure!(
                    e.dims() == [n, hidden],
                    "embeds override must be [{n}, {hidden}], got {:?}",
                    e.dims()
                );
                e.to_dtype(self.dtype)?.reshape((1, n, hidden))?
            }
            None => {
                let embeds = self.embed_tokens.index_select(&ids, 0)?;
                (embeds.to_dtype(DType::F32)? * self.normalizer)?
                    .to_dtype(self.dtype)?
                    .reshape((1, n, hidden))?
            }
        };
        let pli_all = self.per_layer_inputs(&ids, &embeds)?;
        #[cfg(feature = "cuda")]
        if let Some(p) = &self.prefill_prof {
            p.stamp(1)?;
        }
        let pos: Vec<u32> = (past_len..past_len + n).map(|x| x as u32).collect();

        let total = past_len + n;
        let full_mask = cache_mask(n, total, past_len, None, total, &self.device)?;
        let sliding_kept = total.min(self.config.sliding_window + n);
        let sliding_mask = cache_mask(
            n,
            total,
            past_len,
            Some(self.config.sliding_window),
            sliding_kept,
            &self.device,
        )?;
        let mut x = embeds;
        for (i, layer) in self.layers.iter().enumerate() {
            let pli = pli_all
                .narrow(1, i, 1)?
                .reshape((n, self.config.hidden_size_per_layer_input))?;
            let mask = match layer.kind {
                LayerType::SlidingAttention => &sliding_mask,
                LayerType::FullAttention => &full_mask,
            };
            x = self.layer_step(layer, &x, &pos, &pli, kv, i, mask)?;
        }
        let hidden_pre = x.reshape((n, hidden))?;
        let logits = match rows {
            PrefillLogitsRows::Discarded => None,
            PrefillLogitsRows::LastRow => {
                let xn = self.norm.forward(&x.narrow(1, n - 1, 1)?)?;
                let logits = self
                    .lm_head
                    .forward(&xn)?
                    .reshape((1, self.config.vocab_size))?;
                Some(self.step_logits_softcap_f32_capped_serving_default(logits)?)
            }
        };
        #[cfg(feature = "cuda")]
        if let Some(p) = &self.prefill_prof {
            p.end_chunk()?;
        }
        Ok((logits, hidden_pre))
    }

    fn step_logits_softcap_f32_capped_serving_default(&self, logits: Tensor) -> Result<Tensor> {
        let cap = self.config.final_logit_softcapping;
        if !(cap > 0.0 && cap.is_finite()) {
            return Ok(logits);
        }
        #[cfg(feature = "cuda")]
        if self.device.is_cuda() && logits.dtype() == DType::BF16 {
            return crate::gemma4::tanh_softcap_bf16_to_f32_op(&logits, cap, &self.device);
        }
        softcap(&logits.to_dtype(DType::F32)?, cap)
    }

    #[cfg(feature = "cuda")]
    pub fn prefill_prof_take(&self) -> Option<E4bPrefillProfAcc> {
        self.prefill_prof.as_ref().map(|p| p.take())
    }

    #[cfg(feature = "cuda")]
    pub fn prefill_prof_read_one_step_without_accumulating(&self) -> Option<E4bPrefillProfAcc> {
        let p = self.prefill_prof.as_ref()?;
        let stream = nv_layers::cuda_stream::current_stream(&p.dev);
        stream.synchronize().ok()?;
        let ts: Vec<u64> = stream.clone_dtoh(&p.buf).ok()?;
        let ms = |a: usize, b: usize| (ts[b].saturating_sub(ts[a])) as f64 / 1e6;
        let end = p.slot_end();
        let mut acc = E4bPrefillProfAcc::default();
        let mut prev = p.slot_qkv(0);
        let (mut sl, mut fu) = (0f64, 0f64);
        for li in 0..p.n_layers {
            let q = p.slot_qkv(li);
            if li > 0 {
                acc.family_ms[1] += ms(prev, q);
            }
            let a = ms(q, q + 1);
            match p.kinds[li] {
                LayerType::SlidingAttention => {
                    acc.family_ms[2] += a;
                    sl += a;
                }
                LayerType::FullAttention => {
                    acc.family_ms[3] += a;
                    fu += a;
                }
            }
            acc.family_ms[4] += ms(q + 1, q + 2);
            acc.family_ms[5] += ms(q + 2, q + 3);
            prev = q + 3;
        }
        acc.family_ms[6] += ms(prev, end);
        acc.chunks = 1;
        acc.per_chunk_attn_ms.push((sl, fu));
        Some(acc)
    }

    pub fn embed_table(&self) -> &Tensor {
        &self.embed_tokens
    }

    pub fn embed_normalizer(&self) -> f64 {
        self.normalizer
    }

    fn layer_step(
        &self,
        layer: &E4bLayer,
        x: &Tensor,
        pos: &[u32],
        per_layer_input: &Tensor,
        kv: &mut E4bKvMut,
        idx: usize,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let n = x.dim(1)?;
        let hidden = self.config.hidden_size;
        let h0 = x.reshape((n, hidden))?;
        let normed = layer.input_ln.forward(&h0)?;
        let attn = self.attn_cached(layer, &normed, pos, kv, idx, mask)?;
        #[cfg(feature = "cuda")]
        if let Some(p) = &self.prefill_prof {
            p.stamp(p.slot_qkv(idx) + 1)?;
        }
        let attn = layer.post_attn_ln.forward(&attn.reshape((n, hidden))?)?;
        let h = add_scale_op(&h0, &attn, 1.0)?;

        let normed2 = layer.pre_ff_ln.forward(&h)?;
        let act = geglu_fused_op(&layer.gate_up_proj.forward(&normed2)?)?;
        let mlp = layer.post_ff_ln.forward(&layer.down_proj.forward(&act)?)?;
        let h = add_scale_op(&h, &mlp, 1.0)?;
        #[cfg(feature = "cuda")]
        if let Some(p) = &self.prefill_prof {
            p.stamp(p.slot_qkv(idx) + 2)?;
        }

        let gate = layer.per_layer_input_gate.forward(&h)?.gelu()?;
        let gated = (gate.to_dtype(DType::F32)? * per_layer_input)?.to_dtype(self.dtype)?;
        let contrib = layer.per_layer_projection.forward(&gated)?;
        let contrib = layer.post_per_layer_input_norm.forward(&contrib)?;

        let h = add_scale_op(&h, &contrib, layer.layer_scalar as f64)?;
        #[cfg(feature = "cuda")]
        if let Some(p) = &self.prefill_prof {
            p.stamp(p.slot_qkv(idx) + 3)?;
        }

        h.reshape((1, n, hidden)).map_err(Into::into)
    }

    fn attn_cached(
        &self,
        layer: &E4bLayer,
        normed: &Tensor,
        pos: &[u32],
        kv: &mut E4bKvMut,
        idx: usize,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let n = normed.dim(0)?;
        let nh = self.config.num_attention_heads;
        let hd = layer.head_dim;
        let rope = match layer.kind {
            LayerType::SlidingAttention => &self.sliding_rope,
            LayerType::FullAttention => &self.full_rope,
        };
        let pos_t = Tensor::from_vec(pos.to_vec(), (1, n), &self.device)?;
        let nkv = layer.n_kv_heads;
        let qkv = layer.qkv_proj.forward(normed)?;
        #[cfg(feature = "cuda")]
        let mut twin: Option<(Tensor, Tensor)> = None;
        #[cfg(feature = "cuda")]
        let fused_prep = match kv {
            E4bKvMut::FixedPrefill(cache)
                if n > 1 && self.device.is_cuda() =>
            {
                let pos_i32 = Tensor::from_vec(pos.to_vec(), n, &self.device)?;
                let q_rot = cache.prep_and_append_one_kernel(
                    idx,
                    &qkv.reshape((n, (nh + 2 * nkv).min(qkv.dim(D::Minus1)? / hd) * hd))?,
                    layer.q_norm.weight_bf16(),
                    layer.k_norm.as_ref().map(|k| k.weight_bf16()),
                    rope,
                    &pos_i32,
                    nh,
                    nkv,
                    hd,
                    self.config.rms_norm_eps as f32,
                )?;
                let total = *pos.last().context("attn_cached: empty positions")? as usize + 1;
                let src = if layer.k_norm.is_some() {
                    idx
                } else {
                    layer.kv_source.context("kv-shared layer without source")?
                };
                let (kk, vv) = cache.view(src, total)?;
                twin = cache.bf16_twin_views_for_the_gemm_arm(src, total);
                Some((kk, vv, q_rot))
            }
            _ => None,
        };
        #[cfg(not(feature = "cuda"))]
        let fused_prep: Option<(Tensor, Tensor, Tensor)> = None;
        let (k_all, v_all, q_rot) = if let Some(t) = fused_prep {
            t
        } else {
            let q = qkv
                .narrow(D::Minus1, 0, nh * hd)?
                .contiguous()?
                .reshape((1, n, nh, hd))?;
            let q = layer.q_norm.forward(&q)?.to_dtype(DType::F32)?;
            match &layer.k_norm {
                Some(kn) => {
                    let k = qkv
                        .narrow(D::Minus1, nh * hd, nkv * hd)?
                        .contiguous()?
                        .reshape((1, n, nkv, hd))?;
                    let k = kn.forward(&k)?.to_dtype(DType::F32)?;
                    let v = rms_no_weight(
                        &qkv.narrow(D::Minus1, (nh + nkv) * hd, nkv * hd)?
                            .contiguous()?
                            .reshape((1, n, nkv, hd))?
                            .to_dtype(DType::F32)?,
                        self.config.rms_norm_eps,
                    )?;
                    let (q_rot, k_rot) = rope.apply(&q, &k, &pos_t)?;
                    match kv {
                        E4bKvMut::Cat(cache) => {
                            let (nk, nv) = match cache[idx].take() {
                                Some((ok, ov)) => {
                                    (Tensor::cat(&[&ok, &k_rot], 1)?, Tensor::cat(&[&ov, &v], 1)?)
                                }
                                None => (k_rot, v),
                            };
                            cache[idx] = Some((nk.clone(), nv.clone()));
                            (nk, nv, q_rot)
                        }
                        E4bKvMut::FixedPrefill(cache) => {
                            let (kk, vv) = cache.append(idx, &k_rot, &v)?;
                            #[cfg(feature = "cuda")]
                            {
                                twin = cache.bf16_twin_views_for_the_gemm_arm(idx, kk.dim(1)?);
                            }
                            (kk, vv, q_rot)
                        }
                        #[cfg(feature = "cuda")]
                        E4bKvMut::Fp8(cache) => {
                            let total =
                                *pos.last().context("attn_cached: empty positions")? as usize + 1;
                            cache.append(idx, &k_rot, &v)?;
                            let (kk, vv) = cache.view_f32(idx, total)?;
                            (kk, vv, q_rot)
                        }
                    }
                }
                None => {
                    let src = layer.kv_source.context("kv-shared layer without source")?;
                    let (q_rot, _) = rope.apply(&q, &q, &pos_t)?;
                    let (k, v) = match kv {
                        E4bKvMut::Cat(cache) => {
                            cache[src].clone().context("kv source not in cache")?
                        }
                        E4bKvMut::FixedPrefill(cache) => {
                            let total =
                                *pos.last().context("attn_cached: empty positions")? as usize + 1;
                            #[cfg(feature = "cuda")]
                            {
                                twin = cache.bf16_twin_views_for_the_gemm_arm(src, total);
                            }
                            cache.view(src, total)?
                        }
                        #[cfg(feature = "cuda")]
                        E4bKvMut::Fp8(cache) => {
                            let total =
                                *pos.last().context("attn_cached: empty positions")? as usize + 1;
                            cache.view_f32(src, total)?
                        }
                    };
                    (k, v, q_rot)
                }
            }
        };

        #[cfg(feature = "cuda")]
        if let Some(p) = &self.prefill_prof {
            p.stamp(p.slot_qkv(idx))?;
        }
        #[cfg(feature = "cuda")]
        if n == 1 && self.device.is_cuda() {
            let total = k_all.dim(1)?;
            let start = match layer.kind {
                LayerType::SlidingAttention => total.saturating_sub(self.config.sliding_window),
                LayerType::FullAttention => 0,
            };
            let qf = q_rot.reshape((nh, hd))?.contiguous()?;
            let kf = k_all.reshape((total, layer.n_kv_heads, hd))?.contiguous()?;
            let vf = v_all.reshape((total, layer.n_kv_heads, hd))?.contiguous()?;
            let out = attn_decode_f32_op(&qf, &kf, &vf, nh, layer.n_kv_heads, hd, total, start)?;
            let out = out.reshape((1, n, nh * hd))?.to_dtype(self.dtype)?;
            return layer.o_proj.forward(&out);
        }

        let group = nh / layer.n_kv_heads;
        let kept_matches_the_mask_width_so_slice_and_shift_agree_by_construction = mask.dim(1)?;
        let k_len = k_all.dim(1)?;
        let (k_all, v_all) =
            if kept_matches_the_mask_width_so_slice_and_shift_agree_by_construction < k_len {
                let kept = kept_matches_the_mask_width_so_slice_and_shift_agree_by_construction;
                (
                    k_all.narrow(1, k_len - kept, kept)?,
                    v_all.narrow(1, k_len - kept, kept)?,
                )
            } else {
                (k_all, v_all)
            };
        #[cfg(feature = "cuda")]
        if n > 1
            && self.device.is_cuda()
            && matches!(layer.kind, LayerType::FullAttention)
            && k_all.dim(1)? > E4B_FULL_GEMM_ATTN_FROM_THE_SECOND_KEY_TILE_4096_BECAUSE_INSIDE_ONE_TILE_FA2_WINS_THE_2K_CELL_BY_15_MS
        {
            let qg = q_rot
                .squeeze(0)?
                .transpose(0, 1)?
                .to_dtype(DType::BF16)?
                .contiguous()?
                .reshape((layer.n_kv_heads, group * n, hd))?;
            let (kg, vg) = match twin.take() {
                Some((k16, v16)) => (k16, v16),
                None => (
                    k_all
                        .squeeze(0)?
                        .transpose(0, 1)?
                        .to_dtype(DType::BF16)?
                        .contiguous()?,
                    v_all
                        .squeeze(0)?
                        .transpose(0, 1)?
                        .to_dtype(DType::BF16)?
                        .contiguous()?,
                ),
            };
            let out = nv_layers::attn::e4b_full_attn_gqa_folded_f32_scores_tiled_online_softmax(
                &qg, &kg, &vg, n, group, 1.0,
            )?
            .reshape((1, n, nh * hd))?
            .to_dtype(self.dtype)?;
            return layer.o_proj.forward(&out);
        }
        #[cfg(feature = "cuda")]
        if n > 1 && self.device.is_cuda() {
            let cfg = nv_layers::attn::AttnConfig {
                num_heads: nh,
                num_kv_heads: layer.n_kv_heads,
                head_dim: hd,
                softmax_scale: 1.0,
                causal: true,
            };
            let window = match layer.kind {
                LayerType::SlidingAttention => Some(self.config.sliding_window.saturating_sub(1)),
                LayerType::FullAttention => None,
            };
            let qh = q_rot.to_dtype(DType::F16)?.contiguous()?;
            let kh = k_all.to_dtype(DType::F16)?.contiguous()?;
            let vh = v_all.to_dtype(DType::F16)?.contiguous()?;
            let out = nv_layers::attn::flash_attn_windowed(&qh, &kh, &vh, &cfg, window, Some(0))?
                .reshape((1, n, nh * hd))?;
            let out = out.to_dtype(self.dtype)?;
            return layer.o_proj.forward(&out);
        }
        let q = q_rot.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        let k = repeat_kv(&k_all.squeeze(0)?.transpose(0, 1)?, group)?;
        let v = repeat_kv(&v_all.squeeze(0)?.transpose(0, 1)?, group)?;
        let scores = q.matmul(&k.transpose(1, 2)?.contiguous()?)?;
        let scores = scores.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = probs.matmul(&v.contiguous()?)?;
        let out = out
            .transpose(0, 1)?
            .reshape((1, n, nh * hd))?
            .to_dtype(self.dtype)?;
        layer.o_proj.forward(&out)
    }

    #[cfg(feature = "cuda")]
    pub fn generate_graphed(
        &self,
        prompt: &[u32],
        max_new: usize,
        eos: &[u32],
    ) -> Result<Vec<u32>> {
        let graph_batch: usize = std::env::var("NV_E4B_GRAPH_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v| (1..=32).contains(&v))
            .unwrap_or(16);
        ensure!(!prompt.is_empty(), "empty prompt");
        let Device::Cuda(dev) = self.device.clone() else {
            bail!("generate_graphed requires cuda")
        };
        let mut cat: Vec<Option<(Tensor, Tensor)>> = vec![None; self.layers.len()];
        let logits = self.forward_step(prompt, 0, &mut cat)?;

        let max_len = (prompt.len() + max_new + graph_batch + 8)
            .next_power_of_two()
            .max(64);
        let fixed = E4bFixedCache::from_cat(&cat, prompt.len(), max_len, &dev)?;
        let mut dec = GraphedE4bDecoder::new(self, fixed, prompt.len())?;
        let mut next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
        let mut out = vec![next];
        if out.len() >= max_new || eos.contains(&next) {
            return Ok(out);
        }
        next = dec.warm_step(next)?;
        out.push(next);
        'decode: while out.len() < max_new && !eos.contains(out.last().unwrap()) {
            let k = graph_batch.min(max_new - out.len());
            let toks = dec.replay_batch(*out.last().unwrap(), k)?;
            for t in toks {
                out.push(t);
                if out.len() >= max_new || eos.contains(&t) {
                    break 'decode;
                }
            }
        }
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    pub fn graphed_decoder(
        &self,
        cat: &[Option<(Tensor, Tensor)>],
        past_len: usize,
        max_len: usize,
    ) -> Result<GraphedE4bDecoder<'_>> {
        let Device::Cuda(dev) = self.device.clone() else {
            bail!("graphed_decoder requires cuda")
        };
        let fixed = E4bFixedCache::from_cat(cat, past_len, max_len, &dev)?;
        GraphedE4bDecoder::new(self, fixed, past_len)
    }

    #[cfg(feature = "cuda")]
    pub fn generate_fast(&self, prompt: &[u32], max_new: usize, eos: &[u32]) -> Result<Vec<u32>> {
        ensure!(!prompt.is_empty(), "empty prompt");
        let Device::Cuda(dev) = self.device.clone() else {
            bail!("generate_fast requires cuda")
        };
        let mut cat: Vec<Option<(Tensor, Tensor)>> = vec![None; self.layers.len()];
        let mut logits = self.forward_step(prompt, 0, &mut cat)?;
        let max_len = (prompt.len() + max_new + 8).next_power_of_two().max(64);
        let mut fixed = E4bFixedCache::from_cat(&cat, prompt.len(), max_len, &dev)?;
        let mut out = Vec::new();
        for total in (prompt.len()..).take(max_new) {
            let next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            out.push(next);
            if eos.contains(&next) {
                break;
            }
            logits = self.forward_step_fast(next, total, &mut fixed)?;
        }
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn forward_step_fast(
        &self,
        token: u32,
        cur_len: usize,
        fixed: &mut E4bFixedCache,
    ) -> Result<Tensor> {
        let ids = Tensor::from_vec(vec![token], 1, &self.device)?;
        let pos_t = Tensor::from_vec(vec![cur_len as i32], 1, &self.device)?;
        if let Device::Cuda(d) = &self.device {
            marlin_zero_upload(&nv_layers::cuda_stream::current_stream(d))?;
        }
        let logits = self.forward_step_fast_body(&ids, &pos_t, fixed)?;
        logits
            .narrow(1, 0, 1)?
            .reshape(self.config.vocab_size)
            .map_err(Into::into)
    }

    #[cfg(feature = "cuda")]
    fn forward_step_fast_body(
        &self,
        ids: &Tensor,
        pos_t: &Tensor,
        fixed: &mut E4bFixedCache,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let hidden = self.config.hidden_size;
        if let Device::Cuda(d) = &self.device {
            marlin_zero_launch(&nv_layers::cuda_stream::current_stream(d))?;
        }
        if let Some(p) = &self.prefill_prof {
            p.stamp(0)?;
        }
        let emb = gather_rows_bf16_op(&self.embed_tokens, ids)?;
        let embeds = scale_bf16_op(&emb, self.normalizer as f32)?.reshape((1, 1, hidden))?;
        let pli_all = self.per_layer_inputs_fast(ids, &embeds)?;
        if let Some(p) = &self.prefill_prof {
            p.stamp(1)?;
        }
        let Device::Cuda(dev) = self.device.clone() else {
            bail!("cuda only")
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);

        ensure!(pos_t.dtype() == DType::I32, "pos_t must be i32");
        let rc = {
            let pos_ptr = {
                let (p, _g) = fixed.pos.device_ptr_mut(&stream);
                p as *mut i32
            };
            let (ps, _pl) = pos_t.storage_and_layout();
            let pc = match &*ps {
                Storage::Cuda(s) => s,
                _ => bail!("pos_t not cuda"),
            };
            let psl = pc.as_cuda_slice::<i32>()?;
            let (rp, _gr) = psl.device_ptr(&stream);
            unsafe {
                nv_kernels::cuda::incr_pos_rope(
                    stream.cu_stream() as *mut _,
                    pos_ptr,
                    rp as *mut i32,
                )
            }
        };
        ensure!(rc == 0, "incr_pos_rope returned {rc}");
        let mut x = embeds.clone();
        let mut rstd = rstd_op(&embeds, self.config.rms_norm_eps as f32)?;
        let mut normed: Option<Tensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let pli = pli_all.narrow(1, i, 1)?;
            let next_w = self.layers.get(i + 1).map(|l| l.input_ln.weight_bf16());
            let (nx, nrstd, nnormed) = self.layer_step_fast(
                layer,
                &x,
                i,
                pos_t,
                &pli,
                &rstd,
                normed.as_ref(),
                next_w,
                fixed,
            )?;
            x = nx;
            rstd = nrstd;
            normed = nnormed;
        }

        let logits = match &self.lm_head_i8 {
            Some((wq, rs)) => nv_layers::lm_head_resident_quant::lm_head_i8_normed_op(
                wq,
                rs,
                &x,
                self.norm.weight_bf16(),
                &rstd,
                &dev,
            ),
            None => lm_head_normed_op(&self.embed_tokens, &x, self.norm.weight_bf16(), &rstd),
        }?;
        if let Some(p) = &self.prefill_prof {
            p.stamp(p.slot_end())?;
        }
        Ok(logits)
    }

    #[cfg(feature = "cuda")]
    fn per_layer_inputs_fast(&self, ids: &Tensor, inputs_embeds: &Tensor) -> Result<Tensor> {
        let n_layers = self.config.num_hidden_layers;
        let hpl = self.config.hidden_size_per_layer_input;

        let ple_bf = gather_rows_bf16_op(&self.embed_tokens_per_layer, ids)?;
        let ple = cast_scale_f32_op(&ple_bf, self.embed_scale_per_layer as f32)?;

        let proj_bf = self.per_layer_model_projection.forward(inputs_embeds)?;
        let proj = cast_scale_f32_op(
            &proj_bf.reshape((1, n_layers * hpl))?,
            self.per_layer_proj_scale as f32,
        )?;
        let proj = self
            .per_layer_projection_norm
            .forward(&proj.reshape((n_layers, hpl))?)?;

        add_scale_f32_op(
            &proj.reshape((1, n_layers, hpl))?,
            &ple.reshape((1, n_layers, hpl))?,
            self.per_layer_input_scale as f32,
        )
    }

    #[cfg(feature = "cuda")]
    fn layer_step_fast(
        &self,
        layer: &E4bLayer,
        x: &Tensor,
        idx: usize,
        pos_t: &Tensor,
        per_layer_input: &Tensor,
        rstd_in: &Tensor,
        normed_in: Option<&Tensor>,
        next_input_w: Option<&Tensor>,
        fixed: &mut E4bFixedCache,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let hidden = self.config.hidden_size;
        let nh = self.config.num_attention_heads;
        let hd = layer.head_dim;
        let nkv = layer.n_kv_heads;
        let eps = self.config.rms_norm_eps as f32;
        let h0 = x.reshape((1, hidden))?;
        if let Some(p) = &self.prefill_prof {
            p.stamp(p.slot_qkv(idx))?;
        }

        let normed = match normed_in {
            Some(n) => n.reshape((1, hidden))?,
            None => rms_apply_op(&h0, layer.input_ln.weight_bf16(), rstd_in)?,
        };

        let rope = match layer.kind {
            LayerType::SlidingAttention => &self.sliding_rope,
            LayerType::FullAttention => &self.full_rope,
        };
        let window = match layer.kind {
            LayerType::SlidingAttention => self.config.sliding_window,
            LayerType::FullAttention => 0,
        };

        let qkv = layer.qkv_proj.forward(&normed)?;
        match &layer.k_norm {
            Some(kn) => {
                let qf = fixed.qkv_prep(
                    Some(idx),
                    &qkv,
                    0,
                    layer.q_norm.weight_bf16(),
                    Some(kn.weight_bf16()),
                    rope,
                    pos_t,
                    0,
                    nh,
                    nkv,
                    hd,
                    eps,
                )?;
                let attn = fixed.attend(idx, &qf, 0, nh, nkv, hd, window)?;
                self.attn_tail(
                    layer,
                    &h0,
                    &attn,
                    hidden,
                    nh,
                    hd,
                    per_layer_input,
                    next_input_w,
                    idx,
                )
            }
            None => {
                let src = layer.kv_source.context("kv-shared layer without source")?;
                let qf = fixed.qkv_prep(
                    None,
                    &qkv,
                    0,
                    layer.q_norm.weight_bf16(),
                    None,
                    rope,
                    pos_t,
                    0,
                    nh,
                    nkv,
                    hd,
                    eps,
                )?;
                let attn = fixed.attend(src, &qf, 0, nh, nkv, hd, window)?;
                self.attn_tail(
                    layer,
                    &h0,
                    &attn,
                    hidden,
                    nh,
                    hd,
                    per_layer_input,
                    next_input_w,
                    idx,
                )
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn attn_tail(
        &self,
        layer: &E4bLayer,
        h0: &Tensor,
        attn: &Tensor,
        hidden: usize,
        nh: usize,
        hd: usize,
        per_layer_input: &Tensor,
        next_input_w: Option<&Tensor>,
        idx: usize,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let eps = self.config.rms_norm_eps as f32;

        let out = attn.reshape((1, 1, nh * hd))?;
        let fold = e4b_m1_gemv_folds_opt_in_until_the_served_stream_identity_proves_them();
        let folded_o = match &layer.o_proj {
            QLinear::Marlin(m) if fold => m.forward_m1_then_tail_in_the_last_block(
                &out,
                MarlinM1Tail::RmsnormAddScale {
                    norm: &layer.post_attn_ln,
                    res: h0,
                    scale: 1.0,
                    chain_eps: Some(eps),
                    next_w: Some(layer.pre_ff_ln.weight_bf16()),
                },
            )?,
            _ => None,
        };
        let (h, _rstd_h1, normed2) = match folded_o {
            Some(t) => t,
            None => {
                let attn = layer.o_proj.forward(&out)?;
                rmsnorm_add_scale_chain_op(
                    &attn.reshape((1, hidden))?,
                    &layer.post_attn_ln,
                    h0,
                    1.0,
                    Some(eps),
                    Some(layer.pre_ff_ln.weight_bf16()),
                )?
            }
        };
        let normed2 = normed2.context("missing fused pre_ff apply")?;
        if let Some(p) = &self.prefill_prof {
            p.stamp(p.slot_qkv(idx) + 1)?;
        }

        let act = match &layer.gate_up_proj {
            QLinear::GemvM1(g) if fold => {
                g.forward_geglu_pairing_gate_row_n_with_up_row_n_plus_inter(&normed2)?
            }
            gate_up => geglu_fused_op(&gate_up.forward(&normed2)?)?,
        };
        let h = match &layer.down_proj {
            QLinear::GemvM1(d) if fold => {
                d.forward_then_rmsnorm_add_scale_in_the_last_block(
                    &act,
                    &layer.post_ff_ln,
                    &h,
                    1.0,
                    None,
                    None,
                )?
                .0
            }
            down_proj => {
                let down = down_proj.forward(&act)?;
                rmsnorm_add_scale_op(&down.reshape((1, hidden))?, &layer.post_ff_ln, &h, 1.0)?
            }
        };
        if let Some(p) = &self.prefill_prof {
            p.stamp(p.slot_qkv(idx) + 2)?;
        }

        let folded_gate = match &layer.per_layer_input_gate {
            QLinear::Marlin(m) if fold => m.forward_m1_then_tail_in_the_last_block(
                &h,
                MarlinM1Tail::GeluPli {
                    pli: per_layer_input,
                },
            )?,
            _ => None,
        };
        let gated = match folded_gate {
            Some((g, _, _)) => g,
            None => match layer
                .per_layer_input_gate
                .forward_gelu_pli(&h, per_layer_input)
            {
                Some(r) => r?,
                None => gelu_mul(&layer.per_layer_input_gate.forward(&h)?, per_layer_input)?,
            },
        };
        let folded_proj = match &layer.per_layer_projection {
            QLinear::Marlin(m) if fold => m.forward_m1_then_tail_in_the_last_block(
                &gated,
                MarlinM1Tail::RmsnormAddScale {
                    norm: &layer.post_per_layer_input_norm,
                    res: &h,
                    scale: layer.layer_scalar,
                    chain_eps: Some(eps),
                    next_w: next_input_w,
                },
            )?,
            _ => None,
        };
        let (h, rstd_next, normed_next) = match folded_proj {
            Some(t) => t,
            None => {
                let contrib = layer.per_layer_projection.forward(&gated)?;
                rmsnorm_add_scale_chain_op(
                    &contrib.reshape((1, hidden))?,
                    &layer.post_per_layer_input_norm,
                    &h,
                    layer.layer_scalar,
                    Some(eps),
                    next_input_w,
                )?
            }
        };
        let rstd_next = rstd_next.context("missing chained rstd")?;
        if let Some(p) = &self.prefill_prof {
            p.stamp(p.slot_qkv(idx) + 3)?;
        }
        Ok((h.reshape((1, 1, hidden))?, rstd_next, normed_next))
    }
}

#[cfg(feature = "cuda")]
pub fn e4b_kv_fp8_enabled() -> bool {
    nv_config::env_arm_opt_in_set_to_anything_but_0_reread_every_call("NV_E4B_KV_FP8")
}

enum E4bKvMut<'a> {
    Cat(&'a mut [Option<(Tensor, Tensor)>]),
    FixedPrefill(&'a mut E4bPrefillFixedCache),
    #[cfg(feature = "cuda")]
    Fp8(&'a mut E4bFp8Cache),
}

pub struct E4bPrefillFixedCache {
    k: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
    k_bf16_twin_kvh_major_for_the_gemm_arm: Vec<Option<Tensor>>,
    v_bf16_twin_kvh_major_for_the_gemm_arm: Vec<Option<Tensor>>,
    capacity: usize,
    len: usize,
}

impl E4bPrefillFixedCache {
    pub fn seq_len(&self) -> usize {
        self.len
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn begin_step(&self, past_len: usize, n: usize) -> Result<()> {
        ensure!(
            past_len == self.len,
            "E4bPrefillFixedCache: step at past_len {past_len} but cache holds {} tokens",
            self.len
        );
        ensure!(
            past_len + n <= self.capacity,
            "E4bPrefillFixedCache: step to {} exceeds capacity {}",
            past_len + n,
            self.capacity
        );
        Ok(())
    }

    fn append(&mut self, idx: usize, k_rot: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let n = k_rot.dim(1)?;
        let kb = self.k[idx]
            .as_ref()
            .context("append into a kv-shared layer slot")?;
        let vb = self.v[idx]
            .as_ref()
            .context("append into a kv-shared layer slot")?;
        kb.slice_set(&k_rot.contiguous()?, 1, self.len)?;
        vb.slice_set(&v.contiguous()?, 1, self.len)?;
        if let (Some(k16), Some(v16)) = (
            &self.k_bf16_twin_kvh_major_for_the_gemm_arm[idx],
            &self.v_bf16_twin_kvh_major_for_the_gemm_arm[idx],
        ) {
            let kt = k_rot
                .squeeze(0)?
                .transpose(0, 1)?
                .to_dtype(DType::BF16)?
                .contiguous()?;
            let vt = v
                .squeeze(0)?
                .transpose(0, 1)?
                .to_dtype(DType::BF16)?
                .contiguous()?;
            k16.slice_set(&kt, 1, self.len)?;
            v16.slice_set(&vt, 1, self.len)?;
        }
        Ok((
            kb.narrow(1, 0, self.len + n)?,
            vb.narrow(1, 0, self.len + n)?,
        ))
    }

    #[cfg(feature = "cuda")]
    fn prep_and_append_one_kernel(
        &mut self,
        idx: usize,
        qkv: &Tensor,
        q_norm_w: &Tensor,
        k_norm_w: Option<&Tensor>,
        rope: &Rope,
        positions: &Tensor,
        nh: usize,
        nkv: usize,
        hd: usize,
        eps: f32,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let n = qkv.dim(0)?;
        let Device::Cuda(dev) = qkv.device().clone() else {
            bail!("prep_and_append_one_kernel requires cuda")
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let qkv = qkv.contiguous()?;
        let row_stride = qkv.dim(1)?;
        let mut q_dev: CudaSlice<f32> = unsafe { stream.alloc::<f32>(n * nh * hd) }?;
        let cuda_ptr = |t: &Tensor| -> Result<u64> {
            let (st, layout) = t.storage_and_layout();
            match &*st {
                Storage::Cuda(c) => {
                    let off = layout.start_offset();
                    Ok(match t.dtype() {
                        DType::BF16 => {
                            let sl = c.as_cuda_slice::<half::bf16>()?;
                            let (p, _g) = sl.device_ptr(&stream);
                            p + (off * 2) as u64
                        }
                        DType::F32 => {
                            let sl = c.as_cuda_slice::<f32>()?;
                            let (p, _g) = sl.device_ptr(&stream);
                            p + (off * 4) as u64
                        }
                        DType::U32 => {
                            let sl = c.as_cuda_slice::<u32>()?;
                            let (p, _g) = sl.device_ptr(&stream);
                            p + (off * 4) as u64
                        }
                        other => {
                            bail!("prep_and_append_one_kernel: unexpected dtype {other:?}")
                        }
                    })
                }
                _ => bail!("prep_and_append_one_kernel: tensor not on cuda"),
            }
        };
        let qkv_p = cuda_ptr(&qkv)?;
        let qw_p = cuda_ptr(q_norm_w)?;
        let kw_p = match k_norm_w {
            Some(w) => cuda_ptr(w)?,
            None => 0,
        };
        let cos_p = cuda_ptr(rope.cos())?;
        let sin_p = cuda_ptr(rope.sin())?;
        let pos_p = cuda_ptr(positions)?;
        let (kc_p, vc_p, kt_p, vt_p) = if k_norm_w.is_some() {
            let kb = self.k[idx].as_ref().context("prep: no k buffer")?;
            let vb = self.v[idx].as_ref().context("prep: no v buffer")?;
            let kt = self.k_bf16_twin_kvh_major_for_the_gemm_arm[idx].as_ref();
            let vt = self.v_bf16_twin_kvh_major_for_the_gemm_arm[idx].as_ref();
            (
                cuda_ptr(kb)?,
                cuda_ptr(vb)?,
                kt.map(cuda_ptr).transpose()?.unwrap_or(0),
                vt.map(cuda_ptr).transpose()?.unwrap_or(0),
            )
        } else {
            (0, 0, 0, 0)
        };
        let rc = {
            let (qp, _g) = q_dev.device_ptr_mut(&stream);
            unsafe {
                nv_kernels::cuda::e4b_prefill_qkv_prep(
                    stream.cu_stream() as *mut _,
                    qkv_p as *const u16,
                    row_stride as i32,
                    qw_p as *const u16,
                    kw_p as *const u16,
                    cos_p as *const f32,
                    sin_p as *const f32,
                    pos_p as *const i32,
                    qp as *mut f32,
                    kc_p as *mut f32,
                    vc_p as *mut f32,
                    kt_p as *mut u16,
                    vt_p as *mut u16,
                    n as i32,
                    self.len as i32,
                    self.capacity as i32,
                    nh as i32,
                    nkv as i32,
                    hd as i32,
                    eps,
                )
            }
        };
        ensure!(rc == 0, "e4b_prefill_qkv_prep returned {rc}");
        let storage = CudaStorage::wrap_cuda_slice(q_dev, dev);
        let storage = Storage::Cuda(storage);
        Ok(Tensor::from_storage(
            storage,
            (1, n, nh, hd),
            BackpropOp::none(),
            false,
        ))
    }

    #[cfg(feature = "cuda")]
    fn bf16_twin_views_for_the_gemm_arm(
        &self,
        idx: usize,
        total: usize,
    ) -> Option<(Tensor, Tensor)> {
        let k16 = self.k_bf16_twin_kvh_major_for_the_gemm_arm[idx].as_ref()?;
        let v16 = self.v_bf16_twin_kvh_major_for_the_gemm_arm[idx].as_ref()?;
        Some((k16.narrow(1, 0, total).ok()?, v16.narrow(1, 0, total).ok()?))
    }

    fn view(&self, idx: usize, total: usize) -> Result<(Tensor, Tensor)> {
        let kb = self.k[idx].as_ref().context("kv source not in cache")?;
        let vb = self.v[idx].as_ref().context("kv source not in cache")?;
        Ok((kb.narrow(1, 0, total)?, vb.narrow(1, 0, total)?))
    }

    pub fn cat_views(&self) -> Result<Vec<Option<(Tensor, Tensor)>>> {
        let mut out = Vec::with_capacity(self.k.len());
        for (kb, vb) in self.k.iter().zip(self.v.iter()) {
            out.push(match (kb, vb) {
                (Some(kb), Some(vb)) => {
                    Some((kb.narrow(1, 0, self.len)?, vb.narrow(1, 0, self.len)?))
                }
                _ => None,
            });
        }
        Ok(out)
    }
}

#[cfg(feature = "cuda")]
pub struct E4bFp8Cache {
    inner: crate::paged_fp8::PagedGemma4Cache,
    capacity: usize,
}

#[cfg(feature = "cuda")]
impl E4bFp8Cache {
    pub fn new(config: &Gemma4Config, max_seq_len: usize, device: &Device) -> Result<Self> {
        use std::sync::{Arc, Mutex};
        let block_size = 16usize;
        let blocks = max_seq_len.div_ceil(block_size).max(1);
        let cfg = crate::paged_fp8::PagedPoolConfig::from_gemma4_e4b(config, blocks, block_size);
        let pool = crate::paged_fp8::PagedKvFp8Pool::new(cfg, device)?;
        let mut inner =
            crate::paged_fp8::PagedGemma4Cache::new(Arc::new(Mutex::new(pool)), device)?;
        let table: Vec<u32> = (0..blocks as u32).collect();
        inner.set_block_table(&table)?;
        Ok(Self {
            inner,
            capacity: blocks * block_size,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn begin_step(&mut self, past_len: usize, n: usize) -> Result<()> {
        use crate::gemma4::Gemma4Cache;
        ensure!(
            past_len + n <= self.capacity,
            "E4bFp8Cache: step to {} exceeds capacity {}",
            past_len + n,
            self.capacity
        );
        self.inner.prepare_for_decode(past_len, past_len + n)
    }

    fn append(&mut self, layer: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        use crate::gemma4::Gemma4Cache;
        self.inner
            .write_at(layer, &cast_to_bf16(k)?, &cast_to_bf16(v)?)
    }

    fn view_f32(&mut self, layer: usize, len: usize) -> Result<(Tensor, Tensor)> {
        use crate::gemma4::Gemma4Cache;
        let (k, v) = self.inner.view(layer, len)?;
        Ok((cast_to_f32(&k)?, cast_to_f32(&v)?))
    }
}

fn cache_mask(
    n: usize,
    total: usize,
    past: usize,
    window: Option<usize>,
    kept: usize,
    device: &Device,
) -> Result<Tensor> {
    let dropped_columns_are_all_masked_because_kept_ge_window_plus_chunk = total - kept;
    let mut data = vec![0f32; n * kept];
    for r in 0..n {
        let absr = past + r;
        for j in 0..kept {
            let c = dropped_columns_are_all_masked_because_kept_ge_window_plus_chunk + j;
            let mut masked = c > absr;
            if let Some(w) = window {
                if absr + 1 > w && c + w <= absr {
                    masked = true;
                }
            }
            if masked {
                data[r * kept + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (n, kept), device).map_err(Into::into)
}

fn rms_no_weight(x: &Tensor, eps: f64) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let ms = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let denom = (ms + eps)?.sqrt()?;
    xf.broadcast_div(&denom).map_err(Into::into)
}

fn add_scale_op(a: &Tensor, b: &Tensor, scale: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if a.device().is_cuda() && a.dtype() == DType::BF16 {
        return cuda_binary_bf16(a, b, |stream, pa, pb, py, n| unsafe {
            nv_kernels::cuda::residual_add_scale_bf16(stream, pa, pb, py, scale as f32, n)
        });
    }
    let s = (a + b)?;
    (s.to_dtype(DType::F32)? * scale)?
        .to_dtype(a.dtype())
        .map_err(Into::into)
}

#[allow(dead_code)]
fn gelu_mul_op(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if gate.device().is_cuda() && gate.dtype() == DType::BF16 {
        return cuda_binary_bf16(gate, up, |stream, pg, pu, py, n| unsafe {
            nv_kernels::cuda::gelu_tanh_mul_bf16(stream, pg, pu, py, n)
        });
    }
    (gate.gelu()? * up).map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn cuda_binary_bf16<F>(a: &Tensor, b: &Tensor, f: F) -> Result<Tensor>
where
    F: Fn(*mut std::ffi::c_void, *const u16, *const u16, *mut u16, usize) -> i32,
{
    use cudarc::driver::DevicePtr;
    use half::bf16;
    let Device::Cuda(dev) = a.device().clone() else {
        bail!("cuda_binary_bf16 requires cuda")
    };
    let a_c = a.contiguous()?;
    let b_c = b.contiguous()?;
    let dims = a_c.dims().to_vec();
    ensure!(
        dims == b_c.dims(),
        "shape mismatch {:?} vs {:?}",
        dims,
        b_c.dims()
    );
    let n: usize = dims.iter().product();
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let y_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
        &Device::Cuda(dev.clone()),
        DType::BF16,
        &[n],
    )?;
    let rc = {
        let (a_st, _al) = a_c.storage_and_layout();
        let (b_st, _bl) = b_c.storage_and_layout();
        let ac = match &*a_st {
            Storage::Cuda(s) => s,
            _ => bail!("a not cuda"),
        };
        let bc = match &*b_st {
            Storage::Cuda(s) => s,
            _ => bail!("b not cuda"),
        };
        let asl = ac.as_cuda_slice::<bf16>()?;
        let bsl = bc.as_cuda_slice::<bf16>()?;
        let (pa, _ga) = asl.device_ptr(&stream);
        let (pb, _gb) = bsl.device_ptr(&stream);
        let py = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&y_dev)?;
        f(
            stream.cu_stream() as *mut _,
            pa as *const u16,
            pb as *const u16,
            py as *mut u16,
            n,
        )
    };
    ensure!(rc == 0, "elementwise bf16 kernel returned {rc}");
    y_dev
        .reshape(dims)?
        .reshape(a_c.shape())
        .map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn attn_decode_f32_op(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    nh: usize,
    nkv: usize,
    hd: usize,
    total: usize,
    start: usize,
) -> Result<Tensor> {
    use cudarc::driver::DevicePtr;
    let Device::Cuda(dev) = q.device().clone() else {
        bail!("attn_decode_f32_op requires cuda")
    };
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let out_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
        &Device::Cuda(dev.clone()),
        DType::F32,
        &[nh * hd],
    )?;
    let rc = {
        let (qs, _ql) = q.storage_and_layout();
        let (ks, _kl) = k.storage_and_layout();
        let (vs, _vl) = v.storage_and_layout();
        let qc = match &*qs {
            Storage::Cuda(s) => s,
            _ => bail!("q not cuda"),
        };
        let kc = match &*ks {
            Storage::Cuda(s) => s,
            _ => bail!("k not cuda"),
        };
        let vc = match &*vs {
            Storage::Cuda(s) => s,
            _ => bail!("v not cuda"),
        };
        let qsl = qc.as_cuda_slice::<f32>()?;
        let ksl = kc.as_cuda_slice::<f32>()?;
        let vsl = vc.as_cuda_slice::<f32>()?;
        let (qp, _gq) = qsl.device_ptr(&stream);
        let (kp, _gk) = ksl.device_ptr(&stream);
        let (vp, _gv) = vsl.device_ptr(&stream);
        let op = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&out_dev)?;
        unsafe {
            nv_kernels::cuda::attn_decode_f32(
                stream.cu_stream() as *mut _,
                qp as *const f32,
                kp as *const f32,
                vp as *const f32,
                op as *mut f32,
                nh as i32,
                nkv as i32,
                hd as i32,
                total as i32,
                start as i32,
            )
        }
    };
    ensure!(rc == 0, "attn_decode_f32 returned {rc}");
    out_dev.reshape((nh, hd)).map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn cast_to_f32(x: &Tensor) -> Result<Tensor> {
    use cudarc::driver::DevicePtr;
    let Device::Cuda(dev) = x.device().clone() else {
        bail!("cast_to_f32 requires cuda")
    };
    ensure!(x.dtype() == DType::BF16, "cast_to_f32 expects bf16");
    let x_c = x.contiguous()?;
    let dims = x_c.dims().to_vec();
    let n: usize = dims.iter().product();
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let y_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
        &Device::Cuda(dev.clone()),
        DType::F32,
        &[n],
    )?;
    let rc = {
        let (xs, _xl) = x_c.storage_and_layout();
        let xc = match &*xs {
            Storage::Cuda(s) => s,
            _ => bail!("cast_to_f32: x not cuda"),
        };
        let xsl = xc.as_cuda_slice::<half::bf16>()?;
        let (xp, _gx) = xsl.device_ptr(&stream);
        let yp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&y_dev)?;
        unsafe {
            nv_kernels::cuda::cast_bf16_f32(
                stream.cu_stream() as *mut _,
                xp as *const u16,
                yp as *mut f32,
                n as i32,
            )
        }
    };
    ensure!(rc == 0, "cast_bf16_f32 returned {rc}");
    y_dev
        .reshape(dims)?
        .reshape(x_c.shape())
        .map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn cast_to_bf16(x: &Tensor) -> Result<Tensor> {
    use cudarc::driver::DevicePtr;
    let Device::Cuda(dev) = x.device().clone() else {
        bail!("cast_to_bf16 requires cuda")
    };
    ensure!(x.dtype() == DType::F32, "cast_to_bf16 expects f32");
    let x_c = x.contiguous()?;
    let dims = x_c.dims().to_vec();
    let n: usize = dims.iter().product();
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let y_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
        &Device::Cuda(dev.clone()),
        DType::BF16,
        &[n],
    )?;
    let rc = {
        let (xs, _xl) = x_c.storage_and_layout();
        let xc = match &*xs {
            Storage::Cuda(s) => s,
            _ => bail!("cast_to_bf16: x not cuda"),
        };
        let xsl = xc.as_cuda_slice::<f32>()?;
        let (xp, _gx) = xsl.device_ptr(&stream);
        let yp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&y_dev)?;
        unsafe {
            nv_kernels::cuda::cast_f32_bf16(
                stream.cu_stream() as *mut _,
                xp as *const f32,
                yp as *mut u16,
                n as i32,
            )
        }
    };
    ensure!(rc == 0, "cast_f32_bf16 returned {rc}");
    y_dev
        .reshape(dims)?
        .reshape(x_c.shape())
        .map_err(Into::into)
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn rms_no_weight_f32(x: &Tensor, eps: f64) -> Result<Tensor> {
    use cudarc::driver::DevicePtr;
    let Device::Cuda(dev) = x.device().clone() else {
        bail!("rms_no_weight_f32 requires cuda")
    };
    ensure!(x.dtype() == DType::BF16, "rms_no_weight_f32 expects bf16");
    let (rows, dim) = x.dims2()?;
    let x_c = x.contiguous()?;
    let n = rows * dim;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let y_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
        &Device::Cuda(dev.clone()),
        DType::F32,
        &[n],
    )?;
    let rc = {
        let (xs, xl) = x_c.storage_and_layout();
        let (x0, x1) = xl
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("rms_no_weight_f32: x not dense"))?;
        let xc = match &*xs {
            Storage::Cuda(s) => s,
            _ => bail!("rms_no_weight_f32: x not cuda"),
        };
        let xsl = xc.as_cuda_slice::<half::bf16>()?;
        let xview = xsl.slice(x0..x1);
        let (xp, _gx) = xview.device_ptr(&stream);
        let yp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&y_dev)?;
        unsafe {
            nv_kernels::cuda::rms_no_weight_bf16_f32(
                stream.cu_stream() as *mut _,
                xp as *const u16,
                yp as *mut f32,
                rows as i32,
                dim as i32,
                eps as f32,
            )
        }
    };
    ensure!(rc == 0, "rms_no_weight_bf16_f32 returned {rc}");
    y_dev.reshape((rows, dim)).map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn gelu_mul(gate: &Tensor, pli: &Tensor) -> Result<Tensor> {
    use cudarc::driver::DevicePtr;
    use half::bf16;
    let Device::Cuda(dev) = gate.device().clone() else {
        bail!("gelu_mul requires cuda")
    };
    ensure!(gate.dtype() == DType::BF16, "gelu_mul gate expects bf16");
    ensure!(pli.dtype() == DType::F32, "gelu_mul pli expects f32");
    let dims = gate.dims().to_vec();
    let n: usize = dims.iter().product();
    ensure!(
        pli.elem_count() == n,
        "gelu_mul numel mismatch {} vs {}",
        n,
        pli.elem_count()
    );
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let y_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
        &Device::Cuda(dev.clone()),
        DType::BF16,
        &[n],
    )?;
    let rc = {
        let (gs, gl) = gate.storage_and_layout();
        let (ps, pl) = pli.storage_and_layout();
        let (g0, g1) = gl
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("gelu_mul: gate not dense"))?;
        let (p0, p1) = pl
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("gelu_mul: pli not dense"))?;
        ensure!(g1 - g0 == n && p1 - p0 == n, "gelu_mul: range mismatch");
        let gc = match &*gs {
            Storage::Cuda(s) => s,
            _ => bail!("gelu_mul: gate not cuda"),
        };
        let pc = match &*ps {
            Storage::Cuda(s) => s,
            _ => bail!("gelu_mul: pli not cuda"),
        };
        let gsl = gc.as_cuda_slice::<bf16>()?;
        let psl = pc.as_cuda_slice::<f32>()?;
        let gview = gsl.slice(g0..g1);
        let pview = psl.slice(p0..p1);
        let (gp, _gg) = gview.device_ptr(&stream);
        let (pp, _gp) = pview.device_ptr(&stream);
        let yp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&y_dev)?;
        unsafe {
            nv_kernels::cuda::gelu_mul_bf16f32(
                stream.cu_stream() as *mut _,
                gp as *const u16,
                pp as *const f32,
                yp as *mut u16,
                n as i32,
            )
        }
    };
    ensure!(rc == 0, "gelu_mul_bf16f32 returned {rc}");
    y_dev
        .reshape(dims)?
        .reshape(gate.shape())
        .map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn gather_rows_bf16_op(weight: &Tensor, ids: &Tensor) -> Result<Tensor> {
    use cudarc::driver::DevicePtr;
    use half::bf16;
    let Device::Cuda(dev) = weight.device().clone() else {
        bail!("gather_rows_bf16_op requires cuda")
    };
    let (vocab, hidden) = weight.dims2()?;
    ensure!(weight.dtype() == DType::BF16, "gather weight must be bf16");
    let ids_c = ids.flatten_all()?.contiguous()?;
    ensure!(ids_c.dtype() == DType::U32, "gather ids must be u32");
    let n = ids_c.dims()[0];
    let w_c = weight.contiguous()?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let y_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
        &Device::Cuda(dev.clone()),
        DType::BF16,
        &[n * hidden],
    )?;
    let rc = {
        let (ts, _tl) = ids_c.storage_and_layout();
        let (ws, _wl) = w_c.storage_and_layout();
        let tc = match &*ts {
            Storage::Cuda(s) => s,
            _ => bail!("ids not cuda"),
        };
        let wc = match &*ws {
            Storage::Cuda(s) => s,
            _ => bail!("weight not cuda"),
        };
        let tsl = tc.as_cuda_slice::<u32>()?;
        let wsl = wc.as_cuda_slice::<bf16>()?;
        let (tp, _gt) = tsl.device_ptr(&stream);
        let (wp, _gw) = wsl.device_ptr(&stream);
        let yp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&y_dev)?;
        unsafe {
            nv_kernels::cuda::gather_rows_bf16(
                stream.cu_stream() as *mut _,
                wp as *const u16,
                tp as *const i32,
                yp as *mut u16,
                n as i32,
                hidden as i32,
                vocab as i32,
            )
        }
    };
    ensure!(rc == 0, "gather_rows_bf16 returned {rc}");
    y_dev.reshape((n, hidden)).map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn scale_bf16_op(x: &Tensor, scale: f32) -> Result<Tensor> {
    use cudarc::driver::DevicePtr;
    use half::bf16;
    let Device::Cuda(dev) = x.device().clone() else {
        bail!("scale_bf16_op requires cuda")
    };
    ensure!(x.dtype() == DType::BF16, "scale_bf16_op expects bf16");
    let x_c = x.contiguous()?;
    let dims = x_c.dims().to_vec();
    let n: usize = dims.iter().product();
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let y_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
        &Device::Cuda(dev.clone()),
        DType::BF16,
        &[n],
    )?;
    let rc = {
        let (xs, _xl) = x_c.storage_and_layout();
        let xc = match &*xs {
            Storage::Cuda(s) => s,
            _ => bail!("scale_bf16_op: x not cuda"),
        };
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let (xp, _gx) = xsl.device_ptr(&stream);
        let yp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&y_dev)?;
        unsafe {
            nv_kernels::cuda::scale_out_bf16(
                stream.cu_stream() as *mut _,
                xp as *const u16,
                yp as *mut u16,
                scale,
                n,
            )
        }
    };
    ensure!(rc == 0, "scale_out_bf16 returned {rc}");
    y_dev
        .reshape(dims)?
        .reshape(x_c.shape())
        .map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn cast_scale_f32_op(x: &Tensor, scale: f32) -> Result<Tensor> {
    use cudarc::driver::DevicePtr;
    use half::bf16;
    let Device::Cuda(dev) = x.device().clone() else {
        bail!("cast_scale_f32_op requires cuda")
    };
    ensure!(x.dtype() == DType::BF16, "cast_scale_f32_op expects bf16");
    let x_c = x.contiguous()?;
    let dims = x_c.dims().to_vec();
    let n: usize = dims.iter().product();
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let y_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
        &Device::Cuda(dev.clone()),
        DType::F32,
        &[n],
    )?;
    let rc = {
        let (xs, _xl) = x_c.storage_and_layout();
        let xc = match &*xs {
            Storage::Cuda(s) => s,
            _ => bail!("cast_scale_f32_op: x not cuda"),
        };
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let (xp, _gx) = xsl.device_ptr(&stream);
        let yp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&y_dev)?;
        unsafe {
            nv_kernels::cuda::cast_scale_bf16_f32(
                stream.cu_stream() as *mut _,
                xp as *const u16,
                yp as *mut f32,
                scale,
                n as i32,
            )
        }
    };
    ensure!(rc == 0, "cast_scale_bf16_f32 returned {rc}");
    y_dev
        .reshape(dims)?
        .reshape(x_c.shape())
        .map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn add_scale_f32_op(a: &Tensor, b: &Tensor, scale: f32) -> Result<Tensor> {
    use cudarc::driver::DevicePtr;
    let Device::Cuda(dev) = a.device().clone() else {
        bail!("add_scale_f32_op requires cuda")
    };
    ensure!(
        a.dtype() == DType::F32 && b.dtype() == DType::F32,
        "add_scale_f32_op expects f32"
    );
    let a_c = a.contiguous()?;
    let b_c = b.contiguous()?;
    let dims = a_c.dims().to_vec();
    ensure!(dims == b_c.dims(), "add_scale_f32_op shape mismatch");
    let n: usize = dims.iter().product();
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let y_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
        &Device::Cuda(dev.clone()),
        DType::F32,
        &[n],
    )?;
    let rc = {
        let (as_, _al) = a_c.storage_and_layout();
        let (bs, _bl) = b_c.storage_and_layout();
        let ac = match &*as_ {
            Storage::Cuda(s) => s,
            _ => bail!("add_scale_f32_op: a not cuda"),
        };
        let bc = match &*bs {
            Storage::Cuda(s) => s,
            _ => bail!("add_scale_f32_op: b not cuda"),
        };
        let asl = ac.as_cuda_slice::<f32>()?;
        let bsl = bc.as_cuda_slice::<f32>()?;
        let (ap, _ga) = asl.device_ptr(&stream);
        let (bp, _gb) = bsl.device_ptr(&stream);
        let yp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&y_dev)?;
        unsafe {
            nv_kernels::cuda::add_scale_f32(
                stream.cu_stream() as *mut _,
                ap as *const f32,
                bp as *const f32,
                yp as *mut f32,
                scale,
                n as i32,
            )
        }
    };
    ensure!(rc == 0, "add_scale_f32 returned {rc}");
    y_dev
        .reshape(dims)?
        .reshape(a_c.shape())
        .map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn rmsnorm_add_scale_op(
    x: &Tensor,
    norm: &nv_layers::norm::RmsNorm,
    res: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    Ok(rmsnorm_add_scale_chain_op(x, norm, res, scale, None, None)?.0)
}

#[cfg(feature = "cuda")]
fn rmsnorm_add_scale_chain_op(
    x: &Tensor,
    norm: &nv_layers::norm::RmsNorm,
    res: &Tensor,
    scale: f32,
    chain_eps: Option<f32>,
    next_w: Option<&Tensor>,
) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
    use cudarc::driver::DevicePtr;
    use half::bf16;
    let Device::Cuda(dev) = x.device().clone() else {
        bail!("rmsnorm_add_scale_op requires cuda")
    };
    ensure!(
        x.dtype() == DType::BF16 && res.dtype() == DType::BF16,
        "rmsnorm_add_scale_op expects bf16"
    );
    let x_c = x.contiguous()?;
    let res_c = res.contiguous()?;
    let dims = x_c.dims().to_vec();
    ensure!(res_c.elem_count() == x_c.elem_count(), "res numel mismatch");
    let dim = *dims.last().unwrap();
    let rows: usize = dims.iter().product::<usize>() / dim;
    let w = norm.weight_bf16();
    ensure!(
        w.dtype() == DType::BF16 && w.elem_count() == dim,
        "norm weight mismatch"
    );
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let y_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
        &Device::Cuda(dev.clone()),
        DType::BF16,
        &[rows * dim],
    )?;
    let rstd_dev: Option<Tensor> = match chain_eps {
        Some(_) => Some(
            nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
                &Device::Cuda(dev.clone()),
                DType::F32,
                &[rows],
            )?,
        ),
        None => None,
    };
    if next_w.is_some() {
        ensure!(chain_eps.is_some(), "next_w requires chain_eps");
    }
    let normed_dev: Option<Tensor> = match next_w {
        Some(nw) => {
            ensure!(
                nw.dtype() == DType::BF16 && nw.elem_count() == dim,
                "next_w mismatch"
            );
            Some(
                nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
                    &Device::Cuda(dev.clone()),
                    DType::BF16,
                    &[rows * dim],
                )?,
            )
        }
        None => None,
    };
    let rc = {
        let (xs, _xl) = x_c.storage_and_layout();
        let (ws, _wl) = w.storage_and_layout();
        let (rs, _rl) = res_c.storage_and_layout();
        let xc = match &*xs {
            Storage::Cuda(s) => s,
            _ => bail!("x not cuda"),
        };
        let wc = match &*ws {
            Storage::Cuda(s) => s,
            _ => bail!("w not cuda"),
        };
        let rcu = match &*rs {
            Storage::Cuda(s) => s,
            _ => bail!("res not cuda"),
        };
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let wsl = wc.as_cuda_slice::<bf16>()?;
        let rsl = rcu.as_cuda_slice::<bf16>()?;
        let (xp, _gx) = xsl.device_ptr(&stream);
        let (wp, _gw) = wsl.device_ptr(&stream);
        let (rp, _gr) = rsl.device_ptr(&stream);
        let yp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&y_dev)?;
        let rstd_ptr = match rstd_dev.as_ref() {
            Some(b) => nv_layers::step_arena::device_ptr_for_a_kernel_to_write(b)? as *mut f32,
            None => std::ptr::null_mut(),
        };
        let next_w_ptr = match next_w {
            Some(nw) => {
                let (ns, _nl) = nw.storage_and_layout();
                let nc = match &*ns {
                    Storage::Cuda(s) => s,
                    _ => bail!("next_w not cuda"),
                };
                let nsl = nc.as_cuda_slice::<bf16>()?;
                let (np, _gn) = nsl.device_ptr(&stream);
                np as *const u16
            }
            None => std::ptr::null(),
        };
        let normed_ptr = match normed_dev.as_ref() {
            Some(b) => nv_layers::step_arena::device_ptr_for_a_kernel_to_write(b)? as *mut u16,
            None => std::ptr::null_mut(),
        };
        unsafe {
            nv_kernels::cuda::rmsnorm_add_scale_bf16(
                stream.cu_stream() as *mut _,
                xp as *const u16,
                wp as *const u16,
                rp as *const u16,
                yp as *mut u16,
                rstd_ptr,
                next_w_ptr,
                normed_ptr,
                rows as i32,
                dim as i32,
                norm.eps() as f32,
                scale,
                chain_eps.unwrap_or(0.0),
            )
        }
    };
    ensure!(rc == 0, "rmsnorm_add_scale_bf16 returned {rc}");
    let y = y_dev.reshape(dims)?.reshape(x_c.shape())?;
    let rstd = rstd_dev;
    let normed = normed_dev.map(|b| b.reshape(x_c.shape())).transpose()?;
    Ok((y, rstd, normed))
}

#[cfg(feature = "cuda")]
pub(crate) fn rstd_op(x: &Tensor, eps: f32) -> Result<Tensor> {
    use cudarc::driver::DevicePtr;
    use half::bf16;
    let Device::Cuda(dev) = x.device().clone() else {
        bail!("rstd_op requires cuda")
    };
    ensure!(x.dtype() == DType::BF16, "rstd_op expects bf16");
    let dims = x.dims().to_vec();
    let dim = *dims.last().context("rstd_op: scalar input")?;
    let rows = x.elem_count() / dim;
    let x_c = x.contiguous()?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let r_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
        &Device::Cuda(dev.clone()),
        DType::F32,
        &[rows],
    )?;
    let rc = {
        let (xs, xl) = x_c.storage_and_layout();
        let (x0, x1) = xl
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("rstd_op: x not dense"))?;
        let xc = match &*xs {
            Storage::Cuda(s) => s,
            _ => bail!("x not cuda"),
        };
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let xview = xsl.slice(x0..x1);
        let (xp, _gx) = xview.device_ptr(&stream);
        let rp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&r_dev)?;
        unsafe {
            nv_kernels::cuda::rstd_bf16(
                stream.cu_stream() as *mut _,
                xp as *const u16,
                rp as *mut f32,
                rows as i32,
                dim as i32,
                eps,
            )
        }
    };
    ensure!(rc == 0, "rstd_bf16 returned {rc}");
    r_dev.reshape((rows,)).map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn rms_apply_op(x: &Tensor, w_bf16: &Tensor, rstd: &Tensor) -> Result<Tensor> {
    use cudarc::driver::DevicePtr;
    use half::bf16;
    let Device::Cuda(dev) = x.device().clone() else {
        bail!("rms_apply_op requires cuda")
    };
    ensure!(x.dtype() == DType::BF16, "rms_apply_op expects bf16");
    let x_c = x.contiguous()?;
    let dims = x_c.dims().to_vec();
    let n: usize = dims.iter().product();
    let dim = w_bf16.elem_count();
    ensure!(
        dim > 0 && n.is_multiple_of(dim),
        "rms_apply_op weight mismatch"
    );
    ensure!(
        rstd.elem_count() == n / dim,
        "rms_apply_op rstd rows mismatch"
    );
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let y_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
        &Device::Cuda(dev.clone()),
        DType::BF16,
        &[n],
    )?;
    let rc = {
        let (xs, xl) = x_c.storage_and_layout();
        let (x0, x1) = xl
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("rms_apply_op: x not dense"))?;
        let (ws, _wl) = w_bf16.storage_and_layout();
        let (rs, _rl) = rstd.storage_and_layout();
        let xc = match &*xs {
            Storage::Cuda(s) => s,
            _ => bail!("x not cuda"),
        };
        let wc = match &*ws {
            Storage::Cuda(s) => s,
            _ => bail!("w not cuda"),
        };
        let rcu = match &*rs {
            Storage::Cuda(s) => s,
            _ => bail!("rstd not cuda"),
        };
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let xview = xsl.slice(x0..x1);
        let wsl = wc.as_cuda_slice::<bf16>()?;
        let rsl = rcu.as_cuda_slice::<f32>()?;
        let (xp, _gx) = xview.device_ptr(&stream);
        let (wp, _gw) = wsl.device_ptr(&stream);
        let (rp, _gr) = rsl.device_ptr(&stream);
        let yp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&y_dev)?;
        unsafe {
            nv_kernels::cuda::rms_apply_bf16(
                stream.cu_stream() as *mut _,
                xp as *const u16,
                wp as *const u16,
                rp as *const f32,
                yp as *mut u16,
                n as i32,
                dim as i32,
            )
        }
    };
    ensure!(rc == 0, "rms_apply_bf16 returned {rc}");
    y_dev
        .reshape(dims)?
        .reshape(x_c.shape())
        .map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn lm_head_normed_op(embed: &Tensor, x: &Tensor, w_norm: &Tensor, rstd: &Tensor) -> Result<Tensor> {
    use cudarc::driver::DevicePtr;
    use half::bf16;
    let Device::Cuda(dev) = x.device().clone() else {
        bail!("lm_head_normed_op requires cuda")
    };
    let (vocab, hidden) = embed.dims2()?;
    ensure!(x.elem_count() == hidden, "lm_head x mismatch");
    let x_c = x.contiguous()?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let y_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
        &Device::Cuda(dev.clone()),
        DType::BF16,
        &[vocab],
    )?;
    let rc = {
        let (es, _el) = embed.storage_and_layout();
        let (xs, xl) = x_c.storage_and_layout();
        let (x0, x1) = xl
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("lm_head_normed_op: x not dense"))?;
        let (ws, _wl) = w_norm.storage_and_layout();
        let (rs, _rl) = rstd.storage_and_layout();
        let ec = match &*es {
            Storage::Cuda(s) => s,
            _ => bail!("embed not cuda"),
        };
        let xc = match &*xs {
            Storage::Cuda(s) => s,
            _ => bail!("x not cuda"),
        };
        let wc = match &*ws {
            Storage::Cuda(s) => s,
            _ => bail!("w_norm not cuda"),
        };
        let rcu = match &*rs {
            Storage::Cuda(s) => s,
            _ => bail!("rstd not cuda"),
        };
        let esl = ec.as_cuda_slice::<bf16>()?;
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let xview = xsl.slice(x0..x1);
        let wsl = wc.as_cuda_slice::<bf16>()?;
        let rsl = rcu.as_cuda_slice::<f32>()?;
        let (ep, _ge) = esl.device_ptr(&stream);
        let (xp, _gx) = xview.device_ptr(&stream);
        let (wp, _gw) = wsl.device_ptr(&stream);
        let (rp, _gr) = rsl.device_ptr(&stream);
        let yp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&y_dev)?;
        unsafe {
            nv_kernels::cuda::gemv_bf16_normed(
                stream.cu_stream() as *mut _,
                ep as *const u16,
                xp as *const u16,
                wp as *const u16,
                rp as *const f32,
                yp as *mut u16,
                vocab as i32,
                hidden as i32,
            )
        }
    };
    ensure!(rc == 0, "gemv_bf16_normed returned {rc}");
    y_dev.reshape((1usize, 1usize, vocab)).map_err(Into::into)
}

#[cfg(not(feature = "cuda"))]
fn geglu_fused_op(gu: &Tensor) -> Result<Tensor> {
    let last = gu.dims().len() - 1;
    let two_inter = gu.dim(last)?;
    ensure!(two_inter % 2 == 0, "geglu last dim must be even");
    let inter = two_inter / 2;
    let gate = gu.narrow(last, 0, inter)?.contiguous()?;
    let up = gu.narrow(last, inter, inter)?.contiguous()?;
    let act = gate.gelu()?;
    act.mul(&up).map_err(|e| anyhow::anyhow!(e))
}

#[cfg(feature = "cuda")]
fn geglu_fused_op(gu: &Tensor) -> Result<Tensor> {
    use cudarc::driver::DevicePtr;
    use half::bf16;
    let Device::Cuda(dev) = gu.device().clone() else {
        bail!("geglu_fused_op requires cuda")
    };
    ensure!(gu.dtype() == DType::BF16, "geglu_fused_op expects bf16");
    let gu_c = gu.contiguous()?;
    let mut dims = gu_c.dims().to_vec();
    let two_inter = *dims.last().unwrap();
    ensure!(two_inter % 2 == 0, "geglu last dim must be even");
    let inter = two_inter / 2;
    let rows: usize = dims.iter().product::<usize>() / two_inter;
    *dims.last_mut().unwrap() = inter;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let y_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
        &Device::Cuda(dev.clone()),
        DType::BF16,
        &[rows * inter],
    )?;
    let rc = {
        let (gs, _gl) = gu_c.storage_and_layout();
        let gc = match &*gs {
            Storage::Cuda(s) => s,
            _ => bail!("gu not cuda"),
        };
        let gsl = gc.as_cuda_slice::<bf16>()?;
        let (gp, _gg) = gsl.device_ptr(&stream);
        let yp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&y_dev)?;
        unsafe {
            nv_kernels::cuda::gelu_tanh_mul_fused_bf16(
                stream.cu_stream() as *mut _,
                gp as *const u16,
                yp as *mut u16,
                inter as i32,
                rows * inter,
            )
        }
    };
    ensure!(rc == 0, "gelu_tanh_mul_fused_bf16 returned {rc}");
    y_dev.reshape(dims).map_err(Into::into)
}

#[cfg(feature = "cuda")]
pub(crate) struct E4bFixedCache {
    k: Vec<Option<CudaSlice<half::bf16>>>,
    v: Vec<Option<CudaSlice<half::bf16>>>,
    pos: CudaSlice<i32>,

    fan_in: CudaSlice<u32>,
    device: CudaDevice,
}

#[cfg(feature = "cuda")]
impl E4bFixedCache {
    fn from_cat(
        cat: &[Option<(Tensor, Tensor)>],
        plen: usize,
        max_len: usize,
        dev: &CudaDevice,
    ) -> Result<Self> {
        let stream = nv_layers::cuda_stream::current_stream(dev);
        let mut k = Vec::with_capacity(cat.len());
        let mut v = Vec::with_capacity(cat.len());
        for entry in cat {
            match entry {
                Some((kt, vt)) => {
                    let dims = kt.dims().to_vec();
                    let nkv = dims[2];
                    let hd = dims[3];
                    let mut kbuf = unsafe { stream.alloc::<half::bf16>(max_len * nkv * hd)? };
                    let mut vbuf = unsafe { stream.alloc::<half::bf16>(max_len * nkv * hd)? };
                    cast_tensor_front_bf16(kt, &mut kbuf, plen * nkv * hd, &stream)?;
                    cast_tensor_front_bf16(vt, &mut vbuf, plen * nkv * hd, &stream)?;
                    k.push(Some(kbuf));
                    v.push(Some(vbuf));
                }
                None => {
                    k.push(None);
                    v.push(None);
                }
            }
        }
        let pos = stream.clone_htod(&[plen as i32])?;
        let fan_in = stream.alloc_zeros::<u32>(64)?;
        Ok(Self {
            k,
            v,
            pos,
            fan_in,
            device: dev.clone(),
        })
    }

    fn qkv_prep(
        &mut self,
        idx: Option<usize>,
        qkv: &Tensor,
        qkv_row: usize,
        q_norm_w: &Tensor,
        k_norm_w: Option<&Tensor>,
        rope: &Rope,
        rope_pos: &Tensor,
        delta: i32,
        nh: usize,
        nkv: usize,
        hd: usize,
        eps: f32,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let stream = nv_layers::cuda_stream::current_stream(&self.device);
        let qkv_c = qkv.contiguous()?;
        let expected = if k_norm_w.is_some() {
            (nh + 2 * nkv) * hd
        } else {
            nh * hd
        };
        ensure!(
            qkv_c.dtype() == DType::BF16
                && qkv_c.elem_count() % expected == 0
                && qkv_row < qkv_c.elem_count() / expected,
            "qkv_prep: bad qkv shape"
        );
        ensure!(rope_pos.dtype() == DType::I32, "rope_pos must be i32");
        let q_out = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
            &Device::Cuda(self.device.clone()),
            DType::F32,
            &[nh * hd],
        )?;
        let rc = {
            let (qs, _ql) = qkv_c.storage_and_layout();
            let qc = match &*qs {
                Storage::Cuda(s) => s,
                _ => bail!("qkv not cuda"),
            };
            let qsl = qc.as_cuda_slice::<half::bf16>()?;
            let (qp, _gq) = qsl.device_ptr(&stream);
            let qp = qp + (qkv_row * expected * 2) as u64;
            let (qws, _l) = q_norm_w.storage_and_layout();
            let qwc = match &*qws {
                Storage::Cuda(s) => s,
                _ => bail!("q_norm weight not cuda"),
            };
            let qwsl = qwc.as_cuda_slice::<half::bf16>()?;
            let (qwp, _gqw) = qwsl.device_ptr(&stream);
            let (cs, _cl) = rope.cos().storage_and_layout();
            let (ss, _sl) = rope.sin().storage_and_layout();
            let cc = match &*cs {
                Storage::Cuda(s) => s,
                _ => bail!("cos not cuda"),
            };
            let sc = match &*ss {
                Storage::Cuda(s) => s,
                _ => bail!("sin not cuda"),
            };
            let csl = cc.as_cuda_slice::<f32>()?;
            let ssl = sc.as_cuda_slice::<f32>()?;
            let (cp, _gc) = csl.device_ptr(&stream);
            let (sp, _gs) = ssl.device_ptr(&stream);
            let (rs, _rl) = rope_pos.storage_and_layout();
            let rc2 = match &*rs {
                Storage::Cuda(s) => s,
                _ => bail!("rope_pos not cuda"),
            };
            let rsl = rc2.as_cuda_slice::<i32>()?;
            let (rp, _gr) = rsl.device_ptr(&stream);
            let op = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&q_out)?;
            let (kwp, cache_pos_ptr, ckp, cvp) = match k_norm_w {
                Some(kw) => {
                    let (kws, _l) = kw.storage_and_layout();
                    let kwc = match &*kws {
                        Storage::Cuda(s) => s,
                        _ => bail!("k_norm weight not cuda"),
                    };
                    let kwsl = kwc.as_cuda_slice::<half::bf16>()?;
                    let (kwp, _g) = kwsl.device_ptr(&stream);
                    let (pp, _gp) = self.pos.device_ptr(&stream);
                    let i = idx.context("qkv_prep: kv layer without idx")?;
                    let kc = self.k[i].as_mut().context("qkv_prep: no k buffer")?;
                    let (ckp, _gk) = kc.device_ptr_mut(&stream);
                    let vc = self.v[i].as_mut().context("qkv_prep: no v buffer")?;
                    let (cvp, _gv) = vc.device_ptr_mut(&stream);
                    (
                        kwp as *const u16,
                        pp as *const i32,
                        ckp as *mut u16,
                        cvp as *mut u16,
                    )
                }
                None => (
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ),
            };
            unsafe {
                nv_kernels::cuda::qkv_prep(
                    stream.cu_stream() as *mut _,
                    qp as *const u16,
                    qwp as *const u16,
                    kwp,
                    cp as *const f32,
                    sp as *const f32,
                    rp as *const i32,
                    cache_pos_ptr,
                    delta,
                    op as *mut f32,
                    ckp,
                    cvp,
                    nh as i32,
                    nkv as i32,
                    hd as i32,
                    eps,
                )
            }
        };
        ensure!(rc == 0, "qkv_prep returned {rc}");
        q_out.reshape((nh, hd)).map_err(Into::into)
    }

    fn attend(
        &self,
        buf: usize,
        q: &Tensor,
        delta: i32,
        nh: usize,
        nkv: usize,
        hd: usize,
        window: usize,
    ) -> Result<Tensor> {
        use cudarc::driver::DevicePtr;
        let stream = nv_layers::cuda_stream::current_stream(&self.device);
        let q = q.contiguous()?;

        let out_dev = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
            &Device::Cuda(self.device.clone()),
            DType::BF16,
            &[nh * hd],
        )?;

        let grouped = window == 0 && hd == 512 && nh == nkv * 4;
        let capacity = self.k[buf]
            .as_ref()
            .map(|kb| kb.len() / (nkv * hd))
            .unwrap_or(0);
        let scratch_elems = if grouped {
            nv_kernels::cuda::flash_decode_gqa4_hd512_scratch_elems(nh as i32, capacity as i32)
        } else {
            nv_kernels::cuda::flash_splitk_scratch_elems(nh as i32, hd as i32)
        };
        let scratch = nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
            &Device::Cuda(self.device.clone()),
            DType::F32,
            &[scratch_elems],
        )?;
        let rc = {
            let pos_ptr = {
                let (p, _g) = self.pos.device_ptr(&stream);
                p as *const i32
            };
            let kc = self.k[buf].as_ref().context("attend: no k buffer")?;
            let vc = self.v[buf].as_ref().context("attend: no v buffer")?;
            let (ckp, _gk) = kc.device_ptr(&stream);
            let (cvp, _gv) = vc.device_ptr(&stream);
            let (qs, _ql) = q.storage_and_layout();
            let qcuda = match &*qs {
                Storage::Cuda(s) => s,
                _ => bail!("q not cuda"),
            };
            let qsl = qcuda.as_cuda_slice::<f32>()?;
            let (qp, _gq) = qsl.device_ptr(&stream);
            let op = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&out_dev)?;
            let sp = nv_layers::step_arena::device_ptr_for_a_kernel_to_write(&scratch)?;
            let (fp, _gf) = self.fan_in.device_ptr(&stream);
            unsafe {
                if grouped {
                    nv_kernels::cuda::flash_decode_fused_gqa4_hd512(
                        stream.cu_stream() as *mut _,
                        qp as *const f32,
                        ckp as *const u16,
                        cvp as *const u16,
                        op as *mut u16,
                        pos_ptr,
                        delta,
                        sp as *mut f32,
                        fp as *mut u32,
                        nh as i32,
                        nkv as i32,
                        hd as i32,
                        window as i32,
                        capacity as i32,
                    )
                } else {
                    nv_kernels::cuda::flash_decode_fused_bf16kv(
                        stream.cu_stream() as *mut _,
                        qp as *const f32,
                        ckp as *const u16,
                        cvp as *const u16,
                        op as *mut u16,
                        pos_ptr,
                        delta,
                        sp as *mut f32,
                        fp as *mut u32,
                        nh as i32,
                        nkv as i32,
                        hd as i32,
                        window as i32,
                    )
                }
            }
        };
        ensure!(rc == 0, "flash decode (grouped={grouped}) returned {rc}");
        out_dev.reshape((nh, hd)).map_err(Into::into)
    }
}

#[cfg(feature = "cuda")]
fn cast_tensor_front_bf16(
    t: &Tensor,
    dst: &mut CudaSlice<half::bf16>,
    n: usize,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<()> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let t = t.contiguous()?;
    let (ts, _tl) = t.storage_and_layout();
    let tcuda = match &*ts {
        Storage::Cuda(s) => s,
        _ => bail!("cast_tensor_front_bf16: not cuda"),
    };
    let tsl = tcuda.as_cuda_slice::<f32>()?;
    let (sp, _gs) = tsl.device_ptr(stream);
    let (dp, _gd) = dst.device_ptr_mut(stream);
    let rc = unsafe {
        nv_kernels::cuda::cast_f32_bf16(
            stream.cu_stream() as *mut _,
            sp as *const f32,
            dp as *mut u16,
            n as i32,
        )
    };
    ensure!(rc == 0, "cast_f32_bf16 (cache prefill) returned {rc}");
    Ok(())
}

fn repeat_kv(x: &Tensor, group: usize) -> Result<Tensor> {
    if group == 1 {
        return Ok(x.clone());
    }
    let (nkv, seq, hd) = x.dims3()?;
    x.unsqueeze(1)?
        .broadcast_as((nkv, group, seq, hd))?
        .reshape((nkv * group, seq, hd))
        .map_err(Into::into)
}

fn causal_mask(seq: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            data[i * seq + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (seq, seq), device).map_err(Into::into)
}

fn softcap(logits: &Tensor, cap: f32) -> Result<Tensor> {
    if cap <= 0.0 {
        return Ok(logits.clone());
    }
    let cap = cap as f64;
    ((logits / cap)?.tanh()? * cap).map_err(Into::into)
}

#[cfg(feature = "cuda")]
pub struct GraphedE4bDecoder<'m> {
    model: &'m Gemma4E4b,
    cache: E4bFixedCache,
    device: CudaDevice,
    forked: std::sync::Arc<cudarc::driver::CudaStream>,
    runner: nv_kernels::graph::CudaGraphRunner,
    token_buf: CudaSlice<u32>,
    rope_src: CudaSlice<i32>,
    ring: CudaSlice<u32>,
    amax_val: CudaSlice<f32>,
    amax_idx: CudaSlice<i32>,
    host_tok: Box<[u32; 1]>,
    arena: Option<nv_layers::step_arena::StepArena>,
    current_pos: usize,
    call_count: u64,
    capture_active: bool,
}

#[cfg(feature = "cuda")]
const E4B_TOKEN_RING: usize = 64;

#[cfg(feature = "cuda")]
impl<'m> GraphedE4bDecoder<'m> {
    fn new(model: &'m Gemma4E4b, cache: E4bFixedCache, start_len: usize) -> Result<Self> {
        let Device::Cuda(dev) = model.device.clone() else {
            bail!("GraphedE4bDecoder requires cuda")
        };
        let raw_ctx = dev.cuda_stream().context().clone();
        crate::gemma4_batch_graph::graph_teardown::disable_event_tracking_before_capture(&raw_ctx);
        let forked = raw_ctx
            .new_stream()
            .map_err(|e| anyhow::anyhow!("forked stream: {e:?}"))?;
        let token_buf = forked
            .alloc_zeros::<u32>(1)
            .map_err(|e| anyhow::anyhow!("alloc token_buf: {e:?}"))?;
        let rope_src = forked
            .alloc_zeros::<i32>(1)
            .map_err(|e| anyhow::anyhow!("alloc rope_src: {e:?}"))?;
        let ring = forked
            .alloc_zeros::<u32>(E4B_TOKEN_RING)
            .map_err(|e| anyhow::anyhow!("alloc ring: {e:?}"))?;
        let parts = nv_kernels::cuda::argmax_parts();
        let amax_val = forked
            .alloc_zeros::<f32>(parts)
            .map_err(|e| anyhow::anyhow!("alloc amax_val: {e:?}"))?;
        let amax_idx = forked
            .alloc_zeros::<i32>(parts)
            .map_err(|e| anyhow::anyhow!("alloc amax_idx: {e:?}"))?;
        forked
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync after alloc: {e:?}"))?;
        let runner = nv_kernels::graph::CudaGraphRunner::new(forked.clone());
        Ok(Self {
            model,
            cache,
            device: dev,
            forked,
            runner,
            token_buf,
            rope_src,
            ring,
            amax_val,
            amax_idx,
            host_tok: Box::new([0u32; 1]),
            arena: Some(nv_layers::step_arena::StepArena::new()),
            current_pos: start_len,
            call_count: 0,
            capture_active: false,
        })
    }

    pub fn call_count(&self) -> u64 {
        self.call_count
    }

    pub fn capture_active(&self) -> bool {
        self.capture_active
    }

    pub fn kernel_census_of_the_captured_step(&self) -> BTreeMap<String, usize> {
        self.runner.cached_kernel_name_census()
    }

    pub fn captured_node_count(&self) -> usize {
        self.runner.cached_node_count()
    }

    pub fn captured_node_type_census(&self) -> BTreeMap<u32, usize> {
        self.runner
            .cached_node_type_census_type_ids_are_cugraphnodetype()
    }

    pub fn captured_memset_census(&self) -> BTreeMap<(u32, u64), usize> {
        self.runner
            .cached_memset_census_key_is_element_size_and_bytes()
    }

    pub fn captured_mem_alloc_consumer_census(&self) -> BTreeMap<(u64, String), usize> {
        self.runner
            .cached_mem_alloc_consumer_census_key_is_bytes_and_kernel_name()
    }

    pub fn captured_memcpy_consumer_census(&self) -> BTreeMap<(u64, String), usize> {
        self.runner
            .cached_memcpy_consumer_census_key_is_bytes_and_kernel_name()
    }

    fn arena_enter(&mut self) {
        if let Some(a) = self.arena.take() {
            nv_layers::step_arena::install(a);
            nv_layers::step_arena::begin_step();
        }
    }

    fn arena_leave(&mut self) {
        self.arena = nv_layers::step_arena::uninstall();
    }

    pub fn warm_step(&mut self, token_id: u32) -> Result<u32> {
        self.arena_enter();
        let logits = self
            .model
            .forward_step_fast(token_id, self.current_pos, &mut self.cache);
        self.arena_leave();
        let logits = logits?;
        self.call_count += 1;
        self.current_pos += 1;
        logits
            .argmax(D::Minus1)?
            .to_scalar::<u32>()
            .map_err(Into::into)
    }

    pub fn replay_batch(&mut self, seed_token: u32, k: usize) -> Result<Vec<u32>> {
        ensure!((1..=E4B_TOKEN_RING).contains(&k), "bad batch size {k}");

        let ctx = self.device.cuda_stream().context().clone();
        if ctx.is_event_tracking() {
            unsafe { ctx.disable_event_tracking() };
            self.device
                .cuda_stream()
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-capture legacy sync: {e:?}"))?;
        }

        self.host_tok[0] = seed_token;
        let host_tok_slice: &[u32] = &self.host_tok[..];
        self.forked
            .memcpy_htod(host_tok_slice, &mut self.token_buf)
            .map_err(|e| anyhow::anyhow!("seed htod: {e:?}"))?;
        marlin_zero_upload(&self.forked)?;

        let start_pos = self.current_pos;
        self.arena_enter();
        for _ in 0..k {
            let GraphedE4bDecoder {
                model,
                cache,
                device,
                runner,
                token_buf,
                rope_src,
                ring,
                amax_val,
                amax_idx,
                ..
            } = &mut *self;
            let dev = device.clone();
            runner.run(1u64, |s| -> Result<()> {
                nv_layers::cuda_stream::with_stream(s.clone(), || -> Result<()> {
                    nv_layers::step_arena::begin_step();
                    let tok_clone = token_buf
                        .try_clone()
                        .map_err(|e| anyhow::anyhow!("clone token_buf: {e:?}"))?;
                    let pos_clone = rope_src
                        .try_clone()
                        .map_err(|e| anyhow::anyhow!("clone rope_src: {e:?}"))?;
                    let ids = {
                        let storage = CudaStorage::wrap_cuda_slice(tok_clone, dev.clone());
                        let storage = Storage::Cuda(storage);
                        Tensor::from_storage(storage, (1usize,), BackpropOp::none(), false)
                    };
                    let pos_t = {
                        let storage = CudaStorage::wrap_cuda_slice(pos_clone, dev.clone());
                        let storage = Storage::Cuda(storage);
                        Tensor::from_storage(storage, (1usize,), BackpropOp::none(), false)
                    };
                    let logits = model.forward_step_fast_body(&ids, &pos_t, cache)?;
                    argmax_into_bufs(
                        &logits,
                        &cache.pos,
                        token_buf,
                        ring,
                        amax_val,
                        amax_idx,
                        (E4B_TOKEN_RING - 1) as i32,
                        &dev,
                    )
                })
            })?;
            self.capture_active = true;
            self.call_count += 1;
            self.current_pos += 1;
        }
        self.arena_leave();

        self.forked
            .synchronize()
            .map_err(|e| anyhow::anyhow!("forked sync: {e:?}"))?;
        let host: Vec<u32> = self
            .forked
            .clone_dtoh(&self.ring)
            .map_err(|e| anyhow::anyhow!("dtoh ring: {e:?}"))?;
        Ok((0..k)
            .map(|i| host[(start_pos + i) & (E4B_TOKEN_RING - 1)])
            .collect())
    }
}

#[cfg(feature = "cuda")]
impl Drop for GraphedE4bDecoder<'_> {
    fn drop(&mut self) {
        let td = crate::gemma4_batch_graph::graph_teardown::GraphTeardown::new(&self.forked);
        let runner = &mut self.runner;
        td.run(|| runner.invalidate());
    }
}

#[cfg(feature = "cuda")]
fn argmax_into_bufs(
    logits: &Tensor,
    pos: &CudaSlice<i32>,
    token_buf: &mut CudaSlice<u32>,
    ring: &mut CudaSlice<u32>,
    amax_val: &mut CudaSlice<f32>,
    amax_idx: &mut CudaSlice<i32>,
    ring_mask: i32,
    dev: &CudaDevice,
) -> Result<()> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    ensure!(logits.dtype() == DType::BF16, "argmax expects bf16 logits");
    let n = logits.elem_count();
    let logits_c = logits.contiguous()?;
    let stream = nv_layers::cuda_stream::current_stream(dev);
    let rc = {
        let (ls, _ll) = logits_c.storage_and_layout();
        let lc = match &*ls {
            Storage::Cuda(s) => s,
            _ => bail!("logits not cuda"),
        };
        let lsl = lc.as_cuda_slice::<bf16>()?;
        let (lp, _gl) = lsl.device_ptr(&stream);
        let (pp, _gp) = pos.device_ptr(&stream);
        let (tp, _gt) = token_buf.device_ptr_mut(&stream);
        let (rp, _gr) = ring.device_ptr_mut(&stream);
        let (vp, _gv) = amax_val.device_ptr_mut(&stream);
        let (ip, _gi) = amax_idx.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::argmax_bf16(
                stream.cu_stream() as *mut _,
                lp as *const u16,
                n as i32,
                vp as *mut f32,
                ip as *mut i32,
                pp as *const i32,
                tp as *mut u32,
                rp as *mut u32,
                ring_mask,
            )
        }
    };
    ensure!(rc == 0, "argmax_bf16 returned {rc}");
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
mod fast_graph_debug {
    use super::*;

    fn capture_lock() -> std::sync::MutexGuard<'static, ()> {
        static L: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        L.get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn load() -> Option<(Gemma4E4b, Device)> {
        let Ok(dir) = std::env::var("GEMMA4_E4B_DIR") else {
            eprintln!("skip: set GEMMA4_E4B_DIR");
            return None;
        };
        let device = Device::new_cuda(0).ok()?;
        let dir = std::path::Path::new(&dir);
        let cfg = crate::gemma4::Gemma4Config::from_hf_json_file(&dir.join("config.json")).ok()?;
        let weights = nv_weights::WeightLoader::open_dir(dir, &device).ok()?;
        let model = Gemma4E4b::from_loader(cfg, &weights, &device).ok()?;
        Some((model, device))
    }

    fn to_f32(t: &Tensor) -> Vec<f32> {
        t.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    #[test]
    fn fast_ops_vs_candle() {
        let _g = capture_lock();
        let Some((model, device)) = load() else {
            return;
        };
        let hidden = model.config.hidden_size;
        let ids = Tensor::from_vec(vec![563u32], 1, &device).unwrap();

        let e_old = model.embed_tokens.index_select(&ids, 0).unwrap();
        let e_old = (e_old.to_dtype(DType::F32).unwrap() * model.normalizer)
            .unwrap()
            .to_dtype(model.dtype)
            .unwrap()
            .reshape((1, 1, hidden))
            .unwrap();
        let emb = gather_rows_bf16_op(&model.embed_tokens, &ids).unwrap();
        let e_new = scale_bf16_op(&emb, model.normalizer as f32)
            .unwrap()
            .reshape((1, 1, hidden))
            .unwrap();
        let d_embed = maxdiff(&to_f32(&e_old), &to_f32(&e_new));
        eprintln!("embed maxdiff: {d_embed:e}");

        let p_old = model.per_layer_inputs(&ids, &e_old).unwrap();
        let p_new = model.per_layer_inputs_fast(&ids, &e_new).unwrap();
        let d_pli = maxdiff(&to_f32(&p_old), &to_f32(&p_new));
        eprintln!("pli maxdiff:   {d_pli:e}");

        assert!(d_embed < 1e-2, "embed diverged: {d_embed}");
        assert!(d_pli < 1e-2, "per_layer_inputs diverged: {d_pli}");
    }

    #[test]
    fn capture_op_units() {
        let _g = capture_lock();
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("skip: no cuda");
            return;
        };
        let dev = match &device {
            Device::Cuda(d) => d.clone(),
            _ => return,
        };
        let n = 2048usize;
        let xs: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect();
        let ys: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.11).cos() * 2.0).collect();
        let x_bf = Tensor::from_vec(xs.clone(), n, &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let y_bf = Tensor::from_vec(ys.clone(), n, &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let x_f32 = Tensor::from_vec(xs.clone(), n, &device).unwrap();
        let x_rows = x_bf.reshape((8, 256)).unwrap();
        let q4 = x_f32.reshape((1, 1, 8, 256)).unwrap();
        let k4 = Tensor::from_vec(ys.clone(), n, &device)
            .unwrap()
            .reshape((1, 1, 8, 256))
            .unwrap();
        let pos_t = Tensor::from_vec(vec![3i32], 1, &device).unwrap();
        let rope = build_rope(256, 10000.0, 1.0, 64, &device).unwrap();
        dev.cuda_stream().synchronize().unwrap();

        let ops: Vec<(&'static str, Box<dyn Fn() -> Result<Tensor>>)> = vec![
            (
                "cast_to_f32",
                Box::new({
                    let x = x_bf.clone();
                    move || cast_to_f32(&x)
                }),
            ),
            (
                "cast_to_bf16",
                Box::new({
                    let x = x_f32.clone();
                    move || cast_to_bf16(&x)
                }),
            ),
            (
                "rms_no_weight_f32",
                Box::new({
                    let x = x_rows.clone();
                    move || rms_no_weight_f32(&x, 1e-6)
                }),
            ),
            (
                "add_scale_op",
                Box::new({
                    let (a, b) = (x_bf.clone(), y_bf.clone());
                    move || add_scale_op(&a, &b, 0.5)
                }),
            ),
            (
                "gelu_mul_op",
                Box::new({
                    let (a, b) = (x_bf.clone(), y_bf.clone());
                    move || gelu_mul_op(&a, &b)
                }),
            ),
            (
                "gelu_mul",
                Box::new({
                    let (a, b) = (x_bf.clone(), x_f32.clone());
                    move || gelu_mul(&a, &b)
                }),
            ),
            (
                "rope_f32",
                Box::new({
                    let (q, k, p) = (q4.clone(), k4.clone(), pos_t.clone());
                    let rope = rope;
                    move || {
                        let (qr, _kr) = rope.apply(&q, &k, &p)?;
                        Ok(qr)
                    }
                }),
            ),
        ];

        let mut refs: Vec<Vec<f32>> = Vec::new();
        for (_, f) in &ops {
            refs.push(to_f32(&f().unwrap()));
        }
        dev.cuda_stream().synchronize().unwrap();

        let raw_ctx = dev.cuda_stream().context().clone();
        let forked = raw_ctx.new_stream().unwrap();
        let ctx = dev.cuda_stream().context().clone();
        if ctx.is_event_tracking() {
            unsafe { ctx.disable_event_tracking() };
            dev.cuda_stream().synchronize().unwrap();
        }

        for ((name, f), r) in ops.iter().zip(refs.iter()) {
            let mut runner = nv_kernels::graph::CudaGraphRunner::new(forked.clone());
            let mut out: Option<Tensor> = None;
            runner
                .run(1u64, |s| {
                    nv_layers::cuda_stream::with_stream(s.clone(), || {
                        out = Some(f()?);
                        Ok(())
                    })
                })
                .unwrap();
            forked.synchronize().unwrap();
            let c1 = to_f32(out.as_ref().unwrap());
            runner.run(1u64, |_| Ok(())).unwrap();
            forked.synchronize().unwrap();
            let c2 = to_f32(out.as_ref().unwrap());
            eprintln!(
                "{name}: capture maxdiff {:e} replay maxdiff {:e}",
                maxdiff(r, &c1),
                maxdiff(r, &c2)
            );
        }

        crate::gemma4_batch_graph::graph_teardown::GraphTeardown::new(&forked).run(|| {});
    }

    #[test]
    fn capture_op_chains() {
        let _g = capture_lock();
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("skip: no cuda");
            return;
        };
        let dev = match &device {
            Device::Cuda(d) => d.clone(),
            _ => return,
        };
        let h = 2048usize;
        let xs: Vec<f32> = (0..h).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect();
        let x_bf = Tensor::from_vec(xs.clone(), (1, h), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let ws: Vec<f32> = (0..h * h)
            .map(|i| ((i as f32) * 0.013).sin() * 0.02)
            .collect();
        let w = Tensor::from_vec(ws, (h, h), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let lin = Linear::new(w, None).unwrap();
        let nw: Vec<f32> = (0..h)
            .map(|i| 1.0 + ((i as f32) * 0.05).cos() * 0.1)
            .collect();
        let norm = RmsNorm::new(Tensor::from_vec(nw, h, &device).unwrap(), 1e-6);
        dev.cuda_stream().synchronize().unwrap();

        let chains: Vec<(&'static str, Box<dyn Fn() -> Result<Tensor> + '_>)> = vec![
            ("lin", Box::new(|| lin.forward(&x_bf))),
            ("norm", Box::new(|| norm.forward(&x_bf))),
            ("lin>norm", Box::new(|| norm.forward(&lin.forward(&x_bf)?))),
            ("norm>lin", Box::new(|| lin.forward(&norm.forward(&x_bf)?))),
            (
                "lin>norm>add",
                Box::new(|| {
                    let a = norm.forward(&lin.forward(&x_bf)?)?;
                    add_scale_op(&x_bf, &a, 1.0)
                }),
            ),
            (
                "lin>norm>add>norm>lin",
                Box::new(|| {
                    let a = norm.forward(&lin.forward(&x_bf)?)?;
                    let s = add_scale_op(&x_bf, &a, 1.0)?;
                    lin.forward(&norm.forward(&s)?)
                }),
            ),
            (
                "gelu(lin,lin)>lin",
                Box::new(|| {
                    let g = lin.forward(&x_bf)?;
                    let u = lin.forward(&x_bf)?;
                    lin.forward(&gelu_mul_op(&g, &u)?)
                }),
            ),
        ];

        let mut refs: Vec<Vec<f32>> = Vec::new();
        for (_, f) in &chains {
            refs.push(to_f32(&f().unwrap()));
        }
        dev.cuda_stream().synchronize().unwrap();

        let raw_ctx = dev.cuda_stream().context().clone();
        let forked = raw_ctx.new_stream().unwrap();
        let ctx = dev.cuda_stream().context().clone();
        if ctx.is_event_tracking() {
            unsafe { ctx.disable_event_tracking() };
            dev.cuda_stream().synchronize().unwrap();
        }

        for ((name, f), r) in chains.iter().zip(refs.iter()) {
            let mut runner = nv_kernels::graph::CudaGraphRunner::new(forked.clone());
            let mut out: Option<Tensor> = None;
            runner
                .run(1u64, |s| {
                    nv_layers::cuda_stream::with_stream(s.clone(), || {
                        out = Some(f()?);
                        Ok(())
                    })
                })
                .unwrap();
            forked.synchronize().unwrap();
            let c1 = to_f32(out.as_ref().unwrap());
            runner.run(1u64, |_| Ok(())).unwrap();
            forked.synchronize().unwrap();
            let c2 = to_f32(out.as_ref().unwrap());
            eprintln!(
                "{name}: capture maxdiff {:e} replay maxdiff {:e}",
                maxdiff(r, &c1),
                maxdiff(r, &c2)
            );
        }

        crate::gemma4_batch_graph::graph_teardown::GraphTeardown::new(&forked).run(|| {});
    }
}

impl CausalLm for Gemma4E4b {
    fn forward(&mut self, tokens: &[u32], _positions: &[u32]) -> Result<Vec<f32>> {
        Ok(self.trace(tokens)?.logits_last)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}
