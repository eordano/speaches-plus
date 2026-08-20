use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_weights::WeightLoader;

use crate::gemma4::{Gemma4Config, LayerType};
use crate::CausalLm;

const PREFIX: &str = "model.language_model";

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
    anyhow::ensure!(
        weights.has(&name(&packed)),
        "no .weight or .weight_packed for {suffix}"
    );
    #[cfg(feature = "cuda")]
    if matches!(device, Device::Cuda(_)) {
        let (pbytes, scale, out, inp) = read_packed_parts(weights, suffix)?;
        return pick_quant_backend(&pbytes, scale, out, inp, device);
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
    anyhow::ensure!(suffixes.len() >= 2, "merge needs >= 2 linears");
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
    if matches!(device, Device::Cuda(_)) {
        let mut p: Vec<u8> = Vec::new();
        let mut scales: Vec<Tensor> = Vec::new();
        let mut out = 0usize;
        let mut inp = 0usize;
        for sfx in suffixes {
            let (pb, sb, ob, ib) = read_packed_parts(weights, sfx)?;
            if inp == 0 {
                inp = ib;
            }
            anyhow::ensure!(ib == inp, "merged linears must share in_features");
            p.extend_from_slice(&pb);
            scales.push(sb);
            out += ob;
        }
        let refs: Vec<&Tensor> = scales.iter().collect();
        let scale = Tensor::cat(&refs, 0)?;
        return pick_quant_backend(&p, scale, out, inp, device);
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
    anyhow::ensure!(
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
) -> Result<QLinear> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CHOICE: OnceLock<Mutex<HashMap<(usize, usize, usize), bool>>> = OnceLock::new();

    let gs = inp / scale.dim(1)?;
    let marlin_supported = inp % 16 == 0 && out % 64 == 0 && inp % 32 == 0;
    if std::env::var("NV_E4B_FORCE_GEMV").is_ok() || !marlin_supported {
        return Ok(QLinear::Quant(W4a16Linear::from_raw(
            pbytes, scale, out, inp, device,
        )?));
    }
    if std::env::var("NV_E4B_FORCE_MARLIN").is_ok() {
        return Ok(QLinear::Marlin(MarlinLinear::from_raw(
            pbytes, scale, out, inp, device,
        )?));
    }

    let large_gemv = std::env::var("NV_E4B_GEMV_LARGE")
        .map(|v| v != "0")
        .unwrap_or(true);
    if large_gemv && gs == 32 {
        let variant = if inp == 2560 && out == 20480 {
            Some(0)
        } else if inp == 10240 && out == 2560 {
            Some(6)
        } else {
            None
        };
        if let Some(variant) = variant {
            return Ok(QLinear::GemvM1(GemvM1Hybrid {
                gemv: W4a16Linear::from_raw(pbytes, scale.clone(), out, inp, device)?,
                marlin: MarlinLinear::from_raw(pbytes, scale, out, inp, device)?,
                variant,
                y_persist: std::sync::OnceLock::new(),
            }));
        }
    }

    if std::env::var("NV_E4B_GEMV_SMALL").is_ok() && inp <= 3072 && out <= 4096 {
        return Ok(QLinear::Quant(W4a16Linear::from_raw(
            pbytes, scale, out, inp, device,
        )?));
    }

    if std::env::var("NV_E4B_TINY_GEMV").is_ok() && (inp <= 512 || out <= 512) {
        return Ok(QLinear::Quant(W4a16Linear::from_raw(
            pbytes, scale, out, inp, device,
        )?));
    }

    if std::env::var("NV_E4B_AUTOTUNE").is_err() {
        return Ok(QLinear::Marlin(MarlinLinear::from_raw(
            pbytes, scale, out, inp, device,
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
    let dev = match device {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("bench_forward requires cuda"),
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    for _ in 0..3 {
        let _ = f()?;
    }
    stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
    let t0 = std::time::Instant::now();
    for _ in 0..20 {
        let _ = f()?;
    }
    stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
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
    anyhow::ensure!(
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
}

#[cfg(feature = "cuda")]
fn persist_bf16_row(
    lock: &std::sync::OnceLock<(Tensor, u64)>,
    elems: usize,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    dev: &candle_core::CudaDevice,
) -> Result<(Tensor, u64)> {
    use cudarc::driver::DevicePtrMut;
    use half::bf16;
    if let Some((t, p)) = lock.get() {
        return Ok((t.clone(), *p));
    }
    let mut buf: cudarc::driver::CudaSlice<bf16> = unsafe {
        stream
            .alloc::<bf16>(elems)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let ptr = {
        let (p, _g) = buf.device_ptr_mut(stream);
        p as u64
    };
    let storage = candle_core::CudaStorage::wrap_cuda_slice(buf, dev.clone());
    let storage = candle_core::Storage::Cuda(storage);
    let t = Tensor::from_storage(
        storage,
        (1usize, elems),
        candle_core::op::BackpropOp::none(),
        false,
    );
    let _ = lock.set((t, ptr));
    let (t, p) = lock.get().unwrap();
    Ok((t.clone(), *p))
}

#[cfg(feature = "cuda")]
impl GemvM1Hybrid {
    fn persist_y(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        dev: &candle_core::CudaDevice,
    ) -> Result<(Tensor, u64)> {
        persist_bf16_row(&self.y_persist, self.gemv.out_features, stream, dev)
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
        anyhow::ensure!(kin == self.gemv.in_features, "gemv_m1 in mismatch");
        let dev = match &self.gemv.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("gemv_m1 requires cuda"),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let (yt, yp) = self.persist_y(&stream, &dev)?;
        let x2 = x.reshape((1, kin))?.to_dtype(DType::BF16)?.contiguous()?;
        let rc = {
            let (xs, _) = x2.storage_and_layout();
            let (ps, _) = self.gemv.packed.storage_and_layout();
            let (ss, _) = self.gemv.scale.storage_and_layout();
            let xc = match &*xs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("gemv_m1: x not cuda"),
            };
            let pc = match &*ps {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("gemv_m1: packed not cuda"),
            };
            let sc = match &*ss {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("gemv_m1: scale not cuda"),
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
        anyhow::ensure!(rc == 0, "gemv_w4a16_m1 returned {rc}");
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
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_kernels::cuda as nvk;

        let dims = x.dims().to_vec();
        let kin = *dims.last().unwrap();
        anyhow::ensure!(kin == self.in_features, "w4a16 in mismatch");
        let m: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((m, kin))?.to_dtype(DType::BF16)?.contiguous()?;
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("w4a16 requires cuda"),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let mut y_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(m * self.out_features)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        {
            let (xs, _) = x2.storage_and_layout();
            let (ps, _) = self.packed.storage_and_layout();
            let (ss, _) = self.scale.storage_and_layout();
            let xc = match &*xs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("w4a16: x not cuda"),
            };
            let pc = match &*ps {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("w4a16: packed not cuda"),
            };
            let sc = match &*ss {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("w4a16: scale not cuda"),
            };
            let xsl = xc.as_cuda_slice::<bf16>()?;
            let psl = pc.as_cuda_slice::<u32>()?;
            let ssl = sc.as_cuda_slice::<bf16>()?;
            let (xp, _gx) = xsl.device_ptr(&stream);
            let (pp, _gp) = psl.device_ptr(&stream);
            let (sp, _gs2) = ssl.device_ptr(&stream);
            let (yp, _gy) = y_dev.device_ptr_mut(&stream);
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
                    anyhow::bail!("gemv_w4a16 returned {rc}");
                }
            }
        }
        let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
        let storage = candle_core::Storage::Cuda(storage);
        let out = candle_core::Tensor::from_storage(
            storage,
            (m, self.out_features),
            candle_core::op::BackpropOp::none(),
            false,
        );
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(self.out_features);
        out.reshape(out_dims).map_err(Into::into)
    }
}

#[cfg(feature = "cuda")]
impl W4a16Linear {
    fn forward_gelu_pli(&self, x: &Tensor, pli: &Tensor) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        anyhow::ensure!(
            self.in_features <= 3072 && self.group_size >= 32,
            "fused pli shape"
        );
        let dims = x.dims().to_vec();
        anyhow::ensure!(
            *dims.last().unwrap() == self.in_features,
            "fused pli in mismatch"
        );
        let m: usize = dims[..dims.len() - 1].iter().product();
        anyhow::ensure!(m == 1, "fused pli is M=1 only");
        anyhow::ensure!(pli.elem_count() == self.out_features, "pli numel mismatch");
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("fused pli requires cuda"),
        };
        let x2 = x
            .reshape((1, self.in_features))?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let mut y_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(self.out_features)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let rc = {
            let (xs, _) = x2.storage_and_layout();
            let (ps, _) = self.packed.storage_and_layout();
            let (ss, _) = self.scale.storage_and_layout();
            let (pls, pll) = pli.storage_and_layout();
            let (p0, p1) = pll
                .contiguous_offsets()
                .ok_or_else(|| anyhow::anyhow!("pli not dense"))?;
            let xc = match &*xs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("x not cuda"),
            };
            let pc = match &*ps {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("packed not cuda"),
            };
            let sc = match &*ss {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("scale not cuda"),
            };
            let plc = match &*pls {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("pli not cuda"),
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
            let (yp, _g5) = y_dev.device_ptr_mut(&stream);
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
        anyhow::ensure!(rc == 0, "gemv_w4a16_gelu_pli returned {rc}");
        let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
        let storage = candle_core::Storage::Cuda(storage);
        Ok(candle_core::Tensor::from_storage(
            storage,
            (1usize, self.out_features),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(feature = "cuda")]
struct MarlinLinear {
    b_q_marlin: cudarc::driver::CudaSlice<i32>,
    b_scales: Tensor,
    workspace: cudarc::driver::CudaSlice<i32>,

    c_tmp: Option<std::sync::Arc<cudarc::driver::CudaSlice<f32>>>,
    c_persist: std::sync::OnceLock<(Tensor, u64)>,
    out_features: usize,
    in_features: usize,
    group_size: usize,
    device: candle_core::CudaDevice,
}

#[cfg(feature = "cuda")]
struct MarlinZeroRegistry {
    entries: Vec<u64>,
    dev: Option<cudarc::driver::CudaSlice<u64>>,
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
fn marlin_zero_upload(stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> Result<()> {
    use cudarc::driver::DevicePtrMut;
    let mut reg = marlin_zero_registry().lock().unwrap();
    if reg.version == reg.uploaded_version {
        return Ok(());
    }
    if reg.dev.is_none() {
        let mut dev: cudarc::driver::CudaSlice<u64> = stream
            .alloc_zeros::<u64>(2 * MARLIN_ZERO_CAP)
            .map_err(|e| anyhow::anyhow!(e))?;
        let ptr = {
            let (p, _g) = dev.device_ptr_mut(stream);
            p as u64
        };
        reg.dev = Some(dev);
        reg.dev_ptr = ptr;
    }
    let mut host = reg.entries.clone();
    host.resize(2 * MARLIN_ZERO_CAP, 0);
    let count = (reg.entries.len() / 2) as i32;
    let version = reg.version;
    stream
        .memcpy_htod(&host, reg.dev.as_mut().unwrap())
        .map_err(|e| anyhow::anyhow!(e))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
    reg.uploaded_version = version;
    reg.uploaded_count = count;
    Ok(())
}

#[cfg(feature = "cuda")]
fn marlin_zero_launch(stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> Result<()> {
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
    anyhow::ensure!(rc == 0, "multi_zero_bf16 returned {rc}");
    Ok(())
}

#[cfg(feature = "cuda")]
fn marlin_c_tmp_scratch(
    n: usize,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<std::sync::Arc<cudarc::driver::CudaSlice<f32>>> {
    use std::sync::{Arc, Mutex, OnceLock};
    static SCRATCH: OnceLock<Mutex<Option<(usize, Arc<cudarc::driver::CudaSlice<f32>>)>>> =
        OnceLock::new();
    let elems = 64 * n;
    let cell = SCRATCH.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap();
    if let Some((cap, buf)) = guard.as_ref() {
        if *cap >= elems {
            return Ok(buf.clone());
        }
    }
    let buf: Arc<cudarc::driver::CudaSlice<f32>> = Arc::new(
        stream
            .alloc_zeros::<f32>(elems)
            .map_err(|e| anyhow::anyhow!(e))?,
    );
    *guard = Some((elems, buf.clone()));
    Ok(buf)
}

#[cfg(feature = "cuda")]
impl MarlinLinear {
    fn from_raw(
        pbytes: &[u8],
        scale: Tensor,
        out: usize,
        inp: usize,
        device: &Device,
    ) -> Result<Self> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("marlin requires cuda"),
        };
        let group_size = inp / scale.dim(1)?;

        let s = scale.t()?.contiguous()?;
        let ng = s.dim(0)?;
        let n = s.dim(1)?;
        anyhow::ensure!((ng * n) % 64 == 0, "marlin scale reshape: {ng}x{n} not %64");
        let mut perm = Vec::with_capacity(64);
        for i in 0..8u32 {
            for j in 0..8u32 {
                perm.push(i + 8 * j);
            }
        }
        let perm_t = Tensor::from_vec(perm, 64, device)?;
        let b_scales = s
            .reshape((ng * n / 64, 64))?
            .index_select(&perm_t, 1)?
            .reshape((ng, n))?
            .contiguous()?;

        let cols = inp / 8;
        anyhow::ensure!(pbytes.len() == out * cols * 4, "packed size mismatch");
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
        let b_q_packed: cudarc::driver::CudaSlice<i32> =
            stream.clone_htod(&tvec).map_err(|e| anyhow::anyhow!(e))?;
        let mut b_q_marlin: cudarc::driver::CudaSlice<i32> = unsafe {
            stream
                .alloc::<i32>(inp * out / 8)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let rc = {
            let (pp, _gp) = b_q_packed.device_ptr(&stream);
            let (mp, _gm) = b_q_marlin.device_ptr_mut(&stream);
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
        };
        anyhow::ensure!(rc == 0, "marlin_repack returned {rc}");
        drop(b_q_packed);

        let mut elems: i32 = 0;
        unsafe {
            nv_kernels::cuda::marlin_workspace_elems(&mut elems as *mut i32);
        }
        anyhow::ensure!(elems > 0, "marlin workspace elems = {elems}");
        let workspace: cudarc::driver::CudaSlice<i32> = stream
            .clone_htod(&vec![0i32; elems as usize])
            .map_err(|e| anyhow::anyhow!(e))?;

        let c_tmp = if std::env::var("NV_E4B_MARLIN_FP32RED").is_ok() {
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
        let mut buf: cudarc::driver::CudaSlice<bf16> = stream
            .alloc_zeros::<bf16>(self.out_features)
            .map_err(|e| anyhow::anyhow!(e))?;
        let ptr = {
            let (p, _g) = buf.device_ptr_mut(stream);
            p as u64
        };
        if !marlin_zero_register(ptr, self.out_features as u64) {
            return Ok(None);
        }
        let storage = candle_core::CudaStorage::wrap_cuda_slice(buf, self.device.clone());
        let storage = candle_core::Storage::Cuda(storage);
        let t = Tensor::from_storage(
            storage,
            (1usize, self.out_features),
            candle_core::op::BackpropOp::none(),
            false,
        );
        let _ = self.c_persist.set((t, ptr));
        let (t, p) = self.c_persist.get().unwrap();
        Ok(Some((t.clone(), *p)))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        let dims = x.dims().to_vec();
        let kin = *dims.last().unwrap();
        anyhow::ensure!(kin == self.in_features, "marlin in mismatch");
        let m: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((m, kin))?.to_dtype(DType::BF16)?.contiguous()?;
        let stream = nv_layers::cuda_stream::current_stream(&self.device);
        if m == 1 && self.c_tmp.is_none() {
            if let Some((t, cptr)) = self.persist_out(&stream)? {
                let rc = {
                    let (xs, _xl) = x2.storage_and_layout();
                    let (ss, _sl) = self.b_scales.storage_and_layout();
                    let xc = match &*xs {
                        candle_core::Storage::Cuda(s) => s,
                        _ => anyhow::bail!("marlin: x not cuda"),
                    };
                    let sc = match &*ss {
                        candle_core::Storage::Cuda(s) => s,
                        _ => anyhow::bail!("marlin: scales not cuda"),
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
                anyhow::ensure!(rc == 0, "marlin_gemm returned {rc}");
                let mut out_dims = dims[..dims.len() - 1].to_vec();
                out_dims.push(self.out_features);
                return t.reshape(out_dims).map_err(Into::into);
            }
        }
        let mut c_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(m * self.out_features)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let rc = {
            let (xs, _xl) = x2.storage_and_layout();
            let (ss, _sl) = self.b_scales.storage_and_layout();
            let xc = match &*xs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("marlin: x not cuda"),
            };
            let sc = match &*ss {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("marlin: scales not cuda"),
            };
            let xsl = xc.as_cuda_slice::<bf16>()?;
            let ssl = sc.as_cuda_slice::<bf16>()?;
            let (xp, _gx) = xsl.device_ptr(&stream);
            let (sp, _gs) = ssl.device_ptr(&stream);
            let (bp, _gb) = self.b_q_marlin.device_ptr(&stream);
            let (wp, _gw) = self.workspace.device_ptr(&stream);
            let (cp, _gc) = c_dev.device_ptr_mut(&stream);
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
        anyhow::ensure!(rc == 0, "marlin_gemm returned {rc}");
        let storage = candle_core::CudaStorage::wrap_cuda_slice(c_dev, self.device.clone());
        let storage = candle_core::Storage::Cuda(storage);
        let out = candle_core::Tensor::from_storage(
            storage,
            (m, self.out_features),
            candle_core::op::BackpropOp::none(),
            false,
        );
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
    lm_head_i8: Option<(
        cudarc::driver::CudaSlice<i8>,
        cudarc::driver::CudaSlice<f32>,
    )>,
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
        anyhow::ensure!(
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
            if std::env::var("NV_E4B_LMHEAD_INT8").is_ok() && matches!(device, Device::Cuda(_)) {
                Some(quantize_lm_head_i8(&embed_tokens)?)
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
        Ok(Self {
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
        })
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
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");
        #[cfg(feature = "cuda")]
        if matches!(self.device, Device::Cuda(_)) && e4b_kv_fp8_enabled() {
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
        let n = new_ids.len();
        let mut kv = E4bKvMut::Cat(cache);
        let (logits, _hidden) = self.forward_step_inner(new_ids, None, past_len, &mut kv)?;
        logits
            .narrow(0, n - 1, 1)?
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
        let n = new_ids.len();
        let mut kv = E4bKvMut::Cat(cache);
        let (logits, _hidden) = self.forward_step_inner(new_ids, Some(embeds), past_len, &mut kv)?;
        logits
            .narrow(0, n - 1, 1)?
            .reshape(self.config.vocab_size)
            .map_err(Into::into)
    }

    pub fn forward_step_spec(
        &self,
        new_ids: &[u32],
        past_len: usize,
        cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<(Tensor, Tensor)> {
        let mut kv = E4bKvMut::Cat(cache);
        self.forward_step_inner(new_ids, None, past_len, &mut kv)
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
        let (logits, _hidden) = self.forward_step_inner(new_ids, None, past_len, &mut kv)?;
        logits
            .narrow(0, n - 1, 1)?
            .reshape(self.config.vocab_size)
            .map_err(Into::into)
    }

    #[cfg(feature = "cuda")]
    pub fn generate_fp8(&self, prompt: &[u32], max_new: usize, eos: &[u32]) -> Result<Vec<u32>> {
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");
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
    ) -> Result<(Tensor, Tensor)> {
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
                anyhow::ensure!(
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
        let pos: Vec<u32> = (past_len..past_len + n).map(|x| x as u32).collect();

        let total = past_len + n;
        let full_mask = cache_mask(n, total, past_len, None, &self.device)?;
        let sliding_mask = cache_mask(
            n,
            total,
            past_len,
            Some(self.config.sliding_window),
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
        let xn = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&xn)?;
        Ok((logits.reshape((n, self.config.vocab_size))?, hidden_pre))
    }

    pub fn embed_table(&self) -> &Tensor {
        &self.embed_tokens
    }

    pub fn embed_normalizer(&self) -> f64 {
        self.normalizer
    }

    pub fn embed_scaled_row(&self, token: u32) -> Result<Tensor> {
        let ids = Tensor::from_vec(vec![token], 1, &self.device)?;
        let e = self.embed_tokens.index_select(&ids, 0)?;
        (e.to_dtype(DType::F32)? * self.normalizer)?
            .reshape(self.config.hidden_size)
            .map_err(Into::into)
    }

    pub fn spec_shared_kv(
        &self,
        cache: &[Option<(Tensor, Tensor)>],
    ) -> Result<((Tensor, Tensor), (Tensor, Tensor))> {
        let mut sl = None;
        let mut fl = None;
        for (i, l) in self.layers.iter().enumerate() {
            if l.k_norm.is_some() {
                match l.kind {
                    LayerType::SlidingAttention => sl = Some(i),
                    LayerType::FullAttention => fl = Some(i),
                }
            }
        }
        let view = |i: usize| -> Result<(Tensor, Tensor)> {
            let (k, v) = cache[i].as_ref().context("shared kv missing from cache")?;
            Ok((
                k.squeeze(0)?.transpose(0, 1)?.contiguous()?,
                v.squeeze(0)?.transpose(0, 1)?.contiguous()?,
            ))
        };
        Ok((
            view(sl.context("no sliding kv-writing layer")?)?,
            view(fl.context("no full kv-writing layer")?)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
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
        let attn = layer.post_attn_ln.forward(&attn.reshape((n, hidden))?)?;
        let h = add_scale_op(&h0, &attn, 1.0)?;

        let normed2 = layer.pre_ff_ln.forward(&h)?;
        let act = geglu_fused_op(&layer.gate_up_proj.forward(&normed2)?)?;
        let mlp = layer.post_ff_ln.forward(&layer.down_proj.forward(&act)?)?;
        let h = add_scale_op(&h, &mlp, 1.0)?;

        let gate = layer.per_layer_input_gate.forward(&h)?.gelu()?;
        let gated = (gate.to_dtype(DType::F32)? * per_layer_input)?.to_dtype(self.dtype)?;
        let contrib = layer.per_layer_projection.forward(&gated)?;
        let contrib = layer.post_per_layer_input_norm.forward(&contrib)?;

        let h = add_scale_op(&h, &contrib, layer.layer_scalar as f64)?;

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
        let q = qkv
            .narrow(D::Minus1, 0, nh * hd)?
            .contiguous()?
            .reshape((1, n, nh, hd))?;
        let q = layer.q_norm.forward(&q)?.to_dtype(DType::F32)?;

        let (k_all, v_all, q_rot) = match &layer.k_norm {
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
                    E4bKvMut::Cat(cache) => cache[src].clone().context("kv source not in cache")?,
                    #[cfg(feature = "cuda")]
                    E4bKvMut::Fp8(cache) => {
                        let total =
                            *pos.last().context("attn_cached: empty positions")? as usize + 1;
                        cache.view_f32(src, total)?
                    }
                };
                (k, v, q_rot)
            }
        };

        #[cfg(feature = "cuda")]
        if n == 1 && matches!(self.device, Device::Cuda(_)) {
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
            return layer.o_proj.forward(&out).map_err(Into::into);
        }

        let group = nh / layer.n_kv_heads;
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
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("generate_graphed requires cuda"),
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
        'gen: while out.len() < max_new && !eos.contains(out.last().unwrap()) {
            let k = graph_batch.min(max_new - out.len());
            let toks = dec.replay_batch(*out.last().unwrap(), k)?;
            for t in toks {
                out.push(t);
                if out.len() >= max_new || eos.contains(&t) {
                    break 'gen;
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
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("graphed_decoder requires cuda"),
        };
        let fixed = E4bFixedCache::from_cat(cat, past_len, max_len, &dev)?;
        GraphedE4bDecoder::new(self, fixed, past_len)
    }

    #[cfg(feature = "cuda")]
    pub fn generate_fast(&self, prompt: &[u32], max_new: usize, eos: &[u32]) -> Result<Vec<u32>> {
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("generate_fast requires cuda"),
        };
        let mut cat: Vec<Option<(Tensor, Tensor)>> = vec![None; self.layers.len()];
        let mut logits = self.forward_step(prompt, 0, &mut cat)?;
        let max_len = (prompt.len() + max_new + 8).next_power_of_two().max(64);
        let mut fixed = E4bFixedCache::from_cat(&cat, prompt.len(), max_len, &dev)?;
        let mut out = Vec::new();
        let mut total = prompt.len();
        for _ in 0..max_new {
            let next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            out.push(next);
            if eos.contains(&next) {
                break;
            }
            logits = self.forward_step_fast(next, total, &mut fixed)?;
            total += 1;
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
        let emb = gather_rows_bf16_op(&self.embed_tokens, ids)?;
        let embeds = scale_bf16_op(&emb, self.normalizer as f32)?.reshape((1, 1, hidden))?;
        let pli_all = self.per_layer_inputs_fast(ids, &embeds)?;
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("cuda only"),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);

        anyhow::ensure!(pos_t.dtype() == DType::I32, "pos_t must be i32");
        let rc = {
            let pos_ptr = {
                let (p, _g) = fixed.pos.device_ptr_mut(&stream);
                p as *mut i32
            };
            let (ps, _pl) = pos_t.storage_and_layout();
            let pc = match &*ps {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("pos_t not cuda"),
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
        anyhow::ensure!(rc == 0, "incr_pos_rope returned {rc}");
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

        match &self.lm_head_i8 {
            Some((wq, rs)) => {
                lm_head_i8_normed_op(wq, rs, &x, self.norm.weight_bf16(), &rstd, &dev)
            }
            None => lm_head_normed_op(&self.embed_tokens, &x, self.norm.weight_bf16(), &rstd),
        }
    }

    #[cfg(feature = "cuda")]
    fn per_layer_inputs_fast_mk(
        &self,
        ids: &Tensor,
        inputs_embeds: &Tensor,
        m: usize,
    ) -> Result<Tensor> {
        let n_layers = self.config.num_hidden_layers;
        let hpl = self.config.hidden_size_per_layer_input;

        let ple_bf = gather_rows_bf16_op(&self.embed_tokens_per_layer, ids)?;
        let ple = cast_scale_f32_op(&ple_bf, self.embed_scale_per_layer as f32)?;

        let proj_bf = self
            .per_layer_model_projection
            .forward(&inputs_embeds.reshape((m, self.config.hidden_size))?)?;
        let proj = cast_scale_f32_op(
            &proj_bf.reshape((m, n_layers * hpl))?,
            self.per_layer_proj_scale as f32,
        )?;
        let proj = self
            .per_layer_projection_norm
            .forward(&proj.reshape((m * n_layers, hpl))?)?;

        add_scale_f32_op(
            &proj.reshape((m, n_layers, hpl))?,
            &ple.reshape((m, n_layers, hpl))?,
            self.per_layer_input_scale as f32,
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn forward_step_fast_mk(
        &self,
        ids: &Tensor,
        pos_t: &Tensor,
        fixed: &mut E4bFixedCache,
        m: usize,
    ) -> Result<(Tensor, Tensor)> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let hidden = self.config.hidden_size;
        if let Device::Cuda(d) = &self.device {
            marlin_zero_launch(&nv_layers::cuda_stream::current_stream(d))?;
        }
        let emb = gather_rows_bf16_op(&self.embed_tokens, ids)?;
        let embeds = scale_bf16_op(&emb, self.normalizer as f32)?.reshape((1, m, hidden))?;
        let pli_all = self.per_layer_inputs_fast_mk(ids, &embeds, m)?;
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("cuda only"),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);

        anyhow::ensure!(pos_t.dtype() == DType::I32, "pos_t must be i32");
        for _ in 0..m {
            let rc = {
                let pos_ptr = {
                    let (p, _g) = fixed.pos.device_ptr_mut(&stream);
                    p as *mut i32
                };
                let (ps, _pl) = pos_t.storage_and_layout();
                let pc = match &*ps {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("pos_t not cuda"),
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
            anyhow::ensure!(rc == 0, "incr_pos_rope returned {rc}");
        }

        let mut x = embeds.clone();
        let mut rstd = rstd_op(
            &embeds.reshape((m, hidden))?,
            self.config.rms_norm_eps as f32,
        )?;
        let mut normed: Option<Tensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let pli = pli_all
                .narrow(1, i, 1)?
                .reshape((m, self.config.hidden_size_per_layer_input))?
                .contiguous()?;
            let next_w = self.layers.get(i + 1).map(|l| l.input_ln.weight_bf16());
            let (nx, nrstd, nnormed) = self.layer_step_fast_mk(
                layer,
                &x,
                i,
                pos_t,
                &pli,
                &rstd,
                normed.as_ref(),
                next_w,
                fixed,
                m,
            )?;
            x = nx;
            rstd = nrstd;
            normed = nnormed;
        }

        let hidden_pre = x.reshape((m, hidden))?;
        let logits = match &self.lm_head_i8 {
            Some((wq, rs)) => lm_head_i8_normed_mk_op(
                wq,
                rs,
                &hidden_pre,
                self.norm.weight_bf16(),
                &rstd,
                m,
                &dev,
            )?,
            None => {
                anyhow::bail!("forward_step_fast_mk requires NV_E4B_LMHEAD_INT8=1")
            }
        };
        Ok((logits, hidden_pre))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn layer_step_fast_mk(
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
        m: usize,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let hidden = self.config.hidden_size;
        let nh = self.config.num_attention_heads;
        let hd = layer.head_dim;
        let nkv = layer.n_kv_heads;
        let window = match layer.kind {
            LayerType::SlidingAttention => self.config.sliding_window,
            LayerType::FullAttention => 0,
        };
        let eps = self.config.rms_norm_eps as f32;
        let rope = match layer.kind {
            LayerType::SlidingAttention => &self.sliding_rope,
            LayerType::FullAttention => &self.full_rope,
        };
        let h0 = x.reshape((m, hidden))?;
        let normed = match normed_in {
            Some(n) => n.clone(),
            None => rms_apply_op(&h0, layer.input_ln.weight_bf16(), rstd_in)?,
        };
        let qkv = layer
            .qkv_proj
            .forward(&normed)?
            .reshape((m, ()))?
            .contiguous()?;

        let attn_rows = Tensor::zeros((m, nh * hd), DType::BF16, &self.device)?;
        for j in 0..m {
            let delta = (m - 1 - j) as i32;
            let qf = match &layer.k_norm {
                Some(kn) => fixed.qkv_prep(
                    Some(idx),
                    &qkv,
                    j,
                    layer.q_norm.weight_bf16(),
                    Some(kn.weight_bf16()),
                    rope,
                    pos_t,
                    delta,
                    nh,
                    nkv,
                    hd,
                    eps,
                )?,
                None => fixed.qkv_prep(
                    None,
                    &qkv,
                    j,
                    layer.q_norm.weight_bf16(),
                    None,
                    rope,
                    pos_t,
                    delta,
                    nh,
                    nkv,
                    hd,
                    eps,
                )?,
            };
            let buf = match &layer.k_norm {
                Some(_) => idx,
                None => layer.kv_source.context("kv-shared layer without source")?,
            };
            fixed.attend_into(buf, &qf, delta, &attn_rows, j, nh, nkv, hd, window)?;
        }
        self.attn_tail_mk(
            layer,
            &h0,
            &attn_rows,
            hidden,
            per_layer_input,
            next_input_w,
            m,
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn attn_tail_mk(
        &self,
        layer: &E4bLayer,
        h0: &Tensor,
        attn: &Tensor,
        hidden: usize,
        per_layer_input: &Tensor,
        next_input_w: Option<&Tensor>,
        m: usize,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let eps = self.config.rms_norm_eps as f32;

        let attn = layer.o_proj.forward(attn)?;
        let (h, _rstd_h1, normed2) = rmsnorm_add_scale_chain_op(
            &attn.reshape((m, hidden))?,
            &layer.post_attn_ln,
            h0,
            1.0,
            Some(eps),
            Some(layer.pre_ff_ln.weight_bf16()),
        )?;
        let normed2 = normed2.context("missing fused pre_ff apply")?;

        let act = geglu_fused_op(&layer.gate_up_proj.forward(&normed2)?)?;
        let down = layer.down_proj.forward(&act)?;
        let h = rmsnorm_add_scale_op(&down.reshape((m, hidden))?, &layer.post_ff_ln, &h, 1.0)?;

        let gated = gelu_mul(&layer.per_layer_input_gate.forward(&h)?, per_layer_input)?;
        let contrib = layer.per_layer_projection.forward(&gated)?;
        let (h, rstd_next, normed_next) = rmsnorm_add_scale_chain_op(
            &contrib.reshape((m, hidden))?,
            &layer.post_per_layer_input_norm,
            &h,
            layer.layer_scalar,
            Some(eps),
            next_input_w,
        )?;
        let rstd_next = rstd_next.context("missing chained rstd")?;
        Ok((h.reshape((1, m, hidden))?, rstd_next, normed_next))
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
    #[allow(clippy::too_many_arguments)]
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
                )
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let eps = self.config.rms_norm_eps as f32;

        let out = attn.reshape((1, 1, nh * hd))?;
        let attn = layer.o_proj.forward(&out)?;
        let (h, _rstd_h1, normed2) = rmsnorm_add_scale_chain_op(
            &attn.reshape((1, hidden))?,
            &layer.post_attn_ln,
            h0,
            1.0,
            Some(eps),
            Some(layer.pre_ff_ln.weight_bf16()),
        )?;
        let normed2 = normed2.context("missing fused pre_ff apply")?;

        let act = geglu_fused_op(&layer.gate_up_proj.forward(&normed2)?)?;
        let down = layer.down_proj.forward(&act)?;
        let h = rmsnorm_add_scale_op(&down.reshape((1, hidden))?, &layer.post_ff_ln, &h, 1.0)?;

        let gated = match layer
            .per_layer_input_gate
            .forward_gelu_pli(&h, per_layer_input)
        {
            Some(r) => r?,
            None => gelu_mul(&layer.per_layer_input_gate.forward(&h)?, per_layer_input)?,
        };
        let contrib = layer.per_layer_projection.forward(&gated)?;
        let (h, rstd_next, normed_next) = rmsnorm_add_scale_chain_op(
            &contrib.reshape((1, hidden))?,
            &layer.post_per_layer_input_norm,
            &h,
            layer.layer_scalar,
            Some(eps),
            next_input_w,
        )?;
        let rstd_next = rstd_next.context("missing chained rstd")?;
        Ok((h.reshape((1, 1, hidden))?, rstd_next, normed_next))
    }
}

#[cfg(feature = "cuda")]
pub fn e4b_kv_fp8_enabled() -> bool {
    std::env::var("NV_E4B_KV_FP8").is_ok_and(|v| v != "0")
}

enum E4bKvMut<'a> {
    Cat(&'a mut [Option<(Tensor, Tensor)>]),
    #[cfg(feature = "cuda")]
    Fp8(&'a mut E4bFp8Cache),
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
        anyhow::ensure!(
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

pub fn cat_kv_cache_bytes(config: &Gemma4Config, kv_max_seq_len: usize) -> usize {
    (0..config.num_hidden_layers)
        .filter(|i| !config.is_kv_shared_layer(*i))
        .map(|i| {
            let kind = config.layer_kind(i);
            let stride = config.num_kv_heads_for(kind) * config.head_dim_for(kind);
            kv_max_seq_len * stride * std::mem::size_of::<f32>() * 2
        })
        .sum()
}

pub fn truncate_kv_cache(cache: &mut [Option<(Tensor, Tensor)>], len: usize) -> Result<()> {
    for e in cache.iter_mut() {
        if let Some((k, v)) = e.take() {
            let cur = k.dim(1)?;
            if cur > len {
                *e = Some((k.narrow(1, 0, len)?, v.narrow(1, 0, len)?));
            } else {
                *e = Some((k, v));
            }
        }
    }
    Ok(())
}

fn cache_mask(
    n: usize,
    total: usize,
    past: usize,
    window: Option<usize>,
    device: &Device,
) -> Result<Tensor> {
    let mut data = vec![0f32; n * total];
    for r in 0..n {
        let absr = past + r;
        for c in 0..total {
            let mut masked = c > absr;
            if let Some(w) = window {
                if absr + 1 > w && c + w <= absr {
                    masked = true;
                }
            }
            if masked {
                data[r * total + c] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (n, total), device).map_err(Into::into)
}

fn rms_no_weight(x: &Tensor, eps: f64) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let ms = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let denom = (ms + eps)?.sqrt()?;
    xf.broadcast_div(&denom).map_err(Into::into)
}

fn add_scale_op(a: &Tensor, b: &Tensor, scale: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if matches!(a.device(), Device::Cuda(_)) && a.dtype() == DType::BF16 {
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
    if matches!(gate.device(), Device::Cuda(_)) && gate.dtype() == DType::BF16 {
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
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let dev = match a.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("cuda_binary_bf16 requires cuda"),
    };
    let a_c = a.contiguous()?;
    let b_c = b.contiguous()?;
    let dims = a_c.dims().to_vec();
    anyhow::ensure!(
        dims == b_c.dims(),
        "shape mismatch {:?} vs {:?}",
        dims,
        b_c.dims()
    );
    let n: usize = dims.iter().product();
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: cudarc::driver::CudaSlice<bf16> =
        unsafe { stream.alloc::<bf16>(n).map_err(|e| anyhow::anyhow!(e))? };
    let rc = {
        let (a_st, _al) = a_c.storage_and_layout();
        let (b_st, _bl) = b_c.storage_and_layout();
        let ac = match &*a_st {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("a not cuda"),
        };
        let bc = match &*b_st {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("b not cuda"),
        };
        let asl = ac.as_cuda_slice::<bf16>()?;
        let bsl = bc.as_cuda_slice::<bf16>()?;
        let (pa, _ga) = asl.device_ptr(&stream);
        let (pb, _gb) = bsl.device_ptr(&stream);
        let (py, _gy) = y_dev.device_ptr_mut(&stream);
        f(
            stream.cu_stream() as *mut _,
            pa as *const u16,
            pb as *const u16,
            py as *mut u16,
            n,
        )
    };
    anyhow::ensure!(rc == 0, "elementwise bf16 kernel returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = candle_core::Storage::Cuda(storage);
    candle_core::Tensor::from_storage(storage, dims, candle_core::op::BackpropOp::none(), false)
        .reshape(a_c.shape())
        .map_err(Into::into)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
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
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = match q.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("attn_decode_f32_op requires cuda"),
    };
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut out_dev: cudarc::driver::CudaSlice<f32> = unsafe {
        stream
            .alloc::<f32>(nh * hd)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let rc = {
        let (qs, _ql) = q.storage_and_layout();
        let (ks, _kl) = k.storage_and_layout();
        let (vs, _vl) = v.storage_and_layout();
        let qc = match &*qs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("q not cuda"),
        };
        let kc = match &*ks {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("k not cuda"),
        };
        let vc = match &*vs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("v not cuda"),
        };
        let qsl = qc.as_cuda_slice::<f32>()?;
        let ksl = kc.as_cuda_slice::<f32>()?;
        let vsl = vc.as_cuda_slice::<f32>()?;
        let (qp, _gq) = qsl.device_ptr(&stream);
        let (kp, _gk) = ksl.device_ptr(&stream);
        let (vp, _gv) = vsl.device_ptr(&stream);
        let (op, _go) = out_dev.device_ptr_mut(&stream);
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
    anyhow::ensure!(rc == 0, "attn_decode_f32 returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(out_dev, dev);
    let storage = candle_core::Storage::Cuda(storage);
    candle_core::Tensor::from_storage(
        storage,
        (nh, hd),
        candle_core::op::BackpropOp::none(),
        false,
    )
    .reshape((nh, hd))
    .map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn cast_to_f32(x: &Tensor) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("cast_to_f32 requires cuda"),
    };
    anyhow::ensure!(x.dtype() == DType::BF16, "cast_to_f32 expects bf16");
    let x_c = x.contiguous()?;
    let dims = x_c.dims().to_vec();
    let n: usize = dims.iter().product();
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: cudarc::driver::CudaSlice<f32> =
        unsafe { stream.alloc::<f32>(n).map_err(|e| anyhow::anyhow!(e))? };
    let rc = {
        let (xs, _xl) = x_c.storage_and_layout();
        let xc = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("cast_to_f32: x not cuda"),
        };
        let xsl = xc.as_cuda_slice::<half::bf16>()?;
        let (xp, _gx) = xsl.device_ptr(&stream);
        let (yp, _gy) = y_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::cast_bf16_f32(
                stream.cu_stream() as *mut _,
                xp as *const u16,
                yp as *mut f32,
                n as i32,
            )
        }
    };
    anyhow::ensure!(rc == 0, "cast_bf16_f32 returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = candle_core::Storage::Cuda(storage);
    candle_core::Tensor::from_storage(storage, dims, candle_core::op::BackpropOp::none(), false)
        .reshape(x_c.shape())
        .map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn cast_to_bf16(x: &Tensor) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("cast_to_bf16 requires cuda"),
    };
    anyhow::ensure!(x.dtype() == DType::F32, "cast_to_bf16 expects f32");
    let x_c = x.contiguous()?;
    let dims = x_c.dims().to_vec();
    let n: usize = dims.iter().product();
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: cudarc::driver::CudaSlice<bf16> =
        unsafe { stream.alloc::<bf16>(n).map_err(|e| anyhow::anyhow!(e))? };
    let rc = {
        let (xs, _xl) = x_c.storage_and_layout();
        let xc = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("cast_to_bf16: x not cuda"),
        };
        let xsl = xc.as_cuda_slice::<f32>()?;
        let (xp, _gx) = xsl.device_ptr(&stream);
        let (yp, _gy) = y_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::cast_f32_bf16(
                stream.cu_stream() as *mut _,
                xp as *const f32,
                yp as *mut u16,
                n as i32,
            )
        }
    };
    anyhow::ensure!(rc == 0, "cast_f32_bf16 returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = candle_core::Storage::Cuda(storage);
    candle_core::Tensor::from_storage(storage, dims, candle_core::op::BackpropOp::none(), false)
        .reshape(x_c.shape())
        .map_err(Into::into)
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn rms_no_weight_f32(x: &Tensor, eps: f64) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("rms_no_weight_f32 requires cuda"),
    };
    anyhow::ensure!(x.dtype() == DType::BF16, "rms_no_weight_f32 expects bf16");
    let (rows, dim) = x.dims2()?;
    let x_c = x.contiguous()?;
    let n = rows * dim;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: cudarc::driver::CudaSlice<f32> =
        unsafe { stream.alloc::<f32>(n).map_err(|e| anyhow::anyhow!(e))? };
    let rc = {
        let (xs, xl) = x_c.storage_and_layout();
        let (x0, x1) = xl
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("rms_no_weight_f32: x not dense"))?;
        let xc = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("rms_no_weight_f32: x not cuda"),
        };
        let xsl = xc.as_cuda_slice::<half::bf16>()?;
        let xview = xsl.slice(x0..x1);
        let (xp, _gx) = xview.device_ptr(&stream);
        let (yp, _gy) = y_dev.device_ptr_mut(&stream);
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
    anyhow::ensure!(rc == 0, "rms_no_weight_bf16_f32 returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = candle_core::Storage::Cuda(storage);
    candle_core::Tensor::from_storage(
        storage,
        (rows, dim),
        candle_core::op::BackpropOp::none(),
        false,
    )
    .reshape((rows, dim))
    .map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn gelu_mul(gate: &Tensor, pli: &Tensor) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let dev = match gate.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("gelu_mul requires cuda"),
    };
    anyhow::ensure!(gate.dtype() == DType::BF16, "gelu_mul gate expects bf16");
    anyhow::ensure!(pli.dtype() == DType::F32, "gelu_mul pli expects f32");
    let dims = gate.dims().to_vec();
    let n: usize = dims.iter().product();
    anyhow::ensure!(
        pli.elem_count() == n,
        "gelu_mul numel mismatch {} vs {}",
        n,
        pli.elem_count()
    );
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: cudarc::driver::CudaSlice<bf16> =
        unsafe { stream.alloc::<bf16>(n).map_err(|e| anyhow::anyhow!(e))? };
    let rc = {
        let (gs, gl) = gate.storage_and_layout();
        let (ps, pl) = pli.storage_and_layout();
        let (g0, g1) = gl
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("gelu_mul: gate not dense"))?;
        let (p0, p1) = pl
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("gelu_mul: pli not dense"))?;
        anyhow::ensure!(g1 - g0 == n && p1 - p0 == n, "gelu_mul: range mismatch");
        let gc = match &*gs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("gelu_mul: gate not cuda"),
        };
        let pc = match &*ps {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("gelu_mul: pli not cuda"),
        };
        let gsl = gc.as_cuda_slice::<bf16>()?;
        let psl = pc.as_cuda_slice::<f32>()?;
        let gview = gsl.slice(g0..g1);
        let pview = psl.slice(p0..p1);
        let (gp, _gg) = gview.device_ptr(&stream);
        let (pp, _gp) = pview.device_ptr(&stream);
        let (yp, _gy) = y_dev.device_ptr_mut(&stream);
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
    anyhow::ensure!(rc == 0, "gelu_mul_bf16f32 returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = candle_core::Storage::Cuda(storage);
    candle_core::Tensor::from_storage(storage, dims, candle_core::op::BackpropOp::none(), false)
        .reshape(gate.shape())
        .map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn gather_rows_bf16_op(weight: &Tensor, ids: &Tensor) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let dev = match weight.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("gather_rows_bf16_op requires cuda"),
    };
    let (vocab, hidden) = weight.dims2()?;
    anyhow::ensure!(weight.dtype() == DType::BF16, "gather weight must be bf16");
    let ids_c = ids.flatten_all()?.contiguous()?;
    anyhow::ensure!(ids_c.dtype() == DType::U32, "gather ids must be u32");
    let n = ids_c.dims()[0];
    let w_c = weight.contiguous()?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
        stream
            .alloc::<bf16>(n * hidden)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let rc = {
        let (ts, _tl) = ids_c.storage_and_layout();
        let (ws, _wl) = w_c.storage_and_layout();
        let tc = match &*ts {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("ids not cuda"),
        };
        let wc = match &*ws {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("weight not cuda"),
        };
        let tsl = tc.as_cuda_slice::<u32>()?;
        let wsl = wc.as_cuda_slice::<bf16>()?;
        let (tp, _gt) = tsl.device_ptr(&stream);
        let (wp, _gw) = wsl.device_ptr(&stream);
        let (yp, _gy) = y_dev.device_ptr_mut(&stream);
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
    anyhow::ensure!(rc == 0, "gather_rows_bf16 returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = candle_core::Storage::Cuda(storage);
    Ok(candle_core::Tensor::from_storage(
        storage,
        (n, hidden),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
fn scale_bf16_op(x: &Tensor, scale: f32) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("scale_bf16_op requires cuda"),
    };
    anyhow::ensure!(x.dtype() == DType::BF16, "scale_bf16_op expects bf16");
    let x_c = x.contiguous()?;
    let dims = x_c.dims().to_vec();
    let n: usize = dims.iter().product();
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: cudarc::driver::CudaSlice<bf16> =
        unsafe { stream.alloc::<bf16>(n).map_err(|e| anyhow::anyhow!(e))? };
    let rc = {
        let (xs, _xl) = x_c.storage_and_layout();
        let xc = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("scale_bf16_op: x not cuda"),
        };
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let (xp, _gx) = xsl.device_ptr(&stream);
        let (yp, _gy) = y_dev.device_ptr_mut(&stream);
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
    anyhow::ensure!(rc == 0, "scale_out_bf16 returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = candle_core::Storage::Cuda(storage);
    candle_core::Tensor::from_storage(storage, dims, candle_core::op::BackpropOp::none(), false)
        .reshape(x_c.shape())
        .map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn cast_scale_f32_op(x: &Tensor, scale: f32) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("cast_scale_f32_op requires cuda"),
    };
    anyhow::ensure!(x.dtype() == DType::BF16, "cast_scale_f32_op expects bf16");
    let x_c = x.contiguous()?;
    let dims = x_c.dims().to_vec();
    let n: usize = dims.iter().product();
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: cudarc::driver::CudaSlice<f32> =
        unsafe { stream.alloc::<f32>(n).map_err(|e| anyhow::anyhow!(e))? };
    let rc = {
        let (xs, _xl) = x_c.storage_and_layout();
        let xc = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("cast_scale_f32_op: x not cuda"),
        };
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let (xp, _gx) = xsl.device_ptr(&stream);
        let (yp, _gy) = y_dev.device_ptr_mut(&stream);
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
    anyhow::ensure!(rc == 0, "cast_scale_bf16_f32 returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = candle_core::Storage::Cuda(storage);
    candle_core::Tensor::from_storage(storage, dims, candle_core::op::BackpropOp::none(), false)
        .reshape(x_c.shape())
        .map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn add_scale_f32_op(a: &Tensor, b: &Tensor, scale: f32) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = match a.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("add_scale_f32_op requires cuda"),
    };
    anyhow::ensure!(
        a.dtype() == DType::F32 && b.dtype() == DType::F32,
        "add_scale_f32_op expects f32"
    );
    let a_c = a.contiguous()?;
    let b_c = b.contiguous()?;
    let dims = a_c.dims().to_vec();
    anyhow::ensure!(dims == b_c.dims(), "add_scale_f32_op shape mismatch");
    let n: usize = dims.iter().product();
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: cudarc::driver::CudaSlice<f32> =
        unsafe { stream.alloc::<f32>(n).map_err(|e| anyhow::anyhow!(e))? };
    let rc = {
        let (as_, _al) = a_c.storage_and_layout();
        let (bs, _bl) = b_c.storage_and_layout();
        let ac = match &*as_ {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("add_scale_f32_op: a not cuda"),
        };
        let bc = match &*bs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("add_scale_f32_op: b not cuda"),
        };
        let asl = ac.as_cuda_slice::<f32>()?;
        let bsl = bc.as_cuda_slice::<f32>()?;
        let (ap, _ga) = asl.device_ptr(&stream);
        let (bp, _gb) = bsl.device_ptr(&stream);
        let (yp, _gy) = y_dev.device_ptr_mut(&stream);
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
    anyhow::ensure!(rc == 0, "add_scale_f32 returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = candle_core::Storage::Cuda(storage);
    candle_core::Tensor::from_storage(storage, dims, candle_core::op::BackpropOp::none(), false)
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
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("rmsnorm_add_scale_op requires cuda"),
    };
    anyhow::ensure!(
        x.dtype() == DType::BF16 && res.dtype() == DType::BF16,
        "rmsnorm_add_scale_op expects bf16"
    );
    let x_c = x.contiguous()?;
    let res_c = res.contiguous()?;
    let dims = x_c.dims().to_vec();
    anyhow::ensure!(res_c.elem_count() == x_c.elem_count(), "res numel mismatch");
    let dim = *dims.last().unwrap();
    let rows: usize = dims.iter().product::<usize>() / dim;
    let w = norm.weight_bf16();
    anyhow::ensure!(
        w.dtype() == DType::BF16 && w.elem_count() == dim,
        "norm weight mismatch"
    );
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
        stream
            .alloc::<bf16>(rows * dim)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let mut rstd_dev: Option<cudarc::driver::CudaSlice<f32>> = match chain_eps {
        Some(_) => Some(unsafe { stream.alloc::<f32>(rows).map_err(|e| anyhow::anyhow!(e))? }),
        None => None,
    };
    if next_w.is_some() {
        anyhow::ensure!(chain_eps.is_some(), "next_w requires chain_eps");
    }
    let mut normed_dev: Option<cudarc::driver::CudaSlice<bf16>> = match next_w {
        Some(nw) => {
            anyhow::ensure!(
                nw.dtype() == DType::BF16 && nw.elem_count() == dim,
                "next_w mismatch"
            );
            Some(unsafe {
                stream
                    .alloc::<bf16>(rows * dim)
                    .map_err(|e| anyhow::anyhow!(e))?
            })
        }
        None => None,
    };
    let rc = {
        let (xs, _xl) = x_c.storage_and_layout();
        let (ws, _wl) = w.storage_and_layout();
        let (rs, _rl) = res_c.storage_and_layout();
        let xc = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("x not cuda"),
        };
        let wc = match &*ws {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("w not cuda"),
        };
        let rcu = match &*rs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("res not cuda"),
        };
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let wsl = wc.as_cuda_slice::<bf16>()?;
        let rsl = rcu.as_cuda_slice::<bf16>()?;
        let (xp, _gx) = xsl.device_ptr(&stream);
        let (wp, _gw) = wsl.device_ptr(&stream);
        let (rp, _gr) = rsl.device_ptr(&stream);
        let (yp, _gy) = y_dev.device_ptr_mut(&stream);
        let rstd_ptr = match rstd_dev.as_mut() {
            Some(b) => {
                let (p, _g) = b.device_ptr_mut(&stream);
                p as *mut f32
            }
            None => std::ptr::null_mut(),
        };
        let next_w_ptr = match next_w {
            Some(nw) => {
                let (ns, _nl) = nw.storage_and_layout();
                let nc = match &*ns {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("next_w not cuda"),
                };
                let nsl = nc.as_cuda_slice::<bf16>()?;
                let (np, _gn) = nsl.device_ptr(&stream);
                np as *const u16
            }
            None => std::ptr::null(),
        };
        let normed_ptr = match normed_dev.as_mut() {
            Some(b) => {
                let (p, _g) = b.device_ptr_mut(&stream);
                p as *mut u16
            }
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
    anyhow::ensure!(rc == 0, "rmsnorm_add_scale_bf16 returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev.clone());
    let storage = candle_core::Storage::Cuda(storage);
    let y = candle_core::Tensor::from_storage(
        storage,
        dims,
        candle_core::op::BackpropOp::none(),
        false,
    )
    .reshape(x_c.shape())?;
    let rstd = rstd_dev.map(|b| {
        let storage = candle_core::CudaStorage::wrap_cuda_slice(b, dev.clone());
        let storage = candle_core::Storage::Cuda(storage);
        candle_core::Tensor::from_storage(
            storage,
            (rows,),
            candle_core::op::BackpropOp::none(),
            false,
        )
    });
    let normed = match normed_dev {
        Some(b) => {
            let storage = candle_core::CudaStorage::wrap_cuda_slice(b, dev);
            let storage = candle_core::Storage::Cuda(storage);
            Some(
                candle_core::Tensor::from_storage(
                    storage,
                    (rows, dim),
                    candle_core::op::BackpropOp::none(),
                    false,
                )
                .reshape(x_c.shape())?,
            )
        }
        None => None,
    };
    Ok((y, rstd, normed))
}

#[cfg(feature = "cuda")]
pub(crate) fn rstd_op(x: &Tensor, eps: f32) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("rstd_op requires cuda"),
    };
    anyhow::ensure!(x.dtype() == DType::BF16, "rstd_op expects bf16");
    let dims = x.dims().to_vec();
    let dim = *dims.last().context("rstd_op: scalar input")?;
    let rows = x.elem_count() / dim;
    let x_c = x.contiguous()?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut r_dev: cudarc::driver::CudaSlice<f32> =
        unsafe { stream.alloc::<f32>(rows).map_err(|e| anyhow::anyhow!(e))? };
    let rc = {
        let (xs, xl) = x_c.storage_and_layout();
        let (x0, x1) = xl
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("rstd_op: x not dense"))?;
        let xc = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("x not cuda"),
        };
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let xview = xsl.slice(x0..x1);
        let (xp, _gx) = xview.device_ptr(&stream);
        let (rp, _gr) = r_dev.device_ptr_mut(&stream);
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
    anyhow::ensure!(rc == 0, "rstd_bf16 returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(r_dev, dev);
    let storage = candle_core::Storage::Cuda(storage);
    Ok(candle_core::Tensor::from_storage(
        storage,
        (rows,),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
fn rms_apply_op(x: &Tensor, w_bf16: &Tensor, rstd: &Tensor) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("rms_apply_op requires cuda"),
    };
    anyhow::ensure!(x.dtype() == DType::BF16, "rms_apply_op expects bf16");
    let x_c = x.contiguous()?;
    let dims = x_c.dims().to_vec();
    let n: usize = dims.iter().product();
    let dim = w_bf16.elem_count();
    anyhow::ensure!(dim > 0 && n % dim == 0, "rms_apply_op weight mismatch");
    anyhow::ensure!(
        rstd.elem_count() == n / dim,
        "rms_apply_op rstd rows mismatch"
    );
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: cudarc::driver::CudaSlice<bf16> =
        unsafe { stream.alloc::<bf16>(n).map_err(|e| anyhow::anyhow!(e))? };
    let rc = {
        let (xs, xl) = x_c.storage_and_layout();
        let (x0, x1) = xl
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("rms_apply_op: x not dense"))?;
        let (ws, _wl) = w_bf16.storage_and_layout();
        let (rs, _rl) = rstd.storage_and_layout();
        let xc = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("x not cuda"),
        };
        let wc = match &*ws {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("w not cuda"),
        };
        let rcu = match &*rs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("rstd not cuda"),
        };
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let xview = xsl.slice(x0..x1);
        let wsl = wc.as_cuda_slice::<bf16>()?;
        let rsl = rcu.as_cuda_slice::<f32>()?;
        let (xp, _gx) = xview.device_ptr(&stream);
        let (wp, _gw) = wsl.device_ptr(&stream);
        let (rp, _gr) = rsl.device_ptr(&stream);
        let (yp, _gy) = y_dev.device_ptr_mut(&stream);
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
    anyhow::ensure!(rc == 0, "rms_apply_bf16 returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = candle_core::Storage::Cuda(storage);
    candle_core::Tensor::from_storage(storage, dims, candle_core::op::BackpropOp::none(), false)
        .reshape(x_c.shape())
        .map_err(Into::into)
}

#[cfg(feature = "cuda")]
pub(crate) fn quantize_lm_head_i8(
    embed: &Tensor,
) -> Result<(
    cudarc::driver::CudaSlice<i8>,
    cudarc::driver::CudaSlice<f32>,
)> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let dev = match embed.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("quantize_lm_head_i8 requires cuda"),
    };
    let (vocab, hidden) = embed.dims2()?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut wq: cudarc::driver::CudaSlice<i8> = unsafe {
        stream
            .alloc::<i8>(vocab * hidden)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let mut rs: cudarc::driver::CudaSlice<f32> =
        unsafe { stream.alloc::<f32>(vocab).map_err(|e| anyhow::anyhow!(e))? };
    let rc = {
        let (es, _el) = embed.storage_and_layout();
        let ec = match &*es {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("embed not cuda"),
        };
        let esl = ec.as_cuda_slice::<bf16>()?;
        let (ep, _ge) = esl.device_ptr(&stream);
        let (qp, _gq) = wq.device_ptr_mut(&stream);
        let (sp, _gs) = rs.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::rowquant_i8(
                stream.cu_stream() as *mut _,
                ep as *const u16,
                qp as *mut i8,
                sp as *mut f32,
                vocab as i32,
                hidden as i32,
            )
        }
    };
    anyhow::ensure!(rc == 0, "rowquant_i8 returned {rc}");
    stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
    Ok((wq, rs))
}

#[cfg(feature = "cuda")]
pub struct E4bSpecVerifier<'a> {
    model: &'a Gemma4E4b,
    fixed: E4bFixedCache,
    pos_scratch: Tensor,
    device: candle_core::CudaDevice,
    pub committed: usize,
    pub last_token: u32,
    pub last_hidden: Tensor,
}

#[cfg(feature = "cuda")]
impl<'a> E4bSpecVerifier<'a> {
    pub fn new(model: &'a Gemma4E4b, prompt: &[u32], max_total: usize) -> Result<Self> {
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");
        let dev = match &model.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("spec verifier requires cuda"),
        };
        let mut cat: Vec<Option<(Tensor, Tensor)>> = vec![None; model.layers.len()];
        let (logits, hiddens) = model.forward_step_spec(prompt, 0, &mut cat)?;
        let n0 = prompt.len();
        let last_token = logits
            .narrow(0, n0 - 1, 1)?
            .argmax(D::Minus1)?
            .reshape(1)?
            .to_vec1::<u32>()?[0];
        let last_hidden = hiddens
            .narrow(0, n0 - 1, 1)?
            .reshape(model.config.hidden_size)?;
        let max_len = max_total.next_power_of_two().max(64);
        let fixed = E4bFixedCache::from_cat(&cat, n0, max_len, &dev)?;
        let pos_scratch = Tensor::zeros(1, DType::I32, &model.device)?;
        Ok(Self {
            model,
            fixed,
            pos_scratch,
            device: dev,
            committed: n0,
            last_token,
            last_hidden,
        })
    }

    pub fn verify(&mut self, feed: &[u32]) -> Result<(Vec<u32>, Tensor)> {
        let m = feed.len();
        let ids = Tensor::from_vec(feed.to_vec(), m, &self.model.device)?;
        let (logits, hiddens) =
            self.model
                .forward_step_fast_mk(&ids, &self.pos_scratch, &mut self.fixed, m)?;
        let targets = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
        Ok((targets, hiddens))
    }

    pub fn accept(&mut self, accepted: usize, bonus: u32, hiddens: &Tensor) -> Result<()> {
        self.committed += accepted + 1;
        self.last_token = bonus;
        self.last_hidden = hiddens
            .narrow(0, accepted, 1)?
            .reshape(self.model.config.hidden_size)?;
        let stream = nv_layers::cuda_stream::current_stream(&self.device);
        stream
            .memcpy_htod(&[self.committed as i32], &mut self.fixed.pos)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(())
    }

    pub fn drafter_shared_kv(&self) -> Result<((Tensor, Tensor), (Tensor, Tensor))> {
        let mut sl = None;
        let mut fl = None;
        for (i, l) in self.model.layers.iter().enumerate() {
            if l.k_norm.is_some() {
                match l.kind {
                    LayerType::SlidingAttention => sl = Some(i),
                    LayerType::FullAttention => fl = Some(i),
                }
            }
        }
        let view = |i: usize, hd: usize, nkv: usize| -> Result<(Tensor, Tensor)> {
            let len = self.committed;
            let stream = nv_layers::cuda_stream::current_stream(&self.device);
            let mk = |src: &cudarc::driver::CudaSlice<half::bf16>| -> Result<Tensor> {
                let elems = len * nkv * hd;
                let mut buf: cudarc::driver::CudaSlice<half::bf16> = unsafe {
                    stream
                        .alloc::<half::bf16>(elems)
                        .map_err(|e| anyhow::anyhow!(e))?
                };
                let src_view = src.slice(0..elems);
                stream
                    .memcpy_dtod(&src_view, &mut buf)
                    .map_err(|e| anyhow::anyhow!(e))?;
                let storage = candle_core::CudaStorage::wrap_cuda_slice(buf, self.device.clone());
                let storage = candle_core::Storage::Cuda(storage);
                Tensor::from_storage(
                    storage,
                    (len, nkv, hd),
                    candle_core::op::BackpropOp::none(),
                    false,
                )
                .transpose(0, 1)?
                .contiguous()
                .map_err(Into::into)
            };
            let k = self.fixed.k[i].as_ref().context("drafter kv: no k")?;
            let v = self.fixed.v[i].as_ref().context("drafter kv: no v")?;
            Ok((mk(k)?, mk(v)?))
        };
        let sl = sl.context("no sliding kv layer")?;
        let fl = fl.context("no full kv layer")?;
        let lsl = &self.model.layers[sl];
        let lfl = &self.model.layers[fl];
        Ok((
            view(sl, lsl.head_dim, lsl.n_kv_heads)?,
            view(fl, lfl.head_dim, lfl.n_kv_heads)?,
        ))
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn lm_head_i8_normed_mk_op(
    wq: &cudarc::driver::CudaSlice<i8>,
    row_scale: &cudarc::driver::CudaSlice<f32>,
    x: &Tensor,
    w_norm: &Tensor,
    rstd: &Tensor,
    m: usize,
    dev: &candle_core::CudaDevice,
) -> Result<Tensor> {
    i8_normed_mk_op_impl(wq, row_scale, x, w_norm, rstd, m, dev, false)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn i8_mk_h_op(
    wq: &cudarc::driver::CudaSlice<i8>,
    row_scale: &cudarc::driver::CudaSlice<f32>,
    x: &Tensor,
    w_norm: &Tensor,
    rstd: &Tensor,
    m: usize,
    dev: &candle_core::CudaDevice,
) -> Result<Tensor> {
    i8_normed_mk_op_impl(wq, row_scale, x, w_norm, rstd, m, dev, true)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn i8_normed_mk_op_impl(
    wq: &cudarc::driver::CudaSlice<i8>,
    row_scale: &cudarc::driver::CudaSlice<f32>,
    x: &Tensor,
    w_norm: &Tensor,
    rstd: &Tensor,
    m: usize,
    dev: &candle_core::CudaDevice,
    force_h: bool,
) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let hidden = x.elem_count() / m;
    let vocab = wq.len() / hidden;
    anyhow::ensure!(rstd.elem_count() == m, "lm_head_mk: rstd rows mismatch");
    let x_c = x.reshape((m, hidden))?.contiguous()?;
    let stream = nv_layers::cuda_stream::current_stream(dev);
    let mut y_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
        stream
            .alloc::<bf16>(m * vocab)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let rc = {
        let (xs, xl) = x_c.storage_and_layout();
        let (x0, x1) = xl
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("lm_head_mk: x not dense"))?;
        let (ws, _wl) = w_norm.storage_and_layout();
        let (rs2, _rl) = rstd.storage_and_layout();
        let xc = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("x not cuda"),
        };
        let wc = match &*ws {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("w_norm not cuda"),
        };
        let rcu = match &*rs2 {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("rstd not cuda"),
        };
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let xview = xsl.slice(x0..x1);
        let wsl = wc.as_cuda_slice::<bf16>()?;
        let rsl = rcu.as_cuda_slice::<f32>()?;
        let (qp, _gq) = wq.device_ptr(&stream);
        let (scp, _gsc) = row_scale.device_ptr(&stream);
        let (xp, _gx) = xview.device_ptr(&stream);
        let (wp, _gw) = wsl.device_ptr(&stream);
        let (rp, _gr) = rsl.device_ptr(&stream);
        let (yp, _gy) = y_dev.device_ptr_mut(&stream);

        let prenorm = !force_h
            && m <= 8
            && hidden >= 4096
            && std::env::var("NV_LMI8_PRENORM").ok().as_deref() != Some("0");
        unsafe {
            if prenorm {
                let mut xn_dev: cudarc::driver::CudaSlice<bf16> = stream
                    .alloc::<bf16>(m * hidden)
                    .map_err(|e| anyhow::anyhow!(e))?;
                let (xnp, _gxn) = xn_dev.device_ptr_mut(&stream);
                let rc1 = nv_kernels::cuda::normx_mk(
                    stream.cu_stream() as *mut _,
                    xp as *const u16,
                    wp as *const u16,
                    rp as *const f32,
                    xnp as *mut u16,
                    hidden as i32,
                    m as i32,
                );
                if rc1 != 0 {
                    rc1
                } else {
                    nv_kernels::cuda::gemv_i8_prenormed_mk(
                        stream.cu_stream() as *mut _,
                        qp as *const i8,
                        scp as *const f32,
                        xnp as *const u16,
                        yp as *mut u16,
                        vocab as i32,
                        hidden as i32,
                        m as i32,
                    )
                }
            } else if force_h {
                nv_kernels::cuda::gemv_i8_mk_h(
                    stream.cu_stream() as *mut _,
                    qp as *const i8,
                    scp as *const f32,
                    xp as *const u16,
                    wp as *const u16,
                    rp as *const f32,
                    yp as *mut u16,
                    vocab as i32,
                    hidden as i32,
                    m as i32,
                )
            } else {
                nv_kernels::cuda::gemv_i8_normed_mk(
                    stream.cu_stream() as *mut _,
                    qp as *const i8,
                    scp as *const f32,
                    xp as *const u16,
                    wp as *const u16,
                    rp as *const f32,
                    yp as *mut u16,
                    vocab as i32,
                    hidden as i32,
                    m as i32,
                )
            }
        }
    };
    anyhow::ensure!(
        rc == 0,
        "gemv_i8_normed_mk(force_h={force_h}) returned {rc}"
    );
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev.clone());
    let storage = candle_core::Storage::Cuda(storage);
    Ok(candle_core::Tensor::from_storage(
        storage,
        (m, vocab),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
fn lm_head_i8_normed_op(
    wq: &cudarc::driver::CudaSlice<i8>,
    row_scale: &cudarc::driver::CudaSlice<f32>,
    x: &Tensor,
    w_norm: &Tensor,
    rstd: &Tensor,
    dev: &candle_core::CudaDevice,
) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let hidden = x.elem_count();
    let vocab = wq.len() / hidden;
    let x_c = x.contiguous()?;
    let stream = nv_layers::cuda_stream::current_stream(dev);
    let mut y_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
        stream
            .alloc::<bf16>(vocab)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let rc = {
        let (xs, xl) = x_c.storage_and_layout();
        let (x0, x1) = xl
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("lm_head_i8: x not dense"))?;
        let (ws, _wl) = w_norm.storage_and_layout();
        let (rs2, _rl) = rstd.storage_and_layout();
        let xc = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("x not cuda"),
        };
        let wc = match &*ws {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("w_norm not cuda"),
        };
        let rcu = match &*rs2 {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("rstd not cuda"),
        };
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let xview = xsl.slice(x0..x1);
        let wsl = wc.as_cuda_slice::<bf16>()?;
        let rsl = rcu.as_cuda_slice::<f32>()?;
        let (qp, _gq) = wq.device_ptr(&stream);
        let (scp, _gsc) = row_scale.device_ptr(&stream);
        let (xp, _gx) = xview.device_ptr(&stream);
        let (wp, _gw) = wsl.device_ptr(&stream);
        let (rp, _gr) = rsl.device_ptr(&stream);
        let (yp, _gy) = y_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::gemv_i8_normed(
                stream.cu_stream() as *mut _,
                qp as *const i8,
                scp as *const f32,
                xp as *const u16,
                wp as *const u16,
                rp as *const f32,
                yp as *mut u16,
                vocab as i32,
                hidden as i32,
            )
        }
    };
    anyhow::ensure!(rc == 0, "gemv_i8_normed returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev.clone());
    let storage = candle_core::Storage::Cuda(storage);
    Ok(candle_core::Tensor::from_storage(
        storage,
        (1usize, 1usize, vocab),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
fn lm_head_normed_op(embed: &Tensor, x: &Tensor, w_norm: &Tensor, rstd: &Tensor) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("lm_head_normed_op requires cuda"),
    };
    let (vocab, hidden) = embed.dims2()?;
    anyhow::ensure!(x.elem_count() == hidden, "lm_head x mismatch");
    let x_c = x.contiguous()?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
        stream
            .alloc::<bf16>(vocab)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let rc = {
        let (es, _el) = embed.storage_and_layout();
        let (xs, xl) = x_c.storage_and_layout();
        let (x0, x1) = xl
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("lm_head_normed_op: x not dense"))?;
        let (ws, _wl) = w_norm.storage_and_layout();
        let (rs, _rl) = rstd.storage_and_layout();
        let ec = match &*es {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("embed not cuda"),
        };
        let xc = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("x not cuda"),
        };
        let wc = match &*ws {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("w_norm not cuda"),
        };
        let rcu = match &*rs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("rstd not cuda"),
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
        let (yp, _gy) = y_dev.device_ptr_mut(&stream);
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
    anyhow::ensure!(rc == 0, "gemv_bf16_normed returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = candle_core::Storage::Cuda(storage);
    Ok(candle_core::Tensor::from_storage(
        storage,
        (1usize, 1usize, vocab),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

#[cfg(not(feature = "cuda"))]
fn geglu_fused_op(gu: &Tensor) -> Result<Tensor> {
    let last = gu.dims().len() - 1;
    let two_inter = gu.dim(last)?;
    anyhow::ensure!(two_inter % 2 == 0, "geglu last dim must be even");
    let inter = two_inter / 2;
    let gate = gu.narrow(last, 0, inter)?.contiguous()?;
    let up = gu.narrow(last, inter, inter)?.contiguous()?;
    let act = gate.gelu()?;
    act.mul(&up).map_err(|e| anyhow::anyhow!(e))
}

#[cfg(feature = "cuda")]
fn geglu_fused_op(gu: &Tensor) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let dev = match gu.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("geglu_fused_op requires cuda"),
    };
    anyhow::ensure!(gu.dtype() == DType::BF16, "geglu_fused_op expects bf16");
    let gu_c = gu.contiguous()?;
    let mut dims = gu_c.dims().to_vec();
    let two_inter = *dims.last().unwrap();
    anyhow::ensure!(two_inter % 2 == 0, "geglu last dim must be even");
    let inter = two_inter / 2;
    let rows: usize = dims.iter().product::<usize>() / two_inter;
    *dims.last_mut().unwrap() = inter;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut y_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
        stream
            .alloc::<bf16>(rows * inter)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let rc = {
        let (gs, _gl) = gu_c.storage_and_layout();
        let gc = match &*gs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("gu not cuda"),
        };
        let gsl = gc.as_cuda_slice::<bf16>()?;
        let (gp, _gg) = gsl.device_ptr(&stream);
        let (yp, _gy) = y_dev.device_ptr_mut(&stream);
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
    anyhow::ensure!(rc == 0, "gelu_tanh_mul_fused_bf16 returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
    let storage = candle_core::Storage::Cuda(storage);
    Ok(candle_core::Tensor::from_storage(
        storage,
        dims,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
pub(crate) struct E4bFixedCache {
    k: Vec<Option<cudarc::driver::CudaSlice<half::bf16>>>,
    v: Vec<Option<cudarc::driver::CudaSlice<half::bf16>>>,
    pos: cudarc::driver::CudaSlice<i32>,

    fan_in: cudarc::driver::CudaSlice<u32>,
    device: candle_core::CudaDevice,
}

#[cfg(feature = "cuda")]
impl E4bFixedCache {
    fn from_cat(
        cat: &[Option<(Tensor, Tensor)>],
        plen: usize,
        max_len: usize,
        dev: &candle_core::CudaDevice,
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
                    let mut kbuf = unsafe {
                        stream
                            .alloc::<half::bf16>(max_len * nkv * hd)
                            .map_err(|e| anyhow::anyhow!(e))?
                    };
                    let mut vbuf = unsafe {
                        stream
                            .alloc::<half::bf16>(max_len * nkv * hd)
                            .map_err(|e| anyhow::anyhow!(e))?
                    };
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
        let pos = stream
            .clone_htod(&[plen as i32])
            .map_err(|e| anyhow::anyhow!(e))?;
        let fan_in = stream
            .alloc_zeros::<u32>(64)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(Self {
            k,
            v,
            pos,
            fan_in,
            device: dev.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
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
        anyhow::ensure!(
            qkv_c.dtype() == DType::BF16
                && qkv_c.elem_count() % expected == 0
                && qkv_row < qkv_c.elem_count() / expected,
            "qkv_prep: bad qkv shape"
        );
        anyhow::ensure!(rope_pos.dtype() == DType::I32, "rope_pos must be i32");
        let mut q_out: cudarc::driver::CudaSlice<f32> = unsafe {
            stream
                .alloc::<f32>(nh * hd)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let rc = {
            let (qs, _ql) = qkv_c.storage_and_layout();
            let qc = match &*qs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("qkv not cuda"),
            };
            let qsl = qc.as_cuda_slice::<half::bf16>()?;
            let (qp, _gq) = qsl.device_ptr(&stream);
            let qp = qp + (qkv_row * expected * 2) as u64;
            let (qws, _l) = q_norm_w.storage_and_layout();
            let qwc = match &*qws {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("q_norm weight not cuda"),
            };
            let qwsl = qwc.as_cuda_slice::<half::bf16>()?;
            let (qwp, _gqw) = qwsl.device_ptr(&stream);
            let (cs, _cl) = rope.cos().storage_and_layout();
            let (ss, _sl) = rope.sin().storage_and_layout();
            let cc = match &*cs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("cos not cuda"),
            };
            let sc = match &*ss {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("sin not cuda"),
            };
            let csl = cc.as_cuda_slice::<f32>()?;
            let ssl = sc.as_cuda_slice::<f32>()?;
            let (cp, _gc) = csl.device_ptr(&stream);
            let (sp, _gs) = ssl.device_ptr(&stream);
            let (rs, _rl) = rope_pos.storage_and_layout();
            let rc2 = match &*rs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("rope_pos not cuda"),
            };
            let rsl = rc2.as_cuda_slice::<i32>()?;
            let (rp, _gr) = rsl.device_ptr(&stream);
            let (op, _go) = q_out.device_ptr_mut(&stream);
            let (kwp, cache_pos_ptr, ckp, cvp) = match k_norm_w {
                Some(kw) => {
                    let (kws, _l) = kw.storage_and_layout();
                    let kwc = match &*kws {
                        candle_core::Storage::Cuda(s) => s,
                        _ => anyhow::bail!("k_norm weight not cuda"),
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
        anyhow::ensure!(rc == 0, "qkv_prep returned {rc}");
        let storage = candle_core::CudaStorage::wrap_cuda_slice(q_out, self.device.clone());
        let storage = candle_core::Storage::Cuda(storage);
        candle_core::Tensor::from_storage(
            storage,
            (nh, hd),
            candle_core::op::BackpropOp::none(),
            false,
        )
        .reshape((nh, hd))
        .map_err(Into::into)
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
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        let stream = nv_layers::cuda_stream::current_stream(&self.device);
        let q = q.contiguous()?;

        let mut out_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(nh * hd)
                .map_err(|e| anyhow::anyhow!(e))?
        };

        let scratch_elems = nv_kernels::cuda::flash_splitk_scratch_elems(nh as i32, hd as i32);
        let mut scratch: cudarc::driver::CudaSlice<f32> = unsafe {
            stream
                .alloc::<f32>(scratch_elems)
                .map_err(|e| anyhow::anyhow!(e))?
        };
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
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("q not cuda"),
            };
            let qsl = qcuda.as_cuda_slice::<f32>()?;
            let (qp, _gq) = qsl.device_ptr(&stream);
            let (op, _go) = out_dev.device_ptr_mut(&stream);
            let (sp, _gs) = scratch.device_ptr_mut(&stream);
            let (fp, _gf) = self.fan_in.device_ptr(&stream);
            unsafe {
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
        };
        anyhow::ensure!(rc == 0, "flash_decode_fused_bf16kv returned {rc}");
        let storage = candle_core::CudaStorage::wrap_cuda_slice(out_dev, self.device.clone());
        let storage = candle_core::Storage::Cuda(storage);
        candle_core::Tensor::from_storage(
            storage,
            (nh, hd),
            candle_core::op::BackpropOp::none(),
            false,
        )
        .reshape((nh, hd))
        .map_err(Into::into)
    }

    #[allow(clippy::too_many_arguments)]
    fn attend_into(
        &self,
        buf: usize,
        q: &Tensor,
        delta: i32,
        out: &Tensor,
        out_row: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        window: usize,
    ) -> Result<()> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        let stream = nv_layers::cuda_stream::current_stream(&self.device);
        let q = q.contiguous()?;
        anyhow::ensure!(out.dtype() == DType::BF16, "attend_into: out must be bf16");
        anyhow::ensure!(
            out.elem_count() >= (out_row + 1) * nh * hd,
            "attend_into: out too small"
        );

        let scratch_elems = nv_kernels::cuda::flash_splitk_scratch_elems(nh as i32, hd as i32);
        let mut scratch: cudarc::driver::CudaSlice<f32> = unsafe {
            stream
                .alloc::<f32>(scratch_elems)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let rc = {
            let pos_ptr = {
                let (p, _g) = self.pos.device_ptr(&stream);
                p as *const i32
            };
            let kc = self.k[buf].as_ref().context("attend_into: no k buffer")?;
            let vc = self.v[buf].as_ref().context("attend_into: no v buffer")?;
            let (ckp, _gk) = kc.device_ptr(&stream);
            let (cvp, _gv) = vc.device_ptr(&stream);
            let (qs, _ql) = q.storage_and_layout();
            let qcuda = match &*qs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("q not cuda"),
            };
            let qsl = qcuda.as_cuda_slice::<f32>()?;
            let (qp, _gq) = qsl.device_ptr(&stream);
            let (os, _ol) = out.storage_and_layout();
            let ocuda = match &*os {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("attend_into: out not cuda"),
            };
            let osl = ocuda.as_cuda_slice::<bf16>()?;
            let (op, _go) = osl.device_ptr(&stream);
            let op = op + (out_row * nh * hd * 2) as u64;
            let (sp, _gs) = scratch.device_ptr_mut(&stream);
            let (fp, _gf) = self.fan_in.device_ptr(&stream);
            unsafe {
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
        };
        anyhow::ensure!(rc == 0, "flash_decode_fused_bf16kv returned {rc}");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn cast_tensor_front_bf16(
    t: &Tensor,
    dst: &mut cudarc::driver::CudaSlice<half::bf16>,
    n: usize,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<()> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let t = t.contiguous()?;
    let (ts, _tl) = t.storage_and_layout();
    let tcuda = match &*ts {
        candle_core::Storage::Cuda(s) => s,
        _ => anyhow::bail!("cast_tensor_front_bf16: not cuda"),
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
    anyhow::ensure!(rc == 0, "cast_f32_bf16 (cache prefill) returned {rc}");
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
    device: candle_core::CudaDevice,
    forked: std::sync::Arc<cudarc::driver::CudaStream>,
    runner: nv_kernels::graph::CudaGraphRunner,
    token_buf: cudarc::driver::CudaSlice<u32>,
    rope_src: cudarc::driver::CudaSlice<i32>,
    ring: cudarc::driver::CudaSlice<u32>,
    amax_val: cudarc::driver::CudaSlice<f32>,
    amax_idx: cudarc::driver::CudaSlice<i32>,
    host_tok: Box<[u32; 1]>,
    current_pos: usize,
    call_count: u64,
    capture_active: bool,
}

#[cfg(feature = "cuda")]
const E4B_TOKEN_RING: usize = 64;

#[cfg(feature = "cuda")]
impl<'m> GraphedE4bDecoder<'m> {
    fn new(model: &'m Gemma4E4b, cache: E4bFixedCache, start_len: usize) -> Result<Self> {
        let dev = match &model.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("GraphedE4bDecoder requires cuda"),
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

    pub fn warm_step(&mut self, token_id: u32) -> Result<u32> {
        let logits = self
            .model
            .forward_step_fast(token_id, self.current_pos, &mut self.cache)?;
        self.call_count += 1;
        self.current_pos += 1;
        logits
            .argmax(D::Minus1)?
            .to_scalar::<u32>()
            .map_err(Into::into)
    }

    pub fn replay_batch(&mut self, seed_token: u32, k: usize) -> Result<Vec<u32>> {
        anyhow::ensure!(k >= 1 && k <= E4B_TOKEN_RING, "bad batch size {k}");

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
                    let tok_clone = token_buf
                        .try_clone()
                        .map_err(|e| anyhow::anyhow!("clone token_buf: {e:?}"))?;
                    let pos_clone = rope_src
                        .try_clone()
                        .map_err(|e| anyhow::anyhow!("clone rope_src: {e:?}"))?;
                    let ids = {
                        let storage =
                            candle_core::CudaStorage::wrap_cuda_slice(tok_clone, dev.clone());
                        let storage = candle_core::Storage::Cuda(storage);
                        Tensor::from_storage(
                            storage,
                            (1usize,),
                            candle_core::op::BackpropOp::none(),
                            false,
                        )
                    };
                    let pos_t = {
                        let storage =
                            candle_core::CudaStorage::wrap_cuda_slice(pos_clone, dev.clone());
                        let storage = candle_core::Storage::Cuda(storage);
                        Tensor::from_storage(
                            storage,
                            (1usize,),
                            candle_core::op::BackpropOp::none(),
                            false,
                        )
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
#[allow(clippy::too_many_arguments)]
fn argmax_into_bufs(
    logits: &Tensor,
    pos: &cudarc::driver::CudaSlice<i32>,
    token_buf: &mut cudarc::driver::CudaSlice<u32>,
    ring: &mut cudarc::driver::CudaSlice<u32>,
    amax_val: &mut cudarc::driver::CudaSlice<f32>,
    amax_idx: &mut cudarc::driver::CudaSlice<i32>,
    ring_mask: i32,
    dev: &candle_core::CudaDevice,
) -> Result<()> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    anyhow::ensure!(logits.dtype() == DType::BF16, "argmax expects bf16 logits");
    let n = logits.elem_count();
    let logits_c = logits.contiguous()?;
    let stream = nv_layers::cuda_stream::current_stream(dev);
    let rc = {
        let (ls, _ll) = logits_c.storage_and_layout();
        let lc = match &*ls {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("logits not cuda"),
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
    anyhow::ensure!(rc == 0, "argmax_bf16 returned {rc}");
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
    fn graphed_matches_eager_greedy() {
        let _g = capture_lock();
        let Some((model, _device)) = load() else {
            return;
        };
        let prompt = [818u32, 5279, 529, 7001, 563];
        let n = 24;
        let eos = [999999u32];
        let base = model.generate_fast(&prompt, n, &eos).unwrap();
        let graphed = model.generate_graphed(&prompt, n, &eos).unwrap();
        eprintln!("eager:   {:?}", &base);
        eprintln!("graphed: {:?}", &graphed);
        if std::env::var("NV_E4B_FORCE_GEMV").is_ok() {
            assert_eq!(base, graphed, "graphed diverged from eager");
        }
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
