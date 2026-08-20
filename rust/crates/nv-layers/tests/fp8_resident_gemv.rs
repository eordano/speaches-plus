#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_layers::linear::Linear;
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u32() & 0x7f_ffff) as f32 / 8388608.0) * 2.0 - 1.0
    }
    fn bf16s(&mut self, n: usize, gain: f32) -> Vec<bf16> {
        (0..n)
            .map(|_| bf16::from_f32(self.next_f32() * gain))
            .collect()
    }
}

fn cuda_device(test: &str) -> Option<Device> {
    match Device::new_cuda_with_stream(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("{test}: SKIP no CUDA device 0: {e}");
            None
        }
    }
}

fn host_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

#[test]
fn resident_fp8_forward_at_m1_and_m4_matches_the_bf16_dequant_linear() {
    let Some(device) = cuda_device("resident_fp8_forward_at_m1_and_m4_matches_the_bf16_dequant_linear")
    else {
        return;
    };
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let mut rng = Lcg(0x243f6a8885a308d3);
    let (n, k) = (1024usize, 512usize);
    let w_host = rng.bf16s(n * k, 0.05);
    let (wq, scales) = nv_quant::fp8::quantize_e4m3_per_row(&w_host, n, k).unwrap();

    let stream = dev.cuda_stream();
    #[allow(deprecated)]
    let wq_dev: CudaSlice<u8> = stream.clone_htod(&wq).unwrap();
    let runner = Arc::new(Mutex::new(
        nv_quant::fp8::Fp8GemmRunner::new(stream.clone()).unwrap(),
    ));
    let resident = Linear::new_fp8_e4m3_row_scales_without_the_cublaslt_probe(
        wq_dev,
        scales.clone(),
        k,
        n,
        None,
        &device,
        runner,
        nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
    )
    .unwrap();

    let dequant_f32 = nv_quant::fp8::dequantize_e4m3_per_row(&wq, n, k, &scales).unwrap();
    let dequant_bf16: Vec<bf16> = dequant_f32.iter().map(|v| bf16::from_f32(*v)).collect();
    let w_ref = Tensor::from_vec(dequant_bf16, (n, k), &device).unwrap();
    let reference = Linear::new_no_pretranspose(w_ref, None).unwrap();

    for m in [1usize, 4] {
        let x = Tensor::from_vec(rng.bf16s(m * k, 1.0), (m, k), &device).unwrap();
        let got = host_f32(&resident.forward(&x).unwrap());
        let want = host_f32(&reference.forward(&x).unwrap());
        let mut max_rel = 0f32;
        for (g, w) in got.iter().zip(&want) {
            max_rel = max_rel.max((g - w).abs() / w.abs().max(0.25));
        }
        for j in 0..m {
            assert_eq!(
                argmax(&got[j * n..(j + 1) * n]),
                argmax(&want[j * n..(j + 1) * n]),
                "m={m} row {j}: argmax disagrees between resident fp8 gemv and bf16 dequant linear"
            );
        }
        eprintln!("[fp8-resident] synthetic n={n} k={k} m={m} max_rel={max_rel:.3e}");
        assert!(
            max_rel < 2.0e-2,
            "m={m}: resident fp8 forward deviates {max_rel:.3e} from the bf16 dequant linear"
        );
    }
}

#[test]
fn resident_fp8_forward_above_m16_takes_the_pertensor_raw_gemm_and_matches_the_rowcol_scaled_oracle()
{
    let Some(device) = cuda_device(
        "resident_fp8_forward_above_m16_takes_the_pertensor_raw_gemm_and_matches_the_rowcol_scaled_oracle",
    ) else {
        return;
    };
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let mut rng = Lcg(0x452821e638d01377);
    let (n, k) = (1024usize, 512usize);
    let w_host = rng.bf16s(n * k, 0.05);
    let (wq, w_scales) = nv_quant::fp8::quantize_e4m3_per_row(&w_host, n, k).unwrap();

    let stream = dev.cuda_stream();
    #[allow(deprecated)]
    let wq_dev: CudaSlice<u8> = stream.clone_htod(&wq).unwrap();
    let runner = Arc::new(Mutex::new(
        nv_quant::fp8::Fp8GemmRunner::new(stream.clone()).unwrap(),
    ));
    let resident = Linear::new_fp8_e4m3_row_scales_without_the_cublaslt_probe(
        wq_dev,
        w_scales.clone(),
        k,
        n,
        None,
        &device,
        runner,
        nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
    )
    .unwrap();

    let dequant_f32 = nv_quant::fp8::dequantize_e4m3_per_row(&wq, n, k, &w_scales).unwrap();
    let dequant_bf16: Vec<bf16> = dequant_f32.iter().map(|v| bf16::from_f32(*v)).collect();
    let w_ref = Tensor::from_vec(dequant_bf16, (n, k), &device).unwrap();
    let reference = Linear::new_no_pretranspose(w_ref, None).unwrap();

    for m in [17usize, 64] {
        let x_host = rng.bf16s(m * k, 1.0);
        let x = Tensor::from_vec(x_host.clone(), (m, k), &device).unwrap();
        let got = host_f32(&resident.forward(&x).unwrap());

        let mut xq = vec![0u8; m * k];
        let mut a_scales = vec![0f32; m];
        for i in 0..m {
            let row = &x_host[i * k..(i + 1) * k];
            let rmax = row
                .iter()
                .map(|v| v.to_f32().abs())
                .filter(|v| v.is_finite())
                .fold(0f32, f32::max);
            let (scale, inv) = if rmax > 0.0 {
                (rmax / 448.0, 448.0 / rmax)
            } else {
                (0.0, 0.0)
            };
            a_scales[i] = scale;
            for (dst, v) in xq[i * k..(i + 1) * k].iter_mut().zip(row) {
                *dst = float8::F8E4M3::from(v.to_f32() * inv).to_bits();
            }
        }
        let oracle =
            nv_quant::fp8::cpu_e4m3_matmul_row_scaled(&xq, &wq, &a_scales, &w_scales, m, n, k)
                .unwrap();

        let mut max_rel_oracle = 0f32;
        for (g, w) in got.iter().zip(&oracle) {
            max_rel_oracle = max_rel_oracle.max((g - w).abs() / w.abs().max(0.25));
        }
        let want = host_f32(&reference.forward(&x).unwrap());
        let mut num = 0f64;
        let mut den = 0f64;
        for (g, w) in got.iter().zip(&want) {
            num += ((g - w) as f64).powi(2);
            den += (*w as f64).powi(2);
        }
        let rel_rms_ref = (num / den.max(1e-12)).sqrt();
        eprintln!(
            "[fp8-resident-prefill] synthetic n={n} k={k} m={m} max_rel_vs_fp8_oracle={max_rel_oracle:.3e} rel_rms_vs_bf16_dequant={rel_rms_ref:.3e}"
        );
        assert!(
            max_rel_oracle < 2.0e-2,
            "m={m}: the device rowquant+raw-gemm+rowcol-epilogue route deviates \
             {max_rel_oracle:.3e} from the exact fp8 row-scaled oracle; the epilogue is applying \
             the wrong scale to some axis if this is large"
        );
        assert!(
            rel_rms_ref < 0.05,
            "m={m}: fp8 prefill route rel rms {rel_rms_ref:.3e} vs the bf16 dequant linear \
             exceeds the same 0.05 bound the per-tensor fp8 arm holds; activation e4m3 \
             quantization alone sits near 3e-2"
        );
    }
}

fn require_q38() {
    if std::env::var("NV_Q38_TEST").as_deref() != Ok("1") {
        panic!("set NV_Q38_TEST=1 to run the real-checkpoint fp8 resident probes");
    }
}

fn snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN38_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .unwrap_or_else(|e| panic!("no snapshot dir {}: {e}; set NV_QWEN38_DIR", snaps.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("model.safetensors").exists())
        .collect();
    entries.sort();
    entries
        .pop()
        .expect("no hydrated unsloth/Qwen3.8-27B-NVFP4 snapshot; set NV_QWEN38_DIR")
}

fn bench_us(stream: &Arc<CudaStream>, iters: usize, mut launch: impl FnMut()) -> f64 {
    for _ in 0..3 {
        launch();
    }
    stream.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        launch();
    }
    stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

#[test]
#[ignore = "reads the real 1.27 GB fp8 lm_head from the checkpoint; set NV_Q38_TEST=1"]
fn real_lm_head_fp8_gemv_halves_weight_bytes_and_lands_within_15pct_of_bf16_path_bandwidth() {
    require_q38();
    let dir = snapshot_dir();
    let Some(device) =
        cuda_device("real_lm_head_fp8_gemv_halves_weight_bytes_and_lands_within_15pct_of_bf16_path_bandwidth")
    else {
        return;
    };
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = dev.cuda_stream();
    let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("weights");
    let module = "lm_head";
    let shape = weights.shape_of("lm_head.weight").expect("shape");
    let (n, k) = (shape[0], shape[1]);
    assert_eq!(k % 16, 0, "lm_head K must be 16-aligned for the fp8 gemv");

    let bytes = weights.raw_bytes("lm_head.weight").expect("raw fp8 bytes");
    let scale_t = weights.get("lm_head.weight_scale", DType::F32).expect("scales");
    let scale_dims = scale_t.dims().to_vec();
    let scale_vals = host_f32(&scale_t);
    let rows = nv_weights::fp8_row_scales_from(&scale_dims, &scale_vals, n).expect("rows");

    #[allow(deprecated)]
    let wq_dev: CudaSlice<u8> = stream.clone_htod(bytes).unwrap();
    #[allow(deprecated)]
    let rs_dev: CudaSlice<f32> = stream.clone_htod(&rows).unwrap();

    let dequant = nv_layers::linear::fp8_e4m3_rowscale_checkpoint_dequant_linear(
        &weights,
        module,
        n,
        k,
        DType::BF16,
    )
    .expect("bf16 dequant linear");
    let w_bf = dequant.weight().expect("bf16 weight").contiguous().unwrap();

    let mut rng = Lcg(0x452821e638d01377);
    let mut y_fp8: CudaSlice<u16> = stream.alloc_zeros::<u16>(4 * n).unwrap();
    let mut y_bf: CudaSlice<u16> = stream.alloc_zeros::<u16>(4 * n).unwrap();
    let fp8_bytes = (n * k) as f64;
    let bf16_bytes = 2.0 * fp8_bytes;

    for m in [1usize, 4] {
        let x_host = rng.bf16s(m * k, 1.0);
        let x_bits: Vec<u16> = x_host.iter().map(|b| b.to_bits()).collect();
        #[allow(deprecated)]
        let x_dev: CudaSlice<u16> = stream.clone_htod(&x_bits).unwrap();

        let us_fp8 = {
            let (pw, _a) = wq_dev.device_ptr(&stream);
            let (ps, _b) = rs_dev.device_ptr(&stream);
            let (px, _c) = x_dev.device_ptr(&stream);
            let (py, _d) = y_fp8.device_ptr_mut(&stream);
            bench_us(&stream, 20, || {
                let rc = unsafe {
                    nv_kernels::cuda::gemv_e4m3_mk(
                        stream.cu_stream() as *mut c_void,
                        pw as *const u8,
                        ps as *const f32,
                        px as *const u16,
                        py as *mut u16,
                        n as i32,
                        k as i32,
                        m as i32,
                    )
                };
                assert_eq!(rc, 0, "gemv_e4m3_mk rc={rc}");
            })
        };

        let us_bf16 = {
            let (w_storage, wl) = w_bf.storage_and_layout();
            let w_cuda = match &*w_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => unreachable!(),
            };
            let w_slice = w_cuda.as_cuda_slice::<bf16>().unwrap();
            let w_view = w_slice.slice(wl.start_offset()..);
            let (pwb, _e) = w_view.device_ptr(&stream);
            let (px, _c) = x_dev.device_ptr(&stream);
            let (pyb, _f) = y_bf.device_ptr_mut(&stream);
            bench_us(&stream, 20, || {
                let rc = unsafe {
                    if m == 1 {
                        nv_kernels::cuda::gemv_bf16(
                            stream.cu_stream() as *mut c_void,
                            pwb as *const u16,
                            px as *const u16,
                            pyb as *mut u16,
                            n as i32,
                            k as i32,
                        )
                    } else {
                        nv_kernels::cuda::gemm_bf16_mk(
                            stream.cu_stream() as *mut c_void,
                            pwb as *const u16,
                            px as *const u16,
                            pyb as *mut u16,
                            n as i32,
                            k as i32,
                            m as i32,
                        )
                    }
                };
                assert_eq!(rc, 0, "bf16 comparator rc={rc}");
            })
        };

        #[allow(deprecated)]
        let fp8_out = stream.memcpy_dtov(&y_fp8).unwrap();
        #[allow(deprecated)]
        let bf_out = stream.memcpy_dtov(&y_bf).unwrap();
        for j in 0..m {
            let a: Vec<f32> = fp8_out[j * n..(j + 1) * n]
                .iter()
                .map(|v| bf16::from_bits(*v).to_f32())
                .collect();
            let b: Vec<f32> = bf_out[j * n..(j + 1) * n]
                .iter()
                .map(|v| bf16::from_bits(*v).to_f32())
                .collect();
            assert_eq!(
                argmax(&a),
                argmax(&b),
                "m={m} row {j}: real lm_head argmax disagrees between fp8 gemv and bf16 path"
            );
        }

        let gbs_fp8 = fp8_bytes / us_fp8 / 1e3;
        let gbs_bf16 = bf16_bytes / us_bf16 / 1e3;
        eprintln!(
            "[q38-fp8-lmhead] basis: checkpoint={} module={module} n={n} k={k} m={m} | fp8: {:.0} MB read, {us_fp8:.1} us, {gbs_fp8:.0} GB/s | bf16: {:.0} MB read, {us_bf16:.1} us, {gbs_bf16:.0} GB/s | speedup {:.2}x",
            dir.display(),
            fp8_bytes / 1e6,
            bf16_bytes / 1e6,
            us_bf16 / us_fp8,
        );
        if m == 1 {
            assert!(
                gbs_fp8 >= 0.85 * gbs_bf16,
                "m=1 fp8 gemv reaches {gbs_fp8:.0} GB/s on fp8 bytes vs bf16 path {gbs_bf16:.0} GB/s; \
                 below the within-15pct-of-DRAM gate that justifies wiring the resident arm"
            );
            assert!(
                us_fp8 < 0.65 * us_bf16,
                "m=1 fp8 gemv {us_fp8:.1} us must beat 0.65x of bf16 {us_bf16:.1} us to halve decode weight-read time"
            );
        }
    }

    let resident = nv_layers::linear::fp8_e4m3_rowscale_checkpoint_resident_linear(
        &weights,
        module,
        n,
        k,
        &device,
        Arc::new(Mutex::new(
            nv_quant::fp8::Fp8GemmRunner::new(stream.clone()).unwrap(),
        )),
    )
    .expect("resident fp8 linear");
    for m in [1usize, 4] {
        let x = Tensor::from_vec(rng.bf16s(m * k, 1.0), (m, k), &device).unwrap();
        let got = host_f32(&resident.forward(&x).unwrap());
        let want = host_f32(&dequant.forward(&x).unwrap());
        for j in 0..m {
            assert_eq!(
                argmax(&got[j * n..(j + 1) * n]),
                argmax(&want[j * n..(j + 1) * n]),
                "m={m} row {j}: resident arm forward argmax diverges from dequant linear on real lm_head"
            );
        }
        eprintln!("[q38-fp8-lmhead] resident arm forward m={m}: argmax parity ok over {n} logits");
    }
}
