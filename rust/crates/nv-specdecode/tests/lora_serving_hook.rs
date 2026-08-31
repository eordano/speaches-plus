use candle_core::{DType, Device, Tensor};
use nv_layers::linear::Linear;
use nv_specdecode::lora_spec::{ServingDeltaSlice, ServingLoraHook};

fn frand(seed: u64, i: usize) -> f32 {
    let mut z = seed
        .wrapping_add(0x9E3779B97F4A7C15u64.wrapping_mul(i as u64 + 1))
        .wrapping_mul(0xBF58476D1CE4E5B9);
    z ^= z >> 29;
    z = z.wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 32;
    ((z & 0xFFFF) as f32 / 65535.0) - 0.5
}

fn bf16_t(seed: u64, rows: usize, cols: usize, scale: f32, dev: &Device) -> Tensor {
    let v: Vec<f32> = (0..rows * cols).map(|i| frand(seed, i) * scale).collect();
    Tensor::from_vec(v, (rows, cols), dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

fn host_f32(t: &Tensor) -> Vec<f32> {
    t.contiguous()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

const K: usize = 128;
const N: usize = 128;
const S0_W: usize = 64;
const S0_R: usize = 16;
const S1_OFF: usize = 80;
const S1_W: usize = 48;
const S1_R: usize = 8;
const S1_SCALE: f64 = 1.5;

fn cpu_pair(seed: u64, r: usize, width: usize) -> (Tensor, Tensor) {
    let cpu = Device::Cpu;
    (
        bf16_t(seed, r, K, 0.5, &cpu),
        bf16_t(seed ^ 0xB, width, r, 0.8, &cpu),
    )
}

fn slices_on(dev: &Device) -> Vec<ServingDeltaSlice> {
    let (a0, b0) = cpu_pair(21, S0_R, S0_W);
    let (a1, b1) = cpu_pair(22, S1_R, S1_W);
    let mv = |t: &Tensor| t.to_device(dev).unwrap();
    vec![
        ServingDeltaSlice::from_peft_pair("t.s0", 0, S0_W, K, &mv(&a0), &mv(&b0), 1.0).unwrap(),
        ServingDeltaSlice::from_peft_pair("t.s1", S1_OFF, S1_W, K, &mv(&a1), &mv(&b1), S1_SCALE)
            .unwrap(),
    ]
}

fn reference_delta_f32(x_host: &Tensor, m: usize) -> Vec<f32> {
    let xf = x_host.to_dtype(DType::F32).unwrap();
    let mut d = vec![0f32; m * N];
    for (seed, r, off, w, scale) in [
        (21u64, S0_R, 0usize, S0_W, 1.0f64),
        (22u64, S1_R, S1_OFF, S1_W, S1_SCALE),
    ] {
        let (a, b) = cpu_pair(seed, r, w);
        let h = xf
            .matmul(&a.to_dtype(DType::F32).unwrap().t().unwrap().contiguous().unwrap())
            .unwrap();
        let dd = h
            .matmul(&b.to_dtype(DType::F32).unwrap().t().unwrap().contiguous().unwrap())
            .unwrap()
            .affine(scale, 0.0)
            .unwrap();
        let dd = host_f32(&dd);
        for row in 0..m {
            for j in 0..w {
                d[row * N + off + j] += dd[row * w + j];
            }
        }
    }
    d
}

fn parity_max_err(got: &[f32], base: &[f32], delta_ref: &[f32]) -> (f32, f32) {
    let mut max_err = 0f32;
    let mut max_delta = 0f32;
    for i in 0..got.len() {
        max_err = max_err.max((got[i] - (base[i] + delta_ref[i])).abs());
        max_delta = max_delta.max(delta_ref[i].abs());
    }
    (max_err, max_delta)
}

const PARITY_TOL_ABS_COVERS_BF16_ROUNDING_OF_DELTAS_NEAR_5: f32 = 0.1;

#[test]
fn cpu_candle_path_serves_the_same_delta_the_reference_defines() {
    let cpu = Device::Cpu;
    let m = 3usize;
    let w = bf16_t(11, N, K, 0.5, &cpu);
    let linear = Linear::new(w, None).unwrap();
    let x = bf16_t(12, m, K, 1.0, &cpu);
    let base = host_f32(&linear.forward(&x).unwrap());
    linear
        .attach_lora(ServingLoraHook::new(K, N, slices_on(&cpu)).unwrap())
        .unwrap();
    let got = host_f32(&linear.forward(&x).unwrap());
    let delta_ref = reference_delta_f32(&x, m);
    let (max_err, max_delta) = parity_max_err(&got, &base, &delta_ref);
    assert!(
        max_delta > 0.5,
        "reference delta magnitude {max_delta} too small for the tolerance to mean anything"
    );
    assert!(
        max_err <= PARITY_TOL_ABS_COVERS_BF16_ROUNDING_OF_DELTAS_NEAR_5,
        "cpu candle path diverged from the mathematical reference: max err {max_err}"
    );
}

#[cfg(feature = "cuda")]
mod cuda_kernel_parity {
    use super::*;

    const GAP: std::ops::Range<usize> = S0_W..S1_OFF;

    fn bits_of(t: &Tensor) -> Vec<u32> {
        host_f32(
            &t.to_dtype(DType::BF16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap(),
        )
        .into_iter()
        .map(f32::to_bits)
        .collect()
    }

    fn cuda_missing(test: &str) {
        if std::env::var("NV_KERNELS_PARITY_ALLOW_SKIP").as_deref() == Ok("1") {
            eprintln!(
                "SKIP (NV_KERNELS_PARITY_ALLOW_SKIP=1): {test}: no CUDA device 0. Not a pass."
            );
            return;
        }
        panic!(
            "{test}: no CUDA device 0. This is the capture-legal serving-lora kernel parity \
             gate; set NV_KERNELS_PARITY_ALLOW_SKIP=1 to skip on purpose."
        );
    }

    fn hooked_pair(dev: &Device) -> (Linear, Linear) {
        let w = bf16_t(11, N, K, 0.5, &Device::Cpu);
        let clean = Linear::new(w.to_device(dev).unwrap(), None).unwrap();
        let hooked = Linear::new(
            bf16_t(11, N, K, 0.5, &Device::Cpu).to_device(dev).unwrap(),
            None,
        )
        .unwrap();
        hooked
            .attach_lora(ServingLoraHook::new(K, N, slices_on(dev)).unwrap())
            .unwrap();
        (clean, hooked)
    }

    #[test]
    fn m1_kernel_delta_matches_the_reference_and_the_mutation_control_trips() {
        let Ok(cuda) = Device::new_cuda(0) else {
            cuda_missing("m1_kernel_delta_matches_the_reference_and_the_mutation_control_trips");
            return;
        };
        let x_host = bf16_t(12, 1, K, 1.0, &Device::Cpu);
        let x = x_host.to_device(&cuda).unwrap();
        let (clean, hooked) = hooked_pair(&cuda);
        let base = host_f32(&clean.forward(&x).unwrap());
        let got = host_f32(&hooked.forward(&x).unwrap());
        let delta_ref = reference_delta_f32(&x_host, 1);
        let (max_err, max_delta) = parity_max_err(&got, &base, &delta_ref);
        assert!(
            max_delta > 0.5,
            "reference delta magnitude {max_delta} too small to gate on"
        );
        assert!(
            max_err <= PARITY_TOL_ABS_COVERS_BF16_ROUNDING_OF_DELTAS_NEAR_5,
            "m=1 nv-kernels delta diverged from the f32 reference: max err {max_err}"
        );
        let mutated: Vec<f32> = delta_ref.iter().map(|d| d * 1.10).collect();
        let (mutated_err, _) = parity_max_err(&got, &base, &mutated);
        assert!(
            mutated_err > PARITY_TOL_ABS_COVERS_BF16_ROUNDING_OF_DELTAS_NEAR_5,
            "mutation control: a 10% miscomputed reference must fail parity (err {mutated_err}); \
             a tolerance that passes it would also pass a broken kernel"
        );
    }

    #[test]
    fn m1_gap_columns_stay_bitwise_base_and_windows_match_the_full_forward() {
        let Ok(cuda) = Device::new_cuda(0) else {
            cuda_missing("m1_gap_columns_stay_bitwise_base_and_windows_match_the_full_forward");
            return;
        };
        let x = bf16_t(12, 1, K, 1.0, &Device::Cpu).to_device(&cuda).unwrap();
        let (clean, hooked) = hooked_pair(&cuda);
        let base_bits = bits_of(&clean.forward(&x).unwrap());
        let full = hooked.forward(&x).unwrap();
        let full_bits = bits_of(&full);
        assert_eq!(
            &base_bits[GAP], &full_bits[GAP],
            "columns between slices are covered only by a zero stack; the expand kernel must \
             leave them bitwise base"
        );
        assert_ne!(
            &base_bits[0..S0_W],
            &full_bits[0..S0_W],
            "covered columns must change or the kernel path never fired"
        );
        for (off, len) in [(0usize, S0_W), (S1_OFF, S1_W), (32usize, 64usize), (0usize, N)] {
            let win = host_f32(&hooked.forward_rows(&x, off, len).unwrap());
            let full_cols = host_f32(&full.narrow(1, off, len).unwrap());
            let mut max_err = 0f32;
            for i in 0..win.len() {
                max_err = max_err.max((win[i] - full_cols[i]).abs());
            }
            assert!(
                max_err <= 0.02,
                "windowed fused kernel ({off},{len}) diverged from the full forward's columns \
                 by {max_err}"
            );
        }
    }

    #[test]
    fn m2_takes_the_candle_fallback_off_capture_and_matches_the_reference() {
        let Ok(cuda) = Device::new_cuda(0) else {
            cuda_missing("m2_takes_the_candle_fallback_off_capture_and_matches_the_reference");
            return;
        };
        let m = 2usize;
        let x_host = bf16_t(13, m, K, 1.0, &Device::Cpu);
        let x = x_host.to_device(&cuda).unwrap();
        let (clean, hooked) = hooked_pair(&cuda);
        let base = host_f32(&clean.forward(&x).unwrap());
        let got = host_f32(&hooked.forward(&x).unwrap());
        let delta_ref = reference_delta_f32(&x_host, m);
        let (max_err, max_delta) = parity_max_err(&got, &base, &delta_ref);
        assert!(max_delta > 0.5);
        assert!(
            max_err <= PARITY_TOL_ABS_COVERS_BF16_ROUNDING_OF_DELTAS_NEAR_5,
            "m=2 candle fallback diverged from the f32 reference: max err {max_err}; the \
             dispatch boundary must not change the delta a request sees"
        );
    }
}
