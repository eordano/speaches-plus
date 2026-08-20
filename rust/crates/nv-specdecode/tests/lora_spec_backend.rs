use candle_core::{DType, Device, Tensor};
use half::bf16;
use nv_layers::linear::Linear;

fn frand(seed: u64, i: usize) -> f32 {
    let mut z = seed
        .wrapping_add(0x9E3779B97F4A7C15u64.wrapping_mul(i as u64 + 1))
        .wrapping_mul(0xBF58476D1CE4E5B9);
    z ^= z >> 29;
    z = z.wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 32;
    ((z & 0xFFFF) as f32 / 65535.0) - 0.5
}

fn bf_vec(seed: u64, n: usize, scale: f32) -> Vec<bf16> {
    (0..n)
        .map(|i| bf16::from_f32(frand(seed, i) * scale))
        .collect()
}

fn to_tensor2(v: &[bf16], rows: usize, cols: usize, dev: &Device) -> Tensor {
    Tensor::from_vec(v.to_vec(), (rows, cols), dev).unwrap()
}

fn bits_of(t: &Tensor) -> Vec<u16> {
    t.contiguous()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<bf16>()
        .unwrap()
        .into_iter()
        .map(|v| v.to_bits())
        .collect()
}

fn diff_count(a: &[u16], b: &[u16]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

#[test]
fn no_adapter_cpu_forward_is_bitwise_the_pre_lora_algorithm() {
    let dev = Device::Cpu;
    let k = 96usize;
    let n = 80usize;
    let m = 7usize;
    let w = bf_vec(1, n * k, 0.5);
    let x = bf_vec(2, m * k, 1.0);
    let linear = Linear::new(to_tensor2(&w, n, k, &dev), None).unwrap();
    let xt = to_tensor2(&x, m, k, &dev);

    let got = linear.forward(&xt).unwrap();
    assert_eq!(got.dtype(), DType::BF16);

    let w_f32 = to_tensor2(&w, n, k, &dev).to_dtype(DType::F32).unwrap();
    let x_f32 = xt.to_dtype(DType::F32).unwrap();
    let want = x_f32
        .matmul(&w_f32.t().unwrap().contiguous().unwrap())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let gb = bits_of(&got);
    let wb = bits_of(&want);
    let d = diff_count(&gb, &wb);
    println!(
        "no_adapter cpu forward vs f32-upcast reference: {d}/{} words differ",
        gb.len()
    );
    assert_eq!(
        d, 0,
        "no-adapter forward must be bitwise the fallback matmul"
    );
    assert!(!linear.has_lora());
}

#[cfg(feature = "wgpu")]
fn wgpu_missing(test: &str, reason: &str) {
    if std::env::var("NV_KERNELS_WGPU_ALLOW_SKIP").as_deref() == Ok("1") {
        eprintln!(
            "SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1): {test}: no wgpu context: {reason}. Not a pass."
        );
        return;
    }
    panic!(
        "{test}: no wgpu context: {reason}. Set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
    );
}

#[cfg(all(feature = "cuda", feature = "wgpu"))]
fn cuda_missing(test: &str) {
    if std::env::var("NV_KERNELS_PARITY_ALLOW_SKIP").as_deref() == Ok("1") {
        eprintln!("SKIP (NV_KERNELS_PARITY_ALLOW_SKIP=1): {test}: no CUDA device 0. Not a pass.");
        return;
    }
    panic!(
        "{test}: no CUDA device 0. This is the cuda-vs-wgpu LoRA delta bit-exactness gate; set \
         NV_KERNELS_PARITY_ALLOW_SKIP=1 to skip on purpose."
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_hook_through_linear_forward_no_cuda_required() {
    use nv_kernels::wgpu_backend::device::shared_or_reason;
    use nv_layers::lora_slots::LoraModuleSpec;
    use nv_specdecode::lora_spec::{synth_adapter, WgpuLoraRuntime};

    let ctx = match shared_or_reason() {
        Ok(c) => c,
        Err(r) => {
            wgpu_missing("wgpu_hook_through_linear_forward_no_cuda_required", &r);
            return;
        }
    };
    let dev = Device::Cpu;
    let k = 128usize;
    let n = 96usize;
    let m = 6usize;
    let rank = 8usize;

    let w = bf_vec(11, n * k, 0.5);
    let x = bf_vec(12, m * k, 1.0);
    let linear = Linear::new(to_tensor2(&w, n, k, &dev), None).unwrap();
    let xt = to_tensor2(&x, m, k, &dev);
    let base = bits_of(&linear.forward(&xt).unwrap());

    let specs = vec![LoraModuleSpec::new("proj", k, n)];
    let adapter = synth_adapter(&specs, rank, 0.2, 99, &dev).unwrap();
    let runtime = WgpuLoraRuntime::install(
        ctx,
        &[(&linear, vec![LoraModuleSpec::new("proj", k, n)])],
        &adapter,
        rank,
        64,
        &dev,
    )
    .unwrap();
    assert_eq!(runtime.hooked_layers(), 1);
    assert!(linear.has_lora());

    let disarmed = bits_of(&linear.forward(&xt).unwrap());
    let d0 = diff_count(&base, &disarmed);
    println!("disarmed forward vs base: {d0}/{} words differ", base.len());
    assert_eq!(d0, 0);

    runtime.arm(m).unwrap();
    assert!(runtime.armed());
    let armed = linear.forward(&xt).unwrap();
    let armed_bits = bits_of(&armed);
    let d1 = diff_count(&base, &armed_bits);
    println!("armed forward vs base: {d1}/{} words differ", base.len());
    assert!(d1 > 0, "armed wgpu lora must change the output");

    let stack = runtime.manager().stack("proj").unwrap();
    let a = stack.slot_a(0).unwrap().to_dtype(DType::F32).unwrap();
    let b = stack.slot_b(0).unwrap().to_dtype(DType::F32).unwrap();
    let delta = xt
        .to_dtype(DType::F32)
        .unwrap()
        .matmul(&a.t().unwrap().contiguous().unwrap())
        .unwrap()
        .matmul(&b.t().unwrap().contiguous().unwrap())
        .unwrap();
    let base_t = Tensor::from_vec(
        base.iter().map(|&b| bf16::from_bits(b)).collect::<Vec<_>>(),
        (m, n),
        &dev,
    )
    .unwrap();
    let want = base_t
        .to_dtype(DType::F32)
        .unwrap()
        .add(&delta)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let got = armed
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let mut max_err = 0f32;
    for (gr, wr) in got.iter().zip(want.iter()) {
        for (g, w) in gr.iter().zip(wr.iter()) {
            let lim = 3e-2 + 3e-2 * w.abs();
            let err = (g - w).abs();
            assert!(err <= lim, "wgpu armed forward vs oracle: got {g} want {w}");
            max_err = max_err.max(err);
        }
    }
    println!("armed forward vs candle oracle: max abs err {max_err}");

    runtime.disarm();
    let after = bits_of(&linear.forward(&xt).unwrap());
    assert_eq!(
        diff_count(&base, &after),
        0,
        "disarm must restore base bitwise"
    );
    linear.detach_lora();
    assert!(!linear.has_lora());
    let detached = bits_of(&linear.forward(&xt).unwrap());
    assert_eq!(
        diff_count(&base, &detached),
        0,
        "detach must restore base bitwise"
    );
    println!("disarm+detach restore base: 0/{} words differ", base.len());
}

#[cfg(all(feature = "cuda", feature = "wgpu"))]
#[test]
fn cuda_and_wgpu_lora_deltas_are_bit_identical() {
    use nv_kernels::wgpu_backend::device::shared_or_reason;
    use nv_layers::linear::LoraDeltaHook;
    use nv_layers::lora_slots::{LoraDispatch, LoraHook, LoraModuleSpec, LoraSlotManager};
    use nv_specdecode::lora_spec::{synth_adapter, LoraTokenMap, WgpuLoraHook};

    let Ok(cuda) = Device::new_cuda(0) else {
        cuda_missing("cuda_and_wgpu_lora_deltas_are_bit_identical");
        return;
    };
    let ctx = match shared_or_reason() {
        Ok(c) => c,
        Err(r) => {
            wgpu_missing("cuda_and_wgpu_lora_deltas_are_bit_identical", &r);
            return;
        }
    };
    let cpu = Device::Cpu;

    let k = 128usize;
    let q_dim = 64usize;
    let kv_dim = 32usize;
    let n_total = q_dim + 2 * kv_dim;
    let rank = 16usize;
    let max_loras = 2usize;
    let max_tokens = 128usize;

    let specs = || {
        vec![
            LoraModuleSpec::new("q", k, q_dim),
            LoraModuleSpec::new("k", k, kv_dim),
            LoraModuleSpec::new("v", k, kv_dim),
        ]
    };
    let ad0_cuda = synth_adapter(&specs(), rank, 0.2, 1000, &cuda).unwrap();
    let ad1_cuda = synth_adapter(&specs(), 8, 0.3, 2000, &cuda).unwrap();
    let ad0_cpu = synth_adapter(&specs(), rank, 0.2, 1000, &cpu).unwrap();
    let ad1_cpu = synth_adapter(&specs(), 8, 0.3, 2000, &cpu).unwrap();

    let mut mgr_cuda = LoraSlotManager::new(max_loras, rank, &specs(), DType::BF16, &cuda).unwrap();
    let mut mgr_cpu = LoraSlotManager::new(max_loras, rank, &specs(), DType::BF16, &cpu).unwrap();
    let slot0 = mgr_cuda.activate(100, &ad0_cuda).unwrap();
    let slot1 = mgr_cuda.activate(101, &ad1_cuda).unwrap();
    assert_eq!(slot0, mgr_cpu.activate(100, &ad0_cpu).unwrap());
    assert_eq!(slot1, mgr_cpu.activate(101, &ad1_cpu).unwrap());

    let dispatch = LoraDispatch::new(&cuda, max_tokens, max_loras).unwrap();
    let cuda_hook = LoraHook::from_stacks(
        dispatch.clone(),
        &[
            mgr_cuda.stack("q").unwrap(),
            mgr_cuda.stack("k").unwrap(),
            mgr_cuda.stack("v").unwrap(),
        ],
    )
    .unwrap();

    let map = LoraTokenMap::new(max_tokens, max_loras).unwrap();
    let wgpu_hook = WgpuLoraHook::from_stacks(
        ctx,
        map.clone(),
        &[
            mgr_cpu.stack("q").unwrap(),
            mgr_cpu.stack("k").unwrap(),
            mgr_cpu.stack("v").unwrap(),
        ],
    )
    .unwrap();

    let mut total_words = 0usize;
    let mut total_diffs = 0usize;
    for m in [4usize, 96usize] {
        let mapping: Vec<i32> = (0..m)
            .map(|i| match i % 3 {
                0 => slot0 as i32,
                1 => -1,
                _ => slot1 as i32,
            })
            .collect();
        dispatch.set_mapping(&mapping).unwrap();
        map.set_mapping(&mapping).unwrap();

        let x_host = bf_vec(500 + m as u64, m * k, 1.0);
        let x_cuda = to_tensor2(&x_host, m, k, &cuda);
        let x_cpu = to_tensor2(&x_host, m, k, &cpu);

        for (win, wlen, tag) in [
            (None, n_total, "full"),
            (Some((q_dim, kv_dim)), kv_dim, "window_k"),
        ] {
            let zeros = vec![bf16::ZERO; m * wlen];
            let y_cuda = to_tensor2(&zeros, m, wlen, &cuda);
            cuda_hook.apply(&x_cuda, &y_cuda, win).unwrap();
            let cuda_bits = bits_of(&y_cuda);

            let y_cpu = to_tensor2(&zeros, m, wlen, &cpu);
            let out = LoraDeltaHook::apply(wgpu_hook.as_ref(), &x_cpu, &y_cpu, win)
                .unwrap()
                .expect("wgpu hook must return a replacement tensor");
            let wgpu_bits = bits_of(&out);

            let d = diff_count(&cuda_bits, &wgpu_bits);
            println!(
                "m={m} {tag}: cuda vs wgpu lora delta {d}/{} words differ",
                cuda_bits.len()
            );
            total_words += cuda_bits.len();
            total_diffs += d;
            assert_eq!(
                d, 0,
                "m={m} {tag}: cuda and wgpu deltas must be bit-identical"
            );

            let touched = cuda_bits.iter().filter(|&&b| b != 0).count();
            assert!(touched > 0, "m={m} {tag}: delta must be non-trivial");
        }
    }
    println!("total: {total_diffs}/{total_words} words differ across all cases");

    dispatch.disarm();
    map.disarm();
    let m = 4usize;
    let x_host = bf_vec(777, m * k, 1.0);
    let x_cpu = to_tensor2(&x_host, m, k, &cpu);
    let y_cpu = to_tensor2(&vec![bf16::ZERO; m * n_total], m, n_total, &cpu);
    let out = LoraDeltaHook::apply(wgpu_hook.as_ref(), &x_cpu, &y_cpu, None).unwrap();
    assert!(out.is_none(), "disarmed wgpu hook must be a no-op");
}
