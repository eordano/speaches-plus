use half::bf16;
use nv_layers::backend::{
    availability, kind_supports, missing_on, resolve_from, Backend, BackendError, BackendKind,
    BackendSel, KernelId,
};

fn wgpu_require() -> bool {
    std::env::var("NV_KERNELS_WGPU_ALLOW_SKIP").as_deref() != Ok("1")
}

fn cuda_require() -> bool {
    std::env::var("NV_KERNELS_PARITY_ALLOW_SKIP").as_deref() != Ok("1")
}

fn open_backends(test: &str) -> Vec<Backend> {
    let mut out = vec![Backend::open(BackendKind::Cpu).unwrap()];
    if cfg!(feature = "wgpu") {
        match Backend::open(BackendKind::Wgpu) {
            Ok(b) => {
                eprintln!("{test}: wgpu = {}", b.describe());
                out.push(b);
            }
            Err(e) => {
                if wgpu_require() {
                    panic!(
                        "{test}: no wgpu backend: {e}. Set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip \
                         on purpose."
                    );
                }
                eprintln!("{test}: SKIP wgpu: {e}");
            }
        }
    }
    if cfg!(feature = "cuda") {
        match Backend::open(BackendKind::Cuda) {
            Ok(b) => {
                eprintln!("{test}: cuda = {}", b.describe());
                out.push(b);
            }
            Err(e) => {
                if cuda_require() {
                    panic!("{test}: NV_KERNELS_PARITY_REQUIRE=1 but {e}");
                }
                eprintln!("{test}: SKIP cuda: {e}");
            }
        }
    }
    out
}

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
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn small_int(&mut self) -> i32 {
        (self.next_u32() % 5) as i32 - 2
    }
}

#[test]
fn resolver_matrix_is_pure() {
    let ok: &dyn Fn() -> Result<(), String> = &|| Ok(());
    let no: &dyn Fn() -> Result<(), String> = &|| Err("forced probe failure".to_string());

    assert_eq!(
        resolve_from(BackendSel::Auto, ok, ok).unwrap(),
        BackendKind::Cuda
    );
    assert_eq!(
        resolve_from(BackendSel::Auto, no, ok).unwrap(),
        BackendKind::Wgpu
    );
    assert_eq!(
        resolve_from(BackendSel::Auto, no, no).unwrap(),
        BackendKind::Cpu
    );
    assert_eq!(
        resolve_from(BackendSel::Cuda, ok, no).unwrap(),
        BackendKind::Cuda
    );
    assert_eq!(
        resolve_from(BackendSel::Wgpu, no, ok).unwrap(),
        BackendKind::Wgpu
    );
    assert_eq!(
        resolve_from(BackendSel::Cpu, ok, ok).unwrap(),
        BackendKind::Cpu
    );

    match resolve_from(BackendSel::Cuda, no, ok) {
        Err(BackendError::Unavailable { kind, reason }) => {
            assert_eq!(kind, BackendKind::Cuda);
            assert!(reason.contains("forced probe failure"));
        }
        other => panic!("explicit cuda with failing probe must error, got {other:?}"),
    }
    match resolve_from(BackendSel::Wgpu, ok, no) {
        Err(BackendError::Unavailable { kind, reason }) => {
            assert_eq!(kind, BackendKind::Wgpu);
            assert!(reason.contains("forced probe failure"));
        }
        other => panic!("explicit wgpu must not fall back to cuda, got {other:?}"),
    }

    assert_eq!(BackendSel::parse(""), Some(BackendSel::Auto));
    assert_eq!(BackendSel::parse("auto"), Some(BackendSel::Auto));
    assert_eq!(BackendSel::parse("CUDA"), Some(BackendSel::Cuda));
    assert_eq!(BackendSel::parse(" wgpu "), Some(BackendSel::Wgpu));
    assert_eq!(BackendSel::parse("cpu"), Some(BackendSel::Cpu));
    assert_eq!(BackendSel::parse("metal"), None);
    eprintln!("resolver_matrix_is_pure: 8 resolution cases + 6 parse cases checked");
}

#[test]
fn capability_tables_are_honest() {
    for k in KernelId::ALL {
        assert!(kind_supports(BackendKind::Cuda, k), "cuda must support {k}");
    }
    let wgpu_missing = missing_on(BackendKind::Wgpu, &KernelId::ALL);
    assert_eq!(
        wgpu_missing,
        vec![KernelId::MarlinGemmW4a16],
        "wgpu missing set drifted; update kind_supports against src/wgpu_backend/kernels/"
    );
    assert!(
        missing_on(BackendKind::Wgpu, &KernelId::DENSE_DECODE_PATH).is_empty(),
        "every dense-decode-path kernel module must exist on wgpu"
    );
    let cpu_have: Vec<KernelId> = KernelId::ALL
        .into_iter()
        .filter(|k| kind_supports(BackendKind::Cpu, *k))
        .collect();
    assert_eq!(
        cpu_have,
        vec![KernelId::Rmsnorm, KernelId::Silu, KernelId::GemvBf16]
    );
    eprintln!(
        "capability_tables_are_honest: {} kernels; wgpu missing = {:?}; cpu has = {:?}",
        KernelId::ALL.len(),
        wgpu_missing.iter().map(|k| k.name()).collect::<Vec<_>>(),
        cpu_have.iter().map(|k| k.name()).collect::<Vec<_>>()
    );
    for (kind, res) in availability() {
        eprintln!("availability: {kind} -> {res:?}");
    }
}

#[test]
fn gemv_bf16_bitwise_identical_across_all_backends() {
    let backends = open_backends("gemv_bf16_bitwise_identical_across_all_backends");
    let (n, k) = (33usize, 48usize);
    let mut rng = Lcg(0x5eed);
    let w: Vec<u16> = (0..n * k)
        .map(|_| bf16::from_f32(rng.small_int() as f32).to_bits())
        .collect();
    let x: Vec<u16> = (0..k)
        .map(|_| bf16::from_f32(rng.small_int() as f32).to_bits())
        .collect();
    let reference: Vec<u16> = (0..n)
        .map(|row| {
            let mut acc = 0i64;
            for j in 0..k {
                acc += bf16::from_bits(w[row * k + j]).to_f32() as i64
                    * bf16::from_bits(x[j]).to_f32() as i64;
            }
            assert!(acc.abs() <= 256, "test inputs must stay bf16-exact");
            bf16::from_f32(acc as f32).to_bits()
        })
        .collect();

    let nz_ref = reference
        .iter()
        .filter(|b| bf16::from_bits(**b).to_f32() != 0.0)
        .count();
    assert!(
        nz_ref * 2 > n,
        "degenerate reference: only {nz_ref}/{n} rows are nonzero, so a kernel that returned \
         zeros would match it"
    );

    let mut results: Vec<(BackendKind, Vec<u16>)> = Vec::new();
    for b in &backends {
        let y = b.gemv_bf16(&w, &x, n, k).unwrap();
        let vs_ref = y
            .iter()
            .zip(reference.iter())
            .filter(|(a, r)| a != r)
            .count();
        eprintln!(
            "gemv_bf16 n={n} k={k} on {}: {}/{} words differ vs exact integer reference",
            b.kind(),
            vs_ref,
            n
        );
        assert_eq!(vs_ref, 0, "{} diverged from the exact reference", b.kind());
        results.push((b.kind(), y));
    }
    for pair in results.windows(2) {
        let mism = pair[0]
            .1
            .iter()
            .zip(pair[1].1.iter())
            .filter(|(a, b)| a != b)
            .count();
        eprintln!(
            "gemv_bf16: {} vs {}: {}/{} words differ",
            pair[0].0, pair[1].0, mism, n
        );
        assert_eq!(mism, 0, "{} and {} disagree bitwise", pair[0].0, pair[1].0);
    }
    assert!(
        !results.is_empty(),
        "at least the cpu backend must have run"
    );
    eprintln!(
        "gemv_bf16_bitwise_identical_across_all_backends: backends = {:?}",
        results.iter().map(|(k, _)| k.name()).collect::<Vec<_>>()
    );
}

#[test]
fn rmsnorm_and_silu_mul_agree_across_backends() {
    let backends = open_backends("rmsnorm_and_silu_mul_agree_across_backends");
    let (batch, hidden) = (4usize, 256usize);
    let mut rng = Lcg(0xabcdef);
    let x: Vec<f32> = (0..batch * hidden).map(|_| rng.next_f32() * 3.0).collect();
    let wgt: Vec<f32> = (0..hidden).map(|_| rng.next_f32() + 1.5).collect();
    let gate: Vec<f32> = (0..batch * hidden).map(|_| rng.next_f32() * 2.0).collect();

    let rms_host: Vec<f64> = (0..batch)
        .flat_map(|r| {
            let row = &x[r * hidden..(r + 1) * hidden];
            let ms = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / hidden as f64;
            let inv = 1.0 / (ms + 1e-6f64).sqrt();
            (0..hidden)
                .map(|c| row[c] as f64 * inv * wgt[c] as f64)
                .collect::<Vec<f64>>()
        })
        .collect();
    let silu_host: Vec<f64> = x
        .iter()
        .zip(gate.iter())
        .map(|(a, g)| {
            let a = *a as f64;
            (a / (1.0 + (-a).exp())) * *g as f64
        })
        .collect();

    let spread = |v: &[f64]| -> f64 {
        let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    };
    assert!(
        spread(&rms_host) > 1e-3 && spread(&silu_host) > 1e-3,
        "degenerate fixture: rmsnorm spread {:.3e}, silu spread {:.3e}; a kernel returning a \
         constant would pass",
        spread(&rms_host),
        spread(&silu_host)
    );

    let cpu = &backends[0];
    assert_eq!(cpu.kind(), BackendKind::Cpu);
    let rms_ref = cpu.rmsnorm_f32(&x, &wgt, batch, hidden, 1e-6).unwrap();
    let silu_ref = cpu.silu_mul_f32(&x, &gate).unwrap();

    let vs_host = |got: &[f32], want: &[f64]| -> f64 {
        got.iter()
            .zip(want.iter())
            .map(|(a, r)| (*a as f64 - r).abs())
            .fold(0f64, f64::max)
    };
    let cpu_rms_err = vs_host(&rms_ref, &rms_host);
    let cpu_silu_err = vs_host(&silu_ref, &silu_host);
    eprintln!(
        "cpu vs f64 host reference: rmsnorm max_abs {cpu_rms_err:e}, silu_mul max_abs \
         {cpu_silu_err:e}"
    );
    assert!(
        cpu_rms_err < 1e-5,
        "the cpu backend itself disagrees with the f64 host rmsnorm by {cpu_rms_err:e}; it is \
         not fit to be the cross-backend reference"
    );
    assert!(
        cpu_silu_err < 1e-5,
        "the cpu backend itself disagrees with the f64 host silu_mul by {cpu_silu_err:e}"
    );

    for b in &backends[1..] {
        let rms = b.rmsnorm_f32(&x, &wgt, batch, hidden, 1e-6).unwrap();
        let silu = b.silu_mul_f32(&x, &gate).unwrap();
        let rms_max = rms
            .iter()
            .zip(rms_ref.iter())
            .map(|(a, r)| (a - r).abs() as f64)
            .fold(0f64, f64::max);
        let silu_max = silu
            .iter()
            .zip(silu_ref.iter())
            .map(|(a, r)| (a - r).abs() as f64)
            .fold(0f64, f64::max);
        eprintln!(
            "{}: rmsnorm max_abs vs cpu = {rms_max:e} ({} elems), silu_mul max_abs vs cpu = {silu_max:e} ({} elems)",
            b.kind(),
            rms.len(),
            silu.len()
        );
        assert!(rms_max < 1e-5, "{} rmsnorm drifted: {rms_max:e}", b.kind());
        assert!(
            silu_max < 1e-5,
            "{} silu_mul drifted: {silu_max:e}",
            b.kind()
        );
    }
    eprintln!(
        "rmsnorm_and_silu_mul_agree_across_backends: {} backend(s) compared against cpu",
        backends.len() - 1
    );
}

#[test]
fn cuda_and_wgpu_are_bitexact_on_parity_proven_kernels() {
    if !(cfg!(feature = "cuda") && cfg!(feature = "wgpu")) {
        eprintln!(
            "cuda_and_wgpu_are_bitexact_on_parity_proven_kernels: SKIPPED, NOT PASSED -- this \
             test needs BOTH the cuda and wgpu features and this binary was built with \
             cuda={} wgpu={}. It has no file-level #![cfg], so unlike a parity_* suite it does \
             NOT print `0 passed`: it returns from inside a suite whose other tests do real \
             work, and the cross-backend bit-exactness claim vanishes with no 0.00s tell. \
             Rebuild with NVK_FEATURES=cuda,wgpu.",
            cfg!(feature = "cuda"),
            cfg!(feature = "wgpu")
        );
        return;
    }
    let backends = open_backends("cuda_and_wgpu_are_bitexact_on_parity_proven_kernels");
    let cuda = backends.iter().find(|b| b.kind() == BackendKind::Cuda);
    let wgpu = backends.iter().find(|b| b.kind() == BackendKind::Wgpu);
    let (Some(cuda), Some(wgpu)) = (cuda, wgpu) else {
        panic!(
            "cuda_and_wgpu_are_bitexact_on_parity_proven_kernels: this binary has both features \
             but only opened cuda={} wgpu={}. A cross-backend bit-exactness claim needs both \
             devices; nothing was compared. Set NV_KERNELS_PARITY_ALLOW_SKIP=1 or \
             NV_KERNELS_WGPU_ALLOW_SKIP=1 and read the SKIP line above, rather than letting this \
             return green.",
            cuda.is_some(),
            wgpu.is_some()
        );
    };

    let (n, k) = (64usize, 512usize);
    let mut rng = Lcg(0x77aa);
    let w: Vec<u16> = (0..n * k)
        .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
        .collect();
    let x: Vec<u16> = (0..k)
        .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
        .collect();
    let yc = cuda.gemv_bf16(&w, &x, n, k).unwrap();
    let yw = wgpu.gemv_bf16(&w, &x, n, k).unwrap();
    let gemv_mism = yc.iter().zip(yw.iter()).filter(|(a, b)| a != b).count();
    eprintln!("gemv_bf16 random n={n} k={k}: cuda vs wgpu {gemv_mism}/{n} words differ");
    assert_eq!(gemv_mism, 0);

    let (nh, nkv, hd, total, start) = (4usize, 2usize, 64usize, 96usize, 8usize);
    let q: Vec<f32> = (0..nh * hd).map(|_| rng.next_f32()).collect();
    let kc: Vec<f32> = (0..total * nkv * hd).map(|_| rng.next_f32()).collect();
    let vc: Vec<f32> = (0..total * nkv * hd).map(|_| rng.next_f32()).collect();
    let oc = cuda
        .attn_decode_f32(&q, &kc, &vc, nh, nkv, hd, start, total)
        .unwrap();
    let ow = wgpu
        .attn_decode_f32(&q, &kc, &vc, nh, nkv, hd, start, total)
        .unwrap();
    let attn_mism = oc
        .iter()
        .zip(ow.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!(
        "attn_decode_f32 nh={nh} nkv={nkv} hd={hd} start={start} total={total}: cuda vs wgpu {attn_mism}/{} words differ",
        oc.len()
    );
    assert_eq!(attn_mism, 0);
}

#[test]
fn missing_kernel_is_a_clean_error_not_a_fallback() {
    if cfg!(feature = "wgpu") {
        match Backend::open(BackendKind::Wgpu) {
            Ok(b) => {
                let err = b.marlin_workspace_elems().unwrap_err();
                let msg = err.to_string();
                eprintln!("wgpu marlin dispatch -> {msg}");
                assert!(matches!(
                    err,
                    BackendError::MissingKernel {
                        kind: BackendKind::Wgpu,
                        kernel: KernelId::MarlinGemmW4a16
                    }
                ));
                assert!(msg.contains("marlin_gemm_w4a16"));
                assert!(msg.contains("wgpu"));
                assert!(msg.contains("cuda"));
                assert!(
                    b.require(KernelId::MoeGroupedGemmNvfp4).is_ok(),
                    "moe_wgpu::try_forward dispatches wgpu_backend::kernels::moe_grouped_gemm; \
                     requiring it on wgpu must not be an error"
                );
                assert!(b.require(KernelId::FlashDecode).is_ok());
            }
            Err(e) => {
                if wgpu_require() {
                    panic!(
                        "missing_kernel test: no wgpu backend: {e}. Set \
                         NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
                    );
                }
                eprintln!("missing_kernel test: SKIP wgpu part: {e}");
            }
        }
    }

    let cpu = Backend::open(BackendKind::Cpu).unwrap();
    let err = cpu
        .attn_decode_f32(&[0.0; 8], &[0.0; 16], &[0.0; 16], 1, 1, 8, 0, 2)
        .unwrap_err();
    let msg = err.to_string();
    eprintln!("cpu attn_decode dispatch -> {msg}");
    assert!(matches!(
        err,
        BackendError::MissingKernel {
            kind: BackendKind::Cpu,
            kernel: KernelId::AttnDecode
        }
    ));
    assert!(msg.contains("attn_decode"));
    assert!(msg.contains("cpu"));
    assert!(msg.contains("no silent fallback"));
}
