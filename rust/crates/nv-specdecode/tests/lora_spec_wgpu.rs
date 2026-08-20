#![cfg(feature = "wgpu")]

use candle_core::{DType, Device, Tensor};
use half::bf16;
use nv_kernels::wgpu_backend::WgpuContext;
use nv_layers::linear::Linear;
use nv_layers::lora_slots::{LoraAdapter, LoraModuleSpec};
use nv_models::gemma4::Gemma4;
use nv_specdecode::lora_spec::{
    det_rand_tensor, synth_adapter, synth_qkv_adapter, WgpuLoraRuntime,
};

fn adapter_missing(test: &str, what: String) {
    if std::env::var("NV_KERNELS_WGPU_ALLOW_SKIP").as_deref() == Ok("1") {
        eprintln!("SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1): {test}: {what}. Not a pass.");
        return;
    }
    panic!(
        "{test}: {what}. Set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose; this gate refuses \
         to report success without dispatching anything."
    );
}

fn ctx_or_skip(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("{test}: {}", ctx.summary());
            let st = ctx.qualify();
            if !st.qualified {
                adapter_missing(test, format!("adapter not qualified: {:?}", st.reason));
                return None;
            }
            Some(ctx)
        }
        Err(e) => {
            adapter_missing(test, format!("no wgpu adapter: {e}"));
            None
        }
    }
}

fn bits(t: &Tensor) -> Vec<u16> {
    t.flatten_all()
        .unwrap()
        .to_vec1::<bf16>()
        .unwrap()
        .into_iter()
        .map(|v| v.to_bits())
        .collect()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let av = a
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let bv = b
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(av.len(), bv.len());
    av.iter()
        .zip(bv.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn run_case(name: &str, m: usize) {
    let Some(ctx) = ctx_or_skip(name) else { return };
    let device = Device::Cpu;
    let in_f = 96usize;
    let widths = [48usize, 32, 32];
    let out_f: usize = widths.iter().sum();
    let rank = 8usize;

    let weight = det_rand_tensor(11, out_f, in_f, 0.25, &device).unwrap();
    let linear = Linear::new(weight, None).unwrap();

    let names = ["m.q", "m.k", "m.v"];
    let specs: Vec<LoraModuleSpec> = names
        .iter()
        .zip(widths.iter())
        .map(|(n, w)| LoraModuleSpec::new(n.to_string(), in_f, *w))
        .collect();
    let adapter = synth_adapter(&specs, rank, 0.05, 33, &device).unwrap();

    let rt =
        WgpuLoraRuntime::install(ctx, &[(&linear, specs)], &adapter, rank, 256, &device).unwrap();
    assert_eq!(rt.hooked_layers(), 1);
    assert!(!rt.armed());

    let x = det_rand_tensor(77, m, in_f, 0.5, &device).unwrap();
    let base = linear.forward(&x).unwrap();

    rt.arm(m).unwrap();
    assert!(rt.armed());
    let hooked = linear.forward(&x).unwrap();

    rt.disarm();
    assert!(!rt.armed());
    let base2 = linear.forward(&x).unwrap();
    assert_eq!(
        bits(&base),
        bits(&base2),
        "{name}: disarm must restore base output bit-exactly"
    );

    let xf = x.to_dtype(DType::F32).unwrap();
    let mut delta_cols = Vec::new();
    for name in &names {
        let w = &adapter.modules[*name];
        let a = w.a.to_dtype(DType::F32).unwrap();
        let b = w.b.to_dtype(DType::F32).unwrap();
        let d = xf
            .matmul(&a.t().unwrap())
            .unwrap()
            .matmul(&b.t().unwrap())
            .unwrap();
        delta_cols.push((d * adapter.scaling).unwrap());
    }
    let delta = Tensor::cat(&delta_cols, 1).unwrap();
    let want = base.to_dtype(DType::F32).unwrap().add(&delta).unwrap();

    let delta_mag = max_abs_diff(&delta, &delta.zeros_like().unwrap());
    assert!(
        delta_mag > 5e-3,
        "{name}: reference delta magnitude {delta_mag} too small, test would be vacuous"
    );
    let applied = max_abs_diff(&hooked, &base);
    assert!(
        applied > 5e-3,
        "{name}: hooked output identical to base ({applied}), LoRA delta not applied"
    );
    let err = max_abs_diff(&hooked, &want);
    eprintln!("{name}: m={m} delta_mag={delta_mag} applied={applied} err={err}");
    assert!(
        err <= 2e-2,
        "{name}: hooked output deviates from f32 reference by {err}"
    );
    assert_eq!(hooked.dims2().unwrap(), (m, out_f));
}

#[test]
fn wgpu_lora_fused_path_matches_reference() {
    run_case("wgpu_lora_fused", 8);
}

#[test]
fn wgpu_lora_grouped_path_matches_reference() {
    run_case("wgpu_lora_grouped", 96);
}

#[test]
fn gemma4_lora_api_reachable_without_cuda() {
    let install: fn(
        &'static WgpuContext,
        &Gemma4,
        &LoraAdapter,
        usize,
        usize,
    ) -> anyhow::Result<WgpuLoraRuntime> = WgpuLoraRuntime::install_gemma4_qkv;
    let synth: fn(&Gemma4, usize, f64, u64) -> anyhow::Result<LoraAdapter> = synth_qkv_adapter;
    let detach: fn(&Gemma4) = WgpuLoraRuntime::detach_gemma4;
    assert!([install as usize, synth as usize, detach as usize]
        .iter()
        .all(|&p| p != 0));
}
