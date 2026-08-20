
use candle_core::{DType, Device, Tensor};

fn dev() -> Device {
    Device::cuda_if_available(0).unwrap_or(Device::Cpu)
}

fn with_contiguous(a: &Tensor, b: &Tensor, x: &Tensor, scaling: f64) -> Tensor {
    let xr = x.matmul(&a.t().unwrap().contiguous().unwrap()).unwrap();
    let d = xr.matmul(&b.t().unwrap().contiguous().unwrap()).unwrap();
    (d * scaling).unwrap()
}

fn without_contiguous(a: &Tensor, b: &Tensor, x: &Tensor, scaling: f64) -> Tensor {
    let xr = x.matmul(&a.t().unwrap()).unwrap();
    let d = xr.matmul(&b.t().unwrap()).unwrap();
    (d * scaling).unwrap()
}

fn drain(t: &Tensor) {
    let _ = t.sum_all().unwrap().to_scalar::<f32>().unwrap();
}

fn time(f: impl Fn() -> Tensor, iters: usize, _d: &Device) -> f64 {
    drain(&f());
    let start = std::time::Instant::now();
    let mut last = f();
    for _ in 1..iters {
        last = f();
    }
    drain(&last);
    start.elapsed().as_secs_f64() * 1e3 / iters as f64
}

#[test]
#[ignore = "a measurement, not a gate; set NV_LORA_COST=1"]
fn the_contiguous_copy_in_lora_delta_costs_this_much() {
    if std::env::var("NV_LORA_COST").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_LORA_COST=1");
    }
    let d = dev();
    eprintln!("[lora-cost] device: {d:?}");
    println!("  tokens  in_f  out_f   r | contiguous ms | strided ms | speedup | max_abs_diff");
    for (n, inf, outf, r) in [
        (512usize, 4096usize, 4096usize, 16usize),
        (512, 4096, 4096, 64),
        (2048, 4096, 4096, 16),
        (512, 5376, 8192, 16),
        (1, 4096, 4096, 16),
    ] {
        let a = Tensor::randn(0f32, 1.0, (r, inf), &d).unwrap();
        let b = Tensor::randn(0f32, 1.0, (outf, r), &d).unwrap();
        let x = Tensor::randn(0f32, 1.0, (n, inf), &d).unwrap();

        let want = with_contiguous(&a, &b, &x, 2.0);
        let got = without_contiguous(&a, &b, &x, 2.0);
        let diff = (want - got)
            .unwrap()
            .abs()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let iters = if n > 1000 { 50 } else { 200 };
        let c_ms = time(|| with_contiguous(&a, &b, &x, 2.0), iters, &d);
        let s_ms = time(|| without_contiguous(&a, &b, &x, 2.0), iters, &d);
        println!(
            "  {n:6} {inf:5} {outf:6} {r:3} | {c_ms:13.4} | {s_ms:10.4} | {:6.2}x | {diff:.3e}",
            c_ms / s_ms
        );
    }
}
