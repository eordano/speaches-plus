#![cfg(feature = "wgpu")]

use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_weights::WeightLoader;

const DIR_ENV: &str = "NV_MLXQ_DIR";

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn bf16_small(&mut self) -> f32 {
        let v = ((self.next() % 2000) as f32 / 1000.0 - 1.0) * 0.05;
        half::bf16::from_f32(v).to_f32()
    }
}

fn mlx_dir() -> Option<std::path::PathBuf> {
    match std::env::var(DIR_ENV) {
        Ok(d) if !d.is_empty() => {
            let p = std::path::PathBuf::from(&d);
            assert!(p.is_dir(), "{DIR_ENV}={d} is not a directory");
            Some(p)
        }
        _ => {
            eprintln!(
                "SKIP q3d_mlx_affine_graph_oracle: {DIR_ENV} unset; point it at an \
                 mlx-community 4-bit affine snapshot (e.g. Qwen3.8-27B-4bit) to run"
            );
            None
        }
    }
}

#[test]
fn one_real_affine_linear_through_push_gemv_matches_ref_gemv_dense() {
    let Some(dir) = mlx_dir() else {
        return;
    };
    let w = WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("open weights");
    let module = [
        "language_model.model.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.gate_proj",
    ]
    .into_iter()
    .find(|m| w.has(&format!("{m}.scales")))
    .expect("no layer-0 gate_proj with .scales siblings; this is not an mlx affine checkpoint");

    let packed_shape = w
        .shape_of(&format!("{module}.weight"))
        .expect("packed weight shape");
    assert_eq!(packed_shape.len(), 2, "packed weight must be 2-D");
    let (n, k) = (packed_shape[0], packed_shape[1] * 8);
    let lin = q3d::load_affine_w4_lin(&w, module, n, k)
        .expect("load_affine_w4_lin on the real gate_proj");
    drop(w);
    eprintln!("[q3d-mlx-affine-oracle] {module}: n={n} k={k} gs={}", lin.gs);

    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for s in &lin.scales_bf16 {
        let v = f32::from_bits((*s as u32) << 16).abs();
        lo = lo.min(v);
        hi = hi.max(v);
    }
    assert!(
        hi > lo * 1.5,
        "real checkpoint scales span only {lo}..{hi}; with near-uniform scales a mis-indexed \
         scale load is unobservable and this oracle gates nothing"
    );

    let mut r = Lcg(0x9e3779b97f4a7c15);
    let x: Vec<f32> = (0..k).map(|_| r.bf16_small()).collect();
    let host = q3d::HostDenseLin::AffineW4(lin);
    let want = q3d::ref_gemv_dense_for_test(&host, &x);
    let ref_max = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    let nonzero = want.iter().filter(|v| **v != 0.0).count();
    assert!(
        ref_max > 1e-3 && nonzero * 2 >= n,
        "host reference is degenerate (max |out| {ref_max:e}, {nonzero}/{n} rows nonzero); \
         comparing against it would pass on a kernel that writes constants"
    );

    let got = q3d::debug_push_gemv_dense_reading_packed_pairs_for_test(&host, &x)
        .expect("push_gemv through the graph builder");
    assert_eq!(got.len(), n, "packed-pairs readback must yield one value per output row");

    let bound = ref_max * 3.0 / 256.0;
    let mut worst = 0f32;
    let mut over: Vec<String> = Vec::new();
    for (row, (g, v)) in got.iter().zip(want.iter()).enumerate() {
        let err = (g - v).abs();
        worst = worst.max(err / bound.max(f32::MIN_POSITIVE));
        if err > bound {
            over.push(format!("row {row}: got {g} want {v} err {err:e}"));
        }
    }
    eprintln!(
        "[q3d-mlx-affine-oracle] worst err/bound {worst:.3} over {} rows (ref_max {ref_max:.4})",
        got.len()
    );
    assert!(
        over.is_empty(),
        "the graph's affine gemv (gemv_w4a16_affine + q3w_affine_y_pack_pairs) diverged from \
         ref_gemv_dense past 3 bf16 ulp of the row-max: the output is bf16 and both sides \
         accumulate f32, so that bound is the format's own resolution plus reduction-order \
         slack, not a tuned tolerance:\n{}",
        over.join("\n")
    );
}
