#![cfg(feature = "wgpu")]

mod common;
use common::lcg;
use common::require;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::gelu_tanh_mul as k;
use common::ctx_or_skip;

const K_SQRT2_OVER_PI: f32 = 0.797_884_6;
const K_CUBIC_COEFF: f32 = 0.044715;

fn gelu_tanh_mul_ref(gate: f32, up: f32) -> f32 {
    let g3 = gate * gate * gate;
    let inner = K_SQRT2_OVER_PI * (gate + K_CUBIC_COEFF * g3);
    let t = inner.tanh();
    let gelu = 0.5 * gate * (1.0 + t);
    gelu * up
}

fn ref_split(gate: &[u16], up: &[u16]) -> Vec<u16> {
    gate.iter()
        .zip(up.iter())
        .map(|(g, u)| {
            let g = bf16::from_bits(*g).to_f32();
            let u = bf16::from_bits(*u).to_f32();
            bf16::from_f32(gelu_tanh_mul_ref(g, u)).to_bits()
        })
        .collect()
}

fn ref_fused(fused: &[u16], inter: usize, tot_pairs: usize) -> Vec<u16> {
    (0..tot_pairs)
        .map(|idx| {
            let i = idx % inter;
            let bs = idx / inter;
            let off = bs * 2 * inter;
            let g = bf16::from_bits(fused[off + i]).to_f32();
            let u = bf16::from_bits(fused[off + inter + i]).to_f32();
            bf16::from_f32(gelu_tanh_mul_ref(g, u)).to_bits()
        })
        .collect()
}

fn rand_bf16(state: &mut u64, scale: f32) -> u16 {
    let r = (lcg(state) >> 11) as f64 / (1u64 << 53) as f64;
    bf16::from_f32(((r as f32) * 2.0 - 1.0) * scale).to_bits()
}

struct ErrStats {
    max_abs: f32,
    max_rel: f32,
    exact: usize,
    total: usize,
    worst: (f32, f32),
}

fn compare(got: &[u16], want: &[u16]) -> ErrStats {
    let mut s = ErrStats {
        max_abs: 0.0,
        max_rel: 0.0,
        exact: 0,
        total: got.len(),
        worst: (0.0, 0.0),
    };
    for (g, w) in got.iter().zip(want.iter()) {
        if g == w {
            s.exact += 1;
        }
        let gf = bf16::from_bits(*g).to_f32();
        let wf = bf16::from_bits(*w).to_f32();
        let d = (gf - wf).abs();
        if d > s.max_abs {
            s.max_abs = d;
            s.worst = (gf, wf);
        }
        let denom = wf.abs().max(1e-6);
        let rel = d / denom;
        if rel > s.max_rel {
            s.max_rel = rel;
        }
    }
    s
}

fn report(test: &str, s: &ErrStats) {
    eprintln!(
        "{test}: n={} bit_exact={} ({:.4}%) max_abs_err={:e} max_rel_err={:e} worst got={} want={}",
        s.total,
        s.exact,
        100.0 * s.exact as f64 / s.total.max(1) as f64,
        s.max_abs,
        s.max_rel,
        s.worst.0,
        s.worst.1
    );
}

#[test]
fn split_matches_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("split_matches_cpu_oracle") else {
        return;
    };
    let n = 4096 + 13;
    let mut st = 0x2026_0804_dead_beefu64;
    let gate: Vec<u16> = (0..n).map(|_| rand_bf16(&mut st, 6.0)).collect();
    let up: Vec<u16> = (0..n).map(|_| rand_bf16(&mut st, 3.0)).collect();
    let mut y = vec![0u16; n];
    k::gelu_tanh_mul_bf16(ctx, &gate, &up, &mut y, n).expect("split kernel");
    let want = ref_split(&gate, &up);
    let s = compare(&y, &want);
    report("split_matches_cpu_oracle", &s);
    assert!(s.max_rel <= 1.0 / 128.0, "max_rel_err {}", s.max_rel);
    assert!(
        s.exact * 100 >= s.total * 99,
        "only {}/{} bit-exact",
        s.exact,
        s.total
    );
}

#[test]
fn split_handles_saturating_and_edge_inputs() {
    let Some(ctx) = ctx_or_skip("split_handles_saturating_and_edge_inputs") else {
        return;
    };
    let raw: Vec<f32> = vec![
        0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 3.0, -3.0, 10.0, -10.0, 64.0, -64.0, 1024.0, -1024.0,
        1.0e5, -1.0e5, 1.0e-4, -1.0e-4, 6.0, -6.0,
    ];
    let gate: Vec<u16> = raw.iter().map(|v| bf16::from_f32(*v).to_bits()).collect();
    let up: Vec<u16> = raw
        .iter()
        .rev()
        .map(|v| bf16::from_f32(*v).to_bits())
        .collect();
    let n = gate.len();
    let mut y = vec![0u16; n];
    k::gelu_tanh_mul_bf16(ctx, &gate, &up, &mut y, n).expect("split kernel");
    let want = ref_split(&gate, &up);
    for i in 0..n {
        let g = bf16::from_bits(y[i]).to_f32();
        let w = bf16::from_bits(want[i]).to_f32();
        assert!(
            g.is_finite() == w.is_finite(),
            "finiteness mismatch at {i}: got {g} want {w}"
        );
        assert_eq!(
            y[i],
            want[i],
            "edge case {i}: gate={} up={} got={g} want={w}",
            raw[i],
            raw[n - 1 - i]
        );
    }
}

#[test]
fn fused_even_matches_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("fused_even_matches_cpu_oracle") else {
        return;
    };
    let inter = 512usize;
    let rows = 37usize;
    let tot_pairs = inter * rows;
    let mut st = 0x1234_5678_9abc_def0u64;
    let fused: Vec<u16> = (0..rows * 2 * inter)
        .map(|_| rand_bf16(&mut st, 5.0))
        .collect();
    let mut y = vec![0u16; tot_pairs];
    k::gelu_tanh_mul_fused_bf16(ctx, &fused, &mut y, inter, tot_pairs).expect("fused kernel");
    let want = ref_fused(&fused, inter, tot_pairs);
    let s = compare(&y, &want);
    report("fused_even_matches_cpu_oracle", &s);
    assert!(s.max_rel <= 1.0 / 128.0, "max_rel_err {}", s.max_rel);
    assert!(
        s.exact * 100 >= s.total * 99,
        "only {}/{} bit-exact",
        s.exact,
        s.total
    );
}

#[test]
fn fused_odd_inter_uses_general_path() {
    let Some(ctx) = ctx_or_skip("fused_odd_inter_uses_general_path") else {
        return;
    };
    let inter = 71usize;
    let rows = 9usize;
    let tot_pairs = inter * rows;
    let mut st = 0xfeed_face_0000_0001u64;
    let fused: Vec<u16> = (0..rows * 2 * inter)
        .map(|_| rand_bf16(&mut st, 4.0))
        .collect();
    let mut y = vec![0u16; tot_pairs];
    k::gelu_tanh_mul_fused_bf16(ctx, &fused, &mut y, inter, tot_pairs).expect("fused kernel");
    let want = ref_fused(&fused, inter, tot_pairs);
    let s = compare(&y, &want);
    report("fused_odd_inter_uses_general_path", &s);
    assert!(s.max_rel <= 1.0 / 128.0, "max_rel_err {}", s.max_rel);
    assert!(
        s.exact * 100 >= s.total * 99,
        "only {}/{} bit-exact",
        s.exact,
        s.total
    );
}

#[test]
fn fused_agrees_with_split_on_the_same_data() {
    let Some(ctx) = ctx_or_skip("fused_agrees_with_split_on_the_same_data") else {
        return;
    };
    let inter = 128usize;
    let rows = 5usize;
    let tot_pairs = inter * rows;
    let mut st = 0x0bad_c0de_1111_2222u64;
    let fused: Vec<u16> = (0..rows * 2 * inter)
        .map(|_| rand_bf16(&mut st, 8.0))
        .collect();
    let mut gate = Vec::with_capacity(tot_pairs);
    let mut up = Vec::with_capacity(tot_pairs);
    for bs in 0..rows {
        let off = bs * 2 * inter;
        gate.extend_from_slice(&fused[off..off + inter]);
        up.extend_from_slice(&fused[off + inter..off + 2 * inter]);
    }
    let mut y_fused = vec![0u16; tot_pairs];
    let mut y_split = vec![0u16; tot_pairs];
    k::gelu_tanh_mul_fused_bf16(ctx, &fused, &mut y_fused, inter, tot_pairs).expect("fused");
    k::gelu_tanh_mul_bf16(ctx, &gate, &up, &mut y_split, tot_pairs).expect("split");
    assert_eq!(y_fused, y_split);
}

#[test]
fn empty_and_shape_errors() {
    let Some(ctx) = ctx_or_skip("empty_and_shape_errors") else {
        return;
    };
    let mut none: Vec<u16> = Vec::new();
    assert!(k::gelu_tanh_mul_bf16(ctx, &[], &[], &mut none, 0).is_ok());
    assert!(k::gelu_tanh_mul_fused_bf16(ctx, &[], &mut none, 8, 0).is_ok());
    let mut y = vec![0u16; 4];
    assert!(k::gelu_tanh_mul_bf16(ctx, &[0u16; 3], &[0u16; 4], &mut y, 4).is_err());
    assert!(k::gelu_tanh_mul_fused_bf16(ctx, &[0u16; 7], &mut y, 4, 4).is_err());
    assert!(k::gelu_tanh_mul_fused_bf16(ctx, &[0u16; 8], &mut y, 0, 4).is_err());
}

#[test]
fn large_dispatch_folds_correctly() {
    let Some(ctx) = ctx_or_skip("large_dispatch_folds_correctly") else {
        return;
    };
    let inter = 2usize;
    let rows = 200_000usize;
    let tot_pairs = inter * rows;
    let mut st = 0x7777_8888_9999_aaaau64;
    let fused: Vec<u16> = (0..rows * 2 * inter)
        .map(|_| rand_bf16(&mut st, 2.0))
        .collect();
    let mut y = vec![0u16; tot_pairs];
    k::gelu_tanh_mul_fused_bf16(ctx, &fused, &mut y, inter, tot_pairs).expect("fused kernel");
    let want = ref_fused(&fused, inter, tot_pairs);
    let s = compare(&y, &want);
    report("large_dispatch_folds_correctly", &s);
    assert!(s.max_rel <= 1.0 / 128.0, "max_rel_err {}", s.max_rel);
    assert!(
        s.exact * 100 >= s.total * 99,
        "only {}/{} bit-exact",
        s.exact,
        s.total
    );
}
