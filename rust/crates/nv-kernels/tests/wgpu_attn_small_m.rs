#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::attention_fp8_decode as afd;
use nv_kernels::wgpu_backend::kernels::attn_decode;
use nv_kernels::wgpu_backend::kernels::attn_decode_small_m as smk;
use common::LcgShift32TwoSided as Lcg;
use common::time_calls;

fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let e = ((b >> 3) & 0xf) as i32;
    let m = (b & 7) as f32;
    if e == 0 {
        sign * (m / 8.0) * (-6.0f32).exp2()
    } else {
        sign * (1.0 + m / 8.0) * ((e - 7) as f32).exp2()
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn window_start(total: usize, window: usize) -> usize {
    if window > 0 && total > window {
        total - window
    } else {
        0
    }
}

struct Shape {
    label: &'static str,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
}

const SHAPES: [Shape; 2] = [
    Shape {
        label: "sliding_hd256_nkv16",
        n_heads: 32,
        n_kv_heads: 16,
        head_dim: 256,
        window: 512,
    },
    Shape {
        label: "full_hd512_nkv4",
        n_heads: 32,
        n_kv_heads: 4,
        head_dim: 512,
        window: 0,
    },
];

const PAST_LENGTHS: [usize; 3] = [0, 47, 1000];

fn sequential_oracle_f32(
    ctx: &WgpuContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    m_rows: usize,
    total: usize,
    window: usize,
    scaling: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; m_rows * n_heads * head_dim];
    for qi in 0..m_rows {
        let tq = total - (m_rows - 1 - qi);
        let sq = window_start(tq, window);
        let q_row = &q[qi * n_heads * head_dim..(qi + 1) * n_heads * head_dim];
        let k_slice = &k[..tq * n_kv_heads * head_dim];
        let v_slice = &v[..tq * n_kv_heads * head_dim];
        let mut row_out = vec![0f32; n_heads * head_dim];
        attn_decode::attn_decode_f32(
            ctx,
            q_row,
            k_slice,
            v_slice,
            &mut row_out,
            n_heads,
            n_kv_heads,
            head_dim,
            sq,
            tq,
            scaling,
        )
        .unwrap_or_else(|e| panic!("attn_decode_f32 oracle row qi={qi} tq={tq} sq={sq}: {e}"));
        out[qi * n_heads * head_dim..(qi + 1) * n_heads * head_dim].copy_from_slice(&row_out);
    }
    out
}

fn sequential_oracle_bf16(
    ctx: &WgpuContext,
    q: &[f32],
    k: &[u16],
    v: &[u16],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    m_rows: usize,
    total: usize,
    window: usize,
    scaling: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; m_rows * n_heads * head_dim];
    for qi in 0..m_rows {
        let tq = total - (m_rows - 1 - qi);
        let sq = window_start(tq, window);
        let q_row = &q[qi * n_heads * head_dim..(qi + 1) * n_heads * head_dim];
        let k_slice = &k[..tq * n_kv_heads * head_dim];
        let v_slice = &v[..tq * n_kv_heads * head_dim];
        let mut row_out = vec![0f32; n_heads * head_dim];
        smk::attn_decode_small_m_bf16kv(
            ctx,
            q_row,
            k_slice,
            v_slice,
            &mut row_out,
            n_heads,
            n_kv_heads,
            head_dim,
            1,
            tq,
            window,
            scaling,
        )
        .unwrap_or_else(|e| panic!("bf16kv m=1 oracle row qi={qi} tq={tq} sq={sq}: {e}"));
        out[qi * n_heads * head_dim..(qi + 1) * n_heads * head_dim].copy_from_slice(&row_out);
    }
    out
}

struct Fp8Case {
    q: Vec<u16>,
    k: Vec<u8>,
    v: Vec<u8>,
    ks: Vec<f32>,
    vs: Vec<f32>,
}

fn make_fp8_case(shape: &Shape, m_rows: usize, total: usize, seed: u64) -> Fp8Case {
    let mut rng = Lcg(seed);
    let q = rng.bf16_vec(m_rows * shape.n_heads * shape.head_dim, 1.5);
    let k = rng.fp8_vec(total * shape.n_kv_heads * shape.head_dim);
    let v = rng.fp8_vec(total * shape.n_kv_heads * shape.head_dim);
    let ks = rng.scale_vec(total * shape.n_kv_heads);
    let vs = rng.scale_vec(total * shape.n_kv_heads);
    Fp8Case { q, k, v, ks, vs }
}

fn sequential_oracle_fp8(
    ctx: &WgpuContext,
    case: &Fp8Case,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    m_rows: usize,
    total: usize,
    window: usize,
    scaling: f32,
) -> Vec<u16> {
    let mut out = vec![0u16; m_rows * n_heads * head_dim];
    for qi in 0..m_rows {
        let tq = total - (m_rows - 1 - qi);
        let q_row = &case.q[qi * n_heads * head_dim..(qi + 1) * n_heads * head_dim];
        let mut row_out = vec![0u16; n_heads * head_dim];
        afd::attention_fp8_decode(
            ctx,
            q_row,
            &case.k,
            &case.v,
            &case.ks,
            &case.vs,
            &mut row_out,
            &[tq as i32],
            n_heads,
            n_kv_heads,
            head_dim,
            window,
            scaling,
        )
        .unwrap_or_else(|e| panic!("attention_fp8_decode oracle row qi={qi} tq={tq}: {e}"));
        out[qi * n_heads * head_dim..(qi + 1) * n_heads * head_dim].copy_from_slice(&row_out);
    }
    out
}

fn check_shape_fp8(ctx: &WgpuContext, shape: &Shape, past: usize) -> usize {
    let scaling = 1.0f32 / (shape.head_dim as f32).sqrt();
    let mut cases = 0usize;
    for m in 1..=smk::MAX_M {
        let total = past + m;
        let case = make_fp8_case(
            shape,
            m,
            total,
            0xf8f8_5eed ^ ((past as u64) << 40) ^ ((m as u64) << 32) ^ (shape.head_dim as u64),
        );

        let oracle = sequential_oracle_fp8(
            ctx,
            &case,
            shape.n_heads,
            shape.n_kv_heads,
            shape.head_dim,
            m,
            total,
            shape.window,
            scaling,
        );

        let mut got = vec![0u16; m * shape.n_heads * shape.head_dim];
        smk::attn_decode_small_m_fp8(
            ctx,
            &case.q,
            &case.k,
            &case.v,
            &case.ks,
            &case.vs,
            &mut got,
            shape.n_heads,
            shape.n_kv_heads,
            shape.head_dim,
            m,
            total,
            shape.window,
            scaling,
        )
        .unwrap_or_else(|e| {
            panic!(
                "attn_decode_small_m_fp8 {} m={m} past={past}: {e}",
                shape.label
            )
        });

        let diff = oracle
            .iter()
            .zip(got.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diff,
            0,
            "{} m={m} past={past}: {diff}/{} bf16 words differ from {m}x attention_fp8_decode",
            shape.label,
            oracle.len()
        );
        cases += 1;
    }
    eprintln!(
        "fp8 parity: {} past={past} all M=1..={} bit-exact vs production attention_fp8_decode",
        shape.label,
        smk::MAX_M
    );
    cases
}

fn check_shape_f32(ctx: &WgpuContext, shape: &Shape, past: usize) {
    let scaling = 1.0f32 / (shape.head_dim as f32).sqrt();
    for m in 1..=smk::MAX_M {
        let total = past + m;
        let mut rng =
            Lcg(0x5eed_f32a ^ ((past as u64) << 40) ^ ((m as u64) << 32) ^ (shape.head_dim as u64));
        let q = rng.f32_vec(m * shape.n_heads * shape.head_dim, 1.0);
        let k = rng.f32_vec(total * shape.n_kv_heads * shape.head_dim, 1.0);
        let v = rng.f32_vec(total * shape.n_kv_heads * shape.head_dim, 2.0);

        let oracle = sequential_oracle_f32(
            ctx,
            &q,
            &k,
            &v,
            shape.n_heads,
            shape.n_kv_heads,
            shape.head_dim,
            m,
            total,
            shape.window,
            scaling,
        );

        let mut got = vec![0f32; m * shape.n_heads * shape.head_dim];
        smk::attn_decode_small_m_f32(
            ctx,
            &q,
            &k,
            &v,
            &mut got,
            shape.n_heads,
            shape.n_kv_heads,
            shape.head_dim,
            m,
            total,
            shape.window,
            scaling,
        )
        .unwrap_or_else(|e| {
            panic!(
                "attn_decode_small_m_f32 {} m={m} past={past}: {e}",
                shape.label
            )
        });

        let diff = oracle
            .iter()
            .zip(got.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            diff,
            0,
            "{} m={m} past={past}: {diff}/{} f32 words differ from {m}x attn_decode_f32",
            shape.label,
            oracle.len()
        );
    }
    eprintln!(
        "f32 parity: {} past={past} all M=1..={} bit-exact",
        shape.label,
        smk::MAX_M
    );
}

fn check_shape_bf16(ctx: &WgpuContext, shape: &Shape, past: usize) {
    let scaling = 1.0f32 / (shape.head_dim as f32).sqrt();
    for m in 1..=smk::MAX_M {
        let total = past + m;
        let mut rng =
            Lcg(0xb0b1_6ee5 ^ ((past as u64) << 40) ^ ((m as u64) << 32) ^ (shape.head_dim as u64));
        let q = rng.f32_vec(m * shape.n_heads * shape.head_dim, 1.0);
        let k = rng.bf16_vec(total * shape.n_kv_heads * shape.head_dim, 1.0);
        let v = rng.bf16_vec(total * shape.n_kv_heads * shape.head_dim, 2.0);

        let oracle = sequential_oracle_bf16(
            ctx,
            &q,
            &k,
            &v,
            shape.n_heads,
            shape.n_kv_heads,
            shape.head_dim,
            m,
            total,
            shape.window,
            scaling,
        );

        let mut got = vec![0f32; m * shape.n_heads * shape.head_dim];
        smk::attn_decode_small_m_bf16kv(
            ctx,
            &q,
            &k,
            &v,
            &mut got,
            shape.n_heads,
            shape.n_kv_heads,
            shape.head_dim,
            m,
            total,
            shape.window,
            scaling,
        )
        .unwrap_or_else(|e| {
            panic!(
                "attn_decode_small_m_bf16kv {} m={m} past={past}: {e}",
                shape.label
            )
        });

        let diff = oracle
            .iter()
            .zip(got.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            diff,
            0,
            "{} m={m} past={past}: {diff}/{} bf16kv words differ from {m}x its own M=1 mode",
            shape.label,
            oracle.len()
        );
    }
    eprintln!(
        "bf16kv self-parity: {} past={past} all M=1..={} bit-exact",
        shape.label,
        smk::MAX_M
    );
}

#[test]
fn f32_matches_sequential_attn_decode_f32_bitwise() {
    let Some(ctx) = ctx_or_skip("f32_matches_sequential_attn_decode_f32_bitwise") else {
        return;
    };
    for shape in &SHAPES {
        for &past in &PAST_LENGTHS {
            check_shape_f32(ctx, shape, past);
        }
    }
}

#[test]
fn bf16kv_matches_its_own_sequential_m1_calls_bitwise() {
    let Some(ctx) = ctx_or_skip("bf16kv_matches_its_own_sequential_m1_calls_bitwise") else {
        return;
    };
    for shape in &SHAPES {
        for &past in &PAST_LENGTHS {
            check_shape_bf16(ctx, shape, past);
        }
    }
}

#[test]
fn fp8_matches_sequential_attention_fp8_decode_bitwise() {
    let Some(ctx) = ctx_or_skip("fp8_matches_sequential_attention_fp8_decode_bitwise") else {
        return;
    };
    eprintln!(
        "adapter max_compute_workgroup_storage_size={}",
        ctx.caps.max_compute_workgroup_storage_size
    );
    let mut cases = 0usize;
    for shape in &SHAPES {
        for &past in &PAST_LENGTHS {
            cases += check_shape_fp8(ctx, shape, past);
        }
    }
    eprintln!("fp8 parity total: {cases}/54 cases bit-exact");
    assert_eq!(cases, 54);
}

#[test]
fn fp8_tracks_f32_small_m_within_quantization_tolerance() {
    let Some(ctx) = ctx_or_skip("fp8_tracks_f32_small_m_within_quantization_tolerance") else {
        return;
    };
    for shape in &SHAPES {
        for &past in &[0usize, 1000] {
            let scaling = 1.0f32 / (shape.head_dim as f32).sqrt();
            for m in [1usize, 4, 9] {
                let total = past + m;
                let case = make_fp8_case(
                    shape,
                    m,
                    total,
                    0xc0de_c0de ^ ((past as u64) << 40) ^ ((m as u64) << 32),
                );

                let q_f32: Vec<f32> = case.q.iter().map(|b| bf16_to_f32(*b)).collect();
                let hd = shape.head_dim;
                let k_f32: Vec<f32> = case
                    .k
                    .iter()
                    .enumerate()
                    .map(|(i, b)| e4m3_to_f32(*b) * case.ks[i / hd])
                    .collect();
                let v_f32: Vec<f32> = case
                    .v
                    .iter()
                    .enumerate()
                    .map(|(i, b)| e4m3_to_f32(*b) * case.vs[i / hd])
                    .collect();

                let mut ref_out = vec![0f32; m * shape.n_heads * hd];
                smk::attn_decode_small_m_f32(
                    ctx,
                    &q_f32,
                    &k_f32,
                    &v_f32,
                    &mut ref_out,
                    shape.n_heads,
                    shape.n_kv_heads,
                    hd,
                    m,
                    total,
                    shape.window,
                    scaling,
                )
                .unwrap();

                let mut got = vec![0u16; m * shape.n_heads * hd];
                smk::attn_decode_small_m_fp8(
                    ctx,
                    &case.q,
                    &case.k,
                    &case.v,
                    &case.ks,
                    &case.vs,
                    &mut got,
                    shape.n_heads,
                    shape.n_kv_heads,
                    hd,
                    m,
                    total,
                    shape.window,
                    scaling,
                )
                .unwrap();

                let mut worst = 0f32;
                for (idx, (g, r)) in got.iter().zip(ref_out.iter()).enumerate() {
                    let gf = bf16_to_f32(*g);
                    let err = (gf - r).abs();
                    let bound = 1e-2 + 1e-2 * r.abs();
                    assert!(
                        err <= bound,
                        "{} m={m} past={past} idx={idx}: fp8={gf} f32={r} err={err} bound={bound}",
                        shape.label
                    );
                    worst = worst.max(err / (r.abs() + 1e-6));
                }
                eprintln!(
                    "fp8-vs-f32 cross-check: {} m={m} past={past} worst_rel={worst:.2e}",
                    shape.label
                );
            }
        }
    }
}

#[test]
fn dispatch_helper_routes_fp8_and_matches_the_direct_entry() {
    let Some(ctx) = ctx_or_skip("dispatch_helper_routes_fp8_and_matches_the_direct_entry") else {
        return;
    };
    let shape = &SHAPES[0];
    let scaling = 1.0f32 / (shape.head_dim as f32).sqrt();
    let m = 5usize;
    let total = 52usize;
    let case = make_fp8_case(shape, m, total, 0xd15b_a7c4);

    let mut direct = vec![0u16; m * shape.n_heads * shape.head_dim];
    smk::attn_decode_small_m_fp8(
        ctx,
        &case.q,
        &case.k,
        &case.v,
        &case.ks,
        &case.vs,
        &mut direct,
        shape.n_heads,
        shape.n_kv_heads,
        shape.head_dim,
        m,
        total,
        shape.window,
        scaling,
    )
    .unwrap();

    let q_f32: Vec<f32> = case.q.iter().map(|b| bf16_to_f32(*b)).collect();
    let mut via_dispatch = vec![0f32; m * shape.n_heads * shape.head_dim];
    smk::attn_decode_small_m_dispatch(
        ctx,
        &q_f32,
        smk::SmallMKv::Fp8 {
            k: &case.k,
            v: &case.v,
            k_scales: &case.ks,
            v_scales: &case.vs,
        },
        &mut via_dispatch,
        shape.n_heads,
        shape.n_kv_heads,
        shape.head_dim,
        m,
        total,
        shape.window,
        scaling,
    )
    .unwrap();

    for (idx, (d, v)) in direct.iter().zip(via_dispatch.iter()).enumerate() {
        assert_eq!(
            bf16_to_f32(*d).to_bits(),
            v.to_bits(),
            "idx={idx}: dispatch fp8 route diverges from the direct entry"
        );
    }

    let mut out = vec![0f32; shape.n_heads * shape.head_dim];
    for bad_m in [0usize, 10] {
        let e = smk::attn_decode_small_m_dispatch(
            ctx,
            &q_f32,
            smk::SmallMKv::Fp8 {
                k: &case.k,
                v: &case.v,
                k_scales: &case.ks,
                v_scales: &case.vs,
            },
            &mut out,
            shape.n_heads,
            shape.n_kv_heads,
            shape.head_dim,
            bad_m,
            total,
            shape.window,
            scaling,
        )
        .unwrap_err();
        eprintln!("fp8 m={bad_m} rejection: {e}");
    }
}

#[test]
fn dispatch_helper_routes_by_precision_and_validates_m() {
    let Some(ctx) = ctx_or_skip("dispatch_helper_routes_by_precision_and_validates_m") else {
        return;
    };
    let shape = &SHAPES[1];
    let scaling = 1.0f32 / (shape.head_dim as f32).sqrt();
    let m = 3usize;
    let total = 10usize;
    let mut rng = Lcg(0xdead_beef);
    let q = rng.f32_vec(m * shape.n_heads * shape.head_dim, 1.0);
    let k = rng.f32_vec(total * shape.n_kv_heads * shape.head_dim, 1.0);
    let v = rng.f32_vec(total * shape.n_kv_heads * shape.head_dim, 2.0);
    let mut direct = vec![0f32; m * shape.n_heads * shape.head_dim];
    smk::attn_decode_small_m_f32(
        ctx,
        &q,
        &k,
        &v,
        &mut direct,
        shape.n_heads,
        shape.n_kv_heads,
        shape.head_dim,
        m,
        total,
        shape.window,
        scaling,
    )
    .unwrap();
    let mut via_dispatch = vec![0f32; m * shape.n_heads * shape.head_dim];
    smk::attn_decode_small_m_dispatch(
        ctx,
        &q,
        smk::SmallMKv::F32 { k: &k, v: &v },
        &mut via_dispatch,
        shape.n_heads,
        shape.n_kv_heads,
        shape.head_dim,
        m,
        total,
        shape.window,
        scaling,
    )
    .unwrap();
    assert_eq!(direct, via_dispatch);

    let mut out = vec![0f32; shape.n_heads * shape.head_dim];
    let e = smk::attn_decode_small_m_dispatch(
        ctx,
        &q,
        smk::SmallMKv::F32 { k: &k, v: &v },
        &mut out,
        shape.n_heads,
        shape.n_kv_heads,
        shape.head_dim,
        10,
        total,
        shape.window,
        scaling,
    )
    .unwrap_err();
    eprintln!("m=10 rejection: {e}");
    let e = smk::attn_decode_small_m_dispatch(
        ctx,
        &q,
        smk::SmallMKv::F32 { k: &k, v: &v },
        &mut out,
        shape.n_heads,
        shape.n_kv_heads,
        shape.head_dim,
        0,
        total,
        shape.window,
        scaling,
    )
    .unwrap_err();
    eprintln!("m=0 rejection: {e}");
}

fn gpu_idle_pct() -> Option<u32> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let util = text.lines().next()?.trim().parse::<u32>().ok()?;
    Some(100u32.saturating_sub(util))
}

fn idle_pct() -> Option<u32> {
    if let Some(p) = gpu_idle_pct() {
        return Some(p);
    }
    let out = std::process::Command::new("top")
        .args(["-l", "1", "-n", "8", "-o", "cpu"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if line.contains("CPU usage") {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let pct = fields.get(6)?.trim_end_matches('%');
            return pct.parse::<f32>().ok().map(|v| v as u32);
        }
    }
    None
}

fn wait_for_idle(min_pct: u32, timeout: std::time::Duration) -> bool {
    let start = std::time::Instant::now();
    loop {
        match idle_pct() {
            Some(p) if p >= min_pct => {
                eprintln!("idle gate: {p}% idle, proceeding");
                return true;
            }
            Some(p) => eprintln!("idle gate: {p}% idle, waiting"),
            None => eprintln!("idle gate: top unavailable, proceeding without gate"),
        }
        if idle_pct().is_none() || start.elapsed() >= timeout {
            return idle_pct().unwrap_or(100) >= min_pct;
        }
        std::thread::sleep(std::time::Duration::from_secs(20));
    }
}

#[test]
#[ignore = "GPU rate measurement; run explicitly with --ignored"]
fn bench_us_per_call_vs_sequential_m1_at_past1000() {
    let Some(ctx) = ctx_or_skip("bench_us_per_call_vs_sequential_m1_at_past1000") else {
        return;
    };

    let quiet = wait_for_idle(85, std::time::Duration::from_secs(15 * 60));
    if !quiet {
        eprintln!(
            "bench_us_per_call_vs_sequential_m1_at_past1000: PROVISIONAL -- could not reach a quiet window in 15 min"
        );
    }

    let past = 1000usize;
    let warmup = 3usize;
    let iters = 10usize;

    for shape in &SHAPES {
        let scaling = 1.0f32 / (shape.head_dim as f32).sqrt();
        println!(
            "attn_decode_small_m bench {} past={past} quiet_window={} (numbers are {} if quiet_window=false)",
            shape.label,
            quiet,
            if quiet { "measured" } else { "PROVISIONAL" }
        );
        println!(
            "{:>3} {:>16} {:>18} {:>10}",
            "M", "mk_us/call", "m_x_m1_us/call", "speedup"
        );

        for m in 1..=smk::MAX_M {
            let total = past + m;
            let mut rng = Lcg(0xf00d_ba5e ^ ((m as u64) << 32) ^ (shape.head_dim as u64));
            let q = rng.f32_vec(m * shape.n_heads * shape.head_dim, 1.0);
            let k = rng.f32_vec(total * shape.n_kv_heads * shape.head_dim, 1.0);
            let v = rng.f32_vec(total * shape.n_kv_heads * shape.head_dim, 2.0);
            let mut mk_out = vec![0f32; m * shape.n_heads * shape.head_dim];

            let mk_secs = time_calls(
                || {
                    smk::attn_decode_small_m_f32(
                        ctx,
                        &q,
                        &k,
                        &v,
                        &mut mk_out,
                        shape.n_heads,
                        shape.n_kv_heads,
                        shape.head_dim,
                        m,
                        total,
                        shape.window,
                        scaling,
                    )
                    .unwrap();
                },
                warmup,
                iters,
            );
            let mk_us_per_call = mk_secs * 1e6 / iters as f64;

            let m1_secs = time_calls(
                || {
                    for qi in 0..m {
                        let tq = total - (m - 1 - qi);
                        let sq = window_start(tq, shape.window);
                        let q_row = &q[qi * shape.n_heads * shape.head_dim
                            ..(qi + 1) * shape.n_heads * shape.head_dim];
                        let k_slice = &k[..tq * shape.n_kv_heads * shape.head_dim];
                        let v_slice = &v[..tq * shape.n_kv_heads * shape.head_dim];
                        let mut row_out = vec![0f32; shape.n_heads * shape.head_dim];
                        attn_decode::attn_decode_f32(
                            ctx,
                            q_row,
                            k_slice,
                            v_slice,
                            &mut row_out,
                            shape.n_heads,
                            shape.n_kv_heads,
                            shape.head_dim,
                            sq,
                            tq,
                            scaling,
                        )
                        .unwrap();
                    }
                },
                warmup,
                iters,
            );
            let m1_us_per_call = m1_secs * 1e6 / iters as f64;
            let speedup = m1_us_per_call / mk_us_per_call;

            println!(
                "{:>3} {:>16.2} {:>18.2} {:>9.2}x",
                m, mk_us_per_call, m1_us_per_call, speedup
            );
        }

        println!(
            "attn_decode_small_m_fp8 bench {} past={past} quiet_window={} (numbers are {} if quiet_window=false)",
            shape.label,
            quiet,
            if quiet { "measured" } else { "PROVISIONAL" }
        );
        println!(
            "{:>3} {:>16} {:>22} {:>10}",
            "M", "fp8_mk_us/call", "fp8_m_x_prod_us/call", "speedup"
        );

        for m in 1..=smk::MAX_M {
            let total = past + m;
            let case = make_fp8_case(
                shape,
                m,
                total,
                0xbe4c_f8f8 ^ ((m as u64) << 32) ^ (shape.head_dim as u64),
            );
            let mut mk_out = vec![0u16; m * shape.n_heads * shape.head_dim];

            let mk_secs = time_calls(
                || {
                    smk::attn_decode_small_m_fp8(
                        ctx,
                        &case.q,
                        &case.k,
                        &case.v,
                        &case.ks,
                        &case.vs,
                        &mut mk_out,
                        shape.n_heads,
                        shape.n_kv_heads,
                        shape.head_dim,
                        m,
                        total,
                        shape.window,
                        scaling,
                    )
                    .unwrap();
                },
                warmup,
                iters,
            );
            let mk_us_per_call = mk_secs * 1e6 / iters as f64;

            let m1_secs = time_calls(
                || {
                    for qi in 0..m {
                        let tq = total - (m - 1 - qi);
                        let q_row = &case.q[qi * shape.n_heads * shape.head_dim
                            ..(qi + 1) * shape.n_heads * shape.head_dim];
                        let mut row_out = vec![0u16; shape.n_heads * shape.head_dim];
                        afd::attention_fp8_decode(
                            ctx,
                            q_row,
                            &case.k,
                            &case.v,
                            &case.ks,
                            &case.vs,
                            &mut row_out,
                            &[tq as i32],
                            shape.n_heads,
                            shape.n_kv_heads,
                            shape.head_dim,
                            shape.window,
                            scaling,
                        )
                        .unwrap();
                    }
                },
                warmup,
                iters,
            );
            let m1_us_per_call = m1_secs * 1e6 / iters as f64;
            let speedup = m1_us_per_call / mk_us_per_call;

            println!(
                "{:>3} {:>16.2} {:>22.2} {:>9.2}x",
                m, mk_us_per_call, m1_us_per_call, speedup
            );
        }
    }
}
