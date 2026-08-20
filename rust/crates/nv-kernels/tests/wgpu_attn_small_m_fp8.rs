#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::attention_fp8_decode as afd;
use nv_kernels::wgpu_backend::kernels::attn_decode_small_m_fp8 as smk8;
use nv_kernels::wgpu_backend::kernels::flash_decode;
use nv_kernels::wgpu_backend::WgpuError;
use common::idle_pct;
use common::time_calls;
use common::wait_for_idle;

struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn bf16_vec(&mut self, len: usize, scale: f32) -> Vec<u16> {
        (0..len)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_bits())
            .collect()
    }
    fn fp8_vec(&mut self, len: usize) -> Vec<u8> {
        (0..len)
            .map(|_| {
                let mut b = (self.next_u32() & 0xff) as u8;
                if b & 127 == 127 {
                    b &= 0x7e;
                }
                b
            })
            .collect()
    }
    fn scale_vec(&mut self, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| 0.02 + (self.next_f32() * 0.5 + 0.5) * 0.08)
            .collect()
    }
}

struct Shape {
    label: &'static str,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    window: usize,
}

const SHAPES: [Shape; 2] = [
    Shape {
        label: "sliding_hd256_nkv16",
        n_q: 32,
        n_kv: 16,
        head_dim: 256,
        window: 512,
    },
    Shape {
        label: "full_hd512_nkv4",
        n_q: 32,
        n_kv: 4,
        head_dim: 512,
        window: 0,
    },
];

const PAST_LENGTHS: [usize; 3] = [0, 47, 1000];

fn sequential_oracle_fp8(
    ctx: &WgpuContext,
    q: &[u16],
    k_fp8: &[u8],
    v_fp8: &[u8],
    k_scales: &[f32],
    v_scales: &[f32],
    shape: &Shape,
    m_rows: usize,
    total: usize,
    scaling: f32,
) -> Vec<u16> {
    let per_slot = shape.n_kv * shape.head_dim;
    let row_elems = shape.n_q * shape.head_dim;
    let mut out = vec![0u16; m_rows * row_elems];
    for qi in 0..m_rows {
        let tq = total - (m_rows - 1 - qi);
        let q_row = &q[qi * row_elems..(qi + 1) * row_elems];
        let mut row_out = vec![0u16; row_elems];
        afd::attention_fp8_decode(
            ctx,
            q_row,
            &k_fp8[..tq * per_slot],
            &v_fp8[..tq * per_slot],
            &k_scales[..tq * shape.n_kv],
            &v_scales[..tq * shape.n_kv],
            &mut row_out,
            &[tq as i32],
            shape.n_q,
            shape.n_kv,
            shape.head_dim,
            shape.window,
            scaling,
        )
        .unwrap_or_else(|e| panic!("attention_fp8_decode oracle row qi={qi} tq={tq}: {e}"));
        out[qi * row_elems..(qi + 1) * row_elems].copy_from_slice(&row_out);
    }
    out
}

fn check_shape(ctx: &WgpuContext, shape: &Shape, past: usize) {
    let scaling = 1.0f32 / (shape.head_dim as f32).sqrt();
    for m in 1..=smk8::MAX_M {
        let total = past + m;
        let per_slot = shape.n_kv * shape.head_dim;
        let mut rng =
            Lcg(0x8f00_ba11 ^ ((m as u64) << 40) ^ ((past as u64) << 8) ^ shape.head_dim as u64);
        let q = rng.bf16_vec(m * shape.n_q * shape.head_dim, 1.0);
        let k = rng.fp8_vec(total * per_slot);
        let v = rng.fp8_vec(total * per_slot);
        let ks = rng.scale_vec(total * shape.n_kv);
        let vs = rng.scale_vec(total * shape.n_kv);

        let mut got = vec![0u16; m * shape.n_q * shape.head_dim];
        smk8::attn_decode_small_m_fp8(
            ctx,
            &q,
            &k,
            &v,
            &ks,
            &vs,
            &mut got,
            shape.n_q,
            shape.n_kv,
            shape.head_dim,
            m,
            total,
            shape.window,
            scaling,
        )
        .unwrap_or_else(|e| panic!("{} m={m} past={past}: {e}", shape.label));

        let want = sequential_oracle_fp8(ctx, &q, &k, &v, &ks, &vs, shape, m, total, scaling);
        let diffs = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(
            diffs,
            0,
            "{} m={m} past={past}: {diffs}/{} output words differ from attention_fp8_decode x{m}",
            shape.label,
            got.len()
        );
    }
}

#[test]
fn fp8kv_matches_attention_fp8_decode_called_m_times_bitwise() {
    let Some(ctx) = ctx_or_skip("fp8kv_matches_attention_fp8_decode_called_m_times_bitwise") else {
        return;
    };
    for shape in &SHAPES {
        for past in PAST_LENGTHS {
            check_shape(ctx, shape, past);
        }
    }
}

#[test]
fn rejects_bad_geometry_and_m() {
    let Some(ctx) = ctx_or_skip("rejects_bad_geometry_and_m") else {
        return;
    };
    let hd = 64usize;
    let n_q = 4usize;
    let n_kv = 2usize;
    let m = 2usize;
    let total = 4usize;
    let mut rng = Lcg(0xdead);
    let q = rng.bf16_vec(m * n_q * hd, 1.0);
    let k = rng.fp8_vec(total * n_kv * hd);
    let v = rng.fp8_vec(total * n_kv * hd);
    let ks = rng.scale_vec(total * n_kv);
    let vs = rng.scale_vec(total * n_kv);
    let mut out = vec![0u16; m * n_q * hd];

    let call = |m_rows: usize,
                n_q_: usize,
                n_kv_: usize,
                hd_: usize,
                total_: usize,
                out_: &mut Vec<u16>| {
        smk8::attn_decode_small_m_fp8(
            ctx, &q, &k, &v, &ks, &vs, out_, n_q_, n_kv_, hd_, m_rows, total_, 0, 0.125,
        )
    };

    assert!(matches!(
        call(0, n_q, n_kv, hd, total, &mut out).unwrap_err(),
        WgpuError::Shape(_)
    ));
    assert!(matches!(
        call(smk8::MAX_M + 1, n_q, n_kv, hd, total, &mut out).unwrap_err(),
        WgpuError::Shape(_)
    ));
    assert!(matches!(
        call(m, 6, 4, hd, total, &mut out).unwrap_err(),
        WgpuError::Shape(_)
    ));
    assert!(matches!(
        call(m, n_q, n_kv, 96, total, &mut out).unwrap_err(),
        WgpuError::Unsupported(_)
    ));
    assert!(matches!(
        call(m, n_q, n_kv, hd, 1, &mut out).unwrap_err(),
        WgpuError::Shape(_)
    ));

    call(m, n_q, n_kv, hd, total, &mut out).unwrap();
}

#[test]
#[ignore = "GPU rate measurement; run explicitly with --ignored"]
fn bench_fp8_small_m_vs_afd_and_flash_m1_baselines() {
    let Some(ctx) = ctx_or_skip("bench_fp8_small_m_vs_afd_and_flash_m1_baselines") else {
        return;
    };

    let quiet = wait_for_idle(85, std::time::Duration::from_secs(15 * 60));
    if !quiet {
        eprintln!("bench_fp8_small_m_vs_afd_and_flash_m1_baselines: PROVISIONAL -- no quiet window in 15 min");
    }

    let warmup = 3usize;
    let iters = 10usize;

    for past in [200usize, 1000] {
        for shape in &SHAPES {
            let scaling = 1.0f32 / (shape.head_dim as f32).sqrt();
            println!(
                "attn_decode_small_m_fp8 bench {} past={past} quiet_window={quiet} ({})",
                shape.label,
                if quiet { "measured" } else { "PROVISIONAL" }
            );
            println!(
                "{:>3} {:>14} {:>16} {:>18} {:>10} {:>12}",
                "M", "smk8_us/call", "afd_x_m_us/call", "flash_x_m_us/call", "vs_afd", "vs_flash"
            );

            for m in [1usize, 4, 8, 10] {
                let total = past + m;
                let per_slot = shape.n_kv * shape.head_dim;
                let row_elems = shape.n_q * shape.head_dim;
                let mut rng = Lcg(0xf00d ^ ((m as u64) << 32) ^ shape.head_dim as u64);
                let q = rng.bf16_vec(m * row_elems, 1.0);
                let k = rng.fp8_vec(total * per_slot);
                let v = rng.fp8_vec(total * per_slot);
                let ks = rng.scale_vec(total * shape.n_kv);
                let vs = rng.scale_vec(total * shape.n_kv);
                let mut out = vec![0u16; m * row_elems];

                let smk_secs = time_calls(
                    || {
                        smk8::attn_decode_small_m_fp8(
                            ctx,
                            &q,
                            &k,
                            &v,
                            &ks,
                            &vs,
                            &mut out,
                            shape.n_q,
                            shape.n_kv,
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
                let smk_us = smk_secs * 1e6 / iters as f64;

                let afd_secs = time_calls(
                    || {
                        let _ = sequential_oracle_fp8(
                            ctx, &q, &k, &v, &ks, &vs, shape, m, total, scaling,
                        );
                    },
                    warmup,
                    iters,
                );
                let afd_us = afd_secs * 1e6 / iters as f64;

                let splits = flash_decode::DEFAULT_SPLITS;
                let scratch_elems =
                    flash_decode::flash_splitk_scratch_elems(shape.n_q, shape.head_dim, splits)
                        .unwrap();
                let mut scratch = vec![0f32; scratch_elems];
                let mut row_out = vec![0u16; row_elems];
                let flash_secs = time_calls(
                    || {
                        for qi in 0..m {
                            let tq = total - (m - 1 - qi);
                            let q_row = &q[qi * row_elems..(qi + 1) * row_elems];
                            flash_decode::flash_decode_fused_fp8kv(
                                ctx,
                                q_row,
                                &k[..tq * per_slot],
                                &v[..tq * per_slot],
                                &ks[..tq * shape.n_kv],
                                &vs[..tq * shape.n_kv],
                                &mut row_out,
                                &mut scratch,
                                &[tq as i32],
                                shape.n_q,
                                shape.n_kv,
                                shape.head_dim,
                                shape.window,
                                scaling,
                                splits,
                                0,
                            )
                            .unwrap();
                        }
                    },
                    warmup,
                    iters,
                );
                let flash_us = flash_secs * 1e6 / iters as f64;

                println!(
                    "{:>3} {:>14.2} {:>16.2} {:>18.2} {:>9.2}x {:>11.2}x",
                    m,
                    smk_us,
                    afd_us,
                    flash_us,
                    afd_us / smk_us,
                    flash_us / smk_us
                );
            }
        }
    }
}
