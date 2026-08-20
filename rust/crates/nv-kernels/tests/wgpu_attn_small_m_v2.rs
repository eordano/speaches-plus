#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::attn_decode_small_m as smk;
use nv_kernels::wgpu_backend::kernels::attn_decode_small_m_v2 as smv2;
use nv_kernels::wgpu_backend::kernels::flash_decode as fdk;
use common::LcgShift32TwoSided as Lcg;
use common::bf16_enc as bf16_bits_from_f32;

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[derive(Clone, Copy)]
struct Shape {
    label: &'static str,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
    total: usize,
}

const BF16_SHAPES: [Shape; 3] = [
    Shape {
        label: "full_hd256_nkv4",
        n_heads: 8,
        n_kv_heads: 4,
        head_dim: 256,
        window: 0,
        total: 197,
    },
    Shape {
        label: "sliding_hd128_nkv2",
        n_heads: 8,
        n_kv_heads: 2,
        head_dim: 128,
        window: 96,
        total: 200,
    },
    Shape {
        label: "scalar_hd68_nkv4",
        n_heads: 8,
        n_kv_heads: 4,
        head_dim: 68,
        window: 0,
        total: 83,
    },
];

const FP8_SHAPES: [Shape; 3] = [
    Shape {
        label: "full_hd256_nkv4",
        n_heads: 8,
        n_kv_heads: 4,
        head_dim: 256,
        window: 0,
        total: 197,
    },
    Shape {
        label: "sliding_hd128_nkv2",
        n_heads: 8,
        n_kv_heads: 2,
        head_dim: 128,
        window: 96,
        total: 200,
    },
    Shape {
        label: "scalar_hd66_nkv2",
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 66,
        window: 0,
        total: 71,
    },
];

const M_ROWS: [usize; 3] = [1, 3, 8];
const SPLITS: usize = 16;

fn scaling_for(head_dim: usize) -> f32 {
    1.0 / (head_dim as f32).sqrt()
}

fn assert_rows_bit_identical(
    label: &str,
    m_rows: usize,
    row_elems: usize,
    got: &[u16],
    want_rows: &[Vec<u16>],
) {
    for qi in 0..m_rows {
        let got_row = &got[qi * row_elems..(qi + 1) * row_elems];
        let want_row = &want_rows[qi][..];
        let diff = got_row
            .iter()
            .zip(want_row.iter())
            .position(|(a, b)| a != b);
        assert!(
            diff.is_none(),
            "{label}: row qi={qi} diverges from per-row flash at elem {:?} (got {:#06x} want {:#06x})",
            diff,
            got_row[diff.unwrap()],
            want_row[diff.unwrap()],
        );
    }
}

#[test]
fn smv2_bf16kv_rows_are_bit_identical_to_decode_flash() {
    let Some(ctx) = ctx_or_skip("smv2_bf16kv_rows_are_bit_identical_to_decode_flash") else {
        return;
    };
    for shape in BF16_SHAPES {
        for &m in &M_ROWS {
            for fused in [false, true] {
                let mut rng = Lcg(0x5eed_0001 ^ (m as u64) << 8 ^ shape.head_dim as u64);
                let row_elems = shape.n_heads * shape.head_dim;
                let q = rng.f32_vec(m * row_elems, 1.0);
                let k = rng.bf16_vec(shape.total * shape.n_kv_heads * shape.head_dim, 1.0);
                let v = rng.bf16_vec(shape.total * shape.n_kv_heads * shape.head_dim, 1.0);
                let scaling = scaling_for(shape.head_dim);

                let mut got = vec![0u16; m * row_elems];
                let mut scratch = vec![
                    0f32;
                    smv2::scratch_elems(shape.n_heads, shape.head_dim, m, SPLITS)
                        .unwrap()
                ];
                smv2::attn_decode_small_m_v2_bf16kv(
                    ctx,
                    &q,
                    &k,
                    &v,
                    &mut got,
                    &mut scratch,
                    shape.n_heads,
                    shape.n_kv_heads,
                    shape.head_dim,
                    m,
                    shape.total,
                    shape.window,
                    scaling,
                    SPLITS,
                    0,
                    fused,
                )
                .unwrap_or_else(|e| panic!("{} m={m} fused={fused}: v2 {e}", shape.label));

                let mut want_rows: Vec<Vec<u16>> = Vec::with_capacity(m);
                for qi in 0..m {
                    let tq = shape.total - (m - 1 - qi);
                    let q_row = &q[qi * row_elems..(qi + 1) * row_elems];
                    let mut row_out = vec![0u16; row_elems];
                    let mut row_scratch =
                        vec![
                            0f32;
                            fdk::flash_splitk_scratch_elems(shape.n_heads, shape.head_dim, SPLITS)
                                .unwrap()
                        ];
                    if fused {
                        fdk::flash_decode_fused_bf16kv(
                            ctx,
                            q_row,
                            &k,
                            &v,
                            &mut row_out,
                            &mut row_scratch,
                            &[tq as i32],
                            0,
                            shape.n_heads,
                            shape.n_kv_heads,
                            shape.head_dim,
                            shape.window,
                            scaling,
                            SPLITS,
                            0,
                        )
                    } else {
                        fdk::flash_decode_splitk_bf16kv(
                            ctx,
                            q_row,
                            &k,
                            &v,
                            &mut row_out,
                            &mut row_scratch,
                            &[tq as i32],
                            shape.n_heads,
                            shape.n_kv_heads,
                            shape.head_dim,
                            shape.window,
                            scaling,
                            SPLITS,
                            0,
                        )
                    }
                    .unwrap_or_else(|e| panic!("{} qi={qi}: flash anchor {e}", shape.label));
                    want_rows.push(row_out);
                }
                assert_rows_bit_identical(
                    &format!("bf16 {} m={m} fused={fused}", shape.label),
                    m,
                    row_elems,
                    &got,
                    &want_rows,
                );
            }
        }
    }
}

#[test]
fn smv2_fp8kv_rows_are_bit_identical_to_decode_flash() {
    let Some(ctx) = ctx_or_skip("smv2_fp8kv_rows_are_bit_identical_to_decode_flash") else {
        return;
    };
    for shape in FP8_SHAPES {
        for &m in &M_ROWS {
            let mut rng = Lcg(0x5eed_0002 ^ (m as u64) << 8 ^ shape.head_dim as u64);
            let row_elems = shape.n_heads * shape.head_dim;
            let q = rng.bf16_vec(m * row_elems, 1.5);
            let k = rng.fp8_vec(shape.total * shape.n_kv_heads * shape.head_dim);
            let v = rng.fp8_vec(shape.total * shape.n_kv_heads * shape.head_dim);
            let ks = rng.scale_vec(shape.total * shape.n_kv_heads);
            let vs = rng.scale_vec(shape.total * shape.n_kv_heads);
            let scaling = scaling_for(shape.head_dim);

            let mut got = vec![0u16; m * row_elems];
            let mut scratch =
                vec![0f32; smv2::scratch_elems(shape.n_heads, shape.head_dim, m, SPLITS).unwrap()];
            smv2::attn_decode_small_m_v2_fp8kv(
                ctx,
                &q,
                &k,
                &v,
                &ks,
                &vs,
                &mut got,
                &mut scratch,
                shape.n_heads,
                shape.n_kv_heads,
                shape.head_dim,
                m,
                shape.total,
                shape.window,
                scaling,
                SPLITS,
                0,
            )
            .unwrap_or_else(|e| panic!("{} m={m}: v2 {e}", shape.label));

            let mut want_rows: Vec<Vec<u16>> = Vec::with_capacity(m);
            for qi in 0..m {
                let tq = shape.total - (m - 1 - qi);
                let q_row = &q[qi * row_elems..(qi + 1) * row_elems];
                let mut row_out = vec![0u16; row_elems];
                let mut row_scratch =
                    vec![
                        0f32;
                        fdk::flash_splitk_scratch_elems(shape.n_heads, shape.head_dim, SPLITS)
                            .unwrap()
                    ];
                fdk::flash_decode_fused_fp8kv(
                    ctx,
                    q_row,
                    &k,
                    &v,
                    &ks,
                    &vs,
                    &mut row_out,
                    &mut row_scratch,
                    &[tq as i32],
                    shape.n_heads,
                    shape.n_kv_heads,
                    shape.head_dim,
                    shape.window,
                    scaling,
                    SPLITS,
                    0,
                )
                .unwrap_or_else(|e| panic!("{} qi={qi}: flash anchor {e}", shape.label));
                want_rows.push(row_out);
            }
            assert_rows_bit_identical(
                &format!("fp8 {} m={m}", shape.label),
                m,
                row_elems,
                &got,
                &want_rows,
            );
        }
    }
}

#[test]
fn smv2_bf16kv_ring_rows_are_bit_identical_to_decode_flash() {
    let Some(ctx) = ctx_or_skip("smv2_bf16kv_ring_rows_are_bit_identical_to_decode_flash") else {
        return;
    };
    let (n_heads, n_kv_heads, head_dim) = (8usize, 4usize, 256usize);
    let (window, ring, total) = (64usize, 96usize, 300usize);
    let scaling = scaling_for(head_dim);
    for &m in &M_ROWS {
        let mut rng = Lcg(0x5eed_0003 ^ (m as u64) << 8);
        let row_elems = n_heads * head_dim;
        let q = rng.f32_vec(m * row_elems, 1.0);
        let k = rng.bf16_vec(ring * n_kv_heads * head_dim, 1.0);
        let v = rng.bf16_vec(ring * n_kv_heads * head_dim, 1.0);

        let mut got = vec![0u16; m * row_elems];
        let mut scratch = vec![0f32; smv2::scratch_elems(n_heads, head_dim, m, SPLITS).unwrap()];
        smv2::attn_decode_small_m_v2_bf16kv(
            ctx,
            &q,
            &k,
            &v,
            &mut got,
            &mut scratch,
            n_heads,
            n_kv_heads,
            head_dim,
            m,
            total,
            window,
            scaling,
            SPLITS,
            ring,
            false,
        )
        .unwrap_or_else(|e| panic!("ring m={m}: v2 {e}"));

        let mut want_rows: Vec<Vec<u16>> = Vec::with_capacity(m);
        for qi in 0..m {
            let tq = total - (m - 1 - qi);
            let q_row = &q[qi * row_elems..(qi + 1) * row_elems];
            let mut row_out = vec![0u16; row_elems];
            let mut row_scratch =
                vec![0f32; fdk::flash_splitk_scratch_elems(n_heads, head_dim, SPLITS).unwrap()];
            fdk::flash_decode_splitk_bf16kv(
                ctx,
                q_row,
                &k,
                &v,
                &mut row_out,
                &mut row_scratch,
                &[tq as i32],
                n_heads,
                n_kv_heads,
                head_dim,
                window,
                scaling,
                SPLITS,
                ring,
            )
            .unwrap_or_else(|e| panic!("ring qi={qi}: flash anchor {e}"));
            want_rows.push(row_out);
        }
        assert_rows_bit_identical(&format!("bf16 ring m={m}"), m, row_elems, &got, &want_rows);
    }
}

#[test]
fn smv2_f32_matches_bf16_entry_on_bf16_representable_kv() {
    let Some(ctx) = ctx_or_skip("smv2_f32_matches_bf16_entry_on_bf16_representable_kv") else {
        return;
    };
    for shape in [BF16_SHAPES[0], BF16_SHAPES[1]] {
        for &m in &M_ROWS {
            for fused in [false, true] {
                let mut rng = Lcg(0x5eed_0004 ^ (m as u64) << 8 ^ shape.head_dim as u64);
                let row_elems = shape.n_heads * shape.head_dim;
                let q = rng.f32_vec(m * row_elems, 1.0);
                let k_bits = rng.bf16_vec(shape.total * shape.n_kv_heads * shape.head_dim, 1.0);
                let v_bits = rng.bf16_vec(shape.total * shape.n_kv_heads * shape.head_dim, 1.0);
                let k_f32: Vec<f32> = k_bits.iter().map(|b| bf16_to_f32(*b)).collect();
                let v_f32: Vec<f32> = v_bits.iter().map(|b| bf16_to_f32(*b)).collect();
                let scaling = scaling_for(shape.head_dim);

                let mut out_f32 = vec![0f32; m * row_elems];
                let mut scratch_a =
                    vec![
                        0f32;
                        smv2::scratch_elems(shape.n_heads, shape.head_dim, m, SPLITS).unwrap()
                    ];
                smv2::attn_decode_small_m_v2_f32(
                    ctx,
                    &q,
                    &k_f32,
                    &v_f32,
                    &mut out_f32,
                    &mut scratch_a,
                    shape.n_heads,
                    shape.n_kv_heads,
                    shape.head_dim,
                    m,
                    shape.total,
                    shape.window,
                    scaling,
                    SPLITS,
                    fused,
                )
                .unwrap_or_else(|e| panic!("{} m={m} fused={fused}: v2 f32 {e}", shape.label));

                let mut out_bf16 = vec![0u16; m * row_elems];
                let mut scratch_b =
                    vec![
                        0f32;
                        smv2::scratch_elems(shape.n_heads, shape.head_dim, m, SPLITS).unwrap()
                    ];
                smv2::attn_decode_small_m_v2_bf16kv(
                    ctx,
                    &q,
                    &k_bits,
                    &v_bits,
                    &mut out_bf16,
                    &mut scratch_b,
                    shape.n_heads,
                    shape.n_kv_heads,
                    shape.head_dim,
                    m,
                    shape.total,
                    shape.window,
                    scaling,
                    SPLITS,
                    0,
                    fused,
                )
                .unwrap_or_else(|e| panic!("{} m={m} fused={fused}: v2 bf16 {e}", shape.label));

                for (i, (f, b)) in out_f32.iter().zip(out_bf16.iter()).enumerate() {
                    assert_eq!(
                        bf16_bits_from_f32(*f),
                        *b,
                        "{} m={m} fused={fused}: f32 entry diverges from bf16 entry at elem {i} (f32 {f})",
                        shape.label
                    );
                }
                for (i, (a, b)) in scratch_a.iter().zip(scratch_b.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{} m={m} fused={fused}: stage1 scratch diverges at elem {i}",
                        shape.label
                    );
                }
            }
        }
    }
}

#[test]
fn smv2_bf16kv_tracks_the_v1_oracle_numerically() {
    let Some(ctx) = ctx_or_skip("smv2_bf16kv_tracks_the_v1_oracle_numerically") else {
        return;
    };
    for shape in BF16_SHAPES {
        for &m in &M_ROWS {
            let mut rng = Lcg(0x5eed_0005 ^ (m as u64) << 8 ^ shape.head_dim as u64);
            let row_elems = shape.n_heads * shape.head_dim;
            let q = rng.f32_vec(m * row_elems, 1.0);
            let k = rng.bf16_vec(shape.total * shape.n_kv_heads * shape.head_dim, 1.0);
            let v = rng.bf16_vec(shape.total * shape.n_kv_heads * shape.head_dim, 1.0);
            let scaling = scaling_for(shape.head_dim);

            let mut got = vec![0u16; m * row_elems];
            let mut scratch =
                vec![0f32; smv2::scratch_elems(shape.n_heads, shape.head_dim, m, SPLITS).unwrap()];
            smv2::attn_decode_small_m_v2_bf16kv(
                ctx,
                &q,
                &k,
                &v,
                &mut got,
                &mut scratch,
                shape.n_heads,
                shape.n_kv_heads,
                shape.head_dim,
                m,
                shape.total,
                shape.window,
                scaling,
                SPLITS,
                0,
                false,
            )
            .unwrap_or_else(|e| panic!("{} m={m}: v2 {e}", shape.label));

            let mut v1_out = vec![0f32; m * row_elems];
            smk::attn_decode_small_m_bf16kv(
                ctx,
                &q,
                &k,
                &v,
                &mut v1_out,
                shape.n_heads,
                shape.n_kv_heads,
                shape.head_dim,
                m,
                shape.total,
                shape.window,
                scaling,
            )
            .unwrap_or_else(|e| panic!("{} m={m}: v1 oracle {e}", shape.label));

            let mut worst = 0f32;
            for (g, w) in got.iter().zip(v1_out.iter()) {
                let gf = bf16_to_f32(*g);
                let denom = w.abs().max(0.05);
                worst = worst.max((gf - w).abs() / denom);
            }
            assert!(
                worst < 3e-2,
                "{} m={m}: v2 drifts from v1 oracle (worst rel {worst})",
                shape.label
            );
        }
    }
}

const PROD_PASTS: [usize; 4] = [0, 47, 1000, 4000];
const PROD_SHAPES: [(&str, usize, usize, usize); 2] = [
    ("hd256_nkv4_full", 8, 4, 256),
    ("hd128_nkv2_full", 8, 2, 128),
];

#[test]
fn smv2_bf16kv_rows_stay_bit_identical_to_decode_flash_at_production_pasts() {
    let Some(ctx) = ctx_or_skip("smv2_bf16kv_rows_stay_bit_identical_at_production_pasts") else {
        return;
    };
    let mut failures = 0usize;
    let mut compared = 0usize;
    for (label, n_heads, n_kv_heads, head_dim) in PROD_SHAPES {
        for &past in &PROD_PASTS {
            for m in 1..=smv2::MAX_M {
                let total = past + m;
                for fused in [false, true] {
                    let mut rng =
                        Lcg(0xA11CE ^ (m as u64) << 8 ^ (head_dim as u64) << 20 ^ past as u64);
                    let row_elems = n_heads * head_dim;
                    let q = rng.f32_vec(m * row_elems, 1.0);
                    let k = rng.bf16_vec(total * n_kv_heads * head_dim, 1.0);
                    let v = rng.bf16_vec(total * n_kv_heads * head_dim, 1.0);
                    let scaling = scaling_for(head_dim);

                    let mut got = vec![0u16; m * row_elems];
                    let mut scratch =
                        vec![0f32; smv2::scratch_elems(n_heads, head_dim, m, SPLITS).unwrap()];
                    smv2::attn_decode_small_m_v2_bf16kv(
                        ctx,
                        &q,
                        &k,
                        &v,
                        &mut got,
                        &mut scratch,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        m,
                        total,
                        0,
                        scaling,
                        SPLITS,
                        0,
                        fused,
                    )
                    .unwrap_or_else(|e| panic!("{label} past={past} m={m} fused={fused}: v2 {e}"));

                    for qi in 0..m {
                        let tq = total - (m - 1 - qi);
                        let q_row = &q[qi * row_elems..(qi + 1) * row_elems];
                        let mut row_out = vec![0u16; row_elems];
                        let mut row_scratch =
                            vec![
                                0f32;
                                fdk::flash_splitk_scratch_elems(n_heads, head_dim, SPLITS).unwrap()
                            ];
                        if fused {
                            fdk::flash_decode_fused_bf16kv(
                                ctx,
                                q_row,
                                &k,
                                &v,
                                &mut row_out,
                                &mut row_scratch,
                                &[tq as i32],
                                0,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                0,
                                scaling,
                                SPLITS,
                                0,
                            )
                        } else {
                            fdk::flash_decode_splitk_bf16kv(
                                ctx,
                                q_row,
                                &k,
                                &v,
                                &mut row_out,
                                &mut row_scratch,
                                &[tq as i32],
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                0,
                                scaling,
                                SPLITS,
                                0,
                            )
                        }
                        .unwrap_or_else(|e| panic!("{label} anchor qi={qi}: {e}"));

                        assert!(
                            row_out.iter().any(|w| *w != 0),
                            "{label} past={past} m={m} qi={qi}: the per-row flash anchor wrote \
                             an all-zero row, so the comparison below is vacuous"
                        );
                        compared += 1;

                        let got_row = &got[qi * row_elems..(qi + 1) * row_elems];
                        if let Some(pos) =
                            got_row.iter().zip(row_out.iter()).position(|(a, b)| a != b)
                        {
                            failures += 1;
                            eprintln!(
                                "MISMATCH {label} past={past} m={m} fused={fused} qi={qi} \
                                 elem={pos} got={:#06x} want={:#06x}",
                                got_row[pos], row_out[pos]
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(compared > 0, "the sweep compared no rows at all");
    assert_eq!(
        failures, 0,
        "{failures} of {compared} row(s) diverged from per-row flash split-k"
    );
    eprintln!(
        "smv2 production pasts: {compared} rows bit-identical to per-row flash split-k \
         (pasts {PROD_PASTS:?})"
    );
}
