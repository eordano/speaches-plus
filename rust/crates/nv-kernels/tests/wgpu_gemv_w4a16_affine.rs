#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::kernels::gemv_w4a16_affine as aff;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_rows: u32,
    k_elems: u32,
    gs: u32,
    w_row_words: u32,
    scale_row_stride: u32,
    groups_x: u32,
    seg_nq: u32,
    seg_nk: u32,
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn bf16_small(&mut self) -> u16 {
        let v = ((self.next() % 2000) as f32 / 1000.0 - 1.0) * 0.05;
        (v.to_bits() >> 16) as u16
    }
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn ctx_or_skip(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP {test}: no wgpu adapter: {e}");
            None
        }
    }
}

#[test]
fn qkv_fused_entry_writes_each_segment_bitwise_equal_to_per_matrix_mrow4() {
    let Some(ctx) = ctx_or_skip("qkv_fused_entry_writes_each_segment_bitwise_equal_to_per_matrix_mrow4")
    else {
        return;
    };
    let (nq, nk, nv, k, gs) = (192usize, 32usize, 32usize, 512usize, 64usize);
    let n = nq + nk + nv;
    let mut r = Lcg(0x51ab7ee1);
    let words: Vec<u32> = (0..n * k / 8).map(|_| r.next()).collect();
    let scales_b: Vec<u32> = (0..n * (k / gs)).map(|_| r.bf16_small() as u32).collect();
    let biases_b: Vec<u32> = (0..n * (k / gs)).map(|_| r.bf16_small() as u32).collect();
    let x_pack: Vec<u32> = (0..k / 2).map(|_| r.next() & 0x3f803f80).collect();
    let pb = dispatch::storage_from_slice(ctx, "qkv-packed", &words);
    let sb = dispatch::storage_from_slice(ctx, "qkv-scale", &scales_b);
    let bb = dispatch::storage_from_slice(ctx, "qkv-bias", &biases_b);
    let xb = dispatch::storage_from_slice(ctx, "qkv-x", &x_pack);
    let yq = dispatch::storage_from_slice(ctx, "qkv-yq", &vec![0u32; nq / 2]);
    let yk = dispatch::storage_from_slice(ctx, "qkv-yk", &vec![0u32; nk / 2]);
    let yv = dispatch::storage_from_slice(ctx, "qkv-yv", &vec![0u32; nv / 2]);
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, aff::MROW4_ROWS_PER_GROUP);
    let params = Params {
        n_rows: n as u32,
        k_elems: k as u32,
        gs: gs as u32,
        w_row_words: (k / 8) as u32,
        scale_row_stride: (k / gs) as u32,
        groups_x: groups.0,
        seg_nq: nq as u32,
        seg_nk: nk as u32,
    };
    let ub = dispatch::uniform_from(ctx, "qkv-params", &params);
    dispatch::run(
        ctx,
        "qkv-fused",
        &aff::source(),
        aff::QKV_PK_ENTRY_SEGMENTS_MUST_BE_EVEN_ROWS,
        &[
            (0, &pb),
            (1, &sb),
            (2, &xb),
            (3, &yq),
            (4, &ub),
            (aff::BIAS_SLOT, &bb),
            (aff::QKV_Y_K_SLOT, &yk),
            (aff::QKV_Y_V_SLOT, &yv),
        ],
        groups,
    )
    .expect("fused dispatch");
    let got_q: Vec<u32> = dispatch::read_back(ctx, &yq, nq / 2).expect("rb q");
    let got_k: Vec<u32> = dispatch::read_back(ctx, &yk, nk / 2).expect("rb k");
    let got_v: Vec<u32> = dispatch::read_back(ctx, &yv, nv / 2).expect("rb v");

    let whole = dispatch::storage_from_slice(ctx, "qkv-whole", &vec![0u32; n]);
    dispatch::run(
        ctx,
        "qkv-ref",
        &aff::source(),
        aff::MROW4_ENTRY_REQUIRES_GS_MULTIPLE_OF_32,
        &[
            (0, &pb),
            (1, &sb),
            (2, &xb),
            (3, &whole),
            (4, &ub),
            (aff::BIAS_SLOT, &bb),
        ],
        groups,
    )
    .expect("ref dispatch");
    let flat: Vec<u32> = dispatch::read_back(ctx, &whole, n).expect("rb whole");
    let pair = |lo: u32, hi: u32| (lo & 0xffff) | ((hi & 0xffff) << 16);
    for i in 0..nq / 2 {
        assert_eq!(got_q[i], pair(flat[2 * i], flat[2 * i + 1]), "q pair {i}");
    }
    for i in 0..nk / 2 {
        assert_eq!(
            got_k[i],
            pair(flat[nq + 2 * i], flat[nq + 2 * i + 1]),
            "k pair {i}"
        );
    }
    for i in 0..nv / 2 {
        assert_eq!(
            got_v[i],
            pair(flat[nq + nk + 2 * i], flat[nq + nk + 2 * i + 1]),
            "v pair {i}"
        );
    }
}

#[test]
fn affine_block_matches_the_f64_host_reference_at_gs64_shapes() {
    let Some(ctx) = ctx_or_skip("affine_block_matches_the_f64_host_reference_at_gs64_shapes")
    else {
        return;
    };
    for (n, k) in [(64usize, 512usize), (256, 5120), (272, 4096)] {
        let gs = 64usize;
        let mut r = Lcg(0x9e3779b97f4a7c15 ^ ((n * k) as u64));
        let words: Vec<u32> = (0..n * k / 8).map(|_| r.next()).collect();
        let scales_b: Vec<u16> = (0..n * (k / gs)).map(|_| r.bf16_small()).collect();
        let biases_b: Vec<u16> = (0..n * (k / gs)).map(|_| r.bf16_small()).collect();
        let x_b: Vec<u16> = (0..k).map(|_| r.bf16_small()).collect();

        let scales_f: Vec<f32> = scales_b.iter().map(|b| bf16_to_f32(*b)).collect();
        let biases_f: Vec<f32> = biases_b.iter().map(|b| bf16_to_f32(*b)).collect();
        let x_f: Vec<f32> = x_b.iter().map(|b| bf16_to_f32(*b)).collect();
        let w = nv_quant::mlx_affine::dequantize_mlx_affine_linear(
            &words, &scales_f, &biases_f, n, k, gs,
        );
        let want: Vec<f64> = (0..n)
            .map(|row| {
                (0..k)
                    .map(|i| w[row * k + i] as f64 * x_f[i] as f64)
                    .sum::<f64>()
            })
            .collect();

        let widen = |v: &[u16]| -> Vec<u32> { v.iter().map(|b| *b as u32).collect() };
        let pack_x: Vec<u32> = (0..k / 2)
            .map(|i| x_b[2 * i] as u32 | ((x_b[2 * i + 1] as u32) << 16))
            .collect();
        for (entry, rows_per_group) in [
            (aff::BLOCK_ENTRY, aff::ROWS_PER_GROUP),
            (
                aff::MROW4_ENTRY_REQUIRES_GS_MULTIPLE_OF_32,
                aff::MROW4_ROWS_PER_GROUP,
            ),
            (aff::MROW8_PROBE_ENTRY, aff::MROW8_ROWS_PER_GROUP),
            (aff::MROW4H_PROBE_ENTRY, aff::MROW4H_ROWS_PER_GROUP),
        ] {
        let groups = dispatch::workgroup_count_1d(ctx, n as u64, rows_per_group);
        let params = Params {
            n_rows: n as u32,
            k_elems: k as u32,
            gs: gs as u32,
            w_row_words: (k / 8) as u32,
            scale_row_stride: (k / gs) as u32,
            groups_x: groups.0,
            seg_nq: 0,
            seg_nk: 0,
        };
        let pb = dispatch::storage_from_slice(ctx, "aff-packed", &words);
        let sb = dispatch::storage_from_slice(ctx, "aff-scale", &widen(&scales_b));
        let bb = dispatch::storage_from_slice(ctx, "aff-bias", &widen(&biases_b));
        let xb = dispatch::storage_from_slice(ctx, "aff-x", &pack_x);
        let yb = dispatch::storage_from_slice(ctx, "aff-y", &vec![0x7fc0u32; n]);
        let ub = dispatch::uniform_from(ctx, "aff-params", &params);
        dispatch::run(
            ctx,
            "aff-oracle",
            &aff::source(),
            entry,
            &[
                (0, &pb),
                (1, &sb),
                (2, &xb),
                (3, &yb),
                (4, &ub),
                (aff::BIAS_SLOT, &bb),
            ],
            groups,
        )
        .expect("dispatch");
        let got_words: Vec<u32> = dispatch::read_back(ctx, &yb, n).expect("readback");
        let ref_max = want.iter().fold(0f64, |a, b| a.max(b.abs())).max(1e-6);
        let mut worst = 0f64;
        for (row, gw) in got_words.iter().enumerate() {
            let got = bf16_to_f32((gw & 0xffff) as u16) as f64;
            let rel = (got - want[row]).abs() / ref_max;
            worst = worst.max(rel);
            assert!(
                rel < 3.0 / 256.0,
                "n{n} k{k} row {row}: kernel {got} vs f64 host {}, rel {rel:e} exceeds the \
                 bf16-output-plus-f32-accumulation budget",
                want[row]
            );
        }
        eprintln!("AFFINE-ORACLE {entry} n{n} k{k} worst rel {worst:e}");
        }
    }
}


#[test]
#[ignore = "GPU rate measurement; run via nvk.sh probe -- times the affine entries at the four dots.ocr decoder decode GEMVs (qkv and gate_up fused as served, hidden 1536, inter 8960, 12h/2kv head 128) and reports effective GB/s of packed-weight traffic"]
fn affine_rate_at_dots_ocr_decoder_shapes() {
    let Some(ctx) = ctx_or_skip("affine_rate_at_dots_ocr_decoder_shapes") else {
        return;
    };
    let shapes: Vec<(usize, usize, usize)> = vec![
        (2048, 1536, 64),
        (1536, 1536, 64),
        (17920, 1536, 64),
        (1536, 8960, 64),
    ];
    for (n, k, census_count) in shapes {
        rate_one_shape(ctx, n, k, census_count);
    }
}

#[test]
#[ignore = "GPU rate measurement; run via nvk.sh probe -- times the affine block entry against the symmetric gemv_w4a16_block at the qwen3.8 gate_proj shape and reports effective GB/s of packed-weight traffic"]
fn affine_vs_symmetric_rate_at_qwen38_gate_proj_shape() {
    let Some(ctx) = ctx_or_skip("affine_vs_symmetric_rate_at_qwen38_gate_proj_shape") else {
        return;
    };
    let shapes: Vec<(usize, usize, usize)> = vec![
        (17408, 5120, 64),
        (5120, 17408, 64),
        (10240, 5120, 48),
        (6144, 5120, 48),
        (5120, 6144, 64),
        (12288, 5120, 16),
        (1024, 5120, 32),
        (248320, 5120, 1),
    ];
    for (n, k, census_count) in shapes {
        rate_one_shape(ctx, n, k, census_count);
    }
}

fn rate_one_shape(ctx: &WgpuContext, n: usize, k: usize, census_count: usize) {
    let gs = 64usize;
    let _ = census_count;
    let mut r = Lcg(0xabcdef12345);
    let words: Vec<u32> = (0..n * k / 8).map(|_| r.next()).collect();
    let scales_b: Vec<u32> = (0..n * (k / gs)).map(|_| r.bf16_small() as u32).collect();
    let biases_b: Vec<u32> = (0..n * (k / gs)).map(|_| r.bf16_small() as u32).collect();
    let x_pack: Vec<u32> = (0..k / 2).map(|_| r.next() & 0x3f803f80).collect();
    let pb = dispatch::storage_from_slice(ctx, "rate-packed", &words);
    let sb = dispatch::storage_from_slice(ctx, "rate-scale", &scales_b);
    let bb = dispatch::storage_from_slice(ctx, "rate-bias", &biases_b);
    let xb = dispatch::storage_from_slice(ctx, "rate-x", &x_pack);
    let yb = dispatch::storage_from_slice(ctx, "rate-y", &vec![0u32; n]);
    let weight_bytes = (n * k / 2 + n * (k / gs) * 4 * 2) as f64;
    for (label, entry, rows_per_group) in [
        (
            "mrow4",
            aff::MROW4_ENTRY_REQUIRES_GS_MULTIPLE_OF_32,
            aff::MROW4_ROWS_PER_GROUP,
        ),
        ("mrow8", aff::MROW8_PROBE_ENTRY, aff::MROW8_ROWS_PER_GROUP),
        ("mrow4h", aff::MROW4H_PROBE_ENTRY, aff::MROW4H_ROWS_PER_GROUP),
    ] {
        let source = aff::source();
        let groups = dispatch::workgroup_count_1d(ctx, n as u64, rows_per_group);
        let params = Params {
            n_rows: n as u32,
            k_elems: k as u32,
            gs: gs as u32,
            w_row_words: (k / 8) as u32,
            scale_row_stride: (k / gs) as u32,
            groups_x: groups.0,
            seg_nq: 0,
            seg_nk: 0,
        };
        let ub = dispatch::uniform_from(ctx, "rate-params", &params);
        let binds = vec![
            (0u32, &pb),
            (1, &sb),
            (2, &xb),
            (3, &yb),
            (4, &ub),
            (aff::BIAS_SLOT, &bb),
        ];
        for _ in 0..5 {
            dispatch::run(ctx, "rate-warm", &source, entry, &binds, groups).expect("warm");
        }
        let _drain_warm: Vec<u32> = dispatch::read_back(ctx, &yb, 1).expect("drain-warm");
        let laps = 50usize;
        let t0 = std::time::Instant::now();
        for _ in 0..laps {
            dispatch::run(ctx, "rate-lap", &source, entry, &binds, groups).expect("lap");
        }
        let _drain: Vec<u32> = dispatch::read_back(ctx, &yb, 1).expect("drain");
        let per = t0.elapsed().as_secs_f64() / laps as f64;
        eprintln!(
            "AFFINE-RATE {label} n{n} k{k} census{census_count} {:.4} ms/dispatch {:.1} GB/s packed-weight",
            per * 1e3,
            weight_bytes / per / 1e9
        );
    }
}
