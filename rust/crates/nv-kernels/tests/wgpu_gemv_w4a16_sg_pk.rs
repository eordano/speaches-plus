#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use common::d;
use common::dot8;
use common::LcgShift33W4a16Packs as Lcg;
use common::OffParams;
use common::pack_u16;
use common::Params;
use common::q;
use common::require;
use common::tree_sum;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::gemv_w4a16 as gw;

fn sg_ctx_or_skip(test: &str) -> Option<&'static WgpuContext> {
    let ctx = ctx_or_skip(test)?;
    let width = ctx.subgroup_width();
    eprintln!("{test}: probed subgroup width {width:?}");
    if !gw::sg_pk_supported(width) {
        if require() {
            panic!(
                "{test}: probed subgroup width {width:?} is not a multiple of the {} lanes the \
                 sg16 pk kernel packs a row across, so every sg arm here is unreachable and this \
                 gate would report success having compared nothing. Set \
                 NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose.",
                gw::SG_PK_LANES
            );
        }
        eprintln!(
            "{test}: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) sg16 pk unsupported on width {width:?}"
        );
        return None;
    }
    Some(ctx)
}

fn dot32(packed: &[u32], wbase: usize, x: &[u16], kbase: usize) -> f32 {
    let mut a = 0f32;
    for j in 0..4 {
        a = dot8(packed[wbase + j], x, kbase + j * 8, a);
    }
    a
}

fn oracle_row(packed: &[u32], scales: &[u16], x: &[u16], row: usize, k: usize, gs: usize) -> f32 {
    let lanes = gw::SG_PK_LANES as usize;
    let kv = k / 32;
    let wbase = row * (k / 8);
    let sbase = row * (k / gs);
    let accs: Vec<f32> = (0..lanes)
        .map(|lane| {
            let mut acc = 0f32;
            let mut v = lane;
            while v < kv {
                let kbase = v * 32;
                let sc = d(scales[sbase + kbase / gs]);
                acc = sc.mul_add(dot32(packed, wbase + v * 4, x, kbase), acc);
                v += lanes;
            }
            acc
        })
        .collect();
    tree_sum(&accs)
}

fn oracle_packed(
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    n: usize,
    k: usize,
    gs: usize,
) -> Vec<u32> {
    (0..n / 2)
        .map(|p| {
            let lo = bf16::from_f32(oracle_row(packed, scales, x, 2 * p, k, gs)).to_bits() as u32;
            let hi =
                bf16::from_f32(oracle_row(packed, scales, x, 2 * p + 1, k, gs)).to_bits() as u32;
            lo | (hi << 16)
        })
        .collect()
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SplitParams {
    q_rows: u32,
    kv_rows: u32,
    v_off: u32,
    pad0: u32,
}

struct Case {
    packed: Vec<u32>,
    scales: Vec<u16>,
    x: Vec<u16>,
    n: usize,
    k: usize,
    gs: usize,
}

fn mk_case(seed: u64, n: usize, k: usize, gs: usize) -> Case {
    let mut rng = Lcg(seed);
    Case {
        packed: rng.packed(n * (k / 8)),
        scales: rng.scales(n * (k / gs)),
        x: rng.bf16_words(k, 1.5),
        n,
        k,
        gs,
    }
}

fn common_bufs_rows(
    ctx: &WgpuContext,
    case: &Case,
    rows_per_group: u32,
) -> (
    wgpu::Buffer,
    wgpu::Buffer,
    wgpu::Buffer,
    wgpu::Buffer,
    (u32, u32, u32),
) {
    let groups = dispatch::workgroup_count_1d(ctx, case.n as u64, rows_per_group);
    let params = Params {
        n_rows: case.n as u32,
        k_elems: case.k as u32,
        gs: case.gs as u32,
        w_row_words: (case.k / 8) as u32,
        scale_row_stride: (case.k / case.gs) as u32,
        groups_x: groups.0,
    };
    (
        dispatch::storage_from_slice(ctx, "sgpk-packed", &case.packed),
        dispatch::storage_from_slice(ctx, "sgpk-scale", &gw::pack_scale_words(&case.scales)),
        dispatch::storage_from_slice(ctx, "sgpk-x", &pack_u16(&case.x)),
        dispatch::uniform_from(ctx, "sgpk-params", &params),
        groups,
    )
}

fn common_bufs(
    ctx: &WgpuContext,
    case: &Case,
) -> (
    wgpu::Buffer,
    wgpu::Buffer,
    wgpu::Buffer,
    wgpu::Buffer,
    (u32, u32, u32),
) {
    common_bufs_rows(ctx, case, gw::SG_PK_ROWS)
}

fn run_source(
    ctx: &WgpuContext,
    label: &str,
    source: &str,
    entry: &str,
    binds: &[(u32, &wgpu::Buffer)],
    groups: (u32, u32, u32),
) {
    let pipeline = dispatch::cached_compute_pipeline(ctx, label, source, entry)
        .unwrap_or_else(|e| panic!("pipeline {entry}: {e}"));
    let group = dispatch::bind_group(ctx, &pipeline, binds);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
    }
    ctx.queue.submit([enc.finish()]);
    ctx.poll_blocking().expect("poll");
}

fn run_entry(
    ctx: &WgpuContext,
    entry: &str,
    binds: &[(u32, &wgpu::Buffer)],
    groups: (u32, u32, u32),
) {
    run_source(
        ctx,
        "nv_kernels_gemv_w4a16_sg_pk",
        &gw::sg_pk_source(),
        entry,
        binds,
        groups,
    );
}

#[test]
fn sg_pk_matches_the_16_lane_oracle_on_every_e4b_shape() {
    let Some(ctx) = sg_ctx_or_skip("sg_pk_matches_the_16_lane_oracle_on_every_e4b_shape") else {
        return;
    };
    let word_off = 4usize;
    for (name, n, k, gs) in [
        ("qkv", 3072usize, 2560usize, 32usize),
        ("o", 2560, 2048, 32),
        ("gate_up", 20480, 2560, 64),
        ("down", 2560, 10240, 32),
        ("plig", 256, 2048, 32),
        ("plp", 2048, 256, 32),
        ("ragged", 42, 96, 32),
    ] {
        let case = mk_case(0x51_6b_00 ^ (n as u64) ^ ((k as u64) << 24), n, k, gs);
        let (packed, scale, x, params, groups) = common_bufs(ctx, &case);
        let y = dispatch::storage_zeroed(ctx, "sgpk-y", ((n / 2 + word_off) * 4) as u64);
        let off = dispatch::uniform_from(
            ctx,
            "sgpk-off",
            &OffParams {
                dst_word_off: word_off as u32,
                ..Default::default()
            },
        );
        run_entry(
            ctx,
            gw::SG_PK_ENTRY,
            &[
                (1, &scale),
                (3, &y),
                (4, &params),
                (6, &packed),
                (7, &x),
                (30, &off),
            ],
            groups,
        );
        let got: Vec<u32> = dispatch::read_back(ctx, &y, n / 2 + word_off).expect("read_back");
        assert!(
            got[..word_off].iter().all(|w| *w == 0),
            "{name}: words below dst_word_off must stay untouched"
        );
        let want = oracle_packed(&case.packed, &case.scales, &case.x, n, k, gs);
        let mismatch = want
            .iter()
            .zip(got[word_off..].iter())
            .filter(|(a, b)| a != b)
            .count();
        eprintln!(
            "sg_pk {name:<8} n={n:<6} k={k:<6} gs={gs} | {mismatch}/{} words off",
            n / 2
        );
        assert_eq!(
            mismatch, 0,
            "{name}: sg_pk must match the 16-lane oracle bit-exactly"
        );
    }
}

#[test]
fn sg_pk3_routes_q_k_v_including_the_k_eq_v_layout() {
    let Some(ctx) = sg_ctx_or_skip("sg_pk3_routes_q_k_v_including_the_k_eq_v_layout") else {
        return;
    };
    for (name, q_rows, kv_rows, has_v, k, gs) in [
        ("sliding", 2048usize, 512usize, true, 2560usize, 32usize),
        ("full_keqv", 2048, 512, false, 2560, 64),
        ("ragged", 96, 32, true, 128, 32),
    ] {
        let n = q_rows + kv_rows * if has_v { 2 } else { 1 };
        let v_off = if has_v { q_rows + kv_rows } else { q_rows };
        let case = mk_case(0x33_71 ^ (n as u64) ^ ((k as u64) << 20), n, k, gs);
        let (packed, scale, x, params, groups) = common_bufs(ctx, &case);
        let yq = dispatch::storage_zeroed(ctx, "sgpk3-q", (q_rows / 2 * 4) as u64);
        let yk = dispatch::storage_zeroed(ctx, "sgpk3-k", (kv_rows / 2 * 4) as u64);
        let yv = dispatch::storage_zeroed(ctx, "sgpk3-v", (kv_rows / 2 * 4) as u64);
        let sp = dispatch::uniform_from(
            ctx,
            "sgpk3-split",
            &SplitParams {
                q_rows: q_rows as u32,
                kv_rows: kv_rows as u32,
                v_off: v_off as u32,
                pad0: 0,
            },
        );
        run_entry(
            ctx,
            gw::SG_PK3_ENTRY,
            &[
                (1, &scale),
                (4, &params),
                (6, &packed),
                (7, &x),
                (31, &yq),
                (32, &yk),
                (33, &yv),
                (34, &sp),
            ],
            groups,
        );
        let want = oracle_packed(&case.packed, &case.scales, &case.x, n, k, gs);
        let got_q: Vec<u32> = dispatch::read_back(ctx, &yq, q_rows / 2).unwrap();
        let got_k: Vec<u32> = dispatch::read_back(ctx, &yk, kv_rows / 2).unwrap();
        let got_v: Vec<u32> = dispatch::read_back(ctx, &yv, kv_rows / 2).unwrap();
        let mut mismatch = 0usize;
        for (i, w) in got_q.iter().enumerate() {
            if *w != want[i] {
                mismatch += 1;
            }
        }
        for (i, w) in got_k.iter().enumerate() {
            if *w != want[q_rows / 2 + i] {
                mismatch += 1;
            }
        }
        for (i, w) in got_v.iter().enumerate() {
            if *w != want[v_off / 2 + i] {
                mismatch += 1;
            }
        }
        eprintln!(
            "sg_pk3 {name:<10} q={q_rows} kv={kv_rows} v_off={v_off} k={k} gs={gs} | {mismatch} words off"
        );
        assert_eq!(
            mismatch, 0,
            "{name}: sg_pk3 routing must match the oracle bit-exactly"
        );
    }
}

#[test]
fn sg_pkm_matches_the_16_lane_oracle_at_every_row_blocking() {
    let Some(ctx) = sg_ctx_or_skip("sg_pkm_matches_the_16_lane_oracle_at_every_row_blocking")
    else {
        return;
    };
    let word_off = 4usize;
    for mr in [2u32, 4, 8] {
        let src = gw::sg_pk_mr_source(mr);
        let label = format!("nv_kernels_gemv_w4a16_sg_pkm{mr}");
        for (name, n, k, gs) in [
            ("qkv", 3072usize, 2560usize, 32usize),
            ("o", 2560, 2048, 32),
            ("gate_up", 20480, 2560, 64),
            ("down", 2560, 10240, 32),
            ("plig", 256, 2048, 32),
            ("plp", 2048, 256, 32),
            ("ragged", 42, 96, 32),
        ] {
            let case = mk_case(0x51_6b_00 ^ (n as u64) ^ ((k as u64) << 24), n, k, gs);
            let (packed, scale, x, params, groups) =
                common_bufs_rows(ctx, &case, gw::SG_PK_ROWS * mr);
            let y = dispatch::storage_zeroed(ctx, "sgpkm-y", ((n / 2 + word_off) * 4) as u64);
            let off = dispatch::uniform_from(
                ctx,
                "sgpkm-off",
                &OffParams {
                    dst_word_off: word_off as u32,
                    ..Default::default()
                },
            );
            run_source(
                ctx,
                &label,
                &src,
                gw::SG_PKM_ENTRY,
                &[
                    (1, &scale),
                    (3, &y),
                    (4, &params),
                    (6, &packed),
                    (7, &x),
                    (30, &off),
                ],
                groups,
            );
            let got: Vec<u32> = dispatch::read_back(ctx, &y, n / 2 + word_off).expect("read_back");
            assert!(
                got[..word_off].iter().all(|w| *w == 0),
                "mr={mr} {name}: words below dst_word_off must stay untouched"
            );
            let want = oracle_packed(&case.packed, &case.scales, &case.x, n, k, gs);
            let mismatch = want
                .iter()
                .zip(got[word_off..].iter())
                .filter(|(a, b)| a != b)
                .count();
            eprintln!(
                "sg_pkm mr={mr} {name:<8} n={n:<6} k={k:<6} gs={gs} | {mismatch}/{} words off",
                n / 2
            );
            assert_eq!(
                mismatch, 0,
                "mr={mr} {name}: sg_pkm must match the 16-lane oracle bit-exactly"
            );
        }
    }
}

#[test]
fn sg_pkm3_routes_q_k_v_bit_exactly_at_every_row_blocking() {
    let Some(ctx) = sg_ctx_or_skip("sg_pkm3_routes_q_k_v_bit_exactly_at_every_row_blocking") else {
        return;
    };
    for mr in [2u32, 4, 8] {
        let src = gw::sg_pk_mr_source(mr);
        let label = format!("nv_kernels_gemv_w4a16_sg_pkm{mr}");
        for (name, q_rows, kv_rows, has_v, k, gs) in [
            ("sliding", 2048usize, 512usize, true, 2560usize, 32usize),
            ("full_keqv", 2048, 512, false, 2560, 64),
            ("ragged", 96, 32, true, 128, 32),
        ] {
            let n = q_rows + kv_rows * if has_v { 2 } else { 1 };
            let v_off = if has_v { q_rows + kv_rows } else { q_rows };
            let case = mk_case(0x33_71 ^ (n as u64) ^ ((k as u64) << 20), n, k, gs);
            let (packed, scale, x, params, groups) =
                common_bufs_rows(ctx, &case, gw::SG_PK_ROWS * mr);
            let yq = dispatch::storage_zeroed(ctx, "sgpkm3-q", (q_rows / 2 * 4) as u64);
            let yk = dispatch::storage_zeroed(ctx, "sgpkm3-k", (kv_rows / 2 * 4) as u64);
            let yv = dispatch::storage_zeroed(ctx, "sgpkm3-v", (kv_rows / 2 * 4) as u64);
            let sp = dispatch::uniform_from(
                ctx,
                "sgpkm3-split",
                &SplitParams {
                    q_rows: q_rows as u32,
                    kv_rows: kv_rows as u32,
                    v_off: v_off as u32,
                    pad0: 0,
                },
            );
            run_source(
                ctx,
                &label,
                &src,
                gw::SG_PKM3_ENTRY,
                &[
                    (1, &scale),
                    (4, &params),
                    (6, &packed),
                    (7, &x),
                    (31, &yq),
                    (32, &yk),
                    (33, &yv),
                    (34, &sp),
                ],
                groups,
            );
            let want = oracle_packed(&case.packed, &case.scales, &case.x, n, k, gs);
            let got_q: Vec<u32> = dispatch::read_back(ctx, &yq, q_rows / 2).unwrap();
            let got_k: Vec<u32> = dispatch::read_back(ctx, &yk, kv_rows / 2).unwrap();
            let got_v: Vec<u32> = dispatch::read_back(ctx, &yv, kv_rows / 2).unwrap();
            let mut mismatch = 0usize;
            for (i, w) in got_q.iter().enumerate() {
                if *w != want[i] {
                    mismatch += 1;
                }
            }
            for (i, w) in got_k.iter().enumerate() {
                if *w != want[q_rows / 2 + i] {
                    mismatch += 1;
                }
            }
            for (i, w) in got_v.iter().enumerate() {
                if *w != want[v_off / 2 + i] {
                    mismatch += 1;
                }
            }
            eprintln!(
                "sg_pkm3 mr={mr} {name:<10} q={q_rows} kv={kv_rows} v_off={v_off} k={k} gs={gs} | {mismatch} words off"
            );
            assert_eq!(
                mismatch, 0,
                "mr={mr} {name}: sg_pkm3 routing must match the oracle bit-exactly"
            );
        }
    }
}
