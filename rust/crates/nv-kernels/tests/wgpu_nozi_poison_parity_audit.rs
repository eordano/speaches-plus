#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::kernels::{gemv_bf16 as gb, gemv_nvfp4 as g4, gemv_nvfp4_v2 as v2};
use nv_kernels::wgpu_backend::kernels::{graph_decode as gd, rmsnorm, rmsnorm_residual};
use nv_kernels::wgpu_backend::{compose, dispatch, WgpuContext};
mod common;
use common::lcg_hi33_u32 as lcg;
use common::pipeline;
use common::OffParams as GbTreeOffParams;

fn ctx(what: &str) -> &'static WgpuContext {
    let ctx = WgpuContext::shared()
        .unwrap_or_else(|e| panic!("{what}: no wgpu adapter, this proof cannot pass: {e}"));
    eprintln!("{what}: {}", ctx.summary());
    ctx
}

fn bf16_pairs(n: usize, seed: &mut u64) -> Vec<u32> {
    (0..n).map(|_| lcg(seed) & 0x3f7f_3f7f).collect()
}

fn assert_bitwise_eq<T: Copy + PartialEq + std::fmt::Debug>(a: &[T], b: &[T], what: &str) {
    let bad: Vec<String> = a
        .iter()
        .zip(b.iter())
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .take(4)
        .map(|(i, (x, y))| format!("[{i}] zi {x:?} vs nozi+poison {y:?}"))
        .collect();
    assert!(
        bad.is_empty(),
        "{what}: an entry on NOZI_AUDITED_ENTRIES read workgroup memory it had not \
         written: {bad:?}"
    );
}

fn assert_nontrivial(v: &[u32], what: &str) {
    assert!(
        v.iter().any(|x| *x != 0),
        "{what}: the zero-init arm produced an all-zero output, so parity is vacuous"
    );
}

fn moved(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

fn pair_until_moved(
    tripwire: bool,
    mut f: impl FnMut() -> (Vec<u32>, Vec<u32>),
) -> (Vec<u32>, Vec<u32>) {
    let mut last = f();
    if !tripwire {
        return last;
    }
    for _ in 0..15 {
        if moved(&last.0, &last.1) > 0 {
            break;
        }
        last = f();
    }
    last
}

fn assert_tripwire_moved(a: &[u32], b: &[u32], what: &str) {
    let moved = moved(a, b);
    assert!(
        moved > 0,
        "{what}: the deliberately unsafe tripwire read identical words with and without \
         zero-init, so the poison never reached workgroup memory and every parity result \
         beside it is vacuous"
    );
    eprintln!(
        "  tripwire {what}: {moved}/{} words moved under poison",
        a.len()
    );
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ArgmaxParams {
    n: u32,
    nparts: u32,
    ring_mask: i32,
    has_ring: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ArgmaxRowsParams {
    rows: u32,
    n: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ArgmaxCapParams {
    n: u32,
    cap: f32,
    inv_cap: f32,
    softcap: u32,
}

const GD_POISON: &str = "
@compute @workgroup_size(256)
fn gdp_poison_wg(@builtin(local_invocation_id) tid: vec3<u32>) {
    let p = bitcast<f32>(0x7fc0deadu | bitcast<u32>(gd_amf_part_val[0]));
    gd_am_val[tid.x] = p;
    gd_am_idx[tid.x] = i32(0x5eed0000u | tid.x);
}

@compute @workgroup_size(256)
fn gdp_tripwire(@builtin(local_invocation_id) tid: vec3<u32>) {
    let lid = tid.x;
    if (lid == 0u) {
        gd_am_val[0] = 1.0;
        gd_am_idx[0] = 1;
    }
    workgroupBarrier();
    gd_amf_part_val[lid] = gd_am_val[lid];
    gd_amf_part_idx[lid] = gd_am_idx[lid];
}
";

#[test]
fn graph_decode_argmax_entries_are_write_before_read() {
    let ctx = ctx("nozi_graph_decode_argmax");
    let src = format!("{}\n{}", compose(gd::WGSL), GD_POISON);
    let pl_poison = pipeline(ctx, "gdp", &src, "gdp_poison_wg", false);
    let blocks = gd::ARGMAX_BLOCKS as u32;

    let mut seed = 0xa4_9a_11_03u64;
    let vocab = 262_144usize + 37;
    let logits_pk: Vec<u32> = bf16_pairs(vocab, &mut seed);
    let rows = 3usize;
    let rows_f32: Vec<f32> = (0..rows * vocab)
        .map(|_| (lcg(&mut seed) as f32 / u32::MAX as f32) - 0.5)
        .collect();

    let pk_buf = dispatch::storage_from_slice(ctx, "gd-pk", &logits_pk);
    let rows_buf = dispatch::storage_from_slice(ctx, "gd-rows", &rows_f32);
    let pos_buf = dispatch::storage_from_slice(ctx, "gd-pos", &[7i32]);
    let am_p = dispatch::uniform_from(
        ctx,
        "gd-am-p",
        &ArgmaxParams {
            n: vocab as u32,
            nparts: blocks,
            ring_mask: 63,
            has_ring: 1,
        },
    );
    let amf_p = dispatch::uniform_from(
        ctx,
        "gd-amf-p",
        &ArgmaxRowsParams {
            rows: rows as u32,
            n: vocab as u32,
            pad0: 0,
            pad1: 0,
        },
    );
    let amc_p = dispatch::uniform_from(
        ctx,
        "gd-amc-p",
        &ArgmaxCapParams {
            n: vocab as u32,
            cap: 30.0,
            inv_cap: 1.0 / 30.0,
            softcap: 1,
        },
    );

    let mut cells = 0usize;
    for entry in [
        "gdp_tripwire",
        "argmax_bf16_stage1",
        "argmax_bf16_stage2",
        "argmax_f32_rows_stage1",
        "argmax_f32_rows_stage2",
        "argmax_softcap_bf16_stage1",
    ] {
        let pl_zi = pipeline(ctx, "gd-zi", &src, entry, true);
        let pl_nozi = pipeline(ctx, "gd-nozi", &src, entry, false);
        let grid = match entry {
            "argmax_bf16_stage2" => (1u32, 1u32, 1u32),
            "argmax_f32_rows_stage1" => (blocks, rows as u32, 1),
            "argmax_f32_rows_stage2" => (rows as u32, 1, 1),
            _ => (blocks, 1, 1),
        };
        let part_val_seed: Vec<f32> = (0..rows * gd::ARGMAX_BLOCKS)
            .map(|i| ((i % 97) as f32) * 0.125 - 6.0)
            .collect();
        let part_idx_seed: Vec<i32> = (0..rows * gd::ARGMAX_BLOCKS)
            .map(|i| (i * 31 % (vocab - 1)) as i32)
            .collect();

        let run = |pl: &wgpu::ComputePipeline, poison: bool| -> Vec<u32> {
            let part_val = dispatch::storage_from_slice(ctx, "gd-pv", &part_val_seed);
            let part_idx = dispatch::storage_from_slice(ctx, "gd-pi", &part_idx_seed);
            let out = dispatch::storage_zeroed(ctx, "gd-out", (rows * 4).max(4) as u64);
            let token = dispatch::storage_zeroed(ctx, "gd-tok", 4);
            let ring = dispatch::storage_zeroed(ctx, "gd-ring", 64 * 4);
            let capped = dispatch::storage_zeroed(ctx, "gd-capped", (vocab * 4) as u64);
            let binds: Vec<(u32, &wgpu::Buffer)> = match entry {
                "gdp_tripwire" => vec![(55, &part_val), (56, &part_idx)],
                "argmax_bf16_stage1" => {
                    vec![(47, &pk_buf), (48, &part_val), (49, &part_idx), (50, &am_p)]
                }
                "argmax_bf16_stage2" => vec![
                    (48, &part_val),
                    (49, &part_idx),
                    (50, &am_p),
                    (51, &pos_buf),
                    (52, &token),
                    (53, &ring),
                ],
                "argmax_f32_rows_stage1" => vec![
                    (54, &rows_buf),
                    (55, &part_val),
                    (56, &part_idx),
                    (58, &amf_p),
                ],
                "argmax_f32_rows_stage2" => {
                    vec![(55, &part_val), (56, &part_idx), (57, &out), (58, &amf_p)]
                }
                _ => vec![
                    (65, &pk_buf),
                    (66, &capped),
                    (67, &amc_p),
                    (55, &part_val),
                    (56, &part_idx),
                ],
            };
            let bind = dispatch::bind_group(ctx, pl, &binds);
            let pbind = dispatch::bind_group(ctx, &pl_poison, &[(55, &part_val)]);
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                if poison {
                    pass.set_pipeline(&pl_poison);
                    pass.set_bind_group(0, &pbind, &[]);
                    pass.dispatch_workgroups(grid.0, grid.1, grid.2);
                }
                pass.set_pipeline(pl);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(grid.0, grid.1, grid.2);
            }
            ctx.queue.submit([enc.finish()]);
            ctx.poll_blocking().expect("poll");
            let mut got: Vec<u32> = Vec::new();
            got.extend(
                dispatch::read_back::<f32>(ctx, &part_val, rows * gd::ARGMAX_BLOCKS)
                    .expect("pv")
                    .iter()
                    .map(|v| v.to_bits()),
            );
            got.extend(
                dispatch::read_back::<i32>(ctx, &part_idx, rows * gd::ARGMAX_BLOCKS)
                    .expect("pi")
                    .iter()
                    .map(|v| *v as u32),
            );
            got.extend(dispatch::read_back::<u32>(ctx, &out, rows).expect("out"));
            got.extend(dispatch::read_back::<u32>(ctx, &token, 1).expect("tok"));
            got.extend(dispatch::read_back::<u32>(ctx, &ring, 64).expect("ring"));
            got.extend(
                dispatch::read_back::<f32>(ctx, &capped, vocab)
                    .expect("capped")
                    .iter()
                    .map(|v| v.to_bits()),
            );
            got
        };

        let (zi, nozi) = pair_until_moved(entry == "gdp_tripwire", || {
            (run(&pl_zi, false), run(&pl_nozi, true))
        });
        if entry == "gdp_tripwire" {
            assert_tripwire_moved(&zi, &nozi, "gd_am_idx");
            continue;
        }
        assert_nontrivial(&zi, entry);
        assert_bitwise_eq(&zi, &nozi, entry);
        eprintln!("nozi-parity {entry:<28} vocab={vocab} | bit-identical under poison");
        cells += 1;
    }
    assert_eq!(cells, 5, "ran {cells} argmax entries");
}

const GBP_POISON: &str = "
@compute @workgroup_size(256)
fn gbp2_poison_wg(@builtin(local_invocation_id) tid: vec3<u32>) {
    let p = bitcast<f32>(0x7fc0deadu | gemv_bf16_y[0]);
    gemv_bf16_partial[tid.x] = p;
}

@compute @workgroup_size(256)
fn gbp2_tripwire(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    if (tid == 0u) {
        gemv_bf16_partial[0] = 1.0;
    }
    workgroupBarrier();
    let row = wid.x * 256u + tid;
    if (row < gemv_bf16_params.n_rows) {
        gemv_bf16_y[row] = bitcast<u32>(gemv_bf16_partial[tid]);
    }
}
";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct BaseParams {
    n_rows: u32,
    k_elems: u32,
    w_row_words: u32,
    groups_x: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct NormedParams {
    n_rows: u32,
    k_elems: u32,
    w_row_words: u32,
    groups_x: u32,
    rstd: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RowQuantParams {
    n_rows: u32,
    k_elems: u32,
    src_row_words: u32,
    dst_row_words: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct I8Params {
    n_rows: u32,
    k_elems: u32,
    wq_row_words: u32,
    groups_x: u32,
    m_rows: u32,
    x_row_words: u32,
    pad0: u32,
    pad1: u32,
}

#[test]
fn gemv_bf16_epilogue_entries_are_write_before_read() {
    let ctx = ctx("nozi_gemv_bf16_epilogue");
    let src = format!("{}\n{}", compose(gb::WGSL), GBP_POISON);
    let pl_poison = pipeline(ctx, "gbp2", &src, "gbp2_poison_wg", false);

    let mut cells = 0usize;
    let mut tripwires = 0usize;
    for (entry, m) in [
        ("gbp2_tripwire", 1usize),
        ("gemv_bf16_normed", 1),
        ("rowquant_i8", 1),
        ("gemv_i8_normed", 1),
        ("gemv_i8_normed_mk", 1),
        ("gemv_i8_normed_mk", 5),
        ("gemv_i8_normed_mk", 8),
    ] {
        let pl_zi = pipeline(ctx, "gbe-zi", &src, entry, true);
        let pl_nozi = pipeline(ctx, "gbe-nozi", &src, entry, false);
        for n in [512usize, 517] {
            let k = 1024usize;
            let mut seed = 0x11_be_5e_edu64 ^ (n as u64) ^ ((m as u64) << 32);
            let w = bf16_pairs(n * k / 2, &mut seed);
            let x = bf16_pairs(m * k / 2, &mut seed);
            let wn = bf16_pairs(k / 2, &mut seed);
            let wq: Vec<u32> = (0..n * k / 4).map(|_| lcg(&mut seed)).collect();
            let row_scale: Vec<f32> = (0..n)
                .map(|_| (lcg(&mut seed) as f32 / u32::MAX as f32) * 0.01 + 0.001)
                .collect();
            let rstd: Vec<f32> = (0..m.max(1))
                .map(|_| (lcg(&mut seed) as f32 / u32::MAX as f32) * 0.5 + 0.5)
                .collect();

            let rows_per_group = if entry == "rowquant_i8" { 1 } else { 8 };
            let grid = dispatch::workgroup_count_1d(ctx, n as u64, rows_per_group);
            let base_p = dispatch::uniform_from(
                ctx,
                "gbe-base",
                &BaseParams {
                    n_rows: n as u32,
                    k_elems: k as u32,
                    w_row_words: (k / 2) as u32,
                    groups_x: grid.0,
                },
            );
            let normed_p = dispatch::uniform_from(
                ctx,
                "gbe-normed",
                &NormedParams {
                    n_rows: n as u32,
                    k_elems: k as u32,
                    w_row_words: (k / 2) as u32,
                    groups_x: grid.0,
                    rstd: 0.7,
                    ..Default::default()
                },
            );
            let rq_p = dispatch::uniform_from(
                ctx,
                "gbe-rq",
                &RowQuantParams {
                    n_rows: n as u32,
                    k_elems: k as u32,
                    src_row_words: (k / 2) as u32,
                    dst_row_words: (k / 4) as u32,
                    groups_x: grid.0,
                    ..Default::default()
                },
            );
            let i8_p = dispatch::uniform_from(
                ctx,
                "gbe-i8",
                &I8Params {
                    n_rows: n as u32,
                    k_elems: k as u32,
                    wq_row_words: (k / 4) as u32,
                    groups_x: grid.0,
                    m_rows: m as u32,
                    x_row_words: (k / 2) as u32,
                    ..Default::default()
                },
            );
            let wb = dispatch::storage_from_slice(ctx, "gbe-w", &w);
            let xb = dispatch::storage_from_slice(ctx, "gbe-x", &x);
            let wnb = dispatch::storage_from_slice(ctx, "gbe-wn", &wn);
            let wqb = dispatch::storage_from_slice(ctx, "gbe-wq", &wq);
            let rsb = dispatch::storage_from_slice(ctx, "gbe-rs", &row_scale);
            let rstdb = dispatch::storage_from_slice(ctx, "gbe-rstd", &rstd);

            let run = |pl: &wgpu::ComputePipeline, poison: bool| -> Vec<u32> {
                let y = dispatch::storage_zeroed(ctx, "gbe-y", (m * n * 4) as u64);
                let q = dispatch::storage_zeroed(ctx, "gbe-q", (n * k / 4 * 4) as u64);
                let sc = dispatch::storage_zeroed(ctx, "gbe-sc", (n * 4) as u64);
                let binds: Vec<(u32, &wgpu::Buffer)> = match entry {
                    "gbp2_tripwire" => vec![(2, &y), (3, &base_p)],
                    "gemv_bf16_normed" => {
                        vec![(4, &wb), (5, &xb), (6, &wnb), (7, &y), (8, &normed_p)]
                    }
                    "rowquant_i8" => vec![(9, &wb), (10, &q), (11, &sc), (12, &rq_p)],
                    _ => vec![
                        (13, &wqb),
                        (14, &rsb),
                        (15, &xb),
                        (16, &wnb),
                        (17, &rstdb),
                        (18, &y),
                        (19, &i8_p),
                    ],
                };
                let bind = dispatch::bind_group(ctx, pl, &binds);
                let pbind = dispatch::bind_group(ctx, &pl_poison, &[(2, &y)]);
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    if poison {
                        pass.set_pipeline(&pl_poison);
                        pass.set_bind_group(0, &pbind, &[]);
                        pass.dispatch_workgroups(grid.0, grid.1, grid.2);
                    }
                    pass.set_pipeline(pl);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups(grid.0, grid.1, grid.2);
                }
                ctx.queue.submit([enc.finish()]);
                ctx.poll_blocking().expect("poll");
                let mut got = dispatch::read_back::<u32>(ctx, &y, m * n).expect("y");
                got.extend(dispatch::read_back::<u32>(ctx, &q, n * k / 4).expect("q"));
                got.extend(
                    dispatch::read_back::<f32>(ctx, &sc, n)
                        .expect("sc")
                        .iter()
                        .map(|v| v.to_bits()),
                );
                got
            };

            let (zi, nozi) = pair_until_moved(entry == "gbp2_tripwire", || {
                (run(&pl_zi, false), run(&pl_nozi, true))
            });
            if entry == "gbp2_tripwire" {
                assert_tripwire_moved(&zi, &nozi, "gemv_bf16_partial");
                tripwires += 1;
                continue;
            }
            assert_nontrivial(&zi, &format!("{entry} n={n} m={m}"));
            assert_bitwise_eq(&zi, &nozi, &format!("{entry} n={n} m={m}"));
            cells += 1;
        }
        if entry != "gbp2_tripwire" {
            eprintln!("nozi-parity {entry:<24} m={m} | bit-identical under poison");
        }
    }
    assert_eq!(cells, 12, "ran {cells} epilogue cells");
    assert_eq!(tripwires, 2, "ran {tripwires} tripwires");
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
}

const RMS_F32_POISON: &str = "
@compute @workgroup_size(256)
fn rmsf_poison_wg(@builtin(local_invocation_id) tid: vec3<u32>) {
    let p = bitcast<f32>(0x7fc0deadu | rms_y[0]);
    rms_scratch[tid.x] = p;
    if (tid.x == 0u) {
        rms_shared = p;
    }
}

@compute @workgroup_size(256)
fn rmsf_tripwire(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.x + wg.y * nwg.x;
    if (row >= rms_params.batch) {
        return;
    }
    let lid = tid.x;
    if (lid == 0u) {
        rms_scratch[0] = 1.0;
    }
    workgroupBarrier();
    for (var i = lid; i < rms_params.hidden; i = i + 256u) {
        rms_y[row * rms_params.hidden + i] = bitcast<u32>(rms_scratch[i & 255u]);
    }
}
";

const RMSRES_F32_POISON: &str = "
@compute @workgroup_size(256)
fn rmsresf_poison_wg(@builtin(local_invocation_id) tid: vec3<u32>) {
    let p = bitcast<f32>(0x7fc0deadu | rmsres_out[0]);
    rmsres_scratch[tid.x] = p;
    if (tid.x == 0u) {
        rmsres_shared = p;
    }
}
";

#[test]
fn f32_norm_entries_are_write_before_read() {
    let ctx = ctx("nozi_f32_norms");
    let hidden = 2048usize;

    let src = format!("{}\n{}", compose(rmsnorm::WGSL), RMS_F32_POISON);
    let pl_poison = pipeline(ctx, "rmsf-p", &src, "rmsf_poison_wg", false);
    let mut cells = 0usize;
    for entry in ["rmsf_tripwire", "rmsnorm_f32"] {
        let pl_zi = pipeline(ctx, "rmsf-zi", &src, entry, true);
        let pl_nozi = pipeline(ctx, "rmsf-nozi", &src, entry, false);
        for batch in [1usize, 3] {
            let mut seed = 0x33_11_ee_77u64 ^ (batch as u64);
            let x: Vec<f32> = (0..batch * hidden)
                .map(|_| (lcg(&mut seed) as f32 / u32::MAX as f32) - 0.5)
                .collect();
            let w: Vec<f32> = (0..hidden)
                .map(|_| (lcg(&mut seed) as f32 / u32::MAX as f32) + 0.5)
                .collect();
            let p = dispatch::uniform_from(
                ctx,
                "rmsf-p",
                &RmsParams {
                    hidden: hidden as u32,
                    batch: batch as u32,
                    eps: 1e-6,
                    words_per_row: hidden as u32,
                },
            );
            let xb = dispatch::storage_from_slice(ctx, "rmsf-x", &x);
            let wb = dispatch::storage_from_slice(ctx, "rmsf-w", &w);
            let run = |pl: &wgpu::ComputePipeline, poison: bool| -> Vec<u32> {
                let y = dispatch::storage_zeroed(ctx, "rmsf-y", (batch * hidden * 4) as u64);
                let binds: Vec<(u32, &wgpu::Buffer)> = if entry == "rmsf_tripwire" {
                    vec![(2, &y), (3, &p)]
                } else {
                    vec![(0, &xb), (1, &wb), (2, &y), (3, &p)]
                };
                let bind = dispatch::bind_group(ctx, pl, &binds);
                let pbind = dispatch::bind_group(ctx, &pl_poison, &[(2, &y)]);
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    if poison {
                        pass.set_pipeline(&pl_poison);
                        pass.set_bind_group(0, &pbind, &[]);
                        pass.dispatch_workgroups(batch as u32, 1, 1);
                    }
                    pass.set_pipeline(pl);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups(batch as u32, 1, 1);
                }
                ctx.queue.submit([enc.finish()]);
                ctx.poll_blocking().expect("poll");
                dispatch::read_back::<u32>(ctx, &y, batch * hidden).expect("y")
            };
            let (zi, nozi) = pair_until_moved(entry == "rmsf_tripwire", || {
                (run(&pl_zi, false), run(&pl_nozi, true))
            });
            if entry == "rmsf_tripwire" {
                assert_tripwire_moved(&zi, &nozi, "rms_scratch");
                continue;
            }
            assert_nontrivial(&zi, entry);
            assert_bitwise_eq(&zi, &nozi, &format!("{entry} batch={batch}"));
            cells += 1;
        }
    }

    let src_r = format!("{}\n{}", compose(rmsnorm_residual::WGSL), RMSRES_F32_POISON);
    let plr_poison = pipeline(ctx, "rmsresf-p", &src_r, "rmsresf_poison_wg", false);
    let plr_zi = pipeline(ctx, "rmsresf-zi", &src_r, "rmsnorm_residual_f32", true);
    let plr_nozi = pipeline(ctx, "rmsresf-nozi", &src_r, "rmsnorm_residual_f32", false);
    for batch in [1usize, 3] {
        let mut seed = 0x77_ee_11_33u64 ^ (batch as u64);
        let x: Vec<f32> = (0..batch * hidden)
            .map(|_| (lcg(&mut seed) as f32 / u32::MAX as f32) - 0.5)
            .collect();
        let res0: Vec<f32> = (0..batch * hidden)
            .map(|_| (lcg(&mut seed) as f32 / u32::MAX as f32) - 0.5)
            .collect();
        let w: Vec<f32> = (0..hidden)
            .map(|_| (lcg(&mut seed) as f32 / u32::MAX as f32) + 0.5)
            .collect();
        let p = dispatch::uniform_from(
            ctx,
            "rmsresf-p",
            &RmsParams {
                hidden: hidden as u32,
                batch: batch as u32,
                eps: 1e-6,
                words_per_row: hidden as u32,
            },
        );
        let wb = dispatch::storage_from_slice(ctx, "rmsresf-w", &w);
        let run = |pl: &wgpu::ComputePipeline, poison: bool| -> Vec<u32> {
            let xb = dispatch::storage_from_slice(ctx, "rmsresf-x", &x);
            let rb = dispatch::storage_from_slice(ctx, "rmsresf-res", &res0);
            let out = dispatch::storage_zeroed(ctx, "rmsresf-out", (batch * hidden * 4) as u64);
            let bind =
                dispatch::bind_group(ctx, pl, &[(0, &xb), (1, &rb), (2, &wb), (3, &out), (4, &p)]);
            let pbind = dispatch::bind_group(ctx, &plr_poison, &[(3, &out)]);
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                if poison {
                    pass.set_pipeline(&plr_poison);
                    pass.set_bind_group(0, &pbind, &[]);
                    pass.dispatch_workgroups(batch as u32, 1, 1);
                }
                pass.set_pipeline(pl);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(batch as u32, 1, 1);
            }
            ctx.queue.submit([enc.finish()]);
            ctx.poll_blocking().expect("poll");
            let mut got = dispatch::read_back::<u32>(ctx, &rb, batch * hidden).expect("res");
            got.extend(dispatch::read_back::<u32>(ctx, &out, batch * hidden).expect("out"));
            got
        };
        let zi = run(&plr_zi, false);
        let nozi = run(&plr_nozi, true);
        assert_nontrivial(&zi, "rmsnorm_residual_f32");
        assert_bitwise_eq(&zi, &nozi, &format!("rmsnorm_residual_f32 batch={batch}"));
        cells += 1;
    }
    assert_eq!(cells, 4, "ran {cells} f32 norm cells");
    eprintln!("nozi-parity rmsnorm_f32 / rmsnorm_residual_f32 | bit-identical under poison");
}

const NV2_POISON: &str = "
@compute @workgroup_size(NV2_WG)
fn nv2_poison_wg(@builtin(local_invocation_id) tid: vec3<u32>) {
    if (tid.x < NV2_SGS) {
        nv2_pk_bits[tid.x] = (0xdead0000u | tid.x) ^ (nv2_y[0] & 0xffu);
    }
}

@compute @workgroup_size(NV2_WG)
fn nv2_tripwire(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    if (sgid == 0u && lane == 0u) {
        nv2_pk_bits[0] = 1u;
    }
    workgroupBarrier();
    if (lane == 0u) {
        let row = (wid.x + wid.y * nv2_p.groups_x) * NV2_SGS + sgid;
        if (row < nv2_p.n_rows) {
            nv2_y[row >> 1u] = nv2_pk_bits[sgid];
        }
    }
}
";

#[test]
fn nvfp4_v2_packed_entries_are_write_before_read() {
    let ctx = ctx("nozi_nvfp4_v2_pk");
    let width = ctx.subgroup_width();
    assert!(
        ctx.caps.subgroup,
        "the packed nvfp4-v2 entries reduce with subgroup shuffles; without the SUBGROUP \
         feature this proof cannot run and their audit entry is unsupported here"
    );
    assert_eq!(
        width,
        Some(v2::NV2_LANES),
        "nv2_pk_bits is sized NV2_WG/32 and indexed by subgroup id; the write-before-read \
         argument is only valid at a 32-wide subgroup, and this adapter probed {width:?}"
    );

    let cfg = v2::V2Config::default();
    let src = format!("{}\n{}", v2::source(cfg), NV2_POISON);
    let pl_poison = pipeline(ctx, "nv2-p", &src, "nv2_poison_wg", false);
    let subgroups = cfg.subgroups();

    let mut cells = 0usize;
    for entry in ["nv2_tripwire", v2::WARP_PK_ENTRY, v2::FDEC_PK_ENTRY] {
        let pl_zi = pipeline(ctx, "nv2-zi", &src, entry, true);
        let pl_nozi = pipeline(ctx, "nv2-nozi", &src, entry, false);
        for n in [512usize, 508, 505] {
            let k = 2048usize;
            let k_blocks = k / 16;
            let mut seed = 0x0f_4b_2c_91u64 ^ (n as u64);
            let w: Vec<u32> = (0..n * k_blocks * 2).map(|_| lcg(&mut seed)).collect();
            let x: Vec<u32> = (0..k_blocks * 2).map(|_| lcg(&mut seed)).collect();
            let ws: Vec<u32> = (0..g4::swizzled_scale_len(n, k_blocks) / 4)
                .map(|_| lcg(&mut seed) & 0x3f3f_3f3f)
                .collect();
            let xs: Vec<u32> = (0..k_blocks.div_ceil(4))
                .map(|_| lcg(&mut seed) & 0x3f3f_3f3f)
                .collect();

            let grid = dispatch::workgroup_count_1d(ctx, n as u64, subgroups);
            let p = dispatch::uniform_from(ctx, "nv2-pp", &g4::gemv_params(1.0, n, k, grid.0));
            let wb = dispatch::storage_from_slice(ctx, "nv2-w", &w);
            let wsb = dispatch::storage_from_slice(ctx, "nv2-ws", &ws);
            let xb = dispatch::storage_from_slice(ctx, "nv2-x", &x);
            let xsb = dispatch::storage_from_slice(ctx, "nv2-xs", &xs);
            let run = |pl: &wgpu::ComputePipeline, poison: bool| -> Vec<u32> {
                let y = dispatch::storage_zeroed(ctx, "nv2-y", (n.div_ceil(2) * 4) as u64);
                let binds: Vec<(u32, &wgpu::Buffer)> = if entry == "nv2_tripwire" {
                    vec![(v2::PARAMS_SLOT, &p), (v2::Y_SLOT, &y)]
                } else if entry == v2::FDEC_PK_ENTRY {
                    vec![
                        (v2::WS_SLOT, &wsb),
                        (v2::XS_SLOT, &xsb),
                        (v2::PARAMS_SLOT, &p),
                        (v2::Y_SLOT, &y),
                        (v2::W4_SLOT, &wb),
                        (v2::X4_SLOT, &xb),
                    ]
                } else {
                    vec![
                        (v2::W2_SLOT, &wb),
                        (v2::WS_SLOT, &wsb),
                        (v2::X2_SLOT, &xb),
                        (v2::XS_SLOT, &xsb),
                        (v2::PARAMS_SLOT, &p),
                        (v2::Y_SLOT, &y),
                    ]
                };
                let bind = dispatch::bind_group(ctx, pl, &binds);
                let pbind = dispatch::bind_group(ctx, &pl_poison, &[(v2::Y_SLOT, &y)]);
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    if poison {
                        pass.set_pipeline(&pl_poison);
                        pass.set_bind_group(0, &pbind, &[]);
                        pass.dispatch_workgroups(grid.0, grid.1, grid.2);
                    }
                    pass.set_pipeline(pl);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups(grid.0, grid.1, grid.2);
                }
                ctx.queue.submit([enc.finish()]);
                ctx.poll_blocking().expect("poll");
                dispatch::read_back::<u32>(ctx, &y, n.div_ceil(2)).expect("y")
            };
            let (zi, nozi) = pair_until_moved(entry == "nv2_tripwire", || {
                (run(&pl_zi, false), run(&pl_nozi, true))
            });
            if entry == "nv2_tripwire" {
                assert_tripwire_moved(&zi, &nozi, "nv2_pk_bits");
                continue;
            }
            assert_nontrivial(&zi, &format!("{entry} n={n}"));
            assert_bitwise_eq(&zi, &nozi, &format!("{entry} n={n} k={k}"));
            cells += 1;
        }
        if entry != "nv2_tripwire" {
            eprintln!("nozi-parity {entry:<24} | bit-identical under poison");
        }
    }
    assert_eq!(cells, 6, "ran {cells} nvfp4-v2 pk cells");
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GbTreeParams {
    n_rows: u32,
    k_elems: u32,
    w_row_words: u32,
    groups_x: u32,
}

const GB_TREE_POISON: &str = "
@compute @workgroup_size(256)
fn gbt_poison_wg(@builtin(local_invocation_id) tid: vec3<u32>) {
    let p = bitcast<f32>(0x7fc0deadu | gemv_bf16_y[0]);
    gemv_bf16_partial[tid.x] = p;
}

@compute @workgroup_size(256)
fn gbt_tripwire(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    if (tid == 0u) {
        gemv_bf16_partial[0] = 1.0;
    }
    workgroupBarrier();
    let row = wid.x * 256u + tid;
    if (row < gemv_bf16_params.n_rows) {
        gemv_bf16_y[row] = bitcast<u32>(gemv_bf16_partial[tid]);
    }
}
";

#[test]
fn gemv_bf16_tree_entries_are_write_before_read() {
    let ctx = ctx("nozi_gemv_bf16_tree");
    let src = format!("{}\n{}", compose(gb::WGSL), GB_TREE_POISON);
    let pl_poison = pipeline(ctx, "gbt", &src, "gbt_poison_wg", false);
    for (entry, n, k) in [
        ("gbt_tripwire", 512usize, 1024usize),
        (gb::VEC8_ENTRY, 512usize, 1024usize),
        (gb::SCALAR_ENTRY, 512, 1024),
        (gb::V4_TREE_ENTRY, 512, 1024),
        (gb::VEC8_ENTRY, 517, 1024),
        (gb::V4_TREE_ENTRY, 517, 1024),
    ] {
        let mut seed = 0x6e02 ^ (n as u64) ^ ((k as u64) << 17);
        let w = bf16_pairs(n * k / 2, &mut seed);
        let x = bf16_pairs(k / 2, &mut seed);
        let groups = ((n as u32).div_ceil(8), 1u32, 1u32);
        let params = GbTreeParams {
            n_rows: n as u32,
            k_elems: k as u32,
            w_row_words: (k / 2) as u32,
            groups_x: groups.0,
        };
        let p = dispatch::uniform_from(ctx, "gbt-p", &params);
        let wb = dispatch::storage_from_slice(ctx, "gbt-w", &w);
        let xb = dispatch::storage_from_slice(ctx, "gbt-x", &x);
        let run = |zero_init: bool, poison: bool| -> Vec<u32> {
            let pl = pipeline(ctx, "gbt-run", &src, entry, zero_init);
            let y = dispatch::storage_zeroed(ctx, "gbt-y", (n * 4) as u64);
            let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![(2, &y), (3, &p)];
            if entry == "gbt_tripwire" {
            } else if entry == gb::V4_TREE_ENTRY {
                binds.push((20, &wb));
                binds.push((21, &xb));
            } else {
                binds.push((0, &wb));
                binds.push((1, &xb));
            }
            let bind = dispatch::bind_group(ctx, &pl, &binds);
            let pbind = dispatch::bind_group(ctx, &pl_poison, &[(2, &y)]);
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                if poison {
                    pass.set_pipeline(&pl_poison);
                    pass.set_bind_group(0, &pbind, &[]);
                    pass.dispatch_workgroups(groups.0, 1, 1);
                }
                pass.set_pipeline(&pl);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            }
            ctx.queue.submit([enc.finish()]);
            ctx.poll_blocking().expect("poll");
            dispatch::read_back(ctx, &y, n).expect("read_back")
        };
        let tripwire = entry == "gbt_tripwire";
        let (zi, nozi) = pair_until_moved(tripwire, || (run(true, false), run(false, true)));
        if tripwire {
            assert_tripwire_moved(&zi, &nozi, "gemv_bf16_partial");
            continue;
        }
        assert_nontrivial(&zi, &format!("gemv_bf16 {entry} n={n}"));
        assert_bitwise_eq(&zi, &nozi, &format!("gemv_bf16 {entry} n={n} k={k}"));
        eprintln!("nozi-parity {entry:<24} n={n:<5} k={k:<5} | bit-identical under poison");
    }
}

const GB_SG_POISON: &str = "
@compute @workgroup_size(256)
fn sgp_poison_wg(@builtin(local_invocation_id) tid: vec3<u32>) {
    let p = bitcast<f32>(0x7fc0deadu | sg_y[0]);
    if (tid.x < 16u) {
        sg_pk_tot[tid.x] = p;
    }
}

@compute @workgroup_size(256)
fn sgp_tripwire(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    if (tid == 0u) {
        sg_pk_tot[0] = 1.0;
    }
    workgroupBarrier();
    if (tid < 16u) {
        let o = wid.x * 16u + tid;
        if (o < sg_params.n_rows) {
            sg_y[o] = bitcast<u32>(sg_pk_tot[tid]);
        }
    }
}
";

#[test]
fn gemv_bf16_sg_pk_entries_are_write_before_read() {
    let ctx = ctx("nozi_gemv_bf16_sg_pk");
    let width = ctx.subgroup_width();
    eprintln!("nozi_gemv_bf16_sg_pk: probed subgroup width {width:?}");
    let src = format!("{}\n{}", gb::sg_pk_source(), GB_SG_POISON);
    let pl_poison = pipeline(ctx, "sgp", &src, "sgp_poison_wg", false);
    for (entry, rows, n, k) in [
        ("sgp_tripwire", 16u32, 512usize, 1024usize),
        (gb::SG_PK_ENTRY_WG128, 4u32, 512usize, 1024usize),
        (gb::SG_PK_ENTRY_WG256, 8, 512, 1024),
        (gb::SG_PK_ENTRY_WG256, 8, 520, 1024),
    ] {
        let mut seed = 0x6e11 ^ (n as u64) ^ ((rows as u64) << 33);
        let w = bf16_pairs(n * k / 2, &mut seed);
        let x = bf16_pairs(k / 2, &mut seed);
        let groups = ((n as u32).div_ceil(rows), 1u32, 1u32);
        let params = GbTreeParams {
            n_rows: n as u32,
            k_elems: k as u32,
            w_row_words: (k / 2) as u32,
            groups_x: groups.0,
        };
        let p = dispatch::uniform_from(ctx, "sgp-p", &params);
        let off = dispatch::uniform_from(ctx, "sgp-off", &GbTreeOffParams::default());
        let wb = dispatch::storage_from_slice(ctx, "sgp-w", &w);
        let xb = dispatch::storage_from_slice(ctx, "sgp-x", &x);
        let y_words = n.div_ceil(2);
        let run = |zero_init: bool, poison: bool| -> Vec<u32> {
            let pl = pipeline(ctx, "sgp-run", &src, entry, zero_init);
            let y = dispatch::storage_zeroed(ctx, "sgp-y", (y_words * 4) as u64);
            let binds: Vec<(u32, &wgpu::Buffer)> = if entry == "sgp_tripwire" {
                vec![(2, &y), (3, &p)]
            } else {
                vec![(0, &wb), (1, &xb), (2, &y), (3, &p), (30, &off)]
            };
            let bind = dispatch::bind_group(ctx, &pl, &binds);
            let pbind = dispatch::bind_group(ctx, &pl_poison, &[(2, &y)]);
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                if poison {
                    pass.set_pipeline(&pl_poison);
                    pass.set_bind_group(0, &pbind, &[]);
                    pass.dispatch_workgroups(groups.0, 1, 1);
                }
                pass.set_pipeline(&pl);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            }
            ctx.queue.submit([enc.finish()]);
            ctx.poll_blocking().expect("poll");
            dispatch::read_back(ctx, &y, y_words).expect("read_back")
        };
        let tripwire = entry == "sgp_tripwire";
        let (zi, nozi) = pair_until_moved(tripwire, || (run(true, false), run(false, true)));
        if tripwire {
            assert_tripwire_moved(&zi, &nozi, "sg_pk_tot");
            continue;
        }
        assert_nontrivial(&zi, &format!("gemv_bf16 {entry} n={n}"));
        assert_bitwise_eq(&zi, &nozi, &format!("gemv_bf16 {entry} n={n} k={k}"));
        eprintln!("nozi-parity {entry:<24} n={n:<5} k={k:<5} | bit-identical under poison");
    }
}
