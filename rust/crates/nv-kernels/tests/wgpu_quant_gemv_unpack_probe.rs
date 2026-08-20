#![cfg(feature = "wgpu")]

mod common;
use common::ctx;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::quant_gemv::{self, bf16_to_f32, QFormat};
use nv_kernels::wgpu_backend::kernels::{gelu_tanh_mul, gemv_nvfp4};

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / 8_388_608.0) - 1.0
    }
    fn bf16_vec(&mut self, len: usize, scale: f32) -> Vec<u16> {
        (0..len)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_bits())
            .collect()
    }
}

fn cpu_ref_i8_f64(
    wq: &[u32],
    scales: &[f32],
    x: &[u16],
    n: usize,
    k: usize,
    group: usize,
) -> Vec<f64> {
    let g = if group == 0 { k } else { group };
    let per_row = k / g;
    (0..n)
        .map(|r| {
            let mut acc = 0f64;
            for gi in 0..per_row {
                let mut d = 0f64;
                for i in 0..g {
                    let idx = r * k + gi * g + i;
                    let byte = ((wq[idx / 4] >> (8 * (idx % 4))) & 0xff) as u8;
                    d += (byte as i8) as f64 * bf16_to_f32(x[gi * g + i]) as f64;
                }
                acc += d * scales[r * per_row + gi] as f64;
            }
            acc
        })
        .collect()
}

const UNPACK_PROBE_ENTRY: &str = "qg_probe_unpack_tree";
const UNPACK_SG_PROBE_ENTRY: &str = "qg_probe_unpack_sg";

const PROBE_WGSL_KEEPS_THE_UNPACK4XI8_CANDIDATE_MEASURABLE_WITHOUT_SHIPPING_IT: &str = r#"
fn qg_probe_dot16_i8_pk(wv: vec4<u32>, xa: vec4<u32>, xb: vec4<u32>) -> f32 {
    let qa = vec4<f32>(unpack4xI8(wv.x));
    let qb = vec4<f32>(unpack4xI8(wv.y));
    let qc = vec4<f32>(unpack4xI8(wv.z));
    let qd = vec4<f32>(unpack4xI8(wv.w));
    var d = 0.0;
    d = fma(qa.x, bf16_lo(xa.x), d);
    d = fma(qa.y, bf16_hi(xa.x), d);
    d = fma(qa.z, bf16_lo(xa.y), d);
    d = fma(qa.w, bf16_hi(xa.y), d);
    d = fma(qb.x, bf16_lo(xa.z), d);
    d = fma(qb.y, bf16_hi(xa.z), d);
    d = fma(qb.z, bf16_lo(xa.w), d);
    d = fma(qb.w, bf16_hi(xa.w), d);
    d = fma(qc.x, bf16_lo(xb.x), d);
    d = fma(qc.y, bf16_hi(xb.x), d);
    d = fma(qc.z, bf16_lo(xb.y), d);
    d = fma(qc.w, bf16_hi(xb.y), d);
    d = fma(qd.x, bf16_lo(xb.z), d);
    d = fma(qd.y, bf16_hi(xb.z), d);
    d = fma(qd.z, bf16_lo(xb.w), d);
    d = fma(qd.w, bf16_hi(xb.w), d);
    return d;
}

fn qg_probe_group_acc_i8_pk(row: u32, live: bool, lane: u32) -> f32 {
    let kv = select(0u, qg_params.k_elems >> 4u, live);
    let wbase = select(0u, row * (qg_params.k_elems >> 4u), live);
    let sbase = select(0u, row * qg_params.scales_per_row, live);
    let sh = qg_params.group_shift;
    var acc = 0.0;
    for (var v = lane; v < kv; v = v + QG_LANES) {
        let d = qg_probe_dot16_i8_pk(qg_w4[wbase + v], qg_x4[2u * v], qg_x4[2u * v + 1u]);
        acc = fma(qg_row_scale[sbase + (v >> sh)], d, acc);
    }
    return acc;
}

@compute @workgroup_size(256)
fn qg_probe_unpack_tree(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (QG_LANES - 1u);
    let warp = tid / QG_LANES;
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_TREE_ROWS + warp;
    let live = row < qg_params.n_rows;
    let total = qg_reduce(tid, lane, qg_probe_group_acc_i8_pk(row, live, lane));
    if (lane == 0u && live) {
        qg_y[row] = bf16_encode(total);
    }
}

@compute @workgroup_size(128)
fn qg_probe_unpack_sg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_SG_ROWS + sgid;
    let live = row < qg_params.n_rows;
    let total = qg_butterfly(qg_probe_group_acc_i8_pk(row, live, lane));
    if (lane == 0u && live) {
        qg_y[row] = bf16_encode(total);
    }
}
"#;

fn probe_source() -> String {
    let src = quant_gemv::source();
    assert!(
        src.contains("fn qg_group_acc_i8("),
        "the shipped qg_group_acc_i8 is gone; the probes lose their oracle and the g4w pk \
         templates lose their accumulator"
    );
    format!("{src}\n{PROBE_WGSL_KEEPS_THE_UNPACK4XI8_CANDIDATE_MEASURABLE_WITHOUT_SHIPPING_IT}")
}

fn run_entry(
    ctx: &WgpuContext,
    src: &str,
    entry: &str,
    rows_per_group: u32,
    wq: &[u32],
    scales: &[f32],
    x: &[u16],
    n: usize,
    k: usize,
    group: usize,
) -> Vec<u16> {
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, rows_per_group);
    let params = quant_gemv::params_for(n, k, group, groups.0);
    let w_buf = dispatch::storage_from_slice(ctx, "xs-w", wq);
    let s_buf = dispatch::storage_from_slice(ctx, "xs-s", scales);
    let x_buf = dispatch::storage_from_slice(ctx, "xs-x", &quant_gemv::pack_x_bf16(x));
    let y_buf = dispatch::storage_zeroed(ctx, "xs-y", (n * 4) as u64);
    let p_buf = dispatch::uniform_from(ctx, "xs-p", &params);
    dispatch::run(
        ctx,
        "xs-run",
        src,
        entry,
        &[
            (0, &w_buf),
            (1, &s_buf),
            (2, &x_buf),
            (3, &y_buf),
            (4, &p_buf),
        ],
        groups,
    )
    .expect("dispatch");
    let words: Vec<u32> = dispatch::read_back(ctx, &y_buf, n).expect("read back");
    words.iter().map(|w| (*w & 0xffff) as u16).collect()
}

const CASES_SPAN_MODEL_SIZED_AND_SMALL_K: &[(usize, usize, usize)] = &[
    (132, 4352, 128),
    (64, 2176, 16),
    (48, 2176, 0),
    (40, 256, 16),
];

#[test]
fn the_unpack4xi8_probe_arms_are_bit_identical_to_the_shipped_extractbits_arms() {
    let Some(ctx) = ctx("qg_unpack_parity") else {
        return;
    };
    let probe = probe_source();
    let staged = quant_gemv::source();
    let sg_ok = gemv_nvfp4::sg32_ok(ctx);
    for &(n, k, group) in CASES_SPAN_MODEL_SIZED_AND_SMALL_K {
        let mut rng = Lcg(0x5157 ^ (n as u64) << 32 ^ (k as u64) << 8 ^ group as u64);
        let w = rng.bf16_vec(n * k, 1.0);
        let x = rng.bf16_vec(k, 1.0);
        let (wq, scales) = quant_gemv::quantize_groups(&w, n, k, group, QFormat::Int8);

        let shipped = run_entry(
            ctx,
            &staged,
            quant_gemv::INT8_GROUP_ENTRY,
            quant_gemv::TREE_ROWS_PER_GROUP,
            &wq,
            &scales,
            &x,
            n,
            k,
            group,
        );
        let unpack = run_entry(
            ctx,
            &probe,
            UNPACK_PROBE_ENTRY,
            quant_gemv::TREE_ROWS_PER_GROUP,
            &wq,
            &scales,
            &x,
            n,
            k,
            group,
        );
        let diff = shipped
            .iter()
            .zip(unpack.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diff,
            0,
            "n={n} k={k} group={group}: the unpack4xI8 tree probe differs from the shipped \
             extractBits arm in {diff}/{n} rows (first {:?}); unpack4xI8 sign-extends the same \
             bytes int8_decode does and the fma order is copied verbatim, so any diff is a real \
             decode defect, not rounding",
            shipped
                .iter()
                .zip(unpack.iter())
                .position(|(a, b)| a != b)
                .map(|i| (i, shipped[i], unpack[i]))
        );

        if sg_ok {
            let unpack_sg = run_entry(
                ctx,
                &probe,
                UNPACK_SG_PROBE_ENTRY,
                quant_gemv::SG_ROWS_PER_GROUP,
                &wq,
                &scales,
                &x,
                n,
                k,
                group,
            );
            let sg_diff = shipped
                .iter()
                .zip(unpack_sg.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                sg_diff, 0,
                "n={n} k={k} group={group}: the unpack4xI8 subgroup probe differs from the \
                 shipped tree in {sg_diff}/{n} rows; the xor butterfly and the strided tree \
                 share one association order, so this too must be bit-exact"
            );
        }

        let reference = cpu_ref_i8_f64(&wq, &scales, &x, n, k, group);
        let mut worst = 0f64;
        let mut nonzero = 0usize;
        for (i, out) in shipped.iter().enumerate() {
            if *out != 0 {
                nonzero += 1;
            }
            let got = bf16_to_f32(*out) as f64;
            let want = reference[i];
            worst = worst.max((got - want).abs() / want.abs().max(1e-3));
        }
        eprintln!(
            "unpack-parity n={n:<4} k={k:<5} group={group:<4} sg={sg_ok} | 0 diffs | worst rel \
             vs f64 {worst:.4} | {nonzero}/{n} nonzero"
        );
        assert!(
            worst < 0.05,
            "n={n} k={k} group={group}: probe and shipped arm agree but sit {worst} from the \
             f64 reference -- both arms are wrong together"
        );
        assert!(
            nonzero * 4 > n * 3,
            "n={n} k={k} group={group}: {nonzero}/{n} nonzero -- the fixture measured nothing"
        );
    }
}

struct Rig {
    pipeline: std::sync::Arc<wgpu::ComputePipeline>,
    group: wgpu::BindGroup,
    grid: (u32, u32, u32),
}

fn bench(ctx: &WgpuContext, rig: &Rig, iters: usize) -> f64 {
    let submit = |count: usize| {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&rig.pipeline);
            pass.set_bind_group(0, &rig.group, &[]);
            for _ in 0..count {
                pass.dispatch_workgroups(rig.grid.0, rig.grid.1, rig.grid.2);
            }
        }
        ctx.queue.submit([enc.finish()]);
    };
    submit(8);
    ctx.poll_blocking().expect("warmup poll");
    let start = std::time::Instant::now();
    submit(iters);
    ctx.poll_blocking().expect("timed poll");
    start.elapsed().as_secs_f64() * 1e6 / iters as f64
}

const SHIPPED_MAY_TRAIL_A_PROBE_BY_AT_MOST: f64 = 1.05;

#[test]
#[ignore = "kernel-rate suite: run alone, one per process"]
fn the_shipped_int8_group_arms_stay_within_five_percent_of_the_unpack_probes() {
    let Some(ctx) = ctx("qg_x_stage_rate") else {
        return;
    };
    let sg_ok = gemv_nvfp4::sg32_ok(ctx);
    let (n, k, group) = (43008usize, 5376usize, 128usize);
    let mut rng = Lcg(0x9e3779b9);
    let w = rng.bf16_vec(n * k, 1.0);
    let x = rng.bf16_vec(k, 1.0);
    let (wq, scales) = quant_gemv::quantize_groups(&w, n, k, group, QFormat::Int8);
    let bytes = (wq.len() + scales.len()) * 4;

    let tree_grid = dispatch::workgroup_count_1d(ctx, n as u64, quant_gemv::TREE_ROWS_PER_GROUP);
    let sg_grid = dispatch::workgroup_count_1d(ctx, n as u64, quant_gemv::SG_ROWS_PER_GROUP);
    let w_buf = dispatch::storage_from_slice(ctx, "xr-w", &wq);
    let s_buf = dispatch::storage_from_slice(ctx, "xr-s", &scales);
    let x_buf = dispatch::storage_from_slice(ctx, "xr-x", &quant_gemv::pack_x_bf16(&x));
    let y_buf = dispatch::storage_zeroed(ctx, "xr-y", (n * 4) as u64);
    let probe = probe_source();
    let shipped = quant_gemv::source();

    let mk = |label: &str, src: &str, entry: &str, grid: (u32, u32, u32)| {
        let params = quant_gemv::params_for(n, k, group, grid.0);
        let p_buf = dispatch::uniform_from(ctx, label, &params);
        let pipeline = dispatch::cached_compute_pipeline(ctx, label, src, entry).expect("pipeline");
        let group = dispatch::bind_group(
            ctx,
            &pipeline,
            &[
                (0, &w_buf),
                (1, &s_buf),
                (2, &x_buf),
                (3, &y_buf),
                (4, &p_buf),
            ],
        );
        Rig {
            pipeline,
            group,
            grid,
        }
    };
    let mut arms: Vec<(&str, Rig)> = vec![
        (
            "shipped tree",
            mk(
                "xr-ship-tree",
                &shipped,
                quant_gemv::INT8_GROUP_ENTRY,
                tree_grid,
            ),
        ),
        (
            "unpack tree ",
            mk("xr-pk-tree", &probe, UNPACK_PROBE_ENTRY, tree_grid),
        ),
    ];
    if sg_ok {
        arms.push((
            "shipped sg  ",
            mk(
                "xr-ship-sg",
                &shipped,
                quant_gemv::INT8_GROUP_SG_ENTRY,
                sg_grid,
            ),
        ));
        arms.push((
            "unpack sg   ",
            mk("xr-pk-sg", &probe, UNPACK_SG_PROBE_ENTRY, sg_grid),
        ));
    }

    let iters = 200;
    let mut best = vec![f64::MAX; arms.len()];
    for _ in 0..9 {
        for (i, (_, rig)) in arms.iter().enumerate() {
            let us = bench(ctx, rig, iters);
            if us < best[i] {
                best[i] = us;
            }
        }
    }
    eprintln!(
        "int8-group rate n={n} k={k} group={group} sg={sg_ok} over {:.1} MB",
        bytes as f64 / 1e6
    );
    for (i, (name, _)) in arms.iter().enumerate() {
        eprintln!(
            "  {name} {:9.3} us  ({:.1} GB/s)",
            best[i],
            bytes as f64 / best[i] * 1e-3
        );
    }
    assert!(
        best[0] <= best[1] * SHIPPED_MAY_TRAIL_A_PROBE_BY_AT_MOST,
        "the shipped tree arm ({:.3} us) trails the bit-exact unpack4xI8 probe ({:.3} us) by \
         more than five percent on this adapter; the probe is a drop-in accumulator body, so \
         reroute the tree entries to it and keep the extractBits text for the g4w oracles",
        best[0],
        best[1]
    );
    if sg_ok {
        assert!(
            best[2] <= best[3] * SHIPPED_MAY_TRAIL_A_PROBE_BY_AT_MOST,
            "the shipped sg arm ({:.3} us) trails the bit-exact unpack4xI8 probe ({:.3} us) by \
             more than five percent on this adapter; the probe is a drop-in accumulator body, so \
             reroute the sg entries to it and keep the extractBits text for the g4w oracles",
            best[2],
            best[3]
        );
    }
}

#[test]
fn the_gelu_fold_stays_bit_exact_against_the_split_pipeline_on_model_sized_k() {
    let Some(ctx) = ctx("qg_gelu_large_k") else {
        return;
    };
    let sg_ok = gemv_nvfp4::sg32_ok(ctx);
    let folds: &[quant_gemv::GeluFold] = if sg_ok {
        &[quant_gemv::GeluFold::Tree, quant_gemv::GeluFold::Subgroup]
    } else {
        &[quant_gemv::GeluFold::Tree]
    };
    for &(inter, k, group) in &[(68usize, 4352usize, 128usize), (32, 2176, 16)] {
        let n = 2 * inter;
        let mut rng = Lcg(0x6e11 ^ (inter as u64) << 24 ^ k as u64);
        let w = rng.bf16_vec(n * k, 1.0);
        let x = rng.bf16_vec(k, 1.0);
        let (wq, scales) = quant_gemv::quantize_groups(&w, n, k, group, QFormat::Int8);

        let mut rows = vec![0u16; n];
        quant_gemv::gemv_group_bf16(ctx, &wq, &scales, &x, &mut rows, n, k, group, QFormat::Int8)
            .expect("split gemv");
        let mut split = vec![0u16; inter];
        gelu_tanh_mul::gelu_tanh_mul_bf16(ctx, &rows[..inter], &rows[inter..], &mut split, inter)
            .expect("split gelu");

        for fold in folds {
            let mut fused = vec![0u16; inter];
            quant_gemv::gemv_group_gelu_bf16(
                ctx,
                &wq,
                &scales,
                &x,
                &mut fused,
                n,
                k,
                group,
                QFormat::Int8,
                *fold,
            )
            .expect("fused gemv+gelu");
            let diff = split
                .iter()
                .zip(fused.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                diff,
                0,
                "inter={inter} k={k} group={group} fold={fold:?}: fused differs from split in \
                 {diff}/{inter} elements at a model-sized k (first {:?}); the fold suite's cases \
                 stop at k=512, this suite carries the k>=2176 coverage",
                split
                    .iter()
                    .zip(fused.iter())
                    .position(|(a, b)| a != b)
                    .map(|i| (i, split[i], fused[i]))
            );
            eprintln!(
                "gelu large-k inter={inter:<4} k={k:<5} group={group:<4} fold={fold:?} | \
                 bit-exact ({diff} diffs)"
            );
        }
    }
}
