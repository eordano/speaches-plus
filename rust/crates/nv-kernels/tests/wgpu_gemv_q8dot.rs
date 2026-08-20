#![cfg(feature = "wgpu")]

mod common;
use common::wgpu_allow_skip;
use half::{bf16, f16};
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::{compose, dispatch};
use common::ctx_or_skip_quiet_unqualified as ctx_or_skip;

const Q8DOT_WGSL: &str = include_str!("../wgsl/gemv_q8dot.wgsl");
const Q3D_I8_WGSL: &str = include_str!("../wgsl/q3d_gemv_i8.wgsl");
const QUANT_ENTRY: &str = "q8d_quantize_x";
const GEMV_ENTRY: &str = "q8d_gemv";
const GEMV_SMEM_ENTRY: &str = "q8d_gemv_smem";
const GEMV_SG_ENTRY: &str = "q8d_gemv_sg";
const GEMV_ENTRIES: [&str; 3] = [GEMV_ENTRY, GEMV_SMEM_ENTRY, GEMV_SG_ENTRY];
const Q3D_I8_ENTRY: &str = "q3d_gemv_i8";

const GROUP_INDEX_ANCHOR: &str = "let gi = b / q8dg_p.group_blocks;";
const SMEM_GROUP_INDEX_ANCHOR: &str = "let s = q8dg_ws[sbase + gb / q8dg_p.group_blocks];";

const GEMV_VS_INTEGER_ORACLE_REL_TOL_IS_TIGHT_BECAUSE_BOTH_SIDES_DO_EXACT_INT_DOTS: f32 = 1e-4;
const OPFDIV_ULP_SLACK_ALLOWS_ONE_CODE_OF_TIE_FLIP: i32 = 1;
const E2E_VS_F64_ORACLE_REL_TOL_BUDGETS_ONLY_THE_Q8_ACTIVATION_QUANT_ERROR: f32 = 3e-2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct QuantParams {
    k_blocks: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvParams {
    n_rows: u32,
    k_blocks: u32,
    group_blocks: u32,
    groups_x: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Q3q8Params {
    n_rows: u32,
    k_elems: u32,
    groups_x: u32,
    groups_per_row: u32,
    group_shift: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_unit_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn next_i8(&mut self) -> i8 {
        ((self.next_u32() % 255) as i32 - 127) as i8
    }
}

fn gen_x_bf16(k: usize, seed: u64) -> Vec<u16> {
    let mut rng = Lcg::new(seed);
    (0..k)
        .map(|_| bf16::from_f32(rng.next_unit_f32()).to_bits())
        .collect()
}

fn pack_u16(v: &[u16]) -> Vec<u32> {
    v.chunks(2)
        .map(|c| (c[0] as u32) | ((*c.get(1).unwrap_or(&0) as u32) << 16))
        .collect()
}

struct QuantizedActs {
    q_words: Vec<u32>,
    ds_words: Vec<u32>,
}

fn host_quantize(x: &[u16]) -> QuantizedActs {
    assert!(x.len() % 32 == 0, "q8 blocks are 32 elements");
    let k_blocks = x.len() / 32;
    let mut q_words = Vec::with_capacity(k_blocks * 8);
    let mut ds_words = Vec::with_capacity(k_blocks);
    for b in 0..k_blocks {
        let vals: Vec<f32> = x[b * 32..(b + 1) * 32]
            .iter()
            .map(|&bits| bf16::from_bits(bits).to_f32())
            .collect();
        let amax = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let dinv = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut isum = 0i32;
        for i in 0..8 {
            let mut word = 0u32;
            for j in 0..4 {
                let q = (vals[4 * i + j] * dinv).round_ties_even() as i32;
                isum += q;
                word |= ((q as u8) as u32) << (8 * j);
            }
            q_words.push(word);
        }
        let d16 = f16::from_f32(d).to_bits() as u32;
        let dsum16 = f16::from_f32(d * isum as f32).to_bits() as u32;
        ds_words.push(d16 | (dsum16 << 16));
    }
    QuantizedActs { q_words, ds_words }
}

struct WeightsI8 {
    rows: Vec<Vec<i8>>,
    scales: Vec<Vec<f32>>,
    group_blocks: usize,
}

fn gen_weights(n: usize, k: usize, group_elems: usize, seed: u64) -> WeightsI8 {
    assert!(k % 32 == 0 && group_elems % 32 == 0 && k % group_elems == 0);
    let mut rng = Lcg::new(seed);
    let rows: Vec<Vec<i8>> = (0..n)
        .map(|_| (0..k).map(|_| rng.next_i8()).collect())
        .collect();
    let scales: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            (0..k / group_elems)
                .map(|_| 0.004 + 0.008 * (rng.next_u32() as f32 / u32::MAX as f32))
                .collect()
        })
        .collect();
    WeightsI8 {
        rows,
        scales,
        group_blocks: group_elems / 32,
    }
}

fn pack_weight_words(w: &WeightsI8) -> Vec<u32> {
    let mut out = Vec::with_capacity(w.rows.len() * w.rows[0].len() / 4);
    for row in &w.rows {
        for quad in row.chunks(4) {
            let mut word = 0u32;
            for (j, &v) in quad.iter().enumerate() {
                word |= ((v as u8) as u32) << (8 * j);
            }
            out.push(word);
        }
    }
    out
}

fn flat_scales(w: &WeightsI8) -> Vec<f32> {
    w.scales.iter().flatten().copied().collect()
}

fn integer_oracle(w: &WeightsI8, acts: &QuantizedActs) -> Vec<f32> {
    let k_blocks = acts.ds_words.len();
    w.rows
        .iter()
        .zip(&w.scales)
        .map(|(row, srow)| {
            let mut acc = 0f64;
            for b in 0..k_blocks {
                let mut idot = 0i64;
                for i in 0..32 {
                    let xw = acts.q_words[b * 8 + i / 4];
                    let xq = ((xw >> (8 * (i % 4))) & 0xff) as u8 as i8;
                    idot += row[b * 32 + i] as i64 * xq as i64;
                }
                let d = f16::from_bits((acts.ds_words[b] & 0xffff) as u16).to_f64();
                let s = srow[b / w.group_blocks] as f64;
                acc += idot as f64 * d * s;
            }
            acc as f32
        })
        .collect()
}

fn dequant_oracle(w: &WeightsI8, x: &[u16]) -> Vec<f32> {
    w.rows
        .iter()
        .zip(&w.scales)
        .map(|(row, srow)| {
            let mut acc = 0f64;
            for (kk, &wv) in row.iter().enumerate() {
                let s = srow[kk / (w.group_blocks * 32)] as f64;
                acc += wv as f64 * s * bf16::from_bits(x[kk]).to_f64();
            }
            acc as f32
        })
        .collect()
}

fn run_quantize(ctx: &WgpuContext, label: &str, x: &[u16]) -> QuantizedActs {
    let k_blocks = x.len() / 32;
    let params = QuantParams {
        k_blocks: k_blocks as u32,
        ..Default::default()
    };
    let xb = dispatch::storage_from_slice(ctx, "q8d-x", &pack_u16(x));
    let qb = dispatch::storage_from_slice(ctx, "q8d-q", &vec![0u32; k_blocks * 8]);
    let dsb = dispatch::storage_from_slice(ctx, "q8d-ds", &vec![0u32; k_blocks]);
    let ub = dispatch::uniform_from(ctx, "q8d-qp", &params);
    dispatch::run(
        ctx,
        label,
        Q8DOT_WGSL,
        QUANT_ENTRY,
        &[(0, &xb), (1, &qb), (2, &dsb), (3, &ub)],
        (k_blocks.div_ceil(64) as u32, 1, 1),
    )
    .expect("quantize dispatch");
    QuantizedActs {
        q_words: dispatch::read_back(ctx, &qb, k_blocks * 8).expect("q read back"),
        ds_words: dispatch::read_back(ctx, &dsb, k_blocks).expect("ds read back"),
    }
}

fn gemv_groups(ctx: &WgpuContext, entry: &str, n: usize) -> (u32, u32, u32) {
    match entry {
        "q8d_gemv_smem" | "q8d_gemv_sg" => {
            dispatch::workgroup_count_1d(ctx, n.div_ceil(8) as u64, 1)
        }
        _ => dispatch::workgroup_count_1d(ctx, n as u64, 1),
    }
}

fn xdf_from_ds(acts: &QuantizedActs) -> Vec<f32> {
    acts.ds_words
        .iter()
        .map(|w| f16::from_bits((*w & 0xffff) as u16).to_f32())
        .collect()
}

fn run_gemv(
    ctx: &WgpuContext,
    label: &str,
    src: &str,
    entry: &str,
    w: &WeightsI8,
    acts: &QuantizedActs,
) -> Vec<f32> {
    let n = w.rows.len();
    let k_blocks = acts.ds_words.len();
    let groups = gemv_groups(ctx, entry, n);
    let params = GemvParams {
        n_rows: n as u32,
        k_blocks: k_blocks as u32,
        group_blocks: w.group_blocks as u32,
        groups_x: groups.0,
    };
    let wb = dispatch::storage_from_slice(ctx, "q8d-w", &pack_weight_words(w));
    let sb = dispatch::storage_from_slice(ctx, "q8d-s", &flat_scales(w));
    let qb = dispatch::storage_from_slice(ctx, "q8d-xq", &acts.q_words);
    let dsb = dispatch::storage_from_slice(ctx, "q8d-xds", &acts.ds_words);
    let yb = dispatch::storage_from_slice(ctx, "q8d-y", &vec![0x7fc00000u32; n]);
    let ub = dispatch::uniform_from(ctx, "q8d-gp", &params);
    let dfb = dispatch::storage_from_slice(ctx, "q8d-xdf", &xdf_from_ds(acts));
    let mut bindings: Vec<(u32, &wgpu::Buffer)> =
        vec![(4, &wb), (5, &sb), (6, &qb), (8, &ub), (9, &yb)];
    if entry == GEMV_SG_ENTRY {
        bindings.push((11, &dfb));
    } else {
        bindings.push((7, &dsb));
    }
    dispatch::run(ctx, label, src, entry, &bindings, groups).expect("gemv dispatch");
    let words: Vec<u32> = dispatch::read_back(ctx, &yb, n).expect("y read back");
    words.iter().map(|v| f32::from_bits(*v)).collect()
}

fn max_rel_mismatch(got: &[f32], want: &[f32], floor: f32) -> (f32, usize) {
    let mut worst = (0f32, 0usize);
    for (row, (a, b)) in got.iter().zip(want).enumerate() {
        let d = (a - b).abs() / b.abs().max(floor);
        if d > worst.0 {
            worst = (d, row);
        }
    }
    worst
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| (x * x) as f64).sum::<f64>() / v.len() as f64).sqrt() as f32
}

#[test]
fn activation_quantizer_matches_host_emulation_within_opfdiv_ulp_slack() {
    let Some(ctx) = ctx_or_skip("q8dot_quant_parity") else {
        return;
    };
    for &(k, seed) in &[(64usize, 41u64), (416, 42), (5120, 43)] {
        let x = gen_x_bf16(k, seed);
        let got = run_quantize(ctx, "q8d-quant-parity", &x);
        let want = host_quantize(&x);
        for i in 0..want.q_words.len() {
            for j in 0..4 {
                let gq = ((got.q_words[i] >> (8 * j)) & 0xff) as u8 as i8 as i32;
                let wq = ((want.q_words[i] >> (8 * j)) & 0xff) as u8 as i8 as i32;
                assert!(
                    (gq - wq).abs() <= OPFDIV_ULP_SLACK_ALLOWS_ONE_CODE_OF_TIE_FLIP,
                    "k={k} seed={seed}: q elem {} gpu={gq} host={wq}: more than one code \
                     apart cannot come from Vulkan's 2.5-ULP division slack",
                    i * 4 + j
                );
            }
        }
        for (b, (g, w)) in got.ds_words.iter().zip(&want.ds_words).enumerate() {
            let gd = f16::from_bits((*g & 0xffff) as u16).to_f32();
            let wd = f16::from_bits((*w & 0xffff) as u16).to_f32();
            assert!(
                (gd - wd).abs() <= wd.abs() * 2e-3,
                "k={k} seed={seed}: block {b} scale d gpu={gd:e} host={wd:e}: beyond f16 \
                 rounding plus division slack"
            );
        }
    }
}

#[test]
fn gemv_matches_integer_oracle_given_identical_quantized_inputs() {
    let Some(ctx) = ctx_or_skip("q8dot_gemv_parity") else {
        return;
    };
    for &(n, k, group_elems, seed) in &[
        (5usize, 64usize, 32usize, 51u64),
        (33, 416, 32, 52),
        (129, 1024, 128, 53),
        (256, 2048, 256, 54),
    ] {
        let w = gen_weights(n, k, group_elems, seed);
        let x = gen_x_bf16(k, seed ^ 0xfeed);
        let acts = host_quantize(&x);
        let want = integer_oracle(&w, &acts);
        for entry in GEMV_ENTRIES {
            let got = run_gemv(ctx, "q8d-gemv-parity", Q8DOT_WGSL, entry, &w, &acts);
            let (d, row) = max_rel_mismatch(&got, &want, rms(&want));
            assert!(
                d < GEMV_VS_INTEGER_ORACLE_REL_TOL_IS_TIGHT_BECAUSE_BOTH_SIDES_DO_EXACT_INT_DOTS,
                "{entry} n={n} k={k} g={group_elems}: row {row} rel {d:.3e}: integer dots are \
                 exact on both sides, so this can only be an indexing or scale-application bug"
            );
        }
    }
}

#[test]
fn gpu_quantize_feeding_gpu_gemv_tracks_the_f64_dequant_oracle_at_serving_shapes() {
    let Some(ctx) = ctx_or_skip("q8dot_e2e") else {
        return;
    };
    for &(n, k, group_elems, seed) in
        &[(5120usize, 5120usize, 128usize, 61u64), (12288, 5120, 128, 62)]
    {
        let w = gen_weights(n, k, group_elems, seed);
        let x = gen_x_bf16(k, seed ^ 0xe2e);
        let acts = run_quantize(ctx, "q8d-e2e-quant", &x);
        let int_want = integer_oracle(&w, &acts);
        let want = dequant_oracle(&w, &x);
        for entry in GEMV_ENTRIES {
            let got = run_gemv(ctx, "q8d-e2e-gemv", Q8DOT_WGSL, entry, &w, &acts);
            let (di, rowi) = max_rel_mismatch(&got, &int_want, rms(&int_want));
            assert!(
                di < GEMV_VS_INTEGER_ORACLE_REL_TOL_IS_TIGHT_BECAUSE_BOTH_SIDES_DO_EXACT_INT_DOTS,
                "{entry} n={n} k={k}: row {rowi} rel {di:.3e} against the integer oracle on \
                 the GPU-quantized activations: the gemv itself is wrong at serving shapes, \
                 independent of quantizer error"
            );
            let floor = rms(&want);
            let (d, row) = max_rel_mismatch(&got, &want, floor);
            assert!(
                d < E2E_VS_F64_ORACLE_REL_TOL_BUDGETS_ONLY_THE_Q8_ACTIVATION_QUANT_ERROR,
                "{entry} n={n} k={k}: row {row} rel {d:.3e} (floor {floor:.3e}): end-to-end \
                 error exceeds the q8 activation quantization budget"
            );
        }
    }
}

#[test]
fn planted_group_index_swap_is_caught_by_the_parity_harness() {
    let Some(ctx) = ctx_or_skip("q8dot_planted_bug") else {
        return;
    };
    let plants = [
        (
            GEMV_ENTRY,
            GROUP_INDEX_ANCHOR,
            "let gi = b % q8dg_p.group_blocks;",
        ),
        (
            GEMV_SMEM_ENTRY,
            SMEM_GROUP_INDEX_ANCHOR,
            "let s = q8dg_ws[sbase + gb % q8dg_p.group_blocks];",
        ),
    ];
    let (n, k, group_elems, seed) = (65usize, 1024usize, 128usize, 71u64);
    let w = gen_weights(n, k, group_elems, seed);
    let x = gen_x_bf16(k, seed ^ 0xbad);
    let acts = host_quantize(&x);
    let want = integer_oracle(&w, &acts);
    for (entry, anchor, swap) in plants {
        let bad = Q8DOT_WGSL.replacen(anchor, swap, 1);
        assert_ne!(
            bad, Q8DOT_WGSL,
            "{entry}: group-index anchor missing from gemv_q8dot.wgsl; the planted-bug gate \
             is vacuous"
        );
        let got = run_gemv(ctx, "q8d-planted-bug", &bad, entry, &w, &acts);
        let (d, row) = max_rel_mismatch(&got, &want, rms(&want));
        assert!(
            d >= GEMV_VS_INTEGER_ORACLE_REL_TOL_IS_TIGHT_BECAUSE_BOTH_SIDES_DO_EXACT_INT_DOTS,
            "{entry}: group-index swap survived parity (worst rel {d:.3e} at row {row}); \
             per-group scales vary by construction, so the harness would miss a real \
             scale-indexing regression"
        );
    }
}

struct StepPass {
    pipeline: std::sync::Arc<wgpu::ComputePipeline>,
    bind_group: wgpu::BindGroup,
    groups: (u32, u32, u32),
}

fn time_steps(ctx: &WgpuContext, passes: &[StepPass], steps: usize) -> f64 {
    let submit = |count: usize| {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for _ in 0..count {
                for p in passes {
                    pass.set_pipeline(&p.pipeline);
                    pass.set_bind_group(0, &p.bind_group, &[]);
                    pass.dispatch_workgroups(p.groups.0, p.groups.1, p.groups.2);
                }
            }
        }
        ctx.queue.submit([enc.finish()]);
    };
    submit(5);
    ctx.poll_blocking().expect("warmup poll");
    let start = std::time::Instant::now();
    submit(steps);
    ctx.poll_blocking().expect("timed poll");
    start.elapsed().as_secs_f64()
}

#[test]
#[ignore]
fn w8a8_int_dot_loses_to_w8a16_float_fma_here_because_weight_bytes_already_match_bench() {
    let Some(ctx) = ctx_or_skip("q8dot_bench") else {
        return;
    };
    let k = 5120usize;
    let k_blocks = k / 32;
    let group_elems = 128usize;
    for &(n, steps) in &[(5120usize, 400usize), (12288, 200)] {
        let w = gen_weights(n, k, group_elems, 0xd0 ^ n as u64);
        let x = gen_x_bf16(k, 0xd1 ^ n as u64);
        let acts = host_quantize(&x);
        let weight_bytes = (n * k) as f64;

        let wq_words = pack_weight_words(&w);
        let scales = flat_scales(&w);

        let q8_groups = gemv_groups(ctx, GEMV_ENTRY, n);
        let q8_params = GemvParams {
            n_rows: n as u32,
            k_blocks: k_blocks as u32,
            group_blocks: w.group_blocks as u32,
            groups_x: q8_groups.0,
        };
        let q8_wb = dispatch::storage_from_slice(ctx, "q8b-w", &wq_words);
        let q8_sb = dispatch::storage_from_slice(ctx, "q8b-s", &scales);
        let q8_qb = dispatch::storage_from_slice(ctx, "q8b-xq", &acts.q_words);
        let q8_dsb = dispatch::storage_from_slice(ctx, "q8b-xds", &acts.ds_words);
        let q8_yb = dispatch::storage_from_slice(ctx, "q8b-y", &vec![0u32; n]);
        let q8_ub = dispatch::uniform_from(ctx, "q8b-p", &q8_params);
        let q8_pipeline =
            dispatch::cached_compute_pipeline(ctx, "q8b-gemv", Q8DOT_WGSL, GEMV_ENTRY)
                .expect("q8dot pipeline");
        let q8_bind = dispatch::bind_group(
            ctx,
            &q8_pipeline,
            &[
                (4, &q8_wb),
                (5, &q8_sb),
                (6, &q8_qb),
                (7, &q8_dsb),
                (8, &q8_ub),
                (9, &q8_yb),
            ],
        );

        let smem_groups = gemv_groups(ctx, GEMV_SMEM_ENTRY, n);
        let smem_params = GemvParams {
            groups_x: smem_groups.0,
            ..q8_params
        };
        let smem_ub = dispatch::uniform_from(ctx, "q8b-sp", &smem_params);
        let smem_pipeline =
            dispatch::cached_compute_pipeline(ctx, "q8b-gemv-smem", Q8DOT_WGSL, GEMV_SMEM_ENTRY)
                .expect("q8dot smem pipeline");
        let smem_bind = dispatch::bind_group(
            ctx,
            &smem_pipeline,
            &[
                (4, &q8_wb),
                (5, &q8_sb),
                (6, &q8_qb),
                (7, &q8_dsb),
                (8, &smem_ub),
                (9, &q8_yb),
            ],
        );

        let sg_groups = gemv_groups(ctx, GEMV_SG_ENTRY, n);
        let sg_params = GemvParams {
            groups_x: sg_groups.0,
            ..q8_params
        };
        let sg_ub = dispatch::uniform_from(ctx, "q8b-sgp", &sg_params);
        let sg_dfb = dispatch::storage_from_slice(ctx, "q8b-xdf", &xdf_from_ds(&acts));
        let sg_pipeline =
            dispatch::cached_compute_pipeline(ctx, "q8b-gemv-sg", Q8DOT_WGSL, GEMV_SG_ENTRY)
                .expect("q8dot sg pipeline");
        let sg_bind = dispatch::bind_group(
            ctx,
            &sg_pipeline,
            &[
                (4, &q8_wb),
                (5, &q8_sb),
                (6, &q8_qb),
                (8, &sg_ub),
                (9, &q8_yb),
                (11, &sg_dfb),
            ],
        );

        let quant_params = QuantParams {
            k_blocks: k_blocks as u32,
            ..Default::default()
        };
        let quant_xb = dispatch::storage_from_slice(ctx, "q8b-x", &pack_u16(&x));
        let quant_qb = dispatch::storage_from_slice(ctx, "q8b-qq", &vec![0u32; k_blocks * 8]);
        let quant_dsb = dispatch::storage_from_slice(ctx, "q8b-qds", &vec![0u32; k_blocks]);
        let quant_ub = dispatch::uniform_from(ctx, "q8b-qp", &quant_params);
        let _ = (&quant_dsb, &quant_qb, QUANT_ENTRY);
        let quant_pipeline =
            dispatch::cached_compute_pipeline(ctx, "q8b-quant-df", Q8DOT_WGSL, "q8d_quantize_x_df")
                .expect("quant pipeline");
        let quant_bind = dispatch::bind_group(
            ctx,
            &quant_pipeline,
            &[(0, &quant_xb), (1, &q8_qb), (3, &quant_ub), (10, &sg_dfb)],
        );

        let w16_words_per_row = k / 4;
        let group_shift_words = 5u32;
        let w16_params = Q3q8Params {
            n_rows: n as u32,
            k_elems: k as u32,
            groups_x: dispatch::workgroup_count_1d(ctx, n.div_ceil(8) as u64, 1).0,
            groups_per_row: (w16_words_per_row >> group_shift_words) as u32,
            group_shift: group_shift_words,
            ..Default::default()
        };
        let w16_scales: Vec<f32> = (0..n * (w16_words_per_row >> group_shift_words))
            .map(|i| 0.004 + (i % 7) as f32 * 0.001)
            .collect();
        let w16_wb = dispatch::storage_from_slice(ctx, "w16b-w", &wq_words);
        let w16_sb = dispatch::storage_from_slice(ctx, "w16b-s", &w16_scales);
        let w16_xb = dispatch::storage_from_slice(ctx, "w16b-x", &pack_u16(&x));
        let w16_yb = dispatch::storage_from_slice(ctx, "w16b-y", &vec![0u32; n.div_ceil(2)]);
        let w16_ub = dispatch::uniform_from(ctx, "w16b-p", &w16_params);
        let w16_pipeline = dispatch::cached_compute_pipeline(
            ctx,
            "w16b-gemv",
            &compose(Q3D_I8_WGSL),
            Q3D_I8_ENTRY,
        )
        .expect("q3d i8 pipeline");
        let w16_bind = dispatch::bind_group(
            ctx,
            &w16_pipeline,
            &[
                (0, &w16_wb),
                (1, &w16_sb),
                (2, &w16_xb),
                (3, &w16_yb),
                (4, &w16_ub),
            ],
        );

        let w16_pass = StepPass {
            pipeline: w16_pipeline,
            bind_group: w16_bind,
            groups: (w16_params.groups_x, 1, 1),
        };
        let q8_pass = StepPass {
            pipeline: q8_pipeline,
            bind_group: q8_bind,
            groups: q8_groups,
        };
        let quant_pass = StepPass {
            pipeline: quant_pipeline,
            bind_group: quant_bind,
            groups: (k_blocks.div_ceil(64) as u32, 1, 1),
        };
        let smem_pass = StepPass {
            pipeline: smem_pipeline,
            bind_group: smem_bind,
            groups: smem_groups,
        };
        let sg_pass = StepPass {
            pipeline: sg_pipeline,
            bind_group: sg_bind,
            groups: sg_groups,
        };

        let t_w16 = time_steps(ctx, std::slice::from_ref(&w16_pass), steps) / steps as f64;
        let t_q8 = time_steps(ctx, std::slice::from_ref(&q8_pass), steps) / steps as f64;
        let t_smem = time_steps(ctx, std::slice::from_ref(&smem_pass), steps) / steps as f64;
        let t_sg = time_steps(ctx, std::slice::from_ref(&sg_pass), steps) / steps as f64;
        let quant_then_gemv = [quant_pass, sg_pass];
        let t_q8_full = time_steps(ctx, &quant_then_gemv, steps) / steps as f64;

        let gbps = |t: f64| weight_bytes / t / 1e9;
        eprintln!("-- n={n} k={k} i8 weight={:.1} MB", weight_bytes / 1e6);
        eprintln!(
            "   w8a16 float-fma      {:8.4} ms  eff {:7.1} GB/s",
            t_w16 * 1e3,
            gbps(t_w16)
        );
        eprintln!(
            "   w8a8 dot4I8 gemv     {:8.4} ms  eff {:7.1} GB/s  speedup {:.2}x",
            t_q8 * 1e3,
            gbps(t_q8),
            t_w16 / t_q8
        );
        eprintln!(
            "   w8a8 smem 8-row      {:8.4} ms  eff {:7.1} GB/s  speedup {:.2}x",
            t_smem * 1e3,
            gbps(t_smem),
            t_w16 / t_smem
        );
        eprintln!(
            "   w8a8 sg q3d-layout   {:8.4} ms  eff {:7.1} GB/s  speedup {:.2}x",
            t_sg * 1e3,
            gbps(t_sg),
            t_w16 / t_sg
        );
        eprintln!(
            "   w8a8 quant+smem      {:8.4} ms  eff {:7.1} GB/s  speedup {:.2}x (unamortized quant)",
            t_q8_full * 1e3,
            gbps(t_q8_full),
            t_w16 / t_q8_full
        );
    }
}
