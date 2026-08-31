#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use nv_quant::fp8::Fp8GemmRunner;
use std::sync::Arc;

const QWEN38_E4M3_TRUNK_SHAPES_N_K: [(usize, usize); 5] = [
    (5120, 17408),
    (12288, 5120),
    (10240, 5120),
    (5120, 6144),
    (2048, 5120),
];
const M_SWEEP_2_TO_8_THE_VERIFY_ROW_RANGE: std::ops::RangeInclusive<usize> = 2..=8;
const TRAIN_M_LADDER_M1_IS_SERVING: [usize; 5] = [1, 192, 1024, 4096, 8192];
const Q38_27B_PROJECTIONS_ROLE_N_K: [(&str, usize, usize); 10] = [
    ("attn.q_proj_gated", 12288, 5120),
    ("attn.k_proj", 1024, 5120),
    ("attn.v_proj", 1024, 5120),
    ("attn.o_proj", 5120, 6144),
    ("mlp.gate_proj", 17408, 5120),
    ("mlp.up_proj", 17408, 5120),
    ("mlp.down_proj", 5120, 17408),
    ("gdn.in_proj_qkv", 10240, 5120),
    ("gdn.in_proj_z", 6144, 5120),
    ("gdn.out_proj", 5120, 6144),
];

fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

fn rnd_fp8_finite(state: &mut u64) -> u8 {
    loop {
        let b = (lcg(state) >> 40) as u8;
        if (b & 0x7f) != 0x7f {
            return b;
        }
    }
}

#[test]
fn cublaslt_e4m3_row0_per_shape_report_where_the_heuristic_plan_moves_with_m() {
    let stream: Arc<CudaStream> = match CudaContext::new(0) {
        Ok(ctx) => ctx.default_stream(),
        Err(e) => panic!(
            "no CUDA device 0: this is the Lt plan m-stability report and must not report \
             success having executed nothing: {e}"
        ),
    };
    let mut runner = Fp8GemmRunner::new(stream.clone()).expect("Lt runner");
    let m_max = *M_SWEEP_2_TO_8_THE_VERIFY_ROW_RANGE.end();
    let mut any_flip = String::new();
    for (n, k) in QWEN38_E4M3_TRUNK_SHAPES_N_K {
        let mut st = 0x1122_3344_5566_7788u64 ^ ((n as u64) << 20) ^ k as u64;
        let ha: Vec<u8> = (0..m_max * k).map(|_| rnd_fp8_finite(&mut st)).collect();
        let hw: Vec<u8> = (0..n * k).map(|_| rnd_fp8_finite(&mut st)).collect();
        let da: CudaSlice<u8> = stream.clone_htod(&ha).unwrap();
        let dw: CudaSlice<u8> = stream.clone_htod(&hw).unwrap();
        let scale_one: CudaSlice<f32> = stream.clone_htod(&[0.01f32]).unwrap();
        let mut rows0: Vec<(usize, Vec<u16>)> = Vec::new();
        for m in M_SWEEP_2_TO_8_THE_VERIFY_ROW_RANGE {
            let mut d: CudaSlice<half::bf16> =
                stream.alloc_zeros::<half::bf16>(m * n).unwrap();
            let da_m: CudaSlice<u8> = stream.clone_htod(&ha[..m * k]).unwrap();
            runner
                .matmul_e4m3_weight_row(
                    &da_m,
                    &dw,
                    &mut d,
                    m as u64,
                    n as u64,
                    k as u64,
                    &scale_one,
                    &scale_one,
                )
                .unwrap_or_else(|e| panic!("Lt matmul n={n} k={k} m={m}: {e:#}"));
            stream.synchronize().unwrap();
            let out: Vec<half::bf16> = stream.clone_dtoh(&d).unwrap();
            rows0.push((m, out[..n].iter().map(|b| b.to_bits()).collect()));
        }
        let (_, base) = &rows0[0];
        for (m, r) in &rows0 {
            let ndiff = r.iter().zip(base.iter()).filter(|(a, b)| a != b).count();
            if ndiff > 0 {
                any_flip.push_str(&format!("n={n} k={k} m={m}: row0 differs in {ndiff}/{n}\n"));
            }
        }
        let _ = da;
    }
    eprintln!(
        "[lt-plan] row0 m-stability over m=2..8 per shape:\n{}",
        if any_flip.is_empty() {
            "all shapes bitwise stable".to_string()
        } else {
            any_flip.clone()
        }
    );
}

const N5120_K17408_THE_FP8_RESIDENT_DOWN_PROJ_OF_LAYERS_56_TO_63_THE_ONLY_DEFAULT_PATH_LT_TRUNK_CALL_AT_VERIFY_M: (
    usize,
    usize,
) = (5120, 17408);

#[test]
fn cublaslt_e4m3_down_proj_shared_rows_family_map_pins_the_verify_range_plan_split_and_the_m6_chunk_head_is_bitwise_the_m6_call(
) {
    let stream: Arc<CudaStream> = match CudaContext::new(0) {
        Ok(ctx) => ctx.default_stream(),
        Err(e) => panic!(
            "no CUDA device 0: this is the down-proj Lt plan family map and must not report \
             success having executed nothing: {e}"
        ),
    };
    let mut runner = Fp8GemmRunner::new(stream.clone()).expect("Lt runner");
    let (n, k) =
        N5120_K17408_THE_FP8_RESIDENT_DOWN_PROJ_OF_LAYERS_56_TO_63_THE_ONLY_DEFAULT_PATH_LT_TRUNK_CALL_AT_VERIFY_M;
    let m_max = *M_SWEEP_2_TO_8_THE_VERIFY_ROW_RANGE.end();
    let mut st = 0xa5a5_1234_dead_beefu64;
    let ha: Vec<u8> = (0..m_max * k).map(|_| rnd_fp8_finite(&mut st)).collect();
    let hw: Vec<u8> = (0..n * k).map(|_| rnd_fp8_finite(&mut st)).collect();
    let da: CudaSlice<u8> = stream.clone_htod(&ha).unwrap();
    let dw: CudaSlice<u8> = stream.clone_htod(&hw).unwrap();
    let scale_one: CudaSlice<f32> = stream.clone_htod(&[0.01f32]).unwrap();
    let mut outs: Vec<(usize, Vec<u16>)> = Vec::new();
    for m in M_SWEEP_2_TO_8_THE_VERIFY_ROW_RANGE {
        let mut d: CudaSlice<half::bf16> = stream.alloc_zeros::<half::bf16>(m * n).unwrap();
        let da_m: CudaSlice<u8> = stream.clone_htod(&ha[..m * k]).unwrap();
        runner
            .matmul_e4m3_weight_row(
                &da_m,
                &dw,
                &mut d,
                m as u64,
                n as u64,
                k as u64,
                &scale_one,
                &scale_one,
            )
            .unwrap_or_else(|e| panic!("Lt matmul n={n} k={k} m={m}: {e:#}"));
        stream.synchronize().unwrap();
        let out: Vec<half::bf16> = stream.clone_dtoh(&d).unwrap();
        outs.push((m, out.iter().map(|b| b.to_bits()).collect()));
    }
    let mut fam_line = String::new();
    for i in 0..outs.len() {
        let (mi, oi) = &outs[i];
        let fam = (0..i)
            .find(|&j| {
                let (mj, oj) = &outs[j];
                let shared = mi.min(mj) * n;
                oi[..shared] == oj[..shared]
            })
            .map(|j| (b'A' + j as u8) as char)
            .unwrap_or((b'A' + i as u8) as char);
        fam_line.push_str(&format!("m{mi}:{fam} "));
    }
    eprintln!("[lt-down-family] shared-rows bitwise family per m (n={n} k={k}): {fam_line}");
    let _ = &da;
    let m4 = outs.iter().find(|(m, _)| *m == 4).unwrap().1.clone();
    let m6 = outs.iter().find(|(m, _)| *m == 6).unwrap().1.clone();
    assert_eq!(
        m4[..4 * n],
        m6[..4 * n],
        "the m=4 and m=6 down_proj launches no longer share a bitwise plan family; the 4-row \
         span composition rides on {{m4,m5,m6}} being one family at this shape"
    );
    for m in [7usize, 8] {
        let mut d_chunk: CudaSlice<half::bf16> = stream.alloc_zeros::<half::bf16>(m * n).unwrap();
        let da_m: CudaSlice<u8> = stream.clone_htod(&ha[..m * k]).unwrap();
        for start in [0usize, m - 4] {
            runner
                .matmul_e4m3_weight_row_span(
                    &da_m,
                    &dw,
                    &mut d_chunk,
                    start as u64,
                    4,
                    n as u64,
                    k as u64,
                    &scale_one,
                    &scale_one,
                )
                .unwrap_or_else(|e| panic!("span start={start} rows=4 m={m}: {e:#}"));
        }
        stream.synchronize().unwrap();
        let chunked: Vec<u16> = stream
            .clone_dtoh(&d_chunk)
            .unwrap()
            .iter()
            .map(|b| b.to_bits())
            .collect();
        assert_eq!(
            chunked[..4 * n],
            m4[..4 * n],
            "rows 0..3 of the (0,4)+({},4) span composition are not bitwise the single m=4 \
             call; the chunked-verify fix rides on the head span being exactly the production \
             mtp verify launch and the tail overlap rewriting its shared rows bitwise",
            m - 4
        );
    }
}

const VERIFY_TC_SENDS_EVERY_E4M3_PROJECTION_TO_THIS_TILE_SO_THE_SPAN_HEAD_MUST_BE_BITWISE_THE_M4_CALL_AT_EVERY_TRUNK_SHAPE: &str =
    "under NV_Q38_VERIFY_TC=1 the verify-chain scope makes matmul_fp8_small_m_gemv refuse every \
     m>=2 call, so all checkpoint-resident e4m3 projections -- not only the k=17408 down_proj -- \
     ride the rowquant+cublasLt tile at the true verify m, and the heuristic plan families move \
     with m per shape (the m7triage census saw (10240,5120) move at m=7 and (12288,5120) at \
     m=8); the linear.rs chunk runs m in {7,8} as two 4-row spans, rows 0..4 and rows m-4..m, \
     which erases every family boundary at or above m=7 and keeps the deep rows on the \
     {m4,m5,m6} emission family if and only if the head span is bitwise the single m=4 launch \
     at every shape the route serves";

#[test]
fn cublaslt_e4m3_all_trunk_shapes_span_head_is_bitwise_the_m4_call_because_verify_tc_routes_every_projection_to_this_tile(
) {
    let _ = VERIFY_TC_SENDS_EVERY_E4M3_PROJECTION_TO_THIS_TILE_SO_THE_SPAN_HEAD_MUST_BE_BITWISE_THE_M4_CALL_AT_EVERY_TRUNK_SHAPE;
    let stream: Arc<CudaStream> = match CudaContext::new(0) {
        Ok(ctx) => ctx.default_stream(),
        Err(e) => panic!(
            "no CUDA device 0: this is the all-shapes span-composition gate and must not report \
             success having executed nothing: {e}"
        ),
    };
    let mut runner = Fp8GemmRunner::new(stream.clone()).expect("Lt runner");
    let m_max = *M_SWEEP_2_TO_8_THE_VERIFY_ROW_RANGE.end();
    for (n, k) in QWEN38_E4M3_TRUNK_SHAPES_N_K {
        let mut st = 0x5eed_0f_a_c4de_u64 ^ ((n as u64) << 24) ^ k as u64;
        let ha: Vec<u8> = (0..m_max * k).map(|_| rnd_fp8_finite(&mut st)).collect();
        let hw: Vec<u8> = (0..n * k).map(|_| rnd_fp8_finite(&mut st)).collect();
        let dw: CudaSlice<u8> = stream.clone_htod(&hw).unwrap();
        let scale_one: CudaSlice<f32> = stream.clone_htod(&[0.01f32]).unwrap();
        let mut outs: Vec<(usize, Vec<u16>)> = Vec::new();
        for m in M_SWEEP_2_TO_8_THE_VERIFY_ROW_RANGE {
            let mut d: CudaSlice<half::bf16> = stream.alloc_zeros::<half::bf16>(m * n).unwrap();
            let da_m: CudaSlice<u8> = stream.clone_htod(&ha[..m * k]).unwrap();
            runner
                .matmul_e4m3_weight_row(
                    &da_m,
                    &dw,
                    &mut d,
                    m as u64,
                    n as u64,
                    k as u64,
                    &scale_one,
                    &scale_one,
                )
                .unwrap_or_else(|e| panic!("Lt matmul n={n} k={k} m={m}: {e:#}"));
            stream.synchronize().unwrap();
            let out: Vec<half::bf16> = stream.clone_dtoh(&d).unwrap();
            outs.push((m, out.iter().map(|b| b.to_bits()).collect()));
        }
        let mut fam_line = String::new();
        for i in 0..outs.len() {
            let (mi, oi) = &outs[i];
            let fam = (0..i)
                .find(|&j| {
                    let (mj, oj) = &outs[j];
                    let shared = mi.min(mj) * n;
                    oi[..shared] == oj[..shared]
                })
                .map(|j| (b'A' + j as u8) as char)
                .unwrap_or((b'A' + i as u8) as char);
            fam_line.push_str(&format!("m{mi}:{fam} "));
        }
        eprintln!("[lt-tc-family] shared-rows bitwise family per m (n={n} k={k}): {fam_line}");
        let m4 = outs.iter().find(|(m, _)| *m == 4).unwrap().1.clone();
        for m in [7usize, 8] {
            let mut d_chunk: CudaSlice<half::bf16> =
                stream.alloc_zeros::<half::bf16>(m * n).unwrap();
            let da_m: CudaSlice<u8> = stream.clone_htod(&ha[..m * k]).unwrap();
            for start in [0usize, m - 4] {
                runner
                    .matmul_e4m3_weight_row_span(
                        &da_m,
                        &dw,
                        &mut d_chunk,
                        start as u64,
                        4,
                        n as u64,
                        k as u64,
                        &scale_one,
                        &scale_one,
                    )
                    .unwrap_or_else(|e| {
                        panic!("span start={start} rows=4 n={n} k={k} m={m}: {e:#}")
                    });
            }
            stream.synchronize().unwrap();
            let chunked: Vec<u16> = stream
                .clone_dtoh(&d_chunk)
                .unwrap()
                .iter()
                .map(|b| b.to_bits())
                .collect();
            assert_eq!(
                chunked[..4 * n],
                m4[..4 * n],
                "n={n} k={k} m={m}: rows 0..3 of the (0,4)+({},4) span composition are not \
                 bitwise the single m=4 launch; under NV_Q38_VERIFY_TC=1 this shape carries \
                 verify projections and the chunked fix rides on the head span being exactly \
                 the production mtp verify launch at EVERY shape the tile serves, not only the \
                 down_proj",
                m - 4
            );
            let mut d_planned: CudaSlice<half::bf16> =
                stream.alloc_zeros::<half::bf16>(m * n).unwrap();
            runner
                .matmul_e4m3_weight_row_span_with_plan_rows(
                    &da_m,
                    &dw,
                    &mut d_planned,
                    0,
                    m as u64,
                    4,
                    n as u64,
                    k as u64,
                    &scale_one,
                    &scale_one,
                )
                .unwrap_or_else(|e| panic!("planned m={m} plan_rows=4 n={n} k={k}: {e:#}"));
            stream.synchronize().unwrap();
            let planned: Vec<u16> = stream
                .clone_dtoh(&d_planned)
                .unwrap()
                .iter()
                .map(|b| b.to_bits())
                .collect();
            assert_eq!(
                planned,
                chunked,
                "n={n} k={k} m={m}: one m-row launch riding the m4-plan algo is not bitwise \
                 the (0,4)+({},4) span composition; the single-launch arm reclaims the \
                 composition's second full weight read only if it reproduces the composition \
                 bit for bit",
                m - 4
            );
        }
    }
}

const AN_OFFSET_SPAN_MUST_REPRODUCE_THE_ZERO_OFFSET_SPAN_ROW_FOR_ROW_OR_A_TAIL_OVERLAP_CANNOT_RIDE_THE_EMISSION_FAMILY: &str =
    "a (span, m-span) composition leaves verify rows past the head on an m=1 or m=2 heuristic \
     plan whose bitwise family differs from the {m4,m5,m6} family every emission-bearing row \
     of the mtp and suffix verifies rides, so a deep-row near-tie can flip what the round \
     emits against the suffix-off stream; the only way those rows can join the emission family \
     is a second family-member span ending at row m-1, which recomputes overlap rows and is \
     sound if and only if a span's output for a given activation row does not depend on the \
     row's position inside the launch or the pointer offsets -- probed here for the 6-row span \
     at offsets 1 and 2 and for the shipped 4-row span at the offset-3 overlap the m=7 \
     composition rewrites";

#[test]
fn cublaslt_e4m3_offset_spans_reproduce_zero_offset_rows_bitwise_at_every_trunk_shape_so_tail_overlaps_can_ride_the_emission_family(
) {
    let _ = AN_OFFSET_SPAN_MUST_REPRODUCE_THE_ZERO_OFFSET_SPAN_ROW_FOR_ROW_OR_A_TAIL_OVERLAP_CANNOT_RIDE_THE_EMISSION_FAMILY;
    let stream: Arc<CudaStream> = match CudaContext::new(0) {
        Ok(ctx) => ctx.default_stream(),
        Err(e) => panic!(
            "no CUDA device 0: this is the span position-invariance probe and must not report \
             success having executed nothing: {e}"
        ),
    };
    let mut runner = Fp8GemmRunner::new(stream.clone()).expect("Lt runner");
    let m_max = *M_SWEEP_2_TO_8_THE_VERIFY_ROW_RANGE.end();
    for (n, k) in QWEN38_E4M3_TRUNK_SHAPES_N_K {
        let mut st = 0x0ff5_e7a1_1a5e_d001u64 ^ ((n as u64) << 16) ^ k as u64;
        let ha: Vec<u8> = (0..m_max * k).map(|_| rnd_fp8_finite(&mut st)).collect();
        let hw: Vec<u8> = (0..n * k).map(|_| rnd_fp8_finite(&mut st)).collect();
        let da: CudaSlice<u8> = stream.clone_htod(&ha).unwrap();
        let dw: CudaSlice<u8> = stream.clone_htod(&hw).unwrap();
        let scale_one: CudaSlice<f32> = stream.clone_htod(&[0.01f32]).unwrap();
        let read_rows = |d: &CudaSlice<half::bf16>| -> Vec<u16> {
            stream
                .clone_dtoh(d)
                .unwrap()
                .iter()
                .map(|b| b.to_bits())
                .collect()
        };
        for (span, start) in [(6usize, 1usize), (6, 2), (4, 3)] {
            let mut d_zero: CudaSlice<half::bf16> =
                stream.alloc_zeros::<half::bf16>(m_max * n).unwrap();
            let mut d_off: CudaSlice<half::bf16> =
                stream.alloc_zeros::<half::bf16>(m_max * n).unwrap();
            runner
                .matmul_e4m3_weight_row_span(
                    &da,
                    &dw,
                    &mut d_zero,
                    0,
                    span as u64,
                    n as u64,
                    k as u64,
                    &scale_one,
                    &scale_one,
                )
                .unwrap_or_else(|e| panic!("zero-offset span-{span} n={n} k={k}: {e:#}"));
            runner
                .matmul_e4m3_weight_row_span(
                    &da,
                    &dw,
                    &mut d_off,
                    start as u64,
                    span as u64,
                    n as u64,
                    k as u64,
                    &scale_one,
                    &scale_one,
                )
                .unwrap_or_else(|e| panic!("offset-{start} span-{span} n={n} k={k}: {e:#}"));
            stream.synchronize().unwrap();
            let zero = read_rows(&d_zero);
            let off = read_rows(&d_off);
            let shared_rows = span.saturating_sub(start);
            let (a, b) = (
                &zero[start * n..(start + shared_rows) * n],
                &off[start * n..(start + shared_rows) * n],
            );
            let ndiff = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
            assert_eq!(
                ndiff,
                0,
                "n={n} k={k} span={span} offset={start}: rows {start}..{span} of the offset \
                 span differ from the zero-offset span in {ndiff} elements; the m{span} plan is \
                 not position-invariant here and a tail overlap span cannot join its family at \
                 this shape"
            );
        }
        let _ = &da;
    }
    eprintln!(
        "[lt-pos-invariance] offset 6-row and 4-row spans bitwise reproduce zero-offset rows \
         at every trunk shape"
    );
}

#[test]
fn cublaslt_e4m3_row0_training_m_ladder_family_census_and_serving_m1_skew_for_the_27b_projections()
{
    let stream: Arc<CudaStream> = match CudaContext::new(0) {
        Ok(ctx) => ctx.default_stream(),
        Err(e) => panic!(
            "no CUDA device 0: this is the train-vs-serve Lt plan family census and must not \
             report success having executed nothing: {e}"
        ),
    };
    let mut runner = Fp8GemmRunner::new(stream.clone()).expect("Lt runner");
    let m_max = *TRAIN_M_LADDER_M1_IS_SERVING.iter().max().unwrap();
    let mut by_shape: Vec<((usize, usize), Vec<String>)> = Vec::new();
    for (role, n, k) in Q38_27B_PROJECTIONS_ROLE_N_K {
        match by_shape.iter_mut().find(|(s, _)| *s == (n, k)) {
            Some((_, roles)) => roles.push(role.to_string()),
            None => by_shape.push(((n, k), vec![role.to_string()])),
        }
    }
    let mut table = String::new();
    for ((n, k), roles) in by_shape {
        let mut st = 0x1122_3344_5566_7788u64 ^ ((n as u64) << 20) ^ k as u64;
        let ha: Vec<u8> = (0..m_max * k).map(|_| rnd_fp8_finite(&mut st)).collect();
        let hw: Vec<u8> = (0..n * k).map(|_| rnd_fp8_finite(&mut st)).collect();
        let dw: CudaSlice<u8> = stream.clone_htod(&hw).unwrap();
        let scale_one: CudaSlice<f32> = stream.clone_htod(&[0.01f32]).unwrap();
        let mut rows0: Vec<(usize, Vec<u16>)> = Vec::new();
        for m in TRAIN_M_LADDER_M1_IS_SERVING {
            let mut d: CudaSlice<half::bf16> = stream.alloc_zeros::<half::bf16>(m * n).unwrap();
            let da_m: CudaSlice<u8> = stream.clone_htod(&ha[..m * k]).unwrap();
            runner
                .matmul_e4m3_weight_row(
                    &da_m,
                    &dw,
                    &mut d,
                    m as u64,
                    n as u64,
                    k as u64,
                    &scale_one,
                    &scale_one,
                )
                .unwrap_or_else(|e| panic!("Lt matmul n={n} k={k} m={m}: {e:#}"));
            stream.synchronize().unwrap();
            let out: Vec<half::bf16> = stream.clone_dtoh(&d).unwrap();
            rows0.push((m, out[..n].iter().map(|b| b.to_bits()).collect()));
        }
        let mut fams: Vec<usize> = Vec::new();
        for i in 0..rows0.len() {
            let fam = (0..i)
                .find(|&j| rows0[j].1 == rows0[i].1)
                .map(|j| fams[j])
                .unwrap_or_else(|| fams.iter().copied().max().map_or(0, |x| x + 1));
            fams.push(fam);
        }
        let fam_str = rows0
            .iter()
            .zip(&fams)
            .map(|((m, _), f)| format!("m{m}:{}", (b'A' + *f as u8) as char))
            .collect::<Vec<_>>()
            .join(" ");
        let base = &rows0[0].1;
        let base_maxabs = base
            .iter()
            .map(|b| half::bf16::from_bits(*b).to_f32().abs())
            .fold(0f32, f32::max);
        let mut skew = String::new();
        for (i, (m, r)) in rows0.iter().enumerate() {
            if i == 0 || fams[i] == fams[0] {
                continue;
            }
            let ndiff = r.iter().zip(base.iter()).filter(|(a, b)| a != b).count();
            let maxd = r
                .iter()
                .zip(base.iter())
                .map(|(a, b)| {
                    (half::bf16::from_bits(*a).to_f32() - half::bf16::from_bits(*b).to_f32()).abs()
                })
                .fold(0f32, f32::max);
            skew.push_str(&format!(" m{m}(ndiff={ndiff}/{n} maxabsD={maxd:.6})"));
        }
        table.push_str(&format!(
            "n={n} k={k} roles={} fams[{fam_str}] row0_maxabs={base_maxabs:.4} serving_m1_vs:{}\n",
            roles.join(","),
            if skew.is_empty() {
                " none-differ".to_string()
            } else {
                skew
            }
        ));
    }
    eprintln!("[lt-train-census] 27B projections, m ladder {TRAIN_M_LADDER_M1_IS_SERVING:?}, family = bitwise row0 class:\n{table}");
}
