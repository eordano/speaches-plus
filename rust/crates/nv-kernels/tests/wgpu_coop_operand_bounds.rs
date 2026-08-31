#![cfg(feature = "wgpu")]

use half::f16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::gemm_coop_f16 as coop;

mod common;
use common::ctx_or_panic as ctx;

const OPERAND_SLACK_IS_A_QUIET_NAN_SO_ANY_LANE_THAT_READS_PAST_M_OR_N_CANNOT_PRODUCE_A_FINITE_NUMBER:
    u16 = 0x7e00;

const THESE_SHAPES_ARE_WRITTEN_FOR_THE_16X16_FRAGMENT_THIS_ADAPTER_SELECTS: u32 = 16;

const RAGGED_BANDS: [(&str, u32, u32, u32, (u32, u32, u32, u32)); 5] = [
    ("rows48_over16__cols80_over48", 48, 80, 64, (4, 2, 2, 1)),
    ("rows16_over112__cols48_exact", 16, 48, 128, (8, 1, 1, 2)),
    ("rows112_over16__cols96_over32", 112, 96, 64, (2, 4, 2, 1)),
    ("rows64_exact__cols48_over16", 64, 48, 64, (4, 2, 2, 1)),
    ("rows64_exact__cols64_exact", 64, 64, 64, (4, 4, 4, 1)),
];

fn require_coop(c: &WgpuContext) -> coop::CoopGemm {
    match coop::select(c, coop::Operand::F16) {
        Ok(g) => {
            eprintln!("adapter: {}", c.summary());
            eprintln!("coop fragment: {}", g.request().label());
            g
        }
        Err(why) => panic!("cooperative-matrix GEMM unavailable on this adapter: {why}"),
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut r = common::LcgShift32TwoSided(seed);
    (0..len)
        .map(|_| f16::from_f32(r.next_f32() * 0.5).to_f32())
        .collect()
}

fn operand(
    c: &WgpuContext,
    label: &str,
    rows: usize,
    rows_alloc: usize,
    k: usize,
    vals: &[f32],
    pad: u16,
) -> wgpu::Buffer {
    let mut bits = vec![pad; rows_alloc * k];
    for (i, v) in vals.iter().take(rows * k).enumerate() {
        bits[i] = f16::from_f32(*v).to_bits();
    }
    dispatch::storage_from_slice(c, label, &bits)
}

fn run(
    c: &WgpuContext,
    g: coop::CoopGemm,
    m: u32,
    n: u32,
    k: u32,
    cfg: (u32, u32, u32, u32),
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
) -> Vec<f32> {
    let (tm, tn, sg, ku) = cfg;
    g.check_shape(m, n, k)
        .unwrap_or_else(|e| panic!("M={m} N={n} K={k}: {e}"));
    let src = g.source(tm, tn, sg, ku);
    let entry = g.entry(tm, tn, sg, ku);
    let pipeline = dispatch::compute_pipeline(c, "coop-bounds", &src, &entry)
        .unwrap_or_else(|e| panic!("pipeline tm{tm} tn{tn} sg{sg} ku{ku}: {e}"));
    let y = dispatch::storage_zeroed(c, "coop-bounds-y", (m as u64) * (n as u64) * 4);
    let zero = dispatch::storage_from_slice(c, "coop-bounds-zero", &vec![0f32; g.zero_elems()]);
    let (bm, bn) = g.grid(m, n, tm, tn, sg);
    let groups = dispatch::workgroup_count_1d(c, (bm as u64) * (bn as u64), 1);
    let p = coop::CoopGemmParams {
        n_rows: n,
        k_elems: k,
        m_rows: m,
        blocks_n: bn,
        y_stride: n,
        groups_x: groups.0,
        pad0: 0,
        pad1: 0,
    };
    let pbuf = dispatch::uniform_from(c, "coop-bounds-p", &p);
    let bind = dispatch::bind_group_offsets(
        c,
        &pipeline,
        &[(0, w, 0), (1, x, 0), (2, &y, 0), (3, &pbuf, 0), (4, &zero, 0)],
    );
    let mut enc = c.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
    }
    c.queue.submit([enc.finish()]);
    c.poll_blocking().unwrap();
    dispatch::read_back(c, &y, (m as usize) * (n as usize)).unwrap()
}

fn oracle_f64(x: &[f32], w: &[f32], m: u32, n: u32, k: u32) -> Vec<f64> {
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let mut o = vec![0f64; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut s = 0f64;
            for j in 0..k {
                s += x[mi * k + j] as f64 * w[ni * k + j] as f64;
            }
            o[mi * n + ni] = s;
        }
    }
    o
}

fn rel_rms(got: &[f32], want: &[f64]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (g, o) in got.iter().zip(want.iter()) {
        let d = *g as f64 - *o;
        num += d * d;
        den += o * o;
    }
    assert!(den > 0.0, "the oracle is all zeros; the comparison is vacuous");
    (num / den).sqrt()
}

fn bands(g: &coop::CoopGemm, m: u32, n: u32, cfg: (u32, u32, u32, u32)) -> (usize, usize) {
    let (tm, tn, sg, _) = cfg;
    let rows_pb = g.rows_per_block(tm) as usize;
    let cols = g.cols_per_workgroup(tn, sg) as usize;
    (
        (m as usize).next_multiple_of(rows_pb),
        (n as usize).next_multiple_of(cols),
    )
}

#[test]
fn the_stored_output_is_independent_of_what_lies_past_the_operand() {
    let c = ctx();
    let g = require_coop(c);
    assert_eq!(
        g.tile, THESE_SHAPES_ARE_WRITTEN_FOR_THE_16X16_FRAGMENT_THIS_ADAPTER_SELECTS,
        "these shapes are multiples of 16 only; an 8x8-only adapter needs its own table"
    );
    let _ = coop::A_ROW_AND_B_COL_BASES_ARE_CLAMPED_TO_THE_LAST_WHOLE_FRAGMENT_BECAUSE_THE_EPILOGUE_GUARDS_THE_STORE_AND_NEVER_THE_LOAD;
    for (label, m, n, k, cfg) in RAGGED_BANDS {
        let (rows_alloc, cols_alloc) = bands(&g, m, n, cfg);
        let xv = values((m as usize) * (k as usize), 0x51ee_d001);
        let wv = values((n as usize) * (k as usize), 0x51ee_d002);
        let mut got = Vec::new();
        for pad in [
            OPERAND_SLACK_IS_A_QUIET_NAN_SO_ANY_LANE_THAT_READS_PAST_M_OR_N_CANNOT_PRODUCE_A_FINITE_NUMBER,
            0u16,
        ] {
            let xb = operand(c, "cb-x", m as usize, rows_alloc, k as usize, &xv, pad);
            let wb = operand(c, "cb-w", n as usize, cols_alloc, k as usize, &wv, pad);
            got.push(run(c, g, m, n, k, cfg, &xb, &wb));
        }
        let d = fnv1a(bytemuck::cast_slice(&got[0]));
        eprintln!(
            "{label} m{m} n{n} k{k} cfg{cfg:?} alloc{rows_alloc}x{cols_alloc} digest {d:016x}"
        );
        let ndiff = got[0]
            .iter()
            .zip(got[1].iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            ndiff,
            0,
            "{label}: {ndiff} of {} f32 words move when the slack past m={m}/n={n} holds NaN \
             instead of zero. The stored fragments are exactly those the epilogue guard admits, \
             so either that guard is not what we think it is or a NaN is crossing lanes inside \
             coopMultiplyAdd -- both are worth more than the clamp this suite guards",
            got[0].len()
        );
        assert!(
            got[0].iter().all(|v| v.is_finite()),
            "{label}: a non-finite reached the stored output"
        );
        let err = rel_rms(&got[0], &oracle_f64(&xv, &wv, m, n, k));
        assert!(
            err < 5e-3,
            "{label}: rel_rms {err:.3e} against the f64 oracle; the clamped operand base must not \
             move any row or column a stored fragment reads"
        );
    }
}

#[test]
fn a_ragged_row_or_column_band_stays_inside_an_exactly_sized_operand() {
    let c = ctx();
    let g = require_coop(c);
    assert_eq!(g.tile, THESE_SHAPES_ARE_WRITTEN_FOR_THE_16X16_FRAGMENT_THIS_ADAPTER_SELECTS);
    for (label, m, n, k, cfg) in RAGGED_BANDS {
        let xv = values((m as usize) * (k as usize), 0x51ee_d001);
        let wv = values((n as usize) * (k as usize), 0x51ee_d002);
        let xb = operand(c, "cb-x-exact", m as usize, m as usize, k as usize, &xv, 0);
        let wb = operand(c, "cb-w-exact", n as usize, n as usize, k as usize, &wv, 0);
        let y = run(c, g, m, n, k, cfg, &xb, &wb);
        let err = rel_rms(&y, &oracle_f64(&xv, &wv, m, n, k));
        eprintln!("{label} exact-alloc rel_rms {err:.3e} digest {:016x}", fnv1a(bytemuck::cast_slice(&y)));
        assert!(
            err < 5e-3,
            "{label}: rel_rms {err:.3e} with x sized to exactly {m} rows and w to exactly {n}; a \
             consumer must not have to over-allocate for this kernel to be correct"
        );
    }
}

#[test]
fn every_global_operand_load_reads_from_a_clamped_band_base() {
    for tile in coop::TILES {
        let g = coop::CoopGemm::new(tile, coop::Operand::F16);
        for (tm, tn, sg, ku) in [(1u32, 4u32, 1u32, 1u32), (2, 8, 4, 1), (4, 4, 2, 2), (8, 1, 1, 4)]
        {
            let src = g.source(tm, tn, sg, ku);
            assert_eq!(
                src.matches("= min(m0 + ").count() as u32,
                tm,
                "{tile}x{tile} tm{tm} tn{tn} sg{sg} ku{ku}: one clamped A band base per row tile"
            );
            assert_eq!(
                src.matches("= min(n0 + ").count() as u32,
                tn,
                "{tile}x{tile} tm{tm} tn{tn} sg{sg} ku{ku}: one clamped B band base per column tile"
            );
            assert!(
                !src.contains("cg_x[(m0") && !src.contains("cg_w[(n0"),
                "{tile}x{tile} tm{tm} tn{tn} sg{sg} ku{ku}: an operand load still addresses the \
                 unclamped band base, so a ragged last block reads past the operand"
            );
            assert_eq!(
                src.matches("mrows - ").count() as u32,
                tm,
                "the A clamp must be expressed against m_rows, the same extent the store guard uses"
            );
            assert_eq!(src.matches("nrows - ").count() as u32, tn);
        }
    }
    for fmt in [coop::WqFmt::Nvfp4Block16, coop::WqFmt::Fp8RowscalePlain] {
        for (tm, tn, sg, ku) in [(1u32, 4u32, 1u32, 1u32), (4, 4, 2, 2), (8, 1, 1, 4)] {
            let src = coop::source_wq16(fmt, tm, tn, sg, ku);
            assert_eq!(
                src.matches("= min(m0 + ").count() as u32,
                tm,
                "{fmt:?} tm{tm} tn{tn} sg{sg} ku{ku}: the f16-activation arm loads A straight from \
                 cg_x and needs the same clamp the plain arm has"
            );
            assert!(!src.contains("cg_x[(m0"));
            for act in [coop::WqAct::Fp8Rowscale, coop::WqAct::Nvfp4Block16] {
                let staged = coop::source_wq16_act(fmt, act, tm, tn, sg, ku);
                assert_eq!(
                    staged.matches("= min(m0 + ").count(),
                    0,
                    "{fmt:?}x{act:?}: the staged activation arm already guards its own row loads \
                     with gm < mrows and must not grow a second clamp"
                );
            }
        }
    }
}

#[test]
fn the_clamped_base_leaves_a_whole_fragment_inside_the_operand() {
    for tile in coop::TILES {
        let g = coop::CoopGemm::new(tile, coop::Operand::F16);
        let t = tile as u64;
        for rows in [tile, tile * 3, tile * 7, 48, 112, 5120] {
            for tm in [1u32, 2, 4, 8, 16] {
                let rows_pb = g.rows_per_block(tm);
                for band in (0..rows.next_multiple_of(rows_pb)).step_by(tile as usize) {
                    let base = g.clamped_fragment_base_row(rows, band);
                    assert!(
                        base + tile <= rows,
                        "{tile}x{tile} rows={rows} tm={tm} band={band}: clamped base {base} still \
                         leaves the operand"
                    );
                    if band + tile <= rows {
                        assert_eq!(
                            base, band,
                            "{tile}x{tile} rows={rows} band={band}: the clamp must be the identity \
                             on every band the epilogue stores, which is what makes it bit-exact"
                        );
                    }
                    for k in [tile, 64, 5120] {
                        let last = base as u64 * k as u64
                            + (k - tile) as u64
                            + (t - 1) * k as u64
                            + (t - 1);
                        assert!(
                            last < rows as u64 * k as u64,
                            "{tile}x{tile} rows={rows} k={k} band={band}: the last element a \
                             fragment strides to is {last}, past the {} the operand holds",
                            rows as u64 * k as u64
                        );
                    }
                }
            }
        }
    }
}
