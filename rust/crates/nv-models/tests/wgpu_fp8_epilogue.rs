#![cfg(feature = "wgpu")]

mod common;
use common::ord;
use common::Split;
mod hub_snapshot;

use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::gemv_nvfp4;
use nv_kernels::wgpu_backend::kernels::kv_fp8::{decode_e4m3, encode_e4m3};
use nv_kernels::wgpu_backend::kernels::quant_gemv;
use nv_models::gemma4_wgpu as g4w;
use common::LcgOddSeedShift32GaussUnit as Lcg;

const HIDDEN: usize = 5376;
const Q_DIM_SLIDING: usize = 8192;
const KV_DIM_SLIDING: usize = 4096;
const Q_DIM_FULL: usize = 16384;
const KV_DIM_FULL: usize = 2048;

const WHY: &str = "g4w_gemv_fp8_pk / g4w_gemv_fp8_pk3 are the fp8 epilogues that actually ship in \
     gemma4_wgpu.rs. Until this suite existed their only mention outside that file was a string \
     inside an error message: every other fp8 test exercised the standalone gemv_fp8_bf16 kernel \
     at toy sizes (m=8, m=96) while the packed split-scatter epilogue went untested. The tolerance \
     here is ONE bf16 ulp per element against an f64 CPU reference, not an aggregate norm; an \
     aggregate norm on a bf16 output saturates at ~1.5e-3 (half a bf16 ulp) and cannot see an \
     epilogue defect at all.";

fn ctx_or_skip(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("{test}: {}", ctx.summary());
            let st = ctx.qualify();
            if !st.qualified {
                hub_snapshot::precondition_absent(
                    test,
                    &format!("wgpu adapter present but NOT qualified: {:?}", st.reason),
                    "run under rust/scripts/nvk.sh, which wires VK_ICD_FILENAMES and the \
                     vulkan-loader; a qualified adapter exists on this box",
                );
                return None;
            }
            Some(ctx)
        }
        Err(e) => {
            hub_snapshot::precondition_absent(
                test,
                &format!("no wgpu adapter: {e}"),
                "run under rust/scripts/nvk.sh, which wires VK_ICD_FILENAMES and prepends \
                 the store vulkan-loader to LD_LIBRARY_PATH",
            );
            None
        }
    }
}

fn code_table(rng: &mut Lcg) -> [u8; 256] {
    let mut t = [0u8; 256];
    for slot in t.iter_mut() {
        let mut c = encode_e4m3(rng.gauss() * 110.0);
        if c & 0x7f == 0x7f {
            c ^= 1;
        }
        *slot = c;
    }
    t
}

struct Case {
    label: &'static str,
    n: usize,
    k: usize,
    group: usize,
    wq: Vec<u32>,
    scales: Vec<f32>,
    x_packed: Vec<u32>,
    reference: Vec<f32>,
    magnitude: Vec<f32>,
}

impl Case {
    fn build(label: &'static str, n: usize, k: usize, group: usize, seed: u64) -> Self {
        assert!(
            n.is_multiple_of(8),
            "{label}: pk epilogues pack row pairs on an 8-row grid"
        );
        quant_gemv::group_rule(k, group).unwrap_or_else(|e| panic!("{label}: {e}"));
        let per_row = quant_gemv::scales_per_row(k, group);
        let g = if group == 0 { k } else { group };

        let mut rng = Lcg::new(seed);
        let tab = code_table(&mut rng);
        let words = n * k / 4;
        let mut wq = vec![0u32; words];
        for w in wq.iter_mut() {
            let r = rng.next_u32();
            *w = tab[(r & 0xff) as usize] as u32
                | ((tab[((r >> 8) & 0xff) as usize] as u32) << 8)
                | ((tab[((r >> 16) & 0xff) as usize] as u32) << 16)
                | ((tab[(r >> 24) as usize] as u32) << 24);
        }
        let scales: Vec<f32> = (0..n * per_row)
            .map(|_| (0.5 + rng.unit()) * (0.03 / 448.0))
            .collect();
        let x: Vec<u16> = (0..k)
            .map(|_| bf16::from_f32(rng.gauss() * 0.4).to_bits())
            .collect();
        let x_packed = quant_gemv::pack_x_bf16(&x);

        let dec: Vec<f64> = (0..256).map(|b| decode_e4m3(b as u8) as f64).collect();
        let xf: Vec<f64> = x
            .iter()
            .map(|b| f32::from_bits((*b as u32) << 16) as f64)
            .collect();
        let mut reference = vec![0f32; n];
        let mut magnitude = vec![0f32; n];
        for r in 0..n {
            let row = &wq[r * (k / 4)..(r + 1) * (k / 4)];
            let mut acc = 0f64;
            let mut mag = 0f64;
            for (wi, word) in row.iter().enumerate() {
                let base = 4 * wi;
                let s = scales[r * per_row + base / g] as f64;
                let w = *word;
                let t = [
                    dec[(w & 0xff) as usize] * xf[base],
                    dec[((w >> 8) & 0xff) as usize] * xf[base + 1],
                    dec[((w >> 16) & 0xff) as usize] * xf[base + 2],
                    dec[(w >> 24) as usize] * xf[base + 3],
                ];
                acc += (t[0] + t[1] + t[2] + t[3]) * s;
                mag += (t[0].abs() + t[1].abs() + t[2].abs() + t[3].abs()) * s.abs();
            }
            reference[r] = acc as f32;
            magnitude[r] = mag as f32;
        }
        Self {
            label,
            n,
            k,
            group,
            wq,
            scales,
            x_packed,
            reference,
            magnitude,
        }
    }

    fn params(&self, groups_x: u32) -> quant_gemv::QuantGemvParams {
        quant_gemv::params_for(self.n, self.k, self.group, groups_x)
    }
}

pub const NOISE_C: f32 = 16.0;

const FLOOR_DOC: &str = "NOISE FLOOR: the kernel accumulates in f32, so a row whose true dot \
     product sits far below the sum of the magnitudes of its own terms is not resolvable to one \
     bf16 ulp by ANY f32 kernel. Elements with |ref| <= 16*f32::EPSILON*sum|w_i x_i| are reported \
     as UNRESOLVABLE and gated on absolute error against that same floor instead. On the real \
     Gemma4 shapes the floor is ~1.9e-5 while one bf16 ulp of a typical row is ~5.4e-4, so the \
     floor NEVER relaxes a typical element -- it only excuses catastrophic cancellation.";

struct UlpReport {
    max_ulp: i64,
    over_zero: usize,
    over_one: usize,
    worst_row: usize,
    got: f32,
    want: f32,
    nonfinite: usize,
    unresolvable: usize,
    worst_floor_ratio: f32,
    worst_floor_row: usize,
}

fn ulp_report(got: &[u16], want: &[f32], mag: &[f32]) -> UlpReport {
    assert_eq!(got.len(), want.len());
    assert_eq!(got.len(), mag.len());
    let mut r = UlpReport {
        max_ulp: 0,
        over_zero: 0,
        over_one: 0,
        worst_row: 0,
        got: 0.0,
        want: 0.0,
        nonfinite: 0,
        unresolvable: 0,
        worst_floor_ratio: 0.0,
        worst_floor_row: 0,
    };
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let gf = f32::from_bits((*g as u32) << 16);
        if !gf.is_finite() {
            r.nonfinite += 1;
        }
        let floor = NOISE_C * f32::EPSILON * mag[i];
        if w.abs() <= floor {
            r.unresolvable += 1;
            let ratio = if floor > 0.0 {
                (gf - *w).abs() / floor
            } else {
                0.0
            };
            if ratio > r.worst_floor_ratio {
                r.worst_floor_ratio = ratio;
                r.worst_floor_row = i;
            }
            continue;
        }
        let d = (ord(*g) - ord(bf16::from_f32(*w).to_bits())).abs();
        if d > 0 {
            r.over_zero += 1;
        }
        if d > 1 {
            r.over_one += 1;
        }
        if d > r.max_ulp {
            r.max_ulp = d;
            r.worst_row = i;
            r.got = gf;
            r.want = *w;
        }
    }
    r
}

impl UlpReport {
    fn check(&self, who: &str) {
        assert_eq!(
            self.nonfinite, 0,
            "{who} produced non-finite outputs: {self}"
        );
        assert!(
            self.max_ulp <= 1,
            "{who} exceeds one bf16 ulp against the f64 CPU reference: {self}. {WHY}"
        );
        assert!(
            self.worst_floor_ratio <= 1.0,
            "{who} exceeds the f32 accumulation noise floor on a cancellation-limited row: {self}. \
             {FLOOR_DOC}"
        );
    }
    fn caught(&self) -> bool {
        self.max_ulp > 1 || self.worst_floor_ratio > 1.0 || self.nonfinite > 0
    }
}

impl std::fmt::Display for UlpReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max {} bf16 ulp, {} elems >0 ulp, {} elems >1 ulp, worst row {} got {:e} want {:e}, \
             {} non-finite, {} unresolvable (worst {:.3}x noise floor at row {})",
            self.max_ulp,
            self.over_zero,
            self.over_one,
            self.worst_row,
            self.got,
            self.want,
            self.nonfinite,
            self.unresolvable,
            self.worst_floor_ratio,
            self.worst_floor_row
        )
    }
}

fn unpack(words: &[u32], n: usize) -> Vec<u16> {
    let mut out = vec![0u16; n];
    for (i, dst) in out.iter_mut().enumerate() {
        *dst = ((words[i / 2] >> (16 * (i % 2))) & 0xffff) as u16;
    }
    out
}

fn run_pk(
    ctx: &WgpuContext,
    src: &str,
    entry: &str,
    c: &Case,
    sg: bool,
    word_off: usize,
) -> anyhow::Result<(Vec<u16>, Vec<u32>)> {
    run_pk_with_x(ctx, src, entry, c, sg, word_off, &c.x_packed)
}

fn run_pk_with_x(
    ctx: &WgpuContext,
    src: &str,
    entry: &str,
    c: &Case,
    sg: bool,
    word_off: usize,
    x_packed: &[u32],
) -> anyhow::Result<(Vec<u16>, Vec<u32>)> {
    let rows_per_group = g4w::fp8_pk_rows_per_group(sg);
    let grid = dispatch::workgroup_count_1d(ctx, c.n as u64, rows_per_group);
    let params = c.params(grid.0);
    let w = dispatch::storage_from_slice(ctx, "e-w", &c.wq);
    let s = dispatch::storage_from_slice(ctx, "e-s", &c.scales);
    let x = dispatch::storage_from_slice(ctx, "e-x", x_packed);
    let total_words = word_off + c.n / 2;
    let y = dispatch::storage_from_slice(ctx, "e-y", &vec![0xdead_beefu32; total_words]);
    let p = dispatch::uniform_from(ctx, "e-p", &params);
    let off = dispatch::uniform_from(ctx, "e-off", &[word_off as u32, 0u32, 0u32, 0u32]);
    let pipe = dispatch::compute_pipeline(ctx, "fp8-pk-probe", src, entry)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    dispatch::dispatch(
        ctx,
        &pipe,
        &[(0, &w), (1, &s), (2, &x), (3, &y), (4, &p), (30, &off)],
        grid,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let words: Vec<u32> =
        dispatch::read_back(ctx, &y, total_words).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((unpack(&words[word_off..], c.n), words[..word_off].to_vec()))
}

fn run_pk3(
    ctx: &WgpuContext,
    src: &str,
    entry: &str,
    c: &Case,
    sg: bool,
    sp: &Split,
) -> anyhow::Result<(Vec<u16>, Vec<u16>, Vec<u16>)> {
    run_pk3_with_x(ctx, src, entry, c, sg, sp, &c.x_packed)
}

fn run_pk3_with_x(
    ctx: &WgpuContext,
    src: &str,
    entry: &str,
    c: &Case,
    sg: bool,
    sp: &Split,
    x_packed: &[u32],
) -> anyhow::Result<(Vec<u16>, Vec<u16>, Vec<u16>)> {
    assert_eq!(sp.v_off, sp.q_rows + sp.kv_rows);
    assert_eq!(c.n, sp.q_rows + 2 * sp.kv_rows);
    let rows_per_group = g4w::fp8_pk_rows_per_group(sg);
    let grid = dispatch::workgroup_count_1d(ctx, c.n as u64, rows_per_group);
    let params = c.params(grid.0);
    let w = dispatch::storage_from_slice(ctx, "e-w", &c.wq);
    let s = dispatch::storage_from_slice(ctx, "e-s", &c.scales);
    let x = dispatch::storage_from_slice(ctx, "e-x", x_packed);
    let qb = dispatch::storage_from_slice(ctx, "e-q", &vec![0xdead_beefu32; sp.q_rows / 2]);
    let kb = dispatch::storage_from_slice(ctx, "e-k", &vec![0xdead_beefu32; sp.kv_rows / 2]);
    let vb = dispatch::storage_from_slice(ctx, "e-v", &vec![0xdead_beefu32; sp.kv_rows / 2]);
    let p = dispatch::uniform_from(ctx, "e-p", &params);
    let spb = dispatch::uniform_from(
        ctx,
        "e-sp",
        &[sp.q_rows as u32, sp.kv_rows as u32, sp.v_off as u32, 0u32],
    );
    let pipe = dispatch::compute_pipeline(ctx, "fp8-pk3-probe", src, entry)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    dispatch::dispatch(
        ctx,
        &pipe,
        &[
            (0, &w),
            (1, &s),
            (2, &x),
            (4, &p),
            (31, &qb),
            (32, &kb),
            (33, &vb),
            (34, &spb),
        ],
        grid,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let rb = |b: &wgpu::Buffer, n: usize| -> anyhow::Result<Vec<u32>> {
        dispatch::read_back(ctx, b, n).map_err(|e| anyhow::anyhow!("{e}"))
    };
    let qw = rb(&qb, sp.q_rows / 2)?;
    let kw = rb(&kb, sp.kv_rows / 2)?;
    let vw = rb(&vb, sp.kv_rows / 2)?;
    Ok((
        unpack(&qw, sp.q_rows),
        unpack(&kw, sp.kv_rows),
        unpack(&vw, sp.kv_rows),
    ))
}

fn stacked_xs(c: &Case, m: usize, seed: u64) -> Vec<u32> {
    let mut rng = Lcg::new(seed);
    let mut xs = c.x_packed.clone();
    for _ in 1..m {
        let x: Vec<u16> = (0..c.k)
            .map(|_| bf16::from_f32(rng.gauss() * 0.4).to_bits())
            .collect();
        xs.extend(quant_gemv::pack_x_bf16(&x));
    }
    xs
}

fn run_mk(
    ctx: &WgpuContext,
    src: &str,
    entry: &str,
    c: &Case,
    xs: &[u32],
    m: usize,
    sg_grid: bool,
) -> anyhow::Result<Vec<Vec<u16>>> {
    let rows_per_group = g4w::fp8_pk_rows_per_group(sg_grid);
    let grid = dispatch::workgroup_count_1d(ctx, c.n as u64, rows_per_group);
    let params = c.params(grid.0);
    let x_stride_words = (c.k / 2) as u32;
    let y_stride_words = c.n / 2;
    let w = dispatch::storage_from_slice(ctx, "e-w", &c.wq);
    let s = dispatch::storage_from_slice(ctx, "e-s", &c.scales);
    let x = dispatch::storage_from_slice(ctx, "e-x", xs);
    let y = dispatch::storage_from_slice(ctx, "e-y", &vec![0xdead_beefu32; m * y_stride_words]);
    let p = dispatch::uniform_from(ctx, "e-p", &params);
    let mkp = dispatch::uniform_from(
        ctx,
        "e-mkp",
        &[m as u32, x_stride_words, y_stride_words as u32, 0u32],
    );
    let pipe = dispatch::compute_pipeline(ctx, "q8-mk-probe", src, entry)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    dispatch::dispatch(
        ctx,
        &pipe,
        &[(0, &w), (1, &s), (2, &x), (3, &y), (4, &p), (35, &mkp)],
        grid,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let words: Vec<u32> =
        dispatch::read_back(ctx, &y, m * y_stride_words).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((0..m)
        .map(|t| unpack(&words[t * y_stride_words..(t + 1) * y_stride_words], c.n))
        .collect())
}

const MK_PK3_WHY: &str = "g4w_gemm_fp8_mk_pk3 and g4w_gemm_int8_mk_pk3 are the M-row prefill \
     twins of the fused q/k/v split scatter: one dispatch per prefill chunk per attention layer, \
     which is every QKV projection the model performs outside decode. The entry census recorded \
     g4w_gemm_fp8_mk_pk3 as having no gate, and the _pk twin next door was covered while it was \
     not -- mk_tree_twin_is_bitwise_identical_to_both_decode_epilogues drives \
     g4w_gemm_fp8_mk_pk/g4w_gemm_int8_mk_pk and stops there. The pk3 variants are where an index \
     error actually hides: _pk writes one contiguous row range, while _pk3 routes each row to one \
     of THREE destination buffers on two boundaries AND strides each by the token, so a bug lands \
     in the wrong tensor rather than at the wrong value. THE ORACLE IS NOT THE IMPLEMENTATION: the \
     decode pk3 epilogue that the mk twin is compared against is itself gated element by element \
     against an f64 host reference by pk3_split_scatter_matches_cpu_reference_at_gemma4_qkv_shapes \
     in this same file, so bitwise agreement with it inherits that reference. Every row of the M \
     block carries DIFFERENT activations -- with one shared row the token stride is unobservable \
     and the M-row kernel degenerates into M copies of a GEMV.";

const MK_PK3_MUTANT_ROWS: usize = 128;

const MK_PK3_MUTANT_ROWS_DOC: &str = "MK_PK3_MUTANT_ROWS=128: the mutants are token-stride, \
     activation-stride and segment-offset index errors, none of which depends on the row count, so \
     running them at the 20480-row Gemma4 QKV shape would re-upload a 110 MB weight buffer per \
     mutant per format to observe the same bit. The real shapes carry the parity assertions; the \
     small ones carry the proof that a disagreement would be seen.";

fn mk_pk3_mutants() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        (
            "mk-pk3-token-stride-collapsed-onto-token-0",
            "qg_y_q[1u * (qg_split_params.q_rows >> 1u) + (row >> 1u)] = word;",
            "qg_y_q[0u * (qg_split_params.q_rows >> 1u) + (row >> 1u)] = word;",
        ),
        (
            "mk-activation-stride-halved",
            "    let xs4 = g4w_mk_params.x_stride_words >> 2u;",
            "    let xs4 = g4w_mk_params.x_stride_words >> 3u;",
        ),
        (
            "mk-pk3-v-segment-offset-plus-one-pair",
            "                    let vr = row - qg_split_params.v_off;",
            "                    let vr = row - qg_split_params.v_off + 2u;",
        ),
    ]
}

#[allow(clippy::too_many_arguments)]
fn run_mk_pk3(
    ctx: &WgpuContext,
    src: &str,
    entry: &str,
    c: &Case,
    sp: &Split,
    xs: &[u32],
    m: usize,
    sg_grid: bool,
) -> anyhow::Result<Vec<(Vec<u16>, Vec<u16>, Vec<u16>)>> {
    assert_eq!(sp.v_off, sp.q_rows + sp.kv_rows);
    assert_eq!(c.n, sp.q_rows + 2 * sp.kv_rows);
    let rows_per_group = g4w::fp8_pk_rows_per_group(sg_grid);
    let grid = dispatch::workgroup_count_1d(ctx, c.n as u64, rows_per_group);
    let params = c.params(grid.0);
    let q_words = sp.q_rows / 2;
    let kv_words = sp.kv_rows / 2;
    let w = dispatch::storage_from_slice(ctx, "e-w", &c.wq);
    let s = dispatch::storage_from_slice(ctx, "e-s", &c.scales);
    let x = dispatch::storage_from_slice(ctx, "e-x", xs);
    let qb = dispatch::storage_from_slice(ctx, "e-q", &vec![0xdead_beefu32; m * q_words]);
    let kb = dispatch::storage_from_slice(ctx, "e-k", &vec![0xdead_beefu32; m * kv_words]);
    let vb = dispatch::storage_from_slice(ctx, "e-v", &vec![0xdead_beefu32; m * kv_words]);
    let p = dispatch::uniform_from(ctx, "e-p", &params);
    let spb = dispatch::uniform_from(
        ctx,
        "e-sp",
        &[sp.q_rows as u32, sp.kv_rows as u32, sp.v_off as u32, 0u32],
    );
    let mkp = dispatch::uniform_from(ctx, "e-mkp", &[m as u32, (c.k / 2) as u32, 0u32, 0u32]);
    let pipe = dispatch::compute_pipeline(ctx, "q8-mk-pk3-probe", src, entry)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    dispatch::dispatch(
        ctx,
        &pipe,
        &[
            (0, &w),
            (1, &s),
            (2, &x),
            (4, &p),
            (31, &qb),
            (32, &kb),
            (33, &vb),
            (34, &spb),
            (35, &mkp),
        ],
        grid,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let rb = |b: &wgpu::Buffer, n: usize| -> anyhow::Result<Vec<u32>> {
        dispatch::read_back(ctx, b, n).map_err(|e| anyhow::anyhow!("{e}"))
    };
    let qw = rb(&qb, m * q_words)?;
    let kw = rb(&kb, m * kv_words)?;
    let vw = rb(&vb, m * kv_words)?;
    Ok((0..m)
        .map(|t| {
            (
                unpack(&qw[t * q_words..(t + 1) * q_words], sp.q_rows),
                unpack(&kw[t * kv_words..(t + 1) * kv_words], sp.kv_rows),
                unpack(&vw[t * kv_words..(t + 1) * kv_words], sp.kv_rows),
            )
        })
        .collect())
}

#[test]
fn mk_pk3_split_scatter_is_bitwise_identical_to_the_decode_pk3_epilogue() {
    let Some(ctx) = ctx_or_skip("mk_pk3_twin_parity") else {
        return;
    };
    eprintln!("{MK_PK3_WHY}");
    let m = 5usize;
    let mk_src = g4w::mk_q8_shader_source(16);
    let tree_src = g4w::fp8_pk_shader_source(false);
    let mut mutant_cases = 0usize;
    for (label, q_rows, kv_rows, k, group) in [
        (
            "gemma4 qkv sliding",
            Q_DIM_SLIDING,
            KV_DIM_SLIDING,
            HIDDEN,
            0,
        ),
        (
            "gemma4 qkv full-attention",
            Q_DIM_FULL,
            KV_DIM_FULL,
            HIDDEN,
            0,
        ),
        ("grouped-128 small", 32usize, 16usize, 512usize, 128usize),
        ("odd row block", 24, 8, 256, 0),
    ] {
        let n = q_rows + 2 * kv_rows;
        let c = Case::build(
            label,
            n,
            k,
            group,
            0x00c0_dec0_u64 ^ ((n as u64) << 24) ^ k as u64,
        );
        let sp = Split {
            q_rows,
            kv_rows,
            v_off: q_rows + kv_rows,
        };
        let xs = stacked_xs(&c, m, 0xbeef ^ n as u64);
        let stride = k / 2;
        for (fmt, decode_entry, mk_entry) in [
            ("fp8", g4w::FP8_PK3_ENTRY, "g4w_gemm_fp8_mk_pk3"),
            ("int8", "g4w_gemv_int8_pk3", "g4w_gemm_int8_mk_pk3"),
        ] {
            let per_token: Vec<(Vec<u16>, Vec<u16>, Vec<u16>)> = (0..m)
                .map(|t| {
                    let x = &xs[t * stride..(t + 1) * stride];
                    run_pk3_with_x(ctx, &tree_src, decode_entry, &c, false, &sp, x)
                        .expect("decode pk3 dispatch")
                })
                .collect();
            for sg_grid in [false, true] {
                let mk = run_mk_pk3(ctx, &mk_src, mk_entry, &c, &sp, &xs, m, sg_grid)
                    .expect("mk pk3 dispatch");
                for (t, got) in mk.iter().enumerate() {
                    for (seg, g, w) in [
                        ("Q", &got.0, &per_token[t].0),
                        ("K", &got.1, &per_token[t].1),
                        ("V", &got.2, &per_token[t].2),
                    ] {
                        assert_eq!(
                            g, w,
                            "{label}/{fmt} token {t} segment {seg} (sg_grid={sg_grid}): \
                             {mk_entry} diverges bitwise from {decode_entry}. {MK_PK3_WHY}"
                        );
                    }
                }
            }
            if n <= MK_PK3_MUTANT_ROWS {
                for (mn, from, to) in mk_pk3_mutants() {
                    let bad = mutate(&mk_src, from, to);
                    let mk = run_mk_pk3(ctx, &bad, mk_entry, &c, &sp, &xs, m, false)
                        .unwrap_or_else(|e| panic!("{label}/{fmt} mutant {mn} failed: {e}"));
                    let caught = mk.iter().enumerate().any(|(t, got)| {
                        got.0 != per_token[t].0
                            || got.1 != per_token[t].1
                            || got.2 != per_token[t].2
                    });
                    assert!(
                        caught,
                        "{label}/{fmt} mk pk3 mutant {mn} was NOT caught. A split scatter a broken \
                         stride passes is not a gate. {MK_PK3_WHY}"
                    );
                    eprintln!("[{label}/{fmt}] MK PK3 MUTANT {mn}: caught");
                }
                mutant_cases += 1;
            }
            eprintln!(
                "mk_pk3_twin_parity {label}/{fmt}: mk({m} tokens) == decode pk3 bitwise on both \
                 grid sizings, q_rows={q_rows} kv_rows={kv_rows} k={k} group={group}"
            );
        }
    }
    assert!(
        mutant_cases >= 4,
        "only {mutant_cases} case/format pairs were small enough to run the mutants; the parity \
         assertions alone prove agreement, not that a disagreement would be visible, so at least \
         one small shape per format must stay in the corpus. {MK_PK3_MUTANT_ROWS_DOC}"
    );
}

#[test]
fn mk_tree_twin_is_bitwise_identical_to_both_decode_epilogues() {
    let Some(ctx) = ctx_or_skip("mk_twin_parity") else {
        return;
    };

    if !gemv_nvfp4::sg32_ok(ctx) {
        eprintln!("mk_twin_parity: SKIP no 32-wide subgroups; the sg epilogue never runs here");
        return;
    }
    let m = 5usize;
    let mk_src = g4w::mk_q8_shader_source(16);
    let tree_src = g4w::fp8_pk_shader_source(false);
    let sg_src = g4w::fp8_pk_shader_source(true);
    for (label, n, k, group) in [
        ("dense-per-row", 256usize, 1024usize, 0usize),
        ("grouped-128", 256, 1024, 128),
        ("odd-rows", 40, 512, 128),
        ("minimal", 8, 32, 0),
    ] {
        let c = Case::build(
            label,
            n,
            k,
            group,
            0x00c0_ffee ^ ((n as u64) << 32) ^ k as u64,
        );
        let xs = stacked_xs(&c, m, 0x5eed ^ n as u64);
        let stride = k / 2;
        for (fmt, decode_entry, mk_entry) in [
            ("fp8", g4w::FP8_PK_ENTRY, g4w::FP8_MK_PK_ENTRY),
            ("int8", g4w::INT8_PK_ENTRY, g4w::INT8_MK_PK_ENTRY),
        ] {
            let per_token: Vec<(Vec<u16>, Vec<u16>)> = (0..m)
                .map(|t| {
                    let x = &xs[t * stride..(t + 1) * stride];
                    let sg = run_pk_with_x(ctx, &sg_src, decode_entry, &c, true, 0, x)
                        .expect("sg dispatch")
                        .0;
                    let tree = run_pk_with_x(ctx, &tree_src, decode_entry, &c, false, 0, x)
                        .expect("tree dispatch")
                        .0;
                    (sg, tree)
                })
                .collect();
            for (t, (sg, tree)) in per_token.iter().enumerate() {
                assert_eq!(
                    sg, tree,
                    "{label}/{fmt} token {t}: the sg butterfly diverges bitwise from the tree reduction"
                );
            }
            for sg_grid in [false, true] {
                let mk = run_mk(ctx, &mk_src, mk_entry, &c, &xs, m, sg_grid).expect("mk dispatch");
                for (t, row) in mk.iter().enumerate() {
                    assert_eq!(
                        row, &per_token[t].0,
                        "{label}/{fmt} token {t} (sg_grid={sg_grid}): the tree mk twin diverges bitwise from the sg decode epilogue"
                    );
                }
            }
            eprintln!("mk_twin_parity {label}/{fmt}: sg==tree==mk({m} tokens) bitwise on both grid sizings");
        }
    }
}

fn paths(ctx: &WgpuContext) -> Vec<(&'static str, bool)> {
    let mut v = vec![("tree", false)];
    if gemv_nvfp4::sg32_ok(ctx) {
        v.push(("sg", true));
    } else {
        eprintln!("NOTE: no subgroup support, only the tree epilogue is covered on this adapter");
    }
    v
}

fn mutate(src: &str, from: &str, to: &str) -> String {
    assert!(
        src.contains(from),
        "mutation anchor not present in the shipped shader source: {from:?}. This suite is \
         worthless if its mutants no longer apply -- fix the anchors."
    );
    src.replace(from, to)
}

fn pk_mutants(sg: bool) -> Vec<(&'static str, &'static str, &'static str)> {
    let mut m = vec![(
        "row-scale-index-collapsed-to-row-0",
        "acc = fma(qg_row_scale[sbase + (v >> sh)], d, acc);\n    }\n    return acc;\n}\n\nfn qg_group_acc_i8",
        "acc = fma(qg_row_scale[0], d, acc);\n    }\n    return acc;\n}\n\nfn qg_group_acc_i8",
    )];
    if sg {
        m.push((
            "sg-pair-word-halves-swapped",
            "var word = qg_pk_rowbits[sgid];\n    if (row + 1u < qg_params.n_rows) {\n        word = word | (qg_pk_rowbits[sgid + 1u] << 16u);",
            "var word = qg_pk_rowbits[sgid] << 16u;\n    if (row + 1u < qg_params.n_rows) {\n        word = word | qg_pk_rowbits[sgid + 1u];",
        ));
    } else {
        m.push((
            "tree-pair-word-halves-swapped",
            "    return lo | (hi << 16u);\n}\n",
            "    return hi | (lo << 16u);\n}\n",
        ));
        m.push((
            "tree-hi-row-reads-its-own-lane-instead-of-the-next-warp",
            "hi = bf16_encode(qg_partial[tid + QG_LANES]) & 0xffffu;",
            "hi = bf16_encode(qg_partial[tid]) & 0xffffu;",
        ));
    }
    m
}

const LEGACY_WHY: &str = "g4w_gemv_legacy_pk and g4w_gemv_legacy_pk3 were the second of the two \
     NOT-REACHED entries the graph mutation sweep reported, and the entry census closed that half \
     of it: the arms below build them. Being built is not being gated. Until this suite asserted \
     on them, both legacy tests dispatched the legacy entries, computed a report, and printed it \
     -- the pk report was never checked and the pk3 test asserted nothing at all, so every \
     mutation of the legacy WGSL passed. THE BOUND IS THE SHIPPED ONE, and it is not a choice: at \
     group == 0 the legacy epilogue accumulates the SAME products as the shipped one, unscaled, \
     and applies the single row scale once at the end. That is the same f32 sum in a different \
     ORDER, and UlpReport's noise floor is precisely the allowance for order. So the legacy path \
     must meet the bound the shipped path meets, and a red here means the whole-row accumulator is \
     numerically worse than the per-16-element one -- a finding about a path that ships behind \
     set_attn_variant(legacy_epilogue = 1), not a bound to widen. These cases are group == 0 only: \
     with group scales the legacy epilogue is WRONG BY CONSTRUCTION, which is why \
     fp8_contract_freerun uses exactly that combination as a deliberate saboteur.";

fn legacy_mutants(sg: bool) -> Vec<(&'static str, &'static str, &'static str)> {
    if sg {
        vec![
            (
                "sg-legacy-row-scale-collapsed-to-row-0",
                "    let sc = qg_row_scale[select(0u, row, live)];",
                "    let sc = qg_row_scale[0];",
            ),
            (
                "sg-legacy-butterfly-dropped-so-only-one-lane-contributes",
                "    let total = qg_butterfly(raw);",
                "    let total = raw;",
            ),
        ]
    } else {
        vec![
            (
                "tree-legacy-row-scale-collapsed-to-row-0",
                "total * qg_row_scale[row]) & 0xffffu",
                "total * qg_row_scale[0]) & 0xffffu",
            ),
            (
                "tree-legacy-hi-row-reads-its-own-lane-instead-of-the-next-warp",
                "hi = bf16_encode(qg_partial[tid + QG_LANES] * qg_row_scale[row + 1u]) & 0xffffu;",
                "hi = bf16_encode(qg_partial[tid] * qg_row_scale[row + 1u]) & 0xffffu;",
            ),
        ]
    }
}

fn pk3_mutants() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        (
            "q-scatter-word-index-xor-1",
            "        qg_y_q[row >> 1u] = word;",
            "        qg_y_q[(row >> 1u) ^ 1u] = word;",
        ),
        (
            "k-scatter-stride-halved",
            "        qg_y_k[kr >> 1u] = word;",
            "        qg_y_k[kr >> 2u] = word;",
        ),
        (
            "v-scatter-row-offset-plus-one-pair",
            "        let vr = row - qg_split_params.v_off;",
            "        let vr = row - qg_split_params.v_off + 2u;",
        ),
    ]
}

fn check_pk(ctx: &WgpuContext, c: &Case, word_off: usize) {
    for (name, sg) in paths(ctx) {
        let src = g4w::fp8_pk_shader_source(sg);
        let (got, before) =
            run_pk(ctx, &src, g4w::FP8_PK_ENTRY, c, sg, word_off).expect("pk dispatch");
        let r = ulp_report(&got, &c.reference, &c.magnitude);
        eprintln!(
            "[{}][{name}] {} n={} k={} group={} off={word_off}: {r}",
            c.label,
            g4w::FP8_PK_ENTRY,
            c.n,
            c.k,
            c.group
        );
        r.check(&format!("[{}][{name}] {}", c.label, g4w::FP8_PK_ENTRY));
        assert!(
            before.iter().all(|w| *w == 0xdead_beef),
            "[{}][{name}] dst_word_off={word_off} was not honoured: the epilogue wrote below its \
             own offset",
            c.label
        );
        for (mn, from, to) in pk_mutants(sg) {
            let bad = mutate(&src, from, to);
            let (mg, _) =
                run_pk(ctx, &bad, g4w::FP8_PK_ENTRY, c, sg, word_off).unwrap_or_else(|e| {
                    panic!("[{}][{name}] mutant {mn} failed to dispatch: {e}", c.label)
                });
            let mr = ulp_report(&mg, &c.reference, &c.magnitude);
            eprintln!("[{}][{name}] MUTANT {mn}: {mr}", c.label);
            assert!(
                mr.caught(),
                "[{}][{name}] mutant {mn} was NOT caught: {mr}. A gate a broken epilogue passes is \
                 not a gate. {WHY}",
                c.label
            );
        }
    }
}

fn check_pk3(ctx: &WgpuContext, c: &Case, sp: &Split) {
    let want_q = &c.reference[..sp.q_rows];
    let want_k = &c.reference[sp.q_rows..sp.v_off];
    let want_v = &c.reference[sp.v_off..];
    let mag_q = &c.magnitude[..sp.q_rows];
    let mag_k = &c.magnitude[sp.q_rows..sp.v_off];
    let mag_v = &c.magnitude[sp.v_off..];
    for (name, sg) in paths(ctx) {
        let src = g4w::fp8_pk_shader_source(sg);
        let (q, k, v) = run_pk3(ctx, &src, g4w::FP8_PK3_ENTRY, c, sg, sp).expect("pk3 dispatch");
        let (rq, rk, rv) = (
            ulp_report(&q, want_q, mag_q),
            ulp_report(&k, want_k, mag_k),
            ulp_report(&v, want_v, mag_v),
        );
        eprintln!(
            "[{}][{name}] {} n={} k={} group={} q_rows={} kv_rows={} v_off={}\n    Q: {rq}\n    K: {rk}\n    V: {rv}",
            c.label, g4w::FP8_PK3_ENTRY, c.n, c.k, c.group, sp.q_rows, sp.kv_rows, sp.v_off
        );
        for (seg, r) in [("Q", &rq), ("K", &rk), ("V", &rv)] {
            r.check(&format!(
                "[{}][{name}] {} split-scatter segment {seg}",
                c.label,
                g4w::FP8_PK3_ENTRY
            ));
        }
        for (mn, from, to) in pk3_mutants() {
            let bad = mutate(&src, from, to);
            let (mq, mk, mv) =
                run_pk3(ctx, &bad, g4w::FP8_PK3_ENTRY, c, sg, sp).unwrap_or_else(|e| {
                    panic!("[{}][{name}] mutant {mn} failed to dispatch: {e}", c.label)
                });
            let mrs = [
                ("Q", ulp_report(&mq, want_q, mag_q)),
                ("K", ulp_report(&mk, want_k, mag_k)),
                ("V", ulp_report(&mv, want_v, mag_v)),
            ];
            let caught = mrs.iter().any(|(_, r)| r.caught());
            eprintln!(
                "[{}][{name}] MUTANT {mn}: Q {} K {} V {}",
                c.label, mrs[0].1, mrs[1].1, mrs[2].1
            );
            assert!(
                caught,
                "[{}][{name}] pk3 mutant {mn} was NOT caught by any of Q/K/V. The fused q/k/v \
                 split scatter is exactly where an index or stride error hides. {WHY}",
                c.label
            );
        }
    }
}

#[test]
fn the_shipped_fp8_epilogues_are_the_ones_under_test() {
    let tree = g4w::fp8_pk_shader_source(false);
    let sg = g4w::fp8_pk_shader_source(true);
    for src in [&tree, &sg] {
        for e in [
            g4w::FP8_PK_ENTRY,
            g4w::FP8_PK3_ENTRY,
            g4w::FP8_LEGACY_PK_ENTRY,
            g4w::FP8_LEGACY_PK3_ENTRY,
        ] {
            assert!(src.contains(&format!("fn {e}(")), "missing entry {e}");
        }
        assert!(src.contains("fn qg_scatter("));
        assert!(src.contains("fn qg_group_acc_e4m3("));
        assert!(src.contains("fn qg_row_acc_e4m3("));
    }
    assert!(tree.contains("@workgroup_size(256)"));
    assert!(sg.contains("@workgroup_size(128)"));
    assert_eq!(g4w::fp8_pk_rows_per_group(false), 8);
    assert_eq!(g4w::fp8_pk_rows_per_group(true), 4);
    eprintln!("{WHY}");
}

#[test]
fn every_mutation_anchor_still_applies_to_the_shipped_text() {
    let mk = g4w::mk_q8_shader_source(16);
    let mut checked = 0usize;
    for sg in [false, true] {
        let src = g4w::fp8_pk_shader_source(sg);
        for (name, from, _) in pk_mutants(sg)
            .into_iter()
            .chain(legacy_mutants(sg))
            .chain(pk3_mutants())
        {
            assert!(
                src.contains(from),
                "anchor for mutant {name} is gone from the sg={sg} shipped epilogue source: \
                 {from:?}. Every mutant in this file is silently inert the moment its anchor \
                 rots, and the GPU tests that would have caught it do not run on a box with no \
                 adapter -- which is why this check is CPU-only and unconditional."
            );
            checked += 1;
        }
    }
    for (name, from, _) in mk_pk3_mutants() {
        assert!(
            mk.contains(from),
            "anchor for mk pk3 mutant {name} is gone from mk_q8_shader_source(16): {from:?}"
        );
        checked += 1;
    }
    assert!(
        checked >= 12,
        "only {checked} anchors were checked; the mutant lists shrank and this guard is now \
         guarding less than the suite relies on"
    );
    eprintln!("all {checked} mutation anchors still apply to the shipped text");
}

#[test]
fn pk_epilogue_matches_cpu_reference_at_gemma4_o_proj_shapes() {
    let Some(ctx) = ctx_or_skip("pk_epilogue_o_proj") else {
        return;
    };
    let sliding = Case::build("o-proj sliding", HIDDEN, Q_DIM_SLIDING, 0, 0x51d1);
    check_pk(ctx, &sliding, 0);
    let full = Case::build("o-proj full-attention", HIDDEN, Q_DIM_FULL, 0, 0xf011);
    check_pk(ctx, &full, 24);
}

#[test]
fn pk_epilogue_matches_cpu_reference_with_group_scales() {
    let Some(ctx) = ctx_or_skip("pk_epilogue_group") else {
        return;
    };
    let c = Case::build(
        "o-proj sliding group=128",
        HIDDEN,
        Q_DIM_SLIDING,
        128,
        0x91a5,
    );
    assert_eq!(quant_gemv::scales_per_row(c.k, c.group), 64);
    check_pk(ctx, &c, 0);
}

#[test]
fn pk3_split_scatter_matches_cpu_reference_at_gemma4_qkv_shapes() {
    let Some(ctx) = ctx_or_skip("pk3_epilogue_qkv") else {
        return;
    };
    let n = Q_DIM_SLIDING + 2 * KV_DIM_SLIDING;
    let c = Case::build("qkv sliding", n, HIDDEN, 0, 0x9a71);
    check_pk3(
        ctx,
        &c,
        &Split {
            q_rows: Q_DIM_SLIDING,
            kv_rows: KV_DIM_SLIDING,
            v_off: Q_DIM_SLIDING + KV_DIM_SLIDING,
        },
    );
}

#[test]
fn pk3_split_scatter_matches_cpu_reference_at_the_full_attention_qkv_shape() {
    let Some(ctx) = ctx_or_skip("pk3_epilogue_qkv_full") else {
        return;
    };
    let n = Q_DIM_FULL + 2 * KV_DIM_FULL;
    let c = Case::build("qkv full-attention", n, HIDDEN, 0, 0xbeef);
    check_pk3(
        ctx,
        &c,
        &Split {
            q_rows: Q_DIM_FULL,
            kv_rows: KV_DIM_FULL,
            v_off: Q_DIM_FULL + KV_DIM_FULL,
        },
    );
}

pub const MECHANISM: &str = "MECHANISM (job 3). The 2026-08 fp8 fix swapped the epilogue entry \
     from g4w_gemv_legacy_pk/_pk3 (whole-row accumulator, row scale applied once at the end) to \
     g4w_gemv_fp8_pk/_pk3 (qg_group_acc_e4m3: a fresh 16-element partial `d`, then \
     acc = fma(scale, d, acc)). With the shipped group=0 configuration group_shift is 31, so \
     v >> sh is 0 for every v and BOTH forms multiply by exactly the same qg_row_scale[row]. The \
     only remaining difference is summation order and where the constant is applied -- and f32 \
     arithmetic is very nearly equivariant under multiplication by a constant. This test measures \
     both entry points against the same f64 reference on the real Gemma4 shapes and prints how \
     many outputs actually differ.";

fn bitwise_delta(a: &[u16], b: &[u16]) -> (usize, i64) {
    let mut differ = 0usize;
    let mut worst = 0i64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (ord(*x) - ord(*y)).abs();
        if d != 0 {
            differ += 1;
        }
        worst = worst.max(d);
    }
    (differ, worst)
}

#[test]
fn the_legacy_row_scale_epilogue_lands_on_the_same_bits_as_the_shipped_one() {
    let Some(ctx) = ctx_or_skip("legacy_vs_shipped") else {
        return;
    };
    eprintln!("{MECHANISM}");
    let cases = [
        Case::build("o-proj sliding", HIDDEN, Q_DIM_SLIDING, 0, 0x51d1),
        Case::build("o-proj full-attention", HIDDEN, Q_DIM_FULL, 0, 0xf011),
    ];
    let mut total_differ = 0usize;
    let mut total_elems = 0usize;
    for c in &cases {
        for (name, sg) in paths(ctx) {
            let src = g4w::fp8_pk_shader_source(sg);
            let (shipped, _) = run_pk(ctx, &src, g4w::FP8_PK_ENTRY, c, sg, 0).expect("shipped pk");
            let (legacy, _) =
                run_pk(ctx, &src, g4w::FP8_LEGACY_PK_ENTRY, c, sg, 0).expect("legacy pk");
            let rs = ulp_report(&shipped, &c.reference, &c.magnitude);
            let rl = ulp_report(&legacy, &c.reference, &c.magnitude);
            let (differ, worst) = bitwise_delta(&shipped, &legacy);
            eprintln!(
                "[{}][{name}] k={}\n    shipped g4w_gemv_fp8_pk    vs f64 reference: {rs}\n    \
                 legacy  g4w_gemv_legacy_pk vs f64 reference: {rl}\n    shipped vs legacy: \
                 {differ}/{} elements differ, max {worst} bf16 ulp",
                c.label,
                c.k,
                shipped.len()
            );
            rs.check(&format!("[{}][{name}] shipped", c.label));
            rl.check(&format!(
                "[{}][{name}] legacy {}. {LEGACY_WHY}",
                c.label,
                g4w::FP8_LEGACY_PK_ENTRY
            ));
            for (mn, from, to) in legacy_mutants(sg) {
                let bad = mutate(&src, from, to);
                let (mg, _) =
                    run_pk(ctx, &bad, g4w::FP8_LEGACY_PK_ENTRY, c, sg, 0).unwrap_or_else(|e| {
                        panic!(
                            "[{}][{name}] legacy mutant {mn} failed to dispatch: {e}",
                            c.label
                        )
                    });
                let mr = ulp_report(&mg, &c.reference, &c.magnitude);
                eprintln!("[{}][{name}] LEGACY MUTANT {mn}: {mr}", c.label);
                assert!(
                    mr.caught(),
                    "[{}][{name}] legacy mutant {mn} was NOT caught: {mr}. {LEGACY_WHY}",
                    c.label
                );
            }
            total_differ += differ;
            total_elems += shipped.len();
        }
    }
    eprintln!(
        "MECHANISM RESULT: across {total_elems} real-shape outputs the shipped per-16-element \
         epilogue and the legacy whole-row epilogue differ on {total_differ} of them. If that \
         count is 0 the accumulator change cannot be the mechanism behind the model-level swing, \
         and the fp8 story must be written down as empirically accepted rather than explained. \
         See docs/book/04.2-fp8-epilogue-mechanism.md."
    );
}

#[test]
fn the_legacy_pk3_split_scatter_lands_on_the_same_bits_as_the_shipped_one() {
    let Some(ctx) = ctx_or_skip("legacy_vs_shipped_pk3") else {
        return;
    };
    let n = Q_DIM_SLIDING + 2 * KV_DIM_SLIDING;
    let c = Case::build("qkv sliding", n, HIDDEN, 0, 0x9a71);
    let sp = Split {
        q_rows: Q_DIM_SLIDING,
        kv_rows: KV_DIM_SLIDING,
        v_off: Q_DIM_SLIDING + KV_DIM_SLIDING,
    };
    let want = |sp: &Split| {
        (
            c.reference[..sp.q_rows].to_vec(),
            c.reference[sp.q_rows..sp.v_off].to_vec(),
            c.reference[sp.v_off..].to_vec(),
        )
    };
    let mag = |sp: &Split| {
        (
            c.magnitude[..sp.q_rows].to_vec(),
            c.magnitude[sp.q_rows..sp.v_off].to_vec(),
            c.magnitude[sp.v_off..].to_vec(),
        )
    };
    let (wq, wk, wv) = want(&sp);
    let (mq, mk, mv) = mag(&sp);
    for (name, sg) in paths(ctx) {
        let src = g4w::fp8_pk_shader_source(sg);
        let a = run_pk3(ctx, &src, g4w::FP8_PK3_ENTRY, &c, sg, &sp).expect("shipped pk3");
        let b = run_pk3(ctx, &src, g4w::FP8_LEGACY_PK3_ENTRY, &c, sg, &sp).expect("legacy pk3");
        for (seg, x, y) in [("Q", &a.0, &b.0), ("K", &a.1, &b.1), ("V", &a.2, &b.2)] {
            let (differ, worst) = bitwise_delta(x, y);
            eprintln!(
                "[qkv sliding][{name}] {seg}: shipped vs legacy pk3 {differ}/{} elements differ, \
                 max {worst} bf16 ulp",
                x.len()
            );
        }
        for (seg, got, want, mag) in [
            ("Q", &b.0, &wq, &mq),
            ("K", &b.1, &wk, &mk),
            ("V", &b.2, &wv, &mv),
        ] {
            let r = ulp_report(got, want, mag);
            eprintln!("[qkv sliding][{name}] legacy pk3 {seg} vs f64 reference: {r}");
            r.check(&format!(
                "[qkv sliding][{name}] {} split-scatter segment {seg}. {LEGACY_WHY}",
                g4w::FP8_LEGACY_PK3_ENTRY
            ));
        }
        for (mn, from, to) in legacy_mutants(sg).into_iter().chain(pk3_mutants()) {
            let bad = mutate(&src, from, to);
            let (bq, bk, bv) = run_pk3(ctx, &bad, g4w::FP8_LEGACY_PK3_ENTRY, &c, sg, &sp)
                .unwrap_or_else(|e| panic!("[{name}] legacy pk3 mutant {mn} failed: {e}"));
            let rs = [
                ("Q", ulp_report(&bq, &wq, &mq)),
                ("K", ulp_report(&bk, &wk, &mk)),
                ("V", ulp_report(&bv, &wv, &mv)),
            ];
            eprintln!(
                "[qkv sliding][{name}] LEGACY PK3 MUTANT {mn}: Q {} K {} V {}",
                rs[0].1, rs[1].1, rs[2].1
            );
            assert!(
                rs.iter().any(|(_, r)| r.caught()),
                "[{name}] legacy pk3 mutant {mn} was NOT caught by any of Q/K/V. {LEGACY_WHY}"
            );
        }
    }
}
