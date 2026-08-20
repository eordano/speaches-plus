#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dispatch;
use crate::wgpu_backend::{compose, Result, WgpuError};
pub use crate::wgpu_backend::pack::{pack_u16_pairs as pack_x_bf16};

pub const WGSL: &str = include_str!("../../../wgsl/quant_gemv.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;
pub const TREE_ROWS_PER_GROUP: u32 = 8;
pub const SG_ROWS_PER_GROUP: u32 = 4;
pub const INT8_ENTRY: &str = "gemv_int8_rowscale";
pub const FP8_ENTRY: &str = "gemv_fp8_rowscale";
pub const MXFP4_ENTRY: &str = "gemv_mxfp4";
pub const INT8_SG_ENTRY: &str = "gemv_int8_sg";
pub const FP8_SG_ENTRY: &str = "gemv_fp8_sg";
pub const MXFP4_SG_ENTRY: &str = "gemv_mxfp4_sg";
pub const MXFP4_BLOCK: usize = 32;
pub const FP8_GROUP_ENTRY: &str = "gemv_fp8_group";
pub const INT8_GROUP_ENTRY: &str = "gemv_int8_group";
pub const FP8_GROUP_SG_ENTRY: &str = "gemv_fp8_group_sg";
pub const INT8_GROUP_SG_ENTRY: &str = "gemv_int8_group_sg";
pub const INT8_GROUP_GELU_ENTRY: &str = "gemv_int8_group_gelu";
pub const INT8_GROUP_GELU_SG_ENTRY: &str = "gemv_int8_group_gelu_sg";
pub const FP8_GROUP_GELU_ENTRY: &str = "gemv_fp8_group_gelu";
pub const FP8_GROUP_GELU_SG_ENTRY: &str = "gemv_fp8_group_gelu_sg";
pub const GROUP_ALIGN: usize = 16;
pub const ROW_GROUP_SHIFT: u32 = 31;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QFormat {
    E4m3,
    Int8,
}

impl QFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "e4m3" | "fp8" | "fp8e4m3" => Some(QFormat::E4m3),
            "int8" | "i8" | "s8" => Some(QFormat::Int8),
            _ => None,
        }
    }
    pub fn label(&self) -> &'static str {
        match self {
            QFormat::E4m3 => "e4m3",
            QFormat::Int8 => "int8",
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QuantGemvParams {
    pub n_rows: u32,
    pub k_elems: u32,
    pub groups_x: u32,
    pub group_shift: u32,
    pub scales_per_row: u32,
    pub pad1: u32,
    pub pad2: u32,
    pub pad3: u32,
}

pub fn scales_per_row(k: usize, group: usize) -> usize {
    k.checked_div(group).unwrap_or(1)
}

pub fn group_shift(group: usize) -> u32 {
    if group == 0 {
        ROW_GROUP_SHIFT
    } else {
        (group / GROUP_ALIGN).trailing_zeros()
    }
}

pub fn group_rule(k: usize, group: usize) -> Result<()> {
    shape_rule(k)?;
    if group == 0 {
        return Ok(());
    }
    let vecs = group / GROUP_ALIGN;
    if !group.is_multiple_of(GROUP_ALIGN)
        || !vecs.is_power_of_two()
        || !k.is_multiple_of(group)
        || group > k
    {
        return Err(WgpuError::Shape(format!(
            "quant_gemv group must be a multiple of {GROUP_ALIGN} with a power-of-two vector \
             count and divide K; got group={group} K={k}"
        )));
    }
    Ok(())
}

pub fn params_for(n: usize, k: usize, group: usize, groups_x: u32) -> QuantGemvParams {
    QuantGemvParams {
        n_rows: n as u32,
        k_elems: k as u32,
        groups_x,
        group_shift: group_shift(group),
        scales_per_row: scales_per_row(k, group) as u32,
        pad1: 0,
        pad2: 0,
        pad3: 0,
    }
}

pub fn source() -> String {
    compose(WGSL)
}

pub fn shape_rule(k: usize) -> Result<()> {
    if k == 0 || !k.is_multiple_of(MXFP4_BLOCK) {
        return Err(WgpuError::Shape(format!(
            "quant_gemv requires K>0 and K%{MXFP4_BLOCK}==0; got K={k}"
        )));
    }
    Ok(())
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn pack_bytes(src: &[u8]) -> Vec<u32> {
    let n_words = src.len().div_ceil(4);
    let mut out = vec![0u32; n_words];
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .max(1);
    let words_per = n_words.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (ci, chunk) in out.chunks_mut(words_per).enumerate() {
            let lo = ci * words_per * 4;
            let hi = (lo + chunk.len() * 4).min(src.len());
            let src_chunk = &src[lo..hi];
            sc.spawn(move || {
                for (w, c) in chunk.iter_mut().zip(src_chunk.chunks(4)) {
                    let mut v = 0u32;
                    for (i, b) in c.iter().enumerate() {
                        v |= (*b as u32) << (8 * i);
                    }
                    *w = v;
                }
            });
        }
    });
    out
}

fn row_amax(row: &[u16]) -> f32 {
    row.iter().fold(0f32, |a, b| {
        let v = bf16_to_f32(*b);
        if v.is_finite() {
            a.max(v.abs())
        } else {
            a
        }
    })
}

pub fn quantize_groups(
    w: &[u16],
    n: usize,
    k: usize,
    group: usize,
    fmt: QFormat,
) -> (Vec<u32>, Vec<f32>) {
    assert_eq!(w.len(), n * k);
    let g = if group == 0 { k } else { group };
    let per_row = k / g;
    let peak = match fmt {
        QFormat::E4m3 => 448.0f32,
        QFormat::Int8 => 127.0f32,
    };
    let mut bytes = vec![0u8; n * k];
    let mut scales = vec![0f32; n * per_row];

    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .min(n.max(1));
    let rows_per = n.div_ceil(threads);
    std::thread::scope(|sc| {
        let mut b_rest = bytes.as_mut_slice();
        let mut s_rest = scales.as_mut_slice();
        let mut r0 = 0usize;
        while r0 < n {
            let rows = rows_per.min(n - r0);
            let (b_chunk, b_next) = b_rest.split_at_mut(rows * k);
            let (s_chunk, s_next) = s_rest.split_at_mut(rows * per_row);
            b_rest = b_next;
            s_rest = s_next;
            let w_chunk = &w[r0 * k..(r0 + rows) * k];
            sc.spawn(move || {
                for rr in 0..rows {
                    for gi in 0..per_row {
                        let off = rr * k + gi * g;
                        let chunk = &w_chunk[off..off + g];
                        let amax = row_amax(chunk);
                        let (scale, inv) = if amax > 0.0 {
                            (amax / peak, peak / amax)
                        } else {
                            (0.0, 0.0)
                        };
                        s_chunk[rr * per_row + gi] = scale;
                        for (i, b) in chunk.iter().enumerate() {
                            let v = bf16_to_f32(*b) * inv;
                            b_chunk[off + i] = match fmt {
                                QFormat::E4m3 => super::kv_fp8::encode_e4m3(v),
                                QFormat::Int8 => v.round().clamp(-127.0, 127.0) as i8 as u8,
                            };
                        }
                    }
                }
            });
            r0 += rows;
        }
    });
    (pack_bytes(&bytes), scales)
}

pub fn quantize_rows_int8(w: &[u16], n: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    quantize_groups(w, n, k, 0, QFormat::Int8)
}

pub fn quantize_rows_fp8(w: &[u16], n: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    quantize_groups(w, n, k, 0, QFormat::E4m3)
}

pub fn cpu_gemv_groups(
    wq: &[u32],
    scales: &[f32],
    x: &[u16],
    n: usize,
    k: usize,
    group: usize,
    fmt: QFormat,
) -> Vec<f32> {
    let g = if group == 0 { k } else { group };
    let per_row = k / g;
    let mut y = vec![0f32; n];
    for (r, dst) in y.iter_mut().enumerate() {
        let mut acc = 0f32;
        for gi in 0..per_row {
            let mut d = 0f32;
            for i in 0..g {
                let idx = r * k + gi * g + i;
                let byte = ((wq[idx / 4] >> (8 * (idx % 4))) & 0xff) as u8;
                let v = match fmt {
                    QFormat::E4m3 => super::kv_fp8::decode_e4m3(byte),
                    QFormat::Int8 => (byte as i8) as f32,
                };
                d += v * bf16_to_f32(x[gi * g + i]);
            }
            acc += d * scales[r * per_row + gi];
        }
        *dst = acc;
    }
    y
}

fn run_rowscale(
    ctx: &WgpuContext,
    label: &str,
    entry: &str,
    wq: &[u32],
    row_scale: &[f32],
    x: &[u16],
    y: &mut [u16],
    n: usize,
    k: usize,
) -> Result<()> {
    shape_rule(k)?;
    dispatch::check_len("quant_gemv wq", wq.len(), n * k / 4)?;
    dispatch::check_len("quant_gemv row_scale", row_scale.len(), n)?;
    dispatch::check_len("quant_gemv x", x.len(), k)?;
    dispatch::check_len("quant_gemv y", y.len(), n)?;
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, TREE_ROWS_PER_GROUP);
    let params = params_for(n, k, 0, groups.0);
    let w_buf = dispatch::storage_from_slice(ctx, "quant-gemv-w", wq);
    let s_buf = dispatch::storage_from_slice(ctx, "quant-gemv-scale", row_scale);
    let x_buf = dispatch::storage_from_slice(ctx, "quant-gemv-x", &pack_x_bf16(x));
    let y_buf = dispatch::storage_zeroed(ctx, "quant-gemv-y", (n * 4) as u64);
    let p_buf = dispatch::uniform_from(ctx, "quant-gemv-params", &params);
    dispatch::run(
        ctx,
        label,
        &source(),
        entry,
        &[
            (0, &w_buf),
            (1, &s_buf),
            (2, &x_buf),
            (3, &y_buf),
            (4, &p_buf),
        ],
        groups,
    )?;
    let words: Vec<u32> = dispatch::read_back(ctx, &y_buf, n)?;
    for (dst, w) in y.iter_mut().zip(words.iter()) {
        *dst = (*w & 0xffff) as u16;
    }
    Ok(())
}

pub fn gemv_int8_bf16(
    ctx: &WgpuContext,
    wq: &[u32],
    row_scale: &[f32],
    x: &[u16],
    y: &mut [u16],
    n: usize,
    k: usize,
) -> Result<()> {
    run_rowscale(
        ctx,
        "quant-gemv-int8",
        INT8_ENTRY,
        wq,
        row_scale,
        x,
        y,
        n,
        k,
    )
}

pub fn gemv_fp8_bf16(
    ctx: &WgpuContext,
    wq: &[u32],
    row_scale: &[f32],
    x: &[u16],
    y: &mut [u16],
    n: usize,
    k: usize,
) -> Result<()> {
    run_rowscale(ctx, "quant-gemv-fp8", FP8_ENTRY, wq, row_scale, x, y, n, k)
}

pub fn gemv_group_bf16(
    ctx: &WgpuContext,
    wq: &[u32],
    scales: &[f32],
    x: &[u16],
    y: &mut [u16],
    n: usize,
    k: usize,
    group: usize,
    fmt: QFormat,
) -> Result<()> {
    group_rule(k, group)?;
    dispatch::check_len("quant_gemv wq", wq.len(), n * k / 4)?;
    dispatch::check_len(
        "quant_gemv scales",
        scales.len(),
        n * scales_per_row(k, group),
    )?;
    dispatch::check_len("quant_gemv x", x.len(), k)?;
    dispatch::check_len("quant_gemv y", y.len(), n)?;
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, TREE_ROWS_PER_GROUP);
    let params = params_for(n, k, group, groups.0);
    let entry = match fmt {
        QFormat::E4m3 => FP8_GROUP_ENTRY,
        QFormat::Int8 => INT8_GROUP_ENTRY,
    };
    let w_buf = dispatch::storage_from_slice(ctx, "quant-gemv-w", wq);
    let s_buf = dispatch::storage_from_slice(ctx, "quant-gemv-scale", scales);
    let x_buf = dispatch::storage_from_slice(ctx, "quant-gemv-x", &pack_x_bf16(x));
    let y_buf = dispatch::storage_zeroed(ctx, "quant-gemv-y", (n * 4) as u64);
    let p_buf = dispatch::uniform_from(ctx, "quant-gemv-params", &params);
    dispatch::run(
        ctx,
        "quant-gemv-group",
        &source(),
        entry,
        &[
            (0, &w_buf),
            (1, &s_buf),
            (2, &x_buf),
            (3, &y_buf),
            (4, &p_buf),
        ],
        groups,
    )?;
    let words: Vec<u32> = dispatch::read_back(ctx, &y_buf, n)?;
    for (dst, w) in y.iter_mut().zip(words.iter()) {
        *dst = (*w & 0xffff) as u16;
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GeluFold {
    Tree,

    Subgroup,
}

impl GeluFold {
    pub fn entry(self, fmt: QFormat) -> &'static str {
        match (self, fmt) {
            (Self::Tree, QFormat::Int8) => INT8_GROUP_GELU_ENTRY,
            (Self::Subgroup, QFormat::Int8) => INT8_GROUP_GELU_SG_ENTRY,
            (Self::Tree, QFormat::E4m3) => FP8_GROUP_GELU_ENTRY,
            (Self::Subgroup, QFormat::E4m3) => FP8_GROUP_GELU_SG_ENTRY,
        }
    }

    pub fn rows_per_group(self) -> u32 {
        match self {
            Self::Subgroup => SG_ROWS_PER_GROUP,
            Self::Tree => TREE_ROWS_PER_GROUP,
        }
    }
}

pub fn gemv_group_gelu_bf16(
    ctx: &WgpuContext,
    wq: &[u32],
    scales: &[f32],
    x: &[u16],
    y: &mut [u16],
    n: usize,
    k: usize,
    group: usize,
    fmt: QFormat,
    fold: GeluFold,
) -> Result<()> {
    group_rule(k, group)?;
    gelu_fold_rule(n)?;
    let inter = n / 2;
    dispatch::check_len("quant_gemv wq", wq.len(), n * k / 4)?;
    dispatch::check_len(
        "quant_gemv scales",
        scales.len(),
        n * scales_per_row(k, group),
    )?;
    dispatch::check_len("quant_gemv x", x.len(), k)?;
    dispatch::check_len("quant_gemv gelu y", y.len(), inter)?;
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, fold.rows_per_group());
    let params = params_for(n, k, group, groups.0);
    let out_words = inter.div_ceil(2);
    let w_buf = dispatch::storage_from_slice(ctx, "quant-gemv-w", wq);
    let s_buf = dispatch::storage_from_slice(ctx, "quant-gemv-scale", scales);
    let x_buf = dispatch::storage_from_slice(ctx, "quant-gemv-x", &pack_x_bf16(x));
    let y_buf = dispatch::storage_zeroed(ctx, "quant-gemv-gelu-y", (out_words * 4) as u64);
    let p_buf = dispatch::uniform_from(ctx, "quant-gemv-params", &params);
    dispatch::run(
        ctx,
        "quant-gemv-group-gelu",
        &source(),
        fold.entry(fmt),
        &[
            (0, &w_buf),
            (1, &s_buf),
            (2, &x_buf),
            (3, &y_buf),
            (4, &p_buf),
        ],
        groups,
    )?;
    let words: Vec<u32> = dispatch::read_back(ctx, &y_buf, out_words)?;
    for (i, dst) in y.iter_mut().enumerate() {
        *dst = ((words[i / 2] >> (16 * (i % 2))) & 0xffff) as u16;
    }
    Ok(())
}

pub fn gelu_fold_rule(n: usize) -> Result<()> {
    if n == 0 || !n.is_multiple_of(2) {
        return Err(WgpuError::Shape(format!(
            "quant_gemv gelu fold needs an even row count (gate rows then up rows); got n={n}"
        )));
    }
    Ok(())
}

pub fn gemv_mxfp4_bf16(
    ctx: &WgpuContext,
    packed: &[u32],
    scale_bytes: &[u8],
    x: &[u16],
    y: &mut [u16],
    n: usize,
    k: usize,
) -> Result<()> {
    shape_rule(k)?;
    dispatch::check_len("quant_gemv mxfp4 packed", packed.len(), n * k / 8)?;
    dispatch::check_len(
        "quant_gemv mxfp4 scales",
        scale_bytes.len(),
        n * k / MXFP4_BLOCK,
    )?;
    dispatch::check_len("quant_gemv x", x.len(), k)?;
    dispatch::check_len("quant_gemv y", y.len(), n)?;
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, TREE_ROWS_PER_GROUP);
    let params = params_for(n, k, 0, groups.0);
    let w_buf = dispatch::storage_from_slice(ctx, "quant-gemv-mx-w", packed);
    let s_buf = dispatch::storage_from_slice(ctx, "quant-gemv-mx-scale", &pack_bytes(scale_bytes));
    let x_buf = dispatch::storage_from_slice(ctx, "quant-gemv-x", &pack_x_bf16(x));
    let y_buf = dispatch::storage_zeroed(ctx, "quant-gemv-y", (n * 4) as u64);
    let p_buf = dispatch::uniform_from(ctx, "quant-gemv-params", &params);
    dispatch::run(
        ctx,
        "quant-gemv-mxfp4",
        &source(),
        MXFP4_ENTRY,
        &[
            (0, &w_buf),
            (2, &x_buf),
            (3, &y_buf),
            (4, &p_buf),
            (5, &s_buf),
        ],
        groups,
    )?;
    let words: Vec<u32> = dispatch::read_back(ctx, &y_buf, n)?;
    for (dst, w) in y.iter_mut().zip(words.iter()) {
        *dst = (*w & 0xffff) as u16;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_declares_every_entry_point() {
        for e in [
            INT8_ENTRY,
            FP8_ENTRY,
            MXFP4_ENTRY,
            INT8_SG_ENTRY,
            FP8_SG_ENTRY,
            MXFP4_SG_ENTRY,
            FP8_GROUP_ENTRY,
            INT8_GROUP_ENTRY,
            FP8_GROUP_SG_ENTRY,
            INT8_GROUP_SG_ENTRY,
            INT8_GROUP_GELU_ENTRY,
            INT8_GROUP_GELU_SG_ENTRY,
            FP8_GROUP_GELU_ENTRY,
            FP8_GROUP_GELU_SG_ENTRY,
        ] {
            assert!(WGSL.contains(&format!("fn {e}(")), "missing entry {e}");
        }
        assert!(source().contains("fn e4m3_decode("));
        assert!(source().contains("fn int8_decode("));
    }

    #[test]
    fn the_int8_text_the_nv_models_mutation_oracles_anchor_is_frozen() {
        for anchor in [
            "    acc = fma(int8_decode(word, 1u), bf16_hi(xw0), acc);",
            "        let d = qg_dot16_i8(qg_w4[wbase + v], qg_x4[2u * v], qg_x4[2u * v + 1u]);",
            "acc = fma(qg_row_scale[sbase + (v >> sh)], d, acc);\n    }\n    return acc;\n}\n\nfn qg_row_acc_mx",
            "acc = fma(qg_row_scale[sbase + (v >> sh)], d, acc);\n    }\n    return acc;\n}\n\nfn qg_group_acc_i8",
        ] {
            assert!(
                WGSL.contains(anchor),
                "frozen text moved: {anchor:?} -- graph_g4w_int8_epilogue_oracle and \
                 wgpu_fp8_epilogue in nv-models string-anchor mutation probes to it, and the g4w \
                 pk template entries route through qg_group_acc_i8"
            );
        }
    }

    #[test]
    fn the_folded_gelu_constants_match_the_standalone_pass() {
        let split = super::super::gelu_tanh_mul::WGSL;
        for c in ["0.7978845608028654", "0.044715", "10.0"] {
            assert!(split.contains(c), "gelu_tanh_mul.wgsl lost {c}");
            assert!(WGSL.contains(c), "quant_gemv.wgsl lost {c}");
        }
        for op in [
            "let mag = 0.5 * abs(gate) * (1.0 + t) * abs(up);",
            "let clamped = clamp(inner,",
            "select(nv_tanhf(clamped), inner, inner != inner)",
        ] {
            assert!(split.contains(op), "gelu_tanh_mul.wgsl lost `{op}`");
            assert!(WGSL.contains(op), "quant_gemv.wgsl lost `{op}`");
        }
    }

    #[test]
    fn the_gelu_fold_rejects_an_odd_row_count() {
        assert!(gelu_fold_rule(0).is_err());
        assert!(gelu_fold_rule(43007).is_err());
        assert!(gelu_fold_rule(43008).is_ok());
        assert_eq!(GeluFold::Tree.rows_per_group(), TREE_ROWS_PER_GROUP);
        assert_eq!(GeluFold::Subgroup.rows_per_group(), SG_ROWS_PER_GROUP);
    }

    #[test]
    fn every_format_and_reduction_pair_has_its_own_fold_entry() {
        let all = [
            GeluFold::Tree.entry(QFormat::Int8),
            GeluFold::Subgroup.entry(QFormat::Int8),
            GeluFold::Tree.entry(QFormat::E4m3),
            GeluFold::Subgroup.entry(QFormat::E4m3),
        ];
        assert_eq!(
            all,
            [
                INT8_GROUP_GELU_ENTRY,
                INT8_GROUP_GELU_SG_ENTRY,
                FP8_GROUP_GELU_ENTRY,
                FP8_GROUP_GELU_SG_ENTRY
            ]
        );
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a, b, "two fold variants share entry {a}");
            }
        }
        for (entry, acc) in [
            (INT8_GROUP_GELU_ENTRY, "qg_group_acc_i8"),
            (INT8_GROUP_GELU_SG_ENTRY, "qg_group_acc_i8"),
            (FP8_GROUP_GELU_ENTRY, "qg_group_acc_e4m3"),
            (FP8_GROUP_GELU_SG_ENTRY, "qg_group_acc_e4m3"),
        ] {
            let at = WGSL.find(&format!("fn {entry}(")).expect("entry");
            let body = &WGSL[at..];
            let end = body.find("\n}\n").unwrap_or(body.len());
            assert!(
                body[..end].contains(acc),
                "{entry} does not call {acc} -- it decodes the other format"
            );
        }
    }

    #[test]
    fn int8_quantizer_packs_low_byte_first() {
        let w: Vec<u16> = [1.0f32, -1.0, 0.5, 0.25]
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_bits())
            .collect();
        let (packed, scales) = quantize_rows_int8(&w, 1, 4);
        assert_eq!(packed.len(), 1);
        assert!((scales[0] - 1.0 / 127.0).abs() < 1e-9);
        assert_eq!(packed[0] & 0xff, 127);
        assert_eq!((packed[0] >> 8) & 0xff, (-127i8) as u8 as u32);
        assert_eq!((packed[0] >> 16) & 0xff, 64);
        assert_eq!((packed[0] >> 24) & 0xff, 32);
    }

    #[test]
    fn fp8_quantizer_hits_e4m3_max_at_row_amax() {
        let w: Vec<u16> = [2.0f32, -2.0, 1.0, 0.0]
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_bits())
            .collect();
        let (packed, scales) = quantize_rows_fp8(&w, 1, 4);
        assert!((scales[0] - 2.0 / 448.0).abs() < 1e-9);
        assert_eq!(packed[0] & 0xff, 0x7e);
        assert_eq!((packed[0] >> 8) & 0xff, 0xfe);
    }

    #[test]
    fn group_zero_reproduces_the_row_quantizers_bit_for_bit() {
        let w: Vec<u16> = (0..2048)
            .map(|i| half::bf16::from_f32(((i % 97) as f32 - 48.0) * 0.013).to_bits())
            .collect();
        let (a, sa) = quantize_rows_fp8(&w, 4, 512);
        let (b, sb) = quantize_groups(&w, 4, 512, 512, QFormat::E4m3);
        assert_eq!(a, b);
        assert_eq!(sa, sb);
        let (c, sc) = quantize_rows_int8(&w, 4, 512);
        let (d, sd) = quantize_groups(&w, 4, 512, 512, QFormat::Int8);
        assert_eq!(c, d);
        assert_eq!(sc, sd);
    }

    #[test]
    fn group_layout_params_and_rules() {
        assert_eq!(scales_per_row(8192, 0), 1);
        assert_eq!(scales_per_row(8192, 128), 64);
        assert_eq!(group_shift(0), ROW_GROUP_SHIFT);
        assert_eq!(group_shift(16), 0);
        assert_eq!(group_shift(128), 3);
        assert!(group_rule(5376, 0).is_ok());
        assert!(group_rule(5376, 128).is_ok());
        assert!(group_rule(5376, 64).is_ok());
        assert!(group_rule(8192, 96).is_err());
        assert!(group_rule(8192, 48).is_err());
        assert!(group_rule(5376, 1024).is_err());
        assert!(group_rule(5376, 256).is_ok());
        let p = params_for(16, 5376, 128, 2);
        assert_eq!(p.group_shift, 3);
        assert_eq!(p.scales_per_row, 42);
        let r = params_for(16, 5376, 0, 2);
        assert_eq!(r.scales_per_row, 1);
        assert_eq!(336u32 >> r.group_shift, 0);
    }

    #[test]
    fn group_scaling_shrinks_int8_error_far_more_than_e4m3() {
        let mut s = 12345u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f64 / 2147483648.0 - 0.5) as f32
        };
        let k = 8192usize;
        let w: Vec<u16> = (0..k)
            .map(|_| {
                let g: f32 = (0..12).map(|_| next()).sum();
                half::bf16::from_f32(g * 0.02).to_bits()
            })
            .collect();
        let err = |group: usize, fmt: QFormat| -> f64 {
            let (wq, sc) = quantize_groups(&w, 1, k, group, fmt);
            let g = if group == 0 { k } else { group };
            let per = k / g;
            let mut se = 0f64;
            let mut sw = 0f64;
            for i in 0..k {
                let byte = ((wq[i / 4] >> (8 * (i % 4))) & 0xff) as u8;
                let v = match fmt {
                    QFormat::E4m3 => super::super::kv_fp8::decode_e4m3(byte),
                    QFormat::Int8 => (byte as i8) as f32,
                } * sc[i / g.max(1)];
                let _ = per;
                let t = bf16_to_f32(w[i]);
                se += ((t - v) as f64).powi(2);
                sw += (t as f64).powi(2);
            }
            (se / sw).sqrt()
        };
        let f_row = err(0, QFormat::E4m3);
        let f_128 = err(128, QFormat::E4m3);
        let i_row = err(0, QFormat::Int8);
        let i_128 = err(128, QFormat::Int8);
        assert!(
            f_row / f_128 < 1.5,
            "e4m3 group scaling should barely help: {f_row:.4e} -> {f_128:.4e}"
        );
        assert!(
            i_row / i_128 > 1.25,
            "int8 group scaling should help: {i_row:.4e} -> {i_128:.4e}"
        );
        assert!(
            i_128 * 2.0 < f_128,
            "int8 group=128 must beat e4m3 group=128 by >2x: {i_128:.4e} vs {f_128:.4e}"
        );
    }

    #[test]
    fn shape_rule_requires_multiple_of_32() {
        assert!(shape_rule(5376).is_ok());
        assert!(shape_rule(8192).is_ok());
        assert!(shape_rule(48).is_err());
        assert!(shape_rule(0).is_err());
    }
}
