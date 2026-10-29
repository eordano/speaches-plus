use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dispatch;
use crate::wgpu_backend::pack::pack_u16_pairs as pack_u16;
use crate::wgpu_backend::{compose, Result, WgpuError};

pub const WGSL: &str = include_str!("../../../wgsl/gemv_w4a16.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;
pub const LANES_PER_ROW: u32 = 32;
pub const ROWS_PER_GROUP: u32 = 8;
pub const MAX_SHARED_K: usize = 3072;
pub const BLOCK_ENTRY: &str = "gemv_w4a16_block";
pub const ROW_ENTRY: &str = "gemv_w4a16_row";
pub const GELU_PLI_ENTRY: &str = "gemv_w4a16_gelu_pli";
pub const V4_ENTRY: &str = "gemv_w4a16_v4";
pub const SG_ENTRY: &str = "gemv_w4a16_sg_v4";
pub const SG_PK_ENTRY: &str = "gemv_w4a16_sg_pk";
pub const SG_PK3_ENTRY: &str = "gemv_w4a16_sg_pk3";
pub const SG_PKM_ENTRY: &str = "gemv_w4a16_sg_pkm";
pub const SG_PKM3_ENTRY: &str = "gemv_w4a16_sg_pkm3";
pub const SG_MK_PK_ENTRY: &str = "gemv_w4a16_sg_mk_pk";
pub const SG_MK_PK3_ENTRY: &str = "gemv_w4a16_sg_mk_pk3";
pub const SG_MK_MAX: u32 = 16;
pub const SG_PK_LANES: u32 = 16;
pub const SG_PK_WG: u32 = 256;
pub const SG_PK_ROWS: u32 = SG_PK_WG / SG_PK_LANES;
pub const V4_PACKED_SLOT: u32 = 6;
pub const V4_X_SLOT: u32 = 7;

pub const SG_LANE_MIN_WIDTH: u32 = 4;

pub fn sg_lane_width(
    subgroup: bool,
    min_size: u32,
    max_size: u32,
    probed: Option<u32>,
) -> Option<u32> {
    if !subgroup || max_size < min_size {
        return None;
    }
    let width = probed.unwrap_or(min_size);
    if width < SG_LANE_MIN_WIDTH || !width.is_power_of_two() {
        return None;
    }
    Some(width.min(LANES_PER_ROW))
}

pub fn sg_pk_supported(probed_width: Option<u32>) -> bool {
    matches!(probed_width, Some(w) if w >= SG_PK_LANES && w % SG_PK_LANES == 0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleGrain {
    Ge32,

    Ge32Fixed(u32),
    G16,
}

impl ScaleGrain {
    pub fn for_group_size(gs: usize) -> Option<Self> {
        match gs {
            16 => Some(Self::G16),
            g if g >= 32 && g.is_multiple_of(32) => Some(Self::Ge32),
            _ => None,
        }
    }

    pub fn fastest_for_group_size(gs: usize) -> Option<Self> {
        match gs {
            16 => Some(Self::G16),
            g if g >= 32 && g.is_power_of_two() => Some(Self::Ge32Fixed(g.trailing_zeros() - 5)),
            g => Self::for_group_size(g),
        }
    }

    pub fn accepts(self, gs: usize) -> bool {
        match self {
            Self::Ge32 => gs >= 32 && gs.is_multiple_of(32),
            Self::Ge32Fixed(shift) => gs == 32usize << shift,
            Self::G16 => gs == 16,
        }
    }
}

pub fn require_grain(grain: ScaleGrain, gs: usize) -> Result<()> {
    if grain.accepts(gs) {
        return Ok(());
    }
    Err(WgpuError::Shape(format!(
        "gemv_w4a16 kernel built for {grain:?} cannot express GS={gs}; \
         build it with ScaleGrain::fastest_for_group_size({gs})"
    )))
}

pub fn sg_pk_source_for(gs: usize) -> Result<(String, ScaleGrain)> {
    let grain = ScaleGrain::fastest_for_group_size(gs).ok_or_else(|| {
        WgpuError::Shape(format!(
            "gemv_w4a16 sg body has no grain for GS={gs}; needs 16 or a multiple of 32"
        ))
    })?;
    require_grain(grain, gs)?;
    Ok((sg_pk_source_grain(grain), grain))
}

pub fn g16_shape_rule(k: usize) -> Result<()> {
    shape_rule(k, 16)?;
    if !(k / 16).is_multiple_of(2) {
        return Err(WgpuError::Shape(format!(
            "gemv_w4a16 g16 needs an even scale row stride; K={k} gives {}",
            k / 16
        )));
    }
    Ok(())
}

fn sg_scale_step(grain: ScaleGrain) -> String {
    match grain {
        ScaleGrain::Ge32 => concat!(
            "        let sgi = sbase + ((v << 5u) / sgp_params.gs);\n",
            "        let ssw = sgp_scale[sgi >> 1u];\n",
            "        let sc = select(bf16_lo(ssw), bf16_hi(ssw), (sgi & 1u) == 1u);\n",
        )
        .to_string(),
        ScaleGrain::Ge32Fixed(shift) => {
            let idx = if shift == 0 {
                "v".to_string()
            } else {
                format!("(v >> {shift}u)")
            };
            format!(
                "        let sgi = sbase + {idx};\n        let ssw = sgp_scale[sgi >> 1u];\n        let sc = select(bf16_lo(ssw), bf16_hi(ssw), (sgi & 1u) == 1u);\n"
            )
        }
        ScaleGrain::G16 => concat!(
            "        let sgi = sbase + (v << 1u);\n",
            "        let ssw = sgp_scale[sgi >> 1u];\n",
            "        let sc_lo = bf16_lo(ssw);\n",
            "        let sc_hi = bf16_hi(ssw);\n",
        )
        .to_string(),
    }
}

fn sg_acc_step(grain: ScaleGrain, indent: &str, acc: &str, xb: &str) -> String {
    match grain {
        ScaleGrain::Ge32 | ScaleGrain::Ge32Fixed(_) => {
            format!("{indent}{acc} = fma(sc, sgp_dot32(wv, {xb}), {acc});\n")
        }
        ScaleGrain::G16 => format!(
            "{indent}let xb_{acc} = {xb};\n\
             {indent}var a0_{acc} = 0.0;\n\
             {indent}a0_{acc} = sgp_dot8(wv.x, sgp_x4[xb_{acc}], a0_{acc});\n\
             {indent}a0_{acc} = sgp_dot8(wv.y, sgp_x4[xb_{acc} + 1u], a0_{acc});\n\
             {indent}var a1_{acc} = 0.0;\n\
             {indent}a1_{acc} = sgp_dot8(wv.z, sgp_x4[xb_{acc} + 2u], a1_{acc});\n\
             {indent}a1_{acc} = sgp_dot8(wv.w, sgp_x4[xb_{acc} + 3u], a1_{acc});\n\
             {indent}{acc} = fma(sc_lo, a0_{acc}, {acc});\n\
             {indent}{acc} = fma(sc_hi, a1_{acc}, {acc});\n"
        ),
    }
}

const SG_PARAMS: &str = "struct GemvW4A16SgParams {
    n_rows: u32,
    k_elems: u32,
    gs: u32,
    w_row_words: u32,
    scale_row_stride: u32,
    groups_x: u32,
};

";

const SG_MK_BINDINGS: &str = "struct SgSplitParams {
    q_rows: u32,
    kv_rows: u32,
    v_off: u32,
    pad0: u32,
};

struct SgMkParams {
    m: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    dst_word_off: u32,
};

@group(0) @binding(1) var<storage, read> sgp_scale: array<u32>;
@group(0) @binding(3) var<storage, read_write> sgp_y: array<u32>;
@group(0) @binding(4) var<uniform> sgp_params: GemvW4A16SgParams;
@group(0) @binding(6) var<storage, read> sgp_packed4: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> sgp_x4: array<vec4<u32>>;
@group(0) @binding(31) var<storage, read_write> sgp_y_q: array<u32>;
@group(0) @binding(32) var<storage, read_write> sgp_y_k: array<u32>;
@group(0) @binding(33) var<storage, read_write> sgp_y_v: array<u32>;
@group(0) @binding(34) var<uniform> sgp_split: SgSplitParams;
@group(0) @binding(35) var<uniform> sgp_mk: SgMkParams;

";

const SGP_DOT: &str = "fn sgp_dot8(pv: u32, xw: vec4<u32>, acc_in: f32) -> f32 {
    let qe = vec4<f32>(unpack4xU8(pv & 0x0f0f0f0fu)) - vec4<f32>(8.0);
    let qo = vec4<f32>(unpack4xU8((pv >> 4u) & 0x0f0f0f0fu)) - vec4<f32>(8.0);
    let xe = bitcast<vec4<f32>>(xw << vec4<u32>(16u));
    let xo = bitcast<vec4<f32>>(xw & vec4<u32>(0xffff0000u));
    var s = acc_in;
    s = fma(qe.x, xe.x, s);
    s = fma(qo.x, xo.x, s);
    s = fma(qe.y, xe.y, s);
    s = fma(qo.y, xo.y, s);
    s = fma(qe.z, xe.z, s);
    s = fma(qo.z, xo.z, s);
    s = fma(qe.w, xe.w, s);
    s = fma(qo.w, xo.w, s);
    return s;
}

fn sgp_dot32(wv: vec4<u32>, xb: u32) -> f32 {
    var a = 0.0;
    a = sgp_dot8(wv.x, sgp_x4[xb], a);
    a = sgp_dot8(wv.y, sgp_x4[xb + 1u], a);
    a = sgp_dot8(wv.z, sgp_x4[xb + 2u], a);
    a = sgp_dot8(wv.w, sgp_x4[xb + 3u], a);
    return a;
}
";

pub fn sg_pk_source() -> String {
    sg_pk_source_grain(ScaleGrain::Ge32)
}

pub fn sg_pk_source_grain(grain: ScaleGrain) -> String {
    use std::fmt::Write as _;
    let x = SG_PK_LANES;
    let wg = SG_PK_WG;
    let rows = SG_PK_ROWS;
    assert!(x.is_power_of_two() && rows >= 2 && rows.is_multiple_of(2));
    let mut b = String::new();
    write!(
        b,
        "{SG_PARAMS}struct SgPkOffParams {{
    dst_word_off: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}};

struct SgSplitParams {{
    q_rows: u32,
    kv_rows: u32,
    v_off: u32,
    pad0: u32,
}};

@group(0) @binding(1) var<storage, read> sgp_scale: array<u32>;
@group(0) @binding(3) var<storage, read_write> sgp_y: array<u32>;
@group(0) @binding(4) var<uniform> sgp_params: GemvW4A16SgParams;
@group(0) @binding(6) var<storage, read> sgp_packed4: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> sgp_x4: array<vec4<u32>>;
@group(0) @binding(30) var<uniform> sgp_off: SgPkOffParams;
@group(0) @binding(31) var<storage, read_write> sgp_y_q: array<u32>;
@group(0) @binding(32) var<storage, read_write> sgp_y_k: array<u32>;
@group(0) @binding(33) var<storage, read_write> sgp_y_v: array<u32>;
@group(0) @binding(34) var<uniform> sgp_split: SgSplitParams;

var<workgroup> sgp_row_tot: array<f32, {rows}>;

{SGP_DOT}
fn sgp_row_acc(row: u32, live: bool, vlane: u32) -> f32 {{
    let kv = select(0u, sgp_params.k_elems >> 5u, live);
    let wbase4 = select(0u, row * (sgp_params.w_row_words >> 2u), live);
    let sbase = select(0u, row * sgp_params.scale_row_stride, live);
    var acc = 0.0;
    for (var v = vlane; v < kv; v = v + {x}u) {{\n"
    )
    .unwrap();
    b.push_str(&sg_scale_step(grain));
    b.push_str("        let wv = sgp_packed4[wbase4 + v];\n");
    b.push_str(&sg_acc_step(grain, "        ", "acc", "(v << 2u)"));
    b.push_str("    }\n");
    let mut s = x / 2;
    while s >= 1 {
        writeln!(b, "    acc = acc + subgroupShuffleXor(acc, {s}u);").unwrap();
        if s == 1 {
            break;
        }
        s /= 2;
    }
    b.push_str("    return acc;\n}\n\n");
    let header = |b: &mut String, entry: &str| {
        write!(
            b,
            "@compute @workgroup_size({wg})
fn {entry}(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_size) sgsz: u32,
    @builtin(subgroup_invocation_id) slane: u32
) {{
    let vt = sgid * sgsz + slane;
    let slot = vt / {x}u;\n"
        )
        .unwrap();
        writeln!(b, "    let vlane = slane & {}u;", x - 1).unwrap();
        write!(
            b,
            "    let row = (wid.x + wid.y * sgp_params.groups_x) * {rows}u + slot;
    let live = row < sgp_params.n_rows;
    let acc = sgp_row_acc(row, live, vlane);
    if (vlane == 0u) {{
        sgp_row_tot[slot] = acc;
    }}
    workgroupBarrier();
    if (vlane == 0u && live && (slot & 1u) == 0u) {{
        let lo = bf16_encode(acc) & 0xffffu;
        let hi_live = row + 1u < sgp_params.n_rows;
        let hi = bf16_encode(sgp_row_tot[slot + 1u]) & 0xffffu;
        let word = lo | (select(0u, hi, hi_live) << 16u);\n"
        )
        .unwrap();
    };
    header(&mut b, SG_PK_ENTRY);
    b.push_str(
        "        sgp_y[sgp_off.dst_word_off + (row >> 1u)] = word;
    }
}
\n",
    );
    header(&mut b, SG_PK3_ENTRY);
    b.push_str(
        "        if (row < sgp_split.q_rows) {
            sgp_y_q[row >> 1u] = word;
        } else {
            let kr = row - sgp_split.q_rows;
            if (kr < sgp_split.kv_rows) {
                sgp_y_k[kr >> 1u] = word;
            }
            if (row >= sgp_split.v_off) {
                let vr = row - sgp_split.v_off;
                if (vr < sgp_split.kv_rows) {
                    sgp_y_v[vr >> 1u] = word;
                }
            }
        }
    }
}\n",
    );
    compose(&b)
}

pub fn sg_mk_source(mk_max: u32) -> String {
    sg_mk_source_grain(mk_max, ScaleGrain::Ge32)
}

pub fn sg_mk_source_grain(mk_max: u32, grain: ScaleGrain) -> String {
    use std::fmt::Write as _;
    let x = SG_PK_LANES;
    let wg = SG_PK_WG;
    let rows = SG_PK_ROWS;
    assert!((1..=SG_MK_MAX).contains(&mk_max));
    let mut b = String::new();
    write!(
        b,
        "{SG_PARAMS}{SG_MK_BINDINGS}var<workgroup> sgp_row_tot: array<f32, {rows}>;

{SGP_DOT}
fn sgp_mk_row_acc(row: u32, live: bool, vlane: u32, accs: ptr<function, array<f32, {mk_max}>>) {{
    let kv = select(0u, sgp_params.k_elems >> 5u, live);
    let wbase4 = select(0u, row * (sgp_params.w_row_words >> 2u), live);
    let sbase = select(0u, row * sgp_params.scale_row_stride, live);
    let mm = sgp_mk.m;
    let xs4 = sgp_mk.x_stride_words >> 2u;
    for (var v = vlane; v < kv; v = v + {x}u) {{\n"
    )
    .unwrap();
    b.push_str(&sg_scale_step(grain));
    write!(
        b,
        "        let wv = sgp_packed4[wbase4 + v];
        for (var t = 0u; t < {mk_max}u; t = t + 1u) {{
            if (t < mm) {{\n"
    )
    .unwrap();
    match grain {
        ScaleGrain::Ge32 | ScaleGrain::Ge32Fixed(_) => b.push_str(
            "                (*accs)[t] = fma(sc, sgp_dot32(wv, t * xs4 + (v << 2u)), (*accs)[t]);\n",
        ),
        ScaleGrain::G16 => {
            b.push_str("                var acc_t = (*accs)[t];\n");
            b.push_str(&sg_acc_step(
                grain,
                "                ",
                "acc_t",
                "t * xs4 + (v << 2u)",
            ));
            b.push_str("                (*accs)[t] = acc_t;\n");
        }
    }
    b.push_str(
        "            }
        }
    }
}
\n",
    );
    let entry_fn = |b: &mut String, entry: &str, store: &str| {
        write!(
            b,
            "@compute @workgroup_size({wg})
fn {entry}(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_size) sgsz: u32,
    @builtin(subgroup_invocation_id) slane: u32
) {{
    let vt = sgid * sgsz + slane;
    let slot = vt / {x}u;\n"
        )
        .unwrap();
        writeln!(b, "    let vlane = slane & {}u;", x - 1).unwrap();
        write!(
            b,
            "    let row = wid.x * {rows}u + slot;
    let live = row < sgp_params.n_rows;
    var accs: array<f32, {mk_max}>;
    for (var t = 0u; t < {mk_max}u; t = t + 1u) {{
        accs[t] = 0.0;
    }}
    sgp_mk_row_acc(row, live, vlane, &accs);
    for (var t = 0u; t < {mk_max}u; t = t + 1u) {{
        if (t >= sgp_mk.m) {{
            break;
        }}
        var acc = accs[t];\n"
        )
        .unwrap();
        let mut s = x / 2;
        loop {
            writeln!(b, "        acc = acc + subgroupShuffleXor(acc, {s}u);").unwrap();
            if s == 1 {
                break;
            }
            s /= 2;
        }
        b.push_str(
            "        if (vlane == 0u) {
            sgp_row_tot[slot] = acc;
        }
        workgroupBarrier();
        if (vlane == 0u && live && (slot & 1u) == 0u) {
            let lo = bf16_encode(acc) & 0xffffu;
            let hi_live = row + 1u < sgp_params.n_rows;
            let hi = bf16_encode(sgp_row_tot[slot + 1u]) & 0xffffu;
            let word = lo | (select(0u, hi, hi_live) << 16u);\n",
        );
        b.push_str(store);
        b.push_str(
            "        }
        workgroupBarrier();
    }
}
\n",
        );
    };
    entry_fn(
        &mut b,
        SG_MK_PK_ENTRY,
        "            sgp_y[sgp_mk.dst_word_off + t * sgp_mk.y_stride_words + (row >> 1u)] = word;\n",
    );
    entry_fn(
        &mut b,
        SG_MK_PK3_ENTRY,
        concat!(
            "            if (row < sgp_split.q_rows) {\n",
            "                sgp_y_q[t * (sgp_split.q_rows >> 1u) + (row >> 1u)] = word;\n",
            "            } else {\n",
            "                let kr = row - sgp_split.q_rows;\n",
            "                if (kr < sgp_split.kv_rows) {\n",
            "                    sgp_y_k[t * (sgp_split.kv_rows >> 1u) + (kr >> 1u)] = word;\n",
            "                }\n",
            "                if (row >= sgp_split.v_off) {\n",
            "                    let vr = row - sgp_split.v_off;\n",
            "                    if (vr < sgp_split.kv_rows) {\n",
            "                        sgp_y_v[t * (sgp_split.kv_rows >> 1u) + (vr >> 1u)] = word;\n",
            "                    }\n",
            "                }\n",
            "            }\n",
        ),
    );
    compose(&b)
}

pub fn sg_mk_unrolled_source(mk_max: u32) -> String {
    sg_mk_unrolled_source_grain(mk_max, ScaleGrain::Ge32)
}

pub fn sg_mk_unrolled_source_grain(mk_max: u32, grain: ScaleGrain) -> String {
    use std::fmt::Write as _;
    let x = SG_PK_LANES;
    let wg = SG_PK_WG;
    let rows = SG_PK_ROWS;
    assert!((1..=SG_MK_MAX).contains(&mk_max));
    let mut b = String::new();
    write!(
        b,
        "{SG_PARAMS}{SG_MK_BINDINGS}var<workgroup> sgp_row_tot: array<f32, {rows}>;

{SGP_DOT}\n"
    )
    .unwrap();
    let entry_fn = |b: &mut String, entry: &str, store: &dyn Fn(u32) -> String| {
        write!(
            b,
            "@compute @workgroup_size({wg})
fn {entry}(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_size) sgsz: u32,
    @builtin(subgroup_invocation_id) slane: u32
) {{
    let vt = sgid * sgsz + slane;
    let slot = vt / {x}u;\n"
        )
        .unwrap();
        writeln!(b, "    let vlane = slane & {}u;", x - 1).unwrap();
        write!(
            b,
            "    let row = wid.x * {rows}u + slot;
    let live = row < sgp_params.n_rows;
    let kv = select(0u, sgp_params.k_elems >> 5u, live);
    let wbase4 = select(0u, row * (sgp_params.w_row_words >> 2u), live);
    let sbase = select(0u, row * sgp_params.scale_row_stride, live);
    let mm = sgp_mk.m;
    let xs4 = sgp_mk.x_stride_words >> 2u;\n"
        )
        .unwrap();
        for t in 0..mk_max {
            writeln!(b, "    var acc{t} = 0.0;").unwrap();
        }
        writeln!(b, "    for (var v = vlane; v < kv; v = v + {x}u) {{").unwrap();
        b.push_str(&sg_scale_step(grain));
        b.push_str("        let wv = sgp_packed4[wbase4 + v];\n");
        for t in 0..mk_max {
            let xb = if t == 0 {
                "(v << 2u)".to_string()
            } else {
                format!("{t}u * xs4 + (v << 2u)")
            };
            writeln!(b, "        if ({t}u < mm) {{").unwrap();
            b.push_str(&sg_acc_step(grain, "            ", &format!("acc{t}"), &xb));
            b.push_str("        }\n");
        }
        b.push_str("    }\n");
        for t in 0..mk_max {
            writeln!(
                b,
                "    if ({t}u < mm) {{
        var acc = acc{t};"
            )
            .unwrap();
            let mut s = x / 2;
            loop {
                writeln!(b, "        acc = acc + subgroupShuffleXor(acc, {s}u);").unwrap();
                if s == 1 {
                    break;
                }
                s /= 2;
            }
            b.push_str(
                "        if (vlane == 0u) {
            sgp_row_tot[slot] = acc;
        }
        workgroupBarrier();
        if (vlane == 0u && live && (slot & 1u) == 0u) {
            let lo = bf16_encode(acc) & 0xffffu;
            let hi_live = row + 1u < sgp_params.n_rows;
            let hi = bf16_encode(sgp_row_tot[slot + 1u]) & 0xffffu;
            let word = lo | (select(0u, hi, hi_live) << 16u);\n",
            );
            b.push_str(&store(t));
            b.push_str(
                "        }
        workgroupBarrier();
    }\n",
            );
        }
        b.push_str("}\n\n");
    };
    entry_fn(&mut b, SG_MK_PK_ENTRY, &|t| {
        format!(
            "            sgp_y[sgp_mk.dst_word_off + {t}u * sgp_mk.y_stride_words + (row >> 1u)] = word;\n"
        )
    });
    entry_fn(&mut b, SG_MK_PK3_ENTRY, &|t| {
        format!(
            concat!(
                "            if (row < sgp_split.q_rows) {{\n",
                "                sgp_y_q[{t}u * (sgp_split.q_rows >> 1u) + (row >> 1u)] = word;\n",
                "            }} else {{\n",
                "                let kr = row - sgp_split.q_rows;\n",
                "                if (kr < sgp_split.kv_rows) {{\n",
                "                    sgp_y_k[{t}u * (sgp_split.kv_rows >> 1u) + (kr >> 1u)] = word;\n",
                "                }}\n",
                "                if (row >= sgp_split.v_off) {{\n",
                "                    let vr = row - sgp_split.v_off;\n",
                "                    if (vr < sgp_split.kv_rows) {{\n",
                "                        sgp_y_v[{t}u * (sgp_split.kv_rows >> 1u) + (vr >> 1u)] = word;\n",
                "                    }}\n",
                "                }}\n",
                "            }}\n",
            ),
            t = t
        )
    });
    compose(&b)
}

pub fn sg_pk_mr_source(mr: u32) -> String {
    sg_pk_mr_source_grain(mr, ScaleGrain::Ge32)
}

pub fn sg_pk_mr_source_grain(mr: u32, grain: ScaleGrain) -> String {
    use std::fmt::Write as _;
    let x = SG_PK_LANES;
    let wg = SG_PK_WG;
    let slots = SG_PK_ROWS;
    assert!((2..=8).contains(&mr) && mr.is_multiple_of(2));
    let mut b = String::new();
    b.push_str(
        "struct GemvW4A16SgParams {
    n_rows: u32,
    k_elems: u32,
    gs: u32,
    w_row_words: u32,
    scale_row_stride: u32,
    groups_x: u32,
};

struct SgPkOffParams {
    dst_word_off: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct SgSplitParams {
    q_rows: u32,
    kv_rows: u32,
    v_off: u32,
    pad0: u32,
};

@group(0) @binding(1) var<storage, read> sgp_scale: array<u32>;
@group(0) @binding(3) var<storage, read_write> sgp_y: array<u32>;
@group(0) @binding(4) var<uniform> sgp_params: GemvW4A16SgParams;
@group(0) @binding(6) var<storage, read> sgp_packed4: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> sgp_x4: array<vec4<u32>>;
@group(0) @binding(30) var<uniform> sgp_off: SgPkOffParams;
@group(0) @binding(31) var<storage, read_write> sgp_y_q: array<u32>;
@group(0) @binding(32) var<storage, read_write> sgp_y_k: array<u32>;
@group(0) @binding(33) var<storage, read_write> sgp_y_v: array<u32>;
@group(0) @binding(34) var<uniform> sgp_split: SgSplitParams;

fn sgm_dot8(pv: u32, xe: vec4<f32>, xo: vec4<f32>, acc_in: f32) -> f32 {
    let qe = vec4<f32>(unpack4xU8(pv & 0x0f0f0f0fu)) - vec4<f32>(8.0);
    let qo = vec4<f32>(unpack4xU8((pv >> 4u) & 0x0f0f0f0fu)) - vec4<f32>(8.0);
    var s = acc_in;
    s = fma(qe.x, xe.x, s);
    s = fma(qo.x, xo.x, s);
    s = fma(qe.y, xe.y, s);
    s = fma(qo.y, xo.y, s);
    s = fma(qe.z, xe.z, s);
    s = fma(qo.z, xo.z, s);
    s = fma(qe.w, xe.w, s);
    s = fma(qo.w, xo.w, s);
    return s;
}
\n",
    );
    let header = |b: &mut String, entry: &str| {
        write!(
            b,
            "@compute @workgroup_size({wg})
fn {entry}(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_size) sgsz: u32,
    @builtin(subgroup_invocation_id) slane: u32
) {{
    let vt = sgid * sgsz + slane;
    let slot = vt / {x}u;\n"
        )
        .unwrap();
        writeln!(b, "    let vlane = slane & {}u;", x - 1).unwrap();
        write!(
            b,
            "    let row0 = ((wid.x + wid.y * sgp_params.groups_x) * {slots}u + slot) * {mr}u;
    let kv = select(0u, sgp_params.k_elems >> 5u, row0 < sgp_params.n_rows);
    var wbase4: array<u32, {mr}>;
    var sbase: array<u32, {mr}>;
    var accs: array<f32, {mr}>;
    for (var m = 0u; m < {mr}u; m = m + 1u) {{
        let r = row0 + m;
        let ok = r < sgp_params.n_rows;
        wbase4[m] = select(0u, r * (sgp_params.w_row_words >> 2u), ok);
        sbase[m] = select(0u, r * sgp_params.scale_row_stride, ok);
        accs[m] = 0.0;
    }}
    for (var v = vlane; v < kv; v = v + {x}u) {{
        let xb = v << 2u;
        let xw0 = sgp_x4[xb];
        let xw1 = sgp_x4[xb + 1u];
        let xw2 = sgp_x4[xb + 2u];
        let xw3 = sgp_x4[xb + 3u];
        let xe0 = bitcast<vec4<f32>>(xw0 << vec4<u32>(16u));
        let xo0 = bitcast<vec4<f32>>(xw0 & vec4<u32>(0xffff0000u));
        let xe1 = bitcast<vec4<f32>>(xw1 << vec4<u32>(16u));
        let xo1 = bitcast<vec4<f32>>(xw1 & vec4<u32>(0xffff0000u));
        let xe2 = bitcast<vec4<f32>>(xw2 << vec4<u32>(16u));
        let xo2 = bitcast<vec4<f32>>(xw2 & vec4<u32>(0xffff0000u));
        let xe3 = bitcast<vec4<f32>>(xw3 << vec4<u32>(16u));
        let xo3 = bitcast<vec4<f32>>(xw3 & vec4<u32>(0xffff0000u));\n"
        )
        .unwrap();
        match grain {
            ScaleGrain::Ge32 => b.push_str("        let sgo = (v << 5u) / sgp_params.gs;\n"),
            ScaleGrain::Ge32Fixed(0) => b.push_str("        let sgo = v;\n"),
            ScaleGrain::Ge32Fixed(s) => {
                writeln!(b, "        let sgo = v >> {s}u;").unwrap();
            }
            ScaleGrain::G16 => b.push_str("        let sgo = v << 1u;\n"),
        }
        write!(
            b,
            "        for (var m = 0u; m < {mr}u; m = m + 1u) {{
            let sgi = sbase[m] + sgo;
            let ssw = sgp_scale[sgi >> 1u];\n"
        )
        .unwrap();
        match grain {
            ScaleGrain::Ge32 | ScaleGrain::Ge32Fixed(_) => {
                b.push_str(
                    "            let sc = select(bf16_lo(ssw), bf16_hi(ssw), (sgi & 1u) == 1u);
            let wv = sgp_packed4[wbase4[m] + v];
            var a = 0.0;
            a = sgm_dot8(wv.x, xe0, xo0, a);
            a = sgm_dot8(wv.y, xe1, xo1, a);
            a = sgm_dot8(wv.z, xe2, xo2, a);
            a = sgm_dot8(wv.w, xe3, xo3, a);
            accs[m] = fma(sc, a, accs[m]);\n",
                );
            }
            ScaleGrain::G16 => {
                b.push_str(
                    "            let sc_lo = bf16_lo(ssw);
            let sc_hi = bf16_hi(ssw);
            let wv = sgp_packed4[wbase4[m] + v];
            var a0 = 0.0;
            a0 = sgm_dot8(wv.x, xe0, xo0, a0);
            a0 = sgm_dot8(wv.y, xe1, xo1, a0);
            var a1 = 0.0;
            a1 = sgm_dot8(wv.z, xe2, xo2, a1);
            a1 = sgm_dot8(wv.w, xe3, xo3, a1);
            accs[m] = fma(sc_lo, a0, accs[m]);
            accs[m] = fma(sc_hi, a1, accs[m]);\n",
                );
            }
        }
        write!(
            b,
            "        }}
    }}
    var tot: array<f32, {mr}>;
    for (var m = 0u; m < {mr}u; m = m + 1u) {{
        var acc = accs[m];\n"
        )
        .unwrap();
        let mut s = x / 2;
        while s >= 1 {
            writeln!(b, "        acc = acc + subgroupShuffleXor(acc, {s}u);").unwrap();
            if s == 1 {
                break;
            }
            s /= 2;
        }
        write!(
            b,
            "        tot[m] = acc;
    }}
    if (vlane == 0u) {{
        for (var m = 0u; m < {mr}u; m = m + 2u) {{
            let row = row0 + m;
            if (row < sgp_params.n_rows) {{
                let lo = bf16_encode(tot[m]) & 0xffffu;
                let hi_live = row + 1u < sgp_params.n_rows;
                let hi = bf16_encode(tot[m + 1u]) & 0xffffu;
                let word = lo | (select(0u, hi, hi_live) << 16u);\n"
        )
        .unwrap();
    };
    header(&mut b, SG_PKM_ENTRY);
    b.push_str(
        "                sgp_y[sgp_off.dst_word_off + (row >> 1u)] = word;
            }
        }
    }
}
\n",
    );
    header(&mut b, SG_PKM3_ENTRY);
    b.push_str(
        "                if (row < sgp_split.q_rows) {
                    sgp_y_q[row >> 1u] = word;
                } else {
                    let kr = row - sgp_split.q_rows;
                    if (kr < sgp_split.kv_rows) {
                        sgp_y_k[kr >> 1u] = word;
                    }
                    if (row >= sgp_split.v_off) {
                        let vr = row - sgp_split.v_off;
                        if (vr < sgp_split.kv_rows) {
                            sgp_y_v[vr >> 1u] = word;
                        }
                    }
                }
            }
        }
    }
}\n",
    );
    compose(&b)
}

pub fn sg_source(x: u32, wg: u32) -> String {
    use std::fmt::Write as _;
    assert!(x.is_power_of_two() && (4..=32).contains(&x));
    assert!(wg.is_multiple_of(x) && wg >= x);
    let rows = wg / x;
    let mut b = String::new();
    write!(
        b,
        "{SG_PARAMS}@group(0) @binding(0) var<storage, read> sgw_packed4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> sgw_scale: array<u32>;
@group(0) @binding(2) var<storage, read> sgw_x4: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> sgw_y: array<u32>;
@group(0) @binding(4) var<uniform> sgw_params: GemvW4A16SgParams;

fn sgw_dot8(pv: u32, xw: vec4<u32>, acc_in: f32) -> f32 {{
    let qe = vec4<f32>(unpack4xU8(pv & 0x0f0f0f0fu)) - vec4<f32>(8.0);
    let qo = vec4<f32>(unpack4xU8((pv >> 4u) & 0x0f0f0f0fu)) - vec4<f32>(8.0);
    let xe = bitcast<vec4<f32>>(xw << vec4<u32>(16u));
    let xo = bitcast<vec4<f32>>(xw & vec4<u32>(0xffff0000u));
    var s = acc_in;
    s = fma(qe.x, xe.x, s);
    s = fma(qo.x, xo.x, s);
    s = fma(qe.y, xe.y, s);
    s = fma(qo.y, xo.y, s);
    s = fma(qe.z, xe.z, s);
    s = fma(qo.z, xo.z, s);
    s = fma(qe.w, xe.w, s);
    s = fma(qo.w, xo.w, s);
    return s;
}}

fn sgw_dot32(wv: vec4<u32>, xb: u32) -> f32 {{
    var a = 0.0;
    a = sgw_dot8(wv.x, sgw_x4[xb], a);
    a = sgw_dot8(wv.y, sgw_x4[xb + 1u], a);
    a = sgw_dot8(wv.z, sgw_x4[xb + 2u], a);
    a = sgw_dot8(wv.w, sgw_x4[xb + 3u], a);
    return a;
}}

@compute @workgroup_size({wg})
fn {SG_ENTRY}(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_size) sgsz: u32,
    @builtin(subgroup_invocation_id) slane: u32
) {{
    let vt = sgid * sgsz + slane;
    let slot = vt / {x}u;\n"
    )
    .unwrap();
    writeln!(b, "    let vlane = slane & {}u;", x - 1).unwrap();
    write!(
        b,
        "    let row = (wid.x + wid.y * sgw_params.groups_x) * {rows}u + slot;
    let live = row < sgw_params.n_rows;
    let kv = select(0u, sgw_params.k_elems >> 5u, live);
    let wbase4 = select(0u, row * (sgw_params.w_row_words >> 2u), live);
    let sbase = select(0u, row * sgw_params.scale_row_stride, live);
    var acc = 0.0;
    for (var v = vlane; v < kv; v = v + {x}u) {{
        let sc = bf16_decode(sgw_scale[sbase + ((v << 5u) / sgw_params.gs)]);
        acc = fma(sc, sgw_dot32(sgw_packed4[wbase4 + v], v << 2u), acc);
    }}\n"
    )
    .unwrap();
    let mut s = x / 2;
    while s >= 1 {
        writeln!(b, "    acc = acc + subgroupShuffleXor(acc, {s}u);").unwrap();
        if s == 1 {
            break;
        }
        s /= 2;
    }
    b.push_str(
        "    if (vlane == 0u && live) {\n        sgw_y[row] = bf16_encode(acc);\n    }\n}\n",
    );
    compose(&b)
}

const SCRATCH_BYTES: u32 = WORKGROUP_SIZE * 4 + 8 * 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvW4A16Params {
    n_rows: u32,
    k_elems: u32,
    gs: u32,
    w_row_words: u32,
    scale_row_stride: u32,
    groups_x: u32,
}

pub fn entry_for(k: usize) -> &'static str {
    if k <= MAX_SHARED_K {
        BLOCK_ENTRY
    } else {
        ROW_ENTRY
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W4Route {
    Block,
    V4,
    Sg16,
}

pub const V4_ASPECT: usize = 4;

pub fn prefers_v4(n: usize, k: usize) -> bool {
    n != 0 && k / V4_ASPECT >= n
}

pub fn w4_route(n: usize, k: usize, gs: usize, sg16_ok: bool, force_block: bool) -> W4Route {
    let _ = (n, k);
    if force_block || !ScaleGrain::Ge32.accepts(gs) {
        W4Route::Block
    } else if sg16_ok {
        W4Route::Sg16
    } else {
        W4Route::V4
    }
}

pub fn w4_route_grain(
    n: usize,
    k: usize,
    gs: usize,
    sg16_ok: bool,
    force_block: bool,
) -> Option<(W4Route, ScaleGrain)> {
    let grain = ScaleGrain::fastest_for_group_size(gs)?;
    if force_block {
        return Some((W4Route::Block, grain));
    }
    match grain {
        ScaleGrain::Ge32 | ScaleGrain::Ge32Fixed(_) => {
            Some((w4_route(n, k, gs, sg16_ok, false), grain))
        }
        ScaleGrain::G16 if sg16_ok => Some((W4Route::Sg16, grain)),
        ScaleGrain::G16 => Some((W4Route::Block, grain)),
    }
}

pub fn route_rows_per_group(route: W4Route) -> u32 {
    match route {
        W4Route::Sg16 => SG_PK_ROWS,
        W4Route::Block | W4Route::V4 => ROWS_PER_GROUP,
    }
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup_and_scratch(ctx, "gemv_w4a16", WORKGROUP_SIZE, SCRATCH_BYTES)
}

fn check_binding(ctx: &WgpuContext, what: &str, bytes: u64) -> Result<()> {
    if bytes > ctx.caps.max_storage_buffer_binding_size {
        return Err(WgpuError::Unsupported(format!(
            "gemv_w4a16 {what} needs {bytes} bytes; device allows {} per storage binding",
            ctx.caps.max_storage_buffer_binding_size
        )));
    }
    Ok(())
}

pub fn shape_rule(k: usize, gs: usize) -> Result<()> {
    if !k.is_multiple_of(32) || !k.is_multiple_of(gs) || !gs.is_multiple_of(8) {
        return Err(WgpuError::Shape(format!(
            "gemv_w4a16 requires K%32==0, K%GS==0 and GS%8==0; got K={k} GS={gs}"
        )));
    }
    Ok(())
}

fn widen_u16(src: &[u16]) -> Vec<u32> {
    src.iter().map(|v| *v as u32).collect()
}

pub fn pack_scale_words(src: &[u16]) -> Vec<u32> {
    let mut out = vec![0u32; src.len().div_ceil(2).max(1)];
    for (i, w) in out.iter_mut().enumerate() {
        let lo = src.get(2 * i).copied().unwrap_or(0) as u32;
        let hi = src.get(2 * i + 1).copied().unwrap_or(0) as u32;
        *w = lo | (hi << 16);
    }
    out
}

struct Plan {
    packed: wgpu::Buffer,
    scale: wgpu::Buffer,
    x: wgpu::Buffer,
    y: wgpu::Buffer,
    params: wgpu::Buffer,
    groups: (u32, u32, u32),
}

impl Plan {
    fn bindings(&self) -> [(u32, &wgpu::Buffer); 5] {
        [
            (0, &self.packed),
            (1, &self.scale),
            (2, &self.x),
            (3, &self.y),
            (4, &self.params),
        ]
    }
}

fn plan(
    ctx: &WgpuContext,
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    n: usize,
    k: usize,
    gs: usize,
    rows_per_group: u32,
) -> Result<Plan> {
    let groups_per_row = k / gs;
    dispatch::check_len("gemv_w4a16 packed", packed.len(), n * (k / 8))?;
    dispatch::check_len("gemv_w4a16 scale", scales.len(), n * groups_per_row)?;
    dispatch::check_len("gemv_w4a16 x", x.len(), k)?;
    check_device(ctx)?;
    check_binding(ctx, "packed", (n as u64) * (k as u64 / 8) * 4)?;
    check_binding(ctx, "scale", (n as u64) * (groups_per_row as u64) * 4)?;
    check_binding(ctx, "y", (n as u64) * 4)?;

    let groups = dispatch::workgroup_count_1d(ctx, n as u64, rows_per_group);
    let params = GemvW4A16Params {
        n_rows: n as u32,
        k_elems: k as u32,
        gs: gs as u32,
        w_row_words: (k / 8) as u32,
        scale_row_stride: groups_per_row as u32,
        groups_x: groups.0,
    };

    Ok(Plan {
        packed: dispatch::storage_from_slice(ctx, "gemv-w4a16-packed", packed),
        scale: dispatch::storage_from_slice(ctx, "gemv-w4a16-scale", &widen_u16(scales)),
        x: dispatch::storage_from_slice(ctx, "gemv-w4a16-x", &pack_u16(x)),
        y: dispatch::storage_zeroed(ctx, "gemv-w4a16-y", (n * 4) as u64),
        params: dispatch::uniform_from(ctx, "gemv-w4a16-params", &params),
        groups,
    })
}

fn store_rows(ctx: &WgpuContext, plan: &Plan, y: &mut [u16], n: usize) -> Result<()> {
    let words: Vec<u32> = dispatch::read_back(ctx, &plan.y, n)?;
    for (dst, word) in y.iter_mut().zip(words.iter()) {
        *dst = (*word & 0xffff) as u16;
    }
    Ok(())
}

pub fn gemv_w4a16(
    ctx: &WgpuContext,
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    y: &mut [u16],
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<()> {
    if n == 0 || k == 0 || group_size == 0 {
        return Ok(());
    }
    shape_rule(k, group_size)?;
    dispatch::check_len("gemv_w4a16 y", y.len(), n)?;
    let entry = entry_for(k);
    let rows_per_group = if entry == BLOCK_ENTRY {
        ROWS_PER_GROUP
    } else {
        1
    };
    let p = plan(ctx, packed, scales, x, n, k, group_size, rows_per_group)?;
    dispatch::run(
        ctx,
        "nv_kernels_gemv_w4a16",
        &compose(WGSL),
        entry,
        &p.bindings(),
        p.groups,
    )?;
    store_rows(ctx, &p, y, n)
}

pub fn gemv_w4a16_gelu_pli(
    ctx: &WgpuContext,
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    per_layer_input: &[f32],
    y: &mut [u16],
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<()> {
    if n == 0 || k == 0 || group_size < 32 {
        return Err(WgpuError::Shape(format!(
            "gemv_w4a16_gelu_pli requires N>0, K>0 and GS>=32; got N={n} K={k} GS={group_size}"
        )));
    }
    if k > MAX_SHARED_K {
        return Err(WgpuError::Shape(format!(
            "gemv_w4a16_gelu_pli requires K<={MAX_SHARED_K}; got {k}"
        )));
    }
    shape_rule(k, group_size)?;
    dispatch::check_len("gemv_w4a16_gelu_pli y", y.len(), n)?;
    dispatch::check_len("gemv_w4a16_gelu_pli pli", per_layer_input.len(), n)?;
    let p = plan(ctx, packed, scales, x, n, k, group_size, ROWS_PER_GROUP)?;
    let pli = dispatch::storage_from_slice(ctx, "gemv-w4a16-pli", per_layer_input);
    let base = p.bindings();
    let bindings = [base[0], base[1], base[2], base[3], base[4], (5, &pli)];
    dispatch::run(
        ctx,
        "nv_kernels_gemv_w4a16_gelu_pli",
        &compose(WGSL),
        GELU_PLI_ENTRY,
        &bindings,
        p.groups,
    )?;
    store_rows(ctx, &p, y, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_host_guards_match_cuda_and_refuse_every_grain_width_and_shape_the_sg_body_cannot_express(
    ) {
        {
            assert_eq!(entry_for(32), BLOCK_ENTRY);
            assert_eq!(entry_for(MAX_SHARED_K), BLOCK_ENTRY);
            assert_eq!(entry_for(MAX_SHARED_K + 32), ROW_ENTRY);
            assert_eq!(entry_for(4096), ROW_ENTRY);
        }
        {
            assert!(shape_rule(4096, 128).is_ok());
            assert!(shape_rule(4096, 8).is_ok());
            assert!(shape_rule(48, 16).is_err());
            assert!(shape_rule(4096, 12).is_err());
            assert!(shape_rule(4096, 3).is_err());
        }
        {
            assert_eq!(ScaleGrain::for_group_size(16), Some(ScaleGrain::G16));
            assert_eq!(ScaleGrain::for_group_size(32), Some(ScaleGrain::Ge32));
            assert_eq!(ScaleGrain::for_group_size(64), Some(ScaleGrain::Ge32));
            assert_eq!(ScaleGrain::for_group_size(128), Some(ScaleGrain::Ge32));
            assert_eq!(ScaleGrain::for_group_size(8), None);
            assert_eq!(ScaleGrain::for_group_size(48), None);
            assert_eq!(
                ScaleGrain::fastest_for_group_size(32),
                Some(ScaleGrain::Ge32Fixed(0))
            );
            assert_eq!(
                ScaleGrain::fastest_for_group_size(64),
                Some(ScaleGrain::Ge32Fixed(1))
            );
            assert_eq!(
                ScaleGrain::fastest_for_group_size(128),
                Some(ScaleGrain::Ge32Fixed(2))
            );
            assert_eq!(
                ScaleGrain::fastest_for_group_size(96),
                Some(ScaleGrain::Ge32)
            );
            assert_eq!(
                ScaleGrain::fastest_for_group_size(16),
                Some(ScaleGrain::G16)
            );
            assert!(ScaleGrain::Ge32Fixed(0).accepts(32));
            assert!(!ScaleGrain::Ge32Fixed(0).accepts(64));
            assert!(ScaleGrain::Ge32.accepts(64) && !ScaleGrain::Ge32.accepts(16));
            assert!(ScaleGrain::G16.accepts(16) && !ScaleGrain::G16.accepts(32));
            assert!(sg_pk_source_grain(ScaleGrain::Ge32Fixed(0)).contains("let sgi = sbase + v;"));
            assert!(sg_pk_source_grain(ScaleGrain::Ge32Fixed(2))
                .contains("let sgi = sbase + (v >> 2u);"));
            assert!(!sg_pk_source_grain(ScaleGrain::Ge32Fixed(1)).contains("/ sgp_params.gs"));
            assert_eq!(
                w4_route_grain(43008, 5376, 16, true, false),
                Some((W4Route::Sg16, ScaleGrain::G16))
            );
            assert_eq!(
                w4_route_grain(5376, 21504, 16, true, false),
                Some((W4Route::Sg16, ScaleGrain::G16))
            );
            assert_eq!(
                w4_route_grain(5376, 21504, 32, true, false),
                Some((W4Route::Sg16, ScaleGrain::Ge32Fixed(0)))
            );
            assert_eq!(
                w4_route_grain(512, 21504, 32, false, false),
                Some((W4Route::V4, ScaleGrain::Ge32Fixed(0)))
            );
            assert_eq!(
                w4_route_grain(43008, 5376, 96, true, false),
                Some((W4Route::Sg16, ScaleGrain::Ge32))
            );
            assert_eq!(
                w4_route_grain(43008, 5376, 16, false, false),
                Some((W4Route::Block, ScaleGrain::G16))
            );
            assert_eq!(w4_route_grain(43008, 5376, 8, true, false), None);
            assert!(g16_shape_rule(5376).is_ok());
            assert!(g16_shape_rule(21504).is_ok());
            assert!(g16_shape_rule(48).is_err());
        }
        {
            assert!(require_grain(ScaleGrain::Ge32, 16).is_err());
            assert!(require_grain(ScaleGrain::Ge32, 8).is_err());
            assert!(require_grain(ScaleGrain::Ge32, 32).is_ok());
            assert!(require_grain(ScaleGrain::G16, 32).is_err());
            assert!(require_grain(ScaleGrain::G16, 16).is_ok());

            assert!(require_grain(ScaleGrain::Ge32Fixed(0), 64).is_err());
            assert!(require_grain(ScaleGrain::Ge32Fixed(1), 32).is_err());
            assert!(require_grain(ScaleGrain::Ge32Fixed(1), 64).is_ok());
            for gs in [16usize, 32, 64, 128] {
                let (src, grain) = sg_pk_source_for(gs).expect("source");
                assert!(grain.accepts(gs));
                assert_eq!(src, sg_pk_source_grain(grain));
                assert!(src.contains(SG_PK_ENTRY));
                assert!(!src.contains("/ sgp_params.gs"));
            }
            assert!(sg_pk_source_for(96).is_ok());
            assert!(sg_pk_source_for(8).is_err());
            assert!(sg_pk_source_for(48).is_err());
        }
        {
            assert!(sg_pk_supported(Some(16)));
            assert!(sg_pk_supported(Some(32)));
            assert!(sg_pk_supported(Some(64)));
            assert!(!sg_pk_supported(Some(8)));
            assert!(!sg_pk_supported(Some(4)));
            assert!(!sg_pk_supported(None));
        }
        {
            assert_eq!(sg_lane_width(true, 4, 64, Some(32)), Some(32));
            assert_eq!(sg_lane_width(true, 4, 64, Some(16)), Some(16));
            assert_eq!(sg_lane_width(true, 4, 64, None), Some(4));
            assert_eq!(sg_lane_width(true, 32, 32, None), Some(32));
            assert_eq!(sg_lane_width(true, 64, 64, None), Some(32));
            assert_eq!(sg_lane_width(false, 32, 32, Some(32)), None);
            assert_eq!(sg_lane_width(true, 2, 64, None), None);
        }
    }

    const E4B_QKV: (usize, usize) = (3072, 2560);
    const E4B_O: (usize, usize) = (2560, 2048);
    const E4B_GATE_UP: (usize, usize) = (20480, 2560);
    const E4B_DOWN: (usize, usize) = (2560, 10240);
    const B31_GATE_UP: (usize, usize) = (43008, 5376);
    const B31_DOWN: (usize, usize) = (5376, 21504);

    fn route(shape: (usize, usize)) -> W4Route {
        w4_route(shape.0, shape.1, 32, true, false)
    }

    #[test]
    fn w4_route_picks_the_measured_winner_and_falls_back_on_capability_not_aspect_ratio() {
        {
            assert_eq!(route(E4B_GATE_UP), W4Route::Sg16);
            assert_eq!(route(E4B_QKV), W4Route::Sg16);
            assert_eq!(route(E4B_O), W4Route::Sg16);
            assert_eq!(route(B31_GATE_UP), W4Route::Sg16);
            assert_eq!(route(E4B_DOWN), W4Route::Sg16);
            assert_eq!(route(B31_DOWN), W4Route::Sg16);
        }
        {
            for k in [512usize, 2048, 8192, 21504] {
                for n in [64usize, 512, 1024, 4096, 248320] {
                    assert_eq!(w4_route(n, k, 32, true, false), W4Route::Sg16);
                    assert_eq!(w4_route(n, k, 32, false, false), W4Route::V4);
                    assert_eq!(w4_route(n, k, 16, true, false), W4Route::Block);
                    assert_eq!(w4_route(n, k, 32, true, true), W4Route::Block);
                }
            }
            for scale in [1usize, 2, 3, 7] {
                assert_eq!(
                    route((E4B_DOWN.0 * scale, E4B_DOWN.1 * scale)),
                    W4Route::Sg16
                );
                assert_eq!(
                    route((E4B_GATE_UP.0 * scale, E4B_GATE_UP.1 * scale)),
                    W4Route::Sg16
                );
            }
            assert!(prefers_v4(512, 2048));
            assert!(!prefers_v4(512, 2016));
            assert!(prefers_v4(64, 4096));
            assert!(!prefers_v4(4096, 4096));
            assert!(!prefers_v4(0, 4096));
        }
        {
            assert_eq!(w4_route(20480, 2560, 32, true, true), W4Route::Block);
            assert_eq!(w4_route(2560, 10240, 32, true, true), W4Route::Block);
            assert_eq!(w4_route(20480, 2560, 16, true, false), W4Route::Block);
            assert_eq!(w4_route(20480, 2560, 32, false, false), W4Route::V4);
            assert_eq!(w4_route(2560, 10240, 32, false, false), W4Route::V4);
            assert_eq!(route_rows_per_group(W4Route::Sg16), SG_PK_ROWS);
            assert_eq!(route_rows_per_group(W4Route::V4), ROWS_PER_GROUP);
            assert_eq!(route_rows_per_group(W4Route::Block), ROWS_PER_GROUP);
        }
    }

    #[test]
    fn every_source_declares_its_entries_keeps_ge32_codegen_expresses_group_sixteen_and_branches_the_wide_arm_on_divisibility(
    ) {
        {
            assert!(WGSL.contains(BLOCK_ENTRY));
            assert!(WGSL.contains(ROW_ENTRY));
            assert!(WGSL.contains(GELU_PLI_ENTRY));
            assert!(compose(WGSL).contains("fn u4_unpack("));
        }
        {
            assert!(WGSL.contains(V4_ENTRY));
            assert!(WGSL.contains("unpack4xU8"));
        }
        {
            let src = sg_source(4, 256);
            assert!(src.contains(SG_ENTRY));
            assert!(src.contains("subgroupShuffleXor(acc, 2u)"));
            assert!(src.contains("subgroupShuffleXor(acc, 1u)"));
            assert!(src.contains("fn bf16_encode("));
            let wide = sg_source(32, 256);
            assert!(wide.contains("subgroupShuffleXor(acc, 16u)"));
        }
        {
            let src = sg_pk_source();
            assert!(src.contains(SG_PK_ENTRY));
            assert!(src.contains(SG_PK3_ENTRY));
            assert!(src.contains("subgroupShuffleXor(acc, 8u)"));
            assert!(src.contains("subgroupShuffleXor(acc, 1u)"));
            assert!(!src.contains("subgroupShuffleXor(acc, 16u)"));
            assert!(src.contains("sgp_off.dst_word_off"));
            assert!(src.contains("sgp_split.q_rows"));
            assert!(src.contains("fn bf16_encode("));
            assert!(src.contains("bf16_lo(ssw)"));
            assert!(src.contains("sgp_scale[sgi >> 1u]"));
            assert_eq!(SG_PK_ROWS % 2, 0);
        }
        {
            for mr in [2u32, 4, 8] {
                let src = sg_pk_mr_source(mr);
                assert!(src.contains(SG_PKM_ENTRY));
                assert!(src.contains(SG_PKM3_ENTRY));
                assert!(src.contains(&format!("array<f32, {mr}>")));
                assert!(src.contains(&format!("* {mr}u;")));
                assert!(src.contains("subgroupShuffleXor(acc, 8u)"));
                assert!(src.contains("subgroupShuffleXor(acc, 1u)"));
                assert!(!src.contains("subgroupShuffleXor(acc, 16u)"));
                assert!(!src.contains("var<workgroup>"));
                assert!(src.contains("sgp_scale[sgi >> 1u]"));
                assert!(src.contains("sgp_off.dst_word_off"));
                assert!(src.contains("sgp_split.q_rows"));
            }
        }
        {
            for mk_max in [2u32, 8, 16] {
                let src = sg_mk_source(mk_max);
                assert!(src.contains(SG_MK_PK_ENTRY));
                assert!(src.contains(SG_MK_PK3_ENTRY));
                assert!(src.contains(&format!("array<f32, {mk_max}>")));
                assert!(src.contains("subgroupShuffleXor(acc, 8u)"));
                assert!(src.contains("subgroupShuffleXor(acc, 1u)"));
                assert!(!src.contains("subgroupShuffleXor(acc, 16u)"));
                assert!(src.contains("sgp_mk.dst_word_off"));
                assert!(src.contains("sgp_split.q_rows"));
                assert!(src.contains("fn bf16_encode("));
                assert!(src.contains("sgp_scale[sgi >> 1u]"));
            }
        }
        {
            let ge32 = sg_pk_source_grain(ScaleGrain::Ge32);
            assert_eq!(ge32, sg_pk_source());
            assert!(ge32.contains("(v << 5u) / sgp_params.gs"));
            assert!(!ge32.contains("sc_hi"));
            let g16 = sg_pk_source_grain(ScaleGrain::G16);
            assert!(g16.contains("let sgi = sbase + (v << 1u);"));
            assert!(g16.contains("let sc_lo = bf16_lo(ssw);"));
            assert!(g16.contains("let sc_hi = bf16_hi(ssw);"));
            assert!(!g16.contains("/ sgp_params.gs"));
            assert!(g16.contains(SG_PK_ENTRY) && g16.contains(SG_PK3_ENTRY));
            for mk in [1u32, 4, 16] {
                let mk16 = sg_mk_unrolled_source_grain(mk, ScaleGrain::G16);
                assert_eq!(mk16.matches("let sc_lo = bf16_lo(ssw);").count(), 2);
                assert!(mk16.contains(SG_MK_PK_ENTRY) && mk16.contains(SG_MK_PK3_ENTRY));
                assert_eq!(
                    sg_mk_unrolled_source_grain(mk, ScaleGrain::Ge32),
                    sg_mk_unrolled_source(mk)
                );
            }
        }
        {
            for mk in [1u32, 4, 16] {
                assert_eq!(sg_mk_source_grain(mk, ScaleGrain::Ge32), sg_mk_source(mk));
                let ge32 = sg_mk_source(mk);
                assert!(ge32.contains("let sgi = sbase + ((v << 5u) / sgp_params.gs);"));
                assert!(!ge32.contains("sc_hi"));
                let g16 = sg_mk_source_grain(mk, ScaleGrain::G16);
                assert!(g16.contains("let sgi = sbase + (v << 1u);"));
                assert!(g16.contains("let sc_hi = bf16_hi(ssw);"));
                assert!(!g16.contains("/ sgp_params.gs"));
                assert!(g16.contains(SG_MK_PK_ENTRY) && g16.contains(SG_MK_PK3_ENTRY));
                let fixed = sg_mk_source_grain(mk, ScaleGrain::Ge32Fixed(2));
                assert!(fixed.contains("let sgi = sbase + (v >> 2u);"));
                assert!(!fixed.contains("/ sgp_params.gs"));
            }
            for mr in [2u32, 4, 8] {
                assert_eq!(
                    sg_pk_mr_source_grain(mr, ScaleGrain::Ge32),
                    sg_pk_mr_source(mr)
                );
                assert!(sg_pk_mr_source(mr).contains("let sgo = (v << 5u) / sgp_params.gs;"));
                let g16 = sg_pk_mr_source_grain(mr, ScaleGrain::G16);
                assert!(g16.contains("let sgo = v << 1u;"));
                assert!(g16.contains("accs[m] = fma(sc_lo, a0, accs[m]);"));
                assert!(g16.contains("accs[m] = fma(sc_hi, a1, accs[m]);"));
                assert!(!g16.contains("/ sgp_params.gs"));
                assert!(g16.contains(SG_PKM_ENTRY) && g16.contains(SG_PKM3_ENTRY));
                assert!(
                    sg_pk_mr_source_grain(mr, ScaleGrain::Ge32Fixed(0)).contains("let sgo = v;")
                );
                assert!(sg_pk_mr_source_grain(mr, ScaleGrain::Ge32Fixed(1))
                    .contains("let sgo = v >> 1u;"));
            }
        }
        {
            let src = compose(WGSL);
            for f in ["fn w4a16_row_acc_v4(", "fn w4a16_row_acc_block("] {
                let body = &src[src.find(f).expect("body")..];
                let body = &body[..body.find("\n@compute").unwrap_or(body.len())];
                assert!(
                    body.contains("if ((gs & 31u) == 0u)"),
                    "{f} wide-arm gate: {body}"
                );
                assert!(body.contains("kb / gs"));
            }
            let row = &src[src.find("fn gemv_w4a16_row(").expect("row entry")..];
            assert!(row.contains("let wide = (gs & 31u) == 0u;"));
        }
    }

    #[test]
    fn scale_words_carry_one_bf16_each_and_packed_words_two() {
        {
            assert_eq!(widen_u16(&[0x3f80, 0xbf00]), vec![0x3f80u32, 0xbf00u32]);
            assert_eq!(pack_u16(&[0x1234, 0xabcd]), vec![0xabcd_1234u32]);
        }
        {
            assert_eq!(pack_scale_words(&[0x3f80, 0xbf00]), vec![0xbf00_3f80u32]);
            assert_eq!(
                pack_scale_words(&[0x3f80, 0xbf00, 0x1234]),
                vec![0xbf00_3f80u32, 0x0000_1234u32]
            );
            assert_eq!(pack_scale_words(&[]), vec![0u32]);
        }
    }
}
