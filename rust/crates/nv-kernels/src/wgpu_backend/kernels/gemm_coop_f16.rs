#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::qualify::{CoopDecision, CoopRequest, CoopScalar, COOP_UNSAFE_SWEEP_ENV};
use crate::wgpu_backend::{compose_enabled, Result, WgpuError};

pub const TILE: u32 = 8;
pub const TILES: [u32; 2] = [16, 8];
pub const SUBGROUP: u32 = 32;
pub const ENABLES: [&str; 2] = ["f16", "wgpu_cooperative_matrix"];

pub const ACC_FRAGS: u32 = 16;

pub const ACC_LANE_DWORD_BUDGET: u32 = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CoopGemmParams {
    pub n_rows: u32,
    pub k_elems: u32,
    pub m_rows: u32,
    pub blocks_n: u32,
    pub y_stride: u32,
    pub groups_x: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    F16,
    F32,
}

impl Operand {
    fn wgsl(&self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
        }
    }
    fn tag(&self) -> &'static str {
        match self {
            Self::F16 => "h",
            Self::F32 => "f",
        }
    }
    pub fn bytes(&self) -> u32 {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
    pub fn scalar(&self) -> CoopScalar {
        match self {
            Self::F16 => CoopScalar::F16,
            Self::F32 => CoopScalar::F32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoopGemm {
    pub tile: u32,
    pub ab: Operand,
}

impl CoopGemm {
    pub fn new(tile: u32, ab: Operand) -> Self {
        assert!(
            TILES.contains(&tile),
            "wgsl spells only coop_mat8x8 and coop_mat16x16 (naga 30 front/wgsl/parse/conv.rs:561), \
             so a {tile}x{tile} fragment cannot be emitted"
        );
        Self { tile, ab }
    }

    pub fn request(&self) -> CoopRequest {
        CoopRequest::square(self.tile, self.ab.scalar(), CoopScalar::F32)
    }

    pub fn tiles(&self, m: u32, acc: u32) -> (u32, u32) {
        let tm = m.div_ceil(self.tile).clamp(1, acc);
        let tn = (acc / tm).max(1);
        (tm, tn)
    }

    pub fn rows_per_block(&self, tm: u32) -> u32 {
        self.tile * tm
    }

    pub fn cols_per_workgroup(&self, tn: u32, sg: u32) -> u32 {
        self.tile * tn * sg
    }

    pub fn grid(&self, m: u32, n: u32, tm: u32, tn: u32, sg: u32) -> (u32, u32) {
        let blocks_m = m.div_ceil(self.rows_per_block(tm));
        let blocks_n = n.div_ceil(self.cols_per_workgroup(tn, sg));
        (blocks_m, blocks_n)
    }

    pub fn zero_elems(&self) -> usize {
        (self.tile * self.tile) as usize
    }

    pub fn acc_lane_dwords(&self, tm: u32, tn: u32) -> u32 {
        tm * tn * self.tile * self.tile / SUBGROUP
    }

    pub fn acc_fits_a_register_file(&self, tm: u32, tn: u32) -> bool {
        self.acc_lane_dwords(tm, tn) <= ACC_LANE_DWORD_BUDGET
    }

    pub fn check_shape(&self, m: u32, n: u32, k: u32) -> Result<()> {
        let t = self.tile;
        if m == 0 || n == 0 || k == 0 {
            return Err(WgpuError::Shape("zero extent".to_string()));
        }
        if !k.is_multiple_of(t) {
            return Err(WgpuError::Shape(format!("K {k} must be a multiple of {t}")));
        }
        if !m.is_multiple_of(t) {
            return Err(WgpuError::Shape(format!(
                "M {m} must be a multiple of {t}: the epilogue stores whole {t}x{t} fragments and \
                 a ragged last row band would write past row M"
            )));
        }
        if !n.is_multiple_of(t) {
            return Err(WgpuError::Shape(format!(
                "N {n} must be a multiple of {t}: the epilogue stores whole {t}x{t} fragments and \
                 a ragged last column band would write into the next row"
            )));
        }
        Ok(())
    }

    pub fn entry(&self, tm: u32, tn: u32, sg: u32, ku: u32) -> String {
        format!(
            "gemm_coop_{}{}_tm{tm}_tn{tn}_sg{sg}_ku{ku}",
            self.ab.tag(),
            self.tile
        )
    }

    pub fn source(&self, tm: u32, tn: u32, sg: u32, ku: u32) -> String {
        use std::fmt::Write as _;
        assert!(tm >= 1 && tn >= 1 && tm * tn <= 64, "tile too large");
        assert!((1..=8).contains(&sg), "sg out of range");
        assert!((1..=8).contains(&ku), "ku out of range");

        let t = self.tile;
        let wg = sg * SUBGROUP;
        let entry = self.entry(tm, tn, sg, ku);
        let cols = self.cols_per_workgroup(tn, sg);
        let abt = self.ab.wgsl();
        let mut b = String::new();

        writeln!(b, "alias CGA = coop_mat{t}x{t}<{abt}, A>;").unwrap();
        writeln!(b, "alias CGB = coop_mat{t}x{t}<{abt}, B>;").unwrap();
        writeln!(b, "alias CGC = coop_mat{t}x{t}<f32, C>;\n").unwrap();
        b.push_str("struct CoopGemmParams {\n    n_rows: u32,\n    k_elems: u32,\n    m_rows: u32,\n    blocks_n: u32,\n    y_stride: u32,\n    groups_x: u32,\n    pad0: u32,\n    pad1: u32,\n};\n\n");
        writeln!(
            b,
            "@group(0) @binding(0) var<storage, read> cg_w: array<{abt}>;"
        )
        .unwrap();
        writeln!(
            b,
            "@group(0) @binding(1) var<storage, read> cg_x: array<{abt}>;"
        )
        .unwrap();
        b.push_str("@group(0) @binding(2) var<storage, read_write> cg_y: array<f32>;\n");
        b.push_str("@group(0) @binding(3) var<uniform> cg_p: CoopGemmParams;\n");
        b.push_str("@group(0) @binding(4) var<storage, read> cg_zero: array<f32>;\n\n");

        writeln!(b, "@compute @workgroup_size({wg})").unwrap();
        writeln!(b, "fn {entry}(").unwrap();
        b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_index) lidx: u32\n) {\n");
        b.push_str("    let sgid = lidx / 32u;\n");
        b.push_str("    let block = wid.x + wid.y * cg_p.groups_x;\n");
        b.push_str("    let bm = block / cg_p.blocks_n;\n");
        b.push_str("    let bn = block - bm * cg_p.blocks_n;\n");
        writeln!(b, "    let m0 = bm * {}u;", self.rows_per_block(tm)).unwrap();
        writeln!(b, "    let n0 = bn * {cols}u + sgid * {}u;", t * tn).unwrap();
        b.push_str("    let kk = cg_p.k_elems;\n");
        b.push_str("    let ys = cg_p.y_stride;\n");
        b.push_str("    let mrows = cg_p.m_rows;\n");
        b.push_str("    let nrows = cg_p.n_rows;\n");

        for r in 0..tm {
            for j in 0..tn {
                writeln!(b, "    var c{r}_{j} = coopLoadT<CGC>(&cg_zero[0], {t}u);").unwrap();
            }
        }
        writeln!(
            b,
            "    for (var kt = 0u; kt < kk; kt = kt + {}u) {{",
            t * ku
        )
        .unwrap();
        for u in 0..ku {
            for r in 0..tm {
                writeln!(
                    b,
                    "        let a{u}_{r} = coopLoadT<CGA>(&cg_x[(m0 + {}u) * kk + kt + {}u], kk);",
                    r * t,
                    u * t
                )
                .unwrap();
            }
            for j in 0..tn {
                writeln!(
                    b,
                    "        let b{u}_{j} = coopLoad<CGB>(&cg_w[(n0 + {}u) * kk + kt + {}u], kk);",
                    j * t,
                    u * t
                )
                .unwrap();
            }
        }
        for u in 0..ku {
            for r in 0..tm {
                for j in 0..tn {
                    writeln!(
                        b,
                        "        c{r}_{j} = coopMultiplyAdd(a{u}_{r}, b{u}_{j}, c{r}_{j});"
                    )
                    .unwrap();
                }
            }
        }
        b.push_str("    }\n");

        for r in 0..tm {
            for j in 0..tn {
                writeln!(b, "    let sc{r}_{j} = c{r}_{j};").unwrap();
                writeln!(
                    b,
                    "    let yo{r}_{j} = (m0 + {}u) * ys + n0 + {}u;",
                    r * t,
                    j * t
                )
                .unwrap();
                writeln!(
                    b,
                    "    if (m0 + {}u < mrows && n0 + {}u < nrows) {{",
                    r * t,
                    j * t
                )
                .unwrap();
                writeln!(b, "        coopStoreT(sc{r}_{j}, &cg_y[yo{r}_{j}], ys);").unwrap();
                b.push_str("    }\n");
            }
        }
        b.push_str("}\n");

        compose_enabled(&ENABLES, &b)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WqFmt {
    Nvfp4Block16,
    Fp8RowscalePlain,
}

impl WqFmt {
    fn wtag(self) -> &'static str {
        match self {
            WqFmt::Nvfp4Block16 => "w4",
            WqFmt::Fp8RowscalePlain => "w8",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WqAct {
    F16,
    Fp8Rowscale,
    Nvfp4Block16,
}

impl WqAct {
    fn atag(self) -> &'static str {
        match self {
            WqAct::F16 => "a16",
            WqAct::Fp8Rowscale => "a8",
            WqAct::Nvfp4Block16 => "a4",
        }
    }
    pub fn a_bytes(self, m: u32, k: u32) -> f64 {
        let (m, k) = (m as f64, k as f64);
        match self {
            WqAct::F16 => m * k * 2.0,
            WqAct::Fp8Rowscale => m * k + m * 4.0,
            WqAct::Nvfp4Block16 => m * k / 2.0 + m * k / 16.0,
        }
    }
}

const E4M3_PLAIN_DECODE_ALL_INTERMEDIATES_NORMAL_BECAUSE_THIS_ADAPTER_FLUSHES_F32_DENORMALS:
    &str = "
fn wq16_e4m3_plain(bits: u32) -> f32 {
    let s = select(1.0, -1.0, (bits & 128u) != 0u);
    let e = (bits >> 3u) & 15u;
    let m = bits & 7u;
    if (e == 0u) {
        return s * f32(m) * 0.001953125;
    }
    return s * (8.0 + f32(m)) * exp2(f32(e) - 10.0);
}
";

pub fn entry_wq16(fmt: WqFmt, tm: u32, tn: u32, sg: u32, ku: u32) -> String {
    entry_wq16_act(fmt, WqAct::F16, tm, tn, sg, ku)
}

pub fn entry_wq16_act(fmt: WqFmt, act: WqAct, tm: u32, tn: u32, sg: u32, ku: u32) -> String {
    format!(
        "gemm_coop_{}{}_16_tm{tm}_tn{tn}_sg{sg}_ku{ku}",
        fmt.wtag(),
        act.atag()
    )
}

pub fn entry_w4a16(tm: u32, tn: u32, sg: u32, ku: u32) -> String {
    entry_wq16(WqFmt::Nvfp4Block16, tm, tn, sg, ku)
}

pub const W4A16_STAGE_BUDGET_F16_IS_HALF_THE_48K_WORKGROUP_LIMIT: u32 = 24576;

pub fn source_w4a16(tm: u32, tn: u32, sg: u32, ku: u32) -> String {
    source_wq16(WqFmt::Nvfp4Block16, tm, tn, sg, ku)
}

pub fn source_wq16(fmt: WqFmt, tm: u32, tn: u32, sg: u32, ku: u32) -> String {
    source_wq16_act(fmt, WqAct::F16, tm, tn, sg, ku)
}

pub fn wq16_act_stage_elems(act: WqAct, tm: u32, tn: u32, sg: u32, ku: u32) -> u32 {
    let g = CoopGemm::new(16, Operand::F16);
    let b_elems = g.cols_per_workgroup(tn, sg) * g.tile * ku;
    let a_elems = if act == WqAct::F16 {
        0
    } else {
        g.rows_per_block(tm) * g.tile * ku
    };
    b_elems + a_elems
}

pub fn entry_wq16_act_y16(fmt: WqFmt, act: WqAct, tm: u32, tn: u32, sg: u32, ku: u32) -> String {
    format!("{}_y16", entry_wq16_act(fmt, act, tm, tn, sg, ku))
}

pub fn source_wq16_act_y16(fmt: WqFmt, act: WqAct, tm: u32, tn: u32, sg: u32, ku: u32) -> String {
    source_wq16_act_inner(fmt, act, tm, tn, sg, ku, true)
}

pub fn source_wq16_act(fmt: WqFmt, act: WqAct, tm: u32, tn: u32, sg: u32, ku: u32) -> String {
    source_wq16_act_inner(fmt, act, tm, tn, sg, ku, false)
}

fn source_wq16_act_inner(
    fmt: WqFmt,
    act: WqAct,
    tm: u32,
    tn: u32,
    sg: u32,
    ku: u32,
    y_bf16: bool,
) -> String {
    use std::fmt::Write as _;
    let g = CoopGemm::new(16, Operand::F16);
    assert!(tm >= 1 && tn >= 1 && tm * tn <= 64, "tile too large");
    assert!((1..=8).contains(&sg), "sg out of range");
    assert!((1..=4).contains(&ku), "ku out of range");
    let t = g.tile;
    assert!(t == 16, "the wq16 unpack stages one 16-code weight block per k-tile row");
    let wg = sg * SUBGROUP;
    let entry = if y_bf16 {
        entry_wq16_act_y16(fmt, act, tm, tn, sg, ku)
    } else {
        entry_wq16_act(fmt, act, tm, tn, sg, ku)
    };
    let cols = g.cols_per_workgroup(tn, sg);
    let rows_pb = g.rows_per_block(tm);
    let a_pitch = t * ku;
    let a_elems = rows_pb * a_pitch;
    assert!(
        wq16_act_stage_elems(act, tm, tn, sg, ku)
            <= W4A16_STAGE_BUDGET_F16_IS_HALF_THE_48K_WORKGROUP_LIMIT,
        "staged B ({cols} cols x {t} x ku {ku}) plus staged A exceeds the workgroup budget"
    );
    let mut b = String::new();

    writeln!(b, "alias CGA = coop_mat{t}x{t}<f16, A>;").unwrap();
    writeln!(b, "alias CGB = coop_mat{t}x{t}<f16, B>;").unwrap();
    writeln!(b, "alias CGC = coop_mat{t}x{t}<f32, C>;\n").unwrap();
    if y_bf16 {
        b.push_str("struct CoopGemmParams {\n    n_rows: u32,\n    k_elems: u32,\n    m_rows: u32,\n    blocks_n: u32,\n    y_stride: u32,\n    groups_x: u32,\n    alpha: f32,\n    pad1: u32,\n};\n\n");
    } else {
        b.push_str("struct CoopGemmParams {\n    n_rows: u32,\n    k_elems: u32,\n    m_rows: u32,\n    blocks_n: u32,\n    y_stride: u32,\n    groups_x: u32,\n    pad0: u32,\n    pad1: u32,\n};\n\n");
    }
    b.push_str("@group(0) @binding(0) var<storage, read> cg_w4: array<u32>;\n");
    match act {
        WqAct::F16 => b.push_str("@group(0) @binding(1) var<storage, read> cg_x: array<f16>;\n"),
        _ => b.push_str("@group(0) @binding(1) var<storage, read> cg_x: array<u32>;\n"),
    }
    if y_bf16 {
        b.push_str("@group(0) @binding(2) var<storage, read_write> cg_y: array<u32>;\n");
        writeln!(b, "var<workgroup> ysh: array<f32, {}>;", sg * t * t).unwrap();
    } else {
        b.push_str("@group(0) @binding(2) var<storage, read_write> cg_y: array<f32>;\n");
    }
    b.push_str("@group(0) @binding(3) var<uniform> cg_p: CoopGemmParams;\n");
    b.push_str("@group(0) @binding(4) var<storage, read> cg_zero: array<f32>;\n");
    match fmt {
        WqFmt::Nvfp4Block16 => {
            b.push_str("@group(0) @binding(5) var<storage, read> cg_sf: array<u32>;\n")
        }
        WqFmt::Fp8RowscalePlain => {
            b.push_str("@group(0) @binding(5) var<storage, read> cg_sf: array<f32>;\n")
        }
    }
    match act {
        WqAct::F16 => {}
        WqAct::Fp8Rowscale => {
            b.push_str("@group(0) @binding(6) var<storage, read> cg_xsf: array<f32>;\n")
        }
        WqAct::Nvfp4Block16 => {
            b.push_str("@group(0) @binding(6) var<storage, read> cg_xsf: array<u32>;\n")
        }
    }
    b.push('\n');
    if fmt == WqFmt::Fp8RowscalePlain || act == WqAct::Fp8Rowscale {
        b.push_str(E4M3_PLAIN_DECODE_ALL_INTERMEDIATES_NORMAL_BECAUSE_THIS_ADAPTER_FLUSHES_F32_DENORMALS);
    }
    writeln!(b, "var<workgroup> w4bs: array<f16, {}>;", cols * t * ku).unwrap();
    if act != WqAct::F16 {
        writeln!(b, "var<workgroup> xas: array<f16, {a_elems}>;").unwrap();
    }

    writeln!(b, "\n@compute @workgroup_size({wg})").unwrap();
    writeln!(b, "fn {entry}(").unwrap();
    b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_index) lidx: u32\n) {\n");
    b.push_str("    let sgid = lidx / 32u;\n");
    b.push_str("    let block = wid.x + wid.y * cg_p.groups_x;\n");
    b.push_str("    let bm = block / cg_p.blocks_n;\n");
    b.push_str("    let bn = block - bm * cg_p.blocks_n;\n");
    writeln!(b, "    let m0 = bm * {}u;", g.rows_per_block(tm)).unwrap();
    writeln!(b, "    let n0w = bn * {cols}u;").unwrap();
    writeln!(b, "    let n0 = n0w + sgid * {}u;", t * tn).unwrap();
    b.push_str("    let kk = cg_p.k_elems;\n");
    b.push_str("    let ys = cg_p.y_stride;\n");
    b.push_str("    let mrows = cg_p.m_rows;\n");
    b.push_str("    let nrows = cg_p.n_rows;\n");
    match fmt {
        WqFmt::Nvfp4Block16 => {
            b.push_str("    let row_words = kk >> 3u;\n");
            b.push_str("    let sf_row = kk >> 4u;\n");
        }
        WqFmt::Fp8RowscalePlain => {
            b.push_str("    let row_words = kk >> 2u;\n");
        }
    }
    match act {
        WqAct::F16 => {}
        WqAct::Fp8Rowscale => {
            b.push_str("    let xrow_words = kk >> 2u;\n");
        }
        WqAct::Nvfp4Block16 => {
            b.push_str("    let xrow_words = kk >> 3u;\n");
            b.push_str("    let xsf_row = kk >> 4u;\n");
        }
    }

    for r in 0..tm {
        for j in 0..tn {
            writeln!(b, "    var c{r}_{j} = coopLoadT<CGC>(&cg_zero[0], {t}u);").unwrap();
        }
    }
    let tiles = sg * tn;
    writeln!(
        b,
        "    for (var kt = 0u; kt < kk; kt = kt + {}u) {{",
        t * ku
    )
    .unwrap();
    b.push_str("        workgroupBarrier();\n");
    writeln!(
        b,
        "        for (var e = lidx; e < {}u; e = e + {wg}u) {{",
        cols * ku
    )
    .unwrap();
    writeln!(b, "            let u = e / {cols}u;").unwrap();
    writeln!(b, "            let c = e - u * {cols}u;").unwrap();
    b.push_str("            let gr = n0w + c;\n");
    b.push_str("            let ku16 = kt + u * 16u;\n");
    writeln!(b, "            let tile = u * {tiles}u + (c >> 4u);").unwrap();
    b.push_str("            let row = c & 15u;\n");
    writeln!(b, "            let tb = tile * {}u;", t * t).unwrap();
    match fmt {
        WqFmt::Nvfp4Block16 => {
            b.push_str("            var s = 0.0;\n");
            b.push_str("            var w0 = 0u;\n");
            b.push_str("            var w1 = 0u;\n");
            b.push_str("            if (gr < nrows) {\n");
            b.push_str("                let base = gr * row_words + (ku16 >> 3u);\n");
            b.push_str("                w0 = cg_w4[base];\n");
            b.push_str("                w1 = cg_w4[base + 1u];\n");
            b.push_str("                let sbi = gr * sf_row + (ku16 >> 4u);\n");
            b.push_str("                s = ue4m3_decode(byte_at(cg_sf[sbi >> 2u], sbi));\n");
            b.push_str("            }\n");
            b.push_str("            for (var j = 0u; j < 8u; j = j + 1u) {\n");
            writeln!(
                b,
                "                w4bs[tb + j * {t}u + row] = f16(nvfp4_decode(nvfp4_nibble(w0, j)) * s);"
            )
            .unwrap();
            writeln!(
                b,
                "                w4bs[tb + (j + 8u) * {t}u + row] = f16(nvfp4_decode(nvfp4_nibble(w1, j)) * s);"
            )
            .unwrap();
            b.push_str("            }\n");
        }
        WqFmt::Fp8RowscalePlain => {
            b.push_str("            var s = 0.0;\n");
            b.push_str("            var w0 = 0u;\n");
            b.push_str("            var w1 = 0u;\n");
            b.push_str("            var w2 = 0u;\n");
            b.push_str("            var w3 = 0u;\n");
            b.push_str("            if (gr < nrows) {\n");
            b.push_str("                let base = gr * row_words + (ku16 >> 2u);\n");
            b.push_str("                w0 = cg_w4[base];\n");
            b.push_str("                w1 = cg_w4[base + 1u];\n");
            b.push_str("                w2 = cg_w4[base + 2u];\n");
            b.push_str("                w3 = cg_w4[base + 3u];\n");
            b.push_str("                s = cg_sf[gr];\n");
            b.push_str("            }\n");
            for (wi, off) in [("w0", 0u32), ("w1", 4), ("w2", 8), ("w3", 12)] {
                b.push_str("            for (var j = 0u; j < 4u; j = j + 1u) {\n");
                writeln!(
                    b,
                    "                w4bs[tb + (j + {off}u) * {t}u + row] = \
                     f16(wq16_e4m3_plain(byte_at({wi}, j)) * s);"
                )
                .unwrap();
                b.push_str("            }\n");
            }
        }
    }
    b.push_str("        }\n");
    if act != WqAct::F16 {
        writeln!(
            b,
            "        for (var e = lidx; e < {}u; e = e + {wg}u) {{",
            rows_pb * ku
        )
        .unwrap();
        writeln!(b, "            let u = e / {rows_pb}u;").unwrap();
        writeln!(b, "            let r = e - u * {rows_pb}u;").unwrap();
        b.push_str("            let gm = m0 + r;\n");
        b.push_str("            let ku16 = kt + u * 16u;\n");
        writeln!(b, "            let xb = r * {a_pitch}u + u * 16u;").unwrap();
        match act {
            WqAct::F16 => unreachable!(),
            WqAct::Fp8Rowscale => {
                b.push_str("            var s = 0.0;\n");
                b.push_str("            var w0 = 0u;\n");
                b.push_str("            var w1 = 0u;\n");
                b.push_str("            var w2 = 0u;\n");
                b.push_str("            var w3 = 0u;\n");
                b.push_str("            if (gm < mrows) {\n");
                b.push_str("                let base = gm * xrow_words + (ku16 >> 2u);\n");
                b.push_str("                w0 = cg_x[base];\n");
                b.push_str("                w1 = cg_x[base + 1u];\n");
                b.push_str("                w2 = cg_x[base + 2u];\n");
                b.push_str("                w3 = cg_x[base + 3u];\n");
                b.push_str("                s = cg_xsf[gm];\n");
                b.push_str("            }\n");
                for (wi, off) in [("w0", 0u32), ("w1", 4), ("w2", 8), ("w3", 12)] {
                    b.push_str("            for (var j = 0u; j < 4u; j = j + 1u) {\n");
                    writeln!(
                        b,
                        "                xas[xb + j + {off}u] = \
                         f16(wq16_e4m3_plain(byte_at({wi}, j)) * s);"
                    )
                    .unwrap();
                    b.push_str("            }\n");
                }
            }
            WqAct::Nvfp4Block16 => {
                b.push_str("            var s = 0.0;\n");
                b.push_str("            var w0 = 0u;\n");
                b.push_str("            var w1 = 0u;\n");
                b.push_str("            if (gm < mrows) {\n");
                b.push_str("                let base = gm * xrow_words + (ku16 >> 3u);\n");
                b.push_str("                w0 = cg_x[base];\n");
                b.push_str("                w1 = cg_x[base + 1u];\n");
                b.push_str("                let sbi = gm * xsf_row + (ku16 >> 4u);\n");
                b.push_str("                s = ue4m3_decode(byte_at(cg_xsf[sbi >> 2u], sbi));\n");
                b.push_str("            }\n");
                b.push_str("            for (var j = 0u; j < 8u; j = j + 1u) {\n");
                b.push_str("                xas[xb + j] = f16(nvfp4_decode(nvfp4_nibble(w0, j)) * s);\n");
                b.push_str("                xas[xb + j + 8u] = f16(nvfp4_decode(nvfp4_nibble(w1, j)) * s);\n");
                b.push_str("            }\n");
            }
        }
        b.push_str("        }\n");
    }
    b.push_str("        workgroupBarrier();\n");
    for u in 0..ku {
        for r in 0..tm {
            if act == WqAct::F16 {
                writeln!(
                    b,
                    "        let a{u}_{r} = coopLoadT<CGA>(&cg_x[(m0 + {}u) * kk + kt + {}u], kk);",
                    r * t,
                    u * t
                )
                .unwrap();
            } else {
                writeln!(
                    b,
                    "        let a{u}_{r} = coopLoadT<CGA>(&xas[{}u], {a_pitch}u);",
                    r * t * a_pitch + u * t
                )
                .unwrap();
            }
        }
        for j in 0..tn {
            writeln!(
                b,
                "        let b{u}_{j} = coopLoadT<CGB>(&w4bs[({}u + sgid * {tn}u + {j}u) * {}u], {t}u);",
                u * tiles,
                t * t
            )
            .unwrap();
        }
    }
    for u in 0..ku {
        for r in 0..tm {
            for j in 0..tn {
                writeln!(b, "        c{r}_{j} = coopMultiplyAdd(a{u}_{r}, b{u}_{j}, c{r}_{j});").unwrap();
            }
        }
    }
    b.push_str("    }\n");

    if y_bf16 {
        b.push_str("    let ylane = lidx & 31u;\n");
        for r in 0..tm {
            for j in 0..tn {
                b.push_str("    workgroupBarrier();\n");
                writeln!(
                    b,
                    "    if (m0 + {}u < mrows && n0 + {}u < nrows) {{",
                    r * t,
                    j * t
                )
                .unwrap();
                writeln!(b, "        coopStoreT(c{r}_{j}, &ysh[sgid * 256u], 16u);").unwrap();
                b.push_str("    }\n");
                b.push_str("    workgroupBarrier();\n");
                writeln!(
                    b,
                    "    if (m0 + {}u < mrows && n0 + {}u < nrows) {{",
                    r * t,
                    j * t
                )
                .unwrap();
                b.push_str("        for (var p = ylane; p < 128u; p = p + 32u) {\n");
                b.push_str("            let rr = p >> 3u;\n");
                b.push_str("            let pc = (p & 7u) << 1u;\n");
                b.push_str("            let v0 = ysh[sgid * 256u + rr * 16u + pc];\n");
                b.push_str("            let v1 = ysh[sgid * 256u + rr * 16u + pc + 1u];\n");
                writeln!(
                    b,
                    "            cg_y[((m0 + {}u + rr) * ys + n0 + {}u + pc) >> 1u] = \
                     bf16_encode(v0 * cg_p.alpha) | (bf16_encode(v1 * cg_p.alpha) << 16u);",
                    r * t,
                    j * t
                )
                .unwrap();
                b.push_str("        }\n");
                b.push_str("    }\n");
            }
        }
    } else {
        for r in 0..tm {
            for j in 0..tn {
                writeln!(
                    b,
                    "    let yo{r}_{j} = (m0 + {}u) * ys + n0 + {}u;",
                    r * t,
                    j * t
                )
                .unwrap();
                writeln!(
                    b,
                    "    if (m0 + {}u < mrows && n0 + {}u < nrows) {{",
                    r * t,
                    j * t
                )
                .unwrap();
                writeln!(b, "        coopStoreT(c{r}_{j}, &cg_y[yo{r}_{j}], ys);").unwrap();
                b.push_str("    }\n");
            }
        }
    }
    b.push_str("}\n");

    compose_enabled(&ENABLES, &b)
}

pub fn tiles(m: u32, acc: u32) -> (u32, u32) {
    CoopGemm::new(TILE, Operand::F16).tiles(m, acc)
}

pub fn request(ab: Operand) -> CoopRequest {
    CoopGemm::new(TILE, ab).request()
}

pub fn entry_ab(tm: u32, tn: u32, sg: u32, ku: u32, ab: Operand) -> String {
    CoopGemm::new(TILE, ab).entry(tm, tn, sg, ku)
}

pub fn entry(tm: u32, tn: u32, sg: u32, ku: u32) -> String {
    entry_ab(tm, tn, sg, ku, Operand::F16)
}

pub fn cols_per_workgroup(tn: u32, sg: u32) -> u32 {
    CoopGemm::new(TILE, Operand::F16).cols_per_workgroup(tn, sg)
}

pub fn rows_per_block(tm: u32) -> u32 {
    CoopGemm::new(TILE, Operand::F16).rows_per_block(tm)
}

pub fn source(tm: u32, tn: u32, sg: u32, ku: u32) -> String {
    source_ab(tm, tn, sg, ku, Operand::F16)
}

pub fn source_ab(tm: u32, tn: u32, sg: u32, ku: u32, ab: Operand) -> String {
    CoopGemm::new(TILE, ab).source(tm, tn, sg, ku)
}

pub fn check_shape(m: u32, n: u32, k: u32) -> Result<()> {
    CoopGemm::new(TILE, Operand::F16).check_shape(m, n, k)
}

pub fn grid(m: u32, n: u32, tm: u32, tn: u32, sg: u32) -> (u32, u32) {
    CoopGemm::new(TILE, Operand::F16).grid(m, n, tm, tn, sg)
}

fn preflight(ctx: &WgpuContext) -> Option<String> {
    if !ctx.caps.cooperative_matrix {
        return Some(match &ctx.caps.coop_note {
            Some(n) => format!("EXPERIMENTAL_COOPERATIVE_MATRIX not granted: {n}"),
            None => "EXPERIMENTAL_COOPERATIVE_MATRIX not granted".to_string(),
        });
    }
    if !ctx.caps.shader_f16 {
        return Some("SHADER_F16 not available".to_string());
    }
    ctx.caps.subgroup32_reason()
}

pub fn select(ctx: &WgpuContext, ab: Operand) -> std::result::Result<CoopGemm, String> {
    if let Some(why) = preflight(ctx) {
        return Err(why);
    }
    let mut first: Option<String> = None;
    for tile in TILES {
        let g = CoopGemm::new(tile, ab);
        match ctx.caps.coop_decision(&g.request()) {
            CoopDecision::Compile => return Ok(g),
            CoopDecision::CompileUnadvertised(why) => {
                eprintln!("[coop] {COOP_UNSAFE_SWEEP_ENV} is set, compiling anyway: {why}");
                return Ok(g);
            }
            CoopDecision::Skip(why) => {
                if first.is_none() {
                    first = Some(why);
                }
            }
        }
    }
    Err(first.unwrap_or_else(|| {
        format!(
            "no cooperative-matrix fragment shape in {TILES:?} carries {} operands",
            ab.wgsl()
        )
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiling_fills_the_accumulator_budget() {
        assert_eq!(tiles(8, ACC_FRAGS), (1, 16));
        assert_eq!(tiles(16, ACC_FRAGS), (2, 8));
        assert_eq!(tiles(32, ACC_FRAGS), (4, 4));
        assert_eq!(tiles(64, ACC_FRAGS), (8, 2));
        assert_eq!(tiles(64, 32), (8, 4));

        let g16 = CoopGemm::new(16, Operand::F16);
        assert_eq!(g16.tiles(16, ACC_FRAGS), (1, 16));
        assert_eq!(g16.tiles(32, ACC_FRAGS), (2, 8));
        assert_eq!(g16.tiles(64, ACC_FRAGS), (4, 4));
        assert_eq!(g16.tiles(128, ACC_FRAGS), (8, 2));
        assert_eq!(g16.rows_per_block(4), 64);
        assert_eq!(g16.cols_per_workgroup(4, 4), 256);
        assert_eq!(g16.grid(64, 512, 4, 4, 4), (1, 2));
    }

    #[test]
    fn the_accumulator_budget_is_registers_per_lane_not_fragments() {
        let g8 = CoopGemm::new(8, Operand::F16);
        let g16 = CoopGemm::new(16, Operand::F16);
        for (tm, tn) in [(1u32, 4u32), (2, 8), (4, 4), (8, 4), (4, 8)] {
            assert_eq!(
                g8.acc_fits_a_register_file(tm, tn),
                tm * tn <= 32,
                "at 8x8 the dword budget must reproduce the historical tm*tn<=32 filter exactly \
                 ({tm}x{tn})"
            );
        }
        assert_eq!(g8.acc_lane_dwords(4, 8), 64);
        assert_eq!(g16.acc_lane_dwords(4, 4), 128);
        assert!(g16.acc_fits_a_register_file(4, 4));
        assert_eq!(g16.acc_lane_dwords(2, 16), 256);
        assert!(
            !g16.acc_fits_a_register_file(2, 16),
            "32 accumulator fragments of 16x16 f32 are 256 dwords per lane, past any 255-register \
             file; that config wedged this adapter for >15 minutes on one dispatch"
        );
        assert!(!g16.acc_fits_a_register_file(1, 32));
    }

    #[test]
    fn w4a16_source_parses_and_validates_for_the_swept_configs() {
        for fmt in [WqFmt::Nvfp4Block16, WqFmt::Fp8RowscalePlain] {
            for (tm, tn, sg, ku) in [
                (1u32, 4u32, 1u32, 1u32),
                (2, 2, 2, 1),
                (4, 4, 2, 2),
                (8, 2, 2, 2),
                (8, 1, 1, 4),
                (2, 2, 2, 3),
            ] {
                for (src, entry) in [
                    (
                        source_wq16(fmt, tm, tn, sg, ku),
                        entry_wq16(fmt, tm, tn, sg, ku),
                    ),
                    (
                        source_wq16_act_y16(fmt, WqAct::F16, tm, tn, sg, ku),
                        entry_wq16_act_y16(fmt, WqAct::F16, tm, tn, sg, ku),
                    ),
                ] {
                    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
                        panic!("{fmt:?} tm={tm} tn={tn} sg={sg} ku={ku}: parse: {e}")
                    });
                    naga::valid::Validator::new(
                        naga::valid::ValidationFlags::all(),
                        naga::valid::Capabilities::all(),
                    )
                    .validate(&module)
                    .unwrap_or_else(|e| {
                        panic!("{fmt:?} tm={tm} tn={tn} sg={sg} ku={ku}: validate: {e}")
                    });
                    assert!(src.contains(&entry));
                }
            }
        }
    }

    #[test]
    fn wq16_act_arms_parse_validate_and_leave_the_f16_act_arm_untouched() {
        for fmt in [WqFmt::Nvfp4Block16, WqFmt::Fp8RowscalePlain] {
            assert_eq!(
                source_wq16(fmt, 4, 4, 2, 2),
                source_wq16_act(fmt, WqAct::F16, 4, 4, 2, 2),
                "source_wq16 must stay the F16-act arm byte for byte: other lanes wire it"
            );
            assert_eq!(
                entry_wq16(fmt, 4, 4, 2, 2),
                entry_wq16_act(fmt, WqAct::F16, 4, 4, 2, 2)
            );
            for act in [WqAct::Fp8Rowscale, WqAct::Nvfp4Block16] {
                for (tm, tn, sg, ku) in
                    [(2u32, 2u32, 2u32, 1u32), (4, 4, 2, 2), (8, 2, 2, 2), (8, 1, 1, 4)]
                {
                    let src = source_wq16_act(fmt, act, tm, tn, sg, ku);
                    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
                        panic!("{fmt:?}x{act:?} tm={tm} tn={tn} sg={sg} ku={ku}: parse: {e}")
                    });
                    naga::valid::Validator::new(
                        naga::valid::ValidationFlags::all(),
                        naga::valid::Capabilities::all(),
                    )
                    .validate(&module)
                    .unwrap_or_else(|e| {
                        panic!("{fmt:?}x{act:?} tm={tm} tn={tn} sg={sg} ku={ku}: validate: {e}")
                    });
                    assert!(src.contains(&entry_wq16_act(fmt, act, tm, tn, sg, ku)));
                    assert!(src.contains("var<workgroup> xas"));
                    assert_eq!(
                        src.matches("coopLoadT<CGA>(&xas[").count() as u32,
                        tm * ku,
                        "every A fragment must come from the staged tile, not cg_x"
                    );
                }
            }
        }
        assert_eq!(
            entry_wq16_act(WqFmt::Nvfp4Block16, WqAct::Fp8Rowscale, 2, 2, 2, 1),
            "gemm_coop_w4a8_16_tm2_tn2_sg2_ku1"
        );
        assert_eq!(
            entry_wq16_act(WqFmt::Nvfp4Block16, WqAct::Nvfp4Block16, 2, 2, 2, 1),
            "gemm_coop_w4a4_16_tm2_tn2_sg2_ku1"
        );
        assert_eq!(
            entry_wq16_act(WqFmt::Fp8RowscalePlain, WqAct::Fp8Rowscale, 2, 2, 2, 1),
            "gemm_coop_w8a8_16_tm2_tn2_sg2_ku1"
        );
        assert_eq!(
            entry_wq16(WqFmt::Nvfp4Block16, 2, 2, 2, 1),
            "gemm_coop_w4a16_16_tm2_tn2_sg2_ku1"
        );
    }

    #[test]
    fn the_a_stage_budget_counts_both_tiles_and_the_error_names_the_overflow() {
        assert_eq!(wq16_act_stage_elems(WqAct::F16, 8, 1, 1, 4), 1024);
        assert_eq!(wq16_act_stage_elems(WqAct::Fp8Rowscale, 8, 1, 1, 4), 1024 + 8192);
        assert_eq!(
            wq16_act_stage_elems(WqAct::Nvfp4Block16, 4, 4, 2, 2),
            4096 + 2048
        );
        assert_eq!(wq16_act_stage_elems(WqAct::F16, 8, 3, 8, 4), 24576);
        let _fits = source_wq16_act(WqFmt::Nvfp4Block16, WqAct::F16, 8, 3, 8, 4);
        let r = std::panic::catch_unwind(|| {
            source_wq16_act(WqFmt::Nvfp4Block16, WqAct::Fp8Rowscale, 8, 3, 8, 4)
        });
        assert!(
            r.is_err(),
            "a B stage at exactly the budget plus any A stage must refuse to emit"
        );
    }

    #[test]
    fn source_is_fully_unrolled_and_has_no_indexed_accumulator_array() {
        let src = source(2, 8, 4, 1);
        assert!(
            !src.contains("array<CGC"),
            "accumulators must not be an array"
        );
        assert_eq!(src.matches("coopMultiplyAdd").count(), 16);
        assert_eq!(src.matches("coopLoad<CGB>").count(), 8);
        assert_eq!(src.matches("coopLoadT<CGA>").count(), 2);
        assert!(src.starts_with("enable f16;\nenable wgpu_cooperative_matrix;\n"));
        let u2 = source(4, 4, 2, 2);
        assert_eq!(u2.matches("coopMultiplyAdd").count(), 32);
        assert_eq!(u2.matches("coopLoad<CGB>").count(), 8);

        let s16 = CoopGemm::new(16, Operand::F16).source(2, 8, 4, 1);
        assert!(!s16.contains("array<CGC"));
        assert_eq!(s16.matches("coopMultiplyAdd").count(), 16);
        assert_eq!(s16.matches("coopLoad<CGB>").count(), 8);
        assert_eq!(s16.matches("coopLoadT<CGA>").count(), 2);
        assert!(s16.starts_with("enable f16;\nenable wgpu_cooperative_matrix;\n"));
    }

    #[test]
    fn the_tile_is_the_only_thing_that_changes_between_the_two_fragment_shapes() {
        let s8 = source(2, 4, 2, 2);
        let s16 = CoopGemm::new(16, Operand::F16).source(2, 4, 2, 2);
        assert!(s8.contains("coop_mat8x8<f16, A>"));
        assert!(s16.contains("coop_mat16x16<f16, A>"));
        assert!(s8.contains("coopLoadT<CGC>(&cg_zero[0], 8u)"));
        assert!(s16.contains("coopLoadT<CGC>(&cg_zero[0], 16u)"));
        assert!(s8.contains("kt = kt + 16u"), "8x8 with ku=2 steps K by 16");
        assert!(
            s16.contains("kt = kt + 32u"),
            "16x16 with ku=2 steps K by 32"
        );
        assert_eq!(
            s8.matches("coopMultiplyAdd").count(),
            s16.matches("coopMultiplyAdd").count()
        );
        assert_eq!(CoopGemm::new(8, Operand::F16).zero_elems(), 64);
        assert_eq!(CoopGemm::new(16, Operand::F16).zero_elems(), 256);
    }

    #[test]
    fn a_ragged_extent_is_refused_rather_than_written_past() {
        let g16 = CoopGemm::new(16, Operand::F16);
        g16.check_shape(32, 64, 128).expect("aligned shape");
        for (m, n, k) in [(32, 72, 128), (24, 64, 128), (32, 64, 120)] {
            assert!(
                g16.check_shape(m, n, k).is_err(),
                "16x16 must refuse M={m} N={n} K={k}"
            );
        }
        check_shape(32, 72, 128).expect("72 and 128 are both multiples of 8");
        assert!(check_shape(32, 68, 128).is_err());
    }

    #[test]
    fn the_emitted_source_asks_for_exactly_the_declared_request() {
        use crate::wgpu_backend::qualify::coop_requests_in_wgsl;
        for tile in TILES {
            for ab in [Operand::F16, Operand::F32] {
                let g = CoopGemm::new(tile, ab);
                let src = g.source(2, 8, 4, 1);
                assert_eq!(
                    coop_requests_in_wgsl(&src),
                    vec![g.request()],
                    "{:?} {tile}x{tile} source declares something other than {}",
                    ab,
                    g.request().label()
                );
            }
        }
        assert_eq!(request(Operand::F16).label(), "8x8x8 f16xf16->f32");
        assert_eq!(request(Operand::F32).label(), "8x8x8 f32xf32->f32");
        assert_eq!(
            CoopGemm::new(16, Operand::F16).request().label(),
            "16x16x16 f16xf16->f32"
        );
    }

    #[test]
    fn the_entry_point_name_carries_the_fragment_shape() {
        assert_eq!(entry(2, 8, 4, 1), "gemm_coop_h8_tm2_tn8_sg4_ku1");
        assert_eq!(
            CoopGemm::new(16, Operand::F16).entry(2, 8, 4, 1),
            "gemm_coop_h16_tm2_tn8_sg4_ku1"
        );
        assert!(source(2, 8, 4, 1).contains("fn gemm_coop_h8_tm2_tn8_sg4_ku1("));
        assert!(CoopGemm::new(16, Operand::F16)
            .source(2, 8, 4, 1)
            .contains("fn gemm_coop_h16_tm2_tn8_sg4_ku1("));
    }

    #[test]
    fn every_coop_store_argument_is_a_binding_naga_has_already_emitted() {
        for tile in TILES {
            let src = CoopGemm::new(tile, Operand::F16).source(2, 2, 2, 1);
            let stores: Vec<&str> = src
                .lines()
                .map(str::trim)
                .filter(|l| l.contains("coopStoreT("))
                .collect();
            assert_eq!(stores.len(), 4, "{tile}x{tile}: one store per accumulator");
            for line in stores {
                assert!(
                    line.starts_with("coopStoreT(sc")
                        && line.contains("&cg_y[yo")
                        && line.ends_with(", ys);"),
                    "naga 30 pushes Statement::CooperativeStore without first flushing its emitter \
                     (front/wgsl/lower/mod.rs:3762; atomicStore at :3227 does flush), so any \
                     argument built at the call site is never cached and the SPIR-V backend panics \
                     at back/spv/block.rs:4153/4160. Every coopStore argument must be a let \
                     bound in an earlier statement: {line}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "coop_mat8x8 and coop_mat16x16")]
    fn a_fragment_shape_wgsl_cannot_spell_is_refused_at_construction() {
        CoopGemm::new(32, Operand::F16);
    }
}
