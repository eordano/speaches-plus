#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::kernels::gemv_w4a16;
use crate::wgpu_backend::kernels::gemv_w4a16::ScaleGrain;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};
use crate::wgpu_backend::pack::{pack_u16_pairs as pack_u16};

pub const MIN_M: u32 = 1;
pub const MAX_M: u32 = 9;
pub const WORKGROUP_SIZE: u32 = 256;
pub const LANES_PER_ROW: u32 = 32;
pub const ROWS_PER_GROUP: u32 = 8;

const SCRATCH_BYTES: u32 = WORKGROUP_SIZE * 4;

pub const GRAIN: ScaleGrain = ScaleGrain::Ge32;

fn check_grain(group_size: usize) -> Result<()> {
    if !GRAIN.accepts(group_size) {
        return Err(WgpuError::Unsupported(format!(
            "gemm_w4a16_small_m is built for {GRAIN:?} and would silently apply one scale to 32 weights at GS={group_size}; use the sg_mk M-row twin, which has a G16 grain"
        )));
    }
    Ok(())
}

fn entry_name(m: u32) -> String {
    format!("gemm_w4a16_small_m_m{m}")
}

pub fn entry_for(m: u32) -> Result<String> {
    if (MIN_M..=MAX_M).contains(&m) {
        Ok(entry_name(m))
    } else {
        Err(WgpuError::Unsupported(format!(
            "gemm_w4a16_small_m has no entry point for M={m}; supported range is {MIN_M}..={MAX_M}"
        )))
    }
}

const COMPONENTS: [&str; 4] = ["x", "y", "z", "w"];

fn write_preamble(b: &mut String) {
    b.push_str(
        "struct GemmW4A16SmallMParams {\n    n_rows: u32,\n    k_elems: u32,\n    gs: u32,\n    w_row_words: u32,\n    scale_row_stride: u32,\n    groups_x: u32,\n    x_stride_words: u32,\n    y_stride_words: u32,\n};\n\n",
    );
    b.push_str("@group(0) @binding(0) var<storage, read> sm_packed4: array<vec4<u32>>;\n");
    b.push_str("@group(0) @binding(1) var<storage, read> sm_scale: array<u32>;\n");
    b.push_str("@group(0) @binding(2) var<storage, read> sm_x4: array<vec4<u32>>;\n");
    b.push_str("@group(0) @binding(3) var<storage, read_write> sm_y: array<u32>;\n");
    b.push_str("@group(0) @binding(4) var<uniform> sm_params: GemmW4A16SmallMParams;\n\n");
    b.push_str("const SM_LANES: u32 = 32u;\n");
    b.push_str("const SM_ROWS: u32 = 8u;\n\n");
    b.push_str("var<workgroup> sm_partial: array<f32, 256>;\n\n");
}

fn write_entry(b: &mut String, m: u32) {
    use std::fmt::Write as _;
    let entry = entry_name(m);
    writeln!(b, "@compute @workgroup_size({WORKGROUP_SIZE})").unwrap();
    writeln!(b, "fn {entry}(").unwrap();
    b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n");
    b.push_str("    @builtin(local_invocation_id) lid: vec3<u32>\n) {\n");
    b.push_str("    let tid = lid.x;\n");
    b.push_str("    let lane = tid & (SM_LANES - 1u);\n");
    b.push_str("    let warp = tid / SM_LANES;\n");
    b.push_str("    let row = (wid.x + wid.y * sm_params.groups_x) * SM_ROWS + warp;\n");
    b.push_str("    let live = row < sm_params.n_rows;\n");
    b.push_str("    let kv = select(0u, sm_params.k_elems >> 5u, live);\n");
    b.push_str("    let wbase4 = select(0u, row * (sm_params.w_row_words >> 2u), live);\n");
    b.push_str("    let sbase = select(0u, row * sm_params.scale_row_stride, live);\n");
    b.push_str("    let xs4 = sm_params.x_stride_words >> 2u;\n\n");
    writeln!(b, "    var acc: array<f32, {m}>;").unwrap();
    writeln!(b, "    for (var t = 0u; t < {m}u; t = t + 1u) {{").unwrap();
    b.push_str("        acc[t] = 0.0;\n    }\n\n");
    b.push_str("    for (var v = lane; v < kv; v = v + SM_LANES) {\n");
    b.push_str("        let wv = sm_packed4[wbase4 + v];\n");
    b.push_str("        let sc = bf16_decode(sm_scale[sbase + ((v << 5u) / sm_params.gs)]);\n\n");
    for (c, comp) in COMPONENTS.iter().enumerate() {
        writeln!(
            b,
            "        let qe{c} = vec4<f32>(unpack4xU8(wv.{comp} & 0x0f0f0f0fu)) - vec4<f32>(8.0);"
        )
        .unwrap();
        writeln!(
            b,
            "        let qo{c} = vec4<f32>(unpack4xU8((wv.{comp} >> 4u) & 0x0f0f0f0fu)) - vec4<f32>(8.0);"
        )
        .unwrap();
    }
    b.push_str("\n        let xb = v << 2u;\n");
    writeln!(b, "        for (var t = 0u; t < {m}u; t = t + 1u) {{").unwrap();
    b.push_str("            let xbase = t * xs4 + xb;\n");
    for c in 0..4u32 {
        writeln!(b, "            let xw{c} = sm_x4[xbase + {c}u];").unwrap();
    }
    for c in 0..4u32 {
        writeln!(
            b,
            "            let xe{c} = bitcast<vec4<f32>>(xw{c} << vec4<u32>(16u));"
        )
        .unwrap();
        writeln!(
            b,
            "            let xo{c} = bitcast<vec4<f32>>(xw{c} & vec4<u32>(0xffff0000u));"
        )
        .unwrap();
    }
    b.push_str("\n            var a = 0.0;\n");
    for c in 0..4u32 {
        for sub in COMPONENTS {
            writeln!(b, "            a = fma(qe{c}.{sub}, xe{c}.{sub}, a);").unwrap();
            writeln!(b, "            a = fma(qo{c}.{sub}, xo{c}.{sub}, a);").unwrap();
        }
    }
    b.push_str("\n            acc[t] = fma(sc, a, acc[t]);\n");
    b.push_str("        }\n");
    b.push_str("    }\n\n");
    writeln!(b, "    for (var t = 0u; t < {m}u; t = t + 1u) {{").unwrap();
    b.push_str("        sm_partial[tid] = acc[t];\n");
    b.push_str("        workgroupBarrier();\n");
    b.push_str("        for (var stride = SM_LANES >> 1u; stride > 0u; stride = stride >> 1u) {\n");
    b.push_str("            if (lane < stride) {\n");
    b.push_str("                sm_partial[tid] = sm_partial[tid] + sm_partial[tid + stride];\n");
    b.push_str("            }\n");
    b.push_str("            workgroupBarrier();\n");
    b.push_str("        }\n");
    b.push_str("        if (lane == 0u && live) {\n");
    b.push_str(
        "            sm_y[t * sm_params.y_stride_words + row] = bf16_encode(sm_partial[tid]);\n",
    );
    b.push_str("        }\n");
    b.push_str("        workgroupBarrier();\n");
    b.push_str("    }\n");
    b.push_str("}\n\n");
}

pub fn small_m_source() -> String {
    let mut b = String::new();
    write_preamble(&mut b);
    for m in MIN_M..=MAX_M {
        write_entry(&mut b, m);
    }
    compose(&b)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_rows: u32,
    k_elems: u32,
    gs: u32,
    w_row_words: u32,
    scale_row_stride: u32,
    groups_x: u32,
    x_stride_words: u32,
    y_stride_words: u32,
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup_and_scratch(ctx, "gemm_w4a16_small_m", WORKGROUP_SIZE, SCRATCH_BYTES)
}

fn check_binding(ctx: &WgpuContext, what: &str, bytes: u64) -> Result<()> {
    if bytes > ctx.caps.max_storage_buffer_binding_size {
        return Err(WgpuError::Unsupported(format!(
            "gemm_w4a16_small_m {what} needs {bytes} bytes; device allows {} per storage binding",
            ctx.caps.max_storage_buffer_binding_size
        )));
    }
    Ok(())
}

fn widen_u16(src: &[u16]) -> Vec<u32> {
    src.iter().map(|v| *v as u32).collect()
}

pub fn gemm_w4a16_small_m(
    ctx: &WgpuContext,
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    y: &mut [u16],
    n: usize,
    k: usize,
    group_size: usize,
    m: usize,
) -> Result<()> {
    if n == 0 || k == 0 || m == 0 {
        return Ok(());
    }
    let m_u32 = m as u32;
    let entry = entry_for(m_u32)?;
    gemv_w4a16::shape_rule(k, group_size)?;
    check_grain(group_size)?;
    if !k.is_multiple_of(4) {
        return Err(WgpuError::Shape(format!(
            "gemm_w4a16_small_m requires K%4==0 for vec4 packing; got K={k}"
        )));
    }
    dispatch::check_len("gemm_w4a16_small_m packed", packed.len(), n * (k / 8))?;
    let groups_per_row = k / group_size;
    dispatch::check_len("gemm_w4a16_small_m scale", scales.len(), n * groups_per_row)?;
    dispatch::check_len("gemm_w4a16_small_m x", x.len(), m * k)?;
    dispatch::check_len("gemm_w4a16_small_m y", y.len(), m * n)?;
    check_device(ctx)?;
    check_binding(ctx, "packed", (n as u64) * (k as u64 / 8) * 4)?;
    check_binding(ctx, "scale", (n as u64) * (groups_per_row as u64) * 4)?;
    check_binding(ctx, "x", (m as u64) * (k as u64 / 2) * 4)?;
    check_binding(ctx, "y", (m as u64) * (n as u64) * 4)?;

    let groups = dispatch::workgroup_count_1d(ctx, n as u64, ROWS_PER_GROUP);
    let params = Params {
        n_rows: n as u32,
        k_elems: k as u32,
        gs: group_size as u32,
        w_row_words: (k / 8) as u32,
        scale_row_stride: groups_per_row as u32,
        groups_x: groups.0,
        x_stride_words: (k / 2) as u32,
        y_stride_words: n as u32,
    };

    let packed_buf = dispatch::storage_from_slice(ctx, "gemm-w4a16-small-m-packed", packed);
    let scale_buf =
        dispatch::storage_from_slice(ctx, "gemm-w4a16-small-m-scale", &widen_u16(scales));
    let x_buf = dispatch::storage_from_slice(ctx, "gemm-w4a16-small-m-x", &pack_u16(x));
    let y_buf = dispatch::storage_zeroed(ctx, "gemm-w4a16-small-m-y", ((m * n) as u64) * 4);
    let params_buf = dispatch::uniform_from(ctx, "gemm-w4a16-small-m-params", &params);

    dispatch::run(
        ctx,
        "nv_kernels_gemm_w4a16_small_m",
        &small_m_source(),
        &entry,
        &[
            (0, &packed_buf),
            (1, &scale_buf),
            (2, &x_buf),
            (3, &y_buf),
            (4, &params_buf),
        ],
        groups,
    )?;

    let words: Vec<u32> = dispatch::read_back(ctx, &y_buf, m * n)?;
    for (dst, word) in y.iter_mut().zip(words.iter()) {
        *dst = (*word & 0xffff) as u16;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_for_covers_the_supported_range() {
        assert_eq!(entry_for(1).unwrap(), "gemm_w4a16_small_m_m1");
        assert_eq!(entry_for(9).unwrap(), "gemm_w4a16_small_m_m9");
        assert!(entry_for(0).is_err());
        assert!(entry_for(10).is_err());
    }

    #[test]
    fn small_m_source_declares_every_entry_point() {
        let src = small_m_source();
        for m in MIN_M..=MAX_M {
            assert!(src.contains(&entry_name(m)), "missing entry for M={m}");
        }
        assert!(src.contains("unpack4xU8"));
        assert!(src.contains("fn bf16_encode("));
        assert!(src.contains("fn bf16_decode("));
    }

    #[test]
    fn the_grain_guard_refuses_every_group_size_the_body_cannot_express() {
        assert!(gemv_w4a16::shape_rule(5376, 16).is_ok());
        assert!(check_grain(16).is_err());
        assert!(check_grain(8).is_err());
        assert!(check_grain(32).is_ok());
        assert!(check_grain(64).is_ok());
        assert!(check_grain(128).is_ok());
        assert!(small_m_source().contains("(v << 5u) / sm_params.gs"));
    }

    #[test]
    fn params_are_uniform_buffer_sized() {
        assert_eq!(std::mem::size_of::<Params>() % 4, 0);
    }
}
