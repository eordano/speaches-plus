#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dispatch;
use crate::wgpu_backend::{compose, Result, WgpuError};
use crate::wgpu_backend::pack::{pack_u16_pairs as pack_u16};

pub const MAX_M: u32 = 9;
pub const WORKGROUP_SIZE: u32 = 256;
pub const ROWS_PER_GROUP: u32 = 8;
pub const LANES_PER_ROW: u32 = 32;

const SCRATCH_BYTES: u32 = WORKGROUP_SIZE * 4;

pub fn small_m_entry(m: u32, vec8: bool) -> String {
    if vec8 {
        format!("gemm_bf16_small_m_vec8_{m}")
    } else {
        format!("gemm_bf16_small_m_scalar_{m}")
    }
}

pub fn small_m_source(m: u32, vec8: bool) -> String {
    use std::fmt::Write as _;
    assert!((1..=MAX_M).contains(&m), "m must be 1..={MAX_M}, got {m}");

    let mut b = String::new();
    b.push_str(
        "struct GemmSmallMParams {\n    n_rows: u32,\n    k_elems: u32,\n    row_words: u32,\n    groups_x: u32,\n};\n\n",
    );
    b.push_str("@group(0) @binding(0) var<storage, read> smk_w: array<u32>;\n");
    b.push_str("@group(0) @binding(1) var<storage, read> smk_x: array<u32>;\n");
    b.push_str("@group(0) @binding(2) var<storage, read_write> smk_y: array<u32>;\n");
    b.push_str("@group(0) @binding(3) var<uniform> smk_params: GemmSmallMParams;\n\n");
    b.push_str("const SMK_LANES: u32 = 32u;\nconst SMK_ROWS: u32 = 8u;\n\n");
    b.push_str("var<workgroup> smk_partial: array<f32, 256>;\n\n");
    b.push_str(
        "fn smk_row(wid: vec3<u32>, warp: u32) -> u32 {\n    return (wid.x + wid.y * smk_params.groups_x) * SMK_ROWS + warp;\n}\n\n",
    );
    b.push_str(
        "fn smk_reduce(tid: u32, lane: u32, acc: f32) -> f32 {\n    workgroupBarrier();\n    smk_partial[tid] = acc;\n    workgroupBarrier();\n    for (var stride = SMK_LANES >> 1u; stride > 0u; stride = stride >> 1u) {\n        if (lane < stride) {\n            smk_partial[tid] = smk_partial[tid] + smk_partial[tid + stride];\n        }\n        workgroupBarrier();\n    }\n    return smk_partial[tid - lane];\n}\n\n",
    );

    writeln!(b, "@compute @workgroup_size({WORKGROUP_SIZE})").unwrap();
    writeln!(b, "fn {}(", small_m_entry(m, vec8)).unwrap();
    b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>\n) {\n");
    b.push_str("    let tid = lid.x;\n");
    b.push_str("    let lane = tid & (SMK_LANES - 1u);\n");
    b.push_str("    let warp = tid / SMK_LANES;\n");
    b.push_str("    let row = smk_row(wid, warp);\n");
    b.push_str("    let live = row < smk_params.n_rows;\n");

    if vec8 {
        b.push_str("    let kv = select(0u, smk_params.k_elems >> 3u, live);\n");
        b.push_str("    let w_base = select(0u, row * smk_params.row_words, live);\n");
    } else {
        b.push_str("    let k_elems = select(0u, smk_params.k_elems, live);\n");
        b.push_str("    let w_base = select(0u, row * smk_params.row_words, live);\n");
    }

    writeln!(b, "    var acc: array<f32, {m}>;").unwrap();
    writeln!(
        b,
        "    for (var mi = 0u; mi < {m}u; mi = mi + 1u) {{ acc[mi] = 0.0; }}"
    )
    .unwrap();

    if vec8 {
        b.push_str("    for (var v = lane; v < kv; v = v + SMK_LANES) {\n");
        b.push_str("        let wo = w_base + (v << 2u);\n");
        b.push_str("        let xo = v << 2u;\n");
        b.push_str("        for (var j = 0u; j < 4u; j = j + 1u) {\n");
        b.push_str("            let ww = smk_w[wo + j];\n");
        writeln!(
            b,
            "            for (var mi = 0u; mi < {m}u; mi = mi + 1u) {{"
        )
        .unwrap();
        b.push_str("                let xw = smk_x[mi * smk_params.row_words + xo + j];\n");
        b.push_str(
            "                acc[mi] = acc[mi] + (bf16_lo(ww) * bf16_lo(xw) + bf16_hi(ww) * bf16_hi(xw));\n",
        );
        b.push_str("            }\n        }\n    }\n");
    } else {
        b.push_str("    for (var k = lane; k < k_elems; k = k + SMK_LANES) {\n");
        b.push_str("        let wv = bf16_decode(u16_at(smk_w[w_base + (k >> 1u)], k));\n");
        writeln!(b, "        for (var mi = 0u; mi < {m}u; mi = mi + 1u) {{").unwrap();
        b.push_str("            let xv = bf16_decode(u16_at(smk_x[mi * smk_params.row_words + (k >> 1u)], k));\n");
        b.push_str("            acc[mi] = acc[mi] + wv * xv;\n");
        b.push_str("        }\n    }\n");
    }

    writeln!(b, "    for (var mi = 0u; mi < {m}u; mi = mi + 1u) {{").unwrap();
    b.push_str("        let total = smk_reduce(tid, lane, acc[mi]);\n");
    b.push_str("        if (lane == 0u && live) {\n            smk_y[mi * smk_params.n_rows + row] = bf16_encode(total);\n        }\n    }\n");
    b.push_str("}\n");

    compose(&b)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmSmallMParams {
    n_rows: u32,
    k_elems: u32,
    row_words: u32,
    groups_x: u32,
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup_and_scratch(ctx, "gemm_bf16_small_m", WORKGROUP_SIZE, SCRATCH_BYTES)
}

fn check_binding(ctx: &WgpuContext, what: &str, bytes: u64) -> Result<()> {
    if bytes > ctx.caps.max_storage_buffer_binding_size {
        return Err(WgpuError::Unsupported(format!(
            "gemm_bf16_small_m {what} needs {bytes} bytes; device allows {} per storage binding",
            ctx.caps.max_storage_buffer_binding_size
        )));
    }
    Ok(())
}

struct Plan {
    w: wgpu::Buffer,
    x: wgpu::Buffer,
    y: wgpu::Buffer,
    params: wgpu::Buffer,
    vec8: bool,
    groups: (u32, u32, u32),
}

impl Plan {
    fn bindings(&self) -> [(u32, &wgpu::Buffer); 4] {
        [(0, &self.w), (1, &self.x), (2, &self.y), (3, &self.params)]
    }
}

fn plan(ctx: &WgpuContext, w: &[u16], x: &[u16], m: usize, n: usize, k: usize) -> Result<Plan> {
    if m == 0 || (m as u32) > MAX_M {
        return Err(WgpuError::Shape(format!(
            "gemm_bf16_small_m supports 1..={MAX_M} rows; got {m}"
        )));
    }
    if !k.is_multiple_of(2) {
        return Err(WgpuError::Shape(format!(
            "gemm_bf16_small_m K must be even so rows start on a u32 word; got {k}"
        )));
    }
    dispatch::check_len("gemm_bf16_small_m w", w.len(), n * k)?;
    dispatch::check_len("gemm_bf16_small_m x", x.len(), m * k)?;
    check_device(ctx)?;

    let row_words = k / 2;
    check_binding(ctx, "w", (n as u64) * (row_words as u64) * 4)?;
    check_binding(ctx, "y", (m as u64) * (n as u64) * 4)?;

    let groups = dispatch::workgroup_count_1d(ctx, n as u64, ROWS_PER_GROUP);
    let params = GemmSmallMParams {
        n_rows: n as u32,
        k_elems: k as u32,
        row_words: row_words as u32,
        groups_x: groups.0,
    };

    Ok(Plan {
        w: dispatch::storage_from_slice(ctx, "gemm-bf16-small-m-w", &pack_u16(w)),
        x: dispatch::storage_from_slice(ctx, "gemm-bf16-small-m-x", &pack_u16(x)),
        y: dispatch::storage_zeroed(ctx, "gemm-bf16-small-m-y", (m * n * 4) as u64),
        params: dispatch::uniform_from(ctx, "gemm-bf16-small-m-params", &params),
        vec8: k.is_multiple_of(8),
        groups,
    })
}

pub fn gemm_bf16_small_m(
    ctx: &WgpuContext,
    w: &[u16],
    x: &[u16],
    y: &mut [u16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    dispatch::check_len("gemm_bf16_small_m y", y.len(), m * n)?;
    if n == 0 || k == 0 || m == 0 {
        return Ok(());
    }
    let p = plan(ctx, w, x, m, n, k)?;
    let entry = small_m_entry(m as u32, p.vec8);
    let source = small_m_source(m as u32, p.vec8);
    dispatch::run(
        ctx,
        "nv_kernels_gemm_bf16_small_m",
        &source,
        &entry,
        &p.bindings(),
        p.groups,
    )?;
    let words: Vec<u32> = dispatch::read_back(ctx, &p.y, m * n)?;
    for (dst, word) in y.iter_mut().zip(words.iter()) {
        *dst = (*word & 0xffff) as u16;
    }
    Ok(())
}

pub fn gemm_bf16_small_m_probe(
    ctx: &WgpuContext,
    w: &[u16],
    x: &[u16],
    m: usize,
    n: usize,
    k: usize,
    warmup: usize,
    iters: usize,
) -> Result<(Vec<u16>, f64)> {
    let p = plan(ctx, w, x, m, n, k)?;
    let entry = small_m_entry(m as u32, p.vec8);
    let source = small_m_source(m as u32, p.vec8);
    let pipeline =
        dispatch::cached_compute_pipeline(ctx, "nv_kernels_gemm_bf16_small_m", &source, &entry)?;
    let group = dispatch::bind_group(ctx, &pipeline, &p.bindings());
    let submit = |count: usize| {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            for _ in 0..count {
                pass.dispatch_workgroups(p.groups.0, p.groups.1, p.groups.2);
            }
        }
        ctx.queue.submit([enc.finish()]);
    };
    submit(warmup.max(1));
    ctx.poll_blocking()?;

    let start = std::time::Instant::now();
    submit(iters);
    ctx.poll_blocking()?;
    let secs = start.elapsed().as_secs_f64();

    let words: Vec<u32> = dispatch::read_back(ctx, &p.y, m * n)?;
    let y = words.iter().map(|word| (*word & 0xffff) as u16).collect();
    Ok((y, secs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_names_are_stable_and_distinct() {
        assert_eq!(small_m_entry(1, true), "gemm_bf16_small_m_vec8_1");
        assert_eq!(small_m_entry(9, false), "gemm_bf16_small_m_scalar_9");
        assert_ne!(small_m_entry(3, true), small_m_entry(3, false));
    }

    #[test]
    fn generated_source_declares_the_requested_entry_only() {
        for m in 1..=MAX_M {
            let src = small_m_source(m, true);
            assert!(src.contains(&small_m_entry(m, true)));
            assert!(src.contains("fn bf16_encode("));
            assert!(src.contains(&format!("array<f32, {m}>")));
        }
    }

    #[test]
    fn scalar_source_uses_u16_at_not_vec4_unpack() {
        let src = small_m_source(3, false);
        assert!(src.contains("u16_at("));
        assert!(src.contains(&small_m_entry(3, false)));
    }

    #[test]
    #[should_panic]
    fn zero_m_is_rejected_by_the_generator() {
        let _ = small_m_source(0, true);
    }

    #[test]
    fn pack_u16_matches_the_flat_row_major_layout() {
        let src: Vec<u16> = vec![1, 2, 3, 4, 5, 6];
        assert_eq!(
            pack_u16(&src),
            vec![0x0002_0001u32, 0x0004_0003, 0x0006_0005]
        );
    }
}
