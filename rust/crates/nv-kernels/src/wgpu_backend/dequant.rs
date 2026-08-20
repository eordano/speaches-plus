use super::device::WgpuContext;
use super::dispatch;
use super::{Result, WgpuError};

pub const DEQUANT_WGSL: &str = include_str!("../../wgsl/dequant.wgsl");

pub const NVFP4_BLOCK_SIZE: usize = 16;

pub fn compose(body: &str) -> String {
    format!("{DEQUANT_WGSL}\n{body}\n")
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DequantParams {
    pub f0: f32,
    pub f1: f32,
    pub u0: u32,
    pub u1: u32,
}

pub fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    crate::wgpu_backend::pack::pack_u8_min_one_word(bytes)
}

const MAP_HEADER: &str = "\
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
";

const ENCODE_HEADER: &str = "\
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
";

fn map_shader(expr: &str) -> String {
    compose(&format!(
        "{MAP_HEADER}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&dst)) {{ return; }}
    dst[i] = {expr};
}}
"
    ))
}

fn run_map(ctx: &WgpuContext, label: &str, expr: &str, src: &[u32]) -> Result<Vec<f32>> {
    if src.is_empty() {
        return Ok(Vec::new());
    }
    let source = map_shader(expr);
    let src_buf = dispatch::storage_from_slice(ctx, "dequant-src", src);
    let dst_buf = dispatch::storage_zeroed(ctx, "dequant-dst", (src.len() * 4) as u64);
    let groups = dispatch::workgroup_count_1d(ctx, src.len() as u64, 64);
    dispatch::run(
        ctx,
        label,
        &source,
        "main",
        &[(0, &src_buf), (1, &dst_buf)],
        groups,
    )?;
    dispatch::read_back(ctx, &dst_buf, src.len())
}

pub fn gpu_decode_e2m1(ctx: &WgpuContext, nibbles: &[u32]) -> Result<Vec<f32>> {
    run_map(ctx, "e2m1-table", "nvfp4_decode(src[i])", nibbles)
}

pub fn gpu_decode_e2m1_arith(ctx: &WgpuContext, nibbles: &[u32]) -> Result<Vec<f32>> {
    run_map(ctx, "e2m1-arith", "nvfp4_decode_arith(src[i])", nibbles)
}

pub fn gpu_decode_ue4m3(ctx: &WgpuContext, bytes: &[u32]) -> Result<Vec<f32>> {
    run_map(ctx, "ue4m3", "ue4m3_decode(src[i])", bytes)
}

pub fn gpu_decode_e4m3(ctx: &WgpuContext, bytes: &[u32]) -> Result<Vec<f32>> {
    run_map(ctx, "e4m3", "e4m3_decode(src[i])", bytes)
}

pub fn gpu_decode_e5m2(ctx: &WgpuContext, bytes: &[u32]) -> Result<Vec<f32>> {
    run_map(ctx, "e5m2", "e5m2_decode(src[i])", bytes)
}

pub fn gpu_decode_bf16(ctx: &WgpuContext, bits: &[u32]) -> Result<Vec<f32>> {
    run_map(ctx, "bf16-decode", "bf16_decode(src[i])", bits)
}

pub fn gpu_decode_bf16_pairs(ctx: &WgpuContext, words: &[u32]) -> Result<Vec<f32>> {
    if words.is_empty() {
        return Ok(Vec::new());
    }
    let source = compose(
        "\
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= arrayLength(&src)) { return; }
    let word = src[w];
    dst[w * 2u] = bf16_lo(word);
    dst[w * 2u + 1u] = bf16_hi(word);
}
",
    );
    let src_buf = dispatch::storage_from_slice(ctx, "bf16-pairs-src", words);
    let dst_buf = dispatch::storage_zeroed(ctx, "bf16-pairs-dst", (words.len() * 8) as u64);
    let groups = dispatch::workgroup_count_1d(ctx, words.len() as u64, 64);
    dispatch::run(
        ctx,
        "bf16-pairs",
        &source,
        "main",
        &[(0, &src_buf), (1, &dst_buf)],
        groups,
    )?;
    dispatch::read_back(ctx, &dst_buf, words.len() * 2)
}

pub fn gpu_encode_bf16(ctx: &WgpuContext, values: &[f32]) -> Result<Vec<u32>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let source = compose(&format!(
        "{ENCODE_HEADER}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&dst)) {{ return; }}
    dst[i] = bf16_encode(src[i]);
}}
"
    ));
    let src_buf = dispatch::storage_from_slice(ctx, "bf16-encode-src", values);
    let dst_buf = dispatch::storage_zeroed(ctx, "bf16-encode-dst", (values.len() * 4) as u64);
    let groups = dispatch::workgroup_count_1d(ctx, values.len() as u64, 64);
    dispatch::run(
        ctx,
        "bf16-encode",
        &source,
        "main",
        &[(0, &src_buf), (1, &dst_buf)],
        groups,
    )?;
    dispatch::read_back(ctx, &dst_buf, values.len())
}

pub fn gpu_dequantize_nvfp4(
    ctx: &WgpuContext,
    packed: &[u8],
    scales: &[u8],
    n_values: usize,
    global_scale: f32,
) -> Result<Vec<f32>> {
    if n_values == 0 {
        return Ok(Vec::new());
    }
    if !n_values.is_multiple_of(NVFP4_BLOCK_SIZE) {
        return Err(WgpuError::Shape(format!(
            "n_values {n_values} is not a multiple of {NVFP4_BLOCK_SIZE}"
        )));
    }
    dispatch::check_len("nvfp4 packed", packed.len(), n_values / 2)?;
    dispatch::check_len("nvfp4 scales", scales.len(), n_values / NVFP4_BLOCK_SIZE)?;
    let source = compose(
        "\
struct DequantParams { f0: f32, f1: f32, u0: u32, u1: u32 };

@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<uniform> params: DequantParams;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.u0) { return; }
    let block = i / NVFP4_BLOCK_SIZE;
    let nib = nvfp4_nibble(packed[i >> 3u], i);
    let scale_byte = byte_at(scales[block >> 2u], block);
    dst[i] = nvfp4_value_global(nib, scale_byte, params.f0);
}
",
    );
    let packed_words = bytes_to_words(packed);
    let scale_words = bytes_to_words(scales);
    let packed_buf = dispatch::storage_from_slice(ctx, "nvfp4-packed", &packed_words);
    let scale_buf = dispatch::storage_from_slice(ctx, "nvfp4-scales", &scale_words);
    let params = DequantParams {
        f0: global_scale,
        f1: 0.0,
        u0: n_values as u32,
        u1: 0,
    };
    let params_buf = dispatch::uniform_from(ctx, "nvfp4-params", &params);
    let dst_buf = dispatch::storage_zeroed(ctx, "nvfp4-dst", (n_values * 4) as u64);
    let groups = dispatch::workgroup_count_1d(ctx, n_values as u64, 64);
    dispatch::run(
        ctx,
        "nvfp4-dequantize",
        &source,
        "main",
        &[
            (0, &packed_buf),
            (1, &scale_buf),
            (2, &params_buf),
            (3, &dst_buf),
        ],
        groups,
    )?;
    dispatch::read_back(ctx, &dst_buf, n_values)
}

pub fn gpu_decode_int4_group(
    ctx: &WgpuContext,
    packed: &[u32],
    n_values: usize,
    group_scale: f32,
    zero_point: f32,
) -> Result<Vec<f32>> {
    if n_values == 0 {
        return Ok(Vec::new());
    }
    dispatch::check_len("int4 packed", packed.len(), n_values.div_ceil(8))?;
    let source = compose(
        "\
struct DequantParams { f0: f32, f1: f32, u0: u32, u1: u32 };

@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<uniform> params: DequantParams;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.u0) { return; }
    dst[i] = int4_decode(packed[i >> 3u], i, params.f0, params.f1);
}
",
    );
    let packed_buf = dispatch::storage_from_slice(ctx, "int4-packed", packed);
    let params = DequantParams {
        f0: group_scale,
        f1: zero_point,
        u0: n_values as u32,
        u1: 0,
    };
    let params_buf = dispatch::uniform_from(ctx, "int4-params", &params);
    let dst_buf = dispatch::storage_zeroed(ctx, "int4-dst", (n_values * 4) as u64);
    let groups = dispatch::workgroup_count_1d(ctx, n_values as u64, 64);
    dispatch::run(
        ctx,
        "int4-decode",
        &source,
        "main",
        &[(0, &packed_buf), (1, &params_buf), (2, &dst_buf)],
        groups,
    )?;
    dispatch::read_back(ctx, &dst_buf, n_values)
}

pub fn gpu_nvfp4_scale_swizzle_index(
    ctx: &WgpuContext,
    rows: usize,
    k_blocks: usize,
) -> Result<Vec<u32>> {
    let n = rows * k_blocks;
    if n == 0 {
        return Ok(Vec::new());
    }
    let source = compose(
        "\
struct DequantParams { f0: f32, f1: f32, u0: u32, u1: u32 };

@group(0) @binding(0) var<uniform> params: DequantParams;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&dst)) { return; }
    let row = i / params.u1;
    let block = i % params.u1;
    dst[i] = nvfp4_scale_byte_index(row, block, nvfp4_k_tiles(params.u1));
}
",
    );
    let params = DequantParams {
        f0: 0.0,
        f1: 0.0,
        u0: rows as u32,
        u1: k_blocks as u32,
    };
    let params_buf = dispatch::uniform_from(ctx, "swizzle-params", &params);
    let dst_buf = dispatch::storage_zeroed(ctx, "swizzle-dst", (n * 4) as u64);
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, 64);
    dispatch::run(
        ctx,
        "nvfp4-swizzle-index",
        &source,
        "main",
        &[(0, &params_buf), (1, &dst_buf)],
        groups,
    )?;
    dispatch::read_back(ctx, &dst_buf, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_to_words_is_little_endian() {
        assert_eq!(bytes_to_words(&[0x11, 0x22, 0x33, 0x44]), vec![0x44332211]);
        assert_eq!(bytes_to_words(&[0xff]), vec![0x000000ff]);
        assert_eq!(bytes_to_words(&[]), vec![0u32]);
    }

    #[test]
    fn dequant_wgsl_declares_the_primitives() {
        for f in [
            "fn nvfp4_decode(",
            "fn ue4m3_decode(",
            "fn e4m3_decode(",
            "fn e5m2_decode(",
            "fn bf16_decode(",
            "fn bf16_encode(",
            "fn int4_decode(",
            "fn nvfp4_value_global(",
            "fn nvfp4_scale_byte_index(",
        ] {
            assert!(DEQUANT_WGSL.contains(f), "missing {f}");
        }
    }

    #[test]
    fn composed_source_keeps_the_prelude_first() {
        let s = compose("fn body() -> f32 { return 1.0; }");
        assert!(s.starts_with("const NVFP4_BLOCK_SIZE"));
        assert!(s.ends_with("\nfn body() -> f32 { return 1.0; }\n"));
        assert!(s.contains("fn nvfp4_decode("));
    }
}
