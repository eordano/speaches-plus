#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::dequant::{bytes_to_words, NVFP4_BLOCK_SIZE};
use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dispatch;
use crate::wgpu_backend::{compose, compose_enabled, Result, WgpuError};

pub const WGSL: &str = include_str!("../../../wgsl/gemm_nvfp4.wgsl");

pub const SECTION_SPLIT: &str = "const NVFP4_GEMM_SECTION_SPLIT: u32 = 1u;";
pub const SCALAR_ENTRY: &str = "gemm_nvfp4_scalar";
pub const COOP_ENTRY: &str = "gemm_nvfp4_coop";
pub const SCALAR_WORKGROUP_SIZE: u32 = 64;
pub const COOP_WORKGROUP_SIZE: u32 = 128;
pub const COOP_TILE: usize = 16;
pub const COOP_BLOCK_M: usize = 32;
pub const COOP_BLOCK_N: usize = 32;

pub const COOP_ENABLES: [&str; 2] = ["f16", "wgpu_cooperative_matrix"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmPath {
    Auto,
    CoopMat,
    Scalar,
}

impl GemmPath {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::CoopMat => "coop_mat",
            Self::Scalar => "scalar",
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmParams {
    alpha: f32,
    m: u32,
    n: u32,
    k: u32,
    row_words: u32,
    k_tiles: u32,
    tiles_n: u32,
    groups_x: u32,
}

fn split_sections() -> (&'static str, &'static str) {
    match WGSL.split_once(SECTION_SPLIT) {
        Some((head, tail)) => (head, tail),
        None => (WGSL, ""),
    }
}

pub fn scalar_source() -> String {
    compose(split_sections().0)
}

pub fn coop_source() -> String {
    compose_enabled(&COOP_ENABLES, split_sections().1)
}

pub fn k_tiles(k_blocks: usize) -> usize {
    k_blocks.div_ceil(4)
}

pub fn swizzled_scale_len(rows: usize, k_blocks: usize) -> usize {
    rows.div_ceil(128) * 128 * k_tiles(k_blocks) * 4
}

fn env_path() -> Option<GemmPath> {
    match std::env::var("NV_KERNELS_WGPU_GEMM")
        .ok()
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("scalar") => Some(GemmPath::Scalar),
        Some("coop") | Some("coop_mat") | Some("coopmat") => Some(GemmPath::CoopMat),
        _ => None,
    }
}

pub fn resolve_path(ctx: &WgpuContext, requested: GemmPath) -> Result<GemmPath> {
    let requested = match requested {
        GemmPath::Auto => env_path().unwrap_or(GemmPath::Auto),
        other => other,
    };
    match requested {
        GemmPath::Scalar => Ok(GemmPath::Scalar),
        GemmPath::CoopMat => match ctx.caps.coop_gemm_reason() {
            None => Ok(GemmPath::CoopMat),
            Some(why) => Err(WgpuError::Unsupported(format!("coop_mat gemm: {why}"))),
        },
        GemmPath::Auto => {
            if ctx.caps.coop_gemm_tile().is_some() {
                Ok(GemmPath::CoopMat)
            } else {
                Ok(GemmPath::Scalar)
            }
        }
    }
}

pub fn nvfp4_gemm_bf16(
    ctx: &WgpuContext,
    a_packed: &[u8],
    a_scales: &[u8],
    b_packed: &[u8],
    b_scales: &[u8],
    global_scale: f32,
    d: &mut [u16],
    m: usize,
    n: usize,
    k: usize,
    path: GemmPath,
) -> Result<GemmPath> {
    let path = resolve_path(ctx, path)?;
    if m == 0 || n == 0 || k == 0 {
        return Ok(path);
    }
    if !k.is_multiple_of(NVFP4_BLOCK_SIZE) {
        return Err(WgpuError::Shape(format!(
            "K {k} is not a multiple of {NVFP4_BLOCK_SIZE}"
        )));
    }
    let k_blocks = k / NVFP4_BLOCK_SIZE;
    let row_words = k / 8;
    dispatch::check_len("nvfp4 gemm a_packed", a_packed.len(), m * k / 2)?;
    dispatch::check_len("nvfp4 gemm b_packed", b_packed.len(), n * k / 2)?;
    dispatch::check_len("nvfp4 gemm d", d.len(), m * n)?;
    for (what, got, rows) in [
        ("a_scales", a_scales.len(), m),
        ("b_scales", b_scales.len(), n),
    ] {
        let want = swizzled_scale_len(rows, k_blocks);
        if got < want {
            return Err(WgpuError::Shape(format!(
                "nvfp4 gemm {what}: got {got} want at least {want}"
            )));
        }
    }

    let a_words = bytes_to_words(a_packed);
    let a_sf_words = bytes_to_words(a_scales);
    let b_words = bytes_to_words(b_packed);
    let b_sf_words = bytes_to_words(b_scales);

    let a_buf = dispatch::storage_from_slice(ctx, "nvfp4-gemm-a", &a_words);
    let a_sf_buf = dispatch::storage_from_slice(ctx, "nvfp4-gemm-a-sf", &a_sf_words);
    let b_buf = dispatch::storage_from_slice(ctx, "nvfp4-gemm-b", &b_words);
    let b_sf_buf = dispatch::storage_from_slice(ctx, "nvfp4-gemm-b-sf", &b_sf_words);
    let d_buf = dispatch::storage_zeroed(ctx, "nvfp4-gemm-d", (m * n * 4) as u64);

    let blocks_n = n.div_ceil(COOP_BLOCK_N);
    let (groups, source, entry, label, tiles_n) = match path {
        GemmPath::Scalar | GemmPath::Auto => (
            dispatch::workgroup_count_1d(ctx, (m * n) as u64, SCALAR_WORKGROUP_SIZE),
            scalar_source(),
            SCALAR_ENTRY,
            "nvfp4-gemm-scalar",
            n.div_ceil(COOP_TILE),
        ),
        GemmPath::CoopMat => (
            dispatch::workgroup_count_1d(ctx, (m.div_ceil(COOP_BLOCK_M) * blocks_n) as u64, 1),
            coop_source(),
            COOP_ENTRY,
            "nvfp4-gemm-coop",
            blocks_n,
        ),
    };

    let params = GemmParams {
        alpha: global_scale,
        m: m as u32,
        n: n as u32,
        k: k as u32,
        row_words: row_words as u32,
        k_tiles: k_tiles(k_blocks) as u32,
        tiles_n: tiles_n as u32,
        groups_x: groups.0,
    };
    let params_buf = dispatch::uniform_from(ctx, "nvfp4-gemm-params", &params);

    dispatch::run(
        ctx,
        label,
        &source,
        entry,
        &[
            (0, &a_buf),
            (1, &a_sf_buf),
            (2, &b_buf),
            (3, &b_sf_buf),
            (4, &params_buf),
            (5, &d_buf),
        ],
        groups,
    )?;

    let words: Vec<u32> = dispatch::read_back(ctx, &d_buf, m * n)?;
    for (dst, w) in d.iter_mut().zip(words.iter()) {
        *dst = (*w & 0xffff) as u16;
    }
    Ok(path)
}

#[doc(hidden)]
pub fn bench_transfer_floor(
    ctx: &WgpuContext,
    a_packed: &[u8],
    a_scales: &[u8],
    b_packed: &[u8],
    b_scales: &[u8],
    m: usize,
    n: usize,
) -> Result<usize> {
    let a_words = bytes_to_words(a_packed);
    let a_sf_words = bytes_to_words(a_scales);
    let b_words = bytes_to_words(b_packed);
    let b_sf_words = bytes_to_words(b_scales);
    let _a = dispatch::storage_from_slice(ctx, "floor-a", &a_words);
    let _asf = dispatch::storage_from_slice(ctx, "floor-a-sf", &a_sf_words);
    let _b = dispatch::storage_from_slice(ctx, "floor-b", &b_words);
    let _bsf = dispatch::storage_from_slice(ctx, "floor-b-sf", &b_sf_words);
    let d = dispatch::storage_zeroed(ctx, "floor-d", (m * n * 4) as u64);
    let words: Vec<u32> = dispatch::read_back(ctx, &d, m * n)?;
    Ok(words.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_split_and_carry_their_entry_points() {
        let (head, tail) = split_sections();
        assert!(head.contains(SCALAR_ENTRY));
        assert!(!head.contains(COOP_ENTRY));
        assert!(tail.contains(COOP_ENTRY));
        assert!(!tail.contains(SCALAR_ENTRY));
    }

    #[test]
    fn only_the_coop_source_carries_the_enable_directives() {
        let coop = coop_source();
        assert!(coop.starts_with("enable f16;\nenable wgpu_cooperative_matrix;\n"));
        assert!(coop.contains("coopMultiplyAdd("));
        let scalar = scalar_source();
        assert!(!scalar.contains("enable "));
        assert!(!scalar.contains("coopMultiplyAdd("));
    }

    #[test]
    fn the_shipped_coop_shader_asks_for_16x16x16_f16_f32_and_nothing_else() {
        use crate::wgpu_backend::qualify::{coop_requests_in_wgsl, CoopRequest, CoopScalar};
        assert_eq!(
            coop_requests_in_wgsl(&coop_source()),
            vec![CoopRequest::square(16, CoopScalar::F16, CoopScalar::F32)]
        );
        assert!(coop_requests_in_wgsl(&scalar_source()).is_empty());
    }

    #[test]
    fn both_sources_include_the_dequant_prelude() {
        assert!(scalar_source().contains("fn nvfp4_scale_byte_index("));
        assert!(coop_source().contains("fn nvfp4_scale_byte_index("));
    }

    #[test]
    fn the_coop_k_step_is_exactly_one_nvfp4_block() {
        assert_eq!(COOP_TILE, NVFP4_BLOCK_SIZE);
        assert!(coop_source().contains("coopMultiplyAdd(ta, tb, zero)"));
        assert!(coop_source().contains("acc[s] = acc[s] + coop_c[lidx + s * COOP_WG]"));
        assert!(scalar_source().contains("acc = acc + block_dot"));
    }

    #[test]
    fn swizzled_scale_len_matches_the_nv_quant_layout() {
        assert_eq!(swizzled_scale_len(128, 8), 128 * 2 * 4);
        assert_eq!(swizzled_scale_len(129, 4), 256 * 4);
    }
}
