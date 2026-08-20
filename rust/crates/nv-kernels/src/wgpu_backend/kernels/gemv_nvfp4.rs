#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::dequant::{bytes_to_words, NVFP4_BLOCK_SIZE};
use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dispatch;
use crate::wgpu_backend::qualify::GemmStrategy;
use crate::wgpu_backend::{compose, Result, WgpuError};

pub const WGSL: &str = include_str!("../../../wgsl/gemv_nvfp4.wgsl");

#[path = "gemv_nvfp4_lin.rs"]
pub mod lin;

pub const DECODE_BEGIN: &str = "const NVFP4_GEMV_DECODE_BEGIN: u32 = 0u;";
pub const DECODE_END: &str = "const NVFP4_GEMV_DECODE_END: u32 = 0u;";

pub const SECTION_SPLIT: &str = "const NVFP4_GEMV_SECTION_SPLIT: u32 = 1u;";
pub const SECTION_SG: &str = "const NVFP4_GEMV_SECTION_SG: u32 = 2u;";
pub const SECTION_SGW: &str = "const NVFP4_GEMV_SECTION_SGW: u32 = 3u;";
pub const GEMV_ENTRY: &str = "gemv_nvfp4_bf16";
pub const SG_GEMV_ENTRY: &str = "gemv_nvfp4_bf16_sg";
pub const SGU_GEMV_ENTRY: &str = "gemv_nvfp4_bf16_sgu";
pub const SGW_GEMV_ENTRY: &str = "gemv_nvfp4_bf16_sgw";
pub const SGQ_GEMV_ENTRY: &str = "gemv_nvfp4_bf16_sgq";
pub const QUANTIZE_ENTRY: &str = "quantize_row_nvfp4_bf16";
pub const WORKGROUP_SIZE: u32 = 256;
pub const SG_WORKGROUP_SIZE: u32 = 128;
pub const SG_ROWS_PER_GROUP: u32 = 4;
pub const SGW_VWARPS: u32 = 8;
pub const SGW_TILE_BLOCKS: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GemvParams {
    pub alpha: f32,
    pub n_rows: u32,
    pub k_blocks: u32,
    pub k_tiles: u32,
    pub w_row_words: u32,
    pub groups_x: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct QuantParams {
    global_scale: f32,
    k_blocks: u32,
    pad0: u32,
    pad1: u32,
}

fn split_sections() -> (&'static str, &'static str, &'static str, &'static str) {
    let (head, rest) = match WGSL.split_once(SECTION_SPLIT) {
        Some(pair) => pair,
        None => (WGSL, ""),
    };
    let (quant, rest) = match rest.split_once(SECTION_SG) {
        Some(pair) => pair,
        None => (rest, ""),
    };
    let (sg, sgw) = match rest.split_once(SECTION_SGW) {
        Some(pair) => pair,
        None => (rest, ""),
    };
    (head, quant, sg, sgw)
}

pub fn decode_block() -> &'static str {
    let after = match WGSL.split_once(DECODE_BEGIN) {
        Some((_, rest)) => rest,
        None => return "",
    };
    match after.split_once(DECODE_END) {
        Some((body, _)) => body,
        None => after,
    }
}

pub fn with_decode(src: &str, body: &str) -> Option<String> {
    let (head, rest) = src.split_once(DECODE_BEGIN)?;
    let (_, tail) = rest.split_once(DECODE_END)?;
    Some(format!("{head}{DECODE_BEGIN}\n{body}\n{DECODE_END}{tail}"))
}

pub fn gemv_source() -> String {
    compose(split_sections().0)
}

pub fn quantize_source() -> String {
    compose(split_sections().1)
}

pub fn sg_gemv_source() -> String {
    let (head, _, sg, _) = split_sections();
    compose(&format!("{head}\n{sg}"))
}

pub fn subgroup_ok(ctx: &WgpuContext) -> bool {
    ctx.caps.subgroup_width_known() == Some(32)
}

pub fn sg32_ok(ctx: &WgpuContext) -> bool {
    subgroup_ok(ctx)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemvVariant {
    Tree,
    Sg,
    SgU,
}

impl GemvVariant {
    pub fn entry(self) -> &'static str {
        match self {
            Self::Tree => GEMV_ENTRY,
            Self::Sg => SG_GEMV_ENTRY,
            Self::SgU => SGU_GEMV_ENTRY,
        }
    }

    pub fn source(self) -> String {
        match self {
            Self::Tree => gemv_source(),
            Self::Sg | Self::SgU => sg_gemv_source(),
        }
    }

    pub fn rows_per_group(self) -> u32 {
        match self {
            Self::Tree => 1,
            Self::Sg | Self::SgU => SG_ROWS_PER_GROUP,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Tree => "nvfp4-gemv-bf16",
            Self::Sg => "nvfp4-gemv-bf16-sg",
            Self::SgU => SGU_LABEL,
        }
    }
}

pub const SGU_LABEL: &str = "nvfp4-gemv-bf16-sgu";

pub fn select_variant(ctx: &WgpuContext) -> GemvVariant {
    if sg32_ok(ctx) {
        GemvVariant::Sg
    } else {
        GemvVariant::Tree
    }
}

pub const SG_CROSSOVER_BLOCKS_PER_LANE: usize = 2;

pub fn blocks_per_lane(k: usize) -> usize {
    (k / NVFP4_BLOCK_SIZE).div_ceil(WORKGROUP_SIZE as usize)
}

pub fn select_variant_for_shape(ctx: &WgpuContext, _n: usize, k: usize) -> GemvVariant {
    if !sg32_ok(ctx) {
        return GemvVariant::Tree;
    }
    if blocks_per_lane(k) > SG_CROSSOVER_BLOCKS_PER_LANE {
        GemvVariant::Tree
    } else {
        GemvVariant::Sg
    }
}

pub fn sgq_shape_ok(k: usize) -> bool {
    k.is_multiple_of(NVFP4_BLOCK_SIZE) && (k / NVFP4_BLOCK_SIZE).is_multiple_of(4)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SgwConfig {
    pub wg: u32,
    pub rows: u32,
    pub tiled: bool,
    pub stage: bool,
}

impl SgwConfig {
    pub const fn new(wg: u32, rows: u32, tiled: bool, stage: bool) -> Self {
        Self {
            wg,
            rows,
            tiled,
            stage,
        }
    }

    pub fn subgroups_per_row(self) -> u32 {
        (self.wg / 32) / self.rows
    }

    pub fn vwarps(self) -> u32 {
        SGW_VWARPS / self.subgroups_per_row()
    }

    pub fn valid(self) -> bool {
        self.wg.is_multiple_of(32)
            && self.wg >= 32
            && self.rows >= 1
            && (self.wg / 32).is_multiple_of(self.rows)
            && matches!(self.subgroups_per_row(), 1 | 2 | 4 | 8)
            && (!self.stage || self.tiled)
    }

    pub fn workgroup_storage_bytes(self) -> u32 {
        let part = (self.wg / 32) * 4;
        let stage = if self.stage { SGW_TILE_BLOCKS * 12 } else { 0 };
        part + stage
    }

    pub fn supported(self, ctx: &WgpuContext) -> bool {
        self.valid()
            && sg32_ok(ctx)
            && self.wg <= ctx.caps.max_compute_invocations_per_workgroup
            && ctx
                .caps
                .workgroup_storage_fits(self.workgroup_storage_bytes())
    }

    pub fn label(self) -> String {
        format!(
            "nvfp4-gemv-sgw-w{}-r{}-{}{}",
            self.wg,
            self.rows,
            if self.tiled { "tiled" } else { "flat" },
            if self.stage { "-stage" } else { "" }
        )
    }
}

impl Default for SgwConfig {
    fn default() -> Self {
        Self::new(256, 1, false, false)
    }
}

pub fn sgw_source(cfg: SgwConfig) -> String {
    let (head, _, sg, sgw) = split_sections();
    let sgpr = cfg.subgroups_per_row();
    let body = sgw
        .replace(
            "const SGW_WG: u32 = 256u;",
            &format!("const SGW_WG: u32 = {}u;", cfg.wg),
        )
        .replace(
            "const SGW_ROWS: u32 = 1u;",
            &format!("const SGW_ROWS: u32 = {}u;", cfg.rows),
        )
        .replace(
            "const SGW_SGPR: u32 = 8u;",
            &format!("const SGW_SGPR: u32 = {sgpr}u;"),
        )
        .replace(
            "const SGW_VW: u32 = 1u;",
            &format!("const SGW_VW: u32 = {}u;", cfg.vwarps()),
        )
        .replace(
            "const SGW_TILED: u32 = 0u;",
            &format!("const SGW_TILED: u32 = {}u;", u32::from(cfg.tiled)),
        )
        .replace(
            "const SGW_STAGE: u32 = 0u;",
            &format!("const SGW_STAGE: u32 = {}u;", u32::from(cfg.stage)),
        )
        .replace(
            "const SGW_STAGE_LEN: u32 = 1u;",
            &format!(
                "const SGW_STAGE_LEN: u32 = {}u;",
                if cfg.stage { SGW_TILE_BLOCKS } else { 1 }
            ),
        )
        .replace(
            "const SGW_PART_LEN: u32 = 8u;",
            &format!("const SGW_PART_LEN: u32 = {}u;", cfg.wg / 32),
        );
    compose(&format!("{head}\n{sg}\n{body}"))
}

pub const SGW_SHALLOW: SgwConfig = SgwConfig::new(128, 4, false, false);
pub const SGW_DEEP: SgwConfig = SgwConfig::new(256, 2, false, false);

pub fn select_sgw(ctx: &WgpuContext, _n: usize, k: usize) -> Option<SgwConfig> {
    if !sg32_ok(ctx) {
        return None;
    }
    let cfg = if blocks_per_lane(k) > SG_CROSSOVER_BLOCKS_PER_LANE {
        SGW_DEEP
    } else {
        SGW_SHALLOW
    };
    cfg.supported(ctx).then_some(cfg)
}

pub fn gemv_params(alpha: f32, n: usize, k: usize, groups_x: u32) -> GemvParams {
    let k_blocks = k / NVFP4_BLOCK_SIZE;
    GemvParams {
        alpha,
        n_rows: n as u32,
        k_blocks: k_blocks as u32,
        k_tiles: k_tiles(k_blocks) as u32,
        w_row_words: (k / 8) as u32,
        groups_x,
        pad0: 0,
        pad1: 0,
    }
}

pub fn k_tiles(k_blocks: usize) -> usize {
    k_blocks.div_ceil(4)
}

pub fn swizzled_scale_len(n: usize, k_blocks: usize) -> usize {
    n.div_ceil(128) * 128 * k_tiles(k_blocks) * 4
}

pub fn coop_mat_capable(ctx: &WgpuContext) -> bool {
    ctx.caps.gemm_strategy() == GemmStrategy::CoopMat
}

pub fn nvfp4_gemv_bf16(
    ctx: &WgpuContext,
    w_packed: &[u32],
    w_scales: &[u8],
    x_packed: &[u32],
    x_scales: &[u8],
    alpha: f32,
    y: &mut [u16],
    n: usize,
    k: usize,
) -> Result<()> {
    if n == 0 || k == 0 {
        return Ok(());
    }
    if !k.is_multiple_of(NVFP4_BLOCK_SIZE) {
        return Err(WgpuError::Shape(format!(
            "K {k} is not a multiple of {NVFP4_BLOCK_SIZE}"
        )));
    }
    let k_blocks = k / NVFP4_BLOCK_SIZE;
    let row_words = k / 8;
    dispatch::check_len("nvfp4 gemv w_packed", w_packed.len(), n * row_words)?;
    dispatch::check_len("nvfp4 gemv x_packed", x_packed.len(), row_words)?;
    dispatch::check_len("nvfp4 gemv x_scales", x_scales.len(), k_blocks)?;
    dispatch::check_len("nvfp4 gemv y", y.len(), n)?;
    let want_scales = swizzled_scale_len(n, k_blocks);
    if w_scales.len() < want_scales {
        return Err(WgpuError::Shape(format!(
            "nvfp4 gemv w_scales: got {} want at least {want_scales}",
            w_scales.len()
        )));
    }

    let w_scale_words = bytes_to_words(w_scales);
    let x_scale_words = bytes_to_words(x_scales);
    let w_buf = dispatch::storage_from_slice(ctx, "nvfp4-gemv-w", w_packed);
    let ws_buf = dispatch::storage_from_slice(ctx, "nvfp4-gemv-ws", &w_scale_words);
    let x_buf = dispatch::storage_from_slice(ctx, "nvfp4-gemv-x", x_packed);
    let xs_buf = dispatch::storage_from_slice(ctx, "nvfp4-gemv-xs", &x_scale_words);
    let y_buf = dispatch::storage_zeroed(ctx, "nvfp4-gemv-y", (n * 4) as u64);

    let variant = select_variant_for_shape(ctx, n, k);
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, variant.rows_per_group());
    let params = gemv_params(alpha, n, k, groups.0);
    let params_buf = dispatch::uniform_from(ctx, "nvfp4-gemv-params", &params);

    dispatch::run(
        ctx,
        variant.label(),
        &variant.source(),
        variant.entry(),
        &[
            (0, &w_buf),
            (1, &ws_buf),
            (2, &x_buf),
            (3, &xs_buf),
            (4, &params_buf),
            (5, &y_buf),
        ],
        groups,
    )?;

    let words: Vec<u32> = dispatch::read_back(ctx, &y_buf, n)?;
    for (dst, w) in y.iter_mut().zip(words.iter()) {
        *dst = (*w & 0xffff) as u16;
    }
    Ok(())
}

pub fn nvfp4_quantize_row_bf16(
    ctx: &WgpuContext,
    x: &[u16],
    global_scale: f32,
    packed_out: &mut [u32],
    scales_out: &mut [u8],
    k: usize,
) -> Result<()> {
    if k == 0 {
        return Ok(());
    }
    if !k.is_multiple_of(NVFP4_BLOCK_SIZE) {
        return Err(WgpuError::Shape(format!(
            "K {k} is not a multiple of {NVFP4_BLOCK_SIZE}"
        )));
    }
    let k_blocks = k / NVFP4_BLOCK_SIZE;
    dispatch::check_len("nvfp4 quantize x", x.len(), k)?;
    dispatch::check_len("nvfp4 quantize packed_out", packed_out.len(), k / 8)?;
    dispatch::check_len("nvfp4 quantize scales_out", scales_out.len(), k_blocks)?;

    let mut x_words = vec![0u32; k / 2];
    for (j, v) in x.iter().enumerate() {
        x_words[j / 2] |= (*v as u32) << (16 * (j % 2));
    }

    let x_buf = dispatch::storage_from_slice(ctx, "nvfp4-quant-x", &x_words);
    let packed_buf = dispatch::storage_zeroed(ctx, "nvfp4-quant-packed", (k / 8 * 4) as u64);
    let scales_buf = dispatch::storage_zeroed(ctx, "nvfp4-quant-scales", (k_blocks * 4) as u64);
    let params = QuantParams {
        global_scale,
        k_blocks: k_blocks as u32,
        pad0: 0,
        pad1: 0,
    };
    let params_buf = dispatch::uniform_from(ctx, "nvfp4-quant-params", &params);

    dispatch::run(
        ctx,
        "nvfp4-quantize-row-bf16",
        &quantize_source(),
        QUANTIZE_ENTRY,
        &[
            (0, &x_buf),
            (1, &params_buf),
            (2, &packed_buf),
            (3, &scales_buf),
        ],
        (1, 1, 1),
    )?;

    let packed: Vec<u32> = dispatch::read_back(ctx, &packed_buf, k / 8)?;
    packed_out.copy_from_slice(&packed);
    let scales: Vec<u32> = dispatch::read_back(ctx, &scales_buf, k_blocks)?;
    for (dst, w) in scales_out.iter_mut().zip(scales.iter()) {
        *dst = (*w & 0xff) as u8;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_split_and_carry_their_entry_points() {
        let (head, quant, sg, sgw) = split_sections();
        assert!(head.contains(GEMV_ENTRY));
        assert!(!head.contains(QUANTIZE_ENTRY));
        assert!(quant.contains(QUANTIZE_ENTRY));
        assert!(!quant.contains(GEMV_ENTRY));
        assert!(!quant.contains(SG_GEMV_ENTRY));
        assert!(sg.contains(SG_GEMV_ENTRY));
        assert!(!sg.contains(SGW_GEMV_ENTRY));
        assert!(sgw.contains(SGW_GEMV_ENTRY));
    }

    #[test]
    fn legacy_sources_are_untouched_by_the_sgw_section() {
        for src in [gemv_source(), quantize_source(), sg_gemv_source()] {
            assert!(!src.contains(SGW_GEMV_ENTRY));
            assert!(!src.contains("sgw_part"));
            assert!(!src.contains("SGW_WG"));
        }
    }

    #[test]
    fn sgw_ladder_configs_are_all_well_formed() {
        for cfg in [SGW_SHALLOW, SGW_DEEP] {
            assert!(cfg.valid(), "{cfg:?}");
            assert_eq!(cfg.subgroups_per_row() * cfg.vwarps(), SGW_VWARPS);
        }
        assert_eq!(SgwConfig::new(256, 1, false, false).subgroups_per_row(), 8);
        assert_eq!(SgwConfig::new(256, 1, false, false).vwarps(), 1);
        assert_eq!(SgwConfig::new(128, 4, false, false).subgroups_per_row(), 1);
        assert_eq!(SgwConfig::new(128, 4, false, false).vwarps(), 8);
        assert!(!SgwConfig::new(256, 16, false, false).valid());
        assert!(!SgwConfig::new(256, 1, false, true).valid());
    }

    #[test]
    fn sgw_source_substitutes_every_knob() {
        let cfg = SgwConfig::new(128, 2, true, true);
        let src = sgw_source(cfg);
        assert!(src.contains("const SGW_WG: u32 = 128u;"));
        assert!(src.contains("const SGW_ROWS: u32 = 2u;"));
        assert!(src.contains("const SGW_SGPR: u32 = 2u;"));
        assert!(src.contains("const SGW_VW: u32 = 4u;"));
        assert!(src.contains("const SGW_TILED: u32 = 1u;"));
        assert!(src.contains("const SGW_STAGE: u32 = 1u;"));
        assert!(src.contains("const SGW_STAGE_LEN: u32 = 256u;"));
        assert!(src.contains("const SGW_PART_LEN: u32 = 4u;"));
        assert!(src.contains("@workgroup_size(SGW_WG)"));
        assert!(src.contains("fn nvfp4_scale_byte_index("));
        assert!(src.contains("subgroupShuffleXor"));
        assert_eq!(cfg.workgroup_storage_bytes(), 16 + 256 * 12);
    }

    #[test]
    fn blocks_per_lane_matches_the_dispatch_stride() {
        assert_eq!(blocks_per_lane(5376), 2);
        assert_eq!(blocks_per_lane(8192), 2);
        assert_eq!(blocks_per_lane(12288), 3);
        assert_eq!(blocks_per_lane(21504), 6);
        assert_eq!(blocks_per_lane(4096), 1);
    }

    #[test]
    fn the_shape_gate_splits_the_gemma4_projections_where_it_was_measured() {
        let sg_side = [5376usize, 8192];
        let tree_side = [12288usize, 16384, 21504, 32768];
        for k in sg_side {
            assert!(
                blocks_per_lane(k) <= SG_CROSSOVER_BLOCKS_PER_LANE,
                "k={k} should land on the subgroup side of the gate"
            );
        }
        for k in tree_side {
            assert!(
                blocks_per_lane(k) > SG_CROSSOVER_BLOCKS_PER_LANE,
                "k={k} should land on the tree side of the gate"
            );
        }
        assert_eq!(SGW_SHALLOW.rows, SG_ROWS_PER_GROUP);
        assert_eq!(SGW_SHALLOW.subgroups_per_row(), 1);
        assert_eq!(SGW_SHALLOW.vwarps(), SGW_VWARPS);
        assert_eq!(SGW_DEEP.subgroups_per_row(), 4);
        assert_eq!(SGW_DEEP.vwarps(), 2);
    }

    #[test]
    fn the_hot_path_decodes_without_a_lookup_table() {
        for src in [
            gemv_source(),
            sg_gemv_source(),
            sgw_source(SGW_DEEP),
            sgw_source(SGW_SHALLOW),
        ] {
            assert!(src.contains("fn gemv_e2m1_decode("));
            assert!(src.contains("fn gemv_ue4m3_decode("));
            assert!(src.contains("fn gemv_i8map("));
            let body = src.split_once("fn gemv_dot8(").expect("dot8").1;
            let dot8 = body.split_once("\n}").expect("dot8 end").0;
            assert!(
                dot8.contains("dot4I8Packed"),
                "dot8 lost the packed integer dot"
            );
            assert!(dot8.contains("gemv_i8map"), "dot8 lost the swar nibble map");
            assert!(
                !dot8.contains("E2M1_TABLE"),
                "dot8 regressed to a table lookup"
            );
            assert!(
                !dot8.contains("gemv_e2m1_decode"),
                "dot8 regressed to per-element float decode"
            );
        }
    }

    #[test]
    fn the_decode_block_is_marker_wrapped_and_swappable() {
        let block = decode_block();
        for f in [
            "fn gemv_ue4m3_decode(",
            "fn gemv_e2m1_decode(",
            "fn gemv_i8map(",
            "fn gemv_dot8(",
        ] {
            assert!(block.contains(f), "decode block lost {f}");
        }
        for src in [gemv_source(), sg_gemv_source(), sgw_source(SGW_DEEP)] {
            let swapped = with_decode(&src, "// swapped").expect("markers present");
            assert!(!swapped.contains("fn gemv_i8map("));
            assert!(swapped.contains(SG_GEMV_ENTRY) || swapped.contains(GEMV_ENTRY));
        }
        assert!(with_decode("no markers here", "x").is_none());
    }

    #[test]
    fn dead_rows_do_no_block_work_in_any_gemv_entry() {
        let (head, _, sg, sgw) = split_sections();
        assert!(head.contains("let blocks = select(0u, gemv_params.k_blocks, row_live);"));
        assert_eq!(
            sg.matches("let blocks = select(0u, gemv_params.k_blocks, row_live);")
                .count(),
            2
        );
        assert!(sgw.contains("let live_blocks = select(0u, blocks, row_live);"));
        assert!(sgw.contains("kb < live_blocks"));
        assert!(sgw.contains("let quads = select(0u, gemv_params.k_blocks >> 2u, row_live);"));
    }

    #[test]
    fn sgq_is_only_offered_on_quad_aligned_k() {
        assert!(sgq_shape_ok(5376));
        assert!(sgq_shape_ok(8192));
        assert!(sgq_shape_ok(21504));
        assert!(!sgq_shape_ok(16 * 6));
        assert!(!sgq_shape_ok(20));
    }

    #[test]
    fn composed_sources_include_the_dequant_prelude() {
        assert!(gemv_source().contains("fn nvfp4_scale_byte_index("));
        assert!(quantize_source().contains("fn ue4m3_decode("));
    }

    #[test]
    fn non_sg_sources_stay_free_of_subgroup_builtins() {
        assert!(!gemv_source().contains("subgroup"));
        assert!(!quantize_source().contains("subgroup"));
    }

    #[test]
    fn sg_source_carries_both_entries_and_the_butterfly() {
        let src = sg_gemv_source();
        assert!(src.contains(SG_GEMV_ENTRY));
        assert!(src.contains(SGU_GEMV_ENTRY));
        assert!(src.contains(GEMV_ENTRY));
        assert!(src.contains("subgroupShuffleXor"));
        assert!(src.contains("fn nvfp4_scale_byte_index("));
        assert!(src.contains("@workgroup_size(128)"));
    }

    #[test]
    fn variant_dispatch_geometry_matches_the_shader_layout() {
        assert_eq!(GemvVariant::Tree.rows_per_group(), 1);
        assert_eq!(GemvVariant::Sg.rows_per_group(), SG_ROWS_PER_GROUP);
        assert_eq!(SG_WORKGROUP_SIZE / 32, SG_ROWS_PER_GROUP);
        let p = gemv_params(1.5, 40, 5376, 7);
        assert_eq!(p.k_blocks, 336);
        assert_eq!(p.k_tiles, 84);
        assert_eq!(p.w_row_words, 672);
        assert_eq!(p.groups_x, 7);
    }

    #[test]
    fn swizzled_scale_len_matches_the_quant_layout() {
        assert_eq!(swizzled_scale_len(1, 8), 128 * 2 * 4);
        assert_eq!(swizzled_scale_len(129, 4), 256 * 4);
    }
}
