use crate::wgpu_backend::compose;
use crate::wgpu_backend::dequant::NVFP4_BLOCK_SIZE;
use crate::wgpu_backend::device::WgpuContext;

pub const WGSL: &str = include_str!("../../../wgsl/gemv_nvfp4_v2.wgsl");

pub const WARP_ENTRY: &str = "gemv_nvfp4_warp";
pub const WARPQ_ENTRY: &str = "gemv_nvfp4_warpq";
pub const FDEC_ENTRY: &str = "gemv_nvfp4_fdec";
pub const MROW_ENTRY: &str = "gemv_nvfp4_mrow";
pub const MROW2_ENTRY: &str = "gemv_nvfp4_mrow2";
pub const MROWQ_ENTRY: &str = "gemv_nvfp4_mrowq";
pub const FMROW_ENTRY: &str = "gemv_nvfp4_fmrow";
pub const FMLUT_ENTRY: &str = "gemv_nvfp4_fmlut";

pub const WARP_PK_ENTRY: &str = "gemv_nvfp4_warp_pk";
pub const FDEC_PK_ENTRY: &str = "gemv_nvfp4_fdec_pk";
pub const FMLUT_PK_ENTRY: &str = "gemv_nvfp4_fmlut_pk";
pub const MROW_PK_ENTRY: &str = "gemv_nvfp4_mrow_pk";
pub const MROW2_PK_ENTRY: &str = "gemv_nvfp4_mrow2_pk";

pub const DECODE_BEGIN: &str = "const NV2_DECODE_BEGIN: u32 = 0u;";
pub const DECODE_END: &str = "const NV2_DECODE_END: u32 = 0u;";

pub const W2_SLOT: u32 = 0;
pub const WS_SLOT: u32 = 1;
pub const X2_SLOT: u32 = 2;
pub const XS_SLOT: u32 = 3;
pub const PARAMS_SLOT: u32 = 4;
pub const Y_SLOT: u32 = 5;
pub const W4_SLOT: u32 = 6;
pub const X4_SLOT: u32 = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum V2Kernel {
    Warp,
    WarpQ,
    FDec,
    MRow,
    MRow2,
    MRowQ,
    FMRow,
    FMLut,
}

impl V2Kernel {
    pub fn entry(self) -> &'static str {
        match self {
            Self::Warp => WARP_ENTRY,
            Self::WarpQ => WARPQ_ENTRY,
            Self::FDec => FDEC_ENTRY,
            Self::MRow => MROW_ENTRY,
            Self::MRow2 => MROW2_ENTRY,
            Self::MRowQ => MROWQ_ENTRY,
            Self::FMRow => FMROW_ENTRY,
            Self::FMLut => FMLUT_ENTRY,
        }
    }

    pub fn pk_entry(self) -> Option<&'static str> {
        match self {
            Self::Warp => Some(WARP_PK_ENTRY),
            Self::FDec => Some(FDEC_PK_ENTRY),
            Self::FMLut => Some(FMLUT_PK_ENTRY),
            Self::MRow => Some(MROW_PK_ENTRY),
            Self::MRow2 => Some(MROW2_PK_ENTRY),
            _ => None,
        }
    }

    pub fn multi_row(self) -> bool {
        matches!(self, Self::MRow | Self::FMRow | Self::FMLut)
    }

    pub fn rows_per_subgroup(self, mr: u32) -> u32 {
        match self {
            Self::Warp | Self::WarpQ | Self::FDec => 1,
            Self::MRow | Self::FMRow | Self::FMLut => mr,
            Self::MRow2 | Self::MRowQ => 2,
        }
    }

    pub fn vec4_slots(self) -> bool {
        !matches!(self, Self::Warp)
    }

    pub fn blocks_per_iter(self) -> usize {
        match self {
            Self::Warp => 1,
            Self::WarpQ | Self::FDec | Self::MRowQ => 4,
            Self::MRow | Self::MRow2 | Self::FMRow | Self::FMLut => 2,
        }
    }

    pub fn shape_ok(self, k: usize) -> bool {
        k.is_multiple_of(NVFP4_BLOCK_SIZE)
            && (k / NVFP4_BLOCK_SIZE).is_multiple_of(self.blocks_per_iter())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V2Config {
    pub wg: u32,
    pub mr: u32,
}

impl V2Config {
    pub const fn new(wg: u32, mr: u32) -> Self {
        Self { wg, mr }
    }

    pub fn subgroups(self) -> u32 {
        self.wg / 32
    }

    pub fn rows_per_group(self, kernel: V2Kernel) -> u32 {
        self.subgroups() * kernel.rows_per_subgroup(self.mr)
    }

    pub fn valid(self) -> bool {
        self.wg.is_multiple_of(32)
            && self.wg >= 32
            && self.wg <= 1024
            && self.mr >= 1
            && self.mr <= 8
    }
}

impl Default for V2Config {
    fn default() -> Self {
        Self::new(256, 4)
    }
}

pub fn source(cfg: V2Config) -> String {
    let body = WGSL
        .replace(
            "const NV2_WG: u32 = 256u;",
            &format!("const NV2_WG: u32 = {}u;", cfg.wg),
        )
        .replace(
            "const NV2_MR: u32 = 4u;",
            &format!("const NV2_MR: u32 = {}u;", cfg.mr),
        );
    compose(&body)
}

pub const HELPERS_BEGIN: &str = "const NV2_WG: u32 = 256u;";
pub const HELPERS_END: &str = "const NV2_SECTION_WARP: u32 = 1u;";

pub fn helpers(cfg: V2Config) -> String {
    let body = WGSL
        .split_once(HELPERS_BEGIN)
        .map(|(_, rest)| format!("{HELPERS_BEGIN}{rest}"))
        .unwrap_or_default();
    let body = body.split_once(HELPERS_END).map(|(h, _)| h).unwrap_or("");
    body.replace(
        "const NV2_WG: u32 = 256u;",
        &format!("const NV2_WG: u32 = {}u;", cfg.wg),
    )
    .replace(
        "const NV2_MR: u32 = 4u;",
        &format!("const NV2_MR: u32 = {}u;", cfg.mr),
    )
}

pub fn select(n: usize, k: usize) -> (V2Kernel, V2Config) {
    select_slots(n, k, 1)
}

pub const MROW_BEATS_FMLUT_ON_SINGLE_SLOT_DENSE_SHAPES_ON_THIS_LADDER: &str =
    "decode-ALU ablation: mrow(128,2) out-streams fmlut(128,4) at every dense single-slot \
     shape measured -- fmlut's decode is ALU-bound while mrow sits near the stream ceiling \
     (current numbers: perf/runs.jsonl). The routed MoE dispatch keeps fmlut: at 9-slot \
     expert shapes fmlut won the earlier round and mrow was not re-measured there. \
     NV_NVFP4_V2_SELECT forces either by name.";

pub const MROW2_WINS_EVERY_DENSE_SINGLE_SLOT_SHAPE_BUT_SHIPS_ONLY_WITH_THE_MODEL_PK_ARMS: &str =
    "interleaved route A/B (dense_single_slot_route_ab): scalar two-row mrow2 out-streams \
     mrow(128,2) on every dense single-slot shape, both orientations, while wide 32-byte \
     mrowq loses (current numbers: perf/runs.jsonl). The default stays mrow because \
     qwen3_5_moe_wgpu and gemma4_wgpu map pk entries by name and fall back to the tree \
     kernel on an unknown entry -- flipping select_slots before those two match arms know \
     q3w_gemv_nvfp4_mrow2 would silently reroute production to the slowest kernel. \
     NV_NVFP4_V2_SELECT=mrow2 forces it where the entry is dispatched directly.";

pub fn select_slots(n: usize, k: usize, slots: usize) -> (V2Kernel, V2Config) {
    if let Some(forced) = select_override(n, k) {
        return forced;
    }
    let rows = n.saturating_mul(slots.max(1));
    if slots <= 1 {
        let deep = k / NVFP4_BLOCK_SIZE >= 64;
        if deep && rows >= FMLUT_MIN_ROWS && V2Kernel::MRow.shape_ok(k) {
            return (V2Kernel::MRow, V2Config::new(128, 2));
        }
    }
    select_default(rows, k)
}

fn select_override(n: usize, k: usize) -> Option<(V2Kernel, V2Config)> {
    let v = std::env::var("NV_NVFP4_V2_SELECT").ok()?;
    let picked = match v.trim().to_ascii_lowercase().as_str() {
        "fmlut" => (V2Kernel::FMLut, V2Config::new(128, 4)),
        "fdec" => (V2Kernel::FDec, V2Config::new(128, 1)),
        "warp" => (V2Kernel::Warp, V2Config::new(64, 1)),
        "mrow" => (V2Kernel::MRow, V2Config::new(128, 2)),
        "mrow2" => (V2Kernel::MRow2, V2Config::new(128, 2)),
        "mrowq" => (V2Kernel::MRowQ, V2Config::new(128, 2)),
        _ => return None,
    };
    if picked.0.shape_ok(k) && n >= picked.1.mr as usize {
        Some(picked)
    } else {
        None
    }
}

fn select_default(rows: usize, k: usize) -> (V2Kernel, V2Config) {
    let k_blocks = k / NVFP4_BLOCK_SIZE;
    let deep = k_blocks >= 64;
    if deep && rows >= FMLUT_MIN_ROWS && V2Kernel::FMLut.shape_ok(k) {
        (V2Kernel::FMLut, V2Config::new(128, 4))
    } else if deep && V2Kernel::FDec.shape_ok(k) {
        (V2Kernel::FDec, V2Config::new(128, 1))
    } else {
        (V2Kernel::Warp, V2Config::new(64, 1))
    }
}

const FMLUT_MIN_ROWS: usize = 2048;

pub fn subgroup32_ok(ctx: &WgpuContext) -> bool {
    ctx.caps.subgroup && ctx.subgroup_width() == Some(NV2_LANES)
}

pub const NV2_LANES: u32 = 32;

pub fn pk_capable(kernel: V2Kernel, cfg: V2Config) -> bool {
    if !cfg.valid() || kernel.pk_entry().is_none() {
        return false;
    }
    let rps = kernel.rows_per_subgroup(cfg.mr);
    if rps > 1 {
        rps.is_multiple_of(2)
    } else {
        cfg.subgroups().is_multiple_of(2)
    }
}

pub fn select_pk(n: usize, k: usize) -> Option<(V2Kernel, V2Config, &'static str)> {
    select_pk_slots(n, k, 1)
}

pub fn select_pk_slots(
    n: usize,
    k: usize,
    slots: usize,
) -> Option<(V2Kernel, V2Config, &'static str)> {
    let (kernel, cfg) = select_slots(n, k, slots);
    if !kernel.shape_ok(k) || !pk_capable(kernel, cfg) {
        return None;
    }
    if !cfg.rows_per_group(kernel).is_multiple_of(2) {
        return None;
    }
    Some((kernel, cfg, kernel.pk_entry()?))
}

pub fn with_decode(src: &str, body: &str) -> Option<String> {
    let (head, rest) = src.split_once(DECODE_BEGIN)?;
    let (_, tail) = rest.split_once(DECODE_END)?;
    Some(format!("{head}{DECODE_BEGIN}\n{body}\n{DECODE_END}{tail}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_entry_point_is_present_in_the_shader() {
        let src = source(V2Config::default());
        for e in [
            WARP_ENTRY,
            WARPQ_ENTRY,
            FDEC_ENTRY,
            MROW_ENTRY,
            MROW2_ENTRY,
            MROWQ_ENTRY,
            FMROW_ENTRY,
            FMLUT_ENTRY,
        ] {
            assert!(src.contains(&format!("fn {e}(")), "missing {e}");
        }
        assert!(src.contains("fn nvfp4_scale_byte_index("));
        assert!(src.contains("subgroupShuffleXor"));
    }

    #[test]
    fn config_substitution_reaches_both_knobs() {
        let src = source(V2Config::new(128, 2));
        assert!(src.contains("const NV2_WG: u32 = 128u;"));
        assert!(src.contains("const NV2_MR: u32 = 2u;"));
        assert!(!src.contains("const NV2_WG: u32 = 256u;"));
    }

    #[test]
    fn row_geometry_matches_the_shader_layout() {
        let c = V2Config::new(256, 4);
        assert_eq!(c.subgroups(), 8);
        assert_eq!(c.rows_per_group(V2Kernel::Warp), 8);
        assert_eq!(c.rows_per_group(V2Kernel::WarpQ), 8);
        assert_eq!(c.rows_per_group(V2Kernel::MRow), 32);
        assert_eq!(c.rows_per_group(V2Kernel::MRow2), 16);
        assert_eq!(c.rows_per_group(V2Kernel::MRowQ), 16);
        assert_eq!(V2Config::new(128, 2).rows_per_group(V2Kernel::FMRow), 8);
        assert!(c.valid());
        assert!(!V2Config::new(48, 1).valid());
    }

    #[test]
    fn quad_kernels_reject_k_that_is_not_quad_aligned() {
        assert!(V2Kernel::WarpQ.shape_ok(5376));
        assert!(V2Kernel::WarpQ.shape_ok(21504));
        assert!(V2Kernel::WarpQ.shape_ok(512));
        assert!(!V2Kernel::WarpQ.shape_ok(16 * 6));
        assert!(!V2Kernel::MRowQ.shape_ok(16 * 6));
        assert!(V2Kernel::MRowQ.shape_ok(16 * 8));
        assert!(V2Kernel::MRow.shape_ok(16 * 6));
        assert!(V2Kernel::MRow2.shape_ok(16 * 6));
        assert!(!V2Kernel::MRow.shape_ok(16 * 3));
        assert!(V2Kernel::Warp.shape_ok(16 * 3));
    }

    #[test]
    fn select_picks_the_measured_winner_at_every_production_shape() {
        let mrow = (V2Kernel::MRow, V2Config::new(128, 2));
        let fdec = (V2Kernel::FDec, V2Config::new(128, 1));
        let warp = (V2Kernel::Warp, V2Config::new(64, 1));
        assert_eq!(select(43008, 5376), mrow);
        assert_eq!(select(5376, 21504), mrow);
        assert_eq!(select(8192, 2048), mrow);
        assert_eq!(select(2048, 4096), mrow);
        assert_eq!(select(512, 2048), fdec);
        assert_eq!(select(2048, 512), warp);
        assert_eq!(select(4096, 16 * 33), warp);
        for (n, k) in [(43008usize, 5376usize), (512, 2048), (2048, 512)] {
            let (kern, cfg) = select(n, k);
            assert!(kern.shape_ok(k) && cfg.valid());
        }
    }

    #[test]
    fn slots_move_the_moe_stacks_and_nothing_else() {
        let fmlut = (V2Kernel::FMLut, V2Config::new(128, 4));
        let fdec = (V2Kernel::FDec, V2Config::new(128, 1));
        let warp = (V2Kernel::Warp, V2Config::new(64, 1));

        assert_eq!(select_slots(1024, 2048, 9), fmlut);

        assert_eq!(select_slots(512, 2048, 9), fmlut);

        assert_eq!(select_slots(2048, 512, 9), warp);

        for (n, k) in [
            (43008usize, 5376usize),
            (5376, 21504),
            (8192, 2048),
            (2048, 4096),
            (512, 2048),
            (1024, 2048),
            (2048, 512),
            (4096, 16 * 33),
            (300, 1024),
            (37, 2048),
        ] {
            assert_eq!(
                select_slots(n, k, 1),
                select(n, k),
                "n={n} k={k}: the slots-aware path must reproduce the old rule at one slot"
            );
            assert_eq!(select_pk_slots(n, k, 1), select_pk(n, k), "n={n} k={k}");
        }
        assert_eq!(select_slots(1024, 2048, 1), fdec);

        assert_eq!(
            select_pk_slots(1024, 2048, 9).map(|r| r.2),
            Some(FMLUT_PK_ENTRY)
        );
        assert_eq!(
            select_pk_slots(1024, 2048, 1).map(|r| r.2),
            Some(FDEC_PK_ENTRY)
        );
        assert_eq!(
            select_pk_slots(2048, 512, 9).map(|r| r.2),
            Some(WARP_PK_ENTRY)
        );

        assert_eq!(select_slots(1024, 2048, 0), select_slots(1024, 2048, 1));
    }

    #[test]
    fn every_pk_entry_point_is_present_and_pair_packs() {
        let src = source(V2Config::new(128, 4));
        for e in [WARP_PK_ENTRY, FDEC_PK_ENTRY, FMLUT_PK_ENTRY, MROW_PK_ENTRY, MROW2_PK_ENTRY] {
            assert!(src.contains(&format!("fn {e}(")), "missing {e}");
        }
        let pk = src
            .split_once("const NV2_SECTION_PK")
            .expect("pk section")
            .1;
        assert!(pk.contains("nv2_y[(row0 + m) >> 1u]"));
        assert!(pk.contains("nv2_y[row >> 1u]"));
        assert!(pk.contains("(hi << 16u)"));
    }

    #[test]
    fn select_pk_routes_every_production_shape_to_a_pair_packed_entry() {
        for (n, k, want) in [
            (43008usize, 5376usize, MROW_PK_ENTRY),
            (5376, 21504, MROW_PK_ENTRY),
            (8192, 2048, MROW_PK_ENTRY),
            (2048, 4096, MROW_PK_ENTRY),
            (512, 2048, FDEC_PK_ENTRY),
            (2048, 512, WARP_PK_ENTRY),
        ] {
            let (kern, cfg, entry) = select_pk(n, k).expect("pk route");
            assert_eq!(entry, want, "n={n} k={k}");
            assert!(kern.shape_ok(k));
            assert!(cfg.rows_per_group(kern).is_multiple_of(2));
        }
        assert!(select_pk(4096, 16 * 33).is_some());
        assert!(select_pk(4096, 24).is_none());
    }

    #[test]
    fn pk_capability_rejects_odd_row_blocking() {
        assert!(!pk_capable(V2Kernel::FMLut, V2Config::new(128, 3)));
        assert!(pk_capable(V2Kernel::FMLut, V2Config::new(128, 4)));
        assert!(!pk_capable(V2Kernel::Warp, V2Config::new(32, 1)));
        assert!(pk_capable(V2Kernel::Warp, V2Config::new(64, 1)));
        assert!(!pk_capable(V2Kernel::MRow, V2Config::new(128, 3)));
        assert!(pk_capable(V2Kernel::MRow, V2Config::new(128, 2)));
        assert!(pk_capable(V2Kernel::MRow2, V2Config::new(128, 3)));
        assert!(!pk_capable(V2Kernel::MRowQ, V2Config::new(128, 2)));
    }

    #[test]
    fn the_decode_block_is_marker_wrapped_and_swappable() {
        let src = source(V2Config::default());
        let swapped = with_decode(&src, "// gone").expect("markers present");
        assert!(!swapped.contains("fn nv2_i8map("));
        assert!(!swapped.contains("fn nv2_dec4("));
        assert!(!swapped.contains("fn nv2_mdec4("));
        assert!(swapped.contains("fn nv2_ue4m3("));
        assert!(swapped.contains(MROW_ENTRY));
        assert!(with_decode("no markers", "x").is_none());
    }

    #[test]
    fn the_int_path_keeps_the_swar_map_and_the_float_path_does_not() {
        let src = source(V2Config::default());
        let warp = src.split_once("fn gemv_nvfp4_warp(").expect("warp").1;
        let warp = warp.split_once("\nconst NV2_SECTION").expect("warp end").0;
        assert!(warp.contains("nv2_iblock"));
        let fdec = src.split_once("fn gemv_nvfp4_fdec(").expect("fdec").1;
        let fdec = fdec.split_once("\nconst NV2_SECTION").expect("fdec end").0;
        assert!(fdec.contains("nv2_fblock"));
        assert!(!fdec.contains("nv2_iblock"));
    }
}
