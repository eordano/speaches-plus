#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::dequant::{bytes_to_words, NVFP4_BLOCK_SIZE};
use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dispatch;
use crate::wgpu_backend::{compose, compose_enabled, Result, WgpuError};

use super::gemm_nvfp4::{k_tiles, resolve_path, swizzled_scale_len, GemmPath, COOP_ENABLES};

pub const WGSL: &str = include_str!("../../../wgsl/moe_grouped_gemm.wgsl");

pub const SECTION_SPLIT: &str = "const MOE_GEMM_SECTION_SPLIT: u32 = 1u;";
pub const SCALAR_ENTRY: &str = "moe_grouped_gemm_scalar";
pub const HOIST_ENTRY: &str = "moe_grouped_gemm_scalar_hoist";
pub const HOIST_V4_ENTRY: &str = "moe_grouped_gemm_scalar_hoist_v4";
pub const QUAD_ENTRY: &str = "moe_grouped_gemm_scalar_quad";
pub const QUAD_BITS_ENTRY: &str = "moe_grouped_gemm_scalar_quad_bits";
pub const SHARED_A_ENTRY: &str = "moe_grouped_gemm_scalar_shared_a";
pub const COOP_ENTRY: &str = "moe_grouped_gemm_coop";
pub const SCALAR_WORKGROUP_SIZE: u32 = 64;
pub const COOP_WORKGROUP_SIZE: u32 = 128;
pub const COOP_TILE_M: usize = 32;
pub const COOP_TILE_N: usize = 32;
pub const SHARED_A_K_MAX: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarVariant {
    Base,
    Hoist,
    HoistV4,
    Quad,
    QuadBits,
    SharedA,
}

impl ScalarVariant {
    pub fn entry(self) -> &'static str {
        match self {
            Self::Base => SCALAR_ENTRY,
            Self::Hoist => HOIST_ENTRY,
            Self::HoistV4 => HOIST_V4_ENTRY,
            Self::Quad => QUAD_ENTRY,
            Self::QuadBits => QUAD_BITS_ENTRY,
            Self::SharedA => SHARED_A_ENTRY,
        }
    }

    pub fn supports(self, n: usize, k: usize) -> bool {
        match self {
            Self::Base | Self::Hoist => true,
            Self::HoistV4 => k.is_multiple_of(32),
            Self::Quad | Self::QuadBits => k.is_multiple_of(64),
            Self::SharedA => {
                k.is_multiple_of(64)
                    && k <= SHARED_A_K_MAX
                    && n.is_multiple_of(SCALAR_WORKGROUP_SIZE as usize)
            }
        }
    }
}

pub fn select_scalar_variant(n: usize, k: usize) -> ScalarVariant {
    if ScalarVariant::SharedA.supports(n, k) {
        ScalarVariant::SharedA
    } else if ScalarVariant::QuadBits.supports(n, k) {
        ScalarVariant::QuadBits
    } else if ScalarVariant::HoistV4.supports(n, k) {
        ScalarVariant::HoistV4
    } else {
        ScalarVariant::Hoist
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct MoeGemmParams {
    n: u32,
    k: u32,
    row_words: u32,
    k_tiles: u32,
    b_sf_stride_bytes: u32,
    b_words_per_expert: u32,
    total_m: u32,
    groups_x: u32,
    total_tiles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
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

fn row_group_map(expert_offsets: &[i32], total_m: usize) -> Vec<u32> {
    let mut map = vec![0u32; total_m];
    for g in 0..expert_offsets.len() - 1 {
        let lo = expert_offsets[g] as usize;
        let hi = expert_offsets[g + 1] as usize;
        for slot in map.iter_mut().take(hi).skip(lo) {
            *slot = g as u32;
        }
    }
    map
}

fn tile_table(expert_offsets: &[i32], n: usize) -> Vec<u32> {
    let tiles_n = n.div_ceil(COOP_TILE_N);
    let mut table = Vec::new();
    for g in 0..expert_offsets.len() - 1 {
        let lo = expert_offsets[g] as usize;
        let hi = expert_offsets[g + 1] as usize;
        let tiles_m = (hi - lo).div_ceil(COOP_TILE_M);
        for tm in 0..tiles_m {
            for tn in 0..tiles_n {
                table.extend_from_slice(&[
                    g as u32,
                    (lo + tm * COOP_TILE_M) as u32,
                    hi as u32,
                    (tn * COOP_TILE_N) as u32,
                ]);
            }
        }
    }
    table
}

fn validate_groups(
    expert_offsets: &[i32],
    expert_ids: &[i32],
    alphas: &[f32],
    e_total: usize,
) -> Result<usize> {
    let num_groups = expert_ids.len();
    if expert_offsets.len() != num_groups + 1 {
        return Err(WgpuError::Shape(format!(
            "moe grouped gemm expert_offsets: got {} want {}",
            expert_offsets.len(),
            num_groups + 1
        )));
    }
    dispatch::check_len("moe grouped gemm alphas", alphas.len(), num_groups)?;
    if expert_offsets.first() != Some(&0) {
        return Err(WgpuError::Shape(format!(
            "moe grouped gemm expert_offsets must start at 0, got {:?}",
            expert_offsets.first()
        )));
    }
    for w in expert_offsets.windows(2) {
        if w[1] < w[0] {
            return Err(WgpuError::Shape(format!(
                "moe grouped gemm expert_offsets not monotone: {} then {}",
                w[0], w[1]
            )));
        }
    }
    for &id in expert_ids {
        if id < 0 || id as usize >= e_total {
            return Err(WgpuError::Shape(format!(
                "moe grouped gemm expert id {id} out of range 0..{e_total}"
            )));
        }
    }
    Ok(*expert_offsets.last().unwrap() as usize)
}

pub fn moe_grouped_nvfp4_gemm_bf16(
    ctx: &WgpuContext,
    a_packed: &[u8],
    a_scales: &[u8],
    b_packed: &[u8],
    b_scales: &[u8],
    expert_offsets: &[i32],
    expert_ids: &[i32],
    alphas: &[f32],
    d: &mut [u16],
    n: usize,
    k: usize,
    e_total: usize,
    path: GemmPath,
) -> Result<GemmPath> {
    let path = resolve_path(ctx, path)?;
    let total_m = validate_groups(expert_offsets, expert_ids, alphas, e_total)?;
    if total_m == 0 || n == 0 || k == 0 {
        dispatch::check_len("moe grouped gemm d", d.len(), total_m * n)?;
        return Ok(path);
    }
    if !k.is_multiple_of(NVFP4_BLOCK_SIZE) {
        return Err(WgpuError::Shape(format!(
            "K {k} is not a multiple of {NVFP4_BLOCK_SIZE}"
        )));
    }
    let k_blocks = k / NVFP4_BLOCK_SIZE;
    let row_words = k / 8;
    let b_sf_stride = swizzled_scale_len(n, k_blocks);
    dispatch::check_len("moe grouped gemm a_packed", a_packed.len(), total_m * k / 2)?;
    dispatch::check_len(
        "moe grouped gemm b_packed",
        b_packed.len(),
        e_total * n * k / 2,
    )?;
    dispatch::check_len("moe grouped gemm d", d.len(), total_m * n)?;
    let a_sf_want = swizzled_scale_len(total_m, k_blocks);
    if a_scales.len() < a_sf_want {
        return Err(WgpuError::Shape(format!(
            "moe grouped gemm a_scales: got {} want at least {a_sf_want}",
            a_scales.len()
        )));
    }
    let b_sf_want = e_total * b_sf_stride;
    if b_scales.len() < b_sf_want {
        return Err(WgpuError::Shape(format!(
            "moe grouped gemm b_scales: got {} want at least {b_sf_want}",
            b_scales.len()
        )));
    }
    if ctx.caps.max_storage_buffers_per_shader_stage < 7 {
        return Err(WgpuError::Unsupported(format!(
            "moe grouped gemm needs 7 storage bindings in one stage; device allows {}",
            ctx.caps.max_storage_buffers_per_shader_stage
        )));
    }

    let a_words = bytes_to_words(a_packed);
    let a_sf_words = bytes_to_words(a_scales);
    let b_words = bytes_to_words(b_packed);
    let b_sf_words = bytes_to_words(b_scales);
    let meta: Vec<u32> = expert_ids
        .iter()
        .zip(alphas.iter())
        .flat_map(|(id, a)| [*id as u32, a.to_bits()])
        .collect();

    let a_buf = dispatch::storage_from_slice(ctx, "moe-gemm-a", &a_words);
    let a_sf_buf = dispatch::storage_from_slice(ctx, "moe-gemm-a-sf", &a_sf_words);
    let b_buf = dispatch::storage_from_slice(ctx, "moe-gemm-b", &b_words);
    let b_sf_buf = dispatch::storage_from_slice(ctx, "moe-gemm-b-sf", &b_sf_words);
    let d_buf = dispatch::storage_zeroed(ctx, "moe-gemm-d", (total_m * n * 4) as u64);
    let meta_buf = dispatch::storage_from_slice(ctx, "moe-gemm-meta", &meta);

    let (groups, source, entry, label, map, total_tiles, variant) = match path {
        GemmPath::Scalar | GemmPath::Auto => {
            let variant = select_scalar_variant(n, k);
            (
                dispatch::workgroup_count_1d(ctx, (total_m * n) as u64, SCALAR_WORKGROUP_SIZE),
                scalar_source(),
                variant.entry(),
                "moe-gemm-scalar",
                row_group_map(expert_offsets, total_m),
                0usize,
                Some(variant),
            )
        }
        GemmPath::CoopMat => {
            let table = tile_table(expert_offsets, n);
            let tiles = table.len() / 4;
            (
                dispatch::workgroup_count_1d(ctx, tiles as u64, 1),
                coop_source(),
                COOP_ENTRY,
                "moe-gemm-coop",
                table,
                tiles,
                None,
            )
        }
    };
    let map_buf = dispatch::storage_from_slice(ctx, "moe-gemm-map", &map);

    let params = MoeGemmParams {
        n: n as u32,
        k: k as u32,
        row_words: row_words as u32,
        k_tiles: k_tiles(k_blocks) as u32,
        b_sf_stride_bytes: b_sf_stride as u32,
        b_words_per_expert: (n * row_words) as u32,
        total_m: total_m as u32,
        groups_x: groups.0,
        total_tiles: total_tiles as u32,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    let params_buf = dispatch::uniform_from(ctx, "moe-gemm-params", &params);

    let coop_bindings = [
        (0u32, &a_buf),
        (1, &a_sf_buf),
        (2, &b_buf),
        (3, &b_sf_buf),
        (4, &params_buf),
        (5, &d_buf),
        (6, &meta_buf),
        (7, &map_buf),
    ];
    let bindings: Vec<(u32, &wgpu::Buffer)> = match variant {
        None => coop_bindings.to_vec(),
        Some(v) => variant_bindings(
            v,
            &a_buf,
            &a_sf_buf,
            &b_buf,
            &b_sf_buf,
            &params_buf,
            &d_buf,
            &meta_buf,
            &map_buf,
        ),
    };
    dispatch::run(ctx, label, &source, entry, &bindings, groups)?;

    let words: Vec<u32> = dispatch::read_back(ctx, &d_buf, total_m * n)?;
    for (dst, w) in d.iter_mut().zip(words.iter()) {
        *dst = (*w & 0xffff) as u16;
    }
    Ok(path)
}

#[allow(clippy::type_complexity)]
fn variant_bindings<'a>(
    variant: ScalarVariant,
    a: &'a wgpu::Buffer,
    a_sf: &'a wgpu::Buffer,
    b: &'a wgpu::Buffer,
    b_sf: &'a wgpu::Buffer,
    params: &'a wgpu::Buffer,
    d: &'a wgpu::Buffer,
    meta: &'a wgpu::Buffer,
    map: &'a wgpu::Buffer,
) -> Vec<(u32, &'a wgpu::Buffer)> {
    match variant {
        ScalarVariant::Base | ScalarVariant::Hoist => vec![
            (0, a),
            (1, a_sf),
            (2, b),
            (3, b_sf),
            (4, params),
            (5, d),
            (6, meta),
            (7, map),
        ],
        ScalarVariant::HoistV4 | ScalarVariant::Quad | ScalarVariant::QuadBits => vec![
            (1, a_sf),
            (3, b_sf),
            (4, params),
            (5, d),
            (6, meta),
            (7, map),
            (8, a),
            (9, b),
        ],
        ScalarVariant::SharedA => vec![
            (0, a),
            (1, a_sf),
            (3, b_sf),
            (4, params),
            (5, d),
            (6, meta),
            (7, map),
            (9, b),
        ],
    }
}

pub fn moe_grouped_scalar_probe(
    ctx: &WgpuContext,
    a_packed: &[u8],
    a_scales: &[u8],
    b_packed: &[u8],
    b_scales: &[u8],
    expert_offsets: &[i32],
    expert_id_sets: &[Vec<i32>],
    alphas: &[f32],
    n: usize,
    k: usize,
    e_total: usize,
    variant: ScalarVariant,
    warmup: usize,
    iters: usize,
) -> Result<(Vec<u16>, f64)> {
    let first = expert_id_sets
        .first()
        .ok_or_else(|| WgpuError::Shape("probe needs at least one expert id set".into()))?;
    let total_m = validate_groups(expert_offsets, first, alphas, e_total)?;
    if total_m == 0 || n == 0 || k == 0 || !k.is_multiple_of(NVFP4_BLOCK_SIZE) {
        return Err(WgpuError::Shape(format!(
            "probe needs non-empty aligned operands; total_m={total_m} n={n} k={k}"
        )));
    }
    if !variant.supports(n, k) {
        return Err(WgpuError::Shape(format!(
            "{:?} does not support n={n} k={k}",
            variant
        )));
    }
    let k_blocks = k / NVFP4_BLOCK_SIZE;
    let row_words = k / 8;
    let b_sf_stride = swizzled_scale_len(n, k_blocks);
    dispatch::check_len("probe a_packed", a_packed.len(), total_m * k / 2)?;
    dispatch::check_len("probe b_packed", b_packed.len(), e_total * n * k / 2)?;

    let a_buf = dispatch::storage_from_slice(ctx, "moe-probe-a", &bytes_to_words(a_packed));
    let a_sf_buf = dispatch::storage_from_slice(ctx, "moe-probe-a-sf", &bytes_to_words(a_scales));
    let b_buf = dispatch::storage_from_slice(ctx, "moe-probe-b", &bytes_to_words(b_packed));
    let b_sf_buf = dispatch::storage_from_slice(ctx, "moe-probe-b-sf", &bytes_to_words(b_scales));
    let d_buf = dispatch::storage_zeroed(ctx, "moe-probe-d", (total_m * n * 4) as u64);
    let map = row_group_map(expert_offsets, total_m);
    let map_buf = dispatch::storage_from_slice(ctx, "moe-probe-map", &map);

    let groups = dispatch::workgroup_count_1d(ctx, (total_m * n) as u64, SCALAR_WORKGROUP_SIZE);
    let params = MoeGemmParams {
        n: n as u32,
        k: k as u32,
        row_words: row_words as u32,
        k_tiles: k_tiles(k_blocks) as u32,
        b_sf_stride_bytes: b_sf_stride as u32,
        b_words_per_expert: (n * row_words) as u32,
        total_m: total_m as u32,
        groups_x: groups.0,
        total_tiles: 0,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    let params_buf = dispatch::uniform_from(ctx, "moe-probe-params", &params);

    let pipeline = dispatch::cached_compute_pipeline(
        ctx,
        "moe-gemm-scalar-probe",
        &scalar_source(),
        variant.entry(),
    )?;
    let mut bind_groups = Vec::new();
    for ids in expert_id_sets {
        validate_groups(expert_offsets, ids, alphas, e_total)?;
        let meta: Vec<u32> = ids
            .iter()
            .zip(alphas.iter())
            .flat_map(|(id, a)| [*id as u32, a.to_bits()])
            .collect();
        let meta_buf = dispatch::storage_from_slice(ctx, "moe-probe-meta", &meta);
        let bindings = variant_bindings(
            variant,
            &a_buf,
            &a_sf_buf,
            &b_buf,
            &b_sf_buf,
            &params_buf,
            &d_buf,
            &meta_buf,
            &map_buf,
        );
        bind_groups.push(dispatch::bind_group(ctx, &pipeline, &bindings));
    }

    let submit = |count: usize| {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for i in 0..count {
                pass.set_bind_group(0, &bind_groups[i % bind_groups.len()], &[]);
                pass.dispatch_workgroups(groups.0, groups.1, groups.2);
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

    let words: Vec<u32> = dispatch::read_back(ctx, &d_buf, total_m * n)?;
    let out = words.iter().map(|w| (*w & 0xffff) as u16).collect();
    Ok((out, secs))
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
    fn both_sources_include_the_dequant_prelude() {
        assert!(scalar_source().contains("fn nvfp4_scale_byte_index("));
        assert!(coop_source().contains("fn nvfp4_scale_byte_index("));
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
    fn params_are_forty_eight_bytes() {
        assert_eq!(std::mem::size_of::<MoeGemmParams>(), 48);
    }

    #[test]
    fn row_group_map_covers_ragged_and_empty_groups() {
        let map = row_group_map(&[0, 2, 2, 5], 5);
        assert_eq!(map, vec![0, 0, 2, 2, 2]);
    }

    #[test]
    fn tile_table_skips_empty_groups_and_masks_ragged_rows() {
        let table = tile_table(&[0, 33, 33, 40], 40);
        assert_eq!(table.len() / 4, (2 + 1) * 2);
        assert_eq!(&table[0..4], &[0, 0, 33, 0]);
        assert_eq!(&table[4..8], &[0, 0, 33, 32]);
        assert_eq!(&table[8..12], &[0, 32, 33, 0]);
        assert_eq!(&table[16..20], &[2, 33, 40, 0]);
        assert!(table.iter().skip(2).step_by(4).all(|&hi| hi <= 40));
    }
}
