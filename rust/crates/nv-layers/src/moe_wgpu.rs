use anyhow::{Context, Result};

use nv_kernels::wgpu_backend::compose;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch::{self, Chain, GpuBind, GpuTensor, GpuUniform};
use nv_kernels::wgpu_backend::kernels::gather_rows_bf16 as wg_gather;
use nv_kernels::wgpu_backend::kernels::gemm_nvfp4::swizzled_scale_len;
use nv_kernels::wgpu_backend::kernels::moe_grouped_gemm as wg_moe;
use nv_kernels::wgpu_backend::kernels::moe_grouped_gemm::{select_scalar_variant, ScalarVariant};
use nv_kernels::wgpu_backend::kernels::moe_unpermute_scatter as wg_mus;
use nv_kernels::wgpu_backend::kernels::quantize_nvfp4_bf16 as wg_quant;

pub const MIN_TILE: usize = nv_quant::nvfp4::MIN_TILE;
pub const BLOCK_SIZE: usize = nv_quant::nvfp4::BLOCK_SIZE;

const MODE_PLAIN: u32 = 0;
const MODE_SILU_MUL: u32 = 1;

const PACK_WGSL: &str = r#"
struct PackParams {
    words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> pk_src: array<u32>;
@group(0) @binding(1) var<storage, read_write> pk_dst: array<u32>;
@group(0) @binding(2) var<uniform> pk_params: PackParams;

@compute @workgroup_size(256)
fn pack_bf16_pairs(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_count: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = (wg_id.x + wg_id.y * wg_count.x) * 256u + lid.x;
    if (i >= pk_params.words) {
        return;
    }
    let lo = pk_src[2u * i] & 0xffffu;
    let hi = pk_src[2u * i + 1u] & 0xffffu;
    pk_dst[i] = lo | (hi << 16u);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherParams {
    m_total_padded: u32,
    hidden_words: u32,
    n_tokens: i32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuantizeParams {
    rows: u32,
    m_data_rows: u32,
    m_read_rows: u32,
    k: u32,
    k_tiles: u32,
    blocks_per_row: u32,
    rows_per_expert: u32,
    mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmParams {
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackParams {
    words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MusParams {
    n_tokens: u32,
    k: u32,
    hidden: u32,
    row_stride: u32,
    hidden_tiles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

pub struct MoeWgpuExpertSource<'a> {
    pub gate_packed: &'a [u8],
    pub gate_scales_swizzled: &'a [u8],
    pub gate_alpha: f32,
    pub up_packed: &'a [u8],
    pub up_scales_swizzled: &'a [u8],
    pub up_alpha: f32,
    pub down_packed: &'a [u8],
    pub down_scales_swizzled: &'a [u8],
    pub down_alpha: f32,
    pub input_global_gate_up: f32,
    pub input_global_down: f32,
}

struct MatWeights {
    packed: GpuTensor<u32>,
    scales: GpuTensor<u32>,
    alphas: Vec<f32>,
    n: usize,
    k: usize,
}

pub struct MoeWgpuWeights {
    num_experts: usize,
    hidden_size: usize,
    intermediate_size: usize,
    gate: MatWeights,
    up: MatWeights,
    down: MatWeights,
    input_globals_gate_up: Vec<f32>,
    input_globals_down: Vec<f32>,
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    nv_kernels::wgpu_backend::pack::pack_u8_min_one_word(bytes)
}

fn pack_u16_words(src: &[u16]) -> Vec<u32> {
    let mut out: Vec<u32> = src
        .chunks_exact(2)
        .map(|c| (c[0] as u32) | ((c[1] as u32) << 16))
        .collect();
    if out.is_empty() {
        out.push(0);
    }
    out
}

fn upload_mat(
    ctx: &WgpuContext,
    label: &str,
    per_expert_packed: &[&[u8]],
    per_expert_scales: &[&[u8]],
    alphas: Vec<f32>,
    n: usize,
    k: usize,
) -> Result<MatWeights> {
    let e = per_expert_packed.len();
    let packed_bytes = n * k / 2;
    let scale_bytes = swizzled_scale_len(n, k / BLOCK_SIZE);
    let mut packed = Vec::with_capacity(e * packed_bytes);
    let mut scales = Vec::with_capacity(e * scale_bytes);
    for (i, (p, s)) in per_expert_packed
        .iter()
        .zip(per_expert_scales.iter())
        .enumerate()
    {
        anyhow::ensure!(
            p.len() == packed_bytes,
            "{label} expert {i}: packed {} bytes, want {packed_bytes}",
            p.len()
        );
        anyhow::ensure!(
            s.len() == scale_bytes,
            "{label} expert {i}: scales {} bytes, want {scale_bytes}",
            s.len()
        );
        packed.extend_from_slice(p);
        scales.extend_from_slice(s);
    }
    Ok(MatWeights {
        packed: GpuTensor::upload(ctx, label, &bytes_to_words(&packed)),
        scales: GpuTensor::upload(ctx, &format!("{label}-sf"), &bytes_to_words(&scales)),
        alphas,
        n,
        k,
    })
}

impl MoeWgpuWeights {
    pub fn from_expert_sources(
        ctx: &WgpuContext,
        hidden_size: usize,
        intermediate_size: usize,
        sources: &[MoeWgpuExpertSource<'_>],
    ) -> Result<Self> {
        let e = sources.len();
        anyhow::ensure!(e > 0, "MoeWgpuWeights: no experts");
        anyhow::ensure!(
            hidden_size.is_multiple_of(64)
                && intermediate_size.is_multiple_of(BLOCK_SIZE)
                && intermediate_size.is_multiple_of(2),
            "MoeWgpuWeights: hidden {hidden_size} / intermediate {intermediate_size} misaligned"
        );
        for (i, s) in sources.iter().enumerate() {
            anyhow::ensure!(
                s.input_global_gate_up.is_finite() && s.input_global_down.is_finite(),
                "expert {i}: non-finite input global scale"
            );
        }
        let gate = upload_mat(
            ctx,
            "moe-wgpu-gate",
            &sources.iter().map(|s| s.gate_packed).collect::<Vec<_>>(),
            &sources
                .iter()
                .map(|s| s.gate_scales_swizzled)
                .collect::<Vec<_>>(),
            sources.iter().map(|s| s.gate_alpha).collect(),
            intermediate_size,
            hidden_size,
        )?;
        let up = upload_mat(
            ctx,
            "moe-wgpu-up",
            &sources.iter().map(|s| s.up_packed).collect::<Vec<_>>(),
            &sources
                .iter()
                .map(|s| s.up_scales_swizzled)
                .collect::<Vec<_>>(),
            sources.iter().map(|s| s.up_alpha).collect(),
            intermediate_size,
            hidden_size,
        )?;
        let down = upload_mat(
            ctx,
            "moe-wgpu-down",
            &sources.iter().map(|s| s.down_packed).collect::<Vec<_>>(),
            &sources
                .iter()
                .map(|s| s.down_scales_swizzled)
                .collect::<Vec<_>>(),
            sources.iter().map(|s| s.down_alpha).collect(),
            hidden_size,
            intermediate_size,
        )?;
        Ok(Self {
            num_experts: e,
            hidden_size,
            intermediate_size,
            gate,
            up,
            down,
            input_globals_gate_up: sources.iter().map(|s| s.input_global_gate_up).collect(),
            input_globals_down: sources.iter().map(|s| s.input_global_down).collect(),
        })
    }

    pub fn from_loader(
        ctx: &WgpuContext,
        prefix: &str,
        weights: &nv_weights::WeightLoader,
        num_experts: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        #[allow(clippy::type_complexity)]
        let mut raw: Vec<(
            Vec<u8>,
            Vec<u8>,
            f32,
            f32,
            Vec<u8>,
            Vec<u8>,
            f32,
            f32,
            Vec<u8>,
            Vec<u8>,
            f32,
            f32,
        )> = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            let g = read_nvfp4_module(
                weights,
                &format!("{prefix}.experts.{i}.gate_proj"),
                intermediate_size,
                hidden_size,
            )?;
            let u = read_nvfp4_module(
                weights,
                &format!("{prefix}.experts.{i}.up_proj"),
                intermediate_size,
                hidden_size,
            )?;
            let d = read_nvfp4_module(
                weights,
                &format!("{prefix}.experts.{i}.down_proj"),
                hidden_size,
                intermediate_size,
            )?;
            anyhow::ensure!(
                (g.3 - u.3).abs() <= 1e-6,
                "expert {i}: gate and up input_global_scale differ ({} vs {})",
                g.3,
                u.3
            );
            raw.push((g.0, g.1, g.2, g.3, u.0, u.1, u.2, u.3, d.0, d.1, d.2, d.3));
        }
        let sources: Vec<MoeWgpuExpertSource<'_>> = raw
            .iter()
            .map(|r| MoeWgpuExpertSource {
                gate_packed: &r.0,
                gate_scales_swizzled: &r.1,
                gate_alpha: r.2,
                up_packed: &r.4,
                up_scales_swizzled: &r.5,
                up_alpha: r.6,
                down_packed: &r.8,
                down_scales_swizzled: &r.9,
                down_alpha: r.10,
                input_global_gate_up: r.3,
                input_global_down: r.11,
            })
            .collect();
        Self::from_expert_sources(ctx, hidden_size, intermediate_size, &sources)
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
}

fn read_nvfp4_module(
    weights: &nv_weights::WeightLoader,
    module: &str,
    out_features: usize,
    in_features: usize,
) -> Result<(Vec<u8>, Vec<u8>, f32, f32)> {
    use candle_core::DType;
    let packed_name = format!("{module}.weight_packed");
    let scale_name = format!("{module}.weight_scale");
    let gscale_name = format!("{module}.weight_global_scale");
    let input_gscale_name = format!("{module}.input_global_scale");
    let packed_shape = weights
        .shape_of(&packed_name)
        .ok_or_else(|| anyhow::anyhow!("missing {packed_name}"))?;
    anyhow::ensure!(
        packed_shape.len() == 2
            && packed_shape[0] == out_features
            && packed_shape[1] == in_features / 2,
        "{module}: weight_packed shape {:?}, want [{out_features}, {}]",
        packed_shape,
        in_features / 2
    );
    let packed = weights
        .raw_bytes(&packed_name)
        .with_context(|| format!("read {packed_name}"))?
        .to_vec();
    let scale_raw = weights
        .raw_bytes(&scale_name)
        .with_context(|| format!("read {scale_name}"))?
        .to_vec();
    anyhow::ensure!(
        scale_raw.len() == out_features * in_features / BLOCK_SIZE,
        "{module}: weight_scale {} bytes, want {}",
        scale_raw.len(),
        out_features * in_features / BLOCK_SIZE
    );
    let scales =
        nv_quant::nvfp4::swizzle_scales(&scale_raw, out_features, in_features / BLOCK_SIZE);
    let gscale = weights
        .get(&gscale_name, DType::F32)
        .with_context(|| format!("read {gscale_name}"))?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let stored_weight_global = *gscale.first().unwrap_or(&1.0);
    let stored_input_global = if weights.has(&input_gscale_name) {
        let t = weights
            .get(&input_gscale_name, DType::F32)
            .with_context(|| format!("read {input_gscale_name}"))?;
        *t.flatten_all()?.to_vec1::<f32>()?.first().unwrap_or(&1.0)
    } else {
        1.0
    };
    let safe_recip = |x: f32| {
        if x == 0.0 || !x.is_finite() {
            1.0
        } else {
            1.0 / x
        }
    };
    let alpha = safe_recip(stored_weight_global) * safe_recip(stored_input_global);
    Ok((packed, scales, alpha, stored_input_global))
}

struct RoutingPlan {
    active_experts: Vec<usize>,
    m_total_padded: usize,
    src_idx: Vec<i32>,
    inv_perm: Vec<i32>,
}

fn plan_routing(topk_ids: &[u32], n_tokens: usize, k: usize, num_experts: usize) -> RoutingPlan {
    let mut buckets: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_experts];
    for n in 0..n_tokens {
        for j in 0..k {
            let e = topk_ids[n * k + j] as usize;
            buckets[e].push((n as u32, j as u32));
        }
    }
    let mut active_experts = Vec::new();
    for (e, b) in buckets.iter().enumerate() {
        if !b.is_empty() {
            active_experts.push(e);
        }
    }
    let m_total_padded = active_experts.len() * MIN_TILE;
    let mut src_idx = vec![-1i32; m_total_padded];
    let mut inv_perm = vec![-1i32; n_tokens * k];
    for (i, &e) in active_experts.iter().enumerate() {
        let base = i * MIN_TILE;
        for (j, &(token, slot)) in buckets[e].iter().enumerate() {
            if j >= MIN_TILE {
                break;
            }
            src_idx[base + j] = token as i32;
            inv_perm[(token as usize) * k + slot as usize] = (base + j) as i32;
        }
    }
    RoutingPlan {
        active_experts,
        m_total_padded,
        src_idx,
        inv_perm,
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_bindings<'a>(
    variant: ScalarVariant,
    a: &'a dyn GpuBind,
    a_sf: &'a dyn GpuBind,
    b: &'a dyn GpuBind,
    b_sf: &'a dyn GpuBind,
    params: &'a dyn GpuBind,
    d: &'a dyn GpuBind,
    meta: &'a dyn GpuBind,
    map: &'a dyn GpuBind,
) -> Vec<(u32, &'a dyn GpuBind)> {
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

struct GemmStage<'a> {
    label: &'static str,
    mat: &'a MatWeights,
    meta: GpuTensor<u32>,
}

fn gemm_stage<'a>(
    ctx: &WgpuContext,
    label: &'static str,
    mat: &'a MatWeights,
    active: &[usize],
) -> GemmStage<'a> {
    let meta: Vec<u32> = active
        .iter()
        .flat_map(|&e| [e as u32, mat.alphas[e].to_bits()])
        .collect();
    GemmStage {
        label,
        mat,
        meta: GpuTensor::upload(ctx, label, &meta),
    }
}

#[allow(clippy::too_many_arguments)]
fn push_gemm(
    chain: &mut Chain<'_>,
    ctx: &WgpuContext,
    source: &str,
    stage: &GemmStage<'_>,
    a: &GpuTensor<u32>,
    a_sf: &GpuTensor<u32>,
    map: &GpuTensor<u32>,
    d: &GpuTensor<u32>,
    total_m: usize,
) -> Result<GpuUniform<GemmParams>, nv_kernels::wgpu_backend::WgpuError> {
    let n = stage.mat.n;
    let k = stage.mat.k;
    let k_blocks = k / BLOCK_SIZE;
    let groups =
        dispatch::workgroup_count_1d(ctx, (total_m * n) as u64, wg_moe::SCALAR_WORKGROUP_SIZE);
    let params = GemmParams {
        n: n as u32,
        k: k as u32,
        row_words: (k / 8) as u32,
        k_tiles: k_blocks.div_ceil(4) as u32,
        b_sf_stride_bytes: swizzled_scale_len(n, k_blocks) as u32,
        b_words_per_expert: (n * k / 8) as u32,
        total_m: total_m as u32,
        groups_x: groups.0,
        total_tiles: 0,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    let params_buf = GpuUniform::new(ctx, stage.label, &params);
    let variant = select_scalar_variant(n, k);
    let bindings = gemm_bindings(
        variant,
        a,
        a_sf,
        &stage.mat.packed,
        &stage.mat.scales,
        &params_buf,
        d,
        &stage.meta,
        map,
    );
    chain.push(stage.label, source, variant.entry(), &bindings, groups)?;
    Ok(params_buf)
}

#[allow(clippy::too_many_arguments)]
fn push_quantize(
    chain: &mut Chain<'_>,
    ctx: &WgpuContext,
    label: &str,
    x: &dyn GpuBind,
    y: &dyn GpuBind,
    globals: &GpuTensor<f32>,
    packed: &GpuTensor<u32>,
    scales: &GpuTensor<u32>,
    total_m: usize,
    k: usize,
    mode: u32,
) -> Result<GpuUniform<QuantizeParams>, nv_kernels::wgpu_backend::WgpuError> {
    let blocks_per_row = k / BLOCK_SIZE;
    let k_tiles = blocks_per_row.div_ceil(4);
    let params = QuantizeParams {
        rows: total_m as u32,
        m_data_rows: total_m as u32,
        m_read_rows: total_m as u32,
        k: k as u32,
        k_tiles: k_tiles as u32,
        blocks_per_row: blocks_per_row as u32,
        rows_per_expert: MIN_TILE as u32,
        mode,
    };
    let params_buf = GpuUniform::new(ctx, label, &params);
    let groups =
        dispatch::workgroup_count_1d(ctx, (total_m * k_tiles) as u64, wg_quant::WORKGROUP_SIZE);
    chain.push(
        label,
        &compose(wg_quant::WGSL),
        wg_quant::ENTRY,
        &[
            (0, x),
            (1, y),
            (2, globals),
            (3, packed),
            (4, scales),
            (5, params_buf.raw()),
        ],
        groups,
    )?;
    Ok(params_buf)
}

fn push_pack(
    chain: &mut Chain<'_>,
    ctx: &WgpuContext,
    label: &str,
    src: &GpuTensor<u32>,
    dst: &GpuTensor<u32>,
    words: usize,
) -> Result<GpuUniform<PackParams>, nv_kernels::wgpu_backend::WgpuError> {
    let params = PackParams {
        words: words as u32,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    let params_buf = GpuUniform::new(ctx, label, &params);
    let groups = dispatch::workgroup_count_1d(ctx, words as u64, 256);
    chain.push(
        label,
        PACK_WGSL,
        "pack_bf16_pairs",
        &[(0, src), (1, dst), (2, params_buf.raw())],
        groups,
    )?;
    Ok(params_buf)
}

pub fn try_forward(
    w: &MoeWgpuWeights,
    ctx: &WgpuContext,
    x_flat_bf16: &[u16],
    topk_ids: &[u32],
    topk_weights: &[f32],
    n_tokens: usize,
    k: usize,
) -> Result<Option<Vec<f32>>> {
    let hidden = w.hidden_size;
    let inter = w.intermediate_size;
    anyhow::ensure!(
        x_flat_bf16.len() == n_tokens * hidden,
        "x_flat: got {} want {}",
        x_flat_bf16.len(),
        n_tokens * hidden
    );
    anyhow::ensure!(
        topk_ids.len() == n_tokens * k && topk_weights.len() == n_tokens * k,
        "topk arrays: got ids {} weights {} want {}",
        topk_ids.len(),
        topk_weights.len(),
        n_tokens * k
    );
    for &e in topk_ids {
        anyhow::ensure!(
            (e as usize) < w.num_experts,
            "topk id {e} out of range 0..{}",
            w.num_experts
        );
    }
    if n_tokens == 0 {
        return Ok(Some(Vec::new()));
    }
    let mut counts = vec![0usize; w.num_experts];
    for &e in topk_ids {
        counts[e as usize] += 1;
    }
    if counts.iter().any(|&c| c > MIN_TILE) {
        return Ok(None);
    }
    if ctx.caps.max_storage_buffers_per_shader_stage < 7 {
        anyhow::bail!(
            "moe wgpu needs 7 storage bindings in one stage; device allows {}",
            ctx.caps.max_storage_buffers_per_shader_stage
        );
    }

    let plan = plan_routing(topk_ids, n_tokens, k, w.num_experts);
    let a = plan.active_experts.len();
    if a == 0 {
        return Ok(Some(vec![0f32; n_tokens * hidden]));
    }
    let m_total = plan.m_total_padded;

    let mut globals_gu = Vec::with_capacity(a);
    let mut globals_dn = Vec::with_capacity(a);
    for &e in &plan.active_experts {
        globals_gu.push(w.input_globals_gate_up[e]);
        globals_dn.push(w.input_globals_down[e]);
    }
    let map_host: Vec<u32> = (0..m_total).map(|r| (r / MIN_TILE) as u32).collect();

    let x_buf = GpuTensor::upload(ctx, "moe-wgpu-x", &pack_u16_words(x_flat_bf16));
    let src_idx_buf = GpuTensor::upload(ctx, "moe-wgpu-src-idx", &plan.src_idx);
    let inv_perm_buf = GpuTensor::upload(ctx, "moe-wgpu-inv-perm", &plan.inv_perm);
    let weights_buf = GpuTensor::upload(ctx, "moe-wgpu-topk-w", topk_weights);
    let globals_gu_buf = GpuTensor::upload(ctx, "moe-wgpu-glob-gu", &globals_gu);
    let globals_dn_buf = GpuTensor::upload(ctx, "moe-wgpu-glob-dn", &globals_dn);
    let map_buf = GpuTensor::upload(ctx, "moe-wgpu-map", &map_host);
    let dummy_buf: GpuTensor<u32> = GpuTensor::zeroed(ctx, "moe-wgpu-dummy", 1);

    let hidden_words = hidden / 2;
    let x_sorted_buf: GpuTensor<u32> =
        GpuTensor::zeroed(ctx, "moe-wgpu-x-sorted", m_total * hidden_words);
    let hidden_blocks = hidden / BLOCK_SIZE;
    let inter_blocks = inter / BLOCK_SIZE;
    let x_fp4_buf: GpuTensor<u32> = GpuTensor::zeroed(ctx, "moe-wgpu-x-fp4", m_total * hidden / 8);
    let x_sf_buf: GpuTensor<u32> = GpuTensor::zeroed(
        ctx,
        "moe-wgpu-x-sf",
        swizzled_scale_len(m_total, hidden_blocks) / 4,
    );
    let y_gate_buf: GpuTensor<u32> = GpuTensor::zeroed(ctx, "moe-wgpu-y-gate", m_total * inter);
    let y_up_buf: GpuTensor<u32> = GpuTensor::zeroed(ctx, "moe-wgpu-y-up", m_total * inter);
    let gate2_buf: GpuTensor<u32> = GpuTensor::zeroed(ctx, "moe-wgpu-gate2", m_total * inter / 2);
    let up2_buf: GpuTensor<u32> = GpuTensor::zeroed(ctx, "moe-wgpu-up2", m_total * inter / 2);
    let act_fp4_buf: GpuTensor<u32> =
        GpuTensor::zeroed(ctx, "moe-wgpu-act-fp4", m_total * inter / 8);
    let act_sf_buf: GpuTensor<u32> = GpuTensor::zeroed(
        ctx,
        "moe-wgpu-act-sf",
        swizzled_scale_len(m_total, inter_blocks) / 4,
    );
    let y_down_buf: GpuTensor<u32> = GpuTensor::zeroed(ctx, "moe-wgpu-y-down", m_total * hidden);
    let down2_buf: GpuTensor<u32> =
        GpuTensor::zeroed(ctx, "moe-wgpu-down2", m_total * hidden_words);
    let out_buf: GpuTensor<f32> = GpuTensor::zeroed(ctx, "moe-wgpu-out", n_tokens * hidden);

    let gemm_source = wg_moe::scalar_source();
    let gate_stage = gemm_stage(ctx, "moe-wgpu-gemm-gate", &w.gate, &plan.active_experts);
    let up_stage = gemm_stage(ctx, "moe-wgpu-gemm-up", &w.up, &plan.active_experts);
    let down_stage = gemm_stage(ctx, "moe-wgpu-gemm-down", &w.down, &plan.active_experts);

    let gather_params = GpuUniform::new(
        ctx,
        "moe-wgpu-gather-params",
        &GatherParams {
            m_total_padded: m_total as u32,
            hidden_words: hidden_words as u32,
            n_tokens: n_tokens as i32,
            pad0: 0,
        },
    );
    let hidden_tiles = (hidden as u32).div_ceil(wg_mus::WORKGROUP_SIZE);
    let mus_params = GpuUniform::new(
        ctx,
        "moe-wgpu-mus-params",
        &MusParams {
            n_tokens: n_tokens as u32,
            k: k as u32,
            hidden: hidden as u32,
            row_stride: hidden as u32,
            hidden_tiles,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        },
    );

    let err = |e: nv_kernels::wgpu_backend::WgpuError| anyhow::anyhow!("moe wgpu chain: {e}");

    let mut chain = Chain::new(ctx);
    chain
        .push(
            "moe-wgpu-gather",
            wg_gather::WGSL,
            wg_gather::ENTRY,
            &[
                (0, &x_buf),
                (1, &src_idx_buf),
                (2, &x_sorted_buf),
                (3, &gather_params),
            ],
            dispatch::workgroup_count_1d(ctx, m_total as u64, 1),
        )
        .map_err(err)?;
    let _qp = push_quantize(
        &mut chain,
        ctx,
        "moe-wgpu-quant-x",
        &x_sorted_buf,
        &dummy_buf,
        &globals_gu_buf,
        &x_fp4_buf,
        &x_sf_buf,
        m_total,
        hidden,
        MODE_PLAIN,
    )
    .map_err(err)?;
    let _gp = push_gemm(
        &mut chain,
        ctx,
        &gemm_source,
        &gate_stage,
        &x_fp4_buf,
        &x_sf_buf,
        &map_buf,
        &y_gate_buf,
        m_total,
    )
    .map_err(err)?;
    let _up = push_gemm(
        &mut chain,
        ctx,
        &gemm_source,
        &up_stage,
        &x_fp4_buf,
        &x_sf_buf,
        &map_buf,
        &y_up_buf,
        m_total,
    )
    .map_err(err)?;
    let _p1 = push_pack(
        &mut chain,
        ctx,
        "moe-wgpu-pack-gate",
        &y_gate_buf,
        &gate2_buf,
        m_total * inter / 2,
    )
    .map_err(err)?;
    let _p2 = push_pack(
        &mut chain,
        ctx,
        "moe-wgpu-pack-up",
        &y_up_buf,
        &up2_buf,
        m_total * inter / 2,
    )
    .map_err(err)?;
    let _qa = push_quantize(
        &mut chain,
        ctx,
        "moe-wgpu-quant-act",
        &gate2_buf,
        &up2_buf,
        &globals_dn_buf,
        &act_fp4_buf,
        &act_sf_buf,
        m_total,
        inter,
        MODE_SILU_MUL,
    )
    .map_err(err)?;
    let _dp = push_gemm(
        &mut chain,
        ctx,
        &gemm_source,
        &down_stage,
        &act_fp4_buf,
        &act_sf_buf,
        &map_buf,
        &y_down_buf,
        m_total,
    )
    .map_err(err)?;
    let _p3 = push_pack(
        &mut chain,
        ctx,
        "moe-wgpu-pack-down",
        &y_down_buf,
        &down2_buf,
        m_total * hidden_words,
    )
    .map_err(err)?;
    chain
        .push(
            "moe-wgpu-unpermute",
            &compose(wg_mus::WGSL),
            "moe_unpermute_scatter",
            &[
                (0, &down2_buf),
                (1, &weights_buf),
                (2, &inv_perm_buf),
                (3, &out_buf),
                (4, &mus_params),
            ],
            dispatch::workgroup_count_1d(ctx, n_tokens as u64 * hidden_tiles as u64, 1),
        )
        .map_err(err)?;
    chain.submit().map_err(err)?;

    let out = out_buf.download(ctx).map_err(err)?;
    Ok(Some(out))
}
