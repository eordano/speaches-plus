#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::pack::unpack_u8_by_element as words_to_bytes;
use crate::wgpu_backend::kernels::gemv_nvfp4;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};
use crate::wgpu_backend::pack::{pack_u16_even_min_one_word as pack_words};

pub const WGSL: &str = include_str!("../../../wgsl/quantize_nvfp4_bf16.wgsl");

pub const ENTRY: &str = "quantize_nvfp4_bf16";
pub const WORKGROUP_SIZE: u32 = 256;
pub const BLOCK_SIZE: usize = 16;
pub const SF_ROW_TILE: usize = 128;

pub const ACT_GRID_ENTRY: &str = "quantize_row_nvfp4_bf16_grid";
pub const ACT_GRID_ENTRY_WG64: &str = "quantize_row_nvfp4_bf16_grid_wg64";
pub const ACT_GRID_WG: u32 = 256;
pub const ACT_GRID_WG64: u32 = 64;

const ACT_GRID_WGSL: &str = include_str!("../../../wgsl/quantize_nvfp4_bf16_act_grid.wgsl");

pub fn act_grid_source() -> String {
    format!("{}\n{}", gemv_nvfp4::quantize_source(), ACT_GRID_WGSL)
}

pub fn act_grid_groups(k_blocks: u32, wg: u32) -> u32 {
    k_blocks.div_ceil(wg.max(1)).max(1)
}

const MODE_PLAIN: u32 = 0;
const MODE_SILU_MUL: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    rows: u32,
    m_data_rows: u32,
    m_read_rows: u32,
    k: u32,
    k_tiles: u32,
    blocks_per_row: u32,
    rows_per_expert: u32,
    mode: u32,
}

pub fn scale_rows(rows: usize) -> usize {
    rows.div_ceil(SF_ROW_TILE) * SF_ROW_TILE
}

pub fn swizzled_scale_bytes(rows: usize, k: usize) -> usize {
    scale_rows(rows) * (k / BLOCK_SIZE).div_ceil(4) * 4
}

struct Plan {
    rows: usize,
    m_data_rows: usize,
    m_read_rows: usize,
    rows_per_expert: usize,
    k: usize,
    mode: u32,
}

fn run(
    ctx: &WgpuContext,
    plan: Plan,
    x: &[u16],
    y: Option<&[u16]>,
    globals: &[f32],
    packed_out: &mut [u8],
    scales_out: &mut [u8],
) -> Result<()> {
    let Plan {
        rows,
        m_data_rows,
        m_read_rows,
        rows_per_expert,
        k,
        mode,
    } = plan;

    if k == 0 || k % BLOCK_SIZE != 0 {
        return Err(WgpuError::Shape(format!(
            "quantize_nvfp4_bf16: k must be a non-zero multiple of {BLOCK_SIZE}, got {k}"
        )));
    }
    if m_read_rows > m_data_rows || m_data_rows > rows {
        return Err(WgpuError::Shape(format!(
            "quantize_nvfp4_bf16: need m_read_rows {m_read_rows} <= m_data_rows {m_data_rows} <= rows {rows}"
        )));
    }
    if globals.is_empty() {
        return Err(WgpuError::Shape(
            "quantize_nvfp4_bf16: global scale table is empty".to_string(),
        ));
    }
    if rows == 0 {
        return Ok(());
    }
    if x.len() < m_read_rows * k {
        return Err(WgpuError::Shape(format!(
            "x_bf16: got {} want at least {}",
            x.len(),
            m_read_rows * k
        )));
    }
    if let Some(y) = y {
        if y.len() < m_read_rows * k {
            return Err(WgpuError::Shape(format!(
                "y_bf16: got {} want at least {}",
                y.len(),
                m_read_rows * k
            )));
        }
    }

    let blocks_per_row = k / BLOCK_SIZE;
    let k_tiles = blocks_per_row.div_ceil(4);
    let packed_words = m_data_rows * k / 8;
    let scale_words = scale_rows(rows) * k_tiles;
    dispatch::check_len("packed_out", packed_out.len(), m_data_rows * k / 2)?;
    dispatch::check_len("scales_out", scales_out.len(), scale_words * 4)?;

    let x_words = pack_words(&x[..m_read_rows * k]);
    let y_words = match y {
        Some(y) => pack_words(&y[..m_read_rows * k]),
        None => vec![0u32],
    };

    let x_buf = dispatch::storage_from_slice(ctx, "quantize_nvfp4_bf16.x", &x_words);
    let y_buf = dispatch::storage_from_slice(ctx, "quantize_nvfp4_bf16.y", &y_words);
    let globals_buf = dispatch::storage_from_slice(ctx, "quantize_nvfp4_bf16.globals", globals);
    let packed_buf = dispatch::storage_zeroed(
        ctx,
        "quantize_nvfp4_bf16.packed",
        (packed_words * std::mem::size_of::<u32>()) as u64,
    );
    let scales_buf = dispatch::storage_zeroed(
        ctx,
        "quantize_nvfp4_bf16.scales",
        (scale_words * std::mem::size_of::<u32>()) as u64,
    );
    let params = Params {
        rows: rows as u32,
        m_data_rows: m_data_rows as u32,
        m_read_rows: m_read_rows as u32,
        k: k as u32,
        k_tiles: k_tiles as u32,
        blocks_per_row: blocks_per_row as u32,
        rows_per_expert: rows_per_expert.max(1) as u32,
        mode,
    };
    let params_buf = dispatch::uniform_from(ctx, "quantize_nvfp4_bf16.params", &params);

    let source = compose(WGSL);
    let workgroups = dispatch::workgroup_count_1d(ctx, (rows * k_tiles) as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        "quantize_nvfp4_bf16",
        &source,
        ENTRY,
        &[
            (0, &x_buf),
            (1, &y_buf),
            (2, &globals_buf),
            (3, &packed_buf),
            (4, &scales_buf),
            (5, &params_buf),
        ],
        workgroups,
    )?;

    let packed = dispatch::read_back::<u32>(ctx, &packed_buf, packed_words)?;
    let scales = dispatch::read_back::<u32>(ctx, &scales_buf, scale_words)?;
    words_to_bytes(&packed, packed_out);
    words_to_bytes(&scales, scales_out);
    Ok(())
}

fn check_expert_offsets(
    expert_offsets: &[i32],
    num_experts: usize,
    m_per_expert: usize,
) -> Result<()> {
    if expert_offsets.is_empty() {
        return Ok(());
    }
    if expert_offsets.len() != num_experts && expert_offsets.len() != num_experts + 1 {
        return Err(WgpuError::Shape(format!(
            "expert_offsets: got {} want {} or {}",
            expert_offsets.len(),
            num_experts,
            num_experts + 1
        )));
    }
    for (e, off) in expert_offsets.iter().enumerate() {
        let want = (e * m_per_expert) as i32;
        if *off != want {
            return Err(WgpuError::Unsupported(format!(
                "quantize_nvfp4_bf16_per_expert: only uniform expert strides are supported; \
                 expert_offsets[{e}] = {off}, want {want}"
            )));
        }
    }
    Ok(())
}

pub fn quantize_nvfp4_bf16(
    ctx: &WgpuContext,
    x: &[u16],
    global_scale: f32,
    packed_out: &mut [u8],
    scales_out: &mut [u8],
    m_logical: usize,
    m_padded: usize,
    k: usize,
) -> Result<()> {
    let rows = scale_rows(m_padded);
    run(
        ctx,
        Plan {
            rows,
            m_data_rows: m_padded,
            m_read_rows: m_logical,
            rows_per_expert: rows,
            k,
            mode: MODE_PLAIN,
        },
        x,
        None,
        &[global_scale],
        packed_out,
        scales_out,
    )
}

pub fn quantize_nvfp4_bf16_per_expert(
    ctx: &WgpuContext,
    x: &[u16],
    global_scales: &[f32],
    expert_offsets: &[i32],
    packed_out: &mut [u8],
    scales_out: &mut [u8],
    num_experts: usize,
    m_padded: usize,
    k: usize,
) -> Result<()> {
    dispatch::check_len("global_scales", global_scales.len(), num_experts)?;
    check_expert_offsets(expert_offsets, num_experts, m_padded)?;
    let m_total = num_experts * m_padded;
    run(
        ctx,
        Plan {
            rows: m_total,
            m_data_rows: m_total,
            m_read_rows: m_total,
            rows_per_expert: m_padded,
            k,
            mode: MODE_PLAIN,
        },
        x,
        None,
        global_scales,
        packed_out,
        scales_out,
    )
}

pub fn silu_mul_quantize_nvfp4_bf16_per_expert(
    ctx: &WgpuContext,
    gate_up: &[u16],
    global_scales: &[f32],
    expert_offsets: &[i32],
    packed_out: &mut [u8],
    scales_out: &mut [u8],
    num_experts: usize,
    m_padded: usize,
    inter: usize,
) -> Result<()> {
    dispatch::check_len("global_scales", global_scales.len(), num_experts)?;
    check_expert_offsets(expert_offsets, num_experts, m_padded)?;
    let m_total = num_experts * m_padded;
    let half = m_total * inter;
    dispatch::check_len("gate_up", gate_up.len(), 2 * half)?;
    let (gate, up) = gate_up.split_at(half);
    run(
        ctx,
        Plan {
            rows: m_total,
            m_data_rows: m_total,
            m_read_rows: m_total,
            rows_per_expert: m_padded,
            k: inter,
            mode: MODE_SILU_MUL,
        },
        gate,
        Some(up),
        global_scales,
        packed_out,
        scales_out,
    )
}
