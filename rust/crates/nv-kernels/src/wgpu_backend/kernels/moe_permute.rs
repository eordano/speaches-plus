#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dispatch;
use crate::wgpu_backend::{Result, WgpuError};

pub const WGSL: &str = include_str!("../../../wgsl/moe_permute.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;
pub const ENTRY_COUNT: &str = "moe_permute_count";
pub const ENTRY_SCAN: &str = "moe_permute_scan";
pub const ENTRY_BLOCK_SCAN: &str = "moe_permute_block_scan";
pub const ENTRY_ASSIGN: &str = "moe_permute_assign";

const TILE_BYTES: u32 = WORKGROUP_SIZE * 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    total: u32,
    k: u32,
    num_experts: u32,
    num_blocks: u32,
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    if ctx.caps.max_compute_invocations_per_workgroup < WORKGROUP_SIZE
        || ctx.caps.max_compute_workgroup_size_x < WORKGROUP_SIZE
    {
        return Err(WgpuError::Unsupported(format!(
            "moe_permute needs a {WORKGROUP_SIZE}-invocation workgroup; device allows {} (x max {})",
            ctx.caps.max_compute_invocations_per_workgroup, ctx.caps.max_compute_workgroup_size_x
        )));
    }
    if !ctx.caps.workgroup_storage_fits(TILE_BYTES) {
        return Err(WgpuError::Unsupported(format!(
            "moe_permute tile needs {TILE_BYTES} bytes of workgroup storage; device allows {}",
            ctx.caps.max_compute_workgroup_storage_size
        )));
    }
    if ctx.caps.max_storage_buffers_per_shader_stage < 5 {
        return Err(WgpuError::Unsupported(format!(
            "moe_permute needs 5 storage bindings in one stage; device allows {}",
            ctx.caps.max_storage_buffers_per_shader_stage
        )));
    }
    Ok(())
}

pub fn moe_permute(
    ctx: &WgpuContext,
    topk_ids: &[i32],
    expert_offsets: &mut [i32],
    perm: &mut [i32],
    inv_perm: &mut [i32],
    n_tokens: usize,
    k: usize,
    num_experts: usize,
) -> Result<()> {
    if n_tokens == 0 || k == 0 || num_experts == 0 {
        return Ok(());
    }
    let total = n_tokens * k;
    dispatch::check_len("moe_permute topk_ids", topk_ids.len(), total)?;
    dispatch::check_len(
        "moe_permute expert_offsets",
        expert_offsets.len(),
        num_experts + 1,
    )?;
    dispatch::check_len("moe_permute inv_perm", inv_perm.len(), total)?;
    if perm.len() < total {
        return Err(WgpuError::Shape(format!(
            "moe_permute perm: got {} want at least {total}",
            perm.len()
        )));
    }
    if total > u32::MAX as usize || num_experts > u32::MAX as usize {
        return Err(WgpuError::Shape(format!(
            "moe_permute: total {total} / num_experts {num_experts} exceed u32 range"
        )));
    }
    check_device(ctx)?;

    let num_blocks = total.div_ceil(WORKGROUP_SIZE as usize);
    let params = Params {
        total: total as u32,
        k: k as u32,
        num_experts: num_experts as u32,
        num_blocks: num_blocks as u32,
    };

    let ids_buf = dispatch::storage_from_slice(ctx, "moe_permute.topk_ids", topk_ids);
    let counts_buf = dispatch::storage_zeroed(ctx, "moe_permute.counts", (num_experts * 4) as u64);
    let block_counts_buf = dispatch::storage_zeroed(
        ctx,
        "moe_permute.block_counts",
        (num_blocks * num_experts * 4) as u64,
    );
    let offsets_buf = dispatch::storage_zeroed(
        ctx,
        "moe_permute.expert_offsets",
        ((num_experts + 1) * 4) as u64,
    );
    let perm_buf = dispatch::storage_zeroed(ctx, "moe_permute.perm", (total * 4) as u64);
    let inv_buf = dispatch::storage_zeroed(ctx, "moe_permute.inv_perm", (total * 4) as u64);
    let params_buf = dispatch::uniform_from(ctx, "moe_permute.params", &params);

    let token_groups = dispatch::workgroup_count_1d(ctx, total as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        "moe_permute_count",
        WGSL,
        ENTRY_COUNT,
        &[
            (0, &ids_buf),
            (1, &counts_buf),
            (2, &block_counts_buf),
            (6, &params_buf),
        ],
        token_groups,
    )?;

    dispatch::run(
        ctx,
        "moe_permute_scan",
        WGSL,
        ENTRY_SCAN,
        &[(1, &counts_buf), (3, &offsets_buf), (6, &params_buf)],
        (1, 1, 1),
    )?;

    let expert_groups = dispatch::workgroup_count_1d(ctx, num_experts as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        "moe_permute_block_scan",
        WGSL,
        ENTRY_BLOCK_SCAN,
        &[(2, &block_counts_buf), (3, &offsets_buf), (6, &params_buf)],
        expert_groups,
    )?;

    dispatch::run(
        ctx,
        "moe_permute_assign",
        WGSL,
        ENTRY_ASSIGN,
        &[
            (0, &ids_buf),
            (2, &block_counts_buf),
            (4, &perm_buf),
            (5, &inv_buf),
            (6, &params_buf),
        ],
        token_groups,
    )?;

    let offsets_out: Vec<i32> = dispatch::read_back(ctx, &offsets_buf, num_experts + 1)?;
    expert_offsets.copy_from_slice(&offsets_out);
    let perm_out: Vec<i32> = dispatch::read_back(ctx, &perm_buf, total)?;
    perm[..total].copy_from_slice(&perm_out);
    let inv_out: Vec<i32> = dispatch::read_back(ctx, &inv_buf, total)?;
    inv_perm.copy_from_slice(&inv_out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sequential(ids: &[i32], k: usize, e: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let mut counts = vec![0i32; e];
        for &x in ids {
            counts[x as usize] += 1;
        }
        let mut off = vec![0i32; e + 1];
        for i in 0..e {
            off[i + 1] = off[i] + counts[i];
        }
        let mut cur = vec![0i32; e];
        let mut perm = vec![0i32; ids.len()];
        let mut inv = vec![0i32; ids.len()];
        for (t, &x) in ids.iter().enumerate() {
            let ex = x as usize;
            let pos = (off[ex] + cur[ex]) as usize;
            cur[ex] += 1;
            perm[pos] = (t / k) as i32;
            inv[t] = pos as i32;
        }
        (off, perm, inv)
    }

    fn blocked(ids: &[i32], k: usize, e: usize, block: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let nb = ids.len().div_ceil(block);
        let mut counts = vec![0u32; e];
        let mut bc = vec![0u32; nb * e];
        for (t, &x) in ids.iter().enumerate() {
            counts[x as usize] += 1;
            bc[(t / block) * e + x as usize] += 1;
        }
        let mut off = vec![0i32; e + 1];
        let mut acc = 0u32;
        for i in 0..e {
            acc += counts[i];
            off[i + 1] = acc as i32;
        }
        for ex in 0..e {
            let mut base = off[ex] as u32;
            for b in 0..nb {
                let c = bc[b * e + ex];
                bc[b * e + ex] = base;
                base += c;
            }
        }
        let mut perm = vec![0i32; ids.len()];
        let mut inv = vec![0i32; ids.len()];
        for (t, &x) in ids.iter().enumerate() {
            let b = t / block;
            let i = t % block;
            let lo = b * block;
            let rank = ids[lo..t].iter().filter(|&&y| y == x).count() as u32;
            debug_assert!(i < block);
            let pos = (bc[b * e + x as usize] + rank) as usize;
            perm[pos] = (t / k) as i32;
            inv[t] = pos as i32;
        }
        (off, perm, inv)
    }

    #[test]
    fn blocked_rank_matches_sequential_cursors() {
        let mut seed: u64 = 0x243f6a8885a308d3;
        let (n, k, e) = (37usize, 3usize, 11usize);
        let ids: Vec<i32> = (0..n * k)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) as u32 % e as u32) as i32
            })
            .collect();
        for block in [1usize, 4, 8, 256] {
            assert_eq!(
                sequential(&ids, k, e),
                blocked(&ids, k, e, block),
                "block={block}"
            );
        }
    }

    #[test]
    fn params_are_sixteen_bytes() {
        assert_eq!(std::mem::size_of::<Params>(), 16);
    }
}
