#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};
use crate::wgpu_backend::pack::{pack_u16_pairs as pack_u16};

pub const WGSL: &str = include_str!("../../../wgsl/gemv_w4a16_m1_proto.wgsl");

pub const GROUP_SIZE: usize = 32;
pub const MAX_V_STEPS: [u32; 5] = [1, 2, 3, 5, 10];

const SCRATCH_BYTES: u32 = 512 * 4 + 16 * 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variant {
    pub warps: u32,
    pub split: u32,
    pub stream: bool,
}

pub const VARIANTS: [Variant; 8] = [
    Variant {
        warps: 8,
        split: 1,
        stream: true,
    },
    Variant {
        warps: 8,
        split: 1,
        stream: false,
    },
    Variant {
        warps: 4,
        split: 1,
        stream: true,
    },
    Variant {
        warps: 16,
        split: 1,
        stream: true,
    },
    Variant {
        warps: 8,
        split: 2,
        stream: true,
    },
    Variant {
        warps: 16,
        split: 2,
        stream: true,
    },
    Variant {
        warps: 8,
        split: 4,
        stream: true,
    },
    Variant {
        warps: 16,
        split: 4,
        stream: true,
    },
];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ProtoParams {
    n_rows: u32,
    kv: u32,
    w_row_words: u32,
    split: u32,
    rows_per_group: u32,
    max_v: u32,
    groups_x: u32,
    reserved: u32,
}

pub fn variant_for(variant: u32) -> Result<Variant> {
    VARIANTS
        .get(variant as usize)
        .copied()
        .ok_or_else(|| WgpuError::Shape(format!("gemv_w4a16_m1_proto: unknown variant {variant}")))
}

pub fn entry_for(warps: u32) -> Result<&'static str> {
    match warps {
        4 => Ok("gemv_w4a16_m1_proto_w4"),
        8 => Ok("gemv_w4a16_m1_proto_w8"),
        16 => Ok("gemv_w4a16_m1_proto_w16"),
        _ => Err(WgpuError::Unsupported(format!(
            "gemv_w4a16_m1_proto has no entry point for {warps} warps"
        ))),
    }
}

pub fn max_v_for(kv: u32, split: u32) -> Result<u32> {
    let needed = kv.div_ceil(32 * split);
    for step in MAX_V_STEPS {
        if needed <= step {
            return Ok(step);
        }
    }
    Err(WgpuError::Unsupported(format!(
        "gemv_w4a16_m1_proto needs {needed} vector steps; the CUDA launcher stops at {}",
        MAX_V_STEPS[MAX_V_STEPS.len() - 1]
    )))
}

fn check_device(ctx: &WgpuContext, block: u32) -> Result<()> {
    dispatch::require_workgroup_and_scratch(ctx, "gemv_w4a16_m1_proto", block, SCRATCH_BYTES)
}

fn widen_u16(src: &[u16]) -> Vec<u32> {
    src.iter().map(|v| *v as u32).collect()
}

pub fn gemv_w4a16_m1_proto(
    ctx: &WgpuContext,
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    y: &mut [u16],
    n: usize,
    k: usize,
    variant: u32,
) -> Result<()> {
    if n == 0 || k == 0 {
        return Ok(());
    }
    if !k.is_multiple_of(32) {
        return Err(WgpuError::Shape(format!(
            "gemv_w4a16_m1_proto requires K%32==0 with GS={GROUP_SIZE}; got K={k}"
        )));
    }
    let v = variant_for(variant)?;
    let entry = entry_for(v.warps)?;
    let block = v.warps * 32;
    let rows_per_group = v.warps / v.split;
    dispatch::check_len("gemv_w4a16_m1_proto packed", packed.len(), n * (k / 8))?;
    dispatch::check_len(
        "gemv_w4a16_m1_proto scale",
        scales.len(),
        n * (k / GROUP_SIZE),
    )?;
    dispatch::check_len("gemv_w4a16_m1_proto x", x.len(), k)?;
    dispatch::check_len("gemv_w4a16_m1_proto y", y.len(), n)?;
    check_device(ctx, block)?;

    let kv = (k / 32) as u32;
    let max_v = max_v_for(kv, v.split)?;
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, rows_per_group);
    let params = ProtoParams {
        n_rows: n as u32,
        kv,
        w_row_words: (k / 8) as u32,
        split: v.split,
        rows_per_group,
        max_v,
        groups_x: groups.0,
        reserved: 0,
    };

    let pb = dispatch::storage_from_slice(ctx, "proto-packed", packed);
    let sb = dispatch::storage_from_slice(ctx, "proto-scale", &widen_u16(scales));
    let xb = dispatch::storage_from_slice(ctx, "proto-x", &pack_u16(x));
    let yb = dispatch::storage_zeroed(ctx, "proto-y", (n * 4) as u64);
    let ub = dispatch::uniform_from(ctx, "proto-params", &params);

    dispatch::run(
        ctx,
        "nv_kernels_gemv_w4a16_m1_proto",
        &compose(WGSL),
        entry,
        &[(0, &pb), (1, &sb), (2, &xb), (3, &yb), (4, &ub)],
        groups,
    )?;

    let words: Vec<u32> = dispatch::read_back(ctx, &yb, n)?;
    for (dst, word) in y.iter_mut().zip(words.iter()) {
        *dst = (*word & 0xffff) as u16;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_table_matches_the_cuda_switch() {
        assert_eq!(
            variant_for(0).unwrap(),
            Variant {
                warps: 8,
                split: 1,
                stream: true
            }
        );
        assert_eq!(
            variant_for(1).unwrap(),
            Variant {
                warps: 8,
                split: 1,
                stream: false
            }
        );
        assert_eq!(
            variant_for(7).unwrap(),
            Variant {
                warps: 16,
                split: 4,
                stream: true
            }
        );
        assert!(variant_for(8).is_err());
    }

    #[test]
    fn entry_points_cover_every_warp_count() {
        for v in VARIANTS {
            assert!(WGSL.contains(entry_for(v.warps).unwrap()));
        }
    }

    #[test]
    fn max_v_rounds_up_to_the_cuda_template_steps() {
        assert_eq!(max_v_for(32, 1).unwrap(), 1);
        assert_eq!(max_v_for(33, 1).unwrap(), 2);
        assert_eq!(max_v_for(320, 1).unwrap(), 10);
        assert_eq!(max_v_for(320, 4).unwrap(), 3);
        assert!(max_v_for(321, 1).is_err());
    }

    #[test]
    fn params_are_uniform_buffer_sized() {
        assert_eq!(std::mem::size_of::<ProtoParams>() % 16, 0);
    }
}
