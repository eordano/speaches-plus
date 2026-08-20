#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};
use crate::wgpu_backend::pack::{pack_u16_pairs as pack_u16, unpack_u16_pairs as unpack_u16};

pub const WGSL: &str = include_str!("../../../wgsl/rmsnorm.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;

const SCRATCH_BYTES: u32 = WORKGROUP_SIZE * 4 + 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup_and_scratch(ctx, "rmsnorm", WORKGROUP_SIZE, SCRATCH_BYTES)
}

fn check_shapes(x: usize, weight: usize, y: usize, batch: usize, hidden: usize) -> Result<()> {
    dispatch::check_len("rmsnorm x", x, batch * hidden)?;
    dispatch::check_len("rmsnorm weight", weight, hidden)?;
    dispatch::check_len("rmsnorm y", y, batch * hidden)?;
    Ok(())
}

pub fn rmsnorm_f32(
    ctx: &WgpuContext,
    x: &[f32],
    weight: &[f32],
    y: &mut [f32],
    batch: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    check_shapes(x.len(), weight.len(), y.len(), batch, hidden)?;
    if batch == 0 || hidden == 0 {
        return Ok(());
    }
    check_device(ctx)?;

    let params = RmsParams {
        hidden: hidden as u32,
        batch: batch as u32,
        eps,
        words_per_row: hidden as u32,
    };
    let xb = dispatch::storage_from_slice(ctx, "rmsnorm-f32-x", x);
    let wb = dispatch::storage_from_slice(ctx, "rmsnorm-f32-weight", weight);
    let yb = dispatch::storage_zeroed(ctx, "rmsnorm-f32-y", (batch * hidden * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "rmsnorm-f32-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, batch as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_rmsnorm_f32",
        &compose(WGSL),
        "rmsnorm_f32",
        &[(0, &xb), (1, &wb), (2, &yb), (3, &pb)],
        groups,
    )?;

    let out: Vec<f32> = dispatch::read_back(ctx, &yb, batch * hidden)?;
    y.copy_from_slice(&out);
    Ok(())
}

pub fn rmsnorm_bf16(
    ctx: &WgpuContext,
    x: &[u16],
    weight: &[u16],
    y: &mut [u16],
    batch: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    check_shapes(x.len(), weight.len(), y.len(), batch, hidden)?;
    if batch == 0 || hidden == 0 {
        return Ok(());
    }
    if !hidden.is_multiple_of(2) {
        return Err(WgpuError::Shape(format!(
            "rmsnorm bf16 hidden must be even so whole u32 words are written; got {hidden}"
        )));
    }
    check_device(ctx)?;

    let words_per_row = hidden / 2;
    let params = RmsParams {
        hidden: hidden as u32,
        batch: batch as u32,
        eps,
        words_per_row: words_per_row as u32,
    };
    let xb = dispatch::storage_from_slice(ctx, "rmsnorm-bf16-x", &pack_u16(x));
    let wb = dispatch::storage_from_slice(ctx, "rmsnorm-bf16-weight", &pack_u16(weight));
    let yb = dispatch::storage_zeroed(ctx, "rmsnorm-bf16-y", (batch * words_per_row * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "rmsnorm-bf16-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, batch as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_rmsnorm_bf16",
        &compose(WGSL),
        "rmsnorm_bf16",
        &[(0, &xb), (1, &wb), (2, &yb), (3, &pb)],
        groups,
    )?;

    let out: Vec<u32> = dispatch::read_back(ctx, &yb, batch * words_per_row)?;
    unpack_u16(&out, y);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u16_word_packing_round_trips() {
        let src: Vec<u16> = (0u16..64).map(|i| i.wrapping_mul(1031)).collect();
        let words = pack_u16(&src);
        assert_eq!(words.len(), src.len() / 2);
        assert_eq!(words[0], src[0] as u32 | ((src[1] as u32) << 16));
        let mut back = vec![0u16; src.len()];
        unpack_u16(&words, &mut back);
        assert_eq!(back, src);
    }

    #[test]
    fn params_keep_the_sixteen_byte_uniform_layout() {
        assert_eq!(std::mem::size_of::<RmsParams>(), 16);
    }

    #[test]
    fn markstein_division_is_correctly_rounded_for_integer_divisors() {
        let mut state = 0x1234_5678u32;
        for hidden in [
            1usize, 2, 3, 5, 7, 11, 43, 257, 1000, 3000, 4097, 11008, 65535,
        ] {
            let b = hidden as f32;
            let r0 = 1.0f32 / b;
            let y = (-b).mul_add(r0, 1.0).mul_add(r0, r0);
            assert_eq!(y.to_bits(), r0.to_bits(), "hidden={hidden}");
            for _ in 0..2000 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let exp = 40 + (state >> 24) % 170;
                let a = f32::from_bits((exp << 23) | (state & 0x007f_ffff));
                let q = a * y;
                let r = (-b).mul_add(q, a);
                assert_eq!(
                    r.mul_add(y, q).to_bits(),
                    (a / b).to_bits(),
                    "hidden={hidden} a={a:e}"
                );
            }
        }
    }

    #[test]
    fn shape_mismatch_is_reported() {
        let e = check_shapes(10, 4, 8, 2, 4).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
    }
}
