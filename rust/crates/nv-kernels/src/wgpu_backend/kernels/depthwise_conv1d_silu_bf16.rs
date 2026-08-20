#![allow(clippy::too_many_arguments)]

use std::sync::OnceLock;

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dispatch;
use crate::wgpu_backend::{compose, Result, WgpuError};
pub use crate::wgpu_backend::pack::{pack_u16_odd_tail_zeroed_min_one_word as pack_bf16_words, unpack_u16_first_n as unpack_bf16_words};

pub const WGSL: &str = include_str!("../../../wgsl/depthwise_conv1d_silu_bf16.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;

pub const ENTRY: &str = "depthwise_conv1d_silu_bf16";

static SOURCE: OnceLock<String> = OnceLock::new();

fn source() -> &'static str {
    SOURCE.get_or_init(|| compose(WGSL))
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct DwcParams {
    batch: u32,
    channels: u32,
    seq_len: u32,
    ksize: u32,
    n_elems: u32,
    n_words: u32,
    pad0: u32,
    pad1: u32,
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup(ctx, "depthwise_conv1d_silu_bf16", WORKGROUP_SIZE)
}

fn params_for(b: usize, c: usize, t: usize, k: usize) -> Result<DwcParams> {
    let n_elems = b
        .checked_mul(c)
        .and_then(|v| v.checked_mul(t))
        .ok_or_else(|| {
            WgpuError::Shape(format!("depthwise_conv1d: B*C*T overflows ({b},{c},{t})"))
        })?;
    let n_words = n_elems.div_ceil(2);
    let w_elems = c
        .checked_mul(k)
        .ok_or_else(|| WgpuError::Shape(format!("depthwise_conv1d: C*K overflows ({c},{k})")))?;
    for (what, v) in [
        ("B", b),
        ("C", c),
        ("T", t),
        ("K", k),
        ("B*C*T", n_elems),
        ("C*K", w_elems),
    ] {
        if v > u32::MAX as usize {
            return Err(WgpuError::Shape(format!(
                "depthwise_conv1d: {what} = {v} exceeds the u32 index range"
            )));
        }
    }
    Ok(DwcParams {
        batch: b as u32,
        channels: c as u32,
        seq_len: t as u32,
        ksize: k as u32,
        n_elems: n_elems as u32,
        n_words: n_words as u32,
        pad0: 0,
        pad1: 0,
    })
}

pub fn depthwise_conv1d_silu_bf16(
    ctx: &WgpuContext,
    x_bf16: &[u16],
    w_bf16: &[u16],
    y_bf16: &mut [u16],
    b: usize,
    c: usize,
    t: usize,
    k: usize,
) -> Result<()> {
    if b == 0 || c == 0 || t == 0 || k == 0 {
        return Ok(());
    }
    let params = params_for(b, c, t, k)?;
    let n = params.n_elems as usize;
    let words = params.n_words as usize;
    dispatch::check_len("depthwise_conv1d x", x_bf16.len(), n)?;
    dispatch::check_len("depthwise_conv1d w", w_bf16.len(), c * k)?;
    dispatch::check_len("depthwise_conv1d y", y_bf16.len(), n)?;
    check_device(ctx)?;

    let xb = dispatch::storage_from_slice(ctx, "dwc-x", &pack_bf16_words(x_bf16));
    let wb = dispatch::storage_from_slice(ctx, "dwc-w", &pack_bf16_words(w_bf16));
    let yb = dispatch::storage_zeroed(ctx, "dwc-y", (words * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "dwc-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, words as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        "nv_kernels_depthwise_conv1d_silu_bf16",
        source(),
        ENTRY,
        &[(0, &xb), (1, &wb), (2, &yb), (3, &pb)],
        groups,
    )?;

    let out: Vec<u32> = dispatch::read_back(ctx, &yb, words)?;
    unpack_bf16_words(&out, n, y_bf16);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_cover_odd_element_counts() {
        let p = params_for(1, 3, 5, 4).unwrap();
        assert_eq!(p.n_elems, 15);
        assert_eq!(p.n_words, 8);
        assert_eq!(p.ksize, 4);
    }

    #[test]
    fn params_reject_indices_past_u32() {
        if (u32::MAX as usize) < usize::MAX {
            assert!(params_for(1, 1, u32::MAX as usize + 1, 1).is_err());
        }
    }

    #[test]
    fn shader_declares_the_entry_point() {
        assert!(WGSL.contains("fn depthwise_conv1d_silu_bf16("));
        assert!(WGSL.contains("acc = fma(bitcast<f32>(xb << 16u), bitcast<f32>(wb << 16u), acc);"));
    }

    #[test]
    fn shader_keeps_the_denormal_exact_fallback() {
        for needle in [
            "fn dwc_soft_fma(xb: u32, wb: u32, ab: u32) -> u32 {",
            "fn dwc_soft_acc_bits(idx: u32, t: u32, wbase: u32, kmax: u32) -> u32 {",
            "fn dwc_silu_from_bits(ab: u32) -> f32 {",
            "risky = risky || ex == 0u || ew == 0u || (i32(ex) + i32(ew)) < 174;",
            "risky = risky || (seen && (bitcast<u32>(acc) & 0x7fffffffu) < 0x17800000u);",
            "if (risky || nonfinite) {",
            "ab = dwc_soft_acc_bits(idx, t, wbase, kmax);",
        ] {
            assert!(WGSL.contains(needle), "shader lost `{needle}`");
        }
        assert_eq!(0x17800000u32, 47u32 << 23);
        assert_eq!(174i32, 254 - 80);
    }

    #[test]
    fn shader_packs_bf16_bits_without_float_round_trip() {
        for needle in [
            "fn dwc_point_bits(idx: u32) -> u32 {",
            "dwc_y[w] = lo | (hi << 16u);",
            "return 0x7fffu;",
            "return select(0x7f80u, 0x7fffu, (ab >> 31u) == 1u);",
            "return bf16_encode(dwc_silu_from_bits(ab));",
        ] {
            assert!(WGSL.contains(needle), "shader lost `{needle}`");
        }
        assert!(
            !WGSL.contains("bf16_pack("),
            "bf16_pack maps every NaN to 0x7fc0; CUDA's cvt.rn.bf16.f32 emits 0x7fff, so the \
             entry point must assemble the output word from bf16_encode plus explicit \
             inf/nan bits"
        );
    }

    #[test]
    fn soft_fma_window_has_room_for_guard_and_sticky() {
        for needle in [
            "let base = top - 38;",
            "let sub_shift = -149 - base;",
            "if (rbit == 1u && (sticky || (q & 1u) == 1u)) {",
            "fn dwc_u64_shr_jam(v: vec2<u32>, s: u32) -> vec2<u32> {",
        ] {
            assert!(WGSL.contains(needle), "shader lost `{needle}`");
        }
    }

    #[test]
    fn silu_mirrors_the_libdevice_expf_range_reduction() {
        for needle in [
            "const DWC_RR_SCALE: f32 = -0.005724980030208826;",
            "const DWC_RR_STEPS: f32 = 252.0;",
            "const DWC_RR_BIAS: f32 = 12582913.0;",
            "const DWC_RR_UNBIAS: f32 = 12583039.0;",
            "const DWC_LOG2E_HI: f32 = -1.4426950216293335;",
            "const DWC_LOG2E_LO: f32 = -1.925963033500011e-8;",
            "let scale = bitcast<f32>(bitcast<u32>(j) << 23u);",
            "let den = fma(exp2(f), scale, 1.0);",
        ] {
            assert!(WGSL.contains(needle), "shader lost `{needle}`");
        }
        assert!(!WGSL.contains("exp(-acc)"));
    }

    #[test]
    fn silu_emulates_the_subnormal_reciprocal() {
        for needle in [
            "const DWC_SUBNORMAL_DEN: f32 = 8.5070591730234616e37;",
            "const DWC_P2_M149_BITS: u32 = 1249902592u;",
            "const DWC_ULP_BIAS: u32 = 1258291200u;",
            "if (den >= DWC_SUBNORMAL_DEN) {",
            "return dwc_mul_rcp_subnormal(acc, den);",
            "let rem = fma(-d, t, 1.0);",
            "let up = bitcast<f32>(tb + 1u);",
            "let dn = bitcast<f32>(tb - 1u);",
        ] {
            assert!(WGSL.contains(needle), "shader lost `{needle}`");
        }
    }

    #[test]
    fn rounding_correction_never_materialises_a_subnormal_ulp() {
        for needle in [
            "let dulp = bitcast<f32>(bitcast<u32>(d) + (tb & 0x7f800000u) - DWC_ULP_BIAS);",
            "let dulp = bitcast<f32>(bitcast<u32>(d) + ((tb - 1u) & 0x7f800000u) - DWC_ULP_BIAS);",
            "let g = dulp - (rem + rem);",
            "let g = dulp + (rem + rem);",
        ] {
            assert!(WGSL.contains(needle), "shader lost `{needle}`");
        }
        assert!(!WGSL.contains("up - t"));
        assert!(!WGSL.contains("t - dn"));
        assert_eq!(1258291200u32, 150u32 << 23);
    }

    #[test]
    fn subnormal_reciprocal_uses_integer_long_division() {
        for needle in [
            "for (var i = 0u; i < 48u; i = i + 1u) {",
            "r = (r << 1u) | select(0u, 1u, i == 0u);",
            "if (num > cmp || (num == cmp && (base & 1u) == 1u)) {",
            "let p = acc * f32(m);",
            "return bitcast<f32>(bitcast<u32>(p) - DWC_P2_M149_BITS);",
        ] {
            assert!(WGSL.contains(needle), "shader lost `{needle}`");
        }
        assert!(
            !WGSL.contains("1.0 / ds"),
            "the subnormal path must not divide by a power-of-two-scaled denominator: \
             drivers reassociate 1.0/(d*2^-64) into (1.0/d)*2^64 and flush the subnormal to zero"
        );
    }

    #[test]
    fn subnormal_scale_constant_is_two_pow_minus_149() {
        assert_eq!(1249902592u32, 149u32 << 23);
    }
}
