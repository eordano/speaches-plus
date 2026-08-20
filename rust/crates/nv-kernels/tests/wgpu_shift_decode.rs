#![cfg(feature = "wgpu")]

mod common;
use common::wgpu_allow_skip;
use nv_kernels::shift_decode_fold::{
    e2m1_code_shift_decodes_to_f32_subnormal_which_ftz_multipliers_flush,
    e2m1_shift_decode_unscaled, e4m3_code_shift_decodes_to_f32_subnormal_which_ftz_multipliers_flush,
    e4m3_shift_decode_unscaled, E2M1_SHIFT_DECODE_LANDS_2POW126_BELOW_TRUE,
    E4M3_SHIFT_DECODE_LANDS_2POW120_BELOW_TRUE,
};
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::{compose, dispatch};
use common::reference_e2m1;
use common::reference_e4m3;

const FOLD_2POW120_WGSL_BITS: &str = "bitcast<f32>(0x7b800000u)";
const FOLD_2POW126_WGSL_BITS: &str = "bitcast<f32>(0x7e800000u)";

fn ctx_or_skip(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("{test}: {}", ctx.summary());
            let st = ctx.qualify();
            if !st.qualified {
                if !wgpu_allow_skip() {
                    panic!("adapter not qualified: {:?}", st.reason);
                }
                eprintln!("{test}: SKIP adapter not qualified: {:?}", st.reason);
                return None;
            }
            Some(ctx)
        }
        Err(e) => {
            if !wgpu_allow_skip() {
                panic!(
                    "{test}: no wgpu adapter: {e}. This gate refuses to report success without \
                     running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
                );
            }
            eprintln!("{test}: SKIP no wgpu adapter: {e}");
            None
        }
    }
}

fn run_map(ctx: &WgpuContext, label: &str, expr: &str, src: &[u32]) -> Vec<f32> {
    let source = compose(&format!(
        "\
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&dst)) {{ return; }}
    dst[i] = {expr};
}}
"
    ));
    let src_buf = dispatch::storage_from_slice(ctx, "shift-src", src);
    let dst_buf = dispatch::storage_zeroed(ctx, "shift-dst", (src.len() * 4) as u64);
    let groups = dispatch::workgroup_count_1d(ctx, src.len() as u64, 64);
    dispatch::run(
        ctx,
        label,
        &source,
        "main",
        &[(0, &src_buf), (1, &dst_buf)],
        groups,
    )
    .expect("dispatch");
    dispatch::read_back(ctx, &dst_buf, src.len()).expect("read back")
}

#[test]
fn wgpu_e4m3_shift_decode_bits_match_the_host_emulation_for_all_256_codes() {
    let Some(ctx) = ctx_or_skip("wgpu_e4m3_shift_decode_bits_match_the_host_emulation") else {
        return;
    };
    let codes: Vec<u32> = (0u32..256).collect();
    let got = run_map(
        ctx,
        "e4m3-shift-unscaled",
        "e4m3_shift_decode_scale_must_carry_2pow120(src[i])",
        &codes,
    );
    for b in 0u32..256 {
        assert_eq!(
            got[b as usize].to_bits(),
            e4m3_shift_decode_unscaled(b as u8).to_bits(),
            "e4m3 code {b:#04x}: GPU shift decode disagrees with the host bit emulation, so \
             either the shader or a driver denormal flush changed the unscaled bits"
        );
    }
}

fn subnormal_band_verdict(test: &str, exact: usize, flushed: usize, total: usize) {
    assert_eq!(
        exact + flushed,
        total,
        "{test}: a subnormal-band code came back neither bit-exact nor flushed to zero, which is \
         a third adapter behavior this suite has never seen"
    );
    assert!(
        exact == total || flushed == total,
        "{test}: the adapter treated subnormal-band multiply operands inconsistently ({exact} \
         exact, {flushed} flushed of {total})"
    );
    eprintln!(
        "{test}: this adapter {} f32 denormal multiply operands ({exact} exact, {flushed} \
         flushed of {total} subnormal-band codes); a flushing adapter zeroes these codes on any \
         shift-decode route that multiplies before renormalizing, so callers must keep such \
         codes off the f32 fold path there",
        if flushed == 0 { "preserves" } else { "FLUSHES" }
    );
}

#[test]
fn wgpu_e4m3_folded_product_is_exact_off_the_subnormal_band_and_pins_the_flush_behavior() {
    let Some(ctx) = ctx_or_skip("wgpu_e4m3_folded_product") else {
        return;
    };
    let codes: Vec<u32> = (0u32..256).collect();
    let got = run_map(
        ctx,
        "e4m3-shift-folded",
        &format!("e4m3_shift_decode_scale_must_carry_2pow120(src[i]) * {FOLD_2POW120_WGSL_BITS}"),
        &codes,
    );
    assert_eq!(
        f32::from_bits(0x7b80_0000).to_bits(),
        E4M3_SHIFT_DECODE_LANDS_2POW120_BELOW_TRUE.to_bits()
    );
    let mut exact = 0usize;
    let mut flushed = 0usize;
    let mut band = 0usize;
    for b in 0u32..256 {
        let byte = b as u8;
        let g = got[b as usize];
        if byte & 0x7f == 0x7f {
            assert!(
                g.is_finite() && g.abs() > 448.0,
                "e4m3 NaN code {byte:#04x} must shift-decode to a finite value beyond \
                 E4M3_MAX=448, got {g}; callers exclude these codes by contract"
            );
            continue;
        }
        let want = reference_e4m3(byte);
        if e4m3_code_shift_decodes_to_f32_subnormal_which_ftz_multipliers_flush(byte) {
            band += 1;
            if g.to_bits() == want.to_bits() {
                exact += 1;
            } else if g.abs() == 0.0 {
                flushed += 1;
            }
            continue;
        }
        assert_eq!(
            g.to_bits(),
            want.to_bits(),
            "e4m3 code {byte:#04x}: GPU folded product {g:e} vs format reference {want:e}; off \
             the subnormal band the fold multiply must be bit-exact on every adapter"
        );
    }
    subnormal_band_verdict("wgpu_e4m3_folded_product", exact, flushed, band);
}

#[test]
fn wgpu_e2m1_shift_decode_bits_match_the_host_emulation_for_all_16_codes() {
    let Some(ctx) = ctx_or_skip("wgpu_e2m1_shift_decode_bits_match_the_host_emulation") else {
        return;
    };
    let codes: Vec<u32> = (0u32..16).collect();
    let got = run_map(
        ctx,
        "e2m1-shift-unscaled",
        "e2m1_shift_decode_scale_must_carry_2pow126(src[i])",
        &codes,
    );
    for c in 0u32..16 {
        assert_eq!(
            got[c as usize].to_bits(),
            e2m1_shift_decode_unscaled(c as u8).to_bits(),
            "e2m1 code {c:#03x}: GPU shift decode disagrees with the host bit emulation"
        );
    }
}

#[test]
fn wgpu_e2m1_folded_product_is_exact_off_the_subnormal_band_and_pins_the_flush_behavior() {
    let Some(ctx) = ctx_or_skip("wgpu_e2m1_folded_product") else {
        return;
    };
    let codes: Vec<u32> = (0u32..16).collect();
    let got = run_map(
        ctx,
        "e2m1-shift-folded",
        &format!("e2m1_shift_decode_scale_must_carry_2pow126(src[i]) * {FOLD_2POW126_WGSL_BITS}"),
        &codes,
    );
    assert_eq!(
        f32::from_bits(0x7e80_0000).to_bits(),
        E2M1_SHIFT_DECODE_LANDS_2POW126_BELOW_TRUE.to_bits()
    );
    let mut exact = 0usize;
    let mut flushed = 0usize;
    let mut band = 0usize;
    for c in 0u32..16 {
        let code = c as u8;
        let g = got[c as usize];
        let want = reference_e2m1(code);
        if e2m1_code_shift_decodes_to_f32_subnormal_which_ftz_multipliers_flush(code) {
            band += 1;
            if g.to_bits() == want.to_bits() {
                exact += 1;
            } else if g.abs() == 0.0 {
                flushed += 1;
            }
            continue;
        }
        assert_eq!(
            g.to_bits(),
            want.to_bits(),
            "e2m1 code {code:#03x}: GPU folded product {g:e} vs format reference {want:e}; off \
             the subnormal band the fold multiply must be bit-exact on every adapter"
        );
    }
    assert_eq!(
        band, 2,
        "the e2m1 subnormal band is exactly the two half codes 0x1 and 0x9"
    );
    subnormal_band_verdict("wgpu_e2m1_folded_product", exact, flushed, band);
}
