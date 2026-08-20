#![cfg(feature = "wgpu")]

use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_quant::nvfp4::{swizzle_scales, Nvfp4Tensor, BLOCK_SIZE};

fn scales_len(m: usize, k: usize) -> usize {
    m.div_ceil(128) * 128 * (k / BLOCK_SIZE).div_ceil(4) * 4
}

fn check(m: usize, k: usize, global: f32, seed_off: f32) {
    let ctx = WgpuContext::shared().expect("no wgpu adapter");
    let host_bf: Vec<bf16> = (0..m * k)
        .map(|n| bf16::from_f32(((n as f32 * 0.013) + seed_off).sin() * 2.5))
        .collect();

    let rows: Vec<Vec<f32>> = (0..m)
        .map(|i| host_bf[i * k..(i + 1) * k].iter().map(|v| v.to_f32()).collect())
        .collect();
    let q_cpu = Nvfp4Tensor::quantize_rows_with_global(&rows, global);
    let scales_cpu = swizzle_scales(&q_cpu.scales, m, k / BLOCK_SIZE);

    let x: Vec<u16> = host_bf.iter().map(|v| v.to_bits()).collect();
    let mut packed = vec![0u8; m * k / 2];
    let mut scales = vec![0u8; scales_len(m, k)];
    nv_kernels::wgpu_backend::kernels::quantize_nvfp4_bf16::quantize_nvfp4_bf16(
        ctx, &x, global, &mut packed, &mut scales, m, m, k,
    )
    .expect("wgpu quantize_nvfp4_bf16");

    assert!(
        q_cpu.data.iter().any(|b| *b != 0),
        "m={m} k={k} g={global}: the oracle produced all zeros, so a match would be vacuous"
    );
    assert_eq!(q_cpu.data.len(), packed.len(), "packed length disagrees");
    assert_eq!(scales_cpu.len(), scales.len(), "scale length disagrees");

    let nib: usize = q_cpu
        .data
        .iter()
        .zip(packed.iter())
        .map(|(a, b)| usize::from((a & 0x0F) != (b & 0x0F)) + usize::from((a >> 4) != (b >> 4)))
        .sum();
    let sc = scales_cpu.iter().zip(scales.iter()).filter(|(a, b)| a != b).count();
    eprintln!(
        "[wgsl-nvfp4] m={m} k={k} global={global}: {nib}/{} nibbles, {sc}/{} scale bytes differ from the CPU oracle",
        packed.len() * 2,
        scales.len()
    );
    assert_eq!(
        nib, 0,
        "m={m} k={k} global={global}: WGSL disagrees with the CPU oracle on {nib} e2m1 nibbles. \
         No accumulation happens here, so this is an encoding disagreement -- check the ue4m3 \
         encode in wgsl/quantize_nvfp4_bf16.wgsl (see #43: carry into the next binade, do not \
         saturate at 7)."
    );
    assert_eq!(sc, 0, "m={m} k={k} global={global}: {sc} ue4m3 scale bytes differ from the oracle");
}

#[test]
fn wgsl_nvfp4_encode_matches_the_cpu_oracle() {
    check(128, 128, 1.0, 0.0);
    check(128, 128, 0.37, 1.7);
    check(64, 256, 2.5, 0.3);
    check(256, 64, 1.0, 4.2);
}

#[test]
fn wgsl_nvfp4_encode_holds_across_global_scales() {
    for (i, g) in [0.999f32, 1.0, 1.001, 1.999, 2.0].into_iter().enumerate() {
        check(128, 128, g, i as f32 * 0.9);
    }
}
