#![cfg(feature = "wgpu")]

mod common;
use common::bits;
use common::ctx_or_skip;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::{
    fused_norm_chain, residual_scale, rmsnorm, rmsnorm_residual,
};

fn sample(n: usize, seed: u32) -> Vec<u16> {
    let s = seed as f32;
    bits(
        &(0..n)
            .map(|i| ((i as f32) * (0.0007 + s * 0.0003) + s).sin() * (1.7 + s * 0.4))
            .collect::<Vec<f32>>(),
    )
}

fn sample_w(n: usize, seed: u32) -> Vec<u16> {
    let s = seed as f32;
    bits(
        &(0..n)
            .map(|i| 1.0 + ((i as f32) * 0.0011 + s).cos() * 0.6)
            .collect::<Vec<f32>>(),
    )
}

fn diff_count(a: &[u16], b: &[u16]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

const SHAPES: &[(usize, usize)] = &[(1usize, 2048usize), (3, 2048), (1, 512), (2, 1024)];
const EPS: f32 = 1e-6;

#[test]
fn fused_chain_a_matches_composed_sequence_bitwise() {
    let Some(ctx) = ctx_or_skip("fused_chain_a_matches_composed_sequence_bitwise") else {
        return;
    };
    for (seed, &(batch, hidden)) in SHAPES.iter().enumerate() {
        let seed = seed as u32;
        let x = sample(batch * hidden, seed);
        let r0 = sample(batch * hidden, seed + 10);
        let w1 = sample_w(hidden, seed + 20);
        let w2 = sample_w(hidden, seed + 30);

        let mut t = vec![0u16; batch * hidden];
        rmsnorm::rmsnorm_bf16(ctx, &x, &w1, &mut t, batch, hidden, EPS).expect("ref rms");
        let mut want_res = r0.clone();
        let mut want_out = vec![0u16; batch * hidden];
        rmsnorm_residual::rmsnorm_residual_bf16(
            ctx,
            &t,
            &mut want_res,
            &w2,
            &mut want_out,
            batch,
            hidden,
            EPS,
        )
        .expect("ref rmsres");

        let mut got_res = r0.clone();
        let mut got_out = vec![0u16; batch * hidden];
        fused_norm_chain::rms_res_rms_bf16(
            ctx,
            &x,
            &mut got_res,
            &w1,
            &w2,
            &mut got_out,
            batch,
            hidden,
            EPS,
        )
        .expect("fused chain a");

        let dr = diff_count(&got_res, &want_res);
        let dy = diff_count(&got_out, &want_out);
        eprintln!("chain a batch={batch} hidden={hidden} res_diff={dr} out_diff={dy}");
        assert_eq!(
            dr, 0,
            "chain a residual mismatch batch={batch} hidden={hidden}"
        );
        assert_eq!(dy, 0, "chain a out mismatch batch={batch} hidden={hidden}");
    }
}

#[test]
fn fused_chain_b_matches_composed_sequence_bitwise() {
    let Some(ctx) = ctx_or_skip("fused_chain_b_matches_composed_sequence_bitwise") else {
        return;
    };
    for (seed, &(batch, hidden)) in SHAPES.iter().enumerate() {
        let seed = seed as u32;
        let x = sample(batch * hidden, seed + 40);
        let res = sample(batch * hidden, seed + 50);
        let w = sample_w(hidden, seed + 60);

        let mut t = vec![0u16; batch * hidden];
        rmsnorm::rmsnorm_bf16(ctx, &x, &w, &mut t, batch, hidden, EPS).expect("ref rms");
        let mut want_out = vec![0u16; batch * hidden];
        residual_scale::residual_add_scale_bf16(ctx, &res, &t, &mut want_out, 1.0, batch * hidden)
            .expect("ref resadd");

        let mut got_out = vec![0u16; batch * hidden];
        fused_norm_chain::res_of_rms_bf16(ctx, &x, &res, &w, &mut got_out, batch, hidden, EPS, 1.0)
            .expect("fused chain b");

        let dy = diff_count(&got_out, &want_out);
        eprintln!("chain b batch={batch} hidden={hidden} out_diff={dy}");
        assert_eq!(dy, 0, "chain b out mismatch batch={batch} hidden={hidden}");
    }
}

#[test]
fn fused_chain_c_matches_composed_sequence_bitwise() {
    let Some(ctx) = ctx_or_skip("fused_chain_c_matches_composed_sequence_bitwise") else {
        return;
    };
    for (seed, &(batch, hidden)) in SHAPES.iter().enumerate() {
        let seed = seed as u32;
        for scale in [1.0f32, 0.5, std::f32::consts::SQRT_2] {
            let x = sample(batch * hidden, seed + 70);
            let res = sample(batch * hidden, seed + 80);
            let w1 = sample_w(hidden, seed + 90);
            let w2 = sample_w(hidden, seed + 100);

            let mut t = vec![0u16; batch * hidden];
            rmsnorm::rmsnorm_bf16(ctx, &x, &w1, &mut t, batch, hidden, EPS).expect("ref rms1");
            let mut want_out = vec![0u16; batch * hidden];
            residual_scale::residual_add_scale_bf16(
                ctx,
                &res,
                &t,
                &mut want_out,
                scale,
                batch * hidden,
            )
            .expect("ref resadd");
            let mut want_out2 = vec![0u16; batch * hidden];
            rmsnorm::rmsnorm_bf16(ctx, &want_out, &w2, &mut want_out2, batch, hidden, EPS)
                .expect("ref rms2");

            let mut got_out = vec![0u16; batch * hidden];
            let mut got_out2 = vec![0u16; batch * hidden];
            fused_norm_chain::rms_res_rms_next_bf16(
                ctx,
                &x,
                &res,
                &w1,
                &w2,
                &mut got_out,
                &mut got_out2,
                batch,
                hidden,
                EPS,
                scale,
            )
            .expect("fused chain c");

            let dy = diff_count(&got_out, &want_out);
            let dy2 = diff_count(&got_out2, &want_out2);
            eprintln!(
                "chain c batch={batch} hidden={hidden} scale={scale} out_diff={dy} out2_diff={dy2}"
            );
            assert_eq!(dy, 0, "chain c out mismatch batch={batch} hidden={hidden}");
            assert_eq!(
                dy2, 0,
                "chain c out2 mismatch batch={batch} hidden={hidden}"
            );
        }
    }
}
