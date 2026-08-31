#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;

fn cpu_reference(
    y_sorted: &[bf16],
    topk_weights: &[f32],
    inv_perm: &[i32],
    n_tokens: usize,
    k: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut acc = vec![0f32; n_tokens * hidden];
    for n in 0..n_tokens {
        for s in 0..k {
            let slot = n * k + s;
            let sorted_row = inv_perm[slot] as usize;
            let w = topk_weights[slot];
            for h in 0..hidden {
                acc[n * hidden + h] += w * y_sorted[sorted_row * hidden + h].to_f32();
            }
        }
    }
    acc
}

#[test]
fn moe_unpermute_scatter_matches_cpu() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "moe_unpermute_scatter: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("moe_unpermute_scatter: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    let n_tokens = 8usize;
    let k = 4usize;
    let hidden = 512usize;
    let m_total = n_tokens * k;

    let mut inv_perm: Vec<i32> = (0..(n_tokens * k) as i32).collect();
    inv_perm.swap(0, 7);
    inv_perm.swap(3, 12);
    inv_perm.swap(5, 22);

    let y_sorted: Vec<bf16> = (0..m_total * hidden)
        .map(|i| bf16::from_f32(((i as f32) * 0.013).sin() * 1.5))
        .collect();
    let topk_weights: Vec<f32> = (0..n_tokens * k)
        .map(|i| 0.1 + (i as f32 * 0.071).cos().abs() * 0.4)
        .collect();

    let cpu = cpu_reference(&y_sorted, &topk_weights, &inv_perm, n_tokens, k, hidden);

    #[allow(deprecated)]
    let y_dev = stream
        .memcpy_stod(unsafe {
            std::slice::from_raw_parts(y_sorted.as_ptr() as *const u16, y_sorted.len())
        })
        .unwrap();
    #[allow(deprecated)]
    let w_dev = stream.memcpy_stod(&topk_weights).unwrap();
    #[allow(deprecated)]
    let inv_dev = stream.memcpy_stod(&inv_perm).unwrap();
    let mut acc_dev = stream.alloc_zeros::<f32>(n_tokens * hidden).unwrap();

    let rc = {
        let (y_p, _g1) = y_dev.device_ptr(&stream);
        let (w_p, _g2) = w_dev.device_ptr(&stream);
        let (i_p, _g3) = inv_dev.device_ptr(&stream);
        let (a_p, _g4) = acc_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::moe_unpermute_scatter(
                stream.cu_stream() as *mut c_void,
                y_p as *const u16,
                w_p as *const f32,
                i_p as *const i32,
                a_p as *mut f32,
                n_tokens as i32,
                k as i32,
                hidden as i32,
                hidden as i32,
            )
        }
    };
    assert_eq!(rc, 0, "moe_unpermute_scatter rc={rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let got = stream.memcpy_dtov(&acc_dev).unwrap();

    let mut max_abs_diff = 0f32;
    for (g, c) in got.iter().zip(cpu.iter()) {
        let d = (g - c).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    eprintln!("max abs diff GPU vs CPU = {max_abs_diff:.6}");
    assert!(max_abs_diff < 1e-4, "GPU vs CPU diverges by {max_abs_diff}");
}

#[test]
fn moe_unpermute_scatter_verify_shaped_tile_strided_rows_match_cpu_and_skip_padding() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "moe_unpermute_scatter (tile-strided): no CUDA device 0. This gate refuses to \
                 report success without running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on \
                 purpose."
            );
        }
        eprintln!(
            "moe_unpermute_scatter (tile-strided): SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0"
        );
        return;
    };
    let stream = ctx.default_stream();

    let min_tile = 128usize;
    let hidden = 2048usize;
    for &(n_tokens, tiles_per_token) in &[(2usize, 9usize), (3, 9), (5, 9), (3, 8)] {
        let tiles = n_tokens * tiles_per_token;
        let rows = tiles * min_tile;
        let inv_perm: Vec<i32> = (0..tiles as i32).map(|t| t * min_tile as i32).collect();
        let mut y_sorted: Vec<bf16> = vec![bf16::from_f32(f32::NAN); rows * hidden];
        for t in 0..tiles {
            for h in 0..hidden {
                y_sorted[t * min_tile * hidden + h] =
                    bf16::from_f32(((t * hidden + h) as f32 * 0.011).sin() * 1.2);
            }
        }
        let topk_weights: Vec<f32> = (0..tiles)
            .map(|i| 0.1 + (i as f32 * 0.083).cos().abs() * 0.4)
            .collect();
        let cpu = cpu_reference(
            &y_sorted,
            &topk_weights,
            &inv_perm,
            n_tokens,
            tiles_per_token,
            hidden,
        );

        #[allow(deprecated)]
        let y_dev = stream
            .memcpy_stod(unsafe {
                std::slice::from_raw_parts(y_sorted.as_ptr() as *const u16, y_sorted.len())
            })
            .unwrap();
        #[allow(deprecated)]
        let w_dev = stream.memcpy_stod(&topk_weights).unwrap();
        #[allow(deprecated)]
        let inv_dev = stream.memcpy_stod(&inv_perm).unwrap();
        let mut acc_dev = stream.alloc_zeros::<f32>(n_tokens * hidden).unwrap();
        let rc = {
            let (y_p, _g1) = y_dev.device_ptr(&stream);
            let (w_p, _g2) = w_dev.device_ptr(&stream);
            let (i_p, _g3) = inv_dev.device_ptr(&stream);
            let (a_p, _g4) = acc_dev.device_ptr_mut(&stream);
            unsafe {
                nv_kernels::cuda::moe_unpermute_scatter(
                    stream.cu_stream() as *mut c_void,
                    y_p as *const u16,
                    w_p as *const f32,
                    i_p as *const i32,
                    a_p as *mut f32,
                    n_tokens as i32,
                    tiles_per_token as i32,
                    hidden as i32,
                    hidden as i32,
                )
            }
        };
        assert_eq!(rc, 0, "moe_unpermute_scatter rc={rc}");
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        let got = stream.memcpy_dtov(&acc_dev).unwrap();
        let mut max_abs_diff = 0f32;
        let mut nonfinite = 0usize;
        for (g, c) in got.iter().zip(cpu.iter()) {
            if !g.is_finite() {
                nonfinite += 1;
                continue;
            }
            max_abs_diff = max_abs_diff.max((g - c).abs());
        }
        assert_eq!(
            nonfinite, 0,
            "m={n_tokens} tpt={tiles_per_token}: NaN in output -- the kernel read a padding row"
        );
        assert!(
            max_abs_diff < 1e-4,
            "m={n_tokens} tpt={tiles_per_token}: GPU vs CPU diverges by {max_abs_diff}"
        );
        eprintln!(
            "  tile-strided m={n_tokens} tpt={tiles_per_token}: max abs diff {max_abs_diff:.6}, no padding reads"
        );
    }
}

fn cpu_reference_shared_sig_tail(
    y_sorted: &[bf16],
    topk_weights: &[f32],
    inv_perm: &[i32],
    gate_logits: &[bf16],
    n_tokens: usize,
    k: usize,
    hidden: usize,
) -> Vec<bf16> {
    let mut out = vec![bf16::from_f32(0.0); n_tokens * hidden];
    for n in 0..n_tokens {
        let sig = 1.0f32 / (1.0 + (-gate_logits[n].to_f32()).exp());
        for h in 0..hidden {
            let mut acc = 0f32;
            for s in 0..k {
                let slot = n * k + s;
                let sorted_row = inv_perm[slot] as usize;
                let w = if s == k - 1 { sig } else { topk_weights[slot] };
                acc += w * y_sorted[sorted_row * hidden + h].to_f32();
            }
            out[n * hidden + h] = bf16::from_f32(acc);
        }
    }
    out
}

#[test]
fn moe_unpermute_scatter_shared_sig_tail_matches_cpu() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "moe_unpermute_scatter_shared_sig_tail: no CUDA device 0. This gate refuses to \
                 report success without running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on \
                 purpose."
            );
        }
        eprintln!(
            "moe_unpermute_scatter_shared_sig_tail: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0"
        );
        return;
    };
    let stream = ctx.default_stream();

    let n_tokens = 4usize;
    let k = 9usize;
    let hidden = 2048usize;
    let m_total = n_tokens * k;

    let mut inv_perm: Vec<i32> = (0..(n_tokens * k) as i32).collect();
    inv_perm.swap(0, 7);
    inv_perm.swap(3, 12);
    inv_perm.swap(5, 22);

    let y_sorted: Vec<bf16> = (0..m_total * hidden)
        .map(|i| bf16::from_f32(((i as f32) * 0.013).sin() * 1.5))
        .collect();
    let topk_weights: Vec<f32> = (0..n_tokens * k)
        .map(|i| 0.1 + (i as f32 * 0.071).cos().abs() * 0.4)
        .collect();
    let gate_logits: Vec<bf16> = (0..n_tokens)
        .map(|i| bf16::from_f32((i as f32) * 0.9 - 1.7))
        .collect();

    let cpu = cpu_reference_shared_sig_tail(
        &y_sorted,
        &topk_weights,
        &inv_perm,
        &gate_logits,
        n_tokens,
        k,
        hidden,
    );

    #[allow(deprecated)]
    let y_dev = stream
        .memcpy_stod(unsafe {
            std::slice::from_raw_parts(y_sorted.as_ptr() as *const u16, y_sorted.len())
        })
        .unwrap();
    #[allow(deprecated)]
    let w_dev = stream.memcpy_stod(&topk_weights).unwrap();
    #[allow(deprecated)]
    let inv_dev = stream.memcpy_stod(&inv_perm).unwrap();
    #[allow(deprecated)]
    let g_dev = stream
        .memcpy_stod(unsafe {
            std::slice::from_raw_parts(gate_logits.as_ptr() as *const u16, gate_logits.len())
        })
        .unwrap();
    let mut out_dev = stream.alloc_zeros::<u16>(n_tokens * hidden).unwrap();

    let rc = {
        let (y_p, _g1) = y_dev.device_ptr(&stream);
        let (w_p, _g2) = w_dev.device_ptr(&stream);
        let (i_p, _g3) = inv_dev.device_ptr(&stream);
        let (g_p, _g4) = g_dev.device_ptr(&stream);
        let (o_p, _g5) = out_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::moe_unpermute_scatter_shared_sig_tail(
                stream.cu_stream() as *mut c_void,
                y_p as *const u16,
                w_p as *const f32,
                i_p as *const i32,
                g_p as *const u16,
                o_p as *mut u16,
                n_tokens as i32,
                k as i32,
                hidden as i32,
                hidden as i32,
            )
        }
    };
    assert_eq!(rc, 0, "moe_unpermute_scatter_shared_sig_tail rc={rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let got_raw = stream.memcpy_dtov(&out_dev).unwrap();

    let mut max_abs_diff = 0f32;
    let mut exact = true;
    for (g_bits, c) in got_raw.iter().zip(cpu.iter()) {
        let g = bf16::from_bits(*g_bits).to_f32();
        let d = (g - c.to_f32()).abs();
        if *g_bits != c.to_bits() {
            exact = false;
        }
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    eprintln!(
        "shared sig tail: max abs diff GPU vs CPU = {max_abs_diff:.6} bitwise_exact={exact}"
    );
    assert!(
        max_abs_diff < 2e-2,
        "GPU vs CPU diverges by {max_abs_diff}; the bf16 tail rounds once so only bf16-level \
         differences are acceptable"
    );
}
