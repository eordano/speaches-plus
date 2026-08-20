#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use common::lcg;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::gather_rows_bf16::gather_rows_bf16;

fn cpu_reference(
    x: &[u16],
    src_idx: &[i32],
    m_total_padded: usize,
    hidden: usize,
    n_tokens: usize,
) -> Vec<u16> {
    let mut out = vec![0u16; m_total_padded * hidden];
    for r in 0..m_total_padded {
        let s = src_idx[r];
        if s >= 0 && (s as usize) < n_tokens {
            let sb = s as usize * hidden;
            out[r * hidden..r * hidden + hidden].copy_from_slice(&x[sb..sb + hidden]);
        }
    }
    out
}

fn assert_bit_exact(got: &[u16], want: &[u16], hidden: usize, src_idx: &[i32]) -> f32 {
    let mut max_abs = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (bf16::from_bits(*g).to_f32() - bf16::from_bits(*w).to_f32()).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert_eq!(
            g,
            w,
            "row {} col {} (src_idx={}): got bits {:#06x} want {:#06x}",
            i / hidden,
            i % hidden,
            src_idx[i / hidden],
            g,
            w
        );
    }
    max_abs
}

#[test]
fn wgpu_gather_rows_bf16_basic() {
    let Some(ctx) = ctx_or_skip("wgpu_gather_rows_bf16_basic") else {
        return;
    };
    let n_tokens = 5usize;
    let hidden = 256usize;
    let m_total_padded = 16usize;

    let mut x: Vec<u16> = Vec::with_capacity(n_tokens * hidden);
    for n in 0..n_tokens {
        for h in 0..hidden {
            x.push(bf16::from_f32((n * 100 + h) as f32 * 0.001).to_bits());
        }
    }
    let src_idx: Vec<i32> = vec![0, 1, 2, 3, 4, -1, -1, -1, 2, 0, 4, 1, 3, -1, -1, -1];
    assert_eq!(src_idx.len(), m_total_padded);

    let mut out = vec![0xdeadu16; m_total_padded * hidden];
    gather_rows_bf16(
        ctx,
        &x,
        &src_idx,
        &mut out,
        m_total_padded,
        hidden,
        n_tokens,
    )
    .expect("gather_rows_bf16");

    let want = cpu_reference(&x, &src_idx, m_total_padded, hidden, n_tokens);
    let max_abs = assert_bit_exact(&out, &want, hidden, &src_idx);
    eprintln!("wgpu_gather_rows_bf16_basic: max_abs_err={max_abs}");
    assert_eq!(max_abs, 0.0);
}

#[test]
fn wgpu_gather_rows_bf16_pad_only() {
    let Some(ctx) = ctx_or_skip("wgpu_gather_rows_bf16_pad_only") else {
        return;
    };
    let n_tokens = 1usize;
    let hidden = 128usize;
    let m_total_padded = 4usize;

    let x: Vec<u16> = vec![bf16::from_f32(123.0).to_bits(); n_tokens * hidden];
    let src_idx: Vec<i32> = vec![-1; m_total_padded];
    let mut out = vec![0x1234u16; m_total_padded * hidden];
    gather_rows_bf16(
        ctx,
        &x,
        &src_idx,
        &mut out,
        m_total_padded,
        hidden,
        n_tokens,
    )
    .expect("gather_rows_bf16");
    for (i, v) in out.iter().enumerate() {
        assert_eq!(*v, 0, "padding row should be zero, idx {i}");
    }
}

#[test]
fn wgpu_gather_rows_bf16_random_and_oob() {
    let Some(ctx) = ctx_or_skip("wgpu_gather_rows_bf16_random_and_oob") else {
        return;
    };
    let mut st = 0x9e3779b97f4a7c15u64;
    for &(n_tokens, hidden, m_total_padded) in &[
        (1usize, 2usize, 1usize),
        (3, 6, 7),
        (17, 130, 40),
        (64, 1024, 129),
        (7, 2048, 33),
    ] {
        let x: Vec<u16> = (0..n_tokens * hidden)
            .map(|_| (lcg(&mut st) >> 32) as u16)
            .collect();
        let src_idx: Vec<i32> = (0..m_total_padded)
            .map(|_| {
                let r = (lcg(&mut st) >> 33) as i64 % (n_tokens as i64 + 4);
                (r - 2) as i32
            })
            .collect();
        let mut out = vec![0xffffu16; m_total_padded * hidden];
        gather_rows_bf16(
            ctx,
            &x,
            &src_idx,
            &mut out,
            m_total_padded,
            hidden,
            n_tokens,
        )
        .expect("gather_rows_bf16");
        let want = cpu_reference(&x, &src_idx, m_total_padded, hidden, n_tokens);
        let max_abs = assert_bit_exact(&out, &want, hidden, &src_idx);
        eprintln!(
            "wgpu_gather_rows_bf16_random_and_oob: n_tokens={n_tokens} hidden={hidden} m={m_total_padded} max_abs_err={max_abs}"
        );
    }
}

#[test]
fn wgpu_gather_rows_bf16_large_row_count() {
    let Some(ctx) = ctx_or_skip("wgpu_gather_rows_bf16_large_row_count") else {
        return;
    };
    let n_tokens = 8usize;
    let hidden = 32usize;
    let m_total_padded = 70000usize;
    let mut st = 0x243f6a8885a308d3u64;
    let x: Vec<u16> = (0..n_tokens * hidden)
        .map(|_| (lcg(&mut st) >> 32) as u16)
        .collect();
    let src_idx: Vec<i32> = (0..m_total_padded)
        .map(|r| {
            if r % 5 == 0 {
                -1
            } else {
                (r % n_tokens) as i32
            }
        })
        .collect();
    let mut out = vec![0u16; m_total_padded * hidden];
    gather_rows_bf16(
        ctx,
        &x,
        &src_idx,
        &mut out,
        m_total_padded,
        hidden,
        n_tokens,
    )
    .expect("gather_rows_bf16");
    let want = cpu_reference(&x, &src_idx, m_total_padded, hidden, n_tokens);
    let max_abs = assert_bit_exact(&out, &want, hidden, &src_idx);
    eprintln!("wgpu_gather_rows_bf16_large_row_count: max_abs_err={max_abs}");
}

#[test]
fn wgpu_gather_rows_bf16_odd_hidden_is_shape_error() {
    let Some(ctx) = ctx_or_skip("wgpu_gather_rows_bf16_odd_hidden_is_shape_error") else {
        return;
    };
    let x = vec![0u16; 3];
    let src_idx = vec![0i32; 2];
    let mut out = vec![0u16; 6];
    let e = gather_rows_bf16(ctx, &x, &src_idx, &mut out, 2, 3, 1).unwrap_err();
    assert!(
        matches!(e, nv_kernels::wgpu_backend::WgpuError::Shape(_)),
        "{e}"
    );
}
