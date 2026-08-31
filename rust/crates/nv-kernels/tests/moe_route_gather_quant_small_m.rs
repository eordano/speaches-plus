#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use std::ffi::c_void;
use std::sync::Arc;
mod common;
use common::swizzled_dst;

const MIN_TILE: usize = 128;
const E: usize = 256;
const K: usize = 8;
const HIDDEN: usize = 2048;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn unit_f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn swizzled_scale_bytes(rows: usize, cols: usize) -> usize {
    let k_blocks = cols / 16;
    rows.div_ceil(128) * 128 * k_blocks.div_ceil(4) * 4
}

struct FusedOut {
    ids: Vec<i32>,
    weights: Vec<f32>,
    gu_mini: Vec<f32>,
    dn_mini: Vec<f32>,
    x_fp4: Vec<u8>,
    x_sf: Vec<u8>,
}

struct Inputs {
    logits: CudaSlice<f32>,
    x: CudaSlice<u16>,
    globals_gu: CudaSlice<f32>,
    globals_dn: CudaSlice<f32>,
    n_tokens: usize,
    tiles_per_token: usize,
    shared_tail_id: i32,
    mode: i32,
}

fn build_inputs(
    stream: &Arc<CudaStream>,
    rng: &mut Rng,
    n_tokens: usize,
    folded: bool,
    mode: i32,
) -> Inputs {
    let logits_host: Vec<f32> = (0..n_tokens * E).map(|_| rng.unit_f32() * 8.0 - 4.0).collect();
    let x_host: Vec<u16> = (0..n_tokens * HIDDEN)
        .map(|_| half::bf16::from_f32(rng.unit_f32() * 4.0 - 2.0).to_bits())
        .collect();
    let e_total = E + folded as usize;
    let g_gu: Vec<f32> = (0..e_total).map(|_| 0.5 + rng.unit_f32()).collect();
    let g_dn: Vec<f32> = (0..e_total).map(|_| 0.5 + rng.unit_f32()).collect();
    #[allow(deprecated)]
    let logits = stream.memcpy_stod(&logits_host).unwrap();
    #[allow(deprecated)]
    let x = stream.memcpy_stod(&x_host).unwrap();
    #[allow(deprecated)]
    let globals_gu = stream.memcpy_stod(&g_gu).unwrap();
    #[allow(deprecated)]
    let globals_dn = stream.memcpy_stod(&g_dn).unwrap();
    Inputs {
        logits,
        x,
        globals_gu,
        globals_dn,
        n_tokens,
        tiles_per_token: K + folded as usize,
        shared_tail_id: if folded { E as i32 } else { -1 },
        mode,
    }
}

fn run_fused(stream: &Arc<CudaStream>, inp: &Inputs, per_token_loop: bool, quant_ctas: i32) -> FusedOut {
    let n_tokens = inp.n_tokens;
    let tpt = inp.tiles_per_token;
    let tiles = n_tokens * tpt;
    let m_total = tiles * MIN_TILE;
    let sf_bytes = swizzled_scale_bytes(m_total, HIDDEN);
    #[allow(deprecated)]
    let mut ids = stream.memcpy_stod(&vec![-7i32; tiles]).unwrap();
    #[allow(deprecated)]
    let mut weights = stream.memcpy_stod(&vec![-7.0f32; tiles]).unwrap();
    #[allow(deprecated)]
    let mut gu_mini = stream.memcpy_stod(&vec![-7.0f32; tiles]).unwrap();
    #[allow(deprecated)]
    let mut dn_mini = stream.memcpy_stod(&vec![-7.0f32; tiles]).unwrap();
    #[allow(deprecated)]
    let mut x_fp4 = stream.memcpy_stod(&vec![0xAAu8; m_total * HIDDEN / 2]).unwrap();
    #[allow(deprecated)]
    let mut x_sf = stream.memcpy_stod(&vec![0xAAu8; sf_bytes]).unwrap();

    let group_k = HIDDEN / 16;
    let group_k_pad = group_k.div_ceil(4) * 4;
    let calls: Vec<(usize, usize)> = if per_token_loop {
        (0..n_tokens).map(|n| (n, 1usize)).collect()
    } else {
        vec![(0usize, n_tokens)]
    };
    for (n0, span) in calls {
        let (lp, _g1) = inp.logits.device_ptr(stream);
        let (xp, _g2) = inp.x.device_ptr(stream);
        let (ggu, _g3) = inp.globals_gu.device_ptr(stream);
        let (gdn, _g4) = inp.globals_dn.device_ptr(stream);
        let (ip, _g5) = ids.device_ptr_mut(stream);
        let (wp, _g6) = weights.device_ptr_mut(stream);
        let (gm, _g7) = gu_mini.device_ptr_mut(stream);
        let (dm, _g8) = dn_mini.device_ptr_mut(stream);
        let (fp, _g9) = x_fp4.device_ptr_mut(stream);
        let (sp, _g10) = x_sf.device_ptr_mut(stream);
        let rc = unsafe {
            nv_kernels::cuda::moe_route_gather_quant_small_m(
                stream.cu_stream() as *mut c_void,
                (lp + (n0 * E * 4) as u64) as *const f32,
                std::ptr::null(),
                (xp + (n0 * HIDDEN * 2) as u64) as *const u16,
                ggu as *const f32,
                gdn as *const f32,
                (ip + (n0 * tpt * 4) as u64) as *mut i32,
                (wp + (n0 * tpt * 4) as u64) as *mut f32,
                (gm + (n0 * tpt * 4) as u64) as *mut f32,
                (dm + (n0 * tpt * 4) as u64) as *mut f32,
                (fp + (n0 * tpt * MIN_TILE * HIDDEN / 2) as u64) as *mut u8,
                (sp + (n0 * tpt * (MIN_TILE / 128) * group_k_pad * 128) as u64) as *mut u8,
                span as i32,
                E as i32,
                K as i32,
                inp.mode,
                0.0,
                0,
                1.0,
                inp.shared_tail_id,
                HIDDEN as i32,
                MIN_TILE as i32,
                quant_ctas,
            )
        };
        assert_eq!(rc, 0, "moe_route_gather_quant_small_m rc={rc}");
    }
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let out = FusedOut {
        ids: stream.memcpy_dtov(&ids).unwrap(),
        weights: stream.memcpy_dtov(&weights).unwrap(),
        gu_mini: stream.memcpy_dtov(&gu_mini).unwrap(),
        dn_mini: stream.memcpy_dtov(&dn_mini).unwrap(),
        x_fp4: stream.memcpy_dtov(&x_fp4).unwrap(),
        x_sf: stream.memcpy_dtov(&x_sf).unwrap(),
    };
    out
}

fn run_unfused_reference(stream: &Arc<CudaStream>, inp: &Inputs) -> FusedOut {
    let n_tokens = inp.n_tokens;
    let tpt = inp.tiles_per_token;
    let tiles = n_tokens * tpt;
    let m_total = tiles * MIN_TILE;
    let sf_bytes = swizzled_scale_bytes(m_total, HIDDEN);
    #[allow(deprecated)]
    let mut ids = stream.memcpy_stod(&vec![-7i32; tiles]).unwrap();
    #[allow(deprecated)]
    let mut weights = stream.memcpy_stod(&vec![-7.0f32; tiles]).unwrap();
    let mut gu_mini = stream.alloc_zeros::<f32>(tiles).unwrap();
    let mut dn_mini = stream.alloc_zeros::<f32>(tiles).unwrap();
    let mut x_sorted = stream.alloc_zeros::<u16>(m_total * HIDDEN).unwrap();
    #[allow(deprecated)]
    let mut x_fp4 = stream.memcpy_stod(&vec![0xAAu8; m_total * HIDDEN / 2]).unwrap();
    #[allow(deprecated)]
    let mut x_sf = stream.memcpy_stod(&vec![0xAAu8; sf_bytes]).unwrap();
    let mut src_idx_host = vec![-1i32; m_total];
    for t in 0..tiles {
        src_idx_host[t * MIN_TILE] = (t / tpt) as i32;
    }
    #[allow(deprecated)]
    let src_idx = stream.memcpy_stod(&src_idx_host).unwrap();

    {
        let (lp, _g1) = inp.logits.device_ptr(stream);
        let (ip, _g2) = ids.device_ptr_mut(stream);
        let (wp, _g3) = weights.device_ptr_mut(stream);
        let rc = if inp.shared_tail_id >= 0 {
            unsafe {
                nv_kernels::cuda::moe_route_topk_shared_tail(
                    stream.cu_stream() as *mut c_void,
                    lp as *const f32,
                    std::ptr::null(),
                    ip as *mut i32,
                    wp as *mut f32,
                    n_tokens as i32,
                    E as i32,
                    K as i32,
                    inp.mode,
                    0.0,
                    0,
                    1.0,
                    inp.shared_tail_id,
                )
            }
        } else {
            unsafe {
                nv_kernels::cuda::moe_route_topk(
                    stream.cu_stream() as *mut c_void,
                    lp as *const f32,
                    std::ptr::null(),
                    ip as *mut i32,
                    wp as *mut f32,
                    n_tokens as i32,
                    E as i32,
                    K as i32,
                    inp.mode,
                    0.0,
                    0,
                    1.0,
                )
            }
        };
        assert_eq!(rc, 0, "route rc={rc}");
    }
    for (globals, mini) in [(&inp.globals_gu, &mut gu_mini), (&inp.globals_dn, &mut dn_mini)] {
        let (sp, _g1) = globals.device_ptr(stream);
        let (ip, _g2) = ids.device_ptr(stream);
        let (dp, _g3) = mini.device_ptr_mut(stream);
        let rc = unsafe {
            nv_kernels::cuda::gather_f32_by_ids(
                stream.cu_stream() as *mut c_void,
                sp as *const f32,
                ip as *const i32,
                dp as *mut f32,
                tiles as i32,
            )
        };
        assert_eq!(rc, 0, "gather_f32_by_ids rc={rc}");
    }
    {
        let (xp, _g1) = inp.x.device_ptr(stream);
        let (sp, _g2) = src_idx.device_ptr(stream);
        let (op, _g3) = x_sorted.device_ptr_mut(stream);
        let rc = unsafe {
            nv_kernels::cuda::gather_rows_bf16_strided(
                stream.cu_stream() as *mut c_void,
                xp as *const u16,
                sp as *const i32,
                op as *mut u16,
                m_total as i32,
                HIDDEN as i32,
                n_tokens as i32,
                MIN_TILE as i32,
            )
        };
        assert_eq!(rc, 0, "gather_rows_bf16_strided rc={rc}");
    }
    {
        let (xp, _g1) = x_sorted.device_ptr(stream);
        let (fp, _g2) = x_fp4.device_ptr_mut(stream);
        let (sp, _g3) = x_sf.device_ptr_mut(stream);
        let (gp, _g4) = gu_mini.device_ptr(stream);
        let rc = unsafe {
            nv_kernels::cuda::quantize_nvfp4_bf16_per_expert_strided(
                stream.cu_stream() as *mut c_void,
                xp as *const u16,
                fp as *mut u8,
                sp as *mut u8,
                gp as *const f32,
                MIN_TILE as i32,
                m_total as i32,
                HIDDEN as i32,
            )
        };
        assert_eq!(rc, 0, "quantize_nvfp4_bf16_per_expert_strided rc={rc}");
    }
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let out = FusedOut {
        ids: stream.memcpy_dtov(&ids).unwrap(),
        weights: stream.memcpy_dtov(&weights).unwrap(),
        gu_mini: stream.memcpy_dtov(&gu_mini).unwrap(),
        dn_mini: stream.memcpy_dtov(&dn_mini).unwrap(),
        x_fp4: stream.memcpy_dtov(&x_fp4).unwrap(),
        x_sf: stream.memcpy_dtov(&x_sf).unwrap(),
    };
    out
}

fn assert_real_row_regions_match(label: &str, a: &FusedOut, b: &FusedOut, tiles: usize) {
    assert_eq!(a.ids, b.ids, "{label}: topk ids differ");
    for (i, (x, y)) in a.weights.iter().zip(b.weights.iter()).enumerate() {
        assert!(
            x.to_bits() == y.to_bits(),
            "{label}: weight[{i}] {x} != {y} (bitwise)"
        );
    }
    for (i, (x, y)) in a
        .gu_mini
        .iter()
        .zip(b.gu_mini.iter())
        .chain(a.dn_mini.iter().zip(b.dn_mini.iter()))
        .enumerate()
    {
        assert!(x.to_bits() == y.to_bits(), "{label}: mini[{i}] {x} != {y}");
    }
    let group_k = HIDDEN / 16;
    for t in 0..tiles {
        let row = t * MIN_TILE;
        let lo = row * HIDDEN / 2;
        assert_eq!(
            a.x_fp4[lo..lo + HIDDEN / 2],
            b.x_fp4[lo..lo + HIDDEN / 2],
            "{label}: fp4 row bytes differ for tile {t}"
        );
        for kb in 0..group_k {
            let dst = swizzled_dst(row, kb, group_k);
            assert_eq!(
                a.x_sf[dst], b.x_sf[dst],
                "{label}: sf byte differs at tile {t} kb {kb}"
            );
        }
    }
}

#[test]
fn moe_route_gather_quant_small_m_matches_per_token_and_unfused_chain() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "moe_route_gather_quant_small_m: no CUDA device 0. This gate refuses to report \
                 success without running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!(
            "moe_route_gather_quant_small_m: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0"
        );
        return;
    };
    let stream = ctx.default_stream();
    for &folded in &[true, false] {
        for &mode in &[0i32, 1] {
            for &n_tokens in &[1usize, 2, 3, 5, 8] {
                let mut rng = Rng::new(
                    0x9e3779b97f4a7c15u64
                        ^ ((n_tokens as u64) << 8)
                        ^ ((folded as u64) << 20)
                        ^ ((mode as u64) << 24),
                );
                let inp = build_inputs(&stream, &mut rng, n_tokens, folded, mode);
                let tiles = n_tokens * inp.tiles_per_token;
                let one_call = run_fused(&stream, &inp, false, 1);
                let per_token = run_fused(&stream, &inp, true, 1);
                let split_quant = run_fused(&stream, &inp, false, 8);
                let label = format!("m={n_tokens} folded={folded} mode={mode}");
                assert_eq!(
                    one_call.x_fp4, per_token.x_fp4,
                    "{label}: fused small-m vs per-token fp4 bytes"
                );
                assert_eq!(
                    one_call.x_sf, per_token.x_sf,
                    "{label}: fused small-m vs per-token sf bytes"
                );
                assert_eq!(one_call.ids, per_token.ids, "{label}: ids");
                assert_eq!(
                    one_call.x_fp4, split_quant.x_fp4,
                    "{label}: quant_ctas=8 fp4 bytes"
                );
                assert_eq!(one_call.x_sf, split_quant.x_sf, "{label}: quant_ctas=8 sf bytes");
                assert_eq!(one_call.ids, split_quant.ids, "{label}: quant_ctas=8 ids");
                assert_eq!(
                    one_call.weights, split_quant.weights,
                    "{label}: quant_ctas=8 weights"
                );
                assert_eq!(
                    one_call.gu_mini, split_quant.gu_mini,
                    "{label}: quant_ctas=8 gu_mini"
                );
                assert_eq!(
                    one_call.dn_mini, split_quant.dn_mini,
                    "{label}: quant_ctas=8 dn_mini"
                );
                let unfused = run_unfused_reference(&stream, &inp);
                assert_real_row_regions_match(&label, &one_call, &unfused, tiles);
                eprintln!("  {label}: fused==per-token (full-buffer) and fused==unfused chain (real rows) bit-exact");
            }
        }
    }
}
