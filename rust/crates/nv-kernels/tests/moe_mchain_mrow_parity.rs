#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;
use std::sync::Arc;
mod common;
use common::swizzled_dst as swizzled_scale_dst;

const MIN_TILE: usize = 128;
const BLOCK_SIZE: usize = 16;
const E: usize = 256;
const TOPK: usize = 8;
const HIDDEN: usize = 2048;
const INTER: usize = 512;

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

    fn fill_bytes(&mut self, buf: &mut [u8]) {
        let mut it = buf.chunks_exact_mut(8);
        for c in &mut it {
            c.copy_from_slice(&self.next().to_le_bytes());
        }
        let rem = it.into_remainder();
        let last = self.next().to_le_bytes();
        let n = rem.len();
        rem.copy_from_slice(&last[..n]);
    }

    fn scale_byte(&mut self) -> u8 {
        0x30 + (self.next() % 0x18) as u8
    }

    fn range(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn unit_f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn swizzled_scale_bytes(rows: usize, cols: usize) -> usize {
    let k_blocks = cols / BLOCK_SIZE;
    rows.div_ceil(128) * 128 * k_blocks.div_ceil(4) * 4
}

fn verify_shaped_ids(rng: &mut Rng, m: usize, folded: bool, with_invalid: bool) -> Vec<i32> {
    let tiles_per_token = TOPK + folded as usize;
    let mut ids = vec![0i32; m * tiles_per_token];
    let mut token0: Vec<i32> = Vec::new();
    for t in 0..m {
        let mut picked: Vec<i32> = Vec::with_capacity(TOPK);
        while picked.len() < TOPK {
            let cand = if t > 0 && picked.len() < TOPK / 2 && !token0.is_empty() {
                token0[picked.len()]
            } else {
                rng.range(E) as i32
            };
            if !picked.contains(&cand) {
                picked.push(cand);
            }
        }
        if t == 0 {
            token0 = picked.clone();
        }
        for (j, &e) in picked.iter().enumerate() {
            ids[t * tiles_per_token + j] = e;
        }
        if folded {
            ids[t * tiles_per_token + TOPK] = E as i32;
        }
    }
    if with_invalid && m >= 2 {
        ids[tiles_per_token + 2] = -1;
    }
    ids
}

struct Weights {
    b_packed: CudaSlice<u8>,
    b_scales: CudaSlice<u8>,
    alphas: CudaSlice<f32>,
    e_total: usize,
    n: usize,
    k: usize,
}

fn build_weights(
    stream: &Arc<CudaStream>,
    rng: &mut Rng,
    e_total: usize,
    n: usize,
    k: usize,
) -> Weights {
    let group_k = k / BLOCK_SIZE;
    let mut b_packed_host = vec![0u8; e_total * n * k / 2];
    rng.fill_bytes(&mut b_packed_host);
    let sf_per_expert = swizzled_scale_bytes(n, k);
    let mut b_scales_host = vec![0u8; e_total * sf_per_expert];
    for e in 0..e_total {
        let base = e * sf_per_expert;
        for r in 0..n {
            for kb in 0..group_k {
                b_scales_host[base + swizzled_scale_dst(r, kb, group_k)] = rng.scale_byte();
            }
        }
    }
    let alphas_host: Vec<f32> = (0..e_total).map(|_| 0.5 + 1.5 * rng.unit_f32()).collect();
    #[allow(deprecated)]
    let b_packed = stream.memcpy_stod(&b_packed_host).unwrap();
    #[allow(deprecated)]
    let b_scales = stream.memcpy_stod(&b_scales_host).unwrap();
    #[allow(deprecated)]
    let alphas = stream.memcpy_stod(&alphas_host).unwrap();
    Weights {
        b_packed,
        b_scales,
        alphas,
        e_total,
        n,
        k,
    }
}

struct Acts {
    a_packed: CudaSlice<u8>,
    a_scales: CudaSlice<u8>,
    ids: CudaSlice<i32>,
    groups: usize,
}

fn build_acts(stream: &Arc<CudaStream>, rng: &mut Rng, ids_host: &[i32], k: usize) -> Acts {
    let groups = ids_host.len();
    let group_k = k / BLOCK_SIZE;
    let rows = groups * MIN_TILE;
    let mut a_packed_host = vec![0u8; rows * k / 2];
    for g in 0..groups {
        let lo = g * MIN_TILE * k / 2;
        rng.fill_bytes(&mut a_packed_host[lo..lo + k / 2]);
    }
    let mut a_scales_host = vec![0u8; swizzled_scale_bytes(rows, k)];
    for g in 0..groups {
        for kb in 0..group_k {
            a_scales_host[swizzled_scale_dst(g * MIN_TILE, kb, group_k)] = rng.scale_byte();
        }
    }
    #[allow(deprecated)]
    let a_packed = stream.memcpy_stod(&a_packed_host).unwrap();
    #[allow(deprecated)]
    let a_scales = stream.memcpy_stod(&a_scales_host).unwrap();
    #[allow(deprecated)]
    let ids = stream.memcpy_stod(&ids_host.to_vec()).unwrap();
    Acts {
        a_packed,
        a_scales,
        ids,
        groups,
    }
}

fn run_gemv_m1(stream: &Arc<CudaStream>, w: &Weights, a: &Acts) -> Vec<u16> {
    let d_stride = MIN_TILE * w.n;
    let mut d = stream.alloc_zeros::<u16>(a.groups * d_stride).unwrap();
    {
        let (ap, _g1) = a.a_packed.device_ptr(stream);
        let (asp, _g2) = a.a_scales.device_ptr(stream);
        let (bp, _g3) = w.b_packed.device_ptr(stream);
        let (bsp, _g4) = w.b_scales.device_ptr(stream);
        let (alp, _g5) = w.alphas.device_ptr(stream);
        let (dp, _g6) = d.device_ptr_mut(stream);
        let (ip, _g7) = a.ids.device_ptr(stream);
        let rc = unsafe {
            nv_kernels::cuda::moe_grouped_fp4_gemv_m1_bf16(
                stream.cu_stream() as *mut c_void,
                ap as *const u8,
                asp as *const u8,
                bp as *const u8,
                bsp as *const u8,
                alp as *const f32,
                dp as *mut u16,
                ip as *const i32,
                a.groups as i32,
                w.e_total as i32,
                w.n as i32,
                w.k as i32,
                MIN_TILE as i32,
                d_stride as i64,
            )
        };
        assert_eq!(rc, 0, "moe_grouped_fp4_gemv_m1_bf16 rc={rc}");
    }
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    stream.memcpy_dtov(&d).unwrap()
}

#[test]
fn gate_up_silu_quant_fused_bitwise_matches_unfused_chain() {
    let Ok(ctx) = CudaContext::new(0) else {
        panic!("no CUDA device: this parity gate must not silently skip");
    };
    let stream = ctx.default_stream();
    let mut rng = Rng::new(0x36c4a174);
    let mut combos = 0usize;
    for &m in &[1usize, 2, 3, 5, 8] {
        for &folded in &[true, false] {
            for &with_invalid in &[false, true] {
                let e_total = E + folded as usize;
                let ids_host = verify_shaped_ids(&mut rng, m, folded, with_invalid);
                let tiles = ids_host.len();
                let wg = build_weights(&stream, &mut rng, e_total, INTER, HIDDEN);
                let wu = build_weights(&stream, &mut rng, e_total, INTER, HIDDEN);
                let a = build_acts(&stream, &mut rng, &ids_host, HIDDEN);
                let dn_mini_host: Vec<f32> = (0..tiles).map(|_| 0.5 + rng.unit_f32()).collect();
                #[allow(deprecated)]
                let dn_mini = stream.memcpy_stod(&dn_mini_host).unwrap();
                let m_total = tiles * MIN_TILE;

                let y_gate = run_gemv_m1(&stream, &wg, &a);
                let y_up = run_gemv_m1(&stream, &wu, &a);
                #[allow(deprecated)]
                let y_gate_dev = stream.memcpy_stod(&y_gate).unwrap();
                #[allow(deprecated)]
                let y_up_dev = stream.memcpy_stod(&y_up).unwrap();
                let mut ref_fp4 = stream.alloc_zeros::<u8>(m_total * INTER / 2).unwrap();
                let mut ref_sf = stream
                    .alloc_zeros::<u8>(swizzled_scale_bytes(m_total, INTER))
                    .unwrap();
                {
                    let (gp, _g1) = y_gate_dev.device_ptr(&stream);
                    let (up, _g2) = y_up_dev.device_ptr(&stream);
                    let (fp, _g3) = ref_fp4.device_ptr_mut(&stream);
                    let (sp, _g4) = ref_sf.device_ptr_mut(&stream);
                    let (dm, _g5) = dn_mini.device_ptr(&stream);
                    let rc = unsafe {
                        nv_kernels::cuda::silu_mul_quantize_nvfp4_bf16_per_expert_strided(
                            stream.cu_stream() as *mut c_void,
                            gp as *const u16,
                            up as *const u16,
                            fp as *mut u8,
                            sp as *mut u8,
                            dm as *const f32,
                            MIN_TILE as i32,
                            m_total as i32,
                            INTER as i32,
                        )
                    };
                    assert_eq!(rc, 0, "silu_mul_quantize rc={rc}");
                }

                let mut new_fp4 = stream.alloc_zeros::<u8>(m_total * INTER / 2).unwrap();
                let mut new_sf = stream
                    .alloc_zeros::<u8>(swizzled_scale_bytes(m_total, INTER))
                    .unwrap();
                {
                    let (xf, _g1) = a.a_packed.device_ptr(&stream);
                    let (xs, _g2) = a.a_scales.device_ptr(&stream);
                    let (gwp, _g3) = wg.b_packed.device_ptr(&stream);
                    let (gsp, _g4) = wg.b_scales.device_ptr(&stream);
                    let (gap, _g5) = wg.alphas.device_ptr(&stream);
                    let (uwp, _g6) = wu.b_packed.device_ptr(&stream);
                    let (usp, _g7) = wu.b_scales.device_ptr(&stream);
                    let (uap, _g8) = wu.alphas.device_ptr(&stream);
                    let (dm, _g9) = dn_mini.device_ptr(&stream);
                    let (ip, _g10) = a.ids.device_ptr(&stream);
                    let (fp, _g11) = new_fp4.device_ptr_mut(&stream);
                    let (sp, _g12) = new_sf.device_ptr_mut(&stream);
                    let rc = unsafe {
                        nv_kernels::cuda::moe_grouped_fp4_gemv_gate_up_silu_quant_small_m(
                            stream.cu_stream() as *mut c_void,
                            xf as *const u8,
                            xs as *const u8,
                            gwp as *const u8,
                            gsp as *const u8,
                            gap as *const f32,
                            uwp as *const u8,
                            usp as *const u8,
                            uap as *const f32,
                            dm as *const f32,
                            ip as *const i32,
                            fp as *mut u8,
                            sp as *mut u8,
                            tiles as i32,
                            e_total as i32,
                            INTER as i32,
                            HIDDEN as i32,
                            MIN_TILE as i32,
                        )
                    };
                    assert_eq!(rc, 0, "gate_up_silu_quant_small_m rc={rc}");
                }
                stream.synchronize().unwrap();
                #[allow(deprecated)]
                let ref_fp4_h: Vec<u8> = stream.memcpy_dtov(&ref_fp4).unwrap();
                #[allow(deprecated)]
                let new_fp4_h: Vec<u8> = stream.memcpy_dtov(&new_fp4).unwrap();
                #[allow(deprecated)]
                let ref_sf_h: Vec<u8> = stream.memcpy_dtov(&ref_sf).unwrap();
                #[allow(deprecated)]
                let new_sf_h: Vec<u8> = stream.memcpy_dtov(&new_sf).unwrap();
                assert_eq!(
                    ref_fp4_h, new_fp4_h,
                    "fused gate+up+silu+quant fp4 bytes diverge: m={m} folded={folded} \
                     invalid={with_invalid} ids={ids_host:?}"
                );
                assert_eq!(
                    ref_sf_h, new_sf_h,
                    "fused gate+up+silu+quant sf bytes diverge: m={m} folded={folded} \
                     invalid={with_invalid} ids={ids_host:?}"
                );
                combos += 1;
            }
        }
    }
    println!(
        "[mchain-parity] fused gate+up+silu+quant BIT-EXACT vs unfused m1+m1+silu on {combos} combos"
    );
}

#[test]
fn router_gemv_sg_xstage_bitwise_matches_per_row_gemv_bf16_plus_widen() {
    let Ok(ctx) = CudaContext::new(0) else {
        panic!("no CUDA device: this parity gate must not silently skip");
    };
    let stream = ctx.default_stream();
    let mut rng = Rng::new(0x36c4a173);
    for &m in &[1usize, 2, 3, 5, 8] {
        for &with_sg in &[true, false] {
            let x_host: Vec<u16> = (0..m * HIDDEN)
                .map(|_| bf16::from_f32(rng.unit_f32() * 4.0 - 2.0).to_bits())
                .collect();
            let wg_host: Vec<u16> = (0..E * HIDDEN)
                .map(|_| bf16::from_f32(rng.unit_f32() * 0.2 - 0.1).to_bits())
                .collect();
            let wsg_host: Vec<u16> = (0..HIDDEN)
                .map(|_| bf16::from_f32(rng.unit_f32() * 0.2 - 0.1).to_bits())
                .collect();
            #[allow(deprecated)]
            let x = stream.memcpy_stod(&x_host).unwrap();
            #[allow(deprecated)]
            let wg = stream.memcpy_stod(&wg_host).unwrap();
            #[allow(deprecated)]
            let wsg = stream.memcpy_stod(&wsg_host).unwrap();

            let mut logits_ref = vec![0f32; m * E];
            let mut sg_ref = vec![0u16; m];
            for r in 0..m {
                let mut y = stream.alloc_zeros::<u16>(E).unwrap();
                let mut ysg = stream.alloc_zeros::<u16>(1).unwrap();
                {
                    let (xp, _g1) = x.device_ptr(&stream);
                    let xp = xp + (r * HIDDEN * 2) as u64;
                    let (wp, _g2) = wg.device_ptr(&stream);
                    let (yp, _g3) = y.device_ptr_mut(&stream);
                    let rc = unsafe {
                        nv_kernels::cuda::gemv_bf16(
                            stream.cu_stream() as *mut c_void,
                            wp as *const u16,
                            xp as *const u16,
                            yp as *mut u16,
                            E as i32,
                            HIDDEN as i32,
                        )
                    };
                    assert_eq!(rc, 0);
                    let (sp, _g4) = wsg.device_ptr(&stream);
                    let (yg, _g5) = ysg.device_ptr_mut(&stream);
                    let rc = unsafe {
                        nv_kernels::cuda::gemv_bf16(
                            stream.cu_stream() as *mut c_void,
                            sp as *const u16,
                            xp as *const u16,
                            yg as *mut u16,
                            1,
                            HIDDEN as i32,
                        )
                    };
                    assert_eq!(rc, 0);
                }
                stream.synchronize().unwrap();
                #[allow(deprecated)]
                let y_h: Vec<u16> = stream.memcpy_dtov(&y).unwrap();
                #[allow(deprecated)]
                let ysg_h: Vec<u16> = stream.memcpy_dtov(&ysg).unwrap();
                for n in 0..E {
                    logits_ref[r * E + n] = f32::from(bf16::from_bits(y_h[n]));
                }
                sg_ref[r] = ysg_h[0];
            }

            let mut logits_new = stream.alloc_zeros::<f32>(m * E).unwrap();
            let mut sg_new = stream.alloc_zeros::<u16>(m).unwrap();
            let mut x_stage = stream.alloc_zeros::<u16>(m * HIDDEN).unwrap();
            {
                let (xp, _g1) = x.device_ptr(&stream);
                let (wp, _g2) = wg.device_ptr(&stream);
                let (sp, _g3) = wsg.device_ptr(&stream);
                let (lp, _g4) = logits_new.device_ptr_mut(&stream);
                let (gp, _g5) = sg_new.device_ptr_mut(&stream);
                let (stp, _g6) = x_stage.device_ptr_mut(&stream);
                let rc = unsafe {
                    nv_kernels::cuda::moe_router_gemv_sg_xstage_small_m(
                        stream.cu_stream() as *mut c_void,
                        xp as *const u16,
                        wp as *const u16,
                        if with_sg {
                            sp as *const u16
                        } else {
                            std::ptr::null()
                        },
                        lp as *mut f32,
                        if with_sg {
                            gp as *mut u16
                        } else {
                            std::ptr::null_mut()
                        },
                        stp as *mut u16,
                        m as i32,
                        E as i32,
                        HIDDEN as i32,
                    )
                };
                assert_eq!(rc, 0, "moe_router_gemv_sg_xstage_small_m rc={rc}");
            }
            stream.synchronize().unwrap();
            #[allow(deprecated)]
            let logits_h: Vec<f32> = stream.memcpy_dtov(&logits_new).unwrap();
            #[allow(deprecated)]
            let sg_h: Vec<u16> = stream.memcpy_dtov(&sg_new).unwrap();
            #[allow(deprecated)]
            let x_stage_h: Vec<u16> = stream.memcpy_dtov(&x_stage).unwrap();
            let mut logit_diffs = 0usize;
            for i in 0..m * E {
                if logits_h[i].to_bits() != logits_ref[i].to_bits() {
                    logit_diffs += 1;
                }
            }
            assert_eq!(
                logit_diffs, 0,
                "router gemv logits diverge from per-row gemv_bf16+widen: m={m} with_sg={with_sg}"
            );
            if with_sg {
                assert_eq!(sg_h, sg_ref, "router gemv sg logits diverge: m={m}");
            }
            assert_eq!(x_stage_h, x_host, "router gemv x stage diverges: m={m}");
        }
    }
    println!("[mchain-parity] router gemv+sg+xstage BIT-EXACT vs per-row gemv_bf16 on all m");
}
