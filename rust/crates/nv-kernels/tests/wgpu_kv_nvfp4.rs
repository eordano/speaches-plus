#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use nv_kernels::wgpu_backend::compose;
use nv_kernels::wgpu_backend::dequant::bytes_to_words;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::flash_decode as fd;
use nv_kernels::wgpu_backend::kernels::kv_fp8;
use nv_kernels::wgpu_backend::kernels::kv_nvfp4 as kv4;

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32) - 1.0
    }
    fn bf16_vec(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_bits())
            .collect()
    }
}

fn outlier_channelled_k_cache(slots: usize, n_kv: usize, hd: usize, seed: u64) -> Vec<u16> {
    let mut r = Lcg(seed | 1);
    let mut cache = r.bf16_vec(slots * n_kv * hd, 1.5);
    for slot in 0..slots {
        for kvh in 0..n_kv {
            for d in [1usize, hd - 3] {
                let e = (slot * n_kv + kvh) * hd + d;
                let v = half::bf16::from_bits(cache[e]).to_f32() * 40.0;
                cache[e] = half::bf16::from_f32(v).to_bits();
            }
        }
    }
    cache
}

#[test]
fn both_quantizers_match_the_cpu_reference_bit_exactly_across_decode_and_chunk_shapes() {
    let Some(c) = ctx_or_skip("wgpu_kv_nvfp4") else {
        return;
    };
    let n_kv = 2usize;
    let hd = 64usize;
    let slots = 96usize;
    let cache = outlier_channelled_k_cache(slots, n_kv, hd, 0x4b_5eed);
    for (start, tokens) in [(0usize, 1usize), (31, 1), (32, 1), (5, 48), (0, 96), (90, 6)] {
        for k_arm in [false, true] {
            let payload = slots * n_kv * hd / 2;
            let scale_len = if k_arm {
                kv4::k_scale_blocks(slots) * n_kv * hd
            } else {
                slots * n_kv
            };
            let mut gpu_out = vec![0u8; payload];
            let mut gpu_scales = vec![0f32; scale_len];
            let mut cpu_out = vec![0u8; payload];
            let mut cpu_scales = vec![0f32; scale_len];
            if k_arm {
                kv4::quantize_kv_nvfp4_k_channel_blocks(
                    c, &cache, &mut gpu_out, &mut gpu_scales, start, tokens, n_kv, hd, slots,
                )
                .expect("gpu k blocks");
                kv4::cpu_quantize_kv_nvfp4_k_channel_blocks(
                    &cache, &mut cpu_out, &mut cpu_scales, start, tokens, n_kv, hd, slots,
                );
            } else {
                kv4::quantize_kv_nvfp4_v_rows(
                    c, &cache, &mut gpu_out, &mut gpu_scales, start, tokens, n_kv, hd, slots,
                )
                .expect("gpu v rows");
                kv4::cpu_quantize_kv_nvfp4_v_rows(
                    &cache, &mut cpu_out, &mut cpu_scales, start, tokens, n_kv, hd, slots,
                );
            }
            let tag = format!("k_arm={k_arm} start={start} tokens={tokens}");
            let sdiff = gpu_scales
                .iter()
                .zip(cpu_scales.iter())
                .position(|(g, cc)| g.to_bits() != cc.to_bits());
            assert!(
                sdiff.is_none(),
                "{tag}: scale {} differs: gpu {} cpu {}",
                sdiff.unwrap(),
                gpu_scales[sdiff.unwrap()],
                cpu_scales[sdiff.unwrap()]
            );
            let ndiff = gpu_out
                .iter()
                .zip(cpu_out.iter())
                .position(|(g, cc)| g != cc);
            assert!(
                ndiff.is_none(),
                "{tag}: payload byte {} differs: gpu {:#04x} cpu {:#04x}",
                ndiff.unwrap(),
                gpu_out[ndiff.unwrap()],
                cpu_out[ndiff.unwrap()]
            );
        }
    }
}

const ROUND_TRIP_REL_BAND_PROVES_E2M1_ENGAGED_NOT_A_COPY: (f64, f64) = (0.01, 0.5);

#[test]
fn the_k_channel_arm_round_trips_outlier_channels_tighter_than_a_row_scale_would() {
    let n_kv = 2usize;
    let hd = 64usize;
    let slots = 64usize;
    let cache = outlier_channelled_k_cache(slots, n_kv, hd, 0xc4a2_2e1f);
    let payload = slots * n_kv * hd / 2;
    let mut k_out = vec![0u8; payload];
    let mut k_scales = vec![0f32; kv4::k_scale_blocks(slots) * n_kv * hd];
    kv4::cpu_quantize_kv_nvfp4_k_channel_blocks(
        &cache, &mut k_out, &mut k_scales, 0, slots, n_kv, hd, slots,
    );
    let mut v_out = vec![0u8; payload];
    let mut v_scales = vec![0f32; slots * n_kv];
    kv4::cpu_quantize_kv_nvfp4_v_rows(&cache, &mut v_out, &mut v_scales, 0, slots, n_kv, hd, slots);
    let outlier = |d: usize| d == 1 || d == hd - 3;
    let mut ch_err = [0f64; 2];
    let mut row_err = [0f64; 2];
    let mut norm = [0f64; 2];
    for slot in 0..slots {
        for kvh in 0..n_kv {
            for d in 0..hd {
                let side = usize::from(outlier(d));
                let e = (slot * n_kv + kvh) * hd + d;
                let v = f32::from_bits((cache[e] as u32) << 16) as f64;
                let qk = kv4::cpu_dequantize_kv_nvfp4_k(&k_out, &k_scales, slot, n_kv, kvh, hd, d) as f64;
                let qv = kv4::cpu_dequantize_kv_nvfp4_v(&v_out, &v_scales, slot, n_kv, kvh, hd, d) as f64;
                ch_err[side] += (qk - v) * (qk - v);
                row_err[side] += (qv - v) * (qv - v);
                norm[side] += v * v;
            }
        }
    }
    let rms = |err: f64, n: f64| (err / n).sqrt();
    let quiet_channel = rms(ch_err[0], norm[0]);
    let quiet_row = rms(row_err[0], norm[0]);
    eprintln!(
        "[kv-nvfp4] outlier-channel K: quiet-channel rms rel err per-channel {quiet_channel:.5} \
         vs per-row {quiet_row:.5}; outlier channels {:.5} vs {:.5}",
        rms(ch_err[1], norm[1]),
        rms(row_err[1], norm[1])
    );
    assert!(
        quiet_channel * 2.0 < quiet_row,
        "per-channel K scaling must beat a row scale at least 2x on the channels the outliers \
         would flatten (got {quiet_channel:.5} vs {quiet_row:.5}); this is the entire reason the \
         K4 arm exists"
    );
    let (lo, hi) = ROUND_TRIP_REL_BAND_PROVES_E2M1_ENGAGED_NOT_A_COPY;
    assert!(
        quiet_channel > lo && quiet_channel < hi,
        "K4 quiet-channel round-trip rms {quiet_channel} escaped ({lo}, {hi}): below means the \
         quantize never engaged, above means the scales or nibble packing are wrong"
    );
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct FdP {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    total: u32,
    start: u32,
    splits: u32,
    ring: u32,
    out_bf16: u32,
    scaling: f32,
    pad0: u32,
    fused: u32,
    pad2: u32,
    m_rows: u32,
    window: u32,
    pad3: u32,
    pad4: u32,
}

struct QuantizedCaches {
    k8: Vec<u8>,
    v8: Vec<u8>,
    ks8: Vec<f32>,
    vs8: Vec<f32>,
    k4: Vec<u8>,
    ks4: Vec<f32>,
    v4: Vec<u8>,
    vs4: Vec<f32>,
}

fn quantize_all(kc: &[u16], vc: &[u16], slots: usize, n_kv: usize, hd: usize) -> QuantizedCaches {
    let elems = slots * n_kv * hd;
    let mut q = QuantizedCaches {
        k8: vec![0u8; elems],
        v8: vec![0u8; elems],
        ks8: vec![0f32; slots * n_kv],
        vs8: vec![0f32; slots * n_kv],
        k4: vec![0u8; elems / 2],
        ks4: vec![0f32; kv4::k_scale_blocks(slots) * n_kv * hd],
        v4: vec![0u8; elems / 2],
        vs4: vec![0f32; slots * n_kv],
    };
    kv_fp8::cpu_quantize_kv_fp8(kc, &mut q.k8, &mut q.ks8, 0, slots, n_kv, hd, 0);
    kv_fp8::cpu_quantize_kv_fp8(vc, &mut q.v8, &mut q.vs8, 0, slots, n_kv, hd, 0);
    kv4::cpu_quantize_kv_nvfp4_k_channel_blocks(kc, &mut q.k4, &mut q.ks4, 0, slots, n_kv, hd, slots);
    kv4::cpu_quantize_kv_nvfp4_v_rows(vc, &mut q.v4, &mut q.vs4, 0, slots, n_kv, hd, slots);
    q
}

fn dequant_k(q: &QuantizedCaches, k4_arm: bool, slot: usize, n_kv: usize, kvh: usize, hd: usize, d: usize) -> f64 {
    let sink = kv4::KV_NVFP4_SINK_SLOTS_STAY_FP8_TO_ANCHOR_SOFTMAX as usize;
    let e = (slot * n_kv + kvh) * hd + d;
    if !k4_arm || slot < sink {
        (kv_fp8::decode_e4m3(q.k8[e]) * q.ks8[slot * n_kv + kvh]) as f64
    } else {
        kv4::cpu_dequantize_kv_nvfp4_k(&q.k4, &q.ks4, slot, n_kv, kvh, hd, d) as f64
    }
}

fn dequant_v(q: &QuantizedCaches, slot: usize, n_kv: usize, kvh: usize, hd: usize, d: usize) -> f64 {
    let sink = kv4::KV_NVFP4_SINK_SLOTS_STAY_FP8_TO_ANCHOR_SOFTMAX as usize;
    let e = (slot * n_kv + kvh) * hd + d;
    if slot < sink {
        (kv_fp8::decode_e4m3(q.v8[e]) * q.vs8[slot * n_kv + kvh]) as f64
    } else {
        kv4::cpu_dequantize_kv_nvfp4_v(&q.v4, &q.vs4, slot, n_kv, kvh, hd, d) as f64
    }
}

fn f64_attention_reference(
    q: &[f32],
    caches: &QuantizedCaches,
    k4_arm: bool,
    n_heads: usize,
    n_kv: usize,
    hd: usize,
    total: usize,
) -> Vec<f64> {
    let group = n_heads / n_kv;
    let scaling = 1.0 / (hd as f64).sqrt();
    let mut out = vec![0f64; n_heads * hd];
    for h in 0..n_heads {
        let kvh = h / group;
        let mut scores = vec![0f64; total];
        for (p, s) in scores.iter_mut().enumerate() {
            let mut dot = 0f64;
            for d in 0..hd {
                dot += q[h * hd + d] as f64 * dequant_k(caches, k4_arm, p, n_kv, kvh, hd, d);
            }
            *s = dot * scaling;
        }
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut l = 0f64;
        let w: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        for wp in &w {
            l += wp;
        }
        for (p, wp) in w.iter().enumerate() {
            for d in 0..hd {
                out[h * hd + d] += wp / l * dequant_v(caches, p, n_kv, kvh, hd, d);
            }
        }
    }
    out
}

#[test]
fn nvfp4_fold_stage1_tracks_an_f64_attention_reference_for_both_arms() {
    let Some(c) = ctx_or_skip("wgpu_kv_nvfp4") else {
        return;
    };
    let n_heads = 8usize;
    let n_kv = 2usize;
    let hd = 128usize;
    let slots = 160usize;
    let splits = 16u32;
    let mut r = Lcg(0xf01d_4b1d | 1);
    let kc = outlier_channelled_k_cache(slots, n_kv, hd, 0x5eed_0001);
    let vc = r.bf16_vec(slots * n_kv * hd, 1.0);
    let qv: Vec<f32> = (0..n_heads * hd).map(|_| r.next_f32()).collect();
    let caches = quantize_all(&kc, &vc, slots, n_kv, hd);

    let sg_ok = c.caps.subgroup && c.subgroup_width() == Some(32);
    for total in [3usize, 33, 100, 160] {
        for k4_arm in [false, true] {
            for fold in [1u32, 2, 4] {
                for sg in [false, true] {
                    if sg && !sg_ok {
                        continue;
                    }
                    let body = fd::fold_stage1_source_nvfp4(hd as u32, sg, fold, k4_arm);
                    let entry = fd::fold_stage1_entry_nvfp4(hd as u32, sg, fold, k4_arm);
                    let src = compose(&format!("{}\n{}", fd::WGSL, body));
                    let params = FdP {
                        n_heads: n_heads as u32,
                        n_kv: n_kv as u32,
                        head_dim: hd as u32,
                        total: total as u32,
                        splits,
                        scaling: 1.0 / (hd as f32).sqrt(),
                        m_rows: 1,
                        ..Default::default()
                    };
                    let qb = dispatch::storage_from_slice(c, "kv4f-q", &qv);
                    let kb = dispatch::storage_from_slice(c, "kv4f-k8", &bytes_to_words(&caches.k8));
                    let vb = dispatch::storage_from_slice(c, "kv4f-v8", &bytes_to_words(&caches.v8));
                    let ksb = dispatch::storage_from_slice(c, "kv4f-ks8", &caches.ks8);
                    let vsb = dispatch::storage_from_slice(c, "kv4f-vs8", &caches.vs8);
                    let k4b = dispatch::storage_from_slice(c, "kv4f-k4", &bytes_to_words(&caches.k4));
                    let ks4b = dispatch::storage_from_slice(c, "kv4f-ks4", &caches.ks4);
                    let v4b = dispatch::storage_from_slice(c, "kv4f-v4", &bytes_to_words(&caches.v4));
                    let vs4b = dispatch::storage_from_slice(c, "kv4f-vs4", &caches.vs4);
                    let pb = dispatch::uniform_from(c, "kv4f-p", &params);
                    let scratch_elems = n_heads * splits as usize * (hd + 2);
                    let sb = dispatch::storage_zeroed(c, "kv4f-scratch", (scratch_elems * 4) as u64);
                    let ob = dispatch::storage_zeroed(c, "kv4f-out", (n_heads * hd * 4) as u64);
                    let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![
                        (0, &qb),
                        (4, &pb),
                        (5, &kb),
                        (6, &vb),
                        (7, &sb),
                        (8, &ksb),
                        (9, &vsb),
                        (15, &v4b),
                        (17, &vs4b),
                    ];
                    if k4_arm {
                        binds.push((14, &k4b));
                        binds.push((16, &ks4b));
                    }
                    let tag = format!("total={total} k4={k4_arm} fold={fold} sg={sg}");
                    dispatch::run(
                        c,
                        "kv4f-stage1",
                        &src,
                        &entry,
                        &binds,
                        (n_heads as u32 / fold, splits, 1),
                    )
                    .unwrap_or_else(|e| panic!("{tag}: stage1: {e}"));
                    dispatch::run(
                        c,
                        "kv4f-stage2",
                        &src,
                        fd::ENTRY_STAGE2,
                        &[(3, &ob), (4, &pb), (7, &sb)],
                        (n_heads as u32, 1, 1),
                    )
                    .unwrap_or_else(|e| panic!("{tag}: stage2: {e}"));
                    let got: Vec<u32> = dispatch::read_back(c, &ob, n_heads * hd)
                        .unwrap_or_else(|e| panic!("{tag}: read_back: {e}"));
                    let want = f64_attention_reference(&qv, &caches, k4_arm, n_heads, n_kv, hd, total);
                    let mut maxrel = 0f64;
                    let scale = want.iter().fold(0f64, |a, v| a.max(v.abs())).max(1e-9);
                    for (i, w) in got.iter().enumerate() {
                        let g = f32::from_bits(*w) as f64;
                        maxrel = maxrel.max((g - want[i]).abs() / scale);
                    }
                    eprintln!("[kv4-fold] {tag}: max rel vs f64 ref {maxrel:.3e}");
                    assert!(
                        maxrel < 1e-3,
                        "{tag}: stage1+stage2 drifted {maxrel:.3e} from the f64 reference on the \
                         SAME dequantized values; only f32 accumulation order may differ"
                    );
                }
            }
        }
    }
}
