#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::LayerType;
use nv_models::laguna::LagunaConfig;
use nv_models::laguna_fp8::LagunaKvCacheFp8;

fn model_dir() -> Option<std::path::PathBuf> {
    if let Ok(d) = std::env::var("NV_LAGUNA_DIR") {
        return Some(std::path::PathBuf::from(d));
    }
    let home = std::env::var("HOME").ok()?;
    let p = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--poolside--Laguna-XS-2.1-NVFP4/snapshots/main");
    p.is_dir().then_some(p)
}

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 33) as u32;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn rand_kv(seed: u64, t: usize, n_kv: usize, hd: usize) -> Vec<f32> {
    let mut rng = Lcg(seed);
    let mut out = Vec::with_capacity(t * n_kv * hd);
    for tok in 0..t {
        for h in 0..n_kv {
            let mag = 0.5 + 2.0 * ((tok * n_kv + h) % 7) as f32;
            for _ in 0..hd {
                out.push(rng.next_f32() * mag);
            }
        }
    }
    out
}

fn to_bf16_tensor(data: &[f32], shape: (usize, usize, usize, usize), dev: &Device) -> Tensor {
    Tensor::from_vec(data.to_vec(), shape, dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

fn host_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn e4m3_decode(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((byte >> 3) & 0xF) as i32;
    let man = (byte & 0x7) as f32;
    if exp == 0xF && (byte & 0x7) == 0x7 {
        return f32::NAN;
    }
    if exp == 0 {
        sign * man * (2.0f32).powi(-9)
    } else {
        sign * (1.0 + man / 8.0) * (2.0f32).powi(exp - 7)
    }
}

fn e4m3_encode_round_trip(v: f32) -> f32 {
    if v == 0.0 {
        return 0.0;
    }
    let mag = v.abs().min(448.0);
    let mut best = 0.0f32;
    let mut best_d = f32::INFINITY;
    let mut best_byte = 0u8;
    for byte in 0u8..=0x7E {
        let dec = e4m3_decode(byte);
        if dec.is_nan() {
            continue;
        }
        let d = (dec - mag).abs();
        if d < best_d || (d == best_d && byte & 1 == 0 && best_byte & 1 == 1) {
            best_d = d;
            best = dec;
            best_byte = byte;
        }
    }
    best * v.signum()
}

fn quant_ref(x: &[f32], t: usize, n_kv: usize, hd: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for tok in 0..t {
        for h in 0..n_kv {
            let base = (tok * n_kv + h) * hd;
            let amax = x[base..base + hd].iter().fold(0f32, |m, v| m.max(v.abs()));
            let scale = if amax > 0.0 { amax / 448.0 } else { 1.0 };
            let inv = if amax > 0.0 { 448.0 / amax } else { 1.0 };
            for d in 0..hd {
                out[base + d] = e4m3_encode_round_trip(x[base + d] * inv) * scale;
            }
        }
    }
    out
}

fn rel_stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    let denom = a.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
    let mut max_rel = 0f32;
    let mut sum = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs() / denom;
        max_rel = max_rel.max(d);
        sum += d as f64;
    }
    (max_rel, (sum / a.len() as f64) as f32)
}

#[test]
#[ignore]
fn laguna_fp8_unit_probes() {
    if std::env::var("NV_LAGUNA_TEST").is_err() {
        eprintln!("set NV_LAGUNA_TEST=1 to run");
        return;
    }
    let dir = model_dir().expect("Laguna snapshot not found");
    let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let config = LagunaConfig::from_hf_json_str(&raw).unwrap();
    let device = Device::new_cuda(0).unwrap();
    let n_kv = config.num_key_value_heads;
    let hd = config.head_dim;
    let full_idx = config
        .layer_types
        .iter()
        .position(|k| matches!(k, LayerType::FullAttention))
        .unwrap();
    let n_q = config.num_heads_for_layer(full_idx);
    eprintln!("full layer {full_idx}: n_q={n_q} n_kv={n_kv} hd={hd}");

    let t = 12usize;
    let k_host = rand_kv(1, t, n_kv, hd);
    let v_host = rand_kv(2, t, n_kv, hd);
    let k = to_bf16_tensor(&k_host, (1, t, n_kv, hd), &device);
    let v = to_bf16_tensor(&v_host, (1, t, n_kv, hd), &device);
    let k_ref = host_f32(&k);
    let v_ref = host_f32(&v);

    let mut cache = LagunaKvCacheFp8::new(&config, 64, &device, DType::BF16).unwrap();
    cache.prepare_for_decode_dev(0, t).unwrap();
    cache.write_at_impl(full_idx, &k, &v).unwrap();
    let (k_rt, v_rt) = cache.view_bf16(full_idx, t).unwrap();
    let k_rt_h = host_f32(&k_rt);
    let v_rt_h = host_f32(&v_rt);
    let (k_max, k_mean) = rel_stats(&k_ref, &k_rt_h);
    let (v_max, v_mean) = rel_stats(&v_ref, &v_rt_h);
    eprintln!("probe1 round-trip: k max_rel={k_max:.5} mean_rel={k_mean:.5}; v max_rel={v_max:.5} mean_rel={v_mean:.5}");

    let mut cache2 = LagunaKvCacheFp8::new(&config, 64, &device, DType::BF16).unwrap();
    let k_a = to_bf16_tensor(&k_host[..5 * n_kv * hd], (1, 5, n_kv, hd), &device);
    let v_a = to_bf16_tensor(&v_host[..5 * n_kv * hd], (1, 5, n_kv, hd), &device);
    let k_b = to_bf16_tensor(&k_host[5 * n_kv * hd..], (1, 7, n_kv, hd), &device);
    let v_b = to_bf16_tensor(&v_host[5 * n_kv * hd..], (1, 7, n_kv, hd), &device);
    cache2.prepare_for_decode_dev(0, 5).unwrap();
    cache2.write_at_impl(full_idx, &k_a, &v_a).unwrap();
    cache2.advance(5);
    cache2.prepare_for_decode_dev(5, 12).unwrap();
    cache2.write_at_impl(full_idx, &k_b, &v_b).unwrap();
    cache2.advance(7);
    let (k_c, v_c) = cache2.view_bf16(full_idx, t).unwrap();
    let k_c_h = host_f32(&k_c);
    let v_c_h = host_f32(&v_c);
    let k_chunk_eq = k_c_h == k_rt_h;
    let v_chunk_eq = v_c_h == v_rt_h;
    eprintln!("probe2 chunked==single: k {k_chunk_eq} v {v_chunk_eq}");
    if !k_chunk_eq {
        let (m, _) = rel_stats(&k_rt_h, &k_c_h);
        eprintln!("probe2 chunked k max_rel vs single {m:.5}");
    }

    let big_t = t + 2;
    let k_big_host = rand_kv(3, big_t, n_kv, hd);
    let k_big = to_bf16_tensor(&k_big_host, (1, big_t, n_kv, hd), &device);
    let k_view = k_big.narrow(1, 1, t).unwrap().contiguous().unwrap();
    let expect = host_f32(&k_big.narrow(1, 1, t).unwrap());
    let alias = host_f32(&k_big.narrow(1, 0, t).unwrap());
    let mut cache3 = LagunaKvCacheFp8::new(&config, 64, &device, DType::BF16).unwrap();
    cache3.prepare_for_decode_dev(0, t).unwrap();
    cache3.write_at_impl(full_idx, &k_view, &k_view).unwrap();
    let (k3, _v3) = cache3.view_bf16(full_idx, t).unwrap();
    let k3_h = host_f32(&k3);
    let (rel_expect, _) = rel_stats(&expect, &k3_h);
    let (rel_alias, _) = rel_stats(&alias, &k3_h);
    eprintln!(
        "probe3 offset view: rel-to-correct {rel_expect:.5}, rel-to-base-aliased {rel_alias:.5} \
         (small first number = offset honored; small second = OFFSET BUG)"
    );

    let q_host = rand_kv(7, 1, n_q, hd);
    let q = to_bf16_tensor(&q_host, (1, 1, n_q, hd), &device);
    let q_h = host_f32(&q);
    let n_total = t;
    cache.prepare_for_decode_dev(t, t + 1).unwrap();
    let scale = (hd as f32).powf(-0.5);
    let _ = n_total;
    let kq_host = rand_kv(11, 1, n_kv, hd);
    let vq_host = rand_kv(13, 1, n_kv, hd);
    let k1 = to_bf16_tensor(&kq_host, (1, 1, n_kv, hd), &device);
    let v1 = to_bf16_tensor(&vq_host, (1, 1, n_kv, hd), &device);
    cache.write_at_impl(full_idx, &k1, &v1).unwrap();
    let out = cache
        .decode_attention_fp8(full_idx, &q, n_q, None, scale)
        .unwrap()
        .expect("fp8 decode path");
    let out_h = host_f32(&out);
    let (k_all, v_all) = cache.view_bf16(full_idx, t + 1).unwrap();
    let k_all_h = host_f32(&k_all);
    let v_all_h = host_f32(&v_all);
    let total = t + 1;
    let group = n_q / n_kv;
    let mut ref_out = vec![0f32; n_q * hd];
    for qh in 0..n_q {
        let kvh = qh / group;
        let mut scores = vec![0f32; total];
        for i in 0..total {
            let mut s = 0f32;
            for d in 0..hd {
                s += q_h[qh * hd + d] * k_all_h[(i * n_kv + kvh) * hd + d];
            }
            scores[i] = s * scale;
        }
        let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut den = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            den += *s;
        }
        for s in scores.iter_mut() {
            *s /= den;
        }
        for d in 0..hd {
            let mut acc = 0f32;
            for i in 0..total {
                acc += scores[i] * v_all_h[(i * n_kv + kvh) * hd + d];
            }
            ref_out[qh * hd + d] = acc;
        }
    }
    let (o_max, o_mean) = rel_stats(&ref_out, &out_h);
    eprintln!(
        "probe4 decode kernel vs host ref on dequant kv: max_rel={o_max:.5} mean_rel={o_mean:.5}"
    );

    let k1_ref = host_f32(&k1);
    let v1_ref = host_f32(&v1);
    let mut k_cat = k_ref.clone();
    k_cat.extend_from_slice(&k1_ref);
    let mut v_cat = v_ref.clone();
    v_cat.extend_from_slice(&v1_ref);
    let k_exact = quant_ref(&k_cat, total, n_kv, hd);
    let v_exact = quant_ref(&v_cat, total, n_kv, hd);
    let (host_e4m3_max, _) = rel_stats(&k_all_h, &k_exact);
    eprintln!("probe4b host-e4m3 vs gpu-dequant-view k: max_rel={host_e4m3_max:.5} (bf16 rounding bound ~0.004)");
    let mut ref2 = vec![0f32; n_q * hd];
    for qh in 0..n_q {
        let kvh = qh / group;
        let mut scores = vec![0f32; total];
        for i in 0..total {
            let mut s = 0f32;
            for d in 0..hd {
                s += q_h[qh * hd + d] * k_exact[(i * n_kv + kvh) * hd + d];
            }
            scores[i] = s * scale;
        }
        let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut den = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            den += *s;
        }
        for s in scores.iter_mut() {
            *s /= den;
        }
        for d in 0..hd {
            let mut acc = 0f32;
            for i in 0..total {
                acc += scores[i] * v_exact[(i * n_kv + kvh) * hd + d];
            }
            ref2[qh * hd + d] = acc;
        }
    }
    let (e_max, e_mean) = rel_stats(&ref2, &out_h);
    eprintln!(
        "probe4b decode kernel vs EXACT e4m3 host ref: max_rel={e_max:.5} mean_rel={e_mean:.5}"
    );

    assert!(k_max < 0.08, "probe1 k round-trip too lossy: {k_max}");
    assert!(v_max < 0.08, "probe1 v round-trip too lossy: {v_max}");
    assert!(
        k_chunk_eq && v_chunk_eq,
        "probe2 chunked writes differ from single-shot"
    );
    assert!(
        rel_expect < 0.08,
        "probe3 offset-view write corrupted (rel-to-correct {rel_expect}, rel-to-aliased {rel_alias})"
    );
    let _ = o_max;
    assert!(
        host_e4m3_max < 0.006,
        "probe4b host e4m3 model diverges from gpu dequant: {host_e4m3_max}"
    );
    assert!(
        e_max < 0.01,
        "probe4b decode kernel mismatch vs exact e4m3 ref: {e_max}"
    );
}

#[test]
#[ignore]
fn laguna_fp8_gscores_bitmatch_below_cap() {
    if std::env::var("NV_LAGUNA_TEST").is_err() {
        eprintln!("set NV_LAGUNA_TEST=1 to run");
        return;
    }
    let dir = model_dir().expect("Laguna snapshot not found");
    let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let config = LagunaConfig::from_hf_json_str(&raw).unwrap();
    let device = Device::new_cuda(0).unwrap();
    let n_kv = config.num_key_value_heads;
    let hd = config.head_dim;
    let full_idx = config
        .layer_types
        .iter()
        .position(|k| matches!(k, LayerType::FullAttention))
        .unwrap();
    let n_q = config.num_heads_for_layer(full_idx);
    let fp8_cap = LagunaKvCacheFp8::max_seq_len_for_fp8_decode(hd);
    eprintln!("full layer {full_idx}: n_q={n_q} n_kv={n_kv} hd={hd} fp8_cap={fp8_cap}");

    let t = 511usize;
    let total = t + 1;
    let k_host = rand_kv(21, t, n_kv, hd);
    let v_host = rand_kv(22, t, n_kv, hd);
    let kq_host = rand_kv(23, 1, n_kv, hd);
    let vq_host = rand_kv(24, 1, n_kv, hd);
    let q_host = rand_kv(25, 1, n_q, hd);
    let k = to_bf16_tensor(&k_host, (1, t, n_kv, hd), &device);
    let v = to_bf16_tensor(&v_host, (1, t, n_kv, hd), &device);
    let k1 = to_bf16_tensor(&kq_host, (1, 1, n_kv, hd), &device);
    let v1 = to_bf16_tensor(&vq_host, (1, 1, n_kv, hd), &device);
    let q = to_bf16_tensor(&q_host, (1, 1, n_q, hd), &device);
    let scale = (hd as f32).powf(-0.5);

    let mut run = |max_seq: usize, expect_scratch: bool| -> Vec<f32> {
        let mut cache = LagunaKvCacheFp8::new(&config, max_seq, &device, DType::BF16).unwrap();
        assert_eq!(
            cache.uses_score_scratch(),
            expect_scratch,
            "max_seq {max_seq} scratch dispatch"
        );
        cache.prepare_for_decode_dev(0, t).unwrap();
        cache.write_at_impl(full_idx, &k, &v).unwrap();
        cache.advance(t);
        cache.prepare_for_decode_dev(t, total).unwrap();
        cache.write_at_impl(full_idx, &k1, &v1).unwrap();
        let out = cache
            .decode_attention_fp8(full_idx, &q, n_q, None, scale)
            .unwrap()
            .expect("fp8 decode path");
        host_f32(&out)
    };

    let out_smem = run(fp8_cap.min(4096), false);
    let out_gmem = run(fp8_cap + 64, true);
    let n_eq = out_smem
        .iter()
        .zip(out_gmem.iter())
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    eprintln!(
        "gscores-vs-smem bitmatch at n_total={total}: {n_eq}/{} equal",
        out_smem.len()
    );
    assert_eq!(
        n_eq,
        out_smem.len(),
        "gscores kernel must bit-match the smem kernel below the cap"
    );

    let sw = config
        .sliding_window
        .max(1)
        .min(total.saturating_sub(2))
        .max(1);
    let mut run_sw = |max_seq: usize, expect_scratch: bool| -> Vec<f32> {
        let mut cache = LagunaKvCacheFp8::new(&config, max_seq, &device, DType::BF16).unwrap();
        assert_eq!(cache.uses_score_scratch(), expect_scratch);
        cache.prepare_for_decode_dev(0, t).unwrap();
        cache.write_at_impl(full_idx, &k, &v).unwrap();
        cache.advance(t);
        cache.prepare_for_decode_dev(t, total).unwrap();
        cache.write_at_impl(full_idx, &k1, &v1).unwrap();
        let out = cache
            .decode_attention_fp8(full_idx, &q, n_q, Some(sw), scale)
            .unwrap()
            .expect("fp8 decode path");
        host_f32(&out)
    };
    let out_smem_sw = run_sw(fp8_cap.min(4096), false);
    let out_gmem_sw = run_sw(fp8_cap + 64, true);
    let n_eq_sw = out_smem_sw
        .iter()
        .zip(out_gmem_sw.iter())
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    eprintln!(
        "gscores-vs-smem bitmatch (sliding_window={sw}): {n_eq_sw}/{} equal",
        out_smem_sw.len()
    );
    assert_eq!(n_eq_sw, out_smem_sw.len(), "masked path must bit-match too");
}

#[test]
#[ignore]
fn laguna_fp8_gscores_over_cap() {
    if std::env::var("NV_LAGUNA_TEST").is_err() {
        eprintln!("set NV_LAGUNA_TEST=1 to run");
        return;
    }
    let dir = model_dir().expect("Laguna snapshot not found");
    let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let config = LagunaConfig::from_hf_json_str(&raw).unwrap();
    let device = Device::new_cuda(0).unwrap();
    let n_kv = config.num_key_value_heads;
    let hd = config.head_dim;
    let full_idx = config
        .layer_types
        .iter()
        .position(|k| matches!(k, LayerType::FullAttention))
        .unwrap();
    let n_q = config.num_heads_for_layer(full_idx);
    let fp8_cap = LagunaKvCacheFp8::max_seq_len_for_fp8_decode(hd);
    let max_seq = fp8_cap + 1024;
    let t = fp8_cap + 500;
    let total = t + 1;
    eprintln!("over-cap probe: fp8_cap={fp8_cap} max_seq={max_seq} n_total={total}");

    let k_host = rand_kv(31, t, n_kv, hd);
    let v_host = rand_kv(32, t, n_kv, hd);
    let kq_host = rand_kv(33, 1, n_kv, hd);
    let vq_host = rand_kv(34, 1, n_kv, hd);
    let q_host = rand_kv(35, 1, n_q, hd);
    let k = to_bf16_tensor(&k_host, (1, t, n_kv, hd), &device);
    let v = to_bf16_tensor(&v_host, (1, t, n_kv, hd), &device);
    let k1 = to_bf16_tensor(&kq_host, (1, 1, n_kv, hd), &device);
    let v1 = to_bf16_tensor(&vq_host, (1, 1, n_kv, hd), &device);
    let q = to_bf16_tensor(&q_host, (1, 1, n_q, hd), &device);
    let q_h = host_f32(&q);
    let scale = (hd as f32).powf(-0.5);

    let fill_and_decode = |max_seq: usize, expect_scratch: bool, t_fill: usize| {
        let mut cache = LagunaKvCacheFp8::new(&config, max_seq, &device, DType::BF16).unwrap();
        assert_eq!(cache.uses_score_scratch(), expect_scratch);
        let chunk = 256usize;
        let mut off = 0usize;
        while off < t_fill {
            let n = chunk.min(t_fill - off);
            let kc = k.narrow(1, off, n).unwrap();
            let vc = v.narrow(1, off, n).unwrap();
            cache.prepare_for_decode_dev(off, off + n).unwrap();
            cache.write_at_impl(full_idx, &kc, &vc).unwrap();
            cache.advance(n);
            off += n;
        }
        cache.prepare_for_decode_dev(t_fill, t_fill + 1).unwrap();
        cache.write_at_impl(full_idx, &k1, &v1).unwrap();
        let out = cache
            .decode_attention_fp8(full_idx, &q, n_q, None, scale)
            .unwrap()
            .expect("fp8 decode path");
        (cache, host_f32(&out))
    };

    let t_cap = fp8_cap - 1;
    let (_c_smem, out_cap_smem) = fill_and_decode(fp8_cap, false, t_cap);
    let (_c_gmem, out_cap_gmem) = fill_and_decode(max_seq, true, t_cap);
    let n_eq_cap = out_cap_smem
        .iter()
        .zip(out_cap_gmem.iter())
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    eprintln!(
        "at-cap bitmatch (n_total={fp8_cap}, differing max_total strides): {n_eq_cap}/{} equal",
        out_cap_smem.len()
    );
    assert_eq!(
        n_eq_cap,
        out_cap_smem.len(),
        "gscores kernel must bit-match the smem kernel at the cap boundary"
    );

    let (mut cache, out_h) = fill_and_decode(max_seq, true, t);
    let out2 = cache
        .decode_attention_fp8(full_idx, &q, n_q, None, scale)
        .unwrap()
        .expect("fp8 decode path");
    let out2_h = host_f32(&out2);
    let repeat_eq = out_h
        .iter()
        .zip(out2_h.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    eprintln!("over-cap repeat determinism: {repeat_eq}");

    let (k_all, v_all) = cache.view_bf16(full_idx, total).unwrap();
    let k_all_h = host_f32(&k_all);
    let v_all_h = host_f32(&v_all);
    let group = n_q / n_kv;
    let mut ref_out = vec![0f32; n_q * hd];
    for qh in 0..n_q {
        let kvh = qh / group;
        let mut scores = vec![0f32; total];
        for i in 0..total {
            let mut s = 0f32;
            for d in 0..hd {
                s += q_h[qh * hd + d] * k_all_h[(i * n_kv + kvh) * hd + d];
            }
            scores[i] = s * scale;
        }
        let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut den = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            den += *s;
        }
        for s in scores.iter_mut() {
            *s /= den;
        }
        for d in 0..hd {
            let mut acc = 0f32;
            for i in 0..total {
                acc += scores[i] * v_all_h[(i * n_kv + kvh) * hd + d];
            }
            ref_out[qh * hd + d] = acc;
        }
    }
    let (o_max, o_mean) = rel_stats(&ref_out, &out_h);
    eprintln!("over-cap decode vs host ref on dequant kv: max_rel={o_max:.5} mean_rel={o_mean:.5}");
    assert!(repeat_eq, "over-cap decode must be bitwise deterministic");
    assert!(
        o_max < 0.08 && o_mean < 0.005,
        "over-cap decode out of e4m3-class bounds vs dequant-kv host ref: max {o_max} mean {o_mean}"
    );
}

#[test]
#[ignore]
fn laguna_fp8_gscores_65k() {
    if std::env::var("NV_LAGUNA_TEST").is_err() {
        eprintln!("set NV_LAGUNA_TEST=1 to run");
        return;
    }
    let dir = model_dir().expect("Laguna snapshot not found");
    let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let config = LagunaConfig::from_hf_json_str(&raw).unwrap();
    let device = Device::new_cuda(0).unwrap();
    let n_kv = config.num_key_value_heads;
    let hd = config.head_dim;
    let full_idx = config
        .layer_types
        .iter()
        .position(|k| matches!(k, LayerType::FullAttention))
        .unwrap();
    let n_q = config.num_heads_for_layer(full_idx);
    let t = 65_548usize.min(config.max_position_embeddings.saturating_sub(2));
    let total = t + 1;
    let max_seq = total + 512;
    eprintln!("65k probe: n_total={total} max_seq={max_seq}");

    let k_host = rand_kv(41, t, n_kv, hd);
    let v_host = rand_kv(42, t, n_kv, hd);
    let kq_host = rand_kv(43, 1, n_kv, hd);
    let vq_host = rand_kv(44, 1, n_kv, hd);
    let q_host = rand_kv(45, 1, n_q, hd);
    let k = to_bf16_tensor(&k_host, (1, t, n_kv, hd), &device);
    let v = to_bf16_tensor(&v_host, (1, t, n_kv, hd), &device);
    let k1 = to_bf16_tensor(&kq_host, (1, 1, n_kv, hd), &device);
    let v1 = to_bf16_tensor(&vq_host, (1, 1, n_kv, hd), &device);
    let q = to_bf16_tensor(&q_host, (1, 1, n_q, hd), &device);
    let q_h = host_f32(&q);
    let scale = (hd as f32).powf(-0.5);

    let mut cache = LagunaKvCacheFp8::new(&config, max_seq, &device, DType::BF16).unwrap();
    assert!(cache.uses_score_scratch());
    let chunk = 256usize;
    let mut off = 0usize;
    while off < t {
        let n = chunk.min(t - off);
        let kc = k.narrow(1, off, n).unwrap();
        let vc = v.narrow(1, off, n).unwrap();
        cache.prepare_for_decode_dev(off, off + n).unwrap();
        cache.write_at_impl(full_idx, &kc, &vc).unwrap();
        cache.advance(n);
        off += n;
    }
    cache.prepare_for_decode_dev(t, total).unwrap();
    cache.write_at_impl(full_idx, &k1, &v1).unwrap();
    let out = cache
        .decode_attention_fp8(full_idx, &q, n_q, None, scale)
        .unwrap()
        .expect("fp8 decode path");
    let out_h = host_f32(&out);
    let out2 = cache
        .decode_attention_fp8(full_idx, &q, n_q, None, scale)
        .unwrap()
        .expect("fp8 decode path");
    let out2_h = host_f32(&out2);
    let repeat_eq = out_h
        .iter()
        .zip(out2_h.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let finite = out_h.iter().all(|x| x.is_finite());
    eprintln!("65k repeat determinism: {repeat_eq}, all finite: {finite}");

    let (k_all, v_all) = cache.view_bf16(full_idx, total).unwrap();
    let k_all_h = host_f32(&k_all);
    let v_all_h = host_f32(&v_all);
    let group = n_q / n_kv;
    let mut ref_out = vec![0f32; n_q * hd];
    for qh in 0..n_q {
        let kvh = qh / group;
        let mut scores = vec![0f32; total];
        for i in 0..total {
            let mut s = 0f32;
            for d in 0..hd {
                s += q_h[qh * hd + d] * k_all_h[(i * n_kv + kvh) * hd + d];
            }
            scores[i] = s * scale;
        }
        let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut den = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            den += *s;
        }
        for s in scores.iter_mut() {
            *s /= den;
        }
        for d in 0..hd {
            let mut acc = 0f32;
            for i in 0..total {
                acc += scores[i] * v_all_h[(i * n_kv + kvh) * hd + d];
            }
            ref_out[qh * hd + d] = acc;
        }
    }
    let (o_max, o_mean) = rel_stats(&ref_out, &out_h);
    eprintln!("65k decode vs host ref on dequant kv: max_rel={o_max:.5} mean_rel={o_mean:.5}");
    assert!(repeat_eq, "65k decode must be bitwise deterministic");
    assert!(finite, "65k decode produced non-finite outputs");
    assert!(
        o_max < 0.2 && o_mean < 0.01,
        "65k decode out of sanity bounds vs dequant-kv host ref: max {o_max} mean {o_mean} \
         (softmax-cancellation-dominated bound; tight evidence is the at-cap bit-match)"
    );
}
