use candle_core::{Device, Tensor};
use nv_layers::attn::{sdpa, sdpa_with_sinks, AttnConfig};

fn det(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.3137).sin() * 1.3)
        .collect()
}

fn host_sink_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sinks: &[f32],
    sq: usize,
    sk: usize,
    h: usize,
    h_kv: usize,
    d: usize,
    scale: f32,
    window: usize,
) -> Vec<f32> {
    let group = h / h_kv;
    let off = sk - sq;
    let mut out = vec![0f32; sq * h * d];
    for row in 0..sq {
        let last = row + off;
        let first = if window > 0 {
            (last + 1).saturating_sub(window)
        } else {
            0
        };
        for head in 0..h {
            let kv = head / group;
            let mut scores = Vec::new();
            let mut m = sinks[head];
            for t in first..=last {
                let mut dot = 0f32;
                for i in 0..d {
                    dot += q[(row * h + head) * d + i] * k[(t * h_kv + kv) * d + i];
                }
                let s = dot * scale;
                m = m.max(s);
                scores.push(s);
            }
            let mut z = (sinks[head] - m).exp();
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                z += *s;
            }
            for i in 0..d {
                let mut acc = 0f32;
                for (j, t) in (first..=last).enumerate() {
                    acc += scores[j] * v[(t * h_kv + kv) * d + i];
                }
                out[(row * h + head) * d + i] = acc / z;
            }
        }
    }
    out
}

fn run_case(sq: usize, sk: usize, h: usize, h_kv: usize, d: usize, window: usize, sink_scale: f32) {
    let device = Device::Cpu;
    let q_host = det(sq * h * d, 1.0);
    let k_host = det(sk * h_kv * d, 40.0);
    let v_host = det(sk * h_kv * d, 90.0);
    let sinks_host: Vec<f32> = (0..h).map(|i| ((i as f32) * 0.77).cos() * sink_scale).collect();
    let scale = 1.0 / (d as f32).sqrt();

    let q = Tensor::from_vec(q_host.clone(), (1, sq, h, d), &device).unwrap();
    let k = Tensor::from_vec(k_host.clone(), (1, sk, h_kv, d), &device).unwrap();
    let v = Tensor::from_vec(v_host.clone(), (1, sk, h_kv, d), &device).unwrap();
    let sinks = Tensor::from_vec(sinks_host.clone(), h, &device).unwrap();
    let cfg = AttnConfig {
        num_heads: h,
        num_kv_heads: h_kv,
        head_dim: d,
        softmax_scale: scale,
        causal: true,
    };

    let got = sdpa_with_sinks(&q, &k, &v, &cfg, &sinks, window).unwrap();
    let got: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
    let want = host_sink_attention(
        &q_host,
        &k_host,
        &v_host,
        &sinks_host,
        sq,
        sk,
        h,
        h_kv,
        d,
        scale,
        window,
    );
    let mut worst = 0f32;
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        let e = (a - b).abs();
        assert!(
            e < 2e-5,
            "sq={sq} sk={sk} window={window} elem {i}: sdpa_with_sinks {a} vs host sink softmax {b}"
        );
        worst = worst.max(e);
    }
    assert!(worst.is_finite());
}

#[test]
fn appending_a_sink_column_reproduces_the_gpt_oss_max_and_denominator_fold() {
    for (sq, sk) in [(1usize, 1usize), (1, 9), (4, 4), (3, 11), (7, 7)] {
        for window in [0usize, 1, 4, 128] {
            run_case(sq, sk, 4, 2, 8, window, 0.9);
        }
    }
}

#[test]
fn a_sink_far_below_every_score_leaves_plain_sdpa_unchanged() {
    let device = Device::Cpu;
    let (sq, sk, h, d) = (5usize, 5usize, 3usize, 8usize);
    let q = Tensor::from_vec(det(sq * h * d, 2.0), (1, sq, h, d), &device).unwrap();
    let k = Tensor::from_vec(det(sk * h * d, 20.0), (1, sk, h, d), &device).unwrap();
    let v = Tensor::from_vec(det(sk * h * d, 200.0), (1, sk, h, d), &device).unwrap();
    let sinks = Tensor::from_vec(vec![-60f32; h], h, &device).unwrap();
    let cfg = AttnConfig {
        num_heads: h,
        num_kv_heads: h,
        head_dim: d,
        softmax_scale: 1.0 / (d as f32).sqrt(),
        causal: true,
    };
    let plain: Vec<f32> = sdpa(&q, &k, &v, &cfg)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let sunk: Vec<f32> = sdpa_with_sinks(&q, &k, &v, &cfg, &sinks, 0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for (i, (a, b)) in plain.iter().zip(sunk.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "elem {i}: a sink at -60 must contribute nothing, sdpa {a} vs sdpa_with_sinks {b}"
        );
    }
}

#[test]
fn a_sink_far_above_every_score_drives_every_real_key_probability_to_zero() {
    let device = Device::Cpu;
    let (sq, sk, h, d) = (3usize, 6usize, 2usize, 8usize);
    let q = Tensor::from_vec(det(sq * h * d, 5.0), (1, sq, h, d), &device).unwrap();
    let k = Tensor::from_vec(det(sk * h * d, 60.0), (1, sk, h, d), &device).unwrap();
    let v = Tensor::from_vec(vec![1f32; sk * h * d], (1, sk, h, d), &device).unwrap();
    let sinks = Tensor::from_vec(vec![60f32; h], h, &device).unwrap();
    let cfg = AttnConfig {
        num_heads: h,
        num_kv_heads: h,
        head_dim: d,
        softmax_scale: 1.0 / (d as f32).sqrt(),
        causal: true,
    };
    let out: Vec<f32> = sdpa_with_sinks(&q, &k, &v, &cfg, &sinks, 0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for (i, o) in out.iter().enumerate() {
        assert!(
            o.abs() < 1e-9,
            "elem {i}: an all-ones V under a dominating sink must still return ~0, got {o}"
        );
    }
}

#[test]
fn the_sliding_window_keeps_exactly_window_keys_ending_at_the_query() {
    let device = Device::Cpu;
    let (sq, sk, d) = (1usize, 10usize, 4usize);
    let window = 3usize;
    let q = Tensor::zeros((1, sq, 1, d), candle_core::DType::F32, &device).unwrap();
    let k = Tensor::zeros((1, sk, 1, d), candle_core::DType::F32, &device).unwrap();
    let mut v_host = vec![0f32; sk * d];
    for j in 0..sk {
        v_host[j * d + (j % d)] += 1.0;
    }
    let v = Tensor::from_vec(v_host, (1, sk, 1, d), &device).unwrap();
    let cfg = AttnConfig {
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: d,
        softmax_scale: 1.0,
        causal: true,
    };
    let sinks = Tensor::from_vec(vec![f32::NEG_INFINITY], 1usize, &device).unwrap();
    let out: Vec<f32> = sdpa_with_sinks(&q, &k, &v, &cfg, &sinks, window)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let mut want = vec![0f32; d];
    for j in (sk - window)..sk {
        want[j % d] += 1.0 / window as f32;
    }
    for i in 0..d {
        assert!(
            (out[i] - want[i]).abs() < 1e-6,
            "window {window} at key {} must average keys {}..{}: dim {i} got {} want {}",
            sk - 1,
            sk - window,
            sk,
            out[i],
            want[i]
        );
    }
}
