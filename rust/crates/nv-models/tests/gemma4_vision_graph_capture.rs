#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::gemma4_vision::{Gemma4VisionConfig, Gemma4VisionTower, VisionRopeParams};
use nv_models::gemma4_vision_graph::Gemma4VisionGraph;
use std::time::Instant;

const TEXT_HIDDEN: usize = 64;

fn synth_cfg() -> Gemma4VisionConfig {
    Gemma4VisionConfig {
        model_type: Some("gemma4_vision".into()),
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 64,
        patch_size: 16,
        pooling_kernel_size: 2,
        position_embedding_size: 64,
        default_output_length: 64,
        rms_norm_eps: 1e-6,
        attention_bias: false,
        hidden_activation: "gelu_pytorch_tanh".into(),
        use_clipped_linears: false,
        standardize: false,
        rope_parameters: VisionRopeParams {
            rope_theta: 100.0,
            rope_type: None,
        },
        vision_soft_tokens_per_image: None,
    }
}

fn grid_pixels(grid_w: usize, grid_h: usize, pp: usize, seed: u64) -> Vec<f32> {
    let n = grid_w * grid_h * pp;
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push(((s >> 40) & 0xFFFF) as f32 / 65535.0);
    }
    v
}

fn grid_pos_tensor(grid_w: usize, grid_h: usize, device: &Device) -> Tensor {
    let mut v = Vec::with_capacity(grid_w * grid_h * 2);
    for y in 0..grid_h as i64 {
        for x in 0..grid_w as i64 {
            v.push(x);
            v.push(y);
        }
    }
    Tensor::from_vec(v, (grid_w * grid_h, 2), device).unwrap()
}

fn reversed_pos_tensor_forces_the_irregular_legacy_route(
    grid_w: usize,
    grid_h: usize,
    device: &Device,
) -> (Tensor, Vec<usize>) {
    let n = grid_w * grid_h;
    let order: Vec<usize> = (0..n).rev().collect();
    let mut v = Vec::with_capacity(n * 2);
    for &i in &order {
        v.push((i % grid_w) as i64);
        v.push((i / grid_w) as i64);
    }
    (Tensor::from_vec(v, (n, 2), device).unwrap(), order)
}

fn max_rel_diff(a: &Tensor, b: &Tensor) -> f32 {
    let av = a
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let bv = b
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(av.len(), bv.len(), "shape mismatch between eager and graph");
    let mut m = 0f32;
    for (x, y) in av.iter().zip(bv.iter()) {
        assert!(x.is_finite() && y.is_finite(), "non-finite embedding value");
        m = m.max((x - y).abs() / (x.abs().max(y.abs()).max(1e-3)));
    }
    m
}

#[test]
fn vision_graph_matches_eager_per_bucket_and_replays() {
    let Ok(device) = Device::new_cuda_with_stream(0) else {
        eprintln!("SKIP vision_graph_matches_eager_per_bucket_and_replays: no CUDA device");
        return;
    };
    let cfg = synth_cfg();
    let pp = cfg.patch_pixels();
    let tower = Gemma4VisionTower::new_synthetic(cfg, TEXT_HIDDEN, &device, DType::BF16)
        .expect("synthetic tower");
    let graph = Gemma4VisionGraph::new(&device).expect("vision graph on with-stream device");

    for (gw, gh) in [(8usize, 8usize), (12usize, 8usize)] {
        let pixels = grid_pixels(gw, gh, pp, (gw * 100 + gh) as u64);
        let pv = Tensor::from_slice(&pixels, (gw * gh, pp), &device).unwrap();
        let pos = grid_pos_tensor(gw, gh, &device);
        let eager = tower.forward(&pv, &pos).expect("eager full-grid forward");
        let first = graph
            .forward(&tower, &pixels, gw, gh)
            .expect("graphed forward (capture)");
        let m = max_rel_diff(&eager, &first);
        assert!(
            m < 2e-2,
            "bucket {gw}x{gh}: capture-pass output drifted rel {m} from eager"
        );

        let pixels2 = grid_pixels(gw, gh, pp, (gw * 7 + gh * 13) as u64);
        let pv2 = Tensor::from_slice(&pixels2, (gw * gh, pp), &device).unwrap();
        let eager2 = tower.forward(&pv2, &pos).expect("eager forward 2");
        let replay = graph
            .forward(&tower, &pixels2, gw, gh)
            .expect("graphed forward (replay)");
        let m2 = max_rel_diff(&eager2, &replay);
        assert!(
            m2 < 2e-2,
            "bucket {gw}x{gh}: replay with fresh pixels drifted rel {m2} from eager -- the \
             graph is reading stale input"
        );
    }
    assert!(
        graph.capture_active(),
        "two buckets ran but capture_active is false"
    );
    assert!(
        graph.captures() >= 2 && graph.replays() >= 2,
        "expected >=2 captures and >=2 replays, got {} / {}",
        graph.captures(),
        graph.replays()
    );
}

#[test]
fn vision_graph_replay_beats_eager_wall_on_a_repeated_bucket() {
    let Ok(device) = Device::new_cuda_with_stream(0) else {
        eprintln!("SKIP vision_graph_replay_beats_eager_wall_on_a_repeated_bucket: no CUDA");
        return;
    };
    let cfg = synth_cfg();
    let pp = cfg.patch_pixels();
    let tower = Gemma4VisionTower::new_synthetic(cfg, TEXT_HIDDEN, &device, DType::BF16)
        .expect("synthetic tower");
    let graph = Gemma4VisionGraph::new(&device).expect("vision graph");
    let (gw, gh) = (16usize, 16usize);
    let pixels = grid_pixels(gw, gh, pp, 42);
    let pv = Tensor::from_slice(&pixels, (gw * gh, pp), &device).unwrap();
    let pos = grid_pos_tensor(gw, gh, &device);

    for _ in 0..3 {
        let _ = tower.forward(&pv, &pos).unwrap();
        let _ = graph.forward(&tower, &pixels, gw, gh).unwrap();
    }
    device.synchronize().unwrap();

    let (rev_pos, order) = reversed_pos_tensor_forces_the_irregular_legacy_route(gw, gh, &device);
    let pp_stride = pp;
    let mut rev_pixels = vec![0f32; pixels.len()];
    for (slot, &i) in order.iter().enumerate() {
        rev_pixels[slot * pp_stride..(slot + 1) * pp_stride]
            .copy_from_slice(&pixels[i * pp_stride..(i + 1) * pp_stride]);
    }
    let rev_pv = Tensor::from_slice(&rev_pixels, (gw * gh, pp), &device).unwrap();
    let _ = tower.forward(&rev_pv, &rev_pos).unwrap();
    device.synchronize().unwrap();

    let iters = 20;
    let tl = Instant::now();
    for _ in 0..iters {
        let _ = tower.forward(&rev_pv, &rev_pos).unwrap();
    }
    device.synchronize().unwrap();
    let legacy_ms = tl.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = tower.forward(&pv, &pos).unwrap();
    }
    device.synchronize().unwrap();
    let eager_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = graph.forward(&tower, &pixels, gw, gh).unwrap();
    }
    device.synchronize().unwrap();
    let graph_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;

    eprintln!(
        "[vision_graph timing] {gw}x{gh} grid ({} patches): irregular-route \
         {legacy_ms:.3} ms/image, full-grid eager {eager_ms:.3} ms/image, graph \
         {graph_ms:.3} ms/image, graph-vs-eager {:.2}x, capture_active={}",
        gw * gh,
        eager_ms / graph_ms,
        graph.capture_active()
    );
    assert!(graph.capture_active(), "timing ran uncaptured");
    assert!(
        graph_ms < eager_ms,
        "graph replay ({graph_ms:.3} ms) must not be slower than eager ({eager_ms:.3} ms) on a \
         launch-bound tower"
    );
}

fn rel_l2(a: &Tensor, b: &Tensor) -> f32 {
    let av = a
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let bv = b
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(av.len(), bv.len());
    let mut num = 0f64;
    let mut den = 0f64;
    for (x, y) in av.iter().zip(bv.iter()) {
        num += ((x - y) * (x - y)) as f64;
        den += (x * x) as f64;
    }
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

#[test]
#[ignore]
fn capture_bisect_by_layer_count() {
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let layers: usize = std::env::var("NV_VG_BISECT_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut cfg = synth_cfg();
    cfg.num_hidden_layers = layers;
    let pp = cfg.patch_pixels();
    let tower = Gemma4VisionTower::new_synthetic(cfg, TEXT_HIDDEN, &device, DType::BF16)
        .expect("synthetic tower");
    let graph = Gemma4VisionGraph::new(&device).expect("vision graph");
    let (gw, gh) = (8usize, 8usize);
    let pixels = grid_pixels(gw, gh, pp, 3);
    let pv = Tensor::from_slice(&pixels, (gw * gh, pp), &device).unwrap();
    let pos = grid_pos_tensor(gw, gh, &device);
    let eager = tower.forward(&pv, &pos).expect("eager");
    let graphed = graph
        .forward(&tower, &pixels, gw, gh)
        .expect("graphed capture");
    let m = max_rel_diff(&eager, &graphed);
    eprintln!("[vg_bisect] layers={layers} capture_ok max_rel_diff={m}");
    let graphed2 = graph
        .forward(&tower, &pixels, gw, gh)
        .expect("graphed replay");
    let m2 = max_rel_diff(&eager, &graphed2);
    eprintln!(
        "[vg_bisect] layers={layers} replay_ok max_rel_diff={m2} capture_active={}",
        graph.capture_active()
    );
}

fn e4b_snapshot_dir() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snaps = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots");
    for e in std::fs::read_dir(&snaps).ok()?.flatten() {
        let p = e.path();
        if p.join("config.json").is_file() {
            return Some(p);
        }
    }
    None
}

#[test]
#[ignore]
fn e4b_real_tower_graph_parity_and_encode_wall() {
    if std::env::var("NV_GEMMA4_VISION_REAL_TEST").ok().as_deref() != Some("1") {
        panic!(
            "e4b_real_tower_graph_parity_and_encode_wall is #[ignore]d and was asked for by \
             name, but NV_GEMMA4_VISION_REAL_TEST=1 is not set. This is a SKIP, not a pass."
        );
    }
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let dir = e4b_snapshot_dir().expect("E4B hub snapshot not cached");
    let cfg = Gemma4VisionConfig::from_hf_json_file(&dir.join("config.json")).expect("vision cfg");
    let pp = cfg.patch_pixels();
    let pk = cfg.pooling_kernel_size;
    let weights =
        nv_weights::WeightLoader::open_dir(&dir, &device).expect("open E4B weights");
    let tower =
        Gemma4VisionTower::load(cfg, &weights, &device, DType::BF16).expect("load E4B tower");
    let graph = Gemma4VisionGraph::new(&device).expect("vision graph");

    for (side_units, label) in [(8usize, "512px-class"), (16usize, "1024px-class")] {
        let gw = side_units * pk;
        let gh = side_units * pk;
        let pixels = grid_pixels(gw, gh, pp, side_units as u64);
        let pv = Tensor::from_slice(&pixels, (gw * gh, pp), &device).unwrap();
        let pos = grid_pos_tensor(gw, gh, &device);

        let eager = tower.forward(&pv, &pos).expect("eager E4B forward");
        let graphed = graph
            .forward(&tower, &pixels, gw, gh)
            .expect("graphed E4B forward");
        let l2 = rel_l2(&eager, &graphed);
        let m = max_rel_diff(&eager, &graphed);
        eprintln!("[e4b_vision parity] {label}: rel_l2={l2} max_elem_rel={m}");
        assert!(
            l2 < 2e-2,
            "{label}: graphed E4B tower drifted rel_l2 {l2} (max elem rel {m}) from eager"
        );

        let (rev_pos, order) =
            reversed_pos_tensor_forces_the_irregular_legacy_route(gw, gh, &device);
        let mut rev_pixels = vec![0f32; pixels.len()];
        for (slot, &i) in order.iter().enumerate() {
            rev_pixels[slot * pp..(slot + 1) * pp].copy_from_slice(&pixels[i * pp..(i + 1) * pp]);
        }
        let rev_pv = Tensor::from_slice(&rev_pixels, (gw * gh, pp), &device).unwrap();

        for _ in 0..2 {
            let _ = tower.forward(&pv, &pos).unwrap();
            let _ = graph.forward(&tower, &pixels, gw, gh).unwrap();
            let _ = tower.forward(&rev_pv, &rev_pos).unwrap();
        }
        device.synchronize().unwrap();
        let iters = 10;
        let tl = Instant::now();
        for _ in 0..iters {
            let _ = tower.forward(&rev_pv, &rev_pos).unwrap();
        }
        device.synchronize().unwrap();
        let legacy_ms = tl.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = tower.forward(&pv, &pos).unwrap();
        }
        device.synchronize().unwrap();
        let eager_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let t1 = Instant::now();
        for _ in 0..iters {
            let _ = graph.forward(&tower, &pixels, gw, gh).unwrap();
        }
        device.synchronize().unwrap();
        let graph_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        eprintln!(
            "[e4b_vision timing] {label} grid {gw}x{gh} ({} patches, {} soft tokens): \
             irregular-route {legacy_ms:.3} ms/image, full-grid eager {eager_ms:.3} ms/image, \
             graph {graph_ms:.3} ms/image, graph-vs-eager {:.2}x, capture_active={}",
            gw * gh,
            (gw / pk) * (gh / pk),
            eager_ms / graph_ms,
            graph.capture_active()
        );
    }
    assert!(graph.capture_active(), "E4B run never captured");
}

#[test]
fn bucket_byte_budget_evicts_the_largest_bucket_recaptures_on_return_and_refuses_oversize() {
    let Ok(device) = Device::new_cuda_with_stream(0) else {
        eprintln!("SKIP bucket_byte_budget test: no CUDA device");
        return;
    };
    let cfg = synth_cfg();
    let pp = cfg.patch_pixels();
    let tower = Gemma4VisionTower::new_synthetic(cfg, TEXT_HIDDEN, &device, DType::BF16)
        .expect("synthetic tower");

    let small = (8usize, 8usize);
    let mid = (10usize, 10usize);
    let large = (12usize, 12usize);
    let eager_for = |(gw, gh): (usize, usize)| {
        let pixels = grid_pixels(gw, gh, pp, (gw * 31 + gh) as u64);
        let pv = Tensor::from_slice(&pixels, (gw * gh, pp), &device).unwrap();
        let pos = grid_pos_tensor(gw, gh, &device);
        let eager = tower.forward(&pv, &pos).expect("eager forward");
        (pixels, eager)
    };

    let probe = Gemma4VisionGraph::with_bucket_byte_budget(&device, usize::MAX)
        .expect("probe graph with an unbounded budget");
    let mut grid_bytes = std::collections::HashMap::new();
    let mut prev = 0usize;
    for g in [small, mid, large] {
        let (pixels, _) = eager_for(g);
        probe
            .forward(&tower, &pixels, g.0, g.1)
            .expect("probe capture");
        let now = probe.resident_bucket_bytes();
        assert!(now > prev, "each admitted bucket must account resident bytes");
        grid_bytes.insert(g, now - prev);
        prev = now;
    }
    assert_eq!(probe.bucket_count(), 3);
    drop(probe);

    let (b_small, b_mid, b_large) = (grid_bytes[&small], grid_bytes[&mid], grid_bytes[&large]);
    assert!(
        b_small < b_mid && b_mid < b_large,
        "resident-byte accounting must scale with resolution ({b_small} / {b_mid} / {b_large})"
    );

    let budget = b_small + b_mid + b_large - 1;
    let graph = Gemma4VisionGraph::with_bucket_byte_budget(&device, budget)
        .expect("budgeted vision graph");
    for g in [small, mid] {
        let (pixels, eager) = eager_for(g);
        let out = graph.forward(&tower, &pixels, g.0, g.1).expect("capture");
        assert!(max_rel_diff(&eager, &out) < 2e-2);
    }
    assert_eq!(graph.bucket_count(), 2);
    assert_eq!(graph.captures(), 2);

    let (pixels_l, eager_l) = eager_for(large);
    let out_l = graph
        .forward(&tower, &pixels_l, large.0, large.1)
        .expect("admitting the large grid must evict, not fail");
    assert!(max_rel_diff(&eager_l, &out_l) < 2e-2);
    assert_eq!(
        graph.bucket_count(),
        2,
        "admitting {large:?} over the byte budget must evict exactly the largest resident \
         bucket ({mid:?}), leaving small+large"
    );
    assert_eq!(graph.captures(), 3);
    assert!(
        graph.resident_bucket_bytes() <= budget,
        "resident bytes {} exceed the budget {budget} after eviction",
        graph.resident_bucket_bytes()
    );

    let (pixels_m, eager_m) = eager_for(mid);
    let out_m = graph
        .forward(&tower, &pixels_m, mid.0, mid.1)
        .expect("the evicted grid must be re-admitted on its next miss");
    assert!(max_rel_diff(&eager_m, &out_m) < 2e-2);
    assert_eq!(
        graph.captures(),
        4,
        "an evicted grid returns via a fresh capture -- eviction is correct, just slow"
    );
    assert!(graph.resident_bucket_bytes() <= budget);

    let tiny = Gemma4VisionGraph::with_bucket_byte_budget(&device, b_small - 1)
        .expect("tiny-budget vision graph");
    let (pixels_s, eager_s) = eager_for(small);
    for round in 0..2 {
        let out = tiny
            .forward(&tower, &pixels_s, small.0, small.1)
            .expect("an oversize grid must fall back eager, not fail");
        assert!(max_rel_diff(&eager_s, &out) < 2e-2, "round {round}");
    }
    assert_eq!(
        tiny.bucket_count(),
        0,
        "a grid whose bucket alone exceeds the budget must never be admitted"
    );
    assert_eq!(tiny.captures(), 0);
    eprintln!(
        "[vision-budget] bytes small={b_small} mid={b_mid} large={b_large} budget={budget}: \
         evict-largest, recapture-on-return, oversize-refusal all hold"
    );
}
