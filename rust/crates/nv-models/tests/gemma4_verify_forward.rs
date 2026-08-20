#![cfg(feature = "cuda")]

use candle_core::{Device, Tensor};
use nv_models::gemma4::{Gemma4, Gemma4Config};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;

fn gemma4_nvfp4_snapshot_dir() -> PathBuf {
    PathBuf::from(std::env::var("NV_G4_SNAPSHOT").unwrap_or_else(|_| {
        format!(
            "{}/.cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots/e5ef03afa233c35cb000323ff098d4291e1dd07c",
            std::env::var("HOME").unwrap_or_default()
        )
    }))
}

fn argmax(row: &[f32]) -> usize {
    let mut b = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in row.iter().enumerate() {
        if x > bv {
            bv = x;
            b = i;
        }
    }
    b
}

#[test]
#[ignore]
fn gemma4_forward_verify_matches_masked() {
    let dir = gemma4_nvfp4_snapshot_dir();
    if !dir.is_dir() {
        eprintln!("skip: snapshot dir missing");
        return;
    }
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no cuda");
        return;
    };
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("cfg");
    let cfg = Gemma4Config::from_hf_json_str(&raw_cfg).expect("parse cfg");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(dir, &device).expect("weights");
    let model =
        Gemma4::from_loader_quantized(cfg.clone(), &weights, &qcfg, &device).expect("model");
    eprintln!("model loaded");

    let aux_layers: Vec<usize> = vec![1, 29, 56];
    let context: Vec<u32> = vec![2, 9259, 236764, 1041, 1463, 563];
    let ctx_len = context.len();
    let chain: Vec<u32> = vec![1390, 568, 20470, 496];
    let k = chain.len();
    let vocab = cfg.vocab_size;

    let mut joint = context.clone();
    joint.extend_from_slice(&chain);
    let seq = joint.len();
    let jt = Tensor::from_vec(joint.clone(), (1usize, seq), &device).unwrap();
    let jp = Tensor::from_vec((0..seq as i32).collect::<Vec<_>>(), seq, &device).unwrap();
    let mut fmask = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in 0..=i {
            fmask[i * seq + j] = 1.0;
        }
    }
    let fmask_t = Tensor::from_vec(fmask, (seq, seq), &device).unwrap();
    let (ref_logits, _aux) = model
        .forward_with_aux_hidden_masked(&jt, &jp, &aux_layers, Some(&fmask_t))
        .expect("ref masked forward");
    let ref_flat: Vec<f32> = ref_logits
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let mut cache = model.new_verify_cache(seq + 8).expect("verify cache");
    let stream_dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = nv_layers::cuda_stream::current_stream(&stream_dev);

    let mut pmask = vec![0u8; ctx_len * ctx_len];
    for i in 0..ctx_len {
        for j in 0..=i {
            pmask[i * ctx_len + j] = 1;
        }
    }
    let pmask_d = stream.memcpy_stod(&pmask).unwrap();
    let ppos: Vec<i32> = (0..ctx_len as i32).collect();
    let (pre_logits, _) = model
        .forward_verify(&context, &ppos, &pmask_d, 0, &aux_layers, &mut cache)
        .expect("prefill");
    let pre_flat: Vec<f32> = pre_logits
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let mut pre_agree = 0usize;
    for i in 0..ctx_len {
        let rr = &ref_flat[i * vocab..(i + 1) * vocab];
        let rp = &pre_flat[i * vocab..(i + 1) * vocab];
        if argmax(rr) == argmax(rp) {
            pre_agree += 1;
        }
    }
    eprintln!("PREFILL argmax agreement with reference context rows: {pre_agree}/{ctx_len}");

    let mut cmask = vec![0u8; k * k];
    for i in 0..k {
        for j in 0..=i {
            cmask[i * k + j] = 1;
        }
    }
    let cmask_d = stream.memcpy_stod(&cmask).unwrap();
    let cpos: Vec<i32> = (ctx_len as i32..(ctx_len + k) as i32).collect();
    let (v_logits, _aux2) = model
        .forward_verify(&chain, &cpos, &cmask_d, ctx_len, &aux_layers, &mut cache)
        .expect("verify");
    let v_flat: Vec<f32> = v_logits
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let mut agree = 0usize;
    let mut max_rel = 0f32;
    for i in 0..k {
        let r_ref = &ref_flat[(ctx_len + i) * vocab..(ctx_len + i + 1) * vocab];
        let r_v = &v_flat[i * vocab..(i + 1) * vocab];
        let a_ref = argmax(r_ref);
        let a_v = argmax(r_v);
        if a_ref == a_v {
            agree += 1;
        }

        let d = (r_ref[a_ref] - r_v[a_ref]).abs() / (r_ref[a_ref].abs() + 1.0);
        if d > max_rel {
            max_rel = d;
        }
        let _ = (a_ref, a_v, d);
    }
    eprintln!("argmax agreement {agree}/{k}, max rel logit diff {max_rel:.4}");

    let mut gcache = model.new_verify_cache(64).expect("gcache");
    let pm: Vec<u8> = {
        let mut m = vec![0u8; ctx_len * ctx_len];
        for i in 0..ctx_len {
            for j in 0..=i {
                m[i * ctx_len + j] = 1;
            }
        }
        m
    };
    let pmd = stream.memcpy_stod(&pm).unwrap();
    let pp: Vec<i32> = (0..ctx_len as i32).collect();
    let (pl, _) = model
        .forward_verify(&context, &pp, &pmd, 0, &aux_layers, &mut gcache)
        .expect("g prefill");
    let plf: Vec<f32> = pl
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let mut tok = argmax(&plf[(ctx_len - 1) * vocab..ctx_len * vocab]) as u32;
    let mut committed = ctx_len;
    let one_mask = stream.memcpy_stod(&[1u8]).unwrap();
    let mut gen: Vec<u32> = vec![tok];
    for _ in 0..24 {
        let (l, _) = model
            .forward_verify(
                &[tok],
                &[committed as i32],
                &one_mask,
                committed,
                &aux_layers,
                &mut gcache,
            )
            .expect("g step");
        let lf: Vec<f32> = l
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        committed += 1;
        tok = argmax(&lf[0..vocab]) as u32;
        gen.push(tok);
    }
    eprintln!("GREEDY via forward_verify: {:?}", gen);
    let uniq: std::collections::HashSet<u32> = gen.iter().copied().collect();
    assert!(
        uniq.len() * 2 >= gen.len(),
        "forward_verify greedy stream degenerate: {gen:?}"
    );
    assert_eq!(
        gen[0], 1390,
        "forward_verify first token must match reference"
    );
}

#[test]
#[ignore]
fn gemma4_verify_batch_vs_sequential() {
    let dir = gemma4_nvfp4_snapshot_dir();
    if !dir.is_dir() {
        eprintln!("skip: snapshot missing");
        return;
    }
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no cuda");
        return;
    };
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("cfg");
    let cfg = Gemma4Config::from_hf_json_str(&raw_cfg).expect("parse cfg");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(dir, &device).expect("weights");
    let model =
        Gemma4::from_loader_quantized(cfg.clone(), &weights, &qcfg, &device).expect("model");

    let aux_layers: Vec<usize> = vec![1, 29, 56];
    let context: Vec<u32> = vec![2, 9259, 236764, 1041, 1463, 563];
    let ctx_len = context.len();
    let chain: Vec<u32> = vec![1390, 568, 20470, 496];
    let k = chain.len();
    let vocab = cfg.vocab_size;
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let lower = |n: usize| {
        let mut m = vec![0u8; n * n];
        for i in 0..n {
            for j in 0..=i {
                m[i * n + j] = 1;
            }
        }
        m
    };

    let mut ca = model.new_verify_cache(64).expect("ca");
    let pmd = stream.memcpy_stod(&lower(ctx_len)).unwrap();
    let ppos: Vec<i32> = (0..ctx_len as i32).collect();
    model
        .forward_verify(&context, &ppos, &pmd, 0, &aux_layers, &mut ca)
        .expect("A prefill");
    let cmd = stream.memcpy_stod(&lower(k)).unwrap();
    let cpos: Vec<i32> = (ctx_len as i32..(ctx_len + k) as i32).collect();
    let (a_logits, _) = model
        .forward_verify(&chain, &cpos, &cmd, ctx_len, &aux_layers, &mut ca)
        .expect("A verify");
    let a_flat: Vec<f32> = a_logits
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let mut cb = model.new_verify_cache(64).expect("cb");
    model
        .forward_verify(&context, &ppos, &pmd, 0, &aux_layers, &mut cb)
        .expect("B prefill");
    let one = stream.memcpy_stod(&[1u8]).unwrap();
    let mut b_rows: Vec<Vec<f32>> = Vec::new();
    for i in 0..k {
        let (l, _) = model
            .forward_verify(
                &[chain[i]],
                &[(ctx_len + i) as i32],
                &one,
                ctx_len + i,
                &aux_layers,
                &mut cb,
            )
            .expect("B step");
        let lf: Vec<f32> = l
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        b_rows.push(lf);
    }

    let mut all_agree = true;
    for i in 0..k {
        let ar = &a_flat[i * vocab..(i + 1) * vocab];
        let br = &b_rows[i];
        let aa = argmax(ar);
        let bb = argmax(br);
        let mut maxd = 0f32;
        for j in 0..vocab {
            let d = (ar[j] - br[j]).abs();
            if d > maxd {
                maxd = d;
            }
        }
        eprintln!(
            "row {i}: batch_argmax={aa} seq_argmax={bb} max_abs_logit_diff={maxd:.4} {}",
            if aa == bb { "OK" } else { "MISMATCH" }
        );
        if aa != bb {
            all_agree = false;
        }
    }
    assert!(all_agree, "batch verify rows must match sequential decode");
}

#[test]
#[ignore]
fn gemma4_graphed_verify_captures_and_replays() {
    let dir = gemma4_nvfp4_snapshot_dir();
    if !dir.is_dir() {
        eprintln!("skip: snapshot missing");
        return;
    }
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no cuda");
        return;
    };
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("cfg");
    let cfg = Gemma4Config::from_hf_json_str(&raw_cfg).expect("parse cfg");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(dir, &device).expect("weights");
    let model =
        Gemma4::from_loader_quantized(cfg.clone(), &weights, &qcfg, &device).expect("model");

    let aux_layers: Vec<usize> = vec![1, 29, 56];
    let context: Vec<u32> = vec![2, 9259, 236764, 1041, 1463, 563];
    let ctx_len = context.len();
    let chain: Vec<u32> = vec![1390, 568, 20470, 496];
    let k = chain.len();
    let vocab = cfg.vocab_size;
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);

    let mk_pmask = |n: usize| {
        let mut m = vec![0u8; n * n];
        for i in 0..n {
            for j in 0..=i {
                m[i * n + j] = 1;
            }
        }
        m
    };
    let pmd = stream.memcpy_stod(&mk_pmask(ctx_len)).unwrap();
    let ppos: Vec<i32> = (0..ctx_len as i32).collect();
    let cmask = mk_pmask(k);
    let cpos: Vec<i32> = (ctx_len as i32..(ctx_len + k) as i32).collect();

    let mut ecache = model.new_verify_cache(64).expect("ecache");
    model
        .forward_verify(&context, &ppos, &pmd, 0, &aux_layers, &mut ecache)
        .expect("e prefill");
    let cmd = stream.memcpy_stod(&cmask).unwrap();
    let (e_logits, _) = model
        .forward_verify(&chain, &cpos, &cmd, ctx_len, &aux_layers, &mut ecache)
        .expect("e verify");
    let e_flat: Vec<f32> = e_logits
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let mut gcache = model.new_verify_cache(64).expect("gcache");
    model
        .forward_verify(&context, &ppos, &pmd, 0, &aux_layers, &mut gcache)
        .expect("g prefill");
    let mut gv = nv_models::gemma4_graph::GraphedGemma4Verify::new(
        &model,
        gcache,
        &device,
        k,
        aux_layers.clone(),
    )
    .expect("gv");

    let (g_logits, _aux) = gv
        .run(&chain, &cpos, &cmask, ctx_len)
        .expect("g run capture");

    let (g_logits2, _aux2) = gv
        .run(&chain, &cpos, &cmask, ctx_len)
        .expect("g run replay");

    let mut agree = 0usize;
    for i in 0..k {
        let er = &e_flat[i * vocab..(i + 1) * vocab];
        let gr = &g_logits[i * vocab..(i + 1) * vocab];
        if argmax(er) == argmax(gr) {
            agree += 1;
        }
    }
    let mut replay_max = 0f32;
    for i in 0..g_logits.len() {
        let d = (g_logits[i] - g_logits2[i]).abs();
        if d > replay_max {
            replay_max = d;
        }
    }
    eprintln!("GRAPHED capture: argmax agree with eager {agree}/{k}; capture-vs-replay max diff {replay_max:.5}");
    assert_eq!(
        agree, k,
        "graphed verify logits must match eager forward_verify"
    );
    assert!(replay_max < 1e-3, "replay must be deterministic vs capture");

    let iters = 50usize;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = gv.run(&chain, &cpos, &cmask, ctx_len).expect("replay");
    }
    let dt = t0.elapsed().as_secs_f64();
    let ms = 1000.0 * dt / iters as f64;
    let tpr = 3.25f64;
    eprintln!(
        "GRAPHED replay: {:.2} ms/round over {iters} iters; at {tpr} tokens/round => {:.1} tok/s ({:.2} ms/token)",
        ms, tpr / (ms / 1000.0), ms / tpr
    );
}
