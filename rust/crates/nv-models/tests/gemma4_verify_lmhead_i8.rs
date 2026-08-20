#![cfg(feature = "cuda")]

use candle_core::{Device, Tensor};
use nv_models::gemma4::{Gemma4, Gemma4Config};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::Path;

fn gemma4_nvfp4_snapshot_home_default() -> String {
    format!(
        "{}/.cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots/e5ef03afa233c35cb000323ff098d4291e1dd07c",
        std::env::var("HOME").unwrap_or_default()
    )
}

fn snapshot_dir() -> Option<std::path::PathBuf> {
    if let Ok(d) = std::env::var("NV_G4_SNAPSHOT") {
        let p = std::path::PathBuf::from(d);
        if p.join("config.json").exists() {
            return Some(p);
        }
    }
    let p = std::path::PathBuf::from(gemma4_nvfp4_snapshot_home_default());
    if p.join("config.json").exists() {
        return Some(p);
    }
    None
}

const STEPS: usize = 48;
const CONTEXT: [u32; 6] = [2, 9259, 236764, 1041, 1463, 563];
const CHAIN: [u32; 4] = [1390, 568, 20470, 496];

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

fn top2(row: &[f32]) -> ((usize, f32), f32) {
    let mut b1 = (0usize, f32::NEG_INFINITY);
    let mut b2 = f32::NEG_INFINITY;
    for (i, &x) in row.iter().enumerate() {
        if x > b1.1 {
            b2 = b1.1;
            b1 = (i, x);
        } else if x > b2 {
            b2 = x;
        }
    }
    (b1, b1.1 - b2)
}

fn load_model(device: &Device) -> (Gemma4, usize) {
    let dir = snapshot_dir().expect("no 31B snapshot");
    let dir = dir.as_path();
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("cfg");
    let cfg = Gemma4Config::from_hf_json_str(&raw_cfg).expect("parse cfg");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(dir, device).expect("weights");
    let vocab = cfg.vocab_size;
    let model = Gemma4::from_loader_quantized(cfg, &weights, &qcfg, device).expect("model");
    (model, vocab)
}

fn verify_stream(
    model: &Gemma4,
    device: &Device,
    vocab: usize,
    force: Option<&[u32]>,
) -> (Vec<u32>, Vec<Vec<f32>>) {
    let aux_layers: Vec<usize> = vec![];
    let ctx = CONTEXT.to_vec();
    let ctx_len = ctx.len();
    let dev = match device {
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
    let mut cache = model.new_verify_cache(ctx_len + STEPS + 8).expect("cache");
    let pmd = stream.memcpy_stod(&lower(ctx_len)).unwrap();
    let ppos: Vec<i32> = (0..ctx_len as i32).collect();
    let (pl, _) = model
        .forward_verify(&ctx, &ppos, &pmd, 0, &aux_layers, &mut cache)
        .expect("prefill");
    let plf: Vec<f32> = pl
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let last_row = plf[(ctx_len - 1) * vocab..ctx_len * vocab].to_vec();
    let mut tok = argmax(&last_row) as u32;
    let mut committed = ctx_len;
    let one_mask = stream.memcpy_stod(&[1u8]).unwrap();
    let mut toks: Vec<u32> = vec![tok];
    let mut rows: Vec<Vec<f32>> = vec![last_row];
    for step in 0..STEPS - 1 {
        let feed = force.map(|f| f[step]).unwrap_or(tok);
        let (l, _) = model
            .forward_verify(
                &[feed],
                &[committed as i32],
                &one_mask,
                committed,
                &aux_layers,
                &mut cache,
            )
            .expect("step");
        let lf: Vec<f32> = l
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        committed += 1;
        tok = argmax(&lf[0..vocab]) as u32;
        toks.push(tok);
        rows.push(lf[0..vocab].to_vec());
    }
    (toks, rows)
}

fn graphed_round_ms(model: &Gemma4, device: &Device) -> f64 {
    let aux_layers: Vec<usize> = vec![];
    let ctx = CONTEXT.to_vec();
    let ctx_len = ctx.len();
    let k = CHAIN.len();
    let dev = match device {
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
    let mut gcache = model.new_verify_cache(64).expect("gcache");
    let pmd = stream.memcpy_stod(&lower(ctx_len)).unwrap();
    let ppos: Vec<i32> = (0..ctx_len as i32).collect();
    model
        .forward_verify(&ctx, &ppos, &pmd, 0, &aux_layers, &mut gcache)
        .expect("g prefill");
    let cmask = lower(k);
    let cpos: Vec<i32> = (ctx_len as i32..(ctx_len + k) as i32).collect();
    let mut gv =
        nv_models::gemma4_graph::GraphedGemma4Verify::new(model, gcache, device, k, aux_layers)
            .expect("graphed verify");
    let _ = gv.run(&CHAIN, &cpos, &cmask, ctx_len).expect("capture");
    let _ = gv.run(&CHAIN, &cpos, &cmask, ctx_len).expect("warm replay");
    let iters = 50usize;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = gv.run(&CHAIN, &cpos, &cmask, ctx_len).expect("replay");
    }
    1000.0 * t0.elapsed().as_secs_f64() / iters as f64
}

#[test]
#[ignore]
fn verify_lmhead_int8_agreement() {
    if std::env::var("NV_G4_LMI8_TEST").ok().as_deref() != Some("1") {
        eprintln!("SKIP: set NV_G4_LMI8_TEST=1");
        return;
    }
    if snapshot_dir().is_none() {
        eprintln!("SKIP: snapshot missing");
        return;
    }
    let device = Device::new_cuda(0).expect("cuda");

    std::env::set_var("NV_VERIFY_LMHEAD_INT8", "0");
    let (model_bf16, vocab) = load_model(&device);
    std::env::set_var("NV_VERIFY_LMHEAD_INT8", "1");
    let (model_i8, _) = load_model(&device);
    std::env::remove_var("NV_VERIFY_LMHEAD_INT8");

    let (bf16_toks, bf16_rows) = verify_stream(&model_bf16, &device, vocab, None);
    eprintln!("[bf16] greedy stream: {bf16_toks:?}");
    let (_i8_toks, i8_rows) = verify_stream(&model_i8, &device, vocab, Some(&bf16_toks));

    let engaged = bf16_rows
        .iter()
        .zip(&i8_rows)
        .any(|(a, b)| a.iter().zip(b).any(|(x, y)| x != y));
    assert!(
        engaged,
        "int8 and bf16 logits are bitwise identical: NV_VERIFY_LMHEAD_INT8 did not engage"
    );

    let mut agree = 0usize;
    let mut max_rel = 0f32;
    let mut mism: Vec<(usize, f32)> = Vec::new();
    for (i, (br, ir)) in bf16_rows.iter().zip(&i8_rows).enumerate() {
        let ((ab, _), margin) = top2(br);
        let ai = argmax(ir);
        if ab == ai {
            agree += 1;
        } else {
            mism.push((i, margin));
        }
        let denom = br.iter().fold(0f32, |m, x| m.max(x.abs())).max(1.0);
        let d = br
            .iter()
            .zip(ir)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        max_rel = max_rel.max(d / denom);
    }
    eprintln!(
        "AGREE {agree}/{} teacher-forced argmax agreement; max rel logit err {max_rel:.4}; \
         disagreements (pos, bf16 top1-top2 margin): {mism:?}",
        bf16_rows.len()
    );

    let rate = agree as f64 / bf16_rows.len() as f64;
    assert!(
        rate >= 0.90,
        "int8 lm_head argmax agreement collapsed: {agree}/{}",
        bf16_rows.len()
    );
}

#[test]
#[ignore]
fn verify_lmhead_round_time() {
    if std::env::var("NV_G4_LMI8_TEST").ok().as_deref() != Some("1") {
        eprintln!("SKIP: set NV_G4_LMI8_TEST=1");
        return;
    }
    if snapshot_dir().is_none() {
        eprintln!("SKIP: snapshot missing");
        return;
    }
    let device = Device::new_cuda(0).expect("cuda");
    let arm = if std::env::var("NV_VERIFY_LMHEAD_INT8").as_deref() != Ok("0") {
        "int8"
    } else {
        "bf16"
    };
    let (model, _vocab) = load_model(&device);
    let ms = graphed_round_ms(&model, &device);
    eprintln!(
        "ROUND-TIME arm={arm} graphed verify round {ms:.3} ms over 50 iters (M=4 chain, short ctx)"
    );
}
