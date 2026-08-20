#![cfg(feature = "wgpu")]

mod common;
use common::distinct;
use common::envn;
use common::LcgCentered0p1Shift32 as Lcg;
use std::time::Instant;

use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_wgpu::{
    quantize_nvfp4_host, Gemma4Wgpu, HostBf16Lin, HostLayer, HostProj, HostWeights,
};
use common::gemma4_host_weights_nvfp4_ffn as host_weights;

fn ctx_or_panic() {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => eprintln!("[g4d-cp] adapter: {}", ctx.summary()),
        Err(e) => panic!("gemma4 dense chunked prefill needs a wgpu adapter: {e}"),
    }
}

fn config_json(layers: usize, hidden: usize, inter: usize, vocab: usize, window: usize) -> String {
    let mut types = Vec::with_capacity(layers);
    for i in 0..layers {
        types.push(if (i + 1) % 3 == 0 {
            "\"full_attention\""
        } else {
            "\"sliding_attention\""
        });
    }
    format!(
        r#"{{
  "text_config": {{
    "hidden_size": {hidden},
    "intermediate_size": {inter},
    "num_hidden_layers": {layers},
    "num_attention_heads": {nq},
    "num_key_value_heads": {nkv},
    "num_global_key_value_heads": {nkvg},
    "head_dim": {hds},
    "global_head_dim": {hdf},
    "vocab_size": {vocab},
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "sliding_window": {window},
    "final_logit_softcapping": 0.0,
    "layer_types": [{}],
    "attention_k_eq_v": false,
    "hidden_activation": "gelu_pytorch_tanh",
    "num_kv_shared_layers": 0,
    "rope_parameters": {{
      "full_attention": {{"partial_rotary_factor": 0.25, "rope_theta": 1000000.0}},
      "sliding_attention": {{"rope_theta": 10000.0}}
    }}
  }},
  "tie_word_embeddings": true
}}"#,
        types.join(", "),
        nq = envn("NV_G4D_CP_NQ", 4),
        nkv = envn("NV_G4D_CP_NKV", 2),
        nkvg = envn("NV_G4D_CP_NKVG", 1),
        hds = envn("NV_G4D_CP_HD", 128),
        hdf = envn("NV_G4D_CP_HDF", 256),
    )
}

struct Arm {
    next: u32,
    logit_bits: Vec<u32>,
    ms: f64,
}

fn arm_unchunked(m: &mut Gemma4Wgpu, ids: &[u32]) -> Arm {
    m.reset();
    let (last, rest) = ids.split_last().expect("prompt");
    let t0 = Instant::now();
    for t in rest {
        m.prefill_step(*t).expect("prefill step");
    }
    let (next, logits) = m.decode_step_logits(*last).expect("last prompt token");
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / ids.len() as f64;
    Arm {
        next,
        logit_bits: logits.into_iter().map(f32::to_bits).collect(),
        ms,
    }
}

fn arm_chunked(m: &mut Gemma4Wgpu, ids: &[u32]) -> (Arm, usize) {
    m.reset();
    let (last, rest) = ids.split_last().expect("prompt");
    let t0 = Instant::now();
    let done = m.prefill_tokens(rest).expect("prefill_tokens");
    for t in &rest[done..] {
        m.prefill_step(*t).expect("prefill step");
    }
    let (next, logits) = m.decode_step_logits(*last).expect("last prompt token");
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / ids.len() as f64;
    (
        Arm {
            next,
            logit_bits: logits.into_iter().map(f32::to_bits).collect(),
            ms,
        },
        done,
    )
}

#[test]
fn gemma4_dense_chunked_prefill_is_bit_identical() {
    if std::env::var("NV_G4D_CHUNKED_PREFILL").as_deref() != Ok("1") {
        panic!("set NV_G4D_CHUNKED_PREFILL=1 to run this GPU test (it must never silently skip)");
    }
    ctx_or_panic();
    let layers = envn("NV_G4D_CP_LAYERS", 4);
    let hidden = envn("NV_G4D_CP_HIDDEN", 512);
    let inter = envn("NV_G4D_CP_INTER", 1024);
    let vocab = envn("NV_G4D_CP_VOCAB", 2048);

    let windows = if envn("NV_G4D_CP_WINDOWS", 2) >= 2 {
        vec![4096usize, 8]
    } else {
        vec![4096usize]
    };
    for window in windows {
        let raw = config_json(layers, hidden, inter, vocab, window);
        let config = Gemma4Config::from_hf_json_str(&raw).expect("config");
        let w = host_weights(&config, 0x9e3779b9);
        let t_build = Instant::now();
        let mut m = Gemma4Wgpu::new(config, &w, 512).expect("build");
        drop(w);
        eprintln!("[g4d-cp] === sliding_window = {window} ===");
        let cm = m.prefill_chunk_len();
        eprintln!(
            "[g4d-cp] built in {:.2}s: {} decode passes, chunk m={cm}, {} prefill passes/chunk",
            t_build.elapsed().as_secs_f64(),
            m.pass_count(),
            m.prefill_pass_count(),
        );
        assert!(
            cm >= 2,
            "chunked prefill is disabled on this graph; the test would be vacuous"
        );
        assert!(
            m.prefill_pass_count() > 0,
            "prefill pass list is empty; the test would be vacuous"
        );

        let pps: Vec<usize> = match envn("NV_G4D_CP_PP", 0) {
            0 => vec![cm * 3, cm * 2 + cm / 2 + 1, cm + 3, cm - 1],
            n => vec![n],
        };
        for pp in pps {
            if pp < 2 {
                continue;
            }
            let ids: Vec<u32> = (0..pp).map(|i| ((i * 7919 + 13) % vocab) as u32).collect();
            let a = arm_unchunked(&mut m, &ids);
            let (b, done) = arm_chunked(&mut m, &ids);
            let d = distinct(&a.logit_bits);
            assert_eq!(a.logit_bits.len(), vocab);
            assert!(
            d > (vocab / 4).min(1000),
            "pp={pp}: logits are degenerate ({d} distinct of {vocab}); the bit-compare would be vacuous"
        );
            let diff = a
                .logit_bits
                .iter()
                .zip(b.logit_bits.iter())
                .filter(|(x, y)| x != y)
                .count();
            let worst = a
                .logit_bits
                .iter()
                .zip(b.logit_bits.iter())
                .map(|(x, y)| (f32::from_bits(*x) - f32::from_bits(*y)).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "[g4d-cp] pp={pp:3} chunked committed {done:3} of {} prompt tokens \
             | logits {d} distinct | differing lanes {diff}/{vocab} | max |delta| {worst:.3e} \
             | next token {} vs {} | {:.3} vs {:.3} ms/prompt-tok",
                pp - 1,
                a.next,
                b.next,
                a.ms,
                b.ms
            );
            assert_eq!(
            diff, 0,
            "pp={pp}: {diff} of {vocab} logit lanes differ between chunked and unchunked prefill (max |delta| {worst:.3e})"
        );
            assert_eq!(a.next, b.next, "pp={pp}: first sampled token differs");
            if pp > cm {
                assert!(
                    done >= cm,
                    "pp={pp}: chunked path committed only {done} tokens; it did not engage"
                );
            }
        }
    }
    eprintln!("[g4d-cp] BIT-IDENTICAL on every prompt length and both sliding windows");
}

#[test]
#[ignore = "loads the ~20 GB Gemma-4-31B NVFP4 checkpoint; set NV_G4D_CP_REAL=1"]
fn real_gemma4_31b_chunked_prefill_ab() {
    if std::env::var("NV_G4D_CP_REAL").as_deref() != Ok("1") {
        panic!("set NV_G4D_CP_REAL=1 to run this GPU test (it must never silently skip)");
    }
    ctx_or_panic();
    let home = std::env::var("HOME").unwrap();
    let base = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
    let dir = std::fs::read_dir(&base)
        .expect("hub snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").exists())
        .expect("no snapshot with config.json");
    eprintln!("[g4d-cp-real] loading {}", dir.display());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let t = Instant::now();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader).unwrap();
    eprintln!(
        "[g4d-cp-real] host staging {:.1}s",
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    let mut m = Gemma4Wgpu::new(config, &host, 4096).expect("build");
    drop(host);
    let cm = m.prefill_chunk_len();
    eprintln!(
        "[g4d-cp-real] built in {:.1}s: chunk m={cm}, {} prefill passes/chunk",
        t.elapsed().as_secs_f64(),
        m.prefill_pass_count()
    );
    assert!(
        cm >= 2,
        "chunked prefill did not engage on the real checkpoint at default env; \
         the sg-epilogue default flip (8cd497514) is inert here -- record which \
         disabler fired (see the [gemma4_wgpu] boot lines above) before quoting \
         any prefill speedup for this model"
    );

    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let para = "The measurement of record for this repository is the pipeline \
                comparison document, and every number in it carries its basis. ";
    let mut text = String::new();
    let pp_target = envn("NV_G4D_CP_REAL_PP", 1024);
    while tokenizer
        .encode(text.as_str(), false)
        .unwrap()
        .get_ids()
        .len()
        < pp_target
    {
        text.push_str(para);
    }
    let mut ids: Vec<u32> = vec![2];
    ids.extend(tokenizer.encode(text.as_str(), false).unwrap().get_ids());
    ids.truncate(pp_target);
    eprintln!("[g4d-cp-real] prompt of {} real tokens", ids.len());

    let a = arm_unchunked(&mut m, &ids);
    let (b, done) = arm_chunked(&mut m, &ids);
    let diff = a
        .logit_bits
        .iter()
        .zip(b.logit_bits.iter())
        .filter(|(x, y)| x != y)
        .count();
    eprintln!(
        "[g4d-cp-real] pp={} chunked committed {done} | differing lanes {diff} \
         | next {} vs {} | unchunked {:.3} vs chunked {:.3} ms/prompt-tok ({:.2}x)",
        ids.len(),
        a.next,
        b.next,
        a.ms,
        b.ms,
        a.ms / b.ms
    );
    assert_eq!(
        diff, 0,
        "chunked and unchunked logits differ on the real checkpoint"
    );
    assert_eq!(a.next, b.next);
    assert!(done >= cm, "chunked path did not commit a full chunk");
}
