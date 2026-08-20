#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use nv_omni::{
    audio_tokens_for_mel_frames, build_mrope_positions, whisper_log_mel_128, AuTConfig,
    AudioEncoder, ModalitySplice, OmniDeepstack, OmniKvCache, OmniPositions, OmniSpecialIds,
    OmniThinker, OmniThinkerConfig, OmniVisionEncoder,
};

const PREP: &str = "\
NV_OMNI_MODEL_DIR is missing/invalid. Procure the checkpoint:\n\
  cd <repo root>\n\
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; export HF_HUB_DISABLE_XET=1\n\
  df -h .      # require >= 300GB free before starting\n\
  hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir <dst>\n\
  # verify each model-000NN-of-00015.safetensors real byte size (sum 70,523,299,202);\n\
  # any 135-byte shard is an LFS pointer stub = BLOCKER.\n\
  # generate tokenizer.json once (repo ships only vocab.json+merges.txt):\n\
  python3 -c \"from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained('<dst>').save_pretrained('<dst>')\"\n\
  # verify tokenizer.json exists and '<|image_pad|>' round-trips to [151655].\n\
Then rerun with NV_OMNI_TEST=1 NV_OMNI_MODEL_DIR=<dst>.";

fn test_enabled() -> bool {
    std::env::var("NV_OMNI_TEST").as_deref() == Ok("1")
}

fn model_dir() -> PathBuf {
    let d = std::env::var("NV_OMNI_MODEL_DIR")
        .unwrap_or_else(|_| panic!("NV_OMNI_TEST=1 but NV_OMNI_MODEL_DIR unset.\n{PREP}"));
    let dir = PathBuf::from(d);
    let cfg = dir.join("config.json");
    if !cfg.is_file() {
        panic!("NV_OMNI_MODEL_DIR has no config.json.\n{PREP}");
    }
    let idx = dir.join("model.safetensors.index.json");
    if !idx.is_file() {
        panic!("NV_OMNI_MODEL_DIR has no model.safetensors.index.json.\n{PREP}");
    }
    if !dir.join("tokenizer.json").is_file() {
        panic!("NV_OMNI_MODEL_DIR has no tokenizer.json.\n{PREP}");
    }
    for entry in std::fs::read_dir(&dir).unwrap() {
        let p = entry.unwrap().path();
        if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            let len = std::fs::metadata(&p).unwrap().len();
            if len < 1024 {
                panic!("shard {} is a {}-byte LFS pointer stub.\n{PREP}", p.display(), len);
            }
        }
    }
    dir
}

fn loader(dir: &PathBuf, device: &Device) -> Result<nv_weights::WeightLoader> {
    nv_weights::WeightLoader::open_dir(dir, device).context("open weights dir")
}

fn index_prefix_count(dir: &PathBuf, prefix: &str) -> Result<usize> {
    let raw = std::fs::read_to_string(dir.join("model.safetensors.index.json"))?;
    let v: serde_json::Value = serde_json::from_str(&raw)?;
    let map = v
        .get("weight_map")
        .and_then(|m| m.as_object())
        .context("index has no weight_map")?;
    Ok(map.keys().filter(|k| k.starts_with(prefix)).count())
}

fn tokenizer(dir: &PathBuf) -> Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("load tokenizer.json: {e}"))
}

fn encode(tok: &tokenizers::Tokenizer, s: &str) -> Result<Vec<u32>> {
    Ok(tok
        .encode(s, false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec())
}

fn greedy(logits: &Tensor) -> Result<u32> {
    let v: Vec<f32> = logits.to_vec1()?;
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    Ok(best as u32)
}

fn red_image(size: usize) -> (Vec<u8>, usize, usize) {
    let mut rgb = vec![0u8; size * size * 3];
    for px in rgb.chunks_mut(3) {
        px[0] = 220;
        px[1] = 30;
        px[2] = 30;
    }
    (rgb, size, size)
}

#[test]
#[ignore]
fn omni_audio_tower_real_weights_forward() -> Result<()> {
    if !test_enabled() {
        eprintln!("SKIP omni_real_weights: NV_OMNI_TEST unset");
        return Ok(());
    }
    let dir = model_dir();
    let device = Device::new_cuda(0)?;
    let t0 = Instant::now();
    let w = loader(&dir, &device)?;
    let index_count = index_prefix_count(&dir, "thinker.audio_tower.")?;
    assert_eq!(index_count, 525, "index audio_tower tensor count");
    let cfg = AuTConfig::from_hf_config_json(dir.join("config.json"))?;
    let mut enc = AudioEncoder::new(&cfg, &device)?;
    let loaded = enc.load_weights(&w)?;
    assert_eq!(loaded, 525, "audio load_weights count");

    let wav = std::env::var("NV_OMNI_TEST_WAV").expect("NV_OMNI_TEST_WAV (16kHz mono wav) required");
    let reader = hound::WavReader::open(&wav)?;
    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.0))
        .collect::<std::result::Result<_, _>>()?;
    let (mel, frames) = whisper_log_mel_128(&samples)?;
    let mel = Tensor::from_vec(mel, (128, frames), &device)?;
    let out = enc.forward(&mel)?;
    assert_eq!(out.dims(), &[audio_tokens_for_mel_frames(frames), 2048]);
    let f: Vec<f32> = out.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1()?;
    assert!(f.iter().all(|x| x.is_finite()), "audio features must be finite");
    let r0: Vec<f32> = out.i(0)?.to_dtype(candle_core::DType::F32)?.to_vec1()?;
    let r1: Vec<f32> = out.i(1)?.to_dtype(candle_core::DType::F32)?.to_vec1()?;
    assert!(r0 != r1, "audio rows must be non-degenerate");
    eprintln!(
        "omni_audio_tower_real_weights_forward: {} frames -> {} rows in {:.1}s [{}]",
        frames,
        out.dim(0)?,
        t0.elapsed().as_secs_f32(),
        dir.display()
    );
    Ok(())
}

#[test]
#[ignore]
fn omni_vision_tower_real_weights_forward() -> Result<()> {
    if !test_enabled() {
        eprintln!("SKIP omni_real_weights: NV_OMNI_TEST unset");
        return Ok(());
    }
    let dir = model_dir();
    let device = Device::new_cuda(0)?;
    let t0 = Instant::now();
    let w = loader(&dir, &device)?;
    assert_eq!(index_prefix_count(&dir, "thinker.visual.")?, 351, "index visual count");
    let mut enc = OmniVisionEncoder::from_hf_config_json(dir.join("config.json"), &device)?;
    assert_eq!(enc.load_weights(&w)?, 351, "vision load_weights count");

    let (rgb, wpx, hpx) = red_image(512);
    let (patches, grid) = enc.patchify_rgb(&rgb, wpx, hpx, &device)?;
    let (main, deep) = enc.forward(&patches, grid)?;
    let merged = grid.1 * grid.2 / 4;
    assert_eq!(main.dims(), &[merged, 2048]);
    assert_eq!(deep.len(), 3);
    for d in &deep {
        assert_eq!(d.dims(), &[merged, 2048]);
    }
    let f: Vec<f32> = main.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1()?;
    assert!(f.iter().all(|x| x.is_finite()));
    eprintln!(
        "omni_vision_tower_real_weights_forward: grid {:?} -> [{},2048] in {:.1}s [{}]",
        grid,
        merged,
        t0.elapsed().as_secs_f32(),
        dir.display()
    );
    Ok(())
}

fn load_thinker(dir: &PathBuf, device: &Device) -> Result<OmniThinker> {
    let w = loader(dir, device)?;
    let cfg = OmniThinkerConfig::from_hf_config_json(dir.join("config.json"))?;
    let mut thinker = OmniThinker::new(cfg, device)?;
    let n = thinker.load_weights(&w)?;
    assert_eq!(n, 18867, "thinker load_weights count");
    Ok(thinker)
}

#[test]
#[ignore]
fn omni_thinker_text_greedy_paris() -> Result<()> {
    if !test_enabled() {
        eprintln!("SKIP omni_real_weights: NV_OMNI_TEST unset");
        return Ok(());
    }
    let dir = model_dir();
    let device = Device::new_cuda(0)?;
    let t0 = Instant::now();
    let thinker = load_thinker(&dir, &device)?;
    let tok = tokenizer(&dir)?;
    let ids = OmniSpecialIds::from_hf_config_json(dir.join("config.json"))?;

    let prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n";
    let mut tokens = encode(&tok, prompt)?;
    let (pos, mut next) = build_mrope_positions(&tokens, &ids, &[], &[])?;
    let x = thinker.embed_with_splices(&tokens, &[])?.unsqueeze(0)?;
    let mut cache = OmniKvCache::new(thinker.num_layers());
    let mut logits = thinker.forward_step(&x, &pos, &mut cache, None)?;

    let mut out_ids = Vec::new();
    for _ in 0..16 {
        let tid = greedy(&logits)?;
        if tid == ids.im_end || tid == ids.endoftext {
            break;
        }
        out_ids.push(tid);
        tokens.push(tid);
        let xi = thinker.embed_with_splices(&[tid], &[])?.unsqueeze(0)?;
        let pi = OmniPositions::uniform(&[next]);
        next += 1;
        logits = thinker.forward_step(&xi, &pi, &mut cache, None)?;
    }
    let text = tok.decode(&out_ids, true).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    eprintln!(
        "omni_thinker_text_greedy_paris: '{}' in {:.1}s [{}]",
        text,
        t0.elapsed().as_secs_f32(),
        dir.display()
    );
    assert!(text.contains("Paris"), "expected 'Paris', got '{text}'");
    Ok(())
}

fn expand_marker(tokens: &mut Vec<u32>, pad: u32, count: usize) {
    let pos = tokens.iter().position(|&t| t == pad).expect("marker pad present");
    let mut new = Vec::with_capacity(tokens.len() + count - 1);
    new.extend_from_slice(&tokens[..pos]);
    new.extend(std::iter::repeat(pad).take(count));
    new.extend_from_slice(&tokens[pos + 1..]);
    *tokens = new;
}

fn pad_run_start(tokens: &[u32], pad: u32) -> usize {
    tokens.iter().position(|&t| t == pad).unwrap()
}

#[test]
#[ignore]
fn omni_image_grounded_decode() -> Result<()> {
    if !test_enabled() {
        eprintln!("SKIP omni_real_weights: NV_OMNI_TEST unset");
        return Ok(());
    }
    let dir = model_dir();
    let device = Device::new_cuda(0)?;
    let t0 = Instant::now();
    let thinker = load_thinker(&dir, &device)?;
    let tok = tokenizer(&dir)?;
    let ids = OmniSpecialIds::from_hf_config_json(dir.join("config.json"))?;
    let mut vision = OmniVisionEncoder::from_hf_config_json(dir.join("config.json"), &device)?;
    assert_eq!(vision.load_weights(&loader(&dir, &device)?)?, 351);

    let (rgb, wpx, hpx) = red_image(512);
    let (patches, grid) = vision.patchify_rgb(&rgb, wpx, hpx, &device)?;
    let (emb, deep) = vision.forward(&patches, grid)?;
    let n_img = emb.dim(0)?;
    let merged_grid = (1usize, grid.1 / 2, grid.2 / 2);

    let prompt = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What color is this image? Answer with one word.<|im_end|>\n<|im_start|>assistant\n";
    let mut tokens = encode(&tok, prompt)?;
    expand_marker(&mut tokens, ids.image_pad, n_img);
    let splice_pos = pad_run_start(&tokens, ids.image_pad);

    let (pos, mut next) = build_mrope_positions(&tokens, &ids, &[merged_grid], &[])?;
    let x = thinker
        .embed_with_splices(
            &tokens,
            &[ModalitySplice { position: splice_pos, embedding: emb.clone() }],
        )?
        .unsqueeze(0)?;
    let rows: Vec<u32> = (splice_pos as u32..(splice_pos + n_img) as u32).collect();
    let rows = Tensor::from_vec(rows, n_img, &device)?;
    let ds = OmniDeepstack { rows, embeds: deep };
    let mut cache = OmniKvCache::new(thinker.num_layers());
    let mut logits = thinker.forward_step(&x, &pos, &mut cache, Some(&ds))?;

    let mut out_ids = Vec::new();
    for _ in 0..24 {
        let tid = greedy(&logits)?;
        if tid == ids.im_end || tid == ids.endoftext {
            break;
        }
        out_ids.push(tid);
        let xi = thinker.embed_with_splices(&[tid], &[])?.unsqueeze(0)?;
        let pi = OmniPositions::uniform(&[next]);
        next += 1;
        logits = thinker.forward_step(&xi, &pi, &mut cache, None)?;
    }
    let text = tok.decode(&out_ids, true).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    eprintln!(
        "omni_image_grounded_decode: '{}' in {:.1}s [{}]",
        text,
        t0.elapsed().as_secs_f32(),
        dir.display()
    );
    assert!(text.to_lowercase().contains("red"), "expected 'red', got '{text}'");
    Ok(())
}

#[test]
#[ignore]
fn omni_audio_grounded_decode() -> Result<()> {
    if !test_enabled() {
        eprintln!("SKIP omni_real_weights: NV_OMNI_TEST unset");
        return Ok(());
    }
    let dir = model_dir();
    let device = Device::new_cuda(0)?;
    let t0 = Instant::now();
    let thinker = load_thinker(&dir, &device)?;
    let tok = tokenizer(&dir)?;
    let ids = OmniSpecialIds::from_hf_config_json(dir.join("config.json"))?;
    let cfg = AuTConfig::from_hf_config_json(dir.join("config.json"))?;
    let mut audio = AudioEncoder::new(&cfg, &device)?;
    assert_eq!(audio.load_weights(&loader(&dir, &device)?)?, 525);

    let wav = std::env::var("NV_OMNI_TEST_WAV").expect("NV_OMNI_TEST_WAV required");
    let keyword = std::env::var("NV_OMNI_TEST_WAV_KEYWORD").unwrap_or_else(|_| "fox".to_string());
    let reader = hound::WavReader::open(&wav)?;
    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.0))
        .collect::<std::result::Result<_, _>>()?;
    let (mel, frames) = whisper_log_mel_128(&samples)?;
    let mel = Tensor::from_vec(mel, (128, frames), &device)?;
    let emb = audio.forward(&mel)?;
    let n_aud = emb.dim(0)?;
    assert_eq!(n_aud, audio_tokens_for_mel_frames(frames));

    let prompt = "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>What did the speaker say?<|im_end|>\n<|im_start|>assistant\n";
    let mut tokens = encode(&tok, prompt)?;
    expand_marker(&mut tokens, ids.audio_pad, n_aud);
    let splice_pos = pad_run_start(&tokens, ids.audio_pad);

    let (pos, mut next) = build_mrope_positions(&tokens, &ids, &[], &[n_aud])?;
    let x = thinker
        .embed_with_splices(&tokens, &[ModalitySplice { position: splice_pos, embedding: emb }])?
        .unsqueeze(0)?;
    let mut cache = OmniKvCache::new(thinker.num_layers());
    let mut logits = thinker.forward_step(&x, &pos, &mut cache, None)?;

    let mut out_ids = Vec::new();
    for _ in 0..32 {
        let tid = greedy(&logits)?;
        if tid == ids.im_end || tid == ids.endoftext {
            break;
        }
        out_ids.push(tid);
        let xi = thinker.embed_with_splices(&[tid], &[])?.unsqueeze(0)?;
        let pi = OmniPositions::uniform(&[next]);
        next += 1;
        logits = thinker.forward_step(&xi, &pi, &mut cache, None)?;
    }
    let text = tok.decode(&out_ids, true).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    eprintln!(
        "omni_audio_grounded_decode: '{}' in {:.1}s [{}]",
        text,
        t0.elapsed().as_secs_f32(),
        dir.display()
    );
    assert!(
        text.to_lowercase().contains(&keyword.to_lowercase()),
        "expected keyword '{keyword}', got '{text}'"
    );
    Ok(())
}

use candle_core::IndexOp;
