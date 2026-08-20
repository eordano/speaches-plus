use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_tts::{
    CodecDecoderConfig, Qwen3TtsCodecDecoder, Qwen3TtsTalker, Qwen3TtsTalkerConfig,
    Qwen3TtsTokenizer,
};
use nv_weights::WeightLoader;

const SECONDS_PER_FRAME: f64 = 1920.0 / 24_000.0;

fn sync(device: &Device) {
    if let Device::Cuda(_) = device {
        let _ = device.synchronize();
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("NV_TTS_TALKER_DIR").map(PathBuf::from))
        .or_else(nv_tts::qwen3_tts_cache_dir)
        .context("pass model dir as arg1 or set NV_TTS_TALKER_DIR")?;
    let dev_arg = args.get(2).map(|s| s.as_str()).unwrap_or("cpu");
    let mode = args.get(3).map(|s| s.as_str()).unwrap_or("both");
    let max_n: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2048);

    let device = match dev_arg {
        "cuda" => Device::new_cuda(0).context("cuda device")?,
        _ => Device::Cpu,
    };
    eprintln!("model dir: {}", dir.display());
    eprintln!("device: {dev_arg}, mode: {mode}, max_n: {max_n}");

    let t0 = Instant::now();
    let weights = WeightLoader::open_file(&dir.join("model.safetensors"), &device)?;
    let mut talker_cfg = Qwen3TtsTalkerConfig::from_hf_config_file(&dir.join("config.json"))?;
    if args.get(5).map(|s| s.as_str()) == Some("f32") {
        talker_cfg.dtype = DType::F32;
    }
    let mut talker = Qwen3TtsTalker::new(talker_cfg.clone(), &device)?;
    talker.load_weights(&weights)?;
    let mut cp_cfg = CodecDecoderConfig::from_hf_config_file(&dir.join("config.json"))?;
    cp_cfg.dtype = talker_cfg.dtype;
    let mut cp = Qwen3TtsCodecDecoder::new(cp_cfg, &device)?;
    cp.load_weights(&weights)?;
    eprintln!("talker+cp loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let tokenizer = Qwen3TtsTokenizer::from_dir(&dir)?;
    let text = "The quick brown fox jumps over the lazy dog while the orchestra plays a long and winding melody through the evening air.";
    let ids = tokenizer.encode_text(text)?;
    let text_hidden = talker.embed_text_ids(&ids)?;
    eprintln!("text tokens: {}", ids.len());

    let fake_tok = |i: usize| -> u32 { ((i * 37) % 2000) as u32 };

    let checkpoints: Vec<usize> = [256usize, 512, 1024, 2048, 4096, 8192]
        .iter()
        .copied()
        .filter(|&n| n <= max_n)
        .collect();

    if mode == "both" || mode == "uncached" {
        eprintln!("== uncached step_full_frame_with_speaker: per-step wall time vs context ==");
        for &n in &checkpoints {
            let prev: Vec<u32> = (0..n).map(fake_tok).collect();
            let reps = if n >= 2048 { 1 } else { 2 };
            let mut total = 0.0f64;
            for _ in 0..reps {
                let t = Instant::now();
                let (base, _extras) =
                    talker.step_full_frame_with_speaker(&text_hidden, None, &prev, &cp, &[])?;
                sync(&device);
                total += t.elapsed().as_secs_f64();
                let _ = base;
            }
            let per = total / reps as f64;
            println!(
                "uncached n={n} per_step_ms={:.1} implied_rtf={:.4}",
                per * 1e3,
                SECONDS_PER_FRAME / per
            );
        }
    }

    if mode == "both" || mode == "cached" {
        eprintln!("== cached step loop: per-step wall time vs context ==");
        let text_len = text_hidden.dims()[1];
        let mut cache = talker.new_kv_cache(text_len + max_n + 2)?;
        let mut cp_cache = cp.new_kv_cache()?;
        let t = Instant::now();
        let (_base, hidden) = talker.step_cached_with_hidden(&text_hidden, None, &mut cache)?;
        let h = hidden.reshape((1usize, 1usize, talker_cfg.hidden_size))?;
        let _extras = cp.predict_with_cache(&h, &mut cp_cache)?;
        sync(&device);
        println!(
            "cached prefill_ms={:.1} (text_len={text_len})",
            t.elapsed().as_secs_f64() * 1e3
        );

        let mut window_start = Instant::now();
        let mut window_talker = 0.0f64;
        let mut window_cp = 0.0f64;
        let window = 32usize;
        let mut step = 1usize;
        while step < max_n {
            let tok = fake_tok(step);
            let tt = Instant::now();
            let (_b, hidden) =
                talker.step_cached_with_hidden(&text_hidden, Some(tok), &mut cache)?;
            sync(&device);
            window_talker += tt.elapsed().as_secs_f64();
            let ct = Instant::now();
            let h = hidden.reshape((1usize, 1usize, talker_cfg.hidden_size))?;
            let _extras = cp.predict_with_cache(&h, &mut cp_cache)?;
            sync(&device);
            window_cp += ct.elapsed().as_secs_f64();
            step += 1;
            if checkpoints.contains(&step) {
                let per = window_start.elapsed().as_secs_f64() / window as f64;
                let per_t = window_talker / window as f64;
                let per_c = window_cp / window as f64;
                println!(
                    "cached n={step} per_step_ms={:.2} talker_ms={:.2} cp_ms={:.2} implied_rtf={:.2}",
                    per * 1e3,
                    per_t * 1e3,
                    per_c * 1e3,
                    SECONDS_PER_FRAME / per
                );
            }
            if checkpoints.iter().any(|&c| step + window == c) {
                window_start = Instant::now();
                window_talker = 0.0;
                window_cp = 0.0;
            }
        }
    }

    if mode == "vocoder" || mode == "both" {
        eprintln!("== vocoder decode: 10-frame chunk on {dev_arg} ==");
        let (voc, _report) = nv_tts::load_vocoder_from_qwen3_tts(&dir, &device, DType::F32)?;
        let frames: Vec<[u32; nv_omni::vocoder::NUM_CODEBOOKS]> = (0..10)
            .map(|i| {
                let mut row = [0u32; nv_omni::vocoder::NUM_CODEBOOKS];
                for (k, r) in row.iter_mut().enumerate() {
                    *r = fake_tok(i * 16 + k);
                }
                row
            })
            .collect();
        let _ = voc.decode(&frames)?;
        let t = Instant::now();
        let reps = 3;
        for _ in 0..reps {
            let _ = voc.decode(&frames)?;
        }
        let per = t.elapsed().as_secs_f64() / reps as f64;
        println!(
            "vocoder chunk10 decode_ms={:.1} per_frame_ms={:.2} implied_rtf={:.2}",
            per * 1e3,
            per * 1e2,
            10.0 * SECONDS_PER_FRAME / per
        );
    }

    let _ = Tensor::zeros(1, DType::F32, &Device::Cpu);
    Ok(())
}
