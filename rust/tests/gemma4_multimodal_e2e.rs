use std::io::Cursor;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use serde_json::json;

use nv_models::gemma4_audio::{
    Gemma4AudioConfig, Gemma4AudioEmbedder, Gemma4AudioEncoder, Gemma4AudioTower,
};
use nv_models::gemma4_mm_splice::{
    audio_num_soft_tokens, splice_mm_embeddings, GEMMA4_AUDIO_TOKEN_ID, GEMMA4_BOA_TOKEN_ID,
    GEMMA4_BOI_TOKEN_ID, GEMMA4_EOA_TOKEN_ID, GEMMA4_EOI_TOKEN_ID, GEMMA4_IMAGE_TOKEN_ID,
};
use nv_models::gemma4_vision::{Gemma4VisionConfig, Gemma4VisionTower};
use speaches_plus::oapi::chat_multimodal::{
    decode_audio_input, embed_prompt, gemma4_log_mel, messages_have_mm_parts, mm_embeddings,
    parse_mm_messages, plan_prompt, run_towers, segments_from_message, Gemma4MmTowers,
    InputAudioSpec, MmContentPart, PromptSegment,
};

const SMALL_VISION_JSON: &str = r#"{
  "vision_config": {
    "attention_bias": false,
    "default_output_length": 4,
    "head_dim": 64,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 128,
    "intermediate_size": 256,
    "model_type": "gemma4_vision",
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "position_embedding_size": 10240,
    "rms_norm_eps": 1e-06,
    "rope_parameters": { "rope_theta": 100.0, "rope_type": "default" },
    "standardize": false,
    "use_clipped_linears": false
  }
}"#;

const TEXT_HIDDEN: usize = 32;
const VOCAB: usize = 262400;

fn synth_embed_table(hidden: usize, device: &Device) -> Tensor {
    let mut s = 0x5DEECE66Du64;
    let n = VOCAB * hidden;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((s >> 40) & 0xFFFF) as f32 / 65535.0;
        v.push((unit * 2.0 - 1.0) * 0.05);
    }
    Tensor::from_vec(v, (VOCAB, hidden), device).unwrap()
}

fn byte_tokenizer(s: &str) -> anyhow::Result<Vec<u32>> {
    Ok(s.bytes().map(|b| 1000 + b as u32).collect())
}

fn gradient_png_data_url(w: u32, h: u32) -> String {
    let img = image::RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([
            (x * 255 / w.max(1)) as u8,
            (y * 255 / h.max(1)) as u8,
            ((x + y) % 256) as u8,
        ])
    });
    let mut buf = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
        .unwrap();
    use base64::Engine as _;
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&buf)
    )
}

fn sine_wav_base64(seconds: f32, freq: f32, sample_rate: u32) -> String {
    let n = (seconds * sample_rate as f32) as usize;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = Cursor::new(Vec::new());
    {
        let mut w = hound::WavWriter::new(&mut buf, spec).unwrap();
        for i in 0..n {
            let t = i as f32 / sample_rate as f32;
            let s = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5;
            w.write_sample((s * i16::MAX as f32) as i16).unwrap();
        }
        w.finalize().unwrap();
    }
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(buf.into_inner())
}

fn tiny_audio_cfg() -> Gemma4AudioConfig {
    Gemma4AudioConfig {
        model_type: Some("gemma4_audio".into()),
        attention_chunk_size: 4,
        attention_context_left: 3,
        attention_context_right: 0,
        attention_invalid_logits_value: -1.0e9,
        attention_logit_cap: 50.0,
        conv_kernel_size: 5,
        hidden_size: 32,
        num_attention_heads: 4,
        num_hidden_layers: 2,
        output_proj_dims: 24,
        residual_weight: 0.5,
        rms_norm_eps: 1e-6,
        subsampling_conv_channels: vec![8, 4],
        hidden_act: "silu".into(),
        use_clipped_linears: false,
    }
}

fn tiny_audio_tower(device: &Device) -> Gemma4AudioTower {
    let cfg = tiny_audio_cfg();
    let encoder = Gemma4AudioEncoder::synthetic(&cfg, device).unwrap();
    let embedder =
        Gemma4AudioEmbedder::synthetic(cfg.output_proj_dims, TEXT_HIDDEN, 1e-6, device).unwrap();
    Gemma4AudioTower {
        encoder,
        embedder,
        audio_token_id: Some(GEMMA4_AUDIO_TOKEN_ID),
    }
}

fn small_vision_tower(device: &Device) -> Gemma4VisionTower {
    let cfg = Gemma4VisionConfig::from_hf_json_str(SMALL_VISION_JSON).unwrap();
    Gemma4VisionTower::new_synthetic(cfg, TEXT_HIDDEN, device, DType::F32).unwrap()
}

fn rows_bits(t: &Tensor) -> Vec<Vec<u32>> {
    t.to_dtype(DType::F32)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap()
        .into_iter()
        .map(|r| r.into_iter().map(f32::to_bits).collect())
        .collect()
}

#[test]
fn parse_openai_content_shapes() {
    let messages = json!([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [
            {"type": "text", "text": "look:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}},
            {"type": "image_url", "image_url": "/tmp/x.png"},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
        ]}
    ]);
    let parsed = parse_mm_messages(&messages).unwrap();
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].role, "system");
    assert!(matches!(&parsed[0].parts[0], MmContentPart::Text(t) if t == "be brief"));
    assert_eq!(parsed[1].parts.len(), 4);
    assert!(
        matches!(&parsed[1].parts[1], MmContentPart::ImageUrl(s) if s.url().starts_with("data:image/png"))
    );
    assert!(matches!(&parsed[1].parts[2], MmContentPart::ImageUrl(s) if s.url() == "/tmp/x.png"));
    assert!(
        matches!(&parsed[1].parts[3], MmContentPart::InputAudio(a) if a.format == "wav" && a.data == "AAAA")
    );
    assert!(messages_have_mm_parts(&messages));

    let text_only = json!([
        {"role": "user", "content": "hello"},
        {"role": "user", "content": [{"type": "text", "text": "parts"}]}
    ]);
    assert!(!messages_have_mm_parts(&text_only));

    let bad = json!([{"role": "user", "content": [{"type": "video_url", "video_url": "x"}]}]);
    let err = format!("{:#}", parse_mm_messages(&bad).unwrap_err());
    assert!(err.contains("video_url"), "unexpected error: {err}");
}

#[test]
fn text_only_plan_is_bit_identical_to_plain_embedding() {
    let device = Device::Cpu;
    let towers = Gemma4MmTowers::new("synthetic", None, None);
    let segments = vec![PromptSegment::Text("just words".into())];
    let plan = plan_prompt(&towers, &segments, &device, byte_tokenizer).unwrap();
    assert!(!plan.is_multimodal());
    assert_eq!(plan.tokens, byte_tokenizer("just words").unwrap());

    let table = synth_embed_table(TEXT_HIDDEN, &device);
    let scale = (TEXT_HIDDEN as f64).sqrt();
    let via_mm = mm_embeddings(&towers, &plan, &table, scale).unwrap();
    let plain = embed_prompt(&table, scale, &plan.tokens).unwrap();
    assert_eq!(rows_bits(&via_mm), rows_bits(&plain));
}

#[test]
fn synthetic_image_end_to_end() {
    let device = Device::Cpu;
    let towers = Gemma4MmTowers::new("synthetic-e4b", Some(small_vision_tower(&device)), None);

    let messages = json!([
        {"role": "user", "content": [
            {"type": "text", "text": "hi "},
            {"type": "image_url", "image_url": {"url": gradient_png_data_url(96, 96)}},
            {"type": "text", "text": " ok"}
        ]}
    ]);
    assert!(messages_have_mm_parts(&messages));
    let parsed = parse_mm_messages(&messages).unwrap();
    let segments = segments_from_message(&parsed[0]).unwrap();
    let plan = plan_prompt(&towers, &segments, &device, byte_tokenizer).unwrap();

    assert_eq!(plan.images.len(), 1);
    let num_soft = plan.images[0].patches.num_soft_tokens;
    assert_eq!(num_soft, 4);
    assert_eq!(plan.images[0].patches.target_width, 96);
    assert_eq!(plan.images[0].patches.target_height, 96);
    assert_eq!(plan.tokens.len(), 3 + 1 + num_soft + 1 + 3);
    assert_eq!(plan.tokens[3], GEMMA4_BOI_TOKEN_ID);
    for i in 4..4 + num_soft {
        assert_eq!(plan.tokens[i], GEMMA4_IMAGE_TOKEN_ID);
    }
    assert_eq!(plan.tokens[4 + num_soft], GEMMA4_EOI_TOKEN_ID);
    assert_eq!(plan.images[0].position, 4);

    let table = synth_embed_table(TEXT_HIDDEN, &device);
    let scale = (TEXT_HIDDEN as f64).sqrt();
    let out = mm_embeddings(&towers, &plan, &table, scale).unwrap();
    assert_eq!(out.dims(), &[plan.tokens.len(), TEXT_HIDDEN]);

    let out_rows = rows_bits(&out);
    let text_rows = rows_bits(&embed_prompt(&table, scale, &plan.tokens).unwrap());
    let items = run_towers(&towers, &plan, DType::F32).unwrap();
    let tower_rows = rows_bits(&items[0].embedding);
    for i in 0..plan.tokens.len() {
        if (4..4 + num_soft).contains(&i) {
            assert_eq!(out_rows[i], tower_rows[i - 4], "soft token row {i}");
        } else {
            assert_eq!(out_rows[i], text_rows[i], "text row {i}");
        }
    }
}

#[test]
fn synthetic_audio_end_to_end() {
    let device = Device::Cpu;
    let towers = Gemma4MmTowers::new("synthetic-e4b", None, Some(tiny_audio_tower(&device)));

    let wav = sine_wav_base64(0.5, 440.0, 16000);
    let messages = json!([
        {"role": "user", "content": [
            {"type": "text", "text": "t:"},
            {"type": "input_audio", "input_audio": {"data": wav, "format": "wav"}}
        ]}
    ]);
    let parsed = parse_mm_messages(&messages).unwrap();
    let segments = segments_from_message(&parsed[0]).unwrap();

    let PromptSegment::Audio(samples) = &segments[1] else {
        panic!("expected audio segment");
    };
    assert_eq!(samples.len(), 8000);
    let expected_soft = audio_num_soft_tokens(8000, 16000, 750);
    assert_eq!(expected_soft, 13);

    let plan = plan_prompt(&towers, &segments, &device, byte_tokenizer).unwrap();
    assert_eq!(plan.audios.len(), 1);
    assert_eq!(plan.audios[0].num_soft_tokens, 13);
    assert_eq!(plan.audios[0].mel_frames, 49);
    assert_eq!(plan.tokens.len(), 2 + 1 + 13 + 1);
    assert_eq!(plan.tokens[2], GEMMA4_BOA_TOKEN_ID);
    for i in 3..16 {
        assert_eq!(plan.tokens[i], GEMMA4_AUDIO_TOKEN_ID);
    }
    assert_eq!(plan.tokens[16], GEMMA4_EOA_TOKEN_ID);
    assert_eq!(plan.audios[0].position, 3);

    let table = synth_embed_table(TEXT_HIDDEN, &device);
    let scale = (TEXT_HIDDEN as f64).sqrt();
    let out = mm_embeddings(&towers, &plan, &table, scale).unwrap();
    assert_eq!(out.dims(), &[17, TEXT_HIDDEN]);

    let out_rows = rows_bits(&out);
    let text_rows = rows_bits(&embed_prompt(&table, scale, &plan.tokens).unwrap());
    let items = run_towers(&towers, &plan, DType::F32).unwrap();
    let tower_rows = rows_bits(&items[0].embedding);
    for i in 0..17 {
        if (3..16).contains(&i) {
            assert_eq!(out_rows[i], tower_rows[i - 3], "soft token row {i}");
        } else {
            assert_eq!(out_rows[i], text_rows[i], "text row {i}");
        }
    }
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|x| x.is_finite()));
}

#[test]
fn vision_only_model_rejects_audio_cleanly() {
    let device = Device::Cpu;
    let towers = Gemma4MmTowers::new("gemma-4-26B-A4B-it", None, None);
    let segments = vec![PromptSegment::Audio(vec![0.1f32; 16000])];
    let err = format!(
        "{:#}",
        plan_prompt(&towers, &segments, &device, byte_tokenizer).unwrap_err()
    );
    assert!(err.contains("gemma-4-26B-A4B-it"), "err: {err}");
    assert!(err.contains("audio_config is null"), "err: {err}");
    assert!(err.contains("not supported"), "err: {err}");

    let img_err = format!(
        "{:#}",
        plan_prompt(
            &towers,
            &[PromptSegment::Image(image::RgbImage::new(32, 32))],
            &device,
            byte_tokenizer
        )
        .unwrap_err()
    );
    assert!(img_err.contains("vision_config missing"), "err: {img_err}");
}

#[test]
fn audio_too_short_is_clean_error() {
    let device = Device::Cpu;
    let towers = Gemma4MmTowers::new("synthetic", None, Some(tiny_audio_tower(&device)));
    let err = format!(
        "{:#}",
        plan_prompt(
            &towers,
            &[PromptSegment::Audio(vec![0.0f32; 100])],
            &device,
            byte_tokenizer
        )
        .unwrap_err()
    );
    assert!(err.contains("audio too short"), "err: {err}");
}

#[test]
fn unsupported_audio_format_is_clean_error() {
    let spec = InputAudioSpec {
        data: "AAAA".into(),
        format: "mp3".into(),
    };
    let err = format!("{:#}", decode_audio_input(&spec).unwrap_err());
    assert!(err.contains("mp3"), "err: {err}");
    assert!(err.contains("wav"), "err: {err}");
}

#[test]
fn mel_frontend_frame_arithmetic() {
    for secs in [1usize, 2, 5] {
        let n = 16000 * secs;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.3)
            .collect();
        let (mel, frames) = gemma4_log_mel(&samples);
        let expected_frames = (n + 160 - 321) / 160 + 1;
        assert_eq!(frames, expected_frames, "{secs}s");
        assert_eq!(mel.len(), frames * 128);
        let floor = (1.0e-3f32).ln();
        assert!(mel.iter().all(|x| x.is_finite() && *x >= floor - 1e-4));
        assert!(
            mel.iter().any(|x| *x > floor + 1.0),
            "{secs}s: all at floor"
        );
    }
    assert_eq!(audio_num_soft_tokens(16000, 16000, 750), 25);
}

#[test]
fn wav_decode_resamples_to_16k() {
    let wav = sine_wav_base64(0.25, 440.0, 48000);
    let spec = InputAudioSpec {
        data: wav,
        format: "wav".into(),
    };
    let samples = decode_audio_input(&spec).unwrap();
    assert_eq!(samples.len(), 4000);
    let peak = samples.iter().fold(0f32, |m, &x| m.max(x.abs()));
    assert!((peak - 0.5).abs() < 0.05, "peak {peak}");
}

fn e4b_snapshot_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots");
    for e in std::fs::read_dir(&snaps).ok()?.flatten() {
        let p = e.path();
        if p.join("config.json").is_file() {
            return Some(p);
        }
    }
    None
}

const WHY_THE_SYNTHETIC_ROWS_ARE_NOT_MULTIMODAL_COVERAGE: &str =
    "Every row in this file except e4b_real_weights_mm_e2e builds its towers with \
     Gemma4MmTowers::new over Gemma4AudioEncoder::synthetic and Gemma4VisionTower::new_synthetic, \
     against a 128-hidden vision config and a 32-hidden text width invented in this file. They \
     pin the splice arithmetic -- sentinel placement, soft-token counts, which rows come from \
     which tower -- and they are worth having. What they do not touch is a checkpoint: the \
     suite is named gemma4_multimodal_e2e, so a reader seeing its rows go green could easily \
     read that as Gemma-4 multimodal working, when the only row that loads Gemma-4 is gated \
     off. This row states the split so the count cannot be misread.";

const WHY_THE_REAL_WEIGHTS_ROW_STAYS_GATED: &str =
    "e4b_real_weights_mm_e2e needs no GPU and no download -- it is Device::Cpu against an \
     already-cached snapshot -- but it does map the full google/gemma-4-E4B-it checkpoint and \
     run both towers unoptimized, and on a unified-memory box that is not free to anyone \
     measuring alongside it. So the double gate is right and matches its siblings. What was \
     wrong is that NV_GEMMA4_MM_E2E_TEST appeared nowhere in this repo except the two lines \
     below, while every documented sibling gate (NV_WGPU_SERVE_TEST, NV_TOOLS_REAL_TEST, \
     NV_LAGUNA_TEST) is listed in docs/book/08-build-and-testing.md. An opt-in nobody can \
     discover is not an opt-in, it is a delete with extra steps, and this row had never once \
     executed. It has now been run and it passes end to end: it loads the real towers and the \
     real tokenizer, plans a prompt around a 256x192 image and a 1s 16kHz clip, embeds through \
     the checkpoint's own embed_tokens at width 2560, and checks every spliced row against the \
     tower or text row it must equal, bit for bit. Register the variable where its siblings \
     live and it stays reachable.";

#[test]
fn only_one_row_in_this_suite_loads_a_gemma4_checkpoint() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join(std::path::Path::new(file!()).file_name().unwrap()),
    )
    .expect("read this harness's own source");
    let rows = src.lines().filter(|l| l.trim() == "#[test]").count();
    let gated = src.lines().filter(|l| l.trim() == "#[ignore]").count();
    eprintln!(
        "[gemma4_multimodal_e2e] {} of {rows} rows run on SYNTHETIC towers and load no \
         checkpoint; the {gated} real-weights row (e4b_real_weights_mm_e2e) is #[ignore]d \
         behind NV_GEMMA4_MM_E2E_TEST=1 and does not run here.",
        rows - gated
    );
    assert_eq!(
        gated, 1,
        "{WHY_THE_SYNTHETIC_ROWS_ARE_NOT_MULTIMODAL_COVERAGE}\n\n{rows} rows carry {gated} \
         ignore attributes, so the one-real-row split this reports no longer holds"
    );
    assert!(
        rows > gated,
        "{WHY_THE_SYNTHETIC_ROWS_ARE_NOT_MULTIMODAL_COVERAGE}"
    );
}

#[test]
#[ignore]
fn e4b_real_weights_mm_e2e() {
    if std::env::var("NV_GEMMA4_MM_E2E_TEST").ok().as_deref() != Some("1") {
        panic!(
            "{WHY_THE_REAL_WEIGHTS_ROW_STAYS_GATED}\n\ne4b_real_weights_mm_e2e is #[ignore]d, so \
             it was asked for BY NAME, but NV_GEMMA4_MM_E2E_TEST=1 is not set. Re-run with \
             NV_GEMMA4_MM_E2E_TEST=1. This is a SKIP, not a pass."
        );
    }
    let device = Device::Cpu;
    let dir = e4b_snapshot_dir().expect("E4B hub snapshot not cached");
    let towers = Gemma4MmTowers::from_model_dir(&dir, &device).expect("load E4B towers");
    let vision_cfg = towers
        .vision
        .as_ref()
        .expect("E4B has vision")
        .config()
        .clone();
    assert!(towers.audio.is_some(), "E4B has audio_config");

    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer.json");
    let tokenize = |s: &str| -> anyhow::Result<Vec<u32>> {
        let enc = tokenizer
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    };

    let messages = json!([
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this image and transcribe the audio."},
            {"type": "image_url", "image_url": {"url": gradient_png_data_url(256, 192)}},
            {"type": "input_audio", "input_audio": {"data": sine_wav_base64(1.0, 440.0, 16000), "format": "wav"}}
        ]}
    ]);
    assert!(messages_have_mm_parts(&messages));
    let parsed = parse_mm_messages(&messages).expect("parse request");
    let segments = segments_from_message(&parsed[0]).expect("decode media");
    let plan = plan_prompt(&towers, &segments, &device, tokenize).expect("plan prompt");

    let expected_img_soft = vision_cfg.compute_num_soft_tokens(256, 192, None);
    assert_eq!(plan.images[0].patches.num_soft_tokens, expected_img_soft);
    assert_eq!(plan.audios[0].num_soft_tokens, 25);
    let img_run = plan.images[0].position;
    let aud_run = plan.audios[0].position;
    assert_eq!(plan.tokens[img_run - 1], GEMMA4_BOI_TOKEN_ID);
    assert_eq!(
        plan.tokens[img_run + expected_img_soft],
        GEMMA4_EOI_TOKEN_ID
    );
    assert_eq!(plan.tokens[aud_run - 1], GEMMA4_BOA_TOKEN_ID);
    assert_eq!(plan.tokens[aud_run + 25], GEMMA4_EOA_TOKEN_ID);

    let weights = nv_weights::WeightLoader::open_file(dir.join("model.safetensors"), &device)
        .expect("open E4B safetensors");
    let embed = weights
        .get("model.language_model.embed_tokens.weight", DType::F32)
        .expect("load embed_tokens");
    assert_eq!(embed.dims()[1], 2560);
    let scale = (2560f64).sqrt();

    let text = embed_prompt(&embed, scale, &plan.tokens).expect("embed text tokens");
    let items = run_towers(&towers, &plan, text.dtype()).expect("run towers");
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].embedding.dims(), &[expected_img_soft, 2560]);
    assert_eq!(items[1].embedding.dims(), &[25, 2560]);

    let out = splice_mm_embeddings(&text, &plan.tokens, &items).expect("splice");
    assert_eq!(out.dims(), &[plan.tokens.len(), 2560]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|x| x.is_finite()));

    let out_rows = rows_bits(&out);
    let text_rows = rows_bits(&text);
    let img_rows = rows_bits(&items[0].embedding);
    let aud_rows = rows_bits(&items[1].embedding);
    for i in 0..plan.tokens.len() {
        if (img_run..img_run + expected_img_soft).contains(&i) {
            assert_eq!(out_rows[i], img_rows[i - img_run], "image soft row {i}");
        } else if (aud_run..aud_run + 25).contains(&i) {
            assert_eq!(out_rows[i], aud_rows[i - aud_run], "audio soft row {i}");
        } else {
            assert_eq!(out_rows[i], text_rows[i], "text row {i}");
        }
    }
    let mean_abs_img: f32 = {
        let v: Vec<f32> = items[0].embedding.flatten_all().unwrap().to_vec1().unwrap();
        v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32
    };
    assert!(mean_abs_img > 1e-5);
    eprintln!(
        "e4b mm e2e OK: seq={} image_soft={} audio_soft=25 out=[{}, 2560] mean|img|={mean_abs_img}",
        plan.tokens.len(),
        expected_img_soft,
        plan.tokens.len()
    );
}

#[test]
fn marked_tokens_plan_matches_segment_plan() {
    use speaches_plus::oapi::chat_multimodal::{decode_image_ref, plan_from_marked_tokens, MmMedia};
    let device = Device::Cpu;
    let towers = Gemma4MmTowers::new("synthetic-e4b", Some(small_vision_tower(&device)), None);
    let url = gradient_png_data_url(96, 96);
    let img = decode_image_ref(&url).unwrap();

    let segments = vec![
        PromptSegment::Text("hi ".into()),
        PromptSegment::Image(img.clone()),
        PromptSegment::Text(" ok".into()),
    ];
    let via_segments = plan_prompt(&towers, &segments, &device, byte_tokenizer).unwrap();

    let mut rendered = byte_tokenizer("hi ").unwrap();
    rendered.push(GEMMA4_BOI_TOKEN_ID);
    rendered.extend(byte_tokenizer(" ok").unwrap());
    let media = MmMedia {
        images: vec![img],
        audios: vec![],
    };
    let via_markers = plan_from_marked_tokens(&towers, &rendered, &media, &device).unwrap();

    assert_eq!(via_markers.tokens, via_segments.tokens);
    assert_eq!(via_markers.images.len(), 1);
    assert_eq!(via_markers.images[0].position, via_segments.images[0].position);
    assert_eq!(
        via_markers.images[0].patches.num_soft_tokens,
        via_segments.images[0].patches.num_soft_tokens
    );
}

#[test]
fn marked_tokens_plan_rejects_media_count_mismatch() {
    use speaches_plus::oapi::chat_multimodal::{decode_image_ref, plan_from_marked_tokens, MmMedia};
    let device = Device::Cpu;
    let towers = Gemma4MmTowers::new("synthetic-e4b", Some(small_vision_tower(&device)), None);
    let img = decode_image_ref(&gradient_png_data_url(96, 96)).unwrap();

    let mut extra_marker = byte_tokenizer("a").unwrap();
    extra_marker.push(GEMMA4_BOI_TOKEN_ID);
    extra_marker.push(GEMMA4_BOI_TOKEN_ID);
    let one_image = MmMedia {
        images: vec![img.clone()],
        audios: vec![],
    };
    let err = format!(
        "{:#}",
        plan_from_marked_tokens(&towers, &extra_marker, &one_image, &device).unwrap_err()
    );
    assert!(err.contains("more image markers"), "unexpected error: {err}");

    let no_marker = byte_tokenizer("a").unwrap();
    let err = format!(
        "{:#}",
        plan_from_marked_tokens(&towers, &no_marker, &one_image, &device).unwrap_err()
    );
    assert!(
        err.contains("more image_url parts"),
        "unexpected error: {err}"
    );
}
