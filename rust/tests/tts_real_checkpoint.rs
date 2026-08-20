use std::path::PathBuf;
use std::time::Instant;

use speaches_plus::oapi::audio_speech::{AudioSpeech, NV_TTS_SAMPLE_RATE};
use speaches_plus::oapi::audio_speech_nvtts::Qwen3TtsAudioSpeech;

const ENV_GATE: &str = "NV_TTS_REAL_TEST";

const ENV_DIR: &str = "NV_TTS_REAL_DIR";

const ENV_WAV_OUT: &str = "NV_TTS_REAL_WAV";

const ENV_VOICE: &str = "NV_TTS_REAL_VOICE";

const DEFAULT_VOICE: &str = "default";

const SENTENCE: &str =
    "The quick brown fox jumps over the lazy dog while the harbour bell rings twice.";

const SENTENCE_WORDS: f32 = 14.0;

const CODEC_FRAME_SAMPLES: usize = 1920;

const MIN_SECONDS_PER_WORD: f32 = 0.10;

const MAX_SECONDS_PER_WORD: f32 = 1.20;

const MIN_PEAK: f32 = 0.02;

const MIN_RMS: f32 = 1.0e-3;

const LOOP_BLOCK_SAMPLES: usize = 12_000;

const LOOP_CORRELATION_CEILING: f32 = 0.98;

const LOOP_BLOCK_RMS_FLOOR_RATIO: f32 = 0.25;

fn checkpoint_dir() -> PathBuf {
    assert_eq!(
        std::env::var(ENV_GATE).ok().as_deref(),
        Some("1"),
        "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {ENV_GATE}=1"
    );
    let raw = std::env::var(ENV_DIR).unwrap_or_else(|_| {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {ENV_DIR}=<snapshot dir>")
    });
    let dir = PathBuf::from(raw);
    assert!(
        dir.join("model.safetensors").is_file(),
        "{ENV_DIR}={} has no model.safetensors",
        dir.display()
    );
    assert!(
        dir.join("speech_tokenizer/model.safetensors").is_file(),
        "{ENV_DIR}={} has no speech_tokenizer/model.safetensors",
        dir.display()
    );
    dir
}

fn block_correlation(pcm: &[f32]) -> (f32, usize, usize) {
    let blocks: Vec<&[f32]> = pcm.chunks_exact(LOOP_BLOCK_SAMPLES).collect();
    let energies: Vec<f32> = blocks
        .iter()
        .map(|b| (b.iter().map(|s| s * s).sum::<f32>() / b.len() as f32).sqrt())
        .collect();
    let global = (pcm.iter().map(|s| (s * s) as f64).sum::<f64>() / pcm.len().max(1) as f64).sqrt()
        as f32;
    let floor = global * LOOP_BLOCK_RMS_FLOOR_RATIO;
    let mut worst = 0.0f32;
    let mut worst_pair = (0usize, 0usize);
    for i in 0..blocks.len() {
        if energies[i] < floor {
            continue;
        }
        for j in (i + 1)..blocks.len() {
            if energies[j] < floor {
                continue;
            }
            let dot: f32 = blocks[i]
                .iter()
                .zip(blocks[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            let na: f32 = blocks[i].iter().map(|a| a * a).sum::<f32>().sqrt();
            let nb: f32 = blocks[j].iter().map(|b| b * b).sum::<f32>().sqrt();
            if na <= 0.0 || nb <= 0.0 {
                continue;
            }
            let c = (dot / (na * nb)).abs();
            if c > worst {
                worst = c;
                worst_pair = (i, j);
            }
        }
    }
    (worst, worst_pair.0, worst_pair.1)
}

fn write_wav(path: &str, pcm: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: NV_TTS_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).expect("create wav");
    for s in pcm {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        w.write_sample(v).expect("write sample");
    }
    w.finalize().expect("finalize wav");
}

#[test]
#[ignore = "real Qwen3-TTS weights on CPU: set NV_TTS_REAL_TEST=1 and NV_TTS_REAL_DIR"]
fn real_checkpoint_speaks_without_looping() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn,speaches_plus=debug"))
        .with_test_writer()
        .try_init();
    let dir = checkpoint_dir();
    let voice = std::env::var(ENV_VOICE).unwrap_or_else(|_| DEFAULT_VOICE.to_string());

    let t_load = Instant::now();
    let svc = Qwen3TtsAudioSpeech::from_dirs_with(&dir, None, false)
        .unwrap_or_else(|e| panic!("{} failed to load: {e:#}", dir.display()));
    let load_secs = t_load.elapsed().as_secs_f32();

    assert!(
        svc.vocoder_inventory.is_real_qwen3_decoder(),
        "vocoder inventory is not a real Qwen3 decoder: {:?}",
        svc.vocoder_inventory
    );
    assert!(
        !svc.vocoder_report.zero_init_fallback,
        "zero-init vocoder fallback: {:?}",
        svc.vocoder_report.fallback_reason
    );
    assert!(!svc.vocoder.is_zero_init());
    let talker = svc.talker.as_ref().expect("talker must load");
    assert!(talker.has_text_embedding(), "text embedding must load");
    let hidden = talker.config().hidden_size;
    let layers = talker.config().num_hidden_layers;
    let speakers: Vec<String> = talker
        .config()
        .spk_id
        .iter()
        .map(|(n, _)| n.clone())
        .collect();

    eprintln!(
        "checkpoint : {}\nmodel_id   : {}\nload_secs  : {load_secs:.2}\ntalker     : hidden={hidden} layers={layers} profiles_supported={}\nspeakers   : {} ({:?})\nvocoder    : decoder_keys={} upsample={} sample_rate={}",
        dir.display(),
        svc.talker_model_id(),
        svc.profiles_supported,
        speakers.len(),
        speakers,
        svc.vocoder_inventory.decoder_key_count,
        svc.vocoder_inventory.upsample_factor,
        svc.vocoder_inventory.sample_rate,
    );

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("rt");

    if svc.code_predictor.is_none() {
        let err = rt
            .block_on(svc.synthesize(SENTENCE, &voice))
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_else(|| "synthesize SUCCEEDED without a code predictor".to_string());
        panic!(
            "code_predictor did not load for {}; synthesize refuses with: {err}",
            dir.display()
        );
    }

    let t_gen = Instant::now();
    let pcm = rt.block_on(async {
        let mut rx = svc
            .synthesize(SENTENCE, &voice)
            .await
            .unwrap_or_else(|e| panic!("synthesize failed: {e:#}"));
        let mut all: Vec<f32> = Vec::new();
        while let Some(chunk) = rx.recv().await {
            all.extend_from_slice(&chunk);
        }
        all
    });
    let gen_secs = t_gen.elapsed().as_secs_f32();

    let frames = pcm.len() / CODEC_FRAME_SAMPLES;
    let audio_secs = pcm.len() as f32 / NV_TTS_SAMPLE_RATE as f32;
    let peak = pcm.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let rms =
        (pcm.iter().map(|s| (s * s) as f64).sum::<f64>() / pcm.len().max(1) as f64).sqrt() as f32;
    let (corr, bi, bj) = block_correlation(&pcm);

    eprintln!(
        "voice      : {voice}\nsamples    : {} ({audio_secs:.2} s, {frames} codec frames)\ngen_secs   : {gen_secs:.2} (rtf {:.2}x realtime, {:.1} codec frames/s)\npeak       : {peak:.4}\nrms        : {rms:.4e}\nloop_corr  : {corr:.4} (blocks {bi} vs {bj}, {} blocks of {:.2}s)",
        pcm.len(),
        audio_secs / gen_secs.max(1e-6),
        frames as f32 / gen_secs.max(1e-6),
        pcm.len() / LOOP_BLOCK_SAMPLES,
        LOOP_BLOCK_SAMPLES as f32 / NV_TTS_SAMPLE_RATE as f32,
    );

    if let Ok(p) = std::env::var(ENV_WAV_OUT) {
        write_wav(&p, &pcm);
        eprintln!("wav        : {p}");
    }

    assert!(peak > MIN_PEAK, "near-silent output: peak={peak}");
    assert!(rms > MIN_RMS, "near-silent output: rms={rms}");
    assert!(
        audio_secs > SENTENCE_WORDS * MIN_SECONDS_PER_WORD,
        "output too short for {SENTENCE_WORDS} words: {audio_secs:.2} s"
    );
    assert!(
        audio_secs < SENTENCE_WORDS * MAX_SECONDS_PER_WORD,
        "output ran long for {SENTENCE_WORDS} words ({audio_secs:.2} s, {frames} frames): the \
         talker did not reach EOS, which is the repetition-loop signature"
    );
    assert!(
        corr < LOOP_CORRELATION_CEILING,
        "two 0.5 s blocks ({bi}, {bj}) correlate at {corr:.4}: repetition loop"
    );
}
