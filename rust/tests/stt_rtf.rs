use std::path::PathBuf;
use std::time::Instant;

use speaches_plus::stt::{Backend, WhisperEngine};

const ENV_GATE: &str = "STT_RTF_TEST";

const ENV_MODEL_DIR: &str = "STT_RTF_MODEL_DIR";

const ENV_WAV: &str = "STT_RTF_WAV";

const ENV_RTF_FLOOR: &str = "STT_RTF_FLOOR";

const RTF_FLOOR_WITHOUT_ENV_IS_SANITY_ONLY: f32 = 1.0;

const WARMUP_RUNS: usize = 1;

const MEASURED_RUNS: usize = 3;

const MIN_TRANSCRIPT_CHARS_PER_AUDIO_SECOND: f32 = 2.0;

fn required(var: &str, hint: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {var}={hint}")
    })
}

fn load_wav_16k_mono(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "{path} must be 16 kHz");
    assert_eq!(spec.channels, 1, "{path} must be mono");
    match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.expect("sample") as f32 / i16::MAX as f32)
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.expect("sample")).collect(),
    }
}

#[test]
#[ignore = "real whisper weights: set STT_RTF_TEST=1, STT_RTF_MODEL_DIR, STT_RTF_WAV"]
fn stt_rtf_meets_floor() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn,speaches_plus=debug"))
        .with_test_writer()
        .try_init();
    assert_eq!(
        std::env::var(ENV_GATE).ok().as_deref(),
        Some("1"),
        "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {ENV_GATE}=1"
    );
    let model_dir = PathBuf::from(required(ENV_MODEL_DIR, "<dir with whisper-ct2/ or ggml-*.bin>"));
    let wav = required(ENV_WAV, "<16k mono wav>");
    let floor: f32 = std::env::var(ENV_RTF_FLOOR)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(RTF_FLOOR_WITHOUT_ENV_IS_SANITY_ONLY);

    let audio = load_wav_16k_mono(&wav);
    let audio_secs = audio.len() as f32 / 16_000.0;

    let t_load = Instant::now();
    let engine = WhisperEngine::load(&model_dir)
        .unwrap_or_else(|e| panic!("whisper load ({:?}): {e:#}", Backend::from_env()));
    let handle = engine.handle();
    let load_secs = t_load.elapsed().as_secs_f32();

    for _ in 0..WARMUP_RUNS {
        handle
            .transcribe(&audio)
            .unwrap_or_else(|e| panic!("warmup transcribe failed: {e:#}"));
    }

    let mut best_secs = f32::INFINITY;
    let mut text = String::new();
    for _ in 0..MEASURED_RUNS {
        let t0 = Instant::now();
        text = handle
            .transcribe(&audio)
            .unwrap_or_else(|e| panic!("transcribe failed: {e:#}"));
        best_secs = best_secs.min(t0.elapsed().as_secs_f32());
    }
    let rtf = audio_secs / best_secs.max(1e-6);

    eprintln!(
        "stt rtf    : backend={:?} audio={audio_secs:.1}s best_gen={best_secs:.3}s \
         rtf={rtf:.1}x chars={} (load {load_secs:.1}s, {MEASURED_RUNS} runs)\ntext       : {}",
        Backend::from_env(),
        text.len(),
        &text[..text.len().min(120)],
    );

    assert!(
        text.len() as f32 >= MIN_TRANSCRIPT_CHARS_PER_AUDIO_SECOND * audio_secs,
        "transcript suspiciously short ({} chars for {audio_secs:.0}s): {text:?}",
        text.len()
    );
    assert!(
        rtf >= floor,
        "stt rtf {rtf:.1}x is below the floor {floor:.1}x ({ENV_RTF_FLOOR})"
    );
}
