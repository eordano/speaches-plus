use std::path::PathBuf;
use std::time::Instant;

use speaches_plus::tts;

const ENV_GATE: &str = "KOKORO_RTF_TEST";

const ENV_MODEL_DIR: &str = "KOKORO_RTF_MODEL_DIR";

const ENV_RTF_FLOOR: &str = "KOKORO_RTF_FLOOR";

const RTF_FLOOR_WITHOUT_ENV_IS_SANITY_ONLY: f32 = 1.0;

const WARMUP_RUNS: usize = 2;

const MEASURED_RUNS: usize = 5;

const SAMPLE_RATE: f32 = 24_000.0;

const SHORT: &str = "The quick brown fox jumps over the lazy dog.";

const LONG: &str = "The quick brown fox jumps over the lazy dog, while the harbour \
bell rings twice and the sailors wait for the morning tide to turn before setting \
out across the wide grey water with their nets and lanterns. A cold wind moves \
through the rigging, and somewhere beyond the breakwater a gull repeats its one \
hoarse question to the empty pier.";

fn model_dir() -> PathBuf {
    assert_eq!(
        std::env::var(ENV_GATE).ok().as_deref(),
        Some("1"),
        "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {ENV_GATE}=1"
    );
    let raw = std::env::var(ENV_MODEL_DIR).unwrap_or_else(|_| {
        panic!(
            "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {ENV_MODEL_DIR}=<dir with \
             kokoro-v1.0.onnx + voices.bin>"
        )
    });
    let dir = PathBuf::from(raw);
    assert!(
        dir.join("kokoro-v1.0.onnx").is_file(),
        "{ENV_MODEL_DIR}={} has no kokoro-v1.0.onnx",
        dir.display()
    );
    dir
}

#[test]
#[ignore = "real Kokoro ONNX weights: set KOKORO_RTF_TEST=1 and KOKORO_RTF_MODEL_DIR"]
fn kokoro_rtf_meets_floor() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn,speaches_plus=debug"))
        .with_test_writer()
        .try_init();
    let dir = model_dir();
    let floor: f32 = std::env::var(ENV_RTF_FLOOR)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(RTF_FLOOR_WITHOUT_ENV_IS_SANITY_ONLY);

    let t_load = Instant::now();
    let handle = tts::prepared_handle(&dir)
        .unwrap_or_else(|e| panic!("kokoro load failed: {e:#}"))
        .unwrap_or_else(|| panic!("no kokoro model in {}", dir.display()));
    let load_secs = t_load.elapsed().as_secs_f32();

    for _ in 0..WARMUP_RUNS {
        handle
            .synthesize(SHORT, None, None, 1.0)
            .unwrap_or_else(|e| panic!("warmup synth failed: {e:#}"));
    }

    let mut worst_rtf = f32::INFINITY;
    for (label, text) in [("short", SHORT), ("long", LONG)] {
        let mut best_secs = f32::INFINITY;
        let mut audio_secs = 0.0f32;
        for _ in 0..MEASURED_RUNS {
            let t0 = Instant::now();
            let audio = handle
                .synthesize(text, None, None, 1.0)
                .unwrap_or_else(|e| panic!("synth {label} failed: {e:#}"));
            let gen = t0.elapsed().as_secs_f32();
            best_secs = best_secs.min(gen);
            audio_secs = audio.len() as f32 / SAMPLE_RATE;
        }
        let rtf = audio_secs / best_secs.max(1e-6);
        worst_rtf = worst_rtf.min(rtf);
        eprintln!(
            "kokoro rtf : {label:5} audio={audio_secs:6.2}s best_gen={best_secs:6.3}s \
             rtf={rtf:6.1}x (load {load_secs:.1}s, {MEASURED_RUNS} runs)"
        );
    }

    assert!(
        worst_rtf >= floor,
        "kokoro rtf {worst_rtf:.1}x is below the floor {floor:.1}x ({ENV_RTF_FLOOR})"
    );
}
