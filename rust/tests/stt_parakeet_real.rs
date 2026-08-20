use std::time::Instant;

use speaches_plus::stt::parakeet::{parakeet_dir, ParakeetTdt};

fn read_wav_16k_mono(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("open test wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "fixture must be 16k");
    assert_eq!(spec.channels, 1, "fixture must be mono");
    reader
        .samples::<i16>()
        .map(|s| s.expect("wav sample") as f32 / 32768.0)
        .collect()
}

#[test]
#[ignore]
fn parakeet_transcribes_the_committed_espeak_clip_in_realtime() {
    if std::env::var("NV_PARAKEET_TEST").as_deref() != Ok("1") {
        eprintln!(
            "SKIP parakeet_transcribes_the_committed_espeak_clip_in_realtime: set \
             NV_PARAKEET_TEST=1 (needs the istupakov/parakeet-tdt-0.6b-v2-onnx snapshot; \
             ~2.5GB, CPU-only)"
        );
        return;
    }
    let dir = parakeet_dir().expect("parakeet snapshot present");
    let t_load = Instant::now();
    let model = ParakeetTdt::load(&dir).expect("load parakeet");
    eprintln!("[parakeet] load {:?}", t_load.elapsed());

    let audio = read_wav_16k_mono(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/parakeet_fox_16k.wav"
    ));
    let secs = audio.len() as f64 / 16_000.0;
    let t0 = Instant::now();
    let text = model.transcribe(&audio).expect("transcribe");
    let wall = t0.elapsed().as_secs_f64();
    eprintln!(
        "[parakeet] {secs:.1}s audio in {wall:.2}s -> x{:.1} realtime: {text:?}",
        secs / wall
    );
    let lower = text.to_lowercase();
    for needle in ["quick brown fox", "lazy dog", "pelicans", "harbor wall"] {
        assert!(
            lower.contains(needle),
            "transcript must contain {needle:?}, got {text:?}"
        );
    }
    assert!(
        secs / wall > 1.0,
        "CPU decode slower than realtime: {wall:.2}s for {secs:.1}s"
    );
}
