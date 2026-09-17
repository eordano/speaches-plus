use std::path::Path;

use super::stages;
use super::workspace::{producer, ArtifactRef, Workspace};

fn wav(path: &Path, freq: f32) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for i in 0..24_000 {
        let t = i as f32 / 24_000.0;
        w.write_sample((0.2 * (2.0 * std::f32::consts::PI * freq * t).sin() * 32767.0) as i16)
            .unwrap();
    }
    w.finalize().unwrap();
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

fn diar(speakers: &[(&str, Vec<f32>, f64)]) -> serde_json::Value {
    let mut segs = Vec::new();
    let mut t = 0.0;
    for i in 0..12 {
        let (lab, _, _) = &speakers[i % speakers.len()];
        segs.push(
            serde_json::json!({"start": t, "end": t + 20.0, "speaker": lab, "confidence": 0.9}),
        );
        t += 20.0;
    }
    serde_json::json!({"duration": 600.0, "segments": segs,
        "speakers": speakers.iter().map(|(l, e, s)| serde_json::json!({"speaker": l, "seconds": s, "turns": 6, "embedding": e})).collect::<Vec<_>>()})
}

fn tmp(tag: &str) -> stages::TempDir {
    stages::TempDir::new(tag).unwrap()
}

#[test]
fn scan_follows_symlinked_trees_skips_transcript_dirs_and_is_content_addressed() {
    let td = tmp("scan");
    let ws = Workspace::new(td.0.join("ws"));
    ws.init().unwrap();
    let pods = td.0.join("podcasts");
    let meets = td.0.join("meetings");
    wav(&pods.join("Show/2026-01-01-ep.wav"), 220.0);
    wav(&meets.join("2026.08/2026.08.03-weekly.wav"), 330.0);
    wav(&meets.join("2026.08/2026.08.03-weekly/chunk-1.wav"), 440.0);
    wav(&meets.join("2026.08/.hidden/x.wav"), 550.0);
    std::fs::remove_dir(ws.recordings_dir()).unwrap();
    std::fs::create_dir(ws.recordings_dir()).unwrap();
    std::os::unix::fs::symlink(&pods, ws.recordings_dir().join("podcasts")).unwrap();
    std::os::unix::fs::symlink(&meets, ws.recordings_dir().join("meetings")).unwrap();
    let mut m = ws.load().unwrap();
    let (added, updated) = ws.scan(&mut m, false).unwrap();
    assert_eq!((added, updated), (2, 0));
    let paths: Vec<String> = m.by_path().iter().map(|r| r.path.clone()).collect();
    assert_eq!(
        paths,
        [
            "meetings/2026.08/2026.08.03-weekly.wav",
            "podcasts/Show/2026-01-01-ep.wav"
        ]
    );
    let ep = m.by_path()[1].clone();
    assert_eq!(
        (
            ep.podcast.as_deref(),
            ep.date.as_deref(),
            ep.title.as_deref()
        ),
        (Some("Show"), Some("2026-01-01"), Some("ep"))
    );
    assert_eq!(m.by_path()[0].podcast.as_deref(), Some("2026.08"));

    ws.put_artifact(
        &mut m,
        &ep.sha256,
        "transcript",
        &serde_json::json!({"text": "hi"}),
    )
    .unwrap();
    std::fs::rename(
        pods.join("Show/2026-01-01-ep.wav"),
        pods.join("Show/renamed.wav"),
    )
    .unwrap();
    let mut m = ws.load().unwrap();
    let (added, updated) = ws.scan(&mut m, false).unwrap();
    assert_eq!((added, updated), (0, 1), "a renamed file keeps its row");
    let r = &m.rows[&ep.sha256];
    assert_eq!(r.path, "podcasts/Show/renamed.wav");
    assert!(ws.has_artifact(r, "transcript") && r.current("transcript"));
}

#[test]
fn stale_producer_is_reported_not_wiped() {
    let td = tmp("stale");
    let ws = Workspace::new(td.0.join("ws"));
    ws.init().unwrap();
    wav(&ws.recordings_dir().join("a.wav"), 220.0);
    let mut m = ws.load().unwrap();
    ws.scan(&mut m, false).unwrap();
    let sha = m.by_path()[0].sha256.clone();
    ws.put_artifact(
        &mut m,
        &sha,
        "diar",
        &diar(&[("S1", unit(&[1.0, 0.0, 0.0]), 200.0)]),
    )
    .unwrap();
    m.rows
        .get_mut(&sha)
        .unwrap()
        .artifacts
        .get_mut("diar")
        .unwrap()
        .producer = "old/v0".into();
    ws.save(&m).unwrap();
    let m = ws.load().unwrap();
    let st = ws.status(&m).unwrap();
    assert_eq!(st["diar"]["stale"], 1);
    assert_eq!(st["diar"]["done"], 1);
    assert!(m.rows[&sha].stale("diar"));
    assert_eq!(
        producer("diar"),
        "diarizen-16s-hop0.1+wespeaker-resnet293/v1"
    );
    let _ = ArtifactRef {
        sha256: String::new(),
        producer: String::new(),
        bytes: 0,
        created: String::new(),
    };
}

#[test]
fn identify_links_shared_speaker_and_proposes_host_then_clips_flag_overlap() {
    let td = tmp("identify");
    let ws = Workspace::new(td.0.join("ws"));
    ws.init().unwrap();
    let host = unit(&[1.0, 0.1, 0.0]);
    let guests = [
        unit(&[0.0, 1.0, 0.0]),
        unit(&[0.0, 0.0, 1.0]),
        unit(&[0.5, -0.5, 0.5]),
    ];
    for i in 0..guests.len() {
        wav(
            &ws.recordings_dir()
                .join(format!("Pod/2026-01-0{}-ep.wav", i + 1)),
            200.0 + i as f32 * 50.0,
        );
    }
    let mut m = ws.load().unwrap();
    ws.scan(&mut m, false).unwrap();
    let shas: Vec<String> = m.by_path().iter().map(|r| r.sha256.clone()).collect();
    for (i, sha) in shas.iter().enumerate() {
        ws.put_artifact(
            &mut m,
            sha,
            "diar",
            &diar(&[
                ("S0", host.clone(), 300.0),
                ("S1", guests[i].clone(), 280.0),
            ]),
        )
        .unwrap();
    }
    ws.put_artifact(&mut m, &shas[0], "transcript", &serde_json::json!({"segments": [
        {"start": 0.0, "end": 19.0, "text": "Host words here."}, {"start": 20.0, "end": 39.0, "text": "Guest words here."}]})).unwrap();
    let mut log = |_s: String| {};
    let reg = stages::identify(&ws, &m, 60.0, &mut log).unwrap();
    assert_eq!(
        reg.speakers.len(),
        4,
        "one host across 3 recordings plus 3 guests"
    );
    let h = reg
        .speakers
        .iter()
        .find(|s| s.sources.len() == 3)
        .expect("host linked thrice");
    assert_eq!(h.role.as_deref(), Some("host"));
    assert_eq!(h.name_proposed.as_deref(), Some("Pod host"));
    let reg2 = stages::identify(&ws, &m, 60.0, &mut log).unwrap();
    assert_eq!(reg2.speakers.len(), 4, "identify is idempotent");

    let n = stages::clips(&ws, &mut m, 4.0, &mut log).unwrap();
    assert_eq!(n, 3);
    let c = ws.get_artifact(&shas[0], "clips").unwrap();
    let turns = c["clips"].as_array().unwrap();
    assert_eq!(turns.len(), 12);
    assert!(
        turns.iter().all(|t| t["clean"].as_bool().unwrap()),
        "alternating turns never overlap"
    );
    assert_eq!(turns[0]["text_aligned"], "Host words here.");
    assert_eq!(turns[1]["text_aligned"], "Guest words here.");
    let hc = stages::speaker_clips(&ws, &m, &reg, &h.id, 8.0).unwrap();
    assert_eq!(hc.len(), 18);
    assert_eq!(ws.status(&m).unwrap()["clips"]["done"], 3);
}

#[test]
fn pick_sentences_keeps_mid_length_capitalised_sentences_without_years() {
    let texts = vec![
        "The data center build out this year is bigger than anything the industry has done before. In 2024 we saw nothing like it at all in the market. short one. lowercase sentence that is long enough to pass the word count filter easily.".to_string(),
    ];
    let out = stages::pick_sentences(&texts, 10);
    assert_eq!(out, ["The data center build out this year is bigger than anything the industry has done before."]);
}

#[test]
fn chunk_merging_shifts_segment_nested_and_top_level_words() {
    let mut merged = serde_json::json!({"text": "", "segments": []});
    let chunk = serde_json::json!({"text": "Hello", "segments": [{
        "start": 0.5, "end": 2.0, "text": "Hello", "words": [{"start": 0.5, "end": 1.5, "word": "Hello"}]
    }], "words": [{"start": 0.5, "end": 1.5, "word": "Hello"}]});
    stages::merge_transcript(&mut merged, &chunk, 0.0);
    stages::merge_transcript(&mut merged, &chunk, 1500.0);
    assert_eq!(merged["text"], "Hello Hello");
    assert_eq!(merged["segments"][1]["start"], 1500.5);
    assert_eq!(merged["segments"][1]["words"][0]["end"], 1501.5);
    assert_eq!(merged["words"][1]["start"], 1500.5);
    assert_eq!(merged["segments"][0]["words"][0]["start"], 0.5);
}

#[test]
fn scan_retains_successes_when_another_file_disappears() {
    let td = tmp("scan-failure");
    let ws = Workspace::new(&td.0);
    ws.init().unwrap();
    let good = ws.recordings_dir().join("good.wav");
    wav(&good, 220.0);
    let missing = ws.recordings_dir().join("vanished.wav");
    let unreadable = ws.recordings_dir().join("directory.wav");
    std::fs::create_dir(&unreadable).unwrap();
    let mut m = ws.load().unwrap();
    assert_eq!(
        ws.scan_files(&mut m, false, vec![missing, unreadable, good])
            .unwrap(),
        (1, 0)
    );
    assert_eq!(ws.load().unwrap().rows.len(), 1);
}

#[test]
fn rename_waits_for_registry_transaction_and_preserves_identify_updates() {
    use super::workspace::{Registry, Speaker, REGISTRY_GATE};
    let td = tmp("registry-race");
    let ws = Workspace::new(&td.0);
    ws.init().unwrap();
    let mut registry = Registry::default();
    registry.speakers.push(Speaker {
        id: "spk-0001".into(),
        name: None,
        name_proposed: None,
        role: None,
        centroid: vec![1., 0.],
        seconds: 60.,
        sources: vec![],
        extra: Default::default(),
    });
    let guard = REGISTRY_GATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    ws.save_speakers(&mut registry).unwrap();
    let other = ws.clone();
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        let result = other.rename_speaker("spk-0001", "Confirmed Name");
        done_tx.send(result).unwrap();
    });
    started_rx.recv().unwrap();
    assert!(matches!(
        done_rx.recv_timeout(std::time::Duration::from_millis(30)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
    ));
    registry.speakers[0].seconds = 120.;
    registry.speakers[0].role = Some("host".into());
    ws.save_speakers(&mut registry).unwrap();
    drop(guard);
    done_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap()
        .unwrap();
    worker.join().unwrap();
    let registry = ws.load_speakers().unwrap();
    assert_eq!(registry.speakers[0].name.as_deref(), Some("Confirmed Name"));
    assert_eq!(registry.speakers[0].seconds, 120.);
    assert_eq!(registry.speakers[0].role.as_deref(), Some("host"));
}

#[tokio::test]
async fn transcribe_real_wav_chunks_and_continues_after_a_bad_recording() {
    let app = axum::Router::new().route(
        "/",
        axum::routing::post(|| async {
            axum::Json(serde_json::json!({
                "text": "A test sentence.",
                "segments": [{"start": 0.0, "end": 0.25, "text": "A test sentence.",
                    "words": [{"start": 0.0, "end": 0.25, "word": "test"}]}],
                "words": [{"start": 0.0, "end": 0.25, "word": "test"}]
            }))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let td = tmp("transcribe-contract");
    let ws = Workspace::new(&td.0);
    ws.init().unwrap();
    std::fs::write(ws.recordings_dir().join("a-bad.wav"), b"not audio").unwrap();
    std::fs::write(
        ws.recordings_dir().join("b-good.wav"),
        include_bytes!("../../tests/data/parakeet_fox_16k.wav"),
    )
    .unwrap();
    let mut m = ws.load().unwrap();
    ws.scan(&mut m, false).unwrap();
    let mut log = Vec::new();
    let count = stages::transcribe(
        &ws,
        &mut m,
        &reqwest::Client::new(),
        &format!("http://{addr}/"),
        0.05,
        &mut |s| log.push(s),
    )
    .await
    .unwrap();
    server.abort();
    assert_eq!(count, 1);
    assert!(log
        .iter()
        .any(|s| s.contains("transcribe failed for a-bad.wav")));
    let good = m
        .by_path()
        .into_iter()
        .find(|r| r.path == "b-good.wav")
        .unwrap();
    let result = ws.get_artifact(&good.sha256, "transcript").unwrap();
    assert_eq!(result["segments"].as_array().unwrap().len(), 3);
    assert_eq!(result["segments"][2]["words"][0]["end"], 6.25);
    assert_eq!(result["words"][2]["start"], 6.0);
    let persisted = ws.load().unwrap();
    assert_eq!(
        persisted
            .rows
            .values()
            .filter(|r| ws.has_artifact(r, "transcript"))
            .count(),
        1
    );
}
