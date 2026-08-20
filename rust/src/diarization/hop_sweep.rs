use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};

use super::{DiarConfig, DiarSegment, Diarizer, EmbeddingModel, SegmentationModel};

const FIXTURE_REL: &str = "../conformance/fixtures/050-diarization-multispeaker";
const AUDIO_FILE: &str = "audio.wav";
const TRUTH_FILE: &str = "ground_truth.json";
const FIXTURE_SAMPLE_RATE: u32 = 16_000;
const KOKORO_SAMPLE_RATE: usize = 24_000;
const DEFAULT_RATIOS: &str = "0.1,0.2,0.25,0.5";
const MATCH_WINDOW_MS: i64 = 500;

const SPEAKERS: [(&str, &str); 4] = [
    ("SPK_A", "af_heart"),
    ("SPK_B", "am_michael"),
    ("SPK_C", "bf_emma"),
    ("SPK_D", "bm_george"),
];

struct ScriptTurn {
    speaker: usize,
    lead_ms: i64,
    text: &'static str,
}

const SCRIPT: [ScriptTurn; 12] = [
    ScriptTurn {
        speaker: 0,
        lead_ms: 0,
        text: "Good morning everyone, thanks for joining the weekly sync. I want to start with the deployment status before we move on to anything else.",
    },
    ScriptTurn {
        speaker: 1,
        lead_ms: 260,
        text: "The rollout finished last night, and all three regions are reporting green.",
    },
    ScriptTurn {
        speaker: 2,
        lead_ms: -350,
        text: "Nice work.",
    },
    ScriptTurn {
        speaker: 3,
        lead_ms: 180,
        text: "I still see one alert sitting on the Berlin cluster, though.",
    },
    ScriptTurn {
        speaker: 1,
        lead_ms: 150,
        text: "Which alert?",
    },
    ScriptTurn {
        speaker: 3,
        lead_ms: 120,
        text: "Disk pressure on the build node. It has been climbing steadily since Friday afternoon.",
    },
    ScriptTurn {
        speaker: 0,
        lead_ms: -900,
        text: "That will be the log volume again.",
    },
    ScriptTurn {
        speaker: 2,
        lead_ms: 220,
        text: "I can clean up the stale artifacts this afternoon and reclaim most of it.",
    },
    ScriptTurn {
        speaker: 1,
        lead_ms: 160,
        text: "Thanks.",
    },
    ScriptTurn {
        speaker: 3,
        lead_ms: 200,
        text: "While you are in there, please check the retention policy too.",
    },
    ScriptTurn {
        speaker: 2,
        lead_ms: 140,
        text: "Will do.",
    },
    ScriptTurn {
        speaker: 0,
        lead_ms: 280,
        text: "Great. Then let us move on to the next item, which is the tokenizer migration and what it means for the release train.",
    },
];

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TruthTurn {
    index: usize,
    speaker: String,
    voice: String,
    t_start_ms: u64,
    t_end_ms: u64,
    text: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TruthOverlap {
    turn_a: usize,
    turn_b: usize,
    t_start_ms: u64,
    t_end_ms: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct GroundTruth {
    source: String,
    sample_rate: u32,
    duration_ms: u64,
    speakers: Vec<(String, String)>,
    turns: Vec<TruthTurn>,
    overlaps: Vec<TruthOverlap>,
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str) -> Option<f32> {
    std::env::var(key).ok().and_then(|s| s.parse::<f32>().ok())
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_REL)
}

fn sweep_audio_path() -> PathBuf {
    env_path("DIAR_SWEEP_AUDIO").unwrap_or_else(|| fixture_dir().join(AUDIO_FILE))
}

fn sweep_truth_path(audio: &std::path::Path) -> Option<PathBuf> {
    if let Some(p) = env_path("DIAR_SWEEP_TRUTH") {
        return Some(p);
    }
    let sibling = audio.with_file_name(TRUTH_FILE);
    if sibling.exists() {
        return Some(sibling);
    }
    None
}

fn ratios() -> Vec<f32> {
    std::env::var("DIAR_SWEEP_RATIOS")
        .unwrap_or_else(|_| DEFAULT_RATIOS.to_string())
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .filter(|v| *v > 0.0 && *v <= 1.0)
        .collect()
}

fn load_models() -> Result<(SegmentationModel, EmbeddingModel)> {
    let seg_path = env_path("DIAR_SEGMENTATION_MODEL")
        .ok_or_else(|| anyhow!("DIAR_SEGMENTATION_MODEL not set or file missing"))?;
    let emb_path = env_path("DIAR_EMBEDDING_MODEL")
        .ok_or_else(|| anyhow!("DIAR_EMBEDDING_MODEL not set or file missing"))?;
    let t = Instant::now();
    let seg = SegmentationModel::load(&seg_path)?;
    eprintln!("segmentation load : {:.0} ms", ms(t.elapsed()));
    let t = Instant::now();
    let emb = EmbeddingModel::load(&emb_path)?;
    eprintln!("embedding load    : {:.0} ms", ms(t.elapsed()));
    Ok((seg, emb))
}

fn boundaries(segs: &[DiarSegment]) -> Vec<i64> {
    let mut set: BTreeSet<i64> = BTreeSet::new();
    for s in segs {
        set.insert(s.t_start_ms as i64);
        set.insert(s.t_end_ms as i64);
    }
    set.into_iter().collect()
}

fn nearest(sorted: &[i64], v: i64) -> Option<i64> {
    if sorted.is_empty() {
        return None;
    }
    let idx = sorted.partition_point(|&x| x < v);
    let mut best = i64::MAX;
    for i in idx.saturating_sub(1)..(idx + 1).min(sorted.len()) {
        best = best.min((sorted[i] - v).abs());
    }
    if idx < sorted.len() {
        best = best.min((sorted[idx] - v).abs());
    }
    Some(best)
}

struct BoundaryDelta {
    mean_ms: f64,
    max_ms: i64,
    matched: usize,
    ref_unmatched: usize,
    cand_extra: usize,
}

fn boundary_delta(reference: &[i64], candidate: &[i64]) -> BoundaryDelta {
    let mut sum = 0f64;
    let mut max = 0i64;
    let mut matched = 0usize;
    let mut ref_unmatched = 0usize;
    for &r in reference {
        match nearest(candidate, r) {
            Some(d) if d <= MATCH_WINDOW_MS => {
                sum += d as f64;
                max = max.max(d);
                matched += 1;
            }
            _ => ref_unmatched += 1,
        }
    }
    let mut cand_extra = 0usize;
    for &c in candidate {
        match nearest(reference, c) {
            Some(d) if d <= MATCH_WINDOW_MS => {}
            _ => cand_extra += 1,
        }
    }
    BoundaryDelta {
        mean_ms: if matched == 0 {
            0.0
        } else {
            sum / matched as f64
        },
        max_ms: max,
        matched,
        ref_unmatched,
        cand_extra,
    }
}

fn speaker_count(segs: &[DiarSegment]) -> usize {
    segs.iter()
        .map(|s| s.speaker)
        .collect::<BTreeSet<_>>()
        .len()
}

fn speech_ms(segs: &[DiarSegment]) -> u64 {
    segs.iter()
        .map(|s| s.t_end_ms.saturating_sub(s.t_start_ms))
        .sum()
}

fn truth_coverage(truth: &GroundTruth, segs: &[DiarSegment]) -> (usize, usize) {
    let mut covered = 0usize;
    for t in &truth.turns {
        let mid = (t.t_start_ms + t.t_end_ms) / 2;
        if segs.iter().any(|s| s.t_start_ms <= mid && mid < s.t_end_ms) {
            covered += 1;
        }
    }
    (covered, truth.turns.len())
}

fn run_once(
    seg: Arc<SegmentationModel>,
    emb: Arc<EmbeddingModel>,
    audio: &[f32],
    hop_ratio: f32,
) -> Result<(Duration, Vec<DiarSegment>)> {
    let mut cfg = DiarConfig::from_env();
    cfg.hop_ratio = hop_ratio;
    let mut d = Diarizer::new(seg, emb, cfg);
    let t = Instant::now();
    let out = d.diarize_utterance(audio, 0)?;
    Ok((t.elapsed(), out))
}

#[test]
#[ignore]
fn diar_hop_ratio_sweep() -> Result<()> {
    if std::env::var("DIAR_SWEEP").ok().as_deref() != Some("1") {
        eprintln!("skip: set DIAR_SWEEP=1");
        return Ok(());
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let audio_path = sweep_audio_path();
    if !audio_path.exists() {
        return Err(anyhow!(
            "sweep audio {} missing; run diar_build_multispeaker_fixture or set DIAR_SWEEP_AUDIO",
            audio_path.display()
        ));
    }
    let bytes = std::fs::read(&audio_path)?;
    let audio = crate::audio::decode_any_to_16k_mono(&bytes, None)?;
    let audio_s = audio.len() as f64 / FIXTURE_SAMPLE_RATE as f64;

    let truth: Option<GroundTruth> = sweep_truth_path(&audio_path).and_then(|p| {
        std::fs::read(&p)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
    });

    let (seg, emb) = load_models()?;
    let seg = Arc::new(seg);
    let emb = Arc::new(emb);

    let reps = env_usize("DIAR_SWEEP_REPS", 3).max(1);
    let ratios = ratios();
    if ratios.is_empty() {
        return Err(anyhow!("DIAR_SWEEP_RATIOS parsed to nothing"));
    }

    eprintln!(
        "\naudio             : {} ({audio_s:.2} s)",
        audio_path.display()
    );
    match &truth {
        Some(t) => eprintln!(
            "ground truth      : {} turns, {} speakers, {} overlaps",
            t.turns.len(),
            t.speakers.len(),
            t.overlaps.len()
        ),
        None => eprintln!("ground truth      : none (real-audio mode)"),
    }
    eprintln!("reps per setting  : {reps}");

    let mut rows: Vec<(f32, Duration, Vec<DiarSegment>)> = Vec::new();
    for &r in &ratios {
        let mut best = Duration::MAX;
        let mut segs = Vec::new();
        for i in 0..reps {
            let (el, out) = run_once(seg.clone(), emb.clone(), &audio, r)?;
            if i == 0 {
                segs = out;
            }
            best = best.min(el);
        }
        rows.push((r, best, segs));
    }

    let ref_bounds = boundaries(&rows[0].2);
    let ref_ratio = rows[0].0;

    eprintln!("\n===== hop_ratio sweep (reference = {ref_ratio}) =====",);
    eprintln!(
        "{:>9} {:>10} {:>7} {:>9} {:>7} {:>9} {:>10} {:>9} {:>8} {:>7} {:>9}",
        "hop_ratio",
        "best_ms",
        "RTF",
        "segments",
        "spkrs",
        "speech_s",
        "bnd_mean",
        "bnd_max",
        "matched",
        "lost",
        "extra"
    );
    for (r, best, segs) in &rows {
        let b = boundary_delta(&ref_bounds, &boundaries(segs));
        eprintln!(
            "{:>9} {:>10.1} {:>7.4} {:>9} {:>7} {:>9.2} {:>10.1} {:>9} {:>8} {:>7} {:>9}",
            r,
            ms(*best),
            best.as_secs_f64() / audio_s,
            segs.len(),
            speaker_count(segs),
            speech_ms(segs) as f64 / 1000.0,
            b.mean_ms,
            b.max_ms,
            b.matched,
            b.ref_unmatched,
            b.cand_extra
        );
    }

    if let Some(t) = &truth {
        eprintln!(
            "\n{:>9} {:>14} {:>16}",
            "hop_ratio", "turns_covered", "hyp_speakers"
        );
        for (r, _, segs) in &rows {
            let (covered, total) = truth_coverage(t, segs);
            eprintln!(
                "{:>9} {:>14} {:>16}",
                r,
                format!("{covered}/{total}"),
                speaker_count(segs)
            );
        }
    }

    for (r, _, segs) in &rows {
        eprintln!("\n-- segments @ hop_ratio {r} --");
        for s in segs {
            eprintln!(
                "   spk={} {:.2}-{:.2}  conf={:.3}",
                s.speaker,
                s.t_start_ms as f64 / 1000.0,
                s.t_end_ms as f64 / 1000.0,
                s.confidence
            );
        }
    }

    for (r, _, segs) in &rows {
        if segs.is_empty() {
            return Err(anyhow!("hop_ratio {r} produced zero segments"));
        }
    }

    if let Some(limit) = env_f32("DIAR_SWEEP_ASSERT_MS") {
        for (r, _, segs) in &rows {
            let b = boundary_delta(&ref_bounds, &boundaries(segs));
            if b.mean_ms > limit as f64 {
                return Err(anyhow!(
                    "hop_ratio {r} mean boundary delta {:.1} ms exceeds DIAR_SWEEP_ASSERT_MS {limit}",
                    b.mean_ms
                ));
            }
            if b.ref_unmatched > 0 {
                return Err(anyhow!(
                    "hop_ratio {r} lost {} reference boundaries",
                    b.ref_unmatched
                ));
            }
        }
    }

    Ok(())
}

fn kokoro_handle() -> Result<crate::tts::KokoroHandle> {
    let model = env_path("DIAR_FIXTURE_KOKORO_MODEL")
        .ok_or_else(|| anyhow!("DIAR_FIXTURE_KOKORO_MODEL not set or file missing"))?;
    let voices = env_path("DIAR_FIXTURE_KOKORO_VOICES")
        .ok_or_else(|| anyhow!("DIAR_FIXTURE_KOKORO_VOICES not set or file missing"))?;
    let staging = std::env::temp_dir().join(format!("diar-fixture-kokoro-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging)?;
    std::os::unix::fs::symlink(&model, staging.join("kokoro-v1.0.onnx"))?;
    std::os::unix::fs::symlink(&voices, staging.join("voices.bin"))?;
    crate::tts::prepared_handle(&staging)?
        .ok_or_else(|| anyhow!("kokoro prepared_handle returned None"))
}

#[test]
#[ignore]
fn diar_build_multispeaker_fixture() -> Result<()> {
    if std::env::var("DIAR_FIXTURE_BUILD").ok().as_deref() != Some("1") {
        eprintln!("skip: set DIAR_FIXTURE_BUILD=1");
        return Ok(());
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let kokoro = kokoro_handle()?;
    for (_, voice) in SPEAKERS {
        if !kokoro.has_voice(voice) {
            return Err(anyhow!("voices.bin has no voice {voice:?}"));
        }
    }

    let out_dir = std::env::var("DIAR_FIXTURE_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| fixture_dir());
    std::fs::create_dir_all(&out_dir)?;

    let mut track: Vec<f32> = Vec::new();
    let mut turns: Vec<TruthTurn> = Vec::new();
    let mut cursor_ms: i64 = 0;

    for (index, turn) in SCRIPT.iter().enumerate() {
        let (label, voice) = SPEAKERS[turn.speaker];
        let synth = kokoro
            .synthesize(turn.text, Some(voice), Some("en-us"), 1.0)
            .with_context(|| format!("synthesize turn {index} with {voice}"))?;
        let at16k = crate::audio::downmix_and_resample_f32(
            synth.samples(),
            1,
            KOKORO_SAMPLE_RATE,
            FIXTURE_SAMPLE_RATE as usize,
        );
        let trimmed = trim_silence(&at16k);
        let start_ms = (cursor_ms + turn.lead_ms).max(0);
        let start_sample = (start_ms as usize) * FIXTURE_SAMPLE_RATE as usize / 1000;
        let end_sample = start_sample + trimmed.len();
        if track.len() < end_sample {
            track.resize(end_sample, 0.0);
        }
        for (i, v) in trimmed.iter().enumerate() {
            track[start_sample + i] += *v;
        }
        let end_ms = end_sample as i64 * 1000 / FIXTURE_SAMPLE_RATE as i64;
        turns.push(TruthTurn {
            index,
            speaker: label.to_string(),
            voice: voice.to_string(),
            t_start_ms: start_ms as u64,
            t_end_ms: end_ms as u64,
            text: turn.text.to_string(),
        });
        cursor_ms = end_ms;
    }

    let tail = FIXTURE_SAMPLE_RATE as usize / 2;
    track.resize(track.len() + tail, 0.0);

    let peak = track.iter().fold(0f32, |a, v| a.max(v.abs()));
    if peak > 0.0 {
        let gain = 0.89 / peak;
        for v in track.iter_mut() {
            *v *= gain;
        }
    }

    let mut overlaps = Vec::new();
    for i in 0..turns.len() {
        for j in (i + 1)..turns.len() {
            let s = turns[i].t_start_ms.max(turns[j].t_start_ms);
            let e = turns[i].t_end_ms.min(turns[j].t_end_ms);
            if e > s {
                overlaps.push(TruthOverlap {
                    turn_a: i,
                    turn_b: j,
                    t_start_ms: s,
                    t_end_ms: e,
                });
            }
        }
    }
    if overlaps.is_empty() {
        return Err(anyhow!("fixture has no overlapping turns"));
    }

    let audio_path = out_dir.join(AUDIO_FILE);
    write_wav16(&audio_path, &track)?;

    let truth = GroundTruth {
        source: "kokoro-82m-v1.0-onnx".to_string(),
        sample_rate: FIXTURE_SAMPLE_RATE,
        duration_ms: track.len() as u64 * 1000 / FIXTURE_SAMPLE_RATE as u64,
        speakers: SPEAKERS
            .iter()
            .map(|(l, v)| (l.to_string(), v.to_string()))
            .collect(),
        turns,
        overlaps,
    };
    let truth_path = out_dir.join(TRUTH_FILE);
    std::fs::write(&truth_path, serde_json::to_vec_pretty(&truth)?)?;

    eprintln!(
        "wrote {} ({:.2} s, {} bytes)",
        audio_path.display(),
        truth.duration_ms as f64 / 1000.0,
        std::fs::metadata(&audio_path)?.len()
    );
    eprintln!("wrote {}", truth_path.display());
    for t in &truth.turns {
        eprintln!(
            "  turn {:>2} {:<6} {:<12} {:>7.2}-{:>7.2}",
            t.index,
            t.speaker,
            t.voice,
            t.t_start_ms as f64 / 1000.0,
            t.t_end_ms as f64 / 1000.0
        );
    }
    for o in &truth.overlaps {
        eprintln!(
            "  overlap turns {}+{}: {:.2}-{:.2} ({} ms)",
            o.turn_a,
            o.turn_b,
            o.t_start_ms as f64 / 1000.0,
            o.t_end_ms as f64 / 1000.0,
            o.t_end_ms - o.t_start_ms
        );
    }
    Ok(())
}

fn trim_silence(samples: &[f32]) -> Vec<f32> {
    const THRESH: f32 = 0.005;
    let first = samples.iter().position(|v| v.abs() > THRESH);
    let last = samples.iter().rposition(|v| v.abs() > THRESH);
    match (first, last) {
        (Some(a), Some(b)) if b >= a => samples[a..=b].to_vec(),
        _ => samples.to_vec(),
    }
}

fn write_wav16(path: &std::path::Path, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: FIXTURE_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create {}", path.display()))?;
    for v in samples {
        let s = (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        w.write_sample(s)?;
    }
    w.finalize()?;
    Ok(())
}
