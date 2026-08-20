use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;

use super::{framing, DiarConfig, Diarizer, EmbeddingModel, PowersetDecoder, SegmentationModel};

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

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

struct Stage {
    name: &'static str,
    total: Duration,
    calls: usize,
}

impl Stage {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            total: Duration::ZERO,
            calls: 0,
        }
    }
    fn add(&mut self, d: Duration) {
        self.total += d;
        self.calls += 1;
    }
}

fn report(label: &str, audio_s: f64, wall: Duration, stages: &[Stage]) {
    let total_ms = ms(wall);
    eprintln!("\n===== {label} =====");
    eprintln!("audio            : {audio_s:.2} s");
    eprintln!(
        "wall             : {total_ms:.1} ms   (RTF {:.4})",
        total_ms / 1000.0 / audio_s
    );
    eprintln!(
        "{:<22} {:>10} {:>8} {:>9} {:>7}",
        "stage", "total_ms", "calls", "ms/call", "%wall"
    );
    let mut acc = 0.0;
    for s in stages {
        let t = ms(s.total);
        acc += t;
        eprintln!(
            "{:<22} {:>10.1} {:>8} {:>9.3} {:>6.1}%",
            s.name,
            t,
            s.calls,
            if s.calls == 0 {
                0.0
            } else {
                t / s.calls as f64
            },
            100.0 * t / total_ms
        );
    }
    eprintln!(
        "{:<22} {:>10.1} {:>8} {:>9} {:>6.1}%",
        "(unattributed)",
        total_ms - acc,
        "",
        "",
        100.0 * (total_ms - acc) / total_ms
    );
}

fn load_audio() -> Result<Vec<f32>> {
    let path = env_path("DIAR_BENCH_AUDIO")
        .ok_or_else(|| anyhow::anyhow!("DIAR_BENCH_AUDIO not set or file missing"))?;
    let bytes = std::fs::read(&path)?;
    crate::audio::decode_any_to_16k_mono(&bytes, None)
}

fn load_models() -> Result<(SegmentationModel, EmbeddingModel)> {
    let seg_path = env_path("DIAR_SEGMENTATION_MODEL")
        .ok_or_else(|| anyhow::anyhow!("DIAR_SEGMENTATION_MODEL not set or file missing"))?;
    let emb_path = env_path("DIAR_EMBEDDING_MODEL")
        .ok_or_else(|| anyhow::anyhow!("DIAR_EMBEDDING_MODEL not set or file missing"))?;
    let t = Instant::now();
    let seg = SegmentationModel::load(&seg_path)?;
    eprintln!("segmentation load : {:.0} ms", ms(t.elapsed()));
    let t = Instant::now();
    let emb = EmbeddingModel::load(&emb_path)?;
    eprintln!("embedding load    : {:.0} ms", ms(t.elapsed()));
    Ok((seg, emb))
}

#[test]
#[ignore]
fn diar_stage_breakdown() -> Result<()> {
    if std::env::var("DIAR_BENCH").ok().as_deref() != Some("1") {
        eprintln!("skip: set DIAR_BENCH=1");
        return Ok(());
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let audio = load_audio()?;
    let audio_s = audio.len() as f64 / 16_000.0;
    let (seg, emb) = load_models()?;
    let cfg = DiarConfig::from_env();
    eprintln!(
        "config            : chunk={}s hop_ratio={} seg_batch={} emb_batch={} threads={}",
        cfg.chunk_seconds,
        cfg.hop_ratio,
        env_usize("DIAR_SEG_BATCH", 8),
        env_usize("DIAR_EMB_BATCH", 8),
        env_usize("DIAR_INTRA_THREADS", 0)
    );

    let decoder = PowersetDecoder::new(seg.max_speakers_per_chunk(), seg.max_speakers_per_frame());

    let mut st_chunk = Stage::new("slide_chunks");
    let mut st_seg = Stage::new("segmentation onnx");
    let mut st_powerset = Stage::new("powerset decode");
    let mut st_median = Stage::new("median filter");
    let mut st_spans = Stage::new("extract_spans");
    let mut st_embed = Stage::new("embedding (fbank+onnx)");
    let mut st_cluster = Stage::new("clustering");
    let mut st_coalesce = Stage::new("coalesce");

    let wall = Instant::now();

    let t = Instant::now();
    let chunks = framing::slide_chunks(&audio, seg.sample_rate(), cfg.chunk_seconds, cfg.hop_ratio);
    st_chunk.add(t.elapsed());

    let seg_batch = env_usize("DIAR_SEG_BATCH", 8).max(1);
    let mut spans: Vec<framing::ChunkSpans> = Vec::new();
    let mut idx = 0usize;
    for batch in chunks.chunks(seg_batch) {
        let refs: Vec<&[f32]> = batch.iter().map(|c| c.samples.as_slice()).collect();
        let t = Instant::now();
        let logits_batch = seg.run_batch(&refs)?;
        st_seg.add(t.elapsed());
        for (chunk, logits) in batch.iter().zip(logits_batch) {
            let t = Instant::now();
            let multihot = decoder.to_multilabel_hard(&logits);
            st_powerset.add(t.elapsed());
            let t = Instant::now();
            let smoothed = framing::median_filter_multihot(&multihot, cfg.median_filter_window);
            st_median.add(t.elapsed());
            let t = Instant::now();
            let chunk_spans = framing::extract_spans(
                &smoothed,
                seg.frame_rate_hz(),
                chunk.t_offset_ms,
                cfg.min_span_frames,
            );
            st_spans.add(t.elapsed());
            spans.push(framing::ChunkSpans {
                chunk_index: idx,
                spans: chunk_spans,
            });
            idx += 1;
        }
    }

    let total_spans: usize = spans.iter().map(|c| c.spans.len()).sum();
    let mut clusterer = super::OnlineClusterer::new(cfg.clustering_threshold, cfg.max_speakers);
    let mut emitted = Vec::new();
    let mut embedded_samples = 0usize;
    let mut distinct: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    for cs in &spans {
        for span in &cs.spans {
            let span_audio = &audio[span.sample_start..span.sample_end.min(audio.len())];
            if span_audio.len() < emb.min_input_samples() {
                continue;
            }
            embedded_samples += span_audio.len();
            distinct.insert((span.sample_start, span.sample_end.min(audio.len())));
            let t = Instant::now();
            let v = emb.embed(span_audio)?;
            st_embed.add(t.elapsed());
            let t = Instant::now();
            let (cluster_id, score) = clusterer.assign(&v);
            st_cluster.add(t.elapsed());
            emitted.push(super::DiarSegment {
                speaker: cluster_id,
                t_start_ms: span.t_start_ms,
                t_end_ms: span.t_end_ms,
                confidence: score,
            });
        }
    }

    let t = Instant::now();
    let out = framing::coalesce_segments(emitted);
    st_coalesce.add(t.elapsed());

    let wall = wall.elapsed();

    eprintln!(
        "chunks={} spans={} embeds={} distinct_spans={} embedded_audio={:.1}s ({:.1}x input) out_segments={}",
        chunks.len(),
        total_spans,
        st_embed.calls,
        distinct.len(),
        embedded_samples as f64 / 16_000.0,
        embedded_samples as f64 / 16_000.0 / audio_s,
        out.len()
    );

    report(
        "diarization stage breakdown",
        audio_s,
        wall,
        &[
            st_chunk,
            st_seg,
            st_powerset,
            st_median,
            st_spans,
            st_embed,
            st_cluster,
            st_coalesce,
        ],
    );

    Ok(())
}

#[test]
#[ignore]
fn diar_end_to_end() -> Result<()> {
    if std::env::var("DIAR_BENCH").ok().as_deref() != Some("1") {
        eprintln!("skip: set DIAR_BENCH=1");
        return Ok(());
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let audio = load_audio()?;
    let audio_s = audio.len() as f64 / 16_000.0;
    let (seg, emb) = load_models()?;
    let seg = std::sync::Arc::new(seg);
    let emb = std::sync::Arc::new(emb);

    let reps = env_usize("DIAR_BENCH_REPS", 3);
    let mut best = Duration::MAX;
    let mut n_out = 0usize;
    for i in 0..reps {
        let mut d = Diarizer::new(seg.clone(), emb.clone(), DiarConfig::from_env());
        let t = Instant::now();
        let out = d.diarize_utterance(&audio, 0)?;
        let el = t.elapsed();
        n_out = out.len();
        eprintln!("  rep {i}: {:.1} ms  segments={}", ms(el), out.len());
        if i == 0 {
            for s in &out {
                eprintln!(
                    "  SEG spk={} {:.2}-{:.2}",
                    s.speaker,
                    s.t_start_ms as f64 / 1000.0,
                    s.t_end_ms as f64 / 1000.0
                );
            }
        }
        best = best.min(el);
    }
    eprintln!(
        "\ndiarize_utterance best of {reps}: {:.1} ms for {audio_s:.2}s audio (RTF {:.4}), segments={n_out}",
        ms(best),
        best.as_secs_f64() / audio_s
    );
    Ok(())
}

#[test]
#[ignore]
fn diar_concurrency() -> Result<()> {
    if std::env::var("DIAR_BENCH").ok().as_deref() != Some("1") {
        eprintln!("skip: set DIAR_BENCH=1");
        return Ok(());
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let audio = load_audio()?;
    let audio_s = audio.len() as f64 / 16_000.0;
    let (seg, emb) = load_models()?;
    let seg = std::sync::Arc::new(seg);
    let emb = std::sync::Arc::new(emb);

    for &n in &[1usize, 2, 4] {
        let t = Instant::now();
        std::thread::scope(|scope| {
            for _ in 0..n {
                let seg = seg.clone();
                let emb = emb.clone();
                let audio = &audio;
                scope.spawn(move || {
                    let mut d = Diarizer::new(seg, emb, DiarConfig::from_env());
                    let _ = d.diarize_utterance(audio, 0);
                });
            }
        });
        let el = t.elapsed();
        eprintln!(
            "{n} concurrent diarizations: {:.1} ms wall, {:.1} ms/request, aggregate RTF {:.4}",
            ms(el),
            ms(el) / n as f64,
            el.as_secs_f64() / (audio_s * n as f64)
        );
    }
    Ok(())
}

#[test]
#[ignore]
fn embedding_cost_model() -> Result<()> {
    if std::env::var("DIAR_BENCH").ok().as_deref() != Some("1") {
        eprintln!("skip: set DIAR_BENCH=1");
        return Ok(());
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let emb_path = env_path("DIAR_EMBEDDING_MODEL")
        .ok_or_else(|| anyhow::anyhow!("DIAR_EMBEDDING_MODEL not set or file missing"))?;
    let emb = EmbeddingModel::load(&emb_path)?;

    eprintln!("\n-- cost vs input length --");
    eprintln!("{:>10} {:>12} {:>12}", "seconds", "ms", "ms/sec");
    for &secs in &[1usize, 2, 4, 8, 16] {
        let audio = vec![0.01f32; 16_000 * secs];
        let _ = emb.embed(&audio)?;
        let mut best = Duration::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            let v = emb.embed(&audio)?;
            std::hint::black_box(&v);
            best = best.min(t.elapsed());
        }
        eprintln!(
            "{:>10} {:>12.1} {:>12.4}",
            secs,
            ms(best),
            ms(best) / secs as f64
        );
    }

    Ok(())
}

#[test]
#[ignore]
fn embedding_shape_churn() -> Result<()> {
    if std::env::var("DIAR_BENCH").ok().as_deref() != Some("1") {
        eprintln!("skip: set DIAR_BENCH=1");
        return Ok(());
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let emb_path = env_path("DIAR_EMBEDDING_MODEL")
        .ok_or_else(|| anyhow::anyhow!("DIAR_EMBEDDING_MODEL not set or file missing"))?;
    let emb = EmbeddingModel::load(&emb_path)?;
    let n = env_usize("DIAR_BENCH_SHAPES", 30);

    let audio = vec![0.01f32; 16_000 * 4];
    let _ = emb.embed(&audio)?;

    let t = Instant::now();
    for _ in 0..n {
        let v = emb.embed(&audio)?;
        std::hint::black_box(&v);
    }
    let same = t.elapsed();

    let t = Instant::now();
    for i in 0..n {
        let a = vec![0.01f32; 16_000 * 4 + i * 1120];
        let v = emb.embed(&a)?;
        std::hint::black_box(&v);
    }
    let churn = t.elapsed();

    let t = Instant::now();
    for i in 0..n {
        let a = vec![0.01f32; 16_000 * 4 + i * 1120];
        let v = emb.embed(&a)?;
        std::hint::black_box(&v);
    }
    let churn2 = t.elapsed();

    eprintln!(
        "\n{n} inferences, identical shape (4 s)        : {:.1} ms total, {:.2} ms/call",
        ms(same),
        ms(same) / n as f64
    );
    eprintln!(
        "{n} inferences, all-distinct shapes, 1st pass : {:.1} ms total, {:.2} ms/call",
        ms(churn),
        ms(churn) / n as f64
    );
    eprintln!(
        "{n} inferences, same distinct shapes, 2nd pass: {:.1} ms total, {:.2} ms/call",
        ms(churn2),
        ms(churn2) / n as f64
    );
    Ok(())
}

#[test]
#[ignore]
fn fbank_microbench() -> Result<()> {
    if std::env::var("DIAR_BENCH").ok().as_deref() != Some("1") {
        eprintln!("skip: set DIAR_BENCH=1");
        return Ok(());
    }
    let fb = super::fbank::FBank::new(80, 400, 160);
    let audio = vec![0.01f32; 16_000 * 10];
    let mut best = Duration::MAX;
    for _ in 0..10 {
        let t = Instant::now();
        let f = fb.compute(&audio)?;
        std::hint::black_box(&f);
        best = best.min(t.elapsed());
    }
    eprintln!(
        "fbank 10s audio: best {:.2} ms  ({:.4} RTF)",
        ms(best),
        best.as_secs_f64() / 10.0
    );
    Ok(())
}
