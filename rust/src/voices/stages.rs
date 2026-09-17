use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use super::workspace::{Manifest, Recording, Registry, Source, Speaker, Workspace};

pub type Log<'a> = &'a mut (dyn FnMut(String) + Send);

#[derive(Clone, Debug, Default)]
pub struct Endpoints {
    pub stt: Option<String>,
    pub diar: Option<String>,
    pub chunk_minutes: f64,
}

pub async fn post_audio(
    client: &reqwest::Client,
    url: &str,
    wav: Vec<u8>,
    fields: &[(&str, &str)],
) -> Result<Value> {
    let mut form = reqwest::multipart::Form::new().part(
        "file",
        reqwest::multipart::Part::bytes(wav)
            .file_name("a.wav")
            .mime_str("audio/wav")?,
    );
    for (k, v) in fields {
        form = form.text(k.to_string(), v.to_string());
    }
    let resp = client
        .post(url)
        .multipart(form)
        .send()
        .await
        .with_context(|| format!("post {url}"))?;
    let status = resp.status();
    let body = resp.bytes().await?;
    if !status.is_success() {
        return Err(anyhow!(
            "{url}: {status} {}",
            String::from_utf8_lossy(&body[..body.len().min(200)])
        ));
    }
    Ok(serde_json::from_slice(&body)?)
}

pub async fn to_wav(
    src: &Path,
    dst: &Path,
    start: Option<f64>,
    dur: Option<f64>,
    rate: u32,
) -> Result<()> {
    let mut cmd = tokio::process::Command::new("ffmpeg");
    cmd.args(["-nostdin", "-loglevel", "error", "-y"]);
    if let Some(s) = start {
        cmd.args(["-ss", &format!("{s:.3}")]);
    }
    if let Some(d) = dur {
        cmd.args(["-t", &format!("{d:.3}")]);
    }
    cmd.arg("-i").arg(src);
    cmd.args(["-ac", "1", "-ar", &rate.to_string(), "-c:a", "pcm_s16le"]);
    cmd.arg(dst);
    let out = cmd.output().await.context("spawn ffmpeg")?;
    if !out.status.success() {
        return Err(anyhow!(
            "ffmpeg {}: {}",
            src.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(())
}

pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> Result<Self> {
        let p = std::env::temp_dir().join(format!(
            "voices-{tag}-{}-{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&p)?;
        Ok(TempDir(p))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn audio_source(ws: &Workspace, r: &Recording) -> Result<PathBuf> {
    let p = ws.recording_path(r);
    if p.is_file() {
        return Ok(p);
    }
    Err(anyhow!(
        "{}: no local file (source_url downloads are not done on the engine)",
        r.path
    ))
}

fn todo<'a>(m: &'a Manifest, ws: &Workspace, kind: &str) -> Vec<&'a Recording> {
    m.by_path()
        .into_iter()
        .filter(|r| !(r.current(kind) && ws.has_artifact(r, kind)))
        .collect()
}

fn wav_seconds(path: &Path) -> f64 {
    std::fs::metadata(path)
        .map(|m| (m.len().saturating_sub(44)) as f64 / 32_000.0)
        .unwrap_or(0.0)
}

pub async fn transcribe(
    ws: &Workspace,
    m: &mut Manifest,
    client: &reqwest::Client,
    url: &str,
    chunk_minutes: f64,
    log: Log<'_>,
) -> Result<usize> {
    let shas: Vec<String> = todo(m, ws, "transcript")
        .iter()
        .map(|r| r.sha256.clone())
        .collect();
    let mut n = 0;
    for sha in shas {
        let r = m.rows[&sha].clone();
        let res = transcribe_one(ws, client, url, chunk_minutes, &r).await;
        match res {
            Ok((v, dur)) => {
                if let Some(row) = m.rows.get_mut(&sha) {
                    if row.duration_ms.is_none() {
                        row.duration_ms = Some((dur * 1000.0) as i64);
                    }
                }
                ws.put_artifact(m, &sha, "transcript", &v)?;
                n += 1;
                log(format!("transcribed {}", r.path));
            }
            Err(e) => log(format!("transcribe failed for {}: {e:#}", r.path)),
        }
    }
    Ok(n)
}

async fn transcribe_one(
    ws: &Workspace,
    client: &reqwest::Client,
    url: &str,
    chunk_minutes: f64,
    r: &Recording,
) -> Result<(Value, f64)> {
    let td = TempDir::new("stt")?;
    let src = audio_source(ws, r)?;
    let wav = td.0.join("a.wav");
    to_wav(&src, &wav, None, None, 16_000).await?;
    let dur = wav_seconds(&wav);
    let fields = [
        ("response_format", "verbose_json"),
        ("language", "en"),
        ("model", "whisper"),
    ];
    if chunk_minutes > 0.0 && dur > chunk_minutes * 60.0 {
        let mut merged = serde_json::json!({"task": "transcribe", "language": "en", "duration": dur, "text": "", "segments": []});
        let mut t0 = 0.0;
        while t0 < dur {
            let part = td.0.join("part.wav");
            to_wav(&wav, &part, Some(t0), Some(chunk_minutes * 60.0), 16_000).await?;
            let res = post_audio(client, url, std::fs::read(&part)?, &fields).await?;
            merge_transcript(&mut merged, &res, t0);
            t0 += chunk_minutes * 60.0;
        }
        return Ok((merged, dur));
    }
    Ok((
        post_audio(client, url, std::fs::read(&wav)?, &fields).await?,
        dur,
    ))
}

pub fn merge_transcript(merged: &mut Value, chunk: &Value, offset: f64) {
    fn shift(v: &mut Value, offset: f64) {
        for key in ["start", "end"] {
            if let Some(t) = v[key].as_f64() {
                v[key] = serde_json::json!(t + offset);
            }
        }
        if let Some(words) = v["words"].as_array_mut() {
            for word in words {
                shift(word, offset);
            }
        }
    }
    let text = chunk["text"].as_str().unwrap_or("").trim();
    let current = merged["text"].as_str().unwrap_or("").trim();
    merged["text"] = serde_json::json!([current, text]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" "));
    for key in ["segments", "words"] {
        if let Some(items) = chunk[key].as_array() {
            if !merged[key].is_array() {
                merged[key] = serde_json::json!([]);
            }
            for item in items {
                let mut item = item.clone();
                shift(&mut item, offset);
                merged[key].as_array_mut().unwrap().push(item);
            }
        }
    }
}

pub async fn diarize(
    ws: &Workspace,
    m: &mut Manifest,
    client: &reqwest::Client,
    url: &str,
    log: Log<'_>,
) -> Result<usize> {
    let shas: Vec<String> = m
        .by_path()
        .into_iter()
        .filter(|r| {
            !r.current("diar")
                || !ws.has_artifact(r, "diar")
                || ws
                    .get_artifact(&r.sha256, "diar")
                    .map(|v| !v["speakers"].is_array())
                    .unwrap_or(true)
        })
        .map(|r| r.sha256.clone())
        .collect();
    let mut n = 0;
    for sha in shas {
        let r = m.rows[&sha].clone();
        let res: Result<Value> = async {
            let td = TempDir::new("diar")?;
            let wav = td.0.join("a.wav");
            to_wav(&audio_source(ws, &r)?, &wav, None, None, 16_000).await?;
            post_audio(client, url, std::fs::read(&wav)?, &[]).await
        }
        .await;
        match res {
            Ok(v) => {
                let k = v["speakers"].as_array().map(|a| a.len()).unwrap_or(0);
                ws.put_artifact(m, &sha, "diar", &v)?;
                n += 1;
                log(format!("diarized {} speakers={k}", r.path));
            }
            Err(e) => log(format!("diarize failed for {}: {e:#}", r.path)),
        }
    }
    Ok(n)
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-9;
    v.iter().map(|x| x / n).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array()
        .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
        .unwrap_or_default()
}

pub fn identify(ws: &Workspace, m: &Manifest, min_seconds: f64, log: Log<'_>) -> Result<Registry> {
    let _guard = super::workspace::REGISTRY_GATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut reg = ws.load_speakers()?;
    let th = reg.threshold_link;
    let th_merge = reg.threshold_merge;
    let mut linked = 0;
    for r in m.by_path() {
        if !ws.has_artifact(r, "diar") {
            continue;
        }
        let diar = ws.get_artifact(&r.sha256, "diar")?;
        for spk in diar["speakers"].as_array().into_iter().flatten() {
            let secs = spk["seconds"].as_f64().unwrap_or(0.0);
            let e = f32s(&spk["embedding"]);
            if e.is_empty() || secs < min_seconds {
                continue;
            }
            let e = unit(&e);
            let label = spk["speaker"].as_str().unwrap_or("").to_string();
            if reg.speakers.iter().any(|s| {
                s.sources
                    .iter()
                    .any(|x| x.recording == r.sha256 && x.label == label)
            }) {
                continue;
            }
            let mut best: Option<(usize, f32)> = None;
            for (i, s) in reg.speakers.iter().enumerate() {
                if s.centroid.is_empty() {
                    continue;
                }
                let sim = dot(&e, &unit(&s.centroid));
                if best.map(|(_, b)| sim > b).unwrap_or(true) {
                    best = Some((i, sim));
                }
            }
            let src = Source {
                recording: r.sha256.clone(),
                path: r.path.clone(),
                label: label.clone(),
                seconds: (secs * 10.0).round() / 10.0,
                podcast: r.podcast.clone(),
                sim: None,
            };
            let joins = best.filter(|(i, sim)| {
                let s = &reg.speakers[*i];
                let same_rec = s.sources.iter().any(|x| x.recording == r.sha256);
                *sim >= th && (!same_rec || *sim >= th_merge)
            });
            match joins {
                Some((i, sim)) => {
                    let s = &mut reg.speakers[i];
                    if s.sources
                        .iter()
                        .any(|x| x.recording == r.sha256 && x.label == label)
                    {
                        continue;
                    }
                    let w = s.seconds as f32;
                    let mixed: Vec<f32> = s
                        .centroid
                        .iter()
                        .zip(&e)
                        .map(|(c, x)| c * w + x * secs as f32)
                        .collect();
                    s.centroid = unit(&mixed);
                    s.seconds = ((s.seconds + secs) * 10.0).round() / 10.0;
                    s.sources.push(Source {
                        sim: Some((sim as f64 * 1000.0).round() / 1000.0),
                        ..src
                    });
                }
                None => reg.speakers.push(Speaker {
                    id: format!("spk-{:04}", reg.speakers.len() + 1),
                    name: None,
                    name_proposed: None,
                    role: None,
                    centroid: e,
                    seconds: (secs * 10.0).round() / 10.0,
                    sources: vec![src],
                    extra: Default::default(),
                }),
            }
            linked += 1;
        }
    }
    for s in reg.speakers.iter_mut().filter(|s| s.name.is_none()) {
        let mut pods: std::collections::BTreeMap<&str, std::collections::BTreeSet<&str>> =
            Default::default();
        for x in &s.sources {
            if let Some(p) = x.podcast.as_deref() {
                pods.entry(p).or_default().insert(x.recording.as_str());
            }
        }
        if let Some((p, recs)) = pods.iter().max_by_key(|(_, v)| v.len()) {
            if recs.len() >= 3 {
                s.name_proposed = Some(format!("{p} host"));
                s.role = Some("host".to_string());
            }
        }
    }
    ws.save_speakers(&mut reg)?;
    log(format!(
        "identify: {linked} speaker-recordings linked, {} speakers",
        reg.speakers.len()
    ));
    Ok(reg)
}

pub fn clips(ws: &Workspace, m: &mut Manifest, min_seconds: f64, log: Log<'_>) -> Result<usize> {
    let shas: Vec<String> = todo(m, ws, "clips")
        .iter()
        .filter(|r| ws.has_artifact(r, "diar"))
        .map(|r| r.sha256.clone())
        .collect();
    let mut n = 0;
    for sha in shas {
        let r = m.rows[&sha].clone();
        let diar = ws.get_artifact(&sha, "diar")?;
        let wsegs: Vec<Value> = if ws.has_artifact(&r, "transcript") {
            ws.get_artifact(&sha, "transcript")?["segments"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let mut segs: Vec<Value> = diar["segments"].as_array().cloned().unwrap_or_default();
        segs.sort_by(|a, b| {
            a["start"]
                .as_f64()
                .partial_cmp(&b["start"].as_f64())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let f = |v: &Value, k: &str| v[k].as_f64().unwrap_or(0.0);
        let mut merged: Vec<(f64, f64, String, f64)> = Vec::new();
        for s in &segs {
            let spk = s["speaker"].as_str().unwrap_or("").to_string();
            let conf = s["confidence"].as_f64().unwrap_or(1.0);
            match merged.last_mut() {
                Some(last) if last.2 == spk && f(s, "start") - last.1 < 0.3 => {
                    last.1 = f(s, "end");
                    last.3 = last.3.min(conf);
                }
                _ => merged.push((f(s, "start"), f(s, "end"), spk, conf)),
            }
        }
        let dur_s = diar["duration"].as_f64().unwrap_or(0.0) as usize + 2;
        let mut busy: std::collections::BTreeMap<String, Vec<bool>> = Default::default();
        for s in &segs {
            let v = busy
                .entry(s["speaker"].as_str().unwrap_or("").to_string())
                .or_insert_with(|| vec![false; dur_s]);
            let (a, b) = (f(s, "start") as usize, f(s, "end").ceil() as usize);
            for x in v.iter_mut().take(b.min(dur_s)).skip(a) {
                *x = true;
            }
        }
        let mut out = Vec::new();
        for (i, (start, end, spk, conf)) in merged.iter().enumerate() {
            let dur = end - start;
            if dur < min_seconds {
                continue;
            }
            let (a, b) = (*start as usize + 1, *end as usize);
            let clean = !busy
                .iter()
                .filter(|(k, _)| *k != spk)
                .any(|(_, v)| v.iter().take(b.min(v.len())).skip(a).any(|x| *x));
            let text = wsegs
                .iter()
                .filter(|w| {
                    let (ws0, we) = (f(w, "start"), f(w, "end"));
                    ws0 < *end && we > *start && we.min(*end) - ws0.max(*start) > 0.5 * (we - ws0)
                })
                .map(|w| w["text"].as_str().unwrap_or("").trim().to_string())
                .collect::<Vec<_>>()
                .join(" ");
            out.push(serde_json::json!({
                "i": i, "start": (start * 100.0).round() / 100.0, "end": (end * 100.0).round() / 100.0,
                "dur": (dur * 100.0).round() / 100.0, "speaker": spk, "conf": (conf * 1000.0).round() / 1000.0,
                "clean": clean, "text_aligned": text,
            }));
        }
        let clean_n = out
            .iter()
            .filter(|c| c["clean"].as_bool().unwrap_or(false))
            .count();
        let total = out.len();
        ws.put_artifact(
            m,
            &sha,
            "clips",
            &serde_json::json!({"recording": sha, "min_seconds": min_seconds, "clips": out}),
        )?;
        n += 1;
        log(format!("clips {}: {total} turns, {clean_n} clean", r.path));
    }
    Ok(n)
}

#[derive(Clone, Debug)]
pub struct Clip {
    pub recording: String,
    pub i: u64,
    pub start: f64,
    pub dur: f64,
    pub text: String,
}

pub fn speaker_clips(
    ws: &Workspace,
    m: &Manifest,
    reg: &Registry,
    speaker: &str,
    min_seconds: f64,
) -> Result<Vec<Clip>> {
    let spk = reg
        .find(speaker)
        .ok_or_else(|| anyhow!("speaker {speaker:?} not in the registry"))?;
    let mut out = Vec::new();
    for r in m.by_path() {
        let labels: Vec<&str> = spk
            .sources
            .iter()
            .filter(|x| x.recording == r.sha256)
            .map(|x| x.label.as_str())
            .collect();
        if labels.is_empty() || !ws.has_artifact(r, "clips") {
            continue;
        }
        for c in ws.get_artifact(&r.sha256, "clips")?["clips"]
            .as_array()
            .into_iter()
            .flatten()
        {
            let dur = c["dur"].as_f64().unwrap_or(0.0);
            if labels.contains(&c["speaker"].as_str().unwrap_or(""))
                && dur >= min_seconds
                && c["clean"].as_bool().unwrap_or(false)
            {
                out.push(Clip {
                    recording: r.sha256.clone(),
                    i: c["i"].as_u64().unwrap_or(0),
                    start: c["start"].as_f64().unwrap_or(0.0),
                    dur,
                    text: c["text"]
                        .as_str()
                        .or(c["text_aligned"].as_str())
                        .unwrap_or("")
                        .to_string(),
                });
            }
        }
    }
    out.sort_by(|a, b| {
        b.dur
            .partial_cmp(&a.dur)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(out)
}

pub async fn clip_wavs(
    ws: &Workspace,
    m: &Manifest,
    clips: &[Clip],
    limit: usize,
) -> Result<Vec<String>> {
    let dir = ws.clips_dir();
    std::fs::create_dir_all(&dir)?;
    let mut out = Vec::new();
    for c in clips.iter().take(limit) {
        let name = format!("{}-{}.wav", &c.recording[..12.min(c.recording.len())], c.i);
        let p = dir.join(&name);
        if !p.is_file() {
            let r = m
                .rows
                .get(&c.recording)
                .ok_or_else(|| anyhow!("recording {} gone", c.recording))?;
            to_wav(
                &audio_source(ws, r)?,
                &p,
                Some(c.start),
                Some(c.dur.min(30.0)),
                24_000,
            )
            .await?;
        }
        out.push(format!("clips/{name}"));
    }
    Ok(out)
}

pub fn pick_sentences(texts: &[String], n: usize) -> Vec<String> {
    let mut out = Vec::new();
    for t in texts {
        let mut cur = String::new();
        let chars: Vec<char> = t.chars().collect();
        for (i, ch) in chars.iter().enumerate() {
            cur.push(*ch);
            let boundary = matches!(ch, '.' | '?' | '!')
                && chars.get(i + 1).map(|c| c.is_whitespace()).unwrap_or(true);
            if boundary {
                let s = cur.trim().to_string();
                cur.clear();
                let words = s.split_whitespace().count();
                let year = s
                    .as_bytes()
                    .windows(4)
                    .any(|w| w.iter().all(|b| b.is_ascii_digit()));
                if (9..=18).contains(&words)
                    && s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    && !year
                {
                    out.push(s);
                }
            }
        }
    }
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in (1..out.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.swap(i, (state % (i as u64 + 1)) as usize);
    }
    out.truncate(n);
    out
}

pub async fn auto(
    ws: &Workspace,
    client: &reqwest::Client,
    ep: &Endpoints,
    log: Log<'_>,
) -> Result<Value> {
    let mut m = ws.load()?;
    let (added, updated) = ws.scan(&mut m, true)?;
    log(format!("scan: {added} added, {updated} updated"));
    let mut out = serde_json::json!({"scan": {"added": added, "updated": updated}});
    if let Some(url) = &ep.stt {
        out["transcribed"] =
            serde_json::json!(transcribe(ws, &mut m, client, url, ep.chunk_minutes, log).await?);
    }
    if let Some(url) = &ep.diar {
        out["diarized"] = serde_json::json!(diarize(ws, &mut m, client, url, log).await?);
    }
    let reg = identify(ws, &m, 60.0, log)?;
    out["speakers"] = serde_json::json!(reg.speakers.len());
    out["clips"] = serde_json::json!(clips(ws, &mut m, 4.0, log)?);
    if let Value::Object(st) = ws.status(&m)? {
        for (k, v) in st {
            out[k] = v;
        }
    }
    Ok(out)
}
