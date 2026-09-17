use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub static REGISTRY_GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub const SCHEMA_VERSION: u32 = 1;
pub const AUDIO_SUFFIXES: [&str; 9] = [
    "mp3", "m4a", "wav", "flac", "ogg", "opus", "mkv", "mp4", "webm",
];
pub const KINDS: [&str; 3] = ["transcript", "diar", "clips"];

pub fn producer(kind: &str) -> &'static str {
    match kind {
        "transcript" => "whisper-large-v3-turbo-ct2/verbose_json/v1",
        "diar" => "diarizen-16s-hop0.1+wespeaker-resnet293/v1",
        "clips" => "turns>=4s-nonoverlap+aligned-transcript/v2",
        _ => "unknown/v0",
    }
}

pub fn now_iso() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}+00:00",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h)?;
    Ok(format!("{:x}", h.finalize()))
}

pub fn probe_duration_ms(path: &Path) -> Option<i64> {
    let out = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    s.trim().parse::<f64>().ok().map(|d| (d * 1000.0) as i64)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub sha256: String,
    pub producer: String,
    pub bytes: u64,
    pub created: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Recording {
    pub sha256: String,
    pub path: String,
    pub bytes: u64,
    pub mtime: f64,
    #[serde(default)]
    pub duration_ms: Option<i64>,
    #[serde(default)]
    pub source_url: Option<String>,
    #[serde(default)]
    pub podcast: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub date: Option<String>,
    #[serde(default)]
    pub artifacts: BTreeMap<String, ArtifactRef>,
}

impl Recording {
    pub fn current(&self, kind: &str) -> bool {
        self.artifacts
            .get(kind)
            .map(|a| a.producer == producer(kind))
            .unwrap_or(false)
    }

    pub fn stale(&self, kind: &str) -> bool {
        self.artifacts
            .get(kind)
            .map(|a| a.producer != producer(kind))
            .unwrap_or(false)
    }
}

#[derive(Serialize, Deserialize)]
struct ManifestFile {
    schema_version: u32,
    #[serde(default)]
    updated: Option<String>,
    #[serde(default)]
    producers: BTreeMap<String, String>,
    #[serde(default)]
    recordings: Vec<Recording>,
}

#[derive(Default)]
pub struct Manifest {
    pub rows: BTreeMap<String, Recording>,
}

impl Manifest {
    pub fn by_path(&self) -> Vec<&Recording> {
        let mut v: Vec<&Recording> = self.rows.values().collect();
        v.sort_by(|a, b| a.path.cmp(&b.path));
        v
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Source {
    pub recording: String,
    pub path: String,
    pub label: String,
    pub seconds: f64,
    #[serde(default)]
    pub podcast: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sim: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Speaker {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub name_proposed: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default)]
    pub centroid: Vec<f32>,
    #[serde(default)]
    pub seconds: f64,
    #[serde(default)]
    pub sources: Vec<Source>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Registry {
    pub schema_version: u32,
    pub threshold_link: f32,
    pub threshold_merge: f32,
    #[serde(default)]
    pub updated: Option<String>,
    #[serde(default)]
    pub speakers: Vec<Speaker>,
}

impl Default for Registry {
    fn default() -> Self {
        Registry {
            schema_version: SCHEMA_VERSION,
            threshold_link: 0.70,
            threshold_merge: 0.75,
            updated: None,
            speakers: Vec::new(),
        }
    }
}

impl Registry {
    pub fn find(&self, id_or_name: &str) -> Option<&Speaker> {
        self.speakers
            .iter()
            .find(|s| s.id == id_or_name || s.name.as_deref() == Some(id_or_name))
    }
}

#[derive(Clone, Debug)]
pub struct Workspace {
    pub root: PathBuf,
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

impl Workspace {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Workspace { root: root.into() }
    }

    pub fn recordings_dir(&self) -> PathBuf {
        self.root.join("recordings")
    }
    pub fn artifacts_dir(&self) -> PathBuf {
        self.root.join("artifacts")
    }
    pub fn voices_dir(&self) -> PathBuf {
        self.root.join("voices")
    }
    pub fn clips_dir(&self) -> PathBuf {
        self.root.join("clips")
    }
    pub fn manifest_path(&self) -> PathBuf {
        self.root.join("manifest.json")
    }
    pub fn speakers_path(&self) -> PathBuf {
        self.root.join("speakers.json")
    }

    pub fn init(&self) -> Result<()> {
        for d in [
            self.recordings_dir(),
            self.artifacts_dir(),
            self.voices_dir(),
        ] {
            std::fs::create_dir_all(&d)?;
        }
        if !self.manifest_path().exists() {
            self.save(&Manifest::default())?;
        }
        Ok(())
    }

    pub fn load(&self) -> Result<Manifest> {
        let p = self.manifest_path();
        if !p.exists() {
            return Ok(Manifest::default());
        }
        let raw = std::fs::read(&p)?;
        let file: ManifestFile =
            serde_json::from_slice(&raw).with_context(|| format!("parse {}", p.display()))?;
        Ok(Manifest {
            rows: file
                .recordings
                .into_iter()
                .map(|r| (r.sha256.clone(), r))
                .collect(),
        })
    }

    pub fn save(&self, m: &Manifest) -> Result<()> {
        let file = ManifestFile {
            schema_version: SCHEMA_VERSION,
            updated: Some(now_iso()),
            producers: KINDS
                .iter()
                .map(|k| (k.to_string(), producer(k).to_string()))
                .collect(),
            recordings: m.by_path().into_iter().cloned().collect(),
        };
        write_atomic(&self.manifest_path(), &serde_json::to_vec_pretty(&file)?)
    }

    pub fn audio_files(&self) -> Vec<PathBuf> {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            let Ok(rd) = std::fs::read_dir(dir) else {
                return;
            };
            let mut dirs = Vec::new();
            let mut media = Vec::new();
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with('.') {
                    continue;
                }
                let p = e.path();
                if p.is_dir() {
                    dirs.push((name, p));
                } else if p.is_file() {
                    let ext = p
                        .extension()
                        .map(|x| x.to_string_lossy().to_ascii_lowercase())
                        .unwrap_or_default();
                    if AUDIO_SUFFIXES.contains(&ext.as_str()) {
                        media.push(p);
                    }
                }
            }
            let stems: Vec<String> = media
                .iter()
                .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
                .collect();
            out.extend(media);
            dirs.sort();
            for (name, p) in dirs {
                if !stems.contains(&name) {
                    walk(&p, out);
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.recordings_dir(), &mut out);
        out.sort();
        out
    }

    pub fn scan(&self, m: &mut Manifest, probe: bool) -> Result<(usize, usize)> {
        self.scan_files(m, probe, self.audio_files())
    }

    pub(super) fn scan_files(
        &self,
        m: &mut Manifest,
        probe: bool,
        files: Vec<PathBuf>,
    ) -> Result<(usize, usize)> {
        let base = self.recordings_dir();
        let by_path: BTreeMap<String, String> = m
            .rows
            .values()
            .map(|r| (r.path.clone(), r.sha256.clone()))
            .collect();
        let (mut added, mut updated) = (0, 0);
        for p in files {
            let rel = p
                .strip_prefix(&base)
                .map(|x| x.to_string_lossy().to_string())
                .unwrap_or_else(|_| p.to_string_lossy().to_string());
            let st = match std::fs::metadata(&p) {
                Ok(st) => st,
                Err(e) => {
                    tracing::warn!(path = %p.display(), error = %e, "scan skipped file");
                    continue;
                }
            };
            let modified = match st.modified() {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(path = %p.display(), error = %e, "scan skipped file");
                    continue;
                }
            };
            let mtime = modified
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            if let Some(prev) = by_path.get(&rel).and_then(|sha| m.rows.get(sha)) {
                if prev.bytes == st.len() && (prev.mtime - mtime).abs() < 1e-6 {
                    continue;
                }
            }
            let digest = match sha256_file(&p) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!(path = %p.display(), error = %e, "scan skipped file");
                    continue;
                }
            };
            match m.rows.get_mut(&digest) {
                Some(row) => {
                    row.path = rel;
                    row.bytes = st.len();
                    row.mtime = mtime;
                    updated += 1;
                }
                None => {
                    let parts: Vec<&str> = rel.split('/').collect();
                    let stem = Path::new(&rel)
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_default();
                    let b = stem.as_bytes();
                    let dated = b.len() > 11 && b[4] == b'-' && b[7] == b'-' && b[10] == b'-';
                    m.rows.insert(
                        digest.clone(),
                        Recording {
                            sha256: digest,
                            path: rel.clone(),
                            bytes: st.len(),
                            mtime,
                            duration_ms: if probe { probe_duration_ms(&p) } else { None },
                            source_url: None,
                            podcast: (parts.len() >= 2).then(|| parts[parts.len() - 2].to_string()),
                            title: Some(if dated {
                                stem[11..].to_string()
                            } else {
                                stem.clone()
                            }),
                            date: dated.then(|| stem[..10].to_string()),
                            artifacts: BTreeMap::new(),
                        },
                    );
                    added += 1;
                }
            }
        }
        self.save(m)?;
        Ok((added, updated))
    }

    pub fn recording_path(&self, r: &Recording) -> PathBuf {
        self.recordings_dir().join(&r.path)
    }

    pub fn artifact_path(&self, sha256: &str, kind: &str) -> PathBuf {
        self.artifacts_dir()
            .join(&sha256[..2.min(sha256.len())])
            .join(sha256)
            .join(format!("{kind}.json"))
    }

    pub fn has_artifact(&self, r: &Recording, kind: &str) -> bool {
        r.artifacts.contains_key(kind) && self.artifact_path(&r.sha256, kind).is_file()
    }

    pub fn get_artifact(&self, sha256: &str, kind: &str) -> Result<serde_json::Value> {
        let p = self.artifact_path(sha256, kind);
        let raw = std::fs::read(&p).with_context(|| format!("read {}", p.display()))?;
        Ok(serde_json::from_slice(&raw)?)
    }

    pub fn put_artifact(
        &self,
        m: &mut Manifest,
        sha256: &str,
        kind: &str,
        obj: &serde_json::Value,
    ) -> Result<()> {
        let blob = serde_json::to_vec(obj)?;
        write_atomic(&self.artifact_path(sha256, kind), &blob)?;
        let row = m
            .rows
            .get_mut(sha256)
            .ok_or_else(|| anyhow!("recording {sha256} not in manifest"))?;
        row.artifacts.insert(
            kind.to_string(),
            ArtifactRef {
                sha256: format!("{:x}", Sha256::digest(&blob)),
                producer: producer(kind).to_string(),
                bytes: blob.len() as u64,
                created: now_iso(),
            },
        );
        self.save(m)
    }

    pub fn load_speakers(&self) -> Result<Registry> {
        let p = self.speakers_path();
        if !p.exists() {
            return Ok(Registry::default());
        }
        let raw = std::fs::read(&p)?;
        serde_json::from_slice(&raw).with_context(|| format!("parse {}", p.display()))
    }

    pub fn rename_speaker(&self, id: &str, name: &str) -> Result<Option<Speaker>> {
        let _guard = REGISTRY_GATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut registry = self.load_speakers()?;
        let Some(speaker) = registry.speakers.iter_mut().find(|s| s.id == id) else {
            return Ok(None);
        };
        let name = name.trim();
        speaker.name = (!name.is_empty()).then(|| name.to_string());
        let result = speaker.clone();
        self.save_speakers(&mut registry)?;
        Ok(Some(result))
    }

    pub fn save_speakers(&self, reg: &mut Registry) -> Result<()> {
        reg.updated = Some(now_iso());
        write_atomic(&self.speakers_path(), &serde_json::to_vec_pretty(reg)?)
    }

    pub fn status(&self, m: &Manifest) -> Result<serde_json::Value> {
        let rows = m.by_path();
        let hours = rows
            .iter()
            .map(|r| r.duration_ms.unwrap_or(0) as f64)
            .sum::<f64>()
            / 3.6e6;
        let mut out =
            serde_json::json!({"recordings": rows.len(), "hours": (hours * 10.0).round() / 10.0});
        for kind in KINDS {
            let done = rows.iter().filter(|r| self.has_artifact(r, kind)).count();
            let stale = rows.iter().filter(|r| r.stale(kind)).count();
            out[kind] =
                serde_json::json!({"done": done, "todo": rows.len() - done, "stale": stale});
        }
        let reg = self.load_speakers()?;
        out["speakers"] = serde_json::json!({
            "total": reg.speakers.len(),
            "named": reg.speakers.iter().filter(|s| s.name.is_some()).count(),
        });
        let mut voices: Vec<String> = std::fs::read_dir(self.voices_dir())
            .map(|rd| {
                rd.flatten()
                    .filter(|e| {
                        e.path().join("voice.json").is_file()
                            || e.path().join("current/voice.json").is_file()
                    })
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .collect()
            })
            .unwrap_or_default();
        voices.sort();
        out["voices"] = serde_json::json!(voices);
        Ok(out)
    }
}
