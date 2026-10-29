use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub const META_FILE: &str = "voice.json";
pub const PACK_FILE: &str = "pack.npy";
pub const ADAPTER_FILE: &str = "adapter.npz";
pub const SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VoiceKind {
    QwenXvector,

    KokoroStyle,

    KokoroAdapter,
}

impl VoiceKind {
    pub fn is_kokoro(self) -> bool {
        matches!(self, VoiceKind::KokoroStyle | VoiceKind::KokoroAdapter)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoiceMeta {
    pub schema_version: u32,
    pub name: String,
    pub kind: VoiceKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub embedding: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub design_params: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub producer: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoiceProfile {
    pub schema_version: u32,
    pub name: String,
    pub embedding: Vec<f32>,
    pub design_params: Option<serde_json::Value>,
}

impl From<&VoiceProfile> for VoiceMeta {
    fn from(p: &VoiceProfile) -> Self {
        VoiceMeta {
            schema_version: SCHEMA_VERSION,
            name: p.name.clone(),
            kind: VoiceKind::QwenXvector,
            speaker_id: None,
            embedding: p.embedding.clone(),
            design_params: p.design_params.clone(),
            producer: None,
            eval: None,
            created: None,
        }
    }
}

pub struct VoiceProfileStore {
    root: PathBuf,
}

pub type VoiceStore = VoiceProfileStore;

#[derive(Debug)]
pub struct VoiceStoreUnavailable {
    pub root: PathBuf,
    pub reason: String,
}

impl std::fmt::Display for VoiceStoreUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "voice store {} is unavailable: {}",
            self.root.display(),
            self.reason
        )
    }
}

impl std::error::Error for VoiceStoreUnavailable {}

pub const STORE_REOPEN_BACKOFF: std::time::Duration = std::time::Duration::from_secs(5);

#[derive(Default)]
struct HandleSlot {
    store: Option<std::sync::Arc<VoiceProfileStore>>,
    last_failure: Option<std::time::Instant>,
}

pub struct VoiceStoreHandle {
    root: PathBuf,
    reopen_backoff: std::time::Duration,
    slot: std::sync::Mutex<HandleSlot>,
}

impl VoiceStoreHandle {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            reopen_backoff: STORE_REOPEN_BACKOFF,
            slot: std::sync::Mutex::new(HandleSlot::default()),
        }
    }

    pub fn with_reopen_backoff(mut self, backoff: std::time::Duration) -> Self {
        self.reopen_backoff = backoff;
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HandleSlot> {
        self.slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn unavailable(&self, reason: String) -> anyhow::Error {
        anyhow::Error::new(VoiceStoreUnavailable {
            root: self.root.clone(),
            reason,
        })
    }

    pub fn get(&self) -> Result<std::sync::Arc<VoiceProfileStore>> {
        let (opened, last_failure) = {
            let slot = self.lock();
            (slot.store.clone(), slot.last_failure)
        };
        if let Some(store) = opened {
            if store.root().is_dir() {
                return Ok(store);
            }
            self.lock().store = None;
        }
        if let Some(t) = last_failure {
            if t.elapsed() < self.reopen_backoff {
                return Err(self.unavailable(format!(
                    "reopen attempted less than {:?} ago and failed",
                    self.reopen_backoff
                )));
            }
        }
        match VoiceProfileStore::open(&self.root) {
            Ok(store) => {
                let store = std::sync::Arc::new(store);
                let mut slot = self.lock();
                slot.store = Some(store.clone());
                slot.last_failure = None;
                Ok(store)
            }
            Err(err) => {
                self.lock().last_failure = Some(std::time::Instant::now());
                Err(self.unavailable(format!("{err:#}")))
            }
        }
    }

    pub fn is_live(&self) -> bool {
        self.get().is_ok()
    }
}

pub fn is_store_unavailable(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| cause.downcast_ref::<VoiceStoreUnavailable>().is_some())
}

fn is_safe_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.')
        && !name.starts_with('.')
}

impl VoiceProfileStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;
        let store = Self { root };
        store.migrate_legacy()?;
        Ok(store)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn dir_for(&self, name: &str) -> PathBuf {
        let dir = self.root.join(name);
        match std::fs::read_link(dir.join("current")) {
            Ok(target) => dir.join(target),
            Err(_) => dir,
        }
    }

    pub fn path_for(&self, name: &str) -> PathBuf {
        self.dir_for(name).join(META_FILE)
    }

    pub fn file_path(&self, name: &str, file: &str) -> PathBuf {
        self.dir_for(name).join(file)
    }

    fn migrate_legacy(&self) -> Result<()> {
        for entry in std::fs::read_dir(&self.root)? {
            let path = entry?.path();
            if !path.is_file() || path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let bytes = std::fs::read(&path)?;
            let Ok(profile) = serde_json::from_slice::<VoiceProfile>(&bytes) else {
                continue;
            };
            self.put_meta(&VoiceMeta::from(&profile))
                .with_context(|| format!("migrate legacy voice profile {name}"))?;
            std::fs::remove_file(&path)?;
        }
        Ok(())
    }

    pub fn put_meta(&self, meta: &VoiceMeta) -> Result<()> {
        if !is_safe_name(&meta.name) {
            return Err(anyhow!("unsafe voice name {:?}", meta.name));
        }
        let dir = self.dir_for(&meta.name);
        std::fs::create_dir_all(&dir)?;
        let bytes = serde_json::to_vec_pretty(meta)?;
        let tmp = dir.join(format!("{META_FILE}.tmp"));
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, dir.join(META_FILE))?;
        Ok(())
    }

    pub fn get_meta(&self, name: &str) -> Result<VoiceMeta> {
        if !is_safe_name(name) {
            return Err(anyhow!("unsafe voice name {name:?}"));
        }
        let bytes = std::fs::read(self.path_for(name))
            .with_context(|| format!("voice {name:?} not in store"))?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    pub fn exists(&self, name: &str) -> bool {
        is_safe_name(name) && self.path_for(name).is_file()
    }

    pub fn modified(&self, name: &str) -> Option<SystemTime> {
        std::fs::metadata(self.path_for(name))
            .and_then(|m| m.modified())
            .ok()
    }

    pub fn write_file(&self, name: &str, file: &str, bytes: &[u8]) -> Result<()> {
        if !is_safe_name(name) || !is_safe_name(file) {
            return Err(anyhow!("unsafe voice file {name:?}/{file:?}"));
        }
        let dir = self.dir_for(name);
        std::fs::create_dir_all(&dir)?;
        let tmp = dir.join(format!("{file}.tmp"));
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, dir.join(file))?;
        Ok(())
    }

    pub fn read_file(&self, name: &str, file: &str) -> Result<Vec<u8>> {
        if !is_safe_name(name) || !is_safe_name(file) {
            return Err(anyhow!("unsafe voice file {name:?}/{file:?}"));
        }
        std::fs::read(self.file_path(name, file))
            .with_context(|| format!("voice {name:?}: missing {file}"))
    }

    pub fn put(&self, profile: &VoiceProfile) -> Result<()> {
        self.put_meta(&VoiceMeta::from(profile))
    }

    pub fn get(&self, name: &str) -> Result<VoiceProfile> {
        let meta = self.get_meta(name)?;
        if meta.kind != VoiceKind::QwenXvector {
            return Err(anyhow!(
                "voice {name:?} is a {:?} voice, not an x-vector profile",
                meta.kind
            ));
        }
        Ok(VoiceProfile {
            schema_version: 1,
            name: meta.name,
            embedding: meta.embedding,
            design_params: meta.design_params,
        })
    }

    pub fn delete(&self, name: &str) -> Result<()> {
        if !is_safe_name(name) {
            return Err(anyhow!("unsafe voice name {name:?}"));
        }
        let dir = self.root.join(name);
        if dir.is_dir() {
            std::fs::remove_dir_all(dir)?;
        }
        Ok(())
    }

    pub fn list_all(&self) -> Result<Vec<String>> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.root)? {
            let path = entry?.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    if is_safe_name(name) && self.path_for(name).is_file() {
                        out.push(name.to_string());
                    }
                }
            }
        }
        out.sort();
        Ok(out)
    }

    pub fn list_meta(&self) -> Result<Vec<VoiceMeta>> {
        let mut out = Vec::new();
        for name in self.list_all()? {
            if let Ok(meta) = self.get_meta(&name) {
                out.push(meta);
            }
        }
        Ok(out)
    }

    pub fn list(&self) -> Result<Vec<String>> {
        Ok(self
            .list_meta()?
            .into_iter()
            .filter(|m| m.kind == VoiceKind::QwenXvector)
            .map(|m| m.name)
            .collect())
    }

    pub fn list_kokoro(&self) -> Result<Vec<VoiceMeta>> {
        Ok(self
            .list_meta()?
            .into_iter()
            .filter(|m| m.kind.is_kokoro())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn handle_backs_off_after_a_failed_open_and_reopens_after_it() {
        let blocker = std::env::temp_dir().join(format!(
            "nvtts_handle_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&blocker, b"x").unwrap();
        let root = blocker.join("voices");
        let slow = super::VoiceStoreHandle::new(&root);
        let err = slow.get().err().expect("a file blocks the root");
        assert!(super::is_store_unavailable(&err));
        std::fs::remove_file(&blocker).unwrap();
        let err = slow.get().err().expect("still inside the backoff");
        assert!(
            format!("{err:#}").contains("reopen attempted"),
            "inside the backoff no reopen is tried: {err:#}"
        );
        let fast =
            super::VoiceStoreHandle::new(&root).with_reopen_backoff(std::time::Duration::ZERO);
        assert!(fast.is_live(), "a creatable root opens");
        let _ = std::fs::remove_dir_all(&blocker);
    }

    use super::*;

    fn tmp() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nv-voice-store-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn kokoro_kinds_are_listed_separately_and_not_as_profiles() {
        let store = VoiceProfileStore::open(tmp()).unwrap();
        store
            .put_meta(&VoiceMeta {
                schema_version: SCHEMA_VERSION,
                name: "adapted_voice".into(),
                kind: VoiceKind::KokoroStyle,
                speaker_id: Some("spk-1".into()),
                embedding: vec![],
                design_params: None,
                producer: None,
                eval: None,
                created: None,
            })
            .unwrap();
        store
            .write_file("adapted_voice", PACK_FILE, b"\x93NUMPY")
            .unwrap();
        assert!(store.list().unwrap().is_empty());
        assert_eq!(store.list_kokoro().unwrap().len(), 1);
        assert_eq!(store.list_all().unwrap(), vec!["adapted_voice".to_string()]);
        assert!(store.get("adapted_voice").is_err());
        assert_eq!(
            store.read_file("adapted_voice", PACK_FILE).unwrap(),
            b"\x93NUMPY"
        );
        assert!(store.modified("adapted_voice").is_some());
        store.delete("adapted_voice").unwrap();
        assert!(store.list_all().unwrap().is_empty());
    }

}
