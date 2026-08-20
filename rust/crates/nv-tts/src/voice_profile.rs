use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoiceProfile {
    pub schema_version: u32,
    pub name: String,
    pub embedding: Vec<f32>,
    pub design_params: Option<serde_json::Value>,
}

pub struct VoiceProfileStore {
    root: PathBuf,
}

impl VoiceProfileStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn path_for(&self, name: &str) -> PathBuf {
        self.root.join(format!("{name}.json"))
    }

    pub fn put(&self, profile: &VoiceProfile) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(profile)?;
        std::fs::write(self.path_for(&profile.name), bytes)?;
        Ok(())
    }

    pub fn get(&self, name: &str) -> Result<VoiceProfile> {
        let bytes = std::fs::read(self.path_for(name))?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    pub fn delete(&self, name: &str) -> Result<()> {
        let path = self.path_for(name);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    pub fn list(&self) -> Result<Vec<String>> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.root)? {
            let path = entry?.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    out.push(name.to_string());
                }
            }
        }
        out.sort();
        Ok(out)
    }
}
