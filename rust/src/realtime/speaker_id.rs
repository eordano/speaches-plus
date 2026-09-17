use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use anyhow::Result;
use serde::Serialize;

use crate::diarization::embedding::EmbeddingModel;
use crate::voices::workspace::{Registry, Workspace};

pub const MIN_SPEECH_SAMPLES_16K: usize = 16_000;

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct SpeakerMatch {
    pub id: String,
    pub name: Option<String>,
    pub score: f32,
}

#[derive(Clone)]
struct Centroid {
    id: String,
    name: Option<String>,
    unit: Vec<f32>,
}

#[derive(Default)]
struct Cache {
    modified: Option<SystemTime>,
    threshold: f32,
    centroids: Arc<Vec<Centroid>>,
}

pub struct SpeakerIdentifier {
    workspace: Workspace,
    embedding: Arc<EmbeddingModel>,
    cache: Mutex<Cache>,
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-9;
    v.iter().map(|x| x / n).collect()
}

pub fn best_match(embedding: &[f32], registry: &Registry) -> Option<SpeakerMatch> {
    let e = unit(embedding);
    registry
        .speakers
        .iter()
        .filter(|s| s.centroid.len() == e.len())
        .map(|s| {
            let score = unit(&s.centroid)
                .iter()
                .zip(&e)
                .map(|(a, b)| a * b)
                .sum::<f32>();
            SpeakerMatch {
                id: s.id.clone(),
                name: s.name.clone().or_else(|| s.name_proposed.clone()),
                score,
            }
        })
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .filter(|m| m.score >= registry.threshold_link)
}

impl SpeakerIdentifier {
    pub fn new(workspace_root: impl Into<PathBuf>, embedding: Arc<EmbeddingModel>) -> Self {
        Self {
            workspace: Workspace::new(workspace_root),
            embedding,
            cache: Mutex::new(Cache::default()),
        }
    }

    pub fn global(models: &crate::models::Models) -> Option<Arc<Self>> {
        static ID: OnceLock<Option<Arc<SpeakerIdentifier>>> = OnceLock::new();
        ID.get_or_init(|| {
            let root = std::env::var_os("NV_VOICES_DATA_DIR")?;
            let embedding = models.diar_embedding.clone()?;
            Some(Arc::new(Self::new(PathBuf::from(root), embedding)))
        })
        .clone()
    }

    fn centroids(&self) -> Result<(f32, Arc<Vec<Centroid>>)> {
        let path = self.workspace.speakers_path();
        let modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if cache.modified == modified && !cache.centroids.is_empty() {
            return Ok((cache.threshold, cache.centroids.clone()));
        }
        let registry = self.workspace.load_speakers()?;
        let centroids: Vec<Centroid> = registry
            .speakers
            .iter()
            .filter(|s| !s.centroid.is_empty())
            .map(|s| Centroid {
                id: s.id.clone(),
                name: s.name.clone().or_else(|| s.name_proposed.clone()),
                unit: unit(&s.centroid),
            })
            .collect();
        *cache = Cache {
            modified,
            threshold: registry.threshold_link,
            centroids: Arc::new(centroids),
        };
        Ok((cache.threshold, cache.centroids.clone()))
    }

    pub fn identify(&self, audio_16k: &[f32]) -> Result<Option<SpeakerMatch>> {
        if audio_16k.len() < MIN_SPEECH_SAMPLES_16K {
            return Ok(None);
        }
        let (threshold, centroids) = self.centroids()?;
        if centroids.is_empty() {
            return Ok(None);
        }
        let take = audio_16k.len().min(30 * 16_000);
        let e = unit(&self.embedding.embed(&audio_16k[..take])?);
        Ok(centroids
            .iter()
            .filter(|c| c.unit.len() == e.len())
            .map(|c| SpeakerMatch {
                id: c.id.clone(),
                name: c.name.clone(),
                score: c.unit.iter().zip(&e).map(|(a, b)| a * b).sum(),
            })
            .max_by(|a, b| a.score.total_cmp(&b.score))
            .filter(|m| m.score >= threshold))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voices::workspace::Speaker;

    fn speaker(id: &str, name: Option<&str>, centroid: Vec<f32>) -> Speaker {
        serde_json::from_value(serde_json::json!({
            "id": id, "name": name, "name_proposed": null, "centroid": centroid,
            "seconds": 0.0, "sources": []
        }))
        .unwrap()
    }

    #[test]
    fn best_match_picks_the_closest_centroid_above_the_link_threshold() {
        let mut reg = Registry::default();
        reg.speakers = vec![
            speaker("spk-0001", Some("Alice"), vec![1.0, 0.0, 0.0]),
            speaker("spk-0002", None, vec![0.0, 3.0, 0.0]),
            speaker("spk-0003", Some("short"), vec![1.0, 0.0]),
        ];
        let m = best_match(&[0.9, 0.1, 0.0], &reg).expect("match");
        assert_eq!(m.id, "spk-0001");
        assert_eq!(m.name.as_deref(), Some("Alice"));
        assert!(m.score > 0.99 - 0.01);
        let m = best_match(&[0.0, 5.0, 0.0], &reg).expect("unnamed clusters still match");
        assert_eq!((m.id.as_str(), m.name), ("spk-0002", None));
        assert!(
            best_match(&[0.5, 0.5, 0.7], &reg).is_none(),
            "below threshold_link is no speaker, not the nearest one"
        );
        assert!(best_match(&[0.0, 0.0, 1.0], &Registry::default()).is_none());
    }
}
