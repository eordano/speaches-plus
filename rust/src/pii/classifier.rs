use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use super::spans::{assemble_spans, PiiSpan};
use super::viterbi::viterbi_decode;
use crate::vad::ort_err;

pub struct PiiClassifier {
    session: Arc<Mutex<Session>>,
    tokenizer: Tokenizer,
    labels: Vec<String>,
}

impl PiiClassifier {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let (model_path, tokenizer_path, config_path) = resolve_layout(model_dir)?;

        let session = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?
            .with_intra_threads(2)
            .map_err(ort_err)?
            .commit_from_file(&model_path)
            .map_err(ort_err)
            .with_context(|| format!("load PII ONNX model {}", model_path.display()))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("load tokenizer {}: {e}", tokenizer_path.display()))?;

        let labels = load_labels(&config_path)?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer,
            labels,
        })
    }

    pub fn classify_one(&self, text: &str) -> Result<Vec<PiiSpan>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenize: {e}"))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let offsets: Vec<(usize, usize)> = encoding
            .get_offsets()
            .iter()
            .map(|&(s, e)| (s, e))
            .collect();
        let n = input_ids.len();

        let logits = self.run_inference(&input_ids, &attention_mask, n)?;
        let l = self.labels.len();
        let t = n;

        let path = viterbi_decode(&logits, t, l, &self.labels);
        let label_names: Vec<String> = path
            .iter()
            .map(|&i| self.labels[i as usize].clone())
            .collect();
        let attn_i32: Vec<i32> = attention_mask.iter().map(|&v| v as i32).collect();

        Ok(assemble_spans(&label_names, &offsets, &attn_i32))
    }

    pub fn classify_batch(&self, texts: &[String]) -> Result<Vec<Vec<PiiSpan>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings: Vec<_> = texts
            .iter()
            .map(|t| {
                self.tokenizer
                    .encode(t.as_str(), true)
                    .map_err(|e| anyhow!("tokenize: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
        let batch_size = texts.len();

        let mut all_input_ids: Vec<i64> = vec![0; batch_size * max_len];
        let mut all_attention_mask: Vec<i64> = vec![0; batch_size * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            for (j, &id) in ids.iter().enumerate() {
                all_input_ids[i * max_len + j] = id as i64;
            }
            for (j, &m) in mask.iter().enumerate() {
                all_attention_mask[i * max_len + j] = m as i64;
            }
        }

        let logits =
            self.run_batch_inference(&all_input_ids, &all_attention_mask, batch_size, max_len)?;
        let l = self.labels.len();

        let mut results = Vec::with_capacity(batch_size);
        for (i, text) in texts.iter().enumerate() {
            if text.trim().is_empty() {
                results.push(Vec::new());
                continue;
            }
            let enc = &encodings[i];
            let t_len: usize = enc.get_attention_mask().iter().map(|&v| v as usize).sum();
            let seq_logits = &logits[i * max_len * l..i * max_len * l + t_len * l];
            let path = viterbi_decode(seq_logits, t_len, l, &self.labels);
            let label_names: Vec<String> = path
                .iter()
                .map(|&idx| self.labels[idx as usize].clone())
                .collect();
            let offsets: Vec<(usize, usize)> = enc
                .get_offsets()
                .iter()
                .take(t_len)
                .map(|&(s, e)| (s, e))
                .collect();
            let attn_i32: Vec<i32> = enc
                .get_attention_mask()
                .iter()
                .take(t_len)
                .map(|&v| v as i32)
                .collect();
            results.push(assemble_spans(&label_names, &offsets, &attn_i32));
        }

        Ok(results)
    }

    fn run_inference(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        n: usize,
    ) -> Result<Vec<f32>> {
        let in_tensor =
            Tensor::<i64>::from_array(([1usize, n], input_ids.to_vec().into_boxed_slice()))
                .map_err(ort_err)?;
        let mask_tensor =
            Tensor::<i64>::from_array(([1usize, n], attention_mask.to_vec().into_boxed_slice()))
                .map_err(ort_err)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("PII session poisoned"))?;
        let outputs = session
            .run(ort::inputs![
                "input_ids" => in_tensor,
                "attention_mask" => mask_tensor,
            ])
            .map_err(ort_err)?;

        let (_, logits_data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        Ok(logits_data.to_vec())
    }

    fn run_batch_inference(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        let in_tensor = Tensor::<i64>::from_array((
            [batch_size, seq_len],
            input_ids.to_vec().into_boxed_slice(),
        ))
        .map_err(ort_err)?;
        let mask_tensor = Tensor::<i64>::from_array((
            [batch_size, seq_len],
            attention_mask.to_vec().into_boxed_slice(),
        ))
        .map_err(ort_err)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("PII session poisoned"))?;
        let outputs = session
            .run(ort::inputs![
                "input_ids" => in_tensor,
                "attention_mask" => mask_tensor,
            ])
            .map_err(ort_err)?;

        let (_, logits_data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        Ok(logits_data.to_vec())
    }
}

fn find_existing(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|c| c.is_file()).cloned()
}

fn first_existing(what: &str, candidates: &[PathBuf]) -> Result<PathBuf> {
    if let Some(found) = find_existing(candidates) {
        return Ok(found);
    }
    let tried: Vec<String> = candidates.iter().map(|p| p.display().to_string()).collect();
    tracing::warn!(file = %what, tried = %tried.join(", "), "PII model asset not found");
    Err(anyhow!("{what} not found; tried: {}", tried.join(", ")))
}

fn push_unique(out: &mut Vec<PathBuf>, dir: Option<&Path>, name: &str) {
    let Some(dir) = dir else { return };
    if dir.as_os_str().is_empty() {
        return;
    }
    let p = dir.join(name);
    if !out.contains(&p) {
        out.push(p);
    }
}

fn is_onnx_file(p: &Path) -> bool {
    p.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("onnx"))
        && p.is_file()
}

fn base_dir(model_dir: &Path) -> &Path {
    if is_onnx_file(model_dir) {
        model_dir.parent().unwrap_or(model_dir)
    } else {
        model_dir
    }
}

fn model_candidates(model_dir: &Path) -> Vec<PathBuf> {
    if is_onnx_file(model_dir) {
        return vec![model_dir.to_path_buf()];
    }
    vec![
        model_dir.join("model.onnx"),
        model_dir.join("onnx").join("model.onnx"),
    ]
}

fn asset_candidates(model_dir: &Path, model_path: &Path, name: &str) -> Vec<PathBuf> {
    let base = base_dir(model_dir);
    let model_home = model_path.parent();
    let mut out = Vec::new();
    push_unique(&mut out, Some(base), name);
    push_unique(&mut out, model_home, name);
    push_unique(&mut out, base.parent(), name);
    push_unique(&mut out, model_home.and_then(Path::parent), name);
    out
}

pub fn layout_present(model_dir: &Path) -> bool {
    let Some(model_path) = find_existing(&model_candidates(model_dir)) else {
        return false;
    };
    ["tokenizer.json", "config.json"]
        .iter()
        .all(|name| find_existing(&asset_candidates(model_dir, &model_path, name)).is_some())
}

fn resolve_layout(model_dir: &Path) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let model_path = first_existing("model.onnx", &model_candidates(model_dir))?;
    let tokenizer_path = first_existing(
        "tokenizer.json",
        &asset_candidates(model_dir, &model_path, "tokenizer.json"),
    )?;
    let config_path = first_existing(
        "config.json",
        &asset_candidates(model_dir, &model_path, "config.json"),
    )?;
    Ok((model_path, tokenizer_path, config_path))
}

fn load_labels(config_path: &Path) -> Result<Vec<String>> {
    let data = std::fs::read_to_string(config_path)
        .with_context(|| format!("read {}", config_path.display()))?;
    let config: serde_json::Value =
        serde_json::from_str(&data).with_context(|| "parse config.json")?;
    let id2label = config
        .get("id2label")
        .ok_or_else(|| anyhow!("config.json missing id2label"))?
        .as_object()
        .ok_or_else(|| anyhow!("id2label is not an object"))?;

    let mut labels: Vec<(usize, String)> = Vec::with_capacity(id2label.len());
    for (k, v) in id2label.iter() {
        let idx: usize = k
            .parse()
            .with_context(|| format!("parse id2label key: {k}"))?;
        let label = v
            .as_str()
            .ok_or_else(|| anyhow!("id2label value not a string for key {k}"))?
            .to_string();
        labels.push((idx, label));
    }
    labels.sort_by_key(|(i, _)| *i);

    let result: Vec<String> = labels.into_iter().map(|(_, l)| l).collect();
    if result.is_empty() {
        return Err(anyhow!("id2label is empty"));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "speaches-plus-pii-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"{}").unwrap();
    }

    #[test]
    fn flat_layout_resolves_in_model_dir() {
        let dir = temp_dir();
        touch(&dir.join("model.onnx"));
        touch(&dir.join("tokenizer.json"));
        touch(&dir.join("config.json"));

        let (m, t, c) = resolve_layout(&dir).unwrap();
        assert_eq!(m, dir.join("model.onnx"));
        assert_eq!(t, dir.join("tokenizer.json"));
        assert_eq!(c, dir.join("config.json"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn split_layout_from_onnx_subdir_finds_tokenizer_in_parent() {
        let snapshot = temp_dir();
        touch(&snapshot.join("onnx").join("model.onnx"));
        touch(&snapshot.join("tokenizer.json"));
        touch(&snapshot.join("config.json"));

        let (m, t, c) = resolve_layout(&snapshot.join("onnx")).unwrap();
        assert_eq!(m, snapshot.join("onnx").join("model.onnx"));
        assert_eq!(t, snapshot.join("tokenizer.json"));
        assert_eq!(c, snapshot.join("config.json"));
        let _ = std::fs::remove_dir_all(&snapshot);
    }

    #[test]
    fn split_layout_from_snapshot_root_finds_model_under_onnx() {
        let snapshot = temp_dir();
        touch(&snapshot.join("onnx").join("model.onnx"));
        touch(&snapshot.join("tokenizer.json"));
        touch(&snapshot.join("config.json"));

        let (m, t, c) = resolve_layout(&snapshot).unwrap();
        assert_eq!(m, snapshot.join("onnx").join("model.onnx"));
        assert_eq!(t, snapshot.join("tokenizer.json"));
        assert_eq!(c, snapshot.join("config.json"));
        let _ = std::fs::remove_dir_all(&snapshot);
    }

    #[test]
    fn snapshot_root_finds_assets_that_only_exist_under_onnx() {
        let snapshot = temp_dir();
        touch(&snapshot.join("onnx").join("model.onnx"));
        touch(&snapshot.join("onnx").join("tokenizer.json"));
        touch(&snapshot.join("onnx").join("config.json"));

        let (m, t, c) = resolve_layout(&snapshot).unwrap();
        assert_eq!(m, snapshot.join("onnx").join("model.onnx"));
        assert_eq!(t, snapshot.join("onnx").join("tokenizer.json"));
        assert_eq!(c, snapshot.join("onnx").join("config.json"));
        let _ = std::fs::remove_dir_all(&snapshot);
    }

    #[test]
    fn a_direct_onnx_file_path_resolves_against_its_own_directory() {
        let snapshot = temp_dir();
        touch(&snapshot.join("onnx").join("model.onnx"));
        touch(&snapshot.join("tokenizer.json"));
        touch(&snapshot.join("config.json"));

        let (m, t, c) = resolve_layout(&snapshot.join("onnx").join("model.onnx")).unwrap();
        assert_eq!(m, snapshot.join("onnx").join("model.onnx"));
        assert_eq!(t, snapshot.join("tokenizer.json"));
        assert_eq!(c, snapshot.join("config.json"));
        let _ = std::fs::remove_dir_all(&snapshot);
    }

    #[test]
    fn layout_present_accepts_a_snapshot_root_where_the_flat_probe_fails() {
        let snapshot = temp_dir();
        touch(&snapshot.join("onnx").join("model.onnx"));
        touch(&snapshot.join("tokenizer.json"));
        touch(&snapshot.join("config.json"));

        assert!(!snapshot.join("model.onnx").exists());
        assert!(layout_present(&snapshot));
        assert!(layout_present(&snapshot.join("onnx")));
        let _ = std::fs::remove_dir_all(&snapshot);
    }

    #[test]
    fn layout_present_is_false_without_a_model_or_without_the_tokenizer() {
        let root = temp_dir();
        let snapshot = root.join("snap");
        std::fs::create_dir_all(&snapshot).unwrap();
        touch(&snapshot.join("tokenizer.json"));
        touch(&snapshot.join("config.json"));
        assert!(!layout_present(&snapshot));

        touch(&snapshot.join("onnx").join("model.onnx"));
        std::fs::remove_file(snapshot.join("tokenizer.json")).unwrap();
        assert!(!layout_present(&snapshot));

        assert!(!layout_present(&snapshot.join("does-not-exist")));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn missing_tokenizer_error_lists_every_path_tried() {
        let snapshot = temp_dir();
        touch(&snapshot.join("onnx").join("model.onnx"));

        let err = resolve_layout(&snapshot.join("onnx"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("tokenizer.json not found"), "{err}");
        assert!(
            err.contains(
                &snapshot
                    .join("onnx")
                    .join("tokenizer.json")
                    .display()
                    .to_string()
            ),
            "{err}"
        );
        assert!(
            err.contains(&snapshot.join("tokenizer.json").display().to_string()),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&snapshot);
    }
}
