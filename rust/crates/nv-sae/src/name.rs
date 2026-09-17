use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::capture::Span;
use crate::records::Record;
use crate::score::Scored;
use crate::train::Rng;

pub const NAME_STRONG_EXAMPLES: usize = 8;
pub const NAME_SILENT_EXAMPLES: usize = 4;
pub const NAME_EXAMPLE_CHARS: usize = 420;
pub const NAME_MAX_PER_SESSION: usize = 2;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureName {
    pub feature: usize,
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub stance: Option<String>,
    #[serde(default)]
    pub confidence: Option<u32>,
    #[serde(default)]
    pub examples: Vec<usize>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Names {
    pub model: String,
    pub encoder_sha256: String,
    pub names: Vec<FeatureName>,
}

const RECORD_PROMPT: &str = "You are naming one feature of a sparse autoencoder ({width} features, {k} active per message) trained on the mean middle-layer state of a coding assistant reading one chat message or tool call at a time, from logs of human-AI programming sessions. Below are the messages where the feature is strongest, then messages where it is silent. Each is labelled by kind (user, assistant, tool).\n\nDescribe what kind of message activates it: the writer's stance or communicative move (announcing a plan, reporting a root cause, being blocked and asking the human, pasting an error, summarizing finished work, running a particular kind of tool, narrating parallel work), not surface vocabulary. Answer with JSON only:\n{\"name\": \"<2-5 words>\", \"description\": \"<one sentence>\", \"stance\": \"<one of: plan, progress, root-cause, blocked, question, correction, completion-report, error-paste, instruction, tool-read, tool-edit, tool-run, other>\", \"confidence\": <0-100>}\n\nSTRONGEST (value = activation):\n{strong}\n\nSILENT:\n{silent}";

pub struct Namer {
    pub url: String,
    pub model: String,
    http: reqwest::blocking::Client,
}

impl Namer {
    pub fn new(url: &str, model: &str) -> Result<Self> {
        Ok(Namer {
            url: url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            http: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(240))
                .build()?,
        })
    }

    fn ask(&self, prompt: &str) -> Result<serde_json::Value> {
        let body = serde_json::json!({
            "model": self.model, "chat_template_kwargs": {"enable_thinking": false}, "temperature": 0, "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}]
        });
        let mut last = anyhow::anyhow!("no attempts");
        for attempt in 0..4 {
            let r = self
                .http
                .post(format!("{}/v1/chat/completions", self.url))
                .json(&body)
                .send()
                .and_then(|r| r.error_for_status())
                .and_then(|r| r.json::<serde_json::Value>());
            match r {
                Ok(v) => {
                    let content = v
                        .pointer("/choices/0/message/content")
                        .and_then(|c| c.as_str())
                        .unwrap_or("");
                    let (Some(a), Some(b)) = (content.find('{'), content.rfind('}')) else {
                        last = anyhow::anyhow!(
                            "no json in reply: {}",
                            content.chars().take(120).collect::<String>()
                        );
                        continue;
                    };
                    match serde_json::from_str::<serde_json::Value>(&content[a..=b]) {
                        Ok(j) => return Ok(j),
                        Err(e) => last = anyhow::anyhow!("bad json: {e}"),
                    }
                }
                Err(e) => last = e.into(),
            }
            std::thread::sleep(std::time::Duration::from_secs(2 * (attempt + 1)));
        }
        Err(last)
    }

    pub fn name_records(
        &self,
        scored: &Scored,
        records: &[Record],
        features: &[usize],
    ) -> Vec<FeatureName> {
        let n = records.len();
        let dense: Vec<Vec<f32>> = (0..n).map(|i| scored.dense_row(i)).collect();
        let active_per = scored.top_k;
        features
            .iter()
            .map(|&f| {
                let mut order: Vec<usize> = (0..n).filter(|&i| dense[i][f] > 0.0).collect();
                order.sort_by(|a, b| dense[*b][f].partial_cmp(&dense[*a][f]).unwrap());
                let mut per: std::collections::HashMap<&str, usize> =
                    std::collections::HashMap::new();
                let mut strong = Vec::new();
                for i in order {
                    let c = per.entry(records[i].session.as_str()).or_insert(0);
                    if *c >= NAME_MAX_PER_SESSION {
                        continue;
                    }
                    *c += 1;
                    strong.push(i);
                    if strong.len() >= NAME_STRONG_EXAMPLES {
                        break;
                    }
                }
                let mut rng = Rng::new(f as u64 + 1);
                let silent_pool: Vec<usize> = (0..n).filter(|&i| dense[i][f] <= 0.0).collect();
                let silent: Vec<usize> = (0..NAME_SILENT_EXAMPLES.min(silent_pool.len()))
                    .map(|_| silent_pool[rng.below(silent_pool.len())])
                    .collect();
                let fmt = |i: usize| {
                    format!(
                        "({:?}) {}",
                        records[i].kind,
                        clip(&records[i].text, NAME_EXAMPLE_CHARS).replace('\n', " ")
                    )
                    .to_lowercase_kind()
                };
                let strong_txt = if strong.is_empty() {
                    "(never active)".to_string()
                } else {
                    strong
                        .iter()
                        .enumerate()
                        .map(|(j, &i)| format!("{}. [{:.1}] {}", j + 1, dense[i][f], fmt(i)))
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                let silent_txt = silent
                    .iter()
                    .map(|&i| format!("- {}", fmt(i)))
                    .collect::<Vec<_>>()
                    .join("\n");
                let prompt = RECORD_PROMPT
                    .replace("{width}", &scored.width.to_string())
                    .replace("{k}", &active_per.to_string())
                    .replace("{strong}", &strong_txt)
                    .replace("{silent}", &silent_txt);
                let mut out = FeatureName {
                    feature: f,
                    name: None,
                    description: None,
                    stance: None,
                    confidence: None,
                    examples: strong.clone(),
                    error: None,
                };
                match self.ask(&prompt) {
                    Ok(j) => {
                        out.name = j.get("name").and_then(|x| x.as_str()).map(str::to_string);
                        out.description = j
                            .get("description")
                            .and_then(|x| x.as_str())
                            .map(str::to_string);
                        out.stance = j.get("stance").and_then(|x| x.as_str()).map(str::to_string);
                        out.confidence = j
                            .get("confidence")
                            .and_then(|x| x.as_f64())
                            .map(|x| x.clamp(0.0, 100.0) as u32);
                    }
                    Err(e) => out.error = Some(format!("{e:#}")),
                }
                eprintln!("{f} {:?} {:?} {:?}", out.name, out.stance, out.confidence);
                out
            })
            .collect()
    }
}

trait LowerKind {
    fn to_lowercase_kind(self) -> String;
}
impl LowerKind for String {
    fn to_lowercase_kind(self) -> String {
        self.replacen("(User)", "(user)", 1)
            .replacen("(Assistant)", "(assistant)", 1)
            .replacen("(Tool)", "(tool)", 1)
    }
}

pub fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max - 1).collect();
        format!("{cut}…")
    }
}

pub fn sha256_file(path: &std::path::Path) -> Result<String> {
    use sha2::Digest;
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    Ok(format!("{:x}", sha2::Sha256::digest(bytes)))
}

pub const TOKEN_CONTEXT_BEFORE: usize = 18;
pub const TOKEN_CONTEXT_AFTER: usize = 8;
pub const TOKEN_MAX_PER_RECORD: usize = 2;

const TOKEN_PROMPT: &str = "You are naming one feature of a sparse autoencoder ({width} features, {k} active per token) trained on the middle-layer (block 19 of 40) token states of a coding assistant reading exported conversation records. The feature fires on some tokens and not others. Below are the tokens where it fires most strongly (marked «like this», with surrounding text), then a few tokens from records where it never fires.\n\nInfer what the feature detects. Prefer the most specific description that fits all strong cases; if they disagree, say what most share. Answer with JSON only, no prose:\n{\"name\": \"<2-5 word name>\", \"description\": \"<one sentence: what token/context pattern activates it>\", \"stance\": \"<what the marked token itself usually is>\", \"confidence\": <0-100, how certain you are that the strong cases share this one pattern>}\n\nSTRONG ACTIVATIONS (highest first, value = activation / feature scale):\n{strong}\n\nRECORDS WHERE THE FEATURE IS SILENT (random tokens):\n{silent}";

impl Namer {
    pub fn name_tokens(
        &self,
        scored: &Scored,
        spans: &[Span],
        tokenizer: &tokenizers::Tokenizer,
        features: &[usize],
    ) -> Vec<FeatureName> {
        let total = scored.indices.len();
        let mut record_of = vec![usize::MAX; total];
        for (si, sp) in spans.iter().enumerate() {
            for t in sp.start..sp.end {
                record_of[t] = si;
            }
        }
        let ctx = |t: usize| -> String {
            let sp = &spans[record_of[t]];
            let p = t - sp.start;
            let s = p.saturating_sub(TOKEN_CONTEXT_BEFORE);
            let e = (p + TOKEN_CONTEXT_AFTER + 1).min(sp.ids.len());
            let dec = |ids: &[u32]| tokenizer.decode(ids, false).unwrap_or_default();
            format!(
                "{}{}«{}»{}{}",
                if s > 0 { "…" } else { "" },
                dec(&sp.ids[s..p]),
                dec(&sp.ids[p..p + 1]),
                dec(&sp.ids[p + 1..e]),
                if e < sp.ids.len() { "…" } else { "" }
            )
        };
        let mut per_feature: Vec<Vec<(usize, f32)>> = vec![Vec::new(); scored.width];
        for t in 0..total {
            for (f, v) in scored.indices[t].iter().zip(&scored.values[t]) {
                per_feature[*f as usize].push((t, *v));
            }
        }
        features
            .iter()
            .map(|&f| {
                let mut hits = per_feature[f].clone();
                hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let mut per: std::collections::HashMap<usize, usize> =
                    std::collections::HashMap::new();
                let mut strong = Vec::new();
                for (t, v) in hits {
                    let c = per.entry(record_of[t]).or_insert(0);
                    if *c >= TOKEN_MAX_PER_RECORD {
                        continue;
                    }
                    *c += 1;
                    strong.push((t, v));
                    if strong.len() >= NAME_STRONG_EXAMPLES {
                        break;
                    }
                }
                let mut rng = Rng::new(f as u64 + 7);
                let fired: std::collections::HashSet<usize> =
                    per_feature[f].iter().map(|(t, _)| record_of[*t]).collect();
                let silent_records: Vec<usize> =
                    (0..spans.len()).filter(|r| !fired.contains(r)).collect();
                let silent: Vec<usize> = (0..NAME_SILENT_EXAMPLES.min(silent_records.len()))
                    .map(|_| {
                        let sp = &spans[silent_records[rng.below(silent_records.len())]];
                        sp.start + rng.below(sp.end - sp.start)
                    })
                    .collect();
                let strong_txt = if strong.is_empty() {
                    "(never active)".to_string()
                } else {
                    strong
                        .iter()
                        .enumerate()
                        .map(|(j, (t, v))| {
                            format!("{}. [{:.2}] {:?}", j + 1, v / scored.scale[f], ctx(*t))
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                let silent_txt = silent
                    .iter()
                    .map(|t| format!("- {:?}", ctx(*t)))
                    .collect::<Vec<_>>()
                    .join("\n");
                let prompt = TOKEN_PROMPT
                    .replace("{width}", &scored.width.to_string())
                    .replace("{k}", &scored.top_k.to_string())
                    .replace("{strong}", &strong_txt)
                    .replace("{silent}", &silent_txt);
                let mut out = FeatureName {
                    feature: f,
                    name: None,
                    description: None,
                    stance: None,
                    confidence: None,
                    examples: strong.iter().map(|(t, _)| *t).collect(),
                    error: None,
                };
                match self.ask(&prompt) {
                    Ok(j) => {
                        out.name = j.get("name").and_then(|x| x.as_str()).map(str::to_string);
                        out.description = j
                            .get("description")
                            .and_then(|x| x.as_str())
                            .map(str::to_string);
                        out.stance = j.get("stance").and_then(|x| x.as_str()).map(str::to_string);
                        out.confidence = j
                            .get("confidence")
                            .and_then(|x| x.as_f64())
                            .map(|x| x.clamp(0.0, 100.0) as u32);
                    }
                    Err(e) => out.error = Some(format!("{e:#}")),
                }
                eprintln!("{f} {:?} {:?}", out.name, out.confidence);
                out
            })
            .collect()
    }
}
