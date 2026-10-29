use crate::json::Json;
use anyhow::{anyhow, bail, Result};
use tokenizers::Tokenizer;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QType {
    pub fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Question {
    pub id: String,
    pub qtype: QType,
    pub instructions: String,
    pub options: Vec<String>,
    pub keys: Vec<String>,
}

pub fn parse_question(id: &str, def: &Json) -> Result<Question> {
    if def.as_obj().is_none() {
        bail!("question {id:?} must be an object");
    }
    let qtype = match def.get("type").and_then(Json::as_str) {
        Some("choice") => QType::Choice,
        Some("score") => QType::Score,
        Some("noul") => QType::Noul,
        other => bail!(
            "question {id:?} has type {}; use choice, score or noul",
            other.map_or("none".into(), |t| format!("{t:?}"))
        ),
    };
    let instructions = match def.get("instructions") {
        Some(text) if !text.state_text().trim().is_empty() => text.state_text(),
        _ => bail!("question {id:?} has empty instructions"),
    };
    let crit = def.get("criteria");
    let (options, keys) = match qtype {
        QType::Choice => match crit {
            Some(Json::Obj(map)) if map.len() >= 2 => {
                let options = map.iter().map(|(k, v)| if v.is_blank() { k.clone() } else { format!("{k}: {}", v.state_text()) }).collect();
                (options, map.iter().map(|(k, _)| k.clone()).collect())
            }
            Some(Json::Arr(list)) if list.len() >= 2 => {
                let keys: Vec<String> = list.iter().map(Json::state_text).collect();
                (keys.clone(), keys)
            }
            _ => bail!("choice question {id:?} needs criteria: a map of two or more options to descriptions"),
        },
        QType::Score => match crit {
            Some(Json::Arr(list)) if list.len() >= 2 => {
                let options = list.iter().enumerate().map(|(i, c)| format!("level {i}: {}", c.state_text())).collect();
                (options, list.iter().map(Json::state_text).collect())
            }
            _ => bail!("score question {id:?} needs criteria: an ordered list of two or more levels, lowest first"),
        },
        QType::Noul => {
            if let Some(extra) = crit.and_then(Json::as_obj).and_then(|c| c.iter().find(|(k, _)| k != "true" && k != "false")) {
                bail!("noul question {id:?} criteria may only hold \"true\" and \"false\", not {:?}", extra.0);
            }
            let side = |name: &str, fallback: &str| {
                let text = match crit.and_then(|c| c.get(name)) {
                    Some(v) if !v.is_blank() => v.state_text(),
                    _ => fallback.to_string(),
                };
                format!("{name}: {text}")
            };
            (
                vec![side("false", "no, the statement does not hold"), side("true", "yes, the statement holds")],
                vec!["false".into(), "true".into()],
            )
        }
    };
    Ok(Question {
        id: id.to_string(),
        qtype,
        instructions,
        options,
        keys,
    })
}

pub struct Special {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub mask_token: String,
}

pub fn encode(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tok
        .encode(text, false)
        .map_err(|e| anyhow!("tokenize: {e}"))?
        .get_ids()
        .to_vec())
}

#[derive(Clone, Debug)]
pub struct Prefix {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
}

pub fn build_prefix(
    tok: &Tokenizer,
    sp: &Special,
    q: &Question,
    head_max_len: usize,
) -> Result<Prefix> {
    let ins = q.instructions.replace(&sp.mask_token, " ");
    let mut head = encode(tok, &format!("{} question: {ins}", q.qtype.name()))?;
    let mut opts = Vec::with_capacity(q.options.len());
    for o in &q.options {
        let mut ids = vec![sp.mask];
        let body = encode(tok, &format!(" {}", o.replace(&sp.mask_token, " ")))?;
        ids.extend(body.into_iter().take(48));
        opts.push(ids);
    }
    let used = |opts: &[Vec<u32>]| opts.iter().map(Vec::len).sum::<usize>() as isize;
    let mut budget = head_max_len as isize - used(&opts);
    if budget < 16 {
        let per = ((head_max_len as isize - 16) / opts.len().max(1) as isize).max(4) as usize;
        for o in &mut opts {
            o.truncate(per);
        }
        budget = head_max_len as isize - used(&opts);
    }
    head.truncate(budget.max(8) as usize);
    let mut ids = Vec::with_capacity(head.len() + used(&opts) as usize + 3);
    ids.push(sp.cls);
    ids.extend(head);
    ids.push(sp.sep);
    let mut markers = Vec::with_capacity(opts.len());
    for o in opts {
        markers.push(ids.len());
        ids.extend(o);
    }
    ids.push(sp.sep);
    Ok(Prefix { ids, markers })
}

pub fn finish_sequence(
    prefix: &Prefix,
    state_ids: &[u32],
    sp: &Special,
    max_len: usize,
) -> (Vec<u32>, Vec<usize>) {
    let room = max_len.saturating_sub(prefix.ids.len() + 1);
    let mut ids = Vec::with_capacity(max_len.min(prefix.ids.len() + state_ids.len() + 1));
    ids.extend_from_slice(&prefix.ids);
    ids.extend_from_slice(&state_ids[..room.min(state_ids.len())]);
    ids.push(sp.sep);
    ids.truncate(max_len);
    (
        ids,
        prefix
            .markers
            .iter()
            .copied()
            .filter(|m| *m < max_len)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ask(text: &str) -> Question {
        parse_question("x", &Json::parse(text).unwrap()).unwrap()
    }

    #[test]
    fn noul_choice_and_score_questions_render_their_options() {
        let q = ask(r#"{"type": "noul", "instructions": "i"}"#);
        assert_eq!(
            q.options,
            [
                "false: no, the statement does not hold",
                "true: yes, the statement holds"
            ]
        );
        let q = ask(
            r#"{"type": "noul", "instructions": "i", "criteria": {"true": "scam", "false": ""}}"#,
        );
        assert_eq!(
            q.options,
            ["false: no, the statement does not hold", "true: scam"]
        );
        let q = ask(
            r#"{"type": "choice", "instructions": "i", "criteria": {"a": null, "b": "bee", "c": 0}}"#,
        );
        assert_eq!(q.options, ["a", "b: bee", "c: 0"], "blank descriptions render as bare keys");
        assert_eq!(q.keys, ["a", "b", "c"]);
        let q = ask(r#"{"type": "score", "instructions": "i", "criteria": ["low", "high"]}"#);
        assert_eq!(q.options, ["level 0: low", "level 1: high"]);
    }

}
