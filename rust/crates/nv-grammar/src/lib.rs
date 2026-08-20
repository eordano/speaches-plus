use anyhow::{bail, Context, Result};
use regex_automata::{
    dfa::{dense, Automaton, StartKind},
    util::primitives::StateID,
    Anchored, Input,
};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

type Dfa = dense::DFA<Vec<u32>>;

fn build_anchored_dfa_uncached(pattern: &str) -> Result<Dfa> {
    let anchored = format!(r"\A(?:{pattern})\z");
    dense::Builder::new()
        .configure(dense::Config::new().start_kind(StartKind::Anchored))
        .build(&anchored)
        .with_context(|| format!("build anchored dfa for /{anchored}/"))
}

fn build_anchored_dfa(pattern: &str) -> Result<Arc<Dfa>> {
    use std::sync::Mutex;
    static CACHE: std::sync::OnceLock<Mutex<HashMap<String, Arc<Dfa>>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(d) = cache.lock().unwrap().get(pattern) {
        return Ok(d.clone());
    }
    let dfa = Arc::new(build_anchored_dfa_uncached(pattern)?);
    cache
        .lock()
        .unwrap()
        .insert(pattern.to_string(), dfa.clone());
    Ok(dfa)
}

#[derive(Clone, Debug)]
pub enum GrammarSpec {
    JsonSchema(Value),
    Regex(String),
}

pub fn schema_to_regex(schema: &Value) -> Result<String> {
    let mut out = String::new();
    emit_node(schema, &mut out).context("compile json schema to regex")?;
    Ok(out)
}

pub fn choice_to_regex(choices: &[String]) -> String {
    if choices.is_empty() {
        return String::new();
    }
    let alts: Vec<String> = choices.iter().map(|c| regex_escape(c)).collect();
    format!("({})", alts.join("|"))
}

pub fn json_value_regex(max_depth: usize) -> String {
    let leaf = format!("({STRING}|{NUMBER}|true|false|null)");
    let mut value = leaf.clone();
    for _ in 0..max_depth {
        let member = format!("{STRING}{WS}:{WS}{value}");
        let object = format!(r"\{{{WS}({member}({WS},{WS}{member})*)?{WS}\}}");
        let array = format!(r"\[{WS}({value}({WS},{WS}{value})*)?{WS}\]");
        value = format!("({leaf}|{object}|{array})");
    }
    value
}

pub fn json_object_regex(max_depth: usize) -> String {
    let value = json_value_regex(max_depth);
    let member = format!("{STRING}{WS}:{WS}{value}");
    format!(r"\{{{WS}({member}({WS},{WS}{member})*)?{WS}\}}")
}

const WS: &str = r"[ \t\n]{0,20}";
const STRING: &str = r#""([^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*""#;

const INTEGER: &str = r"-?(0|[1-9][0-9]{0,19})";
const NUMBER: &str = r"-?(0|[1-9][0-9]{0,19})(\.[0-9]{1,17})?([eE][-+]?[0-9]{1,4})?";

fn emit_node(node: &Value, out: &mut String) -> Result<()> {
    if let Some(en) = node.get("enum").and_then(|v| v.as_array()) {
        let alts: Vec<String> = en
            .iter()
            .map(|v| regex_escape(&serde_json::to_string(v).unwrap_or_default()))
            .collect();
        out.push('(');
        out.push_str(&alts.join("|"));
        out.push(')');
        return Ok(());
    }
    if let Some(c) = node.get("const") {
        out.push_str(&regex_escape(&serde_json::to_string(c).unwrap_or_default()));
        return Ok(());
    }

    let ty = node.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match ty {
        "string" => out.push_str(STRING),
        "integer" => out.push_str(INTEGER),
        "number" => out.push_str(NUMBER),
        "boolean" => out.push_str("(true|false)"),
        "null" => out.push_str("null"),
        "object" => emit_object(node, out)?,
        "array" => emit_array(node, out)?,
        "" => bail!("schema node missing 'type' (and no enum/const): {node}"),
        other => bail!("unsupported schema type '{other}'"),
    }
    Ok(())
}

fn emit_object(node: &Value, out: &mut String) -> Result<()> {
    let props = node
        .get("properties")
        .and_then(|v| v.as_object())
        .context("object schema needs 'properties'")?;
    let required: Vec<&str> = node
        .get("required")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_else(|| props.keys().map(|s| s.as_str()).collect());

    out.push_str(r"\{");
    out.push_str(WS);
    for (i, key) in required.iter().enumerate() {
        let sub = props
            .get(*key)
            .with_context(|| format!("required key '{key}' not in properties"))?;
        if i > 0 {
            out.push_str(WS);
            out.push(',');
            out.push_str(WS);
        }
        out.push('"');
        out.push_str(&regex_escape(key));
        out.push('"');
        out.push_str(WS);
        out.push(':');
        out.push_str(WS);
        emit_node(sub, out)?;
    }
    out.push_str(WS);
    out.push_str(r"\}");
    Ok(())
}

fn emit_array(node: &Value, out: &mut String) -> Result<()> {
    let items = node.get("items").context("array schema needs 'items'")?;
    let mut item = String::new();
    emit_node(items, &mut item)?;
    out.push_str(r"\[");
    out.push_str(WS);
    out.push_str("((");
    out.push_str(&item);
    out.push_str(")(");
    out.push_str(WS);
    out.push(',');
    out.push_str(WS);
    out.push('(');
    out.push_str(&item);
    out.push_str("))*)?");
    out.push_str(WS);
    out.push_str(r"\]");
    Ok(())
}

fn regex_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        if "\\^$.|?*+()[]{}".contains(c) {
            o.push('\\');
        }
        o.push(c);
    }
    o
}

pub struct JsonConstraint {
    dfa: Arc<Dfa>,
    state: StateID,
    dead: bool,
}

impl JsonConstraint {
    pub fn from_schema(schema: &Value) -> Result<Self> {
        Self::from_regex(&schema_to_regex(schema)?)
    }

    pub fn from_regex(pattern: &str) -> Result<Self> {
        let dfa = build_anchored_dfa(pattern)?;
        let state = dfa
            .start_state_forward(&Input::new("").anchored(Anchored::Yes))
            .map_err(|e| anyhow::anyhow!("dfa start: {e}"))?;
        Ok(Self {
            dfa,
            state,
            dead: false,
        })
    }

    fn peek(&self, state: StateID, byte: u8) -> Option<StateID> {
        let ns = self.dfa.next_state(state, byte);
        if self.dfa.is_dead_state(ns) {
            None
        } else {
            Some(ns)
        }
    }

    pub fn can_terminate(&self) -> bool {
        if self.dead {
            return false;
        }
        let eoi = self.dfa.next_eoi_state(self.state);
        self.dfa.is_match_state(eoi)
    }

    pub fn advance_str(&mut self, s: &str) -> bool {
        for &b in s.as_bytes() {
            match self.peek(self.state, b) {
                Some(ns) => self.state = ns,
                None => {
                    self.dead = true;
                    return false;
                }
            }
        }
        true
    }

    pub fn accepts_str(&self, s: &str) -> bool {
        if self.dead {
            return false;
        }
        let mut st = self.state;
        for &b in s.as_bytes() {
            match self.peek(st, b) {
                Some(ns) => st = ns,
                None => return false,
            }
        }
        true
    }

    pub fn token_mask(&self, token_strs: &[&str]) -> Vec<bool> {
        token_strs.iter().map(|t| self.accepts_str(t)).collect()
    }
}

pub struct VocabBytes {
    bytes: Vec<Box<[u8]>>,
    is_eos: Vec<bool>,
    oov_eos: Vec<u32>,
}

impl VocabBytes {
    pub fn new(token_bytes: Vec<Vec<u8>>, eos_ids: &[u32]) -> Self {
        let n = token_bytes.len();
        let mut is_eos = vec![false; n];
        let mut oov_eos = Vec::new();
        for &e in eos_ids {
            if (e as usize) < n {
                is_eos[e as usize] = true;
            } else {
                eprintln!(
                    "[nv-grammar] WARN: eos id {e} is outside the tokenizer vocab ({n}); \
                     treating it as an EOS in the padded lm_head range"
                );
                oov_eos.push(e);
            }
        }
        oov_eos.sort_unstable();
        oov_eos.dedup();
        Self {
            bytes: token_bytes
                .into_iter()
                .map(|v| v.into_boxed_slice())
                .collect(),
            is_eos,
            oov_eos,
        }
    }

    fn is_oov_eos(&self, id: u32) -> bool {
        self.oov_eos.binary_search(&id).is_ok()
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

struct StateRow {
    allowed: Vec<u64>,
    can_terminate: bool,
}

pub struct GuidedDecoder {
    dfa: Arc<Dfa>,
    vocab: Arc<VocabBytes>,
    state: StateID,

    rows: HashMap<StateID, Arc<StateRow>>,

    computed: usize,
    dead: bool,

    defer_until: Vec<u32>,
    defer_matched: usize,
}

struct RowEntry {
    row: Arc<StateRow>,
    _dfa: Arc<Dfa>,
    _vocab: Arc<VocabBytes>,
}

type RowKey = (usize, usize, StateID);

fn row_cache() -> &'static std::sync::Mutex<HashMap<RowKey, RowEntry>> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<RowKey, RowEntry>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

fn contiguous_suffix_match_len(seq: &[u32], matched: usize, id: u32) -> usize {
    if seq.get(matched) == Some(&id) {
        return matched + 1;
    }
    let mut k = matched;
    while k > 0 {
        k -= 1;
        if seq[k] == id && seq[matched - k..matched] == seq[..k] {
            return k + 1;
        }
    }
    0
}

impl GuidedDecoder {
    pub fn from_schema(schema: &Value, vocab: Arc<VocabBytes>) -> Result<Self> {
        Self::from_regex(&schema_to_regex(schema)?, vocab)
    }

    pub fn from_grammar(spec: &GrammarSpec, vocab: Arc<VocabBytes>) -> Result<Self> {
        match spec {
            GrammarSpec::JsonSchema(s) => Self::from_schema(s, vocab),
            GrammarSpec::Regex(r) => Self::from_regex(r, vocab),
        }
    }

    pub fn from_regex(pattern: &str, vocab: Arc<VocabBytes>) -> Result<Self> {
        let dfa = build_anchored_dfa(pattern)?;
        let state = dfa
            .start_state_forward(&Input::new("").anchored(Anchored::Yes))
            .map_err(|e| anyhow::anyhow!("dfa start: {e}"))?;
        Ok(Self {
            dfa,
            vocab,
            state,
            rows: HashMap::new(),
            computed: 0,
            dead: false,
            defer_until: Vec::new(),
            defer_matched: 0,
        })
    }

    pub fn is_dead(&self) -> bool {
        self.dead
    }

    pub fn set_defer_until_token(&mut self, close_id: u32) {
        self.set_defer_until_sequence(&[close_id]);
    }

    pub fn set_defer_until_sequence(&mut self, close_ids: &[u32]) {
        self.defer_until = close_ids.to_vec();
        self.defer_matched = 0;
    }

    pub fn deferred(&self) -> bool {
        !self.defer_until.is_empty()
    }

    pub fn defer_token(&self) -> Option<u32> {
        self.defer_until.get(self.defer_matched).copied()
    }

    #[must_use = "a close token the logits row cannot hold can never be forced or sampled, so the \
                  deferred grammar never arms and the caller silently receives unconstrained text"]
    pub fn close_token_outside_logits(&self, logits_len: usize) -> Option<u32> {
        self.defer_until
            .iter()
            .copied()
            .find(|&id| id as usize >= logits_len)
    }

    #[must_use = "false means nothing was masked and the deferral still stands; dropping it leaves \
                  the grammar permanently inert"]
    pub fn mask_to_defer_token(&self, logits: &mut [f32]) -> bool {
        let Some(close) = self.defer_token() else {
            return false;
        };
        if logits.get(close as usize).is_none() {
            return false;
        }
        for (id, logit) in logits.iter_mut().enumerate() {
            if id as u32 != close {
                *logit = f32::NEG_INFINITY;
            }
        }
        true
    }

    fn walk(&self, mut st: StateID, bytes: &[u8]) -> Option<StateID> {
        if bytes.is_empty() {
            return None;
        }
        for &b in bytes {
            let ns = self.dfa.next_state(st, b);
            if self.dfa.is_dead_state(ns) {
                return None;
            }
            st = ns;
        }
        Some(st)
    }

    fn compute_row(&self, state: StateID) -> StateRow {
        let n = self.vocab.len();
        let nwords = n.div_ceil(64);
        let mut allowed = vec![0u64; nwords];
        for (id, bytes) in self.vocab.bytes.iter().enumerate() {
            if self.vocab.is_eos[id] {
                continue;
            }
            if self.walk(state, bytes).is_some() {
                allowed[id >> 6] |= 1u64 << (id & 63);
            }
        }
        let eoi = self.dfa.next_eoi_state(state);
        StateRow {
            allowed,
            can_terminate: self.dfa.is_match_state(eoi),
        }
    }

    fn row(&mut self) -> Arc<StateRow> {
        if let Some(r) = self.rows.get(&self.state) {
            return r.clone();
        }
        let key: RowKey = (
            Arc::as_ptr(&self.dfa) as usize,
            Arc::as_ptr(&self.vocab) as usize,
            self.state,
        );
        if let Some(hit) = row_cache()
            .lock()
            .ok()
            .and_then(|c| c.get(&key).map(|e| e.row.clone()))
        {
            self.rows.insert(self.state, hit.clone());
            return hit;
        }

        let r = Arc::new(self.compute_row(self.state));
        self.computed += 1;
        if let Ok(mut c) = row_cache().lock() {
            c.entry(key).or_insert_with(|| RowEntry {
                row: r.clone(),
                _dfa: self.dfa.clone(),
                _vocab: self.vocab.clone(),
            });
        }
        self.rows.insert(self.state, r.clone());
        r
    }

    pub fn can_terminate(&self) -> bool {
        if self.dead {
            return false;
        }
        let eoi = self.dfa.next_eoi_state(self.state);
        self.dfa.is_match_state(eoi)
    }

    pub fn apply_mask(&mut self, logits: &mut [f32]) {
        if self.dead {
            logits.fill(f32::NEG_INFINITY);
            return;
        }
        if self.deferred() {
            return;
        }
        let row = self.row();
        let n = logits.len().min(self.vocab.len());
        for (id, logit) in logits.iter_mut().take(n).enumerate() {
            if self.vocab.is_eos[id] {
                if !row.can_terminate {
                    *logit = f32::NEG_INFINITY;
                }
            } else if (row.allowed[id >> 6] >> (id & 63)) & 1 == 0 {
                *logit = f32::NEG_INFINITY;
            }
        }

        let spare_oov_eos = row.can_terminate && !self.vocab.oov_eos.is_empty();
        for (id, logit) in logits.iter_mut().enumerate().skip(n) {
            if spare_oov_eos && self.vocab.is_oov_eos(id as u32) {
                continue;
            }
            *logit = f32::NEG_INFINITY;
        }
    }

    #[must_use = "false means this token left the DFA and killed the grammar: the decoder is dead, \
                  it can never accept another token, and every later apply_mask can only answer \
                  'no legal token'. A caller that drops it keeps generating past the point where \
                  the schema stopped being enforced and returns unconstrained text under HTTP 200"]
    pub fn advance(&mut self, id: u32) -> bool {
        if self.dead {
            return false;
        }

        if self.deferred() {
            self.defer_matched =
                contiguous_suffix_match_len(&self.defer_until, self.defer_matched, id);
            if self.defer_matched == self.defer_until.len() {
                self.defer_until.clear();
                self.defer_matched = 0;
            }
            return true;
        }
        let idx = id as usize;
        if idx >= self.vocab.len() {
            if self.vocab.is_oov_eos(id) {
                return true;
            }
            self.dead = true;
            return false;
        }
        if self.vocab.is_eos[idx] {
            return true;
        }
        match self.walk(self.state, &self.vocab.bytes[idx]) {
            Some(ns) => {
                self.state = ns;
                true
            }
            None => {
                self.dead = true;
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn full_match(pattern: &str, s: &str) -> bool {
        let mut c = JsonConstraint::from_regex(pattern).unwrap();
        c.advance_str(s) && c.can_terminate()
    }

    #[test]
    fn scalar_types() {
        assert!(full_match(
            &schema_to_regex(&json!({"type":"integer"})).unwrap(),
            "-42"
        ));
        assert!(!full_match(
            &schema_to_regex(&json!({"type":"integer"})).unwrap(),
            "4.2"
        ));
        assert!(full_match(
            &schema_to_regex(&json!({"type":"number"})).unwrap(),
            "3.14"
        ));
        assert!(full_match(
            &schema_to_regex(&json!({"type":"boolean"})).unwrap(),
            "true"
        ));
        assert!(full_match(
            &schema_to_regex(&json!({"type":"string"})).unwrap(),
            "\"hi\""
        ));
        assert!(!full_match(
            &schema_to_regex(&json!({"type":"string"})).unwrap(),
            "hi"
        ));
    }

    #[test]
    fn enum_constraint() {
        let re = schema_to_regex(&json!({"enum":["red","green",2]})).unwrap();
        assert!(full_match(&re, "\"red\""));
        assert!(full_match(&re, "2"));
        assert!(!full_match(&re, "\"blue\""));
    }

    #[test]
    fn json_object_grammar_accepts_and_rejects() {
        let re = json_object_regex(3);

        let v = vocab(&["x", ""], &[1]);
        assert!(
            GuidedDecoder::from_regex(&re, v).is_ok(),
            "json_object DFA builds"
        );
        assert!(full_match(&re, r#"{"a":1}"#));
        assert!(full_match(
            &re,
            r#"{"a": "hi", "b": [1, 2, 3], "c": {"d": true}}"#
        ));
        assert!(full_match(&re, "{}"));
        assert!(!full_match(&re, r#"{"a":1}x"#), "trailing junk rejected");
        assert!(
            !full_match(&re, "[1,2,3]"),
            "top-level array is not an object"
        );
        assert!(!full_match(&re, "42"), "scalar is not an object");
    }

    #[test]
    fn no_content_tokens_allowed_after_complete_object() {
        let schema =
            json!({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]});

        let v = vocab(&["{", "\"a\"", ":", "1", "}", "\n", "x", "{", ""], &[8]);
        let mut d = GuidedDecoder::from_schema(&schema, v).unwrap();
        for t in [0u32, 1, 2, 3, 4] {
            assert!(d.advance(t), "advance token {t} of {{\"a\":1}}");
        }
        assert!(d.can_terminate(), "object is complete");
        let allowed = masked_allowed(&mut d, 9);
        assert!(!allowed[5], "newline after }} must be masked");
        assert!(!allowed[6], "letter after }} must be masked");
        assert!(!allowed[7], "another {{ after }} must be masked");
        assert!(allowed[8], "eos must be allowed");
    }

    #[test]
    fn weather_tool_args_schema_builds() {
        let schema = json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location", "unit"]
        });
        let re = schema_to_regex(&schema).expect("schema->regex");
        eprintln!("weather regex = {re}");

        let v = vocab(
            &[
                "{",
                "\"",
                "location",
                "\":",
                "\"Paris\"",
                ",",
                "unit",
                "\"celsius\"",
                "}",
                "",
            ],
            &[9],
        );
        let d = GuidedDecoder::from_schema(&schema, v);
        assert!(d.is_ok(), "from_schema failed: {:?}", d.err());
        assert!(full_match(&re, r#"{"location":"Paris","unit":"celsius"}"#));
    }

    #[test]
    fn object_required_fields() {
        let schema = json!({
            "type":"object",
            "properties":{"name":{"type":"string"},"age":{"type":"integer"}},
            "required":["name","age"]
        });
        let re = schema_to_regex(&schema).unwrap();
        assert!(full_match(&re, r#"{"name":"Ada","age":36}"#));
        assert!(full_match(&re, "{ \"name\" : \"Ada\" , \"age\" : 36 }"));
        assert!(
            !full_match(&re, r#"{"name":"Ada"}"#),
            "missing required age"
        );
        assert!(
            !full_match(&re, r#"{"age":36,"name":"Ada"}"#),
            "wrong key order"
        );
    }

    #[test]
    fn nested_array_of_objects() {
        let schema = json!({
            "type":"array",
            "items":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}
        });
        let re = schema_to_regex(&schema).unwrap();
        assert!(full_match(&re, "[]"));
        assert!(full_match(&re, r#"[{"x":1}]"#));
        assert!(full_match(&re, r#"[{"x":1},{"x":2}]"#));
        assert!(!full_match(&re, r#"[{"x":1} {"x":2}]"#), "missing comma");
    }

    #[test]
    fn incremental_prefix_and_terminate() {
        let schema =
            json!({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]});
        let mut c = JsonConstraint::from_schema(&schema).unwrap();
        assert!(!c.can_terminate());
        assert!(c.advance_str(r#"{"ok":"#));
        assert!(!c.can_terminate(), "incomplete object");
        assert!(!c.accepts_str("9"));
        assert!(c.accepts_str("true"));
        assert!(c.advance_str("true}"));
        assert!(c.can_terminate());
    }

    #[test]
    fn token_mask_filters_vocab() {
        let c = JsonConstraint::from_schema(&json!({"type":"boolean"})).unwrap();
        let vocab = ["tr", "fa", "xy", "true", "9", "{"];
        let mask = c.token_mask(&vocab);
        assert_eq!(mask, vec![true, true, false, true, false, false]);
    }

    #[test]
    fn dead_after_invalid() {
        let mut c = JsonConstraint::from_schema(&json!({"type":"integer"})).unwrap();
        assert!(!c.advance_str("x"));
        assert!(!c.can_terminate());
        assert!(!c.accepts_str("1"));
    }

    fn vocab(strs: &[&str], eos: &[u32]) -> Arc<VocabBytes> {
        let bytes: Vec<Vec<u8>> = strs.iter().map(|s| s.as_bytes().to_vec()).collect();
        Arc::new(VocabBytes::new(bytes, eos))
    }

    fn masked_allowed(d: &mut GuidedDecoder, n: usize) -> Vec<bool> {
        let mut logits = vec![0.0f32; n];
        d.apply_mask(&mut logits);
        logits.iter().map(|&x| x.is_finite()).collect()
    }

    #[test]
    fn deferred_grammar_arms_after_think_close() {
        let v = vocab(&["true", "false", "blah", "</think>", ""], &[4]);
        let mut d = GuidedDecoder::from_schema(&json!({"type":"boolean"}), v).unwrap();
        d.set_defer_until_token(3);

        let a = masked_allowed(&mut d, 5);
        assert_eq!(a, vec![true; 5], "no constraint while thinking");
        assert!(d.advance(2), "CoT tokens pass without walking the DFA");
        assert!(d.advance(2));
        assert!(d.deferred());

        assert!(d.advance(3), "</think> arms the grammar");
        assert!(!d.deferred());
        let a = masked_allowed(&mut d, 5);
        assert_eq!(
            a,
            vec![true, true, false, false, false],
            "post-think tokens are constrained"
        );
        assert!(d.advance(0));
        assert!(d.can_terminate());
    }

    #[test]
    fn deferral_edge_semantics_are_pinned() {
        let v = vocab(&["true", "false", "blah", "</think>", ""], &[4]);
        let mut d = GuidedDecoder::from_schema(&json!({"type":"boolean"}), v.clone()).unwrap();
        d.set_defer_until_token(3);

        let mut logits = vec![0.0f32; 8];
        d.apply_mask(&mut logits);
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "while deferred nothing is masked, padding tail included"
        );
        assert!(d.advance(4), "EOS passes through during deferral");
        assert!(
            d.advance(7),
            "out-of-vocab id during deferral does not kill"
        );
        assert!(!d.is_dead());

        assert!(d.advance(3), "arm");
        assert!(
            !d.advance(3),
            "close id after arming walks '</think>' bytes and dies"
        );
        assert!(d.is_dead());

        let mut d2 = GuidedDecoder::from_schema(&json!({"type":"boolean"}), v).unwrap();
        d2.set_defer_until_token(3);
        assert!(d2.advance(3));
        assert!(
            !d2.advance(7),
            "out-of-vocab id after arming kills as usual"
        );
    }

    fn harmony_vocab() -> Arc<VocabBytes> {
        vocab(
            &[
                "true",
                "false",
                "blah",
                "<|end|>",
                "<|start|>",
                "final",
                "<|message|>",
                "",
            ],
            &[7],
        )
    }

    const HARMONY_CLOSE: [u32; 4] = [3, 4, 5, 6];

    fn harmony_decoder() -> GuidedDecoder {
        let mut d =
            GuidedDecoder::from_schema(&json!({"type":"boolean"}), harmony_vocab()).unwrap();
        d.set_defer_until_sequence(&HARMONY_CLOSE);
        d
    }

    #[test]
    fn a_close_sequence_arms_only_when_the_whole_of_it_is_contiguous() {
        let mut d = harmony_decoder();
        for (i, t) in HARMONY_CLOSE.iter().take(3).enumerate() {
            assert!(d.advance(*t));
            assert!(
                d.deferred(),
                "token {i} of the close sequence is a prefix, not a close"
            );
        }
        assert!(
            d.advance(2),
            "the model went on thinking instead of closing"
        );
        assert!(d.deferred(), "a prefix followed by prose is coincidence");
        assert_eq!(
            d.defer_token(),
            Some(HARMONY_CLOSE[0]),
            "the broken match restarts at the head rather than sticking mid-sequence"
        );

        assert!(d.advance(HARMONY_CLOSE[3]));
        assert!(
            d.deferred(),
            "every token of the sequence was emitted, but with prose in the middle: a close \
             marker split by thought is not the model closing its thought"
        );

        for t in HARMONY_CLOSE {
            assert!(d.advance(t));
        }
        assert!(!d.deferred(), "the contiguous whole sequence arms");
        assert_eq!(
            masked_allowed(&mut d, 8),
            vec![true, true, false, false, false, false, false, false],
            "post-close tokens are constrained"
        );

        let mut empty =
            GuidedDecoder::from_schema(&json!({"type":"boolean"}), harmony_vocab()).unwrap();
        empty.set_defer_until_sequence(&[]);
        assert!(
            !empty.deferred(),
            "an empty close sequence is no deferral at all, not a grammar that never arms"
        );
    }

    #[test]
    fn a_close_sequence_whose_head_repeats_resumes_at_the_longest_live_match() {
        let mut d =
            GuidedDecoder::from_schema(&json!({"type":"boolean"}), harmony_vocab()).unwrap();
        d.set_defer_until_sequence(&[3, 3, 4]);
        for t in [3u32, 3, 3, 4] {
            assert!(d.advance(t));
        }
        assert!(
            !d.deferred(),
            "the last three tokens were the close sequence; resetting all progress on the \
             repeated head loses a real close"
        );
    }

    #[test]
    fn the_forced_close_emits_the_sequence_one_token_per_step_and_then_arms() {
        let mut d = harmony_decoder();
        for (i, want) in HARMONY_CLOSE.iter().enumerate() {
            let mut logits = vec![0.0f32; 8];
            assert!(
                d.mask_to_defer_token(&mut logits),
                "step {i}: a spent thinking budget must still have a token to force"
            );
            let legal: Vec<u32> = (0..8u32)
                .filter(|id| logits[*id as usize].is_finite())
                .collect();
            assert_eq!(
                legal,
                vec![*want],
                "step {i}: the forced close must walk the sequence one token per step"
            );
            assert!(d.advance(*want));
        }
        assert!(
            !d.deferred(),
            "a forced close that never finishes the sequence is a grammar that never arms"
        );
        assert!(
            !d.mask_to_defer_token(&mut vec![0.0f32; 8]),
            "nothing is left to force once the grammar is live"
        );
        assert_eq!(
            masked_allowed(&mut d, 8),
            vec![true, true, false, false, false, false, false, false],
            "the schema constrains the tokens the reserve left room for"
        );
    }

    #[test]
    fn a_close_token_the_logits_row_cannot_hold_is_reported_before_the_head_looks_forceable() {
        let d = harmony_decoder();
        let last = HARMONY_CLOSE[HARMONY_CLOSE.len() - 1] as usize;
        assert_eq!(
            d.close_token_outside_logits(last + 1),
            None,
            "a row wide enough for every close token can force the whole sequence"
        );
        assert_eq!(
            d.close_token_outside_logits(last),
            Some(HARMONY_CLOSE[HARMONY_CLOSE.len() - 1]),
            "a model whose lm_head is narrower than the tokenizer can never emit this close token"
        );

        let mut logits = vec![0.0f32; last];
        assert!(
            d.mask_to_defer_token(&mut logits),
            "the HEAD of the sequence is in range, so the per-step bool reads healthy while the \
             sequence as a whole is already impossible: only the whole-sequence check catches it"
        );

        let mut narrow = vec![0.0f32; HARMONY_CLOSE[0] as usize];
        assert!(
            !d.mask_to_defer_token(&mut narrow),
            "with the head itself out of range nothing can be masked, and a caller that drops \
             this bool leaves every token legal and the grammar deferred forever"
        );
        assert!(
            narrow.iter().all(|x| x.is_finite()),
            "the failing call must leave the logits untouched rather than half-masked"
        );
    }

    #[test]
    fn eos_inside_a_close_sequence_breaks_it_without_killing_the_decoder() {
        let mut d = harmony_decoder();
        assert!(d.advance(HARMONY_CLOSE[0]));
        assert!(d.advance(7), "EOS passes through during deferral");
        assert!(!d.is_dead());
        assert!(
            d.deferred(),
            "EOS is not part of the model's close marker, so it cannot arm the grammar"
        );
        assert_eq!(
            d.defer_token(),
            Some(HARMONY_CLOSE[0]),
            "half a close marker must not survive EOS"
        );
    }

    #[test]
    fn padded_lm_head_tail_is_masked() {
        let v = vocab(&["true", "false", ""], &[2]);
        let mut d = GuidedDecoder::from_schema(&json!({"type":"boolean"}), v).unwrap();
        let a = masked_allowed(&mut d, 8);
        assert_eq!(
            a,
            vec![true, true, false, false, false, false, false, false],
            "logits past the tokenizer vocab are padding and never legal"
        );
    }

    #[test]
    fn token_index_masks_and_advances_boolean() {
        let v = vocab(&["tr", "ue", "fa", "lse", "true", "9", "{", ""], &[7]);
        let mut d = GuidedDecoder::from_schema(&json!({"type":"boolean"}), v).unwrap();

        let a = masked_allowed(&mut d, 8);
        assert_eq!(a, vec![true, false, true, false, true, false, false, false]);
        assert!(!d.can_terminate(), "nothing emitted yet");

        assert!(d.advance(0));
        let a = masked_allowed(&mut d, 8);
        assert!(a[1], "ue continues 'tr'");
        assert!(!a[2], "fa cannot follow tr");
        assert!(!a[7], "eos masked: 'tr' incomplete");

        assert!(d.advance(1));
        assert!(d.can_terminate());
        let a = masked_allowed(&mut d, 8);
        assert!(a[7], "eos allowed once 'true' complete");
    }

    #[test]
    fn token_index_object_keys_are_forced() {
        let v = vocab(&["{", "\"", "ok", "no", "\":", "true", "}", ""], &[7]);
        let mut d = GuidedDecoder::from_schema(
            &json!({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}),
            v,
        )
        .unwrap();
        let a = masked_allowed(&mut d, 8);
        assert!(a[0], "must open with {{");
        assert!(!a[5], "cannot start with a value");
        assert!(d.advance(0));
        assert!(d.advance(1));
        let a = masked_allowed(&mut d, 8);
        assert!(a[2], "key 'ok' allowed");
        assert!(!a[3], "wrong key 'no' forbidden");
    }

    #[test]
    fn token_index_caches_state_rows() {
        let v = vocab(&["1", "2", "3", ""], &[3]);
        let mut d = GuidedDecoder::from_schema(&json!({"type":"integer"}), v).unwrap();
        let _ = masked_allowed(&mut d, 4);
        let _ = masked_allowed(&mut d, 4);
        assert_eq!(d.rows.len(), 1, "row computed once per state");
    }

    #[test]
    fn state_rows_are_shared_across_decoders() {
        let v = vocab(&["a", "ab", "b", ""], &[3]);
        let pat = "a(ab|b)*";

        let mut cold = GuidedDecoder::from_regex(pat, v.clone()).unwrap();
        let _ = masked_allowed(&mut cold, 4);
        assert!(cold.advance(0));
        let _ = masked_allowed(&mut cold, 4);
        assert!(
            cold.computed > 0,
            "cold decoder must have computed at least one row"
        );

        let mut warm = GuidedDecoder::from_regex(pat, v).unwrap();
        let _ = masked_allowed(&mut warm, 4);
        assert!(warm.advance(0));
        let _ = masked_allowed(&mut warm, 4);
        assert_eq!(
            warm.computed, 0,
            "warm decoder recomputed {} row(s): the process-wide cache is not being hit",
            warm.computed
        );
    }

    #[test]
    fn out_of_vocab_eos_masked_until_terminable() {
        let v = vocab(&["tr", "ue"], &[5]);
        let mut d = GuidedDecoder::from_schema(&json!({"type":"boolean"}), v).unwrap();

        let a = masked_allowed(&mut d, 8);
        assert!(!a[5], "oov eos masked while grammar incomplete");
        assert!(d.advance(0));
        assert!(!masked_allowed(&mut d, 8)[5], "still masked mid-token");
        assert!(d.advance(1));
        assert!(d.can_terminate());
        let a = masked_allowed(&mut d, 8);
        assert!(a[5], "oov eos allowed once grammar can terminate");
    }

    #[test]
    fn out_of_vocab_eos_advance_is_terminal_not_dead() {
        let v = vocab(&["true"], &[5]);
        let mut d = GuidedDecoder::from_schema(&json!({"type":"boolean"}), v).unwrap();
        assert!(d.advance(0));
        assert!(d.can_terminate());
        assert!(d.advance(5), "oov eos accepted after completion");
        assert!(!d.is_dead());
        assert!(!d.advance(6), "a non-eos oov id still kills");
        assert!(d.is_dead());
    }

    #[test]
    fn padding_tail_stays_masked_around_oov_eos() {
        let v = vocab(&["true"], &[5]);
        let mut d = GuidedDecoder::from_schema(&json!({"type":"boolean"}), v).unwrap();

        let a = masked_allowed(&mut d, 8);
        assert_eq!(
            &a[1..],
            &[false; 7],
            "incomplete: entire tail masked, oov eos included"
        );
        assert!(d.advance(0));
        let a = masked_allowed(&mut d, 8);
        assert_eq!(
            a,
            vec![false, false, false, false, false, true, false, false],
            "complete: only the oov eos is spared in the tail"
        );
    }

    #[test]
    fn a_dead_grammar_admits_no_token_rather_than_freeing_the_row() {
        let v = vocab(&["true", "false", "blah", ""], &[3]);
        let mut d = GuidedDecoder::from_schema(&json!({"type":"boolean"}), v).unwrap();
        assert!(
            !d.advance(2),
            "'blah' is not a prefix of any boolean literal, so it must kill the grammar"
        );
        assert!(d.is_dead());
        assert_eq!(
            masked_allowed(&mut d, 8),
            vec![false; 8],
            "a dead DFA has no continuation at all, so returning early and leaving the row \
             untouched says the opposite -- every token is legal. The request then finishes with \
             HTTP 200 and unconstrained text for a caller who asked for a schema; masking the \
             whole row leaves no legal token and every generation loop turns that into an error"
        );
        assert!(!d.can_terminate());
    }

    #[test]
    fn death_masks_the_out_of_vocab_eos_a_live_row_would_spare() {
        let v = vocab(&["true"], &[5]);
        let mut d = GuidedDecoder::from_schema(&json!({"type":"boolean"}), v).unwrap();
        assert!(d.advance(0));
        assert!(
            masked_allowed(&mut d, 8)[5],
            "a completed grammar spares the out-of-vocab eos"
        );
        assert!(!d.advance(6), "a non-eos oov id kills");
        assert_eq!(
            masked_allowed(&mut d, 8),
            vec![false; 8],
            "a dead grammar cannot terminate either, so the oov eos the live row spared must be \
             masked with everything else: sparing it would let the request stop cleanly on a \
             grammar that had already been violated"
        );
    }

    #[test]
    fn state_rows_do_not_cross_vocabs() {
        let pat = "c(cd|d)*";
        let v1 = vocab(&["c", "cd", "d", ""], &[3]);
        let mut a = GuidedDecoder::from_regex(pat, v1).unwrap();
        let _ = masked_allowed(&mut a, 4);
        assert!(a.computed > 0);

        let v2 = vocab(&["c", "cd", "d", "zz", ""], &[4]);
        let mut b = GuidedDecoder::from_regex(pat, v2).unwrap();
        let _ = masked_allowed(&mut b, 5);
        assert!(
            b.computed > 0,
            "a different vocab reused a cached row -- the key is unsound"
        );
    }
}
