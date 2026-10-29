use anyhow::{Context, Result};
use regex_automata::{
    dfa::{dense, Automaton, StartKind},
    util::primitives::StateID,
    Anchored, Input,
};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

mod json;
mod pattern;
mod ranges;
mod schema;

type Dfa = dense::DFA<Vec<u32>>;

const ANY_JSON_VALUE_BYTE_NO_JSON_TEXT_CAN_CONTAIN: u8 = 0x01;
const ANY_JSON_OBJECT_BYTE_NO_JSON_TEXT_CAN_CONTAIN: u8 = 0x02;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct State {
    dfa: StateID,
    json: Option<json::Json>,
}

fn marker(object_only: bool) -> u8 {
    if object_only {
        ANY_JSON_OBJECT_BYTE_NO_JSON_TEXT_CAN_CONTAIN
    } else {
        ANY_JSON_VALUE_BYTE_NO_JSON_TEXT_CAN_CONTAIN
    }
}

fn live(dfa: &Dfa, s: StateID) -> Option<StateID> {
    (!dfa.is_dead_state(s)).then_some(s)
}

fn resume(
    dfa: &Dfa,
    json_markers: bool,
    at: StateID,
    j: json::Json,
    step: json::Step,
    b: u8,
) -> Option<State> {
    let after = || live(dfa, dfa.next_state(at, marker(j.object_only)));
    match step {
        json::Step::Next(json) => Some(State {
            dfa: at,
            json: Some(json),
        }),
        json::Step::Done => Some(State {
            dfa: after()?,
            json: None,
        }),
        json::Step::DoneBeforeThisByte => step_state(
            dfa,
            json_markers,
            State {
                dfa: after()?,
                json: None,
            },
            b,
        ),
        json::Step::Reject => None,
    }
}

fn step_state(dfa: &Dfa, json_markers: bool, st: State, b: u8) -> Option<State> {
    if json_markers && (b == marker(false) || b == marker(true)) {
        return None;
    }
    if let Some(j) = st.json {
        return resume(dfa, json_markers, st.dfa, j, j.step(b), b);
    }
    if let Some(ns) = live(dfa, dfa.next_state(st.dfa, b)) {
        return Some(State {
            dfa: ns,
            json: None,
        });
    }
    if !json_markers {
        return None;
    }
    [false, true]
        .into_iter()
        .filter(|&o| live(dfa, dfa.next_state(st.dfa, marker(o))).is_some())
        .find_map(|o| {
            let j = json::Json::start(o);
            resume(dfa, json_markers, st.dfa, j, j.step(b), b)
        })
}

fn terminates(dfa: &Dfa, st: State) -> bool {
    let at = match st.json {
        None => st.dfa,
        Some(j) if j.complete() => dfa.next_state(st.dfa, marker(j.object_only)),
        Some(_) => return false,
    };
    dfa.is_match_state(dfa.next_eoi_state(at))
}

fn start_state(dfa: &Dfa) -> Result<State> {
    let s = dfa
        .start_state_forward(&Input::new("").anchored(Anchored::Yes))
        .map_err(|e| anyhow::anyhow!("dfa start: {e}"))?;
    Ok(State { dfa: s, json: None })
}

fn build_anchored_dfa_uncached(pattern: &str) -> Result<Dfa> {
    let anchored = format!(r"\A(?:{pattern})\z");
    dense::Builder::new()
        .configure(
            dense::Config::new()
                .start_kind(StartKind::Anchored)
                .dfa_size_limit(Some(DFA_SIZE_LIMIT_64_MIB_REFUSES_SCHEMAS_THAT_EXPLODE_WHEN_DETERMINIZED))
                .determinize_size_limit(Some(DFA_SIZE_LIMIT_64_MIB_REFUSES_SCHEMAS_THAT_EXPLODE_WHEN_DETERMINIZED)),
        )
        .build(&anchored)
        .with_context(|| {
            format!(
                "the grammar does not compile to an automaton within {} MiB (or is not a valid \
                 regular expression); simplify it: fewer optional properties, smaller \
                 maxItems/maxLength, shorter alternations",
                DFA_SIZE_LIMIT_64_MIB_REFUSES_SCHEMAS_THAT_EXPLODE_WHEN_DETERMINIZED >> 20
            )
        })
}

pub const DFA_SIZE_LIMIT_64_MIB_REFUSES_SCHEMAS_THAT_EXPLODE_WHEN_DETERMINIZED: usize = 64 << 20;
pub const DFA_CACHE_256_PATTERNS_THEN_CLEARED_SO_DISTINCT_SCHEMAS_CANNOT_GROW_MEMORY_FOREVER: usize = 256;
pub const ROW_CACHE_4096_MASK_ROWS_THEN_CLEARED_A_ROW_IS_ONE_BIT_PER_VOCAB_TOKEN: usize = 4096;

fn build_anchored_dfa(pattern: &str) -> Result<Arc<Dfa>> {
    use std::sync::Mutex;
    static CACHE: std::sync::OnceLock<Mutex<HashMap<String, Arc<Dfa>>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(d) = cache.lock().unwrap().get(pattern) {
        return Ok(d.clone());
    }
    let dfa = Arc::new(build_anchored_dfa_uncached(pattern)?);
    let mut c = cache.lock().unwrap();
    if c.len() >= DFA_CACHE_256_PATTERNS_THEN_CLEARED_SO_DISTINCT_SCHEMAS_CANNOT_GROW_MEMORY_FOREVER {
        c.clear();
    }
    c.insert(pattern.to_string(), dfa.clone());
    Ok(dfa)
}

#[derive(Clone, Debug)]
pub enum GrammarSpec {
    JsonSchema(Value),
    Regex(String),
    CompiledRegex(String),
}

pub fn schema_to_regex(schema: &Value) -> Result<String> {
    let mut out = String::new();
    schema::Compiler::new(schema)
        .node(schema, "#", &mut out)
        .context("compile json schema to regex")?;
    Ok(out)
}

pub fn choice_to_regex(choices: &[String]) -> String {
    if choices.is_empty() {
        return String::new();
    }
    let alts: Vec<String> = choices.iter().map(|c| regex_escape(c)).collect();
    format!("({})", alts.join("|"))
}

pub fn grammar_to_regex(spec: &GrammarSpec) -> Result<String> {
    match spec {
        GrammarSpec::JsonSchema(s) => schema_to_regex(s),
        GrammarSpec::Regex(r) | GrammarSpec::CompiledRegex(r) => Ok(r.clone()),
    }
}

pub fn check_grammar(spec: &GrammarSpec) -> Result<()> {
    build_anchored_dfa(&grammar_to_regex(spec)?).map(|_| ())
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
    json_markers: bool,
    state: State,
    dead: bool,
}

impl JsonConstraint {
    pub fn from_schema(schema: &Value) -> Result<Self> {
        Self::from_compiled_regex(&schema_to_regex(schema)?)
    }

    pub fn from_regex(pattern: &str) -> Result<Self> {
        Self::new(pattern, false)
    }

    pub fn from_compiled_regex(pattern: &str) -> Result<Self> {
        Self::new(pattern, true)
    }

    fn new(pattern: &str, json_markers: bool) -> Result<Self> {
        let dfa = build_anchored_dfa(pattern)?;
        let state = start_state(&dfa)?;
        Ok(Self {
            dfa,
            json_markers,
            state,
            dead: false,
        })
    }

    pub fn can_terminate(&self) -> bool {
        !self.dead && terminates(&self.dfa, self.state)
    }

    pub fn advance_str(&mut self, s: &str) -> bool {
        for &b in s.as_bytes() {
            match step_state(&self.dfa, self.json_markers, self.state, b) {
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
            match step_state(&self.dfa, self.json_markers, st, b) {
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
    eos: Vec<u32>,
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
        let mut eos = eos_ids.to_vec();
        eos.sort_unstable();
        eos.dedup();
        Self {
            bytes: token_bytes
                .into_iter()
                .map(|v| v.into_boxed_slice())
                .collect(),
            is_eos,
            oov_eos,
            eos,
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

#[derive(Clone, Debug)]
pub struct Checkpoint {
    state: State,
    dead: bool,
    defer_until: Vec<u32>,
    defer_matched: usize,
}

struct StateRow {
    allowed: Vec<u64>,
    can_terminate: bool,
}

pub struct GuidedDecoder {
    dfa: Arc<Dfa>,
    json_markers: bool,
    vocab: Arc<VocabBytes>,
    state: State,

    rows: HashMap<State, Arc<StateRow>>,

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

type RowKey = (usize, usize, bool, State);

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
        Self::from_compiled_regex(&schema_to_regex(schema)?, vocab)
    }

    pub fn from_grammar(spec: &GrammarSpec, vocab: Arc<VocabBytes>) -> Result<Self> {
        match spec {
            GrammarSpec::JsonSchema(s) => Self::from_schema(s, vocab),
            GrammarSpec::Regex(r) => Self::from_regex(r, vocab),
            GrammarSpec::CompiledRegex(r) => Self::from_compiled_regex(r, vocab),
        }
    }

    pub fn from_regex(pattern: &str, vocab: Arc<VocabBytes>) -> Result<Self> {
        Self::new(pattern, false, vocab)
    }

    pub fn from_compiled_regex(pattern: &str, vocab: Arc<VocabBytes>) -> Result<Self> {
        Self::new(pattern, true, vocab)
    }

    fn new(pattern: &str, json_markers: bool, vocab: Arc<VocabBytes>) -> Result<Self> {
        let dfa = build_anchored_dfa(pattern)?;
        let state = start_state(&dfa)?;
        Ok(Self {
            dfa,
            json_markers,
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

    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            state: self.state,
            dead: self.dead,
            defer_until: self.defer_until.clone(),
            defer_matched: self.defer_matched,
        }
    }

    pub fn rollback(&mut self, cp: &Checkpoint) {
        self.state = cp.state;
        self.dead = cp.dead;
        self.defer_until.clone_from(&cp.defer_until);
        self.defer_matched = cp.defer_matched;
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

    fn walk(&self, mut st: State, bytes: &[u8]) -> Option<State> {
        if bytes.is_empty() {
            return None;
        }
        for &b in bytes {
            st = step_state(&self.dfa, self.json_markers, st, b)?;
        }
        Some(st)
    }

    fn compute_row(&self, state: State) -> StateRow {
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
        StateRow {
            allowed,
            can_terminate: terminates(&self.dfa, state),
        }
    }

    fn row(&mut self) -> Arc<StateRow> {
        if let Some(r) = self.rows.get(&self.state) {
            return r.clone();
        }
        let key: RowKey = (
            Arc::as_ptr(&self.dfa) as usize,
            Arc::as_ptr(&self.vocab) as usize,
            self.json_markers,
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
            if c.len() >= ROW_CACHE_4096_MASK_ROWS_THEN_CLEARED_A_ROW_IS_ONE_BIT_PER_VOCAB_TOKEN {
                c.clear();
            }
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
        !self.dead && terminates(&self.dfa, self.state)
    }

    pub fn apply_mask(&mut self, logits: &mut [f32]) {
        if self.dead {
            logits.fill(f32::NEG_INFINITY);
            return;
        }
        if self.deferred() {
            for &id in &self.vocab.eos {
                if let Some(logit) = logits.get_mut(id as usize) {
                    *logit = f32::NEG_INFINITY;
                }
            }
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
        let mut c = JsonConstraint::from_compiled_regex(pattern).unwrap();
        c.advance_str(s) && c.can_terminate()
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

    fn boolean_decoder(v: Arc<VocabBytes>) -> GuidedDecoder {
        GuidedDecoder::from_schema(&json!({"type":"boolean"}), v).unwrap()
    }

    const DEEP_REPLY: &str = r#"{"route": "change", "say": "Gold.", "ops": [{"op": "spawn", "position": {"at": "speaker", "offset": [0, 0.5, -1e2]}, "do": [{"op": "play", "tags": ["a", "b"]}]}]}"#;

    #[test]
    fn schema_regexes_accept_and_reject_the_table() {
        let weather = json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location", "unit"]
        });
        let cases: [(Value, &[&str], &[&str]); 8] = [
            (json!({"type":"integer"}), &["-42"], &["4.2"]),
            (json!({"type":"number"}), &["3.14"], &[]),
            (json!({"type":"boolean"}), &["true"], &[]),
            (json!({"type":"string"}), &["\"hi\""], &["hi"]),
            (json!({"enum":["red","green",2]}), &["\"red\"", "2"], &["\"blue\""]),
            (
                json!({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}),
                &[r#"{"name":"Ada","age":36}"#, "{ \"name\" : \"Ada\" , \"age\" : 36 }"],
                &[r#"{"name":"Ada"}"#, r#"{"age":36,"name":"Ada"}"#],
            ),
            (
                json!({"type":"array","items":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}}),
                &["[]", r#"[{"x":1}]"#, r#"[{"x":1},{"x":2}]"#],
                &[r#"[{"x":1} {"x":2}]"#],
            ),
            (weather.clone(), &[r#"{"location":"Paris","unit":"celsius"}"#], &[r#"{"location":"Paris","unit":"kelvin"}"#]),
        ];
        for (schema, ok, bad) in cases {
            let re = schema_to_regex(&schema).unwrap();
            for s in ok {
                assert!(full_match(&re, s), "{schema} must accept {s}");
            }
            for s in bad {
                assert!(!full_match(&re, s), "{schema} must reject {s}");
            }
        }
        let v = vocab(&["{", "\"", "location", "\":", "\"Paris\"", ",", "unit", "\"celsius\"", "}", ""], &[9]);
        assert!(GuidedDecoder::from_schema(&weather, v).is_ok());
    }

    #[test]
    fn json_object_mode_is_any_object_at_any_depth_with_bounded_runs_nesting_and_string_bytes() {
        let re = schema_to_regex(&json!({"type": "object"})).unwrap();
        assert!(GuidedDecoder::from_compiled_regex(&re, vocab(&["x", ""], &[1])).is_ok());
        for ok in [
            r#"{"a":1}"#,
            r#"{"a": "hi", "b": [1, 2, 3], "c": {"d": true}}"#,
            "{}",
            DEEP_REPLY,
            r#"{"a":[[[[[[[[[[null]]]]]]]]]]}"#,
            "{\n\t\"a\" : [ ] , \"b\" : { } }",
        ] {
            assert!(full_match(&re, ok), "{ok}");
        }
        for bad in [
            r#"{"a":1}x"#,
            "[1,2,3]",
            "42",
            r#"{"a":1,}"#,
            r#"{"a":[1,]}"#,
            r#"{"a" 1}"#,
            r#"{"a":01}"#,
            r#"{"a":tru}"#,
            r#"{"a":[1}"#,
            r#"{"a":{"b":1]}"#,
            r#"{"a":1}}"#,
            r#"{"a":"x"#,
            r#"{a:1}"#,
            r#"{"a":1.}"#,
            r#"{"a":-}"#,
        ] {
            assert!(!full_match(&re, bad), "{bad}");
        }
        assert!(full_match(&re, &format!("{{{}}}", " ".repeat(20))));
        assert!(!full_match(&re, &format!("{{{}}}", " ".repeat(21))));
        assert!(full_match(&re, &format!(r#"{{"a":{}}}"#, "9".repeat(20))));
        assert!(!full_match(&re, &format!(r#"{{"a":{}}}"#, "9".repeat(21))));
        let nested = |n: usize| format!("{{\"a\":{}{}}}", "[".repeat(n), "]".repeat(n));
        assert!(full_match(&re, &nested(63)));
        assert!(!full_match(&re, &nested(64)), "nesting stops at the u64 stack width");
        assert!(full_match(&re, r#"{"a":"\u00e9 \"q\" \\ é 中 🌍"}"#));
        assert!(!full_match(&re, "{\"a\":\"x\ny\"}"));
        assert!(!full_match(&re, r#"{"a":"\x"}"#));
        assert!(!full_match(&re, r#"{"a":"\u12"}"#));
        let mut d = GuidedDecoder::from_compiled_regex(
            &re,
            Arc::new(VocabBytes::new(
                vec![b"{\"a\":\"".to_vec(), vec![0xff], vec![0xc3], vec![0xa9]],
                &[],
            )),
        )
        .unwrap();
        assert!(d.advance(0));
        assert_eq!(
            masked_allowed(&mut d, 4),
            vec![false, false, true, false],
            "only a valid utf-8 lead byte may follow"
        );
        assert!(d.advance(2));
        assert_eq!(
            masked_allowed(&mut d, 4),
            vec![false, false, false, true],
            "then its continuation"
        );
    }

    #[test]
    fn free_json_inside_a_schema_meets_the_schema_on_both_sides() {
        let reply = json!({
            "type": "object",
            "properties": {
                "route": {"enum": ["change", "none"]},
                "say": {"type": "string"},
                "ops": {"type": "array", "items": {"type": "object"}}
            },
            "required": ["route", "say", "ops"]
        });
        let re = schema_to_regex(&reply).unwrap();
        assert!(full_match(&re, DEEP_REPLY));
        assert!(
            !full_match(&re, &DEEP_REPLY.replace("change", "other")),
            "the enum still holds"
        );
        assert!(
            !full_match(&re, r#"{"route": "none", "say": "", "ops": [1]}"#),
            "items stay objects"
        );

        let any_x =
            schema_to_regex(&json!({"type": "object", "properties": {"x": {}}, "required": ["x"]}))
                .unwrap();
        for ok in [
            r#"{"x":12}"#,
            r#"{"x": 12 }"#,
            r#"{"x":-0.5e+3}"#,
            r#"{"x":[1,{"y":null}]}"#,
        ] {
            assert!(full_match(&any_x, ok), "{ok}");
        }
        assert!(!full_match(&any_x, r#"{"x":12 3}"#));

        let any = schema_to_regex(&json!({})).unwrap();
        assert!(
            full_match(&any, "42"),
            "a number ending the output completes the value"
        );
        assert!(full_match(&any, "\"s\""));
        assert!(!full_match(&any, "4 2"));

        let any_items = schema_to_regex(&json!({"type": "array"})).unwrap();
        assert!(full_match(&any_items, r#"[1, "a", {"b": [true]}]"#));
    }

    #[test]
    fn the_json_machine_agrees_with_serde_json_on_every_one_byte_edit() {
        let re = schema_to_regex(&json!({"type": "object"})).unwrap();
        let deep =
            serde_json::to_string(&serde_json::from_str::<Value>(DEEP_REPLY).unwrap()).unwrap();
        let corpus = [
            "{}",
            r#"{"a":1}"#,
            r#"{"a":[1,2.5,-3e2,0],"b":{"c":null,"d":[true,false,""]}}"#,
            &deep,
        ];
        let mut checked = 0;
        for doc in corpus {
            let b = doc.as_bytes();
            for i in 0..=b.len() {
                let mut edits: Vec<Vec<u8>> = b"{}[],:\"01-.eE+a "
                    .iter()
                    .map(|c| [&b[..i], &[*c], &b[i..]].concat())
                    .collect();
                if i < b.len() {
                    edits.push([&b[..i], &b[i + 1..]].concat());
                }
                for e in edits {
                    let Ok(s) = String::from_utf8(e) else {
                        continue;
                    };
                    if s.trim() != s {
                        continue;
                    }
                    let serde = serde_json::from_str::<serde_json::Map<String, Value>>(&s).is_ok();
                    assert_eq!(full_match(&re, &s), serde, "{s}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 3000, "{checked}");
    }

    #[test]
    fn guided_tokens_that_straddle_the_schema_and_free_json_advance_and_mask() {
        let schema = json!({
            "type": "object",
            "properties": {"ops": {"type": "array", "items": {"type": "object"}}},
            "required": ["ops"]
        });
        let toks = [
            r#"{"ops":"#,
            "[",
            "{",
            r#""a""#,
            ":",
            "[0",
            ",0]",
            "}",
            "]",
            "}",
            "x",
            "",
        ];
        let n = toks.len();
        let mut d = GuidedDecoder::from_schema(&schema, vocab(&toks, &[n as u32 - 1])).unwrap();
        for t in [0u32, 1] {
            assert!(d.advance(t), "{}", toks[t as usize]);
        }
        let allowed = masked_allowed(&mut d, n);
        assert!(
            allowed[2] && allowed[8],
            "an op object or the end of the list"
        );
        assert!(!allowed[3] && !allowed[10], "no bare key, no junk");
        for t in [2u32, 3, 4, 5, 6] {
            assert!(d.advance(t), "{}", toks[t as usize]);
        }
        assert!(!d.can_terminate());
        let allowed = masked_allowed(&mut d, n);
        assert!(
            allowed[7] && !allowed[8],
            "the free object closes with }} before the list can"
        );
        for t in [7u32, 8, 9] {
            assert!(d.advance(t), "{}", toks[t as usize]);
        }
        assert!(d.can_terminate());
        let allowed = masked_allowed(&mut d, n);
        assert!(allowed[n - 1], "eos once the whole reply is complete");
        assert!(
            !allowed[2] && !allowed[9] && !allowed[10],
            "no content token after the complete object: no brace, no junk"
        );
    }

    #[test]
    fn json_constraint_prefix_terminate_mask_and_death() {
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

        let c = JsonConstraint::from_schema(&json!({"type":"boolean"})).unwrap();
        let mask = c.token_mask(&["tr", "fa", "xy", "true", "9", "{"]);
        assert_eq!(mask, vec![true, true, false, true, false, false]);

        let mut c = JsonConstraint::from_schema(&json!({"type":"integer"})).unwrap();
        assert!(!c.advance_str("x"));
        assert!(!c.can_terminate());
        assert!(!c.accepts_str("1"), "dead after an invalid byte");
    }

    #[test]
    fn deferral_arms_after_the_close_token_and_its_edges_are_pinned() {
        let v = vocab(&["true", "false", "blah", "</think>", ""], &[4]);
        let mut d = boolean_decoder(v);
        d.set_defer_until_token(3);
        assert_eq!(
            masked_allowed(&mut d, 5),
            vec![true, true, true, true, false],
            "the thought is free text, but EOS before the close marker would end the request with \
             no structured answer, so only EOS is masked while thinking"
        );
        assert!(d.advance(2), "CoT tokens pass without walking the DFA");
        assert!(d.advance(2));
        assert!(d.deferred());
        assert!(d.advance(3), "</think> arms the grammar");
        assert!(!d.deferred());
        assert_eq!(
            masked_allowed(&mut d, 5),
            vec![true, true, false, false, false],
            "post-think tokens are constrained"
        );
        assert!(d.advance(0));
        assert!(d.can_terminate());

        let v = vocab(&["true", "false", "blah", "</think>", ""], &[4, 6]);
        let mut d = boolean_decoder(v.clone());
        d.set_defer_until_token(3);
        assert_eq!(
            masked_allowed(&mut d, 8),
            vec![true, true, true, true, false, true, false, true],
            "while deferred only the EOS ids are masked, the one past the tokenizer vocab \
             included; the rest of the padding tail stays as free as the thought"
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

        let mut d2 = boolean_decoder(v);
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
        let mut d = boolean_decoder(harmony_vocab());
        d.set_defer_until_sequence(&HARMONY_CLOSE);
        d
    }

    #[test]
    fn a_close_sequence_arms_only_when_contiguous_resumes_on_a_repeated_head_and_survives_eos() {
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

        let mut empty = boolean_decoder(harmony_vocab());
        empty.set_defer_until_sequence(&[]);
        assert!(
            !empty.deferred(),
            "an empty close sequence is no deferral at all, not a grammar that never arms"
        );

        let mut repeated_head = boolean_decoder(harmony_vocab());
        repeated_head.set_defer_until_sequence(&[3, 3, 4]);
        for t in [3u32, 3, 3, 4] {
            assert!(repeated_head.advance(t));
        }
        assert!(
            !repeated_head.deferred(),
            "the last three tokens were the close sequence; resetting all progress on the \
             repeated head loses a real close"
        );

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
    fn the_forced_close_walks_the_sequence_and_reports_a_token_the_row_cannot_hold() {
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
            !d.mask_to_defer_token(&mut [0.0f32; 8]),
            "nothing is left to force once the grammar is live"
        );
        assert_eq!(
            masked_allowed(&mut d, 8),
            vec![true, true, false, false, false, false, false, false],
            "the schema constrains the tokens the reserve left room for"
        );

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
    fn the_padding_tail_stays_masked_and_only_an_out_of_vocab_eos_is_spared_once_terminable() {
        let mut d = boolean_decoder(vocab(&["true", "false", ""], &[2]));
        assert_eq!(
            masked_allowed(&mut d, 8),
            vec![true, true, false, false, false, false, false, false],
            "logits past the tokenizer vocab are padding and never legal"
        );

        let mut d = boolean_decoder(vocab(&["tr", "ue"], &[5]));
        assert!(!masked_allowed(&mut d, 8)[5], "oov eos masked while grammar incomplete");
        assert!(d.advance(0));
        assert!(!masked_allowed(&mut d, 8)[5], "still masked mid-token");
        assert!(d.advance(1));
        assert!(d.can_terminate());
        assert!(masked_allowed(&mut d, 8)[5], "oov eos allowed once grammar can terminate");

        let mut d = boolean_decoder(vocab(&["true"], &[5]));
        assert_eq!(
            &masked_allowed(&mut d, 8)[1..],
            &[false; 7],
            "incomplete: entire tail masked, oov eos included"
        );
        assert!(d.advance(0));
        assert_eq!(
            masked_allowed(&mut d, 8),
            vec![false, false, false, false, false, true, false, false],
            "complete: only the oov eos is spared in the tail"
        );
        assert!(d.advance(5), "oov eos accepted after completion");
        assert!(!d.is_dead());
        assert!(!d.advance(6), "a non-eos oov id still kills");
        assert!(d.is_dead());
    }

    #[test]
    fn token_index_masks_and_advances_booleans_and_forces_object_keys() {
        let mut d = boolean_decoder(vocab(&["tr", "ue", "fa", "lse", "true", "9", "{", ""], &[7]));
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
        assert!(masked_allowed(&mut d, 8)[7], "eos allowed once 'true' complete");

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
    fn state_rows_are_computed_once_shared_across_decoders_and_keyed_by_vocab() {
        let mut d = GuidedDecoder::from_schema(&json!({"type":"integer"}), vocab(&["1", "2", "3", ""], &[3])).unwrap();
        let _ = masked_allowed(&mut d, 4);
        let _ = masked_allowed(&mut d, 4);
        assert_eq!(d.rows.len(), 1, "row computed once per state");

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

        let pat = "c(cd|d)*";
        let mut a = GuidedDecoder::from_regex(pat, vocab(&["c", "cd", "d", ""], &[3])).unwrap();
        let _ = masked_allowed(&mut a, 4);
        assert!(a.computed > 0);
        let mut b = GuidedDecoder::from_regex(pat, vocab(&["c", "cd", "d", "zz", ""], &[4])).unwrap();
        let _ = masked_allowed(&mut b, 5);
        assert!(
            b.computed > 0,
            "a different vocab reused a cached row -- the key is unsound"
        );
    }

    #[test]
    fn a_dead_grammar_admits_no_token_not_even_the_out_of_vocab_eos() {
        let mut d = boolean_decoder(vocab(&["true", "false", "blah", ""], &[3]));
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

        let mut d = boolean_decoder(vocab(&["true"], &[5]));
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
    fn marker_bytes_open_free_json_only_in_compiled_regexes_and_are_never_output() {
        let mut digit_free =
            GuidedDecoder::from_regex("[^0-9]*", vocab(&["5", "a", "\"x\"", "{}", ""], &[4])).unwrap();
        assert_eq!(
            masked_allowed(&mut digit_free, 5),
            vec![false, true, true, true, true],
            "[^0-9]* forbids every digit; its class also admits U+0001 and U+0002, and reading \
             those as the any-JSON markers let a JSON number through where the caller's regex \
             forbids it"
        );
        assert!(
            !digit_free.advance(0),
            "a digit leaves the request regex, so advancing it must kill the decoder"
        );
        assert!(digit_free.is_dead());

        let choice = GrammarSpec::Regex(choice_to_regex(&["\u{1}".into(), "\u{2}".into()]));
        let mut control_choice =
            GuidedDecoder::from_grammar(&choice, vocab(&["5", "{}", "\u{1}", "\u{2}", ""], &[4]))
                .unwrap();
        assert_eq!(
            masked_allowed(&mut control_choice, 5),
            vec![false, false, true, true, false],
            "a guided_choice of U+0001 and U+0002 admits exactly those two strings, not any JSON \
             value or object in their place"
        );

        let parameterless_tool_args =
            schema_to_regex(&json!({"type": "object", "properties": {}})).unwrap();
        let call = format!("<call>{parameterless_tool_args}</call>");
        let shared_vocab_so_a_row_cached_with_markers_would_be_reused =
            vocab(&["<call>", "{}", "{\"a\":[1]}", "</call>", ""], &[4]);
        let v = || shared_vocab_so_a_row_cached_with_markers_would_be_reused.clone();
        let mut compiled =
            GuidedDecoder::from_grammar(&GrammarSpec::CompiledRegex(call.clone()), v()).unwrap();
        assert!(compiled.advance(0));
        let allowed = masked_allowed(&mut compiled, 5);
        assert!(
            allowed[1] && allowed[2],
            "the compiler's any-object marker inside an assembled regex still admits any object"
        );
        assert!(compiled.advance(2) && compiled.advance(3) && compiled.can_terminate());
        let mut request = GuidedDecoder::from_grammar(&GrammarSpec::Regex(call), v()).unwrap();
        assert!(request.advance(0));
        assert_eq!(
            masked_allowed(&mut request, 5),
            vec![false; 5],
            "the same text sent as a request regex is read literally: its \\x02 is the byte U+0002, \
             so the caller's regex alone decides what is legal, even where the process-wide row \
             cache already holds the compiled reading of the same DFA state"
        );

        let call = GrammarSpec::CompiledRegex(format!(
            r#"<tool_call>\{{"name": "ping", "arguments": {parameterless_tool_args}\}}</tool_call>"#
        ));
        let toks = [r#"<tool_call>{"name": "ping", "arguments": "#, "{}", "\u{2}", "}</tool_call>", ""];
        let mut required_call = GuidedDecoder::from_grammar(&call, vocab(&toks, &[4])).unwrap();
        assert!(required_call.advance(0));
        assert_eq!(
            masked_allowed(&mut required_call, 5),
            vec![false, true, false, false, false],
            "a parameterless required tool call takes a JSON object as its arguments; the raw \
             byte U+0002 is the compiler's any-object marker, not an object, and emitting it \
             would hand the client unparsable arguments"
        );
        let before = required_call.checkpoint();
        assert!(
            !required_call.advance(2),
            "advancing the raw marker byte must kill the grammar, not step over the marker"
        );
        required_call.rollback(&before);
        assert!(required_call.advance(1) && required_call.advance(3) && required_call.can_terminate());

        let free_x = json!({"type": "object", "properties": {"x": {}}, "required": ["x"]});
        let toks = ["{\"x\":", "1", "\u{1}", "\u{2}", "}", ""];
        let mut free_value = GuidedDecoder::from_schema(&free_x, vocab(&toks, &[5])).unwrap();
        assert!(free_value.advance(0));
        assert_eq!(
            masked_allowed(&mut free_value, 6),
            vec![true, true, false, false, false, false],
            "a free JSON value may be a nested object or a number, but neither marker byte stands \
             in for one"
        );
        let mut json_object_mode =
            GuidedDecoder::from_schema(&json!({"type": "object"}), vocab(&["{}", "\u{2}", "\u{1}", ""], &[3]))
                .unwrap();
        assert_eq!(
            masked_allowed(&mut json_object_mode, 4),
            vec![true, false, false, false],
            "json_object mode answers with an object, never with the marker byte it compiles to"
        );
    }
}
