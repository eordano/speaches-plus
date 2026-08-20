use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SsePush {
    Sent,
    Closed,
    TimedOut,
}

pub(crate) fn sse_send_timeout() -> std::time::Duration {
    let ms = std::env::var("NV_SSE_SEND_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(10_000)
        .max(1);
    std::time::Duration::from_millis(ms)
}

#[allow(dead_code)]
pub(crate) fn push_event_blocking(
    tx: &mpsc::Sender<ChatEvent>,
    ev: ChatEvent,
    timeout: std::time::Duration,
) -> SsePush {
    use tokio::sync::mpsc::error::TrySendError;
    let deadline = std::time::Instant::now() + timeout;
    let mut ev = ev;
    loop {
        match tx.try_send(ev) {
            Ok(()) => return SsePush::Sent,
            Err(TrySendError::Closed(_)) => return SsePush::Closed,
            Err(TrySendError::Full(back)) => {
                if std::time::Instant::now() >= deadline {
                    return SsePush::TimedOut;
                }
                ev = back;
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
    }
}

#[allow(dead_code)]
pub(crate) async fn push_event_async(
    tx: &mpsc::Sender<ChatEvent>,
    ev: ChatEvent,
    timeout: std::time::Duration,
) -> SsePush {
    match tokio::time::timeout(timeout, tx.send(ev)).await {
        Ok(Ok(())) => SsePush::Sent,
        Ok(Err(_)) => SsePush::Closed,
        Err(_) => SsePush::TimedOut,
    }
}

#[allow(dead_code)]
pub(crate) fn log_sse_abort(outcome: SsePush, path: &str) {
    match outcome {
        SsePush::TimedOut => tracing::warn!(
            path,
            "SSE client stopped reading: send deadline (NV_SSE_SEND_TIMEOUT_MS) hit; \
             aborting this generation and releasing engine resources"
        ),
        SsePush::Closed => tracing::debug!(path, "SSE client disconnected; aborting generation"),
        SsePush::Sent => {}
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn effective_max_new(requested: usize, default_max_new: usize) -> usize {
    if requested == 0 {
        default_max_new
    } else {
        requested
    }
}

pub(crate) fn stream_text_delta<'a>(emitted: &str, new_text: &'a str) -> &'a str {
    let mut split = 0usize;
    for (ec, (ni, nc)) in emitted.chars().zip(new_text.char_indices()) {
        if ec != nc {
            break;
        }
        split = ni + nc.len_utf8();
    }
    &new_text[split..]
}

pub(crate) struct StreamEmitter {
    pub(crate) sent: String,
    pub(crate) stop: Vec<String>,
    pub(crate) max_stop: usize,

    pub(crate) scanned: usize,
    pub(crate) stopped: bool,
    pub(crate) matched: Option<String>,
}

impl StreamEmitter {
    pub(crate) fn new(stop_strings: &[String]) -> Self {
        let stop: Vec<String> = stop_strings
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        let max_stop = stop.iter().map(|s| s.len()).max().unwrap_or(0);
        Self {
            sent: String::new(),
            stop,
            max_stop,
            scanned: 0,
            stopped: false,
            matched: None,
        }
    }

    pub(crate) fn step(&mut self, full: &str) -> (String, bool) {
        if self.stopped {
            return (String::new(), true);
        }

        let mut vis_end = full.len();
        while let Some(c) = full[..vis_end].chars().next_back() {
            if c == '\u{FFFD}' {
                vis_end -= c.len_utf8();
            } else {
                break;
            }
        }
        let visible = &full[..vis_end];
        if !visible.starts_with(self.sent.as_str()) {
            let piece = stream_text_delta(&self.sent, visible).to_string();
            self.sent = visible.to_string();
            self.scanned = self.sent.len();
            return (piece, false);
        }

        self.scanned = self.scanned.min(visible.len());
        while !visible.is_char_boundary(self.scanned) {
            self.scanned -= 1;
        }
        if self.max_stop > 0 {
            let mut hit: Option<usize> = None;
            for s in &self.stop {
                if let Some(rel) = visible[self.scanned..].find(s.as_str()) {
                    let abs = self.scanned + rel;
                    hit = Some(hit.map_or(abs, |h: usize| h.min(abs)));
                }
            }
            if let Some(m) = hit {
                self.matched = self
                    .stop
                    .iter()
                    .find(|s| visible[m..].starts_with(s.as_str()))
                    .cloned();
                let cut = m.max(self.sent.len());
                let piece = visible[self.sent.len()..cut].to_string();
                self.sent.push_str(&piece);
                self.stopped = true;
                return (piece, true);
            }
        }

        let mut emit_to = visible.len();
        if self.max_stop > 0 {
            let tail_start = visible
                .len()
                .saturating_sub(self.max_stop.saturating_sub(1));
            let mut idx = tail_start;
            while idx < visible.len() && !visible.is_char_boundary(idx) {
                idx += 1;
            }
            while idx < visible.len() {
                let suffix = &visible[idx..];
                if self.stop.iter().any(|s| s.starts_with(suffix)) {
                    emit_to = idx;
                    break;
                }
                idx += suffix.chars().next().map(|c| c.len_utf8()).unwrap_or(1);
            }
        }
        let emit_to = emit_to.max(self.sent.len());
        self.scanned = emit_to;
        let piece = visible[self.sent.len()..emit_to].to_string();
        self.sent.push_str(&piece);
        (piece, false)
    }

    pub(crate) fn finish(&mut self, full: &str) -> String {
        if self.stopped {
            return String::new();
        }
        let (mut piece, hit) = self.step(full);
        if !hit && full.len() > self.sent.len() && full.starts_with(self.sent.as_str()) {
            piece.push_str(&full[self.sent.len()..]);
            self.sent = full.to_string();
        }
        piece
    }
}

pub(crate) const DETOK_WINDOW_MAX: usize = 48;
pub(crate) const DETOK_WINDOW_KEEP: usize = 12;

use crate::oapi::chat::TOOL_WIRE_TOKENS;

pub(crate) fn decode_keeping_wire(
    tokenizer: &tokenizers::Tokenizer,
    ids: &[u32],
) -> Result<String, String> {
    let wire: Vec<(u32, &str)> = TOOL_WIRE_TOKENS
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t).map(|id| (id, *t)))
        .collect();
    if wire.is_empty() {
        return tokenizer.decode(ids, true).map_err(|e| e.to_string());
    }
    let mut out = String::new();
    let mut run: Vec<u32> = Vec::new();
    for id in ids {
        match wire.iter().find(|(w, _)| w == id) {
            Some((_, text)) => {
                if !run.is_empty() {
                    out.push_str(&tokenizer.decode(&run, true).map_err(|e| e.to_string())?);
                    run.clear();
                }
                out.push_str(text);
            }
            None => run.push(*id),
        }
    }
    if !run.is_empty() {
        out.push_str(&tokenizer.decode(&run, true).map_err(|e| e.to_string())?);
    }
    Ok(out)
}

pub(crate) struct IncrementalDetok {
    pub(crate) tokenizer: Arc<tokenizers::Tokenizer>,
    pub(crate) window: Vec<u32>,
    pub(crate) stable: String,
    pub(crate) acc: String,
    pub(crate) next_attempt: usize,

    pub(crate) wire: Vec<(u32, &'static str)>,
}

impl IncrementalDetok {
    pub(crate) fn new(tokenizer: Arc<tokenizers::Tokenizer>) -> Self {
        let wire = TOOL_WIRE_TOKENS
            .iter()
            .filter_map(|t| tokenizer.token_to_id(t).map(|id| (id, *t)))
            .collect();
        Self {
            tokenizer,
            window: Vec::new(),
            stable: String::new(),
            acc: String::new(),
            next_attempt: DETOK_WINDOW_MAX,
            wire,
        }
    }

    fn wire_text(&self, id: u32) -> Option<&'static str> {
        self.wire.iter().find(|(w, _)| *w == id).map(|(_, t)| *t)
    }

    pub(crate) fn push(&mut self, id: u32) -> anyhow::Result<&str> {

        if let Some(text) = self.wire_text(id) {
            let w = self
                .tokenizer
                .decode(&self.window, true)
                .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;
            self.acc.truncate(self.stable.len());
            self.stable.push_str(&w);
            self.stable.push_str(text);
            self.acc.push_str(&w);
            self.acc.push_str(text);
            self.window.clear();
            self.next_attempt = DETOK_WINDOW_MAX;
            return Ok(&self.acc);
        }
        self.window.push(id);
        let w = self
            .tokenizer
            .decode(&self.window, true)
            .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;
        self.acc.truncate(self.stable.len());
        self.acc.push_str(&w);
        if self.window.len() >= self.next_attempt {
            self.next_attempt = if self.advance_cut(&w) {
                DETOK_WINDOW_MAX
            } else {
                self.window.len() + DETOK_WINDOW_KEEP
            };
        }
        Ok(&self.acc)
    }

    pub(crate) fn advance_cut(&mut self, w: &str) -> bool {
        let j = self.window.len() - DETOK_WINDOW_KEEP;
        let Ok(head) = self.tokenizer.decode(&self.window[..j], true) else {
            return false;
        };
        if head.ends_with('\u{FFFD}') || !w.starts_with(head.as_str()) {
            return false;
        }
        let Ok(tail) = self.tokenizer.decode(&self.window[j..], true) else {
            return false;
        };
        if w.len() != head.len() + tail.len() || !w.ends_with(tail.as_str()) {
            return false;
        }
        self.stable.push_str(&head);
        self.window.drain(..j);
        true
    }
}

pub(crate) fn token_text_and_bytes(
    tokenizer: &tokenizers::Tokenizer,
    id: u32,
) -> (String, Vec<u8>) {
    let s = tokenizer.decode(&[id], false).unwrap_or_default();
    let bytes = s.as_bytes().to_vec();
    (s, bytes)
}

pub(crate) fn build_logprob_entry(
    tokenizer: &tokenizers::Tokenizer,
    out: &SampleOutput,
) -> crate::oapi::chat::LogprobEntry {
    use crate::oapi::chat::{LogprobEntry, TopLogprob};
    let (token, bytes) = token_text_and_bytes(tokenizer, out.token);
    let top_logprobs = out
        .top
        .iter()
        .map(|&(id, lp)| {
            let (t, b) = token_text_and_bytes(tokenizer, id);
            TopLogprob {
                token: t,
                logprob: lp,
                bytes: b,
            }
        })
        .collect();
    LogprobEntry {
        token,
        logprob: out.logprob.unwrap_or(f32::NEG_INFINITY),
        bytes,
        top_logprobs,
    }
}

#[cfg(test)]
mod wire_grammar_tests {
    use super::*;

    use crate::oapi::chat::NATIVE_WIRE_TOKENS as NATIVE_GRAMMAR;
    use crate::oapi::tool_parse::HERMES_WIRE_TOKENS as HERMES_GRAMMAR;

    const FRAMING: &str = "<|im_end|>";

    const DELIMITER_SPECIAL: &str = "DELIMITER_SPECIAL";

    fn grammar_tokenizer(delimiters_are_special: bool) -> Arc<tokenizers::Tokenizer> {
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 7, "content": "<tool_call>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": DELIMITER_SPECIAL},
                {"id": 8, "content": "</tool_call>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": DELIMITER_SPECIAL},
                {"id": 9, "content": "<|tool_call>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": DELIMITER_SPECIAL},
                {"id": 10, "content": "<tool_call|>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": DELIMITER_SPECIAL},
                {"id": 11, "content": "<|\"|>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": DELIMITER_SPECIAL},
                {"id": 12, "content": "<|im_end|>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": {"type": "Fuse"},
            "model": {"type": "WordLevel", "vocab": {
                "<unk>": 0,
                "{\"name\":": 1,
                "\"get_weather\"": 2,
                ",\"arguments\":{\"city\":\"Oslo, NO\"}}": 3,
                "call:get_weather{city:": 4,
                "Oslo, NO": 5,
                "}": 6
            }, "unk_token": "<unk>"}
        }"#;
        let json = json.replace(
            DELIMITER_SPECIAL,
            if delimiters_are_special {
                "true"
            } else {
                "false"
            },
        );
        Arc::new(
            json.parse::<tokenizers::Tokenizer>()
                .expect("test tokenizer"),
        )
    }

    fn hermes_call_ids(tok: &tokenizers::Tokenizer) -> Vec<u32> {
        vec![
            id(tok, HERMES_GRAMMAR[0]),
            1,
            2,
            3,
            id(tok, HERMES_GRAMMAR[1]),
            id(tok, FRAMING),
        ]
    }

    fn id(tok: &tokenizers::Tokenizer, t: &str) -> u32 {
        tok.token_to_id(t)
            .unwrap_or_else(|| panic!("fixture vocab lacks {t}, so this suite proves nothing"))
    }

    fn stream_and_finish(tok: &Arc<tokenizers::Tokenizer>, ids: &[u32]) -> String {
        let mut d = IncrementalDetok::new(tok.clone());
        let mut streamed = String::new();
        for i in ids {
            streamed = d.push(*i).expect("push").to_string();
        }
        let full = decode_keeping_wire(tok, ids).expect("decode");
        assert_eq!(
            streamed, full,
            "the streamed and full decodes disagree, so StopEmitter::finish drops the tail"
        );
        full
    }

    #[test]
    fn the_wire_list_is_exactly_the_grammars_the_parser_understands() {
        for t in NATIVE_GRAMMAR.iter().chain(HERMES_GRAMMAR.iter()) {
            assert!(
                TOOL_WIRE_TOKENS.contains(t),
                "{t:?} is a delimiter the tool parser scans for, so a decode that drops \
                 it hands the parser a body it cannot recognise"
            );
        }
        assert_eq!(
            TOOL_WIRE_TOKENS.len(),
            NATIVE_GRAMMAR.len() + HERMES_GRAMMAR.len(),
            "TOOL_WIRE_TOKENS carries an entry no parser grammar claims, so a special \
             token would be written into user-visible text: {TOOL_WIRE_TOKENS:?}"
        );
    }

    #[test]
    fn the_fixture_makes_every_delimiter_a_special_token() {
        let tok = grammar_tokenizer(true);
        for t in TOOL_WIRE_TOKENS.iter().chain([&FRAMING]) {
            let i = id(&tok, t);
            assert_eq!(
                tok.decode(&[i], true).unwrap(),
                "",
                "{t} is not special in the fixture, so nothing here reproduces the \
                 checkpoints where these delimiters are added-special tokens"
            );
        }
    }

    #[test]
    fn a_hermes_tool_call_survives_the_stream_and_parses() {
        let tok = grammar_tokenizer(true);
        let ids = hermes_call_ids(&tok);
        let full = stream_and_finish(&tok, &ids);
        let parsed = crate::oapi::chat::parse_model_tool_calls(&full, None);
        assert_eq!(
            parsed.tool_calls.len(),
            1,
            "a Qwen/Hermes tool call came back as content {:?} instead of tool_calls: the \
             handler branch at oapi::chat then leaves finish_reason \"stop\" and a caller \
             that asked for tools gets prose that happens to look like JSON. Decoded {full:?}",
            parsed.content
        );
        assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
        assert_eq!(
            parsed.tool_calls[0].function.arguments,
            r#"{"city":"Oslo, NO"}"#
        );
        assert!(
            parsed.content.is_none(),
            "leftover content {:?}",
            parsed.content
        );
        assert!(
            !full.contains(FRAMING),
            "a non-wire special leaked into user-visible text: {full:?}"
        );
    }

    #[test]
    fn a_native_tool_call_survives_the_stream_and_parses() {
        let tok = grammar_tokenizer(true);
        let s = id(&tok, NATIVE_GRAMMAR[2]);
        let ids = [
            id(&tok, NATIVE_GRAMMAR[0]),
            4,
            s,
            5,
            s,
            6,
            id(&tok, NATIVE_GRAMMAR[1]),
            id(&tok, FRAMING),
        ];
        let full = stream_and_finish(&tok, &ids);
        let parsed = crate::oapi::chat::parse_model_tool_calls(&full, None);
        assert_eq!(
            parsed.tool_calls.len(),
            1,
            "a Gemma-4 native tool call came back as content {:?} instead of tool_calls. \
             Decoded {full:?}",
            parsed.content
        );
        assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
        assert_eq!(
            parsed.tool_calls[0].function.arguments,
            r#"{"city":"Oslo, NO"}"#
        );
    }

    #[test]
    fn delimiters_the_checkpoint_never_flagged_special_decode_to_the_same_text() {
        let tok = grammar_tokenizer(false);
        let ids = hermes_call_ids(&tok);
        let plain = tok.decode(&ids, true).expect("decode");
        assert!(
            plain.contains(HERMES_GRAMMAR[0]) && plain.contains(HERMES_GRAMMAR[1]),
            "skip_special_tokens already dropped a delimiter the fixture declared \
             non-special, so this fixture does not model the Qwen packaging: {plain:?}"
        );
        let full = stream_and_finish(&tok, &ids);
        assert_eq!(
            full, plain,
            "putting the Hermes pair on the wire list changed the served text on a \
             checkpoint that never flagged those delimiters special"
        );
        assert_eq!(
            crate::oapi::chat::parse_model_tool_calls(&full, None)
                .tool_calls
                .len(),
            1,
            "decoded {full:?}"
        );
    }
}

#[cfg(test)]
mod wire_token_tests {
    use super::*;

    fn real_tokenizer() -> Arc<tokenizers::Tokenizer> {
        let dir = std::env::var("NV_CHAT_MODEL_DIR").expect(
            "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: NV_CHAT_MODEL_DIR unset",
        );
        let path = std::path::Path::new(&dir).join("tokenizer.json");
        Arc::new(tokenizers::Tokenizer::from_file(&path).expect("tokenizer.json"))
    }

    use crate::oapi::chat::NATIVE_WIRE_TOKENS as NATIVE_GRAMMAR;
    use crate::oapi::tool_parse::HERMES_WIRE_TOKENS as HERMES_GRAMMAR;

    #[test]
    #[ignore]
    fn the_tool_wire_tokens_exist_in_the_checkpoint() {
        if std::env::var("NV_TOOL_WIRE_TEST").as_deref() != Ok("1") {
            panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_TOOL_WIRE_TEST=1");
        }
        let tok = real_tokenizer();
        let d = IncrementalDetok::new(tok.clone());
        let mut whole = 0usize;
        for g in [&NATIVE_GRAMMAR[..], &HERMES_GRAMMAR[..]] {
            let n = g.iter().filter(|t| tok.token_to_id(t).is_some()).count();
            assert!(
                n == 0 || n == g.len(),
                "the checkpoint declares {n} of {g:?}: half a grammar means the \
                 emitter writes some delimiters and drops the rest, which the tool \
                 parser cannot recover from"
            );
            whole += usize::from(n == g.len());
        }
        assert!(
            whole > 0,
            "no tool-call grammar resolves against this checkpoint, so the wire \
             list is a no-op here and tool calling is broken with everything green: \
             resolved {:?} of {:?}",
            d.wire,
            TOOL_WIRE_TOKENS
        );
        for t in TOOL_WIRE_TOKENS {
            let Some(id) = tok.token_to_id(t) else {
                continue;
            };
            assert_eq!(
                tok.decode(&[id], true).unwrap(),
                "",
                "{t} is not special, so it was never being stripped and this fix \
                 is solving a problem that does not exist"
            );
        }
    }

    #[test]
    #[ignore]
    fn a_tool_call_survives_incremental_detokenisation() {
        if std::env::var("NV_TOOL_WIRE_TEST").as_deref() != Ok("1") {
            panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_TOOL_WIRE_TEST=1");
        }
        let tok = real_tokenizer();
        let call = "<|tool_call>call:get_weather{city:<|\"|>Oslo<|\"|>}<tool_call|>";
        let ids = tok.encode(call, false).expect("encode").get_ids().to_vec();
        assert!(
            ids.iter().any(|i| tok.decode(&[*i], true).unwrap().is_empty()),
            "the encoding contains no special token, so this proves nothing"
        );

        let mut d = IncrementalDetok::new(tok.clone());
        let mut last = String::new();
        for id in &ids {
            last = d.push(*id).expect("push").to_string();
        }
        assert_eq!(last, call, "emitter output must reproduce the call verbatim");

        let parsed = crate::oapi::chat::parse_model_tool_calls(&last, None);
        assert_eq!(parsed.tool_calls.len(), 1, "parsed: {last:?}");
        assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
        assert_eq!(
            parsed.tool_calls[0].function.arguments,
            r#"{"city":"Oslo"}"#
        );
    }

    #[test]
    #[ignore]
    fn plain_decoding_of_the_same_ids_loses_the_call() {
        if std::env::var("NV_TOOL_WIRE_TEST").as_deref() != Ok("1") {
            panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_TOOL_WIRE_TEST=1");
        }
        let tok = real_tokenizer();
        let call = "<|tool_call>call:get_weather{city:<|\"|>Oslo<|\"|>}<tool_call|>";
        let ids = tok.encode(call, false).expect("encode").get_ids().to_vec();
        let stripped = tok.decode(&ids, true).expect("decode");
        assert_ne!(stripped, call);
        let parsed = crate::oapi::chat::parse_model_tool_calls(&stripped, None);
        assert!(
            parsed.tool_calls.is_empty(),
            "skip_special_tokens decoding produced {stripped:?}, which parsed -- \
             then the delimiters were never load-bearing"
        );
    }

    #[test]
    #[ignore]
    fn the_full_decode_agrees_with_the_streamed_one() {
        if std::env::var("NV_TOOL_WIRE_TEST").as_deref() != Ok("1") {
            panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_TOOL_WIRE_TEST=1");
        }
        let tok = real_tokenizer();
        let text = "sure<|tool_call>call:f{x:<|\"|>a<|\"|>}<tool_call|>all done";
        let ids = tok.encode(text, false).expect("encode").get_ids().to_vec();

        let mut d = IncrementalDetok::new(tok.clone());
        let mut streamed = String::new();
        for id in &ids {
            streamed = d.push(*id).expect("push").to_string();
        }
        let full = decode_keeping_wire(&tok, &ids).expect("decode");

        assert_eq!(streamed, full, "streamed and full decode disagree");
        assert_eq!(full, text, "round trip changed the text");
        assert!(
            full.starts_with(streamed.as_str()),
            "StopEmitter::finish would drop the tail"
        );
        assert!(
            full.ends_with("all done"),
            "text after the call was lost: {full:?}"
        );

        let plain = tok.decode(&ids, true).expect("decode");
        assert!(
            !plain.starts_with(streamed.as_str()),
            "if the plain decode still prefixes the stream, this test is not \
             pinning the defect it claims to"
        );
    }

    #[test]
    #[ignore]
    fn other_special_tokens_are_still_dropped() {
        if std::env::var("NV_TOOL_WIRE_TEST").as_deref() != Ok("1") {
            panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_TOOL_WIRE_TEST=1");
        }
        let tok = real_tokenizer();
        let eos = tok
            .token_to_id("<end_of_turn>")
            .or_else(|| tok.token_to_id("<eos>"))
            .expect("an end-of-turn token");
        assert_eq!(tok.decode(&[eos], true).unwrap(), "", "not special");
        let hi = tok.encode("hi", false).expect("encode").get_ids().to_vec();

        let mut d = IncrementalDetok::new(tok.clone());
        let mut last = String::new();
        for id in hi.iter().chain(std::iter::once(&eos)) {
            last = d.push(*id).expect("push").to_string();
        }
        assert!(
            !last.contains("<end_of_turn>") && !last.contains("<eos>"),
            "framing token leaked into user-visible text: {last:?}"
        );
    }
}
