#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot, Barrier, Mutex as TokioMutex, Semaphore};

use super::fuzz::Lcg;
use super::transport::EventSink;

pub const SPEECH_STARTED: &str = "input_audio_buffer.speech_started";
pub const SPEECH_STOPPED: &str = "input_audio_buffer.speech_stopped";
pub const COMMITTED: &str = "input_audio_buffer.committed";
pub const PARTIAL: &str = "input_audio_buffer.partial_transcription";
pub const ITEM_ADDED: &str = "conversation.item.added";
pub const TRANSCRIPT_DONE: &str = "conversation.item.input_audio_transcription.completed";
pub const DIARIZATION: &str = "conversation.item.diarization";
pub const RESPONSE_CREATED: &str = "response.created";
pub const RESPONSE_DONE: &str = "response.done";
pub const OUTPUT_ITEM_ADDED: &str = "response.output_item.added";
pub const AUDIO_DELTA: &str = "response.output_audio.delta";
pub const SESSION_CREATED: &str = "session.created";
pub const ERROR: &str = "error";

const I3_CHAIN: [&str; 5] = [
    SPEECH_STARTED,
    SPEECH_STOPPED,
    COMMITTED,
    ITEM_ADDED,
    TRANSCRIPT_DONE,
];

const I4_PRE_ANNOUNCE: [&str; 5] = [
    SPEECH_STARTED,
    SPEECH_STOPPED,
    COMMITTED,
    ITEM_ADDED,
    OUTPUT_ITEM_ADDED,
];

const I4_ANNOUNCE: [&str; 3] = [COMMITTED, ITEM_ADDED, OUTPUT_ITEM_ADDED];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    pub invariant: &'static str,
    pub index: usize,
    pub detail: String,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} violated at trace[{}]: {}",
            self.invariant, self.index, self.detail
        )
    }
}

fn v(invariant: &'static str, index: usize, detail: impl Into<String>) -> Violation {
    Violation {
        invariant,
        index,
        detail: detail.into(),
    }
}

pub fn ids_of(vs: &[Violation]) -> BTreeSet<&'static str> {
    vs.iter().map(|x| x.invariant).collect()
}

pub fn ty(e: &Value) -> &str {
    e.get("type").and_then(Value::as_str).unwrap_or("")
}

pub fn item_of(e: &Value) -> Option<String> {
    if let Some(s) = e.get("item_id").and_then(Value::as_str) {
        return Some(s.to_string());
    }
    if let Some(s) = e
        .get("item")
        .and_then(|i| i.get("id"))
        .and_then(Value::as_str)
    {
        return Some(s.to_string());
    }
    None
}

pub fn resp_of(e: &Value) -> Option<String> {
    if let Some(s) = e.get("response_id").and_then(Value::as_str) {
        return Some(s.to_string());
    }
    if let Some(s) = e
        .get("response")
        .and_then(|r| r.get("id"))
        .and_then(Value::as_str)
    {
        return Some(s.to_string());
    }
    None
}

pub fn render(trace: &[Value]) -> String {
    let mut out = String::new();
    for (i, e) in trace.iter().enumerate() {
        let item = item_of(e).unwrap_or_default();
        let resp = resp_of(e).unwrap_or_default();
        out.push_str(&format!("  [{i:3}] {:<52}", ty(e)));
        if !item.is_empty() {
            out.push_str(&format!(" item={item}"));
        }
        if !resp.is_empty() {
            out.push_str(&format!(" resp={resp}"));
        }
        out.push('\n');
    }
    out
}

pub fn i1_session_created_first(trace: &[Value]) -> Vec<Violation> {
    let mut out = Vec::new();
    if trace.is_empty() {
        return out;
    }
    let at = trace.iter().position(|e| ty(e) == SESSION_CREATED);
    match at {
        None => out.push(v("I1", 0, "no session.created in trace")),
        Some(0) => {}
        Some(n) => out.push(v(
            "I1",
            n,
            format!(
                "session.created is at index {n}; {} preceded it on the wire",
                ty(&trace[0])
            ),
        )),
    }
    out
}

pub fn i2_event_ids_monotone(trace: &[Value]) -> Vec<Violation> {
    let mut out = Vec::new();
    let mut prev: Option<String> = None;
    for (i, e) in trace.iter().enumerate() {
        let Some(id) = e.get("event_id").and_then(Value::as_str) else {
            out.push(v("I2", i, format!("{} carries no event_id", ty(e))));
            continue;
        };
        if let Some(p) = &prev {
            if id <= p.as_str() {
                out.push(v(
                    "I2",
                    i,
                    format!("event_id {id} does not exceed previous {p}"),
                ));
            }
        }
        prev = Some(id.to_string());
    }
    out
}

pub fn i3_input_item_chain(trace: &[Value]) -> Vec<Violation> {
    let mut first: HashMap<(String, &str), usize> = HashMap::new();
    for (i, e) in trace.iter().enumerate() {
        let t = ty(e);
        if let Some(slot) = I3_CHAIN.iter().find(|c| **c == t) {
            if let Some(item) = item_of(e) {
                first.entry((item, slot)).or_insert(i);
            }
        }
    }
    let items: BTreeSet<String> = first.keys().map(|(it, _)| it.clone()).collect();
    let mut out = Vec::new();
    for item in items {
        let mut last_seen: Option<(&str, usize)> = None;
        for stage in I3_CHAIN {
            let Some(&idx) = first.get(&(item.clone(), stage)) else {
                continue;
            };
            if let Some((prev_stage, prev_idx)) = last_seen {
                if idx < prev_idx {
                    out.push(v(
                        "I3",
                        idx,
                        format!(
                            "item {item}: {stage} at {idx} precedes {prev_stage} at {prev_idx}"
                        ),
                    ));
                }
            }
            last_seen = Some((stage, idx));
        }
    }
    out
}

pub fn i4_no_unannounced_item_refs(trace: &[Value]) -> Vec<Violation> {
    let mut announced: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for (i, e) in trace.iter().enumerate() {
        let t = ty(e);
        let Some(item) = item_of(e) else {
            continue;
        };
        let pre = I4_PRE_ANNOUNCE.contains(&t);
        if !pre && !announced.contains(&item) {
            out.push(v(
                "I4",
                i,
                format!("{t} references item {item} which was never announced"),
            ));
        }
        if I4_ANNOUNCE.contains(&t) {
            announced.insert(item);
        }
    }
    out
}

pub fn i5_no_partial_after_completed(trace: &[Value]) -> Vec<Violation> {
    let mut completed: HashMap<String, usize> = HashMap::new();
    let mut out = Vec::new();
    for (i, e) in trace.iter().enumerate() {
        let t = ty(e);
        let Some(item) = item_of(e) else {
            continue;
        };
        if t == TRANSCRIPT_DONE {
            completed.entry(item).or_insert(i);
        } else if t == PARTIAL {
            if let Some(&done_at) = completed.get(&item) {
                out.push(v(
                    "I5",
                    i,
                    format!(
                        "partial_transcription for {item} at {i} follows completed at {done_at}"
                    ),
                ));
            }
        }
    }
    out
}

pub fn i6_response_brackets(trace: &[Value]) -> Vec<Violation> {
    let mut positions: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();
    for (i, e) in trace.iter().enumerate() {
        if let Some(r) = resp_of(e) {
            positions.entry(r).or_default().push((i, ty(e).to_string()));
        }
    }
    let mut out = Vec::new();
    for (rid, evs) in positions {
        let (first_i, first_t) = evs.first().cloned().unwrap();
        let (last_i, last_t) = evs.last().cloned().unwrap();
        let has_done = evs.iter().any(|(_, t)| t == RESPONSE_DONE);
        if first_t != RESPONSE_CREATED {
            out.push(v(
                "I6",
                first_i,
                format!("response {rid}: first event is {first_t}, not response.created"),
            ));
        }
        if has_done && last_t != RESPONSE_DONE {
            out.push(v(
                "I6",
                last_i,
                format!("response {rid}: last event is {last_t}, not response.done"),
            ));
        }
    }
    out
}

pub fn i7_nothing_after_response_done(trace: &[Value]) -> Vec<Violation> {
    let mut done_at: HashMap<String, usize> = HashMap::new();
    let mut out = Vec::new();
    for (i, e) in trace.iter().enumerate() {
        let t = ty(e);
        if t == AUDIO_DELTA && resp_of(e).is_none() {
            out.push(v(
                "I7",
                i,
                "response.output_audio.delta carries no response_id, so no ordering check on it can engage",
            ));
        }
        let Some(rid) = resp_of(e) else {
            continue;
        };
        if let Some(&d) = done_at.get(&rid) {
            out.push(v(
                "I7",
                i,
                format!("{t} for response {rid} at {i} follows response.done at {d}"),
            ));
        }
        if t == RESPONSE_DONE {
            done_at.entry(rid).or_insert(i);
        }
    }
    out
}

pub fn i8_single_open_response(trace: &[Value]) -> Vec<Violation> {
    let mut open: BTreeSet<String> = BTreeSet::new();
    let mut out = Vec::new();
    for (i, e) in trace.iter().enumerate() {
        let t = ty(e);
        let Some(rid) = resp_of(e) else {
            continue;
        };
        if t == RESPONSE_CREATED {
            open.insert(rid);
            if open.len() > 1 {
                out.push(v(
                    "I8",
                    i,
                    format!("{} responses open simultaneously: {:?}", open.len(), open),
                ));
            }
        } else if t == RESPONSE_DONE {
            open.remove(&rid);
        }
    }
    out
}

pub fn i9_no_audio_after_done(trace: &[Value]) -> Vec<Violation> {
    let mut done_at: HashMap<String, usize> = HashMap::new();
    let mut open: BTreeSet<String> = BTreeSet::new();
    let mut last_done: Option<usize> = None;
    let mut out = Vec::new();
    for (i, e) in trace.iter().enumerate() {
        let t = ty(e);
        let rid = resp_of(e);
        match t {
            RESPONSE_CREATED => {
                if let Some(r) = rid.clone() {
                    open.insert(r);
                }
            }
            RESPONSE_DONE => {
                if let Some(r) = rid.clone() {
                    open.remove(&r);
                    done_at.entry(r).or_insert(i);
                }
                last_done = Some(i);
            }
            AUDIO_DELTA => match &rid {
                Some(r) => {
                    if let Some(&d) = done_at.get(r) {
                        out.push(v(
                            "I9",
                            i,
                            format!("audio frame for response {r} at {i} written after its response.done at {d}"),
                        ));
                    }
                }
                None => {
                    if open.is_empty() {
                        if let Some(d) = last_done {
                            out.push(v(
                                "I9",
                                i,
                                format!("unattributable audio frame at {i} written with no response open, after response.done at {d}"),
                            ));
                        }
                    }
                }
            },
            _ => {}
        }
    }
    out
}

pub fn i10_single_pacer(peak_concurrent_pacers: i64) -> Vec<Violation> {
    if peak_concurrent_pacers > 1 {
        vec![v(
            "I10",
            0,
            format!("{peak_concurrent_pacers} pacers wrote to one session sink concurrently"),
        )]
    } else {
        Vec::new()
    }
}

pub fn i11_timeline_monotone(trace: &[Value]) -> Vec<Violation> {
    let mut out = Vec::new();
    let mut prev_end: Option<(usize, u64)> = None;
    for (i, e) in trace.iter().enumerate() {
        if ty(e) != SPEECH_STOPPED {
            continue;
        }
        let Some(ms) = e.get("audio_end_ms").and_then(Value::as_u64) else {
            continue;
        };
        if let Some((pi, pms)) = prev_end {
            if ms < pms {
                out.push(v(
                    "I11",
                    i,
                    format!("audio_end_ms went backwards: {pms} at {pi} then {ms} at {i}"),
                ));
            }
        }
        prev_end = Some((i, ms));
    }
    let mut prev_played: HashMap<String, (usize, u64)> = HashMap::new();
    for (i, e) in trace.iter().enumerate() {
        let Some(ms) = e.get("played_ms").and_then(Value::as_u64) else {
            continue;
        };
        let rid = resp_of(e).unwrap_or_else(|| "<none>".to_string());
        if let Some((pi, pms)) = prev_played.get(&rid) {
            if ms < *pms {
                out.push(v(
                    "I11",
                    i,
                    format!(
                        "response {rid}: played_ms went backwards: {pms} at {pi} then {ms} at {i}"
                    ),
                ));
            }
        }
        prev_played.insert(rid, (i, ms));
    }
    out
}

pub fn i12_emit_order_equals_wire_order(trace: &[Value]) -> Vec<Violation> {
    let mut out = Vec::new();
    let mut prev: Option<(usize, u64)> = None;
    for (i, e) in trace.iter().enumerate() {
        let Some(s) = e.get("harness_seq").and_then(Value::as_u64) else {
            continue;
        };
        if let Some((pi, ps)) = prev {
            if s < ps {
                out.push(v(
                    "I12",
                    i,
                    format!(
                        "emit seq {s} ({}) reached the wire after seq {ps} at index {pi}",
                        ty(e)
                    ),
                ));
            }
        }
        prev = Some((i, s));
    }
    out
}

pub fn i12_fragments_contiguous(frames: &[String]) -> Vec<Violation> {
    let mut current: Option<(String, usize, usize)> = None;
    let mut out = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        let parsed: Value = match serde_json::from_str(f) {
            Ok(p) => p,
            Err(e) => {
                out.push(v("I12", i, format!("frame is not JSON: {e}")));
                continue;
            }
        };
        let kind = ty(&parsed);
        let id = parsed
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        match kind {
            "full_message" => {
                if let Some((open_id, idx, total)) = &current {
                    out.push(v(
                        "I12",
                        i,
                        format!(
                            "full_message for {id} interleaved into fragment run of {open_id} ({}/{total})",
                            idx + 1
                        ),
                    ));
                }
            }
            "partial_message" => {
                let idx = parsed
                    .get("fragment_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                let total = parsed
                    .get("total_fragments")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                match &current {
                    None => {
                        if idx != 0 {
                            out.push(v(
                                "I12",
                                i,
                                format!("fragment run for {id} starts at {idx}"),
                            ));
                        }
                        current = Some((id, idx, total));
                    }
                    Some((open_id, prev_idx, _)) => {
                        if *open_id != id {
                            out.push(v(
                                "I12",
                                i,
                                format!("fragment of {id} interleaved into open run of {open_id}"),
                            ));
                        } else if idx != prev_idx + 1 {
                            out.push(v(
                                "I12",
                                i,
                                format!("fragment {idx} of {id} follows {prev_idx}"),
                            ));
                        }
                        current = Some((id, idx, total));
                    }
                }
                if let Some((_, cur, tot)) = &current {
                    if *cur + 1 == *tot {
                        current = None;
                    }
                }
            }
            other => out.push(v("I12", i, format!("unknown envelope {other}"))),
        }
    }
    if let Some((id, idx, total)) = current {
        out.push(v(
            "I12",
            frames.len(),
            format!("fragment run for {id} ended early at {idx}/{total}"),
        ));
    }
    out
}

pub fn i13_diarization_after_transcript(trace: &[Value]) -> Vec<Violation> {
    let mut first: HashMap<(String, &'static str), usize> = HashMap::new();
    for (i, e) in trace.iter().enumerate() {
        let t = ty(e);
        let key = match t {
            ITEM_ADDED => ITEM_ADDED,
            TRANSCRIPT_DONE => TRANSCRIPT_DONE,
            DIARIZATION => DIARIZATION,
            _ => continue,
        };
        if let Some(item) = item_of(e) {
            first.entry((item, key)).or_insert(i);
        }
    }
    let mut out = Vec::new();
    let diar: Vec<(String, usize)> = first
        .iter()
        .filter(|((_, k), _)| *k == DIARIZATION)
        .map(|((it, _), i)| (it.clone(), *i))
        .collect();
    for (item, di) in diar {
        for gate in [ITEM_ADDED, TRANSCRIPT_DONE] {
            match first.get(&(item.clone(), gate)) {
                None => out.push(v(
                    "I13",
                    di,
                    format!("diarization for {item} but {gate} never appeared"),
                )),
                Some(&gi) if di < gi => out.push(v(
                    "I13",
                    di,
                    format!("diarization for {item} at {di} precedes {gate} at {gi}"),
                )),
                _ => {}
            }
        }
    }
    out
}

pub fn i15_no_silent_drops(ledger: &[u64], trace: &[Value]) -> Vec<Violation> {
    let delivered: BTreeSet<u64> = trace
        .iter()
        .filter_map(|e| e.get("harness_seq").and_then(Value::as_u64))
        .collect();
    let missing: Vec<u64> = ledger
        .iter()
        .copied()
        .filter(|s| !delivered.contains(s))
        .collect();
    if missing.is_empty() {
        return Vec::new();
    }
    let errors = trace.iter().filter(|e| ty(e) == ERROR).count();
    if errors >= missing.len() {
        return Vec::new();
    }
    vec![v(
        "I15",
        trace.len(),
        format!(
            "{} emitted events never reached the client and only {errors} error events were sent (missing seqs {:?})",
            missing.len(),
            missing
        ),
    )]
}

pub fn check_trace(trace: &[Value]) -> Vec<Violation> {
    let mut out = Vec::new();
    out.extend(i1_session_created_first(trace));
    out.extend(i2_event_ids_monotone(trace));
    out.extend(i3_input_item_chain(trace));
    out.extend(i4_no_unannounced_item_refs(trace));
    out.extend(i5_no_partial_after_completed(trace));
    out.extend(i6_response_brackets(trace));
    out.extend(i7_nothing_after_response_done(trace));
    out.extend(i8_single_open_response(trace));
    out.extend(i9_no_audio_after_done(trace));
    out.extend(i11_timeline_monotone(trace));
    out.extend(i12_emit_order_equals_wire_order(trace));
    out.extend(i13_diarization_after_transcript(trace));
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    None,
    SessionCreatedLate,
    DiarizationEarly,
    DiarizationForDroppedItem,
    AudioAfterDone,
    ConcurrentResponses,
    PartialAfterCompleted,
    DropEvent,
    EmitOrderInversion,
    PlayedMsRegression,
    TranscriptBeforeCommit,
}

pub const ALL_FAULTS: [Fault; 10] = [
    Fault::SessionCreatedLate,
    Fault::DiarizationEarly,
    Fault::DiarizationForDroppedItem,
    Fault::AudioAfterDone,
    Fault::ConcurrentResponses,
    Fault::PartialAfterCompleted,
    Fault::DropEvent,
    Fault::EmitOrderInversion,
    Fault::PlayedMsRegression,
    Fault::TranscriptBeforeCommit,
];

impl Fault {
    pub fn slug(self) -> &'static str {
        match self {
            Fault::None => "none",
            Fault::SessionCreatedLate => "session_created_late",
            Fault::DiarizationEarly => "diarization_early",
            Fault::DiarizationForDroppedItem => "diarization_for_dropped_item",
            Fault::AudioAfterDone => "audio_after_done",
            Fault::ConcurrentResponses => "concurrent_responses",
            Fault::PartialAfterCompleted => "partial_after_completed",
            Fault::DropEvent => "drop_event",
            Fault::EmitOrderInversion => "emit_order_inversion",
            Fault::PlayedMsRegression => "played_ms_regression",
            Fault::TranscriptBeforeCommit => "transcript_before_commit",
        }
    }

    pub fn from_slug(s: &str) -> Option<Fault> {
        ALL_FAULTS.iter().copied().find(|f| f.slug() == s)
    }

    pub fn expected(self) -> &'static [&'static str] {
        match self {
            Fault::None => &[],
            Fault::SessionCreatedLate => &["I1"],
            Fault::DiarizationEarly => &["I13"],
            Fault::DiarizationForDroppedItem => &["I4", "I13"],
            Fault::AudioAfterDone => &["I7", "I9"],
            Fault::ConcurrentResponses => &["I8", "I10"],
            Fault::PartialAfterCompleted => &["I5"],
            Fault::DropEvent => &["I15"],
            Fault::EmitOrderInversion => &["I12", "I2"],
            Fault::PlayedMsRegression => &["I11"],
            Fault::TranscriptBeforeCommit => &["I3"],
        }
    }
}

pub fn injected_fault() -> Fault {
    std::env::var("ORDER_HARNESS_INJECT")
        .ok()
        .and_then(|s| Fault::from_slug(&s))
        .unwrap_or(Fault::None)
}

struct Emitter {
    sink: EventSink,
    seq: AtomicU64,
    ids: AtomicU64,
    gate: TokioMutex<()>,
    fault: Fault,
    ledger: StdMutex<Vec<u64>>,
    pacers: AtomicI64,
    pacer_peak: AtomicI64,
}

impl Emitter {
    fn new(sink: EventSink, fault: Fault) -> Self {
        Self {
            sink,
            seq: AtomicU64::new(0),
            ids: AtomicU64::new(0),
            gate: TokioMutex::new(()),
            fault,
            ledger: StdMutex::new(Vec::new()),
            pacers: AtomicI64::new(0),
            pacer_peak: AtomicI64::new(0),
        }
    }

    async fn emit_inner(&self, mut value: Value, deliver: bool) {
        let inversion = self.fault == Fault::EmitOrderInversion;
        if !inversion {
            let _g = self.gate.lock().await;
            let s = self.seq.fetch_add(1, Ordering::SeqCst);
            value["harness_seq"] = json!(s);
            value["event_id"] = json!(format!(
                "evt_{:024}",
                self.ids.fetch_add(1, Ordering::SeqCst)
            ));
            self.ledger.lock().expect("ledger").push(s);
            if deliver {
                self.sink.send_value(&value).await;
            }
            return;
        }
        let (s, e) = {
            let _g = self.gate.lock().await;
            (
                self.seq.fetch_add(1, Ordering::SeqCst),
                self.ids.fetch_add(1, Ordering::SeqCst),
            )
        };
        value["harness_seq"] = json!(s);
        value["event_id"] = json!(format!("evt_{e:024}"));
        self.ledger.lock().expect("ledger").push(s);
        if s % 2 == 1 {
            tokio::time::sleep(Duration::from_millis(4)).await;
        }
        if deliver {
            self.sink.send_value(&value).await;
        }
    }

    async fn emit(&self, value: Value) {
        self.emit_inner(value, true).await;
    }

    async fn emit_dropped(&self, value: Value) {
        self.emit_inner(value, false).await;
    }

    fn pacer_open(&self) {
        let now = self.pacers.fetch_add(1, Ordering::SeqCst) + 1;
        self.pacer_peak.fetch_max(now, Ordering::SeqCst);
    }

    fn pacer_close(&self) {
        self.pacers.fetch_sub(1, Ordering::SeqCst);
    }
}

pub struct SessionRun {
    pub trace: Vec<Value>,
    pub ledger: Vec<u64>,
    pub pacer_peak: i64,
}

impl SessionRun {
    pub fn violations(&self) -> Vec<Violation> {
        let mut out = check_trace(&self.trace);
        out.extend(i10_single_pacer(self.pacer_peak));
        out.extend(i15_no_silent_drops(&self.ledger, &self.trace));
        out
    }
}

async fn audio_lane(
    em: Arc<Emitter>,
    resp_id: String,
    assistant_item: String,
    frames: u64,
    fault: Fault,
) {
    em.pacer_open();
    let mut played: u64 = 0;
    for f in 0..frames {
        tokio::time::sleep(Duration::from_millis(2)).await;
        played += 20;
        let reported = if fault == Fault::PlayedMsRegression && f == frames - 2 {
            played.saturating_sub(200)
        } else {
            played
        };
        em.emit(json!({
            "type": AUDIO_DELTA,
            "response_id": resp_id,
            "item_id": assistant_item,
            "output_index": 0,
            "content_index": 0,
            "delta": "AAAA",
            "played_ms": reported,
        }))
        .await;
    }
    em.pacer_close();
}

async fn run_response(
    em: Arc<Emitter>,
    sess: usize,
    k: usize,
    fault: Fault,
    sem: Arc<Semaphore>,
    barrier: Option<Arc<Barrier>>,
) {
    let resp_id = format!("resp_s{sess}_{k}");
    let assistant_item = format!("item_a_s{sess}_{k}");
    let _permit = if fault == Fault::ConcurrentResponses {
        None
    } else {
        Some(sem.acquire().await.expect("semaphore"))
    };
    if let Some(b) = &barrier {
        b.wait().await;
    }
    em.emit(json!({
        "type": RESPONSE_CREATED,
        "response": {"id": resp_id, "status": "in_progress"},
    }))
    .await;
    em.emit(json!({
        "type": OUTPUT_ITEM_ADDED,
        "response_id": resp_id,
        "output_index": 0,
        "item": {"id": assistant_item, "role": "assistant"},
    }))
    .await;

    let frames = if fault == Fault::AudioAfterDone {
        14
    } else {
        6
    };
    let audio = tokio::spawn(audio_lane(
        em.clone(),
        resp_id.clone(),
        assistant_item.clone(),
        frames,
        fault,
    ));

    for d in 0..4u32 {
        tokio::time::sleep(Duration::from_millis(3)).await;
        em.emit(json!({
            "type": "response.output_audio_transcript.delta",
            "response_id": resp_id,
            "item_id": assistant_item,
            "output_index": 0,
            "content_index": 0,
            "delta": format!("tok{d} "),
        }))
        .await;
    }

    let mut audio = Some(audio);
    if fault != Fault::AudioAfterDone {
        if let Some(h) = audio.take() {
            let _ = h.await;
        }
    }
    em.emit(json!({
        "type": RESPONSE_DONE,
        "response": {
            "id": resp_id,
            "object": "realtime.response",
            "status": if fault == Fault::AudioAfterDone { "cancelled" } else { "completed" },
            "audio_end_ms": 0,
            "output": [],
        },
    }))
    .await;
    if let Some(h) = audio.take() {
        let _ = h.await;
    }
}

async fn tail_lane(
    em: Arc<Emitter>,
    sess: usize,
    k: usize,
    item_id: String,
    audio_end_ms: u64,
    seed: u64,
    fault: Fault,
    sem: Arc<Semaphore>,
    barrier: Option<Arc<Barrier>>,
    diar_gate: oneshot::Sender<()>,
) {
    let mut rng = Lcg::new(seed);
    let slow = k.is_multiple_of(2);
    let stt_ms = if slow {
        22 + rng.next() % 10
    } else {
        4 + rng.next() % 4
    };

    for p in 0..2u32 {
        tokio::time::sleep(Duration::from_millis(stt_ms / 4 + 1)).await;
        em.emit(json!({
            "type": PARTIAL,
            "item_id": item_id,
            "transcript": format!("part{p}"),
            "audio_end_ms": audio_end_ms,
        }))
        .await;
    }
    tokio::time::sleep(Duration::from_millis(stt_ms)).await;
    em.emit(json!({
        "type": ITEM_ADDED,
        "item": {"id": item_id, "role": "user", "status": "completed"},
    }))
    .await;

    let completed = json!({
        "type": TRANSCRIPT_DONE,
        "item_id": item_id,
        "content_index": 0,
        "transcript": format!("utterance {k}"),
    });
    if fault == Fault::DropEvent && k == 0 {
        em.emit_dropped(completed).await;
    } else {
        em.emit(completed).await;
    }

    if fault == Fault::PartialAfterCompleted && k == 0 {
        em.emit(json!({
            "type": PARTIAL,
            "item_id": item_id,
            "transcript": "stale partial",
            "audio_end_ms": audio_end_ms,
        }))
        .await;
    }

    let _ = diar_gate.send(());
    run_response(em, sess, k, fault, sem, barrier).await;
}

pub async fn drive_session(sess: usize, items: usize, seed: u64, fault: Fault) -> SessionRun {
    let (tx, mut rx) = mpsc::channel::<String>(8192);
    let em = Arc::new(Emitter::new(EventSink::WebSocket(tx), fault));
    let mut rng = Lcg::new(seed ^ ((sess as u64) << 32));

    let created = json!({
        "type": SESSION_CREATED,
        "session": {"id": format!("sess_{sess}")},
    });
    if fault != Fault::SessionCreatedLate {
        em.emit(created.clone()).await;
    }

    let sem = Arc::new(Semaphore::new(1));
    let barrier = if fault == Fault::ConcurrentResponses {
        Some(Arc::new(Barrier::new(items)))
    } else {
        None
    };

    let mut tails = Vec::new();
    let mut diar_tasks = Vec::new();
    let mut audio_end: u64 = 0;

    for k in 0..items {
        let item_id = format!("item_s{sess}_{k}");
        audio_end += 400 + rng.next() % 600;
        em.emit(json!({
            "type": SPEECH_STARTED,
            "item_id": item_id,
            "audio_start_ms": audio_end.saturating_sub(300),
        }))
        .await;
        if k == 0 && fault == Fault::SessionCreatedLate {
            em.emit(created.clone()).await;
        }
        tokio::time::sleep(Duration::from_millis(1 + rng.next() % 3)).await;
        em.emit(json!({
            "type": SPEECH_STOPPED,
            "item_id": item_id,
            "audio_end_ms": audio_end,
        }))
        .await;

        let (gate_tx, gate_rx) = oneshot::channel::<()>();
        let em_diar = em.clone();
        let item_for_diar = item_id.clone();
        let early = matches!(
            fault,
            Fault::DiarizationEarly | Fault::DiarizationForDroppedItem
        );
        diar_tasks.push(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(1)).await;
            if !early && gate_rx.await.is_err() {
                return;
            }
            em_diar
                .emit(json!({
                    "type": DIARIZATION,
                    "item_id": item_for_diar,
                    "audio_end_ms": audio_end,
                    "segments": [],
                }))
                .await;
        }));

        if fault == Fault::DiarizationForDroppedItem && k == 0 {
            continue;
        }

        if fault == Fault::TranscriptBeforeCommit && k == 0 {
            em.emit(json!({
                "type": TRANSCRIPT_DONE,
                "item_id": item_id,
                "content_index": 0,
                "transcript": "early",
            }))
            .await;
        }

        em.emit(json!({"type": COMMITTED, "item_id": item_id}))
            .await;

        tails.push(tokio::spawn(tail_lane(
            em.clone(),
            sess,
            k,
            item_id,
            audio_end,
            seed.wrapping_mul(31).wrapping_add(k as u64),
            fault,
            sem.clone(),
            barrier.clone(),
            gate_tx,
        )));
    }

    for t in tails {
        let _ = t.await;
    }
    for t in diar_tasks {
        let _ = t.await;
    }

    let mut trace = Vec::new();
    while let Ok(text) = rx.try_recv() {
        match serde_json::from_str::<Value>(&text) {
            Ok(vv) => trace.push(vv),
            Err(e) => panic!("harness sink produced non-JSON: {e}"),
        }
    }

    let ledger = em.ledger.lock().expect("ledger").clone();
    let pacer_peak = em.pacer_peak.load(Ordering::SeqCst);
    SessionRun {
        trace,
        ledger,
        pacer_peak,
    }
}

pub async fn drive_fleet(
    sessions: usize,
    items: usize,
    seed: u64,
    fault: Fault,
) -> Vec<SessionRun> {
    let mut handles = Vec::new();
    for s in 0..sessions {
        handles.push(tokio::spawn(drive_session(s, items, seed, fault)));
    }
    let mut out = Vec::new();
    for h in handles {
        out.push(h.await.expect("session task"));
    }
    out
}

fn report(runs: &[SessionRun]) -> (Vec<Violation>, String) {
    let mut all = Vec::new();
    let mut text = String::new();
    for (i, r) in runs.iter().enumerate() {
        let vs = r.violations();
        if !vs.is_empty() {
            text.push_str(&format!(
                "\nsession {i}: {} violation(s)\n{}\ntrace:\n{}",
                vs.len(),
                vs.iter()
                    .map(|x| format!("  {x}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
                render(&r.trace)
            ));
        }
        all.extend(vs);
    }
    (all, text)
}

#[cfg(test)]
mod checker_selftest {
    use super::*;

    fn t(events: Vec<Value>) -> Vec<Value> {
        events
            .into_iter()
            .enumerate()
            .map(|(i, mut e)| {
                e["event_id"] = json!(format!("evt_{i:024}"));
                e["harness_seq"] = json!(i as u64);
                e
            })
            .collect()
    }

    fn created(id: &str) -> Value {
        json!({"type": RESPONSE_CREATED, "response": {"id": id}})
    }

    fn done(id: &str) -> Value {
        json!({"type": RESPONSE_DONE, "response": {"id": id, "status": "completed"}})
    }

    fn audio(id: &str) -> Value {
        json!({"type": AUDIO_DELTA, "response_id": id, "item_id": "item_a", "delta": "x"})
    }

    #[test]
    fn i1_flags_a_late_session_created() {
        let ok = t(vec![
            json!({"type": SESSION_CREATED}),
            json!({"type": ERROR}),
        ]);
        let bad = t(vec![
            json!({"type": ERROR}),
            json!({"type": SESSION_CREATED}),
        ]);
        assert!(i1_session_created_first(&ok).is_empty());
        assert_eq!(i1_session_created_first(&bad).len(), 1);
        assert!(!i1_session_created_first(&t(vec![json!({"type": ERROR})])).is_empty());
    }

    #[test]
    fn i2_flags_missing_and_inverted_event_ids() {
        let ok = t(vec![json!({"type": ERROR}), json!({"type": ERROR})]);
        assert!(i2_event_ids_monotone(&ok).is_empty());
        let mut inverted = ok.clone();
        inverted[1]["event_id"] = json!("evt_000000000000000000000000");
        assert_eq!(i2_event_ids_monotone(&inverted).len(), 1);
        let missing = vec![json!({"type": ERROR})];
        assert_eq!(i2_event_ids_monotone(&missing).len(), 1);
    }

    #[test]
    fn i3_flags_a_swapped_chain() {
        let ok = t(vec![
            json!({"type": SPEECH_STARTED, "item_id": "i1"}),
            json!({"type": SPEECH_STOPPED, "item_id": "i1"}),
            json!({"type": COMMITTED, "item_id": "i1"}),
            json!({"type": ITEM_ADDED, "item": {"id": "i1"}}),
            json!({"type": TRANSCRIPT_DONE, "item_id": "i1"}),
        ]);
        assert!(i3_input_item_chain(&ok).is_empty());
        let mut bad = ok.clone();
        bad.swap(2, 3);
        assert!(!i3_input_item_chain(&bad).is_empty());
    }

    #[test]
    fn i4_flags_a_dangling_item_reference() {
        let ok = t(vec![
            json!({"type": COMMITTED, "item_id": "i1"}),
            json!({"type": DIARIZATION, "item_id": "i1"}),
        ]);
        assert!(i4_no_unannounced_item_refs(&ok).is_empty());
        let bad = t(vec![
            json!({"type": SPEECH_STARTED, "item_id": "i1"}),
            json!({"type": DIARIZATION, "item_id": "i1"}),
        ]);
        assert_eq!(i4_no_unannounced_item_refs(&bad).len(), 1);
    }

    #[test]
    fn i5_flags_a_stale_partial() {
        let bad = t(vec![
            json!({"type": TRANSCRIPT_DONE, "item_id": "i1"}),
            json!({"type": PARTIAL, "item_id": "i1"}),
        ]);
        assert_eq!(i5_no_partial_after_completed(&bad).len(), 1);
        let ok = t(vec![
            json!({"type": PARTIAL, "item_id": "i1"}),
            json!({"type": TRANSCRIPT_DONE, "item_id": "i1"}),
        ]);
        assert!(i5_no_partial_after_completed(&ok).is_empty());
    }

    #[test]
    fn i6_flags_missing_created_and_trailing_events() {
        let ok = t(vec![created("r1"), audio("r1"), done("r1")]);
        assert!(i6_response_brackets(&ok).is_empty());
        let no_created = t(vec![audio("r1"), done("r1")]);
        assert_eq!(i6_response_brackets(&no_created).len(), 1);
        let trailing = t(vec![created("r1"), done("r1"), audio("r1")]);
        assert_eq!(i6_response_brackets(&trailing).len(), 1);
    }

    #[test]
    fn i7_flags_post_done_events_and_unscoped_audio() {
        let bad = t(vec![created("r1"), done("r1"), audio("r1")]);
        assert_eq!(i7_nothing_after_response_done(&bad).len(), 1);
        let unscoped = t(vec![json!({"type": AUDIO_DELTA, "delta": "x"})]);
        assert_eq!(i7_nothing_after_response_done(&unscoped).len(), 1);
        let ok = t(vec![created("r1"), audio("r1"), done("r1")]);
        assert!(i7_nothing_after_response_done(&ok).is_empty());
    }

    #[test]
    fn i8_flags_overlapping_responses() {
        let ok = t(vec![created("r1"), done("r1"), created("r2"), done("r2")]);
        assert!(i8_single_open_response(&ok).is_empty());
        let bad = t(vec![created("r1"), created("r2"), done("r1"), done("r2")]);
        assert_eq!(i8_single_open_response(&bad).len(), 1);
    }

    #[test]
    fn i9_flags_orphan_audio_including_the_unscoped_case() {
        let scoped = t(vec![created("r1"), done("r1"), audio("r1")]);
        assert_eq!(i9_no_audio_after_done(&scoped).len(), 1);
        let unscoped = t(vec![
            created("r1"),
            done("r1"),
            json!({"type": AUDIO_DELTA, "delta": "x"}),
        ]);
        assert_eq!(i9_no_audio_after_done(&unscoped).len(), 1);
        let next_response = t(vec![
            created("r1"),
            done("r1"),
            created("r2"),
            audio("r2"),
            done("r2"),
        ]);
        assert!(i9_no_audio_after_done(&next_response).is_empty());
    }

    #[test]
    fn i10_flags_two_pacers() {
        assert!(i10_single_pacer(1).is_empty());
        assert_eq!(i10_single_pacer(2).len(), 1);
    }

    #[test]
    fn i11_flags_backwards_timelines() {
        let bad_end = t(vec![
            json!({"type": SPEECH_STOPPED, "item_id": "i1", "audio_end_ms": 900}),
            json!({"type": SPEECH_STOPPED, "item_id": "i2", "audio_end_ms": 400}),
        ]);
        assert_eq!(i11_timeline_monotone(&bad_end).len(), 1);
        let mut a = audio("r1");
        a["played_ms"] = json!(200);
        let mut b = audio("r1");
        b["played_ms"] = json!(100);
        assert_eq!(i11_timeline_monotone(&t(vec![a, b])).len(), 1);
    }

    #[test]
    fn i12_flags_wire_order_inversion() {
        let mut trace = t(vec![json!({"type": ERROR}), json!({"type": ERROR})]);
        assert!(i12_emit_order_equals_wire_order(&trace).is_empty());
        trace[1]["harness_seq"] = json!(0u64);
        trace[0]["harness_seq"] = json!(1u64);
        assert_eq!(i12_emit_order_equals_wire_order(&trace).len(), 1);
    }

    #[test]
    fn i13_flags_early_diarization() {
        let ok = t(vec![
            json!({"type": ITEM_ADDED, "item": {"id": "i1"}}),
            json!({"type": TRANSCRIPT_DONE, "item_id": "i1"}),
            json!({"type": DIARIZATION, "item_id": "i1"}),
        ]);
        assert!(i13_diarization_after_transcript(&ok).is_empty());
        let bad = t(vec![
            json!({"type": DIARIZATION, "item_id": "i1"}),
            json!({"type": ITEM_ADDED, "item": {"id": "i1"}}),
            json!({"type": TRANSCRIPT_DONE, "item_id": "i1"}),
        ]);
        assert_eq!(i13_diarization_after_transcript(&bad).len(), 2);
    }

    #[test]
    fn i15_flags_silent_drops_but_accepts_an_error_event() {
        let trace = t(vec![json!({"type": ERROR})]);
        assert!(i15_no_silent_drops(&[0], &trace).is_empty());
        assert!(i15_no_silent_drops(&[0, 1], &trace).is_empty());
        assert_eq!(i15_no_silent_drops(&[0, 1, 2], &trace).len(), 1);
        let quiet = t(vec![json!({"type": COMMITTED, "item_id": "i1"})]);
        assert_eq!(i15_no_silent_drops(&[0, 1], &quiet).len(), 1);
    }

    fn client_supplied_speak_bracket() -> Vec<Value> {
        vec![
            json!({"type": SESSION_CREATED}),
            json!({"type": ITEM_ADDED, "item": {"id": "item_ABC", "role": "assistant", "status": "completed"}}),
            json!({"type": RESPONSE_CREATED, "response": {"id": "resp_1", "status": "in_progress"}}),
            json!({"type": OUTPUT_ITEM_ADDED, "response_id": "resp_1", "output_index": 0, "item": {"id": "item_XYZ"}}),
            json!({"type": "response.content_part.added", "response_id": "resp_1", "item_id": "item_XYZ"}),
            json!({"type": "response.output_audio_transcript.delta", "response_id": "resp_1", "item_id": "item_XYZ", "delta": "The build finished."}),
            json!({"type": AUDIO_DELTA, "response_id": "resp_1", "item_id": "item_XYZ", "delta": "AAAA"}),
            json!({"type": "response.output_audio_transcript.done", "response_id": "resp_1", "item_id": "item_XYZ"}),
            json!({"type": "response.output_audio.done", "response_id": "resp_1", "item_id": "item_XYZ"}),
            json!({"type": "response.content_part.done", "response_id": "resp_1", "item_id": "item_XYZ"}),
            json!({"type": "response.output_item.done", "response_id": "resp_1", "output_index": 0, "item": {"id": "item_XYZ"}}),
            json!({"type": RESPONSE_DONE, "response": {"id": "resp_1", "status": "completed"}}),
        ]
    }

    #[test]
    fn client_supplied_speak_trace_is_clean() {
        let trace = t(client_supplied_speak_bracket());
        assert!(
            ids_of(&check_trace(&trace)).is_empty(),
            "{}",
            render(&trace)
        );
    }

    #[test]
    fn client_supplied_speak_without_response_created_trips_i6() {
        let mut evs = client_supplied_speak_bracket();
        evs.remove(2);
        let trace = t(evs);
        assert!(ids_of(&check_trace(&trace)).contains("I6"));
    }
}

#[cfg(test)]
mod tier1 {
    use super::*;
    use crate::types::{EventId, ItemId, MonoF32At24k, ResponseId};

    use super::super::audio_out_ws::WsAudioPacer;
    use super::super::framing;
    use super::super::state::Topic;
    use super::super::wire::{
        ErrorPayload, EventSeq, OutboundEvent, ResponsePayload, ResponseStatus,
        ResponseStatusDetails, ResponseStatusReason,
    };

    fn tag(ev: &OutboundEvent) -> &'static str {
        use OutboundEvent::*;
        match ev {
            SessionCreated { .. } => "SessionCreated",
            SessionUpdated { .. } => "SessionUpdated",
            SessionDone { .. } => "SessionDone",
            SpeechStarted { .. } => "SpeechStarted",
            SpeechStopped { .. } => "SpeechStopped",
            BufferCommitted { .. } => "BufferCommitted",
            BufferCleared => "BufferCleared",
            PartialTranscription { .. } => "PartialTranscription",
            ItemAdded { .. } => "ItemAdded",
            ItemDeleted { .. } => "ItemDeleted",
            ItemTruncatedClientAck { .. } => "ItemTruncatedClientAck",
            AssistantTruncated { .. } => "AssistantTruncated",
            TranscriptionCompleted { .. } => "TranscriptionCompleted",
            TranscriptionDelta { .. } => "TranscriptionDelta",
            TranscriptionFailed { .. } => "TranscriptionFailed",
            ItemDone { .. } => "ItemDone",
            ItemRetrieved { .. } => "ItemRetrieved",
            ResponseCreated { .. } => "ResponseCreated",
            ResponseOutputItemAdded { .. } => "ResponseOutputItemAdded",
            ResponseOutputItemDone { .. } => "ResponseOutputItemDone",
            ResponseContentPartAdded { .. } => "ResponseContentPartAdded",
            ResponseContentPartDone { .. } => "ResponseContentPartDone",
            ResponseOutputAudioTranscriptDelta { .. } => "ResponseOutputAudioTranscriptDelta",
            ResponseOutputAudioTranscriptDone { .. } => "ResponseOutputAudioTranscriptDone",
            ResponseOutputAudioDelta { .. } => "ResponseOutputAudioDelta",
            ResponseOutputAudioDone { .. } => "ResponseOutputAudioDone",
            ResponseOutputTextDelta { .. } => "ResponseOutputTextDelta",
            ResponseOutputTextDone { .. } => "ResponseOutputTextDone",
            ResponseFunctionCallArgumentsDelta { .. } => "ResponseFunctionCallArgumentsDelta",
            ResponseFunctionCallArgumentsDone { .. } => "ResponseFunctionCallArgumentsDone",
            ResponseToolProgress { .. } => "ResponseToolProgress",
            ResponseCancelled { .. } => "ResponseCancelled",
            ResponseDone { .. } => "ResponseDone",
            OutputAudioBufferCleared => "OutputAudioBufferCleared",
            OutputAudioBufferStarted { .. } => "OutputAudioBufferStarted",
            OutputAudioBufferStopped { .. } => "OutputAudioBufferStopped",
            RateLimitsUpdated { .. } => "RateLimitsUpdated",
            Error { .. } => "Error",
            Diarization { .. } => "Diarization",
        }
    }

    fn all_variants() -> Vec<OutboundEvent> {
        use OutboundEvent::*;
        let item = || ItemId::new("item_x");
        let resp = || ResponseId::new("resp_x");
        vec![
            SessionCreated { session: json!({}) },
            SessionUpdated { session: json!({}) },
            SessionDone { reason: "x".into() },
            SpeechStarted {
                item_id: item(),
                audio_start_ms: 0,
            },
            SpeechStopped {
                item_id: item(),
                audio_end_ms: 1,
            },
            BufferCommitted { item_id: item() },
            BufferCleared,
            PartialTranscription {
                item_id: item(),
                transcript: "t".into(),
                audio_end_ms: 1,
            },
            ItemAdded {
                item: json!({"id": "item_x"}),
            },
            ItemDeleted { item_id: item() },
            ItemTruncatedClientAck {
                item_id: item(),
                content_index: 0,
                audio_end_ms: 0,
            },
            AssistantTruncated {
                event_id: EventId::new("evt_x"),
                item_id: item(),
                audio_end_ms: 0,
                transcript: "t".into(),
            },
            TranscriptionCompleted {
                item_id: item(),
                content_index: 0,
                transcript: "t".into(),
            },
            TranscriptionDelta {
                item_id: item(),
                content_index: 0,
                delta: "d".into(),
            },
            TranscriptionFailed {
                item_id: item(),
                content_index: 0,
                error: json!({}),
            },
            ItemDone {
                item: json!({"id": "item_x"}),
            },
            ItemRetrieved {
                item: json!({"id": "item_x"}),
            },
            ResponseCreated {
                response: json!({"id": "resp_x"}),
            },
            ResponseOutputItemAdded {
                response_id: resp(),
                output_index: 0,
                item: json!({"id": "item_a"}),
            },
            ResponseOutputItemDone {
                response_id: resp(),
                output_index: 0,
                item: json!({"id": "item_a"}),
            },
            ResponseContentPartAdded {
                response_id: resp(),
                item_id: item(),
                output_index: 0,
                content_index: 0,
                part: json!({}),
            },
            ResponseContentPartDone {
                response_id: resp(),
                item_id: item(),
                output_index: 0,
                content_index: 0,
                part: json!({}),
            },
            ResponseOutputAudioTranscriptDelta {
                response_id: resp(),
                item_id: item(),
                output_index: 0,
                content_index: 0,
                delta: "d".into(),
            },
            ResponseOutputAudioTranscriptDone {
                response_id: resp(),
                item_id: item(),
                output_index: 0,
                content_index: 0,
                transcript: "t".into(),
            },
            ResponseOutputAudioDelta {
                response_id: resp(),
                item_id: item(),
                output_index: 0,
                content_index: 0,
                delta: "d".into(),
            },
            ResponseOutputAudioDone {
                response_id: resp(),
                item_id: item(),
                output_index: 0,
                content_index: 0,
            },
            ResponseOutputTextDelta {
                response_id: resp(),
                item_id: item(),
                output_index: 0,
                content_index: 0,
                delta: "d".into(),
            },
            ResponseOutputTextDone {
                response_id: resp(),
                item_id: item(),
                output_index: 0,
                content_index: 0,
                text: "t".into(),
            },
            ResponseFunctionCallArgumentsDelta {
                response_id: resp(),
                item_id: item(),
                output_index: 0,
                call_id: "c".into(),
                delta: "d".into(),
            },
            ResponseFunctionCallArgumentsDone {
                response_id: resp(),
                item_id: item(),
                output_index: 0,
                call_id: "c".into(),
                arguments: "{}".into(),
            },
            ResponseToolProgress {
                response_id: resp(),
                item_id: item(),
                output_index: 0,
                progress: json!({}),
            },
            ResponseCancelled {
                response_id: resp(),
            },
            ResponseDone {
                response: ResponsePayload {
                    id: resp(),
                    status: ResponseStatus::Cancelled,
                    status_details: Some(ResponseStatusDetails {
                        reason: ResponseStatusReason::BargeIn,
                        error: None,
                    }),
                    ..Default::default()
                },
            },
            OutputAudioBufferCleared,
            OutputAudioBufferStarted {
                response_id: resp(),
            },
            OutputAudioBufferStopped {
                response_id: resp(),
            },
            RateLimitsUpdated {
                rate_limits: json!([]),
            },
            Error {
                error: ErrorPayload::for_code("x", "y"),
            },
            Diarization {
                item_id: item(),
                audio_end_ms: 0,
                elapsed_ms: None,
                segments: vec![],
            },
        ]
    }

    #[test]
    fn variant_table_is_exhaustive() {
        let evs = all_variants();
        let tags: BTreeSet<&'static str> = evs.iter().map(tag).collect();
        assert_eq!(
            tags.len(),
            evs.len(),
            "duplicate variants in the harness table"
        );
        assert_eq!(
            evs.len(),
            39,
            "OutboundEvent variant count changed; update the harness table"
        );
    }

    fn i17_divergences() -> Vec<(&'static str, Topic, Topic)> {
        all_variants()
            .iter()
            .filter_map(|ev| {
                let a = Topic::classify(ev.type_name());
                let b = ev.topic();
                if a == b {
                    None
                } else {
                    Some((ev.type_name(), a, b))
                }
            })
            .collect()
    }

    #[test]
    fn i17_topic_classifiers_agree() {
        let d = i17_divergences();
        assert!(
            d.is_empty(),
            "I17 violated for {} event types:\n{}",
            d.len(),
            d.iter()
                .map(|(t, a, b)| format!("  {t}: classify={a:?} topic()={b:?}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    #[test]
    fn i17_divergence_is_exactly_the_documented_set() {
        let got: BTreeSet<&'static str> = i17_divergences().iter().map(|(t, _, _)| *t).collect();
        let expect: BTreeSet<&'static str> = BTreeSet::new();
        assert_eq!(
            got, expect,
            "I17 divergence set moved; harness expectation is stale"
        );
    }

    #[test]
    fn i17_no_outbound_variant_classifies_as_other() {
        let stragglers: Vec<&'static str> = all_variants()
            .iter()
            .filter(|ev| ev.topic() == Topic::Other)
            .map(|ev| ev.type_name())
            .collect();
        assert!(
            stragglers.is_empty(),
            "OutboundEvent::topic() delegates to Topic::classify, whose prefix table has no arm \
             for {} event type(s):\n{}\nAdd a prefix arm in state::Topic::classify.",
            stragglers.len(),
            stragglers
                .iter()
                .map(|t| format!("  {t}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    #[test]
    fn i17_topic_table_is_pinned_per_variant() {
        let expect: &[(&str, Topic)] = &[
            ("session.created", Topic::Session),
            ("session.updated", Topic::Session),
            ("session.done", Topic::Session),
            ("rate_limits.updated", Topic::Session),
            ("input_audio_buffer.speech_started", Topic::Buffer),
            ("input_audio_buffer.speech_stopped", Topic::Buffer),
            ("input_audio_buffer.committed", Topic::Buffer),
            ("input_audio_buffer.cleared", Topic::Buffer),
            ("input_audio_buffer.partial_transcription", Topic::Buffer),
            ("output_audio_buffer.cleared", Topic::Buffer),
            ("output_audio_buffer.started", Topic::Buffer),
            ("output_audio_buffer.stopped", Topic::Buffer),
            ("conversation.item.added", Topic::Item),
            ("conversation.item.deleted", Topic::Item),
            ("conversation.item.truncated", Topic::Item),
            ("conversation.item.assistant_truncated", Topic::Item),
            (
                "conversation.item.input_audio_transcription.completed",
                Topic::Item,
            ),
            (
                "conversation.item.input_audio_transcription.delta",
                Topic::Item,
            ),
            (
                "conversation.item.input_audio_transcription.failed",
                Topic::Item,
            ),
            ("conversation.item.done", Topic::Item),
            ("conversation.item.retrieved", Topic::Item),
            ("conversation.item.diarization", Topic::Item),
            ("response.created", Topic::Response),
            ("response.output_item.added", Topic::Response),
            ("response.output_item.done", Topic::Response),
            ("response.content_part.added", Topic::Response),
            ("response.content_part.done", Topic::Response),
            ("response.output_audio_transcript.delta", Topic::Response),
            ("response.output_audio_transcript.done", Topic::Response),
            ("response.output_audio.delta", Topic::Response),
            ("response.output_audio.done", Topic::Response),
            ("response.output_text.delta", Topic::Response),
            ("response.output_text.done", Topic::Response),
            ("response.function_call_arguments.delta", Topic::Response),
            ("response.function_call_arguments.done", Topic::Response),
            ("response.tool_progress", Topic::Response),
            ("response.cancelled", Topic::Response),
            ("response.done", Topic::Response),
            ("error", Topic::Error),
        ];
        assert_eq!(
            expect.len(),
            all_variants().len(),
            "pinned topic table is stale; update it alongside all_variants()"
        );
        let pinned: BTreeSet<&str> = expect.iter().map(|(t, _)| *t).collect();
        let actual: BTreeSet<&str> = all_variants().iter().map(|ev| ev.type_name()).collect();
        assert_eq!(pinned, actual, "pinned topic table names drifted");
        for (name, want) in expect {
            assert_eq!(
                Topic::classify(name),
                *want,
                "Topic::classify({name:?}) changed"
            );
        }
        for ev in all_variants() {
            let want = expect
                .iter()
                .find(|(t, _)| *t == ev.type_name())
                .map(|(_, w)| *w)
                .expect("pinned");
            assert_eq!(ev.topic(), want, "topic() for {} changed", ev.type_name());
        }
    }

    fn missing_event_id() -> Vec<&'static str> {
        let seq = EventSeq::new();
        all_variants()
            .iter()
            .filter(|ev| {
                let v = ev.to_wire_value(&seq).expect("serialize");
                v.get("event_id").and_then(Value::as_str).is_none()
            })
            .map(|ev| ev.type_name())
            .collect()
    }

    #[test]
    fn i2_every_outbound_event_carries_event_id() {
        let missing = missing_event_id();
        assert!(
            missing.is_empty(),
            "I2 violated: {} of {} event types have no event_id field:\n  {}",
            missing.len(),
            all_variants().len(),
            missing.join("\n  ")
        );
    }

    #[test]
    fn i2_missing_event_id_surface_is_known() {
        let missing = missing_event_id();
        assert_eq!(
            missing.len(),
            0,
            "I2 surface changed (was 0 of 39 types lacking event_id): {missing:?}"
        );
    }

    #[test]
    fn i2_event_ids_are_strictly_increasing_across_every_variant() {
        let seq = EventSeq::new();
        let mut ids = Vec::new();
        for ev in all_variants() {
            let v = ev.to_wire_value(&seq).expect("serialize");
            ids.push(
                v.get("event_id")
                    .and_then(Value::as_str)
                    .unwrap_or_else(|| panic!("no event_id on {}", ev.type_name()))
                    .to_string(),
            );
        }
        assert_eq!(ids.len(), all_variants().len());
        for w in ids.windows(2) {
            assert!(
                w[0] < w[1],
                "I2 violated: event_id {} did not sort before {}",
                w[0],
                w[1]
            );
        }
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "duplicate event_id issued");
    }

    async fn ws_audio_trace(frames: usize) -> Vec<Value> {
        let (tx, mut rx) = mpsc::channel::<String>(1024);
        let seq = Arc::new(EventSeq::new());
        let played = Arc::new(AtomicU64::new(0));
        let mut pacer = WsAudioPacer::start(tx, seq, played, "pcm16_24k", "resp_a", "item_assist");
        let samples = 24_000 * 20 / 1000 * frames;
        pacer
            .play(MonoF32At24k::new(vec![0.0; samples]))
            .await
            .expect("play");
        drop(pacer);
        let mut trace = Vec::new();
        while let Ok(t) = rx.try_recv() {
            trace.push(serde_json::from_str(&t).expect("json"));
        }
        trace
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn i7_ws_audio_delta_carries_response_scope() {
        let trace = ws_audio_trace(2).await;
        assert!(!trace.is_empty(), "pacer produced no frames");
        for (i, e) in trace.iter().enumerate() {
            assert_eq!(ty(e), AUDIO_DELTA);
            for field in ["response_id", "item_id", "output_index", "content_index"] {
                assert!(
                    e.get(field).is_some(),
                    "I7 violated: frame {i} lacks {field}; keys are {:?}",
                    e.as_object().map(|o| o.keys().cloned().collect::<Vec<_>>())
                );
            }
        }
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn i7_ws_audio_delta_surface_is_response_scoped() {
        let trace = ws_audio_trace(3).await;
        assert_eq!(trace.len(), 3, "expected three 20ms frames");
        let vs = i7_nothing_after_response_done(&trace);
        assert!(
            vs.is_empty(),
            "scoped WS audio deltas must raise no I7 violation, got {vs:?}"
        );
        let keys: Vec<String> = trace[0]
            .as_object()
            .expect("object")
            .keys()
            .cloned()
            .collect();
        assert_eq!(
            keys,
            vec![
                "content_index",
                "delta",
                "event_id",
                "item_id",
                "output_index",
                "response_id",
                "type"
            ]
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn i7_ws_audio_lane_now_catches_post_done_deltas() {
        let frames = ws_audio_trace(2).await;
        let rid = resp_of(&frames[0]).expect("scoped");
        let mut trace = vec![json!({"type": RESPONSE_DONE, "response": {"id": rid}})];
        trace.extend(frames);
        let vs = i7_nothing_after_response_done(&trace);
        assert_eq!(
            vs.len(),
            2,
            "every post-done WS audio delta must be flagged, got {vs:?}"
        );
        assert!(vs.iter().all(|x| x.invariant == "I7"));
        assert!(vs
            .iter()
            .all(|x| x.detail.contains("follows response.done")));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn i16_ws_pacer_stalls_unboundedly_with_no_error_event() {
        let (tx, _rx) = mpsc::channel::<String>(1);
        let seq = Arc::new(EventSeq::new());
        let played = Arc::new(AtomicU64::new(0));
        let mut pacer = WsAudioPacer::start(tx, seq, played, "pcm16_24k", "resp_a", "item_assist");
        let samples = 24_000 * 20 / 1000 * 6;
        let r = tokio::time::timeout(
            Duration::from_millis(500),
            pacer.play(MonoF32At24k::new(vec![0.0; samples])),
        )
        .await;
        assert!(
            r.is_err(),
            "I16 expectation stale: WsAudioPacer returned {r:?} against an undrained sink; it now has a bounded path"
        );
    }

    fn oversized_event() -> Value {
        json!({"type": "conversation.item.added", "item": {"id": "item_x", "blob": "a".repeat(4000)}})
    }

    #[tokio::test(flavor = "current_thread")]
    async fn i12_fragment_interleaving_is_detected() {
        let (tx, mut rx) = mpsc::channel::<String>(256);
        let big = framing::frame_event(&oversized_event()).expect("frame");
        assert!(big.len() >= 4, "need a multi-fragment event");
        let small = framing::frame_event(&json!({"type": SESSION_CREATED})).expect("frame");

        let tx_a = tx.clone();
        let a = tokio::spawn(async move {
            for f in big {
                tx_a.send(f).await.expect("send");
                tokio::time::sleep(Duration::from_millis(4)).await;
            }
        });
        let tx_b = tx.clone();
        let b = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(2)).await;
            for f in small {
                tx_b.send(f).await.expect("send");
            }
        });
        let _ = a.await;
        let _ = b.await;
        drop(tx);

        let mut frames = Vec::new();
        while let Some(f) = rx.recv().await {
            frames.push(f);
        }
        let vs = i12_fragments_contiguous(&frames);
        assert!(
            vs.iter().any(|x| x.detail.contains("interleaved")),
            "harness failed to catch fragment interleaving: {vs:?}\nframes: {}",
            frames.len()
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn i12_serialized_fragments_pass() {
        let (tx, mut rx) = mpsc::channel::<String>(256);
        let lock = Arc::new(TokioMutex::new(()));
        let big = framing::frame_event(&oversized_event()).expect("frame");
        let small = framing::frame_event(&json!({"type": SESSION_CREATED})).expect("frame");

        let tx_a = tx.clone();
        let l_a = lock.clone();
        let a = tokio::spawn(async move {
            let _g = l_a.lock().await;
            for f in big {
                tx_a.send(f).await.expect("send");
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        });
        let tx_b = tx.clone();
        let l_b = lock.clone();
        let b = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(1)).await;
            let _g = l_b.lock().await;
            for f in small {
                tx_b.send(f).await.expect("send");
            }
        });
        let _ = a.await;
        let _ = b.await;
        drop(tx);

        let mut frames = Vec::new();
        while let Some(f) = rx.recv().await {
            frames.push(f);
        }
        let vs = i12_fragments_contiguous(&frames);
        assert!(vs.is_empty(), "serialized framing flagged: {vs:?}");
    }
}

#[cfg(test)]
mod driver {
    use super::*;

    const SEEDS: [u64; 6] = [1, 7, 1337, 424242, 9_000_001, 0xDEAD_BEEF];

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn harness_is_green_under_correct_ordering() {
        let fault = injected_fault();
        for seed in SEEDS {
            let runs = drive_fleet(4, 3, seed, fault).await;
            let total: usize = runs.iter().map(|r| r.trace.len()).sum();
            assert!(total > 100, "seed {seed}: implausibly small trace {total}");
            let (vs, text) = report(&runs);
            assert!(
                vs.is_empty(),
                "seed {seed}: {} ordering violation(s) [{:?}]{}",
                vs.len(),
                ids_of(&vs),
                text
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn injected_faults_turn_the_harness_red() {
        for fault in ALL_FAULTS {
            let runs = drive_fleet(2, 3, 4242, fault).await;
            let (vs, _) = report(&runs);
            let got = ids_of(&vs);
            for want in fault.expected() {
                assert!(
                    got.contains(want),
                    "fault {} did not trip {want}; harness reported {got:?}\n{}",
                    fault.slug(),
                    vs.iter()
                        .map(|x| format!("  {x}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                );
            }
            eprintln!(
                "fault {:<28} -> {:?} ({} violations)",
                fault.slug(),
                got,
                vs.len()
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn i14_cross_lane_audio_and_transcript_interleaving_is_not_flagged() {
        let runs = drive_fleet(1, 2, 5, Fault::None).await;
        let trace = &runs[0].trace;
        let audio: Vec<usize> = trace
            .iter()
            .enumerate()
            .filter(|(_, e)| ty(e) == AUDIO_DELTA)
            .map(|(i, _)| i)
            .collect();
        let caption: Vec<usize> = trace
            .iter()
            .enumerate()
            .filter(|(_, e)| ty(e) == "response.output_audio_transcript.delta")
            .map(|(i, _)| i)
            .collect();
        assert!(!audio.is_empty() && !caption.is_empty());
        assert!(
            audio.iter().any(|a| caption.iter().any(|c| c < a))
                && audio.iter().any(|a| caption.iter().any(|c| c > a)),
            "the two lanes did not interleave, so I14 is untested here"
        );
        assert!(
            runs[0].violations().is_empty(),
            "I14 is a declared non-guarantee but the harness flagged the interleaving"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn slow_lane_finishes_first_at_least_once() {
        let runs = drive_fleet(1, 4, 11, Fault::None).await;
        let trace = &runs[0].trace;
        let mut committed: Vec<(String, usize)> = Vec::new();
        let mut completed: Vec<(String, usize)> = Vec::new();
        for (i, e) in trace.iter().enumerate() {
            match ty(e) {
                COMMITTED => committed.push((item_of(e).unwrap_or_default(), i)),
                TRANSCRIPT_DONE => completed.push((item_of(e).unwrap_or_default(), i)),
                _ => {}
            }
        }
        let mut crossed = false;
        for a in 0..committed.len() {
            for b in 0..committed.len() {
                if a == b {
                    continue;
                }
                let (ia, ib) = (committed[a].1, committed[b].1);
                let ca = completed
                    .iter()
                    .find(|(it, _)| *it == committed[a].0)
                    .map(|(_, i)| *i);
                let cb = completed
                    .iter()
                    .find(|(it, _)| *it == committed[b].0)
                    .map(|(_, i)| *i);
                if let (Some(ca), Some(cb)) = (ca, cb) {
                    if ia < ib && ca > cb {
                        crossed = true;
                    }
                }
            }
        }
        assert!(
            crossed,
            "no item pair completed out of commit order; the driver is not exercising lane reordering\n{}",
            render(trace)
        );
    }
}

#[cfg(test)]
mod tier2 {
    use super::*;
    use crate::ids::CounterIdSource;
    use crate::models::Models;
    use crate::RealtimeQuery;

    use super::super::inspector;
    use super::super::Intent;
    use super::super::Session;

    fn models() -> Option<Arc<Models>> {
        match Models::get_or_init() {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("SKIP tier2 (models unavailable): {e}");
                None
            }
        }
    }

    fn query() -> RealtimeQuery {
        RealtimeQuery {
            intent: Some("transcription".into()),
            model: None,
            transcription_model: None,
            voice: None,
            speech_model: None,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tier2_i1_session_created_wins_the_attach_race() {
        let Some(models) = models() else {
            return;
        };
        for round in 0..64u32 {
            let session = Arc::new(Session::with_dependencies(
                query(),
                models.clone(),
                Intent::Transcription,
                None,
                Arc::new(CounterIdSource::new()),
                inspector::default_sink(),
            ));
            let (tx, mut rx) = mpsc::channel::<String>(1024);
            let racer = session.clone();
            let race = tokio::spawn(async move {
                for i in 0..32u32 {
                    racer
                        .emit_event(json!({
                            "type": "conversation.item.added",
                            "item": {"id": format!("item_race_{i}")},
                        }))
                        .await;
                    tokio::task::yield_now().await;
                }
            });
            session.attach_websocket(tx).await;
            let _ = race.await;
            let mut trace = Vec::new();
            while let Ok(t) = rx.try_recv() {
                trace.push(serde_json::from_str::<Value>(&t).expect("json"));
            }
            let vs = i1_session_created_first(&trace);
            assert!(vs.is_empty(), "round {round}: {:?}\n{}", vs, render(&trace));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tier2_i12_concurrent_emitters_preserve_order() {
        let Some(models) = models() else {
            return;
        };
        let session = Arc::new(Session::with_dependencies(
            query(),
            models,
            Intent::Transcription,
            None,
            Arc::new(CounterIdSource::new()),
            inspector::default_sink(),
        ));
        let (tx, mut rx) = mpsc::channel::<String>(8192);
        session.attach_websocket(tx).await;

        let seq = Arc::new(AtomicU64::new(0));
        let stamp = Arc::new(TokioMutex::new(()));
        let mut tasks = Vec::new();
        for lane in 0..8u32 {
            let s = session.clone();
            let seq = seq.clone();
            let stamp = stamp.clone();
            tasks.push(tokio::spawn(async move {
                for i in 0..32u32 {
                    let _g = stamp.lock().await;
                    let n = seq.fetch_add(1, Ordering::SeqCst);
                    s.emit_event(json!({
                        "type": "conversation.item.added",
                        "item": {"id": format!("item_{lane}_{i}")},
                        "harness_seq": n,
                    }))
                    .await;
                    drop(_g);
                }
            }));
        }
        for t in tasks {
            let _ = t.await;
        }
        let mut trace = Vec::new();
        while let Ok(t) = rx.try_recv() {
            trace.push(serde_json::from_str::<Value>(&t).expect("json"));
        }
        let vs = i12_emit_order_equals_wire_order(&trace);
        assert!(vs.is_empty(), "I12: {:?}", &vs[..vs.len().min(8)]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tier2_i12_event_id_monotonic_under_unserialized_emitters() {
        let Some(models) = models() else {
            eprintln!("SKIP tier2 (models unavailable)");
            return;
        };
        let session = Arc::new(Session::with_dependencies(
            query(),
            models,
            Intent::Transcription,
            None,
            Arc::new(CounterIdSource::new()),
            inspector::default_sink(),
        ));
        let (tx, mut rx) = mpsc::channel::<String>(8192);
        session.attach_websocket(tx).await;

        let seq = Arc::new(AtomicU64::new(0));
        let mut tasks = Vec::new();
        for lane in 0..8u32 {
            let s = session.clone();
            let seq = seq.clone();
            tasks.push(tokio::spawn(async move {
                for i in 0..32u32 {
                    let n = seq.fetch_add(1, Ordering::SeqCst);
                    tokio::task::yield_now().await;
                    s.emit_event(json!({
                        "type": "conversation.item.added",
                        "item": {"id": format!("item_{lane}_{i}")},
                        "harness_seq": n,
                    }))
                    .await;
                }
            }));
        }
        for t in tasks {
            let _ = t.await;
        }
        let mut trace = Vec::new();
        while let Ok(t) = rx.try_recv() {
            trace.push(serde_json::from_str::<Value>(&t).expect("json"));
        }
        let ids: Vec<String> = trace
            .iter()
            .filter_map(|e| {
                e.get("event_id")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect();
        assert_eq!(
            ids.len(),
            trace.len(),
            "every delivered event must carry an event_id"
        );
        assert_eq!(trace.len(), 8 * 32 + 1, "no events may be dropped");
        let inversions: Vec<String> = ids
            .windows(2)
            .filter(|w| w[1] <= w[0])
            .map(|w| format!("{} reached the wire before {}", w[0], w[1]))
            .collect();
        assert!(
            inversions.is_empty(),
            "server-assigned event_id out of order on the wire ({} of {}): {:?}",
            inversions.len(),
            ids.len(),
            &inversions[..inversions.len().min(8)]
        );
    }
}

#[cfg(test)]
mod cancel_energy {
    use super::*;
    use std::sync::atomic::AtomicBool;

    use base64::engine::general_purpose::STANDARD as B64;
    use base64::Engine as _;

    use crate::types::MonoF32At24k;

    use super::super::audio_out_ws::{AudioPacer, WsAudioPacer};
    use super::super::cancel::SessionCancel;
    use super::super::pipeline::TtsAbort;
    use super::super::session::InflightGuard;

    const SESSION_DONE: &str = "session.done";

    const SENTENCES: usize = 8;
    const SYNTH_MS: u64 = 40;
    const SENTENCE_AUDIO_MS: usize = 60;
    const CANCEL_AFTER_MS: u64 = 90;
    const STRAGGLER_WINDOW_MS: u64 = 900;
    const FRAME_PAYLOAD_BYTES: usize = 24_000 / 1000 * 20 * 2;

    const RESP_ID: &str = "resp_bargein";
    const ITEM_ID: &str = "item_assistant";

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum CancelMode {
        DeliveryOnly,
        StopSynthesis,
        TeardownToday,
        TeardownCancelled,
    }

    struct BargeinRun {
        trace: Vec<Value>,
        terminal: &'static str,
        synth_started: u64,
        synth_after_cancel: u64,
    }

    impl BargeinRun {
        fn frames_after_done(&self) -> usize {
            let done = self
                .trace
                .iter()
                .position(|e| ty(e) == self.terminal)
                .unwrap_or(self.trace.len());
            self.trace[done..]
                .iter()
                .filter(|e| ty(e) == AUDIO_DELTA)
                .count()
        }

        fn frames_before_done(&self) -> usize {
            let done = self
                .trace
                .iter()
                .position(|e| ty(e) == self.terminal)
                .unwrap_or(self.trace.len());
            self.trace[..done]
                .iter()
                .filter(|e| ty(e) == AUDIO_DELTA)
                .count()
        }

        fn torn_frames(&self) -> Vec<usize> {
            self.trace
                .iter()
                .enumerate()
                .filter(|(_, e)| ty(e) == AUDIO_DELTA)
                .filter(|(_, e)| {
                    let ok = e
                        .get("delta")
                        .and_then(Value::as_str)
                        .and_then(|d| B64.decode(d).ok())
                        .map(|b| b.len() == FRAME_PAYLOAD_BYTES)
                        .unwrap_or(false);
                    let scoped = resp_of(e).is_some() && item_of(e).is_some();
                    !(ok && scoped)
                })
                .map(|(i, _)| i)
                .collect()
        }
    }

    async fn drive_bargein(mode: CancelMode) -> BargeinRun {
        let (tx, mut rx) = mpsc::channel::<String>(8192);
        let sink = EventSink::WebSocket(tx.clone());
        let seq = Arc::new(super::super::wire::EventSeq::new());
        let played = Arc::new(AtomicU64::new(0));

        sink.send_value(&json!({
            "type": RESPONSE_CREATED,
            "response": {"id": RESP_ID, "status": "in_progress"},
        }))
        .await;

        let pacer = AudioPacer::WebSocket(WsAudioPacer::start(
            tx.clone(),
            seq,
            played,
            "pcm16_24k",
            RESP_ID,
            ITEM_ID,
        ));

        let (sentence_tx, mut sentence_rx) = mpsc::channel::<String>(64);
        let synth_started = Arc::new(AtomicU64::new(0));
        let synth_after_cancel = Arc::new(AtomicU64::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));

        let started_ctr = synth_started.clone();
        let after_ctr = synth_after_cancel.clone();
        let cancel_flag = cancelled.clone();
        let worker_body = async move {
            let mut pacer = pacer;
            while let Some(_sentence) = sentence_rx.recv().await {
                let started_ctr = started_ctr.clone();
                let after_ctr = after_ctr.clone();
                let cancel_flag = cancel_flag.clone();
                let audio = tokio::task::spawn_blocking(move || {
                    started_ctr.fetch_add(1, Ordering::SeqCst);
                    if cancel_flag.load(Ordering::SeqCst) {
                        after_ctr.fetch_add(1, Ordering::SeqCst);
                    }
                    std::thread::sleep(Duration::from_millis(SYNTH_MS));
                    vec![0.0f32; 24_000 * SENTENCE_AUDIO_MS / 1000]
                })
                .await
                .expect("synth join");
                if pacer.play(MonoF32At24k::new(audio)).await.is_err() {
                    break;
                }
            }
        };

        let teardown = SessionCancel::new();
        let worker_body = teardown.wrap_unit(worker_body);

        let tts_abort = Arc::new(TtsAbort::new());
        let worker = match mode {
            CancelMode::DeliveryOnly => tokio::spawn(worker_body),
            CancelMode::StopSynthesis
            | CancelMode::TeardownToday
            | CancelMode::TeardownCancelled => tts_abort.spawn(RESP_ID, worker_body),
        };

        let parent = tokio::spawn(teardown.wrap_unit(async move {
            let mut worker = worker;
            for i in 0..SENTENCES {
                if sentence_tx.send(format!("sentence {i}.")).await.is_err() {
                    break;
                }
            }
            drop(sentence_tx);
            let _ = (&mut worker).await;
        }));

        tokio::time::sleep(Duration::from_millis(CANCEL_AFTER_MS)).await;
        cancelled.store(true, Ordering::SeqCst);
        match mode {
            CancelMode::DeliveryOnly => {
                parent.abort();
            }
            CancelMode::StopSynthesis => {
                parent.abort();
                assert!(
                    tts_abort.cancel(RESP_ID).await,
                    "TtsAbort had no registered worker for {RESP_ID}"
                );
            }
            CancelMode::TeardownToday => {}
            CancelMode::TeardownCancelled => {
                teardown.cancel().await;
            }
        }

        let terminal = match mode {
            CancelMode::DeliveryOnly | CancelMode::StopSynthesis => {
                sink.send_value(&json!({
                    "type": RESPONSE_DONE,
                    "response": {
                        "id": RESP_ID,
                        "object": "realtime.response",
                        "status": "cancelled",
                        "status_details": {"reason": "turn_detected"},
                        "output": [],
                    },
                }))
                .await;
                RESPONSE_DONE
            }
            CancelMode::TeardownToday | CancelMode::TeardownCancelled => {
                sink.send_value(&json!({
                    "type": SESSION_DONE,
                    "reason": "client_closed",
                }))
                .await;
                SESSION_DONE
            }
        };

        tokio::time::sleep(Duration::from_millis(STRAGGLER_WINDOW_MS)).await;
        drop(sink);
        drop(tx);

        let mut trace = Vec::new();
        while let Ok(t) = rx.try_recv() {
            trace.push(serde_json::from_str::<Value>(&t).expect("harness sink produced non-JSON"));
        }
        BargeinRun {
            trace,
            terminal,
            synth_started: synth_started.load(Ordering::SeqCst),
            synth_after_cancel: synth_after_cancel.load(Ordering::SeqCst),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn i9_cancel_stops_synthesis_not_just_delivery() {
        let run = drive_bargein(CancelMode::StopSynthesis).await;
        eprintln!(
            "synth_started={} synth_after_cancel={} frames_before_done={} frames_after_done={}",
            run.synth_started,
            run.synth_after_cancel,
            run.frames_before_done(),
            run.frames_after_done(),
        );
        assert!(
            run.frames_before_done() >= 2,
            "vacuous run: the pacer never wrote audio before the cancel\n{}",
            render(&run.trace)
        );
        assert!(
            run.torn_frames().is_empty(),
            "torn or unscoped audio frames at {:?}\n{}",
            run.torn_frames(),
            render(&run.trace)
        );
        let vs = i9_no_audio_after_done(&run.trace);
        assert!(
            vs.is_empty(),
            "I9 violated by {} audio frame(s) after response.done{{cancelled}}:\n{}\n{}",
            vs.len(),
            vs.iter()
                .take(4)
                .map(|x| format!("  {x}"))
                .collect::<Vec<_>>()
                .join("\n"),
            render(&run.trace)
        );
        assert!(
            run.synth_after_cancel <= 1,
            "cancellation kept the TTS worker synthesizing: {} synth call(s) started after cancel \
             (total {})",
            run.synth_after_cancel,
            run.synth_started,
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn i9_delivery_only_cancel_is_the_regression_this_pins() {
        let run = drive_bargein(CancelMode::DeliveryOnly).await;
        eprintln!(
            "delivery-only: synth_started={} synth_after_cancel={} frames_after_done={}",
            run.synth_started,
            run.synth_after_cancel,
            run.frames_after_done(),
        );
        let vs = i9_no_audio_after_done(&run.trace);
        assert!(
            !vs.is_empty(),
            "aborting only the parent task no longer leaks audio past response.done; the I9 \
             regression pin is stale and drive_bargein needs rewriting\n{}",
            render(&run.trace)
        );
        assert!(
            run.synth_after_cancel >= 3,
            "delivery-only cancel synthesized only {} sentence(s) after cancel; the pin is too \
             weak to prove the energy leak",
            run.synth_after_cancel,
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn i9_cancel_avoids_measurable_synthesis_work() {
        let leaky = drive_bargein(CancelMode::DeliveryOnly).await;
        let fixed = drive_bargein(CancelMode::StopSynthesis).await;
        let avoided = leaky.synth_started.saturating_sub(fixed.synth_started);
        eprintln!(
            "synthesis avoided on barge-in: {avoided} of {} sentences \
             ({} -> {} synth calls, {} -> {} frames after done)",
            leaky.synth_started,
            leaky.synth_started,
            fixed.synth_started,
            leaky.frames_after_done(),
            fixed.frames_after_done(),
        );
        assert!(
            avoided >= 3,
            "aborting the TTS worker avoided only {avoided} synth call(s) ({} -> {})",
            leaky.synth_started,
            fixed.synth_started,
        );
        assert!(
            fixed.frames_after_done() == 0,
            "{} frame(s) still reached the sink after response.done{{cancelled}}",
            fixed.frames_after_done(),
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn teardown_without_session_cancel_is_the_leak_this_pins() {
        let run = drive_bargein(CancelMode::TeardownToday).await;
        eprintln!(
            "teardown-today: synth_started={} synth_after_teardown={} frames_after_done={}",
            run.synth_started,
            run.synth_after_cancel,
            run.frames_after_done(),
        );
        assert!(
            run.synth_after_cancel >= 3,
            "a client disconnect that stops nothing synthesized only {} sentence(s) after \
             teardown; the pin is too weak to prove the energy leak",
            run.synth_after_cancel,
        );
        assert!(
            run.frames_after_done() > 0,
            "disconnect no longer leaks audio past session.done; this regression pin is stale\n{}",
            render(&run.trace)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn teardown_cancel_avoids_measurable_synthesis_work() {
        let leaky = drive_bargein(CancelMode::TeardownToday).await;
        let fixed = drive_bargein(CancelMode::TeardownCancelled).await;
        let avoided = leaky.synth_started.saturating_sub(fixed.synth_started);
        eprintln!(
            "synthesis avoided on disconnect: {avoided} of {} sentences \
             ({} -> {} synth calls, {} -> {} frames after done)",
            leaky.synth_started,
            leaky.synth_started,
            fixed.synth_started,
            leaky.frames_after_done(),
            fixed.frames_after_done(),
        );
        assert!(
            avoided >= 3,
            "session cancellation on disconnect avoided only {avoided} synth call(s) ({} -> {})",
            leaky.synth_started,
            fixed.synth_started,
        );
        assert!(
            fixed.synth_after_cancel <= 1,
            "disconnect kept the TTS worker synthesizing: {} synth call(s) started after teardown \
             (total {})",
            fixed.synth_after_cancel,
            fixed.synth_started,
        );
        assert!(
            fixed.frames_after_done() == 0,
            "{} frame(s) still reached the sink after session.done\n{}",
            fixed.frames_after_done(),
            render(&fixed.trace)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn inflight_accounting_survives_an_aborted_emit() {
        let counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let (_tx, rx) = mpsc::channel::<()>(1);
        let held = counter.clone();
        let task = tokio::spawn(async move {
            let _guard = InflightGuard::try_acquire(&held, 4).expect("under cap");
            let mut rx = rx;
            rx.recv().await
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(counter.load(Ordering::SeqCst), 1, "guard did not register");
        task.abort();
        let _ = task.await;
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "aborting a task mid-emit leaked an outbound_inflight slot; the session would \
             permanently shed events once the leak reaches the cap",
        );
    }

    #[test]
    fn inflight_guard_rejects_over_cap_without_leaking() {
        let counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let a = InflightGuard::try_acquire(&counter, 2).expect("1 of 2");
        let b = InflightGuard::try_acquire(&counter, 2).expect("2 of 2");
        assert_eq!(InflightGuard::try_acquire(&counter, 2).err(), Some(3));
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        drop(a);
        drop(b);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tts_abort_release_leaves_a_completed_worker_alone() {
        let tts_abort = TtsAbort::new();
        let ran = Arc::new(AtomicBool::new(false));
        let flag = ran.clone();
        let worker = tts_abort.spawn("resp_normal", async move {
            flag.store(true, Ordering::SeqCst);
            7u32
        });
        assert_eq!(worker.await.expect("worker joined"), 7);
        assert!(ran.load(Ordering::SeqCst));
        tts_abort.release("resp_normal");
        assert!(
            !tts_abort.cancel("resp_normal").await,
            "release must leave nothing for a later cancel to abort"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tts_abort_ignores_a_foreign_response_id() {
        let tts_abort = TtsAbort::new();
        let worker = tts_abort.spawn("resp_a", async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
        });
        assert!(!tts_abort.cancel("resp_b").await);
        assert!(!worker.is_finished());
        assert!(tts_abort.cancel("resp_a").await);
        assert!(worker.await.unwrap_err().is_cancelled());
    }
}

#[cfg(test)]
mod client_supplied {
    use super::*;
    use std::sync::atomic::AtomicBool;

    use crate::ids::CounterIdSource;
    use crate::models::Models;
    use crate::types::{ItemId, Millis, ResponseId};
    use crate::RealtimeQuery;

    use super::super::events;
    use super::super::inspector;
    use super::super::pipeline;
    use super::super::session::Session;
    use super::super::state::{select_response_source, ResponseSource, VadPhase};
    use super::super::wire::OutboundEvent;
    use super::super::Intent;

    const SPEAK_TEXT: &str = "The build finished in forty seconds.";
    const NO_USER_MESSAGE: &str = "no user message in conversation to respond to";
    const SPEECH_ACTIVE: &str = "cannot create a response while input speech is active";

    const SPEAK_BRACKET_PORTAL_PARSES: [&str; 9] = [
        RESPONSE_CREATED,
        OUTPUT_ITEM_ADDED,
        "response.content_part.added",
        "response.output_audio_transcript.delta",
        "response.output_audio_transcript.done",
        "response.output_audio.done",
        "response.content_part.done",
        "response.output_item.done",
        RESPONSE_DONE,
    ];

    fn models() -> Option<Arc<Models>> {
        match Models::get_or_init() {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("SKIP client_supplied (models unavailable): {e}");
                None
            }
        }
    }

    fn query() -> RealtimeQuery {
        RealtimeQuery {
            intent: Some("conversation".into()),
            model: None,
            transcription_model: None,
            voice: None,
            speech_model: None,
        }
    }

    async fn conversation_session() -> Option<(Arc<Session>, mpsc::Receiver<String>)> {
        let models = models()?;
        let session = Arc::new(Session::with_dependencies(
            query(),
            models,
            Intent::Conversation,
            None,
            Arc::new(CounterIdSource::new()),
            inspector::default_sink(),
        ));
        let (tx, rx) = mpsc::channel::<String>(4096);
        session.attach_websocket(tx).await;
        Some((session, rx))
    }

    async fn inbound(session: &Arc<Session>, value: Value) {
        session
            .handle_client_event("test", bytes::Bytes::from(value.to_string()))
            .await
            .expect("inbound event accepted");
    }

    fn drain(rx: &mut mpsc::Receiver<String>) -> Vec<Value> {
        let mut out = Vec::new();
        while let Ok(t) = rx.try_recv() {
            out.push(serde_json::from_str(&t).expect("json"));
        }
        out
    }

    async fn collect_until_response_done(rx: &mut mpsc::Receiver<String>) -> Vec<Value> {
        let mut out = Vec::new();
        while let Ok(Some(t)) = tokio::time::timeout(Duration::from_secs(20), rx.recv()).await {
            let value: Value = serde_json::from_str(&t).expect("json");
            let terminal = ty(&value) == RESPONSE_DONE;
            out.push(value);
            if terminal {
                break;
            }
        }
        out
    }

    async fn wait_until_no_response_is_active(session: &Arc<Session>) {
        let waited = tokio::time::timeout(Duration::from_secs(60), async {
            loop {
                if session.state.lock().await.current_response.is_none() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(
            waited.is_ok(),
            "the response never cleared, so the next response.create would hit RESPONSE_ALREADY_ACTIVE"
        );
    }

    const SPEAK_ITEM_ID: &str = "portal_item_1";

    fn assistant_item_create(text: &str) -> Value {
        json!({
            "type": "conversation.item.create",
            "event_id": "portal_spk_1",
            "item": {
                "id": SPEAK_ITEM_ID,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            },
        })
    }

    fn speak_response_create() -> Value {
        json!({
            "type": "response.create",
            "event_id": "portal_rsp_1",
            "response": {"speak_item_id": SPEAK_ITEM_ID},
        })
    }

    fn user_item_create(text: &str) -> Value {
        json!({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "status": "completed",
                "content": [{"type": "input_text", "text": text}],
            },
        })
    }

    fn find(trace: &[Value], t: &str) -> Option<Value> {
        trace.iter().find(|e| ty(e) == t).cloned()
    }

    fn index_of(trace: &[Value], t: &str) -> Option<usize> {
        trace.iter().position(|e| ty(e) == t)
    }

    fn error_messages(trace: &[Value]) -> Vec<String> {
        trace
            .iter()
            .filter(|e| ty(e) == ERROR)
            .filter_map(|e| {
                e.get("error")
                    .and_then(|x| x.get("message"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect()
    }

    async fn speakable_flags(session: &Arc<Session>) -> Vec<bool> {
        session
            .state
            .lock()
            .await
            .conversation
            .iter()
            .map(|i| i.client_speakable)
            .collect()
    }

    async fn transcripts(session: &Arc<Session>) -> Vec<String> {
        session
            .state
            .lock()
            .await
            .conversation
            .iter()
            .map(|i| i.transcript().unwrap_or_default().to_string())
            .collect()
    }

    fn assert_clean(trace: &[Value]) {
        let vs = check_trace(trace);
        assert!(
            vs.is_empty(),
            "{:?}\n{}\n{}",
            ids_of(&vs),
            vs.iter()
                .map(|x| format!("  {x}"))
                .collect::<Vec<_>>()
                .join("\n"),
            render(trace)
        );
    }

    struct OpenResponse {
        id: ResponseId,
        played_ms: Arc<AtomicU64>,
        transcript_so_far: Arc<TokioMutex<String>>,
        wire_opened: Arc<AtomicBool>,
    }

    async fn open_response(
        session: &Arc<Session>,
        response_id: &str,
        item_id: &str,
        emit_brackets: bool,
    ) -> OpenResponse {
        let played_ms = Arc::new(AtomicU64::new(0));
        let transcript_so_far = Arc::new(TokioMutex::new(String::new()));
        let wire_opened = Arc::new(AtomicBool::new(false));
        let handle = tokio::spawn(async { std::future::pending::<()>().await });
        session
            .register_response(
                ResponseId::new(response_id),
                handle,
                played_ms.clone(),
                ItemId::new(item_id),
                transcript_so_far.clone(),
                wire_opened.clone(),
            )
            .await;
        if emit_brackets {
            events::emit_response_open_brackets(session, response_id, item_id).await;
            wire_opened.store(true, Ordering::Release);
            session.mark_streaming(response_id).await;
            *transcript_so_far.lock().await = SPEAK_TEXT.to_string();
            events::emit_audio_transcript_delta(session, response_id, item_id, SPEAK_TEXT).await;
            for _ in 0..2 {
                session
                    .emit(OutboundEvent::ResponseOutputAudioDelta {
                        response_id: ResponseId::new(response_id),
                        item_id: ItemId::new(item_id),
                        output_index: 0,
                        content_index: 0,
                        delta: "AAAA".to_string(),
                    })
                    .await;
            }
        }
        OpenResponse {
            id: ResponseId::new(response_id),
            played_ms,
            transcript_so_far,
            wire_opened,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn item_create_marks_only_nonblank_assistant_text_client_speakable() {
        let Some((session, mut rx)) = conversation_session().await else {
            return;
        };
        inbound(&session, assistant_item_create(SPEAK_TEXT)).await;
        inbound(&session, user_item_create("when did the build finish")).await;
        inbound(&session, assistant_item_create("   \n ")).await;
        assert_eq!(
            speakable_flags(&session).await,
            vec![true, false, false],
            "only a non-blank assistant item written by the client may be spoken"
        );
        assert_clean(&drain(&mut rx));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn client_supplied_speak_bracket_is_order_clean_and_matches_the_portal_contract() {
        let Some((session, mut rx)) = conversation_session().await else {
            return;
        };
        inbound(&session, assistant_item_create(SPEAK_TEXT)).await;
        let open = open_response(&session, "resp_speak_1", "item_spoken_1", true).await;
        events::emit_bracket_close(&session, "resp_speak_1", "item_spoken_1", SPEAK_TEXT).await;
        events::emit_response_done(
            &session,
            "resp_speak_1",
            "completed",
            Some(SPEAK_TEXT.to_string()),
            None,
            2140,
        )
        .await;
        session.clear_response_if_matches(&open.id).await;
        let trace = drain(&mut rx);
        assert_clean(&trace);

        let echoed = find(&trace, ITEM_ADDED).expect("the client item is echoed back");
        let client_item = item_of(&echoed).expect("echoed item carries an id");
        assert_ne!(
            client_item, "item_spoken_1",
            "the response item id is always new; reusing the client's id would make two items share one id"
        );

        let bracket: Vec<&str> = trace
            .iter()
            .map(|e| ty(e))
            .filter(|t| t.starts_with("response.") && *t != AUDIO_DELTA)
            .collect();
        assert_eq!(
            bracket,
            SPEAK_BRACKET_PORTAL_PARSES.to_vec(),
            "portal is coded against exactly this bracket\n{}",
            render(&trace)
        );

        let part = find(&trace, "response.content_part.added").expect("content part opened");
        assert_eq!(
            part["part"]["type"],
            json!("audio"),
            "the emitted part type is 'audio', not 'output_audio'; portal must parse what is emitted"
        );

        let done = find(&trace, RESPONSE_DONE).expect("response.done");
        assert_eq!(done["response"]["status"], json!("completed"));
        assert_eq!(done["response"]["output"][0]["id"], json!("item_spoken_1"));
        assert_eq!(
            done["response"]["output"][0]["content"][0]["transcript"],
            json!(SPEAK_TEXT)
        );
        assert_eq!(
            transcripts(&session).await,
            vec![SPEAK_TEXT.to_string()],
            "the client's item already carries the text; a second copy would double it in build_chat_messages"
        );
        drop(open.played_ms);
        drop(open.transcript_so_far);
        drop(open.wire_opened);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn client_cancel_mid_client_supplied_speech_closes_the_bracket_cleanly() {
        let Some((session, mut rx)) = conversation_session().await else {
            return;
        };
        inbound(&session, assistant_item_create(SPEAK_TEXT)).await;
        let _open = open_response(&session, "resp_speak_1", "item_spoken_1", true).await;
        inbound(&session, json!({"type": "response.cancel"})).await;
        let trace = drain(&mut rx);
        assert_clean(&trace);

        let done =
            find(&trace, RESPONSE_DONE).expect("a cancelled response still closes its bracket");
        assert_eq!(done["response"]["status"], json!("cancelled"));
        assert_eq!(
            done["response"]["status_details"]["reason"],
            json!("client_cancelled")
        );
        let item_done = find(&trace, "response.output_item.done").expect("close cascade");
        assert_eq!(item_done["item"]["status"], json!("incomplete"));
        assert!(
            error_messages(&trace).is_empty(),
            "cancellation is a normal terminal outcome; portal must not raise chat_err for it: {:?}",
            error_messages(&trace)
        );
        assert!(
            find(&trace, "conversation.item.assistant_truncated").is_none(),
            "the played_ms snapshot is always 0, so emit_server_truncate early-returns and portal cannot learn how much was spoken"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bargein_during_client_supplied_speech_closes_the_bracket_cleanly() {
        let Some((session, mut rx)) = conversation_session().await else {
            return;
        };
        inbound(&session, assistant_item_create(SPEAK_TEXT)).await;
        let _open = open_response(&session, "resp_speak_1", "item_spoken_1", true).await;
        pipeline::commit_bargein(&session, "item_user_next", 1234).await;
        let trace = drain(&mut rx);
        assert_clean(&trace);

        let done = find(&trace, RESPONSE_DONE).expect("barge-in closes the bracket");
        assert_eq!(done["response"]["status"], json!("cancelled"));
        assert_eq!(
            done["response"]["status_details"]["reason"],
            json!("barge_in")
        );
        assert!(
            error_messages(&trace).is_empty(),
            "barge-in is not a failure: {:?}",
            error_messages(&trace)
        );
        let done_at = index_of(&trace, RESPONSE_DONE).expect("done");
        let started_at =
            index_of(&trace, SPEECH_STARTED).expect("the interrupting turn is announced");
        assert!(
            started_at > done_at,
            "speech_started follows the cancelled bracket\n{}",
            render(&trace)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bargein_on_a_client_supplied_response_keeps_one_copy_in_history() {
        let Some((session, mut rx)) = conversation_session().await else {
            return;
        };
        inbound(&session, assistant_item_create(SPEAK_TEXT)).await;
        let _open = open_response(&session, "resp_speak_1", SPEAK_ITEM_ID, true).await;
        pipeline::commit_bargein(&session, "item_user_next", 1234).await;
        let _ = drain(&mut rx);
        let copies = transcripts(&session)
            .await
            .iter()
            .filter(|t| t.as_str() == SPEAK_TEXT)
            .count();
        assert_eq!(
            copies, 1,
            "a client-supplied response reuses the client's own item id, so the cancel path's apply_truncate_to_conversation finds that item and marks it incomplete instead of pushing a second copy; two copies would be fed to build_chat_messages and build_eou_context as separate assistant turns"
        );
        assert_eq!(
            transcripts(&session).await.len(),
            1,
            "the truncate path must not add an item of its own; consumption of client_speakable is covered by the_speak_path_never_reaches_the_llm, which drives the real selection"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_before_the_wire_opened_emits_no_bracket_at_all() {
        let Some((session, mut rx)) = conversation_session().await else {
            return;
        };
        inbound(&session, assistant_item_create(SPEAK_TEXT)).await;
        let _open = open_response(&session, "resp_speak_1", "item_spoken_1", false).await;
        inbound(&session, json!({"type": "response.cancel"})).await;
        let trace = drain(&mut rx);
        assert_clean(&trace);
        assert!(find(&trace, RESPONSE_CREATED).is_none());
        assert!(
            find(&trace, RESPONSE_DONE).is_none(),
            "a response cancelled before its brackets opened is closed silently, so portal's one-outstanding gate must also key on a timeout\n{}",
            render(&trace)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn response_create_with_a_user_item_newest_still_generates() {
        let Some((session, mut rx)) = conversation_session().await else {
            return;
        };
        inbound(&session, assistant_item_create(SPEAK_TEXT)).await;
        inbound(&session, user_item_create("what about tuesday")).await;
        {
            let state = session.state.lock().await;
            assert_eq!(
                select_response_source(&state.conversation, false, None),
                ResponseSource::Generate {
                    prompt: "what about tuesday".to_string()
                },
                "a bare response.create generates regardless of what the newest item is"
            );
        }
        let llm_configured = session.llm_config.is_some();
        inbound(&session, json!({"type": "response.create"})).await;
        assert_eq!(
            speakable_flags(&session).await,
            vec![true, false],
            "the generate path must not consume client_speakable"
        );
        let trace = collect_until_response_done(&mut rx).await;
        assert_clean(&trace);
        assert!(
            !error_messages(&trace).iter().any(|m| m == NO_USER_MESSAGE),
            "the user turn is the prompt: {:?}",
            error_messages(&trace)
        );
        assert!(
            find(&trace, RESPONSE_CREATED).is_some(),
            "generate opens a real response bracket\n{}",
            render(&trace)
        );
        if !llm_configured {
            let done = find(&trace, RESPONSE_DONE).expect("done");
            assert_eq!(
                done["response"]["status_details"]["reason"],
                json!("llm_error"),
                "with no CHAT_COMPLETION_BASE_URL the generate path fails at the LLM, which is proof it went to the LLM at all"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_speak_path_never_reaches_the_llm() {
        let Some((session, mut rx)) = conversation_session().await else {
            return;
        };
        inbound(&session, assistant_item_create(SPEAK_TEXT)).await;
        inbound(&session, speak_response_create()).await;
        assert_eq!(
            speakable_flags(&session).await,
            vec![false],
            "selection consumes the item under the same lock that read it"
        );
        let trace = collect_until_response_done(&mut rx).await;
        assert_clean(&trace);
        let done = find(&trace, RESPONSE_DONE).expect("done");
        assert_ne!(
            done["response"]["status_details"]["reason"],
            json!("llm_error"),
            "an llm_error on a speakable last item means select_response_source chose Generate\n{}",
            render(&trace)
        );
        assert_eq!(
            transcripts(&session).await,
            vec![SPEAK_TEXT.to_string()],
            "append_assistant_item is skipped for ClientSupplied, so the text stays in history exactly once"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_second_response_create_after_a_speak_falls_to_the_generate_path() {
        let Some((session, mut rx)) = conversation_session().await else {
            return;
        };
        inbound(&session, assistant_item_create(SPEAK_TEXT)).await;
        inbound(&session, speak_response_create()).await;
        let _ = collect_until_response_done(&mut rx).await;
        wait_until_no_response_is_active(&session).await;
        let _ = drain(&mut rx);
        inbound(&session, speak_response_create()).await;
        let trace = drain(&mut rx);
        assert!(
            error_messages(&trace)
                .iter()
                .any(|m| m.contains("already been spoken")),
            "naming a consumed item must be refused, never re-spoken: {:?}\n{}",
            error_messages(&trace),
            render(&trace)
        );
        assert!(find(&trace, RESPONSE_CREATED).is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn response_create_carrying_input_forces_the_generate_path() {
        let Some((session, mut rx)) = conversation_session().await else {
            return;
        };
        inbound(&session, assistant_item_create(SPEAK_TEXT)).await;
        inbound(
            &session,
            json!({"type": "response.create", "response": {"input": []}}),
        )
        .await;
        let trace = drain(&mut rx);
        assert_clean(&trace);
        assert!(
            error_messages(&trace).iter().any(|m| m == NO_USER_MESSAGE),
            "R1: a present input array forces Generate even for an empty array: {:?}",
            error_messages(&trace)
        );
        assert!(find(&trace, RESPONSE_CREATED).is_none());
        assert_eq!(
            speakable_flags(&session).await,
            vec![true],
            "a create routed to Generate must not consume the speakable item"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn response_create_while_input_speech_is_active_is_rejected_without_consuming() {
        let Some((session, mut rx)) = conversation_session().await else {
            return;
        };
        inbound(&session, assistant_item_create(SPEAK_TEXT)).await;
        {
            let mut state = session.state.lock().await;
            state.vad = VadPhase::Speaking {
                item_id: ItemId::new("item_user_live"),
                audio_start_ms: Millis(0),
            };
        }
        inbound(&session, speak_response_create()).await;
        let trace = drain(&mut rx);
        assert_clean(&trace);
        assert!(
            error_messages(&trace).iter().any(|m| m == SPEECH_ACTIVE),
            "the speak path must refuse to race register_response against an open mic: {:?}",
            error_messages(&trace)
        );
        assert!(
            find(&trace, RESPONSE_CREATED).is_none(),
            "no orphaned bracket may be emitted\n{}",
            render(&trace)
        );
        assert_eq!(
            speakable_flags(&session).await,
            vec![true],
            "a rejected create must leave the item speakable so portal can retry after speech_stopped"
        );
    }
}
