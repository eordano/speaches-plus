
use std::fs;
use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::Arc;

use serde_json::{json, Value};
use speaches_plus::realtime::state::{
    ConversationItem, RespPhase, ResponseRuntime, SessionPhase, SessionState, VadPhase,
};
use speaches_plus::types::{ItemId, Millis, ResponseId};

const TRACE_FIXTURE_FLOOR: usize = 15;

const W_CHECKS: &[&str] = &[
    "W1_response_done_per_created",
    "W2_delta_only_between_created_and_done",
    "W3_committed_after_stopped_before_created",
    "W4_response_done_carries_audio_end_ms",
    "W6_no_response_events_after_done",
    "W7_assistant_truncated_paired_with_cancelled_done",
    "W8_client_create_paired_with_server_created",
];

fn repo_conformance_root() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest.ancestors() {
        let cand = ancestor.join("conformance");
        if cand.is_dir() {
            return Some(cand);
        }
    }
    None
}

fn canonical_lib_dir() -> PathBuf {
    let root = repo_conformance_root()
        .expect("conformance/ root must exist somewhere up from CARGO_MANIFEST_DIR");
    let lib = root.join("lib");
    assert!(
        lib.join("trace_invariants.py").is_file(),
        "canonical lib not found at {}/trace_invariants.py",
        lib.display()
    );
    lib
}

fn fixtures_root() -> PathBuf {
    let root = repo_conformance_root().expect("conformance/ root must exist");
    let cand = root.join("fixtures");
    assert!(cand.is_dir(), "missing {}", cand.display());
    cand
}

fn load_fixture_events(expected: &std::path::Path) -> Vec<serde_json::Value> {
    let text =
        fs::read_to_string(expected).unwrap_or_else(|e| panic!("read {}: {e}", expected.display()));
    let mut events = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("{}:{}: parse: {e}", expected.display(), i + 1));
        events.push(v);
    }
    events
}

fn import_canonical(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let lib_dir = canonical_lib_dir();
    let sys = py.import("sys")?;
    let path = sys.getattr("path")?;
    let lib_str = lib_dir.to_str().expect("lib dir path is utf-8").to_string();
    let mut already = false;
    let len: usize = path.len()?;
    for i in 0..len {
        let item = path.get_item(i)?;
        if let Ok(s) = item.extract::<String>() {
            if s == lib_str {
                already = true;
                break;
            }
        }
    }
    if !already {
        path.call_method1("insert", (0, lib_str))?;
    }
    py.import("trace_invariants")
}

fn json_value_to_py<'py>(py: Python<'py>, v: &serde_json::Value) -> PyResult<Bound<'py, PyAny>> {
    let json_mod = py.import("json")?;
    let s = serde_json::to_string(v).expect("re-serialize json");
    let any = json_mod.call_method1("loads", (s,))?;
    Ok(any)
}

fn events_to_py_list<'py>(
    py: Python<'py>,
    events: &[serde_json::Value],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for ev in events {
        list.append(json_value_to_py(py, ev)?)?;
    }
    Ok(list)
}

fn run_w_checks(
    py: Python<'_>,
    canonical: &Bound<'_, PyModule>,
    events: &[serde_json::Value],
) -> PyResult<Vec<(String, Vec<String>)>> {
    let checks: Bound<'_, PyDict> = canonical.getattr("CHECKS")?.cast_into()?;
    let py_events = events_to_py_list(py, events)?;
    let mut out = Vec::new();
    for name in W_CHECKS {
        let func = checks
            .get_item(*name)?
            .unwrap_or_else(|| panic!("canonical CHECKS missing {name}"));
        let res = func.call1((py_events.clone(), py.None()))?;
        let viols: Vec<String> = res.extract()?;
        out.push(((*name).to_string(), viols));
    }
    Ok(out)
}

#[test]
fn canonical_passes_every_conformance_fixture() {
    let fixtures = fixtures_root();
    let mut count = 0usize;
    let mut failures: Vec<String> = Vec::new();

    let result: PyResult<()> = Python::attach(|py| {
        let canonical = import_canonical(py)?;
        for entry in fs::read_dir(&fixtures).expect("read fixtures dir") {
            let entry = entry.expect("dir entry");
            if !entry.file_type().expect("ft").is_dir() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let expected = entry.path().join("expected.jsonl");
            if !expected.is_file() {
                continue;
            }
            count += 1;
            let events = load_fixture_events(&expected);
            let results = run_w_checks(py, &canonical, &events)?;
            let mut local: Vec<String> = Vec::new();
            for (check_name, viols) in &results {
                if !viols.is_empty() {
                    local.push(format!(
                        "  [FAIL] {check_name}\n    - {}",
                        viols.join("\n    - ")
                    ));
                }
            }
            if !local.is_empty() {
                failures.push(format!("{name}:\n{}", local.join("\n")));
            }
        }
        Ok(())
    });

    result.expect("canonical lib FFI failed (Python interpreter or import error)");
    eprintln!(
        "ran the W-check battery over {count} trace fixture(s) under {}",
        fixtures.display()
    );
    assert!(
        count >= TRACE_FIXTURE_FLOOR,
        "walked {count} trace fixture(s) under {}, floor is {TRACE_FIXTURE_FLOOR}. `count > 0` \
         was the only floor here and it cannot notice a corpus shrinking from {TRACE_FIXTURE_FLOOR} \
         to one. If fixtures were retired, lower the floor in the same commit and say which \
         families stopped being covered.",
        fixtures.display()
    );
    if !failures.is_empty() {
        panic!(
            "canonical lib reported failures on {} fixture(s):\n{}",
            failures.len(),
            failures.join("\n---\n")
        );
    }
}

#[test]
fn expected_traces_use_only_known_event_types() {
    use std::collections::HashSet;

    let known: HashSet<&str> = [
        "session.created",
        "session.updated",
        "session.done",
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "input_audio_buffer.cleared",
        "input_audio_buffer.partial_transcription",
        "conversation.item.added",
        "conversation.item.deleted",
        "conversation.item.truncated",
        "conversation.item.assistant_truncated",
        "conversation.item.input_audio_transcription.completed",
        "conversation.item.input_audio_transcription.failed",
        "response.created",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_audio_transcript.delta",
        "response.output_audio_transcript.done",
        "response.output_audio.delta",
        "response.output_audio.done",
        "response.done",
        "error",
        "session.update",
        "response.create",
        "response.cancel",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "input_audio_buffer.clear",
        "conversation.item.create",
        "conversation.item.delete",
        "conversation.item.truncate",
        "conversation.item.diarization",
    ]
    .into_iter()
    .collect();

    let fixtures = fixtures_root();
    let mut unknown: Vec<String> = Vec::new();
    for entry in fs::read_dir(&fixtures).expect("read fixtures dir") {
        let entry = entry.expect("dir entry");
        if !entry.file_type().expect("ft").is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let expected = entry.path().join("expected.jsonl");
        if !expected.is_file() {
            continue;
        }
        for ev in load_fixture_events(&expected) {
            let ty = ev.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if !known.contains(ty) {
                unknown.push(format!("{name}: unknown type {ty:?}"));
            }
        }
    }
    assert!(
        unknown.is_empty(),
        "fixtures contain wire types Rust does not know about; either add them to OutboundEvent / the allowlist, or fix the fixture:\n  - {}",
        unknown.join("\n  - ")
    );
}

fn dummy_runtime() -> ResponseRuntime {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("rt");
    let handle = rt.spawn(async {});
    ResponseRuntime {
        handle,
        transcript_so_far: Arc::new(tokio::sync::Mutex::new(String::new())),
        wire_opened: Arc::new(AtomicBool::new(false)),
    }
}

fn resp_active(s: &SessionState) -> bool {
    matches!(
        s.resp,
        RespPhase::Created { .. } | RespPhase::Streaming { .. } | RespPhase::Drain { .. }
    )
}

fn num(op: &Value, key: &str) -> i64 {
    op.get(key).and_then(Value::as_i64).unwrap_or(0)
}

fn str_of(op: &Value, key: &str) -> String {
    op.get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn replay(ops: &[Value]) -> Vec<Value> {
    let mut s = SessionState::default();
    let mut trace: Vec<Value> = Vec::new();
    let mut started = false;
    let mut planned: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
    let mut played: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();

    macro_rules! emit {
        ($v:expr) => {
            trace.push($v)
        };
    }
    macro_rules! maybe_start {
        () => {{
            if !started {
                s.session = SessionPhase::Active {
                    created_at_ms: Millis(0),
                };
                trace.push(json!({"type":"session.created","session":{"id":"sess_1"}}));
                started = true;
            }
        }};
    }

    for (i, op) in ops.iter().enumerate() {
        let name = op.get("op").and_then(Value::as_str).unwrap_or("");
        match name {
            "markActive" => {
                maybe_start!();
            }
            "session_update" => {
                maybe_start!();
                emit!(json!({"type":"session.updated","session":{"id":"sess_1"}}));
            }
            "session_update_invalid" => {
                maybe_start!();
                let code = {
                    let c = str_of(op, "code");
                    if c.is_empty() {
                        "session_update_invalid".to_string()
                    } else {
                        c
                    }
                };
                emit!(json!({"type":"error","code":code,"message":str_of(op,"message")}));
                emit!(json!({"type":"session.updated","session":{"id":"sess_1"}}));
            }
            "vad_speech_start" => {
                maybe_start!();
                let item = str_of(op, "item_id");
                let start_ms = num(op, "start_ms");
                if resp_active(&s) {
                    let ep = s.resp.epoch().map(|e| e.0 as i64).unwrap_or(0);
                    let pl = *played.get(&ep).unwrap_or(&0);
                    let rid = s
                        .resp
                        .id()
                        .map(|r| r.as_str().to_string())
                        .unwrap_or_default();
                    let iid = s
                        .resp
                        .item_id()
                        .map(|x| x.as_str().to_string())
                        .unwrap_or_default();
                    emit!(json!({"type":"response.done","response":{
                        "id":rid,"status":"cancelled","audio_end_ms":pl,"output":[{"id":iid}]}}));
                    emit!(json!({"type":"conversation.item.assistant_truncated",
                        "item_id":iid,"audio_end_ms":pl}));
                    s.resp_retire_to_none().expect("retire on bargein");
                }
                match &s.vad {
                    VadPhase::Speaking { .. } => {}
                    VadPhase::Stopped {
                        item_id,
                        audio_start_ms,
                        ..
                    } => {
                        let (iid, st) = (item_id.clone(), *audio_start_ms);
                        s.vad = VadPhase::Speaking {
                            item_id: iid,
                            audio_start_ms: st,
                        };
                    }
                    VadPhase::Silent => {
                        s.vad = VadPhase::Speaking {
                            item_id: ItemId::new(item.clone()),
                            audio_start_ms: Millis(start_ms as u64),
                        };
                        emit!(json!({"type":"input_audio_buffer.speech_started",
                            "item_id":item,"audio_start_ms":start_ms}));
                    }
                }
            }
            "vad_speech_end" => {
                maybe_start!();
                let end_ms = num(op, "end_ms");
                if let VadPhase::Speaking {
                    item_id,
                    audio_start_ms,
                } = &s.vad
                {
                    let (iid, st) = (item_id.clone(), *audio_start_ms);
                    let iid_str = iid.as_str().to_string();
                    s.vad = VadPhase::Stopped {
                        item_id: iid,
                        audio_start_ms: st,
                        audio_end_ms: Millis(end_ms as u64),
                    };
                    emit!(json!({"type":"input_audio_buffer.speech_stopped",
                        "item_id":iid_str,"audio_end_ms":end_ms}));
                }
            }
            "commit_fire" => {
                maybe_start!();
                if let VadPhase::Stopped { item_id, .. } = &s.vad {
                    let iid = item_id.as_str().to_string();
                    s.vad = VadPhase::Silent;
                    s.conversation
                        .push(ConversationItem::new_user_audio(iid.clone()));
                    emit!(json!({"type":"input_audio_buffer.committed","item_id":iid}));
                    emit!(
                        json!({"type":"conversation.item.added","item":{"id":iid,"role":"user"}})
                    );
                }
            }
            "transcription_complete" => {
                maybe_start!();
                emit!(
                    json!({"type":"conversation.item.input_audio_transcription.completed",
                    "item_id":str_of(op,"item_id"),"transcript":str_of(op,"transcript")})
                );
            }
            "response_create" => {
                maybe_start!();
                let rid = str_of(op, "resp_id");
                let iid = str_of(op, "item_id");
                s.resp_create_from_none(
                    ResponseId::new(rid.clone()),
                    ItemId::new(iid),
                    dummy_runtime(),
                )
                .unwrap_or_else(|e| panic!("op {i} response_create: {e:?}"));
                let mut resp = json!({"id":rid});
                if let Some(instr) = op.get("instructions").and_then(Value::as_str) {
                    resp["instructions"] = json!(instr);
                }
                emit!(json!({"type":"response.created","response":resp}));
            }
            "audio_delta" => {
                let ab = {
                    let a = num(op, "audio_bytes");
                    if a <= 0 {
                        1024
                    } else {
                        a
                    }
                };
                emit!(json!({"type":"response.output_audio.delta",
                    "response_id":str_of(op,"resp_id"),"audio":{"audio_bytes":ab}}));
            }
            "llm_complete" => {
                let ep = num(op, "epoch");
                *planned.entry(ep).or_insert(0) += num(op, "planned_ms");
                if matches!(s.resp, RespPhase::Created { .. }) {
                    s.resp_advance_to_streaming(Arc::new(AtomicU64::new(0)))
                        .expect("advance");
                }
                if matches!(s.resp, RespPhase::Streaming { .. }) {
                    s.resp_drain(planned[&ep] as u64).expect("drain");
                }
            }
            "audio_drained" => {
                let ep = num(op, "epoch");
                let pl = num(op, "played_ms");
                played.insert(ep, pl);
                if matches!(s.resp, RespPhase::Drain { .. })
                    && pl >= *planned.get(&ep).unwrap_or(&0)
                {
                    let rid = s
                        .resp
                        .id()
                        .map(|r| r.as_str().to_string())
                        .unwrap_or_default();
                    emit!(json!({"type":"response.output_audio.done",
                        "response":{"id":rid,"audio_end_ms":pl}}));
                    emit!(json!({"type":"response.done",
                        "response":{"id":rid,"status":"completed","audio_end_ms":pl}}));
                    s.resp_retire_to_none().expect("retire on drained");
                }
            }
            "response_failed" => {
                let pl = num(op, "played_ms");
                let reason = {
                    let r = str_of(op, "reason");
                    if r.is_empty() {
                        "llm_error".to_string()
                    } else {
                        r
                    }
                };
                assert!(
                    resp_active(&s),
                    "op {i} response_failed: no in-flight response"
                );
                let rid = s
                    .resp
                    .id()
                    .map(|r| r.as_str().to_string())
                    .unwrap_or_default();
                let iid = s
                    .resp
                    .item_id()
                    .map(|x| x.as_str().to_string())
                    .unwrap_or_default();
                emit!(json!({"type":"response.done","response":{
                    "id":rid,"status":"failed","audio_end_ms":pl,
                    "status_details":{"reason":reason},"output":[{"id":iid}]}}));
                s.resp_retire_to_none().expect("retire on failed");
            }
            "response_drain_cap_expired" => {
                let ep = num(op, "epoch");
                let pl = num(op, "played_ms");
                *planned.entry(ep).or_insert(0) += num(op, "planned_ms");
                if matches!(s.resp, RespPhase::Created { .. }) {
                    s.resp_advance_to_streaming(Arc::new(AtomicU64::new(0)))
                        .expect("advance");
                }
                if matches!(s.resp, RespPhase::Streaming { .. }) {
                    s.resp_drain(planned[&ep] as u64).expect("drain");
                }
                played.insert(ep, pl);
                if matches!(s.resp, RespPhase::Drain { .. }) {
                    let rid = s
                        .resp
                        .id()
                        .map(|r| r.as_str().to_string())
                        .unwrap_or_default();
                    let iid = s
                        .resp
                        .item_id()
                        .map(|x| x.as_str().to_string())
                        .unwrap_or_default();
                    emit!(json!({"type":"response.done","response":{
                        "id":rid,"status":"incomplete","audio_end_ms":pl,
                        "status_details":{"reason":"drain_cap"},"output":[{"id":iid}]}}));
                    s.resp_retire_to_none().expect("retire on drain_cap");
                }
            }
            other => panic!("op {i}: unknown op {other:?}"),
        }
    }
    trace
}

fn canonicalise(py: Python<'_>, events: &[Value]) -> PyResult<Vec<Value>> {
    import_canonical(py)?;
    let trace_diff = py.import("trace_diff")?;
    let py_events = events_to_py_list(py, events)?;
    let res = trace_diff.call_method1("canonicalise_events", (py_events,))?;
    let json_mod = py.import("json")?;
    let s: String = json_mod.call_method1("dumps", (res,))?.extract()?;
    Ok(serde_json::from_str(&s).expect("parse canonicalised json"))
}

#[test]
fn replays_input_jsonl_against_rust_fsm() {
    let fixtures = fixtures_root();
    let mut failures: Vec<String> = Vec::new();
    let mut count = 0usize;

    let result: PyResult<()> = Python::attach(|py| {
        for entry in fs::read_dir(&fixtures).expect("read fixtures dir") {
            let entry = entry.expect("dir entry");
            if !entry.file_type().expect("ft").is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let input = entry.path().join("input.jsonl");
            let expected = entry.path().join("expected.jsonl");
            if !input.is_file() || !expected.is_file() {
                continue;
            }
            count += 1;
            let ops = load_fixture_events(&input);
            let actual = replay(&ops);
            let expected_events = load_fixture_events(&expected);
            let canon_actual = canonicalise(py, &actual)?;
            let canon_expected = canonicalise(py, &expected_events)?;
            let n = canon_actual.len().max(canon_expected.len());
            for i in 0..n {
                let a = canon_actual.get(i);
                let e = canon_expected.get(i);
                if a != e {
                    failures.push(format!(
                        "{name}: diverge at {i}\n    expected={}\n    actual  ={}",
                        e.map(|v| v.to_string()).unwrap_or_else(|| "<none>".into()),
                        a.map(|v| v.to_string()).unwrap_or_else(|| "<none>".into()),
                    ));
                    break;
                }
            }
        }
        Ok(())
    });
    result.expect("canonical lib FFI failed");
    eprintln!("replayed {count} input.jsonl fixture(s) against the Rust FSM");
    assert!(
        count >= TRACE_FIXTURE_FLOOR,
        "replayed {count} fixture(s), floor is {TRACE_FIXTURE_FLOOR}"
    );
    assert!(
        failures.is_empty(),
        "Rust FSM replay diverged from expected.jsonl on {} fixture(s):\n{}",
        failures.len(),
        failures.join("\n---\n")
    );
}

#[test]
fn canonical_flags_w4_on_all_statuses() {
    let result: PyResult<()> = Python::attach(|py| {
        let canonical = import_canonical(py)?;
        for status in ["completed", "cancelled", "incomplete", "failed"] {
            let trace = vec![
                serde_json::json!({
                    "type": "session.created",
                    "session": {"id": "sess_1"},
                }),
                serde_json::json!({
                    "type": "response.created",
                    "response": {"id": "resp_1"},
                }),
                serde_json::json!({
                    "type": "response.done",
                    "response": {"id": "resp_1", "status": status},
                }),
            ];
            let results = run_w_checks(py, &canonical, &trace)?;
            let w4 = results
                .iter()
                .find(|(n, _)| n == "W4_response_done_carries_audio_end_ms")
                .expect("W4 entry");
            assert!(
                !w4.1.is_empty(),
                "canonical must flag W4 for status={status}; produced no violations"
            );
            let joined = w4.1.join("\n");
            assert!(
                joined.contains("audio_end_ms"),
                "canonical W4 output for status={status} lacked audio_end_ms diagnostic:\n{joined}"
            );
        }
        Ok(())
    });
    result.expect("canonical lib FFI failed");
}
