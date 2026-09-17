use std::sync::{LazyLock, Mutex};

#[derive(Clone, Debug, serde::Serialize)]
pub struct WorkerInfo {
    pub kv_kind: &'static str,
    pub prefix_supported: bool,
    pub max_seq: usize,
    pub chunk: usize,
    pub max_entries: usize,
    pub max_bytes: usize,
    pub draft_enabled: bool,
    pub draft_packed_linears: usize,
    pub draft_fallback_linears: usize,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Snapshot {
    pub worker_generation: u64,
    pub worker: Option<WorkerInfo>,
    pub request_id: u64,
    pub active_requests: usize,
    pub interleaved: bool,
    pub prompt_tokens: usize,
    pub selected_kind: &'static str,
    pub selected_position: usize,
    pub copy_starts: u64,
    pub copy_declared_bytes: u64,
    pub last_copy_kind: &'static str,
    pub budget_refusals: u64,
    pub last_refusal_kind: &'static str,
    pub last_refusal_bytes: usize,
    pub lru_entries: usize,
    pub charged_bytes: usize,
    pub history_len: usize,
    pub frontier: usize,
    pub graph_propose_rounds: usize,
    pub eager_propose_rounds: usize,
    pub outcome: &'static str,
    pub emitted_ids: Vec<u32>,
}

static SNAPSHOT: LazyLock<Mutex<Snapshot>> = LazyLock::new(|| Mutex::new(Snapshot::default()));

fn record(f: impl FnOnce(&mut Snapshot)) {
    if std::env::var("NV_LAGUNA_SERVE_STATS").as_deref() == Ok("1") {
        f(&mut SNAPSHOT.lock().expect("Laguna prefix diagnostics mutex"));
    }
}

pub fn snapshot() -> Snapshot {
    SNAPSHOT
        .lock()
        .expect("Laguna prefix diagnostics mutex")
        .clone()
}

pub fn worker_started(info: WorkerInfo) {
    record(|s| {
        s.interleaved |= s.active_requests != 0;
        s.worker_generation += 1;
        s.worker = Some(info);
        s.lru_entries = 0;
        s.charged_bytes = 0;
        s.history_len = 0;
        s.frontier = 0;
    });
}

pub fn request_started(prompt_tokens: usize) {
    record(|s| {
        s.interleaved |= s.active_requests != 0;
        s.active_requests += 1;
        s.request_id += 1;
        s.prompt_tokens = prompt_tokens;
        s.selected_kind = "cold";
        s.selected_position = 0;
        s.graph_propose_rounds = 0;
        s.eager_propose_rounds = 0;
        s.outcome = "running";
        s.emitted_ids.clear();
    });
}

pub fn draft_round(graph: bool) {
    record(|s| {
        if graph {
            s.graph_propose_rounds += 1;
        } else {
            s.eager_propose_rounds += 1;
        }
    });
}

pub fn selected(kind: &'static str, position: usize) {
    record(|s| {
        s.selected_kind = kind;
        s.selected_position = position;
    });
}

pub fn copy_started(kind: &'static str, declared_bytes: usize) {
    record(|s| {
        s.copy_starts += 1;
        s.copy_declared_bytes += declared_bytes as u64;
        s.last_copy_kind = kind;
    });
}

pub fn budget_refused(kind: &'static str, declared_bytes: usize) {
    record(|s| {
        s.budget_refusals += 1;
        s.last_refusal_kind = kind;
        s.last_refusal_bytes = declared_bytes;
    });
}

pub fn state(lru_entries: usize, charged_bytes: usize, history_len: usize, frontier: usize) {
    record(|s| {
        s.lru_entries = lru_entries;
        s.charged_bytes = charged_bytes;
        s.history_len = history_len;
        s.frontier = frontier;
    });
}

pub fn request_finished(outcome: &'static str, emitted_ids: &[u32]) {
    record(|s| {
        s.interleaved |= s.active_requests != 1;
        s.active_requests = s.active_requests.saturating_sub(1);
        s.outcome = outcome;
        s.emitted_ids = emitted_ids.to_vec();
    });
}
