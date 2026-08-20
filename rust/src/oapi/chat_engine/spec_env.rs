#[cfg(feature = "cuda")]
use super::*;

pub(crate) const NV_CHAT_CONCURRENCY_DEFAULT: usize = 16;

pub(crate) const NV_CHAT_CONCURRENCY_MAX: usize = 64;

pub(crate) const NV_CHAT_QUEUE_MS_DEFAULT: u64 = 3000;

pub(crate) fn chat_permits(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(NV_CHAT_CONCURRENCY_DEFAULT)
        .min(NV_CHAT_CONCURRENCY_MAX)
}

pub(crate) fn chat_queue_ms(raw: Option<&str>) -> u64 {
    raw.and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(NV_CHAT_QUEUE_MS_DEFAULT)
}

pub(crate) async fn acquire_permit_bounded(
    sem: std::sync::Arc<tokio::sync::Semaphore>,
    permits: usize,
    queue: std::time::Duration,
) -> anyhow::Result<tokio::sync::OwnedSemaphorePermit> {
    let started = std::time::Instant::now();
    match tokio::time::timeout(queue, sem.acquire_owned()).await {
        Ok(Ok(permit)) => {
            let waited = started.elapsed();
            if waited > std::time::Duration::from_millis(50) {
                tracing::debug!(
                    waited_ms = waited.as_millis() as u64,
                    permits,
                    "chat request queued before acquiring a slot"
                );
            }
            Ok(permit)
        }
        Ok(Err(_)) => Err(anyhow::anyhow!("chat concurrency semaphore closed")),
        Err(_) => Err(anyhow::Error::new(crate::oapi::chat::EngineBusy::new(
            permits,
            queue.as_millis() as u64,
        ))),
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct ChatGate {
    pub(crate) sem: Arc<tokio::sync::Semaphore>,
    pub(crate) permits: usize,
    pub(crate) queue: std::time::Duration,
}

#[cfg(feature = "cuda")]
pub(crate) fn chat_gate() -> &'static ChatGate {
    static G: std::sync::OnceLock<ChatGate> = std::sync::OnceLock::new();
    G.get_or_init(|| {
        let raw = std::env::var("NV_CHAT_CONCURRENCY").ok();
        let permits = chat_permits(raw.as_deref());
        let queue_ms = chat_queue_ms(std::env::var("NV_CHAT_QUEUE_MS").ok().as_deref());
        tracing::info!(
            permits,
            queue_ms,
            backstop_default = NV_CHAT_CONCURRENCY_DEFAULT,
            "gemma4 chat concurrency gate armed. NV_CHAT_CONCURRENCY is a SAFETY BACKSTOP \
             against runaway task fan-out, NOT the throughput limiter: the real gate is VRAM \
             admission (see the 'vram admission gate armed' line and its \
             max_concurrent_upper_bound), which prices each request by the KV it actually needs \
             and sheds with 503 when the budget is exhausted. A semaphore permit is a poor cost \
             proxy - a 200-token and a 100k-token prompt each take one. Do NOT lower this to \
             tune concurrency; lower NV_VRAM_BUDGET_GIB or raise NV_ADMIT_TRANSIENT_GIB instead. \
             NV_CHAT_QUEUE_MS bounds the wait before a 503 engine_busy."
        );
        if permits < NV_CHAT_CONCURRENCY_DEFAULT {
            tracing::warn!(
                permits,
                backstop_default = NV_CHAT_CONCURRENCY_DEFAULT,
                "NV_CHAT_CONCURRENCY is set BELOW the backstop default, so the semaphore - not \
                 VRAM admission - is likely the binding constraint on chat concurrency. That \
                 caps throughput without capping VRAM. Unset it unless you are deliberately \
                 serializing (e.g. 1 for determinism debugging)."
            );
        }
        ChatGate {
            sem: Arc::new(tokio::sync::Semaphore::new(permits)),
            permits,
            queue: std::time::Duration::from_millis(queue_ms),
        }
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn chat_semaphore() -> Arc<tokio::sync::Semaphore> {
    chat_gate().sem.clone()
}

#[cfg(feature = "cuda")]
pub(crate) async fn acquire_chat_permit() -> anyhow::Result<tokio::sync::OwnedSemaphorePermit> {
    let g = chat_gate();
    acquire_permit_bounded(g.sem.clone(), g.permits, g.queue).await
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Eagle3Gate {
    NotRequested,

    Enabled,

    DegradedWarn,

    RequiredFail,
}

pub(crate) fn eagle3_gate(
    spec_requested: bool,
    required: bool,
    drafter_loaded: bool,
) -> Eagle3Gate {
    match (spec_requested, drafter_loaded, required) {
        (false, _, _) => Eagle3Gate::NotRequested,
        (true, true, _) => Eagle3Gate::Enabled,
        (true, false, true) => Eagle3Gate::RequiredFail,
        (true, false, false) => Eagle3Gate::DegradedWarn,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TargetDims {
    pub(crate) model_id: String,
    pub(crate) hidden_size: usize,
    pub(crate) vocab_size: usize,
    pub(crate) num_hidden_layers: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DrafterDims {
    pub(crate) fc_in_dim: usize,
    pub(crate) target_vocab_size: usize,
    pub(crate) aux_layer_ids: Vec<usize>,
}

pub(crate) fn drafter_target_mismatch(target: &TargetDims, drafter: &DrafterDims) -> Vec<String> {
    let mut out = Vec::new();
    let n_aux = drafter.aux_layer_ids.len();
    if n_aux == 0 {
        out.push(
            "aux_hidden_state_layer_ids: the drafter declares none, so there is nothing for the \
             target to emit and the fc projection can never be fed"
                .to_string(),
        );
    } else {
        let expected = n_aux.saturating_mul(target.hidden_size);
        if drafter.fc_in_dim != expected {
            out.push(format!(
                "hidden_size: drafter fc_in={} over {n_aux} aux layers implies a target \
                 hidden_size of {}, but this target has hidden_size={} (fc_in would have to be \
                 {expected})",
                drafter.fc_in_dim,
                drafter.fc_in_dim / n_aux,
                target.hidden_size,
            ));
        }
    }
    for &id in &drafter.aux_layer_ids {
        let zero_based = id.saturating_sub(1);
        if zero_based >= target.num_hidden_layers {
            out.push(format!(
                "aux_hidden_state_layer_ids: drafter asks for layer {id} (0-based {zero_based}) \
                 but this target has only {} layers",
                target.num_hidden_layers
            ));
        }
    }
    if drafter.target_vocab_size != target.vocab_size {
        out.push(format!(
            "vocab_size: drafter was trained against target_vocab_size={} but this target has \
             vocab_size={} (d2t maps draft ids into the target vocab and t2d is indexed by \
             target ids)",
            drafter.target_vocab_size, target.vocab_size
        ));
    }
    out
}

pub(crate) fn drafter_mismatch_message(
    kind: &str,
    dir: &str,
    target: &TargetDims,
    problems: &[String],
) -> String {
    format!(
        "{kind} drafter at {dir} does not match target model {}: {}. Spec-decode is DISABLED \
         for this engine and serving continues NON-SPECULATIVE (~half throughput) rather than \
         failing every request at generation time. Point NV_{}_DRAFT_DIR at a drafter trained \
         for this target, or unset it. Set NV_{}_REQUIRED=1 to make this fatal at startup \
         instead.",
        target.model_id,
        problems.join("; "),
        kind.to_ascii_uppercase(),
        kind.to_ascii_uppercase(),
    )
}

pub(crate) fn eagle3_required(raw: Option<&str>) -> bool {
    match raw {
        None => false,
        Some(v) => {
            let v = v.trim();
            !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false"))
        }
    }
}

pub(crate) fn dflash_required(raw: Option<&str>) -> bool {
    eagle3_required(raw)
}

pub(crate) fn drafter_wants_dflash(kind: &str) -> bool {
    matches!(kind, "dflash" | "auto" | "route")
}

pub(crate) fn dflash_spec_requested(no_spec: bool, drafter_kind: &str) -> bool {
    !no_spec && drafter_wants_dflash(drafter_kind)
}

pub(crate) fn nv_no_spec(raw: Option<&str>) -> bool {
    eagle3_required(raw)
}

pub(crate) fn env_flag_enabled(raw: Option<&str>) -> bool {
    raw.is_some_and(|v| v != "0")
}

pub(crate) fn spec_defer_drafter_from(raw: Option<&str>) -> bool {
    raw != Some("0")
}

pub(crate) fn eagle3_graph_chain_from(raw: Option<&str>) -> bool {
    raw != Some("0")
}

pub(crate) fn spec_requested(no_spec: bool, use_eagle3_set: bool, draft_dir_set: bool) -> bool {
    !no_spec && (use_eagle3_set || draft_dir_set)
}

pub(crate) fn spec_gate_for_request(no_spec: bool, use_eagle3_set: bool, greedy: bool) -> bool {
    !no_spec && (use_eagle3_set || greedy)
}

#[cfg(feature = "cuda")]
pub(crate) fn spec_gate_for_request_env(greedy: bool) -> bool {
    spec_gate_for_request(
        nv_no_spec(std::env::var("NV_NO_SPEC").ok().as_deref()),
        env_flag_enabled(std::env::var("NV_USE_EAGLE3").ok().as_deref()),
        greedy,
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn log_spec_degraded_rate_limited() {
    use std::sync::atomic::{AtomicU64, Ordering};
    static LAST_LOG_SECS: AtomicU64 = AtomicU64::new(0);
    const PERIOD_SECS: u64 = 60;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let last = LAST_LOG_SECS.load(Ordering::Relaxed);
    if now.saturating_sub(last) >= PERIOD_SECS
        && LAST_LOG_SECS
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    {
        tracing::error!(
            "NV_USE_EAGLE3 is set but the Eagle3 drafter is not loaded: serving \
             NON-SPECULATIVE decode at roughly half throughput. A harness can assert this \
             per request: /v1/chat/completions and /v1/completions answer with \
             `x-spec-decode: on|degraded|off`, and /v1/models carries the same value as \
             `spec_decode` on each chat model. Both report the ENGINE-level state decided at \
             load time (was a drafter requested, did it load) -- not per-request eligibility, \
             which additionally requires greedy sampling and no guided decode/logprobs/ \
             logit_bias. Set NV_EAGLE3_REQUIRED=1 to make this fatal at startup."
        );
    }
}

pub(crate) const EAGLE3_K_MIN: usize = 2;
pub(crate) const EAGLE3_K_MAX: usize = 64;
pub(crate) const EAGLE3_K_DEFAULT: usize = 3;
pub(crate) const EAGLE3_K_SHORT_DEFAULT: usize = 4;
pub(crate) const EAGLE3_K_CTX_GATE: usize = 8192;
pub(crate) const SPEC_VERIFY_HEADROOM: usize = 16;

pub(crate) const SPEC_PREFILL_CHUNK_MIN: usize = 16;
pub(crate) const SPEC_PREFILL_CHUNK_MAX: usize = 65535;
pub(crate) const SPEC_PREFILL_CHUNK_DEFAULT: usize = 1024;

pub(crate) fn resolve_cond_mode(raw: Option<&str>, use_drafter_kv: bool) -> (String, bool, bool) {
    let requested = raw.unwrap_or("shift");
    let (mode, forced) = match requested {
        "shift-force" => ("shift", true),
        m => (m, false),
    };
    if use_drafter_kv || forced {
        return (mode.to_string(), forced, false);
    }

    let needs_downgrade = matches!(mode, "shift" | "bonus");
    if needs_downgrade {
        (String::new(), false, true)
    } else {
        (mode.to_string(), false, false)
    }
}

pub(crate) fn spec_prefill_chunk(raw: Option<&str>) -> usize {
    let requested = match raw {
        None => return SPEC_PREFILL_CHUNK_DEFAULT,
        Some(s) => match s.trim().parse::<usize>() {
            Ok(v) => v,
            Err(_) => return SPEC_PREFILL_CHUNK_DEFAULT,
        },
    };
    let c = requested.clamp(SPEC_PREFILL_CHUNK_MIN, SPEC_PREFILL_CHUNK_MAX);
    if c != requested {
        tracing::warn!(
            requested,
            clamped = c,
            "NV_SPEC_PREFILL_CHUNK out of range [{SPEC_PREFILL_CHUNK_MIN}, {SPEC_PREFILL_CHUNK_MAX}]; clamped"
        );
    }
    c
}

pub(crate) fn eagle3_k_default(prompt_len: usize) -> usize {
    if prompt_len < EAGLE3_K_CTX_GATE {
        EAGLE3_K_SHORT_DEFAULT
    } else {
        EAGLE3_K_DEFAULT
    }
}

pub(crate) const SPEC_CTX_DISABLE_OFF: usize = usize::MAX;

pub(crate) fn spec_ctx_disable(raw: Option<&str>) -> usize {
    match raw.map(str::trim) {
        None | Some("") => SPEC_CTX_DISABLE_OFF,
        Some(s) => match s.parse::<usize>() {
            Ok(0) => SPEC_CTX_DISABLE_OFF,
            Ok(v) => v,
            Err(_) => SPEC_CTX_DISABLE_OFF,
        },
    }
}

pub(crate) fn eagle3_k(raw: Option<&str>, prompt_len: usize) -> usize {
    let requested = match raw {
        None => return eagle3_k_default(prompt_len),
        Some(s) => match s.trim().parse::<usize>() {
            Ok(v) => v,
            Err(_) => return eagle3_k_default(prompt_len),
        },
    };
    let k = requested.clamp(EAGLE3_K_MIN, EAGLE3_K_MAX);
    if k != requested {
        tracing::warn!(
            requested,
            clamped = k,
            "NV_EAGLE3_K out of range [{EAGLE3_K_MIN}, {EAGLE3_K_MAX}]; clamped"
        );
    }
    k
}

pub(crate) const DFLASH_K_DEFAULT: usize = 8;

pub(crate) fn dflash_k(raw: Option<&str>, block_size: usize) -> usize {
    dflash_k_with_default(raw, block_size, DFLASH_K_DEFAULT)
}

pub(crate) fn dflash_prose_k(raw: Option<&str>, block_size: usize, base: usize) -> usize {
    let cap = block_size
        .saturating_add(1)
        .clamp(EAGLE3_K_MIN, EAGLE3_K_MAX);
    match raw.and_then(|s| s.trim().parse::<usize>().ok()) {
        Some(v) => v.clamp(EAGLE3_K_MIN, cap),
        None => base,
    }
}

pub(crate) fn dflash_k_with_default(raw: Option<&str>, block_size: usize, default: usize) -> usize {
    let cap = block_size
        .saturating_add(1)
        .clamp(EAGLE3_K_MIN, EAGLE3_K_MAX);
    let requested = match raw {
        None => default,
        Some(s) => match s.trim().parse::<usize>() {
            Ok(v) => v,
            Err(_) => default,
        },
    };
    let k = requested.clamp(EAGLE3_K_MIN, cap);
    if k != requested {
        tracing::warn!(
            requested,
            clamped = k,
            block_size,
            "NV_DFLASH_K out of range [{EAGLE3_K_MIN}, {cap}]; clamped"
        );
    }
    k
}

pub(crate) const ADAPTIVE_K_MAX_DEFAULT: usize = 8;

pub(crate) fn adaptive_k_enabled(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

pub(crate) fn adaptive_k_graph(raw_max: Option<&str>, k_env: usize) -> usize {
    let m = raw_max
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(ADAPTIVE_K_MAX_DEFAULT);
    m.clamp(EAGLE3_K_MIN, EAGLE3_K_MAX).max(k_env)
}
#[cfg(feature = "cuda")]
pub(crate) const SPEC_SNAPSHOT_KEYS: &[&str] = &[
    "NV_ADAPTIVE_K",
    "NV_ADAPTIVE_K_MAX",
    "NV_DEBUG_DETERMINISM",
    "NV_DEBUG_GRAPH",
    "NV_DFLASH_GRAPH_EAGER",
    "NV_DFLASH_K",
    "NV_DFLASH_K_PROSE",
    "NV_DFLASH_NO_GRAPH",
    "NV_DRAFTER",
    "NV_DRAFTER_AUTO_SWITCH_TOKENS",
    "NV_EAGLE3_COND",
    "NV_EAGLE3_GRAPH_CHAIN",
    "NV_EAGLE3_K",
    "NV_EAGLE3_NO_DEVICE_CHAIN",
    "NV_EAGLE3_NO_DRAFTER_KV",
    "NV_EAGLE3_NO_GRAPH_CACHE",
    "NV_EAGLE3_TREE",
    "NV_EAGLE3_UNGRAPHED",
    "NV_MK_VERIFY_HD512",
    "NV_NO_SPEC",
    "NV_PROF_CHAT",
    "NV_PROF_PREFILL",
    "NV_ROUTE_CTX_GATE",
    "NV_SPEC_CTX_DISABLE",
    "NV_SPEC_DEFER_DRAFTER",
    "NV_SPEC_NO_GPU_ACCEPT",
    "NV_SPEC_PREFILL_CHUNK",
    "NV_SUFFIX_DRAFTER",
    "NV_SUFFIX_MIN_MATCH",
    "NV_USE_EAGLE3",
];

#[cfg(feature = "cuda")]
pub(crate) struct SpecEnvSnapshot {
    pub(crate) adaptive_k: Option<String>,
    pub(crate) adaptive_k_max: Option<String>,
    pub(crate) debug_determinism: bool,
    pub(crate) debug_graph: bool,
    pub(crate) dflash_graph_eager: bool,
    pub(crate) dflash_k: Option<String>,
    pub(crate) dflash_k_prose: Option<String>,
    pub(crate) dflash_no_graph: bool,
    pub(crate) drafter: Option<String>,
    pub(crate) drafter_auto_switch_tokens: Option<String>,
    pub(crate) eagle3_cond: Option<String>,
    pub(crate) eagle3_graph_chain: Option<String>,
    pub(crate) eagle3_k: Option<String>,
    pub(crate) eagle3_no_device_chain: bool,
    pub(crate) eagle3_no_drafter_kv: bool,
    pub(crate) eagle3_no_graph_cache: bool,
    pub(crate) eagle3_tree: bool,
    pub(crate) eagle3_ungraphed: Option<String>,
    pub(crate) mk_verify_hd512: Option<String>,
    pub(crate) no_spec: Option<String>,
    pub(crate) prof_chat: Option<String>,
    pub(crate) prof_prefill: bool,
    pub(crate) route_ctx_gate: Option<String>,
    pub(crate) spec_ctx_disable: Option<String>,
    pub(crate) spec_defer_drafter: Option<String>,
    pub(crate) spec_no_gpu_accept: bool,
    pub(crate) spec_prefill_chunk: Option<String>,
    pub(crate) suffix_drafter: Option<String>,
    pub(crate) suffix_min_match: Option<String>,
    pub(crate) use_eagle3: Option<String>,
}

#[cfg(feature = "cuda")]
impl SpecEnvSnapshot {
    pub(crate) fn capture() -> Self {
        Self::capture_with(&|k| std::env::var(k).ok(), &|k| {
            std::env::var_os(k).is_some()
        })
    }

    fn capture_with(get: &dyn Fn(&str) -> Option<String>, has: &dyn Fn(&str) -> bool) -> Self {
        Self {
            adaptive_k: get("NV_ADAPTIVE_K"),
            adaptive_k_max: get("NV_ADAPTIVE_K_MAX"),
            debug_determinism: has("NV_DEBUG_DETERMINISM"),
            debug_graph: has("NV_DEBUG_GRAPH"),
            dflash_graph_eager: has("NV_DFLASH_GRAPH_EAGER"),
            dflash_k: get("NV_DFLASH_K"),
            dflash_k_prose: get("NV_DFLASH_K_PROSE"),
            dflash_no_graph: has("NV_DFLASH_NO_GRAPH"),
            drafter: get("NV_DRAFTER"),
            drafter_auto_switch_tokens: get("NV_DRAFTER_AUTO_SWITCH_TOKENS"),
            eagle3_cond: get("NV_EAGLE3_COND"),
            eagle3_graph_chain: get("NV_EAGLE3_GRAPH_CHAIN"),
            eagle3_k: get("NV_EAGLE3_K"),
            eagle3_no_device_chain: has("NV_EAGLE3_NO_DEVICE_CHAIN"),
            eagle3_no_drafter_kv: has("NV_EAGLE3_NO_DRAFTER_KV"),
            eagle3_no_graph_cache: has("NV_EAGLE3_NO_GRAPH_CACHE"),
            eagle3_tree: get("NV_EAGLE3_TREE").is_some(),
            eagle3_ungraphed: get("NV_EAGLE3_UNGRAPHED"),
            mk_verify_hd512: get("NV_MK_VERIFY_HD512"),
            no_spec: get("NV_NO_SPEC"),
            prof_chat: get("NV_PROF_CHAT"),
            prof_prefill: has("NV_PROF_PREFILL"),
            route_ctx_gate: get("NV_ROUTE_CTX_GATE"),
            spec_ctx_disable: get("NV_SPEC_CTX_DISABLE"),
            spec_defer_drafter: get("NV_SPEC_DEFER_DRAFTER"),
            spec_no_gpu_accept: has("NV_SPEC_NO_GPU_ACCEPT"),
            spec_prefill_chunk: get("NV_SPEC_PREFILL_CHUNK"),
            suffix_drafter: get("NV_SUFFIX_DRAFTER"),
            suffix_min_match: get("NV_SUFFIX_MIN_MATCH"),
            use_eagle3: get("NV_USE_EAGLE3"),
        }
    }

    pub(crate) fn profile_line(&self) -> String {
        fn s(v: &Option<String>) -> String {
            v.clone().unwrap_or_else(|| "-".into())
        }
        format!(
            "drafter={} no_spec={} use_eagle3={} eagle3_k={} dflash_k={} dflash_k_prose={} \
             adaptive_k={} adaptive_k_max={} cond={} graph_chain={} prefill_chunk={} \
             ctx_disable={} route_ctx_gate={} auto_switch_tokens={} defer_drafter={} suffix={} suffix_min={} ungraphed={} tree={} hd512={} \
             no_drafter_kv={} no_graph_cache={} no_device_chain={} no_gpu_accept={} \
             dflash_no_graph={} dflash_graph_eager={} prof_prefill={} debug_det={} debug_graph={}",
            s(&self.drafter),
            s(&self.no_spec),
            s(&self.use_eagle3),
            s(&self.eagle3_k),
            s(&self.dflash_k),
            s(&self.dflash_k_prose),
            s(&self.adaptive_k),
            s(&self.adaptive_k_max),
            s(&self.eagle3_cond),
            s(&self.eagle3_graph_chain),
            s(&self.spec_prefill_chunk),
            s(&self.spec_ctx_disable),
            s(&self.route_ctx_gate),
            s(&self.drafter_auto_switch_tokens),
            s(&self.spec_defer_drafter),
            s(&self.suffix_drafter),
            s(&self.suffix_min_match),
            s(&self.eagle3_ungraphed),
            self.eagle3_tree,
            s(&self.mk_verify_hd512),
            self.eagle3_no_drafter_kv,
            self.eagle3_no_graph_cache,
            self.eagle3_no_device_chain,
            self.spec_no_gpu_accept,
            self.dflash_no_graph,
            self.dflash_graph_eager,
            self.prof_prefill,
            self.debug_determinism,
            self.debug_graph,
        )
    }
}

#[cfg(test)]
mod chat_queue_tests {
    use super::{
        acquire_permit_bounded, chat_permits, chat_queue_ms, NV_CHAT_CONCURRENCY_DEFAULT,
        NV_CHAT_CONCURRENCY_MAX, NV_CHAT_QUEUE_MS_DEFAULT,
    };
    use crate::oapi::chat::EngineBusy;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::sync::Semaphore;

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn concurrency_default_is_a_backstop_above_what_vram_admission_allows() {
        assert_eq!(
            NV_CHAT_CONCURRENCY_DEFAULT, 16,
            "the semaphore is a safety backstop, not the throughput limiter; VRAM admission \
             is the primary gate and fits ~9 short-prompt requests at the measured capacity \
             and transient pad (numbers: perf/runs.jsonl). Lowering this back under that \
             number makes the semaphore bind first again."
        );
        assert!(
            NV_CHAT_CONCURRENCY_DEFAULT > 9,
            "backstop must sit strictly above the measured VRAM-admission fit"
        );
        assert_eq!(chat_permits(None), NV_CHAT_CONCURRENCY_DEFAULT);
    }

    #[test]
    fn concurrency_env_still_overrides_in_both_directions() {
        assert_eq!(chat_permits(Some("1")), 1);
        assert_eq!(chat_permits(Some("3")), 3);
        assert_eq!(chat_permits(Some(" 32 ")), 32);
        assert_eq!(chat_permits(Some("999")), NV_CHAT_CONCURRENCY_MAX);
        assert_eq!(chat_permits(Some("0")), NV_CHAT_CONCURRENCY_DEFAULT);
        assert_eq!(chat_permits(Some("")), NV_CHAT_CONCURRENCY_DEFAULT);
        assert_eq!(chat_permits(Some("lots")), NV_CHAT_CONCURRENCY_DEFAULT);
    }

    #[test]
    fn queue_ms_defaults_when_unset_or_garbage() {
        assert_eq!(chat_queue_ms(None), NV_CHAT_QUEUE_MS_DEFAULT);
        assert_eq!(chat_queue_ms(Some("")), NV_CHAT_QUEUE_MS_DEFAULT);
        assert_eq!(chat_queue_ms(Some("soon")), NV_CHAT_QUEUE_MS_DEFAULT);
        assert_eq!(chat_queue_ms(Some("-1")), NV_CHAT_QUEUE_MS_DEFAULT);
        assert_eq!(NV_CHAT_QUEUE_MS_DEFAULT, 3000);
    }

    #[test]
    fn queue_ms_parses_and_zero_means_no_wait() {
        assert_eq!(chat_queue_ms(Some("500")), 500);
        assert_eq!(chat_queue_ms(Some(" 12000 ")), 12000);
        assert_eq!(chat_queue_ms(Some("0")), 0);
    }

    #[tokio::test]
    async fn bounded_acquire_sheds_as_engine_busy_once_the_window_expires() {
        let sem = Arc::new(Semaphore::new(1));
        let held = acquire_permit_bounded(sem.clone(), 1, Duration::from_millis(5_000))
            .await
            .expect("first acquire");
        let t0 = Instant::now();
        let err = acquire_permit_bounded(sem.clone(), 1, Duration::from_millis(80))
            .await
            .expect_err("second acquire must shed, not block forever");
        let elapsed = t0.elapsed();
        let busy = err
            .downcast_ref::<EngineBusy>()
            .expect("shed must be an EngineBusy so chat.rs maps it to 503 engine_busy");
        assert_eq!(busy.permits, 1);
        assert_eq!(busy.waited_ms, 80);
        assert!(
            elapsed >= Duration::from_millis(70) && elapsed < Duration::from_millis(3_000),
            "shed should land at the queue deadline, took {elapsed:?}"
        );
        drop(held);
        let _reacquired = acquire_permit_bounded(sem, 1, Duration::from_millis(5_000))
            .await
            .expect("slot is reusable after release");
    }

    #[tokio::test]
    async fn bounded_acquire_proceeds_when_a_slot_frees_inside_the_window() {
        let sem = Arc::new(Semaphore::new(1));
        let held = acquire_permit_bounded(sem.clone(), 1, Duration::from_millis(5_000))
            .await
            .expect("first acquire");
        let s2 = sem.clone();
        let waiter = tokio::spawn(async move {
            acquire_permit_bounded(s2, 1, Duration::from_millis(5_000))
                .await
                .map(|_| ())
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(held);
        waiter
            .await
            .expect("join")
            .expect("a waiter inside the window must get the freed slot, not a shed");
    }
}

#[cfg(test)]
mod drafter_dim_tests {
    use super::{drafter_mismatch_message, drafter_target_mismatch, DrafterDims, TargetDims};

    fn gemma4_31b() -> TargetDims {
        TargetDims {
            model_id: "nvidia/Gemma-4-31B-IT-NVFP4".into(),
            hidden_size: 5376,
            vocab_size: 262144,
            num_hidden_layers: 60,
        }
    }

    fn gemma4_e4b() -> TargetDims {
        TargetDims {
            model_id: "google/gemma-4-E4B-it".into(),
            hidden_size: 2560,
            vocab_size: 262144,
            num_hidden_layers: 42,
        }
    }

    fn speculator_31b() -> DrafterDims {
        DrafterDims {
            fc_in_dim: 3 * 5376,
            target_vocab_size: 262144,
            aux_layer_ids: vec![2, 30, 57],
        }
    }

    #[test]
    fn the_shipped_pair_is_accepted() {
        assert_eq!(
            drafter_target_mismatch(&gemma4_31b(), &speculator_31b()),
            Vec::<String>::new()
        );
    }

    #[test]
    fn the_31b_speculator_is_rejected_against_e4b() {
        let problems = drafter_target_mismatch(&gemma4_e4b(), &speculator_31b());
        assert_eq!(
            problems.len(),
            2,
            "expected exactly the hidden-size and aux-layer-range disagreements, got {problems:?}"
        );
        assert!(problems[0].contains("hidden_size"), "{problems:?}");
        assert!(problems[0].contains("16128"), "{problems:?}");
        assert!(problems[0].contains("5376"), "{problems:?}");
        assert!(problems[0].contains("2560"), "{problems:?}");
        assert!(problems[0].contains("7680"), "{problems:?}");
        assert!(
            problems[1].contains("aux_hidden_state_layer_ids"),
            "{problems:?}"
        );
        assert!(problems[1].contains("57"), "{problems:?}");
        assert!(problems[1].contains("42"), "{problems:?}");
    }

    #[test]
    fn hidden_size_check_mirrors_the_runtime_ensure() {
        for (n_aux, hidden) in [(3usize, 5376usize), (2, 2560), (5, 4096)] {
            let d = DrafterDims {
                fc_in_dim: n_aux * hidden,
                target_vocab_size: 7,
                aux_layer_ids: vec![1; n_aux],
            };
            let t = TargetDims {
                model_id: "t".into(),
                hidden_size: hidden,
                vocab_size: 7,
                num_hidden_layers: 8,
            };
            assert!(drafter_target_mismatch(&t, &d).is_empty());
            let mut off_by_one = t.clone();
            off_by_one.hidden_size = hidden + 1;
            assert_eq!(drafter_target_mismatch(&off_by_one, &d).len(), 1);
        }
    }

    #[test]
    fn a_right_sized_drafter_with_the_wrong_aux_count_is_rejected() {
        let d = DrafterDims {
            fc_in_dim: 3 * 5376,
            target_vocab_size: 262144,
            aux_layer_ids: vec![2, 30],
        };
        let problems = drafter_target_mismatch(&gemma4_31b(), &d);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains("hidden_size"), "{problems:?}");
    }

    #[test]
    fn zero_aux_layers_is_rejected_without_dividing_by_zero() {
        let d = DrafterDims {
            fc_in_dim: 16128,
            target_vocab_size: 262144,
            aux_layer_ids: vec![],
        };
        let problems = drafter_target_mismatch(&gemma4_31b(), &d);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains("declares none"), "{problems:?}");
    }

    #[test]
    fn the_last_layer_id_is_in_range_and_one_past_it_is_not() {
        let ok = DrafterDims {
            fc_in_dim: 3 * 5376,
            target_vocab_size: 262144,
            aux_layer_ids: vec![2, 30, 60],
        };
        assert!(drafter_target_mismatch(&gemma4_31b(), &ok).is_empty());
        let bad = DrafterDims {
            aux_layer_ids: vec![2, 30, 61],
            ..ok
        };
        assert_eq!(drafter_target_mismatch(&gemma4_31b(), &bad).len(), 1);
    }

    #[test]
    fn vocab_disagreement_is_reported_in_both_directions() {
        for v in [32000usize, 262145] {
            let t = TargetDims {
                vocab_size: v,
                ..gemma4_31b()
            };
            let problems = drafter_target_mismatch(&t, &speculator_31b());
            assert_eq!(problems.len(), 1, "{problems:?}");
            assert!(problems[0].contains("vocab_size"), "{problems:?}");
            assert!(problems[0].contains("262144"), "{problems:?}");
            assert!(problems[0].contains(&v.to_string()), "{problems:?}");
        }
    }

    #[test]
    fn the_message_names_both_sides_the_dir_and_the_env_knob() {
        let target = gemma4_e4b();
        let problems = drafter_target_mismatch(&target, &speculator_31b());
        let msg = drafter_mismatch_message("eagle3", "/hub/speculator.eagle3", &target, &problems);
        assert!(msg.contains("/hub/speculator.eagle3"), "{msg}");
        assert!(msg.contains("google/gemma-4-E4B-it"), "{msg}");
        assert!(msg.contains("NV_EAGLE3_DRAFT_DIR"), "{msg}");
        assert!(msg.contains("NV_EAGLE3_REQUIRED=1"), "{msg}");
        assert!(msg.contains("DISABLED"), "{msg}");
        let df = drafter_mismatch_message("dflash", "/hub/dflash", &target, &problems);
        assert!(df.contains("NV_DFLASH_DRAFT_DIR"), "{df}");
        assert!(df.contains("NV_DFLASH_REQUIRED=1"), "{df}");
    }
}

#[cfg(test)]
mod ctx_disable_tests {
    use super::{spec_ctx_disable, SPEC_CTX_DISABLE_OFF};

    #[test]
    fn ctx_disable_default_and_zero_are_off() {
        assert_eq!(spec_ctx_disable(None), SPEC_CTX_DISABLE_OFF);
        assert_eq!(spec_ctx_disable(Some("")), SPEC_CTX_DISABLE_OFF);
        assert_eq!(spec_ctx_disable(Some("0")), SPEC_CTX_DISABLE_OFF);
        assert_eq!(spec_ctx_disable(Some("nope")), SPEC_CTX_DISABLE_OFF);
    }

    #[test]
    fn ctx_disable_parses_threshold() {
        assert_eq!(spec_ctx_disable(Some("8192")), 8192);
        assert_eq!(spec_ctx_disable(Some(" 12000 ")), 12000);
    }
}

#[cfg(all(test, feature = "cuda"))]
mod snapshot_tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn spec_env_knobs_sorted_and_deduped() {
        for w in SPEC_SNAPSHOT_KEYS.windows(2) {
            assert!(
                w[0] < w[1],
                "SPEC_SNAPSHOT_KEYS out of order: {} >= {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn capture_reads_exactly_the_registered_knobs() {
        let seen = RefCell::new(Vec::<String>::new());
        let get = |k: &str| {
            seen.borrow_mut().push(k.to_string());
            None::<String>
        };
        let has = |k: &str| {
            seen.borrow_mut().push(k.to_string());
            false
        };
        let _ = SpecEnvSnapshot::capture_with(&get, &has);
        let mut seen = seen.into_inner();
        seen.sort();
        seen.dedup();
        let mut reg: Vec<String> = SPEC_SNAPSHOT_KEYS.iter().map(|s| s.to_string()).collect();
        reg.sort();
        assert_eq!(seen, reg);
    }
}
