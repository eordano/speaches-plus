use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

pub const GIB: f64 = (1u64 << 30) as f64;

pub const VRAM_PLAN_TOLERANCE_ENV: &str = "NV_VRAM_PLAN_TOLERANCE_GIB";
pub const VRAM_PLAN_STRICT_ENV_1_MAKES_THE_POST_BOOT_RECONCILE_FATAL: &str = "NV_VRAM_PLAN_STRICT";
pub const VRAM_PLAN_AUX_ENV_RESERVES_HEADROOM_FOR_SPEECH_AND_OCR_ENGINES: &str =
    "NV_VRAM_PLAN_AUX_GIB";

pub const RECONCILE_TOLERANCE_GIB_DEFAULT_COVERS_ALLOCATOR_ROUNDING_AND_GRAPH_POOLS: f64 = 3.0;
pub const CUDA_CONTEXT_AND_CUBLAS_WORKSPACE_SLACK_GIB: f64 = 1.5;
pub const FAMILY_GRAPH_CAPTURE_AND_TRANSIENT_SLACK_GIB: f64 = 2.0;

pub const KV_SOLVE_FLOOR_TOKENS_BELOW_WHICH_SERVING_IS_NOT_WORTH_BOOTING: usize = 4096;
pub const KV_SOLVE_GRANULARITY_1024_ROUNDS_DOWN_SO_A_SOLVED_KV_NEVER_EXCEEDS_BUDGET: usize = 1024;

pub const KV_MAX_SEQ_LEN_GLOBAL_ENV_THE_DEFAULT_EVERY_ENGINE_INHERITS: &str = "NV_KV_MAX_SEQ_LEN";
pub const KV_MAX_SEQ_LEN_Q38_ENV_PER_ENGINE_OVERRIDE_SO_THE_FLAGSHIP_LINEAR_ATTN_ENGINE_SERVES_FULL_CONTEXT: &str =
    "NV_KV_MAX_SEQ_LEN_Q38";
pub const KV_MAX_SEQ_LEN_GEMMA4_ENV_PER_ENGINE_OVERRIDE_SO_THE_DENSE_KV_VISION_ENGINE_STOPS_DRAGGING_THE_GLOBAL_KNOB_DOWN: &str =
    "NV_KV_MAX_SEQ_LEN_GEMMA4";
pub const KV_SOLVE_ORDER_IS_ENGINE_BOOT_ORDER_NV_CHAT_MODEL_DIRS_LEFT_TO_RIGHT_Q38_FIRST_ON_THE_GPU_HOST_SO_EACH_UNSET_ENGINE_TAKES_THE_LARGEST_KV_THE_REMAINING_BUDGET_ADMITS_AND_LATER_ENGINES_SOLVE_ONLY_WHAT_IT_LEFT: &str =
    "NV_CHAT_MODEL_DIRS";

pub const Q38_BATCH_KV_ENV_CAPS_LANE_POOL_DEPTH_SO_PREPAID_BATCH_LANES_STAY_SHALLOW_WHILE_SOLO_REQUESTS_GO_DEEP: &str =
    "NV_Q38_BATCH_KV";
pub const Q38_BATCH_KV_DEFAULT_CAP_TOKENS_LEDGER_MEASURED_1167_MIB_PER_LANE_AT_THIS_DEPTH: usize =
    32768;

pub const Q36_BATCH_KV_ENV_CAPS_LANE_POOL_DEPTH_SO_PREPAID_BATCH_LANES_STAY_SHALLOW_WHILE_SOLO_REQUESTS_GO_DEEP: &str =
    "NV_Q36_BATCH_KV";
pub const Q36_BATCH_KV_DEFAULT_CAP_TOKENS_SAME_32768_AS_Q38_BECAUSE_THE_Q36_LANE_COST_IS_DOMINATED_BY_ITS_DEPTH_CONSTANT_NOT_KV: usize =
    32768;
pub const Q36_BATCH_LANE_CONST_MIB_ENGINE_BOOT_MEASURED_4P24_GIB_PER_LANE_AT_832_AND_4P32_AT_8384_SO_ENGINE_BUFFERS_PLUS_GDN_STATE_DOMINATE_AND_KV_ADDS_THE_PER_TOKEN_TAIL: u64 =
    4342;

pub const KV_MAX_SEQ_LEN_UNMODELED_FAMILY_ENVS_PER_ENGINE_OVERRIDES_SO_ONE_FAMILY_LIFTS_WITHOUT_MOVING_EVERY_COBOOTED_ENGINE:
    &[(&str, &str)] = &[
    ("qwen3", "NV_KV_MAX_SEQ_LEN_QWEN3"),
    ("gemma4-moe", "NV_KV_MAX_SEQ_LEN_GEMMA4_MOE"),
    ("laguna", "NV_KV_MAX_SEQ_LEN_LAGUNA"),
    ("gpt-oss", "NV_KV_MAX_SEQ_LEN_GPTOSS"),
    ("omni", "NV_KV_MAX_SEQ_LEN_OMNI"),
];

pub fn per_engine_kv_env_for_family(family: &str) -> Option<&'static str> {
    match family {
        "qwen3.5-dense" => {
            return Some(KV_MAX_SEQ_LEN_Q38_ENV_PER_ENGINE_OVERRIDE_SO_THE_FLAGSHIP_LINEAR_ATTN_ENGINE_SERVES_FULL_CONTEXT)
        }
        "gemma4" => {
            return Some(KV_MAX_SEQ_LEN_GEMMA4_ENV_PER_ENGINE_OVERRIDE_SO_THE_DENSE_KV_VISION_ENGINE_STOPS_DRAGGING_THE_GLOBAL_KNOB_DOWN)
        }
        _ => {}
    }
    KV_MAX_SEQ_LEN_UNMODELED_FAMILY_ENVS_PER_ENGINE_OVERRIDES_SO_ONE_FAMILY_LIFTS_WITHOUT_MOVING_EVERY_COBOOTED_ENGINE
        .iter()
        .find(|(f, _)| *f == family)
        .map(|(_, env)| *env)
}

pub fn kv_binding_from(
    per_engine_env: Option<&'static str>,
    per_engine_raw: Option<&str>,
    global_raw: Option<&str>,
) -> Option<(usize, &'static str)> {
    let parse = |s: Option<&str>| {
        s.and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|n| *n > 0)
    };
    if let (Some(env), Some(n)) = (per_engine_env, parse(per_engine_raw)) {
        return Some((n, env));
    }
    parse(global_raw).map(|n| {
        (
            n,
            KV_MAX_SEQ_LEN_GLOBAL_ENV_THE_DEFAULT_EVERY_ENGINE_INHERITS,
        )
    })
}

pub fn q38_batch_lane_kv_from(kv_max: usize, raw: Option<&str>) -> usize {
    raw.and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(Q38_BATCH_KV_DEFAULT_CAP_TOKENS_LEDGER_MEASURED_1167_MIB_PER_LANE_AT_THIS_DEPTH)
        .min(kv_max.max(1))
}

pub fn q36_batch_lane_kv_from(kv_max: usize, raw: Option<&str>) -> usize {
    raw.and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(
            Q36_BATCH_KV_DEFAULT_CAP_TOKENS_SAME_32768_AS_Q38_BECAUSE_THE_Q36_LANE_COST_IS_DOMINATED_BY_ITS_DEPTH_CONSTANT_NOT_KV,
        )
        .min(kv_max.max(1))
}

pub const MIB: u64 = 1 << 20;

pub const VRAM_PLAN_KV_CONCURRENT_ENV_RESERVES_WORST_CASE_REQUEST_KV_SLOTS: &str =
    "NV_VRAM_PLAN_KV_CONCURRENT";
pub const KV_CONCURRENT_DEFAULT_1_LANE_POOLS_AND_THE_ADMISSION_GATE_BOUND_THE_REST: usize = 1;

pub const PRIOR_ENGINES_ITEM_LABEL_STRIPPED_ON_RECORD_SO_EACH_ENGINE_COUNTS_ONCE: &str =
    "prior-engines-and-aux";

pub const AUX_SPEECH_STACK_STEADY_MIB_MEASURED_LEDGER_ARM_AUX_WHISPER_CT2: u64 = 2638;
pub const AUX_EMBEDDING_STEADY_MIB_CKPT_SHARE_OF_MEASURED_LEDGER_ARM_AUX_EMBED_TTS: u64 = 1196;
pub const AUX_QWEN3_TTS_0P6B_FP8_CUDA_MIB_ONE_STREAM_PEAK_3912_MEASURED_IN_PROCESS_PLUS_1416_THE_F32_LEDGERS_SECOND_STREAM: u64 =
    5328;
pub const AUX_QWEN3_TTS_1P7B_FP8_CUDA_MIB_ONE_STREAM_PEAK_4813_MEASURED_IN_PROCESS_PLUS_1416_THE_F32_LEDGERS_SECOND_STREAM: u64 =
    6229;
pub const AUX_VOXTRAL_TTS_4B_BF16_CUDA_MIB_MEASURED_STANDALONE_PROCESS_AFTER_WARMUP: u64 = 17000;
pub const VOXTRAL_TTS_SPEECH_COMPONENT: &str = "voxtral-tts";
pub const AUX_QWEN3_TTS_1P7B_VOICEDESIGN_BF16_CUDA_MIB_MEASURED_STANDALONE_PROCESS_ONE_STREAM: u64 =
    6912;
pub const AUX_DOTS_OCR_STEADY_MIB_CKPT_SHARE_OF_MEASURED_LEDGER_OCR_PAIR_NET_14P6_GIB: u64 = 7068;
pub const AUX_DEEPSEEK_OCR2_STEADY_MIB_CKPT_SHARE_OF_MEASURED_LEDGER_OCR_PAIR_NET_14P6_GIB: u64 =
    7882;

pub const GEMMA4_STEADY_OVERHEAD_MIB_MEASURED_LEDGER_44684_MINUS_HEADERS_MINUS_I8_HEAD: u64 = 12219;
pub const GEMMA4_STEADY_OVERHEAD_MIB_UNDER_ATTN_NVFP4_MEASURED_LEDGER_28588_ARM: u64 = 7745;
pub const NVFP4_RESIDENT_NUMERATOR_9_BYTES_PER_32_BF16_BYTES_4BIT_PLUS_FP8_SCALE_PER_16: u64 = 9;
pub const NVFP4_RESIDENT_DENOMINATOR_32: u64 = 32;

pub const Q38_MTP_RUNTIME_STEADY_MIB_MEASURED_LEDGER_23435_MINUS_SOLO_21889_MINUS_CKPT_HEAD: u64 =
    736;

#[derive(Clone, Debug)]
pub struct PlanItem {
    pub label: String,
    pub bytes: u64,
    pub basis: String,
    pub unmeasured: bool,
}

#[derive(Clone, Debug, Default)]
pub struct KvShrinkKnob {
    pub env: &'static str,
    pub kv_max: usize,
    pub lanes: usize,
    pub approx_bytes_per_token_per_lane: u64,
    pub prior_engines_bytes_per_token_at_same_kv: u64,
    pub floor: usize,
}

impl KvShrinkKnob {
    pub fn process_bytes_per_token(&self) -> u64 {
        self.approx_bytes_per_token_per_lane.max(1) * self.lanes.max(1) as u64
            + self.prior_engines_bytes_per_token_at_same_kv
    }
}

pub fn kv_shrink_option(k: &KvShrinkKnob, over_bytes: u64) -> Option<String> {
    let per = k.process_bytes_per_token();
    let shave_tokens = over_bytes.div_ceil(per) as usize;
    if k.kv_max <= shave_tokens {
        return None;
    }
    let g = KV_SOLVE_GRANULARITY_1024_ROUNDS_DOWN_SO_A_SOLVED_KV_NEVER_EXCEEDS_BUDGET;
    let target = ((k.kv_max - shave_tokens) / g) * g;
    (target >= k.floor).then(|| {
        format!(
            "set {}={} (currently {}; frees {:.2} GiB)",
            k.env,
            target,
            k.kv_max,
            ((k.kv_max - target) as u64 * per) as f64 / GIB
        )
    })
}

#[derive(Clone, Debug)]
pub struct BucketShrinkKnob {
    pub env: &'static str,
    pub sizes: Vec<usize>,
    pub bytes_per_lane: u64,
}

#[derive(Clone, Debug)]
pub struct VramPlan {
    pub model_dir: String,
    pub family: &'static str,
    pub budget_bytes: u64,
    pub items: Vec<PlanItem>,
    pub refusal_enforced_for_this_family: bool,
    pub kv_knob: Option<KvShrinkKnob>,
    pub prior_kv_knobs: Vec<KvShrinkKnob>,
    pub bucket_knob: Option<BucketShrinkKnob>,
    pub drafter_off_knob: Option<(u64, &'static str)>,
}

impl VramPlan {
    pub fn new(model_dir: &Path, family: &'static str, budget_bytes: u64, enforced: bool) -> Self {
        Self {
            model_dir: model_dir.display().to_string(),
            family,
            budget_bytes,
            items: Vec::new(),
            refusal_enforced_for_this_family: enforced,
            kv_knob: None,
            prior_kv_knobs: Vec::new(),
            bucket_knob: None,
            drafter_off_knob: None,
        }
    }

    pub fn push(&mut self, label: impl Into<String>, bytes: u64, basis: impl Into<String>) {
        self.items.push(PlanItem {
            label: label.into(),
            bytes,
            basis: basis.into(),
            unmeasured: false,
        });
    }

    fn push_unmeasured(&mut self, label: &str, bytes: u64, basis: impl Into<String>) {
        self.items.push(PlanItem {
            label: label.into(),
            bytes,
            basis: basis.into(),
            unmeasured: true,
        });
    }

    pub fn planned_total_bytes(&self) -> u64 {
        self.items.iter().map(|i| i.bytes).sum()
    }

    pub fn fits(&self) -> bool {
        self.planned_total_bytes() <= self.budget_bytes
    }

    pub fn render(&self) -> String {
        let mut out = format!(
            "vram plan for {} (family {}, budget {:.2} GiB)\n",
            self.model_dir,
            self.family,
            self.budget_bytes as f64 / GIB
        );
        let width = self
            .items
            .iter()
            .map(|i| i.label.len())
            .max()
            .unwrap_or(8)
            .max(8);
        for i in &self.items {
            out.push_str(&format!(
                "  {:<width$}  {:>9.3} GiB  {}{}\n",
                i.label,
                i.bytes as f64 / GIB,
                if i.unmeasured {
                    "UNMEASURED CURRENT RESIDENCY: "
                } else {
                    ""
                },
                i.basis,
                width = width
            ));
        }
        let total = self.planned_total_bytes();
        let verdict = if self.fits() {
            format!(
                "headroom {:.3} GiB",
                (self.budget_bytes - total) as f64 / GIB
            )
        } else {
            format!(
                "OVER by {:.3} GiB",
                (total - self.budget_bytes) as f64 / GIB
            )
        };
        out.push_str(&format!(
            "  {:<width$}  {:>9.3} GiB  budget {:.2} GiB -> {}\n",
            "TOTAL",
            total as f64 / GIB,
            self.budget_bytes as f64 / GIB,
            verdict,
            width = width
        ));
        if self.items.iter().any(|i| i.unmeasured) {
            out.push_str("  INCOMPLETE ESTIMATE: declared headroom is not a bound on unmeasured components\n");
        }
        out
    }

    pub fn ranked_changes_to_fit(&self) -> Vec<String> {
        let total = self.planned_total_bytes();
        let over = total.saturating_sub(self.budget_bytes);
        if over == 0 {
            return vec!["already fits".into()];
        }
        let mut out = Vec::new();
        let mut suggested_envs: Vec<&'static str> = Vec::new();
        if let Some(k) = &self.kv_knob {
            suggested_envs.push(k.env);
            if let Some(line) = kv_shrink_option(k, over) {
                out.push(line);
            }
        }
        for k in &self.prior_kv_knobs {
            if suggested_envs.contains(&k.env) {
                continue;
            }
            suggested_envs.push(k.env);
            if let Some(line) = kv_shrink_option(k, over) {
                out.push(line);
            }
        }
        if let Some(b) = &self.bucket_knob {
            if b.sizes.len() > 1 {
                let lanes_now = *b.sizes.last().unwrap_or(&1) as u64;
                for cut in (0..b.sizes.len() - 1).rev() {
                    let lanes_then = b.sizes[cut] as u64;
                    if (lanes_now - lanes_then) * b.bytes_per_lane >= over {
                        let kept: Vec<String> =
                            b.sizes[..=cut].iter().map(|s| s.to_string()).collect();
                        out.push(format!(
                            "set {}={} (frees {:.2} GiB)",
                            b.env,
                            kept.join(","),
                            ((lanes_now - lanes_then) * b.bytes_per_lane) as f64 / GIB
                        ));
                        break;
                    }
                }
            }
        }
        if let Some((bytes, hint)) = &self.drafter_off_knob {
            if *bytes >= over {
                out.push(format!("{hint} (frees {:.2} GiB)", *bytes as f64 / GIB));
            }
        }
        out.push(format!(
            "raise NV_VRAM_BUDGET_GIB to at least {:.2}",
            (total as f64 / GIB) + 0.01
        ));
        out
    }

    pub fn ranked_changes_rendered(&self) -> String {
        self.ranked_changes_to_fit()
            .iter()
            .enumerate()
            .map(|(i, c)| format!("  option {}: {c}", i + 1))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn smallest_change_to_fit(&self) -> String {
        self.ranked_changes_to_fit()
            .into_iter()
            .next()
            .unwrap_or_else(|| "already fits".into())
    }

    pub fn enforce(&self) -> anyhow::Result<()> {
        if self.fits() {
            return Ok(());
        }
        if !self.refusal_enforced_for_this_family {
            tracing::warn!(
                "vram plan does not fit but the refusal is not enforced for this family -- expect CUDA OOM under load\n{}smallest changes that would fit, ranked:\n{}",
                self.render(),
                self.ranked_changes_rendered()
            );
            return Ok(());
        }
        anyhow::bail!(
            "vram plan exceeds NV_VRAM_BUDGET_GIB; refusing to boot a server that would OOM.\n{}smallest changes that would fit, ranked:\n{}",
            self.render(),
            self.ranked_changes_rendered()
        )
    }
}

pub fn device_total_vram_bytes_via_nvidia_smi() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .and_then(|l| l.trim().parse::<u64>().ok())
        .map(|mib| mib * 1024 * 1024)
}

pub fn env_budget_bytes() -> u64 {
    if let Some(v) = std::env::var("NV_VRAM_BUDGET_GIB")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
    {
        return (v * GIB) as u64;
    }
    let total = crate::oapi::admission::device_total_vram_bytes()
        .or_else(device_total_vram_bytes_via_nvidia_smi);
    match total {
        Some(t) => (t as f64 * crate::oapi::admission::DEFAULT_BUDGET_FRACTION) as u64,
        None => (crate::oapi::admission::default_budget_gib() * GIB) as u64,
    }
}

pub fn env_gib(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(default)
}

pub fn safetensors_shards(model_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let index = model_dir.join("model.safetensors.index.json");
    if index.is_file() {
        let text = std::fs::read_to_string(&index)?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let mut names: Vec<String> = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .map(|m| {
                m.values()
                    .filter_map(|f| f.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names.dedup();
        if !names.is_empty() {
            return Ok(names.into_iter().map(|n| model_dir.join(n)).collect());
        }
    }
    let mut out = Vec::new();
    for e in std::fs::read_dir(model_dir)? {
        let p = e?.path();
        if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            out.push(p);
        }
    }
    out.sort();
    anyhow::ensure!(
        !out.is_empty(),
        "no safetensors shards under {}",
        model_dir.display()
    );
    Ok(out)
}

pub fn shard_tensor_bytes(path: &Path) -> anyhow::Result<Vec<(String, u64)>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8)?;
    let header_len = u64::from_le_bytes(len8);
    anyhow::ensure!(
        header_len <= 256 * 1024 * 1024,
        "unreasonable safetensors header length {header_len} in {}",
        path.display()
    );
    let mut buf = vec![0u8; header_len as usize];
    f.read_exact(&mut buf)?;
    let v: serde_json::Value = serde_json::from_slice(&buf)?;
    let obj = v
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("safetensors header is not an object"))?;
    let mut out = Vec::with_capacity(obj.len());
    for (name, meta) in obj {
        if name == "__metadata__" {
            continue;
        }
        let offs = meta
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .ok_or_else(|| anyhow::anyhow!("tensor {name} has no data_offsets"))?;
        let (a, b) = (
            offs.first().and_then(|x| x.as_u64()).unwrap_or(0),
            offs.get(1).and_then(|x| x.as_u64()).unwrap_or(0),
        );
        out.push((name.clone(), b.saturating_sub(a)));
    }
    Ok(out)
}

pub fn checkpoint_tensor_bytes(model_dir: &Path) -> anyhow::Result<Vec<(String, u64)>> {
    let mut out = Vec::new();
    for shard in safetensors_shards(model_dir)? {
        out.extend(shard_tensor_bytes(&shard)?);
    }
    Ok(out)
}

pub fn sum_where(entries: &[(String, u64)], pred: impl Fn(&str) -> bool) -> u64 {
    entries
        .iter()
        .filter(|(n, _)| pred(n))
        .map(|(_, b)| *b)
        .sum()
}

pub fn solve_max_kv_tokens(
    remaining_bytes: u64,
    bytes_per_token_per_lane: u64,
    lanes: usize,
    cap_tokens: usize,
) -> Option<usize> {
    let per = bytes_per_token_per_lane.max(1) * lanes.max(1) as u64;
    let raw = (remaining_bytes / per) as usize;
    let g = KV_SOLVE_GRANULARITY_1024_ROUNDS_DOWN_SO_A_SOLVED_KV_NEVER_EXCEEDS_BUDGET;
    let n = (raw / g) * g;
    let n = n.min(cap_tokens);
    (n >= KV_SOLVE_FLOOR_TOKENS_BELOW_WHICH_SERVING_IS_NOT_WORTH_BOOTING).then_some(n)
}

pub fn solve_max_kv_tokens_monotone(
    fits: impl Fn(usize) -> bool,
    cap_tokens: usize,
) -> Option<usize> {
    let g = KV_SOLVE_GRANULARITY_1024_ROUNDS_DOWN_SO_A_SOLVED_KV_NEVER_EXCEEDS_BUDGET;
    let floor = KV_SOLVE_FLOOR_TOKENS_BELOW_WHICH_SERVING_IS_NOT_WORTH_BOOTING;
    if cap_tokens >= floor && fits(cap_tokens) {
        return Some(cap_tokens);
    }
    let (mut lo, mut hi) = (floor / g, cap_tokens / g);
    if hi <= lo || !fits(lo * g) {
        return None;
    }
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if fits(mid * g) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo * g)
}

pub fn solve_largest_bucket_prefix_that_fits(
    sizes_sorted: &[usize],
    bytes_per_lane: u64,
    remaining_bytes: u64,
) -> Vec<usize> {
    let mut kept: Vec<usize> = Vec::new();
    for &s in sizes_sorted {
        if (s as u64) * bytes_per_lane <= remaining_bytes {
            kept.push(s);
        } else {
            break;
        }
    }
    kept
}

pub fn kv_concurrent_cap_from(raw: Option<&str>) -> usize {
    raw.and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(KV_CONCURRENT_DEFAULT_1_LANE_POOLS_AND_THE_ADMISSION_GATE_BOUND_THE_REST)
}

pub fn kv_concurrent_cap() -> usize {
    kv_concurrent_cap_from(
        std::env::var(VRAM_PLAN_KV_CONCURRENT_ENV_RESERVES_WORST_CASE_REQUEST_KV_SLOTS)
            .ok()
            .as_deref(),
    )
}

#[cfg(any(feature = "cuda", test))]
fn qwen_lazy_kv_slots(
    batch_on: bool,
    mtp_selected: bool,
    solo_via_lanes: bool,
    kv_slots: usize,
) -> usize {
    if batch_on && solo_via_lanes && !mtp_selected {
        0
    } else {
        kv_slots
    }
}

#[derive(Clone, Debug)]
pub struct LoadedSpeechProvider {
    pub component: &'static str,
    pub provider: &'static str,
}

static LOADED_SPEECH_PROVIDERS: OnceLock<Vec<LoadedSpeechProvider>> = OnceLock::new();

pub fn observe_loaded_speech_stack(models: &crate::models::Models) {
    let mut providers = Vec::new();
    if let Some(stt) = models.whisper_opt() {
        let component = match stt {
            crate::stt::WhisperHandle::Ct2 { .. } => "whisper-ct2",
            crate::stt::WhisperHandle::WhisperCpp { .. } => "whisper.cpp",
            crate::stt::WhisperHandle::Parakeet { .. } => "parakeet-encoder",
            crate::stt::WhisperHandle::Voxtral { .. } => "voxtral-realtime",
        };
        providers.push(LoadedSpeechProvider {
            component,
            provider: stt.ep().unwrap_or("unreported"),
        });
        if matches!(stt, crate::stt::WhisperHandle::Parakeet { .. }) {
            providers.push(LoadedSpeechProvider {
                component: "parakeet-nemo/joint",
                provider: "unreported",
            });
        }
    }
    if let Some(kokoro) = models.kokoro.as_ref() {
        providers.push(LoadedSpeechProvider {
            component: "kokoro",
            provider: kokoro.provider(),
        });
    }
    if let Some(v) = models.voxtral_tts.as_ref() {
        providers.push(LoadedSpeechProvider {
            component: VOXTRAL_TTS_SPEECH_COMPONENT,
            provider: v.device_name(),
        });
    }
    if let Some(seg) = models.diar_segmentation.as_ref() {
        providers.push(LoadedSpeechProvider {
            component: "diar-segmentation",
            provider: seg.provider(),
        });
    }
    if let Some(embedding) = models.diar_embedding.as_ref() {
        providers.push(LoadedSpeechProvider {
            component: "diar-embedding",
            provider: embedding.provider(),
        });
    }
    let _ = LOADED_SPEECH_PROVIDERS.set(providers);
}

#[derive(Clone, Debug, Default)]
pub struct AuxEnv {
    pub aux_override_gib: Option<f64>,
    pub speech_stack: bool,
    pub loaded_speech: Option<Vec<LoadedSpeechProvider>>,
    pub embedding: bool,
    pub tts_talker: bool,
    pub tts_talker_on_cuda: bool,
    pub tts_talker_1p7b: bool,
    pub tts_voice_design: bool,
    pub tts_voice_design_on_cuda: bool,
    pub dots_ocr: bool,
    pub deepseek_ocr: bool,
    pub diar_gpu_limit_mib: Option<u64>,
}

impl AuxEnv {
    fn speech_is_cpu_only(&self) -> bool {
        self.loaded_speech.as_ref().is_some_and(|providers| {
            providers.iter().all(|p| {
                p.provider.eq_ignore_ascii_case("cpu")
                    || p.provider.eq_ignore_ascii_case("CPUExecutionProvider")
            })
        })
    }

    fn voxtral_tts_on_cuda(&self) -> bool {
        self.loaded_speech.as_ref().is_some_and(|providers| {
            providers
                .iter()
                .any(|p| p.component == VOXTRAL_TTS_SPEECH_COMPONENT && p.provider == "cuda")
        })
    }

    fn speech_provider_basis(&self) -> String {
        match &self.loaded_speech {
            None => "loaded providers unreported".into(),
            Some(providers) if providers.is_empty() => "no speech handles loaded".into(),
            Some(providers) => providers
                .iter()
                .map(|p| format!("{}={}", p.component, p.provider))
                .collect::<Vec<_>>()
                .join(", "),
        }
    }

    pub fn from_env() -> Self {
        let tts_talker = nv_config::env_arm_present_at_all_reread_every_call("NV_TTS_TALKER_DIR");
        let tts_voice_design = nv_config::env_arm_present_at_all_reread_every_call(
            crate::oapi::audio_speech_nvtts::ENV_VOICE_DESIGN_DIR,
        );
        let tts_on_cuda = (tts_talker || tts_voice_design)
            && crate::oapi::audio_speech_nvtts::qwen3_tts_device().is_cuda();
        Self {
            aux_override_gib: std::env::var(
                VRAM_PLAN_AUX_ENV_RESERVES_HEADROOM_FOR_SPEECH_AND_OCR_ENGINES,
            )
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0),
            speech_stack: LOADED_SPEECH_PROVIDERS.get().is_some()
                || nv_config::env_arm_present_at_all_reread_every_call("SPEACHES_PLUS_MODELS"),
            loaded_speech: LOADED_SPEECH_PROVIDERS.get().cloned(),
            embedding: nv_config::env_arm_present_at_all_reread_every_call("NV_EMBEDDING_MODEL_DIR"),
            tts_talker,
            tts_talker_on_cuda: tts_talker && tts_on_cuda,
            tts_talker_1p7b: std::env::var("NV_TTS_TALKER_DIR")
                .is_ok_and(|dir| dir.contains("1.7B")),
            tts_voice_design,
            tts_voice_design_on_cuda: tts_voice_design && tts_on_cuda,
            dots_ocr: nv_config::env_arm_present_at_all_reread_every_call("NV_OCR_DOTS_DIR"),
            deepseek_ocr: nv_config::env_arm_present_at_all_reread_every_call("NV_OCR_DEEPSEEK_DIR"),
            diar_gpu_limit_mib: std::env::var("DIAR_GPU_MEM_LIMIT_MB")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .filter(|v| *v > 0),
        }
    }
}

pub fn push_aux_items(plan: &mut VramPlan, aux: &AuxEnv) {
    if let Some(gib) = aux.aux_override_gib {
        plan.push(
            "aux-engines-reserved",
            (gib * GIB) as u64,
            format!(
                "{} supplies an operator reservation, not measured itemization",
                VRAM_PLAN_AUX_ENV_RESERVES_HEADROOM_FOR_SPEECH_AND_OCR_ENGINES
            ),
        );
        return;
    }
    if aux.speech_stack {
        let providers = aux.speech_provider_basis();
        if aux.speech_is_cpu_only() {
            plan.push(
                "aux-speech-stack",
                0,
                format!("CPU-only or absent loaded speech handles; {providers}"),
            );
        } else {
            plan.push_unmeasured(
                "aux-speech-stack",
                AUX_SPEECH_STACK_STEADY_MIB_MEASURED_LEDGER_ARM_AUX_WHISPER_CT2 * MIB,
                format!("legacy Whisper-CT2 reserve 2638 MiB; current provider mix/arenas unmeasured; {providers}"),
            );
        }
    }
    if aux.embedding {
        plan.push_unmeasured(
            "aux-qwen3-embedding",
            AUX_EMBEDDING_STEADY_MIB_CKPT_SHARE_OF_MEASURED_LEDGER_ARM_AUX_EMBED_TTS * MIB,
            "legacy checkpoint-ratio reserve 1196 MiB; current embedding device/residency unmeasured",
        );
    }
    if aux.tts_talker && aux.tts_talker_on_cuda {
        let (mib, size) = if aux.tts_talker_1p7b {
            (AUX_QWEN3_TTS_1P7B_FP8_CUDA_MIB_ONE_STREAM_PEAK_4813_MEASURED_IN_PROCESS_PLUS_1416_THE_F32_LEDGERS_SECOND_STREAM, "1.7B")
        } else {
            (AUX_QWEN3_TTS_0P6B_FP8_CUDA_MIB_ONE_STREAM_PEAK_3912_MEASURED_IN_PROCESS_PLUS_1416_THE_F32_LEDGERS_SECOND_STREAM, "0.6B")
        };
        plan.push(
            "aux-qwen3-tts-talker",
            mib * MIB,
            &format!("qwen3_tts_device() is CUDA: {size} fp8 talker and code-predictor projections over bf16 activations, F32 vocoder, two streams in flight"),
        );
    } else if aux.tts_talker {
        plan.push(
            "aux-qwen3-tts-talker",
            0,
            "qwen3_tts_device() is the CPU: talker, code predictor and vocoder load there",
        );
    }
    if aux.voxtral_tts_on_cuda() {
        plan.push(
            "aux-voxtral-tts",
            AUX_VOXTRAL_TTS_4B_BF16_CUDA_MIB_MEASURED_STANDALONE_PROCESS_AFTER_WARMUP * MIB,
            "voxtral-tts loaded on CUDA: bf16 backbone, flow head and codec resident after warmup",
        );
    }
    if aux.tts_voice_design && aux.tts_voice_design_on_cuda {
        plan.push(
            "aux-qwen3-tts-voicedesign",
            AUX_QWEN3_TTS_1P7B_VOICEDESIGN_BF16_CUDA_MIB_MEASURED_STANDALONE_PROCESS_ONE_STREAM
                * MIB,
            "cold-tier resident on CUDA: BF16 1.7B VoiceDesign talker and code predictor, F32 \
             vocoder, one stream in flight; resident only between a request and its idle or \
             pressure eviction",
        );
    } else if aux.tts_voice_design {
        plan.push(
            "aux-qwen3-tts-voicedesign",
            0,
            "qwen3_tts_device() is the CPU: the VoiceDesign talker loads there",
        );
    }
    if aux.dots_ocr {
        plan.push_unmeasured(
            "aux-dots-ocr",
            AUX_DOTS_OCR_STEADY_MIB_CKPT_SHARE_OF_MEASURED_LEDGER_OCR_PAIR_NET_14P6_GIB * MIB,
            "ckpt-ratio share of VRAM-LEDGER OCR-pair net 14.6 GiB (aux arm minus split arms \
             plus shared context)",
        );
    }
    if aux.deepseek_ocr {
        plan.push_unmeasured(
            "aux-deepseek-ocr-2",
            AUX_DEEPSEEK_OCR2_STEADY_MIB_CKPT_SHARE_OF_MEASURED_LEDGER_OCR_PAIR_NET_14P6_GIB * MIB,
            "ckpt-ratio share of VRAM-LEDGER OCR-pair net 14.6 GiB (aux arm minus split arms \
             plus shared context)",
        );
    }
    if let Some(mib) = aux.diar_gpu_limit_mib {
        if !aux.speech_is_cpu_only() {
            plan.push_unmeasured(
                "aux-diarization-ort-arena",
                mib.saturating_mul(MIB),
                "DIAR_GPU_MEM_LIMIT_MB is a per-session arena limit, not aggregate residency; this single-limit reserve does not bound Parakeet/diar sessions or Kokoro",
            );
        }
    }
}

pub fn push_prior_or_aux_process_items(plan: &mut VramPlan, prior_bytes: u64, aux: &AuxEnv) {
    if prior_bytes > 0 {
        plan.push(
            PRIOR_ENGINES_ITEM_LABEL_STRIPPED_ON_RECORD_SO_EACH_ENGINE_COUNTS_ONCE,
            prior_bytes,
            "running sum of previously enforced boot plans in this process; every enforce sees \
             the process-wide total, not this model dir alone",
        );
    } else {
        push_aux_items(plan, aux);
    }
}

static BOOT_PLANS: OnceLock<Mutex<Vec<(u64, VramPlan)>>> = OnceLock::new();
static BOOT_PLAN_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn boot_plans() -> &'static Mutex<Vec<(u64, VramPlan)>> {
    BOOT_PLANS.get_or_init(|| Mutex::new(Vec::new()))
}

#[must_use]
pub struct RecordedBootPlan {
    seq: u64,
}

impl Drop for RecordedBootPlan {
    fn drop(&mut self) {
        unrecord_boot_plan_when_its_owner_drops_so_neither_failed_boots_nor_dropped_engines_leave_phantom_prior_engine_bytes(self.seq);
    }
}

fn unrecord_boot_plan_when_its_owner_drops_so_neither_failed_boots_nor_dropped_engines_leave_phantom_prior_engine_bytes(
    seq: u64,
) {
    boot_plans().lock().unwrap().retain(|(s, _)| *s != seq);
}

pub fn record_boot_plan(plan: &VramPlan) -> RecordedBootPlan {
    let mut own = plan.clone();
    own.items.retain(|i| {
        i.label != PRIOR_ENGINES_ITEM_LABEL_STRIPPED_ON_RECORD_SO_EACH_ENGINE_COUNTS_ONCE
    });
    let seq = BOOT_PLAN_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    boot_plans().lock().unwrap().push((seq, own));
    RecordedBootPlan { seq }
}

pub fn recorded_kv_bytes_per_token_at(kv_max: usize, env: &str) -> u64 {
    boot_plans()
        .lock()
        .unwrap()
        .iter()
        .filter_map(|(_, p)| p.kv_knob.as_ref())
        .filter(|k| k.kv_max == kv_max && k.env == env)
        .map(|k| k.approx_bytes_per_token_per_lane.max(1) * k.lanes.max(1) as u64)
        .sum()
}

pub fn recorded_kv_knobs() -> Vec<KvShrinkKnob> {
    boot_plans()
        .lock()
        .unwrap()
        .iter()
        .filter_map(|(_, p)| p.kv_knob.clone())
        .collect()
}

pub fn recorded_planned_total_bytes() -> u64 {
    boot_plans()
        .lock()
        .unwrap()
        .iter()
        .map(|(_, p)| p.planned_total_bytes())
        .sum()
}

pub const GEMMA4_KV_RESERVE_LABEL_PREFIX_IS_WHAT_THE_RUNTIME_GATE_DISPENSES: &str = "kv-reserve@";

pub fn recorded_planned_bytes_outside_the_gemma4_kv_reserve() -> u64 {
    boot_plans()
        .lock()
        .unwrap()
        .iter()
        .flat_map(|(_, p)| {
            p.items.iter().filter(move |i| {
                !(p.family == "gemma4"
                    && i.label
                        .starts_with(GEMMA4_KV_RESERVE_LABEL_PREFIX_IS_WHAT_THE_RUNTIME_GATE_DISPENSES))
            })
        })
        .map(|i| i.bytes)
        .sum()
}

fn recorded_unmeasured_items() -> Vec<String> {
    boot_plans()
        .lock()
        .unwrap()
        .iter()
        .flat_map(|(_, plan)| plan.items.iter())
        .filter(|item| item.unmeasured)
        .map(|item| item.label.clone())
        .collect()
}

fn reconcile_verdict(
    planned: u64,
    measured: u64,
    tolerance: u64,
    incomplete: bool,
) -> &'static str {
    if measured > planned.saturating_add(tolerance) {
        "EXCEEDS PLAN"
    } else if incomplete {
        "WITHIN DECLARED BYTES; INCOMPLETE ESTIMATE"
    } else {
        "WITHIN PLAN"
    }
}

pub fn process_vram_bytes_via_nvidia_smi() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    let me = std::process::id();
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    for line in text.lines() {
        let mut it = line.split(',').map(|f| f.trim());
        let (Some(pid), Some(mib)) = (it.next(), it.next()) else {
            continue;
        };
        if pid.parse::<u32>().ok() == Some(me) {
            return mib.parse::<u64>().ok().map(|m| m * 1024 * 1024);
        }
    }
    None
}

pub fn reconcile_boot_plans_against_nvidia_smi_once() {
    let planned = recorded_planned_total_bytes();
    if planned == 0 {
        return;
    }
    let Some(measured) = process_vram_bytes_via_nvidia_smi() else {
        tracing::warn!(
            planned_gib = format!("{:.2}", planned as f64 / GIB),
            "vram plan reconcile skipped: nvidia-smi did not report this process"
        );
        return;
    };
    let tol = (env_gib(
        VRAM_PLAN_TOLERANCE_ENV,
        RECONCILE_TOLERANCE_GIB_DEFAULT_COVERS_ALLOCATOR_ROUNDING_AND_GRAPH_POOLS,
    ) * GIB) as u64;
    let within = measured <= planned.saturating_add(tol);
    let unmeasured = recorded_unmeasured_items();
    let line = format!(
        "vram plan reconcile: planned {:.2} GiB, nvidia-smi measured {:.2} GiB, tolerance {:.2} GiB -> {}; unmeasured={:?}; sample after synchronous startup, asynchronous warmup/later growth not bounded",
        planned as f64 / GIB,
        measured as f64 / GIB,
        tol as f64 / GIB,
        reconcile_verdict(planned, measured, tol, !unmeasured.is_empty()),
        unmeasured
    );
    eprintln!("{line}");
    if within {
        if unmeasured.is_empty() {
            tracing::info!("{line}");
        } else {
            tracing::warn!("{line}");
        }
        return;
    }
    tracing::error!(
        "{line}; measured residency exceeds declared estimates/reservations; component attribution remains unresolved"
    );
    if std::env::var(VRAM_PLAN_STRICT_ENV_1_MAKES_THE_POST_BOOT_RECONCILE_FATAL).as_deref()
        == Ok("1")
    {
        panic!("{line}; NV_VRAM_PLAN_STRICT=1 makes this fatal");
    }
}

pub const VERIFY_LMHEAD_INT8_SHIPS_ON_SIGNED_OFF_2026_09_02_V2_NORMX_MK_PLUS_GEMV_I8_PRENORMED_MK_AT_K_GE_4096_48_OF_48_TEACHER_FORCED_AGREEMENT_AND_ZERO_RESTORES_THE_BF16_VERIFY_HEAD: bool =
    true;

pub fn verify_lmhead_int8_from(raw: Option<&str>) -> bool {
    match raw {
        Some("0") => false,
        Some(_) => true,
        None => VERIFY_LMHEAD_INT8_SHIPS_ON_SIGNED_OFF_2026_09_02_V2_NORMX_MK_PLUS_GEMV_I8_PRENORMED_MK_AT_K_GE_4096_48_OF_48_TEACHER_FORCED_AGREEMENT_AND_ZERO_RESTORES_THE_BF16_VERIFY_HEAD,
    }
}

#[cfg(feature = "cuda")]
pub use family::*;

#[cfg(feature = "cuda")]
mod family {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum PlanFamily {
        Gemma4,
        Qwen35Dense,
        Qwen35Moe { grouped_expert_copy: bool },
        Unmodeled(&'static str),
        Laguna,
        Qwen4Exp,
    }

    pub const QWEN4_EXP_PLAN_IS_ADVISORY_UNTIL_A_LEDGER_MEASURES_ITS_LOADER_TRANSIENTS: bool = false;

    pub const LAGUNA_PLAN_IS_ADVISORY_ONLY_HANDOFF_PREFLIGHTS_BOTH_CACHES_AGAINST_LIVE_FREE_MEMORY: bool = false;

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct LagunaSpecServeShape {
        pub full_layers: u64,
        pub sliding_layers: u64,
        pub moe_layers: u64,
        pub kv_heads: u64,
        pub head_dim: u64,
        pub max_query_heads: u64,
        pub window: u64,
        pub chunk: u64,
        pub capacity: u64,
        pub num_experts: u64,
        pub hidden: u64,
        pub moe_intermediate: u64,
    }

    impl LagunaSpecServeShape {
        pub fn from_config_json(raw_cfg: &str, capacity: u64, chunk: u64) -> anyhow::Result<Self> {
            let v: serde_json::Value = serde_json::from_str(raw_cfg)?;
            let field = |name: &str| -> anyhow::Result<u64> {
                v.get(name)
                    .and_then(|x| x.as_u64())
                    .ok_or_else(|| anyhow::anyhow!("laguna config lacks {name}"))
            };
            let layer_types: Vec<&str> = v
                .get("layer_types")
                .and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(|t| t.as_str()).collect())
                .unwrap_or_default();
            let layers = field("num_hidden_layers")?;
            anyhow::ensure!(
                layer_types.len() as u64 == layers,
                "laguna layer_types must list every layer"
            );
            let sliding_layers = layer_types
                .iter()
                .filter(|t| **t == "sliding_attention")
                .count() as u64;
            anyhow::ensure!(
                field("num_key_value_heads")? > 0,
                "Laguna ledger requires positive KV heads"
            );
            let model_config = nv_models::laguna::LagunaConfig::from_hf_json_str(raw_cfg)?;
            let moe_layers = (0..model_config.num_hidden_layers)
                .filter(|idx| model_config.is_moe_layer(*idx))
                .count() as u64;
            let max_query_heads = v
                .get("num_attention_heads_per_layer")
                .and_then(|x| x.as_array())
                .and_then(|a| a.iter().filter_map(|h| h.as_u64()).max())
                .unwrap_or(0)
                .max(field("num_attention_heads")?);
            anyhow::ensure!(
                capacity > 0
                    && capacity <= 262144
                    && matches!(chunk, 256 | 512 | 1024 | 2048 | 4096 | 8192),
                "Laguna ledger requires capacity 1..=262144 and supported chunk"
            );
            anyhow::ensure!(
                layer_types
                    .iter()
                    .all(|t| matches!(*t, "full_attention" | "sliding_attention")),
                "Laguna ledger requires valid layer types"
            );
            Ok(Self {
                full_layers: layers - sliding_layers,
                sliding_layers,
                moe_layers,
                kv_heads: field("num_key_value_heads")?,
                head_dim: field("head_dim")?,
                max_query_heads,
                window: field("sliding_window")?.max(1),
                chunk,
                capacity,
                num_experts: field("num_experts")?,
                hidden: field("hidden_size")?,
                moe_intermediate: field("moe_intermediate_size")?,
            })
        }

        pub fn sliding_slots(&self) -> anyhow::Result<u64> {
            Ok(self.capacity.min(laguna_sum(&[self.window, self.chunk])?))
        }

        pub fn bf16_cache_bytes(&self) -> anyhow::Result<u64> {
            laguna_product(&[
                4,
                self.kv_heads,
                self.head_dim,
                laguna_sum(&[
                    laguna_product(&[self.full_layers, self.capacity])?,
                    laguna_product(&[self.sliding_layers, self.sliding_slots()?])?,
                ])?,
            ])
        }

        pub fn fp8_cache_bytes(&self) -> anyhow::Result<u64> {
            let full = laguna_product(&[
                self.full_layers,
                self.capacity,
                laguna_sum(&[
                    laguna_product(&[2, self.kv_heads, self.head_dim])?,
                    laguna_product(&[8, self.kv_heads])?,
                ])?,
            ])?;
            let sliding = laguna_product(&[
                4,
                self.sliding_layers,
                self.sliding_slots()?,
                self.kv_heads,
                self.head_dim,
            ])?;
            let scores_scratch_upper_bound =
                laguna_product(&[4, self.max_query_heads, self.capacity])?;
            laguna_sum(&[full, sliding, scores_scratch_upper_bound])
        }

        pub fn grouped_expert_copy_bytes(&self, fold_shared: bool) -> anyhow::Result<u64> {
            let elements = laguna_product(&[self.hidden, self.moe_intermediate])?;
            anyhow::ensure!(
                self.hidden.is_multiple_of(16) && self.moe_intermediate.is_multiple_of(16),
                "Laguna packed expert geometry is not block16 aligned"
            );
            let scale_bytes = |rows: u64, cols: u64| -> anyhow::Result<u64> {
                laguna_product(&[
                    laguna_sum(&[rows, 127])? / 128,
                    128,
                    laguna_sum(&[cols / 16, 3])? / 4,
                    4,
                ])
            };
            let gate = laguna_sum(&[
                elements / 2,
                scale_bytes(self.moe_intermediate, self.hidden)?,
            ])?;
            let down = laguna_sum(&[
                elements / 2,
                scale_bytes(self.hidden, self.moe_intermediate)?,
            ])?;
            let per_expert = laguna_sum(&[laguna_product(&[2, gate])?, down, 5 * 4])?;
            laguna_product(&[
                self.moe_layers,
                laguna_sum(&[self.num_experts, u64::from(fold_shared)])?,
                per_expert,
            ])
        }
    }

    fn laguna_sum(values: &[u64]) -> anyhow::Result<u64> {
        values
            .iter()
            .try_fold(0u64, |a, b| a.checked_add(*b))
            .ok_or_else(|| anyhow::anyhow!("Laguna ledger addition overflow"))
    }

    fn laguna_product(values: &[u64]) -> anyhow::Result<u64> {
        values
            .iter()
            .try_fold(1u64, |a, b| a.checked_mul(*b))
            .ok_or_else(|| anyhow::anyhow!("Laguna ledger multiplication overflow"))
    }

    pub fn laguna_draft_ctx_ring_bytes(draft_cfg: &str) -> anyhow::Result<u64> {
        let v: serde_json::Value = serde_json::from_str(draft_cfg)?;
        let n = |name: &str| {
            v.get(name)
                .and_then(|x| x.as_u64())
                .ok_or_else(|| anyhow::anyhow!("Laguna draft config lacks {name}"))
        };
        let block = v
            .get("dflash_config")
            .and_then(|c| c.get("block_size"))
            .and_then(|n| n.as_u64())
            .ok_or_else(|| anyhow::anyhow!("Laguna draft config lacks block_size"))?;
        let window = n("sliding_window")?.max(1);
        let row = laguna_product(&[n("num_key_value_heads")?, n("head_dim")?, 2])?;
        laguna_sum(&[
            laguna_product(&[
                n("num_hidden_layers")?,
                2,
                laguna_sum(&[window, block])?,
                row,
            ])?,
            laguna_product(&[window, row])?,
        ])
    }

    fn laguna_env_usize(name: &str) -> Option<usize> {
        std::env::var(name)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|n| *n > 0)
    }

    fn laguna_spec_capacity(raw_cfg: &str) -> anyhow::Result<u64> {
        let v: serde_json::Value = serde_json::from_str(raw_cfg)?;
        let model_max = v
            .get("max_position_embeddings")
            .and_then(|x| x.as_u64())
            .unwrap_or(0) as usize;
        let bound = kv_env_binding(per_engine_kv_env_for_family("laguna")).map(|(n, _)| n);
        let kv_max = crate::oapi::chat_engine::kv_max_seq_len_resolved(bound, model_max);
        Ok(crate::oapi::chat_engine::spec_serve_max_seq(
            laguna_env_usize("NV_LAGUNA_SPEC_MAX_SEQ"),
            kv_max,
        ) as u64)
    }

    fn plan_laguna(
        model_dir: &Path,
        raw_cfg: &str,
        budget_bytes: u64,
        boot_shape: BootPlanShape,
    ) -> anyhow::Result<BootPlan> {
        let batch_selected = std::env::var("NV_LAGUNA_BATCH")
            .map(|v| !v.trim().is_empty() && v.trim() != "0")
            .unwrap_or(false);
        let spec_b2 = nv_models::laguna_serve::batch_m2_spec_enabled_from_env();
        if spec_b2 {
            nv_models::laguna_serve::validate_m2_spec_selection()?;
        }
        let prepared_b2 = spec_b2 || nv_models::laguna_serve::batch_m2_graph_enabled_from_env();
        if prepared_b2 {
            nv_models::laguna_serve::validate_m2_graph_selection()?;
        }
        if boot_shape != BootPlanShape::Full
            || (batch_selected && !prepared_b2)
            || nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_RING_ATTN")
            || nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_FP8_BF16_FULL")
        {
            let mut boot = plan_unmodeled(model_dir, "laguna", budget_bytes)?;
            boot.plan.push("laguna-topology-unmodeled", 0,
                "Laguna retained-cache ledger covers only the full solo worker with compacting CUDA caches and resident grouped experts; this selected topology needs its own plan");
            return Ok(boot);
        }
        let mut plan = new_process_plan(
            model_dir,
            "laguna",
            budget_bytes,
            LAGUNA_PLAN_IS_ADVISORY_ONLY_HANDOFF_PREFLIGHTS_BOTH_CACHES_AGAINST_LIVE_FREE_MEMORY,
        );
        match checkpoint_tensor_bytes(model_dir) {
            Ok(t) => plan.push("weights-resident", sum_where(&t, |_| true), format!("source safetensors bytes, {} tensors; transformed residents/temporary loader allocations remain within separately estimated slack, not exact allocation tracking", t.len())),
            Err(e) => plan.push("weights-resident", 0, format!("header parse failed: {e:#}")),
        }
        let chunk = laguna_env_usize("NV_LAGUNA_PREFILL_CHUNK").unwrap_or(1024) as u64;
        let capacity = laguna_spec_capacity(raw_cfg)?;
        let shape = LagunaSpecServeShape::from_config_json(raw_cfg, capacity, chunk)?;
        plan.push("experts-grouped-copy", shape.grouped_expert_copy_bytes(true)?,
            format!("NVFP4 grouped-copy estimate: {} MoE layers x ({} routed + 1 shared) experts; packed bytes, padded block16 scales and five f32 coefficients per expert; retained per-expert weights charged separately", shape.moe_layers, shape.num_experts));
        let handoff = std::env::var("NV_LAGUNA_PREFILL_BF16_HANDOFF").as_deref() == Ok("1");
        let fp8 = handoff || nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_FP8_KV");
        let ring = format!(
            "cap {} full x {} layers, ring {} x {} sliding layers, chunk {}",
            shape.capacity,
            shape.full_layers,
            shape.sliding_slots()?,
            shape.sliding_layers,
            shape.chunk
        );
        if prepared_b2 {
            plan.push("kv-destinations-e4m3-b2", laguna_product(&[2, shape.fp8_cache_bytes()?])?,
                format!("two independent FixedOne full-capacity caches; e4m3 K/V + f32 scales, bf16 sliding rings and legacy f32 scores scratch; shared BF16 source and draft separately charged when selected; {ring}"));
            let prefill = if handoff {
                "one shared BF16 prefill runner"
            } else {
                "two direct-FP8 prefill runners"
            };
            plan.push("b2-graph-pools", 0, format!("retained physical masks 01/10/11; M1/M2 plain or M1/M2/M3/M6 fixed-K2 graphs, per-lane M1/M3 XQA, {prefill} and grouped workspaces are covered only by estimated graph/transient slack, not exact allocation tracking"));
        } else if fp8 {
            plan.push("kv-destination-e4m3", shape.fp8_cache_bytes()?, format!("LagunaKvCacheFp8: e4m3 K/V + f32 scales on full layers, bf16 sliding ring, f32 scores scratch; {ring}"));
        } else {
            plan.push(
                "kv-destination-bf16",
                shape.bf16_cache_bytes()?,
                format!("LagunaKvCache bf16; {ring}"),
            );
        }
        if handoff {
            plan.push("kv-handoff-source-bf16", shape.bf16_cache_bytes()?, format!("NV_LAGUNA_PREFILL_BF16_HANDOFF=1 retains one full-capacity bf16 prefill source, shared by the two destinations in B2; {ring}"));
        }
        let draft = std::env::var("NV_LAGUNA_SERVE_DRAFT")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(true);
        let draft_dir = crate::oapi::chat_engine::laguna_dflash_dir();
        if let (true, Some(dir)) = (draft, draft_dir) {
            match checkpoint_tensor_bytes(&dir) {
                Ok(t) => plan.push("draft-weights", sum_where(&t, |_| true), format!("DFlash source checkpoint bytes (conservative source-storage accounting, not proof of resident packed format), {} tensors, {}", t.len(), dir.display())),
                Err(e) => plan.push("draft-weights", 0, format!("draft header parse failed: {e:#}")),
            }
            let draft_cfg = std::fs::read_to_string(dir.join("config.json"))?;
            plan.push("draft-ctx-ring", laguna_product(&[if spec_b2 { 2 } else { 1 }, laguna_draft_ctx_ring_bytes(&draft_cfg)?])?,
                "DflashCtxRing bf16 K/V (window + block) per draft layer plus window scratch, per physical speculative lane");
            if spec_b2 {
                plan.push("b2-spec-context-commit-staging", laguna_sum(&[laguna_product(&[2, 5, 3, shape.hidden, 2])?, 2 * 3 * 4])?, "five retained three-row BF16 auxiliary inputs and I32 positions per physical lane; commit graph pools remain inside the graph reserve estimate");
                plan.push("b2-spec-outputs-and-sliding", laguna_sum(&[laguna_product(&[6, shape.hidden, 5, 2])?, laguna_product(&[8, serde_json::from_str::<serde_json::Value>(raw_cfg)?.get("vocab_size").and_then(|v| v.as_u64()).ok_or_else(|| anyhow::anyhow!("Laguna config lacks vocab_size"))?, 4])?, 2 * 798_816])?, "six F32 verification logit rows plus two retained plain-owner rows, five BF16 auxiliary tap blocks and two M3 sliding scratch/fan-in buffers; XQA scratch and graph pools remain estimated");
            }
        }
        common_slack(&mut plan);
        Ok(BootPlan {
            plan,
            kv_max_seq_len: 0,
            batch_bucket_sizes: None,
            batch_lane_kv: None,
            gemma4_batch_pool: None,
        })
    }

    #[derive(Clone, Debug)]
    pub struct BootPlan {
        pub plan: VramPlan,
        pub kv_max_seq_len: usize,
        pub batch_bucket_sizes: Option<Vec<usize>>,
        pub batch_lane_kv: Option<usize>,
        pub gemma4_batch_pool: Option<Gemma4BatchPool>,
    }

    fn kv_env_binding(per_engine_env: Option<&'static str>) -> Option<(usize, &'static str)> {
        kv_binding_from(
            per_engine_env,
            per_engine_env
                .and_then(|e| std::env::var(e).ok())
                .as_deref(),
            std::env::var(KV_MAX_SEQ_LEN_GLOBAL_ENV_THE_DEFAULT_EVERY_ENGINE_INHERITS)
                .ok()
                .as_deref(),
        )
    }

    fn common_slack(plan: &mut VramPlan) {
        plan.push(
            "slack-cuda-context",
            (CUDA_CONTEXT_AND_CUBLAS_WORKSPACE_SLACK_GIB * GIB) as u64,
            "CUDA_CONTEXT_AND_CUBLAS_WORKSPACE_SLACK_GIB",
        );
        plan.push(
            "slack-graph-transient",
            (FAMILY_GRAPH_CAPTURE_AND_TRANSIENT_SLACK_GIB * GIB) as u64,
            "FAMILY_GRAPH_CAPTURE_AND_TRANSIENT_SLACK_GIB",
        );
    }

    fn new_process_plan(
        model_dir: &Path,
        family: &'static str,
        budget_bytes: u64,
        enforced: bool,
    ) -> VramPlan {
        let mut plan = VramPlan::new(model_dir, family, budget_bytes, enforced);
        push_prior_or_aux_process_items(
            &mut plan,
            recorded_planned_total_bytes(),
            &AuxEnv::from_env(),
        );
        plan.prior_kv_knobs = recorded_kv_knobs();
        plan
    }

    fn drafter_dir_config_row_elems(dir: &Path) -> Option<(usize, usize)> {
        let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
        let v: serde_json::Value = serde_json::from_str(&text).ok()?;
        let top = v
            .get("text_config")
            .or_else(|| v.get("transformer_layer_config"))
            .unwrap_or(&v);
        let nkv = top.get("num_key_value_heads")?.as_u64()? as usize;
        let hd = top.get("head_dim")?.as_u64()? as usize;
        let layers = top
            .get("num_hidden_layers")
            .and_then(|x| x.as_u64())
            .unwrap_or(1) as usize;
        Some((nkv * hd, layers))
    }

    struct DrafterPlan {
        weights_bytes: u64,
        eagle3_row_elems: usize,
        dflash_row_elems: usize,
        dir_shown: String,
    }

    fn gemma4_drafter_plan() -> Option<DrafterPlan> {
        let no_spec =
            crate::oapi::chat_engine::nv_no_spec(std::env::var("NV_NO_SPEC").ok().as_deref());
        if no_spec {
            return None;
        }
        let kind =
            crate::oapi::chat_engine::nv_drafter_kind(std::env::var("NV_DRAFTER").ok().as_deref());
        let want_eagle3 = kind != "dflash";
        let want_dflash = matches!(kind, "dflash" | "auto" | "route");
        let mut weights = 0u64;
        let mut e3 = 0usize;
        let mut df = 0usize;
        let mut dirs = Vec::new();
        if want_eagle3 {
            if let Some(dir) = std::env::var_os("NV_EAGLE3_DRAFT_DIR").map(PathBuf::from) {
                if dir.is_dir() {
                    if let Ok(t) = checkpoint_tensor_bytes(&dir) {
                        weights += sum_where(&t, |_| true);
                        if let Some((row, _)) = drafter_dir_config_row_elems(&dir) {
                            e3 = row;
                        }
                        dirs.push(dir.display().to_string());
                    }
                }
            }
        }
        if want_dflash {
            if let Some(dir) = std::env::var_os("NV_DFLASH_DRAFT_DIR").map(PathBuf::from) {
                if dir.is_dir() {
                    if let Ok(t) = checkpoint_tensor_bytes(&dir) {
                        weights += sum_where(&t, |_| true);
                        if let Some((row, layers)) = drafter_dir_config_row_elems(&dir) {
                            df = row * layers.max(1);
                        }
                        dirs.push(dir.display().to_string());
                    }
                }
            }
        }
        if dirs.is_empty() {
            return None;
        }
        Some(DrafterPlan {
            weights_bytes: weights,
            eagle3_row_elems: e3,
            dflash_row_elems: df,
            dir_shown: dirs.join(","),
        })
    }

    fn gemma4_kv_bytes_at(
        config: &nv_models::gemma4::Gemma4Config,
        kv_max: usize,
        charge: crate::oapi::chat_engine::DrafterKvCharge,
    ) -> u64 {
        let rings = nv_models::gemma4::kv_ring_enabled();
        let (rows, row_elems) = charge.rows_elems(kv_max);
        let b = nv_models::gemma4::kv_budget_capped(config, kv_max, true, rings, row_elems, rows);
        (b.worst_total() + nv_models::gemma4::gqa512_verify_scratch_bytes(config)) as u64
    }

    fn gemma4_attn_proj_planned_nvfp4() -> bool {
        matches!(
            std::env::var("NV_ATTN_PROJ_QUANT").ok().as_deref(),
            Some("1")
                | Some("on")
                | Some("true")
                | Some("yes")
                | Some("all")
                | Some("nvfp4")
                | Some("fp4")
                | Some("qkv+o")
        )
    }

    fn gemma4_i8_lmhead_planned() -> bool {
        let raw = std::env::var("NV_VERIFY_LMHEAD_INT8").ok();
        let planned = super::verify_lmhead_int8_from(raw.as_deref());
        assert_eq!(
            planned,
            raw.as_deref() != Some("0"),
            "verify_lmhead_int8_from must stay in lockstep with the raw \
             NV_VERIFY_LMHEAD_INT8 != 0 reads in nv-models gemma4 load_model, or the plan \
             prices a head the engine never builds"
        );
        planned
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum BootPlanShape {
        Full,
        AdapterServingLeanKeepsTheI8DecodeHeadNeverBuildsDrafters,
    }

    pub const GEMMA4_BATCH_ENGINE_ENV_BOOTS_THE_PAGED_POOL_THE_PLAN_SIZES: &str = "NV_BATCH_ENGINE";
    pub const GEMMA4_BATCH_BLOCK_SIZE_IS_THE_SCHEDULER_BLOCK: usize = 16;
    pub const GEMMA4_BATCH_KV_FRACTION_DEFAULT_OF_THE_BUDGET_THE_FIXED_ITEMS_LEAVE: f64 = 0.25;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Gemma4BatchPool {
        pub max_batch_size: usize,
        pub num_blocks: usize,
        pub pool_blocks: usize,
        pub lanes: usize,
        pub graph: bool,
        pub bytes: u64,
    }

    pub fn gemma4_batch_pool_at(
        config: &nv_models::gemma4::Gemma4Config,
        kv_max: usize,
        budget_left_after_fixed_items: u64,
    ) -> Gemma4BatchPool {
        let env = |key: &str| std::env::var(key).ok();
        let block = GEMMA4_BATCH_BLOCK_SIZE_IS_THE_SCHEDULER_BLOCK;
        let max_batch_size = env("NV_BATCH_MAX_SEQS")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(8)
            .max(1);
        let hybrid = nv_models::gemma4::kv_ring_enabled();
        let lanes = if hybrid { max_batch_size } else { 0 };
        let graph = !hybrid && env("NV_BATCH_GRAPH").is_some_and(|v| v != "0");
        let scratch_blocks = if graph {
            nv_models::gemma4_batch_graph::BucketPlan::from_env().max_bucket()
        } else {
            0
        };
        let pool_bytes = |full_blocks: usize| {
            let cfg = if lanes > 0 {
                nv_models::paged_fp8::PagedPoolConfig::from_gemma4_hybrid(
                    config,
                    full_blocks,
                    block,
                    lanes,
                )
            } else {
                nv_models::paged_fp8::PagedPoolConfig::from_gemma4(config, full_blocks, block)
            };
            cfg.pool_bytes() as u64
        };
        let constant = pool_bytes(scratch_blocks);
        let per_block = (pool_bytes(scratch_blocks + 1) - constant).max(1);
        let want = (max_batch_size * kv_max.div_ceil(block)).max(block);
        let num_blocks = match env("NV_BATCH_KV_BLOCKS").and_then(|s| s.parse::<usize>().ok()) {
            Some(n) => n.max(block),
            None => {
                let fraction = env("NV_BATCH_KV_FRACTION")
                    .and_then(|s| s.parse::<f64>().ok())
                    .filter(|f| *f > 0.0 && *f < 1.0)
                    .unwrap_or(GEMMA4_BATCH_KV_FRACTION_DEFAULT_OF_THE_BUDGET_THE_FIXED_ITEMS_LEAVE);
                let share = ((budget_left_after_fixed_items as f64 * fraction) as u64)
                    .saturating_sub(constant);
                ((share / per_block) as usize).max(block).min(want)
            }
        };
        let pool_blocks = num_blocks + scratch_blocks;
        Gemma4BatchPool {
            max_batch_size,
            num_blocks,
            pool_blocks,
            lanes,
            graph,
            bytes: pool_bytes(pool_blocks),
        }
    }

    pub fn plan_gemma4(
        model_dir: &Path,
        raw_cfg: &str,
        budget_bytes: u64,
    ) -> anyhow::Result<BootPlan> {
        plan_gemma4_shaped(model_dir, raw_cfg, budget_bytes, BootPlanShape::Full)
    }

    pub fn plan_gemma4_shaped(
        model_dir: &Path,
        raw_cfg: &str,
        budget_bytes: u64,
        shape: BootPlanShape,
    ) -> anyhow::Result<BootPlan> {
        let config = nv_models::gemma4::Gemma4Config::from_hf_json_str(raw_cfg)?;
        let tensors = checkpoint_tensor_bytes(model_dir)?;
        let weights = sum_where(&tensors, |_| true);
        let mut plan = new_process_plan(model_dir, "gemma4", budget_bytes, true);
        let apq = gemma4_attn_proj_planned_nvfp4();
        if apq {
            let attn_proj = sum_where(&tensors, |n| {
                n.contains("self_attn") && n.contains("_proj.weight")
            });
            let attn_nvfp4 = attn_proj
                * NVFP4_RESIDENT_NUMERATOR_9_BYTES_PER_32_BF16_BYTES_4BIT_PLUS_FP8_SCALE_PER_16
                / NVFP4_RESIDENT_DENOMINATOR_32;
            plan.push(
                "weights-resident",
                weights - attn_proj + attn_nvfp4,
                format!(
                    "safetensors headers, {} tensors; NV_ATTN_PROJ_QUANT requants {:.2} GiB of \
                     self_attn *_proj.weight to nvfp4 resident {:.2} GiB (in-place during load, \
                     no staging spike -- VRAM-LEDGER g4-apq arm)",
                    tensors.len(),
                    attn_proj as f64 / GIB,
                    attn_nvfp4 as f64 / GIB
                ),
            );
        } else {
            plan.push(
                "weights-resident",
                weights,
                format!("safetensors headers, {} tensors", tensors.len()),
            );
        }
        if gemma4_i8_lmhead_planned() {
            plan.push(
                "i8-lmhead-copy",
                (config.vocab_size * config.hidden_size) as u64,
                "vocab x hidden x 1B verify head built unless NV_VERIFY_LMHEAD_INT8=0 \
                 (VRAM-LEDGER measured -1344 MiB with it off, exact match); the adapter-serving \
                 lean shape retains the copy so the armed NV_G4_LMHEAD_I8 m=1 decode epilogue \
                 reads i8 like the base engine instead of falling back to the 2x-wider bf16 rows",
            );
        }
        let overhead_mib = if apq {
            GEMMA4_STEADY_OVERHEAD_MIB_UNDER_ATTN_NVFP4_MEASURED_LEDGER_28588_ARM
        } else {
            GEMMA4_STEADY_OVERHEAD_MIB_MEASURED_LEDGER_44684_MINUS_HEADERS_MINUS_I8_HEAD
        };
        plan.push(
            "capture-pools-workspaces-ctx-measured",
            overhead_mib * MIB,
            "VRAM-LEDGER gemma4 measured boot steady minus header-derived residents: \
             env-invariant capture pools + workspaces + mm graph buckets + CUDA context \
             (identical at kv 32768 and 131072, fp8 and bf16 flags)",
        );
        let drafter = match shape {
            BootPlanShape::Full => gemma4_drafter_plan(),
            BootPlanShape::AdapterServingLeanKeepsTheI8DecodeHeadNeverBuildsDrafters => None,
        };
        let charge = crate::oapi::chat_engine::DrafterKvCharge::from_env(
            drafter.as_ref().map(|d| d.eagle3_row_elems).unwrap_or(0),
            drafter.as_ref().map(|d| d.dflash_row_elems).unwrap_or(0),
        );
        if let Some(d) = &drafter {
            plan.push(
                "drafter-weights",
                d.weights_bytes,
                format!(
                    "safetensors headers of {} (counts drafter embed even when later shared with the target and freed)",
                    d.dir_shown
                ),
            );
            plan.drafter_off_knob = Some((d.weights_bytes, "set NV_NO_SPEC=1"));
        }
        let kv_slots = kv_concurrent_cap();
        let fixed = plan.planned_total_bytes();
        let batch_engine = shape == BootPlanShape::Full
            && std::env::var(GEMMA4_BATCH_ENGINE_ENV_BOOTS_THE_PAGED_POOL_THE_PLAN_SIZES).is_ok();
        let batch_pool_at = |kv: usize| {
            batch_engine
                .then(|| gemma4_batch_pool_at(&config, kv, budget_bytes.saturating_sub(fixed)))
        };
        let cap = config.max_position_embeddings.max(1);
        let per_engine_env = per_engine_kv_env_for_family("gemma4");
        let binding = kv_env_binding(per_engine_env);
        let kv_knob_env = binding.map(|(_, env)| env).unwrap_or_else(|| {
            per_engine_env.unwrap_or(KV_MAX_SEQ_LEN_GLOBAL_ENV_THE_DEFAULT_EVERY_ENGINE_INHERITS)
        });
        let (kv_max, kv_env_shown) = match binding {
            Some((n, env)) => (n.min(cap), format!("bound by {env}")),
            None => (
                solve_max_kv_tokens_monotone(
                    |kv| {
                        fixed
                            + batch_pool_at(kv).map_or(0, |p| p.bytes)
                            + gemma4_kv_bytes_at(&config, kv, charge) * kv_slots as u64
                            <= budget_bytes
                    },
                    cap,
                )
                .unwrap_or(KV_SOLVE_FLOOR_TOKENS_BELOW_WHICH_SERVING_IS_NOT_WORTH_BOOTING),
                format!(
                    "SOLVED to the largest kv the remaining budget admits in engine boot order \
                     ({}); pin with {kv_knob_env}",
                    KV_SOLVE_ORDER_IS_ENGINE_BOOT_ORDER_NV_CHAT_MODEL_DIRS_LEFT_TO_RIGHT_Q38_FIRST_ON_THE_GPU_HOST_SO_EACH_UNSET_ENGINE_TAKES_THE_LARGEST_KV_THE_REMAINING_BUDGET_ADMITS_AND_LATER_ENGINES_SOLVE_ONLY_WHAT_IT_LEFT
                ),
            ),
        };
        let batch_pool = batch_pool_at(kv_max);
        if let Some(p) = batch_pool {
            plan.push(
                format!("batch-paged-kv-pool@{}blocks", p.pool_blocks),
                p.bytes,
                format!(
                    "PREPAID at boot by build_gemma4_batch_engine ({}): {} blocks x {} slots, \
                     NV_BATCH_MAX_SEQS={} sequences at kv_max capped at NV_BATCH_KV_FRACTION of \
                     the budget the fixed items leave unless NV_BATCH_KV_BLOCKS pins it, plus {} \
                     graph scratch blocks and {} hybrid ring lanes; PagedPoolConfig::pool_bytes, \
                     the device pool's upper bound",
                    GEMMA4_BATCH_ENGINE_ENV_BOOTS_THE_PAGED_POOL_THE_PLAN_SIZES,
                    p.num_blocks,
                    GEMMA4_BATCH_BLOCK_SIZE_IS_THE_SCHEDULER_BLOCK,
                    p.max_batch_size,
                    p.pool_blocks - p.num_blocks,
                    p.lanes
                ),
            );
        }
        let kv_bytes = gemma4_kv_bytes_at(&config, kv_max, charge);
        plan.push(
            format!(
                "{GEMMA4_KV_RESERVE_LABEL_PREFIX_IS_WHAT_THE_RUNTIME_GATE_DISPENSES}kv_max={kv_max}x{kv_slots}req"
            ),
            kv_bytes * kv_slots as u64,
            format!(
                "worst-case per-request KV, LAZY at boot but a request-time liability the plan \
                 prepays x{kv_slots} ({}): gemma4::kv_budget_capped fp8=true rings={} + gqa512 \
                 verify scratch, drafter rows folded in; kv_window() caps every request at this \
                 kv_max by construction; kv_max {kv_env_shown}",
                VRAM_PLAN_KV_CONCURRENT_ENV_RESERVES_WORST_CASE_REQUEST_KV_SLOTS,
                nv_models::gemma4::kv_ring_enabled()
            ),
        );
        plan.kv_knob = Some(KvShrinkKnob {
            env: kv_knob_env,
            kv_max,
            lanes: kv_slots,
            approx_bytes_per_token_per_lane: kv_bytes / kv_max.max(1) as u64,
            prior_engines_bytes_per_token_at_same_kv: recorded_kv_bytes_per_token_at(
                kv_max,
                kv_knob_env,
            ),
            floor: KV_SOLVE_FLOOR_TOKENS_BELOW_WHICH_SERVING_IS_NOT_WORTH_BOOTING,
        });
        Ok(BootPlan {
            plan,
            kv_max_seq_len: kv_max,
            batch_bucket_sizes: None,
            batch_lane_kv: None,
            gemma4_batch_pool: batch_pool,
        })
    }

    struct Qwen35Geometry {
        n_full: usize,
        n_lin: usize,
        bytes_per_token_full_layers: u64,
        lin_state_bytes_per_lane: u64,
        scores_scratch_bytes_per_token_per_lane: u64,
    }

    fn qwen35_geometry(trunk: &nv_models::qwen3_5_moe::Qwen3MoeConfig) -> Qwen35Geometry {
        let n_full = trunk
            .layer_types
            .iter()
            .filter(|t| **t == nv_models::qwen3_5_moe::LayerType::FullAttention)
            .count();
        let n_lin = trunk.layer_types.len() - n_full;
        let nkv = trunk.num_key_value_heads;
        let hd = trunk.head_dim;
        let per_token_per_full_layer = (2 * nkv * hd + 2 * nkv * 4) as u64;
        let conv_dim = 2 * trunk.linear_num_key_heads * trunk.linear_key_head_dim
            + trunk.linear_num_value_heads * trunk.linear_value_head_dim;
        let lin_state_per_layer = (trunk.linear_num_value_heads
            * trunk.linear_key_head_dim
            * trunk.linear_value_head_dim
            * 4
            + conv_dim * trunk.linear_conv_kernel_dim * 4) as u64;
        Qwen35Geometry {
            n_full,
            n_lin,
            bytes_per_token_full_layers: per_token_per_full_layer * n_full as u64,
            lin_state_bytes_per_lane: lin_state_per_layer * n_lin as u64,
            scores_scratch_bytes_per_token_per_lane: (trunk.num_attention_heads * 4) as u64,
        }
    }

    fn qwen35_per_lane_bytes(geo: &Qwen35Geometry, kv_max: usize) -> u64 {
        (geo.bytes_per_token_full_layers + geo.scores_scratch_bytes_per_token_per_lane)
            * kv_max as u64
            + geo.lin_state_bytes_per_lane
    }

    fn qwen36_batch_engine_lane_bytes(geo: &Qwen35Geometry, kv_max: usize) -> u64 {
        Q36_BATCH_LANE_CONST_MIB_ENGINE_BOOT_MEASURED_4P24_GIB_PER_LANE_AT_832_AND_4P32_AT_8384_SO_ENGINE_BUFFERS_PLUS_GDN_STATE_DOMINATE_AND_KV_ADDS_THE_PER_TOKEN_TAIL
            * MIB
            + (geo.bytes_per_token_full_layers + geo.scores_scratch_bytes_per_token_per_lane)
                * kv_max as u64
    }

    pub fn plan_qwen35(
        model_dir: &Path,
        raw_cfg: &str,
        dense: bool,
        grouped_expert_copy: bool,
        budget_bytes: u64,
    ) -> anyhow::Result<BootPlan> {
        let trunk = if dense {
            nv_models::qwen3_5_moe::Qwen3_5DenseConfig::from_hf_json_str(raw_cfg)?.trunk()
        } else {
            nv_models::qwen3_5_moe::Qwen3MoeConfig::from_hf_json_str(raw_cfg)?
        };
        let geo = qwen35_geometry(&trunk);
        let tensors = checkpoint_tensor_bytes(model_dir)?;
        let mtp_selected = nv_specdecode::qwen38_mtp::mtp_drafter_selected_from_env();
        let mtp_override = nv_specdecode::qwen38_mtp::mtp_draft_dir_override_from_env();
        let is_mtp_tensor = |n: &str| n.starts_with(nv_specdecode::qwen38_mtp::MTP_TENSOR_PREFIX);
        let weights = sum_where(&tensors, |n| !is_mtp_tensor(n));
        let family: &'static str = if dense {
            "qwen3.5-dense"
        } else {
            "qwen3.5-moe"
        };
        let mut plan = new_process_plan(model_dir, family, budget_bytes, true);
        plan.push(
            "weights-resident",
            weights,
            format!(
                "safetensors headers, {} tensors, mtp tensors excluded",
                tensors.len()
            ),
        );
        if mtp_selected {
            let mtp_dir = mtp_override
                .clone()
                .unwrap_or_else(|| model_dir.to_path_buf());
            let mtp_file = mtp_dir.join(nv_specdecode::qwen38_mtp::MTP_WEIGHTS_FILE_NAME);
            let mtp_bytes = if mtp_file.is_file() {
                shard_tensor_bytes(&mtp_file)
                    .map(|t| sum_where(&t, |_| true))
                    .unwrap_or(0)
            } else {
                sum_where(&tensors, is_mtp_tensor)
            };
            plan.push(
                "mtp-head-weights",
                mtp_bytes,
                "safetensors headers of the mtp.* tensors (embed and lm_head are shared with the target)",
            );
            if dense {
                plan.push(
                    "mtp-draft-runtime-measured",
                    Q38_MTP_RUNTIME_STEADY_MIB_MEASURED_LEDGER_23435_MINUS_SOLO_21889_MINUS_CKPT_HEAD
                        * MIB,
                    "VRAM-LEDGER q38+mtp arm: fp8/lmhead-nvfp4 draft copies + draft KV + verify \
                     graphs beyond the mtp ckpt bytes; the +2164 MiB quantize-staging boot SPIKE \
                     rides on slack-graph-transient",
                );
            } else {
                plan.push_unmeasured(
                    "mtp-draft-runtime",
                    Q38_MTP_RUNTIME_STEADY_MIB_MEASURED_LEDGER_23435_MINUS_SOLO_21889_MINUS_CKPT_HEAD
                        * MIB,
                    "no VRAM-LEDGER arm measures the qwen3.5-moe MTP session (draft KV + captured \
                     verify graphs), so the q38+mtp arm's measured runtime stands in",
                );
            }
        }
        if grouped_expert_copy {
            let expert_bytes = sum_where(&tensors, |n| n.contains(".experts."));
            plan.push(
                "grouped-moe-expert-copy",
                expert_bytes,
                "safetensors headers of .experts. tensors, duplicated on device by install_grouped_moe",
            );
        }
        common_slack(&mut plan);
        let fixed = plan.planned_total_bytes();
        let batch_on = if dense {
            nv_models::qwen3_5_moe::qwen38_batch::nv_q38_batch_env_opt_in_nv_q38_batch_1_the_serving_loop_routes_batch_xor_spec_per_request_group()
        } else {
            nv_models::qwen3_5_moe::qwen36_batch::nv_q36_batch_env_opt_in_nv_q36_batch_1_the_serving_loop_routes_batch_xor_spec_per_request_group()
        };
        let (batch_kv_env, batch_kv_default, batch_sizes_env, batch_engine_name) = if dense {
            (
                Q38_BATCH_KV_ENV_CAPS_LANE_POOL_DEPTH_SO_PREPAID_BATCH_LANES_STAY_SHALLOW_WHILE_SOLO_REQUESTS_GO_DEEP,
                Q38_BATCH_KV_DEFAULT_CAP_TOKENS_LEDGER_MEASURED_1167_MIB_PER_LANE_AT_THIS_DEPTH,
                "NV_Q38_BATCH_SIZES",
                "Qwen38BatchLanes",
            )
        } else {
            (
                Q36_BATCH_KV_ENV_CAPS_LANE_POOL_DEPTH_SO_PREPAID_BATCH_LANES_STAY_SHALLOW_WHILE_SOLO_REQUESTS_GO_DEEP,
                Q36_BATCH_KV_DEFAULT_CAP_TOKENS_SAME_32768_AS_Q38_BECAUSE_THE_Q36_LANE_COST_IS_DOMINATED_BY_ITS_DEPTH_CONSTANT_NOT_KV,
                "NV_Q36_BATCH_SIZES",
                "Qwen36BatchLanes",
            )
        };
        let requested_sizes: Vec<usize> = if !batch_on {
            vec![1]
        } else if dense {
            nv_models::qwen3_5_moe::qwen38_batch::q38_batch_bucket_plan_env_nv_q38_batch_sizes()
                .sizes()
                .to_vec()
        } else {
            nv_models::qwen3_5_moe::qwen36_batch::q36_batch_bucket_plan_env_nv_q36_batch_sizes()
                .sizes()
                .to_vec()
        };
        let kv_slots = kv_concurrent_cap();
        let lazy_slots = qwen_lazy_kv_slots(
            batch_on,
            mtp_selected,
            crate::oapi::chat_engine::solo_via_lanes_env_opt_in_nv_q38_solo_via_lanes_1_a_lone_request_runs_as_a_one_lane_group(),
            kv_slots,
        );
        let cap = trunk.max_position_embeddings.max(1);
        let per_engine_env = per_engine_kv_env_for_family(family);
        let batch_kv_raw = if batch_on {
            std::env::var(batch_kv_env).ok()
        } else {
            None
        };
        let lane_kv_at = |kv: usize| {
            if !batch_on {
                kv
            } else if dense {
                q38_batch_lane_kv_from(kv, batch_kv_raw.as_deref())
            } else {
                q36_batch_lane_kv_from(kv, batch_kv_raw.as_deref())
            }
        };
        let pool_lane_bytes_at = |kv: usize| {
            if dense {
                qwen35_per_lane_bytes(&geo, kv)
            } else {
                qwen36_batch_engine_lane_bytes(&geo, kv)
            }
        };
        let binding = kv_env_binding(per_engine_env);
        let (kv_max, kept_sizes) = match binding {
            Some((n, _)) => {
                let kv = n.min(cap);
                let remaining = budget_bytes
                    .saturating_sub(fixed)
                    .saturating_sub(qwen35_per_lane_bytes(&geo, kv) * lazy_slots as u64);
                let kept = solve_largest_bucket_prefix_that_fits(
                    &requested_sizes,
                    pool_lane_bytes_at(lane_kv_at(kv)),
                    remaining,
                );
                (kv, kept)
            }
            None => {
                let mut chosen = (
                    KV_SOLVE_FLOOR_TOKENS_BELOW_WHICH_SERVING_IS_NOT_WORTH_BOOTING,
                    Vec::new(),
                );
                for cut in (0..requested_sizes.len()).rev() {
                    let lanes = requested_sizes[cut];
                    let fits = |kv: usize| {
                        fixed
                            + pool_lane_bytes_at(lane_kv_at(kv)) * lanes as u64
                            + qwen35_per_lane_bytes(&geo, kv) * lazy_slots as u64
                            <= budget_bytes
                    };
                    if let Some(kv) = solve_max_kv_tokens_monotone(fits, cap) {
                        chosen = (kv, requested_sizes[..=cut].to_vec());
                        break;
                    }
                }
                chosen
            }
        };
        let kv_knob_env = match binding {
            Some((_, env)) => env,
            None => per_engine_env
                .unwrap_or(KV_MAX_SEQ_LEN_GLOBAL_ENV_THE_DEFAULT_EVERY_ENGINE_INHERITS),
        };
        let kv_env_shown = match binding {
            Some((_, env)) => format!("bound by {env}"),
            None => format!(
                "SOLVED to the largest kv the remaining budget admits in engine boot order \
                 ({}); pin with {kv_knob_env}",
                KV_SOLVE_ORDER_IS_ENGINE_BOOT_ORDER_NV_CHAT_MODEL_DIRS_LEFT_TO_RIGHT_Q38_FIRST_ON_THE_GPU_HOST_SO_EACH_UNSET_ENGINE_TAKES_THE_LARGEST_KV_THE_REMAINING_BUDGET_ADMITS_AND_LATER_ENGINES_SOLVE_ONLY_WHAT_IT_LEFT
            ),
        };
        let kept_sizes = if kept_sizes.is_empty() {
            vec![1]
        } else {
            kept_sizes
        };
        let pool_lanes = kept_sizes.last().copied().unwrap_or(1).max(1);
        let lane_kv = lane_kv_at(kv_max);
        let per_pool_lane = pool_lane_bytes_at(lane_kv);
        let per_req = qwen35_per_lane_bytes(&geo, kv_max);
        if batch_on {
            plan.push(
                format!("kv-pools@kv={lane_kv}x{pool_lanes}lanes"),
                per_pool_lane * pool_lanes as u64,
                format!(
                    "PREPAID at boot by {batch_engine_name} sized from this plan: {} full-attn \
                     layers fp8 kv+scales, {} linear-attn state lanes, scores scratch nh*kv*4{}; \
                     lane depth = {} (default min(kv_max, {})) so shallow batched throughput \
                     coexists with deep solo context",
                    geo.n_full,
                    geo.n_lin,
                    if dense {
                        String::new()
                    } else {
                        format!(
                            ", plus the engine-boot-measured {} MiB/lane constant (engine \
                             buffers + GDN state dominate)",
                            Q36_BATCH_LANE_CONST_MIB_ENGINE_BOOT_MEASURED_4P24_GIB_PER_LANE_AT_832_AND_4P32_AT_8384_SO_ENGINE_BUFFERS_PLUS_GDN_STATE_DOMINATE_AND_KV_ADDS_THE_PER_TOKEN_TAIL
                        )
                    },
                    batch_kv_env,
                    batch_kv_default
                ),
            );
            if lazy_slots > 0 {
                let route = if mtp_selected { "spec" } else { "solo" };
                plan.push(
                    format!("kv-reserve-{route}-route@kv_max={kv_max}x{lazy_slots}req"),
                    per_req * lazy_slots as u64,
                    format!(
                        "singleton requests allocate MTP or eager KV outside the prepaid lanes; \
                         only solo-via-lanes without MTP avoids this reserve; kv_window() caps each \
                         at kv_max by construction; kv_max {kv_env_shown}"
                    ),
                );
            }
        } else {
            plan.push(
                format!("kv-reserve@kv_max={kv_max}x{lazy_slots}req"),
                per_req * lazy_slots as u64,
                format!(
                    "worst-case per-request KV, LAZY at boot but a request-time liability the \
                     plan prepays x{lazy_slots} ({}): {} full-attn layers fp8 kv+scales, {} \
                     linear-attn state lanes, scores scratch nh*kv*4; kv_window() caps every \
                     request at this kv_max by construction; kv_max {kv_env_shown}",
                    VRAM_PLAN_KV_CONCURRENT_ENV_RESERVES_WORST_CASE_REQUEST_KV_SLOTS,
                    geo.n_full,
                    geo.n_lin
                ),
            );
        }
        let kv_scaling_lanes = if batch_on {
            lazy_slots + if lane_kv == kv_max { pool_lanes } else { 0 }
        } else {
            lazy_slots.max(1)
        };
        if kv_scaling_lanes > 0 {
            plan.kv_knob = Some(KvShrinkKnob {
                env: kv_knob_env,
                kv_max,
                lanes: kv_scaling_lanes,
                approx_bytes_per_token_per_lane: geo.bytes_per_token_full_layers
                    + geo.scores_scratch_bytes_per_token_per_lane,
                prior_engines_bytes_per_token_at_same_kv: recorded_kv_bytes_per_token_at(
                    kv_max,
                    kv_knob_env,
                ),
                floor: KV_SOLVE_FLOOR_TOKENS_BELOW_WHICH_SERVING_IS_NOT_WORTH_BOOTING,
            });
        }
        if batch_on {
            plan.bucket_knob = Some(BucketShrinkKnob {
                env: batch_sizes_env,
                sizes: kept_sizes.clone(),
                bytes_per_lane: per_pool_lane,
            });
            if kept_sizes.len() < requested_sizes.len() {
                tracing::warn!(
                    requested = ?requested_sizes,
                    kept = ?kept_sizes,
                    "vram plan shrank {batch_sizes_env} to the largest prefix that fits the budget"
                );
            }
        }
        Ok(BootPlan {
            plan,
            kv_max_seq_len: kv_max,
            batch_bucket_sizes: batch_on.then_some(kept_sizes),
            batch_lane_kv: batch_on.then_some(lane_kv),
            gemma4_batch_pool: None,
        })
    }

    #[test]
    fn qwen_plain_batch_full_cap_reserves_the_singleton_before_solving() {
        let trunk = nv_models::qwen3_5_moe::Qwen3_5DenseConfig::from_hf_json_str(include_str!(
            "../../crates/nv-models/tests/qwen3_8_27b_config.json"
        ))
        .unwrap()
        .trunk();
        let geo = qwen35_geometry(&trunk);
        let cap = 262144;
        let per_lane = qwen35_per_lane_bytes(&geo, cap);
        let lazy_slots = qwen_lazy_kv_slots(true, false, false, 1);
        let reserve = per_lane * lazy_slots as u64;
        let buckets = [1, 2, 4];
        let kept = |budget: u64| {
            solve_largest_bucket_prefix_that_fits(
                &buckets,
                per_lane,
                budget.saturating_sub(reserve),
            )
        };
        assert_eq!(geo.bytes_per_token_full_layers, 33280);
        assert_eq!(reserve, per_lane, "the eager singleton owns another cache");
        assert_eq!(kept(4 * per_lane), vec![1, 2]);
        assert_eq!(kept(5 * per_lane), vec![1, 2, 4]);
        assert_eq!(kept(5 * per_lane - 1), vec![1, 2]);
        assert!(kept(2 * per_lane - 1).is_empty());

        let fits = |kv| qwen35_per_lane_bytes(&geo, kv) * (4 + lazy_slots as u64) <= 4 * per_lane;
        let solved = solve_max_kv_tokens_monotone(fits, cap).unwrap();
        assert!(solved < cap && fits(solved));
        assert!(!fits(
            solved + KV_SOLVE_GRANULARITY_1024_ROUNDS_DOWN_SO_A_SOLVED_KV_NEVER_EXCEEDS_BUDGET
        ));
        let pool_only = qwen_lazy_kv_slots(true, false, true, 1);
        assert_eq!(
            solve_max_kv_tokens_monotone(
                |kv| qwen35_per_lane_bytes(&geo, kv) * (4 + pool_only as u64) <= 4 * per_lane,
                cap,
            ),
            Some(cap),
            "solo-via-lanes reuses a prepaid cache when no MTP head is loaded"
        );
    }

    fn plan_unmodeled(
        model_dir: &Path,
        family: &'static str,
        budget_bytes: u64,
    ) -> anyhow::Result<BootPlan> {
        let mut plan = new_process_plan(model_dir, family, budget_bytes, false);
        match checkpoint_tensor_bytes(model_dir) {
            Ok(t) => {
                let weights = sum_where(&t, |_| true);
                plan.push(
                    "weights-resident",
                    weights,
                    format!("safetensors headers, {} tensors; kv/scratch NOT modeled for this family, plan is advisory", t.len()),
                );
            }
            Err(e) => plan.push("weights-resident", 0, format!("header parse failed: {e:#}")),
        }
        common_slack(&mut plan);
        Ok(BootPlan {
            plan,
            kv_max_seq_len: 0,
            batch_bucket_sizes: None,
            batch_lane_kv: None,
            gemma4_batch_pool: None,
        })
    }

    fn plan_qwen4_exp(model_dir: &Path, raw_cfg: &str, budget_bytes: u64) -> anyhow::Result<BootPlan> {
        let cfg = nv_models::qwen4_exp::Qwen4ExpConfig::from_hf_json_str(raw_cfg)?;
        let mut plan = new_process_plan(
            model_dir,
            "qwen4_exp",
            budget_bytes,
            QWEN4_EXP_PLAN_IS_ADVISORY_UNTIL_A_LEDGER_MEASURES_ITS_LOADER_TRANSIENTS,
        );
        let host_table = |n: &str| n.contains(".ple_embedding.ngram_embedding.");
        let vision = |n: &str| n.starts_with("model.visual.");
        match checkpoint_tensor_bytes(model_dir) {
            Ok(t) => {
                plan.push(
                    "weights-resident",
                    sum_where(&t, |n| !host_table(n) && !vision(n)),
                    format!("safetensors bytes of every tensor the loader keeps on the device, {} tensors; fp8 side layers and the NVFP4 draft experts land below this", t.len()),
                );
                plan.push(
                    "ple-ngram-table-host-pinned",
                    0,
                    format!("{:.1} GiB of n-gram embedding shards are pinned in host RAM (NV_Q4EXP_PLE_HOST=1 gathers from the mmap instead), never resident on the device", sum_where(&t, host_table) as f64 / GIB),
                );
                plan.push(
                    "vision-tower-on-demand",
                    0,
                    format!("{:.2} GiB of model.visual.* loads as the qwen4exp-vision cold-tier resident on the first image request and is evicted when idle or under VRAM pressure; NV_Q4EXP_VISION=0 keeps it on disk and images go through the OCR bridge", sum_where(&t, vision) as f64 / GIB),
                );
            }
            Err(e) => plan.push("weights-resident", 0, format!("header parse failed: {e:#}")),
        }
        let per_env = per_engine_kv_env_for_family("qwen4_exp");
        let (kv_max_seq_len, knob) = kv_binding_from(
            per_env,
            per_env.and_then(|e| std::env::var(e).ok()).as_deref(),
            std::env::var(KV_MAX_SEQ_LEN_GLOBAL_ENV_THE_DEFAULT_EVERY_ENGINE_INHERITS).ok().as_deref(),
        )
        .map(|(n, env)| (n.min(cfg.max_position_embeddings), env))
        .unwrap_or((cfg.max_position_embeddings, KV_MAX_SEQ_LEN_GLOBAL_ENV_THE_DEFAULT_EVERY_ENGINE_INHERITS));
        let slots = cfg.full_attention.iter().filter(|a| **a).count() + cfg.mtp_num_hidden_layers;
        let per_token = (2 * cfg.num_key_value_heads * cfg.head_dim * 2 + cfg.indexer_head_dim * 2 + cfg.indexer_head_dim * 2 / cfg.indexer_compress_ratio) as u64;
        let short = nv_models::qwen4_exp::SESSION_MIN_LEN_8192_SHORT_REQUESTS_SHARE_ONE_STATE_SO_THEY_NEVER_OVERWRITE_THE_LONG_SESSIONS_PREFIX;
        plan.push(
            "kv-state-sessions",
            per_token * slots as u64 * 2 * kv_max_seq_len as u64,
            format!("{slots} attention slots (every 4th layer plus the MTP head) x {per_token} B/token x 2 sessions x {kv_max_seq_len} ({knob}) tokens; both are kept between requests and grow toward the engine window in 32768-token steps from {short}, the long one with its conversation and the short one, which serves prompts up to {short} tokens, with a request's max_tokens"),
        );
        let cap = nv_models::qwen4_exp::prefix_ckpt_cap();
        plan.push(
            "prefix-checkpoints",
            2 * cap as u64 * cfg.prefix_ckpt_bytes() as u64,
            format!("{cap} grid states per session (NV_Q4EXP_PREFIX_CKPTS) of every GDN state and the PLE window, {} B each", cfg.prefix_ckpt_bytes()),
        );
        common_slack(&mut plan);
        Ok(BootPlan { plan, kv_max_seq_len, batch_bucket_sizes: None, batch_lane_kv: None, gemma4_batch_pool: None })
    }

    pub fn plan_and_enforce_boot_shaped(
        family: PlanFamily,
        model_dir: &Path,
        raw_cfg: &str,
        shape: BootPlanShape,
    ) -> anyhow::Result<(BootPlan, RecordedBootPlan)> {
        let budget = env_budget_bytes();
        let boot = match family {
            PlanFamily::Gemma4 => plan_gemma4_shaped(model_dir, raw_cfg, budget, shape)?,
            PlanFamily::Qwen35Dense => plan_qwen35(model_dir, raw_cfg, true, false, budget)?,
            PlanFamily::Qwen35Moe { grouped_expert_copy } => {
                plan_qwen35(model_dir, raw_cfg, false, grouped_expert_copy, budget)?
            }
            PlanFamily::Unmodeled(name) => plan_unmodeled(model_dir, name, budget)?,
            PlanFamily::Laguna => plan_laguna(model_dir, raw_cfg, budget, shape)?,
            PlanFamily::Qwen4Exp => plan_qwen4_exp(model_dir, raw_cfg, budget)?,
        };
        eprintln!("{}", boot.plan.render());
        tracing::info!("{}", boot.plan.render());
        boot.plan.enforce()?;
        let recorded = record_boot_plan(&boot.plan);
        Ok((boot, recorded))
    }
}

#[cfg(test)]
pub(crate) struct ScopedEnvRestoredOnDrop {
    saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
    _env: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl ScopedEnvRestoredOnDrop {
    pub(crate) fn with(vars: &[(&'static str, Option<&str>)]) -> Self {
        let _env = crate::oapi::lock_serializing_every_test_that_mutates_the_process_global_env();
        let saved = vars
            .iter()
            .map(|&(key, value)| {
                let prev = std::env::var_os(key);
                match value {
                    Some(v) => unsafe { std::env::set_var(key, v) },
                    None => unsafe { std::env::remove_var(key) },
                }
                (key, prev)
            })
            .collect();
        Self { saved, _env }
    }
}

#[cfg(test)]
impl Drop for ScopedEnvRestoredOnDrop {
    fn drop(&mut self) {
        for (key, prev) in self.saved.drain(..).rev() {
            match prev {
                Some(v) => unsafe { std::env::set_var(key, v) },
                None => unsafe { std::env::remove_var(key) },
            }
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) fn synthetic_checkpoint_dir(tag: &str, tensors: &[(&str, u64)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut offset = 0u64;
    let header: serde_json::Map<String, serde_json::Value> = tensors
        .iter()
        .map(|&(name, bytes)| {
            let entry = serde_json::json!({
                "dtype": "U8", "shape": [bytes], "data_offsets": [offset, offset + bytes]
            });
            offset += bytes;
            (name.to_string(), entry)
        })
        .collect();
    let header = serde_json::Value::Object(header).to_string();
    let mut blob = (header.len() as u64).to_le_bytes().to_vec();
    blob.extend_from_slice(header.as_bytes());
    std::fs::write(dir.join("model.safetensors"), blob).unwrap();
    dir
}

#[cfg(test)]
static THE_PROCESS_WIDE_LEDGER_SERIALIZES_ITS_TESTS: Mutex<()> = Mutex::new(());

#[cfg(test)]
pub(crate) fn ledger_tests_run_one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    THE_PROCESS_WIDE_LEDGER_SERIALIZES_ITS_TESTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn qwen_lazy_kv_reservation_tracks_singletons_and_preserves_the_configured_cap() {
        for kv_slots in [1, 3] {
            for batch_on in [false, true] {
                for mtp_selected in [false, true] {
                    assert_eq!(
                        qwen_lazy_kv_slots(batch_on, mtp_selected, false, kv_slots),
                        kv_slots,
                        "plain and MTP singletons allocate outside the lane pool"
                    );
                    assert_eq!(
                        qwen_lazy_kv_slots(batch_on, mtp_selected, true, kv_slots),
                        if batch_on && !mtp_selected {
                            0
                        } else {
                            kv_slots
                        },
                        "solo-via-lanes cannot bypass an MTP head or a non-batched engine"
                    );
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn laguna_cache_and_retained_pool_ledger_has_checked_geometry_and_shape_counts() {
        let mut value = serde_json::json!({
            "num_hidden_layers": 40,
            "layer_types": (0..40).map(|i| if i % 4 == 0 { "full_attention" } else { "sliding_attention" }).collect::<Vec<_>>(),
            "mlp_only_layers": [0], "num_key_value_heads": 8, "head_dim": 128,
            "num_attention_heads": 48,
            "num_attention_heads_per_layer": (0..40).map(|i| if i % 4 == 0 { 48 } else { 64 }).collect::<Vec<_>>(),
            "sliding_window": 512, "num_experts": 256, "hidden_size": 2048,
            "moe_intermediate_size": 512, "vocab_size": 154880, "intermediate_size": 8192,
            "max_position_embeddings": 262144, "rms_norm_eps": 0.00001,
            "num_experts_per_tok": 8, "shared_expert_intermediate_size": 512,
            "rope_parameters": {
                "full_attention": { "rope_theta": 500000 },
                "sliding_attention": { "rope_theta": 500000 }
            }
        });
        let raw = value.to_string();
        let shape = LagunaSpecServeShape::from_config_json(&raw, 262144, 1024).unwrap();
        assert_eq!(shape.sliding_slots().unwrap(), 1536);
        assert_eq!(shape.bf16_cache_bytes().unwrap(), 10_926_161_920);
        assert_eq!(shape.fp8_cache_bytes().unwrap(), 5_792_333_824);
        assert_eq!(
            shape.grouped_expert_copy_bytes(false).unwrap(),
            17_666_608_128
        );
        assert_eq!(
            shape.grouped_expert_copy_bytes(true).unwrap(),
            17_735_618_316
        );
        let larger = LagunaSpecServeShape::from_config_json(&raw, 262144, 2048).unwrap();
        assert_eq!(
            larger.bf16_cache_bytes().unwrap() - shape.bf16_cache_bytes().unwrap(),
            125_829_120
        );
        assert_eq!(
            larger.fp8_cache_bytes().unwrap() - shape.fp8_cache_bytes().unwrap(),
            125_829_120
        );
        let small = LagunaSpecServeShape::from_config_json(&raw, 33, 1024).unwrap();
        assert_eq!(small.sliding_slots().unwrap(), 33);
        assert_eq!(small.bf16_cache_bytes().unwrap(), 5_406_720);
        assert!(LagunaSpecServeShape::from_config_json(&raw, 0, 1024).is_err());
        assert!(LagunaSpecServeShape::from_config_json(&raw, 262145, 1024).is_err());
        assert!(LagunaSpecServeShape::from_config_json(&raw, 262144, 513).is_err());
        assert!(LagunaSpecServeShape::from_config_json(&raw, 262144, 4096).is_ok());
        assert!(LagunaSpecServeShape::from_config_json(&raw, 262144, 8192).is_ok());
        let mut selectors = value.clone();
        selectors["mlp_layer_types"] = serde_json::json!(vec!["dense"; 40]);
        assert_eq!(
            LagunaSpecServeShape::from_config_json(&selectors.to_string(), 262144, 1024)
                .unwrap()
                .moe_layers,
            0
        );
        selectors["mlp_layer_types"][0] = serde_json::json!("sparse");
        assert_eq!(
            LagunaSpecServeShape::from_config_json(&selectors.to_string(), 262144, 1024)
                .unwrap()
                .moe_layers,
            1,
            "explicit sparse overrides mlp_only membership"
        );
        selectors.as_object_mut().unwrap().remove("mlp_layer_types");
        selectors["mlp_only_layers"] = serde_json::json!([0, 0]);
        assert_eq!(
            LagunaSpecServeShape::from_config_json(&selectors.to_string(), 262144, 1024)
                .unwrap()
                .moe_layers,
            39,
            "duplicate dense membership does not remove another layer"
        );
        selectors["decoder_sparse_step"] = serde_json::json!(2);
        selectors["mlp_only_layers"] = serde_json::json!([1, 1]);
        assert_eq!(
            LagunaSpecServeShape::from_config_json(&selectors.to_string(), 262144, 1024)
                .unwrap()
                .moe_layers,
            19
        );
        selectors["num_experts"] = serde_json::json!(0);
        assert_eq!(
            LagunaSpecServeShape::from_config_json(&selectors.to_string(), 262144, 1024)
                .unwrap()
                .moe_layers,
            0
        );
        selectors["num_key_value_heads"] = serde_json::json!(0);
        assert!(
            LagunaSpecServeShape::from_config_json(&selectors.to_string(), 262144, 1024).is_err()
        );
        value["head_dim"] = serde_json::json!(u64::MAX);
        assert!(
            LagunaSpecServeShape::from_config_json(&value.to_string(), 262144, 1024)
                .unwrap()
                .bf16_cache_bytes()
                .is_err()
        );
        value["sliding_window"] = serde_json::json!(u64::MAX);
        assert!(
            LagunaSpecServeShape::from_config_json(&value.to_string(), 262144, 1024)
                .unwrap()
                .sliding_slots()
                .is_err()
        );
    }

    #[test]
    fn verify_lmhead_int8_ships_on_when_unset_and_zero_is_the_bf16_escape() {
        const {
            assert!(
                VERIFY_LMHEAD_INT8_SHIPS_ON_SIGNED_OFF_2026_09_02_V2_NORMX_MK_PLUS_GEMV_I8_PRENORMED_MK_AT_K_GE_4096_48_OF_48_TEACHER_FORCED_AGREEMENT_AND_ZERO_RESTORES_THE_BF16_VERIFY_HEAD,
                "the int8 verify lm_head ships on, and NV_VERIFY_LMHEAD_INT8=0 restores the bf16 \
                 verify head"
            );
        }
        assert!(
            verify_lmhead_int8_from(None),
            "unset must resolve to the shipped int8 head"
        );
        assert!(
            !verify_lmhead_int8_from(Some("0")),
            "NV_VERIFY_LMHEAD_INT8=0 must restore the bf16 verify head"
        );
        assert!(
            verify_lmhead_int8_from(Some("1")),
            "NV_VERIFY_LMHEAD_INT8=1 must keep the int8 head"
        );
    }

    fn plan_of(budget: u64) -> VramPlan {
        VramPlan::new(Path::new("/x/model"), "test", budget, true)
    }

    #[test]
    fn a_dropped_engines_recorded_plan_leaves_the_prior_sum_so_swap_reloads_fit_where_stacking_refuses(
    ) {
        let _own_ledger = ledger_tests_run_one_at_a_time();
        let budget = 100 * MIB;
        let own = 40 * MIB;
        let ledger_before_this_test = recorded_planned_total_bytes();
        let enforce_and_record = || -> (anyhow::Result<()>, Option<RecordedBootPlan>) {
            let prior = recorded_planned_total_bytes() - ledger_before_this_test;
            let mut p = plan_of(budget);
            if prior > 0 {
                p.push(
                    PRIOR_ENGINES_ITEM_LABEL_STRIPPED_ON_RECORD_SO_EACH_ENGINE_COUNTS_ONCE,
                    prior,
                    "prior engines",
                );
            }
            p.push("weights-resident", own, "headers");
            let verdict = p.enforce();
            let record = verdict.is_ok().then(|| record_boot_plan(&p));
            (verdict, record)
        };

        let (base_ok, base_record) = enforce_and_record();
        base_ok.expect("base engine fits an empty ledger");
        assert_eq!(
            recorded_planned_total_bytes() - ledger_before_this_test,
            own,
            "recording must count the engine's OWN bytes once; if the prior-engines item were \
             not stripped on record, every later engine would re-count every earlier one"
        );

        let (v1_ok, v1_record) = enforce_and_record();
        v1_ok.expect("base + v1 fits the budget");
        assert_eq!(
            recorded_planned_total_bytes() - ledger_before_this_test,
            2 * own
        );

        let (v2_stacked, v2_stacked_record) = enforce_and_record();
        assert!(
            v2_stacked.is_err() && v2_stacked_record.is_none(),
            "base + v1 + v2 exceeds the budget, so enforcing v2 while v1 is still recorded \
             must refuse and must leave nothing in the ledger"
        );

        drop(v1_record);
        assert_eq!(
            recorded_planned_total_bytes() - ledger_before_this_test,
            own,
            "dropping an engine's RecordedBootPlan must decrement the enforced sum by exactly \
             what its record added; a sum that never decrements makes every \
             build-new-then-swap reload accumulate phantom prior-engine bytes until refusal"
        );

        let (v2_after_drop, v2_record) = enforce_and_record();
        v2_after_drop.expect(
            "after v1's record drops, base + v2 must fit: this is the swap-reload sequence \
             (enforce base, enforce v1, drop v1, enforce v2)",
        );
        assert_eq!(
            recorded_planned_total_bytes() - ledger_before_this_test,
            2 * own
        );
        drop(v2_record);
        drop(base_record);
        assert_eq!(
            recorded_planned_total_bytes() - ledger_before_this_test,
            0,
            "every record dropped must return the ledger to its pre-test level"
        );
    }

    #[test]
    fn enforce_refuses_over_budget_with_itemization_and_fix() {
        let mut p = plan_of(100 * MIB);
        p.push("weights-resident", 90 * MIB, "headers");
        p.push("kv-pools", 30 * MIB, "formula");
        p.kv_knob = Some(KvShrinkKnob {
            env: "NV_KV_MAX_SEQ_LEN",
            kv_max: 30 * 1024,
            lanes: 1,
            approx_bytes_per_token_per_lane: 1024,
            prior_engines_bytes_per_token_at_same_kv: 0,
            floor: 4096,
        });
        let err = p.enforce().expect_err("20 MiB over budget must refuse");
        let msg = format!("{err:#}");
        assert!(msg.contains("weights-resident"), "{msg}");
        assert!(msg.contains("kv-pools"), "{msg}");
        assert!(msg.contains("NV_KV_MAX_SEQ_LEN="), "{msg}");
        let suggested: usize = msg
            .split("NV_KV_MAX_SEQ_LEN=")
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse().ok())
            .expect("fix must name a concrete kv_max");
        assert!(suggested * 1024 + 90 * MIB as usize <= 100 * MIB as usize);
        assert_eq!(suggested % 1024, 0);
        let mut p = plan_of(10 * MIB);
        p.push("weights-resident", 20 * MIB, "headers");
        let msg = format!("{:#}", p.enforce().unwrap_err());
        assert!(msg.contains("NV_VRAM_BUDGET_GIB"), "without knobs the fix is a budget raise: {msg}");
        let mut p = VramPlan::new(Path::new("/x"), "unmodeled", 10 * MIB, false);
        p.push("weights-resident", 20 * MIB, "headers");
        assert!(p.enforce().is_ok(), "an unenforced family warns instead of refusing");
    }

    #[test]
    fn bucket_fix_drops_the_smallest_sufficient_suffix() {
        let mut p = plan_of(100 * MIB);
        p.push("kv-pools", 108 * MIB, "8 lanes x 13.5 MiB");
        p.bucket_knob = Some(BucketShrinkKnob {
            env: "NV_Q38_BATCH_SIZES",
            sizes: vec![1, 2, 4, 8],
            bytes_per_lane: (13.5 * MIB as f64) as u64,
        });
        let fix = p.smallest_change_to_fit();
        assert!(
            fix.starts_with("set NV_Q38_BATCH_SIZES=1,2,4"),
            "bucket cut must be the top-ranked change when no kv knob applies: {fix}"
        );
    }

    #[test]
    fn solve_max_kv_tokens_is_tight_and_granular() {
        let per = 1024u64;
        let n = solve_max_kv_tokens(100 * MIB, per, 1, 1 << 20).unwrap();
        assert_eq!(n % 1024, 0);
        assert!(n as u64 * per <= 100 * MIB);
        assert!((n + 1024) as u64 * per > 100 * MIB);
        assert_eq!(solve_max_kv_tokens(100 * MIB, per, 1, 8192), Some(8192));
        assert_eq!(solve_max_kv_tokens(MIB, per, 1, 1 << 20), None);
        let lanes4 = solve_max_kv_tokens(100 * MIB, per, 4, 1 << 20).unwrap();
        assert!(lanes4 as u64 * per * 4 <= 100 * MIB);
    }

    #[test]
    fn bucket_prefix_keeps_only_what_fits() {
        let sizes = [1usize, 2, 4, 8];
        assert_eq!(
            solve_largest_bucket_prefix_that_fits(&sizes, 10 * MIB, 45 * MIB),
            vec![1, 2, 4]
        );
        assert_eq!(
            solve_largest_bucket_prefix_that_fits(&sizes, 10 * MIB, 100 * MIB),
            vec![1, 2, 4, 8]
        );
        assert_eq!(
            solve_largest_bucket_prefix_that_fits(&sizes, 10 * MIB, 5 * MIB),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn synthetic_safetensors_header_bytes_are_exact() {
        let dir = std::env::temp_dir().join(format!("vram_plan_st_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let header = serde_json::json!({
            "__metadata__": {"format": "pt"},
            "model.layers.0.w": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
            "model.experts.0.w": {"dtype": "F32", "shape": [4], "data_offsets": [12, 28]}
        })
        .to_string();
        let mut blob = (header.len() as u64).to_le_bytes().to_vec();
        blob.extend_from_slice(header.as_bytes());
        blob.extend_from_slice(&[0u8; 28]);
        let f = dir.join("model.safetensors");
        std::fs::write(&f, &blob).unwrap();
        let entries = checkpoint_tensor_bytes(&dir).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(sum_where(&entries, |_| true), 28);
        assert_eq!(sum_where(&entries, |n| n.contains(".experts.")), 16);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn render_carries_every_item_and_the_verdict() {
        let mut p = plan_of(100 * MIB);
        p.push("a", 60 * MIB, "b1");
        p.push("b", 40 * MIB, "b2");
        assert_eq!(p.planned_total_bytes(), 100 * MIB);
        assert!(p.fits());
        p.push("c", 1, "b3");
        assert!(!p.fits());
        let mut p = plan_of(64 * MIB);
        p.push("weights-resident", 32 * MIB, "headers");
        p.push("slack-cuda-context", 16 * MIB, "const");
        let r = p.render();
        assert!(r.contains("weights-resident"));
        assert!(r.contains("slack-cuda-context"));
        assert!(r.contains("TOTAL"));
        assert!(r.contains("headroom"));
        let mut over = plan_of(1);
        over.push("weights-resident", 32 * MIB, "headers");
        assert!(over.render().contains("OVER by"));
    }

    #[test]
    fn recording_counts_live_engines_and_a_dropped_record_unrecords() {
        let _own = ledger_tests_run_one_at_a_time();
        let before = recorded_planned_total_bytes();
        let mut failed = plan_of(100 * MIB);
        failed.push("weights-resident", 999 * MIB, "headers");
        drop(record_boot_plan(&failed));
        assert_eq!(
            recorded_planned_total_bytes(),
            before,
            "a record dropped before boot completion must leave the ledger untouched; the \
             qwen36 e2e loud-failure phase needed a child process exactly because it did not"
        );
        let mut p = plan_of(100 * MIB);
        p.push("weights-resident", 7 * MIB, "headers");
        let live_engine_a = record_boot_plan(&p);
        assert_eq!(recorded_planned_total_bytes(), before + 7 * MIB);
        drop(record_boot_plan(&failed));
        assert_eq!(
            recorded_planned_total_bytes(),
            before + 7 * MIB,
            "unrecording a failed boot must remove only its own entry, not live engines'"
        );
        let mut q = plan_of(200 * MIB);
        push_prior_or_aux_process_items(&mut q, 90 * MIB, &AuxEnv::default());
        q.push("weights-resident", 11 * MIB, "headers");
        assert_eq!(q.planned_total_bytes(), 101 * MIB);
        let live_engine_b = record_boot_plan(&q);
        assert_eq!(
            recorded_planned_total_bytes(),
            before + 18 * MIB,
            "the prior-engines item is a view of already-recorded plans; re-recording it would \
             double-count every engine after the second"
        );
        let retry_prior = recorded_planned_total_bytes() - before;
        drop(record_boot_plan(&failed));
        assert_eq!(
            recorded_planned_total_bytes() - before,
            retry_prior,
            "a failed-boot-then-retry sequence must present the retry the same prior-engine \
             bytes a first boot would have seen"
        );
        drop(live_engine_b);
        drop(live_engine_a);
        assert_eq!(
            recorded_planned_total_bytes(),
            before,
            "every ledger entry is owned by exactly one handle, so dropping both live engines \
             must restore the ledger exactly"
        );
    }

    #[test]
    fn second_engine_enforce_sees_the_process_running_total_ledger_oom_scenario() {
        const GIB_U: u64 = 1 << 30;
        let budget = (76.5 * GIB_U as f64) as u64;
        let mut q38 = VramPlan::new(Path::new("/hub/q38"), "qwen3.5-dense", budget, true);
        q38.push(
            "everything",
            (58.495 * GIB_U as f64) as u64,
            "VRAM-LEDGER full-profile: per-model q38 plan total at kv 262144",
        );
        assert!(q38.fits(), "the ledger's per-model q38 plan passed alone");
        let mut g4 = VramPlan::new(Path::new("/hub/gemma4"), "gemma4", budget, true);
        g4.push(
            "everything",
            (55.798 * GIB_U as f64) as u64,
            "VRAM-LEDGER full-profile: per-model gemma4 plan total at kv 262144",
        );
        assert!(
            g4.fits(),
            "the ledger's per-model gemma4 plan passed alone -- exactly the bug: the real boot \
             OOM'd at 97 GiB on a ~125 GiB requirement"
        );
        push_prior_or_aux_process_items(&mut g4, q38.planned_total_bytes(), &AuxEnv::default());
        g4.kv_knob = Some(KvShrinkKnob {
            env: "NV_KV_MAX_SEQ_LEN",
            kv_max: 262144,
            lanes: 1,
            approx_bytes_per_token_per_lane: 81920,
            prior_engines_bytes_per_token_at_same_kv: 100_608,
            floor: 4096,
        });
        let err = g4
            .enforce()
            .expect_err("q38+gemma4 at 262144 must be refused at plan time, not OOM at load");
        let msg = format!("{err:#}");
        assert!(
            msg.contains(PRIOR_ENGINES_ITEM_LABEL_STRIPPED_ON_RECORD_SO_EACH_ENGINE_COUNTS_ONCE),
            "{msg}"
        );
        assert!(
            msg.contains("smallest changes that would fit, ranked:"),
            "{msg}"
        );
        assert!(
            msg.contains("option 1: set NV_KV_MAX_SEQ_LEN="),
            "the shared kv env drives BOTH engines' pools, so with the prior engines' \
             per-token rate folded in the top-ranked change must be a concrete kv shrink: {msg}"
        );
        assert!(
            msg.contains("NV_VRAM_BUDGET_GIB"),
            "the budget raise is always the last-resort option: {msg}"
        );
    }

    #[test]
    fn ranked_changes_list_every_viable_knob_in_capability_order() {
        let mut p = plan_of(100 * MIB);
        p.push("weights-resident", 80 * MIB, "headers");
        p.push("kv-pools", 40 * MIB, "formula");
        p.kv_knob = Some(KvShrinkKnob {
            env: "NV_KV_MAX_SEQ_LEN",
            kv_max: 40 * 1024,
            lanes: 1,
            approx_bytes_per_token_per_lane: 1024,
            prior_engines_bytes_per_token_at_same_kv: 0,
            floor: 4096,
        });
        p.bucket_knob = Some(BucketShrinkKnob {
            env: "NV_Q38_BATCH_SIZES",
            sizes: vec![1, 2, 4, 8],
            bytes_per_lane: 10 * MIB,
        });
        p.drafter_off_knob = Some((30 * MIB, "set NV_NO_SPEC=1"));
        let ranked = p.ranked_changes_to_fit();
        assert_eq!(ranked.len(), 4, "{ranked:?}");
        assert!(
            ranked[0].starts_with("set NV_KV_MAX_SEQ_LEN="),
            "{ranked:?}"
        );
        assert!(
            ranked[1].starts_with("set NV_Q38_BATCH_SIZES="),
            "{ranked:?}"
        );
        assert!(ranked[2].starts_with("set NV_NO_SPEC=1"), "{ranked:?}");
        assert!(
            ranked[3].starts_with("raise NV_VRAM_BUDGET_GIB"),
            "{ranked:?}"
        );
    }

    #[test]
    fn kv_concurrent_cap_defaults_to_one_slot_and_rejects_zero() {
        assert_eq!(kv_concurrent_cap_from(None), 1);
        assert_eq!(kv_concurrent_cap_from(Some("0")), 1);
        assert_eq!(kv_concurrent_cap_from(Some("junk")), 1);
        assert_eq!(kv_concurrent_cap_from(Some("3")), 3);
    }

    #[test]
    fn aux_itemization_keeps_legacy_reserves_excludes_cpu_tts_and_honors_override() {
        let all = AuxEnv {
            aux_override_gib: None,
            speech_stack: true,
            loaded_speech: None,
            embedding: true,
            tts_talker: true,
            tts_talker_on_cuda: false,
            tts_voice_design: false,
            tts_voice_design_on_cuda: false,
            dots_ocr: true,
            deepseek_ocr: true,
            diar_gpu_limit_mib: Some(2048),
        };
        let mut p = plan_of(u64::MAX);
        push_aux_items(&mut p, &all);
        assert_eq!(p.items.len(), 6);
        assert_eq!(
            p.planned_total_bytes(),
            (2638 + 1196 + 7068 + 7882 + 2048) * MIB
        );
        let tts = p
            .items
            .iter()
            .find(|item| item.label == "aux-qwen3-tts-talker")
            .unwrap();
        assert_eq!(tts.bytes, 0);
        assert!(!tts.unmeasured);
        assert!(tts.basis.contains("the CPU"));
        assert_eq!(p.items.iter().filter(|item| item.unmeasured).count(), 5);
        assert!(p.render().contains("INCOMPLETE ESTIMATE"));
        assert_eq!(
            (AUX_DOTS_OCR_STEADY_MIB_CKPT_SHARE_OF_MEASURED_LEDGER_OCR_PAIR_NET_14P6_GIB
                + AUX_DEEPSEEK_OCR2_STEADY_MIB_CKPT_SHARE_OF_MEASURED_LEDGER_OCR_PAIR_NET_14P6_GIB),
            14950,
            "the split must re-add to the measured OCR-pair net"
        );
        let mut o = plan_of(u64::MAX);
        push_aux_items(
            &mut o,
            &AuxEnv {
                aux_override_gib: Some(24.0),
                ..all
            },
        );
        assert_eq!(o.items.len(), 1);
        assert_eq!(o.planned_total_bytes(), (24.0 * GIB) as u64);
    }

    #[test]
    fn a_cuda_voxtral_tts_books_its_measured_residency_and_a_cpu_one_books_nothing() {
        for (provider, booked) in [("cuda", true), ("cpu", false)] {
            let aux = AuxEnv {
                speech_stack: true,
                loaded_speech: Some(vec![LoadedSpeechProvider {
                    component: VOXTRAL_TTS_SPEECH_COMPONENT,
                    provider,
                }]),
                ..AuxEnv::default()
            };
            let mut p = plan_of(u64::MAX);
            push_aux_items(&mut p, &aux);
            let item = p.items.iter().find(|i| i.label == "aux-voxtral-tts");
            assert_eq!(item.is_some(), booked, "provider {provider}");
            if let Some(item) = item {
                assert_eq!(
                    item.bytes,
                    AUX_VOXTRAL_TTS_4B_BF16_CUDA_MIB_MEASURED_STANDALONE_PROCESS_AFTER_WARMUP * MIB
                );
                assert!(!item.unmeasured);
            }
        }
    }

    #[test]
    fn a_cuda_tts_talker_books_its_measured_residency() {
        let mut p = plan_of(u64::MAX);
        push_aux_items(
            &mut p,
            &AuxEnv {
                tts_talker: true,
                tts_talker_on_cuda: true,
                tts_voice_design: true,
                tts_voice_design_on_cuda: true,
                ..AuxEnv::default()
            },
        );
        assert_eq!(p.items.len(), 2);
        assert_eq!(p.items[0].label, "aux-qwen3-tts-talker");
        assert_eq!(p.items[1].label, "aux-qwen3-tts-voicedesign");
        assert!(p.items.iter().all(|item| !item.unmeasured));
        assert_eq!(
            p.planned_total_bytes(),
            (AUX_QWEN3_TTS_0P6B_FP8_CUDA_MIB_ONE_STREAM_PEAK_3912_MEASURED_IN_PROCESS_PLUS_1416_THE_F32_LEDGERS_SECOND_STREAM
                + AUX_QWEN3_TTS_1P7B_VOICEDESIGN_BF16_CUDA_MIB_MEASURED_STANDALONE_PROCESS_ONE_STREAM)
                * MIB
        );
    }

    #[test]
    fn aux_providers_distinguish_cpu_handles_from_cuda_and_unreported_sessions() {
        let providers = |provider| {
            ["kokoro", "diar-segmentation", "diar-embedding"]
                .into_iter()
                .map(|component| LoadedSpeechProvider {
                    component,
                    provider,
                })
                .collect()
        };
        let mut aux = AuxEnv {
            speech_stack: true,
            loaded_speech: Some(providers("CPUExecutionProvider")),
            diar_gpu_limit_mib: Some(2048),
            ..AuxEnv::default()
        };
        let mut cpu = plan_of(u64::MAX);
        push_aux_items(&mut cpu, &aux);
        assert_eq!(cpu.planned_total_bytes(), 0);
        assert!(!cpu.items.iter().any(|item| item.unmeasured));
        aux.loaded_speech = Some(providers("CUDAExecutionProvider"));
        let mut cuda = plan_of(u64::MAX);
        push_aux_items(&mut cuda, &aux);
        assert_eq!(cuda.planned_total_bytes(), (2638 + 2048) * MIB);
        assert!(cuda.items.iter().all(|item| item.unmeasured));
        for component in ["kokoro", "diar-segmentation", "diar-embedding"] {
            assert!(cuda
                .render()
                .contains(&format!("{component}=CUDAExecutionProvider")));
        }
        aux.loaded_speech = Some(vec![
            LoadedSpeechProvider {
                component: "parakeet-encoder",
                provider: "cpu",
            },
            LoadedSpeechProvider {
                component: "parakeet-nemo/joint",
                provider: "unreported",
            },
        ]);
        assert!(!aux.speech_is_cpu_only());
        let mut partial = plan_of(u64::MAX);
        push_aux_items(&mut partial, &aux);
        assert_eq!(partial.planned_total_bytes(), (2638 + 2048) * MIB);
        assert!(partial.render().contains("parakeet-nemo/joint=unreported"));
        aux.loaded_speech = Some(Vec::new());
        let mut absent = plan_of(u64::MAX);
        push_aux_items(&mut absent, &aux);
        assert_eq!(absent.planned_total_bytes(), 0);
    }

    #[test]
    fn aux_reconcile_reports_incomplete_estimates_without_changing_tolerance() {
        assert_eq!(reconcile_verdict(10, 13, 3, false), "WITHIN PLAN");
        assert_eq!(
            reconcile_verdict(10, 13, 3, true),
            "WITHIN DECLARED BYTES; INCOMPLETE ESTIMATE"
        );
        for incomplete in [false, true] {
            assert_eq!(reconcile_verdict(10, 14, 3, incomplete), "EXCEEDS PLAN");
        }
        assert_eq!(
            reconcile_verdict(u64::MAX, u64::MAX, 3, true),
            "WITHIN DECLARED BYTES; INCOMPLETE ESTIMATE"
        );
    }

    #[test]
    fn aux_unmeasured_entries_follow_record_owner_without_double_counting() {
        let _own = ledger_tests_run_one_at_a_time();
        let before_bytes = recorded_planned_total_bytes();
        let before_unknown = recorded_unmeasured_items();
        let mut first = plan_of(u64::MAX);
        first.push_unmeasured("test-unknown-aux", 10 * MIB, "unknown resident bytes");
        let first_record = record_boot_plan(&first);
        let mut second = plan_of(u64::MAX);
        push_prior_or_aux_process_items(&mut second, 10 * MIB, &AuxEnv::default());
        second.push("test-second-engine", 5 * MIB, "headers");
        let second_record = record_boot_plan(&second);
        assert_eq!(recorded_planned_total_bytes(), before_bytes + 15 * MIB);
        assert_eq!(recorded_unmeasured_items().len(), before_unknown.len() + 1);
        drop(first_record);
        assert_eq!(recorded_planned_total_bytes(), before_bytes + 5 * MIB);
        assert_eq!(recorded_unmeasured_items(), before_unknown);
        drop(second_record);
        assert_eq!(recorded_planned_total_bytes(), before_bytes);
    }

    #[test]
    fn per_engine_kv_env_overrides_the_global_and_falls_back_in_order() {
        let q38 = per_engine_kv_env_for_family("qwen3.5-dense").unwrap();
        let g4 = per_engine_kv_env_for_family("gemma4").unwrap();
        assert_eq!(q38, "NV_KV_MAX_SEQ_LEN_Q38");
        assert_eq!(g4, "NV_KV_MAX_SEQ_LEN_GEMMA4");
        assert_eq!(per_engine_kv_env_for_family("qwen3.5-moe"), None);
        for (family, env) in [
            ("qwen3", "NV_KV_MAX_SEQ_LEN_QWEN3"),
            ("gemma4-moe", "NV_KV_MAX_SEQ_LEN_GEMMA4_MOE"),
            ("laguna", "NV_KV_MAX_SEQ_LEN_LAGUNA"),
            ("gpt-oss", "NV_KV_MAX_SEQ_LEN_GPTOSS"),
            ("omni", "NV_KV_MAX_SEQ_LEN_OMNI"),
        ] {
            assert_eq!(
                per_engine_kv_env_for_family(family),
                Some(env),
                "every unmodeled serving family needs its per-engine kv lift env"
            );
        }
        assert_eq!(
            kv_binding_from(Some(q38), Some("262144"), Some("16384")),
            Some((262144, q38))
        );
        assert_eq!(
            kv_binding_from(Some(q38), None, Some("16384")),
            Some((16384, "NV_KV_MAX_SEQ_LEN"))
        );
        assert_eq!(
            kv_binding_from(Some(q38), Some("junk"), Some("16384")),
            Some((16384, "NV_KV_MAX_SEQ_LEN"))
        );
        assert_eq!(kv_binding_from(Some(q38), Some("0"), None), None);
        assert_eq!(
            kv_binding_from(None, Some("262144"), Some("8192")),
            Some((8192, "NV_KV_MAX_SEQ_LEN")),
            "an engine without a per-engine env key must ignore the per-engine raw value"
        );
        assert_eq!(kv_binding_from(Some(q38), None, None), None);
    }

    #[test]
    fn q38_batch_lane_kv_defaults_to_min_of_kv_max_and_32768_and_env_overrides_within_kv_max() {
        assert_eq!(q38_batch_lane_kv_from(262144, None), 32768);
        assert_eq!(q38_batch_lane_kv_from(16384, None), 16384);
        assert_eq!(q38_batch_lane_kv_from(262144, Some("65536")), 65536);
        assert_eq!(q38_batch_lane_kv_from(262144, Some("junk")), 32768);
        assert_eq!(q38_batch_lane_kv_from(262144, Some("0")), 32768);
        assert_eq!(
            q38_batch_lane_kv_from(32768, Some("262144")),
            32768,
            "a lane deeper than the engine kv window is dead weight"
        );
    }

    #[test]
    fn q38_full_context_with_capped_lanes_and_solved_gemma4_fits_86_gib_on_the_ledger_measured_steadies(
    ) {
        let budget = 86 * (GIB as u64);
        let composed_fixed_mib: u64 = 73463 - 1345;
        let q38_bytes_per_token: u64 = 16 * (2 * 4 * 256 + 2 * 4 * 4) + 24 * 4;
        let q38_gdn_state_per_lane: u64 =
            48 * (48 * 128 * 128 * 4 + (2 * 16 * 128 + 48 * 128) * 4 * 4);
        let q38_lane = |kv: u64| q38_bytes_per_token * kv + q38_gdn_state_per_lane;
        let g4_bytes_per_token: u64 = 10 * 16 * 256 * 2;
        let g4_rings: u64 = 50 * 1024 * 16 * 256 * 2;
        let mut p = plan_of(budget);
        p.push(
            "composed-measured-boot-steady-minus-prepaid-lanes",
            composed_fixed_mib * MIB,
            "GPU-86GB verify2 boot steady 73463 MiB (aux + q38 + mtp + gemma4 apq + eagle3 \
             + mm towers) minus its 1.314 GiB kv-pools@16384x2, per the VRAM-LEDGER methodology",
        );
        p.push(
            "kv-pools@kv=32768x2lanes",
            2 * q38_lane(32768),
            "ledger q38-batch arms: ~1167 MiB per lane at kv 32768",
        );
        p.push(
            "kv-reserve-spec-route@kv_max=262144x1req",
            q38_lane(262144),
            "ledger solo-KV formula: 16 full layers x 4 kv_heads x 256 hd x 2 fp8 + scales, \
             8.0 GiB at 262144 + gdn state",
        );
        let fixed = p.planned_total_bytes();
        let solved = solve_max_kv_tokens_monotone(
            |kv| fixed + g4_rings + kv as u64 * g4_bytes_per_token <= budget,
            262144,
        )
        .expect("gemma4 must solve above the serving floor");
        assert!(
            (32768..=65536).contains(&solved),
            "the ledger steadies leave gemma4 a solved kv in the 32-64k band, got {solved}"
        );
        p.push(
            format!("kv-reserve@kv_max={solved}x1req"),
            g4_rings + solved as u64 * g4_bytes_per_token,
            "ledger gemma4 per-request fp8 KV rate 10.0 GiB @131k + sliding rings",
        );
        assert!(p.fits(), "{}", p.render());
        assert!(
            p.budget_bytes - p.planned_total_bytes() < (1u64 << 30),
            "the solver must be tight to within one kv granule's worth of budget: {}",
            p.render()
        );
    }

    #[test]
    fn refusal_names_the_per_engine_env_when_it_is_the_binding_knob() {
        let mut p = plan_of(100 * MIB);
        p.push("kv-reserve@kv_max=262144x1req", 130 * MIB, "forced deep");
        p.kv_knob = Some(KvShrinkKnob {
            env: per_engine_kv_env_for_family("gemma4").unwrap(),
            kv_max: 262144,
            lanes: 1,
            approx_bytes_per_token_per_lane: 1024,
            prior_engines_bytes_per_token_at_same_kv: 0,
            floor: 4096,
        });
        let msg = format!("{:#}", p.enforce().unwrap_err());
        assert!(msg.contains("set NV_KV_MAX_SEQ_LEN_GEMMA4="), "{msg}");
        assert!(
            !msg.contains("set NV_KV_MAX_SEQ_LEN="),
            "the global env is not the binding knob here and must not be suggested: {msg}"
        );

        let mut p = plan_of(100 * MIB);
        p.push(
            PRIOR_ENGINES_ITEM_LABEL_STRIPPED_ON_RECORD_SO_EACH_ENGINE_COUNTS_ONCE,
            90 * MIB,
            "prior q38 at full context",
        );
        p.push(
            "kv-reserve@kv_max=4096x1req",
            20 * MIB,
            "own kv already at the floor",
        );
        p.kv_knob = Some(KvShrinkKnob {
            env: per_engine_kv_env_for_family("gemma4").unwrap(),
            kv_max: 4096,
            lanes: 1,
            approx_bytes_per_token_per_lane: 1024,
            prior_engines_bytes_per_token_at_same_kv: 0,
            floor: 4096,
        });
        p.prior_kv_knobs = vec![KvShrinkKnob {
            env: per_engine_kv_env_for_family("qwen3.5-dense").unwrap(),
            kv_max: 262144,
            lanes: 1,
            approx_bytes_per_token_per_lane: 34_000,
            prior_engines_bytes_per_token_at_same_kv: 0,
            floor: 4096,
        }];
        let ranked = p.ranked_changes_to_fit();
        assert!(
            ranked[0].starts_with("set NV_KV_MAX_SEQ_LEN_Q38="),
            "shrinking the prior engine's per-engine kv is the only kv lever left: {ranked:?}"
        );
        assert!(
            ranked
                .last()
                .unwrap()
                .starts_with("raise NV_VRAM_BUDGET_GIB"),
            "{ranked:?}"
        );
        let target: usize = ranked[0]
            .split("NV_KV_MAX_SEQ_LEN_Q38=")
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse().ok())
            .expect("fix must name a concrete kv");
        assert_eq!(target % 1024, 0);
        assert!(
            (262144 - target) as u64 * 34_000 >= 10 * MIB,
            "the suggested shrink must free at least the overage"
        );
    }

    #[test]
    fn first_engine_gets_aux_items_and_later_engines_get_the_prior_sum() {
        let aux = AuxEnv {
            speech_stack: true,
            ..AuxEnv::default()
        };
        let mut first = plan_of(u64::MAX);
        push_prior_or_aux_process_items(&mut first, 0, &aux);
        assert_eq!(first.items.len(), 1);
        assert_eq!(first.items[0].label, "aux-speech-stack");
        let mut second = plan_of(u64::MAX);
        push_prior_or_aux_process_items(&mut second, 5 * MIB, &aux);
        assert_eq!(second.items.len(), 1);
        assert_eq!(
            second.items[0].label,
            PRIOR_ENGINES_ITEM_LABEL_STRIPPED_ON_RECORD_SO_EACH_ENGINE_COUNTS_ONCE,
            "aux is reserved once by the first plan and flows to later plans inside the prior sum"
        );
    }

    #[test]
    fn kv_reservation_scales_linearly_with_the_concurrent_slot_cap() {
        let per_request = 3 * (GIB as u64);
        for slots in 1..=4usize {
            let mut p = plan_of(20 * (GIB as u64));
            p.push(
                format!("kv-reserve@kv_max=32768x{slots}req"),
                per_request * slots as u64,
                "reservation arithmetic",
            );
            assert_eq!(p.planned_total_bytes(), per_request * slots as u64);
            assert_eq!(p.fits(), slots <= 6);
        }
    }

    #[test]
    fn monotone_solver_matches_the_linear_solver_on_a_linear_cost() {
        let per = 4096u64;
        let budget = 512 * MIB;
        let linear = solve_max_kv_tokens(budget, per, 1, 1 << 20).unwrap();
        let mono = solve_max_kv_tokens_monotone(|kv| kv as u64 * per <= budget, 1 << 20).unwrap();
        assert_eq!(linear, mono);
        assert!(solve_max_kv_tokens_monotone(|kv| kv <= 100, 1 << 20).is_none());
        assert_eq!(
            solve_max_kv_tokens_monotone(|_| true, 8192),
            Some(8192),
            "cap is honored before any bisection"
        );
    }

    #[test]
    fn a_dropped_engine_releases_its_ledger_bytes_so_sequential_harnesses_see_true_peak() {
        let _own = ledger_tests_run_one_at_a_time();
        let before = recorded_planned_total_bytes();
        let mut a = plan_of(86 * MIB);
        a.push("weights-resident", 60 * MIB, "engine A");
        let engine_a_owns_its_record = record_boot_plan(&a);
        assert_eq!(recorded_planned_total_bytes(), before + 60 * MIB);
        drop(engine_a_owns_its_record);
        assert_eq!(
            recorded_planned_total_bytes(),
            before,
            "boot-complete-drop must leave no phantom bytes: the c=16 concurrency harness \
             needed NV_VRAM_BUDGET_GIB=128 for a true peak that fits 86 exactly because a \
             completed boot leaked its record forever"
        );
        let mut b = plan_of(86 * MIB);
        push_prior_or_aux_process_items(
            &mut b,
            recorded_planned_total_bytes() - before,
            &AuxEnv::default(),
        );
        b.push(
            "weights-resident",
            70 * MIB,
            "engine B alone fits the budget but engine A's phantom 60 MiB would refuse it",
        );
        assert!(
            b.fits(),
            "the second sequential boot must be admitted on the true ledger, not refused on \
             the dropped engine's bytes: {}",
            b.render()
        );
        b.enforce()
            .expect("an in-budget second boot after a dropped first engine must not refuse");
        drop(record_boot_plan(&b));
        assert_eq!(recorded_planned_total_bytes(), before);
    }
}
