pub(crate) fn nv_drafter_kind(raw: Option<&str>) -> &'static str {
    match raw.map(str::trim) {
        Some("dflash") => "dflash",
        Some("auto") => "auto",
        Some("route") => "route",
        Some("mtp") => "mtp",
        Some("eagle3") | Some("") | None => "eagle3",
        Some(other) => {
            tracing::warn!(
                requested = other,
                "NV_DRAFTER must be 'eagle3', 'dflash', 'auto', 'route' or 'mtp'; defaulting to eagle3"
            );
            "eagle3"
        }
    }
}

pub(crate) fn suffix_drafter_enabled(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some(v) if !v.is_empty() && v != "0")
}

pub(crate) fn suffix_min_match(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&m| (1..=64).contains(&m))
        .unwrap_or(4)
}

pub(crate) fn prompt_looks_codeish(prompt: &str) -> bool {
    if prompt.contains("```") {
        return true;
    }
    let mut hits = 0usize;
    for pat in [
        "#include",
        "def ",
        "fn ",
        "class ",
        "import ",
        "function ",
        "();",
        "):",
        "=>",
        "});",
        "&&",
        "||",
        "::",
    ] {
        hits += prompt.matches(pat).count();
    }
    if hits >= 3 {
        return true;
    }
    let lines = prompt.lines().count().max(1);
    let sym_lines = prompt
        .lines()
        .filter(|l| {
            let t = l.trim_end();
            t.ends_with(';') || t.ends_with('{') || t.ends_with('}') || t.ends_with(") {")
        })
        .count();
    sym_lines * 4 >= lines && sym_lines >= 2
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DrafterArm {
    Eagle3,
    DFlash,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PromptClass {
    Code,
    Prose,
}

pub(crate) fn classify_prompt(prompt: &str) -> PromptClass {
    if prompt_looks_codeish(prompt) {
        PromptClass::Code
    } else {
        PromptClass::Prose
    }
}

pub(crate) fn route_drafter_arm(codeish: bool, dflash_ema: f64, eagle3_ema: f64) -> DrafterArm {
    if codeish || dflash_ema >= eagle3_ema {
        DrafterArm::DFlash
    } else {
        DrafterArm::Eagle3
    }
}

pub(crate) const ROUTE_CTX_GATE_DEFAULT: usize = 2048;

pub(crate) fn route_ctx_gate(raw: Option<&str>) -> usize {
    match raw.and_then(|s| s.trim().parse::<usize>().ok()) {
        Some(v) if v > 0 => v,
        _ => ROUTE_CTX_GATE_DEFAULT,
    }
}

pub(crate) const DFLASH_WINS_THROUGH_8K_BUT_ACCEPT_COLLAPSES_BY_32K_SO_AUTO_HANDS_OFF_TO_EAGLE3_AT_16384_PROMPT_TOKENS:
    usize = 16384;

pub(crate) fn drafter_auto_switch_tokens(raw: Option<&str>) -> usize {
    match raw.and_then(|s| s.trim().parse::<usize>().ok()) {
        Some(v) if v > 0 => v,
        _ => DFLASH_WINS_THROUGH_8K_BUT_ACCEPT_COLLAPSES_BY_32K_SO_AUTO_HANDS_OFF_TO_EAGLE3_AT_16384_PROMPT_TOKENS,
    }
}

pub(crate) fn drafter_arm_name(arm: DrafterArm) -> &'static str {
    match arm {
        DrafterArm::Eagle3 => "eagle3",
        DrafterArm::DFlash => "dflash",
    }
}

pub(crate) static LAST_ROUTED_DRAFTER_ARM_RECORDED_SO_SERVE_TESTS_CAN_ASSERT_AUTO_ROUTING:
    std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub(crate) fn record_last_routed_drafter_arm(arm: DrafterArm) {
    let code = match arm {
        DrafterArm::Eagle3 => 1u8,
        DrafterArm::DFlash => 2u8,
    };
    LAST_ROUTED_DRAFTER_ARM_RECORDED_SO_SERVE_TESTS_CAN_ASSERT_AUTO_ROUTING
        .store(code, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn last_routed_drafter_arm_name() -> Option<&'static str> {
    match LAST_ROUTED_DRAFTER_ARM_RECORDED_SO_SERVE_TESTS_CAN_ASSERT_AUTO_ROUTING
        .load(std::sync::atomic::Ordering::Relaxed)
    {
        1 => Some("eagle3"),
        2 => Some("dflash"),
        _ => None,
    }
}

pub(crate) fn route_arm_for_ctx(prompt_tokens: usize, ctx_gate: usize) -> DrafterArm {
    if prompt_tokens >= ctx_gate {
        DrafterArm::Eagle3
    } else {
        DrafterArm::DFlash
    }
}

pub(crate) fn drafter_row_elems_charge(eagle3_row_elems: usize, dflash_row_elems: usize) -> usize {
    eagle3_row_elems.max(dflash_row_elems)
}

pub(crate) fn resolve_drafter_arm(
    kind: &str,
    class: PromptClass,
    prompt_tokens: usize,
    ctx_gate: usize,
    dflash_loaded: bool,
    eagle3_loaded: bool,
    dflash_ema: f64,
    eagle3_ema: f64,
) -> Option<DrafterArm> {
    match (dflash_loaded, eagle3_loaded) {
        (false, false) => None,
        (true, false) => Some(DrafterArm::DFlash),
        (false, true) => Some(DrafterArm::Eagle3),
        (true, true) => Some(match kind {
            "route" | "auto" => route_arm_for_ctx(prompt_tokens, ctx_gate),
            _ => route_drafter_arm(class == PromptClass::Code, dflash_ema, eagle3_ema),
        }),
    }
}

pub(crate) static ARM_EMA_EAGLE3: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static ARM_EMA_DFLASH: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(crate) const ARM_EMA_DEFAULT: f64 = 2.0;
pub(crate) const ARM_EMA_ALPHA: f64 = 0.3;

pub(crate) fn arm_ema_cell(arm: DrafterArm) -> &'static std::sync::atomic::AtomicU64 {
    match arm {
        DrafterArm::Eagle3 => &ARM_EMA_EAGLE3,
        DrafterArm::DFlash => &ARM_EMA_DFLASH,
    }
}

pub(crate) fn arm_ema_get(arm: DrafterArm) -> f64 {
    let bits = arm_ema_cell(arm).load(std::sync::atomic::Ordering::Relaxed);
    if bits == 0 {
        ARM_EMA_DEFAULT
    } else {
        f64::from_bits(bits)
    }
}

pub(crate) fn arm_ema_step(old_bits: u64, accepted_per_round: f64) -> u64 {
    let old = if old_bits == 0 {
        ARM_EMA_DEFAULT
    } else {
        f64::from_bits(old_bits)
    };
    let new = (1.0 - ARM_EMA_ALPHA) * old + ARM_EMA_ALPHA * accepted_per_round;
    new.to_bits()
}

pub(crate) fn arm_ema_observe(arm: DrafterArm, accepted_per_round: f64) {
    if !accepted_per_round.is_finite() {
        return;
    }
    let _ = arm_ema_cell(arm).fetch_update(
        std::sync::atomic::Ordering::Relaxed,
        std::sync::atomic::Ordering::Relaxed,
        |bits| Some(arm_ema_step(bits, accepted_per_round)),
    );
}
