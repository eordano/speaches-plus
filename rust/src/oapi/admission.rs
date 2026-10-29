use std::sync::{Mutex, OnceLock};
use std::time::Duration;

pub const REJECT_PREFIX: &str = "vram-admission-reject:";

pub const DEFAULT_BUDGET_FRACTION: f64 = 0.8;

const FALLBACK_BUDGET_GIB_DELIBERATELY_SMALL_SHEDDING_RECOVERS_OOM_DOES_NOT: f64 = 16.0;
const DEFAULT_STATIC_GIB: f64 = 40.0;
const DEFAULT_QUEUE_MS: u64 = 3000;
const DEFAULT_TRANSIENT_PAD_GIB: f64 = 2.0;

const GIB: f64 = (1u64 << 30) as f64;

#[cfg(feature = "cuda")]
pub fn device_total_vram_bytes() -> Option<u64> {
    nv_layers::cudarc::driver::result::mem_get_info()
        .ok()
        .map(|(_free, total)| total as u64)
}

#[cfg(not(feature = "cuda"))]
pub fn device_total_vram_bytes() -> Option<u64> {
    None
}

pub fn default_budget_gib() -> f64 {
    if let Some(v) = std::env::var("NV_VRAM_BUDGET_GIB")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
    {
        return v;
    }
    match device_total_vram_bytes() {
        Some(total) => (total as f64 / GIB) * DEFAULT_BUDGET_FRACTION,
        None => {
            tracing::warn!(
                fallback_gib =
                    FALLBACK_BUDGET_GIB_DELIBERATELY_SMALL_SHEDDING_RECOVERS_OOM_DOES_NOT,
                "could not read device total VRAM; falling back to a deliberately \
                 SMALL budget. Set NV_VRAM_BUDGET_GIB explicitly if this host has \
                 more to give -- an under-estimate sheds, an over-estimate OOMs."
            );
            FALLBACK_BUDGET_GIB_DELIBERATELY_SMALL_SHEDDING_RECOVERS_OOM_DOES_NOT
        }
    }
}

static GATE: OnceLock<Option<std::sync::Arc<VramGate>>> = OnceLock::new();
static DRAFTER_ROW_ELEMS: OnceLock<usize> = OnceLock::new();

#[derive(Debug, Default)]
struct GateState {
    used: u64,
    retained: u64,
    active: u32,
}

#[derive(Debug)]
enum GateCapacity {
    #[cfg(test)]
    Fixed(u64),
    BudgetLessTheLedgerOutsideTheGemma4KvReserveFlooredByMeasuredStatic {
        budget: u64,
        measured_static: u64,
    },
}

impl GateCapacity {
    fn bytes(&self) -> u64 {
        match *self {
            #[cfg(test)]
            GateCapacity::Fixed(capacity) => capacity,
            GateCapacity::BudgetLessTheLedgerOutsideTheGemma4KvReserveFlooredByMeasuredStatic {
                budget,
                measured_static,
            } => budget.saturating_sub(measured_static.max(
                crate::oapi::vram_plan::recorded_planned_bytes_outside_the_gemma4_kv_reserve(),
            )),
        }
    }
}

#[derive(Debug)]
pub struct VramGate {
    capacity: GateCapacity,
    transient_pad: u64,
    queue: Duration,
    state: Mutex<GateState>,
    notify: tokio::sync::Notify,
}

#[derive(Debug)]
pub struct VramGuard {
    gate: std::sync::Arc<VramGate>,
    charge: u64,
    sticky: u64,
}

impl VramGuard {
    pub fn set_sticky(&mut self, sticky: u64) {
        self.sticky = sticky;
    }
}

impl Drop for VramGuard {
    fn drop(&mut self) {
        {
            let mut st = self.gate.state.lock().unwrap();
            let kept = if self.sticky > 0 {
                let grown = st.retained.max(self.sticky);
                let kept = grown - st.retained;
                st.retained = grown;
                kept
            } else {
                0
            };
            st.used = st.used.saturating_sub(self.charge.saturating_sub(kept));
            st.active = st.active.saturating_sub(1);
            tracing::info!(
                released_mib = (self.charge.saturating_sub(kept)) >> 20,
                retained_mib = st.retained >> 20,
                in_flight_mib = st.used >> 20,
                active = st.active,
                "vram admission: released"
            );
        }
        self.gate.notify.notify_waiters();
    }
}

#[derive(Debug)]
pub struct Rejected {
    pub needed: u64,
    pub used: u64,
    pub capacity: u64,
    pub active: u32,
    pub waited: Duration,
}

impl VramGate {
    fn new(capacity: GateCapacity, transient_pad: u64, queue: Duration) -> Self {
        Self {
            capacity,
            transient_pad,
            queue,
            state: Mutex::new(GateState::default()),
            notify: tokio::sync::Notify::new(),
        }
    }

    pub async fn admit(
        self: &std::sync::Arc<Self>,
        sticky: u64,
        extra: u64,
        pad: u64,
        label: &str,
    ) -> Result<VramGuard, Rejected> {
        let started = tokio::time::Instant::now();
        let deadline = started + self.queue;
        let mut waited_once = false;
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            {
                let mut st = self.state.lock().unwrap();
                notified.as_mut().enable();
                let capacity = self.capacity.bytes();
                let charge = sticky
                    .saturating_sub(st.retained)
                    .saturating_add(extra)
                    .saturating_add(pad);
                let fits = st.used.saturating_add(charge) <= capacity;
                if fits || st.active == 0 {
                    if !fits {
                        tracing::warn!(
                            label,
                            charge_mib = charge >> 20,
                            capacity_mib = capacity >> 20,
                            "vram admission: sole request exceeds the accounted budget; \
                             admitting anyway (startup gate is the real bound)"
                        );
                    }
                    st.used = st.used.saturating_add(charge);
                    st.active += 1;
                    tracing::info!(
                        label,
                        charge_mib = charge >> 20,
                        in_flight_mib = st.used >> 20,
                        capacity_mib = capacity >> 20,
                        active = st.active,
                        queued = waited_once,
                        "vram admission: admitted"
                    );
                    return Ok(VramGuard {
                        gate: self.clone(),
                        charge,
                        sticky,
                    });
                }
                let now = tokio::time::Instant::now();
                if now >= deadline {
                    let r = Rejected {
                        needed: charge,
                        used: st.used,
                        capacity,
                        active: st.active,
                        waited: now.saturating_duration_since(started),
                    };
                    tracing::warn!(
                        label,
                        needed_mib = r.needed >> 20,
                        in_flight_mib = r.used >> 20,
                        capacity_mib = r.capacity >> 20,
                        active = r.active,
                        "vram admission: rejected after bounded wait"
                    );
                    return Err(r);
                }
            }
            waited_once = true;
            let now = tokio::time::Instant::now();
            let _ = tokio::time::timeout(deadline.saturating_duration_since(now), notified).await;
        }
    }

    pub fn retry_after_secs(&self) -> u64 {
        (self.queue.as_millis() as u64).div_ceil(1000).max(1)
    }
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(default)
}

pub fn init_gemma4(measured_static_bytes: Option<u64>, drafter_row_elems: usize) {
    let _ = DRAFTER_ROW_ELEMS.set(drafter_row_elems);
    let _ = GATE.get_or_init(|| {
        if nv_config::env_arm_present_at_all_reread_every_call("NV_ADMIT_DISABLE") {
            tracing::warn!(
                "NV_ADMIT_DISABLE set: concurrent VRAM admission control is OFF. The primary \
                 chat gate is gone, so the NV_CHAT_CONCURRENCY semaphore is the ONLY limiter - \
                 and it is sized as a high backstop, not as a VRAM proxy. Set NV_CHAT_CONCURRENCY \
                 explicitly (low) if you run with admission disabled, or expect CUDA OOM."
            );
            return None;
        }
        let budget = (default_budget_gib() * GIB) as u64;
        let measured_static =
            measured_static_bytes.unwrap_or((DEFAULT_STATIC_GIB * GIB) as u64);
        let capacity =
            GateCapacity::BudgetLessTheLedgerOutsideTheGemma4KvReserveFlooredByMeasuredStatic {
                budget,
                measured_static,
            };
        let armed = capacity.bytes();
        let transient_pad =
            (env_f64("NV_ADMIT_TRANSIENT_GIB", DEFAULT_TRANSIENT_PAD_GIB) * GIB) as u64;
        let queue = Duration::from_millis(env_u64("NV_ADMIT_QUEUE_MS", DEFAULT_QUEUE_MS));
        let upper_bound = armed.checked_div(transient_pad).map_or(-1, |v| v as i64);
        tracing::info!(
            capacity_gib = format!("{:.2}", armed as f64 / GIB),
            capacity_mib = armed >> 20,
            static_gib = format!("{:.2}", measured_static as f64 / GIB),
            ledger_outside_the_gemma4_kv_reserve_gib = format!(
                "{:.2}",
                crate::oapi::vram_plan::recorded_planned_bytes_outside_the_gemma4_kv_reserve()
                    as f64
                    / GIB
            ),
            budget_gib = format!("{:.2}", budget as f64 / GIB),
            transient_pad_gib = format!("{:.2}", transient_pad as f64 / GIB),
            transient_pad_mib = transient_pad >> 20,
            max_concurrent_upper_bound = upper_bound,
            queue_ms = queue.as_millis() as u64,
            "vram admission gate armed: this is the PRIMARY chat concurrency gate. Every charge \
             includes the transient pad, so max_concurrent_upper_bound = capacity / pad is a hard \
             ceiling on concurrent admissions (-1 means the pad is 0 and there is no such \
             ceiling); real concurrency is lower because each request also charges its own KV. \
             NV_CHAT_CONCURRENCY is only a backstop above this. The capacity re-derives from the \
             vram plan ledger at every admission: the budget minus the larger of the measured \
             static and every recorded plan item outside the gemma4 kv-reserve, so other \
             engines' request reserves and engines booted or dropped later are never spent here."
        );
        Some(std::sync::Arc::new(VramGate::new(
            capacity,
            transient_pad,
            queue,
        )))
    });
}

pub fn gate() -> Option<std::sync::Arc<VramGate>> {
    GATE.get().cloned().flatten()
}

pub fn drafter_row_elems() -> usize {
    DRAFTER_ROW_ELEMS.get().copied().unwrap_or(0)
}

pub fn retry_after_secs() -> u64 {
    gate().map(|g| g.retry_after_secs()).unwrap_or(2)
}

pub fn rejected_error(r: &Rejected) -> anyhow::Error {
    anyhow::Error::new(crate::oapi::chat::EngineBusy::new(
        r.active as usize,
        r.waited.as_millis() as u64,
    ))
    .context(format!(
        "{REJECT_PREFIX} request needs {:.2} GiB of VRAM headroom but {:.2} of {:.2} GiB \
         is already in flight across {} request(s); waited {} ms; retry shortly",
        r.needed as f64 / GIB,
        r.used as f64 / GIB,
        r.capacity as f64 / GIB,
        r.active,
        r.waited.as_millis(),
    ))
}

pub async fn admit_or_bail(
    sticky: u64,
    extra: u64,
    label: &str,
) -> anyhow::Result<Option<VramGuard>> {
    let Some(g) = gate() else { return Ok(None) };
    let pad = g.transient_pad;
    match g.admit(sticky, extra, pad, label).await {
        Ok(guard) => Ok(Some(guard)),
        Err(r) => Err(rejected_error(&r)),
    }
}

pub async fn admit_or_bail_measured(
    sticky: u64,
    extra: u64,
    label: &str,
) -> anyhow::Result<Option<VramGuard>> {
    let Some(g) = gate() else { return Ok(None) };
    const DEFAULT_BATCH_PAD_MIB: u64 = 64;
    let pad = match std::env::var("NV_ADMIT_BATCH_MIB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        Some(mib) => mib << 20,
        None => DEFAULT_BATCH_PAD_MIB << 20,
    };
    match g.admit(sticky, extra, pad, label).await {
        Ok(guard) => Ok(Some(guard)),
        Err(r) => Err(rejected_error(&r)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use std::sync::Arc;

    const MIB: u64 = 1 << 20;

    fn gate_of(capacity: u64, pad: u64, queue_ms: u64) -> Arc<VramGate> {
        Arc::new(VramGate::new(
            GateCapacity::Fixed(capacity),
            pad,
            Duration::from_millis(queue_ms),
        ))
    }

    fn used(g: &Arc<VramGate>) -> u64 {
        g.state.lock().unwrap().used
    }

    fn retained(g: &Arc<VramGate>) -> u64 {
        g.state.lock().unwrap().retained
    }

    #[tokio::test]
    async fn charge_and_release_roundtrip() {
        let g = gate_of(100 * MIB, 0, 50);
        let a = g.admit(0, 30 * MIB, g.transient_pad, "t").await.unwrap();
        assert_eq!(used(&g), 30 * MIB);
        let b = g.admit(0, 40 * MIB, g.transient_pad, "t").await.unwrap();
        assert_eq!(used(&g), 70 * MIB);
        drop(a);
        assert_eq!(used(&g), 40 * MIB);
        drop(b);
        assert_eq!(used(&g), 0);
        assert_eq!(g.state.lock().unwrap().active, 0);
    }

    #[tokio::test]
    async fn transient_pad_is_added_to_every_charge() {
        let g = gate_of(100 * MIB, 5 * MIB, 50);
        let a = g.admit(0, 10 * MIB, g.transient_pad, "t").await.unwrap();
        assert_eq!(used(&g), 15 * MIB);
        drop(a);
        assert_eq!(used(&g), 0);
    }

    #[tokio::test]
    async fn sticky_retention_highwater() {
        let g = gate_of(100 * MIB, 0, 50);
        let a = g
            .admit(20 * MIB, 3 * MIB, g.transient_pad, "spec")
            .await
            .unwrap();
        assert_eq!(used(&g), 23 * MIB);
        drop(a);
        assert_eq!(retained(&g), 20 * MIB);
        assert_eq!(used(&g), 20 * MIB);

        let b = g
            .admit(20 * MIB, 3 * MIB, g.transient_pad, "spec")
            .await
            .unwrap();
        assert_eq!(used(&g), 23 * MIB);
        drop(b);
        assert_eq!(used(&g), 20 * MIB);

        let c = g
            .admit(50 * MIB, 3 * MIB, g.transient_pad, "spec")
            .await
            .unwrap();
        assert_eq!(used(&g), 20 * MIB + 30 * MIB + 3 * MIB);
        drop(c);
        assert_eq!(retained(&g), 50 * MIB);
        assert_eq!(used(&g), 50 * MIB);
    }

    #[tokio::test]
    async fn third_concurrent_request_rejected_then_admitted_after_release() {
        let g = gate_of(100 * MIB, 0, 80);
        let a = g.admit(0, 40 * MIB, g.transient_pad, "t").await.unwrap();
        let b = g.admit(0, 40 * MIB, g.transient_pad, "t").await.unwrap();
        let err = g.admit(0, 40 * MIB, g.transient_pad, "t").await;
        assert!(err.is_err());
        let r = err.err().unwrap();
        assert_eq!(r.needed, 40 * MIB);
        assert_eq!(r.used, 80 * MIB);

        let g2 = g.clone();
        let waiter =
            tokio::spawn(async move { g2.admit(0, 40 * MIB, g2.transient_pad, "t").await.ok() });
        tokio::time::sleep(Duration::from_millis(10)).await;
        drop(a);
        let w = waiter
            .await
            .unwrap()
            .expect("waiter should admit after release");
        assert_eq!(w.charge, 40 * MIB);
        assert_eq!(used(&g), 80 * MIB);
        drop(w);
        drop(b);
        assert_eq!(used(&g), 0);
    }

    #[tokio::test]
    async fn clearing_sticky_releases_the_full_charge_on_drop() {
        let g = gate_of(100 * MIB, 0, 50);
        let mut a = g
            .admit(20 * MIB, 3 * MIB, g.transient_pad, "spec")
            .await
            .unwrap();
        assert_eq!(used(&g), 23 * MIB);
        a.set_sticky(0);
        drop(a);
        assert_eq!(used(&g), 0);
        assert_eq!(retained(&g), 0);
    }

    #[tokio::test]
    async fn uncredited_full_charge_converted_to_retained_at_store_time() {
        let g = gate_of(100 * MIB, 0, 50);
        let mut a = g.admit(0, 23 * MIB, g.transient_pad, "spec").await.unwrap();
        assert_eq!(used(&g), 23 * MIB);
        assert_eq!(retained(&g), 0);
        a.set_sticky(20 * MIB);
        drop(a);
        assert_eq!(retained(&g), 20 * MIB);
        assert_eq!(used(&g), 20 * MIB);
    }

    #[tokio::test]
    async fn concurrent_spec_pair_lease_accounting_is_conservative() {
        let g = gate_of(100 * MIB, 0, 50);

        let mut r1 = g.admit(0, 23 * MIB, g.transient_pad, "spec").await.unwrap();
        r1.set_sticky(20 * MIB);
        drop(r1);
        assert_eq!(retained(&g), 20 * MIB);
        assert_eq!(used(&g), 20 * MIB);

        let r2 = g
            .admit(20 * MIB, 3 * MIB, g.transient_pad, "spec")
            .await
            .unwrap();
        assert_eq!(used(&g), 23 * MIB);
        let mut r3 = g.admit(0, 23 * MIB, g.transient_pad, "spec").await.unwrap();
        assert_eq!(used(&g), 46 * MIB);

        r3.set_sticky(0);
        drop(r3);
        assert_eq!(used(&g), 23 * MIB);
        drop(r2);
        assert_eq!(used(&g), 20 * MIB);
        assert_eq!(retained(&g), 20 * MIB);
    }

    #[tokio::test]
    async fn sole_request_over_budget_is_admitted() {
        let g = gate_of(10 * MIB, 0, 50);
        let a = g.admit(0, 50 * MIB, g.transient_pad, "t").await.unwrap();
        assert_eq!(used(&g), 50 * MIB);
        let err = g.admit(0, 1, g.transient_pad, "t").await;
        assert!(err.is_err());
        drop(a);
        assert_eq!(used(&g), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn panic_in_holder_releases_via_raii() {
        let g = gate_of(100 * MIB, 0, 50);
        let g2 = g.clone();
        let jh = tokio::spawn(async move {
            let _guard = g2.admit(0, 60 * MIB, g2.transient_pad, "t").await.unwrap();
            panic!("holder dies");
        });
        assert!(jh.await.is_err());
        assert_eq!(used(&g), 0);
        assert_eq!(g.state.lock().unwrap().active, 0);
    }

    #[tokio::test]
    async fn rejected_waited_is_measured_not_the_configured_queue() {
        let g = gate_of(100 * MIB, 0, 60);
        let _a = g.admit(0, 60 * MIB, g.transient_pad, "t").await.unwrap();
        let r = g
            .admit(0, 60 * MIB, g.transient_pad, "t")
            .await
            .err()
            .unwrap();
        assert!(
            r.waited > g.queue,
            "waited {:?} is the configured queue {:?}, not measured",
            r.waited,
            g.queue
        );
        assert!(r.waited < Duration::from_secs(5), "waited {:?}", r.waited);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn waiter_registers_before_the_state_lock_is_released() {
        for _ in 0..32 {
            let g = gate_of(100 * MIB, 0, 8_000);
            let a = g.admit(0, 60 * MIB, g.transient_pad, "t").await.unwrap();
            let g2 = g.clone();
            let waiter =
                tokio::spawn(
                    async move { g2.admit(0, 60 * MIB, g2.transient_pad, "t").await.is_ok() },
                );
            tokio::task::yield_now().await;
            drop(a);
            let ok = tokio::time::timeout(Duration::from_secs(3), waiter)
                .await
                .expect("waiter missed the release wakeup")
                .unwrap();
            assert!(ok);
        }
    }

    #[tokio::test]
    async fn rejection_is_typed_engine_busy_and_keeps_the_vram_arithmetic() {
        let g = gate_of(100 * MIB, 0, 40);
        let _a = g.admit(0, 100 * MIB, g.transient_pad, "t").await.unwrap();
        let r = g
            .admit(0, 40 * MIB, g.transient_pad, "t")
            .await
            .err()
            .unwrap();
        let err = rejected_error(&r);

        let busy = err.downcast_ref::<crate::oapi::chat::EngineBusy>().expect(
            "a VRAM shed must carry EngineBusy through the anyhow context chain so any \
                 caller that surfaces the error synchronously answers 503 rather than 500",
        );
        assert_eq!(busy.permits, 1);
        assert!(busy.waited_ms >= 40, "waited_ms {}", busy.waited_ms);

        let msg = format!("{err:#}");
        assert!(
            msg.starts_with(REJECT_PREFIX) && msg.contains("GiB"),
            "the shed message must LEAD with the reject prefix and keep the VRAM arithmetic: \
             the surviving 500 path (chat.rs ChatEvent::Error) only sees this string, so the \
             prefix is the only handle a fix there has: {msg}"
        );

        let resp = crate::oapi::chat::engine_start_error_response(&err);
        assert_eq!(
            resp.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "engine_start_error_response is the SYNCHRONOUS shed path. NvEngineChat::generate \
             spawns and returns Ok(()), so today the gemma4 shed does NOT reach here - it \
             arrives as ChatEvent::Error and becomes 500 engine_error. Fixing that needs \
             chat.rs / chat_engine.rs, not this file."
        );
    }

    #[tokio::test]
    async fn an_admitted_request_carries_no_busy_error() {
        let g = gate_of(100 * MIB, 0, 40);
        let guard = g.admit(0, 40 * MIB, g.transient_pad, "t").await;
        assert!(
            guard.is_ok(),
            "a request that fits must not shed; otherwise the 503 assertion above is vacuous"
        );
        assert_eq!(used(&g), 40 * MIB);
    }

    #[tokio::test]
    async fn rejection_waits_the_whole_admit_queue_window_before_shedding() {
        let queue_ms = 250u64;
        let g = gate_of(100 * MIB, 0, queue_ms);
        let _a = g.admit(0, 100 * MIB, g.transient_pad, "t").await.unwrap();
        let t0 = std::time::Instant::now();
        let r = g
            .admit(0, 40 * MIB, g.transient_pad, "t")
            .await
            .err()
            .unwrap();
        let elapsed = t0.elapsed();
        assert!(
            elapsed >= Duration::from_millis(queue_ms),
            "shed at {elapsed:?}, before the NV_ADMIT_QUEUE_MS window of {queue_ms} ms"
        );
        assert!(
            elapsed < Duration::from_millis(3_000),
            "shed late: {elapsed:?}"
        );
        assert!(r.waited >= Duration::from_millis(queue_ms));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_admit_math_never_oversubscribes() {
        let g = gate_of(100 * MIB, 0, 200);
        let mut handles = Vec::new();
        for _ in 0..8 {
            let g2 = g.clone();
            handles.push(tokio::spawn(async move {
                match g2.admit(0, 30 * MIB, g2.transient_pad, "t").await {
                    Ok(guard) => {
                        assert!(used(&g2) <= 100 * MIB);
                        tokio::time::sleep(Duration::from_millis(20)).await;
                        drop(guard);
                        true
                    }
                    Err(_) => false,
                }
            }));
        }
        let mut admitted = 0;
        for h in handles {
            if h.await.unwrap() {
                admitted += 1;
            }
        }
        assert!(admitted >= 3, "admitted only {admitted}");
        assert_eq!(used(&g), 0);
    }

    #[test]
    fn the_gemma4_gate_capacity_is_net_of_a_co_booted_q38_request_reserve() {
        use crate::oapi::vram_plan::{
            record_boot_plan, recorded_planned_total_bytes, VramPlan,
        };
        let gib = 1u64 << 30;
        let _ledger = crate::oapi::vram_plan::ledger_tests_run_one_at_a_time();
        let unrelated_ledger_bytes = recorded_planned_total_bytes();
        let _env = crate::oapi::vram_plan::ScopedEnvRestoredOnDrop::with(&[
            ("NV_VRAM_BUDGET_GIB", Some("86")),
            ("NV_ADMIT_DISABLE", None),
        ]);
        let budget = 86 * gib;
        let mut q38 = VramPlan::new(std::path::Path::new("/hub/q38"), "qwen3.5-dense", budget, true);
        q38.push("weights-resident", 40 * gib, "headers");
        q38.push(
            "kv-reserve-spec-route@kv_max=262144x1req",
            9 * gib,
            "lazy spec KV a live q38 request allocates",
        );
        let q38_record = record_boot_plan(&q38);
        let mut g4 = VramPlan::new(std::path::Path::new("/hub/gemma4"), "gemma4", budget, true);
        g4.push("weights-resident", 30 * gib, "headers");
        g4.push(
            format!(
                "{}kv_max=65536x1req",
                crate::oapi::vram_plan::GEMMA4_KV_RESERVE_LABEL_PREFIX_IS_WHAT_THE_RUNTIME_GATE_DISPENSES
            ),
            5 * gib,
            "lazy gemma4 request KV",
        );
        let _g4 = record_boot_plan(&g4);
        let measured_static = 70 * gib;
        init_gemma4(Some(measured_static), 0);
        let gate = gate().expect("the gate arms without NV_ADMIT_DISABLE");
        let co_booted = gate.capacity.bytes();
        assert_eq!(
            co_booted,
            budget - (unrelated_ledger_bytes + 79 * gib).max(measured_static),
            "the q38 weights and the gemma4 weights fill 70 GiB of a measured 70 GiB static, so \
             the capacity must also hold back q38's planned 9 GiB spec reserve"
        );
        assert!(
            co_booted <= budget - measured_static - 9 * gib,
            "capacity {} GiB lets gemma4 requests spend q38's planned 9 GiB spec reserve",
            co_booted as f64 / GIB
        );
        let mut later = VramPlan::new(std::path::Path::new("/hub/later"), "qwen3", budget, false);
        later.push("weights-resident", 2 * gib, "an engine booted after gemma4 measured");
        let later_record = record_boot_plan(&later);
        assert_eq!(
            gate.capacity.bytes(),
            co_booted - 2 * gib,
            "an engine that records after the gate armed re-arms it"
        );
        drop(later_record);
        assert_eq!(gate.capacity.bytes(), co_booted, "a dropped engine re-arms it back");
        drop(q38_record);
        assert_eq!(
            gate.capacity.bytes(),
            budget - (unrelated_ledger_bytes + 30 * gib).max(measured_static),
            "with q38 gone the measured static is the floor again"
        );
    }
}
