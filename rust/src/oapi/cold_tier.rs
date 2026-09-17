use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tracing::{info, warn};

pub const IDLE_SECS_ENV: &str = "NV_COLD_TIER_IDLE_SECS";
pub const PRELOAD_ENV: &str = "NV_COLD_TIER_PRELOAD";
pub const CHAT_MIN_FREE_GIB_ENV: &str = "NV_COLD_TIER_CHAT_MIN_FREE_GIB";
pub const FIT_MIN_FREE_GIB_ENV: &str = "NV_COLD_TIER_FIT_MIN_FREE_GIB";

const DEFAULT_IDLE_SECS: u64 = 600;
const RETRY_AFTER_FAILED_LOAD: Duration = Duration::from_secs(30);
const DEFAULT_CHAT_MIN_FREE_GIB: f64 = 2.0;
const DEFAULT_FIT_MIN_FREE_GIB: f64 = 6.0;
const SWEEP_EVERY: Duration = Duration::from_secs(30);
const GIB: f64 = (1u64 << 30) as f64;

pub trait Resident: Send + Sync {
    fn name(&self) -> &'static str;
    fn estimate_bytes(&self) -> u64;
    fn is_loaded(&self) -> bool;
    fn last_used(&self) -> Option<Instant>;
    fn evict(&self, reason: &str) -> bool;
    fn load_blocking(&self) -> bool;
}

type Loader<T> = Arc<dyn Fn() -> Option<Arc<T>> + Send + Sync>;

struct SlotState<T> {
    value: Option<Arc<T>>,
    last_used: Option<Instant>,
    failed_at: Option<Instant>,
}

pub struct ColdSlot<T> {
    name: &'static str,
    estimate_bytes: u64,
    loader: Loader<T>,
    state: Mutex<SlotState<T>>,
    load_lock: tokio::sync::Mutex<()>,
}

impl<T: Send + Sync + 'static> ColdSlot<T> {
    fn build(name: &'static str, estimate_bytes: u64, loader: Loader<T>, value: Option<Arc<T>>) -> Arc<Self> {
        Arc::new(Self {
            name,
            estimate_bytes,
            loader,
            state: Mutex::new(SlotState {
                last_used: value.as_ref().map(|_| Instant::now()),
                value,
                failed_at: None,
            }),
            load_lock: tokio::sync::Mutex::new(()),
        })
    }

    pub fn demand(name: &'static str, estimate_bytes: u64, loader: Loader<T>) -> Arc<Self> {
        let slot = Self::build(name, estimate_bytes, loader, None);
        register(slot.clone());
        slot
    }

    pub fn resident(name: &'static str, value: Arc<T>) -> Arc<Self> {
        Self::build(name, 0, Arc::new(|| None), Some(value))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, SlotState<T>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn peek(&self) -> Option<Arc<T>> {
        let mut st = self.lock();
        let v = st.value.clone();
        if v.is_some() {
            st.last_used = Some(Instant::now());
        }
        v
    }

    fn in_retry_backoff(&self) -> bool {
        self.lock()
            .failed_at
            .is_some_and(|t| t.elapsed() < RETRY_AFTER_FAILED_LOAD)
    }

    fn store(&self, loaded: Option<Arc<T>>, started: Instant, reason: &str) -> Option<Arc<T>> {
        let mut st = self.lock();
        match loaded {
            Some(v) => {
                st.value = Some(v.clone());
                st.last_used = Some(Instant::now());
                st.failed_at = None;
                info!(
                    resident = self.name,
                    load_ms = started.elapsed().as_millis() as u64,
                    estimate_gib = format!("{:.2}", self.estimate_bytes as f64 / GIB),
                    free_gib = free_gib_label(),
                    reason,
                    "cold tier: resident loaded"
                );
                Some(v)
            }
            None => {
                st.failed_at = Some(Instant::now());
                warn!(
                    resident = self.name,
                    retry_secs = RETRY_AFTER_FAILED_LOAD.as_secs(),
                    free_gib = free_gib_label(),
                    reason,
                    "cold tier: resident load failed; requests answer 503 until the retry window passes"
                );
                None
            }
        }
    }

    pub async fn get(&self) -> Option<Arc<T>> {
        if let Some(v) = self.peek() {
            return Some(v);
        }
        let _load = self.load_lock.lock().await;
        if let Some(v) = self.peek() {
            return Some(v);
        }
        if self.in_retry_backoff() {
            return None;
        }
        let loader = self.loader.clone();
        let (name, need) = (self.name, self.estimate_bytes);
        let started = Instant::now();
        let loaded = tokio::task::spawn_blocking(move || {
            ensure_free(need, name, Some(name));
            loader()
        })
        .await
        .ok()
        .flatten();
        self.store(loaded, started, "request")
    }

    pub fn load_failed(&self) -> bool {
        self.lock().failed_at.is_some()
    }
}

impl<T: Send + Sync + 'static> Resident for ColdSlot<T> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn estimate_bytes(&self) -> u64 {
        self.estimate_bytes
    }

    fn is_loaded(&self) -> bool {
        self.lock().value.is_some()
    }

    fn last_used(&self) -> Option<Instant> {
        self.lock().last_used
    }

    fn evict(&self, reason: &str) -> bool {
        let taken = {
            let mut st = self.lock();
            st.value.take()
        };
        let Some(value) = taken else {
            return false;
        };
        let refs = Arc::strong_count(&value) - 1;
        let before = free_gib_label();
        drop(value);
        synchronize_device();
        info!(
            resident = self.name,
            reason,
            estimate_gib = format!("{:.2}", self.estimate_bytes as f64 / GIB),
            in_flight_refs = refs,
            free_gib_before = before,
            free_gib_after = free_gib_label(),
            "cold tier: resident evicted"
        );
        true
    }

    fn load_blocking(&self) -> bool {
        if self.is_loaded() {
            return true;
        }
        let started = Instant::now();
        let loaded = (self.loader)();
        self.store(loaded, started, "preload").is_some()
    }
}

static REGISTRY: Mutex<Vec<Arc<dyn Resident>>> = Mutex::new(Vec::new());

fn register(r: Arc<dyn Resident>) {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner()).push(r);
}

pub fn residents() -> Vec<Arc<dyn Resident>> {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_gib_bytes(key: &str, default: f64) -> u64 {
    let gib = std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(default);
    (gib * GIB) as u64
}

pub fn chat_min_free_bytes() -> u64 {
    env_gib_bytes(CHAT_MIN_FREE_GIB_ENV, DEFAULT_CHAT_MIN_FREE_GIB)
}

pub fn fit_min_free_bytes() -> u64 {
    env_gib_bytes(FIT_MIN_FREE_GIB_ENV, DEFAULT_FIT_MIN_FREE_GIB)
}

pub fn preload_requested() -> bool {
    std::env::var(PRELOAD_ENV).as_deref() == Ok("1")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeReport {
    pub free_before: Option<u64>,
    pub free_after: Option<u64>,
    pub evicted: Vec<&'static str>,
    pub satisfied: bool,
}

pub fn ensure_free(need_bytes: u64, reason: &str, exclude: Option<&str>) -> FreeReport {
    let mut report = FreeReport {
        free_before: None,
        free_after: None,
        evicted: Vec::new(),
        satisfied: true,
    };
    if need_bytes == 0 {
        return report;
    }
    let mut candidates: Vec<Arc<dyn Resident>> = residents()
        .into_iter()
        .filter(|r| r.is_loaded() && exclude != Some(r.name()))
        .collect();
    if candidates.is_empty() {
        return report;
    }
    let Some(free) = device_free_bytes() else {
        return report;
    };
    report.free_before = Some(free);
    report.free_after = Some(free);
    if free >= need_bytes {
        return report;
    }
    candidates.sort_by_key(|r| r.last_used());
    for r in candidates {
        if r.evict(reason) {
            report.evicted.push(r.name());
        }
        let now = device_free_bytes().unwrap_or(free);
        report.free_after = Some(now);
        if now >= need_bytes {
            break;
        }
    }
    report.satisfied = report.free_after.is_some_and(|f| f >= need_bytes);
    warn!(
        reason,
        need_gib = format!("{:.2}", need_bytes as f64 / GIB),
        free_gib_before = format!("{:.2}", free as f64 / GIB),
        free_gib_after = report
            .free_after
            .map(|f| format!("{:.2}", f as f64 / GIB))
            .unwrap_or_default(),
        evicted = ?report.evicted,
        satisfied = report.satisfied,
        "cold tier: VRAM pressure eviction"
    );
    report
}

pub async fn ensure_free_async(need_bytes: u64, reason: &'static str) -> FreeReport {
    let any_loaded = residents().iter().any(|r| r.is_loaded());
    if need_bytes == 0 || !any_loaded {
        return FreeReport {
            free_before: None,
            free_after: None,
            evicted: Vec::new(),
            satisfied: true,
        };
    }
    tokio::task::spawn_blocking(move || ensure_free(need_bytes, reason, None))
        .await
        .unwrap_or(FreeReport {
            free_before: None,
            free_after: None,
            evicted: Vec::new(),
            satisfied: true,
        })
}

pub fn sweep_idle() -> Vec<&'static str> {
    let idle = env_u64(IDLE_SECS_ENV, DEFAULT_IDLE_SECS);
    if idle == 0 {
        return Vec::new();
    }
    let idle = Duration::from_secs(idle);
    residents()
        .into_iter()
        .filter(|r| r.is_loaded() && r.last_used().is_some_and(|t| t.elapsed() >= idle))
        .filter(|r| r.evict("idle"))
        .map(|r| r.name())
        .collect()
}

pub fn spawn_idle_sweeper() {
    tokio::spawn(async {
        loop {
            tokio::time::sleep(SWEEP_EVERY).await;
            let _ = tokio::task::spawn_blocking(sweep_idle).await;
        }
    });
}

pub async fn preload_if_requested() {
    let names: Vec<&'static str> = residents().iter().map(|r| r.name()).collect();
    if names.is_empty() {
        return;
    }
    if !preload_requested() {
        info!(
            residents = ?names,
            idle_secs = env_u64(IDLE_SECS_ENV, DEFAULT_IDLE_SECS),
            "cold tier: residents load on first request and unload when idle; {PRELOAD_ENV}=1 loads them at boot"
        );
        return;
    }
    let _ = tokio::task::spawn_blocking(|| {
        for r in residents() {
            r.load_blocking();
        }
    })
    .await;
}

fn free_gib_label() -> String {
    device_free_bytes()
        .map(|f| format!("{:.2}", f as f64 / GIB))
        .unwrap_or_else(|| "?".into())
}

#[cfg(feature = "cuda")]
fn cuda_ctx() -> Option<Arc<nv_layers::cudarc::driver::CudaContext>> {
    use std::sync::OnceLock;
    static CTX: OnceLock<Option<Arc<nv_layers::cudarc::driver::CudaContext>>> = OnceLock::new();
    CTX.get_or_init(|| nv_layers::cudarc::driver::CudaContext::new(0).ok())
        .clone()
}

#[cfg(feature = "cuda")]
pub fn device_free_bytes() -> Option<u64> {
    let ctx = cuda_ctx()?;
    ctx.bind_to_thread().ok()?;
    nv_layers::cudarc::driver::result::mem_get_info()
        .ok()
        .map(|(free, _total)| free as u64)
}

#[cfg(feature = "cuda")]
fn synchronize_device() {
    if let Some(ctx) = cuda_ctx() {
        let _ = ctx.synchronize();
    }
}

#[cfg(not(feature = "cuda"))]
pub fn device_free_bytes() -> Option<u64> {
    None
}

#[cfg(not(feature = "cuda"))]
fn synchronize_device() {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Big(u32);

    fn counting_loader(loads: Arc<AtomicUsize>, ok: bool) -> Loader<Big> {
        Arc::new(move || {
            loads.fetch_add(1, Ordering::SeqCst);
            ok.then(|| Arc::new(Big(7)))
        })
    }

    #[tokio::test]
    async fn a_demand_slot_loads_once_and_hands_out_the_same_arc() {
        let loads = Arc::new(AtomicUsize::new(0));
        let slot = ColdSlot::build("t-once", 1, counting_loader(loads.clone(), true), None);
        assert!(!slot.is_loaded());
        let a = slot.get().await.expect("loads");
        let b = slot.get().await.expect("cached");
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert_eq!(a.0, 7);
    }

    #[tokio::test]
    async fn eviction_drops_the_shared_ref_and_the_next_get_reloads() {
        let loads = Arc::new(AtomicUsize::new(0));
        let slot = ColdSlot::build("t-evict", 1, counting_loader(loads.clone(), true), None);
        let held = slot.get().await.expect("loads");
        assert!(slot.evict("test"));
        assert!(!slot.is_loaded());
        assert!(!slot.evict("test"), "a second evict has nothing to drop");
        assert_eq!(held.0, 7, "an in-flight holder keeps its engine");
        slot.get().await.expect("reloads");
        assert_eq!(loads.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_failed_load_backs_off_and_does_not_hammer_the_loader() {
        let loads = Arc::new(AtomicUsize::new(0));
        let slot = ColdSlot::build("t-fail", 1, counting_loader(loads.clone(), false), None);
        assert!(slot.get().await.is_none());
        assert!(slot.get().await.is_none());
        assert_eq!(loads.load(Ordering::SeqCst), 1, "second call sits inside the retry window");
    }

    #[test]
    fn a_resident_slot_is_loaded_from_the_start_and_evictable() {
        let slot = ColdSlot::resident("t-res", Arc::new(Big(1)));
        assert!(slot.is_loaded());
        assert_eq!(slot.peek().map(|b| b.0), Some(1));
        assert!(slot.evict("test"));
        assert!(slot.peek().is_none());
    }

    #[test]
    fn idle_zero_disables_the_sweep() {
        let _env = crate::oapi::lock_serializing_every_test_that_mutates_the_process_global_env();
        std::env::set_var(IDLE_SECS_ENV, "0");
        assert!(sweep_idle().is_empty());
        std::env::remove_var(IDLE_SECS_ENV);
    }

    #[test]
    fn a_failed_slot_reports_it_until_a_load_succeeds() {
        let slot = ColdSlot::build("t-flag", 1, Arc::new(|| None::<Arc<Big>>), None);
        assert!(!slot.load_failed());
        assert!(!slot.load_blocking());
        assert!(slot.load_failed());
    }

    #[test]
    fn ensure_free_without_a_device_reports_nothing_and_evicts_nothing() {
        let report = ensure_free(0, "test", None);
        assert!(report.satisfied);
        assert!(report.evicted.is_empty());
    }

    #[test]
    fn gib_env_parsing_rejects_garbage_and_negatives() {
        let _env = crate::oapi::lock_serializing_every_test_that_mutates_the_process_global_env();
        std::env::set_var(CHAT_MIN_FREE_GIB_ENV, "-1");
        assert_eq!(chat_min_free_bytes(), (DEFAULT_CHAT_MIN_FREE_GIB * GIB) as u64);
        std::env::set_var(CHAT_MIN_FREE_GIB_ENV, "0.5");
        assert_eq!(chat_min_free_bytes(), (0.5 * GIB) as u64);
        std::env::set_var(CHAT_MIN_FREE_GIB_ENV, "nope");
        assert_eq!(chat_min_free_bytes(), (DEFAULT_CHAT_MIN_FREE_GIB * GIB) as u64);
        std::env::remove_var(CHAT_MIN_FREE_GIB_ENV);
    }
}
