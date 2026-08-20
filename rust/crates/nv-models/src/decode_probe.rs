use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NV_PROF_DECODE_PHASES").is_some())
}

fn report_every() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("NV_PROF_DECODE_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or(256)
    })
}

static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
static PEAK_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicU64 = AtomicU64::new(0);

pub fn enter() -> usize {
    let n = IN_FLIGHT.fetch_add(1, Ordering::SeqCst) + 1;
    PEAK_IN_FLIGHT.fetch_max(n, Ordering::SeqCst);
    n
}

pub fn leave() {
    IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
}

#[derive(Default, Clone, Copy)]
struct Bucket {
    n: u64,
    wall_ms: f64,
    launch_ms: f64,
    sync_ms: f64,
    dtoh_ms: f64,
    argmax_ms: f64,
    gpu_ms: f64,
    gpu_n: u64,
}

fn acc() -> &'static Mutex<BTreeMap<usize, Bucket>> {
    static A: OnceLock<Mutex<BTreeMap<usize, Bucket>>> = OnceLock::new();
    A.get_or_init(|| Mutex::new(BTreeMap::new()))
}

pub struct Sample {
    pub in_flight: usize,
    pub wall_ms: f64,
    pub launch_ms: f64,
    pub sync_ms: f64,
    pub dtoh_ms: f64,
    pub argmax_ms: f64,

    pub gpu_ms: Option<f64>,
}

pub fn record(s: Sample) {
    {
        let mut g = acc().lock().unwrap_or_else(|p| p.into_inner());
        let b = g.entry(s.in_flight).or_default();
        b.n += 1;
        b.wall_ms += s.wall_ms;
        b.launch_ms += s.launch_ms;
        b.sync_ms += s.sync_ms;
        b.dtoh_ms += s.dtoh_ms;
        b.argmax_ms += s.argmax_ms;
        if let Some(g_ms) = s.gpu_ms {
            b.gpu_ms += g_ms;
            b.gpu_n += 1;
        }
    }
    let c = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if c.is_multiple_of(report_every()) {
        dump("periodic");
    }
}

pub fn gpu_coverage() -> (u64, u64) {
    let g = acc().lock().unwrap_or_else(|p| p.into_inner());
    g.values().fold((0, 0), |(gn, n), b| (gn + b.gpu_n, n + b.n))
}

pub fn dump(tag: &str) {
    let g = acc().lock().unwrap_or_else(|p| p.into_inner());
    let peak = PEAK_IN_FLIGHT.load(Ordering::SeqCst);
    let live = IN_FLIGHT.load(Ordering::SeqCst);
    eprintln!(
        "[decode-probe] === {tag} === total_calls={} live_in_flight={live} peak_in_flight={peak}",
        CALLS.load(Ordering::Relaxed)
    );
    for (k, b) in g.iter() {
        if b.n == 0 {
            continue;
        }
        let n = b.n as f64;
        let gpu = if b.gpu_n > 0 {
            b.gpu_ms / b.gpu_n as f64
        } else {
            f64::NAN
        };
        let wall = b.wall_ms / n;
        let ratio = if wall > 0.0 {
            gpu / wall * 100.0
        } else {
            f64::NAN
        };
        eprintln!(
            "[decode-probe] in_flight={k:<3} n={:<6} wall={wall:8.2}ms gpu_graph={gpu:8.2}ms \
             launch={:7.3}ms sync={:8.2}ms dtoh={:6.3}ms argmax={:6.3}ms gpu/wall={ratio:5.1}%",
            b.n,
            b.launch_ms / n,
            b.sync_ms / n,
            b.dtoh_ms / n,
            b.argmax_ms / n,
        );
    }
}
