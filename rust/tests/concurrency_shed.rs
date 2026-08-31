use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use speaches_plus::oapi::gate::SurfaceGate;

const A_SHED_REQUEST_MUST_NEVER_REACH_THE_MODEL: &str =
    "the stub backend is entered once per 200 and never once per 503; if a shed request had \
     reached it the permit would have been taken after the work rather than before it, and the \
     semaphore would be decorative";

const A_WAITER_INSIDE_SPAWN_BLOCKING_HOLDS_A_POOL_THREAD: &str =
    "every model lock in rust/src/oapi is taken inside the blocking closure, so a surface that \
     queues for its turn there parks a blocking-pool thread per waiter and one hammered \
     endpoint starves every other surface at tokio's 512-thread default; taking the permit \
     first bounds a surface's pool occupancy to its permit count";

const PERMITS_BOUND_CONCURRENT_ENTRIES_NOT_MERELY_CONCURRENT_REQUESTS: &str =
    "a gate that admits N requests but lets N+1 into the model is a gate in name only";

const BLOCKING_THREADS_IN_THE_STARVATION_PROBE: usize = 2;
const HAMMER_REQUESTS: usize = 8;
const HAMMER_WORK_MS: u64 = 120;
const PROBE_SETTLE_MS: u64 = 150;

#[derive(Default)]
struct StubModel {
    lock: Mutex<()>,
    entered: AtomicUsize,
    live: AtomicUsize,
    peak: AtomicUsize,
}

impl StubModel {
    fn enter(&self, work: Duration) {
        self.entered.fetch_add(1, Ordering::SeqCst);
        let live = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(live, Ordering::SeqCst);
        let _held = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        std::thread::sleep(work);
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn permit_then_blocking(
    gate: Arc<SurfaceGate>,
    model: Arc<StubModel>,
    work: Duration,
) -> StatusCode {
    let permit = match gate.acquire().await {
        Ok(p) => p,
        Err(busy) => return busy.into_response().status(),
    };
    let m = model.clone();
    tokio::task::spawn_blocking(move || m.enter(work))
        .await
        .expect("blocking join");
    drop(permit);
    StatusCode::OK
}

async fn model_lock_inside_the_blocking_closure(model: Arc<StubModel>, work: Duration) {
    let _ = tokio::task::spawn_blocking(move || model.enter(work)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_shed_request_never_enters_the_backend() {
    let gate = Arc::new(SurfaceGate::new("/v1/stub", 2, 80));
    let model = Arc::new(StubModel::default());
    let work = Duration::from_millis(400);

    let mut inflight = Vec::new();
    for _ in 0..12 {
        let (g, m) = (gate.clone(), model.clone());
        inflight.push(tokio::spawn(
            async move { permit_then_blocking(g, m, work).await },
        ));
    }
    let mut ok = 0usize;
    let mut shed = 0usize;
    for h in inflight {
        match h.await.expect("join") {
            StatusCode::OK => ok += 1,
            StatusCode::SERVICE_UNAVAILABLE => shed += 1,
            other => panic!("unexpected status {other}"),
        }
    }

    assert_eq!(ok + shed, 12);
    assert!(ok >= 2, "the gate admitted {ok} of 12 with 2 permits");
    assert!(
        shed >= 8,
        "only {shed} of 12 shed against an 80 ms window and 400 ms of work: the queue deadline \
         is not being applied"
    );
    assert_eq!(
        model.entered.load(Ordering::SeqCst),
        ok,
        "{A_SHED_REQUEST_MUST_NEVER_REACH_THE_MODEL}"
    );
    let peak = model.peak.load(Ordering::SeqCst);
    assert!(
        peak <= gate.permits(),
        "{peak} concurrent entries against {} permits. \
         {PERMITS_BOUND_CONCURRENT_ENTRIES_NOT_MERELY_CONCURRENT_REQUESTS}",
        gate.permits()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_saturated_surface_answers_503_with_the_surface_busy_code() {
    let gate = SurfaceGate::new("/v1/stub", 1, 40);
    let _held = gate.acquire().await.expect("hold the only permit");
    let busy = gate.acquire().await.expect_err("second must shed");
    assert_eq!(busy.surface, "/v1/stub");
    assert_eq!(busy.permits, 1);

    let resp = busy.into_response();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = axum::body::to_bytes(resp.into_body(), 64 * 1024)
        .await
        .expect("shed body");
    let v: serde_json::Value = serde_json::from_slice(&body).expect("shed body is json");
    assert_eq!(
        v["error"]["code"], "surface_busy",
        "a shed must be machine-distinguishable from every other 503: {v}"
    );
    let message = v["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("/v1/stub") && message.contains("at capacity"),
        "the shed message must name the surface and the reason: {message:?}"
    );
}

fn probe_latency_under_hammer(permit_before_spawn_blocking: bool) -> Duration {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .max_blocking_threads(BLOCKING_THREADS_IN_THE_STARVATION_PROBE)
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(async move {
        let gate = Arc::new(SurfaceGate::new("/v1/hammered", 1, 60_000));
        let model = Arc::new(StubModel::default());
        let work = Duration::from_millis(HAMMER_WORK_MS);
        let mut hammer = Vec::new();
        for _ in 0..HAMMER_REQUESTS {
            let (g, m) = (gate.clone(), model.clone());
            hammer.push(tokio::spawn(async move {
                if permit_before_spawn_blocking {
                    permit_then_blocking(g, m, work).await;
                } else {
                    model_lock_inside_the_blocking_closure(m, work).await;
                }
            }));
        }
        tokio::time::sleep(Duration::from_millis(PROBE_SETTLE_MS)).await;
        let t0 = Instant::now();
        tokio::task::spawn_blocking(|| ()).await.expect("probe join");
        let probe = t0.elapsed();
        for h in hammer {
            let _ = h.await;
        }
        probe
    })
}

#[test]
fn taking_the_permit_before_spawn_blocking_keeps_an_unrelated_surface_answering() {
    let ordered = probe_latency_under_hammer(true);
    let inverted = probe_latency_under_hammer(false);
    eprintln!("unrelated spawn_blocking latency: permit-first {ordered:?}, blocking-first {inverted:?}");
    assert!(
        ordered < Duration::from_millis(HAMMER_WORK_MS),
        "an unrelated spawn_blocking waited {ordered:?} even though the hammered surface may \
         hold at most one pool thread. {A_WAITER_INSIDE_SPAWN_BLOCKING_HOLDS_A_POOL_THREAD}"
    );
    assert!(
        inverted >= Duration::from_millis(HAMMER_WORK_MS * 2),
        "the inverted ordering did not starve the pool ({inverted:?}); with only {} blocking \
         threads and {HAMMER_REQUESTS} unpermitted entrants the probe must queue behind them, \
         so this differential is measuring nothing",
        BLOCKING_THREADS_IN_THE_STARVATION_PROBE
    );
    assert!(
        inverted > ordered * 4,
        "no separation between the two orderings: {ordered:?} permit-first vs {inverted:?} \
         blocking-first. {A_WAITER_INSIDE_SPAWN_BLOCKING_HOLDS_A_POOL_THREAD}"
    );
}

fn blank_cfg_test_modules(src: &str) -> String {
    const MARK: &str = "#[cfg(test)]";
    let mut out = src.to_string();
    let mut from = 0usize;
    loop {
        let Some(rel) = out[from..].find(MARK) else {
            return out;
        };
        let at = from + rel;
        let after = at + MARK.len();
        let head = out[after..].trim_start();
        let head = head.strip_prefix("pub ").unwrap_or(head).trim_start();
        if !head.starts_with("mod ") {
            from = after;
            continue;
        }
        let Some(rel_open) = out[after..].find('{') else {
            return out;
        };
        let open = after + rel_open;
        let mut depth = 0i32;
        let mut close = None;
        for (i, c) in out[open..].char_indices() {
            if c == '{' {
                depth += 1;
            } else if c == '}' {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + i);
                    break;
                }
            }
        }
        let Some(close) = close else {
            return out;
        };
        out = format!("{}{}", &out[..at], &out[close + 1..]);
        from = at;
    }
}

fn oapi_rust_files() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        entries.sort();
        for p in entries {
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                out.push(p);
            }
        }
    }
    let mut out = Vec::new();
    walk(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src/oapi"),
        &mut out,
    );
    out
}

fn first_index_of_any(haystack: &str, needles: &[&str]) -> Option<usize> {
    needles.iter().filter_map(|n| haystack.find(n)).min()
}

fn all_indices_of_any(haystack: &str, needles: &[&str]) -> Vec<usize> {
    let mut out = Vec::new();
    for n in needles {
        let mut from = 0usize;
        while let Some(rel) = haystack[from..].find(n) {
            out.push(from + rel);
            from += rel + n.len();
        }
    }
    out.sort_unstable();
    out
}

const ACQUIRE: [&str; 3] = [
    ".acquire_with_deadline(",
    ".acquire()",
    "acquire_permit_bounded(",
];

fn ordering_violations(name: &str, live: &str) -> Vec<String> {
    let mut out = Vec::new();
    let Some(first_blocking) = live.find("spawn_blocking") else {
        return out;
    };
    let Some(first_acquire) = first_index_of_any(live, &ACQUIRE) else {
        out.push(format!(
            "{name}: reaches for SurfaceGate but never acquires a permit"
        ));
        return out;
    };
    if first_acquire > first_blocking {
        out.push(format!(
            "{name}: spawn_blocking at byte {first_blocking} precedes the permit at byte \
             {first_acquire}"
        ));
    }
    let late: Vec<usize> = all_indices_of_any(live, &ACQUIRE)
        .into_iter()
        .filter(|a| *a > first_blocking)
        .collect();
    if !late.is_empty() {
        out.push(format!(
            "{name}: {} permit acquisition(s) sit after the first spawn_blocking (bytes {late:?})",
            late.len()
        ));
    }
    out
}

#[test]
fn the_ordering_scanner_catches_a_permit_taken_inside_the_blocking_closure() {
    let inverted = "async fn handle(rt: State<R>) -> Response {\n    \
                    let joined = tokio::task::spawn_blocking(move || {\n        \
                    let _permit = gate.acquire();\n        engine.recognize(&bytes)\n    \
                    }).await;\n}\n";
    let found = ordering_violations("inverted.rs", inverted);
    assert!(
        !found.is_empty(),
        "the scanner accepted a handler that acquires its permit inside spawn_blocking"
    );

    let ordered = "async fn handle(rt: State<R>) -> Response {\n    \
                   let _permit = gate.acquire_with_deadline(client).await;\n    \
                   let joined = tokio::task::spawn_blocking(move || engine.recognize(&bytes))\n    \
                   .await;\n}\n";
    assert!(
        ordering_violations("ordered.rs", ordered).is_empty(),
        "the scanner rejected the production ordering, so it would be red on main and could \
         never catch a real inversion"
    );

    let ungated = "async fn handle() { tokio::task::spawn_blocking(|| work()).await; }\n";
    assert_eq!(
        ordering_violations("ungated.rs", ungated).len(),
        1,
        "a file that names SurfaceGate but never takes a permit must be reported, not skipped"
    );
}

#[test]
fn every_gated_oapi_surface_takes_its_permit_before_spawn_blocking() {
    let files = oapi_rust_files();
    assert!(
        files.len() >= 20,
        "only {} files under src/oapi; the walk is broken and this gate proves nothing",
        files.len()
    );

    let mut gated: BTreeSet<String> = BTreeSet::new();
    let mut violations: Vec<String> = Vec::new();
    for f in &files {
        let Ok(raw) = std::fs::read_to_string(f) else {
            continue;
        };
        let live = blank_cfg_test_modules(&raw);
        if !live.contains("spawn_blocking") {
            continue;
        }
        if !(live.contains("SurfaceGate") || live.contains("acquire_permit_bounded")) {
            continue;
        }
        let name = f
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        gated.insert(name.clone());
        violations.extend(ordering_violations(&name, &live));
    }

    for expected in ["ocr.rs", "text_embeddings.rs"] {
        assert!(
            gated.contains(expected),
            "src/oapi/{expected} no longer reads as a permit-gated blocking surface; the scan \
             found {gated:?}. Either the gate moved or this walk stopped seeing it, and either \
             way the ordering is unchecked there"
        );
    }
    assert!(
        violations.is_empty(),
        "{A_WAITER_INSIDE_SPAWN_BLOCKING_HOLDS_A_POOL_THREAD}\n  {}",
        violations.join("\n  ")
    );
}

#[test]
fn the_cfg_test_blanker_hides_test_permits_without_hiding_production_ones() {
    let src = "fn h() { let p = g.acquire().await; spawn_blocking(|| {}); }\n\
               #[cfg(test)]\nmod tests {\n  fn t() { let q = g.acquire().await; }\n}\n";
    let live = blank_cfg_test_modules(src);
    assert!(live.contains("spawn_blocking"));
    assert_eq!(
        live.matches(".acquire()").count(),
        1,
        "the blanker left a test-module permit in the live text: {live:?}"
    );

    let nested = "#[cfg(test)]\nmod tests {\n  mod inner { fn f() {} }\n}\nfn after() {}\n";
    assert_eq!(
        blank_cfg_test_modules(nested).trim(),
        "fn after() {}",
        "the blanker did not brace-match past a nested module"
    );

    let cfg_test_fn = "#[cfg(test)]\nfn helper() {}\nfn after() {}\n#[cfg(test)]\nmod t { fn x() {} }\n";
    let live = blank_cfg_test_modules(cfg_test_fn);
    assert!(
        live.contains("fn helper") && live.contains("fn after"),
        "a #[cfg(test)] item that is not a module must not truncate the file: {live:?}"
    );
    assert!(
        !live.contains("mod t"),
        "the blanker stopped at the first non-module #[cfg(test)] and missed the module after \
         it: {live:?}"
    );
}
