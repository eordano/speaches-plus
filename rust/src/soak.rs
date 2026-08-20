use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct SoakConfig {
    pub endpoint_url: String,
    pub rps: f64,
    pub duration: Duration,
    pub request_body: String,
    pub content_type: String,
    pub report_every: Duration,
    pub max_error_rate: f64,
    pub max_rss_growth: f64,
}

impl Default for SoakConfig {
    fn default() -> Self {
        Self {
            endpoint_url: String::new(),
            rps: 1.0,
            duration: Duration::from_secs(60),
            request_body: r#"{"model":"speaches-plus/echo","messages":[{"role":"user","content":"hi"}],"stream":false,"max_tokens":4}"#.to_string(),
            content_type: "application/json".to_string(),
            report_every: Duration::from_secs(30),
            max_error_rate: 0.0001,
            max_rss_growth: 0.05,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SoakResult {
    pub elapsed: Duration,
    pub total_requests: u64,
    pub total_errors: u64,
    pub error_rate: f64,
    pub achieved_rps: f64,
    pub rss_bytes_start: u64,
    pub rss_bytes_end: u64,
    pub rss_growth: f64,
    pub fd_count_start: u64,
    pub fd_count_end: u64,
    pub passed: bool,
    pub fail_reason: Option<String>,
}

pub fn read_rss_bytes() -> u64 {
    let txt = match std::fs::read_to_string("/proc/self/status") {
        Ok(s) => s,
        Err(_) => return 0,
    };
    for line in txt.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if let (Some(num), Some(unit)) = (parts.first(), parts.get(1)) {
                let kb: u64 = num.parse().unwrap_or(0);
                let mul: u64 = match *unit {
                    "kB" => 1024,
                    "MB" => 1024 * 1024,
                    _ => 1024,
                };
                return kb * mul;
            }
        }
    }
    0
}

pub fn read_open_fd_count() -> u64 {
    let path = std::path::Path::new("/proc/self/fd");
    match std::fs::read_dir(path) {
        Ok(it) => it.count() as u64,
        Err(_) => 0,
    }
}

pub fn read_gpu_used_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let txt = String::from_utf8_lossy(&out.stdout);
    txt.lines()
        .next()
        .and_then(|l| l.trim().parse::<u64>().ok())
}

#[derive(Debug, Clone, Default)]
pub struct LatencyStats {
    samples_ms: Vec<f64>,
}

impl LatencyStats {
    pub fn new() -> Self {
        Self {
            samples_ms: Vec::new(),
        }
    }

    pub fn record(&mut self, d: Duration) {
        self.samples_ms.push(d.as_secs_f64() * 1000.0);
    }

    pub fn record_ms(&mut self, ms: f64) {
        self.samples_ms.push(ms);
    }

    pub fn merge(&mut self, other: &LatencyStats) {
        self.samples_ms.extend_from_slice(&other.samples_ms);
    }

    pub fn count(&self) -> usize {
        self.samples_ms.len()
    }

    pub fn percentile(&self, p: f64) -> f64 {
        if self.samples_ms.is_empty() {
            return 0.0;
        }
        let mut v = self.samples_ms.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p = p.clamp(0.0, 100.0);
        let rank = (p / 100.0) * ((v.len() - 1) as f64);
        let lo = rank.floor() as usize;
        let hi = rank.ceil() as usize;
        if lo == hi {
            v[lo]
        } else {
            let frac = rank - lo as f64;
            v[lo] * (1.0 - frac) + v[hi] * frac
        }
    }

    pub fn min(&self) -> f64 {
        if self.samples_ms.is_empty() {
            return 0.0;
        }
        self.samples_ms
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    }

    pub fn max(&self) -> f64 {
        self.samples_ms.iter().cloned().fold(0.0, f64::max)
    }

    pub fn mean(&self) -> f64 {
        if self.samples_ms.is_empty() {
            return 0.0;
        }
        self.samples_ms.iter().sum::<f64>() / self.samples_ms.len() as f64
    }

    pub fn summary_line(&self, label: &str) -> String {
        if self.samples_ms.is_empty() {
            return format!("{label}: (no samples)");
        }
        format!(
            "{label}: n={} min={:.1} p50={:.1} p95={:.1} p99={:.1} max={:.1} mean={:.1} ms",
            self.count(),
            self.min(),
            self.percentile(50.0),
            self.percentile(95.0),
            self.percentile(99.0),
            self.max(),
            self.mean(),
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct ResourceSample {
    pub t_secs: f64,
    pub rss_mib: u64,
    pub gpu_used_mib: Option<u64>,
    pub fds: u64,
}

#[derive(Debug, Default)]
pub struct ResourceTrack {
    pub samples: Vec<ResourceSample>,
}

impl ResourceTrack {
    pub fn capture(&mut self, t_secs: f64) {
        self.samples.push(ResourceSample {
            t_secs,
            rss_mib: read_rss_bytes() / (1024 * 1024),
            gpu_used_mib: read_gpu_used_mib(),
            fds: read_open_fd_count(),
        });
    }

    pub fn rss_growth_mib(&self) -> i64 {
        match (self.samples.first(), self.samples.last()) {
            (Some(a), Some(b)) => b.rss_mib as i64 - a.rss_mib as i64,
            _ => 0,
        }
    }

    pub fn gpu_growth_mib(&self) -> Option<i64> {
        let first = self.samples.iter().find_map(|s| s.gpu_used_mib)?;
        let last = self.samples.iter().rev().find_map(|s| s.gpu_used_mib)?;
        Some(last as i64 - first as i64)
    }

    pub fn monotonic_rss_runs(&self) -> usize {
        let mut longest = 0usize;
        let mut cur = 0usize;
        for w in self.samples.windows(2) {
            if w[1].rss_mib > w[0].rss_mib {
                cur += 1;
                longest = longest.max(cur);
            } else {
                cur = 0;
            }
        }
        longest
    }

    pub fn print_table(&self) {
        println!(
            "  {:>8}  {:>10}  {:>10}  {:>6}",
            "t(s)", "rss(MiB)", "gpu(MiB)", "fds"
        );
        for s in &self.samples {
            let gpu = s
                .gpu_used_mib
                .map(|g| g.to_string())
                .unwrap_or_else(|| "n/a".to_string());
            println!(
                "  {:>8.1}  {:>10}  {:>10}  {:>6}",
                s.t_secs, s.rss_mib, gpu, s.fds
            );
        }
    }
}

#[derive(Default)]
struct Counters {
    total: AtomicU64,
    errors: AtomicU64,
}

pub async fn run_soak(cfg: SoakConfig) -> SoakResult {
    let counters = Arc::new(Counters::default());
    let rss_start = read_rss_bytes();
    let fd_start = read_open_fd_count();
    let start = Instant::now();
    let deadline = start + cfg.duration;
    let mut last_report = start;
    let interval_ns = if cfg.rps > 0.0 {
        (1_000_000_000.0 / cfg.rps) as u64
    } else {
        1_000_000_000
    };

    let client = reqwest::Client::builder()
        .pool_idle_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let mut next_at = start;
    while Instant::now() < deadline {
        let now = Instant::now();
        if now < next_at {
            tokio::time::sleep(next_at - now).await;
        }
        let counters_c = counters.clone();
        let url = cfg.endpoint_url.clone();
        let body = cfg.request_body.clone();
        let ct = cfg.content_type.clone();
        let client_c = client.clone();
        tokio::spawn(async move {
            counters_c.total.fetch_add(1, Ordering::Relaxed);
            let res = client_c
                .post(&url)
                .header("content-type", &ct)
                .body(body)
                .send()
                .await;
            match res {
                Ok(r) if r.status().is_success() => {
                    let _ = r.bytes().await;
                }
                _ => {
                    counters_c.errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        next_at += Duration::from_nanos(interval_ns);

        if last_report.elapsed() >= cfg.report_every {
            let total = counters.total.load(Ordering::Relaxed);
            let errors = counters.errors.load(Ordering::Relaxed);
            let rss = read_rss_bytes();
            let fds = read_open_fd_count();
            let elapsed = start.elapsed().as_secs_f64();
            let rate = if total > 0 {
                errors as f64 / total as f64
            } else {
                0.0
            };
            println!(
                "[soak] t={elapsed:.1}s reqs={total} errs={errors} err_rate={:.4}% rps={:.2} rss={}MB fds={}",
                rate * 100.0,
                total as f64 / elapsed.max(0.001),
                rss / (1024 * 1024),
                fds,
            );
            last_report = Instant::now();
        }
    }

    let settle_deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let outstanding = counters.total.load(Ordering::Relaxed) as i64
            - counters.errors.load(Ordering::Relaxed) as i64;
        let _ = outstanding;
        if Instant::now() >= settle_deadline {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let elapsed = start.elapsed();
    let total = counters.total.load(Ordering::Relaxed);
    let errors = counters.errors.load(Ordering::Relaxed);
    let rss_end = read_rss_bytes();
    let fd_end = read_open_fd_count();
    let error_rate = if total > 0 {
        errors as f64 / total as f64
    } else {
        0.0
    };
    let rss_growth = if rss_start > 0 {
        (rss_end as f64 - rss_start as f64) / rss_start as f64
    } else {
        0.0
    };
    let mut fail_reason = None;
    if error_rate > cfg.max_error_rate {
        fail_reason = Some(format!(
            "error_rate {:.4}% exceeds max {:.4}%",
            error_rate * 100.0,
            cfg.max_error_rate * 100.0
        ));
    } else if rss_growth > cfg.max_rss_growth {
        fail_reason = Some(format!(
            "rss_growth {:.2}% exceeds max {:.2}%",
            rss_growth * 100.0,
            cfg.max_rss_growth * 100.0
        ));
    }
    let passed = fail_reason.is_none();
    SoakResult {
        elapsed,
        total_requests: total,
        total_errors: errors,
        error_rate,
        achieved_rps: total as f64 / elapsed.as_secs_f64().max(0.001),
        rss_bytes_start: rss_start,
        rss_bytes_end: rss_end,
        rss_growth,
        fd_count_start: fd_start,
        fd_count_end: fd_end,
        passed,
        fail_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_rss_bytes_returns_nonzero_under_linux() {
        let rss = read_rss_bytes();
        if cfg!(target_os = "linux") {
            assert!(rss > 0, "expected RSS > 0 on Linux, got {rss}");
        }
    }

    #[test]
    fn read_open_fd_count_returns_at_least_three() {
        let n = read_open_fd_count();
        if cfg!(target_os = "linux") {
            assert!(n >= 3, "expected >= 3 fds (stdin/stdout/stderr), got {n}");
        }
    }

    #[test]
    fn soak_config_default_is_sensible() {
        let cfg = SoakConfig::default();
        assert!(cfg.rps > 0.0);
        assert!(cfg.duration.as_secs() > 0);
        assert!(cfg.max_error_rate > 0.0);
        assert!(cfg.max_rss_growth > 0.0);
    }

    #[test]
    fn latency_percentiles_are_monotonic_and_correct() {
        let mut s = LatencyStats::new();
        for ms in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0] {
            s.record_ms(ms);
        }
        assert_eq!(s.count(), 10);
        assert!((s.min() - 10.0).abs() < 1e-9);
        assert!((s.max() - 100.0).abs() < 1e-9);
        let p50 = s.percentile(50.0);
        let p95 = s.percentile(95.0);
        let p99 = s.percentile(99.0);
        assert!(p50 <= p95 && p95 <= p99, "p50={p50} p95={p95} p99={p99}");
        assert!((p50 - 55.0).abs() < 1e-6, "p50 interpolated = {p50}");
        assert!((s.mean() - 55.0).abs() < 1e-9);
    }

    #[test]
    fn latency_empty_is_safe() {
        let s = LatencyStats::new();
        assert_eq!(s.count(), 0);
        assert_eq!(s.percentile(99.0), 0.0);
        assert_eq!(s.min(), 0.0);
        assert_eq!(s.max(), 0.0);
        assert_eq!(s.mean(), 0.0);
        assert!(s.summary_line("x").contains("no samples"));
    }

    #[test]
    fn latency_merge_combines_samples() {
        let mut a = LatencyStats::new();
        a.record_ms(1.0);
        let mut b = LatencyStats::new();
        b.record_ms(2.0);
        b.record_ms(3.0);
        a.merge(&b);
        assert_eq!(a.count(), 3);
        assert!((a.mean() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn resource_track_growth_and_monotonic_detection() {
        let mut t = ResourceTrack::default();
        t.samples = vec![
            ResourceSample {
                t_secs: 0.0,
                rss_mib: 100,
                gpu_used_mib: Some(1000),
                fds: 10,
            },
            ResourceSample {
                t_secs: 1.0,
                rss_mib: 110,
                gpu_used_mib: Some(1010),
                fds: 10,
            },
            ResourceSample {
                t_secs: 2.0,
                rss_mib: 120,
                gpu_used_mib: Some(1020),
                fds: 11,
            },
            ResourceSample {
                t_secs: 3.0,
                rss_mib: 115,
                gpu_used_mib: Some(1020),
                fds: 11,
            },
        ];
        assert_eq!(t.rss_growth_mib(), 15);
        assert_eq!(t.gpu_growth_mib(), Some(20));
        assert_eq!(t.monotonic_rss_runs(), 2);
    }

    #[test]
    fn resource_track_handles_missing_gpu() {
        let mut t = ResourceTrack::default();
        t.samples = vec![
            ResourceSample {
                t_secs: 0.0,
                rss_mib: 100,
                gpu_used_mib: None,
                fds: 10,
            },
            ResourceSample {
                t_secs: 1.0,
                rss_mib: 100,
                gpu_used_mib: None,
                fds: 10,
            },
        ];
        assert_eq!(t.gpu_growth_mib(), None);
        assert_eq!(t.rss_growth_mib(), 0);
    }
}
