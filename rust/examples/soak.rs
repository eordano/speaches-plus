use std::time::Duration;

use clap::Parser;
use speaches_plus::soak::{run_soak, SoakConfig};

#[derive(Debug, Parser)]
#[command(name = "soak", about = "speaches-plus soak harness")]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8000/v1/chat/completions")]
    endpoint: String,

    #[arg(long, default_value_t = 5.0)]
    rps: f64,

    #[arg(long, default_value_t = 60)]
    duration_sec: u64,

    #[arg(long, default_value_t = 30)]
    report_every_sec: u64,

    #[arg(
        long,
        default_value = r#"{"model":"speaches-plus/echo","messages":[{"role":"user","content":"hi"}],"stream":false,"max_tokens":4}"#
    )]
    body: String,

    #[arg(long, default_value_t = 0.0001)]
    max_error_rate: f64,

    #[arg(long, default_value_t = 0.05)]
    max_rss_growth: f64,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let cfg = SoakConfig {
        endpoint_url: args.endpoint,
        rps: args.rps,
        duration: Duration::from_secs(args.duration_sec),
        request_body: args.body,
        content_type: "application/json".to_string(),
        report_every: Duration::from_secs(args.report_every_sec),
        max_error_rate: args.max_error_rate,
        max_rss_growth: args.max_rss_growth,
    };
    eprintln!(
        "[soak] target={} rps={} duration={}s",
        cfg.endpoint_url, cfg.rps, args.duration_sec
    );
    let result = run_soak(cfg).await;
    println!(
        "[soak] DONE elapsed={:.1}s total={} errors={} error_rate={:.4}% rps={:.2} rss_start={}MB rss_end={}MB growth={:.2}% fds_start={} fds_end={}",
        result.elapsed.as_secs_f64(),
        result.total_requests,
        result.total_errors,
        result.error_rate * 100.0,
        result.achieved_rps,
        result.rss_bytes_start / (1024 * 1024),
        result.rss_bytes_end / (1024 * 1024),
        result.rss_growth * 100.0,
        result.fd_count_start,
        result.fd_count_end,
    );
    if let Some(reason) = result.fail_reason {
        eprintln!("[soak] FAIL: {reason}");
        std::process::exit(1);
    }
    eprintln!("[soak] PASS");
}
