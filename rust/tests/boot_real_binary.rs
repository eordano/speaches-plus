use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const SERVER_BIN: &str = env!("CARGO_BIN_EXE_speaches-plus");

const NO_MODEL_BOOT_DEADLINE: Duration = Duration::from_secs(120);

const PASSTHROUGH_ENV_SO_THE_CHILD_FINDS_LOADERS_NOT_MODELS: [&str; 8] = [
    "PATH",
    "HOME",
    "LD_LIBRARY_PATH",
    "VK_ICD_FILENAMES",
    "ORT_DYLIB_PATH",
    "ESPEAK_DATA_PATH",
    "TMPDIR",
    "NV_PUNKT_DATA",
];

const EVERY_ABSENT_MODEL_FILE_MUST_LEAVE_ONE_BOOT_LINE_NAMING_IT: [&str; 4] = [
    "silero_vad.onnx",
    "kokoro-v1.0.onnx",
    "diarizen-segmentation.onnx",
    "wespeaker-resnet293-LM.onnx",
];

struct BootedServer {
    child: Child,
    port: u16,
    log_path: PathBuf,
    _models_dir: Option<tempdir::Scratch>,
}

mod tempdir {
    use std::path::PathBuf;

    pub struct Scratch(pub PathBuf);

    impl Scratch {
        pub fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "speaches-boot-real-{}-{tag}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("create scratch dir");
            Self(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

impl Drop for BootedServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl BootedServer {
    fn log(&self) -> String {
        std::fs::read_to_string(&self.log_path).unwrap_or_default()
    }

    fn wait_until_serving(&mut self, deadline: Duration) -> String {
        let t0 = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().expect("try_wait") {
                panic!(
                    "server exited during boot with {status}; log:\n{}",
                    self.log()
                );
            }
            if let Some((code, body)) = http(self.port, &get_request("/v1/models")) {
                assert_eq!(code, 200, "/v1/models must answer 200 on a booted server; body: {body}");
                return body;
            }
            assert!(
                t0.elapsed() < deadline,
                "server did not answer /v1/models within {deadline:?}; log:\n{}",
                self.log()
            );
            std::thread::sleep(Duration::from_millis(250));
        }
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral")
        .local_addr()
        .expect("local addr")
        .port()
}

fn get_request(path: &str) -> String {
    format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
}

fn post_request(path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn http(port: u16, request: &str) -> Option<(u16, String)> {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_secs(2)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(600))).ok()?;
    s.set_write_timeout(Some(Duration::from_secs(10))).ok()?;
    s.write_all(request.as_bytes()).ok()?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).ok()?;
    let text = String::from_utf8_lossy(&raw).into_owned();
    let status: u16 = text.split_whitespace().nth(1)?.parse().ok()?;
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    Some((status, body))
}

fn spawn_hermetic(tag: &str, extra_env: &[(&str, String)]) -> BootedServer {
    let models = tempdir::Scratch::new(tag);
    let port = free_port();
    let log_path = std::env::temp_dir().join(format!(
        "speaches-boot-real-{}-{tag}.log",
        std::process::id()
    ));
    let log = std::fs::File::create(&log_path).expect("create boot log");
    let mut cmd = Command::new(SERVER_BIN);
    cmd.env_clear();
    for k in PASSTHROUGH_ENV_SO_THE_CHILD_FINDS_LOADERS_NOT_MODELS {
        if let Some(v) = std::env::var_os(k) {
            cmd.env(k, v);
        }
    }
    cmd.env("UVICORN_HOST", "127.0.0.1");
    cmd.env("UVICORN_PORT", port.to_string());
    cmd.env("SPEACHES_PLUS_MODELS", &models.0);
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    cmd.stdout(Stdio::from(log.try_clone().expect("clone log handle")));
    cmd.stderr(Stdio::from(log));
    let child = cmd.spawn().expect("spawn the real server binary");
    BootedServer {
        child,
        port,
        log_path,
        _models_dir: Some(models),
    }
}

#[test]
fn real_binary_boots_binds_serves_v1_models_and_names_every_missing_model_file_loudly() {
    let mut server = spawn_hermetic("nomodels", &[]);
    let body = server.wait_until_serving(NO_MODEL_BOOT_DEADLINE);
    assert!(
        body.contains("\"object\""),
        "/v1/models body is not the OpenAI list shape: {body}"
    );
    let (health, _) = http(server.port, &get_request("/health")).expect("/health reachable");
    assert_eq!(health, 200);

    let log = server.log();
    for file in EVERY_ABSENT_MODEL_FILE_MUST_LEAVE_ONE_BOOT_LINE_NAMING_IT {
        let named_line = log
            .lines()
            .find(|l| l.contains(file) && (l.contains("WARN") || l.contains("ERROR")));
        assert!(
            named_line.is_some(),
            "absent {file} left no WARN boot line naming it (silent downgrade); log:\n{log}"
        );
        let line = named_line.unwrap();
        assert!(
            line.contains("disabled") || line.contains("unavailable") || line.contains("503"),
            "the {file} boot line must say what its absence disables, got: {line}"
        );
    }
    assert!(
        log.contains("NV_CHAT_MODEL_DIR(S) not set"),
        "with no chat env the boot log must say the env is unset; log:\n{log}"
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn real_binary_with_chat_env_set_but_unloadable_never_claims_env_not_set() {
    let bogus = std::env::temp_dir().join(format!(
        "speaches-boot-real-{}-no-such-checkpoint",
        std::process::id()
    ));
    let mut server = spawn_hermetic(
        "envset",
        &[("NV_CHAT_MODEL_DIR", bogus.display().to_string())],
    );
    server.wait_until_serving(NO_MODEL_BOOT_DEADLINE);
    let log = server.log();
    assert!(
        !log.contains("NV_CHAT_MODEL_DIR(S) not set"),
        "NV_CHAT_MODEL_DIR was set, yet boot blamed unset env (the item-18b lie); log:\n{log}"
    );
    assert!(
        log.contains("no chat engine loaded"),
        "boot must state that configured chat env produced no engine; log:\n{log}"
    );
    assert!(
        log.contains("cuda_feature") && log.contains("wgpu_feature"),
        "boot must name the compiled feature set as the place to look; log:\n{log}"
    );
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "boots the real server binary on the wgpu backend with real weights (~22.6 GB \
            Qwen3.8-27B-NVFP4 by default) and runs one completion; set NV_BOOT_E2E_TEST=1 \
            with NV_WGPU_CHAT_MODEL_DIRS pointing at a checkpoint dir; needs an idle GPU"]
fn real_binary_boots_qwen38_wgpu_serves_models_and_completes_one_request() {
    if std::env::var("NV_BOOT_E2E_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_BOOT_E2E_TEST != 1");
        return;
    }
    let dirs = std::env::var("NV_WGPU_CHAT_MODEL_DIRS")
        .expect("NV_BOOT_E2E_TEST=1 requires NV_WGPU_CHAT_MODEL_DIRS");
    let max_seq = std::env::var("NV_WGPU_CHAT_MAX_SEQ").unwrap_or_else(|_| "8192".into());
    let mut server = spawn_hermetic(
        "realweights",
        &[
            ("NV_SERVE_BACKEND", "wgpu".into()),
            ("NV_WGPU_CHAT_MODEL_DIRS", dirs),
            ("NV_WGPU_CHAT_MAX_SEQ", max_seq),
        ],
    );
    let body = server.wait_until_serving(Duration::from_secs(900));
    let v: serde_json::Value = serde_json::from_str(&body).expect("/v1/models is JSON");
    let model_id = v["data"][0]["id"]
        .as_str()
        .unwrap_or_else(|| panic!("no model id in /v1/models: {body}"))
        .to_string();

    let req = serde_json::json!({
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "max_tokens": 64,
        "temperature": 0.0
    });
    let (code, resp) = http(
        server.port,
        &post_request("/v1/chat/completions", &req.to_string()),
    )
    .expect("completion request reachable");
    assert_eq!(code, 200, "completion failed: {resp}");
    let rv: serde_json::Value = serde_json::from_str(&resp).expect("completion body is JSON");
    let content = rv["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    let reasoning = rv["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .unwrap_or_default();
    assert!(
        !content.trim().is_empty() || !reasoning.trim().is_empty(),
        "empty completion: {resp}"
    );

    let log = server.log();
    assert!(
        log.contains("chat models loaded"),
        "boot log must report the loaded chat models; log tail:\n{}",
        &log[log.len().saturating_sub(4000)..]
    );
    assert!(
        !log.contains("NV_CHAT_MODEL_DIR(S) not set"),
        "chat served yet boot claimed no chat env; log tail:\n{}",
        &log[log.len().saturating_sub(4000)..]
    );
}
