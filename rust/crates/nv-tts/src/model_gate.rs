use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::tokenizer::{QWEN3_TTS_SNAPSHOTS_UNDER_HOME, QWEN3_TTS_SNAPSHOT_KEY_FILE};

pub const MODEL_GATED_TEST_COUNT: usize = 7;

const CALL_SITE_MARKER: &str = "model_gate::require(";

const GATED_SOURCES: [(&str, &str); 4] = [
    ("codec_decoder.rs", include_str!("codec_decoder.rs")),
    ("code_predictor.rs", include_str!("code_predictor.rs")),
    ("tokenizer.rs", include_str!("tokenizer.rs")),
    ("vocoder_loader.rs", include_str!("vocoder_loader.rs")),
];

static EXERCISED: AtomicUsize = AtomicUsize::new(0);
static SKIPPED: AtomicUsize = AtomicUsize::new(0);

fn loud(line: String) {
    let mut err = std::io::stderr().lock();
    let _ = writeln!(err, "{line}");
    let _ = err.flush();
}

enum Gate {
    Ready(PathBuf),
    Absent(String),
}

fn resolve() -> Gate {
    let Some(home) = std::env::var_os("HOME") else {
        return Gate::Absent("HOME is unset".into());
    };
    let root = PathBuf::from(home).join(QWEN3_TTS_SNAPSHOTS_UNDER_HOME);
    let entries = match std::fs::read_dir(&root) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Gate::Absent(format!("{} does not exist", root.display()))
        }
        Err(e) => panic!(
            "{} exists but cannot be read ({e}); a model directory that is present and unreadable \
             is a broken cache, and skipping it would hide the breakage behind a green suite",
            root.display()
        ),
    };
    let mut snapshots: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry.unwrap_or_else(|e| {
            panic!(
                "{} exists but an entry under it cannot be read ({e}); a model directory that is \
                 present and unreadable is a broken cache, not an absent one",
                root.display()
            )
        });
        let p = entry.path();
        if p.is_dir() {
            snapshots.push(p);
        }
    }
    if snapshots.is_empty() {
        return Gate::Absent(format!("no snapshot directory under {}", root.display()));
    }
    snapshots.sort();
    for p in &snapshots {
        if p.join(QWEN3_TTS_SNAPSHOT_KEY_FILE).is_file() {
            return Gate::Ready(p.clone());
        }
    }
    panic!(
        "{} holds {} snapshot directory/ies but none carries a readable \
         {QWEN3_TTS_SNAPSHOT_KEY_FILE} (dangling blob symlinks, an interrupted download, or lost \
         permissions); a present but unreadable model directory must fail, not skip",
        root.display(),
        snapshots.len()
    );
}

pub fn require(test: &str) -> Option<PathBuf> {
    match resolve() {
        Gate::Ready(dir) => {
            let run = EXERCISED.fetch_add(1, Ordering::SeqCst) + 1;
            let skipped = SKIPPED.load(Ordering::SeqCst);
            loud(format!(
                "[nv-tts model-gate] RUN  {test} (exercised {run}, skipped {skipped}, of \
                 {MODEL_GATED_TEST_COUNT} model-gated)"
            ));
            Some(dir)
        }
        Gate::Absent(why) => {
            let skipped = SKIPPED.fetch_add(1, Ordering::SeqCst) + 1;
            let run = EXERCISED.load(Ordering::SeqCst);
            loud(format!(
                "[nv-tts model-gate] SKIP {test} -- {why} (exercised {run}, skipped {skipped}, of \
                 {MODEL_GATED_TEST_COUNT} model-gated)"
            ));
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_model_gated_test_is_on_the_roster() {
        let mut total = 0usize;
        let mut per_file: Vec<(&str, usize)> = Vec::new();
        for (name, src) in GATED_SOURCES {
            let n = src.matches(CALL_SITE_MARKER).count();
            total += n;
            per_file.push((name, n));
        }
        assert_eq!(
            total, MODEL_GATED_TEST_COUNT,
            "MODEL_GATED_TEST_COUNT is {MODEL_GATED_TEST_COUNT} but {total} tests call the gate \
             ({per_file:?}); the count printed next to every skip would be a lie"
        );

        match resolve() {
            Gate::Ready(dir) => loud(format!(
                "[nv-tts model-gate] snapshot {} is usable: the {MODEL_GATED_TEST_COUNT} \
                 model-gated tests run for real",
                dir.display()
            )),
            Gate::Absent(why) => loud(format!(
                "[nv-tts model-gate] no usable snapshot ({why}): all {MODEL_GATED_TEST_COUNT} \
                 model-gated tests are skipped, and the suite below proves nothing about them"
            )),
        }
    }
}
