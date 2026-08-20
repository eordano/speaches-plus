
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const WINDOW: usize = 10;

fn chat_engine_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/oapi/chat_engine")
}

fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut chars = src.chars().peekable();
    let mut in_line = false;
    let mut in_block = false;
    let mut in_str = false;
    while let Some(c) = chars.next() {
        if in_line {
            if c == '\n' {
                in_line = false;
                out.push(c);
            }
            continue;
        }
        if in_block {
            if c == '*' && chars.peek() == Some(&'/') {
                chars.next();
                in_block = false;
            } else if c == '\n' {
                out.push(c);
            }
            continue;
        }
        if in_str {
            if c == '\\' {
                chars.next();
            } else if c == '"' {
                in_str = false;
            } else if c == '\n' {
                out.push(c);
            }
            continue;
        }
        match c {
            '/' if chars.peek() == Some(&'/') => {
                chars.next();
                in_line = true;
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                in_block = true;
            }
            '"' => {
                in_str = true;
                out.push(' ');
            }
            _ => out.push(c),
        }
    }
    out
}

fn decode_loops() -> BTreeMap<String, String> {
    let mut found = BTreeMap::new();
    for entry in std::fs::read_dir(chat_engine_dir()).expect("chat_engine dir").flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();

        if name == "sampling.rs" {
            continue;
        }
        let src = strip_comments(&std::fs::read_to_string(&path).expect("read"));
        if src.contains(".sample(") {
            found.insert(name, src);
        }
    }
    found
}

#[test]
fn every_decode_loop_bails_when_the_sampling_mask_leaves_nothing_legal() {
    let loops = decode_loops();
    assert!(
        loops.len() >= 4,
        "discovery found only {} decode loops ({:?}). qwen.rs, gemma4_loop.rs, \
         gemma4_moe_loop.rs and laguna_loop.rs all call sample(), so a smaller number \
         means the scan broke, not that the loops went away -- and a broken scan passes \
         this test vacuously.",
        loops.len(),
        loops.keys().collect::<Vec<_>>()
    );

    let mut unchecked: Vec<String> = Vec::new();
    let mut total_sites = 0usize;
    for (name, src) in &loops {
        let lines: Vec<&str> = src.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if !line.contains(".sample(") {
                continue;
            }
            total_sites += 1;
            let end = (i + WINDOW).min(lines.len());
            let near = lines[i..end].join("\n");
            if !near.contains("exhausted") {
                unchecked.push(format!("{name}:{}: {}", i + 1, line.trim()));
            }
        }
    }

    assert!(
        total_sites >= 4,
        "found {total_sites} sample() call sites across {} files; the scan is not \
         matching call sites any more",
        loops.len()
    );
    assert!(
        unchecked.is_empty(),
        "these sample() call sites do not inspect SampleOutput::exhausted within {WINDOW} \
         lines:\n  {}\n\nA dead guided decoder masks every candidate to -inf. A loop that \
         samples anyway emits the masked token and continues, answering HTTP 200 with \
         finish_reason=stop and unconstrained text to a caller who asked for a schema. \
         Bail like gemma4_loop.rs does.",
        unchecked.join("\n  ")
    );
}
