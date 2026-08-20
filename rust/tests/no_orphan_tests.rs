
use std::path::{Path, PathBuf};

fn asserts(body: &str) -> bool {
    body.contains("assert!")
        || body.contains("assert_eq!")
        || body.contains("assert_ne!")
        || body.contains("panic!")
}

fn is_test_attr(line: &str) -> bool {
    let t = line.trim();
    t.starts_with("#[test")
        || t.starts_with("#[tokio::test")
        || t.starts_with("#[rstest")
        || t.starts_with("#[test_case")
        || t.starts_with("#[proptest")
        || t.starts_with("#[bench")
        || t.starts_with("#[ignore")
        || t.starts_with("#[should_panic")
        || t.starts_with("#[cfg(")
        || t.starts_with("#[allow(")
}

fn rust_files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(root) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
            if name == "target" || name == "vendor" || name.starts_with('.') {
                continue;
            }
            rust_files(&p, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(p);
        }
    }
}

fn orphans(src: &str, in_tests_file: bool) -> Vec<String> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    let mut test_mod_depth: Option<i32> = None;
    let mut depth: i32 = 0;

    for (i, raw) in lines.iter().enumerate() {
        let line = *raw;
        if test_mod_depth.is_none() {
            let t = line.trim();
            if t.starts_with("mod ") && t.contains("test") && t.ends_with('{') {
                test_mod_depth = Some(depth);
            }
        }
        depth += line.matches('{').count() as i32;
        depth -= line.matches('}').count() as i32;
        if let Some(d) = test_mod_depth {
            if depth <= d {
                test_mod_depth = None;
            }
        }
        let in_test_ctx = in_tests_file || test_mod_depth.is_some();
        if !in_test_ctx {
            continue;
        }

        let t = line.trim();
        let zero_arg = (t.starts_with("fn ") || t.starts_with("async fn ")) && t.contains("()");
        if !zero_arg {
            continue;
        }
        let name = t
            .trim_start_matches("async ")
            .trim_start_matches("fn ")
            .split('(')
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        if name == "main" {
            continue;
        }

        let mut body = String::new();
        let mut d: i32 = 0;
        let mut started = false;
        for l in lines.iter().skip(i) {
            body.push_str(l);
            body.push('\n');
            d += l.matches('{').count() as i32;
            d -= l.matches('}').count() as i32;
            if l.contains('{') {
                started = true;
            }
            if started && d <= 0 {
                break;
            }
        }
        if !asserts(&body) {
            continue;
        }

        let mut attached = false;
        let mut j = i;
        while j > 0 {
            j -= 1;
            let prev = lines[j].trim();
            if prev.is_empty() || prev.starts_with("///") || prev.starts_with("//") {
                continue;
            }
            if is_test_attr(prev) {
                if prev.starts_with("#[test") || prev.starts_with("#[tokio::test")
                    || prev.starts_with("#[rstest") || prev.starts_with("#[test_case")
                    || prev.starts_with("#[proptest") || prev.starts_with("#[bench")
                {
                    attached = true;
                    break;
                }
                continue;
            }
            let ends_a_preceding_item_so_cannot_be_an_attr_continuation =
                prev.starts_with("#[")
                    || prev.ends_with('{')
                    || prev.ends_with('}')
                    || prev.ends_with(';');
            if ends_a_preceding_item_so_cannot_be_an_attr_continuation {
                break;
            }
        }
        if attached {
            continue;
        }

        let referenced = src
            .match_indices(name.as_str())
            .filter(|(at, _)| {
                let before = src[..*at].trim_end();
                !before.ends_with("fn")
            })
            .count()
            > 0;
        if !referenced {
            out.push(format!("{name} (line {})", i + 1));
        }
    }
    out
}

fn repo_rust_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn no_test_is_silently_detached_from_its_attribute() {
    let root = repo_rust_root();
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    rust_files(&root.join("tests"), &mut files);
    rust_files(&root.join("crates"), &mut files);
    assert!(
        files.len() > 200,
        "only {} rust files found under {}; the walk is broken and this gate is \
         proving nothing",
        files.len(),
        root.display()
    );

    let mut found: Vec<String> = Vec::new();
    for f in &files {
        let Ok(src) = std::fs::read_to_string(f) else {
            continue;
        };
        if f.file_name().and_then(|n| n.to_str()) == Some("no_orphan_tests.rs") {
            continue;
        }
        let in_tests_file = f.components().any(|c| c.as_os_str() == "tests");
        for o in orphans(&src, in_tests_file) {
            found.push(format!("{}: {o}", f.display()));
        }
    }
    assert!(
        found.is_empty(),
        "these zero-argument, assertion-bearing functions sit in test context \
         with no test attribute, so they are compiled and never run:\n  {}\n\n\
         If one is a fixture, give it a parameter or drop its assertions. If it \
         is a test, its `#[test]` was probably captured by a neighbour.",
        found.join("\n  ")
    );
}

#[test]
fn the_scanner_catches_an_attribute_captured_by_a_neighbour() {
    let src = r#"
mod tests {
    #[test]
    /// An edit inserted this test between the attribute and its function.
    #[test]
    fn the_inserted_one() {
        assert!(true);
    }

    fn session_slot_is_released_by_drop_session() {
        assert_eq!(1, 1);
    }

    fn a_fixture(n: usize) -> usize {
        assert!(n > 0);
        n
    }
}
"#;
    let found = orphans(src, false);
    assert!(
        found.iter().any(|f| f.contains("session_slot_is_released_by_drop_session")),
        "scanner missed the detached test: {found:?}"
    );
    assert!(
        !found.iter().any(|f| f.contains("a_fixture")),
        "scanner flagged a parameterised fixture: {found:?}"
    );
    assert!(
        !found.iter().any(|f| f.contains("the_inserted_one")),
        "scanner flagged an attached test: {found:?}"
    );
}

#[test]
fn the_scanner_sees_through_a_multiline_ignore_reason_to_the_test_attribute_above_it() {
    let src = r#"
#[test]
#[ignore = "loads a ~16 GB checkpoint and drives a ladder; \
            set NV_SOME_REAL_TEST=1 to run"]
fn attached_behind_a_multiline_ignore() {
    assert!(true);
}

const AFTER: usize = 1;

fn detached_after_a_terminated_item() {
    assert_eq!(AFTER, 1);
}
"#;
    let found = orphans(src, true);
    assert!(
        !found.iter().any(|f| f.contains("attached_behind_a_multiline_ignore")),
        "an #[ignore] reason string spanning lines is still one attribute; flagging the \
         test under it makes every real-weight suite in this repo a false positive: {found:?}"
    );
    assert!(
        found.iter().any(|f| f.contains("detached_after_a_terminated_item")),
        "a line ending in ';' terminates the item above, so the fn after it has no \
         attribute path upward and must still be flagged: {found:?}"
    );
}

#[test]
fn tracked_rust_files_stay_comment_stripped() {
    let root = repo_rust_root();
    let ls = std::process::Command::new("git")
        .args(["-C"])
        .arg(&root)
        .args(["ls-files", "-z", "--", "*.rs"])
        .output()
        .expect("git must be runnable: the strip gate cannot enumerate tracked files without it");
    assert!(
        ls.status.success(),
        "git ls-files failed: {}",
        String::from_utf8_lossy(&ls.stderr)
    );
    let files: Vec<&str> = std::str::from_utf8(&ls.stdout)
        .expect("git ls-files -z output is utf-8 paths")
        .split('\0')
        .filter(|p| !p.is_empty())
        .collect();
    assert!(
        files.len() > 200,
        "only {} tracked rust files under {}; the enumeration is broken and this gate is \
         proving nothing",
        files.len(),
        root.display()
    );
    let stripper = root.join("..").join("scripts").join("strip-comments.py");
    assert!(
        stripper.exists(),
        "scripts/strip-comments.py is the canonical formatter this gate runs; it is gone from {}",
        stripper.display()
    );
    let out = std::process::Command::new("python3")
        .arg(&stripper)
        .arg("--check")
        .args(&files)
        .current_dir(&root)
        .output()
        .expect(
            "python3 must be on PATH (the nvk.sh devshell provides it); a gate that skips when \
             its interpreter is missing reports green for a check that never ran",
        );
    assert!(
        out.status.success(),
        "comments have re-accumulated in tracked rust files; run \
         `python3 scripts/strip-comments.py rust/` from the repo root, moving any load-bearing \
         rationale into a constant NAME or an assert message first:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
