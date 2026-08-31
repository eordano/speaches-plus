use std::collections::BTreeSet;
use std::path::PathBuf;

const SHARED_HELPER_MODULES_PATH_INCLUDED_BY_SUITES_AND_DECLARING_NO_TESTS_THEMSELVES: &[&str] = &[];

const CENSUS_CUDA_ONLY: &[&str] = &[
    "chain_graph.rs",
    "dflash_gemma4_chain_geometry.rs",
    "lora_laguna_install.rs",
    "qwen38_chunked_prefill_real_ab.rs",
    "qwen38_lora_parity.rs",
    "qwen38_mtp_anchor_mismatch.rs",
    "qwen38_mtp_draft_profile.rs",
    "qwen38_mtp_k8_eager_probe.rs",
    "qwen38_mtp_k_sweep.rs",
    "qwen38_mtp_lmfuse_gate.rs",
    "qwen38_mtp_real_ab.rs",
    "qwen38_mtp_synthetic_forward.rs",
    "qwen38_pctree_headroom.rs",
    "qwen38_prefill_session_phases.rs",
    "qwen38_rowshared_gate.rs",
    "qwen38_sufspec_gate.rs",
    "qwen38_sufspec_headroom.rs",
    "qwen38_tctile_cert.rs",
    "qwen38_tctile_gate.rs",
    "qwen38_tree_gate.rs",
    "qwen38_verify_m67_layer_census.rs",
    "qwen38_verify_m_invariance.rs",
];

const CENSUS_CUDA_AND_WGPU: &[&str] = &[];

const CENSUS_WGPU_ONLY: &[&str] = &[
    "lora_spec_wgpu.rs",
    "qwen38_mtp_wgpu_inc2.rs",
    "qwen38_mtp_wgpu_inc2_real.rs",
    "wgpu_spec.rs",
    "wgpu_spec_host_ref.rs",
    "wgpu_spec_real_gemma4.rs",
    "wgpu_spec_real_lockstep.rs",
    "wgpu_spec_smallm.rs",
];

const CENSUS_UNCONDITIONAL_THESE_SUITES_ASSERT_ON_HOST_MATH_OR_SOURCE_TEXT_NOT_ON_A_DEVICE:
    &[&str] = &[
    "feature_census.rs",
    "lora_spec_backend.rs",
    "lora_token_map.rs",
    "prompt_fidelity.rs",
    "qwen38_mtp_ctx_depth.rs",
    "qwen38_mtp_geometry.rs",
    "qwen38_mtp_inc2_round.rs",
];

const CULLED_SUITES_NO_DELETION_FROM_THIS_DIRECTORY_HAS_EVER_BEEN_RECORDED: &[&str] = &[];

const SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED: &[(&str, &str)] = &[
    ("qwen38_chunked_prefill_real_ab.rs", "NV_Q38_CHUNK_AB"),
    (
        "qwen38_lora_parity.rs",
        "NV_Q38_LORA NV_Q38_LORA_MERGED_DIR",
    ),
    ("qwen38_mtp_anchor_mismatch.rs", "NV_Q38_MTP"),
    (
        "qwen38_mtp_ctx_depth.rs",
        "NV_QWEN38_SERVING_TEST NV_QWEN38_WGPU_TEST",
    ),
    ("qwen38_mtp_draft_profile.rs", "NV_Q38_MTP"),
    ("qwen38_mtp_k8_eager_probe.rs", "NV_Q38_MTP"),
    ("qwen38_mtp_k_sweep.rs", "NV_Q38_MTP"),
    ("qwen38_mtp_wgpu_inc2_real.rs", "NV_Q38_MTP_INC2_TEST"),
    ("qwen38_pctree_headroom.rs", "NV_QWEN38_SERVING_TEST"),
    ("qwen38_prefill_session_phases.rs", "NV_QWEN38_SERVING_TEST"),
    ("qwen38_rowshared_gate.rs", "NV_QWEN38_SERVING_TEST"),
    ("qwen38_sufspec_gate.rs", "NV_QWEN38_SERVING_TEST"),
    ("qwen38_sufspec_headroom.rs", "NV_QWEN38_SERVING_TEST"),
    ("qwen38_tctile_cert.rs", "NV_QWEN38_SERVING_TEST"),
    ("qwen38_tctile_gate.rs", "NV_QWEN38_SERVING_TEST"),
    ("qwen38_tree_gate.rs", "NV_QWEN38_SERVING_TEST"),
    ("qwen38_verify_m67_layer_census.rs", "NV_QWEN38_SERVING_TEST"),
    ("qwen38_verify_m_invariance.rs", "NV_QWEN38_SERVING_TEST"),
];

const UNCONDITIONAL_FILES_WHOSE_TESTS_HIDE_BEHIND_THEIR_OWN_CFG_SO_THE_HEADER_LIES: &[(
    &str,
    usize,
)] = &[("lora_spec_backend.rs", 4), ("qwen38_mtp_ctx_depth.rs", 6)];

const HEADER_CUDA_AND_WGPU: &str = "#![cfg(all(feature = \"cuda\", feature = \"wgpu\"))]";
const HEADER_CUDA_ONLY: &str = "#![cfg(feature = \"cuda\")]";
const HEADER_WGPU_ONLY: &str = "#![cfg(feature = \"wgpu\")]";

const CUDA_ONLY_CLASS_COLLAPSES_BELOW: usize = 8;
const WGPU_ONLY_CLASS_COLLAPSES_BELOW: usize = 6;

fn suite_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn crate_level_cfg_lines(src: &str) -> Vec<String> {
    src.lines()
        .filter(|l| l.starts_with("#![cfg("))
        .map(|l| l.trim_end().to_string())
        .collect()
}

fn unindented_positive_cfg_attributes(src: &str) -> usize {
    src.lines()
        .filter(|l| {
            l.starts_with("#[cfg(feature")
                || l.starts_with("#[cfg(all(feature")
                || l.starts_with("#[cfg(any(feature")
        })
        .count()
}

const TEST_ATTRIBUTE_SPELLINGS_CARGO_RUNS_AS_A_TEST_NOT_ONLY_PLAIN_TEST: &[&str] = &[
    "#[test]",
    "#[test(",
    "#[tokio::test",
    "#[rstest",
    "#[test_case",
    "#[proptest",
    "#[bench",
];

fn is_a_test_attribute(t: &str) -> bool {
    TEST_ATTRIBUTE_SPELLINGS_CARGO_RUNS_AS_A_TEST_NOT_ONLY_PLAIN_TEST
        .iter()
        .any(|a| t.starts_with(a))
}

struct SourceFn {
    name: String,
    is_test: bool,
    is_ignored: bool,
    is_zero_arg_bool_fn: bool,
    body: Vec<String>,
    body_never_closed: bool,
}

fn braces_outside_char_literals(l: &str) -> (usize, usize) {
    let stripped = l.replace("'{'", "").replace("'}'", "");
    (
        stripped.matches('{').count(),
        stripped.matches('}').count(),
    )
}

fn source_fns(src: &str) -> Vec<SourceFn> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out: Vec<SourceFn> = Vec::new();
    let mut attr_depth = 0i32;
    let mut in_attr = false;
    let mut saw_test = false;
    let mut saw_ignore = false;
    let mut i = 0usize;
    while i < lines.len() {
        let t = lines[i].trim();
        if in_attr {
            attr_depth += t.matches('[').count() as i32 - t.matches(']').count() as i32;
            in_attr = attr_depth > 0;
            i += 1;
            continue;
        }
        if t.starts_with("#[") {
            if is_a_test_attribute(t) {
                saw_test = true;
            }
            if t.starts_with("#[ignore") {
                saw_ignore = true;
            }
            attr_depth = t.matches('[').count() as i32 - t.matches(']').count() as i32;
            in_attr = attr_depth > 0;
            i += 1;
            continue;
        }
        if t.is_empty() {
            i += 1;
            continue;
        }
        let starts_a_fn = t.starts_with("fn ")
            || t.starts_with("pub fn ")
            || t.starts_with("async fn ")
            || t.starts_with("pub async fn ");
        if !starts_a_fn {
            saw_test = false;
            saw_ignore = false;
            i += 1;
            continue;
        }
        let name = t
            .split("fn ")
            .nth(1)
            .unwrap_or("")
            .split('(')
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        let mut body: Vec<String> = Vec::new();
        let mut signature = String::new();
        let mut d = 0i32;
        let mut started = false;
        let mut j = i;
        while j < lines.len() {
            let l = lines[j];
            if !started {
                signature.push_str(l);
            }
            body.push(l.to_string());
            let (opens, closes) = braces_outside_char_literals(l);
            d += opens as i32 - closes as i32;
            if opens > 0 {
                started = true;
            }
            if started && d <= 0 {
                break;
            }
            j += 1;
        }
        let closed = started && d <= 0;
        out.push(SourceFn {
            name,
            is_test: saw_test,
            is_ignored: saw_test && saw_ignore,
            is_zero_arg_bool_fn: !saw_test
                && signature.contains("()")
                && signature.contains("-> bool"),
            body,
            body_never_closed: !closed,
        });
        saw_test = false;
        saw_ignore = false;
        i = if closed { j + 1 } else { i + 1 };
    }
    out
}

fn declared_and_ignored_test_fns(src: &str) -> (usize, usize) {
    let fns = source_fns(src);
    (
        fns.iter().filter(|f| f.is_test).count(),
        fns.iter().filter(|f| f.is_ignored).count(),
    )
}

fn env_flag_helper_names(fns: &[SourceFn]) -> BTreeSet<String> {
    fns.iter()
        .filter(|f| f.is_zero_arg_bool_fn && f.body.iter().any(|l| l.contains("env::var")))
        .map(|f| f.name.clone())
        .collect()
}

const A_BLOCK_OPENS_WITH: char = '{';
const A_BLOCK_CLOSES_WITH: char = '}';

fn enclosing_guard(body: &[String], k: usize) -> String {
    let mut d = 0i32;
    let mut open = 0usize;
    for j in (0..=k).rev() {
        d += body[j].matches('}').count() as i32 - body[j].matches('{').count() as i32;
        if d < 0 {
            open = j;
            break;
        }
    }
    let mut first = open;
    while first > 0 && open - first < 8 {
        let prev = body[first - 1].trim_end();
        let ends_a_finished_statement = prev.is_empty()
            || prev.ends_with(';')
            || prev.ends_with(A_BLOCK_OPENS_WITH)
            || prev.ends_with(A_BLOCK_CLOSES_WITH);
        if ends_a_finished_statement {
            break;
        }
        first -= 1;
    }
    body[first..=open]
        .iter()
        .map(|l| l.trim())
        .collect::<Vec<_>>()
        .join(" ")
}

fn tests_that_return_green_when_an_opt_in_env_var_is_absent(src: &str) -> Vec<String> {
    let fns = source_fns(src);
    let flags = env_flag_helper_names(&fns);
    let mut out: Vec<String> = Vec::new();
    for f in fns.iter().filter(|f| f.is_test) {
        if f.body_never_closed {
            out.push(format!(
                "{}: this census could not find the end of the function body, so its returns were \
                 never examined; an unbalanced brace inside a string literal defeats the walk and \
                 an unexaminable suite must be an error, not a silent pass",
                f.name
            ));
            continue;
        }
        for (k, l) in f.body.iter().enumerate() {
            let t = l.trim();
            if t != "return;" && t != "return Ok(());" {
                continue;
            }
            let guard = enclosing_guard(&f.body, k);
            let gates_on_env = guard.contains("env::var")
                || flags.iter().any(|n| guard.contains(&format!("{n}()")));
            if !gates_on_env {
                continue;
            }
            let fires_only_when_the_variable_is_deliberately_set = (guard.contains("==")
                || guard.contains("is_ok()"))
                && !guard.contains("!=")
                && !guard.contains("is_err()")
                && !guard.contains("is_none()")
                && !flags.iter().any(|n| guard.contains(&format!("!{n}()")));
            if fires_only_when_the_variable_is_deliberately_set {
                continue;
            }
            out.push(format!("{} returns under the guard `{}`", f.name, guard.trim()));
        }
    }
    out
}

const A_WALK_THAT_READS_FEWER_FILES_THAN_THIS_IS_BROKEN_NOT_CLEAN: usize = 22;

fn suite_sources() -> Vec<(String, String)> {
    let dir = suite_dir();
    let mut out: Vec<(String, String)> = Vec::new();
    let mut push = |p: &PathBuf| {
        if p.extension().and_then(|s| s.to_str()) != Some("rs") {
            return;
        }
        let name = p
            .strip_prefix(&dir)
            .unwrap_or(p)
            .to_string_lossy()
            .to_string();
        let src = std::fs::read_to_string(p)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
        out.push((name, src));
    };
    for e in std::fs::read_dir(&dir).expect("suite directory").flatten() {
        let p = e.path();
        if p.is_dir() {
            for e2 in std::fs::read_dir(&p).expect("suite subdirectory").flatten() {
                push(&e2.path());
            }
        } else {
            push(&p);
        }
    }
    out.sort();
    out
}

#[derive(Default)]
struct Corpus {
    cuda_only: Vec<String>,
    cuda_and_wgpu: Vec<String>,
    wgpu_only: Vec<String>,
    unconditional: Vec<String>,
    helpers: Vec<String>,
    cfg_headers_this_census_cannot_classify: Vec<String>,
    files_declaring_no_tests_and_not_declared_helpers: Vec<String>,
    every_test_ignored: Vec<String>,
    unconditional_with_per_item_cfg: Vec<(String, usize)>,
    subdirectory_targets_a_file_walk_cannot_see: Vec<String>,
    subdirectory_helper_modules: Vec<String>,
}

fn corpus() -> Corpus {
    let dir = suite_dir();
    let rd = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read the suite directory {}: {e}", dir.display()));
    let mut c = Corpus::default();
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            let dname = p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();
            if p.join("main.rs").exists() {
                c.subdirectory_targets_a_file_walk_cannot_see.push(dname);
            } else if p.join("mod.rs").exists() {
                c.subdirectory_helper_modules.push(dname);
            }
            continue;
        }
        if p.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let name = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let src = std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
        let is_helper = SHARED_HELPER_MODULES_PATH_INCLUDED_BY_SUITES_AND_DECLARING_NO_TESTS_THEMSELVES
            .contains(&name.as_str());
        let (declared, ignored) = declared_and_ignored_test_fns(&src);
        if is_helper {
            c.helpers.push(name.clone());
        } else if declared == 0 {
            c.files_declaring_no_tests_and_not_declared_helpers
                .push(name.clone());
        } else if declared == ignored {
            c.every_test_ignored.push(name.clone());
        }
        let cfgs = crate_level_cfg_lines(&src);
        if cfgs.len() > 1 {
            c.cfg_headers_this_census_cannot_classify.push(format!(
                "{name}: {} crate-level #![cfg] lines, which this census cannot reduce to one \
                 class: {cfgs:?}",
                cfgs.len()
            ));
            continue;
        }
        let class = match cfgs.first().map(String::as_str) {
            None => 0u8,
            Some(HEADER_CUDA_AND_WGPU) => 1,
            Some(HEADER_CUDA_ONLY) => 2,
            Some(HEADER_WGPU_ONLY) => 3,
            Some(other) => {
                c.cfg_headers_this_census_cannot_classify
                    .push(format!("{name}: unknown spelling {other}"));
                continue;
            }
        };
        if is_helper {
            continue;
        }
        match class {
            1 => c.cuda_and_wgpu.push(name),
            2 => c.cuda_only.push(name),
            3 => c.wgpu_only.push(name),
            _ => {
                let n = unindented_positive_cfg_attributes(&src);
                if n > 0 {
                    c.unconditional_with_per_item_cfg.push((name.clone(), n));
                }
                c.unconditional.push(name);
            }
        }
    }
    for v in [
        &mut c.cuda_only,
        &mut c.cuda_and_wgpu,
        &mut c.wgpu_only,
        &mut c.unconditional,
        &mut c.helpers,
        &mut c.cfg_headers_this_census_cannot_classify,
        &mut c.files_declaring_no_tests_and_not_declared_helpers,
        &mut c.every_test_ignored,
        &mut c.subdirectory_targets_a_file_walk_cannot_see,
        &mut c.subdirectory_helper_modules,
    ] {
        v.sort();
    }
    c.unconditional_with_per_item_cfg.sort();
    c
}

fn census_delta(
    class: &str,
    recorded_in: &str,
    on_disk: &[String],
    recorded: &[&str],
) -> Option<String> {
    let disk: BTreeSet<&str> = on_disk.iter().map(String::as_str).collect();
    let rec: BTreeSet<&str> = recorded.iter().copied().collect();
    let added: Vec<&str> = disk.difference(&rec).copied().collect();
    let dropped: Vec<&str> = rec.difference(&disk).copied().collect();
    if added.is_empty() && dropped.is_empty() {
        return None;
    }
    let mut m = format!(
        "{class}: {} suites on disk, {} recorded in {recorded_in}",
        disk.len(),
        rec.len()
    );
    for n in added {
        m.push_str(&format!(
            "\n      + {n} is on disk and not in {recorded_in}. The whole fix is one line: add \
             \"{n}\", to {recorded_in}. Name the commit that added it with\n          git log \
             --oneline --diff-filter=A -- rust/crates/nv-specdecode/tests/{n}"
        ));
    }
    for n in dropped {
        m.push_str(&format!(
            "\n      - {n} is in {recorded_in} and not on disk. The whole fix is one line: delete \
             \"{n}\", from {recorded_in}, and record the lost coverage in \
             CULLED_SUITES_NO_DELETION_FROM_THIS_DIRECTORY_HAS_EVER_BEEN_RECORDED. Name the commit \
             that removed it with\n          git log --oneline --diff-filter=D -- \
             rust/crates/nv-specdecode/tests/{n}"
        ));
    }
    Some(m)
}

fn out_of_order(recorded_in: &str, recorded: &[&str]) -> Option<String> {
    for w in recorded.windows(2) {
        if w[0] >= w[1] {
            return Some(format!(
                "{recorded_in} is not in strictly ascending order: {:?} precedes {:?}. Keep it \
                 sorted, or two lanes that each add a suite will both append and the duplicate \
                 will read as a correct count.",
                w[0], w[1]
            ));
        }
    }
    None
}

#[test]
fn features_compiled_into_this_test_run() {
    let cuda = cfg!(feature = "cuda");
    let wgpu = cfg!(feature = "wgpu");
    let c = corpus();
    eprintln!("nv-specdecode tests compiled with cuda={cuda} wgpu={wgpu}");
    eprintln!(
        "nvk.sh defaults every NVK_PKG other than nv-kernels to `cuda` ALONE, so the natural \
         `NVK_PKG=nv-specdecode nvk.sh test` silently compiles every wgpu suite to nothing and \
         prints `0 passed` in 0.00s. Pass NVK_FEATURES=cuda,wgpu for the full surface."
    );
    assert!(
        cuda || wgpu,
        "neither `cuda` nor `wgpu` is enabled, so {} of this crate's {} suites compiled to empty \
         binaries. A run in this configuration is not evidence of anything.",
        c.cuda_only.len() + c.cuda_and_wgpu.len() + c.wgpu_only.len(),
        c.cuda_only.len() + c.cuda_and_wgpu.len() + c.wgpu_only.len() + c.unconditional.len(),
    );
}

#[test]
fn the_suite_corpus_is_the_one_this_census_names_file_by_file() {
    let c = corpus();
    let classes: [(&str, &str, &[String], &[&str]); 4] = [
        (
            "cuda-only",
            "CENSUS_CUDA_ONLY",
            c.cuda_only.as_slice(),
            CENSUS_CUDA_ONLY,
        ),
        (
            "cuda+wgpu",
            "CENSUS_CUDA_AND_WGPU",
            c.cuda_and_wgpu.as_slice(),
            CENSUS_CUDA_AND_WGPU,
        ),
        (
            "wgpu-only",
            "CENSUS_WGPU_ONLY",
            c.wgpu_only.as_slice(),
            CENSUS_WGPU_ONLY,
        ),
        (
            "unconditional",
            "CENSUS_UNCONDITIONAL_THESE_SUITES_ASSERT_ON_HOST_MATH_OR_SOURCE_TEXT_NOT_ON_A_DEVICE",
            c.unconditional.as_slice(),
            CENSUS_UNCONDITIONAL_THESE_SUITES_ASSERT_ON_HOST_MATH_OR_SOURCE_TEXT_NOT_ON_A_DEVICE,
        ),
    ];
    let total =
        c.cuda_only.len() + c.cuda_and_wgpu.len() + c.wgpu_only.len() + c.unconditional.len();
    eprintln!(
        "nv-specdecode suite corpus: {total} suites on disk = {} cuda-only + {} cuda+wgpu + {} \
         wgpu-only + {} unconditional, plus {} shared helper module(s) {:?} that declare no tests \
         of their own",
        c.cuda_only.len(),
        c.cuda_and_wgpu.len(),
        c.wgpu_only.len(),
        c.unconditional.len(),
        c.helpers.len(),
        c.helpers,
    );

    let mut wrong: Vec<String> = Vec::new();
    for (class, recorded_in, on_disk, recorded) in classes {
        wrong.extend(out_of_order(recorded_in, recorded));
        wrong.extend(census_delta(class, recorded_in, on_disk, recorded));
    }
    assert!(
        wrong.is_empty(),
        "the suite corpus moved without this census moving with it.\n  {}\n\nEvery line above is \
         its own fix and its own investigation: the name to add or delete, the constant to edit, \
         and the git command that names the commit responsible. Make the edit in the SAME commit \
         as the suite and put those names in the message. This census WALKS the directory rather \
         than trusting a written list precisely so that a suite cannot be absent from it: the \
         hand-maintained version of this file named 9 files while 26 existed, and the seventeen it \
         never named -- every qwen38_mtp_* suite among them -- could be deleted, broken or left \
         entirely #[ignore]d without a single artifact in this crate noticing.",
        wrong.join("\n  ")
    );
}

#[test]
fn no_file_in_this_directory_escapes_classification_by_carrying_a_header_spelling_this_census_cannot_read(
) {
    let c = corpus();
    assert!(
        c.cfg_headers_this_census_cannot_classify.is_empty(),
        "these files carry a crate-level #![cfg] this census cannot reduce to one of its four \
         classes: {:?}.\nAn unreadable header is how a suite silently changes class, and a class \
         that silently loses a member is exactly the hole this file exists to close. Teach \
         crate_level_cfg_lines and corpus() the new spelling, or restate the header as one of \
         HEADER_CUDA_ONLY / HEADER_WGPU_ONLY / HEADER_CUDA_AND_WGPU.",
        c.cfg_headers_this_census_cannot_classify,
    );
    eprintln!(
        "subdirectories of tests/ holding a shared mod.rs, which cargo builds into each dependent \
         suite rather than as a target of its own: {:?}",
        c.subdirectory_helper_modules,
    );
    assert!(
        c.subdirectory_targets_a_file_walk_cannot_see.is_empty(),
        "these subdirectories of tests/ hold a main.rs: {:?}.\ncargo builds tests/<dir>/main.rs as \
         its own test target, and this census walks FILES in tests/ -- so such a target is exactly \
         the suite-that-never-runs this file exists to make impossible, hiding one directory level \
         below where anyone would look. Either flatten it to tests/<dir>.rs, or teach corpus() to \
         descend and classify it.",
        c.subdirectory_targets_a_file_walk_cannot_see,
    );
    assert!(
        c.cuda_only.len() >= CUDA_ONLY_CLASS_COLLAPSES_BELOW
            && c.wgpu_only.len() >= WGPU_ONLY_CLASS_COLLAPSES_BELOW,
        "the two feature-gated classes collapsed ({} cuda-only, {} wgpu-only), which is what a \
         changed header spelling looks like from here even when every per-file delta is clean",
        c.cuda_only.len(),
        c.wgpu_only.len(),
    );
}

#[test]
fn every_file_excluded_as_a_shared_helper_declares_no_tests_and_is_path_included_by_a_real_suite() {
    let dir = suite_dir();
    let mut wrong: Vec<String> = Vec::new();
    for helper in SHARED_HELPER_MODULES_PATH_INCLUDED_BY_SUITES_AND_DECLARING_NO_TESTS_THEMSELVES {
        let p = dir.join(helper);
        let Ok(src) = std::fs::read_to_string(&p) else {
            wrong.push(format!(
                "{helper} is excluded from the census as a shared helper but does not exist"
            ));
            continue;
        };
        let (declared, _) = declared_and_ignored_test_fns(&src);
        if declared != 0 {
            wrong.push(format!(
                "{helper} is excluded from the census as a shared helper but declares {declared} \
                 #[test] fn(s); a file that asserts is a suite and belongs in a census class"
            ));
        }
        let include = format!("#[path = \"{helper}\"]");
        let included_by: Vec<String> = std::fs::read_dir(&dir)
            .expect("suite directory")
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("rs"))
            .filter(|p| p.file_name().and_then(|s| s.to_str()) != Some(helper))
            .filter(|p| {
                std::fs::read_to_string(p)
                    .map(|s| s.contains(&include))
                    .unwrap_or(false)
            })
            .filter_map(|p| p.file_name().and_then(|s| s.to_str()).map(String::from))
            .collect();
        if included_by.is_empty() {
            wrong.push(format!(
                "{helper} is excluded from the census as a shared helper but no suite carries \
                 {include}; an unreferenced helper is dead weight cargo still builds as its own \
                 test target, and excluding it hides that"
            ));
        } else {
            eprintln!("{helper} is path-included by {included_by:?}");
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n  "));

    let c = corpus();
    assert!(
        c.files_declaring_no_tests_and_not_declared_helpers.is_empty(),
        "these files in tests/ declare no #[test] fn this census can see, yet are not named in \
         SHARED_HELPER_MODULES_PATH_INCLUDED_BY_SUITES_AND_DECLARING_NO_TESTS_THEMSELVES: \
         {:?}.\ncargo builds every tests/*.rs as its own binary, so such a file prints `ok. 0 \
         passed` and reads as a passing suite forever. Either it is a helper -- name it, and this \
         census will prove the claim -- or it declares its tests through an attribute spelling \
         outside TEST_ATTRIBUTE_SPELLINGS_CARGO_RUNS_AS_A_TEST_NOT_ONLY_PLAIN_TEST, which \
         already carries #[test], #[tokio::test], #[rstest], #[test_case], #[proptest] and \
         #[bench] -- add the new spelling there rather than excluding the file.",
        c.files_declaring_no_tests_and_not_declared_helpers,
    );
}

#[test]
fn every_suite_whose_tests_are_all_ignored_is_named_here_with_the_env_var_that_opts_it_in() {
    let c = corpus();
    let dir = suite_dir();
    let recorded: BTreeSet<&str> = SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED
        .iter()
        .map(|(n, _)| *n)
        .collect();
    let found: BTreeSet<&str> = c.every_test_ignored.iter().map(String::as_str).collect();
    let mut wrong: Vec<String> = Vec::new();
    for n in found.difference(&recorded) {
        wrong.push(format!(
            "+ {n}: every #[test] in it is #[ignore]d, so `nvk.sh test` prints `ok. 0 passed` for \
             it and a green crate says nothing about the property it gates. Add it to \
             SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED with the env \
             var that runs it."
        ));
    }
    for n in recorded.difference(&found) {
        wrong.push(format!(
            "- {n}: recorded as entirely-#[ignore]d, but it now has at least one test that runs \
             by default (or the file is gone). Drop the entry."
        ));
    }
    for (n, vars) in SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED {
        let Ok(src) = std::fs::read_to_string(dir.join(n)) else {
            continue;
        };
        for var in vars.split_whitespace() {
            if !src.contains(var) {
                wrong.push(format!(
                    "! {n} is recorded as opting in through {var}, which does not appear in its \
                     source. A documented opt-in nobody can perform is worse than none."
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "the entirely-ignored suites moved:\n  {}\n\nThis class is the second blind spot behind \
         the unregistered one: such a suite is present, registered, compiled, and still proves \
         nothing until someone sets its variable. Every entry here loads a multi-gigabyte \
         Qwen3.8-27B snapshot, which is why they are #[ignore]d by design -- but by design or not, \
         a plain run of this crate reports `ok. 0 passed` for {} of its {} suites.",
        wrong.join("\n  "),
        SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED.len(),
        c.cuda_only.len() + c.cuda_and_wgpu.len() + c.wgpu_only.len() + c.unconditional.len(),
    );
    eprintln!(
        "PRESENT BUT SILENT UNLESS OPTED IN: {:?}",
        SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED
    );
}

#[test]
fn every_unconditional_file_whose_tests_carry_their_own_cfg_is_named_here_with_that_count() {
    let c = corpus();
    let found: Vec<(String, usize)> = c.unconditional_with_per_item_cfg.clone();
    let recorded: Vec<(String, usize)> =
        UNCONDITIONAL_FILES_WHOSE_TESTS_HIDE_BEHIND_THEIR_OWN_CFG_SO_THE_HEADER_LIES
            .iter()
            .map(|(n, k)| ((*n).to_string(), *k))
            .collect();
    assert_eq!(
        found,
        recorded,
        "a file with no crate-level #![cfg] reads as `runs everywhere` from a header walk, and \
         these files defeat that reading by gating individual tests with their own #[cfg]. \
         lora_spec_backend.rs is the case that matters: with one feature it still reports a \
         non-zero pass count from its CPU test while cuda_and_wgpu_lora_deltas_are_bit_identical \
         -- the whole point of the file -- is erased. qwen38_mtp_ctx_depth.rs is worse: all three \
         of its tests are BOTH per-test-#[cfg]'d and #[ignore]d, so the file is invisible twice \
         over. Update the count here in the same commit that changes the gating."
    );
    eprintln!("unconditional header, per-test #[cfg] inside: {found:?}");
}

#[test]
fn no_suite_reports_a_pass_by_returning_when_an_opt_in_env_var_it_gates_itself_on_is_absent() {
    let sources = suite_sources();
    assert!(
        sources.len() >= A_WALK_THAT_READS_FEWER_FILES_THAN_THIS_IS_BROKEN_NOT_CLEAN,
        "this gate read only {} files under {}; a walk that finds nothing proves nothing",
        sources.len(),
        suite_dir().display(),
    );
    let mut wrong: Vec<String> = Vec::new();
    for (name, src) in &sources {
        for m in tests_that_return_green_when_an_opt_in_env_var_is_absent(src) {
            wrong.push(format!("{name}: {m}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "these tests return instead of failing when the environment variable they gate \
         themselves on is absent, so the runner counts them as PASSED having executed \
         nothing:\n  {}\n\nA suite whose precondition is absent must fail or be ignored, never \
         return green. An `#[ignore]`d suite at least prints `0 passed`, which this census can \
         assert on; a skip-and-return prints `1 passed` and reads as coverage. The fix is one of \
         two shapes: panic! with a message naming the variable (correct when the suite is \
         already #[ignore]d, because --ignored IS the opt-in and a second gate behind it can \
         only make --ignored lie), or #[ignore] plus that same panic (correct when the suite \
         still runs by default). A guard that fires only when a variable is deliberately SET -- \
         an explicit NV_*_ALLOW_SKIP opt-out -- is not this defect and is not reported here.",
        wrong.join("\n  ")
    );
}

#[test]
fn the_skip_return_scanner_reads_the_absent_direction_and_leaves_a_deliberate_opt_out_alone() {
    let fixture = "fn device_or_none() -> Option<usize> {\n\
                   \x20   None\n\
                   }\n\
                   \n\
                   fn bench_gated() -> bool {\n\
                   \x20   std::env::var(\"NV_X_BENCH\").as_deref() == Ok(\"1\")\n\
                   }\n\
                   \n\
                   #[test]\n\
                   fn skips_on_a_direct_env_absence() {\n\
                   \x20   if std::env::var(\"NV_X_TEST\").as_deref() != Ok(\"1\") {\n\
                   \x20       eprintln!(\"SKIP: set NV_X_TEST=1\");\n\
                   \x20       return;\n\
                   \x20   }\n\
                   \x20   assert!(true);\n\
                   }\n\
                   \n\
                   #[test]\n\
                   fn skips_through_a_bool_returning_env_flag_helper() {\n\
                   \x20   if !bench_gated() {\n\
                   \x20       return;\n\
                   \x20   }\n\
                   \x20   assert!(true);\n\
                   }\n\
                   \n\
                   #[test]\n\
                   fn skips_only_when_a_variable_is_deliberately_set() {\n\
                   \x20   if std::env::var(\"NV_X_ALLOW_SKIP\").as_deref() == Ok(\"1\") {\n\
                   \x20       return;\n\
                   \x20   }\n\
                   \x20   assert!(true);\n\
                   }\n\
                   \n\
                   #[test]\n\
                   fn returns_early_on_a_fact_about_the_machine_not_a_variable() {\n\
                   \x20   let n = 3usize;\n\
                   \x20   if n == 0 {\n\
                   \x20       return;\n\
                   \x20   }\n\
                   \x20   assert!(true);\n\
                   }\n\
                   \n\
                   #[test]\n\
                   fn asserts_on_the_variable_then_returns_on_an_absent_device() {\n\
                   \x20   assert_eq!(\n\
                   \x20       std::env::var(\"NV_X_BENCH\").ok().as_deref(),\n\
                   \x20       Some(\"1\"),\n\
                   \x20       \"set NV_X_BENCH=1 -- this loads the GPU\"\n\
                   \x20   );\n\
                   \x20   let Some(_d) = device_or_none() else {\n\
                   \x20       return;\n\
                   \x20   };\n\
                   \x20   assert!(true);\n\
                   }\n";
    let found = tests_that_return_green_when_an_opt_in_env_var_is_absent(fixture);
    assert!(
        found.iter().any(|f| f.contains("skips_on_a_direct_env_absence")),
        "the scanner must see a direct env::var guard whose absent branch returns: {found:?}"
    );
    assert!(
        found
            .iter()
            .any(|f| f.contains("skips_through_a_bool_returning_env_flag_helper")),
        "a zero-argument helper returning bool from an env::var read is the same gate one call \
         away; a scanner that only reads env::var textually misses every `if !bench_gated()`: \
         {found:?}"
    );
    assert!(
        !found
            .iter()
            .any(|f| f.contains("skips_only_when_a_variable_is_deliberately_set")),
        "a guard that fires only when a variable IS set is a deliberate, named opt-out, not a \
         suite going green because a precondition is missing: {found:?}"
    );
    assert!(
        !found
            .iter()
            .any(|f| f.contains("returns_early_on_a_fact_about_the_machine_not_a_variable")),
        "an early return that names no environment variable is out of this gate's scope: \
         {found:?}"
    );
    assert!(
        !found
            .iter()
            .any(|f| f.contains("asserts_on_the_variable_then_returns_on_an_absent_device")),
        "the guard this scanner reads is the block the return sits in, not the six lines above \
         it. A suite that asserts on its variable and then returns because the DEVICE is absent \
         is already the correct shape; reading the assert_eq! above the `let ... else` as the \
         guard makes every such suite a false positive, which is how a gate stops being \
         believed: {found:?}"
    );
    assert_eq!(found.len(), 2, "exactly two of the five fixtures qualify: {found:?}");
}

#[test]
fn the_attribute_scanner_reads_multi_line_ignore_attributes_and_ignores_them_inside_strings() {
    let fixture = "#![cfg(feature = \"cuda\")]\n\
                   #[test]\n\
                   fn plain() {}\n\
                   \n\
                   #[test]\n\
                   #[ignore]\n\
                   fn bare_ignore() {}\n\
                   \n\
                   #[cfg(feature = \"wgpu\")]\n\
                   #[test]\n\
                   #[ignore = \"needs a device; set NV_X=1 -- proves \\\n\
                   something across two lines\"]\n\
                   fn multi_line_ignore() {}\n\
                   \n\
                   fn helper_that_only_mentions_attributes() {\n\
                       panic!(\"this is already #[ignore]d, so --ignored IS the opt-in\");\n\
                   }\n";
    assert_eq!(
        declared_and_ignored_test_fns(fixture),
        (3, 2),
        "the scanner must count three #[test] fns of which two are ignored: a bare #[ignore], a \
         multi-line #[ignore = \"...\"] whose closing bracket lands on a later line, and it must \
         not be fooled by a `#[ignore]` mentioned inside a panic message"
    );
    assert_eq!(
        crate_level_cfg_lines(fixture),
        vec![HEADER_CUDA_ONLY.to_string()],
        "an unindented crate-level #![cfg] is what classification reads; a per-item #[cfg] with \
         one bang fewer is not a header and must not be mistaken for one"
    );
    assert_eq!(
        unindented_positive_cfg_attributes(fixture),
        1,
        "the per-item scanner must see the one #[cfg(feature = \"wgpu\")] guarding a test and must \
         not count the crate-level #![cfg] header as one of them"
    );

    let attribute_macro_forms = "#[tokio::test]\n\
                                 async fn plain_tokio() {}\n\
                                 \n\
                                 #[tokio::test(flavor = \"multi_thread\", worker_threads = 2)]\n\
                                 #[ignore = \"boots the real binary\"]\n\
                                 async fn configured_tokio() {}\n\
                                 \n\
                                 #[rstest]\n\
                                 fn a_fixture_driven_case() {}\n\
                                 \n\
                                 #[test_case(1)]\n\
                                 fn a_table_driven_case() {}\n\
                                 \n\
                                 fn a_plain_helper() {}\n";
    let char_literal_braces = "fn depth_of(s: &str) -> usize {\n\
                               \x20   let mut depth = 0usize;\n\
                               \x20   for b in s.bytes() {\n\
                               \x20       match b {\n\
                               \x20           b'{' => depth += 1,\n\
                               \x20           b'}' => depth -= 1,\n\
                               \x20           _ => {}\n\
                               \x20       }\n\
                               \x20   }\n\
                               \x20   depth\n\
                               }\n\
                               \n\
                               #[test]\n\
                               fn a_suite_declared_after_that_helper() {\n\
                               \x20   assert_eq!(depth_of(\"\"), 0);\n\
                               }\n";
    assert_eq!(
        declared_and_ignored_test_fns(char_literal_braces),
        (1, 0),
        "a `b'{{'` or `'}}'` char literal is not a block delimiter. Counting it as one leaves the \
         helper above looking unterminated, the walk swallows every test below it, and the file \
         reads as a test-free helper -- which is how nv-kernels' cuda_ue4m3_subnormal_codec.rs, \
         a suite with eight tests, went invisible the first time this scanner ran."
    );
    assert_eq!(
        declared_and_ignored_test_fns(attribute_macro_forms),
        (4, 1),
        "cargo runs #[tokio::test] fns as tests, so a file whose only tests carry that attribute \
         is a real suite -- not the test-free helper a plain-#[test] scanner reads it as. The \
         rust root crate has 12 such files, 8 of them with no plain #[test] at all, and a census \
         written against the old scanner would have excluded or red-flagged every one. \
         #[rstest], #[test_case], #[proptest] and #[bench] are counted for the same reason: \
         TEST_ATTRIBUTE_SPELLINGS_CARGO_RUNS_AS_A_TEST_NOT_ONLY_PLAIN_TEST is the whole list."
    );
}

#[test]
fn the_culled_suites_are_still_gone_and_their_coverage_is_still_missing() {
    let dir = suite_dir();
    let resurrected: Vec<&str> = CULLED_SUITES_NO_DELETION_FROM_THIS_DIRECTORY_HAS_EVER_BEEN_RECORDED
        .iter()
        .copied()
        .filter(|s| dir.join(format!("{s}.rs")).exists())
        .collect();
    assert!(
        resurrected.is_empty(),
        "CULLED_SUITES_NO_DELETION_FROM_THIS_DIRECTORY_HAS_EVER_BEEN_RECORDED names \
         {resurrected:?}, which exist again. Move them into the matching CENSUS_* list in the same \
         commit that restored them; the directory walk will demand it anyway."
    );
    eprintln!(
        "A directory walk can only report what is present. This list is the only place a deletion \
         from nv-specdecode/tests can be recorded, and it is empty: \
         {CULLED_SUITES_NO_DELETION_FROM_THIS_DIRECTORY_HAS_EVER_BEEN_RECORDED:?}"
    );
}

#[cfg(not(feature = "wgpu"))]
#[test]
#[allow(non_snake_case)]
fn the_wgpu_spec_suites_were_CFGD_OUT_of_this_binary_SKIPPED_no_wgpu_feature() {
    let c = corpus();
    eprintln!(
        "SKIPPED, NOT PASSED: the {} #![cfg(feature = \"wgpu\")] suites compiled to nothing in \
         this run: {:?}. wgpu_spec_host_ref is the ONLY host-f64 oracle on the 13 shipped sp_* \
         kernels -- every other numeric gate in this crate compares one GPU path against another, \
         which cannot see a uniformly wrong kernel (measured: the host-ref gate catches 13/13 \
         planted kernel bugs, everything else in the crate catches 1/13). Losing it to a missing \
         feature flag is the single most expensive skip here. Separately, wgpu_spec_real_lockstep's \
         six tests are pure CPU table decoders and wgpu_spec_real_gemma4's two official_template \
         tests need no GPU at all -- the cfg header alone is what hides them, because the \
         backend-agnostic types they use live in the wgpu-gated nv_specdecode::wgpu_spec module. \
         Re-run with NVK_FEATURES=cuda,wgpu. The list is DERIVED from the directory, not written \
         down.",
        c.wgpu_only.len(),
        c.wgpu_only,
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
#[allow(non_snake_case)]
fn the_cuda_spec_suites_were_CFGD_OUT_of_this_binary_SKIPPED_no_cuda_feature() {
    let c = corpus();
    eprintln!(
        "SKIPPED, NOT PASSED: the {} #![cfg(feature = \"cuda\")] suites compiled to nothing in \
         this run: {:?}. Re-run with NVK_FEATURES=cuda (or cuda,wgpu). The list is DERIVED from \
         the directory, not written down.",
        c.cuda_only.len(),
        c.cuda_only,
    );
}

#[cfg(not(all(feature = "cuda", feature = "wgpu")))]
#[test]
#[allow(non_snake_case)]
fn the_cross_backend_lora_delta_check_was_CFGD_OUT_SKIPPED_needs_cuda_and_wgpu() {
    let c = corpus();
    eprintln!(
        "SKIPPED, NOT PASSED: no file in this directory carries a cfg(all(cuda, wgpu)) HEADER \
         ({:?}), so the cross-backend claim lives inside a file whose header says unconditional: \
         {:?}. lora_spec_backend.rs uses per-test #[cfg] rather than a file header, so with one \
         feature the suite still reports a non-zero pass count from its CPU test while the \
         bit-exactness claim -- cuda_and_wgpu_lora_deltas_are_bit_identical, the whole point of the \
         file -- is erased.",
        c.cuda_and_wgpu, c.unconditional_with_per_item_cfg,
    );
}
