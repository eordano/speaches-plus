use std::path::{Path, PathBuf};

use nv_ocr::{BackendKind, DeepSeekMode, OcrEngine};

const FIXTURES_IN_SORTED_ORDER: [&str; 5] = [
    "071-ocr-layout-invoice",
    "071-ocr-layout-labnotes",
    "071-ocr-layout-letter",
    "071-ocr-layout-newspaper",
    "071-ocr-layout-report",
];

const CLASSICAL_GATES_ARE_CALIBRATED_ON_TESSDATA_BEST: &str = "tessdata_best/eng.traineddata";

const A_MISSING_GATE_MUST_PANIC_NOT_SKIP: &str =
    "a null or absent CER threshold is a dead gate that reads as a pass; the runner names the \
     field instead of scoring the page";

const NEUTRAL_IS_THE_PRIMARY_GATE_BECAUSE_RAW_CHARGES_MARKDOWN_AS_ERRORS: &str =
    "raw CER counts every table pipe, heading hash and line break a correct transcription emits, \
     so a backend that reads the page perfectly and decorates it scores worse than one that \
     misreads it flatly";

pub fn markup_neutral(s: &str) -> String {
    let mut words: Vec<String> = Vec::new();
    for raw in s.lines() {
        let line = strip_html_tags(raw);
        if is_rule_or_table_separator(&line) {
            continue;
        }
        let line = strip_leading_block_markers(&line);
        let line = strip_inline_markup(&line);
        for w in line.split_whitespace() {
            words.push(w.to_string());
        }
    }
    words.join(" ")
}

fn strip_html_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0usize;
    for c in s.chars() {
        match c {
            '<' => depth += 1,
            '>' => {
                if depth > 0 {
                    depth -= 1;
                    out.push(' ');
                } else {
                    out.push(c);
                }
            }
            _ => {
                if depth == 0 {
                    out.push(c);
                }
            }
        }
    }
    out
}

fn is_rule_or_table_separator(s: &str) -> bool {
    let t = s.trim();
    if t.is_empty() {
        return false;
    }
    let mut saw_rule_char = false;
    for c in t.chars() {
        match c {
            '-' | '=' | '_' | '*' => saw_rule_char = true,
            '|' | ':' | '+' | ' ' | '\t' => {}
            _ => return false,
        }
    }
    saw_rule_char
}

fn strip_leading_block_markers(s: &str) -> String {
    let mut t = s.trim_start();
    loop {
        let before = t;
        t = t.trim_start_matches(['#', '>']);
        if let Some(rest) = t.strip_prefix("- ") {
            t = rest;
        } else if let Some(rest) = t.strip_prefix("* ") {
            t = rest;
        } else if let Some(rest) = t.strip_prefix("+ ") {
            t = rest;
        }
        t = t.trim_start();
        if t == before {
            break;
        }
    }
    t.to_string()
}

fn strip_inline_markup(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '*' | '`' | '|' => ' ',
            other => other,
        })
        .collect()
}

pub fn cer(reference: &str, hypothesis: &str) -> f64 {
    let r: Vec<char> = reference.chars().collect();
    let h: Vec<char> = hypothesis.chars().collect();
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for i in 1..=r.len() {
        let mut curr = vec![i; h.len() + 1];
        for j in 1..=h.len() {
            let sub = prev[j - 1] + usize::from(r[i - 1] != h[j - 1]);
            curr[j] = sub.min(prev[j] + 1).min(curr[j - 1] + 1);
        }
        prev = curr;
    }
    prev[h.len()] as f64 / r.len().max(1) as f64
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../conformance/fixtures")
}

struct Fixture {
    name: String,
    expected_text: String,
    png: Vec<u8>,
    cer_max_classical: f64,
    cer_max_classical_neutral: f64,
}

fn gate_field(v: &serde_json::Value, name: &str, fixture: &str) -> f64 {
    v.get("gates")
        .and_then(|g| g.get(name))
        .and_then(|x| x.as_f64())
        .unwrap_or_else(|| {
            panic!("{fixture}: gates.{name} is missing or null. {A_MISSING_GATE_MUST_PANIC_NOT_SKIP}")
        })
}

fn load_fixture(name: &str) -> Fixture {
    let dir = fixtures_dir().join(name);
    let manifest = dir.join("fixture.json");
    let text = std::fs::read_to_string(&manifest)
        .unwrap_or_else(|e| panic!("{}: {e}", manifest.display()));
    let v: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("{}: {e}", manifest.display()));
    assert_eq!(
        v.get("comparison_strategy").and_then(|s| s.as_str()),
        Some("cer_markup_neutral+layout_gates"),
        "{name}: this runner only implements the cer_markup_neutral+layout_gates strategy"
    );
    let input = v
        .get("input")
        .and_then(|s| s.as_str())
        .unwrap_or_else(|| panic!("{name}: fixture.json has no input"));
    let png = std::fs::read(dir.join(input))
        .unwrap_or_else(|e| panic!("{}/{input}: {e}", dir.display()));
    Fixture {
        name: name.to_string(),
        expected_text: v
            .get("expected_text")
            .and_then(|s| s.as_str())
            .unwrap_or_else(|| panic!("{name}: fixture.json has no expected_text"))
            .to_string(),
        png,
        cer_max_classical: gate_field(&v, "cer_max_classical", name),
        cer_max_classical_neutral: gate_field(&v, "cer_max_classical_neutral", name),
    }
}

fn tessdata_root() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NV_OCR_TESSDATA") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let cache = PathBuf::from(home).join(".cache/ocr-testdata");
    cache.exists().then_some(cache)
}

fn classical_traineddata() -> Option<PathBuf> {
    let root = tessdata_root()?;
    if root.is_file() {
        return Some(root);
    }
    for candidate in [
        CLASSICAL_GATES_ARE_CALIBRATED_ON_TESSDATA_BEST,
        "eng.traineddata",
        "tessdata_fast/eng.traineddata",
    ] {
        let p = root.join(candidate);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn classical_engine() -> Option<(OcrEngine, PathBuf)> {
    let path = classical_traineddata()?;
    let engine = OcrEngine::from_traineddata(&path, BackendKind::Classical)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Some((engine, path))
}

struct Scored {
    name: String,
    raw: f64,
    neutral: f64,
    raw_gate: f64,
    neutral_gate: f64,
}

fn score_all_classical(engine: &OcrEngine) -> Vec<Scored> {
    let mut out = Vec::new();
    for name in FIXTURES_IN_SORTED_ORDER {
        let f = load_fixture(name);
        let got = engine
            .recognize_mode(&f.png, DeepSeekMode::FreeOcr)
            .unwrap_or_else(|e| panic!("{name}: classical recognize failed: {e}"));
        out.push(Scored {
            name: f.name,
            raw: cer(&f.expected_text, &got.text),
            neutral: cer(
                &markup_neutral(&f.expected_text),
                &markup_neutral(&got.text),
            ),
            raw_gate: f.cer_max_classical,
            neutral_gate: f.cer_max_classical_neutral,
        });
    }
    out
}

fn skip_note() -> String {
    format!(
        "SKIP: no {CLASSICAL_GATES_ARE_CALIBRATED_ON_TESSDATA_BEST} under \
         NV_OCR_TESSDATA or ~/.cache/ocr-testdata -- run \
         rust/crates/nv-ocr/scripts/fetch-testdata.sh"
    )
}

#[test]
fn every_071_fixture_declares_all_four_cer_gates() {
    let mut seen = 0usize;
    for name in FIXTURES_IN_SORTED_ORDER {
        let dir = fixtures_dir().join(name);
        let text = std::fs::read_to_string(dir.join("fixture.json"))
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        let v: serde_json::Value = serde_json::from_str(&text).expect("fixture.json");
        for field in [
            "cer_max_classical",
            "cer_max_classical_neutral",
            "cer_max_model",
            "cer_max_model_neutral",
        ] {
            let g = gate_field(&v, field, name);
            assert!(
                g > 0.0 && g <= 1.0,
                "{name}: gates.{field} = {g} is outside (0, 1]"
            );
        }
        assert!(
            gate_field(&v, "cer_max_classical_neutral", name) <= gate_field(&v, "cer_max_classical", name),
            "{name}: the neutral gate must be at least as tight as the raw one. \
             {NEUTRAL_IS_THE_PRIMARY_GATE_BECAUSE_RAW_CHARGES_MARKDOWN_AS_ERRORS}"
        );
        assert!(dir.join("input.png").is_file(), "{name}: no input.png");
        seen += 1;
    }
    assert_eq!(
        seen, 5,
        "the 071 family is five pages; the walk found {seen}"
    );
}

#[test]
fn markup_neutral_strips_decoration_and_collapses_whitespace() {
    assert_eq!(markup_neutral("| a | b |"), "a b");
    assert_eq!(markup_neutral("|---|:---:|\n| a | b |"), "a b");
    assert_eq!(markup_neutral("## Totals\n\nSubtotal:  445.50"), "Totals Subtotal: 445.50");
    assert_eq!(markup_neutral("<td>Qty</td><td>40</td>"), "Qty 40");
    assert_eq!(markup_neutral("**Total:** 481.14"), "Total: 481.14");
    assert_eq!(markup_neutral("- first\n- second"), "first second");
    assert_eq!(markup_neutral("> quoted line"), "quoted line");
    assert_eq!(markup_neutral("a\nb\n\n  c  "), "a b c");
    assert_eq!(markup_neutral("`code` span"), "code span");
    assert_eq!(markup_neutral("Hex bolts M8"), "Hex bolts M8");
}

#[test]
fn markup_neutral_is_idempotent_and_never_grows_the_string() {
    let cases = [
        "| a | b |\n|---|---|\n| 1 | 2 |",
        "## Heading\n\ntext with  spaces\n",
        "<table><tr><td>x</td></tr></table>",
        "plain sentence with no markup at all",
        "",
    ];
    for c in cases {
        let once = markup_neutral(c);
        let twice = markup_neutral(&once);
        assert_eq!(once, twice, "not idempotent on {c:?}");
        assert!(
            once.chars().count() <= c.chars().count(),
            "neutralization grew {c:?} into {once:?}"
        );
    }
}

#[test]
fn neutralizing_a_markdown_rendering_of_the_invoice_beats_scoring_it_raw() {
    let f = load_fixture("071-ocr-layout-invoice");
    let markdown = format!(
        "# INVOICE\n\n| Item | Qty |\n|---|---|\n{}\n",
        f.expected_text.replace('\n', " |\n| ")
    );
    let raw = cer(&f.expected_text, &markdown);
    let neutral = cer(&markup_neutral(&f.expected_text), &markup_neutral(&markdown));
    assert!(
        neutral < raw,
        "markup neutralization did not reduce the distance for a decorated but faithful \
         transcription: raw {raw:.4} vs neutral {neutral:.4}. \
         {NEUTRAL_IS_THE_PRIMARY_GATE_BECAUSE_RAW_CHARGES_MARKDOWN_AS_ERRORS}"
    );
}

#[test]
fn cer_is_a_normalized_edit_distance() {
    assert_eq!(cer("abc", "abc"), 0.0);
    assert!((cer("abc", "abd") - 1.0 / 3.0).abs() < 1e-12);
    assert!((cer("abc", "") - 1.0).abs() < 1e-12);
    assert!((cer("kitten", "sitting") - 3.0 / 6.0).abs() < 1e-12);
}

#[test]
fn e2e_layout_fixtures_classical_gates() {
    let Some((engine, path)) = classical_engine() else {
        eprintln!("{}", skip_note());
        return;
    };
    let scored = score_all_classical(&engine);
    assert_eq!(scored.len(), 5, "the runner must score all five 071 pages");
    let mut over: Vec<String> = Vec::new();
    for s in &scored {
        eprintln!(
            "{:<24} neutral {:.4} (gate {:.4})  raw {:.4} (gate {:.4})",
            s.name, s.neutral, s.neutral_gate, s.raw, s.raw_gate
        );
        if s.neutral > s.neutral_gate {
            over.push(format!(
                "{}: neutral {:.4} > {:.4}",
                s.name, s.neutral, s.neutral_gate
            ));
        }
        if s.raw > s.raw_gate {
            over.push(format!("{}: raw {:.4} > {:.4}", s.name, s.raw, s.raw_gate));
        }
    }
    assert!(
        over.is_empty(),
        "classical OCR regressed past the 071 gates in {}:\n  {}",
        path.display(),
        over.join("\n  ")
    );
}

#[test]
#[ignore]
fn report_layout_fixture_cers() {
    let Some((engine, path)) = classical_engine() else {
        eprintln!("{}", skip_note());
        return;
    };
    let scored = score_all_classical(&engine);
    let mean: f64 = scored.iter().map(|s| s.neutral).sum::<f64>() / scored.len() as f64;
    println!("traineddata: {}", path.display());
    println!("| fixture | neutral | gate | raw | gate |");
    println!("|---|---|---|---|---|");
    for s in &scored {
        println!(
            "| {} | {:.4} | {:.2} | {:.4} | {:.2} |",
            s.name.trim_start_matches("071-ocr-layout-"),
            s.neutral,
            s.neutral_gate,
            s.raw,
            s.raw_gate
        );
    }
    println!("classical headline (mean neutral): {mean:.4}");
    assert_eq!(scored.len(), 5, "the report must cover all five 071 pages");
}
