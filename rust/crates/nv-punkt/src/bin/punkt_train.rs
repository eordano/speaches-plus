use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::exit;

use nv_punkt::{PunktTrainer, CURATED_ABBREVS};

fn clean(text: &str) -> String {
    let mut out = String::new();
    let mut in_fence = false;
    let mut in_src = false;
    for line in text.lines() {
        let lt = line.trim_start();
        if lt.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        let low = lt.to_ascii_lowercase();
        if low.starts_with("#+begin_") {
            in_src = true;
            continue;
        }
        if low.starts_with("#+end_") {
            in_src = false;
            continue;
        }
        if in_fence || in_src {
            continue;
        }
        if low.starts_with("#+") || lt.starts_with('|') {
            continue;
        }
        if lt.is_empty() {
            out.push('\n');
            continue;
        }
        let stripped = lt.trim_start_matches(['#', '*', '>']).trim_start();
        out.push_str(stripped);
        out.push('\n');
    }
    out
}

fn clean_type(s: &str) -> bool {
    s == "##number##"
        || (!s.is_empty()
            && s.chars().any(|c| c.is_ascii_alphabetic())
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '\'')))
}

fn main() {
    let mut out: Option<PathBuf> = None;
    let mut inputs: Vec<PathBuf> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = args.next().map(PathBuf::from),
            other => inputs.push(PathBuf::from(other)),
        }
    }
    if inputs.is_empty() {
        eprintln!("usage: punkt-train [--out DIR] <corpus files...>");
        exit(2);
    }

    let mut trainer = PunktTrainer::new();
    let mut total_bytes = 0usize;
    for p in &inputs {
        let raw = match fs::read_to_string(p) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip {}: {e}", p.display());
                continue;
            }
        };
        let text = clean(&raw);
        total_bytes += text.len();
        trainer.train(&text);
    }
    trainer.finalize();

    let params = trainer.params();
    let mut abbrevs: Vec<&str> = params
        .abbrev_types
        .iter()
        .map(String::as_str)
        .filter(|s| {
            s.chars().count() >= 2
                && s.chars().count() <= 12
                && clean_type(s)
                && !CURATED_ABBREVS.contains(s)
        })
        .collect();
    abbrevs.sort_unstable();

    let mut starters: Vec<&str> = params
        .sent_starters
        .iter()
        .map(String::as_str)
        .filter(|s| clean_type(s) && s.chars().count() <= 15)
        .collect();
    starters.sort_unstable();

    let mut collocs: Vec<(&str, &str)> = params
        .collocations
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .filter(|(a, b)| clean_type(a) && clean_type(b) && a.len() <= 20 && b.len() <= 20)
        .collect();
    collocs.sort_unstable();

    let ortho: BTreeMap<&str, u8> = params
        .ortho_context
        .iter()
        .filter(|(t, &f)| {
            let n = t.chars().count();
            f != 0
                && clean_type(t)
                && (2..=20).contains(&n)
                && !(n == 2 && t.ends_with('.'))
                && trainer.type_count(t) + trainer.type_count(&format!("{t}.")) >= 3
        })
        .map(|(t, &f)| (t.as_str(), f))
        .collect();

    eprintln!(
        "corpus {} bytes | abbrevs {} | starters {} | collocations {} | ortho {}",
        total_bytes,
        abbrevs.len(),
        starters.len(),
        collocs.len(),
        ortho.len()
    );

    if let Some(dir) = out {
        fs::create_dir_all(&dir).expect("create out dir");
        let join = |lines: Vec<String>| lines.join("\n") + "\n";
        fs::write(
            dir.join("abbrev_types.txt"),
            join(abbrevs.iter().map(|a| a.to_string()).collect()),
        )
        .expect("write abbrev_types.txt");
        fs::write(
            dir.join("sent_starters.txt"),
            join(starters.iter().map(|s| s.to_string()).collect()),
        )
        .expect("write sent_starters.txt");
        fs::write(
            dir.join("collocations.tab"),
            join(collocs.iter().map(|(a, b)| format!("{a}\t{b}")).collect()),
        )
        .expect("write collocations.tab");
        fs::write(
            dir.join("ortho_context.tab"),
            join(ortho.iter().map(|(t, f)| format!("{t}\t{f}")).collect()),
        )
        .expect("write ortho_context.tab");
        eprintln!("wrote punkt_tab tables to {}", dir.display());
    }
}
