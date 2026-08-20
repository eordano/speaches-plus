use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const MIN_SIBLING_MODULES_CHECKED: usize = 20;

const WHY_THE_DECLARATION_MUST_BE_THE_DIRECTORYS_OWN: &str =
    "`mod x;` in a/mod.rs declares a/x.rs and nothing else. A global set of every `mod` name \
     anywhere under src/ accepts b/x.rs as soon as some unrelated a/mod.rs says `mod x;`, and \
     the tree carries several such names (types, audio, wire, vad, diarization, gemma). A \
     planted src/vad/types.rs, declared nowhere, passed the global-set version of this gate.";

fn module_file_of(dir: &Path) -> Option<PathBuf> {
    let inner = dir.join("mod.rs");
    if inner.is_file() {
        return Some(inner);
    }
    let outer = dir.with_extension("rs");
    outer.is_file().then_some(outer)
}

fn module_dirs(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    let mut subdirs = Vec::new();
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            subdirs.push(p);
        }
    }
    subdirs.sort();
    for d in subdirs {
        if module_file_of(&d).is_some() {
            out.push(d.clone());
        }
        module_dirs(&d, out);
    }
}

fn collect_rs_files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_rs_files(&p, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(p);
        }
    }
}

fn mod_names_in(text: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for line in text.lines() {
        let t = line.trim();
        let rest = t
            .strip_prefix("pub mod ")
            .or_else(|| t.strip_prefix("mod "))
            .or_else(|| t.strip_prefix("pub(crate) mod "))
            .or_else(|| t.strip_prefix("pub(super) mod "));
        if let Some(rest) = rest {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                names.insert(name);
            }
        }
    }
    names
}

fn path_targets_under(src: &Path) -> BTreeSet<String> {
    let mut files = Vec::new();
    collect_rs_files(src, &mut files);
    let mut out = BTreeSet::new();
    for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else {
            continue;
        };
        for line in text.lines() {
            if let Some(rest) = line.trim().strip_prefix("#[path = \"") {
                if let Some(end) = rest.find('"') {
                    out.insert(rest[..end].to_string());
                }
            }
        }
    }
    out
}

fn siblings_of(dir: &Path) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in rd.flatten() {
        let p = e.path();
        let is_dir_module = p.is_dir() && module_file_of(&p).is_some();
        let is_rs = p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("rs");
        if !is_dir_module && !is_rs {
            continue;
        }
        let Some(stem) = (if is_dir_module {
            p.file_name().and_then(|s| s.to_str())
        } else {
            p.file_stem().and_then(|s| s.to_str())
        }) else {
            continue;
        };
        if matches!(stem, "mod" | "lib" | "main") {
            continue;
        }
        if out.iter().any(|(s, _)| s == stem) {
            continue;
        }
        let file = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        out.push((stem.to_string(), file));
    }
    out.sort();
    out
}

fn undeclared_in_dir(
    declared_here: &BTreeSet<String>,
    path_targets: &BTreeSet<String>,
    siblings: &[(String, String)],
) -> Vec<String> {
    siblings
        .iter()
        .filter(|(stem, file)| {
            !declared_here.contains(stem.as_str()) && !path_targets.contains(file.as_str())
        })
        .map(|(stem, file)| format!("`{file}` exists but `mod {stem};` is not declared there"))
        .collect()
}

#[test]
fn every_sibling_module_is_declared_in_its_own_module_file() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut dirs = Vec::new();
    module_dirs(&src, &mut dirs);
    assert!(
        !dirs.is_empty(),
        "found no module directory under {} -- the walker is broken, not the tree",
        src.display()
    );

    let path_targets = path_targets_under(&src);
    let mut units: Vec<(PathBuf, PathBuf)> = vec![(src.clone(), src.join("lib.rs"))];
    for d in &dirs {
        let f = module_file_of(d).expect("module_dirs only yields dirs with a module file");
        units.push((d.clone(), f));
    }

    let mut problems: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for (dir, module_file) in &units {
        let text = std::fs::read_to_string(module_file)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", module_file.display()));
        let declared_here = mod_names_in(&text);
        let siblings = siblings_of(dir);
        checked += siblings.len();
        for p in undeclared_in_dir(&declared_here, &path_targets, &siblings) {
            problems.push(format!(
                "{}: {p} in {} (and no `#[path]` anywhere under src/ points at it) -- a clean \
                 checkout will not compile if anything references it",
                dir.strip_prefix(&src).unwrap_or(dir).display(),
                module_file
                    .strip_prefix(&src)
                    .unwrap_or(module_file)
                    .display()
            ));
        }
    }

    assert!(
        checked > MIN_SIBLING_MODULES_CHECKED,
        "only checked {checked} sibling modules across {} module file(s) -- too few to be a real \
         guard, the walker is probably wrong",
        units.len()
    );
    assert!(
        problems.is_empty(),
        "undeclared module(s) -- this is the exact shape that broke main. \
         {WHY_THE_DECLARATION_MUST_BE_THE_DIRECTORYS_OWN}\n  {}",
        problems.join("\n  ")
    );
    eprintln!(
        "[mod-guard] {checked} sibling modules across {} module file(s), each declared by its own",
        units.len()
    );
}

#[test]
fn the_scanner_catches_a_module_whose_name_is_declared_only_in_another_directory() {
    let declared_here: BTreeSet<String> = ["sibling_declared_here".to_string()]
        .into_iter()
        .collect();
    let path_targets: BTreeSet<String> = ["reached_by_path.rs".to_string()].into_iter().collect();
    let siblings = vec![
        (
            "sibling_declared_here".to_string(),
            "sibling_declared_here.rs".to_string(),
        ),
        ("reached_by_path".to_string(), "reached_by_path.rs".to_string()),
        ("types".to_string(), "types.rs".to_string()),
    ];

    let found = undeclared_in_dir(&declared_here, &path_targets, &siblings);
    assert_eq!(
        found.len(),
        1,
        "exactly the sibling this directory never declares must be flagged. \
         {WHY_THE_DECLARATION_MUST_BE_THE_DIRECTORYS_OWN}\ngot: {found:?}"
    );
    assert!(
        found[0].contains("types.rs"),
        "the flagged entry must be the undeclared one: {found:?}"
    );
}

#[test]
fn the_crate_root_and_the_two_thousand_eighteen_style_directories_are_all_walked() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut dirs = Vec::new();
    module_dirs(&src, &mut dirs);

    let outer_style: Vec<&PathBuf> = dirs
        .iter()
        .filter(|d| !d.join("mod.rs").is_file() && d.with_extension("rs").is_file())
        .collect();
    assert!(
        !outer_style.is_empty(),
        "src/ carries `foo.rs` + `foo/` module directories (chat_engine, chat_engine_wgpu); a \
         walker that only accepts `foo/mod.rs` never checks a single file inside them"
    );

    assert!(
        src.join("lib.rs").is_file(),
        "the crate root is walked as a module directory whose module file is lib.rs; without it \
         every src/*.rs at the top level goes unchecked"
    );
    let root_siblings = siblings_of(&src);
    assert!(
        root_siblings.len() >= 10,
        "only {} top-level siblings found under {}; the crate-root unit is not doing its job",
        root_siblings.len(),
        src.display()
    );
}
