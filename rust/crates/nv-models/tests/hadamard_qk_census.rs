use std::path::{Path, PathBuf};

const SOURCES: [&str; 2] = [
    "crates/nv-models/src/gemma4.rs",
    "crates/nv-models/src/gemma4_batch_graph.rs",
];

const WINDOW: usize = 3;

fn rust_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p
}

#[test]
fn every_rope_apply_rotates_q_and_k_together() {
    let root = rust_root();
    let mut sites = 0usize;
    let mut naked: Vec<String> = Vec::new();
    for rel in SOURCES {
        let path: &Path = &root.join(rel);
        let src = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("census cannot read {}: {e}. A missing source is a \
                                        finding, not a pass", path.display()));
        let lines: Vec<&str> = src.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if !line.contains("rope.apply(") {
                continue;
            }
            sites += 1;
            let covered = lines[i + 1..(i + 1 + WINDOW).min(lines.len())]
                .iter()
                .any(|l| l.contains("maybe_rotate_qk"));
            if !covered {
                naked.push(format!("{rel}:{}", i + 1));
            }
        }
    }
    assert!(
        sites >= 4,
        "the census found only {sites} rope.apply sites across {SOURCES:?}. It previously found \
         four, so either a path moved or this test is now looking at the wrong files and would \
         pass while checking nothing"
    );
    assert!(
        naked.is_empty(),
        "these rope.apply sites do not rotate Q and K within {WINDOW} lines: {naked:?}. Q and K \
         must be rotated TOGETHER at every site or not at all: the cache stores K in whatever \
         basis its writer used, and a reader that skips the rotation computes Q.K in a \
         different basis and silently attends to the wrong thing. Route the pair through \
         hadamard_kv::maybe_rotate_qk so one flag decides for all of them"
    );
}
