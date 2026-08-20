use std::path::{Path, PathBuf};

const SHARED_WGSL_FNS: &[&str] = &[
    "g4w_gemv_bf16_vec8_pk",
    "g4w_gemv_bf16_vec8_pk3",
    "g4w_pair_word",
    "g4w_vec8_acc",
    "g4w_rope_bf16_f32",
    "g4w_flash_splitk_stage2_pk",
];

const ESCAPED_TEMPLATE_FN: &str = "g4w_pair_word";

const ESCAPED_TEMPLATE_HOST: &str = "gemma4_e4b_wgpu.rs";

fn wgsl_fn_body(src: &str, name: &str) -> Option<String> {
    let start = src.find(&format!("fn {name}("))?;
    let open = start + src[start..].find('{')?;
    let mut depth = 0usize;
    for (i, c) in src[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let body = &src[start..open + i + 1];
                    return Some(body.split_whitespace().collect::<Vec<_>>().join(" "));
                }
            }
            _ => {}
        }
    }
    None
}

fn shader_sources() -> Vec<(String, String)> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut roots: Vec<PathBuf> = vec![manifest.join("src")];
    roots.push(manifest.join("../nv-kernels/wgsl"));
    let mut out = Vec::new();
    for root in roots {
        let dir = std::fs::read_dir(&root)
            .unwrap_or_else(|e| panic!("read {}: {e}", root.display()));
        for e in dir.filter_map(|e| e.ok()) {
            let p = e.path();
            let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
            if ext != "rs" && ext != "wgsl" {
                continue;
            }
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            if let Ok(s) = std::fs::read_to_string(&p) {
                out.push((name, s));
            }
        }
    }
    assert!(
        out.len() > 40,
        "found only {} shader-bearing sources; this guard is scanning the wrong roots and \
         would report every function as singly-defined by seeing none of them",
        out.len()
    );
    out
}

#[test]
fn every_shared_wgsl_fn_is_defined_in_exactly_one_place() {
    let sources = shader_sources();
    for name in SHARED_WGSL_FNS {
        let needle = format!("fn {name}(");
        let mut hosts: Vec<&str> = sources
            .iter()
            .filter(|(f, s)| s.contains(&needle) && !is_escaped_template(name, f))
            .map(|(f, _)| f.as_str())
            .collect();
        hosts.sort_unstable();
        assert_eq!(
            hosts.len(),
            1,
            "`fn {name}(` is defined in {hosts:?}. A shared WGSL function copied into a second \
             source is the drift this guard exists to stop: the copies compile, agree on the day \
             they are made, and diverge silently afterwards. Edit the single definition, or \
             compose from it"
        );
    }
}

fn is_escaped_template(name: &str, file: &str) -> bool {
    name == ESCAPED_TEMPLATE_FN && file == ESCAPED_TEMPLATE_HOST
}

#[test]
fn the_escaped_template_copy_still_matches_the_shared_body() {
    let sources = shader_sources();
    let needle = format!("fn {ESCAPED_TEMPLATE_FN}(");
    let shared = sources
        .iter()
        .find(|(f, s)| s.contains(&needle) && f.ends_with(".wgsl"))
        .map(|(_, s)| wgsl_fn_body(s, ESCAPED_TEMPLATE_FN).expect("shared body parses"))
        .expect("the shared definition is in a .wgsl file");
    let host = sources
        .iter()
        .find(|(f, _)| f == ESCAPED_TEMPLATE_HOST)
        .map(|(_, s)| s.replace("\\n", "\n"))
        .expect("the escaped-template host exists");
    let template = wgsl_fn_body(&host, ESCAPED_TEMPLATE_FN).expect(
        "the escaped template copy is gone from the host -- if it now composes from the shared \
         source, delete ESCAPED_TEMPLATE_FN so the exactly-one-place check covers it too",
    );
    assert_eq!(
        template, shared,
        "the escaped `{ESCAPED_TEMPLATE_FN}` template has DIVERGED from the shared body. It is \
         the one copy this guard permits, because it is re-emitted through a format! generator \
         that cannot include_str! -- so it is also the one copy nothing else can catch"
    );
}
