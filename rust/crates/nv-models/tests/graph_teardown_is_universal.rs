
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const EXPECTED_ENGINES: usize = 13;

const EXPECTED_ENGINE_DROPS: usize = 15;

const EXPECTED_CRATES_WITH_ENGINES: usize = 2;

const CAPTURE_EVIDENCE: &str = "CudaGraph";

const RUNNER_DEFINITION: &str = "pub struct CudaGraphRunner";

const RETURNS_THE_MEMPOOL: [&str; 2] = ["GraphTeardown", "cuDeviceGraphMemTrim"];

const RELEASES_THE_LEGACY_STREAM_QUANT_CACHES: [&str; 1] = ["GraphTeardown"];

const CAPTURE_AWARE_TEARDOWN_CONSTRUCTOR: &str = "GraphTeardown::for_capture";

const OWNED_ONLY_TEARDOWN_CONSTRUCTOR: &str = "GraphTeardown::new(";

fn workspace_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate is nested two levels under the cargo workspace")
        .to_path_buf()
}

fn source_roots() -> Vec<(String, PathBuf)> {
    let ws = workspace_dir();
    let mut roots = Vec::new();
    let bin_src = ws.join("src");
    if bin_src.is_dir() {
        roots.push(("<bin>".to_string(), bin_src));
    }
    if let Ok(rd) = std::fs::read_dir(ws.join("crates")) {
        let mut crates: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        crates.sort();
        for c in crates {
            let src = c.join("src");
            if src.is_dir() {
                let name = c
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                roots.push((name, src));
            }
        }
    }
    roots
}

struct SourceFile {
    krate: String,
    rel: String,
    code: String,
}

fn rust_sources(krate: &str, root: &Path, dir: &Path, out: &mut Vec<SourceFile>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            rust_sources(krate, root, &p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            if let Ok(s) = std::fs::read_to_string(&p) {
                let rel = p.strip_prefix(root).unwrap_or(&p).to_string_lossy();
                out.push(SourceFile {
                    krate: krate.to_string(),
                    rel: format!("{krate}/src/{rel}"),
                    code: code_only(&s),
                });
            }
        }
    }
}

fn walk_first_party_sources() -> Vec<SourceFile> {
    let mut out = Vec::new();
    for (krate, root) in source_roots() {
        rust_sources(&krate, &root, &root, &mut out);
    }
    out
}

fn code_only(src: &str) -> String {
    let b = src.as_bytes();
    let n = b.len();
    let mut out: Vec<u8> = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < n {
        let c = b[i];
        if c == b'/' && i + 1 < n && b[i + 1] == b'/' {
            while i < n && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c == b'/' && i + 1 < n && b[i + 1] == b'*' {
            let mut depth = 1usize;
            i += 2;
            while i < n && depth > 0 {
                if b[i] == b'/' && i + 1 < n && b[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if b[i] == b'*' && i + 1 < n && b[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            continue;
        }
        if c == b'r' && !prev_is_ident(&out) {
            if let Some(next) = skip_raw_string(b, i) {
                out.push(b'"');
                out.push(b'"');
                i = next;
                continue;
            }
        }
        if c == b'"' {
            i += 1;
            while i < n {
                if b[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if b[i] == b'"' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            out.push(b'"');
            out.push(b'"');
            continue;
        }
        if c == b'\'' {
            if let Some(next) = skip_char_literal(b, i) {
                out.push(b'\'');
                out.push(b'\'');
                i = next;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    blank_cfg_test_modules_because_a_test_scoped_runner_is_not_a_serving_engine(
        String::from_utf8_lossy(&out).into_owned(),
    )
}

fn blank_cfg_test_modules_because_a_test_scoped_runner_is_not_a_serving_engine(
    code: String,
) -> String {
    let mut out = code.into_bytes();
    let needle = b"#[cfg(";
    let mut from = 0usize;
    while let Some(pos) = out[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + from)
    {
        let attr_open = pos + needle.len() - 1;
        let mut pdepth = 0usize;
        let mut attr_close = attr_open;
        for (k, c) in out[attr_open..].iter().enumerate() {
            match c {
                b'(' => pdepth += 1,
                b')' => {
                    pdepth -= 1;
                    if pdepth == 0 {
                        attr_close = attr_open + k;
                        break;
                    }
                }
                _ => {}
            }
        }
        let attr_text = String::from_utf8_lossy(&out[attr_open..attr_close]).into_owned();
        let cfg_gates_on_test = attr_text
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|tok| tok == "test");
        if !cfg_gates_on_test {
            from = attr_close + 1;
            continue;
        }
        let Some(open) = out[attr_close..]
            .iter()
            .position(|c| *c == b'{')
            .map(|p| p + attr_close)
        else {
            break;
        };
        if out[attr_close..open].iter().any(|c| *c == b';') {
            from = attr_close + 1;
            continue;
        }
        let mut depth = 0usize;
        let mut end = open;
        for (k, c) in out[open..].iter().enumerate() {
            match c {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + k;
                        break;
                    }
                }
                _ => {}
            }
        }
        for c in &mut out[pos..=end] {
            if *c != b'\n' {
                *c = b' ';
            }
        }
        from = end + 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn prev_is_ident(out: &[u8]) -> bool {
    match out.last() {
        Some(c) => (c.is_ascii_alphanumeric() || *c == b'_') && *c != b'b',
        None => false,
    }
}

fn skip_raw_string(b: &[u8], start: usize) -> Option<usize> {
    let n = b.len();
    let mut i = start + 1;
    let hashes = {
        let h = i;
        while i < n && b[i] == b'#' {
            i += 1;
        }
        i - h
    };
    if i >= n || b[i] != b'"' {
        return None;
    }
    i += 1;
    while i < n {
        if b[i] == b'"'
            && b[i + 1..]
                .iter()
                .take(hashes)
                .filter(|c| **c == b'#')
                .count()
                == hashes
        {
            return Some(i + 1 + hashes);
        }
        i += 1;
    }
    Some(n)
}

fn skip_char_literal(b: &[u8], start: usize) -> Option<usize> {
    let n = b.len();
    if start + 1 >= n {
        return None;
    }
    if b[start + 1] == b'\\' {
        let mut i = start + 3;
        while i < n && i < start + 16 {
            if b[i] == b'\'' {
                return Some(i + 1);
            }
            i += 1;
        }
        return None;
    }
    if start + 2 < n && b[start + 2] == b'\'' {
        return Some(start + 3);
    }
    None
}

fn defines_the_graph_runner(code: &str) -> bool {
    code.contains(RUNNER_DEFINITION)
}

fn is_graph_engine(code: &str) -> bool {
    code.contains(CAPTURE_EVIDENCE) && !defines_the_graph_runner(code)
}

const DROP_IMPL_HEADER: &str = "Drop for";

fn token_starts_at(code: &str, at: usize) -> bool {
    !code[..at]
        .bytes()
        .next_back()
        .is_some_and(|c| c.is_ascii_alphanumeric() || c == b'_')
}

fn leading_ident(s: &str) -> String {
    s.trim_start()
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

fn balanced_block(s: &str, from: usize, open: char, close: char) -> &str {
    let mut depth = 0i32;
    let mut end = s.len().saturating_sub(1);
    for (i, ch) in s[from..].char_indices() {
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                end = from + i;
                break;
            }
        }
    }
    &s[from..=end]
}

struct DropImpl {
    ty: String,
    body: String,
}

fn drop_impls(code: &str) -> Vec<DropImpl> {
    let mut out = Vec::new();
    let mut base = 0usize;
    while let Some(rel) = code[base..].find(DROP_IMPL_HEADER) {
        let at = base + rel;
        let after = at + DROP_IMPL_HEADER.len();
        base = after;
        if !token_starts_at(code, at) {
            continue;
        }
        let Some(rel_open) = code[after..].find('{') else {
            break;
        };
        out.push(DropImpl {
            ty: leading_ident(&code[after..]),
            body: balanced_block(code, after + rel_open, '{', '}').to_string(),
        });
    }
    out
}

fn declared_type_body<'a>(code: &'a str, ty: &str) -> Option<&'a str> {
    if ty.is_empty() {
        return None;
    }
    for kw in ["struct ", "enum ", "union "] {
        let mut base = 0usize;
        while let Some(rel) = code[base..].find(kw) {
            let at = base + rel;
            base = at + kw.len();
            if !token_starts_at(code, at) {
                continue;
            }
            let rest = &code[base..];
            let name = leading_ident(rest);
            if name != ty {
                continue;
            }
            let mut i = rest.len() - rest.trim_start().len() + name.len();
            let bytes = rest.as_bytes();
            while i < bytes.len() {
                match bytes[i] {
                    b'{' => return Some(balanced_block(rest, i, '{', '}')),
                    b'(' => return Some(balanced_block(rest, i, '(', ')')),
                    b';' => return Some(""),
                    _ => i += 1,
                }
            }
            return Some("");
        }
    }
    None
}

fn mentions_type(body: &str, ty: &str) -> bool {
    let mut base = 0usize;
    while let Some(rel) = body[base..].find(ty) {
        let at = base + rel;
        base = at + ty.len();
        let ends_token = !body[base..]
            .bytes()
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == b'_');
        if token_starts_at(body, at) && ends_token {
            return true;
        }
    }
    false
}

fn types_declared_here_whose_bodies_name_a_captured_graph(code: &str) -> Vec<String> {
    let mut out = Vec::new();
    for kw in ["struct ", "enum ", "union "] {
        let mut base = 0usize;
        while let Some(rel) = code[base..].find(kw) {
            let at = base + rel;
            base = at + kw.len();
            if !token_starts_at(code, at) {
                continue;
            }
            let name = leading_ident(&code[base..]);
            if !name.is_empty()
                && declared_type_body(code, &name).is_some_and(|b| b.contains(CAPTURE_EVIDENCE))
                && !out.contains(&name)
            {
                out.push(name);
            }
        }
    }
    out
}

fn holds_a_captured_graph(code: &str, ty: &str) -> Option<bool> {
    let body = declared_type_body(code, ty)?;
    if body.contains(CAPTURE_EVIDENCE) {
        return Some(true);
    }
    Some(
        types_declared_here_whose_bodies_name_a_captured_graph(code)
            .iter()
            .any(|t| t != ty && mentions_type(body, t)),
    )
}

fn engine_drops(code: &str) -> Vec<DropImpl> {
    drop_impls(code)
        .into_iter()
        .filter(|d| holds_a_captured_graph(code, &d.ty).unwrap_or(true))
        .collect()
}

fn drop_body_satisfies(body: &str, tokens: &[&str]) -> bool {
    tokens.iter().any(|t| body.contains(t))
}

fn tears_down_the_mempool(code: &str) -> bool {
    let ds = engine_drops(code);
    !ds.is_empty()
        && ds
            .iter()
            .all(|d| drop_body_satisfies(&d.body, &RETURNS_THE_MEMPOOL))
}

fn owned_only_teardown_arguments(code: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut base = 0usize;
    while let Some(rel) = code[base..].find(OWNED_ONLY_TEARDOWN_CONSTRUCTOR) {
        let open = base + rel + OWNED_ONLY_TEARDOWN_CONSTRUCTOR.len() - 1;
        base = open + 1;
        let arg = balanced_block(code, open, '(', ')');
        let inner = arg
            .strip_prefix('(')
            .and_then(|s| s.strip_suffix(')'))
            .unwrap_or(arg);
        out.push(inner.trim().to_string());
    }
    out
}

fn engines(sources: &[SourceFile]) -> Vec<&SourceFile> {
    sources
        .iter()
        .filter(|f| is_graph_engine(&f.code))
        .collect()
}

fn offending_engine_drops(engines: &[&SourceFile], tokens: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for f in engines {
        for d in engine_drops(&f.code) {
            if !drop_body_satisfies(&d.body, tokens) {
                out.push(format!("{}::{}", f.rel, d.ty));
            }
        }
    }
    out
}

fn engine_drop_names(engines: &[&SourceFile]) -> Vec<String> {
    let mut out = Vec::new();
    for f in engines {
        for d in engine_drops(&f.code) {
            out.push(format!("{}::{}", f.rel, d.ty));
        }
    }
    out
}

#[test]
fn every_graph_engine_returns_its_mempool_on_teardown() {
    let sources = walk_first_party_sources();
    assert!(
        !sources.is_empty(),
        "walked no .rs files under {}; discovery is broken",
        workspace_dir().display()
    );

    let engines = engines(&sources);
    let offenders = offending_engine_drops(&engines, &RETURNS_THE_MEMPOOL);

    assert_eq!(
        engines.len(),
        EXPECTED_ENGINES,
        "expected exactly {EXPECTED_ENGINES} graph engine(s) among {} files, found {}. A FLOOR \
         would have let three stop matching {CAPTURE_EVIDENCE} unnoticed. If an engine was added \
         or removed on purpose, change EXPECTED_ENGINES in the same commit: {:?}",
        sources.len(),
        engines.len(),
        engines.iter().map(|f| f.rel.as_str()).collect::<Vec<_>>()
    );
    let found = engine_drop_names(&engines);
    assert_eq!(
        found.len(),
        EXPECTED_ENGINE_DROPS,
        "expected exactly {EXPECTED_ENGINE_DROPS} engine `Drop` impl(s) across {} engine file(s), \
         found {}: {found:?}\n\n\
         An engine with NO Drop at all is invisible to every offender list below, so this count is \
         the only thing that catches it -- and it is equally the only thing standing between the \
         per-Drop discriminator and a silent regression, since a discriminator that stops \
         recognising a type as graph-holding drops that engine out of every assertion and leaves \
         the gate green. If an engine was added or removed on purpose, change EXPECTED_ENGINE_DROPS \
         in the same commit.",
        engines.len(),
        found.len()
    );
    eprintln!(
        "graph teardown: {} files walked, {} graph engine(s), {} engine Drop impl(s)",
        sources.len(),
        engines.len(),
        found.len()
    );

    assert!(
        offenders.is_empty(),
        "graph engine(s) never return the graph mempool on teardown: {offenders:?}\n\n\
         Destroying a captured graph does NOT hand back its reserved physical pages -- only \
         cuDeviceGraphMemTrim does -- so the next component's plain eager allocation fails with \
         CUDA_ERROR_INVALID_VALUE inside an unrelated forward (#59). Route teardown through \
         graph_teardown::GraphTeardown. The call must stand inside the `impl Drop` block for that \
         very type: this gate does not follow a helper, and it does not let a compliant Drop \
         beside it vouch for one that omits the step."
    );
}

#[test]
fn every_graph_engine_releases_the_legacy_stream_quant_caches_on_teardown() {
    let sources = walk_first_party_sources();
    let engines = engines(&sources);
    assert_eq!(
        engine_drop_names(&engines).len(),
        EXPECTED_ENGINE_DROPS,
        "engine Drop discovery disagrees with the mempool gate; fix that first"
    );
    let offenders = offending_engine_drops(&engines, &RELEASES_THE_LEGACY_STREAM_QUANT_CACHES);
    assert!(
        offenders.is_empty(),
        "graph engine(s) tear down without releasing nv_quant's LEGACY-stream caches: \
         {offenders:?}\n\n\
         nv_quant keeps a cublasLt handle and a 64 MiB nvfp4 workspace in process-wide statics \
         keyed on the raw CUstream pointer, and the entries keyed on the context's default stream \
         hold CudaEvents belonging to THIS CudaContext. The next component opens its own \
         CudaContext, finds those entries under the same key, and records this context's event \
         onto its own stream; the driver refuses, cudarc cannot return an error from a SyncOnDrop \
         and stashes it in the new context's error_state, and the next check_err -- which every \
         bind_to_thread does -- surfaces it as CUDA_ERROR_INVALID_VALUE from a plain eager \
         allocation inside an unrelated forward. A hand-rolled teardown releases only the streams \
         it forked, so cuDeviceGraphMemTrim plus release_stream_resources(self.forked) is NOT \
         compliance here. Route the Drop through graph_teardown::GraphTeardown, which releases \
         owned streams AND the legacy stream and then trims."
    );
}

#[test]
fn no_engine_hands_a_borrowed_capture_stream_to_the_owned_only_teardown_constructor() {
    let sources = walk_first_party_sources();
    let mut owned_only = 0usize;
    let mut capture_aware = 0usize;
    let mut offenders: Vec<String> = Vec::new();
    for f in &sources {
        capture_aware += f.code.matches(CAPTURE_AWARE_TEARDOWN_CONSTRUCTOR).count();
        for arg in owned_only_teardown_arguments(&f.code) {
            owned_only += 1;
            if arg.contains("capture") {
                offenders.push(format!(
                    "{}: {OWNED_ONLY_TEARDOWN_CONSTRUCTOR}{arg})",
                    f.rel
                ));
            }
        }
    }
    assert!(
        owned_only > 0 && capture_aware > 0,
        "found {owned_only} {OWNED_ONLY_TEARDOWN_CONSTRUCTOR} call(s) and {capture_aware} \
         {CAPTURE_AWARE_TEARDOWN_CONSTRUCTOR} call(s); this gate is pure string matching, so a \
         renamed constructor makes it match nothing and pass vacuously"
    );
    assert!(
        offenders.is_empty(),
        "engine(s) build teardown from a CaptureStream's stream with the owned-only constructor: \
         {offenders:?}\n\n\
         On a candle_core::Device::new_cuda_with_stream device CaptureStream::for_device does not \
         fork -- stream() IS candle's device stream, shared by the eager path and by every other \
         engine on that device. GraphTeardown::new then calls \
         nv_quant::release_stream_resources on it, destroying the raw cublasLt handle and freeing \
         the 64 MiB nvfp4 workspace whose addresses are already baked into another live engine's \
         captured graph; the next replay reads freed memory. CaptureStream::owns_stream is the \
         answer, and {CAPTURE_AWARE_TEARDOWN_CONSTRUCTOR} is the constructor that asks it. Reach \
         for {OWNED_ONLY_TEARDOWN_CONSTRUCTOR} only with a stream this engine forked itself."
    );
}

#[test]
fn the_walk_reaches_engines_outside_the_crate_hosting_this_test() {
    let sources = walk_first_party_sources();
    let engines = engines(&sources);
    let crates: BTreeSet<&str> = engines.iter().map(|f| f.krate.as_str()).collect();
    assert!(
        crates.len() >= EXPECTED_CRATES_WITH_ENGINES,
        "engines found in {crates:?}, fewer than {EXPECTED_CRATES_WITH_ENGINES} crates. This gate \
         used to walk only its own crate's src, so nv-specdecode's two graph engines -- neither of \
         which had any impl Drop at all -- were invisible to it. Narrowing the walk back to one \
         crate reintroduces exactly that blindness."
    );
    assert!(
        crates.iter().any(|c| *c != env!("CARGO_PKG_NAME")),
        "every engine found lives in {}, the crate hosting this test; the cross-crate walk is not \
         reaching sibling crates",
        env!("CARGO_PKG_NAME")
    );
}

#[test]
fn the_discriminator_separates_an_engine_from_a_plain_module() {
    let engine_ok = code_only("let g: CudaGraph = ...;\nimpl Drop for E { fn drop(&mut self) { GraphTeardown::new(&s).run(|| r.invalidate()); } }");
    assert!(is_graph_engine(&engine_ok));
    assert!(tears_down_the_mempool(&engine_ok));

    let engine_legacy = code_only("let g: CudaGraph = ...;\nimpl Drop for E { fn drop(&mut self) { unsafe { cuDeviceGraphMemTrim(devh) }; } }");
    assert!(is_graph_engine(&engine_legacy));
    assert!(
        tears_down_the_mempool(&engine_legacy),
        "the four engines that predate the shared module hand-roll the trim and must stay legal"
    );
    assert!(
        !drop_body_satisfies(
            &engine_drops(&engine_legacy)[0].body,
            &RELEASES_THE_LEGACY_STREAM_QUANT_CACHES
        ),
        "a hand-rolled trim returns the mempool but leaves the legacy-stream cublasLt and nvfp4 \
         entries behind; the two properties must not share one verdict"
    );

    let two_drops_one_compliant = code_only(
        "let g: CudaGraph = ...;\nimpl Drop for A { fn drop(&mut self) { unsafe { cuDeviceGraphMemTrim(d) }; } }\nimpl Drop for B { fn drop(&mut self) { r.invalidate(); } }",
    );
    assert_eq!(engine_drops(&two_drops_one_compliant).len(), 2);
    assert!(
        !tears_down_the_mempool(&two_drops_one_compliant),
        "a compliant Drop must not vouch for the one beside it: laguna_step_graph.rs holds three \
         Drop impls, gemma4_graph.rs two, and both deepseek_ocr graph files two, so file-level \
         matching let a new engine inherit a pass from a neighbour"
    );

    let helper_beside_an_engine = code_only(
        "struct CtxErrDrain(Arc<CudaContext>);\nimpl Drop for CtxErrDrain { fn drop(&mut self) { let _ = self.0.check_err(); } }\nstruct E { runner: CudaGraphRunner }\nimpl Drop for E { fn drop(&mut self) { unsafe { cuDeviceGraphMemTrim(d) }; } }",
    );
    assert_eq!(
        engine_drops(&helper_beside_an_engine)
            .iter()
            .map(|d| d.ty.clone())
            .collect::<Vec<_>>(),
        vec!["E".to_string()],
        "a Drop whose type holds no captured graph is not a teardown: CtxErrDrain is a newtype \
         over an Arc<CudaContext> that drains the deferred error, and requiring it to trim would \
         make the gate red for something that is not the defect"
    );
    assert!(tears_down_the_mempool(&helper_beside_an_engine));

    let engine_beside_a_helper = code_only(
        "struct CtxErrDrain(Arc<CudaContext>);\nimpl Drop for CtxErrDrain { fn drop(&mut self) { let _ = self.0.check_err(); } }\nstruct E { runner: CudaGraphRunner }\nimpl Drop for E { fn drop(&mut self) { r.invalidate(); } }",
    );
    assert!(
        !tears_down_the_mempool(&engine_beside_a_helper),
        "excluding the helper Drop must not also excuse the engine Drop in the same file"
    );

    let runner_behind_a_state_struct = code_only(
        "struct GraphState { runner: CudaGraphRunner }\nstruct E { state: Mutex<GraphState> }\nimpl Drop for E { fn drop(&mut self) { unsafe { cuDeviceGraphMemTrim(d) }; } }\nstruct CtxErrDrain(Arc<CudaContext>);\nimpl Drop for CtxErrDrain { fn drop(&mut self) { let _ = self.0.check_err(); } }",
    );
    assert_eq!(
        engine_drops(&runner_behind_a_state_struct)
            .iter()
            .map(|d| d.ty.clone())
            .collect::<Vec<_>>(),
        vec!["E".to_string()],
        "an engine whose runner sits one type away -- gemma4_vision_graph.rs keeps its \
         CudaGraphRunner inside Mutex<GraphState> -- is still an engine, and the helper Drop \
         beside it is still not one. Reading only the Drop type's own field list dropped \
         Gemma4VisionGraph out of the census the day GraphState was introduced, and only the \
         pinned EXPECTED_ENGINE_DROPS count caught it"
    );

    let drop_for_a_type_declared_elsewhere = code_only(
        "let g: CudaGraph = ...;\nimpl Drop for Imported { fn drop(&mut self) { r.invalidate(); } }",
    );
    assert_eq!(
        engine_drops(&drop_for_a_type_declared_elsewhere).len(),
        1,
        "a Drop whose type this walk cannot see must fail CLOSED; treating an unknown type as a \
         non-engine is how an engine slips the gate by moving its struct one file over"
    );

    assert_eq!(
        owned_only_teardown_arguments(&code_only(
            "let td = GraphTeardown::new(self.capture.stream());"
        )),
        vec!["self.capture.stream()".to_string()],
        "the argument scanner must survive nested parentheses, or defect #1 reads as compliant"
    );
    assert_eq!(
        owned_only_teardown_arguments(&code_only("let td = GraphTeardown::new(&self.forked);")),
        vec!["&self.forked".to_string()]
    );
    assert!(owned_only_teardown_arguments(&code_only(
        "let td = GraphTeardown::for_capture(&self.capture);"
    ))
    .is_empty());

    let engine_bad = code_only(
        "let g: CudaGraph = ...;\nimpl Drop for E { fn drop(&mut self) { r.invalidate(); } }",
    );

    let trim_outside_drop = code_only("let g: CudaGraph = ...;\nfn reset(&mut self) { cuDeviceGraphMemTrim(o); }\nimpl Drop for E { fn drop(&mut self) { r.invalidate(); } }");
    assert!(
        !tears_down_the_mempool(&trim_outside_drop),
        "a trim reachable only from reset()/prefill() is not a teardown -- this is what scored \
         graph_engine.rs compliant while it leaked on every drop"
    );
    let token_in_a_comment = code_only("let g: CudaGraph = ...;\n/// Reach it as GraphTeardown.\nimpl Drop for E { fn drop(&mut self) { r.invalidate(); } }");
    assert!(
        !tears_down_the_mempool(&token_in_a_comment),
        "a doc comment naming the token must not buy a pass -- gemma4_batch_graph.rs:104 has \
         exactly such a comment"
    );
    let token_in_a_string = code_only(
        "let g: CudaGraph = ...;\nimpl Drop for E { fn drop(&mut self) { eprintln!(\"no GraphTeardown here\"); } }",
    );
    assert!(
        !tears_down_the_mempool(&token_in_a_string),
        "a string literal naming the token is the comment bypass with quotes instead of slashes"
    );
    let brace_in_a_string = code_only(
        "let g: CudaGraph = ...;\nimpl Drop for E { fn drop(&mut self) { bail!(\"}}}}\"); unsafe { cuDeviceGraphMemTrim(d) }; } }",
    );
    assert!(
        tears_down_the_mempool(&brace_in_a_string),
        "an unbalanced brace inside a literal must not close the Drop block early and hide a real \
         trim"
    );
    let raw_string_with_slashes = code_only(
        "let g: CudaGraph = ...;\nconst J: &str = r#\"{\"a\": \"//\"}\"#;\nimpl Drop for E { fn drop(&mut self) { unsafe { cuDeviceGraphMemTrim(d) }; } }",
    );
    assert!(
        tears_down_the_mempool(&raw_string_with_slashes),
        "a raw string is not a comment: eagle3_loader.rs carries one, and mis-lexing it would eat \
         the Drop that follows"
    );
    let byte_char_quote = code_only(
        "let g: CudaGraph = ...;\nfn f(b: u8) -> bool { b == b'\"' }\nimpl Drop for E { fn drop(&mut self) { unsafe { cuDeviceGraphMemTrim(d) }; } }",
    );
    assert!(
        tears_down_the_mempool(&byte_char_quote),
        "a quote inside a char literal must not open a string -- dots_ocr/parse.rs has b'\"'"
    );
    let lifetime_is_not_a_literal = code_only(
        "let g: CudaGraph = ...;\nimpl<'a> Drop for E<'a> { fn drop(&mut self) { unsafe { cuDeviceGraphMemTrim(d) }; } }",
    );
    assert!(
        tears_down_the_mempool(&lifetime_is_not_a_literal),
        "a lifetime apostrophe must not open a char literal and swallow the Drop body, and a \
         generic engine's header reads `impl<'a> Drop for E<'a>` -- matching `impl Drop` missed \
         every one of them"
    );
    let another_trait_ending_in_drop = code_only(
        "let g: CudaGraph = ...;\nimpl NoDrop for E { fn f(&self) { unsafe { cuDeviceGraphMemTrim(d) }; } }",
    );
    assert!(
        !tears_down_the_mempool(&another_trait_ending_in_drop),
        "only the Drop trait counts: a trait whose name merely ends in Drop is not a teardown"
    );
    assert!(is_graph_engine(&engine_bad));
    assert!(
        !tears_down_the_mempool(&engine_bad),
        "an engine that destroys a graph and stops there is exactly the #59 defect"
    );

    let runner_module =
        code_only("pub struct CudaGraphRunner { cached: HashMap<u64, RawGraph> }\nimpl Drop for RawGraph { fn drop(&mut self) { graph::destroy(g); } }");
    assert!(
        !is_graph_engine(&runner_module),
        "the module that DEFINES the runner is the primitive, not an engine: its Drop destroys one \
         graph while the owning engine still holds others, and the trim belongs to the owner"
    );

    let not_an_engine = code_only("fn helper() { let x = 1; }");
    assert!(!is_graph_engine(&not_an_engine));
}
