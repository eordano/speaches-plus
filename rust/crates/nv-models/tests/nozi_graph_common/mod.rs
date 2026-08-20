#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};

use nv_kernels::wgpu_backend::dispatch;

pub const SUBGROUP_LANES: u32 = 32;

pub fn to_msl(tag: &str, source: &str) -> String {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{tag}: wgsl parse: {}", e.message()));
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{tag}: validate: {e}"));
    let mut opts = naga::back::msl::Options {
        lang_version: (3, 0),
        ..Default::default()
    };
    opts.lang_version = (3, 0);
    naga::back::msl::write_string(
        &module,
        &info,
        &opts,
        &naga::back::msl::PipelineOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{tag}: msl-out: {e}"))
    .0
}

#[derive(Clone, Debug)]
pub struct GraphEntry {
    pub src: String,
    pub name: String,
    pub workgroup_vars: Vec<String>,
    pub thread_arrays: Vec<String>,
}

fn wgsl_name(msl_name: &str) -> String {
    msl_name.strip_suffix('_').unwrap_or(msl_name).to_string()
}

pub fn entries_of(tag: &str, msl: &str) -> Vec<GraphEntry> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while let Some(at) = msl[i..].find("kernel void ") {
        let start = i + at;
        let rest = &msl[start..];
        let name_end = rest.find('(').expect("kernel signature");
        let name = rest["kernel void ".len()..name_end].trim().to_string();
        let end = rest.find("\n}\n").unwrap_or(rest.len());
        let mut workgroup_vars = Vec::new();
        let mut thread_arrays = Vec::new();
        for l in rest[..end].lines() {
            let t = l.trim().trim_start_matches(", ");
            if let Some((head, tail)) = t.split_once(' ') {
                if head.starts_with("type_") && tail.ends_with(" = {};") {
                    thread_arrays.push(tail.trim_end_matches(" = {};").to_string());
                }
            }
            if let Some(decl) = t.strip_prefix("threadgroup ") {
                if let Some(v) = decl.rsplit(' ').next() {
                    workgroup_vars.push(v.trim_end_matches(',').to_string());
                }
            }
        }
        out.push(GraphEntry {
            src: tag.to_string(),
            name: wgsl_name(&name),
            workgroup_vars,
            thread_arrays,
        });
        i = start + name_end;
    }
    out
}

pub fn all_graph_sources() -> Vec<(&'static str, String)> {
    let mut v = Vec::new();
    v.extend(nv_models::gemma4_wgpu::nozi_audit_sources());
    v.extend(nv_models::gemma4_moe_wgpu::nozi_audit_sources());
    v.extend(nv_models::gpt_oss_wgpu::nozi_audit_sources());
    v.extend(nv_models::qwen3_5_moe_wgpu::nozi_audit_sources());
    v.extend(nv_models::qwen3_5_dense_wgpu::nozi_audit_sources());
    v
}

pub fn all_graph_entries() -> Vec<GraphEntry> {
    let mut out = Vec::new();
    for (tag, src) in all_graph_sources() {
        out.extend(entries_of(tag, &to_msl(tag, &src)));
    }
    out
}

pub fn audited_graph_names() -> Vec<&'static str> {
    dispatch::nozi_audited_entries()
        .iter()
        .copied()
        .filter(|e| {
            e.starts_with("g4m_")
                || e.starts_with("g4w_")
                || e.starts_with("gow_")
                || e.starts_with("q3w_")
        })
        .collect()
}

fn word_bounded(s: &str, at: usize, len: usize) -> bool {
    let pre = s[..at].chars().next_back();
    let post = s[at + len..].chars().next();
    let free = |c: Option<char>| !matches!(c, Some(c) if c.is_alphanumeric() || c == '_');
    free(pre) && free(post)
}

fn close_of(s: &str, open: usize, o: char, c: char) -> usize {
    let mut d = 0i32;
    for (k, ch) in s[open..].char_indices() {
        if ch == o {
            d += 1;
        } else if ch == c {
            d -= 1;
            if d == 0 {
                return open + k;
            }
        }
    }
    s.len()
}

pub fn eval_u32(expr: &str, consts: &BTreeMap<String, u32>) -> Option<u32> {
    let e = expr.trim();
    if e.is_empty() || e.contains('(') {
        return None;
    }
    for op in ['+', '-'] {
        if let Some(p) = e.rfind(op) {
            if p > 0 {
                let a = eval_u32(&e[..p], consts)?;
                let b = eval_u32(&e[p + 1..], consts)?;
                return if op == '+' {
                    a.checked_add(b)
                } else {
                    a.checked_sub(b)
                };
            }
        }
    }
    for op in ['*', '/'] {
        if let Some(p) = e.rfind(op) {
            let a = eval_u32(&e[..p], consts)?;
            let b = eval_u32(&e[p + 1..], consts)?;
            return if op == '*' {
                a.checked_mul(b)
            } else {
                a.checked_div(b)
            };
        }
    }
    e.strip_suffix('u')
        .unwrap_or(e)
        .parse::<u32>()
        .ok()
        .or_else(|| consts.get(e).copied())
}

pub fn wgsl_consts(src: &str) -> BTreeMap<String, u32> {
    let mut out = BTreeMap::new();
    for line in src.lines() {
        let Some(rest) = line.trim().strip_prefix("const ") else {
            continue;
        };
        let Some((decl, val)) = rest.split_once('=') else {
            continue;
        };
        let mut parts = decl.split(':');
        let name = parts.next().unwrap_or("").trim();
        if parts.next().map(str::trim) != Some("u32") {
            continue;
        }
        if let Some(v) = eval_u32(val.trim().trim_end_matches(';'), &out) {
            out.insert(name.to_string(), v);
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct WgDecl {
    pub name: String,
    pub elems: Option<u32>,
    pub len_tok: Option<String>,
}

pub fn wg_decls(src: &str, consts: &BTreeMap<String, u32>) -> Vec<WgDecl> {
    let mut out = Vec::new();
    for line in src.lines() {
        let Some(rest) = line.trim().strip_prefix("var<workgroup>") else {
            continue;
        };
        let Some((name, ty)) = rest.trim().trim_end_matches(';').split_once(':') else {
            continue;
        };
        let (name, ty) = (name.trim().to_string(), ty.trim());
        match ty.strip_prefix("array<").and_then(|s| s.strip_suffix('>')) {
            Some(inner) => {
                let (_, len) = inner.rsplit_once(',').unwrap_or_else(|| {
                    panic!("workgroup array {name} has no length: {ty}");
                });
                let len = len.trim();
                out.push(WgDecl {
                    name,
                    elems: eval_u32(len, consts),
                    len_tok: Some(len.to_string()),
                });
            }
            None => out.push(WgDecl {
                name,
                elems: None,
                len_tok: None,
            }),
        }
    }
    out
}

struct RawFn {
    name: String,
    params: String,
    body: String,
    wg: Option<String>,
}

fn preceding_wg(src: &str, at: usize) -> Option<String> {
    let head = &src[..at];
    let p = head.rfind("@workgroup_size(")?;
    if head[p..].contains('}') || head[p..].contains("fn ") {
        return None;
    }
    let rest = &head[p + "@workgroup_size(".len()..];
    Some(rest[..rest.find(')')?].trim().to_string())
}

fn scan_fns(src: &str) -> Vec<RawFn> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while let Some(at) = src[i..].find("fn ") {
        let p = i + at;
        i = p + 3;
        if !word_bounded(src, p, 2) {
            continue;
        }
        let rest = &src[p + 3..];
        let Some(po) = rest.find('(') else { continue };
        let name = rest[..po].trim().to_string();
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        let po = p + 3 + po;
        let pc = close_of(src, po, '(', ')');
        let Some(bo) = src[pc..].find('{').map(|k| pc + k) else {
            continue;
        };
        let bc = close_of(src, bo, '{', '}');
        out.push(RawFn {
            name,
            params: src[po + 1..pc].to_string(),
            body: src[bo + 1..bc].to_string(),
            wg: preceding_wg(src, p),
        });
        i = bc;
    }
    out
}

fn param_names(params: &str) -> Vec<String> {
    split_top(params)
        .into_iter()
        .filter_map(|p| {
            let p = p.trim();
            let p = p.rsplit_once(')').map(|(_, r)| r).unwrap_or(p);
            p.split_once(':')
                .map(|(n, _)| n.trim().to_string())
                .filter(|n| !n.is_empty())
        })
        .collect()
}

fn split_top(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let (mut d, mut start) = (0i32, 0usize);
    for (k, c) in s.char_indices() {
        match c {
            '(' | '<' => d += 1,
            ')' | '>' => d -= 1,
            ',' if d == 0 => {
                out.push(s[start..k].to_string());
                start = k + 1;
            }
            _ => {}
        }
    }
    if !s[start..].trim().is_empty() {
        out.push(s[start..].to_string());
    }
    out
}

fn substitute(body: &str, params: &[String], args: &[String]) -> String {
    let mut out = body.to_string();
    for (p, a) in params.iter().zip(args.iter()) {
        let mut next = String::with_capacity(out.len());
        let mut i = 0usize;
        while let Some(at) = out[i..].find(p.as_str()) {
            let at = i + at;
            next.push_str(&out[i..at]);
            if word_bounded(&out, at, p.len()) {
                next.push_str(a.trim());
            } else {
                next.push_str(&out[at..at + p.len()]);
            }
            i = at + p.len();
        }
        next.push_str(&out[i..]);
        out = next;
    }
    out.replace("return;", "")
        .replace("return ", "let pz_ret = ")
}

const INLINE_PASSES: usize = 256;

fn inline_calls(body: &str, helpers: &BTreeMap<String, (Vec<String>, String)>) -> String {
    let mut cur = body.to_string();
    for _ in 0..INLINE_PASSES {
        let Some((at, name)) = helpers
            .keys()
            .filter_map(|n| {
                let mut from = 0usize;
                loop {
                    let at = from + cur[from..].find(&format!("{n}("))?;
                    if word_bounded(&cur, at, n.len()) {
                        return Some((at, n.clone()));
                    }
                    from = at + n.len();
                }
            })
            .min_by_key(|(at, _)| *at)
        else {
            return cur;
        };
        let (params, hbody) = &helpers[&name];
        let po = at + name.len();
        let pc = close_of(&cur, po, '(', ')');
        let args = split_top(&cur[po + 1..pc]);
        let sub = substitute(hbody, params, &args);
        let after = cur[pc + 1..].trim_start();
        let stmt_call = after.starts_with(';')
            && cur[..at]
                .trim_end()
                .chars()
                .next_back()
                .is_none_or(|c| matches!(c, ';' | '{' | '}'));
        cur = if stmt_call {
            let end = pc + 1 + cur[pc + 1..].find(';').unwrap_or(0) + 1;
            format!("{}\n{sub}\n{}", &cur[..at], &cur[end..])
        } else {
            let start = cur[..at]
                .rfind([';', '{', '}'])
                .map(|k| k + 1)
                .unwrap_or(0);
            format!(
                "{}\n{sub}\n{}0{}",
                &cur[..start],
                &cur[start..at],
                &cur[pc + 1..]
            )
        };
    }
    panic!("helper inlining did not settle in {INLINE_PASSES} passes");
}

#[derive(Clone, Debug)]
pub struct EntrySrc {
    pub name: String,
    pub wg: u32,
    pub lanes: BTreeSet<String>,
    pub sgids: BTreeSet<String>,
    pub sglanes: BTreeSet<String>,
    pub body: String,
}

fn builtin_idents(params: &str, which: &str) -> Vec<String> {
    let tag = format!("@builtin({which})");
    split_top(params)
        .into_iter()
        .filter(|p| p.contains(&tag))
        .filter_map(|p| {
            let after = p.split_once(&tag)?.1;
            after.split_once(':').map(|(n, _)| n.trim().to_string())
        })
        .collect()
}

fn lane_idents(params: &str, body: &str) -> BTreeSet<String> {
    let mut out: BTreeSet<String> = builtin_idents(params, "local_invocation_index")
        .into_iter()
        .collect();
    for v in builtin_idents(params, "local_invocation_id") {
        out.insert(format!("{v}.x"));
    }
    for _ in 0..4 {
        let before = out.len();
        for line in body.lines() {
            let Some(rest) = line.trim().strip_prefix("let ") else {
                continue;
            };
            let Some((n, v)) = rest.trim_end_matches(';').split_once('=') else {
                continue;
            };
            if out.contains(v.trim()) {
                out.insert(n.split(':').next().unwrap_or("").trim().to_string());
            }
        }
        if out.len() == before {
            break;
        }
    }
    out
}

pub fn entries_src(src: &str, consts: &BTreeMap<String, u32>) -> Vec<EntrySrc> {
    let raw = scan_fns(src);
    let names: Vec<String> = wg_decls(src, consts).into_iter().map(|d| d.name).collect();
    let mut touch: BTreeSet<String> = BTreeSet::new();
    for _ in 0..raw.len().max(1) {
        let before = touch.len();
        for f in raw.iter().filter(|f| f.wg.is_none()) {
            let hit = names
                .iter()
                .chain(touch.iter())
                .any(|n| f.body.match_indices(n.as_str()).any(|(at, _)| word_bounded(&f.body, at, n.len())));
            if hit {
                touch.insert(f.name.clone());
            }
        }
        if touch.len() == before {
            break;
        }
    }
    let helpers: BTreeMap<String, (Vec<String>, String)> = raw
        .iter()
        .filter(|f| f.wg.is_none() && touch.contains(&f.name))
        .map(|f| (f.name.clone(), (param_names(&f.params), f.body.clone())))
        .collect();
    raw.iter()
        .filter_map(|f| {
            let wg = f.wg.as_ref()?;
            let wg = eval_u32(wg, consts)
                .unwrap_or_else(|| panic!("{}: unresolved @workgroup_size({wg})", f.name));
            let body = inline_calls(&f.body, &helpers);
            Some(EntrySrc {
                name: f.name.clone(),
                wg,
                lanes: lane_idents(&f.params, &body),
                sgids: builtin_idents(&f.params, "subgroup_id")
                    .into_iter()
                    .collect(),
                sglanes: builtin_idents(&f.params, "subgroup_invocation_id")
                    .into_iter()
                    .collect(),
                body,
            })
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct Access {
    pub idx: Option<String>,
    pub write: bool,
    pub guards: Vec<String>,
}

pub fn accesses(body: &str, var: &str) -> Vec<Access> {
    let mut out = Vec::new();
    let mut stack: Vec<String> = Vec::new();
    let mut boundary = 0usize;
    let mut paren = 0i32;
    let mut i = 0usize;
    while i < body.len() {
        let c = body[i..].chars().next().unwrap();
        match c {
            '(' => {
                paren += 1;
                i += 1;
                continue;
            }
            ')' => {
                paren -= 1;
                i += 1;
                continue;
            }
            '{' => {
                stack.push(body[boundary..i].trim().to_string());
                boundary = i + 1;
                i += 1;
                continue;
            }
            '}' => {
                stack.pop();
                boundary = i + 1;
                i += 1;
                continue;
            }
            ';' if paren == 0 => {
                boundary = i + 1;
                i += 1;
                continue;
            }
            _ => {}
        }
        if body[i..].starts_with(var) && word_bounded(body, i, var.len()) {
            let after = &body[i + var.len()..];
            let ws = after.len() - after.trim_start().len();
            let rest = after.trim_start();
            let (idx, tail_at) = if rest.starts_with('[') {
                let ob = i + var.len() + ws;
                let cb = close_of(body, ob, '[', ']');
                (Some(body[ob + 1..cb].trim().to_string()), cb + 1)
            } else {
                (None, i + var.len())
            };
            let tail = body[tail_at..].trim_start();
            let write = tail.starts_with('=') && !tail.starts_with("==");
            out.push(Access {
                idx,
                write,
                guards: stack.clone(),
            });
            i = tail_at;
            continue;
        }
        i += c.len_utf8();
    }
    out
}

#[derive(Clone, Debug, PartialEq)]
pub enum Extent {
    Lane,
    Subgroup,
    Scalar,
    Prefix(String),
    Unproven(String),
}

fn guard_bound(guard: &str) -> Option<String> {
    let clauses: Vec<&str> = guard.split(';').collect();
    let pick = if guard.trim_start().starts_with("for") && clauses.len() >= 2 {
        clauses[1]
    } else {
        clauses[0]
    };
    let (_, r) = pick.split_once('<')?;
    let r = r.trim_start_matches('=');
    let end = r
        .find([')', ';'])
        .unwrap_or(r.len())
        .min(r.find("&&").unwrap_or(r.len()))
        .min(r.find("||").unwrap_or(r.len()));
    let r: String = r[..end].split_whitespace().collect::<Vec<_>>().join(" ");
    (!r.is_empty()).then_some(r)
}

pub fn extent_of(e: &EntrySrc, d: &WgDecl) -> Option<Extent> {
    let acc = accesses(&e.body, &d.name);
    if acc.is_empty() {
        return None;
    }
    let writes: Vec<&Access> = acc.iter().filter(|a| a.write).collect();
    if writes.is_empty() {
        return Some(Extent::Unproven(format!("{} is never written", d.name)));
    }
    let Some(len) = d.elems else {
        let first_w = acc.iter().position(|a| a.write).unwrap_or(usize::MAX);
        let first_r = acc.iter().position(|a| !a.write).unwrap_or(usize::MAX);
        return Some(if first_w < first_r {
            Extent::Scalar
        } else {
            Extent::Unproven(format!("{} is read before any thread writes it", d.name))
        });
    };
    let lane_cover = writes.iter().any(|a| {
        a.guards.iter().all(|g| g.is_empty())
            && a.idx.as_deref().is_some_and(|i| e.lanes.contains(i))
    });
    if lane_cover && len <= e.wg {
        return Some(Extent::Lane);
    }
    let sg_cover = writes.iter().any(|a| {
        a.idx.as_deref().is_some_and(|i| e.sgids.contains(i))
            && a.guards.iter().all(|g| {
                g.is_empty()
                    || e.sglanes
                        .iter()
                        .any(|l| g.contains(&format!("{l} == 0u")) && !g.contains("&&"))
            })
    });
    if sg_cover && len.saturating_mul(SUBGROUP_LANES) <= e.wg {
        return Some(Extent::Subgroup);
    }
    let bounds: BTreeSet<String> = writes
        .iter()
        .filter_map(|a| a.guards.iter().rev().find_map(|g| guard_bound(g)))
        .collect();
    if bounds.len() == 1 && writes.len() == writes.iter().filter(|a| !a.guards.is_empty()).count() {
        return Some(Extent::Prefix(bounds.into_iter().next().unwrap()));
    }
    Some(Extent::Unproven(format!(
        "{}: len {len} wg {} lane_cover {lane_cover} sg_cover {sg_cover} write bounds {bounds:?}",
        d.name, e.wg
    )))
}

pub fn unbounded_reads(e: &EntrySrc, d: &WgDecl, bound: &str) -> Vec<String> {
    let writes: BTreeSet<String> = accesses(&e.body, &d.name)
        .into_iter()
        .filter(|a| a.write)
        .filter_map(|a| a.idx)
        .collect();
    let mut out: BTreeSet<String> = BTreeSet::new();
    for a in accesses(&e.body, &d.name).into_iter().filter(|a| !a.write) {
        let Some(idx) = a.idx else { continue };
        let loop_var = e.body.contains(&format!("for (var {idx} ="));
        if !loop_var && writes.contains(&idx) {
            continue;
        }
        let ident = !idx.is_empty()
            && idx
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '.');
        let covered = ident
            && a.guards.iter().any(|g| {
                let g: String = g.split_whitespace().collect::<Vec<_>>().join(" ");
                g.contains(&format!("{idx} < {bound}")) || g.contains(&format!("{idx} <= {bound}"))
            });
        if !covered {
            out.insert(idx);
        }
    }
    out.into_iter().collect()
}
