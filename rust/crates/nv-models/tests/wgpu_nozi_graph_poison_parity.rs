#![cfg(feature = "wgpu")]

use std::collections::BTreeMap;

use nv_kernels::wgpu_backend::{dispatch, WgpuContext};

mod nozi_graph_common;
use nozi_graph_common as census;

const BUF_WORDS: usize = 1 << 16;

const CMP_WORDS: usize = 1 << 12;

const PARAM_FILLS: [u32; 3] = [8, 3, 64];

const RETRIES: usize = 6;

fn ctx(what: &str) -> &'static WgpuContext {
    let ctx = WgpuContext::shared()
        .unwrap_or_else(|e| panic!("{what}: no wgpu adapter, this proof cannot pass: {e}"));
    ctx
}

fn param_value(field: &str, count: u32, legal: bool) -> u32 {
    if legal && (field == "rot_half" || field == "rot_dim") {
        return (count / 2).max(1);
    }
    count
}

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 33) as u32
}

#[derive(Clone, Debug)]
struct Binding {
    slot: u32,
    kind: BindKind,
    ty: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum BindKind {
    StorageRead,
    StorageRw,
    Uniform,
}

#[derive(Clone, Debug)]
struct WgVar {
    name: String,

    len: Option<String>,
    elem: String,
}

fn parse_workgroup_vars(src: &str) -> Vec<WgVar> {
    let mut out = Vec::new();
    for line in src.lines() {
        let t = line.trim();
        let Some(rest) = t.strip_prefix("var<workgroup>") else {
            continue;
        };
        let Some((name, ty)) = rest.trim().trim_end_matches(';').split_once(':') else {
            continue;
        };
        let name = name.trim().to_string();
        let ty = ty.trim();
        if let Some(inner) = ty.strip_prefix("array<").and_then(|s| s.strip_suffix('>')) {
            let (elem, len) = inner
                .rsplit_once(',')
                .unwrap_or_else(|| panic!("workgroup array {name} has no length: {ty}"));
            out.push(WgVar {
                name,
                len: Some(len.trim().to_string()),
                elem: elem.trim().to_string(),
            });
        } else {
            out.push(WgVar {
                name,
                len: None,
                elem: ty.to_string(),
            });
        }
    }
    out
}

fn parse_bindings(src: &str) -> Vec<Binding> {
    let mut out = Vec::new();
    for line in src.lines() {
        let t = line.trim();
        if !t.starts_with("@group(0)") {
            assert!(
                !t.starts_with("@group("),
                "this harness binds group 0 only, and a source grew another group: {t}"
            );
            continue;
        }
        let slot: u32 = t
            .split("@binding(")
            .nth(1)
            .and_then(|s| s.split(')').next())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or_else(|| panic!("cannot read a binding slot out of: {t}"));
        let kind = if t.contains("var<uniform>") {
            BindKind::Uniform
        } else if t.contains("read_write") {
            BindKind::StorageRw
        } else {
            BindKind::StorageRead
        };
        let ty = t
            .rsplit_once(':')
            .map(|(_, r)| r.trim().trim_end_matches(';').trim().to_string())
            .unwrap_or_default();
        out.push(Binding { slot, kind, ty });
    }
    out.sort_by_key(|b| b.slot);
    out.dedup_by_key(|b| b.slot);
    out
}

fn parse_structs(src: &str) -> BTreeMap<String, Option<Vec<(String, String)>>> {
    let mut out = BTreeMap::new();
    let mut i = 0usize;
    while let Some(at) = src[i..].find("struct ") {
        let start = i + at + "struct ".len();
        let Some(brace) = src[start..].find('{') else {
            break;
        };
        let name = src[start..start + brace].trim().to_string();
        let body_start = start + brace + 1;
        let Some(close) = src[body_start..].find('}') else {
            break;
        };
        let body = &src[body_start..body_start + close];
        let mut fields = Vec::new();
        let mut ok = true;
        for f in body.split(',') {
            let f = f.trim();
            if f.is_empty() {
                continue;
            }
            let Some((fname, ty)) = f.split_once(':') else {
                continue;
            };
            let ty = ty.trim();
            match ty {
                "u32" | "i32" | "f32" => fields.push((fname.trim().to_string(), ty.to_string())),
                _ => ok = false,
            }
        }
        out.insert(name, if ok { Some(fields) } else { None });
        i = body_start + close;
    }
    out
}

struct EntryText {
    wg_size: String,
    params: String,
    body: String,
}

fn extract_entry(src: &str, entry: &str) -> Option<EntryText> {
    let needle = format!("fn {entry}(");
    let mut from = 0usize;
    loop {
        let at = from + src[from..].find(&needle)?;

        let head = &src[..at];
        let ws = head.rfind("@workgroup_size(").map(|p| {
            let rest = &head[p + "@workgroup_size(".len()..];
            rest[..rest.find(')').unwrap_or(0)].trim().to_string()
        });
        let Some(wg_size) = ws else {
            from = at + needle.len();
            continue;
        };

        let between = &head[head.rfind("@workgroup_size(").unwrap()..];
        if between.matches("fn ").count() > 0 {
            from = at + needle.len();
            continue;
        }
        let popen = at + needle.len() - 1;
        let mut depth = 0i32;
        let mut pclose = popen;
        for (k, c) in src[popen..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        pclose = popen + k;
                        break;
                    }
                }
                _ => {}
            }
        }
        let params = src[popen + 1..pclose].trim().to_string();
        let bopen = pclose + src[pclose..].find('{')?;
        let mut depth = 0i32;
        let mut bclose = bopen;
        for (k, c) in src[bopen..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        bclose = bopen + k;
                        break;
                    }
                }
                _ => {}
            }
        }
        return Some(EntryText {
            wg_size,
            params,
            body: src[bopen + 1..bclose].to_string(),
        });
    }
}

fn poison_stmt(v: &WgVar, idx: &str) -> String {
    match (v.elem.as_str(), &v.len) {
        ("f32", Some(len)) => format!(
            "if ({idx} < u32({len})) {{ {}[{idx}] = bitcast<f32>(0x7fc0deadu | ({idx} << 4u)); }}",
            v.name
        ),
        ("u32", Some(len)) => format!(
            "if ({idx} < u32({len})) {{ {}[{idx}] = 0x5eed0000u | {idx}; }}",
            v.name
        ),
        ("i32", Some(len)) => format!(
            "if ({idx} < u32({len})) {{ {}[{idx}] = i32(0x5eed0000u | {idx}); }}",
            v.name
        ),
        ("f32", None) => format!(
            "if ({idx} == 0u) {{ {} = bitcast<f32>(0x7fc0deadu); }}",
            v.name
        ),
        ("u32", None) => format!("if ({idx} == 0u) {{ {} = 0x5eed0001u; }}", v.name),
        ("i32", None) => format!("if ({idx} == 0u) {{ {} = i32(0x5eed0001u); }}", v.name),
        _ => panic!(
            "the poison injector has no pattern for workgroup var {} of type {}{}; it must \
             refuse rather than leave the array un-poisoned and call the entry proved",
            v.name,
            v.elem,
            v.len.as_deref().unwrap_or("")
        ),
    }
}

fn poisoned_twin(e: &EntryText, name: &str, vars: &[WgVar]) -> String {
    assert!(
        !e.params.contains("local_invocation_index"),
        "{name} already takes local_invocation_index; the injector would declare it twice"
    );
    let bound = vars
        .iter()
        .filter_map(|v| v.len.as_ref().map(|l| format!("u32({l})")))
        .fold(String::from("1u"), |acc, l| format!("max({acc}, {l})"));
    let fills: Vec<String> = vars.iter().map(|v| poison_stmt(v, "pz_i")).collect();
    let sep = if e.params.trim().is_empty() {
        ""
    } else {
        ",\n    "
    };
    format!(
        "@compute @workgroup_size({wg})\nfn {name}(\n    {params}{sep}@builtin(local_invocation_index) pz_lidx: u32\n) {{\n\
         var pz_i = pz_lidx;\n    loop {{\n        if (pz_i >= {bound}) {{ break; }}\n        {fills}\n        pz_i = pz_i + u32({wg});\n    }}\n    workgroupBarrier();\n{body}}}\n",
        wg = e.wg_size,
        params = e.params,
        sep = sep,
        bound = bound,
        fills = fills.join("\n        "),
        body = e.body,
    )
}

fn tripwire_plain(
    name: &str,
    wg: &str,
    v: &WgVar,
    out: &Binding,
    out_name: &str,
) -> Option<String> {
    let len = v.len.as_ref()?;
    let read = match v.elem.as_str() {
        "f32" => format!("bitcast<u32>({}[pz_tw_lidx])", v.name),
        "u32" => format!("{}[pz_tw_lidx]", v.name),
        "i32" => format!("u32({}[pz_tw_lidx])", v.name),
        _ => return None,
    };

    let store = match out.ty.as_str() {
        "array<f32>" => "bitcast<f32>(v)",
        "array<i32>" => "i32(v)",
        "array<u32>" => "v",
        _ => return None,
    };
    let one = match v.elem.as_str() {
        "f32" => "1.0",
        "i32" => "1",
        _ => "1u",
    };
    Some(format!(
        "@compute @workgroup_size({wg})\nfn {name}(@builtin(local_invocation_id) pz_tid: vec3<u32>) {{\n\
         let pz_tw_lidx = pz_tid.x;\n    if (pz_tw_lidx == 0u) {{ {arr}[0] = {one}; }}\n    workgroupBarrier();\n\
         if (pz_tw_lidx < u32({len})) {{ let v = {read}; {out_name}[pz_tw_lidx] = {store}; }}\n}}\n",
        wg = wg,
        len = len,
        arr = v.name,
        one = one,
        read = read,
        out_name = out_name,
        store = store,
    ))
}

fn layout(ctx: &WgpuContext, binds: &[Binding]) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = binds
        .iter()
        .map(|b| wgpu::BindGroupLayoutEntry {
            binding: b.slot,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: match b.kind {
                    BindKind::Uniform => wgpu::BufferBindingType::Uniform,
                    BindKind::StorageRead => wgpu::BufferBindingType::Storage { read_only: true },
                    BindKind::StorageRw => wgpu::BufferBindingType::Storage { read_only: false },
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    ctx.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nozi-graph-poison"),
            entries: &entries,
        })
}

fn build(
    ctx: &WgpuContext,
    src: &str,
    entry: &str,
    zero_init: bool,
    pl: &wgpu::PipelineLayout,
) -> Result<wgpu::ComputePipeline, String> {
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
    let p = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(pl),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[],
                zero_initialize_workgroup_memory: zero_init,
            },
            cache: None,
        });
    match pollster::block_on(scope.pop()) {
        Some(e) => Err(format!("{e}")),
        None => Ok(p),
    }
}

struct Fixture {
    bufs: Vec<wgpu::Buffer>,
    bind: wgpu::BindGroup,
}

fn fixture(
    ctx: &WgpuContext,
    binds: &[Binding],
    structs: &BTreeMap<String, Option<Vec<(String, String)>>>,
    bgl: &wgpu::BindGroupLayout,
    seed0: u64,
    count: u32,
    legal: bool,
) -> Option<Fixture> {
    let mut seed = seed0;
    let mut bufs = Vec::new();
    for b in binds {
        let buf = match b.kind {
            BindKind::Uniform => {
                let fields = structs.get(&b.ty)?.as_ref()?;
                let mut w: Vec<u32> = Vec::with_capacity(fields.len().max(4));
                for (fname, fty) in fields {
                    w.push(match fty.as_str() {
                        "f32" => 1.0f32.to_bits(),
                        _ => param_value(fname, count, legal),
                    });
                }
                while w.len() < 64 {
                    w.push(count);
                }
                let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pz-uniform"),
                    size: (w.len() * 4) as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                ctx.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&w));
                buf
            }
            _ => {
                let data: Vec<u32> = (0..BUF_WORDS)
                    .map(|_| lcg(&mut seed) & 0x3f7f_3f7f)
                    .collect();
                dispatch::storage_from_slice(ctx, "pz-storage", &data)
            }
        };
        bufs.push(buf);
    }
    let entries: Vec<wgpu::BindGroupEntry> = binds
        .iter()
        .zip(bufs.iter())
        .map(|(b, buf)| wgpu::BindGroupEntry {
            binding: b.slot,
            resource: buf.as_entire_binding(),
        })
        .collect();
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pz-bind"),
        layout: bgl,
        entries: &entries,
    });
    Some(Fixture { bufs, bind })
}

fn run(
    ctx: &WgpuContext,
    pipeline: &wgpu::ComputePipeline,
    fx: &Fixture,
    binds: &[Binding],
) -> Vec<u32> {
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &fx.bind, &[]);

        pass.dispatch_workgroups(1, 1, 1);
    }
    ctx.queue.submit([enc.finish()]);
    ctx.poll_blocking().expect("poll");
    let mut out = Vec::new();
    for (b, buf) in binds.iter().zip(fx.bufs.iter()) {
        if b.kind == BindKind::StorageRw {
            out.extend(dispatch::read_back::<u32>(ctx, buf, CMP_WORDS).expect("readback"));
        }
    }
    out
}

fn moved(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

fn run_nonempty(
    ctx: &WgpuContext,
    pipeline: &wgpu::ComputePipeline,
    mk: &dyn Fn() -> Fixture,
    binds: &[Binding],
) -> Option<Vec<u32>> {
    for _ in 0..RETRIES {
        let v = run(ctx, pipeline, &mk(), binds);
        if v.iter().any(|w| *w != 0) {
            return Some(v);
        }
    }
    None
}

#[derive(Default)]
struct Tally {
    discharged: Vec<String>,
    vacuous: Vec<String>,
    unbuildable: Vec<String>,
    unstable: Vec<String>,
}

#[test]
fn every_audited_graph_entry_is_write_before_read_under_an_in_dispatch_poison() {
    let ctx = ctx("nozi-graph-poison");
    eprintln!("nozi-graph-poison: {}", ctx.summary());
    let audited = census::audited_graph_names();
    assert_eq!(
        audited.len(),
        41,
        "the graph share of NOZI_AUDITED_ENTRIES moved"
    );

    let mut tally = Tally::default();
    let mut tripwires = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for (tag, src) in census::all_graph_sources() {
        let here: Vec<&str> = audited
            .iter()
            .copied()
            .filter(|e| extract_entry(&src, e).is_some())
            .collect();
        if here.is_empty() {
            continue;
        }

        let reach: BTreeMap<String, Vec<String>> =
            census::entries_of(tag, &census::to_msl(tag, &src))
                .into_iter()
                .map(|e| (e.name, e.workgroup_vars))
                .collect();
        let binds = parse_bindings(&src);
        let structs = parse_structs(&src);
        let all_wg = parse_workgroup_vars(&src);
        let bgl = layout(ctx, &binds);
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(tag),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let Some(rw) = binds
            .iter()
            .find(|b| b.kind == BindKind::StorageRw)
            .cloned()
        else {
            for e in &here {
                tally.unbuildable.push(format!(
                    "{tag}::{e} (source has no read_write binding to observe)"
                ));
            }
            continue;
        };
        let rw_name = src
            .lines()
            .map(str::trim)
            .find(|l| l.starts_with("@group(0)") && l.contains(&format!("@binding({})", rw.slot)))
            .and_then(|l| l.split("> ").nth(1))
            .and_then(|l| l.split(':').next())
            .map(|s| s.trim().to_string())
            .expect("read_write binding name");

        let mut tw_ok = false;
        let mut tw_why = String::from("no array-shaped workgroup var, or no observable output");
        for v in all_wg.iter().filter(|v| v.len.is_some()) {
            let Some(tw) = tripwire_plain("pz_tw", "256", v, &rw, &rw_name) else {
                continue;
            };
            let tw_src = format!("{src}\n{tw}");
            let Some(text) = extract_entry(&tw_src, "pz_tw") else {
                continue;
            };
            let twin = poisoned_twin(&text, "pz_pz_tw", std::slice::from_ref(v));
            let full = format!("{tw_src}\n{twin}");

            let (a, b) = match (
                build(ctx, &full, "pz_tw", true, &pl),
                build(ctx, &full, "pz_pz_tw", false, &pl),
            ) {
                (Ok(a), Ok(b)) => (a, b),
                (x, y) => {
                    tw_why = format!("build: {:?} / {:?}", x.err(), y.err());
                    continue;
                }
            };
            let mk = || {
                fixture(
                    ctx,
                    &binds,
                    &structs,
                    &bgl,
                    0x51ee_d001,
                    PARAM_FILLS[0],
                    true,
                )
                .expect("fixture")
            };
            let (Some(zi), Some(pz)) = (
                run_nonempty(ctx, &a, &mk, &binds),
                run_nonempty(ctx, &b, &mk, &binds),
            ) else {
                tw_why = format!("{} produced an all-zero readback {RETRIES} times", v.name);
                continue;
            };
            let m = moved(&zi, &pz);
            if m > 0 {
                tw_ok = true;
                tripwires += 1;
                eprintln!(
                    "  tripwire {tag} on {}: {m} words moved under poison",
                    v.name
                );
                break;
            }
            tw_why = format!(
                "{} moved 0 words (zi[0..6]={:08x?} pz[0..6]={:08x?}, out={rw_name} slot {})",
                v.name,
                &zi[..6.min(zi.len())],
                &pz[..6.min(pz.len())],
                rw.slot
            );
        }
        assert!(
            tw_ok,
            "{tag}: the tripwire read identical words with and without zero-init ({tw_why}), so \
             the poison never reached workgroup memory and every parity result on this source \
             would be vacuous"
        );

        for e in here {
            let Some(text) = extract_entry(&src, e) else {
                continue;
            };

            let names = reach.get(e).cloned().unwrap_or_default();
            let used: Vec<WgVar> = all_wg
                .iter()
                .filter(|v| names.iter().any(|n| n.trim_end_matches('_') == v.name))
                .cloned()
                .collect();
            assert!(
                !used.is_empty(),
                "{tag}::{e}: the census says it touches {names:?} and the poison injector \
                 resolved none of them to a declaration -- it would then poison nothing and \
                 report a pass"
            );
            let twin_name = format!("pz_{e}");
            let twin = poisoned_twin(&text, &twin_name, &used);
            let twin_src = format!("{src}\n{twin}");
            let zi_pl = match build(ctx, &src, e, true, &pl) {
                Ok(p) => p,
                Err(msg) => {
                    tally
                        .unbuildable
                        .push(format!("{tag}::{e} (zi build: {msg})"));
                    continue;
                }
            };
            let pz_pl = match build(ctx, &twin_src, &twin_name, false, &pl) {
                Ok(p) => p,
                Err(msg) => {
                    tally
                        .unbuildable
                        .push(format!("{tag}::{e} (poisoned twin build: {msg})"));
                    continue;
                }
            };

            let mut moved_at: Option<String> = None;
            let mut judged: Vec<u32> = Vec::new();
            let mut unjudged: Vec<String> = Vec::new();
            for fill in PARAM_FILLS {
                let seed = 0x9e37_79b9u64 ^ u64::from(fill);
                let mk =
                    || fixture(ctx, &binds, &structs, &bgl, seed, fill, true).expect("fixture");
                let Some(a) = run_nonempty(ctx, &zi_pl, &mk, &binds) else {
                    unjudged.push(format!("fill={fill} wrote nothing in {RETRIES} tries"));
                    continue;
                };
                let b = run(ctx, &pz_pl, &mk(), &binds);
                let a2 = run(ctx, &zi_pl, &mk(), &binds);
                let drift = moved(&a, &a2);
                if drift > 0 {
                    unjudged.push(format!("fill={fill} A!=A' by {drift} words"));
                    continue;
                }
                let m = moved(&a, &b);
                if m > 0 {
                    let sample: Vec<String> = a
                        .iter()
                        .zip(b.iter())
                        .enumerate()
                        .filter(|(_, (x, y))| x != y)
                        .take(4)
                        .map(|(i, (x, y))| format!("[{i}] zi {x:08x} vs poisoned {y:08x}"))
                        .collect();
                    moved_at = Some(format!(
                        "{tag}::{e}: fill={fill}, null control clean (A==A'), {m} words moved \
                         under poison: {sample:?}"
                    ));
                    break;
                }
                judged.push(fill);
            }
            match moved_at {
                Some(f) => failures.push(f),
                None if judged.is_empty() => tally
                    .vacuous
                    .push(format!("{tag}::{e} (no shape was judgeable: {unjudged:?})")),
                None => {
                    tally
                        .discharged
                        .push(format!("{tag}::{e} @fills{judged:?}"));
                    if !unjudged.is_empty() {
                        tally.unstable.push(format!("{tag}::{e} {unjudged:?}"));
                    }
                }
            }
        }
    }

    eprintln!(
        "\nnozi-graph-poison: {} declaration sites DISCHARGED, {} vacuous, {} unbuildable, \
         {} unstable, {tripwires} live tripwires",
        tally.discharged.len(),
        tally.vacuous.len(),
        tally.unbuildable.len(),
        tally.unstable.len()
    );
    eprintln!("discharged: {:#?}", tally.discharged);
    eprintln!("vacuous (NOT discharged): {:#?}", tally.vacuous);
    eprintln!("unbuildable (NOT discharged): {:#?}", tally.unbuildable);
    eprintln!(
        "shapes dropped by the null control (site still discharged on the rest): {:#?}",
        tally.unstable
    );
    let names: std::collections::BTreeSet<&str> = tally
        .discharged
        .iter()
        .filter_map(|s| s.split("::").nth(1))
        .map(|s| s.split(" @").next().unwrap_or(s))
        .collect();
    eprintln!(
        "distinct audited NAMES with at least one discharged site: {}/41",
        names.len()
    );

    assert!(
        failures.is_empty(),
        "these entries are on NOZI_AUDITED_ENTRIES and read workgroup memory they had not \
         written -- they ship with zero-init disabled, so this is a live correctness bug, \
         not a test failure: {failures:#?}"
    );

    assert!(
        tally.vacuous.is_empty() && tally.unbuildable.is_empty(),
        "these sites were not judged at any shape: vacuous {:#?} unbuildable {:#?}",
        tally.vacuous,
        tally.unbuildable
    );
    assert_eq!(
        tally.discharged.len(),
        54,
        "the poison harness judged {} of the 54 declaration sites",
        tally.discharged.len()
    );
    assert_eq!(
        names.len(),
        41,
        "expected all 38 audited graph NAMES covered"
    );
}

#[test]
fn attn_norm_rope_reads_unwritten_workgroup_memory_when_rot_half_exceeds_half_the_head() {
    let ctx = ctx("nozi-graph-rope-bound");
    let mut checked = 0usize;
    for (tag, src) in census::all_graph_sources() {
        for entry in ["g4m_attn_norm_rope", "q3w_attn_norm_rope"] {
            let Some(text) = extract_entry(&src, entry) else {
                continue;
            };
            let binds = parse_bindings(&src);
            let structs = parse_structs(&src);
            let all_wg = parse_workgroup_vars(&src);
            let reach: BTreeMap<String, Vec<String>> =
                census::entries_of(tag, &census::to_msl(tag, &src))
                    .into_iter()
                    .map(|e| (e.name, e.workgroup_vars))
                    .collect();
            let names = reach.get(entry).cloned().unwrap_or_default();
            let used: Vec<WgVar> = all_wg
                .iter()
                .filter(|v| names.iter().any(|n| n.trim_end_matches('_') == v.name))
                .cloned()
                .collect();
            let bgl = layout(ctx, &binds);
            let pl = ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(tag),
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 0,
                });
            let twin_name = format!("pz_{entry}");
            let twin_src = format!("{src}\n{}", poisoned_twin(&text, &twin_name, &used));
            let zi = build(ctx, &src, entry, true, &pl).expect("zi build");
            let pz = build(ctx, &twin_src, &twin_name, false, &pl).expect("twin build");
            let fill = PARAM_FILLS[0];
            let seed = 0x9e37_79b9u64 ^ u64::from(fill);

            let mk = |legal| fixture(ctx, &binds, &structs, &bgl, seed, fill, legal).unwrap();
            let a = run(ctx, &zi, &mk(false), &binds);
            let a2 = run(ctx, &zi, &mk(false), &binds);
            assert_eq!(
                moved(&a, &a2),
                0,
                "{tag}::{entry}: the zero-init arm does not reproduce itself, so nothing here \
                 can be attributed to the poison"
            );
            let b = run(ctx, &pz, &mk(false), &binds);
            let m = moved(&a, &b);
            assert!(
                m > 0,
                "{tag}::{entry}: at rot_half == head_dim the rope mix reads buf[d + rot_half], \
                 which the entry never wrote, and the poison did NOT change the answer. Either \
                 the kernel grew a guard -- in which case delete this test and say so -- or the \
                 harness has stopped poisoning and every proof beside it is vacuous"
            );

            let c = run(ctx, &zi, &mk(true), &binds);
            let d = run(ctx, &pz, &mk(true), &binds);
            assert_eq!(
                moved(&c, &d),
                0,
                "{tag}::{entry}: at rot_half == head_dim/2 the entry must be write-before-read"
            );
            eprintln!(
                "  {tag}::{entry}: rot_half=head_dim moves {m} words, rot_half=head_dim/2 moves 0"
            );
            checked += 1;
        }
    }
    assert!(
        checked >= 3,
        "expected the rope-bound control to reach all three attn_norm_rope declaration sites, \
         reached {checked}"
    );
}
