#![cfg(feature = "wgpu")]

mod nozi_graph_common;
use nozi_graph_common::*;

const AUDITED_ENTRIES: usize = 68;

const AUDITED_GRAPH_ENTRIES: usize = 41;

const PREFIX_WRITE_SITES: &[(&str, &str, &str, &str, &[&str])] = &[
    (
        "g4w:head_prep",
        "g4w_head_prep",
        "hp_a",
        "words",
        &[
            "elem + 1u + half >> 1u",
            "elem + 1u >> 1u",
            "elem + half >> 1u",
            "elem >> 1u",
            "pair >> 1u",
        ],
    ),
    (
        "g4w:head_prep",
        "g4w_head_prep",
        "hp_b",
        "words",
        &["d >> 1u", "d0 + j >> 1u"],
    ),
    (
        "g4m:attn",
        "g4m_attn_norm_rope",
        "ga_buf",
        "hd",
        &["e0 + 1u + rh", "e0 + rh", "i"],
    ),
    ("g4m:attn", "g4m_attn_decode", "gd_qs", "hd", &[]),
    ("gow:attn", "gow_attn_decode", "gad_qs", "hd", &[]),
    ("q3m:delta", "q3w_delta_recurrent", "dr_kb", "dk", &[]),
    ("q3m:delta", "q3w_delta_recurrent", "dr_qb", "dk", &[]),
    (
        "q3m:attn",
        "q3w_attn_norm_rope",
        "ar_buf",
        "hd",
        &["e0 + 1u + rh", "e0 + rh", "i"],
    ),
    ("q3d:delta", "q3w_delta_recurrent", "dr_kb", "dk", &[]),
    ("q3d:delta", "q3w_delta_recurrent", "dr_qb", "dk", &[]),
    (
        "q3d:delta",
        "q3w_delta_head_fused",
        "fh_q",
        "dq_p.d_k",
        &[
            "j", "j + 1u", "j + 2u", "j + 3u", "j + 4u", "j + 5u", "j + 6u", "j + 7u",
        ],
    ),
    (
        "q3d:delta",
        "q3w_delta_head_fused",
        "fh_k",
        "dq_p.d_k",
        &[
            "i", "i + 1u", "i + 2u", "i + 3u", "i + 4u", "i + 5u", "i + 6u", "i + 7u", "j",
            "j + 1u", "j + 2u", "j + 3u", "j + 4u", "j + 5u", "j + 6u", "j + 7u",
        ],
    ),
    (
        "q3d:attn",
        "q3w_attn_norm_rope",
        "ar_buf",
        "hd",
        &["e0 + 1u + ar_p.rot_half", "e0 + ar_p.rot_half", "i"],
    ),
    (
        "q3d:attn",
        "q3w_attn_qk_norm_rope_qcast",
        "ar_buf",
        "hd",
        &["e0 + 1u + af_p.rot_half", "e0 + af_p.rot_half", "i"],
    ),
    ("q3d:attn", "q3w_attn_decode", "ad_qs", "hd", &[]),
];

const COVERING_WRITE_CLASSES: (usize, usize, usize, usize) = (59, 8, 15, 15);

const ROPE_HALVING_SITES: usize = 15;

const EXTENT_GATES: &[(&str, &str, &str, &str)] = &[
    ("g4m:attn", "gd_qs", "gemma4_moe_wgpu", "MAX_HEAD_DIM"),
    ("g4m:attn", "ga_buf", "gemma4_moe_wgpu", "MAX_HEAD_DIM"),
    ("gow:attn", "gad_qs", "gpt_oss_wgpu", "MAX_HEAD_DIM"),
    ("q3d:attn", "ad_qs", "qwen3_5_dense_wgpu", "MAX_HEAD_DIM"),
    ("q3d:attn", "ar_buf", "qwen3_5_dense_wgpu", "MAX_HEAD_DIM"),
    ("q3d:delta", "dr_kb", "qwen3_5_dense_wgpu", "MAX_LIN_HEAD_DIM"),
    ("q3d:delta", "dr_qb", "qwen3_5_dense_wgpu", "MAX_LIN_HEAD_DIM"),
    ("q3d:delta", "fh_q", "qwen3_5_dense_wgpu", "MAX_LIN_HEAD_DIM"),
    ("q3d:delta", "fh_k", "qwen3_5_dense_wgpu", "MAX_LIN_HEAD_DIM"),
    ("q3m:attn", "ar_buf", "qwen3_5_moe_wgpu", "MAX_HEAD_DIM"),
    ("q3m:delta", "dr_kb", "qwen3_5_moe_wgpu", "MAX_LIN_HEAD_DIM"),
    ("q3m:delta", "dr_qb", "qwen3_5_moe_wgpu", "MAX_LIN_HEAD_DIM"),
];

const BUILD_SIZED_PREFIX_SITES: &[(&str, &str)] =
    &[("g4w:head_prep", "hp_a"), ("g4w:head_prep", "hp_b")];

const MODEL_SOURCES: &[(&str, &str)] = &[
    ("gemma4_moe_wgpu", include_str!("../src/gemma4_moe_wgpu.rs")),
    ("gpt_oss_wgpu", include_str!("../src/gpt_oss_wgpu.rs")),
    (
        "qwen3_5_moe_wgpu",
        include_str!("../src/qwen3_5_moe_wgpu.rs"),
    ),
    (
        "qwen3_5_dense_wgpu",
        include_str!("../src/qwen3_5_dense_wgpu.rs"),
    ),
];

fn rust_usize_const(module: &str, name: &str) -> u32 {
    let (_, src) = MODEL_SOURCES
        .iter()
        .find(|(m, _)| *m == module)
        .unwrap_or_else(|| panic!("{module} is not on MODEL_SOURCES"));
    src.lines()
        .find_map(|l| {
            let rest = l.trim().strip_prefix("const ")?.strip_prefix(name)?;
            rest.strip_prefix(": usize =")?
                .trim()
                .trim_end_matches(';')
                .trim()
                .parse()
                .ok()
        })
        .unwrap_or_else(|| panic!("{module} no longer declares const {name}: usize"))
}

#[test]
fn every_audited_graph_entry_is_reachable_from_a_generated_source() {
    let all = all_graph_entries();
    let audited = audited_graph_names();
    assert_eq!(
        audited.len(),
        AUDITED_GRAPH_ENTRIES,
        "the graph share of NOZI_AUDITED_ENTRIES moved; this census and the poison-parity \
         suite beside it both key on that split"
    );
    let mut missing: Vec<&str> = Vec::new();
    let mut sites = 0usize;
    for e in &audited {
        let hits: Vec<&GraphEntry> = all.iter().filter(|g| g.name == *e).collect();
        if hits.is_empty() {
            missing.push(e);
        }
        sites += hits.len();
    }
    eprintln!(
        "nozi-graph-census: {} audited graph entries, {} declaration sites across {} sources",
        audited.len(),
        sites,
        all_graph_sources().len()
    );
    assert!(
        missing.is_empty(),
        "these audited entries are on NOZI_AUDITED_ENTRIES but no generated source in this \
         crate declares them -- either the list is stale or this census stopped covering a \
         source: {missing:?}"
    );
}

#[test]
fn audited_names_declared_by_more_than_one_graph_are_pinned() {
    let all = all_graph_entries();
    let mut dupes: Vec<String> = Vec::new();
    for e in audited_graph_names() {
        let hits: Vec<&GraphEntry> = all.iter().filter(|g| g.name == e).collect();
        if hits.len() > 1 {
            let srcs: Vec<&str> = hits.iter().map(|h| h.src.as_str()).collect();
            dupes.push(format!("{e} @ {srcs:?}"));
        }
    }
    eprintln!("nozi-graph-census duplicate declaration sites: {dupes:#?}");
    assert!(
        !dupes.is_empty(),
        "the duplicate set went empty -- if the graphs stopped sharing entry names that is a \
         real simplification, but this test is now measuring nothing and must be retired \
         rather than left green"
    );
}

#[test]
fn no_audited_graph_entry_is_discharged_by_declaring_no_workgroup_memory() {
    let all = all_graph_entries();
    let mut trivially_safe: Vec<String> = Vec::new();
    let mut with_wg = 0usize;
    let mut inventory: Vec<String> = Vec::new();
    for e in audited_graph_names() {
        for g in all.iter().filter(|g| g.name == e) {
            if g.workgroup_vars.is_empty() {
                trivially_safe.push(format!("{}::{}", g.src, g.name));
            } else {
                with_wg += 1;
            }
            inventory.push(format!(
                "{}::{} wg={:?} spill={:?}",
                g.src, g.name, g.workgroup_vars, g.thread_arrays
            ));
        }
    }
    eprintln!("nozi-graph-census inventory:\n{}", inventory.join("\n"));
    eprintln!(
        "nozi-graph-census: {with_wg} declaration sites declare workgroup memory, {} do not",
        trivially_safe.len()
    );
    assert!(
        trivially_safe.is_empty(),
        "these audited graph entries declare no workgroup memory and can be discharged \
         mechanically -- record them as discharged instead of leaving the shortcut \
         un-taken: {trivially_safe:?}"
    );
}

#[test]
fn every_audited_graph_site_writes_the_extent_it_later_reads() {
    assert_eq!(
        nv_kernels::wgpu_backend::dispatch::nozi_audited_entries().len(),
        AUDITED_ENTRIES,
        "NOZI_AUDITED_ENTRIES changed size; this crate proves the {AUDITED_GRAPH_ENTRIES} graph \
         names and nv-kernels' own poison suites prove the rest, so a new member is uncovered \
         until one of the two lists is extended"
    );
    let audited = audited_graph_names();
    assert_eq!(audited.len(), AUDITED_GRAPH_ENTRIES, "the graph share moved");

    let mut lane = 0usize;
    let mut subgroup = 0usize;
    let mut scalar = 0usize;
    let mut prefix: Vec<(String, String, String, String, Vec<String>)> = Vec::new();
    let mut unproven: Vec<String> = Vec::new();
    let mut inventory: Vec<String> = Vec::new();
    let mut declared: std::collections::BTreeMap<(String, String), u32> = Default::default();

    for (tag, src) in all_graph_sources() {
        let consts = wgsl_consts(&src);
        let decls = wg_decls(&src, &consts);
        for e in entries_src(&src, &consts) {
            if !audited.contains(&e.name.as_str()) {
                continue;
            }
            for d in &decls {
                let Some(x) = extent_of(&e, d) else { continue };
                inventory.push(format!("{tag}::{}::{} {x:?}", e.name, d.name));
                match &x {
                    Extent::Lane => lane += 1,
                    Extent::Subgroup => subgroup += 1,
                    Extent::Scalar => scalar += 1,
                    Extent::Prefix(b) => {
                        declared.insert(
                            (tag.to_string(), d.name.clone()),
                            d.elems.unwrap_or_else(|| {
                                panic!("{tag}::{} has an unresolved length", d.name)
                            }),
                        );
                        prefix.push((
                            tag.to_string(),
                            e.name.clone(),
                            d.name.clone(),
                            b.clone(),
                            unbounded_reads(&e, d, b),
                        ))
                    }
                    Extent::Unproven(why) => {
                        unproven.push(format!("{tag}::{}::{}: {why}", e.name, d.name))
                    }
                }
            }
        }
    }

    eprintln!("nozi-extent inventory:\n{}", inventory.join("\n"));
    eprintln!(
        "nozi-extent: {lane} lane-covered, {subgroup} subgroup-covered, {scalar} \
         broadcast scalars, {} runtime-prefix writes, {} unproven",
        prefix.len(),
        unproven.len()
    );
    for (tag, entry, var, bound, reads) in &prefix {
        eprintln!("  prefix {tag}::{entry}::{var} bound={bound} reads_outside={reads:?}");
    }

    assert!(
        unproven.is_empty(),
        "these audited graph entries read workgroup memory whose written extent this census \
         cannot establish -- they ship with zero-init disabled, so an unproven extent is an \
         unaudited entry: {unproven:#?}"
    );

    let recorded: Vec<(String, String, String, String, Vec<String>)> = PREFIX_WRITE_SITES
        .iter()
        .map(|(t, e, v, b, r)| {
            (
                t.to_string(),
                e.to_string(),
                v.to_string(),
                b.to_string(),
                r.iter().map(|s| s.to_string()).collect(),
            )
        })
        .collect();
    let missing: Vec<&(String, String, String, String, Vec<String>)> =
        prefix.iter().filter(|p| !recorded.contains(p)).collect();
    let stale: Vec<&(String, String, String, String, Vec<String>)> =
        recorded.iter().filter(|r| !prefix.contains(r)).collect();
    assert!(
        missing.is_empty() && stale.is_empty(),
        "the runtime-prefix write table no longer matches the shaders. A site whose bound token \
         or whose set of reads outside that bound moved has had its safety argument changed and \
         must be re-derived, not re-pinned. new/changed: {missing:#?} gone/changed: {stale:#?}"
    );
    assert_eq!(
        (lane, subgroup, scalar, prefix.len()),
        COVERING_WRITE_CLASSES,
        "the covering-write class mix moved; a site that stopped being lane-covered has had its \
         write narrowed"
    );

    let mut ungated: Vec<String> = Vec::new();
    for (tag, _, var, _, _) in &prefix {
        let key = (tag.clone(), var.clone());
        if BUILD_SIZED_PREFIX_SITES.contains(&(tag.as_str(), var.as_str())) {
            continue;
        }
        let Some((_, _, module, gate)) = EXTENT_GATES
            .iter()
            .find(|(t, v, _, _)| t == tag && v == var)
        else {
            ungated.push(format!("{tag}::{var} has no config gate on EXTENT_GATES"));
            continue;
        };
        let cap = rust_usize_const(module, gate);
        let len = declared[&key];
        if len < cap {
            ungated.push(format!(
                "{tag}::{var} is {len} elements but {module}::{gate} lets a config ask for {cap}"
            ));
        }
    }
    eprintln!(
        "nozi-extent gates: {} runtime-prefix arrays checked against their config gate, {} \
         sized at pipeline build",
        prefix.len() - BUILD_SIZED_PREFIX_SITES.len(),
        BUILD_SIZED_PREFIX_SITES.len()
    );
    assert!(
        ungated.is_empty(),
        "a runtime-prefix write only covers the reads beside it while the runtime bound cannot \
         exceed the declared array. These arrays and the constants that cap them have drifted \
         apart, so a legal config now writes a prefix and reads past it: {ungated:#?}"
    );
}

#[test]
fn every_host_rot_half_is_derived_by_halving_the_head() {
    let mut sites = 0usize;
    let mut bad: Vec<String> = Vec::new();
    for (tag, src) in MODEL_SOURCES {
        for (n, line) in src.lines().enumerate() {
            let Some(init) = line.trim().strip_prefix("rot_half:") else {
                continue;
            };
            let init = init.trim().trim_end_matches(',').trim();
            let packed: String = init.split_whitespace().collect();
            if packed == "u32" {
                continue;
            }
            sites += 1;
            if !packed.starts_with('(') || !packed.contains("/2)") {
                bad.push(format!("{tag}:{}: rot_half: {init}", n + 1));
            }
        }
    }
    eprintln!("nozi-rope: {sites} host rot_half initializers, all halving: {}", bad.is_empty());
    assert!(
        bad.is_empty(),
        "the rope entries on NOZI_AUDITED_ENTRIES write only elements [0, head_dim) of their \
         workgroup buffer and then read element d + rot_half for d < rot_half, so 2 * rot_half \
         > head_dim reads workgroup memory nothing wrote. That is the exact defect \
         wgpu_nozi_graph_poison_parity's rope-bound control provokes on purpose. Nothing in the \
         kernel prevents it -- only these host initializers do: {bad:#?}"
    );
    assert_eq!(
        sites, ROPE_HALVING_SITES,
        "the number of host rot_half initializers moved; a new dispatch site is unaudited until \
         it is counted here. This count is deliberately BROADER than NOZI_AUDITED_ENTRIES: \
         gow_rope and gow_pf_rope are not on that list, but the halving invariant is not a \
         zero-init property. A rope kernel that writes [0, head_dim) and reads e0 + rot_half \
         returns a WRONG value past its write either way -- garbage without zero-init, a \
         deterministic 0.0 with it. Count every site, not only the audited ones"
    );
}
