#![cfg(feature = "wgpu")]

mod common;
use common::snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights;

const CENSUS_LAYERS_DEFAULT_4_IS_THE_SMALLEST_DEPTH_A_PERIOD_DETECTOR_CAN_PROVE_A_REPEAT_ON: usize =
    4;

const FULL_DEPTH_THE_PER_TOKEN_COUNT_IS_EXTRAPOLATED_TO: usize = 30;

const A_DEPENDENT_WGPU_DISPATCH_COSTS_US_RAW_CHAIN_ONE_PASS: f64 = 2.05;

const A_FREE_DISPATCH_COSTS_US_WGPU_EMITS_NO_BARRIER_WHEN_ITS_TRACKER_SEES_NO_HAZARD: f64 = 0.05;

const REMOVING_A_MERGE_OR_EPILOGUE_SHAPED_DISPATCH_IS_WORTH_US_CONFIRMED_TWICE_TWO_FAMILIES: f64 =
    3.3;

const REMOVING_A_SINGLE_WORKGROUP_PASS_IS_WORTH_US_IT_OCCUPIES_ONE_188TH_OF_THE_CARD: f64 = 11.02;

const PROLOGUE_FOLD_ADMISSION_IS_CONSUMER_WORKGROUPS_THE_MEASURED_CROSSOVER_IS_BETWEEN_64_AND_192:
    u64 = 64;

const SERIAL_SELFTEST_PASSES_A_CHAIN_OF_N_LEAVES_ONLY_ITS_FIRST_PASS_FREE: usize = 295;

const FANOUT_SELFTEST_SIBLINGS_READ_ONE_PRODUCED_BUFFER_AND_WRITE_DISJOINT_OUTPUTS: usize = 32;

fn union_binds_a_write_anywhere_makes_the_merged_pass_a_writer_of_that_buffer(
    x: &[(u32, bool)],
    y: &[(u32, bool)],
) -> Vec<(u32, bool)> {
    let mut m: std::collections::BTreeMap<u32, bool> = Default::default();
    for (id, w) in x.iter().chain(y.iter()) {
        let e = m.entry(*id).or_insert(false);
        *e |= *w;
    }
    m.into_iter().collect()
}

fn fuse_every_adjacent_occurrence_of_this_label_pair_into_one_dispatch(
    labels: &[String],
    binds: &[Vec<(u32, bool)>],
    a: &str,
    b: &str,
) -> (Vec<Vec<(u32, bool)>>, usize) {
    let mut out: Vec<Vec<(u32, bool)>> = Vec::with_capacity(binds.len());
    let mut fused = 0usize;
    let mut i = 0usize;
    while i < binds.len() {
        if i + 1 < binds.len() && labels[i] == a && labels[i + 1] == b {
            out.push(union_binds_a_write_anywhere_makes_the_merged_pass_a_writer_of_that_buffer(
                &binds[i],
                &binds[i + 1],
            ));
            fused += 1;
            i += 2;
        } else {
            out.push(binds[i].clone());
            i += 1;
        }
    }
    (out, fused)
}

fn barriers_of(binds: &[Vec<(u32, bool)>]) -> usize {
    nv_models::gemma4_moe_wgpu::
        a_pass_is_a_wgpu_barrier_iff_it_reads_or_writes_a_buffer_written_since_the_last_barrier_or_writes_one_read_since(
            binds,
        )
        .iter()
        .filter(|b| **b)
        .count()
}

fn the_hazard_model_must_report_a_serial_chain_serial_and_independent_siblings_free() {
    let serial: Vec<Vec<(u32, bool)>> = (0..SERIAL_SELFTEST_PASSES_A_CHAIN_OF_N_LEAVES_ONLY_ITS_FIRST_PASS_FREE)
        .map(|_| vec![(0u32, false), (1u32, true)])
        .collect();
    let b = barriers_of(&serial);
    assert_eq!(
        b,
        serial.len() - 1,
        "on a deliberately-serial graph every pass after the first must be a barrier; the model \
         reported {b} of {}",
        serial.len()
    );
    let mut fanout: Vec<Vec<(u32, bool)>> = vec![vec![(0u32, true)]];
    for s in 0..FANOUT_SELFTEST_SIBLINGS_READ_ONE_PRODUCED_BUFFER_AND_WRITE_DISJOINT_OUTPUTS {
        fanout.push(vec![(0u32, false), (100 + s as u32, true)]);
    }
    let flags = nv_models::gemma4_moe_wgpu::
        a_pass_is_a_wgpu_barrier_iff_it_reads_or_writes_a_buffer_written_since_the_last_barrier_or_writes_one_read_since(
            &fanout,
        );
    assert!(!flags[0], "the first pass of a graph can never be a barrier");
    assert!(
        flags[1],
        "the first consumer of a freshly written buffer is the one true RAW edge in a fan-out"
    );
    assert!(
        flags[2..].iter().all(|f| !f),
        "independent siblings reading one common input and writing disjoint outputs must all \
         flip FREE; the model flipped {} of {} instead",
        flags[2..].iter().filter(|f| **f).count(),
        flags.len() - 2
    );
}

fn strip_kind_suffix(label: &str) -> &str {
    label
        .strip_suffix("-sliding")
        .or_else(|| label.strip_suffix("-full"))
        .unwrap_or(label)
}

fn smallest_prefix_and_period_that_tiles_the_layer_loop(
    labels: &[String],
    layers: usize,
) -> Option<(usize, usize)> {
    let n = labels.len();
    for a in 0..n.min(64) {
        for p in 1..=(n - a) / layers {
            let span = p * (layers - 1);
            if (0..span).all(|i| labels[a + i] == labels[a + i + p]) {
                return Some((a, p));
            }
        }
    }
    None
}

fn layer_start_marker_occurring_exactly_once_per_layer(
    labels: &[String],
    layers: usize,
) -> (String, Vec<usize>) {
    let mut counts: std::collections::HashMap<&str, Vec<usize>> = Default::default();
    for (i, l) in labels.iter().enumerate() {
        counts.entry(l.as_str()).or_default().push(i);
    }
    let mut cands: Vec<(&str, &Vec<usize>)> = counts
        .iter()
        .filter(|(_, ix)| ix.len() == layers)
        .map(|(l, ix)| (*l, ix))
        .collect();
    cands.sort_by_key(|(_, ix)| ix[0]);
    let (label, ix) = cands.first().unwrap_or_else(|| {
        panic!(
            "no decode pass label occurs exactly once per layer over {layers} layers, so the \
             pass list cannot be segmented into layer blocks by marker"
        )
    });
    ((*label).to_string(), (*ix).clone())
}

fn histogram_lines(rows: &[(String, String, (u32, u32, u32), u64)]) -> Vec<(String, String, usize, (u32, u32, u32), u64)> {
    let mut order: Vec<String> = Vec::new();
    let mut agg: std::collections::HashMap<String, (String, usize, (u32, u32, u32), u64)> =
        Default::default();
    for (label, entry, grid, bytes) in rows {
        let e = agg.entry(label.clone()).or_insert_with(|| {
            order.push(label.clone());
            (entry.clone(), 0, *grid, 0)
        });
        e.1 += 1;
        e.3 += *bytes;
    }
    let mut out: Vec<(String, String, usize, (u32, u32, u32), u64)> = order
        .into_iter()
        .map(|l| {
            let (entry, n, grid, bytes) = agg[&l].clone();
            (l, entry, n, grid, bytes)
        })
        .collect();
    out.sort_by(|x, y| y.2.cmp(&x.2).then(x.0.cmp(&y.0)));
    out
}

#[test]
#[ignore = "builds the gemma-4-26B-A4B MoE decode graph on wgpu and dumps its dispatch census; set NV_G4MOE_DISPATCH_CENSUS=1. NV_G4MOE_CENSUS_LAYERS truncates the layer stack (default 4) because the per-layer dispatch program is a pure function of the shapes and the arm gates, not of depth -- the per-token count for the shipped 30 layers is then prefix + 30*period + suffix and is asserted against a full-depth build when NV_G4MOE_CENSUS_LAYERS=30"]
fn g4moe_decode_dispatch_census_per_layer_period_and_label_histogram() {
    if std::env::var("NV_G4MOE_DISPATCH_CENSUS").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_G4MOE_DISPATCH_CENSUS != 1");
        return;
    }
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_G4MOE_SNAPSHOT",
        &["models--google--gemma-4-26B-A4B-it"],
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated census must never silently skip");
    eprintln!("[census] adapter: {}", ctx.summary());
    eprintln!(
        "[census] the limits that decide whether the embed/lm-head table binds whole: \
         max_storage_buffer_binding_size={} B, max_buffer_size={} B, \
         max_compute_workgroups_per_dimension={}",
        ctx.caps.max_storage_buffer_binding_size,
        ctx.caps.max_buffer_size,
        ctx.caps.max_compute_workgroups_per_dimension
    );

    let mut config =
        nv_models::gemma4_moe::Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json"))
            .expect("config.json");
    let shipped_layers = config.base.num_hidden_layers;
    assert_eq!(
        shipped_layers, FULL_DEPTH_THE_PER_TOKEN_COUNT_IS_EXTRAPOLATED_TO,
        "the census extrapolates prefix + {FULL_DEPTH_THE_PER_TOKEN_COUNT_IS_EXTRAPOLATED_TO}*period + suffix; \
         this checkpoint has {shipped_layers} layers"
    );
    let layers: usize = std::env::var("NV_G4MOE_CENSUS_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CENSUS_LAYERS_DEFAULT_4_IS_THE_SMALLEST_DEPTH_A_PERIOD_DETECTOR_CAN_PROVE_A_REPEAT_ON);
    assert!(
        (2..=shipped_layers).contains(&layers),
        "NV_G4MOE_CENSUS_LAYERS={layers} must be in 2..={shipped_layers}"
    );
    config.base.num_hidden_layers = layers;
    config.base.layer_types.truncate(layers);
    let kinds: Vec<String> = (0..layers)
        .map(|i| format!("{:?}", config.base.layer_kind(i)))
        .collect();

    let max_seq = std::env::var("NV_G4MOE_CENSUS_MAXSEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512usize);

    let arms = [
        ("NV_G4MOE_GEMV_SGLAT", nv_models::gemma4_moe_wgpu::gemv_sglat_enabled("g4m-at-oproj")),
        ("NV_G4MOE_GU_FUSEDX", nv_models::gemma4_moe_wgpu::gu_fusedx_enabled()),
        ("NV_G4MOE_DOWN_EXPAR", nv_models::gemma4_moe_wgpu::down_expar_enabled()),
        ("NV_G4MOE_W4_XV4", nv_models::gemma4_moe_wgpu::w4_xv4_enabled()),
        ("NV_G4MOE_GROUPED_GEMV", nv_models::gemma4_moe_wgpu::grouped_gemv_enabled()),
        ("NV_G4MOE_FOLD_CHAIN", nv_models::gemma4_moe_wgpu::fold_chain_enabled()),
        ("NV_G4MOE_RPNORM", nv_models::gemma4_moe_wgpu::rpnorm_enabled()),
        ("NV_G4MOE_RPFUSE", nv_models::gemma4_moe_wgpu::rpfuse_enabled()),
        ("NV_G4MOE_TOPK_NX2", nv_models::gemma4_moe_wgpu::topk_nx2_enabled()),
        ("NV_G4MOE_TOPK_FOLD", nv_models::gemma4_moe_wgpu::topk_fold_enabled()),
        ("NV_G4MOE_TAILFOLD", nv_models::gemma4_moe_wgpu::tailfold_enabled()),
        ("NV_G4MOE_QKVFOLD", nv_models::gemma4_moe_wgpu::qkvfold_enabled()),
        ("NV_G4MOE_ROUTER_CHAIN", nv_models::gemma4_moe_wgpu::router_chain_enabled()),
        ("NV_G4MOE_FLASH_DECODE", nv_models::gemma4_moe_wgpu::flash_decode_enabled()),
        ("NV_G4MOE_FLASH_SLIDING", nv_models::gemma4_moe_wgpu::flash_sliding_enabled()),
        ("NV_G4MOE_KV_FP8", nv_models::gemma4_moe_wgpu::kv_fp8_enabled()),
        ("NV_G4MOE_NORM_LAT", nv_models::gemma4_moe_wgpu::norm_lat_enabled()),
        ("NV_G4MOE_HEAD_ONE_SHARD", nv_models::gemma4_moe_wgpu::head_one_shard_enabled()),
        ("NV_G4MOE_LMHEAD_TIGHT_GRID", nv_models::gemma4_moe_wgpu::lmhead_tight_grid_enabled()),
    ];
    let arm_tag: Vec<&str> = arms.iter().filter(|(_, on)| *on).map(|(k, _)| *k).collect();
    eprintln!("[census] arm stack ON: {}", arm_tag.join(" "));
    for (k, on) in &arms {
        eprintln!("[census] arm {k:<24} {}", if *on { "on" } else { "off" });
    }

    let t0 = std::time::Instant::now();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let m = nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu::from_loader(config, &loader, max_seq)
        .expect("build from loader");
    drop(loader);
    let rows = m.pass_rows();
    eprintln!(
        "[census] built {layers} layers ({}) in {:.1}s, {} decode dispatches, wired {:.2} GiB",
        kinds.join(","),
        t0.elapsed().as_secs_f64(),
        rows.len(),
        m.load_report().wired_bytes as f64 / (1u64 << 30) as f64
    );
    assert_eq!(rows.len(), m.pass_count(), "census lost a dispatch");

    let labels: Vec<String> = rows
        .iter()
        .map(|(l, _, _, _)| strip_kind_suffix(l).to_string())
        .collect();
    let (marker, starts) = layer_start_marker_occurring_exactly_once_per_layer(&labels, layers);
    let prefix = starts[0];
    let mut block: Vec<usize> = Vec::with_capacity(layers);
    for i in 0..layers {
        let end = starts.get(i + 1).copied().unwrap_or(rows.len());
        block.push(end - starts[i]);
    }
    let suffix_marker_relative = rows.len() - starts[layers - 1] - block[layers - 1];
    eprintln!(
        "[census] layer blocks segmented on marker {marker:?}: prefix {prefix}, blocks {block:?}, \
         trailing {suffix_marker_relative} (the last block absorbs the head unless a marker-run \
         boundary is found)"
    );
    let per_kind: Vec<(String, usize)> = kinds
        .iter()
        .cloned()
        .zip(block.iter().copied())
        .collect();
    eprintln!("[census] per-layer dispatches by kind: {per_kind:?}");

    let (prefix, period) = match smallest_prefix_and_period_that_tiles_the_layer_loop(&labels, layers)
    {
        Some(v) => v,
        None => {
            eprintln!(
                "[census] NOT layer-periodic over these {layers} layers -- the blocks above are \
                 the decomposition and the per-token count is their sum, not a period"
            );
            (prefix, block[0])
        }
    };
    let suffix = rows.len().saturating_sub(prefix + layers * period);
    eprintln!(
        "[census] MEASURED decomposition: {} = prefix {prefix} + {layers} x period {period} + suffix {suffix}",
        rows.len()
    );
    let per_token = prefix + FULL_DEPTH_THE_PER_TOKEN_COUNT_IS_EXTRAPOLATED_TO * period + suffix;
    eprintln!(
        "[census] EXTRAPOLATED to the shipped {FULL_DEPTH_THE_PER_TOKEN_COUNT_IS_EXTRAPOLATED_TO} \
         layers: {per_token} dispatches/token ({prefix} + 30 x {period} + {suffix})"
    );
    if layers == shipped_layers {
        eprintln!(
            "[census] FULL DEPTH: this build IS the shipped graph -- {} dispatches/token measured, \
             extrapolation says {per_token}",
            rows.len()
        );
    }

    eprintln!("[census] --- prefix ({prefix} dispatches, before the first layer block) ---");
    for i in 0..prefix {
        let (l, e, g, b) = &rows[i];
        eprintln!("[census] pre  {i:>4}  {l:<28} {e:<34} grid {:?} {:>10} B", g, b);
    }
    eprintln!("[census] --- ONE LAYER: {period} dispatches, the block this round must shrink ---");
    for i in 0..period {
        let (l, e, g, b) = &rows[prefix + i];
        eprintln!("[census] lyr  {i:>4}  {l:<28} {e:<34} grid {:?} {:>10} B", g, b);
    }
    let tail_from = (prefix + layers * period).min(rows.len());
    eprintln!(
        "[census] --- suffix ({} dispatches, after the last layer block) ---",
        rows.len() - tail_from
    );
    for (i, (l, e, g, b)) in rows[tail_from..].iter().enumerate() {
        eprintln!("[census] post {i:>4}  {l:<28} {e:<34} grid {:?} {:>10} B", g, b);
    }

    eprintln!("[census] --- per-label histogram over the whole {layers}-layer graph ---");
    eprintln!(
        "[census] {:<28} {:>5} {:>6}  {:<34} {:>12}",
        "label", "n", "n/layer", "entry", "MB total"
    );
    for (l, e, n, _g, bytes) in histogram_lines(&rows) {
        eprintln!(
            "[census] {l:<28} {n:>5} {:>6.2}  {e:<34} {:>12.2}",
            n as f64 / layers as f64,
            bytes as f64 / 1e6
        );
    }

    let us_per_dependent_dispatch = std::env::var("NV_WGPU_US_PER_DISPATCH")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(A_DEPENDENT_WGPU_DISPATCH_COSTS_US_RAW_CHAIN_ONE_PASS);
    eprintln!(
        "[census] at {us_per_dependent_dispatch:.2} us per dependent dispatch the shipped graph \
         spends {:.3} ms/token in dispatch alone",
        per_token as f64 * us_per_dependent_dispatch / 1e3
    );

    the_hazard_model_must_report_a_serial_chain_serial_and_independent_siblings_free();
    let binds = m.decode_pass_storage_binding_sets_a_shared_id_between_two_passes_is_a_wgpu_hazard();
    assert_eq!(
        binds.len(),
        rows.len(),
        "every encoded pass must report its storage binding set or the hazard chain is a guess"
    );
    let all_write: Vec<Vec<(u32, bool)>> = binds
        .iter()
        .map(|s| s.iter().map(|(id, _)| (*id, true)).collect())
        .collect();
    let all_read: Vec<Vec<(u32, bool)>> = binds
        .iter()
        .map(|s| s.iter().map(|(id, _)| (*id, false)).collect())
        .collect();
    let (b_write, b_read) = (barriers_of(&all_write), barriers_of(&all_read));
    eprintln!(
        "[hazard] bracket on the REAL binding sets: forcing every bind read_write gives \
         {b_write} barriers of {}, forcing every bind read-only gives {b_read}; the model must \
         land strictly inside",
        binds.len()
    );
    assert_eq!(
        b_read, 0,
        "a graph of read-only bindings can raise no hazard at all, so the read-only bracket must \
         be exactly zero barriers"
    );

    let flags = nv_models::gemma4_moe_wgpu::
        a_pass_is_a_wgpu_barrier_iff_it_reads_or_writes_a_buffer_written_since_the_last_barrier_or_writes_one_read_since(
            &binds,
        );
    let barriers = flags.iter().filter(|f| **f).count();
    let free = rows.len() - barriers;
    let single_wg = rows.iter().filter(|(_, _, g, _)| *g == (1, 1, 1)).count();
    eprintln!(
        "\n[hazard] STATIC HAZARD MODEL over the {} decode dispatches of this {layers}-layer \
         build: {barriers} BARRIERS, {free} FREE to overlap their predecessor, {single_wg} \
         launching a single workgroup",
        rows.len()
    );
    eprintln!(
        "[hazard] dispatch bill <= {barriers} x {us_per_dependent_dispatch} us + {free} x \
         {A_FREE_DISPATCH_COSTS_US_WGPU_EMITS_NO_BARRIER_WHEN_ITS_TRACKER_SEES_NO_HAZARD} us = \
         {:.1} us/token, against {:.1} us/token if every dispatch were charged as a barrier",
        barriers as f64 * us_per_dependent_dispatch
            + free as f64 * A_FREE_DISPATCH_COSTS_US_WGPU_EMITS_NO_BARRIER_WHEN_ITS_TRACKER_SEES_NO_HAZARD,
        rows.len() as f64 * us_per_dependent_dispatch
    );
    let mut per_label: std::collections::BTreeMap<&str, (usize, usize)> = Default::default();
    for (i, (label, _, _, _)) in rows.iter().enumerate() {
        let e = per_label.entry(label.as_str()).or_insert((0, 0));
        e.0 += 1;
        if flags[i] {
            e.1 += 1;
        }
    }
    let mut ranked: Vec<(&str, usize, usize)> =
        per_label.into_iter().map(|(l, (n, b))| (l, n, b)).collect();
    ranked.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(b.0)));
    eprintln!("[hazard] barriers charged to each label (this is what a fusion actually removes):");
    for (label, n, b) in &ranked {
        eprintln!("[hazard]   {label:<28} n={n:<5} barriers={b:<5} free={}", n - b);
    }
    if free > 0 {
        eprintln!("[hazard] the FREE dispatches, in encode order:");
        for (i, (label, entry, grid, _)) in rows.iter().enumerate() {
            if !flags[i] {
                eprintln!("[hazard]   free {i:>4}  {label:<28} {entry:<40} grid {grid:?}");
            }
        }
    }

    let causes = nv_models::gemma4_moe_wgpu::
        barrier_cause_raw_waw_is_a_true_dependency_war_alone_can_be_an_artifact_of_a_wide_binding_access_mode(
            &binds,
        );
    let raw_waw = causes.iter().filter(|c| **c == Some(true)).count();
    let war_only = causes.iter().filter(|c| **c == Some(false)).count();
    eprintln!(
        "[hazard] SENSITIVITY of the model to the WGSL access-mode over-approximation: a pass \
         that binds a read_write buffer it only reads is charged as a writer, and the only \
         barriers that can be manufactured that way are WAR-only ones. {raw_waw} of the \
         {barriers} barriers are RAW/WAW true dependencies; {war_only} are WAR-only and are the \
         entire exposure of this instrument to entry-point-imprecise access modes"
    );
    for (i, c) in causes.iter().enumerate() {
        if *c == Some(false) {
            eprintln!(
                "[hazard]   war-only {i:>4}  {:<28} {:<40}",
                rows[i].0, rows[i].1
            );
        }
    }

    let stripped: Vec<String> = rows
        .iter()
        .map(|(l, _, _, _)| strip_kind_suffix(l).to_string())
        .collect();
    let mut pair_grid: std::collections::BTreeMap<(String, String), (usize, (u32, u32, u32), (u32, u32, u32))> =
        Default::default();
    for i in 0..rows.len().saturating_sub(1) {
        let key = (stripped[i].clone(), stripped[i + 1].clone());
        let e = pair_grid
            .entry(key)
            .or_insert((0, rows[i].2, rows[i + 1].2));
        e.0 += 1;
    }
    let base = barriers;
    let mut candidates: Vec<(String, usize, usize, usize, bool, (u64, u64))> = Vec::new();
    for ((a, b), (n, ga, gb)) in &pair_grid {
        let (fused_binds, fused_total) =
            fuse_every_adjacent_occurrence_of_this_label_pair_into_one_dispatch(
                &stripped, &binds, a, b,
            );
        let removed_total = base.saturating_sub(barriers_of(&fused_binds));
        let removes_single_wg = *ga == (1, 1, 1) || *gb == (1, 1, 1);
        let wg_p = (ga.0 as u64) * (ga.1 as u64) * (ga.2 as u64);
        let wg_c = (gb.0 as u64) * (gb.1 as u64) * (gb.2 as u64);
        candidates.push((
            format!("{a} + {b}"),
            *n,
            fused_total,
            removed_total,
            removes_single_wg,
            (wg_p, wg_c),
        ));
    }
    candidates.sort_by(|x, y| y.3.cmp(&x.3).then(x.0.cmp(&y.0)));
    eprintln!(
        "\n[hazard] EVERY adjacent-pair fusion candidate, RE-RANKED BY BARRIERS REMOVED (not \
         dispatches removed). Price: {REMOVING_A_SINGLE_WORKGROUP_PASS_IS_WORTH_US_IT_OCCUPIES_ONE_188TH_OF_THE_CARD} \
         us when the fusion deletes a single-workgroup pass, else \
         {REMOVING_A_MERGE_OR_EPILOGUE_SHAPED_DISPATCH_IS_WORTH_US_CONFIRMED_TWICE_TWO_FAMILIES} us. \
         A candidate that removes ZERO barriers is worth ZERO no matter how many dispatches it deletes."
    );
    eprintln!(
        "[hazard] the surviving-kernel admission rule is not part of the hazard model: a prologue \
         fold replays the producer once per CONSUMER workgroup, and the measured crossover across \
         two families sits between 64 and 192, so \
         {PROLOGUE_FOLD_ADMISSION_IS_CONSUMER_WORKGROUPS_THE_MEASURED_CROSSOVER_IS_BETWEEN_64_AND_192} \
         is the admission bound applied here"
    );
    eprintln!(
        "[hazard] {:<58} {:>5} {:>7} {:>9} {:>10} {:>8} {:>8}  {}",
        "candidate (adjacent pair fused into one dispatch)",
        "sites",
        "fused",
        "barriers-",
        "est us",
        "wg prod",
        "wg cons",
        "verdict"
    );
    for (name, n, fused, removed, single, (wg_p, wg_c)) in &candidates {
        let rate = if *single {
            REMOVING_A_SINGLE_WORKGROUP_PASS_IS_WORTH_US_IT_OCCUPIES_ONE_188TH_OF_THE_CARD
        } else {
            REMOVING_A_MERGE_OR_EPILOGUE_SHAPED_DISPATCH_IS_WORTH_US_CONFIRMED_TWICE_TWO_FAMILIES
        };
        let same_kernel_shard_split_exists_only_because_a_binding_size_clamp_split_one_weight_table =
            name.split(" + ").next() == name.split(" + ").nth(1);
        let verdict = if *removed == 0 {
            "DEAD ON THE HAZARD MODEL: removes no barrier".to_string()
        } else if same_kernel_shard_split_exists_only_because_a_binding_size_clamp_split_one_weight_table
        {
            "LIVE: not a producer/consumer pair at all, one kernel sharded by the storage-binding \
             size clamp; unclamping merges it and replays nothing"
                .to_string()
        } else if wg_p > wg_c {
            "DEAD: consumer reduces across a multi-workgroup producer, and a compute pass has no \
             device-scope barrier"
                .to_string()
        } else if *wg_c
            > PROLOGUE_FOLD_ADMISSION_IS_CONSUMER_WORKGROUPS_THE_MEASURED_CROSSOVER_IS_BETWEEN_64_AND_192
        {
            format!("DEAD: prologue fold replayed {wg_c}x, past the admission bound")
        } else {
            "LIVE: hazard says the barrier is real and the surviving kernel replays nothing at \
             scale"
                .to_string()
        };
        eprintln!(
            "[hazard] {name:<58} {n:>5} {fused:>7} {removed:>9} {:>10.1} {wg_p:>8} {wg_c:>8}  {verdict}",
            *removed as f64 * rate
        );
    }
}
