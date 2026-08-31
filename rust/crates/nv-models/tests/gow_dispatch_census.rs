#![cfg(feature = "wgpu")]

mod common;
use common::snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights;

use nv_kernels::wgpu_backend::dispatch;
use nv_models::gpt_oss_wgpu as gow;

const CENSUS_GATE_ENV: &str = "NV_GOW_CENSUS_TEST";

const PROFILED_DECODE_STEPS_8_IS_ENOUGH_TO_PROVE_THE_COUNT_IS_PER_STEP_CONSTANT: usize = 8;

const CENSUS_MAX_SEQ_KEEPS_KV_TINY_BECAUSE_DISPATCH_COUNT_DOES_NOT_DEPEND_ON_DEPTH: usize = 320;

const A_DEPENDENT_WGPU_DISPATCH_COSTS_US: f64 = 2.05;

fn per_layer_share(total: usize, layers: usize) -> String {
    if total.is_multiple_of(layers) {
        format!("{}/layer", total / layers)
    } else {
        format!("{:.3}/layer", total as f64 / layers as f64)
    }
}

#[test]
#[ignore = "loads the ~13 GB gpt-oss-20b MXFP4 on wgpu; set NV_GOW_CENSUS_TEST=1 -- prints the \
            per-decode-token dependent-dispatch census (pass list, per-label histogram, per-layer \
            decomposition) and cross-checks it against the NV_WGPU_PROFILE instrument"]
fn gow_decode_dispatch_census_per_token_and_per_layer() {
    if std::env::var(CENSUS_GATE_ENV).as_deref() != Ok("1") {
        eprintln!("[skip] {CENSUS_GATE_ENV}=1 not set");
        return;
    }
    std::env::set_var("NV_WGPU_PROFILE", "1");

    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_GPTOSS_DIR",
        &["models--openai--gpt-oss-20b"],
    );
    let cfg = gow::GptOssConfig::from_hf_json_file(dir.join("config.json")).expect("config.json");
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated census must never silently skip");
    eprintln!(
        "[census] adapter: {} | {:?} | {:?} | driver {} {}",
        ctx.info.name, ctx.info.backend, ctx.info.device_type, ctx.info.driver, ctx.info.driver_info
    );
    eprintln!("[census] timestamp_query cap: {}", ctx.caps.timestamp_query);
    assert!(
        dispatch::profile::enabled(),
        "the census reads its cross-check from the NV_WGPU_PROFILE instrument; the arm cache \
         must have taken the value this test set before any other read"
    );

    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let mut gpu = gow::GptOssWgpu::from_loader(
        cfg.clone(),
        &loader,
        CENSUS_MAX_SEQ_KEEPS_KV_TINY_BECAUSE_DISPATCH_COUNT_DOES_NOT_DEPEND_ON_DEPTH,
    )
    .unwrap_or_else(|e| panic!("build from loader: {e:#}"));

    let rows = gpu.decode_pass_labels_and_grids_every_entry_is_one_dependent_dispatch();
    let head_start = gpu.decode_head_start_passes_below_it_also_run_on_a_prefill_only_step();
    let layers = cfg.num_hidden_layers;
    assert_eq!(
        rows.len(),
        gpu.pass_count(),
        "the label+grid dump must enumerate exactly the encoded pass list"
    );

    eprintln!(
        "\n[census] gpt-oss-20b wgpu decode graph: {} dependent dispatches per token \
         ({} body + {} head), {} layers = {:.1} us of pure dispatch latency at \
         {A_DEPENDENT_WGPU_DISPATCH_COSTS_US} us each",
        rows.len(),
        head_start,
        rows.len() - head_start,
        layers,
        rows.len() as f64 * A_DEPENDENT_WGPU_DISPATCH_COSTS_US
    );

    let exits: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter(|(_, (l, _))| l.starts_with("gow-rmsres-next") || l.starts_with("gow-resadd"))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        exits.len(),
        layers,
        "each layer leaves through exactly one residual pass, and that landmark is what makes \
         the per-layer split a measurement instead of a division"
    );
    let mut spans: Vec<usize> = Vec::with_capacity(layers);
    let mut start = rows
        .iter()
        .position(|(l, _)| l.starts_with("gow-at-"))
        .expect("layer 0 opens with an attention pass");
    for (li, &e) in exits.iter().enumerate() {
        spans.push(e + 1 - start);
        let _ = li;
        start = e + 1;
    }
    let in_layers: usize = spans.iter().sum();
    let mut span_hist: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (li, &n) in spans.iter().enumerate() {
        span_hist.entry(n).or_default().push(li);
    }
    eprintln!(
        "[census] {in_layers} of the {} dispatches sit inside a layer span ({} outside: prologue \
         gathers, layer-0 rmsnorm, final rmsnorm, lm head, argmax); layer spans:",
        rows.len(),
        rows.len() - in_layers
    );
    for (n, lis) in &span_hist {
        eprintln!(
            "  {n:>3} dispatches x {} layers (e.g. layer {})",
            lis.len(),
            lis[0]
        );
    }

    let mut hist: std::collections::BTreeMap<String, (usize, usize)> =
        std::collections::BTreeMap::new();
    for (label, grid) in &rows {
        let e = hist.entry(label.clone()).or_insert((0, 0));
        e.0 += 1;
        if *grid == (1, 1, 1) {
            e.1 += 1;
        }
    }
    let mut by_count: Vec<(String, usize, usize)> =
        hist.into_iter().map(|(k, (n, s))| (k, n, s)).collect();
    by_count.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    eprintln!("\n[census] per-label dispatch histogram (per decode token):");
    for (label, n, single) in &by_count {
        eprintln!(
            "  {label:<56} n={n:<5} {:<12} single-workgroup={single}",
            per_layer_share(*n, layers)
        );
    }

    let single_wg: usize = rows.iter().filter(|(_, g)| *g == (1, 1, 1)).count();
    eprintln!(
        "\n[census] {single_wg} of {} dispatches launch a SINGLE workgroup: those are pure \
         latency, not work",
        rows.len()
    );

    let first_layer_marker = "gow-rmsres-post";
    let boundaries: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter(|(_, (l, _))| l.starts_with(first_layer_marker))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        boundaries.len(),
        layers,
        "one post-attention rmsnorm-residual per layer is the census's layer landmark"
    );
    eprintln!("\n[census] first layer's pass sequence, in encode order:");
    let l0_end = rows
        .iter()
        .position(|(l, _)| l.starts_with("gow-rmsres-next") || l.starts_with("gow-resadd"))
        .expect("a layer-exit residual pass");
    for (i, (label, grid)) in rows.iter().take(l0_end + 1).enumerate() {
        eprintln!("  {i:>3}  {label:<56} grid={grid:?}");
    }
    eprintln!("\n[census] head (from index {head_start}):");
    for (i, (label, grid)) in rows.iter().enumerate().skip(head_start) {
        eprintln!("  {i:>3}  {label:<56} grid={grid:?}");
    }

    let binds = gpu.decode_pass_storage_binding_sets_a_shared_id_between_two_passes_is_a_wgpu_hazard();
    assert_eq!(
        binds.len(),
        rows.len(),
        "every encoded pass must report its storage binding set or the hazard chain is a guess"
    );
    let mut read_since: std::collections::BTreeSet<u32> = Default::default();
    let mut written_since: std::collections::BTreeSet<u32> = Default::default();
    let mut barrier_at: Vec<bool> = Vec::with_capacity(rows.len());
    for set in &binds {
        let raw_or_waw = set.iter().any(|(id, _)| written_since.contains(id));
        let war = set
            .iter()
            .any(|(id, w)| *w && (read_since.contains(id) || written_since.contains(id)));
        let conflicts = raw_or_waw || war;
        barrier_at.push(conflicts);
        if conflicts {
            read_since.clear();
            written_since.clear();
        }
        for (id, w) in set {
            if *w {
                written_since.insert(*id);
            } else {
                read_since.insert(*id);
            }
        }
    }
    let barriers = barrier_at.iter().filter(|b| **b).count();
    let free = rows.len() - barriers;
    let dependent_us = 2.05f64;
    let free_us = 0.05f64;
    eprintln!(
        "\n[census] HAZARD CHAIN over the storage bindings, read/write taken from the WGSL \
         access mode (read-only bindings can never raise a hazard; uniforms are not tracked): \
         {barriers} barriers, {free} dispatches free to overlap their predecessor"
    );
    eprintln!(
        "[census] dispatch bill <= {barriers} x {dependent_us} us + {free} x {free_us} us = \
         {:.1} us/token, against the {:.1} us/token that charging all {} at {dependent_us} us gives",
        barriers as f64 * dependent_us + free as f64 * free_us,
        rows.len() as f64 * dependent_us,
        rows.len()
    );
    let mut starts: std::collections::BTreeMap<&str, (usize, usize)> = Default::default();
    for (i, (label, _)) in rows.iter().enumerate() {
        let e = starts.entry(label.as_str()).or_insert((0, 0));
        e.0 += 1;
        if barrier_at[i] {
            e.1 += 1;
        }
    }
    let mut ranked: Vec<(&str, usize, usize)> =
        starts.into_iter().map(|(l, (n, b))| (l, n, b)).collect();
    ranked.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(b.0)));
    eprintln!("[census] barriers charged to each label (this is what a fusion actually removes):");
    for (label, n, b) in &ranked {
        eprintln!("  {label:<56} n={n:<5} barriers={b}");
    }

    dispatch::profile::reset();
    let mut tok = 2000u32;
    for _ in 0..PROFILED_DECODE_STEPS_8_IS_ENOUGH_TO_PROVE_THE_COUNT_IS_PER_STEP_CONSTANT {
        tok = gpu.decode_step(tok).expect("decode step");
    }
    let observed = dispatch::profile::dispatches();
    eprintln!(
        "\n{}",
        dispatch::profile::table("gow-decode", rows.len())
    );
    assert_eq!(
        observed as usize,
        rows.len() * PROFILED_DECODE_STEPS_8_IS_ENOUGH_TO_PROVE_THE_COUNT_IS_PER_STEP_CONSTANT,
        "the instrument must see exactly the encoded pass list once per decode step; a mismatch \
         means part of the graph bypasses encode_pass_list/submit_profiled and the census is blind"
    );
}
