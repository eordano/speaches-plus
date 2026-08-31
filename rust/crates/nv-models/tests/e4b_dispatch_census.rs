#![cfg(feature = "wgpu")]

mod common;
use common::env_usize;
use common::med;
use common::snapshot_dir;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::WgpuContext;
use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_e4b_wgpu::{Gemma4E4bWgpu, ProfMode};
use std::collections::BTreeMap;
use std::time::Instant;

const EVERY_ADJACENT_PAIR_IN_ONE_COMPUTE_PASS_IS_A_DEPENDENT_EDGE_PER_WEBGPU_SPEC: &str =
    "WebGPU orders dispatches inside a compute pass and inserts the memory barrier between them, \
     so the decode graph's adjacency list IS its dependency list and every adjacent pair is a \
     fusion candidate";

const DEPENDENT_DISPATCH_US_MEASURED_THIS_SESSION_NOT_THE_CUDA_DELTA: f64 = 2.05;

const PROMPT: [u32; 8] = [2, 818, 3029, 529, 6081, 603, 563, 1596];

fn lo_of(xs: &[f64]) -> f64 {
    xs.iter().cloned().fold(f64::INFINITY, f64::min)
}

fn base_of(label: &str) -> &str {
    match label.find(" [") {
        Some(i) => &label[..i],
        None => label,
    }
}

fn quiet_card_or_refuse(where_: &str) {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output();
    let Ok(out) = out else {
        return;
    };
    let used: u64 = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .and_then(|l| l.trim().parse().ok())
        .unwrap_or(0);
    eprintln!("[census] card at {where_}: {used} MiB used");
}

type Arm = (&'static str, &'static [(&'static str, &'static str)]);

const ARMS: &[Arm] = &[
    (
        "FUSE_QKV: fused_attn_q/k/v -> one fused_attn_qkv",
        &[("NV_E4B_WGPU_FUSE_QKV", "1")],
    ),
    (
        "FUSE_GELU: gelu_mul folded into the gate_up gemv",
        &[("NV_E4B_WGPU_FUSE_GELU", "1")],
    ),
    (
        "FUSE_GELU+FOLD_FNCA: also absorbs fused_norm_a",
        &[
            ("NV_E4B_WGPU_FUSE_GELU", "1"),
            ("NV_E4B_WGPU_FOLD_FNCA", "1"),
        ],
    ),
    (
        "FUSE_PLGATE: gatemul folded into the pl_gate gemv",
        &[("NV_E4B_WGPU_FUSE_PLGATE", "1")],
    ),
    (
        "FUSE_PLGATE+FOLD_FNCB: also absorbs fused_norm_b",
        &[
            ("NV_E4B_WGPU_FUSE_PLGATE", "1"),
            ("NV_E4B_WGPU_FOLD_FNCB", "1"),
        ],
    ),
    (
        "FOLD_FNCC: fused_norm_c deferred into the next layer",
        &[("NV_E4B_WGPU_FOLD_FNCC", "1")],
    ),
    (
        "FUSE_HEAD: softcap folded into argmax stage 1",
        &[("NV_E4B_WGPU_FUSE_HEAD", "1")],
    ),
    (
        "every fold above, together",
        &[
            ("NV_E4B_WGPU_FUSE_HEAD", "1"),
            ("NV_E4B_WGPU_FUSE_QKV", "1"),
            ("NV_E4B_WGPU_FUSE_GELU", "1"),
            ("NV_E4B_WGPU_FOLD_FNCA", "1"),
            ("NV_E4B_WGPU_FUSE_PLGATE", "1"),
            ("NV_E4B_WGPU_FOLD_FNCB", "1"),
            ("NV_E4B_WGPU_FOLD_FNCC", "1"),
        ],
    ),
    (
        "MERGE+EPILOGUE ONLY: every fold except the two wide prologue folds",
        &[
            ("NV_E4B_WGPU_FUSE_HEAD", "1"),
            ("NV_E4B_WGPU_FUSE_QKV", "1"),
            ("NV_E4B_WGPU_FUSE_GELU", "1"),
            ("NV_E4B_WGPU_FUSE_PLGATE", "1"),
            ("NV_E4B_WGPU_FOLD_FNCB", "1"),
        ],
    ),
    (
        "TAIL_FNCA: fused_norm_a pulled BACK into the o-proj gemv tail",
        &[("NV_E4B_WGPU_TAIL_FNCA", "1")],
    ),
    (
        "TAIL_FNCC: fused_norm_c pulled BACK into the pl-proj gemv tail",
        &[("NV_E4B_WGPU_TAIL_FNCC", "1")],
    ),
    (
        "TAIL_FNCA+TAIL_FNCC: both backward folds, no other arm",
        &[
            ("NV_E4B_WGPU_TAIL_FNCA", "1"),
            ("NV_E4B_WGPU_TAIL_FNCC", "1"),
        ],
    ),
    (
        "TAIL_GATE_ONLY null probe: the arrival fence, folding nothing",
        &[("NV_E4B_WGPU_TAIL_GATE_ONLY", "1")],
    ),
    (
        "SHAPESTACK+TAIL_FNCA+TAIL_FNCC: the 473 stack plus both tails",
        &[
            ("NV_E4B_WGPU_FUSE_HEAD", "1"),
            ("NV_E4B_WGPU_FUSE_QKV", "1"),
            ("NV_E4B_WGPU_FUSE_GELU", "1"),
            ("NV_E4B_WGPU_FUSE_PLGATE", "1"),
            ("NV_E4B_WGPU_FOLD_FNCB", "1"),
            ("NV_E4B_WGPU_TAIL_FNCA", "1"),
            ("NV_E4B_WGPU_TAIL_FNCC", "1"),
        ],
    ),
    (
        "all except FOLD_FNCA (the 1280-workgroup prologue fold)",
        &[
            ("NV_E4B_WGPU_FUSE_HEAD", "1"),
            ("NV_E4B_WGPU_FUSE_QKV", "1"),
            ("NV_E4B_WGPU_FUSE_GELU", "1"),
            ("NV_E4B_WGPU_FUSE_PLGATE", "1"),
            ("NV_E4B_WGPU_FOLD_FNCB", "1"),
            ("NV_E4B_WGPU_FOLD_FNCC", "1"),
        ],
    ),
];

fn arm_selected(name: &str) -> bool {
    match std::env::var("NV_E4B_CENSUS_ARM_FILTER") {
        Ok(f) if !f.is_empty() => f
            .split(',')
            .any(|p| !p.is_empty() && name.contains(p)),
        _ => true,
    }
}

fn logits_bit_trace(m: &mut Gemma4E4bWgpu, feed: &[u32]) -> Vec<Vec<u32>> {
    m.reset();
    feed.iter()
        .map(|&t| {
            let (_, lg) = m.decode_step_logits(t).expect("logits");
            lg.iter().map(|x| x.to_bits()).collect()
        })
        .collect()
}

fn wall_floor_and_median(m: &mut Gemma4E4bWgpu, warm: usize, steps: usize) -> (f64, f64) {
    m.reset();
    let mut t = 0u32;
    for &p in &PROMPT {
        t = m.decode_step(p).expect("prompt");
    }
    for _ in 0..warm {
        t = m.decode_step(t).expect("warm");
    }
    let mut xs = Vec::with_capacity(steps);
    for _ in 0..steps {
        let s = Instant::now();
        t = m.decode_step(t).expect("step");
        xs.push(s.elapsed().as_secs_f64() * 1e3);
    }
    (lo_of(&xs), med(&mut xs))
}

#[test]
#[ignore = "loads the E4B QAT checkpoint and prints the decode dispatch census; set NV_E4B_CENSUS=1"]
fn e4b_decode_dispatch_census() {
    assert_eq!(
        std::env::var("NV_E4B_CENSUS").ok().as_deref(),
        Some("1"),
        "set NV_E4B_CENSUS=1 -- a silent skip here would report a pass"
    );
    let ctx = WgpuContext::shared().expect("wgpu adapter");
    eprintln!("adapter: {}", ctx.summary());
    quiet_card_or_refuse("build");

    let dir = snapshot_dir();
    eprintln!("checkpoint: {}", dir.display());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("config");
    let n_layers = config.num_hidden_layers;
    let kv_shared = config.num_kv_shared_layers;
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("safetensors");
    let max_seq = env_usize("NV_E4B_CENSUS_SEQ", 512);
    let t0 = Instant::now();
    let mut m = Gemma4E4bWgpu::from_loader(config, &loader, max_seq).expect("build graph");
    let n = m.pass_count();
    let (blk, v4, sg) = m.w4_route_census();
    eprintln!(
        "built in {:.1}s -- {n} decode dispatches/token, {n_layers} layers ({kv_shared} kv-shared), \
         {:.3} GB weights/token, w4 route block/v4/sg16 {blk}/{v4}/{sg}",
        t0.elapsed().as_secs_f64(),
        m.weight_bytes_per_token() as f64 / 1e9
    );
    assert!(
        sg > 0,
        "no sg16 w4 projections in the graph; this is not the shipping route and the census would \
         describe a graph serving never runs"
    );

    let labels: Vec<String> = (0..n).map(|i| m.pass_label(i).to_string()).collect();
    let bases: Vec<&str> = labels.iter().map(|l| base_of(l)).collect();

    let mut hist: BTreeMap<&str, usize> = BTreeMap::new();
    for b in &bases {
        *hist.entry(b).or_insert(0) += 1;
    }
    let mut hist: Vec<(&str, usize)> = hist.into_iter().collect();
    hist.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    eprintln!("\n==== per-label dispatch histogram, one decode token ====");
    eprintln!("{:<36} {:>6} {:>8} {:>10}", "label", "n", "% of n", "us @2.05");
    for (l, c) in &hist {
        eprintln!(
            "{l:<36} {c:>6} {:>7.1}% {:>10.1}",
            100.0 * *c as f64 / n as f64,
            *c as f64 * DEPENDENT_DISPATCH_US_MEASURED_THIS_SESSION_NOT_THE_CUDA_DELTA
        );
    }

    let anchor = hist
        .iter()
        .filter(|(_, c)| *c == n_layers)
        .map(|(l, _)| *l)
        .min_by_key(|l| bases.iter().position(|b| b == l).unwrap_or(usize::MAX))
        .unwrap_or("flash_stage1");
    let at: Vec<usize> = (0..n).filter(|&i| bases[i] == anchor).collect();
    assert_eq!(
        at.len(),
        n_layers,
        "anchor {anchor} occurs {} times, not once per layer; the segmentation below would be \
         fiction",
        at.len()
    );
    let prologue = at[0];
    let epilogue_start = *at.last().unwrap();
    eprintln!(
        "\n==== structural decomposition, anchored on the first per-layer label `{anchor}` ====\n\
         prologue before layer 0 : {prologue} dispatches\n\
         layer blocks            : {n_layers} blocks spanning [{prologue}, {epilogue_start})+last",
    );

    let mut segs: Vec<(usize, Vec<String>)> = Vec::with_capacity(n_layers);
    for (li, &start) in at.iter().enumerate() {
        let end = if li + 1 < at.len() { at[li + 1] } else { n };
        segs.push((li, labels[start..end].to_vec()));
    }
    let last_len = segs.last().map(|s| s.1.len()).unwrap_or(0);
    let typical: usize = if n_layers > 1 {
        segs[..n_layers - 1].iter().map(|s| s.1.len()).sum::<usize>() / (n_layers - 1)
    } else {
        last_len
    };
    let epilogue_guess = last_len.saturating_sub(typical);
    eprintln!(
        "  block sizes: {:?}",
        segs.iter().map(|s| s.1.len()).collect::<Vec<_>>()
    );
    eprintln!(
        "  the final block carries the epilogue: {last_len} = ~{typical} layer + ~{epilogue_guess} \
         head/argmax"
    );

    let mut shapes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (li, s) in &segs {
        let sig = s
            .iter()
            .map(|l| base_of(l).to_string())
            .collect::<Vec<_>>()
            .join(" -> ");
        shapes.entry(sig).or_default().push(*li);
    }
    eprintln!(
        "\n==== distinct layer-block shapes ({} of {n_layers} layers collapse to {} shapes) ====",
        n_layers,
        shapes.len()
    );
    let mut shapes: Vec<(String, Vec<usize>)> = shapes.into_iter().collect();
    shapes.sort_by_key(|(_, v)| v[0]);
    for (sig, members) in &shapes {
        let k = sig.split(" -> ").count();
        eprintln!(
            "\n  {k} dispatches x {} layers {:?}:",
            members.len(),
            members
        );
        for (j, step) in sig.split(" -> ").enumerate() {
            let idx = at[members[0]] + j;
            eprintln!(
                "    {j:>3}  {step:<34} {:?}  bound {:.2} MB widest {:.2} MB",
                m.pass_grid(idx),
                m.pass_bound_bytes(idx).0 as f64 / 1e6,
                m.pass_bound_bytes(idx).1 as f64 / 1e6
            );
        }
    }

    eprintln!("\n==== prologue, in order ====");
    for i in 0..prologue {
        eprintln!("  {i:>3}  {:<40} {:?}", labels[i], m.pass_grid(i));
    }
    eprintln!("\n==== tail after the last layer block's typical span ====");
    for i in (n - epilogue_guess.min(n))..n {
        eprintln!("  {i:>3}  {:<40} {:?}", labels[i], m.pass_grid(i));
    }

    let mut pairs: BTreeMap<(String, String), usize> = BTreeMap::new();
    for i in 0..n.saturating_sub(1) {
        *pairs
            .entry((bases[i].to_string(), bases[i + 1].to_string()))
            .or_insert(0) += 1;
    }
    let mut pairs: Vec<((String, String), usize)> = pairs.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!(
        "\n==== adjacent DEPENDENT pairs, ranked by dispatches a merge would remove ====\n\
         {EVERY_ADJACENT_PAIR_IN_ONE_COMPUTE_PASS_IS_A_DEPENDENT_EDGE_PER_WEBGPU_SPEC}"
    );
    eprintln!("{:<34} -> {:<34} {:>6} {:>10}", "producer", "consumer", "n", "us @2.05");
    for ((a, b), c) in pairs.iter().take(40) {
        eprintln!(
            "{a:<34} -> {b:<34} {c:>6} {:>10.1}",
            *c as f64 * DEPENDENT_DISPATCH_US_MEASURED_THIS_SESSION_NOT_THE_CUDA_DELTA
        );
    }

    let warm = env_usize("NV_E4B_CENSUS_WARM", 4);
    let steps = env_usize("NV_E4B_CENSUS_STEPS", 32);
    assert!(
        PROMPT.len() + warm + 3 * steps + 8 < max_seq,
        "{warm} warm + 3x{steps} rounds overruns a {max_seq}-slot kv cache"
    );
    quiet_card_or_refuse("wall");
    let (a_lo, a_med) = wall_floor_and_median(&mut m, warm, steps);
    m.set_preenc(false);
    let (b_lo, b_med) = wall_floor_and_median(&mut m, warm, steps);
    m.set_preenc(true);
    let (a2_lo, a2_med) = wall_floor_and_median(&mut m, warm, steps);
    let drift = 100.0 * (a2_lo - a_lo) / a_lo;
    eprintln!(
        "\n==== wall clock, interleaved A/B/A, warmup discarded ====\n\
         preenc on  (shipping default) : {a_lo:.3} ms floor, {a_med:.3} ms median\n\
         preenc off                    : {b_lo:.3} ms floor, {b_med:.3} ms median\n\
         preenc on, again              : {a2_lo:.3} ms floor, {a2_med:.3} ms median  (drift \
         {drift:+.2}%)"
    );
    if drift.abs() >= 5.0 {
        eprintln!(
            "  VOID: the A arm moved {drift:+.2}% between its two visits, so every wall number in \
             this run is a blend of kernel and weather. The dispatch COUNTS and the bit-identity \
             verdicts below are structural and survive; the milliseconds do not. This run will \
             fail at the end so the void cannot be read as a pass."
        );
    }

    let disp_ms = n as f64 * DEPENDENT_DISPATCH_US_MEASURED_THIS_SESSION_NOT_THE_CUDA_DELTA / 1e3;
    eprintln!(
        "\n==== the dispatch bill ====\n\
         {n} dependent dispatches x {DEPENDENT_DISPATCH_US_MEASURED_THIS_SESSION_NOT_THE_CUDA_DELTA} \
         us = {disp_ms:.3} ms of a {a_med:.3} ms token ({:.1}%)",
        100.0 * disp_ms / a_med
    );

    if env_usize("NV_E4B_CENSUS_PROF", 1) == 1 && ctx.caps.timestamp_query {
        dispatch::profile::reset();
        assert!(
            m.set_prof_mode(ProfMode::PerDispatch),
            "timestamp_query advertised but PerDispatch profiling refused"
        );
        m.reset();
        let mut t = 0u32;
        for &p in &PROMPT {
            t = m.decode_step(p).expect("prof prompt");
        }
        let prof_steps = env_usize("NV_E4B_CENSUS_PROF_STEPS", 8);
        for _ in 0..prof_steps {
            t = m.decode_step(t).expect("prof step");
        }
        let _ = t;
        m.set_prof_mode(ProfMode::Off);
        let observed = dispatch::profile::dispatches();
        eprintln!(
            "\n==== runtime confirmation: the static list is what the GPU ran ====\n{}",
            dispatch::profile::table("e4b-decode", n)
        );
        let expect = (PROMPT.len() + prof_steps) as u64 * n as u64;
        assert_eq!(
            observed, expect,
            "the profiler saw {observed} dispatches where the static pass list predicts {expect} \
             ({} steps x {n}); the census is not describing the graph that ran",
            PROMPT.len() + prof_steps
        );
    }

    if env_usize("NV_E4B_CENSUS_ARMS", 0) == 1 {
        let feed: Vec<u32> = PROMPT.iter().chain(PROMPT.iter()).copied().collect();
        let want = logits_bit_trace(&mut m, &feed);
        eprintln!(
            "\n==== fusion arms already built into gemma4_e4b_wgpu, all DEFAULT-OFF ====\n\
             each row rebuilds the whole graph with the arm on, counts its dispatches, then gates \
             bit-identity against the default arm's f32 logits over {} fed tokens before any \
             timing is quoted",
            feed.len()
        );
        eprintln!(
            "{:<46} {:>7} {:>8} {:>9} {:>14}",
            "arm", "passes", "removed", "us @2.05", "bit-identical"
        );
        for (name, vars) in ARMS {
            if !arm_selected(name) {
                continue;
            }
            for (k, v) in *vars {
                std::env::set_var(k, v);
            }
            let built = Gemma4E4bWgpu::from_loader(
                Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("config"),
                &loader,
                max_seq,
            );
            for (k, _) in *vars {
                std::env::remove_var(k);
            }
            let mut arm = match built {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("{name:<46} REFUSED TO BUILD: {e}");
                    continue;
                }
            };
            let an = arm.pass_count();
            let got = logits_bit_trace(&mut arm, &feed);
            let bits_equal = got == want;
            let first_diff = got
                .iter()
                .zip(want.iter())
                .enumerate()
                .find_map(|(s, (g, w))| {
                    g.iter()
                        .zip(w.iter())
                        .position(|(a, b)| a != b)
                        .map(|c| (s, c))
                });
            eprintln!(
                "{name:<46} {an:>7} {:>8} {:>9.1} {:>14}",
                n as i64 - an as i64,
                (n as i64 - an as i64) as f64
                    * DEPENDENT_DISPATCH_US_MEASURED_THIS_SESSION_NOT_THE_CUDA_DELTA,
                if bits_equal {
                    "yes".to_string()
                } else {
                    match first_diff {
                        Some((s, c)) => format!("NO @tok{s}/col{c}"),
                        None => "NO".to_string(),
                    }
                }
            );
            let arm_hist: BTreeMap<&str, usize> =
                (0..an).fold(BTreeMap::new(), |mut acc, i| {
                    *acc.entry(base_of(arm.pass_label(i))).or_insert(0) += 1;
                    acc
                });
            for (l, c) in &hist {
                let now = arm_hist.get(l).copied().unwrap_or(0);
                if now != *c {
                    eprintln!("      {l:<36} {c:>5} -> {now:<5}");
                }
            }
            for (l, c) in &arm_hist {
                if !hist.iter().any(|(h, _)| h == l) {
                    eprintln!("      {l:<36} {:>5} -> {c:<5}  (new)", 0);
                }
            }
            if env_usize("NV_E4B_CENSUS_ARM_PROF", 0) == 1 && ctx.caps.timestamp_query {
                dispatch::profile::reset();
                assert!(
                    arm.set_prof_mode(ProfMode::PerDispatch),
                    "timestamp_query advertised but PerDispatch profiling refused on arm {name}"
                );
                arm.reset();
                let mut t = 0u32;
                for &p in &PROMPT {
                    t = arm.decode_step(p).expect("arm prof prompt");
                }
                for _ in 0..8 {
                    t = arm.decode_step(t).expect("arm prof step");
                }
                let _ = t;
                arm.set_prof_mode(ProfMode::Off);
                eprintln!("{}", dispatch::profile::table(name, an));
            }
            if env_usize("NV_E4B_CENSUS_ARM_WALL", 0) == 1 {
                quiet_card_or_refuse(name);
                let (base1, _) = wall_floor_and_median(&mut m, warm, steps);
                let (armlo, armmed) = wall_floor_and_median(&mut arm, warm, steps);
                let (base2, basemed) = wall_floor_and_median(&mut m, warm, steps);
                let d = 100.0 * (base2 - base1) / base1;
                eprintln!(
                    "      interleaved A/B/A: default {:.3} ms floor ({basemed:.3} med, drift \
                     {d:+.2}%) vs arm {armlo:.3} ms floor ({armmed:.3} med) -> {:+.2}%{}",
                    base1.min(base2),
                    100.0 * (armlo - base1.min(base2)) / base1.min(base2),
                    if bits_equal {
                        ""
                    } else {
                        "   NOT BIT-IDENTICAL: this timing is fp-reorder class, quote it only \
                         through docs/book/08.2-quality-verdicts.md"
                    }
                );
            }
        }
    }

    assert_eq!(
        hist.iter()
            .find(|(l, _)| *l == "flash_stage1")
            .map(|(_, c)| *c),
        Some(n_layers),
        "flash stage 1 does not run once per layer; the segmentation above is fiction"
    );
    assert!(
        (300..=900).contains(&n),
        "decode dispatch count moved to {n}; re-derive the fusion ledger before trusting it"
    );
    assert!(
        drift.abs() < 5.0,
        "the A arm moved {drift:+.2}% between its two visits; the box moved under the instrument \
         and no millisecond printed above may be published"
    );
}
