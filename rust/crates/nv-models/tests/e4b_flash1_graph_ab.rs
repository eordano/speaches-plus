#![cfg(feature = "wgpu")]

mod common;
use common::env_usize;
use common::med;
use common::snapshot_dir;
use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu;
use std::time::Instant;

fn lo_of(xs: &[f64]) -> f64 {
    xs.iter().cloned().fold(f64::INFINITY, f64::min)
}

const PROMPT: &[u32] = &[
    2, 106, 1645, 108, 3048, 573, 4926, 576, 7127, 235265, 107, 108,
];

fn padded_prompt(ctx: usize) -> Vec<u32> {
    let mut v = PROMPT.to_vec();
    let mut i = 0usize;
    while v.len() < ctx {
        v.push(FILLER[i % FILLER.len()]);
        i += 1;
    }
    v
}

const FILLER: &[u32] = &[
    573, 4926, 576, 7127, 1671, 611, 573, 2149, 576, 573, 3821, 235269,
];

fn wall(m: &mut Gemma4E4bWgpu, warm: usize, steps: usize) -> (f64, f64) {
    let ctx = env_usize("NV_E4B_FLASH1_CTX", 0);
    m.reset();
    let mut t = if ctx > PROMPT.len() {
        m.prefill_prompt(&padded_prompt(ctx)).expect("prefill")
    } else {
        m.prefill_prompt(PROMPT).expect("prefill")
    };
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

fn greedy_stream(m: &mut Gemma4E4bWgpu, n: usize) -> (Vec<u32>, u64) {
    m.reset();
    let mut t = m.prefill_prompt(PROMPT).expect("prefill");
    let mut out = Vec::with_capacity(n);
    out.push(t);
    for _ in 1..n {
        t = m.decode_step(t).expect("step");
        out.push(t);
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for t in &out {
        h ^= *t as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    assert!(
        out.iter().collect::<std::collections::HashSet<_>>().len() > 2,
        "the greedy stream is {out:?} -- a constant stream compares constants to constants and \
         would agree across any mutation"
    );
    (out, h)
}

fn probe(m: &Gemma4E4bWgpu, n: usize, reps: usize) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let s = Instant::now();
        m.probe_prefix(n).expect("prefix");
        best = best.min(s.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn replicate(
    m: &mut Gemma4E4bWgpu,
    label: &str,
    lo: usize,
    hi: usize,
    reps: usize,
) -> (f64, f64, f64, usize) {
    let n = m.probe_append_class(label, None, lo);
    assert!(n > 0, "no dispatches labelled {label} reached the graph");
    let per_copy = n / lo;
    let a = probe(m, m.pass_count(), reps);
    let a_null = probe(m, m.pass_count(), reps);
    m.probe_append_class(label, None, hi);
    let b = probe(m, m.pass_count(), reps);
    m.probe_append_class(label, None, lo);
    let a2 = probe(m, m.pass_count(), reps);
    m.probe_append_clear();
    let base = 0.5 * (a + a2);
    let d = ((hi - lo) * per_copy) as f64;
    (
        (b - base) / d * 1e3,
        (a_null - a) / (lo * per_copy) as f64 * 1e3,
        100.0 * (a2 - a) / a,
        per_copy,
    )
}

#[test]
#[ignore = "loads the E4B QAT checkpoint; set NV_E4B_FLASH1_AB=1"]
fn e4b_flash1_graph_arm() {
    assert_eq!(
        std::env::var("NV_E4B_FLASH1_AB").ok().as_deref(),
        Some("1"),
        "set NV_E4B_FLASH1_AB=1 -- a silent skip here would report a pass"
    );
    let dir = snapshot_dir();
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("config");
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("safetensors");
    let max_seq = env_usize("NV_E4B_FLASH1_SEQ", 640);
    let t0 = Instant::now();
    let mut m = Gemma4E4bWgpu::from_loader(config, &loader, max_seq).expect("build graph");

    let (entry, hds) = m.flash1_route();
    let want_sg = std::env::var("NV_E4B_WGPU_FLASH1_SG").ok().as_deref() != Some("0");
    let want_hd = std::env::var("NV_E4B_WGPU_FLASH1_HD").ok().as_deref() != Some("0");
    assert_eq!(
        entry.trim_end_matches("_sd").ends_with("sg_fp8"),
        want_sg,
        "flash1 entry {entry} does not match NV_E4B_WGPU_FLASH1_SG"
    );
    assert_eq!(
        hds.len() > 1,
        want_hd,
        "flash1 head_dim set {hds:?} does not match NV_E4B_WGPU_FLASH1_HD"
    );
    if want_hd {
        assert_eq!(
            hds.iter().find(|(hd, _)| *hd == 256).map(|(_, n)| *n),
            Some(35),
            "head_dim 256 pipeline serves the wrong layer count: {hds:?}"
        );
    }
    eprintln!(
        "\narm: entry {entry}, head_dim pipelines {hds:?}, {} passes/token, built in {:.1}s",
        m.pass_count(),
        t0.elapsed().as_secs_f64()
    );

    let ctx = env_usize("NV_E4B_FLASH1_CTX", 0).max(PROMPT.len());
    eprintln!("context under measurement: {ctx} prompt tokens + warm + steps");
    let g = env_usize("NV_E4B_FLASH1_GREEDY", 24);
    let (stream, hash) = greedy_stream(&mut m, g);
    for rep in 1..env_usize("NV_E4B_FLASH1_GREEDY_REPS", 3) {
        let (s2, h2) = greedy_stream(&mut m, g);
        assert_eq!(
            h2, hash,
            "greedy stream is not self-reproducing in one process: rep 0 {stream:?} vs rep {rep}              {s2:?} -- the arm cannot be compared to any other arm until this is explained"
        );
    }
    eprintln!(
        "greedy stream fnv1a {hash:#018x} (self-reproducing)  first 12 {:?}",
        &stream[..12]
    );
    m.reset();

    let warm = env_usize("NV_E4B_FLASH1_WARM", 6);
    let steps = env_usize("NV_E4B_FLASH1_STEPS", 32);
    assert!(
        ctx + warm + steps + 8 < max_seq,
        "{ctx}-token prompt + {warm} warm + {steps} steps overruns a {max_seq}-slot kv cache"
    );
    let (pre_lo, pre_med) = wall(&mut m, warm, steps);
    m.set_preenc(false);
    let (lo, md) = wall(&mut m, warm, steps);
    let (lo2, md2) = wall(&mut m, warm, steps);
    eprintln!(
        "wall preenc on : {pre_lo:.3} ms floor ({:.1} tok/s), {pre_med:.3} median",
        1e3 / pre_lo
    );
    eprintln!(
        "wall preenc off: {lo:.3} ms floor ({:.1} tok/s), {md:.3} median",
        1e3 / lo
    );
    eprintln!(
        "wall A-prime   : {lo2:.3} ms floor ({:.1} tok/s), {md2:.3} median  (drift {:+.2}%)",
        1e3 / lo2,
        100.0 * (lo2 - lo) / lo
    );

    let pos = m.current_pos();
    m.probe_at(0, pos.min(max_seq - 1)).expect("probe_at");
    for label in ["flash_stage1", "flash_stage2"] {
        let (us, null_us, drift, per_copy) = replicate(&mut m, label, 1, 6, 10);
        eprintln!(
            "{label:>14}: {per_copy:>3} dispatches, {us:>7.2} us each  (null {null_us:+.2} us, \
             A-prime drift {drift:+.2}%)  -> {:.3} ms/token",
            us * per_copy as f64 / 1e3
        );
    }
}
