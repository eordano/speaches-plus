#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::WgpuContext;
use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu;

fn snapshot_dir() -> std::path::PathBuf {
    match std::env::var("NV_E4B_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => {
            let home = std::env::var("HOME").unwrap();
            let base = std::path::PathBuf::from(home).join(
                ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots",
            );
            std::fs::read_dir(&base)
                .expect("hub snapshot dir")
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.join("config.json").exists())
                .expect("no snapshot with config.json")
        }
    }
}

fn gate() {
    assert_eq!(
        std::env::var("NV_E4B_HOST_SYNC").ok().as_deref(),
        Some("1"),
        "set NV_E4B_HOST_SYNC=1 -- this suite loads the real checkpoint and measures the GPU"
    );
}

fn load(max_seq: usize) -> (&'static WgpuContext, Gemma4E4bWgpu) {
    let ctx = WgpuContext::shared().expect("wgpu adapter");
    eprintln!("adapter: {}", ctx.summary());
    let dir = snapshot_dir();
    eprintln!("checkpoint: {}", dir.display());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let t0 = std::time::Instant::now();
    let m = Gemma4E4bWgpu::from_loader(config, &loader, max_seq).unwrap();
    eprintln!(
        "loaded in {:.1}s -- {} decode passes/token, {:.3} GB weights/token, staged_read={}",
        t0.elapsed().as_secs_f64(),
        m.pass_count(),
        m.weight_bytes_per_token() as f64 / 1e9,
        m.staged_read()
    );
    let mut hist: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for i in 0..m.pass_count() {
        *hist.entry(m.pass_label(i)).or_insert(0) += 1;
    }
    let mut rows: Vec<_> = hist.into_iter().collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    eprintln!("decode graph pipelines:");
    for (label, n) in rows {
        eprintln!("  {label:<32} x{n}");
    }
    (ctx, m)
}

const PROMPT: [u32; 8] = [2, 818, 3029, 529, 6081, 603, 563, 1596];

fn seed(m: &mut Gemma4E4bWgpu) -> u32 {
    m.reset();
    let mut last = 0u32;
    for &t in &PROMPT {
        last = m.decode_step(t).unwrap();
    }
    last
}

fn run_steps(m: &mut Gemma4E4bWgpu, start: u32, n: usize) -> (Vec<u32>, f64) {
    let mut out = Vec::with_capacity(n);
    let mut t = start;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        t = m.decode_step(t).unwrap();
        out.push(t);
    }
    (out, t0.elapsed().as_secs_f64() * 1e3 / n as f64)
}

fn run_chain(m: &mut Gemma4E4bWgpu, start: u32, n: usize, k: usize) -> (Vec<u32>, f64) {
    let mut out = Vec::with_capacity(n);
    let mut t = start;
    let t0 = std::time::Instant::now();
    while out.len() < n {
        let want = k.min(n - out.len());
        let b = m.decode_chain(t, want).unwrap();
        t = *b.last().unwrap();
        out.extend(b);
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / out.len() as f64;
    (out, ms)
}

fn run_pipe(m: &mut Gemma4E4bWgpu, start: u32, n: usize) -> (Vec<u32>, f64) {
    let mut out = Vec::with_capacity(n);
    let t0 = std::time::Instant::now();
    let mut t = m.decode_step_pipelined(Some(start)).unwrap();
    out.push(t);
    while out.len() < n {
        t = m.decode_step_pipelined(None).unwrap();
        out.push(t);
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / out.len() as f64;
    let dropped = m.decode_pipe_abort().unwrap();
    assert!(dropped <= 1, "pipe left {dropped} steps in flight");
    let _ = t;
    (out, ms)
}

fn fit(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let h = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    ((sy - h * sx) / n, h)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n.is_multiple_of(2) {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

#[test]
#[ignore]
fn e4b_host_term_interleaved() {
    gate();
    let rounds: usize = std::env::var("NV_E4B_HS_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let block: usize = std::env::var("NV_E4B_HS_BLOCK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let (_ctx, mut m) = load(2048);

    let s = seed(&mut m);
    let _ = run_steps(&mut m, s, 24);

    let arms: [(&str, bool, bool, usize); 6] = [
        ("plain  wait k=1", false, false, 1),
        ("staged wait k=1", true, false, 1),
        ("staged spin k=1", true, true, 1),
        ("pipe   depth1  ", true, true, 0),
        ("plain  wait k=8", false, false, 8),
        ("staged spin k=8", true, true, 8),
    ];
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut base: Option<Vec<u32>> = None;
    for _ in 0..rounds {
        for (i, (_, staged, spin, k)) in arms.iter().enumerate() {
            m.set_staged_read(*staged);
            m.set_spin_read(*spin);
            let s = seed(&mut m);
            let (toks, ms) = match *k {
                0 => run_pipe(&mut m, s, block),
                1 => run_steps(&mut m, s, block),
                k => run_chain(&mut m, s, block, k),
            };
            match &base {
                None => base = Some(toks),
                Some(b) => assert_eq!(&toks, b, "arm {i} diverged from the reference stream"),
            }
            samples[i].push(ms);
        }
    }
    eprintln!(
        "BIT-EXACT: {} arm-blocks produced one identical stream",
        rounds * arms.len()
    );
    let mut med = Vec::new();
    for (i, (tag, _, _, _)) in arms.iter().enumerate() {
        let mut v = samples[i].clone();
        let mm = median(&mut v);
        eprintln!(
            "{tag}: median {mm:.4} ms/tok  min {:.4}  max {:.4}  over {rounds} rounds x {block} tokens",
            v[0],
            v[v.len() - 1]
        );
        med.push(mm);
    }
    eprintln!(
        "== H (plain wait k=1 minus k=8) = {:.4} ms/tok ==",
        (med[0] - med[4]) / (1.0 - 1.0 / 8.0)
    );

    eprintln!("-- paired within-round deltas vs the same round's plain-wait k=1 arm --");
    for (i, (tag, _, _, _)) in arms.iter().enumerate() {
        let ref_i = 0;
        if i == ref_i {
            continue;
        }
        let mut d: Vec<f64> = (0..rounds)
            .map(|r| samples[i][r] - samples[ref_i][r])
            .collect();
        let dm = median(&mut d);
        eprintln!(
            "{tag}: paired median {dm:+.4} ms/tok  ({:+.2}%)  p25 {:+.4}  p75 {:+.4}",
            100.0 * dm / med[ref_i],
            d[rounds / 4],
            d[rounds * 3 / 4]
        );
    }
}

#[test]
#[ignore]
fn e4b_staged_readback_ab() {
    gate();
    let n: usize = std::env::var("NV_E4B_HS_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(160);
    let (_ctx, mut m) = load(1024);

    let s = seed(&mut m);
    let _ = run_steps(&mut m, s, 24);

    let mut rows: Vec<(bool, usize, Vec<u32>, f64)> = Vec::new();
    for rep in 0..2 {
        for staged in [true, false] {
            m.set_staged_read(staged);
            assert_eq!(m.staged_read(), staged);
            let s = seed(&mut m);
            let (toks, ms) = run_steps(&mut m, s, n);
            eprintln!(
                "rep{rep} staged={staged:<5} decode_step {ms:.4} ms/tok  first8={:?}",
                &toks[..8]
            );
            rows.push((staged, rep, toks, ms));
        }
    }

    let base = rows[0].2.clone();
    for (staged, rep, toks, _) in &rows {
        assert_eq!(
            *toks, base,
            "staged={staged} rep={rep} diverged from the staged reference stream"
        );
    }
    eprintln!("BIT-EXACT: all 4 arms produced the identical {n}-token stream");

    let on: Vec<f64> = rows.iter().filter(|r| r.0).map(|r| r.3).collect();
    let off: Vec<f64> = rows.iter().filter(|r| !r.0).map(|r| r.3).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    eprintln!(
        "== staged readback: ON {:.4} ms/tok  OFF {:.4} ms/tok  delta {:+.4} ms/tok ({:+.2}%) ==",
        mean(&on),
        mean(&off),
        mean(&on) - mean(&off),
        100.0 * (mean(&on) - mean(&off)) / mean(&off)
    );
}

#[test]
#[ignore]
fn e4b_chain_k_sweep() {
    gate();
    let n: usize = std::env::var("NV_E4B_HS_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(160);
    let (_ctx, mut m) = load(1024);
    let s = seed(&mut m);
    let _ = run_steps(&mut m, s, 24);

    let mut base: Option<Vec<u32>> = None;
    for staged in [true, false] {
        m.set_staged_read(staged);
        let mut pts: Vec<(f64, f64)> = Vec::new();
        for k in [1usize, 2, 4, 8] {
            let s = seed(&mut m);
            let (toks, ms) = if k == 1 {
                run_steps(&mut m, s, n)
            } else {
                run_chain(&mut m, s, n, k)
            };
            match &base {
                None => base = Some(toks.clone()),
                Some(b) => assert_eq!(
                    toks, *b,
                    "staged={staged} k={k} diverged from the k=1 reference stream"
                ),
            }
            eprintln!("staged={staged:<5} k={k}: {ms:.4} ms/tok");
            pts.push((1.0 / k as f64, ms));
        }
        let (g, h) = fit(&pts);
        eprintln!(
            "staged={staged:<5} fit decode(k) = G + H/k -> G {g:.4} ms  H {h:.4} ms  residuals {:?}",
            pts.iter()
                .map(|(x, y)| format!("{:+.4}", y - (g + h * x)))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
#[ignore]
fn e4b_pipe_matches_stepped_bitwise() {
    gate();
    let n: usize = std::env::var("NV_E4B_HS_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let (_ctx, mut m) = load(1024);

    let s = seed(&mut m);
    let (reference, _) = run_steps(&mut m, s, n);

    let s = seed(&mut m);
    let (piped, _) = run_pipe(&mut m, s, n);
    assert_eq!(piped, reference, "all-pipe stream diverged");

    let s = seed(&mut m);
    let mut mixed = Vec::with_capacity(n);
    let mut t = s;
    while mixed.len() < n {
        let want = (n - mixed.len()).min(5);
        t = m.decode_step_pipelined(Some(t)).unwrap();
        mixed.push(t);
        for _ in 1..want {
            t = m.decode_step_pipelined(None).unwrap();
            mixed.push(t);
        }
        let dropped = m.decode_pipe_abort().unwrap();
        assert_eq!(
            dropped, 1,
            "abort should discard exactly the lookahead step"
        );
        assert_eq!(m.decode_pipe_inflight(), 0);
        if mixed.len() < n {
            t = m.decode_step(t).unwrap();
            mixed.push(t);
        }
    }
    mixed.truncate(n);
    assert_eq!(mixed, reference, "pipe/step interleaved stream diverged");
    eprintln!("BIT-EXACT: pipe, pipe+abort+step, and pure decode_step agree over {n} tokens");
}

const PREFILL_RATE_ROUNDS: usize = 3;
const PREFILL_ATTR_CHUNKS: usize = 8;

#[test]
#[ignore]
fn e4b_prefill_rate_2048_and_attribution() {
    gate();
    let total: usize = std::env::var("NV_E4B_PF_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);
    let (ctx, mut m) = load(total + 128);
    let cm = m.prefill_chunk_len();
    assert!(
        cm >= 2,
        "prefill pass list disabled -- a rate here would measure decode_step, not prefill"
    );
    eprintln!(
        "prefill chunk m={cm}, {} passes/chunk",
        m.prefill_pass_count()
    );
    let toks: Vec<u32> = (0..total).map(|i| PROMPT[i % PROMPT.len()]).collect();

    m.reset();
    m.prefill_tokens(&toks[..(2 * cm).min(total)]).unwrap();
    ctx.poll_blocking().unwrap();

    let mut rates: Vec<f64> = Vec::new();
    for round in 0..PREFILL_RATE_ROUNDS {
        m.reset();
        let t0 = std::time::Instant::now();
        let done = m.prefill_tokens(&toks).unwrap();
        ctx.poll_blocking().unwrap();
        let dt = t0.elapsed().as_secs_f64();
        assert!(
            done + cm > total,
            "prefill_tokens consumed {done} of {total}; tail fell back further than one chunk"
        );
        let rate = done as f64 / dt;
        eprintln!("round{round}: {done} tokens in {dt:.3}s = {rate:.1} tok/s");
        rates.push(rate);
    }
    eprintln!(
        "== e4b prefill wall rate @ {total}: median {:.1} tok/s over {PREFILL_RATE_ROUNDS} rounds (chunk m={cm}) ==",
        median(&mut rates)
    );

    m.reset();
    let mid = (total / 2 / cm) * cm;
    m.prefill_tokens(&toks[..mid]).unwrap();
    ctx.poll_blocking().unwrap();
    let mut agg: std::collections::BTreeMap<String, (usize, f64)> = Default::default();
    let mut chunk_ms: Vec<f64> = Vec::new();
    for _ in 0..PREFILL_ATTR_CHUNKS {
        let rows = match m.prefill_chunk_profiled(&vec![PROMPT[0]; cm]) {
            Ok(rows) => rows,
            Err(e) => {
                eprintln!("attribution skipped: {e}");
                return;
            }
        };
        let mut tot = 0.0;
        for (label, ns) in rows {
            let e = agg.entry(label).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ns;
            tot += ns;
        }
        chunk_ms.push(tot / 1e6);
    }
    let gpu_ms = median(&mut chunk_ms);
    eprintln!(
        "GPU pass total per chunk @depth~{mid}: median {gpu_ms:.3} ms -> dispatch-only {:.1} tok/s",
        cm as f64 * 1e3 / gpu_ms
    );
    let grand: f64 = agg.values().map(|v| v.1).sum();
    let mut rows: Vec<(String, usize, f64)> =
        agg.into_iter().map(|(l, (n, ns))| (l, n, ns)).collect();
    rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    eprintln!("-- prefill attribution over {PREFILL_ATTR_CHUNKS} chunks --");
    for (label, n, ns) in rows {
        eprintln!(
            "{label:<32} x{:<5} {:>9.3} ms/chunk  {:>5.1}%",
            n / PREFILL_ATTR_CHUNKS,
            ns / 1e6 / PREFILL_ATTR_CHUNKS as f64,
            100.0 * ns / grand
        );
    }
}

#[test]
#[ignore]
fn e4b_deep_kv_decode_rate() {
    gate();
    let (_ctx, mut m) = load(8192);
    let t = seed(&mut m);
    let splits = nv_kernels::wgpu_backend::kernels::flash_decode::splits_env();
    let warm = run_steps(&mut m, t, 8);
    let _ = warm;
    for &depth in &[512usize, 2048, 4096, 8000] {
        m.restore_pos(depth).unwrap();
        let (_toks, ms) = run_steps(&mut m, t, 64);
        m.restore_pos(depth).unwrap();
        let (_toks2, ms2) = run_steps(&mut m, t, 64);
        eprintln!(
            "splits={splits} depth={depth}: {:.4} ms/tok (repeat {:.4})",
            ms.min(ms2),
            ms.max(ms2)
        );
    }
}

#[test]
#[ignore = "timing instrument; set NV_E4B_HOST_SYNC=1"]
fn e4b_pass_total_gpu_ns() {
    gate();
    let n: usize = std::env::var("NV_E4B_HS_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let (_ctx, mut m) = load(1024);
    eprintln!(
        "FUSE_HEAD={:?} FUSE_ATTN={:?} passes/token={}",
        std::env::var("NV_E4B_WGPU_FUSE_HEAD").ok(),
        std::env::var("NV_E4B_WGPU_FUSE_ATTN").ok(),
        m.pass_count()
    );
    let s = seed(&mut m);
    let _ = run_steps(&mut m, s, 16);
    assert!(
        m.set_prof_mode(nv_models::gemma4_e4b_wgpu::ProfMode::PassTotal),
        "no timestamp queries"
    );
    let mut last = seed(&mut m);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        last = m.decode_step(last).unwrap();
        v.push(m.prof_pass_total_ns() / 1e6);
    }
    let mut w = v.clone();
    let med = median(&mut w);
    eprintln!(
        "GPU pass total: median {med:.5} ms  p10 {:.5}  p25 {:.5}  p75 {:.5}  min {:.5}  over {n} steps",
        w[n / 10],
        w[n / 4],
        w[n * 3 / 4],
        w[0]
    );
    m.set_prof_mode(nv_models::gemma4_e4b_wgpu::ProfMode::Off);

    let rounds = 24usize;
    let block = 32usize;
    let mut b: Vec<f64> = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let s = seed(&mut m);
        let (_, ms) = run_pipe(&mut m, s, block);
        b.push(ms);
    }
    let bm = median(&mut b);
    eprintln!(
        "pipe wall: median {bm:.5} ms/tok  p25 {:.5}  p75 {:.5}  min {:.5}  over {rounds}x{block}",
        b[rounds / 4],
        b[rounds * 3 / 4],
        b[0]
    );
}

#[test]
#[ignore]
fn e4b_spec_round_cost() {
    gate();
    let reps: usize = std::env::var("NV_E4B_HS_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    let (_ctx, mut m) = load(1024);
    let rows = m.verify_max_rows();
    eprintln!("verify_max_rows = {rows}");
    assert!(rows >= 2, "verify path unavailable");

    let s = seed(&mut m);
    let _ = run_steps(&mut m, s, 24);

    for staged in [true, false] {
        m.set_staged_read(staged);
        let mut last = seed(&mut m);
        for _ in 0..8 {
            last = m.decode_step(last).unwrap();
        }

        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            last = m.decode_step(last).unwrap();
        }
        let d = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        eprintln!("staged={staged:<5} d (decode_step)          {d:.4} ms");

        for mb in [2usize, 3, 5, 9] {
            if mb > rows {
                continue;
            }
            let pos = m.current_pos();
            let batch: Vec<u32> = std::iter::repeat_n(last, mb).collect();
            m.verify_chain(&batch).unwrap();
            m.truncate_to(pos).unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                let _ = m.verify_chain(&batch).unwrap();
                m.truncate_to(pos).unwrap();
            }
            let v = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
            eprintln!(
                "staged={staged:<5} v(mb={mb})                  {v:.4} ms  tau* = {:.3}",
                v / d
            );
        }
    }
}
