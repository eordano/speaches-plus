#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::kernels::gemv_w4a16::ScaleGrain;
use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu;
use std::path::PathBuf;

const HUB: &str = ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots";

fn ckpt_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NV_E4B_W4_DIR") {
        return PathBuf::from(d);
    }
    let base = PathBuf::from(std::env::var("HOME").expect("HOME")).join(HUB);
    std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("no pack-quantized E4B snapshot dir {}: {e}", base.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").exists())
        .expect("no snapshot with config.json")
}

struct Arm {
    tag: String,
    grain: ScaleGrain,
    census: (usize, usize, usize),
    ms: Vec<f64>,
}

impl Arm {
    fn best(&self) -> f64 {
        self.ms.iter().copied().fold(f64::INFINITY, f64::min)
    }
    fn median(&self) -> f64 {
        let mut v = self.ms.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }
}

fn env_var() -> &'static str {
    "NV_E4B_W4_GRAIN"
}

fn build(tag: &str, grain_env: &str, dir: &std::path::Path, max_seq: usize) -> Gemma4E4bWgpu {
    std::env::set_var(env_var(), grain_env);
    build_here(tag, dir, max_seq)
}

fn build_here(tag: &str, dir: &std::path::Path, max_seq: usize) -> Gemma4E4bWgpu {
    let cfg = Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("parse config");
    let loader =
        nv_weights::WeightLoader::open_dir(dir, &candle_core::Device::Cpu).expect("open weights");
    assert!(
        loader.has("model.language_model.layers.0.mlp.down_proj.weight_packed"),
        "{tag}: this proof needs a pack-quantized checkpoint; that one is bf16"
    );
    let t0 = std::time::Instant::now();
    let m = Gemma4E4bWgpu::from_loader(cfg, &loader, max_seq).expect("from_loader");
    eprintln!(
        "[{tag}] from_loader {:.1}s, grain {:?}, route census {:?}, {} passes/token",
        t0.elapsed().as_secs_f64(),
        m.w4_scale_grain(),
        m.w4_route_census(),
        m.pass_count()
    );
    m
}

fn run(m: &mut Gemma4E4bWgpu, prompt: &[u32], n_new: usize) -> (Vec<u32>, Vec<Vec<u32>>, f64) {
    m.reset();
    let mut next = 0u32;
    for t in prompt {
        next = m.decode_step(*t).expect("prefill step");
    }
    let mut toks = Vec::with_capacity(n_new);
    let mut logits = Vec::with_capacity(n_new);
    let t0 = std::time::Instant::now();
    for _ in 0..n_new {
        toks.push(next);
        let (t, l) = m.decode_step_logits(next).expect("decode step");
        logits.push(l.iter().map(|v| v.to_bits()).collect());
        next = t;
    }
    let ms = 1000.0 * t0.elapsed().as_secs_f64() / n_new as f64;
    (toks, logits, ms)
}

fn warm(m: &mut Gemma4E4bWgpu, prompt: &[u32], n_new: usize) {
    timed(m, prompt, n_new);
}

fn timed(m: &mut Gemma4E4bWgpu, prompt: &[u32], n_new: usize) -> f64 {
    m.reset();
    let mut next = 0u32;
    for t in prompt {
        next = m.decode_step(*t).expect("prefill step");
    }
    let t0 = std::time::Instant::now();
    for _ in 0..n_new {
        next = m.decode_step(next).expect("decode step");
    }
    1000.0 * t0.elapsed().as_secs_f64() / n_new as f64
}

fn diff_words(a: &[Vec<u32>], b: &[Vec<u32>]) -> (usize, usize) {
    let mut differ = 0usize;
    let mut total = 0usize;
    for (ra, rb) in a.iter().zip(b.iter()) {
        assert_eq!(ra.len(), rb.len(), "logit row width moved between arms");
        total += ra.len();
        differ += ra.iter().zip(rb.iter()).filter(|(x, y)| x != y).count();
    }
    (differ, total)
}

#[test]
#[ignore]
fn fixed_shift_grain_reaches_from_loader_and_moves_no_bit() {
    let dir = ckpt_dir();
    eprintln!("checkpoint: {}", dir.display());
    let max_seq: usize = 512;
    let n_new: usize = std::env::var("NV_E4B_GRAIN_NEW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);
    let reps: usize = std::env::var("NV_E4B_GRAIN_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let bos = tok.token_to_id("<bos>").expect("<bos>");
    let mut prompt = vec![bos];
    prompt.extend(
        tok.encode("The capital of France is", false)
            .expect("encode")
            .get_ids()
            .iter()
            .copied(),
    );

    let mut ctl_a = build("ctl-a", "0", &dir, max_seq);
    let mut ctl_b = build("ctl-b", "0", &dir, max_seq);
    let mut fixed = build("fixed", "1", &dir, max_seq);
    std::env::remove_var(env_var());

    assert_eq!(
        ctl_a.w4_scale_grain(),
        ScaleGrain::Ge32,
        "NV_E4B_W4_GRAIN=0 must pin the runtime-divide grain"
    );
    assert_eq!(
        fixed.w4_scale_grain(),
        ScaleGrain::Ge32Fixed(0),
        "the pack-quantized E4B checkpoint is uniformly gs=32; the default must fold it to a shift"
    );
    let census = fixed.w4_route_census();
    assert_eq!(
        census,
        ctl_a.w4_route_census(),
        "the grain must not move the route"
    );
    assert!(
        census.2 > 0 && census.0 == 0 && census.1 == 0,
        "every w4 projection must take the sg body or this measures nothing: {census:?}"
    );

    let (ctl_toks, ctl_logits, _) = run(&mut ctl_a, &prompt, n_new);
    let (fix_toks, fix_logits, _) = run(&mut fixed, &prompt, n_new);
    let text = tok.decode(&fix_toks, false).expect("decode text");
    eprintln!("fixed-grain continuation: {text:?}");
    assert!(
        text.to_lowercase().contains("paris"),
        "greedy continuation should name Paris, got {text:?}"
    );

    let fast =
        nv_kernels::wgpu_backend::kernels::gemv_w4a16::sg_pk_source_grain(ScaleGrain::Ge32Fixed(0));
    let slow = nv_kernels::wgpu_backend::kernels::gemv_w4a16::sg_pk_source_grain(ScaleGrain::Ge32);
    assert!(
        slow.contains("/ sgp_params.gs") && !fast.contains("/ sgp_params.gs"),
        "both grains generate the same inner loop; the flip is textual only"
    );

    let (differ, total) = diff_words(&ctl_logits, &fix_logits);
    eprintln!(
        "bit-exactness: {differ}/{total} logit words differ over {n_new} steps ({} vocab)",
        ctl_logits[0].len()
    );
    assert_eq!(fix_toks, ctl_toks, "the grain flip moved a sampled token");
    assert_eq!(differ, 0, "the grain flip moved a logit bit");

    let mut arms = [
        Arm {
            tag: "ctl-a Ge32".into(),
            grain: ctl_a.w4_scale_grain(),
            census: ctl_a.w4_route_census(),
            ms: Vec::new(),
        },
        Arm {
            tag: "ctl-b Ge32".into(),
            grain: ctl_b.w4_scale_grain(),
            census: ctl_b.w4_route_census(),
            ms: Vec::new(),
        },
        Arm {
            tag: "fixed Ge32Fixed(0)".into(),
            grain: fixed.w4_scale_grain(),
            census: fixed.w4_route_census(),
            ms: Vec::new(),
        },
    ];
    for m in [&mut ctl_a, &mut ctl_b, &mut fixed] {
        warm(m, &prompt, n_new);
    }
    for r in 0..reps {
        for (i, m) in [&mut ctl_a, &mut ctl_b, &mut fixed].into_iter().enumerate() {
            let ms = timed(m, &prompt, n_new);
            eprintln!("rep {r} {:<20} {ms:.3} ms/tok", arms[i].tag);
            arms[i].ms.push(ms);
        }
    }
    for a in &arms {
        eprintln!(
            "{:<20} grain {:?} census {:?} | best {:.3} median {:.3} ms/tok",
            a.tag,
            a.grain,
            a.census,
            a.best(),
            a.median()
        );
    }
    let null = 100.0 * (arms[1].best() - arms[0].best()) / arms[0].best();
    let win = 100.0 * (arms[2].best() - arms[0].best()) / arms[0].best();
    eprintln!(
        "null control (Ge32 vs Ge32) {null:+.2}% | fixed-shift vs Ge32 {win:+.2}% (best of {reps}, {n_new} tokens, one process)"
    );
    assert!(
        win < null,
        "the fixed-shift grain did not beat the null control: {win:+.2}% vs {null:+.2}%"
    );
}

#[test]
#[ignore]
fn aspect_ratio_route_still_loses_at_grain_parity() {
    let dir = ckpt_dir();
    eprintln!("checkpoint: {}", dir.display());
    let max_seq: usize = 512;
    let n_new: usize = std::env::var("NV_E4B_GRAIN_NEW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);
    let reps: usize = std::env::var("NV_E4B_GRAIN_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let bos = tok.token_to_id("<bos>").expect("<bos>");
    let mut prompt = vec![bos];
    prompt.extend(
        tok.encode("The capital of France is", false)
            .expect("encode")
            .get_ids()
            .iter()
            .copied(),
    );

    std::env::remove_var("NV_E4B_WGPU_W4_ROUTE");
    let mut ctl_a = build_here("ctl-a cap-route", &dir, max_seq);
    let mut ctl_b = build_here("ctl-b cap-route", &dir, max_seq);
    std::env::set_var("NV_E4B_WGPU_W4_ROUTE", "1");
    let mut aspect = build_here("aspect-route", &dir, max_seq);
    std::env::remove_var("NV_E4B_WGPU_W4_ROUTE");

    let cap_census = ctl_a.w4_route_census();
    let asp_census = aspect.w4_route_census();
    assert_eq!(cap_census, ctl_b.w4_route_census());
    assert_ne!(
        cap_census, asp_census,
        "NV_E4B_WGPU_W4_ROUTE=1 did not move a single projection off the sg body; \
         this arm measures nothing"
    );
    eprintln!("census capability {cap_census:?} vs aspect-ratio {asp_census:?}");

    let (cap_toks, cap_logits, _) = run(&mut ctl_a, &prompt, n_new);
    let (asp_toks, asp_logits, _) = run(&mut aspect, &prompt, n_new);
    let (differ, total) = diff_words(&cap_logits, &asp_logits);
    eprintln!(
        "aspect-ratio route vs capability route: {differ}/{total} logit words differ, tokens {}",
        if cap_toks == asp_toks {
            "equal"
        } else {
            "DIFFER"
        }
    );

    let mut ms = [Vec::new(), Vec::new(), Vec::new()];
    let tags = ["ctl-a cap", "ctl-b cap", "aspect"];
    for m in [&mut ctl_a, &mut ctl_b, &mut aspect] {
        warm(m, &prompt, n_new);
    }
    for r in 0..reps {
        for (i, m) in [&mut ctl_a, &mut ctl_b, &mut aspect]
            .into_iter()
            .enumerate()
        {
            let t = timed(m, &prompt, n_new);
            eprintln!("rep {r} {:<12} {t:.3} ms/tok", tags[i]);
            ms[i].push(t);
        }
    }
    let best = |v: &Vec<f64>| v.iter().copied().fold(f64::INFINITY, f64::min);
    let null = 100.0 * (best(&ms[1]) - best(&ms[0])) / best(&ms[0]);
    let asp = 100.0 * (best(&ms[2]) - best(&ms[0])) / best(&ms[0]);
    eprintln!(
        "null control {null:+.2}% | aspect-ratio route {asp:+.2}% (best of {reps}, {n_new} tokens, one process)"
    );
    assert!(
        asp > null,
        "the aspect-ratio route beat the capability route at grain parity ({asp:+.2}% vs null {null:+.2}%) \
         -- the falsified predicate would have to be re-examined before it is deleted"
    );
}
