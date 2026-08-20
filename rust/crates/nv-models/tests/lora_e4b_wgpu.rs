#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use common::TINY_E4B_CONFIG;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use half::bf16;
use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_e4b_wgpu::{
    lora_site_probe, E4bHostLayer, E4bHostWeights, E4bLora, Gemma4E4bWgpu, HostLin,
};

const PREFIX: &str = "model.language_model";
const SYNTH_RANK: usize = 4;
const SYNTH_ALPHA: f64 = 8.0;
const SYNTH_SCALING: f32 = 2.0;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn pick(&mut self, table: &[f32]) -> f32 {
        table[(self.next() >> 33) as usize % table.len()]
    }
}

const GRID: f32 = 1.0 / 64.0;

fn coarse_weight(rng: &mut Rng) -> f32 {
    let step = ((rng.next() >> 33) % 33) as i32 - 16;
    step as f32 * GRID
}

fn coarse_norm(rng: &mut Rng) -> f32 {
    let step = ((rng.next() >> 33) % 17) as i32 - 8;
    1.0 + step as f32 * GRID
}

fn coarse_vec(rng: &mut Rng, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| bf16::from_f32(coarse_weight(rng)).to_bits())
        .collect()
}

fn coarse_norm_vec(rng: &mut Rng, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| bf16::from_f32(coarse_norm(rng)).to_bits())
        .collect()
}

fn tiny_host_weights(config: &Gemma4Config, seed: u64) -> E4bHostWeights {
    let mut rng = Rng(seed);
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let hpl = config.hidden_size_per_layer_input;
    let n_q = config.num_attention_heads;
    let n_layers = config.num_hidden_layers;

    let mut layers = Vec::new();
    for i in 0..n_layers {
        let kind = config.layer_kind(i);
        let hd = config.head_dim_for(kind);
        let nkv = config.num_kv_heads_for(kind);
        let q_dim = n_q * hd;
        let kv_dim = nkv * hd;
        let kv_source = config.kv_source_layer(i);
        let has_v = !matches!(
            (kind, config.attention_k_eq_v),
            (LayerType::FullAttention, true)
        );
        let qkv_rows = match kv_source {
            Some(_) => q_dim,
            None => q_dim + kv_dim * if has_v { 2 } else { 1 },
        };
        let k_norm = match kv_source {
            Some(_) => Vec::new(),
            None => coarse_norm_vec(&mut rng, hd),
        };
        layers.push(E4bHostLayer {
            kind,
            kv_source,
            input_ln: coarse_norm_vec(&mut rng, hidden),
            post_attn_ln: coarse_norm_vec(&mut rng, hidden),
            pre_ff_ln: coarse_norm_vec(&mut rng, hidden),
            post_ff_ln: coarse_norm_vec(&mut rng, hidden),
            post_per_layer_input_norm: coarse_norm_vec(&mut rng, hidden),
            q_norm: coarse_norm_vec(&mut rng, hd),
            k_norm,
            layer_scalar: 0.875,
            has_v,
            qkv: HostLin::new(coarse_vec(&mut rng, qkv_rows * hidden), qkv_rows, hidden),
            o: HostLin::new(coarse_vec(&mut rng, hidden * q_dim), hidden, q_dim),
            gate_up: HostLin::new(coarse_vec(&mut rng, 2 * inter * hidden), 2 * inter, hidden),
            down: HostLin::new(coarse_vec(&mut rng, hidden * inter), hidden, inter),
            per_layer_input_gate: HostLin::new(coarse_vec(&mut rng, hpl * hidden), hpl, hidden),
            per_layer_projection: HostLin::new(coarse_vec(&mut rng, hidden * hpl), hidden, hpl),
        });
    }

    let ple_row = n_layers * hpl;
    E4bHostWeights {
        embed: coarse_vec(&mut rng, config.vocab_size * hidden),
        embed_per_layer: coarse_vec(&mut rng, config.vocab_size_per_layer() * ple_row),
        per_layer_model_projection: HostLin::new(
            coarse_vec(&mut rng, ple_row * hidden),
            ple_row,
            hidden,
        ),
        per_layer_projection_norm: coarse_norm_vec(&mut rng, hpl),
        final_norm: coarse_norm_vec(&mut rng, hidden),
        layers,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fill {
    Dense,
    ZeroA,
    ZeroB,
}

struct Synth {
    rank: usize,
    alpha: f64,
    mods: BTreeMap<String, (Vec<f32>, Vec<f32>)>,
}

fn synth_adapter(config: &Gemma4Config, seed: u64, fill: Fill) -> Synth {
    const A_VALS: [f32; 5] = [0.0, 0.125, -0.125, 0.125, -0.125];
    const B_VALS: [f32; 5] = [0.0, 0.25, -0.25, 0.25, -0.25];
    let mut rng = Rng(seed);
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let n_q = config.num_attention_heads;
    let rank = SYNTH_RANK;
    let mut mods = BTreeMap::new();
    let mut emit = |rng: &mut Rng, name: String, k: usize, n: usize| {
        let a: Vec<f32> = (0..rank * k)
            .map(|_| {
                if fill == Fill::ZeroA {
                    0.0
                } else {
                    rng.pick(&A_VALS)
                }
            })
            .collect();
        let b: Vec<f32> = (0..n * rank)
            .map(|_| {
                if fill == Fill::ZeroB {
                    0.0
                } else {
                    rng.pick(&B_VALS)
                }
            })
            .collect();
        mods.insert(name, (a, b));
    };
    for li in 0..config.num_hidden_layers {
        let kind = config.layer_kind(li);
        let hd = config.head_dim_for(kind);
        let nkv = config.num_kv_heads_for(kind);
        let q_dim = n_q * hd;
        let kv_dim = nkv * hd;
        let p = format!("{PREFIX}.layers.{li}");
        emit(&mut rng, format!("{p}.self_attn.q_proj"), hidden, q_dim);
        if config.kv_source_layer(li).is_none() {
            emit(&mut rng, format!("{p}.self_attn.k_proj"), hidden, kv_dim);
            emit(&mut rng, format!("{p}.self_attn.v_proj"), hidden, kv_dim);
        }
        emit(&mut rng, format!("{p}.self_attn.o_proj"), q_dim, hidden);
        emit(&mut rng, format!("{p}.mlp.gate_proj"), hidden, inter);
        emit(&mut rng, format!("{p}.mlp.up_proj"), hidden, inter);
        emit(&mut rng, format!("{p}.mlp.down_proj"), inter, hidden);
    }
    Synth {
        rank,
        alpha: SYNTH_ALPHA,
        mods,
    }
}

fn write_safetensors(path: &Path, tensors: &mut [(String, Vec<usize>, Vec<f32>)]) {
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, shape, vals) in tensors.iter() {
        let start = data.len();
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let mut hdr = serde_json::to_string(&serde_json::Value::Object(header))
        .unwrap()
        .into_bytes();
    while !hdr.len().is_multiple_of(8) {
        hdr.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + hdr.len() + data.len());
    out.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
    out.extend_from_slice(&hdr);
    out.extend_from_slice(&data);
    std::fs::write(path, out).expect("write adapter_model.safetensors");
}

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_adapter(tag: &str, synth: &Synth) -> Scratch {
    let dir = std::env::temp_dir().join(format!(
        "nv-lora-e4b-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create adapter dir");
    let cfg = serde_json::json!({
        "peft_type": "LORA",
        "r": synth.rank,
        "lora_alpha": synth.alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_rslora": false,
        "use_dora": false,
        "modules_to_save": serde_json::Value::Null,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    });
    std::fs::write(
        dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&cfg).unwrap(),
    )
    .expect("write adapter_config.json");

    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    for (module, (a, b)) in &synth.mods {
        let k = a.len() / synth.rank;
        let n = b.len() / synth.rank;
        tensors.push((
            format!("base_model.model.{module}.lora_A.weight"),
            vec![synth.rank, k],
            a.clone(),
        ));
        tensors.push((
            format!("base_model.model.{module}.lora_B.weight"),
            vec![n, synth.rank],
            b.clone(),
        ));
    }
    write_safetensors(&dir.join("adapter_model.safetensors"), &mut tensors);
    Scratch(dir)
}

fn merge_rows(w: &mut [u16], k: usize, row_off: usize, a: &[f32], b: &[f32], rank: usize) {
    let n = b.len() / rank;
    let a_bf: Vec<f32> = a.iter().map(|v| bf16::from_f32(*v).to_f32()).collect();
    let b_bf: Vec<f32> = b
        .iter()
        .map(|v| bf16::from_f32(bf16::from_f32(*v).to_f32() * SYNTH_SCALING).to_f32())
        .collect();
    for row in 0..n {
        for col in 0..k {
            let mut d = 0f32;
            let mut d64 = 0f64;
            for r in 0..rank {
                d += b_bf[row * rank + r] * a_bf[r * k + col];
                d64 += b_bf[row * rank + r] as f64 * a_bf[r * k + col] as f64;
            }
            assert_eq!(d as f64, d64, "delta accumulation left the exact grid");
            if d == 0.0 {
                continue;
            }
            let idx = (row_off + row) * k + col;
            let base = bf16::from_bits(w[idx]).to_f32();
            let merged = base + d;
            let enc = bf16::from_f32(merged);
            assert_eq!(
                enc.to_f32(),
                merged,
                "merged weight {merged} is not exactly representable in bf16"
            );
            w[idx] = enc.to_bits();
        }
    }
}

fn merge_adapter(weights: &mut E4bHostWeights, config: &Gemma4Config, synth: &Synth) {
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let n_q = config.num_attention_heads;
    let rank = synth.rank;
    for li in 0..config.num_hidden_layers {
        let kind = config.layer_kind(li);
        let hd = config.head_dim_for(kind);
        let nkv = config.num_kv_heads_for(kind);
        let q_dim = n_q * hd;
        let kv_dim = nkv * hd;
        let p = format!("{PREFIX}.layers.{li}");
        let layer = &mut weights.layers[li];
        let get = |name: &str| synth.mods.get(name);

        if let Some((a, b)) = get(&format!("{p}.self_attn.q_proj")) {
            merge_rows(&mut layer.qkv.w, hidden, 0, a, b, rank);
        }
        if config.kv_source_layer(li).is_none() {
            if let Some((a, b)) = get(&format!("{p}.self_attn.k_proj")) {
                merge_rows(&mut layer.qkv.w, hidden, q_dim, a, b, rank);
            }
            if let Some((a, b)) = get(&format!("{p}.self_attn.v_proj")) {
                merge_rows(&mut layer.qkv.w, hidden, q_dim + kv_dim, a, b, rank);
            }
        }
        if let Some((a, b)) = get(&format!("{p}.self_attn.o_proj")) {
            merge_rows(&mut layer.o.w, q_dim, 0, a, b, rank);
        }
        if let Some((a, b)) = get(&format!("{p}.mlp.gate_proj")) {
            merge_rows(&mut layer.gate_up.w, hidden, 0, a, b, rank);
        }
        if let Some((a, b)) = get(&format!("{p}.mlp.up_proj")) {
            merge_rows(&mut layer.gate_up.w, hidden, inter, a, b, rank);
        }
        if let Some((a, b)) = get(&format!("{p}.mlp.down_proj")) {
            merge_rows(&mut layer.down.w, inter, 0, a, b, rank);
        }
    }
}

fn flip_lsb(w: &mut [u16]) {
    for v in w.iter_mut() {
        *v ^= 1;
    }
}

fn perturb_one_ulp(weights: &mut E4bHostWeights) {
    for layer in weights.layers.iter_mut() {
        flip_lsb(&mut layer.qkv.w);
        flip_lsb(&mut layer.o.w);
        flip_lsb(&mut layer.gate_up.w);
        flip_lsb(&mut layer.down.w);
    }
}

const STEPS: [u32; 12] = [7, 300, 42, 511, 0, 13, 99, 250, 8, 9, 10, 11];

fn run_steps(m: &mut Gemma4E4bWgpu) -> Vec<Vec<f32>> {
    run_steps_full(m).1
}

fn run_steps_full(m: &mut Gemma4E4bWgpu) -> (Vec<u32>, Vec<Vec<f32>>) {
    let mut toks = Vec::new();
    let mut logits = Vec::new();
    for t in STEPS.iter() {
        let (tok, lg) = m.decode_step_logits(*t).unwrap();
        toks.push(tok);
        logits.push(lg);
    }
    (toks, logits)
}

fn assert_bit_identical(a: &[Vec<f32>], b: &[Vec<f32>], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: step count");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.len(), y.len(), "{what}: step {i} logit count");
        for (j, (p, q)) in x.iter().zip(y.iter()).enumerate() {
            assert_eq!(
                p.to_bits(),
                q.to_bits(),
                "{what}: step {i} logit {j}: {p} vs {q}"
            );
        }
    }
}

fn max_abs_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    let mut m = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        for (p, q) in x.iter().zip(y.iter()) {
            m = m.max((p - q).abs());
        }
    }
    m
}

fn expected_lora_passes(config: &Gemma4Config) -> usize {
    let mut n = 0usize;
    for li in 0..config.num_hidden_layers {
        let qkv_segs = if config.kv_source_layer(li).is_some() {
            1
        } else {
            3
        };
        n += 1 + 2 * qkv_segs;
        n += 3 * (1 + 2);
    }
    n
}

#[test]
fn compiling_lora_in_without_an_adapter_leaves_the_graph_identical() {
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let weights = tiny_host_weights(&config, 0xa11ce);

    let mut plain = Gemma4E4bWgpu::new(
        Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
        &weights,
        64,
    )
    .unwrap();
    let mut gated = Gemma4E4bWgpu::new_with_lora(
        Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
        &weights,
        64,
        None,
    )
    .unwrap();

    assert_eq!(
        plain.pass_count(),
        gated.pass_count(),
        "no adapter must not change the decode dispatch count"
    );
    assert_eq!(
        plain.prefill_pass_count(),
        gated.prefill_pass_count(),
        "no adapter must not change the prefill dispatch count"
    );
    assert_eq!(gated.lora_passes(), 0);
    eprintln!(
        "passes per decode token, no adapter: {}",
        plain.pass_count()
    );

    let a = run_steps(&mut plain);
    let b = run_steps(&mut gated);
    assert_bit_identical(&a, &b, "no-adapter graph");
}

#[test]
fn a_zero_adapter_leaves_logits_bit_identical() {
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let weights = tiny_host_weights(&config, 0xa11ce);

    let mut base = Gemma4E4bWgpu::new(
        Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
        &weights,
        64,
    )
    .unwrap();
    let want = run_steps(&mut base);
    let base_passes = base.pass_count();
    let base_prefill = base.prefill_pass_count();
    drop(base);

    for (tag, fill) in [("zerob", Fill::ZeroB), ("zeroa", Fill::ZeroA)] {
        let synth = synth_adapter(&config, 0x2e40, fill);
        let dir = write_adapter(tag, &synth);
        let lora = E4bLora::from_peft_dir(&dir.0, &config).unwrap();
        assert_eq!(lora.rank(), SYNTH_RANK);
        assert_eq!(lora.matched_modules(), synth.mods.len());

        let mut m = Gemma4E4bWgpu::new_with_lora(
            Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
            &weights,
            64,
            Some(&lora),
        )
        .unwrap();
        assert!(
            m.pass_count() > base_passes,
            "{tag}: the lora graph must record extra dispatches"
        );
        assert_eq!(
            m.pass_count() - base_passes,
            lora.total_pass_count(),
            "{tag}: decode pass delta"
        );
        assert_eq!(
            m.prefill_pass_count() - base_prefill,
            lora.total_pass_count(),
            "{tag}: prefill pass delta"
        );
        let got = run_steps(&mut m);
        assert_bit_identical(&want, &got, &format!("{tag} adapter"));
    }
}

#[test]
fn lora_pass_geometry_matches_the_layer_structure() {
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let synth = synth_adapter(&config, 0x9e01, Fill::Dense);
    let dir = write_adapter("geom", &synth);
    let lora = E4bLora::from_peft_dir(&dir.0, &config).unwrap();
    assert_eq!(lora.layer_count(), config.num_hidden_layers);
    assert_eq!(
        lora.total_pass_count(),
        expected_lora_passes(&config),
        "one fused dispatch plus a widen/repack pair per packed destination"
    );
    assert_eq!(lora.total_pass_count(), 88);
    for li in 0..config.num_hidden_layers {
        let want = if config.kv_source_layer(li).is_some() {
            12
        } else {
            16
        };
        assert_eq!(lora.layer_pass_count(li), want, "layer {li}");
    }
}

#[test]
fn a_real_adapter_shifts_logits_and_tracks_the_merged_weight_oracle() {
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let weights = tiny_host_weights(&config, 0xa11ce);
    let synth = synth_adapter(&config, 0xd317a1, Fill::Dense);
    let dir = write_adapter("dense", &synth);
    let lora = E4bLora::from_peft_dir(&dir.0, &config).unwrap();
    assert_eq!(lora.matched_modules(), synth.mods.len());

    let mut merged_weights = tiny_host_weights(&config, 0xa11ce);
    merge_adapter(&mut merged_weights, &config, &synth);

    let mut base = Gemma4E4bWgpu::new(
        Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
        &weights,
        64,
    )
    .unwrap();
    let (t_base, l_base) = run_steps_full(&mut base);
    drop(base);

    let mut with_lora = Gemma4E4bWgpu::new_with_lora(
        Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
        &weights,
        64,
        Some(&lora),
    )
    .unwrap();
    let (t_lora, l_lora) = run_steps_full(&mut with_lora);
    drop(with_lora);

    let mut merged = Gemma4E4bWgpu::new(
        Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
        &merged_weights,
        64,
    )
    .unwrap();
    let (t_merged, l_merged) = run_steps_full(&mut merged);
    drop(merged);

    for (i, step) in l_lora.iter().enumerate() {
        assert!(
            step.iter().all(|v| v.is_finite()),
            "step {i}: non-finite logits with lora attached"
        );
    }

    let mut nudged_weights = tiny_host_weights(&config, 0xa11ce);
    perturb_one_ulp(&mut nudged_weights);
    let mut nudged = Gemma4E4bWgpu::new(
        Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
        &nudged_weights,
        64,
    )
    .unwrap();
    let l_nudged = run_steps(&mut nudged);
    drop(nudged);

    let shift = max_abs_diff(&l_base, &l_lora);
    let residual = max_abs_diff(&l_merged, &l_lora);
    let merged_shift = max_abs_diff(&l_base, &l_merged);
    let noise = max_abs_diff(&l_base, &l_nudged);
    eprintln!(
        "max |lora - base| = {shift}, max |merged - base| = {merged_shift}, \
         max |lora - merged| = {residual}, one-ulp noise floor = {noise}"
    );
    assert!(
        shift > 1.0,
        "the adapter must move the logits, got max |delta| {shift}"
    );
    assert!(
        merged_shift > 1.0,
        "the merged-weight oracle must move the logits too, got {merged_shift}"
    );
    assert!(
        noise > 0.0,
        "the one-ulp control must move the logits, otherwise it calibrates nothing"
    );
    assert!(
        shift > 3.0 * noise,
        "the adapter's effect {shift} is not clear of the bf16 noise floor {noise}"
    );
    assert!(
        residual < 1.5 * noise,
        "runtime lora disagrees with the merged-weight oracle by {residual}, \
         more than the {noise} this graph moves under a one-ulp weight perturbation"
    );
    let total: usize = l_base.iter().map(|s| s.len()).sum();
    let moved = l_base
        .iter()
        .zip(l_lora.iter())
        .flat_map(|(a, b)| a.iter().zip(b.iter()))
        .filter(|(p, q)| (*p - *q).abs() > noise)
        .count();
    eprintln!("logits moved by more than the noise floor: {moved}/{total}");
    assert!(
        moved * 10 > total,
        "only {moved}/{total} logits moved by more than the {noise} noise floor"
    );
    eprintln!("greedy tokens base   {t_base:?}");
    eprintln!("greedy tokens lora   {t_lora:?}");
    eprintln!("greedy tokens merged {t_merged:?}");
    assert_eq!(
        t_lora, t_merged,
        "runtime lora and the merged-weight oracle must agree on the greedy token stream"
    );
}

fn prompt_then_steps(m: &mut Gemma4E4bWgpu, prompt: &[u32]) -> Vec<Vec<f32>> {
    m.prefill_tokens(prompt).unwrap();
    STEPS[..6]
        .iter()
        .map(|t| m.decode_step_logits(*t).unwrap().1)
        .collect()
}

#[test]
fn prefill_honours_the_adapter_and_a_zero_adapter_stays_bit_identical() {
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let weights = tiny_host_weights(&config, 0xa11ce);
    let prompt: Vec<u32> = (0..25u32).map(|i| (i * 17 + 3) % 512).collect();

    let mut base = Gemma4E4bWgpu::new(
        Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
        &weights,
        128,
    )
    .unwrap();
    assert!(
        base.prefill_pass_count() > 0,
        "this test needs the prefill graph; do not run it with NV_E4B_WGPU_PREFILL=0"
    );
    assert!(base.prefill_chunk_len() > 1);
    let l_base = prompt_then_steps(&mut base, &prompt);
    drop(base);

    let zero = synth_adapter(&config, 0x2e40, Fill::ZeroB);
    let zdir = write_adapter("pfzero", &zero);
    let zlora = E4bLora::from_peft_dir(&zdir.0, &config).unwrap();
    let mut m0 = Gemma4E4bWgpu::new_with_lora(
        Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
        &weights,
        128,
        Some(&zlora),
    )
    .unwrap();
    let l_zero = prompt_then_steps(&mut m0, &prompt);
    drop(m0);
    assert_bit_identical(&l_base, &l_zero, "zero adapter through prefill");

    let synth = synth_adapter(&config, 0xd317a1, Fill::Dense);
    let dir = write_adapter("pfdense", &synth);
    let lora = E4bLora::from_peft_dir(&dir.0, &config).unwrap();
    let mut merged_weights = tiny_host_weights(&config, 0xa11ce);
    merge_adapter(&mut merged_weights, &config, &synth);

    let mut m1 = Gemma4E4bWgpu::new_with_lora(
        Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
        &weights,
        128,
        Some(&lora),
    )
    .unwrap();
    let l_lora = prompt_then_steps(&mut m1, &prompt);
    drop(m1);

    let mut m2 = Gemma4E4bWgpu::new(
        Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
        &merged_weights,
        128,
    )
    .unwrap();
    let l_merged = prompt_then_steps(&mut m2, &prompt);
    drop(m2);

    let mut nudged_weights = tiny_host_weights(&config, 0xa11ce);
    perturb_one_ulp(&mut nudged_weights);
    let mut m3 = Gemma4E4bWgpu::new(
        Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
        &nudged_weights,
        128,
    )
    .unwrap();
    let l_nudged = prompt_then_steps(&mut m3, &prompt);
    drop(m3);

    let shift = max_abs_diff(&l_base, &l_lora);
    let residual = max_abs_diff(&l_merged, &l_lora);
    let noise = max_abs_diff(&l_base, &l_nudged);
    eprintln!(
        "prefill: max |lora - base| = {shift}, max |lora - merged| = {residual}, \
         one-ulp noise floor = {noise}"
    );
    assert!(
        shift > 3.0 * noise,
        "prefill shift {shift} vs noise {noise}"
    );
    assert!(
        residual < 1.5 * noise,
        "prefill runtime lora disagrees with the merged oracle by {residual} vs noise {noise}"
    );
}

#[test]
fn an_adapter_that_matches_no_projection_is_rejected() {
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let mut mods = BTreeMap::new();
    mods.insert(
        "vision_tower.blocks.0.attn.qkv.linear".to_string(),
        (
            vec![0.125f32; SYNTH_RANK * 8],
            vec![0.25f32; 8 * SYNTH_RANK],
        ),
    );
    let synth = Synth {
        rank: SYNTH_RANK,
        alpha: SYNTH_ALPHA,
        mods,
    };
    let dir = write_adapter("nomatch", &synth);
    let msg = match E4bLora::from_peft_dir(&dir.0, &config) {
        Ok(l) => panic!(
            "an adapter with no text-tower module must be rejected, got {} matches",
            l.matched_modules()
        ),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        msg.contains("matched no gemma4-e4b text projection"),
        "unexpected error: {msg}"
    );
}

const LADDER: [f32; 8] = [-1.0, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0];
const B_LADDER: [f32; 6] = [-0.5, -0.25, -0.125, 0.125, 0.25, 0.5];

fn ladder_bits(rng: &mut Rng, n: usize, table: &[f32]) -> Vec<u16> {
    (0..n)
        .map(|_| bf16::from_f32(rng.pick(table)).to_bits())
        .collect()
}

fn dec(x: u16) -> f32 {
    bf16::from_bits(x).to_f32()
}

fn cpu_oracle(
    x: &[u16],
    a: &[u16],
    b: &[u16],
    y: &mut [u16],
    widths: &[usize],
    m: usize,
    rank: usize,
    k: usize,
) {
    let total_w: usize = widths.iter().sum();
    let mut b_off = 0usize;
    let mut slice_start = 0usize;
    for (s, w) in widths.iter().enumerate() {
        for row in 0..m {
            let mut h = vec![0f32; rank];
            for (r, hr) in h.iter_mut().enumerate() {
                let mut acc = 0f32;
                let mut acc64 = 0f64;
                for kk in 0..k {
                    let xv = dec(x[row * k + kk]);
                    let av = dec(a[s * rank * k + r * k + kk]);
                    acc = xv.mul_add(av, acc);
                    acc64 += xv as f64 * av as f64;
                }
                assert_eq!(acc as f64, acc64, "shrink accumulation left the exact grid");
                *hr = acc;
            }
            for n in 0..*w {
                let mut acc = 0f32;
                let mut acc64 = 0f64;
                for (r, hr) in h.iter().enumerate() {
                    let bv = dec(b[b_off + n * rank + r]);
                    acc = hr.mul_add(bv, acc);
                    acc64 += *hr as f64 * bv as f64;
                }
                assert_eq!(acc as f64, acc64, "expand accumulation left the exact grid");
                let idx = row * total_w + slice_start + n;
                let sum = dec(y[idx]) + acc;
                assert_eq!(
                    sum as f64,
                    dec(y[idx]) as f64 + acc as f64,
                    "y accumulation left the exact grid"
                );
                y[idx] = bf16::from_f32(sum).to_bits();
            }
        }
        b_off += w * rank;
        slice_start += w;
    }
}

fn probe_case(m: usize, rank: usize, k: usize, widths: &[usize], segs: &[usize], seed: u64) {
    let mut rng = Rng(seed);
    let total_w: usize = widths.iter().sum();
    assert_eq!(total_w, segs.iter().sum::<usize>());
    let x = ladder_bits(&mut rng, m * k, &LADDER);
    let a = ladder_bits(&mut rng, widths.len() * rank * k, &LADDER);
    let b = ladder_bits(&mut rng, total_w * rank, &B_LADDER);
    let y0 = ladder_bits(&mut rng, m * total_w, &LADDER);

    let mut segments: Vec<Vec<u16>> = Vec::new();
    let mut off = 0usize;
    for w in segs {
        let mut seg = vec![0u16; m * w];
        for row in 0..m {
            seg[row * w..(row + 1) * w]
                .copy_from_slice(&y0[row * total_w + off..row * total_w + off + w]);
        }
        segments.push(seg);
        off += w;
    }

    lora_site_probe(&x, &a, &b, &mut segments, widths, segs, m, rank, k).unwrap();

    let mut flat = vec![0u16; m * total_w];
    let mut off = 0usize;
    for (seg, w) in segments.iter().zip(segs.iter()) {
        for row in 0..m {
            flat[row * total_w + off..row * total_w + off + w]
                .copy_from_slice(&seg[row * w..(row + 1) * w]);
        }
        off += w;
    }

    let mut oracle = y0.clone();
    cpu_oracle(&x, &a, &b, &mut oracle, widths, m, rank, k);
    for (i, (got, want)) in flat.iter().zip(oracle.iter()).enumerate() {
        assert_eq!(
            got,
            want,
            "cpu oracle mismatch at {i}: {} vs {}",
            dec(*got),
            dec(*want)
        );
    }

    use nv_kernels::wgpu_backend::kernels::lora;
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared().unwrap();
    let meta = lora::LoraMeta::prepare(&vec![0i32; m], 1);
    let a_slices: Vec<&[u16]> = (0..widths.len())
        .map(|s| &a[s * rank * k..(s + 1) * rank * k])
        .collect();
    let mut b_slices: Vec<&[u16]> = Vec::new();
    let mut off = 0usize;
    for w in widths {
        b_slices.push(&b[off..off + w * rank]);
        off += w * rank;
    }
    let mut kernel_y = y0.clone();
    lora::lora_fused(
        ctx,
        &x,
        &a_slices,
        &b_slices,
        &mut kernel_y,
        &meta,
        widths,
        m,
        rank,
        k,
        0,
        total_w,
        total_w,
        1.0,
    )
    .unwrap();
    assert_eq!(
        flat, kernel_y,
        "the graph-resident site path diverged from nv-kernels lora_fused"
    );
}

#[test]
fn the_resident_site_path_matches_the_kernel_and_a_cpu_oracle_bitwise() {
    if ctx_or_skip().is_none() {
        return;
    }
    probe_case(1, 4, 32, &[16, 8, 8], &[16, 8, 8], 0x51de);
    probe_case(3, 4, 32, &[16, 8, 8], &[16, 8, 8], 0x51df);
    probe_case(1, 8, 64, &[32, 32], &[64], 0x51e0);
    probe_case(5, 16, 32, &[24, 8], &[24, 8], 0x51e1);
    probe_case(17, 4, 32, &[16], &[16], 0x51e2);
}

#[test]
#[ignore]
fn real_peft_adapter_builds_e4b_lora_sites() {
    let dir = match std::env::var("NV_LORA_REAL_ADAPTER_DIR") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("set NV_LORA_REAL_ADAPTER_DIR to the pinned PEFT directory");
            return;
        }
    };
    let cfg_path = std::env::var("NV_LORA_REAL_BASE_CONFIG")
        .expect("set NV_LORA_REAL_BASE_CONFIG to the base model config.json");
    let raw = std::fs::read_to_string(&cfg_path).unwrap();
    let config = Gemma4Config::from_hf_json_str(&raw).unwrap();
    let lora = E4bLora::from_peft_dir(&dir, &config).unwrap();
    eprintln!(
        "real adapter: rank {} matched {} modules over {} layers, +{} passes/token",
        lora.rank(),
        lora.matched_modules(),
        lora.layer_count(),
        lora.total_pass_count()
    );
    assert_eq!(lora.layer_count(), config.num_hidden_layers);

    let n_layers = config.num_hidden_layers;
    let shared = (0..n_layers)
        .filter(|i| config.kv_source_layer(*i).is_some())
        .count();
    let want_matched = 5 * n_layers + 2 * (n_layers - shared);
    assert_eq!(
        lora.matched_modules(),
        want_matched,
        "every projection this graph computes must be hooked: {n_layers} layers, {shared} kv-shared"
    );

    let skipped = lora.skipped_modules();
    eprintln!("skipped text-tower modules: {}", skipped.len());
    assert_eq!(
        skipped.len(),
        2 * shared,
        "the only text modules without a home are k/v on kv-shared layers"
    );
    for name in skipped {
        assert!(
            name.ends_with("self_attn.k_proj") || name.ends_with("self_attn.v_proj"),
            "unexpected skipped module {name}"
        );
        let li: usize = name
            .trim_start_matches("model.language_model.layers.")
            .split('.')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert!(
            config.kv_source_layer(li).is_some(),
            "{name} was dropped but layer {li} does compute its own k/v"
        );
    }
    assert_eq!(
        lora.total_pass_count(),
        (n_layers - shared) * 16 + shared * 12
    );
}
