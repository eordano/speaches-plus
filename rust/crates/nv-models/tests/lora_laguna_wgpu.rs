#![cfg(feature = "laguna-wip")]
#![cfg(feature = "wgpu")]

mod common;
mod hub_snapshot;
use common::CFG;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use half::bf16;
use nv_models::laguna::LagunaConfig;
use nv_models::laguna_wgpu::config::{FfnKind, LagunaShapes};
use nv_models::laguna_wgpu::weights::{
    HostAttention, HostBf16ExpertStack, HostBf16Lin, HostDenseMlp, HostExperts, HostFfn,
    HostLayer, HostLin, HostMoe, HostWeights,
};
use nv_models::laguna_wgpu::{LagunaLora, LagunaWgpu};

const MAX_SEQ: usize = 24;
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

fn coarse_lin(rng: &mut Rng, n: usize, k: usize) -> HostLin {
    HostLin::Bf16(HostBf16Lin {
        w: coarse_vec(rng, n * k),
        n,
        k,
    })
}

fn coarse_expert_stack(rng: &mut Rng, e: usize, n: usize, k: usize) -> HostExperts {
    HostExperts::Bf16(HostBf16ExpertStack {
        w: coarse_vec(rng, e * n * k),
        e,
        n,
        k,
    })
}

fn config_of() -> LagunaConfig {
    LagunaConfig::from_hf_json_str(CFG).unwrap()
}

fn shapes_of(cfg: &LagunaConfig) -> LagunaShapes {
    LagunaShapes::derive(cfg, MAX_SEQ).unwrap()
}

fn have_gpu() -> bool {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(_) => true,
        Err(e) => {
            eprintln!("SKIP: no wgpu adapter: {e}");
            false
        }
    }
}

fn tiny_host_weights(shapes: &LagunaShapes, seed: u64) -> HostWeights {
    let mut rng = Rng(seed);
    let hidden = shapes.hidden_size;
    let mut layers = Vec::with_capacity(shapes.num_layers);
    for li in 0..shapes.num_layers {
        let ls = *shapes.layer(li);
        let attn = HostAttention {
            q: coarse_lin(&mut rng, ls.q_rows, hidden),
            k: coarse_lin(&mut rng, ls.kv_rows, hidden),
            v: coarse_lin(&mut rng, ls.kv_rows, hidden),
            o: coarse_lin(&mut rng, hidden, ls.q_rows),
            g: (ls.gate_rows > 0).then(|| coarse_lin(&mut rng, ls.gate_rows, hidden)),
            q_norm: coarse_norm_vec(&mut rng, ls.head_dim),
            k_norm: coarse_norm_vec(&mut rng, ls.head_dim),
        };
        let ffn = if ls.ffn_kind == FfnKind::Moe {
            let n_e = shapes.num_experts;
            let inter = shapes.moe_intermediate_size;
            let sinter = shapes.shared_expert_intermediate_size;
            HostFfn::Moe(Box::new(HostMoe {
                router: HostBf16Lin {
                    w: coarse_vec(&mut rng, n_e * hidden),
                    n: n_e,
                    k: hidden,
                },
                selection_bias: (0..n_e).map(|e| 0.05 * (e as f32 - 1.5)).collect(),
                experts_gate: coarse_expert_stack(&mut rng, n_e, inter, hidden),
                experts_up: coarse_expert_stack(&mut rng, n_e, inter, hidden),
                experts_down: coarse_expert_stack(&mut rng, n_e, hidden, inter),
                shared_gate: coarse_lin(&mut rng, sinter, hidden),
                shared_up: coarse_lin(&mut rng, sinter, hidden),
                shared_down: coarse_lin(&mut rng, hidden, sinter),
            }))
        } else {
            let inter = ls.ffn_intermediate;
            HostFfn::Dense(Box::new(HostDenseMlp {
                gate: coarse_lin(&mut rng, inter, hidden),
                up: coarse_lin(&mut rng, inter, hidden),
                down: coarse_lin(&mut rng, hidden, inter),
            }))
        };
        layers.push(HostLayer {
            kind: ls.attn_kind,
            input_ln: coarse_norm_vec(&mut rng, hidden),
            post_attn_ln: coarse_norm_vec(&mut rng, hidden),
            attn,
            ffn,
        });
    }
    HostWeights {
        embed: coarse_vec(&mut rng, shapes.vocab_size * hidden),
        final_norm: coarse_norm_vec(&mut rng, hidden),
        lm_head: coarse_vec(&mut rng, shapes.vocab_size * hidden),
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
    moe_mlp_mods: usize,
}

fn synth_adapter(shapes: &LagunaShapes, seed: u64, fill: Fill, with_moe_mlp: bool) -> Synth {
    const A_VALS: [f32; 5] = [0.0, 0.125, -0.125, 0.125, -0.125];
    const B_VALS: [f32; 5] = [0.0, 0.25, -0.25, 0.25, -0.25];
    let mut rng = Rng(seed);
    let hidden = shapes.hidden_size;
    let rank = SYNTH_RANK;
    let mut mods = BTreeMap::new();
    let mut moe_mlp_mods = 0usize;
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
    for li in 0..shapes.num_layers {
        let ls = *shapes.layer(li);
        let p = format!("model.layers.{li}");
        emit(&mut rng, format!("{p}.self_attn.q_proj"), hidden, ls.q_rows);
        emit(&mut rng, format!("{p}.self_attn.k_proj"), hidden, ls.kv_rows);
        emit(&mut rng, format!("{p}.self_attn.v_proj"), hidden, ls.kv_rows);
        emit(&mut rng, format!("{p}.self_attn.o_proj"), ls.q_rows, hidden);
        if ls.gate_rows > 0 {
            emit(&mut rng, format!("{p}.self_attn.g_proj"), hidden, ls.gate_rows);
        }
        if ls.ffn_kind == FfnKind::Dense {
            let inter = ls.ffn_intermediate;
            emit(&mut rng, format!("{p}.mlp.gate_proj"), hidden, inter);
            emit(&mut rng, format!("{p}.mlp.up_proj"), hidden, inter);
            emit(&mut rng, format!("{p}.mlp.down_proj"), inter, hidden);
        } else if with_moe_mlp {
            let inter = ls.ffn_intermediate;
            emit(
                &mut rng,
                format!("{p}.mlp.experts.0.gate_proj"),
                hidden,
                inter,
            );
            emit(&mut rng, format!("{p}.mlp.experts.1.up_proj"), hidden, inter);
            emit(
                &mut rng,
                format!("{p}.mlp.shared_expert.down_proj"),
                inter,
                hidden,
            );
            moe_mlp_mods += 3;
        }
    }
    Synth {
        rank,
        alpha: SYNTH_ALPHA,
        mods,
        moe_mlp_mods,
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
        "nv-lora-laguna-{tag}-{}-{:?}",
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
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "g_proj",
                           "gate_proj", "up_proj", "down_proj"],
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

fn merge_rows(w: &mut [u16], k: usize, a: &[f32], b: &[f32], rank: usize) {
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
            let idx = row * k + col;
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

fn bf16_of(lin: &mut HostLin) -> &mut HostBf16Lin {
    match lin {
        HostLin::Bf16(l) => l,
        HostLin::Nvfp4(_) => panic!("tiny weights are bf16 by construction"),
    }
}

fn merge_adapter(weights: &mut HostWeights, shapes: &LagunaShapes, synth: &Synth) {
    let rank = synth.rank;
    for li in 0..shapes.num_layers {
        let ls = *shapes.layer(li);
        let p = format!("model.layers.{li}");
        let layer = &mut weights.layers[li];
        let merge_into = |lin: &mut HostLin, name: String| {
            if let Some((a, b)) = synth.mods.get(&name) {
                let l = bf16_of(lin);
                merge_rows(&mut l.w, l.k, a, b, rank);
            }
        };
        merge_into(&mut layer.attn.q, format!("{p}.self_attn.q_proj"));
        merge_into(&mut layer.attn.k, format!("{p}.self_attn.k_proj"));
        merge_into(&mut layer.attn.v, format!("{p}.self_attn.v_proj"));
        merge_into(&mut layer.attn.o, format!("{p}.self_attn.o_proj"));
        if let Some(g) = layer.attn.g.as_mut() {
            if let Some((a, b)) = synth.mods.get(&format!("{p}.self_attn.g_proj")) {
                let l = bf16_of(g);
                merge_rows(&mut l.w, l.k, a, b, rank);
            }
        }
        if ls.ffn_kind == FfnKind::Dense {
            if let HostFfn::Dense(d) = &mut layer.ffn {
                merge_into(&mut d.gate, format!("{p}.mlp.gate_proj"));
                merge_into(&mut d.up, format!("{p}.mlp.up_proj"));
                merge_into(&mut d.down, format!("{p}.mlp.down_proj"));
            }
        }
    }
}

fn flip_lsb(w: &mut [u16]) {
    for v in w.iter_mut() {
        *v ^= 1;
    }
}

fn perturb_one_ulp(weights: &mut HostWeights) {
    for layer in weights.layers.iter_mut() {
        flip_lsb(&mut bf16_of(&mut layer.attn.q).w);
        flip_lsb(&mut bf16_of(&mut layer.attn.k).w);
        flip_lsb(&mut bf16_of(&mut layer.attn.v).w);
        flip_lsb(&mut bf16_of(&mut layer.attn.o).w);
        if let Some(g) = layer.attn.g.as_mut() {
            flip_lsb(&mut bf16_of(g).w);
        }
        if let HostFfn::Dense(d) = &mut layer.ffn {
            flip_lsb(&mut bf16_of(&mut d.gate).w);
            flip_lsb(&mut bf16_of(&mut d.up).w);
            flip_lsb(&mut bf16_of(&mut d.down).w);
        }
    }
}

const STEPS: [u32; 12] = [7, 30, 42, 91, 0, 13, 95, 25, 8, 9, 10, 11];

fn run_steps_full(m: &mut LagunaWgpu) -> (Vec<u32>, Vec<Vec<f32>>) {
    let mut toks = Vec::new();
    let mut logits = Vec::new();
    for t in STEPS.iter() {
        let (tok, lg) = m.decode_step_logits(*t).unwrap();
        toks.push(tok);
        logits.push(lg);
    }
    (toks, logits)
}

fn run_steps(m: &mut LagunaWgpu) -> Vec<Vec<f32>> {
    run_steps_full(m).1
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

fn expected_lora_passes(shapes: &LagunaShapes) -> usize {
    shapes
        .layers
        .iter()
        .map(|ls| {
            let attn_sites = 2 + usize::from(ls.gate_rows > 0);
            let mlp_sites = if ls.ffn_kind == FfnKind::Dense { 2 } else { 0 };
            2 * (attn_sites + mlp_sites)
        })
        .sum()
}

#[test]
fn compiling_lora_in_without_an_adapter_leaves_the_graph_identical() {
    if !have_gpu() {
        return;
    }
    let cfg = config_of();
    let shapes = shapes_of(&cfg);
    let weights = tiny_host_weights(&shapes, 0xa11ce);

    let mut plain = LagunaWgpu::new(cfg.clone(), &weights, MAX_SEQ).unwrap();
    let mut gated = LagunaWgpu::new_with_lora(cfg, &weights, MAX_SEQ, None).unwrap();

    assert_eq!(
        plain.pass_count(),
        gated.pass_count(),
        "no adapter must not change the decode dispatch count"
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
fn a_zero_adapter_leaves_logits_bit_identical_and_counts_its_passes() {
    if !have_gpu() {
        return;
    }
    let cfg = config_of();
    let shapes = shapes_of(&cfg);
    let weights = tiny_host_weights(&shapes, 0xa11ce);

    let mut base = LagunaWgpu::new(cfg.clone(), &weights, MAX_SEQ).unwrap();
    let want = run_steps(&mut base);
    let base_passes = base.pass_count();
    drop(base);

    for (tag, fill) in [("zerob", Fill::ZeroB), ("zeroa", Fill::ZeroA)] {
        let synth = synth_adapter(&shapes, 0x2e40, fill, false);
        let dir = write_adapter(tag, &synth);
        let lora = LagunaLora::from_peft_dir(&dir.0, &cfg).unwrap();
        assert_eq!(lora.rank(), SYNTH_RANK);
        assert_eq!(lora.matched_modules(), synth.mods.len());

        let mut m =
            LagunaWgpu::new_with_lora(cfg.clone(), &weights, MAX_SEQ, Some(&lora)).unwrap();
        assert_eq!(
            m.pass_count() - base_passes,
            lora.total_pass_count(),
            "{tag}: decode pass delta"
        );
        let got = run_steps(&mut m);
        assert_bit_identical(&want, &got, &format!("{tag} adapter"));
    }
}

#[test]
fn lora_pass_geometry_matches_the_layer_structure() {
    let cfg = config_of();
    let shapes = shapes_of(&cfg);
    let synth = synth_adapter(&shapes, 0x9e01, Fill::Dense, true);
    let dir = write_adapter("geom", &synth);
    let lora = LagunaLora::from_peft_dir(&dir.0, &cfg).unwrap();
    assert_eq!(lora.layer_count(), shapes.num_layers);
    assert_eq!(
        lora.matched_modules(),
        synth.mods.len() - synth.moe_mlp_mods,
        "every projection this graph computes must be hooked; moe mlp modules have no site"
    );
    assert_eq!(
        lora.skipped_modules().len(),
        synth.moe_mlp_mods,
        "expert-stack and shared-expert mlp modules must land in skipped, not vanish"
    );
    for name in lora.skipped_modules() {
        assert!(
            name.contains(".mlp."),
            "unexpected skipped module {name}"
        );
    }
    assert!(
        lora.skipped_modules()
            .iter()
            .any(|n| n.contains(".mlp.experts.")),
        "an experts.N module name must be visible in the skip accounting"
    );
    assert!(
        lora.skipped_modules()
            .iter()
            .any(|n| n.contains(".mlp.shared_expert.")),
        "a shared_expert module name must be visible in the skip accounting"
    );
    assert_eq!(
        lora.total_pass_count(),
        expected_lora_passes(&shapes),
        "one pk shrink/expand pair per site: qkv/g/o on every layer, gate_up/down on dense"
    );
    assert_eq!(lora.total_pass_count(), 32);
    for li in 0..shapes.num_layers {
        let want = if shapes.layer(li).ffn_kind == FfnKind::Dense {
            10
        } else {
            6
        };
        assert_eq!(lora.layer_pass_count(li), want, "layer {li}");
    }
}

#[test]
fn a_real_adapter_shifts_logits_and_tracks_the_merged_weight_oracle() {
    if !have_gpu() {
        return;
    }
    let cfg = config_of();
    let shapes = shapes_of(&cfg);
    let weights = tiny_host_weights(&shapes, 0xa11ce);
    let synth = synth_adapter(&shapes, 0xd317a1, Fill::Dense, false);
    let dir = write_adapter("dense", &synth);
    let lora = LagunaLora::from_peft_dir(&dir.0, &cfg).unwrap();
    assert_eq!(lora.matched_modules(), synth.mods.len());

    let mut merged_weights = tiny_host_weights(&shapes, 0xa11ce);
    merge_adapter(&mut merged_weights, &shapes, &synth);

    let mut base = LagunaWgpu::new(cfg.clone(), &weights, MAX_SEQ).unwrap();
    let (t_base, l_base) = run_steps_full(&mut base);
    drop(base);

    let mut with_lora =
        LagunaWgpu::new_with_lora(cfg.clone(), &weights, MAX_SEQ, Some(&lora)).unwrap();
    let (t_lora, l_lora) = run_steps_full(&mut with_lora);
    drop(with_lora);

    let mut merged = LagunaWgpu::new(cfg.clone(), &merged_weights, MAX_SEQ).unwrap();
    let (t_merged, l_merged) = run_steps_full(&mut merged);
    drop(merged);

    for (i, step) in l_lora.iter().enumerate() {
        assert!(
            step.iter().all(|v| v.is_finite()),
            "step {i}: non-finite logits with lora attached"
        );
    }

    let mut nudged_weights = tiny_host_weights(&shapes, 0xa11ce);
    perturb_one_ulp(&mut nudged_weights);
    let mut nudged = LagunaWgpu::new(cfg, &nudged_weights, MAX_SEQ).unwrap();
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
        noise > 0.0,
        "the one-ulp control must move the logits, otherwise it calibrates nothing"
    );
    assert!(
        shift > 3.0 * noise,
        "the adapter's effect {shift} is not clear of the bf16 noise floor {noise}"
    );
    assert!(
        merged_shift > 3.0 * noise,
        "the merged-weight oracle must move the logits too, got {merged_shift} vs noise {noise}"
    );
    assert!(
        residual < 3.0 * noise,
        "runtime lora disagrees with the merged-weight oracle by {residual} against a one-ulp \
         perturbation floor of {noise}; the bound is 3x (not the e4b suite's 1.5x) because the \
         moe router consumes the perturbed hidden state and an expert flip amplifies sub-ulp \
         projection differences"
    );
    assert!(
        residual < 0.1 * shift,
        "the lora-vs-merged residual {residual} must stay an order of magnitude under the \
         adapter's own effect {shift}"
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

#[test]
#[ignore]
fn real_laguna_adapter_decode_ab_perfb() {
    if std::env::var("NV_LAGUNA_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping: set NV_LAGUNA_WGPU_TEST=1 to run");
        return;
    }
    if !have_gpu() {
        return;
    }
    let Some(dir) = hub_snapshot::dir_from_env_or_hub(
        "NV_LAGUNA_DIR",
        "poolside/Laguna-XS-2.1-NVFP4",
        &["config.json", "*.safetensors"],
    ) else {
        eprintln!("SKIP: no laguna snapshot; set NV_LAGUNA_DIR");
        return;
    };
    let cfg = LagunaConfig::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let max_seq: usize = std::env::var("NV_LAGUNA_WGPU_MAXSEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let n_new: usize = std::env::var("NV_LAGUNA_WGPU_NEW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let shapes = LagunaShapes::derive(&cfg, max_seq).unwrap();
    let synth = synth_adapter(&shapes, 0xd317a1, Fill::Dense, false);
    let sdir = write_adapter("perfab", &synth);
    let lora = LagunaLora::from_peft_dir(&sdir.0, &cfg).unwrap();
    eprintln!(
        "adapter: rank {} matched {} modules, +{} passes/token",
        lora.rank(),
        lora.matched_modules(),
        lora.total_pass_count()
    );

    let prompt: Vec<u32> = (0..16u32).map(|i| 2 + (i * 97) % 1000).collect();
    let vocab = shapes.vocab_size as u32;
    let measure = |label: &str, m: &mut LagunaWgpu| -> f64 {
        for t in &prompt {
            m.decode_step(*t % vocab).unwrap();
        }
        let t0 = std::time::Instant::now();
        let mut next = 7u32;
        for _ in 0..n_new {
            next = m.decode_step(next % vocab).unwrap();
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / n_new as f64;
        eprintln!(
            "LORA-LAGUNA-AB arm={label} passes={} ms_per_tok={ms:.3}",
            m.pass_count()
        );
        ms
    };

    let mut base = LagunaWgpu::from_loader(cfg.clone(), &loader, max_seq).unwrap();
    let base_passes = base.pass_count();
    let base_ms = measure("base", &mut base);
    drop(base);

    let mut adapted =
        LagunaWgpu::from_loader_with_lora(cfg, &loader, max_seq, Some(&lora)).unwrap();
    assert_eq!(
        adapted.pass_count() - base_passes,
        lora.total_pass_count(),
        "adapter decode pass delta must be the pk ledger's 2 passes/site"
    );
    let lora_ms = measure("adapter", &mut adapted);
    eprintln!(
        "LORA-LAGUNA-AB summary base_ms={base_ms:.3} adapter_ms={lora_ms:.3} \
         delta_ms={:.3} lora_passes={}",
        lora_ms - base_ms,
        lora.total_pass_count()
    );
}

#[test]
fn an_adapter_that_matches_no_projection_is_rejected() {
    let cfg = config_of();
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
        moe_mlp_mods: 0,
    };
    let dir = write_adapter("nomatch", &synth);
    let msg = match LagunaLora::from_peft_dir(&dir.0, &cfg) {
        Ok(l) => panic!(
            "an adapter with no laguna module must be rejected, got {} matches",
            l.matched_modules()
        ),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        msg.contains("matched no laguna projection"),
        "unexpected error: {msg}"
    );
}
