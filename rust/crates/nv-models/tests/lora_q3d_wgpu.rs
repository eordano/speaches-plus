#![cfg(feature = "wgpu")]

mod common;
use common::have_gpu;
use common::q3d_lora_synth::{
    synth_adapter, write_adapter, Fill, Scratch, Synth, PREFIX, SYNTH_SCALING,
};
use common::tiny_config_q3d_mixed_layers as tiny_config;
use common::tiny_weights_q3d as tiny_weights;
use half::bf16;

use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_moe::LayerType;
use q3d::{Q3dLora, Qwen3_5DenseWgpu};

fn load_lora(dir: &Scratch, cfg: &nv_models::qwen3_5_moe::Qwen3_5DenseConfig) -> Q3dLora {
    Q3dLora::from_peft_dir(&dir.0, cfg).expect("build q3d lora sites")
}

const STEP_TOKENS: [u32; 6] = [3, 11, 5, 40, 2, 19];

fn run_steps(m: &mut Qwen3_5DenseWgpu) -> Vec<Vec<f32>> {
    STEP_TOKENS
        .iter()
        .map(|t| m.decode_step_logits(*t).expect("decode step").1)
        .collect()
}

fn assert_bit_identical(a: &[Vec<f32>], b: &[Vec<f32>], what: &str) {
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.len(), y.len(), "{what}: step {i} logit count");
        for (j, (p, q)) in x.iter().zip(y.iter()).enumerate() {
            assert!(
                p.to_bits() == q.to_bits(),
                "{what}: step {i} logit {j} differs: {p} vs {q}"
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

#[test]
fn the_pk_tiles_shader_validates_without_a_gpu() {
    let src = nv_kernels::wgpu_backend::compose(include_str!(
        "../../nv-kernels/wgsl/lora_pk_tiles.wgsl"
    ));
    for entry in ["lora_pk_shrink_tiles", "lora_pk_expand_tiles"] {
        assert!(src.contains(&format!("fn {entry}(")), "{entry} entry point vanished");
    }
    let module = naga::front::wgsl::parse_str(&src)
        .unwrap_or_else(|e| panic!("lora_pk_tiles wgsl parse: {}", e.message()));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("lora_pk_tiles wgsl validate: {e:?}"));
}

fn bf16_ulp_key(bits: u16) -> i32 {
    let mag = (bits & 0x7fff) as i32;
    if bits & 0x8000 != 0 {
        -mag
    } else {
        mag
    }
}

fn probe_rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((s >> 33) as u32 as f64 / u32::MAX as f64) as f32 - 0.5) * 2.0
    }
}

#[test]
fn real_dimension_pk_site_matches_the_legacy_path_and_an_f64_oracle() {
    if !have_gpu() {
        return;
    }
    let cases: [(usize, Vec<usize>, Vec<usize>, usize, usize); 3] = [
        (5120, vec![12288, 1024, 1024], vec![12288, 1024, 1024], 1, 8),
        (5120, vec![10240, 6144, 48, 48], vec![10240, 6144, 96], 1, 8),
        (17408, vec![5120], vec![5120], 4, 8),
    ];
    for (case, (k, widths, segs, m, rank)) in cases.iter().enumerate() {
        let mut rng = probe_rng(0x9a3d_ab00 + case as u64);
        let enc = |v: f32| bf16::from_f32(v).to_bits();
        let x: Vec<u16> = (0..m * k).map(|_| enc(rng())).collect();
        let a: Vec<u16> = (0..widths.len() * rank * k).map(|_| enc(rng() * 0.02)).collect();
        let b_len: usize = widths.iter().map(|w| w * rank).sum();
        let b: Vec<u16> = (0..b_len).map(|_| enc(rng() * 0.25)).collect();
        let y0: Vec<Vec<u16>> = segs
            .iter()
            .map(|w| (0..m * w).map(|_| enc(rng())).collect())
            .collect();

        let mut y_legacy = y0.clone();
        nv_models::wgpu_lora::lora_site_probe(
            &x, &a, &b, &mut y_legacy, widths, segs, *m, *rank, *k,
        )
        .expect("legacy probe");
        let mut y_pk = y0.clone();
        nv_models::wgpu_lora::lora_site_probe_pk(
            &x, &a, &b, &mut y_pk, widths, segs, *m, *rank, *k,
        )
        .expect("pk probe");

        let dec = |u: u16| bf16::from_bits(u).to_f32();
        let mut slice_col_starts = Vec::new();
        let mut acc = 0usize;
        for w in widths {
            slice_col_starts.push(acc);
            acc += w;
        }
        let mut oracle: Vec<Vec<u16>> = y0.clone();
        for row in 0..*m {
            let mut h = vec![0f64; widths.len() * rank];
            for (si, _) in widths.iter().enumerate() {
                for r in 0..*rank {
                    let a_base = (si * rank + r) * k;
                    let mut s = 0f64;
                    for kk in 0..*k {
                        s += dec(x[row * k + kk]) as f64 * dec(a[a_base + kk]) as f64;
                    }
                    h[si * rank + r] = s;
                }
            }
            let mut b_off = 0usize;
            for (si, w) in widths.iter().enumerate() {
                for col in 0..*w {
                    let mut d = 0f64;
                    for r in 0..*rank {
                        d += h[si * rank + r] * dec(b[b_off + col * rank + r]) as f64;
                    }
                    let gcol = slice_col_starts[si] + col;
                    let mut seg = 0usize;
                    let mut seg_start = 0usize;
                    while seg_start + segs[seg] <= gcol {
                        seg_start += segs[seg];
                        seg += 1;
                    }
                    let idx = row * segs[seg] + (gcol - seg_start);
                    let base = dec(oracle[seg][idx]);
                    oracle[seg][idx] = bf16::from_f32(base + d as f32).to_bits();
                }
                b_off += w * rank;
            }
        }

        for (seg, w) in segs.iter().enumerate() {
            for j in 0..(m * w) {
                let l = bf16_ulp_key(y_legacy[seg][j]);
                let p = bf16_ulp_key(y_pk[seg][j]);
                let o = bf16_ulp_key(oracle[seg][j]);
                assert!(
                    (l - p).abs() <= 1,
                    "case {case} seg {seg} elem {j}: legacy {l} vs pk {p} beyond 1 bf16 ulp; \
                     the k-tiled kernel is not an order-only change of the legacy site"
                );
                assert!(
                    (p - o).abs() <= 2,
                    "case {case} seg {seg} elem {j}: pk {p} vs f64 oracle {o} beyond 2 bf16 \
                     ulps; a tile-boundary or routing bug drops or doubles delta mass"
                );
            }
        }
    }
}

#[test]
fn compiling_lora_in_without_an_adapter_leaves_the_graph_identical() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x9a3d_0001);
    let mut plain = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32).expect("plain build");
    let mut with_none =
        Qwen3_5DenseWgpu::new_with_lora(cfg.clone(), &hw, 32, None).expect("lora-none build");
    let a = run_steps(&mut plain);
    let b = run_steps(&mut with_none);
    assert_bit_identical(&a, &b, "no-adapter graph");
}

#[test]
fn a_zero_adapter_leaves_logits_bit_identical_and_keeps_the_verify_graph_choice() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x9a3d_0002);
    let mut plain = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32).expect("plain build");
    let want = run_steps(&mut plain);
    let plain_chunk = plain.prefill_chunk_len();
    let plain_dedicated = plain.verify_chain_rides_dedicated_graph();
    for (tag, fill) in [("zerob", Fill::ZeroB), ("zeroa", Fill::ZeroA)] {
        let synth = synth_adapter(&cfg, 0x5eed ^ tag.len() as u64, fill, None);
        let dir = write_adapter(tag, &synth);
        let lora = load_lora(&dir, &cfg);
        assert!(lora.matched_modules() > 0, "{tag}: no modules matched");
        let mut m =
            Qwen3_5DenseWgpu::new_with_lora(cfg.clone(), &hw, 32, Some(&lora)).expect("build");
        assert!(m.lora_resident());
        assert_eq!(
            m.prefill_chunk_len(),
            plain_chunk,
            "{tag}: the chunked prefill graph carries lora sites and must build at full width"
        );
        assert_eq!(
            m.verify_chain_rides_dedicated_graph(),
            plain_dedicated,
            "{tag}: the dedicated verify graph carries lora sites, so its presence must not \
             depend on the adapter"
        );
        let got = run_steps(&mut m);
        assert_bit_identical(&want, &got, &format!("{tag} adapter"));
    }
}

#[test]
fn lora_pass_geometry_matches_the_layer_structure() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x9a3d_0003);
    let plain = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32).expect("plain build");
    let synth = synth_adapter(&cfg, 0xa11a, Fill::Coarse, None);
    let dir = write_adapter("geom", &synth);
    let lora = load_lora(&dir, &cfg);
    let attn_layers = cfg
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::FullAttention))
        .count();
    let linear_layers = cfg.num_hidden_layers - attn_layers;
    let sites = attn_layers * 2 + cfg.num_hidden_layers * 2 + linear_layers * 2;
    assert_eq!(
        lora.total_pass_count(),
        2 * sites,
        "each site is one k-tiled shrink pass writing per-tile partials plus one packed-write \
         expand pass covering all of its destination segments; the deltanet in-projections \
         share one site because they share the normed activation"
    );
    let with = Qwen3_5DenseWgpu::new_with_lora(cfg.clone(), &hw, 32, Some(&lora)).expect("build");
    assert_eq!(
        with.pass_count(),
        plain.pass_count() + lora.total_pass_count(),
        "the decode list must grow by exactly the lora pass count"
    );
    assert!(
        plain.prefill_pass_count() > 0,
        "this gate needs the chunked prefill graph; do not run it with NV_WGPU_PREFILL_M=0"
    );
    assert_eq!(
        with.prefill_pass_count(),
        plain.prefill_pass_count() + lora.total_pass_count(),
        "the chunked prefill list must grow by exactly the lora pass count: every projection \
         with a decode site needs the matching prefill site, or prefilled KV rows embed \
         un-adapted projections"
    );
}

#[test]
fn a_real_adapter_shifts_logits() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x9a3d_0004);
    let mut plain = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32).expect("plain build");
    let base = run_steps(&mut plain);
    let synth = synth_adapter(&cfg, 0xd1ff, Fill::Coarse, None);
    let dir = write_adapter("real", &synth);
    let lora = load_lora(&dir, &cfg);
    let mut m =
        Qwen3_5DenseWgpu::new_with_lora(cfg.clone(), &hw, 32, Some(&lora)).expect("build");
    let got = run_steps(&mut m);
    let diff = max_abs_diff(&base, &got);
    assert!(
        diff > 1e-3,
        "a nonzero adapter left logits unchanged (max abs diff {diff}); the sites are wired \
         but inert"
    );
}

#[test]
fn a_value_lane_only_q_adapter_is_rejected_under_attn_output_gate() {
    let cfg = tiny_config();
    assert!(cfg.attn_output_gate, "fixture must exercise the gate quirk");
    let q_half = cfg.num_attention_heads * cfg.head_dim;
    let synth = synth_adapter(&cfg, 0xbadb, Fill::Coarse, Some(q_half));
    let dir = write_adapter("gatereject", &synth);
    let err = Q3dLora::from_peft_dir(&dir.0, &cfg)
        .err()
        .expect("a half-width q adapter must be refused");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("value|gate"),
        "refusal must name the gate-lane hazard, got: {msg}"
    );
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
            w[idx] = bf16::from_f32(base + d).to_bits();
        }
    }
}

fn bf16_of(lin: &mut q3d::HostDenseLin) -> (&mut Vec<u16>, usize) {
    let q3d::HostDenseLin::Bf16(l) = lin else {
        panic!("the tiny q3d fixture stores every projection as bf16")
    };
    (&mut l.w, l.k)
}

fn merge_lin(lin: &mut q3d::HostDenseLin, name: String, synth: &Synth) {
    if let Some((a, b)) = synth.mods.get(&name) {
        let (w, k) = bf16_of(lin);
        merge_rows(w, k, 0, a, b, synth.rank);
    }
}

fn merge_bf16(
    lin: &mut nv_models::qwen3_5_moe_wgpu::HostBf16Lin,
    row_off: usize,
    name: String,
    synth: &Synth,
) {
    if let Some((a, b)) = synth.mods.get(&name) {
        merge_rows(&mut lin.w, lin.k, row_off, a, b, synth.rank);
    }
}

fn merge_adapter(
    weights: &mut q3d::HostDenseWeights,
    cfg: &nv_models::qwen3_5_moe::Qwen3_5DenseConfig,
    synth: &Synth,
) {
    for (li, layer) in weights.layers.iter_mut().enumerate() {
        let p = format!("{PREFIX}.{li}");
        assert!(
            matches!(cfg.layer_types[li], LayerType::FullAttention)
                == matches!(layer.mixer, q3d::HostDenseMixer::Attn(_)),
            "fixture layer kinds must match the config"
        );
        match &mut layer.mixer {
            q3d::HostDenseMixer::Attn(a) => {
                merge_lin(&mut a.q, format!("{p}.self_attn.q_proj"), synth);
                merge_lin(&mut a.k, format!("{p}.self_attn.k_proj"), synth);
                merge_lin(&mut a.v, format!("{p}.self_attn.v_proj"), synth);
                merge_lin(&mut a.o, format!("{p}.self_attn.o_proj"), synth);
            }
            q3d::HostDenseMixer::Delta(d) => {
                let n_vh = d.in_proj_ab.n / 2;
                merge_bf16(
                    &mut d.in_proj_qkv,
                    0,
                    format!("{p}.linear_attn.in_proj_qkv"),
                    synth,
                );
                merge_bf16(
                    &mut d.in_proj_z,
                    0,
                    format!("{p}.linear_attn.in_proj_z"),
                    synth,
                );
                merge_bf16(
                    &mut d.in_proj_ab,
                    0,
                    format!("{p}.linear_attn.in_proj_a"),
                    synth,
                );
                merge_bf16(
                    &mut d.in_proj_ab,
                    n_vh,
                    format!("{p}.linear_attn.in_proj_b"),
                    synth,
                );
                merge_bf16(
                    &mut d.out_proj,
                    0,
                    format!("{p}.linear_attn.out_proj"),
                    synth,
                );
            }
        }
        merge_lin(&mut layer.mlp.gate, format!("{p}.mlp.gate_proj"), synth);
        merge_lin(&mut layer.mlp.up, format!("{p}.mlp.up_proj"), synth);
        merge_lin(&mut layer.mlp.down, format!("{p}.mlp.down_proj"), synth);
    }
}

fn flip_lsb(w: &mut [u16]) {
    for v in w.iter_mut() {
        *v ^= 1;
    }
}

fn perturb_one_ulp(weights: &mut q3d::HostDenseWeights) {
    for layer in weights.layers.iter_mut() {
        match &mut layer.mixer {
            q3d::HostDenseMixer::Attn(a) => {
                flip_lsb(bf16_of(&mut a.q).0);
                flip_lsb(bf16_of(&mut a.k).0);
                flip_lsb(bf16_of(&mut a.v).0);
                flip_lsb(bf16_of(&mut a.o).0);
            }
            q3d::HostDenseMixer::Delta(d) => {
                flip_lsb(&mut d.in_proj_qkv.w);
                flip_lsb(&mut d.in_proj_z.w);
                flip_lsb(&mut d.in_proj_ab.w);
                flip_lsb(&mut d.out_proj.w);
            }
        }
        flip_lsb(bf16_of(&mut layer.mlp.gate).0);
        flip_lsb(bf16_of(&mut layer.mlp.up).0);
        flip_lsb(bf16_of(&mut layer.mlp.down).0);
    }
}

const PF_MAX_SEQ: usize = 64;

fn prompt_then_steps(m: &mut Qwen3_5DenseWgpu, prompt: &[u32]) -> Vec<Vec<f32>> {
    let done = m.prefill_tokens(prompt).expect("chunked prefill");
    for t in &prompt[done..] {
        m.prefill_step(*t).expect("prefill tail step");
    }
    run_steps(m)
}

fn pf_prompt() -> Vec<u32> {
    (0..40u32).map(|i| (i * 17 + 3) % 64).collect()
}

#[test]
fn multi_chunk_prefill_honours_the_adapter_and_a_zero_adapter_stays_bit_identical() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x9a3d_0005);
    let prompt = pf_prompt();

    let mut base = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, PF_MAX_SEQ).expect("plain build");
    assert!(
        base.prefill_chunk_len() >= 2,
        "this gate needs the chunked prefill graph; do not run it with NV_WGPU_PREFILL_M=0"
    );
    assert!(
        prompt.len() > 2 * base.prefill_chunk_len(),
        "the prompt must span multiple chunks to prove chunk-boundary KV correctness"
    );
    let l_base = prompt_then_steps(&mut base, &prompt);
    drop(base);

    let zero = synth_adapter(&cfg, 0x2e40, Fill::ZeroB, None);
    let zdir = write_adapter("pfzero", &zero);
    let zlora = load_lora(&zdir, &cfg);
    let mut m0 =
        Qwen3_5DenseWgpu::new_with_lora(cfg.clone(), &hw, PF_MAX_SEQ, Some(&zlora)).expect("build");
    assert!(
        m0.prefill_chunk_len() >= 2,
        "an adapter engine must keep the chunked prefill graph"
    );
    let l_zero = prompt_then_steps(&mut m0, &prompt);
    drop(m0);
    assert_bit_identical(&l_base, &l_zero, "zero adapter through multi-chunk prefill");

    let synth = synth_adapter(&cfg, 0xd317a1, Fill::Coarse, None);
    let dir = write_adapter("pfcoarse", &synth);
    let lora = load_lora(&dir, &cfg);
    let mut m1 =
        Qwen3_5DenseWgpu::new_with_lora(cfg.clone(), &hw, PF_MAX_SEQ, Some(&lora)).expect("build");
    let l_lora = prompt_then_steps(&mut m1, &prompt);
    drop(m1);

    let mut merged_weights = tiny_weights(&cfg, 0x9a3d_0005);
    merge_adapter(&mut merged_weights, &cfg, &synth);
    let mut m2 = Qwen3_5DenseWgpu::new(cfg.clone(), &merged_weights, PF_MAX_SEQ).expect("build");
    let l_merged = prompt_then_steps(&mut m2, &prompt);
    drop(m2);

    let mut nudged_weights = tiny_weights(&cfg, 0x9a3d_0005);
    perturb_one_ulp(&mut nudged_weights);
    let mut m3 = Qwen3_5DenseWgpu::new(cfg.clone(), &nudged_weights, PF_MAX_SEQ).expect("build");
    let l_nudged = prompt_then_steps(&mut m3, &prompt);
    drop(m3);

    let shift = max_abs_diff(&l_base, &l_lora);
    let residual = max_abs_diff(&l_merged, &l_lora);
    let noise = max_abs_diff(&l_base, &l_nudged);
    eprintln!(
        "pf: max |lora - base| = {shift}, max |lora - merged| = {residual}, one-ulp noise \
         floor = {noise}"
    );
    assert!(
        shift > 3.0 * noise,
        "a nonzero adapter left multi-chunk prefill within the {noise} noise floor (shift \
         {shift}); the prefill sites are wired but inert"
    );
    assert!(
        residual < 2.0 * noise,
        "prefill runtime lora disagrees with the merged-weight oracle by {residual} vs the \
         {noise} one-ulp noise floor (bound 2x: the deltanet gating nonlinearity amplifies \
         merge rounding to ~1.5x, while a missing or mis-wired site measures 15x+); un-adapted \
         KV or recurrent rows are what this catches"
    );
}
