#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use common::LcgCentered0p1Shift33 as Lcg;
use common::TINY_E4B_CONFIG;

use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_e4b_wgpu as e4b;
use nv_models::gemma4_e4b_wgpu::{E4bHostLayer, E4bHostWeights, Gemma4E4bWgpu, HostLin};

static ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE:
    std::sync::Mutex<()> = std::sync::Mutex::new(());

struct EnvGuard {
    _serialized: std::sync::MutexGuard<'static, ()>,
    keys: Vec<&'static str>,
}

fn env_on(pairs: &[(&'static str, &str)]) -> EnvGuard {
    let g = ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let mut keys = Vec::new();
    for (k, v) in pairs {
        std::env::set_var(k, v);
        keys.push(*k);
    }
    EnvGuard {
        _serialized: g,
        keys,
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for k in &self.keys {
            std::env::remove_var(k);
        }
    }
}

fn tiny_e4b_host_weights(config: &Gemma4Config, seed: u64) -> E4bHostWeights {
    let mut rng = Lcg(seed);
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let hpl = config.hidden_size_per_layer_input;
    let n_q = config.num_attention_heads;

    let mut layers = Vec::new();
    for i in 0..config.num_hidden_layers {
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
            None => rng.bf16_vec_around_one(hd),
        };
        layers.push(E4bHostLayer {
            kind,
            kv_source,
            input_ln: rng.bf16_vec_around_one(hidden),
            post_attn_ln: rng.bf16_vec_around_one(hidden),
            pre_ff_ln: rng.bf16_vec_around_one(hidden),
            post_ff_ln: rng.bf16_vec_around_one(hidden),
            post_per_layer_input_norm: rng.bf16_vec_around_one(hidden),
            q_norm: rng.bf16_vec_around_one(hd),
            k_norm,
            layer_scalar: 0.9,
            has_v,
            qkv: HostLin::new(rng.bf16_vec(qkv_rows * hidden), qkv_rows, hidden),
            o: HostLin::new(rng.bf16_vec(hidden * q_dim), hidden, q_dim),
            gate_up: HostLin::new(rng.bf16_vec(2 * inter * hidden), 2 * inter, hidden),
            down: HostLin::new(rng.bf16_vec(hidden * inter), hidden, inter),
            per_layer_input_gate: HostLin::new(rng.bf16_vec(hpl * hidden), hpl, hidden),
            per_layer_projection: HostLin::new(rng.bf16_vec(hidden * hpl), hidden, hpl),
        });
    }

    let ple_row = config.num_hidden_layers * hpl;
    E4bHostWeights {
        embed: rng.bf16_vec(config.vocab_size * hidden),
        embed_per_layer: rng.bf16_vec(config.vocab_size_per_layer() * ple_row),
        per_layer_model_projection: HostLin::new(rng.bf16_vec(ple_row * hidden), ple_row, hidden),
        per_layer_projection_norm: rng.bf16_vec_around_one(hpl),
        final_norm: rng.bf16_vec_around_one(hidden),
        layers,
    }
}

const TINY_SHALLOW_KNEE_8: &str = "8";
const TINY_BAND2_KNEE_24: &str = "24";
const TINY_MAX_SEQ_64: usize = 64;

#[test]
fn the_measured_band_boundaries_are_a_pure_function_of_context_tokens_and_never_move_below_a_knee() {
    let shallow = 16u32;
    let band1 = Some((2048u32, 64u32));
    let band2 = Some((4096u32, 94u32));
    for tokens in [1u32, 256, 2048] {
        assert_eq!(
            e4b::deep_band_splits_for_context_tokens(tokens, shallow, band1, band2),
            shallow,
            "at or below the shallow knee the gate must return the pinned split count unchanged \
             at {tokens} tokens; {}",
            e4b::DEEP2_BAND_RULE
        );
    }
    for (tokens, want) in [(2049u32, 64u32), (4096, 64), (4097, 94), (122880, 94)] {
        assert_eq!(
            e4b::deep_band_splits_for_context_tokens(tokens, shallow, band1, band2),
            want,
            "the band at {tokens} tokens moved off the measured ladder; {}",
            e4b::DEEP2_BAND_RULE
        );
    }
    assert_eq!(
        e4b::deep_band_splits_for_context_tokens(122880, shallow, band1, None),
        64,
        "with the second band absent every depth above the first knee must keep the shipped 64, \
         or a zero-env build silently changes its grid"
    );
}

#[test]
fn every_gate_env_is_default_off_so_a_zero_env_build_keeps_one_deep_band_and_its_scratch() {
    let _g = env_on(&[]);
    assert!(
        !e4b::deep_depthgate(),
        "{} must be opt-in: this is an fp-reorder arm and a performance default is never flipped",
        e4b::E4B_DEEP_DEPTHGATE_ENV
    );
    assert!(
        e4b::deep2_split_arm().is_none(),
        "the second deep band must not exist with {} unset",
        e4b::E4B_DEEP_DEPTHGATE_ENV
    );
    assert!(
        e4b::sliding_splits_cap_a_sliding_layer_scans_at_most_its_window_so_its_band_saturates_there(
            512
        )
        .is_none(),
        "keying the sliding layers on min(total, window) is part of the gate, not a new default"
    );
    assert_eq!(
        e4b::scratch_splits(),
        e4b::deep_splits().max(e4b::decode_splits()).max(e4b::FLASH_SPLITS),
        "an unarmed build must provision exactly the split-k rows it provisioned before the gate"
    );
}

#[test]
fn the_scratch_provision_covers_every_band_the_gate_can_pick_or_a_deep_step_writes_out_of_bounds() {
    let _g = env_on(&[(e4b::E4B_DEEP_DEPTHGATE_ENV, "1")]);
    let provisioned = e4b::scratch_splits();
    let (band1, band2) = (e4b::deep_split_arm(), e4b::deep2_split_arm());
    assert!(
        band2.is_some(),
        "arming {} must produce a second band, or this gate is inert",
        e4b::E4B_DEEP_DEPTHGATE_ENV
    );
    for tokens in [1u32, 2049, 4097, 32768, 131072] {
        assert!(
            e4b::deep_band_splits_for_context_tokens(tokens, e4b::decode_splits(), band1, band2)
                <= provisioned,
            "at {tokens} tokens the gate picks more splits than the {provisioned} scratch rows \
             provisioned; stage1 indexes the split-k scratch by split and would write past it"
        );
    }
    assert_eq!(
        provisioned,
        e4b::deep2_splits(),
        "the widest band the gate can pick is the second one, so it sets the scratch width"
    );
}

#[test]
fn naga_validates_the_second_band_stage2_at_every_split_count_the_gate_can_bake() {
    for splits in [16u32, 47, 64, 80, 94, 110, 128, 141, 256] {
        let src = e4b::e4b_flash2_deep_module_source(splits);
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("band-2 stage2 at {splits} splits does not parse: {e:?}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("band-2 stage2 at {splits} splits fails validation: {e:?}"));
        assert!(
            src.contains(&format!("let splits = {splits}u;")),
            "the generated stage2 must bake the band's own split count: e4b reduces the split-k \
             partials with a hand-sized loop, so a band whose stage2 was baked at another count \
             merges the wrong number of partials"
        );
    }
}

fn tiny_model(pairs: &[(&'static str, &str)]) -> Option<(EnvGuard, Gemma4E4bWgpu)> {
    ctx_or_skip()?;
    let g = env_on(pairs);
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let weights = tiny_e4b_host_weights(&config, 0x5eed);
    let m = Gemma4E4bWgpu::new(
        Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
        &weights,
        TINY_MAX_SEQ_64,
    )
    .expect("tiny e4b wgpu build");
    Some((g, m))
}

#[test]
fn the_gate_off_grid_never_moves_past_the_shipped_deep_band_and_the_gate_on_grid_follows_the_bands()
{
    let Some((_g, m)) = tiny_model(&[
        (e4b::E4B_SPLIT_DEPTH_ENV, TINY_SHALLOW_KNEE_8),
        (e4b::E4B_DEEP_SPLITS_ENV, "24"),
    ]) else {
        return;
    };
    let shallow = e4b::decode_splits();
    for (total, want_full) in [(4usize, shallow), (12, 24), (40, 24)] {
        let (full, sliding) = m.flash1_grid_y_as_encoded_for_total(total);
        assert!(!full.is_empty() && !sliding.is_empty(), "tiny config has both layer kinds");
        assert!(
            full.iter().all(|&y| y == want_full) && sliding.iter().all(|&y| y == want_full),
            "gate off at total {total}: full {full:?} sliding {sliding:?} must both be \
             {want_full}, because an unarmed build keeps one deep band and gates neither kind \
             separately"
        );
    }
    drop(_g);

    let Some((_g2, m2)) = tiny_model(&[
        (e4b::E4B_SPLIT_DEPTH_ENV, TINY_SHALLOW_KNEE_8),
        (e4b::E4B_DEEP_SPLITS_ENV, "24"),
        (e4b::E4B_DEEP_DEPTHGATE_ENV, "1"),
        (e4b::E4B_DEEP2_KNEE_ENV, TINY_BAND2_KNEE_24),
        (e4b::E4B_DEEP2_SPLITS_ENV, "47"),
    ]) else {
        return;
    };
    for (total, want_full) in [(4usize, shallow), (12, 24), (40, 47)] {
        let (full, sliding) = m2.flash1_grid_y_as_encoded_for_total(total);
        assert!(
            full.iter().all(|&y| y == want_full),
            "gate on at total {total}: full-attention grid.y {full:?} must be {want_full}; {}",
            e4b::DEEP2_BAND_RULE
        );
        assert!(
            sliding.iter().all(|&y| y == shallow),
            "gate on at total {total}: sliding grid.y {sliding:?} must stay at the shallow \
             {shallow}, because a sliding layer scans at most its window and the gate keys it on \
             min(total, window)"
        );
    }
    let (_, band1, band2) = m2.split_bands_shallow_deep_deep2();
    assert_eq!(band1, Some((8, 24)));
    assert_eq!(band2, Some((24, 47)));
}

#[test]
fn the_sliding_cap_is_the_band_the_window_itself_picks_and_never_a_band_the_window_cannot_reach() {
    let shallow = 16u32;
    let band1 = Some((2048u32, 64u32));
    let band2 = Some((4096u32, 94u32));
    let cap = |w: u32| e4b::deep_band_splits_for_context_tokens(w, shallow, band1, band2);
    assert_eq!(
        cap(512),
        shallow,
        "gemma4-E4B's 512-position sliding window never reaches the 2048 deep knee, so every \
         sliding layer keeps the shallow split count at every depth; splitting a 512-position \
         scan 94 ways leaves five positions per split"
    );
    for (window, want) in [(2048u32, shallow), (2049, 64), (4096, 64), (8192, 94)] {
        assert_eq!(
            cap(window),
            want,
            "a model whose window is {window} must cap its sliding layers at the band {window} \
             positions themselves pick, not at the band the full-attention layers reached"
        );
    }
    for tokens in [1u32, 600, 4096, 131072] {
        let full = e4b::deep_band_splits_for_context_tokens(tokens, shallow, band1, band2);
        assert!(
            cap(512).min(full) <= full,
            "the sliding band at {tokens} tokens must never exceed the full-attention band, or \
             the twin the builder baked and the uniform the step writes disagree"
        );
    }
}

#[test]
fn arming_the_gate_below_its_knees_is_bit_for_bit_the_unarmed_decode() {
    let steps: Vec<u32> = vec![7, 300, 42, 511, 0, 13, 99, 250];
    let run = |pairs: &[(&'static str, &str)]| -> Option<Vec<Vec<f32>>> {
        let (_g, mut m) = tiny_model(pairs)?;
        Some(
            steps
                .iter()
                .map(|t| m.decode_step_logits(*t).unwrap().1)
                .collect(),
        )
    };
    let base = match run(&[(e4b::E4B_SPLIT_DEPTH_ENV, "4096")]) {
        Some(v) => v,
        None => return,
    };
    let armed = run(&[
        (e4b::E4B_SPLIT_DEPTH_ENV, "4096"),
        (e4b::E4B_DEEP_DEPTHGATE_ENV, "1"),
        (e4b::E4B_DEEP2_KNEE_ENV, "8192"),
    ])
    .expect("the adapter answered once already");
    let mut diff = 0usize;
    for (a, b) in base.iter().zip(armed.iter()) {
        for (x, y) in a.iter().zip(b.iter()) {
            if x.to_bits() != y.to_bits() {
                diff += 1;
            }
        }
    }
    assert_eq!(
        diff, 0,
        "with both knees above every position the harness reaches, the armed gate must be the \
         unarmed decode bit for bit; a difference here means the gate changed the reduction order \
         on a path it was supposed to leave alone"
    );
}
