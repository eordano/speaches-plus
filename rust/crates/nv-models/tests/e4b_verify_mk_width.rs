#![cfg(feature = "wgpu")]

mod common;
use common::TINY_E4B_CONFIG;
use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_e4b_wgpu::{E4bHostLayer, E4bHostWeights, Gemma4E4bWgpu, HostLin};
use common::LcgShift33Centered0p1 as Lcg;

const VERIFY_M_ENV: &str = "NV_E4B_WGPU_VERIFY_M";

fn ctx() -> &'static nv_kernels::wgpu_backend::WgpuContext {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(c) => c,
        Err(e) => panic!("no wgpu adapter: {e}"),
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

fn tiny_model(max_seq: usize) -> Gemma4E4bWgpu {
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let weights = tiny_e4b_host_weights(&config, 0x5eed);
    Gemma4E4bWgpu::new(config, &weights, max_seq).unwrap()
}

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn with_verify_m<T>(v: Option<usize>, f: impl FnOnce() -> T) -> T {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    match v {
        Some(v) => std::env::set_var(VERIFY_M_ENV, v.to_string()),
        None => std::env::remove_var(VERIFY_M_ENV),
    }
    let out = f();
    std::env::remove_var(VERIFY_M_ENV);
    out
}

fn compare(a: &mut Gemma4E4bWgpu, b: &mut Gemma4E4bWgpu, cap: usize, rounds: usize) {
    let vocab = a.config().vocab_size;
    let prefix: Vec<u32> = (0..21u32).map(|i| (i * 37 + 5) % vocab as u32).collect();
    for &t in &prefix {
        a.decode_step(t).unwrap();
        b.decode_step(t).unwrap();
    }
    let mut rng = Lcg(0xc0ffee);
    let mut hidden_rows = 0usize;
    for round in 0..rounds {
        let k = 1 + round % cap;
        let batch: Vec<u32> = (0..k).map(|_| rng.token(vocab)).collect();
        let va = a.verify_chain(&batch).unwrap();
        let vb = b.verify_chain(&batch).unwrap();
        assert_eq!(va, vb, "round {round} (mb={k}): argmax stream differs");
        for row in 0..k {
            let ha = a.verify_hidden_row(row).unwrap();
            let hb = b.verify_hidden_row(row).unwrap();
            assert_eq!(
                ha.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                hb.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "round {round} row {row}: hidden state is not bit-identical"
            );
            hidden_rows += 1;
        }
        let commit = 1 + (round * 7) % k;
        a.advance(commit).unwrap();
        b.advance(commit).unwrap();
    }
    eprintln!("  {rounds} rounds, {hidden_rows} hidden rows compared bit-for-bit");
}

#[test]
fn narrow_verify_width_is_bit_exact_against_the_prefill_width() {
    eprintln!("adapter: {}", ctx().summary());
    let mut wide = with_verify_m(None, || tiny_model(64));
    let wide_cap = wide.verify_max_rows();
    assert!(
        wide_cap >= 3,
        "verify_max_rows {wide_cap} too small to narrow"
    );
    for narrow_m in [1usize, 2, 3] {
        let mut narrow = with_verify_m(Some(narrow_m), || tiny_model(64));
        assert_eq!(
            narrow.verify_max_rows(),
            narrow_m,
            "verify width did not reach verify_max_rows"
        );
        assert_eq!(
            narrow.prefill_chunk_len(),
            wide.prefill_chunk_len(),
            "narrowing verify must not touch the prefill chunk"
        );
        eprintln!("verify_m={narrow_m} (wide cap {wide_cap}):");
        wide.reset();
        narrow.reset();
        compare(&mut wide, &mut narrow, narrow_m, 9);
    }
}

#[test]
fn narrow_verify_chain_still_matches_stepped_decode() {
    eprintln!("adapter: {}", ctx().summary());
    for narrow_m in [2usize, 3] {
        let mut m = with_verify_m(Some(narrow_m), || tiny_model(64));
        assert_eq!(m.verify_max_rows(), narrow_m);
        let vocab = m.config().vocab_size;
        let prefix: Vec<u32> = (0..21u32).map(|i| (i * 37 + 5) % vocab as u32).collect();
        for &t in &prefix {
            m.decode_step(t).unwrap();
        }
        let mut rng = Lcg(0xfeedbee5);
        for round in 0..8 {
            let k = 1 + round % narrow_m;
            let batch: Vec<u32> = (0..k).map(|_| rng.token(vocab)).collect();
            let pos0 = m.current_pos();
            let va = m.verify_chain(&batch).unwrap();
            assert_eq!(m.current_pos(), pos0, "verify_chain must not move pos");
            let sa: Vec<u32> = batch.iter().map(|&t| m.decode_step(t).unwrap()).collect();
            m.truncate_to(pos0).unwrap();
            assert_eq!(
                va, sa,
                "verify_m={narrow_m} round {round}: argmax != stepped"
            );
            m.advance(1 + (round * 3) % k).unwrap();
        }
        eprintln!("verify_m={narrow_m}: 8 rounds match stepped decode");
    }
}

fn qat_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("NV_E4B_DIR") {
        return std::path::PathBuf::from(d);
    }
    let base = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots");
    std::fs::read_dir(&base)
        .expect("QAT hub snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").exists())
        .expect("no QAT snapshot with config.json")
}

fn load_qat(max_seq: usize) -> Gemma4E4bWgpu {
    let dir = qat_dir();
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    Gemma4E4bWgpu::from_loader(config, &loader, max_seq).unwrap()
}

#[test]
#[ignore]
fn narrow_verify_width_is_bit_exact_on_the_real_qat_checkpoint() {
    assert_eq!(
        std::env::var("NV_E4B_HOST_SYNC").ok().as_deref(),
        Some("1"),
        "set NV_E4B_HOST_SYNC=1 -- this loads the real checkpoint"
    );
    eprintln!("adapter: {}", ctx().summary());
    eprintln!("checkpoint: {}", qat_dir().display());
    let mut wide = with_verify_m(None, || load_qat(512));
    let narrow_m = 3usize;
    let mut narrow = with_verify_m(Some(narrow_m), || load_qat(512));
    assert_eq!(narrow.verify_max_rows(), narrow_m);
    assert!(wide.verify_max_rows() > narrow_m);
    eprintln!(
        "wide cap {} vs narrow cap {}, prefill chunk {} both",
        wide.verify_max_rows(),
        narrow.verify_max_rows(),
        wide.prefill_chunk_len()
    );
    compare(&mut wide, &mut narrow, narrow_m, 12);
}
