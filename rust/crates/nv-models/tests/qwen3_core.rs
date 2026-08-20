use candle_core::{DType, Device, Tensor};
use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;
use nv_layers::norm::RmsNorm;
use nv_models::qwen3::{KvCache, Qwen3, Qwen3Config, Qwen3Layer};
use nv_models::CausalLm;
use nv_weights::WeightLoader;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 32) as u32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * scale).collect()
    }
}

fn tiny_cfg() -> Qwen3Config {
    Qwen3Config {
        hidden_size: 8,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        intermediate_size: 16,
        vocab_size: 13,
        max_position_embeddings: 32,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        bos_token_id: 0,
        eos_token_id: 1,
        torch_dtype: None,
        sliding_window: None,
    }
}

fn base_json_map() -> serde_json::Map<String, serde_json::Value> {
    let v = serde_json::json!({
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 16,
        "vocab_size": 13,
        "max_position_embeddings": 32,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": true,
        "bos_token_id": 0,
        "eos_token_id": 1
    });
    v.as_object().expect("base config json is an object").clone()
}

fn mat(rng: &mut Lcg, out: usize, inp: usize, dev: &Device) -> Tensor {
    Tensor::from_vec(rng.vec(out * inp, 0.3), (out, inp), dev).expect("build weight matrix")
}

fn norm_w(rng: &mut Lcg, dim: usize, dev: &Device) -> Tensor {
    let v: Vec<f32> = (0..dim).map(|_| 1.0 + rng.next_f32() * 0.1).collect();
    Tensor::from_vec(v, (dim,), dev).expect("build norm weight")
}

fn tiny_layer(cfg: &Qwen3Config, rng: &mut Lcg, dev: &Device) -> Qwen3Layer {
    let h = cfg.hidden_size;
    let qd = cfg.num_attention_heads * cfg.head_dim;
    let kvd = cfg.num_key_value_heads * cfg.head_dim;
    let inter = cfg.intermediate_size;
    let eps = cfg.rms_norm_eps;
    Qwen3Layer::new(
        RmsNorm::new(norm_w(rng, h, dev), eps),
        Linear::new(mat(rng, qd, h, dev), None).expect("q_proj"),
        Linear::new(mat(rng, kvd, h, dev), None).expect("k_proj"),
        Linear::new(mat(rng, kvd, h, dev), None).expect("v_proj"),
        Linear::new(mat(rng, h, qd, dev), None).expect("o_proj"),
        RmsNorm::new(norm_w(rng, cfg.head_dim, dev), eps),
        RmsNorm::new(norm_w(rng, cfg.head_dim, dev), eps),
        RmsNorm::new(norm_w(rng, h, dev), eps),
        Mlp::new(
            Linear::new(mat(rng, inter, h, dev), None).expect("gate_proj"),
            Linear::new(mat(rng, inter, h, dev), None).expect("up_proj"),
            Linear::new(mat(rng, h, inter, dev), None).expect("down_proj"),
        )
        .expect("mlp"),
    )
}

fn tiny_model(seed: u64) -> Qwen3 {
    let dev = Device::Cpu;
    let cfg = tiny_cfg();
    let mut rng = Lcg::new(seed);
    let embed = mat(&mut rng, cfg.vocab_size, cfg.hidden_size, &dev);
    let layers: Vec<Qwen3Layer> = (0..cfg.num_hidden_layers)
        .map(|_| tiny_layer(&cfg, &mut rng, &dev))
        .collect();
    let final_norm = RmsNorm::new(norm_w(&mut rng, cfg.hidden_size, &dev), cfg.rms_norm_eps);
    let lm_head = Linear::new(embed.clone(), None).expect("tied lm_head");
    Qwen3::from_parts(cfg, embed, layers, final_norm, lm_head, &dev, DType::F32)
        .expect("assemble tiny qwen3 from parts")
}

fn tokens_tensor(ids: &[u32], dev: &Device) -> Tensor {
    Tensor::from_vec(ids.to_vec(), (1usize, ids.len()), dev).expect("tokens tensor")
}

fn positions_tensor(pos: &[u32], dev: &Device) -> Tensor {
    Tensor::from_vec(pos.to_vec(), (pos.len(),), dev).expect("positions tensor")
}

fn last_row(logits: &Tensor) -> Vec<f32> {
    let dims = logits.dims();
    logits
        .narrow(1, dims[1] - 1, 1)
        .expect("narrow last position")
        .flatten_all()
        .expect("flatten logits row")
        .to_vec1::<f32>()
        .expect("logits row to host")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

static FIXTURE_DIR: OnceLock<PathBuf> = OnceLock::new();

fn fixture_dir() -> &'static Path {
    FIXTURE_DIR.get_or_init(|| {
        let d = std::env::temp_dir().join(format!("qwen3_core_fixture_{}", std::process::id()));
        std::fs::create_dir_all(&d).expect("create per-process fixture dir");
        d
    })
}

fn one_layer_stripped_name_weights(include_q_norm: bool) -> HashMap<String, Tensor> {
    let dev = Device::Cpu;
    let mut rng = Lcg::new(0xFEED5EED);
    let cfg = tiny_cfg();
    let h = cfg.hidden_size;
    let qd = cfg.num_attention_heads * cfg.head_dim;
    let kvd = cfg.num_key_value_heads * cfg.head_dim;
    let inter = cfg.intermediate_size;
    let mut m: HashMap<String, Tensor> = HashMap::new();
    m.insert(
        "embed_tokens.weight".into(),
        mat(&mut rng, cfg.vocab_size, h, &dev),
    );
    m.insert("norm.weight".into(), norm_w(&mut rng, h, &dev));
    let p = "layers.0";
    m.insert(
        format!("{p}.input_layernorm.weight"),
        norm_w(&mut rng, h, &dev),
    );
    m.insert(
        format!("{p}.post_attention_layernorm.weight"),
        norm_w(&mut rng, h, &dev),
    );
    m.insert(
        format!("{p}.self_attn.q_proj.weight"),
        mat(&mut rng, qd, h, &dev),
    );
    m.insert(
        format!("{p}.self_attn.k_proj.weight"),
        mat(&mut rng, kvd, h, &dev),
    );
    m.insert(
        format!("{p}.self_attn.v_proj.weight"),
        mat(&mut rng, kvd, h, &dev),
    );
    m.insert(
        format!("{p}.self_attn.o_proj.weight"),
        mat(&mut rng, h, qd, &dev),
    );
    if include_q_norm {
        m.insert(
            format!("{p}.self_attn.q_norm.weight"),
            norm_w(&mut rng, cfg.head_dim, &dev),
        );
    }
    m.insert(
        format!("{p}.self_attn.k_norm.weight"),
        norm_w(&mut rng, cfg.head_dim, &dev),
    );
    m.insert(
        format!("{p}.mlp.gate_proj.weight"),
        mat(&mut rng, inter, h, &dev),
    );
    m.insert(
        format!("{p}.mlp.up_proj.weight"),
        mat(&mut rng, inter, h, &dev),
    );
    m.insert(
        format!("{p}.mlp.down_proj.weight"),
        mat(&mut rng, h, inter, &dev),
    );
    m
}

fn stripped_fixture_loader(file_stem: &str, include_q_norm: bool) -> WeightLoader {
    let path = fixture_dir().join(format!("{file_stem}.safetensors"));
    if !path.exists() {
        candle_core::safetensors::save(&one_layer_stripped_name_weights(include_q_norm), &path)
            .expect("write safetensors fixture");
    }
    WeightLoader::open_file(&path, &Device::Cpu).expect("open safetensors fixture")
}

#[test]
fn hf_eos_array_collapses_to_its_first_id_and_scalar_eos_survives_verbatim() {
    let mut m = base_json_map();
    m.insert("eos_token_id".into(), serde_json::json!([151645, 151643]));
    let cfg = Qwen3Config::from_hf_json_str(&serde_json::Value::Object(m.clone()).to_string())
        .expect("array-form eos_token_id must parse, real Qwen3 configs ship it as a list");
    assert_eq!(
        cfg.eos_token_id, 151645,
        "an eos array must collapse to element ZERO; picking any other element changes which token stops generation"
    );
    m.insert("eos_token_id".into(), serde_json::json!([151643, 151645]));
    let cfg_rev = Qwen3Config::from_hf_json_str(&serde_json::Value::Object(m.clone()).to_string())
        .expect("reversed eos array must also parse");
    assert_eq!(
        cfg_rev.eos_token_id, 151643,
        "reversed array must yield the OTHER id: if both orders return the same value the rule became min/max/known-id instead of first"
    );
    m.insert("eos_token_id".into(), serde_json::json!(7));
    let cfg_scalar = Qwen3Config::from_hf_json_str(&serde_json::Value::Object(m).to_string())
        .expect("scalar eos_token_id must keep parsing");
    assert_eq!(
        cfg_scalar.eos_token_id, 7,
        "scalar eos must pass through untouched; a normalizer that rewrites scalars would corrupt hand-written configs"
    );
}

#[test]
fn hf_optional_fields_default_to_none_and_file_parse_agrees_with_str_parse() {
    let bare = serde_json::Value::Object(base_json_map()).to_string();
    let cfg = Qwen3Config::from_hf_json_str(&bare)
        .expect("config without torch_dtype/sliding_window must parse, most checkpoints omit them");
    assert!(
        cfg.torch_dtype.is_none() && cfg.sliding_window.is_none(),
        "absent optional fields must deserialize as None, not a made-up default that downstream code would trust"
    );
    let mut m = base_json_map();
    m.insert("torch_dtype".into(), serde_json::json!("bfloat16"));
    m.insert("sliding_window".into(), serde_json::json!(4096));
    let cfg_full = Qwen3Config::from_hf_json_str(&serde_json::Value::Object(m).to_string())
        .expect("config with optional fields present must parse");
    assert_eq!(
        cfg_full.torch_dtype.as_deref(),
        Some("bfloat16"),
        "present torch_dtype must round-trip or dtype selection upstream goes blind"
    );
    assert_eq!(
        cfg_full.sliding_window,
        Some(4096),
        "present sliding_window must round-trip or windowed-attention checkpoints load as full-attention"
    );
    let p = fixture_dir().join("bare_config.json");
    std::fs::write(&p, &bare).expect("write config fixture");
    let from_file = Qwen3Config::from_hf_json_file(&p).expect("file parse of the same bytes");
    assert_eq!(
        from_file.eos_token_id, cfg.eos_token_id,
        "from_hf_json_file must be the same parser as from_hf_json_str, not a second diverging one"
    );
}

#[test]
fn kv_append_advances_current_len_only_after_the_final_layer_writes() {
    let dev = Device::Cpu;
    let cfg = tiny_cfg();
    let mut cache = KvCache::new(&cfg, 8, &dev, DType::F32).expect("two-layer cache");
    let k = Tensor::zeros((1usize, 3usize, 1usize, 4usize), DType::F32, &dev).expect("k");
    let v = k.clone();
    cache.append(0, &k, &v).expect("append layer 0");
    assert_eq!(
        cache.current_len(),
        0,
        "advancing after a non-final layer would make layer 1 write its keys 3 slots past layer 0's, silently corrupting attention"
    );
    cache.append(1, &k, &v).expect("append layer 1");
    assert_eq!(
        cache.current_len(),
        3,
        "after the final layer appends, the shared write cursor must move by exactly the token count"
    );
    let mut one = tiny_cfg();
    one.num_hidden_layers = 1;
    let mut single = KvCache::new(&one, 8, &dev, DType::F32).expect("one-layer cache");
    single.append(0, &k, &v).expect("append sole layer");
    assert_eq!(
        single.current_len(),
        3,
        "negative control: with one layer, layer 0 IS the final layer and must advance; a hardcoded layer-index rule would leave this at 0"
    );
}

#[test]
fn kv_write_at_accepts_the_declared_shape_and_rejects_every_shape_and_bound_violation() {
    let dev = Device::Cpu;
    let cfg = tiny_cfg();
    let mut cache = KvCache::new(&cfg, 8, &dev, DType::F32).expect("cache");
    let ok = Tensor::zeros((1usize, 2usize, 1usize, 4usize), DType::F32, &dev).expect("ok kv");
    cache.write_at(0, 0, &ok, &ok).expect(
        "negative control: the exactly-declared [1,t,kv_heads,head_dim] shape must be accepted, otherwise every rejection below is vacuous",
    );
    let wrong_heads =
        Tensor::zeros((1usize, 2usize, 2usize, 4usize), DType::F32, &dev).expect("wh");
    assert!(
        cache.write_at(0, 0, &wrong_heads, &wrong_heads).is_err(),
        "a q-head-count tensor written into a kv-head cache would alias memory across heads"
    );
    let wrong_dim = Tensor::zeros((1usize, 2usize, 1usize, 3usize), DType::F32, &dev).expect("wd");
    assert!(
        cache.write_at(0, 0, &wrong_dim, &wrong_dim).is_err(),
        "a head_dim mismatch must be refused before slice_assign misplaces every value after the first row"
    );
    let rank3 = Tensor::zeros((2usize, 1usize, 4usize), DType::F32, &dev).expect("r3");
    assert!(
        cache.write_at(0, 0, &rank3, &rank3).is_err(),
        "rank-3 input must be refused, the cache layout is strictly [batch, t, kv_heads, head_dim]"
    );
    assert!(
        cache.write_at(0, 7, &ok, &ok).is_err(),
        "start 7 + t 2 overruns max_seq_len 8; accepting it would wrap or clobber the tail of the buffer"
    );
    assert!(
        cache.write_at(2, 0, &ok, &ok).is_err(),
        "layer index equal to layer count must be refused, not silently dropped or panicked on"
    );
    let v_longer = Tensor::zeros((1usize, 3usize, 1usize, 4usize), DType::F32, &dev).expect("vl");
    assert!(
        cache.write_at(0, 0, &ok, &v_longer).is_err(),
        "k and v with different token counts would desynchronize keys from values at attention time"
    );
}

#[test]
fn kv_view_returns_exactly_len_rows_with_written_values_at_their_offset_and_zeros_before() {
    let dev = Device::Cpu;
    let cfg = tiny_cfg();
    let mut cache = KvCache::new(&cfg, 8, &dev, DType::F32).expect("cache");
    let payload: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
    let k = Tensor::from_vec(payload.clone(), (1usize, 2usize, 1usize, 4usize), &dev).expect("k");
    let v = Tensor::from_vec(
        payload.iter().map(|x| x * 10.0).collect::<Vec<f32>>(),
        (1usize, 2usize, 1usize, 4usize),
        &dev,
    )
    .expect("v");
    cache.write_at(1, 2, &k, &v).expect("write at offset 2");
    let (kv, vv) = cache.view(1, 4).expect("view 4 rows");
    assert_eq!(
        kv.dims(),
        [1, 4, 1, 4],
        "view must narrow the time axis to the requested length, a full-buffer view leaks stale future slots into attention"
    );
    let k_host = kv.flatten_all().expect("fk").to_vec1::<f32>().expect("hk");
    let v_host = vv.flatten_all().expect("fv").to_vec1::<f32>().expect("hv");
    assert!(
        k_host[..8].iter().all(|x| *x == 0.0),
        "rows before the write offset must still be the zeros they were initialized to"
    );
    assert_eq!(
        &k_host[8..16],
        payload.as_slice(),
        "the k payload must land at time offset 2, not at 0; an offset bug shifts every cached key"
    );
    assert_eq!(
        v_host[8], 10.0,
        "v must be stored in the v buffer, not mirrored from k"
    );
    let (k0, _) = cache.view(0, 4).expect("other layer view");
    let k0_host = k0.flatten_all().expect("f0").to_vec1::<f32>().expect("h0");
    assert!(
        k0_host.iter().all(|x| *x == 0.0),
        "a write to layer 1 must not bleed into layer 0's buffer"
    );
    cache.view(1, 8).expect("view at exactly max_seq_len must succeed");
    assert!(
        cache.view(1, 9).is_err(),
        "view past max_seq_len must be refused, narrow would otherwise panic deep in candle"
    );
    assert!(
        cache.view(2, 1).is_err(),
        "view of a nonexistent layer must be refused"
    );
}

#[test]
fn kv_reset_rewinds_the_cursor_without_shrinking_capacity_so_the_next_prefill_overwrites() {
    let dev = Device::Cpu;
    let cfg = tiny_cfg();
    let mut cache = KvCache::new(&cfg, 8, &dev, DType::F32).expect("cache");
    let old = Tensor::from_vec(vec![5f32; 12], (1usize, 3usize, 1usize, 4usize), &dev).expect("o");
    cache.append(0, &old, &old).expect("l0");
    cache.append(1, &old, &old).expect("l1");
    assert_eq!(cache.current_len(), 3, "precondition: three tokens cached");
    cache.reset();
    assert_eq!(
        cache.current_len(),
        0,
        "reset must rewind the cursor to zero so the next sequence starts from position 0"
    );
    assert_eq!(
        cache.max_seq_len(),
        8,
        "reset must not shrink capacity; a fresh prefill of 8 tokens must still fit"
    );
    let fresh =
        Tensor::from_vec(vec![9f32; 8], (1usize, 2usize, 1usize, 4usize), &dev).expect("f");
    cache.append(0, &fresh, &fresh).expect("l0 again");
    cache.append(1, &fresh, &fresh).expect("l1 again");
    assert_eq!(cache.current_len(), 2, "post-reset append restarts at slot 0");
    let (k, _) = cache.get(0).expect("get after reset");
    let host = k.flatten_all().expect("f").to_vec1::<f32>().expect("h");
    assert!(
        host.iter().all(|x| *x == 9.0),
        "get() after reset must expose only the new sequence; a surviving 5.0 means the old sequence leaks into the new one's attention"
    );
}

#[test]
fn incremental_decode_through_the_kv_cache_matches_single_shot_prefill_logits() {
    let dev = Device::Cpu;
    let model = tiny_model(0xA11CE);
    let ids: [u32; 6] = [3, 7, 1, 9, 4, 11];
    let pos: [u32; 6] = [0, 1, 2, 3, 4, 5];

    let mut cache_full = model.new_kv_cache(16).expect("full cache");
    let logits_full = model
        .forward(
            &tokens_tensor(&ids, &dev),
            &positions_tensor(&pos, &dev),
            &mut cache_full,
        )
        .expect("single-shot prefill");
    assert_eq!(
        logits_full.dims(),
        [1, 6, 13],
        "logits must be [1, seq, vocab]; any other shape breaks every sampler downstream"
    );
    let full_last = last_row(&logits_full);

    let mut cache_inc = model.new_kv_cache(16).expect("incremental cache");
    model
        .forward(
            &tokens_tensor(&ids[..5], &dev),
            &positions_tensor(&pos[..5], &dev),
            &mut cache_inc,
        )
        .expect("5-token prefill");
    assert_eq!(
        cache_inc.current_len(),
        5,
        "forward must advance the cache by exactly the prefill length or the decode step writes over live keys"
    );
    let logits_step = model
        .forward(
            &tokens_tensor(&ids[5..], &dev),
            &positions_tensor(&pos[5..], &dev),
            &mut cache_inc,
        )
        .expect("1-token decode");
    let step_last = last_row(&logits_step);
    let diff = max_abs_diff(&full_last, &step_last);
    assert!(
        diff < 1e-4,
        "prefill+decode must reproduce single-shot logits (max|diff|={diff}); divergence here means the cache view, write offset, or rope position drifted and every generated token after the first is wrong"
    );

    let mut cache_wrong = model.new_kv_cache(16).expect("wrong-position cache");
    model
        .forward(
            &tokens_tensor(&ids[..5], &dev),
            &positions_tensor(&pos[..5], &dev),
            &mut cache_wrong,
        )
        .expect("prefill for negative control");
    let logits_wrong = model
        .forward(
            &tokens_tensor(&ids[5..], &dev),
            &positions_tensor(&[0u32], &dev),
            &mut cache_wrong,
        )
        .expect("decode at deliberately wrong position 0");
    let wrong_diff = max_abs_diff(&full_last, &last_row(&logits_wrong));
    assert!(
        wrong_diff > 1e-3,
        "negative control: decoding token 5 at rope position 0 must move the logits (max|diff|={wrong_diff}); if it does not, positions are ignored and the parity gate above proves nothing"
    );
}

#[test]
fn forward_rejects_unbatched_tokens_multi_batch_tokens_and_position_length_mismatch() {
    let dev = Device::Cpu;
    let model = tiny_model(0xB0B);
    let mut cache = model.new_kv_cache(16).expect("cache");
    let flat = Tensor::from_vec(vec![1u32, 2, 3], (3usize,), &dev).expect("flat tokens");
    let err = model
        .forward(&flat, &positions_tensor(&[0, 1, 2], &dev), &mut cache)
        .expect_err("rank-1 tokens must be refused, the cache and attention path assume batch dim 1");
    assert!(
        format!("{err:#}").contains("must be [1, seq]"),
        "the rejection must name the expected shape so callers can fix their reshape, got: {err:#}"
    );
    let b2 = Tensor::from_vec(vec![1u32, 2, 3, 4], (2usize, 2usize), &dev).expect("b2");
    assert!(
        model
            .forward(&b2, &positions_tensor(&[0, 1], &dev), &mut cache)
            .is_err(),
        "batch 2 must be refused: the single-sequence KvCache would interleave two conversations"
    );
    let toks = tokens_tensor(&[1, 2, 3], &dev);
    let err2 = model
        .forward(&toks, &positions_tensor(&[0, 1], &dev), &mut cache)
        .expect_err("2 positions for 3 tokens must be refused before rope reads out of bounds");
    assert!(
        format!("{err2:#}").contains("positions must be"),
        "the mismatch error must point at positions, got: {err2:#}"
    );
    assert_eq!(
        cache.current_len(),
        0,
        "a rejected forward must not advance the cache; a half-advanced cursor poisons the next real call"
    );
}

#[test]
fn causal_lm_shim_refuses_the_slice_api_but_still_reports_the_config_vocab_size() {
    let mut model = tiny_model(0xD0D0);
    let vocab = CausalLm::vocab_size(&model);
    assert_eq!(
        vocab, 13,
        "vocab_size must come from the config's vocab_size (13), not hidden_size (8) or any other field; samplers size their logit buffers from this"
    );
    let err = CausalLm::forward(&mut model, &[1, 2], &[0, 1])
        .expect_err("the slice-based CausalLm::forward is an unimplemented shim and must fail loudly rather than return garbage logits");
    assert!(
        format!("{err:#}").contains("not implemented"),
        "the shim error must say it is unimplemented and route callers to the tensor API, got: {err:#}"
    );
}

#[test]
fn from_loader_accepts_checkpoints_whose_tensor_names_lack_the_model_prefix() {
    let loader = stripped_fixture_loader("stripped_full", true);
    let cfg = Qwen3Config {
        num_hidden_layers: 1,
        ..tiny_cfg()
    };
    let model = Qwen3::from_loader(cfg, &loader, &Device::Cpu).expect(
        "a checkpoint storing embed_tokens.weight/layers.0.* without the model. prefix must load via the stripped-name fallback; real exports ship both conventions",
    );
    assert_eq!(
        model.config().vocab_size,
        13,
        "the loaded model must carry the config it was built from"
    );
    let loader_missing = stripped_fixture_loader("stripped_no_qnorm", false);
    let cfg2 = Qwen3Config {
        num_hidden_layers: 1,
        ..tiny_cfg()
    };
    let err = match Qwen3::from_loader(cfg2, &loader_missing, &Device::Cpu) {
        Ok(_) => panic!("negative control: with q_norm.weight absent under BOTH naming conventions the load must fail; if it succeeds the fallback degraded into inventing tensors"),
        Err(e) => e,
    };
    assert!(
        format!("{err:#}").contains("q_norm"),
        "the failure must name the missing tensor so the checkpoint can be diagnosed, got: {err:#}"
    );
}

#[test]
fn tied_embeddings_skip_the_lm_head_tensor_and_untied_configs_require_it() {
    let loader = stripped_fixture_loader("stripped_full", true);
    let tied = Qwen3Config {
        num_hidden_layers: 1,
        tie_word_embeddings: true,
        ..tiny_cfg()
    };
    Qwen3::from_loader(tied, &loader, &Device::Cpu).expect(
        "tie_word_embeddings=true must load from a checkpoint with NO lm_head.weight; tied checkpoints legitimately omit it",
    );
    let untied = Qwen3Config {
        num_hidden_layers: 1,
        tie_word_embeddings: false,
        ..tiny_cfg()
    };
    let err = match Qwen3::from_loader(untied, &loader, &Device::Cpu) {
        Ok(_) => panic!("negative control: tie=false against the same lm_head-less file must fail; silently reusing the embedding here would mask a truncated checkpoint"),
        Err(e) => e,
    };
    assert!(
        format!("{err:#}").contains("lm_head.weight"),
        "the untied failure must name lm_head.weight, got: {err:#}"
    );
}

#[test]
fn from_loader_rejects_an_embedding_whose_shape_disagrees_with_the_config() {
    let loader = stripped_fixture_loader("stripped_full", true);
    let wrong_vocab = Qwen3Config {
        num_hidden_layers: 1,
        vocab_size: 12,
        ..tiny_cfg()
    };
    let err = match Qwen3::from_loader(wrong_vocab, &loader, &Device::Cpu) {
        Ok(_) => panic!("a config claiming vocab 12 against a [13, 8] embedding must be refused; loading it would misalign every token id past the mismatch"),
        Err(e) => e,
    };
    assert!(
        format!("{err:#}").contains("embedding shape mismatch"),
        "the refusal must identify the embedding shape check, got: {err:#}"
    );
}
