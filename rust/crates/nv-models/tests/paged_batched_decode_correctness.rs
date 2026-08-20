#![cfg(feature = "cuda")]

mod common;
use common::argmax;
use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::{Gemma4, Gemma4Config};
use nv_models::paged_fp8::{PagedGemma4Cache, PagedKvFp8Pool, PagedPoolConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

fn gemma4_nvfp4_snapshot_home_default() -> String {
    format!(
        "{}/.cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots/e5ef03afa233c35cb000323ff098d4291e1dd07c",
        std::env::var("HOME").unwrap_or_default()
    )
}

fn tokenize(tok: &Tokenizer, prompt: &str) -> Vec<u32> {
    let enc = tok.encode(prompt, false).expect("tokenize");
    let mut ids: Vec<u32> = enc.get_ids().to_vec();
    ids.insert(0, 2);
    ids
}

fn prefill(
    model: &Gemma4,
    device: &Device,
    cache: &mut PagedGemma4Cache,
    ids: &[u32],
) -> (u32, usize) {
    let seq = ids.len();
    let tokens = Tensor::from_vec(ids.to_vec(), (1usize, seq), device).expect("tokens");
    let positions: Vec<i32> = (0..seq as i32).collect();
    let pos = Tensor::from_vec(positions, seq, device).expect("pos");
    let logits = model
        .forward_with_cache(&tokens, &pos, cache)
        .expect("prefill");
    let v: Vec<f32> = logits
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let vocab = model.config().vocab_size;
    let last = &v[(seq - 1) * vocab..seq * vocab];
    (argmax(last), seq)
}

fn make_cache(
    pool: &Arc<Mutex<PagedKvFp8Pool>>,
    device: &Device,
    table: &[u32],
) -> PagedGemma4Cache {
    let mut c = PagedGemma4Cache::new(pool.clone(), device).expect("cache");
    c.set_block_table(table).expect("block table");
    c
}

fn snapshot_dir() -> Option<std::path::PathBuf> {
    [
        std::env::var("NV_G4_SNAPSHOT").unwrap_or_default(),
        gemma4_nvfp4_snapshot_home_default(),
    ]
    .into_iter()
    .map(std::path::PathBuf::from)
    .find(|p| p.join("config.json").is_file())
}

fn shared_model(dir: &Path, device: &Device) -> &'static Mutex<(Gemma4, Tokenizer)> {
    static MODEL: std::sync::OnceLock<Mutex<(Gemma4, Tokenizer)>> = std::sync::OnceLock::new();
    MODEL.get_or_init(|| {
        pin_the_gemm_algos_this_suite_was_validated_under();
        Mutex::new(load(dir, device))
    })
}

fn pin_the_gemm_algos_this_suite_was_validated_under() {
    if std::env::var_os("NV_BF16_ALGO_PIN").is_some() {
        return;
    }
    std::env::set_var("NV_BF16_ALGO_PIN", "6x5376x8192=1;7x5376x8192=2");
}

fn load(dir: &Path, device: &Device) -> (Gemma4, Tokenizer) {
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config");
    let cfg = Gemma4Config::from_hf_json_str(&raw_cfg).expect("parse cfg");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(dir, device).expect("weights");
    let model = Gemma4::from_loader_quantized(cfg, &weights, &qcfg, device).expect("model");
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    (model, tok)
}

#[test]
fn decode_attention_is_invariant_to_query_row_offset() {
    use nv_layers::attn::{flash_attn_windowed, AttnConfig};
    use nv_models::gemma4::causal_attention_chunked;

    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let n_q = 32usize;
    let total = 40usize;
    let b = 2usize;

    for (n_kv, head_dim, label) in [(16usize, 256usize, "sliding"), (4usize, 512usize, "global")] {
        let q_all = Tensor::rand(-1f32, 1f32, (1, b, n_q, head_dim), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let k = Tensor::rand(-1f32, 1f32, (1, total, n_kv, head_dim), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let v = Tensor::rand(-1f32, 1f32, (1, total, n_kv, head_dim), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let cfg = AttnConfig {
            num_heads: n_q,
            num_kv_heads: n_kv,
            head_dim,
            softmax_scale: 1.0,
            causal: true,
        };
        for i in 0..b {
            let q_view = q_all.narrow(1, i, 1).unwrap().contiguous().unwrap();
            let q_copy = q_all.narrow(1, i, 1).unwrap().force_contiguous().unwrap();
            let same_bytes: Vec<f32> = q_view
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let copy_bytes: Vec<f32> = q_copy
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(same_bytes, copy_bytes, "{label} slot {i}: view != copy");

            let f32v = |t: &Tensor| -> Vec<f32> {
                t.to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap()
            };
            let out_view =
                f32v(&flash_attn_windowed(&q_view, &k, &v, &cfg, None, Some(0)).unwrap());
            let out_copy =
                f32v(&flash_attn_windowed(&q_copy, &k, &v, &cfg, None, Some(0)).unwrap());
            let out_ref = f32v(
                &causal_attention_chunked(&q_copy, &k, &v, n_q, n_kv, head_dim, 1, total - 1)
                    .unwrap(),
            );
            let maxdiff = |a: &[f32], c: &[f32]| -> f32 {
                a.iter()
                    .zip(c)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max)
            };
            eprintln!(
                "{label} slot {i}: |flash(view)-flash(copy)|={:.3e} |flash(copy)-chunked|={:.3e} |flash(view)-chunked|={:.3e}",
                maxdiff(&out_view, &out_copy),
                maxdiff(&out_copy, &out_ref),
                maxdiff(&out_view, &out_ref)
            );
            if head_dim == 512 {
                assert!(
                    maxdiff(&out_copy, &out_ref) > 1e-1,
                    "{label} slot {i}: FA2's hd512 seq_q=1 kernel now agrees with the reference \
                     ({:.3e}) -- drop the FullAttention detours in gemma4.rs and delete this arm",
                    maxdiff(&out_copy, &out_ref)
                );
            } else {
                assert_eq!(
                    out_view, out_copy,
                    "{label} slot {i}: flash on an offset query view != flash on a copied row"
                );
                assert!(
                    maxdiff(&out_view, &out_ref) < 5e-2,
                    "{label} slot {i}: flash on an offset query view is {:.3e} off the reference",
                    maxdiff(&out_view, &out_ref)
                );
            }
        }
    }
}

#[test]
fn paged_batched_decode_same_prompt_rows_agree() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skip: no Gemma-4-31B snapshot found");
        return;
    };
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let guard = shared_model(dir.as_path(), &device).lock().unwrap();
    let (model, tok) = (&guard.0, &guard.1);

    let block_size = 16usize;
    let num_blocks = 128usize;
    let pool_cfg = PagedPoolConfig::from_gemma4(model.config(), num_blocks, block_size);
    let pool = Arc::new(Mutex::new(
        PagedKvFp8Pool::new(pool_cfg, &device).expect("pool"),
    ));
    let table_a: Vec<u32> = (0..64u32).collect();
    let table_b: Vec<u32> = (64..128u32).collect();

    let ids = tokenize(&tok, "The capital of France is");
    let steps = 6usize;

    let mut c0 = make_cache(&pool, &device, &table_a);
    let mut c1 = make_cache(&pool, &device, &table_b);
    let (t0, p0) = prefill(&model, &device, &mut c0, &ids);
    let (t1, p1) = prefill(&model, &device, &mut c1, &ids);
    assert_eq!(t0, t1, "prefill already differs between the two caches");
    let (mut tok0, mut pos0) = (t0, p0);
    let (mut tok1, mut pos1) = (t1, p1);
    let mut out0 = vec![tok0];
    let mut out1 = vec![tok1];
    for _ in 0..steps {
        let mut caches: Vec<&mut PagedGemma4Cache> = vec![&mut c0, &mut c1];
        let logits = model
            .forward_decode_batched(&[tok0, tok1], &[pos0, pos1], &mut caches)
            .expect("batched decode");
        let v: Vec<f32> = logits
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let vocab = model.config().vocab_size;
        let row0 = &v[0..vocab];
        let row1 = &v[vocab..2 * vocab];
        if row0 != row1 {
            let n_diff = row0.iter().zip(row1).filter(|(a, b)| a != b).count();
            eprintln!("logit rows differ in {n_diff}/{vocab} entries at pos {pos0}");
        }
        tok0 = argmax(row0);
        tok1 = argmax(row1);
        pos0 += 1;
        pos1 += 1;
        out0.push(tok0);
        out1.push(tok1);
    }
    eprintln!("row0: {:?}", tok.decode(&out0, false).unwrap_or_default());
    eprintln!("row1: {:?}", tok.decode(&out1, false).unwrap_or_default());
    assert_eq!(out0, out1, "identical prompts decoded differently per slot");
}

#[test]
fn paged_batched_decode_is_slot_invariant() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skip: no Gemma-4-31B snapshot found");
        return;
    };
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let guard = shared_model(dir.as_path(), &device).lock().unwrap();
    let (model, tok) = (&guard.0, &guard.1);

    let block_size = 16usize;
    let num_blocks = 128usize;
    let pool_cfg = PagedPoolConfig::from_gemma4(model.config(), num_blocks, block_size);
    let pool = Arc::new(Mutex::new(
        PagedKvFp8Pool::new(pool_cfg, &device).expect("pool"),
    ));
    let table_a: Vec<u32> = (0..64u32).collect();
    let table_b: Vec<u32> = (64..128u32).collect();

    let ids_a = tokenize(&tok, "The capital of France is");
    let ids_b = tokenize(&tok, "def fibonacci(n):");
    let steps = 24usize;

    let run = |first: &[u32], second: &[u32]| -> (Vec<u32>, Vec<u32>) {
        let mut c0 = make_cache(&pool, &device, &table_a);
        let mut c1 = make_cache(&pool, &device, &table_b);
        let (t0, p0) = prefill(&model, &device, &mut c0, first);
        let (t1, p1) = prefill(&model, &device, &mut c1, second);
        let (mut tok0, mut pos0) = (t0, p0);
        let (mut tok1, mut pos1) = (t1, p1);
        let mut out0 = vec![tok0];
        let mut out1 = vec![tok1];
        for _ in 0..steps {
            let mut caches: Vec<&mut PagedGemma4Cache> = vec![&mut c0, &mut c1];
            let logits = model
                .forward_decode_batched(&[tok0, tok1], &[pos0, pos1], &mut caches)
                .expect("batched decode");
            let v: Vec<f32> = logits
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let vocab = model.config().vocab_size;
            tok0 = argmax(&v[0..vocab]);
            tok1 = argmax(&v[vocab..2 * vocab]);
            pos0 += 1;
            pos1 += 1;
            out0.push(tok0);
            out1.push(tok1);
        }
        (out0, out1)
    };

    let (ab_a, ab_b) = run(&ids_a, &ids_b);
    let (ba_b, ba_a) = run(&ids_b, &ids_a);

    eprintln!(
        "A in slot0: {:?}",
        tok.decode(&ab_a, false).unwrap_or_default()
    );
    eprintln!(
        "A in slot1: {:?}",
        tok.decode(&ba_a, false).unwrap_or_default()
    );
    eprintln!(
        "B in slot1: {:?}",
        tok.decode(&ab_b, false).unwrap_or_default()
    );
    eprintln!(
        "B in slot0: {:?}",
        tok.decode(&ba_b, false).unwrap_or_default()
    );

    assert_eq!(
        ab_a,
        ba_a,
        "sequence A decoded differently in slot 0 vs slot 1:\n  slot0 {:?}\n  slot1 {:?}",
        tok.decode(&ab_a, false).unwrap_or_default(),
        tok.decode(&ba_a, false).unwrap_or_default()
    );
    assert_eq!(
        ab_b,
        ba_b,
        "sequence B decoded differently in slot 1 vs slot 0:\n  slot1 {:?}\n  slot0 {:?}",
        tok.decode(&ab_b, false).unwrap_or_default(),
        tok.decode(&ba_b, false).unwrap_or_default()
    );
}

#[test]
fn paged_batched_decode_matches_solo() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skip: no Gemma-4-31B snapshot found");
        return;
    };
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let guard = shared_model(dir.as_path(), &device).lock().unwrap();
    let (model, tok) = (&guard.0, &guard.1);

    let block_size = 16usize;
    let num_blocks = 128usize;
    let pool_cfg = PagedPoolConfig::from_gemma4(model.config(), num_blocks, block_size);
    let pool = Arc::new(Mutex::new(
        PagedKvFp8Pool::new(pool_cfg, &device).expect("pool"),
    ));

    let table_a: Vec<u32> = (0..64u32).collect();
    let table_b: Vec<u32> = (64..128u32).collect();

    let ids0 = tokenize(&tok, "The capital of France is");
    let ids1 = tokenize(&tok, "def fibonacci(n):");
    let steps = 24usize;

    let mut c0 = make_cache(&pool, &device, &table_a);
    let mut c1 = make_cache(&pool, &device, &table_b);
    let (t0, p0) = prefill(&model, &device, &mut c0, &ids0);
    let (t1, p1) = prefill(&model, &device, &mut c1, &ids1);
    let (mut tok0, mut pos0) = (t0, p0);
    let (mut tok1, mut pos1) = (t1, p1);
    let mut batched0 = vec![tok0];
    let mut batched1 = vec![tok1];
    for _ in 0..steps {
        let mut caches: Vec<&mut PagedGemma4Cache> = vec![&mut c0, &mut c1];
        let logits = model
            .forward_decode_batched(&[tok0, tok1], &[pos0, pos1], &mut caches)
            .expect("batched decode");
        let v: Vec<f32> = logits
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let vocab = model.config().vocab_size;
        tok0 = argmax(&v[0..vocab]);
        tok1 = argmax(&v[vocab..2 * vocab]);
        pos0 += 1;
        pos1 += 1;
        batched0.push(tok0);
        batched1.push(tok1);
    }

    let solo = |ids: &[u32], table: &[u32]| -> Vec<u32> {
        let mut c = make_cache(&pool, &device, table);
        let (t, p) = prefill(&model, &device, &mut c, ids);
        let (mut tk, mut ps) = (t, p);
        let mut out = vec![tk];
        for _ in 0..steps {
            let mut caches: Vec<&mut PagedGemma4Cache> = vec![&mut c];
            let logits = model
                .forward_decode_batched(&[tk], &[ps], &mut caches)
                .expect("solo decode");
            let v: Vec<f32> = logits
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            tk = argmax(&v);
            ps += 1;
            out.push(tk);
        }
        out
    };
    let solo0 = solo(&ids0, &table_a);
    let solo1 = solo(&ids1, &table_b);

    eprintln!(
        "seq0 batched: {:?}",
        tok.decode(&batched0, false).unwrap_or_default()
    );
    eprintln!(
        "seq0 solo   : {:?}",
        tok.decode(&solo0, false).unwrap_or_default()
    );
    eprintln!(
        "seq1 batched: {:?}",
        tok.decode(&batched1, false).unwrap_or_default()
    );
    eprintln!(
        "seq1 solo   : {:?}",
        tok.decode(&solo1, false).unwrap_or_default()
    );

    let s0 = tok
        .decode(&batched0, false)
        .unwrap_or_default()
        .to_lowercase();
    let s1 = tok
        .decode(&batched1, false)
        .unwrap_or_default()
        .to_lowercase();
    assert!(!s0.is_empty() && !s1.is_empty(), "empty batched output");
    assert!(
        s0.contains("paris"),
        "seq0 (capital of France) lost its topic under batching: {s0:?}"
    );
    assert!(
        !s1.contains("paris"),
        "seq1 (a code prompt) bled seq0's Paris content -- KV cross-contamination: {s1:?}"
    );

    assert_eq!(
        batched0,
        solo0,
        "seq0 batched tokens differ from solo:\n  batched {:?}\n  solo    {:?}",
        tok.decode(&batched0, false).unwrap_or_default(),
        tok.decode(&solo0, false).unwrap_or_default()
    );
    assert_eq!(
        batched1,
        solo1,
        "seq1 batched tokens differ from solo:\n  batched {:?}\n  solo    {:?}",
        tok.decode(&batched1, false).unwrap_or_default(),
        tok.decode(&solo1, false).unwrap_or_default()
    );
}
