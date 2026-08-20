#![cfg(feature = "cuda")]

mod common;
use common::argmax;
use common::VOCAB_512 as VOCAB;
use common::write_tiny_model;
use candle_core::{DType, Device, Tensor};
use half::bf16;
use nv_layers::lora_slots::{
    LoraAdapter, LoraDispatch, LoraModuleSpec, LoraModuleWeights, LoraSlotManager,
};
use nv_models::gemma4::{Gemma4, Gemma4Config};
use nv_models::gemma4_batch_graph::{BucketPlan, Gemma4BatchGraphFamily, SlotUpdate};
use nv_models::paged_fp8::{PagedGemma4Cache, PagedKvFp8Pool, PagedPoolConfig};
use nv_weights::WeightLoader;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use nv_layers::lora_slots::LoraHook;

const BLOCK_SIZE: usize = 16;

fn prefill(model: &Gemma4, device: &Device, cache: &mut PagedGemma4Cache, ids: &[u32]) -> u32 {
    let seq = ids.len();
    let tokens = Tensor::from_vec(ids.to_vec(), (1usize, seq), device).unwrap();
    let positions: Vec<i32> = (0..seq as i32).collect();
    let pos = Tensor::from_vec(positions, seq, device).unwrap();
    let logits = model.forward_with_cache(&tokens, &pos, cache).unwrap();
    let v: Vec<f32> = logits
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    argmax(&v[(seq - 1) * VOCAB..seq * VOCAB])
}

#[test]
#[ignore]
fn graphed_batch_matches_eager_active_slot() {
    if std::env::var("NV_BATCH_GRAPH_TEST").ok().as_deref() != Some("1") {
        if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "set NV_BATCH_GRAPH_TEST=1 to run this gate, or \
                 NV_MODELS_ALLOW_SKIP=1 to skip it on purpose. Reporting \
                 \"ok\" in 0.00s for a CUDA-graph test is how three of these \
                 sat green while one of them was broken."
            );
        }
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_BATCH_GRAPH_TEST=1 to run");
        return;
    }
    let Ok(device) = Device::new_cuda(0) else {
        panic!(
            "gemma4_batch_graph is a GPU test and CUDA device 0 did not open. \
             This gate refuses to report success without running."
        );
    };

    static BATCH_FIXTURE: std::sync::OnceLock<std::path::PathBuf> =
        std::sync::OnceLock::new();
    let dir = BATCH_FIXTURE
        .get_or_init(|| {
            let d = std::env::temp_dir()
                .join(format!("{}-{}", "nv-gemma4-batch-graph-tiny", std::process::id()));
            write_tiny_model(&d);
            d
        })
        .clone();
    let cfg =
        Gemma4Config::from_hf_json_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
            .unwrap();
    let weights = WeightLoader::open_dir(&dir, &device).unwrap();
    let model = Arc::new(Gemma4::from_loader(cfg, &weights, &device).unwrap());

    let num_blocks = 8usize;
    let scratch_blocks = 2usize;
    let pool_cfg =
        PagedPoolConfig::from_gemma4(model.config(), num_blocks + scratch_blocks, BLOCK_SIZE);
    let pool = Arc::new(Mutex::new(PagedKvFp8Pool::new(pool_cfg, &device).unwrap()));

    let table_graph: Vec<u32> = vec![0, 1, 2, 3];
    let table_eager: Vec<u32> = vec![4, 5, 6, 7];
    assert!(table_graph.iter().all(|b| !table_eager.contains(b)));

    let prompt: Vec<u32> = (1..15).collect();
    let steps = 6usize;

    let mut eager_cache = PagedGemma4Cache::new(pool.clone(), &device).unwrap();
    eager_cache.set_block_table(&table_eager).unwrap();
    let first = prefill(&model, &device, &mut eager_cache, &prompt);
    let mut eager_tokens = vec![first];
    {
        let mut tok = first;
        let mut pos = prompt.len();
        for _ in 0..steps {
            let mut caches: Vec<&mut PagedGemma4Cache> = vec![&mut eager_cache];
            let logits = model
                .forward_decode_batched(&[tok], &[pos], &mut caches)
                .unwrap();
            let v: Vec<f32> = logits
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            tok = argmax(&v[0..VOCAB]);
            pos += 1;
            eager_tokens.push(tok);
        }
    }

    let mut graph_cache = PagedGemma4Cache::new(pool.clone(), &device).unwrap();
    graph_cache.set_block_table(&table_graph).unwrap();
    let g_first = prefill(&model, &device, &mut graph_cache, &prompt);
    assert_eq!(
        g_first, first,
        "prefill argmax must agree before graphs start"
    );

    let plan = BucketPlan::new(vec![2]);
    let mut family = Gemma4BatchGraphFamily::new(
        model.clone(),
        pool.clone(),
        &device,
        plan,
        num_blocks as u32,
        64,
    )
    .unwrap();

    let mut graph_tokens = vec![g_first];
    {
        let mut tok = g_first;
        let mut pos = prompt.len();
        for _ in 0..steps {
            let rows = family
                .step(&[SlotUpdate {
                    token: tok,
                    pos: pos as i32,
                    n_total: pos as i32 + 1,
                    block_table: table_graph.clone(),
                    lora_slot: -1,
                }])
                .unwrap();
            assert_eq!(rows.len(), 1);
            tok = argmax(&rows[0]);
            pos += 1;
            graph_tokens.push(tok);
        }
    }

    eprintln!("eager tokens: {eager_tokens:?}");
    eprintln!("graph tokens: {graph_tokens:?}");
    eprintln!(
        "captures={} replays={}",
        family.captures(),
        family.replays()
    );

    assert_eq!(
        graph_tokens, eager_tokens,
        "graphed active slot must match the eager batched decode oracle token-for-token"
    );
    assert_eq!(
        family.captures(),
        1,
        "one graph per batch size. These steps run positions 14..20, which the old \
         ctx bucketing split across buckets 16 and 32 and captured twice"
    );
    assert_eq!(
        family.replays(),
        steps as u64 - 1,
        "every step after the first must replay. Together with the token equality \
         above this is the evidence that context does not shape the graph: one body \
         captured at n_total=15 is replayed at 16..20 and still matches the eager \
         oracle, because the decode attention grid is dim3(n_q, splits) and the \
         length is read on-device from n_total_dev"
    );
}

fn det_bf16(seed: u64, rows: usize, cols: usize, mag: f32, device: &Device) -> Tensor {
    let mut v = Vec::with_capacity(rows * cols);
    for i in 0..rows * cols {
        let mut z = seed
            .wrapping_add(0x9E3779B97F4A7C15u64.wrapping_mul(i as u64 + 1))
            .wrapping_mul(0xBF58476D1CE4E5B9);
        z ^= z >> 29;
        z = z.wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 32;
        let u = ((z & 0xFFFF) as f32 / 65535.0) - 0.5;
        v.push(bf16::from_f32(u * 2.0 * mag));
    }
    Tensor::from_vec(v, (rows, cols), device).unwrap()
}

fn qkv_names(layer: usize, has_v: bool) -> Vec<String> {
    let mut v = vec![format!("l{layer}.q"), format!("l{layer}.k")];
    if has_v {
        v.push(format!("l{layer}.v"));
    }
    v
}

fn qkv_widths(attn: &nv_models::gemma4::Gemma4Attention) -> Vec<usize> {
    if attn.has_v {
        vec![attn.q_dim, attn.kv_dim, attn.kv_dim]
    } else {
        vec![attn.q_dim, attn.kv_dim]
    }
}

fn qkv_specs(model: &Gemma4) -> Vec<LoraModuleSpec> {
    let mut specs = Vec::new();
    for (i, layer) in model.layers().iter().enumerate() {
        let attn = &layer.self_attn;
        let k_in = attn.qkv_proj.in_features();
        for (name, w) in qkv_names(i, attn.has_v).into_iter().zip(qkv_widths(attn)) {
            specs.push(LoraModuleSpec::new(name, k_in, w));
        }
    }
    specs
}

fn synth_qkv_adapter(
    model: &Gemma4,
    rank: usize,
    mag: f32,
    seed: u64,
    zero_b: bool,
) -> LoraAdapter {
    let device = model.device();
    let mut modules = HashMap::new();
    for (i, layer) in model.layers().iter().enumerate() {
        let attn = &layer.self_attn;
        let k_in = attn.qkv_proj.in_features();
        for (name, w) in qkv_names(i, attn.has_v).into_iter().zip(qkv_widths(attn)) {
            let sa = seed ^ ((i as u64) << 8) ^ name.len() as u64;
            let a = det_bf16(sa ^ 0xA, rank, k_in, mag, device);
            let b = if zero_b {
                Tensor::zeros((w, rank), DType::BF16, device).unwrap()
            } else {
                det_bf16(sa ^ 0xB, w, rank, mag, device)
            };
            modules.insert(name, LoraModuleWeights { a, b });
        }
    }
    LoraAdapter {
        scaling: 1.0,
        modules,
    }
}

struct LoraInstall {
    manager: LoraSlotManager,
    dispatch: Arc<LoraDispatch>,
    slot: usize,
}

fn install_qkv_lora(
    model: &Gemma4,
    adapter: &LoraAdapter,
    id: u64,
    max_rank: usize,
    max_tokens: usize,
) -> LoraInstall {
    let device = model.device();
    let specs = qkv_specs(model);
    let dispatch = LoraDispatch::new(device, max_tokens, 1).unwrap();
    let mut manager = LoraSlotManager::new(1, max_rank, &specs, DType::BF16, device).unwrap();
    let slot = manager.activate(id, adapter).unwrap();
    for (i, layer) in model.layers().iter().enumerate() {
        let attn = &layer.self_attn;
        let names = qkv_names(i, attn.has_v);
        let stacks: Vec<&_> = names.iter().map(|n| manager.stack(n).unwrap()).collect();
        let hook = LoraHook::from_stacks(dispatch.clone(), &stacks).unwrap();
        attn.qkv_proj.attach_lora(hook).unwrap();
    }
    LoraInstall {
        manager,
        dispatch,
        slot,
    }
}

fn make_pool(
    model: &Gemma4,
    device: &Device,
    kv_blocks: usize,
    scratch: usize,
) -> Arc<Mutex<PagedKvFp8Pool>> {
    let cfg = PagedPoolConfig::from_gemma4(model.config(), kv_blocks + scratch, BLOCK_SIZE);
    Arc::new(Mutex::new(PagedKvFp8Pool::new(cfg, device).unwrap()))
}

fn prefill_logits(
    model: &Gemma4,
    device: &Device,
    cache: &mut PagedGemma4Cache,
    ids: &[u32],
) -> Vec<f32> {
    let seq = ids.len();
    let tokens = Tensor::from_vec(ids.to_vec(), (1usize, seq), device).unwrap();
    let positions: Vec<i32> = (0..seq as i32).collect();
    let pos = Tensor::from_vec(positions, seq, device).unwrap();
    let logits = model.forward_with_cache(&tokens, &pos, cache).unwrap();
    let v: Vec<f32> = logits
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    v[(seq - 1) * VOCAB..seq * VOCAB].to_vec()
}

fn graph_decode(
    family: &mut Gemma4BatchGraphFamily,
    table: &[u32],
    first: u32,
    start_pos: usize,
    steps: usize,
) -> (Vec<Vec<f32>>, Vec<u32>) {
    let mut logits = Vec::new();
    let mut toks = vec![first];
    let mut tok = first;
    let mut pos = start_pos;
    for _ in 0..steps {
        let rows = family
            .step(&[SlotUpdate {
                token: tok,
                pos: pos as i32,
                n_total: pos as i32 + 1,
                block_table: table.to_vec(),
                lora_slot: -1,
            }])
            .unwrap();
        assert_eq!(rows.len(), 1);
        logits.push(rows[0].clone());
        tok = argmax(&rows[0]);
        pos += 1;
        toks.push(tok);
    }
    (logits, toks)
}

fn max_abs_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    let mut m = 0.0f32;
    for (ra, rb) in a.iter().zip(b) {
        for (x, y) in ra.iter().zip(rb) {
            m = m.max((x - y).abs());
        }
    }
    m
}

#[test]
#[ignore]
fn lora_batch_graph_gates() {
    if std::env::var("NV_BATCH_GRAPH_TEST").ok().as_deref() != Some("1") {
        if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "set NV_BATCH_GRAPH_TEST=1 to run this gate, or \
                 NV_MODELS_ALLOW_SKIP=1 to skip it on purpose. Reporting \
                 \"ok\" in 0.00s for a CUDA-graph test is how three of these \
                 sat green while one of them was broken."
            );
        }
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_BATCH_GRAPH_TEST=1 to run");
        return;
    }
    let Ok(device) = Device::new_cuda(0) else {
        panic!(
            "gemma4_batch_graph is a GPU test and CUDA device 0 did not open. \
             This gate refuses to report success without running."
        );
    };

    static GATES_FIXTURE: std::sync::OnceLock<std::path::PathBuf> =
        std::sync::OnceLock::new();
    let dir = GATES_FIXTURE
        .get_or_init(|| {
            let d = std::env::temp_dir()
                .join(format!("{}-{}", "nv-gemma4-lora-gates-tiny", std::process::id()));
            write_tiny_model(&d);
            d
        })
        .clone();
    let cfg =
        Gemma4Config::from_hf_json_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
            .unwrap();
    let weights = WeightLoader::open_dir(&dir, &device).unwrap();

    let model_base = Arc::new(Gemma4::from_loader(cfg.clone(), &weights, &device).unwrap());
    let model_lora = Arc::new(Gemma4::from_loader(cfg.clone(), &weights, &device).unwrap());

    let rank = 8usize;
    let max_seq = 64usize;
    let prompt: Vec<u32> = (1..15).collect();
    let steps = 4usize;
    let table: Vec<u32> = vec![0, 1, 2, 3];
    let plan = || BucketPlan::new(vec![1]);

    let pool_b = make_pool(&model_base, &device, 8, 2);
    let mut cache_b = PagedGemma4Cache::new(pool_b.clone(), &device).unwrap();
    cache_b.set_block_table(&table).unwrap();
    let base_prefill = prefill_logits(&model_base, &device, &mut cache_b, &prompt);
    let first_b = argmax(&base_prefill);
    let mut fam_b =
        Gemma4BatchGraphFamily::new(model_base.clone(), pool_b, &device, plan(), 8, 64)
            .unwrap();
    let (l_base, tok_base) = graph_decode(&mut fam_b, &table, first_b, prompt.len(), steps);

    let zero = synth_qkv_adapter(&model_lora, rank, 0.25, 100, true);
    let inst = install_qkv_lora(&model_lora, &zero, 1, rank, max_seq);
    let pool_z = make_pool(&model_lora, &device, 8, 2);
    let mut cache_z = PagedGemma4Cache::new(pool_z.clone(), &device).unwrap();
    cache_z.set_block_table(&table).unwrap();
    inst.dispatch
        .set_mapping(&vec![inst.slot as i32; prompt.len()])
        .unwrap();
    let zero_prefill = prefill_logits(&model_lora, &device, &mut cache_z, &prompt);
    let first_z = argmax(&zero_prefill);
    let mut fam_z =
        Gemma4BatchGraphFamily::new(model_lora.clone(), pool_z, &device, plan(), 8, 64)
            .unwrap();
    fam_z.arm_lora(inst.dispatch.clone(), inst.slot).unwrap();
    let (l_zero, tok_zero) = graph_decode(&mut fam_z, &table, first_z, prompt.len(), steps);

    let prefill_zero_diff = base_prefill
        .iter()
        .zip(&zero_prefill)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let zero_diff = max_abs_diff(&l_base, &l_zero);
    eprintln!("=== GATE (a) EXACT: zero adapter (B=0) vs no-LoRA ===");
    eprintln!("  base tokens: {tok_base:?}");
    eprintln!("  zero tokens: {tok_zero:?}");
    eprintln!("  prefill max|base-zero| = {prefill_zero_diff:.3e} (bar: 0)");
    eprintln!("  decode  max|base-zero| = {zero_diff:.3e} (bar: 0)");
    assert_eq!(
        first_z, first_b,
        "zero-adapter prefill argmax must match base"
    );
    assert_eq!(
        prefill_zero_diff, 0.0,
        "zero adapter must be bit-identical at prefill"
    );
    assert_eq!(
        zero_diff, 0.0,
        "zero adapter must be bit-identical through the decode graph"
    );
    assert_eq!(tok_zero, tok_base);

    let nz_a = synth_qkv_adapter(&model_lora, rank, 0.25, 200, false);
    let mut inst = inst;
    let slot_a = inst.manager.activate(2, &nz_a).unwrap();
    assert_eq!(slot_a, inst.slot, "single slot must be reused on swap");
    let pool_nz = make_pool(&model_lora, &device, 8, 2);
    let mut cache_nz = PagedGemma4Cache::new(pool_nz.clone(), &device).unwrap();
    cache_nz.set_block_table(&table).unwrap();
    inst.dispatch
        .set_mapping(&vec![slot_a as i32; prompt.len()])
        .unwrap();
    let nz_prefill = prefill_logits(&model_lora, &device, &mut cache_nz, &prompt);
    let first_nz = argmax(&nz_prefill);
    let mut fam_nz =
        Gemma4BatchGraphFamily::new(model_lora.clone(), pool_nz, &device, plan(), 8, 64)
            .unwrap();
    fam_nz.arm_lora(inst.dispatch.clone(), slot_a).unwrap();
    let (l_nz, tok_nz) = graph_decode(&mut fam_nz, &table, first_nz, prompt.len(), steps);
    let nz_diff = max_abs_diff(&l_base, &l_nz);
    eprintln!("=== GATE (b)(i) CORRECTNESS: nonzero adapter changes decode logits ===");
    eprintln!("  nonzero tokens: {tok_nz:?}");
    eprintln!("  decode max|base-nonzero| = {nz_diff:.4} (bar: > 1e-2)");
    assert!(nz_diff > 1e-2, "nonzero adapter must move the logits");

    let attn0 = &model_lora.layers()[0].self_attn;
    let k_in = attn0.qkv_proj.in_features();
    let m_site = 3usize;
    let x = det_bf16(0x515e, m_site, k_in, 0.5, &device);
    inst.dispatch
        .set_mapping(&vec![slot_a as i32; m_site])
        .unwrap();
    let y_lora = attn0.qkv_proj.forward(&x).unwrap();
    inst.dispatch.disarm();
    let y_base = attn0.qkv_proj.forward(&x).unwrap();
    let meas = (&y_lora - &y_base)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    let x_h = x.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap();
    let widths = qkv_widths(attn0);
    let names = qkv_names(0, attn0.has_v);
    let out_features: usize = widths.iter().sum();
    let mut refd = vec![vec![0f32; out_features]; m_site];
    let mut col = 0usize;
    for (name, w) in names.iter().zip(&widths) {
        let wgt = &nz_a.modules[name];
        let a_h = wgt
            .a
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let b_h = wgt
            .b
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for (mi, xr) in x_h.iter().enumerate() {
            let mut shrink = vec![0f32; rank];
            for (r, sh) in shrink.iter_mut().enumerate() {
                let mut acc = 0f32;
                for (k, xv) in xr.iter().enumerate() {
                    acc += a_h[r][k] * xv;
                }
                *sh = acc;
            }
            for o in 0..*w {
                let mut acc = 0f32;
                for (r, sh) in shrink.iter().enumerate() {
                    acc += b_h[o][r] * sh;
                }
                refd[mi][col + o] = acc;
            }
        }
        col += w;
    }
    let mut num = 0f64;
    let mut den = 0f64;
    for (mr, rr) in meas.iter().zip(&refd) {
        for (mv, rv) in mr.iter().zip(rr) {
            num += (*mv as f64 - *rv as f64).powi(2);
            den += (*rv as f64).powi(2);
        }
    }
    let rel_rms = (num / den.max(1e-30)).sqrt();
    eprintln!("=== GATE (b)(ii) CORRECTNESS: site delta vs independent CPU B*(A*x) ===");
    eprintln!(
        "  elements={} rel-RMS={rel_rms:.3e} (bar: < 5e-2)",
        m_site * out_features
    );
    assert!(
        rel_rms < 5e-2,
        "site delta must match CPU reference (rel-RMS {rel_rms})"
    );

    let nz_b = synth_qkv_adapter(&model_lora, rank, 0.25, 300, false);
    inst.dispatch
        .set_mapping(&vec![slot_a as i32; m_site])
        .unwrap();
    let y_a = attn0
        .qkv_proj
        .forward(&x)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let slot_b = inst.manager.activate(3, &nz_b).unwrap();
    inst.dispatch
        .set_mapping(&vec![slot_b as i32; m_site])
        .unwrap();
    let y_b = attn0
        .qkv_proj
        .forward(&x)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let slot_a2 = inst.manager.activate(2, &nz_a).unwrap();
    inst.dispatch
        .set_mapping(&vec![slot_a2 as i32; m_site])
        .unwrap();
    let y_a2 = attn0
        .qkv_proj
        .forward(&x)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let ab = max_abs_diff(&y_a, &y_b);
    let aa = max_abs_diff(&y_a, &y_a2);
    eprintln!("=== GATE (c) HOT-SWAP: A -> B -> A on the resident slot ===");
    eprintln!("  slots: A={slot_a} B={slot_b} A'={slot_a2} (single resident slot)");
    eprintln!("  max|A - B|  = {ab:.4}   (bar: > 1e-2, B must differ)");
    eprintln!("  max|A - A'| = {aa:.3e} (bar: 0, re-arm reproduces A)");
    assert!(
        ab > 1e-2,
        "adapter B must produce a different output than A"
    );
    assert_eq!(
        aa, 0.0,
        "re-arming A after a swap must reproduce A bit-exactly"
    );

    eprintln!("ALL LORA STAGE-1 GATES PASSED");
    let _ = inst;
}

fn install_qkv_lora_multi(
    model: &Gemma4,
    max_loras: usize,
    max_rank: usize,
    max_tokens: usize,
) -> (LoraSlotManager, Arc<LoraDispatch>) {
    let device = model.device();
    let specs = qkv_specs(model);
    let dispatch = LoraDispatch::new(device, max_tokens, max_loras).unwrap();
    let manager = LoraSlotManager::new(max_loras, max_rank, &specs, DType::BF16, device).unwrap();
    for (i, layer) in model.layers().iter().enumerate() {
        let attn = &layer.self_attn;
        let names = qkv_names(i, attn.has_v);
        let stacks: Vec<&_> = names.iter().map(|n| manager.stack(n).unwrap()).collect();
        let hook = LoraHook::from_stacks(dispatch.clone(), &stacks).unwrap();
        attn.qkv_proj.attach_lora(hook).unwrap();
    }
    (manager, dispatch)
}

fn row_stats(meas: &[f32], refr: &[f32]) -> (f64, f32) {
    let mut num = 0f64;
    let mut den = 0f64;
    let mut max_abs = 0f32;
    for (m, r) in meas.iter().zip(refr) {
        num += (*m as f64 - *r as f64).powi(2);
        den += (*r as f64).powi(2);
        max_abs = max_abs.max((m - r).abs());
    }
    ((num / den.max(1e-30)).sqrt(), max_abs)
}

fn upd(token: u32, pos: usize, block: u32, lora_slot: i32) -> SlotUpdate {
    SlotUpdate {
        token,
        pos: pos as i32,
        n_total: pos as i32 + 1,
        block_table: vec![block],
        lora_slot,
    }
}

#[test]
#[ignore]
fn lora_batch_graph_multi_gates() {
    if std::env::var("NV_BATCH_GRAPH_TEST").ok().as_deref() != Some("1") {
        if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "set NV_BATCH_GRAPH_TEST=1 to run this gate, or \
                 NV_MODELS_ALLOW_SKIP=1 to skip it on purpose. Reporting \
                 \"ok\" in 0.00s for a CUDA-graph test is how three of these \
                 sat green while one of them was broken."
            );
        }
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_BATCH_GRAPH_TEST=1 to run");
        return;
    }

    std::env::set_var("NV_DETERMINISTIC", "1");
    let Ok(device) = Device::new_cuda(0) else {
        panic!(
            "gemma4_batch_graph is a GPU test and CUDA device 0 did not open. \
             This gate refuses to report success without running."
        );
    };

    static MULTI_FIXTURE: std::sync::OnceLock<std::path::PathBuf> =
        std::sync::OnceLock::new();
    let dir = MULTI_FIXTURE
        .get_or_init(|| {
            let d = std::env::temp_dir()
                .join(format!("{}-{}", "nv-gemma4-lora-multi-tiny", std::process::id()));
            write_tiny_model(&d);
            d
        })
        .clone();
    let cfg =
        Gemma4Config::from_hf_json_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
            .unwrap();
    let weights = WeightLoader::open_dir(&dir, &device).unwrap();

    let model_base = Arc::new(Gemma4::from_loader(cfg.clone(), &weights, &device).unwrap());
    let model_lora = Arc::new(Gemma4::from_loader(cfg.clone(), &weights, &device).unwrap());

    let rank = 8usize;
    let max_loras = 2usize;
    let max_tokens = 64usize;
    let prompt: Vec<u32> = (1..9).collect();
    let plen = prompt.len();
    let pos = plen;
    let plan = || BucketPlan::new(vec![4]);

    let (mut manager, dispatch) = install_qkv_lora_multi(&model_lora, max_loras, rank, max_tokens);
    let ad_a = synth_qkv_adapter(&model_lora, rank, 0.25, 200, false);
    let ad_b = synth_qkv_adapter(&model_lora, rank, 0.25, 300, false);
    let slot_a = manager.activate(10, &ad_a).unwrap();
    let slot_b = manager.activate(11, &ad_b).unwrap();
    assert_ne!(
        slot_a, slot_b,
        "A and B must occupy distinct resident slots"
    );
    eprintln!(
        "=== Stage 2 setup: max_loras={max_loras} resident A=slot{slot_a} B=slot{slot_b} ==="
    );

    let pool = make_pool(&model_lora, &device, 9, 4);
    let mut fam =
        Gemma4BatchGraphFamily::new(model_lora.clone(), pool.clone(), &device, plan(), 9, 64)
            .unwrap();
    fam.arm_lora_multi(dispatch.clone()).unwrap();

    let pool_base = make_pool(&model_base, &device, 1, 4);
    let mut fam_base = Gemma4BatchGraphFamily::new(
        model_base.clone(),
        pool_base.clone(),
        &device,
        plan(),
        1,
        64,
    )
    .unwrap();

    let prefill_cohort = |cache: &mut PagedGemma4Cache, slot: Option<i32>| -> Vec<f32> {
        match slot {
            Some(s) => dispatch.set_mapping(&vec![s; plen]).unwrap(),
            None => dispatch.disarm(),
        }
        prefill_logits(&model_lora, &device, cache, &prompt)
    };
    let new_cache = |pool: &Arc<Mutex<PagedKvFp8Pool>>, block: u32| -> PagedGemma4Cache {
        let mut c = PagedGemma4Cache::new(pool.clone(), &device).unwrap();
        c.set_block_table(&[block]).unwrap();
        c
    };

    let mut c_a_het = new_cache(&pool, 0);
    let mut c_a_ref = new_cache(&pool, 1);
    let mut c_b_het = new_cache(&pool, 2);
    let mut c_b_ref = new_cache(&pool, 3);
    let mut c_base_het = new_cache(&pool, 4);
    let mut c_base_ref = new_cache(&pool, 5);
    let la = prefill_cohort(&mut c_a_het, Some(slot_a as i32));
    let _ = prefill_cohort(&mut c_a_ref, Some(slot_a as i32));
    let lb = prefill_cohort(&mut c_b_het, Some(slot_b as i32));
    let _ = prefill_cohort(&mut c_b_ref, Some(slot_b as i32));
    let lbase = prefill_cohort(&mut c_base_het, None);
    let _ = prefill_cohort(&mut c_base_ref, None);
    let tok_a = argmax(&la);
    let tok_b = argmax(&lb);
    let tok_base = argmax(&lbase);

    let mut c_base_true = new_cache(&pool_base, 0);
    let lbase_true = prefill_logits(&model_base, &device, &mut c_base_true, &prompt);
    let tok_base_true = argmax(&lbase_true);
    assert_eq!(
        tok_base_true, tok_base,
        "base prefill argmax must agree across models"
    );
    let true_base = fam_base
        .step(&[upd(tok_base, pos, 0, -1)])
        .unwrap()
        .remove(0);

    let ref_a = fam
        .step(&[upd(tok_a, pos, 1, slot_a as i32)])
        .unwrap()
        .remove(0);
    let ref_b = fam
        .step(&[upd(tok_b, pos, 3, slot_b as i32)])
        .unwrap()
        .remove(0);
    let ref_base = fam.step(&[upd(tok_base, pos, 5, -1)]).unwrap().remove(0);
    assert_eq!(
        fam.captures(),
        1,
        "single (bucket=4,ctx=16) shape must capture exactly once"
    );
    let nodes_after_refs = fam.node_count();

    let het = fam
        .step(&[
            upd(tok_a, pos, 0, slot_a as i32),
            upd(tok_b, pos, 2, slot_b as i32),
            upd(tok_base, pos, 4, -1),
        ])
        .unwrap();
    assert_eq!(het.len(), 3);

    let (_, a_exact) = row_stats(&het[2], &true_base);
    let (_, refbase_exact) = row_stats(&ref_base, &true_base);
    eprintln!("=== GATE (a) EXACT: in-batch -1 row vs TRUE no-LoRA base ===");
    eprintln!("  batch=[A,B,-1]; adapters A/B active on the other two rows");
    eprintln!("  max|true_base - het_row(-1)| = {a_exact:.3e} (bar: 0)");
    eprintln!("  max|true_base - ref(-1)|     = {refbase_exact:.3e} (bar: 0)");
    assert_eq!(
        a_exact, 0.0,
        "in-batch -1 row must be bit-identical to true base"
    );
    assert_eq!(
        refbase_exact, 0.0,
        "single -1 row must be bit-identical to true base"
    );

    let (a_rms, a_max) = row_stats(&het[0], &ref_a);
    let (b_rms, b_max) = row_stats(&het[1], &ref_b);
    let (_, ab_diff) = row_stats(&ref_a, &ref_b);
    let (_, a_vs_base) = row_stats(&het[0], &true_base);
    eprintln!("=== GATE (b) HETEROGENEITY: het rows vs A-alone / B-alone ===");
    eprintln!("  row0 (A): rel-RMS={a_rms:.3e} max-abs={a_max:.3e} (bar: < 5e-2)");
    eprintln!("  row1 (B): rel-RMS={b_rms:.3e} max-abs={b_max:.3e} (bar: < 5e-2)");
    eprintln!("  sanity max|A_ref - B_ref| = {ab_diff:.4} (adapters differ, bar > 1e-2)");
    eprintln!("  sanity max|A_row - base|  = {a_vs_base:.4} (adapter moves logits, bar > 1e-2)");
    assert!(a_rms < 5e-2, "het A row must match A-alone reference");
    assert!(b_rms < 5e-2, "het B row must match B-alone reference");
    assert!(
        ab_diff > 1e-2,
        "adapters A and B must produce different logits"
    );
    assert!(a_vs_base > 1e-2, "adapter A must move the logits off base");

    let attn0 = &model_lora.layers()[0].self_attn;
    let k_in = attn0.qkv_proj.in_features();
    let x = det_bf16(0x777, 4, k_in, 0.5, &device);
    dispatch
        .set_mapping(&[slot_a as i32, slot_b as i32, -1, -1])
        .unwrap();
    let y_lora = attn0.qkv_proj.forward(&x).unwrap();
    dispatch.disarm();
    let y_base = attn0.qkv_proj.forward(&x).unwrap();
    let meas = (&y_lora - &y_base)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let x_h = x.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap();
    let widths = qkv_widths(attn0);
    let names = qkv_names(0, attn0.has_v);
    let out_features: usize = widths.iter().sum();

    let cpu_delta = |adapter: &LoraAdapter, xr: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; out_features];
        let mut col = 0usize;
        for (name, w) in names.iter().zip(&widths) {
            let wgt = &adapter.modules[name];
            let a_h = wgt
                .a
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            let b_h = wgt
                .b
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            let mut shrink = vec![0f32; rank];
            for (r, sh) in shrink.iter_mut().enumerate() {
                let mut acc = 0f32;
                for (k, xv) in xr.iter().enumerate() {
                    acc += a_h[r][k] * xv;
                }
                *sh = acc;
            }
            for o in 0..*w {
                let mut acc = 0f32;
                for (r, sh) in shrink.iter().enumerate() {
                    acc += b_h[o][r] * sh;
                }
                out[col + o] = acc;
            }
            col += w;
        }
        out
    };
    let ref0 = cpu_delta(&ad_a, &x_h[0]);
    let ref1 = cpu_delta(&ad_b, &x_h[1]);
    let (o0_rms, _) = row_stats(&meas[0], &ref0);
    let (o1_rms, _) = row_stats(&meas[1], &ref1);
    let row2_maxabs = meas[2].iter().fold(0f32, |m, v| m.max(v.abs()));
    let row3_maxabs = meas[3].iter().fold(0f32, |m, v| m.max(v.abs()));
    eprintln!("=== GATE (b) site oracle: per-row delta vs CPU B*(A*x) ===");
    eprintln!("  row0 (A) rel-RMS={o0_rms:.3e}  row1 (B) rel-RMS={o1_rms:.3e} (bar < 5e-2)");
    eprintln!(
        "  row2 (-1) max|delta|={row2_maxabs:.3e}  row3 (-1) max|delta|={row3_maxabs:.3e} (bar 0)"
    );
    assert!(
        o0_rms < 5e-2 && o1_rms < 5e-2,
        "per-row grouped-GEMM must match CPU oracle"
    );
    assert_eq!(row2_maxabs, 0.0, "row2 (-1) must receive zero delta");
    assert_eq!(row3_maxabs, 0.0, "row3 (-1) must receive zero delta");

    let cap_before = fam.captures();
    let replays_before = fam.replays();
    let nodes_before = fam.node_count();

    let mut c_b2 = new_cache(&pool, 6);
    let mut c_base2 = new_cache(&pool, 7);
    let mut c_a2 = new_cache(&pool, 8);
    let tok_b2 = argmax(&prefill_cohort(&mut c_b2, Some(slot_b as i32)));
    let tok_base2 = argmax(&prefill_cohort(&mut c_base2, None));
    let tok_a2 = argmax(&prefill_cohort(&mut c_a2, Some(slot_a as i32)));
    let het2 = fam
        .step(&[
            upd(tok_b2, pos, 6, slot_b as i32),
            upd(tok_base2, pos, 7, -1),
            upd(tok_a2, pos, 8, slot_a as i32),
        ])
        .unwrap();
    let cap_after = fam.captures();
    let nodes_after = fam.node_count();
    let (c_b_rms, _) = row_stats(&het2[0], &ref_b);
    let (_, c_base_exact) = row_stats(&het2[1], &true_base);
    let (c_a_rms, _) = row_stats(&het2[2], &ref_a);
    eprintln!("=== GATE (c) NO RE-CAPTURE: [A,B,-1] captured, [B,-1,A] replayed ===");
    eprintln!("  captures: before={cap_before} after={cap_after} (must stay == 1)");
    eprintln!(
        "  replays:  before={replays_before} after={} (+1 for the new assignment)",
        fam.replays()
    );
    eprintln!(
        "  node_count: after_refs={nodes_after_refs} before_c={nodes_before} after_c={nodes_after} (constant)"
    );
    eprintln!("  permuted per-row: row0(B) rel-RMS={c_b_rms:.3e}  row2(A) rel-RMS={c_a_rms:.3e}  row1(-1) max-abs={c_base_exact:.3e}");
    assert_eq!(cap_before, 1, "graph must have captured exactly once");
    assert_eq!(
        cap_after, cap_before,
        "a different per-row assignment must NOT re-capture"
    );
    assert_eq!(
        nodes_before, nodes_after,
        "captured node count must be constant across replays"
    );
    assert_eq!(
        nodes_after_refs, nodes_before,
        "node count constant since first capture"
    );
    assert!(
        c_b_rms < 5e-2 && c_a_rms < 5e-2,
        "permuted assignment must produce correct per-row outputs"
    );
    assert_eq!(
        c_base_exact, 0.0,
        "permuted -1 row must stay bit-identical to base"
    );

    dispatch
        .set_mapping(&[slot_a as i32, slot_b as i32, -1, -1])
        .unwrap();
    let d1 = attn0
        .qkv_proj
        .forward(&x)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    dispatch
        .set_mapping(&[slot_a as i32, slot_b as i32, -1, -1])
        .unwrap();
    let d2 = attn0
        .qkv_proj
        .forward(&x)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let det_max = max_abs_diff(&d1, &d2);
    eprintln!("=== GATE (d) DETERMINISM: NV_DETERMINISTIC=1, same het mapping twice ===");
    eprintln!("  max|run1 - run2| = {det_max:.3e} (bar: 0)");
    assert_eq!(
        det_max, 0.0,
        "deterministic heterogeneous apply must be bit-identical"
    );

    eprintln!("ALL LORA STAGE-2 GATES PASSED");
    let _ = (manager, c_a_ref, c_b_ref, c_base_ref, c_base_true);
}
