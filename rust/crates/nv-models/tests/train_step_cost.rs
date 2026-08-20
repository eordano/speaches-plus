#![cfg(feature = "cuda")]

mod common;
use common::config_json;
use common::HEAD_DIM_128 as HEAD_DIM;
use common::HIDDEN_128 as HIDDEN;
use common::INTER_256 as INTER;
use common::LcgTop24TwoSided as Lcg;
use common::N_KV_1 as N_KV;
use common::N_LAYERS_2 as N_LAYERS;
use common::N_Q_2 as N_Q;
use common::ones_tensor;
use common::rand_tensor;
use common::VOCAB_512 as VOCAB;
use candle_core::{DType, Device, Tensor};
use nv_models::train_runner::{run, select_device, TrainArgs};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const SEQ: usize = 32;

fn write_tiny_model(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    let mut rng = Lcg(0x5eed_cafe_f00d_0087);
    let mut t: HashMap<String, Tensor> = HashMap::new();
    t.insert(
        "model.language_model.embed_tokens.weight".into(),
        rand_tensor(&mut rng, (VOCAB, HIDDEN), 1.0),
    );
    t.insert("model.language_model.norm.weight".into(), ones_tensor(HIDDEN));
    t.insert(
        "lm_head.weight".into(),
        rand_tensor(&mut rng, (VOCAB, HIDDEN), 1.0),
    );
    for i in 0..N_LAYERS {
        let p = format!("model.language_model.layers.{i}");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            t.insert(format!("{p}.{norm}.weight"), ones_tensor(HIDDEN));
        }
        t.insert(format!("{p}.layer_scalar"), ones_tensor(1));
        t.insert(
            format!("{p}.self_attn.q_proj.weight"),
            rand_tensor(&mut rng, (N_Q * HEAD_DIM, HIDDEN), 0.3),
        );
        t.insert(
            format!("{p}.self_attn.k_proj.weight"),
            rand_tensor(&mut rng, (N_KV * HEAD_DIM, HIDDEN), 0.3),
        );
        t.insert(
            format!("{p}.self_attn.v_proj.weight"),
            rand_tensor(&mut rng, (N_KV * HEAD_DIM, HIDDEN), 0.3),
        );
        t.insert(
            format!("{p}.self_attn.o_proj.weight"),
            rand_tensor(&mut rng, (HIDDEN, N_Q * HEAD_DIM), 0.3),
        );
        t.insert(format!("{p}.self_attn.q_norm.weight"), ones_tensor(HEAD_DIM));
        t.insert(format!("{p}.self_attn.k_norm.weight"), ones_tensor(HEAD_DIM));
        t.insert(
            format!("{p}.mlp.gate_proj.weight"),
            rand_tensor(&mut rng, (INTER, HIDDEN), 0.3),
        );
        t.insert(
            format!("{p}.mlp.up_proj.weight"),
            rand_tensor(&mut rng, (INTER, HIDDEN), 0.3),
        );
        t.insert(
            format!("{p}.mlp.down_proj.weight"),
            rand_tensor(&mut rng, (HIDDEN, INTER), 0.3),
        );
    }
    candle_core::safetensors::save(&t, dir.join("model.safetensors")).unwrap();
}

fn write_dataset(path: &Path, examples: usize) {
    let mut rng = Lcg(0xd00d_1234_5678_9abc);
    let mut out = String::new();
    for _ in 0..examples {
        let ids: Vec<u32> = (0..SEQ)
            .map(|_| ((rng.next_f32().abs() * VOCAB as f32) as u32) % VOCAB as u32)
            .collect();
        out.push_str(&format!("{{\"ids\":{ids:?}}}\n"));
    }
    std::fs::write(path, out).unwrap();
}

fn args(base: &Path, data: &Path, out: &Path, steps: usize) -> TrainArgs {
    TrainArgs {
        base: base.to_path_buf(),
        data: data.to_path_buf(),
        out: out.to_path_buf(),
        rank: 8,
        alpha: 16.0,
        targets: vec!["q".into(), "v".into()],
        steps,
        lr: 1e-4,
        seed: 7,
    }
}

fn ms_per_step(base: &Path, data: &Path, dir: &Path, few: usize, many: usize) -> f64 {
    let t0 = std::time::Instant::now();
    run(&args(base, data, &dir.join("out-few"), few)).unwrap();
    let few_s = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    run(&args(base, data, &dir.join("out-many"), many)).unwrap();
    let many_s = t1.elapsed().as_secs_f64();
    (many_s - few_s) * 1e3 / (many - few) as f64
}

#[test]
#[ignore = "a measurement, not a gate; set NV_TRAIN_COST=1"]
fn a_training_step_costs_this_much_per_example() {
    if std::env::var("NV_TRAIN_COST").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_TRAIN_COST=1");
    }
    let dir: PathBuf = std::env::temp_dir().join(format!("nv-train-cost-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join("base");
    write_tiny_model(&base);

    println!("  examples | ms/step | ms/example");
    let mut prev: Option<(usize, f64)> = None;
    for examples in [1usize, 8, 32] {
        let data = dir.join(format!("data-{examples}.jsonl"));
        write_dataset(&data, examples);

        let many = 10 + (1600 / examples).max(50);
        let ms = ms_per_step(&base, &data, &dir, 10, many);
        assert!(
            ms > 0.0,
            "{examples} examples measured at {ms:.2} ms/step: the difference method is \
             being swamped by run-to-run variance, so the number means nothing -- raise \
             the step counts rather than reading it"
        );
        println!("  {examples:8} | {ms:7.2} | {:10.3}", ms / examples as f64);
        if let Some((pn, pms)) = prev {
            let grew = ms / pms;
            let n_grew = examples as f64 / pn as f64;
            println!(
                "           examples x{n_grew:.0} -> step x{grew:.2}  (x{n_grew:.0} means the batch \
                 dimension is the whole cost)"
            );
        }
        prev = Some((examples, ms));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "needs cuda; set NV_TRAIN_COST=1"]
fn the_three_ops_the_batch_port_died_on_work_when_positions_are_u32() {
    if std::env::var("NV_TRAIN_COST").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_TRAIN_COST=1");
    }
    let d = Device::new_cuda(0).expect("cuda device 0");
    let seq = 8usize;
    let batch = 4usize;
    let pos = Tensor::from_vec((0..seq as u32).collect::<Vec<_>>(), seq, &d).unwrap();

    let catted = Tensor::cat(&vec![&pos; batch], 0).expect("cat: was copy2d CUDA_ERROR_NOT_FOUND");
    assert_eq!(catted.dims(), &[batch * seq]);

    let bcast = pos
        .unsqueeze(0)
        .unwrap()
        .broadcast_as((batch, seq))
        .unwrap()
        .contiguous()
        .expect("contiguous: was copy_strided_src CUDA_ERROR_NOT_FOUND")
        .flatten_all()
        .unwrap();
    assert_eq!(bcast.to_vec1::<u32>().unwrap(), catted.to_vec1::<u32>().unwrap());

    let (n_kv, n_q, hd) = (2usize, 4usize, 3usize);
    let k = Tensor::ones((batch * n_kv, seq, hd), DType::F32, &d).unwrap();
    let map: Vec<u32> = (0..batch * n_q)
        .map(|i| ((i / n_q) * n_kv + (i % n_q) / (n_q / n_kv)) as u32)
        .collect();
    let map_t = Tensor::from_vec(map, batch * n_q, &d).unwrap();
    let expanded = k.index_select(&map_t, 0).expect("index_select on dim 0 of 3-D");
    assert_eq!(expanded.dims(), &[batch * n_q, seq, hd]);
}

#[test]
#[ignore = "needs a tiny base; set NV_TRAIN_COST=1"]
fn training_actually_reduces_the_loss_on_whichever_device_is_selected() {
    if std::env::var("NV_TRAIN_COST").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_TRAIN_COST=1");
    }
    let dir = std::env::temp_dir().join(format!("nv-train-learns-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let base = dir.join("base");
    write_tiny_model(&base);
    let data = dir.join("data.jsonl");
    write_dataset(&data, 8);

    let mut a = args(&base, &data, &dir.join("out"), 12);
    a.lr = 1e-2;
    let summary = run(&a).unwrap();
    let losses = summary.losses;
    assert_eq!(losses.len(), 12);

    let first = losses[0];
    let last = losses[losses.len() - 1];
    eprintln!("[learns] device {:?}  first {first:.6e}  last {last:.6e}", select_device().unwrap());
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "a non-finite loss means the step diverged, not that it trained: {losses:?}"
    );
    assert!(
        last < first,
        "loss did not fall over 12 steps ({first:.6e} -> {last:.6e}). Either no gradient \
         reached the LoRA parameters -- which is what a BackpropOp::none path on the \
         forward would do, silently -- or the optimiser is not stepping."
    );
    let drop = (first - last) / first.abs();
    assert!(
        drop > 1e-4,
        "loss moved by only {drop:.3e} relative over 12 steps at lr=1e-2, which is closer \
         to numerical drift than to learning"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
