#![cfg(feature = "wgpu")]

mod common;
#[path = "../../nv-train/tests/mtp_fixture/mod.rs"]
mod mtp_fixture;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use candle_core::Device;
use common::mtp_tiny_weights_q3d as mtp_tiny_weights;
use common::q3d_lora_synth::{synth_adapter, write_adapter, Fill};
use common::tiny_config_q3d_mixed_layers as tiny_config;
use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_dense_wgpu::{
    mtp_speculator_deltas_from_adapter_dir, MtpHostWeights, Q3dLora, Qwen3_5DenseWgpu,
    MTP_SPECULATOR_LORA_TARGETS_ARE_THE_8_HEAD_PROJECTIONS_NORMS_EMBED_AND_LM_HEAD_ARE_NEVER_TARGETS as HEAD_TARGETS,
};
use nv_models::qwen3_5_moe::Qwen3_5DenseConfig;
use nv_train::qwen38::{run_jsonl, run_jsonl_with_mtp_cotrain, Q38JsonlArgs, Q38MtpCoTrainArgs};
use nv_train::qwen38_fixture::write_tiny_checkpoint;
use nv_train::{SPECULATOR_KIND_KEY, SPECULATOR_SUBDIR};

static ENV_LOCK_THESE_TESTS_MUTATE_PROCESS_GLOBAL_ENV_SO_MUST_NOT_INTERLEAVE:
    std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK_THESE_TESTS_MUTATE_PROCESS_GLOBAL_ENV_SO_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

const TINY_MAX_SEQ: usize = 64;

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "nv-q3d-specmerge-{name}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

const PROSE: &str =
    "The battle system alternates a top down command map with a third person action phase, so a \
     single turn mixes planning and marksmanship. Critics praised the art direction and the \
     score, while faulting a story that reuses the structure of the first game.";

fn write_jsonl(root: &Path) -> PathBuf {
    let row = |s: &str| format!("{}\n", serde_json::json!({ "text": s }));
    let data = root.join("train.jsonl");
    std::fs::write(
        &data,
        format!(
            "{}{}{}",
            row(&PROSE[0..48]),
            row(&PROSE[48..110]),
            row(&PROSE[110..150])
        ),
    )
    .unwrap();
    data
}

fn jsonl_args(base: &Path, data: &Path, out: PathBuf) -> Q38JsonlArgs {
    Q38JsonlArgs {
        base: base.to_path_buf(),
        data: data.to_path_buf(),
        eval_data: None,
        out,
        grad_layers: 2,
        rank: 4,
        alpha: 4.0,
        targets: vec![
            "q_proj".into(),
            "v_proj".into(),
            "gate_proj".into(),
            "down_proj".into(),
        ],
        steps: 5,
        lr: 5e-3,
        b_lr_ratio: 1.0,
        seed: 19,
    }
}

fn co_args() -> Q38MtpCoTrainArgs {
    Q38MtpCoTrainArgs {
        rank: 4,
        alpha: 4.0,
        speculator_lr_ratio: 4.0,
    }
}

fn fixture_serving_config(raw: &str) -> Qwen3_5DenseConfig {
    let mut v: serde_json::Value = serde_json::from_str(raw).unwrap();
    v["eos_token_id"] = serde_json::json!(1);
    v["text_config"]["max_position_embeddings"] = serde_json::json!(TINY_MAX_SEQ);
    Qwen3_5DenseConfig::from_hf_json_str(&v.to_string())
        .expect("the training fixture config plus an eos id parses as the serving config")
}

fn cotrained_fixture(name: &str) -> (PathBuf, PathBuf, PathBuf) {
    let root = scratch(name);
    let base = root.join("base");
    let cfg = nv_train::qwen38_fixture::write_tiny_checkpoint_with_head_dim(&base, 32);
    mtp_fixture::write_tiny_mtp_head(&base, &cfg);
    let data = write_jsonl(&root);
    let out = root.join("adapter");
    let (_, co) =
        run_jsonl_with_mtp_cotrain(&jsonl_args(&base, &data, out.clone()), &co_args(), &Device::Cpu)
            .expect("the landed co-train fn on the landed fixture");
    assert_eq!(co.speculator_dir, out.join(SPECULATOR_SUBDIR));
    (root, base, out)
}

fn write_speculator(
    adapter_dir: &Path,
    kind: &str,
    rank: usize,
    alpha: f64,
    mods: &BTreeMap<String, (Vec<f32>, Vec<f32>)>,
) {
    let main_path = adapter_dir.join("adapter_config.json");
    let mut main_cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&main_path).unwrap()).unwrap();
    main_cfg[SPECULATOR_KIND_KEY] = serde_json::Value::String(kind.to_string());
    std::fs::write(&main_path, serde_json::to_string_pretty(&main_cfg).unwrap()).unwrap();

    let sdir = adapter_dir.join(SPECULATOR_SUBDIR);
    std::fs::create_dir_all(&sdir).unwrap();
    let mut scfg = serde_json::json!({
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_rslora": false,
        "use_dora": false,
        "modules_to_save": serde_json::Value::Null,
        "target_modules": ["fc", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    });
    scfg[SPECULATOR_KIND_KEY] = serde_json::Value::String(kind.to_string());
    std::fs::write(
        sdir.join("adapter_config.json"),
        serde_json::to_string_pretty(&scfg).unwrap(),
    )
    .unwrap();
    let mut t = std::collections::HashMap::new();
    for (module, (a, b)) in mods {
        let k = a.len() / rank;
        let n = b.len() / rank;
        t.insert(
            format!("base_model.model.{module}.lora_A.weight"),
            candle_core::Tensor::from_vec(a.clone(), (rank, k), &Device::Cpu).unwrap(),
        );
        t.insert(
            format!("base_model.model.{module}.lora_B.weight"),
            candle_core::Tensor::from_vec(b.clone(), (n, rank), &Device::Cpu).unwrap(),
        );
    }
    candle_core::safetensors::save(&t, sdir.join("adapter_model.safetensors")).unwrap();
}

fn head_shapes_in_head_bits_order(cfg: &Qwen3_5DenseConfig) -> [(&'static str, usize, usize); 8] {
    let hidden = cfg.hidden_size;
    let hd = cfg.head_dim;
    let q_out = cfg.num_attention_heads * hd * 2;
    let kv_out = cfg.num_key_value_heads * hd;
    let inter = cfg.intermediate_size;
    [
        ("mtp.fc", hidden, 2 * hidden),
        ("mtp.layers.0.self_attn.q_proj", q_out, hidden),
        ("mtp.layers.0.self_attn.k_proj", kv_out, hidden),
        ("mtp.layers.0.self_attn.v_proj", kv_out, hidden),
        ("mtp.layers.0.self_attn.o_proj", hidden, cfg.num_attention_heads * hd),
        ("mtp.layers.0.mlp.gate_proj", inter, hidden),
        ("mtp.layers.0.mlp.up_proj", inter, hidden),
        ("mtp.layers.0.mlp.down_proj", hidden, inter),
    ]
}

fn head_mods(
    cfg: &Qwen3_5DenseConfig,
    rank: usize,
    fill_b: impl Fn(usize) -> f32,
) -> BTreeMap<String, (Vec<f32>, Vec<f32>)> {
    let mut mods = BTreeMap::new();
    for (name, out, k) in head_shapes_in_head_bits_order(cfg) {
        let a: Vec<f32> = (0..rank * k)
            .map(|i| if i % 3 == 0 { 0.125 } else { -0.0625 })
            .collect();
        let b: Vec<f32> = (0..out * rank).map(&fill_b).collect();
        mods.insert(name.to_string(), (a, b));
    }
    mods
}

fn head_bits(w: &MtpHostWeights) -> Vec<(&'static str, Vec<u16>)> {
    let lin = |l: &q3d::HostDenseLin| match l {
        q3d::HostDenseLin::Bf16(b) => b.w.clone(),
        _ => panic!("tiny mtp host weights are all bf16"),
    };
    vec![
        ("fc", w.fc.w.clone()),
        ("q", lin(&w.attn.q)),
        ("k", lin(&w.attn.k)),
        ("v", lin(&w.attn.v)),
        ("o", lin(&w.attn.o)),
        ("gate", lin(&w.mlp.gate)),
        ("up", lin(&w.mlp.up)),
        ("down", lin(&w.mlp.down)),
    ]
}

fn assert_head_bits_equal(want: &MtpHostWeights, got: &MtpHostWeights, ctx: &str) {
    for ((name, a), (_, b)) in head_bits(want).iter().zip(head_bits(got).iter()) {
        assert_eq!(
            a, b,
            "{ctx}: drafter weight {name} is not bitwise identical to the untouched head"
        );
    }
}

#[test]
fn an_artifact_without_speculator_yields_no_deltas_and_an_untagged_subdir_refuses() {
    let cfg = tiny_config();
    let synth = synth_adapter(&cfg, 0x5bec_0001, Fill::Coarse, None);
    let dir = write_adapter("specnone", &synth);
    assert!(
        mtp_speculator_deltas_from_adapter_dir(&dir.0)
            .expect("a plain adapter loads clean")
            .is_none(),
        "an artifact without speculator/ must merge nothing so attach stays on today's path"
    );

    std::fs::create_dir_all(dir.0.join(SPECULATOR_SUBDIR)).unwrap();
    let err = mtp_speculator_deltas_from_adapter_dir(&dir.0)
        .expect_err("a speculator/ subdir without the main-config kind key is malformed");
    assert!(
        format!("{err:#}").contains(SPECULATOR_KIND_KEY),
        "the refusal must name the missing kind key; got: {err:#}"
    );
}

#[test]
fn an_eagle3_kind_artifact_refuses_on_the_mtp_drafter_engine() {
    let cfg = tiny_config();
    let synth = synth_adapter(&cfg, 0x5bec_0002, Fill::Coarse, None);
    let dir = write_adapter("speceagle", &synth);
    write_speculator(&dir.0, "eagle3", 4, 4.0, &head_mods(&cfg, 4, |_| 0.0));
    let err = mtp_speculator_deltas_from_adapter_dir(&dir.0)
        .expect_err("an eagle3 speculator has no qwen3.8 MTP head to merge into");
    assert!(
        format!("{err:#}").contains("eagle3"),
        "the refusal must name the mismatched kind; got: {err:#}"
    );
}

const MINUS_ZERO_BF16_BITS_REAL_CHECKPOINTS_CARRY_THEM_BUT_THE_LCG_FIXTURE_NEVER_DOES: u16 =
    0x8000;

#[test]
fn a_zero_delta_speculator_merges_to_bitwise_identical_head_weights() {
    let cfg = tiny_config();
    let mut want = mtp_tiny_weights(&cfg, 0x2e40_0001);
    let mut got = mtp_tiny_weights(&cfg, 0x2e40_0001);
    want.fc.w[0] = MINUS_ZERO_BF16_BITS_REAL_CHECKPOINTS_CARRY_THEM_BUT_THE_LCG_FIXTURE_NEVER_DOES;
    got.fc.w[0] = MINUS_ZERO_BF16_BITS_REAL_CHECKPOINTS_CARRY_THEM_BUT_THE_LCG_FIXTURE_NEVER_DOES;
    let synth = synth_adapter(&cfg, 0x5bec_0003, Fill::Coarse, None);
    let dir = write_adapter("speczero", &synth);
    write_speculator(
        &dir.0,
        nv_train::qwen38::QWEN38_MTP_SPECULATOR_KIND,
        4,
        8.0,
        &head_mods(&cfg, 4, |_| 0.0),
    );
    let deltas = mtp_speculator_deltas_from_adapter_dir(&dir.0)
        .expect("load")
        .expect("tagged artifact");
    assert_eq!(
        deltas.scaling, 2.0,
        "alpha=8 r=4 must load as scaling 2, proving the zero-delta identity below also covers \
         a non-unit scaling"
    );
    assert_eq!(got.merge_speculator(&deltas).expect("merge"), 8);
    assert_eq!(
        got.fc.w[0], MINUS_ZERO_BF16_BITS_REAL_CHECKPOINTS_CARRY_THEM_BUT_THE_LCG_FIXTURE_NEVER_DOES,
        "a -0.0 drafter weight lost its sign bit under a zero delta: adding literal 0.0 turns \
         -0.0 into +0.0, which is exactly what the merge's skip-zero guard exists to prevent"
    );
    assert_head_bits_equal(&want, &got, "zero-B speculator");
}

#[test]
fn a_nonzero_delta_at_scaling_two_merges_to_w_plus_two_b_dot_a_rounded_once_to_bf16() {
    let cfg = tiny_config();
    let want = mtp_tiny_weights(&cfg, 0x2e40_0005);
    let mut got = mtp_tiny_weights(&cfg, 0x2e40_0005);
    let rank = 4usize;
    let mods = head_mods(&cfg, rank, |i| if i % 2 == 0 { 0.25 } else { -0.125 });
    let synth = synth_adapter(&cfg, 0x5bec_0008, Fill::Coarse, None);
    let dir = write_adapter("specscale", &synth);
    write_speculator(
        &dir.0,
        nv_train::qwen38::QWEN38_MTP_SPECULATOR_KIND,
        rank,
        8.0,
        &mods,
    );
    let deltas = mtp_speculator_deltas_from_adapter_dir(&dir.0)
        .expect("load")
        .expect("tagged artifact");
    assert_eq!(deltas.scaling, 2.0, "alpha=8 r=4 must merge at scaling 2");
    assert_eq!(got.merge_speculator(&deltas).expect("merge"), 8);
    let want_bits = head_bits(&want);
    let got_bits = head_bits(&got);
    let mut moved = 0usize;
    for (i, (module, out, k)) in head_shapes_in_head_bits_order(&cfg).iter().enumerate() {
        let (a, b) = &mods[*module];
        for row in 0..*out {
            for col in 0..*k {
                let ba: f32 = (0..rank).map(|r| b[row * rank + r] * a[r * k + col]).sum();
                let w0 = half::bf16::from_bits(want_bits[i].1[row * k + col]).to_f32();
                let expect = half::bf16::from_f32(w0 + 2.0 * ba).to_bits();
                let got_w = got_bits[i].1[row * k + col];
                assert_eq!(
                    got_w, expect,
                    "{module} [{row},{col}]: merged bits differ from bf16(w + 2*(B.A)); the \
                     power-of-two fills make every product and partial sum exact in f32, so any \
                     mismatch is a dropped or mis-folded scaling, not accumulation order"
                );
                moved += (got_w != want_bits[i].1[row * k + col]) as usize;
            }
        }
    }
    assert!(
        moved > 0,
        "a nonzero-B scaling-2 speculator moved zero weights, so the exactness check above \
         degenerated to the identity and pins nothing"
    );
}

#[test]
fn mismatched_shapes_or_modules_refuse_whole_with_no_partial_merge() {
    let cfg = tiny_config();
    let kind = nv_train::qwen38::QWEN38_MTP_SPECULATOR_KIND;

    let bad_fc = {
        let mut mods = head_mods(&cfg, 4, |i| if i % 2 == 0 { 0.25 } else { -0.25 });
        let hidden = cfg.hidden_size;
        mods.insert(
            "mtp.fc".to_string(),
            (vec![0.125f32; 4 * hidden], vec![0.25f32; hidden * 4]),
        );
        mods
    };
    let synth = synth_adapter(&cfg, 0x5bec_0004, Fill::Coarse, None);
    let dir = write_adapter("specbadfc", &synth);
    write_speculator(&dir.0, kind, 4, 4.0, &bad_fc);
    let deltas = mtp_speculator_deltas_from_adapter_dir(&dir.0)
        .expect("shape mismatches surface at merge, where the drafter's real dims live")
        .expect("tagged artifact");
    let want = mtp_tiny_weights(&cfg, 0x2e40_0002);
    let mut got = mtp_tiny_weights(&cfg, 0x2e40_0002);
    let err = got
        .merge_speculator(&deltas)
        .expect_err("mtp.fc trained against half the fc in-features is a mismatched drafter");
    assert!(
        format!("{err:#}").contains("mtp.fc"),
        "the refusal must name the module; got: {err:#}"
    );
    assert_head_bits_equal(&want, &got, "refused fc merge");

    let bad_q = {
        let mut mods = head_mods(&cfg, 4, |i| if i % 2 == 0 { 0.25 } else { -0.25 });
        let ungated_rows = cfg.num_attention_heads * cfg.head_dim;
        mods.insert(
            "mtp.layers.0.self_attn.q_proj".to_string(),
            (
                vec![0.125f32; 4 * cfg.hidden_size],
                vec![0.25f32; ungated_rows * 4],
            ),
        );
        mods
    };
    let synth = synth_adapter(&cfg, 0x5bec_0005, Fill::Coarse, None);
    let dir = write_adapter("specbadq", &synth);
    write_speculator(&dir.0, kind, 4, 4.0, &bad_q);
    let deltas = mtp_speculator_deltas_from_adapter_dir(&dir.0)
        .expect("load")
        .expect("tagged artifact");
    let mut got = mtp_tiny_weights(&cfg, 0x2e40_0003);
    let err = got
        .merge_speculator(&deltas)
        .expect_err("a q_proj delta covering only the value lanes of the gated projection");
    assert!(
        format!("{err:#}").contains("q_proj"),
        "the refusal must name q_proj; got: {err:#}"
    );

    let mut with_norm = head_mods(&cfg, 4, |_| 0.0);
    with_norm.insert(
        "mtp.norm".to_string(),
        (vec![0.1f32; 4 * cfg.hidden_size], vec![0.1f32; cfg.hidden_size * 4]),
    );
    let synth = synth_adapter(&cfg, 0x5bec_0006, Fill::Coarse, None);
    let dir = write_adapter("specnorm", &synth);
    write_speculator(&dir.0, kind, 4, 4.0, &with_norm);
    let err = mtp_speculator_deltas_from_adapter_dir(&dir.0)
        .expect_err("a delta naming a norm must refuse, never be skipped");
    assert!(
        format!("{err:#}").contains("mtp.norm"),
        "the refusal must name the offending module; got: {err:#}"
    );

    let synth = synth_adapter(&cfg, 0x5bec_0007, Fill::Coarse, None);
    let dir = write_adapter("specrank", &synth);
    write_speculator(&dir.0, kind, 3, 3.0, &head_mods(&cfg, 3, |_| 0.1));
    let scfg_path = dir.0.join(SPECULATOR_SUBDIR).join("adapter_config.json");
    let mut scfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&scfg_path).unwrap()).unwrap();
    scfg["r"] = serde_json::json!(4);
    std::fs::write(&scfg_path, serde_json::to_string_pretty(&scfg).unwrap()).unwrap();
    let err = mtp_speculator_deltas_from_adapter_dir(&dir.0)
        .expect_err("tensors at rank 3 against a config saying r=4");
    assert!(
        format!("{err:#}").contains("r=4"),
        "the refusal must state the configured rank; got: {err:#}"
    );
}

#[test]
fn the_cotrained_artifact_loads_all_8_head_modules_and_moves_the_merged_weights() {
    let (root, _base, out) = cotrained_fixture("cpu-load");
    let deltas = mtp_speculator_deltas_from_adapter_dir(&out)
        .expect("the landed writer's artifact loads clean")
        .expect("co-training tags the main config with the mtp kind");
    assert_eq!(deltas.entries.len(), 8, "all 8 head modules ship");
    for e in &deltas.entries {
        assert!(
            HEAD_TARGETS.contains(&e.module.as_str()),
            "unexpected module {} in the co-trained artifact",
            e.module
        );
        assert_eq!(e.rank, 4);
    }
    assert_eq!(
        deltas.scaling, 1.0,
        "the landed co-train path saves alpha == r, so its artifacts merge at unit scaling"
    );

    let cfg32 = fixture_serving_config(&nv_train::qwen38_fixture::tiny_config_json_with_head_dim(
        32,
    ));
    let want = mtp_tiny_weights(&cfg32, 0x2e40_0004);
    let mut got = mtp_tiny_weights(&cfg32, 0x2e40_0004);
    assert_eq!(got.merge_speculator(&deltas).expect("merge"), 8);
    let moved: usize = head_bits(&want)
        .iter()
        .zip(head_bits(&got).iter())
        .map(|((_, a), (_, b))| a.iter().zip(b).filter(|(x, y)| x != y).count())
        .sum();
    assert!(
        moved > 0,
        "a trained speculator merged into the head moved zero weights, so either every lora_B \
         is zero (untrained artifact) or the merge is a no-op"
    );
    let _ = std::fs::remove_dir_all(&root);
}

fn ids_from(vocab: usize, n: usize, salt: u32) -> Vec<u32> {
    (0..n)
        .map(|i| ((i as u32 * 7 + salt * 13 + 1) % (vocab as u32 - 1)) + 1)
        .collect()
}

fn prefill_like_serving(m: &mut Qwen3_5DenseWgpu, prompt: &[u32]) -> u32 {
    let (last, rest) = prompt.split_last().expect("non-empty prompt");
    let done = m.prefill_tokens(rest).expect("chunked prefill");
    for &t in &rest[done..] {
        m.prefill_step(t).expect("prefill step");
    }
    m.decode_step(*last).expect("anchor decode step")
}

fn padded_arm_env_set() {
    std::env::set_var("NV_WGPU_PREFILL_M", "8");
    std::env::remove_var(q3d::VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM);
    std::env::set_var(
        q3d::PF_ATTN_TILED_ENV_DEFAULT_ON_SET_0_FOR_THE_SCORES_SLAB_ARM,
        "1",
    );
}

fn arm_env_clear() {
    std::env::remove_var("NV_WGPU_PREFILL_M");
    std::env::remove_var(q3d::PF_ATTN_TILED_ENV_DEFAULT_ON_SET_0_FOR_THE_SCORES_SLAB_ARM);
}

enum DraftPolicy {
    MtpOwn,
    Junk,
    Clairvoyant,
}

fn fixture_engine(
    cfg: &Qwen3_5DenseConfig,
    loader: &nv_weights::WeightLoader,
    lora: Option<&Q3dLora>,
    adapter_dir: Option<&Path>,
) -> (Qwen3_5DenseWgpu, usize) {
    let mut m = Qwen3_5DenseWgpu::from_loader_with_lora(cfg.clone(), loader, TINY_MAX_SEQ, lora)
        .expect("build fixture engine");
    let merged = m
        .mtp_attach_with_adapter(loader, adapter_dir)
        .expect("attach the fixture mtp head");
    (m, merged)
}

fn mtp_stream(
    m: &mut Qwen3_5DenseWgpu,
    vocab: usize,
    prompt: &[u32],
    n: usize,
    k: usize,
    policy: &DraftPolicy,
    reference: &[u32],
) -> (Vec<u32>, usize) {
    let mut last = prefill_like_serving(m, prompt);
    let mut out = vec![last];
    let mut accepted_total = 0usize;
    while out.len() < n {
        let rows = m.verify_max_rows();
        assert!(rows >= 2, "NV_WGPU_PREFILL_M did not reach the M-row prefill graph");
        if m.current_pos() + rows.max(m.prefill_chunk_len()) > TINY_MAX_SEQ {
            last = m.decode_step(last).expect("tail decode step");
            out.push(last);
            continue;
        }
        let want = k.min(rows - 1);
        let own = m.mtp_draft_round(last, want).expect("mtp draft round");
        let drafts: Vec<u32> = match policy {
            DraftPolicy::MtpOwn => own,
            DraftPolicy::Junk => (0..want as u32).map(|j| (j * 5 + 2) % vocab as u32).collect(),
            DraftPolicy::Clairvoyant => (0..want)
                .map(|j| *reference.get(out.len() + j).unwrap_or(&own[j.min(own.len() - 1)]))
                .collect(),
        };
        let mut batch = vec![last];
        batch.extend_from_slice(&drafts);
        let amax = m.verify_chain(&batch).expect("verify chain");
        let mut acc = 0usize;
        while acc < drafts.len() && amax[acc] == drafts[acc] {
            acc += 1;
        }
        let bonus = amax[acc];
        m.advance(acc + 1).expect("advance the accepted prefix");
        m.mtp_post_verify(&batch[1..=acc]).expect("mtp post verify");
        accepted_total += acc;
        out.extend_from_slice(&drafts[..acc]);
        out.push(bonus);
        last = bonus;
    }
    out.truncate(n);
    (out, accepted_total)
}

const MERGED_DRAFT_POLICY_INVARIANCE_IS_THE_BAR: &str =
    "every emitted token is gated by the trunk verify argmax, so a merged speculator may only \
     change WHAT is drafted, never the served stream: any draft policy must emit the adapter's \
     pure M=1 stream byte-identically";

#[test]
fn merged_speculator_moves_draft_logits_while_the_served_stream_stays_the_adapter_m1_stream() {
    let _env = env_lock();
    assert!(
        common::have_gpu(),
        "no wgpu adapter: a skipped merge-serving proof reads as a passed one, so this suite \
         fails instead of skipping"
    );
    padded_arm_env_set();
    let (root, base, out) = cotrained_fixture("gpu-e2e");
    let plain_out = root.join("plain");
    let data = root.join("train.jsonl");
    run_jsonl(&jsonl_args(&base, &data, plain_out.clone()), &Device::Cpu)
        .expect("the plain trainer on the same fixture");

    let cfg =
        fixture_serving_config(&std::fs::read_to_string(base.join("config.json")).unwrap());
    let loader = nv_weights::WeightLoader::open_dir(&base, &Device::Cpu)
        .expect("open the fixture checkpoint");
    let lora = Q3dLora::from_peft_dir(&out, &cfg).expect("build lora sites from the artifact");
    let prompt = ids_from(cfg.vocab_size, 9, 3);
    let n = 16;

    let reference = {
        let mut m =
            Qwen3_5DenseWgpu::from_loader_with_lora(cfg.clone(), &loader, TINY_MAX_SEQ, Some(&lora))
                .expect("build the m1 reference engine");
        let mut last = prefill_like_serving(&mut m, &prompt);
        let mut outv = vec![last];
        while outv.len() < n {
            last = m.decode_step(last).expect("reference decode step");
            outv.push(last);
        }
        outv
    };

    let probe = |adapter_dir: Option<&Path>| -> (Vec<f32>, usize) {
        let (mut m, merged) = fixture_engine(&cfg, &loader, Some(&lora), adapter_dir);
        let last = prefill_like_serving(&mut m, &prompt);
        m.mtp_draft_round(last, 1).expect("probe draft round");
        (
            m.debug_mtp_last_draft_logits_for_test().expect("draft logits"),
            merged,
        )
    };
    let (unmerged_logits, none_merged) = probe(None);
    let (plain_dir_logits, plain_merged) = probe(Some(&plain_out));
    assert_eq!(none_merged, 0);
    assert_eq!(
        plain_merged, 0,
        "the plain artifact carries no speculator, so attach-with-adapter must merge nothing"
    );
    let plain_bit_diff = unmerged_logits
        .iter()
        .zip(&plain_dir_logits)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        plain_bit_diff, 0,
        "an artifact without speculator/ must leave the drafter byte-identical to an attach \
         that never saw an adapter dir"
    );

    let (merged_logits, merged) = probe(Some(&out));
    assert_eq!(merged, 8, "the co-trained artifact merges all 8 head modules");
    assert!(
        merged_logits
            .iter()
            .zip(&unmerged_logits)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "the co-trained speculator deltas merged into the head left the draft logits \
         bit-identical, so the merge never reached the resident drafter"
    );

    for policy in [DraftPolicy::MtpOwn, DraftPolicy::Junk, DraftPolicy::Clairvoyant] {
        let (mut m, merged) = fixture_engine(&cfg, &loader, Some(&lora), Some(&out));
        assert_eq!(merged, 8);
        let (stream, accepted) =
            mtp_stream(&mut m, cfg.vocab_size, &prompt, n, 3, &policy, &reference);
        assert_eq!(stream, reference, "{MERGED_DRAFT_POLICY_INVARIANCE_IS_THE_BAR}");
        if matches!(policy, DraftPolicy::Clairvoyant) {
            assert!(
                accepted > 0,
                "clairvoyant drafts from the reference stream never accepted, so the merged \
                 drafter's accept/commit paths never ran and the invariance above is vacuous"
            );
        }
    }
    arm_env_clear();
    let _ = std::fs::remove_dir_all(&root);
}
