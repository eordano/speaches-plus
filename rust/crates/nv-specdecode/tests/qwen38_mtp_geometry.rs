mod hub_dirs;

use nv_specdecode::qwen38_mtp::{
    assert_mtp_chain_depth_fits_verify_lm_head_rows_ceiling, mtp_chain_depth, mtp_drafter_selected,
    mtp_round_hidden_reanchor_index, mtp_verify_rows_per_round, read_safetensors_header_shapes,
    resolve_mtp_weight_files, validate_mtp_named_shapes, verify_lm_head_f32_fast_path_rows_ceiling,
    verify_lm_head_rows_per_call_ceiling, Qwen38MtpGeometry, MTP_CHAIN_DEPTH_DEFAULT,
    MTP_WEIGHTS_FILE_NAME, QWEN38_27B_MTP_GEOMETRY,
};
use nv_specdecode::{accept_prefix_argmax, build_chain_batch};

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";

#[test]
fn expected_shapes_match_the_verified_official_index_arithmetic() {
    let g = QWEN38_27B_MTP_GEOMETRY;
    let shapes = g.expected_tensor_shapes();
    assert_eq!(shapes.len(), 15);
    assert_eq!(shapes["mtp.fc.weight"], vec![5120, 10240]);
    assert_eq!(shapes["mtp.layers.0.self_attn.q_proj.weight"], vec![12288, 5120]);
    assert_eq!(shapes["mtp.layers.0.self_attn.k_proj.weight"], vec![1024, 5120]);
    assert_eq!(shapes["mtp.layers.0.self_attn.v_proj.weight"], vec![1024, 5120]);
    assert_eq!(shapes["mtp.layers.0.self_attn.o_proj.weight"], vec![5120, 6144]);
    assert_eq!(shapes["mtp.layers.0.self_attn.q_norm.weight"], vec![256]);
    assert_eq!(shapes["mtp.layers.0.self_attn.k_norm.weight"], vec![256]);
    assert_eq!(shapes["mtp.layers.0.mlp.gate_proj.weight"], vec![17408, 5120]);
    assert_eq!(shapes["mtp.layers.0.mlp.up_proj.weight"], vec![17408, 5120]);
    assert_eq!(shapes["mtp.layers.0.mlp.down_proj.weight"], vec![5120, 17408]);
    for n in [
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
    ] {
        assert_eq!(shapes[n], vec![5120], "{n}");
    }
}

#[test]
fn attn_output_gate_doubles_q_rows_and_nothing_else() {
    let mut g = QWEN38_27B_MTP_GEOMETRY;
    assert_eq!(g.q_proj_out(), 2 * 24 * 256);
    g.attn_output_gate = false;
    assert_eq!(g.q_proj_out(), 24 * 256);
    assert_eq!(g.kv_proj_out(), 4 * 256);
    assert_eq!(g.o_proj_in(), 24 * 256);
    assert_eq!(g.fc_in(), 10240);
}

#[test]
fn validate_rejects_missing_extra_and_misshapen_tensors() {
    let g = QWEN38_27B_MTP_GEOMETRY;
    let good = g.expected_tensor_shapes();
    validate_mtp_named_shapes(&g, &good).unwrap();

    let mut missing = good.clone();
    missing.remove("mtp.fc.weight");
    let e = validate_mtp_named_shapes(&g, &missing).unwrap_err().to_string();
    assert!(e.contains("mtp.fc.weight"), "{e}");

    let mut extra = good.clone();
    extra.insert("mtp.layers.1.mlp.gate_proj.weight".into(), vec![17408, 5120]);
    let e = validate_mtp_named_shapes(&g, &extra).unwrap_err().to_string();
    assert!(e.contains("mtp.layers.1.mlp.gate_proj.weight"), "{e}");

    let mut misshapen = good.clone();
    misshapen.insert("mtp.fc.weight".into(), vec![5120, 5120]);
    let e = validate_mtp_named_shapes(&g, &misshapen).unwrap_err().to_string();
    assert!(e.contains("expected shape [5120, 10240]"), "{e}");
}

#[test]
fn the_downloaded_nvfp4_mtp_shard_matches_the_expected_geometry_when_present() {
    let snap = hub_dirs::snapshot(NVFP4_REPO, &[MTP_WEIGHTS_FILE_NAME]);
    let Some(dir) = snap else {
        if std::env::var("NV_Q38_REQUIRE_MTP_SHARD").as_deref() == Ok("1") {
            panic!(
                "NV_Q38_REQUIRE_MTP_SHARD=1 but no {NVFP4_REPO} snapshot carries \
                 {MTP_WEIGHTS_FILE_NAME}; the downloader may still be fetching the main shard"
            );
        }
        eprintln!(
            "SKIP: {NVFP4_REPO} snapshot has no {MTP_WEIGHTS_FILE_NAME} yet; set \
             NV_Q38_REQUIRE_MTP_SHARD=1 to make this a failure once the download finishes"
        );
        return;
    };
    let named = read_safetensors_header_shapes(&dir.join(MTP_WEIGHTS_FILE_NAME)).unwrap();
    validate_mtp_named_shapes(&QWEN38_27B_MTP_GEOMETRY, &named).unwrap();
}

const SM_86_ADA_OPTIN_BYTES: usize = 101_376;
const SM_80_AMPERE_OPTIN_BYTES: usize = 166_912;
const SM_90_120_OPTIN_BYTES: usize = 232_448;

#[test]
fn rows_ceiling_recomputed_for_hidden_5120_and_the_107_reference_point_5376() {
    for (hidden, limit, want_f32, want_overall) in [
        (5120usize, 0usize, 8usize, 8usize),
        (5120, SM_86_ADA_OPTIN_BYTES, 4, 8),
        (5120, SM_80_AMPERE_OPTIN_BYTES, 7, 14),
        (5120, SM_90_120_OPTIN_BYTES, 8, 16),
        (5376, 0, 8, 8),
        (5376, SM_86_ADA_OPTIN_BYTES, 4, 8),
        (5376, SM_90_120_OPTIN_BYTES, 8, 16),
    ] {
        assert_eq!(
            verify_lm_head_f32_fast_path_rows_ceiling(hidden, limit).unwrap(),
            want_f32,
            "f32 fast path: hidden={hidden} limit={limit}"
        );
        assert_eq!(
            verify_lm_head_rows_per_call_ceiling(hidden, limit).unwrap(),
            want_overall,
            "overall: hidden={hidden} limit={limit}"
        );
    }
    assert!(verify_lm_head_rows_per_call_ceiling(5121, 0).is_err());
    assert!(verify_lm_head_rows_per_call_ceiling(0, 0).is_err());
}

#[test]
fn default_and_max_chain_depths_fit_the_recomputed_ceiling() {
    let k = MTP_CHAIN_DEPTH_DEFAULT;
    assert_eq!(mtp_verify_rows_per_round(k), 4);
    for limit in [0, SM_86_ADA_OPTIN_BYTES, SM_80_AMPERE_OPTIN_BYTES, SM_90_120_OPTIN_BYTES] {
        assert_mtp_chain_depth_fits_verify_lm_head_rows_ceiling(5120, k, limit).unwrap();
    }
    for k in 1..=7usize {
        assert_mtp_chain_depth_fits_verify_lm_head_rows_ceiling(5120, k, 0).unwrap();
    }
    let e = assert_mtp_chain_depth_fits_verify_lm_head_rows_ceiling(5120, 8, 0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("#107"), "{e}");
    assert!(assert_mtp_chain_depth_fits_verify_lm_head_rows_ceiling(5120, 0, 0).is_err());
}

#[test]
fn env_contract_nv_drafter_mtp_and_k_clamps() {
    assert!(mtp_drafter_selected(Some("mtp")));
    assert!(mtp_drafter_selected(Some(" mtp ")));
    assert!(!mtp_drafter_selected(Some("eagle3")));
    assert!(!mtp_drafter_selected(Some("dflash")));
    assert!(!mtp_drafter_selected(Some("auto")));
    assert!(!mtp_drafter_selected(Some("")));
    assert!(!mtp_drafter_selected(None));

    assert_eq!(mtp_chain_depth(None), MTP_CHAIN_DEPTH_DEFAULT);
    assert_eq!(mtp_chain_depth(Some("garbage")), MTP_CHAIN_DEPTH_DEFAULT);
    assert_eq!(mtp_chain_depth(Some("0")), MTP_CHAIN_DEPTH_DEFAULT);
    assert_eq!(mtp_chain_depth(Some("1")), 1);
    assert_eq!(mtp_chain_depth(Some("5")), 5);
    assert_eq!(mtp_chain_depth(Some("7")), 7);
    assert_eq!(mtp_chain_depth(Some("8")), 7);
    assert_eq!(mtp_chain_depth(Some("99")), 7);
    for k in 1..=7usize {
        assert_mtp_chain_depth_fits_verify_lm_head_rows_ceiling(
            5120,
            mtp_chain_depth(Some(&k.to_string())),
            0,
        )
        .unwrap();
    }
}

fn oracle_tok(pos: usize) -> u32 {
    ((pos * 13 + 5) % 97) as u32
}

#[test]
fn mock_mtp_rounds_emit_exactly_the_greedy_stream_and_reanchor_on_the_accepted_row() {
    let k = MTP_CHAIN_DEPTH_DEFAULT;
    let prompt_len = 9usize;
    let mut committed = prompt_len;
    let mut anchor = oracle_tok(prompt_len);
    let mut emitted_stream: Vec<u32> = vec![anchor];
    let mut reanchor_rows: Vec<usize> = Vec::new();

    let correct_per_round = [k, 0, 1, 2, k, 0];
    for &n_correct in &correct_per_round {
        let l = committed;
        let drafts: Vec<u32> = (0..k)
            .map(|j| {
                if j < n_correct {
                    oracle_tok(l + 1 + j)
                } else {
                    oracle_tok(l + 1 + j) ^ 1
                }
            })
            .collect();
        let batch = build_chain_batch(anchor, &drafts, k + 1, true).unwrap();
        assert_eq!(batch.len(), mtp_verify_rows_per_round(k));
        assert_eq!(batch[0], anchor);
        assert_eq!(&batch[1..], &drafts[..]);

        let greedy: Vec<u32> = (0..k + 1).map(|i| oracle_tok(l + 1 + i)).collect();
        let acc = accept_prefix_argmax(&batch, &greedy).unwrap();
        assert_eq!(acc.draft_accepted, n_correct.min(k));
        assert_eq!(acc.commit_len, n_correct.min(k) + 1);
        assert_eq!(acc.next_bonus, oracle_tok(l + 1 + acc.draft_accepted));

        let reanchor = mtp_round_hidden_reanchor_index(acc.draft_accepted);
        assert!(reanchor < batch.len());
        assert_eq!(batch[reanchor], oracle_tok(l + reanchor));
        reanchor_rows.push(reanchor);

        for &t in &drafts[..acc.draft_accepted] {
            emitted_stream.push(t);
        }
        emitted_stream.push(acc.next_bonus);
        committed = l + acc.commit_len;
        anchor = acc.next_bonus;
    }

    let expected: Vec<u32> = (prompt_len..committed + 1).map(oracle_tok).collect();
    assert_eq!(
        emitted_stream, expected,
        "the mtp block loop must be lossless against the greedy oracle stream"
    );
    assert_eq!(reanchor_rows, vec![3, 0, 1, 2, 3, 0]);
}

#[test]
fn weight_file_resolution_prefers_the_dedicated_shard_then_the_index_then_a_single_file() {
    let root = std::env::temp_dir().join(format!(
        "q38_mtp_resolve_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mk = |name: &str| {
        let d = root.join(name);
        std::fs::create_dir_all(&d).unwrap();
        d
    };
    let dedicated = mk("dedicated");
    std::fs::write(dedicated.join(MTP_WEIGHTS_FILE_NAME), b"x").unwrap();

    let indexed = mk("indexed");
    std::fs::write(
        indexed.join("model.safetensors.index.json"),
        serde_json::json!({"weight_map": {
            "mtp.fc.weight": "model-00018-of-00018.safetensors",
            "mtp.norm.weight": "model-00018-of-00018.safetensors",
            "lm_head.weight": "model-00017-of-00018.safetensors"
        }})
        .to_string(),
    )
    .unwrap();

    let empty = mk("empty");

    let got = resolve_mtp_weight_files(None, &dedicated).unwrap();
    assert_eq!(got, vec![dedicated.join(MTP_WEIGHTS_FILE_NAME)]);

    let got = resolve_mtp_weight_files(Some(&dedicated), &empty).unwrap();
    assert_eq!(got, vec![dedicated.join(MTP_WEIGHTS_FILE_NAME)]);

    let got = resolve_mtp_weight_files(None, &indexed).unwrap();
    assert_eq!(got, vec![indexed.join("model-00018-of-00018.safetensors")]);

    let e = resolve_mtp_weight_files(None, &empty).unwrap_err().to_string();
    assert!(e.contains("no MTP weights found"), "{e}");
    assert!(e.contains("NV_DRAFTER=mtp"), "{e}");

    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn geometry_matches_the_downloaded_config_json_when_present() {
    let Some(dir) = hub_dirs::snapshot(NVFP4_REPO, &["config.json"]) else {
        eprintln!("SKIP: no {NVFP4_REPO} snapshot with config.json");
        return;
    };
    let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let t = &v["text_config"];
    let g = Qwen38MtpGeometry {
        hidden: t["hidden_size"].as_u64().unwrap() as usize,
        heads: t["num_attention_heads"].as_u64().unwrap() as usize,
        kv_heads: t["num_key_value_heads"].as_u64().unwrap() as usize,
        head_dim: t["head_dim"].as_u64().unwrap() as usize,
        intermediate: t["intermediate_size"].as_u64().unwrap() as usize,
        attn_output_gate: t["attn_output_gate"].as_bool().unwrap(),
    };
    assert_eq!(g, QWEN38_27B_MTP_GEOMETRY);
    assert_eq!(t["mtp_num_hidden_layers"].as_u64(), Some(1));
    assert_eq!(t["mtp_use_dedicated_embeddings"].as_bool(), Some(false));
}
