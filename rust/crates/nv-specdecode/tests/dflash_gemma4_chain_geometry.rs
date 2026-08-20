#![cfg(feature = "cuda")]

use nv_specdecode::chain::{
    aux_row_extract, build_chain_batch, chain_positions, lower_tri_mask, ChainState,
};
use nv_specdecode::dflash::DFlashSpeculatorConfig;

const REDHAT_GEMMA4_31B_DFLASH_CONFIG_SHAPE: &str = r#"{
  "aux_hidden_state_layer_ids": [1, 17, 29, 47, 58],
  "block_size": 8,
  "draft_vocab_size": 32000,
  "mask_token_id": 4,
  "speculators_config": {
    "algorithm": "dflash",
    "proposal_methods": [
      {"proposal_type": "greedy", "speculative_tokens": 8, "verifier_accept_k": 1}
    ],
    "verifier": {"name_or_path": "google/gemma-4-31B-it"}
  },
  "target_hidden_size": null,
  "tie_word_embeddings": false,
  "transformer_layer_config": {
    "head_dim": 256,
    "hidden_size": 5376,
    "intermediate_size": 21504,
    "max_position_embeddings": 262144,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 5,
    "num_key_value_heads": 16,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "vocab_size": 262144
  }
}"#;

const GEMMA4_31B_TARGET_HIDDEN: usize = 5376;
const GEMMA4_31B_FULL_ATTN_N_Q: usize = 32;
const GEMMA4_31B_FULL_ATTN_N_KV: usize = 4;
const GEMMA4_31B_FULL_ATTN_HD: usize = 512;
const GQA512_LAUNCHER_REFUSES_M_ABOVE: usize = 8;
const DFLASH_K_DEFAULT_MIRRORS_SPEC_ENV: usize = 8;

fn dflash_default_k(block_size: usize) -> usize {
    DFLASH_K_DEFAULT_MIRRORS_SPEC_ENV.clamp(1, block_size.saturating_add(1))
}

fn drafter_config() -> DFlashSpeculatorConfig {
    DFlashSpeculatorConfig::from_hf_json_str(REDHAT_GEMMA4_31B_DFLASH_CONFIG_SHAPE)
        .expect("the checked-in shape of the RedHatAI gemma-4-31B dflash config must parse")
}

#[test]
fn drafter_geometry_matches_the_shipped_checkpoint() {
    let cfg = drafter_config();
    assert_eq!(cfg.hidden_size, 5376);
    assert_eq!(
        cfg.target_hidden_size, GEMMA4_31B_TARGET_HIDDEN,
        "target_hidden_size is null in the shipped json and must fall back to hidden_size"
    );
    assert_eq!(cfg.num_hidden_layers, 5);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 16);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.block_size, 8);
    assert_eq!(cfg.mask_token_id, 4);
    assert_eq!(cfg.aux_hidden_state_layer_ids, vec![1, 17, 29, 47, 58]);
    assert_eq!(
        cfg.q_out_dim(),
        8192,
        "drafter attention width is 32 heads x hd 256, wider than its 5376 hidden"
    );
    assert_eq!(cfg.kv_out_dim(), 4096);
    assert_eq!(cfg.fc_in_dim(), 5 * GEMMA4_31B_TARGET_HIDDEN);
    assert_eq!(cfg.query_rows(), cfg.block_size + 1);
}

#[test]
fn chain_verify_buffers_for_this_drafter_have_the_asserted_shapes() {
    let cfg = drafter_config();
    let k = dflash_default_k(cfg.block_size);
    assert_eq!(k, 8);

    let hidden = cfg.target_hidden_size;
    let n_aux = cfg.aux_hidden_state_layer_ids.len();
    let fc_in = cfg.fc_in_dim();

    let bonus: u32 = 7;
    let draft: Vec<u32> = (0..cfg.block_size as u32).map(|t| 100 + t).collect();
    let batch = build_chain_batch(bonus, &draft, k, true).expect("shift chain batch");
    assert_eq!(batch.len(), k);
    assert_eq!(batch[0], bonus);

    let committed = 37usize;
    let positions = chain_positions(committed, k);
    assert_eq!(positions.len(), k);
    assert_eq!(positions[0], committed as i32);

    let mask = lower_tri_mask(k);
    assert_eq!(mask.len(), k * k);
    assert!(
        nv_models::gemma4::verify_mask_is_chain(&mask, k),
        "the dflash loop verifies a linear chain, never a tree"
    );

    let gaux = vec![0f32; n_aux * k * hidden];
    for slot in 0..k {
        let row = aux_row_extract(&gaux, n_aux, k, hidden, slot).expect("aux row in range");
        assert_eq!(
            row.len(),
            fc_in,
            "every committed slot must yield exactly one fc-input row for the drafter"
        );
    }
    assert!(aux_row_extract(&gaux, n_aux, k, hidden, k).is_err());

    let prompt: Vec<u32> = (0..committed as u32).collect();
    let mut st = ChainState::new(&prompt, fc_in).expect("chain state");
    st.assert_round_start(k, committed + k + 1)
        .expect("round fits the verify cache");
    let aux_row = vec![0.5f32; fc_in];
    for &tok in batch.iter().take(3) {
        st.commit_token(tok, &aux_row).expect("lockstep commit");
    }
    assert_eq!(st.aux_rows(), st.committed());
}

#[test]
fn verify_side_kernel_gates_cover_the_dflash_k() {
    let cfg = drafter_config();
    let k = dflash_default_k(cfg.block_size);

    assert!(
        nv_models::gemma4::gqa512_verify_geometry(
            GEMMA4_31B_FULL_ATTN_N_Q,
            GEMMA4_31B_FULL_ATTN_N_KV,
            GEMMA4_31B_FULL_ATTN_HD
        ),
        "gemma-4-31B full-attention layers are 32q/4kv/hd512, so the gqa512 verify kernel engages"
    );
    assert!(
        k <= GQA512_LAUNCHER_REFUSES_M_ABOVE,
        "dflash k={k} must stay within the gqa512 launcher M cap or full-attention verify \
         falls off the fast path"
    );
    assert!(
        nv_models::gemma4::lm_head_i8_rows_per_call_gate(None, GQA512_LAUNCHER_REFUSES_M_ABOVE)
            >= k,
        "the int8 lm_head must take all {k} dflash verify rows in ONE mk call: the legacy 4-row \
         chunking concatenated via candle Tensor::cat, which a forked-stream verify capture \
         cannot record, so every replay read freed memory as CUDA_ERROR_ILLEGAL_ADDRESS (#107)"
    );
    assert_eq!(
        nv_models::gemma4::lm_head_i8_rows_per_call_gate(Some("1"), 8),
        nv_models::gemma4::LM_HEAD_I8_LEGACY_CHUNK_ROWS_PREDATING_THE_MK_M16_LAUNCHER,
        "NV_VERIFY_LMHEAD_I8_CHUNK4=1 keeps the old eager-only chunking reachable for A/B"
    );
}
