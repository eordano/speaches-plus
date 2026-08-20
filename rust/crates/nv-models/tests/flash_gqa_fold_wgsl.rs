#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::compose;
use nv_kernels::wgpu_backend::kernels as wk;
use nv_models::gemma4_e4b_wgpu as e4b;

const E4B_HEAD_DIM: u32 = 256;

const E4B_GROUP: usize = 4;

fn to_msl(tag: &str, source: &str) -> String {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{tag}: wgsl parse: {}", e.message()));
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{tag}: validate: {e}"));
    let opts = naga::back::msl::Options {
        lang_version: (3, 0),
        ..Default::default()
    };
    naga::back::msl::write_string(
        &module,
        &info,
        &opts,
        &naga::back::msl::PipelineOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{tag}: msl-out: {e}"))
    .0
}

fn count(hay: &str, needle: &str) -> usize {
    hay.matches(needle).count()
}

#[test]
fn folded_decode_stage1_compiles_at_every_supported_fold() {
    for sg in [false, true] {
        for fold in [1u32, 2, 4, 8] {
            let src = format!(
                "{}\n{}",
                compose(wk::flash_decode::WGSL),
                wk::flash_decode::fold_stage1_source(E4B_HEAD_DIM, sg, fold)
            );
            let tag = format!("fold hd={E4B_HEAD_DIM} sg={sg} f={fold}");
            let msl = to_msl(&tag, &src);
            let entry = wk::flash_decode::fold_stage1_entry(E4B_HEAD_DIM, sg, fold);
            assert!(
                msl.contains(&entry),
                "{tag}: emitted MSL is missing entry point {entry}"
            );
        }
    }
}

#[test]
fn folded_decode_stage1_sd_compiles_for_both_reduce_strategies_and_shipping_folds() {
    for hd in [128u32, 256] {
        for sg in [false, true] {
            for fold in [1u32, 2, 3, 4, 6, 8] {
                let src = format!(
                    "{}\n{}",
                    compose(wk::flash_decode::WGSL),
                    wk::flash_decode::fold_stage1_source_sd(hd, sg, fold)
                );
                let tag = format!("fold-sd hd={hd} sg={sg} f={fold}");
                let msl = to_msl(&tag, &src);
                let entry = wk::flash_decode::fold_stage1_entry_sd(hd, sg, fold);
                assert!(
                    msl.contains(&entry),
                    "{tag}: emitted MSL is missing entry point {entry}; models select sg from \
                     adapter caps at runtime, so the wb fallback is every non-32-wide-subgroup \
                     card's only route to the folded decode"
                );
            }
        }
    }
}

#[test]
fn folded_decode_stage1_reads_each_kv_row_once_per_workgroup() {
    let accs = (E4B_HEAD_DIM / 32) as usize;
    for fold in [1u32, 2, 4, 8] {
        let src = wk::flash_decode::fold_stage1_source(E4B_HEAD_DIM, false, fold);
        assert_eq!(
            count(&src, "fd_k_fp8("),
            5,
            "fold {fold}: the vec4 path must load 4 K bytes per lane step plus the scalar \
             fallback's single load, independent of how many query heads share the row"
        );
        assert_eq!(
            count(&src, "fd_v_fp8("),
            accs,
            "fold {fold}: V must be loaded once per accumulator slot and reused by every folded \
             query head, so the count may not scale with the fold"
        );
        assert_eq!(
            count(&src, "fd_k_scales["),
            1,
            "fold {fold}: one K scale fetch per position"
        );
        assert_eq!(
            count(&src, "fd_v_scales["),
            1,
            "fold {fold}: one V scale fetch per position"
        );
    }
}

#[test]
fn folded_decode_stage1_keeps_one_chain_per_query_head() {
    let accs = E4B_HEAD_DIM / 32;
    for fold in [1u32, 2, 4, 8] {
        let src = wk::flash_decode::fold_stage1_source(E4B_HEAD_DIM, false, fold);
        for j in 0..fold {
            assert!(
                src.contains(&format!("var m{j} = fd_neg_inf();")),
                "fold {fold}: chain {j} has no running max"
            );
            assert!(
                src.contains(&format!("var a{j}_{} = 0.0;", accs - 1)),
                "fold {fold}: chain {j} is missing its widest accumulator"
            );
            assert!(
                src.contains(&format!("_epilogue(lid, lane, warp, hd, ((h0 + {j}u)")),
                "fold {fold}: chain {j} never reaches a stage1 slot"
            );
        }
        assert_eq!(
            count(&src, "_epilogue(lid, lane, warp, hd, (("),
            fold as usize,
            "fold {fold}: exactly one epilogue call per folded head"
        );
    }
}

#[test]
fn fold_factor_must_divide_the_gqa_group() {
    assert_eq!(
        e4b::gqa_group_of(&sample_config()),
        E4B_GROUP,
        "the E4B decode fold is bounded by 8 query heads over 2 KV heads"
    );
    for bad in [3usize, 5, 6, 7] {
        assert!(
            E4B_GROUP % bad != 0,
            "{bad} was picked because it does not divide {E4B_GROUP}"
        );
    }
}

#[test]
fn prefill_qtile_stage1_compiles_at_every_supported_tile() {
    for tile in [1u32, 2, 4, 8] {
        let src = format!(
            "{}\n{}",
            compose(wk::flash_decode::WGSL),
            e4b::flash1_mk_qtile_source(E4B_HEAD_DIM, tile)
        );
        let tag = format!("qtile hd={E4B_HEAD_DIM} t={tile}");
        let msl = to_msl(&tag, &src);
        let entry = e4b::flash1_mk_qtile_entry(E4B_HEAD_DIM, tile);
        assert!(
            msl.contains(&entry),
            "{tag}: emitted MSL is missing entry point {entry}"
        );
    }
}

#[test]
fn prefill_qtile_walks_the_kv_slab_once_per_tile_not_once_per_row() {
    let accs = (E4B_HEAD_DIM / 32) as usize;
    for tile in [1u32, 2, 4, 8] {
        let src = e4b::flash1_mk_qtile_source(E4B_HEAD_DIM, tile);
        assert_eq!(
            count(&src, "fd_k_fp8("),
            5,
            "tile {tile}: the K row is loaded once per lane step and shared by every query row in \
             the tile; a count that scales with the tile means the interchange was undone"
        );
        assert_eq!(
            count(&src, "fd_v_fp8("),
            accs,
            "tile {tile}: V is loaded once per accumulator slot and shared by the tile"
        );
        assert_eq!(
            count(&src, "for (var r = 0u; r < rounds; r = r + 1u)"),
            1,
            "tile {tile}: the position loop must be the inner-most walk over the slab, entered \
             once per tile rather than once per query row"
        );
        assert!(
            src.contains(&format!("for (var q0 = 0u; q0 < mr; q0 = q0 + {tile}u)")),
            "tile {tile}: query rows must advance a tile at a time"
        );
    }
}

#[test]
fn prefill_qtile_position_origin_is_row_independent() {
    let src = e4b::flash1_mk_qtile_source(E4B_HEAD_DIM, 4);
    assert!(
        src.contains("let base = split * FD_WARPS;"),
        "the tiled kernel shares one split/warp position origin across the tile, which is only \
         equal to g4w_win_start(total_qi) + split * FD_WARPS when the window is unbounded; the \
         dispatch must therefore restrict it to full-attention layers"
    );
    assert!(
        !src.contains("g4w_win_start"),
        "a sliding window would give every query row its own origin and hence its own \
         position-to-warp assignment, which is exactly the bit-exactness the interchange trades \
         away; the tiled kernel must not pretend to handle it"
    );
}

const E4B_SHAPED_CONFIG: &str = r#"{
  "text_config": {
    "hidden_size": 2560,
    "intermediate_size": 8192,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "global_head_dim": 256,
    "vocab_size": 512,
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-6,
    "sliding_window": 512,
    "layer_types": ["sliding_attention", "full_attention",
                    "sliding_attention", "full_attention"],
    "attention_k_eq_v": false,
    "hidden_activation": "gelu_pytorch_tanh",
    "rope_parameters": {
      "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
      "sliding_attention": {"rope_theta": 10000.0}
    }
  },
  "tie_word_embeddings": true
}"#;

fn sample_config() -> nv_models::gemma4::Gemma4Config {
    nv_models::gemma4::Gemma4Config::from_hf_json_str(E4B_SHAPED_CONFIG)
        .expect("sample e4b-shaped config")
}

#[test]
fn every_new_flash_knob_is_inert_until_its_env_var_is_set() {

    assert_eq!(
        wk::flash_decode::gqa_fold_env(E4B_GROUP),
        wk::flash_decode::DEFAULT_GQA_FOLD,
        "the shipped fold must be the measured one"
    );
    assert!(
        wk::flash_decode::DEFAULT_GQA_FOLD <= 2,
        "fold 4 and 8 were measured slower at every context; the default must \
         not grow past the factor that was actually measured to win"
    );
    assert_eq!(
        wk::flash_decode::splits_env(),
        wk::flash_decode::DEFAULT_SPLITS,
        "raising splits repartitions the stage2 reduction, so results move by up to 3.66e-7 \
         relative. That is a correctness change, not a tuning knob, and it may not arrive by \
         default"
    );
    assert_eq!(
        e4b::prefill_qtile(),
        1,
        "tiling the query axis holds tile*head_dim/32 more accumulators live per thread and \
         will spill before it pays on some geometries"
    );
}
