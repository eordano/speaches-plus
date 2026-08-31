use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

const SHARED_HELPER_MODULES_PATH_INCLUDED_BY_SUITES_AND_DECLARING_NO_TESTS_THEMSELVES: &[&str] = &[];

const CULLED_SUITES_NO_DELETION_FROM_THIS_DIRECTORY_HAS_EVER_BEEN_RECORDED: &[&str] = &[];

const HEADER_CUDA_AND_WGPU: &str = "#![cfg(all(feature = \"cuda\", feature = \"wgpu\"))]";
const HEADER_CUDA_ONLY: &str = "#![cfg(feature = \"cuda\")]";
const HEADER_WGPU_ONLY: &str = "#![cfg(feature = \"wgpu\")]";
const HEADER_LAGUNA_WIP: &str = "#![cfg(feature = \"laguna-wip\")]";
const HEADER_CUDA_OR_WGPU_SPELLED_WGPU_FIRST: &str =
    "#![cfg(any(feature = \"wgpu\", feature = \"cuda\"))]";
const HEADER_CUDA_OR_WGPU_SPELLED_CUDA_FIRST: &str =
    "#![cfg(any(feature = \"cuda\", feature = \"wgpu\"))]";

const TOTAL_SUITES_ON_DISK_INCLUDING_THIS_CENSUS: usize = 283;
const THE_ROUTINE_NVK_DEFAULT_FOR_NV_MODELS_IS_CUDA_ALONE_WHICH_BLANKS_THIS_MANY_SUITES: usize =
    157;
const SUITES_UNABLE_TO_ASSERT_ON_A_ROUTINE_CUDA_ONLY_RUN_BLANKED_OR_EVERY_TEST_IGNORED: usize =
    231;
const SUITES_UNABLE_TO_ASSERT_EVEN_UNDER_CUDA_AND_WGPU_LAGUNA_BLANKED_OR_EVERY_TEST_IGNORED:
    usize = 120;

const CUDA_ONLY_CLASS_COLLAPSES_BELOW: usize = 80;
const WGPU_ONLY_CLASS_COLLAPSES_BELOW: usize = 130;
const LAGUNA_WIP_CLASS_COLLAPSES_BELOW: usize = 8;

const A_WALK_THAT_READS_FEWER_FILES_THAN_THIS_IS_BROKEN_NOT_CLEAN: usize = 283;

const CENSUS_CUDA_ONLY: &[&str] = &[
    "chunked_causal_attention.rs",
    "decode_probe_gpu_graph.rs",
    "deepseek_ocr_bench.rs",
    "deepseek_ocr_graph_batch.rs",
    "deepseek_ocr_sam_flash.rs",
    "depth_correctness_sweep.rs",
    "dots_ocr_bench.rs",
    "dots_ocr_single_page.rs",
    "e4b_greedy_agree.rs",
    "e4b_prefill_session_phases.rs",
    "failed_ctor_rounds_hold_no_vram.rs",
    "gemma4_attnproj_gemv_floor.rs",
    "gemma4_batch_graph.rs",
    "gemma4_capture_stream_policy.rs",
    "gemma4_ctx_scaling.rs",
    "gemma4_graph_dtod_arm_keys.rs",
    "gemma4_graph_logits_into.rs",
    "gemma4_kv_share.rs",
    "gemma4_moe_expert_btgemm_oracle.rs",
    "gemma4_moe_prefill_destall.rs",
    "gemma4_prefill_w4a4.rs",
    "gemma4_serving_ctx.rs",
    "gemma4_sliding_kv.rs",
    "gemma4_smallop_fusion.rs",
    "gemma4_spec_prefill_hd512_flash.rs",
    "gemma4_verify_forward.rs",
    "gemma4_verify_lmhead_i8.rs",
    "gemma4_vision_graph_capture.rs",
    "gemma4_vision_nvfp4.rs",
    "gemma4_w4a4_flip_gate.rs",
    "got_ocr_bench.rs",
    "got_ocr_real.rs",
    "gpt_oss_cuda_tiny.rs",
    "laguna_batch_decode.rs",
    "laguna_ctx_scaling.rs",
    "laguna_dflash_verify.rs",
    "laguna_fp8_unit.rs",
    "laguna_graph_gate.rs",
    "laguna_moe_fold_parity.rs",
    "laguna_ring_kv.rs",
    "laguna_spec_serving_loop.rs",
    "paged_batched_decode_correctness.rs",
    "paged_block_table_growth.rs",
    "paged_derive_v_decode_ab.rs",
    "paged_derive_v_step_time.rs",
    "paged_derive_v_vram.rs",
    "paged_mk_prefill_unit.rs",
    "paged_ring_decode_ab.rs",
    "ppl_e4b_cuda.rs",
    "ppl_gemma4.rs",
    "ppl_gemma4_moe_cuda.rs",
    "ppl_laguna.rs",
    "ppl_qwen3.rs",
    "ppl_qwen35_dense.rs",
    "ppl_qwen3_moe.rs",
    "q38qual_cross_engine.rs",
    "qwen35_9b_cuda_decode_timing.rs",
    "qwen35_ctx_scaling.rs",
    "qwen36_batch_decode.rs",
    "qwen36_chat_validate.rs",
    "qwen36_ctx_scaling.rs",
    "qwen36_cuda_decode_timing.rs",
    "qwen36_freerun_base_rate.rs",
    "qwen36_moe_m1_ab.rs",
    "qwen36_moe_verify_chain_parity.rs",
    "qwen36_moe_verify_msweep.rs",
    "qwen36_moecap_stall_bisect.rs",
    "qwen36_mtp_econ.rs",
    "qwen36_mtp_evidence.rs",
    "qwen36_mtp_serving_session.rs",
    "qwen36_serving_ctx.rs",
    "qwen36_verify_trunk_arms_parity.rs",
    "qwen38_batch_decode.rs",
    "qwen38_batch_step_profile.rs",
    "qwen38_decode_split.rs",
    "qwen38_prefill_depth.rs",
    "qwen38_prefill_graph.rs",
    "qwen38_prefill_pp512.rs",
    "qwen38_prefill_seam.rs",
    "qwen38_prime_ckpt.rs",
    "qwen38_quant_arm_probe.rs",
    "qwen38_serving_ctx.rs",
    "qwen38_vision_e2e_smoke.rs",
    "qwen3_5_grouped_gemm_oracle.rs",
    "qwen3_5_grouped_moe_stream_2x2.rs",
    "qwen3_ctx_scaling.rs",
    "qwen_graph_reset_bisect.rs",
    "train_step_cost.rs",
    "vram_engine_cycle_csan.rs",
    "vram_soak.rs",
    "xppl_qwen38.rs",
];

const CENSUS_CUDA_AND_WGPU: &[&str] = &[
    "e4b_ppl_cuda_wgpu_position_diff.rs",
    "qwen38_image_splice_batched_vs_walk.rs",
];

const CENSUS_CUDA_OR_WGPU_EITHER_BACKEND_ALONE_COMPILES_THESE: &[&str] = &[
    "ctx196k_conformance.rs",
    "e4b_cuda_reference_divergence.rs",
];

const CENSUS_WGPU_ONLY: &[&str] = &[
    "deepseek_ocr_decoder_wgpu.rs",
    "dots_ocr_vision_wgpu.rs",
    "e4b_ctx_synthfill.rs",
    "e4b_deep_depthgate.rs",
    "e4b_dispatch_census.rs",
    "e4b_dispatch_floor.rs",
    "e4b_flash1_graph_ab.rs",
    "e4b_flash_stage1_rate.rs",
    "e4b_host_sync.rs",
    "e4b_norm_chain.rs",
    "e4b_pass_budget.rs",
    "e4b_shallow_split_null.rs",
    "e4b_verify_mk_width.rs",
    "e4b_w4_scale_grain.rs",
    "e4b_wgpu_prefill_session_phases.rs",
    "flash_gqa_fold_bench.rs",
    "flash_gqa_fold_parity.rs",
    "flash_gqa_fold_wgsl.rs",
    "flash_splits_runtime.rs",
    "g4moe_decode_dispatch_census.rs",
    "g4moe_depth_stack_parity.rs",
    "g4moe_grouped_wall_ab.rs",
    "gemma4_31b_w4_ptq_quantizer.rs",
    "gemma4_31b_w4a16_requant.rs",
    "gemma4_batch_decode.rs",
    "gemma4_dense_chunked_prefill.rs",
    "gemma4_dense_flash_fold_parity.rs",
    "gemma4_dense_host_shape.rs",
    "gemma4_e4b_wgpu.rs",
    "gemma4_e4b_wgpu_na.rs",
    "gemma4_e4b_wgpu_na_attn.rs",
    "gemma4_e4b_wgpu_verify.rs",
    "gemma4_lmhead_i8_mk.rs",
    "gemma4_moe_ct_nvfp4_loader.rs",
    "gemma4_moe_dispatch_budget.rs",
    "gemma4_moe_gemv_wide.rs",
    "gemma4_moe_gemv_wide_ab.rs",
    "gemma4_moe_grouped_gemv.rs",
    "gemma4_moe_i8_lmhead_route.rs",
    "gemma4_moe_int8_dense_shapes.rs",
    "gemma4_moe_nozi_price.rs",
    "gemma4_moe_router_topk_ab.rs",
    "gemma4_moe_w4_gemv_ab.rs",
    "gemma4_moe_wgpu.rs",
    "gemma4_moe_wgpu_depthgate.rs",
    "gemma4_moe_wgpu_flash_decode.rs",
    "gemma4_moe_wgpu_fold_chain.rs",
    "gemma4_moe_wgpu_kv_ring.rs",
    "gemma4_moe_wgpu_prefill.rs",
    "gemma4_moe_wgpu_verify.rs",
    "gemma4_nvfp4_chunked_prefill.rs",
    "gemma4_pf_flash_tolerance.rs",
    "gemma4_pf_wide_m.rs",
    "gemma4_wgpu_fusion.rs",
    "gemma4_wgpu_kv_ring.rs",
    "gemma4_wgpu_quant.rs",
    "gemma4_wgpu_shared_gate.rs",
    "gemma4_wgpu_verify.rs",
    "gow_dispatch_census.rs",
    "gow_dncomb_wall_ab.rs",
    "gow_shallow_lmhead_cost_probe.rs",
    "gpt_oss_wgpu.rs",
    "gpt_oss_wgpu_accfile.rs",
    "gpt_oss_wgpu_bit_arms.rs",
    "gpt_oss_wgpu_chunked_prefill.rs",
    "gpt_oss_wgpu_ctx_synthfill.rs",
    "gpt_oss_wgpu_default_arms.rs",
    "gpt_oss_wgpu_depthgate.rs",
    "gpt_oss_wgpu_kv8.rs",
    "gpt_oss_wgpu_mx_sg.rs",
    "gpt_oss_wgpu_verify.rs",
    "graph_entry_census_oracle.rs",
    "graph_g4w_gather2_oracle.rs",
    "graph_g4w_head_prep_oracle.rs",
    "graph_g4w_int8_epilogue_oracle.rs",
    "graph_g4w_mk_bf16_oracle.rs",
    "graph_g4w_norm_chain_oracle.rs",
    "graph_g4w_quant_row_oracle.rs",
    "graph_gow_router_topk_oracle.rs",
    "graph_q3d_attn_chain_oracle.rs",
    "graph_q3d_attn_oracle.rs",
    "graph_q3d_decode_head_oracle.rs",
    "graph_q3d_delta_decode_oracle.rs",
    "graph_q3d_delta_front_oracle.rs",
    "graph_q3d_deltanet_oracle.rs",
    "graph_q3d_elementwise_oracle.rs",
    "graph_q3d_fused_decode_identity.rs",
    "graph_q3d_gemv_i8_oracle.rs",
    "graph_q3d_prefill_attn_oracle.rs",
    "graph_q3d_prefill_gemm_oracle.rs",
    "lora_e4b_wgpu.rs",
    "lora_q3d_wgpu.rs",
    "lora_q3d_wgpu_verify_mtp.rs",
    "ppl_e4b.rs",
    "ppl_gemma4_moe.rs",
    "ppl_gemma4_wgpu.rs",
    "ppl_gptoss.rs",
    "q3d_delta_mlx_port.rs",
    "q3d_delta_scan_wy.rs",
    "q3d_delta_scan_wy_v2.rs",
    "q3d_device_argmax_vs_host_on_real_weights.rs",
    "q3d_flash_stage1_rate.rs",
    "q3d_kv_fp8_tp.rs",
    "q3d_kv_nvfp4.rs",
    "q3d_mlp_w4a8.rs",
    "q3d_mlx_affine_graph_oracle.rs",
    "q3d_mlx_affine_real_smoke.rs",
    "q3d_multichunk_prefill_state_vs_stepped.rs",
    "q3d_pf_attn_mk.rs",
    "q3d_resume_checkpoint.rs",
    "q3d_wgpu_mtp_speculator_merge.rs",
    "qwen36_moe_vision_splice.rs",
    "qwen36_proj_int8_battery.rs",
    "qwen38_rotkv_probe.rs",
    "qwen3_5_dense_nvfp4_chunked_prefill.rs",
    "qwen3_5_dense_w8_chunked_prefill.rs",
    "qwen3_5_dense_wgpu.rs",
    "qwen3_5_dense_wgpu_fused_decode.rs",
    "qwen3_5_dense_wgpu_mtp.rs",
    "qwen3_5_dense_wgpu_w8.rs",
    "qwen3_5_moe_attn_fusion.rs",
    "qwen3_5_moe_delta_split.rs",
    "qwen3_5_moe_fusion.rs",
    "qwen3_5_moe_gemv_bf16_load.rs",
    "qwen3_5_moe_pass_budget.rs",
    "qwen3_5_moe_quant_rate.rs",
    "qwen3_5_moe_wgpu.rs",
    "qwen3_5_moe_wgpu_chunked_prefill.rs",
    "qwen3_5_moe_wgpu_mrow_prefill.rs",
    "qwen3_5_wgpu_verify_chain.rs",
    "qwen3_8_dense_wgpu_real.rs",
    "qwen3_8_vision_tower_real.rs",
    "qwen3_mrope_rows_wgpu.rs",
    "wgpu_correct_gemma4_pk_variants.rs",
    "wgpu_ctx_scaling.rs",
    "wgpu_default_flip_quality.rs",
    "wgpu_fp8_epilogue.rs",
    "wgpu_fp8_formats.rs",
    "wgpu_fp8_granularity.rs",
    "wgpu_g4moe_decode_profile_probe.rs",
    "wgpu_gemma4_nvfp4_v2_routing.rs",
    "wgpu_nozi_graph_census.rs",
    "wgpu_nozi_graph_poison_parity.rs",
    "wgpu_q36_decode_profile_probe.rs",
    "wgpu_qwen35_nvfp4_v2_routing.rs",
];

const CENSUS_LAGUNA_WIP_BLANKED_EVEN_UNDER_CUDA_WGPU_BECAUSE_LAGUNA_WIP_IS_ITS_OWN_OPT_IN_FEATURE: &[&str] = &[
    "laguna_wgpu_attn.rs",
    "laguna_wgpu_chunked_prefill.rs",
    "laguna_wgpu_config.rs",
    "laguna_wgpu_ctx_scaling.rs",
    "laguna_wgpu_dense.rs",
    "laguna_wgpu_model.rs",
    "laguna_wgpu_moe.rs",
    "laguna_wgpu_real.rs",
    "laguna_wgpu_verify.rs",
    "laguna_wgpu_w8.rs",
];

const CENSUS_UNCONDITIONAL_THESE_COMPILE_AND_CAN_ASSERT_WITH_NO_FEATURE_AT_ALL: &[&str] = &[
    "deepseek_ocr_oracle.rs",
    "f8_probe.rs",
    "feature_census.rs",
    "fp8_contract_freerun.rs",
    "fp8_contract_prompts.rs",
    "gemma4_audio_gate.rs",
    "gemma4_embeds_override_parity.rs",
    "gemma4_mm_splice.rs",
    "gemma4_vision_gate.rs",
    "gguf_gemma26b_gpu.rs",
    "gguf_sidecar_synthesis.rs",
    "gguf_tiny_roundtrip.rs",
    "graph_teardown_is_universal.rs",
    "hadamard_qk_census.rs",
    "laguna_attn_gate_excl.rs",
    "laguna_prompts.rs",
    "lora_delta_cost.rs",
    "prefill_chunk_floor_census.rs",
    "prefix_reuse_bound.rs",
    "prompt_fidelity.rs",
    "qwen38_mm_splice.rs",
    "qwen38_mtp_head_real.rs",
    "qwen38_vision_tensor_map.rs",
    "qwen3_5_moe_flip_gate.rs",
    "qwen3_8_config_contract.rs",
    "qwen3_8_geometry_smoke.rs",
    "qwen3_core.rs",
    "train_attention_oracle.rs",
    "train_batched_forward.rs",
    "train_eagle3_cotrain.rs",
    "train_moe_adapter_artifact.rs",
    "train_moe_attention_oracle.rs",
    "wgpu_shader_drift_guard.rs",
];

const SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED: &[(&str, &str)] = &[
    ("ctx196k_conformance.rs", "HOME NV_CTX196K_CONFORMANCE NV_MODELS_ALLOW_SKIP"),
    ("decode_probe_gpu_graph.rs", "NV_GPU_GRAPH_TEST"),
    ("deepseek_ocr_bench.rs", "NV_DSOCR_BENCH"),
    ("deepseek_ocr_oracle.rs", "HOME NV_DSOCR2_CPU NV_DSOCR2_DTYPE"),
    ("deepseek_ocr_sam_flash.rs", "HOME NV_DSOCR_DIR NV_SAM_FLASH_TEST"),
    ("dots_ocr_bench.rs", "HOME NV_DOTS_BENCH NV_DOTS_DIR"),
    ("dots_ocr_single_page.rs", "HOME NV_DOTS_DIR"),
    ("e4b_ctx_synthfill.rs", "NV_E4B_CTX_TEST NV_E4B_DEEPGATE_AB"),
    ("e4b_cuda_reference_divergence.rs", "HOME NV_DETERMINISTIC NV_E4B_DIR NV_E4B_XREF_FREEZE NV_E4B_XREF_STREAM NV_E4B_XREF_TEST"),
    ("e4b_dispatch_census.rs", "NV_E4B_CENSUS NV_E4B_CENSUS_ARM_FILTER"),
    ("e4b_dispatch_floor.rs", "NV_E4B_DISPATCH_FLOOR"),
    ("e4b_flash1_graph_ab.rs", "NV_E4B_FLASH1_AB NV_E4B_WGPU_FLASH1_HD NV_E4B_WGPU_FLASH1_SG"),
    ("e4b_flash_stage1_rate.rs", "NV_E4B_DEEP_FOLD NV_E4B_DEEP_PROBE NV_E4B_DEPTH_ATTR NV_E4B_FLASH_RATE"),
    ("e4b_greedy_agree.rs", "NV_E4B_AGREE_TEST"),
    ("e4b_host_sync.rs", "HOME NV_E4B_DIR NV_E4B_HOST_SYNC NV_E4B_HS_BLOCK NV_E4B_HS_REPS NV_E4B_HS_ROUNDS NV_E4B_HS_TOKENS NV_E4B_PF_TOKENS NV_E4B_WGPU_FUSE_ATTN NV_E4B_WGPU_FUSE_HEAD"),
    ("e4b_pass_budget.rs", "NV_E4B_BUDGET"),
    ("e4b_ppl_cuda_wgpu_position_diff.rs", ""),
    ("e4b_prefill_session_phases.rs", "NV_CTX_TOKENS"),
    ("e4b_w4_scale_grain.rs", "HOME NV_E4B_GRAIN_NEW NV_E4B_GRAIN_REPS NV_E4B_W4_DIR"),
    ("e4b_wgpu_prefill_session_phases.rs", "NV_CTX_TOKENS NV_E4B_SERVING_TEST NV_WGPU_WARMUP"),
    ("flash_gqa_fold_bench.rs", "NV_WGPU_FOLD_BENCH"),
    ("g4moe_decode_dispatch_census.rs", "NV_G4MOE_CENSUS_LAYERS NV_G4MOE_CENSUS_MAXSEQ NV_G4MOE_DISPATCH_CENSUS NV_WGPU_US_PER_DISPATCH"),
    ("g4moe_grouped_wall_ab.rs", "NV_G4MOE_AB_BASE NV_G4MOE_AB_LADDER NV_G4MOE_AB_RECORD_ARMS NV_G4MOE_WALL_AB NV_G4MOE_WALL_LADDER NV_G4MOE_WALL_MARGINAL_BASE NV_G4MOE_WALL_RUNGS"),
    ("gemma4_attnproj_gemv_floor.rs", "HOME NV_G4_GEMV_DIAG NV_G4_GEMV_FLOOR"),
    ("gemma4_batch_graph.rs", "NV_BATCH_GRAPH_TEST NV_MODELS_ALLOW_SKIP"),
    ("gemma4_ctx_scaling.rs", "NV_GEMMA4_CTX_TEST"),
    ("gemma4_e4b_wgpu_na.rs", "NV_E4B_WGPU_TEST NV_NA_REPS NV_PREFILL_M_LADDER"),
    ("gemma4_e4b_wgpu_na_attn.rs", "NV_E4B_WGPU_TEST NV_NA_ATTN_M NV_NA_REPS"),
    ("gemma4_graph_logits_into.rs", "NV_GEMMA4_GRAPH_LOGITS_TEST"),
    ("gemma4_moe_dispatch_budget.rs", "NV_G4MOE_BUDGET_LAYERS NV_GGUF_PATH"),
    ("gemma4_moe_gemv_wide_ab.rs", "HOME NV_G4MOE_GEMV_WIDE NV_G4MOE_WIDE_CLASSES NV_G4MOE_WIDE_LAYERS NV_G4MOE_WIDE_PROMPT NV_G4MOE_WIDE_TOKENIZER NV_GGUF_PATH"),
    ("gemma4_moe_nozi_price.rs", "NV_WGPU_NOZI"),
    ("gemma4_moe_prefill_destall.rs", "HOME NV_CTX_TOKENS NV_G4MOE_PREFILL_TEST NV_G4MOE_SNAPSHOT NV_MODELS_ALLOW_SKIP"),
    ("gemma4_moe_router_topk_ab.rs", "NV_G4MOE_ROUTER_TOPK NV_G4MOE_TOPK_LAYERS NV_GGUF_PATH"),
    ("gemma4_moe_w4_gemv_ab.rs", "NV_G4MOE_W4_GEMV NV_G4MOE_W4_LAYERS NV_GGUF_PATH"),
    ("gemma4_serving_ctx.rs", "NV_G4_SERVING_ARM NV_GEMMA4_SERVING_TEST NV_MODELS_ALLOW_SKIP"),
    ("gemma4_spec_prefill_hd512_flash.rs", "HOME NV_CTX_TOKENS NV_G4_SNAPSHOT NV_PPL_CORPUS NV_SPEC_PREFILL_FLASH NV_SPEC_PREFILL_HD512_BF16 NV_SPEC_PREFILL_HD512_EAGER"),
    ("gemma4_verify_forward.rs", "HOME NV_G4_ROUTE_CROSSOVER NV_G4_ROUTE_IDENTITY NV_G4_SNAPSHOT NV_NVFP4_GEMV_ROWS NV_NVFP4_M1_GEMV"),
    ("gemma4_verify_lmhead_i8.rs", "HOME NV_G4_LMI8_TEST NV_G4_SNAPSHOT NV_VERIFY_LMHEAD_INT8"),
    ("gemma4_w4a4_flip_gate.rs", "NV_CHAT_EVAL_STEPS NV_PREFILL_W4A4 NV_W4A4_ARM"),
    ("got_ocr_bench.rs", "NV_GOT_OCR_BENCH"),
    ("gow_dispatch_census.rs", ""),
    ("gow_dncomb_wall_ab.rs", "NV_GOW_DNCOMB_AB NV_GOW_DNCOMB_AB_ARM_B NV_GOW_DNCOMB_AB_BUILD_B_FIRST"),
    ("gow_shallow_lmhead_cost_probe.rs", "NV_GOW_LMHEAD_COST_PROBE"),
    ("gpt_oss_wgpu_ctx_synthfill.rs", "NV_GPTOSS_AB_ENV"),
    ("laguna_ctx_scaling.rs", "HOME NV_CTX_TOKENS NV_LAGUNA_CTX_TEST NV_LAGUNA_DIR"),
    ("laguna_dflash_verify.rs", "NV_LAGUNA_DFLASH NV_LAGUNA_TEST"),
    ("laguna_fp8_unit.rs", "HOME NV_LAGUNA_DIR NV_LAGUNA_TEST"),
    ("laguna_moe_fold_parity.rs", "NV_LAGUNA_CHAIN_GRAPH NV_LAGUNA_CHAIN_STEPS NV_LAGUNA_CTX_TEST"),
    ("laguna_ring_kv.rs", "NV_LAGUNA_TEST"),
    ("laguna_spec_serving_loop.rs", "NV_DFLASH_DRAFT_DIR NV_LAGUNA_DFLASH NV_LAGUNA_TEST"),
    ("laguna_wgpu_ctx_scaling.rs", "NV_CTX_TOKENS NV_LAGUNA_WGPU_CTX_TEST"),
    ("laguna_wgpu_real.rs", "NV_LAGUNA_MAXNEW NV_LAGUNA_PROMPT NV_LAGUNA_STEPS"),
    ("lora_delta_cost.rs", "NV_LORA_COST"),
    ("paged_block_table_growth.rs", ""),
    ("paged_derive_v_decode_ab.rs", "NV_CHAT_MODEL_DIR NV_KV_DERIVE_V_AB"),
    ("paged_derive_v_step_time.rs", "NV_CHAT_MODEL_DIR NV_KV_DERIVE_V_TIME"),
    ("paged_derive_v_vram.rs", "NV_CHAT_MODEL_DIR NV_KV_DERIVE_V_TEST"),
    ("paged_mk_prefill_unit.rs", "NV_PAGED_MK_UNIT"),
    ("paged_ring_decode_ab.rs", "NV_ALIGN_CHUNKS NV_CHAT_MODEL_DIR NV_PAGED_RING_AB NV_PAGED_RING_AB_CONTROL NV_PAGED_RING_AB_FULLONLY NV_PAGED_RING_AB_TOKENS NV_PREFILL_W4A4"),
    ("ppl_e4b.rs", ""),
    ("ppl_e4b_cuda.rs", "NV_E4B_PPL_DEBUG_CHUNK"),
    ("ppl_gemma4.rs", "NV_GEMMA4_CHAIN_STEPS NV_GREEDY_CHAT NV_GREEDY_FP8_CACHE NV_GREEDY_GRAPH NV_PPL_DEBUG_N NV_PPL_DEBUG_PROBE_START NV_PPL_DEBUG_TAIL_ONLY NV_PPL_GRAPHED"),
    ("ppl_gemma4_moe.rs", "HOME NV_G4MOE_SNAPSHOT"),
    ("ppl_gemma4_moe_cuda.rs", "HOME NV_G4MOE_CUDA_CTX_TEST NV_G4MOE_SNAPSHOT"),
    ("ppl_gemma4_wgpu.rs", "HOME NV_G4_WGPU_W8_FFN NV_GEMMA4_MOE_DIR"),
    ("ppl_gptoss.rs", ""),
    ("ppl_laguna.rs", "NV_MOE_SHARED_FOLD"),
    ("ppl_qwen3.rs", "HF_HUB_CACHE HOME NV_QWEN3_DIR NV_QWEN3_WEIGHTS"),
    ("ppl_qwen35_dense.rs", ""),
    ("ppl_qwen3_moe.rs", "HOME NV_QWEN36_DIR"),
    ("q38qual_cross_engine.rs", "NV_Q38QUAL NV_Q38QUAL_FILE NV_Q38QUAL_GEN_OUT NV_Q38QUAL_JOBS NV_Q38QUAL_NLL_OUT NV_QWEN38_DIR"),
    ("q3d_device_argmax_vs_host_on_real_weights.rs", "NV_ARGMAX_AB_MAX_SEQ NV_ARGMAX_AB_PAD_TO NV_QWEN38_DIR NV_QWEN38_PROMPT NV_QWEN38_WGPU_TEST"),
    ("q3d_mlx_affine_real_smoke.rs", "NV_MLXQ_LAYERS NV_MLXQ_PROMPT NV_Q3D_AFFINE_STEPPED_PF_THE_PRE_V2_BASIS_26TOK_1_10S_1613TOK_38_5MS"),
    ("q3d_multichunk_prefill_state_vs_stepped.rs", "NV_QWEN38_DIR NV_QWEN38_WGPU_TEST NV_STATE_DIFF_TEXT"),
    ("qwen35_9b_cuda_decode_timing.rs", "NV_QWEN35_Q NV_QWEN35_TIMING"),
    ("qwen35_ctx_scaling.rs", "NV_QWEN35_CHAIN_GRAPHED NV_QWEN35_CHAIN_STEPS NV_QWEN35_Q"),
    ("qwen36_batch_decode.rs", "NV_CTX_TOKENS NV_Q36_BATCH_ROWWISE NV_Q36_BATCH_TEST NV_Q36_BRATE NV_Q36_GDN_FP8 NV_Q36_LM_HEAD_FP8"),
    ("qwen36_chat_validate.rs", "NV_QWEN36_ARM NV_QWEN36_CHAT NV_QWEN36_Q"),
    ("qwen36_ctx_scaling.rs", "NV_QWEN36_CTX_TEST"),
    ("qwen36_cuda_decode_timing.rs", "NV_QWEN36_ARM NV_QWEN36_Q NV_QWEN36_TIMING"),
    ("qwen36_freerun_base_rate.rs", "NV_Q95_ARM NV_Q95_COTENANT NV_Q95_NEW_TOKENS NV_Q95_RUNS NV_QWEN36_ARM NV_QWEN36_FREERUN_BASE NV_QWEN36_Q"),
    ("qwen36_moe_m1_ab.rs", "HOME NV_MOE_FP4_DECODE_GEMV NV_QWEN36_M1_AB NV_QWEN36_M1_AB_STEPS"),
    ("qwen36_moe_verify_chain_parity.rs", "HOME NV_Q36_MOE_SHARED_FOLD NV_QWEN36_DIR NV_QWEN36_MCHAIN_PARITY NV_QWEN36_VCHAIN_PARITY"),
    ("qwen36_moe_verify_msweep.rs", "HOME NV_GRAPH_OFF NV_QWEN36_DIR NV_QWEN36_MSWEEP NV_QWEN36_MSWEEP_M1 NV_QWEN36_MSWEEP_ROWS"),
    ("qwen36_moecap_stall_bisect.rs", "HOME NV_MOECAP_ENV_LABEL NV_QWEN36_DIR NV_QWEN36_MOECAP"),
    ("qwen36_mtp_econ.rs", "HOME NV_Q36_ECON_CAPTURED_BASELINE_ONLY_DEPTH NV_Q36_ECON_KS NV_Q36_MTP_P_MIN NV_QWEN36_DIR NV_QWEN36_MTP_ECON NV_QWEN36_MTP_ECON_P_MIN"),
    ("qwen36_mtp_evidence.rs", "NV_QWEN36_MTP NV_QWEN36_Q"),
    ("qwen36_mtp_serving_session.rs", "HOME NV_QWEN36_DIR NV_QWEN36_MTP_SERVE NV_QWEN36_Q"),
    ("qwen36_proj_int8_battery.rs", "NV_FLIP_ARMS NV_FLIP_ASSERT_ACCEPT NV_FLIP_PROMPTS NV_QWEN36_DIR"),
    ("qwen36_serving_ctx.rs", "NV_MODELS_ALLOW_SKIP NV_Q36_STEP_DUMP NV_QWEN36_SERVING_TEST"),
    ("qwen38_batch_step_profile.rs", "NV_GRAPH_OFF NV_Q38_BATCH_GEMM NV_Q38_BATCH_PROF NV_Q38_BPROF_BS NV_Q38_BPROF_PARITY_TEST NV_Q38_BPROF_TEST NV_Q38_PREFILL_LENS NV_Q38_PREFILL_PROBE NV_QWEN38_DIR"),
    ("qwen38_decode_split.rs", "NV_CTX_TOKENS NV_GRAPH_OFF NV_MODELS_ALLOW_SKIP NV_PROF_DECODE NV_QWEN38_SERVING_TEST"),
    ("qwen38_image_splice_batched_vs_walk.rs", "HOME NV_Q3D_PF_COOP NV_QWEN38_DIR"),
    ("qwen38_mtp_head_real.rs", "HOME NV_QWEN38_DIR NV_QWEN38_MTP_TEST"),
    ("qwen38_prefill_depth.rs", "NV_CTX_TOKENS NV_MODELS_ALLOW_SKIP NV_PROF_DECODE NV_Q38_GDN_CHUNK_PREFILL NV_QWEN38_SERVING_TEST"),
    ("qwen38_prefill_graph.rs", "NV_CTX_TOKENS NV_MODELS_ALLOW_SKIP NV_QWEN38_SERVING_TEST"),
    ("qwen38_prefill_seam.rs", "NV_CTX_TOKENS NV_MODELS_ALLOW_SKIP NV_QWEN38_SERVING_TEST"),
    ("qwen38_prime_ckpt.rs", "HOME NV_CTX_TOKENS NV_MODELS_ALLOW_SKIP NV_QWEN38_SERVING_TEST"),
    ("qwen38_quant_arm_probe.rs", "NV_Q38_TEST NV_QWEN38_DIR"),
    ("qwen38_rotkv_probe.rs", "NV_QWEN38_DIR NV_QWEN38_LAYERS"),
    ("qwen38_serving_ctx.rs", "NV_MODELS_ALLOW_SKIP NV_QWEN38_SERVING_TEST"),
    ("qwen38_vision_e2e_smoke.rs", "NV_QWEN38_DIR"),
    ("qwen3_5_grouped_gemm_oracle.rs", "NV_MOE_ORACLE_TEST"),
    ("qwen3_5_grouped_moe_stream_2x2.rs", "NV_MOE_STREAM_2X2 NV_MOE_STREAM_ARM"),
    ("qwen3_ctx_scaling.rs", "HF_HUB_CACHE HOME NV_CTX_TOKENS NV_QWEN3_CTX_TEST NV_QWEN3_DIR NV_QWEN3_WEIGHTS"),
    ("qwen_graph_reset_bisect.rs", "HOME NV_QWEN_GRAPH_BISECT"),
    ("train_step_cost.rs", "NV_TRAIN_COST"),
    ("vram_soak.rs", ""),
    ("wgpu_g4moe_decode_profile_probe.rs", "HOME NV_GEMMA4_MOE_DIR NV_GEMMA4_MOE_WGPU_TEST"),
    ("wgpu_q36_decode_profile_probe.rs", "HOME NV_QWEN36_DIR NV_QWEN36_WGPU_TEST"),
    ("xppl_qwen38.rs", "NV_Q38_XPPL NV_QWEN38_DIR NV_XPPL_CHUNKS NV_XPPL_FILE"),
];

const SUITES_WITH_TESTS_THAT_RETURN_GREEN_WHEN_AN_OPT_IN_ENV_VAR_IS_ABSENT_THIS_LIST_MAY_ONLY_SHRINK: &[(&str, usize)] = &[
    ("deepseek_ocr_bench.rs", 2),
    ("deepseek_ocr_decoder_wgpu.rs", 1),
    ("deepseek_ocr_graph_batch.rs", 1),
    ("deepseek_ocr_sam_flash.rs", 2),
    ("dots_ocr_bench.rs", 1),
    ("dots_ocr_single_page.rs", 1),
    ("e4b_ctx_synthfill.rs", 2),
    ("e4b_cuda_reference_divergence.rs", 2),
    ("e4b_greedy_agree.rs", 1),
    ("e4b_prefill_session_phases.rs", 5),
    ("fp8_contract_freerun.rs", 2),
    ("g4moe_decode_dispatch_census.rs", 1),
    ("g4moe_grouped_wall_ab.rs", 2),
    ("gemma4_attnproj_gemv_floor.rs", 2),
    ("gemma4_batch_graph.rs", 3),
    ("gemma4_ctx_scaling.rs", 2),
    ("gemma4_e4b_wgpu.rs", 7),
    ("gemma4_e4b_wgpu_na.rs", 2),
    ("gemma4_e4b_wgpu_na_attn.rs", 1),
    ("gemma4_e4b_wgpu_verify.rs", 1),
    ("gemma4_graph_logits_into.rs", 2),
    ("gemma4_moe_expert_btgemm_oracle.rs", 1),
    ("gemma4_moe_prefill_destall.rs", 3),
    ("gemma4_moe_wgpu.rs", 2),
    ("gemma4_moe_wgpu_prefill.rs", 1),
    ("gemma4_pf_flash_tolerance.rs", 1),
    ("gemma4_serving_ctx.rs", 3),
    ("gemma4_spec_prefill_hd512_flash.rs", 1),
    ("gemma4_verify_forward.rs", 2),
    ("gemma4_verify_lmhead_i8.rs", 2),
    ("gemma4_wgpu_fusion.rs", 2),
    ("gemma4_wgpu_quant.rs", 4),
    ("gguf_gemma26b_gpu.rs", 2),
    ("got_ocr_bench.rs", 1),
    ("got_ocr_real.rs", 1),
    ("gow_dispatch_census.rs", 1),
    ("gow_dncomb_wall_ab.rs", 1),
    ("gpt_oss_wgpu.rs", 1),
    ("gpt_oss_wgpu_chunked_prefill.rs", 1),
    ("gpt_oss_wgpu_verify.rs", 1),
    ("laguna_batch_decode.rs", 1),
    ("laguna_ctx_scaling.rs", 3),
    ("laguna_dflash_verify.rs", 3),
    ("laguna_fp8_unit.rs", 4),
    ("laguna_moe_fold_parity.rs", 1),
    ("laguna_ring_kv.rs", 1),
    ("laguna_spec_serving_loop.rs", 1),
    ("laguna_wgpu_ctx_scaling.rs", 1),
    ("laguna_wgpu_verify.rs", 1),
    ("ppl_gemma4.rs", 1),
    ("ppl_gptoss.rs", 2),
    ("q3d_device_argmax_vs_host_on_real_weights.rs", 1),
    ("q3d_kv_fp8_tp.rs", 7),
    ("q3d_kv_nvfp4.rs", 3),
    ("q3d_multichunk_prefill_state_vs_stepped.rs", 1),
    ("q3d_pf_attn_mk.rs", 4),
    ("qwen36_ctx_scaling.rs", 1),
    ("qwen36_mtp_econ.rs", 1),
    ("qwen36_serving_ctx.rs", 2),
    ("qwen38_decode_split.rs", 1),
    ("qwen38_mtp_head_real.rs", 1),
    ("qwen38_prefill_depth.rs", 4),
    ("qwen38_prefill_graph.rs", 3),
    ("qwen38_prefill_pp512.rs", 4),
    ("qwen38_prefill_seam.rs", 1),
    ("qwen38_prime_ckpt.rs", 1),
    ("qwen38_serving_ctx.rs", 2),
    ("qwen3_5_dense_w8_chunked_prefill.rs", 2),
    ("qwen3_5_dense_wgpu.rs", 1),
    ("qwen3_5_grouped_gemm_oracle.rs", 5),
    ("qwen3_5_grouped_moe_stream_2x2.rs", 1),
    ("qwen3_5_moe_delta_split.rs", 1),
    ("qwen3_5_moe_wgpu.rs", 2),
    ("qwen3_5_moe_wgpu_mrow_prefill.rs", 3),
    ("qwen3_8_config_contract.rs", 1),
    ("wgpu_ctx_scaling.rs", 17),
    ("wgpu_fp8_granularity.rs", 1),
    ("wgpu_g4moe_decode_profile_probe.rs", 1),
    ("wgpu_q36_decode_profile_probe.rs", 1),
];

const UNCONDITIONAL_FILES_WHOSE_TESTS_HIDE_BEHIND_THEIR_OWN_CFG_SO_THE_HEADER_LIES: &[(&str, usize)] = &[
    ("fp8_contract_freerun.rs", 1),
    ("gemma4_embeds_override_parity.rs", 1),
    ("gguf_tiny_roundtrip.rs", 1),
    ("qwen38_mm_splice.rs", 2),
    ("qwen3_5_moe_flip_gate.rs", 4),
    ("qwen3_8_config_contract.rs", 1),
    ("qwen3_8_geometry_smoke.rs", 7),
    ("train_batched_forward.rs", 1),
];

fn suite_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn literal_content_blanked(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let n = b.len();
    let mut out = b.clone();
    let mut i = 0usize;
    let blank = |out: &mut Vec<char>, k: usize| {
        if out[k] != '\n' {
            out[k] = ' ';
        }
    };
    let raw_hashes = |b: &[char], mut k: usize| -> Option<(usize, usize)> {
        let mut h = 0usize;
        while k < b.len() && b[k] == '#' {
            h += 1;
            k += 1;
        }
        if k < b.len() && b[k] == '"' {
            Some((k + 1, h))
        } else {
            None
        }
    };
    while i < n {
        let c = b[i];
        let prev_is_ident = i > 0 && (b[i - 1].is_alphanumeric() || b[i - 1] == '_');
        if (c == 'r' || (c == 'b' && i + 1 < n && b[i + 1] == 'r')) && !prev_is_ident {
            let start = if c == 'b' { i + 2 } else { i + 1 };
            if let Some((mut k, h)) = raw_hashes(&b, start) {
                loop {
                    if k >= n {
                        i = k;
                        break;
                    }
                    if b[k] == '"' && (1..=h).all(|q| k + q < n && b[k + q] == '#') {
                        i = k + 1 + h;
                        break;
                    }
                    blank(&mut out, k);
                    k += 1;
                }
                continue;
            }
        }
        if c == '"' {
            let mut k = i + 1;
            while k < n {
                if b[k] == '\\' {
                    blank(&mut out, k);
                    if k + 1 < n {
                        blank(&mut out, k + 1);
                    }
                    k += 2;
                    continue;
                }
                if b[k] == '"' {
                    break;
                }
                blank(&mut out, k);
                k += 1;
            }
            i = k + 1;
            continue;
        }
        if c == '\'' {
            if i + 1 < n && b[i + 1] == '\\' {
                let mut k = i + 2;
                while k < n && b[k] != '\'' {
                    blank(&mut out, k);
                    k += 1;
                }
                blank(&mut out, i + 1);
                i = k + 1;
                continue;
            }
            if i + 2 < n && b[i + 2] == '\'' {
                blank(&mut out, i + 1);
                i += 3;
                continue;
            }
            i += 1;
            continue;
        }
        i += 1;
    }
    out.into_iter().collect()
}

fn crate_level_cfg_lines(raw: &str) -> Vec<String> {
    raw.lines()
        .filter(|l| l.starts_with("#![cfg("))
        .map(|l| l.trim_end().to_string())
        .collect()
}

fn unindented_positive_cfg_attributes(raw: &str) -> usize {
    raw.lines()
        .filter(|l| {
            l.starts_with("#[cfg(feature")
                || l.starts_with("#[cfg(all(feature")
                || l.starts_with("#[cfg(any(feature")
        })
        .count()
}

const TEST_ATTRIBUTE_SPELLINGS_CARGO_RUNS_AS_A_TEST_NOT_ONLY_PLAIN_TEST: &[&str] = &[
    "#[test]",
    "#[test(",
    "#[tokio::test",
    "#[rstest",
    "#[test_case",
    "#[proptest",
    "#[bench",
];

fn is_a_test_attribute(t: &str) -> bool {
    TEST_ATTRIBUTE_SPELLINGS_CARGO_RUNS_AS_A_TEST_NOT_ONLY_PLAIN_TEST
        .iter()
        .any(|a| t.starts_with(a))
}

struct SourceFn {
    name: String,
    is_test: bool,
    is_ignored: bool,
    is_zero_arg_bool_fn: bool,
    body: Vec<String>,
    body_never_closed: bool,
}

fn source_fns(src: &str) -> Vec<SourceFn> {
    let blanked = literal_content_blanked(src);
    let lines: Vec<&str> = blanked.lines().collect();
    let mut out: Vec<SourceFn> = Vec::new();
    let mut attr_depth = 0i32;
    let mut in_attr = false;
    let mut saw_test = false;
    let mut saw_ignore = false;
    let mut i = 0usize;
    while i < lines.len() {
        let t = lines[i].trim();
        if in_attr {
            attr_depth += t.matches('[').count() as i32 - t.matches(']').count() as i32;
            in_attr = attr_depth > 0;
            i += 1;
            continue;
        }
        if t.starts_with("#[") {
            if is_a_test_attribute(t) {
                saw_test = true;
            }
            if t.starts_with("#[ignore") {
                saw_ignore = true;
            }
            attr_depth = t.matches('[').count() as i32 - t.matches(']').count() as i32;
            in_attr = attr_depth > 0;
            i += 1;
            continue;
        }
        if t.is_empty() {
            i += 1;
            continue;
        }
        let starts_a_fn = t.starts_with("fn ")
            || t.starts_with("pub fn ")
            || t.starts_with("async fn ")
            || t.starts_with("pub async fn ");
        if !starts_a_fn {
            saw_test = false;
            saw_ignore = false;
            i += 1;
            continue;
        }
        let name = t
            .split("fn ")
            .nth(1)
            .unwrap_or("")
            .split('(')
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        let mut body: Vec<String> = Vec::new();
        let mut signature = String::new();
        let mut d = 0i32;
        let mut started = false;
        let mut j = i;
        while j < lines.len() {
            let l = lines[j];
            if !started {
                signature.push_str(l);
            }
            body.push(l.to_string());
            let opens = l.matches('{').count();
            let closes = l.matches('}').count();
            d += opens as i32 - closes as i32;
            if opens > 0 {
                started = true;
            }
            if started && d <= 0 {
                break;
            }
            j += 1;
        }
        let closed = started && d <= 0;
        out.push(SourceFn {
            name,
            is_test: saw_test,
            is_ignored: saw_test && saw_ignore,
            is_zero_arg_bool_fn: !saw_test
                && signature.contains("()")
                && signature.contains("-> bool"),
            body,
            body_never_closed: !closed,
        });
        saw_test = false;
        saw_ignore = false;
        i = if closed { j + 1 } else { i + 1 };
    }
    out
}

fn declared_and_ignored_test_fns(src: &str) -> (usize, usize) {
    let fns = source_fns(src);
    (
        fns.iter().filter(|f| f.is_test).count(),
        fns.iter().filter(|f| f.is_ignored).count(),
    )
}

fn env_flag_helper_names(fns: &[SourceFn]) -> BTreeSet<String> {
    fns.iter()
        .filter(|f| f.is_zero_arg_bool_fn && f.body.iter().any(|l| l.contains("env::var")))
        .map(|f| f.name.clone())
        .collect()
}

const A_BLOCK_OPENS_WITH: char = '{';
const A_BLOCK_CLOSES_WITH: char = '}';

fn enclosing_guard(body: &[String], k: usize) -> String {
    let mut d = 0i32;
    let mut open = 0usize;
    for j in (0..=k).rev() {
        d += body[j].matches('}').count() as i32 - body[j].matches('{').count() as i32;
        if d < 0 {
            open = j;
            break;
        }
    }
    let mut first = open;
    while first > 0 && open - first < 8 {
        let prev = body[first - 1].trim_end();
        let ends_a_finished_statement = prev.is_empty()
            || prev.ends_with(';')
            || prev.ends_with(A_BLOCK_OPENS_WITH)
            || prev.ends_with(A_BLOCK_CLOSES_WITH);
        if ends_a_finished_statement {
            break;
        }
        first -= 1;
    }
    body[first..=open]
        .iter()
        .map(|l| l.trim())
        .collect::<Vec<_>>()
        .join(" ")
}

fn tests_that_return_green_when_an_opt_in_env_var_is_absent(src: &str) -> Vec<String> {
    let fns = source_fns(src);
    let flags = env_flag_helper_names(&fns);
    let mut out: Vec<String> = Vec::new();
    for f in fns.iter().filter(|f| f.is_test) {
        if f.body_never_closed {
            out.push(format!(
                "{}: this census could not find the end of the function body, so its returns were \
                 never examined; an unbalanced brace the literal blanker failed to erase defeats \
                 the walk and an unexaminable suite must be an error, not a silent pass",
                f.name
            ));
            continue;
        }
        for (k, l) in f.body.iter().enumerate() {
            let t = l.trim();
            if t != "return;" && t != "return Ok(());" {
                continue;
            }
            let guard = enclosing_guard(&f.body, k);
            let gates_on_env = guard.contains("env::var")
                || flags.iter().any(|n| guard.contains(&format!("{n}()")));
            if !gates_on_env {
                continue;
            }
            let fires_only_when_the_variable_is_deliberately_set = (guard.contains("==")
                || guard.contains("is_ok()"))
                && !guard.contains("!=")
                && !guard.contains("is_err()")
                && !guard.contains("is_none()")
                && !flags.iter().any(|n| guard.contains(&format!("!{n}()")));
            if fires_only_when_the_variable_is_deliberately_set {
                continue;
            }
            out.push(format!("{} returns under the guard `{}`", f.name, guard.trim()));
        }
    }
    out
}

fn suite_sources() -> Vec<(String, String)> {
    let dir = suite_dir();
    let mut out: Vec<(String, String)> = Vec::new();
    let mut push = |p: &PathBuf| {
        if p.extension().and_then(|s| s.to_str()) != Some("rs") {
            return;
        }
        let name = p
            .strip_prefix(&dir)
            .unwrap_or(p)
            .to_string_lossy()
            .to_string();
        let src = std::fs::read_to_string(p)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
        out.push((name, src));
    };
    for e in std::fs::read_dir(&dir).expect("suite directory").flatten() {
        let p = e.path();
        if p.is_dir() {
            for e2 in std::fs::read_dir(&p).expect("suite subdirectory").flatten() {
                push(&e2.path());
            }
        } else {
            push(&p);
        }
    }
    out.sort();
    out
}

#[derive(Default)]
struct Corpus {
    cuda_only: Vec<String>,
    cuda_and_wgpu: Vec<String>,
    cuda_or_wgpu: Vec<String>,
    wgpu_only: Vec<String>,
    laguna_wip: Vec<String>,
    unconditional: Vec<String>,
    helpers: Vec<String>,
    cfg_headers_this_census_cannot_classify: Vec<String>,
    files_declaring_no_tests_and_not_declared_helpers: Vec<String>,
    every_test_ignored: Vec<String>,
    unconditional_with_per_item_cfg: Vec<(String, usize)>,
    subdirectory_targets_a_file_walk_cannot_see: Vec<String>,
    subdirectory_helper_modules: Vec<String>,
}

impl Corpus {
    fn total(&self) -> usize {
        self.cuda_only.len()
            + self.cuda_and_wgpu.len()
            + self.cuda_or_wgpu.len()
            + self.wgpu_only.len()
            + self.laguna_wip.len()
            + self.unconditional.len()
    }
    fn blanked_under_the_routine_cuda_only_default(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .wgpu_only
            .iter()
            .chain(self.laguna_wip.iter())
            .chain(self.cuda_and_wgpu.iter())
            .cloned()
            .collect();
        v.sort();
        v
    }
}

fn corpus() -> Corpus {
    let dir = suite_dir();
    let rd = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read the suite directory {}: {e}", dir.display()));
    let mut c = Corpus::default();
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            let dname = p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();
            if p.join("main.rs").exists() {
                c.subdirectory_targets_a_file_walk_cannot_see.push(dname);
            } else if p.join("mod.rs").exists() {
                c.subdirectory_helper_modules.push(dname);
            }
            continue;
        }
        if p.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let name = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let src = std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
        let is_helper =
            SHARED_HELPER_MODULES_PATH_INCLUDED_BY_SUITES_AND_DECLARING_NO_TESTS_THEMSELVES
                .contains(&name.as_str());
        let (declared, ignored) = declared_and_ignored_test_fns(&src);
        if is_helper {
            c.helpers.push(name.clone());
        } else if declared == 0 {
            c.files_declaring_no_tests_and_not_declared_helpers
                .push(name.clone());
        } else if declared == ignored {
            c.every_test_ignored.push(name.clone());
        }
        let cfgs = crate_level_cfg_lines(&src);
        if cfgs.len() > 1 {
            c.cfg_headers_this_census_cannot_classify.push(format!(
                "{name}: {} crate-level #![cfg] lines, which this census cannot reduce to one \
                 class: {cfgs:?}",
                cfgs.len()
            ));
            continue;
        }
        let class = match cfgs.first().map(String::as_str) {
            None => 0u8,
            Some(HEADER_CUDA_AND_WGPU) => 1,
            Some(HEADER_CUDA_ONLY) => 2,
            Some(HEADER_WGPU_ONLY) => 3,
            Some(HEADER_LAGUNA_WIP) => 4,
            Some(HEADER_CUDA_OR_WGPU_SPELLED_WGPU_FIRST)
            | Some(HEADER_CUDA_OR_WGPU_SPELLED_CUDA_FIRST) => 5,
            Some(other) => {
                c.cfg_headers_this_census_cannot_classify
                    .push(format!("{name}: unknown spelling {other}"));
                continue;
            }
        };
        if is_helper {
            continue;
        }
        match class {
            1 => c.cuda_and_wgpu.push(name),
            2 => c.cuda_only.push(name),
            3 => c.wgpu_only.push(name),
            4 => c.laguna_wip.push(name),
            5 => c.cuda_or_wgpu.push(name),
            _ => {
                let n = unindented_positive_cfg_attributes(&src);
                if n > 0 {
                    c.unconditional_with_per_item_cfg.push((name.clone(), n));
                }
                c.unconditional.push(name);
            }
        }
    }
    for v in [
        &mut c.cuda_only,
        &mut c.cuda_and_wgpu,
        &mut c.cuda_or_wgpu,
        &mut c.wgpu_only,
        &mut c.laguna_wip,
        &mut c.unconditional,
        &mut c.helpers,
        &mut c.cfg_headers_this_census_cannot_classify,
        &mut c.files_declaring_no_tests_and_not_declared_helpers,
        &mut c.every_test_ignored,
        &mut c.subdirectory_targets_a_file_walk_cannot_see,
        &mut c.subdirectory_helper_modules,
    ] {
        v.sort();
    }
    c.unconditional_with_per_item_cfg.sort();
    c
}

fn census_delta(
    class: &str,
    recorded_in: &str,
    on_disk: &[String],
    recorded: &[&str],
) -> Option<String> {
    let disk: BTreeSet<&str> = on_disk.iter().map(String::as_str).collect();
    let rec: BTreeSet<&str> = recorded.iter().copied().collect();
    let added: Vec<&str> = disk.difference(&rec).copied().collect();
    let dropped: Vec<&str> = rec.difference(&disk).copied().collect();
    if added.is_empty() && dropped.is_empty() {
        return None;
    }
    let mut m = format!(
        "{class}: {} suites on disk, {} recorded in {recorded_in}",
        disk.len(),
        rec.len()
    );
    for n in added {
        m.push_str(&format!(
            "\n      + {n} is on disk and not in {recorded_in}. The whole fix is one line: add \
             \"{n}\", to {recorded_in}. Name the commit that added it with\n          git log \
             --oneline --diff-filter=A -- rust/crates/nv-models/tests/{n}"
        ));
    }
    for n in dropped {
        m.push_str(&format!(
            "\n      - {n} is in {recorded_in} and not on disk. The whole fix is one line: delete \
             \"{n}\", from {recorded_in}, and record the lost coverage in \
             CULLED_SUITES_NO_DELETION_FROM_THIS_DIRECTORY_HAS_EVER_BEEN_RECORDED. Name the commit \
             that removed it with\n          git log --oneline --diff-filter=D -- \
             rust/crates/nv-models/tests/{n}"
        ));
    }
    Some(m)
}

fn out_of_order(recorded_in: &str, recorded: &[&str]) -> Option<String> {
    for w in recorded.windows(2) {
        if w[0] >= w[1] {
            return Some(format!(
                "{recorded_in} is not in strictly ascending order: {:?} precedes {:?}. Keep it \
                 sorted, or two lanes that each add a suite will both append and the duplicate \
                 will read as a correct count.",
                w[0], w[1]
            ));
        }
    }
    None
}

#[test]
fn features_compiled_into_this_test_run() {
    let cuda = cfg!(feature = "cuda");
    let wgpu = cfg!(feature = "wgpu");
    let laguna = cfg!(feature = "laguna-wip");
    let c = corpus();
    eprintln!("nv-models tests compiled with cuda={cuda} wgpu={wgpu} laguna-wip={laguna}");
    eprintln!(
        "nvk.sh defaults every NVK_PKG other than nv-kernels to `cuda` ALONE, so the natural \
         `NVK_PKG=nv-models nvk.sh test` silently compiles {} of this crate's {} suites to \
         nothing, each printing `0 passed` in 0.00s. Pass NVK_FEATURES=cuda,wgpu for the wgpu \
         surface, and cuda,wgpu,laguna-wip for all of it.",
        c.blanked_under_the_routine_cuda_only_default().len(),
        c.total(),
    );
    assert!(
        cuda || wgpu,
        "neither `cuda` nor `wgpu` is enabled, so {} of this crate's {} suites compiled to empty \
         binaries. A run in this configuration is not evidence of anything.",
        c.total() - c.unconditional.len(),
        c.total(),
    );
}

#[test]
fn the_suite_corpus_is_the_one_this_census_names_file_by_file() {
    let c = corpus();
    let classes: [(&str, &str, &[String], &[&str]); 6] = [
        (
            "cuda-only",
            "CENSUS_CUDA_ONLY",
            c.cuda_only.as_slice(),
            CENSUS_CUDA_ONLY,
        ),
        (
            "cuda+wgpu",
            "CENSUS_CUDA_AND_WGPU",
            c.cuda_and_wgpu.as_slice(),
            CENSUS_CUDA_AND_WGPU,
        ),
        (
            "cuda-or-wgpu",
            "CENSUS_CUDA_OR_WGPU_EITHER_BACKEND_ALONE_COMPILES_THESE",
            c.cuda_or_wgpu.as_slice(),
            CENSUS_CUDA_OR_WGPU_EITHER_BACKEND_ALONE_COMPILES_THESE,
        ),
        (
            "wgpu-only",
            "CENSUS_WGPU_ONLY",
            c.wgpu_only.as_slice(),
            CENSUS_WGPU_ONLY,
        ),
        (
            "laguna-wip",
            "CENSUS_LAGUNA_WIP_BLANKED_EVEN_UNDER_CUDA_WGPU_BECAUSE_LAGUNA_WIP_IS_ITS_OWN_OPT_IN_FEATURE",
            c.laguna_wip.as_slice(),
            CENSUS_LAGUNA_WIP_BLANKED_EVEN_UNDER_CUDA_WGPU_BECAUSE_LAGUNA_WIP_IS_ITS_OWN_OPT_IN_FEATURE,
        ),
        (
            "unconditional",
            "CENSUS_UNCONDITIONAL_THESE_COMPILE_AND_CAN_ASSERT_WITH_NO_FEATURE_AT_ALL",
            c.unconditional.as_slice(),
            CENSUS_UNCONDITIONAL_THESE_COMPILE_AND_CAN_ASSERT_WITH_NO_FEATURE_AT_ALL,
        ),
    ];
    eprintln!(
        "nv-models suite corpus: {} suites on disk = {} cuda-only + {} cuda+wgpu + {} \
         cuda-or-wgpu + {} wgpu-only + {} laguna-wip + {} unconditional, plus {} shared helper \
         module dir(s) {:?} built into their dependents rather than as targets",
        c.total(),
        c.cuda_only.len(),
        c.cuda_and_wgpu.len(),
        c.cuda_or_wgpu.len(),
        c.wgpu_only.len(),
        c.laguna_wip.len(),
        c.unconditional.len(),
        c.subdirectory_helper_modules.len(),
        c.subdirectory_helper_modules,
    );
    let mut wrong: Vec<String> = Vec::new();
    for (class, recorded_in, on_disk, recorded) in classes {
        wrong.extend(out_of_order(recorded_in, recorded));
        wrong.extend(census_delta(class, recorded_in, on_disk, recorded));
    }
    assert!(
        wrong.is_empty(),
        "the suite corpus moved without this census moving with it.\n  {}\n\nEvery line above is \
         its own fix and its own investigation: the name to add or delete, the constant to edit, \
         and the git command that names the commit responsible. Make the edit in the SAME commit \
         as the suite and put those names in the message. This census WALKS the directory rather \
         than trusting a written list, so a suite cannot be silently absent from it: before this \
         file existed, two measurement rounds based regression evidence on green counts from this \
         crate while 157 of its suites compiled to nothing.",
        wrong.join("\n  ")
    );
    assert_eq!(
        c.total(),
        TOTAL_SUITES_ON_DISK_INCLUDING_THIS_CENSUS,
        "the suite total moved. If a suite was added or deleted deliberately, update \
         TOTAL_SUITES_ON_DISK_INCLUDING_THIS_CENSUS in the same commit; the per-class lists above \
         will already have demanded the name."
    );
}

#[test]
fn no_file_in_this_directory_escapes_classification_by_carrying_a_header_spelling_this_census_cannot_read(
) {
    let c = corpus();
    assert!(
        c.cfg_headers_this_census_cannot_classify.is_empty(),
        "these files carry a crate-level #![cfg] this census cannot reduce to one of its six \
         classes: {:?}.\nAn unreadable header is how a suite silently changes class, and a \
         silently rebucketed class is exactly the defect that once made a hand count of this \
         crate report 128 blanked suites when the true figure was 137 -- `laguna-wip` bucketed as \
         unconditional. Teach corpus() the new spelling, or restate the header as one of \
         HEADER_CUDA_ONLY / HEADER_WGPU_ONLY / HEADER_CUDA_AND_WGPU / HEADER_LAGUNA_WIP / \
         HEADER_CUDA_OR_WGPU_SPELLED_*.",
        c.cfg_headers_this_census_cannot_classify,
    );
    assert!(
        c.subdirectory_targets_a_file_walk_cannot_see.is_empty(),
        "these subdirectories of tests/ hold a main.rs: {:?}.\ncargo builds tests/<dir>/main.rs \
         as its own test target, and this census walks FILES in tests/ -- so such a target is \
         exactly the suite-that-never-runs this file exists to make impossible, hiding one \
         directory level below where anyone would look. Either flatten it to tests/<dir>.rs, or \
         teach corpus() to descend and classify it.",
        c.subdirectory_targets_a_file_walk_cannot_see,
    );
    assert!(
        c.cuda_only.len() >= CUDA_ONLY_CLASS_COLLAPSES_BELOW
            && c.wgpu_only.len() >= WGPU_ONLY_CLASS_COLLAPSES_BELOW
            && c.laguna_wip.len() >= LAGUNA_WIP_CLASS_COLLAPSES_BELOW,
        "a feature-gated class collapsed ({} cuda-only, {} wgpu-only, {} laguna-wip), which is \
         what a changed header spelling looks like from here even when every per-file delta is \
         clean",
        c.cuda_only.len(),
        c.wgpu_only.len(),
        c.laguna_wip.len(),
    );
}

#[test]
fn every_file_excluded_as_a_shared_helper_declares_no_tests_and_is_path_included_by_a_real_suite() {
    let dir = suite_dir();
    let mut wrong: Vec<String> = Vec::new();
    for helper in SHARED_HELPER_MODULES_PATH_INCLUDED_BY_SUITES_AND_DECLARING_NO_TESTS_THEMSELVES {
        let p = dir.join(helper);
        let Ok(src) = std::fs::read_to_string(&p) else {
            wrong.push(format!(
                "{helper} is excluded from the census as a shared helper but does not exist"
            ));
            continue;
        };
        let (declared, _) = declared_and_ignored_test_fns(&src);
        if declared != 0 {
            wrong.push(format!(
                "{helper} is excluded from the census as a shared helper but declares {declared} \
                 #[test] fn(s); a file that asserts is a suite and belongs in a census class"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n  "));
    let c = corpus();
    assert!(
        c.files_declaring_no_tests_and_not_declared_helpers.is_empty(),
        "these files in tests/ declare no #[test] fn this census can see, yet are not named in \
         SHARED_HELPER_MODULES_PATH_INCLUDED_BY_SUITES_AND_DECLARING_NO_TESTS_THEMSELVES: \
         {:?}.\ncargo builds every tests/*.rs as its own binary, so such a file prints `ok. 0 \
         passed` and reads as a passing suite forever. Either it is a helper -- name it, and this \
         census will prove the claim -- or it declares its tests through an attribute spelling \
         outside TEST_ATTRIBUTE_SPELLINGS_CARGO_RUNS_AS_A_TEST_NOT_ONLY_PLAIN_TEST -- add the new \
         spelling there rather than excluding the file -- or, as with \
         gow_shallow_lmhead_cost_probe.rs the first time this scanner ran, a string literal is \
         defeating the fn walk and literal_content_blanked needs the new literal form.",
        c.files_declaring_no_tests_and_not_declared_helpers,
    );
}

#[test]
fn every_suite_whose_tests_are_all_ignored_is_named_here_with_the_env_vars_it_reads() {
    let c = corpus();
    let dir = suite_dir();
    let recorded: BTreeSet<&str> =
        SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED
            .iter()
            .map(|(n, _)| *n)
            .collect();
    let found: BTreeSet<&str> = c.every_test_ignored.iter().map(String::as_str).collect();
    let mut wrong: Vec<String> = Vec::new();
    for n in found.difference(&recorded) {
        wrong.push(format!(
            "+ {n}: every #[test] in it is #[ignore]d, so `nvk.sh test` prints `ok. 0 passed` for \
             it and a green crate says nothing about the property it gates. Add it to \
             SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED with the env \
             vars it reads."
        ));
    }
    for n in recorded.difference(&found) {
        wrong.push(format!(
            "- {n}: recorded as entirely-#[ignore]d, but it now has at least one test that runs \
             by default (or the file is gone). Drop the entry."
        ));
    }
    for (n, vars) in SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED {
        let Ok(src) = std::fs::read_to_string(dir.join(n)) else {
            continue;
        };
        for var in vars.split_whitespace() {
            if !src.contains(var) {
                wrong.push(format!(
                    "! {n} is recorded as reading {var}, which does not appear in its source. A \
                     documented opt-in nobody can perform is worse than none."
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "the entirely-ignored suites moved:\n  {}\n\nThis class is the second blind spot behind \
         the blanked one: such a suite is present, registered, compiled, and still proves nothing \
         until someone runs it with --ignored and its env vars. By design or not, a plain run of \
         this crate reports `ok. 0 passed` for {} of its {} suites on this axis alone.",
        wrong.join("\n  "),
        SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED.len(),
        c.total(),
    );
    eprintln!(
        "PRESENT BUT SILENT UNLESS OPTED IN ({} suites): {:?}",
        SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED.len(),
        SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED
            .iter()
            .map(|(n, _)| *n)
            .collect::<Vec<_>>(),
    );
}

#[test]
fn every_unconditional_file_whose_tests_carry_their_own_cfg_is_named_here_with_that_count() {
    let c = corpus();
    let found: Vec<(String, usize)> = c.unconditional_with_per_item_cfg.clone();
    let recorded: Vec<(String, usize)> =
        UNCONDITIONAL_FILES_WHOSE_TESTS_HIDE_BEHIND_THEIR_OWN_CFG_SO_THE_HEADER_LIES
            .iter()
            .map(|(n, k)| ((*n).to_string(), *k))
            .collect();
    assert_eq!(
        found, recorded,
        "a file with no crate-level #![cfg] reads as `runs everywhere` from a header walk, and \
         these files defeat that reading by gating individual tests with their own #[cfg]. With \
         one feature such a file still reports a non-zero pass count from its unconditional tests \
         while the gated ones are erased. Update the count here in the same commit that changes \
         the gating."
    );
    eprintln!("unconditional header, per-test #[cfg] inside: {found:?}");
}

#[test]
fn the_skip_return_ratchet_holds_suites_that_go_green_without_their_env_var_may_only_shrink() {
    let sources = suite_sources();
    assert!(
        sources.len() >= A_WALK_THAT_READS_FEWER_FILES_THAN_THIS_IS_BROKEN_NOT_CLEAN,
        "this gate read only {} files under {}; a walk that finds nothing proves nothing",
        sources.len(),
        suite_dir().display(),
    );
    let mut found: BTreeMap<String, usize> = BTreeMap::new();
    for (name, src) in &sources {
        let n = tests_that_return_green_when_an_opt_in_env_var_is_absent(src).len();
        if n > 0 {
            found.insert(name.clone(), n);
        }
    }
    let recorded: BTreeMap<String, usize> =
        SUITES_WITH_TESTS_THAT_RETURN_GREEN_WHEN_AN_OPT_IN_ENV_VAR_IS_ABSENT_THIS_LIST_MAY_ONLY_SHRINK
            .iter()
            .map(|(n, k)| ((*n).to_string(), *k))
            .collect();
    let mut wrong: Vec<String> = Vec::new();
    for (name, n) in &found {
        match recorded.get(name) {
            None => wrong.push(format!(
                "+ {name}: {n} test(s) return instead of failing when the env var they gate on is \
                 absent, and the suite is not in the pinned list. Do not add it: fix it. A suite \
                 whose precondition is absent must panic naming the variable (correct when \
                 already #[ignore]d, because --ignored IS the opt-in) or be #[ignore]d with that \
                 panic. Details: {}",
                tests_that_return_green_when_an_opt_in_env_var_is_absent(
                    &sources.iter().find(|(s, _)| s == name).unwrap().1
                )
                .join("; ")
            )),
            Some(k) if n > k => wrong.push(format!(
                "+ {name}: grew from {k} to {n} skip-returning test(s). The ratchet only turns one \
                 way: fix the new one instead of re-pinning."
            )),
            Some(k) if n < k => wrong.push(format!(
                "- {name}: shrank from {k} to {n}. Good -- pin the new smaller count in \
                 SUITES_WITH_TESTS_THAT_RETURN_GREEN_WHEN_AN_OPT_IN_ENV_VAR_IS_ABSENT_THIS_LIST_MAY_ONLY_SHRINK \
                 so it cannot grow back."
            )),
            Some(_) => {}
        }
    }
    for (name, k) in &recorded {
        if !found.contains_key(name) {
            wrong.push(format!(
                "- {name}: pinned at {k} skip-returning test(s) but now has none (or the file is \
                 gone). Delete the entry so the ratchet tightens."
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "the skip-return ledger moved:\n  {}\n\nEvery pinned entry is a test that the runner \
         counts as PASSED having executed nothing when its env var is absent: `1 passed` that \
         reads as coverage. {} suites carry {} such tests today; the pin exists so the number can \
         only fall.",
        wrong.join("\n  "),
        recorded.len(),
        recorded.values().sum::<usize>(),
    );
    eprintln!(
        "GO GREEN WITHOUT THEIR ENV VAR ({} suites, {} tests): {:?}",
        recorded.len(),
        recorded.values().sum::<usize>(),
        recorded.keys().collect::<Vec<_>>(),
    );
}

#[test]
fn the_headline_blind_spot_counts_are_pinned_so_they_cannot_drift_upward_unnoticed() {
    let c = corpus();
    let blanked = c.blanked_under_the_routine_cuda_only_default();
    eprintln!(
        "BLANKED under the routine `NVK_PKG=nv-models` default (cuda alone), {} of {} suites, \
         each printing `ok. 0 passed` in 0.00s: {:?}",
        blanked.len(),
        c.total(),
        blanked,
    );
    assert_eq!(
        blanked.len(),
        THE_ROUTINE_NVK_DEFAULT_FOR_NV_MODELS_IS_CUDA_ALONE_WHICH_BLANKS_THIS_MANY_SUITES,
        "the count of suites the routine cuda-only default compiles to nothing moved. If it \
         SHRANK, a suite gained a reachable configuration: re-pin the smaller number. If it GREW, \
         a suite just joined the blind spot -- decide out loud, in the commit that moves this \
         number, that one more silent `0 passed` is intended.",
    );
    let ignored: BTreeSet<&str> = c.every_test_ignored.iter().map(String::as_str).collect();
    let mut unable_routine: BTreeSet<&str> = blanked.iter().map(String::as_str).collect();
    unable_routine.extend(ignored.iter());
    let mut unable_cuda_wgpu: BTreeSet<&str> = c.laguna_wip.iter().map(String::as_str).collect();
    unable_cuda_wgpu.extend(ignored.iter());
    eprintln!(
        "UNABLE TO ASSERT on a routine cuda-only run (blanked by cfg, or every test #[ignore]d), \
         {} of {} suites: {:?}",
        unable_routine.len(),
        c.total(),
        unable_routine,
    );
    eprintln!(
        "UNABLE TO ASSERT even under NVK_FEATURES=cuda,wgpu, {} of {} suites: {:?}",
        unable_cuda_wgpu.len(),
        c.total(),
        unable_cuda_wgpu,
    );
    assert_eq!(
        unable_routine.len(),
        SUITES_UNABLE_TO_ASSERT_ON_A_ROUTINE_CUDA_ONLY_RUN_BLANKED_OR_EVERY_TEST_IGNORED,
        "the union of cfg-blanked and entirely-#[ignore]d suites moved for the routine cuda-only \
         run. Smaller: re-pin and keep the gain. Larger: a suite went dark -- name it in the \
         commit before re-pinning.",
    );
    assert_eq!(
        unable_cuda_wgpu.len(),
        SUITES_UNABLE_TO_ASSERT_EVEN_UNDER_CUDA_AND_WGPU_LAGUNA_BLANKED_OR_EVERY_TEST_IGNORED,
        "the union of laguna-wip-blanked and entirely-#[ignore]d suites moved for a \
         NVK_FEATURES=cuda,wgpu run. Smaller: re-pin and keep the gain. Larger: a suite went dark \
         -- name it in the commit before re-pinning.",
    );
}

#[test]
fn the_attribute_scanner_reads_multi_line_ignore_attributes_and_ignores_them_inside_strings() {
    let fixture = "#![cfg(feature = \"cuda\")]\n\
                   #[test]\n\
                   fn plain() {}\n\
                   \n\
                   #[test]\n\
                   #[ignore]\n\
                   fn bare_ignore() {}\n\
                   \n\
                   #[cfg(feature = \"wgpu\")]\n\
                   #[test]\n\
                   #[ignore = \"needs a device; set NV_X=1 -- proves \\\n\
                   something across two lines\"]\n\
                   fn multi_line_ignore() {}\n\
                   \n\
                   fn helper_that_only_mentions_attributes() {\n\
                       panic!(\"this is already #[ignore]d, so --ignored IS the opt-in\");\n\
                   }\n";
    assert_eq!(
        declared_and_ignored_test_fns(fixture),
        (3, 2),
        "the scanner must count three #[test] fns of which two are ignored: a bare #[ignore], a \
         multi-line #[ignore = \"...\"] whose closing bracket lands on a later line, and it must \
         not be fooled by a `#[ignore]` mentioned inside a panic message"
    );
    assert_eq!(
        crate_level_cfg_lines(fixture),
        vec![HEADER_CUDA_ONLY.to_string()],
        "an unindented crate-level #![cfg] is what classification reads; a per-item #[cfg] with \
         one bang fewer is not a header and must not be mistaken for one"
    );
    assert_eq!(
        unindented_positive_cfg_attributes(fixture),
        1,
        "the per-item scanner must see the one #[cfg(feature = \"wgpu\")] guarding a test and \
         must not count the crate-level #![cfg] header as one of them"
    );
    let attribute_macro_forms = "#[tokio::test]\n\
                                 async fn plain_tokio() {}\n\
                                 \n\
                                 #[tokio::test(flavor = \"multi_thread\", worker_threads = 2)]\n\
                                 #[ignore = \"boots the real binary\"]\n\
                                 async fn configured_tokio() {}\n\
                                 \n\
                                 #[rstest]\n\
                                 fn a_fixture_driven_case() {}\n\
                                 \n\
                                 #[test_case(1)]\n\
                                 fn a_table_driven_case() {}\n\
                                 \n\
                                 fn a_plain_helper() {}\n";
    assert_eq!(
        declared_and_ignored_test_fns(attribute_macro_forms),
        (4, 1),
        "cargo runs #[tokio::test] fns as tests, so a file whose only tests carry that attribute \
         is a real suite -- not the test-free helper a plain-#[test] scanner reads it as. \
         #[rstest], #[test_case], #[proptest] and #[bench] are counted for the same reason: \
         TEST_ATTRIBUTE_SPELLINGS_CARGO_RUNS_AS_A_TEST_NOT_ONLY_PLAIN_TEST is the whole list."
    );
}

#[test]
fn the_literal_blanker_erases_braces_and_fn_spellings_inside_strings_so_the_walk_survives_them() {
    let lone_brace_in_a_string = "#[test]\n\
                                  fn parses_malformed_json() {\n\
                                  \x20   assert!(from_json(\"{\").is_err());\n\
                                  \x20   let mut header = String::from(\"{\");\n\
                                  \x20   header.push('x');\n\
                                  }\n\
                                  \n\
                                  #[test]\n\
                                  fn a_suite_declared_after_that_test() {\n\
                                  \x20   assert_eq!(1, 1);\n\
                                  }\n";
    assert_eq!(
        declared_and_ignored_test_fns(lone_brace_in_a_string),
        (2, 0),
        "a lone `{{` inside a string literal is not a block delimiter. Counting it as one leaves \
         the first test looking unterminated and the walk reports it unexaminable -- which is how \
         gemma4_audio_gate.rs's from_full_hf_json_str(\"{{\") probe and \
         gemma4_moe_ct_nvfp4_loader.rs's String::from(\"{{\") header builder defeated the sibling \
         crates' brace counter the first time this scanner ran here."
    );
    let wgsl_shader_in_a_raw_string = "const SRC: &str = r#\"\n\
                                       fn bf16_lo(w: u32) -> f32 { return bitcast<f32>(w); }\n\
                                       fn butterfly32(v: f32) -> f32 {\n\
                                       \x20   return v;\n\
                                       }\n\
                                       \"#;\n\
                                       \n\
                                       #[test]\n\
                                       #[ignore = \"release-only wall-clock probe\"]\n\
                                       fn the_real_test_below_the_shader() {\n\
                                       \x20   assert!(SRC.len() > 0);\n\
                                       }\n";
    assert_eq!(
        declared_and_ignored_test_fns(wgsl_shader_in_a_raw_string),
        (1, 1),
        "WGSL source inside a raw string spells `fn name(...) {{` at column zero, and a walk that \
         reads string content as code turns each shader function into a phantom Rust fn whose \
         body swallows the real #[test] below -- which is how gow_shallow_lmhead_cost_probe.rs \
         read as a test-free file the first time this scanner ran. literal_content_blanked must \
         erase the entire literal body, not only its braces."
    );
    assert_eq!(
        literal_content_blanked("let a = ('{', b'}', \"{[\", r##\"]}\"##); { }"),
        "let a = (' ', b' ', \"  \", r##\"  \"##); { }",
        "every literal form -- char, byte char, plain string, hash-delimited raw string -- must \
         come back blanked to spaces with its delimiters and all code braces intact"
    );
}

#[test]
fn the_skip_return_scanner_reads_the_absent_direction_and_leaves_a_deliberate_opt_out_alone() {
    let fixture = "fn device_or_none() -> Option<usize> {\n\
                   \x20   None\n\
                   }\n\
                   \n\
                   fn bench_gated() -> bool {\n\
                   \x20   std::env::var(\"NV_X_BENCH\").as_deref() == Ok(\"1\")\n\
                   }\n\
                   \n\
                   #[test]\n\
                   fn skips_on_a_direct_env_absence() {\n\
                   \x20   if std::env::var(\"NV_X_TEST\").as_deref() != Ok(\"1\") {\n\
                   \x20       eprintln!(\"SKIP: set NV_X_TEST=1\");\n\
                   \x20       return;\n\
                   \x20   }\n\
                   \x20   assert!(true);\n\
                   }\n\
                   \n\
                   #[test]\n\
                   fn skips_through_a_bool_returning_env_flag_helper() {\n\
                   \x20   if !bench_gated() {\n\
                   \x20       return;\n\
                   \x20   }\n\
                   \x20   assert!(true);\n\
                   }\n\
                   \n\
                   #[test]\n\
                   fn skips_only_when_a_variable_is_deliberately_set() {\n\
                   \x20   if std::env::var(\"NV_X_ALLOW_SKIP\").as_deref() == Ok(\"1\") {\n\
                   \x20       return;\n\
                   \x20   }\n\
                   \x20   assert!(true);\n\
                   }\n\
                   \n\
                   #[test]\n\
                   fn returns_early_on_a_fact_about_the_machine_not_a_variable() {\n\
                   \x20   let n = 3usize;\n\
                   \x20   if n == 0 {\n\
                   \x20       return;\n\
                   \x20   }\n\
                   \x20   assert!(true);\n\
                   }\n\
                   \n\
                   #[test]\n\
                   fn asserts_on_the_variable_then_returns_on_an_absent_device() {\n\
                   \x20   assert_eq!(\n\
                   \x20       std::env::var(\"NV_X_BENCH\").ok().as_deref(),\n\
                   \x20       Some(\"1\"),\n\
                   \x20       \"set NV_X_BENCH=1 -- this loads the GPU\"\n\
                   \x20   );\n\
                   \x20   let Some(_d) = device_or_none() else {\n\
                   \x20       return;\n\
                   \x20   };\n\
                   \x20   assert!(true);\n\
                   }\n";
    let found = tests_that_return_green_when_an_opt_in_env_var_is_absent(fixture);
    assert!(
        found.iter().any(|f| f.contains("skips_on_a_direct_env_absence")),
        "the scanner must see a direct env::var guard whose absent branch returns: {found:?}"
    );
    assert!(
        found
            .iter()
            .any(|f| f.contains("skips_through_a_bool_returning_env_flag_helper")),
        "a zero-argument helper returning bool from an env::var read is the same gate one call \
         away; a scanner that only reads env::var textually misses every `if !bench_gated()`: \
         {found:?}"
    );
    assert!(
        !found
            .iter()
            .any(|f| f.contains("skips_only_when_a_variable_is_deliberately_set")),
        "a guard that fires only when a variable IS set is a deliberate, named opt-out, not a \
         suite going green because a precondition is missing: {found:?}"
    );
    assert!(
        !found
            .iter()
            .any(|f| f.contains("returns_early_on_a_fact_about_the_machine_not_a_variable")),
        "an early return that names no environment variable is out of this gate's scope: \
         {found:?}"
    );
    assert!(
        !found
            .iter()
            .any(|f| f.contains("asserts_on_the_variable_then_returns_on_an_absent_device")),
        "the guard this scanner reads is the block the return sits in, not the six lines above \
         it. A suite that asserts on its variable and then returns because the DEVICE is absent \
         is already the correct shape; reading the assert_eq! above the `let ... else` as the \
         guard makes every such suite a false positive, which is how a gate stops being believed: \
         {found:?}"
    );
    assert_eq!(found.len(), 2, "exactly two of the five fixtures qualify: {found:?}");
}

#[test]
fn the_culled_suites_are_still_gone_and_their_coverage_is_still_missing() {
    let dir = suite_dir();
    let resurrected: Vec<&str> =
        CULLED_SUITES_NO_DELETION_FROM_THIS_DIRECTORY_HAS_EVER_BEEN_RECORDED
            .iter()
            .copied()
            .filter(|s| dir.join(format!("{s}.rs")).exists())
            .collect();
    assert!(
        resurrected.is_empty(),
        "CULLED_SUITES_NO_DELETION_FROM_THIS_DIRECTORY_HAS_EVER_BEEN_RECORDED names \
         {resurrected:?}, which exist again. Move them into the matching CENSUS_* list in the \
         same commit that restored them; the directory walk will demand it anyway."
    );
}

#[cfg(not(feature = "wgpu"))]
#[test]
#[allow(non_snake_case)]
fn the_wgpu_model_suites_were_CFGD_OUT_of_this_binary_SKIPPED_no_wgpu_feature() {
    let c = corpus();
    eprintln!(
        "SKIPPED, NOT PASSED: the {} #![cfg(feature = \"wgpu\")] suites compiled to nothing in \
         this run: {:?}. This is the DEFAULT shape of `NVK_PKG=nv-models nvk.sh test` -- nvk.sh \
         gives cuda,wgpu only to nv-kernels. Re-run with NVK_FEATURES=cuda,wgpu. The list is \
         DERIVED from the directory, not written down.",
        c.wgpu_only.len(),
        c.wgpu_only,
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
#[allow(non_snake_case)]
fn the_cuda_model_suites_were_CFGD_OUT_of_this_binary_SKIPPED_no_cuda_feature() {
    let c = corpus();
    eprintln!(
        "SKIPPED, NOT PASSED: the {} #![cfg(feature = \"cuda\")] suites compiled to nothing in \
         this run: {:?}. Re-run with NVK_FEATURES=cuda (or cuda,wgpu). The list is DERIVED from \
         the directory, not written down.",
        c.cuda_only.len(),
        c.cuda_only,
    );
}

#[cfg(not(feature = "laguna-wip"))]
#[test]
#[allow(non_snake_case)]
fn the_laguna_wip_suites_were_CFGD_OUT_SKIPPED_wgpu_alone_does_not_enable_laguna_wip() {
    let c = corpus();
    eprintln!(
        "SKIPPED, NOT PASSED: the {} #![cfg(feature = \"laguna-wip\")] suites compiled to nothing \
         in this run: {:?}. `laguna-wip = [\"wgpu\"]` in Cargo.toml means laguna-wip PULLS IN \
         wgpu, not the reverse -- NVK_FEATURES=cuda,wgpu still blanks every one of these, which \
         is exactly the reading that once turned a hand count of 137 blanked suites into 128. \
         Re-run with NVK_FEATURES=cuda,wgpu,laguna-wip. The list is DERIVED from the directory, \
         not written down.",
        c.laguna_wip.len(),
        c.laguna_wip,
    );
}
