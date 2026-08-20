use nv_models::laguna::{
    attn_quant_gate_check, lmhead_fp8_mode, lmhead_quant_gate_check, LmHeadFp8Mode,
};

#[test]
fn attn_w8_and_fp8_gates_are_mutually_exclusive() {
    std::env::remove_var("NV_LAGUNA_ATTN_W8");
    std::env::remove_var("NV_LAGUNA_ATTN_FP8");
    assert!(attn_quant_gate_check().is_ok());

    std::env::set_var("NV_LAGUNA_ATTN_W8", "1");
    assert!(attn_quant_gate_check().is_ok());

    std::env::set_var("NV_LAGUNA_ATTN_FP8", "shape");
    let err = attn_quant_gate_check().expect_err("both gates set must error");
    assert!(
        err.to_string().contains("mutually exclusive"),
        "unexpected error: {err}"
    );

    std::env::remove_var("NV_LAGUNA_ATTN_W8");
    assert!(attn_quant_gate_check().is_ok());
    std::env::remove_var("NV_LAGUNA_ATTN_FP8");
}

#[test]
fn lmhead_int8_and_fp8_gates_are_mutually_exclusive() {
    std::env::remove_var("NV_LAGUNA_LMHEAD_INT8");
    std::env::remove_var("NV_LAGUNA_LMHEAD_FP8");
    assert!(lmhead_quant_gate_check().is_ok());

    std::env::set_var("NV_LAGUNA_LMHEAD_FP8", "1");
    assert!(lmhead_quant_gate_check().is_ok());

    std::env::set_var("NV_LAGUNA_LMHEAD_INT8", "1");
    let err = lmhead_quant_gate_check().expect_err("both lm_head gates set must error");
    assert!(
        err.to_string().contains("mutually exclusive"),
        "unexpected error: {err}"
    );

    std::env::set_var("NV_LAGUNA_LMHEAD_FP8", "0");
    assert!(
        lmhead_quant_gate_check().is_ok(),
        "INT8 with FP8=0 (force-off) must be allowed"
    );

    assert_eq!(lmhead_fp8_mode(), LmHeadFp8Mode::ForceOff);

    std::env::remove_var("NV_LAGUNA_LMHEAD_INT8");
    assert!(lmhead_quant_gate_check().is_ok());
    std::env::set_var("NV_LAGUNA_LMHEAD_FP8", "1");
    assert_eq!(lmhead_fp8_mode(), LmHeadFp8Mode::ForceOn);
    std::env::remove_var("NV_LAGUNA_LMHEAD_FP8");
    assert_eq!(lmhead_fp8_mode(), LmHeadFp8Mode::DefaultScoped);
}
