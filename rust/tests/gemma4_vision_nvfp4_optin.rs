use candle_core::DType;
use speaches_plus::oapi::chat_multimodal::{
    vision_weight_mode_from, VISION_NVFP4_STAYS_OPT_IN_UNTIL_AN_IMAGE_ACCURACY_FIXTURE_EXISTS,
};

#[test]
fn nvfp4_is_opt_in_and_the_default_stays_f32() {
    assert_eq!(vision_weight_mode_from(None), (DType::F32, false));
    assert_eq!(vision_weight_mode_from(Some("bf16")), (DType::BF16, false));
    assert_eq!(vision_weight_mode_from(Some("int8")), (DType::U8, false));
    assert_eq!(vision_weight_mode_from(Some("nvfp4")), (DType::BF16, true));
    assert_eq!(vision_weight_mode_from(Some("unknown")), (DType::F32, false));
}

#[test]
fn the_documented_opt_in_spelling_engages_nvfp4() {
    let (key, value) = VISION_NVFP4_STAYS_OPT_IN_UNTIL_AN_IMAGE_ACCURACY_FIXTURE_EXISTS
        .split_once('=')
        .expect("opt-in constant must be a KEY=VALUE spelling");
    assert_eq!(key, "NV_MM_VISION_DTYPE");
    let (dtype, nvfp4) = vision_weight_mode_from(Some(value));
    assert_eq!(dtype, DType::BF16);
    assert!(nvfp4, "the documented spelling must engage the nvfp4 opt-in");
}
