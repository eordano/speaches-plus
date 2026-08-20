use candle_core::{DType, Device, Tensor};
use nv_models::gemma4_mm_splice::{
    audio_num_soft_tokens, expand_audio_placeholder, expand_image_placeholder, placeholder_runs,
    splice_mm_embeddings, MmItem, Modality, GEMMA4_AUDIO_TOKEN_ID, GEMMA4_BOA_TOKEN_ID,
    GEMMA4_BOI_TOKEN_ID, GEMMA4_EOA_TOKEN_ID, GEMMA4_EOI_TOKEN_ID, GEMMA4_IMAGE_TOKEN_ID,
};

const HIDDEN: usize = 8;

fn rows(seq: usize, base: f32) -> Tensor {
    let data: Vec<f32> = (0..seq * HIDDEN).map(|i| base + i as f32).collect();
    Tensor::from_vec(data, (seq, HIDDEN), &Device::Cpu).unwrap()
}

fn v(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1().unwrap()
}

#[test]
fn splice_preserves_length_replaces_runs_and_leaves_text_rows_bit_identical() {
    let tokens: Vec<u32> = vec![
        1,
        2,
        GEMMA4_IMAGE_TOKEN_ID,
        GEMMA4_IMAGE_TOKEN_ID,
        GEMMA4_IMAGE_TOKEN_ID,
        7,
        GEMMA4_AUDIO_TOKEN_ID,
        GEMMA4_AUDIO_TOKEN_ID,
        9,
    ];
    let text = rows(tokens.len(), 0.0);
    let img = rows(3, 1000.0);
    let aud = rows(2, 2000.0);
    let out = splice_mm_embeddings(
        &text,
        &tokens,
        &[
            MmItem { modality: Modality::Image, position: 2, embedding: img.clone() },
            MmItem { modality: Modality::Audio, position: 6, embedding: aud.clone() },
        ],
    )
    .unwrap();
    assert_eq!(
        out.dims(),
        text.dims(),
        "splicing must never change the sequence length, or every later position is shifted"
    );
    let o = v(&out);
    let t = v(&text);
    for row in [0usize, 1, 5, 8] {
        assert_eq!(
            &o[row * HIDDEN..(row + 1) * HIDDEN],
            &t[row * HIDDEN..(row + 1) * HIDDEN],
            "text row {row} was disturbed by the splice"
        );
    }
    assert_eq!(&o[2 * HIDDEN..5 * HIDDEN], &v(&img)[..], "image rows not spliced verbatim");
    assert_eq!(&o[6 * HIDDEN..8 * HIDDEN], &v(&aud)[..], "audio rows not spliced verbatim");
}

#[test]
fn items_arriving_out_of_order_are_matched_to_runs_by_position() {
    let tokens: Vec<u32> = vec![
        GEMMA4_IMAGE_TOKEN_ID,
        1,
        GEMMA4_IMAGE_TOKEN_ID,
    ];
    let text = rows(3, 0.0);
    let a = rows(1, 100.0);
    let b = rows(1, 200.0);
    let out = splice_mm_embeddings(
        &text,
        &tokens,
        &[
            MmItem { modality: Modality::Image, position: 2, embedding: b.clone() },
            MmItem { modality: Modality::Image, position: 0, embedding: a.clone() },
        ],
    )
    .unwrap();
    let o = v(&out);
    assert_eq!(&o[..HIDDEN], &v(&a)[..], "run at 0 must get the position-0 item, not the first item given");
    assert_eq!(&o[2 * HIDDEN..], &v(&b)[..], "run at 2 must get the position-2 item");
}

#[test]
fn no_placeholders_passes_the_embeddings_through_untouched() {
    let tokens = vec![1u32, 2, 3];
    let text = rows(3, 5.0);
    let out = splice_mm_embeddings(&text, &tokens, &[]).unwrap();
    assert_eq!(v(&out), v(&text));
}

#[test]
fn every_mismatch_is_an_error_never_a_panic_or_a_silent_partial_splice() {
    let tokens: Vec<u32> = vec![1, GEMMA4_IMAGE_TOKEN_ID, GEMMA4_IMAGE_TOKEN_ID, 4];
    let text = rows(4, 0.0);
    let good = || MmItem { modality: Modality::Image, position: 1, embedding: rows(2, 9.0) };

    let cases: Vec<(&str, Vec<MmItem>)> = vec![
        ("runs but no items", vec![]),
        ("item count mismatch", vec![good(), good()]),
        ("misaligned position", vec![MmItem { position: 2, ..good() }]),
        ("wrong modality", vec![MmItem { modality: Modality::Audio, ..good() }]),
        ("wrong row count", vec![MmItem { embedding: rows(3, 9.0), ..good() }]),
        ("wrong dtype", vec![MmItem {
            embedding: rows(2, 9.0).to_dtype(DType::F64).unwrap(),
            ..good()
        }]),
    ];
    for (label, items) in cases {
        assert!(
            splice_mm_embeddings(&text, &tokens, &items).is_err(),
            "{label}: accepted -- a wrong splice ships corrupted embeddings into the forward"
        );
    }
    assert!(
        splice_mm_embeddings(&text, &tokens, &[good()]).is_ok(),
        "the negative control failed: the well-formed case must splice, or every row above \
         proves rejection of the setup rather than of its corruption"
    );
}

#[test]
fn expanded_placeholders_cannot_merge_because_the_frame_tokens_break_the_run() {
    let one = expand_image_placeholder(3);
    assert_eq!(one[0], GEMMA4_BOI_TOKEN_ID);
    assert_eq!(*one.last().unwrap(), GEMMA4_EOI_TOKEN_ID);
    let two_images: Vec<u32> = one.iter().chain(one.iter()).cloned().collect();
    let runs = placeholder_runs(&two_images);
    assert_eq!(
        runs.len(),
        2,
        "two adjacent expanded images must stay two runs; without the BOI/EOI framing the \
         scanner would merge same-id neighbours and both towers' rows would land in one run"
    );
    assert!(runs.iter().all(|r| r.len == 3 && r.modality == Modality::Image));

    let audio = expand_audio_placeholder(2);
    assert_eq!(audio, vec![
        GEMMA4_BOA_TOKEN_ID,
        GEMMA4_AUDIO_TOKEN_ID,
        GEMMA4_AUDIO_TOKEN_ID,
        GEMMA4_EOA_TOKEN_ID,
    ]);
}

#[test]
fn audio_soft_token_count_matches_the_conv_stack_arithmetic_and_its_cap() {
    assert_eq!(
        audio_num_soft_tokens(16000, 16000, 1500),
        25,
        "1 s at 16 kHz: 99 mel frames through two stride-2 convs is 25 soft tokens"
    );
    assert_eq!(audio_num_soft_tokens(16000, 16000, 10), 10, "the cap binds");
    assert_eq!(audio_num_soft_tokens(0, 16000, 1500), 0, "no samples, no tokens");
    assert_eq!(audio_num_soft_tokens(100, 16000, 1500), 0, "sub-frame audio yields nothing");
    assert_eq!(audio_num_soft_tokens(16000, 0, 1500), 0, "zero rate must not divide by zero");
    let short = audio_num_soft_tokens(8000, 16000, 1500);
    let long = audio_num_soft_tokens(32000, 16000, 1500);
    assert!(short < 25 && long > 25, "the count must grow with duration: {short} vs {long}");
}
