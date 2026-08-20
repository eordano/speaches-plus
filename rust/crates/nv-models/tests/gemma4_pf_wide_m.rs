#![cfg(feature = "wgpu")]

mod common;
use common::config_json_wrapped_text_config as config_json;
use common::LcgCentered0p1Shift32 as Lcg;
use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_wgpu::{
    pf_gemm_dispatches_per_projection_one_tile_per_mk_max_rows,
    pf_passes_per_chunk_when_no_projection_is_nvfp4, quantize_nvfp4_host, Gemma4Wgpu,
    HostBf16Lin, HostLayer, HostProj, HostWeights,
    PF_EMBED_PASSES_GATHER_SCALE_PLUS_MM_SPLICE, PF_FIXED_PASSES_PER_DENSE_LAYER_RMS8_ROPE2_KVQ2_GELU1,
    PF_WIDE_M_MAX_VIA_MK_MAX_ROW_GEMM_TILES_AT_256B_ALIGNED_OFFSETS,
};
use common::gemma4_host_weights_quant_ffn_opt as host_weights;

fn ctx_or_panic() {
    if std::env::var("NV_G4PFW").as_deref() != Ok("1") {
        panic!("set NV_G4PFW=1 to run this GPU test (it must never silently skip)");
    }
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => eprintln!("[g4pfw] adapter: {}", ctx.summary()),
        Err(e) => panic!("the wide-m prefill gate needs a wgpu adapter: {e}"),
    }
}

fn prompt_for(len: usize, vocab: usize) -> Vec<u32> {
    (0..len)
        .map(|i| (((i + 1) * 7919 + 13) % vocab) as u32)
        .collect()
}

struct EnvPin(&'static str, Option<String>);

impl EnvPin {
    fn set(key: &'static str, value: &str) -> Self {
        let saved = std::env::var(key).ok();
        std::env::set_var(key, value);
        EnvPin(key, saved)
    }
}

impl Drop for EnvPin {
    fn drop(&mut self) {
        match self.1.take() {
            Some(v) => std::env::set_var(self.0, v),
            None => std::env::remove_var(self.0),
        }
    }
}

fn prime_then_logits(m: &mut Gemma4Wgpu, prompt: &[u32]) -> (u32, Vec<u32>) {
    let (last, rest) = prompt.split_last().expect("prompt");
    let done = m.prefill_tokens(rest).expect("prefill_tokens");
    for t in &rest[done..] {
        m.prefill_step(*t).expect("prefill step");
    }
    let (tok, lg) = m.decode_step_logits(*last).expect("decode step logits");
    (tok, lg.into_iter().map(f32::to_bits).collect())
}

fn distinct(bits: &[u32]) -> usize {
    let mut s: Vec<u32> = bits.to_vec();
    s.sort_unstable();
    s.dedup();
    s.len()
}

fn expected_passes(layers: usize, m: usize, nvfp4_ffn_survives: bool) -> usize {
    if !nvfp4_ffn_survives {
        return pf_passes_per_chunk_when_no_projection_is_nvfp4(layers, m);
    }
    let tiles = pf_gemm_dispatches_per_projection_one_tile_per_mk_max_rows(m);
    let int8_attn_qkv_and_o_tiled = 2 * tiles;
    let nvfp4_gate_up_and_down_quant_plus_slot_gemv = 2 * 2;
    layers
        * (PF_FIXED_PASSES_PER_DENSE_LAYER_RMS8_ROPE2_KVQ2_GELU1
            + int8_attn_qkv_and_o_tiled
            + nvfp4_gate_up_and_down_quant_plus_slot_gemv
            + 2 * m)
        + PF_EMBED_PASSES_GATHER_SCALE_PLUS_MM_SPLICE
}

fn run_case(quant_ffn: bool, w8_off: bool) {
    let _w8_pin = w8_off.then(|| EnvPin::set("NV_G4_WGPU_W8_FFN", "off"));
    let nvfp4_ffn_survives = quant_ffn && w8_off;
    let layers = 4usize;
    let hidden = 512usize;
    let inter = 1024usize;
    let vocab = 2048usize;
    let window = 32usize;
    let max_seq = 256usize;
    let wide_m = PF_WIDE_M_MAX_VIA_MK_MAX_ROW_GEMM_TILES_AT_256B_ALIGNED_OFFSETS;
    let prompt = prompt_for(2 * wide_m + 10, vocab);

    let raw = config_json(layers, hidden, inter, vocab, window);
    let config = Gemma4Config::from_hf_json_str(&raw).expect("config");
    let w = host_weights(&config, 0x9e3779b9, quant_ffn);

    let (tok_wide, bits_wide, passes_wide) = {
        let _pin = EnvPin::set("NV_G4_WGPU_PF_M", &wide_m.to_string());
        let mut m = Gemma4Wgpu::new(config.clone(), &w, max_seq).expect("build wide-m graph");
        assert_eq!(
            m.prefill_chunk_len(),
            wide_m,
            "NV_G4_WGPU_PF_M={wide_m} did not widen the chunk (quant_ffn={quant_ffn}); a \
             [gemma4_wgpu] boot disabler fired above and the rest of this test would compare \
             the narrow arm against itself"
        );
        let (tok, bits) = prime_then_logits(&mut m, &prompt);
        (tok, bits, m.prefill_pass_count())
    };
    assert_eq!(
        passes_wide,
        expected_passes(layers, wide_m, nvfp4_ffn_survives),
        "wide-m pass list length disagrees with the named pass-count math (quant_ffn={quant_ffn} \
         w8_off={w8_off}; a projection landing in an unplanned format changes the count)"
    );

    let (tok_narrow, bits_narrow, narrow_m, passes_narrow) = {
        let mut m = Gemma4Wgpu::new(config.clone(), &w, max_seq).expect("build default graph");
        let nm = m.prefill_chunk_len();
        assert!(
            nm >= 2 && nm < wide_m,
            "the default chunk must stay narrower than the wide arm, got {nm}"
        );
        let (tok, bits) = prime_then_logits(&mut m, &prompt);
        (tok, bits, nm, m.prefill_pass_count())
    };
    assert_eq!(
        passes_narrow,
        expected_passes(layers, narrow_m, nvfp4_ffn_survives),
        "default-m pass list length disagrees with the named pass-count math"
    );

    let (tok_step, bits_step) = {
        let _pin = EnvPin::set("NV_G4_WGPU_PF_M", "0");
        let mut m = Gemma4Wgpu::new(config.clone(), &w, max_seq).expect("build step graph");
        assert_eq!(
            m.prefill_chunk_len(),
            0,
            "NV_G4_WGPU_PF_M=0 must disable chunking so this arm primes through the decode graph"
        );
        prime_then_logits(&mut m, &prompt)
    };

    assert!(
        distinct(&bits_step) > 1,
        "step-primed logits are constant; the comparison below would be vacuous"
    );
    let diff_wide = bits_step
        .iter()
        .zip(bits_wide.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff_wide, 0,
        "quant_ffn={quant_ffn} w8_off={w8_off}: {diff_wide} of {vocab} logit lanes differ \
         between the {wide_m}-row tiled prefill and the step-primed decode path"
    );
    let diff_narrow = bits_step
        .iter()
        .zip(bits_narrow.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff_narrow, 0,
        "quant_ffn={quant_ffn} w8_off={w8_off}: the m={narrow_m} control arm drifted from the \
         step-primed path, so a wide-m diff would not implicate the tiling"
    );
    assert_eq!(tok_wide, tok_step);
    assert_eq!(tok_narrow, tok_step);
    eprintln!(
        "[g4pfw] quant_ffn={quant_ffn} w8_off={w8_off}: m={wide_m} ({passes_wide} passes/chunk) \
         and m={narrow_m} ({passes_narrow} passes/chunk) both bit-identical to step priming \
         over {} lanes",
        bits_step.len()
    );
}

#[test]
fn wide_m_tiled_prefill_matches_step_and_narrow_chunk_bit_exactly() {
    ctx_or_panic();
    run_case(false, false);
    run_case(true, false);
    run_case(true, true);
}
