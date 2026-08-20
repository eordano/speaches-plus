#![cfg(feature = "wgpu")]

mod common;
use common::TINY_CONFIG;
use std::sync::Mutex;

use nv_kernels::wgpu_backend::kernels::quant_gemv::QFormat;
use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_wgpu::{
    set_attn_variant, AttnQuant, AttnVariant, Gemma4Wgpu, HostBf16Lin, HostLayer, HostProj,
    HostWeights, ATTN_VARIANT_DEFAULT,
};
use common::LcgCentered0p1Shift32F64 as Lcg;
use common::ctx_or_skip_bool as ctx_or_skip;

fn variant(fmt: QFormat, group: usize, lo: usize, hi: usize) -> AttnVariant {
    AttnVariant {
        on: true,
        quant: AttnQuant { fmt, group, lo, hi },
        legacy_epilogue: 0,
    }
}

fn variant_off() -> AttnVariant {
    AttnVariant {
        on: false,
        ..ATTN_VARIANT_DEFAULT
    }
}

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn tiny_host_weights(config: &Gemma4Config, seed: u64) -> HostWeights {
    let mut rng = Lcg(seed);
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let n_q = config.num_attention_heads;
    let mut layers = Vec::new();
    for i in 0..config.num_hidden_layers {
        let kind = config.layer_kind(i);
        let hd = config.head_dim_for(kind);
        let nkv = config.num_kv_heads_for(kind);
        let q_dim = n_q * hd;
        let kv_dim = nkv * hd;
        let has_v = !matches!(
            (kind, config.attention_k_eq_v),
            (LayerType::FullAttention, true)
        );
        let qkv_rows = q_dim + kv_dim * if has_v { 2 } else { 1 };
        layers.push(HostLayer {
            kind,
            input_ln: rng.bf16_vec_around_one(hidden),
            post_attn_ln: rng.bf16_vec_around_one(hidden),
            pre_ff_ln: rng.bf16_vec_around_one(hidden),
            post_ff_ln: rng.bf16_vec_around_one(hidden),
            q_norm: rng.bf16_vec_around_one(hd),
            k_norm: rng.bf16_vec_around_one(hd),
            layer_scalar: 1.0,
            has_v,
            qkv: HostProj::Bf16(HostBf16Lin {
                w: rng.bf16_vec(qkv_rows * hidden),
                n: qkv_rows,
                k: hidden,
            }),
            o: HostProj::Bf16(HostBf16Lin {
                w: rng.bf16_vec(hidden * q_dim),
                n: hidden,
                k: q_dim,
            }),
            gate_up: HostProj::Bf16(HostBf16Lin {
                w: rng.bf16_vec(2 * inter * hidden),
                n: 2 * inter,
                k: hidden,
            }),
            down: HostProj::Bf16(HostBf16Lin {
                w: rng.bf16_vec(hidden * inter),
                n: hidden,
                k: inter,
            }),
        });
    }
    HostWeights {
        embed: rng.bf16_vec(config.vocab_size * hidden),
        final_norm: rng.bf16_vec_around_one(hidden),
        layers,
    }
}

fn clear_env() {
    set_attn_variant(None);
}

#[test]
fn attn_quant_default_is_int8_128_and_labels_and_windows_behave() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    clear_env();
    let d = nv_models::gemma4_wgpu::attn_quant_config();
    assert_eq!(d.fmt, QFormat::Int8);
    assert_eq!(d.group, 128);
    assert!(d.covers(0) && d.covers(59) && d.covers(usize::MAX - 1));
    assert_eq!(d.label(), "int8/128");

    let e = AttnQuant {
        fmt: QFormat::E4m3,
        group: 0,
        lo: 0,
        hi: usize::MAX,
    };
    assert_eq!(e.label(), "e4m3/row");

    let q = AttnQuant {
        fmt: QFormat::Int8,
        group: 128,
        lo: 4,
        hi: 9,
    };
    assert!(!q.covers(3) && q.covers(4) && q.covers(8) && !q.covers(9));
    assert_eq!(q.label(), "int8/128@[4,9)");
}

const R1R5_FLIP_EVIDENCE: &str = "\
R1R5 (int8 attention projections + int8 FFN) is DEFAULT-ON for the gemma4 dense wgpu graph.
The first ACCEPT was measured with PER-ROW int8 attention while attn_quant_config() had since
moved int8 to group-128 scales, so the blessed and the shipping configuration were not the same
thing; the battery was re-run under the shipping group-128 config and ACCEPTed strictly harder
than the per-row run it replaced on every gate (worst-control mean KL, max step KL, max rho,
answer-text identity, greedy token-exactness). Current numbers: perf/runs.jsonl.
The attention variant is hardcoded (env knobs removed): to revert, edit
ATTN_VARIANT_DEFAULT in gemma4_wgpu.rs; W8_FFN stays env-revertible via NV_G4_WGPU_W8_FFN=off.";

fn clear_w8_env() {
    for v in [
        "NV_G4_WGPU_W8_FFN",
        "NV_G4_WGPU_W8_FFN_GROUP",
        "NV_WGPU_LMHEAD_INT8",
    ] {
        std::env::remove_var(v);
    }
}

#[test]
fn r1r5_defaults_match_the_recorded_verdict_and_stay_constructible() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    clear_env();
    clear_w8_env();

    let line = nv_models::gemma4_wgpu::weight_format_boot_line();
    eprintln!("{line}");
    eprintln!("{R1R5_FLIP_EVIDENCE}");
    for want in [
        "attn.qkv+o=int8/128",
        "ffn.gate_up=int8/128 if nvfp4",
        "ffn.down=int8/128 if nvfp4",
        "lm_head=checkpoint",
    ] {
        assert!(
            line.contains(want),
            "the shipped default must report {want}, got: {line}\n{R1R5_FLIP_EVIDENCE}"
        );
    }

    set_attn_variant(Some(variant(QFormat::E4m3, 0, 0, usize::MAX)));
    std::env::set_var("NV_G4_WGPU_W8_FFN", "off");
    let rev = nv_models::gemma4_wgpu::weight_format_boot_line();
    eprintln!("{rev}");
    for want in [
        "attn.qkv+o=e4m3/row",
        "ffn.gate_up=checkpoint",
        "ffn.down=checkpoint",
    ] {
        assert!(
            rev.contains(want),
            "the pre-flip configuration must stay constructible via set_attn_variant; \
             expected {want}, got: {rev}"
        );
    }

    set_attn_variant(Some(variant_off()));
    assert!(
        nv_models::gemma4_wgpu::weight_format_boot_line().contains("attn.qkv+o=checkpoint"),
        "an off attn variant must still drop attention quantization entirely"
    );

    clear_env();
    clear_w8_env();
}

#[test]
fn synthetic_every_format_and_group_runs_and_ranks_as_predicted() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !ctx_or_skip() {
        return;
    }
    clear_env();
    let config = Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap();
    let weights = tiny_host_weights(&config, 0xf8f8);
    let steps: Vec<u32> = vec![
        7, 300, 42, 511, 0, 13, 99, 250, 8, 9, 10, 11, 12, 14, 15, 16,
    ];

    let run = |fmt: Option<(QFormat, usize)>| -> (u64, usize, Vec<(u32, Vec<f32>)>) {
        match fmt {
            None => set_attn_variant(Some(variant_off())),
            Some((f, g)) => set_attn_variant(Some(variant(f, g, 0, usize::MAX))),
        }
        let m = Gemma4Wgpu::new(
            Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap(),
            &weights,
            64,
        );
        clear_env();
        let mut m = m.unwrap();
        let out: Vec<(u32, Vec<f32>)> = steps
            .iter()
            .map(|t| m.decode_step_logits(*t).unwrap())
            .collect();
        (m.weight_bytes_per_token(), m.pass_count(), out)
    };

    let (base_bytes, base_passes, base) = run(None);
    let mut rows: Vec<(String, f64, u64, usize)> = Vec::new();
    for (f, g) in [
        (QFormat::E4m3, 0usize),
        (QFormat::E4m3, 128),
        (QFormat::E4m3, 32),
        (QFormat::Int8, 0),
        (QFormat::Int8, 128),
        (QFormat::Int8, 32),
    ] {
        let (bytes, passes, out) = run(Some((f, g)));
        let f = f.label();
        let mut se = 0f64;
        let mut sr = 0f64;
        let mut agree = 0usize;
        for ((bt, bl), (at, al)) in base.iter().zip(out.iter()) {
            assert!(
                al.iter().all(|v| v.is_finite()),
                "{f}/{g}: non-finite logits"
            );
            for (x, y) in bl.iter().zip(al.iter()) {
                se += ((x - y) as f64).powi(2);
                sr += (*x as f64).powi(2);
            }
            if bt == at {
                agree += 1;
            }
        }
        let rel = (se / sr).sqrt();
        eprintln!(
            "  {f}/{g:<4} logit rms_rel {rel:.4e}  argmax {agree}/{}  bytes/token {bytes} ({:+.2}%)  passes {passes}",
            steps.len(),
            100.0 * (bytes as f64 - base_bytes as f64) / base_bytes as f64
        );
        assert_eq!(
            passes, base_passes,
            "{f}/{g} must not change the dispatch count"
        );
        assert!(bytes < base_bytes, "{f}/{g} must reduce weight bytes");
        rows.push((format!("{f}/{g}"), rel, bytes, passes));
    }
    let get = |k: &str| rows.iter().find(|(n, _, _, _)| n == k).unwrap().1;
    assert!(
        get("e4m3/0") / get("e4m3/32") < 1.6,
        "e4m3 logit error should be granularity-insensitive: {:.3e} -> {:.3e}",
        get("e4m3/0"),
        get("e4m3/32")
    );
    assert!(
        get("int8/128") < get("e4m3/0"),
        "int8 group=128 must beat e4m3 per-row end to end: {:.3e} vs {:.3e}",
        get("int8/128"),
        get("e4m3/0")
    );
}

#[test]
fn only_the_qkv_and_o_projections_are_swept_into_the_quantized_path() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !ctx_or_skip() {
        return;
    }
    clear_env();
    let config = Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap();
    let weights = tiny_host_weights(&config, 0xabcd);
    let n_q = config.num_attention_heads;

    let build = |group: usize, on: bool| -> u64 {
        if on {
            set_attn_variant(Some(variant(QFormat::Int8, group, 0, usize::MAX)));
        } else {
            set_attn_variant(Some(variant_off()));
        }
        let m = Gemma4Wgpu::new(
            Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap(),
            &weights,
            64,
        );
        clear_env();
        m.unwrap().weight_bytes_per_token()
    };

    for group in [0usize, 128, 32] {
        let mut expect = 0u64;
        for i in 0..config.num_hidden_layers {
            let kind = config.layer_kind(i);
            let hd = config.head_dim_for(kind);
            let nkv = config.num_kv_heads_for(kind);
            let q_dim = n_q * hd;
            let kv_dim = nkv * hd;
            let has_v = !matches!(
                (kind, config.attention_k_eq_v),
                (LayerType::FullAttention, true)
            );
            let qkv_rows = q_dim + kv_dim * if has_v { 2 } else { 1 };
            for (n, k) in [(qkv_rows, config.hidden_size), (config.hidden_size, q_dim)] {
                let per = nv_kernels::wgpu_backend::kernels::quant_gemv::scales_per_row(k, group);
                expect += (2 * n * k) as u64 - ((n * k) as u64 + 4 * (n * per) as u64);
            }
        }
        let bf16 = build(0, false);
        let q = build(group, true);
        eprintln!(
            "  group={group:<4} saving {} bytes/token, analytic qkv+o-only saving {expect}",
            bf16 - q
        );
        assert_eq!(
            bf16 - q,
            expect,
            "group={group}: the byte saving must equal exactly the qkv and o projections. A \
             mismatch means a norm, the mlp, or the embedding got swept into the quantized path."
        );
    }
}

#[test]
fn layer_window_restricts_quantization_to_the_named_range() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !ctx_or_skip() {
        return;
    }
    clear_env();
    let config = Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap();
    let weights = tiny_host_weights(&config, 0x1234);
    let build = |window: Option<(usize, usize)>| -> u64 {
        match window {
            None => set_attn_variant(Some(variant_off())),
            Some((lo, hi)) => set_attn_variant(Some(variant(QFormat::Int8, 128, lo, hi))),
        }
        let m = Gemma4Wgpu::new(
            Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap(),
            &weights,
            64,
        );
        clear_env();
        m.unwrap().weight_bytes_per_token()
    };
    let bf16 = build(None);
    let all = build(Some((0, 6)));
    let none = build(Some((0, 0)));
    let half = build(Some((0, 3)));
    eprintln!(
        "  bytes/token bf16 {bf16}, window[0,0) {none}, window[0,3) {half}, window[0,6) {all}"
    );
    assert_eq!(none, bf16, "an empty window must leave every layer in bf16");
    let tail = build(Some((3, 6)));
    assert!(
        all < half && half < bf16,
        "byte count must fall monotonically with window width"
    );
    assert_eq!(
        (bf16 - half) + (bf16 - tail),
        bf16 - all,
        "disjoint windows must account for exactly the whole saving"
    );
}
