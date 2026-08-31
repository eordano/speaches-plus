#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::kernels::flash_decode as fd;
use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseConfig;
use nv_models::qwen3_5_moe::LayerType;
use nv_models::qwen3_5_moe_wgpu::HostBf16Lin;

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32) - 1.0
    }
    fn bf16_vec(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_bits())
            .collect()
    }
    fn f32_vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_f32())
            .collect()
    }
}

fn bf16_lin(r: &mut Lcg, n: usize, k: usize, scale: f32) -> HostBf16Lin {
    HostBf16Lin {
        w: r.bf16_vec(n * k, scale),
        n,
        k,
    }
}

fn norm_vec(r: &mut Lcg, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| half::bf16::from_f32(1.0 + 0.1 * r.next_f32()).to_bits())
        .collect()
}

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static L: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    L.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn gpu_present_or_explicit_skip() -> bool {
    if nv_kernels::wgpu_backend::WgpuContext::shared().is_ok() {
        return true;
    }
    if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() == Ok("1") {
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): no wgpu adapter");
        return false;
    }
    panic!("no wgpu adapter; this gate must never silently skip");
}

fn tiny_config() -> Qwen3_5DenseConfig {
    Qwen3_5DenseConfig {
        hidden_size: 128,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 32,
        intermediate_size: 192,
        vocab_size: 64,
        max_position_embeddings: 96,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        partial_rotary_factor: 0.25,
        bos_token_id: None,
        eos_token_id: 1,
        layer_types: vec![
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ],
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        attn_output_gate: true,
        tie_word_embeddings: false,
    }
}

fn tiny_weights(cfg: &Qwen3_5DenseConfig, seed: u64) -> q3d::HostDenseWeights {
    let mut r = Lcg::new(seed);
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let hd = cfg.head_dim;
    let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let ks = cfg.linear_conv_kernel_dim;

    let mut layers = Vec::new();
    for li in 0..cfg.num_hidden_layers {
        let mixer = match cfg.layer_types[li] {
            LayerType::LinearAttention => {
                q3d::HostDenseMixer::Delta(Box::new(nv_models::qwen3_5_moe_wgpu::HostDeltaNet {
                    in_proj_qkv: bf16_lin(&mut r, conv_dim, hidden, 0.12),
                    in_proj_z: bf16_lin(&mut r, value_dim, hidden, 0.12),
                    in_proj_ab: bf16_lin(&mut r, 2 * cfg.linear_num_value_heads, hidden, 0.12),
                    conv1d: r.f32_vec(conv_dim * ks, 0.4),
                    a_log: r.f32_vec(cfg.linear_num_value_heads, 0.5),
                    dt_bias: r.f32_vec(cfg.linear_num_value_heads, 0.5),
                    norm_w: norm_vec(&mut r, cfg.linear_value_head_dim),
                    out_proj: bf16_lin(&mut r, hidden, value_dim, 0.12),
                }))
            }
            LayerType::FullAttention => {
                let q_out = cfg.num_attention_heads * hd * 2;
                let kv_out = cfg.num_key_value_heads * hd;
                q3d::HostDenseMixer::Attn(Box::new(q3d::HostDenseAttention {
                    q: bf16_lin(&mut r, q_out, hidden, 0.12).into(),
                    k: bf16_lin(&mut r, kv_out, hidden, 0.12).into(),
                    v: bf16_lin(&mut r, kv_out, hidden, 0.12).into(),
                    o: bf16_lin(&mut r, hidden, cfg.num_attention_heads * hd, 0.12).into(),
                    q_norm: norm_vec(&mut r, hd),
                    k_norm: norm_vec(&mut r, hd),
                    qkv_bias_bf16: vec![],
                }))
            }
        };
        layers.push(q3d::HostDenseLayer {
            input_ln: norm_vec(&mut r, hidden),
            post_attn_ln: norm_vec(&mut r, hidden),
            mixer,
            delta_fp8: Default::default(),
            mlp: q3d::HostDenseMlp {
                gate: bf16_lin(&mut r, inter, hidden, 0.15).into(),
                up: bf16_lin(&mut r, inter, hidden, 0.15).into(),
                down: bf16_lin(&mut r, hidden, inter, 0.15).into(),
            },
        });
    }

    q3d::HostDenseWeights {
        embed: r.bf16_vec(cfg.vocab_size * hidden, 0.6),
        final_norm: norm_vec(&mut r, hidden),
        lm_head: r.bf16_vec(cfg.vocab_size * hidden, 0.2),
        layers,
    }
}

const ARM_ENV: &str = q3d::KV_FP8_TP_ENV_DEFAULT_ON_SET_0_FOR_THE_POSITION_MAJOR_STAGE1;

const MAX_SEQ: usize = 96;

const PROMPT_LEN_CROSSES_A_16_ROW_CHUNK_SO_BOTH_KT_WRITERS_RUN: usize = 39;

const GREEDY_STEPS: usize = 8;

const KNEE_ENV: &str = q3d::KV_FP8_TP_MIN_TOTAL_ENV_TP_RUNS_ONLY_AT_TOTAL_AT_OR_ABOVE_THIS_PLAIN_INTEGER;

fn build(cfg: &Qwen3_5DenseConfig, hw: &q3d::HostDenseWeights, tp: bool) -> q3d::Qwen3_5DenseWgpu {
    std::env::set_var(ARM_ENV, if tp { "1" } else { "0" });
    std::env::set_var(KNEE_ENV, "0");
    let gpu = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), hw, MAX_SEQ).expect(
        "build a pure arm: knee pinned to 0 because the unset default is now the depth gate",
    );
    std::env::remove_var(KNEE_ENV);
    std::env::remove_var(ARM_ENV);
    gpu
}

fn build_gated(
    cfg: &Qwen3_5DenseConfig,
    hw: &q3d::HostDenseWeights,
    min_total: u32,
) -> q3d::Qwen3_5DenseWgpu {
    std::env::set_var(ARM_ENV, "1");
    std::env::set_var(KNEE_ENV, min_total.to_string());
    let gpu = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), hw, MAX_SEQ).expect("build gated");
    std::env::remove_var(KNEE_ENV);
    std::env::remove_var(ARM_ENV);
    gpu
}

fn prompt() -> Vec<u32> {
    (0..PROMPT_LEN_CROSSES_A_16_ROW_CHUNK_SO_BOTH_KT_WRITERS_RUN as u32)
        .map(|i| (i * 7 + 3) % 64)
        .collect()
}

fn drive(gpu: &mut q3d::Qwen3_5DenseWgpu) -> (usize, Vec<u32>) {
    let tokens = prompt();
    let done = gpu.prefill_tokens(&tokens).expect("prefill_tokens");
    for t in &tokens[done..] {
        gpu.prefill_step(*t).expect("tail prefill step");
    }
    let mut ids = Vec::new();
    let mut next = 2u32;
    for _ in 0..GREEDY_STEPS {
        next = gpu.decode_step(next).expect("greedy step");
        assert!((next as usize) < 64, "greedy token out of vocab");
        ids.push(next);
    }
    (tokens.len() + GREEDY_STEPS, ids)
}

fn as_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn tp_attn_state_groups(
    bufs: &[(u64, Vec<u32>)],
    n_kv: usize,
    hd: usize,
) -> Vec<(usize, Vec<&Vec<u32>>)> {
    let kc = (MAX_SEQ * n_kv * hd * 2) as u64;
    let kc8 = (MAX_SEQ * n_kv * hd) as u64;
    let sc = (MAX_SEQ * n_kv * 4) as u64;
    let want = [kc, kc, kc8, kc8, sc, sc, kc8];
    let mut groups = Vec::new();
    let mut i = 0;
    while i + want.len() <= bufs.len() {
        if (0..want.len()).all(|j| bufs[i + j].0 == want[j]) {
            groups.push((i, bufs[i..i + want.len()].iter().map(|(_, w)| w).collect()));
            i += want.len();
        } else {
            i += 1;
        }
    }
    groups
}

const GROUP_OFF_KC8_POSITION_MAJOR: usize = 2;

const GROUP_OFF_KC8T_TRANSPOSED: usize = 6;

fn assert_kt_is_a_bit_exact_permutation(
    gpu: &q3d::Qwen3_5DenseWgpu,
    cfg: &Qwen3_5DenseConfig,
    pos: usize,
) {
    let n_kv = cfg.num_key_value_heads;
    let hd = cfg.head_dim;
    let bufs = gpu.debug_state_buffer_words_for_test();
    let groups = tp_attn_state_groups(&bufs, n_kv, hd);
    let n_attn = cfg
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::FullAttention))
        .count();
    assert_eq!(
        groups.len(),
        n_attn,
        "expected one bf16+fp8+kt state group per full-attention layer"
    );
    for (gi, (_, g)) in groups.iter().enumerate() {
        let kc8 = as_bytes(g[GROUP_OFF_KC8_POSITION_MAJOR]);
        let kt = as_bytes(g[GROUP_OFF_KC8T_TRANSPOSED]);
        let live = pos * n_kv * hd;
        assert!(
            kc8[..live].iter().any(|b| *b != 0),
            "layer group {gi}: an all-zero fp8 K cache would make the permutation gate vacuous"
        );
        for slot in 0..pos {
            for kvh in 0..n_kv {
                for d4 in 0..hd / 4 {
                    let pm_off = (slot * n_kv + kvh) * hd + d4 * 4;
                    let kt_off = fd::k_transposed_word_index(MAX_SEQ, n_kv, hd, kvh, slot, d4) * 4;
                    assert_eq!(
                        &kc8[pm_off..pm_off + 4],
                        &kt[kt_off..kt_off + 4],
                        "layer group {gi} slot {slot} kvh {kvh} d4 {d4}: the kt cache must hold \
                         the same fp8 bytes at the transposed word for every chunk-prefilled and \
                         stepped position, nothing requantized"
                    );
                }
            }
        }
    }
}

#[test]
fn the_kt_cache_is_a_bit_exact_permutation_of_the_position_major_k_cache_after_chunked_prefill_and_decode(
) {
    let _g = env_lock();
    if !gpu_present_or_explicit_skip() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x7f8e_0001);
    let mut gpu = build(&cfg, &hw, true);
    let (pos, ids) = drive(&mut gpu);
    eprintln!("[kv-fp8-tp] {pos} positions landed, greedy tail {ids:?}");
    assert_kt_is_a_bit_exact_permutation(&gpu, &cfg, pos);
}

#[test]
fn nothing_moves_with_the_env_unset_and_tp_adds_exactly_one_state_buffer_per_attn_layer() {
    let _g = env_lock();
    if !gpu_present_or_explicit_skip() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x7f8e_0002);
    let n_attn = cfg
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::FullAttention))
        .count();
    let count = |tp: bool| build(&cfg, &hw, tp).debug_state_buffer_words_for_test().len();
    let base = count(false);
    assert_eq!(
        count(true),
        base + n_attn,
        "tp adds one transposed K state buffer per attention layer and nothing else"
    );
}

const TP_LOGIT_REL_STRUCTURAL_CEILING_REASSOC_ONLY: f32 = 5e-2;

#[test]
fn tp_greedy_stream_matches_the_fold_arm_and_its_logits_differ_only_by_reassociation() {
    let _g = env_lock();
    if !gpu_present_or_explicit_skip() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x7f8e_0003);
    let tokens = prompt();
    let run = |tp: bool| {
        let mut gpu = build(&cfg, &hw, tp);
        let labels = gpu.debug_step_pass_labels_for_test();
        let tp_stage1 = labels.iter().filter(|l| *l == "q3d-at-flash1-tp").count();
        let tp_writer = labels.iter().filter(|l| *l == "q3d-at-kvq-kt").count();
        let n_attn = cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, LayerType::FullAttention))
            .count();
        let want = if tp { n_attn } else { 0 };
        assert_eq!(
            (tp_stage1, tp_writer),
            (want, want),
            "tp={tp}: every full-attention layer must route stage1 to the transposed arm and \
             record the kt row writer, or the A/B silently measures the shipping arm"
        );
        let done = gpu.prefill_tokens(&tokens).expect("prefill_tokens");
        for t in &tokens[done..] {
            gpu.prefill_step(*t).expect("tail prefill step");
        }
        let mut ids = Vec::new();
        let mut logits = Vec::new();
        let mut next = 2u32;
        for _ in 0..GREEDY_STEPS {
            let (tok, row) = gpu.decode_step_logits(next).expect("greedy step");
            next = tok;
            ids.push(tok);
            logits.extend(row);
        }
        (ids, logits)
    };
    let (base_ids, base_logits) = run(false);
    let (tp_ids, tp_logits) = run(true);
    assert_eq!(
        base_ids, tp_ids,
        "the tp arm sums the same fp8 bytes under the same scales in a different order, so the \
         greedy stream must not move; a trip is a layout or scale-anchor bug, not tolerance"
    );
    assert_eq!(base_logits.len(), tp_logits.len());
    let mut max_rel = 0.0f32;
    for (a, b) in base_logits.iter().zip(&tp_logits) {
        let rel = (a - b).abs() / a.abs().max(b.abs()).max(1e-6);
        max_rel = max_rel.max(rel);
        assert!(
            rel <= TP_LOGIT_REL_STRUCTURAL_CEILING_REASSOC_ONLY,
            "logit {a:e} vs {b:e} beyond the structural ceiling: reassociation drift compounds \
             in ulps, not percent, so this is a wrong-position-set or wrong-scale bug"
        );
    }
    let (tp_ids2, tp_logits2) = run(true);
    assert_eq!(
        (tp_ids, tp_logits2 == tp_logits),
        (tp_ids2, true),
        "tp must be build-deterministic"
    );
    eprintln!("[kv-fp8-tp] greedy tails match; max logit rel drift {max_rel:e}");
}

const CHUNKED_PROMPT_TOKENS: usize = 32;

const CROSSOVER_MID_DECODE_TOTALS_RUN_40_TO_47: u32 = 44;

struct Stream {
    ids: Vec<u32>,
    step_logit_bits: Vec<Vec<u32>>,
    tail_prefill_flags: Vec<Option<bool>>,
    decode_flags: Vec<Option<bool>>,
}

fn drive_chunks_then_stepped_tail_then_greedy(gpu: &mut q3d::Qwen3_5DenseWgpu) -> Stream {
    let tokens = prompt();
    let done = gpu
        .prefill_tokens(&tokens[..CHUNKED_PROMPT_TOKENS])
        .expect("prefill_tokens");
    assert_eq!(
        done, CHUNKED_PROMPT_TOKENS,
        "the chunked prefix must land whole so every run splits chunk-vs-step identically"
    );
    let mut tail_prefill_flags = Vec::new();
    for t in &tokens[done..] {
        gpu.prefill_step(*t).expect("tail prefill step");
        tail_prefill_flags.push(gpu.debug_last_step_stage1_engaged_tp_for_test());
    }
    let mut ids = Vec::new();
    let mut step_logit_bits = Vec::new();
    let mut decode_flags = Vec::new();
    let mut next = 2u32;
    for _ in 0..GREEDY_STEPS {
        let (tok, row) = gpu.decode_step_logits(next).expect("greedy step");
        next = tok;
        ids.push(tok);
        step_logit_bits.push(row.iter().map(|v| v.to_bits()).collect());
        decode_flags.push(gpu.debug_last_step_stage1_engaged_tp_for_test());
    }
    Stream {
        ids,
        step_logit_bits,
        tail_prefill_flags,
        decode_flags,
    }
}

#[test]
fn a_knee_below_every_total_is_bit_for_bit_the_tp_arm() {
    let _g = env_lock();
    if !gpu_present_or_explicit_skip() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x7f8e_0004);
    let tp = drive_chunks_then_stepped_tail_then_greedy(&mut build(&cfg, &hw, true));
    let gated = drive_chunks_then_stepped_tail_then_greedy(&mut build_gated(&cfg, &hw, 1));
    assert_eq!(
        (gated.ids.clone(), gated.step_logit_bits == tp.step_logit_bits),
        (tp.ids.clone(), true),
        "with the knee at 1 every total engages tp, so the gated build must be the tp arm to \
         the bit; drift means the depth gate is not pure dispatch selection"
    );
    for s in [&tp, &gated] {
        assert!(
            s.decode_flags.iter().all(|f| *f == Some(true))
                && s.tail_prefill_flags.iter().all(|f| *f == Some(true)),
            "every stepped total is above this knee, so every step (prefill tail included) must \
             report the tp arm"
        );
    }
}

#[test]
fn a_knee_above_every_total_is_bit_for_bit_the_position_major_arm_and_the_kt_cache_stays_maintained(
) {
    let _g = env_lock();
    if !gpu_present_or_explicit_skip() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x7f8e_0005);
    let fold = drive_chunks_then_stepped_tail_then_greedy(&mut build(&cfg, &hw, false));
    let mut g = build_gated(&cfg, &hw, MAX_SEQ as u32 + 1);
    let gated = drive_chunks_then_stepped_tail_then_greedy(&mut g);
    assert_eq!(
        (gated.ids.clone(), gated.step_logit_bits == fold.step_logit_bits),
        (fold.ids.clone(), true),
        "below the knee the gated build must be the position-major arm to the bit; drift means \
         the off tp dispatch still ran or the fold twin binds the wrong cache"
    );
    assert!(
        gated.decode_flags.iter().all(|f| *f == Some(false))
            && gated.tail_prefill_flags.iter().all(|f| *f == Some(false)),
        "every stepped total is below this knee, so every step must report the position-major arm"
    );
    assert!(
        fold.decode_flags.iter().all(|f| f.is_none()),
        "the NV_Q3D_KV_FP8_TP=0 build records a single arm and must report no routing at all"
    );
    assert_kt_is_a_bit_exact_permutation(
        &g,
        &cfg,
        PROMPT_LEN_CROSSES_A_16_ROW_CHUNK_SO_BOTH_KT_WRITERS_RUN + GREEDY_STEPS,
    );
}

#[test]
fn crossing_the_knee_mid_stream_switches_arms_per_step_and_only_reassociation_separates_the_logits(
) {
    let _g = env_lock();
    if !gpu_present_or_explicit_skip() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x7f8e_0003);
    let n_attn = cfg
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::FullAttention))
        .count();
    let knee = CROSSOVER_MID_DECODE_TOTALS_RUN_40_TO_47;
    let fold = drive_chunks_then_stepped_tail_then_greedy(&mut build(&cfg, &hw, false));
    let tp = drive_chunks_then_stepped_tail_then_greedy(&mut build(&cfg, &hw, true));
    let mut g = build_gated(&cfg, &hw, knee);
    let count_arms = |labels: &[String]| {
        (
            labels.iter().filter(|l| *l == "q3d-at-flash1").count(),
            labels.iter().filter(|l| *l == "q3d-at-flash1-tp").count(),
        )
    };
    assert_eq!(
        count_arms(&g.debug_step_pass_labels_for_test()),
        (n_attn, n_attn),
        "both stage1 arms must stay recorded per full-attention layer; the gate selects at \
         encode and cannot select what was never recorded"
    );
    assert_eq!(
        (
            count_arms(&g.debug_step_pass_labels_at_total_for_test(knee - 1)),
            count_arms(&g.debug_step_pass_labels_at_total_for_test(knee)),
        ),
        ((n_attn, 0), (0, n_attn)),
        "one total below the knee must encode only the position-major arm and the knee itself \
         only the tp arm, in every full-attention layer"
    );
    let gated = drive_chunks_then_stepped_tail_then_greedy(&mut g);
    assert_eq!(
        (gated.ids.clone(), &fold.ids == &tp.ids),
        (fold.ids.clone(), true),
        "both pure arms produce this greedy stream, so the arm-switching build must too"
    );
    assert!(
        gated.tail_prefill_flags.iter().all(|f| *f == Some(false)),
        "the stepped prefill tail runs at totals 33..=39, all below this knee"
    );
    let first_decode_total = (PROMPT_LEN_CROSSES_A_16_ROW_CHUNK_SO_BOTH_KT_WRITERS_RUN + 1) as u32;
    for (i, bits) in gated.step_logit_bits.iter().enumerate() {
        let total = first_decode_total + i as u32;
        assert_eq!(
            gated.decode_flags[i],
            Some(total >= knee),
            "decode step at total {total} ran the wrong arm"
        );
        if total < knee {
            assert_eq!(
                bits, &fold.step_logit_bits[i],
                "below the knee logits must match the position-major arm bit for bit: no tp \
                 dispatch has run yet, so the runs are byte-identical"
            );
        } else {
            for (arm_name, arm) in [("position-major", &fold), ("tp", &tp)] {
                for (gb, ab) in bits.iter().zip(&arm.step_logit_bits[i]) {
                    let (a, b) = (f32::from_bits(*gb), f32::from_bits(*ab));
                    let rel = (a - b).abs() / a.abs().max(b.abs()).max(1e-6);
                    assert!(
                        rel <= TP_LOGIT_REL_STRUCTURAL_CEILING_REASSOC_ONLY,
                        "total {total} vs the {arm_name} arm: {a:e} vs {b:e} beyond the pinned \
                         reassociation ceiling. Bitwise equality is not available above the \
                         knee: the KV bytes and recurrent states this step reads were produced \
                         under the other arm's reassociated attention outputs, so only \
                         reassociation-class drift may separate the streams"
                    );
                }
            }
        }
    }
}

#[test]
fn the_unset_default_knee_is_the_reanchored_constant_and_selects_fold_below_tp_at_or_above() {
    let _g = env_lock();
    if !gpu_present_or_explicit_skip() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x7f8e_0006);
    std::env::set_var(ARM_ENV, "1");
    std::env::remove_var(KNEE_ENV);
    let g = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, MAX_SEQ)
        .expect("build with the unset-default knee");
    std::env::remove_var(ARM_ENV);
    let knee = q3d::TP_ENGAGES_PAST_THE_MEASURED_CROSSOVER_DEPTH_REANCHORED_BY_THE_Q38_TPKNEE_ROWS;
    let n_attn = cfg
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::FullAttention))
        .count();
    let count_arms = |labels: &[String]| {
        (
            labels.iter().filter(|l| *l == "q3d-at-flash1").count(),
            labels.iter().filter(|l| *l == "q3d-at-flash1-tp").count(),
        )
    };
    assert_eq!(
        (
            count_arms(&g.debug_step_pass_labels_at_total_for_test(knee - 1)),
            count_arms(&g.debug_step_pass_labels_at_total_for_test(knee)),
        ),
        ((n_attn, 0), (0, n_attn)),
        "with the env unset the depth gate must sit exactly at the q38 tpknee measured \
         crossover: position-major stage1 one below it, tp stage1 at it, in every \
         full-attention layer"
    );
}
