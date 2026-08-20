#![cfg(feature = "wgpu")]

mod common;
use common::prompt_for;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::gemv_bf16::{I8_NORMED_ENTRY, ROWS_PER_GROUP};
use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_wgpu::{
    mk_i8_lmhead_shader_source, Gemma4Wgpu, HostBf16Lin, HostLayer, HostProj, HostWeights,
    LMHEAD_I8_MK_PK_ENTRY, MK_MAX,
};
use common::LcgShift32Centered0p1I8 as Lcg;

fn ctx_or_panic() -> &'static nv_kernels::wgpu_backend::WgpuContext {
    if std::env::var("NV_G4BD").as_deref() != Ok("1") {
        panic!("set NV_G4BD=1 to run this GPU test (it must never silently skip)");
    }
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("[g4lmi8] adapter: {}", ctx.summary());
            ctx
        }
        Err(e) => panic!("the int8 lm_head M-row twin gate needs a wgpu adapter: {e}"),
    }
}

fn pack_u16(v: &[u16]) -> Vec<u32> {
    v.chunks(2)
        .map(|c| (c[0] as u32) | ((c.get(1).copied().unwrap_or(0) as u32) << 16))
        .collect()
}

fn pack_i8(v: &[i8]) -> Vec<u32> {
    v.chunks(4)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u32, |w, (i, b)| w | (((*b as u8) as u32) << (8 * i)))
        })
        .collect()
}

fn pack_lo16_like_the_pack16_pass(unpacked: &[u32]) -> Vec<u32> {
    unpacked
        .chunks(2)
        .map(|c| (c[0] & 0xffff) | ((c[1] & 0xffff) << 16))
        .collect()
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvI8Params {
    n_rows: u32,
    k_elems: u32,
    wq_row_words: u32,
    groups_x: u32,
    m_rows: u32,
    x_row_words: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MkParams {
    m: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    dst_word_off: u32,
}

#[test]
fn lmhead_i8_mk_twin_source_validates_for_every_slot_count_without_a_gpu() {
    for m in 1..=MK_MAX {
        let src = mk_i8_lmhead_shader_source(m);
        assert!(
            src.contains(&format!("fn {LMHEAD_I8_MK_PK_ENTRY}(")),
            "mk_i8_lmhead_shader_source({m}) no longer emits {LMHEAD_I8_MK_PK_ENTRY}; the \
             runtime pipeline build would fail the same way, but this check needs no adapter"
        );
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("m={m}: wgsl parse: {}", e.message()));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("m={m}: naga validate (uniformity included): {e:?}"));
    }
}

#[test]
fn lmhead_i8_mk_twin_is_bit_identical_to_the_single_row_entry_plus_pack16() {
    let ctx = ctx_or_panic();
    let k = 1024usize;
    for &m in &[2usize, 3, 8, MK_MAX] {
        for &n in &[96usize, 70] {
            assert!(n % 2 == 0, "the packed store pairs rows, so n must be even");
            let mut rng = Lcg(0x51ed_c0de ^ ((m as u64) << 32) ^ n as u64);
            let wq = rng.i8_vec(n * k);
            let mut row_scale: Vec<f32> = (0..n).map(|_| 0.01 + rng.next_f32().abs()).collect();
            row_scale[0] = 0.0;
            row_scale[1] = f32::from_bits(0x0000_1234);
            let x_rows: Vec<Vec<u16>> = (0..m).map(|_| rng.bf16_vec(k)).collect();
            let wn = rng.bf16_vec_around_one(k);
            let rstd: Vec<f32> = (0..m).map(|_| 0.9 + rng.next_f32()).collect();

            let src = mk_i8_lmhead_shader_source(m);
            let groups = dispatch::workgroup_count_1d(ctx, n as u64, ROWS_PER_GROUP);
            let wq_buf = dispatch::storage_from_slice(ctx, "g4lmi8-wq", &pack_i8(&wq));
            let rs_buf = dispatch::storage_from_slice(ctx, "g4lmi8-rs", &row_scale);
            let wn_buf = dispatch::storage_from_slice(ctx, "g4lmi8-wn", &pack_u16(&wn));

            let mut solo_packed: Vec<Vec<u32>> = Vec::with_capacity(m);
            for j in 0..m {
                let x_buf = dispatch::storage_from_slice(ctx, "g4lmi8-x1", &pack_u16(&x_rows[j]));
                let rstd_buf = dispatch::storage_from_slice(ctx, "g4lmi8-rstd1", &[rstd[j]]);
                let y_buf = dispatch::storage_zeroed(ctx, "g4lmi8-y1", (n * 4) as u64);
                let p = dispatch::uniform_from(
                    ctx,
                    "g4lmi8-p1",
                    &GemvI8Params {
                        n_rows: n as u32,
                        k_elems: k as u32,
                        wq_row_words: (k / 4) as u32,
                        groups_x: groups.0,
                        m_rows: 1,
                        x_row_words: (k / 2) as u32,
                        pad0: 0,
                        pad1: 0,
                    },
                );
                dispatch::run(
                    ctx,
                    "g4lmi8-solo",
                    &src,
                    I8_NORMED_ENTRY,
                    &[
                        (13, &wq_buf),
                        (14, &rs_buf),
                        (15, &x_buf),
                        (16, &wn_buf),
                        (17, &rstd_buf),
                        (18, &y_buf),
                        (19, &p),
                    ],
                    groups,
                )
                .expect("single-row gemv_i8_normed dispatch");
                let un: Vec<u32> = dispatch::read_back(ctx, &y_buf, n).expect("read solo y");
                solo_packed.push(pack_lo16_like_the_pack16_pass(&un));
            }

            let x_all: Vec<u16> = x_rows.iter().flatten().copied().collect();
            let x_buf = dispatch::storage_from_slice(ctx, "g4lmi8-xm", &pack_u16(&x_all));
            let rstd_buf = dispatch::storage_from_slice(ctx, "g4lmi8-rstdm", &rstd);
            let y_stride = n;
            for &off in &[0usize, n / 2] {
                let y_buf =
                    dispatch::storage_zeroed(ctx, "g4lmi8-ym", (m * y_stride * 4) as u64);
                let p = dispatch::uniform_from(
                    ctx,
                    "g4lmi8-pm",
                    &GemvI8Params {
                        n_rows: n as u32,
                        k_elems: k as u32,
                        wq_row_words: (k / 4) as u32,
                        groups_x: groups.0,
                        m_rows: m as u32,
                        x_row_words: (k / 2) as u32,
                        pad0: 0,
                        pad1: 0,
                    },
                );
                let mkp = dispatch::uniform_from(
                    ctx,
                    "g4lmi8-mkp",
                    &MkParams {
                        m: m as u32,
                        x_stride_words: (k / 2) as u32,
                        y_stride_words: y_stride as u32,
                        dst_word_off: off as u32,
                    },
                );
                dispatch::run(
                    ctx,
                    "g4lmi8-mk",
                    &src,
                    LMHEAD_I8_MK_PK_ENTRY,
                    &[
                        (13, &wq_buf),
                        (14, &rs_buf),
                        (15, &x_buf),
                        (16, &wn_buf),
                        (17, &rstd_buf),
                        (18, &y_buf),
                        (19, &p),
                        (35, &mkp),
                    ],
                    groups,
                )
                .expect("M-row lm_head twin dispatch");
                let got: Vec<u32> =
                    dispatch::read_back(ctx, &y_buf, m * y_stride).expect("read twin y");
                for t in 0..m {
                    for w in 0..n / 2 {
                        assert_eq!(
                            got[off + t * y_stride + w],
                            solo_packed[t][w],
                            "m={m} n={n} dst_word_off={off} slot {t} word {w}: the twin's packed \
                             word differs from gemv_i8_normed followed by the pack16 pairing \
                             (lo16 of row 2w, hi16 of row 2w+1); the twin must keep the solo \
                             entry's accumulation order per slot"
                        );
                    }
                }
            }
            eprintln!(
                "[g4lmi8] m={m} n={n} k={k}: twin bit-identical to solo+pack16 at both offsets"
            );
        }
    }
}

fn config_json(layers: usize, hidden: usize, inter: usize, vocab: usize) -> String {
    let mut types = Vec::with_capacity(layers);
    for i in 0..layers {
        types.push(if (i + 1) % 3 == 0 {
            "\"full_attention\""
        } else {
            "\"sliding_attention\""
        });
    }
    format!(
        r#"{{
  "text_config": {{
    "hidden_size": {hidden},
    "intermediate_size": {inter},
    "num_hidden_layers": {layers},
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "head_dim": 128,
    "global_head_dim": 256,
    "vocab_size": {vocab},
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "sliding_window": 4096,
    "final_logit_softcapping": 0.0,
    "layer_types": [{}],
    "attention_k_eq_v": false,
    "hidden_activation": "gelu_pytorch_tanh",
    "num_kv_shared_layers": 0,
    "rope_parameters": {{
      "full_attention": {{"partial_rotary_factor": 0.25, "rope_theta": 1000000.0}},
      "sliding_attention": {{"rope_theta": 10000.0}}
    }}
  }},
  "tie_word_embeddings": true
}}"#,
        types.join(", ")
    )
}

fn host_weights(config: &Gemma4Config, seed: u64) -> HostWeights {
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
        let mk_proj = |rng: &mut Lcg, n: usize, k: usize| {
            HostProj::Bf16(HostBf16Lin {
                w: rng.bf16_vec(n * k),
                n,
                k,
            })
        };
        layers.push(HostLayer {
            kind,
            input_ln: rng.bf16_vec_around_one(hidden),
            post_attn_ln: rng.bf16_vec_around_one(hidden),
            pre_ff_ln: rng.bf16_vec_around_one(hidden),
            post_ff_ln: rng.bf16_vec_around_one(hidden),
            q_norm: rng.bf16_vec_around_one(hd),
            k_norm: rng.bf16_vec_around_one(hd),
            layer_scalar: 0.9,
            has_v,
            qkv: mk_proj(&mut rng, qkv_rows, hidden),
            o: mk_proj(&mut rng, hidden, q_dim),
            gate_up: mk_proj(&mut rng, 2 * inter, hidden),
            down: mk_proj(&mut rng, hidden, inter),
        });
    }
    HostWeights {
        embed: rng.bf16_vec(config.vocab_size * hidden),
        final_norm: rng.bf16_vec_around_one(hidden),
        layers,
    }
}

fn solo_logits(m: &mut Gemma4Wgpu, prompt: &[u32], steps: usize) -> (Vec<u32>, Vec<Vec<u32>>) {
    m.reset_slot(0).expect("reset slot");
    let (last, rest) = prompt.split_last().expect("prompt");
    let done = m.prefill_tokens(rest).expect("prefill_tokens");
    for t in &rest[done..] {
        m.prefill_step(*t).expect("prefill step");
    }
    let mut toks = Vec::with_capacity(steps);
    let mut bits = Vec::with_capacity(steps);
    let mut t = *last;
    for _ in 0..steps {
        let (n, lg) = m.decode_step_logits(t).expect("decode step logits");
        bits.push(lg.into_iter().map(f32::to_bits).collect::<Vec<u32>>());
        toks.push(n);
        t = n;
    }
    (toks, bits)
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

#[test]
fn batched_int8_lmhead_matches_solo_decode_bit_exactly() {
    ctx_or_panic();
    let layers = 2usize;
    let hidden = 512usize;
    let inter = 1024usize;
    let vocab = 2048usize;
    let slots = 3usize;
    let steps = 6usize;
    let max_seq = 128usize;

    let raw = config_json(layers, hidden, inter, vocab);
    let config = Gemma4Config::from_hf_json_str(&raw).expect("config");
    let w = host_weights(&config, 0x9e3779b9);

    let control_bits = {
        let mut control =
            Gemma4Wgpu::new(config.clone(), &w, max_seq).expect("build bf16-lm_head control");
        solo_logits(&mut control, &prompt_for(0, 17, vocab), 1).1
    };

    let pin = EnvPin::set("NV_WGPU_LMHEAD_INT8", "1");
    let mut single = Gemma4Wgpu::new(config.clone(), &w, max_seq).expect("build int8 single");
    let mut batched =
        Gemma4Wgpu::new_batched(config, &w, max_seq, slots).expect("build int8 batched");
    drop(w);
    assert_eq!(
        batched.batch_slots(),
        slots,
        "the int8 lm_head M-row twin must keep batch decode engaged; a [gemma4_wgpu] boot \
         disabler fired above and the rest of this test would be vacuous"
    );

    let prompts: Vec<Vec<u32>> = (0..slots)
        .map(|j| prompt_for(j, 17 + 3 * j, vocab))
        .collect();
    let solo: Vec<(Vec<u32>, Vec<Vec<u32>>)> = (0..slots)
        .map(|j| solo_logits(&mut single, &prompts[j], steps))
        .collect();
    assert_ne!(
        solo[0].1[0], control_bits[0],
        "int8 and bf16 lm_head logits are bitwise identical: NV_WGPU_LMHEAD_INT8 did not \
         engage and this whole comparison is the bf16 twin's, not the int8 twin's"
    );
    for j in 1..slots {
        assert!(
            (0..steps).any(|i| solo[j].1[i] != solo[0].1[i]),
            "slot {j}'s solo logits equal slot 0's at every step; the cross-slot compare \
             would be vacuous"
        );
    }

    let mut cur: Vec<u32> = Vec::with_capacity(slots);
    for (j, p) in prompts.iter().enumerate() {
        cur.push(batched.prefill_slot(j, p).expect("prefill slot"));
        assert_eq!(
            cur[j], solo[j].0[0],
            "slot {j}: prefill through the batched model already disagrees with the solo run"
        );
    }
    for i in 1..steps {
        let nx = batched.decode_step_batch(&cur).expect("decode_step_batch");
        let lg = batched.batch_logits().expect("batch logits");
        assert_eq!(lg.len(), slots * vocab);
        for j in 0..slots {
            let want = &solo[j].1[i];
            let got: Vec<u32> = lg[j * vocab..(j + 1) * vocab]
                .iter()
                .map(|x| x.to_bits())
                .collect();
            let diff = want.iter().zip(got.iter()).filter(|(x, y)| x != y).count();
            assert_eq!(
                diff, 0,
                "step {i} slot {j}: {diff} of {vocab} int8 lm_head logit lanes differ from \
                 the same sequence run alone through the single-row entry"
            );
            assert_eq!(
                nx[j], solo[j].0[i],
                "step {i} slot {j}: sampled token differs"
            );
        }
        cur = nx;
    }
    drop(pin);
    eprintln!(
        "[g4lmi8] B={slots}: int8 lm_head batch decode bit-identical to solo for {steps} \
         steps x {vocab} lanes"
    );
}
