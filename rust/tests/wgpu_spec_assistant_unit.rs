#[path = "../src/oapi/chat_engine_wgpu/spec.rs"]
#[allow(dead_code)]
mod spec;

use spec::{ChainVerifyTarget, SpecKnobs, SpecLoop};

fn mix(mut h: u64, t: u32) -> u64 {
    h = h.wrapping_add(0x9e3779b97f4a7c15).wrapping_add(t as u64);
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
    h ^ (h >> 31)
}

fn hash_model(seed: u64, vocab: u32) -> impl Fn(&[u32]) -> u32 + Copy {
    move |ctx: &[u32]| {
        let mut h = seed;
        for &t in ctx {
            h = mix(h, t);
        }
        (h % vocab as u64) as u32
    }
}

struct FnTarget<F: Fn(&[u32]) -> u32> {
    ctx: Vec<u32>,
    scratch: Vec<u32>,
    f: F,
    cap: usize,
}

impl<F: Fn(&[u32]) -> u32> FnTarget<F> {
    fn new(f: F, cap: usize) -> Self {
        Self {
            ctx: Vec::new(),
            scratch: Vec::new(),
            f,
            cap,
        }
    }

    fn prefill(&mut self, prompt: &[u32]) -> u32 {
        let mut last = 0;
        for &t in prompt {
            let amax = self.verify_chain(&[t]).unwrap();
            self.advance(1).unwrap();
            last = amax[0];
        }
        last
    }
}

impl<F: Fn(&[u32]) -> u32> ChainVerifyTarget for FnTarget<F> {
    fn verify_chain(&mut self, batch: &[u32]) -> anyhow::Result<Vec<u32>> {
        anyhow::ensure!(!batch.is_empty() && batch.len() <= self.cap);
        self.scratch = batch.to_vec();
        let mut probe = self.ctx.clone();
        let mut amax = Vec::with_capacity(batch.len());
        for &t in batch {
            probe.push(t);
            amax.push((self.f)(&probe));
        }
        Ok(amax)
    }

    fn advance(&mut self, n: usize) -> anyhow::Result<()> {
        anyhow::ensure!(n >= 1 && n <= self.scratch.len());
        self.ctx.extend_from_slice(&self.scratch[..n]);
        self.scratch.clear();
        Ok(())
    }

    fn capacity(&self) -> usize {
        self.cap
    }
}

fn greedy_stream<F: Fn(&[u32]) -> u32>(f: &F, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut seq = prompt.to_vec();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let t = f(&seq);
        out.push(t);
        seq.push(t);
    }
    out
}

fn injected_draft<F: Fn(&[u32]) -> u32>(
    f: &F,
    ctx: &[u32],
    bonus: u32,
    len: usize,
    good_prefix: usize,
) -> Vec<u32> {
    let mut probe = ctx.to_vec();
    probe.push(bonus);
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let good = f(&probe);
        let t = if i < good_prefix { good } else { good ^ 1 };
        out.push(t);
        probe.push(t);
    }
    out
}

#[test]
fn round_with_draft_is_lossless_for_arbitrary_drafts() {
    let vocab = 64u32;
    for seed in 0..24u64 {
        let f = hash_model(seed.wrapping_mul(77) | 5, vocab);
        let prompt: Vec<u32> = (0..12)
            .map(|i| (mix(seed, i) % vocab as u64) as u32)
            .collect();
        let n = 96usize;
        let reference = greedy_stream(&f, &prompt, n);

        let mut target = FnTarget::new(f, 9);
        let mut bonus = target.prefill(&prompt);
        let mut sl = SpecLoop::new(SpecKnobs::parse(Some("1"), Some("8"), None));
        let mut out: Vec<u32> = vec![bonus];
        let mut round = 0u64;
        while out.len() < n {
            let good = (mix(seed, round as u32 + 9000) % 9) as usize;
            let len = (mix(seed, round as u32 + 5000) % 9)
                .max(good as u64)
                .min(target.capacity() as u64 - 1) as usize;
            let draft = injected_draft(&f, &target.ctx, bonus, len, good);
            let emitted = sl.round_with_draft(&mut target, bonus, draft).unwrap();
            assert!(!emitted.is_empty());
            out.extend_from_slice(&emitted);
            bonus = *emitted.last().unwrap();
            round += 1;
        }
        out.truncate(n);
        assert_eq!(
            out,
            reference[..out.len()],
            "seed {seed}: spec stream diverged from greedy"
        );
        let stats = sl.stats();
        assert_eq!(stats.emitted, stats.rounds + stats.accepted);
    }
}

#[test]
fn round_with_draft_truncates_to_capacity() {
    let f = hash_model(3, 32);
    let mut target = FnTarget::new(f, 4);
    let bonus = target.prefill(&[1, 2, 3]);
    let mut sl = SpecLoop::new(SpecKnobs::parse(Some("1"), Some("8"), None));
    let draft = vec![9u32; 20];
    let emitted = sl.round_with_draft(&mut target, bonus, draft).unwrap();
    assert!(!emitted.is_empty() && emitted.len() <= 4);
    assert_eq!(sl.stats().drafted, 3);
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn wgpu_spec_assistant_unit_wgpu_gated_mod_is_cfg_out_this_is_a_skip_not_a_pass() {
    eprintln!(
        "mod wgpu_gated compiled OUT (no `wgpu` feature): 4 tests vanished while the 6 ungated \
         spec-math tests in this file still reported a plausible pass count, so the cfg-out \
         left no trace in the summary line at all. Re-run with NVK_PKG=speaches-plus \
         NVK_FEATURES=cuda,wgpu."
    );
}

#[cfg(feature = "wgpu")]
mod wgpu_gated {
    use nv_kernels::wgpu_backend::kernels::kv_fp8::{div_rn, encode_e4m3, FP8_E4M3_MAX};
    use speaches_plus::oapi::chat_engine_wgpu::learned::dequant_kv_rows;

    fn synth_value(slot: usize, head: usize, d: usize) -> f32 {
        ((slot * 31 + head * 7 + d) as f32 * 0.37 + 0.11).sin() * (1.0 + head as f32)
    }

    fn build_cache(slots: usize, n_kv: usize, hd: usize) -> (Vec<u32>, Vec<f32>) {
        let mut fp8 = vec![0u32; slots * n_kv * hd / 4];
        let mut scales = vec![0f32; slots * n_kv];
        for slot in 0..slots {
            for h in 0..n_kv {
                let vals: Vec<f32> = (0..hd).map(|d| synth_value(slot, h, d)).collect();
                let amax = vals.iter().fold(0f32, |a, v| a.max(v.abs()));
                let (scale, inv) = if amax > 0.0 {
                    (div_rn(amax, FP8_E4M3_MAX), div_rn(FP8_E4M3_MAX, amax))
                } else {
                    (1.0, 1.0)
                };
                scales[slot * n_kv + h] = scale;
                for (d, v) in vals.iter().enumerate() {
                    let idx = (slot * n_kv + h) * hd + d;
                    fp8[idx / 4] |= (encode_e4m3(v * inv) as u32) << (8 * (idx % 4));
                }
            }
        }
        (fp8, scales)
    }

    #[test]
    fn dequant_kv_rows_matches_slot_layout() {
        let (slots, n_kv, hd) = (6usize, 2usize, 8usize);
        let (fp8, scales) = build_cache(slots, n_kv, hd);
        let t = dequant_kv_rows(&fp8, &scales, n_kv, hd, 1, 4).expect("dequant");
        assert_eq!(t.dims(), &[n_kv, 4, hd]);
        let host: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        for h in 0..n_kv {
            for i in 0..4 {
                let slot = 1 + i;
                for d in 0..hd {
                    let got = host[(h * 4 + i) * hd + d];
                    let want = synth_value(slot, h, d);
                    let tol = want.abs().max(0.05) * 0.08;
                    assert!(
                        (got - want).abs() <= tol,
                        "kv[{h}][{i}][{d}]: got {got}, want ~{want}"
                    );
                }
            }
        }
    }

    fn words_per_slot_of(n_kv: usize, hd: usize) -> usize {
        n_kv * hd / 4
    }

    #[test]
    fn kv_mirror_incremental_append_matches_full_dequant() {
        use speaches_plus::oapi::chat_engine_wgpu::learned::KvMirror;
        let (slots, n_kv, hd) = (10usize, 2usize, 8usize);
        let (k_fp8, k_scales) = build_cache(slots, n_kv, hd);
        let (mut v_fp8, mut v_scales) = build_cache(slots, n_kv, hd);
        v_fp8.rotate_left(words_per_slot_of(n_kv, hd));
        v_scales.rotate_left(n_kv);
        let words_per_slot = n_kv * hd / 4;
        let mut mirror = KvMirror::new(n_kv, hd);
        for (start, len) in [(0usize, 3usize), (3, 1), (4, 6)] {
            let kf = &k_fp8[start * words_per_slot..(start + len) * words_per_slot];
            let vf = &v_fp8[start * words_per_slot..(start + len) * words_per_slot];
            let ks = &k_scales[start * n_kv..(start + len) * n_kv];
            let vs = &v_scales[start * n_kv..(start + len) * n_kv];
            mirror.append(kf, vf, ks, vs, len).expect("append");
            assert_eq!(mirror.len(), start + len);
        }
        for (start, len) in [(0usize, 10usize), (2, 5), (9, 1)] {
            let (mk, mv) = mirror.tensors(start, len).expect("tensors");
            let fk = dequant_kv_rows(&k_fp8, &k_scales, n_kv, hd, start, len).expect("dequant k");
            let fv = dequant_kv_rows(&v_fp8, &v_scales, n_kv, hd, start, len).expect("dequant v");
            assert_eq!(mk.dims(), fk.dims());
            let a: Vec<f32> = mk.flatten_all().unwrap().to_vec1().unwrap();
            let b: Vec<f32> = fk.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(a, b, "K mirror diverged at view ({start},{len})");
            let a: Vec<f32> = mv.flatten_all().unwrap().to_vec1().unwrap();
            let b: Vec<f32> = fv.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(a, b, "V mirror diverged at view ({start},{len})");
        }
        mirror.clear();
        assert!(mirror.is_empty());
        assert!(mirror.tensors(0, 1).is_err());
    }

    #[test]
    fn kv_mirror_rejects_short_buffers() {
        use speaches_plus::oapi::chat_engine_wgpu::learned::KvMirror;
        let (k_fp8, k_scales) = build_cache(2, 2, 8);
        let mut mirror = KvMirror::new(2, 8);
        assert!(mirror
            .append(&k_fp8[..1], &k_fp8, &k_scales, &k_scales, 2)
            .is_err());
        assert!(mirror
            .append(&k_fp8, &k_fp8, &k_scales[..1], &k_scales, 2)
            .is_err());
        assert_eq!(mirror.len(), 0);
        mirror
            .append(&k_fp8, &k_fp8, &k_scales, &k_scales, 2)
            .unwrap();
        assert!(mirror.tensors(1, 2).is_err());
    }

    #[test]
    fn dequant_kv_rows_rejects_short_buffers() {
        let (fp8, scales) = build_cache(2, 2, 8);
        assert!(dequant_kv_rows(&fp8, &scales, 2, 8, 1, 4).is_err());
        assert!(dequant_kv_rows(&fp8, &scales[..2], 2, 8, 0, 2).is_err());
    }

    #[test]
    #[ignore]
    fn wgpu_forward_matches_host_forward_on_real_checkpoints() {
        use nv_models::gemma4::Gemma4Config;
        use nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu;
        use speaches_plus::oapi::chat_engine_wgpu::learned::AssistantSpecDrafter;

        let Ok(model_dir) =
            std::env::var("NV_WGPU_SERVE_DIR").or_else(|_| std::env::var("NV_E4B_DIR"))
        else {
            panic!(
                "this test is #[ignore]d, so it was asked for BY NAME, but neither \
                 NV_WGPU_SERVE_DIR nor NV_E4B_DIR names the E4B checkpoint. Returning here \
                 prints `1 passed` in 0.00s having compared nothing. This is a SKIP, not a pass."
            )
        };
        let model_dir = std::path::PathBuf::from(model_dir);
        let home = std::env::var("HOME").unwrap();
        let snaps = std::path::PathBuf::from(home).join(
            ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-q4_0-unquantized-assistant/snapshots",
        );
        let assistant_dir = std::fs::read_dir(&snaps)
            .ok()
            .and_then(|mut it| it.next())
            .and_then(|e| e.ok())
            .map(|e| e.path());
        let Some(assistant_dir) = assistant_dir else {
            panic!(
                "the E4B target checkpoint was named but \
                 google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant is not in the HF cache, so \
                 the drafter half of this comparison cannot run. This is a SKIP, not a pass."
            )
        };
        let _ = tracing_subscriber::fmt()
            .with_env_filter("speaches_plus=debug")
            .with_writer(std::io::stderr)
            .try_init();
        let raw = std::fs::read_to_string(model_dir.join("config.json")).expect("config");
        let cfg = Gemma4Config::from_hf_json_str(&raw).expect("target config");
        let weights = nv_weights::WeightLoader::open_dir(&model_dir, &candle_core::Device::Cpu)
            .expect("open target weights");
        let mut model =
            Gemma4E4bWgpu::from_loader(cfg.clone(), &weights, 256).expect("build wgpu model");

        let prompt: Vec<u32> = vec![
            2, 651, 892, 1204, 7, 3021, 118, 42, 9077, 65, 331, 4, 88, 2077,
        ];
        let mut last = 0u32;
        for &t in &prompt {
            last = model.decode_step(t).expect("prefill step");
        }
        let amax = model.verify_chain(&[last]).expect("verify round");
        model.advance(1).expect("advance");
        let hidden = model.verify_hidden_row(0).expect("hidden row");
        let bonus = amax[0];

        let mut host =
            AssistantSpecDrafter::load(&model_dir, &assistant_dir, &cfg).expect("host drafter");
        let mut gpu =
            AssistantSpecDrafter::load(&model_dir, &assistant_dir, &cfg).expect("gpu drafter");
        gpu.attach_gpu(&model);
        assert!(gpu.gpu_active(), "wgpu drafter forward did not come up");

        for k in [1usize, 4, 8] {
            let a = host
                .propose(&model, bonus, &hidden, k)
                .expect("host propose");
            let b = gpu.propose(&model, bonus, &hidden, k).expect("gpu propose");
            eprintln!("k={k}: host {a:?} vs wgpu {b:?}");
            assert_eq!(a, b, "k={k}: wgpu forward diverged from host forward");
        }
    }

    #[test]
    #[ignore]
    fn decode_hidden_matches_verify_hidden() {
        use nv_models::gemma4::Gemma4Config;
        use nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu;

        let Ok(model_dir) =
            std::env::var("NV_WGPU_SERVE_DIR").or_else(|_| std::env::var("NV_E4B_DIR"))
        else {
            panic!(
                "this test is #[ignore]d, so it was asked for BY NAME, but neither \
                 NV_WGPU_SERVE_DIR nor NV_E4B_DIR names the E4B checkpoint. Returning here \
                 prints `1 passed` in 0.00s having compared nothing. This is a SKIP, not a pass."
            )
        };
        let model_dir = std::path::PathBuf::from(model_dir);
        let raw = std::fs::read_to_string(model_dir.join("config.json")).expect("config");
        let cfg = Gemma4Config::from_hf_json_str(&raw).expect("target config");
        let weights = nv_weights::WeightLoader::open_dir(&model_dir, &candle_core::Device::Cpu)
            .expect("open target weights");
        let mut model = Gemma4E4bWgpu::from_loader(cfg, &weights, 256).expect("build wgpu model");
        let mut last = 2u32;
        for i in 0..40u32 {
            last = model
                .decode_step(((i * 37) % 200000) + 5)
                .expect("prefill step");
        }
        let vt = model.verify_chain(&[last]).expect("verify");
        let vh = model.verify_hidden_row(0).expect("verify hidden");
        let dt = model.decode_step(last).expect("decode");
        let dh = model.decode_hidden_row().expect("decode hidden");
        assert_eq!(vt[0], dt, "verify and decode disagree on the next token");
        assert_eq!(vh.len(), dh.len());
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        for (a, b) in vh.iter().zip(dh.iter()) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / a.abs().max(1e-3));
        }
        eprintln!("[hidden] max_abs {max_abs:.4} max_rel {max_rel:.4}");
        assert!(
            max_rel < 0.05,
            "decode hidden diverges from verify hidden: max_abs {max_abs} max_rel {max_rel}"
        );
    }

    #[test]
    #[ignore]
    fn wgpu_drafter_round_cost_breakdown() {
        use nv_models::gemma4::Gemma4Config;
        use nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu;
        use speaches_plus::oapi::chat_engine_wgpu::learned::AssistantSpecDrafter;

        let Ok(model_dir) =
            std::env::var("NV_WGPU_SERVE_DIR").or_else(|_| std::env::var("NV_E4B_DIR"))
        else {
            panic!(
                "this test is #[ignore]d, so it was asked for BY NAME, but neither \
                 NV_WGPU_SERVE_DIR nor NV_E4B_DIR names the E4B checkpoint. Returning here \
                 prints `1 passed` in 0.00s having compared nothing. This is a SKIP, not a pass."
            )
        };
        let model_dir = std::path::PathBuf::from(model_dir);
        let home = std::env::var("HOME").unwrap();
        let snaps = std::path::PathBuf::from(home).join(
            ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-q4_0-unquantized-assistant/snapshots",
        );
        let Some(assistant_dir) = std::fs::read_dir(&snaps)
            .ok()
            .and_then(|mut it| it.next())
            .and_then(|e| e.ok())
            .map(|e| e.path())
        else {
            panic!(
                "the E4B target checkpoint was named but \
                 google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant is not in the HF cache, so \
                 the drafter half of this comparison cannot run. This is a SKIP, not a pass."
            )
        };
        let raw = std::fs::read_to_string(model_dir.join("config.json")).expect("config");
        let cfg = Gemma4Config::from_hf_json_str(&raw).expect("target config");
        let weights = nv_weights::WeightLoader::open_dir(&model_dir, &candle_core::Device::Cpu)
            .expect("open target weights");
        let mut model =
            Gemma4E4bWgpu::from_loader(cfg.clone(), &weights, 1024).expect("build wgpu model");

        let mut last = 2u32;
        for i in 0..30u32 {
            last = model
                .decode_step(((i * 37) % 200000) + 5)
                .expect("prefill step");
        }
        let reps = 50;
        let vrows = model.verify_max_rows().min(9);
        let batch = vec![last; vrows];
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            model.verify_chain(&batch).expect("verify");
        }
        eprintln!(
            "[bench] verify m={vrows} at ctx 31: {:.2} ms",
            t0.elapsed().as_secs_f64() * 1000.0 / reps as f64
        );
        for i in 30..450u32 {
            last = model
                .decode_step(((i * 37) % 200000) + 5)
                .expect("prefill step");
        }
        let amax = model.verify_chain(&[last]).expect("verify round");
        model.advance(1).expect("advance");
        let hidden = model.verify_hidden_row(0).expect("hidden row");
        let bonus = amax[0];

        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            model.decode_step(bonus).expect("decode");
            model.truncate_to(model.current_pos() - 1).expect("rewind");
        }
        let decode_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

        let m = model.prefill_chunk_len();
        let chunk = vec![bonus; m];
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            model.prefill_chunk(&chunk).expect("prefill chunk");
            model.sync().expect("sync");
            model.truncate_to(model.current_pos() - m).expect("rewind");
        }
        eprintln!(
            "[bench] prefill_chunk m={m} at ctx 451: {:.2} ms",
            t0.elapsed().as_secs_f64() * 1000.0 / reps as f64
        );

        let batch = vec![bonus; vrows];
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            model.verify_chain(&batch).expect("verify");
        }
        let verify_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            model.verify_hidden_row(0).expect("hidden");
        }
        let hidden_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

        let mut gpu =
            AssistantSpecDrafter::load(&model_dir, &assistant_dir, &cfg).expect("gpu drafter");
        gpu.attach_gpu(&model);
        assert!(gpu.gpu_active());
        let _ = gpu.propose(&model, bonus, &hidden, 8).expect("warmup");
        for k in [1usize, 4, 8] {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                gpu.propose(&model, bonus, &hidden, k).expect("propose");
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
            eprintln!("[bench] gpu propose k={k}: {ms:.2} ms/round");
        }
        eprintln!(
            "[bench] decode_step {decode_ms:.2} ms | verify m={vrows} {verify_ms:.2} ms | hidden_row {hidden_ms:.2} ms (ctx 451)"
        );
    }

    #[test]
    #[ignore]
    fn assistant_drafter_loads_and_proposes_on_real_checkpoints() {
        use candle_core::{DType, Device, Tensor};
        use nv_models::gemma4::Gemma4Config;
        use nv_specdecode::gemma4_assistant::FixedSharedKv;
        use speaches_plus::oapi::chat_engine_wgpu::learned::AssistantSpecDrafter;

        let Ok(model_dir) =
            std::env::var("NV_WGPU_SERVE_DIR").or_else(|_| std::env::var("NV_E4B_DIR"))
        else {
            panic!(
                "this test is #[ignore]d, so it was asked for BY NAME, but neither \
                 NV_WGPU_SERVE_DIR nor NV_E4B_DIR names the E4B checkpoint. Returning here \
                 prints `1 passed` in 0.00s having compared nothing. This is a SKIP, not a pass."
            )
        };
        let model_dir = std::path::PathBuf::from(model_dir);
        let home = std::env::var("HOME").unwrap();
        let snaps = std::path::PathBuf::from(home).join(
            ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-q4_0-unquantized-assistant/snapshots",
        );
        let assistant_dir = std::fs::read_dir(&snaps)
            .ok()
            .and_then(|mut it| it.next())
            .and_then(|e| e.ok())
            .map(|e| e.path());
        let Some(assistant_dir) = assistant_dir else {
            panic!(
                "the E4B target checkpoint was named but \
                 google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant is not in the HF cache, so \
                 the drafter half of this comparison cannot run. This is a SKIP, not a pass."
            )
        };
        let cfg =
            Gemma4Config::from_hf_json_file(&model_dir.join("config.json")).expect("target config");
        let d = AssistantSpecDrafter::load(&model_dir, &assistant_dir, &cfg).expect("load drafter");
        assert_eq!(d.sliding_layer(), 22);
        assert_eq!(d.full_layer(), 23);

        let row = d.embed_scaled(2).expect("embed row");
        assert_eq!(row.dims(), &[cfg.hidden_size]);
        let vals: Vec<f32> = row.to_vec1().unwrap();
        assert!(vals.iter().all(|x| x.is_finite()));
        assert!(vals.iter().any(|x| *x != 0.0));

        let kv_len = 16usize;
        let dev = Device::Cpu;
        let mk = |n_kv: usize, hd: usize, seed: f32| -> Tensor {
            let data: Vec<f32> = (0..n_kv * kv_len * hd)
                .map(|i| ((i as f32) * 0.011 + seed).sin() * 0.2)
                .collect();
            Tensor::from_vec(data, (n_kv, kv_len, hd), &dev).unwrap()
        };
        let kv = FixedSharedKv {
            sliding: (mk(2, 256, 0.3), mk(2, 256, 1.1)),
            full: (mk(2, 512, 2.2), mk(2, 512, 3.3)),
        };
        let hidden: Vec<f32> = (0..cfg.hidden_size)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let hidden_t = Tensor::from_vec(hidden.clone(), cfg.hidden_size, &dev)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        drop(hidden_t);
        let out = d
            .propose_from_parts(2, &hidden, kv_len, 4, &kv)
            .expect("propose");
        assert!(!out.is_empty() && out.len() <= 4);
        for &t in &out {
            assert!((t as usize) < cfg.vocab_size);
        }
        eprintln!("assistant chain draft: {out:?}");
    }
}
