#![cfg(any(feature = "cuda", feature = "metal"))]

mod common;
use common::cer;
use common::fixture_expected_text;
use common::fixture_rgb;
use common::FIXTURES;
use common::wer;
mod hub_snapshot;

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use nv_models::dots_ocr::{DotsMode, DotsOcrPipeline, GenerateOptions};

const DOTS_MEAN_CER_GATE_MEASURED_0_0736_BF16_0_1913_NVFP4_ON_070_071_CUDA_C0111CE_DEB48F3: f64 =
    0.10;

fn serving_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    #[cfg(feature = "metal")]
    if let Ok(d) = Device::new_metal(0) {
        return d;
    }
    Device::Cpu
}

fn snapshot_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("NV_DOTS_DIR") {
        let p = PathBuf::from(d);
        return p.is_dir().then_some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries.into_iter().next()
}

#[test]
#[ignore]
fn dots_ocr_baseline_speed_and_cer_over_070_071_fixtures() {
    if std::env::var("NV_DOTS_BENCH").as_deref() != Ok("1") {
        hub_snapshot::precondition_absent(
            "dots_ocr_baseline_speed_and_cer_over_070_071_fixtures",
            "NV_DOTS_BENCH != 1",
            "set NV_DOTS_BENCH=1 and NV_DOTS_DIR to a rednote-hilab/dots.ocr snapshot (or leave unset to use the hub cache)",
        );
        return;
    }
    let dir = snapshot_dir().expect("dots.ocr snapshot present");
    let revision = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    let device = serving_device();
    let load_t0 = Instant::now();
    let pipe = DotsOcrPipeline::load(&dir, &device).expect("load");
    let precision = pipe.precision();
    eprintln!(
        "[dots-bench] load elapsed {:?} precision={precision:?}",
        load_t0.elapsed()
    );

    let opts = GenerateOptions {
        max_new_tokens: 2048,
        ..GenerateOptions::default()
    };

    let mut cers: Vec<f64> = Vec::new();
    let mut wers: Vec<f64> = Vec::new();
    let mut timed_secs: Vec<f64> = Vec::new();
    let mut timed_tokens: Vec<usize> = Vec::new();

    let stride = std::env::var("NV_DOTS_FIXTURE_STRIDE").ok().and_then(|v| {
        let (w, n) = v.split_once('/')?;
        Some((w.parse::<usize>().ok()?, n.parse::<usize>().ok()?))
    });
    for (i, name) in FIXTURES.iter().enumerate() {
        if let Some((w, n)) = stride {
            if i % n != w {
                continue;
            }
        }
        let img = fixture_rgb(name);
        let want = fixture_expected_text(name);
        let prep_t0 = Instant::now();
        let prep = pipe.prepare(&img).expect("prepare");
        let prep_s = prep_t0.elapsed().as_secs_f64();
        let n_vis = prep.num_vision_tokens();
        let t0 = Instant::now();
        let res = pipe
            .recognize(&img, DotsMode::PlainOcr, &opts)
            .expect("recognize");
        let elapsed = t0.elapsed().as_secs_f64();
        let e_cer = cer(&res.text, &want);
        let e_wer = wer(&res.text, &want);
        let tok_s = res.generated_tokens as f64 / elapsed.max(1e-6);
        eprintln!(
            "[dots-bench] {name}: {elapsed:.3}s, {} generated tokens, {tok_s:.1} tok/s, \
             src_px={}x{} vision_tokens={n_vis} prompt_tokens={} prepare_s={prep_s:.3} \
             CER={e_cer:.4} WER={e_wer:.4}",
            res.generated_tokens,
            img.w,
            img.h,
            res.prompt_tokens,
        );
        cers.push(e_cer);
        wers.push(e_wer);
        if i > 0 {
            eprintln!(
                "GEN-ARM dots-ocr fixture={name} vision_tokens={n_vis} prefill_tokens={} gen={} decode_steps={} prepare_s={prep_s:.3} decode_tok_s={tok_s:.2} cer={e_cer:.4} wer={e_wer:.4} precision={precision:?} basis=whole_recognize_wall_incl_vision_and_prefill_prepare_timed_separately_outside_the_wall checkpoint=rednote-hilab/dots.ocr@{revision}",
                res.prompt_tokens, res.generated_tokens, res.generated_tokens,
            );
            timed_secs.push(elapsed);
            timed_tokens.push(res.generated_tokens);
        }
    }

    let n = cers.len();
    let mean_cer = cers.iter().sum::<f64>() / n as f64;
    let mean_wer = wers.iter().sum::<f64>() / n as f64;
    let total_secs: f64 = timed_secs.iter().sum();
    let total_tokens: usize = timed_tokens.iter().sum();
    let mean_ms_per_image = total_secs * 1e3 / timed_secs.len() as f64;
    let pages_per_sec = timed_secs.len() as f64 / total_secs;
    eprintln!(
        "[dots-bench] WORKER stride={stride:?} wall_s={:.2} fixtures_run={}",
        load_t0.elapsed().as_secs_f64(),
        cers.len()
    );
    eprintln!(
        "[dots-bench] SUMMARY basis=rednote-hilab/dots.ocr snapshot={} device={device:?} precision={precision:?} budget={:?} mode=PlainOcr prompt=\"Extract the text content from this image.\" max_new_tokens={} images={n} (1 warmup discarded from timing) \
         mean_ms_per_image={mean_ms_per_image:.1} pages_per_sec={pages_per_sec:.3} total_tokens={total_tokens} tok_per_s={:.1} mean_CER={mean_cer:.4} mean_WER={mean_wer:.4}",
        dir.display(),
        pipe.pixel_budget(),
        opts.max_new_tokens,
        total_tokens as f64 / total_secs,
    );

    if stride.is_some() {
        eprintln!(
            "[dots-bench] STRIDE-MODE: worker shard of a throughput arm; the corpus-wide \
             asserts are the full-run gate and do not apply to a shard"
        );
        return;
    }
    assert_eq!(cers.len(), FIXTURES.len(), "must score every fixture image");
    assert!(total_tokens > 0, "generated zero tokens across the timed set");
    assert!(
        mean_cer < DOTS_MEAN_CER_GATE_MEASURED_0_0736_BF16_0_1913_NVFP4_ON_070_071_CUDA_C0111CE_DEB48F3,
        "mean CER {mean_cer:.4} at precision={precision:?} exceeds the pinned gate \
         {DOTS_MEAN_CER_GATE_MEASURED_0_0736_BF16_0_1913_NVFP4_ON_070_071_CUDA_C0111CE_DEB48F3}; \
         a weight-format default cannot flip until its arm passes this bound \
         (nvfp4 measured 0.1913: catastrophic on 070-ocr-multiword-boxes and 071-ocr-layout-letter)"
    );
}

#[test]
#[ignore]
fn dots_decoder_bare_step_time_no_vision_isolates_the_candle_metal_decode_wall() {
    if std::env::var("NV_DOTS_BENCH").as_deref() != Ok("1") {
        eprintln!("SKIP: NV_DOTS_BENCH != 1");
        return;
    }
    let snap = snapshot_dir().expect("dots snapshot");
    let device = serving_device();
    let precision = nv_models::dots_ocr::DotsPrecision::from_env_for(&device).expect("precision");
    let pipe = nv_models::dots_ocr::DotsOcrPipeline::load_with_precision(
        &snap, &device, precision,
    )
    .expect("load dots pipeline");
    let Some(dec) = pipe.decoder() else {
        eprintln!(
            "SKIP: this probe times the CANDLE decode wall, and the pipeline built no candle \
             decoder because the graph is serving. Re-run with {}=candle.",
            nv_models::dots_ocr::pipeline::DOTS_DEC_ENGINE_ENV
        );
        return;
    };
    let mut cache = dec.new_kv_cache(512).expect("kv cache");
    let prompt: Vec<u32> = (1u32..=64).collect();
    let emb = dec.embed_tokens(&prompt).expect("embed");
    let t0 = std::time::Instant::now();
    let v = dec.last_logits(&emb, 0, &mut cache).expect("prefill");
    let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut tok = v
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();
    let mut step_ms: Vec<f64> = Vec::new();
    for i in 0..32usize {
        let t1 = std::time::Instant::now();
        let e = dec.embed_tokens(&[tok]).expect("embed1");
        let v = dec
            .last_logits(&e, prompt.len() + i, &mut cache)
            .expect("step");
        tok = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        step_ms.push(t1.elapsed().as_secs_f64() * 1e3);
    }
    step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = step_ms[step_ms.len() / 2];
    eprintln!(
        "DOTS-DEC-PROBE precision={precision:?} device={device:?} prefill64_ms={prefill_ms:.1} \
         step_ms_min={:.1} med={med:.1} max={:.1} steps=32",
        step_ms[0],
        step_ms[step_ms.len() - 1]
    );
}

#[test]
#[ignore]
fn dots_page_budget_splits_vision_from_prefill_from_decode() {
    if std::env::var("NV_DOTS_BENCH").as_deref() != Ok("1") {
        eprintln!("SKIP: NV_DOTS_BENCH != 1");
        return;
    }
    let snap = snapshot_dir().expect("dots snapshot");
    let device = serving_device();
    let pipe = nv_models::dots_ocr::DotsOcrPipeline::load(&snap, &device).expect("load pipeline");
    let precision = pipe.precision();
    for name in ["070-ocr-paragraph", "070-ocr-photo-noise-surround", "071-ocr-layout-invoice"] {
        let img = fixture_rgb(name);
        let t0 = std::time::Instant::now();
        let prep = pipe.prepare(&img).expect("prepare");
        let prep_ms = t0.elapsed().as_secs_f64() * 1e3;
        let n_vis = prep.num_vision_tokens();

        let mut vision_ms = Vec::new();
        let mut one_ms = Vec::new();
        let mut whole_ms = Vec::new();
        let mut gen = 0usize;
        let mut one_tok_opts = nv_models::dots_ocr::GenerateOptions::default();
        one_tok_opts.max_new_tokens = 1;
        for _ in 0..REPS {
            let t = std::time::Instant::now();
            let feats = pipe.encode_vision(&prep).expect("encode_vision");
            vision_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let _ = feats;

            let t = std::time::Instant::now();
            pipe.recognize(&img, nv_models::dots_ocr::DotsMode::PlainOcr, &one_tok_opts)
                .expect("recognize 1 token");
            one_ms.push(t.elapsed().as_secs_f64() * 1e3);

            let t = std::time::Instant::now();
            let res = pipe
                .recognize(&img, nv_models::dots_ocr::DotsMode::PlainOcr, &Default::default())
                .expect("recognize");
            whole_ms.push(t.elapsed().as_secs_f64() * 1e3);
            gen = res.generated_tokens;
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let spread = |v: &Vec<f64>| (v[v.len() - 1] - v[0]) / v[0].max(1e-9) * 100.0;
        let vis = med(&mut vision_ms);
        let one = med(&mut one_ms);
        let whole = med(&mut whole_ms);
        let _ = DECODE_IS_THE_DIFFERENCE_OF_TWO_WALLS_THAT_BOTH_CONTAIN_VISION_PREFILL_AND_ONE_STEP_SO_IT_ASSUMES_NO_PHASE_SHARE_UNLIKE_THE_42_21_36_SPLIT_IT_REPLACES_WHICH_WAS_NEVER_MEASURED;
        let decode_ms = whole - one;
        let decode_tok_s = (gen.saturating_sub(1)) as f64 / (decode_ms / 1e3).max(1e-9);
        let prefill_ms = one - vis - (decode_ms / (gen.saturating_sub(1)).max(1) as f64);
        eprintln!(
            "DOTS-BUDGET fixture={name} precision={precision:?} device={device:?} \
             vision_tokens={n_vis} gen={gen} reps={REPS} prepare_ms={prep_ms:.0} \
             vision_ms={vis:.0} prefill_ms={prefill_ms:.0} decode_ms={decode_ms:.0} \
             whole_ms={whole:.0} decode_tok_s={decode_tok_s:.1} \
             shares={:.2}/{:.2}/{:.2} rep_spread_pct={:.1}/{:.1}/{:.1}",
            vis / whole,
            prefill_ms / whole,
            decode_ms / whole,
            spread(&vision_ms),
            spread(&one_ms),
            spread(&whole_ms),
        );
    }
}

const REPS: usize = 3;

const DECODE_IS_THE_DIFFERENCE_OF_TWO_WALLS_THAT_BOTH_CONTAIN_VISION_PREFILL_AND_ONE_STEP_SO_IT_ASSUMES_NO_PHASE_SHARE_UNLIKE_THE_42_21_36_SPLIT_IT_REPLACES_WHICH_WAS_NEVER_MEASURED: () = ();
