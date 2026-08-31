#![cfg(feature = "wgpu")]

mod hub_snapshot;

use std::time::Instant;

use nv_models::dots_ocr::{DotsDecoderConfig, DotsDecoderWgpuGraph, DotsGraphQuant, RegionRow};
use nv_weights::WeightLoader;

const CONTEXT_512_IS_THE_CROP_PROMPT_CLASS: usize = 512;

#[test]
#[ignore]
fn batched_step_price_serial_steps_vs_m_chunks_bounds_the_b_gt_1_decode_win() {
    if std::env::var("NV_DOTS_SCALE").as_deref() != Ok("1") {
        hub_snapshot::precondition_absent(
            "batched_step_price",
            "NV_DOTS_SCALE != 1",
            "set NV_DOTS_SCALE=1 to run the batched-step pricing probe",
        );
        return;
    }
    let dir = hub_snapshot::dir_from_env_or_hub(
        "NV_DOTS_DIR",
        "rednote-hilab/dots.ocr",
        &["config.json", "tokenizer.json", "*.safetensors"],
    )
    .expect("dots.ocr snapshot present");
    let cfg = DotsDecoderConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
    let loader =
        WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("open dots weights");
    let mut g = DotsDecoderWgpuGraph::from_loader(cfg, &loader, 4096, DotsGraphQuant::AffineW4)
        .expect("load w4 graph");
    eprintln!(
        "[bstep] graph loaded: {} passes/step, {} prefill passes, widths {:?}",
        g.pass_count(),
        g.prefill_pass_count(),
        g.prefill_widths(),
    );
    let toks: Vec<u32> = (0..CONTEXT_512_IS_THE_CROP_PROMPT_CLASS as u32)
        .map(|i| 1000 + (i % 4000))
        .collect();
    let done = g.prefill_tokens(&toks).expect("context prefill");
    assert_eq!(done, toks.len());
    for i in 0..8u32 {
        g.decode_step(2000 + i).expect("warm step");
    }

    let n_steps = 64usize;
    let t0 = Instant::now();
    for i in 0..n_steps as u32 {
        g.decode_step(2100 + i).expect("timed step");
    }
    let serial_full_ms = t0.elapsed().as_secs_f64() * 1e3 / n_steps as f64;

    let t0 = Instant::now();
    for i in 0..n_steps as u32 {
        g.prefill_step(2200 + i).expect("trunk step");
    }
    let _ = g.read_logits().expect("sync");
    let serial_trunk_ms = t0.elapsed().as_secs_f64() * 1e3 / n_steps as f64;

    let mut chunk_ms = Vec::new();
    for &m in &[32usize, 128] {
        if !g.prefill_widths().contains(&m) {
            eprintln!("[bstep] width {m} not built, skipping");
            chunk_ms.push((m, f64::NAN));
            continue;
        }
        let reps = (256 / m).max(2);
        let t0 = Instant::now();
        for _ in 0..reps {
            let done = g.prefill_tokens(&toks[..m]).expect("m chunk");
            assert_eq!(done, m);
        }
        let _ = g.read_logits().expect("sync");
        chunk_ms.push((m, t0.elapsed().as_secs_f64() * 1e3 / reps as f64));
    }

    eprintln!(
        "DOTS-BSTEP serial_full_ms_per_tok={serial_full_ms:.3} \
         serial_trunk_nosync_ms_per_tok={serial_trunk_ms:.3} ctx={} steps={n_steps} \
         basis=w4_graph_span4096_flock_ctx_grows_during_timing \
         caveat=chunk_attention_reads_the_whole_prefix_causally_a_batched_step_reads_B_short_ranges",
        CONTEXT_512_IS_THE_CROP_PROMPT_CLASS,
    );
    for (m, ms) in &chunk_ms {
        let per_row = ms / *m as f64;
        let ceiling = serial_trunk_ms / per_row;
        eprintln!(
            "DOTS-BSTEP chunk_m={m} chunk_ms={ms:.3} trunk_ms_per_row={per_row:.4} \
             batched_trunk_ceiling_vs_serial={ceiling:.1}x"
        );
    }
    assert!(serial_full_ms > 0.0 && serial_trunk_ms > 0.0);
}

#[test]
#[ignore]
fn batched_region_decode_real_model_ab_vs_serial_sessions() {
    if std::env::var("NV_DOTS_SCALE").as_deref() != Ok("1") {
        hub_snapshot::precondition_absent(
            "batched_region_decode_real_model_ab",
            "NV_DOTS_SCALE != 1",
            "set NV_DOTS_SCALE=1 to run the batched-decode A/B probe",
        );
        return;
    }
    let dir = hub_snapshot::dir_from_env_or_hub(
        "NV_DOTS_DIR",
        "rednote-hilab/dots.ocr",
        &["config.json", "tokenizer.json", "*.safetensors"],
    )
    .expect("dots.ocr snapshot present");
    let cfg = DotsDecoderConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
    let loader =
        WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("open dots weights");
    let mut g = DotsDecoderWgpuGraph::from_loader(cfg, &loader, 32768, DotsGraphQuant::AffineW4)
        .expect("load w4 graph");
    let (slots, cap) = g.region_capacity().expect("region machinery at span 32768");
    eprintln!("[bstep-ab] slots={slots} stride_cap={cap}");
    let prompt_len = 200usize;
    let gen_steps = 32usize;
    let prompts: Vec<Vec<u32>> = (0..slots)
        .map(|s| {
            (0..prompt_len)
                .map(|i| 1000 + ((s * 37 + i * 13) % 4000) as u32)
                .collect()
        })
        .collect();

    let serial_n = 8usize;
    let mut serial_tokens: Vec<Vec<u32>> = Vec::new();
    let mut serial_decode = 0.0f64;
    for p in prompts.iter().take(serial_n) {
        g.reset().expect("reset");
        let logits = g.prefill_prompt(p).expect("serial prefill");
        let mut tok = DotsDecoderWgpuGraph::greedy_token(&logits);
        let mut out = vec![tok];
        let t = Instant::now();
        for _ in 1..gen_steps {
            tok = g.decode_step(tok).expect("serial step");
            out.push(tok);
        }
        serial_decode += t.elapsed().as_secs_f64();
        serial_tokens.push(out);
    }
    let serial_wall = serial_decode;

    let stride = 32768 / slots;
    let make_rows = |g: &mut DotsDecoderWgpuGraph| -> Vec<RegionRow> {
        let mut rows = Vec::new();
        for (s, p) in prompts.iter().enumerate() {
            let len = g.region_prefill(s, p, 0, None).expect("region prefill");
            rows.push(RegionRow {
                token: *p.last().unwrap(),
                base: (s * stride) as u32,
                len: len as u32,
            });
        }
        rows
    };
    g.reset().expect("reset");
    let mut warm = make_rows(&mut g);
    for _ in 0..2 {
        let toks = g.bstep_tokens(&warm).expect("warm bstep");
        for (r, t) in toks.iter().enumerate() {
            warm[r].len += 1;
            warm[r].token = *t;
        }
    }
    g.reset().expect("reset");
    let t1 = Instant::now();
    let rows_init = make_rows(&mut g);
    let mut rows = rows_init;
    let _ = g.read_logits().expect("drain the async region prefills before timing decode");
    let bpf_wall = t1.elapsed().as_secs_f64();
    let t2 = Instant::now();
    let mut batched: Vec<Vec<u32>> = vec![Vec::new(); slots];
    for _ in 0..gen_steps {
        let toks = g.bstep_tokens(&rows).expect("bstep");
        for (r, t) in toks.iter().enumerate() {
            batched[r].push(*t);
            rows[r].len += 1;
            rows[r].token = *t;
        }
    }
    let bdec_wall = t2.elapsed().as_secs_f64();

    let mut agree = 0usize;
    let mut total = 0usize;
    let mut region_exact = 0usize;
    for r in 0..serial_n {
        let same = serial_tokens[r]
            .iter()
            .zip(&batched[r])
            .take_while(|(a, b)| a == b)
            .count();
        agree += same;
        total += gen_steps;
        if same == gen_steps {
            region_exact += 1;
        }
        if same < gen_steps {
            eprintln!(
                "[bstep-ab] region {r}: diverges at step {same} (serial {:?} vs batched {:?})",
                &serial_tokens[r][same.saturating_sub(1)..(same + 2).min(gen_steps)],
                &batched[r][same.saturating_sub(1)..(same + 2).min(gen_steps)],
            );
        }
    }
    g.reset().expect("reset");
    let len0 = g
        .region_prefill(0, &prompts[0], 0, None)
        .expect("solo region prefill");
    let mut solo_row = vec![RegionRow {
        token: *prompts[0].last().unwrap(),
        base: 0,
        len: len0 as u32,
    }];
    let mut solo: Vec<u32> = Vec::new();
    for _ in 0..gen_steps {
        let t = g.bstep_tokens(&solo_row).expect("solo bstep")[0];
        solo.push(t);
        solo_row[0].len += 1;
        solo_row[0].token = t;
    }
    assert_eq!(
        solo, batched[0],
        "ROW INDEPENDENCE VIOLATED: a region's batched tokens depend on which other rows \
         ride the batch -- cross-region contamination, the failure class this gate exists \
         to catch (GEMM rows, ranged attention, rope and kv-write are all per-row \
         addressing and must be bit-identical solo vs batched)"
    );
    eprintln!("DOTS-BSTEP-INDEP region0 solo==batched over {gen_steps} steps: BIT-IDENTICAL");

    g.reset().expect("reset");
    rows = make_rows(&mut g);
    let _ = g.read_logits().expect("drain");
    for nrows in [1usize, 8, 32] {
        let mut probe_rows: Vec<RegionRow> = rows[..nrows].to_vec();
        let t = Instant::now();
        for _ in 0..8 {
            let toks = g.bstep_tokens(&probe_rows).expect("probe bstep");
            for (r, tk) in toks.iter().enumerate() {
                probe_rows[r].len += 1;
                probe_rows[r].token = *tk;
            }
        }
        eprintln!(
            "DOTS-BSTEP-SPLIT rows={nrows} ms_per_bstep={:.2}",
            t.elapsed().as_secs_f64() * 1e3 / 8.0
        );
    }
    let serial_ms_tok = serial_wall * 1e3 / (serial_n * gen_steps) as f64;
    let batched_ms_tok = bdec_wall * 1e3 / (slots * gen_steps) as f64;
    eprintln!(
        "DOTS-BSTEP-AB serial_n={serial_n} serial_wall_s={serial_wall:.2} \
         serial_ms_per_tok={serial_ms_tok:.3} batched_slots={slots} \
         batched_prefill_s={bpf_wall:.2} batched_decode_s={bdec_wall:.2} \
         batched_ms_per_tok={batched_ms_tok:.3} speedup={:.1}x \
         prefix_agree={agree}/{total} regions_exact={region_exact}/{serial_n} \
         basis=w4_graph_span32768_200tok_prompts_32steps_flock_greedy \
         caveat=agreement_is_reported_not_gated_m32_gemm_vs_m1_gemv_trunks_reassociate",
        serial_ms_tok / batched_ms_tok,
    );
    assert!(
        agree as f64 / total as f64 > 0.5,
        "batched decode disagrees with serial sessions on most tokens; that is a \
         correctness bug, not reassociation ({agree}/{total})"
    );
}
