use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::Device;
use nv_models::deepseek_ocr::{
    build_prompt_tokens, decoder::prof, preprocess, DecoderPrecision, DeepSeekOcr2Pipeline,
    ResolutionMode, RgbImage, PROMPT_FREE_OCR,
};

use nv_models::deepseek_ocr::default_snapshot_dir as default_dir;

fn load_rgb(path: &PathBuf) -> Result<RgbImage> {
    RgbImage::decode_file(path)
}

#[derive(Default, Clone)]
struct Acc {
    prep: f64,
    vision: f64,
    tokens: f64,
    embed: f64,
    prefill: f64,
    total: f64,
    pages: f64,
}

fn main() -> Result<()> {
    let mut images: Vec<PathBuf> = Vec::new();
    let mut rounds = 3usize;
    let mut conc: Vec<usize> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--rounds" => {
                rounds = args.next().context("--rounds needs a value")?.parse()?;
            }
            "--conc" => {
                conc = args
                    .next()
                    .context("--conc needs a value")?
                    .split(',')
                    .map(|s| s.parse::<usize>().unwrap_or(1))
                    .collect();
            }
            other => images.push(PathBuf::from(other)),
        }
    }
    anyhow::ensure!(
        !images.is_empty(),
        "usage: dsocr_front_prof [--rounds N] [--conc 1,2,4] <image>..."
    );

    let dir = default_dir().context("DeepSeek-OCR-2 snapshot not found; set NV_DSOCR_DIR")?;
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    eprintln!("loading DeepSeek-OCR-2 from {}", dir.display());
    let pipe = Arc::new(DeepSeekOcr2Pipeline::load(
        &dir,
        &device,
        DecoderPrecision::Bf16,
    )?);
    let decoder = pipe.decoder_arc();

    let mut pages: Vec<(PathBuf, RgbImage)> = Vec::new();
    for p in &images {
        pages.push((p.clone(), load_rgb(p)?));
    }

    if std::env::var("NV_H2D_CHECK").is_ok() {
        let mut bad = 0usize;
        for (path, img) in &pages {
            let prep = preprocess::prepare(img, ResolutionMode::Gundam)?;
            std::env::set_var("NV_DSOCR_H2D_PINNED", "0");
            let a = pipe.vision().encode_prepared(&prep)?;
            std::env::set_var("NV_DSOCR_H2D_PINNED", "1");
            let b = pipe.vision().encode_prepared(&prep)?;
            let av: Vec<f32> = a
                .flatten_all()?
                .to_dtype(candle_core::DType::F32)?
                .to_vec1()?;
            let bv: Vec<f32> = b
                .flatten_all()?
                .to_dtype(candle_core::DType::F32)?
                .to_vec1()?;
            let diff = av
                .iter()
                .zip(&bv)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            println!(
                "{}: vision features {} elems, {diff} bit-differing",
                path.file_name().unwrap().to_string_lossy(),
                av.len()
            );
            bad += diff;
        }
        println!(
            "H2D pinned-bf16 vs pageable-f32+cast: {}",
            if bad == 0 { "BIT-IDENTICAL" } else { "DIFFERS" }
        );
        return Ok(());
    }

    let run_front = |img: &RgbImage, a: &mut Acc| -> Result<()> {
        let t_page = Instant::now();
        let t = Instant::now();
        let prep = preprocess::prepare(img, ResolutionMode::Gundam)?;
        a.prep += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let feats = pipe.vision().encode_prepared(&prep)?;
        device.synchronize()?;
        a.vision += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let toks = build_prompt_tokens(
            |s| pipe.encode_text(s),
            PROMPT_FREE_OCR,
            prep.vision_tokens(),
        )?;
        a.tokens += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let x = decoder.embed_tokens_with_vision(&toks, Some(&feats))?;
        device.synchronize()?;
        a.embed += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut cache = decoder.new_kv_cache(toks.len() + 512)?;
        let _h = decoder.forward_embeds_hidden(&x, &mut cache)?;
        device.synchronize()?;
        a.prefill += t.elapsed().as_secs_f64();

        a.pages += 1.0;
        a.total += t_page.elapsed().as_secs_f64();
        Ok(())
    };

    let mut acc = Acc::default();
    for round in 0..rounds + 1 {
        let mut a = Acc::default();
        for (_, img) in &pages {
            run_front(img, &mut a)?;
        }
        if round == 0 {
            prof::reset();
            continue;
        }
        acc.prep += a.prep;
        acc.vision += a.vision;
        acc.tokens += a.tokens;
        acc.embed += a.embed;
        acc.prefill += a.prefill;
        acc.total += a.total;
        acc.pages += a.pages;
        eprintln!(
            "round {round}: front {:.1} ms/page (prep {:.1} vision {:.1} prefill {:.1})",
            a.total / a.pages * 1e3,
            a.prep / a.pages * 1e3,
            a.vision / a.pages * 1e3,
            a.prefill / a.pages * 1e3
        );
    }

    let p = acc.pages;
    let ms = |v: f64| v / p * 1e3;
    println!("pages {p}");
    println!("| front-end stage | ms/page | share |");
    println!("|---|---|---|");
    let rows = [
        ("preprocess (CPU)", acc.prep),
        ("vision tower (H2D+SAM+comp+flow+proj)", acc.vision),
        ("prompt tokenize (CPU)", acc.tokens),
        ("embed + vision splice", acc.embed),
        ("decoder prefill", acc.prefill),
    ];
    for (n, v) in rows {
        println!("| {n} | {:.1} | {:.1}% |", ms(v), v / acc.total * 100.0);
    }
    println!("| **front end total** | {:.1} | 100% |", ms(acc.total));
    println!("front-end pages/s (serial) {:.3}", p / acc.total);

    let sub = prof::report(p);
    if !sub.trim().is_empty() {
        println!("\n{sub}");
    }

    if conc.is_empty() {
        return Ok(());
    }

    let preps: Vec<Arc<preprocess::PreparedViews>> = pages
        .iter()
        .map(|(_, img)| {
            Arc::new(preprocess::prepare(img, ResolutionMode::Gundam).expect("prepare"))
        })
        .collect();

    println!("\n| front threads | pages/s | ms/page/thread | scaling |");
    println!("|---|---|---|---|");
    let mut base = 0f64;
    for &c in &conc {
        let reps = 3usize;
        let barrier = Arc::new(Barrier::new(c + 1));
        let hs: Vec<_> = (0..c)
            .map(|t| {
                let pipe = pipe.clone();
                let dec = decoder.clone();
                let preps = preps.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || -> usize {
                    let dev = pipe.device().clone();
                    let warm = &preps[t % preps.len()];
                    let f = pipe.vision().encode_prepared(warm).expect("warm vision");
                    let tk = build_prompt_tokens(
                        |s| pipe.encode_text(s),
                        PROMPT_FREE_OCR,
                        warm.vision_tokens(),
                    )
                    .expect("tok");
                    let x = dec.embed_tokens_with_vision(&tk, Some(&f)).expect("embed");
                    let mut cache = dec.new_kv_cache(tk.len() + 512).expect("cache");
                    let _ = dec
                        .forward_embeds_hidden(&x, &mut cache)
                        .expect("warm prefill");
                    dev.synchronize().expect("sync");
                    barrier.wait();
                    let mut n = 0usize;
                    for r in 0..reps {
                        let prep = &preps[(t + r) % preps.len()];
                        let f = pipe.vision().encode_prepared(prep).expect("vision");
                        let tk = build_prompt_tokens(
                            |s| pipe.encode_text(s),
                            PROMPT_FREE_OCR,
                            prep.vision_tokens(),
                        )
                        .expect("tok");
                        let x = dec.embed_tokens_with_vision(&tk, Some(&f)).expect("embed");
                        let mut cache = dec.new_kv_cache(tk.len() + 512).expect("cache");
                        let _ = dec.forward_embeds_hidden(&x, &mut cache).expect("prefill");
                        n += 1;
                    }
                    dev.synchronize().expect("sync");
                    n
                })
            })
            .collect();
        barrier.wait();
        let t0 = Instant::now();
        let mut total = 0usize;
        for h in hs {
            total += h.join().expect("front thread");
        }
        let dt = t0.elapsed().as_secs_f64();
        let rate = total as f64 / dt;
        if base == 0.0 {
            base = rate;
        }
        println!(
            "| {c} | {:.3} | {:.1} | {:.2}x |",
            rate,
            dt * 1e3 / (total as f64 / c as f64),
            rate / base
        );
    }

    #[cfg(feature = "cuda")]
    {
        use nv_models::deepseek_ocr::DsocrDecodeGraph;
        let cap = decoder.config().max_position_embeddings;
        let prep = preps[0].clone();
        let feats = Arc::new(pipe.vision().encode_prepared(&prep)?);
        let toks = Arc::new(build_prompt_tokens(
            |s| pipe.encode_text(s),
            PROMPT_FREE_OCR,
            prep.vision_tokens(),
        )?);
        let gopts = nv_models::deepseek_ocr::GenerateOptions {
            max_new_tokens: 128,
            ..Default::default()
        };
        let mut g = DsocrDecodeGraph::new(decoder.clone(), cap)?;
        let _ = g.generate(&toks, Some(&feats), &gopts)?;
        device.synchronize()?;

        let steps = 192usize;
        let probe: [u32; 8] = [100, 200, 300, 400, 500, 600, 700, 800];
        let run = |g: &mut DsocrDecodeGraph| -> f64 {
            g.reset();
            let t = Instant::now();
            for i in 0..steps {
                g.step(probe[i % probe.len()]).expect("step");
                let _ = g.logits_host().expect("logits");
            }
            t.elapsed().as_secs_f64() * 1e3 / steps as f64
        };
        let solo = run(&mut g);

        let stop = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(Barrier::new(2));
        let s2 = stop.clone();
        let d2 = done.clone();
        let g2 = gate.clone();
        let pv = pipe.clone();
        let dv = decoder.clone();
        let pp = prep.clone();
        let front = std::thread::spawn(move || {
            g2.wait();
            while !s2.load(Ordering::Relaxed) {
                let f = pv.vision().encode_prepared(&pp).expect("vision");
                let tk =
                    build_prompt_tokens(|s| pv.encode_text(s), PROMPT_FREE_OCR, pp.vision_tokens())
                        .expect("tok");
                let x = dv.embed_tokens_with_vision(&tk, Some(&f)).expect("embed");
                let mut cache = dv.new_kv_cache(tk.len() + 512).expect("cache");
                let _ = dv.forward_embeds_hidden(&x, &mut cache).expect("prefill");
                d2.fetch_add(1, Ordering::Relaxed);
            }
        });
        gate.wait();
        let loaded = run(&mut g);
        stop.store(true, Ordering::Relaxed);
        front.join().expect("front thread");
        println!(
            "\ndecode-step solo={solo:.3} ms/step | under {} concurrent front (vision+prefill) \
             passes={loaded:.3} ms/step | ratio={:.2}",
            done.load(Ordering::Relaxed),
            solo / loaded
        );
    }

    Ok(())
}
