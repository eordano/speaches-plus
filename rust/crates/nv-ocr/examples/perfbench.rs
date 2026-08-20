use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use nv_ocr::binarize::{binarize_sauvola, ink_is_dark, SAUVOLA_K, SAUVOLA_WINDOW};
use nv_ocr::ctc;
use nv_ocr::layout::extract_lines;
use nv_ocr::lstm::input_from_grey;
use nv_ocr::raster;
use nv_ocr::vgsl::Network;
use nv_ocr::{BackendKind, LstmModel, OcrEngine};

fn cer(reference: &str, hypothesis: &str) -> f64 {
    let r: Vec<char> = reference.chars().collect();
    let h: Vec<char> = hypothesis.chars().collect();
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for i in 1..=r.len() {
        let mut curr = vec![i; h.len() + 1];
        for j in 1..=h.len() {
            let sub = prev[j - 1] + usize::from(r[i - 1] != h[j - 1]);
            curr[j] = sub.min(prev[j] + 1).min(curr[j - 1] + 1);
        }
        prev = curr;
    }
    prev[h.len()] as f64 / r.len().max(1) as f64
}

fn layer_label(n: &Network) -> String {
    match n {
        Network::Input { height, depth } => format!("Input(h={height},d={depth})"),
        Network::Series(v) => format!("Series[{}]", v.len()),
        Network::Convolve { half_x, half_y } => format!("Convolve({half_x},{half_y})"),
        Network::Maxpool { x_scale, y_scale } => format!("Maxpool({x_scale},{y_scale})"),
        Network::ReversedX(inner) => format!("RevX<{}>", layer_label(inner)),
        Network::ReversedY(inner) => format!("RevY<{}>", layer_label(inner)),
        Network::TransposedXY(inner) => format!("TransXY<{}>", layer_label(inner)),
        Network::Lstm(l) => format!(
            "Lstm(ni={},ns={}{})",
            l.ni,
            l.ns,
            if l.summarizing { ",sum" } else { "" }
        ),
        Network::Fc { weights, .. } => format!("Fc({}x{})", weights.rows, weights.cols),
    }
}

struct StageAcc {
    labels: Vec<String>,
    times: Vec<Duration>,
}

impl StageAcc {
    fn new() -> Self {
        Self {
            labels: Vec::new(),
            times: Vec::new(),
        }
    }

    fn add(&mut self, label: &str, d: Duration) {
        if let Some(i) = self.labels.iter().position(|l| l == label) {
            self.times[i] += d;
        } else {
            self.labels.push(label.to_string());
            self.times.push(d);
        }
    }
}

fn forward_timed(
    net: &Network,
    input: &nv_ocr::lstm::Tensor,
    acc: &mut StageAcc,
) -> nv_ocr::lstm::Tensor {
    match net {
        Network::Series(stack) => {
            let mut t = input.clone();
            for (i, n) in stack.iter().enumerate() {
                let t0 = Instant::now();
                t = n.forward(&t).unwrap();
                acc.add(&format!("net[{i:02}] {}", layer_label(n)), t0.elapsed());
            }
            t
        }
        other => {
            let t0 = Instant::now();
            let t = other.forward(input).unwrap();
            acc.add(&format!("net[00] {}", layer_label(other)), t0.elapsed());
            t
        }
    }
}

fn corpus_pages(dir: &Path) -> Vec<(PathBuf, String)> {
    let mut pages: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension().map(|x| x == "png").unwrap_or(false)).then_some(p)
        })
        .collect();
    pages.sort();
    pages
        .into_iter()
        .map(|p| {
            let txt = p.with_extension("txt");
            let gt = std::fs::read_to_string(&txt).unwrap_or_default();
            (p, gt.trim().to_string())
        })
        .collect()
}

fn run_stage(model_path: &Path, pages: &[(PathBuf, String)]) {
    let model = LstmModel::from_file(model_path).unwrap();
    eprintln!(
        "model spec: {}  int_mode: {}  version: {:?}",
        model.spec, model.int_mode, model.version
    );
    let input_h = model.input_height().unwrap();
    let mut acc = StageAcc::new();
    let mut total_lines = 0usize;
    let mut total_chars = 0usize;
    let mut cer_sum = 0.0f64;
    let wall0 = Instant::now();
    for (path, gt) in pages {
        let bytes = std::fs::read(path).unwrap();
        let t0 = Instant::now();
        let grey = raster::load(&bytes).unwrap();
        acc.add("png-decode", t0.elapsed());
        let t0 = Instant::now();
        let _bin = binarize_sauvola(&grey, SAUVOLA_WINDOW, SAUVOLA_K);
        acc.add("binarize-sauvola(standalone)", t0.elapsed());
        let t0 = Instant::now();
        let layout = extract_lines(&grey);
        acc.add("layout-extract_lines(incl binarize)", t0.elapsed());
        let mut page_text = String::new();
        for line in &layout.lines {
            let t0 = Instant::now();
            let strip = if ink_is_dark(&line.grey_strip) {
                line.grey_strip.clone()
            } else {
                line.grey_strip.invert()
            };
            let input = input_from_grey(&strip, input_h, model.int_mode).unwrap();
            acc.add("line-input-prep", t0.elapsed());
            let out = forward_timed(&model.network, &input, &mut acc);
            let t0 = Instant::now();
            let logits = out.into_logits().unwrap();
            let steps =
                ctc::beam_decode(&logits, model.null_code, ctc::DEFAULT_BEAM_WIDTH).unwrap();
            acc.add("ctc-beam-decode", t0.elapsed());
            let t0 = Instant::now();
            let chars = ctc::codes_to_unichars(&steps, &model.recoder, &model.unicharset);
            acc.add("ctc-codes-to-unichars", t0.elapsed());
            if !page_text.is_empty() {
                page_text.push('\n');
            }
            for c in &chars {
                page_text.push_str(&c.text);
            }
            total_lines += 1;
        }
        total_chars += page_text.chars().count();
        cer_sum += cer(gt, page_text.trim());
    }
    let wall = wall0.elapsed();
    let sum_no_standalone: Duration = acc
        .labels
        .iter()
        .zip(&acc.times)
        .filter(|(l, _)| !l.contains("standalone"))
        .map(|(_, t)| *t)
        .sum();
    println!(
        "\n== stage breakdown ({} pages, {} lines, {} chars) ==",
        pages.len(),
        total_lines,
        total_chars
    );
    println!(
        "{:<44} {:>10} {:>7} {:>10}",
        "stage", "total ms", "%", "ms/page"
    );
    for (l, t) in acc.labels.iter().zip(&acc.times) {
        let ms = t.as_secs_f64() * 1e3;
        let pct = if l.contains("standalone") {
            f64::NAN
        } else {
            100.0 * t.as_secs_f64() / sum_no_standalone.as_secs_f64()
        };
        println!(
            "{:<44} {:>10.1} {:>6.1}% {:>10.2}",
            l,
            ms,
            pct,
            ms / pages.len() as f64
        );
    }
    println!(
        "wall {:.1} ms  ({:.2} ms/page, {:.3} pages/s)  mean CER {:.4}",
        wall.as_secs_f64() * 1e3,
        wall.as_secs_f64() * 1e3 / pages.len() as f64,
        pages.len() as f64 / wall.as_secs_f64(),
        cer_sum / pages.len() as f64
    );
}

fn run_e2e(model_path: &Path, pages: &[(PathBuf, String)], threads: usize) {
    let engine = OcrEngine::from_traineddata(model_path, BackendKind::Classical).unwrap();
    let inputs: Vec<(Vec<u8>, &String)> = pages
        .iter()
        .map(|(p, gt)| (std::fs::read(p).unwrap(), gt))
        .collect();
    let wall0 = Instant::now();
    let cer_sum: f64 = if threads <= 1 {
        let mut s = 0.0;
        for (bytes, gt) in &inputs {
            let out = engine.recognize(bytes).unwrap();
            s += cer(gt, out.text.trim());
        }
        s
    } else {
        let next = AtomicUsize::new(0);
        let results: Vec<std::sync::Mutex<f64>> =
            inputs.iter().map(|_| std::sync::Mutex::new(0.0)).collect();
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= inputs.len() {
                        break;
                    }
                    let (bytes, gt) = &inputs[i];
                    let out = engine.recognize(bytes).unwrap();
                    *results[i].lock().unwrap() = cer(gt, out.text.trim());
                });
            }
        });
        results.iter().map(|m| *m.lock().unwrap()).sum()
    };
    let wall = wall0.elapsed();
    println!(
        "e2e threads={} pages={} wall {:.1} ms  ({:.2} ms/page, {:.3} pages/s)  mean CER {:.4}",
        threads,
        pages.len(),
        wall.as_secs_f64() * 1e3,
        wall.as_secs_f64() * 1e3 / pages.len() as f64,
        pages.len() as f64 / wall.as_secs_f64(),
        cer_sum / pages.len() as f64
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: perfbench <eng.traineddata> <corpus-dir> <stage|e2e> [threads]");
        std::process::exit(2);
    }
    let model_path = PathBuf::from(&args[1]);
    let pages = corpus_pages(Path::new(&args[2]));
    assert!(!pages.is_empty(), "no pages in corpus dir");
    if std::env::var_os("NV_OCR_THREADS").is_none() {
        std::env::set_var("NV_OCR_THREADS", "1");
    }
    match args[3].as_str() {
        "stage" => run_stage(&model_path, &pages),
        "e2e" => {
            let threads = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(1);
            run_e2e(&model_path, &pages, threads)
        }
        m => {
            eprintln!("unknown mode {m}");
            std::process::exit(2);
        }
    }
}
