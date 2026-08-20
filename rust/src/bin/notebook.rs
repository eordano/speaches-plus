use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::Device;
use nv_models::deepseek_ocr::{
    DecoderPrecision, DeepSeekOcr2Pipeline, GenerateOptions as DsGenerateOptions, ResolutionMode,
    RgbImage, PROMPT_FREE_OCR,
};
use nv_models::dots_ocr::{
    DotsMode, DotsOcrPipeline, GenerateOptions as DotsGenerateOptions, PixelBudget,
};
use serde::Serialize;

const FIGURE_CATEGORIES: [&str; 1] = ["Picture"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReaderKind {
    Layout,
    DotsCrop,
    DsocrCrop,
}

impl ReaderKind {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "layout" | "dots-page" | "page" => Some(ReaderKind::Layout),
            "dots-crop" | "crop" => Some(ReaderKind::DotsCrop),
            "dsocr-crop" | "deepseek-crop" => Some(ReaderKind::DsocrCrop),
            _ => None,
        }
    }

    #[allow(dead_code)]
    fn as_str(self) -> &'static str {
        match self {
            ReaderKind::Layout => "layout",
            ReaderKind::DotsCrop => "dots-crop",
            ReaderKind::DsocrCrop => "dsocr-crop",
        }
    }

    fn needs_crops(self) -> bool {
        !matches!(self, ReaderKind::Layout)
    }
}

#[derive(Debug)]
struct RegionRead {
    text: String,
    looped: bool,
}

trait RegionReader {
    fn name(&self) -> &'static str;
    fn read(&self, crop: &RgbImage) -> Result<RegionRead>;
}

struct DotsCropReader<'a> {
    pipeline: &'a DotsOcrPipeline,
    max_new_tokens: usize,
}

impl RegionReader for DotsCropReader<'_> {
    fn name(&self) -> &'static str {
        "dots-crop"
    }

    fn read(&self, crop: &RgbImage) -> Result<RegionRead> {
        let opts = DotsGenerateOptions {
            max_new_tokens: self.max_new_tokens,
            ..Default::default()
        };
        let res = self.pipeline.recognize(crop, DotsMode::PlainOcr, &opts)?;
        Ok(RegionRead {
            text: res.text.trim().to_string(),
            looped: res.looped,
        })
    }
}

struct DsocrCropReader {
    pipeline: DeepSeekOcr2Pipeline,
    max_new_tokens: usize,
}

impl RegionReader for DsocrCropReader {
    fn name(&self) -> &'static str {
        "dsocr-crop"
    }

    fn read(&self, crop: &RgbImage) -> Result<RegionRead> {
        let opts = DsGenerateOptions {
            max_new_tokens: self.max_new_tokens,
            ..Default::default()
        };
        let mode = if crop.w.max(crop.h) > 1024 {
            ResolutionMode::Gundam
        } else {
            ResolutionMode::Base1024
        };
        let (text, looped) = self
            .pipeline
            .recognize_flagged(crop, PROMPT_FREE_OCR, mode, &opts)?;
        Ok(RegionRead {
            text: text.trim().to_string(),
            looped,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct FigureRecord {
    index: usize,
    file: String,
    bbox: [i64; 4],
    width: usize,
    height: usize,
}

#[derive(Debug, Clone, Serialize)]
struct RegionRecord {
    order: usize,
    category: String,
    bbox: Option<[f32; 4]>,
    chars: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    agreement: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct PageReport {
    page: usize,
    source: String,
    markdown: String,
    width: usize,
    height: usize,
    layout_ms: f64,
    read_ms: f64,
    elements: usize,
    text_regions: usize,
    figures: Vec<FigureRecord>,
    regions: Vec<RegionRecord>,
    flags: Vec<String>,
    score: f64,
    needs_review: bool,
    reader: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    mean_agreement: Option<f64>,
}

#[derive(Debug, Serialize)]
struct RunReport {
    reader: String,
    cross_check: bool,
    pages: Vec<PageReport>,
    pages_total: usize,
    pages_needing_review: usize,
    figures_total: usize,
    wall_s: f64,
    caveat: &'static str,
}

const CAVEAT: &str = "Structural confidence only: the decoder exposes no token logprobs, so `score` is a weighted penalty over layout/decode failure modes (plus optional two-read agreement), NOT a calibrated probability. It has been exercised on synthetic handwriting proxies only.";

struct Args {
    inputs: Vec<PathBuf>,
    out: PathBuf,
    reader: ReaderKind,
    cross_check: bool,
    figure_pad: i64,
    figure_min_frac: f64,
    crop_pad: i64,
    crop_min_side: usize,
    max_new_tokens: usize,
    review_threshold: f64,
    pdf_dpi: u32,
    dots_dir: Option<PathBuf>,
    dsocr_dir: Option<PathBuf>,
    keep_tables: bool,
    quiet: bool,
}

fn usage() -> &'static str {
    "usage: notebook [options] <page.png|page.jpg|scan.pdf ...>

  --out DIR              output directory (default ./notebook-out)
  --reader NAME          layout | dots-crop | dsocr-crop  (env NV_NOTEBOOK_READER)
  --cross-check          read every text region twice (layout pass + crop reader)
                         and report per-region agreement
  --figure-pad PX        padding around figure crops in original pixels (default 12)
  --figure-min-frac F    drop figure boxes below this fraction of page area (default 0.0008)
  --crop-pad PX          padding around text-region crops (default 8)
  --crop-min-side PX     upscale text crops whose short side is below this (default 64)
  --max-new-tokens N     decoder cap per call (default 16384 page / 4096 crop)
  --review-threshold F   flag pages scoring below this (default 0.75)
  --pdf-dpi N            rasterization dpi for .pdf inputs via pdftoppm (default 300)
  --dots-dir DIR         dots.ocr checkpoint (default: HF hub snapshot)
  --dsocr-dir DIR        DeepSeek-OCR-2 checkpoint (default: HF hub snapshot)
  --keep-tables          crop Table regions as figures as well as transcribing them
  --quiet                suppress per-page progress
"
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        inputs: Vec::new(),
        out: PathBuf::from("notebook-out"),
        reader: std::env::var("NV_NOTEBOOK_READER")
            .ok()
            .and_then(|v| ReaderKind::parse(&v))
            .unwrap_or(ReaderKind::Layout),
        cross_check: false,
        figure_pad: 12,
        figure_min_frac: 0.0008,
        crop_pad: 8,
        crop_min_side: 64,
        max_new_tokens: 0,
        review_threshold: 0.75,
        pdf_dpi: 300,
        dots_dir: None,
        dsocr_dir: None,
        keep_tables: false,
        quiet: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut next = |name: &str| -> Result<String> {
            it.next()
                .with_context(|| format!("{name} requires a value"))
        };
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            "--out" => a.out = PathBuf::from(next("--out")?),
            "--reader" => {
                let v = next("--reader")?;
                a.reader = ReaderKind::parse(&v).with_context(|| {
                    format!("unknown reader {v:?}; expected layout, dots-crop or dsocr-crop")
                })?;
            }
            "--cross-check" => a.cross_check = true,
            "--figure-pad" => a.figure_pad = next("--figure-pad")?.parse()?,
            "--figure-min-frac" => a.figure_min_frac = next("--figure-min-frac")?.parse()?,
            "--crop-pad" => a.crop_pad = next("--crop-pad")?.parse()?,
            "--crop-min-side" => a.crop_min_side = next("--crop-min-side")?.parse()?,
            "--max-new-tokens" => a.max_new_tokens = next("--max-new-tokens")?.parse()?,
            "--review-threshold" => a.review_threshold = next("--review-threshold")?.parse()?,
            "--pdf-dpi" => a.pdf_dpi = next("--pdf-dpi")?.parse()?,
            "--dots-dir" => a.dots_dir = Some(PathBuf::from(next("--dots-dir")?)),
            "--dsocr-dir" => a.dsocr_dir = Some(PathBuf::from(next("--dsocr-dir")?)),
            "--keep-tables" => a.keep_tables = true,
            "--quiet" => a.quiet = true,
            other if other.starts_with('-') => bail!("unknown flag {other}\n\n{}", usage()),
            other => a.inputs.push(PathBuf::from(other)),
        }
    }
    if a.inputs.is_empty() {
        bail!("no input pages given\n\n{}", usage());
    }
    if a.cross_check && !a.reader.needs_crops() {
        a.reader = ReaderKind::DotsCrop;
    }
    Ok(a)
}

fn hub_snapshot(repo: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo)
        .join("snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries.into_iter().next()
}

fn device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return d;
        }
    }
    Device::Cpu
}

fn decode_rgb_file(path: &Path) -> Result<RgbImage> {
    RgbImage::decode_file(path)
}

fn write_png(path: &Path, img: &RgbImage) -> Result<()> {
    let buf: image::RgbImage =
        image::ImageBuffer::from_raw(img.w as u32, img.h as u32, img.data.clone())
            .context("figure crop buffer has the wrong length")?;
    buf.save(path)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn expand_pdf(path: &Path, dpi: u32, scratch: &Path) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(scratch)?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("page")
        .to_string();
    let prefix = scratch.join(&stem);
    let status = std::process::Command::new("pdftoppm")
        .arg("-r")
        .arg(dpi.to_string())
        .arg("-png")
        .arg(path)
        .arg(&prefix)
        .status()
        .with_context(|| {
            format!(
                "run pdftoppm for {} (install poppler-utils, or pre-rasterize the pdf yourself)",
                path.display()
            )
        })?;
    if !status.success() {
        bail!("pdftoppm failed on {}", path.display());
    }
    let mut out: Vec<PathBuf> = std::fs::read_dir(scratch)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("png")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(&stem))
        })
        .collect();
    out.sort();
    Ok(out)
}

fn clamp_box(b: [f32; 4], w: usize, h: usize, pad: i64) -> Option<(usize, usize, usize, usize)> {
    let x0 = (b[0].min(b[2]).floor() as i64 - pad).clamp(0, w as i64);
    let y0 = (b[1].min(b[3]).floor() as i64 - pad).clamp(0, h as i64);
    let x1 = (b[0].max(b[2]).ceil() as i64 + pad).clamp(0, w as i64);
    let y1 = (b[1].max(b[3]).ceil() as i64 + pad).clamp(0, h as i64);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some((x0 as usize, y0 as usize, x1 as usize, y1 as usize))
}

fn upscale_short_side(img: &RgbImage, min_side: usize) -> (RgbImage, f64) {
    let short = img.w.min(img.h);
    if short == 0 || short >= min_side {
        return (img.clone(), 1.0);
    }
    let scale = (min_side as f64 / short as f64).min(8.0);
    let ow = ((img.w as f64 * scale).round() as usize).max(1);
    let oh = ((img.h as f64 * scale).round() as usize).max(1);
    (
        nv_models::deepseek_ocr::preprocess::resize_rgb(img, ow, oh),
        scale,
    )
}

fn norm_for_compare(s: &str) -> Vec<char> {
    let mut out: Vec<char> = Vec::with_capacity(s.len());
    let mut last_space = true;
    for c in s.chars() {
        let c = c.to_ascii_lowercase();
        if c.is_whitespace() {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
            continue;
        }
        if c.is_alphanumeric() {
            out.push(c);
            last_space = false;
        }
    }
    while out.last() == Some(&' ') {
        out.pop();
    }
    out
}

fn levenshtein(a: &[char], b: &[char]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

fn agreement(a: &str, b: &str) -> f64 {
    let na = norm_for_compare(a);
    let nb = norm_for_compare(b);
    if na.is_empty() && nb.is_empty() {
        return 1.0;
    }
    const CAP: usize = 4000;
    let na = &na[..na.len().min(CAP)];
    let nb = &nb[..nb.len().min(CAP)];
    let d = levenshtein(na, nb) as f64;
    let m = na.len().max(nb.len()) as f64;
    (1.0 - d / m).clamp(0.0, 1.0)
}

fn crop_overran(layout_text: &str, crop_text: &str) -> bool {
    let l = layout_text.trim().chars().count();
    let c = crop_text.trim().chars().count();
    if l == 0 {
        return c > 2000;
    }
    c > 3 * l + 40
}

fn heading_prefix(cat: &str, text: &str) -> &'static str {
    if text.trim_start().starts_with('#') {
        return "";
    }
    match cat {
        "Title" => "# ",
        "Section-header" => "## ",
        _ => "",
    }
}

fn slug(path: &Path, page: usize) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("page")
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>();
    format!("{page:03}-{stem}")
}

struct PageOutcome {
    report: PageReport,
    markdown: String,
}

#[allow(clippy::too_many_arguments)]
fn process_page(
    args: &Args,
    dots: &DotsOcrPipeline,
    reader: Option<&dyn RegionReader>,
    src: &Path,
    page_no: usize,
    out_dir: &Path,
    images_dir: &Path,
) -> Result<PageOutcome> {
    let orig = decode_rgb_file(src)?;
    let (ow, oh) = (orig.w, orig.h);
    let page_area = (ow * oh) as f64;

    let t = Instant::now();
    let page_opts = DotsGenerateOptions {
        max_new_tokens: if args.max_new_tokens > 0 {
            args.max_new_tokens
        } else {
            16384
        },
        ..Default::default()
    };
    let layout = dots.recognize(&orig, DotsMode::LayoutAll, &page_opts)?;
    let layout_ms = t.elapsed().as_secs_f64() * 1e3;

    let mut flags: Vec<String> = Vec::new();
    if layout.looped {
        flags.push("layout-loop".into());
    }
    if layout.page.truncated {
        flags.push("layout-truncated".into());
    }
    if layout.generated_tokens >= page_opts.max_new_tokens {
        flags.push("token-cap".into());
    }
    if layout.page.elements.is_empty() {
        flags.push("no-elements".into());
    }

    let t = Instant::now();
    let mut md = String::new();
    let mut figures: Vec<FigureRecord> = Vec::new();
    let mut regions: Vec<RegionRecord> = Vec::new();
    let mut agreements: Vec<f64> = Vec::new();
    let mut bad_bbox = 0usize;
    let mut unknown_cat = 0usize;
    let mut empty_text = 0usize;
    let mut crop_loops = 0usize;
    let mut overruns = 0usize;
    let mut text_regions = 0usize;
    let mut cat_counts: BTreeMap<String, usize> = BTreeMap::new();
    let base = slug(src, page_no);

    md.push_str(&format!("# Page {page_no}\n\n"));

    for (order, el) in layout.page.elements.iter().enumerate() {
        let cat = el.category.clone().unwrap_or_else(|| "Text".into());
        *cat_counts.entry(cat.clone()).or_default() += 1;
        if !el.category_is_known() {
            unknown_cat += 1;
        }
        let boxed = el.bbox.and_then(|b| clamp_box(b, ow, oh, 0));
        if el.bbox.is_some() && boxed.is_none() {
            bad_bbox += 1;
        }

        let is_figure =
            FIGURE_CATEGORIES.contains(&cat.as_str()) || (args.keep_tables && cat == "Table");

        if is_figure {
            let Some(b) = el.bbox else {
                regions.push(RegionRecord {
                    order,
                    category: cat,
                    bbox: None,
                    chars: 0,
                    agreement: None,
                    note: Some("figure without a bbox; nothing to crop".into()),
                });
                continue;
            };
            let Some((x0, y0, x1, y1)) = clamp_box(b, ow, oh, args.figure_pad) else {
                regions.push(RegionRecord {
                    order,
                    category: cat,
                    bbox: Some(b),
                    chars: 0,
                    agreement: None,
                    note: Some("figure bbox degenerate after clamping".into()),
                });
                continue;
            };
            let area = ((x1 - x0) * (y1 - y0)) as f64;
            if area / page_area < args.figure_min_frac {
                regions.push(RegionRecord {
                    order,
                    category: cat,
                    bbox: Some(b),
                    chars: 0,
                    agreement: None,
                    note: Some(format!(
                        "figure dropped: {:.4}% of page is below --figure-min-frac",
                        100.0 * area / page_area
                    )),
                });
                continue;
            }
            let crop = orig.crop(x0, y0, x1, y1);
            let idx = figures.len() + 1;
            let name = format!("{base}-fig{idx:02}.png");
            write_png(&images_dir.join(&name), &crop)?;
            figures.push(FigureRecord {
                index: idx,
                file: format!("images/{name}"),
                bbox: [x0 as i64, y0 as i64, x1 as i64, y1 as i64],
                width: crop.w,
                height: crop.h,
            });
            md.push_str(&format!(
                "![figure {idx} from page {page_no}](images/{name})\n\n"
            ));
            regions.push(RegionRecord {
                order,
                category: cat,
                bbox: Some(b),
                chars: 0,
                agreement: None,
                note: None,
            });
            continue;
        }

        text_regions += 1;
        let layout_text = el.text.clone().unwrap_or_default();
        let mut chosen = layout_text.trim().to_string();
        let mut agree: Option<f64> = None;
        let mut note: Option<String> = None;

        if let Some(rd) = reader {
            match el.bbox.and_then(|b| clamp_box(b, ow, oh, args.crop_pad)) {
                Some((x0, y0, x1, y1)) => {
                    let crop = orig.crop(x0, y0, x1, y1);
                    let (crop, _) = upscale_short_side(&crop, args.crop_min_side);
                    match rd.read(&crop) {
                        Ok(r) => {
                            if r.looped {
                                crop_loops += 1;
                            }
                            if args.cross_check {
                                agree = Some(agreement(&layout_text, &r.text));
                                agreements.push(agree.unwrap());
                            }
                            if crop_overran(&layout_text, &r.text) {
                                overruns += 1;
                                note = Some(format!(
                                    "crop read ran away ({} chars from a {} char region); kept the layout-pass text",
                                    r.text.trim().chars().count(),
                                    layout_text.trim().chars().count()
                                ));
                            } else if !r.text.trim().is_empty() {
                                chosen = r.text.trim().to_string();
                            } else if !layout_text.trim().is_empty() {
                                note = Some(
                                    "crop reader returned nothing; kept the layout-pass text"
                                        .into(),
                                );
                            }
                        }
                        Err(e) => {
                            note = Some(format!("crop read failed ({e:#}); kept layout-pass text"));
                        }
                    }
                }
                None => {
                    note = Some("no usable bbox for a crop read; kept layout-pass text".into());
                }
            }
        }

        if chosen.trim().is_empty() {
            empty_text += 1;
        } else {
            md.push_str(heading_prefix(&cat, &chosen));
            md.push_str(chosen.trim());
            md.push_str("\n\n");
        }

        regions.push(RegionRecord {
            order,
            category: cat,
            bbox: el.bbox,
            chars: chosen.chars().count(),
            agreement: agree,
            note,
        });
    }

    let read_ms = t.elapsed().as_secs_f64() * 1e3;

    if bad_bbox > 0 {
        flags.push(format!("bad-bbox:{bad_bbox}"));
    }
    if unknown_cat > 0 {
        flags.push(format!("unknown-category:{unknown_cat}"));
    }
    if empty_text > 0 {
        flags.push(format!("empty-text-region:{empty_text}"));
    }
    if crop_loops > 0 {
        flags.push(format!("crop-loop:{crop_loops}"));
    }
    if overruns > 0 {
        flags.push(format!("crop-overrun:{overruns}"));
    }
    if text_regions == 0 && figures.is_empty() && !layout.page.elements.is_empty() {
        flags.push("nothing-extracted".into());
    }

    let mean_agree = if agreements.is_empty() {
        None
    } else {
        Some(agreements.iter().sum::<f64>() / agreements.len() as f64)
    };
    if let Some(m) = mean_agree {
        if m < 0.6 {
            flags.push(format!("two-read-disagreement:{m:.2}"));
        }
    }

    let mut score = 1.0f64;
    if layout.looped {
        score -= 0.5;
    }
    if layout.page.truncated {
        score -= 0.35;
    }
    if layout.generated_tokens >= page_opts.max_new_tokens {
        score -= 0.3;
    }
    if layout.page.elements.is_empty() {
        score -= 0.9;
    }
    let denom = layout.page.elements.len().max(1) as f64;
    score -= 0.5 * (bad_bbox as f64 / denom);
    score -= 0.3 * (unknown_cat as f64 / denom);
    score -= 0.4 * (empty_text as f64 / denom);
    score -= 0.3 * (crop_loops as f64 / denom);
    score -= 0.6 * (overruns as f64 / denom);
    if let Some(m) = mean_agree {
        score -= 0.5 * (1.0 - m);
    }
    let score = score.clamp(0.0, 1.0);
    let needs_review = score < args.review_threshold || !flags.is_empty();

    let md_name = format!("{base}.md");
    std::fs::write(out_dir.join(&md_name), &md)
        .with_context(|| format!("write {}", out_dir.join(&md_name).display()))?;

    Ok(PageOutcome {
        report: PageReport {
            page: page_no,
            source: src.display().to_string(),
            markdown: md_name,
            width: ow,
            height: oh,
            layout_ms,
            read_ms,
            elements: layout.page.elements.len(),
            text_regions,
            figures,
            regions,
            flags,
            score,
            needs_review,
            reader: reader.map(|r| r.name()).unwrap_or("layout").to_string(),
            mean_agreement: mean_agree,
        },
        markdown: md,
    })
}

fn review_markdown(run: &RunReport, threshold: f64) -> String {
    let mut s = String::new();
    s.push_str("# Notebook digitization report\n\n");
    s.push_str(&format!(
        "- reader: `{}`\n- cross-check: {}\n- pages: {}\n- figures extracted: {}\n- pages flagged for review: {} (threshold {threshold})\n- wall: {:.1} s\n\n",
        run.reader, run.cross_check, run.pages_total, run.figures_total, run.pages_needing_review, run.wall_s
    ));
    s.push_str("> ");
    s.push_str(CAVEAT);
    s.push_str("\n\n## Pages\n\n");
    s.push_str("| page | source | score | regions | figures | flags |\n");
    s.push_str("|---|---|---|---|---|---|\n");
    for p in &run.pages {
        let name = Path::new(&p.source)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&p.source);
        s.push_str(&format!(
            "| {} | {} | {:.2}{} | {} | {} | {} |\n",
            p.page,
            name,
            p.score,
            if p.needs_review { " ⚠" } else { "" },
            p.text_regions,
            p.figures.len(),
            if p.flags.is_empty() {
                "-".to_string()
            } else {
                p.flags.join(", ")
            },
        ));
    }
    let flagged: Vec<&PageReport> = run.pages.iter().filter(|p| p.needs_review).collect();
    if !flagged.is_empty() {
        s.push_str("\n## Needs a human look\n\n");
        for p in flagged {
            s.push_str(&format!("### Page {} -- {}\n\n", p.page, p.source));
            s.push_str(&format!(
                "score {:.2}; flags: {}\n\n",
                p.score,
                if p.flags.is_empty() {
                    "-".into()
                } else {
                    p.flags.join(", ")
                }
            ));
            for r in &p.regions {
                if let Some(note) = &r.note {
                    s.push_str(&format!(
                        "- region {} ({}): {}\n",
                        r.order, r.category, note
                    ));
                }
                if let Some(a) = r.agreement {
                    if a < 0.6 {
                        s.push_str(&format!(
                            "- region {} ({}): two reads agree only {:.0}%\n",
                            r.order,
                            r.category,
                            a * 100.0
                        ));
                    }
                }
            }
            s.push('\n');
        }
    }
    s
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let started = Instant::now();

    std::fs::create_dir_all(&args.out)?;
    let images_dir = args.out.join("images");
    std::fs::create_dir_all(&images_dir)?;
    let scratch = args.out.join("_pdf");

    let mut pages: Vec<PathBuf> = Vec::new();
    for input in &args.inputs {
        if input
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("pdf"))
            == Some(true)
        {
            pages.extend(expand_pdf(input, args.pdf_dpi, &scratch)?);
        } else {
            pages.push(input.clone());
        }
    }

    let dots_dir = args
        .dots_dir
        .clone()
        .or_else(|| std::env::var("NV_OCR_DOTS_DIR").ok().map(PathBuf::from))
        .or_else(|| {
            [
                "models--dots-studio--dots.ocr",
                "models--rednote-hilab--dots.ocr",
            ]
            .iter()
            .find_map(|r| hub_snapshot(r))
        })
        .context("no dots.ocr checkpoint: pass --dots-dir or set NV_OCR_DOTS_DIR")?;
    let dev = device();
    if !args.quiet {
        eprintln!(
            "notebook: layout from {} on {:?}, budget {:?}",
            dots_dir.display(),
            dev,
            PixelBudget::from_env()
        );
    }
    let dots = DotsOcrPipeline::load(&dots_dir, &dev).context("load dots.ocr")?;

    let crop_cap = if args.max_new_tokens > 0 {
        args.max_new_tokens
    } else {
        4096
    };
    let reader: Option<Box<dyn RegionReader + '_>> = match args.reader {
        ReaderKind::Layout => None,
        ReaderKind::DotsCrop => Some(Box::new(DotsCropReader {
            pipeline: &dots,
            max_new_tokens: crop_cap,
        })),
        ReaderKind::DsocrCrop => {
            let dir = args
                .dsocr_dir
                .clone()
                .or_else(|| std::env::var("NV_DSOCR_DIR").ok().map(PathBuf::from))
                .or_else(|| hub_snapshot("models--deepseek-ai--DeepSeek-OCR-2"))
                .context("no DeepSeek-OCR-2 checkpoint: pass --dsocr-dir or set NV_DSOCR_DIR")?;
            let pipeline = DeepSeekOcr2Pipeline::load(&dir, &dev, DecoderPrecision::Bf16)
                .context("load DeepSeek-OCR-2")?;
            Some(Box::new(DsocrCropReader {
                pipeline,
                max_new_tokens: crop_cap,
            }))
        }
    };

    let mut reports: Vec<PageReport> = Vec::new();
    let mut combined = String::new();
    for (i, page) in pages.iter().enumerate() {
        let n = i + 1;
        if !args.quiet {
            eprintln!("notebook: page {n}/{} {}", pages.len(), page.display());
        }
        let outcome = process_page(
            &args,
            &dots,
            reader.as_deref(),
            page,
            n,
            &args.out,
            &images_dir,
        )?;
        if !args.quiet {
            eprintln!(
                "  score {:.2} elements {} figures {} layout {:.0} ms read {:.0} ms{}",
                outcome.report.score,
                outcome.report.elements,
                outcome.report.figures.len(),
                outcome.report.layout_ms,
                outcome.report.read_ms,
                if outcome.report.flags.is_empty() {
                    String::new()
                } else {
                    format!(" flags [{}]", outcome.report.flags.join(", "))
                }
            );
        }
        combined.push_str(&outcome.markdown);
        combined.push_str("\n---\n\n");
        reports.push(outcome.report);
    }

    let figures_total: usize = reports.iter().map(|p| p.figures.len()).sum();
    let flagged = reports.iter().filter(|p| p.needs_review).count();
    let run = RunReport {
        reader: reader.map(|r| r.name()).unwrap_or("layout").to_string(),
        cross_check: args.cross_check,
        pages_total: reports.len(),
        pages_needing_review: flagged,
        figures_total,
        wall_s: started.elapsed().as_secs_f64(),
        pages: reports,
        caveat: CAVEAT,
    };

    std::fs::write(args.out.join("notebook.md"), &combined)?;
    std::fs::write(
        args.out.join("report.json"),
        serde_json::to_string_pretty(&run)? + "\n",
    )?;
    std::fs::write(
        args.out.join("review.md"),
        review_markdown(&run, args.review_threshold),
    )?;
    if scratch.is_dir() {
        let _ = std::fs::remove_dir_all(&scratch);
    }

    println!(
        "notebook: {} pages, {} figures, {} flagged for review -> {}",
        run.pages_total,
        run.figures_total,
        run.pages_needing_review,
        args.out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nv_models::dots_ocr::LayoutElement;

    fn el(cat: &str, bbox: Option<[f32; 4]>, text: Option<&str>) -> LayoutElement {
        LayoutElement {
            bbox,
            category: Some(cat.to_string()),
            text: text.map(|s| s.to_string()),
        }
    }

    #[test]
    fn reader_kind_round_trips_and_rejects_junk() {
        assert_eq!(ReaderKind::parse("layout"), Some(ReaderKind::Layout));
        assert_eq!(ReaderKind::parse("dots-crop"), Some(ReaderKind::DotsCrop));
        assert_eq!(ReaderKind::parse("dsocr-crop"), Some(ReaderKind::DsocrCrop));
        assert_eq!(ReaderKind::parse("qwen3-vl"), None);
        assert!(!ReaderKind::Layout.needs_crops());
        assert!(ReaderKind::DotsCrop.needs_crops());
        assert_eq!(ReaderKind::DsocrCrop.as_str(), "dsocr-crop");
    }

    #[test]
    fn crops_come_from_original_pixels_not_the_model_input() {
        let img = RgbImage::from_fn(1000, 800, |x, y| {
            if (200..400).contains(&x) && (100..300).contains(&y) {
                [255, 0, 0]
            } else {
                [255, 255, 255]
            }
        });
        let e = el("Picture", Some([200.0, 100.0, 400.0, 300.0]), None);
        let (x0, y0, x1, y1) = clamp_box(e.bbox.unwrap(), img.w, img.h, 0).unwrap();
        let crop = img.crop(x0, y0, x1, y1);
        assert_eq!((crop.w, crop.h), (200, 200));
        assert_eq!(crop.get(0, 0), [255, 0, 0]);
        assert_eq!(crop.get(199, 199), [255, 0, 0]);
    }

    #[test]
    fn clamp_box_pads_but_stays_inside_the_page() {
        assert_eq!(
            clamp_box([5.0, 5.0, 20.0, 20.0], 100, 100, 10),
            Some((0, 0, 30, 30))
        );
        assert_eq!(
            clamp_box([90.0, 90.0, 99.0, 99.0], 100, 100, 20),
            Some((70, 70, 100, 100))
        );
        assert_eq!(clamp_box([10.0, 10.0, 10.0, 40.0], 100, 100, 0), None);
        assert_eq!(clamp_box([-50.0, -50.0, -10.0, -10.0], 100, 100, 0), None);
    }

    #[test]
    fn clamp_box_normalizes_inverted_corners() {
        assert_eq!(
            clamp_box([40.0, 60.0, 10.0, 20.0], 100, 100, 0),
            Some((10, 20, 40, 60))
        );
    }

    #[test]
    fn upscale_lifts_thin_line_crops_and_leaves_big_ones_alone() {
        let thin = RgbImage::filled(400, 16, [255, 255, 255]);
        let (up, scale) = upscale_short_side(&thin, 64);
        assert_eq!(up.h, 64);
        assert_eq!(up.w, 1600);
        assert!((scale - 4.0).abs() < 1e-9);
        let big = RgbImage::filled(400, 200, [255, 255, 255]);
        let (same, s) = upscale_short_side(&big, 64);
        assert_eq!((same.w, same.h), (400, 200));
        assert_eq!(s, 1.0);
    }

    #[test]
    fn agreement_is_markup_and_case_neutral() {
        assert_eq!(agreement("Hello world", "hello   world"), 1.0);
        assert_eq!(agreement("**Hello** world", "Hello world"), 1.0);
        assert!(agreement("the cat sat", "the dog sat") < 1.0);
        assert!(agreement("the cat sat", "the dog sat") > 0.7);
        assert!(agreement("completely different", "xyz") < 0.3);
        assert_eq!(agreement("", ""), 1.0);
        assert_eq!(agreement("", "abc"), 0.0);
    }

    #[test]
    fn heading_prefixes_follow_the_dots_categories() {
        assert_eq!(heading_prefix("Title", "Lecture 6"), "# ");
        assert_eq!(heading_prefix("Section-header", "Fick's law"), "## ");
        assert_eq!(heading_prefix("Text", "body"), "");
        assert_eq!(heading_prefix("List-item", "1. a"), "");
    }

    #[test]
    fn crop_overrun_catches_the_measured_essay_hallucination() {
        let title = "Lecture 6 - Diffusion";
        let essay = "Lecture 6 - Diffusion\n\n## Introduction\n\n".to_string()
            + &"Diffusion is the process of movement of a substance. ".repeat(40);
        assert!(crop_overran(title, &essay));
        assert!(!crop_overran(title, "Lecture 6 - Diffusion"));
        assert!(!crop_overran(
            "a",
            "a slightly longer but plausible re-read"
        ));
        assert!(!crop_overran(
            "",
            "short crop text with no layout counterpart"
        ));
        assert!(crop_overran("", &"x".repeat(2001)));
    }

    #[test]
    fn heading_prefix_does_not_double_up_on_dots_own_markdown() {
        assert_eq!(heading_prefix("Title", "# Lecture 6 - Diffusion"), "");
        assert_eq!(heading_prefix("Section-header", "  ## already"), "");
    }

    #[test]
    fn figure_category_set_excludes_transcribable_regions() {
        assert!(FIGURE_CATEGORIES.contains(&"Picture"));
        assert!(!FIGURE_CATEGORIES.contains(&"Table"));
        assert!(!FIGURE_CATEGORIES.contains(&"Formula"));
        assert!(el("Picture", None, None).is_picture());
        assert!(!el("Text", None, Some("x")).is_picture());
    }

    #[test]
    fn slug_is_filesystem_safe_and_page_ordered() {
        let s = slug(Path::new("/scans/notebook 1992 (a).jpg"), 7);
        assert_eq!(s, "007-notebook-1992--a-");
        assert!(!s.contains('/'));
        assert!(!s.contains(' '));
        assert!(slug(Path::new("b.png"), 2) > slug(Path::new("a.png"), 1));
    }

    #[test]
    fn levenshtein_matches_known_distances() {
        let a: Vec<char> = "kitten".chars().collect();
        let b: Vec<char> = "sitting".chars().collect();
        assert_eq!(levenshtein(&a, &b), 3);
        assert_eq!(levenshtein(&[], &b), 7);
        assert_eq!(levenshtein(&a, &a), 0);
    }

    #[test]
    fn norm_for_compare_drops_punctuation_and_collapses_space() {
        assert_eq!(
            norm_for_compare("  Hello, *world*! \n\n 42 ")
                .iter()
                .collect::<String>(),
            "hello world 42"
        );
    }
}
