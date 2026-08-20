use std::path::PathBuf;

use nv_models::deepseek_ocr::{ResolutionMode, RgbImage};
use nv_ocr::ResolutionHint;
use speaches_plus::oapi::ocr::{decide_auto, grey_from_rgb, page_metrics_full, PageMetrics};

fn synthetic_page(w: usize, h: usize, band_px: usize, pitch: usize) -> RgbImage {
    RgbImage::from_fn(w, h, |_x, y| {
        let phase = y % pitch;
        if phase < band_px {
            [20, 20, 20]
        } else {
            [245, 245, 245]
        }
    })
}

fn metrics_of(img: &RgbImage) -> PageMetrics {
    page_metrics_full(&grey_from_rgb(img))
}

#[test]
fn hint_parse_roundtrips() {
    for s in ["auto", "tiled", "base1024", "base768"] {
        assert_eq!(ResolutionHint::parse(s).unwrap().as_str(), s);
    }
    assert_eq!(ResolutionHint::parse("gundam"), Some(ResolutionHint::Tiled));
    assert_eq!(ResolutionHint::parse(""), Some(ResolutionHint::Auto));
    assert_eq!(ResolutionHint::parse("nope"), None);
}

const MEASURED: [(usize, usize, usize, f32, f32, &str); 25] = [
    (1400, 1900, 11, 0.0117, 0.0000, "071 real-05-labnotes"),
    (1400, 1900, 20, 0.0164, 0.0000, "071 real-04-letter"),
    (1400, 1900, 21, 0.0213, 0.0000, "071 real-01-invoice"),
    (1400, 1900, 20, 0.0227, 0.0000, "071 real-03-report"),
    (1400, 1900, 17, 0.0279, 0.0000, "071 real-02-newspaper"),
    (1186, 1677, 45, 0.0534, 0.0017, "scan book_en_Pyomo_016"),
    (
        1457,
        2064,
        49,
        0.0584,
        0.1546,
        "scan book_en_excursion_0596",
    ),
    (1191, 1684, 41, 0.0695, 0.1220, "scan eastmoney_5"),
    (1191, 1684, 38, 0.0707, 0.0442, "scan yanbaor2_whitepaper_6"),
    (1677, 1186, 18, 0.0713, 0.0000, "scan PPT_1001115_011"),
    (1684, 1191, 9, 0.0726, 0.0231, "scan yanbaoPPT_4820"),
    (1191, 1684, 43, 0.0744, 0.2335, "scan jiaocai_2527"),
    (1191, 1684, 71, 0.0880, 0.7188, "scan scihub_ana20363_5"),
    (1677, 1186, 13, 0.0971, 0.1631, "scan yanbao_SE17_10"),
    (1684, 2381, 68, 0.0979, 0.3954, "scan newspaper_bf784"),
    (1684, 2381, 57, 0.1041, 0.8457, "scan newspaper_d4ed0"),
    (
        1684,
        1191,
        22,
        0.1087,
        0.0000,
        "scan PPT_EnglishToAmerican_017",
    ),
    (1191, 1684, 35, 0.1133, 0.1022, "scan notes_1ba14_99"),
    (1191, 1684, 28, 0.1176, 0.2500, "scan notes_1ba14_77"),
    (1191, 1684, 79, 0.1264, 0.4645, "scan scihub_jcrimjus_8"),
    (1684, 1191, 4, 0.1268, 0.9915, "scan jiaocai_377"),
    (1683, 1191, 14, 0.1468, 0.0101, "scan yanbaoPPT_825"),
    (
        1684,
        1191,
        13,
        0.1506,
        0.1591,
        "scan PPT_BaharMartonosi_006",
    ),
    (
        1684,
        1191,
        14,
        0.1533,
        0.0000,
        "scan PPT_english-guidelines_011",
    ),
    (1684, 1191, 5, 0.1899, 0.0104, "scan yanbaoPPT_335"),
];

const GATE_NED: f32 = 0.02;

fn measured_metrics(w: usize, h: usize, bands: usize, ink: f32) -> PageMetrics {
    let scale = (1024.0 / w.max(h) as f32).min(1.0);
    PageMetrics {
        width: w,
        height: h,
        bands,
        band_px: 0.0,
        scale,
        scaled_band_px: 0.0,
        ink_frac: ink,
        ds_stroke_px: 0.0,
        ds_separability: 0.0,
        ds_acutance: 0.0,
    }
}

#[test]
fn auto_never_picks_base1024_on_a_page_it_would_damage() {
    let mut picked = 0usize;
    for (w, h, bands, ink, ab_ned, name) in MEASURED {
        let (mode, why) = decide_auto(&measured_metrics(w, h, bands, ink));
        if mode == ResolutionMode::Base1024 {
            picked += 1;
            assert!(
                ab_ned <= GATE_NED,
                "auto picked base1024 ({why}) on {name}, whose measured base1024-vs-tiled \
                 markup-neutral NED is {ab_ned:.4} (> {GATE_NED})"
            );
        }
    }
    assert_eq!(
        picked, 5,
        "coverage changed: auto should pick base1024 on exactly the 5 low-ink 071 pages"
    );
}

#[test]
fn auto_is_a_no_op_on_real5_scanning() {
    let mut damaged = 0usize;
    let mut scan_picks = 0usize;
    for (w, h, bands, ink, ab_ned, name) in MEASURED {
        if ab_ned > GATE_NED {
            damaged += 1;
        }
        if name.starts_with("scan ")
            && decide_auto(&measured_metrics(w, h, bands, ink)).0 == ResolutionMode::Base1024
        {
            scan_picks += 1;
        }
    }
    assert_eq!(
        damaged, 14,
        "base1024 still damages 14 of the 25 measured pages"
    );
    assert_eq!(
        scan_picks, 0,
        "auto claims no Real5-Scanning page; that is the honest coverage, not a bug"
    );
}

#[test]
fn pages_that_never_tile_pick_base1024() {
    let img = synthetic_page(600, 700, 12, 30);
    let (mode, why) = decide_auto(&metrics_of(&img));
    assert_eq!(mode, ResolutionMode::Base1024);
    assert_eq!(why, "small-page-tiling-is-a-noop");
}

#[test]
fn sparse_text_on_a_large_page_picks_base1024() {
    let img = synthetic_page(2400, 3200, 6, 200);
    let m = metrics_of(&img);
    assert!(m.bands >= 4, "{m:?}");
    assert!(m.ink_frac <= 0.04, "{m:?}");
    let (mode, why) = decide_auto(&m);
    assert_eq!(mode, ResolutionMode::Base1024);
    assert_eq!(why, "sparse-page-survives-downscale");
}

#[test]
fn dense_text_on_a_large_page_keeps_tiling() {
    let img = synthetic_page(3400, 4400, 14, 40);
    let m = metrics_of(&img);
    assert!(m.bands >= 4, "{m:?}");
    assert!(m.ink_frac > 0.04, "{m:?}");
    let (mode, why) = decide_auto(&m);
    assert_eq!(mode, ResolutionMode::Gundam);
    assert_eq!(why, "dense-page-needs-tiles");
}

#[test]
fn blank_large_page_keeps_tiling() {
    let img = RgbImage::from_fn(2000, 3000, |_x, _y| [250, 250, 250]);
    let (mode, why) = decide_auto(&metrics_of(&img));
    assert_eq!(mode, ResolutionMode::Gundam);
    assert_eq!(why, "too-few-text-bands");
}

#[test]
fn band_estimate_tracks_the_drawn_band_height() {
    for band in [10usize, 24, 48] {
        let img = synthetic_page(1600, 2200, band, band * 3);
        let m = metrics_of(&img);
        let err = (m.band_px - band as f32).abs() / band as f32;
        assert!(err <= 0.2, "band={band} estimated={} ({m:?})", m.band_px);
    }
}

fn corpus_images(root: &PathBuf) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(root) else {
        return out;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(corpus_images(&p));
        } else if matches!(
            p.extension().and_then(|s| s.to_str()),
            Some("png") | Some("jpg") | Some("jpeg")
        ) {
            out.push(p);
        }
    }
    out.sort();
    out
}

#[test]
#[ignore]
fn dump_corpus_metrics() {
    let Ok(root) = std::env::var("NV_OCR_POLICY_CORPUS") else {
        eprintln!("SKIP: set NV_OCR_POLICY_CORPUS to an image directory");
        return;
    };
    let images = corpus_images(&PathBuf::from(&root));
    println!("path\tw\th\tbands\tband_px\tscale\tscaled_band_px\tink_frac\tds_stroke\tds_sep\tds_acut\tpick\twhy\tms");
    for p in images {
        let bytes = std::fs::read(&p).unwrap();
        let t = std::time::Instant::now();
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        let (w, h) = (decoded.width() as usize, decoded.height() as usize);
        let img = RgbImage::from_fn(w, h, |x, y| {
            let px = decoded.get_pixel(x as u32, y as u32);
            [px[0], px[1], px[2]]
        });
        let t_metrics = std::time::Instant::now();
        let m = metrics_of(&img);
        let metrics_ms = t_metrics.elapsed().as_secs_f64() * 1e3;
        let (pick, why) = decide_auto(&m);
        let _ = t;
        println!(
            "{}\t{}\t{}\t{}\t{:.1}\t{:.4}\t{:.2}\t{:.4}\t{:.1}\t{:.4}\t{:.4}\t{:?}\t{}\t{:.1}",
            p.display(),
            m.width,
            m.height,
            m.bands,
            m.band_px,
            m.scale,
            m.scaled_band_px,
            m.ink_frac,
            m.ds_stroke_px,
            m.ds_separability,
            m.ds_acutance,
            pick,
            why,
            metrics_ms
        );
    }
}
