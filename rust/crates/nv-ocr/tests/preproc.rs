use nv_ocr::binarize::{
    binarize_otsu, binarize_sauvola, histogram, ink_is_dark, otsu_threshold, SAUVOLA_K,
    SAUVOLA_WINDOW,
};
use nv_ocr::line::{normalize_line, PAD_COLS};
use nv_ocr::raster::{load, IntegralImage};
use nv_ocr::GreyImage;
use rand_core::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use std::io::Cursor;

fn encode_png(img: &image::RgbImage) -> Vec<u8> {
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(img.clone())
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .unwrap();
    bytes
}

#[test]
fn bt601_luma_pinned() {
    let mut img = image::RgbImage::new(2, 2);
    img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
    img.put_pixel(1, 0, image::Rgb([0, 255, 0]));
    img.put_pixel(0, 1, image::Rgb([0, 0, 255]));
    img.put_pixel(1, 1, image::Rgb([255, 255, 255]));
    let g = load(&encode_png(&img)).unwrap();
    assert_eq!((g.w, g.h), (2, 2));
    assert_eq!(g.data, vec![76, 150, 29, 255]);
}

#[test]
fn integral_matches_bruteforce() {
    let mut rng = Pcg64Mcg::seed_from_u64(0x0c51);
    let g = GreyImage::from_fn(64, 64, |_, _| (rng.next_u64() & 0xff) as u8);
    let integral = IntegralImage::build(&g);
    let mut rects = Vec::new();
    for _ in 0..20 {
        let x0 = (rng.next_u64() % 64) as usize;
        let y0 = (rng.next_u64() % 64) as usize;
        let x1 = x0 + 1 + (rng.next_u64() as usize % (64 - x0));
        let y1 = y0 + 1 + (rng.next_u64() as usize % (64 - y0));
        rects.push((x0, y0, x1, y1));
    }
    rects.push((0, 0, 64, 64));
    for (x0, y0, x1, y1) in rects {
        let mut sum = 0u64;
        let mut sq = 0u64;
        for y in y0..y1 {
            for x in x0..x1 {
                let v = g.get(x, y) as u64;
                sum += v;
                sq += v * v;
            }
        }
        assert_eq!(integral.rect_sum(x0, y0, x1, y1), sum);
        assert_eq!(integral.rect_sq_sum(x0, y0, x1, y1), sq);
    }
}

fn otsu_reference(hist: &[u32; 256]) -> u8 {
    let total: f64 = hist.iter().map(|&n| n as f64).sum();
    let mut best = 0u8;
    let mut best_score = -1.0;
    for t in 0..=255usize {
        let w0: f64 = hist[..=t].iter().map(|&n| n as f64).sum();
        let w1 = total - w0;
        if w0 == 0.0 || w1 == 0.0 {
            continue;
        }
        let m0: f64 = hist[..=t]
            .iter()
            .enumerate()
            .map(|(v, &n)| v as f64 * n as f64)
            .sum::<f64>()
            / w0;
        let m1: f64 = hist[t + 1..]
            .iter()
            .enumerate()
            .map(|(v, &n)| (v + t + 1) as f64 * n as f64)
            .sum::<f64>()
            / w1;
        let score = w0 * w1 * (m0 - m1) * (m0 - m1);
        if score > best_score {
            best_score = score;
            best = t as u8;
        }
    }
    best
}

#[test]
fn otsu_threshold_bimodal_pinned() {
    let mut hist = [0u32; 256];
    hist[10] = 60;
    hist[200] = 40;
    assert_eq!(otsu_threshold(&hist), 10);
    assert_eq!(otsu_reference(&hist), 10);

    let mut hist2 = [0u32; 256];
    hist2[40] = 300;
    hist2[45] = 50;
    hist2[220] = 700;
    hist2[210] = 100;
    assert_eq!(otsu_threshold(&hist2), otsu_reference(&hist2));
}

#[test]
fn otsu_bimodal_image_classifies_exactly() {
    let g = GreyImage::from_fn(100, 10, |x, _| if x < 30 { 40 } else { 220 });
    let bin = binarize_otsu(&g);
    for y in 0..10 {
        for x in 0..100 {
            assert_eq!(bin.get(x, y), x < 30);
        }
    }
}

#[test]
fn otsu_uniform_image_no_ink() {
    let g = GreyImage::from_fn(50, 50, |_, _| 200);
    assert_eq!(binarize_otsu(&g).ink_count(), 0);
    assert_eq!(
        binarize_sauvola(&g, SAUVOLA_WINDOW, SAUVOLA_K).ink_count(),
        0
    );
}

#[test]
fn polarity_inversion_same_ink() {
    let g = GreyImage::from_fn(80, 40, |x, y| {
        if (10..14).contains(&x) && (10..30).contains(&y) {
            20
        } else {
            240
        }
    });
    let inv = g.invert();
    assert!(ink_is_dark(&g));
    assert!(!ink_is_dark(&inv));
    let b1 = binarize_otsu(&g);
    let b2 = binarize_otsu(&inv);
    let s1 = binarize_sauvola(&g, SAUVOLA_WINDOW, SAUVOLA_K);
    let s2 = binarize_sauvola(&inv, SAUVOLA_WINDOW, SAUVOLA_K);
    for y in 0..40 {
        for x in 0..80 {
            assert_eq!(b1.get(x, y), b2.get(x, y));
            assert_eq!(s1.get(x, y), s2.get(x, y));
        }
    }
    assert_eq!(b1.ink_count(), 80);
}

fn gradient_strokes() -> (GreyImage, Vec<(usize, usize)>) {
    let w = 400;
    let h = 80;
    let mut strokes = Vec::new();
    let g = GreyImage::from_fn(w, h, |x, y| {
        let bg = 60 + (x * 195 / (w - 1)) as i32;
        let in_stroke = (20..60).contains(&y) && x >= 20 && (x % 25) < 3;
        if in_stroke {
            (bg - 55).max(0) as u8
        } else {
            bg as u8
        }
    });
    for y in 20..60 {
        for x in 20..w {
            if (x % 25) < 3 {
                strokes.push((x, y));
            }
        }
    }
    (g, strokes)
}

#[test]
fn sauvola_beats_otsu_on_gradient() {
    let (g, strokes) = gradient_strokes();
    let stroke_set: std::collections::HashSet<(usize, usize)> = strokes.iter().copied().collect();
    let sau = binarize_sauvola(&g, SAUVOLA_WINDOW, SAUVOLA_K);
    let ots = binarize_otsu(&g);
    let mut sau_hit = 0usize;
    let mut ots_hit = 0usize;
    for &(x, y) in &strokes {
        if sau.get(x, y) {
            sau_hit += 1;
        }
        if ots.get(x, y) {
            ots_hit += 1;
        }
    }
    let mut sau_bg_err = 0usize;
    let mut ots_bg_err = 0usize;
    let mut bg_total = 0usize;
    for y in 0..g.h {
        for x in 0..g.w {
            if stroke_set.contains(&(x, y)) {
                continue;
            }
            bg_total += 1;
            if sau.get(x, y) {
                sau_bg_err += 1;
            }
            if ots.get(x, y) {
                ots_bg_err += 1;
            }
        }
    }
    assert!(sau_hit as f64 >= 0.95 * strokes.len() as f64);
    assert!(sau_bg_err as f64 <= 0.02 * bg_total as f64);
    let ots_bad =
        ots_bg_err as f64 > 0.10 * bg_total as f64 || (ots_hit as f64) < 0.5 * strokes.len() as f64;
    assert!(ots_bad);
}

#[test]
fn histogram_counts_all_pixels() {
    let mut rng = Pcg64Mcg::seed_from_u64(7);
    let g = GreyImage::from_fn(31, 17, |_, _| (rng.next_u64() & 0xff) as u8);
    let hist = histogram(&g);
    assert_eq!(hist.iter().map(|&n| n as usize).sum::<usize>(), 31 * 17);
}

#[test]
fn normalize_scales_height_and_inverts() {
    let strip = GreyImage::from_fn(40, 20, |x, y| {
        if (8..32).contains(&x) && (4..16).contains(&y) {
            0
        } else {
            255
        }
    });
    let n = normalize_line(&strip, 40);
    assert_eq!(n.h, 40);
    assert_eq!(n.w, 80 + 2 * PAD_COLS);
    for v in &n.data {
        assert!((0.0..=1.0).contains(v));
    }
    assert!(n.at(20, PAD_COLS + 40) > 0.9);
    assert!(n.at(2, PAD_COLS + 2) < 0.1);
    for row in 0..n.h {
        for c in 0..PAD_COLS {
            assert_eq!(n.at(row, c), 0.0);
            assert_eq!(n.at(row, n.w - 1 - c), 0.0);
        }
    }
}

#[test]
fn normalize_polarity_agnostic() {
    let strip = GreyImage::from_fn(60, 30, |x, y| {
        if (10..50).contains(&x) && (8..22).contains(&y) {
            10
        } else {
            245
        }
    });
    let a = normalize_line(&strip, 36);
    let b = normalize_line(&strip.invert(), 36);
    assert_eq!(a.w, b.w);
    let mut max_diff = 0.0f32;
    for i in 0..a.data.len() {
        max_diff = max_diff.max((a.data[i] - b.data[i]).abs());
    }
    assert!(max_diff <= 10.5 / 255.0 + 1e-6);
}

#[test]
fn normalize_blank_and_empty() {
    let blank = GreyImage::from_fn(30, 10, |_, _| 230);
    let n = normalize_line(&blank, 36);
    assert!(n.data.iter().all(|&v| v <= 25.0 / 255.0 + 1e-6));
    let empty = GreyImage::new(0, 0);
    let n2 = normalize_line(&empty, 36);
    assert_eq!(n2.w, 0);
    assert!(n2.data.is_empty());
}
