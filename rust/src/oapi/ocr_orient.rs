use nv_models::deepseek_ocr::RgbImage;
use nv_ocr::{binarize, GreyImage};

use super::ocr::{downscale_box, grey_from_rgb};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentRotation {
    Upright,
    ApplyCw90,
    Apply180,
    ApplyCcw90,
}

pub const AUTO_ORIENT_ENV: &str = "NV_OCR_AUTO_ORIENT";

const DETECT_LONG_SIDE_PX_PRESERVES_DENSE_LINE_PITCH: usize = 1280;

const POLARITY_MARGIN_KEEPS_CURRENT_ON_TIES: f64 = 1.1;

const MIN_INK_ROWS_FOR_A_VERDICT: usize = 6;

pub fn auto_orient_enabled() -> bool {
    std::env::var(AUTO_ORIENT_ENV).map(|v| v == "1").unwrap_or(false)
}

pub fn rot_cw90_rgb(img: &RgbImage) -> RgbImage {
    let (w, h) = (img.w, img.h);
    let mut out = RgbImage::new(h, w);
    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * 3;
            let dst = (x * h + (h - 1 - y)) * 3;
            out.data[dst..dst + 3].copy_from_slice(&img.data[src..src + 3]);
        }
    }
    out
}

pub fn rot_180_rgb(img: &RgbImage) -> RgbImage {
    let n = img.w * img.h;
    let mut out = RgbImage::new(img.w, img.h);
    for i in 0..n {
        let src = i * 3;
        let dst = (n - 1 - i) * 3;
        out.data[dst..dst + 3].copy_from_slice(&img.data[src..src + 3]);
    }
    out
}

pub fn rot_ccw90_rgb(img: &RgbImage) -> RgbImage {
    rot_180_rgb(&rot_cw90_rgb(img))
}

pub fn apply_rotation_rgb(img: RgbImage, rot: ContentRotation) -> RgbImage {
    match rot {
        ContentRotation::Upright => img,
        ContentRotation::ApplyCw90 => rot_cw90_rgb(&img),
        ContentRotation::Apply180 => rot_180_rgb(&img),
        ContentRotation::ApplyCcw90 => rot_ccw90_rgb(&img),
    }
}

fn rot_cw90_grey(g: &GreyImage) -> GreyImage {
    let (w, h) = (g.w, g.h);
    let mut out = GreyImage::new(h, w);
    for y in 0..h {
        for x in 0..w {
            out.data[x * h + (h - 1 - y)] = g.data[y * w + x];
        }
    }
    out
}

fn rot_180_grey(g: &GreyImage) -> GreyImage {
    let mut out = GreyImage::new(g.w, g.h);
    let n = g.w * g.h;
    for i in 0..n {
        out.data[n - 1 - i] = g.data[i];
    }
    out
}

const PEN_INK_PERCENTILE_EXCLUDES_RULED_GRID_LINES: f64 = 0.06;

const GLYPH_INK_PERCENTILE_KEEPS_TOUCHING_CURSIVE_LINES_APART: f64 = 0.03;

fn ink_threshold_at_percentile(g: &GreyImage, pct: f64) -> (u8, bool) {
    let hist = binarize::histogram(g);
    let otsu = binarize::otsu_threshold(&hist);
    let dark = binarize::ink_is_dark(g);
    let total: u64 = hist.iter().map(|&c| c as u64).sum();
    let budget = (total as f64 * pct) as u64;
    let mut cum = 0u64;
    if dark {
        for v in 0..256usize {
            cum += hist[v] as u64;
            if cum >= budget {
                return ((v as u8).min(otsu), true);
            }
        }
        (otsu, true)
    } else {
        for v in (0..256usize).rev() {
            cum += hist[v] as u64;
            if cum >= budget {
                return ((v as u8).max(otsu), false);
            }
        }
        (otsu, false)
    }
}

const PAGE_BOX_AT_HALF_PAPER_DENSITY_DROPS_THE_DESK_AROUND_A_PHOTOGRAPHED_BOOK: f64 = 0.5;

const PAGE_BOX_UNDER_A_QUARTER_OF_THE_FRAME_IS_NOT_A_PAGE: usize = 4;

fn page_region(g: &GreyImage) -> GreyImage {
    let hist = binarize::histogram(g);
    let otsu = binarize::otsu_threshold(&hist);
    let dark = binarize::ink_is_dark(g);
    let paper = |v: u8| if dark { v > otsu } else { v <= otsu };
    let mut rows = vec![0u32; g.h];
    let mut cols = vec![0u32; g.w];
    for y in 0..g.h {
        for x in 0..g.w {
            if paper(g.data[y * g.w + x]) {
                rows[y] += 1;
                cols[x] += 1;
            }
        }
    }
    let span = |v: &[u32]| {
        let mx = v.iter().copied().max().unwrap_or(0) as f64;
        let cut =
            (mx * PAGE_BOX_AT_HALF_PAPER_DENSITY_DROPS_THE_DESK_AROUND_A_PHOTOGRAPHED_BOOK) as u32;
        let a = v.iter().position(|&c| c > cut).unwrap_or(0);
        let b = v
            .iter()
            .rposition(|&c| c > cut)
            .map(|i| i + 1)
            .unwrap_or(v.len());
        (a, b.max(a + 1))
    };
    let (y0, y1) = span(&rows);
    let (x0, x1) = span(&cols);
    if (x1 - x0) * (y1 - y0) * PAGE_BOX_UNDER_A_QUARTER_OF_THE_FRAME_IS_NOT_A_PAGE < g.w * g.h {
        return g.clone();
    }
    let mut out = GreyImage::new(x1 - x0, y1 - y0);
    for y in y0..y1 {
        let d = (y - y0) * out.w;
        out.data[d..d + out.w].copy_from_slice(&g.data[y * g.w + x0..y * g.w + x1]);
    }
    out
}

struct Blob {
    cx: f64,
    cy: f64,
    span: usize,
    area: f64,
    elongation: f64,
    orientation: f64,
}

fn ink_blobs(g: &GreyImage, pct: f64) -> Vec<Blob> {
    let (thr, dark) = ink_threshold_at_percentile(g, pct);
    let is_ink = |v: u8| if dark { v <= thr } else { v > thr };
    let (w, h) = (g.w, g.h);
    let mut seen = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();
    let mut out = Vec::new();
    for start in 0..w * h {
        if seen[start] || !is_ink(g.data[start]) {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0f64, 0f64, 0f64, 0f64, 0f64);
        let mut n = 0f64;
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
        while let Some(p) = stack.pop() {
            let (x, y) = (p % w, p / w);
            let (fx, fy) = (x as f64, y as f64);
            sx += fx;
            sy += fy;
            sxx += fx * fx;
            syy += fy * fy;
            sxy += fx * fy;
            n += 1.0;
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            for (dx, dy) in [
                (1i32, 0i32),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let q = ny as usize * w + nx as usize;
                if !seen[q] && is_ink(g.data[q]) {
                    seen[q] = true;
                    stack.push(q);
                }
            }
        }
        let (cx, cy) = (sx / n, sy / n);
        let (mxx, myy, mxy) = (sxx / n - cx * cx, syy / n - cy * cy, sxy / n - cx * cy);
        let tr = mxx + myy;
        let root = ((mxx - myy) * (mxx - myy) + 4.0 * mxy * mxy)
            .max(0.0)
            .sqrt();
        let (big, small) = ((tr + root) / 2.0, (tr - root) / 2.0);
        out.push(Blob {
            cx,
            cy,
            span: (x1 - x0 + 1).max(y1 - y0 + 1),
            area: n,
            elongation: if small > 0.05 { big / small } else { f64::MAX },
            orientation: 0.5 * (2.0 * mxy).atan2(mxx - myy),
        });
    }
    out
}

const ALIGNED_WINDOW_DEG_MAKES_UNIFORM_DIRECTIONS_SCORE_ONE_SIXTH: f64 = 15.0;

fn axis_of(dirs: &[f64]) -> (f64, f64) {
    let (mut cs, mut sn) = (0.0f64, 0.0f64);
    for &t in dirs {
        cs += (2.0 * t).cos();
        sn += (2.0 * t).sin();
    }
    let axis = 0.5 * sn.atan2(cs);
    let window = ALIGNED_WINDOW_DEG_MAKES_UNIFORM_DIRECTIONS_SCORE_ONE_SIXTH.to_radians();
    let aligned = dirs
        .iter()
        .filter(|&&t| {
            let d = (t - axis).rem_euclid(std::f64::consts::PI);
            d.min(std::f64::consts::PI - d) <= window
        })
        .count();
    (axis, aligned as f64 / dirs.len().max(1) as f64)
}

const NEIGHBOUR_REACH_IN_GLYPH_SPANS: f64 = 3.0;

const GLYPH_SPAN_OVER_A_TWELFTH_OF_THE_FRAME_IS_A_RULE_NOT_A_GLYPH: usize = 12;

const MIN_GLYPHS_FOR_A_NEIGHBOUR_VERDICT: usize = 12;

fn glyph_neighbour_axis(blobs: &[Blob], frame: usize) -> Option<(f64, f64)> {
    let span_cap = frame / GLYPH_SPAN_OVER_A_TWELFTH_OF_THE_FRAME_IS_A_RULE_NOT_A_GLYPH;
    let glyphs: Vec<&Blob> = blobs
        .iter()
        .filter(|b| b.area >= 4.0 && b.span <= span_cap)
        .collect();
    if glyphs.len() < MIN_GLYPHS_FOR_A_NEIGHBOUR_VERDICT {
        return None;
    }
    let mut spans: Vec<usize> = glyphs.iter().map(|b| b.span).collect();
    spans.sort_unstable();
    let reach = spans[spans.len() / 2] as f64 * NEIGHBOUR_REACH_IN_GLYPH_SPANS;
    if reach < 2.0 {
        return None;
    }
    let minx = glyphs.iter().map(|b| b.cx).fold(f64::MAX, f64::min);
    let miny = glyphs.iter().map(|b| b.cy).fold(f64::MAX, f64::min);
    let gw = ((glyphs.iter().map(|b| b.cx).fold(f64::MIN, f64::max) - minx) / reach) as usize + 1;
    let gh = ((glyphs.iter().map(|b| b.cy).fold(f64::MIN, f64::max) - miny) / reach) as usize + 1;
    let cell_of = |b: &Blob| {
        (
            (((b.cx - minx) / reach) as usize).min(gw - 1),
            (((b.cy - miny) / reach) as usize).min(gh - 1),
        )
    };
    let mut cells: Vec<Vec<usize>> = vec![Vec::new(); gw * gh];
    for (i, b) in glyphs.iter().enumerate() {
        let (bx, by) = cell_of(b);
        cells[by * gw + bx].push(i);
    }
    let mut dirs = Vec::with_capacity(glyphs.len());
    for (i, a) in glyphs.iter().enumerate() {
        let (bx, by) = cell_of(a);
        let mut best = f64::MAX;
        let mut bv = (0.0f64, 0.0f64);
        for cy in by.saturating_sub(1)..=(by + 1).min(gh - 1) {
            for cx in bx.saturating_sub(1)..=(bx + 1).min(gw - 1) {
                for &j in &cells[cy * gw + cx] {
                    if i == j {
                        continue;
                    }
                    let dx = glyphs[j].cx - a.cx;
                    let dy = glyphs[j].cy - a.cy;
                    let d = dx * dx + dy * dy;
                    if d < best {
                        best = d;
                        bv = (dx, dy);
                    }
                }
            }
        }
        if best < 1.0 || best > reach * reach {
            continue;
        }
        dirs.push(bv.1.atan2(bv.0));
    }
    if dirs.len() < MIN_GLYPHS_FOR_A_NEIGHBOUR_VERDICT {
        return None;
    }
    Some(axis_of(&dirs))
}

const RULE_ELONGATION_MIN_SEPARATES_A_LINE_FROM_A_GLYPH: f64 = 4.0;

const RULE_MIN_AREA_STOPS_SPECKLE_FROM_VOTING_A_DEGENERATE_ORIENTATION: f64 = 8.0;

fn rule_elongation_axis(blobs: &[Blob]) -> Option<(f64, f64)> {
    let rules: Vec<f64> = blobs
        .iter()
        .filter(|b| {
            b.area >= RULE_MIN_AREA_STOPS_SPECKLE_FROM_VOTING_A_DEGENERATE_ORIENTATION
                && b.elongation >= RULE_ELONGATION_MIN_SEPARATES_A_LINE_FROM_A_GLYPH
        })
        .map(|b| b.orientation)
        .collect();
    if rules.len() < MIN_GLYPHS_FOR_A_NEIGHBOUR_VERDICT {
        return None;
    }
    Some(axis_of(&rules))
}

const ALIGNED_FLOOR_ADMITS_CURSIVE_AT_0_30_REJECTS_ENGRAVING_TEXTURE_AT_0_13: f64 = 0.20;

const RULE_VOTE_FLOOR_TRUSTS_ONLY_A_NEAR_UNANIMOUS_ELONGATION: f64 = 0.80;

fn text_line_axis_deg(g: &GreyImage) -> Option<f64> {
    let frame = g.w.max(g.h);
    let blobs = ink_blobs(g, GLYPH_INK_PERCENTILE_KEEPS_TOUCHING_CURSIVE_LINES_APART);
    if let Some((axis, aligned)) = glyph_neighbour_axis(&blobs, frame) {
        if aligned >= ALIGNED_FLOOR_ADMITS_CURSIVE_AT_0_30_REJECTS_ENGRAVING_TEXTURE_AT_0_13 {
            return Some(axis.to_degrees().rem_euclid(180.0));
        }
    }
    let (axis, aligned) = rule_elongation_axis(&blobs)?;
    (aligned >= RULE_VOTE_FLOOR_TRUSTS_ONLY_A_NEAR_UNANIMOUS_ELONGATION)
        .then(|| axis.to_degrees().rem_euclid(180.0))
}

fn margin_spreads(g: &GreyImage) -> Option<(f64, f64)> {
    let (thr, dark) = ink_threshold_at_percentile(g, PEN_INK_PERCENTILE_EXCLUDES_RULED_GRID_LINES);
    let is_ink = |v: u8| if dark { v <= thr } else { v > thr };
    let rows: Vec<u32> = (0..g.h)
        .map(|y| {
            g.data[y * g.w..(y + 1) * g.w]
                .iter()
                .filter(|&&v| is_ink(v))
                .count() as u32
        })
        .collect();
    let cut = rows.iter().copied().max().unwrap_or(0) / 8;
    let mut firsts = Vec::new();
    let mut lasts = Vec::new();
    for y in 0..g.h {
        if rows[y] <= cut || rows[y] == 0 {
            continue;
        }
        let row = &g.data[y * g.w..(y + 1) * g.w];
        if let Some(first) = row.iter().position(|&v| is_ink(v)) {
            let last = row.iter().rposition(|&v| is_ink(v)).unwrap_or(first);
            firsts.push(first as f64);
            lasts.push(last as f64);
        }
    }
    if firsts.len() < MIN_INK_ROWS_FOR_A_VERDICT {
        return None;
    }
    let spread = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
    };
    Some((spread(&firsts), spread(&lasts)))
}

fn looks_left_aligned(g: &GreyImage) -> Option<bool> {
    let (first, last) = margin_spreads(g)?;
    if first * POLARITY_MARGIN_KEEPS_CURRENT_ON_TIES < last {
        return Some(true);
    }
    if last * POLARITY_MARGIN_KEEPS_CURRENT_ON_TIES < first {
        return Some(false);
    }
    None
}

const AXIS_DEADBAND_DEG_KEEPS_UPRIGHT_ON_TIES: f64 = 10.0;

pub fn detect_content_rotation(grey: &GreyImage) -> ContentRotation {
    let page = page_region(grey);
    let long = page.w.max(page.h).max(1);
    let scale = (DETECT_LONG_SIDE_PX_PRESERVES_DENSE_LINE_PITCH as f32 / long as f32).min(1.0);
    let g = downscale_box(&page, scale);

    let lines_run_down_the_page = match text_line_axis_deg(&g) {
        Some(deg) => (deg - 90.0).abs() < 45.0 - AXIS_DEADBAND_DEG_KEEPS_UPRIGHT_ON_TIES,
        None => false,
    };

    if lines_run_down_the_page {
        match looks_left_aligned(&rot_cw90_grey(&g)) {
            Some(false) => ContentRotation::ApplyCcw90,
            _ => ContentRotation::ApplyCw90,
        }
    } else {
        match looks_left_aligned(&g) {
            Some(false) => ContentRotation::Apply180,
            _ => ContentRotation::Upright,
        }
    }
}

pub fn reencode_rotated(bytes: &[u8], rot: ContentRotation) -> anyhow::Result<Vec<u8>> {
    let img = nv_imgdec::decode_rgb8(bytes)?;
    let rotated = match rot {
        ContentRotation::Upright => img,
        ContentRotation::ApplyCw90 => nv_imgdec::image::imageops::rotate90(&img),
        ContentRotation::Apply180 => nv_imgdec::image::imageops::rotate180(&img),
        ContentRotation::ApplyCcw90 => nv_imgdec::image::imageops::rotate270(&img),
    };
    let mut out = std::io::Cursor::new(Vec::new());
    rotated.write_to(&mut out, nv_imgdec::image::ImageFormat::Png)?;
    Ok(out.into_inner())
}

pub fn maybe_auto_orient(img: RgbImage) -> RgbImage {
    if !auto_orient_enabled() {
        return img;
    }
    let rot = detect_content_rotation(&grey_from_rgb(&img));
    if rot != ContentRotation::Upright {
        tracing::info!(?rot, w = img.w, h = img.h, "ocr auto-orient rotating content");
    }
    apply_rotation_rgb(img, rot)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_left_aligned_page() -> GreyImage {
        let (w, h) = (200usize, 160usize);
        let mut g = GreyImage::new(w, h);
        g.data.fill(245);
        let mut seed = 0x243f_6a88u32;
        for line in 0..14 {
            let y0 = 10 + line * 10;
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let len = 90 + (seed >> 24) as usize % 90;
            for dy in 0..3 {
                for x in 8..8 + len.min(w - 16) {
                    g.data[(y0 + dy) * w + x] = 20;
                }
            }
        }
        g
    }

    fn as_rgb(g: &GreyImage) -> RgbImage {
        RgbImage::from_fn(g.w, g.h, |x, y| {
            let v = g.data[y * g.w + x];
            [v, v, v]
        })
    }

    #[test]
    fn upright_page_is_left_alone() {
        assert_eq!(
            detect_content_rotation(&synth_left_aligned_page()),
            ContentRotation::Upright
        );
    }

    #[test]
    fn every_rotation_is_detected_and_undone() {
        let upright = synth_left_aligned_page();
        let cases = [
            (rot_cw90_grey(&upright), ContentRotation::ApplyCcw90),
            (rot_180_grey(&upright), ContentRotation::Apply180),
            (rot_cw90_grey(&rot_180_grey(&upright)), ContentRotation::ApplyCw90),
        ];
        for (rotated, expect) in cases {
            assert_eq!(
                detect_content_rotation(&rotated),
                expect,
                "a page rotated so that applying {expect:?} restores it must be detected"
            );
        }
    }

    #[test]
    fn rotate_field_reencode_round_trips_dimensions() {
        let img = as_rgb(&synth_left_aligned_page());
        let mut png = std::io::Cursor::new(Vec::new());
        nv_imgdec::image::RgbImage::from_raw(img.w as u32, img.h as u32, img.data.clone())
            .unwrap()
            .write_to(&mut png, nv_imgdec::image::ImageFormat::Png)
            .unwrap();
        let rotated = reencode_rotated(&png.into_inner(), ContentRotation::ApplyCw90)
            .expect("reencode cw90");
        let back = nv_imgdec::decode_rgb8(&rotated).expect("decode rotated png");
        assert_eq!(
            (back.width() as usize, back.height() as usize),
            (img.h, img.w),
            "cw90 must swap dimensions in the re-encoded image"
        );
    }

    #[test]
    fn rgb_rotations_round_trip() {
        let img = as_rgb(&synth_left_aligned_page());
        let back = rot_ccw90_rgb(&rot_cw90_rgb(&img));
        assert_eq!(back.data, img.data, "cw then ccw must be the identity");
        let back2 = rot_180_rgb(&rot_180_rgb(&img));
        assert_eq!(back2.data, img.data, "180 twice must be the identity");
    }

    const UPRIGHT_FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../conformance/fixtures");

    const UPRIGHT_FIXTURE_PAGES_LEFT_EXACTLY_ALONE: usize = 15;

    fn grey_from_png(path: &std::path::Path) -> GreyImage {
        grey_from_rgb(&RgbImage::decode_file(path).expect("decode fixture page"))
    }

    #[test]
    fn upright_conformance_pages_are_never_given_a_quarter_turn() {
        let mut pages: Vec<std::path::PathBuf> = std::fs::read_dir(UPRIGHT_FIXTURES)
            .expect("conformance fixtures dir")
            .filter_map(|e| e.ok().map(|e| e.path().join("input.png")))
            .filter(|p| p.exists())
            .collect();
        pages.sort();
        assert!(
            pages.len() >= 16,
            "the upright regression needs the 070/071 fixture pages, found {}",
            pages.len()
        );
        let mut left_alone = 0usize;
        for page in &pages {
            let rot = detect_content_rotation(&grey_from_png(page));
            assert!(
                matches!(rot, ContentRotation::Upright | ContentRotation::Apply180),
                "an upright conformance page must never be turned sideways, {page:?} got {rot:?}"
            );
            left_alone += (rot == ContentRotation::Upright) as usize;
        }
        assert!(
            left_alone >= UPRIGHT_FIXTURE_PAGES_LEFT_EXACTLY_ALONE,
            "only {left_alone} of {} upright pages were left exactly alone; the 180 polarity \
             test is the weak half of the detector and this is its regression bar",
            pages.len()
        );
    }

    #[test]
    fn orient_corpus_sweep_when_pointed_at_real_photos() {
        let Ok(dir) = std::env::var("NV_OCR_ORIENT_CORPUS") else {
            return;
        };
        let mut correct = 0usize;
        let mut total = 0usize;
        for entry in std::fs::read_dir(&dir).expect("read corpus dir") {
            let path = entry.expect("entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("jpeg") {
                continue;
            }
            let bytes = std::fs::read(&path).expect("read photo");
            let img = RgbImage::decode(&bytes).expect("decode photo");
            let rot = detect_content_rotation(&grey_from_rgb(&img));
            total += 1;
            correct += (rot == ContentRotation::ApplyCw90) as usize;
            eprintln!(
                "orient {:?} -> {rot:?} ({}x{})",
                path.file_name().unwrap(),
                img.w,
                img.h
            );
        }
        eprintln!("orient corpus: {correct}/{total} detected as ApplyCw90");
        assert!(total > 0, "corpus dir had no jpeg files");
        assert!(
            correct * 9 >= total * 8,
            "the handwritten-journal corpus is entirely ApplyCw90 ground truth and the \
             detector must recover at least 8 in 9, got {correct}/{total}"
        );
    }
}
