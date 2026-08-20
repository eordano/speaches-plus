use std::sync::OnceLock;

use anyhow::Result;

fn prep_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("NV_DSOCR_PREP_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
                    .min(16)
            })
    })
}

fn par_rows<T, F>(buf: &mut [T], row_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let rows = buf.len().checked_div(row_len).unwrap_or(0);
    let nt = prep_threads().min(rows).max(1);
    if nt == 1 || rows <= 1 {
        for (y, r) in buf.chunks_mut(row_len).enumerate() {
            f(y, r);
        }
        return;
    }
    let per = rows.div_ceil(nt);
    std::thread::scope(|s| {
        for (ci, chunk) in buf.chunks_mut(per * row_len).enumerate() {
            let fr = &f;
            s.spawn(move || {
                for (j, r) in chunk.chunks_mut(row_len).enumerate() {
                    fr(ci * per + j, r);
                }
            });
        }
    });
}

#[derive(Clone, Debug)]
pub struct RgbImage {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl RgbImage {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            data: vec![0u8; w * h * 3],
        }
    }

    pub fn decode(bytes: &[u8]) -> anyhow::Result<Self> {
        let img = nv_imgdec::decode_rgb8(bytes)?;
        let (w, h) = (img.width() as usize, img.height() as usize);
        Ok(Self {
            w,
            h,
            data: img.into_raw(),
        })
    }

    pub fn decode_file(path: &std::path::Path) -> anyhow::Result<Self> {
        use anyhow::Context as _;
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Self::decode(&bytes).with_context(|| format!("decode {}", path.display()))
    }

    pub fn filled(w: usize, h: usize, rgb: [u8; 3]) -> Self {
        let mut data = Vec::with_capacity(w * h * 3);
        for _ in 0..w * h {
            data.extend_from_slice(&rgb);
        }
        Self { w, h, data }
    }

    pub fn from_fn(w: usize, h: usize, mut f: impl FnMut(usize, usize) -> [u8; 3]) -> Self {
        let mut img = Self::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let p = f(x, y);
                let o = (y * w + x) * 3;
                img.data[o..o + 3].copy_from_slice(&p);
            }
        }
        img
    }

    pub fn get(&self, x: usize, y: usize) -> [u8; 3] {
        let o = (y * self.w + x) * 3;
        [self.data[o], self.data[o + 1], self.data[o + 2]]
    }

    pub fn crop(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> RgbImage {
        let mut out = RgbImage::new(x1 - x0, y1 - y0);
        for y in y0..y1 {
            let src = &self.data[(y * self.w + x0) * 3..(y * self.w + x1) * 3];
            let dy = y - y0;
            out.data[dy * out.w * 3..(dy + 1) * out.w * 3].copy_from_slice(src);
        }
        out
    }
}

fn cubic_filter(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

fn precompute_coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ss = 1.0 / filterscale;
    let mut out = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = (((center + support + 0.5) as i64) as usize).min(in_size);
        let mut k: Vec<f64> = (xmin..xmax)
            .map(|x| cubic_filter((x as f64 + 0.5 - center) * ss))
            .collect();
        let ww: f64 = k.iter().sum();
        if ww != 0.0 {
            for v in &mut k {
                *v /= ww;
            }
        }
        out.push((xmin, k));
    }
    out
}

pub fn resize_plane_f32(src: &[f32], w: usize, h: usize, ow: usize, oh: usize) -> Vec<f32> {
    let hc = precompute_coeffs(w, ow);
    let vc = precompute_coeffs(h, oh);
    let mut tmp = vec![0f32; ow * h];
    for y in 0..h {
        for (x, (xmin, k)) in hc.iter().enumerate() {
            let mut acc = 0f64;
            for (i, kv) in k.iter().enumerate() {
                acc += src[y * w + xmin + i] as f64 * kv;
            }
            tmp[y * ow + x] = acc as f32;
        }
    }
    let mut out = vec![0f32; ow * oh];
    for (y, (ymin, k)) in vc.iter().enumerate() {
        for x in 0..ow {
            let mut acc = 0f64;
            for (i, kv) in k.iter().enumerate() {
                acc += tmp[(ymin + i) * ow + x] as f64 * kv;
            }
            out[y * ow + x] = acc as f32;
        }
    }
    out
}

pub fn resize_rgb(img: &RgbImage, ow: usize, oh: usize) -> RgbImage {
    if ow == img.w && oh == img.h {
        return img.clone();
    }
    let hc = precompute_coeffs(img.w, ow);
    let vc = precompute_coeffs(img.h, oh);
    let clip = |v: f64| -> u8 { (v + 0.5).floor().clamp(0.0, 255.0) as u8 };
    let mut tmp = vec![0u8; ow * img.h * 3];
    par_rows(&mut tmp, ow * 3, |y, row| {
        for (x, (xmin, k)) in hc.iter().enumerate() {
            for c in 0..3 {
                let mut acc = 0f64;
                for (i, kv) in k.iter().enumerate() {
                    acc += img.data[(y * img.w + xmin + i) * 3 + c] as f64 * kv;
                }
                row[x * 3 + c] = clip(acc);
            }
        }
    });
    let mut out = RgbImage::new(ow, oh);
    par_rows(&mut out.data, ow * 3, |y, row| {
        let (ymin, k) = &vc[y];
        for x in 0..ow {
            for c in 0..3 {
                let mut acc = 0f64;
                for (i, kv) in k.iter().enumerate() {
                    acc += tmp[((ymin + i) * ow + x) * 3 + c] as f64 * kv;
                }
                row[x * 3 + c] = clip(acc);
            }
        }
    });
    out
}

pub fn round_half_even(v: f64) -> i64 {
    let f = v.floor();
    let d = v - f;
    let up = d > 0.5 || (d == 0.5 && (f as i64) % 2 != 0);
    if up {
        f as i64 + 1
    } else {
        f as i64
    }
}

pub fn letterbox_pad(img: &RgbImage, size: usize, fill: [u8; 3]) -> RgbImage {
    let im_ratio = img.w as f64 / img.h as f64;
    let (mut tw, mut th) = (size, size);
    if (im_ratio - 1.0).abs() > f64::EPSILON {
        if im_ratio > 1.0 {
            let nh = round_half_even(img.h as f64 / img.w as f64 * size as f64).max(1) as usize;
            if nh != size {
                th = nh;
            }
        } else {
            let nw = round_half_even(img.w as f64 / img.h as f64 * size as f64).max(1) as usize;
            if nw != size {
                tw = nw;
            }
        }
    }
    let resized = resize_rgb(img, tw, th);
    let mut out = RgbImage::filled(size, size, fill);
    let x0 = (size - tw) / 2;
    let y0 = (size - th) / 2;
    for y in 0..th {
        let dst = ((y + y0) * size + x0) * 3;
        out.data[dst..dst + tw * 3].copy_from_slice(&resized.data[y * tw * 3..(y + 1) * tw * 3]);
    }
    out
}

pub fn rotate_rgb(img: &RgbImage, deg: f64, fill: [u8; 3]) -> RgbImage {
    if deg == 0.0 {
        return img.clone();
    }
    let rad = deg.to_radians();
    let (s, c) = (rad.sin(), rad.cos());
    let (w, h) = (img.w as f64, img.h as f64);
    let ow = (w * c.abs() + h * s.abs()).round().max(1.0) as usize;
    let oh = (w * s.abs() + h * c.abs()).round().max(1.0) as usize;
    let (cx, cy) = ((w - 1.0) / 2.0, (h - 1.0) / 2.0);
    let (ocx, ocy) = ((ow as f64 - 1.0) / 2.0, (oh as f64 - 1.0) / 2.0);
    let mut out = RgbImage::filled(ow, oh, fill);
    for oy in 0..oh {
        let dy = oy as f64 - ocy;
        for ox in 0..ow {
            let dx = ox as f64 - ocx;
            let sx = c * dx + s * dy + cx;
            let sy = -s * dx + c * dy + cy;
            if sx < 0.0 || sy < 0.0 {
                continue;
            }
            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            if x0 + 1 >= img.w || y0 + 1 >= img.h {
                if x0 < img.w && y0 < img.h {
                    let p = img.get(x0, y0);
                    let o = (oy * ow + ox) * 3;
                    out.data[o..o + 3].copy_from_slice(&p);
                }
                continue;
            }
            let fx = sx - x0 as f64;
            let fy = sy - y0 as f64;
            let o = (oy * ow + ox) * 3;
            let i00 = (y0 * img.w + x0) * 3;
            let i10 = i00 + 3;
            let i01 = i00 + img.w * 3;
            let i11 = i01 + 3;
            for ch in 0..3 {
                let a = img.data[i00 + ch] as f64 * (1.0 - fx) + img.data[i10 + ch] as f64 * fx;
                let b = img.data[i01 + ch] as f64 * (1.0 - fx) + img.data[i11 + ch] as f64 * fx;
                out.data[o + ch] = (a * (1.0 - fy) + b * fy + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

pub fn border_fill(img: &RgbImage) -> [u8; 3] {
    let mut ch: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let step = (img.w.max(img.h) / 256).max(1);
    let push = |p: [u8; 3], ch: &mut [Vec<u8>; 3]| {
        for c in 0..3 {
            ch[c].push(p[c]);
        }
    };
    for x in (0..img.w).step_by(step) {
        push(img.get(x, 0), &mut ch);
        push(img.get(x, img.h - 1), &mut ch);
    }
    for y in (0..img.h).step_by(step) {
        push(img.get(0, y), &mut ch);
        push(img.get(img.w - 1, y), &mut ch);
    }
    let mut out = [0u8; 3];
    for c in 0..3 {
        ch[c].sort_unstable();
        out[c] = ch[c][ch[c].len() / 2];
    }
    out
}

fn gray_downsample(img: &RgbImage, target_long: usize) -> (Vec<f32>, usize, usize) {
    let long = img.w.max(img.h);
    let f = (long as f64 / target_long as f64).ceil().max(1.0) as usize;
    let (dw, dh) = ((img.w / f).max(1), (img.h / f).max(1));
    let mut out = vec![0f32; dw * dh];
    let inv = 1.0 / (f * f) as f32;
    for oy in 0..dh {
        for ox in 0..dw {
            let mut acc = 0f32;
            for y in oy * f..(oy + 1) * f {
                let row = y * img.w * 3;
                for x in ox * f..(ox + 1) * f {
                    let o = row + x * 3;
                    acc += img.data[o] as f32 * 0.299
                        + img.data[o + 1] as f32 * 0.587
                        + img.data[o + 2] as f32 * 0.114;
                }
            }
            out[oy * dw + ox] = acc * inv;
        }
    }
    (out, dw, dh)
}

fn ink_mask(gray: &[f32], w: usize, h: usize) -> Vec<f32> {
    let r = (w.min(h) / 40).clamp(2, 24) as i64;
    let mut sat = vec![0f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0f64;
        for x in 0..w {
            row += gray[y * w + x] as f64;
            sat[(y + 1) * (w + 1) + x + 1] = sat[y * (w + 1) + x + 1] + row;
        }
    }
    let area = |x0: i64, y0: i64, x1: i64, y1: i64| -> (f64, f64) {
        let x0 = x0.clamp(0, w as i64) as usize;
        let y0 = y0.clamp(0, h as i64) as usize;
        let x1 = x1.clamp(0, w as i64) as usize;
        let y1 = y1.clamp(0, h as i64) as usize;
        let n = ((x1 - x0) * (y1 - y0)) as f64;
        let s = sat[y1 * (w + 1) + x1] - sat[y0 * (w + 1) + x1] - sat[y1 * (w + 1) + x0]
            + sat[y0 * (w + 1) + x0];
        (s, n.max(1.0))
    };
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let (s, n) = area(
                x as i64 - r,
                y as i64 - r,
                x as i64 + r + 1,
                y as i64 + r + 1,
            );
            let local = (s / n) as f32;
            let v = local - gray[y * w + x] - 6.0;
            out[y * w + x] = if v > 0.0 { v } else { 0.0 };
        }
    }
    out
}

const SKEW_MAX_DEG: f64 = 20.0;
const SKEW_DEAD_ZONE_DEG: f64 = 0.30;
const SKEW_MIN_GAIN: f64 = 1.02;
const COVERAGE_FLOOR: f64 = 0.3;

fn projection_score(ink: &[f32], w: usize, h: usize, deg: f64) -> f64 {
    let rad = deg.to_radians();
    let (s, c) = (rad.sin(), rad.cos());
    let cx = (w as f64 - 1.0) / 2.0;
    let off = (w as f64 - 1.0) * s.abs() / 2.0 + 1.0;
    let n = ((w as f64 - 1.0) * s.abs() + (h as f64 - 1.0) * c.abs()).ceil() as usize + 3;
    let mut prof = vec![0f64; n];
    let mut cov = vec![0f64; n];
    for y in 0..h {
        let yc = y as f64 * c + off;
        for x in 0..w {
            let v = ink[y * w + x];
            if v <= 0.0 {
                continue;
            }
            let b = yc - (x as f64 - cx) * s;
            let bi = b.floor();
            let fr = b - bi;
            let bi = bi as usize;
            prof[bi] += v as f64 * (1.0 - fr);
            prof[bi + 1] += v as f64 * fr;
        }
    }
    for y in 0..h {
        let yc = y as f64 * c + off;
        for x in (0..w).step_by(4) {
            let b = yc - (x as f64 - cx) * s;
            cov[b.floor() as usize] += 1.0;
        }
    }
    let cmax = cov.iter().copied().fold(0.0f64, f64::max);
    if cmax <= 0.0 {
        return 0.0;
    }
    let floor = COVERAGE_FLOOR * cmax;
    let mut s1 = 0f64;
    let mut s2 = 0f64;
    let mut nv = 0usize;
    for b in 0..n {
        if cov[b] < floor {
            continue;
        }
        let d = prof[b] / cov[b];
        s1 += d;
        s2 += d * d;
        nv += 1;
    }
    if s1 <= 0.0 || nv == 0 {
        return 0.0;
    }
    nv as f64 * s2 / (s1 * s1)
}

pub fn estimate_skew_deg(img: &RgbImage) -> (f64, f64) {
    let (gray, w, h) = gray_downsample(img, 700);
    if w < 64 || h < 64 {
        return (0.0, 1.0);
    }
    let ink = ink_mask(&gray, w, h);
    let score = |deg: f64| projection_score(&ink, w, h, deg);
    let base = score(0.0);
    if base <= 0.0 {
        return (0.0, 1.0);
    }
    let mut best = 0.0f64;
    let mut best_score = base;
    let scan = |from: f64, to: f64, step: f64, best: &mut f64, best_score: &mut f64| {
        let n = ((to - from) / step).round() as i64;
        for i in 0..=n {
            let deg = from + i as f64 * step;
            if deg.abs() > SKEW_MAX_DEG {
                continue;
            }
            let s = score(deg);
            if s > *best_score {
                *best_score = s;
                *best = deg;
            }
        }
    };
    scan(-SKEW_MAX_DEG, SKEW_MAX_DEG, 0.5, &mut best, &mut best_score);
    let c = best;
    scan(c - 0.5, c + 0.5, 0.1, &mut best, &mut best_score);
    let c = best;
    scan(c - 0.1, c + 0.1, 0.02, &mut best, &mut best_score);
    (best, best_score / base)
}

pub fn deskew(img: &RgbImage) -> (RgbImage, f64) {
    let (deg, gain) = estimate_skew_deg(img);
    if deg.abs() < SKEW_DEAD_ZONE_DEG || gain < SKEW_MIN_GAIN {
        return (img.clone(), 0.0);
    }
    (rotate_rgb(img, -deg, border_fill(img)), deg)
}

pub fn select_tile_grid(
    w: usize,
    h: usize,
    min_num: usize,
    max_num: usize,
    tile: usize,
) -> (usize, usize) {
    let aspect = w as f64 / h as f64;
    let mut ratios: Vec<(usize, usize)> = Vec::new();
    for i in 1..=max_num {
        for j in 1..=max_num {
            let p = i * j;
            if p >= min_num && p <= max_num {
                ratios.push((i, j));
            }
        }
    }
    ratios.sort_by_key(|&(i, j)| (i * j, i, j));
    let mut best = (1, 1);
    let mut best_diff = f64::INFINITY;
    let area = (w * h) as f64;
    for &(i, j) in &ratios {
        let diff = (aspect - i as f64 / j as f64).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (i, j);
        } else if diff == best_diff && area > 0.5 * (tile * tile * i * j) as f64 {
            best = (i, j);
        }
    }
    best
}

pub fn dynamic_tiles(img: &RgbImage, tile: usize) -> (Vec<RgbImage>, (usize, usize)) {
    let max_num = gundam_max_tiles();
    let (gx, gy) = select_tile_grid(img.w, img.h, max_num.min(2), max_num, tile);
    let resized = resize_rgb(img, tile * gx, tile * gy);
    let mut tiles = Vec::with_capacity(gx * gy);
    for idx in 0..gx * gy {
        let cx = (idx % gx) * tile;
        let cy = (idx / gx) * tile;
        tiles.push(resized.crop(cx, cy, cx + tile, cy + tile));
    }
    (tiles, (gx, gy))
}

fn norm_lut() -> &'static [f32; 256] {
    static L: OnceLock<[f32; 256]> = OnceLock::new();
    L.get_or_init(|| {
        let mut t = [0f32; 256];
        for (v, slot) in t.iter_mut().enumerate() {
            *slot = v as f32 / 255.0 * 2.0 - 1.0;
        }
        t
    })
}

pub fn normalize_chw(img: &RgbImage) -> Vec<f32> {
    let n = img.w * img.h;
    let lut = norm_lut();
    let mut out = vec![0f32; 3 * n];
    par_rows(&mut out, n, |c, plane| {
        for (i, slot) in plane.iter_mut().enumerate() {
            *slot = lut[img.data[i * 3 + c] as usize];
        }
    });
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolutionMode {
    Gundam,
    Base1024,
    Base768,
}

pub const GLOBAL_SIZE: usize = 1024;
pub const TILE_SIZE: usize = 768;
pub const PAD_FILL: [u8; 3] = [127, 127, 127];

pub fn num_queries(size: usize) -> usize {
    ((size / 16) as f64 / 4.0).ceil() as usize
}

#[derive(Clone, Debug)]
pub struct PreparedViews {
    pub global: Vec<f32>,
    pub global_size: usize,
    pub tiles: Vec<Vec<f32>>,
    pub tile_size: usize,
    pub crop_grid: (usize, usize),
}

impl PreparedViews {
    pub fn base_tokens(&self) -> usize {
        let q = num_queries(self.global_size);
        q * q
    }

    pub fn tile_tokens(&self) -> usize {
        let q = num_queries(self.tile_size);
        self.tiles.len() * q * q
    }

    pub fn vision_tokens(&self) -> usize {
        self.base_tokens() + 1 + self.tile_tokens()
    }
}

pub fn deskew_enabled() -> bool {
    std::env::var("NV_DSOCR_DESKEW")
        .map(|v| v != "0")
        .unwrap_or(true)
}

pub fn fast_res_override(requested: ResolutionMode) -> ResolutionMode {
    match std::env::var("NV_DSOCR_FAST_RES").ok().as_deref() {
        Some("base1024") => ResolutionMode::Base1024,
        Some("base768") => ResolutionMode::Base768,
        _ => requested,
    }
}

pub fn gundam_max_tiles() -> usize {
    std::env::var("NV_DSOCR_MAX_TILES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(6)
}

pub fn prepare(img: &RgbImage, mode: ResolutionMode) -> Result<PreparedViews> {
    anyhow::ensure!(img.w > 0 && img.h > 0, "empty image");
    let mode = fast_res_override(mode);
    let owned;
    let img = if deskew_enabled() {
        let (deg, gain) = estimate_skew_deg(img);
        let apply = deg.abs() >= SKEW_DEAD_ZONE_DEG && gain >= SKEW_MIN_GAIN;
        if std::env::var("NV_DSOCR_DESKEW_DEBUG").is_ok() {
            eprintln!("[dsocr] deskew {deg:+.2} deg gain={gain:.4} applied={apply}");
        }
        if apply {
            owned = rotate_rgb(img, -deg, border_fill(img));
            &owned
        } else {
            img
        }
    } else {
        img
    };
    match mode {
        ResolutionMode::Gundam => {
            let (tiles, crop_grid) = if img.w <= TILE_SIZE && img.h <= TILE_SIZE {
                (Vec::new(), (1, 1))
            } else {
                let (t, g) = dynamic_tiles(img, TILE_SIZE);
                (t, g)
            };
            let global = letterbox_pad(img, GLOBAL_SIZE, PAD_FILL);
            Ok(PreparedViews {
                global: normalize_chw(&global),
                global_size: GLOBAL_SIZE,
                tiles: tiles.iter().map(normalize_chw).collect(),
                tile_size: TILE_SIZE,
                crop_grid,
            })
        }
        ResolutionMode::Base1024 => {
            let global = letterbox_pad(img, GLOBAL_SIZE, PAD_FILL);
            Ok(PreparedViews {
                global: normalize_chw(&global),
                global_size: GLOBAL_SIZE,
                tiles: Vec::new(),
                tile_size: TILE_SIZE,
                crop_grid: (1, 1),
            })
        }
        ResolutionMode::Base768 => {
            let resized = resize_rgb(img, TILE_SIZE, TILE_SIZE);
            Ok(PreparedViews {
                global: normalize_chw(&resized),
                global_size: TILE_SIZE,
                tiles: Vec::new(),
                tile_size: TILE_SIZE,
                crop_grid: (1, 1),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_resize_is_noop() {
        let img = RgbImage::from_fn(8, 6, |x, y| [(x * 30) as u8, (y * 40) as u8, 7]);
        let out = resize_rgb(&img, 8, 6);
        assert_eq!(out.data, img.data);
    }

    #[test]
    fn constant_image_resize_stays_constant() {
        let img = RgbImage::filled(100, 60, [200, 10, 90]);
        let out = resize_rgb(&img, 37, 23);
        for p in out.data.chunks(3) {
            assert_eq!(p, [200, 10, 90]);
        }
    }

    #[test]
    fn plane_resize_preserves_linear_ramp_mean() {
        let src: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let out = resize_plane_f32(&src, 8, 8, 4, 4);
        let mean_in: f32 = src.iter().sum::<f32>() / 64.0;
        let mean_out: f32 = out.iter().sum::<f32>() / 16.0;
        assert!((mean_in - mean_out).abs() < 0.5, "{mean_in} vs {mean_out}");
    }

    #[test]
    fn letterbox_wide_image_centers_vertically() {
        let img = RgbImage::filled(200, 100, [255, 255, 255]);
        let out = letterbox_pad(&img, 1024, PAD_FILL);
        assert_eq!(out.w, 1024);
        assert_eq!(out.h, 1024);
        assert_eq!(out.get(0, 0), PAD_FILL);
        assert_eq!(out.get(512, 512), [255, 255, 255]);
        assert_eq!(out.get(512, 1023), PAD_FILL);
        assert_eq!(out.get(512, 256), [255, 255, 255]);
        assert_eq!(out.get(512, 255), PAD_FILL);
    }

    #[test]
    fn tile_grid_matches_reference_cases() {
        assert_eq!(select_tile_grid(2000, 1000, 2, 6, 768), (2, 1));
        assert_eq!(select_tile_grid(1000, 2000, 2, 6, 768), (1, 2));
        assert_eq!(select_tile_grid(1500, 1000, 2, 6, 768), (3, 2));
        assert_eq!(select_tile_grid(900, 900, 2, 6, 768), (2, 2));
        assert_eq!(select_tile_grid(3000, 500, 2, 6, 768), (6, 1));
    }

    #[test]
    fn dynamic_tiles_are_row_major() {
        let img = RgbImage::from_fn(2000, 1000, |x, _| {
            if x < 1000 {
                [0, 0, 0]
            } else {
                [255, 255, 255]
            }
        });
        let (tiles, grid) = dynamic_tiles(&img, 768);
        assert_eq!(grid, (2, 1));
        assert_eq!(tiles.len(), 2);
        assert_eq!(tiles[0].get(100, 100), [0, 0, 0]);
        assert_eq!(tiles[1].get(700, 100), [255, 255, 255]);
    }

    #[test]
    fn normalize_range_and_layout() {
        let img = RgbImage::from_fn(
            2,
            1,
            |x, _| if x == 0 { [0, 255, 127] } else { [255, 0, 127] },
        );
        let v = normalize_chw(&img);
        assert_eq!(v.len(), 6);
        assert!((v[0] + 1.0).abs() < 1e-6);
        assert!((v[1] - 1.0).abs() < 1e-6);
        assert!((v[2] - 1.0).abs() < 1e-6);
        assert!((v[3] + 1.0).abs() < 1e-6);
        assert!(v[4].abs() < 0.01);
        assert!(v[5].abs() < 0.01);
    }

    #[test]
    fn gundam_token_counts() {
        let small = RgbImage::filled(600, 400, [10, 10, 10]);
        let p = prepare(&small, ResolutionMode::Gundam).unwrap();
        assert_eq!(p.tiles.len(), 0);
        assert_eq!(p.crop_grid, (1, 1));
        assert_eq!(p.vision_tokens(), 257);

        let big = RgbImage::filled(2000, 1000, [10, 10, 10]);
        let p = prepare(&big, ResolutionMode::Gundam).unwrap();
        assert_eq!(p.crop_grid, (2, 1));
        assert_eq!(p.tiles.len(), 2);
        assert_eq!(p.vision_tokens(), 257 + 288);
    }

    #[test]
    fn base_modes_token_counts() {
        let img = RgbImage::filled(500, 700, [80, 80, 80]);
        let p = prepare(&img, ResolutionMode::Base1024).unwrap();
        assert_eq!(p.vision_tokens(), 257);
        let p = prepare(&img, ResolutionMode::Base768).unwrap();
        assert_eq!(p.global_size, 768);
        assert_eq!(p.vision_tokens(), 145);
        assert_eq!(p.global.len(), 3 * 768 * 768);
    }

    fn ruled_page(w: usize, h: usize) -> RgbImage {
        RgbImage::from_fn(w, h, |_, y| {
            if y % 16 < 5 && y > 20 && y < h - 20 {
                [20, 20, 20]
            } else {
                [245, 245, 245]
            }
        })
    }

    #[test]
    fn rotate_zero_is_identity() {
        let img = ruled_page(64, 48);
        let out = rotate_rgb(&img, 0.0, PAD_FILL);
        assert_eq!(out.w, img.w);
        assert_eq!(out.data, img.data);
    }

    #[test]
    fn skew_estimator_recovers_synthetic_rotation() {
        let base = ruled_page(400, 300);
        for truth in [-3.0f64, 2.0, 4.5, -12.0, 16.0] {
            let rot = rotate_rgb(&base, truth, [245, 245, 245]);
            let (deg, gain) = estimate_skew_deg(&rot);
            assert!(
                (deg - truth).abs() < 0.5,
                "truth {truth} estimated {deg} gain {gain}"
            );
            assert!(gain > SKEW_MIN_GAIN, "gain {gain} for truth {truth}");
        }
    }

    #[test]
    fn skew_estimator_is_flat_on_axis_aligned_text() {
        let img = ruled_page(400, 300);
        let (deg, _) = estimate_skew_deg(&img);
        assert!(
            deg.abs() < SKEW_DEAD_ZONE_DEG,
            "estimated {deg} on straight page"
        );
        let (out, applied) = deskew(&img);
        assert_eq!(applied, 0.0);
        assert_eq!(out.data, img.data);
    }

    #[test]
    fn deskew_on_blank_page_is_a_noop() {
        let img = RgbImage::filled(300, 200, [250, 250, 250]);
        let (out, applied) = deskew(&img);
        assert_eq!(applied, 0.0);
        assert_eq!(out.data, img.data);
    }

    #[test]
    fn deskew_straightens_a_rotated_page() {
        let base = ruled_page(400, 300);
        let rot = rotate_rgb(&base, 4.0, [245, 245, 245]);
        let (fixed, applied) = deskew(&rot);
        assert!((applied - 4.0).abs() < 0.5, "applied {applied}");
        let (residual, _) = estimate_skew_deg(&fixed);
        assert!(residual.abs() < SKEW_DEAD_ZONE_DEG, "residual {residual}");
    }

    #[test]
    fn border_fill_picks_the_margin_colour() {
        let img = RgbImage::from_fn(200, 150, |x, y| {
            if x > 20 && x < 180 && y > 20 && y < 130 {
                [10, 20, 30]
            } else {
                [200, 100, 50]
            }
        });
        assert_eq!(border_fill(&img), [200, 100, 50]);
    }

    #[test]
    fn deskew_leaves_a_straight_page_bit_identical_through_prepare() {
        let img = ruled_page(900, 900);
        let (pre, applied) = deskew(&img);
        assert_eq!(applied, 0.0);
        let a = prepare(&img, ResolutionMode::Gundam).unwrap();
        let b = prepare(&pre, ResolutionMode::Gundam).unwrap();
        assert_eq!(a.crop_grid, b.crop_grid);
        assert_eq!(a.global, b.global);
        assert_eq!(a.tiles, b.tiles);
    }

    #[test]
    #[ignore]
    fn scan_skew_angles_from_env() {
        let Ok(list) = std::env::var("NV_DSOCR_SKEW_SCAN") else {
            return;
        };
        for path in list.split(':').filter(|s| !s.is_empty()) {
            let img = RgbImage::decode_file(std::path::Path::new(path)).unwrap();
            let t = std::time::Instant::now();
            let (deg, gain) = estimate_skew_deg(&img);
            println!(
                "{deg:+7.2} gain={gain:.4} {:.2}s {}x{} {path}",
                t.elapsed().as_secs_f64(),
                img.w,
                img.h
            );
        }
    }

    #[test]
    fn num_queries_matches_reference() {
        assert_eq!(num_queries(1024), 16);
        assert_eq!(num_queries(768), 12);
    }
}
