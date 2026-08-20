use crate::raster::IntegralImage;
use crate::{BinImage, GreyImage};

pub const SAUVOLA_WINDOW: usize = 25;
pub const SAUVOLA_K: f32 = 0.2;
pub const BG_MIN_LONG: usize = 2000;
pub const BG_CELL: usize = 64;
pub const BG_MIN_SEPARATION: u8 = 40;
pub const BG_MIN_NOISE_DENSITY: f32 = 0.04;
pub const BG_MIN_KEEP_FRAC: f32 = 0.15;
pub const BG_MIN_CLEAR_FRAC: f32 = 0.02;

impl BinImage {
    pub fn ink_count(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }
}

pub fn histogram(g: &GreyImage) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for &v in &g.data {
        hist[v as usize] += 1;
    }
    hist
}

pub fn otsu_threshold(hist: &[u32; 256]) -> u8 {
    let total: u64 = hist.iter().map(|&n| n as u64).sum();
    if total == 0 {
        return 0;
    }
    let weighted_total: u64 = hist
        .iter()
        .enumerate()
        .map(|(v, &n)| v as u64 * n as u64)
        .sum();
    let mut best_t = 0u8;
    let mut best_score = -1.0f64;
    let mut w0 = 0u64;
    let mut sum0 = 0u64;
    for t in 0..=255usize {
        w0 += hist[t] as u64;
        sum0 += t as u64 * hist[t] as u64;
        let w1 = total - w0;
        if w0 == 0 || w1 == 0 {
            continue;
        }
        let m0 = sum0 as f64 / w0 as f64;
        let m1 = (weighted_total - sum0) as f64 / w1 as f64;
        let score = w0 as f64 * w1 as f64 * (m0 - m1) * (m0 - m1);
        if score > best_score {
            best_score = score;
            best_t = t as u8;
        }
    }
    best_t
}

pub fn ink_is_dark(g: &GreyImage) -> bool {
    let hist = histogram(g);
    let thr = otsu_threshold(&hist);
    let total: u64 = hist.iter().map(|&n| n as u64).sum();
    if total == 0 {
        return true;
    }
    let dark: u64 = hist[..=thr as usize].iter().map(|&n| n as u64).sum();
    (dark as f64) < 0.65 * total as f64
}

pub fn binarize_otsu(g: &GreyImage) -> BinImage {
    let hist = histogram(g);
    let thr = otsu_threshold(&hist);
    let dark_ink = ink_is_dark(g);
    let mut bin = BinImage::new(g.w, g.h);
    for y in 0..g.h {
        for x in 0..g.w {
            let v = g.data[y * g.w + x];
            let ink = if dark_ink { v <= thr } else { v > thr };
            if ink {
                bin.set(x, y, true);
            }
        }
    }
    bin
}

pub fn binarize_sauvola(g: &GreyImage, window: usize, k: f32) -> BinImage {
    let mut bin = BinImage::new(g.w, g.h);
    if g.w == 0 || g.h == 0 {
        return bin;
    }
    let work;
    let src = if ink_is_dark(g) {
        g
    } else {
        work = g.invert();
        &work
    };
    let integral = IntegralImage::build(src);
    let half = window / 2;
    let (sums, sqs, stride) = integral.planes();
    let w = src.w;
    for y in 0..src.h {
        let y0 = y.saturating_sub(half);
        let y1 = (y + half + 1).min(src.h);
        let top = &sums[y0 * stride..(y0 + 1) * stride];
        let bot = &sums[y1 * stride..(y1 + 1) * stride];
        let topq = &sqs[y0 * stride..(y0 + 1) * stride];
        let botq = &sqs[y1 * stride..(y1 + 1) * stride];
        let dy = (y1 - y0) as f64;
        let row = &src.data[y * w..(y + 1) * w];
        for x in 0..w {
            let x0 = x.saturating_sub(half);
            let x1 = (x + half + 1).min(w);
            let n = (x1 - x0) as f64 * dy;
            let s = (bot[x1] + top[x0] - top[x1] - bot[x0]) as f64;
            let sq = (botq[x1] + topq[x0] - topq[x1] - botq[x0]) as f64;
            let mean = s / n;
            let var = (sq / n - mean * mean).max(0.0);
            let std = var.sqrt();
            let t = mean * (1.0 + k as f64 * (std / 128.0 - 1.0));
            if (row[x] as f64) <= t {
                bin.set(x, y, true);
            }
        }
    }
    bin
}

pub fn cell_density(bin: &BinImage, cell: usize) -> (Vec<f32>, usize, usize) {
    let gw = bin.w.div_ceil(cell);
    let gh = bin.h.div_ceil(cell);
    if gw == 0 || gh == 0 {
        return (Vec::new(), 0, 0);
    }
    let words_per_cell = (cell / 64).max(1);
    let mut ink = vec![0u32; gw * gh];
    for y in 0..bin.h {
        let gy = y / cell;
        let row = &bin.bits[y * bin.stride..(y + 1) * bin.stride];
        for (wi, w) in row.iter().enumerate() {
            if *w == 0 {
                continue;
            }
            ink[gy * gw + (wi / words_per_cell).min(gw - 1)] += w.count_ones();
        }
    }
    let mut out = vec![0f32; gw * gh];
    for gy in 0..gh {
        let rows = (bin.h - gy * cell).min(cell);
        for gx in 0..gw {
            let cols = (bin.w - gx * cell).min(cell);
            let n = (rows * cols).max(1) as f32;
            out[gy * gw + gx] = ink[gy * gw + gx] as f32 / n;
        }
    }
    (out, gw, gh)
}

pub fn cell_mean(g: &GreyImage, cell: usize) -> (Vec<u8>, usize, usize) {
    let gw = g.w.div_ceil(cell);
    let gh = g.h.div_ceil(cell);
    if gw == 0 || gh == 0 {
        return (Vec::new(), 0, 0);
    }
    let mut sum = vec![0u64; gw * gh];
    for y in 0..g.h {
        let base = (y / cell) * gw;
        let row = &g.data[y * g.w..(y + 1) * g.w];
        for (x, &v) in row.iter().enumerate() {
            sum[base + x / cell] += v as u64;
        }
    }
    let mut out = vec![0u8; gw * gh];
    for gy in 0..gh {
        let rows = (g.h - gy * cell).min(cell);
        for gx in 0..gw {
            let cols = (g.w - gx * cell).min(cell);
            let n = (rows * cols).max(1) as u64;
            out[gy * gw + gx] = (sum[gy * gw + gx] / n) as u8;
        }
    }
    (out, gw, gh)
}

fn largest_blob(paper: &[bool], gw: usize, gh: usize) -> Vec<bool> {
    let mut label = vec![u32::MAX; gw * gh];
    let mut best_id = u32::MAX;
    let mut best_n = 0usize;
    let mut next = 0u32;
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..gw * gh {
        if !paper[start] || label[start] != u32::MAX {
            continue;
        }
        let id = next;
        next += 1;
        let mut n = 0usize;
        stack.push(start);
        label[start] = id;
        while let Some(i) = stack.pop() {
            n += 1;
            let (x, y) = (i % gw, i / gw);
            let push = |nx: usize, ny: usize, st: &mut Vec<usize>, lab: &mut Vec<u32>| {
                let j = ny * gw + nx;
                if paper[j] && lab[j] == u32::MAX {
                    lab[j] = id;
                    st.push(j);
                }
            };
            if x > 0 {
                push(x - 1, y, &mut stack, &mut label);
            }
            if x + 1 < gw {
                push(x + 1, y, &mut stack, &mut label);
            }
            if y > 0 {
                push(x, y - 1, &mut stack, &mut label);
            }
            if y + 1 < gh {
                push(x, y + 1, &mut stack, &mut label);
            }
        }
        if n > best_n {
            best_n = n;
            best_id = id;
        }
    }
    let core: Vec<bool> = label.iter().map(|&l| l == best_id).collect();
    let mut keep = core.clone();
    for gy in 0..gh {
        for gx in 0..gw {
            if !core[gy * gw + gx] {
                continue;
            }
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let nx = gx as i64 + dx;
                    let ny = gy as i64 + dy;
                    if nx >= 0 && ny >= 0 && (nx as usize) < gw && (ny as usize) < gh {
                        keep[ny as usize * gw + nx as usize] = true;
                    }
                }
            }
        }
    }
    keep
}

pub fn suppress_background(g: &GreyImage, bin: &BinImage) -> Option<GreyImage> {
    if g.w.max(g.h) < BG_MIN_LONG || g.w < BG_CELL * 4 || g.h < BG_CELL * 4 {
        return None;
    }
    let (means, gw, gh) = cell_mean(g, BG_CELL);
    if gw == 0 {
        return None;
    }
    let dark_ink = ink_is_dark(g);
    let mut chist = [0u32; 256];
    for &m in &means {
        chist[m as usize] += 1;
    }
    let thr = otsu_threshold(&chist);
    let mut lo_sum = 0u64;
    let mut lo_n = 0u64;
    let mut hi_sum = 0u64;
    let mut hi_n = 0u64;
    for (v, &c) in chist.iter().enumerate() {
        if v <= thr as usize {
            lo_sum += v as u64 * c as u64;
            lo_n += c as u64;
        } else {
            hi_sum += v as u64 * c as u64;
            hi_n += c as u64;
        }
    }
    if lo_n == 0 || hi_n == 0 {
        return None;
    }
    if hi_sum / hi_n - lo_sum / lo_n < BG_MIN_SEPARATION as u64 {
        return None;
    }
    let page: Vec<bool> = means
        .iter()
        .map(|&m| if dark_ink { m > thr } else { m <= thr })
        .collect();
    let blob = largest_blob(&page, gw, gh);
    let mut cx0 = gw;
    let mut cy0 = gh;
    let mut cx1 = 0usize;
    let mut cy1 = 0usize;
    for (i, &k) in blob.iter().enumerate() {
        if k {
            cx0 = cx0.min(i % gw);
            cy0 = cy0.min(i / gw);
            cx1 = cx1.max(i % gw + 1);
            cy1 = cy1.max(i / gw + 1);
        }
    }
    if cx1 <= cx0 || cy1 <= cy0 {
        return None;
    }
    let cells = (cx1 - cx0) * (cy1 - cy0);
    let keep_frac = cells as f32 / (gw * gh) as f32;
    if keep_frac < BG_MIN_KEEP_FRAC || 1.0 - keep_frac < BG_MIN_CLEAR_FRAC {
        return None;
    }
    let (density, dw, _) = cell_density(bin, BG_CELL);
    if dw != gw {
        return None;
    }
    let mut noise = 0f32;
    let mut noise_n = 0usize;
    for gy in 0..gh {
        for gx in 0..gw {
            if gx < cx0 || gx >= cx1 || gy < cy0 || gy >= cy1 {
                noise += density[gy * gw + gx];
                noise_n += 1;
            }
        }
    }
    if noise_n == 0 || noise / (noise_n as f32) < BG_MIN_NOISE_DENSITY {
        return None;
    }
    let x0 = cx0 * BG_CELL;
    let y0 = cy0 * BG_CELL;
    let x1 = (cx1 * BG_CELL).min(g.w);
    let y1 = (cy1 * BG_CELL).min(g.h);
    let mut hist = [0u32; 256];
    let mut n = 0u32;
    for y in (y0..y1).step_by(4) {
        for x in (x0..x1).step_by(4) {
            hist[g.data[y * g.w + x] as usize] += 1;
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    let mut acc = 0u32;
    let mut fill = 255u8;
    for (v, &c) in hist.iter().enumerate() {
        acc += c;
        if acc * 2 >= n {
            fill = v as u8;
            break;
        }
    }
    let mut out = g.clone();
    for y in 0..g.h {
        let inside_y = y >= y0 && y < y1;
        for x in 0..g.w {
            if !inside_y || x < x0 || x >= x1 {
                out.data[y * g.w + x] = fill;
            }
        }
    }
    Some(out)
}
