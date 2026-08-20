use crate::binarize::{binarize_sauvola, suppress_background, SAUVOLA_K, SAUVOLA_WINDOW};
use crate::{BinImage, GreyImage, Line, PixelRect, WordBox};

pub const SKEW_NARROW_DEG: f32 = 5.0;
pub const SKEW_COARSE_DEG: f32 = 45.0;
pub const SKEW_COARSE_STEP: f32 = 1.0;
pub const SKEW_COARSE_LONG: f32 = 500.0;
pub const SKEW_COARSE_PTS: usize = 24_000;
pub const SKEW_WIDE_MARGIN: f64 = 1.25;
pub const SPECKLE_MIN_COMPS: usize = 2000;
pub const SPECKLE_MIN_HEIGHT_FRAC: f32 = 0.35;
pub const SPECKLE_MIN_KEPT: usize = 48;
pub const COLUMN_MIN_BANDS: usize = 6;
pub const COLUMN_MIN_WIDTH_FRAC: f32 = 0.25;
pub const COLUMN_GUTTER_MIN_FRAC: f32 = 0.02;
pub const COLUMN_GUTTER_MIN_HEIGHTS: f32 = 1.2;
pub const COLUMN_MIN_FILL_FRAC: f32 = 0.6;

impl PixelRect {
    pub fn width(&self) -> i32 {
        self.right - self.left
    }

    pub fn height(&self) -> i32 {
        self.bottom - self.top
    }

    pub fn union(&self, other: &PixelRect) -> PixelRect {
        PixelRect {
            left: self.left.min(other.left),
            top: self.top.min(other.top),
            right: self.right.max(other.right),
            bottom: self.bottom.max(other.bottom),
        }
    }

    pub fn intersection_area(&self, other: &PixelRect) -> i64 {
        let w = (self.right.min(other.right) - self.left.max(other.left)).max(0) as i64;
        let h = (self.bottom.min(other.bottom) - self.top.max(other.top)).max(0) as i64;
        w * h
    }

    pub fn area(&self) -> i64 {
        self.width().max(0) as i64 * self.height().max(0) as i64
    }

    pub fn iou(&self, other: &PixelRect) -> f64 {
        let inter = self.intersection_area(other);
        let union = self.area() + other.area() - inter;
        if union <= 0 {
            0.0
        } else {
            inter as f64 / union as f64
        }
    }
}

#[derive(Debug, Clone)]
pub struct Component {
    pub bbox: PixelRect,
    pub area: usize,
    pub cx: f32,
    pub cy: f32,
}

struct Dsu {
    parent: Vec<u32>,
}

impl Dsu {
    fn new() -> Self {
        Dsu { parent: Vec::new() }
    }

    fn make(&mut self) -> u32 {
        let id = self.parent.len() as u32;
        self.parent.push(id);
        id
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let gp = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = gp;
            x = gp;
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.parent[hi as usize] = lo;
        }
    }
}

pub fn connected_components(bin: &BinImage) -> Vec<Component> {
    let w = bin.w;
    let h = bin.h;
    if w == 0 || h == 0 {
        return Vec::new();
    }
    const NONE: u32 = u32::MAX;
    let mut labels = vec![NONE; w * h];
    let mut dsu = Dsu::new();
    for y in 0..h {
        for x in 0..w {
            if !bin.get(x, y) {
                continue;
            }
            let mut label = NONE;
            let mut neigh = [NONE; 4];
            let mut n = 0;
            if x > 0 && labels[y * w + x - 1] != NONE {
                neigh[n] = labels[y * w + x - 1];
                n += 1;
            }
            if y > 0 {
                if x > 0 && labels[(y - 1) * w + x - 1] != NONE {
                    neigh[n] = labels[(y - 1) * w + x - 1];
                    n += 1;
                }
                if labels[(y - 1) * w + x] != NONE {
                    neigh[n] = labels[(y - 1) * w + x];
                    n += 1;
                }
                if x + 1 < w && labels[(y - 1) * w + x + 1] != NONE {
                    neigh[n] = labels[(y - 1) * w + x + 1];
                    n += 1;
                }
            }
            for &nb in &neigh[..n] {
                if label == NONE {
                    label = nb;
                } else {
                    dsu.union(label, nb);
                }
            }
            if label == NONE {
                label = dsu.make();
            }
            labels[y * w + x] = label;
        }
    }
    let mut root_index = vec![NONE; dsu.parent.len()];
    let mut comps: Vec<(i32, i32, i32, i32, usize, u64, u64)> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let l = labels[y * w + x];
            if l == NONE {
                continue;
            }
            let root = dsu.find(l);
            let idx = if root_index[root as usize] == NONE {
                let idx = comps.len() as u32;
                root_index[root as usize] = idx;
                comps.push((x as i32, y as i32, x as i32, y as i32, 0, 0, 0));
                idx
            } else {
                root_index[root as usize]
            };
            let c = &mut comps[idx as usize];
            c.0 = c.0.min(x as i32);
            c.1 = c.1.min(y as i32);
            c.2 = c.2.max(x as i32);
            c.3 = c.3.max(y as i32);
            c.4 += 1;
            c.5 += x as u64;
            c.6 += y as u64;
        }
    }
    comps
        .into_iter()
        .map(|(l, t, r, b, area, sx, sy)| Component {
            bbox: PixelRect {
                left: l,
                top: t,
                right: r + 1,
                bottom: b + 1,
            },
            area,
            cx: sx as f32 / area as f32,
            cy: sy as f32 / area as f32,
        })
        .collect()
}

fn skew_score(pts: &[(f32, f32)], deg: f32) -> f64 {
    let (s, c) = deg.to_radians().sin_cos();
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    let mut ys = Vec::with_capacity(pts.len());
    for &(x, y) in pts {
        let yp = s * x + c * y;
        min_y = min_y.min(yp);
        max_y = max_y.max(yp);
        ys.push(yp);
    }
    let bins = ((max_y - min_y) as usize) + 3;
    let mut hist = vec![0f64; bins];
    for yp in ys {
        let t = yp - min_y;
        let i = t.floor() as usize;
        let f = (t - t.floor()) as f64;
        hist[i] += 1.0 - f;
        hist[i + 1] += f;
    }
    hist.windows(2).map(|w| (w[1] - w[0]) * (w[1] - w[0])).sum()
}

fn coarse_skew_center(pts: &[(f32, f32)], w: usize, h: usize) -> Option<f32> {
    let long = w.max(h) as f32;
    let r = (long / SKEW_COARSE_LONG).max(1.0);
    let stride = (pts.len() / SKEW_COARSE_PTS).max(1);
    let small: Vec<(f32, f32)> = pts
        .iter()
        .step_by(stride)
        .map(|&(x, y)| (x / r, y / r))
        .collect();
    if small.len() < 32 {
        return None;
    }
    let mut best = 0.0f32;
    let mut best_score = -1.0f64;
    let mut near_best = -1.0f64;
    let mut deg = -SKEW_COARSE_DEG;
    while deg <= SKEW_COARSE_DEG + 0.001 {
        let score = skew_score(&small, deg);
        if deg.abs() <= SKEW_NARROW_DEG && score > near_best {
            near_best = score;
        }
        if score > best_score || (score == best_score && deg.abs() < best.abs()) {
            best_score = score;
            best = deg;
        }
        deg += SKEW_COARSE_STEP;
    }
    if best.abs() <= SKEW_NARROW_DEG || best_score < near_best * SKEW_WIDE_MARGIN {
        None
    } else {
        Some(best)
    }
}

pub fn estimate_skew_scored(bin: &BinImage) -> (f32, f64) {
    let mut pts = Vec::new();
    for y in 0..bin.h {
        for x in 0..bin.w {
            if bin.get(x, y) {
                pts.push((x as f32, y as f32));
            }
        }
    }
    if pts.len() > 400_000 {
        let stride = pts.len() / 200_000;
        pts = pts.into_iter().step_by(stride.max(1)).collect();
    }
    if pts.len() < 32 {
        return (0.0, 1.0);
    }
    let score0 = skew_score(&pts, 0.0);
    let center = coarse_skew_center(&pts, bin.w, bin.h);
    let (lo, hi) = match center {
        Some(c) => (c - 1.0, c + 1.0),
        None => (-SKEW_NARROW_DEG, SKEW_NARROW_DEG),
    };
    let mut best = center.unwrap_or(0.0);
    let mut best_score = -1.0f64;
    let mut deg = lo;
    while deg <= hi + 0.001 {
        let score = skew_score(&pts, deg);
        if score > best_score || (score == best_score && deg.abs() < best.abs()) {
            best_score = score;
            best = deg;
        }
        deg += 0.25;
    }
    let coarse = best;
    let mut fine = coarse;
    let mut fine_score = best_score;
    let mut d = coarse - 0.3;
    while d <= coarse + 0.301 {
        let score = skew_score(&pts, d);
        if score > fine_score || (score == fine_score && d.abs() < fine.abs()) {
            fine_score = score;
            fine = d;
        }
        d += 0.05;
    }
    let ratio = if score0 > 0.0 {
        fine_score / score0
    } else {
        f64::MAX
    };
    (fine, ratio)
}

pub fn estimate_skew(bin: &BinImage) -> f32 {
    estimate_skew_scored(bin).0
}

#[derive(Debug, Clone, Copy)]
pub struct Deskew {
    pub deg: f32,
    pub in_w: usize,
    pub in_h: usize,
    pub out_w: usize,
    pub out_h: usize,
    sin: f32,
    cos: f32,
    in_cx: f32,
    in_cy: f32,
    out_cx: f32,
    out_cy: f32,
}

impl Deskew {
    pub fn new(w: usize, h: usize, deg: f32, expand: bool) -> Self {
        let (s, c) = deg.to_radians().sin_cos();
        let (ow, oh) = if expand {
            let fw = w as f32;
            let fh = h as f32;
            (
                (fw * c.abs() + fh * s.abs()).ceil().max(1.0) as usize,
                (fw * s.abs() + fh * c.abs()).ceil().max(1.0) as usize,
            )
        } else {
            (w, h)
        };
        Deskew {
            deg,
            in_w: w,
            in_h: h,
            out_w: ow,
            out_h: oh,
            sin: s,
            cos: c,
            in_cx: (w as f32 - 1.0) * 0.5,
            in_cy: (h as f32 - 1.0) * 0.5,
            out_cx: (ow as f32 - 1.0) * 0.5,
            out_cy: (oh as f32 - 1.0) * 0.5,
        }
    }

    pub fn out_to_in(&self, x: f32, y: f32) -> (f32, f32) {
        let dx = x - self.out_cx;
        let dy = y - self.out_cy;
        (
            self.cos * dx + self.sin * dy + self.in_cx,
            -self.sin * dx + self.cos * dy + self.in_cy,
        )
    }

    pub fn map_rect_back(&self, r: &PixelRect) -> PixelRect {
        let corners = [
            (r.left as f32, r.top as f32),
            (r.right as f32, r.top as f32),
            (r.left as f32, r.bottom as f32),
            (r.right as f32, r.bottom as f32),
        ];
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        for (x, y) in corners {
            let (xi, yi) = self.out_to_in(x, y);
            min_x = min_x.min(xi);
            min_y = min_y.min(yi);
            max_x = max_x.max(xi);
            max_y = max_y.max(yi);
        }
        PixelRect {
            left: (min_x.floor() as i32).clamp(0, self.in_w as i32),
            top: (min_y.floor() as i32).clamp(0, self.in_h as i32),
            right: (max_x.ceil() as i32).clamp(0, self.in_w as i32),
            bottom: (max_y.ceil() as i32).clamp(0, self.in_h as i32),
        }
    }
}

pub fn rotate_grey_with(g: &GreyImage, dsk: &Deskew, bg: u8) -> GreyImage {
    let mut out = GreyImage::new(dsk.out_w, dsk.out_h);
    for yo in 0..dsk.out_h {
        for xo in 0..dsk.out_w {
            let (xi, yi) = dsk.out_to_in(xo as f32, yo as f32);
            out.data[yo * dsk.out_w + xo] =
                g.sample_bilinear(xi, yi, bg).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

pub fn rotate_grey(g: &GreyImage, deg: f32, bg: u8) -> GreyImage {
    rotate_grey_with(g, &Deskew::new(g.w, g.h, deg, false), bg)
}

fn median_i32(vals: &mut [i32]) -> i32 {
    if vals.is_empty() {
        return 0;
    }
    vals.sort_unstable();
    vals[vals.len() / 2]
}

fn weighted_median(hist: &[u64]) -> i32 {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return 0;
    }
    let half = total.div_ceil(2);
    let mut acc = 0u64;
    for (v, &w) in hist.iter().enumerate() {
        acc += w;
        if acc >= half {
            return v as i32;
        }
    }
    0
}

pub fn speckle_filter(comps: Vec<Component>) -> Vec<Component> {
    if comps.len() < SPECKLE_MIN_COMPS {
        return comps;
    }
    let maxh = comps.iter().map(|c| c.bbox.height()).max().unwrap_or(0);
    if maxh <= 0 {
        return comps;
    }
    let mut by_count = vec![0u64; maxh as usize + 1];
    let mut by_area = vec![0u64; maxh as usize + 1];
    for c in &comps {
        by_count[c.bbox.height() as usize] += 1;
        by_area[c.bbox.height() as usize] += c.area as u64;
    }
    let count_med = weighted_median(&by_count);
    let area_med = weighted_median(&by_area);
    if 2 * count_med >= area_med {
        return comps;
    }
    let min_h = (area_med as f32 * SPECKLE_MIN_HEIGHT_FRAC).round() as i32;
    if comps.iter().filter(|c| c.bbox.height() >= min_h).count() < SPECKLE_MIN_KEPT {
        return comps;
    }
    comps
        .into_iter()
        .filter(|c| c.bbox.height() >= min_h)
        .collect()
}

fn line_bands(comps: &[Component], img_h: usize) -> Vec<Vec<usize>> {
    if comps.is_empty() || img_h == 0 {
        return Vec::new();
    }
    let mut heights: Vec<i32> = comps.iter().map(|c| c.bbox.height()).collect();
    let med_h = median_i32(&mut heights).max(1);
    let mut cover = vec![0f32; img_h];
    for c in comps {
        let half = (c.bbox.height() as f32 * 0.25).clamp(1.0, med_h as f32 * 0.5);
        let t = (c.cy - half).floor().max(0.0) as usize;
        let b = ((c.cy + half).ceil().max(0.0) as usize).min(img_h);
        for row in cover.iter_mut().take(b).skip(t.min(b)) {
            *row += c.bbox.width().max(1) as f32;
        }
    }
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut start = None;
    for (y, &v) in cover.iter().enumerate() {
        if v > 0.0 {
            if start.is_none() {
                start = Some(y);
            }
        } else if let Some(s) = start.take() {
            runs.push((s, y));
        }
    }
    if let Some(s) = start {
        runs.push((s, img_h));
    }
    let gap_max = ((med_h as f32 * 0.3) as usize).max(1);
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for run in runs {
        if let Some(last) = merged.last_mut() {
            if run.0 - last.1 <= gap_max {
                last.1 = run.1;
                continue;
            }
        }
        merged.push(run);
    }
    let mut bands: Vec<Vec<usize>> = vec![Vec::new(); merged.len()];
    for (i, c) in comps.iter().enumerate() {
        let mid = c.cy.round().max(0.0) as usize;
        let mut assigned = None;
        for (bi, &(s, e)) in merged.iter().enumerate() {
            if mid >= s && mid < e {
                assigned = Some(bi);
                break;
            }
        }
        let bi = assigned.unwrap_or_else(|| {
            let mut best = 0;
            let mut best_d = usize::MAX;
            for (bi, &(s, e)) in merged.iter().enumerate() {
                let d = if mid < s {
                    s - mid
                } else {
                    mid.saturating_sub(e)
                };
                if d < best_d {
                    best_d = d;
                    best = bi;
                }
            }
            best
        });
        bands[bi].push(i);
    }
    let band_max_h: Vec<i32> = bands
        .iter()
        .map(|b| b.iter().map(|&i| comps[i].bbox.height()).max().unwrap_or(0))
        .collect();
    let strong: Vec<bool> = band_max_h.iter().map(|&h| 2 * h >= med_h).collect();
    if strong.iter().any(|&s| s) && strong.iter().any(|&s| !s) {
        let centers: Vec<f32> = merged.iter().map(|&(s, e)| (s + e) as f32 * 0.5).collect();
        let mut moved: Vec<(usize, usize)> = Vec::new();
        for (bi, band) in bands.iter().enumerate() {
            if strong[bi] {
                continue;
            }
            for &ci in band {
                let cy = comps[ci].cy;
                let mut best = None;
                let mut best_d = f32::MAX;
                for (oi, &ok) in strong.iter().enumerate() {
                    if !ok {
                        continue;
                    }
                    let d = (centers[oi] - cy).abs();
                    if d < best_d {
                        best_d = d;
                        best = Some(oi);
                    }
                }
                if let Some(oi) = best {
                    moved.push((oi, ci));
                }
            }
        }
        for (bi, band) in bands.iter_mut().enumerate() {
            if !strong[bi] {
                band.clear();
            }
        }
        for (oi, ci) in moved {
            bands[oi].push(ci);
        }
    }
    bands.retain(|b| !b.is_empty());
    bands
}

fn band_left_right(comps: &[Component], band: &[usize]) -> (i32, i32) {
    let mut l = i32::MAX;
    let mut r = i32::MIN;
    for &i in band {
        l = l.min(comps[i].bbox.left);
        r = r.max(comps[i].bbox.right);
    }
    (l, r)
}

fn band_top(comps: &[Component], band: &[usize]) -> i32 {
    band.iter()
        .map(|&i| comps[i].bbox.top)
        .min()
        .unwrap_or(i32::MAX)
}

fn ink_mask(comps: &[Component], band: &[usize], img_w: usize) -> Vec<bool> {
    let mut mask = vec![false; img_w];
    for &i in band {
        let l = comps[i].bbox.left.max(0) as usize;
        let r = (comps[i].bbox.right.max(0) as usize).min(img_w);
        for cell in mask.iter_mut().take(r).skip(l.min(r)) {
            *cell = true;
        }
    }
    mask
}

fn column_of(cuts: &[i32], x: i32) -> usize {
    cuts.partition_point(|&c| c <= x)
}

fn validate_split(free: &[bool], left: i32, right: i32, med_h: i32) -> Option<Vec<i32>> {
    let span = right - left;
    if span <= 0 {
        return None;
    }
    let min_gutter = ((span as f32 * COLUMN_GUTTER_MIN_FRAC).ceil() as i32)
        .max((med_h as f32 * COLUMN_GUTTER_MIN_HEIGHTS).ceil() as i32)
        .max(2);
    let mut gutters: Vec<(i32, i32)> = Vec::new();
    let mut start: Option<i32> = None;
    for x in left..right {
        if free[x as usize] {
            if start.is_none() {
                start = Some(x);
            }
        } else if let Some(s) = start.take() {
            if x - s >= min_gutter {
                gutters.push((s, x));
            }
        }
    }
    if gutters.is_empty() {
        return None;
    }
    let min_col = (span as f32 * COLUMN_MIN_WIDTH_FRAC) as i32;
    let mut x0 = left;
    for &(gs, ge) in &gutters {
        if gs - x0 < min_col {
            return None;
        }
        x0 = ge;
    }
    if right - x0 < min_col {
        return None;
    }
    Some(gutters.iter().map(|&(gs, ge)| (gs + ge) / 2).collect())
}

fn columns_filled(comps: &[Component], bands: &[Vec<usize>], cuts: &[i32]) -> bool {
    let ncols = cuts.len() + 1;
    let mut counts = vec![0usize; ncols];
    for band in bands {
        let mut seen = vec![false; ncols];
        for &ci in band {
            seen[column_of(cuts, comps[ci].cx.round() as i32)] = true;
        }
        for (k, &s) in seen.iter().enumerate() {
            if s {
                counts[k] += 1;
            }
        }
    }
    let need = COLUMN_MIN_FILL_FRAC * bands.len() as f32;
    counts.iter().all(|&c| c as f32 >= need)
}

fn detect_column_zones(
    comps: &[Component],
    bands: &[Vec<usize>],
    img_w: usize,
) -> Vec<(usize, usize, Vec<i32>)> {
    let n = bands.len();
    if n < COLUMN_MIN_BANDS || img_w == 0 {
        return Vec::new();
    }
    let mut heights: Vec<i32> = comps.iter().map(|c| c.bbox.height()).collect();
    let med_h = median_i32(&mut heights).max(1);
    let masks: Vec<Vec<bool>> = bands.iter().map(|b| ink_mask(comps, b, img_w)).collect();
    let bounds: Vec<(i32, i32)> = bands.iter().map(|b| band_left_right(comps, b)).collect();
    let mut zones: Vec<(usize, usize, Vec<i32>)> = Vec::new();
    let mut i = 0;
    while i + COLUMN_MIN_BANDS <= n {
        let mut free: Vec<bool> = masks[i].iter().map(|&v| !v).collect();
        let (mut left, mut right) = bounds[i];
        let mut best: Option<(usize, Vec<i32>)> = None;
        let mut j = i + 1;
        while j < n {
            let next: Vec<bool> = free
                .iter()
                .zip(masks[j].iter())
                .map(|(&f, &m)| f && !m)
                .collect();
            let nl = left.min(bounds[j].0);
            let nr = right.max(bounds[j].1);
            let Some(cuts) = validate_split(&next, nl, nr, med_h) else {
                break;
            };
            free = next;
            left = nl;
            right = nr;
            if j + 1 - i >= COLUMN_MIN_BANDS {
                best = Some((j + 1, cuts));
            }
            j += 1;
        }
        match best {
            Some((end, cuts)) if columns_filled(comps, &bands[i..end], &cuts) => {
                zones.push((i, end, cuts));
                i = end;
            }
            _ => i += 1,
        }
    }
    zones
}

fn plan_reading_order(comps: &[Component], img_w: usize, img_h: usize) -> Vec<Vec<usize>> {
    let mut bands = line_bands(comps, img_h);
    bands.sort_by_key(|b| band_top(comps, b));
    let zones = detect_column_zones(comps, &bands, img_w);
    if zones.is_empty() {
        return bands;
    }
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut i = 0;
    for (start, end, cuts) in zones {
        while i < start {
            out.push(std::mem::take(&mut bands[i]));
            i += 1;
        }
        let mut cols: Vec<Vec<usize>> = vec![Vec::new(); cuts.len() + 1];
        for band in &bands[start..end] {
            for &ci in band {
                cols[column_of(&cuts, comps[ci].cx.round() as i32)].push(ci);
            }
        }
        for col in cols {
            if col.is_empty() {
                continue;
            }
            let sub: Vec<Component> = col.iter().map(|&ci| comps[ci].clone()).collect();
            let mut sub_bands = line_bands(&sub, img_h);
            sub_bands.sort_by_key(|b| band_top(&sub, b));
            for b in sub_bands {
                out.push(b.into_iter().map(|k| col[k]).collect());
            }
        }
        i = end;
    }
    while i < bands.len() {
        out.push(std::mem::take(&mut bands[i]));
        i += 1;
    }
    out
}

fn split_words(comps: &[Component], order: &[usize], x_height: f32) -> Vec<Vec<usize>> {
    if order.is_empty() {
        return Vec::new();
    }
    let mut gaps: Vec<f32> = Vec::new();
    let mut run_right = comps[order[0]].bbox.right;
    for &i in &order[1..] {
        let gap = (comps[i].bbox.left - run_right).max(0) as f32;
        if gap > 0.0 {
            gaps.push(gap);
        }
        run_right = run_right.max(comps[i].bbox.right);
    }
    let med_gap = if gaps.is_empty() {
        0.0
    } else {
        let mut sorted = gaps.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };
    let thr = (2.0 * med_gap).max(0.3 * x_height).max(2.0);
    let mut words: Vec<Vec<usize>> = vec![vec![order[0]]];
    let mut run_right = comps[order[0]].bbox.right;
    for &i in &order[1..] {
        let gap = (comps[i].bbox.left - run_right).max(0) as f32;
        if gap > thr {
            words.push(Vec::new());
        }
        words.last_mut().unwrap().push(i);
        run_right = run_right.max(comps[i].bbox.right);
    }
    words
}

pub struct LayoutResult {
    pub angle_deg: f32,
    pub lines: Vec<Line>,
}

pub fn extract_lines(grey: &GreyImage) -> LayoutResult {
    if grey.w == 0 || grey.h == 0 {
        return LayoutResult {
            angle_deg: 0.0,
            lines: Vec::new(),
        };
    }
    let bin = binarize_sauvola(grey, SAUVOLA_WINDOW, SAUVOLA_K);
    let (angle, skew_ratio) = estimate_skew_scored(&bin);
    let rotated = angle.abs() > 0.05 && skew_ratio >= 1.15;
    let angle = if rotated { angle } else { 0.0 };
    let dsk = Deskew::new(grey.w, grey.h, angle, angle.abs() > SKEW_NARROW_DEG);
    let (mut work, mut work_bin) = if rotated {
        let bg = grey.median_value();
        let rg = rotate_grey_with(grey, &dsk, bg);
        let rb = binarize_sauvola(&rg, SAUVOLA_WINDOW, SAUVOLA_K);
        (rg, rb)
    } else {
        (grey.clone(), bin)
    };
    if let Some(cleaned) = suppress_background(&work, &work_bin) {
        work_bin = binarize_sauvola(&cleaned, SAUVOLA_WINDOW, SAUVOLA_K);
        work = cleaned;
    }
    let comps: Vec<Component> = speckle_filter(
        connected_components(&work_bin)
            .into_iter()
            .filter(|c| c.area >= 3 && c.bbox.height() >= 2)
            .collect(),
    );
    let mut lines = Vec::new();
    for band in plan_reading_order(&comps, work.w, work.h) {
        let mut order = band.clone();
        order.sort_by_key(|&i| comps[i].bbox.left);
        let mut heights: Vec<i32> = order.iter().map(|&i| comps[i].bbox.height()).collect();
        let med_h = median_i32(&mut heights).max(1);
        let x_height = med_h as f32;
        let mut bottoms: Vec<i32> = order
            .iter()
            .map(|&i| &comps[i])
            .filter(|c| c.bbox.height() as f32 >= 0.6 * med_h as f32)
            .map(|c| c.bbox.bottom)
            .collect();
        let baseline = if bottoms.is_empty() {
            comps[order[0]].bbox.bottom as f32
        } else {
            median_i32(&mut bottoms) as f32
        };
        let mut bbox = comps[order[0]].bbox;
        for &i in &order[1..] {
            bbox = bbox.union(&comps[i].bbox);
        }
        let words: Vec<WordBox> = split_words(&comps, &order, x_height)
            .into_iter()
            .map(|group| {
                let mut r = comps[group[0]].bbox;
                for &i in &group[1..] {
                    r = r.union(&comps[i].bbox);
                }
                if rotated {
                    r = dsk.map_rect_back(&r);
                }
                WordBox { rect: r }
            })
            .collect();
        let pad = 3;
        let grey_strip = work.crop_clamped(
            bbox.left - pad,
            bbox.top - pad,
            bbox.right + pad,
            bbox.bottom + pad,
        );
        let out_bbox = if rotated {
            dsk.map_rect_back(&bbox)
        } else {
            bbox
        };
        lines.push(Line {
            bbox: out_bbox,
            words,
            grey_strip,
            baseline_y: baseline,
            x_height,
        });
    }
    LayoutResult {
        angle_deg: angle,
        lines,
    }
}
