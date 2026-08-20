use crate::{Error, GreyImage};

pub fn load(bytes: &[u8]) -> Result<GreyImage, Error> {
    let rgb = nv_imgdec::decode_rgb8(bytes).map_err(|e| Error::Decode(format!("{e:#}")))?;
    let (w, h) = rgb.dimensions();
    let mut g = GreyImage::new(w as usize, h as usize);
    for (i, p) in rgb.pixels().enumerate() {
        let [r, gr, b] = p.0;
        g.data[i] = ((299 * r as u32 + 587 * gr as u32 + 114 * b as u32 + 500) / 1000) as u8;
    }
    Ok(g)
}

impl GreyImage {
    pub fn from_fn(w: usize, h: usize, mut f: impl FnMut(usize, usize) -> u8) -> Self {
        let mut g = GreyImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                g.data[y * w + x] = f(x, y);
            }
        }
        g
    }

    pub fn invert(&self) -> Self {
        GreyImage {
            w: self.w,
            h: self.h,
            data: self.data.iter().map(|v| 255 - v).collect(),
        }
    }

    pub fn median_value(&self) -> u8 {
        if self.data.is_empty() {
            return 0;
        }
        let mut hist = [0u32; 256];
        for &v in &self.data {
            hist[v as usize] += 1;
        }
        let half = (self.data.len() as u32).div_ceil(2);
        let mut acc = 0u32;
        for (v, &n) in hist.iter().enumerate() {
            acc += n;
            if acc >= half {
                return v as u8;
            }
        }
        255
    }

    pub fn crop_clamped(&self, left: i32, top: i32, right: i32, bottom: i32) -> GreyImage {
        let l = left.max(0).min(self.w as i32) as usize;
        let t = top.max(0).min(self.h as i32) as usize;
        let r = right.max(l as i32).min(self.w as i32) as usize;
        let b = bottom.max(t as i32).min(self.h as i32) as usize;
        let mut out = GreyImage::new(r - l, b - t);
        for y in t..b {
            let src = &self.data[y * self.w + l..y * self.w + r];
            let dy = y - t;
            out.data[dy * (r - l)..(dy + 1) * (r - l)].copy_from_slice(src);
        }
        out
    }

    pub fn sample_bilinear(&self, x: f32, y: f32, bg: u8) -> f32 {
        if self.w == 0 || self.h == 0 {
            return bg as f32;
        }
        if x < -1.0 || y < -1.0 || x > self.w as f32 || y > self.h as f32 {
            return bg as f32;
        }
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;
        let px = |xi: i64, yi: i64| -> f32 {
            if xi < 0 || yi < 0 || xi >= self.w as i64 || yi >= self.h as i64 {
                bg as f32
            } else {
                self.data[yi as usize * self.w + xi as usize] as f32
            }
        };
        let x0i = x0 as i64;
        let y0i = y0 as i64;
        let top = px(x0i, y0i) * (1.0 - fx) + px(x0i + 1, y0i) * fx;
        let bot = px(x0i, y0i + 1) * (1.0 - fx) + px(x0i + 1, y0i + 1) * fx;
        top * (1.0 - fy) + bot * fy
    }
}

pub struct IntegralImage {
    pub w: usize,
    pub h: usize,
    sum: Vec<u64>,
    sq: Vec<u64>,
}

impl IntegralImage {
    pub fn build(g: &GreyImage) -> Self {
        let w = g.w;
        let h = g.h;
        let stride = w + 1;
        let mut sum = vec![0u64; stride * (h + 1)];
        let mut sq = vec![0u64; stride * (h + 1)];
        for y in 0..h {
            let mut row_sum = 0u64;
            let mut row_sq = 0u64;
            for x in 0..w {
                let v = g.data[y * w + x] as u64;
                row_sum += v;
                row_sq += v * v;
                sum[(y + 1) * stride + x + 1] = sum[y * stride + x + 1] + row_sum;
                sq[(y + 1) * stride + x + 1] = sq[y * stride + x + 1] + row_sq;
            }
        }
        IntegralImage { w, h, sum, sq }
    }

    pub fn planes(&self) -> (&[u64], &[u64], usize) {
        (&self.sum, &self.sq, self.w + 1)
    }

    pub fn rect_sum(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> u64 {
        let s = self.w + 1;
        self.sum[y1 * s + x1] + self.sum[y0 * s + x0]
            - self.sum[y0 * s + x1]
            - self.sum[y1 * s + x0]
    }

    pub fn rect_sq_sum(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> u64 {
        let s = self.w + 1;
        self.sq[y1 * s + x1] + self.sq[y0 * s + x0] - self.sq[y0 * s + x1] - self.sq[y1 * s + x0]
    }
}
