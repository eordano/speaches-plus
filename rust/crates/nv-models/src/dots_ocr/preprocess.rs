use anyhow::Result;

use crate::deepseek_ocr::preprocess::{resize_rgb, RgbImage};

pub const PATCH_SIZE: usize = 14;
pub const MERGE_SIZE: usize = 2;
pub const FACTOR: usize = PATCH_SIZE * MERGE_SIZE;
pub const TEMPORAL_PATCH_SIZE: usize = 1;
pub const PATCH_DIM: usize = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE;

pub const DEFAULT_MIN_PIXELS: usize = 3136;
pub const DEFAULT_MAX_PIXELS: usize = 11_289_600;

pub const IMAGE_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const IMAGE_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

#[derive(Clone, Copy, Debug)]
pub struct PixelBudget {
    pub min_pixels: usize,
    pub max_pixels: usize,
}

impl Default for PixelBudget {
    fn default() -> Self {
        Self {
            min_pixels: DEFAULT_MIN_PIXELS,
            max_pixels: DEFAULT_MAX_PIXELS,
        }
    }
}

impl PixelBudget {
    pub fn from_env() -> Self {
        let mut b = Self::default();
        if let Ok(v) = std::env::var("NV_DOTS_MIN_PIXELS") {
            if let Ok(n) = v.trim().parse::<usize>() {
                b.min_pixels = n.max(FACTOR * FACTOR);
            }
        }
        if let Ok(v) = std::env::var("NV_DOTS_MAX_PIXELS") {
            if let Ok(n) = v.trim().parse::<usize>() {
                b.max_pixels = n.max(b.min_pixels);
            }
        }
        b
    }
}

use crate::deepseek_ocr::preprocess::round_half_even;

fn round_by_factor(n: f64, factor: usize) -> usize {
    (round_half_even(n / factor as f64).max(1) as usize) * factor
}

fn floor_by_factor(n: f64, factor: usize) -> usize {
    ((n / factor as f64).floor().max(1.0) as usize) * factor
}

fn ceil_by_factor(n: f64, factor: usize) -> usize {
    ((n / factor as f64).ceil().max(1.0) as usize) * factor
}

pub fn smart_resize(h: usize, w: usize, budget: PixelBudget) -> Result<(usize, usize)> {
    anyhow::ensure!(h > 0 && w > 0, "smart_resize: empty image {w}x{h}");
    let ratio = h.max(w) as f64 / h.min(w) as f64;
    anyhow::ensure!(
        ratio <= 200.0,
        "smart_resize: aspect ratio {ratio:.1} exceeds 200 ({w}x{h})"
    );
    let mut h_bar = round_by_factor(h as f64, FACTOR);
    let mut w_bar = round_by_factor(w as f64, FACTOR);
    if h_bar * w_bar > budget.max_pixels {
        let beta = ((h * w) as f64 / budget.max_pixels as f64).sqrt();
        h_bar = floor_by_factor(h as f64 / beta, FACTOR);
        w_bar = floor_by_factor(w as f64 / beta, FACTOR);
    } else if h_bar * w_bar < budget.min_pixels {
        let beta = (budget.min_pixels as f64 / (h * w) as f64).sqrt();
        h_bar = ceil_by_factor(h as f64 * beta, FACTOR);
        w_bar = ceil_by_factor(w as f64 * beta, FACTOR);
    }
    Ok((h_bar, w_bar))
}

#[derive(Clone, Debug)]
pub struct PreparedImage {
    pub patches: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
    pub resized_h: usize,
    pub resized_w: usize,
    pub orig_h: usize,
    pub orig_w: usize,
}

impl PreparedImage {
    pub fn num_patches(&self) -> usize {
        self.grid_h * self.grid_w
    }

    pub fn num_vision_tokens(&self) -> usize {
        self.grid_h * self.grid_w / (MERGE_SIZE * MERGE_SIZE)
    }

    pub fn scale_to_orig(&self, bbox: [f32; 4]) -> [f32; 4] {
        let sx = self.orig_w as f32 / self.resized_w as f32;
        let sy = self.orig_h as f32 / self.resized_h as f32;
        [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]
    }
}

pub fn prepare(img: &RgbImage, budget: PixelBudget) -> Result<PreparedImage> {
    let (rh, rw) = smart_resize(img.h, img.w, budget)?;
    let resized = resize_rgb(img, rw, rh);
    let grid_h = rh / PATCH_SIZE;
    let grid_w = rw / PATCH_SIZE;
    anyhow::ensure!(
        grid_h.is_multiple_of(MERGE_SIZE) && grid_w.is_multiple_of(MERGE_SIZE),
        "prepare: grid {grid_h}x{grid_w} not divisible by merge size {MERGE_SIZE}"
    );

    let mut normed = vec![0f32; 3 * rh * rw];
    for c in 0..3 {
        let inv = 1.0 / IMAGE_STD[c];
        let mean = IMAGE_MEAN[c];
        let plane = &mut normed[c * rh * rw..(c + 1) * rh * rw];
        for i in 0..rh * rw {
            plane[i] = (resized.data[i * 3 + c] as f32 / 255.0 - mean) * inv;
        }
    }

    let n = grid_h * grid_w;
    let mut patches = vec![0f32; n * PATCH_DIM];
    let blocks_h = grid_h / MERGE_SIZE;
    let blocks_w = grid_w / MERGE_SIZE;
    let mut row = 0usize;
    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            for sy in 0..MERGE_SIZE {
                for sx in 0..MERGE_SIZE {
                    let py = (by * MERGE_SIZE + sy) * PATCH_SIZE;
                    let px = (bx * MERGE_SIZE + sx) * PATCH_SIZE;
                    let dst = &mut patches[row * PATCH_DIM..(row + 1) * PATCH_DIM];
                    for c in 0..3 {
                        let plane = &normed[c * rh * rw..(c + 1) * rh * rw];
                        for ky in 0..PATCH_SIZE {
                            let src = (py + ky) * rw + px;
                            let off = c * PATCH_SIZE * PATCH_SIZE + ky * PATCH_SIZE;
                            dst[off..off + PATCH_SIZE]
                                .copy_from_slice(&plane[src..src + PATCH_SIZE]);
                        }
                    }
                    row += 1;
                }
            }
        }
    }

    Ok(PreparedImage {
        patches,
        grid_h,
        grid_w,
        resized_h: rh,
        resized_w: rw,
        orig_h: img.h,
        orig_w: img.w,
    })
}

pub fn position_ids(grid_h: usize, grid_w: usize) -> Vec<(u32, u32)> {
    let blocks_h = grid_h / MERGE_SIZE;
    let blocks_w = grid_w / MERGE_SIZE;
    let mut out = Vec::with_capacity(grid_h * grid_w);
    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            for sy in 0..MERGE_SIZE {
                for sx in 0..MERGE_SIZE {
                    out.push(((by * MERGE_SIZE + sy) as u32, (bx * MERGE_SIZE + sx) as u32));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_rounds_to_factor() {
        let (h, w) = smart_resize(2200, 1700, PixelBudget::default()).unwrap();
        assert_eq!(h % FACTOR, 0);
        assert_eq!(w % FACTOR, 0);
        assert_eq!((h, w), (2212, 1708));
    }

    #[test]
    fn smart_resize_shrinks_over_max_pixels() {
        let budget = PixelBudget {
            min_pixels: 3136,
            max_pixels: 100_000,
        };
        let (h, w) = smart_resize(2000, 1000, budget).unwrap();
        assert!(h * w <= budget.max_pixels, "{h}x{w}");
        assert_eq!(h % FACTOR, 0);
        assert_eq!(w % FACTOR, 0);
        let ar = h as f64 / w as f64;
        assert!((ar - 2.0).abs() < 0.15, "aspect drifted: {ar}");
    }

    #[test]
    fn smart_resize_grows_under_min_pixels() {
        let budget = PixelBudget::default();
        let (h, w) = smart_resize(10, 10, budget).unwrap();
        assert!(h * w >= budget.min_pixels, "{h}x{w}");
        assert_eq!(h % FACTOR, 0);
        assert_eq!(w % FACTOR, 0);
    }

    #[test]
    fn smart_resize_rejects_extreme_aspect() {
        assert!(smart_resize(1, 500, PixelBudget::default()).is_err());
    }

    #[test]
    fn patchify_is_merge_block_major() {
        let img = RgbImage::from_fn(FACTOR * 2, FACTOR * 2, |x, y| {
            let px = (x / PATCH_SIZE) as u8;
            let py = (y / PATCH_SIZE) as u8;
            [px, py, 0]
        });
        let prep = prepare(&img, PixelBudget::default()).unwrap();
        assert_eq!((prep.resized_h, prep.resized_w), (FACTOR * 2, FACTOR * 2));
        assert_eq!((prep.grid_h, prep.grid_w), (4, 4));
        assert_eq!(prep.num_patches(), 16);
        assert_eq!(prep.num_vision_tokens(), 4);

        let ids = position_ids(prep.grid_h, prep.grid_w);
        assert_eq!(ids.len(), 16);
        assert_eq!(&ids[..4], &[(0, 0), (0, 1), (1, 0), (1, 1)]);
        assert_eq!(&ids[4..8], &[(0, 2), (0, 3), (1, 2), (1, 3)]);
        assert_eq!(&ids[8..12], &[(2, 0), (2, 1), (3, 0), (3, 1)]);
        assert_eq!(&ids[12..], &[(2, 2), (2, 3), (3, 2), (3, 3)]);

        let decode = |row: usize| -> (u8, u8) {
            let d = &prep.patches[row * PATCH_DIM..(row + 1) * PATCH_DIM];
            let r = (d[0] * IMAGE_STD[0] + IMAGE_MEAN[0]) * 255.0;
            let g = (d[PATCH_SIZE * PATCH_SIZE] * IMAGE_STD[1] + IMAGE_MEAN[1]) * 255.0;
            (r.round() as u8, g.round() as u8)
        };
        for (row, (py, px)) in ids.iter().enumerate() {
            assert_eq!(decode(row), (*px as u8, *py as u8), "row {row}");
        }
    }

    #[test]
    fn prepared_scales_bbox_back_to_original() {
        let img = RgbImage::filled(100, 200, [128, 128, 128]);
        let prep = prepare(&img, PixelBudget::default()).unwrap();
        let full = prep.scale_to_orig([0.0, 0.0, prep.resized_w as f32, prep.resized_h as f32]);
        assert!((full[2] - 100.0).abs() < 1e-3);
        assert!((full[3] - 200.0).abs() < 1e-3);
    }
}
