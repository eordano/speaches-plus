use crate::binarize::ink_is_dark;
use crate::GreyImage;

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedLine {
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl NormalizedLine {
    pub fn at(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.w + col]
    }
}

pub const PAD_COLS: usize = 2;

pub fn normalize_line(strip: &GreyImage, target_h: usize) -> NormalizedLine {
    if strip.w == 0 || strip.h == 0 || target_h == 0 {
        return NormalizedLine {
            h: target_h,
            w: 0,
            data: Vec::new(),
        };
    }
    let src;
    let g = if ink_is_dark(strip) {
        strip
    } else {
        src = strip.invert();
        &src
    };
    let scale = target_h as f32 / g.h as f32;
    let out_w = ((g.w as f32 * scale).round() as usize).max(1);
    let w = out_w + 2 * PAD_COLS;
    let mut data = vec![0.0f32; target_h * w];
    let sx = g.w as f32 / out_w as f32;
    let sy = g.h as f32 / target_h as f32;
    for row in 0..target_h {
        let yin = (row as f32 + 0.5) * sy - 0.5;
        for col in 0..out_w {
            let xin = (col as f32 + 0.5) * sx - 0.5;
            let v = g.sample_bilinear(xin, yin, 255);
            data[row * w + PAD_COLS + col] = (255.0 - v) / 255.0;
        }
    }
    NormalizedLine {
        h: target_h,
        w,
        data,
    }
}
