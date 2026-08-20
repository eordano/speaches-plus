use anyhow::{Context, Result};
use image::{Rgba, RgbaImage};
use std::io::Cursor;

#[derive(Debug, Clone)]
pub struct RedactRect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FillMode {
    Solid,
    Shuffle,
}

pub fn render_redactions(
    image_bytes: &[u8],
    rects: &[RedactRect],
    fill_mode: FillMode,
    fill_color: [u8; 3],
) -> Result<Vec<u8>> {
    let img = nv_imgdec::decode_oriented(image_bytes).context("decode image")?;
    let mut rgba = img.to_rgba8();

    let (w, h) = (rgba.width(), rgba.height());
    for rect in rects {
        let Some((x0, y0, x1, y1)) = clamp_rect(rect, w, h) else {
            continue;
        };

        match fill_mode {
            FillMode::Solid => {
                fill_solid(&mut rgba, x0, y0, x1, y1, fill_color);
            }
            FillMode::Shuffle => {
                fill_shuffle(&mut rgba, x0, y0, x1, y1, rect);
            }
        }
    }

    let mut buf = Cursor::new(Vec::new());
    let encoder = image::codecs::png::PngEncoder::new(&mut buf);
    image::ImageEncoder::write_image(
        encoder,
        rgba.as_raw(),
        rgba.width(),
        rgba.height(),
        image::ExtendedColorType::Rgba8,
    )
    .context("encode PNG")?;

    Ok(buf.into_inner())
}

fn clamp_rect(rect: &RedactRect, width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
    let x0 = rect.left.clamp(0, width as i32) as u32;
    let y0 = rect.top.clamp(0, height as i32) as u32;
    let x1 = rect.right.clamp(0, width as i32) as u32;
    let y1 = rect.bottom.clamp(0, height as i32) as u32;
    (x0 < x1 && y0 < y1).then_some((x0, y0, x1, y1))
}

fn fill_solid(img: &mut RgbaImage, x0: u32, y0: u32, x1: u32, y1: u32, color: [u8; 3]) {
    let pixel = Rgba([color[0], color[1], color[2], 255]);
    for y in y0..y1 {
        for x in x0..x1 {
            img.put_pixel(x, y, pixel);
        }
    }
}

fn fill_shuffle(img: &mut RgbaImage, x0: u32, y0: u32, x1: u32, y1: u32, rect: &RedactRect) {
    let mut buckets: [u32; 4096] = [0; 4096];
    let mut color_sums: [(u64, u64, u64); 4096] = [(0, 0, 0); 4096];

    for y in y0..y1 {
        for x in x0..x1 {
            let p = img.get_pixel(x, y);
            let bucket = quantize_pixel(p);
            buckets[bucket] += 1;
            color_sums[bucket].0 += p[0] as u64;
            color_sums[bucket].1 += p[1] as u64;
            color_sums[bucket].2 += p[2] as u64;
        }
    }

    let (top1_idx, top2_idx) = find_top_two(&buckets);

    let c1 = centroid(&color_sums[top1_idx], buckets[top1_idx]);
    let c2 = centroid(&color_sums[top2_idx], buckets[top2_idx]);

    let seed = (rect.left ^ rect.top ^ rect.right) as u32;
    let mut rng = SimpleRng::new(seed);

    for y in y0..y1 {
        for x in x0..x1 {
            let color = if rng.next_bit() { c1 } else { c2 };
            img.put_pixel(x, y, color);
        }
    }
}

fn quantize_pixel(p: &Rgba<u8>) -> usize {
    let r = (p[0] >> 4) as usize;
    let g = (p[1] >> 4) as usize;
    let b = (p[2] >> 4) as usize;
    (r << 8) | (g << 4) | b
}

fn find_top_two(buckets: &[u32; 4096]) -> (usize, usize) {
    let mut top1 = 0usize;
    let mut top2 = 0usize;

    for i in 1..4096 {
        if buckets[i] > buckets[top1] {
            top2 = top1;
            top1 = i;
        } else if buckets[i] > buckets[top2] && i != top1 {
            top2 = i;
        }
    }

    (top1, top2)
}

fn centroid(sums: &(u64, u64, u64), count: u32) -> Rgba<u8> {
    if count == 0 {
        return Rgba([0, 0, 0, 255]);
    }
    let c = count as u64;
    Rgba([
        (sums.0 / c) as u8,
        (sums.1 / c) as u8,
        (sums.2 / c) as u8,
        255,
    ])
}

struct SimpleRng {
    state: u32,
}

impl SimpleRng {
    fn new(seed: u32) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_bit(&mut self) -> bool {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        (self.state & 1) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(left: i32, top: i32, right: i32, bottom: i32) -> RedactRect {
        RedactRect {
            left,
            top,
            right,
            bottom,
        }
    }

    #[test]
    fn a_rect_entirely_left_of_the_image_redacts_nothing() {
        assert_eq!(clamp_rect(&rect(-100, -100, -50, -50), 900, 340), None);
    }

    #[test]
    fn a_rect_entirely_right_of_the_image_redacts_nothing() {
        assert_eq!(clamp_rect(&rect(1200, 400, 1300, 500), 900, 340), None);
    }

    #[test]
    fn a_straddling_rect_is_clamped_to_the_image() {
        assert_eq!(
            clamp_rect(&rect(-20, -10, 200, 60), 900, 340),
            Some((0, 0, 200, 60))
        );
        assert_eq!(
            clamp_rect(&rect(800, 300, 5000, 9000), 900, 340),
            Some((800, 300, 900, 340))
        );
    }

    #[test]
    fn an_inverted_rect_redacts_nothing() {
        assert_eq!(clamp_rect(&rect(600, 200, 100, 50), 900, 340), None);
    }

    #[test]
    fn an_off_image_rect_leaves_every_pixel_untouched() {
        let mut img = RgbaImage::from_pixel(8, 8, Rgba([255, 255, 255, 255]));
        let before = img.as_raw().clone();
        let (w, h) = (img.width(), img.height());
        if let Some((x0, y0, x1, y1)) = clamp_rect(&rect(-9, -9, -1, -1), w, h) {
            fill_solid(&mut img, x0, y0, x1, y1, [0, 0, 0]);
        }
        assert_eq!(img.as_raw(), &before);
    }
}
