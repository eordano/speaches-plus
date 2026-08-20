pub use image;

use std::io::Cursor;

use anyhow::{bail, Context, Result};
use image::ImageDecoder;

pub const MAX_DECODE_PIXELS_CAPS_RGB8_BUFFER_AT_512MB: u64 = 178_000_000;

pub const ALPHA_MATTE_WHITE_KEEPS_TRANSPARENT_DOC_INK_READABLE: u16 = 255;

pub const SUPPORTED_IMAGE_FORMATS: &str = "png, jpeg, webp, gif, bmp, tiff";

pub fn decode_oriented(bytes: &[u8]) -> Result<image::DynamicImage> {
    let reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .context("sniff image format")?;
    if reader.format().is_none() {
        bail!("unrecognized image format (supported: {SUPPORTED_IMAGE_FORMATS})");
    }
    let mut decoder = reader.into_decoder().context("open image decoder")?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let (w, h) = decoder.dimensions();
    if w == 0 || h == 0 {
        bail!("image has zero width or height");
    }
    let px = w as u64 * h as u64;
    if px > MAX_DECODE_PIXELS_CAPS_RGB8_BUFFER_AT_512MB {
        bail!(
            "image is {w}x{h} ({px} pixels), over the \
             {MAX_DECODE_PIXELS_CAPS_RGB8_BUFFER_AT_512MB}-pixel decode cap"
        );
    }
    let mut img = image::DynamicImage::from_decoder(decoder).context("decode image")?;
    img.apply_orientation(orientation);
    Ok(img)
}

pub fn decode_rgb8(bytes: &[u8]) -> Result<image::RgbImage> {
    let img = decode_oriented(bytes)?;
    if !img.color().has_alpha() {
        return Ok(img.to_rgb8());
    }
    let rgba = img.into_rgba8();
    let (w, h) = rgba.dimensions();
    let matte = ALPHA_MATTE_WHITE_KEEPS_TRANSPARENT_DOC_INK_READABLE;
    let mut out = image::RgbImage::new(w, h);
    for (o, p) in out.pixels_mut().zip(rgba.pixels()) {
        let a = p[3] as u16;
        for c in 0..3 {
            o[c] = ((p[c] as u16 * a + matte * (255 - a)) / 255) as u8;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient_rgb(w: u32, h: u32) -> image::RgbImage {
        image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x * 40) as u8, (y * 40) as u8, 200])
        })
    }

    fn encode(img: image::DynamicImage, fmt: image::ImageFormat) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        img.write_to(&mut buf, fmt).expect("encode fixture");
        buf.into_inner()
    }

    #[test]
    fn every_supported_container_decodes_to_the_same_pixels() {
        use image::ImageFormat as F;
        let src = gradient_rgb(6, 4);
        for fmt in [F::Png, F::WebP, F::Gif, F::Bmp, F::Tiff] {
            let bytes = encode(image::DynamicImage::ImageRgb8(src.clone()), fmt);
            let got = decode_rgb8(&bytes)
                .unwrap_or_else(|e| panic!("{fmt:?} must decode, got {e:#}"));
            assert_eq!(got.dimensions(), (6, 4), "{fmt:?} changed dimensions");
            for (a, b) in got.pixels().zip(src.pixels()) {
                for c in 0..3 {
                    assert!(
                        (a[c] as i16 - b[c] as i16).abs() <= 8,
                        "{fmt:?} pixel drifted beyond palette quantization: {a:?} vs {b:?}"
                    );
                }
            }
        }
        let jpg = encode(image::DynamicImage::ImageRgb8(src.clone()), F::Jpeg);
        let got = decode_rgb8(&jpg).expect("jpeg must decode");
        assert_eq!(got.dimensions(), (6, 4));
    }

    #[test]
    fn grayscale_and_sixteen_bit_sources_decode() {
        let l8 = image::GrayImage::from_fn(5, 3, |x, _| image::Luma([(x * 50) as u8]));
        let bytes = encode(image::DynamicImage::ImageLuma8(l8), image::ImageFormat::Png);
        let got = decode_rgb8(&bytes).expect("L8 png");
        assert_eq!(got.dimensions(), (5, 3));
        assert_eq!(got.get_pixel(4, 0), &image::Rgb([200, 200, 200]));

        let r16 = image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::from_fn(3, 3, |x, y| {
            image::Rgb([x as u16 * 20_000, y as u16 * 20_000, 65_535])
        });
        let bytes = encode(image::DynamicImage::ImageRgb16(r16), image::ImageFormat::Png);
        let got = decode_rgb8(&bytes).expect("RGB16 png");
        assert_eq!(got.dimensions(), (3, 3));
        assert_eq!(got.get_pixel(0, 0)[2], 255, "16-bit full-scale blue must map to 255");
    }

    #[test]
    fn transparent_pixels_flatten_to_white_not_black() {
        let rgba = image::RgbaImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                image::Rgba([0, 0, 0, 0])
            } else {
                image::Rgba([0, 0, 0, 128])
            }
        });
        let bytes = encode(image::DynamicImage::ImageRgba8(rgba), image::ImageFormat::Png);
        let got = decode_rgb8(&bytes).expect("rgba png");
        assert_eq!(
            got.get_pixel(0, 0),
            &image::Rgb([255, 255, 255]),
            "fully transparent background must read as paper, not black"
        );
        let half = got.get_pixel(1, 0)[0];
        assert!(
            (126..=128).contains(&half),
            "half-transparent black must blend toward white, got {half}"
        );
    }

    fn jpeg_with_exif_orientation(img: &image::RgbImage, orientation: u16) -> Vec<u8> {
        let jpg = encode(
            image::DynamicImage::ImageRgb8(img.clone()),
            image::ImageFormat::Jpeg,
        );
        let mut app1 = vec![0xFF, 0xE1, 0, 34];
        app1.extend_from_slice(b"Exif\0\0");
        app1.extend_from_slice(&[0x49, 0x49, 0x2A, 0x00, 8, 0, 0, 0]);
        app1.extend_from_slice(&1u16.to_le_bytes());
        app1.extend_from_slice(&0x0112u16.to_le_bytes());
        app1.extend_from_slice(&3u16.to_le_bytes());
        app1.extend_from_slice(&1u32.to_le_bytes());
        app1.extend_from_slice(&orientation.to_le_bytes());
        app1.extend_from_slice(&[0, 0]);
        app1.extend_from_slice(&0u32.to_le_bytes());
        let mut out = jpg[..2].to_vec();
        out.extend_from_slice(&app1);
        out.extend_from_slice(&jpg[2..]);
        out
    }

    #[test]
    fn exif_rotated_phone_jpegs_are_uprighted_before_any_model_sees_them() {
        let src = image::RgbImage::from_fn(4, 2, |x, y| {
            if x == 3 && y == 0 {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([0, 0, 255])
            }
        });
        let bytes = jpeg_with_exif_orientation(&src, 6);
        let got = decode_rgb8(&bytes).expect("exif jpeg");
        assert_eq!(
            got.dimensions(),
            (2, 4),
            "orientation 6 must rotate 90 degrees, swapping dimensions"
        );
        let p = got.get_pixel(1, 3);
        assert!(
            p[0] > 128 && p[2] < 128,
            "marker pixel must land where a 90-degree clockwise rotation puts it, got {p:?}"
        );
    }

    fn bmp_header_claiming(w: u32, h: u32) -> Vec<u8> {
        let mut b = b"BM".to_vec();
        b.extend_from_slice(&54u32.to_le_bytes());
        b.extend_from_slice(&[0; 4]);
        b.extend_from_slice(&54u32.to_le_bytes());
        b.extend_from_slice(&40u32.to_le_bytes());
        b.extend_from_slice(&(w as i32).to_le_bytes());
        b.extend_from_slice(&(h as i32).to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&24u16.to_le_bytes());
        b.extend_from_slice(&[0; 24]);
        b
    }

    #[test]
    fn decompression_bombs_fail_before_allocating() {
        let err = decode_rgb8(&bmp_header_claiming(20_000, 20_000))
            .expect_err("400-megapixel claim must be rejected")
            .to_string();
        assert!(err.contains("decode cap"), "unexpected error: {err}");
    }

    #[test]
    fn junk_bytes_name_the_supported_formats() {
        let err = decode_rgb8(b"definitely not an image")
            .expect_err("junk must not decode")
            .to_string();
        assert!(err.contains("supported: png, jpeg"), "unexpected error: {err}");
    }

    #[test]
    fn corpus_sweep_real_world_files_decode_or_error_cleanly() {
        let Ok(dir) = std::env::var("NV_IMGDEC_CORPUS") else {
            return;
        };
        let mut ok = 0usize;
        let mut rejected = 0usize;
        let mut failures: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(&dir).expect("read corpus dir") {
            let path = entry.expect("corpus dir entry").path();
            if !path.is_file() {
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let bytes = std::fs::read(&path).expect("read corpus file");
            let expect_reject = name.starts_with("bad-");
            match (decode_rgb8(&bytes), expect_reject) {
                (Ok(img), false) => {
                    assert!(img.width() > 0 && img.height() > 0, "{name}: empty decode");
                    ok += 1;
                    eprintln!("ok       {name}: {}x{}", img.width(), img.height());
                }
                (Err(e), true) => {
                    rejected += 1;
                    eprintln!("rejected {name}: {e:#}");
                }
                (Ok(img), true) => failures.push(format!(
                    "{name}: decoded {}x{} but a bad-* file must be rejected",
                    img.width(),
                    img.height()
                )),
                (Err(e), false) => failures.push(format!("{name}: {e:#}")),
            }
        }
        assert!(
            failures.is_empty(),
            "corpus sweep: {ok} decoded, {rejected} rejected, failures:\n{}",
            failures.join("\n")
        );
        eprintln!("corpus sweep: {ok} decoded, {rejected} rejected as expected");
    }

    #[test]
    fn truncated_files_error_instead_of_panicking() {
        let full = encode(
            image::DynamicImage::ImageRgb8(gradient_rgb(32, 32)),
            image::ImageFormat::Png,
        );
        for cut in [8, 24, full.len() / 2] {
            assert!(
                decode_rgb8(&full[..cut]).is_err(),
                "truncated png at {cut} bytes must error cleanly"
            );
        }
    }
}
