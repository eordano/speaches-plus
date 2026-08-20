use serde::{Deserialize, Serialize};

use super::ocr::OcrToken;
use super::spans::PiiSpan;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct LabeledRect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
    pub label: String,
}

pub fn map_spans(
    tokens: &[OcrToken],
    spans: &[PiiSpan],
    img_width: u32,
    img_height: u32,
) -> Vec<LabeledRect> {
    let mut result = Vec::new();

    for span in spans {
        let overlapping: Vec<&OcrToken> = tokens
            .iter()
            .filter(|t| t.start < span.end_exclusive && t.end_exclusive > span.start)
            .collect();

        if overlapping.is_empty() {
            continue;
        }

        let mut rects: Vec<(i32, i32, i32, i32)> = overlapping
            .iter()
            .map(|t| (t.rect.left, t.rect.top, t.rect.right, t.rect.bottom))
            .collect();

        rects.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

        let merged = merge_rects(&rects);

        for (left, top, right, bottom) in merged {
            let left = (left - 4).max(0);
            let top = (top - 4).max(0);
            let right = (right + 4).min(img_width as i32);
            let bottom = (bottom + 4).min(img_height as i32);

            result.push(LabeledRect {
                left,
                top,
                right,
                bottom,
                label: span.label.clone(),
            });
        }
    }

    result
}

fn merge_rects(rects: &[(i32, i32, i32, i32)]) -> Vec<(i32, i32, i32, i32)> {
    if rects.is_empty() {
        return Vec::new();
    }

    let mut merged: Vec<(i32, i32, i32, i32)> = Vec::new();
    merged.push(rects[0]);

    for &(left, top, right, bottom) in &rects[1..] {
        let last = merged.last_mut().unwrap();
        let h_gap = left - last.2;
        let min_height = (last.3 - last.1).min(bottom - top);
        let overlap_top = last.1.max(top);
        let overlap_bottom = last.3.min(bottom);
        let v_overlap = (overlap_bottom - overlap_top).max(0);

        let should_merge = h_gap < 8 && min_height > 0 && v_overlap * 2 > min_height;

        if should_merge {
            last.0 = last.0.min(left);
            last.1 = last.1.min(top);
            last.2 = last.2.max(right);
            last.3 = last.3.max(bottom);
        } else {
            merged.push((left, top, right, bottom));
        }
    }

    merged
}
