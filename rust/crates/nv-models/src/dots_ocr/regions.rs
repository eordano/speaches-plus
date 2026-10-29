use super::parse::LayoutPage;

pub const LAYOUT_MAX_PIXELS_ENV: &str = "NV_DOTS_LAYOUT_MAX_PIXELS";
pub const CROP_MAX_PIXELS_ENV: &str = "NV_DOTS_CROP_MAX_PIXELS";

pub const DEFAULT_LAYOUT_MAX_PIXELS_1MP_KEEPS_THE_DETECT_PASS_CHEAP: usize = 1_000_000;
pub const DEFAULT_CROP_MAX_PIXELS_500K_IS_ABOUT_640_VISION_TOKENS: usize = 500_000;

const VERTICAL_MARGIN_FRAC_COVERS_HANDWRITING_DESCENDERS: f32 = 0.08;
const HORIZONTAL_MARGIN_FRAC: f32 = 0.03;
const MERGE_MIN_OVERLAP_FRAC_OF_SMALLER_BECAUSE_ADJACENT_LAYOUT_BANDS_MERELY_TOUCH: f32 = 0.2;
const ROW_MEMBERSHIP_MIN_OVERLAP_FRAC_OF_REGION_HEIGHT: f32 = 0.5;
const MIN_REGION_SIDE_PX: f32 = 6.0;
const MAX_REGION_ASPECT_SMART_RESIZE_REJECTS_ABOVE_200: f32 = 150.0;

pub const LAYOUT_USABLE_MIN_COVERAGE_FRAC_BELOW_THIS_THE_DETECTOR_MISSED_THE_PAGE: f32 = 0.15;

pub fn env_pixels(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    pub bbox: [f32; 4],
    pub category: Option<String>,
}

impl Region {
    pub fn width(&self) -> f32 {
        self.bbox[2] - self.bbox[0]
    }

    pub fn height(&self) -> f32 {
        self.bbox[3] - self.bbox[1]
    }

    pub fn pixel_bounds(&self, img_w: usize, img_h: usize) -> (usize, usize, usize, usize) {
        let x0 = (self.bbox[0].floor().max(0.0) as usize).min(img_w.saturating_sub(1));
        let y0 = (self.bbox[1].floor().max(0.0) as usize).min(img_h.saturating_sub(1));
        let x1 = (self.bbox[2].ceil() as usize).clamp(x0 + 1, img_w);
        let y1 = (self.bbox[3].ceil() as usize).clamp(y0 + 1, img_h);
        (x0, y0, x1, y1)
    }
}

fn area(b: &[f32; 4]) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

fn intersection_area(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let w = a[2].min(b[2]) - a[0].max(b[0]);
    let h = a[3].min(b[3]) - a[1].max(b[1]);
    w.max(0.0) * h.max(0.0)
}

fn overlaps_substantially(a: &[f32; 4], b: &[f32; 4]) -> bool {
    let smaller = area(a).min(area(b));
    smaller > 0.0
        && intersection_area(a, b) / smaller
            >= MERGE_MIN_OVERLAP_FRAC_OF_SMALLER_BECAUSE_ADJACENT_LAYOUT_BANDS_MERELY_TOUCH
}

fn union(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

fn expand_and_clamp(b: [f32; 4], img_w: f32, img_h: f32) -> [f32; 4] {
    let my = (b[3] - b[1]) * VERTICAL_MARGIN_FRAC_COVERS_HANDWRITING_DESCENDERS;
    let mx = (b[2] - b[0]) * HORIZONTAL_MARGIN_FRAC;
    [
        (b[0] - mx).max(0.0),
        (b[1] - my).max(0.0),
        (b[2] + mx).min(img_w),
        (b[3] + my).min(img_h),
    ]
}

fn usable(b: &[f32; 4]) -> bool {
    let w = b[2] - b[0];
    let h = b[3] - b[1];
    if w < MIN_REGION_SIDE_PX || h < MIN_REGION_SIDE_PX {
        return false;
    }
    w.max(h) / w.min(h) <= MAX_REGION_ASPECT_SMART_RESIZE_REJECTS_ABOVE_200
}

fn merge_substantial_overlaps(regions: &mut Vec<Region>) {
    let mut merged_any = true;
    while merged_any {
        merged_any = false;
        'outer: for i in 0..regions.len() {
            for j in (i + 1)..regions.len() {
                if overlaps_substantially(&regions[i].bbox, &regions[j].bbox) {
                    let other = regions.remove(j);
                    regions[i].bbox = union(&regions[i].bbox, &other.bbox);
                    if regions[i].category != other.category {
                        regions[i].category = None;
                    }
                    merged_any = true;
                    break 'outer;
                }
            }
        }
    }
}

fn largest_gap_split(regions: &[Region], lo_axis: usize) -> Option<(f32, usize, Vec<usize>)> {
    let hi_axis = lo_axis + 2;
    let mut idx: Vec<usize> = (0..regions.len()).collect();
    idx.sort_by(|&a, &b| {
        regions[a].bbox[lo_axis]
            .partial_cmp(&regions[b].bbox[lo_axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut best: Option<(f32, usize)> = None;
    let mut running_end = regions[idx[0]].bbox[hi_axis];
    for (pos, &i) in idx.iter().enumerate().skip(1) {
        let gap = regions[i].bbox[lo_axis] - running_end;
        if gap > 0.0 && best.is_none_or(|(g, _)| gap > g) {
            best = Some((gap, pos));
        }
        running_end = running_end.max(regions[i].bbox[hi_axis]);
    }
    best.map(|(gap, pos)| (gap, pos, idx))
}

fn rows_top_to_bottom_then_left_to_right(regions: &mut Vec<Region>) {
    regions.sort_by(|a, b| {
        a.bbox[1]
            .partial_cmp(&b.bbox[1])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut rows: Vec<(f32, f32, Vec<Region>)> = Vec::new();
    for r in regions.drain(..) {
        let joined = rows.last_mut().is_some_and(|(y0, y1, _)| {
            r.bbox[3].min(*y1) - r.bbox[1].max(*y0)
                >= r.height() * ROW_MEMBERSHIP_MIN_OVERLAP_FRAC_OF_REGION_HEIGHT
        });
        match (joined, rows.last_mut()) {
            (true, Some((y0, y1, members))) => {
                *y0 = y0.min(r.bbox[1]);
                *y1 = y1.max(r.bbox[3]);
                members.push(r);
            }
            _ => rows.push((r.bbox[1], r.bbox[3], vec![r])),
        }
    }
    for (_, _, mut members) in rows {
        members.sort_by(|a, b| {
            a.bbox[0]
                .partial_cmp(&b.bbox[0])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        regions.append(&mut members);
    }
}

fn xy_cut_reading_order(mut regions: Vec<Region>) -> Vec<Region> {
    if regions.len() <= 1 {
        return regions;
    }
    let y_split = largest_gap_split(&regions, 1);
    let x_split = largest_gap_split(&regions, 0);
    let chosen = match (y_split, x_split) {
        (Some(y), Some(x)) => Some(if y.0 >= x.0 { y } else { x }),
        (y, x) => y.or(x),
    };
    let Some((_, pos, idx)) = chosen else {
        rows_top_to_bottom_then_left_to_right(&mut regions);
        return regions;
    };
    let take: Vec<Region> = idx.iter().map(|&i| regions[i].clone()).collect();
    let (first, second) = take.split_at(pos);
    let mut out = xy_cut_reading_order(first.to_vec());
    out.extend(xy_cut_reading_order(second.to_vec()));
    out
}

pub fn plan_regions(
    page: &LayoutPage,
    img_w: usize,
    img_h: usize,
    include_pictures: bool,
) -> Vec<Region> {
    let (fw, fh) = (img_w as f32, img_h as f32);
    let mut regions: Vec<Region> = page
        .elements
        .iter()
        .filter_map(|e| {
            let b = e.bbox?;
            if e.is_picture() && !include_pictures {
                return None;
            }
            let b = [b[0].max(0.0), b[1].max(0.0), b[2].min(fw), b[3].min(fh)];
            usable(&b).then(|| Region {
                bbox: b,
                category: e.category.clone(),
            })
        })
        .collect();
    merge_substantial_overlaps(&mut regions);
    let mut regions = xy_cut_reading_order(regions);
    expand_without_annexing_neighbors(&mut regions, fw, fh);
    regions.retain(|r| usable(&r.bbox));
    regions
}

fn expand_without_annexing_neighbors(regions: &mut [Region], fw: f32, fh: f32) {
    let raws: Vec<[f32; 4]> = regions.iter().map(|r| r.bbox).collect();
    for (i, region) in regions.iter_mut().enumerate() {
        let raw = raws[i];
        let mut e = expand_and_clamp(raw, fw, fh);
        for (j, other) in raws.iter().enumerate() {
            if j == i || intersection_area(&raw, other) > 0.0 || intersection_area(&e, other) == 0.0
            {
                continue;
            }
            if other[3] <= raw[1] {
                e[1] = e[1].max(other[3]);
            }
            if other[1] >= raw[3] {
                e[3] = e[3].min(other[1]);
            }
            if other[2] <= raw[0] {
                e[0] = e[0].max(other[2]);
            }
            if other[0] >= raw[2] {
                e[2] = e[2].min(other[0]);
            }
        }
        region.bbox = e;
    }
}

pub fn coverage_frac(regions: &[Region], img_w: usize, img_h: usize) -> f32 {
    if img_w == 0 || img_h == 0 {
        return 0.0;
    }
    regions.iter().map(|r| area(&r.bbox)).sum::<f32>() / (img_w as f32 * img_h as f32)
}

pub fn layout_is_usable(regions: &[Region], img_w: usize, img_h: usize) -> bool {
    !regions.is_empty()
        && coverage_frac(regions, img_w, img_h)
            >= LAYOUT_USABLE_MIN_COVERAGE_FRAC_BELOW_THIS_THE_DETECTOR_MISSED_THE_PAGE
}

pub fn category_prefix(category: Option<&str>, text: &str) -> &'static str {
    if text.trim_start().starts_with('#') {
        return "";
    }
    match category {
        Some("Title") => "# ",
        Some("Section-header") => "## ",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dots_ocr::parse::parse_layout_json;

    fn page(raw: &str) -> LayoutPage {
        parse_layout_json(raw)
    }

    #[test]
    fn expansion_adds_descender_margin_clamps_to_the_image_and_stops_at_the_title_above() {
        let p = page(r#"[{"bbox":[0,0,100,100],"category":"Text"}]"#);
        let regs = plan_regions(&p, 102, 200, false);
        assert_eq!(regs.len(), 1);
        let b = regs[0].bbox;
        assert_eq!(b[0], 0.0);
        assert_eq!(b[1], 0.0);
        assert_eq!(b[2], 102.0);
        assert!(
            (b[3] - 108.0).abs() < 1e-3,
            "bottom margin 8% of height: {b:?}"
        );
        let p = page(
            r#"[{"bbox":[0,0,300,15],"category":"Title"},
                {"bbox":[0,16,600,700],"category":"Text"}]"#,
        );
        let regs = plan_regions(&p, 1000, 1000, false);
        assert_eq!(regs.len(), 2);
        assert_eq!(regs[0].category.as_deref(), Some("Title"));
        assert!(
            (regs[1].bbox[1] - 15.0).abs() < 1e-3,
            "body top margin must clip at the title bottom: {regs:?}"
        );
        assert!(
            regs[1].bbox[3] > 700.0,
            "the unopposed bottom margin survives: {regs:?}"
        );
    }

    #[test]
    fn touching_bands_stay_separate_overlaps_merge_and_mixed_category_merges_lose_the_category() {
        let p = page(
            r#"[{"bbox":[10,10,100,50],"category":"Title"},
                {"bbox":[12,12,102,52],"category":"Text"}]"#,
        );
        let regs = plan_regions(&p, 1000, 1000, false);
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].category, None);
        let p = page(
            r#"[{"bbox":[10,10,100,50],"category":"Text"},
                {"bbox":[10,48,100,90],"category":"Text"},
                {"bbox":[10,300,100,340],"category":"Text"},
                {"bbox":[15,305,105,345],"category":"Text"}]"#,
        );
        let regs = plan_regions(&p, 1000, 1000, false);
        assert_eq!(regs.len(), 3, "{regs:?}");
        let merged = regs.iter().find(|r| r.bbox[3] > 300.0).unwrap();
        assert!(
            merged.bbox[2] >= 105.0 && merged.bbox[1] <= 300.0,
            "{merged:?}"
        );
    }

    #[test]
    fn reading_order_is_row_major_without_a_gap_column_major_with_one_and_headers_lead() {
        let p = page(
            r#"[{"bbox":[600,100,900,200],"category":"Text"},
                {"bbox":[100,100,500,200],"category":"Text"},
                {"bbox":[600,250,900,350],"category":"Text"},
                {"bbox":[100,250,500,350],"category":"Text"}]"#,
        );
        let regs = plan_regions(&p, 1000, 1000, false);
        assert_eq!(regs.len(), 4);
        let xs: Vec<f32> = regs.iter().map(|r| r.bbox[0]).collect();
        let ys: Vec<f32> = regs.iter().map(|r| r.bbox[1]).collect();
        assert!(xs[0] < 200.0 && xs[1] < 200.0, "left column first: {xs:?}");
        assert!(
            xs[2] > 500.0 && xs[3] > 500.0,
            "right column second: {xs:?}"
        );
        assert!(
            ys[0] < ys[1] && ys[2] < ys[3],
            "top to bottom within columns: {ys:?}"
        );
        let p = page(
            r#"[{"bbox":[600,300,900,800],"category":"Text"},
                {"bbox":[100,300,500,800],"category":"Text"},
                {"bbox":[100,50,900,200],"category":"Section-header"}]"#,
        );
        let regs = plan_regions(&p, 1000, 1000, false);
        assert_eq!(regs.len(), 3);
        assert_eq!(regs[0].category.as_deref(), Some("Section-header"));
        assert!(
            regs[1].bbox[0] < 200.0 && regs[2].bbox[0] > 500.0,
            "{regs:?}"
        );
        let p = page(
            r#"[{"bbox":[290,20,600,55],"category":"Text","text":"row1-right"},
                {"bbox":[10,10,100,45],"category":"Text","text":"row1-left"},
                {"bbox":[90,50,300,100],"category":"Text","text":"row2"}]"#,
        );
        let regs = plan_regions(&p, 1000, 1000, false);
        assert_eq!(regs.len(), 3);
        assert!(regs[0].bbox[0] < 100.0, "row 1 left first: {regs:?}");
        assert!(regs[1].bbox[0] > 250.0, "row 1 right second: {regs:?}");
        assert!(regs[2].bbox[1] > 40.0, "row 2 last: {regs:?}");
    }

    #[test]
    fn degenerate_boxes_missing_boxes_and_unrequested_pictures_yield_no_regions() {
        let p = page(
            r#"[{"bbox":[0,0,2,2],"category":"Text"},
                {"bbox":[0,100,4000,102],"category":"Text"}]"#,
        );
        assert!(plan_regions(&p, 5000, 5000, false).is_empty());
        let p = page(r#"[{"category":"Text","text":"no box"}]"#);
        assert!(plan_regions(&p, 1000, 1000, false).is_empty());
        let raw = r#"[{"bbox":[10,10,100,100],"category":"Picture"},
                      {"bbox":[10,300,100,400],"category":"Text"}]"#;
        let p = page(raw);
        assert_eq!(plan_regions(&p, 1000, 1000, false).len(), 1);
        assert_eq!(plan_regions(&p, 1000, 1000, true).len(), 2);
    }

    #[test]
    fn a_single_tiny_region_is_an_unusable_layout_but_half_page_coverage_is_usable() {
        let tiny = vec![Region {
            bbox: [900.0, 557.0, 2946.0, 1105.0],
            category: Some("Text".to_string()),
        }];
        assert!(coverage_frac(&tiny, 5642, 3761) < 0.06);
        assert!(!layout_is_usable(&tiny, 5642, 3761));
        let half = vec![Region {
            bbox: [0.0, 0.0, 884.0, 1123.0],
            category: Some("Text".to_string()),
        }];
        assert!(layout_is_usable(&half, 1400, 1400));
        assert!(!layout_is_usable(&[], 1400, 1400));
    }

    #[test]
    fn category_prefixes_pixel_bounds_and_env_pixels_helpers_behave() {
        assert_eq!(env_pixels("NV_DOTS_TEST_UNSET_PIXELS_XYZ", 42), 42);
        let r = Region {
            bbox: [10.4, -3.0, 99.2, 1005.0],
            category: None,
        };
        assert_eq!(r.pixel_bounds(100, 1000), (10, 0, 100, 1000));
        assert_eq!(category_prefix(Some("Title"), "Report"), "# ");
        assert_eq!(category_prefix(Some("Section-header"), "Intro"), "## ");
        assert_eq!(category_prefix(Some("Section-header"), "## Intro"), "");
        assert_eq!(category_prefix(Some("Text"), "body"), "");
        assert_eq!(category_prefix(None, "body"), "");
    }
}
