use anyhow::Result;

use super::parse::{LayoutElement, LayoutPage};
use super::preprocess::{sample_ink, smart_resize, PixelBudget};
use crate::deepseek_ocr::preprocess::RgbImage;

pub const TIERED_ENV: &str = "NV_DOTS_TIERED";
pub const TIERED_MIN_DOWNSCALE_ENV: &str = "NV_DOTS_TIERED_MIN_DOWNSCALE";
pub const TIERED_MIN_INPUT_PX_ENV: &str = "NV_DOTS_TIERED_MIN_INPUT_PX";

pub const DEFAULT_TIERED_MIN_DOWNSCALE_0_70_KEEPS_EVERY_INPUT_THE_16_FIXTURE_CORPUS_COVERS_ON_SINGLE_PASS: f64 =
    0.70;

pub const DEFAULT_TIERED_MIN_INPUT_PX_3_2MP_SITS_ABOVE_THE_2_7MP_CORPUS_CEILING_AND_UNDER_THE_MEASURED_WALL_CROSSOVER: usize =
    3_200_000;

pub const AUTO_ENGAGES_ON_PIXELS_BECAUSE_SINGLE_PASS_VISION_COST_IS_QUADRATIC_IN_PATCHES: &str =
    "the MP ladder on one dense page measured single-pass walls of 9.9/16.9/33.1/56.5 s at \
     3/5/8/11.2MP against tiered walls of 11.4/12.1/12.6/14.3 s, with tiered neutral CER at \
     or under single-pass at every rung, and the screens matrix repeats it (4K: 6.6 s tiered \
     vs 31.0 s native single-pass at higher word recall). The wall crossover is between 3 and \
     4MP; 3.2MP keeps every input the 16-fixture corpus covers (max 2.66MP) on today's \
     single-pass byte-identically, engages 1440p screens and up, and costs at most the \
     measured +1.5 s on a 3MP-class page relative to the crossover ideal while gaining \
     accuracy there. The downscale floor stays as a second trigger for callers running a \
     reduced NV_DOTS_MAX_PIXELS budget, where quality loss precedes the wall argument.";

pub const AUTO_IS_THE_ARBITRATED_SERVING_DEFAULT_AND_ZERO_RESTORES_THE_PRE_FLIP_PATH_EXACTLY: &str =
    "NV_DOTS_TIERED unset means auto, by arbitration on the 090-093 scale corpus: tiered \
     wins wall AND neutral CER on every measured input above the threshold (15-100MP pages \
     17.0-27.0 s vs 56.5-60.6 s single-pass; 4K screens 6.6 s at word recall 0.92 vs 31.0 s \
     at 0.67), and the 16-fixture corpus sits under the pixel threshold so its twelve zeros \
     cannot move -- re-stamped after the flip. NV_DOTS_TIERED=0 is the escape that restores \
     the always-single-pass path byte-identically; =1 forces tiling on everything.";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TierPolicy {
    Off,
    #[default]
    Auto,
    Always,
}

impl TierPolicy {
    pub fn name(self) -> &'static str {
        match self {
            TierPolicy::Off => "0",
            TierPolicy::Auto => "auto",
            TierPolicy::Always => "1",
        }
    }

    pub fn parse(raw: Option<&str>, default: Self) -> Result<Self> {
        match raw {
            None | Some("") => Ok(default),
            Some("0") | Some("off") => Ok(TierPolicy::Off),
            Some("auto") => Ok(TierPolicy::Auto),
            Some("1") | Some("always") => Ok(TierPolicy::Always),
            Some(other) => anyhow::bail!(
                "{TIERED_ENV}={other:?} is not a dots.ocr tier policy; use auto (tiered when \
                 the single-pass downscale falls below the floor), 0/off (single pass \
                 always), or 1/always (tiered always)"
            ),
        }
    }

    pub fn from_env() -> Result<Self> {
        let _ = AUTO_IS_THE_ARBITRATED_SERVING_DEFAULT_AND_ZERO_RESTORES_THE_PRE_FLIP_PATH_EXACTLY;
        Self::parse(std::env::var(TIERED_ENV).ok().as_deref(), Self::default())
    }
}

pub fn min_downscale_from_env() -> f64 {
    std::env::var(TIERED_MIN_DOWNSCALE_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| (0.0..=1.0).contains(v))
        .unwrap_or(DEFAULT_TIERED_MIN_DOWNSCALE_0_70_KEEPS_EVERY_INPUT_THE_16_FIXTURE_CORPUS_COVERS_ON_SINGLE_PASS)
}

pub fn min_input_px_from_env() -> usize {
    std::env::var(TIERED_MIN_INPUT_PX_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_TIERED_MIN_INPUT_PX_3_2MP_SITS_ABOVE_THE_2_7MP_CORPUS_CEILING_AND_UNDER_THE_MEASURED_WALL_CROSSOVER)
}

pub fn single_pass_downscale(w: usize, h: usize, budget: PixelBudget) -> Result<f64> {
    let (rh, rw) = smart_resize(h, w, budget)?;
    Ok(((rh * rw) as f64 / (h * w) as f64).sqrt().min(1.0))
}

pub fn engages(w: usize, h: usize, budget: PixelBudget, policy: TierPolicy) -> Result<bool> {
    match policy {
        TierPolicy::Off => Ok(false),
        TierPolicy::Always => Ok(true),
        TierPolicy::Auto => {
            let _ = AUTO_ENGAGES_ON_PIXELS_BECAUSE_SINGLE_PASS_VISION_COST_IS_QUADRATIC_IN_PATCHES;
            Ok(w * h > min_input_px_from_env()
                || single_pass_downscale(w, h, budget)? < min_downscale_from_env())
        }
    }
}

pub const CLASSICAL_SCOUT_EXISTS_BECAUSE_MODEL_T0_COST_IS_DECODE_LENGTH_NOT_RESOLUTION: &str =
    "the layout-only model pass generates ~25-30 JSON tokens per region at ~5-8 ms a step, \
     so a busy page pays seconds of T0 decode that no input downscale removes (measured \
     4.7-10.9 s on 14-25-region pages). Plain-text tiling needs only boxes and reading \
     order, which connected components over dark-luma ink cells produce in milliseconds on \
     the CPU; semantic categories (Table, Formula, Picture) exist only on the model scout, \
     which stays the path for layout-mode output. The scout reads dark-on-light ink: when \
     more than a quarter of samples are dark the page is inverted or photographic and the \
     caller must fall back to the model scout.";

pub const SAME_ROW_BOXES_WITHIN_TWO_CELLS_MERGE_OR_A_BOX_EDGE_SPLITS_A_LINE_AND_CLIPS_GLYPHS: &str =
    "on the gazette the raw components split a wrapped line into two boxes 130 times for \
     121 entries, and one split edge clipped a digit out of a price; two boxes sharing half \
     their vertical extent within two cells of each other are one line of text.";

pub const CLASSICAL_GRID_TARGET_512_CELLS_KEEPS_CC_UNDER_A_MILLISECOND: usize = 512;
pub const CLASSICAL_INKY_CELL_MIN_SAMPLES_2_REJECTS_LONE_SPECKLES: usize = 2;
pub const CLASSICAL_MIN_BLOCK_CELLS_4_DROPS_DUST: usize = 4;
pub const CLASSICAL_DARK_PAGE_INK_FRACTION_CEILING_0_25: f64 = 0.25;

pub const CLASSICAL_MIN_MEDIAN_BOX_HEIGHT_64PX_SMALL_GLYPH_PAGES_SPLIT_AT_THEIR_OWN_WHITESPACE: usize =
    64;

pub const CLASSICAL_BUSY_PAGE_MIN_BOXES_96_SITS_BETWEEN_THE_5K_SCREENS_52_FRAGMENTS_AND_THE_GAZETTES_130_ENTRIES: usize =
    96;

pub const SMALL_GLYPH_PAGES_GO_TO_THE_MODEL_SCOUT_BECAUSE_CC_FRAGMENTS_UI_TEXT: &str =
    "on a 1080p screen the connected components split code and menu lines at their wide \
     internal whitespace (indentation, tab stops, menu spacing exceed two cells) and the \
     line fragments OCR as garbage -- word recall 0.556 against the model scout's 0.972 on \
     the same screen. Pages whose median component height sits under 64 px are small-glyph \
     pages of exactly that shape, so the scout hands them to the model scout, which \
     measured 0.86-0.97 recall across the screens matrix. The guard exempts pages with 96 \
     or more components: the 121-entry gazette produces 130 boxes at a 48 px median and \
     reads 0.967 -- the busy-print class the classical scout exists for -- while the \
     worst screen measured is the 5K editor at 52 fragment boxes, which the first 48-box \
     carve-out wrongly admitted and this threshold returns to the model scout.";

pub const A_DARK_THEME_IS_A_LIGHT_PAGE_INVERTED_SO_BRIGHT_POINTS_BECOME_THE_INK: &str =
    "when dark-luma points exceed the ceiling the page is a dark theme or a photograph; \
     re-sampling with luma >= 192 as ink reads light-on-dark UI text (chrome and panel \
     greys sit at luma 25-66, the text at ~222), and only when the bright fraction also \
     exceeds the ceiling does the scout refuse and leave the page to the model.";

enum ScoutRead {
    Unreadable,
    Boxes(Vec<([usize; 4], usize)>, usize),
}

fn scout_components(img: &RgbImage) -> ScoutRead {
    let mut ink = sample_ink(img);
    if ink.sampled == 0 {
        return ScoutRead::Unreadable;
    }
    if ink.xs.len() as f64 / ink.sampled as f64 > CLASSICAL_DARK_PAGE_INK_FRACTION_CEILING_0_25 {
        let _ = A_DARK_THEME_IS_A_LIGHT_PAGE_INVERTED_SO_BRIGHT_POINTS_BECOME_THE_INK;
        ink = super::preprocess::sample_bright_ink(img);
        if ink.xs.len() as f64 / ink.sampled as f64
            > CLASSICAL_DARK_PAGE_INK_FRACTION_CEILING_0_25
        {
            return ScoutRead::Unreadable;
        }
    }
    let cell = (img.w.max(img.h)).div_ceil(CLASSICAL_GRID_TARGET_512_CELLS_KEEPS_CC_UNDER_A_MILLISECOND).max(8);
    let gw = img.w.div_ceil(cell);
    let gh = img.h.div_ceil(cell);
    let mut counts = vec![0u16; gw * gh];
    for (x, y) in ink.xs.iter().zip(&ink.ys) {
        let cx = (*x as usize / cell).min(gw - 1);
        let cy = (*y as usize / cell).min(gh - 1);
        counts[cy * gw + cx] = counts[cy * gw + cx].saturating_add(1);
    }
    let inky: Vec<bool> = counts
        .iter()
        .map(|&c| c as usize >= CLASSICAL_INKY_CELL_MIN_SAMPLES_2_REJECTS_LONE_SPECKLES)
        .collect();
    let mut dilated = vec![false; gw * gh];
    for y in 0..gh {
        for x in 0..gw {
            if inky[y * gw + x] {
                for dy in y.saturating_sub(1)..(y + 2).min(gh) {
                    for dx in x.saturating_sub(1)..(x + 2).min(gw) {
                        dilated[dy * gw + dx] = true;
                    }
                }
            }
        }
    }
    let mut label = vec![0u32; gw * gh];
    let mut next = 0u32;
    let mut boxes: Vec<([usize; 4], usize)> = Vec::new();
    let mut stack = Vec::new();
    for start in 0..gw * gh {
        if !dilated[start] || label[start] != 0 {
            continue;
        }
        next += 1;
        let mut bb = [gw, gh, 0usize, 0usize];
        let mut cells = 0usize;
        stack.push(start);
        label[start] = next;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % gw, i / gw);
            if inky[i] {
                cells += 1;
                bb[0] = bb[0].min(x);
                bb[1] = bb[1].min(y);
                bb[2] = bb[2].max(x);
                bb[3] = bb[3].max(y);
            }
            for dy in y.saturating_sub(1)..(y + 2).min(gh) {
                for dx in x.saturating_sub(1)..(x + 2).min(gw) {
                    let j = dy * gw + dx;
                    if dilated[j] && label[j] == 0 {
                        label[j] = next;
                        stack.push(j);
                    }
                }
            }
        }
        if cells >= CLASSICAL_MIN_BLOCK_CELLS_4_DROPS_DUST && bb[0] <= bb[2] {
            boxes.push((bb, cells));
        }
    }
    let _ = SAME_ROW_BOXES_WITHIN_TWO_CELLS_MERGE_OR_A_BOX_EDGE_SPLITS_A_LINE_AND_CLIPS_GLYPHS;
    let mut boxes = boxes;
    let mut merged_any = true;
    while merged_any {
        merged_any = false;
        'outer: for i in 0..boxes.len() {
            for j in (i + 1)..boxes.len() {
                let (a, b) = (boxes[i].0, boxes[j].0);
                let v_overlap =
                    (a[3].min(b[3]) + 1).saturating_sub(a[1].max(b[1]));
                let min_h = (a[3] - a[1]).min(b[3] - b[1]) + 1;
                let h_gap = if a[2] < b[0] {
                    b[0] - a[2]
                } else if b[2] < a[0] {
                    a[0] - b[2]
                } else {
                    0
                };
                if 2 * v_overlap >= min_h && h_gap <= 2 {
                    let cells = boxes[i].1 + boxes[j].1;
                    let (b2, _) = boxes.remove(j);
                    boxes[i].0 = [
                        a[0].min(b2[0]),
                        a[1].min(b2[1]),
                        a[2].max(b2[2]),
                        a[3].max(b2[3]),
                    ];
                    boxes[i].1 = cells;
                    merged_any = true;
                    break 'outer;
                }
            }
        }
    }
    ScoutRead::Boxes(boxes, cell)
}

fn median_box_height_px(boxes: &[([usize; 4], usize)], cell: usize) -> usize {
    let mut heights: Vec<usize> = boxes
        .iter()
        .map(|(bb, _)| (bb[3] - bb[1] + 1) * cell)
        .collect();
    heights.sort_unstable();
    heights[heights.len() / 2]
}

pub fn classical_scout(img: &RgbImage) -> Option<LayoutPage> {
    let _ = CLASSICAL_SCOUT_EXISTS_BECAUSE_MODEL_T0_COST_IS_DECODE_LENGTH_NOT_RESOLUTION;
    let ScoutRead::Boxes(boxes, cell) = scout_components(img) else {
        return None;
    };
    if !boxes.is_empty()
        && boxes.len() < CLASSICAL_BUSY_PAGE_MIN_BOXES_96_SITS_BETWEEN_THE_5K_SCREENS_52_FRAGMENTS_AND_THE_GAZETTES_130_ENTRIES
    {
        let _ = SMALL_GLYPH_PAGES_GO_TO_THE_MODEL_SCOUT_BECAUSE_CC_FRAGMENTS_UI_TEXT;
        if median_box_height_px(&boxes, cell)
            < CLASSICAL_MIN_MEDIAN_BOX_HEIGHT_64PX_SMALL_GLYPH_PAGES_SPLIT_AT_THEIR_OWN_WHITESPACE
        {
            return None;
        }
    }
    let elements = boxes
        .into_iter()
        .map(|(bb, _)| LayoutElement {
            bbox: Some([
                (bb[0] * cell) as f32,
                (bb[1] * cell) as f32,
                ((bb[2] + 1) * cell).min(img.w) as f32,
                ((bb[3] + 1) * cell).min(img.h) as f32,
            ]),
            category: Some("Text".to_string()),
            text: None,
        })
        .collect();
    Some(LayoutPage {
        elements,
        truncated: false,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubThresholdClass {
    Document,
    ScreenLike,
    Undecided,
}

pub const SCREENLIKE_MEDIAN_COMPONENT_UNDER_0_04_OF_PAGE_HEIGHT_SEPARATES_UI_FROM_RENDERED_LINES: f64 =
    0.04;
pub const SCREENLIKE_MIN_COMPONENTS_4_A_STRIP_OR_A_BANNER_IS_NOT_A_DESKTOP: usize = 4;

pub const SUB_THRESHOLD_SCREENS_DETOUR_TO_TILING_BECAUSE_NATIVE_READING_DROPS_A_THIRD_OF_THE_UI: &str =
    "below the tier threshold a 1080p screenshot reads word recall 0.75 native single-pass, \
     0.92 downscaled and 0.97 tiled (ncer 0.06 tiled against 0.34 downscaled), so \
     screen-like sub-threshold pages detour to the tiling composition instead. Screen-like \
     means the ink components read as UI: at least four components and a median component \
     height under 4 percent of the page height -- the 070 strips put a text line across a \
     third of their height and the 071 renders read 5-20 percent, while UI glyphs sit at \
     1-3 percent; the corpus guard suite asserts every conformance fixture classifies away \
     from the detour, because the twelve zeros are not available as collateral. \
     NV_DOTS_SCREEN_DETOUR=0 pins sub-threshold inputs to the plain single pass.";

pub fn classify_sub_threshold(img: &RgbImage) -> SubThresholdClass {
    let ScoutRead::Boxes(boxes, cell) = scout_components(img) else {
        return SubThresholdClass::Undecided;
    };
    if boxes.len() < SCREENLIKE_MIN_COMPONENTS_4_A_STRIP_OR_A_BANNER_IS_NOT_A_DESKTOP {
        return SubThresholdClass::Undecided;
    }
    let frac = median_box_height_px(&boxes, cell) as f64 / img.h as f64;
    if frac < SCREENLIKE_MEDIAN_COMPONENT_UNDER_0_04_OF_PAGE_HEIGHT_SEPARATES_UI_FROM_RENDERED_LINES
    {
        SubThresholdClass::ScreenLike
    } else {
        SubThresholdClass::Document
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_defaults_to_auto_zero_escapes_to_single_pass_and_garbage_is_rejected() {
        for raw in [None, Some("")] {
            assert_eq!(
                TierPolicy::parse(raw, TierPolicy::default()).unwrap(),
                TierPolicy::Auto,
                "the arbitrated serving default is auto"
            );
        }
        for raw in [Some("0"), Some("off")] {
            assert_eq!(
                TierPolicy::parse(raw, TierPolicy::Auto).unwrap(),
                TierPolicy::Off,
                "{raw:?} must restore the pre-flip single-pass path"
            );
        }
        assert_eq!(TierPolicy::parse(Some("auto"), TierPolicy::Off).unwrap(), TierPolicy::Auto);
        for raw in [Some("1"), Some("always")] {
            assert_eq!(TierPolicy::parse(raw, TierPolicy::Off).unwrap(), TierPolicy::Always);
        }
        let err = TierPolicy::parse(Some("yes"), TierPolicy::Off).unwrap_err();
        assert!(err.to_string().contains(TIERED_ENV), "{err}");
    }

    #[test]
    fn the_16_fixture_corpus_domain_stays_single_pass_under_auto_by_construction() {
        let budget = PixelBudget::default();
        for (w, h) in [(1400, 1900), (1920, 1080), (1500, 1000)] {
            assert!(
                w * h
                    <= DEFAULT_TIERED_MIN_INPUT_PX_3_2MP_SITS_ABOVE_THE_2_7MP_CORPUS_CEILING_AND_UNDER_THE_MEASURED_WALL_CROSSOVER,
                "{w}x{h} must sit under the auto threshold"
            );
            assert!(!engages(w, h, budget, TierPolicy::Auto).unwrap());
            assert!(!engages(w, h, budget, TierPolicy::Off).unwrap());
        }
    }

    #[test]
    fn screens_from_1440p_up_and_every_scale_fixture_engage_tiering_under_auto() {
        let budget = PixelBudget::default();
        for (w, h) in [
            (2560, 1440),
            (3840, 2160),
            (5120, 1440),
            (3248, 4592),
            (4956, 7016),
            (5936, 8408),
            (8426, 5960),
            (8400, 11888),
        ] {
            assert!(engages(w, h, budget, TierPolicy::Auto).unwrap(), "{w}x{h}");
            assert!(!engages(w, h, budget, TierPolicy::Off).unwrap());
            assert!(engages(w, h, budget, TierPolicy::Always).unwrap());
        }
    }

    #[test]
    fn a_reduced_pixel_budget_trips_the_downscale_floor_even_under_the_px_threshold() {
        let budget = PixelBudget {
            min_pixels: 3136,
            max_pixels: 1_000_000,
        };
        assert!(engages(1920, 1080, budget, TierPolicy::Auto).unwrap());
    }

    #[test]
    fn downscale_saturates_at_one_for_small_inputs() {
        let d = single_pass_downscale(400, 300, PixelBudget::default()).unwrap();
        assert_eq!(d, 1.0);
    }

    fn lined_block(x: usize, y: usize, b: &[usize; 4], pitch: usize) -> bool {
        x >= b[0] && x < b[2] && y >= b[1] && y < b[3] && (y - b[1]) % pitch < pitch / 2
    }

    #[test]
    fn classical_scout_boxes_two_separated_text_blocks() {
        let a = [50usize, 60, 350, 160];
        let b = [450usize, 300, 750, 500];
        let img = RgbImage::from_fn(800, 600, |x, y| {
            if lined_block(x, y, &a, 20) || lined_block(x, y, &b, 20) {
                [0, 0, 0]
            } else {
                [255, 255, 255]
            }
        });
        let page = classical_scout(&img).expect("light page must scout");
        assert_eq!(page.elements.len(), 2, "{:?}", page.elements);
        for (want, got) in [a, b].iter().zip(&page.elements) {
            let g = got.bbox.unwrap();
            assert!(
                g[0] <= want[0] as f32
                    && g[1] <= want[1] as f32
                    && g[2] >= want[2] as f32 - 8.0
                    && g[3] >= want[3] as f32 - 8.0,
                "box {g:?} must cover {want:?}"
            );
        }
    }

    #[test]
    fn dark_theme_ui_scouts_by_inverted_ink_and_a_bimodal_photo_page_refuses() {
        let darkui = RgbImage::from_fn(800, 600, |x, y| {
            if lined_block(x, y, &[100, 100, 700, 300], 20) {
                [222, 222, 222]
            } else {
                [24, 25, 28]
            }
        });
        let page = classical_scout(&darkui).expect("dark theme must scout by bright ink");
        assert_eq!(page.elements.len(), 1, "{:?}", page.elements);
        let b = page.elements[0].bbox.unwrap();
        assert!(b[0] <= 100.0 && b[2] >= 692.0 && b[3] >= 292.0, "{b:?}");

        let photo = RgbImage::from_fn(400, 400, |x, _| {
            if (x / 4) % 2 == 0 {
                [10, 10, 10]
            } else {
                [250, 250, 250]
            }
        });
        assert!(
            classical_scout(&photo).is_none(),
            "a page that is mostly dark AND mostly bright is a photograph, not a theme"
        );

        let solid = RgbImage::filled(400, 400, [20, 20, 20]);
        let page = classical_scout(&solid).expect("a solid dark page has no bright ink");
        assert!(page.elements.is_empty(), "no text, no boxes: {:?}", page.elements);
    }

    #[test]
    fn a_page_of_thin_ui_strips_defers_to_the_model_scout() {
        let img = RgbImage::from_fn(800, 600, |_, y| {
            if (100..116).contains(&y) || (200..216).contains(&y) || (300..316).contains(&y) {
                [0, 0, 0]
            } else {
                [255, 255, 255]
            }
        });
        assert!(
            classical_scout(&img).is_none(),
            "16px strips are small-glyph fragments and must defer"
        );
    }

    #[test]
    fn a_busy_page_of_small_entries_keeps_the_classical_scout() {
        let img = RgbImage::from_fn(1600, 3400, |x, y| {
            let col = x / 400;
            let in_col = x % 400 < 300 && x % 400 > 20;
            let row = y % 100;
            if in_col && col < 4 && y > 40 && y < 3360 && (20..52).contains(&row) {
                [0, 0, 0]
            } else {
                [255, 255, 255]
            }
        });
        let page = classical_scout(&img).expect("128 small entries are busy print, not a screen");
        assert!(
            page.elements.len() >= CLASSICAL_BUSY_PAGE_MIN_BOXES_96_SITS_BETWEEN_THE_5K_SCREENS_52_FRAGMENTS_AND_THE_GAZETTES_130_ENTRIES,
            "{} boxes",
            page.elements.len()
        );
    }

    #[test]
    fn classical_scout_refuses_nothing_light_and_grey_linework_is_invisible() {
        let grey = RgbImage::from_fn(800, 600, |x, y| {
            if lined_block(x, y, &[100, 100, 700, 500], 24) {
                [176, 176, 176]
            } else {
                [255, 255, 255]
            }
        });
        let page = classical_scout(&grey).expect("grey linework page is light");
        assert!(
            page.elements.is_empty(),
            "light-grey linework must not read as text ink: {:?}",
            page.elements
        );
    }
}
