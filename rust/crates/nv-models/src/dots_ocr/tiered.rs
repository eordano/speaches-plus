use anyhow::Result;

use super::parse::{LayoutElement, LayoutPage};
use super::preprocess::{sample_ink, smart_resize, PixelBudget};
use crate::deepseek_ocr::preprocess::RgbImage;

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

pub const AUTO_IS_THE_ARBITRATED_SERVING_DEFAULT: &str =
    "auto is the serving default by arbitration on the 090-093 scale corpus: tiered wins \
     wall AND neutral CER on every measured input above the threshold (15-100MP pages \
     17.0-27.0 s vs 56.5-60.6 s single-pass; 4K screens 6.6 s at word recall 0.92 vs 31.0 s \
     at 0.67), and the 16-fixture corpus sits under the pixel threshold so its twelve zeros \
     cannot move. Off and Always are reachable only through the per-request tiering hint.";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TierPolicy {
    Off,
    #[default]
    Auto,
    Always,
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
            let _ = AUTO_IS_THE_ARBITRATED_SERVING_DEFAULT;
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

pub const INK_BRIDGE_MAX_GAP_8_CELLS_A_GUTTER_IS_WHITE_AT_EVERY_SAMPLE: usize = 8;

pub const SUB_THRESHOLD_INK_IN_A_SAME_ROW_GAP_IS_A_SPLIT_LINE_NOT_A_GUTTER: &str =
    "stride-4 sampling leaves the cells beside a word space under the two-sample inky \
     threshold, so a wrapped entry can split into a two-line left component and a one-line \
     tail whose crop then starts mid-word and stitches as a displaced paragraph after its \
     own entry ('TO LET: cast iron peat stove with / Box 103.' then 'ue kit. 29 pounds or \
     near offer.' -- nine of the gazette's 121 entries carried the artifact, the entire \
     seam class of its ncer). The gap of such a split still holds sampled ink, the clipped \
     glyph slivers themselves, while a column gutter is white at every sample: on a page \
     already accepted for classical crops, same-row boxes therefore also merge across gaps \
     of up to eight cells when any sample landed in the gap band, which rejoins all nine \
     split entries and zero gutters on the measured scale corpus. The bridge runs strictly \
     AFTER the acceptance guards and never feeds classify_sub_threshold, because bridged \
     boxes move the box-count and median-height statistics those verdicts read (the uw49 \
     screen's median crosses the 64 px line under bridged merges) and no routing decision \
     may move with it.";

pub const CLASSICAL_GRID_TARGET_512_CELLS_KEEPS_CC_UNDER_A_MILLISECOND: usize = 512;
pub const CLASSICAL_INKY_CELL_MIN_SAMPLES_2_REJECTS_LONE_SPECKLES: usize = 2;
pub const CLASSICAL_MIN_BLOCK_CELLS_4_DROPS_DUST: usize = 4;
pub const CLASSICAL_DARK_PAGE_INK_FRACTION_CEILING_0_25: f64 = 0.25;

pub const CLASSICAL_MIN_MEDIAN_BOX_HEIGHT_64PX_SMALL_GLYPH_PAGES_SPLIT_AT_THEIR_OWN_WHITESPACE:
    usize = 64;

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

pub const DARK_THEMED_MEANS_MOSTLY_DARK_WITH_READABLE_BRIGHT_INK_AND_A_PHOTO_IS_NEITHER: &str =
    "the detector reuses the scout's own two ink reads: a page whose dark-luma fraction \
     exceeds the 0.25 ceiling is inverted or photographic, and it is a dark THEME exactly \
     when the bright-ink re-read stays under the same ceiling (light-on-dark text is sparse \
     bright points over dark ground, a photograph is dense in both reads and refuses). This \
     is the same verdict scout_components already reaches internally; exposing it lets the \
     serving router invert the page before the tower, where inversion measured recall \
     0.667 -> 0.944 and 41 -> 7 s on the 4K dark theme. The corpus and every light fixture \
     must answer false -- the twelve zeros are not available as collateral.";

pub fn page_is_dark_themed(img: &RgbImage) -> bool {
    let _ = DARK_THEMED_MEANS_MOSTLY_DARK_WITH_READABLE_BRIGHT_INK_AND_A_PHOTO_IS_NEITHER;
    let ink = sample_ink(img);
    if ink.sampled == 0
        || ink.xs.len() as f64 / ink.sampled as f64 <= CLASSICAL_DARK_PAGE_INK_FRACTION_CEILING_0_25
    {
        return false;
    }
    let bright = super::preprocess::sample_bright_ink(img);
    bright.sampled > 0
        && bright.xs.len() as f64 / bright.sampled as f64
            <= CLASSICAL_DARK_PAGE_INK_FRACTION_CEILING_0_25
}

struct ScoutBoxes {
    boxes: Vec<([usize; 4], usize)>,
    cell: usize,
    counts: Vec<u16>,
    grid_w: usize,
}

enum ScoutRead {
    Unreadable,
    Boxes(ScoutBoxes),
}

fn scout_components(img: &RgbImage) -> ScoutRead {
    let mut ink = sample_ink(img);
    if ink.sampled == 0 {
        return ScoutRead::Unreadable;
    }
    if ink.xs.len() as f64 / ink.sampled as f64 > CLASSICAL_DARK_PAGE_INK_FRACTION_CEILING_0_25 {
        let _ = A_DARK_THEME_IS_A_LIGHT_PAGE_INVERTED_SO_BRIGHT_POINTS_BECOME_THE_INK;
        ink = super::preprocess::sample_bright_ink(img);
        if ink.xs.len() as f64 / ink.sampled as f64 > CLASSICAL_DARK_PAGE_INK_FRACTION_CEILING_0_25
        {
            return ScoutRead::Unreadable;
        }
    }
    let cell = (img.w.max(img.h))
        .div_ceil(CLASSICAL_GRID_TARGET_512_CELLS_KEEPS_CC_UNDER_A_MILLISECOND)
        .max(8);
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
    let mut merged_any = true;
    while merged_any {
        merged_any = false;
        'outer: for i in 0..boxes.len() {
            for j in (i + 1)..boxes.len() {
                let (a, b) = (boxes[i].0, boxes[j].0);
                let v_overlap = (a[3].min(b[3]) + 1).saturating_sub(a[1].max(b[1]));
                let min_h = (a[3] - a[1]).min(b[3] - b[1]) + 1;
                let h_gap = if a[2] < b[0] {
                    b[0] - a[2]
                } else {
                    a[0].saturating_sub(b[2])
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
    ScoutRead::Boxes(ScoutBoxes {
        boxes,
        cell,
        counts,
        grid_w: gw,
    })
}

fn bridge_split_lines(sb: &mut ScoutBoxes) {
    let _ = SUB_THRESHOLD_INK_IN_A_SAME_ROW_GAP_IS_A_SPLIT_LINE_NOT_A_GUTTER;
    let gw = sb.grid_w;
    let boxes = &mut sb.boxes;
    let mut merged_any = true;
    while merged_any {
        merged_any = false;
        'outer: for i in 0..boxes.len() {
            for j in (i + 1)..boxes.len() {
                let (a, b) = (boxes[i].0, boxes[j].0);
                let v_overlap = (a[3].min(b[3]) + 1).saturating_sub(a[1].max(b[1]));
                let min_h = (a[3] - a[1]).min(b[3] - b[1]) + 1;
                if 2 * v_overlap < min_h {
                    continue;
                }
                let h_gap = if a[2] < b[0] {
                    b[0] - a[2]
                } else {
                    a[0].saturating_sub(b[2])
                };
                let gap_inked = h_gap > 2
                    && h_gap <= INK_BRIDGE_MAX_GAP_8_CELLS_A_GUTTER_IS_WHITE_AT_EVERY_SAMPLE
                    && {
                        let (gx0, gx1) = if a[2] < b[0] {
                            (a[2] + 1, b[0])
                        } else {
                            (b[2] + 1, a[0])
                        };
                        (a[1].max(b[1])..=a[3].min(b[3]))
                            .any(|y| sb.counts[y * gw + gx0..y * gw + gx1].iter().any(|&c| c > 0))
                    };
                if h_gap <= 2 || gap_inked {
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
    let ScoutRead::Boxes(mut sb) = scout_components(img) else {
        return None;
    };
    if !sb.boxes.is_empty()
        && sb.boxes.len() < CLASSICAL_BUSY_PAGE_MIN_BOXES_96_SITS_BETWEEN_THE_5K_SCREENS_52_FRAGMENTS_AND_THE_GAZETTES_130_ENTRIES
    {
        let _ = SMALL_GLYPH_PAGES_GO_TO_THE_MODEL_SCOUT_BECAUSE_CC_FRAGMENTS_UI_TEXT;
        if median_box_height_px(&sb.boxes, sb.cell)
            < CLASSICAL_MIN_MEDIAN_BOX_HEIGHT_64PX_SMALL_GLYPH_PAGES_SPLIT_AT_THEIR_OWN_WHITESPACE
        {
            return None;
        }
    }
    bridge_split_lines(&mut sb);
    let ScoutBoxes { boxes, cell, .. } = sb;
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

pub const SCREENLIKE_MEDIAN_COMPONENT_UNDER_0_04_OF_PAGE_HEIGHT_SEPARATES_UI_FROM_RENDERED_LINES:
    f64 = 0.04;
pub const SCREENLIKE_MIN_COMPONENTS_4_A_STRIP_OR_A_BANNER_IS_NOT_A_DESKTOP: usize = 4;

pub const SUB_THRESHOLD_SCREENS_DETOUR_TO_TILING_BECAUSE_NATIVE_READING_DROPS_A_THIRD_OF_THE_UI:
    &str =
    "below the tier threshold a 1080p screenshot reads word recall 0.75 native single-pass, \
     0.92 downscaled and 0.97 tiled (ncer 0.06 tiled against 0.34 downscaled), so \
     screen-like sub-threshold pages detour to the tiling composition instead. Screen-like \
     means the ink components read as UI: at least four components and a median component \
     height under 4 percent of the page height -- the 070 strips put a text line across a \
     third of their height and the 071 renders read 5-20 percent, while UI glyphs sit at \
     1-3 percent; the corpus guard suite asserts every conformance fixture classifies away \
     from the detour, because the twelve zeros are not available as collateral.";

pub const SCREENLIKE_REQUIRES_LANDSCAPE_BECAUSE_SPARSE_PORTRAIT_RENDERS_SHARE_THE_LINE_FRACTION:
    &str = "the 071 invoice render measures the same 0.02 median line fraction as a desktop \
     screenshot -- sparse rendered lines and UI text are indistinguishable by relative \
     component height alone -- but every such render is portrait while desktops are \
     landscape, and the landscape corpus strips clear the fraction test at 0.3 of their \
     height. A portrait phone screenshot therefore misses the detour and stays on the \
     single pass; that miss is the recorded price of never detouring a document render.";

pub fn classify_sub_threshold(img: &RgbImage) -> SubThresholdClass {
    let _ = SCREENLIKE_REQUIRES_LANDSCAPE_BECAUSE_SPARSE_PORTRAIT_RENDERS_SHARE_THE_LINE_FRACTION;
    if img.w <= img.h {
        return SubThresholdClass::Undecided;
    }
    let ScoutRead::Boxes(sb) = scout_components(img) else {
        return SubThresholdClass::Undecided;
    };
    if sb.boxes.len() < SCREENLIKE_MIN_COMPONENTS_4_A_STRIP_OR_A_BANNER_IS_NOT_A_DESKTOP {
        return SubThresholdClass::Undecided;
    }
    let frac = median_box_height_px(&sb.boxes, sb.cell) as f64 / img.h as f64;
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
    fn auto_tiering_engages_above_the_px_threshold_or_a_reduced_budget_and_never_on_the_fixture_corpus(
    ) {
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
        let reduced = PixelBudget {
            min_pixels: 3136,
            max_pixels: 1_000_000,
        };
        assert!(engages(1920, 1080, reduced, TierPolicy::Auto).unwrap());
        assert_eq!(single_pass_downscale(400, 300, budget).unwrap(), 1.0);
    }

    fn lined_block(x: usize, y: usize, b: &[usize; 4], pitch: usize) -> bool {
        x >= b[0] && x < b[2] && y >= b[1] && y < b[3] && (y - b[1]) % pitch < pitch / 2
    }

    #[test]
    fn classical_scout_boxes_text_blocks_light_or_dark_keeps_busy_print_and_refuses_photos() {
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
        assert!(
            page.elements.is_empty(),
            "no text, no boxes: {:?}",
            page.elements
        );
        let busy = RgbImage::from_fn(1600, 3400, |x, y| {
            let col = x / 400;
            let in_col = x % 400 < 300 && x % 400 > 20;
            let row = y % 100;
            if in_col && col < 4 && y > 40 && y < 3360 && (20..52).contains(&row) {
                [0, 0, 0]
            } else {
                [255, 255, 255]
            }
        });
        let page = classical_scout(&busy).expect("128 small entries are busy print, not a screen");
        assert!(
            page.elements.len() >= CLASSICAL_BUSY_PAGE_MIN_BOXES_96_SITS_BETWEEN_THE_5K_SCREENS_52_FRAGMENTS_AND_THE_GAZETTES_130_ENTRIES,
            "{} boxes",
            page.elements.len()
        );
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

    fn split_line_page(with_trickle: bool) -> RgbImage {
        RgbImage::from_fn(800, 600, move |x, y| {
            let line1_left = (96..400).contains(&x) && (96..160).contains(&y);
            let line2 = (96..240).contains(&x) && (176..240).contains(&y);
            let line1_tail = (432..640).contains(&x) && (96..160).contains(&y);
            let trickle = with_trickle && x == 412 && y == 100;
            if line1_left || line2 || line1_tail || trickle {
                [0, 0, 0]
            } else {
                [255, 255, 255]
            }
        })
    }

    #[test]
    fn a_split_wrapped_line_rejoins_across_an_ink_gap_after_the_raw_read_but_a_white_gutter_stays_split(
    ) {
        let page = classical_scout(&split_line_page(false)).expect("light page must scout");
        assert_eq!(
            page.elements.len(),
            2,
            "a gap that is white at every sample is a gutter, never a split line: {:?}",
            page.elements
        );
        let img = split_line_page(true);
        let ScoutRead::Boxes(mut sb) = scout_components(&img) else {
            panic!("light page must scout");
        };
        assert_eq!(
            sb.boxes.len(),
            2,
            "the raw read that feeds the routing guards must keep the two-cell-only law: {:?}",
            sb.boxes
        );
        bridge_split_lines(&mut sb);
        assert_eq!(sb.boxes.len(), 1, "the bridge must rejoin: {:?}", sb.boxes);
        let page = classical_scout(&img).expect("light page must scout");
        assert_eq!(
            page.elements.len(),
            1,
            "the sampled glyph sliver in the gap band proves the gap is inside a line, so \
             the tail must rejoin its entry: {:?}",
            page.elements
        );
        let b = page.elements[0].bbox.unwrap();
        assert!(
            b[0] <= 96.0 && b[2] >= 632.0 && b[3] >= 232.0,
            "the merged box must cover both lines and the tail: {b:?}"
        );
    }

    #[test]
    fn thin_ui_strips_defer_to_the_model_scout_and_grey_linework_is_invisible() {
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
}
