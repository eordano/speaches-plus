use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_weights::WeightLoader;
use tokenizers::Tokenizer;

#[cfg(feature = "wgpu")]
use super::decoder::EOS_TOKEN_IDS;
use super::decoder::{
    build_prompt_tokens, DotsDecoder, DotsDecoderConfig, GenerateOptions, GenerateOutcome,
    PromptStyle,
};
#[cfg(feature = "wgpu")]
use super::decoder_wgpu::DotsQ3dDecoder;
#[cfg(feature = "wgpu")]
use super::decoder_wgpu_graph::{DotsDecoderWgpuGraph, DotsGraphQuant};
use super::parse::{parse_layout_json, plain_text_fallback, LayoutElement, LayoutPage};
use super::preprocess::{
    prepare, sample_ink, InkPoints, PixelBudget, PreparedImage, DEFAULT_MIN_PIXELS,
};
use super::regions::{
    category_prefix, env_pixels, layout_is_usable, plan_regions, CROP_MAX_PIXELS_ENV,
    DEFAULT_CROP_MAX_PIXELS_500K_IS_ABOUT_640_VISION_TOKENS,
    DEFAULT_LAYOUT_MAX_PIXELS_1MP_KEEPS_THE_DETECT_PASS_CHEAP, LAYOUT_MAX_PIXELS_ENV,
};

const LAYOUT_RETRY_TRIPLES_THE_BUDGET_BECAUSE_A_LOOPING_DETECT_PASS_MEANS_UNDERRESOLVED_INK: usize =
    3;
const LAYOUT_KEEPS_AT_LEAST_A_NINTH_OF_SOURCE_PIXELS_OR_HANDWRITING_LOOPS: usize = 9;
use super::vision::{DotsVisionConfig, DotsVisionTower};
#[cfg(feature = "wgpu")]
use super::vision_wgpu::DotsVisionTowerWgpu;
use crate::deepseek_ocr::preprocess::RgbImage;

pub const PROMPT_LAYOUT_ALL_EN: &str = "Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.\n\n1. Bbox format: [x1, y1, x2, y2]\n\n2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n\n3. Text Extraction & Formatting Rules:\n    - Picture: For the 'Picture' category, the text field should be omitted.\n    - Formula: Format its text as LaTeX.\n    - Table: Format its text as HTML.\n    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n4. Constraints:\n    - The output text must be the original text from the image, with no translation.\n    - All layout elements must be sorted according to human reading order.\n\n5. Final Output: The entire output must be a single JSON object.\n";

pub const PROMPT_LAYOUT_ONLY_EN: &str = "Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.";

pub const PROMPT_OCR: &str = "Extract the text content from this image.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DotsPrecision {
    Bf16,
    Nvfp4,
    W4,
    W8,
}

pub const DOTS_PRECISION_ENV: &str = "NV_DOTS_DEC_PRECISION";

pub const W4_IS_THE_METAL_SERVING_DEFAULT_AND_BF16_STAYS_THE_MASTER_NUMERICS_ARM: &str =
    "The metal serving default is affine gs=64 on the graph, by the fastest-possible-defaults \
     directive: the consistent w4 arm holds all twelve stable pages at exactly CER 0.0000, its \
     numerics are the mlx-community 4-bit release bit for bit, it serves 121 ms/page faster \
     today (4419.0 vs 4539.9 canonical means), and it is the only arm whose decode floor keeps \
     dropping with byte-bound wins -- 0.96 GB/step against bf16's immovable 3.4. \
     NV_DOTS_DEC_PRECISION=bf16 is the master-numerics arm, one env away. The mixed \
     composition -- bf16 kv cache with quantised decode weights -- broke a stable page and is \
     reachable only through the explicit NV_DOTS_DEC_W4 measurement env, never through any \
     default. Cost carried by the default: quantising the trunk at load, measured 8.5 s \
     against bf16's 3.3 s cold start. CUDA and CPU defaults are unchanged.";

impl DotsPrecision {
    pub fn default_for(device: &Device) -> Self {
        let _ = W4_IS_THE_METAL_SERVING_DEFAULT_AND_BF16_STAYS_THE_MASTER_NUMERICS_ARM;
        if cfg!(feature = "wgpu") && device.is_metal() {
            return DotsPrecision::W4;
        }
        DotsPrecision::Bf16
    }

    pub fn from_env_for(device: &Device) -> Result<Self> {
        Self::parse(
            std::env::var(DOTS_PRECISION_ENV).ok().as_deref(),
            Self::default_for(device),
        )
    }

    pub fn parse(raw: Option<&str>, default: Self) -> Result<Self> {
        match raw {
            None | Some("") => Ok(default),
            Some("bf16") => Ok(DotsPrecision::Bf16),
            Some("nvfp4") => Ok(DotsPrecision::Nvfp4),
            Some("w4") => Ok(DotsPrecision::W4),
            Some("w8") => Ok(DotsPrecision::W8),
            Some(other) => anyhow::bail!(
                "{DOTS_PRECISION_ENV}={other:?} is not a dots.ocr decoder precision; use bf16, \
                 nvfp4 (cuda), w4 (affine gs=64 on the metal graph, ggml Q4K via candle \
                 elsewhere), or w8 (ggml Q8_0, any device)"
            ),
        }
    }
}

pub const DOTS_VISION_BACKEND_ENV: &str = "NV_DOTS_VISION_BACKEND";

pub const DOTS_FAST_PATH_DEFAULTS_ARE_SCOPED_TO_METAL_BECAUSE_THAT_IS_THE_ONLY_ADAPTER_THEY_WERE_MEASURED_ON: &str =
    "The wgpu tower and the graphed wgpu decoder default on for a Metal serving device and \
     nowhere else. Every number that justifies them -- the tower's 1.78 s/page over the candle \
     tower at identical trajectories, the graph's 9.93 s against 35.47 s composed over an invoice \
     page -- was taken on an Apple M2 Ultra, and the kernels behind them are fitted to an 8x8x8 \
     cooperative-matrix adapter with a 32768 byte workgroup budget. A CUDA box builds the same \
     wgpu feature and would silently inherit a default nobody measured there, so the predicate is \
     the serving device and not the feature flag. NV_DOTS_VISION_BACKEND and NV_DOTS_DEC_ENGINE \
     still name either arm explicitly on any device, and =candle restores the pre-flip path.";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DotsVisionBackend {
    #[default]
    Candle,
    Wgpu,
}

impl DotsVisionBackend {
    pub fn name(self) -> &'static str {
        match self {
            DotsVisionBackend::Candle => "candle",
            DotsVisionBackend::Wgpu => "wgpu",
        }
    }

    pub fn default_for(device: &Device) -> Self {
        let _ = DOTS_FAST_PATH_DEFAULTS_ARE_SCOPED_TO_METAL_BECAUSE_THAT_IS_THE_ONLY_ADAPTER_THEY_WERE_MEASURED_ON;
        if cfg!(feature = "wgpu") && device.is_metal() {
            DotsVisionBackend::Wgpu
        } else {
            DotsVisionBackend::Candle
        }
    }

    pub fn parse(raw: Option<&str>, default: Self) -> Result<Self> {
        match raw {
            None | Some("") => Ok(default),
            Some("0") | Some("candle") => Ok(DotsVisionBackend::Candle),
            Some("wgpu") => Ok(DotsVisionBackend::Wgpu),
            Some(other) => anyhow::bail!(
                "{DOTS_VISION_BACKEND_ENV}={other:?} is not a dots.ocr vision backend; \
                 use wgpu (the default on metal) or candle (also =0)"
            ),
        }
    }

    pub fn from_env_for(device: &Device) -> Result<Self> {
        Self::parse(
            std::env::var(DOTS_VISION_BACKEND_ENV).ok().as_deref(),
            Self::default_for(device),
        )
    }
}

pub const LOADING_THE_CANDLE_DECODER_BESIDE_A_LIVE_GRAPH_DUPLICATES_THE_WHOLE_TRUNK_FOR_NOTHING:
    &str = "graph_generate returns None only when there is no graph, so a live graph serves EVERY
     generate and the candle decoder beside it is never called. Building it anyway put a second
     full copy of the trunk in memory -- about 3 GiB of bf16 beside the graph's own 7.1 GiB --
     which is why the decoder is built only when the graph is not serving. This mattered the
     moment the graph became the default on metal: before that the duplicate was rare because
     the graph was opt-in, and after it, it was on every single load. It is also the first thing
     to suspect behind an out-of-memory kill on a small box, ahead of anything a 4-bit weight
     format would fix.";

pub const DOTS_DEC_ENGINE_ENV: &str = "NV_DOTS_DEC_ENGINE";

pub const DOTS_VISION_HANDOFF_ENV: &str = "NV_DOTS_VISION_HANDOFF";

pub const VISION_HANDOFF_STAYS_ON_DEVICE_BECAUSE_TOWER_AND_GRAPH_SHARE_ONE_WGPU_QUEUE: &str =
    "The wgpu tower and the graph decoder both run on WgpuContext::shared(), so the merged \
     features never need the host: encode_resident packs them to the bf16 words the prefill \
     splice consumes and generate_gpu_rows copies them buffer-to-buffer, deleting the 21 MB \
     merger readback, the candle bf16 staging round trip (a third Metal context), the 21 MB \
     download back to host, the host repack and the 10.5 MB splice re-upload on an invoice-size \
     page. The splice words are byte-identical to the shipped host chain because every host hop \
     only ever rounded once (see HANDOFF_SPLICE_WORDS_MATCH...). NV_DOTS_VISION_HANDOFF=0 \
     restores the host handoff everywhere, region crops included.";

#[cfg(feature = "wgpu")]
fn vision_handoff_enabled() -> bool {
    let _ = VISION_HANDOFF_STAYS_ON_DEVICE_BECAUSE_TOWER_AND_GRAPH_SHARE_ONE_WGPU_QUEUE;
    std::env::var(DOTS_VISION_HANDOFF_ENV).ok().as_deref() != Some("0")
}

pub const DOTS_REGION_HANDOFF_ENV: &str = "NV_DOTS_REGION_HANDOFF";

pub const REGION_CROPS_RIDE_THE_SAME_RESIDENT_SPLICE_BECAUSE_THE_TIERED_PATH_PAID_THE_HOST_HOP_PER_CROP: &str =
    "The tiered route hands vision rows to the decoder once per crop, so the host seam the \
     whole-page handoff deleted once per page multiplied by the region count: batch-encoded \
     crops staged through candle and re-downloaded before the prefill repack, bucket-miss \
     crops rode the full candle round trip, and the batched-region prefill re-packed every \
     row on the host. encode_batch_resident packs the batched merger output per crop with \
     the gated dvh pack pass, encode_resident serves bucket misses, and region_prefill \
     splices buffer-to-buffer -- the same once-rounded bf16 words either way (see \
     HANDOFF_SPLICE_WORDS_MATCH...). NV_DOTS_REGION_HANDOFF=0 restores the host rows on \
     region paths alone; NV_DOTS_VISION_HANDOFF=0 restores them everywhere. q3d and candle \
     engines never see gpu rows -- the producer gates on the graph serving alone.";

#[cfg(feature = "wgpu")]
fn region_handoff_enabled() -> bool {
    let _ = REGION_CROPS_RIDE_THE_SAME_RESIDENT_SPLICE_BECAUSE_THE_TIERED_PATH_PAID_THE_HOST_HOP_PER_CROP;
    vision_handoff_enabled() && std::env::var(DOTS_REGION_HANDOFF_ENV).ok().as_deref() != Some("0")
}

#[cfg(feature = "wgpu")]
enum PreRows {
    Host(Vec<f32>),
    Gpu(super::decoder_wgpu_graph::VisionRowsGpu),
}

#[cfg(not(feature = "wgpu"))]
type PreRows = Vec<f32>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DotsDecoderEngine {
    #[default]
    Candle,
    Graph,
    Q3d,
}

impl DotsDecoderEngine {
    pub fn name(self) -> &'static str {
        match self {
            DotsDecoderEngine::Candle => "candle",
            DotsDecoderEngine::Graph => "graph",
            DotsDecoderEngine::Q3d => "q3d",
        }
    }

    pub fn default_for(device: &Device, precision: DotsPrecision) -> Self {
        let _ = DOTS_FAST_PATH_DEFAULTS_ARE_SCOPED_TO_METAL_BECAUSE_THAT_IS_THE_ONLY_ADAPTER_THEY_WERE_MEASURED_ON;
        let _ = THE_GRAPH_CARRIES_NO_GGML_ARM_SO_A_GGML_PRECISION_MUST_KEEP_THE_CANDLE_ENGINE;
        let graphable = matches!(
            precision,
            DotsPrecision::Bf16 | DotsPrecision::Nvfp4 | DotsPrecision::W4
        );
        if cfg!(feature = "wgpu") && device.is_metal() && graphable {
            DotsDecoderEngine::Graph
        } else {
            DotsDecoderEngine::Candle
        }
    }

    pub fn parse(raw: Option<&str>, default: Self) -> Result<Self> {
        match raw {
            None | Some("") => Ok(default),
            Some("0") | Some("candle") => Ok(DotsDecoderEngine::Candle),
            Some("graph") => Ok(DotsDecoderEngine::Graph),
            Some("q3d") => Ok(DotsDecoderEngine::Q3d),
            Some(other) => anyhow::bail!(
                "{DOTS_DEC_ENGINE_ENV}={other:?} is not a dots.ocr decoder engine; \
                 use graph (the graphed wgpu decoder, the default on metal at a graphable \
                 precision), candle (also =0), or q3d (the qwen2-bias arm of the q3d wgpu \
                 graph fed from the mlx-community 4-bit snapshot)"
            ),
        }
    }

    pub fn from_env_for(device: &Device, precision: DotsPrecision) -> Result<Self> {
        Self::parse(
            std::env::var(DOTS_DEC_ENGINE_ENV).ok().as_deref(),
            Self::default_for(device, precision),
        )
    }
}

pub const THE_GRAPH_CARRIES_NO_GGML_ARM_SO_A_GGML_PRECISION_MUST_KEEP_THE_CANDLE_ENGINE: &str =
    "w8 (ggml Q8_0) has no graph arm and is served by the candle decoder through candle's \
     QMatMul; asking for graph+w8 explicitly errors rather than silently downgrading, because \
     a caller who names both has stated an intent the engine cannot honour. w4 on the graph is \
     NOT ggml: it quantizes the bf16 master to mlx affine gs=64 at load -- the exact tensors of \
     the mlx-community 4-bit release -- serves prefill from their dequantisation and decode \
     from the packed codes. With engine=candle, w4 keeps its old ggml Q4K meaning.";

enum VisionTower {
    Candle(DotsVisionTower),
    #[cfg(feature = "wgpu")]
    Wgpu(DotsVisionTowerWgpu),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DotsMode {
    #[default]
    LayoutAll,
    LayoutOnly,
    PlainOcr,
}

impl DotsMode {
    pub fn prompt(self) -> &'static str {
        match self {
            DotsMode::LayoutAll => PROMPT_LAYOUT_ALL_EN,
            DotsMode::LayoutOnly => PROMPT_LAYOUT_ONLY_EN,
            DotsMode::PlainOcr => PROMPT_OCR,
        }
    }

    pub fn emits_json(self) -> bool {
        !matches!(self, DotsMode::PlainOcr)
    }
}

#[derive(Debug, Clone)]
pub struct DotsPageResult {
    pub page: LayoutPage,
    pub text: String,
    pub raw: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub looped: bool,
    pub hit_eos: bool,
    pub grid: (usize, usize),
}

pub const INK_RECALL_RESCUES_SPARSE_SHEETS_THE_PAGE_AREA_FLOOR_WRONGLY_SENDS_BACK_TO_SINGLE_PASS:
    &str =
    "layout_is_usable gates on region area over PAGE area (0.15), which reads a large sheet \
     with small text -- an A0 plan at 50MP measures 0.02 -- as a detector miss and falls back \
     to the whole-page pass that cannot resolve exactly that input. The ink check accepts the \
     layout instead when the region union contains at least 85 percent of sampled dark-luma \
     points, which is what a detector that actually saw the page produces; light-grey \
     linework sits above the luma threshold on purpose so drawings do not count as text ink. \
     Pages with almost no dark samples return None and keep the old page-area verdict.";

pub const INK_RECALL_USABLE_MIN_085_A_DETECTOR_BOXING_85PCT_OF_THE_INK_SAW_THE_PAGE: f64 = 0.85;
pub const MIN_INK_POINTS_200_BELOW_WHICH_THE_PAGE_IS_BLANK_TO_THE_INK_CHECK: usize = 200;

fn ink_recall(ink: &InkPoints, regions: &[super::regions::Region]) -> Option<f64> {
    if ink.xs.len() < MIN_INK_POINTS_200_BELOW_WHICH_THE_PAGE_IS_BLANK_TO_THE_INK_CHECK {
        return None;
    }
    let hit = ink
        .xs
        .iter()
        .zip(&ink.ys)
        .filter(|&(&x, &y)| {
            regions
                .iter()
                .any(|r| x >= r.bbox[0] && x <= r.bbox[2] && y >= r.bbox[1] && y <= r.bbox[3])
        })
        .count();
    Some(hit as f64 / ink.xs.len() as f64)
}

pub const HEADING_ONLY_CROPS_BAIT_WHITESPACE_ESSAYS_SO_HEADINGS_MERGE_INTO_THE_BLOCK_BELOW: &str =
    "a crop holding just a large-font heading is mostly white space, and the decoder answers \
     it with an invented essay or an eleven-fold repetition of the heading (measured on the \
     090-scale corpus: Appendix B grew a 500-token vibration-survey essay, Ordering and Lead \
     Times repeated eleven times). Merging a Title/Section-header/Page-header region into the \
     next region in reading order, when it sits directly above it, removes the bait: the crop \
     then reads heading plus body like any dense page. A heading left unmerged (page end, no \
     block below) keeps a 256-token cap instead -- long generation from a one-line crop is \
     the failure signature, never legitimate output.";

fn heading_like(category: Option<&str>) -> bool {
    matches!(
        category,
        Some("Title") | Some("Section-header") | Some("Page-header")
    )
}

const HEADING_MERGE_MAX_GAP_2X_HEADING_HEIGHT: f32 = 2.0;
const UNMERGED_HEADING_CROP_TOKEN_CAP: usize = 256;

pub const LAYOUT_ONLY_STAYS_SINGLE_PASS_BECAUSE_A_BOXES_ONLY_CALLER_MUST_NOT_PAY_CROP_OCR: &str =
    "recognize_cropped answers every engaged mode with per-region plain OCR, which is the \
     LayoutAll contract (bbox, category, text; Picture regions are omitted because \
     plan_regions drops them for cropping) and the PlainOcr contract stitched in reading \
     order -- but a LayoutOnly caller asked for boxes without text, and tiling would spend \
     the whole crop-OCR wall producing text it did not ask for. LayoutOnly therefore always \
     rides the single detect pass, whose own cost the recognize_cropped ladder already \
     bounds at a 3MP-class budget when it is the one calling.";

struct RegionOcr {
    elements: Vec<LayoutElement>,
    prompt_tokens: usize,
    generated_tokens: usize,
    looped: bool,
    hit_eos: bool,
}

pub const T1_BATCH_ENV: &str = "NV_DOTS_T1_BATCH";
pub const SCREEN_DETOUR_ENV: &str = "NV_DOTS_SCREEN_DETOUR";
pub const T1_HANDOFF_ENV: &str = "NV_DOTS_T1_HANDOFF";

pub const T1_HANDOFF_DEFAULT_0_LIVE_ROWS_KEEPS_THE_BATCH_TO_ITS_LAST_ROW: usize = 0;

pub const A_THIN_BATCH_TAIL_KEEPS_BATCHING_BECAUSE_THE_GAZETTE_LADDER_MEASURED_EVERY_HANDOFF_THRESHOLD_LOSING: &str =
    "the sparse-batch red predicted a win from retiring a thin batch tail (under the ~5.2 \
     live-row break-even) into serial continuation -- each survivor re-prefills its prompt \
     plus generated-so-far through the steel serial path over its already-encoded vision \
     rows and decodes on at the serial per-token price. The interleaved gazette ladder \
     measured the opposite: thresholds 2/3/5 read 19.69/19.70/20.00 s against 19.50-19.53 s \
     with the handoff off, token-identical output (gen 3147, ncer 0.0194) on every arm -- at \
     drain time the survivors' remaining tails are short, so the re-prefill costs more than \
     the thin bsteps it replaces. The handoff therefore defaults off; NV_DOTS_T1_HANDOFF=N \
     retires survivors at N live rows for pages whose drain holds a long-generation \
     straggler, the unmeasured win case. A handed-off continuation re-argues ngram bans \
     against real logits like any serial generation, but its ban window restarts at the \
     handoff and the prefill-vs-decode trunk reassociation makes the tail token-equal \
     class, never bitwise -- covered by the tiered quality suites.";

#[cfg(feature = "wgpu")]
fn t1_handoff_max_live_rows() -> usize {
    std::env::var(T1_HANDOFF_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(T1_HANDOFF_DEFAULT_0_LIVE_ROWS_KEEPS_THE_BATCH_TO_ITS_LAST_ROW)
}

pub const DARK_INVERT_ENV: &str = "NV_DOTS_DARK_INVERT";

pub const DARK_PAGES_INVERT_BEFORE_THE_TOWER_BECAUSE_THE_MODEL_WAS_TRAINED_ON_INK_OVER_LIGHT: &str =
    "a dark theme is a light page inverted, and the tower, the scout and the tier ladder \
     were all measured on ink-over-light: the 4K dark screen reads recall 0.6667 / ncer \
     0.3196 in 41 s native (one boxed region, truncated) against 0.9444 / 0.1300 in 7.2 s \
     inverted (nine regions, tiling engages). recognize_auto therefore inverts a detected \
     dark page before any routing, for PlainOcr only -- layout modes are unmeasured and \
     keep the native read. The inversion is PREPROCESSING, not routing: it composes with \
     every tiering policy and caller hint unchanged, and NV_DOTS_DARK_INVERT=0 is its own \
     operator kill-switch restoring the native read. Detection is the scout's paired ink \
     reads (page_is_dark_themed), gated so the corpus and every light fixture answer \
     false; the single-pass recognize() entry is untouched, so the family band cannot \
     move. The remaining unmeasured edge: sub-threshold dark pages that are not \
     screen-like invert and then ride the single pass.";

pub const SERIAL_CROP_ENCODES_BATCH_BY_GRID_BECAUSE_TOWER_COST_IS_PER_CALL_NOT_PER_PATCH: &str =
    "a page under the 24-region floor decodes serially by the sparse-batch rule, but its \
     crops still paid one tower call each (~49 ms fixed upload/dispatch/readback against \
     under 1 ms compute on a small crop), so the serial path buckets same-grid crops \
     through the same encode_batch the batched-decode path rides -- gated bitwise against \
     single encodes -- and stages the rows through the identical bf16 tensor conversion, \
     leaving decode arithmetic untouched byte for byte. NV_DOTS_T1_BATCH=0 restores \
     per-crop encodes together with sequential decode.";

pub const T1_BATCH_MAX_CROP_VISION_TOKENS_896_IS_THE_STRIDE_BOUND_NOW_THAT_RANGED_PREFILL_IS_STEEL: usize =
    896;

pub const T1_BATCH_MIN_REGIONS_24_A_STEP_COSTS_THE_FULL_TRUNK_SO_SPARSE_BATCHES_LOSE_TO_SERIAL:
    usize = 24;

pub const SPARSE_BATCHES_LOSE_BECAUSE_THE_TRUNK_PRICE_IS_PER_STEP_NOT_PER_ROW: &str =
    "a batched step costs the full m=32 trunk (~40-45 ms) whether one row or thirty-two are \
     live, so the win requires enough live rows on average: the 130-entry gazette keeps the \
     batch full (~98 steps for ~3145 tokens, 22.6 s end-to-end vs 37.5 sequential) while \
     the 4K screen's 11 skew-length regions average two or three live rows and read 10.8 s \
     batched against 6.5 serial. Pages under 24 regions take the serial crop path; the \
     refinement left on the table is straggler handoff, retiring a thin batch tail into \
     serial continuation, which needs a serial step that reads a region's KV stride.";

pub const A_BANNED_ROW_REDOES_SERIALLY_BECAUSE_THE_BATCH_KEEPS_NO_PER_ROW_LOGITS_TO_REARGUE: &str =
    "the serial path answers a fired ngram ban by re-argmaxing the step's logits with the \
     banned tokens excluded and generating on; the batch persists only each row's argmax \
     token, so a banned row stopping there kept its repetition and read ncer 0.24 against \
     0.06 serial on the 1080p screen. A region whose ban fires is therefore retired and \
     redone through the serial crop path, which restores exact serial semantics at the \
     price of one crop re-run on the rare looping region.";

pub const T1_BATCH_MAX_PROMPT_KEEPS_LONG_PROMPT_PAGES_ON_STEEL_PREFILL: &str =
    "region prefill rides the steel causal path through ranged KV mirror conversion and \
     prices UNDER the serial padded-chunk prefill (0.87x at 200-token prompts, 0.74x at \
     700), so the ceiling is no longer the naive-attention regression that once kept any \
     crop over 256 vision tokens serial (4K screen: 10.9 s batched vs 6.5 serial in that \
     era); it is the 1024-row stride itself: 896 vision tokens plus the text prompt and a \
     generation floor must fit under the region's reserved-row cap, and any crop over the \
     bound still sends the whole page down the serial path.";

pub const BATCHED_T1_RIDES_THE_GRAPHS_REGION_MACHINERY_AND_ESCAPES_TO_SERIAL_CROPS: &str =
    "crop decode is 54 percent of the sequential tiered wall and the graph's batched region \
     step serves 32 concurrent generations at 1.4 ms/token against 7.8 serial (5.5x, \
     region-prefill at cost parity), so tiled OCR decodes its crops through region strides \
     with retire-and-refill. Vision stays per-crop; a region whose prompt cannot fit the \
     1023-row stride sends the whole set down the serial path, as does NV_DOTS_T1_BATCH=0, \
     a non-graph engine, or a single region. Ngram bans stop a region instead of re-arguing \
     its logits, and loop truncation applies at onset host-side exactly as the serial path \
     truncates plain OCR.";

pub const CROP_PACKING_IS_A_MEASURED_NEGATIVE_UNION_BOXES_SHIP_WHITESPACE_THROUGH_THE_TOWER: &str =
    "stacking consecutive same-column Text regions into one crop halved T1 prefill on the \
     121-entry gazette (12.0 to 6.0 s over 130 to 41 crops) but the union boxes carried \
     the inter-entry gaps as pixels, growing vision from 6.4 to 8.4 s, and multi-entry \
     crops reintroduced entry skips (ncer 0.0199 to 0.0289; a whole entry dropped) -- net \
     -2.8 s wall for worse text on the one page shaped for it, and a quality wash \
     elsewhere. The packer is deleted on that measurement; the standing T1 levers are \
     batched B>1 vision over SMALL crops (steel attention is batch-native) and \
     multi-sequence decode, which own 17 and 54 percent of the sequential wall.";

pub const STACKED_HEADINGS_CHAIN_THE_MERGE_OR_A_CATEGORY_LESS_PAIR_STILL_BAITS_THE_ESSAY: &str =
    "a title over a section line is two heading regions in reading order; merging the first \
     into the second must keep the pair heading-like, because a merged pair whose category \
     was erased could never merge into the body below it and became exactly the \
     mostly-whitespace crop the heading merge exists to remove (measured on the dark \
     document render: the title+section crop invented a nine-item watch-care essay). The \
     chain stops at the first non-heading absorption, which erases the category as before \
     so the crop reads as plain text and no heading token cap applies.";

fn merge_headings_into_following_regions(
    regions: Vec<super::regions::Region>,
) -> Vec<super::regions::Region> {
    let _ = STACKED_HEADINGS_CHAIN_THE_MERGE_OR_A_CATEGORY_LESS_PAIR_STILL_BAITS_THE_ESSAY;
    let mut out: Vec<super::regions::Region> = Vec::with_capacity(regions.len());
    let mut it = regions.into_iter().peekable();
    while let Some(mut r) = it.next() {
        while heading_like(r.category.as_deref()) {
            let Some(next) = it.peek() else { break };
            let gap = next.bbox[1] - r.bbox[3];
            let x_overlap = next.bbox[0] < r.bbox[2] && r.bbox[0] < next.bbox[2];
            if !(x_overlap
                && gap > -r.height()
                && gap < r.height() * HEADING_MERGE_MAX_GAP_2X_HEADING_HEIGHT)
            {
                break;
            }
            let next = it.next().expect("peeked region present");
            let absorbed_a_heading = heading_like(next.category.as_deref());
            r.bbox = [
                r.bbox[0].min(next.bbox[0]),
                r.bbox[1].min(next.bbox[1]),
                r.bbox[2].max(next.bbox[2]),
                r.bbox[3].max(next.bbox[3]),
            ];
            r.category = if absorbed_a_heading {
                next.category
            } else {
                None
            };
        }
        out.push(r);
    }
    out
}

pub const RELOAD_REBUILDS_BESIDE_THE_RESIDENT_DECODER_SO_PEAK_IS_TWICE_ONE_ARM: &str =
    "reload_adapter constructs the replacement decoder while the old one still holds the \
     slot, so peak decoder memory during a reload is twice one arm (~23 GiB on the w4 \
     graph); dropping the old arm first would halve the peak but a failed rebuild would \
     then leave the pipeline with no decoder at all, and an explicit unload-then-load \
     already gives a caller that trade when memory is tight. reload_adapter(None) is a \
     true unload: the explicit argument wins over NV_DOTS_GRAPH_LORA_DIR, which only the \
     boot path reads.";

pub const DOTS_GRAPH_FROM_MLX_ENV_SKIPS_THE_REDNOTE_BF16_INTERMEDIATE: &str =
    "NV_DOTS_GRAPH_FROM_MLX";

#[cfg(feature = "wgpu")]
fn graph_from_mlx_enabled() -> bool {
    std::env::var(DOTS_GRAPH_FROM_MLX_ENV_SKIPS_THE_REDNOTE_BF16_INTERMEDIATE)
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[cfg(feature = "wgpu")]
fn graph_quant_for(precision: DotsPrecision) -> Result<DotsGraphQuant> {
    match precision {
        DotsPrecision::Bf16 => Ok(DotsGraphQuant::Bf16),
        DotsPrecision::Nvfp4 => Ok(DotsGraphQuant::Nvfp4),
        DotsPrecision::W4 => Ok(DotsGraphQuant::AffineW4),
        other => anyhow::bail!(
            "{DOTS_DEC_ENGINE_ENV}=graph carries no ggml arm, so it cannot serve \
             {DOTS_PRECISION_ENV}={other:?}; use bf16, nvfp4 or w4"
        ),
    }
}

pub const FROM_PARTS_IS_THE_BROWSER_SEAM_AND_MUST_STAY_THE_EXACT_METAL_SERVING_ARM: &str =
    "from_parts_wgpu_graph_w4 is load_with_precision's Wgpu-tower + AffineW4-graph arm fed from \
     an owned-bytes WeightLoader instead of a checkpoint dir, so a browser that downloaded the \
     same safetensors builds the same tower and the same graph and its texts gate bitwise \
     against the dir-loaded pipeline (nv-ocr-web tests/native_parity.rs). Anything this \
     constructor decides differently from that arm -- backend, engine, quant, tokenizer \
     sanitize -- breaks the parity gate, which is the point of having it. The candle Device is \
     Cpu because the graph arm never stages activations through candle; adapter reload keeps \
     the dir-based path and is not offered here.";

pub struct DotsOcrPipeline {
    vision: VisionTower,
    vision_backend: DotsVisionBackend,
    decoder: Option<DotsDecoder>,
    engine: DotsDecoderEngine,
    #[cfg(feature = "wgpu")]
    graph: Option<std::sync::Mutex<DotsDecoderWgpuGraph>>,
    #[cfg(feature = "wgpu")]
    q3d: Option<std::sync::Mutex<DotsQ3dDecoder>>,
    tokenizer: Tokenizer,
    budget: PixelBudget,
    style: PromptStyle,
    device: Device,
    precision: DotsPrecision,
    #[cfg(feature = "wgpu")]
    dir: std::path::PathBuf,
    gpu_handoff_crops: std::sync::atomic::AtomicU64,
}

pub fn dots_weight_dtype_bf16_on_every_device_whose_gemm_takes_it_which_is_cuda_and_metal_f32_only_on_cpu(
    device: &Device,
) -> DType {
    device.bf16_default_to_f32()
}

fn default_rope_span() -> usize {
    std::env::var("NV_DOTS_MAX_SEQ")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(32768)
}

impl DotsOcrPipeline {
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        Self::load_with_precision(dir, device, DotsPrecision::from_env_for(device)?)
    }

    pub fn load_with_precision(
        dir: &Path,
        device: &Device,
        precision: DotsPrecision,
    ) -> Result<Self> {
        let engine = DotsDecoderEngine::from_env_for(device, precision)?;
        #[cfg(not(feature = "cuda"))]
        if precision == DotsPrecision::Nvfp4 && engine != DotsDecoderEngine::Graph {
            anyhow::bail!(
                "dots.ocr: the candle DotsPrecision::Nvfp4 decoder requires the cuda feature; \
                 rebuild with --features cuda, or use DotsPrecision::Bf16, or set \
                 {DOTS_DEC_ENGINE_ENV}=graph whose nvfp4 arm is wgpu and runs on any adapter"
            );
        }
        let weights = WeightLoader::open_dir(dir, device)
            .with_context(|| format!("open dots.ocr checkpoint dir {}", dir.display()))?;
        let cfg = DotsDecoderConfig::from_hf_json_file(&dir.join("config.json"))
            .context("load dots.ocr config.json")?;
        let dtype = dots_weight_dtype_bf16_on_every_device_whose_gemm_takes_it_which_is_cuda_and_metal_f32_only_on_cpu(device);
        let vision_backend = DotsVisionBackend::from_env_for(device)?;
        let vision = match vision_backend {
            DotsVisionBackend::Candle => VisionTower::Candle(
                DotsVisionTower::from_loader(
                    &weights,
                    "vision_tower.",
                    DotsVisionConfig::default(),
                    device,
                    dtype,
                )
                .context("load dots.ocr vision tower")?,
            ),
            #[cfg(feature = "wgpu")]
            DotsVisionBackend::Wgpu => VisionTower::Wgpu(
                DotsVisionTowerWgpu::from_loader(
                    &weights,
                    "vision_tower.",
                    DotsVisionConfig::default(),
                )
                .context("load dots.ocr wgpu vision tower")?,
            ),
            #[cfg(not(feature = "wgpu"))]
            DotsVisionBackend::Wgpu => anyhow::bail!(
                "{DOTS_VISION_BACKEND_ENV}=wgpu requires the wgpu feature; \
                 rebuild with NVK_FEATURES=cuda,wgpu"
            ),
        };
        #[cfg(feature = "wgpu")]
        let q3d = match engine {
            DotsDecoderEngine::Q3d => {
                let d = DotsQ3dDecoder::from_snapshot()
                    .context("load dots.ocr q3d wgpu decoder from the mlx 4-bit snapshot")?;
                eprintln!(
                    "[dots] {DOTS_DEC_ENGINE_ENV}=q3d: mlx-affine w4, {} passes per decode step",
                    d.pass_count()
                );
                Some(std::sync::Mutex::new(d))
            }
            _ => None,
        };
        #[cfg(feature = "wgpu")]
        let graph = match engine {
            DotsDecoderEngine::Candle | DotsDecoderEngine::Q3d => None,
            DotsDecoderEngine::Graph => {
                let quant = graph_quant_for(precision)?;
                let g = if quant == DotsGraphQuant::AffineW4 && graph_from_mlx_enabled() {
                    let mlx_dir = super::decoder_wgpu::dots_mlx_snapshot_dir()?;
                    let mlx_loader =
                        WeightLoader::open_dir(&mlx_dir, &Device::Cpu).with_context(|| {
                            format!("open mlx dots.ocr-4bit snapshot {}", mlx_dir.display())
                        })?;
                    eprintln!(
                        "[dots] {DOTS_GRAPH_FROM_MLX_ENV_SKIPS_THE_REDNOTE_BF16_INTERMEDIATE}=1: \
                         w4 graph from {}",
                        mlx_dir.display()
                    );
                    DotsDecoderWgpuGraph::from_loader_mlx(
                        cfg.clone(),
                        &mlx_loader,
                        default_rope_span(),
                    )
                    .context("load dots.ocr graphed wgpu decoder from the mlx snapshot")?
                } else {
                    DotsDecoderWgpuGraph::from_loader(
                        cfg.clone(),
                        &weights,
                        default_rope_span(),
                        quant,
                    )
                    .context("load dots.ocr graphed wgpu decoder")?
                };
                eprintln!(
                    "[dots] {DOTS_DEC_ENGINE_ENV}=graph: {} quant, {} passes per decode step, \
                     {:.3} GiB",
                    quant.name(),
                    g.pass_count(),
                    g.vram_report().total_bytes as f64 / (1u64 << 30) as f64
                );
                Some(std::sync::Mutex::new(g))
            }
        };
        #[cfg(not(feature = "wgpu"))]
        if engine == DotsDecoderEngine::Graph {
            anyhow::bail!(
                "{DOTS_DEC_ENGINE_ENV}=graph requires the wgpu feature; \
                 rebuild with NVK_FEATURES=metal,wgpu"
            );
        }
        #[cfg(feature = "wgpu")]
        let graph_serves_every_generate = graph.is_some() || q3d.is_some();
        #[cfg(not(feature = "wgpu"))]
        let graph_serves_every_generate = false;
        let _ =
            LOADING_THE_CANDLE_DECODER_BESIDE_A_LIVE_GRAPH_DUPLICATES_THE_WHOLE_TRUNK_FOR_NOTHING;
        let decoder = if graph_serves_every_generate {
            None
        } else {
            Some(match precision {
                DotsPrecision::Bf16 => {
                    DotsDecoder::from_loader(cfg, &weights, device, dtype, default_rope_span())
                        .context("load dots.ocr decoder bf16")?
                }
                DotsPrecision::Nvfp4 => {
                    #[cfg(feature = "cuda")]
                    {
                        DotsDecoder::from_loader_nvfp4(cfg, &weights, device, default_rope_span())
                            .context("load dots.ocr decoder nvfp4")?
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        DotsDecoder::from_loader(cfg, &weights, device, dtype, default_rope_span())
                        .context(
                            "load the bf16 candle decoder that stands unused behind the graph's \
                             own nvfp4 arm, which is the one that serves on a non-cuda build",
                        )?
                    }
                }
                DotsPrecision::W4 => DotsDecoder::from_loader_ggml(
                    cfg,
                    &weights,
                    device,
                    dtype,
                    candle_core::quantized::GgmlDType::Q4K,
                    default_rope_span(),
                )
                .context("load dots.ocr decoder w4 (ggml Q4K)")?,
                DotsPrecision::W8 => DotsDecoder::from_loader_ggml(
                    cfg,
                    &weights,
                    device,
                    dtype,
                    candle_core::quantized::GgmlDType::Q8_0,
                    default_rope_span(),
                )
                .context("load dots.ocr decoder w8 (ggml Q8_0)")?,
            })
        };
        let tokenizer = nv_tokenizer::load_tokenizer(&dir.join("tokenizer.json"))
            .context("load dots.ocr tokenizer.json")?;
        Ok(Self {
            vision,
            vision_backend,
            decoder,
            engine,
            #[cfg(feature = "wgpu")]
            graph,
            #[cfg(feature = "wgpu")]
            q3d,
            tokenizer,
            budget: PixelBudget::from_env(),
            style: PromptStyle::from_env(),
            device: device.clone(),
            precision,
            #[cfg(feature = "wgpu")]
            dir: dir.to_path_buf(),
            gpu_handoff_crops: std::sync::atomic::AtomicU64::new(0),
        })
    }

    #[cfg(feature = "wgpu")]
    pub fn from_parts_wgpu_graph_w4(
        weights: &WeightLoader,
        config_json: &str,
        tokenizer_json: &[u8],
        kv_span: usize,
    ) -> Result<Self> {
        Self::from_parts_wgpu_graph_w4_lora(weights, config_json, tokenizer_json, kv_span, None)
    }

    #[cfg(feature = "wgpu")]
    pub fn from_parts_wgpu_graph_w4_lora(
        weights: &WeightLoader,
        config_json: &str,
        tokenizer_json: &[u8],
        kv_span: usize,
        lora: Option<super::decoder_wgpu_graph::DgLora>,
    ) -> Result<Self> {
        Self::from_parts_wgpu_graph_w4_impl(
            weights,
            config_json,
            tokenizer_json,
            kv_span,
            lora,
            false,
        )
    }

    #[cfg(feature = "wgpu")]
    pub fn from_parts_wgpu_graph_w4_mlx_lora(
        weights: &WeightLoader,
        config_json: &str,
        tokenizer_json: &[u8],
        kv_span: usize,
        lora: Option<super::decoder_wgpu_graph::DgLora>,
    ) -> Result<Self> {
        Self::from_parts_wgpu_graph_w4_impl(
            weights,
            config_json,
            tokenizer_json,
            kv_span,
            lora,
            true,
        )
    }

    #[cfg(feature = "wgpu")]
    fn from_parts_wgpu_graph_w4_impl(
        weights: &WeightLoader,
        config_json: &str,
        tokenizer_json: &[u8],
        kv_span: usize,
        lora: Option<super::decoder_wgpu_graph::DgLora>,
        from_mlx: bool,
    ) -> Result<Self> {
        let _ = FROM_PARTS_IS_THE_BROWSER_SEAM_AND_MUST_STAY_THE_EXACT_METAL_SERVING_ARM;
        let cfg = DotsDecoderConfig::from_hf_json_str(config_json)
            .context("parse dots.ocr config.json")?;
        let vision = VisionTower::Wgpu(
            DotsVisionTowerWgpu::from_loader(weights, "vision_tower.", DotsVisionConfig::default())
                .context("load dots.ocr wgpu vision tower")?,
        );
        let lora = match lora {
            Some(l) => Some(l),
            None => super::decoder_wgpu_graph::DgLora::from_env(&cfg)?,
        };
        let graph = if from_mlx {
            DotsDecoderWgpuGraph::from_loader_mlx_with_lora(cfg, weights, kv_span, lora)
                .context("load dots.ocr graphed wgpu decoder from the mlx snapshot")?
        } else {
            DotsDecoderWgpuGraph::from_loader_with_lora(
                cfg,
                weights,
                kv_span,
                DotsGraphQuant::AffineW4,
                lora,
            )
            .context("load dots.ocr graphed wgpu decoder")?
        };
        let mut tokenizer = Tokenizer::from_bytes(tokenizer_json)
            .map_err(|e| anyhow::anyhow!("dots.ocr tokenizer.json from bytes: {e}"))?;
        nv_tokenizer::sanitize_for_serving(&mut tokenizer);
        Ok(Self {
            vision,
            vision_backend: DotsVisionBackend::Wgpu,
            decoder: None,
            engine: DotsDecoderEngine::Graph,
            graph: Some(std::sync::Mutex::new(graph)),
            q3d: None,
            tokenizer,
            budget: PixelBudget::from_env(),
            style: PromptStyle::from_env(),
            device: Device::Cpu,
            precision: DotsPrecision::W4,
            dir: std::path::PathBuf::new(),
            gpu_handoff_crops: std::sync::atomic::AtomicU64::new(0),
        })
    }

    pub fn precision(&self) -> DotsPrecision {
        self.precision
    }

    pub fn gpu_handoff_crops(&self) -> u64 {
        self.gpu_handoff_crops
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(feature = "wgpu")]
    pub fn visenc_packs(&self) -> u64 {
        match &self.vision {
            VisionTower::Wgpu(t) => t.visenc_packs(),
            VisionTower::Candle(_) => 0,
        }
    }

    #[cfg(feature = "wgpu")]
    pub fn rpf_chunks(&self) -> u64 {
        self.graph
            .as_ref()
            .and_then(|g| g.lock().ok())
            .map(|g| g.rpf_chunks())
            .unwrap_or(0)
    }

    #[cfg(feature = "wgpu")]
    pub fn rpf_packs(&self) -> u64 {
        self.graph
            .as_ref()
            .and_then(|g| g.lock().ok())
            .map(|g| g.rpf_packs())
            .unwrap_or(0)
    }

    #[cfg(feature = "wgpu")]
    pub fn bstep_bhead_steps(&self) -> u64 {
        self.graph
            .as_ref()
            .and_then(|g| g.lock().ok())
            .map(|g| g.bstep_bhead_steps())
            .unwrap_or(0)
    }

    #[cfg(feature = "wgpu")]
    pub fn bstep_bhead_mma_steps(&self) -> u64 {
        self.graph
            .as_ref()
            .and_then(|g| g.lock().ok())
            .map(|g| g.bstep_bhead_mma_steps())
            .unwrap_or(0)
    }

    #[cfg(feature = "wgpu")]
    pub fn trunk_splitk_families(&self) -> u64 {
        self.graph
            .as_ref()
            .and_then(|g| g.lock().ok())
            .map(|g| g.trunk_splitk_families())
            .unwrap_or(0)
    }

    #[cfg(feature = "wgpu")]
    pub fn battn_v2(&self) -> bool {
        self.graph
            .as_ref()
            .and_then(|g| g.lock().ok())
            .map(|g| g.battn_v2())
            .unwrap_or(false)
    }

    #[cfg(feature = "wgpu")]
    pub fn trunk_bf16out_families(&self) -> u64 {
        self.graph
            .as_ref()
            .and_then(|g| g.lock().ok())
            .map(|g| g.trunk_bf16out_families())
            .unwrap_or(0)
    }

    #[cfg(feature = "wgpu")]
    pub fn trunk_abf16_families(&self) -> u64 {
        self.graph
            .as_ref()
            .and_then(|g| g.lock().ok())
            .map(|g| g.trunk_abf16_families())
            .unwrap_or(0)
    }

    #[cfg(feature = "wgpu")]
    pub fn brope_kvw_fused_layers(&self) -> u64 {
        self.graph
            .as_ref()
            .and_then(|g| g.lock().ok())
            .map(|g| g.brope_kvw_fused_layers())
            .unwrap_or(0)
    }

    #[cfg(feature = "wgpu")]
    pub fn reload_q3d_with_adapter(
        &mut self,
        adapter_dir: Option<&std::path::Path>,
    ) -> Result<bool> {
        let Some(slot) = &self.q3d else {
            anyhow::bail!(
                "adapter hot-load rides the q3d decoder arm and this pipeline serves \
                 {}; set {DOTS_DEC_ENGINE_ENV}=q3d (or =graph, which also hot-loads)",
                self.engine.name()
            );
        };
        let d = DotsQ3dDecoder::from_snapshot_with_lora(adapter_dir)
            .context("rebuild dots q3d decoder with adapter")?;
        let resident = d.lora_resident();
        *slot.lock().expect("q3d decoder mutex") = d;
        Ok(resident)
    }

    #[cfg(feature = "wgpu")]
    pub fn reload_adapter(&mut self, adapter_dir: Option<&std::path::Path>) -> Result<bool> {
        let _ = RELOAD_REBUILDS_BESIDE_THE_RESIDENT_DECODER_SO_PEAK_IS_TWICE_ONE_ARM;
        match self.engine {
            DotsDecoderEngine::Q3d => self.reload_q3d_with_adapter(adapter_dir),
            DotsDecoderEngine::Graph => {
                let Some(slot) = &self.graph else {
                    anyhow::bail!(
                        "engine says graph but no graph decoder is resident; refusing a \
                         blind rebuild"
                    );
                };
                let cfg = DotsDecoderConfig::from_hf_json_file(&self.dir.join("config.json"))
                    .context("reload dots.ocr config.json")?;
                let lora = adapter_dir
                    .map(|d| super::decoder_wgpu_graph::DgLora::from_peft_dir(d, &cfg))
                    .transpose()
                    .context("load dots graph lora adapter")?;
                let weights =
                    WeightLoader::open_dir(&self.dir, &self.device).with_context(|| {
                        format!("reopen dots.ocr checkpoint dir {}", self.dir.display())
                    })?;
                let quant = graph_quant_for(self.precision)?;
                let resident = lora.is_some();
                let g = DotsDecoderWgpuGraph::from_loader_with_lora(
                    cfg,
                    &weights,
                    default_rope_span(),
                    quant,
                    lora,
                )
                .context("rebuild dots graph decoder with adapter")?;
                *slot.lock().expect("graph decoder mutex") = g;
                Ok(resident)
            }
            DotsDecoderEngine::Candle => anyhow::bail!(
                "adapter hot-load rides the q3d or graph decoder arms and this pipeline \
                 serves candle; set {DOTS_DEC_ENGINE_ENV}=graph or =q3d"
            ),
        }
    }

    pub fn prompt_style(&self) -> PromptStyle {
        self.style
    }

    pub fn set_prompt_style(&mut self, style: PromptStyle) {
        self.style = style;
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn decoder(&self) -> Option<&DotsDecoder> {
        let _ =
            LOADING_THE_CANDLE_DECODER_BESIDE_A_LIVE_GRAPH_DUPLICATES_THE_WHOLE_TRUNK_FOR_NOTHING;
        self.decoder.as_ref()
    }

    pub fn vision(&self) -> Option<&DotsVisionTower> {
        match &self.vision {
            VisionTower::Candle(t) => Some(t),
            #[cfg(feature = "wgpu")]
            VisionTower::Wgpu(_) => None,
        }
    }

    pub fn vision_backend(&self) -> DotsVisionBackend {
        self.vision_backend
    }

    pub fn decoder_engine(&self) -> DotsDecoderEngine {
        self.engine
    }

    #[cfg(feature = "wgpu")]
    pub fn graph_kv_cache_raw(&self) -> Option<Result<(usize, Vec<(Vec<u32>, Vec<u32>)>)>> {
        self.graph.as_ref().map(|g| {
            let g = g.lock().unwrap();
            g.kv_cache_raw().map(|c| (g.current_pos(), c))
        })
    }

    #[cfg(feature = "wgpu")]
    fn graph_generate(
        &self,
        tokens: &[u32],
        vision_start: usize,
        feats: &Tensor,
        opts: &GenerateOptions,
    ) -> Result<Option<GenerateOutcome>> {
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = (tokens, vision_start, feats, opts);
            return Ok(None);
        }
        #[cfg(feature = "wgpu")]
        {
            let Some(graph) = &self.graph else {
                return Ok(None);
            };
            let t_dl = std::time::Instant::now();
            let rows: Vec<f32> = feats.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            if super::decoder_wgpu_graph::page_profile_enabled() {
                eprintln!(
                    "DOTS-PAGE-PROFILE   vision_download_to_host_s={:.3} ({} values)",
                    t_dl.elapsed().as_secs_f64(),
                    rows.len()
                );
            }
            let mut graph = graph
                .lock()
                .map_err(|e| anyhow::anyhow!("dots.ocr graph decoder lock: {e}"))?;
            Ok(Some(graph.generate(
                tokens,
                vision_start,
                Some(&rows),
                opts,
            )?))
        }
    }

    #[cfg(feature = "wgpu")]
    fn q3d_generate(
        &self,
        tokens: &[u32],
        vision_start: usize,
        feats: &Tensor,
        opts: &GenerateOptions,
    ) -> Result<Option<GenerateOutcome>> {
        let Some(q3d) = &self.q3d else {
            return Ok(None);
        };
        let rows: Vec<f32> = feats.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let mut d = q3d
            .lock()
            .map_err(|e| anyhow::anyhow!("dots.ocr q3d decoder lock: {e}"))?;
        Ok(Some(d.generate(tokens, vision_start, Some(&rows), opts)?))
    }

    #[cfg(not(feature = "wgpu"))]
    fn q3d_generate(
        &self,
        _tokens: &[u32],
        _vision_start: usize,
        _feats: &Tensor,
        _opts: &GenerateOptions,
    ) -> Result<Option<GenerateOutcome>> {
        Ok(None)
    }

    #[cfg(not(feature = "wgpu"))]
    fn graph_generate(
        &self,
        _tokens: &[u32],
        _vision_start: usize,
        _feats: &Tensor,
        _opts: &GenerateOptions,
    ) -> Result<Option<GenerateOutcome>> {
        Ok(None)
    }

    pub fn encode_vision(&self, prep: &PreparedImage) -> Result<Tensor> {
        match &self.vision {
            VisionTower::Candle(t) => t.encode(prep),
            #[cfg(feature = "wgpu")]
            VisionTower::Wgpu(t) => {
                let merged = t.encode(prep)?;
                let rows = prep.num_vision_tokens();
                let cols = t.config().hidden_size;
                anyhow::ensure!(
                    merged.len() == rows * cols,
                    "wgpu vision tower produced {} values, expected {rows} x {cols}",
                    merged.len()
                );
                let dtype = dots_weight_dtype_bf16_on_every_device_whose_gemm_takes_it_which_is_cuda_and_metal_f32_only_on_cpu(&self.device);
                let t = std::time::Instant::now();
                let out = Tensor::from_vec(merged, (rows, cols), &Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_device(&self.device)?;
                if super::decoder_wgpu_graph::page_profile_enabled() {
                    eprintln!(
                        "DOTS-PAGE-PROFILE   vision_stage_to_candle_s={:.3} ({rows}x{cols})",
                        t.elapsed().as_secs_f64()
                    );
                }
                Ok(out)
            }
        }
    }

    pub fn pixel_budget(&self) -> PixelBudget {
        self.budget
    }

    pub fn set_pixel_budget(&mut self, budget: PixelBudget) {
        self.budget = budget;
    }

    pub fn encode_text(&self, s: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?
            .get_ids()
            .to_vec())
    }

    pub fn prepare(&self, img: &RgbImage) -> Result<PreparedImage> {
        prepare(img, self.budget)
    }

    pub fn recognize(
        &self,
        img: &RgbImage,
        mode: DotsMode,
        opts: &GenerateOptions,
    ) -> Result<DotsPageResult> {
        self.recognize_with_budget(img, mode, opts, self.budget)
    }

    pub fn recognize_with_budget(
        &self,
        img: &RgbImage,
        mode: DotsMode,
        opts: &GenerateOptions,
        budget: PixelBudget,
    ) -> Result<DotsPageResult> {
        let t0 = std::time::Instant::now();
        let prep = prepare(img, budget)?;
        let prepare_s = t0.elapsed().as_secs_f64();
        self.recognize_prepared(&prep, mode, opts, None, prepare_s)
    }

    fn stage_precomputed_vision_rows(
        &self,
        merged: Vec<f32>,
        prep: &PreparedImage,
    ) -> Result<Tensor> {
        let rows = prep.num_vision_tokens();
        anyhow::ensure!(
            rows > 0 && merged.len().is_multiple_of(rows),
            "precomputed vision rows hold {} values across {rows} tokens",
            merged.len()
        );
        let cols = merged.len() / rows;
        #[cfg(feature = "wgpu")]
        if let VisionTower::Wgpu(t) = &self.vision {
            anyhow::ensure!(
                cols == t.config().hidden_size,
                "precomputed vision rows are {cols} wide, tower hidden is {}",
                t.config().hidden_size
            );
        }
        let dtype = dots_weight_dtype_bf16_on_every_device_whose_gemm_takes_it_which_is_cuda_and_metal_f32_only_on_cpu(&self.device);
        Ok(Tensor::from_vec(merged, (rows, cols), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(&self.device)?)
    }

    fn recognize_prepared(
        &self,
        prep: &PreparedImage,
        mode: DotsMode,
        opts: &GenerateOptions,
        pre_rows: Option<PreRows>,
        prepare_s: f64,
    ) -> Result<DotsPageResult> {
        let _ = SERIAL_CROP_ENCODES_BATCH_BY_GRID_BECAUSE_TOWER_COST_IS_PER_CALL_NOT_PER_PATCH;
        #[cfg(feature = "wgpu")]
        let profile = super::decoder_wgpu_graph::page_profile_enabled();
        #[cfg(not(feature = "wgpu"))]
        let profile = false;
        let t0 = std::time::Instant::now();
        let t1 = std::time::Instant::now();
        #[cfg(feature = "wgpu")]
        let (pre_gpu, pre_host) = match pre_rows {
            Some(PreRows::Gpu(g)) => (Some(g), None),
            Some(PreRows::Host(v)) => (None, Some(v)),
            None => (None, None),
        };
        #[cfg(not(feature = "wgpu"))]
        let pre_host = pre_rows;
        #[cfg(feature = "wgpu")]
        let gpu_rows = match pre_gpu {
            Some(g) => {
                anyhow::ensure!(
                    self.graph.is_some() && self.q3d.is_none(),
                    "pre-packed gpu vision rows engage only when the graph serves alone"
                );
                Some(g)
            }
            None if pre_host.is_none()
                && self.graph.is_some()
                && self.q3d.is_none()
                && vision_handoff_enabled() =>
            {
                match &self.vision {
                    VisionTower::Wgpu(t) => t.encode_resident(prep)?,
                    _ => None,
                }
            }
            None => None,
        };
        #[cfg(feature = "wgpu")]
        let gpu_handoff = gpu_rows.is_some();
        #[cfg(not(feature = "wgpu"))]
        let gpu_handoff = false;
        if gpu_handoff {
            self.gpu_handoff_crops
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let feats = if gpu_handoff {
            None
        } else {
            Some(match pre_host {
                Some(rows) => self.stage_precomputed_vision_rows(rows, prep)?,
                None => self.encode_vision(prep)?,
            })
        };
        let vision_s = t1.elapsed().as_secs_f64();
        let n_vis = prep.num_vision_tokens();
        #[cfg(feature = "wgpu")]
        if let Some(gv) = &gpu_rows {
            anyhow::ensure!(
                gv.rows == n_vis,
                "resident vision produced {} tokens, expected {n_vis}",
                gv.rows
            );
        }
        if let Some(feats) = &feats {
            anyhow::ensure!(
                feats.dim(0)? == n_vis,
                "vision produced {} tokens, expected {n_vis}",
                feats.dim(0)?
            );
        }
        let t2 = std::time::Instant::now();
        let (tokens, vision_start) =
            build_prompt_tokens(|s| self.encode_text(s), mode.prompt(), n_vis, self.style)?;
        let prompt_s = t2.elapsed().as_secs_f64();
        let t3 = std::time::Instant::now();
        #[cfg(feature = "wgpu")]
        let outcome = if let Some(gv) = &gpu_rows {
            let mut graph = self
                .graph
                .as_ref()
                .expect("the resident handoff engages only with a live graph")
                .lock()
                .map_err(|e| anyhow::anyhow!("dots.ocr graph decoder lock: {e}"))?;
            graph.generate_gpu_rows(&tokens, vision_start, gv, opts)?
        } else {
            let feats = feats.as_ref().expect("host handoff staged features");
            match self.q3d_generate(&tokens, vision_start, feats, opts)? {
                Some(o) => o,
                None => match self.graph_generate(&tokens, vision_start, feats, opts)? {
                    Some(o) => o,
                    None => self
                        .decoder
                        .as_ref()
                        .context(
                            "dots.ocr: no wgpu-engine outcome and no candle decoder; the candle \
                             decoder is built only when neither wgpu engine is serving",
                        )?
                        .generate(&tokens, vision_start, Some(feats), opts)?,
                },
            }
        };
        #[cfg(not(feature = "wgpu"))]
        let outcome = {
            let feats = feats.as_ref().expect("host handoff staged features");
            match self.q3d_generate(&tokens, vision_start, feats, opts)? {
                Some(o) => o,
                None => match self.graph_generate(&tokens, vision_start, feats, opts)? {
                    Some(o) => o,
                    None => self
                        .decoder
                        .as_ref()
                        .context(
                            "dots.ocr: no wgpu-engine outcome and no candle decoder; the candle \
                             decoder is built only when neither wgpu engine is serving",
                        )?
                        .generate(&tokens, vision_start, Some(feats), opts)?,
                },
            }
        };
        let generate_s = t3.elapsed().as_secs_f64();
        let t4 = std::time::Instant::now();
        let result = self.page_result_from_outcome(prep, mode, tokens.len(), outcome)?;
        if profile {
            let post_s = t4.elapsed().as_secs_f64();
            let wall = prepare_s + t0.elapsed().as_secs_f64();
            eprintln!(
                "DOTS-PAGE-PROFILE wall_s={wall:.3} = prepare {prepare_s:.3} + vision \
                 {vision_s:.3} + prompt {prompt_s:.3} + generate {generate_s:.3} + post \
                 {post_s:.3}  (residual {:.3})",
                wall - prepare_s - vision_s - prompt_s - generate_s - post_s
            );
        }
        Ok(result)
    }

    fn page_result_from_outcome(
        &self,
        prep: &PreparedImage,
        mode: DotsMode,
        prompt_len: usize,
        outcome: GenerateOutcome,
    ) -> Result<DotsPageResult> {
        let mut out_tokens = outcome.tokens;
        let looped = outcome.loop_detection.is_some();
        if let Some(d) = &outcome.loop_detection {
            if d.period > 0 || !mode.emits_json() {
                out_tokens.truncate(d.onset);
            }
        }
        let raw = self
            .tokenizer
            .decode(&out_tokens, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;
        let mut page = if mode.emits_json() {
            parse_layout_json(&raw)
        } else {
            plain_text_fallback(&raw)
        };
        page.truncated |= !outcome.hit_eos;
        page.rescale(
            prep.orig_w as f32 / prep.resized_w as f32,
            prep.orig_h as f32 / prep.resized_h as f32,
        );
        let text = page.to_plain_text();
        Ok(DotsPageResult {
            page,
            text,
            raw,
            prompt_tokens: prompt_len,
            generated_tokens: out_tokens.len(),
            looped,
            hit_eos: outcome.hit_eos,
            grid: (prep.grid_h, prep.grid_w),
        })
    }

    pub const PUMPED_SERVING_IS_THE_GRAPH_PLUS_RESIDENT_TOWER_ARM_ALONE_BECAUSE_ONLY_ITS_READBACKS_ARE_ASYNC:
        &'static str =
        "recognize_pumped awaits every GPU->CPU readback (the browser event loop cannot host a \
         blocking wait), and the only readbacks that have pumped twins are the graph decoder's \
         token/logit fetches and the resident vision-row splice; the host-merger tower arm, the \
         q3d engine and the candle decoder read back inside code with no async twin, so this \
         entry refuses them instead of deadlocking a browser tab";

    #[cfg(feature = "wgpu")]
    pub async fn recognize_pumped(
        &mut self,
        img: &RgbImage,
        mode: DotsMode,
        opts: &GenerateOptions,
    ) -> Result<DotsPageResult> {
        let _ = Self::PUMPED_SERVING_IS_THE_GRAPH_PLUS_RESIDENT_TOWER_ARM_ALONE_BECAUSE_ONLY_ITS_READBACKS_ARE_ASYNC;
        anyhow::ensure!(
            self.graph.is_some() && self.q3d.is_none(),
            "recognize_pumped serves only the wgpu graph decoder"
        );
        let prep = prepare(img, self.budget)?;
        let gpu_rows = self.encode_resident_rows(&prep)?;
        let Some(gv) = gpu_rows else {
            anyhow::bail!(
                "the tower produced no resident vision rows (host-merger arm or an ablation \
                 env); {}",
                Self::PUMPED_SERVING_IS_THE_GRAPH_PLUS_RESIDENT_TOWER_ARM_ALONE_BECAUSE_ONLY_ITS_READBACKS_ARE_ASYNC
            )
        };
        let n_vis = prep.num_vision_tokens();
        anyhow::ensure!(
            gv.rows == n_vis,
            "resident vision produced {} tokens, expected {n_vis}",
            gv.rows
        );
        self.gpu_handoff_crops
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (tokens, vision_start) =
            build_prompt_tokens(|s| self.encode_text(s), mode.prompt(), n_vis, self.style)?;
        let outcome = {
            let graph = self
                .graph
                .as_mut()
                .expect("checked above")
                .get_mut()
                .map_err(|e| anyhow::anyhow!("dots.ocr graph decoder lock: {e}"))?;
            graph
                .generate_gpu_rows_pumped(&tokens, vision_start, &gv, opts)
                .await?
        };
        self.page_result_from_outcome(&prep, mode, tokens.len(), outcome)
    }

    #[cfg(feature = "wgpu")]
    pub fn encode_resident_rows(
        &self,
        prep: &PreparedImage,
    ) -> Result<Option<super::decoder_wgpu_graph::VisionRowsGpu>> {
        match &self.vision {
            VisionTower::Wgpu(t) => t.encode_resident(prep),
            _ => Ok(None),
        }
    }

    pub fn recognize_auto(
        &self,
        img: &RgbImage,
        mode: DotsMode,
        opts: &GenerateOptions,
    ) -> Result<DotsPageResult> {
        self.recognize_auto_with_policy(img, mode, opts, None, false)
    }

    pub const THE_TIERING_HINT_IS_PER_REQUEST_AND_OUTRANKS_THE_TIERED_ENV_BUT_NOT_THE_DETOUR_KILL_SWITCH:
        &'static str =
        "an integrator who names its input class knows more than a machine-wide default, so an \
         explicit per-request policy beats NV_DOTS_TIERED (a caller sending off or force has \
         stated intent for this one page); the hint never reroutes LayoutOnly, whose boxes-only \
         contract already refuses tiling, and force_screen_detour still answers only PlainOcr \
         because the detour composes plain text. NV_DOTS_SCREEN_DETOUR=0 remains the operator \
         kill-switch over the detour composition, hint or no hint -- an operator disabling a \
         path outranks a caller requesting it. An absent hint is recognize_auto exactly.";

    pub fn recognize_auto_with_policy(
        &self,
        img: &RgbImage,
        mode: DotsMode,
        opts: &GenerateOptions,
        policy_override: Option<super::tiered::TierPolicy>,
        force_screen_detour: bool,
    ) -> Result<DotsPageResult> {
        let _ = LAYOUT_ONLY_STAYS_SINGLE_PASS_BECAUSE_A_BOXES_ONLY_CALLER_MUST_NOT_PAY_CROP_OCR;
        let _ = Self::THE_TIERING_HINT_IS_PER_REQUEST_AND_OUTRANKS_THE_TIERED_ENV_BUT_NOT_THE_DETOUR_KILL_SWITCH;
        let inverted_dark_page;
        let img = if mode == DotsMode::PlainOcr
            && std::env::var(DARK_INVERT_ENV).ok().as_deref() != Some("0")
            && super::tiered::page_is_dark_themed(img)
        {
            let _ =
                DARK_PAGES_INVERT_BEFORE_THE_TOWER_BECAUSE_THE_MODEL_WAS_TRAINED_ON_INK_OVER_LIGHT;
            inverted_dark_page = RgbImage {
                w: img.w,
                h: img.h,
                data: img.data.iter().map(|b| 255 - b).collect(),
            };
            &inverted_dark_page
        } else {
            img
        };
        let policy = match policy_override {
            Some(p) => p,
            None => super::tiered::TierPolicy::from_env()?,
        };
        if mode != DotsMode::LayoutOnly
            && super::tiered::engages(img.w, img.h, self.budget, policy)?
        {
            if mode == DotsMode::PlainOcr {
                if let Some(res) = self.recognize_tiled_classical(img, opts)? {
                    return Ok(res);
                }
            }
            return self.recognize_cropped(img, opts);
        }
        let _ = super::tiered::SUB_THRESHOLD_SCREENS_DETOUR_TO_TILING_BECAUSE_NATIVE_READING_DROPS_A_THIRD_OF_THE_UI;
        if mode == DotsMode::PlainOcr
            && std::env::var(SCREEN_DETOUR_ENV).ok().as_deref() != Some("0")
            && (force_screen_detour
                || (policy == super::tiered::TierPolicy::Auto
                    && super::tiered::classify_sub_threshold(img)
                        == super::tiered::SubThresholdClass::ScreenLike))
        {
            if let Some(res) = self.recognize_tiled_classical(img, opts)? {
                return Ok(res);
            }
            return self.recognize_cropped(img, opts);
        }
        self.recognize(img, mode, opts)
    }

    #[cfg(feature = "wgpu")]
    fn same_grid_batched_rows(
        &self,
        preps: &[PreparedImage],
        resident: bool,
    ) -> Result<Vec<Option<PreRows>>> {
        let mut rows: Vec<Option<PreRows>> = (0..preps.len()).map(|_| None).collect();
        let VisionTower::Wgpu(t) = &self.vision else {
            return Ok(rows);
        };
        let mut buckets: std::collections::HashMap<(usize, usize), Vec<usize>> =
            std::collections::HashMap::new();
        for (i, p) in preps.iter().enumerate() {
            buckets.entry((p.grid_h, p.grid_w)).or_default().push(i);
        }
        let profile = super::decoder_wgpu_graph::page_profile_enabled();
        let mut census: Vec<String> = Vec::new();
        if resident {
            if let Some(crops) = t.encode_pack_resident(preps)? {
                for (i, crop) in crops.into_iter().enumerate() {
                    rows[i] = Some(PreRows::Gpu(crop));
                }
                if profile {
                    let mut b: Vec<_> = buckets
                        .iter()
                        .map(|((gh, gw), idxs)| (gh * gw, idxs.len()))
                        .collect();
                    b.sort_unstable();
                    eprintln!(
                        "DOTS-VISENC-PACKED crops={} rows_total={} buckets={b:?}",
                        preps.len(),
                        preps.iter().map(PreparedImage::num_patches).sum::<usize>(),
                    );
                }
                return Ok(rows);
            }
        }
        for idxs in buckets.values() {
            if idxs.len() < 2 {
                continue;
            }
            let refs: Vec<&PreparedImage> = idxs.iter().map(|&i| &preps[i]).collect();
            let t_bucket = std::time::Instant::now();
            if resident {
                if let Some(crops) = t.encode_batch_resident(&refs)? {
                    for (crop, &i) in crops.into_iter().zip(idxs.iter()) {
                        rows[i] = Some(PreRows::Gpu(crop));
                    }
                    if profile {
                        let p = &preps[idxs[0]];
                        census.push(format!(
                            "{}x{}:b{}:n{}:{:.4}s",
                            p.grid_h,
                            p.grid_w,
                            idxs.len(),
                            p.num_patches(),
                            t_bucket.elapsed().as_secs_f64(),
                        ));
                    }
                    continue;
                }
            }
            if let Some(batched) = t.encode_batch(&refs)? {
                for (k, &i) in idxs.iter().enumerate() {
                    rows[i] = Some(PreRows::Host(batched[k].clone()));
                }
                if profile {
                    let p = &preps[idxs[0]];
                    census.push(format!(
                        "{}x{}:b{}:n{}:host:{:.4}s",
                        p.grid_h,
                        p.grid_w,
                        idxs.len(),
                        p.num_patches(),
                        t_bucket.elapsed().as_secs_f64(),
                    ));
                }
            }
        }
        if profile {
            let misses: Vec<usize> = buckets
                .values()
                .filter(|v| v.len() < 2)
                .map(|v| preps[v[0]].num_patches())
                .collect();
            census.sort();
            eprintln!(
                "DOTS-VISENC-BUCKETS crops={} rows_total={} forwards={} miss_forwards={} \
                 miss_rows={} buckets=[{}]",
                preps.len(),
                preps.iter().map(PreparedImage::num_patches).sum::<usize>(),
                census.len(),
                misses.len(),
                misses.iter().sum::<usize>(),
                census.join(" "),
            );
        }
        Ok(rows)
    }

    #[cfg(feature = "wgpu")]
    fn ocr_regions_batched(
        &self,
        img: &RgbImage,
        regions: &[super::regions::Region],
        opts: &GenerateOptions,
    ) -> Result<Option<RegionOcr>> {
        let _ = BATCHED_T1_RIDES_THE_GRAPHS_REGION_MACHINERY_AND_ESCAPES_TO_SERIAL_CROPS;
        let _ =
            T1_BATCH_MIN_REGIONS_24_A_STEP_COSTS_THE_FULL_TRUNK_SO_SPARSE_BATCHES_LOSE_TO_SERIAL;
        if std::env::var(T1_BATCH_ENV).ok().as_deref() == Some("0")
            || regions.len()
                < T1_BATCH_MIN_REGIONS_24_A_STEP_COSTS_THE_FULL_TRUNK_SO_SPARSE_BATCHES_LOSE_TO_SERIAL
        {
            return Ok(None);
        }
        let Some(graph) = &self.graph else {
            return Ok(None);
        };
        let mut graph = graph
            .lock()
            .map_err(|e| anyhow::anyhow!("dots.ocr graph decoder lock: {e}"))?;
        let Some((slots, cap)) = graph.region_capacity() else {
            return Ok(None);
        };
        let profile = super::decoder_wgpu_graph::page_profile_enabled();
        let t_page = std::time::Instant::now();
        let crop_max = env_pixels(
            CROP_MAX_PIXELS_ENV,
            DEFAULT_CROP_MAX_PIXELS_500K_IS_ABOUT_640_VISION_TOKENS,
        );
        let img_px = img.w * img.h;
        struct RegionJob {
            tokens: Vec<u32>,
            vision_start: usize,
            rows: PreRows,
            max_new: usize,
            generated: Vec<u32>,
            hit_eos: bool,
            banned_stop: bool,
            done: bool,
        }
        let mut preps = Vec::with_capacity(regions.len());
        for region in regions {
            let (x0, y0, x1, y1) = region.pixel_bounds(img.w, img.h);
            let crop = img.crop(x0, y0, x1, y1);
            let crop_px = (x1 - x0) * (y1 - y0);
            let dpi_floor = if img_px > self.budget.max_pixels {
                crop_px * self.budget.max_pixels / img_px
            } else {
                crop_px
            };
            let crop_budget = PixelBudget {
                min_pixels: DEFAULT_MIN_PIXELS,
                max_pixels: crop_max.max(dpi_floor),
            };
            preps.push(prepare(&crop, crop_budget)?);
        }
        let _ = T1_BATCH_MAX_PROMPT_KEEPS_LONG_PROMPT_PAGES_ON_STEEL_PREFILL;
        if preps
            .iter()
            .any(|p| p.num_vision_tokens() > T1_BATCH_MAX_CROP_VISION_TOKENS_896_IS_THE_STRIDE_BOUND_NOW_THAT_RANGED_PREFILL_IS_STEEL)
        {
            return Ok(None);
        }
        let resident = region_handoff_enabled();
        let t_enc = std::time::Instant::now();
        let vision_rows = self.same_grid_batched_rows(&preps, resident)?;
        if profile {
            graph.drain();
        }
        let enc_batch_s = t_enc.elapsed().as_secs_f64();
        let batched_crops = vision_rows.iter().filter(|r| r.is_some()).count();
        let mut enc_fallback_s = 0.0f64;
        let mut rows_f32_total = 0usize;
        let mut jobs: Vec<RegionJob> = Vec::with_capacity(regions.len());
        for ((region, prep), pre_rows) in regions.iter().zip(&preps).zip(vision_rows) {
            let rows: PreRows = match pre_rows {
                Some(r) => r,
                None => {
                    let t_fb = std::time::Instant::now();
                    let mut got: Option<PreRows> = None;
                    if resident {
                        if let VisionTower::Wgpu(t) = &self.vision {
                            got = t.encode_resident(prep)?.map(PreRows::Gpu);
                        }
                    }
                    let r = match got {
                        Some(g) => g,
                        None => PreRows::Host(
                            self.encode_vision(prep)?
                                .to_dtype(candle_core::DType::F32)?
                                .flatten_all()?
                                .to_vec1()?,
                        ),
                    };
                    enc_fallback_s += t_fb.elapsed().as_secs_f64();
                    r
                }
            };
            if let PreRows::Host(r) = &rows {
                rows_f32_total += r.len();
            }
            let n_vis = prep.num_vision_tokens();
            if let PreRows::Gpu(g) = &rows {
                anyhow::ensure!(
                    g.rows == n_vis,
                    "resident crop encode produced {} tokens, expected {n_vis}",
                    g.rows
                );
                self.gpu_handoff_crops
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let (tokens, vision_start) =
                build_prompt_tokens(|s| self.encode_text(s), PROMPT_OCR, n_vis, self.style)?;
            let max_new = if heading_like(region.category.as_deref()) {
                opts.max_new_tokens.min(UNMERGED_HEADING_CROP_TOKEN_CAP)
            } else {
                opts.max_new_tokens
            };
            if tokens.len() + max_new.min(64) >= cap {
                return Ok(None);
            }
            jobs.push(RegionJob {
                tokens,
                vision_start,
                rows,
                max_new,
                generated: Vec::new(),
                hit_eos: false,
                banned_stop: false,
                done: false,
            });
        }
        graph.reset()?;
        let stride = cap + 1;
        let mut active: Vec<(usize, super::decoder_wgpu_graph::RegionRow)> = Vec::new();
        let mut next_job = 0usize;
        let fill = |graph: &mut super::decoder_wgpu_graph::DotsDecoderWgpuGraph,
                    active: &mut Vec<(usize, super::decoder_wgpu_graph::RegionRow)>,
                    next_job: &mut usize,
                    jobs: &[RegionJob],
                    pf_s: &mut f64,
                    pf_drain_s: &mut f64|
         -> Result<()> {
            let mut used: Vec<bool> = vec![false; slots];
            for (_, r) in active.iter() {
                used[r.base as usize / stride] = true;
            }
            let before = active.len();
            let free: Vec<usize> = (0..slots).filter(|s| !used[*s]).collect();
            let jobs_left = jobs.len() - *next_job;
            let defer = super::decoder_wgpu_graph::rpf_defer_rows();
            if defer > 0 && before > 0 && jobs_left > free.len() && free.len() < defer {
                return Ok(());
            }
            let picked: Vec<(usize, usize)> = free
                .iter()
                .take(jobs_left)
                .map(|slot| {
                    let ji = *next_job;
                    *next_job += 1;
                    (*slot, ji)
                })
                .collect();
            if picked.is_empty() {
                return Ok(());
            }
            let pack_jobs: Vec<super::decoder_wgpu_graph::RegionPackJob> = picked
                .iter()
                .map(|(slot, ji)| {
                    let j = &jobs[*ji];
                    super::decoder_wgpu_graph::RegionPackJob {
                        slot: *slot,
                        prompt_tokens: &j.tokens,
                        vision_start: j.vision_start,
                        src: match &j.rows {
                            PreRows::Host(r) => {
                                super::decoder_wgpu_graph::RegionRowsSrc::Host(r.as_slice())
                            }
                            PreRows::Gpu(g) => super::decoder_wgpu_graph::RegionRowsSrc::Gpu(g),
                        },
                    }
                })
                .collect();
            let t_pf = std::time::Instant::now();
            let lens = graph.region_prefill_pack(&pack_jobs)?;
            *pf_s += t_pf.elapsed().as_secs_f64();
            for ((slot, ji), len) in picked.iter().zip(lens) {
                active.push((
                    *ji,
                    super::decoder_wgpu_graph::RegionRow {
                        token: *jobs[*ji].tokens.last().expect("prompt is non-empty"),
                        base: (slot * stride) as u32,
                        len: len as u32,
                    },
                ));
            }
            if profile && active.len() > before {
                eprintln!(
                    "DOTS-RPF-BURST filled={} jobs_left={}",
                    active.len() - before,
                    jobs.len() - *next_job
                );
                let t_drain = std::time::Instant::now();
                graph.drain();
                *pf_drain_s += t_drain.elapsed().as_secs_f64();
            }
            Ok(())
        };
        let mut pf_s = 0.0f64;
        let mut pf_drain_s = 0.0f64;
        fill(
            &mut graph,
            &mut active,
            &mut next_job,
            &jobs,
            &mut pf_s,
            &mut pf_drain_s,
        )?;
        let handoff_max = t1_handoff_max_live_rows();
        let t_bstep = std::time::Instant::now();
        let mut bsteps = 0u64;
        let mut live_curve: Vec<u32> = Vec::new();
        while !active.is_empty() {
            bsteps += 1;
            live_curve.push(active.len() as u32);
            let rows: Vec<super::decoder_wgpu_graph::RegionRow> =
                active.iter().map(|(_, r)| *r).collect();
            let toks = graph.bstep_tokens(&rows)?;
            let mut retired = false;
            for (i, tok) in toks.iter().enumerate() {
                let (ji, row) = &mut active[i];
                let job = &mut jobs[*ji];
                if EOS_TOKEN_IDS.contains(tok) {
                    job.hit_eos = true;
                    job.done = true;
                } else {
                    job.generated.push(*tok);
                    row.len += 1;
                    row.token = *tok;
                    if job.generated.len() >= job.max_new || (row.len as usize + 1) >= cap {
                        job.done = true;
                    } else if let Some(n) = opts.ngram_size.filter(|n| *n > 0) {
                        let banned = crate::deepseek_ocr::decoder::banned_tokens_windowed_ngram(
                            &job.generated,
                            n,
                            opts.ngram_window,
                            &opts.ngram_whitelist,
                        );
                        if !banned.is_empty() {
                            job.banned_stop = true;
                            job.done = true;
                        }
                    }
                }
                retired |= job.done;
            }
            if retired {
                active.retain(|(ji, _)| !jobs[*ji].done);
                fill(
                    &mut graph,
                    &mut active,
                    &mut next_job,
                    &jobs,
                    &mut pf_s,
                    &mut pf_drain_s,
                )?;
                if next_job >= jobs.len() && !active.is_empty() && active.len() <= handoff_max {
                    let _ = A_THIN_BATCH_TAIL_KEEPS_BATCHING_BECAUSE_THE_GAZETTE_LADDER_MEASURED_EVERY_HANDOFF_THRESHOLD_LOSING;
                    for (ji, _) in active.drain(..) {
                        let job = &mut jobs[ji];
                        job.done = true;
                        let rem = job.max_new.saturating_sub(job.generated.len());
                        if rem == 0 {
                            continue;
                        }
                        let mut resume = job.tokens.clone();
                        resume.extend_from_slice(&job.generated);
                        let hopts = GenerateOptions {
                            max_new_tokens: rem,
                            ..opts.clone()
                        };
                        let outcome = match &job.rows {
                            PreRows::Host(r) => {
                                graph.generate(&resume, job.vision_start, Some(r), &hopts)?
                            }
                            PreRows::Gpu(g) => {
                                graph.generate_gpu_rows(&resume, job.vision_start, g, &hopts)?
                            }
                        };
                        job.generated.extend_from_slice(&outcome.tokens);
                        job.hit_eos = outcome.hit_eos;
                    }
                }
            }
        }
        if profile {
            let (bstep_cpu_s, bstep_read_s, bstep_n) = graph.take_bstep_split();
            let sum_live: u64 = live_curve.iter().map(|l| *l as u64).sum();
            let full = live_curve.iter().filter(|l| **l as usize == slots).count();
            let le16 = live_curve.iter().filter(|l| **l <= 16).count();
            let le8 = live_curve.iter().filter(|l| **l <= 8).count();
            eprintln!(
                "DOTS-BSTEP-OCC bsteps={bsteps} slots={slots} sum_live={sum_live} \
                 occ_mean={:.3} steps_full={full} steps_le16={le16} steps_le8={le8} \
                 bstep_cpu_s={bstep_cpu_s:.3} bstep_read_s={bstep_read_s:.3} bstep_n={bstep_n} \
                 pf_drain_s={pf_drain_s:.3} live_curve={live_curve:?}",
                sum_live as f64 / (bsteps.max(1) as f64 * slots as f64),
            );
            let rpf_repack_s = graph.take_region_repack_s();
            let resident_crops = jobs
                .iter()
                .filter(|j| matches!(j.rows, PreRows::Gpu(_)))
                .count();
            eprintln!(
                "DOTS-REGION-PROFILE regions={} batched_enc={batched_crops} fallback_enc={} \
                 resident_crops={resident_crops} \
                 enc_batch_s={enc_batch_s:.3} enc_fallback_s={enc_fallback_s:.3} \
                 host_rows_mb={:.1} pf_s={pf_s:.3} (of which host repack {rpf_repack_s:.3}) \
                 bstep_s={:.3} bsteps={bsteps} page_s={:.3}",
                jobs.len(),
                jobs.len() - batched_crops,
                rows_f32_total as f64 * 4.0 / 1e6,
                t_bstep.elapsed().as_secs_f64(),
                t_page.elapsed().as_secs_f64(),
            );
        }
        drop(graph);
        let mut out = RegionOcr {
            elements: Vec::with_capacity(regions.len()),
            prompt_tokens: 0,
            generated_tokens: 0,
            looped: false,
            hit_eos: true,
        };
        for (region, job) in regions.iter().zip(jobs.iter_mut()) {
            if job.banned_stop {
                let _ = A_BANNED_ROW_REDOES_SERIALLY_BECAUSE_THE_BATCH_KEEPS_NO_PER_ROW_LOGITS_TO_REARGUE;
                let (x0, y0, x1, y1) = region.pixel_bounds(img.w, img.h);
                let crop = img.crop(x0, y0, x1, y1);
                let crop_px = (x1 - x0) * (y1 - y0);
                let dpi_floor = if img_px > self.budget.max_pixels {
                    crop_px * self.budget.max_pixels / img_px
                } else {
                    crop_px
                };
                let crop_budget = PixelBudget {
                    min_pixels: DEFAULT_MIN_PIXELS,
                    max_pixels: crop_max.max(dpi_floor),
                };
                let crop_opts = GenerateOptions {
                    max_new_tokens: job.max_new,
                    ..opts.clone()
                };
                let res =
                    self.recognize_with_budget(&crop, DotsMode::PlainOcr, &crop_opts, crop_budget)?;
                out.prompt_tokens += res.prompt_tokens;
                out.generated_tokens += res.generated_tokens;
                out.looped |= res.looped;
                out.hit_eos &= res.hit_eos;
                let body = res.text.trim();
                let text = format!(
                    "{}{}",
                    category_prefix(region.category.as_deref(), body),
                    body
                );
                out.elements.push(LayoutElement {
                    bbox: Some(region.bbox),
                    category: region.category.clone(),
                    text: Some(text),
                });
                continue;
            }
            let mut toks = std::mem::take(&mut job.generated);
            let detection = crate::deepseek_ocr::decoder::detect_loop(&toks);
            let looped = detection.is_some() || job.banned_stop;
            if let Some(d) = &detection {
                toks.truncate(d.onset);
            }
            out.prompt_tokens += job.tokens.len();
            out.generated_tokens += toks.len();
            out.looped |= looped;
            out.hit_eos &= job.hit_eos;
            let raw = self
                .tokenizer
                .decode(&toks, true)
                .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;
            let body = raw.trim();
            let text = format!(
                "{}{}",
                category_prefix(region.category.as_deref(), body),
                body
            );
            out.elements.push(LayoutElement {
                bbox: Some(region.bbox),
                category: region.category.clone(),
                text: Some(text),
            });
        }
        Ok(Some(out))
    }

    fn ocr_regions(
        &self,
        img: &RgbImage,
        regions: &[super::regions::Region],
        opts: &GenerateOptions,
    ) -> Result<RegionOcr> {
        #[cfg(feature = "wgpu")]
        if let Some(batched) = self.ocr_regions_batched(img, regions, opts)? {
            return Ok(batched);
        }
        let crop_max = env_pixels(
            CROP_MAX_PIXELS_ENV,
            DEFAULT_CROP_MAX_PIXELS_500K_IS_ABOUT_640_VISION_TOKENS,
        );
        let _ = CROP_PACKING_IS_A_MEASURED_NEGATIVE_UNION_BOXES_SHIP_WHITESPACE_THROUGH_THE_TOWER;
        let img_px = img.w * img.h;
        let mut out = RegionOcr {
            elements: Vec::with_capacity(regions.len()),
            prompt_tokens: 0,
            generated_tokens: 0,
            looped: false,
            hit_eos: true,
        };
        let mut preps = Vec::with_capacity(regions.len());
        let mut prep_secs = Vec::with_capacity(regions.len());
        for region in regions {
            let (x0, y0, x1, y1) = region.pixel_bounds(img.w, img.h);
            let crop = img.crop(x0, y0, x1, y1);
            let crop_px = (x1 - x0) * (y1 - y0);
            let dpi_floor_matching_a_whole_page_pass_at_the_serving_budget =
                if img_px > self.budget.max_pixels {
                    crop_px * self.budget.max_pixels / img_px
                } else {
                    crop_px
                };
            let crop_budget = PixelBudget {
                min_pixels: DEFAULT_MIN_PIXELS,
                max_pixels: crop_max
                    .max(dpi_floor_matching_a_whole_page_pass_at_the_serving_budget),
            };
            let t = std::time::Instant::now();
            preps.push(prepare(&crop, crop_budget)?);
            prep_secs.push(t.elapsed().as_secs_f64());
        }
        let _ = SERIAL_CROP_ENCODES_BATCH_BY_GRID_BECAUSE_TOWER_COST_IS_PER_CALL_NOT_PER_PATCH;
        #[cfg(feature = "wgpu")]
        let mut pre_rows = if std::env::var(T1_BATCH_ENV).ok().as_deref() == Some("0") {
            (0..preps.len())
                .map(|_| None)
                .collect::<Vec<Option<PreRows>>>()
        } else {
            let resident = self.graph.is_some() && self.q3d.is_none() && region_handoff_enabled();
            self.same_grid_batched_rows(&preps, resident)?
        };
        #[cfg(not(feature = "wgpu"))]
        let mut pre_rows: Vec<Option<PreRows>> = (0..preps.len()).map(|_| None).collect();
        for (i, region) in regions.iter().enumerate() {
            let crop_opts = if heading_like(region.category.as_deref()) {
                GenerateOptions {
                    max_new_tokens: opts.max_new_tokens.min(UNMERGED_HEADING_CROP_TOKEN_CAP),
                    ..opts.clone()
                }
            } else {
                opts.clone()
            };
            let res = self.recognize_prepared(
                &preps[i],
                DotsMode::PlainOcr,
                &crop_opts,
                pre_rows[i].take(),
                prep_secs[i],
            )?;
            out.prompt_tokens += res.prompt_tokens;
            out.generated_tokens += res.generated_tokens;
            out.looped |= res.looped;
            out.hit_eos &= res.hit_eos;
            let body = res.text.trim();
            let text = format!(
                "{}{}",
                category_prefix(region.category.as_deref(), body),
                body
            );
            out.elements.push(LayoutElement {
                bbox: Some(region.bbox),
                category: region.category.clone(),
                text: Some(text),
            });
        }
        Ok(out)
    }

    pub fn recognize_tiled_classical(
        &self,
        img: &RgbImage,
        opts: &GenerateOptions,
    ) -> Result<Option<DotsPageResult>> {
        #[cfg(feature = "wgpu")]
        let t_t0 = std::time::Instant::now();
        let Some(scouted) = super::tiered::classical_scout(img) else {
            return Ok(None);
        };
        let regions = plan_regions(&scouted, img.w, img.h, false);
        if regions.is_empty() {
            return Ok(None);
        }
        #[cfg(feature = "wgpu")]
        let t0_s = t_t0.elapsed().as_secs_f64();
        #[cfg(feature = "wgpu")]
        let t_t1 = std::time::Instant::now();
        let t1 = self.ocr_regions(img, &regions, opts)?;
        #[cfg(feature = "wgpu")]
        if super::decoder_wgpu_graph::page_profile_enabled() {
            eprintln!(
                "DOTS-TIER-PROFILE scout=classical t0_scout_s={t0_s:.3} t1_regions_s={:.3} \
                 regions={} src_px={}x{}",
                t_t1.elapsed().as_secs_f64(),
                regions.len(),
                img.w,
                img.h,
            );
        }
        let page = LayoutPage {
            elements: t1.elements,
            truncated: !t1.hit_eos,
        };
        let text = page.to_plain_text();
        Ok(Some(DotsPageResult {
            page,
            text,
            raw: String::new(),
            prompt_tokens: t1.prompt_tokens,
            generated_tokens: t1.generated_tokens,
            looped: t1.looped,
            hit_eos: t1.hit_eos,
            grid: (0, 0),
        }))
    }

    pub fn recognize_cropped(
        &self,
        img: &RgbImage,
        opts: &GenerateOptions,
    ) -> Result<DotsPageResult> {
        let _ =
            INK_RECALL_RESCUES_SPARSE_SHEETS_THE_PAGE_AREA_FLOOR_WRONGLY_SENDS_BACK_TO_SINGLE_PASS;
        let ink = sample_ink(img);
        let usable_by_ink = |planned: &[super::regions::Region]| {
            ink_recall(&ink, planned).is_some_and(|r| {
                r >= INK_RECALL_USABLE_MIN_085_A_DETECTOR_BOXING_85PCT_OF_THE_INK_SAW_THE_PAGE
            })
        };
        #[cfg(feature = "wgpu")]
        let profile = super::decoder_wgpu_graph::page_profile_enabled();
        #[cfg(not(feature = "wgpu"))]
        let profile = false;
        let t_t0 = std::time::Instant::now();
        let layout_max = env_pixels(
            LAYOUT_MAX_PIXELS_ENV,
            DEFAULT_LAYOUT_MAX_PIXELS_1MP_KEEPS_THE_DETECT_PASS_CHEAP,
        );
        let retry_max =
            LAYOUT_RETRY_TRIPLES_THE_BUDGET_BECAUSE_A_LOOPING_DETECT_PASS_MEANS_UNDERRESOLVED_INK
                * layout_max;
        let downscale_floor =
            img.w * img.h / LAYOUT_KEEPS_AT_LEAST_A_NINTH_OF_SOURCE_PIXELS_OR_HANDWRITING_LOOPS;
        let first_max = downscale_floor.clamp(layout_max, retry_max);
        let mut ladder = vec![first_max];
        if retry_max > first_max {
            ladder.push(retry_max);
        }
        let mut layout = None;
        let mut regions = Vec::new();
        for max_pixels in ladder {
            let budget = PixelBudget {
                min_pixels: DEFAULT_MIN_PIXELS,
                max_pixels,
            };
            let attempt = self.recognize_with_budget(img, DotsMode::LayoutOnly, opts, budget)?;
            let planned = plan_regions(&attempt.page, img.w, img.h, false);
            let usable = !attempt.page.truncated
                && (layout_is_usable(&planned, img.w, img.h) || usable_by_ink(&planned));
            let keep = usable || layout.is_none() || planned.len() > regions.len();
            if keep {
                layout = Some(attempt);
                regions = planned;
            }
            if usable {
                break;
            }
        }
        let layout = layout.expect("both layout attempts recorded a result");
        let t0_s = t_t0.elapsed().as_secs_f64();
        if !(layout_is_usable(&regions, img.w, img.h) || usable_by_ink(&regions)) {
            if profile {
                eprintln!(
                    "DOTS-TIER-PROFILE t0_layout_s={t0_s:.3} regions={} UNUSABLE -> whole-page \
                     plain ocr fallback",
                    regions.len()
                );
            }
            return self.recognize(img, DotsMode::PlainOcr, opts);
        }
        let _ = HEADING_ONLY_CROPS_BAIT_WHITESPACE_ESSAYS_SO_HEADINGS_MERGE_INTO_THE_BLOCK_BELOW;
        let regions = merge_headings_into_following_regions(regions);
        let t_t1 = std::time::Instant::now();
        let t1 = self.ocr_regions(img, &regions, opts)?;
        let prompt_tokens = layout.prompt_tokens + t1.prompt_tokens;
        let generated_tokens = layout.generated_tokens + t1.generated_tokens;
        let looped = layout.looped | t1.looped;
        let hit_eos = t1.hit_eos;
        let t1_s = t_t1.elapsed().as_secs_f64();
        let t_t2 = std::time::Instant::now();
        let page = LayoutPage {
            elements: t1.elements,
            truncated: layout.page.truncated || !hit_eos,
        };
        let text = page.to_plain_text();
        if profile {
            eprintln!(
                "DOTS-TIER-PROFILE t0_layout_s={t0_s:.3} t1_regions_s={t1_s:.3} \
                 t2_stitch_s={:.3} regions={} src_px={}x{}",
                t_t2.elapsed().as_secs_f64(),
                regions.len(),
                img.w,
                img.h,
            );
        }
        Ok(DotsPageResult {
            page,
            text,
            raw: layout.raw,
            prompt_tokens,
            generated_tokens,
            looped,
            hit_eos,
            grid: layout.grid,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_checkpoint_reports_the_directory() {
        let Err(err) = DotsOcrPipeline::load(Path::new("/nonexistent/dots-ocr"), &Device::Cpu)
        else {
            panic!("load unexpectedly succeeded")
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("dots.ocr"),
            "error should name the model: {msg}"
        );
    }

    #[test]
    fn a_ggml_precision_never_defaults_into_the_graph_because_the_graph_would_reject_it() {
        for device in [Device::Cpu] {
            for precision in [DotsPrecision::W4, DotsPrecision::W8] {
                assert_eq!(
                    DotsDecoderEngine::default_for(&device, precision),
                    DotsDecoderEngine::Candle,
                    "{precision:?} has no cpu graph arm, so defaulting the graph on would turn \
                     every load at this precision into an error instead of a slower page"
                );
            }
        }
        if let Ok(metal) = Device::new_metal(0) {
            assert_eq!(
                DotsDecoderEngine::default_for(&metal, DotsPrecision::W8),
                DotsDecoderEngine::Candle,
                "w8 is still ggml-only and must keep candle even where the graph is default"
            );
            for precision in [DotsPrecision::W4, DotsPrecision::Bf16] {
                assert_eq!(
                    DotsDecoderEngine::default_for(&metal, precision),
                    DotsDecoderEngine::Graph,
                    "{precision:?} is graphable and metal is where the graph was measured"
                );
            }
        }
    }

    #[test]
    fn metal_default_precision_is_w4_and_bf16_is_the_one_env_master_numerics_escape() {
        if let Ok(metal) = Device::new_metal(0) {
            assert_eq!(
                DotsPrecision::default_for(&metal),
                if cfg!(feature = "wgpu") {
                    DotsPrecision::W4
                } else {
                    DotsPrecision::Bf16
                },
                "the metal serving default is the affine-w4 graph wherever wgpu can serve it"
            );
            assert_eq!(
                DotsPrecision::parse(Some("bf16"), DotsPrecision::default_for(&metal)).unwrap(),
                DotsPrecision::Bf16,
                "bf16 must stay one env away as the master-numerics arm"
            );
        }
        assert_eq!(
            DotsPrecision::parse(Some("w4"), DotsPrecision::Bf16).unwrap(),
            DotsPrecision::W4
        );
    }

    #[test]
    fn vision_backend_unset_takes_the_device_default_and_zero_or_candle_still_escape_to_the_prior_tower(
    ) {
        for raw in [None, Some("")] {
            for default in [DotsVisionBackend::Candle, DotsVisionBackend::Wgpu] {
                assert_eq!(
                    DotsVisionBackend::parse(raw, default).unwrap(),
                    default,
                    "{raw:?} must defer to the device default, not pin one arm"
                );
            }
        }
        for raw in [Some("0"), Some("candle")] {
            assert_eq!(
                DotsVisionBackend::parse(raw, DotsVisionBackend::Wgpu).unwrap(),
                DotsVisionBackend::Candle,
                "{raw:?} is the escape and must beat a wgpu default"
            );
        }
        assert_eq!(
            DotsVisionBackend::parse(Some("wgpu"), DotsVisionBackend::Candle).unwrap(),
            DotsVisionBackend::Wgpu
        );
        let err = DotsVisionBackend::parse(Some("metal"), DotsVisionBackend::Candle).unwrap_err();
        assert!(
            err.to_string().contains(DOTS_VISION_BACKEND_ENV),
            "rejection must name the env var: {err}"
        );
    }

    #[test]
    fn decoder_engine_unset_takes_the_device_default_and_zero_or_candle_still_escape_to_the_candle_decoder(
    ) {
        for raw in [None, Some("")] {
            for default in [DotsDecoderEngine::Candle, DotsDecoderEngine::Graph] {
                assert_eq!(
                    DotsDecoderEngine::parse(raw, default).unwrap(),
                    default,
                    "{raw:?} must defer to the device default, not pin one arm"
                );
            }
        }
        for raw in [Some("0"), Some("candle")] {
            assert_eq!(
                DotsDecoderEngine::parse(raw, DotsDecoderEngine::Graph).unwrap(),
                DotsDecoderEngine::Candle,
                "{raw:?} is the escape and must beat a graph default"
            );
        }
        assert_eq!(
            DotsDecoderEngine::parse(Some("graph"), DotsDecoderEngine::Candle).unwrap(),
            DotsDecoderEngine::Graph
        );
        let err = DotsDecoderEngine::parse(Some("wgpu"), DotsDecoderEngine::Candle).unwrap_err();
        assert!(
            err.to_string().contains(DOTS_DEC_ENGINE_ENV),
            "rejection must name the env var: {err}"
        );
    }

    #[test]
    fn precision_escape_env_overrides_and_garbage_is_rejected() {
        let default = DotsPrecision::default_for(&Device::Cpu);
        assert_eq!(DotsPrecision::parse(None, default).unwrap(), default);
        assert_eq!(DotsPrecision::parse(Some(""), default).unwrap(), default);
        assert_eq!(
            DotsPrecision::parse(Some("bf16"), DotsPrecision::Nvfp4).unwrap(),
            DotsPrecision::Bf16
        );
        assert_eq!(
            DotsPrecision::parse(Some("nvfp4"), DotsPrecision::Bf16).unwrap(),
            DotsPrecision::Nvfp4
        );
        let err = DotsPrecision::parse(Some("int4"), default).unwrap_err();
        assert!(err.to_string().contains(DOTS_PRECISION_ENV));
    }

    #[test]
    fn cpu_default_precision_is_bf16_nvfp4_kernels_are_cuda_only() {
        assert_eq!(
            DotsPrecision::default_for(&Device::Cpu),
            DotsPrecision::Bf16
        );
    }

    #[test]
    fn weight_dtype_follows_device_bf16_capability_not_a_cuda_only_test_so_metal_stops_paying_f32()
    {
        assert_eq!(
            dots_weight_dtype_bf16_on_every_device_whose_gemm_takes_it_which_is_cuda_and_metal_f32_only_on_cpu(
                &Device::Cpu
            ),
            DType::F32,
            "cpu has no bf16 gemm and must stay f32"
        );
        assert!(
            !Device::Cpu.supports_bf16(),
            "the selection delegates to candle's own capability predicate, which reports \
             Cuda and Metal as bf16-capable and Cpu as not; a cuda-only test silently \
             dropped Metal to f32 and doubled its weight traffic"
        );
        #[cfg(feature = "cuda")]
        if let Ok(cuda) = Device::new_cuda(0) {
            assert_eq!(
                dots_weight_dtype_bf16_on_every_device_whose_gemm_takes_it_which_is_cuda_and_metal_f32_only_on_cpu(
                    &cuda
                ),
                DType::BF16
            );
        }
    }

    #[test]
    fn heading_regions_merge_into_the_block_below_and_lone_headings_survive() {
        use super::super::regions::Region;
        let heading = Region {
            bbox: [100.0, 100.0, 500.0, 160.0],
            category: Some("Section-header".to_string()),
        };
        let body = Region {
            bbox: [90.0, 180.0, 900.0, 400.0],
            category: Some("Text".to_string()),
        };
        let merged = merge_headings_into_following_regions(vec![heading.clone(), body.clone()]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].bbox, [90.0, 100.0, 900.0, 400.0]);
        assert_eq!(merged[0].category, None, "merged crop reads as plain text");

        let far_body = Region {
            bbox: [90.0, 700.0, 900.0, 900.0],
            category: Some("Text".to_string()),
        };
        let kept = merge_headings_into_following_regions(vec![heading.clone(), far_body]);
        assert_eq!(kept.len(), 2, "a distant block must not merge");

        let beside = Region {
            bbox: [600.0, 100.0, 900.0, 160.0],
            category: Some("Text".to_string()),
        };
        let kept = merge_headings_into_following_regions(vec![heading.clone(), beside]);
        assert_eq!(kept.len(), 2, "a block with no x overlap must not merge");

        let lone = merge_headings_into_following_regions(vec![heading]);
        assert_eq!(lone.len(), 1);
        assert!(heading_like(lone[0].category.as_deref()));
        assert!(!heading_like(Some("Text")));
        assert!(!heading_like(None));
    }

    #[test]
    fn stacked_headings_chain_into_the_body_instead_of_stranding_a_category_less_pair() {
        use super::super::regions::Region;
        let title = Region {
            bbox: [100.0, 60.0, 700.0, 130.0],
            category: Some("Title".to_string()),
        };
        let section = Region {
            bbox: [100.0, 170.0, 600.0, 220.0],
            category: Some("Section-header".to_string()),
        };
        let body = Region {
            bbox: [90.0, 280.0, 900.0, 700.0],
            category: Some("Text".to_string()),
        };
        let merged =
            merge_headings_into_following_regions(vec![title.clone(), section.clone(), body]);
        assert_eq!(merged.len(), 1, "{merged:?}");
        assert_eq!(merged[0].bbox, [90.0, 60.0, 900.0, 700.0]);
        assert_eq!(
            merged[0].category, None,
            "the body absorption ends the chain as plain text"
        );

        let pair_at_page_end = merge_headings_into_following_regions(vec![title, section]);
        assert_eq!(pair_at_page_end.len(), 1);
        assert!(
            heading_like(pair_at_page_end[0].category.as_deref()),
            "a heading pair with no body below stays heading-like so the token cap applies: {:?}",
            pair_at_page_end[0].category
        );
    }

    #[test]
    fn prompt_bodies_carry_the_canonical_recipe() {
        assert!(DotsMode::LayoutAll.prompt().contains("human reading order"));
        assert!(DotsMode::LayoutAll.prompt().contains("[x1, y1, x2, y2]"));
        assert!(DotsMode::LayoutOnly
            .prompt()
            .contains("Do not output the corresponding text"));
        assert!(DotsMode::LayoutAll.emits_json());
        assert!(!DotsMode::PlainOcr.emits_json());
    }
}
