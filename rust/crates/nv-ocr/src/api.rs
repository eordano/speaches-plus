use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::layout::extract_lines;
use crate::traineddata::{LstmLineRecognizer, LstmModel};
use crate::{raster, Error, Line, LineText, OcrResult, OcrToken};

pub trait PageAnalyzer: Send + Sync {
    fn analyze(&self, image_bytes: &[u8]) -> Result<Vec<Line>, Error>;
}

pub trait OcrBackend: Send + Sync {
    fn recognize_lines(&self, lines: &[Line]) -> Result<Vec<LineText>, Error>;
}

pub trait LineRecognizer: Send + Sync {
    fn recognize_line(&self, line: &Line) -> Result<LineText, Error>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Classical,
    DeepSeek,
    DotsOcr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeepSeekMode {
    #[default]
    FreeOcr,
    Markdown,
}

impl DeepSeekMode {
    pub fn prompt(self) -> &'static str {
        match self {
            DeepSeekMode::FreeOcr => "<image>\nFree OCR. ",
            DeepSeekMode::Markdown => "<image>\n<|grounding|>Convert the document to markdown. ",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResolutionHint {
    #[default]
    Auto,
    Tiled,
    Base1024,
    Base768,
}

impl ResolutionHint {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" | "" => Some(ResolutionHint::Auto),
            "tiled" | "gundam" => Some(ResolutionHint::Tiled),
            "base1024" | "base" => Some(ResolutionHint::Base1024),
            "base768" => Some(ResolutionHint::Base768),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ResolutionHint::Auto => "auto",
            ResolutionHint::Tiled => "tiled",
            ResolutionHint::Base1024 => "base1024",
            ResolutionHint::Base768 => "base768",
        }
    }
}

pub trait DeepSeekOcr2Model: Send + Sync {
    fn recognize_page(&self, image_bytes: &[u8], mode: DeepSeekMode) -> Result<OcrResult, Error>;

    fn recognize_page_hinted(
        &self,
        image_bytes: &[u8],
        mode: DeepSeekMode,
        hint: ResolutionHint,
    ) -> Result<OcrResult, Error> {
        let _ = hint;
        self.recognize_page(image_bytes, mode)
    }

    fn recognize_page_budgeted(
        &self,
        image_bytes: &[u8],
        mode: DeepSeekMode,
        hint: ResolutionHint,
        max_new_override: Option<usize>,
    ) -> Result<OcrResult, Error> {
        let _ = max_new_override;
        self.recognize_page_hinted(image_bytes, mode, hint)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DotsOcrMode {
    #[default]
    LayoutAll,
    LayoutOnly,
    PlainOcr,
}

impl DotsOcrMode {
    pub fn emits_layout(self) -> bool {
        !matches!(self, DotsOcrMode::PlainOcr)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields)
)]
pub struct LayoutBox {
    pub order: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[f32; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct LayoutPageResult {
    pub elements: Vec<LayoutBox>,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct DotsPageResult {
    #[serde(flatten)]
    pub result: OcrResult,
    pub layout: LayoutPageResult,
}

pub trait DotsOcrModel: Send + Sync {
    fn recognize_page(&self, image_bytes: &[u8], mode: DotsOcrMode) -> Result<OcrResult, Error>;

    fn recognize_layout(
        &self,
        image_bytes: &[u8],
        mode: DotsOcrMode,
    ) -> Result<DotsPageResult, Error> {
        Ok(DotsPageResult {
            result: self.recognize_page(image_bytes, mode)?,
            layout: LayoutPageResult::default(),
        })
    }

    fn recognize_layout_budgeted(
        &self,
        image_bytes: &[u8],
        mode: DotsOcrMode,
        max_new_override: Option<usize>,
    ) -> Result<DotsPageResult, Error> {
        let _ = max_new_override;
        self.recognize_layout(image_bytes, mode)
    }
}

pub struct ClassicalAnalyzer;

impl PageAnalyzer for ClassicalAnalyzer {
    fn analyze(&self, image_bytes: &[u8]) -> Result<Vec<Line>, Error> {
        let grey = raster::load(image_bytes)?;
        Ok(extract_lines(&grey).lines)
    }
}

pub struct ClassicalBackend {
    recognizer: Box<dyn LineRecognizer>,
}

impl ClassicalBackend {
    pub fn new(recognizer: Box<dyn LineRecognizer>) -> Self {
        Self { recognizer }
    }
}

pub fn thread_budget() -> usize {
    match std::env::var("NV_OCR_THREADS") {
        Ok(v) => v.trim().parse::<usize>().unwrap_or(1).max(1),
        Err(_) => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    }
}

impl OcrBackend for ClassicalBackend {
    fn recognize_lines(&self, lines: &[Line]) -> Result<Vec<LineText>, Error> {
        let threads = thread_budget().min(lines.len().max(1));
        if threads <= 1 {
            return lines
                .iter()
                .map(|line| self.recognizer.recognize_line(line))
                .collect();
        }
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut parts: Vec<Vec<(usize, Result<LineText, Error>)>> = Vec::new();
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| {
                        let mut local = Vec::new();
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if i >= lines.len() {
                                break;
                            }
                            local.push((i, self.recognizer.recognize_line(&lines[i])));
                        }
                        local
                    })
                })
                .collect();
            for h in handles {
                parts.push(h.join().expect("ocr line worker panicked"));
            }
        });
        let mut out: Vec<Option<LineText>> = (0..lines.len()).map(|_| None).collect();
        for (i, r) in parts.into_iter().flatten() {
            out[i] = Some(r?);
        }
        Ok(out.into_iter().map(|o| o.unwrap()).collect())
    }
}

enum EngineInner {
    Lines {
        analyzer: Box<dyn PageAnalyzer>,
        backend: Box<dyn OcrBackend>,
    },
    DeepSeek(Box<dyn DeepSeekOcr2Model>),
    DotsOcr(Box<dyn DotsOcrModel>),
}

pub struct OcrEngine {
    inner: EngineInner,
}

impl OcrEngine {
    pub fn new(analyzer: Box<dyn PageAnalyzer>, backend: Box<dyn OcrBackend>) -> Self {
        Self {
            inner: EngineInner::Lines { analyzer, backend },
        }
    }

    pub fn from_deepseek(model: Box<dyn DeepSeekOcr2Model>) -> Self {
        Self {
            inner: EngineInner::DeepSeek(model),
        }
    }

    pub fn from_dots(model: Box<dyn DotsOcrModel>) -> Self {
        Self {
            inner: EngineInner::DotsOcr(model),
        }
    }

    pub fn dots_mode_for(mode: DeepSeekMode) -> DotsOcrMode {
        match mode {
            DeepSeekMode::FreeOcr => DotsOcrMode::LayoutAll,
            DeepSeekMode::Markdown => DotsOcrMode::LayoutAll,
        }
    }

    pub fn from_traineddata(path: &Path, kind: BackendKind) -> Result<Self, Error> {
        match kind {
            BackendKind::Classical => {
                let file = if path.is_dir() {
                    path.join("eng.traineddata")
                } else {
                    path.to_path_buf()
                };
                let model = LstmModel::from_file(&file)?;
                let recognizer = LstmLineRecognizer::new(model);
                Ok(Self::new(
                    Box::new(ClassicalAnalyzer),
                    Box::new(ClassicalBackend::new(Box::new(recognizer))),
                ))
            }
            BackendKind::DeepSeek => Err(Error::NotWired(
                "deepseek backend loads from a checkpoint directory via OcrEngine::from_deepseek; the nv-models deepseek_ocr implementation provides the DeepSeekOcr2Model",
            )),
            BackendKind::DotsOcr => Err(Error::NotWired(
                "dots.ocr backend loads from a checkpoint directory via OcrEngine::from_dots; the nv-models dots_ocr implementation provides the DotsOcrModel",
            )),
        }
    }

    pub fn backend_kind(&self) -> BackendKind {
        match self.inner {
            EngineInner::Lines { .. } => BackendKind::Classical,
            EngineInner::DeepSeek(_) => BackendKind::DeepSeek,
            EngineInner::DotsOcr(_) => BackendKind::DotsOcr,
        }
    }

    pub fn recognize(&self, image_bytes: &[u8]) -> Result<OcrResult, Error> {
        self.recognize_mode(image_bytes, DeepSeekMode::FreeOcr)
    }

    pub fn recognize_mode(
        &self,
        image_bytes: &[u8],
        mode: DeepSeekMode,
    ) -> Result<OcrResult, Error> {
        match &self.inner {
            EngineInner::Lines { analyzer, backend } => {
                let lines = analyzer.analyze(image_bytes)?;
                let texts = backend.recognize_lines(&lines)?;
                if texts.len() != lines.len() {
                    return Err(Error::Backend(format!(
                        "backend returned {} line texts for {} lines",
                        texts.len(),
                        lines.len()
                    )));
                }
                Ok(assemble(&lines, &texts))
            }
            EngineInner::DeepSeek(model) => model.recognize_page(image_bytes, mode),
            EngineInner::DotsOcr(model) => {
                model.recognize_page(image_bytes, Self::dots_mode_for(mode))
            }
        }
    }

    pub fn recognize_hinted(
        &self,
        image_bytes: &[u8],
        mode: DeepSeekMode,
        hint: ResolutionHint,
    ) -> Result<OcrResult, Error> {
        match &self.inner {
            EngineInner::DeepSeek(model) => model.recognize_page_hinted(image_bytes, mode, hint),
            _ => self.recognize_mode(image_bytes, mode),
        }
    }

    pub fn recognize_hinted_budgeted(
        &self,
        image_bytes: &[u8],
        mode: DeepSeekMode,
        hint: ResolutionHint,
        max_new_override: Option<usize>,
    ) -> Result<OcrResult, Error> {
        match &self.inner {
            EngineInner::DeepSeek(model) => {
                model.recognize_page_budgeted(image_bytes, mode, hint, max_new_override)
            }
            _ => self.recognize_hinted(image_bytes, mode, hint),
        }
    }

    pub fn recognize_dots(
        &self,
        image_bytes: &[u8],
        mode: DotsOcrMode,
    ) -> Result<OcrResult, Error> {
        match &self.inner {
            EngineInner::DotsOcr(model) => model.recognize_page(image_bytes, mode),
            _ => Err(Error::NotWired(
                "recognize_dots requires an OcrEngine built with OcrEngine::from_dots",
            )),
        }
    }

    pub fn recognize_dots_layout(
        &self,
        image_bytes: &[u8],
        mode: DotsOcrMode,
    ) -> Result<DotsPageResult, Error> {
        match &self.inner {
            EngineInner::DotsOcr(model) => model.recognize_layout(image_bytes, mode),
            _ => Err(Error::NotWired(
                "recognize_dots_layout requires an OcrEngine built with OcrEngine::from_dots",
            )),
        }
    }

    pub fn recognize_dots_layout_budgeted(
        &self,
        image_bytes: &[u8],
        mode: DotsOcrMode,
        max_new_override: Option<usize>,
    ) -> Result<DotsPageResult, Error> {
        match &self.inner {
            EngineInner::DotsOcr(model) => {
                model.recognize_layout_budgeted(image_bytes, mode, max_new_override)
            }
            _ => self.recognize_dots_layout(image_bytes, mode),
        }
    }
}

fn assemble(lines: &[Line], texts: &[LineText]) -> OcrResult {
    let mut text = String::new();
    let mut tokens = Vec::new();
    for (i, (line, lt)) in lines.iter().zip(texts.iter()).enumerate() {
        if i > 0 {
            text.push('\n');
        }
        let base = text.len();
        text.push_str(&lt.text);
        for (wi, (ws, we)) in split_words(&lt.text).into_iter().enumerate() {
            let rect = line.words.get(wi).map(|w| w.rect).unwrap_or(line.bbox);
            tokens.push(OcrToken {
                start: base + ws,
                end_exclusive: base + we,
                rect,
                confidence: span_confidence(lt, ws, we),
            });
        }
    }
    OcrResult {
        text,
        tokens,
        truncated: false,
        looped: false,
    }
}

fn split_words(text: &str) -> Vec<(usize, usize)> {
    let mut words = Vec::new();
    let mut start = None;
    for (idx, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(s) = start.take() {
                words.push((s, idx));
            }
        } else if start.is_none() {
            start = Some(idx);
        }
    }
    if let Some(s) = start {
        words.push((s, text.len()));
    }
    words
}

fn span_confidence(lt: &LineText, ws: usize, we: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut n = 0usize;
    for span in &lt.spans {
        if span.byte_start < we && span.byte_end > ws {
            sum += span.confidence;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f32
    }
}
