use std::path::Path;

use crate::api::LineRecognizer;
use crate::ctc;
use crate::lstm;
use crate::unicharset::{Recoder, Unicharset};
use crate::vgsl::Network;
use crate::{CharSpan, Error, GreyImage, Line, LineText, Logits, TessdataComponents};

pub const TD_LSTM: usize = 17;
pub const TD_LSTM_PUNC_DAWG: usize = 18;
pub const TD_LSTM_WORD_DAWG: usize = 19;
pub const TD_LSTM_NUMBER_DAWG: usize = 20;
pub const TD_LSTM_UNICHARSET: usize = 21;
pub const TD_LSTM_RECODER: usize = 22;
pub const TD_VERSION: usize = 23;

const TF_INT_MODE: i32 = 1;

fn err(msg: impl Into<String>) -> Error {
    Error::Traineddata(msg.into())
}

pub fn quantize_requested() -> bool {
    matches!(
        std::env::var("NV_OCR_INT8").ok().as_deref(),
        Some("1") | Some("int8") | Some("true")
    )
}

pub struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8], Error> {
        if self.remaining() < n {
            return Err(err(format!(
                "unexpected eof at {} reading {} bytes",
                self.pos, n
            )));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.bytes(1)?[0])
    }

    pub fn i8(&mut self) -> Result<i8, Error> {
        Ok(self.bytes(1)?[0] as i8)
    }

    pub fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    pub fn i32(&mut self) -> Result<i32, Error> {
        Ok(i32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    pub fn f32(&mut self) -> Result<f32, Error> {
        Ok(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    pub fn f64(&mut self) -> Result<f64, Error> {
        Ok(f64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }

    pub fn string(&mut self) -> Result<String, Error> {
        let n = self.u32()? as usize;
        if n > self.remaining() {
            return Err(err(format!(
                "string length {} exceeds remaining {}",
                n,
                self.remaining()
            )));
        }
        Ok(String::from_utf8_lossy(self.bytes(n)?).into_owned())
    }
}

pub struct Traineddata {
    components: Vec<Option<Vec<u8>>>,
}

impl Traineddata {
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        if data.len() < 4 {
            return Err(err("truncated traineddata header"));
        }
        let n = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        if n == 0 || n > 512 {
            return Err(err(format!("implausible traineddata entry count {}", n)));
        }
        let table_end = 4 + 8 * n;
        if data.len() < table_end {
            return Err(err("truncated traineddata offset table"));
        }
        let offsets: Vec<i64> = (0..n)
            .map(|i| i64::from_le_bytes(data[4 + 8 * i..12 + 8 * i].try_into().unwrap()))
            .collect();
        let mut components = vec![None; n];
        for i in 0..n {
            if offsets[i] < 0 {
                continue;
            }
            let start = offsets[i] as usize;
            let end = offsets[i + 1..]
                .iter()
                .find(|&&o| o >= 0)
                .map(|&o| o as usize)
                .unwrap_or(data.len());
            if start > end || end > data.len() {
                return Err(err(format!("bad offsets for traineddata entry {}", i)));
            }
            components[i] = Some(data[start..end].to_vec());
        }
        Ok(Self { components })
    }

    pub fn component(&self, kind: usize) -> Option<&[u8]> {
        self.components.get(kind).and_then(|c| c.as_deref())
    }

    pub fn version(&self) -> Option<String> {
        self.component(TD_VERSION).map(|b| {
            String::from_utf8_lossy(b)
                .trim_end_matches(['\0', '\n'])
                .to_string()
        })
    }

    pub fn into_components(self) -> TessdataComponents {
        let mut c = self.components;
        let mut take = |i: usize| -> Option<Vec<u8>> { c.get_mut(i).and_then(|x| x.take()) };
        let version = take(TD_VERSION).map(|b| {
            String::from_utf8_lossy(&b)
                .trim_end_matches(['\0', '\n'])
                .to_string()
        });
        TessdataComponents {
            lstm: take(TD_LSTM),
            lstm_punc_dawg: take(TD_LSTM_PUNC_DAWG),
            lstm_word_dawg: take(TD_LSTM_WORD_DAWG),
            lstm_number_dawg: take(TD_LSTM_NUMBER_DAWG),
            lstm_unicharset: take(TD_LSTM_UNICHARSET),
            lstm_recoder: take(TD_LSTM_RECODER),
            version,
        }
    }
}

pub struct LstmModel {
    pub network: Network,
    pub spec: String,
    pub int_mode: bool,
    pub null_code: usize,
    pub unicharset: Unicharset,
    pub recoder: Recoder,
    pub version: Option<String>,
}

pub struct RecognizedLine {
    pub text: String,
    pub chars: Vec<ctc::DecodedChar>,
    pub x_scale: f32,
    pub timesteps: usize,
}

impl LstmModel {
    pub fn from_file(path: &Path) -> Result<Self, Error> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, Error> {
        let td = Traineddata::parse(data)?;
        let version = td.version();
        let lstm_bytes = td
            .component(TD_LSTM)
            .ok_or_else(|| err("missing lstm component"))?;
        let mut cur = Cursor::new(lstm_bytes);
        let network = Network::deserialize(&mut cur)?;
        let spec = cur.string()?;
        let training_flags = cur.i32()?;
        let _training_iteration = cur.i32()?;
        let _sample_iteration = cur.i32()?;
        let null_code = cur.i32()?;
        let _adam_beta = cur.f32()?;
        let _learning_rate = cur.f32()?;
        let _momentum = cur.f32()?;
        if cur.remaining() != 0 {
            return Err(err(format!(
                "{} trailing bytes in lstm component",
                cur.remaining()
            )));
        }
        if null_code < 0 {
            return Err(err(format!("negative null_char {}", null_code)));
        }
        let ucs_bytes = td
            .component(TD_LSTM_UNICHARSET)
            .ok_or_else(|| err("missing lstm-unicharset component"))?;
        let unicharset = Unicharset::parse(ucs_bytes)?;
        let rec_bytes = td
            .component(TD_LSTM_RECODER)
            .ok_or_else(|| err("missing lstm-recoder component"))?;
        let recoder = Recoder::deserialize(&mut Cursor::new(rec_bytes))?;
        let mut int_mode = training_flags & TF_INT_MODE != 0;
        let mut network = network;
        if !int_mode && quantize_requested() {
            network.quantize_to_int8();
            int_mode = true;
        }
        Ok(Self {
            network,
            spec,
            int_mode,
            null_code: null_code as usize,
            unicharset,
            recoder,
            version,
        })
    }

    pub fn input_height(&self) -> Result<usize, Error> {
        self.network
            .input_height()
            .ok_or_else(|| Error::Network("network has no Input layer".into()))
    }

    pub fn num_classes(&self) -> Result<usize, Error> {
        self.network
            .output_classes()
            .ok_or_else(|| Error::Network("network has no output layer".into()))
    }

    pub fn forward_grey(&self, grey: &GreyImage) -> Result<Logits, Error> {
        let input_h = self.input_height()?;
        let input = lstm::input_from_grey(grey, input_h, self.int_mode)?;
        let out = self.network.forward(&input)?;
        out.into_logits()
    }

    pub fn recognize_grey(&self, grey: &GreyImage) -> Result<RecognizedLine, Error> {
        let logits = self.forward_grey(grey)?;
        let steps = ctc::beam_decode(&logits, self.null_code, ctc::DEFAULT_BEAM_WIDTH)?;
        let chars = ctc::codes_to_unichars(&steps, &self.recoder, &self.unicharset);
        let text: String = chars.iter().map(|c| c.text.as_str()).collect();
        let x_scale = if logits.timesteps == 0 {
            0.0
        } else {
            grey.w as f32 / logits.timesteps as f32
        };
        Ok(RecognizedLine {
            text,
            chars,
            x_scale,
            timesteps: logits.timesteps,
        })
    }
}

pub struct LstmLineRecognizer {
    model: LstmModel,
}

impl LstmLineRecognizer {
    pub fn new(model: LstmModel) -> Self {
        Self { model }
    }

    pub fn model(&self) -> &LstmModel {
        &self.model
    }
}

impl LineRecognizer for LstmLineRecognizer {
    fn recognize_line(&self, line: &Line) -> Result<LineText, Error> {
        let strip = if crate::binarize::ink_is_dark(&line.grey_strip) {
            line.grey_strip.clone()
        } else {
            line.grey_strip.invert()
        };
        let rec = self.model.recognize_grey(&strip)?;
        let mut spans = Vec::with_capacity(rec.chars.len());
        let mut byte = 0usize;
        for (i, ch) in rec.chars.iter().enumerate() {
            let t_end = rec.chars.get(i + 1).map(|n| n.t).unwrap_or(rec.timesteps);
            spans.push(CharSpan {
                byte_start: byte,
                byte_end: byte + ch.text.len(),
                t_start: ch.t,
                t_end,
                confidence: ch.prob,
            });
            byte += ch.text.len();
        }
        Ok(LineText {
            text: rec.text,
            spans,
        })
    }
}
