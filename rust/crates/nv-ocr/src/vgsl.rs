use crate::lstm::{self, Tensor};
use crate::traineddata::Cursor;
use crate::Error;

const NT_NAMES: [&str; 27] = [
    "Invalid",
    "Input",
    "Convolve",
    "Maxpool",
    "Parallel",
    "Replicated",
    "ParBidiLSTM",
    "DepParUDLSTM",
    "Par2dLSTM",
    "Series",
    "Reconfig",
    "RTLReversed",
    "TTBReversed",
    "XYTranspose",
    "LSTM",
    "SummLSTM",
    "Logistic",
    "LinLogistic",
    "LinTanh",
    "Tanh",
    "Relu",
    "Linear",
    "Softmax",
    "SoftmaxNoCTC",
    "LSTMSoftmax",
    "LSTMBinarySoftmax",
    "TensorFlow",
];

const NT_INPUT: usize = 1;
const NT_CONVOLVE: usize = 2;
const NT_MAXPOOL: usize = 3;
const NT_PARALLEL: usize = 4;
const NT_SERIES: usize = 9;
const NT_RECONFIG: usize = 10;
const NT_XREVERSED: usize = 11;
const NT_YREVERSED: usize = 12;
const NT_XYTRANSPOSE: usize = 13;
const NT_LSTM: usize = 14;
const NT_LSTM_SUMMARY: usize = 15;
const NT_LOGISTIC: usize = 16;
const NT_TANH: usize = 19;
const NT_RELU: usize = 20;
const NT_LINEAR: usize = 21;
const NT_SOFTMAX: usize = 22;
const NT_SOFTMAX_NO_CTC: usize = 23;

const NF_LAYER_SPECIFIC_LR: i32 = 64;

const MODE_INT8: u8 = 1;
const MODE_DOUBLE: u8 = 128;

const MAX_DIM: usize = 1 << 20;
const MAX_WEIGHT_BYTES: usize = 512 << 20;
const MAX_NET_DEPTH: usize = 64;

fn err(msg: impl Into<String>) -> Error {
    Error::Network(msg.into())
}

pub enum Weights {
    Float(Vec<f32>),
    Int8 { w: Vec<i8>, scales: Vec<f32> },
}

pub struct WeightMatrix {
    pub rows: usize,
    pub cols: usize,
    pub weights: Weights,
}

impl WeightMatrix {
    pub fn deserialize(cur: &mut Cursor) -> Result<Self, Error> {
        let mode = cur.u8()?;
        if mode & MODE_DOUBLE == 0 {
            return Err(err("legacy float weight matrix format unsupported"));
        }
        let int_mode = mode & MODE_INT8 != 0;
        let rows = cur.u32()? as usize;
        let cols = cur.u32()? as usize;
        if rows == 0 || cols < 2 || rows > MAX_DIM || cols > MAX_DIM {
            return Err(err(format!("implausible weight matrix {}x{}", rows, cols)));
        }
        let elems = rows
            .checked_mul(cols)
            .ok_or_else(|| err(format!("weight matrix {}x{} overflows", rows, cols)))?;
        let file_bytes = elems
            .checked_mul(if int_mode { 1 } else { 8 })
            .ok_or_else(|| err(format!("weight matrix {}x{} overflows", rows, cols)))?;
        if file_bytes > MAX_WEIGHT_BYTES {
            return Err(err(format!(
                "weight matrix {}x{} exceeds size cap",
                rows, cols
            )));
        }
        if file_bytes > cur.remaining() {
            return Err(err(format!(
                "weight matrix {}x{} needs {} bytes, {} remain",
                rows,
                cols,
                file_bytes,
                cur.remaining()
            )));
        }
        if int_mode {
            let _empty = cur.i8()?;
            let w: Vec<i8> = cur.bytes(elems)?.iter().map(|&b| b as i8).collect();
            let nscales = cur.u32()? as usize;
            if nscales < rows || nscales > rows + 64 {
                return Err(err(format!("scale count {} for {} rows", nscales, rows)));
            }
            if nscales.saturating_mul(8) > cur.remaining() {
                return Err(err(format!(
                    "{} scales need {} bytes, {} remain",
                    nscales,
                    nscales * 8,
                    cur.remaining()
                )));
            }
            let mut scales = Vec::with_capacity(rows);
            for i in 0..nscales {
                let v = cur.f64()?;
                if i < rows {
                    scales.push((v / 127.0) as f32);
                }
            }
            Ok(Self {
                rows,
                cols,
                weights: Weights::Int8 { w, scales },
            })
        } else {
            let _empty = cur.f64()?;
            let mut w = Vec::with_capacity(elems);
            for _ in 0..elems {
                w.push(cur.f64()? as f32);
            }
            Ok(Self {
                rows,
                cols,
                weights: Weights::Float(w),
            })
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.cols - 1
    }

    pub fn int_mode(&self) -> bool {
        matches!(self.weights, Weights::Int8 { .. })
    }

    pub fn quantize_to_int8(&mut self) {
        let Weights::Float(f) = &self.weights else {
            return;
        };
        let (rows, cols) = (self.rows, self.cols);
        let mut w = vec![0i8; rows * cols];
        let mut scales = vec![0f32; rows];
        for r in 0..rows {
            let row = &f[r * cols..(r + 1) * cols];
            let maxabs = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let inv = if maxabs > 0.0 { 127.0 / maxabs } else { 0.0 };
            for (dst, &v) in w[r * cols..(r + 1) * cols].iter_mut().zip(row) {
                *dst = (v * inv).round().clamp(-127.0, 127.0) as i8;
            }
            scales[r] = maxabs / (127.0 * 127.0);
        }
        self.weights = Weights::Int8 { w, scales };
    }
}

pub struct LstmLayer {
    pub ni: usize,
    pub ns: usize,
    pub na: usize,
    pub summarizing: bool,
    pub gates: [WeightMatrix; 4],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FcKind {
    Logistic,
    Tanh,
    Relu,
    Linear,
    Softmax,
    SoftmaxNoCtc,
}

pub enum Network {
    Input { height: usize, depth: usize },
    Series(Vec<Network>),
    Convolve { half_x: usize, half_y: usize },
    Maxpool { x_scale: usize, y_scale: usize },
    ReversedX(Box<Network>),
    ReversedY(Box<Network>),
    TransposedXY(Box<Network>),
    Lstm(LstmLayer),
    Fc { kind: FcKind, weights: WeightMatrix },
}

impl Network {
    pub fn deserialize(cur: &mut Cursor) -> Result<Self, Error> {
        Self::deserialize_at(cur, 0)
    }

    fn deserialize_at(cur: &mut Cursor, depth: usize) -> Result<Self, Error> {
        if depth > MAX_NET_DEPTH {
            return Err(err(format!(
                "network nesting exceeds depth {}",
                MAX_NET_DEPTH
            )));
        }
        let mut ty = cur.u8()? as usize;
        if ty == 0 {
            let name = cur.string()?;
            ty = NT_NAMES
                .iter()
                .position(|&n| n == name)
                .ok_or_else(|| err(format!("unknown network type name {:?}", name)))?;
        }
        let _training = cur.u8()?;
        let _needs_backprop = cur.u8()?;
        let flags = cur.i32()?;
        let ni = cur.i32()?;
        let no = cur.i32()?;
        let _num_weights = cur.i32()?;
        let _name = cur.string()?;
        if ni < 0 || no <= 0 {
            return Err(err(format!("bad layer dims ni={} no={}", ni, no)));
        }
        let ni = ni as usize;
        let no = no as usize;
        match ty {
            NT_INPUT => {
                let _batch = cur.i32()?;
                let height = cur.i32()?;
                let _width = cur.i32()?;
                let depth = cur.i32()?;
                let _loss_type = cur.i32()?;
                if height <= 0 || depth <= 0 {
                    return Err(err(format!("bad input shape h={} d={}", height, depth)));
                }
                Ok(Network::Input {
                    height: height as usize,
                    depth: depth as usize,
                })
            }
            NT_CONVOLVE => {
                let half_x = cur.i32()?;
                let half_y = cur.i32()?;
                if half_x < 0 || half_y < 0 {
                    return Err(err("negative convolve half-window"));
                }
                Ok(Network::Convolve {
                    half_x: half_x as usize,
                    half_y: half_y as usize,
                })
            }
            NT_MAXPOOL => {
                let x_scale = cur.i32()?;
                let y_scale = cur.i32()?;
                if x_scale <= 0 || y_scale <= 0 {
                    return Err(err("non-positive maxpool scale"));
                }
                Ok(Network::Maxpool {
                    x_scale: x_scale as usize,
                    y_scale: y_scale as usize,
                })
            }
            NT_SERIES | NT_XREVERSED | NT_YREVERSED | NT_XYTRANSPOSE | NT_PARALLEL
            | NT_RECONFIG => {
                let count = cur.u32()? as usize;
                if count == 0 || count > 4096 {
                    return Err(err(format!("implausible plumbing stack size {}", count)));
                }
                if count > cur.remaining() {
                    return Err(err(format!(
                        "plumbing stack of {} children exceeds {} remaining bytes",
                        count,
                        cur.remaining()
                    )));
                }
                let mut stack = Vec::with_capacity(count);
                for _ in 0..count {
                    stack.push(Network::deserialize_at(cur, depth + 1)?);
                }
                if flags & NF_LAYER_SPECIFIC_LR != 0 {
                    let n = cur.u32()? as usize;
                    if n.saturating_mul(4) > cur.remaining() {
                        return Err(err(format!(
                            "{} learning rates exceed {} remaining bytes",
                            n,
                            cur.remaining()
                        )));
                    }
                    for _ in 0..n {
                        let _lr = cur.f32()?;
                    }
                }
                match ty {
                    NT_SERIES => Ok(Network::Series(stack)),
                    NT_XREVERSED | NT_YREVERSED | NT_XYTRANSPOSE => {
                        if stack.len() != 1 {
                            return Err(err(format!(
                                "reversed wrapper with {} children",
                                stack.len()
                            )));
                        }
                        let inner = Box::new(stack.into_iter().next().unwrap());
                        Ok(match ty {
                            NT_XREVERSED => Network::ReversedX(inner),
                            NT_YREVERSED => Network::ReversedY(inner),
                            _ => Network::TransposedXY(inner),
                        })
                    }
                    _ => Err(err(format!("unsupported plumbing type {}", NT_NAMES[ty]))),
                }
            }
            NT_LSTM | NT_LSTM_SUMMARY => {
                let na = cur.i32()?;
                if na <= 0 {
                    return Err(err("non-positive lstm na"));
                }
                let g0 = WeightMatrix::deserialize(cur)?;
                let g1 = WeightMatrix::deserialize(cur)?;
                let g2 = WeightMatrix::deserialize(cur)?;
                let g3 = WeightMatrix::deserialize(cur)?;
                let layer = LstmLayer {
                    ni,
                    ns: no,
                    na: na as usize,
                    summarizing: ty == NT_LSTM_SUMMARY,
                    gates: [g0, g1, g2, g3],
                };
                for g in &layer.gates {
                    if g.rows != layer.ns || g.num_inputs() != layer.na {
                        return Err(err(format!(
                            "lstm gate matrix {}x{} does not match ns={} na={}",
                            g.rows, g.cols, layer.ns, layer.na
                        )));
                    }
                }
                Ok(Network::Lstm(layer))
            }
            NT_LOGISTIC | NT_TANH | NT_RELU | NT_LINEAR | NT_SOFTMAX | NT_SOFTMAX_NO_CTC => {
                let weights = WeightMatrix::deserialize(cur)?;
                if weights.rows != no || weights.num_inputs() != ni {
                    return Err(err(format!(
                        "fc matrix {}x{} does not match ni={} no={}",
                        weights.rows, weights.cols, ni, no
                    )));
                }
                let kind = match ty {
                    NT_LOGISTIC => FcKind::Logistic,
                    NT_TANH => FcKind::Tanh,
                    NT_RELU => FcKind::Relu,
                    NT_LINEAR => FcKind::Linear,
                    NT_SOFTMAX => FcKind::Softmax,
                    _ => FcKind::SoftmaxNoCtc,
                };
                Ok(Network::Fc { kind, weights })
            }
            _ => Err(err(format!(
                "unsupported network type {}",
                NT_NAMES.get(ty).copied().unwrap_or("?")
            ))),
        }
    }

    pub fn quantize_to_int8(&mut self) {
        match self {
            Network::Series(stack) => stack.iter_mut().for_each(|n| n.quantize_to_int8()),
            Network::ReversedX(n) | Network::ReversedY(n) | Network::TransposedXY(n) => {
                n.quantize_to_int8()
            }
            Network::Lstm(l) => l.gates.iter_mut().for_each(|g| g.quantize_to_int8()),
            Network::Fc { weights, .. } => weights.quantize_to_int8(),
            _ => {}
        }
    }

    pub fn input_height(&self) -> Option<usize> {
        match self {
            Network::Input { height, .. } => Some(*height),
            Network::Series(stack) => stack.first().and_then(|n| n.input_height()),
            _ => None,
        }
    }

    pub fn output_classes(&self) -> Option<usize> {
        match self {
            Network::Fc { weights, .. } => Some(weights.rows),
            Network::Series(stack) => stack.last().and_then(|n| n.output_classes()),
            _ => None,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        match self {
            Network::Input { height, depth } => {
                if input.h != *height || input.d != *depth {
                    return Err(err(format!(
                        "input tensor {}x{}x{} does not match network input h={} d={}",
                        input.h, input.w, input.d, height, depth
                    )));
                }
                Ok(input.clone())
            }
            Network::Series(stack) => {
                let mut t = input.clone();
                for n in stack {
                    t = n.forward(&t)?;
                }
                Ok(t)
            }
            Network::Convolve { half_x, half_y } => Ok(lstm::conv_forward(input, *half_x, *half_y)),
            Network::Maxpool { x_scale, y_scale } => {
                lstm::maxpool_forward(input, *x_scale, *y_scale)
            }
            Network::ReversedX(inner) => {
                let rev = lstm::reverse_x(input);
                let out = inner.forward(&rev)?;
                Ok(lstm::reverse_x(&out))
            }
            Network::ReversedY(inner) => {
                let rev = lstm::reverse_y(input);
                let out = inner.forward(&rev)?;
                Ok(lstm::reverse_y(&out))
            }
            Network::TransposedXY(inner) => {
                let tr = lstm::transpose_xy(input);
                let out = inner.forward(&tr)?;
                Ok(lstm::transpose_xy(&out))
            }
            Network::Lstm(layer) => lstm::lstm_forward(layer, input),
            Network::Fc { kind, weights } => lstm::fc_forward(*kind, weights, input),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lstm::{matvec_f32_scalar, matvec_i8_scalar};

    #[test]
    fn quantize_matvec_tracks_float_within_int8_error() {
        let rows = 5;
        let cols = 9;
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 % 51) as f32 / 50.0) - 0.5)
            .collect();
        let mut m = WeightMatrix {
            rows,
            cols,
            weights: Weights::Float(w.clone()),
        };
        let act: Vec<f32> = (0..cols - 1).map(|i| ((i * 13 % 7) as f32 / 6.0) - 0.5).collect();
        let mut f_out = vec![0f32; rows];
        matvec_f32_scalar(rows, cols, &w, &act, &mut f_out);
        m.quantize_to_int8();
        let Weights::Int8 { w: qw, scales } = &m.weights else {
            panic!("quantize_to_int8 must yield int8 weights");
        };
        let qa: Vec<i8> = act
            .iter()
            .map(|v| (v * 127.0).round().clamp(-127.0, 127.0) as i8)
            .collect();
        let mut i_out = vec![0f32; rows];
        matvec_i8_scalar(rows, cols, qw, scales, &qa, &mut i_out);
        for (f, i) in f_out.iter().zip(&i_out) {
            assert!(
                (f - i).abs() < 0.05,
                "int8 matvec {i} strays from float {f}"
            );
        }
    }
}
