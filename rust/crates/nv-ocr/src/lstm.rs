use crate::vgsl::{FcKind, LstmLayer, WeightMatrix, Weights};
use crate::{Error, GreyImage, Logits};

const STATE_CLIP: f32 = 100.0;

#[derive(Clone)]
pub enum Buf {
    F32(Vec<f32>),
    I8(Vec<i8>),
}

#[derive(Clone)]
pub struct Tensor {
    pub h: usize,
    pub w: usize,
    pub d: usize,
    pub buf: Buf,
}

pub enum StepView<'a> {
    F32(&'a [f32]),
    I8(&'a [i8]),
}

fn quant127(v: f32) -> i8 {
    (v * 127.0).round().clamp(-127.0, 127.0) as i8
}

fn quant128(v: f32) -> i8 {
    (v * 128.0).round().clamp(-127.0, 127.0) as i8
}

impl Tensor {
    pub fn zeros(h: usize, w: usize, d: usize, int_mode: bool) -> Self {
        let n = h * w * d;
        let buf = if int_mode {
            Buf::I8(vec![0; n])
        } else {
            Buf::F32(vec![0.0; n])
        };
        Self { h, w, d, buf }
    }

    pub fn int_mode(&self) -> bool {
        matches!(self.buf, Buf::I8(_))
    }

    pub fn steps(&self) -> usize {
        self.h * self.w
    }

    pub fn step_view(&self, t: usize) -> StepView<'_> {
        let r = t * self.d..(t + 1) * self.d;
        match &self.buf {
            Buf::F32(f) => StepView::F32(&f[r]),
            Buf::I8(i) => StepView::I8(&i[r]),
        }
    }

    pub fn step_f32(&self, t: usize) -> Result<&[f32], Error> {
        match &self.buf {
            Buf::F32(f) => Ok(&f[t * self.d..(t + 1) * self.d]),
            Buf::I8(_) => Err(Error::Network("expected float tensor, got int8".into())),
        }
    }

    pub fn write_step(&mut self, t: usize, vals: &[f32]) {
        let off = t * self.d;
        match &mut self.buf {
            Buf::F32(f) => f[off..off + vals.len()].copy_from_slice(vals),
            Buf::I8(i) => {
                for (dst, &v) in i[off..off + vals.len()].iter_mut().zip(vals) {
                    *dst = quant127(v);
                }
            }
        }
    }

    pub fn copy_part(
        &mut self,
        dst_t: usize,
        dst_off: usize,
        src: &Tensor,
        src_t: usize,
        len: usize,
    ) {
        let d = dst_t * self.d + dst_off;
        let s = src_t * src.d;
        match (&mut self.buf, &src.buf) {
            (Buf::F32(dv), Buf::F32(sv)) => dv[d..d + len].copy_from_slice(&sv[s..s + len]),
            (Buf::I8(dv), Buf::I8(sv)) => dv[d..d + len].copy_from_slice(&sv[s..s + len]),
            _ => panic!("tensor mode mismatch in copy_part"),
        }
    }

    pub fn into_logits(self) -> Result<Logits, Error> {
        if self.h != 1 {
            return Err(Error::Network(format!(
                "expected height-1 output, got h={}",
                self.h
            )));
        }
        match self.buf {
            Buf::F32(data) => Ok(Logits {
                data,
                timesteps: self.w,
                classes: self.d,
            }),
            Buf::I8(_) => Err(Error::Network(
                "output tensor is int8, expected float".into(),
            )),
        }
    }
}

pub fn matvec_f32_scalar(rows: usize, cols: usize, w: &[f32], u: &[f32], out: &mut [f32]) {
    let n = cols - 1;
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut sum = row[n];
        for j in 0..n {
            sum += row[j] * u[j];
        }
        out[r] = sum;
    }
}

pub fn matvec_i8_scalar(
    rows: usize,
    cols: usize,
    w: &[i8],
    scales: &[f32],
    u: &[i8],
    out: &mut [f32],
) {
    let n = cols - 1;
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut total: i32 = row[n] as i32 * 127;
        for j in 0..n {
            total += row[j] as i32 * u[j] as i32;
        }
        out[r] = total as f32 * scales[r];
    }
}

pub fn matvec_f32(rows: usize, cols: usize, w: &[f32], u: &[f32], out: &mut [f32]) {
    crate::simd::matvec_f32(rows, cols, w, u, out);
}

pub fn matvec_i8(rows: usize, cols: usize, w: &[i8], scales: &[f32], u: &[i8], out: &mut [f32]) {
    crate::simd::matvec_i8(rows, cols, w, scales, u, out);
}

pub fn dot(wm: &WeightMatrix, u: &StepView, out: &mut [f32]) -> Result<(), Error> {
    match (&wm.weights, u) {
        (Weights::Float(w), StepView::F32(u)) => {
            matvec_f32(wm.rows, wm.cols, w, u, out);
            Ok(())
        }
        (Weights::Int8 { w, scales }, StepView::I8(u)) => {
            matvec_i8(wm.rows, wm.cols, w, scales, u, out);
            Ok(())
        }
        (Weights::Float(_), StepView::I8(_)) => Err(Error::Network(
            "int8 activations fed to float weights".into(),
        )),
        (Weights::Int8 { .. }, StepView::F32(_)) => Err(Error::Network(
            "float activations fed to int8 weights".into(),
        )),
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn softmax_inplace(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

fn stats_ile(hist: &[u32; 256], frac: f32) -> f32 {
    let total: u32 = hist.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let target = frac * total as f32;
    let mut cum = 0.0f32;
    for (v, &c) in hist.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let next = cum + c as f32;
        if next >= target {
            let within = ((target - cum) / c as f32).clamp(0.0, 1.0);
            return v as f32 + within;
        }
        cum = next;
    }
    255.0
}

pub fn compute_black_white(g: &GreyImage) -> (f32, f32) {
    let mut mins = [0u32; 256];
    let mut maxes = [0u32; 256];
    if g.w >= 3 && g.h > 0 {
        let y = g.h / 2;
        let mut prev = g.get(0, y) as i32;
        let mut curr = g.get(1, y) as i32;
        for x in 1..g.w - 1 {
            let next = g.get(x + 1, y) as i32;
            if (curr < prev && curr <= next) || (curr <= prev && curr < next) {
                mins[curr as usize] += 1;
            }
            if (curr > prev && curr >= next) || (curr >= prev && curr > next) {
                maxes[curr as usize] += 1;
            }
            prev = curr;
            curr = next;
        }
    }
    if mins.iter().all(|&c| c == 0) {
        mins[0] = 1;
    }
    if maxes.iter().all(|&c| c == 0) {
        maxes[255] = 1;
    }
    (stats_ile(&mins, 0.25), stats_ile(&maxes, 0.75))
}

pub fn input_from_grey(grey: &GreyImage, target_h: usize, int_mode: bool) -> Result<Tensor, Error> {
    if grey.w == 0 || grey.h == 0 || target_h == 0 {
        return Err(Error::Network("empty line image".into()));
    }
    let scaled = if grey.h == target_h {
        grey.clone()
    } else {
        let scale = target_h as f32 / grey.h as f32;
        let out_w = ((grey.w as f32 * scale).round() as usize).max(1);
        let sx = grey.w as f32 / out_w as f32;
        let sy = grey.h as f32 / target_h as f32;
        GreyImage::from_fn(out_w, target_h, |x, y| {
            let xin = (x as f32 + 0.5) * sx - 0.5;
            let yin = (y as f32 + 0.5) * sy - 0.5;
            grey.sample_bilinear(xin, yin, 255)
                .round()
                .clamp(0.0, 255.0) as u8
        })
    };
    let (black, white) = compute_black_white(&scaled);
    let mut contrast = (white - black) / 2.0;
    if contrast <= 0.0 {
        contrast = 1.0;
    }
    let mut t = Tensor::zeros(target_h, scaled.w, 1, int_mode);
    for y in 0..target_h {
        for x in 0..scaled.w {
            let v = (scaled.get(x, y) as f32 - black) / contrast - 1.0;
            let step = y * scaled.w + x;
            match &mut t.buf {
                Buf::F32(f) => f[step] = v,
                Buf::I8(i) => i[step] = quant128(v),
            }
        }
    }
    Ok(t)
}

pub fn conv_forward(input: &Tensor, half_x: usize, half_y: usize) -> Tensor {
    let ni = input.d;
    let y_span = 2 * half_y + 1;
    let x_span = 2 * half_x + 1;
    let no = ni * y_span * x_span;
    let mut out = Tensor::zeros(input.h, input.w, no, input.int_mode());
    let h = input.h;
    let w = input.w;
    match (&mut out.buf, &input.buf) {
        (Buf::F32(dst), Buf::F32(src)) => conv_fill(src, dst, h, w, ni, half_x, half_y),
        (Buf::I8(dst), Buf::I8(src)) => conv_fill(src, dst, h, w, ni, half_x, half_y),
        _ => unreachable!(),
    }
    out
}

fn conv_fill<T: Copy>(
    src: &[T],
    dst: &mut [T],
    h: usize,
    w: usize,
    ni: usize,
    half_x: usize,
    half_y: usize,
) {
    let y_span = 2 * half_y + 1;
    let x_span = 2 * half_x + 1;
    let no = ni * y_span * x_span;
    let mut sys = vec![0usize; y_span];
    for y in 0..h {
        for (i, sy) in sys.iter_mut().enumerate() {
            *sy = ((y + i) as isize - half_y as isize).clamp(0, h as isize - 1) as usize * w;
        }
        let row_off = y * w * no;
        for x in 0..w {
            let mut o = row_off + x * no;
            for dx in 0..x_span {
                let sx = ((x + dx) as isize - half_x as isize).clamp(0, w as isize - 1) as usize;
                if ni == 1 {
                    for &syw in &sys {
                        dst[o] = src[syw + sx];
                        o += 1;
                    }
                } else {
                    for &syw in &sys {
                        let s = (syw + sx) * ni;
                        dst[o..o + ni].copy_from_slice(&src[s..s + ni]);
                        o += ni;
                    }
                }
            }
        }
    }
}

pub fn maxpool_forward(input: &Tensor, x_scale: usize, y_scale: usize) -> Result<Tensor, Error> {
    if input.h == 0 || input.w == 0 {
        return Err(Error::Network(format!(
            "maxpool {}x{} on empty {}x{} input",
            x_scale, y_scale, input.w, input.h
        )));
    }
    let out_h = (input.h / y_scale).max(1);
    let out_w = (input.w / x_scale).max(1);
    let d = input.d;
    let mut out = Tensor::zeros(out_h, out_w, d, input.int_mode());
    for oy in 0..out_h {
        for ox in 0..out_w {
            let out_t = oy * out_w + ox;
            for dy in 0..y_scale {
                for dx in 0..x_scale {
                    let sy = oy * y_scale + dy;
                    let sx = ox * x_scale + dx;
                    if sy >= input.h || sx >= input.w {
                        continue;
                    }
                    let src_t = sy * input.w + sx;
                    match (&mut out.buf, &input.buf) {
                        (Buf::F32(o), Buf::F32(i)) => {
                            for k in 0..d {
                                let s = i[src_t * d + k];
                                let dst = &mut o[out_t * d + k];
                                if dx == 0 && dy == 0 || s > *dst {
                                    *dst = s;
                                }
                            }
                        }
                        (Buf::I8(o), Buf::I8(i)) => {
                            for k in 0..d {
                                let s = i[src_t * d + k];
                                let dst = &mut o[out_t * d + k];
                                if dx == 0 && dy == 0 || s > *dst {
                                    *dst = s;
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }
    }
    Ok(out)
}

pub fn reverse_x(input: &Tensor) -> Tensor {
    let mut out = Tensor::zeros(input.h, input.w, input.d, input.int_mode());
    for y in 0..input.h {
        for x in 0..input.w {
            let src_t = y * input.w + x;
            let dst_t = y * input.w + (input.w - 1 - x);
            out.copy_part(dst_t, 0, input, src_t, input.d);
        }
    }
    out
}

pub fn reverse_y(input: &Tensor) -> Tensor {
    let mut out = Tensor::zeros(input.h, input.w, input.d, input.int_mode());
    for y in 0..input.h {
        for x in 0..input.w {
            let src_t = y * input.w + x;
            let dst_t = (input.h - 1 - y) * input.w + x;
            out.copy_part(dst_t, 0, input, src_t, input.d);
        }
    }
    out
}

pub fn transpose_xy(input: &Tensor) -> Tensor {
    let mut out = Tensor::zeros(input.w, input.h, input.d, input.int_mode());
    for y in 0..input.h {
        for x in 0..input.w {
            let src_t = y * input.w + x;
            let dst_t = x * input.h + y;
            out.copy_part(dst_t, 0, input, src_t, input.d);
        }
    }
    out
}

pub fn lstm_forward(layer: &LstmLayer, input: &Tensor) -> Result<Tensor, Error> {
    let ni = layer.ni;
    let ns = layer.ns;
    let na = layer.na;
    if input.d != ni {
        return Err(Error::Network(format!(
            "lstm expects depth {}, input has {}",
            ni, input.d
        )));
    }
    if na != ni + ns {
        return Err(Error::Network(format!(
            "lstm na {} != ni {} + ns {}",
            na, ni, ns
        )));
    }
    let int_mode = input.int_mode();
    let (out_h, out_w) = if layer.summarizing {
        (input.h, 1)
    } else {
        (input.h, input.w)
    };
    let mut out = Tensor::zeros(out_h, out_w, ns, int_mode);
    let mut src_f = vec![0f32; na];
    let mut src_i = vec![0i8; na];
    let mut state = vec![0f32; ns];
    let mut output = vec![0f32; ns];
    let mut gates = [
        vec![0f32; ns],
        vec![0f32; ns],
        vec![0f32; ns],
        vec![0f32; ns],
    ];
    for y in 0..input.h {
        state.fill(0.0);
        output.fill(0.0);
        for x in 0..input.w {
            let t = y * input.w + x;
            match input.step_view(t) {
                StepView::F32(s) => src_f[..ni].copy_from_slice(s),
                StepView::I8(s) => src_i[..ni].copy_from_slice(s),
            }
            if int_mode {
                for k in 0..ns {
                    src_i[ni + k] = quant127(output[k]);
                }
            } else {
                src_f[ni..na].copy_from_slice(&output);
            }
            let view = if int_mode {
                StepView::I8(&src_i)
            } else {
                StepView::F32(&src_f)
            };
            for (g, wm) in gates.iter_mut().zip(layer.gates.iter()) {
                dot(wm, &view, g)?;
            }
            for k in 0..ns {
                let ci = gates[0][k].tanh();
                let gi = sigmoid(gates[1][k]);
                let gf = sigmoid(gates[2][k]);
                let go = sigmoid(gates[3][k]);
                let s = (state[k] * gf + ci * gi).clamp(-STATE_CLIP, STATE_CLIP);
                state[k] = s;
                output[k] = s.tanh() * go;
            }
            if layer.summarizing {
                if x == input.w - 1 {
                    out.write_step(y, &output);
                }
            } else {
                out.write_step(t, &output);
            }
        }
    }
    Ok(out)
}

pub fn fc_forward(kind: FcKind, wm: &WeightMatrix, input: &Tensor) -> Result<Tensor, Error> {
    if wm.num_inputs() != input.d {
        return Err(Error::Network(format!(
            "fc expects depth {}, input has {}",
            wm.num_inputs(),
            input.d
        )));
    }
    let no = wm.rows;
    let softmax = matches!(kind, FcKind::Softmax | FcKind::SoftmaxNoCtc);
    let out_int = input.int_mode() && !softmax;
    let mut out = Tensor::zeros(input.h, input.w, no, out_int);
    let mut line = vec![0f32; no];
    for t in 0..input.steps() {
        dot(wm, &input.step_view(t), &mut line)?;
        match kind {
            FcKind::Tanh => {
                for v in line.iter_mut() {
                    *v = v.tanh();
                }
            }
            FcKind::Logistic => {
                for v in line.iter_mut() {
                    *v = sigmoid(*v);
                }
            }
            FcKind::Relu => {
                for v in line.iter_mut() {
                    *v = v.max(0.0);
                }
            }
            FcKind::Linear => {}
            FcKind::Softmax | FcKind::SoftmaxNoCtc => softmax_inplace(&mut line),
        }
        out.write_step(t, &line);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip(h: usize, w: usize, vals: &[f32]) -> Tensor {
        let mut t = Tensor::zeros(h, w, 1, false);
        if let Buf::F32(b) = &mut t.buf {
            b.copy_from_slice(vals);
        }
        t
    }

    #[test]
    fn maxpool_1px_wide_strip_does_not_collapse() {
        let vals: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let t = strip(48, 1, &vals);
        let out = maxpool_forward(&t, 3, 3).expect("degenerate strip must not error");
        assert_eq!((out.h, out.w, out.d), (16, 1, 1));
        if let Buf::F32(b) = &out.buf {
            assert_eq!(b[0], 2.0);
            assert_eq!(b[15], 47.0);
        } else {
            panic!("expected f32 output");
        }
    }

    #[test]
    fn maxpool_1px_tall_strip_does_not_collapse() {
        let vals: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let t = strip(1, 48, &vals);
        let out = maxpool_forward(&t, 3, 3).expect("degenerate strip must not error");
        assert_eq!((out.h, out.w, out.d), (1, 16, 1));
        if let Buf::F32(b) = &out.buf {
            assert_eq!(b[0], 2.0);
            assert_eq!(b[15], 47.0);
        } else {
            panic!("expected f32 output");
        }
    }

    #[test]
    fn maxpool_1x1_input_survives() {
        let t = strip(1, 1, &[5.0]);
        let out = maxpool_forward(&t, 3, 3).unwrap();
        assert_eq!((out.h, out.w, out.d), (1, 1, 1));
        if let Buf::F32(b) = &out.buf {
            assert_eq!(b[0], 5.0);
        }
    }

    #[test]
    fn maxpool_empty_input_errors() {
        let t = Tensor::zeros(0, 4, 1, false);
        assert!(maxpool_forward(&t, 3, 3).is_err());
    }
}
