use nv_ocr::lstm::{
    conv_forward, dot, fc_forward, matvec_f32, matvec_i8, maxpool_forward, reverse_x, reverse_y,
    softmax_inplace, transpose_xy, Buf, StepView, Tensor,
};
use nv_ocr::vgsl::{FcKind, WeightMatrix, Weights};

fn f32_tensor(h: usize, w: usize, d: usize, data: Vec<f32>) -> Tensor {
    assert_eq!(data.len(), h * w * d);
    Tensor {
        h,
        w,
        d,
        buf: Buf::F32(data),
    }
}

fn i8_tensor(h: usize, w: usize, d: usize, data: Vec<i8>) -> Tensor {
    assert_eq!(data.len(), h * w * d);
    Tensor {
        h,
        w,
        d,
        buf: Buf::I8(data),
    }
}

fn f32_data(t: &Tensor) -> &[f32] {
    match &t.buf {
        Buf::F32(v) => v,
        _ => panic!("expected f32"),
    }
}

fn i8_data(t: &Tensor) -> &[i8] {
    match &t.buf {
        Buf::I8(v) => v,
        _ => panic!("expected i8"),
    }
}

#[test]
fn matvec_f32_applies_bias() {
    let w = vec![1.0, 2.0, 0.5, -1.0, 0.0, 3.0];
    let u = vec![10.0, 100.0];
    let mut out = vec![0.0; 2];
    matvec_f32(2, 3, &w, &u, &mut out);
    assert_eq!(out[0], 10.0 + 200.0 + 0.5);
    assert_eq!(out[1], -10.0 + 3.0);
}

#[test]
fn matvec_i8_matches_tesseract_semantics() {
    let w: Vec<i8> = vec![64, -32, 10];
    let scales = vec![(0.02 / 127.0) as f32];
    let u: Vec<i8> = vec![100, -50];
    let mut out = vec![0.0; 1];
    matvec_i8(1, 3, &w, &scales, &u, &mut out);
    assert!((out[0] - 1.4598425).abs() < 1e-5, "got {}", out[0]);
}

#[test]
fn dot_rejects_mode_mismatch() {
    let wm = WeightMatrix {
        rows: 1,
        cols: 2,
        weights: Weights::Float(vec![1.0, 0.0]),
    };
    let mut out = vec![0.0];
    assert!(dot(&wm, &StepView::I8(&[1]), &mut out).is_err());
    assert!(dot(&wm, &StepView::F32(&[1.0]), &mut out).is_ok());
}

#[test]
fn conv_gathers_x_major_then_y_with_clamped_border() {
    let input = f32_tensor(2, 2, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let out = conv_forward(&input, 1, 1);
    assert_eq!(out.h, 2);
    assert_eq!(out.w, 2);
    assert_eq!(out.d, 9);
    let d = f32_data(&out);
    assert_eq!(&d[0..9], &[1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 2.0, 2.0, 4.0]);
    let t3 = &d[3 * 9..4 * 9];
    assert_eq!(t3, &[1.0, 3.0, 3.0, 2.0, 4.0, 4.0, 2.0, 4.0, 4.0]);
}

#[test]
fn conv_preserves_int8_raw_values() {
    let input = i8_tensor(1, 2, 1, vec![100, -100]);
    let out = conv_forward(&input, 1, 0);
    let d = i8_data(&out);
    assert_eq!(&d[0..3], &[100, 100, -100]);
    assert_eq!(&d[3..6], &[100, -100, -100]);
}

#[test]
fn maxpool_floors_output_dims_and_takes_max() {
    let mut data = vec![0.0f32; 36];
    for (i, v) in data.iter_mut().enumerate() {
        *v = i as f32;
    }
    let input = f32_tensor(6, 6, 1, data);
    let out = maxpool_forward(&input, 3, 3).unwrap();
    assert_eq!((out.h, out.w), (2, 2));
    assert_eq!(f32_data(&out), &[14.0, 17.0, 32.0, 35.0]);
    let odd = f32_tensor(1, 7, 1, vec![1.0; 7]);
    let out = maxpool_forward(&odd, 3, 1).unwrap();
    assert_eq!(out.w, 2);
}

#[test]
fn maxpool_int8() {
    let input = i8_tensor(1, 4, 1, vec![-5, 3, 100, -100]);
    let out = maxpool_forward(&input, 2, 1).unwrap();
    assert_eq!(i8_data(&out), &[3, 100]);
}

#[test]
fn reverse_and_transpose_roundtrip() {
    let input = f32_tensor(2, 3, 2, (0..12).map(|i| i as f32).collect());
    let rx = reverse_x(&input);
    assert_eq!(f32_data(&rx)[0..2], [4.0, 5.0]);
    let back = reverse_x(&rx);
    assert_eq!(f32_data(&back), f32_data(&input));
    let ry = reverse_y(&input);
    assert_eq!(f32_data(&ry)[0..2], [6.0, 7.0]);
    assert_eq!(f32_data(&reverse_y(&ry)), f32_data(&input));
    let tr = transpose_xy(&input);
    assert_eq!((tr.h, tr.w, tr.d), (3, 2, 2));
    assert_eq!(f32_data(&tr)[2..4], [6.0, 7.0]);
    let back = transpose_xy(&tr);
    assert_eq!(f32_data(&back), f32_data(&input));
}

#[test]
fn fc_softmax_produces_float_distribution_from_int_input() {
    let wm = WeightMatrix {
        rows: 3,
        cols: 3,
        weights: Weights::Int8 {
            w: vec![127, 0, 0, 0, 127, 0, 0, 0, 0],
            scales: vec![1.0 / 127.0; 3],
        },
    };
    let input = i8_tensor(1, 2, 2, vec![127, 0, 0, 127]);
    let out = fc_forward(FcKind::Softmax, &wm, &input).unwrap();
    assert!(!out.int_mode());
    let d = f32_data(&out);
    for t in 0..2 {
        let row = &d[t * 3..(t + 1) * 3];
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
    assert!(d[0] > d[1] && d[0] > d[2]);
    assert!(d[4] > d[3] && d[4] > d[5]);
}

#[test]
fn fc_tanh_int_output_requantizes() {
    let wm = WeightMatrix {
        rows: 1,
        cols: 2,
        weights: Weights::Int8 {
            w: vec![127, 0],
            scales: vec![10.0 / 127.0],
        },
    };
    let input = i8_tensor(1, 1, 1, vec![127]);
    let out = fc_forward(FcKind::Tanh, &wm, &input).unwrap();
    assert!(out.int_mode());
    assert_eq!(i8_data(&out), &[127]);
}

#[test]
fn softmax_inplace_normalizes() {
    let mut v = vec![1.0, 2.0, 3.0];
    softmax_inplace(&mut v);
    let sum: f32 = v.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(v[2] > v[1] && v[1] > v[0]);
}

#[test]
fn tensor_write_step_quantizes_half_away_from_zero() {
    let mut t = Tensor::zeros(1, 1, 2, true);
    t.write_step(0, &[0.5, -2.0]);
    assert_eq!(i8_data(&t), &[64, -127]);
}
