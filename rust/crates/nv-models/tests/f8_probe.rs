#[test]
fn int8_row_quant_ops_exist_on_cuda_where_candle_f8_casts_do_not() {
    let device = match candle_core::Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: no cuda: {e}");
            return;
        }
    };
    let w = candle_core::Tensor::randn(0f32, 0.02, (64, 32), &device).unwrap();
    let scale = (w.abs().unwrap().max_keepdim(1).unwrap() / 127.0)
        .unwrap()
        .clamp(1e-12, 1e12)
        .unwrap();
    let q = w
        .broadcast_div(&scale)
        .unwrap()
        .round()
        .unwrap()
        .affine(1.0, 128.0)
        .unwrap()
        .clamp(0.0, 255.0)
        .unwrap()
        .to_dtype(candle_core::DType::U8)
        .unwrap();
    let back = q
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .affine(1.0, -128.0)
        .unwrap()
        .broadcast_mul(&scale)
        .unwrap();
    let err = ((&w - &back)
        ).unwrap()
        .abs()
        .unwrap()
        .max_keepdim(1)
        .unwrap()
        .max_keepdim(0)
        .unwrap()
        .reshape(1)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()[0];
    assert!(err < 0.001, "int8 row roundtrip err {err}");

    let f8 = w.to_dtype(candle_core::DType::F8E4M3);
    assert!(
        f8.is_err(),
        "candle CUDA f8 casts started working (kernel name cast_f32_f8e4m3 vs \
         cast_f32_f8_e4m3); consider reinstating fp8 vision tower storage"
    );
}
