#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use half::bf16;
use nv_layers::linear::Linear;

#[test]
fn stream_epoch_registry_bumps_and_reads() {
    let key = 0xdead_beef_usize;
    let e0 = nv_quant::stream_epoch(key);
    nv_quant::bump_stream_epoch(key);
    nv_quant::bump_stream_epoch(key);
    assert_eq!(nv_quant::stream_epoch(key), e0 + 2);
    assert_eq!(nv_quant::stream_epoch(key ^ 1), 0);
}

fn quantized_linear(device: &Device, k: usize, n: usize) -> anyhow::Result<Linear> {
    let cuda = match device {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("cuda device required"),
    };
    let runner = std::sync::Arc::new(std::sync::Mutex::new(
        nv_quant::nvfp4::Nvfp4GemmRunner::new(cuda.cuda_stream().clone())?,
    ));
    let w = Tensor::randn(0f32, 0.05, (n, k), device)?.to_dtype(DType::BF16)?;
    Linear::from_bf16_quantized_nvfp4_dev(&w, None, device, runner)
}

#[test]
#[ignore]
fn staging_evicts_on_epoch_bump_and_zeros_on_shrink() -> anyhow::Result<()> {
    if std::env::var("NV_STAGING_TEST").as_deref() != Ok("1") {
        panic!(
            "staging_evicts_on_epoch_bump_and_zeros_on_shrink: NV_STAGING_TEST != 1. This test is \
             already #[ignore]d, so asking for it by name or with --ignored IS the opt-in; the \
             extra env var only made `--ignored` report `1 passed` in 0.00s having run nothing. \
             It needs a CUDA device and 256x128 of synthetic weights -- no checkpoint. Set \
             NV_STAGING_TEST=1 to run it."
        );
    }
    let device = Device::new_cuda(0)?;
    let cuda = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = cuda.cuda_stream();
    let key = stream.cu_stream() as usize;
    let (k, n) = (256usize, 128usize);
    let lin = quantized_linear(&device, k, n)?;

    let x5 = Tensor::full(bf16::from_f32(0.5), (5usize, k), &device)?.to_dtype(DType::BF16)?;
    let _ = lin.forward(&x5)?;
    let (_e, hwm) = lin
        .nvfp4_staging_diag(key, nv_quant::nvfp4::MIN_TILE)
        .expect("staging entry after first forward");
    assert_eq!(hwm, 5, "high-water mark tracks logical rows");

    let x2 = Tensor::full(bf16::from_f32(0.5), (2usize, k), &device)?.to_dtype(DType::BF16)?;
    let y2a = lin.forward(&x2)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let (_e, hwm) = lin
        .nvfp4_staging_diag(key, nv_quant::nvfp4::MIN_TILE)
        .unwrap();
    assert_eq!(hwm, 2, "shrink path ran and reset the mark");

    let epoch_before = lin
        .nvfp4_staging_diag(key, nv_quant::nvfp4::MIN_TILE)
        .unwrap()
        .0;
    nv_quant::release_stream_resources(key);
    let y2b = lin.forward(&x2)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let epoch_after = lin
        .nvfp4_staging_diag(key, nv_quant::nvfp4::MIN_TILE)
        .unwrap()
        .0;
    assert!(
        epoch_after > epoch_before,
        "entry rebuilt under the new epoch after release_stream_resources"
    );
    assert_eq!(y2a, y2b, "eviction is value-preserving");
    Ok(())
}
