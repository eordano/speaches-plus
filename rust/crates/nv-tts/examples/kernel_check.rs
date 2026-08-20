use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use nv_layers::{RmsNorm, Rope, RopeConfig, RopeKind};

fn max_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let b = b.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let d = a.sub(&b)?.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?;
    Ok(d)
}

fn main() -> Result<()> {
    let gpu = Device::new_cuda(0)?;
    let cpu = Device::Cpu;

    let t = 9usize;
    let heads = 16usize;
    let kv = 8usize;
    let hd = 128usize;

    let q_host: Vec<f32> = (0..(t * heads * hd))
        .map(|i| ((i * 31 % 97) as f32 - 48.0) * 0.02)
        .collect();
    let k_host: Vec<f32> = (0..(t * kv * hd))
        .map(|i| ((i * 17 % 89) as f32 - 44.0) * 0.02)
        .collect();
    let pos_host: Vec<u32> = (0..t as u32).collect();

    let mk = |dev: &Device, dt: DType| -> Result<(Tensor, Tensor, Tensor)> {
        let q = Tensor::from_vec(q_host.clone(), (1, t, heads, hd), dev)?.to_dtype(dt)?;
        let k = Tensor::from_vec(k_host.clone(), (1, t, kv, hd), dev)?.to_dtype(dt)?;
        let p = Tensor::from_vec(pos_host.clone(), (1, t), dev)?;
        Ok((q, k, p))
    };

    let rope_cfg = RopeConfig {
        head_dim: hd,
        max_seq_len: 32768,
        base: 1_000_000.0,
        kind: RopeKind::Standard,
    };
    let rope_cpu = Rope::new(rope_cfg, &cpu)?;
    let rope_gpu = Rope::new(rope_cfg, &gpu)?;

    let (qc, kc, pc) = mk(&cpu, DType::F32)?;
    let (q_ref, k_ref) = rope_cpu.apply(&qc, &kc, &pc)?;

    let (qg32, kg32, pg) = mk(&gpu, DType::F32)?;
    let (qo, ko) = rope_gpu.apply(&qg32, &kg32, &pg)?;
    println!(
        "rope cuda f32 vs cpu f32: q={:.6} k={:.6}",
        max_diff(&qo, &q_ref)?,
        max_diff(&ko, &k_ref)?
    );

    let (qgb, kgb, pg) = mk(&gpu, DType::BF16)?;
    let (qo, ko) = rope_gpu.apply(&qgb, &kgb, &pg)?;
    println!(
        "rope cuda bf16 vs cpu f32: q={:.6} k={:.6}",
        max_diff(&qo, &q_ref)?,
        max_diff(&ko, &k_ref)?
    );

    let h = 1024usize;
    let w_host: Vec<f32> = (0..h)
        .map(|i| 1.0 + ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let x_host: Vec<f32> = (0..(t * h))
        .map(|i| ((i * 13 % 101) as f32 - 50.0) * 0.03)
        .collect();

    let norm_cpu = RmsNorm::new(Tensor::from_vec(w_host.clone(), h, &cpu)?, 1e-6);
    let x_cpu = Tensor::from_vec(x_host.clone(), (1, t, h), &cpu)?;
    let y_ref = norm_cpu.forward(&x_cpu)?;

    let norm_gpu32 = RmsNorm::new(Tensor::from_vec(w_host.clone(), h, &gpu)?, 1e-6);
    let x_gpu32 = Tensor::from_vec(x_host.clone(), (1, t, h), &gpu)?;
    println!(
        "rmsnorm cuda f32 vs cpu f32: {:.6}",
        max_diff(&norm_gpu32.forward(&x_gpu32)?, &y_ref)?
    );

    let norm_gpub = RmsNorm::new(
        Tensor::from_vec(w_host.clone(), h, &gpu)?.to_dtype(DType::BF16)?,
        1e-6,
    );
    let x_gpub = x_gpu32.to_dtype(DType::BF16)?;
    println!(
        "rmsnorm cuda bf16 vs cpu f32: {:.6}",
        max_diff(&norm_gpub.forward(&x_gpub)?, &y_ref)?
    );

    let norm_cpub = RmsNorm::new(
        Tensor::from_vec(w_host.clone(), h, &cpu)?.to_dtype(DType::BF16)?,
        1e-6,
    );
    let x_cpub = x_cpu.to_dtype(DType::BF16)?;
    println!(
        "rmsnorm cpu bf16 vs cpu f32: {:.6}",
        max_diff(&norm_cpub.forward(&x_cpub)?, &y_ref)?
    );

    let (qcb, kcb, pcb) = mk(&cpu, DType::BF16)?;
    let (qo, ko) = rope_cpu.apply(&qcb, &kcb, &pcb)?;
    println!(
        "rope cpu bf16 vs cpu f32: q={:.6} k={:.6}",
        max_diff(&qo, &q_ref)?,
        max_diff(&ko, &k_ref)?
    );

    use nv_layers::Linear;
    let out_f = 3072usize;
    let lw_host: Vec<f32> = (0..(out_f * h))
        .map(|i| ((i * 29 % 113) as f32 - 56.0) * 0.004)
        .collect();
    let lin_cpu = Linear::new(Tensor::from_vec(lw_host.clone(), (out_f, h), &cpu)?, None)?;
    let y_lin_ref = lin_cpu.forward(&x_cpu)?;
    let lin_gpu32 = Linear::new(Tensor::from_vec(lw_host.clone(), (out_f, h), &gpu)?, None)?;
    println!(
        "linear cuda f32 vs cpu f32: {:.6}",
        max_diff(&lin_gpu32.forward(&x_gpu32)?, &y_lin_ref)?
    );
    let lin_gpub = Linear::new(
        Tensor::from_vec(lw_host.clone(), (out_f, h), &gpu)?.to_dtype(DType::BF16)?,
        None,
    )?;
    println!(
        "linear cuda bf16 vs cpu f32: {:.6}",
        max_diff(&lin_gpub.forward(&x_gpub)?, &y_lin_ref)?
    );
    let lin_cpub = Linear::new(
        Tensor::from_vec(lw_host.clone(), (out_f, h), &cpu)?.to_dtype(DType::BF16)?,
        None,
    )?;
    println!(
        "linear cpu bf16 vs cpu f32: {:.6}",
        max_diff(&lin_cpub.forward(&x_cpub)?, &y_lin_ref)?
    );

    for (dt, name) in [(DType::F32, "f32"), (DType::BF16, "bf16")] {
        let buf = Tensor::zeros((1usize, 16, kv, hd), dt, &gpu)?;
        let src = Tensor::from_vec(k_host.clone(), (1, t, kv, hd), &gpu)?.to_dtype(dt)?;
        buf.slice_set(&src, 1, 0)?;
        let view = buf.narrow(1, 0, t)?;
        let d = max_diff(&view, &src)?;
        let buf2 = Tensor::zeros((1usize, 16, kv, hd), dt, &gpu)?;
        let one = src.narrow(1, 3, 1)?.contiguous()?;
        buf2.slice_set(&src.narrow(1, 0, 3)?.contiguous()?, 1, 0)?;
        buf2.slice_set(&one, 1, 3)?;
        let d2 = max_diff(&buf2.narrow(1, 3, 1)?, &one)?;
        println!("slice_set cuda {name}: full={d:.6} offset={d2:.6}");
    }

    let w128_host: Vec<f32> = (0..hd)
        .map(|i| 1.0 + ((i % 5) as f32 - 2.0) * 0.05)
        .collect();
    let n4_cpu = RmsNorm::new(Tensor::from_vec(w128_host.clone(), hd, &cpu)?, 1e-6);
    let y4_ref = n4_cpu.forward(&qc)?;
    let n4_gpub = RmsNorm::new(
        Tensor::from_vec(w128_host.clone(), hd, &gpu)?.to_dtype(DType::BF16)?,
        1e-6,
    );
    println!(
        "rmsnorm4d cuda bf16 vs cpu f32: {:.6}",
        max_diff(&n4_gpub.forward(&qgb)?, &y4_ref)?
    );
    let n4_cpub = RmsNorm::new(
        Tensor::from_vec(w128_host, hd, &cpu)?.to_dtype(DType::BF16)?,
        1e-6,
    );
    println!(
        "rmsnorm4d cpu bf16 vs cpu f32: {:.6}",
        max_diff(&n4_cpub.forward(&qcb)?, &y4_ref)?
    );

    Ok(())
}
