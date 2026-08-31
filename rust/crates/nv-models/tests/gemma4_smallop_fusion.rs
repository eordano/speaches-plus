#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use half::bf16;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};

extern "C" {
    fn nv_kernels_g4_qknorm_rope_vnorm_bf16(
        stream: *mut std::ffi::c_void,
        q_raw: *const u16,
        k_raw: *const u16,
        v_raw: *const u16,
        q_norm_w: *const u16,
        k_norm_w: *const u16,
        v_norm_w: *const u16,
        cos_tab: *const f32,
        sin_tab: *const f32,
        positions: *const i32,
        q_out: *mut u16,
        k_out: *mut u16,
        v_out: *mut u16,
        batch: i32,
        n_q: i32,
        n_kv: i32,
        hd: i32,
        eps: f32,
    ) -> i32;
}

fn lcg_bf16(n: usize, seed: &mut u64, scale: f32) -> Vec<bf16> {
    (0..n)
        .map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((*seed >> 33) as f64 / (1u64 << 31) as f64) as f32 - 0.5;
            bf16::from_f32(u * scale)
        })
        .collect()
}

fn bits(t: &Tensor) -> Vec<u16> {
    t.flatten_all()
        .unwrap()
        .to_vec1::<bf16>()
        .unwrap()
        .into_iter()
        .map(|x| x.to_bits())
        .collect()
}

fn eager_chain(
    q_raw: &Tensor,
    k_raw: &Tensor,
    v_raw: &Tensor,
    qn: &RmsNorm,
    kn: &RmsNorm,
    vn: &RmsNorm,
    rope: &Rope,
    pos: &Tensor,
    b: usize,
    n_q: usize,
    n_kv: usize,
    hd: usize,
) -> (Tensor, Tensor, Tensor) {
    let q = q_raw.reshape((1usize, b, n_q, hd)).unwrap();
    let q_normed = qn.forward(&q).unwrap();
    let k = k_raw.reshape((1usize, b, n_kv, hd)).unwrap();
    let k_normed = kn.forward(&k).unwrap();
    let v = v_raw.reshape((1usize, b, n_kv, hd)).unwrap();
    let v_normed = vn.forward(&v).unwrap();
    let (q_rot, k_rot) = rope.apply(&q_normed, &k_normed, pos).unwrap();
    (
        q_rot.contiguous().unwrap(),
        k_rot.contiguous().unwrap(),
        v_normed.contiguous().unwrap(),
    )
}

#[test]
fn fused_qknorm_rope_vnorm_kernel_matches_the_eager_norm_rope_chain_bitwise() {
    let device = match Device::new_cuda_with_stream(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP no cuda device: {e}");
            return;
        }
    };
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let mut seed = 0x9e3779b97f4a7c15u64;
    for &(b, n_q, n_kv, hd, partial) in
        &[(1usize, 16usize, 8usize, 128usize, false), (4, 16, 8, 128, false), (2, 8, 4, 256, true)]
    {
        let eps = 1e-6f64;
        let q_host = lcg_bf16(b * n_q * hd, &mut seed, 6.0);
        let k_host = lcg_bf16(b * n_kv * hd, &mut seed, 6.0);
        let v_host = lcg_bf16(b * n_kv * hd, &mut seed, 6.0);
        let qw_host = lcg_bf16(hd, &mut seed, 2.0);
        let kw_host = lcg_bf16(hd, &mut seed, 2.0);
        let vw_host = vec![bf16::from_f32(1.0); hd];
        let q_raw = Tensor::from_vec(q_host, (1usize, b, n_q * hd), &device).unwrap();
        let k_raw = Tensor::from_vec(k_host, (1usize, b, n_kv * hd), &device).unwrap();
        let v_raw = Tensor::from_vec(v_host, (1usize, b, n_kv * hd), &device).unwrap();
        let qn = RmsNorm::new(
            Tensor::from_vec(qw_host, hd, &device).unwrap().to_dtype(DType::BF16).unwrap(),
            eps,
        );
        let kn = RmsNorm::new(
            Tensor::from_vec(kw_host, hd, &device).unwrap().to_dtype(DType::BF16).unwrap(),
            eps,
        );
        let vn = RmsNorm::new(
            Tensor::from_vec(vw_host, hd, &device).unwrap().to_dtype(DType::BF16).unwrap(),
            eps,
        );
        let half_dim = hd / 2;
        let angles = if partial { half_dim / 2 } else { half_dim };
        let mut inv_freq = vec![0f32; half_dim];
        for (i, f) in inv_freq[..angles].iter_mut().enumerate() {
            *f = 1.0 / 10000f32.powf((i as f32 * 2.0) / (hd as f32));
        }
        let rope = Rope::from_inv_freq(
            RopeConfig { head_dim: hd, max_seq_len: 512, base: 10000.0, kind: RopeKind::Standard },
            &inv_freq,
            &device,
        )
        .unwrap();
        let positions: Vec<i32> = (0..b as i32).map(|i| 17 + 31 * i).collect();
        let pos_t = Tensor::from_vec(positions, b, &device).unwrap();

        let (q_ref, k_ref, v_ref) =
            eager_chain(&q_raw, &k_raw, &v_raw, &qn, &kn, &vn, &rope, &pos_t, b, n_q, n_kv, hd);

        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let mut q_out = unsafe { stream.alloc::<bf16>(b * n_q * hd).unwrap() };
        let mut k_out = unsafe { stream.alloc::<bf16>(b * n_kv * hd).unwrap() };
        let mut v_out = unsafe { stream.alloc::<bf16>(b * n_kv * hd).unwrap() };
        let rc = {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let ptr_bf16 = |t: &Tensor| -> u64 {
                let (st, l) = t.storage_and_layout();
                let c = match &*st {
                    candle_core::Storage::Cuda(c) => {
                        c.as_cuda_slice::<bf16>().unwrap().slice(l.start_offset()..)
                    }
                    _ => panic!("not cuda"),
                };
                let (p, _g) = c.device_ptr(&stream);
                p
            };
            let ptr_f32 = |t: &Tensor| -> u64 {
                let (st, l) = t.storage_and_layout();
                let c = match &*st {
                    candle_core::Storage::Cuda(c) => {
                        c.as_cuda_slice::<f32>().unwrap().slice(l.start_offset()..)
                    }
                    _ => panic!("not cuda"),
                };
                let (p, _g) = c.device_ptr(&stream);
                p
            };
            let ptr_i32 = |t: &Tensor| -> u64 {
                let (st, l) = t.storage_and_layout();
                let c = match &*st {
                    candle_core::Storage::Cuda(c) => {
                        c.as_cuda_slice::<i32>().unwrap().slice(l.start_offset()..)
                    }
                    _ => panic!("not cuda"),
                };
                let (p, _g) = c.device_ptr(&stream);
                p
            };
            let (pqo, _g1) = q_out.device_ptr_mut(&stream);
            let (pko, _g2) = k_out.device_ptr_mut(&stream);
            let (pvo, _g3) = v_out.device_ptr_mut(&stream);
            unsafe {
                nv_kernels_g4_qknorm_rope_vnorm_bf16(
                    stream.cu_stream() as *mut _,
                    ptr_bf16(&q_raw) as *const u16,
                    ptr_bf16(&k_raw) as *const u16,
                    ptr_bf16(&v_raw) as *const u16,
                    ptr_bf16(qn.weight_bf16()) as *const u16,
                    ptr_bf16(kn.weight_bf16()) as *const u16,
                    ptr_bf16(vn.weight_bf16()) as *const u16,
                    ptr_f32(&rope.cos().contiguous().unwrap()) as *const f32,
                    ptr_f32(&rope.sin().contiguous().unwrap()) as *const f32,
                    ptr_i32(&pos_t) as *const i32,
                    pqo as *mut u16,
                    pko as *mut u16,
                    pvo as *mut u16,
                    b as i32,
                    n_q as i32,
                    n_kv as i32,
                    hd as i32,
                    eps as f32,
                )
            }
        };
        assert_eq!(rc, 0, "fused kernel rc for shape b={b} n_q={n_q} n_kv={n_kv} hd={hd}");
        stream.synchronize().unwrap();
        let wrap = |s: cudarc::driver::CudaSlice<bf16>, heads: usize| {
            let st = candle_core::CudaStorage::wrap_cuda_slice(s, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(st),
                (1usize, b, heads, hd),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        let q_f = wrap(q_out, n_q);
        let k_f = wrap(k_out, n_kv);
        let v_f = wrap(v_out, n_kv);

        for (name, r, f) in [("q", &q_ref, &q_f), ("k", &k_ref, &k_f), ("v", &v_ref, &v_f)] {
            let rb = bits(r);
            let fb = bits(f);
            assert_eq!(rb.len(), fb.len());
            let mismatches = rb.iter().zip(&fb).filter(|(a, b)| a != b).count();
            let mut max_abs = 0f32;
            for (a, c) in rb.iter().zip(&fb) {
                let d = (bf16::from_bits(*a).to_f32() - bf16::from_bits(*c).to_f32()).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
            assert_eq!(
                mismatches,
                0,
                "{name} fused-vs-eager bitwise mismatch: {mismatches}/{} elems differ \
                 (max abs {max_abs:e}) at b={b} n_q={n_q} n_kv={n_kv} hd={hd} partial={partial}; \
                 NV_G4_QKNORM_ROPE_FUSED must stay default-off until this is zero",
                rb.len()
            );
        }
        eprintln!(
            "BITWISE-EQUAL fused qknorm_rope_vnorm b={b} n_q={n_q} n_kv={n_kv} hd={hd} partial={partial}"
        );
    }
}

