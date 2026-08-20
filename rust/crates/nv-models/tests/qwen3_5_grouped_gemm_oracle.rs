#![cfg(feature = "cuda")]

mod common;
use common::htod_f32;
use common::htod_u8;
use common::LcgInc1HalfCentered as Lcg;
use common::sf_swizzled;
mod hub_snapshot;

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Tensor};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_layers::moe_grouped::{forward_grouped, MoeGroupedWeights};
use nv_quant::nvfp4::{swizzle_scales, Nvfp4GemmRunner, Nvfp4Tensor, BLOCK_SIZE};

fn gated() -> bool {
    if std::env::var("NV_MOE_ORACLE_TEST").is_err() {
        eprintln!("SKIP: set NV_MOE_ORACLE_TEST=1 to run the grouped-MoE oracle arbitration");
        return false;
    }
    true
}

fn cuda() -> Option<Device> {
    match Device::new_cuda(0) {
        Ok(d) => Some(d),
        Err(e) => {
            hub_snapshot::precondition_absent(
                "qwen3_5_grouped_gemm_oracle",
                &format!("CUDA device 0 did not open: {e}"),
                "run on a machine with a CUDA device; this box has one (Blackwell sm_120)",
            );
            None
        }
    }
}

fn stream_of(device: &Device) -> Arc<CudaStream> {
    match device {
        Device::Cuda(d) => d.cuda_stream(),
        _ => unreachable!(),
    }
}

fn lattice_rows(n: usize, k: usize, seed: u64) -> Vec<Vec<f32>> {
    assert!(k % BLOCK_SIZE == 0);
    let mut rng = Lcg(seed);
    (0..n)
        .map(|_| {
            let mut row = vec![0f32; k];
            for b in 0..k / BLOCK_SIZE {
                let base = b * BLOCK_SIZE;
                let six_at = (rng.next_u32() as usize) % BLOCK_SIZE;
                for i in 0..BLOCK_SIZE {
                    let r = rng.next_u32();
                    row[base + i] = if i == six_at {
                        if r & 1 == 0 {
                            6.0
                        } else {
                            -6.0
                        }
                    } else if r & 2 == 0 {
                        0.0
                    } else {
                        let mag = [1.0f32, 2.0, 3.0, 4.0][(r >> 2) as usize % 4];
                        if r & 1 == 0 {
                            mag
                        } else {
                            -mag
                        }
                    };
                }
            }
            row
        })
        .collect()
}

fn quantize_exact(rows: &[Vec<f32>]) -> Nvfp4Tensor {
    let t = Nvfp4Tensor::quantize_rows(rows);
    let deq = t.dequantize();
    for (r, (a, b)) in rows.iter().zip(&deq).enumerate() {
        for (c, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                x, y,
                "lattice row {r} col {c} not lossless under NVFP4: {x} vs {y}"
            );
        }
    }
    t
}

fn oracle_bf16(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<bf16> {
    let m = a.len();
    let n = b.len();
    let mut out = Vec::with_capacity(m * n);
    for row in a {
        for col in b {
            let mut acc: i64 = 0;
            let mut abs: i64 = 0;
            for (x, y) in row.iter().zip(col) {
                let p = (*x as i64) * (*y as i64);
                acc += p;
                abs += p.abs();
            }
            assert!(abs < (1 << 24), "oracle abs-sum {abs} not exact in fp32");
            out.push(bf16::from_f32(acc as f32));
        }
    }
    out
}

fn ulp_stats(got: &[bf16], want: &[bf16]) -> (usize, u32, f64) {
    assert_eq!(got.len(), want.len());
    let mut mismatches = 0usize;
    let mut max_ulp = 0u32;
    let mut max_abs = 0f64;
    for (g, w) in got.iter().zip(want) {
        if g.to_bits() != w.to_bits() {
            mismatches += 1;
            let a = g.to_bits() as i32;
            let b = w.to_bits() as i32;

            let ord = |x: i32| if x & 0x8000 != 0 { -(x & 0x7FFF) } else { x };
            max_ulp = max_ulp.max((ord(a) - ord(b)).unsigned_abs());
            max_abs = max_abs.max((g.to_f32() as f64 - w.to_f32() as f64).abs());
        }
    }
    (mismatches, max_ulp, max_abs)
}

#[allow(deprecated)]
fn htod_i32(stream: &Arc<CudaStream>, v: &[i32]) -> CudaSlice<i32> {
    stream.memcpy_stod(v).expect("htod i32")
}

struct GroupedArgs<'a> {
    a_data: &'a [u8],
    a_sf: &'a [u8],
    b_data: &'a [u8],
    b_sf: &'a [u8],
    alphas: &'a [f32],
    expert_offsets: &'a [i32],
    sf_offsets: &'a [i32],
    problem_sizes: &'a [i32],
    active_ids: &'a [i32],
    n: i32,
    k: i32,
    m_total: usize,
}

fn run_grouped_kernel(
    stream: &Arc<CudaStream>,
    args: &GroupedArgs,
    decode_variant: bool,
) -> Result<Vec<bf16>, i32> {
    let a_dev = htod_u8(stream, args.a_data);
    let asf_dev = htod_u8(stream, args.a_sf);
    let b_dev = htod_u8(stream, args.b_data);
    let bsf_dev = htod_u8(stream, args.b_sf);
    let al_dev = htod_f32(stream, args.alphas);
    let eo_dev = htod_i32(stream, args.expert_offsets);
    let sfo_dev = htod_i32(stream, args.sf_offsets);
    let ps_dev = htod_i32(stream, args.problem_sizes);
    let aid_dev = htod_i32(stream, args.active_ids);
    let mut ms: CudaSlice<u8> = unsafe { stream.alloc::<u8>(128 * 1024).unwrap() };
    let mut ws: CudaSlice<u8> = unsafe { stream.alloc::<u8>(16 * 1024 * 1024).unwrap() };
    let mut d: CudaSlice<bf16> = unsafe {
        stream
            .alloc::<bf16>(args.m_total * args.n as usize)
            .unwrap()
    };

    {
        let (ap, _g1) = a_dev.device_ptr(stream);
        let (asp, _g2) = asf_dev.device_ptr(stream);
        let (bp, _g3) = b_dev.device_ptr(stream);
        let (bsp, _g4) = bsf_dev.device_ptr(stream);
        let (alp, _g5) = al_dev.device_ptr(stream);
        let (eop, _g6) = eo_dev.device_ptr(stream);
        let (sfp, _g7) = sfo_dev.device_ptr(stream);
        let (psp, _g8) = ps_dev.device_ptr(stream);
        let (aip, _g9) = aid_dev.device_ptr(stream);
        let (msp, _g10) = ms.device_ptr_mut(stream);
        let (wsp, _g11) = ws.device_ptr_mut(stream);
        let (dp, _g12) = d.device_ptr_mut(stream);

        let f = if decode_variant {
            nv_kernels::cuda::cutlass_moe_grouped_fp4_gemm_sm120_bf16_decode
        } else {
            nv_kernels::cuda::cutlass_moe_grouped_fp4_gemm_sm120_bf16
        };
        unsafe {
            f(
                stream.cu_stream() as *mut c_void,
                ap as *const c_void,
                asp as *const c_void,
                bp as *const c_void,
                bsp as *const c_void,
                alp as *const f32,
                dp as *mut c_void,
                eop as *const i32,
                sfp as *const i32,
                psp as *const i32,
                aip as *const i32,
                args.n,
                args.k,
                args.active_ids.len() as i32,
                args.k as i64,
                args.k as i64,
                args.n as i64,
                msp as *mut c_void,
                128 * 1024,
                wsp as *mut c_void,
                16 * 1024 * 1024,
            )?;
        }
    }
    stream.synchronize().expect("sync");
    #[allow(deprecated)]
    let out = stream.memcpy_dtov(&d).expect("dtoh");
    Ok(out)
}

#[test]
#[ignore]
fn rung0_device_quantizer_lossless_and_layout_matches() {
    if !gated() {
        return;
    }
    let Some(device) = cuda() else { return };
    let stream = stream_of(&device);

    let (m, k) = (128usize, 2048usize);
    let rows = lattice_rows(m, k, 0xA11CE);
    let host_q = quantize_exact(&rows);
    let host_sf = sf_swizzled(&host_q);

    let x_bf: Vec<bf16> = rows
        .iter()
        .flat_map(|r| r.iter().map(|v| bf16::from_f32(*v)))
        .collect();
    #[allow(deprecated)]
    let x_dev = stream.memcpy_stod(&x_bf).expect("htod bf16");
    let mut fp4: CudaSlice<u8> = unsafe { stream.alloc::<u8>(m * k / 2).unwrap() };
    let mut sf: CudaSlice<u8> = unsafe { stream.alloc::<u8>(host_sf.len()).unwrap() };
    let globals = htod_f32(&stream, &[1.0]);

    let rc = {
        let (xp, _g1) = x_dev.device_ptr(&stream);
        let (fp, _g2) = fp4.device_ptr_mut(&stream);
        let (sp, _g3) = sf.device_ptr_mut(&stream);
        let (gp, _g4) = globals.device_ptr(&stream);
        unsafe {
            nv_kernels::cuda::quantize_nvfp4_bf16_per_expert(
                stream.cu_stream() as *mut c_void,
                xp as *const u16,
                fp as *mut u8,
                sp as *mut u8,
                gp as *const f32,
                m as i32,
                m as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "quantize_nvfp4_bf16_per_expert rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let dev_fp4 = stream.memcpy_dtov(&fp4).unwrap();
    #[allow(deprecated)]
    let dev_sf = stream.memcpy_dtov(&sf).unwrap();

    let data_eq = dev_fp4 == host_q.data;
    let sf_eq = dev_sf == host_sf;
    eprintln!("[rung0] packed bytes equal: {data_eq}  swizzled sf bytes equal: {sf_eq}");
    if !(data_eq && sf_eq) {
        let dev_vals = nv_quant::nvfp4::dequantize_packed_swizzled(&dev_fp4, &dev_sf, m, k, 1.0);
        let mut worst = 0f32;
        for (i, v) in dev_vals.iter().enumerate() {
            worst = worst.max((v - rows[i / k][i % k]).abs());
        }
        eprintln!("[rung0] decoded max abs diff vs lattice: {worst}");
        assert_eq!(
            worst, 0.0,
            "device quantizer is not lossless on the lattice"
        );
    }
}

#[test]
#[ignore]
fn rung1_grouped_kernel_vs_exact_oracle_single_expert() {
    if !gated() {
        return;
    }
    let Some(device) = cuda() else { return };
    let stream = stream_of(&device);

    for (label, n, k) in [("gate/up", 512usize, 2048usize), ("down", 2048, 512)] {
        let m = 128usize;
        let a_rows = lattice_rows(m, k, 0xB0B0 ^ n as u64);
        let b_rows = lattice_rows(n, k, 0xC0C0 ^ k as u64);
        let a_q = quantize_exact(&a_rows);
        let b_q = quantize_exact(&b_rows);
        let a_sf = sf_swizzled(&a_q);
        let b_sf = sf_swizzled(&b_q);
        let want = oracle_bf16(&a_rows, &b_rows);

        let args = GroupedArgs {
            a_data: &a_q.data,
            a_sf: &a_sf,
            b_data: &b_q.data,
            b_sf: &b_sf,
            alphas: &[1.0],
            expert_offsets: &[0],
            sf_offsets: &[0],
            problem_sizes: &[m as i32, n as i32, k as i32],
            active_ids: &[0],
            n: n as i32,
            k: k as i32,
            m_total: m,
        };

        let got = run_grouped_kernel(&stream, &args, false).expect("grouped kernel rc");
        let (mis, max_ulp, max_abs) = ulp_stats(&got, &want);
        eprintln!(
            "[rung1 prefill {label}] m={m} n={n} k={k}: mismatches {mis}/{} max_ulp {max_ulp} max_abs {max_abs}",
            got.len()
        );
        assert!(
            max_ulp <= 1,
            "grouped kernel ({label}) deviates from the exact oracle by {max_ulp} ulp \
             (mismatches {mis}, max_abs {max_abs}) -- kernel defect, not quantization noise"
        );

        match run_grouped_kernel(&stream, &args, true) {
            Ok(got_dec) => {
                let (mis, max_ulp, max_abs) = ulp_stats(&got_dec, &want);
                eprintln!(
                    "[rung1 decode-tile {label}] mismatches {mis}/{} max_ulp {max_ulp} max_abs {max_abs}",
                    got_dec.len()
                );
                assert!(
                    max_ulp <= 1,
                    "decode-tile kernel ({label}) deviates by {max_ulp} ulp (mismatches {mis})"
                );
            }
            Err(rc) => eprintln!(
                "[rung1 decode-tile {label}] kernel rc={rc} (variant unavailable at this shape)"
            ),
        }

        let mut b_sf_bad = b_sf.clone();
        let poisoned = b_sf_bad.iter().position(|&s| s != 0).expect("nonzero sf");
        b_sf_bad[poisoned] = nv_quant::nvfp4::encode_ue4m3(2.0);
        let args_bad = GroupedArgs {
            b_sf: &b_sf_bad,
            ..args
        };
        let got_bad = run_grouped_kernel(&stream, &args_bad, false).expect("poisoned rc");
        let (mis_bad, ulp_bad, abs_bad) = ulp_stats(&got_bad, &want);
        eprintln!(
            "[rung1 control {label}] poisoned sf -> mismatches {mis_bad} max_ulp {ulp_bad} max_abs {abs_bad}"
        );
        assert!(
            mis_bad > 0 && ulp_bad > 1,
            "negative control failed: a poisoned block scale was not detected \
             ({label}: mismatches {mis_bad}, max_ulp {ulp_bad}) -- the oracle has no power"
        );
    }
}

#[test]
#[ignore]
fn rung2_grouped_kernel_vs_oracle_four_experts_permuted() {
    if !gated() {
        return;
    }
    let Some(device) = cuda() else { return };
    let stream = stream_of(&device);

    let (n, k, e_count, tile) = (512usize, 2048usize, 4usize, 128usize);
    let m_total = e_count * tile;
    let a_rows = lattice_rows(m_total, k, 0xD1D1);
    let a_q = quantize_exact(&a_rows);
    let a_sf = sf_swizzled(&a_q);

    let experts: Vec<Vec<Vec<f32>>> = (0..e_count)
        .map(|e| lattice_rows(n, k, 0xE0E0 + e as u64 * 7919))
        .collect();
    let mut b_data = Vec::new();
    let mut b_sf = Vec::new();
    for rows in &experts {
        let q = quantize_exact(rows);
        b_data.extend_from_slice(&q.data);
        b_sf.extend_from_slice(&sf_swizzled(&q));
    }

    let active_ids: Vec<i32> = vec![2, 0, 3, 1];
    let expert_offsets: Vec<i32> = (0..e_count).map(|i| (i * tile) as i32).collect();
    let sf_offsets = expert_offsets.clone();
    let mut problem_sizes = Vec::new();
    for _ in 0..e_count {
        problem_sizes.extend_from_slice(&[tile as i32, n as i32, k as i32]);
    }

    let args = GroupedArgs {
        a_data: &a_q.data,
        a_sf: &a_sf,
        b_data: &b_data,
        b_sf: &b_sf,
        alphas: &[1.0; 4],
        expert_offsets: &expert_offsets,
        sf_offsets: &sf_offsets,
        problem_sizes: &problem_sizes,
        active_ids: &active_ids,
        n: n as i32,
        k: k as i32,
        m_total,
    };
    let got = run_grouped_kernel(&stream, &args, false).expect("grouped kernel rc");

    for (g, &eid) in active_ids.iter().enumerate() {
        let a_slice = &a_rows[g * tile..(g + 1) * tile];
        let want = oracle_bf16(a_slice, &experts[eid as usize]);
        let got_g = &got[g * tile * n..(g + 1) * tile * n];
        let (mis, max_ulp, max_abs) = ulp_stats(got_g, &want);
        eprintln!(
            "[rung2] group {g} -> expert {eid}: mismatches {mis}/{} max_ulp {max_ulp} max_abs {max_abs}",
            want.len()
        );
        assert!(
            max_ulp <= 1,
            "group {g} (expert {eid}) deviates from its expert's oracle by {max_ulp} ulp -- \
             grouping/indexing defect"
        );
    }
}

fn grouped_weights_from(
    stream: &Arc<CudaStream>,
    runner: &Arc<Mutex<Nvfp4GemmRunner>>,
    experts: &[(Nvfp4Tensor, Nvfp4Tensor, Nvfp4Tensor)],
    hidden: usize,
    inter: usize,
) -> MoeGroupedWeights {
    let e = experts.len();
    let mut gate_p = Vec::new();
    let mut gate_s = Vec::new();
    let mut up_p = Vec::new();
    let mut up_s = Vec::new();
    let mut down_p = Vec::new();
    let mut down_s = Vec::new();
    for (g, u, d) in experts {
        gate_p.extend_from_slice(&g.data);
        gate_s.extend_from_slice(&sf_swizzled(g));
        up_p.extend_from_slice(&u.data);
        up_s.extend_from_slice(&sf_swizzled(u));
        down_p.extend_from_slice(&d.data);
        down_s.extend_from_slice(&sf_swizzled(d));
    }
    let ones = vec![1.0f32; e];
    MoeGroupedWeights {
        num_experts: e,
        hidden_size: hidden,
        intermediate_size: inter,
        gate_w: htod_u8(stream, &gate_p),
        gate_w_scales: htod_u8(stream, &gate_s),
        gate_alphas: htod_f32(stream, &ones),
        gate_a_stride_elems: hidden as i64,
        gate_b_stride_elems: hidden as i64,
        gate_c_stride_elems: inter as i64,
        up_w: htod_u8(stream, &up_p),
        up_w_scales: htod_u8(stream, &up_s),
        up_alphas: htod_f32(stream, &ones),
        down_w: htod_u8(stream, &down_p),
        down_w_scales: htod_u8(stream, &down_s),
        down_alphas: htod_f32(stream, &ones),
        down_a_stride_elems: inter as i64,
        down_b_stride_elems: inter as i64,
        down_c_stride_elems: hidden as i64,
        runner: runner.clone(),
        input_globals_gate_up: htod_f32(stream, &ones),
        input_globals_down: htod_f32(stream, &ones),
        input_globals_gate_up_host: ones.clone(),
        input_globals_down_host: ones,
    }
}

fn rand_expert(seed: u64, hidden: usize, inter: usize) -> (Nvfp4Tensor, Nvfp4Tensor, Nvfp4Tensor) {
    let mut rng = Lcg(seed);
    let mk = |n: usize, k: usize, rng: &mut Lcg| {
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..k).map(|_| rng.next_f32()).collect())
            .collect();
        Nvfp4Tensor::quantize_rows(&rows)
    };
    (
        mk(inter, hidden, &mut rng),
        mk(inter, hidden, &mut rng),
        mk(hidden, inter, &mut rng),
    )
}

fn max_rel_diff(a: &[f32], b: &[f32]) -> f64 {
    let scale = a
        .iter()
        .fold(0f64, |acc, v| acc.max((*v as f64).abs()))
        .max(1e-6);
    a.iter()
        .zip(b)
        .fold(0f64, |acc, (x, y)| acc.max((*x as f64 - *y as f64).abs()))
        / scale
}

fn forward_to_vec(
    w: &MoeGroupedWeights,
    x: &Tensor,
    topk_ids: &[u32],
    topk_weights: &[f32],
    n_tokens: usize,
    k: usize,
    device: &Device,
) -> Vec<f32> {
    forward_grouped(w, w, x, topk_ids, topk_weights, n_tokens, k, device)
        .expect("forward_grouped")
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

#[test]
#[ignore]
fn rung3_forward_grouped_routing_invariants() {
    if !gated() {
        return;
    }
    let Some(device) = cuda() else { return };
    let stream = stream_of(&device);
    let runner = Arc::new(Mutex::new(
        Nvfp4GemmRunner::new(stream.clone()).expect("nvfp4 runner"),
    ));

    let (hidden, inter) = (2048usize, 512usize);
    let n_tokens = 16usize;

    let mut rng = Lcg(0xF00D);
    let x_f: Vec<f32> = (0..n_tokens * hidden).map(|_| rng.next_f32()).collect();
    let x = Tensor::from_vec(x_f, (n_tokens, hidden), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let proto = rand_expert(0xAB, hidden, inter);
    let clone_of = |t: &Nvfp4Tensor| Nvfp4Tensor {
        data: t.data.clone(),
        scales: t.scales.clone(),
        rows: t.rows,
        cols: t.cols,
    };
    let identical: Vec<_> = (0..8)
        .map(|_| (clone_of(&proto.0), clone_of(&proto.1), clone_of(&proto.2)))
        .collect();
    let w8 = grouped_weights_from(&stream, &runner, &identical, hidden, inter);
    let w1 = grouped_weights_from(
        &stream,
        &runner,
        &identical[..1]
            .iter()
            .map(|(a, b, c)| (clone_of(a), clone_of(b), clone_of(c)))
            .collect::<Vec<_>>(),
        hidden,
        inter,
    );

    let mut ids = Vec::with_capacity(n_tokens * 2);
    let mut wts = Vec::with_capacity(n_tokens * 2);
    for t in 0..n_tokens {
        let a = (rng.next_u32() % 8) as u32;
        let b = (a + 1 + rng.next_u32() % 7) % 8;
        ids.extend_from_slice(&[a, b]);
        wts.extend_from_slice(&[0.5, 0.5]);
        let _ = t;
    }
    let out_multi = forward_to_vec(&w8, &x, &ids, &wts, n_tokens, 2, &device);
    let ids1: Vec<u32> = vec![0; n_tokens];
    let wts1: Vec<f32> = vec![1.0; n_tokens];
    let out_single = forward_to_vec(&w1, &x, &ids1, &wts1, n_tokens, 1, &device);
    let d = max_rel_diff(&out_single, &out_multi);
    eprintln!("[rung3a] identical-experts collapse: max_rel_diff {d:.3e}");
    assert!(
        d < 1e-5,
        "identical-expert routing changed the output by {d:.3e} rel -- routing/scatter defect"
    );

    let distinct: Vec<_> = (0..4)
        .map(|e| rand_expert(0x100 + e as u64, hidden, inter))
        .collect();
    let perm = [2usize, 0, 3, 1];
    let mut permuted: Vec<Option<(Nvfp4Tensor, Nvfp4Tensor, Nvfp4Tensor)>> =
        (0..4).map(|_| None).collect();
    for (old, &new) in perm.iter().enumerate() {
        let (a, b, c) = &distinct[old];
        permuted[new] = Some((clone_of(a), clone_of(b), clone_of(c)));
    }
    let permuted: Vec<_> = permuted.into_iter().map(|o| o.unwrap()).collect();

    let w_orig = grouped_weights_from(&stream, &runner, &distinct, hidden, inter);
    let w_perm = grouped_weights_from(&stream, &runner, &permuted, hidden, inter);

    let mut ids_o = Vec::new();
    let mut wts2 = Vec::new();
    for _ in 0..n_tokens {
        let a = (rng.next_u32() % 4) as u32;
        let b = (a + 1 + rng.next_u32() % 3) % 4;
        ids_o.extend_from_slice(&[a, b]);
        wts2.extend_from_slice(&[0.5, 0.5]);
    }
    let ids_p: Vec<u32> = ids_o.iter().map(|&e| perm[e as usize] as u32).collect();

    let out_o = forward_to_vec(&w_orig, &x, &ids_o, &wts2, n_tokens, 2, &device);
    let out_p = forward_to_vec(&w_perm, &x, &ids_p, &wts2, n_tokens, 2, &device);
    let d = max_rel_diff(&out_o, &out_p);
    eprintln!("[rung3b] expert-permutation commutation: max_rel_diff {d:.3e}");
    assert!(
        d < 1e-5,
        "permuting experts changed the output by {d:.3e} rel -- expert indexing defect"
    );
}

#[test]
#[ignore]
fn rung4_host_runner_vs_exact_oracle() {
    if !gated() {
        return;
    }
    let Some(device) = cuda() else { return };
    let stream = stream_of(&device);
    let mut runner = Nvfp4GemmRunner::new(stream.clone()).expect("nvfp4 runner");

    for (label, n, k) in [("gate/up", 512usize, 2048usize), ("down", 2048, 512)] {
        let m = 128usize;
        let a_rows = lattice_rows(m, k, 0xB0B0 ^ n as u64);
        let b_rows = lattice_rows(n, k, 0xC0C0 ^ k as u64);
        let a_q = quantize_exact(&a_rows);
        let b_q = quantize_exact(&b_rows);
        let a_sf = sf_swizzled(&a_q);
        let b_sf = sf_swizzled(&b_q);
        let want = oracle_bf16(&a_rows, &b_rows);

        let a_dev = htod_u8(&stream, &a_q.data);
        let asf_dev = htod_u8(&stream, &a_sf);
        let b_dev = htod_u8(&stream, &b_q.data);
        let bsf_dev = htod_u8(&stream, &b_sf);
        let alpha_dev = htod_f32(&stream, &[1.0]);
        let mut d: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(m * n).unwrap() };

        runner
            .matmul_scaled_alpha_dev(
                &a_dev, &asf_dev, &b_dev, &bsf_dev, &mut d, m as u64, n as u64, k as u64,
                &alpha_dev, 1.0,
            )
            .expect("host runner gemm");
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        let got = stream.memcpy_dtov(&d).unwrap();
        let (mis, max_ulp, max_abs) = ulp_stats(&got, &want);
        eprintln!(
            "[rung4 {label}] m={m} n={n} k={k}: mismatches {mis}/{} max_ulp {max_ulp} max_abs {max_abs}",
            got.len()
        );
        assert!(
            max_ulp <= 1,
            "host-path GEMM ({label}) deviates from the exact oracle by {max_ulp} ulp \
             (mismatches {mis}, max_abs {max_abs})"
        );
    }
}
