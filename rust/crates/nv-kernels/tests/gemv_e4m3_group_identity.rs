#![cfg(feature = "cuda")]

mod common;
use common::LcgMask23TwoSided as Lcg;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;

pub const A_ROW_OF_THE_GROUP_KERNEL_WALKS_THE_SAME_SIXTEEN_BYTE_CHUNKS_IN_THE_SAME_ORDER_AS_THE_UNTILED_KERNEL_BECAUSE_THE_K_TILE_IS_A_WHOLE_NUMBER_OF_WARP_STRIDES: &str =
    "lane l accumulates chunks l, l+32, l+64, ... inside each tile and the tiles run in \
     increasing k, so concatenating them reproduces the untiled lane sequence exactly; the \
     per-row butterfly reduction and the row_scale multiply are unchanged, so identity is a \
     property of the loop shape and not of a tolerance";

pub const THE_GROUP_KERNEL_IS_CERTIFIED_AGAINST_THE_ROUTE_IT_REPLACES_NOT_AGAINST_A_HOST_REFERENCE_BECAUSE_ONLY_THE_FORMER_IS_THE_BASIS_THE_SERVED_BYTES_CAME_FROM: &str =
    "gemv_e4m3_mk at the same (n, k, m) is the basis; a host f64 dot product would certify \
     accuracy and say nothing about whether the served bytes moved";

fn alloc_shape(
    stream: &Arc<CudaStream>,
    rng: &mut Lcg,
    n: usize,
    k: usize,
    m: usize,
) -> (CudaSlice<u8>, CudaSlice<f32>, CudaSlice<u16>) {
    let wq: Vec<u8> = (0..n * k).map(|_| (rng.next_u32() & 0xff) as u8).collect();
    let row_scale: Vec<f32> = (0..n).map(|_| 0.01 + rng.next_f32().abs() * 0.01).collect();
    let x = rng.bf16_words(m * k, 1.0);
    #[allow(deprecated)]
    let dw = stream.clone_htod(&wq).unwrap();
    #[allow(deprecated)]
    let ds = stream.clone_htod(&row_scale).unwrap();
    #[allow(deprecated)]
    let dx = stream.clone_htod(&x).unwrap();
    (dw, ds, dx)
}

#[test]
#[ignore = "real-shape kernel certificate, cuda-gated; set NV_E4M3_GROUP_TEST=1 -- every output bf16 of gemv_e4m3_group must equal gemv_e4m3_mk's at the same (n, k, m), on the shapes where the shipped route still runs"]
fn gemv_e4m3_group_rows_are_bit_identical_to_the_untiled_gemv_e4m3_mk_wherever_that_route_still_runs()
{
    if std::env::var("NV_E4M3_GROUP_TEST").as_deref() != Ok("1") {
        panic!(
            "NV_E4M3_GROUP_TEST is not 1, so this suite executed nothing. It is already \
             #[ignore]d, which makes --ignored the opt-in; a second env gate behind that can only \
             make an --ignored run print `1 passed` for a certificate nobody checked. A suite \
             whose precondition is absent must fail or be ignored, never return green."
        );
    }
    let ctx = CudaContext::new(0).expect("cuda device 0");
    let stream = ctx.default_stream();
    let mut rng = Lcg(0x243f6a8885a308d3);
    let mut cells = 0usize;
    let mut ran = 0usize;
    for (n, k) in [(1024usize, 5120usize), (2048, 2048), (512, 1024), (256, 512)] {
        for m in 1usize..=16 {
            for r in [1i32, 2, 4] {
                let (dw, ds, dx) = alloc_shape(&stream, &mut rng, n, k, m);
                let mut y_ref: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();
                let mut y_grp: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();
                let rc_ref = {
                    let (pw, _a) = dw.device_ptr(&stream);
                    let (ps, _b) = ds.device_ptr(&stream);
                    let (px, _c) = dx.device_ptr(&stream);
                    let (py, _d) = y_ref.device_ptr_mut(&stream);
                    unsafe {
                        cuda::gemv_e4m3_mk(
                            stream.cu_stream() as *mut c_void,
                            pw as *const u8,
                            ps as *const f32,
                            px as *const u16,
                            py as *mut u16,
                            n as i32,
                            k as i32,
                            m as i32,
                        )
                    }
                };
                cells += 1;
                if rc_ref != 0 {
                    continue;
                }
                let rc_grp = {
                    let (pw, _a) = dw.device_ptr(&stream);
                    let (ps, _b) = ds.device_ptr(&stream);
                    let (px, _c) = dx.device_ptr(&stream);
                    let (py, _d) = y_grp.device_ptr_mut(&stream);
                    unsafe {
                        cuda::gemv_e4m3_group(
                            stream.cu_stream() as *mut c_void,
                            pw as *const u8,
                            ps as *const f32,
                            px as *const u16,
                            py as *mut u16,
                            n as i32,
                            k as i32,
                            m as i32,
                            r,
                        )
                    }
                };
                assert_eq!(rc_grp, 0, "gemv_e4m3_group rc={rc_grp} n={n} k={k} m={m} r={r}");
                stream.synchronize().unwrap();
                #[allow(deprecated)]
                let a = stream.memcpy_dtov(&y_ref).unwrap();
                #[allow(deprecated)]
                let b = stream.memcpy_dtov(&y_grp).unwrap();
                let diff = a.iter().zip(b.iter()).filter(|(p, q)| p != q).count();
                eprintln!(
                    "E4M3-GROUP-IDENTITY n={n} k={k} m={m} rows_per_warp={r} differing={diff}/{}",
                    a.len()
                );
                assert_eq!(
                    diff, 0,
                    "gemv_e4m3_group is not bit-identical to gemv_e4m3_mk at n={n} k={k} m={m} \
                     rows_per_warp={r}: {diff} of {} bf16 outputs differ, so the arm is an \
                     fp-reorder and not the same arithmetic",
                    a.len()
                );
                ran += 1;
            }
        }
    }
    eprintln!("E4M3-GROUP-IDENTITY compared={ran} of {cells} cells (the rest the basis refuses)");
    assert!(
        ran >= 100,
        "only {ran} of {cells} cells had a live basis; a certificate that compares almost nothing \
         is not a certificate"
    );
}
