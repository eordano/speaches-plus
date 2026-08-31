#![cfg(feature = "cuda")]

use candle_core::Device;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;

const WHY: &str = "NV_Q38_GDN_AB_WIDE replaces the batch GDN's per-lane a/b projection loop with \
                   one gemm_bf16_mk launch per layer. That is only allowed to ship as a launch \
                   count change if row j of the wide call is the SAME BITS as call j of the loop \
                   it replaces, because the a/b outputs drive the gated delta rule's recurrent \
                   state and a one-ulp difference there compounds over every future step. This \
                   suite is the certificate; the arm's constant cites it by name.";

const THE_GDN_AB_SHAPE_IS_TWICE_THE_VALUE_HEAD_COUNT_BY_HIDDEN: (usize, usize) = (96, 5120);

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 33) & 0x7f_ffff) as f32 / 8388608.0) * 2.0 - 1.0
    }
    fn bits(&mut self, n: usize, gain: f32) -> Vec<u16> {
        (0..n)
            .map(|_| bf16::from_f32(self.next_f32() * gain).to_bits())
            .collect()
    }
}

#[test]
#[ignore = "needs CUDA device 0; set NV_GDN_AB_WIDE_TEST=1 -- proves gemm_bf16_mk row j is \
            bit-identical to gemv_bf16 on row j at the qwen3.8 GDN a/b projection shape for \
            every lane count the batch route can form"]
fn gemm_bf16_mk_rows_are_bit_identical_to_the_looped_gemv_bf16_at_the_gdn_ab_shape() {
    if std::env::var("NV_GDN_AB_WIDE_TEST").as_deref() != Ok("1") {
        panic!("set NV_GDN_AB_WIDE_TEST=1 to run this GPU test (it must never silently skip)");
    }
    eprintln!("[gdn-ab-wide] {WHY}");
    let device = Device::new_cuda_with_stream(0).expect("cuda device 0");
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut rng = Lcg(0x51ed_270b_9ea1_1e3d);

    let shapes = [
        THE_GDN_AB_SHAPE_IS_TWICE_THE_VALUE_HEAD_COUNT_BY_HIDDEN,
        (96usize, 2048usize),
        (48, 4096),
        (192, 5120),
    ];
    let mut cells = 0usize;
    for (n, k) in shapes {
        let w_bits = rng.bits(n * k, 0.05);
        let x_bits = rng.bits(16 * k, 1.0);
        #[allow(deprecated)]
        let w_dev: CudaSlice<u16> = stream.clone_htod(&w_bits).unwrap();
        #[allow(deprecated)]
        let x_dev: CudaSlice<u16> = stream.clone_htod(&x_bits).unwrap();

        for m in 1..=16usize {
            let mut y_wide: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();
            let rc = {
                let (wp, _a) = w_dev.device_ptr(&stream);
                let (xp, _b) = x_dev.device_ptr(&stream);
                let (yp, _c) = y_wide.device_ptr_mut(&stream);
                unsafe {
                    nv_kernels::cuda::gemm_bf16_mk(
                        stream.cu_stream() as *mut c_void,
                        wp as *const u16,
                        xp as *const u16,
                        yp as *mut u16,
                        n as i32,
                        k as i32,
                        m as i32,
                    )
                }
            };
            assert_eq!(rc, 0, "gemm_bf16_mk rc={rc} at n={n} k={k} m={m}");

            let mut y_loop: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();
            for row in 0..m {
                let x_row = x_dev.slice(row * k..(row + 1) * k);
                let mut y_row = y_loop.slice_mut(row * n..(row + 1) * n);
                let rc = {
                    let (wp, _a) = w_dev.device_ptr(&stream);
                    let (xp, _b) = x_row.device_ptr(&stream);
                    let (yp, _c) = y_row.device_ptr_mut(&stream);
                    unsafe {
                        nv_kernels::cuda::gemv_bf16(
                            stream.cu_stream() as *mut c_void,
                            wp as *const u16,
                            xp as *const u16,
                            yp as *mut u16,
                            n as i32,
                            k as i32,
                        )
                    }
                };
                assert_eq!(rc, 0, "gemv_bf16 rc={rc} at n={n} k={k} row={row}");
            }
            stream.synchronize().unwrap();
            #[allow(deprecated)]
            let got = stream.memcpy_dtov(&y_wide).unwrap();
            #[allow(deprecated)]
            let want = stream.memcpy_dtov(&y_loop).unwrap();
            let differing = got
                .iter()
                .zip(want.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                differing, 0,
                "n={n} k={k} m={m}: {differing} of {} bf16 outputs differ between the wide \
                 gemm_bf16_mk and the looped gemv_bf16. {WHY}",
                m * n
            );
            cells += 1;
        }
        eprintln!("[gdn-ab-wide] n={n} k={k}: m=1..16 all bit-identical");
    }
    assert_eq!(
        cells,
        shapes.len() * 16,
        "the ladder must cover every lane count at every shape or it is not a certificate"
    );
    eprintln!("[gdn-ab-wide] {cells} (shape, m) cells, 0 differing bf16 outputs");
}
