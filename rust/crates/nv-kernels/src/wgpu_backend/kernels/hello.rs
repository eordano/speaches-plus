#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result};

pub const WGSL: &str = include_str!("../../../wgsl/hello.wgsl");

pub const ENTRY: &str = "hello_fill";
pub const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct HelloParams {
    n: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup(ctx, "hello", WORKGROUP_SIZE)
}

pub fn hello_launch(ctx: &WgpuContext, out: &mut [f32]) -> Result<()> {
    if out.is_empty() {
        return Ok(());
    }
    check_device(ctx)?;
    let params = HelloParams {
        n: out.len() as u32,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    let ob = dispatch::storage_zeroed(ctx, "hello-out", (out.len() * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "hello-params", &params);
    let groups = dispatch::workgroup_count_1d(ctx, out.len() as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        "nv_kernels_hello_launch",
        &compose(WGSL),
        ENTRY,
        &[(0, &ob), (1, &pb)],
        groups,
    )?;
    let got: Vec<f32> = dispatch::read_back(ctx, &ob, out.len())?;
    out.copy_from_slice(&got);
    Ok(())
}

pub fn capability(ctx: &WgpuContext) -> Result<crate::wgpu_backend::qualify::Capabilities> {
    Ok(ctx.caps.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_declares_the_entry_point() {
        assert!(WGSL.contains(ENTRY));
    }

    #[test]
    fn params_are_uniform_buffer_sized() {
        assert_eq!(std::mem::size_of::<HelloParams>() % 16, 0);
    }

    fn require() -> bool {
        std::env::var("NV_KERNELS_PARITY_REQUIRE").as_deref() == Ok("1")
    }

    fn wgpu_ctx(test: &str) -> Option<&'static WgpuContext> {
        match WgpuContext::shared() {
            Ok(c) if c.qualify().qualified => Some(c),
            Ok(c) => {
                if require() {
                    panic!("{test}: adapter not qualified: {:?}", c.qualify().reason);
                }
                eprintln!("{test}: SKIP adapter not qualified");
                None
            }
            Err(e) => {
                if require() {
                    panic!("{test}: no wgpu adapter: {e}");
                }
                eprintln!("{test}: SKIP no wgpu adapter: {e}");
                None
            }
        }
    }

    #[test]
    fn hello_fills_the_index_ramp() {
        let Some(ctx) = wgpu_ctx("hello") else {
            return;
        };
        for n in [1usize, 255, 256, 257, 1000] {
            let mut out = vec![-1.0f32; n];
            hello_launch(ctx, &mut out).unwrap();
            for (i, v) in out.iter().enumerate() {
                assert_eq!(v.to_bits(), (i as f32).to_bits(), "n={n} idx={i}");
            }
        }
        eprintln!("hello: index ramp exact for n in {{1,255,256,257,1000}}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn hello_matches_cuda_bit_for_bit() {
        use cudarc::driver::{CudaContext, CudaSlice, DevicePtrMut};

        let Some(ctx) = wgpu_ctx("hello_cuda") else {
            return;
        };
        let stream = match CudaContext::new(0) {
            Ok(c) => c.default_stream(),
            Err(e) => {
                if require() {
                    panic!("hello_cuda: no CUDA device 0: {e}");
                }
                eprintln!("hello_cuda: SKIP no CUDA device 0: {e}");
                return;
            }
        };
        let mut checked = 0usize;
        for n in [1usize, 255, 256, 257, 1000, 4097] {
            let mut dout: CudaSlice<f32> = stream.alloc_zeros::<f32>(n).unwrap();
            let rc = {
                let (op, _g) = dout.device_ptr_mut(&stream);
                unsafe {
                    crate::cuda::hello_launch(stream.cu_stream() as *mut _, op as *mut f32, n)
                }
            };
            assert_eq!(rc, 0, "cuda hello_launch rc={rc} n={n}");
            stream.synchronize().unwrap();
            #[allow(deprecated)]
            let golden = stream.memcpy_dtov(&dout).unwrap();
            let mut got = vec![f32::NAN; n];
            hello_launch(ctx, &mut got).unwrap();
            for (i, (a, b)) in golden.iter().zip(got.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "n={n} idx={i} cuda={a} wgpu={b}");
            }
            checked += n;
        }
        eprintln!("hello: {checked} f32 lanes bit-identical to CUDA across 6 sizes");
    }
}
