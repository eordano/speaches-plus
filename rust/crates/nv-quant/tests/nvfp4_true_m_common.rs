#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use std::ffi::c_void;
use std::sync::Arc;

#[allow(dead_code)]
pub fn quantize_dev(
    stream: &Arc<cudarc::driver::CudaStream>,
    x: &CudaSlice<u16>,
    m_padded: usize,
    m_logical: usize,
    k: usize,
    stored_global: f32,
) -> (CudaSlice<u8>, CudaSlice<u8>) {
    let blocks = k / 16;
    let packed_bytes = m_padded * k / 2;
    let scales_bytes = ((m_padded + 127) / 128) * 128 * ((blocks + 3) / 4) * 4;

    let mut packed: CudaSlice<u8> = unsafe { stream.alloc::<u8>(packed_bytes).unwrap() };
    let mut scales: CudaSlice<u8> = unsafe { stream.alloc::<u8>(scales_bytes).unwrap() };
    let rc = {
        let (xp, _gx) = x.device_ptr(stream);
        let (pp, _gp) = packed.device_ptr_mut(stream);
        let (sp, _gs) = scales.device_ptr_mut(stream);
        unsafe {
            nv_kernels::cuda::quantize_nvfp4_bf16(
                stream.cu_stream() as *mut c_void,
                xp as *const u16,
                pp as *mut u8,
                sp as *mut u8,
                stored_global,
                m_padded as i32,
                m_logical as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "quantize_nvfp4_bf16 rc={rc}");
    (packed, scales)
}
