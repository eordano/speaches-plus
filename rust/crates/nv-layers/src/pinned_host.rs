use anyhow::Result;
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DevicePtr, DeviceRepr, ValidAsZeroBits};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub const PINNED_HOST_ALLOCATIONS_REFUSE_RATHER_THAN_FALL_BACK_TO_PAGEABLE_MEMORY_BECAUSE_A_PAGEABLE_COPY_SERIALIZES_THE_STREAM_AND_SWAPS_UNDER_PRESSURE: &str =
    "cuMemHostAlloc failed; the handover tier refuses to fall back to pageable memory, whose \
     copies are staged synchronously through a driver bounce buffer and whose pages the kernel \
     may swap out under host-memory pressure";

static PINNED_BYTES: AtomicUsize = AtomicUsize::new(0);

pub fn pinned_host_bytes_live() -> usize {
    PINNED_BYTES.load(Ordering::Relaxed)
}

pub struct PinnedHostBuf {
    ptr: *mut c_void,
    len: usize,
}

unsafe impl Send for PinnedHostBuf {}
unsafe impl Sync for PinnedHostBuf {}

impl PinnedHostBuf {
    pub fn alloc(stream: &Arc<CudaStream>, len: usize, what: &str) -> Result<Self> {
        stream.context().bind_to_thread()?;
        let ptr = unsafe { cudarc::driver::result::malloc_host(len.max(1), 0) }.map_err(|e| {
            anyhow::anyhow!(
                "{what}: {len} bytes: {e:?}: {}",
                PINNED_HOST_ALLOCATIONS_REFUSE_RATHER_THAN_FALL_BACK_TO_PAGEABLE_MEMORY_BECAUSE_A_PAGEABLE_COPY_SERIALIZES_THE_STREAM_AND_SWAPS_UNDER_PRESSURE
            )
        })?;
        PINNED_BYTES.fetch_add(len, Ordering::Relaxed);
        Ok(Self { ptr, len })
    }

    pub fn from_bytes(stream: &Arc<CudaStream>, src: &[u8], what: &str) -> Result<Self> {
        let mut b = Self::alloc(stream, src.len(), what)?;
        b.as_mut_slice().copy_from_slice(src);
        Ok(b)
    }

    pub fn download<T: DeviceRepr>(stream: &Arc<CudaStream>, src: &CudaSlice<T>, what: &str) -> Result<Self> {
        let bytes = src.len() * std::mem::size_of::<T>();
        let mut b = Self::alloc(stream, bytes, what)?;
        let (p, _g) = src.device_ptr(stream);
        unsafe { cudarc::driver::result::memcpy_dtoh_async(b.as_mut_slice(), p, stream.cu_stream()) }?;
        stream.synchronize()?;
        Ok(b)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
    }

    pub fn upload_on<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        compute: &Arc<CudaStream>,
        copy: &Arc<CudaStream>,
    ) -> Result<(CudaSlice<T>, CudaEvent)> {
        upload_pinned_bytes_on(self.as_slice(), compute, copy)
    }

    pub fn download_into_on<T: DeviceRepr>(
        &mut self,
        src: &CudaSlice<T>,
        compute: &Arc<CudaStream>,
        copy: &Arc<CudaStream>,
    ) -> Result<CudaEvent> {
        anyhow::ensure!(src.len() * std::mem::size_of::<T>() == self.len, "pinned download: size mismatch");
        copy.wait(&compute.record_event(None)?)?;
        let (p, _g) = src.device_ptr(compute);
        unsafe { cudarc::driver::result::memcpy_dtoh_async(self.as_mut_slice(), p, copy.cu_stream()) }?;
        Ok(copy.record_event(None)?)
    }
}

impl Drop for PinnedHostBuf {
    fn drop(&mut self) {
        PINNED_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        unsafe {
            let _ = cudarc::driver::result::free_host(self.ptr);
        }
    }
}

pub struct PinnedBudget {
    cap: usize,
    used: AtomicUsize,
}

impl PinnedBudget {
    pub fn new(cap_bytes: usize) -> Self {
        Self { cap: cap_bytes, used: AtomicUsize::new(0) }
    }

    pub fn from_env_gib(key: &str, default_gib: f64) -> Self {
        let gib = std::env::var(key).ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|v| v.is_finite() && *v >= 0.0).unwrap_or(default_gib);
        Self::new((gib * (1u64 << 30) as f64) as usize)
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    pub fn try_take(&self, bytes: usize) -> bool {
        self.used
            .try_update(Ordering::AcqRel, Ordering::Acquire, |u| (u + bytes <= self.cap).then_some(u + bytes))
            .is_ok()
    }

    pub fn give_back(&self, bytes: usize) {
        self.used.update(Ordering::AcqRel, Ordering::Acquire, |u| u.saturating_sub(bytes));
    }
}

pub fn upload_pinned_bytes_on<T: DeviceRepr + ValidAsZeroBits>(
    bytes: &[u8],
    compute: &Arc<CudaStream>,
    copy: &Arc<CudaStream>,
) -> Result<(CudaSlice<T>, CudaEvent)> {
    let n = bytes.len() / std::mem::size_of::<T>();
    anyhow::ensure!(n * std::mem::size_of::<T>() == bytes.len(), "pinned upload: {} bytes is not a whole number of elements", bytes.len());
    let dst: CudaSlice<T> = unsafe { compute.alloc::<T>(n.max(1)) }?;
    copy.wait(&compute.record_event(None)?)?;
    if n > 0 {
        let (p, _g) = dst.device_ptr(compute);
        unsafe { cudarc::driver::result::memcpy_htod_async(p, bytes, copy.cu_stream()) }?;
    }
    Ok((dst, copy.record_event(None)?))
}
