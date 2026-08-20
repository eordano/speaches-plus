use serde::{Deserialize, Serialize};

pub mod algo_pin;
pub mod fp8;
pub mod matmul;
pub mod mxfp4;
pub mod nvfp4;

#[cfg(feature = "cuda")]
pub fn release_stream_resources(cu_stream_key: usize) {
    matmul::release_stream_state(cu_stream_key);
    nvfp4::release_stream_workspace(cu_stream_key);
    bump_stream_epoch(cu_stream_key);
}

#[cfg(feature = "cuda")]
pub fn stream_cache_key(stream: &cudarc::driver::CudaStream) -> usize {
    const CU_STREAM_PER_THREAD: usize = 0x2;
    let raw = stream.cu_stream() as usize;
    if raw != CU_STREAM_PER_THREAD {
        return raw;
    }
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(1);
    thread_local! {
        static TID: usize = NEXT.fetch_add(1, Ordering::Relaxed);
    }
    TID.with(|t| CU_STREAM_PER_THREAD | (t << 8))
}

static STREAM_EPOCHS: std::sync::Mutex<Option<std::collections::HashMap<usize, u64>>> =
    std::sync::Mutex::new(None);

pub fn stream_epoch(cu_stream_key: usize) -> u64 {
    STREAM_EPOCHS
        .lock()
        .map(|g| {
            g.as_ref()
                .and_then(|m| m.get(&cu_stream_key).copied())
                .unwrap_or(0)
        })
        .unwrap_or(0)
}

pub fn bump_stream_epoch(cu_stream_key: usize) {
    if let Ok(mut g) = STREAM_EPOCHS.lock() {
        *g.get_or_insert_with(Default::default)
            .entry(cu_stream_key)
            .or_insert(0) += 1;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum QuantScheme {
    None,
    Fp8E4m3,
    Fp8E5m2,
    Nvfp4,
    Mxfp4,
    AwqInt4,
    GptqInt4,
    Marlin,
}

#[derive(Clone, Copy, Debug)]
pub enum LinearKind {
    Bf16,
    Fp8E4m3 { a_scale: f32, b_scale: f32 },
    Nvfp4,
    Mxfp4,
}

impl LinearKind {
    pub fn scheme(&self) -> QuantScheme {
        match self {
            LinearKind::Bf16 => QuantScheme::None,
            LinearKind::Fp8E4m3 { .. } => QuantScheme::Fp8E4m3,
            LinearKind::Nvfp4 => QuantScheme::Nvfp4,
            LinearKind::Mxfp4 => QuantScheme::Mxfp4,
        }
    }
}
