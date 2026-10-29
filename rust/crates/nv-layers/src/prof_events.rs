#[cfg(feature = "cuda")]
use std::sync::{Mutex, OnceLock};

pub const NV_TRAIN_PROF_EVENTS_ENV_OPT_IN_1_RECORDS_A_CUDA_EVENT_PAIR_AROUND_EVERY_SPANNED_KERNEL_FAMILY_AND_SUMS_THEM_AT_TAKE: &str =
    "NV_TRAIN_PROF_EVENTS";

pub fn enabled() -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call(
        NV_TRAIN_PROF_EVENTS_ENV_OPT_IN_1_RECORDS_A_CUDA_EVENT_PAIR_AROUND_EVERY_SPANNED_KERNEL_FAMILY_AND_SUMS_THEM_AT_TAKE,
    )
}

#[cfg(feature = "cuda")]
struct Pair {
    name: &'static str,
    start: cudarc::driver::CudaEvent,
    end: cudarc::driver::CudaEvent,
}

#[cfg(feature = "cuda")]
static PAIRS: OnceLock<Mutex<Vec<Pair>>> = OnceLock::new();

#[cfg(feature = "cuda")]
pub struct Span {
    name: &'static str,
    start: Option<cudarc::driver::CudaEvent>,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
}

#[cfg(feature = "cuda")]
impl Drop for Span {
    fn drop(&mut self) {
        let Some(start) = self.start.take() else {
            return;
        };
        if let Ok(end) = self
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        {
            PAIRS
                .get_or_init(|| Mutex::new(Vec::new()))
                .lock()
                .unwrap()
                .push(Pair {
                    name: self.name,
                    start,
                    end,
                });
        }
    }
}

#[cfg(feature = "cuda")]
pub fn span(
    name: &'static str,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Option<Span> {
    if !enabled() {
        return None;
    }
    let start = stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .ok()?;
    Some(Span {
        name,
        start: Some(start),
        stream: stream.clone(),
    })
}

#[cfg(feature = "cuda")]
pub fn take_summing_ms_per_name() -> Vec<(&'static str, f64, usize)> {
    let pairs: Vec<Pair> =
        std::mem::take(&mut *PAIRS.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap());
    let mut out: Vec<(&'static str, f64, usize)> = Vec::new();
    for p in pairs {
        let ms = p.start.elapsed_ms(&p.end).unwrap_or(0.0) as f64;
        match out.iter_mut().find(|(n, _, _)| *n == p.name) {
            Some(e) => {
                e.1 += ms;
                e.2 += 1;
            }
            None => out.push((p.name, ms, 1)),
        }
    }
    out
}

#[cfg(not(feature = "cuda"))]
pub fn take_summing_ms_per_name() -> Vec<(&'static str, f64, usize)> {
    Vec::new()
}
