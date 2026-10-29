use candle_core::Device;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

pub const NV_TRAIN_PROF_ENV_OPT_IN_1_SYNCS_THE_DEVICE_AT_EVERY_PHASE_BOUNDARY_AND_ACCUMULATES_MS_PER_PHASE_FOR_THE_STEP_PROFILE:
    &str = "NV_TRAIN_PROF";

static PHASES: OnceLock<Mutex<Vec<(&'static str, f64)>>> = OnceLock::new();
static OPTIMIZER_TIMELINE: OnceLock<Mutex<Option<(Vec<Instant>, bool)>>> = OnceLock::new();

pub fn start_optimizer_timeline() {
    assert!(
        enabled(),
        "optimizer timeline requires synchronized phase instrumentation"
    );
    let mut timeline = OPTIMIZER_TIMELINE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap();
    assert!(
        timeline.is_none(),
        "only one trainer may record a process timeline"
    );
    *timeline = Some((Vec::new(), true));
}

fn steady_intervals(stamps: &[Instant], steps: usize, warmup: usize) -> anyhow::Result<Vec<f64>> {
    anyhow::ensure!(
        stamps.len() == steps,
        "missing or excess optimizer-end stamps"
    );
    anyhow::ensure!(
        warmup > 0 && warmup < steps,
        "discard the unmeasured first step and retain steady steps"
    );
    let intervals: Vec<f64> = stamps
        .windows(2)
        .map(|w| w[1].saturating_duration_since(w[0]).as_secs_f64())
        .collect();
    anyhow::ensure!(
        intervals.iter().all(|v| *v > 0.0),
        "nonmonotonic optimizer-end stamps"
    );
    Ok(intervals.into_iter().skip(warmup - 1).collect())
}

pub fn take_optimizer_step_seconds(steps: usize, warmup: usize) -> anyhow::Result<Vec<f64>> {
    let (stamps, valid) = OPTIMIZER_TIMELINE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| anyhow::anyhow!("optimizer timeline was not started"))?;
    anyhow::ensure!(valid, "optimizer closure or device synchronization failed");
    steady_intervals(&stamps, steps, warmup)
}

pub fn enabled() -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call(
        NV_TRAIN_PROF_ENV_OPT_IN_1_SYNCS_THE_DEVICE_AT_EVERY_PHASE_BOUNDARY_AND_ACCUMULATES_MS_PER_PHASE_FOR_THE_STEP_PROFILE,
    )
}

fn sync(device: &Device) -> bool {
    device.synchronize().is_ok()
}

#[cfg(feature = "cuda")]
fn current_pool_memory() -> Result<(u64, u64), cudarc::driver::result::DriverError> {
    use cudarc::driver::sys;
    let mut device = 0;
    let mut pool = std::ptr::null_mut();
    let mut used = 0u64;
    let mut reserved = 0u64;
    unsafe {
        sys::cuCtxGetDevice(&mut device).result()?;
        sys::cuDeviceGetMemPool(&mut pool, device).result()?;
        sys::cuMemPoolGetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            (&mut used as *mut u64).cast(),
        )
        .result()?;
        sys::cuMemPoolGetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            (&mut reserved as *mut u64).cast(),
        )
        .result()?;
    }
    Ok((used, reserved))
}

fn record_device_memory(device: &Device, phase: &str, boundary: &str) {
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        match cudarc::driver::result::mem_get_info() {
            Ok((free, total)) => eprintln!(
                "TRAIN_DEVICE_MEMORY phase={phase} boundary={boundary} free_bytes={free} total_bytes={total}"
            ),
            Err(error) => eprintln!(
                "TRAIN_DEVICE_MEMORY phase={phase} boundary={boundary} unavailable={error:?}"
            ),
        }
        match current_pool_memory() {
            Ok((used, reserved)) => eprintln!(
                "TRAIN_POOL_MEMORY phase={phase} boundary={boundary} used_bytes={used} reserved_bytes={reserved}"
            ),
            Err(error) => eprintln!(
                "TRAIN_POOL_MEMORY phase={phase} boundary={boundary} unavailable={error:?}"
            ),
        }
    }
    let _ = (device, phase, boundary);
}

pub fn phase<T>(
    device: &Device,
    name: &'static str,
    f: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    if !enabled() {
        return f();
    }
    let before_ok = sync(device);
    record_device_memory(device, name, "before");
    let t0 = Instant::now();
    let out = f();
    let after_ok = sync(device);
    record_device_memory(device, name, "after");
    if name == "optimizer_step" {
        let stamp = Instant::now();
        if let Some(timeline) = OPTIMIZER_TIMELINE.get() {
            if let Some((stamps, valid)) = timeline.lock().unwrap().as_mut() {
                *valid &= before_ok && after_ok && out.is_ok();
                stamps.push(stamp);
            }
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    PHASES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((name, ms));
    out
}

pub fn take() -> Vec<(&'static str, f64)> {
    PHASES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .map(|mut v| std::mem::take(&mut *v))
        .unwrap_or_default()
}

pub fn summary_ms_per_phase(rows: &[(&'static str, f64)]) -> Vec<(&'static str, f64, usize)> {
    let mut out: Vec<(&'static str, f64, usize)> = Vec::new();
    for (n, ms) in rows {
        match out.iter_mut().find(|(k, _, _)| k == n) {
            Some(e) => {
                e.1 += ms;
                e.2 += 1;
            }
            None => out.push((n, *ms, 1)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn optimizer_end_intervals_discard_five_warmup_steps_and_reject_incomplete_or_nonmonotonic_receipts() {
        let origin = Instant::now();
        let stamps: Vec<_> = (0..30)
            .map(|i| origin + Duration::from_secs(i * i + i))
            .collect();
        let seconds = steady_intervals(&stamps, 30, 5).unwrap();
        assert_eq!(seconds.len(), 25);
        assert_eq!(seconds[0], 10.0);
        assert_eq!(seconds[24], 58.0);
        let t = Instant::now();
        assert!(steady_intervals(&[t], 30, 5).is_err());
        assert!(steady_intervals(&[t, t], 2, 1).is_err());
        assert!(steady_intervals(&[t + Duration::from_secs(1), t], 2, 1).is_err());
        assert!(steady_intervals(&[t], 1, 0).is_err());
    }

}
