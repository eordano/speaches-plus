use std::cell::Cell;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{self, ThreadId};

pub const A_DEVICE_WIDE_SYNCHRONIZE_ON_ANY_THREAD_INVALIDATES_EVERY_CAPTURE_IN_FLIGHT_IN_THE_PROCESS:
    &str = "cuCtxSynchronize fails with CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED and turns every \
     capture in flight in the context into CUDA_ERROR_STREAM_CAPTURE_INVALIDATED -- even a \
     CU_STREAM_CAPTURE_MODE_RELAXED capture on a CU_STREAM_NON_BLOCKING stream owned by another \
     thread. cublasDestroy and cublasLtDestroy synchronize the device the same way while \
     returning success themselves, and cudarc's CudaContext::new_stream synchronizes a fresh \
     context before its first stream, which every candle Device::new_cuda_with_stream does. A \
     driver-API probe on a CUDA host reproduced each 3 out of 3 times while cublasCreate, cuMemFree, \
     cuModuleUnload, cuStreamDestroy, curand teardown and cuDeviceGraphMemTrim left the capture \
     intact. Every capture therefore holds a CaptureInFlight and every one of those calls holds a \
     NoCaptureInFlight, so the two never overlap";

struct State {
    captures: usize,
    excluder: Option<ThreadId>,
    depth: usize,
}

static STATE: Mutex<State> = Mutex::new(State {
    captures: 0,
    excluder: None,
    depth: 0,
});
static CHANGED: Condvar = Condvar::new();

thread_local! {
    static CAPTURES_ON_THIS_THREAD: Cell<usize> = const { Cell::new(0) };
}

fn state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

fn wait(s: MutexGuard<'static, State>) -> MutexGuard<'static, State> {
    CHANGED.wait(s).unwrap_or_else(PoisonError::into_inner)
}

fn another_thread_excludes_captures(s: &State, me: ThreadId) -> bool {
    s.excluder.is_some_and(|d| d != me)
}

#[must_use = "hold the CaptureInFlight from begin_capture until the capture has ended"]
pub struct CaptureInFlight(());

pub fn capture_in_flight() -> CaptureInFlight {
    let me = thread::current().id();
    let mut s = state();
    while another_thread_excludes_captures(&s, me) {
        s = wait(s);
    }
    s.captures += 1;
    CAPTURES_ON_THIS_THREAD.with(|c| c.set(c.get() + 1));
    CaptureInFlight(())
}

impl Drop for CaptureInFlight {
    fn drop(&mut self) {
        CAPTURES_ON_THIS_THREAD.with(|c| c.set(c.get() - 1));
        state().captures -= 1;
        CHANGED.notify_all();
    }
}

#[must_use = "hold the NoCaptureInFlight until the device-wide synchronizing call it covers has returned"]
pub struct NoCaptureInFlight(());

pub fn no_capture_in_flight() -> NoCaptureInFlight {
    let me = thread::current().id();
    let mine = CAPTURES_ON_THIS_THREAD.with(Cell::get);
    let mut s = state();
    while another_thread_excludes_captures(&s, me) {
        s = wait(s);
    }
    s.excluder = Some(me);
    s.depth += 1;
    while s.captures > mine {
        s = wait(s);
    }
    NoCaptureInFlight(())
}

impl Drop for NoCaptureInFlight {
    fn drop(&mut self) {
        let mut s = state();
        s.depth -= 1;
        if s.depth == 0 {
            s.excluder = None;
        }
        drop(s);
        CHANGED.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{mpsc, Arc, Mutex as StdMutex};
    use std::time::Duration;

    static SERIAL: StdMutex<()> = StdMutex::new(());

    #[test]
    fn a_synchronizing_call_waits_for_the_capture_another_thread_has_in_flight() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let capture = capture_in_flight();
        let excluded = Arc::new(AtomicBool::new(false));
        let flag = excluded.clone();
        let excluder = thread::spawn(move || {
            let _d = no_capture_in_flight();
            flag.store(true, Ordering::SeqCst);
        });
        thread::sleep(Duration::from_millis(100));
        assert!(
            !excluded.load(Ordering::SeqCst),
            "a device-wide synchronizing call must not run while another thread's capture is in flight"
        );
        drop(capture);
        excluder.join().unwrap();
        assert!(excluded.load(Ordering::SeqCst));
    }

    #[test]
    fn a_capture_waits_for_the_synchronizing_call_another_thread_is_making() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let exclusion = no_capture_in_flight();
        let (began, captured) = mpsc::channel();
        let capturer = thread::spawn(move || {
            let _c = capture_in_flight();
            began.send(()).unwrap();
        });
        assert!(
            captured.recv_timeout(Duration::from_millis(100)).is_err(),
            "a capture must not begin while another thread makes a device-wide synchronizing call"
        );
        drop(exclusion);
        captured.recv_timeout(Duration::from_secs(5)).unwrap();
        capturer.join().unwrap();
    }

    #[test]
    fn nested_exclusion_and_a_capture_on_the_excluding_thread_do_not_deadlock() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let (done, finished) = mpsc::channel();
        let worker = thread::spawn(move || {
            let _outer = no_capture_in_flight();
            let _inner = no_capture_in_flight();
            let _capture = capture_in_flight();
            let _innermost = no_capture_in_flight();
            done.send(()).unwrap();
        });
        finished
            .recv_timeout(Duration::from_secs(5))
            .expect("the excluding thread must not wait on itself");
        worker.join().unwrap();
        let _after = no_capture_in_flight();
        let _capture_after = capture_in_flight();
    }
}
