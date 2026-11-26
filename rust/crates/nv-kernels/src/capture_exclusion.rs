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

pub const ONE_THREAD_AT_A_TIME_OWNS_CAPTURES_AND_EXCLUSIONS_SO_A_WAITER_ONLY_WAITS_ON_THE_OWNER:
    &str = "capture_in_flight and no_capture_in_flight wait for the same condition, that no other \
     thread holds a capture or an exclusion, and then take theirs; one thread may nest both. One \
     thread at a time therefore owns the capture machinery, every waiting thread waits on that \
     owner, and these guards never make the owner wait, so they cannot deadlock. Nothing is \
     counted or claimed while waiting: a capture counted before it queues on a separate lock, or an \
     exclusion claimed before the captures drain, lets an exclusion taken inside a capture hang on \
     a capture queued behind it or on an excluder waiting for that very capture \
     (projects/speaches-plus/proof, finding F01)";

struct State {
    owner: Option<ThreadId>,
    captures: usize,
    exclusions: usize,
}

static STATE: Mutex<State> = Mutex::new(State {
    owner: None,
    captures: 0,
    exclusions: 0,
});
static CHANGED: Condvar = Condvar::new();

fn state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

fn wait(s: MutexGuard<'static, State>) -> MutexGuard<'static, State> {
    CHANGED.wait(s).unwrap_or_else(PoisonError::into_inner)
}

fn another_thread_owns(s: &State, me: ThreadId) -> bool {
    s.owner.is_some_and(|o| o != me)
}

fn own() -> MutexGuard<'static, State> {
    let me = thread::current().id();
    let mut s = state();
    while another_thread_owns(&s, me) {
        s = wait(s);
    }
    s.owner = Some(me);
    s
}

fn disown(release: impl FnOnce(&mut State)) {
    let mut s = state();
    release(&mut s);
    if s.captures == 0 && s.exclusions == 0 {
        s.owner = None;
    }
    drop(s);
    CHANGED.notify_all();
}

#[must_use = "hold the CaptureInFlight from begin_capture until the capture has ended"]
pub struct CaptureInFlight(());

pub fn capture_in_flight() -> CaptureInFlight {
    own().captures += 1;
    CaptureInFlight(())
}

impl Drop for CaptureInFlight {
    fn drop(&mut self) {
        disown(|s| s.captures -= 1);
    }
}

#[must_use = "hold the NoCaptureInFlight until the device-wide synchronizing call it covers has returned"]
pub struct NoCaptureInFlight(());

pub fn no_capture_in_flight() -> NoCaptureInFlight {
    own().exclusions += 1;
    NoCaptureInFlight(())
}

impl Drop for NoCaptureInFlight {
    fn drop(&mut self) {
        disown(|s| s.exclusions -= 1);
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

    #[test]
    fn two_threads_never_hold_captures_at_once() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let capture = capture_in_flight();
        let (began, captured) = mpsc::channel();
        let second = thread::spawn(move || {
            let _c = capture_in_flight();
            began.send(()).unwrap();
        });
        assert!(
            captured.recv_timeout(Duration::from_millis(100)).is_err(),
            "a second thread's capture must wait for the first to end: captures are serialized here, not by a separate lock"
        );
        drop(capture);
        captured.recv_timeout(Duration::from_secs(5)).unwrap();
        second.join().unwrap();
    }

    #[test]
    fn an_exclusion_inside_a_capture_does_not_wait_on_a_capture_queued_behind_it() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let (capturing, is_capturing) = mpsc::channel();
        let (go, wait_go) = mpsc::channel::<()>();
        let (done, is_done) = mpsc::channel();
        let (ended, has_ended) = mpsc::channel::<()>();
        let first = thread::spawn(move || {
            {
                let _capture = capture_in_flight();
                capturing.send(()).unwrap();
                wait_go.recv().unwrap();
                let _nested = no_capture_in_flight();
                done.send(()).unwrap();
            }
            let _ = ended.send(());
        });
        is_capturing.recv_timeout(Duration::from_secs(5)).unwrap();
        let queued = thread::spawn(move || {
            let _capture = capture_in_flight();
            let _ = has_ended.recv_timeout(Duration::from_secs(10));
        });
        thread::sleep(Duration::from_millis(100));
        go.send(()).unwrap();
        is_done.recv_timeout(Duration::from_secs(5)).expect(
            "an exclusion taken inside a capture must not wait for a capture queued behind it: that capture waits for the first to end",
        );
        first.join().unwrap();
        queued.join().unwrap();
    }

    #[test]
    fn an_exclusion_inside_a_capture_does_not_wait_on_an_excluder_waiting_for_that_capture() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let (capturing, is_capturing) = mpsc::channel();
        let (go, wait_go) = mpsc::channel::<()>();
        let (done, is_done) = mpsc::channel();
        let first = thread::spawn(move || {
            let _capture = capture_in_flight();
            capturing.send(()).unwrap();
            wait_go.recv().unwrap();
            let _nested = no_capture_in_flight();
            done.send(()).unwrap();
        });
        is_capturing.recv_timeout(Duration::from_secs(5)).unwrap();
        let excluder = thread::spawn(|| {
            let _d = no_capture_in_flight();
        });
        thread::sleep(Duration::from_millis(100));
        go.send(()).unwrap();
        is_done.recv_timeout(Duration::from_secs(5)).expect(
            "an exclusion taken inside a capture must not wait on another thread's exclusion that is itself waiting for this capture",
        );
        first.join().unwrap();
        excluder.join().unwrap();
    }
}
