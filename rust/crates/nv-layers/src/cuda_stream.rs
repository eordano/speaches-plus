#[cfg(feature = "cuda")]
use std::cell::RefCell;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use candle_core::CudaDevice;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaStream;

#[cfg(feature = "cuda")]
thread_local! {
    static EXPLICIT_STREAM: RefCell<Option<Arc<CudaStream>>> = const { RefCell::new(None) };
}

#[cfg(feature = "cuda")]
struct Restore(Option<Arc<CudaStream>>);

#[cfg(feature = "cuda")]
impl Drop for Restore {
    fn drop(&mut self) {
        let _ = EXPLICIT_STREAM.try_with(|s| match self.0.take() {
            Some(p) => {
                let _ = s.borrow_mut().replace(p);
            }
            None => {
                let _ = s.borrow_mut().take();
            }
        });
    }
}

#[cfg(feature = "cuda")]
pub fn with_stream<R, F: FnOnce() -> R>(stream: Arc<CudaStream>, f: F) -> R {
    let _restore = Restore(EXPLICIT_STREAM.with(|s| s.borrow_mut().replace(stream)));
    f()
}

#[cfg(feature = "cuda")]
pub fn current_stream(dev: &CudaDevice) -> Arc<CudaStream> {
    EXPLICIT_STREAM.with(|s| s.borrow().clone().unwrap_or_else(|| dev.cuda_stream()))
}

#[cfg(feature = "cuda")]
pub fn sync_legacy_then_forked(dev: &CudaDevice, forked: &Arc<CudaStream>) -> anyhow::Result<()> {
    dev.cuda_stream()
        .synchronize()
        .map_err(|e| anyhow::anyhow!("legacy sync: {e:?}"))?;
    forked
        .synchronize()
        .map_err(|e| anyhow::anyhow!("forked sync: {e:?}"))
}

#[cfg(feature = "cuda")]
pub fn is_overridden() -> bool {
    EXPLICIT_STREAM.with(|s| s.borrow().is_some())
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn panic_inside_the_scope_does_not_leak_the_override_to_the_next_caller() {
        let Ok(dev) = candle_core::Device::new_cuda(0) else {
            eprintln!("SKIP: no cuda device");
            return;
        };
        let candle_core::Device::Cuda(cd) = &dev else {
            eprintln!("SKIP: device is not cuda");
            return;
        };
        let stream = cd.cuda_stream();

        assert!(
            !is_overridden(),
            "a fresh thread must start with no override"
        );

        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_stream(stream.clone(), || panic!("kernel blew up mid-scope"));
        }));
        assert!(
            caught.is_err(),
            "the panic must propagate, not be swallowed"
        );

        assert!(
            !is_overridden(),
            "with_stream restored the thread-local only on the normal-return path, so an \
             unwind left this thread pinned to a stream owned by the failed request. On the \
             spawn_blocking pools that serve /v1/ocr that is cross-request contamination: the \
             next request scheduled onto this thread would run its kernels on a stream \
             belonging to a dropped graph."
        );
    }

    #[test]
    fn nested_scopes_restore_the_outer_stream_even_when_the_inner_one_panics() {
        let Ok(dev) = candle_core::Device::new_cuda(0) else {
            eprintln!("SKIP: no cuda device");
            return;
        };
        let candle_core::Device::Cuda(cd) = &dev else {
            eprintln!("SKIP: device is not cuda");
            return;
        };
        let ctx = cd.cuda_stream().context().clone();
        let outer = ctx.new_stream().expect("fork outer stream");
        let inner = ctx.new_stream().expect("fork inner stream");
        assert_ne!(
            outer.cu_stream(),
            inner.cu_stream(),
            "the two scopes must use genuinely different streams or this test proves nothing \
             -- cd.cuda_stream() hands back the same default stream every time, which made an \
             earlier version of this test pass against the very bug it was written to catch"
        );

        with_stream(outer.clone(), || {
            assert!(is_overridden());
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                with_stream(inner, || panic!("inner scope fails"));
            }));
            assert!(caught.is_err());
            assert!(
                is_overridden(),
                "the outer scope is still live and must still be overridden"
            );
            let cur = current_stream(cd);
            assert_eq!(
                cur.cu_stream(),
                outer.cu_stream(),
                "the inner unwind must restore the OUTER stream, not clear the override"
            );
        });

        assert!(!is_overridden(), "leaving the outer scope clears it");
    }
}
