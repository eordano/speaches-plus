
use crate::gemma4_batch_graph::capture_stream::CaptureStream;
use cudarc::driver::{CudaContext, CudaStream};
use std::sync::Arc;

pub const EVENT_TRACKING_GOES_OFF_BEFORE_THE_FIRST_CAPTURE_VISIBLE_ALLOC_NOT_JUST_BEFORE_CAPTURE:
    &str =
    "with cudarc event tracking on, every alloc attaches read/write CudaEvents to the slice and \
     every device_ptr() records into them; a buffer allocated with tracking on and later touched \
     inside a capture records events on the capture stream and poisons the graph. Call \
     disable_event_tracking_before_capture before the engine forks its capture stream and before \
     it allocates any buffer the capture will touch";

pub fn disable_event_tracking_before_capture(ctx: &Arc<CudaContext>) {
    if ctx.is_event_tracking() {
        let _ = ctx.default_stream().synchronize();
        unsafe { ctx.disable_event_tracking() };
    }
}

pub const ONLY_THE_ENGINE_THAT_FORKED_A_STREAM_MAY_RELEASE_ITS_QUANT_CACHES: &str =
    "GraphTeardown synchronizes every stream it is given but releases nv_quant's per-stream \
     cublasLt and nvfp4 caches ONLY for streams this engine forked, plus the legacy stream that \
     no engine forks and every context shares. A stream handed over by CaptureStream::for_device \
     without a fork is borrowed, not owned, and is released by nobody. \
     RELEASING_A_BORROWED_DEVICE_STREAM_FREES_A_LIVE_ENGINES_WORKSPACE says why. Build teardown \
     with GraphTeardown::for_capture and the distinction is made for you; GraphTeardown::new \
     asserts by its argument that the engine forked the stream itself";

pub const A_CTOR_THAT_ERRORS_AFTER_FORKING_STRANDS_WHAT_ITS_WARMUP_INSTALLED: &str =
    "graph-engine ctors fork their capture/aux streams first and can fail afterwards (alloc, \
     workspace install, capture probe). Any gemm run on a forked stream before the failure \
     lazily installs up to ~96 MiB of nv_quant per-stream statics -- the 64 MiB nvfp4 workspace, \
     the 32 MiB det-gemm workspace, a cublasLt handle -- keyed on the raw CUstream pointer, and \
     the teardown that would release them lives in the Drop of the engine that never got built. \
     Retried ctors (laguna_serve retries proposer/verify-graph init) turn that into an unbounded \
     per-round leak. CtorForkGuard::fork is therefore the only way an engine ctor may fork a \
     stream: the guard releases every forked stream's per-stream caches on error-path drop, and \
     the_built_engine_owns_teardown_now() disarms it once the engine's own GraphTeardown-bearing \
     Drop exists to take over";

#[must_use = "bind the CtorForkGuard for the whole ctor body; dropping it immediately releases \
              the streams it forked. Call the_built_engine_owns_teardown_now() just before \
              returning Ok(engine)"]
pub struct CtorForkGuard {
    forked: Vec<Arc<CudaStream>>,
    engine_built: bool,
}

impl CtorForkGuard {
    pub fn new() -> Self {
        Self {
            forked: Vec::new(),
            engine_built: false,
        }
    }

    pub fn fork(
        &mut self,
        ctx: &Arc<CudaContext>,
    ) -> Result<Arc<CudaStream>, cudarc::driver::DriverError> {
        let s = ctx.new_stream()?;
        self.forked.push(s.clone());
        Ok(s)
    }

    pub fn the_built_engine_owns_teardown_now(mut self) {
        self.engine_built = true;
    }
}

impl Default for CtorForkGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CtorForkGuard {
    fn drop(&mut self) {
        if self.engine_built {
            return;
        }
        let _ = A_CTOR_THAT_ERRORS_AFTER_FORKING_STRANDS_WHAT_ITS_WARMUP_INSTALLED;
        for s in &self.forked {
            let _ = s.synchronize();
            nv_quant::release_stream_resources(s.cu_stream() as usize);
        }
    }
}

#[must_use = "a GraphTeardown that is never .run() leaves the graph mempool untrimmed and \
              nv_quant's per-stream caches stale, which poisons the next component in this process"]
pub struct GraphTeardown {

    owned: Vec<Arc<CudaStream>>,
    borrowed_and_shared_with_every_other_engine: Vec<Arc<CudaStream>>,

    legacy: Arc<CudaStream>,
    ordinal: usize,
}

impl GraphTeardown {
    fn on_context(ctx: &Arc<CudaContext>) -> Self {
        Self {
            owned: Vec::new(),
            borrowed_and_shared_with_every_other_engine: Vec::new(),
            legacy: ctx.default_stream(),
            ordinal: ctx.ordinal(),
        }
    }

    pub fn new(primary: &Arc<CudaStream>) -> Self {
        let mut td = Self::on_context(primary.context());
        td.owned.push(primary.clone());
        td
    }

    pub fn for_a_stream_this_engine_did_not_fork(primary: &Arc<CudaStream>) -> Self {
        let mut td = Self::on_context(primary.context());
        td.borrowed_and_shared_with_every_other_engine
            .push(primary.clone());
        td
    }

    pub fn for_capture(capture: &CaptureStream) -> Self {
        if capture.owns_stream() {
            Self::new(capture.stream())
        } else {
            Self::for_a_stream_this_engine_did_not_fork(capture.stream())
        }
    }

    pub fn with_stream(mut self, s: &Arc<CudaStream>) -> Self {
        self.owned.push(s.clone());
        self
    }

    fn quiesce(&self) {
        let _ = self.legacy.synchronize();
        for s in self
            .owned
            .iter()
            .chain(self.borrowed_and_shared_with_every_other_engine.iter())
        {
            let _ = s.synchronize();
        }
    }

    pub fn run(self, invalidate: impl FnOnce()) {
        self.probe("entry");
        self.quiesce();
        invalidate();
        self.quiesce();
        self.probe("graphs-destroyed");

        for s in self.owned.iter().chain(std::iter::once(&self.legacy)) {
            nv_quant::release_stream_resources(s.cu_stream() as usize);
        }
        self.probe("caches-released");

        self.trim();
        self.probe("trimmed");
    }

    fn trim(&self) {
        if let Ok(devh) = cudarc::driver::result::device::get(self.ordinal as i32) {
            let _ = unsafe { cudarc::driver::sys::cuDeviceGraphMemTrim(devh) };
        }
    }

    fn probe(&self, tag: &str) {
        if std::env::var("NV_GRAPH_TEARDOWN_DEBUG").as_deref() != Ok("1") {
            return;
        }
        let deferred = match self.legacy.context().check_err() {
            Ok(()) => "clean".to_string(),
            Err(e) => format!("DEFERRED {e:?}"),
        };
        let (reserved, used) = graph_mem(self.ordinal);
        eprintln!("[graph-teardown] {tag}: {deferred} graphmem reserved={reserved} used={used}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemma4_batch_graph::capture_stream::RELEASING_A_BORROWED_DEVICE_STREAM_FREES_A_LIVE_ENGINES_WORKSPACE;
    use candle_core::Device;

    const NO_CUDA: &str =
        "no CUDA device 0: this suite is the only gate on which streams teardown \
                           may release and must not report success having executed nothing";

    #[test]
    fn teardown_releases_the_legacy_stream_no_engine_forks_and_every_context_shares() {
        let Ok(device) = Device::new_cuda(0) else {
            panic!("{NO_CUDA}");
        };
        let Device::Cuda(d) = &device else {
            unreachable!()
        };
        let ctx = d.cuda_stream().context().clone();
        let legacy_key = ctx.default_stream().cu_stream() as usize;
        assert_eq!(
            legacy_key,
            d.cuda_stream().cu_stream() as usize,
            "GraphTeardown calls the stream it releases `legacy` because it is the one candle runs \
             every eager op on, and nv_quant keys its cublasLt and nvfp4 caches on that raw \
             CUstream. If CudaContext::default_stream() is not the stream a Device::new_cuda \
             CudaDevice launches on, then teardown releases a key nothing installed and #59/#69 is \
             untouched no matter how green the source walk in \
             tests/graph_teardown_is_universal.rs reads"
        );
        let forked = ctx.new_stream().expect("fork a stream off this context");
        let installed = nv_quant::stream_epoch(legacy_key);

        GraphTeardown::new(&forked).run(|| {});
        assert!(
            nv_quant::stream_epoch(legacy_key) > installed,
            "GraphTeardown::run released the streams it was handed and left the LEGACY stream's \
             nv_quant caches installed. That is the whole of #59/#69: no engine forks the legacy \
             stream and every CudaContext on the device shares it, so its cublasLt handle and 64 \
             MiB nvfp4 workspace -- holding CudaEvents from THIS context -- are what the next \
             component finds under the same raw CUstream key and records onto its own stream, \
             surfacing as CUDA_ERROR_INVALID_VALUE from an unrelated eager allocation. The source \
             walk in tests/graph_teardown_is_universal.rs can only prove every engine CALLS \
             GraphTeardown; this is the only test that proves calling it releases the legacy \
             stream, so with it gone the four engines routed here in #69 would be compliant and \
             still broken"
        );
    }

    #[test]
    fn teardown_of_a_borrowed_device_stream_leaves_the_shared_quant_caches_installed() {
        let Ok(device) = Device::new_cuda_with_stream(0) else {
            panic!("{NO_CUDA}");
        };
        let Device::Cuda(d) = &device else {
            unreachable!()
        };
        let shared_key = d.cuda_stream().cu_stream() as usize;
        let cs = CaptureStream::for_device(&device).expect("CaptureStream on the stream device");
        assert!(
            !cs.owns_stream(),
            "Device::new_cuda_with_stream hands candle a real stream, so for_device must borrow it \
             rather than fork; without that this test proves nothing"
        );
        let installed = nv_quant::stream_epoch(shared_key);

        GraphTeardown::for_capture(&cs).run(|| {});
        assert_eq!(
            nv_quant::stream_epoch(shared_key),
            installed,
            "GraphTeardown::for_capture released the nv_quant caches of a stream this engine only \
             borrows. {RELEASING_A_BORROWED_DEVICE_STREAM_FREES_A_LIVE_ENGINES_WORKSPACE}. \
             {ONLY_THE_ENGINE_THAT_FORKED_A_STREAM_MAY_RELEASE_ITS_QUANT_CACHES}"
        );

        drop(cs);
        assert_eq!(
            nv_quant::stream_epoch(shared_key),
            installed,
            "dropping a CaptureStream that borrows candle's device stream released that stream's \
             nv_quant caches. {RELEASING_A_BORROWED_DEVICE_STREAM_FREES_A_LIVE_ENGINES_WORKSPACE}"
        );
    }

    #[test]
    fn a_ctor_guard_dropped_on_the_error_path_releases_its_forked_streams_quant_caches() {
        let Ok(device) = Device::new_cuda(0) else {
            panic!("{NO_CUDA}");
        };
        let Device::Cuda(d) = &device else {
            unreachable!()
        };
        let ctx = d.cuda_stream().context().clone();
        let mut key = 0usize;
        let mut epoch_at_install = 0u64;
        let failed_ctor = (|| -> anyhow::Result<()> {
            let mut guard = CtorForkGuard::new();
            let forked = guard.fork(&ctx)?;
            key = forked.cu_stream() as usize;
            epoch_at_install = nv_quant::stream_epoch(key);
            nv_quant::nvfp4::ensure_workspace_for_stream(&forked)?;
            anyhow::bail!("planted ctor failure after the warmup installed the 64 MiB workspace")
        })();
        assert!(
            failed_ctor.is_err(),
            "the planted ctor failure must propagate or this test exercises nothing"
        );
        assert!(
            nv_quant::stream_epoch(key) > epoch_at_install,
            "a CtorForkGuard dropped by an erroring ctor left the forked stream's nv_quant \
             caches installed. {A_CTOR_THAT_ERRORS_AFTER_FORKING_STRANDS_WHAT_ITS_WARMUP_INSTALLED}"
        );
    }

    #[test]
    fn a_disarmed_ctor_guard_leaves_the_built_engines_quant_caches_installed() {
        let Ok(device) = Device::new_cuda(0) else {
            panic!("{NO_CUDA}");
        };
        let Device::Cuda(d) = &device else {
            unreachable!()
        };
        let ctx = d.cuda_stream().context().clone();
        let mut guard = CtorForkGuard::new();
        let forked = guard.fork(&ctx).expect("fork through the guard");
        let key = forked.cu_stream() as usize;
        nv_quant::nvfp4::ensure_workspace_for_stream(&forked).expect("install the workspace");
        let epoch_at_install = nv_quant::stream_epoch(key);
        guard.the_built_engine_owns_teardown_now();
        assert_eq!(
            nv_quant::stream_epoch(key),
            epoch_at_install,
            "a DISARMED CtorForkGuard released the caches of a successfully built engine; the \
             engine's own captured graphs bake those workspace addresses, so this free is a \
             use-after-free on the next replay. Disarming must hand teardown to the engine's Drop \
             untouched"
        );
        GraphTeardown::new(&forked).run(|| {});
    }

    #[test]
    fn teardown_of_a_forked_stream_releases_the_quant_caches_that_engine_installed() {
        let Ok(device) = Device::new_cuda(0) else {
            panic!("{NO_CUDA}");
        };
        let cs = CaptureStream::for_device(&device).expect("CaptureStream on the legacy device");
        assert!(
            cs.owns_stream(),
            "Device::new_cuda puts candle on the legacy NULL stream, so for_device must fork and \
             own the capture stream; without that this test proves nothing"
        );
        let owned_key = cs.stream().cu_stream() as usize;
        let installed = nv_quant::stream_epoch(owned_key);

        GraphTeardown::for_capture(&cs).run(|| {});
        assert!(
            nv_quant::stream_epoch(owned_key) > installed,
            "GraphTeardown::for_capture left the nv_quant caches of a stream this engine forked \
             behind. Those entries hold CudaEvents belonging to this CudaContext; the next \
             component looks them up under the same raw CUstream key from a fresh context and the \
             driver refuses the cross-context record, surfacing as CUDA_ERROR_INVALID_VALUE in an \
             unrelated eager forward. \
             {ONLY_THE_ENGINE_THAT_FORKED_A_STREAM_MAY_RELEASE_ITS_QUANT_CACHES}"
        );
    }
}

fn graph_mem(ordinal: usize) -> (u64, u64) {
    use cudarc::driver::sys;
    let Ok(devh) = cudarc::driver::result::device::get(ordinal as i32) else {
        return (0, 0);
    };
    let mut reserved: u64 = 0;
    let mut used: u64 = 0;
    unsafe {
        let _ = sys::cuDeviceGetGraphMemAttribute(
            devh,
            sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT,
            &mut reserved as *mut u64 as *mut std::ffi::c_void,
        );
        let _ = sys::cuDeviceGetGraphMemAttribute(
            devh,
            sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT,
            &mut used as *mut u64 as *mut std::ffi::c_void,
        );
    }
    (reserved, used)
}
