use anyhow::{anyhow, Result};
use cudarc::driver::result as cu_result;
use cudarc::driver::safe::CudaStream;
use cudarc::driver::sys::{self, CUgraphInstantiate_flags_enum, CUstreamCaptureMode_enum};
use std::collections::HashMap;
use std::sync::Arc;

pub fn capture_lock() -> &'static std::sync::Mutex<()> {
    static L: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    L.get_or_init(|| std::sync::Mutex::new(()))
}

static MEMPOOL_TRIMS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn mempool_trims() -> u64 {
    MEMPOOL_TRIMS.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn trim_device_graph_mempool_because_graph_destroy_keeps_the_reserved_pages(ordinal: usize) {
    if let Ok(devh) = cu_result::device::get(ordinal as i32) {
        let _ = unsafe { sys::cuDeviceGraphMemTrim(devh) };
    }
    MEMPOOL_TRIMS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

pub struct CudaGraphRunner {
    stream: Arc<CudaStream>,
    cached: HashMap<u64, RawGraph>,
}

struct RawGraph {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}

impl RawGraph {
    fn launch(&self) -> Result<()> {
        self.launch_on(self.stream.cu_stream())
    }

    fn launch_on(&self, cu_stream: sys::CUstream) -> Result<()> {
        self.stream
            .context()
            .bind_to_thread()
            .map_err(|e| anyhow!("bind_to_thread failed: {e:?}"))?;
        unsafe { cu_result::graph::launch(self.cu_graph_exec, cu_stream) }
            .map_err(|e| anyhow!("graph launch failed: {e:?}"))
    }
}

impl Drop for RawGraph {
    fn drop(&mut self) {
        let _ = self.stream.context().bind_to_thread();
        let exec = std::mem::replace(&mut self.cu_graph_exec, std::ptr::null_mut());
        if !exec.is_null() {
            unsafe {
                let _ = cu_result::graph::exec_destroy(exec);
            }
        }
        let graph = std::mem::replace(&mut self.cu_graph, std::ptr::null_mut());
        if !graph.is_null() {
            unsafe {
                let _ = cu_result::graph::destroy(graph);
            }
        }
    }
}

unsafe impl Send for CudaGraphRunner {}

struct EndOpenCaptureOnUnwindBecauseAPanicInTheCapturedClosureOtherwiseLeavesTheStreamInCaptureMode
{
    stream: Arc<CudaStream>,
    armed: bool,
}

impl Drop
    for EndOpenCaptureOnUnwindBecauseAPanicInTheCapturedClosureOtherwiseLeavesTheStreamInCaptureMode
{
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let _ = self.stream.context().bind_to_thread();
        if let Ok(g) = unsafe { cu_result::stream::end_capture(self.stream.cu_stream()) } {
            if !g.is_null() {
                unsafe {
                    let _ = cu_result::graph::destroy(g);
                }
            }
        }
    }
}

impl CudaGraphRunner {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self {
            stream,
            cached: HashMap::new(),
        }
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn invalidate(&mut self) {
        self.cached.clear();
    }

    pub fn invalidate_token(&mut self, shape_token: u64) {
        self.cached.remove(&shape_token);
    }

    pub fn has_cached(&self) -> bool {
        !self.cached.is_empty()
    }

    pub fn has_cached_token(&self, shape_token: u64) -> bool {
        self.cached.contains_key(&shape_token)
    }

    pub fn cached_node_count(&self) -> usize {
        self.cached
            .values()
            .map(|g| raw_graph_node_count(g.cu_graph))
            .sum()
    }

    pub fn probe_capture(&self) -> Result<()> {
        let _capture_guard = capture_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.stream
            .synchronize()
            .map_err(|e| anyhow!("probe pre-capture synchronize failed: {e:?}"))?;
        self.stream
            .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| anyhow!("begin_capture failed: {e:?}"))?;
        let _ = self.stream.context().bind_to_thread();
        let end_result = unsafe { cu_result::stream::end_capture(self.stream.cu_stream()) };
        if let Ok(g) = end_result {
            if !g.is_null() {
                unsafe {
                    let _ = cu_result::graph::destroy(g);
                }
            }
        }
        Ok(())
    }

    pub fn run<F>(&mut self, shape_token: u64, f: F) -> Result<()>
    where
        F: FnOnce(&Arc<CudaStream>) -> Result<()>,
    {
        self.run_on(shape_token, None, f)
    }

    pub fn run_on<F>(
        &mut self,
        shape_token: u64,
        launch_stream: Option<&Arc<CudaStream>>,
        f: F,
    ) -> Result<()>
    where
        F: FnOnce(&Arc<CudaStream>) -> Result<()>,
    {
        if let Some(graph) = self.cached.get(&shape_token) {
            match launch_stream {
                Some(s) => graph
                    .launch_on(s.cu_stream())
                    .map_err(|e| anyhow!("graph replay (explicit stream) failed: {e:?}"))?,
                None => graph
                    .launch()
                    .map_err(|e| anyhow!("graph replay failed: {e:?}"))?,
            }
            return Ok(());
        }

        let _capture_guard = capture_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        self.stream
            .synchronize()
            .map_err(|e| anyhow!("pre-capture synchronize failed: {e:?}"))?;

        self.stream
            .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| anyhow!("begin_capture failed: {e:?}"))?;

        let mut unwind_guard =
            EndOpenCaptureOnUnwindBecauseAPanicInTheCapturedClosureOtherwiseLeavesTheStreamInCaptureMode {
                stream: self.stream.clone(),
                armed: true,
            };
        let user_result = f(&self.stream);

        let bind_result = self.stream.context().bind_to_thread();
        let end_result = unsafe { cu_result::stream::end_capture(self.stream.cu_stream()) };
        unwind_guard.armed = false;

        let mut cu_graph: sys::CUgraph = std::ptr::null_mut();
        if let Ok(g) = &end_result {
            cu_graph = *g;
        }
        if user_result.is_err() || bind_result.is_err() || end_result.is_err() || cu_graph.is_null()
        {
            self.destroy_the_failed_graph_and_trim(cu_graph);
            user_result?;
            bind_result.map_err(|e| anyhow!("bind_to_thread failed: {e:?}"))?;
            end_result.map_err(|e| anyhow!("end_capture failed: {e:?}"))?;
            return Err(anyhow!("end_capture returned no graph"));
        }

        let n_nodes = raw_graph_node_count(cu_graph);
        eprintln!("[cuda-graph] captured shape_token={shape_token}: {n_nodes} nodes");
        if graph_mem_debug() {
            report_graph_mem(
                cu_graph,
                shape_token,
                self.stream.context().ordinal() as i32,
            );
        }

        let cu_graph_exec = match unsafe {
            cu_result::graph::instantiate(
                cu_graph,
                CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
        } {
            Ok(exec) => exec,
            Err(e) => {
                self.destroy_the_failed_graph_and_trim(cu_graph);
                return Err(anyhow!("graph instantiate failed: {e:?}"));
            }
        };
        let graph = RawGraph {
            cu_graph,
            cu_graph_exec,
            stream: self.stream.clone(),
        };
        if let Err(e) = graph.launch() {
            drop(graph);
            trim_device_graph_mempool_because_graph_destroy_keeps_the_reserved_pages(
                self.stream.context().ordinal(),
            );
            return Err(anyhow!("first graph launch failed: {e:?}"));
        }
        self.cached.insert(shape_token, graph);
        Ok(())
    }

    fn destroy_the_failed_graph_and_trim(&self, cu_graph: sys::CUgraph) {
        if !cu_graph.is_null() {
            unsafe {
                let _ = cu_result::graph::destroy(cu_graph);
            }
        }
        trim_device_graph_mempool_because_graph_destroy_keeps_the_reserved_pages(
            self.stream.context().ordinal(),
        );
    }
}

fn graph_mem_debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NV_DEBUG_GRAPH_MEM").is_some())
}

fn report_graph_mem(cu_graph: sys::CUgraph, shape_token: u64, device: i32) {
    let total = raw_graph_node_count(cu_graph);
    let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); total];
    let mut n = total;
    let rc = unsafe { sys::cuGraphGetNodes(cu_graph, nodes.as_mut_ptr(), &mut n) };
    if rc != sys::CUresult::CUDA_SUCCESS {
        eprintln!("[graph-mem] shape_token={shape_token}: cuGraphGetNodes failed {rc:?}");
        return;
    }
    let (mut allocs, mut frees, mut kernels) = (0usize, 0usize, 0usize);
    for node in nodes.iter().take(n) {
        let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        if unsafe { sys::cuGraphNodeGetType(*node, &mut ty) } != sys::CUresult::CUDA_SUCCESS {
            continue;
        }
        match ty {
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC => allocs += 1,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE => frees += 1,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => kernels += 1,
            _ => {}
        }
    }
    let attr = |a: sys::CUgraphMem_attribute| -> i64 {
        let mut v: u64 = 0;
        let rc = unsafe {
            sys::cuDeviceGetGraphMemAttribute(
                device,
                a,
                &mut v as *mut u64 as *mut std::ffi::c_void,
            )
        };
        if rc == sys::CUresult::CUDA_SUCCESS {
            v as i64
        } else {
            -1
        }
    };
    let used = attr(sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT);
    let used_hi = attr(sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_HIGH);
    let resv = attr(sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT);
    let resv_hi = attr(sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH);
    let mib = |b: i64| {
        if b < 0 {
            -1.0
        } else {
            b as f64 / (1024.0 * 1024.0)
        }
    };
    eprintln!(
        "[graph-mem] shape_token={shape_token} nodes={n} kernel={kernels} mem_alloc={allocs} \
         mem_free={frees} pool_used={:.1}MiB pool_used_high={:.1}MiB pool_reserved={:.1}MiB \
         pool_reserved_high={:.1}MiB",
        mib(used),
        mib(used_hi),
        mib(resv),
        mib(resv_hi)
    );
}

fn raw_graph_node_count(cu_graph: sys::CUgraph) -> usize {
    let mut n: usize = 0;
    let rc = unsafe { sys::cuGraphGetNodes(cu_graph, std::ptr::null_mut(), &mut n) };
    if rc != sys::CUresult::CUDA_SUCCESS {
        return 0;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_CUDA: &str = "no CUDA device 0: this test is the only gate on capture-mode recovery \
                           after a panic inside the captured closure and must not report success \
                           having executed nothing";

    #[test]
    fn a_panic_inside_the_captured_closure_leaves_the_stream_out_of_capture_mode_for_the_next_capture(
    ) {
        let Ok(ctx) = cudarc::driver::CudaContext::new(0) else {
            panic!("{NO_CUDA}");
        };
        let stream = ctx.new_stream().expect("fork a capture stream");
        let mut runner = CudaGraphRunner::new(stream.clone());
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = runner.run(0x7e57_0001, |_| panic!("mid-capture panic on purpose"));
        }));
        assert!(
            unwound.is_err(),
            "the planted panic must unwind out of run(); if it stopped panicking this test no \
             longer exercises the unwind guard"
        );
        runner.run(0x7e57_0002, |_| Ok(())).expect(
            "a capture after a mid-capture panic failed: the unwind guard did not end the open \
             capture, so the stream is still in capture mode and every later capture or eager \
             launch on it inherits the stashed CUDA_ERROR_STREAM_CAPTURE state",
        );
        assert!(runner.has_cached_token(0x7e57_0002));
        assert!(
            !runner.has_cached_token(0x7e57_0001),
            "the panicked capture must not have cached a graph"
        );
    }

    #[test]
    fn a_failed_capture_routes_through_the_mempool_trim_and_the_next_capture_still_works() {
        let Ok(ctx) = cudarc::driver::CudaContext::new(0) else {
            panic!("{NO_CUDA}");
        };
        let stream = ctx.new_stream().expect("fork a capture stream");
        let mut runner = CudaGraphRunner::new(stream.clone());
        let trims_before = mempool_trims();
        let failed = runner.run(0x7e57_0003, |_| {
            Err(anyhow!("planted capture failure on purpose"))
        });
        assert!(
            failed.is_err(),
            "the planted closure error must propagate out of run()"
        );
        assert!(
            !runner.has_cached_token(0x7e57_0003),
            "the failed capture must not have cached a graph"
        );
        assert!(
            mempool_trims() > trims_before,
            "the capture-failure arm did not reach \
             trim_device_graph_mempool_because_graph_destroy_keeps_the_reserved_pages. Destroying \
             a graph marks its alloc nodes unused but does NOT hand the reserved physical pages \
             back; every failure arm in run_on (closure error, end_capture error, instantiate \
             error, first-launch error) must funnel through the trim, or the reservation stacks \
             until engine Drop -- the qwen38_batch/graph_engine engines trim and this runner's \
             failure arms must match that discipline"
        );
        runner
            .run(0x7e57_0004, |_| Ok(()))
            .expect("a capture after a failed capture must still work");
    }

    #[test]
    fn invalidate_token_destroys_one_graph_and_leaves_the_rest_cached() {
        let Ok(ctx) = cudarc::driver::CudaContext::new(0) else {
            panic!("{NO_CUDA}");
        };
        let stream = ctx.new_stream().expect("fork a capture stream");
        let mut runner = CudaGraphRunner::new(stream.clone());
        runner.run(0x7e57_0005, |_| Ok(())).expect("capture A");
        runner.run(0x7e57_0006, |_| Ok(())).expect("capture B");
        runner.invalidate_token(0x7e57_0005);
        assert!(
            !runner.has_cached_token(0x7e57_0005),
            "invalidate_token must drop exactly the named graph -- the vision bucket eviction \
             path relies on it to destroy an evicted bucket's graph without recapturing every \
             surviving bucket"
        );
        assert!(
            runner.has_cached_token(0x7e57_0006),
            "invalidate_token must not touch other cached graphs"
        );
    }
}
