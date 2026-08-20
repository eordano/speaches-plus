use anyhow::Result;
use candle_core::Device;
use cudarc::driver::CudaStream;
use std::sync::Arc;

pub const A_CANDLE_BODY_ESCAPES_A_FORKED_CAPTURE: &str = "a CUDA graph whose captured body calls \
     candle-core -- any Tensor op: matmul, contiguous, narrow, reshape, embedding lookup, \
     candle_flash_attn -- must be captured ON THE DEVICE STREAM. candle launches only on \
     candle_core::CudaDevice::cuda_stream(); nv_layers::cuda_stream::with_stream cannot redirect \
     it, because only nv-* kernels consult that thread-local. Under \
     CU_STREAM_CAPTURE_MODE_RELAXED (nv-kernels/src/graph.rs) work submitted to any OTHER stream \
     is not captured: it runs eagerly while the capture is open, its outputs are baked into the \
     captured nv-* kernels by address, and those buffers are freed with the closure's temporaries. \
     Replay then reads freed memory -- measured on GraphedQwen3Moe as 3187 nodes captured instead \
     of 4270, then CUDA_ERROR_ILLEGAL_ADDRESS. Build the device with \
     candle_core::Device::new_cuda_with_stream so cuda_stream() is a real non-NULL stream, which \
     is what CaptureStream::for_device then captures on";

pub const A_FORKED_CAPTURE_IS_CORRECT_ONLY_FOR_A_RAW_BODY: &str = "CaptureStream::for_device forks \
     when the device is on the legacy NULL stream, because cuStreamBeginCapture refuses the NULL \
     stream outright. A forked capture stream is unconditionally correct only for a body whose \
     every launch is a raw stream-parameterized nv-* kernel, with the candle handoff done as a \
     memcpy_dtod outside the capture (laguna_graph.rs is the worked example). An engine whose body \
     IS a candle forward has exactly two honest responses, and silence is not one of them: \
     require_capture_of_a_candle_body, which refuses outright; or \
     forked_candle_capture_is_an_asserted_coincidence, which says so out loud and names the suite \
     that decodes through the graph and through the eager path and requires them to agree";

pub const RELEASING_A_BORROWED_DEVICE_STREAM_FREES_A_LIVE_ENGINES_WORKSPACE: &str =
    "nv_quant::release_stream_resources destroys the raw cublasLt handle cached for a stream and \
     frees that stream's 64 MiB nvfp4 workspace. Both live in process-wide statics keyed on the \
     raw CUstream pointer, installed once by whichever component asked first and never \
     refcounted. When CaptureStream::for_device does NOT fork -- a device built with \
     candle_core::Device::new_cuda_with_stream -- stream() IS candle's device stream, shared by \
     the eager path and by every other engine on that device, so releasing it when one engine \
     drops frees NVFP4 staging buffers whose addresses are already baked into another live \
     engine's captured graph, and the next replay reads freed memory. Only the engine that forked \
     its own stream may release it: ask CaptureStream::owns_stream, or hand the CaptureStream to \
     GraphTeardown::for_capture and let it ask for you";

pub struct CaptureStream {
    stream: Arc<CudaStream>,
    owns_stream: bool,
}

impl Drop for CaptureStream {
    fn drop(&mut self) {
        if !self.owns_stream {
            return;
        }
        let _ = self.stream.synchronize();
        nv_quant::release_stream_resources(self.stream.cu_stream() as usize);
    }
}

impl CaptureStream {
    pub fn for_device(device: &Device) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("CaptureStream::for_device requires a CUDA device"),
        };
        let device_stream = dev.cuda_stream();
        let ctx = device_stream.context().clone();
        crate::gemma4_batch_graph::graph_teardown::disable_event_tracking_before_capture(&ctx);

        let (stream, owns_stream) = if device_stream.cu_stream().is_null() {
            let forked = ctx
                .new_stream()
                .map_err(|e| anyhow::anyhow!("CaptureStream: fork capture stream: {e:?}"))?;
            (forked, true)
        } else {
            (device_stream, false)
        };

        let cs = Self {
            stream,
            owns_stream,
        };
        nv_quant::nvfp4::ensure_workspace_for_stream(&cs.stream)?;
        let _ = nv_quant::matmul::TensorCoreGemm::new(cs.stream.clone())?;
        Ok(cs)
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn owns_stream(&self) -> bool {
        self.owns_stream
    }

    pub fn candle_launches_reach_this_stream(&self) -> bool {
        !self.owns_stream
    }

    pub fn require_capture_of_a_candle_body(&self, engine: &str) -> Result<()> {
        anyhow::ensure!(
            self.candle_launches_reach_this_stream(),
            "{engine}: refusing to capture a CUDA graph on a forked stream -- \
             {A_CANDLE_BODY_ESCAPES_A_FORKED_CAPTURE}. {A_FORKED_CAPTURE_IS_CORRECT_ONLY_FOR_A_RAW_BODY}"
        );
        Ok(())
    }

    pub fn forked_candle_capture_is_an_asserted_coincidence(&self, engine: &str, gate: &str) {
        if self.candle_launches_reach_this_stream() {
            return;
        }
        static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        ONCE.get_or_init(|| {
            eprintln!(
                "[capture-stream] {engine} is capturing a candle body on a FORKED stream. That is \
                 legal here only because the body's every launch happens to be an nv-* kernel \
                 that honours nv_layers::cuda_stream::with_stream -- measured, not assumed: on \
                 nvidia/Gemma-4-31B-IT-NVFP4 the captured graph has the same 4133 nodes whether \
                 the device is Device::new_cuda or Device::new_cuda_with_stream, so nothing \
                 escapes today. The suite that keeps it true is {gate}; it decodes through the \
                 graph and through the eager path and requires bit-equality across capture AND \
                 replay. {A_CANDLE_BODY_ESCAPES_A_FORKED_CAPTURE}"
            );
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_forked_capture_stream_refuses_a_candle_body_and_names_the_device_constructor() {
        let Ok(device) = Device::new_cuda(0) else {
            panic!(
                "no CUDA device 0: this suite is the only gate on the capture-stream policy and \
                 must not report success having executed nothing"
            );
        };
        let cs = CaptureStream::for_device(&device).expect("CaptureStream on the legacy device");
        assert!(
            cs.owns_stream(),
            "Device::new_cuda puts candle on the legacy NULL stream, which cuStreamBeginCapture \
             refuses, so for_device must fork and own the stream"
        );
        assert!(!cs.candle_launches_reach_this_stream());
        let err = cs
            .require_capture_of_a_candle_body("test-engine")
            .expect_err("a forked capture stream must refuse a candle body, not capture it");
        let msg = format!("{err:#}");
        for needle in [
            "Device::new_cuda_with_stream",
            "CU_STREAM_CAPTURE_MODE_RELAXED",
            "test-engine",
        ] {
            assert!(
                msg.contains(needle),
                "the refusal must name {needle}; it is the only durable record of why the \
                 configuration is illegal. got: {msg}"
            );
        }
    }

    #[test]
    fn a_device_built_with_a_stream_is_captured_on_that_very_stream() {
        let Ok(device) = Device::new_cuda_with_stream(0) else {
            panic!(
                "no CUDA device 0: this suite is the only gate on the capture-stream policy and \
                 must not report success having executed nothing"
            );
        };
        let Device::Cuda(d) = &device else {
            unreachable!()
        };
        let device_stream = d.cuda_stream();
        assert!(
            !device_stream.cu_stream().is_null(),
            "Device::new_cuda_with_stream must hand candle a real non-NULL stream"
        );
        let cs = CaptureStream::for_device(&device).expect("CaptureStream on the stream device");
        assert!(!cs.owns_stream());
        assert!(cs.candle_launches_reach_this_stream());
        assert_eq!(
            cs.stream().cu_stream(),
            device_stream.cu_stream(),
            "the capture stream must be the SAME handle candle launches on, or every candle op in \
             the captured body escapes the capture"
        );
        cs.require_capture_of_a_candle_body("test-engine")
            .expect("a device-stream capture must be accepted");
    }

    #[test]
    fn for_device_disables_event_tracking_before_it_allocates_the_nvfp4_workspace() {
        let Ok(device) = Device::new_cuda_with_stream(0) else {
            panic!(
                "no CUDA device 0: this suite is the only gate on the capture-stream policy and \
                 must not report success having executed nothing"
            );
        };
        let Device::Cuda(d) = &device else {
            unreachable!()
        };
        let ctx = d.cuda_stream().context().clone();
        let cs = CaptureStream::for_device(&device).expect("CaptureStream on the stream device");
        assert!(
            !ctx.is_event_tracking(),
            "CaptureStream::for_device must disable cudarc event tracking BEFORE it allocates the \
             64 MiB nvfp4 workspace and before the engine allocates any buffer the capture will \
             touch; a slice allocated with tracking on carries read/write CudaEvents that record \
             into the capture and then record an error on teardown. Constructing a CaptureStream \
             is therefore not free -- it disables tracking for the whole CONTEXT -- so an engine \
             whose eager candle path is still live must not build one in its constructor: {}",
            crate::graph_engine::THE_EAGER_ALL_NAN_IS_INSTALL_GROUPED_MOE_NOT_THIS_DISABLE_AND_NEEDS_A_HOST_STALL
        );
        assert!(
            std::ptr::eq(
                Arc::as_ptr(cs.stream().context()),
                Arc::as_ptr(d.cuda_stream().context())
            ),
            "the capture stream must belong to the context whose tracking was just disabled, or \
             this test disarmed a different context than the engine will capture on"
        );
    }
}
