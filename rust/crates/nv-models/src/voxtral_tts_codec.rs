use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use half::bf16;
use nv_layers::cpu_qmatvec::{arm, matmul_rows_bf16, matmul_rows_f32};

pub const CODEC_FRAME_TOKENS: usize = 37;
pub const CODEC_SAMPLE_RATE: usize = 24000;
pub const CODEC_SAMPLES_PER_FRAME: usize = 1920;
pub const CODEC_END_OF_AUDIO_SEMANTIC_ID: i32 = 1;
pub const CODEC_N_AUDIO_SPECIAL: i32 = 2;
pub const CODEC_WEIGHTS_FILE: &str = "consolidated.safetensors";

const DIM: usize = 1024;
const HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const FFN: usize = 4096;
const LATENT: usize = 292;
const SEMANTIC_ROWS: usize = 8192;
const SEMANTIC_DIM: usize = 256;
const PATCH: usize = 240;
const ACOUSTIC_LEVELS: f32 = 21.0;
const STAGE_BLOCKS: [usize; 4] = [1, 3, 5, 7];
const STAGE_WINDOWS: [usize; 4] = [2, 4, 8, 16];
const UPSAMPLER_BLOCKS: [usize; 3] = [2, 4, 6];
const INPUT_CONV_TAPS: usize = 3;
const UPSAMPLER_TAPS: usize = 2;
const OUTPUT_CONV_TAPS: usize = 7;
const QK_NORM_EPS: f32 = 1e-6;
const BLOCK_NORM_EPS: f32 = 1e-2;
const CLUSTER_USAGE_FLOOR: f32 = 1e-5;
const ALIBI_SLOPE_RATIO_FOR_8_HEADS_IS_ONE_HALF_SO_HEAD_0_HAS_SLOPE_1: f32 = 0.5;
#[cfg(feature = "cuda")]
const SCRATCH_GROWS_TO_THE_LARGEST_CALL_AND_STARTS_AT_THIS_MANY_FRAMES: usize = 64;

pub const THE_CONVS_FOLD_AND_RUN_IN_F32_ON_EVERY_DEVICE: &str =
    "the weight-norm fold is computed in f32 and its convs multiply an f32 copy of their input; \
     rounding the folded weights to bf16 moves the waveform enough that the STT port drops the \
     Spanish fixture's diacritics, so the three transformer stages stay bf16 and the input conv, \
     the upsamplers and the output conv do not";

pub const THE_CPU_CODEC_IS_ONE_HAND_WRITTEN_STEP_OVER_HOST_BF16_WEIGHTS: &str =
    "on CPU a frame runs as host f32 loops around matmul_rows_bf16: each causal conv is one \
     M-row matvec over its taps stacked per output row, q, k and v share one weight read and so \
     do w1 and w3, the windowed ALiBi attention, the norms and the layer scales are loops over \
     the rows, the transformer weights are the checkpoint's bf16 bits and the folded weight-norm \
     convs stay the f32 values the fold computes, so a frame reads each weight once";

pub const THE_CUDA_CODEC_IS_ONE_GRAPH_PER_CHUNK_SIZE_OVER_DEVICE_RESIDENT_STATE: &str =
    "on CUDA a chunk of n frames is one CUDA graph keyed on n: the host builds the latent rows \
     and uploads them, the graph runs the im2col, the cuBLAS GEMMs (bf16 with f32 accumulation \
     for the stages, f32 for the folded convs), the fused norm, residual, attention and ring \
     commit kernels against a frame counter, the k-1 trailing conv columns and a window-slot K/V \
     ring per layer that live in device buffers, and the host reads the samples back once; the \
     scratch and the graphs belong to the state because the graph bakes their addresses";

pub const THE_CUDA_CODEC_OWNS_A_NON_BLOCKING_STREAM_AND_A_CUBLAS_HANDLE_BECAUSE_THE_ROUTE_DECODES_BESIDE_A_REPLAYING_FRAME_GRAPH: &str =
    "the route's decoder thread runs the codec while the synthesis thread replays the TTS frame \
     graph on the candle device stream with the device's cuBLAS handle; a capture on that stream \
     would swallow the other thread's launches and two threads on one handle race on its stream \
     binding, so the codec forks its own non-blocking stream, binds its own handle to it for every \
     GEMM, serialises its decodes behind one mutex, and takes the capture-exclusion gate around \
     the handle's creation and destruction, which synchronise the device";

struct HostLayer {
    qkv: Vec<u16>,
    wo: Vec<u16>,
    gate_up: Vec<u16>,
    w2: Vec<u16>,
    q_norm: Vec<u16>,
    k_norm: Vec<u16>,
    attention_norm: Vec<u16>,
    ffn_norm: Vec<u16>,
    attention_scale: Vec<u16>,
    ffn_scale: Vec<u16>,
}

struct HostWeights {
    semantic: Vec<u16>,
    input_conv: Vec<f32>,
    layers: Vec<HostLayer>,
    upsamplers: [Vec<f32>; 3],
    output_conv: Vec<f32>,
}

enum Arm {
    Cpu(CpuCodec),
    #[cfg(feature = "cuda")]
    Cuda(CudaCodec),
}

#[derive(Default)]
pub struct CodecState {
    cpu: CpuCodecState,
    #[cfg(feature = "cuda")]
    cuda: Option<CudaCodecState>,
}

pub struct VoxtralTtsCodec {
    arm: Arm,
    device: Device,
}

pub fn frames_before_end_of_audio(codes: &[i32]) -> &[i32] {
    let frames = codes.len() / CODEC_FRAME_TOKENS;
    let cut = (0..frames)
        .find(|&t| codes[t * CODEC_FRAME_TOKENS] == CODEC_END_OF_AUDIO_SEMANTIC_ID)
        .unwrap_or(frames);
    &codes[..cut * CODEC_FRAME_TOKENS]
}

fn bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn frame_codes(frame: &[i32]) -> Result<(usize, [f32; CODEC_FRAME_TOKENS - 1])> {
    let s = frame[0] - CODEC_N_AUDIO_SPECIAL;
    ensure!((0..SEMANTIC_ROWS as i32).contains(&s), "semantic code {} out of range", frame[0]);
    let mut acoustic = [0f32; CODEC_FRAME_TOKENS - 1];
    for (a, &c) in acoustic.iter_mut().zip(&frame[1..]) {
        let c = c - CODEC_N_AUDIO_SPECIAL;
        ensure!((0..ACOUSTIC_LEVELS as i32).contains(&c), "acoustic code {} out of range", c);
        *a = (c * 2) as f32 / (ACOUSTIC_LEVELS - 1.0) - 1.0;
    }
    Ok((s as usize, acoustic))
}

fn alibi_slopes() -> [f32; HEADS] {
    let mut slopes = [1f32; HEADS];
    for h in 1..HEADS {
        slopes[h] = slopes[h - 1] * ALIBI_SLOPE_RATIO_FOR_8_HEADS_IS_ONE_HALF_SO_HEAD_0_HAS_SLOPE_1;
    }
    slopes
}

fn fold_weight_norm(weights: &candle_core::safetensors::MmapedSafetensors, prefix: &str) -> Result<Tensor> {
    let g = weights
        .load(&format!("{prefix}.parametrizations.weight.original0"), &Device::Cpu)?
        .to_dtype(DType::F32)?;
    let v = weights
        .load(&format!("{prefix}.parametrizations.weight.original1"), &Device::Cpu)?
        .to_dtype(DType::F32)?;
    let norm = v.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
    Ok(v.broadcast_mul(&g.broadcast_div(&norm)?)?)
}

fn transpose_conv_as_causal_conv(w_in_out_k: &Tensor) -> Result<Tensor> {
    let w = w_in_out_k.transpose(0, 1)?;
    let phase = |j: usize| -> Result<Tensor> { Ok(Tensor::stack(&[w.i((.., .., j + 2))?, w.i((.., .., j))?], 2)?) };
    Ok(Tensor::cat(&[phase(0)?, phase(1)?], 0)?.contiguous()?)
}

fn conv_rows_out_tap_in(w_out_in_k: &Tensor) -> Result<Vec<f32>> {
    Ok(w_out_in_k.transpose(1, 2)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?)
}

impl HostWeights {
    fn load(dir: &Path) -> Result<Self> {
        let path = dir.join(CODEC_WEIGHTS_FILE);
        let weights = unsafe { candle_core::safetensors::MmapedSafetensors::new(&path) }
            .with_context(|| format!("mmap {}", path.display()))?;
        let tensor = |name: &str| -> Result<Tensor> {
            weights
                .load(&format!("audio_tokenizer.{name}"), &Device::Cpu)
                .with_context(|| format!("load audio_tokenizer.{name}"))
        };
        let bits = |name: &str| -> Result<Vec<u16>> {
            Ok(tensor(name)?.to_dtype(DType::BF16)?.flatten_all()?.to_vec1::<bf16>()?.iter().map(|v| v.to_bits()).collect())
        };
        let cat = |names: &[String]| -> Result<Vec<u16>> {
            let mut out = Vec::new();
            for n in names {
                out.extend(bits(n)?);
            }
            Ok(out)
        };
        let usage: Vec<f32> = tensor("quantizer.semantic_codebook.cluster_usage")?.to_dtype(DType::F32)?.to_vec1()?;
        let sums: Vec<f32> = tensor("quantizer.semantic_codebook.embedding_sum")?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        ensure!(usage.len() == SEMANTIC_ROWS && sums.len() == SEMANTIC_ROWS * SEMANTIC_DIM, "semantic codebook shape");
        let semantic = sums
            .iter()
            .enumerate()
            .map(|(i, s)| bf16::from_f32(s / usage[i / SEMANTIC_DIM].max(CLUSTER_USAGE_FLOOR)).to_bits())
            .collect();
        let layer = |block: usize, l: usize| -> Result<HostLayer> {
            let p = format!("decoder_blocks.{block}.layers.{l}");
            Ok(HostLayer {
                qkv: cat(&[format!("{p}.attention.wq.weight"), format!("{p}.attention.wk.weight"), format!("{p}.attention.wv.weight")])?,
                wo: bits(&format!("{p}.attention.wo.weight"))?,
                gate_up: cat(&[format!("{p}.feed_forward.w1.weight"), format!("{p}.feed_forward.w3.weight")])?,
                w2: bits(&format!("{p}.feed_forward.w2.weight"))?,
                q_norm: bits(&format!("{p}.attention.q_norm.weight"))?,
                k_norm: bits(&format!("{p}.attention.k_norm.weight"))?,
                attention_norm: bits(&format!("{p}.attention_norm.weight"))?,
                ffn_norm: bits(&format!("{p}.ffn_norm.weight"))?,
                attention_scale: bits(&format!("{p}.attention_scale"))?,
                ffn_scale: bits(&format!("{p}.ffn_scale"))?,
            })
        };
        let _ = THE_CONVS_FOLD_AND_RUN_IN_F32_ON_EVERY_DEVICE;
        let upsampler = |block: usize| -> Result<Vec<f32>> {
            conv_rows_out_tap_in(&transpose_conv_as_causal_conv(&fold_weight_norm(
                &weights,
                &format!("audio_tokenizer.decoder_blocks.{block}.conv"),
            )?)?)
        };
        Ok(Self {
            semantic,
            input_conv: conv_rows_out_tap_in(&fold_weight_norm(&weights, "audio_tokenizer.decoder_blocks.0.conv")?)?,
            layers: STAGE_BLOCKS.iter().flat_map(|&b| [(b, 0), (b, 1)]).map(|(b, l)| layer(b, l)).collect::<Result<_>>()?,
            upsamplers: [upsampler(UPSAMPLER_BLOCKS[0])?, upsampler(UPSAMPLER_BLOCKS[1])?, upsampler(UPSAMPLER_BLOCKS[2])?],
            output_conv: conv_rows_out_tap_in(&fold_weight_norm(&weights, "audio_tokenizer.output_proj.conv")?)?,
        })
    }
}

impl VoxtralTtsCodec {
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let host = HostWeights::load(dir)?;
        let arm = match device {
            Device::Cpu => Arm::Cpu(CpuCodec::new(host)?),
            #[cfg(feature = "cuda")]
            Device::Cuda(dev) => Arm::Cuda(CudaCodec::new(host, dev)?),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => bail!("the Voxtral TTS codec's CUDA arm needs the cuda feature"),
            Device::Metal(_) => bail!("the Voxtral TTS codec has a CUDA arm, a CPU arm and a wgpu twin, not a Metal arm"),
        };
        Ok(Self { arm, device: device.clone() })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        DType::BF16
    }

    pub fn decode(&self, codes: &[i32]) -> Result<Vec<f32>> {
        self.decode_frames(&mut CodecState::default(), frames_before_end_of_audio(codes))
    }

    pub fn decode_frames(&self, state: &mut CodecState, frames: &[i32]) -> Result<Vec<f32>> {
        ensure!(
            !frames.is_empty() && frames.len() % CODEC_FRAME_TOKENS == 0,
            "codec frames are [T, {CODEC_FRAME_TOKENS}] with T >= 1, got {} values",
            frames.len()
        );
        match &self.arm {
            Arm::Cpu(cpu) => cpu.decode_frames(&mut state.cpu, frames),
            #[cfg(feature = "cuda")]
            Arm::Cuda(cuda) => cuda.decode_frames(&mut state.cuda, frames),
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use cudarc::cublas::{sys as cb, CudaBlas};
    use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr, ValidAsZeroBits};
    use nv_kernels::capture_exclusion::no_capture_in_flight;
    use nv_kernels::graph::CudaGraphRunner;
    use nv_kernels::voxtral_tts_codec as k;
    use std::ffi::c_void;
    use std::mem::ManuallyDrop;
    use std::sync::{Arc, Mutex};

    struct Lin<T> {
        w: CudaSlice<T>,
        n: usize,
        k: usize,
        name: &'static str,
    }

    struct Conv {
        lin: Lin<f32>,
        channels: usize,
        taps: usize,
        pad: i32,
    }

    struct Layer {
        qkv: Lin<u16>,
        wo: Lin<u16>,
        gate_up: Lin<u16>,
        w2: Lin<u16>,
        q_norm: CudaSlice<u16>,
        k_norm: CudaSlice<u16>,
        attention_norm: CudaSlice<u16>,
        ffn_norm: CudaSlice<u16>,
        attention_scale: CudaSlice<u16>,
        ffn_scale: CudaSlice<u16>,
    }

    pub(super) struct CudaCodec {
        stream: Arc<CudaStream>,
        blas: ManuallyDrop<CudaBlas>,
        decodes: Mutex<()>,
        semantic: Vec<u16>,
        input_conv: Conv,
        layers: Vec<Layer>,
        upsamplers: [Conv; 3],
        output_conv: Conv,
    }

    struct Scratch {
        cap: usize,
        lat: CudaSlice<u16>,
        cols: CudaSlice<f32>,
        conv_out: CudaSlice<f32>,
        x: CudaSlice<u16>,
        h: CudaSlice<u16>,
        qkv: CudaSlice<u16>,
        qn: CudaSlice<u16>,
        kn: CudaSlice<u16>,
        o: CudaSlice<u16>,
        t: CudaSlice<u16>,
        gu: CudaSlice<u16>,
        act: CudaSlice<u16>,
        out: CudaSlice<f32>,
    }

    pub(super) struct CudaCodecState {
        stream: Arc<CudaStream>,
        runner: CudaGraphRunner,
        ctl: CudaSlice<i32>,
        lat_hist: CudaSlice<u16>,
        hist_up: [CudaSlice<u16>; 3],
        hist_out: CudaSlice<u16>,
        rings: Vec<(CudaSlice<u16>, CudaSlice<u16>)>,
        scratch: Scratch,
    }

    impl Drop for CudaCodecState {
        fn drop(&mut self) {
            let teardown = nv_quant::graph_teardown::GraphTeardown::new(&self.stream);
            let runner = &mut self.runner;
            teardown.run_owned_streams_only(|| runner.invalidate());
        }
    }

    impl Drop for CudaCodec {
        fn drop(&mut self) {
            let _device_sync_gate = no_capture_in_flight();
            let _ = self.stream.synchronize();
            unsafe { ManuallyDrop::drop(&mut self.blas) };
        }
    }

    fn ptr<T>(s: &CudaSlice<T>, stream: &CudaStream) -> *mut T {
        s.device_ptr(stream).0 as *mut T
    }

    fn rc(code: i32, what: &str) -> Result<()> {
        ensure!(code == 0, "voxtral tts codec {what} returned {code}");
        Ok(())
    }

    fn zeros<T: DeviceRepr + ValidAsZeroBits>(stream: &Arc<CudaStream>, len: usize) -> Result<CudaSlice<T>> {
        Ok(stream.alloc_zeros::<T>(len.max(1))?)
    }

    impl CudaCodec {
        pub(super) fn new(host: HostWeights, dev: &candle_core::CudaDevice) -> Result<Self> {
            let _ = THE_CUDA_CODEC_OWNS_A_NON_BLOCKING_STREAM_AND_A_CUBLAS_HANDLE_BECAUSE_THE_ROUTE_DECODES_BESIDE_A_REPLAYING_FRAME_GRAPH;
            let ctx = dev.cuda_stream().context().clone();
            nv_quant::graph_teardown::disable_event_tracking_before_capture(&ctx);
            let (stream, blas) = {
                let _device_sync_gate = no_capture_in_flight();
                let stream = ctx.new_stream()?;
                let blas = CudaBlas::new(stream.clone())?;
                (stream, blas)
            };
            let up = |name: &'static str, v: &[u16], n: usize, k: usize| -> Result<Lin<u16>> {
                ensure!(v.len() == n * k, "{name}: {} elements, want {n}x{k}", v.len());
                Ok(Lin { w: stream.clone_htod(v)?, n, k, name })
            };
            let conv = |name: &'static str, v: &[f32], n: usize, channels: usize, taps: usize, pad: i32| -> Result<Conv> {
                ensure!(v.len() == n * channels * taps, "{name}: {} elements, want {n}x{taps}x{channels}", v.len());
                Ok(Conv { lin: Lin { w: stream.clone_htod(v)?, n, k: channels * taps, name }, channels, taps, pad })
            };
            let vec = |v: &[u16]| -> Result<CudaSlice<u16>> { Ok(stream.clone_htod(v)?) };
            let layers = host
                .layers
                .iter()
                .map(|l| {
                    Ok(Layer {
                        qkv: up("qkv", &l.qkv, 3 * DIM, DIM)?,
                        wo: up("wo", &l.wo, DIM, DIM)?,
                        gate_up: up("gate_up", &l.gate_up, 2 * FFN, DIM)?,
                        w2: up("w2", &l.w2, DIM, FFN)?,
                        q_norm: vec(&l.q_norm)?,
                        k_norm: vec(&l.k_norm)?,
                        attention_norm: vec(&l.attention_norm)?,
                        ffn_norm: vec(&l.ffn_norm)?,
                        attention_scale: vec(&l.attention_scale)?,
                        ffn_scale: vec(&l.ffn_scale)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let codec = Self {
                blas: ManuallyDrop::new(blas),
                decodes: Mutex::new(()),
                semantic: host.semantic,
                input_conv: conv("input_conv", &host.input_conv, DIM, LATENT, INPUT_CONV_TAPS, k::PAD_REPLICATE)?,
                layers,
                upsamplers: [
                    conv("upsampler", &host.upsamplers[0], 2 * DIM, DIM, UPSAMPLER_TAPS, k::PAD_ZERO)?,
                    conv("upsampler", &host.upsamplers[1], 2 * DIM, DIM, UPSAMPLER_TAPS, k::PAD_ZERO)?,
                    conv("upsampler", &host.upsamplers[2], 2 * DIM, DIM, UPSAMPLER_TAPS, k::PAD_ZERO)?,
                ],
                output_conv: conv("output_conv", &host.output_conv, PATCH, DIM, OUTPUT_CONV_TAPS, k::PAD_REFLECT)?,
                stream,
            };
            let (x16, y16) = (zeros::<u16>(&codec.stream, DIM)?, zeros::<u16>(&codec.stream, 3 * DIM)?);
            codec.gemm_bf16(&x16, &codec.layers[0].qkv, &y16, 1)?;
            let (x32, y32) = (zeros::<f32>(&codec.stream, codec.input_conv.lin.k)?, zeros::<f32>(&codec.stream, DIM)?);
            let lin = &codec.input_conv.lin;
            codec.gemm(cb::cudaDataType_t::CUDA_R_32F, ptr(&lin.w, &codec.stream) as u64, lin.n, lin.k, ptr(&x32, &codec.stream) as u64, ptr(&y32, &codec.stream) as u64, 1, lin.name)?;
            codec.stream.synchronize()?;
            Ok(codec)
        }

        fn scratch(&self, cap: usize) -> Result<Scratch> {
            let s = &self.stream;
            let rows = 8 * cap;
            Ok(Scratch {
                cap,
                lat: zeros(s, cap * LATENT)?,
                cols: zeros(s, rows * OUTPUT_CONV_TAPS * DIM)?,
                conv_out: zeros(s, rows * DIM)?,
                x: zeros(s, rows * DIM)?,
                h: zeros(s, rows * DIM)?,
                qkv: zeros(s, rows * 3 * DIM)?,
                qn: zeros(s, rows * DIM)?,
                kn: zeros(s, rows * DIM)?,
                o: zeros(s, rows * DIM)?,
                t: zeros(s, rows * DIM)?,
                gu: zeros(s, rows * 2 * FFN)?,
                act: zeros(s, rows * FFN)?,
                out: zeros(s, rows * PATCH)?,
            })
        }

        fn state(&self, n: usize) -> Result<CudaCodecState> {
            let s = &self.stream;
            Ok(CudaCodecState {
                stream: s.clone(),
                runner: CudaGraphRunner::new(s.clone()),
                ctl: zeros(s, 1)?,
                lat_hist: zeros(s, (INPUT_CONV_TAPS - 1) * LATENT)?,
                hist_up: [zeros(s, DIM)?, zeros(s, DIM)?, zeros(s, DIM)?],
                hist_out: zeros(s, (OUTPUT_CONV_TAPS - 1) * DIM)?,
                rings: (0..2 * STAGE_WINDOWS.len())
                    .map(|i| Ok((zeros(s, STAGE_WINDOWS[i / 2] * DIM)?, zeros(s, STAGE_WINDOWS[i / 2] * DIM)?)))
                    .collect::<Result<_>>()?,
                scratch: self.scratch(n.max(SCRATCH_GROWS_TO_THE_LARGEST_CALL_AND_STARTS_AT_THIS_MANY_FRAMES))?,
            })
        }

        pub(super) fn decode_frames(&self, state: &mut Option<CudaCodecState>, frames: &[i32]) -> Result<Vec<f32>> {
            let _ = THE_CUDA_CODEC_IS_ONE_GRAPH_PER_CHUNK_SIZE_OVER_DEVICE_RESIDENT_STATE;
            let _one_decode_at_a_time_on_the_codec_stream_and_handle = self.decodes.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            self.stream.context().bind_to_thread()?;
            let n = frames.len() / CODEC_FRAME_TOKENS;
            let mut lat = Vec::with_capacity(n * LATENT);
            for frame in frames.chunks_exact(CODEC_FRAME_TOKENS) {
                let (s, acoustic) = frame_codes(frame)?;
                lat.extend_from_slice(&self.semantic[s * SEMANTIC_DIM..][..SEMANTIC_DIM]);
                lat.extend(acoustic.iter().map(|a| bf16::from_f32(*a).to_bits()));
            }
            let st = match state {
                Some(st) => st,
                None => state.insert(self.state(n)?),
            };
            if st.scratch.cap < n {
                st.runner.invalidate();
                st.scratch = self.scratch(n)?;
            }
            self.stream.memcpy_htod(&lat, &mut st.scratch.lat.slice_mut(0..lat.len()))?;
            let CudaCodecState { runner, ctl, lat_hist, hist_up, hist_out, rings, scratch, .. } = st;
            runner.run(n as u64, |_| self.chunk(n, ctl, lat_hist, hist_up, hist_out, rings, scratch))?;
            let samples = self.stream.clone_dtoh(&scratch.out.slice(0..n * 8 * PATCH))?;
            Ok(samples.iter().map(|s| s.clamp(-1.0, 1.0)).collect())
        }

        fn cu(&self) -> *mut c_void {
            self.stream.cu_stream() as *mut c_void
        }

        fn gemm(&self, ty: cb::cudaDataType_t, lin_w: u64, n: usize, kk: usize, x: u64, y: u64, m: usize, name: &str) -> Result<()> {
            let (alpha, beta) = (1f32, 0f32);
            unsafe {
                cudarc::cublas::result::gemm_ex(
                    *self.blas.handle(),
                    cb::cublasOperation_t::CUBLAS_OP_T,
                    cb::cublasOperation_t::CUBLAS_OP_N,
                    n as i32,
                    m as i32,
                    kk as i32,
                    &alpha as *const f32 as *const c_void,
                    lin_w as *const c_void,
                    ty,
                    kk as i32,
                    x as *const c_void,
                    ty,
                    kk as i32,
                    &beta as *const f32 as *const c_void,
                    y as *mut c_void,
                    ty,
                    n as i32,
                    cb::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cb::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
            }
            .with_context(|| format!("voxtral tts codec gemm {name} m={m}"))
        }

        fn gemm_bf16(&self, x: &CudaSlice<u16>, lin: &Lin<u16>, y: &CudaSlice<u16>, m: usize) -> Result<()> {
            let s = &self.stream;
            self.gemm(cb::cudaDataType_t::CUDA_R_16BF, ptr(&lin.w, s) as u64, lin.n, lin.k, ptr(x, s) as u64, ptr(y, s) as u64, m, lin.name)
        }

        fn conv(&self, conv: &Conv, src: &CudaSlice<u16>, hist: &CudaSlice<u16>, ctl: &CudaSlice<i32>, sc: &Scratch, rows: usize, dst: &CudaSlice<f32>) -> Result<()> {
            let s = &self.stream;
            let (c, taps) = (conv.channels as i32, conv.taps as i32);
            rc(
                unsafe { k::nv_kernels_voxtral_tts_codec_im2col(self.cu(), ptr(src, s), ptr(hist, s), ptr(ctl, s), ptr(&sc.cols, s), rows as i32, c, taps, conv.pad) },
                "im2col",
            )?;
            rc(unsafe { k::nv_kernels_voxtral_tts_codec_carry(self.cu(), ptr(&sc.cols, s), ptr(hist, s), rows as i32, c, taps) }, "carry")?;
            let lin = &conv.lin;
            self.gemm(cb::cudaDataType_t::CUDA_R_32F, ptr(&lin.w, s) as u64, lin.n, lin.k, ptr(&sc.cols, s) as u64, ptr(dst, s) as u64, rows, lin.name)
        }

        fn cast_norm(&self, sc: &Scratch, w: &CudaSlice<u16>, rows: usize) -> Result<()> {
            let s = &self.stream;
            rc(
                unsafe { k::nv_kernels_voxtral_tts_codec_cast_norm(self.cu(), ptr(&sc.conv_out, s), ptr(&sc.x, s), ptr(w, s), ptr(&sc.h, s), rows as i32, BLOCK_NORM_EPS) },
                "cast_norm",
            )
        }

        fn residual_norm(&self, sc: &Scratch, scale: &CudaSlice<u16>, next: Option<&CudaSlice<u16>>, rows: usize) -> Result<()> {
            let s = &self.stream;
            let w = next.map_or(std::ptr::null(), |w| ptr(w, s) as *const u16);
            rc(
                unsafe { k::nv_kernels_voxtral_tts_codec_residual_norm(self.cu(), ptr(&sc.x, s), ptr(&sc.t, s), ptr(scale, s), w, ptr(&sc.h, s), rows as i32, BLOCK_NORM_EPS) },
                "residual_norm",
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn chunk(
            &self,
            n: usize,
            ctl: &CudaSlice<i32>,
            lat_hist: &CudaSlice<u16>,
            hist_up: &[CudaSlice<u16>; 3],
            hist_out: &CudaSlice<u16>,
            rings: &[(CudaSlice<u16>, CudaSlice<u16>)],
            sc: &Scratch,
        ) -> Result<()> {
            let s = &self.stream;
            let scale = (1.0 / (HEAD_DIM as f64).sqrt()) as f32;
            self.conv(&self.input_conv, &sc.lat, lat_hist, ctl, sc, n, &sc.conv_out)?;
            self.cast_norm(sc, &self.layers[0].attention_norm, n)?;
            for (stage, &window) in STAGE_WINDOWS.iter().enumerate() {
                let m = n << stage;
                let rpf = 1 << stage;
                for l in 0..2 {
                    let layer = &self.layers[2 * stage + l];
                    let ring = &rings[2 * stage + l];
                    self.gemm_bf16(&sc.h, &layer.qkv, &sc.qkv, m)?;
                    rc(
                        unsafe {
                            k::nv_kernels_voxtral_tts_codec_qk_norm(self.cu(), ptr(&sc.qkv, s), ptr(&layer.q_norm, s), ptr(&layer.k_norm, s), ptr(&sc.qn, s), ptr(&sc.kn, s), m as i32, QK_NORM_EPS)
                        },
                        "qk_norm",
                    )?;
                    rc(
                        unsafe {
                            k::nv_kernels_voxtral_tts_codec_attn(
                                self.cu(),
                                ptr(&sc.qn, s),
                                ptr(&sc.kn, s),
                                ptr(&sc.qkv, s),
                                ptr(&ring.0, s),
                                ptr(&ring.1, s),
                                ptr(ctl, s),
                                ptr(&sc.o, s),
                                m as i32,
                                window as i32,
                                rpf,
                                scale,
                            )
                        },
                        "attn",
                    )?;
                    rc(
                        unsafe {
                            k::nv_kernels_voxtral_tts_codec_kv_commit(self.cu(), ptr(&sc.kn, s), ptr(&sc.qkv, s), ptr(&ring.0, s), ptr(&ring.1, s), ptr(ctl, s), m as i32, window as i32, rpf)
                        },
                        "kv_commit",
                    )?;
                    self.gemm_bf16(&sc.o, &layer.wo, &sc.t, m)?;
                    self.residual_norm(sc, &layer.attention_scale, Some(&layer.ffn_norm), m)?;
                    self.gemm_bf16(&sc.h, &layer.gate_up, &sc.gu, m)?;
                    rc(unsafe { k::nv_kernels_voxtral_tts_codec_silu_mul(self.cu(), ptr(&sc.gu, s), ptr(&sc.act, s), m as i32, FFN as i32) }, "silu_mul")?;
                    self.gemm_bf16(&sc.act, &layer.w2, &sc.t, m)?;
                    let next = (l == 0).then(|| &self.layers[2 * stage + 1].attention_norm);
                    self.residual_norm(sc, &layer.ffn_scale, next, m)?;
                }
                if stage < UPSAMPLER_BLOCKS.len() {
                    self.conv(&self.upsamplers[stage], &sc.x, &hist_up[stage], ctl, sc, m, &sc.conv_out)?;
                    self.cast_norm(sc, &self.layers[2 * stage + 2].attention_norm, 2 * m)?;
                }
            }
            self.conv(&self.output_conv, &sc.x, hist_out, ctl, sc, 8 * n, &sc.out)?;
            rc(unsafe { k::nv_kernels_voxtral_tts_codec_advance(self.cu(), ptr(ctl, s), n as i32) }, "advance")
        }
    }
}

#[cfg(feature = "cuda")]
use cuda::{CudaCodec, CudaCodecState};

enum CpuRows {
    Bf16(Vec<u16>),
    F32(Vec<f32>),
}

struct CpuLinear {
    w: CpuRows,
    n: usize,
    k: usize,
}

impl CpuLinear {
    fn bf16(w: Vec<u16>, n: usize) -> Result<Self> {
        ensure!(!w.is_empty() && w.len() % n == 0, "{} bf16 weights do not hold {n} rows", w.len());
        let k = w.len() / n;
        Ok(Self { w: CpuRows::Bf16(w), n, k })
    }

    fn conv(w: Vec<f32>, n: usize) -> Result<Self> {
        ensure!(!w.is_empty() && w.len() % n == 0, "{} f32 weights do not hold {n} rows", w.len());
        let k = w.len() / n;
        Ok(Self { w: CpuRows::F32(w), n, k })
    }

    fn forward_columns(&self, x: &[f32]) -> Result<(Vec<f32>, usize)> {
        let m = x.len() / self.k;
        let mut out_t = vec![0f32; self.n * m];
        match &self.w {
            CpuRows::Bf16(w) => matmul_rows_bf16(arm(), w, self.k, x, &mut out_t)?,
            CpuRows::F32(w) => matmul_rows_f32(arm(), w, self.k, x, &mut out_t)?,
        }
        Ok((out_t, m))
    }

    fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let (out_t, m) = self.forward_columns(x)?;
        Ok(if m == 1 { out_t } else { transpose_blocked(&out_t, self.n, m) })
    }

    fn forward_silu_mul(&self, x: &[f32]) -> Result<Vec<f32>> {
        let (gu, m) = self.forward_columns(x)?;
        let inter = self.n / 2;
        let silu_mul = |g: f32, u: f32| g / (1.0 + (-g).exp()) * u;
        if m == 1 {
            return Ok((0..inter).map(|i| silu_mul(gu[i], gu[inter + i])).collect());
        }
        let mut act = vec![0f32; m * inter];
        for i0 in (0..inter).step_by(TRANSPOSE_TILE) {
            for j in 0..m {
                for i in i0..(i0 + TRANSPOSE_TILE).min(inter) {
                    act[j * inter + i] = silu_mul(gu[i * m + j], gu[(inter + i) * m + j]);
                }
            }
        }
        Ok(act)
    }
}

const TRANSPOSE_TILE: usize = 16;

fn transpose_blocked(out_t: &[f32], n: usize, m: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * m];
    for i0 in (0..n).step_by(TRANSPOSE_TILE) {
        for j in 0..m {
            for i in i0..(i0 + TRANSPOSE_TILE).min(n) {
                out[j * n + i] = out_t[i * m + j];
            }
        }
    }
    out
}

struct CpuLayer {
    wqkv: CpuLinear,
    wo: CpuLinear,
    w13: CpuLinear,
    w2: CpuLinear,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    attention_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    attention_scale: Vec<f32>,
    ffn_scale: Vec<f32>,
}

struct CpuCodec {
    semantic: Vec<u16>,
    input_conv: CpuLinear,
    layers: Vec<CpuLayer>,
    upsamplers: Vec<CpuLinear>,
    output_conv: CpuLinear,
    slopes: [f32; HEADS],
}

#[derive(Default)]
pub struct CpuCodecState {
    input: Option<Vec<f32>>,
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    up: [Option<Vec<f32>>; 3],
    output: Option<Vec<f32>>,
}

fn rms_rows(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let n = w.len();
    for (x, o) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
        for i in 0..n {
            o[i] = x[i] * inv * w[i];
        }
    }
}

fn causal_conv(
    lin: &CpuLinear,
    x: &[f32],
    c: usize,
    taps: usize,
    hist: &mut Option<Vec<f32>>,
    first: impl FnOnce(&[f32]) -> Result<Vec<f32>>,
) -> Result<Vec<f32>> {
    let t = x.len() / c;
    let mut padded = match hist.take() {
        Some(h) => h,
        None => first(x)?,
    };
    padded.extend_from_slice(x);
    *hist = Some(padded[t * c..].to_vec());
    let mut rows = Vec::with_capacity(t * taps * c);
    for i in 0..t {
        rows.extend_from_slice(&padded[i * c..(i + taps) * c]);
    }
    lin.forward(&rows)
}

impl CpuCodec {
    fn new(host: HostWeights) -> Result<Self> {
        let _ = THE_CPU_CODEC_IS_ONE_HAND_WRITTEN_STEP_OVER_HOST_BF16_WEIGHTS;
        let f32s = |v: &[u16]| v.iter().map(|b| bits_to_f32(*b)).collect::<Vec<f32>>();
        let layers = host
            .layers
            .into_iter()
            .map(|l| {
                Ok(CpuLayer {
                    wqkv: CpuLinear::bf16(l.qkv, 3 * DIM)?,
                    wo: CpuLinear::bf16(l.wo, DIM)?,
                    w13: CpuLinear::bf16(l.gate_up, 2 * FFN)?,
                    w2: CpuLinear::bf16(l.w2, DIM)?,
                    q_norm: f32s(&l.q_norm),
                    k_norm: f32s(&l.k_norm),
                    attention_norm: f32s(&l.attention_norm),
                    ffn_norm: f32s(&l.ffn_norm),
                    attention_scale: f32s(&l.attention_scale),
                    ffn_scale: f32s(&l.ffn_scale),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let [u0, u1, u2] = host.upsamplers;
        Ok(Self {
            semantic: host.semantic,
            input_conv: CpuLinear::conv(host.input_conv, DIM)?,
            layers,
            upsamplers: vec![CpuLinear::conv(u0, 2 * DIM)?, CpuLinear::conv(u1, 2 * DIM)?, CpuLinear::conv(u2, 2 * DIM)?],
            output_conv: CpuLinear::conv(host.output_conv, PATCH)?,
            slopes: alibi_slopes(),
        })
    }

    fn decode_frames(&self, st: &mut CpuCodecState, frames: &[i32]) -> Result<Vec<f32>> {
        let t = frames.len() / CODEC_FRAME_TOKENS;
        let mut latent = Vec::with_capacity(t * LATENT);
        for frame in frames.chunks_exact(CODEC_FRAME_TOKENS) {
            let (s, acoustic) = frame_codes(frame)?;
            latent.extend(self.semantic[s * SEMANTIC_DIM..][..SEMANTIC_DIM].iter().map(|b| bits_to_f32(*b)));
            latent.extend_from_slice(&acoustic);
        }
        let mut x = causal_conv(&self.input_conv, &latent, LATENT, INPUT_CONV_TAPS, &mut st.input, |x| Ok(x[..LATENT].repeat(2)))?;
        if st.kv.is_empty() {
            st.kv = (0..self.layers.len()).map(|_| Default::default()).collect();
        }
        for (s, window) in STAGE_WINDOWS.iter().enumerate() {
            for l in 2 * s..2 * s + 2 {
                x = self.layer(&self.layers[l], &x, &mut st.kv[l], *window)?;
            }
            if s < UPSAMPLER_BLOCKS.len() {
                let y = causal_conv(&self.upsamplers[s], &x, DIM, UPSAMPLER_TAPS, &mut st.up[s], |_| Ok(vec![0f32; DIM]))?;
                x = y.chunks_exact(2 * DIM).flat_map(|r| r.iter().copied()).collect();
            }
        }
        let patches = causal_conv(&self.output_conv, &x, DIM, OUTPUT_CONV_TAPS, &mut st.output, |x| {
            ensure!(x.len() / DIM > 6, "reflect pad needs 6 columns beyond the first, got {}", x.len() / DIM);
            Ok((1..=6).rev().flat_map(|r| x[r * DIM..(r + 1) * DIM].iter().copied()).collect())
        })?;
        ensure!(self.output_conv.n == PATCH, "output patch width {} != {PATCH}", self.output_conv.n);
        Ok(patches.iter().map(|v| v.clamp(-1.0, 1.0)).collect())
    }

    fn layer(&self, l: &CpuLayer, x: &[f32], kv: &mut (Vec<f32>, Vec<f32>), window: usize) -> Result<Vec<f32>> {
        let n = x.len() / DIM;
        let mut h = vec![0f32; x.len()];
        rms_rows(x, &l.attention_norm, BLOCK_NORM_EPS, &mut h);
        let qkv = l.wqkv.forward(&h)?;
        let mut q = vec![0f32; n * DIM];
        for r in 0..n {
            let row = &qkv[r * 3 * DIM..(r + 1) * 3 * DIM];
            rms_rows(&row[..DIM], &l.q_norm, QK_NORM_EPS, &mut q[r * DIM..(r + 1) * DIM]);
            let at = kv.0.len();
            kv.0.resize(at + DIM, 0.0);
            rms_rows(&row[DIM..2 * DIM], &l.k_norm, QK_NORM_EPS, &mut kv.0[at..]);
            kv.1.extend_from_slice(&row[2 * DIM..]);
        }
        let m = kv.0.len() / DIM;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut attn = vec![0f32; n * DIM];
        let mut p = Vec::with_capacity(window + 1);
        for i in 0..n {
            let qpos = m - n + i;
            let j0 = qpos.saturating_sub(window);
            for hd in 0..HEADS {
                let qh = &q[i * DIM + hd * HEAD_DIM..][..HEAD_DIM];
                p.clear();
                for j in j0..=qpos {
                    let kh = &kv.0[j * DIM + hd * HEAD_DIM..][..HEAD_DIM];
                    p.push(qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale + self.slopes[hd] * (j as f32 - qpos as f32));
                }
                let max = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for v in p.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                let o = &mut attn[i * DIM + hd * HEAD_DIM..][..HEAD_DIM];
                for (jj, j) in (j0..=qpos).enumerate() {
                    let w = p[jj] / sum;
                    let vh = &kv.1[j * DIM + hd * HEAD_DIM..][..HEAD_DIM];
                    for (o, v) in o.iter_mut().zip(vh) {
                        *o += w * v;
                    }
                }
            }
        }
        let drop = (m - window.min(m)) * DIM;
        kv.0.drain(..drop);
        kv.1.drain(..drop);
        let o = l.wo.forward(&attn)?;
        let mut x: Vec<f32> = x.iter().zip(&o).enumerate().map(|(i, (x, o))| x + o * l.attention_scale[i % DIM]).collect();
        rms_rows(&x, &l.ffn_norm, BLOCK_NORM_EPS, &mut h);
        let f = l.w2.forward(&l.w13.forward_silu_mul(&h)?)?;
        for (i, (x, f)) in x.iter_mut().zip(&f).enumerate() {
            *x += f * l.ffn_scale[i % DIM];
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_causal_conv_rewrite_of_the_transposed_conv_matches_candle_conv_transpose1d_trimmed_on_the_right() {
        let dev = Device::Cpu;
        let (c_in, c_out, len) = (3, 5, 4);
        let w = Tensor::randn(0f32, 1.0, (c_in, c_out, 4), &dev).unwrap();
        let x = Tensor::randn(0f32, 1.0, (1, c_in, len), &dev).unwrap();
        let full = x.conv_transpose1d(&w, 0, 0, 2, 1, 1).unwrap().narrow(2, 0, 2 * len).unwrap();
        let padded = Tensor::cat(&[x.narrow(2, 0, 1).unwrap().zeros_like().unwrap(), x], 2).unwrap();
        let y = padded.conv1d(&transpose_conv_as_causal_conv(&w).unwrap(), 0, 1, 1, 1).unwrap();
        let ours = y.reshape((2, c_out, len)).unwrap().permute((1, 2, 0)).unwrap().reshape((1, c_out, 2 * len)).unwrap();
        let diff = (full - ours).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-5, "transposed conv rewrite differs by {diff}");
    }

    #[test]
    fn frames_cut_at_the_first_end_of_audio_frame() {
        let mut codes = vec![5; CODEC_FRAME_TOKENS * 3];
        codes[CODEC_FRAME_TOKENS * 2] = CODEC_END_OF_AUDIO_SEMANTIC_ID;
        assert_eq!(frames_before_end_of_audio(&codes).len(), CODEC_FRAME_TOKENS * 2);
        assert_eq!(frames_before_end_of_audio(&codes[..CODEC_FRAME_TOKENS * 2]).len(), CODEC_FRAME_TOKENS * 2);
    }
}
