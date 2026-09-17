use anyhow::{anyhow, Result};
use std::collections::BTreeMap;

pub use nv_weights::WeightLoader;

pub trait WeightSource {
    fn shard_names(&self) -> Vec<String>;
    fn take_shard(&mut self, name: &str) -> Result<Vec<u8>>;
}

impl WeightSource for BTreeMap<String, Vec<u8>> {
    fn shard_names(&self) -> Vec<String> {
        self.keys().cloned().collect()
    }
    fn take_shard(&mut self, name: &str) -> Result<Vec<u8>> {
        self.remove(name)
            .ok_or_else(|| anyhow!("missing weight shard {name}"))
    }
}

#[cfg(feature = "browser")]
pub mod q38;

#[cfg(feature = "browser")]
pub mod train;

pub mod kokoro_web;
#[cfg(all(feature = "browser", target_arch = "wasm32"))]
pub mod wasm;
#[cfg(all(feature = "browser", target_arch = "wasm32"))]
pub mod wasm_kokoro;

#[cfg(feature = "browser")]
mod browser {
    use super::WeightSource;
    use anyhow::{Context, Result};
    use candle_core::Device;
    use nv_models::dots_ocr::{
        DotsMode, DotsOcrPipeline, GenerateOptions, LayoutElement, RgbImage,
    };
    use nv_weights::WeightLoader;

    pub const BROWSER_KV_SPAN_DEFAULT_4096_KEEPS_PER_LAYER_KV_UNDER_5_MB_AND_CAPS_LOAD_TIME: usize =
        4096;

    pub const CONFIG_JSON_NAME: &str = "config.json";
    pub const TOKENIZER_JSON_NAME: &str = "tokenizer.json";

    pub const MLX_CONFIG_QUANTIZATION_BLOCK_IS_THE_EXPLICIT_OPT_IN_TO_THE_PREQUANTIZED_GRAPH: &str =
        "serving the mlx-community dots.ocr-4bit files IS the opt-in: its config.json declares \
         {mode: affine, bits: 4, group_size: 64} and init builds the graph from the snapshot's \
         own u4 words and bf16 scale/bias rows (bitwise the load-time quantizer's output, so the \
         same graph the rednote arm builds) instead of quantizing a 5.7 GB bf16 checkpoint that \
         cannot transit wasm32's 4 GiB linear memory; a rednote config.json carries no \
         quantization block and keeps the default-off rednote path per the flip law";

    pub const ATTN_REG_STAGES_16480_BYTES_SO_A_DEFAULT_LIMIT_BROWSER_MUST_KEEP_IT_OFF: u32 = 16_480;

    #[derive(Clone, Copy, Debug)]
    pub struct OcrWebOptions {
        pub kv_span: usize,
    }

    #[derive(Clone, Debug)]
    pub struct AdapterBytes {
        pub config_json: String,
        pub safetensors: Vec<u8>,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct AdapterReport {
        pub matched: usize,
        pub rank: usize,
    }

    impl Default for OcrWebOptions {
        fn default() -> Self {
            Self {
                kv_span:
                    BROWSER_KV_SPAN_DEFAULT_4096_KEEPS_PER_LAYER_KV_UNDER_5_MB_AND_CAPS_LOAD_TIME,
            }
        }
    }

    #[derive(Clone, Debug)]
    pub struct OcrPageReport {
        pub regions: usize,
        pub prompt_tokens: usize,
        pub generated_tokens: usize,
        pub looped: bool,
        pub hit_eos: bool,
    }

    pub struct OcrWebSession {
        pipe: DotsOcrPipeline,
        kv_span: usize,
        adapter: Option<AdapterReport>,
    }

    #[cfg(target_arch = "wasm32")]
    async fn install_browser_wgpu_context() -> Result<()> {
        use nv_kernels::wgpu_backend::WgpuContext;
        let ctx = WgpuContext::new_browser()
            .await
            .map_err(|e| anyhow::anyhow!("browser wgpu context: {e}"))?;
        if ctx.caps.max_compute_workgroup_storage_size
            < ATTN_REG_STAGES_16480_BYTES_SO_A_DEFAULT_LIMIT_BROWSER_MUST_KEEP_IT_OFF
            && std::env::var_os(nv_models::dots_ocr::decoder_wgpu_graph::ATTN_REG_ENV).is_none()
        {
            std::env::set_var(nv_models::dots_ocr::decoder_wgpu_graph::ATTN_REG_ENV, "0");
        }
        WgpuContext::install_shared(ctx)
            .map_err(|e| anyhow::anyhow!("install browser wgpu context: {e}"))?;
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn ensure_browser_gpu() -> Result<&'static nv_kernels::wgpu_backend::WgpuContext> {
        use std::sync::atomic::{AtomicBool, Ordering};
        static INSTALLED: AtomicBool = AtomicBool::new(false);
        if !INSTALLED.swap(true, Ordering::SeqCst) {
            install_browser_wgpu_context().await?;
        }
        nv_kernels::wgpu_backend::WgpuContext::shared()
            .map_err(|e| anyhow::anyhow!("shared browser wgpu context: {e}"))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn ensure_browser_gpu() -> Result<&'static nv_kernels::wgpu_backend::WgpuContext> {
        nv_kernels::wgpu_backend::WgpuContext::shared()
            .map_err(|e| anyhow::anyhow!("shared wgpu context: {e}"))
    }

    pub async fn init_from_weights(source: impl WeightSource) -> Result<OcrWebSession> {
        init_with_options(source, OcrWebOptions::default()).await
    }

    pub async fn init_with_options(
        source: impl WeightSource,
        opts: OcrWebOptions,
    ) -> Result<OcrWebSession> {
        init_with_adapter(source, opts, None).await
    }

    pub async fn init_with_adapter(
        mut source: impl WeightSource,
        opts: OcrWebOptions,
        adapter: Option<AdapterBytes>,
    ) -> Result<OcrWebSession> {
        #[cfg(target_arch = "wasm32")]
        ensure_browser_gpu().await?;
        let mut config_json: Option<Vec<u8>> = None;
        let mut tokenizer_json: Option<Vec<u8>> = None;
        let mut shards: Vec<(String, Vec<u8>)> = Vec::new();
        for name in source.shard_names() {
            if name == CONFIG_JSON_NAME {
                config_json = Some(source.take_shard(&name)?);
            } else if name == TOKENIZER_JSON_NAME {
                tokenizer_json = Some(source.take_shard(&name)?);
            } else if name.ends_with(".safetensors") {
                let bytes = source.take_shard(&name)?;
                shards.push((name, bytes));
            }
        }
        let config_json =
            String::from_utf8(config_json.context("weight source carries no config.json")?)
                .context("config.json is not utf-8")?;
        let tokenizer_json = tokenizer_json.context("weight source carries no tokenizer.json")?;
        anyhow::ensure!(
            !shards.is_empty(),
            "weight source carries no .safetensors shards"
        );
        let loader = WeightLoader::open_from_named_bytes(shards, &Device::Cpu)?;
        let (lora, adapter_report) = match adapter {
            None => (None, None),
            Some(a) => {
                let cfg = nv_models::dots_ocr::DotsDecoderConfig::from_hf_json_str(&config_json)
                    .context("parse dots.ocr config.json for adapter dims")?;
                let peft = nv_weights::lora_adapter::LoraAdapter::from_named_bytes(
                    &a.config_json,
                    a.safetensors,
                    &Device::Cpu,
                )
                .context("parse adapter bytes")?;
                let lora = nv_models::dots_ocr::decoder_wgpu_graph::DgLora::from_adapter(
                    &peft,
                    &cfg,
                    "adapter bytes",
                )?;
                let report = AdapterReport {
                    matched: lora.matched(),
                    rank: lora.rank(),
                };
                (Some(lora), Some(report))
            }
        };
        let _ = MLX_CONFIG_QUANTIZATION_BLOCK_IS_THE_EXPLICIT_OPT_IN_TO_THE_PREQUANTIZED_GRAPH;
        let quant = nv_weights::QuantizationConfig::from_hf_json_str(&config_json)
            .context("read the config.json quantization block")?;
        let pipe = if quant.scheme == nv_weights::QuantScheme::MlxAffineW4 {
            anyhow::ensure!(
                quant.effective_group_size() == Some(64),
                "mlx affine-u4 checkpoint declares group_size {:?}, the w4 graph reads gs 64",
                quant.effective_group_size()
            );
            DotsOcrPipeline::from_parts_wgpu_graph_w4_mlx_lora(
                &loader,
                &config_json,
                &tokenizer_json,
                opts.kv_span,
                lora,
            )?
        } else {
            DotsOcrPipeline::from_parts_wgpu_graph_w4_lora(
                &loader,
                &config_json,
                &tokenizer_json,
                opts.kv_span,
                lora,
            )?
        };
        Ok(OcrWebSession {
            pipe,
            kv_span: opts.kv_span,
            adapter: adapter_report,
        })
    }

    impl OcrWebSession {
        pub fn kv_span(&self) -> usize {
            self.kv_span
        }

        pub fn adapter(&self) -> Option<AdapterReport> {
            self.adapter
        }

        pub fn pipeline(&self) -> &DotsOcrPipeline {
            &self.pipe
        }

        pub async fn ocr_page(
            &mut self,
            image_bytes: &[u8],
            on_region: &mut dyn FnMut(LayoutElement),
        ) -> Result<OcrPageReport> {
            let img = RgbImage::decode(image_bytes).context("decode page image")?;
            let opts = GenerateOptions::default();
            let res = self.pipe.recognize(&img, DotsMode::LayoutAll, &opts)?;
            Ok(Self::stream_page(res, on_region))
        }

        pub async fn ocr_page_pumped(
            &mut self,
            image_bytes: &[u8],
            on_region: &mut dyn FnMut(LayoutElement),
        ) -> Result<OcrPageReport> {
            let img = RgbImage::decode(image_bytes).context("decode page image")?;
            let opts = GenerateOptions::default();
            let res = self
                .pipe
                .recognize_pumped(&img, DotsMode::LayoutAll, &opts)
                .await?;
            Ok(Self::stream_page(res, on_region))
        }

        fn stream_page(
            res: nv_models::dots_ocr::DotsPageResult,
            on_region: &mut dyn FnMut(LayoutElement),
        ) -> OcrPageReport {
            let report = OcrPageReport {
                regions: res.page.elements.len(),
                prompt_tokens: res.prompt_tokens,
                generated_tokens: res.generated_tokens,
                looped: res.looped,
                hit_eos: res.hit_eos,
            };
            for e in res.page.elements {
                on_region(e);
            }
            report
        }

        pub async fn build_train_sample_pumped(
            &self,
            spec: &PageSample,
            mode: DotsMode,
        ) -> Result<crate::train::TrainSample> {
            let _ = SAMPLE_ASSEMBLY_MIRRORS_NVK_DOTS_VISION_DUMP_WITH_RESIDENT_BF16_ROWS;
            let img = RgbImage::decode(&spec.image).context("decode sample image")?;
            let img = match spec.bbox {
                Some([x1, y1, x2, y2]) => {
                    let m = BBOX_MARGIN_PX_MATCHES_NVK_DOTS_VISION_DUMP as f64;
                    let (w, h) = (img.w as f64, img.h as f64);
                    let cx1 = (x1 - m).max(0.0).min(w - 1.0) as usize;
                    let cy1 = (y1 - m).max(0.0).min(h - 1.0) as usize;
                    let cx2 = (x2 + m).max(0.0).min(w) as usize;
                    let cy2 = (y2 + m).max(0.0).min(h) as usize;
                    anyhow::ensure!(
                        cx2 > cx1 && cy2 > cy1,
                        "degenerate sample bbox {:?}",
                        spec.bbox
                    );
                    img.crop(cx1, cy1, cx2, cy2)
                }
                None => img,
            };
            let prep = self.pipe.prepare(&img)?;
            let gv = self.pipe.encode_resident_rows(&prep)?.ok_or_else(|| {
                anyhow::anyhow!(
                    "the tower produced no resident vision rows; browser sample building rides \
                     the resident splice and its pumped readback alone"
                )
            })?;
            let n_vis = prep.num_vision_tokens();
            anyhow::ensure!(
                gv.rows == n_vis,
                "resident vision produced {} rows, expected {n_vis}",
                gv.rows
            );
            let ctx = ensure_browser_gpu().await?;
            let rows =
                nv_models::dots_ocr::decoder_wgpu_graph::vision_rows_host_pumped(ctx, &gv).await?;
            let encode = |s: &str| self.pipe.encode_text(s);
            let (mut ids, vision_start) = nv_models::dots_ocr::build_prompt_tokens(
                encode,
                mode.prompt(),
                n_vis,
                self.pipe.prompt_style(),
            )?;
            let completion_start = ids.len();
            ids.extend(self.pipe.encode_text(&spec.text)?);
            ids.push(nv_models::dots_ocr::ENDOFASSISTANT_TOKEN_ID);
            Ok(crate::train::TrainSample {
                ids,
                completion_start,
                vision_start,
                n_vis,
                rows,
            })
        }
    }

    pub const BBOX_MARGIN_PX_MATCHES_NVK_DOTS_VISION_DUMP: u32 = 8;

    pub const SAMPLE_ASSEMBLY_MIRRORS_NVK_DOTS_VISION_DUMP_WITH_RESIDENT_BF16_ROWS: &str =
        "ids are build_prompt_tokens(mode prompt, n_vis, style) ++ encode(target text) ++ \
         ENDOFASSISTANT, completion_start at the prompt boundary, exactly the \
         nvk-dots-vision-dump example; the vision rows come from the resident splice read \
         back through the pump, so they carry the handoff's bf16 rounding where the native \
         dump stores the host merger's f32 -- same values the serving graph consumes, one \
         rounding class from the dump";

    #[derive(Clone, Debug)]
    pub struct PageSample {
        pub image: Vec<u8>,
        pub text: String,
        pub bbox: Option<[f64; 4]>,
    }
}

#[cfg(feature = "browser")]
pub use browser::{
    ensure_browser_gpu, init_from_weights, init_with_adapter, init_with_options, AdapterBytes,
    AdapterReport, OcrPageReport, OcrWebOptions, OcrWebSession, PageSample,
    ATTN_REG_STAGES_16480_BYTES_SO_A_DEFAULT_LIMIT_BROWSER_MUST_KEEP_IT_OFF,
    BBOX_MARGIN_PX_MATCHES_NVK_DOTS_VISION_DUMP,
    BROWSER_KV_SPAN_DEFAULT_4096_KEEPS_PER_LAYER_KV_UNDER_5_MB_AND_CAPS_LOAD_TIME,
    CONFIG_JSON_NAME, TOKENIZER_JSON_NAME,
};
