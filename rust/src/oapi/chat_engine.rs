use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(any(feature = "cuda", test))]
use rand_core::Rng;
#[cfg(any(feature = "cuda", test))]
use rand_core::SeedableRng;
#[cfg(any(feature = "cuda", test))]
use rand_pcg::Pcg64;
use tokio::sync::mpsc;
#[cfg(feature = "cuda")]
use tracing::warn;

#[cfg(feature = "cuda")]
use crate::oapi::chat::{render_chat_prompt, ChatMessageIn};
use crate::oapi::chat::{ChatEngine, ChatEvent, ChatGenerateRequest};

#[cfg(feature = "cuda")]
pub(crate) use candle_core::IndexOp;

mod batch;
mod build;
#[cfg(feature = "cuda")]
mod capture;
mod gemma4_loop;
mod gemma4_moe_loop;
mod gpt_oss_loop;
mod laguna_loop;
#[cfg(feature = "cuda")]
mod lora_cuda;
#[cfg(feature = "cuda")]
mod omni_loop;
mod qwen;
mod qwen4_exp_loop;
#[cfg(feature = "cuda")]
mod qwen4_exp_mm;
mod qwen38_sched;
#[cfg(any(feature = "cuda", test))]
mod route;
mod sampling;
#[cfg(any(feature = "cuda", test))]
mod spec_env;
#[cfg(any(feature = "cuda", test, kani))]
mod spec_window;
#[cfg(any(feature = "cuda", test))]
mod stream;

#[cfg(feature = "cuda")]
pub(crate) use build::*;
pub use build::{
    allow_unknown_model, allow_unknown_model_from, default_engine, default_engine_from_env,
    live_registry, publish_live_registry, registry_from_env, ChatRegistry, NvEngineChat,
    RemoveDenied, ALLOW_UNKNOWN_MODEL_ENV,
};
#[cfg(feature = "cuda")]
pub(crate) use gemma4_loop::*;
#[cfg(feature = "cuda")]
pub(crate) use gemma4_moe_loop::*;
#[cfg(feature = "cuda")]
pub(crate) use gpt_oss_loop::*;
#[cfg(feature = "cuda")]
pub(crate) use laguna_loop::*;
#[cfg(feature = "cuda")]
pub use lora_cuda::{
    cuda_adapter_tag_that_changes_when_the_weights_change, cuda_base_dirs, select_cuda_base,
    CudaLoraAdapterEngine, Qwen4ExpAdapterEngine, QWEN4_EXP_LORA_ADAPTER_DIRS_ENV,
    hot_swap_qwen4_exp_adapter, qwen4_exp_adapter_engine, qwen4_exp_adapter_metrics, qwen4_exp_adapter_registry, unload_qwen4_exp_adapter,
};
#[cfg(feature = "cuda")]
pub(crate) use omni_loop::*;
#[cfg(feature = "cuda")]
pub(crate) use qwen::*;
#[cfg(feature = "cuda")]
pub(crate) use qwen38_sched::*;

#[cfg_attr(not(any(test, feature = "cuda")), allow(unused_imports))]
pub(crate) use batch::*;
#[cfg(any(feature = "cuda", test))]
pub(crate) use route::*;
pub(crate) use sampling::*;
#[cfg(any(feature = "cuda", test))]
pub(crate) use spec_env::*;
#[cfg(any(feature = "cuda", test, kani))]
pub(crate) use spec_window::*;
#[cfg(any(feature = "cuda", test))]
pub(crate) use stream::*;

pub struct EchoEngine {
    pub model_id: String,
    pub reply: String,
}

impl EchoEngine {
    pub fn new(model_id: impl Into<String>, reply: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            reply: reply.into(),
        }
    }
}

#[async_trait::async_trait]
impl ChatEngine for EchoEngine {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn generate(
        &self,
        req: ChatGenerateRequest,
        tx: mpsc::Sender<ChatEvent>,
    ) -> anyhow::Result<()> {
        let full = self.reply.clone();
        let (reply, matched) = req
            .stop
            .iter()
            .filter(|s| !s.is_empty())
            .filter_map(|s| full.find(s.as_str()).map(|at| (at, s.clone())))
            .min_by_key(|(at, _)| *at)
            .map(|(at, s)| (full[..at].to_string(), Some(s)))
            .unwrap_or((full, None));

        let prompt_tokens = req.prompt.split_whitespace().count() as u32;
        let max_new = req.max_new_tokens;
        tokio::spawn(async move {
            let _ = tx.send(ChatEvent::Started { prompt_tokens }).await;

            let words: Vec<&str> = reply.split_whitespace().collect();
            let mut emitted = 0u32;
            for (i, w) in words.iter().enumerate() {
                if (emitted as usize) >= max_new {
                    break;
                }
                let piece = if i == 0 {
                    (*w).to_string()
                } else {
                    format!(" {w}")
                };
                if tx.send(ChatEvent::TextDelta(piece)).await.is_err() {
                    return;
                }
                emitted += 1;
            }
            if let Some(stop_sequence) = matched {
                let _ = tx.send(ChatEvent::StoppedBy { stop_sequence }).await;
            }
            let _ = tx
                .send(ChatEvent::Done {
                    finish_reason: "stop".into(),
                    completion_tokens: emitted,
                })
                .await;
        });
        Ok(())
    }
}

#[cfg(feature = "cuda")]
#[async_trait::async_trait]
impl ChatEngine for NvEngineChat {
    fn capture_info(&self) -> crate::oapi::capture::CaptureInfo {
        capture::capture_info(self)
    }

    async fn capture_layer(
        &self,
        req: crate::oapi::capture::LayerCaptureRequest,
    ) -> anyhow::Result<crate::oapi::capture::LayerCaptureResponse> {
        capture::capture_layer(self, req).await
    }

    fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    fn spec_decode_status(&self) -> Option<&'static str> {
        self.inner.spec_status
    }

    fn supports_mm_input(&self) -> bool {
        matches!(self.inner.family, ModelFamily::Omni)
            || (matches!(
                self.inner.family,
                ModelFamily::Gemma4 | ModelFamily::Gemma4E4b | ModelFamily::Gemma4Moe
            ) && self.inner.mm_towers.is_some())
            || matches!(&self.inner.model, LoadedModel::Qwen4Exp(m) if qwen4_exp_mm::vision_for(m).is_some())
    }

    fn supports_audio_input(&self) -> bool {
        !matches!(self.inner.family, ModelFamily::Qwen4Exp)
    }

    fn mm_markers(&self) -> (&'static str, &'static str) {
        match self.inner.family {
            ModelFamily::Omni => (OMNI_IMAGE_MARKER, OMNI_AUDIO_MARKER),
            _ => (
                crate::oapi::chat::GEMMA4_IMAGE_MARKER,
                crate::oapi::chat::GEMMA4_AUDIO_MARKER,
            ),
        }
    }

    fn render_prompt(&self, messages: &[ChatMessageIn]) -> String {
        match self.inner.family {
            ModelFamily::Qwen3 => render_chat_prompt(messages),
            ModelFamily::Gemma4 | ModelFamily::Gemma4E4b | ModelFamily::Gemma4Moe => {
                render_gemma4_prompt(messages)
            }
            ModelFamily::Qwen3_5Moe | ModelFamily::Qwen4Exp => render_qwen3_5_moe_prompt(messages),
            ModelFamily::Laguna => render_laguna_prompt(messages),
            ModelFamily::Omni | ModelFamily::GptOss => render_chat_prompt(messages),
        }
    }

    fn official_template(&self) -> Option<&crate::oapi::chat_template::ChatTemplate> {
        self.inner.chat_template.as_deref()
    }

    fn runtime_lora_twin_sharing_these_resident_weights(
        &self,
        adapter_id: &str,
        adapter_dir: &Path,
    ) -> Option<anyhow::Result<Arc<dyn ChatEngine>>> {
        lora_cuda::qwen4_exp_runtime_lora_engine(&self.inner, adapter_id, adapter_dir)
    }

    async fn generate(
        &self,
        req: ChatGenerateRequest,
        tx: mpsc::Sender<ChatEvent>,
    ) -> anyhow::Result<()> {
        let tokenizer = self.inner.tokenizer.clone();
        let device = self.inner.device.clone();
        let kv_max_seq_len = self.inner.kv_max_seq_len;
        let default_max_new = self.inner.default_max_new_tokens as usize;
        let eos_ids = self.inner.eos_token_ids.clone();
        let bos_token_id = self.inner.bos_token_id;
        let family = self.inner.family;
        let eagle3 = self.inner.eagle3.clone();
        let dflash = self.inner.dflash.clone();
        let laguna_spec = self.inner.laguna_spec.clone();
        let qwen_moe_dispatch = self.inner.qwen_moe_dispatch.clone();
        let qwen_mtp = self.inner.qwen_mtp.clone();
        let mm_towers = self.inner.mm_towers.clone();
        let loaded = match &self.inner.model {
            LoadedModel::Qwen3(m) => LoadedModel::Qwen3(m.clone()),
            LoadedModel::Gemma4(m) => LoadedModel::Gemma4(m.clone()),
            LoadedModel::Gemma4E4b(m) => LoadedModel::Gemma4E4b(m.clone()),
            LoadedModel::Gemma4Moe(m) => LoadedModel::Gemma4Moe(m.clone()),
            LoadedModel::Qwen3_5Moe(m) => LoadedModel::Qwen3_5Moe(m.clone()),
            LoadedModel::Laguna(m) => LoadedModel::Laguna(m.clone()),
            LoadedModel::Omni(m) => LoadedModel::Omni(m.clone()),
            LoadedModel::GptOss(m) => LoadedModel::GptOss(m.clone()),
            LoadedModel::Qwen4Exp(m) => LoadedModel::Qwen4Exp(m.clone()),
        };

        if self.inner.spec_status == Some("degraded") {
            log_spec_degraded_rate_limited();
        }
        let snap = SpecEnvSnapshot::capture();
        let mm_present = req
            .mm
            .as_ref()
            .is_some_and(|m| !(m.images.is_empty() && m.audios.is_empty()));
        let spec_capable = matches!(family, ModelFamily::Gemma4)
            && !mm_present
            && (eagle3.is_some() || dflash.is_some())
            && spec_gate_for_request(
                nv_no_spec(snap.no_spec.as_deref()),
                env_flag_enabled(snap.use_eagle3.as_deref()),
                sampling_params_from(&req).is_greedy(),
            )
            && req.guided.is_none()
            && req.logit_bias.is_empty();
        let spec_eligible = spec_capable && !req.logprobs;
        if spec_capable && req.logprobs {
            tracing::warn!(
                "logprobs requested: speculative decoding disabled for this request; \
                 falling back to the non-spec path (expect ~25-40% lower throughput)"
            );
        }
        let engine_eligible = matches!(family, ModelFamily::Gemma4)
            && !mm_present
            && self.inner.gemma4_engine.is_some()
            && req.guided.is_none()
            && req.logit_bias.is_empty()
            && !req.logprobs
            && !spec_eligible;
        if engine_eligible {
            let handle = self.inner.gemma4_engine.clone().unwrap();
            let tokenizer = tokenizer.clone();
            let eos_ids = eos_ids.clone();
            tokio::spawn(async move {
                if let Err(err) = run_gemma4_via_engine(
                    handle,
                    tokenizer,
                    req,
                    kv_max_seq_len,
                    default_max_new,
                    &eos_ids,
                    bos_token_id,
                    &tx,
                )
                .await
                {
                    let _ = tx.send(ChatEvent::Error(format!("{err:#}"))).await;
                }
            });
            return Ok(());
        }

        tokio::spawn(async move {
            let res = match (family, loaded) {
                (ModelFamily::Qwen3, LoadedModel::Qwen3(m)) => {
                    run_sampling_qwen3(
                        m,
                        tokenizer,
                        device,
                        req,
                        kv_max_seq_len,
                        default_max_new,
                        &eos_ids,
                        &tx,
                    )
                    .await
                }
                (ModelFamily::Gemma4, LoadedModel::Gemma4(m)) => {
                    run_sampling_gemma4(
                        m,
                        tokenizer,
                        device,
                        req,
                        kv_max_seq_len,
                        default_max_new,
                        &eos_ids,
                        bos_token_id,
                        eagle3,
                        dflash,
                        mm_towers,
                        &snap,
                        &tx,
                    )
                    .await
                }
                (ModelFamily::Gemma4E4b, LoadedModel::Gemma4E4b(m)) => {
                    run_sampling_gemma4_e4b(
                        m,
                        tokenizer,
                        device,
                        req,
                        kv_max_seq_len,
                        default_max_new,
                        &eos_ids,
                        bos_token_id,
                        mm_towers,
                        &tx,
                    )
                    .await
                }
                (ModelFamily::Gemma4Moe, LoadedModel::Gemma4Moe(m)) => {
                    run_sampling_gemma4_moe(
                        m,
                        tokenizer,
                        device,
                        req,
                        kv_max_seq_len,
                        default_max_new,
                        &eos_ids,
                        bos_token_id,
                        mm_towers,
                        &tx,
                    )
                    .await
                }
                (ModelFamily::Qwen3_5Moe, LoadedModel::Qwen3_5Moe(m)) => {
                    run_sampling_qwen3_5_moe(
                        m,
                        tokenizer,
                        device,
                        req,
                        kv_max_seq_len,
                        default_max_new,
                        &eos_ids,
                        qwen_moe_dispatch,
                        qwen_mtp,
                        &tx,
                    )
                    .await
                }
                (ModelFamily::Omni, LoadedModel::Omni(m)) => {
                    run_sampling_omni(
                        m,
                        tokenizer,
                        device,
                        req,
                        kv_max_seq_len,
                        default_max_new,
                        &eos_ids,
                        &tx,
                    )
                    .await
                }
                (ModelFamily::Laguna, LoadedModel::Laguna(m)) => {
                    run_sampling_laguna(
                        m,
                        tokenizer,
                        device,
                        req,
                        kv_max_seq_len,
                        default_max_new,
                        &eos_ids,
                        laguna_spec,
                        &tx,
                    )
                    .await
                }
                (ModelFamily::Qwen4Exp, LoadedModel::Qwen4Exp(m)) => {
                    qwen4_exp_loop::run_sampling_qwen4_exp(
                        m,
                        tokenizer,
                        req,
                        kv_max_seq_len,
                        default_max_new,
                        &eos_ids,
                        None,
                        &tx,
                    )
                    .await
                }
                (ModelFamily::GptOss, LoadedModel::GptOss(m)) => {
                    run_sampling_gpt_oss(
                        m,
                        tokenizer,
                        req,
                        kv_max_seq_len,
                        default_max_new,
                        &eos_ids,
                        &tx,
                    )
                    .await
                }
                _ => Err(anyhow::anyhow!("engine family/model variant mismatch")),
            };
            if let Err(err) = res {
                let _ = tx.send(ChatEvent::Error(format!("{err:#}"))).await;
            }
        });
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
#[async_trait::async_trait]
impl ChatEngine for NvEngineChat {
    fn model_id(&self) -> &str {
        "nv-engine/cuda-only"
    }

    async fn generate(
        &self,
        _req: ChatGenerateRequest,
        tx: mpsc::Sender<ChatEvent>,
    ) -> anyhow::Result<()> {
        tokio::spawn(async move {
            let _ = tx
                .send(ChatEvent::Error(
                    "NvEngineChat requires the `cuda` cargo feature".into(),
                ))
                .await;
        });
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn render_gemma4_prompt(messages: &[ChatMessageIn]) -> String {
    let mut out = String::new();
    out.push_str("<bos>");
    let mut idx = 0usize;
    if !messages.is_empty() && (messages[0].role == "system" || messages[0].role == "developer") {
        out.push_str("<|turn>system\n");
        out.push_str(messages[0].text().trim());
        out.push_str("<turn|>\n");
        idx = 1;
    }
    for m in &messages[idx..] {
        let role = if m.role == "assistant" {
            "model"
        } else {
            m.role.as_str()
        };
        out.push_str("<|turn>");
        out.push_str(role);
        out.push('\n');
        out.push_str(m.text().trim());
        out.push_str("<turn|>\n");
    }
    out.push_str("<|turn>model\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn delta(s: &str) -> ChatEvent {
        ChatEvent::TextDelta(s.to_string())
    }

    #[test]
    fn push_blocking_detects_a_closed_receiver_and_times_out_on_a_stalled_reader() {
        let (tx, rx) = mpsc::channel::<ChatEvent>(2);
        drop(rx);
        let t0 = std::time::Instant::now();
        let out = push_event_blocking(&tx, delta("a"), std::time::Duration::from_secs(30));
        assert_eq!(out, SsePush::Closed);
        assert!(t0.elapsed() < std::time::Duration::from_secs(1));
        {
            let (tx, _rx) = mpsc::channel::<ChatEvent>(1);
            assert_eq!(
                push_event_blocking(&tx, delta("fill"), std::time::Duration::from_millis(10)),
                SsePush::Sent
            );
            let t0 = std::time::Instant::now();
            let out = push_event_blocking(&tx, delta("b"), std::time::Duration::from_millis(60));
            let dt = t0.elapsed();
            assert_eq!(out, SsePush::TimedOut);
            assert!(
                dt >= std::time::Duration::from_millis(55),
                "returned early: {dt:?}"
            );
            assert!(dt < std::time::Duration::from_secs(5), "overslept: {dt:?}");
        }
        {
            let (tx, rx) = mpsc::channel::<ChatEvent>(1);
            assert_eq!(
                push_event_blocking(&tx, delta("fill"), std::time::Duration::from_millis(10)),
                SsePush::Sent
            );
            let dropper = std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(30));
                drop(rx);
            });
            let out = push_event_blocking(&tx, delta("b"), std::time::Duration::from_secs(30));
            assert_eq!(out, SsePush::Closed);
            dropper.join().unwrap();
        }
    }

    #[tokio::test]
    async fn push_async_times_out_and_detects_close() {
        let (tx, rx) = mpsc::channel::<ChatEvent>(1);
        assert_eq!(
            push_event_async(&tx, delta("fill"), std::time::Duration::from_millis(10)).await,
            SsePush::Sent
        );
        assert_eq!(
            push_event_async(&tx, delta("b"), std::time::Duration::from_millis(40)).await,
            SsePush::TimedOut
        );
        drop(rx);
        assert_eq!(
            push_event_async(&tx, delta("c"), std::time::Duration::from_millis(40)).await,
            SsePush::Closed
        );
    }

    #[test]
    fn stream_delta_cases() {
        for (sent, full, want) in [
            ("", "he", "he"),
            ("he", "hello", "llo"),
            ("hello", "hello", ""),
            ("hello", "hell", ""),
            ("hello", "", ""),
            ("abcdef", "abXY", "XY"),
            ("abc", "xyz", "xyz"),
            ("x\u{fffd}", "x\u{1f600}", "\u{1f600}"),
            ("caf", "caf\u{e9}", "\u{e9}"),
            ("caf\u{e9}", "caf\u{e9}s", "s"),
            ("\u{20ac}\u{20ac}", "\u{20ac}", ""),
        ] {
            assert_eq!(stream_text_delta(sent, full), want, "sent={sent:?} full={full:?}");
        }
    }

    fn run_emitter(stop: &[&str], steps: &[&str]) -> (String, bool, String) {
        let stop: Vec<String> = stop.iter().map(|s| s.to_string()).collect();
        let mut em = StreamEmitter::new(&stop);
        let mut out = String::new();
        let mut hit = false;
        for s in steps {
            let (piece, h) = em.step(s);
            out.push_str(&piece);
            if h {
                hit = true;
                break;
            }
        }
        if !hit {
            let last = steps.last().copied().unwrap_or("");
            out.push_str(&em.finish(last));
        }
        let sent = em.sent.clone();
        (out, hit, sent)
    }

    #[test]
    fn emitter_streams_plain_appends_and_holds_back_a_trailing_fffd_until_the_char_completes() {
        let steps = [
            "\u{fffd}",
            "\u{fffd}\u{fffd}",
            "\u{fffd}\u{fffd}\u{fffd}",
            "\u{13000}",
        ];
        let (out, hit, _) = run_emitter(&[], &steps);
        assert_eq!(out, "\u{13000}");
        assert!(!hit);

        let steps = [
            "\u{fffd}",
            "\u{13000}",
            "\u{13000}\u{fffd}",
            "\u{13000}\u{fffd}\u{fffd}",
            "\u{13000}\u{13001}",
        ];
        let (out, _, _) = run_emitter(&[], &steps);
        assert_eq!(out, "\u{13000}\u{13001}");

        let steps = ["ok \u{fffd}", "ok \u{fffd}\u{fffd}"];
        let (out, hit, _) = run_emitter(&[], &steps);
        assert!(!hit);
        assert_eq!(out, "ok \u{fffd}\u{fffd}");
        {
            let (out, hit, _) = run_emitter(&[], &["he", "hello", "hello world"]);
            assert_eq!(out, "hello world");
            assert!(!hit);
        }
    }

    #[test]
    fn emitter_stop_cases() {
        let cases: &[(&[&str], &[&str], &str)] = &[
            (&["three"], &["one ", "one two ", "one two three"], "one two "),
            (&["Hell"], &["Hello"], ""),
            (&["ab"], &["x", "xaby"], "x"),
            (&["\u{3002}"], &["猫", "猫が好き", "猫が好き\u{3002}犬も"], "猫が好き"),
            (&["two", "one"], &["zero one two"], "zero "),
        ];
        for (stop, steps, want) in cases {
            let (out, hit, _) = run_emitter(stop, steps);
            assert_eq!(out, *want, "stop={stop:?} steps={steps:?}");
            assert!(hit, "stop={stop:?} steps={steps:?}");
        }
        {
            let (out, hit, _) = run_emitter(&["END"], &["EN", "ENOUGH done"]);
            assert_eq!(out, "ENOUGH done");
            assert!(!hit);

            let (out, hit, _) = run_emitter(&["END"], &["EN", "END"]);
            assert_eq!(out, "");
            assert!(hit);

            let (out, hit, _) = run_emitter(&["END"], &["say E", "say EN", "say END now"]);
            assert_eq!(out, "say ");
            assert!(hit);
        }
        {
            let stop = vec!["X".to_string()];
            let mut em = StreamEmitter::new(&stop);
            let (p1, h1) = em.step("aX");
            assert_eq!((p1.as_str(), h1), ("a", true));
            let (p2, h2) = em.step("aXbc");
            assert_eq!((p2.as_str(), h2), ("", true));
            assert_eq!(em.finish("aXbc"), "");
        }
        {
            let (out, hit, _) = run_emitter(&["STOP"], &["a\u{fffd}", "aSTOP\u{fffd}"]);
            assert_eq!(out, "a");
            assert!(hit);
        }
    }

    fn bytefallback_tokenizer() -> Arc<tokenizers::Tokenizer> {
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 20, "content": "<eos>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": null,
            "decoder": {"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"}
            ]},
            "model": {"type": "WordLevel", "vocab": {
                "<unk>": 0, "▁hello": 1, "▁world": 2, "!": 3,
                "<0xE2>": 4, "<0x82>": 5, "<0xAC>": 6, "<0xFF>": 7,
                "▁café": 8, "猫": 9, "▁": 10,
                "<0xF0>": 11, "<0x9F>": 12, "<0x98>": 13, "<0x80>": 14,
                "<0xE1>": 15, "<0x9A>": 16, "<0xA0>": 17, "<0x9B>": 18
            }, "unk_token": "<unk>"}
        }"#;
        Arc::new(
            json.parse::<tokenizers::Tokenizer>()
                .expect("test tokenizer"),
        )
    }

    fn assert_settled_text_is_a_prefix_released_when_no_byte_run_is_open(
        tok: &tokenizers::Tokenizer,
        detok: &IncrementalDetok,
        settled: &str,
        full: &str,
        id: u32,
    ) {
        assert!(
            full.starts_with(settled),
            "push returned {settled:?}, not a prefix of the exact decode {full:?}"
        );
        if !tok.id_to_token(id).is_some_and(|t| t.starts_with("<0x")) {
            assert_eq!(
                settled, full,
                "a non-byte token closes every byte run, so push must release the whole decode"
            );
        }
        assert_eq!(detok.acc, full, "the detok window drifted from the full decode");
    }

    fn assert_incremental_matches_full(tok: &Arc<tokenizers::Tokenizer>, ids: &[u32]) {
        let mut detok = IncrementalDetok::new(tok.clone());
        for i in 0..ids.len() {
            let settled = detok.push(ids[i]).expect("push").to_string();
            let full = tok.decode(&ids[..=i], true).expect("full decode");
            assert_settled_text_is_a_prefix_released_when_no_byte_run_is_open(
                tok, &detok, &settled, &full, ids[i],
            );
        }
    }

    #[test]
    fn incremental_detok_matches_full_decode_stays_bounded_and_streams_byte_runs_once() {
        let tok = bytefallback_tokenizer();

        assert_incremental_matches_full(&tok, &[1, 2, 3]);

        assert_incremental_matches_full(&tok, &[1, 4, 5, 6, 3]);

        assert_incremental_matches_full(&tok, &[1, 4, 5, 3, 2]);

        assert_incremental_matches_full(&tok, &[1, 4, 7, 5, 6, 3]);

        assert_incremental_matches_full(&tok, &[8, 9, 11, 12, 13, 14, 9, 3]);

        assert_incremental_matches_full(&tok, &[1, 20, 2, 20, 20, 3]);

        assert_incremental_matches_full(&tok, &[7, 7, 7, 4, 5, 6, 7]);
        {
            let tok = bytefallback_tokenizer();
            let cycle = [1u32, 2, 3, 8, 9, 10, 4, 5, 6, 11, 12, 13, 14];
            let ids: Vec<u32> = (0..400).map(|i| cycle[i % cycle.len()]).collect();
            let mut detok = IncrementalDetok::new(tok.clone());
            let mut max_window = 0usize;
            for i in 0..ids.len() {
                let settled = detok.push(ids[i]).expect("push").to_string();
                max_window = max_window.max(detok.window.len());
                let full = tok.decode(&ids[..=i], true).expect("full decode");
                assert_settled_text_is_a_prefix_released_when_no_byte_run_is_open(
                    &tok, &detok, &settled, &full, ids[i],
                );
            }
            assert!(
                max_window < 2 * DETOK_WINDOW_MAX,
                "window grew to {max_window}, expected < {}",
                2 * DETOK_WINDOW_MAX
            );
            assert!(!detok.stable.is_empty(), "cut never advanced");
        }
        {
            let tok = bytefallback_tokenizer();
            let runes = [15u32, 16, 17, 15, 18, 12];
            assert_eq!(tok.decode(&runes, true).unwrap(), "\u{16A0}\u{16DF}");
            for ids in [&runes[..], &[15, 16, 17, 15, 18, 12, 3], &[15, 16, 17, 7, 3]] {
                assert_eq!(
                    stream_through_detok_and_emitter(&tok, ids),
                    tok.decode(ids, true).unwrap(),
                    "ids {ids:?}: a ByteFallback run decodes all-or-nothing, so text handed to the \
                     emitter while the run is open can be retracted, and the client then receives it twice"
                );
            }
        }
    }

    fn stream_through_detok_and_emitter(tok: &Arc<tokenizers::Tokenizer>, ids: &[u32]) -> String {
        let mut detok = IncrementalDetok::new(tok.clone());
        let mut em = StreamEmitter::new(&[]);
        let mut out = String::new();
        for id in ids {
            out.push_str(&em.step(detok.push(*id).expect("push")).0);
        }
        out.push_str(&em.finish(&decode_keeping_wire(tok, ids).expect("decode")));
        out
    }

    #[test]
    fn registry_resolves_by_model_and_default_and_try_load_names_the_first_missing_file() {
        let a: Arc<dyn ChatEngine> = Arc::new(EchoEngine::new("model-a", "x"));
        let b: Arc<dyn ChatEngine> = Arc::new(EchoEngine::new("model-b", "y"));
        let reg = ChatRegistry::from_engines(vec![a, b]).expect("non-empty");
        assert_eq!(
            reg.model_ids(),
            &["model-a".to_string(), "model-b".to_string()]
        );
        assert_eq!(reg.resolve(None).unwrap().model_id(), "model-a");
        assert_eq!(reg.resolve(Some("")).unwrap().model_id(), "model-a");
        assert_eq!(reg.resolve(Some("model-b")).unwrap().model_id(), "model-b");
        assert!(reg.resolve(Some("missing")).is_none());
        assert!(reg.contains("model-a"));
        assert!(!reg.contains("missing"));

        let a: Arc<dyn ChatEngine> = Arc::new(EchoEngine::new("dup", "x"));
        let b: Arc<dyn ChatEngine> = Arc::new(EchoEngine::new("dup", "y"));
        let reg = ChatRegistry::from_engines(vec![a, b]).expect("non-empty");
        assert_eq!(reg.model_ids(), &["dup".to_string()]);
        {
            let tmp = tempdir_like("nveng-missing");
            for (present, missing) in [
                (None, "config.json"),
                (Some("config.json"), "tokenizer.json"),
                (Some("tokenizer.json"), "safetensors"),
            ] {
                if let Some(f) = present {
                    fs::write(tmp.join(f), "{}").unwrap();
                }
                let err = NvEngineChat::try_load(&tmp).err().expect("expected Err");
                let msg = format!("{err}");
                assert!(msg.contains(missing), "got: {msg}");
            }
            cleanup(&tmp);
        }
    }

    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn default_engine_from_env_unset_returns_none() {
        let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = EnvGuard::unset("NV_CHAT_MODEL_DIR");
        assert!(default_engine_from_env().is_none());
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[should_panic(expected = "NvEngineChat::try_load failed")]
    fn default_engine_from_env_bad_dir_panics() {
        let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = EnvGuard::set("NV_CHAT_MODEL_DIR", "/does/not/exist/xyzzy");
        let _ = default_engine_from_env();
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn default_engine_from_env_non_cuda_returns_none_without_panic() {
        let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = EnvGuard::set("NV_CHAT_MODEL_DIR", "/does/not/exist/xyzzy");
        assert!(default_engine_from_env().is_none());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn load_eagle3_state_returns_none_when_env_unset_or_dir_missing() {
        let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let device = candle_core::Device::Cpu;
        {
            let _guard = EnvGuard::unset("NV_EAGLE3_DRAFT_DIR");
            let state = load_eagle3_state(&device, None);
            assert!(state.is_none(), "no env var must mean spec-decode disabled");
        }
        let _guard = EnvGuard::set(
            "NV_EAGLE3_DRAFT_DIR",
            "/definitely/not/a/real/eagle3/dir/xyzzy",
        );
        let state = load_eagle3_state(&device, None);
        assert!(state.is_none(), "missing dir must yield None, not panic");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn detect_family_routes_by_arch_model_type_and_moe_flag() {
        for (cfg, want) in [
            (
                r#"{"architectures":["Gemma4ForConditionalGeneration"],"model_type":"gemma4"}"#,
                ModelFamily::Gemma4,
            ),
            (r#"{"model_type":"gemma4"}"#, ModelFamily::Gemma4),
            (
                r#"{"architectures":["Gemma4AssistantForCausalLM"],"model_type":"gemma4_assistant"}"#,
                ModelFamily::Gemma4,
            ),
            (
                r#"{"architectures":["Gemma4ForConditionalGeneration"],"model_type":"gemma4",
            "text_config":{"enable_moe_block":true,"num_experts":128}}"#,
                ModelFamily::Gemma4Moe,
            ),
            (r#"{"model_type":"gemma4","enable_moe_block":true}"#, ModelFamily::Gemma4Moe),
            (
                r#"{"model_type":"gemma4","text_config":{"enable_moe_block":false}}"#,
                ModelFamily::Gemma4,
            ),
            (
                r#"{"architectures":["Qwen3_5MoeForConditionalGeneration"],"model_type":"qwen3_5_moe"}"#,
                ModelFamily::Qwen3_5Moe,
            ),
            (r#"{"architectures":["Qwen3_5MoeForCausalLM"]}"#, ModelFamily::Qwen3_5Moe),
            (r#"{"model_type":"qwen3_5_moe"}"#, ModelFamily::Qwen3_5Moe),
        ] {
            assert_eq!(detect_family(cfg).unwrap(), want, "{cfg}");
        }
        assert!(detect_family(r#"{"model_type":"llama"}"#).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gemma4_and_qwen_prompts_render_their_turn_markers_and_eos_ids_parse() {
        use crate::oapi::chat::{ChatMessageIn, MessageContent};
        let msgs = vec![
            ChatMessageIn {
                role: "system".into(),
                content: Some(MessageContent::Text("S".into())),
                ..Default::default()
            },
            ChatMessageIn {
                role: "user".into(),
                content: Some(MessageContent::Text("U".into())),
                ..Default::default()
            },
        ];
        let p = render_gemma4_prompt(&msgs);
        assert!(p.starts_with("<bos>"));
        assert!(p.contains("<|turn>system\nS<turn|>\n"));
        assert!(p.contains("<|turn>user\nU<turn|>\n"));
        assert!(p.ends_with("<|turn>model\n"));
        {
            use crate::oapi::chat::{ChatMessageIn, MessageContent};
            let msgs = vec![
                ChatMessageIn {
                    role: "system".into(),
                    content: Some(MessageContent::Text("S".into())),
                    ..Default::default()
                },
                ChatMessageIn {
                    role: "user".into(),
                    content: Some(MessageContent::Text("U".into())),
                    ..Default::default()
                },
            ];
            let p = render_qwen3_5_moe_prompt(&msgs);
            assert!(p.contains("<|im_start|>system\nS<|im_end|>\n"));
            assert!(p.contains("<|im_start|>user\nU<|im_end|>\n"));
            assert!(p.ends_with("<|im_start|>assistant\n<think>\n"));
        }
        {
            assert_eq!(parse_eos_ids(r#"{"eos_token_id":7}"#), Some(vec![7]));
            assert_eq!(
                parse_eos_ids(r#"{"eos_token_id":[1,106]}"#),
                Some(vec![1, 106])
            );
            assert_eq!(parse_eos_ids(r#"{}"#), None);
        }
    }

    fn tempdir_like(prefix: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("speaches-plus-{prefix}-{pid}-{nanos}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn cleanup(p: &Path) {
        let _ = fs::remove_dir_all(p);
    }

    struct EnvGuard {
        key: &'static str,
        prev: Option<std::ffi::OsString>,
    }
    impl EnvGuard {
        fn set(key: &'static str, val: &str) -> Self {
            let prev = std::env::var_os(key);
            unsafe { std::env::set_var(key, val) };
            Self { key, prev }
        }
        fn unset(key: &'static str) -> Self {
            let prev = std::env::var_os(key);
            unsafe { std::env::remove_var(key) };
            Self { key, prev }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => unsafe { std::env::set_var(self.key, v) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    use rand_core::SeedableRng;
    use rand_pcg::Pcg64;

    #[test]
    fn sampler_is_greedy_at_zero_temperature_restricts_to_the_head_and_is_seed_reproducible() {
        let logits = vec![0.1_f32, -1.0, 0.5, 9.9, 0.2];
        for seed in [0u64, 1, 42, 1_000_000] {
            let mut rng = Pcg64::seed_from_u64(seed);
            let tok = sample_logits(&logits, 0.0, None, None, &mut rng);
            assert_eq!(tok, 3, "greedy must pick argmax (seed={seed})");
        }

        let mut rng = Pcg64::seed_from_u64(7);
        let tok = sample_logits(&logits, 1.0, Some(1), None, &mut rng);
        assert_eq!(tok, 3);
        {
            let mut rng = Pcg64::seed_from_u64(99);
            for (logits, top_k, top_p, allowed) in [
                (
                    vec![10.0_f32, -100.0, -100.0, 10.0, -100.0, -100.0],
                    Some(2),
                    None,
                    [0u32, 3],
                ),
                (
                    vec![5.0_f32, 4.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    None,
                    Some(0.5),
                    [0, 1],
                ),
            ] {
                for _ in 0..500 {
                    let tok = sample_logits(&logits, 1.0, top_k, top_p, &mut rng);
                    assert!(
                        allowed.contains(&tok),
                        "top_k={top_k:?} top_p={top_p:?} must restrict to {allowed:?}, got {tok}"
                    );
                }
            }
        }
        {
            let logits = vec![0.4_f32, 0.7, -0.1, 0.2, 0.5, -0.3, 0.0, 0.6];
            let draws_a: Vec<u32> = {
                let mut rng = Pcg64::seed_from_u64(123);
                (0..20)
                    .map(|_| sample_logits(&logits, 1.0, None, None, &mut rng))
                    .collect()
            };
            let draws_b: Vec<u32> = {
                let mut rng = Pcg64::seed_from_u64(123);
                (0..20)
                    .map(|_| sample_logits(&logits, 1.0, None, None, &mut rng))
                    .collect()
            };
            assert_eq!(draws_a, draws_b, "same seed must produce same draws");

            let draws_c: Vec<u32> = {
                let mut rng = Pcg64::seed_from_u64(456);
                (0..20)
                    .map(|_| sample_logits(&logits, 1.0, None, None, &mut rng))
                    .collect()
            };
            assert_ne!(
                draws_a, draws_c,
                "different seeds should diverge over 20 draws"
            );
        }
    }

    fn spec_params(
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> nv_layers::sampler::SamplingParams {
        nv_layers::sampler::SamplingParams {
            temperature,
            top_k,
            top_p,
            min_p: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
        }
    }

    #[test]
    fn spec_rejection_sampling_matches_the_target_distribution_and_respects_topk() {
        for (logits, params, drafts, n, tol) in [
            (
                vec![2.0f32, 1.0, 0.5, 0.0, -1.0, 1.7],
                spec_params(0.8, None, None),
                vec![0u32, 3, 5],
                400_000u64,
                0.005f64,
            ),
            (
                vec![3.0f32, 2.4, 1.0, -2.0, -5.0, 0.3, 2.9],
                spec_params(1.0, None, Some(0.8)),
                vec![0u32, 2, 4, 6],
                150_000u64,
                0.008f64,
            ),
        ] {
            let target = nv_layers::sampler::distribution(&logits, &params);
            if params.top_p.is_some() {
                assert!(target.contains(&0.0), "top_p must zero some tail");
            }
            for drafted in drafts {
                let mut s =
                    ChatSampler::new(params, 0xC0FFEE ^ drafted as u64, None, vec![], false, 0);
                let mut hist = vec![0u64; logits.len()];
                for _ in 0..n {
                    let out = match s.accept_draft(&logits, drafted) {
                        DraftOutcome::Accept => drafted,
                        DraftOutcome::Reject(r) => r,
                    };
                    hist[out as usize] += 1;
                }
                for t in 0..logits.len() {
                    let emp = hist[t] as f64 / n as f64;
                    let tgt = target[t] as f64;
                    assert!(
                        (emp - tgt).abs() < tol,
                        "drafted={drafted} token={t}: empirical {emp:.5} vs target {tgt:.5}"
                    );
                }
            }
        }
        {
            let logits = vec![10.0f32, -100.0, -100.0, 10.0, -100.0];
            let params = spec_params(1.0, Some(2), None);
            let mut s = ChatSampler::new(params, 7, None, vec![], false, 0);

            for _ in 0..2000 {
                match s.accept_draft(&logits, 1) {
                    DraftOutcome::Accept => panic!("out-of-nucleus token accepted"),
                    DraftOutcome::Reject(r) => {
                        assert!(r == 0 || r == 3, "replacement {r} not in top-2")
                    }
                }
            }
        }
    }

    #[test]
    fn repetition_penalty_seeds_from_the_prompt_and_logprobs_report_the_raw_distribution() {
        let logits = vec![2.0f32, 3.0];
        let params = nv_layers::sampler::SamplingParams {
            repetition_penalty: 4.0,
            ..spec_params(0.0, None, None)
        };

        let mut unseeded = ChatSampler::new(params, 1, None, vec![], false, 0);
        assert_eq!(unseeded.sample(&logits).token, 1);

        let mut seeded = ChatSampler::new(params, 1, None, vec![], false, 0);
        seeded.seed_prompt(&[1, 1]);
        assert_eq!(seeded.sample(&logits).token, 0);

        let mut spec = ChatSampler::new(params, 1, None, vec![], false, 0);
        spec.seed_prompt(&[1]);
        match spec.accept_draft(&logits, 1) {
            DraftOutcome::Reject(r) => assert_eq!(r, 0),
            DraftOutcome::Accept => panic!("penalized prompt token must lose the argmax"),
        }
        {
            let logits = vec![2.0f32, 3.0];
            let params = nv_layers::sampler::SamplingParams {
                presence_penalty: 5.0,
                frequency_penalty: 5.0,
                ..spec_params(0.0, None, None)
            };
            let mut s = ChatSampler::new(params, 1, None, vec![], false, 0);
            s.seed_prompt(&[1]);
            assert!(s.prompt_tokens.is_empty());
            assert_eq!(s.sample(&logits).token, 1);
        }
        {
            let logits = vec![1.0f32, 2.0, 0.5, -1.0];
            let raw = nv_layers::sampler::logprobs_full(&logits, 1.0);

            let params = spec_params(0.8, None, None);
            let mut s = ChatSampler::new(params, 42, None, vec![], true, 2);
            let out = s.sample(&logits);
            let lp = out.logprob.expect("logprobs requested");
            assert!(
                (lp - raw[out.token as usize]).abs() < 1e-6,
                "sampled-token logprob {lp} != raw {}",
                raw[out.token as usize]
            );
            assert_eq!(out.top.len(), 2);
            assert_eq!(out.top[0].0, 1, "top-1 must be the raw argmax");
            assert!((out.top[0].1 - raw[1]).abs() < 1e-6);

            let scaled = nv_layers::sampler::logprobs_full(&logits, 0.8);
            assert!((raw[1] - scaled[1]).abs() > 1e-3);

            let params = nv_layers::sampler::SamplingParams {
                repetition_penalty: 2.0,
                ..spec_params(0.9, None, None)
            };
            let mut s = ChatSampler::new(params, 7, None, vec![(1, -100.0)], true, 1);
            let out = s.sample(&logits);
            let lp = out.logprob.expect("logprobs requested");
            assert!(
                (lp - raw[out.token as usize]).abs() < 1e-6,
                "sampled-token logprob must come from raw logits"
            );
            assert_eq!(
                out.top[0].0, 1,
                "top logprob from raw logits, ignoring bias"
            );
            assert!((out.top[0].1 - raw[1]).abs() < 1e-6);
        }
    }

    #[test]
    fn spec_acceptance_is_exactly_the_leviathan_case_split_and_greedy_short_circuits_on_the_gpu() {
        let mut lcg: u64 = 0x5EED_CAFE;
        let mut next = |m: u64| {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (lcg >> 33) % m
        };
        let mut checked_accept = 0usize;
        let mut checked_reject = 0usize;
        for case in 0..2000u64 {
            let n = 3 + (next(8) as usize);
            let logits: Vec<f32> = (0..n).map(|_| (next(8000) as f32 / 1000.0) - 4.0).collect();
            let temp = 0.3 + next(150) as f32 / 100.0;
            let top_p = if next(2) == 0 {
                Some(0.5 + next(45) as f32 / 100.0)
            } else {
                None
            };
            let params = spec_params(temp, None, top_p);
            let drafted = next(n as u64) as u32;
            let seed = 0xABCD ^ case;

            let mut probe = ChatSampler::new(params, seed, None, vec![], false, 0);
            let u1 = probe.uniform_f64();
            let u2 = probe.uniform_f64();

            let probs = nv_layers::sampler::distribution(&logits, &params);
            let px = probs.get(drafted as usize).copied().unwrap_or(0.0) as f64;

            let mut s = ChatSampler::new(params, seed, None, vec![], false, 0);
            let out = s.accept_draft(&logits, drafted);
            if u1 < px {
                assert!(
                    matches!(out, DraftOutcome::Accept),
                    "case {case}: u1={u1} < p(x)={px} must accept"
                );
                checked_accept += 1;
            } else {
                let want = nv_layers::sampler::residual_sample_checked(&probs, drafted, u2)
                    .unwrap_or_else(|| nv_layers::sampler::argmax(&logits));
                match out {
                    DraftOutcome::Reject(r) => {
                        assert_eq!(r, want, "case {case}: residual draw mismatch");
                        assert!(
                            px >= 1.0 - 1e-6 || r != drafted,
                            "case {case}: residual returned the rejected draft"
                        );
                    }
                    DraftOutcome::Accept => {
                        panic!("case {case}: u1={u1} >= p(x)={px} must reject")
                    }
                }
                checked_reject += 1;
            }
        }
        assert!(
            checked_accept > 300 && checked_reject > 300,
            "both branches must be exercised: {checked_accept} accepts, {checked_reject} rejects"
        );
        {
            let params = spec_params(0.0, None, None);
            let mut s = ChatSampler::new(params, 42, None, vec![], false, 0);
            assert!(s.pure_greedy());

            let mut state = 0x9e3779b97f4a7c15u64;
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / (1u64 << 31) as f32) * 20.0 - 10.0
            };
            for case in 0..200 {
                let n = 64 + (case % 7) * 13;
                let mut logits: Vec<f32> = (0..n).map(|_| next()).collect();
                match case % 5 {
                    0 => {
                        logits[3] = 42.0;
                        logits[n - 2] = 42.0;
                    }
                    1 => logits[7] = f32::NAN,
                    2 => logits[0] = f32::INFINITY,
                    3 => logits[n - 1] = 99.0,
                    _ => {}
                }
                let am = nv_layers::sampler::argmax_checked(&logits).unwrap();
                for drafted in [am, (am + 1) % n as u32, 0, (n - 1) as u32] {
                    match s.accept_draft(&logits, drafted) {
                        DraftOutcome::Accept => assert_eq!(drafted, am, "case {case}"),
                        DraftOutcome::Reject(r) => {
                            assert_ne!(drafted, am, "case {case}");
                            assert_eq!(r, am, "case {case}: replacement must be argmax");
                        }
                    }
                }
                assert_eq!(s.draw_from_logits(&logits), am, "case {case}");
            }

            let mut p = spec_params(0.0, None, None);
            p.repetition_penalty = 1.2;
            assert!(!ChatSampler::new(p, 1, None, vec![], false, 0).pure_greedy());
            let p2 = spec_params(0.7, None, None);
            assert!(!ChatSampler::new(p2, 1, None, vec![], false, 0).pure_greedy());
            let p3 = spec_params(0.0, None, None);
            assert!(!ChatSampler::new(p3, 1, None, vec![(5, 1.0)], false, 0).pure_greedy());
        }
    }

    #[test]
    fn verify_cache_capacity_rounds_to_the_grain_and_graph_reuse_needs_the_same_k() {
        assert_eq!(verify_cache_capacity(0), VERIFY_CACHE_GRAIN);
        assert_eq!(verify_cache_capacity(1), VERIFY_CACHE_GRAIN);
        assert_eq!(
            verify_cache_capacity(VERIFY_CACHE_GRAIN),
            VERIFY_CACHE_GRAIN
        );
        assert_eq!(
            verify_cache_capacity(VERIFY_CACHE_GRAIN + 1),
            2 * VERIFY_CACHE_GRAIN
        );
        for needed in [1usize, 98, 700, 836, 1524, 2048, 2049, 4097, 100_000] {
            let cap = verify_cache_capacity(needed);
            assert!(cap >= needed);
            assert!(
                cap < needed + VERIFY_CACHE_GRAIN,
                "needed={needed} cap={cap} overshoots by a full grain"
            );
        }
        {
            assert!(verify_graph_reusable(8, 4096, 8, 4096));
            assert!(verify_graph_reusable(8, 4096, 8, 700));
            assert!(!verify_graph_reusable(8, 4096, 8, 4097));
            assert!(!verify_graph_reusable(4, 4096, 8, 700));
        }
    }

    #[test]
    fn kv_window_invariants_hold_over_a_small_grid_and_clamp_oversized_requests() {
        for kv_max_seq_len in 0..14usize {
            for prompt_len in 0..14usize {
                for max_new in 0..14usize {
                    assert_kv_window_invariants(prompt_len, max_new, kv_max_seq_len);
                    if let Some((cache_len, clamped)) =
                        kv_window(prompt_len, max_new, kv_max_seq_len)
                    {
                        for step in 0..clamped {
                            assert_kv_step_in_bounds(prompt_len, step, cache_len, kv_max_seq_len);
                        }
                    }
                }
            }
        }
        {
            assert_eq!(kv_window(6, 1000, 8), Some((8, 1)));
            assert_eq!(kv_window(7, 1000, 8), Some((8, 0)));
            assert_eq!(kv_window(8, 1000, 8), None);
            assert_eq!(kv_window(2, 3, 4096), Some((6, 3)));
            for kv_max_seq_len in 0..14usize {
                for prompt_len in 0..14usize {
                    assert_eq!(
                        kv_window(prompt_len, 7, kv_max_seq_len).is_none(),
                        prompt_len >= kv_max_seq_len,
                        "prompt_len={prompt_len} kv_max_seq_len={kv_max_seq_len}"
                    );
                }
            }
        }
    }

    struct DropProbe {
        cap: usize,
        log: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.log.borrow_mut().push(format!("drop {}", self.cap));
        }
    }

    #[test]
    fn rebuilds_and_k_changes_free_the_old_verifier_first_and_reuse_keeps_at_most_one_live() {
        let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut slot = Some(DropProbe {
            cap: 1024,
            log: log.clone(),
        });
        let built = take_reusable_or_build::<DropProbe, ()>(
            &mut slot,
            |gv| verify_graph_reusable(8, gv.cap, 8, 4096),
            || {
                log.borrow_mut().push("alloc 4096".to_string());
                Ok(DropProbe {
                    cap: 4096,
                    log: log.clone(),
                })
            },
        )
        .unwrap();
        assert_eq!(built.cap, 4096);
        assert!(slot.is_none());
        assert_eq!(
            log.borrow().as_slice(),
            ["drop 1024".to_string(), "alloc 4096".to_string()],
            "the old verify cache must be released before the larger one is allocated"
        );
        {
            let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
            let mut slot = Some(DropProbe {
                cap: 4096,
                log: log.clone(),
            });
            let reused = take_reusable_or_build::<DropProbe, ()>(
                &mut slot,
                |gv| verify_graph_reusable(8, gv.cap, 8, 836),
                || -> Result<DropProbe, ()> {
                    panic!("must not rebuild when the cached verifier fits")
                },
            )
            .unwrap();
            assert_eq!(reused.cap, 4096);
            assert!(
                slot.is_none(),
                "the verifier must be moved out, not aliased"
            );
            assert!(log.borrow().is_empty());
            drop(reused);
            assert_eq!(log.borrow().as_slice(), ["drop 4096".to_string()]);
        }
        {
            let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
            let k = 8usize;
            let needed = |prompt: usize, max_new: usize| prompt + max_new + k + 16;

            let mut slot: Option<DropProbe> = None;
            let mut builds = 0usize;
            let mut caps: Vec<usize> = Vec::new();

            for (prompt, max_new) in [
                (300usize, 512usize),
                (10, 64),
                (280, 500),
                (900, 600),
                (50, 128),
            ] {
                let need = needed(prompt, max_new);
                let gv = take_reusable_or_build::<DropProbe, ()>(
                    &mut slot,
                    |gv| verify_graph_reusable(k, gv.cap, k, need),
                    || {
                        builds += 1;
                        log.borrow_mut().push(format!("alloc {need}"));
                        Ok(DropProbe {
                            cap: verify_cache_capacity(need),
                            log: log.clone(),
                        })
                    },
                )
                .unwrap();
                assert!(gv.cap >= need, "capacity {} < needed {need}", gv.cap);
                caps.push(gv.cap);
                slot = Some(gv);
            }

            assert_eq!(builds, 2, "only the two capacity-raising requests rebuild");
            assert!(
                caps.windows(2).all(|w| w[1] >= w[0]),
                "capacity must ratchet monotonically: {caps:?}"
            );
            assert!(slot.is_some(), "the verifier must be put back for reuse");

            let live = log
                .borrow()
                .iter()
                .scan(0isize, |n, ev| {
                    *n += if ev.starts_with("alloc") { 1 } else { -1 };
                    Some(*n)
                })
                .max()
                .unwrap();
            assert_eq!(
                live,
                1,
                "two verify caches were resident at once: {:?}",
                log.borrow()
            );
        }
        {
            let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
            let mut slot = Some(DropProbe {
                cap: 4096,
                log: log.clone(),
            });
            let built = take_reusable_or_build::<DropProbe, ()>(
                &mut slot,
                |gv| verify_graph_reusable(4, gv.cap, 8, 836),
                || {
                    log.borrow_mut().push("alloc 1024".to_string());
                    Ok(DropProbe {
                        cap: 1024,
                        log: log.clone(),
                    })
                },
            )
            .unwrap();
            assert_eq!(built.cap, 1024);
            assert_eq!(
                log.borrow().as_slice(),
                ["drop 4096".to_string(), "alloc 1024".to_string()]
            );
        }
    }

    #[test]
    fn k_and_chunk_env_knobs_clamp_into_range() {
        assert_eq!(eagle3_k(None, 0), 4);
        assert_eq!(eagle3_k(None, 0), EAGLE3_K_SHORT_DEFAULT);
        assert_eq!(
            eagle3_k(None, EAGLE3_K_CTX_GATE - 1),
            EAGLE3_K_SHORT_DEFAULT
        );
        assert_eq!(eagle3_k(None, EAGLE3_K_CTX_GATE), EAGLE3_K_DEFAULT);
        assert_eq!(eagle3_k(None, 16384), EAGLE3_K_DEFAULT);
        assert_eq!(eagle3_k(Some("not-a-number"), 0), EAGLE3_K_SHORT_DEFAULT);
        assert_eq!(eagle3_k(Some("not-a-number"), 16384), EAGLE3_K_DEFAULT);
        assert_eq!(eagle3_k(Some(""), 0), EAGLE3_K_SHORT_DEFAULT);
        assert_eq!(eagle3_k(Some("0"), 0), EAGLE3_K_MIN);
        assert_eq!(eagle3_k(Some("1"), 0), EAGLE3_K_MIN);
        assert_eq!(eagle3_k(Some(" 4 "), 16384), 4);
        assert_eq!(eagle3_k(Some("3"), 0), 3);
        assert_eq!(eagle3_k(Some("64"), 0), EAGLE3_K_MAX);
        assert_eq!(eagle3_k(Some("65"), 0), EAGLE3_K_MAX);
        assert_eq!(eagle3_k(Some("18446744073709551615"), 0), EAGLE3_K_MAX);
        {
            assert_eq!(dflash_k(None, 8), DFLASH_K_DEFAULT);
            assert_eq!(dflash_k(Some("9"), 8), 9);
            assert_eq!(dflash_k(Some("12"), 8), 9);
            assert_eq!(dflash_k(Some("0"), 8), EAGLE3_K_MIN);
            assert_eq!(dflash_k(Some("1"), 8), EAGLE3_K_MIN);
            assert_eq!(dflash_k(Some("not-a-number"), 8), DFLASH_K_DEFAULT);
            assert_eq!(dflash_k(Some(""), 8), DFLASH_K_DEFAULT);
            assert_eq!(dflash_k(None, 16), DFLASH_K_DEFAULT);
            assert_eq!(dflash_k(Some("15"), 16), 15);
            assert_eq!(dflash_k(Some("17"), 16), 17);
            assert_eq!(dflash_k(Some("18"), 16), 17);
            assert_eq!(dflash_k(Some("200"), 200), EAGLE3_K_MAX);
            assert_eq!(dflash_k(None, 0), EAGLE3_K_MIN);
        }
        {
            assert_eq!(dflash_prose_k(None, 8, 8), 8);
            assert_eq!(dflash_prose_k(Some(""), 8, 8), 8);
            assert_eq!(dflash_prose_k(Some("junk"), 8, 8), 8);
            assert_eq!(dflash_prose_k(Some("4"), 8, 8), 4);
            assert_eq!(dflash_prose_k(Some(" 4 "), 8, 8), 4);
            assert_eq!(dflash_prose_k(Some("0"), 8, 8), EAGLE3_K_MIN);
            assert_eq!(dflash_prose_k(Some("64"), 8, 8), 9);
            assert_eq!(dflash_prose_k(Some("12"), 16, 16), 12);
        }
        {
            assert_eq!(spec_prefill_chunk(None), SPEC_PREFILL_CHUNK_DEFAULT);
            assert_eq!(spec_prefill_chunk(Some("junk")), SPEC_PREFILL_CHUNK_DEFAULT);
            assert_eq!(spec_prefill_chunk(Some("")), SPEC_PREFILL_CHUNK_DEFAULT);
            assert_eq!(spec_prefill_chunk(Some("0")), SPEC_PREFILL_CHUNK_MIN);
            assert_eq!(spec_prefill_chunk(Some("512")), 512);
            assert_eq!(spec_prefill_chunk(Some(" 2048 ")), 2048);

            assert_eq!(spec_prefill_chunk(Some("65536")), SPEC_PREFILL_CHUNK_MAX);
            assert_eq!(
                spec_prefill_chunk(Some("18446744073709551615")),
                SPEC_PREFILL_CHUNK_MAX
            );
        }
    }

    #[test]
    fn drafter_kind_and_flag_env_knobs_parse_their_shapes() {
        assert_eq!(nv_drafter_kind(None), "eagle3");
        assert_eq!(nv_drafter_kind(Some("")), "eagle3");
        assert_eq!(nv_drafter_kind(Some("eagle3")), "eagle3");
        assert_eq!(nv_drafter_kind(Some("dflash")), "dflash");
        assert_eq!(nv_drafter_kind(Some(" dflash ")), "dflash");
        assert_eq!(nv_drafter_kind(Some("auto")), "auto");
        assert_eq!(nv_drafter_kind(Some(" auto ")), "auto");
        assert_eq!(nv_drafter_kind(Some("route")), "route");
        assert_eq!(nv_drafter_kind(Some(" route ")), "route");
        assert_eq!(nv_drafter_kind(Some("mtp")), "mtp");
        assert_eq!(nv_drafter_kind(Some("bogus")), "eagle3");
        {
            assert!(!suffix_drafter_enabled(None));
            assert!(!suffix_drafter_enabled(Some("0")));
            assert!(!suffix_drafter_enabled(Some("")));
            assert!(!suffix_drafter_enabled(Some(" ")));
            assert!(suffix_drafter_enabled(Some("1")));
            assert!(suffix_drafter_enabled(Some("on")));

            assert_eq!(suffix_min_match(None), 4);
            assert_eq!(suffix_min_match(Some("")), 4);
            assert_eq!(suffix_min_match(Some("bogus")), 4);
            assert_eq!(suffix_min_match(Some("0")), 4);
            assert_eq!(suffix_min_match(Some("65")), 4);
            assert_eq!(suffix_min_match(Some("1")), 1);
            assert_eq!(suffix_min_match(Some(" 8 ")), 8);
            assert_eq!(suffix_min_match(Some("64")), 64);
        }
        {
            assert_eq!(route_ctx_gate(None), ROUTE_CTX_GATE_DEFAULT);
            assert_eq!(route_ctx_gate(Some("")), ROUTE_CTX_GATE_DEFAULT);
            assert_eq!(route_ctx_gate(Some("junk")), ROUTE_CTX_GATE_DEFAULT);
            assert_eq!(route_ctx_gate(Some("0")), ROUTE_CTX_GATE_DEFAULT);
            assert_eq!(route_ctx_gate(Some("-1")), ROUTE_CTX_GATE_DEFAULT);
            assert_eq!(route_ctx_gate(Some("4096")), 4096);
            assert_eq!(route_ctx_gate(Some(" 3072 ")), 3072);
        }
        {
            assert!(!adaptive_k_enabled(None));
            assert!(!adaptive_k_enabled(Some("0")));
            assert!(!adaptive_k_enabled(Some("")));
            assert!(adaptive_k_enabled(Some("1")));
            assert!(adaptive_k_enabled(Some(" 1 ")));
            assert!(!adaptive_k_enabled(Some("yes")));

            assert_eq!(adaptive_k_graph(None, 3), ADAPTIVE_K_MAX_DEFAULT);
            assert_eq!(adaptive_k_graph(None, 12), 12);
            assert_eq!(adaptive_k_graph(Some("4"), 3), 4);
            assert_eq!(adaptive_k_graph(Some("1"), 3), 3);
            assert_eq!(adaptive_k_graph(Some("1"), 2), EAGLE3_K_MIN);
            assert_eq!(adaptive_k_graph(Some("junk"), 3), ADAPTIVE_K_MAX_DEFAULT);
            assert_eq!(adaptive_k_graph(Some("9999"), 3), EAGLE3_K_MAX);
        }
        {
            assert!(!env_flag_enabled(None));
            assert!(!env_flag_enabled(Some("0")));
            assert!(env_flag_enabled(Some("1")));
            assert!(env_flag_enabled(Some("")));
            assert!(env_flag_enabled(Some("true")));
        }
        {
            for f in [
                eagle3_required as fn(Option<&str>) -> bool,
                dflash_required,
                nv_no_spec,
            ] {
                for off in [None, Some(""), Some("0"), Some(" 0 "), Some("false"), Some("FALSE")] {
                    assert!(!f(off), "{off:?}");
                }
                for on in [Some("1"), Some("true"), Some("yes")] {
                    assert!(f(on), "{on:?}");
                }
            }
        }
    }

    #[test]
    fn drafter_arm_routing_follows_the_decision_table_keyed_on_context_and_code_heuristic() {
        use DrafterArm::{DFlash, Eagle3};
        use PromptClass::{Code, Prose};

        let gate = ROUTE_CTX_GATE_DEFAULT;
        for kind in ["eagle3", "dflash", "auto", "route"] {
            for class in [Code, Prose] {
                for ctx in [16usize, gate + 16] {
                    assert_eq!(
                        resolve_drafter_arm(kind, class, ctx, gate, false, false, 2.0, 2.0),
                        None
                    );
                    assert_eq!(
                        resolve_drafter_arm(kind, class, ctx, gate, true, false, 0.0, 9.0),
                        Some(DFlash)
                    );
                    assert_eq!(
                        resolve_drafter_arm(kind, class, ctx, gate, false, true, 9.0, 0.0),
                        Some(Eagle3)
                    );
                }
            }
        }

        for class in [Code, Prose] {
            assert_eq!(
                resolve_drafter_arm("route", class, 16, gate, true, true, 0.0, 9.0),
                Some(DFlash)
            );
            assert_eq!(
                resolve_drafter_arm("route", class, gate, gate, true, true, 9.0, 0.0),
                Some(Eagle3)
            );
        }

        let auto_gate = drafter_auto_switch_tokens(None);
        for class in [Code, Prose] {
            assert_eq!(
                resolve_drafter_arm("auto", class, 16, auto_gate, true, true, 0.5, 9.0),
                Some(DFlash)
            );
            assert_eq!(
                resolve_drafter_arm(
                    "auto",
                    class,
                    auto_gate - 1,
                    auto_gate,
                    true,
                    true,
                    0.5,
                    9.0
                ),
                Some(DFlash)
            );
            assert_eq!(
                resolve_drafter_arm("auto", class, auto_gate, auto_gate, true, true, 9.0, 0.5),
                Some(Eagle3)
            );
            assert_eq!(
                resolve_drafter_arm("auto", class, 32768, auto_gate, true, true, 9.0, 0.5),
                Some(Eagle3)
            );
        }
        {
            assert!(prompt_looks_codeish(
                "fix this:\n```rust\nfn main() {}\n```"
            ));
            assert!(prompt_looks_codeish(
                "#include <stdio.h>\nint main(void) {\n  printf(\"hi\");\n  return 0;\n}\n"
            ));
            assert!(prompt_looks_codeish(
                "import os\ndef walk(root):\n    for d in os.listdir(root):\n        print(d)\nclass Foo:\n    pass"
            ));
            assert!(!prompt_looks_codeish(
                "Tell me about the history of Uruguay and its relationship with Argentina."
            ));
            assert!(!prompt_looks_codeish(
                "Write a short story about a lighthouse keeper who befriends a seagull.\n\nMake it about 500 words."
            ));
            assert!(!prompt_looks_codeish(""));
            assert_eq!(
                classify_prompt("fix this:\n```rust\nfn main() {}\n```"),
                PromptClass::Code
            );
            assert_eq!(
                classify_prompt("Write a short story about a lighthouse keeper."),
                PromptClass::Prose
            );
            assert_eq!(classify_prompt(""), PromptClass::Prose);
        }
        {
            assert_eq!(route_drafter_arm(true, 0.5, 3.0), DrafterArm::DFlash);
            assert_eq!(route_drafter_arm(false, 2.5, 2.0), DrafterArm::DFlash);
            assert_eq!(route_drafter_arm(false, 2.0, 2.0), DrafterArm::DFlash);
            assert_eq!(route_drafter_arm(false, 1.5, 2.0), DrafterArm::Eagle3);
        }
        {
            use DrafterArm::{DFlash, Eagle3};
            use PromptClass::{Code, Prose};

            let gate = ROUTE_CTX_GATE_DEFAULT;
            assert_eq!(route_arm_for_ctx(0, gate), DFlash);
            assert_eq!(route_arm_for_ctx(gate - 1, gate), DFlash);
            assert_eq!(route_arm_for_ctx(gate, gate), Eagle3);
            assert_eq!(route_arm_for_ctx(gate + 1, gate), Eagle3);

            for ctx in [0usize, 256, gate - 1] {
                assert_eq!(
                    resolve_drafter_arm("route", Prose, ctx, gate, true, true, 0.0, 9.0),
                    resolve_drafter_arm("route", Code, ctx, gate, true, true, 0.0, 9.0)
                );
            }
            for ctx in [gate, 16_000usize] {
                assert_eq!(
                    resolve_drafter_arm("route", Code, ctx, gate, true, true, 9.0, 0.0),
                    Some(Eagle3)
                );
            }
        }
    }

    #[test]
    fn auto_switch_tokens_defaults_to_the_measured_dflash_collapse_boundary() {
        assert_eq!(drafter_auto_switch_tokens(None), 16384);
        assert_eq!(
            drafter_auto_switch_tokens(None),
            DFLASH_WINS_THROUGH_8K_BUT_ACCEPT_COLLAPSES_BY_32K_SO_AUTO_HANDS_OFF_TO_EAGLE3_AT_16384_PROMPT_TOKENS
        );
        assert_eq!(drafter_auto_switch_tokens(Some("")), 16384);
        assert_eq!(drafter_auto_switch_tokens(Some("0")), 16384);
        assert_eq!(drafter_auto_switch_tokens(Some("bogus")), 16384);
        assert_eq!(drafter_auto_switch_tokens(Some("8192")), 8192);
        assert_eq!(drafter_auto_switch_tokens(Some(" 4096 ")), 4096);
    }

    #[test]
    fn drafter_row_elems_charge_is_max_not_sum() {
        let eagle3 = 4096usize;
        let dflash = 20_480usize;

        assert_eq!(drafter_row_elems_charge(eagle3, dflash), dflash);
        assert_ne!(drafter_row_elems_charge(eagle3, dflash), eagle3 + dflash);
        assert_eq!(drafter_row_elems_charge(dflash, eagle3), dflash);

        assert_eq!(drafter_row_elems_charge(eagle3, 0), eagle3);
        assert_eq!(drafter_row_elems_charge(0, dflash), dflash);
        assert_eq!(drafter_row_elems_charge(0, 0), 0);

        for (e, d) in [(4096usize, 20_480usize), (0, 20_480), (4096, 0)] {
            let charge = drafter_row_elems_charge(e, d);
            assert!(charge >= e && charge >= d);
            assert!(charge <= e + d);
        }
    }

    #[test]
    fn arm_ema_defaults_moves_and_its_step_seeds_then_blends() {
        assert_eq!(arm_ema_get(DrafterArm::Eagle3), ARM_EMA_DEFAULT);
        arm_ema_observe(DrafterArm::Eagle3, 4.0);
        let after = arm_ema_get(DrafterArm::Eagle3);
        assert!(after > ARM_EMA_DEFAULT && after < 4.0);
        arm_ema_observe(DrafterArm::Eagle3, f64::NAN);
        assert_eq!(arm_ema_get(DrafterArm::Eagle3), after);
        {
            let seeded = f64::from_bits(arm_ema_step(0, ARM_EMA_DEFAULT));
            assert_eq!(seeded, ARM_EMA_DEFAULT);
            let moved = f64::from_bits(arm_ema_step(ARM_EMA_DEFAULT.to_bits(), 4.0));
            assert_eq!(
                moved,
                (1.0 - ARM_EMA_ALPHA) * ARM_EMA_DEFAULT + ARM_EMA_ALPHA * 4.0
            );
        }
    }

    #[test]
    fn adaptive_k_starts_in_range_grows_on_acceptance_shrinks_on_rejection_and_ignores_bad_timings() {
        let a = AdaptiveK::new(8, 3);
        assert_eq!(a.k_eff(), 3);
        let a = AdaptiveK::new(8, 64);
        assert_eq!(a.k_eff(), 8);
        let a = AdaptiveK::new(8, 1);
        assert_eq!(a.k_eff(), EAGLE3_K_MIN);
        {
            let mut a = AdaptiveK::new(8, 3);
            for _ in 0..200 {
                let offered = a.k_eff() - 1;
                a.observe(offered, offered, 1.2 * offered as f64, 30.0);
            }
            assert_eq!(a.k_eff(), 8, "p_ema={} should drive k to k_graph", a.p_ema);
        }
        {
            let mut a = AdaptiveK::new(8, 8);
            for _ in 0..200 {
                let offered = a.k_eff() - 1;
                a.observe(offered, 0, 2.5 * offered as f64, 30.0);
            }
            assert!(
                a.k_eff() <= 3,
                "k should shrink under rejection, got {} (p_ema={})",
                a.k_eff(),
                a.p_ema
            );
            for k in EAGLE3_K_MIN..=8 {
                assert!(a.cost_ms_per_tok(k).is_finite());
            }
        }
        {
            let mut a = AdaptiveK::new(8, 3);
            a.observe(2, 9, f64::NAN, -1.0);
            assert!(a.p_ema <= 1.0 && a.p_ema >= 0.0);
            assert!((a.d_graph_ms - 1.2).abs() < 1e-9);
            assert!((a.d_eager_ms - 2.4).abs() < 1e-9);
            assert!((a.verify_ms - 30.0).abs() < 1e-9);
            a.observe(0, 0, 5.0, 30.0);
            assert!(a.p_ema > 0.0);
        }
    }

    #[test]
    fn spec_gate_matrices_for_eagle3_dflash_and_per_request_requests() {
        use super::Eagle3Gate::*;

        assert_eq!(eagle3_gate(false, false, false), NotRequested);
        assert_eq!(eagle3_gate(false, true, false), NotRequested);
        assert_eq!(eagle3_gate(false, false, true), NotRequested);
        assert_eq!(eagle3_gate(false, true, true), NotRequested);

        assert_eq!(eagle3_gate(true, false, true), Enabled);
        assert_eq!(eagle3_gate(true, true, true), Enabled);

        assert_eq!(eagle3_gate(true, false, false), DegradedWarn);
        assert_eq!(eagle3_gate(true, true, false), RequiredFail);
        {
            assert!(drafter_wants_dflash("dflash"));
            assert!(drafter_wants_dflash("auto"));
            assert!(drafter_wants_dflash("route"));
            assert!(!drafter_wants_dflash("eagle3"));
            assert!(dflash_spec_requested(false, "dflash"));
            assert!(dflash_spec_requested(false, "auto"));
            assert!(dflash_spec_requested(false, "route"));
            assert!(!dflash_spec_requested(false, "eagle3"));
            assert!(!dflash_spec_requested(true, "dflash"));
            assert!(!dflash_spec_requested(true, "auto"));
        }
        {
            assert!(spec_requested(false, false, true));
            assert!(spec_requested(false, true, false));
            assert!(spec_requested(false, true, true));
            assert!(!spec_requested(false, false, false));

            assert!(!spec_requested(true, false, false));
            assert!(!spec_requested(true, true, false));
            assert!(!spec_requested(true, false, true));
            assert!(!spec_requested(true, true, true));
        }
        {
            assert!(spec_gate_for_request(false, false, true));
            assert!(spec_gate_for_request(false, true, true));
            assert!(spec_gate_for_request(false, true, false));

            assert!(!spec_gate_for_request(false, false, false));

            assert!(!spec_gate_for_request(true, false, true));
            assert!(!spec_gate_for_request(true, true, true));
            assert!(!spec_gate_for_request(true, true, false));
            assert!(!spec_gate_for_request(true, false, false));
        }
    }

    fn seq_input(seq_id: u64, position: usize) -> nv_engine::SeqInput {
        nv_engine::SeqInput {
            seq_id,
            position,
            token: 0,
            block_table: Vec::new(),
        }
    }

    #[test]
    fn graph_decode_eligible_passes_batch_and_max_total_and_requires_cache_len_lockstep() {
        assert!(!graph_decode_eligible(&[], |_, _| true, |_| Some(0usize)));
        assert!(!graph_decode_eligible(
            &[seq_input(1, 0)],
            |_, _| false,
            |_| Some(0usize),
        ));
        let items = [seq_input(1, 10), seq_input(2, 5)];
        assert!(graph_decode_eligible(
            &items,
            |_, _| true,
            |id| Some(if id == 1 { 10 } else { 5 })
        ));
        assert!(!graph_decode_eligible(
            &items,
            |_, _| true,
            |id| Some(if id == 1 { 9 } else { 5 })
        ));
        assert!(!graph_decode_eligible(
            &items,
            |_, _| true,
            |id| Some(if id == 1 { 10 } else { 6 })
        ));
        assert!(!graph_decode_eligible(
            &items,
            |_, _| true,
            |id| (id == 1).then_some(10)
        ));
        {
            let items = [seq_input(7, 10), seq_input(9, 5)];
            let lens = |id: u64| Some(if id == 7 { 10 } else { 5 });
            assert!(graph_decode_eligible(
                &items,
                |batch, max_total| {
                    assert_eq!(batch, 2);
                    assert_eq!(max_total, 11);
                    true
                },
                lens,
            ));
        }
    }

    #[test]
    fn spec_verify_cache_len_never_overflows_covers_every_row_and_never_sizes_past_the_kv_window() {
        let vals = [
            0usize,
            1,
            2,
            13,
            4096,
            usize::MAX / 2,
            usize::MAX - 2,
            usize::MAX - 1,
            usize::MAX,
        ];
        for &kv_max_seq_len in &vals {
            for &prompt_len in &vals {
                for &max_new in &vals {
                    for k in [EAGLE3_K_MIN, 8, EAGLE3_K_MAX] {
                        let got = spec_verify_cache_len(prompt_len, max_new, k, kv_max_seq_len);
                        match kv_window(prompt_len, max_new, kv_max_seq_len) {
                            None => assert_eq!(
                                got, None,
                                "sized a cache for a prompt that does not fit \
                                 (prompt_len={prompt_len} max_new={max_new} kv={kv_max_seq_len})"
                            ),
                            Some((cache_len, clamped)) => {
                                assert_eq!(cache_len, prompt_len + clamped + 1);
                                match got {
                                    Some(max_seq) => assert!(
                                        max_seq >= cache_len + k + SPEC_VERIFY_HEADROOM
                                            && max_seq > cache_len,
                                        "max_seq={max_seq} does not cover the speculative \
                                         window above cache_len={cache_len} (k={k})"
                                    ),
                                    None => assert!(
                                        cache_len.checked_add(k + SPEC_VERIFY_HEADROOM).is_none(),
                                        "refused a request that fits in usize \
                                         (cache_len={cache_len} k={k})"
                                    ),
                                }
                            }
                        }
                    }
                }
            }
        }
        {
            for kv_max_seq_len in 0..14usize {
                for prompt_len in 0..14usize {
                    for max_new in 0..14usize {
                        for k in [2usize, 3, 8] {
                            let Some(max_seq) =
                                spec_verify_cache_len(prompt_len, max_new, k, kv_max_seq_len)
                            else {
                                continue;
                            };
                            let (_, clamped) = kv_window(prompt_len, max_new, kv_max_seq_len).unwrap();
                            let mut committed = prompt_len;
                            let mut emitted = 0usize;
                            while emitted < clamped {
                                for row in committed..committed + k {
                                    assert!(
                                        row < max_seq,
                                        "verify row {row} past max_seq={max_seq} \
                                         (prompt_len={prompt_len} max_new={max_new} k={k})"
                                    );
                                }
                                for _ in 0..k {
                                    if emitted >= clamped {
                                        break;
                                    }
                                    committed += 1;
                                    emitted += 1;
                                }
                            }
                            for row in committed..committed + k {
                                assert!(row < max_seq, "post-stop verify row {row} past {max_seq}");
                            }
                        }
                    }
                }
            }
        }
        {
            let vals = [0usize, 1, 2, 13, 255, 256, 4096, usize::MAX / 2, usize::MAX];
            for &kv_max_seq_len in &vals {
                for &prompt_len in &vals {
                    for &max_new in &vals {
                        for k in [EAGLE3_K_MIN, 8, EAGLE3_K_MAX] {
                            if let Some((max_seq, clamped)) =
                                spec_verify_window(prompt_len, max_new, k, kv_max_seq_len)
                            {
                                assert!(
                                    max_seq <= kv_max_seq_len,
                                    "max_seq={max_seq} > kv={kv_max_seq_len}"
                                );
                                assert!(clamped <= max_new, "grew max_new");
                                assert!(max_seq >= prompt_len, "cache cannot hold the prefill");
                                if clamped > 0 {
                                    assert!(
                                        prompt_len + clamped + 1 + k + SPEC_VERIFY_HEADROOM
                                            <= kv_max_seq_len
                                    );
                                }

                                assert!(kv_window(prompt_len, max_new, kv_max_seq_len).is_some());
                            } else {
                                assert!(kv_window(prompt_len, max_new, kv_max_seq_len).is_none());
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn spec_verify_window_allocation_covers_every_row_tightens_the_legacy_sizing_and_leaves_the_common_case_unclamped() {
        for kv_max_seq_len in [64usize, 256, 512, 4096, 8192, 16384] {
            for k in [EAGLE3_K_MIN, 3, 4, 8, EAGLE3_K_MAX] {
                for prompt_len in [1usize, 2, 17, 63, 100, 255, 511] {
                    if prompt_len >= kv_max_seq_len {
                        continue;
                    }

                    for max_new in [usize::MAX / 4, kv_max_seq_len, kv_max_seq_len * 4, 64] {
                        let Some((max_seq, clamped)) =
                            spec_verify_window(prompt_len, max_new, k, kv_max_seq_len)
                        else {
                            continue;
                        };
                        let alloc = verify_cache_capacity(max_seq).min(kv_max_seq_len);
                        assert!(
                            alloc >= max_seq,
                            "allocated {alloc} rows for a max_seq of {max_seq} \
                             (prompt_len={prompt_len} max_new={max_new} k={k} kv={kv_max_seq_len})"
                        );

                        assert!(prompt_len <= alloc, "prefill past the allocation");
                        let mut committed = prompt_len;
                        let mut emitted = 0usize;
                        while emitted < clamped {
                            assert!(
                                committed + k <= alloc,
                                "verify rows [{committed}, {}) run past the {alloc}-row \
                                 allocation (prompt_len={prompt_len} max_new={max_new} \
                                 k={k} kv={kv_max_seq_len})",
                                committed + k
                            );
                            for _ in 0..k {
                                if emitted >= clamped {
                                    break;
                                }
                                committed += 1;
                                emitted += 1;
                            }
                        }

                        assert!(clamped == 0 || committed < alloc);
                    }
                }
            }
        }
        {
            let (prompt_len, max_new, k, kv) = (100usize, 1_000_000usize, 4usize, 16384usize);
            let legacy = spec_verify_cache_len(prompt_len, max_new, k, kv).unwrap();
            let legacy_alloc = verify_cache_capacity(legacy).min(kv);
            assert!(
                legacy_alloc < legacy,
                "expected the legacy sizing to be truncated by the window \
                 (max_seq={legacy} alloc={legacy_alloc})"
            );

            let (_, clamped) = kv_window(prompt_len, max_new, kv).unwrap();
            let last_verify_start = prompt_len + clamped - 1;
            assert!(
                last_verify_start + k > legacy_alloc,
                "expected an out-of-bounds append at rows [{last_verify_start}, {}) \
                 into a {legacy_alloc}-row cache",
                last_verify_start + k
            );

            let (max_seq, clamped2) = spec_verify_window(prompt_len, max_new, k, kv).unwrap();
            let alloc = verify_cache_capacity(max_seq).min(kv);
            assert!(alloc >= max_seq);
            assert!(prompt_len + clamped2 - 1 + k <= alloc);
            assert!(clamped2 < clamped, "the clamp must actually tighten");
        }
        {
            for (prompt_len, max_new, k, kv) in [
                (190usize, 256usize, 4usize, 8192usize),
                (1, 256, 4, 8192),
                (2048, 1024, 8, 16384),
            ] {
                let (max_seq, clamped) = spec_verify_window(prompt_len, max_new, k, kv).unwrap();
                assert_eq!(clamped, max_new);
                assert_eq!(max_seq, prompt_len + max_new + 1 + k + SPEC_VERIFY_HEADROOM);
            }
        }
        {
            for kv_max in [4096usize, 8192, 16384] {
                for k in [2usize, 4, 8, 64] {
                    for prompt_len in [1usize, 100, kv_max / 2, kv_max - 2, kv_max - 1] {
                        let Some(max_seq) = spec_verify_cache_len(prompt_len, kv_max * 2, k, kv_max)
                        else {
                            continue;
                        };
                        let alloc = verify_cache_capacity(max_seq);
                        assert!(
                            alloc >= max_seq,
                            "alloc {alloc} < needed {max_seq} (kv_max={kv_max} k={k} prompt={prompt_len})"
                        );

                        let old = verify_cache_capacity(max_seq).min(kv_max);
                        if prompt_len < kv_max - 1 && prompt_len + 1 < kv_max {
                            let committed_max = prompt_len + (kv_max - prompt_len - 1) + 1;
                            if committed_max == kv_max {
                                assert!(
                                    old < max_seq,
                                    "expected the legacy min() clamp to undersize at the window"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn suffix_rounds_interleave_with_drafter_kv_bookkeeping_in_lockstep() {
        use nv_specdecode::chain::{
            accept_prefix_argmax, aux_row_extract, build_chain_batch, chain_positions,
            lower_tri_mask, ChainJudgment, ChainState, ChainVerifier, ChainVerifyOut,
        };
        use nv_specdecode::{suffix_arm_wins, AcceptEma, SuffixAutomaton};

        struct ScriptedVerifier {
            target: Vec<u32>,
            k: usize,
            calls: usize,
        }

        impl ChainVerifier for ScriptedVerifier {
            fn verify_chain(
                &mut self,
                batch: &[u32],
                positions: &[i32],
                mask: &[u8],
                committed: usize,
                want_logits: bool,
            ) -> anyhow::Result<ChainVerifyOut> {
                assert!(!want_logits);
                assert_eq!(batch.len(), self.k);
                assert_eq!(positions, &chain_positions(committed, self.k)[..]);
                assert_eq!(mask, &lower_tri_mask(self.k)[..]);
                self.calls += 1;
                let amax: Vec<u32> = (0..self.k)
                    .map(|i| self.target[committed + i + 1])
                    .collect();
                Ok(ChainVerifyOut {
                    judgment: ChainJudgment::Argmax(amax),
                    aux: vec![1.0f32; self.k],
                })
            }
        }

        let k = 4usize;
        let fc_in = 1usize;
        let n_layers_aux = 1usize;
        let hidden = 1usize;
        let suffix_min = 2usize;

        let target: Vec<u32> = (0..70usize)
            .map(|i| {
                if i < 30 {
                    100 + (i % 3) as u32
                } else {
                    200 + (i % 4) as u32
                }
            })
            .collect();
        let prompt_len = 9usize;

        let mut context: Vec<u32> = target[..prompt_len].to_vec();
        let mut committed = context.len();
        let mut st = ChainState::new(&context, fc_in).unwrap();
        let mut verifier = ScriptedVerifier {
            target: target.clone(),
            k,
            calls: 0,
        };

        let mut aux_rows = context.len();
        let mut aux_pending_rows = 0usize;
        let mut aux_base = 0usize;
        let mut drafter_kv_len = 0usize;

        let mut sam = SuffixAutomaton::new();
        let mut sam_fed = 0usize;
        let mut drafter_ema = AcceptEma::new(0.2, 2.0);

        let mut bonus = target[committed];
        let mut kinds: Vec<bool> = Vec::new();
        let mut rounds_since_model_draft = 0usize;
        let mut multi_round_stale_aux_seen = false;

        for _round in 0..40 {
            if committed >= 50 {
                break;
            }

            st.assert_round_start(k, 4096).unwrap();
            aux_rows += aux_pending_rows;
            aux_pending_rows = 0;
            assert_eq!(
                aux_rows,
                context.len() - aux_base,
                "aux tail must cover exactly [aux_base, context.len())"
            );

            sam.extend_slice(&context[sam_fed..]);
            sam.extend(bonus);
            sam_fed = context.len() + 1;
            let mut suffix_draft: Option<Vec<u32>> = None;
            if let Some(p) = sam.propose(k - 1, suffix_min) {
                if suffix_arm_wins(p.tokens.len(), suffix_min, p.match_len, drafter_ema.value()) {
                    suffix_draft = Some(p.tokens);
                }
            }
            let from_suffix = suffix_draft.is_some();
            kinds.push(from_suffix);

            let draft = if let Some(sd) = suffix_draft.take() {
                pad_suffix_draft(sd, k, bonus)
            } else {
                if rounds_since_model_draft >= 2 {
                    multi_round_stale_aux_seen = true;
                }
                assert_eq!(
                    aux_rows,
                    context.len() - aux_base,
                    "chain_draft_cached tail contract: aux rows must be ctx - aux_base"
                );
                assert!(
                    drafter_kv_len <= context.len(),
                    "drafter KV is append-only and must not be ahead of the context"
                );
                assert!(
                    aux_base <= drafter_kv_len,
                    "aux tail must start within the encoded drafter prefix"
                );
                drafter_kv_len = context.len();
                (0..k - 1).map(|j| target[committed + 1 + j]).collect()
            };
            assert_eq!(draft.len(), k - 1);

            if !from_suffix {
                aux_rows = 0;
                aux_base = context.len();
                rounds_since_model_draft = 0;
            } else {
                rounds_since_model_draft += 1;
            }

            let batch = build_chain_batch(bonus, &draft, k, true).unwrap();
            let positions = chain_positions(committed, k);
            let out = verifier
                .verify_chain(&batch, &positions, &lower_tri_mask(k), committed, false)
                .unwrap();
            let amax = match &out.judgment {
                ChainJudgment::Argmax(a) => a.clone(),
                ChainJudgment::Logits { .. } => unreachable!(),
            };
            let acc = accept_prefix_argmax(&batch, &amax).unwrap();
            assert!(acc.commit_len >= 1);

            let ema_before = drafter_ema.value();
            for (i, &tok) in batch[..acc.commit_len].iter().enumerate() {
                let row = aux_row_extract(&out.aux, n_layers_aux, k, hidden, i).unwrap();
                assert_eq!(row.len(), fc_in);
                st.commit_token(tok, &row).unwrap();
                context.push(tok);
                committed += 1;
                aux_pending_rows += 1;
                assert_eq!(
                    tok,
                    target[committed - 1],
                    "greedy stream must match the model"
                );
            }
            assert_eq!(committed, context.len());

            if from_suffix {
                assert_eq!(
                    drafter_ema.value(),
                    ema_before,
                    "suffix rounds must not perturb the model-drafter EMA"
                );
            } else {
                drafter_ema.observe(acc.commit_len - 1);
            }

            bonus = acc.next_bonus;
        }

        assert!(
            committed >= 50,
            "loop stalled before reaching the target length"
        );
        assert_eq!(context[..], target[..context.len()]);
        assert_eq!(st.committed(), committed);
        assert_eq!(st.aux_rows(), committed);

        let suffix_rounds = kinds.iter().filter(|&&s| s).count();
        let model_rounds = kinds.len() - suffix_rounds;
        assert!(suffix_rounds >= 2, "kinds={kinds:?}");
        assert!(model_rounds >= 2, "kinds={kinds:?}");
        assert!(
            kinds.windows(2).any(|w| w[0] && !w[1]),
            "need a model-drafter round right after a suffix round: {kinds:?}"
        );
        assert!(
            kinds.windows(2).any(|w| !w[0] && w[1]),
            "need a suffix round right after a model-drafter round: {kinds:?}"
        );
        assert!(
            multi_round_stale_aux_seen,
            "no model round consumed an aux tail spanning multiple suffix rounds: {kinds:?}"
        );
        assert!(verifier.calls >= kinds.len());
        {
            assert_eq!(pad_suffix_draft(vec![7], 4, 99), vec![7, 7, 7]);
            assert_eq!(pad_suffix_draft(vec![5, 6], 4, 99), vec![5, 6, 6]);
            assert_eq!(pad_suffix_draft(vec![1, 2, 3], 4, 99), vec![1, 2, 3]);
            assert_eq!(pad_suffix_draft(vec![1, 2, 3, 4], 4, 99), vec![1, 2, 3, 4]);
            assert_eq!(pad_suffix_draft(Vec::new(), 4, 99), vec![99, 99, 99]);
            assert_eq!(pad_suffix_draft(vec![8], 2, 99), vec![8]);
        }
    }

    #[test]
    fn adaptive_k_converges_to_the_cost_optimal_k_and_hysteresis_keeps_current_on_near_ties() {
        for &(p, d, v) in &[
            (0.9f64, 1.0f64, 30.0f64),
            (0.4, 2.5, 30.0),
            (0.6, 1.5, 50.0),
        ] {
            let mut a = AdaptiveK::new(8, 3);
            for _ in 0..400 {
                let offered = a.k_eff() - 1;
                let expected_acc =
                    (1..=offered).map(|i| p.powi(i as i32)).sum::<f64>().round() as usize;
                a.observe(offered, expected_acc.min(offered), d * offered as f64, v);
            }
            let brute_best = (EAGLE3_K_MIN..=8)
                .min_by(|&x, &y| {
                    a.cost_ms_per_tok(x)
                        .partial_cmp(&a.cost_ms_per_tok(y))
                        .unwrap()
                })
                .unwrap();
            let ratio = a.cost_ms_per_tok(a.k_eff()) / a.cost_ms_per_tok(brute_best);
            assert!(
                ratio <= 1.0 / ADAPTIVE_K_HYSTERESIS + 1e-9,
                "p={p} d={d} v={v}: settled k={} cost-ratio {ratio} vs best k={brute_best}",
                a.k_eff()
            );
        }
        {
            let mut a = AdaptiveK::new(8, 4);
            a.p_ema = 0.5;
            a.d_graph_ms = 1.2;
            a.d_eager_ms = 1.2;
            a.verify_ms = 30.0;
            let before = a.k_eff();
            let chosen = a.choose();
            let ratio = a.cost_ms_per_tok(chosen) / a.cost_ms_per_tok(before);
            assert!(
                chosen == before || ratio < ADAPTIVE_K_HYSTERESIS,
                "switch {before}->{chosen} without clearing hysteresis (ratio {ratio})"
            );
        }
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn route_ctx_gate_default_is_the_measured_crossover() {
        assert_eq!(ROUTE_CTX_GATE_DEFAULT, 2048);
        assert!(ROUTE_CTX_GATE_DEFAULT < EAGLE3_K_CTX_GATE);
        assert_eq!(
            route_arm_for_ctx(1024, ROUTE_CTX_GATE_DEFAULT),
            DrafterArm::DFlash
        );
        assert_eq!(
            route_arm_for_ctx(2048, ROUTE_CTX_GATE_DEFAULT),
            DrafterArm::Eagle3
        );
        assert_eq!(
            route_arm_for_ctx(4096, ROUTE_CTX_GATE_DEFAULT),
            DrafterArm::Eagle3
        );
    }

    #[test]
    fn sse_send_timeout_default_is_ten_seconds() {
        if !nv_config::env_arm_present_at_all_reread_every_call("NV_SSE_SEND_TIMEOUT_MS") {
            assert_eq!(sse_send_timeout(), std::time::Duration::from_millis(10_000));
        }
    }
}
