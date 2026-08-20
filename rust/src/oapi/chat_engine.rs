use std::path::{Path, PathBuf};
use std::sync::Arc;

use rand_core::Rng;
use rand_core::SeedableRng;
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
mod gemma4_loop;
mod gemma4_moe_loop;
mod gpt_oss_loop;
mod laguna_loop;
#[cfg(feature = "cuda")]
mod omni_loop;
mod qwen;
mod qwen38_sched;
mod route;
mod sampling;
mod spec_env;
mod spec_window;
mod stream;

#[cfg(feature = "cuda")]
pub(crate) use build::*;
pub use build::{
    allow_unknown_model, allow_unknown_model_from, default_engine, default_engine_from_env,
    registry_from_env, ChatRegistry, NvEngineChat, ALLOW_UNKNOWN_MODEL_ENV,
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
pub(crate) use omni_loop::*;
#[cfg(feature = "cuda")]
pub(crate) use qwen::*;
#[cfg_attr(not(any(test, feature = "cuda")), allow(unused_imports))]
pub(crate) use qwen38_sched::*;

#[cfg_attr(not(any(test, feature = "cuda")), allow(unused_imports))]
pub(crate) use batch::*;
#[cfg_attr(not(any(test, feature = "cuda")), allow(unused_imports))]
pub(crate) use route::*;
pub(crate) use sampling::*;
pub(crate) use spec_env::*;
#[cfg_attr(not(any(test, feature = "cuda")), allow(unused_imports))]
pub(crate) use spec_window::*;
#[cfg_attr(not(any(test, feature = "cuda")), allow(unused_imports))]
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
            ModelFamily::Qwen3_5Moe => render_qwen3_5_moe_prompt(messages),
            ModelFamily::Laguna => render_laguna_prompt(messages),
            ModelFamily::Omni | ModelFamily::GptOss => render_chat_prompt(messages),
        }
    }

    fn official_template(&self) -> Option<&crate::oapi::chat_template::ChatTemplate> {
        self.inner.chat_template.as_deref()
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
    fn push_blocking_sends_when_capacity_available() {
        let (tx, mut rx) = mpsc::channel::<ChatEvent>(2);
        let out = push_event_blocking(&tx, delta("a"), std::time::Duration::from_millis(50));
        assert_eq!(out, SsePush::Sent);
        assert!(matches!(rx.try_recv(), Ok(ChatEvent::TextDelta(s)) if s == "a"));
    }

    #[test]
    fn push_blocking_detects_closed_receiver_immediately() {
        let (tx, rx) = mpsc::channel::<ChatEvent>(2);
        drop(rx);
        let t0 = std::time::Instant::now();
        let out = push_event_blocking(&tx, delta("a"), std::time::Duration::from_secs(30));
        assert_eq!(out, SsePush::Closed);
        assert!(t0.elapsed() < std::time::Duration::from_secs(1));
    }

    #[test]
    fn push_blocking_times_out_on_full_channel_with_stalled_reader() {
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

    #[test]
    fn push_blocking_full_channel_detects_reader_disconnect_mid_wait() {
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

    #[test]
    fn push_blocking_recovers_when_reader_drains_before_deadline() {
        let (tx, mut rx) = mpsc::channel::<ChatEvent>(1);
        assert_eq!(
            push_event_blocking(&tx, delta("fill"), std::time::Duration::from_millis(10)),
            SsePush::Sent
        );
        let drainer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(30));
            let _ = rx.blocking_recv();
            rx
        });
        let out = push_event_blocking(&tx, delta("b"), std::time::Duration::from_secs(30));
        assert_eq!(out, SsePush::Sent);
        let _rx = drainer.join().unwrap();
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
    fn sse_send_timeout_default_is_ten_seconds() {
        if std::env::var_os("NV_SSE_SEND_TIMEOUT_MS").is_none() {
            assert_eq!(sse_send_timeout(), std::time::Duration::from_millis(10_000));
        }
    }

    #[test]
    fn stream_delta_pure_append() {
        assert_eq!(stream_text_delta("", "he"), "he");
        assert_eq!(stream_text_delta("he", "hello"), "llo");
        assert_eq!(stream_text_delta("hello", "hello"), "");
    }

    #[test]
    fn stream_delta_shorter_rewrite_is_empty() {
        assert_eq!(stream_text_delta("hello", "hell"), "");
        assert_eq!(stream_text_delta("hello", ""), "");
    }

    #[test]
    fn stream_delta_mid_string_rewrite_resumes_at_divergence() {
        assert_eq!(stream_text_delta("abcdef", "abXY"), "XY");
        assert_eq!(stream_text_delta("abc", "xyz"), "xyz");
    }

    #[test]
    fn stream_delta_multibyte_boundary() {
        assert_eq!(stream_text_delta("x\u{fffd}", "x\u{1f600}"), "\u{1f600}");
        assert_eq!(stream_text_delta("caf", "caf\u{e9}"), "\u{e9}");
        assert_eq!(stream_text_delta("caf\u{e9}", "caf\u{e9}s"), "s");
        assert_eq!(stream_text_delta("\u{20ac}\u{20ac}", "\u{20ac}"), "");
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
    fn emitter_plain_append_streams_everything() {
        let (out, hit, _) = run_emitter(&[], &["he", "hello", "hello world"]);
        assert_eq!(out, "hello world");
        assert!(!hit);
    }

    #[test]
    fn emitter_holds_back_trailing_fffd_until_char_completes() {
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
    }

    #[test]
    fn emitter_flushes_genuinely_incomplete_final_char() {
        let steps = ["ok \u{fffd}", "ok \u{fffd}\u{fffd}"];
        let (out, hit, _) = run_emitter(&[], &steps);
        assert!(!hit);
        assert_eq!(out, "ok \u{fffd}\u{fffd}");
    }

    #[test]
    fn emitter_stop_excluded_and_matches_at_end() {
        let (out, hit, _) = run_emitter(&["three"], &["one ", "one two ", "one two three"]);
        assert_eq!(out, "one two ");
        assert!(hit);
    }

    #[test]
    fn emitter_stop_inside_single_token() {
        let (out, hit, _) = run_emitter(&["Hell"], &["Hello"]);
        assert_eq!(out, "");
        assert!(hit);
    }

    #[test]
    fn emitter_stop_interior_occurrence_found() {
        let (out, hit, _) = run_emitter(&["ab"], &["x", "xaby"]);
        assert_eq!(out, "x");
        assert!(hit);
    }

    #[test]
    fn emitter_stop_prefix_held_back_until_disambiguated() {
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

    #[test]
    fn emitter_multibyte_stop_string() {
        let (out, hit, _) = run_emitter(&["\u{3002}"], &["猫", "猫が好き", "猫が好き\u{3002}犬も"]);
        assert_eq!(out, "猫が好き");
        assert!(hit);
    }

    #[test]
    fn emitter_earliest_stop_wins() {
        let (out, hit, _) = run_emitter(&["two", "one"], &["zero one two"]);
        assert_eq!(out, "zero ");
        assert!(hit);
    }

    #[test]
    fn emitter_after_stop_emits_nothing() {
        let stop = vec!["X".to_string()];
        let mut em = StreamEmitter::new(&stop);
        let (p1, h1) = em.step("aX");
        assert_eq!((p1.as_str(), h1), ("a", true));
        let (p2, h2) = em.step("aXbc");
        assert_eq!((p2.as_str(), h2), ("", true));
        assert_eq!(em.finish("aXbc"), "");
    }

    #[test]
    fn emitter_fffd_holdback_and_stop_interact() {
        let (out, hit, _) = run_emitter(&["STOP"], &["a\u{fffd}", "aSTOP\u{fffd}"]);
        assert_eq!(out, "a");
        assert!(hit);
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
                "<0xF0>": 11, "<0x9F>": 12, "<0x98>": 13, "<0x80>": 14
            }, "unk_token": "<unk>"}
        }"#;
        Arc::new(
            json.parse::<tokenizers::Tokenizer>()
                .expect("test tokenizer"),
        )
    }

    fn assert_incremental_matches_full(tok: &Arc<tokenizers::Tokenizer>, ids: &[u32]) {
        let mut detok = IncrementalDetok::new(tok.clone());
        for i in 0..ids.len() {
            let inc = detok.push(ids[i]).expect("push").to_string();
            let full = tok.decode(&ids[..=i], true).expect("full decode");
            assert_eq!(inc, full, "divergence at step {i} of {:?}", &ids[..=i]);
        }
    }

    #[test]
    fn incremental_detok_matches_full_decode_torture() {
        let tok = bytefallback_tokenizer();

        assert_incremental_matches_full(&tok, &[1, 2, 3]);

        assert_incremental_matches_full(&tok, &[1, 4, 5, 6, 3]);

        assert_incremental_matches_full(&tok, &[1, 4, 5, 3, 2]);

        assert_incremental_matches_full(&tok, &[1, 4, 7, 5, 6, 3]);

        assert_incremental_matches_full(&tok, &[8, 9, 11, 12, 13, 14, 9, 3]);

        assert_incremental_matches_full(&tok, &[1, 20, 2, 20, 20, 3]);

        assert_incremental_matches_full(&tok, &[7, 7, 7, 4, 5, 6, 7]);
    }

    #[test]
    fn incremental_detok_window_stays_bounded_and_exact() {
        let tok = bytefallback_tokenizer();
        let cycle = [1u32, 2, 3, 8, 9, 10, 4, 5, 6, 11, 12, 13, 14];
        let ids: Vec<u32> = (0..400).map(|i| cycle[i % cycle.len()]).collect();
        let mut detok = IncrementalDetok::new(tok.clone());
        let mut max_window = 0usize;
        for i in 0..ids.len() {
            let inc = detok.push(ids[i]).expect("push").to_string();
            max_window = max_window.max(detok.window.len());
            let full = tok.decode(&ids[..=i], true).expect("full decode");
            assert_eq!(inc, full, "divergence at step {i}");
        }
        assert!(
            max_window < 2 * DETOK_WINDOW_MAX,
            "window grew to {max_window}, expected < {}",
            2 * DETOK_WINDOW_MAX
        );
        assert!(!detok.stable.is_empty(), "cut never advanced");
    }

    #[test]
    #[ignore]
    fn detok_scaling_bench_real_tokenizer() {
        let Ok(path) = std::env::var("NV_DETOK_BENCH_TOKENIZER") else {
            return;
        };
        let tok = Arc::new(tokenizers::Tokenizer::from_file(&path).expect("tokenizer.json"));
        let text = "The quick brown fox 日本語のテキスト \u{1F680} jumps over 1234 lazy dogs. "
            .repeat(1500);
        let ids: Vec<u32> = tok
            .encode(text.as_str(), false)
            .expect("encode")
            .get_ids()
            .to_vec();
        let n = ids.len().min(16384);
        eprintln!("corpus: {n} tokens");

        let t0 = std::time::Instant::now();
        let mut detok = IncrementalDetok::new(tok.clone());
        let mut last_len = 0usize;
        for &id in &ids[..n] {
            last_len = detok.push(id).expect("push").len();
        }
        let inc_total = t0.elapsed().as_secs_f64();
        let full = tok.decode(&ids[..n], true).expect("full decode");
        assert_eq!(detok.acc, full, "incremental != full at n={n}");
        eprintln!(
            "incremental: total {:.3}s ({:.4} ms/tok, flat), final text {} bytes",
            inc_total,
            inc_total / n as f64 * 1000.0,
            last_len
        );

        for l in [1024usize, 2048, 4096, 8192, 16384] {
            if l > n {
                break;
            }
            let reps = 20usize;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                let _ = tok.decode(&ids[..l], true).expect("decode");
            }
            let per = t.elapsed().as_secs_f64() / reps as f64 * 1000.0;
            eprintln!("full re-decode at len {l}: {per:.3} ms per call");
        }

        let m = n.min(4096);
        let t0 = std::time::Instant::now();
        for i in 1..=m {
            let _ = tok.decode(&ids[..i], true).expect("decode");
        }
        let quad = t0.elapsed().as_secs_f64();
        eprintln!(
            "true quadratic loop to {m}: total {:.3}s ({:.4} ms/tok avg)",
            quad,
            quad / m as f64 * 1000.0
        );
    }

    #[test]
    fn incremental_detok_all_invalid_bytes_exact() {
        let tok = bytefallback_tokenizer();
        let ids: Vec<u32> = std::iter::repeat_n(7u32, 120).collect();
        assert_incremental_matches_full(&tok, &ids);
    }

    #[test]
    fn try_load_missing_config_errors_cleanly() {
        let tmp = tempdir_like("nveng-no-config");
        let err = NvEngineChat::try_load(&tmp).err().expect("expected Err");
        let msg = format!("{err}");
        assert!(msg.contains("config.json"), "got: {msg}");
        cleanup(&tmp);
    }

    #[test]
    fn try_load_missing_tokenizer_errors_cleanly() {
        let tmp = tempdir_like("nveng-no-tok");
        fs::write(tmp.join("config.json"), "{}").unwrap();
        let err = NvEngineChat::try_load(&tmp).err().expect("expected Err");
        let msg = format!("{err}");
        assert!(msg.contains("tokenizer.json"), "got: {msg}");
        cleanup(&tmp);
    }

    #[test]
    fn try_load_missing_safetensors_errors_cleanly() {
        let tmp = tempdir_like("nveng-no-st");
        fs::write(tmp.join("config.json"), "{}").unwrap();
        fs::write(tmp.join("tokenizer.json"), "{}").unwrap();
        let err = NvEngineChat::try_load(&tmp).err().expect("expected Err");
        let msg = format!("{err}");
        assert!(msg.contains("safetensors"), "got: {msg}");
        cleanup(&tmp);
    }

    #[test]
    fn registry_resolves_by_model_and_default() {
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
    }

    #[test]
    fn registry_dedups_duplicate_ids() {
        let a: Arc<dyn ChatEngine> = Arc::new(EchoEngine::new("dup", "x"));
        let b: Arc<dyn ChatEngine> = Arc::new(EchoEngine::new("dup", "y"));
        let reg = ChatRegistry::from_engines(vec![a, b]).expect("non-empty");
        assert_eq!(reg.model_ids(), &["dup".to_string()]);
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
    fn load_eagle3_state_returns_none_when_env_unset() {
        let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = EnvGuard::unset("NV_EAGLE3_DRAFT_DIR");
        let device = candle_core::Device::Cpu;
        let state = load_eagle3_state(&device, None);
        assert!(state.is_none(), "no env var must mean spec-decode disabled");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn load_eagle3_state_returns_none_when_dir_missing() {
        let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = EnvGuard::set(
            "NV_EAGLE3_DRAFT_DIR",
            "/definitely/not/a/real/eagle3/dir/xyzzy",
        );
        let device = candle_core::Device::Cpu;
        let state = load_eagle3_state(&device, None);
        assert!(state.is_none(), "missing dir must yield None, not panic");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn detect_family_qwen3_via_arch() {
        let cfg = r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}"#;
        let fam = detect_family(cfg).unwrap();
        assert_eq!(fam, ModelFamily::Qwen3);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn detect_family_gemma4_via_arch() {
        let cfg = r#"{"architectures":["Gemma4ForConditionalGeneration"],"model_type":"gemma4"}"#;
        let fam = detect_family(cfg).unwrap();
        assert_eq!(fam, ModelFamily::Gemma4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn detect_family_gemma4_via_model_type_only() {
        let cfg = r#"{"model_type":"gemma4"}"#;
        let fam = detect_family(cfg).unwrap();
        assert_eq!(fam, ModelFamily::Gemma4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn detect_family_gemma4_assistant_routes_to_gemma4() {
        let cfg =
            r#"{"architectures":["Gemma4AssistantForCausalLM"],"model_type":"gemma4_assistant"}"#;
        let fam = detect_family(cfg).unwrap();
        assert_eq!(fam, ModelFamily::Gemma4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn detect_family_gemma4_moe_via_enable_moe_block() {
        let cfg = r#"{"architectures":["Gemma4ForConditionalGeneration"],"model_type":"gemma4",
            "text_config":{"enable_moe_block":true,"num_experts":128}}"#;
        assert_eq!(detect_family(cfg).unwrap(), ModelFamily::Gemma4Moe);
        let top = r#"{"model_type":"gemma4","enable_moe_block":true}"#;
        assert_eq!(detect_family(top).unwrap(), ModelFamily::Gemma4Moe);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn detect_family_gemma4_moe_false_flag_stays_dense() {
        let cfg = r#"{"model_type":"gemma4","text_config":{"enable_moe_block":false}}"#;
        assert_eq!(detect_family(cfg).unwrap(), ModelFamily::Gemma4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn detect_family_qwen3_5_moe_via_arch() {
        let cfg = r#"{"architectures":["Qwen3_5MoeForConditionalGeneration"],"model_type":"qwen3_5_moe"}"#;
        let fam = detect_family(cfg).unwrap();
        assert_eq!(fam, ModelFamily::Qwen3_5Moe);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn detect_family_qwen3_5_moe_via_arch_causallm() {
        let cfg = r#"{"architectures":["Qwen3_5MoeForCausalLM"]}"#;
        let fam = detect_family(cfg).unwrap();
        assert_eq!(fam, ModelFamily::Qwen3_5Moe);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn detect_family_qwen3_5_moe_via_model_type_only() {
        let cfg = r#"{"model_type":"qwen3_5_moe"}"#;
        let fam = detect_family(cfg).unwrap();
        assert_eq!(fam, ModelFamily::Qwen3_5Moe);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn render_qwen3_5_moe_prompt_uses_im_markers_and_thinking_stub() {
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

    #[cfg(feature = "cuda")]
    #[test]
    fn detect_family_unknown_errors() {
        let cfg = r#"{"model_type":"llama"}"#;
        assert!(detect_family(cfg).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn parse_eos_ids_handles_scalar_and_array() {
        assert_eq!(parse_eos_ids(r#"{"eos_token_id":7}"#), Some(vec![7]));
        assert_eq!(
            parse_eos_ids(r#"{"eos_token_id":[1,106]}"#),
            Some(vec![1, 106])
        );
        assert_eq!(parse_eos_ids(r#"{}"#), None);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn render_gemma4_prompt_uses_turn_markers() {
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
            std::env::set_var(key, val);
            Self { key, prev }
        }
        fn unset(key: &'static str) -> Self {
            let prev = std::env::var_os(key);
            std::env::remove_var(key);
            Self { key, prev }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    use rand_core::SeedableRng;
    use rand_pcg::Pcg64;

    #[test]
    fn sampler_greedy_when_temperature_is_zero() {
        let logits = vec![0.1_f32, -1.0, 0.5, 9.9, 0.2];
        for seed in [0u64, 1, 42, 1_000_000] {
            let mut rng = Pcg64::seed_from_u64(seed);
            let tok = sample_logits(&logits, 0.0, None, None, &mut rng);
            assert_eq!(tok, 3, "greedy must pick argmax (seed={seed})");
        }

        let mut rng = Pcg64::seed_from_u64(7);
        let tok = sample_logits(&logits, 1.0, Some(1), None, &mut rng);
        assert_eq!(tok, 3);
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
    fn spec_rejection_sampling_matches_target_distribution() {
        let logits = vec![2.0f32, 1.0, 0.5, 0.0, -1.0, 1.7];
        let params = spec_params(0.8, None, None);
        let target = nv_layers::sampler::distribution(&logits, &params);
        let n = 400_000u64;

        for drafted in [0u32, 3, 5] {
            let mut s = ChatSampler::new(params, 0xC0FFEE ^ drafted as u64, None, vec![], false, 0);
            let mut hist = vec![0u64; logits.len()];
            for _ in 0..n {
                let out = match s.accept_draft(&logits, drafted) {
                    DraftOutcome::Accept => drafted,
                    DraftOutcome::Reject(r) => r,
                };
                hist[out as usize] += 1;
            }
            for t in 0..logits.len() {
                let emp = hist[t] as f32 / n as f32;
                let tgt = target[t];
                assert!(
                    (emp - tgt).abs() < 0.005,
                    "drafted={drafted} token={t}: empirical {emp:.4} vs target {tgt:.4}"
                );
            }
        }
    }

    #[test]
    fn spec_rejection_matches_target_under_topp_with_zeroed_tail() {
        let logits = vec![3.0f32, 2.4, 1.0, -2.0, -5.0, 0.3, 2.9];
        let params = spec_params(1.0, None, Some(0.8));
        let target = nv_layers::sampler::distribution(&logits, &params);
        assert!(target.contains(&0.0), "top_p must zero some tail");
        let n = 150_000u64;

        for drafted in [0u32, 2, 4, 6] {
            let mut s = ChatSampler::new(params, 0xA11CE ^ drafted as u64, None, vec![], false, 0);
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
                    (emp - tgt).abs() < 0.008,
                    "drafted={drafted} token={t}: empirical {emp:.5} vs target {tgt:.5}"
                );
            }
        }
    }

    #[test]
    fn spec_rejection_accept_rate_equals_target_mass() {
        let logits = vec![1.5f32, 0.2, -0.7, 2.2, 0.9];
        let params = spec_params(1.0, None, None);
        let target = nv_layers::sampler::distribution(&logits, &params);
        let n = 150_000u64;

        for drafted in 0..logits.len() as u32 {
            let mut s = ChatSampler::new(params, 99 + drafted as u64, None, vec![], false, 0);
            let accepts = (0..n)
                .filter(|_| matches!(s.accept_draft(&logits, drafted), DraftOutcome::Accept))
                .count() as f64
                / n as f64;
            assert!(
                (accepts - target[drafted as usize] as f64).abs() < 0.008,
                "drafted={drafted}: accept rate {accepts:.5} vs p {:.5}",
                target[drafted as usize]
            );
        }
    }

    #[test]
    fn spec_rejection_replacement_never_repeats_the_draft() {
        let logits = vec![1.0f32, 1.1, 0.9, 1.05];
        let params = spec_params(1.0, None, None);
        for drafted in 0..logits.len() as u32 {
            let mut s = ChatSampler::new(params, 5 + drafted as u64, None, vec![], false, 0);
            for _ in 0..5_000 {
                if let DraftOutcome::Reject(r) = s.accept_draft(&logits, drafted) {
                    assert_ne!(r, drafted, "residual must exclude the rejected draft");
                }
            }
        }
    }

    #[test]
    fn spec_rejection_respects_topk() {
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

    #[test]
    fn spec_greedy_gpu_shortcircuit_contract() {
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

    #[test]
    fn spec_greedy_reduces_to_argmax() {
        let logits = vec![0.1f32, 5.0, 0.2, -3.0];
        let params = spec_params(0.0, None, None);
        let mut s = ChatSampler::new(params, 123, None, vec![], false, 0);
        for _ in 0..100 {
            assert!(matches!(s.accept_draft(&logits, 1), DraftOutcome::Accept));
            match s.accept_draft(&logits, 0) {
                DraftOutcome::Reject(r) => assert_eq!(r, 1, "greedy replacement must be argmax"),
                DraftOutcome::Accept => panic!("greedy must reject non-argmax draft"),
            }
        }

        assert_eq!(s.draw_from_logits(&logits), 1);
    }

    #[test]
    fn repetition_penalty_seeds_from_prompt_tokens() {
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
    }

    #[test]
    fn logprobs_report_raw_untempered_distribution() {
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
    }

    #[test]
    fn logprobs_raw_even_with_bias_and_penalties() {
        let logits = vec![1.0f32, 2.0, 0.5, -1.0];
        let raw = nv_layers::sampler::logprobs_full(&logits, 1.0);
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

    #[test]
    fn seed_prompt_noop_without_repetition_penalty() {
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

    #[test]
    fn sampler_reproducible_with_seed() {
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

    #[test]
    fn spec_acceptance_is_exactly_leviathan_dirac_case_split() {
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
    }

    #[test]
    fn spec_emitted_law_equals_target_distribution_analytically() {
        let logits = vec![1.2f32, -0.4, 2.0, 0.1, -1.5, 0.7];
        for &(temp, top_p) in &[(0.8f32, None), (1.0, Some(0.85f32))] {
            let params = spec_params(temp, None, top_p);
            let probs = nv_layers::sampler::distribution(&logits, &params);
            for drafted in 0..logits.len() as u32 {
                let px = probs[drafted as usize] as f64;

                let grid = 200_000u32;
                let mut law = vec![0.0f64; logits.len()];
                for step in 0..grid {
                    let u2 = (step as f64 + 0.5) / grid as f64;
                    if let Some(r) =
                        nv_layers::sampler::residual_sample_checked(&probs, drafted, u2)
                    {
                        law[r as usize] += (1.0 - px) / grid as f64;
                    } else {
                        law[nv_layers::sampler::argmax(&logits) as usize] +=
                            (1.0 - px) / grid as f64;
                    }
                }
                law[drafted as usize] += px;
                for t in 0..logits.len() {
                    assert!(
                        (law[t] - probs[t] as f64).abs() < 2e-5,
                        "temp={temp} top_p={top_p:?} drafted={drafted} token {t}: \
                         law {} vs target {}",
                        law[t],
                        probs[t]
                    );
                }
            }
        }
    }

    #[test]
    fn sampler_top_k_filters_correctly() {
        let logits = vec![10.0_f32, -100.0, -100.0, 10.0, -100.0, -100.0];
        let mut rng = Pcg64::seed_from_u64(99);
        for _ in 0..500 {
            let tok = sample_logits(&logits, 1.0, Some(2), None, &mut rng);
            assert!(
                tok == 0 || tok == 3,
                "top_k=2 must restrict to {{0,3}}, got {tok}"
            );
        }
    }

    #[test]
    fn sampler_top_p_cumulative() {
        let logits = vec![5.0_f32, 4.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut rng = Pcg64::seed_from_u64(2024);
        for _ in 0..500 {
            let tok = sample_logits(&logits, 1.0, None, Some(0.5), &mut rng);
            assert!(
                tok == 0 || tok == 1,
                "top_p=0.5 must restrict to head, got {tok}"
            );
        }
    }

    #[test]
    fn sampler_temperature_softens_distribution() {
        let logits = vec![0.0_f32; 10];
        let mut rng = Pcg64::seed_from_u64(31337);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            seen.insert(sample_logits(&logits, 5.0, None, None, &mut rng));
            if seen.len() >= 3 {
                break;
            }
        }
        assert!(
            seen.len() >= 3,
            "high-temperature uniform sampler should explore (got {} unique ids)",
            seen.len()
        );
    }

    #[test]
    fn verify_cache_capacity_rounds_up_to_grain() {
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
        for needed in [1usize, 700, 2048, 2049, 100_000] {
            assert!(verify_cache_capacity(needed) >= needed);
        }
    }

    #[test]
    fn kv_window_invariants_hold_over_small_grid() {
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
    }

    #[test]
    fn verify_graph_reuse_requires_same_k_and_enough_capacity() {
        assert!(verify_graph_reusable(8, 4096, 8, 4096));
        assert!(verify_graph_reusable(8, 4096, 8, 700));
        assert!(!verify_graph_reusable(8, 4096, 8, 4097));
        assert!(!verify_graph_reusable(4, 4096, 8, 700));
    }

    #[test]
    fn verify_cache_capacity_never_exceeds_need_by_a_grain() {
        for needed in [1usize, 98, 700, 836, 1524, 4097, 100_000] {
            let cap = verify_cache_capacity(needed);
            assert!(cap >= needed);
            assert!(
                cap < needed + VERIFY_CACHE_GRAIN,
                "needed={needed} cap={cap} overshoots by a full grain"
            );
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
    fn rebuild_frees_the_old_verifier_before_allocating_the_new_one() {
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
    }

    #[test]
    fn reused_verifier_is_not_dropped_and_slot_is_emptied() {
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

    #[test]
    fn verify_slot_reuses_across_requests_with_at_most_one_live_verifier() {
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

    #[test]
    fn changing_k_rebuilds_and_releases_the_old_verifier_first() {
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

    #[test]
    fn kv_window_invariants_hold_at_usize_extremes() {
        let vals = [
            0usize,
            1,
            2,
            4096,
            usize::MAX / 2,
            usize::MAX - 2,
            usize::MAX - 1,
            usize::MAX,
        ];
        for &kv_max_seq_len in &vals {
            for &prompt_len in &vals {
                for &max_new in &vals {
                    assert_kv_window_invariants(prompt_len, max_new, kv_max_seq_len);
                }
            }
        }
    }

    #[test]
    fn kv_window_none_exactly_when_prompt_does_not_fit() {
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

    #[test]
    fn kv_window_clamps_oversized_request_to_capacity() {
        assert_eq!(kv_window(6, 1000, 8), Some((8, 1)));
        assert_eq!(kv_window(7, 1000, 8), Some((8, 0)));
        assert_eq!(kv_window(8, 1000, 8), None);
        assert_eq!(kv_window(2, 3, 4096), Some((6, 3)));
    }

    #[test]
    fn eagle3_k_clamps_env_into_range() {
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
    }

    #[test]
    fn dflash_k_defaults_and_clamps_to_block_size_plus_one() {
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

    #[test]
    fn nv_drafter_kind_defaults_to_eagle3() {
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
    }

    #[test]
    fn suffix_drafter_env_parsing() {
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

    #[test]
    fn codeish_prompt_heuristic() {
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
    }

    #[test]
    fn drafter_arm_routing_prefers_dflash_for_code_and_best_ema_otherwise() {
        assert_eq!(route_drafter_arm(true, 0.5, 3.0), DrafterArm::DFlash);
        assert_eq!(route_drafter_arm(false, 2.5, 2.0), DrafterArm::DFlash);
        assert_eq!(route_drafter_arm(false, 2.0, 2.0), DrafterArm::DFlash);
        assert_eq!(route_drafter_arm(false, 1.5, 2.0), DrafterArm::Eagle3);
    }

    #[test]
    fn classify_prompt_splits_code_and_prose() {
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

    #[test]
    fn resolve_drafter_arm_decision_table() {
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
                resolve_drafter_arm("auto", class, auto_gate - 1, auto_gate, true, true, 0.5, 9.0),
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
    fn route_arm_is_keyed_on_context_not_prompt_class() {
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

    #[test]
    fn route_ctx_gate_env_parsing() {
        assert_eq!(route_ctx_gate(None), ROUTE_CTX_GATE_DEFAULT);
        assert_eq!(route_ctx_gate(Some("")), ROUTE_CTX_GATE_DEFAULT);
        assert_eq!(route_ctx_gate(Some("junk")), ROUTE_CTX_GATE_DEFAULT);
        assert_eq!(route_ctx_gate(Some("0")), ROUTE_CTX_GATE_DEFAULT);
        assert_eq!(route_ctx_gate(Some("-1")), ROUTE_CTX_GATE_DEFAULT);
        assert_eq!(route_ctx_gate(Some("4096")), 4096);
        assert_eq!(route_ctx_gate(Some(" 3072 ")), 3072);
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
    fn dflash_k_is_the_single_k_policy_for_every_drafter_kind() {
        assert_eq!(dflash_k(None, 16), DFLASH_K_DEFAULT);
        assert_eq!(dflash_k(None, 8), DFLASH_K_DEFAULT);
        assert_eq!(dflash_k(Some("16"), 16), 16);
        assert_eq!(dflash_k(Some("0"), 16), EAGLE3_K_MIN);
    }

    #[test]
    fn dflash_prose_k_env_overrides_base_only_when_set() {
        assert_eq!(dflash_prose_k(None, 8, 8), 8);
        assert_eq!(dflash_prose_k(Some(""), 8, 8), 8);
        assert_eq!(dflash_prose_k(Some("junk"), 8, 8), 8);
        assert_eq!(dflash_prose_k(Some("4"), 8, 8), 4);
        assert_eq!(dflash_prose_k(Some(" 4 "), 8, 8), 4);
        assert_eq!(dflash_prose_k(Some("0"), 8, 8), EAGLE3_K_MIN);
        assert_eq!(dflash_prose_k(Some("64"), 8, 8), 9);
        assert_eq!(dflash_prose_k(Some("12"), 16, 16), 12);
    }

    #[test]
    fn arm_ema_defaults_and_moves() {
        assert_eq!(arm_ema_get(DrafterArm::Eagle3), ARM_EMA_DEFAULT);
        arm_ema_observe(DrafterArm::Eagle3, 4.0);
        let after = arm_ema_get(DrafterArm::Eagle3);
        assert!(after > ARM_EMA_DEFAULT && after < 4.0);
        arm_ema_observe(DrafterArm::Eagle3, f64::NAN);
        assert_eq!(arm_ema_get(DrafterArm::Eagle3), after);
    }

    #[test]
    fn arm_ema_step_seeds_default_and_blends() {
        let seeded = f64::from_bits(arm_ema_step(0, ARM_EMA_DEFAULT));
        assert_eq!(seeded, ARM_EMA_DEFAULT);
        let moved = f64::from_bits(arm_ema_step(ARM_EMA_DEFAULT.to_bits(), 4.0));
        assert_eq!(
            moved,
            (1.0 - ARM_EMA_ALPHA) * ARM_EMA_DEFAULT + ARM_EMA_ALPHA * 4.0
        );
    }

    #[test]
    fn adaptive_k_env_parsing() {
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

    #[test]
    fn adaptive_k_starts_at_init_and_stays_in_range() {
        let a = AdaptiveK::new(8, 3);
        assert_eq!(a.k_eff(), 3);
        let a = AdaptiveK::new(8, 64);
        assert_eq!(a.k_eff(), 8);
        let a = AdaptiveK::new(8, 1);
        assert_eq!(a.k_eff(), EAGLE3_K_MIN);
    }

    #[test]
    fn adaptive_k_high_acceptance_grows_toward_k_graph() {
        let mut a = AdaptiveK::new(8, 3);
        for _ in 0..200 {
            let offered = a.k_eff() - 1;
            a.observe(offered, offered, 1.2 * offered as f64, 30.0);
        }
        assert_eq!(a.k_eff(), 8, "p_ema={} should drive k to k_graph", a.p_ema);
    }

    #[test]
    fn adaptive_k_low_acceptance_shrinks_k() {
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

    #[test]
    fn adaptive_k_observe_clamps_and_ignores_bad_timings() {
        let mut a = AdaptiveK::new(8, 3);
        a.observe(2, 9, f64::NAN, -1.0);
        assert!(a.p_ema <= 1.0 && a.p_ema >= 0.0);
        assert!((a.d_graph_ms - 1.2).abs() < 1e-9);
        assert!((a.d_eager_ms - 2.4).abs() < 1e-9);
        assert!((a.verify_ms - 30.0).abs() < 1e-9);
        a.observe(0, 0, 5.0, 30.0);
        assert!(a.p_ema > 0.0);
    }

    #[test]
    fn adaptive_k_converges_to_cost_optimal_k() {
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
    }

    #[test]
    fn adaptive_k_hysteresis_keeps_current_on_near_ties() {
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

    #[test]
    fn spec_prefill_chunk_clamps_env_into_range() {
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

    #[test]
    fn eagle3_gate_matrix() {
        use super::Eagle3Gate::*;

        assert_eq!(eagle3_gate(false, false, false), NotRequested);
        assert_eq!(eagle3_gate(false, true, false), NotRequested);
        assert_eq!(eagle3_gate(false, false, true), NotRequested);
        assert_eq!(eagle3_gate(false, true, true), NotRequested);

        assert_eq!(eagle3_gate(true, false, true), Enabled);
        assert_eq!(eagle3_gate(true, true, true), Enabled);

        assert_eq!(eagle3_gate(true, false, false), DegradedWarn);
        assert_eq!(eagle3_gate(true, true, false), RequiredFail);
    }

    #[test]
    fn eagle3_required_parses_env_shapes() {
        assert!(!eagle3_required(None));
        assert!(!eagle3_required(Some("")));
        assert!(!eagle3_required(Some("0")));
        assert!(!eagle3_required(Some(" 0 ")));
        assert!(!eagle3_required(Some("false")));
        assert!(!eagle3_required(Some("FALSE")));
        assert!(eagle3_required(Some("1")));
        assert!(eagle3_required(Some("true")));
        assert!(eagle3_required(Some("yes")));
    }

    #[test]
    fn dflash_required_parses_the_same_env_shapes_as_eagle3() {
        assert!(!dflash_required(None));
        assert!(!dflash_required(Some("")));
        assert!(!dflash_required(Some("0")));
        assert!(!dflash_required(Some(" 0 ")));
        assert!(!dflash_required(Some("false")));
        assert!(!dflash_required(Some("FALSE")));
        assert!(dflash_required(Some("1")));
        assert!(dflash_required(Some("true")));
        assert!(dflash_required(Some("yes")));
    }

    #[test]
    fn drafter_wants_dflash_covers_every_kind_that_loads_one() {
        assert!(drafter_wants_dflash("dflash"));
        assert!(drafter_wants_dflash("auto"));
        assert!(drafter_wants_dflash("route"));
        assert!(!drafter_wants_dflash("eagle3"));
    }

    #[test]
    fn dflash_spec_requested_tracks_kind_and_no_spec() {
        assert!(dflash_spec_requested(false, "dflash"));
        assert!(dflash_spec_requested(false, "auto"));
        assert!(dflash_spec_requested(false, "route"));
        assert!(!dflash_spec_requested(false, "eagle3"));
        assert!(!dflash_spec_requested(true, "dflash"));
        assert!(!dflash_spec_requested(true, "auto"));
    }

    #[test]
    fn dflash_gate_matrix_mirrors_eagle3() {
        use super::Eagle3Gate::*;

        let gate = |kind: &str, required: bool, loaded: bool| {
            eagle3_gate(dflash_spec_requested(false, kind), required, loaded)
        };

        assert_eq!(gate("eagle3", true, false), NotRequested);
        assert_eq!(gate("eagle3", false, false), NotRequested);

        assert_eq!(gate("dflash", false, true), Enabled);
        assert_eq!(gate("dflash", true, true), Enabled);
        assert_eq!(gate("auto", true, true), Enabled);
        assert_eq!(gate("route", true, true), Enabled);

        assert_eq!(gate("dflash", false, false), DegradedWarn);
        assert_eq!(gate("dflash", true, false), RequiredFail);
        assert_eq!(gate("auto", true, false), RequiredFail);
        assert_eq!(gate("route", true, false), RequiredFail);

        assert_eq!(
            eagle3_gate(dflash_spec_requested(true, "dflash"), true, false),
            NotRequested
        );
    }

    #[test]
    fn env_flag_enabled_treats_zero_as_off_for_presence_flags() {
        assert!(!env_flag_enabled(None));
        assert!(!env_flag_enabled(Some("0")));
        assert!(env_flag_enabled(Some("1")));
        assert!(env_flag_enabled(Some("")));
        assert!(env_flag_enabled(Some("true")));
    }

    #[test]
    fn spec_defer_drafter_ships_default_on_and_zero_disables() {
        assert!(spec_defer_drafter_from(None));
        assert!(!spec_defer_drafter_from(Some("0")));
        assert!(spec_defer_drafter_from(Some("1")));
        assert!(spec_defer_drafter_from(Some("")));
    }

    #[test]
    fn eagle3_graph_chain_ships_default_on_and_zero_disables() {
        assert!(eagle3_graph_chain_from(None));
        assert!(!eagle3_graph_chain_from(Some("0")));
        assert!(eagle3_graph_chain_from(Some("1")));
        assert!(eagle3_graph_chain_from(Some("")));
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
    fn graph_decode_eligible_passes_batch_and_max_total_to_the_family() {
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

    #[test]
    fn graph_decode_eligible_rejects_empty_and_family_refusal() {
        let lens = |_: u64| Some(0usize);
        assert!(!graph_decode_eligible(&[], |_, _| true, lens));
        assert!(!graph_decode_eligible(
            &[seq_input(1, 0)],
            |_, _| false,
            lens,
        ));
    }

    #[test]
    fn graph_decode_eligible_requires_cache_len_position_lockstep_per_seq() {
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
    }

    #[test]
    fn nv_no_spec_parses_env_shapes() {
        assert!(!nv_no_spec(None));
        assert!(!nv_no_spec(Some("")));
        assert!(!nv_no_spec(Some("0")));
        assert!(!nv_no_spec(Some(" 0 ")));
        assert!(!nv_no_spec(Some("false")));
        assert!(!nv_no_spec(Some("FALSE")));
        assert!(nv_no_spec(Some("1")));
        assert!(nv_no_spec(Some("true")));
        assert!(nv_no_spec(Some("yes")));
    }

    #[test]
    fn spec_requested_defaults_on_when_drafter_dir_set() {
        assert!(spec_requested(false, false, true));
        assert!(spec_requested(false, true, false));
        assert!(spec_requested(false, true, true));
        assert!(!spec_requested(false, false, false));

        assert!(!spec_requested(true, false, false));
        assert!(!spec_requested(true, true, false));
        assert!(!spec_requested(true, false, true));
        assert!(!spec_requested(true, true, true));
    }

    #[test]
    fn spec_gate_for_request_matrix() {
        assert!(spec_gate_for_request(false, false, true));
        assert!(spec_gate_for_request(false, true, true));
        assert!(spec_gate_for_request(false, true, false));

        assert!(!spec_gate_for_request(false, false, false));

        assert!(!spec_gate_for_request(true, false, true));
        assert!(!spec_gate_for_request(true, true, true));
        assert!(!spec_gate_for_request(true, true, false));
        assert!(!spec_gate_for_request(true, false, false));
    }

    #[test]
    fn resolve_cond_mode_default_and_explicit_modes_with_drafter_kv() {
        assert_eq!(
            resolve_cond_mode(None, true),
            ("shift".into(), false, false)
        );
        assert_eq!(
            resolve_cond_mode(Some("shift"), true),
            ("shift".into(), false, false)
        );
        assert_eq!(
            resolve_cond_mode(Some("bonus"), true),
            ("bonus".into(), false, false)
        );
        assert_eq!(
            resolve_cond_mode(Some("legacy"), true),
            ("legacy".into(), false, false)
        );
        assert_eq!(resolve_cond_mode(Some(""), true), ("".into(), false, false));

        assert_eq!(
            resolve_cond_mode(Some("shift-force"), true),
            ("shift".into(), true, false)
        );
    }

    #[test]
    fn resolve_cond_mode_downgrades_cached_modes_without_drafter_kv() {
        assert_eq!(resolve_cond_mode(None, false), ("".into(), false, true));
        assert_eq!(
            resolve_cond_mode(Some("shift"), false),
            ("".into(), false, true)
        );
        assert_eq!(
            resolve_cond_mode(Some("bonus"), false),
            ("".into(), false, true)
        );

        assert_eq!(
            resolve_cond_mode(Some("legacy"), false),
            ("legacy".into(), false, false)
        );
        assert_eq!(
            resolve_cond_mode(Some(""), false),
            ("".into(), false, false)
        );
    }

    #[test]
    fn resolve_cond_mode_shift_force_bypasses_the_downgrade_guard() {
        assert_eq!(
            resolve_cond_mode(Some("shift-force"), false),
            ("shift".into(), true, false)
        );
    }

    #[test]
    fn spec_verify_cache_len_never_overflows_or_undersizes() {
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
    }

    #[test]
    fn spec_verify_cache_len_covers_every_speculative_row() {
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

    #[test]
    fn spec_verify_cache_len_matches_the_legacy_formula_when_it_was_safe() {
        for (prompt_len, max_new, k, kv_max_seq_len) in [
            (300usize, 512usize, 8usize, 4096usize),
            (1, 1, 2, 4096),
            (2048, 1024, 16, 8192),
        ] {
            let legacy = prompt_len + max_new + k + 16;
            assert_eq!(
                spec_verify_cache_len(prompt_len, max_new, k, kv_max_seq_len),
                Some(legacy + 1),
                "prompt_len={prompt_len} max_new={max_new} k={k}"
            );
        }
    }

    #[test]
    fn spec_verify_cache_len_rejects_the_overflow_that_the_legacy_formula_wrapped() {
        let prompt_len = usize::MAX / 2;
        let max_new = usize::MAX / 2;
        let k = EAGLE3_K_MAX;
        let kv_max_seq_len = usize::MAX;
        let legacy = prompt_len
            .wrapping_add(max_new)
            .wrapping_add(k)
            .wrapping_add(SPEC_VERIFY_HEADROOM);
        assert!(
            legacy < prompt_len,
            "legacy formula was expected to wrap here, got {legacy}"
        );
        assert_eq!(
            spec_verify_cache_len(prompt_len, max_new, k, kv_max_seq_len),
            None
        );
    }

    #[test]
    fn spec_verify_window_allocation_covers_every_speculative_row() {
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
    }

    #[test]
    fn legacy_spec_verify_cache_len_oversizes_past_the_capped_allocation() {
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

    #[test]
    fn spec_verify_window_never_sizes_past_the_kv_window() {
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

    #[test]
    fn spec_verify_window_leaves_the_common_case_unclamped() {
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

    #[test]
    fn spec_verify_cache_len_refuses_when_the_prompt_does_not_fit() {
        assert_eq!(spec_verify_cache_len(8, 1, 8, 8), None);
        assert_eq!(spec_verify_cache_len(9, 1, 8, 8), None);
        assert_eq!(spec_verify_cache_len(0, 1, 8, 0), None);
    }

    #[test]
    fn verify_cache_alloc_covers_the_full_window_clamp() {
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

    #[test]
    fn short_suffix_proposals_pad_with_the_last_token_up_to_k_minus_one() {
        assert_eq!(pad_suffix_draft(vec![7], 4, 99), vec![7, 7, 7]);
        assert_eq!(pad_suffix_draft(vec![5, 6], 4, 99), vec![5, 6, 6]);
        assert_eq!(pad_suffix_draft(vec![1, 2, 3], 4, 99), vec![1, 2, 3]);
        assert_eq!(pad_suffix_draft(vec![1, 2, 3, 4], 4, 99), vec![1, 2, 3, 4]);
        assert_eq!(pad_suffix_draft(Vec::new(), 4, 99), vec![99, 99, 99]);
        assert_eq!(pad_suffix_draft(vec![8], 2, 99), vec![8]);
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
    }
}
