#![allow(dead_code)]

#[path = "../../../nv-models/tests/official_template/mod.rs"]
pub mod official_template;

use anyhow::Result;
use candle_core::{Device, Tensor};
use nv_models::gemma4::LayerType;
use nv_models::gemma4_moe::Gemma4MoeConfig;
use nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu;
use nv_specdecode::chain::{accept_prefix_argmax, build_chain_batch};
use nv_specdecode::gemma4_assistant::{
    AssistantLayerType, MoeAssistantHost, MoeSharedKvEngineTap, MoeSharedKvGeometry,
};
use official_template::OfficialTemplate;
use std::path::PathBuf;

pub const RETOKENIZATION_DRIFT_PAD_128_KEEPS_THE_CHAT_WRAP_AT_OR_ABOVE_THE_RUNG: usize = 128;

pub fn gate_loud(key: &str) {
    if std::env::var(key).ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {key}=1");
    }
}

pub fn env_dir(key: &str) -> PathBuf {
    let v = std::env::var(key)
        .unwrap_or_else(|_| panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {key}"));
    let p = PathBuf::from(v);
    assert!(p.is_dir(), "{key} is not a directory: {}", p.display());
    p
}

pub fn env_usize(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(v) if !v.is_empty() => v
            .parse()
            .unwrap_or_else(|_| panic!("{key} must be a plain integer, got {v:?}")),
        _ => default,
    }
}

pub fn trunk_snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NV_G4MOE_SNAPSHOT") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("g4moe snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file() && p.join("tokenizer.json").is_file())
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no complete gemma-4-26B-A4B-it snapshot under HOME hub; set NV_G4MOE_SNAPSHOT")
}

pub fn force_the_nonflash_decode_graph_verify_chain_refuses_the_flash_default() {
    std::env::set_var("NV_G4MOE_FLASH_DECODE", "0");
}

pub fn build_engine(dir: &PathBuf, max_seq: usize) -> Gemma4MoeWgpu {
    let config =
        Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("trunk config.json");
    assert!(
        max_seq <= config.base.max_position_embeddings,
        "max_seq {max_seq} beyond max_position_embeddings {}",
        config.base.max_position_embeddings
    );
    let t0 = std::time::Instant::now();
    let loader =
        nv_weights::WeightLoader::open_dir(dir, &Device::Cpu).expect("open trunk safetensors");
    let m = Gemma4MoeWgpu::from_loader(config, &loader, max_seq).expect("build Gemma4MoeWgpu");
    eprintln!(
        "[G4WGPU-BUILD] built in {:.1}s wired={:.2} GiB prefill_m={} verify_rows={}",
        t0.elapsed().as_secs_f64(),
        m.load_report().wired_bytes as f64 / (1u64 << 30) as f64,
        m.prefill_chunk_len(),
        m.verify_max_rows()
    );
    m
}

pub fn drafter_device() -> Device {
    match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[G4WGPU-SETUP] no cuda device for the drafter ({e}); candle CPU");
            Device::Cpu
        }
    }
}

pub struct EngineTap<'a>(pub &'a Gemma4MoeWgpu);

fn moe_layer_type_for(lt: AssistantLayerType) -> LayerType {
    match lt {
        AssistantLayerType::Sliding => LayerType::SlidingAttention,
        AssistantLayerType::Full => LayerType::FullAttention,
    }
}

impl MoeSharedKvEngineTap for EngineTap<'_> {
    fn committed_kv_rows(&self) -> usize {
        self.0.current_pos()
    }

    fn shared_kv_geometry(&self, lt: AssistantLayerType) -> Result<MoeSharedKvGeometry> {
        let g = self.0.assistant_kv_geometry(moe_layer_type_for(lt))?;
        Ok(MoeSharedKvGeometry {
            n_kv: g.n_kv,
            head_dim: g.head_dim,
        })
    }

    fn shared_kv_rows_f32_layout_len_by_n_kv_by_head_dim(
        &self,
        lt: AssistantLayerType,
        start: usize,
        len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let g = self.0.assistant_kv_geometry(moe_layer_type_for(lt))?;
        let (k, v) = self
            .0
            .assistant_kv_rows_f32_layout_len_by_n_kv_by_head_dim(
                moe_layer_type_for(lt),
                start,
                len,
            )?;
        Ok((
            Tensor::from_vec(k, (len, g.n_kv, g.head_dim), &Device::Cpu)?,
            Tensor::from_vec(v, (len, g.n_kv, g.head_dim), &Device::Cpu)?,
        ))
    }
}

pub fn load_host(m: &Gemma4MoeWgpu, trunk_dir: &PathBuf, assistant_dir: &PathBuf) -> MoeAssistantHost {
    let device = drafter_device();
    let host = MoeAssistantHost::load(
        assistant_dir,
        trunk_dir,
        m.config().base.hidden_size,
        m.max_seq(),
        &device,
    )
    .expect("load MoeAssistantHost");
    let acfg = host.config();
    assert_eq!(
        acfg.head_dim,
        m.config().base.head_dim_for(LayerType::SlidingAttention)
    );
    assert_eq!(
        acfg.global_head_dim,
        m.config().base.head_dim_for(LayerType::FullAttention)
    );
    host
}

pub struct Primed {
    pub anchor: u32,
    pub hidden: Vec<f32>,
    pub prefill_s: f64,
}

pub fn prime_prompt(m: &mut Gemma4MoeWgpu, prompt: &[u32]) -> Result<Primed> {
    anyhow::ensure!(prompt.len() >= 2, "prompt too short to prime");
    m.reset()?;
    let t0 = std::time::Instant::now();
    let head = &prompt[..prompt.len() - 1];
    let done = m.prefill_tokens(head)?;
    for t in &head[done..] {
        m.prefill_step(*t)?;
    }
    let anchor = m.decode_step(prompt[prompt.len() - 1])?;
    anyhow::ensure!(m.current_pos() == prompt.len(), "prime landed off-depth");
    let hidden = m.decode_prenorm_hidden_row_the_gemma4_assistant_drafter_consumes()?;
    Ok(Primed {
        anchor,
        hidden,
        prefill_s: t0.elapsed().as_secs_f64(),
    })
}

pub struct SpecRoundOut {
    pub submitted: usize,
    pub accepted: usize,
    pub emitted: Vec<u32>,
    pub draft_s: f64,
    pub verify_s: f64,
}

pub fn spec_round(
    m: &mut Gemma4MoeWgpu,
    host: &mut MoeAssistantHost,
    anchor: &mut u32,
    hidden: &mut Vec<f32>,
    k: usize,
) -> Result<SpecRoundOut> {
    let rows = m.verify_max_rows();
    anyhow::ensure!(k + 1 <= rows, "k {k} needs {} verify rows, have {rows}", k + 1);
    let td = std::time::Instant::now();
    let mut draft = host.propose_from_engine(&EngineTap(m), *anchor, hidden, k)?;
    let draft_s = td.elapsed().as_secs_f64();
    draft.truncate(rows - 1);
    anyhow::ensure!(!draft.is_empty(), "drafter proposed nothing");
    let tv = std::time::Instant::now();
    let batch = build_chain_batch(*anchor, &draft, draft.len() + 1, true)?;
    let amax = m.verify_chain(&batch)?;
    let acc = accept_prefix_argmax(&batch, &amax)?;
    m.advance(acc.commit_len)?;
    let mut emitted = Vec::with_capacity(acc.commit_len);
    emitted.extend_from_slice(&draft[..acc.commit_len - 1]);
    emitted.push(acc.next_bonus);
    *hidden =
        m.verify_prenorm_hidden_row_the_gemma4_assistant_drafter_consumes(acc.commit_len - 1)?;
    *anchor = acc.next_bonus;
    let verify_s = tv.elapsed().as_secs_f64();
    Ok(SpecRoundOut {
        submitted: draft.len(),
        accepted: acc.draft_accepted,
        emitted,
        draft_s,
        verify_s,
    })
}

pub fn corpus_ids(tok: &tokenizers::Tokenizer) -> Vec<u32> {
    let corpus = std::env::var("NV_G4DRAFT_CORPUS").unwrap_or_else(|_| {
        panic!(
            "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_G4DRAFT_CORPUS to a raw \
             text file; a tiled paragraph reads as an upper bracket, not an acceptance number"
        )
    });
    let text = std::fs::read_to_string(&corpus).expect("corpus file");
    tok.encode(text.as_str(), false)
        .expect("encode corpus")
        .get_ids()
        .to_vec()
}

pub fn chat_wrapped_prompt(
    tmpl: &OfficialTemplate,
    tok: &tokenizers::Tokenizer,
    corpus_ids: &[u32],
    offset: usize,
    depth: usize,
) -> Vec<u32> {
    let body = tok
        .decode(
            &corpus_ids[offset
                ..offset + depth + RETOKENIZATION_DRIFT_PAD_128_KEEPS_THE_CHAT_WRAP_AT_OR_ABOVE_THE_RUNG],
            true,
        )
        .expect("decode corpus slice");
    let rendered = tmpl.render_user(&format!(
        "Continue this document, staying in its style:\n\n{body}"
    ));
    let ids: Vec<u32> = tok
        .encode(rendered.as_str(), false)
        .expect("encode rendered prompt")
        .get_ids()
        .to_vec();
    assert!(
        ids.len() >= depth,
        "chat wrap produced {} tokens for a {depth} rung; the template cannot shrink its body",
        ids.len()
    );
    ids
}

pub fn assistant_ckpt_dir() -> PathBuf {
    env_dir("NV_G4DRAFT_ASSISTANT_DIR")
}

pub fn trunk_tokenizer(dir: &PathBuf) -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("trunk tokenizer.json")
}

pub fn median_ms(samples: &mut Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}
