#![allow(dead_code)]
#![allow(unused_imports)]

#[cfg(feature = "cuda")]
use minijinja::{context, Environment, Value as JinjaValue};
#[cfg(feature = "cuda")]
use std::path::Path;
#[cfg(feature = "wgpu")]
use nv_specdecode::wgpu_spec::SpecDims;
#[cfg(feature = "cuda")]
use nv_specdecode::qwen38_mtp::{Qwen38DenseMtpHead, Qwen38MtpGraphedDecodeSession};

#[cfg(feature = "cuda")]
pub fn raise_exception(msg: String) -> Result<JinjaValue, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

#[cfg(feature = "cuda")]
pub fn strftime_now(_fmt: String) -> Result<JinjaValue, minijinja::Error> {
    Ok(JinjaValue::from("1970-01-01"))
}

#[cfg(feature = "cuda")]
pub fn render_real_chat_template_no_thinking(dir: &Path, question: &str) -> String {
    let source = std::fs::read_to_string(dir.join("chat_template.jinja"))
        .expect("chat_template.jinja must ship with the checkpoint");
    let mut env = Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", raise_exception);
    env.add_function("strftime_now", strftime_now);
    env.add_template_owned("chat", source).expect("template compiles");
    env.get_template("chat")
        .unwrap()
        .render(context! {
            messages => vec![serde_json::json!({"role": "user", "content": question})],
            add_generation_prompt => true,
            enable_thinking => false,
        })
        .expect("the official template must render a plain user turn")
}

#[cfg(feature = "cuda")]
pub struct GraphedRunRecord {
    pub ids: Vec<u32>,
    pub stats: nv_models::qwen3_5_mtp::MtpSpecStats,
    pub round_emitted: Vec<usize>,
    pub round_wall_ms: Vec<f64>,
}

#[cfg(feature = "cuda")]
pub fn generate_graphed(
    engine: &mut nv_models::graph_engine::GraphedQwen3Moe,
    mtp: &Qwen38DenseMtpHead,
    k: usize,
    prompt: &[u32],
    max_new: usize,
    stop: &[u32],
    junk_drafts: Option<u32>,
) -> GraphedRunRecord {
    let mut session = Qwen38MtpGraphedDecodeSession::start(engine, mtp, k, prompt)
        .expect("graphed mtp session start");
    let junk: Option<Vec<u32>> = junk_drafts.map(|t| vec![t; k]);
    let anchor = session.anchor();
    let mut ids: Vec<u32> = vec![anchor];
    let mut round_emitted: Vec<usize> = Vec::new();
    let mut round_wall_ms: Vec<f64> = Vec::new();
    if stop.contains(&anchor) {
        return GraphedRunRecord {
            ids,
            stats: session.stats,
            round_emitted,
            round_wall_ms,
        };
    }
    'rounds: while ids.len() < max_new && session.round_fits() {
        let t0 = std::time::Instant::now();
        let emitted = match &junk {
            Some(j) => session
                .round_with_drafts_from_a_clairvoyant_test_oracle(j)
                .expect("graphed junk round"),
            None => session.round().expect("graphed round"),
        };
        round_wall_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        let mut taken = 0usize;
        for &t in emitted.iter() {
            ids.push(t);
            taken += 1;
            if stop.contains(&t) {
                round_emitted.push(taken);
                break 'rounds;
            }
            if ids.len() >= max_new {
                round_emitted.push(taken);
                break 'rounds;
            }
        }
        round_emitted.push(taken);
    }
    GraphedRunRecord {
        ids,
        stats: session.stats,
        round_emitted,
        round_wall_ms,
    }
}

#[cfg(feature = "wgpu")]
pub fn spec_dims_tiny() -> SpecDims {
    SpecDims {
        h: 32,
        nh: 4,
        nkv: 2,
        hd: 8,
        inter: 64,
        vocab: 96,
        max_seq: 192,
        eps: 1e-5,
        rope_theta: 10000.0,
    }
}
