#![recursion_limit = "256"]

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use serde_json::{json, Value};

const NO_TRAINER_EXIT_4_IS_THE_NIGHTLY_CONTRACT_FOR_A_BASE_THIS_TRAINER_CANNOT_TRAIN: i32 = 4;

#[derive(Parser)]
#[command(name = "nv-train-nvfp4", about = "NVFP4-native LoRA for Qwen3.8-Flash-Next (qwen4_exp): train against the served forward, evaluate and generate through the CUDA engine's runtime LoRA")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Lora(LoraCmd),
    Eval(EvalCmd),
    Generate(GenCmd),
    BenchLayer(BenchCmd),
    Fixture(FixtureCmd),
}

#[derive(Parser)]
struct FixtureCmd {
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 0x0004_e7a0)]
    seed: u64,
}

#[cfg(feature = "cuda")]
mod fixture {
    pub fn write(dir: &std::path::Path, seed: u64) -> anyhow::Result<serde_json::Value> {
        std::fs::create_dir_all(dir)?;
        use nv_models::qwen4_exp::lora::tiny;
        let mut cfg: serde_json::Value = serde_json::from_str(&tiny::config())?;
        cfg["model_type"] = serde_json::json!("qwen4_exp");
        cfg["architectures"] = serde_json::json!(["Qwen4ExpForConditionalGeneration"]);
        std::fs::write(dir.join("config.json"), cfg.to_string())?;
        candle_core::safetensors::save(&tiny::weights(seed), dir.join("model.safetensors"))?;
        let vocab: serde_json::Map<String, serde_json::Value> = (0..tiny::vocab()).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
        let tok = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
            "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false},
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "t0"}
        });
        std::fs::write(dir.join("tokenizer.json"), tok.to_string())?;
        Ok(serde_json::json!({"fixture": "qwen4_exp", "dir": dir.display().to_string(), "vocab": tiny::vocab(), "eos": "t1"}))
    }
}

#[derive(Parser)]
struct LoraCmd {
    #[arg(long)]
    base: PathBuf,
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 400)]
    steps: usize,
    #[arg(long, default_value_t = 20261011)]
    seed: u64,
    #[arg(long)]
    metrics: Option<PathBuf>,
    #[arg(long, default_value_t = 8)]
    rank: usize,
    #[arg(long)]
    alpha: Option<f64>,
    #[arg(long, default_value = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")]
    targets: String,
    #[arg(long, default_value_t = 6)]
    window: usize,
    #[arg(long, default_value_t = 1e-4)]
    lr: f64,
    #[arg(long, default_value_t = 2048)]
    max_seq: usize,
    #[arg(long, default_value_t = 6.0)]
    vram_reserve_gib: f64,
    #[arg(long, default_value_t = 4.0)]
    prefix_cache_gib: f64,
    #[arg(long)]
    store_gcroot: Option<PathBuf>,
    #[arg(long, default_value = "expert-a")]
    expert_lora: String,
    #[arg(long, default_value_t = 16)]
    expert_rank: usize,
    #[arg(long)]
    expert_alpha: Option<f64>,
    #[arg(long, default_value_t = 6)]
    expert_layers: usize,
    #[arg(long, default_value_t = 0.0)]
    host_stacks_gib: f64,
    #[arg(long, default_value_t = false)]
    host_optimizer: bool,
}

#[derive(Parser)]
struct EvalCmd {
    #[arg(long)]
    base: PathBuf,
    #[arg(long)]
    adapter: PathBuf,
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 2048)]
    max_seq: usize,
}

#[derive(Parser)]
struct GenCmd {
    #[arg(long)]
    base: PathBuf,
    #[arg(long)]
    adapter: Option<PathBuf>,
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 0)]
    limit: usize,
    #[arg(long, default_value_t = 256)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 2048)]
    max_seq: usize,
}

#[derive(Parser)]
struct BenchCmd {
    #[arg(long)]
    base: PathBuf,
    #[arg(long, default_value_t = 22)]
    layer: usize,
    #[arg(long, default_value_t = 512)]
    tokens: usize,
    #[arg(long, default_value_t = nv_bench_experts_default())]
    experts: usize,
}

fn nv_bench_experts_default() -> usize {
    128
}

fn main() {
    let cli = Cli::parse();
    match run(cli) {
        Ok(v) => println!("{v}"),
        Err(e) if e.downcast_ref::<NoTrainer>().is_some() => {
            println!("{}", e.downcast_ref::<NoTrainer>().unwrap().0);
            std::process::exit(NO_TRAINER_EXIT_4_IS_THE_NIGHTLY_CONTRACT_FOR_A_BASE_THIS_TRAINER_CANNOT_TRAIN);
        }
        Err(e) => {
            eprintln!("nv-train-nvfp4: {e:#}");
            println!("{}", json!({"error": format!("{e:#}")}));
            std::process::exit(1);
        }
    }
}

#[derive(Debug)]
struct NoTrainer(Value);

impl std::fmt::Display for NoTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for NoTrainer {}

#[cfg(not(feature = "cuda"))]
fn run(cli: Cli) -> Result<Value> {
    let base = match &cli.cmd {
        Cmd::Lora(a) => a.base.clone(),
        Cmd::Eval(a) => a.base.clone(),
        Cmd::Generate(a) => a.base.clone(),
        Cmd::BenchLayer(a) => a.base.clone(),
        Cmd::Fixture(a) => a.out.clone(),
    };
    Err(NoTrainer(json!({"error": "no_trainer", "reason": format!("this build has no cuda feature; {} needs the CUDA engine", base.display())})).into())
}

#[cfg(feature = "cuda")]
fn run(cli: Cli) -> Result<Value> {
    match cli.cmd {
        Cmd::Lora(a) => cuda::lora(a),
        Cmd::Eval(a) => cuda::eval(a),
        Cmd::Generate(a) => cuda::generate(a),
        Cmd::BenchLayer(a) => cuda::bench_layer(a),
        Cmd::Fixture(a) => fixture::write(&a.out, a.seed),
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use anyhow::Context;
    use std::io::Write as _;
    use std::path::Path;
    use candle_core::Device;
    use nv_models::qwen4_exp::lora::expert::ExpertLoraForm;
    use nv_models::qwen4_exp::lora::{served_row_ce, ExpertLoraArgs, LoraTrainArgs, Qwen4ExpLoraTrainer, TokenRow};
    use nv_models::qwen4_exp::Qwen4Exp;
    use nv_specdecode::lora_spec::Qwen4ExpServingAdapter;
    use speaches_plus::oapi::chat_template::ChatTemplate;
    use std::sync::Arc;

    fn require_qwen4_exp(base: &Path) -> Result<()> {
        let raw = std::fs::read_to_string(base.join("config.json")).with_context(|| format!("read {}/config.json", base.display()))?;
        let v: Value = serde_json::from_str(&raw)?;
        let mt = v.get("model_type").or_else(|| v.get("text_config").and_then(|t| t.get("model_type"))).and_then(|x| x.as_str()).unwrap_or("");
        if mt != "qwen4_exp" && mt != "qwen4_exp_text" {
            return Err(NoTrainer(json!({"error": "no_trainer", "model_type": mt, "reason": "nv-train-nvfp4 trains qwen4_exp (Qwen3.8-Flash-Next) bases only"})).into());
        }
        Ok(())
    }

    fn read_rows(path: &Path) -> Result<Vec<Value>> {
        let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        raw.lines().filter(|l| !l.trim().is_empty()).enumerate().map(|(i, l)| serde_json::from_str(l).with_context(|| format!("{}:{}", path.display(), i + 1))).collect()
    }

    fn sha256_file(path: &Path) -> Result<String> {
        use sha2::Digest as _;
        Ok(hex::encode(sha2::Sha256::digest(std::fs::read(path)?)))
    }

    fn write_text(path: &Path, text: &str) -> Result<()> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let mut f = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
        f.write_all(text.as_bytes())?;
        Ok(())
    }

    pub const MTP_IS_LEFT_UNLOADED_BECAUSE_THE_ADAPTER_NEVER_SERVES_SPECULATIVELY: &str = "NV_Q4EXP_MTP";
    pub const PLE_TABLE_STAYS_IN_THE_MMAP_BECAUSE_PINNING_48_GB_OF_HOST_RAM_DOES_NOT_FIT_BESIDE_TRAINING: &str = "NV_Q4EXP_PLE_HOST";
    pub const THE_TOKEN_TABLE_STAYS_IN_PINNED_HOST_RAM_SO_TRAINING_NEVER_SPENDS_VRAM_ON_IT: &str = "NV_Q4EXP_EMBED_DEVICE";

    struct Base {
        model: Qwen4Exp,
        tokenizer: tokenizers::Tokenizer,
        template: Option<Arc<ChatTemplate>>,
        eos: Vec<u32>,
        device: Device,
    }

    fn open(base: &Path) -> Result<Base> {
        require_qwen4_exp(base)?;
        unsafe { std::env::set_var(MTP_IS_LEFT_UNLOADED_BECAUSE_THE_ADAPTER_NEVER_SERVES_SPECULATIVELY, "0") };
        if std::env::var_os(PLE_TABLE_STAYS_IN_THE_MMAP_BECAUSE_PINNING_48_GB_OF_HOST_RAM_DOES_NOT_FIT_BESIDE_TRAINING).is_none() {
            unsafe { std::env::set_var(PLE_TABLE_STAYS_IN_THE_MMAP_BECAUSE_PINNING_48_GB_OF_HOST_RAM_DOES_NOT_FIT_BESIDE_TRAINING, "1") };
        }
        unsafe { std::env::remove_var(THE_TOKEN_TABLE_STAYS_IN_PINNED_HOST_RAM_SO_TRAINING_NEVER_SPENDS_VRAM_ON_IT) };
        let device = Device::new_cuda_with_stream(0)?;
        let model = Qwen4Exp::load(base, &device)?;
        anyhow::ensure!(
            model.table_placement()["token_embedding"] != "device" || model.config().tie_word_embeddings,
            "the token table landed in VRAM; the trainer keeps it in pinned host RAM like the engine"
        );
        model.set_base_identity(nv_models::qwen4_exp::lora::expert::base_identity_for_dir(base)?);
        let tokenizer = tokenizers::Tokenizer::from_file(base.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let mut eos = vec![model.config().eos_token_id];
        if let Ok(raw) = std::fs::read_to_string(base.join("generation_config.json")) {
            let v: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
            match v.get("eos_token_id") {
                Some(Value::Array(a)) => eos.extend(a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32)),
                Some(Value::Number(n)) => eos.extend(n.as_u64().map(|x| x as u32)),
                _ => {}
            }
        }
        eos.extend(tokenizer.token_to_id("<|im_end|>"));
        eos.sort_unstable();
        eos.dedup();
        Ok(Base { model, tokenizer, template: ChatTemplate::load(base), eos, device })
    }

    fn fallback_prompt(messages: &Value) -> String {
        let mut out = String::new();
        for m in messages.as_array().into_iter().flatten() {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = match m.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => String::new(),
            };
            out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        out + "<|im_start|>assistant\n"
    }

    fn prompt_and_completion(b: &Base, row: &Value) -> Result<(String, Option<String>)> {
        let completion = row.get("completion").and_then(|c| c.as_str()).map(str::to_string);
        if let Some(p) = row.get("prompt").and_then(|p| p.as_str()) {
            return Ok((p.to_string(), completion.map(|c| c + "<|im_end|>")));
        }
        if let Some(t) = row.get("text").and_then(|t| t.as_str()) {
            return Ok((String::new(), Some(t.to_string())));
        }
        let messages = row.get("messages").cloned().context("row has none of prompt | text | messages")?;
        let Some(t) = &b.template else {
            return Ok((fallback_prompt(&messages), completion.map(|c| c + "<|im_end|>")));
        };
        let prompt = t.render(&messages, None, true).unwrap_or_else(|_| fallback_prompt(&messages));
        let Some(c) = completion else {
            return Ok((prompt, None));
        };
        let mut full = messages.as_array().cloned().unwrap_or_default();
        full.push(json!({"role": "assistant", "content": c}));
        match t.render(&Value::Array(full), None, false) {
            Ok(f) if f.starts_with(&prompt) && f.len() > prompt.len() => Ok((prompt.clone(), Some(f[prompt.len()..].trim_end_matches('\n').to_string()))),
            _ => Ok((prompt, Some(c + "<|im_end|>"))),
        }
    }

    fn encode(b: &Base, s: &str) -> Result<Vec<u32>> {
        Ok(b.tokenizer.encode(s, false).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?.get_ids().to_vec())
    }

    fn token_row(b: &Base, row: &Value, max_seq: usize) -> Result<TokenRow> {
        let (prompt, completion) = prompt_and_completion(b, row)?;
        let mut p = encode(b, &prompt)?;
        let mut c = encode(b, &completion.context("row has no completion to train or score")?)?;
        if p.len() + c.len() > max_seq {
            c.truncate((max_seq / 2).max(1));
            let keep = max_seq.saturating_sub(c.len());
            p = p[p.len().saturating_sub(keep)..].to_vec();
        }
        let completion_start = p.len().max(1);
        let mut ids = p;
        ids.extend(c);
        anyhow::ensure!(ids.len() >= 2 && completion_start < ids.len(), "row tokenizes to no trainable completion");
        Ok(TokenRow { ids, completion_start })
    }

    fn vram_used_gib() -> f64 {
        nv_models::qwen4_exp::vram_used_gib()
    }

    pub fn lora(a: LoraCmd) -> Result<Value> {
        let b = open(&a.base)?;
        let rows: Vec<TokenRow> = read_rows(&a.data)?.iter().filter_map(|r| token_row(&b, r, a.max_seq).ok()).collect();
        anyhow::ensure!(!rows.is_empty(), "{} holds no trainable row", a.data.display());
        let targets: Vec<String> = a.targets.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty() && s != "none").collect();
        let alpha = a.alpha.unwrap_or(2.0 * a.rank as f64);
        let form = ExpertLoraForm::parse(&a.expert_lora)?;
        let expert = ExpertLoraArgs { form, rank: a.expert_rank, alpha: a.expert_alpha.unwrap_or(2.0 * a.expert_rank as f64), layers: a.expert_layers };
        let args = LoraTrainArgs {
            rank: a.rank,
            alpha,
            targets: targets.clone(),
            window: a.window,
            steps: a.steps,
            lr: a.lr,
            seed: a.seed,
            clip: 1.0,
            vram_reserve_gib: a.vram_reserve_gib,
            prefix_cache_gib: a.prefix_cache_gib,
            expert: expert.clone(),
            host_stacks_gib: a.host_stacks_gib,
            host_optimizer: a.host_optimizer,
        };
        let loaded_gib = vram_used_gib();
        let mut tr = Qwen4ExpLoraTrainer::new(&b.model, args, a.max_seq)?;
        let t0 = std::time::Instant::now();
        let mut peak = loaded_gib;
        let recs = tr.train(&rows, |step, r| {
            peak = peak.max(vram_used_gib());
            if step % 10 == 0 {
                eprintln!("[nv-train-nvfp4] step {step} row {} tokens {} served-loss {:.4} twin-loss {:.4} grad-norm {:.3e} {:.0} ms", r.row, r.tokens, r.exact_loss, r.twin_loss, r.grad_norm, r.backward_ms);
            }
        })?;
        let secs = t0.elapsed().as_secs_f64();
        let base_name = a.base.display().to_string();
        let entries = tr.export_for_base(&a.out, &base_name, b.model.base_identity())?;
        let k = (recs.len() / 10).max(1);
        let mean = |f: &dyn Fn(&nv_models::qwen4_exp::lora::StepRecord) -> f32, r: &[nv_models::qwen4_exp::lora::StepRecord]| r.iter().map(f).sum::<f32>() / r.len().max(1) as f32;
        let tokens: usize = recs.iter().map(|r| r.tokens).sum();
        let mut m = json!({
            "trainer": "nvfp4", "base": base_name, "adapter": a.out.display().to_string(), "rows": rows.len(), "steps": recs.len(),
            "rank": a.rank, "alpha": alpha, "exported_alpha": a.rank, "targets": targets, "window": tr.window(),
            "first_trainable_layer": tr.first_trainable, "modules": entries.len(), "trainable_params": tr.trainable_params(),
            "seed": a.seed, "lr": a.lr, "max_seq": a.max_seq,
            "loss_first": mean(&|r| r.exact_loss, &recs[..k]), "loss_last": mean(&|r| r.exact_loss, &recs[recs.len() - k..]),
            "twin_loss_first": mean(&|r| r.twin_loss, &recs[..k]), "twin_loss_last": mean(&|r| r.twin_loss, &recs[recs.len() - k..]),
            "nonfinite_steps": recs.iter().filter(|r| r.skipped_nonfinite).count(),
            "seconds": (secs * 10.0).round() / 10.0, "tokens": tokens, "tokens_per_s": tokens as f64 / secs.max(1e-9),
            "stack_build_s": tr.stack_build_ms / 1000.0, "expert_stacks_resident": tr.stacks_resident(),
            "vram_gib_after_load": loaded_gib, "vram_gib_peak_sampled": peak,
            "data_sha256": sha256_file(&a.data)?,
            "expert_lora": form.as_str(), "expert_rank": expert.rank, "expert_alpha": expert.alpha,
            "expert_layers": tr.experts.iter().map(|e| e.layer).collect::<Vec<_>>(), "expert_params": tr.expert_params(),
            "base_identity": b.model.base_identity(),
            "host_stacks_gib": a.host_stacks_gib, "stacks_parked_on_host": tr.stacks_parked_on_host(),
            "host_stack_bytes": tr.host_stack_bytes(), "stack_uploads": tr.stack_uploads,
            "host_optimizer": a.host_optimizer, "optimizer_device_bytes": tr.optimizer_device_bytes(), "optimizer_host_bytes": tr.optimizer_host_bytes(),
            "optimizer_ms_mean": recs.iter().map(|r| r.optimizer_ms).sum::<f64>() / recs.len().max(1) as f64,
            "pinned_host_bytes": nv_layers::pinned_host::pinned_host_bytes_live(),
            "tables": b.model.table_placement(),
        });
        if let Some(root) = &a.store_gcroot {
            let out = std::process::Command::new("nix-store").args(["--add", &a.out.display().to_string()]).output().context("nix-store --add")?;
            anyhow::ensure!(out.status.success(), "nix-store --add {}: {}", a.out.display(), String::from_utf8_lossy(&out.stderr));
            let path = String::from_utf8(out.stdout)?.trim().to_string();
            let r = std::process::Command::new("nix-store").args(["--add-root", &root.display().to_string(), "--indirect", "--realise", &path]).output().context("nix-store --add-root")?;
            anyhow::ensure!(r.status.success(), "gc root {}: {}", root.display(), String::from_utf8_lossy(&r.stderr));
            m["store_path"] = json!(path);
            m["gcroot"] = json!(root.display().to_string());
        }
        let text = serde_json::to_string_pretty(&m)?;
        write_text(&a.out.join("metrics.json"), &(text.clone() + "\n"))?;
        if let Some(p) = &a.metrics {
            write_text(p, &(text + "\n"))?;
        }
        Ok(m)
    }

    fn serving(b: &Base, adapter: &Path) -> Result<Qwen4ExpServingAdapter> {
        let candle_core::Device::Cuda(dev) = &b.device else { anyhow::bail!("not cuda") };
        let copy = nv_layers::cuda_stream::current_stream(dev).fork()?;
        let host = nv_specdecode::qwen4_exp_adapters::HostStagedAdapter::read(adapter, &copy)?;
        host.validate(&b.model, b.model.base_identity())?;
        host.to_device(&b.model, &copy)
    }

    pub fn eval(a: EvalCmd) -> Result<Value> {
        let b = open(&a.base)?;
        let rows = read_rows(&a.data)?;
        let serve = serving(&b, &a.adapter)?;
        let mut state = b.model.new_state(a.max_seq.max(2))?;
        let mut ce = |row: &TokenRow| -> Result<f32> { served_row_ce(&b.model, row, &mut state) };
        let (mut ad, mut base, mut ids, mut groups) = (vec![], vec![], vec![], vec![]);
        for (i, r) in rows.iter().enumerate() {
            let Ok(tr) = token_row(&b, r, a.max_seq) else {
                continue;
            };
            Qwen4ExpServingAdapter::detach(&b.model);
            base.push(ce(&tr)?);
            serve.attach(&b.model)?;
            ad.push(ce(&tr)?);
            Qwen4ExpServingAdapter::detach(&b.model);
            ids.push(r.get("row_id").cloned().unwrap_or(json!(i)));
            groups.push(r.get("group").or_else(|| r.get("split_key")).cloned().unwrap_or(json!(i.to_string())));
        }
        let n = ad.len();
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
        let res = json!({"n": n, "row_ids": ids, "groups": groups, "eval_ce_adapter": ad, "eval_ce_base": base,
                         "eval_loss_adapter": mean(&ad), "eval_loss_base": mean(&base)});
        write_text(&a.out, &(res.to_string() + "\n"))?;
        Ok(json!({"n": n, "eval_loss_adapter": mean(&ad), "eval_loss_base": mean(&base)}))
    }

    pub fn generate(a: GenCmd) -> Result<Value> {
        let b = open(&a.base)?;
        let mut rows = read_rows(&a.data)?;
        if a.limit > 0 {
            rows.truncate(a.limit);
        }
        if let Some(dir) = &a.adapter {
            serving(&b, dir)?.attach(&b.model)?;
        }
        let mut state = b.model.new_state(a.max_seq + a.max_new_tokens + 1)?;
        let mut lines = String::new();
        for (i, r) in rows.iter().enumerate() {
            let mut prompt = encode(&b, &prompt_and_completion(&b, r)?.0)?;
            if prompt.len() > a.max_seq {
                prompt = prompt[prompt.len() - a.max_seq..].to_vec();
            }
            state.reset(b.model.config(), &b.device)?;
            let mut logits = b.model.forward(&prompt, &mut state)?;
            let mut out = Vec::new();
            for _ in 0..a.max_new_tokens {
                let tok = logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?;
                if b.eos.contains(&tok) {
                    break;
                }
                out.push(tok);
                logits = b.model.forward(&[tok], &mut state)?;
            }
            let answer = b.tokenizer.decode(&out, true).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
            lines.push_str(&(json!({"row_id": r.get("row_id").cloned().unwrap_or(json!(i)), "answer": answer}).to_string() + "\n"));
        }
        Qwen4ExpServingAdapter::detach(&b.model);
        write_text(&a.out, &lines)?;
        Ok(json!({"generated": rows.len()}))
    }

    pub fn bench_layer(a: BenchCmd) -> Result<Value> {
        require_qwen4_exp(&a.base)?;
        let device = Device::new_cuda_with_stream(0)?;
        nv_models::qwen4_exp::lora::bench_real_layer(&a.base, a.layer, a.tokens, a.experts, &device)
    }
}
