mod hub_snapshot;
mod official_template;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use nv_models::gemma4_gguf::gemma4_moe_config_from_gguf;
use nv_models::gemma4_moe::Gemma4Moe;
use nv_models::CausalLm;
use nv_weights::GgufLoader;
use official_template::OfficialTemplate;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

const GGUF_FILE: &str = "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf";

fn gguf_gemma26b_home_default() -> String {
    format!(
        "{}/.cache/gguf-gemma26b/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        std::env::var("HOME").unwrap_or_default()
    )
}

const TEMPLATE_REPO: &str = "google/gemma-4-26B-A4B-it";

fn gated() -> bool {
    std::env::var("NV_GGUF_NATIVE_Q4").ok().as_deref() == Some("1")
}

fn gguf_candidates() -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Ok(p) = std::env::var("NV_GGUF_PATH") {
        out.push(std::path::PathBuf::from(p));
    }
    out.push(std::path::PathBuf::from(gguf_gemma26b_home_default()));
    let home = std::env::var("HOME").unwrap_or_default();
    if !home.is_empty() {
        let home = std::path::PathBuf::from(home);
        out.push(home.join(".cache/gguf-gemma26b").join(GGUF_FILE));
        let snaps = home
            .join(".cache/huggingface/hub")
            .join("models--unsloth--gemma-4-26B-A4B-it-qat-GGUF/snapshots");
        if let Ok(rd) = std::fs::read_dir(&snaps) {
            for e in rd.flatten() {
                out.push(e.path().join(GGUF_FILE));
            }
        }
    }
    out
}

fn gguf_path(test: &str) -> Option<String> {
    for p in gguf_candidates() {
        if p.exists() {
            return Some(p.to_string_lossy().into_owned());
        }
    }
    let looked: Vec<String> = gguf_candidates()
        .iter()
        .map(|p| p.display().to_string())
        .collect();
    hub_snapshot::precondition_absent(
        test,
        &format!("no gemma-4-26B QAT GGUF; looked in {looked:?}"),
        "realize the hf-models GGUF entry added by ff5dc939c (task #53), or set \
         NV_GGUF_PATH; download to /tank, NOT to zroot",
    );
    None
}

fn template_dir(test: &str) -> Option<PathBuf> {
    let dir = hub_snapshot::dir_from_env_or_hub(
        "NV_GGUF_TEMPLATE_DIR",
        TEMPLATE_REPO,
        &["chat_template.jinja", "tokenizer_config.json"],
    );
    if dir.is_none() {
        hub_snapshot::precondition_absent(
            test,
            &format!("no cached {TEMPLATE_REPO} snapshot carrying chat_template.jinja"),
            "hf download google/gemma-4-26B-A4B-it, or set NV_GGUF_TEMPLATE_DIR to a \
             26B snapshot dir. Do NOT substitute the 31B or E4B template: the vocab is \
             shared across gemma-4 sizes but the templates are three different files",
        );
    }
    dir
}

fn cuda_device(test: &str) -> Device {
    Device::new_cuda(0)
        .unwrap_or_else(|e| panic!("{test} is a GPU test; CUDA device 0 did not open: {e}"))
}

fn rel_rms(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut num = 0f64;
    let mut den = 0f64;
    let mut maxabs = 0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x - y) as f64;
        num += d * d;
        den += (y as f64) * (y as f64);
        maxabs = maxabs.max((x - y).abs());
    }
    let rr = if den > 0.0 {
        (num / den).sqrt() as f32
    } else {
        num.sqrt() as f32
    };
    (rr, maxabs)
}

fn argmax(v: &[f32]) -> (usize, f32) {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    (bi, bv)
}

fn gpu_mem_used_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

struct GgufVocab {
    tokens: Vec<String>,
    id_of: HashMap<String, u32>,
}

impl GgufVocab {
    fn load(g: &GgufLoader) -> Result<Self> {
        let tokens: Vec<String> = g.md_list("tokenizer.ggml.tokens", |v| {
            v.to_string()
                .map(|s| s.to_string())
                .map_err(|e| anyhow::anyhow!("token not string: {e}"))
        })?;
        let mut id_of = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            id_of.entry(t.clone()).or_insert(i as u32);
        }
        Ok(Self { tokens, id_of })
    }

    fn id(&self, piece: &str) -> Option<u32> {
        self.id_of.get(piece).copied()
    }

    fn encode_text(&self, text: &str) -> Vec<u32> {
        let norm = text.replace(' ', "\u{2581}");
        let chars: Vec<char> = norm.chars().collect();
        let unk = self.id("<unk>").unwrap_or(3);
        let mut ids = Vec::new();
        let mut i = 0usize;
        while i < chars.len() {
            let end = (i + 32).min(chars.len());
            let mut hit = None;
            for j in (i + 1..=end).rev() {
                let piece: String = chars[i..j].iter().collect();
                if let Some(id) = self.id(&piece) {
                    hit = Some((id, j));
                    break;
                }
            }
            if let Some((id, j)) = hit {
                ids.push(id);
                i = j;
            } else {
                let mut buf = [0u8; 4];
                let s = chars[i].encode_utf8(&mut buf);
                let mut any = false;
                for b in s.bytes() {
                    if let Some(id) = self.id(&format!("<0x{b:02X}>")) {
                        ids.push(id);
                        any = true;
                    }
                }
                if !any {
                    ids.push(unk);
                }
                i += 1;
            }
        }
        ids
    }

    fn special_at(&self, chars: &[char], i: usize) -> Option<(u32, usize)> {
        let end = (i + 40).min(chars.len());
        let mut best: Option<(u32, usize)> = None;
        for j in i + 1..end {
            if chars[j] != '>' {
                continue;
            }
            let cand: String = chars[i..=j].iter().collect();
            if let Some(id) = self.id(&cand) {
                best = Some((id, j + 1));
            }
        }
        best
    }

    fn encode_rendered(&self, rendered: &str) -> Vec<u32> {
        let chars: Vec<char> = rendered.chars().collect();
        let mut ids = Vec::new();
        let mut lit = String::new();
        let mut i = 0usize;
        while i < chars.len() {
            if chars[i] == '<' {
                if let Some((id, next)) = self.special_at(&chars, i) {
                    if !lit.is_empty() {
                        ids.extend(self.encode_text(&lit));
                        lit.clear();
                    }
                    ids.push(id);
                    i = next;
                    continue;
                }
                let run: String = chars[i..(i + 40).min(chars.len())].iter().collect();
                let looks_special = run.starts_with("<|")
                    || run
                        .find('>')
                        .is_some_and(|e| run[..=e].ends_with("|>") && e > 1);
                assert!(
                    !looks_special,
                    "the chat template emitted the marker at {run:?}, and this GGUF's \
                     tokenizer.ggml.tokens has no such piece. Encoding it as literal text \
                     would prompt the model off-distribution and every tok/s and answer \
                     below would be measured on it silently. Check that the template dir \
                     ({TEMPLATE_REPO}) matches the GGUF's model size."
                );
            }
            lit.push(chars[i]);
            i += 1;
        }
        if !lit.is_empty() {
            ids.extend(self.encode_text(&lit));
        }
        ids
    }

    fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let Some(p) = self.tokens.get(id as usize) else {
                continue;
            };

            if p.len() == 6 && p.starts_with("<0x") && p.ends_with('>') {
                if let Ok(b) = u8::from_str_radix(&p[3..5], 16) {
                    bytes.push(b);
                    continue;
                }
            }
            if skip_special && p.starts_with('<') && p.ends_with('>') && p.len() > 2 {
                continue;
            }
            for ch in p.chars() {
                if ch == '\u{2581}' {
                    bytes.push(b' ');
                } else {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn build_model(path: &str, device: &Device, n_layers: Option<usize>) -> Result<Gemma4Moe> {
    let loader = GgufLoader::open(std::path::Path::new(path), device)?;
    let mut cfg = gemma4_moe_config_from_gguf(&loader)?;
    if let Some(n) = n_layers {
        if n < cfg.base.num_hidden_layers {
            cfg.base.num_hidden_layers = n;
            cfg.base.layer_types.truncate(n);
        }
    }
    Gemma4Moe::from_loader_dtype(cfg, &loader, device, DType::BF16)
}

#[test]
#[ignore = "builds a CPU oracle and a GPU model from a 26 GB GGUF; --ignored NV_GGUF_NATIVE_Q4=1"]
fn gguf_gpu_exact_gate_6layer() {
    let Some(path) = gguf_path("gguf_gpu_exact_gate_6layer") else {
        return;
    };
    if !gated() {
        hub_snapshot::precondition_absent(
            "gguf_gpu_exact_gate_6layer",
            "NV_GGUF_NATIVE_Q4 != 1",
            "set NV_GGUF_NATIVE_Q4=1; this arm builds two 26B models",
        );
        return;
    }
    let cuda = cuda_device("gguf_gpu_exact_gate_6layer");

    let n_layers = 6usize;
    eprintln!("building CPU oracle ({n_layers} layers, bf16) ...");
    let t0 = Instant::now();
    let mut cpu = build_model(&path, &Device::Cpu, Some(n_layers)).expect("cpu model");
    eprintln!("  cpu built in {:?}", t0.elapsed());
    eprintln!("building GPU model ({n_layers} layers, bf16) ...");
    let t0 = Instant::now();
    let mut gpu = build_model(&path, &cuda, Some(n_layers)).expect("gpu model");
    eprintln!(
        "  gpu built in {:?}  vram_used={:?} MiB",
        t0.elapsed(),
        gpu_mem_used_mib()
    );

    let mut seq: Vec<u32> = vec![2, 106, 1645, 108, 4521];
    let mut cpu_traj = Vec::new();
    let mut gpu_traj = Vec::new();
    let mut max_relrms = 0f32;
    let mut traj_match = true;

    for step in 0..6usize {
        let positions: Vec<u32> = (0..seq.len() as u32).collect();
        let cpu_logits = cpu.forward(&seq, &positions).expect("cpu forward");
        let gpu_logits = gpu.forward(&seq, &positions).expect("gpu forward");
        assert_eq!(cpu_logits.len(), gpu_logits.len());
        assert!(gpu_logits.iter().all(|x| x.is_finite()), "gpu non-finite");
        let (rr, ma) = rel_rms(&gpu_logits, &cpu_logits);
        max_relrms = max_relrms.max(rr);
        let (ci, _) = argmax(&cpu_logits);
        let (gi, _) = argmax(&gpu_logits);
        if ci != gi {
            traj_match = false;
        }
        eprintln!(
            "step {step}: rel-RMS={rr:.6} max-abs={ma:.5}  cpu_top1={ci} gpu_top1={gi} match={}",
            ci == gi
        );
        cpu_traj.push(ci as u32);
        gpu_traj.push(gi as u32);

        seq.push(ci as u32);
    }

    eprintln!("cpu(oracle) trajectory: {cpu_traj:?}");
    eprintln!("gpu           trajectory: {gpu_traj:?}");
    eprintln!("max per-step logit rel-RMS = {max_relrms:.6}");

    let pass = traj_match;
    eprintln!(
        "VERDICT (exact greedy agreement vs CPU Q4_0->bf16 oracle): {}  [logit rel-RMS diag: max {:.4}]",
        if pass { "PASS" } else { "FAIL" },
        max_relrms
    );
    assert_eq!(
        cpu_traj,
        vec![392u32, 12, 62, 62, 658, 52],
        "CPU oracle must reproduce the Stage-1 trajectory (harness soundness)"
    );
    assert!(
        pass,
        "GPU bf16 greedy trajectory diverges from the CPU oracle: gpu={gpu_traj:?} cpu={cpu_traj:?}"
    );
}

#[test]
#[ignore = "loads the full 30-layer 26B GGUF onto the GPU; --ignored NV_GGUF_NATIVE_Q4=1"]
fn gguf_gpu_full_depth() {
    let Some(path) = gguf_path("gguf_gpu_full_depth") else {
        return;
    };
    let Some(tdir) = template_dir("gguf_gpu_full_depth") else {
        return;
    };
    if !gated() {
        hub_snapshot::precondition_absent(
            "gguf_gpu_full_depth",
            "NV_GGUF_NATIVE_Q4 != 1",
            "set NV_GGUF_NATIVE_Q4=1; this arm loads the full 30-layer model",
        );
        return;
    }
    let cuda = cuda_device("gguf_gpu_full_depth");

    let want_layers: Option<usize> = std::env::var("NV_GGUF_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok());
    let mem_before = gpu_mem_used_mib();
    eprintln!(
        "building {} GPU model (bf16) ...",
        want_layers
            .map(|n| format!("{n}-layer"))
            .unwrap_or_else(|| "FULL 30-layer".into())
    );
    let t0 = Instant::now();
    let model = build_model(&path, &cuda, want_layers).expect("full gpu model");
    let load_dt = t0.elapsed();
    let mem_after = gpu_mem_used_mib();
    let n_loaded = model.config().base.num_hidden_layers;
    eprintln!(
        "model built ({n_loaded} layers) in {:?}  vram: before={mem_before:?} after={mem_after:?} MiB  ({:.2} GB/layer)",
        load_dt,
        (mem_after.unwrap_or(0).saturating_sub(mem_before.unwrap_or(0)) as f64) / 1024.0 / n_loaded as f64
    );

    let loader =
        GgufLoader::open(std::path::Path::new(&path), &Device::Cpu).expect("open for vocab");
    let vocab = GgufVocab::load(&loader).expect("load vocab");
    eprintln!("vocab size = {}", vocab.tokens.len());
    let eot = vocab.id("<turn|>");
    let eos = vocab.id("<eos>").unwrap_or(1);

    let tmpl = OfficialTemplate::load(&tdir);
    let question = "What is the capital of France? Answer in one sentence.";
    let rendered = tmpl.render_user(question);
    eprintln!("template: {}", tmpl.source_path.display());
    eprintln!("rendered prompt: {rendered:?}");
    let prompt = vocab.encode_rendered(&rendered);
    eprintln!("special ids: eot={eot:?} eos={eos}");
    eprintln!("prompt {} tokens: {:?}", prompt.len(), prompt);
    eprintln!("prompt re-decoded: {:?}", vocab.decode(&prompt, false));
    assert!(
        !prompt.is_empty(),
        "the template rendered no prompt at all: {rendered:?}"
    );

    let max_len = prompt.len() + 32;
    let mut cache = model.new_kv_cache(max_len).expect("kv cache");

    let pos: Vec<i32> = (0..prompt.len() as i32).collect();
    let toks_t = Tensor::from_vec(prompt.clone(), (1usize, prompt.len()), &cuda).unwrap();
    let pos_t = Tensor::from_vec(pos, prompt.len(), &cuda).unwrap();
    let t_prefill = Instant::now();
    let logits = model
        .forward_with_cache(&toks_t, &pos_t, &mut cache)
        .expect("prefill");
    let prefill_dt = t_prefill.elapsed();
    let vocab_n = model.config().base.vocab_size;
    let last = logits
        .narrow(1, prompt.len() - 1, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(last.len(), vocab_n);
    let (mut cur, _) = argmax(&last);

    let n_new = 16usize;
    let mut generated: Vec<u32> = Vec::new();
    let t_dec = Instant::now();
    for _ in 0..n_new {
        generated.push(cur as u32);
        if Some(cur as u32) == eot || cur as u32 == eos {
            break;
        }
        let cur_len = cache.current_len();
        let tt = Tensor::from_vec(vec![cur as u32], (1usize, 1), &cuda).unwrap();
        let pt = Tensor::from_vec(vec![cur_len as i32], 1usize, &cuda).unwrap();
        let lg = model
            .forward_with_cache(&tt, &pt, &mut cache)
            .expect("decode step");
        let row = lg.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (ni, _) = argmax(&row);
        cur = ni;
    }
    let dec_dt = t_dec.elapsed();
    let tok_s = generated.len() as f64 / dec_dt.as_secs_f64();

    let text = vocab.decode(&generated, true);
    eprintln!("--------------------------------------------------------------");
    eprintln!("prefill: {} tokens in {:?}", prompt.len(), prefill_dt);
    eprintln!(
        "decode : {} tokens in {:?}  => {tok_s:.2} tok/s (incremental, KV-cached)",
        generated.len(),
        dec_dt
    );
    eprintln!("vram used (nvidia-smi) = {:?} MiB", gpu_mem_used_mib());
    eprintln!("generated ids: {generated:?}");
    eprintln!("CONTINUATION: {text:?}");
    eprintln!("--------------------------------------------------------------");

    assert!(!generated.is_empty(), "no tokens generated");
    assert!(
        generated.iter().all(|&t| (t as usize) < vocab_n),
        "generated tokens in vocab"
    );
    let _ = cache.current_len();
}

fn toy_vocab(tokens: &[&str]) -> GgufVocab {
    let tokens: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
    let mut id_of = HashMap::with_capacity(tokens.len());
    for (i, t) in tokens.iter().enumerate() {
        id_of.entry(t.clone()).or_insert(i as u32);
    }
    GgufVocab { tokens, id_of }
}

const TOY_PIECES: [&str; 14] = [
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "user",
    "model",
    "thought",
    "\n",
    "<0x0A>",
    "x",
];

fn toy_pieces_without(drop: &str) -> Vec<&'static str> {
    TOY_PIECES.iter().copied().filter(|p| *p != drop).collect()
}

#[test]
fn the_26b_generation_prompt_carries_the_channel_opener() {
    let Some(dir) = template_dir("the_26b_generation_prompt_carries_the_channel_opener") else {
        return;
    };
    let tmpl = OfficialTemplate::load(&dir);
    let question = "What is the capital of France? Answer in one sentence.";
    let rendered = tmpl.render_user(question);
    eprintln!("template: {}", tmpl.source_path.display());
    eprintln!("rendered: {rendered:?}");
    for marker in ["<|turn>user", "<turn|>", "<|turn>model"] {
        assert!(
            rendered.contains(marker),
            "the {TEMPLATE_REPO} render is missing {marker:?}: {rendered:?}"
        );
    }
    assert!(
        rendered.ends_with("<|channel>thought\n<channel|>"),
        "the {TEMPLATE_REPO} generation prompt must end on the thought-channel \
         opener; a prompt without it is off-distribution even with perfect turn \
         markers. got {rendered:?}"
    );
    let hand_built = format!("{}<|turn>user\n{question}<turn|>\n<|turn>model\n", tmpl.bos);
    assert_ne!(
        rendered, hand_built,
        "if the render ever equals the hand-built wrapper this gate is measuring \
         nothing, because the wrapper is what it exists to reject"
    );
}

#[test]
fn encode_rendered_keeps_every_marker_a_single_piece() {
    let v = toy_vocab(&TOY_PIECES);
    let rendered = "<bos><|turn>user\nx<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
    let ids = v.encode_rendered(rendered);
    assert_eq!(
        ids,
        vec![2, 4, 8, 11, 13, 5, 11, 4, 9, 11, 6, 10, 11, 7],
        "markers must survive as one id each; got {ids:?}"
    );
    assert_eq!(v.decode(&ids, false), rendered);
    let shredded = toy_vocab(&toy_pieces_without("<|channel>")).encode_text("<|channel>");
    assert!(
        shredded.len() > 1,
        "a marker the vocab lacks must shred under plain-text encoding -- that \
         silent shred is exactly what encode_rendered turns into a panic, and if \
         it does not happen this test proves nothing (got {shredded:?})"
    );
}

#[test]
#[should_panic(expected = "has no such piece")]
fn encode_rendered_refuses_a_marker_the_vocab_lacks() {
    let v = toy_vocab(&toy_pieces_without("<|channel>"));
    v.encode_rendered("<bos><|turn>model\n<|channel>thought\n<channel|>");
}
