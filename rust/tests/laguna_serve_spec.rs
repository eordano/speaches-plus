#[path = "common/chat_eval_core.rs"]
mod harness_self_test_no_server_code;

use harness_self_test_no_server_code::*;
use serde_json::{json, Value};
use speaches_plus::oapi::chat_template::ChatTemplate;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const LAGUNA_REPO: &str = "poolside/Laguna-XS-2.1-NVFP4";

pub const PACK_PREFIX: &str = "pack-poolside--Laguna-XS-2.1-NVFP4-";

pub const WHY_A_LAGUNA_PACK: &str = "The Laguna track hand-built its prompts in 30 files as \
     \"〈|EOS|〉<user>{q}</user>\\n<assistant></think>\". That literal is wrong three ways against \
     the snapshot's own chat_template.jinja: (1) the template supplies a default Poolside system \
     persona that the literal omits entirely, (2) generation_config.json sets \
     default_chat_template_kwargs.enable_thinking=true so the shipped generation prompt is \
     <assistant><think>, not <assistant></think>, and (3) nothing in that track stopped on \
     eos_token_id [2, 24] (2 = 〈|EOS|〉, 24 = </assistant>), so every run continued past the end \
     of the assistant turn and began hallucinating a new one. Take prompts from a pack rendered \
     through the snapshot's own template.";

pub const SCAFFOLD_SENTINEL: &str = "\u{3014}LAGUNA_USER_BODY\u{3015}";

pub const SCAFFOLD_LABEL: &str = "scaffold-user";

pub const WHY_A_SCAFFOLD: &str = "Several Laguna harnesses sweep prefill length (needle-in-a-\
     haystack at 65536 ids, ladders up to max_position_embeddings), so a fixed pack of rendered \
     prompts cannot serve them. The pack therefore also carries a SCAFFOLD: the official render of \
     a single user turn whose content is one sentinel. Splitting that render on the sentinel yields \
     the exact prefix and suffix the template puts around any single-user-turn body, so a harness \
     can build a prompt of any length with prefix+body+suffix and get a byte-identical result to \
     calling the template. emit_laguna_prompt_pack proves that identity against every body in the \
     suite plus the long/needle shapes the harnesses actually use, so it is a checked equivalence \
     rather than an assumption about BPE not merging across the <user> boundary.";

pub const LAGUNA_ALLOW_SKIP_ENV: &str = "NV_LAGUNA_ALLOW_SKIP";

pub const WHY_A_MISSING_LAGUNA_IS_NOT_A_PASS: &str = "These three tests need no GPU and load no \
     weights: they read chat_template.jinja, generation_config.json and tokenizer.json out of a \
     cached snapshot. Returning early when the snapshot is absent printed `3 passed` in 0.00s \
     having rendered nothing, which is the shape this repo panics on everywhere else \
     (wgpu_serving_unit, wgpu_gptoss_serving_ab, http_wgpu_bos_guard). Point NV_LAGUNA_DIR or \
     HF_HUB_CACHE at a hub that carries the snapshot, or set NV_LAGUNA_ALLOW_SKIP=1 to skip on \
     purpose.";

pub fn open_or_panic(test: &str) -> Option<LagunaPromptSource> {
    match LagunaPromptSource::open() {
        Ok(src) => Some(src),
        Err(e) if std::env::var(LAGUNA_ALLOW_SKIP_ENV).as_deref() == Ok("1") => {
            eprintln!("SKIP {test}: {e}. {WHY_A_MISSING_LAGUNA_IS_NOT_A_PASS}");
            None
        }
        Err(e) => panic!("{test}: {e}. {WHY_A_MISSING_LAGUNA_IS_NOT_A_PASS}"),
    }
}

pub const REPEATED_UNIT: &str =
    "The quick brown fox jumps over the lazy dog near the quiet river bank. ";

pub const NEEDLE_HEAD: &str = "Remember this: the secret code is 7391.\n";

pub const NEEDLE_TAIL: &str =
    "\nWhat is the secret code I told you at the beginning? Answer with just the code.";

pub fn thinking_on_kwargs() -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    m.insert("enable_thinking".to_string(), json!(true));
    m
}

pub fn thinking_off_kwargs() -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    m.insert("enable_thinking".to_string(), json!(false));
    m
}

pub fn hub_roots() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if p.is_dir() && !out.contains(&p) {
            out.push(p);
        }
    };
    if let Ok(v) = std::env::var("HF_HUB_CACHE") {
        push(PathBuf::from(v));
    }
    push(PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/huggingface/hub"));
    out
}

pub fn laguna_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("NV_LAGUNA_DIR") {
        let p = PathBuf::from(d);
        if p.join("config.json").is_file() {
            return Some(p);
        }
    }
    let leaf = format!("models--{}", LAGUNA_REPO.replace('/', "--"));
    for root in hub_roots() {
        let snaps = root.join(&leaf).join("snapshots");
        let Ok(rd) = std::fs::read_dir(&snaps) else {
            continue;
        };
        let mut dirs: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.join("chat_template.jinja").is_file())
            .collect();
        dirs.sort();
        if let Some(p) = dirs.into_iter().next() {
            return Some(p);
        }
    }
    None
}

pub fn snapshot_id(dir: &Path) -> String {
    dir.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".into())
}

pub fn scratch_dir() -> PathBuf {
    if let Ok(v) = std::env::var("NV_CHAT_EVAL_OUT") {
        return PathBuf::from(v);
    }
    PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/nvk-tmp/chat-eval")
}

pub struct LagunaPromptSource {
    pub dir: PathBuf,
    pub snapshot: String,
    pub template: std::sync::Arc<ChatTemplate>,
    pub tokenizer: tokenizers::Tokenizer,
    pub stops: StopSet,
    pub template_digest: String,
    pub template_bytes: usize,
}

impl LagunaPromptSource {
    pub fn open() -> anyhow::Result<Self> {
        let dir = laguna_dir().ok_or_else(|| {
            anyhow::anyhow!(
                "{LAGUNA_REPO} is not cached in any hub root {:?}",
                hub_roots()
            )
        })?;
        let snapshot = snapshot_id(&dir);
        let template = ChatTemplate::load(&dir)
            .ok_or_else(|| anyhow::anyhow!("{} has no loadable chat template", dir.display()))?;
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let stops = StopSet::from_generation_config(&dir)?;
        let (template_digest, template_bytes) = template_digest_of_dir(&dir)?;
        Ok(Self {
            dir,
            snapshot,
            template,
            tokenizer,
            stops,
            template_digest,
            template_bytes,
        })
    }

    pub fn prompt(
        &self,
        label: &str,
        kind: PromptKind,
        messages: &Value,
        kwargs: &BTreeMap<String, Value>,
    ) -> anyhow::Result<TemplatedPrompt> {
        let rendered = self
            .template
            .render_with_kwargs(messages, None, true, kwargs)?;
        let ids = self
            .tokenizer
            .encode(rendered.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode {label}: {e}"))?
            .get_ids()
            .to_vec();
        Ok(TemplatedPrompt::from_official_render(
            label,
            kind,
            LAGUNA_REPO,
            &self.snapshot,
            &self.template_digest,
            self.template_bytes,
            rendered,
            ids,
        ))
    }

    pub fn long_prompt(
        &self,
        label: &str,
        target_ids: usize,
        kwargs: &BTreeMap<String, Value>,
    ) -> anyhow::Result<TemplatedPrompt> {
        const UNIT: &str = "The scheduler dispatches one warp per row of the projection weight, \
             accumulates in fp32, and writes the result back through a coalesced store. ";
        let mut body = String::new();
        let mut last: Option<TemplatedPrompt> = None;
        for _ in 0..4096 {
            body.push_str(UNIT);
            let msgs = json!([{
                "role": "user",
                "content": format!("Read this note and then summarise it in one sentence.\n\n{body}")
            }]);
            let p = self.prompt(label, PromptKind::OpenEnded, &msgs, kwargs)?;
            let n = p.ids.len();
            last = Some(p);
            if n >= target_ids {
                break;
            }
        }
        last.ok_or_else(|| anyhow::anyhow!("long_prompt {label} produced nothing"))
    }

    pub fn scaffold_prompt(
        &self,
        tag: &str,
        kwargs: &BTreeMap<String, Value>,
    ) -> anyhow::Result<TemplatedPrompt> {
        let msgs = json!([{"role":"user","content": SCAFFOLD_SENTINEL}]);
        let p = self.prompt(
            &format!("{tag}/{SCAFFOLD_LABEL}"),
            PromptKind::OpenEnded,
            &msgs,
            kwargs,
        )?;
        let n = p.rendered.matches(SCAFFOLD_SENTINEL).count();
        anyhow::ensure!(
            n == 1,
            "the scaffold sentinel appears {n} times in the {tag} render; it must appear exactly \
             once for the split to be well defined. {WHY_A_SCAFFOLD}"
        );
        Ok(p)
    }

    pub fn prove_scaffold(
        &self,
        tag: &str,
        kwargs: &BTreeMap<String, Value>,
        bodies: &[String],
    ) -> anyhow::Result<(String, String)> {
        let s = self.scaffold_prompt(tag, kwargs)?;
        let (prefix, suffix) = s
            .rendered
            .split_once(SCAFFOLD_SENTINEL)
            .ok_or_else(|| anyhow::anyhow!("{tag} scaffold lost its sentinel"))?;
        anyhow::ensure!(!bodies.is_empty(), "prove_scaffold needs at least one body");
        for b in bodies {
            let official = self.template.render_with_kwargs(
                &json!([{"role":"user","content": b}]),
                None,
                true,
                kwargs,
            )?;
            let built = format!("{prefix}{b}{suffix}");
            anyhow::ensure!(
                built == official,
                "scaffold prefix+body+suffix disagrees with the official render for a {} byte \
                 body under {tag}. {WHY_A_SCAFFOLD}\nbuilt   : {:?}\nofficial: {:?}",
                b.len(),
                built.chars().take(160).collect::<String>(),
                official.chars().take(160).collect::<String>(),
            );
        }
        Ok((prefix.to_string(), suffix.to_string()))
    }

    pub fn suite_bodies() -> Vec<(&'static str, PromptKind, &'static str)> {
        vec![
            (
                "control-arithmetic",
                PromptKind::Control,
                "What is 2 + 2? Reply with the number only.",
            ),
            (
                "control-capital",
                PromptKind::Control,
                "What is the capital of France? Reply with the city name only.",
            ),
            (
                "control-literal",
                PromptKind::Control,
                "Reply with exactly the word BANANA and nothing else.",
            ),
            (
                "openended-code",
                PromptKind::OpenEnded,
                "Write a Python function that returns the square of a number.",
            ),
            (
                "openended-explain",
                PromptKind::OpenEnded,
                "Explain in two sentences why the sky is blue.",
            ),
            (
                "openended-prose",
                PromptKind::OpenEnded,
                "Summarize the history of computing, era by era, in a few sentences.",
            ),
        ]
    }

    pub fn scaffold_proof_bodies() -> Vec<String> {
        let mut out: Vec<String> = Self::suite_bodies()
            .into_iter()
            .map(|(_, _, b)| b.to_string())
            .collect();
        out.push(REPEATED_UNIT.repeat(64));
        out.push(format!(
            "{NEEDLE_HEAD}{}{NEEDLE_TAIL}",
            REPEATED_UNIT.repeat(512)
        ));
        out.push("Hello, my name is".to_string());
        out.push(String::new());
        out.push("trailing space ".to_string());
        out.push("no-trailing-space".to_string());
        out.push("ends with a newline\n".to_string());
        out.push("\u{3008}|EOS|\u{3009}".to_string());
        out
    }

    pub fn suite(&self) -> anyhow::Result<Vec<TemplatedPrompt>> {
        let mut out = Vec::new();
        for (tag, kwargs) in [
            ("think-on", thinking_on_kwargs()),
            ("think-off", thinking_off_kwargs()),
        ] {
            for (name, kind, body) in Self::suite_bodies() {
                out.push(self.prompt(
                    &format!("{tag}/{name}"),
                    kind,
                    &json!([{"role":"user","content": body}]),
                    &kwargs,
                )?);
            }
            out.push(self.long_prompt(&format!("{tag}/longctx-512"), 512, &kwargs)?);
            out.push(self.long_prompt(&format!("{tag}/longctx-2048"), 2048, &kwargs)?);
            self.prove_scaffold(tag, &kwargs, &Self::scaffold_proof_bodies())?;
            out.push(self.scaffold_prompt(tag, &kwargs)?);
        }
        Ok(out)
    }

    pub fn pack(&self, prompts: Vec<TemplatedPrompt>) -> PromptPack {
        PromptPack {
            model_repo: LAGUNA_REPO.to_string(),
            snapshot: self.snapshot.clone(),
            template_digest: self.template_digest.clone(),
            template_bytes: self.template_bytes,
            stop_ids: self.stops.ids.clone(),
            stop_source: self.stops.source.clone(),
            prompts,
        }
    }

    pub fn pack_path(&self) -> PathBuf {
        let tail = self.snapshot.chars().take(8).collect::<String>();
        scratch_dir().join(format!("{PACK_PREFIX}{tail}.json"))
    }
}

#[test]
fn the_shipped_template_disagrees_with_the_hand_built_laguna_wrapper() {
    let Some(src) = open_or_panic("the_shipped_template_disagrees_with_the_hand_built_laguna_wrapper")
    else {
        return;
    };
    let q = "What is the capital of France? Reply with the city name only.";
    let msgs = json!([{"role":"user","content": q}]);
    let hand_built = format!("〈|EOS|〉<user>{q}</user>\n<assistant></think>");
    let shipped = src
        .template
        .render(&msgs, None, true)
        .expect("render with the snapshot's own defaults");

    eprintln!("hand-built (30 files): {hand_built:?}");
    eprintln!("shipped default      : {shipped:?}");
    eprintln!("stop set             : {}", src.stops);

    assert_ne!(hand_built, shipped, "{WHY_A_LAGUNA_PACK}");
    assert!(
        shipped.contains("made by Poolside"),
        "the shipped render carries a default system persona: {shipped:?}"
    );
    assert!(
        shipped.ends_with("<assistant><think>"),
        "the shipped default is thinking-ON: {shipped:?}"
    );
    assert_eq!(
        src.stops.ids,
        vec![2, 24],
        "eos_token_id must be [2 = 〈|EOS|〉, 24 = </assistant>]"
    );

    let hand_ids = src
        .tokenizer
        .encode(hand_built.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    let ship_ids = src
        .tokenizer
        .encode(shipped.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    eprintln!(
        "hand-built tokenizes to {} ids, shipped default to {} ids (delta {})",
        hand_ids.len(),
        ship_ids.len(),
        ship_ids.len() as i64 - hand_ids.len() as i64
    );
    assert!(
        ship_ids.len() > hand_ids.len(),
        "the omitted system block must cost tokens"
    );
}

#[test]
fn the_pack_scaffold_reproduces_the_official_render_for_any_body() {
    let Some(src) = open_or_panic("the_pack_scaffold_reproduces_the_official_render_for_any_body")
    else {
        return;
    };
    let bodies = LagunaPromptSource::scaffold_proof_bodies();
    assert!(bodies.len() >= 8, "too few proof bodies to mean anything");
    for (tag, kwargs) in [
        ("think-on", thinking_on_kwargs()),
        ("think-off", thinking_off_kwargs()),
    ] {
        let (prefix, suffix) = src
            .prove_scaffold(tag, &kwargs, &bodies)
            .expect("scaffold must reproduce the official render");
        eprintln!("{tag} scaffold prefix {prefix:?}");
        eprintln!("{tag} scaffold suffix {suffix:?}");
        assert!(
            prefix.contains("made by Poolside"),
            "{tag} scaffold prefix dropped the default system persona: {prefix:?}"
        );

        let persona_start = prefix
            .find("<system>")
            .expect("prefix carries a <system> block");
        let persona_end = prefix.find("</system>\n").expect("prefix closes <system>") + 10;
        let mut mutilated = prefix.clone();
        mutilated.replace_range(persona_start..persona_end, "");
        let body = "What is the capital of France? Reply with the city name only.";
        let official = src
            .template
            .render_with_kwargs(
                &json!([{"role":"user","content": body}]),
                None,
                true,
                &kwargs,
            )
            .expect("official render");
        assert_eq!(
            format!("{prefix}{body}{suffix}"),
            official,
            "{tag}: the intact scaffold must match"
        );
        assert_ne!(
            format!("{mutilated}{body}{suffix}"),
            official,
            "{tag}: dropping the whole <system> persona block from the scaffold still matched the \
             official render, so this equality check cannot detect the exact defect the Laguna \
             track shipped for 23 files and is worthless as a gate"
        );
        let hand_built = format!("\u{3008}|EOS|\u{3009}<user>{body}</user>\n<assistant></think>");
        assert_ne!(
            format!("{prefix}{body}{suffix}"),
            hand_built,
            "{tag}: the scaffold reproduced the hand-built wrapper, which would mean the wrapper \
             was right all along"
        );
        eprintln!(
            "{tag}: scaffold==official for {} bodies; persona-stripped scaffold != official; \
             hand-built wrapper != official",
            bodies.len()
        );
    }
}

#[test]
fn emit_laguna_prompt_pack() {
    let Some(src) = open_or_panic("emit_laguna_prompt_pack") else {
        return;
    };
    let prompts = src.suite().expect("render the laguna suite");
    let pack = src.pack(prompts);
    assert!(
        pack.controls() >= 2,
        "a pack with fewer than two controls cannot carry an A/B claim"
    );
    let p = src.pack_path();
    pack.write_json(&p).expect("write pack");
    eprintln!(
        "wrote {} ({} prompts, {} controls, {})",
        p.display(),
        pack.prompts.len(),
        pack.controls(),
        pack.stop_set()
    );
    for pr in &pack.prompts {
        eprintln!("  {:<28} [{}] {} ids", pr.label, pr.kind, pr.ids.len());
    }
    eprintln!("{WHY_A_LAGUNA_PACK}");
}

#[cfg(feature = "cuda")]
mod serving {
    use super::*;
    use speaches_plus::oapi::chat::{ChatEngine, ChatEvent, ChatGenerateRequest};
    use speaches_plus::oapi::chat_engine::NvEngineChat;
    use std::time::Instant;

    #[derive(Debug, Clone)]
    struct RunOut {
        text: String,
        finish: String,
        completion: u32,
        elapsed_s: f64,
    }

    fn greedy_req(prompt: &str, max_new: usize) -> ChatGenerateRequest {
        ChatGenerateRequest {
            prompt: prompt.to_string(),
            max_new_tokens: max_new,
            stop: Vec::new(),
            seed: Some(0),
            temperature: Some(0.0),
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
            guided: None,
            guided_think_close: None,
            logit_bias: Vec::new(),
            logprobs: false,
            top_logprobs: 0,
            kv_resume: None,
            kv_store: None,
            mm: None,
        }
    }

    async fn run_once(engine: &NvEngineChat, prompt: &str, max_new: usize) -> RunOut {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<ChatEvent>(256);
        let t0 = Instant::now();
        engine
            .generate(greedy_req(prompt, max_new), tx)
            .await
            .expect("generate");
        let mut text = String::new();
        let mut finish = String::new();
        let mut completion = 0u32;
        while let Some(ev) = rx.recv().await {
            match ev {
                ChatEvent::Started { .. } => {}
                ChatEvent::PromptCached { .. }
                | ChatEvent::StoppedBy { .. }
                | ChatEvent::ReasoningDelta(_) => {}
                ChatEvent::TextDelta(p) => text.push_str(&p),
                ChatEvent::Logprob(_) => {}
                ChatEvent::Done {
                    finish_reason,
                    completion_tokens,
                } => {
                    finish = finish_reason;
                    completion = completion_tokens;
                    break;
                }
                ChatEvent::Error(e) => panic!("engine error: {e}"),
            }
        }
        RunOut {
            text,
            finish,
            completion,
            elapsed_s: t0.elapsed().as_secs_f64(),
        }
    }

    fn report(tag: &str, name: &str, r: &RunOut) {
        let per_tok = if r.completion > 0 {
            1000.0 * r.elapsed_s / r.completion as f64
        } else {
            f64::NAN
        };
        eprintln!(
            "[laguna_serve_spec] {name} {tag}: {} tokens in {:.1} ms wall ({:.2} ms/token, {:.1} tok/s) finish={}",
            r.completion,
            1000.0 * r.elapsed_s,
            per_tok,
            1000.0 / per_tok,
            r.finish
        );
        let words: Vec<&str> = r.text.split_whitespace().collect();
        let distinct: std::collections::BTreeSet<&str> = words.iter().copied().collect();
        let mut longest_run = 0usize;
        let mut run = 0usize;
        for w in words.windows(2) {
            run = if w[0] == w[1] { run + 1 } else { 0 };
            longest_run = longest_run.max(run);
        }
        eprintln!(
            "[laguna_serve_spec] {name} {tag}: words={} distinct={} ratio={:.3} longest_repeat_run={longest_run}",
            words.len(),
            distinct.len(),
            if words.is_empty() {
                0.0
            } else {
                distinct.len() as f64 / words.len() as f64
            }
        );
        eprintln!("[laguna_serve_spec] {name} {tag} text: {:?}", r.text);
    }

    fn serving_cases(src: &LagunaPromptSource) -> Vec<(String, String, usize)> {
        let mut out = Vec::new();
        for pr in src.suite().expect("render the laguna suite") {
            if !pr.label.starts_with("think-off/")
                || pr.label.contains("longctx")
                || pr.label.ends_with(SCAFFOLD_LABEL)
            {
                continue;
            }
            let max_new = if pr.label.contains("control") {
                192
            } else {
                64
            };
            out.push((pr.label.clone(), pr.rendered.clone(), max_new));
        }
        out
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn laguna_serve_spec_matches_normal_greedy() {
        if std::env::var("NV_LAGUNA_TEST").is_err() {
            panic!(
                "set NV_LAGUNA_TEST=1 to run this GPU test (it must never silently skip); \
                 optionally NV_LAGUNA_DIR / NV_LAGUNA_DFLASH_DIR"
            );
        }
        let src = LagunaPromptSource::open().expect("Laguna-XS-2.1-NVFP4 snapshot");
        eprintln!(
            "[laguna_serve_spec] prompts rendered through {}/chat_template.jinja; {}",
            src.dir.display(),
            src.stops
        );
        std::env::set_var("NV_LAGUNA_SERVE_SPEC", "1");
        std::env::set_var("NV_LAGUNA_SERVE_DRAFT", "0");
        let t0 = Instant::now();
        let engine = NvEngineChat::try_load(&src.dir).expect("load laguna serving engine");
        eprintln!(
            "[laguna_serve_spec] engine loaded (incl. spec warmup) in {:.1}s",
            t0.elapsed().as_secs_f64()
        );

        let mut ended = 0usize;
        let cases = serving_cases(&src);
        assert!(!cases.is_empty(), "no serving cases rendered");
        for (name, prompt, max_new) in cases {
            std::env::set_var("NV_LAGUNA_SERVE_SPEC", "1");
            let _ = run_once(&engine, &prompt, max_new).await;
            let spec = run_once(&engine, &prompt, max_new).await;
            report("spec", &name, &spec);

            std::env::set_var("NV_LAGUNA_SERVE_SPEC", "0");
            let _ = run_once(&engine, &prompt, max_new).await;
            let normal = run_once(&engine, &prompt, max_new).await;
            report("normal", &name, &normal);

            assert_eq!(spec.text, normal.text, "case {name}: text mismatch");
            assert_eq!(spec.finish, normal.finish, "case {name}: finish mismatch");
            assert_eq!(
                spec.completion, normal.completion,
                "case {name}: completion_tokens mismatch"
            );
            assert!(spec.completion > 0, "case {name}: no tokens");
            if spec.finish == "stop" {
                ended += 1;
            }
            if name.contains("control") {
                assert_eq!(
                    spec.finish, "stop",
                    "control {name} did not end its turn within {max_new} tokens; a control that \
                     runs to length is a broken control. {WHY_A_LAGUNA_PACK}"
                );
            }
        }
        eprintln!("[laguna_serve_spec] {ended} case(s) ended their turn on eos_token_id");
        std::env::set_var("NV_LAGUNA_SERVE_SPEC", "1");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn laguna_serve_spec_draft_smoke() {
        if std::env::var("NV_LAGUNA_TEST").is_err()
            || std::env::var("NV_LAGUNA_SERVE_DRAFT").is_err()
        {
            panic!(
                "set NV_LAGUNA_TEST=1 and NV_LAGUNA_SERVE_DRAFT=1 to run the draft-mode smoke \
                 (it must never silently skip)"
            );
        }
        let src = LagunaPromptSource::open().expect("Laguna-XS-2.1-NVFP4 snapshot");
        std::env::set_var("NV_LAGUNA_SERVE_SPEC", "1");
        let t0 = Instant::now();
        let engine = NvEngineChat::try_load(&src.dir).expect("load laguna serving engine");
        eprintln!(
            "[laguna_serve_spec] draft engine loaded in {:.1}s",
            t0.elapsed().as_secs_f64()
        );

        for (name, prompt, max_new) in serving_cases(&src) {
            std::env::set_var("NV_LAGUNA_SERVE_SPEC", "1");
            let _ = run_once(&engine, &prompt, max_new).await;
            let spec = run_once(&engine, &prompt, max_new).await;
            report("draft-spec", &name, &spec);

            std::env::set_var("NV_LAGUNA_SERVE_SPEC", "0");
            let normal = run_once(&engine, &prompt, max_new).await;
            report("normal", &name, &normal);

            assert!(spec.completion > 0, "case {name}: no tokens");
            assert!(
                spec.finish == "stop" || spec.finish == "length",
                "case {name}: unexpected finish {}",
                spec.finish
            );
            let agree = spec
                .text
                .bytes()
                .zip(normal.text.bytes())
                .take_while(|(a, b)| a == b)
                .count();
            eprintln!(
                "[laguna_serve_spec] draft {name}: byte-identical={} matching prefix {agree}/{} bytes (draft mode is greedy-class, identity not asserted)",
                spec.text == normal.text,
                normal.text.len()
            );
        }
        std::env::set_var("NV_LAGUNA_SERVE_SPEC", "1");
    }
}
