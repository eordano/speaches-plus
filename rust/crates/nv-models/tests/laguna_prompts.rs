#![allow(dead_code)]

#[path = "../../../tests/common/chat_eval_core.rs"]
pub mod harness_self_test_no_server_code;

pub use harness_self_test_no_server_code::*;

use std::path::{Path, PathBuf};

pub const LAGUNA_REPO: &str = "poolside/Laguna-XS-2.1-NVFP4";

pub const PACK_PREFIX: &str = "pack-poolside--Laguna-XS-2.1-NVFP4-";

pub const EMIT_CMD: &str = "NVK_LANE=<lane> NVK_PKG=speaches-plus NVK_FEATURES= \
     rust/scripts/nvk.sh test --test laguna_serve_spec emit_laguna_prompt_pack -- --nocapture";

pub const WHY_A_LAGUNA_PACK: &str = "MEASURED 2026-08-07 against \
     models--poolside--Laguna-XS-2.1-NVFP4/snapshots/d32afde8: the wrapper this track hand-built \
     in 30 files, \"〈|EOS|〉<user>{q}</user>\\n<assistant></think>\", tokenizes to 22 ids while the \
     snapshot's own chat_template.jinja renders the same question to 55 ids. The 33-token delta is \
     a default Poolside system persona the literal drops entirely. The literal is also \
     thinking-OFF while generation_config.json ships \
     default_chat_template_kwargs.enable_thinking=true, so it is not the prompt this model is \
     served with either. Take prompts from a pack rendered through the snapshot's own template.";

pub const WHY_EOS: &str =
    "Laguna eos_token_id is [2, 24]: 2 is 〈|EOS|〉 and 24 is </assistant>. A \
     harness that does not stop on 24 keeps decoding after the assistant turn ends and the model \
     starts a fresh turn -- which is exactly how a control-token-leak signature was manufactured \
     elsewhere in this repo and then read as proof of kernel corruption.";

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

pub fn pack_search_dirs() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if p.is_dir() && !out.contains(&p) {
            out.push(p);
        }
    };
    if let Ok(v) = std::env::var("NV_CHAT_EVAL_OUT") {
        push(PathBuf::from(v));
    }
    push(
        PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join(".cache/nvk-tmp/chat-eval"),
    );
    out
}

fn candidate_packs() -> Vec<PathBuf> {
    if let Ok(v) = std::env::var("NV_LAGUNA_PACK") {
        return vec![PathBuf::from(v)];
    }
    let mut out = Vec::new();
    for d in pack_search_dirs() {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                let p = e.path();
                let name = p
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                if name.starts_with(PACK_PREFIX) && name.ends_with(".json") {
                    out.push(p);
                }
            }
        }
    }
    out.sort();
    out
}

pub fn resolve_pack(weights_dir: &Path) -> anyhow::Result<(PathBuf, PromptPack)> {
    let cands = candidate_packs();
    anyhow::ensure!(
        !cands.is_empty(),
        "no Laguna prompt pack in {:?}. {WHY_A_LAGUNA_PACK}\nEmit one with:\n  {EMIT_CMD}",
        pack_search_dirs()
    );
    let mut rejected = Vec::new();
    for p in &cands {
        match PromptPack::load_for_snapshot(p, weights_dir) {
            Ok(pack) => return Ok((p.clone(), pack)),
            Err(e) => rejected.push(format!("{}: {e}", p.display())),
        }
    }
    anyhow::bail!(
        "no Laguna pack matches the chat template shipped by {}. {WHY_A_LAGUNA_PACK}\nRe-emit \
         with:\n  {EMIT_CMD}\nrejected candidates:\n  {}",
        weights_dir.display(),
        rejected.join("\n  ")
    )
}

pub const SCAFFOLD_LABEL: &str = "scaffold-user";

pub const SCAFFOLD_SENTINEL: &str = "\u{3014}LAGUNA_USER_BODY\u{3015}";

pub const WHY_A_SCAFFOLD: &str = "A fixed pack of rendered prompts cannot serve a harness that \
     sweeps prefill length up to max_position_embeddings, so the pack also carries the official \
     render of one user turn whose content is a sentinel. Split on the sentinel and you have the \
     exact prefix and suffix the template wraps around any single-turn body. \
     laguna_serve_spec.rs::the_pack_scaffold_reproduces_the_official_render_for_any_body proves \
     prefix+body+suffix is byte-identical to calling the template, for the suite bodies plus the \
     long and needle shapes, and proves that a scaffold with the <system> persona stripped is NOT \
     identical -- so the equality is a gate that can deny, not a tautology.";

pub const REPEATED_UNIT: &str =
    "The quick brown fox jumps over the lazy dog near the quiet river bank. ";

pub const WHY_VARIED: &str = "REPEATED_UNIT pads a prompt by repeating ONE sentence, so every \
     prompt built through filled_ids/ids_at_least is a degenerate corpus. A greedy decode over it \
     continues the only pattern it was shown and emits a period-p loop whose p is the token length \
     of the unit itself, which tail_cycle then flags -- in EVERY arm, before any arm has had a \
     chance to differ. That makes such a prompt useless as a token-identity gate: a 1-ULP prefill \
     difference merely phase-shifts the same loop, so a reported token difference carries no \
     information about the change under test. Any harness that judges trajectory health, or that \
     compares two trajectories token by token, must pad from VARIED_CORPUS instead, and must build \
     the prompt AT its target length rather than slicing a longer one -- a slice drops the \
     scaffold suffix that closes the user turn and opens the assistant turn, leaving the model \
     continuing raw filler mid-sentence.";

pub const VARIED_HEAD: &str = "Read the following passage carefully, then explain in your own \
     words what it describes and why it mattered.\n\n";

pub const VARIED_CORPUS: &str = "\
For most of the age of sail a ship could find how far north or south it lay within a \
few miles, and yet have almost no idea how far east or west it had come. Latitude gives \
itself away to anyone with a clear horizon and a table of declinations, because the sun \
climbs to a predictable height at noon. Longitude hides, because the earth turns, and \
turning makes every meridian look like every other one. A navigator who wanted to place \
himself on the width of an ocean had to know, at the instant of local noon, what hour it \
then was at some fixed reference port. Every four minutes of difference between those two \
clocks is one degree of arc, and one degree near the equator is sixty nautical miles. The \
whole problem therefore collapses into a single unglamorous requirement: carry the time of \
the home port with you, accurately, for months, across salt air and heavy weather.\n\n\
That requirement defeated the best instrument makers in Europe for generations. A pendulum \
regulates a clock beautifully on land and uselessly at sea, since the swing depends on \
gravity and on the platform holding still, and a deck does neither. Heat swells a balance \
wheel and slows it; cold shrinks it and speeds it. Oil thickens in the cold, thins in the \
heat, and gums up with age, so friction wanders instead of staying put. Damp rusts steel \
pivots. Any one of these faults, left uncorrected, throws a reckoning off by degrees, and a \
reckoning wrong by degrees puts a ship onto rocks its captain believed lay well over the \
horizon. In 1707 a squadron under Sir Cloudesley Shovell misjudged its position returning \
from Gibraltar and struck the Scilly Isles, drowning something close to two thousand men \
within sight of home. Parliament responded seven years later with a prize, offering twenty \
thousand pounds for a method that could fix longitude to half a degree after a voyage to \
the West Indies.\n\n\
Most of the learned opinion of the day expected the answer to come from astronomy rather \
than from machinery. The moon moves against the fixed stars quickly enough to serve as a \
clock face, and if its position could be tabulated finely enough, a navigator could read the \
hour off the sky. That approach eventually worked, but it demanded a skilled observer, a \
steady sight, clear weather, and roughly four hours of arithmetic per fix. A Lincolnshire \
carpenter named John Harrison took the unfashionable view that the answer was a better \
clock, and spent the rest of his working life proving it.\n\n\
His solutions were mechanical rather than theoretical, and each one attacked a specific \
enemy. To beat thermal drift he built a gridiron of brass and steel rods whose different \
expansion rates cancelled, holding the effective length of the regulator constant as the \
temperature moved. To beat friction he devised an escapement of springy wooden arms that \
released the train with so little rubbing that it needed no lubrication at all, and he cut \
the parts from lignum vitae, a timber so resinous it oils itself. To beat the motion of the \
deck he coupled two balances with springs so that they swung in opposition, which meant a \
lurch that hurried one retarded the other by the same amount. His early sea clocks were \
enormous, some of them near to a hundredweight, and they looked less like timepieces than \
like small engines.\n\n\
The decisive change came when Harrison abandoned scale entirely. Watching the accuracy of a \
pocket watch made for him by a London craftsman, he concluded that a fast, small oscillator \
disturbed by the sea recovers far more quickly than a slow, heavy one. The instrument he \
built on that insight was about five inches across and could be held in one hand. Sent to \
Jamaica in 1761 in the care of his son, it came home having lost only a few seconds over \
eighty one days, an error corresponding to well under half a degree of longitude and \
comfortably inside the terms of the prize.\n\n\
Collecting the money proved harder than earning it. The board judging the award was \
dominated by astronomers with a competing method, and it demanded repeat trials, then the \
surrender of the mechanism, then the construction of copies by other hands. Harrison was \
past seventy before an intervention by the king secured him most of the sum, and he never \
received the prize as such. The vindication was practical instead. Copies of his design, \
simplified for manufacture by others, went to sea in quantity, and by the end of the \
century an officer could buy a chronometer, check it against a known meridian, and trust \
it. Cook carried one on his second voyage and praised it without reservation. The \
instrument had turned an open question of natural philosophy into a piece of standard \
equipment, which is usually the last thing that happens to a hard problem and the surest \
sign that it has been solved.";

pub const NEEDLE_HEAD: &str = "Remember this: the secret code is 7391.\n";

pub const NEEDLE_TAIL: &str =
    "\nWhat is the secret code I told you at the beginning? Answer with just the code.";

pub const WHY_NON_DEGENERATE: &str = "Acceptance rate and tokens/round are the two metrics where \
     BROKEN output makes the number look BETTER: a repetition loop is trivially draftable, so a \
     collapsed verifier reports a flattering acceptance. This repo has already published a 13.90 \
     ms/tok eagle3 figure taken while the model was looping, and has already read a control-token \
     leak -- manufactured by a harness that decoded past its stop set -- as proof of kernel \
     corruption. Gate every rate on a verifier trajectory that terminates, does not cycle, and \
     does not emit the template's own turn markers.";

#[derive(Clone, Debug)]
pub struct LagunaScaffold {
    pub mode: String,
    pub prefix: String,
    pub suffix: String,
}

impl LagunaScaffold {
    pub fn render(&self, body: &str) -> String {
        format!("{}{}{}", self.prefix, body, self.suffix)
    }
}

pub struct LagunaBuilder {
    pub scaffold: LagunaScaffold,
    pub tokenizer: tokenizers::Tokenizer,
    pub stops: StopSet,
}

impl LagunaBuilder {
    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec())
    }

    pub fn ids(&self, body: &str) -> anyhow::Result<Vec<u32>> {
        self.encode(&self.scaffold.render(body))
    }

    pub fn filled_ids(
        &self,
        head: &str,
        tail: &str,
        target: usize,
        cap: usize,
    ) -> anyhow::Result<Vec<u32>> {
        let per = self.encode(REPEATED_UNIT)?.len().max(1);
        let ceiling = cap.saturating_sub(40);
        anyhow::ensure!(
            ceiling > 4 * per,
            "cap {cap} leaves no room for a templated prompt (unit is {per} ids)"
        );
        let target = target.min(ceiling - 4 * per);
        let build = |reps: usize| -> anyhow::Result<Vec<u32>> {
            self.ids(&format!("{head}{}{tail}", REPEATED_UNIT.repeat(reps)))
        };
        let mut reps = 0usize;
        let mut ids = build(reps)?;
        for _ in 0..64 {
            if ids.len() >= target {
                anyhow::ensure!(
                    ids.len() < ceiling,
                    "smallest prompt reaching {target} ids is {} ids, over the {ceiling} ceiling",
                    ids.len()
                );
                return Ok(ids);
            }
            reps += ((target - ids.len()) / per).max(1);
            ids = build(reps)?;
        }
        anyhow::bail!("filled_ids did not converge on target {target} under cap {cap}")
    }

    pub fn ids_at_least(&self, target: usize, cap: usize) -> anyhow::Result<Vec<u32>> {
        self.filled_ids(
            "Read this note and then summarise it in one sentence.\n\n",
            "",
            target,
            cap,
        )
    }

    pub fn needle_ids(&self, target: usize, cap: usize) -> anyhow::Result<Vec<u32>> {
        self.filled_ids(NEEDLE_HEAD, NEEDLE_TAIL, target, cap)
    }

    pub fn depth_ids(&self, body: &str, target: usize, cap: usize) -> anyhow::Result<Vec<u32>> {
        self.filled_ids(
            "Background reading, not related to the task:\n",
            &format!("\nNow the actual task.\n{body}"),
            target,
            cap,
        )
    }

    pub fn varied_ids(&self, target: usize) -> anyhow::Result<Vec<u32>> {
        let words: Vec<&str> = VARIED_CORPUS.split_whitespace().collect();
        anyhow::ensure!(!words.is_empty(), "VARIED_CORPUS is empty");
        let build = |w: usize| -> anyhow::Result<Vec<u32>> {
            self.ids(&format!("{VARIED_HEAD}{}", words[..w].join(" ")))
        };
        let full = build(words.len())?.len();
        anyhow::ensure!(
            full >= target,
            "VARIED_CORPUS renders to {full} ids, short of the {target} requested. {WHY_VARIED}"
        );
        let (mut lo, mut hi) = (0usize, words.len());
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if build(mid)?.len() <= target {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let ids = build(lo)?;
        anyhow::ensure!(
            ids.len() <= target && ids.len() + VARIED_BAND >= target,
            "varied_ids({target}) landed on {} ids, outside the -{VARIED_BAND} band; the corpus \
             has a word longer than the band or a head longer than the target",
            ids.len()
        );
        Ok(ids)
    }
}

pub const VARIED_BAND: usize = 16;

pub fn template_markers(dir: &Path) -> Vec<String> {
    let Ok(src) = std::fs::read_to_string(dir.join("chat_template.jinja")) else {
        return Vec::new();
    };
    let ch: Vec<char> = src.chars().collect();
    let mut out: Vec<String> = Vec::new();
    for (i, c) in ch.iter().enumerate() {
        if *c != '<' {
            continue;
        }
        let mut inner = String::new();
        let mut k = i + 1;
        while k < ch.len() {
            let d = ch[k];
            if d == '>' {
                break;
            }
            if d.is_whitespace() || d == '"' || d == '\'' || d == '<' || inner.chars().count() > 24
            {
                inner.clear();
                break;
            }
            inner.push(d);
            k += 1;
        }
        if inner.is_empty() || k >= ch.len() {
            continue;
        }
        let name = inner.trim_start_matches('/');
        if name.is_empty() || !name.chars().all(|x| x.is_ascii_lowercase() || x == '_') {
            continue;
        }
        let run = format!("<{inner}>");
        if !out.contains(&run) {
            out.push(run);
        }
    }
    out.sort();
    out
}

pub fn expected_during_generation(marker: &str) -> bool {
    matches!(marker, "<think>" | "</think>")
}

pub fn tail_cycle(tokens: &[u32], max_period: usize, min_reps: usize) -> Option<(usize, usize)> {
    let n = tokens.len();
    for p in 1..=max_period.min(n / min_reps.max(2)) {
        let last = &tokens[n - p..];
        let mut reps = 1usize;
        while (reps + 1) * p <= n && &tokens[n - (reps + 1) * p..n - reps * p] == last {
            reps += 1;
        }
        if reps >= min_reps && p * reps >= 12 {
            return Some((p, reps));
        }
    }
    None
}

#[derive(Clone, Debug)]
pub struct Degeneracy {
    pub label: String,
    pub steps: usize,
    pub terminated: bool,
    pub stop_token: Option<u32>,
    pub cycle: Option<(usize, usize)>,
    pub leaks: Vec<String>,
    pub text: String,
}

impl Degeneracy {
    pub fn is_degenerate(&self) -> bool {
        self.cycle.is_some() || !self.leaks.is_empty()
    }

    pub fn reasons(&self) -> Vec<String> {
        let mut v = Vec::new();
        if let Some((p, r)) = self.cycle {
            v.push(format!(
                "REPETITION LOOP: the last {} of {} tokens are a period-{p} block repeated {r} times",
                p * r,
                self.steps
            ));
        }
        if !self.leaks.is_empty() {
            v.push(format!(
                "CONTROL-TOKEN LEAK: the completion contains the template's own turn markers {:?}",
                self.leaks
            ));
        }
        v
    }
}

impl std::fmt::Display for Degeneracy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[laguna] {}: {} step(s), {}, cycle {:?}, leaks {:?}",
            self.label,
            self.steps,
            match self.stop_token {
                Some(t) => format!("ENDED-TURN on stop id {t}"),
                None => "NOT-TERMINATED".to_string(),
            },
            self.cycle,
            self.leaks
        )
    }
}

pub fn inspect_trajectory(
    label: &str,
    tokens: &[u32],
    stops: &StopSet,
    tokenizer: &tokenizers::Tokenizer,
    markers: &[String],
) -> Degeneracy {
    let (kept, stop, _) = truncate_at_stop(tokens, stops);
    let body: Vec<u32> = match stop {
        Some(_) => kept[..kept.len() - 1].to_vec(),
        None => kept.clone(),
    };
    let text = tokenizer.decode(&body, false).unwrap_or_default();
    let leaks: Vec<String> = markers
        .iter()
        .filter(|m| !expected_during_generation(m))
        .filter(|m| text.contains(m.as_str()))
        .cloned()
        .collect();
    Degeneracy {
        label: label.to_string(),
        steps: kept.len(),
        terminated: stop.is_some(),
        stop_token: stop,
        cycle: tail_cycle(&body, 16, 4),
        leaks,
        text,
    }
}

pub fn assert_publishable(d: &Degeneracy, require_termination: bool) {
    eprintln!("{d}");
    assert!(
        !d.is_degenerate(),
        "{} produced a degenerate trajectory, so no acceptance rate or tokens/round taken from it \
         is publishable:\n  {}\n  completion {:?}\n{WHY_NON_DEGENERATE}",
        d.label,
        d.reasons().join("\n  "),
        d.text.chars().take(400).collect::<String>()
    );
    if require_termination {
        assert!(
            d.terminated,
            "{} never reached a stop id in {} steps. {WHY_EOS}",
            d.label, d.steps
        );
    }
}

pub fn thinking_mode() -> &'static str {
    match std::env::var("NV_LAGUNA_THINKING").ok().as_deref() {
        Some("0") | Some("off") | Some("false") => "think-off",
        _ => "think-on",
    }
}

pub fn in_mode(pack: &PromptPack) -> Vec<&TemplatedPrompt> {
    let want = format!("{}/", thinking_mode());
    pack.prompts
        .iter()
        .filter(|p| p.label.starts_with(&want) && !p.label.ends_with(SCAFFOLD_LABEL))
        .collect()
}

pub fn scaffold_of(pack: &PromptPack) -> anyhow::Result<LagunaScaffold> {
    let mode = thinking_mode();
    let want = format!("{mode}/{SCAFFOLD_LABEL}");
    let p = pack
        .prompts
        .iter()
        .find(|p| p.label == want)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "pack carries no {want}; re-emit it with:\n  {EMIT_CMD}\n{WHY_A_SCAFFOLD}"
            )
        })?;
    let n = p.rendered.matches(SCAFFOLD_SENTINEL).count();
    anyhow::ensure!(n == 1, "{want} has {n} sentinels, expected exactly 1");
    let (prefix, suffix) = p.rendered.split_once(SCAFFOLD_SENTINEL).unwrap();
    Ok(LagunaScaffold {
        mode: mode.to_string(),
        prefix: prefix.to_string(),
        suffix: suffix.to_string(),
    })
}

pub fn named<'a>(pack: &'a PromptPack, suffix: &str) -> anyhow::Result<&'a TemplatedPrompt> {
    let want = format!("{}/{suffix}", thinking_mode());
    pack.prompts
        .iter()
        .find(|p| p.label == want)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "pack has no prompt {want}; it has {:?}",
                pack.prompts
                    .iter()
                    .map(|p| p.label.as_str())
                    .collect::<Vec<_>>()
            )
        })
}

pub fn controls(pack: &PromptPack) -> Vec<&TemplatedPrompt> {
    in_mode(pack)
        .into_iter()
        .filter(|p| p.kind == PromptKind::Control)
        .collect()
}

pub fn describe(pack: &PromptPack, path: &Path) -> String {
    format!(
        "pack {} @ {} :: template {} ({} bytes), {}\nmode {} ({} prompt(s), {} control(s)). \
         NV_LAGUNA_THINKING=0 selects the thinking-off renders.\n{WHY_EOS}",
        path.display(),
        pack.snapshot,
        pack.template_digest,
        pack.template_bytes,
        pack.stop_set(),
        thinking_mode(),
        in_mode(pack).len(),
        controls(pack).len(),
    )
}

pub struct LagunaEval {
    pub dir: PathBuf,
    pub pack_path: PathBuf,
    pub pack: PromptPack,
}

impl LagunaEval {
    pub fn open() -> anyhow::Result<Self> {
        let dir = laguna_dir().ok_or_else(|| {
            anyhow::anyhow!(
                "{LAGUNA_REPO} is not cached in any hub root {:?}",
                hub_roots()
            )
        })?;
        let (pack_path, pack) = resolve_pack(&dir)?;
        Ok(Self {
            dir,
            pack_path,
            pack,
        })
    }

    pub fn stops(&self) -> StopSet {
        self.pack.stop_set()
    }

    pub fn describe(&self) -> String {
        describe(&self.pack, &self.pack_path)
    }

    pub fn get(&self, suffix: &str) -> anyhow::Result<&TemplatedPrompt> {
        named(&self.pack, suffix)
    }

    pub fn ids(&self, suffix: &str) -> anyhow::Result<Vec<u32>> {
        Ok(self.get(suffix)?.ids.clone())
    }

    pub fn controls(&self) -> Vec<&TemplatedPrompt> {
        controls(&self.pack)
    }

    pub fn scaffold(&self) -> anyhow::Result<LagunaScaffold> {
        scaffold_of(&self.pack)
    }

    pub fn tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer> {
        tokenizers::Tokenizer::from_file(self.dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))
    }

    pub fn markers(&self) -> Vec<String> {
        template_markers(&self.dir)
    }

    pub fn builder(&self) -> anyhow::Result<LagunaBuilder> {
        Ok(LagunaBuilder {
            scaffold: self.scaffold()?,
            tokenizer: self.tokenizer()?,
            stops: self.stops(),
        })
    }

    pub fn inspect(
        &self,
        label: &str,
        tokens: &[u32],
        tokenizer: &tokenizers::Tokenizer,
    ) -> Degeneracy {
        inspect_trajectory(label, tokens, &self.stops(), tokenizer, &self.markers())
    }
}

pub fn truncate_at_stop(tokens: &[u32], stops: &StopSet) -> (Vec<u32>, Option<u32>, usize) {
    match tokens.iter().position(|t| stops.contains(*t)) {
        Some(i) => (tokens[..=i].to_vec(), Some(tokens[i]), tokens.len() - i - 1),
        None => (tokens.to_vec(), None, 0),
    }
}

pub fn report_termination(label: &str, tokens: &[u32], stops: &StopSet, max_new: usize) -> bool {
    let (kept, stop, dropped) = truncate_at_stop(tokens, stops);
    match stop {
        Some(t) => {
            eprintln!(
                "[laguna] {label}: ENDED-TURN at token {} of {max_new} on stop id {t} ({dropped} \
                 token(s) after the stop would have been free-run by a non-EOS-aware harness)",
                kept.len()
            );
            true
        }
        None => {
            eprintln!(
                "[laguna] {label}: MAX-NEW({max_new}) NOT-TERMINATED after {} tokens. {WHY_EOS}",
                tokens.len()
            );
            false
        }
    }
}

#[test]
fn a_laguna_pack_is_discoverable_and_matches_the_cached_snapshot() {
    let Some(dir) = laguna_dir() else {
        eprintln!("skip: {LAGUNA_REPO} not cached in {:?}", hub_roots());
        return;
    };
    match resolve_pack(&dir) {
        Ok((p, pack)) => {
            eprintln!("{}", describe(&pack, &p));
            for pr in in_mode(&pack) {
                eprintln!("  {:<28} [{}] {} ids", pr.label, pr.kind, pr.ids.len());
            }
            assert!(
                controls(&pack).len() >= 2,
                "a pack with fewer than two controls in mode {} cannot carry an A/B claim",
                thinking_mode()
            );
            assert_eq!(pack.stop_ids, vec![2, 24], "{WHY_EOS}");
        }
        Err(e) => {
            eprintln!("no usable pack yet: {e}");
        }
    }
}

#[test]
fn the_hand_built_wrapper_is_not_what_the_template_produces() {
    let Some(dir) = laguna_dir() else {
        eprintln!("skip: {LAGUNA_REPO} not cached");
        return;
    };
    let tmpl = std::fs::read_to_string(dir.join("chat_template.jinja")).expect("chat_template");
    assert!(
        tmpl.contains("made by Poolside"),
        "the shipped template declares a default system persona; if this stops being true the \
         33-token delta measured in {WHY_A_LAGUNA_PACK} has changed and every Laguna quality \
         number needs re-measuring"
    );
    assert!(
        tmpl.contains("enable_thinking"),
        "the shipped template branches on enable_thinking"
    );
    let gen =
        std::fs::read_to_string(dir.join("generation_config.json")).expect("generation_config");
    let v: serde_json::Value = serde_json::from_str(&gen).expect("parse generation_config");
    assert_eq!(
        v.pointer("/default_chat_template_kwargs/enable_thinking"),
        Some(&serde_json::json!(true)),
        "the shipped default is thinking-ON; a harness that hardcodes </think> is measuring the \
         other mode"
    );
    eprintln!(
        "snapshot {} ships a default system persona and enable_thinking=true",
        dir.display()
    );
}

#[test]
fn the_pack_carries_a_scaffold_that_rebuilds_the_official_render() {
    let Some(dir) = laguna_dir() else {
        eprintln!("skip: {LAGUNA_REPO} not cached");
        return;
    };
    let Ok((path, pack)) = resolve_pack(&dir) else {
        eprintln!("skip: no usable pack yet; emit one with\n  {EMIT_CMD}");
        return;
    };
    let sc = scaffold_of(&pack).expect("pack carries a scaffold");
    eprintln!("pack {} scaffold[{}]", path.display(), sc.mode);
    eprintln!("  prefix {:?}", sc.prefix);
    eprintln!("  suffix {:?}", sc.suffix);
    assert!(
        sc.prefix.contains("made by Poolside"),
        "the scaffold prefix must carry the default system persona the hand-built wrapper drops"
    );
    let body = "What is the capital of France? Reply with the city name only.";
    let rebuilt = sc.render(body);
    let official = in_mode(&pack)
        .into_iter()
        .find(|p| p.label.ends_with("control-capital"))
        .expect("pack has control-capital");
    assert_eq!(
        rebuilt, official.rendered,
        "prefix+body+suffix must reproduce the packed official render byte for byte"
    );

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let ids = tok
        .encode(rebuilt.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();
    assert_eq!(
        ids, official.ids,
        "a scaffold-built prompt must tokenize identically to the packed render"
    );
    eprintln!(
        "scaffold-built prompt == packed official render, {} ids",
        ids.len()
    );
}

#[test]
fn the_scaffold_builder_hits_every_length_the_laguna_harnesses_ask_for() {
    let Some(dir) = laguna_dir() else {
        eprintln!("skip: {LAGUNA_REPO} not cached");
        return;
    };
    let Ok(ev) = LagunaEval::open() else {
        eprintln!("skip: no usable pack yet; emit one with\n  {EMIT_CMD}");
        return;
    };
    let b = ev.builder().expect("builder");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg: serde_json::Value = serde_json::from_str(&raw).expect("parse config");
    let max_pos = cfg["max_position_embeddings"]
        .as_u64()
        .expect("max_position") as usize;

    for (target, cap) in [
        (128usize, max_pos),
        (480, max_pos),
        (512, max_pos),
        (720, max_pos),
        (8192, max_pos),
        (16384, max_pos),
        (264, 512),
    ] {
        let ids = b
            .ids_at_least(target, cap)
            .unwrap_or_else(|e| panic!("ids_at_least({target}, {cap}): {e}"));
        assert!(
            ids.len() >= target,
            "ids_at_least({target}, {cap}) returned {} ids",
            ids.len()
        );
        assert!(
            ids.len() < cap.saturating_sub(40),
            "over ceiling: {}",
            ids.len()
        );
        eprintln!("ids_at_least({target}, {cap}) -> {} ids", ids.len());
    }

    let needle = b.needle_ids(65536, max_pos).expect("needle at 65536");
    assert!(needle.len() >= 65536, "needle {}", needle.len());
    let text = ev.tokenizer().unwrap().decode(&needle, false).unwrap();
    assert!(
        text.contains("made by Poolside") && text.contains("secret code is 7391"),
        "the needle prompt must carry both the official system persona and the needle"
    );
    assert!(
        text.ends_with(&ev.scaffold().unwrap().suffix),
        "the needle prompt must close the user turn and open the assistant turn"
    );
    eprintln!("needle_ids(65536) -> {} ids", needle.len());

    let big = b
        .needle_ids(max_pos.saturating_sub(200), max_pos)
        .expect("needle at max_pos-200");
    assert!(big.len() < max_pos - 40, "needle at cap: {}", big.len());
    eprintln!(
        "needle_ids(max_pos-200={}) -> {} ids",
        max_pos - 200,
        big.len()
    );
}

#[test]
fn the_varied_corpus_is_not_a_repeated_block_and_the_padded_one_is() {
    let Some(_dir) = laguna_dir() else {
        eprintln!("skip: {LAGUNA_REPO} not cached");
        return;
    };
    let Ok(ev) = LagunaEval::open() else {
        eprintln!("skip: no usable pack yet; emit one with\n  {EMIT_CMD}");
        return;
    };
    let b = ev.builder().expect("builder");
    let suffix = ev.scaffold().expect("scaffold").suffix;
    let tok = ev.tokenizer().expect("tokenizer");

    for target in [500usize, 700] {
        let ids = b
            .varied_ids(target)
            .unwrap_or_else(|e| panic!("varied_ids({target}): {e}"));
        assert!(
            ids.len() <= target && ids.len() + VARIED_BAND >= target,
            "varied_ids({target}) -> {} ids",
            ids.len()
        );
        let text = tok.decode(&ids, false).expect("decode");
        assert!(
            text.ends_with(&suffix),
            "varied_ids({target}) must close the user turn and open the assistant turn; a SLICE \
             of a longer prompt does not, which is half of why the pre-fix gate was red. \
             {WHY_VARIED}"
        );
        assert!(
            tail_cycle(&ids, 32, 3).is_none(),
            "varied_ids({target}) is itself a repeated block, so it cannot gate anything"
        );
        eprintln!(
            "varied_ids({target}) -> {} ids, no tail cycle, scaffold intact",
            ids.len()
        );
    }

    let two = b.encode(&REPEATED_UNIT.repeat(2)).expect("encode 2x").len();
    let three = b.encode(&REPEATED_UNIT.repeat(3)).expect("encode 3x").len();
    let in_context = three - two;
    let padded_body = b
        .encode(&REPEATED_UNIT.repeat(45))
        .expect("encode the padded body");
    let body = &padded_body[..padded_body.len() - 1];
    let (p, r) = tail_cycle(body, 32, 3).unwrap_or_else(|| {
        panic!(
            "the body ids_at_least pads with is supposed to be ONE sentence repeated; if it no \
             longer cycles, this deny-side check has stopped denying and must be rewritten. \
             in-context period {in_context}, body {} ids, tail {:?}",
            body.len(),
            &body[body.len() - 40..]
        )
    });
    assert_eq!(
        p, in_context,
        "the period the detector reports must be the period REPEATED_UNIT actually repeats at"
    );
    eprintln!(
        "REPEATED_UNIT repeats at period {in_context}; a 45x padded body reads as a period-{p} \
         block x {r} reps -- the exact shape the model echoed back as the period-15 completion \
         loop that kept this gate permanently red"
    );
}

#[test]
fn the_template_markers_are_read_off_the_shipped_template() {
    let Some(dir) = laguna_dir() else {
        eprintln!("skip: {LAGUNA_REPO} not cached");
        return;
    };
    let m = template_markers(&dir);
    eprintln!("markers derived from chat_template.jinja: {m:?}");
    for want in [
        "<user>",
        "</user>",
        "<assistant>",
        "</assistant>",
        "<system>",
    ] {
        assert!(m.iter().any(|x| x == want), "{want} missing from {m:?}");
    }
    assert!(
        m.iter().filter(|x| !expected_during_generation(x)).count() >= 6,
        "too few leak markers to be a useful check: {m:?}"
    );
}

#[test]
fn a_repetition_loop_and_a_marker_leak_are_both_caught_and_clean_output_is_not() {
    let stops = StopSet {
        ids: vec![2, 24],
        source: "unit".into(),
    };
    let markers: Vec<String> = [
        "<user>",
        "</user>",
        "<assistant>",
        "</assistant>",
        "</think>",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    let mut healthy: Vec<u32> = (100..140).collect();
    healthy.push(24);
    assert_eq!(
        tail_cycle(&healthy, 16, 4),
        None,
        "healthy run must not cycle"
    );

    let mut looped: Vec<u32> = vec![100, 101, 102];
    looped.extend(std::iter::repeat_n(777, 24));
    let (p, r) = tail_cycle(&looped, 16, 4).expect("a 24x repeated token is a loop");
    assert_eq!(p, 1);
    assert!(r >= 24, "reps {r}");

    let mut cycle3: Vec<u32> = vec![50];
    for _ in 0..8 {
        cycle3.extend_from_slice(&[11, 12, 13]);
    }
    let (p3, r3) = tail_cycle(&cycle3, 16, 4).expect("a period-3 cycle is a loop");
    assert_eq!((p3, r3), (3, 8));

    assert_eq!(
        tail_cycle(&[7, 7, 7, 7], 16, 4),
        None,
        "four identical tokens cover only 4 positions and must stay under the bar"
    );

    let leaks: Vec<String> = markers
        .iter()
        .filter(|m| !expected_during_generation(m))
        .filter(|m| "Paris.</assistant><user>and again".contains(m.as_str()))
        .cloned()
        .collect();
    assert_eq!(leaks.len(), 2, "{leaks:?}");
    assert!(
        !markers
            .iter()
            .filter(|m| !expected_during_generation(m))
            .any(|m| "Paris.".contains(m.as_str())),
        "clean text must not be flagged"
    );

    let clean = Degeneracy {
        label: "clean".into(),
        steps: 6,
        terminated: true,
        stop_token: Some(24),
        cycle: None,
        leaks: Vec::new(),
        text: "Paris.".into(),
    };
    assert!(!clean.is_degenerate());
    assert_publishable(&clean, true);

    let broken = Degeneracy {
        label: "looped".into(),
        steps: 96,
        terminated: false,
        stop_token: None,
        cycle: Some((1, 96)),
        leaks: vec!["<user>".into()],
        text: "aaaa".into(),
    };
    assert!(broken.is_degenerate());
    assert_eq!(broken.reasons().len(), 2, "{:?}", broken.reasons());
    let caught = std::panic::catch_unwind(|| assert_publishable(&broken, false));
    assert!(
        caught.is_err(),
        "assert_publishable accepted a 96-token repetition loop with a control-token leak; a gate \
         that cannot deny is not a gate. {WHY_NON_DEGENERATE}"
    );
    let _ = stops;
}

#[test]
fn truncating_at_the_stop_token_keeps_the_stop_and_drops_the_rest() {
    let stops = StopSet {
        ids: vec![2, 24],
        source: "unit".into(),
    };
    let (kept, stop, dropped) = truncate_at_stop(&[7, 8, 24, 9, 10], &stops);
    assert_eq!(kept, vec![7, 8, 24]);
    assert_eq!(stop, Some(24));
    assert_eq!(dropped, 2);

    let (kept, stop, dropped) = truncate_at_stop(&[7, 8, 9], &stops);
    assert_eq!(kept, vec![7, 8, 9]);
    assert_eq!(stop, None);
    assert_eq!(dropped, 0);

    let (kept, stop, _) = truncate_at_stop(&[2], &stops);
    assert_eq!(kept, vec![2]);
    assert_eq!(stop, Some(2));
}
