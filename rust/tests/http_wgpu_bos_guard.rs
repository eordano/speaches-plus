#[cfg(not(feature = "wgpu"))]
#[test]
fn http_wgpu_bos_guard_is_cfg_out_without_the_wgpu_feature_this_is_a_skip_not_a_pass() {
    eprintln!(
        "http_wgpu_bos_guard compiled OUT (no `wgpu` feature): 9 tests vanished and the suite \
         printed 0 passed AND 0 ignored. Six are synthetic BOS-declaration logic and three scan \
         cached snapshots that are present; none needs a GPU. Re-run with \
         NVK_PKG=speaches-plus NVK_FEATURES=cuda,wgpu."
    );
}

#[cfg(feature = "wgpu")]
mod gated {

    use std::path::{Path, PathBuf};

    use speaches_plus::oapi::chat::{ChatMessageIn, MessageContent};
    use speaches_plus::oapi::chat_engine_wgpu::{
        bos_declaration_from_json, classify_wgpu_model, probe_prompt_head, prompt_bos_id,
        BosDeclaration, PromptHeadProbe,
    };

    fn user(text: &str) -> ChatMessageIn {
        ChatMessageIn {
            role: "user".into(),
            content: Some(MessageContent::Text(text.into())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn probe_messages() -> Vec<ChatMessageIn> {
        vec![user("Name the capital of France in one word.")]
    }

    fn hub_roots() -> Vec<PathBuf> {
        let mut roots = Vec::new();
        for key in ["NV_TEMPLATE_HUB", "HF_HUB_CACHE"] {
            if let Ok(p) = std::env::var(key) {
                if !p.is_empty() {
                    roots.push(PathBuf::from(p));
                }
            }
        }
        if let Ok(home) = std::env::var("HOME") {
            roots.push(PathBuf::from(home).join(".cache/huggingface/hub"));
        }
        roots.retain(|r| r.is_dir());
        roots
    }

    const BOS_GUARD_ALLOW_SKIP: &str = "NV_BOS_GUARD_ALLOW_SKIP";

    fn skipping_is_opted_into() -> bool {
        std::env::var(BOS_GUARD_ALLOW_SKIP).as_deref() == Ok("1")
    }

    fn snapshots_or_panic(needle: &str, test: &str) -> Vec<PathBuf> {
        let dirs = snapshots_matching(needle);
        if dirs.is_empty() && !skipping_is_opted_into() {
            panic!(
                "{test}: no cached wgpu-servable models--*{needle}* snapshot under {:?}. With no \
                 snapshot this gate asserts nothing about any checkpoint and still prints ok, \
                 which is the shape that let three never-run tests sit green. Point \
                 NV_TEMPLATE_HUB or HF_HUB_CACHE at a hub that has one, or set \
                 {BOS_GUARD_ALLOW_SKIP}=1 to skip on purpose.",
                hub_roots()
            );
        }
        dirs
    }

    fn snapshots_matching(needle: &str) -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = Vec::new();
        for root in hub_roots() {
            let Ok(rd) = std::fs::read_dir(&root) else {
                continue;
            };
            for repo in rd.flatten() {
                let name = repo.file_name().to_string_lossy().to_ascii_lowercase();
                if !name.starts_with("models--") || !name.contains(&needle.to_ascii_lowercase()) {
                    continue;
                }
                let Ok(snaps) = std::fs::read_dir(repo.path().join("snapshots")) else {
                    continue;
                };
                for snap in snaps.flatten() {
                    let d = snap.path().canonicalize().unwrap_or_else(|_| snap.path());
                    if !d.join("tokenizer.json").exists() {
                        continue;
                    }
                    let Ok(raw_cfg) = std::fs::read_to_string(d.join("config.json")) else {
                        continue;
                    };
                    match classify_wgpu_model(&raw_cfg) {
                        Ok(kind) => {
                            eprintln!("wgpu-servable: {} [{}]", d.display(), kind.label());
                            out.push(d);
                        }
                        Err(err) => eprintln!("not wgpu-servable, skipping {}: {err}", d.display()),
                    }
                }
            }
        }
        out.sort();
        out.dedup();
        out
    }

    fn describe(dir: &Path, p: &PromptHeadProbe) -> String {
        format!(
            "{}\n  dir            = {}\n  tokenizer decl = {}\n  eos            = {:?}\n  \
         generation_config bos_token_id = {:?}\n  bos actually prepended        = {:?}\n  \
         prompt_ids[..4]  = {:?}  (len {})\n  legacy_ids[..4]  = {:?}  (len {})",
            p.model_id,
            dir.display(),
            p.bos_declaration.label(),
            p.eos_ids,
            p.generation_config_bos,
            p.bos_token_id,
            &p.prompt_ids[..p.prompt_ids.len().min(4)],
            p.prompt_ids.len(),
            &p.legacy_prompt_ids[..p.legacy_prompt_ids.len().min(4)],
            p.legacy_prompt_ids.len(),
        )
    }

    #[test]
    fn qwen_shaped_tokenizer_config_declares_no_bos() {
        let cfg = r#"{"bos_token": null, "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"}"#;
        assert_eq!(bos_declaration_from_json(cfg), BosDeclaration::Absent);
        assert_eq!(
            prompt_bos_id(
                &bos_declaration_from_json(cfg),
                &|t| match t {
                    "<|endoftext|>" => Some(248044),
                    "<|im_end|>" => Some(248046),
                    _ => None,
                },
                &[248044, 248046],
                "qwen-shaped",
            ),
            None,
            "a model whose tokenizer_config declares bos_token: null must get no BOS, no matter \
         what generation_config.json says"
        );
    }

    #[test]
    fn gemma_shaped_tokenizer_config_declares_bos() {
        let cfg = r#"{"bos_token": "<bos>", "eos_token": "<eos>"}"#;
        assert_eq!(
            bos_declaration_from_json(cfg),
            BosDeclaration::Token("<bos>".into())
        );
        assert_eq!(
            prompt_bos_id(
                &bos_declaration_from_json(cfg),
                &|t| match t {
                    "<bos>" => Some(2),
                    "<eos>" => Some(1),
                    _ => None,
                },
                &[1, 106, 50],
                "gemma-shaped",
            ),
            Some(2)
        );
    }

    #[test]
    fn a_declared_bos_that_is_also_eos_is_never_prepended() {
        let cfg = r#"{"bos_token": "<|endoftext|>", "eos_token": "<|im_end|>"}"#;
        assert_eq!(
            bos_declaration_from_json(cfg),
            BosDeclaration::Token("<|endoftext|>".into())
        );
        assert_eq!(
            prompt_bos_id(
                &bos_declaration_from_json(cfg),
                &|t| (t == "<|endoftext|>").then_some(248044),
                &[248044, 248046],
                "bos-is-eos",
            ),
            None,
            "an end-of-text token at position 0 corrupts the prompt; the declared BOS must be \
         dropped when it is a member of the EOS set"
        );
    }

    #[test]
    fn add_bos_token_false_suppresses_a_declared_bos() {
        let cfg = r#"{"add_bos_token": false, "bos_token": "<bos>"}"#;
        assert_eq!(bos_declaration_from_json(cfg), BosDeclaration::Suppressed);
        assert_eq!(
            prompt_bos_id(
                &bos_declaration_from_json(cfg),
                &|t| (t == "<bos>").then_some(2),
                &[1],
                "suppressed",
            ),
            None
        );
    }

    #[test]
    fn a_declared_bos_absent_from_the_vocabulary_is_not_prepended() {
        let cfg = r#"{"bos_token": "<not-in-vocab>"}"#;
        assert_eq!(
            prompt_bos_id(
                &bos_declaration_from_json(cfg),
                &|_| None,
                &[1],
                "missing-from-vocab",
            ),
            None
        );
    }

    #[test]
    fn object_form_bos_token_is_read_from_content() {
        let cfg = r#"{"bos_token": {"content": "<bos>", "lstrip": false}}"#;
        assert_eq!(
            bos_declaration_from_json(cfg),
            BosDeclaration::Token("<bos>".into())
        );
    }

    #[test]
    fn cached_qwen_snapshots_never_start_a_prompt_with_an_eos_token() {
        let dirs = snapshots_or_panic("qwen", "cached_qwen_snapshots_never_start_a_prompt_with_an_eos_token");
        if dirs.is_empty() {
            eprintln!(
                "cached_qwen_snapshots_never_start_a_prompt_with_an_eos_token: SKIP \
                 ({BOS_GUARD_ALLOW_SKIP}=1) no cached wgpu-servable models--*Qwen* snapshot"
            );
            return;
        }
        let msgs = probe_messages();
        let mut checked = 0usize;
        let mut legacy_bug_seen = 0usize;
        for dir in &dirs {
            let probe = match probe_prompt_head(dir, &msgs) {
                Ok(p) => p,
                Err(err) => {
                    eprintln!("skip {} ({err:#})", dir.display());
                    continue;
                }
            };
            eprintln!("{}", describe(dir, &probe));
            checked += 1;
            let first = probe.prompt_ids[0];
            assert!(
                !probe.eos_ids.contains(&first),
                "{}: first prompt token {first} is a member of the EOS set {:?}. The rendered \
             official-template prompt must never begin with an end-of-text token.\n{}",
                probe.model_id,
                probe.eos_ids,
                describe(dir, &probe),
            );
            if probe
                .legacy_prompt_ids
                .first()
                .is_some_and(|t| probe.eos_ids.contains(t))
            {
                legacy_bug_seen += 1;
            }
        }
        assert!(
            checked > 0,
            "found {} cached Qwen snapshots but none had a loadable chat template + tokenizer",
            dirs.len()
        );
        assert!(
            legacy_bug_seen > 0,
            "no cached Qwen snapshot reproduces the generation_config.json BOS trap any more \
         (checked {checked}). Either the corpus changed or this guard has stopped exercising \
         the bug it was written for; re-derive it instead of deleting it."
        );
    }

    #[test]
    fn cached_gemma_snapshots_still_start_a_prompt_with_bos() {
        let dirs = snapshots_or_panic("gemma", "cached_gemma_snapshots_still_start_a_prompt_with_bos");
        if dirs.is_empty() {
            eprintln!(
                "cached_gemma_snapshots_still_start_a_prompt_with_bos: SKIP \
                 ({BOS_GUARD_ALLOW_SKIP}=1) no cached wgpu-servable models--*gemma* snapshot"
            );
            return;
        }
        let msgs = probe_messages();
        let mut checked = 0usize;
        for dir in &dirs {
            let probe = match probe_prompt_head(dir, &msgs) {
                Ok(p) => p,
                Err(err) => {
                    eprintln!("skip {} ({err:#})", dir.display());
                    continue;
                }
            };
            eprintln!("{}", describe(dir, &probe));
            checked += 1;
            let first = probe.prompt_ids[0];
            assert!(
                !probe.eos_ids.contains(&first),
                "{}: first prompt token {first} is in the EOS set {:?}\n{}",
                probe.model_id,
                probe.eos_ids,
                describe(dir, &probe),
            );
            assert_eq!(
                probe.bos_declaration,
                BosDeclaration::Token("<bos>".into()),
                "{}: gemma tokenizers declare bos_token \"<bos>\"; the fix must not have taken \
             the BOS away from the family that needs it\n{}",
                probe.model_id,
                describe(dir, &probe),
            );
            assert_eq!(
                probe.bos_token_id,
                Some(2),
                "{}: gemma BOS is token id 2\n{}",
                probe.model_id,
                describe(dir, &probe),
            );
            assert_eq!(
                first,
                2,
                "{}: the gemma chat template emits {{{{ bos_token }}}} itself, so the rendered \
             prompt must already begin with id 2 and must not be double-prefixed\n{}",
                probe.model_id,
                describe(dir, &probe),
            );
            assert_ne!(
                probe.prompt_ids.get(1).copied(),
                Some(2),
                "{}: BOS was prepended on top of the template's own BOS\n{}",
                probe.model_id,
                describe(dir, &probe),
            );
        }
        assert!(checked > 0, "no cached gemma snapshot was checkable");
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum HeadOrigin {
        NotEos,
        CheckpointDualRoleBos,
        EnginePrepended,
        Unexplained,
    }

    fn head_origin(p: &PromptHeadProbe) -> HeadOrigin {
        let first = p.prompt_ids[0];
        if !p.eos_ids.contains(&first) {
            return HeadOrigin::NotEos;
        }
        if p.bos_token_id == Some(first) {
            return HeadOrigin::EnginePrepended;
        }
        let declared_bos_twice = matches!(p.bos_declaration, BosDeclaration::Token(_))
            && p.generation_config_bos == Some(first);
        if p.bos_token_id.is_none() && declared_bos_twice {
            HeadOrigin::CheckpointDualRoleBos
        } else {
            HeadOrigin::Unexplained
        }
    }

    #[test]
    fn an_eos_member_prompt_head_is_only_ever_the_checkpoints_own_declared_bos() {
        let dirs = snapshots_or_panic(
            "models--",
            "an_eos_member_prompt_head_is_only_ever_the_checkpoints_own_declared_bos",
        );
        if dirs.is_empty() {
            eprintln!(
                "an_eos_member_prompt_head_is_only_ever_the_checkpoints_own_declared_bos: SKIP \
                 ({BOS_GUARD_ALLOW_SKIP}=1) no cached wgpu-servable hub snapshots"
            );
            return;
        }
        eprintln!(
            "an EOS-member prompt head is a defect only when this engine put it there. \
             poolside/Laguna-XS-2.1-NVFP4 declares id 2 as bos_token_id AND as a member of \
             eos_token_id [2, 24], and its own chat_template.jinja opens by emitting that \
             token: the earlier form of this guard \
             (every_wgpu_servable_snapshot_has_a_non_eos_prompt_head) asserted a property that \
             checkpoint violates by design, so it was unconditionally red and could never fire \
             for the hazard it was written for."
        );
        let msgs = probe_messages();
        let mut checked = 0usize;
        let mut dual_role = 0usize;
        let mut legacy_traps = 0usize;
        let mut bad: Vec<String> = Vec::new();
        for dir in &dirs {
            let Ok(probe) = probe_prompt_head(dir, &msgs) else {
                continue;
            };
            checked += 1;
            let first = probe.prompt_ids[0];
            let origin = head_origin(&probe);
            eprintln!(
                "{:<48} first={first:<8} eos={:?} bos={:?} generation_config_bos={:?} {origin:?}",
                probe.model_id, probe.eos_ids, probe.bos_token_id, probe.generation_config_bos
            );
            match origin {
                HeadOrigin::NotEos => {}
                HeadOrigin::CheckpointDualRoleBos => {
                    dual_role += 1;
                    assert_ne!(
                        probe.prompt_ids.get(1).copied(),
                        Some(first),
                        "{}: this checkpoint declares id {first} as both its BOS and an EOS \
                     member and its template already emits it; the engine must not have added a \
                     second copy on top\n{}",
                        probe.model_id,
                        describe(dir, &probe),
                    );
                }
                HeadOrigin::EnginePrepended | HeadOrigin::Unexplained => {
                    bad.push(format!("[{origin:?}]\n{}", describe(dir, &probe)));
                }
            }
            if let Some(gen_bos) = probe.generation_config_bos {
                if probe.eos_ids.contains(&gen_bos) {
                    legacy_traps += 1;
                    assert_ne!(
                        probe.bos_token_id,
                        Some(gen_bos),
                        "{}: bos_id_from_dir() reads bos_token_id {gen_bos} straight out of \
                     generation_config.json and this checkpoint also lists {gen_bos} in its EOS \
                     set, so adopting it splices a stop id into position 0 of a token stream \
                     every downstream consumer reads by id. The serving path must resolve BOS \
                     through prompt_bos_id(), which drops an EOS-member BOS.\n{}",
                        probe.model_id,
                        describe(dir, &probe),
                    );
                }
            }
        }
        eprintln!(
            "checked {checked} canonicalized snapshots: {dual_role} declare one id as both BOS \
             and EOS, {legacy_traps} carry a generation_config.json bos_token_id that is an EOS \
             member"
        );
        assert!(
            bad.is_empty(),
            "{} of {checked} cached chat snapshots open on an EOS member that the checkpoint \
         does not itself declare as BOS. EnginePrepended = prompt_bos_id() adopted an id from \
         the EOS set and this engine put it at position 0; Unexplained = the template emits an \
         EOS member the checkpoint never declares as BOS in tokenizer_config.json AND \
         generation_config.json.\n{}",
            bad.len(),
            bad.join("\n"),
        );
        assert!(
            checked > 0,
            "no cached snapshot had a loadable chat template"
        );
    }

    const CUDA_PROMPT_LOOPS: [&str; 2] = [
        "src/oapi/chat_engine/gemma4_loop.rs",
        "src/oapi/chat_engine/gemma4_moe_loop.rs",
    ];
    const SHARED_POSITIONAL_RULE: &str = "splice_bos_at_position_0_only";
    const NAMED_RATIONALE: &str = "BOS_IS_A_POSITION_0_ROLE_AND_THE_SAME_ID_LATER_IS_A_STOP";
    const SERVING_CALL_SHAPE: &str = "bos_token_id, eos_ids)";
    const CUDA_PROMPT_BUILDING_SITES: usize = 4;

    #[test]
    fn every_cuda_bos_splice_in_the_gemma_loops_goes_through_the_shared_positional_rule() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut raw_splices: Vec<String> = Vec::new();
        let mut call_sites: Vec<String> = Vec::new();
        let mut rule_definitions = 0usize;
        let mut rationale_present = false;
        for rel in CUDA_PROMPT_LOOPS {
            let path = root.join(rel);
            let src = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
            rationale_present |= src.contains(NAMED_RATIONALE);
            for (n, line) in src.lines().enumerate() {
                let at = format!("{rel}:{}: {}", n + 1, line.trim());
                if line.contains(&format!("fn {SHARED_POSITIONAL_RULE}(")) {
                    rule_definitions += 1;
                }
                if line.contains(&format!("{SHARED_POSITIONAL_RULE}(&mut"))
                    && line.contains(SERVING_CALL_SHAPE)
                {
                    call_sites.push(at.clone());
                }
                if line.contains(".insert(0, bos_token_id)") {
                    raw_splices.push(at);
                }
            }
        }
        assert_eq!(
            rule_definitions, 1,
            "the CUDA gemma loops must share exactly one BOS rule named {SHARED_POSITIONAL_RULE}; \
             found {rule_definitions} definitions"
        );
        assert!(
            rationale_present,
            "the const {NAMED_RATIONALE} carries why the rule is positional. \
             scripts/strip-comments.py deletes // and /// alike, so a name is the only durable \
             place for it; if the const is gone the reasoning is gone"
        );
        assert_eq!(
            raw_splices.len(),
            1,
            "BOS may be inserted at position 0 in exactly one place, inside {SHARED_POSITIONAL_RULE}. \
             A per-call-site `insert(0, bos_token_id)` cannot see whether the id is also an EOS \
             member, and that is how four CUDA sites spliced a stop id into position 0 of prompts \
             for checkpoints that declare one id in both roles.\n{}",
            raw_splices.join("\n")
        );
        assert!(
            raw_splices[0].starts_with(CUDA_PROMPT_LOOPS[0]),
            "the one raw splice must be the one inside {SHARED_POSITIONAL_RULE}, in \
             {}, not {}",
            CUDA_PROMPT_LOOPS[0],
            raw_splices[0]
        );
        assert_eq!(
            call_sites.len(),
            CUDA_PROMPT_BUILDING_SITES,
            "expected {CUDA_PROMPT_BUILDING_SITES} serving sites -- the ones handed the engine's \
             `{SERVING_CALL_SHAPE}` pair: run_gemma4_via_engine, run_sampling_gemma4, \
             run_sampling_gemma4_e4b, run_sampling_gemma4_moe -- to call \
             {SHARED_POSITIONAL_RULE}; found {}:\n{}",
            call_sites.len(),
            call_sites.join("\n")
        );
    }
}
