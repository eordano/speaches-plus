#![cfg(any(feature = "wgpu", feature = "cuda"))]
#![allow(dead_code)]

mod official_template;

use official_template::OfficialTemplate;

struct XrefCase {
    prompt: &'static str,
    prompt_ids: &'static [u32],
    cuda_stream: &'static [u32],
}

const STEPS: usize = 32;
const EOS: [u32; 2] = [1, 106];

const FROZEN_BASIS: &str = "pack-quantized";

const CASES: [XrefCase; 4] = [
    XrefCase {
        prompt: "What is the capital of France, and what river runs through it?",
        prompt_ids: &[
            2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001, 236764, 532, 1144, 8858, 8784,
            1343, 625, 236881, 106, 107, 105, 4368, 107,
        ],
        cuda_stream: &[
            818, 5279, 529, 7001, 563, 5213, 50429, 125546, 532, 506, 8858, 600, 8784, 1343, 625,
            563, 506, 5213, 159003, 84750, 106,
        ],
    },
    XrefCase {
        prompt: "Summarize in one sentence why the sky is blue.",
        prompt_ids: &[
            2, 105, 2364, 107, 160773, 969, 528, 886, 13315, 3217, 506, 7217, 563, 3730, 236761,
            106, 107, 105, 4368, 107,
        ],
        cuda_stream: &[
            818, 7217, 7412, 3730, 1547, 529, 496, 20284, 2760, 121707, 19389, 236764, 1298, 506,
            10824, 236789, 236751, 11661, 141891, 1826, 20532, 3730, 57583, 529, 26808, 919, 11974,
            1082, 4890, 2604, 57583, 236761,
        ],
    },
    XrefCase {
        prompt: "Write a haiku about mountains.",
        prompt_ids: &[
            2, 105, 2364, 107, 6974, 496, 678, 20517, 1003, 13254, 236761, 106, 107, 105, 4368, 107,
        ],
        cuda_stream: &[
            55421, 47941, 74942, 236764, 107, 174103, 506, 14958, 607, 54530, 23338, 236764, 107,
            136657, 236764, 162038, 2473, 236761, 106,
        ],
    },
    XrefCase {
        prompt: "List three uses for a paperclip.",
        prompt_ids: &[
            2, 105, 2364, 107, 1613, 1806, 6178, 573, 496, 3627, 13758, 236761, 106, 107, 105,
            4368, 107,
        ],
        cuda_stream: &[
            8291, 659, 1806, 3364, 6178, 573, 496, 3627, 13758, 236787, 108, 236770, 236761, 5213,
            134942, 13009, 3075, 53121, 1174, 563, 1061, 1346, 6672, 532, 7334, 1161, 236761, 107,
            236778, 236761, 5213, 2205,
        ],
    },
];

const FREEZE_HOWTO: &str = "CUDA reference streams are EMPTY (the E4B store pins were GC'd when this \
fixture was authored). To freeze them, on a CUDA box with an E4B snapshot present run:\n\
  NV_E4B_XREF_FREEZE=1 NV_DETERMINISTIC=1 NV_E4B_DIR=<snapshot> \\\n\
  NVK_LANE=<lane> NVK_PKG=nv-models NVK_FEATURES=cuda \\\n\
  rust/scripts/nvk.sh test --release --test e4b_cuda_reference_divergence -- --ignored --nocapture\n\
then paste the printed prompt_ids/cuda_stream arrays into CASES and the printed \
basis into FROZEN_BASIS in tests/e4b_cuda_reference_divergence.rs.";

fn frozen() -> bool {
    !FROZEN_BASIS.is_empty()
        && CASES
            .iter()
            .all(|c| !c.prompt_ids.is_empty() && !c.cuda_stream.is_empty())
}

fn templated(dir: &std::path::Path, user: &str) -> String {
    let rendered = OfficialTemplate::load(dir).render_user(user);
    assert!(
        rendered.starts_with("<bos>"),
        "official template must emit BOS itself: {rendered:?}"
    );
    rendered
}

fn e4b_snapshot_dir() -> std::path::PathBuf {
    match std::env::var("NV_E4B_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => {
            let home = std::env::var("HOME").unwrap();
            let base = std::path::PathBuf::from(home)
                .join(".cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots");
            std::fs::read_dir(&base)
                .expect("set NV_E4B_DIR or hydrate the hub snapshot")
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.join("config.json").exists())
                .expect("no snapshot with config.json")
        }
    }
}

fn basis_of(loader: &nv_weights::WeightLoader) -> &'static str {
    if loader.has("model.language_model.layers.0.mlp.down_proj.weight_packed") {
        "pack-quantized"
    } else {
        "dense-bf16"
    }
}

fn print_frozen_slice(name: &str, ids: &[u32]) {
    println!(
        "        {name}: &[{}],",
        ids.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn e4b_cuda_reference_freeze() {
    if std::env::var("NV_E4B_XREF_FREEZE").ok().as_deref() != Some("1") {
        eprintln!("SKIP: set NV_E4B_XREF_FREEZE=1 (CUDA freeze arm of the xref fixture)");
        return;
    }
    assert_eq!(
        std::env::var("NV_DETERMINISTIC").ok().as_deref(),
        Some("1"),
        "freeze must run with NV_DETERMINISTIC=1 so the reference stream is reproducible"
    );
    let device = candle_core::Device::new_cuda(0).expect("cuda device required to freeze");
    let dir = e4b_snapshot_dir();
    eprintln!("E4B checkpoint: {}", dir.display());
    let cfg = nv_models::gemma4::Gemma4Config::from_hf_json_file(&dir.join("config.json"))
        .expect("config");
    assert!(
        cfg.has_per_layer_embeddings(),
        "not an E4B checkpoint: {dir:?}"
    );
    let loader = nv_weights::WeightLoader::open_dir(&dir, &device).expect("weights");
    let basis = basis_of(&loader);
    eprintln!("checkpoint basis: {basis}");
    let model =
        nv_models::gemma4_e4b::Gemma4E4b::from_loader(cfg, &loader, &device).expect("load E4b");

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    println!("FROZEN_BASIS: \"{basis}\"");
    for case in &CASES {
        let ids: Vec<u32> = tok
            .encode(templated(&dir, case.prompt).as_str(), false)
            .expect("encode")
            .get_ids()
            .to_vec();
        let stream = model
            .generate(&ids, STEPS, &EOS)
            .expect("cuda greedy generate");
        assert!(
            stream.len() >= 8,
            "prompt {:?}: CUDA stream too short ({}) to serve as a reference",
            case.prompt,
            stream.len()
        );
        println!("    XrefCase {{");
        println!("        prompt: {:?},", case.prompt);
        print_frozen_slice("prompt_ids", &ids);
        print_frozen_slice("cuda_stream", &stream);
        println!("    }},");
        eprintln!(
            "[freeze] {:?} -> {} tok: {:?}",
            case.prompt,
            stream.len(),
            tok.decode(&stream, false).unwrap_or_default()
        );
    }
    println!("paste the blocks above into CASES and FROZEN_BASIS, then re-run the replay arm");
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore]
fn e4b_wgpu_divergence_vs_cuda_reference() {
    if std::env::var("NV_E4B_XREF_TEST").ok().as_deref() != Some("1") {
        eprintln!("SKIP: set NV_E4B_XREF_TEST=1 (wgpu replay arm of the xref fixture)");
        return;
    }
    let ctx = match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("SKIP: no wgpu adapter ({e})");
            return;
        }
    };
    eprintln!("adapter: {}", ctx.summary());
    assert!(frozen(), "{FREEZE_HOWTO}");

    let dir = e4b_snapshot_dir();
    eprintln!("E4B checkpoint: {}", dir.display());
    let config = nv_models::gemma4::Gemma4Config::from_hf_json_file(&dir.join("config.json"))
        .expect("config");
    assert!(
        config.has_per_layer_embeddings(),
        "not an E4B checkpoint: {dir:?}"
    );
    let vocab = config.vocab_size;
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("weights");
    let basis = basis_of(&loader);
    assert_eq!(
        basis, FROZEN_BASIS,
        "checkpoint basis mismatch: the CUDA reference was frozen on {FROZEN_BASIS}, this \
         snapshot is {basis} — same-checkpoint replay is the whole point of the watch"
    );

    let need = CASES
        .iter()
        .map(|c| c.prompt_ids.len() + c.cuda_stream.len() + 1)
        .max()
        .unwrap();
    let max_seq = need.next_multiple_of(64).max(256);
    let t0 = std::time::Instant::now();
    let mut m = if std::env::var("NV_E4B_XREF_STREAM").ok().as_deref() == Some("1") {
        nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu::from_loader(config.clone(), &loader, max_seq)
            .expect("Gemma4E4bWgpu::from_loader")
    } else {
        let host = nv_models::gemma4_e4b_wgpu::e4b_host_weights_from_loader(&config, &loader)
            .expect("host weights");
        nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu::new(config.clone(), &host, max_seq)
            .expect("Gemma4E4bWgpu::new")
    };
    eprintln!(
        "wgpu model up in {:.1}s, max_seq {max_seq}",
        t0.elapsed().as_secs_f32()
    );
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");

    let mut total_scored = 0usize;
    let mut total_diverged = 0usize;
    for (pi, case) in CASES.iter().enumerate() {
        assert!(
            case.cuda_stream.len() >= 8,
            "frozen stream for prompt {pi} too short to score"
        );
        assert!(
            case.prompt_ids
                .iter()
                .chain(case.cuda_stream)
                .all(|&t| (t as usize) < vocab),
            "frozen fixture for prompt {pi} carries out-of-vocab ids"
        );

        m.reset();
        let mut pred = 0u32;
        for &t in case.prompt_ids {
            pred = m.decode_step(t).expect("wgpu prefill step");
        }
        let mut tf_div = 0usize;
        let mut tf_first: Option<usize> = None;
        for (i, &want) in case.cuda_stream.iter().enumerate() {
            assert!(
                (pred as usize) < vocab,
                "wgpu produced out-of-vocab token {pred}"
            );
            if pred != want {
                tf_div += 1;
                tf_first.get_or_insert(i);
            }
            if i + 1 < case.cuda_stream.len() {
                pred = m.decode_step(want).expect("wgpu teacher-forced step");
            }
        }

        m.reset();
        let mut next = 0u32;
        for &t in case.prompt_ids {
            next = m.decode_step(t).expect("wgpu prefill step");
        }
        let mut own: Vec<u32> = Vec::with_capacity(case.cuda_stream.len());
        for _ in 0..case.cuda_stream.len() {
            own.push(next);
            if EOS.contains(&next) {
                break;
            }
            next = m.decode_step(next).expect("wgpu free-run step");
        }
        let fr_first = own
            .iter()
            .zip(case.cuda_stream)
            .position(|(a, b)| a != b)
            .or_else(|| (own.len() < case.cuda_stream.len()).then_some(own.len()));
        let fr_div = own
            .iter()
            .zip(case.cuda_stream)
            .filter(|(a, b)| a != b)
            .count()
            + case.cuda_stream.len().saturating_sub(own.len());

        total_scored += case.cuda_stream.len();
        total_diverged += tf_div;
        eprintln!(
            "[xref p{pi}] teacher-forced {tf_div}/{} diverged, first {tf_first:?}; \
             free-run {fr_div}/{} diverged, first {fr_first:?}",
            case.cuda_stream.len(),
            case.cuda_stream.len(),
        );
        eprintln!(
            "[xref p{pi}] cuda: {:?}",
            tok.decode(case.cuda_stream, false).unwrap_or_default()
        );
        eprintln!(
            "[xref p{pi}] wgpu: {:?}",
            tok.decode(&own, false).unwrap_or_default()
        );
    }

    eprintln!(
        "XREF-SUMMARY {total_diverged}/{total_scored} teacher-forced positions diverged from \
         the CUDA reference across {} external chat prompts on adapter {:?}. This is a watch, \
         not a gate: no divergence threshold is asserted.",
        CASES.len(),
        ctx.summary()
    );
    assert!(
        total_scored >= 32,
        "too few scored positions for a meaningful watch: {total_scored}"
    );
}
