#![allow(dead_code)]

pub const TILED_PROMPT_PARAGRAPH_SAME_TEXT_AS_THE_CTX_DEPTH_LADDER_UPPER_BRACKET: &str =
    "Polynesian wayfinders crossed thousands of miles of open ocean by reading star \
     paths, swell patterns, and the flight of birds, long before the magnetic compass reached \
     Europe. Their voyaging canoes carried taro, breadfruit, and chickens between island chains \
     separated by weeks of sailing. ";

pub const FREEFORM_CORPUS_ENV_SAME_KNOB_AS_THE_DEPTH_LADDER: &str = "NV_Q38_DEPTH_PROMPT_CORPUS";
pub const FREEFORM_CORPUS_DEFAULT_THE_ADAK_CAMPAIGN_PG1342: &str = ".cache/ppl-corpus/pg1342.txt";

pub const VARIANT_RENAME_PAIRS_INDEX_0_IS_THE_ORIGINAL_SO_VARIANT_0_STAYS_BITWISE_STABLE:
    [(&str, &str, &str, &str); 4] = [
    ("compute_window_mean", "rolling_mean", "values", "samples"),
    ("accumulate_bucket_sum", "bucket_total", "entries", "items"),
    ("scan_prefix_products", "prefix_prod", "inputs", "xs"),
    ("fold_pairwise_max", "pair_max", "series", "points"),
];

pub fn agentic_transcript_text(blocks: usize) -> String {
    agentic_transcript_text_variant(blocks, 0)
}

pub fn agentic_transcript_text_variant(blocks: usize, variant: usize) -> String {
    let (old_fn, new_fn, old_arg, new_arg) =
        VARIANT_RENAME_PAIRS_INDEX_0_IS_THE_ORIGINAL_SO_VARIANT_0_STAYS_BITWISE_STABLE
            [variant % 4];
    let base = 1000 * variant;
    let mut s = format!(
        "An automated coding agent applies the same refactor to every module in a list. For \
         each module it reads the file, then writes the file back with the function renamed \
         from `{old_fn}` to `{new_fn}` and the parameter `{old_arg}` renamed to \
         `{new_arg}`. The transcript so far:\n\n",
    );
    for j in 0..blocks {
        let i = base + j;
        s.push_str(&format!(
            "<tool_call>{{\"name\": \"read_file\", \"arguments\": {{\"path\": \
             \"src/metrics/module_{i:03}.rs\"}}}}</tool_call>\n<tool_result>pub fn \
             {old_fn}({old_arg}: &[f64], width: usize) -> Vec<f64> {{\n    let mut out \
             = Vec::with_capacity({old_arg}.len());\n    let mut acc = 0.0f64;\n    for (i, v) in \
             {old_arg}.iter().enumerate() {{\n        acc += v;\n        if i >= width {{ acc -= \
             {old_arg}[i - width]; }}\n        let n = (i + 1).min(width);\n        out.push(acc \
             / n as f64);\n    }}\n    out\n}}</tool_result>\n<tool_call>{{\"name\": \
             \"write_file\", \"arguments\": {{\"path\": \"src/metrics/module_{i:03}.rs\", \
             \"content\": \"pub fn {new_fn}({new_arg}: &[f64], width: usize) -> Vec<f64> \
             {{\\n    let mut out = Vec::with_capacity({new_arg}.len());\\n    let mut acc = \
             0.0f64;\\n    for (i, v) in {new_arg}.iter().enumerate() {{\\n        acc += v;\\n  \
             if i >= width {{ acc -= {new_arg}[i - width]; }}\\n        let n = (i + \
             1).min(width);\\n        out.push(acc / n as f64);\\n    }}\\n    \
             out\\n}}\"}}}}</tool_call>\n<tool_result>ok</tool_result>\n"
        ));
    }
    s
}

pub fn tokens_of(tok: &tokenizers::Tokenizer, text: &str, want: usize, label: &str) -> Vec<u32> {
    let ids: Vec<u32> = tok
        .encode(text, false)
        .expect("encode class prompt")
        .get_ids()
        .to_vec();
    assert!(
        ids.len() >= want,
        "{label} prompt tokenized to {} tokens, below the requested {want}",
        ids.len()
    );
    ids[..want].to_vec()
}

pub fn three_class_prompts(
    tok: &tokenizers::Tokenizer,
    want: usize,
) -> Vec<(&'static str, Vec<u32>)> {
    three_class_prompt_variants(tok, want, 0)
}

pub fn three_class_prompt_variants(
    tok: &tokenizers::Tokenizer,
    want: usize,
    variant: usize,
) -> Vec<(&'static str, Vec<u32>)> {
    let agentic = tokens_of(
        tok,
        &agentic_transcript_text_variant(4 + want / 128, variant),
        want,
        "agentic",
    );
    let tiled: Vec<u32> = {
        let seed: Vec<u32> = tok
            .encode(TILED_PROMPT_PARAGRAPH_SAME_TEXT_AS_THE_CTX_DEPTH_LADDER_UPPER_BRACKET, false)
            .expect("encode paragraph")
            .get_ids()
            .to_vec();
        let rot = (variant * 17) % seed.len();
        (0..want).map(|i| seed[(i + rot) % seed.len()]).collect()
    };
    let freeform = {
        let path = std::env::var(FREEFORM_CORPUS_ENV_SAME_KNOB_AS_THE_DEPTH_LADDER)
            .unwrap_or_else(|_| {
                format!(
                    "{}/{}",
                    std::env::var("HOME").expect("HOME"),
                    FREEFORM_CORPUS_DEFAULT_THE_ADAK_CAMPAIGN_PG1342
                )
            });
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: free-prose corpus {path} \
                 unreadable ({e}); set {FREEFORM_CORPUS_ENV_SAME_KNOB_AS_THE_DEPTH_LADDER}"
            )
        });
        let ids: Vec<u32> = tok
            .encode(text.as_str(), false)
            .expect("encode class prompt")
            .get_ids()
            .to_vec();
        let start = variant * want;
        assert!(
            ids.len() >= start + want,
            "freeform corpus holds {} tokens, below variant {variant}'s slice \
             [{start}..{}]; a shorter corpus would silently reuse variant 0's text",
            ids.len(),
            start + want
        );
        ids[start..start + want].to_vec()
    };
    vec![("agentic", agentic), ("tiled", tiled), ("freeform", freeform)]
}

pub fn fnv1a64_over_u32_le(tokens: &[u32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}
