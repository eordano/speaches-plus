use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const NORMAL_BITCAST_WGSL: &str = "0x3c000000u";

const NORMAL_BITCAST_RUST: &str = "0x3c00_0000";

const SUBNORMAL_STEP_LITERAL: &str = "0.001953125";

const SUBNORMAL_STEP_CONST: &str = "UE4M3_SUBNORMAL_STEP";

const SUBNORMAL_BRANCH_MUST_SIT_WITHIN_LINES_OF_THE_BITCAST: usize = 4;

const KNOWN_DECODERS: [&str; 3] = [
    "nv-kernels/src/wgpu_backend/kernels/gemv_nvfp4_lin.rs",
    "nv-kernels/wgsl/gemv_nvfp4.wgsl",
    "nv-kernels/wgsl/gemv_nvfp4_v2.wgsl",
];

const WGSL_ENTRY_POINT: &str = "@compute";

const SHADER_TEXT_STILL_ASSEMBLED_IN_RUST: [(&str, &str); 14] = [
    (
        "nv-kernels/src/wgpu_backend/dequant.rs",
        "the gpu_decode_* probes wrap a caller-supplied decode expression in a one-entry map shader",
    ),
    (
        "nv-kernels/src/wgpu_backend/device.rs",
        "SG_PROBE_SRC asks the adapter its subgroup size before any kernel can be specialised",
    ),
    (
        "nv-kernels/src/wgpu_backend/dispatch.rs",
        "the coop-matrix guard compiles a do-nothing entry to test the guard, not to run work",
    ),
    (
        "nv-kernels/src/wgpu_backend/kernels/flash_decode.rs",
                "fold-variant stage1 sources are composed per hd_max/sg/fold/tile over the extracted \
         flash_decode_fold_*.wgsl, and the shift-decode twins are anchored rewrites of stock \
         entries extracted from flash_decode.wgsl; the only \"@compute\" in Rust text is the \
         search needle that locates each stock entry attribute",
    ),
    (
        "nv-kernels/src/wgpu_backend/kernels/gemm_bf16_small_m.rs",
        "workgroup size is baked per variant",
    ),
    (
        "nv-kernels/src/wgpu_backend/kernels/gemm_coop_f16.rs",
        "workgroup size and tile shape are baked per variant",
    ),
    (
        "nv-kernels/src/wgpu_backend/kernels/gemm_w4a16_small_m.rs",
        "workgroup size is baked per variant",
    ),
    (
        "nv-kernels/src/wgpu_backend/kernels/gemv_bf16.rs",
        "workgroup size and fold factor are baked per variant over the extracted gemv_bf16_sg.wgsl",
    ),
    (
        "nv-kernels/src/wgpu_backend/kernels/gemv_w4a16.rs",
        "workgroup size and group size are baked per variant",
    ),
    (
        "nv-models/src/gemma4_e4b_wgpu.rs",
        "the multi-token paths unroll per row count over extracted e4b_*.wgsl prologues",
    ),
    (
        "nv-models/src/gemma4_moe_wgpu.rs",
        "the prefill body unrolls per tile over the extracted g4m_prefill_head.wgsl",
    ),
    (
        "nv-models/src/gemma4_wgpu.rs",
        "mk_bf16_source and mk_q8_source unroll per token count over the extracted g4w_mk_params.wgsl",
    ),
    (
        "nv-models/src/qwen3_5_dense_wgpu.rs",
        "gemm_mk_source unrolls per row count",
    ),
    (
        "nv-models/src/qwen3_5_moe_wgpu.rs",
        "pf_gemm_bf16_mrow_source unrolls its accumulator array and workgroup reduction \
         tile per token-row count",
    ),
];

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("nv-kernels sits in the crates directory")
        .to_path_buf()
}

fn collect(dir: &Path, ext: &str, out: &mut BTreeMap<String, String>) {
    let root = crates_dir();
    for e in std::fs::read_dir(dir).unwrap_or_else(|_| panic!("{} is readable", dir.display())) {
        let p = e.expect("dir entry").path();
        if p.is_dir() {
            collect(&p, ext, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some(ext) {
            let label = p
                .strip_prefix(&root)
                .expect("every source sits under crates/")
                .to_string_lossy()
                .replace('\\', "/");
            out.insert(label, std::fs::read_to_string(&p).expect("source reads"));
        }
    }
}

fn rust_shader_roots() -> [PathBuf; 2] {
    let c = crates_dir();
    let roots = [c.join("nv-kernels/src"), c.join("nv-models/src")];
    for r in &roots {
        assert!(
            r.is_dir(),
            "{} is missing, so this census would silently guard less than it claims: a shader \
             assembled in Rust is exactly the copy the 382aa6dd1 fix was able to miss",
            r.display()
        );
    }
    roots
}

fn sources() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    collect(&crates_dir().join("nv-kernels/wgsl"), "wgsl", &mut out);
    for r in rust_shader_roots() {
        collect(&r, "rs", &mut out);
    }
    out
}

fn lines_matching(src: &str, needles: &[&str]) -> Vec<usize> {
    src.lines()
        .enumerate()
        .filter(|(_, l)| needles.iter().any(|n| l.contains(n)))
        .map(|(i, _)| i)
        .collect()
}

#[test]
fn every_ue4m3_decode_handles_the_subnormal_band() {
    let mut found: Vec<String> = Vec::new();
    let mut bad: Vec<String> = Vec::new();
    for (name, src) in sources() {
        let sites = lines_matching(&src, &[NORMAL_BITCAST_WGSL, NORMAL_BITCAST_RUST]);
        if sites.is_empty() {
            continue;
        }
        found.push(name.clone());
        let branches = lines_matching(&src, &[SUBNORMAL_STEP_LITERAL, SUBNORMAL_STEP_CONST]);
        for s in sites {
            let guarded = branches.iter().any(|b| {
                b.abs_diff(s) <= SUBNORMAL_BRANCH_MUST_SIT_WITHIN_LINES_OF_THE_BITCAST
            });
            if !guarded {
                bad.push(format!("{name}:{}", s + 1));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "these sites reconstruct a ue4m3 scale by adding {NORMAL_BITCAST_WGSL} to a shifted \
         mantissa, which yields (1 + m/8) * 2^-7 -- the NORMAL formula -- for a biased exponent \
         of 0, where e4m3fn defines m * 2^-9. No {SUBNORMAL_STEP_LITERAL} or \
         {SUBNORMAL_STEP_CONST} sits within \
         {SUBNORMAL_BRANCH_MUST_SIT_WITHIN_LINES_OF_THE_BITCAST} lines of them, so the smallest 8 \
         scale codes decode ~15x too large against checkpoint weight bytes. The window is what \
         makes this bite: a mention of the constant elsewhere in the same file -- a unit test \
         asserting on it, a second decoder that is correct -- must not vouch for this site. \
         382aa6dd1 fixed this in dequant.wgsl and gemv_nvfp4.wgsl and missed these: {bad:?}"
    );
    assert_eq!(
        found, KNOWN_DECODERS,
        "the set of sources reconstructing a ue4m3 scale changed. A new one must carry the \
         subnormal branch before it is added here, and a removed one means this census is \
         guarding less than it claims"
    );
}

#[test]
fn the_subnormal_step_is_two_to_the_minus_nine() {
    let step: f32 = SUBNORMAL_STEP_LITERAL.parse().expect("literal parses");
    assert_eq!(
        step,
        (2.0f32).powi(-9),
        "e4m3fn encodes a biased exponent of 0 as m * 2^-9; any other step silently rescales \
         the smallest 8 codes"
    );
    for m in 0u32..8 {
        let decoded = m as f32 * step;
        let wrong = f32::from_bits((m << 20) + 0x3c00_0000);
        assert!(
            m == 0 || decoded < wrong,
            "code {m}: the subnormal value {decoded} must be below what the normal formula \
             {wrong} would have produced, or the branch is not doing anything"
        );
    }
}

#[test]
fn every_shader_assembled_in_rust_is_listed_so_none_can_hide_from_the_decode_census() {
    let listed: Vec<&str> = SHADER_TEXT_STILL_ASSEMBLED_IN_RUST
        .iter()
        .map(|(p, _)| *p)
        .collect();
    let mut sorted = listed.clone();
    sorted.sort_unstable();
    assert_eq!(
        listed, sorted,
        "keep SHADER_TEXT_STILL_ASSEMBLED_IN_RUST sorted so it compares against a walk directly"
    );
    let found: Vec<String> = sources()
        .into_iter()
        .filter(|(name, src)| name.ends_with(".rs") && src.contains(WGSL_ENTRY_POINT))
        .map(|(name, _)| name)
        .collect();
    assert_eq!(
        found, listed,
        "a WGSL entry point declared inside a Rust string is invisible to any audit that walks \
         wgsl/ -- which is how gemv_nvfp4_v2 kept decoding the smallest 8 scale codes ~15x too \
         large after 382aa6dd1 fixed its siblings. A fixed shader belongs in nv-kernels/wgsl \
         behind include_str!; only text whose shape depends on a specialisation constant may be \
         assembled here, and then its fixed prologue still belongs in a .wgsl file the generator \
         composes. Whatever is left must be listed with the reason it cannot be a file"
    );
}
