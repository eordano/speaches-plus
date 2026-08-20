#![cfg(feature = "wgpu")]

mod common;
use common::pack;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use common::LcgOddSeedShift33SignedUnitG4w as Lcg;

const TAG: &str = "g4w:quant_row_pk";
const ENTRY: &str = "g4w_quant_row_pk";
const BLOCK_FN: &str = "g4w_qz_block";

const WHY: &str = "g4w_quant_row_pk is the activation quantizer of the nvfp4 path in the Gemma-4 \
     dense graph: every nvfp4 GEMV in the model is preceded by one dispatch of it, and its output \
     -- packed E2M1 nibbles plus one UE4M3 scale byte per sixteen elements -- is the entire input \
     the GEMV sees. g4w_qz_block is the per-block body it calls. Neither is named by any test in \
     the workspace. It is also the one kernel in this graph whose output is not a number but an \
     ENCODING: a wrong nibble is a wrong weight, a wrong scale byte is a wrong sixteen of them, \
     and nothing downstream of it can report a tolerance that would separate the two.";

const ORACLE_IS_NOT_THE_KERNEL: &str = "The reference is the definition of nvfp4 row \
     quantization, computed on the host in f64 and compared BIT EXACTLY because the output is an \
     encoding: the block scale is the global scale times the block's largest magnitude over six, \
     rounded to UE4M3; each element is that element divided by the DECODED scale, clamped to the \
     E2M1 range, and rounded to the nearest E2M1 code with ties going to the smaller magnitude. \
     It is not a call into gemv_nvfp4::quantize_rows and not a second copy of the WGSL, and it \
     never consults the shader's own encoder helpers.";

const THE_SCALE_EXPONENT_MUST_REACH_ZERO: &str = "The screen asserts, block by block, that the \
     corpus actually produces UE4M3 scale bytes with a BIASED EXPONENT OF ZERO -- the subnormal \
     scale regime, where the byte is a plain multiple of 2^-9 and the encoder takes a completely \
     different branch. Two earlier nvfp4 fixtures in this tree drew scale bytes that never got \
     there, so a correct oracle proved nothing about half the encoder. The same screen requires a \
     normal exponent, a saturated 0x7e, and a scale that underflows to the zero byte, because \
     each is a branch of q_encode_scale and each leads to a different divisor for the elements.";

const SLACK_WORDS: usize = 16;

const POISON: u32 = 0xdead_beef;

const DOUBLE_ROUNDING_DOC: &str = "The reference encodes the EXACT product of the global scale and \
     the block's local scale; the shipped encoder carries that product through one f32 rounding \
     of the two significands first. The two can only part company when the exact product and its \
     f32 rounding land on opposite sides of a UE4M3 boundary, so each block is screened for \
     exactly that and a corpus that put one there is rejected. It is a screen on the FIXTURE, not \
     a tolerance on the result -- the comparison itself is bit exact, and an exact tie is not a \
     disagreement because both round it up.";

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_g4w_quant_row_oracle needs a real wgpu adapter; a skipped numeric gate reads as a \
         passed one, so this panics rather than returning early",
    )
}

fn source() -> String {
    let all = nv_models::gemma4_wgpu::nozi_audit_sources();
    let hit = all.into_iter().find(|(t, _)| *t == TAG).unwrap_or_else(|| {
        panic!(
            "gemma4_wgpu::nozi_audit_sources() no longer exposes {TAG}; this gate compiles the \
             SHIPPED text and cannot fall back to a copy"
        )
    });
    let src = hit.1;
    for f in [ENTRY, BLOCK_FN] {
        assert!(
            src.contains(&format!("fn {f}(")),
            "{TAG} no longer declares {f}; the entry moved and this gate is now testing nothing"
        );
    }
    src
}

fn ue4m3_encode(p: f64) -> u8 {
    if !(p > 0.0) || !p.is_finite() {
        return 0;
    }
    let mut e = 0i32;
    let mut sig = p;
    while sig >= 2.0 {
        sig /= 2.0;
        e += 1;
    }
    while sig < 1.0 {
        sig *= 2.0;
        e -= 1;
    }
    if e < -6 {
        let n = (p * 512.0 + 0.5).floor() as i64;
        if n <= 0 {
            return 0;
        }
        if n <= 7 {
            return n as u8;
        }
        return 0x08;
    }
    if e > 8 || (e == 8 && sig > 1.75) {
        return 0x7e;
    }
    let mut mant = ((sig - 1.0) * 8.0 + 0.5).floor() as i32;
    let mut eo = e;
    if mant > 7 {
        mant = 0;
        eo += 1;
    }
    let biased = (eo + 7).clamp(0, 15) as u32;
    let enc = ((biased << 3) | mant as u32) as u8;
    if enc == 0x7f {
        0x7e
    } else {
        enc
    }
}

fn ue4m3_decode(byte: u8) -> f64 {
    let e = ((byte >> 3) & 15) as i32;
    let m = (byte & 7) as f64;
    if e == 0 {
        m * 0.001953125
    } else {
        (2f64).powi(e - 7) * (1.0 + m / 8.0)
    }
}

fn e2m1_encode(x: f32) -> u8 {
    let sign = if x.to_bits() >> 31 == 1 { 8u8 } else { 0 };
    let a = x.abs() as f64;
    let mag = if a <= 0.25 {
        0
    } else if a <= 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a <= 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a <= 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    sign | mag
}

fn bf16_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuantParams {
    global_scale: f32,
    k_blocks: u32,
    pad0: u32,
    pad1: u32,
}

struct Case {
    label: &'static str,
    global: f32,
    x: Vec<u16>,
}

struct Expect {
    packed: Vec<u32>,
    scale_words: Vec<u32>,
    scale_bytes: Vec<u8>,
    nibbles: Vec<u8>,
}

impl Case {
    fn build(label: &'static str, blocks: usize, global: f32, mag: f32, seed: u64) -> Self {
        let mut rng = Lcg::new(seed);
        let x: Vec<u16> = (0..blocks * 16)
            .map(|_| half::bf16::from_f32(rng.next_f32() * mag).to_bits())
            .collect();
        Self { label, global, x }
    }

    fn with(mut self, edits: &[(usize, u16)]) -> Self {
        for (i, bits) in edits {
            self.x[*i] = *bits;
        }
        self
    }

    fn blocks(&self) -> usize {
        self.x.len() / 16
    }

    fn stored(&self) -> f32 {
        let mag = self.global.to_bits() & 0x7fff_ffff;
        if mag == 0 || mag >= 0x7f80_0000 {
            1.0
        } else {
            self.global
        }
    }

    fn expect(&self) -> Expect {
        let stored = self.stored();
        let n = self.blocks();
        let mut packed = vec![0u32; n * 2];
        let mut scale_bytes = vec![0u8; n];
        let mut nibbles = vec![0u8; n * 16];
        for kb in 0..n {
            let vals = &self.x[kb * 16..kb * 16 + 16];
            let mut amax_bits = 0u16;
            for v in vals {
                let mag = v & 0x7fff;
                if mag <= 0x7f80 && mag > amax_bits {
                    amax_bits = mag;
                }
            }
            let local_scale: f32 = if amax_bits != 0 {
                (bf16_f32(amax_bits) as f64 / 6.0) as f32
            } else {
                1.0
            };
            let product = stored as f64 * local_scale as f64;
            let byte = ue4m3_encode(product);
            self.screen_scale(kb, product, byte);
            scale_bytes[kb] = byte;
            let inv: f32 = if byte == 0 {
                stored
            } else {
                (stored as f64 / ue4m3_decode(byte)) as f32
            };
            let mut w = [0u32; 2];
            for i in 0..8 {
                let code = |j: usize| -> u8 {
                    let v = bf16_f32(vals[j]);
                    let q = (v as f64 * inv as f64) as f32;
                    let q = if q.is_nan() { -6.0 } else { q.clamp(-6.0, 6.0) };
                    e2m1_encode(q)
                };
                let lo = code(2 * i);
                let hi = code(2 * i + 1);
                nibbles[kb * 16 + 2 * i] = lo;
                nibbles[kb * 16 + 2 * i + 1] = hi;
                let b = (((hi & 15) << 4) | (lo & 15)) as u32;
                w[i / 4] |= b << (8 * (i % 4));
            }
            packed[kb * 2] = w[0];
            packed[kb * 2 + 1] = w[1];
        }
        let mut scale_words = vec![0u32; n.div_ceil(4)];
        for (g, word) in scale_words.iter_mut().enumerate() {
            for j in 0..4 {
                let kb = 4 * g + j;
                if kb < n {
                    *word |= (scale_bytes[kb] as u32) << (8 * j);
                }
            }
        }
        Expect {
            packed,
            scale_words,
            scale_bytes,
            nibbles,
        }
    }

    fn screen_scale(&self, kb: usize, product: f64, byte: u8) {
        assert_eq!(
            ue4m3_encode((product as f32) as f64),
            byte,
            "{}: block {kb} has a scale product {product:e} whose f32 rounding crosses a UE4M3 \
             boundary, where the shipped encoder and this reference may legitimately disagree. \
             {DOUBLE_ROUNDING_DOC}",
            self.label
        );
    }

    fn params(&self) -> QuantParams {
        QuantParams {
            global_scale: self.global,
            k_blocks: self.blocks() as u32,
            pad0: 0,
            pad1: 0,
        }
    }
}

struct Got {
    packed: Vec<u32>,
    scale_words: Vec<u32>,
    packed_slack: Vec<u32>,
    scale_slack: Vec<u32>,
}

const PIPELINE_IS_BUILT_ONCE_PER_SOURCE: &str = "The probe pipeline is compiled once per shader \
     text and reused across the corpus. Compiling it per case would rebuild the same MSL for \
     every one of them, which is the difference between this suite taking seconds and taking \
     minutes on a box that is also serving.";

fn pipeline(ctx: &WgpuContext, src: &str) -> anyhow::Result<wgpu::ComputePipeline> {
    dispatch::compute_pipeline(ctx, "g4w-quant-row-pk-probe", src, ENTRY)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

fn run(ctx: &WgpuContext, pipe: &wgpu::ComputePipeline, c: &Case) -> anyhow::Result<Got> {
    let n = c.blocks();
    let pw = n * 2;
    let sw = n.div_ceil(4);
    let x = dispatch::storage_from_slice(ctx, "qz-x", &pack(&c.x));
    let p = dispatch::uniform_from(ctx, "qz-p", &c.params());
    let pk = dispatch::storage_from_slice(ctx, "qz-packed", &vec![POISON; pw + SLACK_WORDS]);
    let sc = dispatch::storage_from_slice(ctx, "qz-scales", &vec![POISON; sw + SLACK_WORDS]);
    let grid = dispatch::workgroup_count_1d(ctx, n as u64, 256);
    dispatch::dispatch(ctx, pipe, &[(0, &x), (1, &p), (2, &pk), (3, &sc)], grid)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let pv: Vec<u32> =
        dispatch::read_back(ctx, &pk, pw + SLACK_WORDS).map_err(|e| anyhow::anyhow!("{e}"))?;
    let sv: Vec<u32> =
        dispatch::read_back(ctx, &sc, sw + SLACK_WORDS).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(Got {
        packed: pv[..pw].to_vec(),
        scale_words: sv[..sw].to_vec(),
        packed_slack: pv[pw..].to_vec(),
        scale_slack: sv[sw..].to_vec(),
    })
}

const NAN_BITS: u16 = 0x7fc0;

fn corpus() -> Vec<Case> {
    vec![
        Case::build("normal scale exponent", 64, 1.0, 1.0, 0x0000_4a01),
        Case::build("subnormal scale exponent", 48, 0.01, 1.0, 0x0000_5b02),
        Case::build("saturating scale, every element clamps", 32, 4000.0, 1.0, 0x0000_6c03),
        Case::build("scale underflows to the zero byte", 32, 1e-5, 1.0, 0x0000_7d04),
        Case::build("global scale of zero is replaced by one", 32, 0.0, 1.0, 0x0000_8e05),
        Case::build("block tail past the last full scale word", 302, 1.0, 1.0, 0x0000_9f06)
            .with(&[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
                    (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0),
                    (32, NAN_BITS)]),
    ]
}

struct Regimes {
    subnormal_exp: usize,
    normal_exp: usize,
    saturated: usize,
    zero_byte: usize,
    zero_nibble: usize,
    max_nibble: usize,
    negative_nibble: usize,
    zero_block: usize,
}

fn regimes(cases: &[Case]) -> Regimes {
    let mut r = Regimes {
        subnormal_exp: 0,
        normal_exp: 0,
        saturated: 0,
        zero_byte: 0,
        zero_nibble: 0,
        max_nibble: 0,
        negative_nibble: 0,
        zero_block: 0,
    };
    for c in cases {
        let e = c.expect();
        for b in &e.scale_bytes {
            match *b {
                0 => r.zero_byte += 1,
                0x7e => r.saturated += 1,
                b if (b >> 3) & 15 == 0 => r.subnormal_exp += 1,
                _ => r.normal_exp += 1,
            }
        }
        for nb in &e.nibbles {
            if nb & 7 == 0 {
                r.zero_nibble += 1;
            }
            if nb & 7 == 7 {
                r.max_nibble += 1;
            }
            if nb & 8 != 0 {
                r.negative_nibble += 1;
            }
        }
        for kb in 0..c.blocks() {
            if c.x[kb * 16..kb * 16 + 16].iter().all(|v| *v == 0) {
                r.zero_block += 1;
            }
        }
    }
    r
}

#[test]
fn the_corpus_reaches_every_branch_of_the_scale_encoder() {
    let cases = corpus();
    let r = regimes(&cases);
    for (what, n) in [
        ("a UE4M3 scale byte with a biased exponent of zero", r.subnormal_exp),
        ("a UE4M3 scale byte with a normal exponent", r.normal_exp),
        ("a scale saturated at 0x7e", r.saturated),
        ("a scale that underflowed to the zero byte", r.zero_byte),
        ("an all-zero block, where the local scale falls back to one", r.zero_block),
        ("an element that encodes to the zero code", r.zero_nibble),
        ("an element that reaches the E2M1 maximum", r.max_nibble),
        ("a negative element", r.negative_nibble),
    ] {
        assert!(
            n > 0,
            "the corpus never produces {what}, so that branch is untested while every case \
             passes. {THE_SCALE_EXPONENT_MUST_REACH_ZERO}"
        );
    }
    eprintln!(
        "[quant-row-oracle] scale bytes: {} subnormal-exponent, {} normal, {} saturated, {} zero; \
         nibbles: {} zero, {} at the E2M1 maximum, {} negative; {} all-zero blocks",
        r.subnormal_exp,
        r.normal_exp,
        r.saturated,
        r.zero_byte,
        r.zero_nibble,
        r.max_nibble,
        r.negative_nibble,
        r.zero_block
    );
    eprintln!("{WHY}\n{ORACLE_IS_NOT_THE_KERNEL}\n{THE_SCALE_EXPONENT_MUST_REACH_ZERO}");
}

fn compare(c: &Case, got: &Got, e: &Expect) -> Option<String> {
    for (kb, w) in got.scale_words.iter().enumerate() {
        if *w != e.scale_words[kb] {
            return Some(format!(
                "scale word {kb} is {:#010x}, the f64 reference says {:#010x}",
                w, e.scale_words[kb]
            ));
        }
    }
    for (i, w) in got.packed.iter().enumerate() {
        if *w != e.packed[i] {
            return Some(format!(
                "packed word {i} (block {}) is {:#010x}, the f64 reference says {:#010x}",
                i / 2,
                w,
                e.packed[i]
            ));
        }
    }
    if got.packed_slack.iter().any(|w| *w != POISON)
        || got.scale_slack.iter().any(|w| *w != POISON)
    {
        return Some(format!(
            "{} wrote past the {} blocks it was given",
            ENTRY,
            c.blocks()
        ));
    }
    None
}

#[test]
fn quant_row_pk_matches_an_f64_nvfp4_encoder() {
    let ctx = ctx();
    eprintln!("[quant-row-oracle] adapter: {}\n{WHY}", ctx.info.name);
    let src = source();
    eprintln!("{PIPELINE_IS_BUILT_ONCE_PER_SOURCE}");
    let pipe = pipeline(ctx, &src).expect("clean pipeline");
    for c in corpus() {
        let got = run(ctx, &pipe, &c).unwrap_or_else(|e| panic!("{}: {e}", c.label));
        let e = c.expect();
        if let Some(d) = compare(&c, &got, &e) {
            panic!(
                "[{}] {ENTRY} disagrees with the f64 nvfp4 encoder at global_scale={} over {} \
                 blocks: {d}. {ORACLE_IS_NOT_THE_KERNEL}",
                c.label,
                c.global,
                c.blocks()
            );
        }
        eprintln!(
            "[{}] {ENTRY}: {} blocks encode bit-exactly at global_scale={}",
            c.label,
            c.blocks(),
            c.global
        );
    }
}

const MUTANTS: [(&str, &str, &str); 6] = [
    (
        "block-base-strides-by-the-word-count-instead-of-the-element-count",
        "    let base = kb * NVFP4_BLOCK_SIZE;",
        "    let base = kb * 8u;",
    ),
    (
        "local-scale-divides-the-block-maximum-by-three-instead-of-six",
        "        local_scale = q_div_small(bf16_decode(amax_bits), 3u, 1);",
        "        local_scale = q_div_small(bf16_decode(amax_bits), 3u, 0);",
    ),
    (
        "packed-pair-halves-swapped",
        "        let packed = ((hi & 15u) << 4u) | (lo & 15u);",
        "        let packed = ((lo & 15u) << 4u) | (hi & 15u);",
    ),
    (
        "scale-word-packs-the-third-lane-over-the-second",
        "            | (g4w_qz_sbytes[lid.x + 2u] << 16u)",
        "            | (g4w_qz_sbytes[lid.x + 2u] << 8u)",
    ),
    (
        "block-maximum-admits-nan-magnitudes",
        "        if (mag <= 0x7f80u && mag > amax_bits) {",
        "        if (mag > amax_bits) {",
    ),
    (
        "degenerate-global-scale-falls-back-to-two-instead-of-one",
        "    let stored = select(g, 1.0, bad);",
        "    let stored = select(g, 2.0, bad);",
    ),
];

const WHY_THE_GUARD_MUTANT_CHANGES_THE_FALLBACK_RATHER_THAN_REMOVING_IT: &str = "The fallback \
     mutant replaces 1.0 with 2.0 instead of deleting the select. Deleting it lets a global scale \
     of zero reach q_norm_parts, whose normalising loop shifts a zero mantissa left forever: the \
     dispatch never returns, the device wedges and the suite hangs rather than fails. That is a \
     load-bearing property of the guard and the reason it is not merely a nicety, but a mutant \
     that hangs is not a mutant -- it proves nothing a timeout could be read as. Changing the \
     fallback VALUE keeps the branch observable and the dispatch finite.";

const NAN_MUTANT: usize = 4;
const DEGENERATE_GLOBAL_MUTANT: usize = 5;

fn mutate(src: &str, from: &str, to: &str) -> String {
    assert!(
        src.contains(from),
        "mutation anchor not present in the shipped quantizer: {from:?}. This gate is worthless \
         if its mutants no longer apply -- fix the anchors."
    );
    src.replace(from, to)
}

#[test]
fn every_mutation_anchor_still_applies_to_the_shipped_quantizer() {
    let src = source();
    eprintln!("{WHY_THE_GUARD_MUTANT_CHANGES_THE_FALLBACK_RATHER_THAN_REMOVING_IT}");
    for (name, from, to) in MUTANTS {
        assert!(
            src.contains(from),
            "anchor for mutant {name} is gone from {TAG}: {from:?}. A mutant whose anchor rotted \
             is silently inert, and the GPU tests that would have caught that do not run on a box \
             with no adapter -- which is why this check is CPU-only and unconditional."
        );
        assert_ne!(
            from, to,
            "mutant {name} replaces text with itself and can never turn anything red"
        );
    }
}

#[test]
fn every_quantizer_mutant_is_caught_by_this_corpus() {
    let ctx = ctx();
    let src = source();
    let cases = corpus();
    let expects: Vec<Expect> = cases.iter().map(|c| c.expect()).collect();
    for (name, from, to) in MUTANTS {
        let bad = mutate(&src, from, to);
        let pipe = pipeline(ctx, &bad).expect("mutant pipeline");
        let mut caught_by: Vec<&str> = Vec::new();
        for (c, e) in cases.iter().zip(expects.iter()) {
            let hit = match run(ctx, &pipe, c) {
                Ok(got) => compare(c, &got, e).is_some(),
                Err(_) => true,
            };
            if hit {
                caught_by.push(c.label);
            }
        }
        assert!(
            !caught_by.is_empty(),
            "mutant {name} was NOT caught by any case in the corpus. A quantizer whose encoding a \
             broken branch passes is not a gate. {WHY} {THE_SCALE_EXPONENT_MUST_REACH_ZERO}"
        );
        eprintln!("MUTANT {name}: caught by {caught_by:?}");
    }
}

#[test]
fn only_the_cases_that_reach_a_branch_can_see_its_mutant() {
    let ctx = ctx();
    let src = source();
    let cases = corpus();
    for (which, reached_by) in [
        (NAN_MUTANT, "block tail past the last full scale word"),
        (DEGENERATE_GLOBAL_MUTANT, "global scale of zero is replaced by one"),
    ] {
        let (name, from, to) = MUTANTS[which];
        let bad = mutate(&src, from, to);
        let pipe = pipeline(ctx, &bad).expect("mutant pipeline");
        let mut caught: Vec<&str> = Vec::new();
        for c in &cases {
            let hit = match run(ctx, &pipe, c) {
                Ok(got) => compare(c, &got, &c.expect()).is_some(),
                Err(_) => true,
            };
            if hit {
                caught.push(c.label);
            }
        }
        assert_eq!(
            caught,
            vec![reached_by],
            "mutant {name} was caught by {caught:?}; it can only be reached by the case that \
             drives its branch, so a wider list means some other case now carries that input by \
             accident and a narrower one means the branch is no longer reached at all. \
             {THE_SCALE_EXPONENT_MUST_REACH_ZERO}"
        );
        eprintln!("MUTANT {name}: caught only by {caught:?}, as its branch requires");
    }
}
