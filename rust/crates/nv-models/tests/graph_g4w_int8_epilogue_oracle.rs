#![cfg(feature = "wgpu")]

mod common;
use common::ord;
use common::Split;
use common::unpack;
use nv_kernels::wgpu_backend::kernels::gemv_nvfp4;
use nv_kernels::wgpu_backend::kernels::quant_gemv;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use nv_models::gemma4_wgpu as g4w;
use common::LcgOddSeedShift32GaussUnit as Lcg;

const PK3_ENTRY: &str = "g4w_gemv_int8_pk3";

const WHY: &str = "g4w_gemv_int8_pk and g4w_gemv_int8_pk3 are the int8 decode epilogues of the \
     Gemma-4 dense graph -- the packed and the fused q/k/v split-scatter arms of the W8 FFN and \
     the int8 lm-head. The entry census records both as having no gate of their own: the only \
     suite that names them is mk_tree_twin_is_bitwise_identical_to_both_decode_epilogues, which \
     uses g4w_gemv_int8_pk as the ORACLE for the int8 M-row twin, and \
     mk_pk3_split_scatter_is_bitwise_identical_to_the_decode_pk3_epilogue, which does the same \
     with the pk3 arm. A defect in one arm shows up there as disagreement; a defect shared by \
     every arm does not, and the two int8 arms share their whole accumulator. Only the fp8 \
     sibling carries an f64 chain, and the int8 accumulator is DIFFERENT TEXT: qg_group_acc_i8 \
     over qg_dot16_i8 over int8_decode, none of which any fp8 case reaches.";

const ORACLE_IS_NOT_THE_KERNEL: &str = "The reference is an f64 host computation written from the \
     definition of a group-scaled int8 GEMV: for each 16-element group, the sum of the SIGNED \
     byte times the bf16 activation, scaled by that group's f32 scale, accumulated over the row. \
     It is not a call into a shipped kernel and it is not the fp8 twin -- the fp8 twin decodes \
     e4m3, which is exactly the difference this suite exists to cover.";

const NOISE_C: f64 = 16.0;

const BOUNDS_DOC: &str = "ONE bf16 ulp per element against the f64 reference, not an aggregate \
     norm: an aggregate norm over a bf16 output saturates at half a bf16 ulp and cannot see an \
     epilogue defect at all. NOISE_C=16 is the f32 accumulation floor -- a row whose true dot \
     product sits below 16*f32::EPSILON times the sum of the magnitudes of its own terms is not \
     resolvable to one bf16 ulp by ANY f32 kernel, and those rows are gated on absolute error \
     against that floor instead.";

const THE_CORPUS_REACHES_BOTH_SCALE_REGIMES: &str = "Both group regimes are in the corpus and \
     neither is optional. At group == 0 the shipped group_shift is 31, every `v >> sh` is 0 and \
     the whole row shares one scale, so a mutant that collapses the scale INDEX is invisible; \
     only the grouped cases can see it. The grouped cases in turn cannot see a defect in the \
     row-scale degenerate path. A corpus with one of the two would report a gate it does not \
     have. The weight bytes are drawn across the full signed range including the extremes, so \
     the sign of the decode is observable rather than a property of the draw.";

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_g4w_int8_epilogue_oracle needs a real wgpu adapter; a skipped numeric gate reads \
         as a passed one, so this panics rather than returning early",
    )
}

fn source(sg: bool) -> String {
    let src = g4w::fp8_pk_shader_source(sg);
    for e in [g4w::INT8_PK_ENTRY, PK3_ENTRY] {
        assert!(
            src.contains(&format!("fn {e}(")),
            "fp8_pk_shader_source(sg={sg}) no longer declares {e}; the entry moved and this gate \
             is now testing nothing. {WHY}"
        );
    }
    src
}

fn paths(ctx: &WgpuContext) -> Vec<(&'static str, bool)> {
    let mut v = vec![("tree", false)];
    if gemv_nvfp4::sg32_ok(ctx) {
        v.push(("sg", true));
    } else {
        eprintln!("NOTE: no 32-wide subgroups here, only the tree epilogue is covered");
    }
    v
}

struct Case {
    label: &'static str,
    n: usize,
    k: usize,
    group: usize,
    wq: Vec<u32>,
    scales: Vec<f32>,
    x_packed: Vec<u32>,
    reference: Vec<f64>,
    magnitude: Vec<f64>,
}

impl Case {
    fn build(label: &'static str, n: usize, k: usize, group: usize, seed: u64) -> Self {
        assert!(
            n.is_multiple_of(8),
            "{label}: the pk epilogues pair rows on an 8-row grid"
        );
        quant_gemv::group_rule(k, group).unwrap_or_else(|e| panic!("{label}: {e}"));
        let per_row = quant_gemv::scales_per_row(k, group);
        let g = if group == 0 { k } else { group };

        let mut rng = Lcg::new(seed);
        let mut bytes = vec![0i8; n * k];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = match i % 64 {
                0 => -128,
                1 => 127,
                2 => 0,
                3 => -1,
                _ => (rng.next_u32() & 0xff) as u8 as i8,
            };
        }
        let mut wq = vec![0u32; n * k / 4];
        for (i, b) in bytes.iter().enumerate() {
            wq[i / 4] |= ((*b as u8) as u32) << (8 * (i % 4));
        }
        let scales: Vec<f32> = (0..n * per_row)
            .map(|_| (0.5 + rng.unit()) * (0.03 / 127.0))
            .collect();
        let x: Vec<u16> = (0..k)
            .map(|_| half::bf16::from_f32(rng.gauss() * 0.4).to_bits())
            .collect();
        let x_packed = quant_gemv::pack_x_bf16(&x);
        let xf: Vec<f64> = x
            .iter()
            .map(|b| f32::from_bits((*b as u32) << 16) as f64)
            .collect();

        let mut reference = vec![0f64; n];
        let mut magnitude = vec![0f64; n];
        for r in 0..n {
            let mut acc = 0f64;
            let mut mag = 0f64;
            for v in 0..k / 16 {
                let s = scales[r * per_row + (v * 16) / g] as f64;
                let mut d = 0f64;
                let mut dm = 0f64;
                for i in 0..16 {
                    let p = bytes[r * k + v * 16 + i] as f64 * xf[v * 16 + i];
                    d += p;
                    dm += p.abs();
                }
                acc += s * d;
                mag += s.abs() * dm;
            }
            reference[r] = acc;
            magnitude[r] = mag;
        }
        let c = Self {
            label,
            n,
            k,
            group,
            wq,
            scales,
            x_packed,
            reference,
            magnitude,
        };
        c.screen(&bytes);
        c
    }

    fn screen(&self, bytes: &[i8]) {
        for want in [-128i8, 127, 0, -1] {
            assert!(
                bytes.contains(&want),
                "{}: no weight byte equals {want}, so this fixture never reaches the edge of the \
                 signed range and a decode that dropped the sign could pass. \
                 {THE_CORPUS_REACHES_BOTH_SCALE_REGIMES}",
                self.label
            );
        }
        let resolvable = self
            .reference
            .iter()
            .zip(self.magnitude.iter())
            .filter(|(r, m)| r.abs() > NOISE_C * (f32::EPSILON as f64) * **m)
            .count();
        assert!(
            resolvable * 4 >= self.n * 3,
            "{}: only {resolvable} of {} rows are above the f32 accumulation floor, so most of \
             this fixture is gated on the floor rather than on one bf16 ulp. {BOUNDS_DOC}",
            self.label,
            self.n
        );
    }

    fn params(&self, groups_x: u32) -> quant_gemv::QuantGemvParams {
        quant_gemv::params_for(self.n, self.k, self.group, groups_x)
    }
}

struct Report {
    max_ulp: i64,
    worst_row: usize,
    got: f32,
    want: f64,
    nonfinite: usize,
    unresolvable: usize,
    worst_floor_ratio: f64,
}

fn report(got: &[u16], want: &[f64], mag: &[f64]) -> Report {
    let mut r = Report {
        max_ulp: 0,
        worst_row: 0,
        got: 0.0,
        want: 0.0,
        nonfinite: 0,
        unresolvable: 0,
        worst_floor_ratio: 0.0,
    };
    for (i, g) in got.iter().enumerate() {
        let gf = f32::from_bits((*g as u32) << 16);
        if !gf.is_finite() {
            r.nonfinite += 1;
        }
        let floor = NOISE_C * (f32::EPSILON as f64) * mag[i];
        if want[i].abs() <= floor {
            r.unresolvable += 1;
            if floor > 0.0 {
                r.worst_floor_ratio = r
                    .worst_floor_ratio
                    .max(((gf as f64) - want[i]).abs() / floor);
            }
            continue;
        }
        let d = (ord(*g) - ord(half::bf16::from_f64(want[i]).to_bits())).abs();
        if d > r.max_ulp {
            r.max_ulp = d;
            r.worst_row = i;
            r.got = gf;
            r.want = want[i];
        }
    }
    r
}

impl Report {
    fn caught(&self) -> bool {
        self.max_ulp > 1 || self.worst_floor_ratio > 1.0 || self.nonfinite > 0
    }
    fn check(&self, who: &str) {
        assert_eq!(self.nonfinite, 0, "{who} produced non-finite outputs: {self}");
        assert!(
            self.max_ulp <= 1,
            "{who} exceeds one bf16 ulp against the f64 host reference: {self}. \
             {ORACLE_IS_NOT_THE_KERNEL} {BOUNDS_DOC}"
        );
        assert!(
            self.worst_floor_ratio <= 1.0,
            "{who} exceeds the f32 accumulation noise floor on a cancellation-limited row: \
             {self}. {BOUNDS_DOC}"
        );
    }
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max {} bf16 ulp at row {} (got {:e} want {:e}), {} non-finite, {} unresolvable \
             (worst {:.3}x noise floor)",
            self.max_ulp,
            self.worst_row,
            self.got,
            self.want,
            self.nonfinite,
            self.unresolvable,
            self.worst_floor_ratio
        )
    }
}

fn run_pk(
    ctx: &WgpuContext,
    src: &str,
    entry: &str,
    c: &Case,
    sg: bool,
) -> anyhow::Result<Vec<u16>> {
    let grid = dispatch::workgroup_count_1d(ctx, c.n as u64, g4w::fp8_pk_rows_per_group(sg));
    let w = dispatch::storage_from_slice(ctx, "i8-w", &c.wq);
    let s = dispatch::storage_from_slice(ctx, "i8-s", &c.scales);
    let x = dispatch::storage_from_slice(ctx, "i8-x", &c.x_packed);
    let y = dispatch::storage_from_slice(ctx, "i8-y", &vec![0xdead_beefu32; c.n / 2]);
    let p = dispatch::uniform_from(ctx, "i8-p", &c.params(grid.0));
    let off = dispatch::uniform_from(ctx, "i8-off", &[0u32, 0, 0, 0]);
    let pipe = dispatch::compute_pipeline(ctx, "g4w-int8-pk-probe", src, entry)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    dispatch::dispatch(
        ctx,
        &pipe,
        &[(0, &w), (1, &s), (2, &x), (3, &y), (4, &p), (30, &off)],
        grid,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let words: Vec<u32> =
        dispatch::read_back(ctx, &y, c.n / 2).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(unpack(&words, c.n))
}

fn run_pk3(
    ctx: &WgpuContext,
    src: &str,
    c: &Case,
    sg: bool,
    sp: &Split,
) -> anyhow::Result<(Vec<u16>, Vec<u16>, Vec<u16>)> {
    assert_eq!(sp.v_off, sp.q_rows + sp.kv_rows);
    assert_eq!(c.n, sp.q_rows + 2 * sp.kv_rows);
    let grid = dispatch::workgroup_count_1d(ctx, c.n as u64, g4w::fp8_pk_rows_per_group(sg));
    let w = dispatch::storage_from_slice(ctx, "i8-w", &c.wq);
    let s = dispatch::storage_from_slice(ctx, "i8-s", &c.scales);
    let x = dispatch::storage_from_slice(ctx, "i8-x", &c.x_packed);
    let qb = dispatch::storage_from_slice(ctx, "i8-q", &vec![0xdead_beefu32; sp.q_rows / 2]);
    let kb = dispatch::storage_from_slice(ctx, "i8-k", &vec![0xdead_beefu32; sp.kv_rows / 2]);
    let vb = dispatch::storage_from_slice(ctx, "i8-v", &vec![0xdead_beefu32; sp.kv_rows / 2]);
    let p = dispatch::uniform_from(ctx, "i8-p", &c.params(grid.0));
    let spb = dispatch::uniform_from(
        ctx,
        "i8-sp",
        &[sp.q_rows as u32, sp.kv_rows as u32, sp.v_off as u32, 0u32],
    );
    let pipe = dispatch::compute_pipeline(ctx, "g4w-int8-pk3-probe", src, PK3_ENTRY)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    dispatch::dispatch(
        ctx,
        &pipe,
        &[
            (0, &w),
            (1, &s),
            (2, &x),
            (4, &p),
            (31, &qb),
            (32, &kb),
            (33, &vb),
            (34, &spb),
        ],
        grid,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let rb = |b: &wgpu::Buffer, n: usize| -> anyhow::Result<Vec<u32>> {
        dispatch::read_back(ctx, b, n).map_err(|e| anyhow::anyhow!("{e}"))
    };
    Ok((
        unpack(&rb(&qb, sp.q_rows / 2)?, sp.q_rows),
        unpack(&rb(&kb, sp.kv_rows / 2)?, sp.kv_rows),
        unpack(&rb(&vb, sp.kv_rows / 2)?, sp.kv_rows),
    ))
}

const MUTANTS: [(&str, &str, &str); 4] = [
    (
        "int8-decode-drops-the-sign",
        "    return f32(extractBits(bitcast<i32>(word), 8u * (elem & 3u), 8u));",
        "    return f32(extractBits(word, 8u * (elem & 3u), 8u));",
    ),
    (
        "i8-dot-second-lane-reads-the-low-half-twice",
        "    acc = fma(int8_decode(word, 1u), bf16_hi(xw0), acc);",
        "    acc = fma(int8_decode(word, 1u), bf16_lo(xw0), acc);",
    ),
    (
        "i8-group-scale-index-collapsed-to-the-row-base",
        "acc = fma(qg_row_scale[sbase + (v >> sh)], d, acc);\n    }\n    return acc;\n}\n\nfn qg_row_acc_mx",
        "acc = fma(qg_row_scale[sbase], d, acc);\n    }\n    return acc;\n}\n\nfn qg_row_acc_mx",
    ),
    (
        "i8-accumulator-reads-the-low-activation-vector-twice",
        "        let d = qg_dot16_i8(qg_w4[wbase + v], qg_x4[2u * v], qg_x4[2u * v + 1u]);",
        "        let d = qg_dot16_i8(qg_w4[wbase + v], qg_x4[2u * v], qg_x4[2u * v]);",
    ),
];

const SCALE_INDEX_MUTANT: usize = 2;

const SIGN_MUTANT: usize = 0;

fn mutate(src: &str, from: &str, to: &str) -> String {
    assert!(
        src.contains(from),
        "mutation anchor not present in the shipped q8 epilogue source: {from:?}. This gate is \
         worthless if its mutants no longer apply -- fix the anchors."
    );
    src.replace(from, to)
}

fn corpus() -> Vec<Case> {
    vec![
        Case::build("row scale, whole row shares one", 256, 1024, 0, 0x0000_51d1),
        Case::build("grouped-128", 256, 1024, 128, 0x0000_91a5),
        Case::build("odd row block, grouped-128", 40, 512, 128, 0x0000_0dd0),
        Case::build("minimal", 8, 32, 0, 0x0000_1c1a),
    ]
}

fn pk3_cases() -> Vec<(Case, Split)> {
    vec![
        (
            Case::build("fused qkv, row scale", 64, 256, 0, 0x0000_9a71),
            Split {
                q_rows: 32,
                kv_rows: 16,
                v_off: 48,
            },
        ),
        (
            Case::build("fused qkv, grouped-128", 64, 256, 128, 0x0000_beef),
            Split {
                q_rows: 32,
                kv_rows: 16,
                v_off: 48,
            },
        ),
    ]
}

#[test]
fn every_mutation_anchor_still_applies_to_the_shipped_int8_text() {
    for sg in [false, true] {
        let src = source(sg);
        for (name, from, to) in MUTANTS {
            assert!(
                src.contains(from),
                "anchor for mutant {name} is gone from the sg={sg} shipped epilogue source: \
                 {from:?}. A mutant whose anchor rotted is silently inert, and the GPU tests that \
                 would have caught that do not run on a box with no adapter -- which is why this \
                 check is CPU-only and unconditional."
            );
            assert_ne!(
                from, to,
                "mutant {name} replaces text with itself and can never turn anything red"
            );
        }
        assert!(
            src.contains("fn qg_group_acc_i8(") && src.contains("fn qg_dot16_i8("),
            "the int8 accumulator this suite gates is gone from the sg={sg} source; the entries \
             would then be sharing the fp8 arm's text and this suite would be a second fp8 gate"
        );
    }
    eprintln!("{WHY}\n{ORACLE_IS_NOT_THE_KERNEL}\n{THE_CORPUS_REACHES_BOTH_SCALE_REGIMES}");
}

#[test]
fn int8_pk_matches_an_f64_host_reference() {
    let ctx = ctx();
    eprintln!("[int8-epilogue-oracle] adapter: {}\n{WHY}", ctx.info.name);
    for (name, sg) in paths(ctx) {
        let src = source(sg);
        for c in corpus() {
            let got = run_pk(ctx, &src, g4w::INT8_PK_ENTRY, &c, sg)
                .unwrap_or_else(|e| panic!("{}/{name}: pk dispatch: {e}", c.label));
            let r = report(&got, &c.reference, &c.magnitude);
            eprintln!(
                "[{}][{name}] {} n={} k={} group={}: {r}",
                c.label,
                g4w::INT8_PK_ENTRY,
                c.n,
                c.k,
                c.group
            );
            r.check(&format!("[{}][{name}] {}", c.label, g4w::INT8_PK_ENTRY));
        }
    }
}

#[test]
fn int8_pk3_split_scatter_matches_an_f64_host_reference() {
    let ctx = ctx();
    for (name, sg) in paths(ctx) {
        let src = source(sg);
        for (c, sp) in pk3_cases() {
            let (q, k, v) = run_pk3(ctx, &src, &c, sg, &sp)
                .unwrap_or_else(|e| panic!("{}/{name}: pk3 dispatch: {e}", c.label));
            for (seg, got, from, to) in [
                ("Q", &q, 0usize, sp.q_rows),
                ("K", &k, sp.q_rows, sp.v_off),
                ("V", &v, sp.v_off, c.n),
            ] {
                let r = report(got, &c.reference[from..to], &c.magnitude[from..to]);
                eprintln!("[{}][{name}] {PK3_ENTRY} segment {seg}: {r}", c.label);
                r.check(&format!(
                    "[{}][{name}] {PK3_ENTRY} split-scatter segment {seg}",
                    c.label
                ));
            }
        }
    }
}

#[test]
fn every_int8_mutant_is_caught_by_this_corpus() {
    let ctx = ctx();
    for (name, sg) in paths(ctx) {
        let src = source(sg);
        for (mn, from, to) in MUTANTS {
            let bad = mutate(&src, from, to);
            let mut caught_by: Vec<&str> = Vec::new();
            for c in corpus() {
                let caught = match run_pk(ctx, &bad, g4w::INT8_PK_ENTRY, &c, sg) {
                    Ok(got) => report(&got, &c.reference, &c.magnitude).caught(),
                    Err(_) => true,
                };
                if caught {
                    caught_by.push(c.label);
                }
            }
            for (c, sp) in pk3_cases() {
                let caught = match run_pk3(ctx, &bad, &c, sg, &sp) {
                    Ok((q, k, v)) => {
                        report(&q, &c.reference[..sp.q_rows], &c.magnitude[..sp.q_rows]).caught()
                            || report(
                                &k,
                                &c.reference[sp.q_rows..sp.v_off],
                                &c.magnitude[sp.q_rows..sp.v_off],
                            )
                            .caught()
                            || report(&v, &c.reference[sp.v_off..], &c.magnitude[sp.v_off..])
                                .caught()
                    }
                    Err(_) => true,
                };
                if caught {
                    caught_by.push(c.label);
                }
            }
            assert!(
                !caught_by.is_empty(),
                "[{name}] mutant {mn} was NOT caught by any case in the corpus. An int8 epilogue a \
                 broken accumulator passes is not a gate. {WHY} \
                 {THE_CORPUS_REACHES_BOTH_SCALE_REGIMES}"
            );
            eprintln!("[{name}] MUTANT {mn}: caught by {caught_by:?}");
        }
    }
}

#[test]
fn only_the_grouped_cases_can_see_the_scale_index() {
    let ctx = ctx();
    let src = source(false);
    let (name, from, to) = MUTANTS[SCALE_INDEX_MUTANT];
    assert_eq!(name, "i8-group-scale-index-collapsed-to-the-row-base");
    let bad = mutate(&src, from, to);
    let mut grouped_caught = 0usize;
    let mut row_scale_survivals = 0usize;
    for c in corpus() {
        let caught = match run_pk(ctx, &bad, g4w::INT8_PK_ENTRY, &c, false) {
            Ok(got) => report(&got, &c.reference, &c.magnitude).caught(),
            Err(_) => true,
        };
        if c.group == 0 {
            if !caught {
                row_scale_survivals += 1;
            }
        } else if caught {
            grouped_caught += 1;
        }
    }
    assert!(
        grouped_caught > 0,
        "no grouped case caught the collapsed scale index, so the corpus has stopped reaching the \
         regime where the per-group scale differs from the row's first scale and the index is \
         untested. {THE_CORPUS_REACHES_BOTH_SCALE_REGIMES}"
    );
    assert!(
        row_scale_survivals > 0,
        "a group == 0 case also caught the collapsed scale index, which contradicts the shipped \
         group_shift of 31: at group == 0 every `v >> sh` is already 0 and the mutation is the \
         identity. This suite is then measuring something other than what it names."
    );
}

#[test]
fn the_int8_decode_this_suite_gates_is_no_part_of_the_fp8_arm() {
    let ctx = ctx();
    let src = source(false);
    let (name, from, to) = MUTANTS[SIGN_MUTANT];
    assert_eq!(name, "int8-decode-drops-the-sign");
    let bad = mutate(&src, from, to);
    let c = Case::build("attribution probe", 64, 256, 0, 0x0000_a771);
    let clean = run_pk(ctx, &src, g4w::FP8_PK_ENTRY, &c, false).expect("fp8 clean");
    let mutated = run_pk(ctx, &bad, g4w::FP8_PK_ENTRY, &c, false).expect("fp8 mutated");
    assert_eq!(
        clean, mutated,
        "dropping the sign from int8_decode moved the fp8 epilogue's output, so the redness this \
         suite reports for {} could belong to the fp8 arm that wgpu_fp8_epilogue.rs already \
         gates, and the int8 entries would still have no gate of their own. {WHY}",
        g4w::INT8_PK_ENTRY
    );
    let i8_clean = run_pk(ctx, &src, g4w::INT8_PK_ENTRY, &c, false).expect("int8 clean");
    let i8_bad = run_pk(ctx, &bad, g4w::INT8_PK_ENTRY, &c, false).expect("int8 mutated");
    assert_ne!(
        i8_clean, i8_bad,
        "dropping the sign from int8_decode did NOT move the int8 epilogue's output, so this \
         attribution proves nothing and the fixture never reached a negative weight byte"
    );
}
