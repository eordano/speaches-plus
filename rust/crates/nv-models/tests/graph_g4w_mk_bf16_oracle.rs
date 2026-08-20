#![cfg(feature = "wgpu")]

mod common;
use common::ord;
use common::Split;
use common::unpack;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use nv_models::gemma4_wgpu as g4w;
use common::LcgOddSeedShift33SignedUnitG4w as Lcg;

const PK_ENTRY: &str = "g4w_gemm_bf16_mk_pk";
const PK3_ENTRY: &str = "g4w_gemm_bf16_mk_pk3";

const WHY: &str = "g4w_gemm_bf16_mk_pk and g4w_gemm_bf16_mk_pk3 are the M-row bf16 projections \
     the Gemma-4 dense graph runs for every prefill chunk: one dispatch per chunk per projection, \
     which is where the whole prompt is spent. They were the two entries the runtime entry census \
     recorded as NOT-REACHED rather than merely ungated -- strictly worse, because no mutation of \
     their text could be turned red by anything in the workspace. Their WGSL is BUILT IN RUST by \
     mk_bf16_source(), which unrolls the token loop, so it is not one of the files the WGSL \
     extraction moved into nv-kernels/wgsl and no test could reach it by reading a .wgsl. The \
     accessor mk_bf16_shader_source is what closes that, and build_mk_pipelines composes the \
     shipped source through the same function, so this suite cannot drift onto a parallel copy.";

const ORACLE_IS_NOT_THE_KERNEL: &str = "The reference is an f64 host dot product written from the \
     definition of the projection -- sum over K of the bf16 weight times the bf16 activation, \
     rounded once to bf16 at the end. It is not a call into any shipped kernel and not a \
     comparison against the decode twin: the sg == tree == mk chain that covers the q8 mk entries \
     proves the three arms agree, which a defect present in all three survives.";

const EVERY_TOKEN_CARRIES_ITS_OWN_ACTIVATIONS: &str = "Every row of the M block is given DIFFERENT \
     activations. With one shared activation row the token stride is unobservable and the M-row \
     kernel degenerates into M copies of a GEMV -- a stride mutant would then pass. The state this \
     kernel carries is the unrolled per-token guard `if (Tu < mm)`, so the corpus runs m = 1, \
     where every guard but the first must be off, and m = MK_ROWS, where none of them may be; and \
     the destination is always sized for MK_ROWS tokens with the tail past m left poisoned, so a \
     guard wired open is caught by what it writes rather than by an out-of-bounds store whose \
     result is the robust-access rule's to decide.";

const MK_ROWS: usize = 16;

const TAIL_POISON: u32 = 0xdead_beef;

const NOISE_C: f64 = 16.0;

const BOUNDS_DOC: &str = "The bound is ONE bf16 ulp per element against the f64 reference, not an \
     aggregate norm: an aggregate norm over a bf16 output saturates at half a bf16 ulp and cannot \
     see a projection defect at all. NOISE_C=16 spells the same f32 accumulation floor the fp8 \
     epilogue suite uses: a row whose true dot product sits below 16*f32::EPSILON times the sum of \
     the magnitudes of its own terms is not resolvable to one bf16 ulp by ANY f32 kernel, so those \
     rows are gated on absolute error against that floor instead. The floor never relaxes a \
     typical row -- it only excuses catastrophic cancellation, and the corpus screen rejects any \
     case where it would relax more than a quarter of the outputs.";

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_g4w_mk_bf16_oracle needs a real wgpu adapter; a skipped numeric gate reads as a \
         passed one, so this panics rather than returning early",
    )
}

fn source() -> String {
    let src = g4w::mk_bf16_shader_source(MK_ROWS);
    for e in [PK_ENTRY, PK3_ENTRY] {
        assert!(
            src.contains(&format!("fn {e}(")),
            "mk_bf16_shader_source({MK_ROWS}) no longer declares {e}; the entry moved and this \
             gate is now testing nothing. {WHY}"
        );
    }
    src
}

fn pack(src: &[u16]) -> Vec<u32> {
    let mut out = vec![0u32; src.len().div_ceil(2).max(1)];
    for (i, v) in src.iter().enumerate() {
        out[i / 2] |= (*v as u32) << (16 * (i % 2));
    }
    out
}

fn bf16_to_f64(bits: u16) -> f64 {
    f32::from_bits((bits as u32) << 16) as f64
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvParams {
    n_rows: u32,
    k_elems: u32,
    w_row_words: u32,
    groups_x: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MkParams {
    m: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    dst_word_off: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SplitParams {
    q_rows: u32,
    kv_rows: u32,
    v_off: u32,
    pad0: u32,
}

struct Case {
    label: &'static str,
    n: usize,
    k: usize,
    m: usize,
    w: Vec<u16>,
    x: Vec<u16>,
    reference: Vec<Vec<f64>>,
    magnitude: Vec<Vec<f64>>,
}

impl Case {
    fn build(label: &'static str, n: usize, k: usize, m: usize, seed: u64) -> Self {
        assert!(
            n.is_multiple_of(8),
            "{label}: the bf16 mk grid places 8 rows per workgroup and pairs them into words"
        );
        assert!(
            k.is_multiple_of(8),
            "{label}: the vec8 inner loop consumes 8 elements per iteration"
        );
        assert!(
            (1..=MK_ROWS).contains(&m),
            "{label}: m={m} is outside the unroll the shader was generated for"
        );
        let mut rng = Lcg::new(seed);
        let w = rng.bf16_vec(n * k, 0.12);
        let x = rng.bf16_vec(m * k, 0.4);
        let mut reference = Vec::with_capacity(m);
        let mut magnitude = Vec::with_capacity(m);
        for t in 0..m {
            let mut r = vec![0f64; n];
            let mut mag = vec![0f64; n];
            for row in 0..n {
                let mut acc = 0f64;
                let mut ma = 0f64;
                for j in 0..k {
                    let p = bf16_to_f64(w[row * k + j]) * bf16_to_f64(x[t * k + j]);
                    acc += p;
                    ma += p.abs();
                }
                r[row] = acc;
                mag[row] = ma;
            }
            reference.push(r);
            magnitude.push(mag);
        }
        let c = Self {
            label,
            n,
            k,
            m,
            w,
            x,
            reference,
            magnitude,
        };
        c.screen();
        c
    }

    fn screen(&self) {
        let mut distinct = std::collections::BTreeSet::new();
        for t in 0..self.m {
            distinct.insert(&self.x[t * self.k..(t + 1) * self.k]);
        }
        assert_eq!(
            distinct.len(),
            self.m,
            "{}: two tokens were dealt the same activation row, so a collapsed token stride would \
             be invisible. {EVERY_TOKEN_CARRIES_ITS_OWN_ACTIVATIONS}",
            self.label
        );
        let resolvable = self
            .reference
            .iter()
            .zip(self.magnitude.iter())
            .flat_map(|(r, m)| r.iter().zip(m.iter()))
            .filter(|(r, m)| r.abs() > NOISE_C * (f32::EPSILON as f64) * **m)
            .count();
        assert!(
            resolvable * 4 >= self.n * self.m * 3,
            "{}: only {resolvable} of {} outputs are above the f32 accumulation floor, so most of \
             this fixture is gated on the floor rather than on one bf16 ulp and the case proves \
             far less than it appears to. {BOUNDS_DOC}",
            self.label,
            self.n * self.m
        );
    }

    fn params(&self, groups_x: u32) -> GemvParams {
        GemvParams {
            n_rows: self.n as u32,
            k_elems: self.k as u32,
            w_row_words: (self.k / 2) as u32,
            groups_x,
        }
    }
}

struct Report {
    max_ulp: i64,
    worst: (usize, usize),
    got: f32,
    want: f64,
    nonfinite: usize,
    unresolvable: usize,
    worst_floor_ratio: f64,
}

fn report(got: &[Vec<u16>], want: &[Vec<f64>], mag: &[Vec<f64>]) -> Report {
    let mut r = Report {
        max_ulp: 0,
        worst: (0, 0),
        got: 0.0,
        want: 0.0,
        nonfinite: 0,
        unresolvable: 0,
        worst_floor_ratio: 0.0,
    };
    for (t, row) in got.iter().enumerate() {
        for (i, g) in row.iter().enumerate() {
            let gf = f32::from_bits((*g as u32) << 16);
            if !gf.is_finite() {
                r.nonfinite += 1;
            }
            let floor = NOISE_C * (f32::EPSILON as f64) * mag[t][i];
            let w = want[t][i];
            if w.abs() <= floor {
                r.unresolvable += 1;
                let ratio = if floor > 0.0 {
                    ((gf as f64) - w).abs() / floor
                } else {
                    0.0
                };
                r.worst_floor_ratio = r.worst_floor_ratio.max(ratio);
                continue;
            }
            let d = (ord(*g) - ord(half::bf16::from_f64(w).to_bits())).abs();
            if d > r.max_ulp {
                r.max_ulp = d;
                r.worst = (t, i);
                r.got = gf;
                r.want = w;
            }
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
            "max {} bf16 ulp at token {} row {} (got {:e} want {:e}), {} non-finite, {} \
             unresolvable (worst {:.3}x noise floor)",
            self.max_ulp,
            self.worst.0,
            self.worst.1,
            self.got,
            self.want,
            self.nonfinite,
            self.unresolvable,
            self.worst_floor_ratio
        )
    }
}

struct PkRun {
    rows: Vec<Vec<u16>>,
    below_offset: Vec<u32>,
    tail: Vec<u32>,
}

fn run_pk(ctx: &WgpuContext, src: &str, c: &Case, word_off: usize) -> anyhow::Result<PkRun> {
    let grid = dispatch::workgroup_count_1d(ctx, c.n as u64, 8);
    let stride = c.n / 2;
    let total = word_off + MK_ROWS * stride;
    let w = dispatch::storage_from_slice(ctx, "mkb-w", &pack(&c.w));
    let x = dispatch::storage_from_slice(ctx, "mkb-x", &pack(&c.x));
    let y = dispatch::storage_from_slice(ctx, "mkb-y", &vec![TAIL_POISON; total]);
    let p = dispatch::uniform_from(ctx, "mkb-p", &c.params(grid.0));
    let mkp = dispatch::uniform_from(
        ctx,
        "mkb-mkp",
        &MkParams {
            m: c.m as u32,
            x_stride_words: (c.k / 2) as u32,
            y_stride_words: stride as u32,
            dst_word_off: word_off as u32,
        },
    );
    let pipe = dispatch::compute_pipeline(ctx, "g4w-mk-bf16-pk-probe", src, PK_ENTRY)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    dispatch::dispatch(
        ctx,
        &pipe,
        &[(0, &w), (1, &x), (2, &y), (3, &p), (35, &mkp)],
        grid,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let words: Vec<u32> = dispatch::read_back(ctx, &y, total).map_err(|e| anyhow::anyhow!("{e}"))?;
    let rows = (0..c.m)
        .map(|t| {
            let base = word_off + t * stride;
            unpack(&words[base..base + stride], c.n)
        })
        .collect();
    Ok(PkRun {
        rows,
        below_offset: words[..word_off].to_vec(),
        tail: words[word_off + c.m * stride..].to_vec(),
    })
}

type Pk3Out = Vec<(Vec<u16>, Vec<u16>, Vec<u16>)>;

fn run_pk3(ctx: &WgpuContext, src: &str, c: &Case, sp: &Split) -> anyhow::Result<Pk3Out> {
    assert_eq!(sp.v_off, sp.q_rows + sp.kv_rows);
    assert_eq!(c.n, sp.q_rows + 2 * sp.kv_rows);
    let grid = dispatch::workgroup_count_1d(ctx, c.n as u64, 8);
    let qw = sp.q_rows / 2;
    let kvw = sp.kv_rows / 2;
    let w = dispatch::storage_from_slice(ctx, "mkb-w", &pack(&c.w));
    let x = dispatch::storage_from_slice(ctx, "mkb-x", &pack(&c.x));
    let qb = dispatch::storage_from_slice(ctx, "mkb-q", &vec![TAIL_POISON; c.m * qw]);
    let kb = dispatch::storage_from_slice(ctx, "mkb-k", &vec![TAIL_POISON; c.m * kvw]);
    let vb = dispatch::storage_from_slice(ctx, "mkb-v", &vec![TAIL_POISON; c.m * kvw]);
    let p = dispatch::uniform_from(ctx, "mkb-p", &c.params(grid.0));
    let spb = dispatch::uniform_from(
        ctx,
        "mkb-sp",
        &SplitParams {
            q_rows: sp.q_rows as u32,
            kv_rows: sp.kv_rows as u32,
            v_off: sp.v_off as u32,
            pad0: 0,
        },
    );
    let mkp = dispatch::uniform_from(
        ctx,
        "mkb-mkp",
        &MkParams {
            m: c.m as u32,
            x_stride_words: (c.k / 2) as u32,
            y_stride_words: 0,
            dst_word_off: 0,
        },
    );
    let pipe = dispatch::compute_pipeline(ctx, "g4w-mk-bf16-pk3-probe", src, PK3_ENTRY)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    dispatch::dispatch(
        ctx,
        &pipe,
        &[
            (0, &w),
            (1, &x),
            (3, &p),
            (31, &qb),
            (32, &kb),
            (33, &vb),
            (34, &spb),
            (35, &mkp),
        ],
        grid,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let rb = |b: &wgpu::Buffer, n: usize| -> anyhow::Result<Vec<u32>> {
        dispatch::read_back(ctx, b, n).map_err(|e| anyhow::anyhow!("{e}"))
    };
    let q = rb(&qb, c.m * qw)?;
    let k = rb(&kb, c.m * kvw)?;
    let v = rb(&vb, c.m * kvw)?;
    Ok((0..c.m)
        .map(|t| {
            (
                unpack(&q[t * qw..(t + 1) * qw], sp.q_rows),
                unpack(&k[t * kvw..(t + 1) * kvw], sp.kv_rows),
                unpack(&v[t * kvw..(t + 1) * kvw], sp.kv_rows),
            )
        })
        .collect())
}

fn segment(rows: &[Vec<f64>], from: usize, to: usize) -> Vec<Vec<f64>> {
    rows.iter().map(|r| r[from..to].to_vec()).collect()
}

fn pk3_segment(got: &Pk3Out, which: usize) -> Vec<Vec<u16>> {
    got.iter()
        .map(|(q, k, v)| match which {
            0 => q.clone(),
            1 => k.clone(),
            _ => v.clone(),
        })
        .collect()
}

const MUTANTS: [(&str, &str, &str); 6] = [
    (
        "activation-token-stride-collapsed-onto-token-0",
        "let xw = gemv_bf16_x[1u * xs + xo + j];",
        "let xw = gemv_bf16_x[0u * xs + xo + j];",
    ),
    (
        "weight-high-half-reads-the-low-half",
        "            let wh = bf16_hi(ww);\n",
        "            let wh = bf16_lo(ww);\n",
    ),
    (
        "pk-output-token-stride-collapsed-onto-token-0",
        "gemv_bf16_y[g4w_mk_params.dst_word_off + 1u * g4w_mk_params.y_stride_words + (row >> 1u)]",
        "gemv_bf16_y[g4w_mk_params.dst_word_off + 0u * g4w_mk_params.y_stride_words + (row >> 1u)]",
    ),
    (
        "pk3-q-segment-token-stride-collapsed-onto-token-0",
        "g4w_y_q[1u * (g4w_split_params.q_rows >> 1u) + (row >> 1u)] = word;",
        "g4w_y_q[0u * (g4w_split_params.q_rows >> 1u) + (row >> 1u)] = word;",
    ),
    (
        "pk3-v-segment-offset-plus-one-pair",
        "                    let vr = row - g4w_split_params.v_off;\n",
        "                    let vr = row - g4w_split_params.v_off + 2u;\n",
    ),
    (
        "token-count-guard-wired-open-to-the-full-unroll",
        "    let mm = g4w_mk_params.m;\n",
        "    let mm = 16u;\n",
    ),
];

const PK_MUTANTS: [usize; 3] = [0, 1, 2];
const PK3_MUTANTS: [usize; 4] = [0, 1, 3, 4];
const GUARD_MUTANT: usize = 5;

fn mutate(src: &str, from: &str, to: &str) -> String {
    assert!(
        src.contains(from),
        "mutation anchor not present in mk_bf16_shader_source: {from:?}. This gate is worthless \
         if its mutants no longer apply -- fix the anchors."
    );
    src.replace(from, to)
}

fn pk_cases() -> Vec<Case> {
    vec![
        Case::build("dense projection block", 256, 1024, 5, 0x000b_f160),
        Case::build("odd row block", 40, 512, 3, 0x0000_0dd0),
        Case::build("single token, every unrolled guard off but one", 64, 256, 1, 0x0000_0511),
        Case::build("full unroll, no guard off", 32, 128, MK_ROWS, 0x0000_f011),
    ]
}

fn pk3_case() -> (Case, Split) {
    (
        Case::build("fused qkv split scatter", 64, 256, 5, 0x0000_9a71),
        Split {
            q_rows: 32,
            kv_rows: 16,
            v_off: 48,
        },
    )
}

#[test]
fn every_mutation_anchor_still_applies_to_the_shipped_mk_bf16_text() {
    let src = source();
    for (name, from, to) in MUTANTS {
        assert!(
            src.contains(from),
            "anchor for mutant {name} is gone from mk_bf16_shader_source({MK_ROWS}): {from:?}. A \
             mutant whose anchor rotted is silently inert, and the GPU tests that would have \
             caught that do not run on a box with no adapter -- which is why this check is \
             CPU-only and unconditional."
        );
        assert_ne!(
            from, to,
            "mutant {name} replaces text with itself and can never turn anything red"
        );
    }
    let assigned: std::collections::BTreeSet<usize> = PK_MUTANTS
        .iter()
        .chain(PK3_MUTANTS.iter())
        .chain([GUARD_MUTANT].iter())
        .copied()
        .collect();
    assert_eq!(
        assigned.len(),
        MUTANTS.len(),
        "{} of {} mutants are assigned to a corpus; an unassigned mutant is one nothing ever runs",
        assigned.len(),
        MUTANTS.len()
    );
    eprintln!("{WHY}\n{ORACLE_IS_NOT_THE_KERNEL}\n{EVERY_TOKEN_CARRIES_ITS_OWN_ACTIVATIONS}");
}

#[test]
fn the_shipped_mk_bf16_source_is_the_one_this_gate_compiles() {
    let src = source();
    assert!(
        src.contains("fn gemv_bf16_reduce("),
        "mk_bf16_shader_source no longer carries the bf16 gemv prelude it is composed with, so \
         the accessor has stopped returning what build_mk_pipelines builds. {WHY}"
    );
    assert!(
        src.contains(&format!("acc{} = 0.0;", MK_ROWS - 1)),
        "mk_bf16_shader_source({MK_ROWS}) did not unroll {MK_ROWS} token accumulators; the gate \
         would then never reach the tokens it claims to cover"
    );
    let narrow = g4w::mk_bf16_shader_source(2);
    assert!(
        !narrow.contains("acc2 = 0.0;"),
        "mk_bf16_shader_source(2) unrolled more accumulators than it was asked for, so mk_max is \
         not the knob it appears to be"
    );
}

#[test]
fn mk_bf16_pk_matches_an_f64_host_reference() {
    let ctx = ctx();
    eprintln!("[mk-bf16-oracle] adapter: {}\n{WHY}", ctx.info.name);
    let src = source();
    for c in pk_cases() {
        for word_off in [0usize, 24] {
            let run = run_pk(ctx, &src, &c, word_off)
                .unwrap_or_else(|e| panic!("{}: pk dispatch: {e}", c.label));
            let r = report(&run.rows, &c.reference, &c.magnitude);
            eprintln!(
                "[{}] {PK_ENTRY} n={} k={} m={} off={word_off}: {r}",
                c.label, c.n, c.k, c.m
            );
            r.check(&format!("[{}] {PK_ENTRY}", c.label));
            assert!(
                run.below_offset.iter().all(|w| *w == TAIL_POISON),
                "[{}] dst_word_off={word_off} was not honoured: the M-row epilogue wrote below \
                 its own offset",
                c.label
            );
            assert!(
                run.tail.iter().all(|w| *w == TAIL_POISON),
                "[{}] the M-row epilogue wrote past token m={}; the per-token guard is not \
                 holding and the destination of a shorter prefill chunk would be corrupted",
                c.label,
                c.m
            );
        }
    }
}

#[test]
fn mk_bf16_pk3_split_scatter_matches_an_f64_host_reference() {
    let ctx = ctx();
    let src = source();
    let (c, sp) = pk3_case();
    let got = run_pk3(ctx, &src, &c, &sp).expect("pk3 dispatch");
    for (which, seg, from, to) in [
        (0usize, "Q", 0usize, sp.q_rows),
        (1, "K", sp.q_rows, sp.v_off),
        (2, "V", sp.v_off, c.n),
    ] {
        let r = report(
            &pk3_segment(&got, which),
            &segment(&c.reference, from, to),
            &segment(&c.magnitude, from, to),
        );
        eprintln!("[{}] {PK3_ENTRY} segment {seg}: {r}", c.label);
        r.check(&format!("[{}] {PK3_ENTRY} segment {seg}", c.label));
    }
}

#[test]
fn every_mk_bf16_mutant_is_caught_by_this_corpus() {
    let ctx = ctx();
    let src = source();
    let cases = pk_cases();
    for i in PK_MUTANTS {
        let (name, from, to) = MUTANTS[i];
        let bad = mutate(&src, from, to);
        let mut caught_by: Vec<&str> = Vec::new();
        for c in &cases {
            let caught = match run_pk(ctx, &bad, c, 0) {
                Ok(run) => report(&run.rows, &c.reference, &c.magnitude).caught(),
                Err(_) => true,
            };
            if caught {
                caught_by.push(c.label);
            }
        }
        assert!(
            !caught_by.is_empty(),
            "mutant {name} was NOT caught by any pk case. An M-row projection a broken stride \
             passes is not a gate. {WHY} {EVERY_TOKEN_CARRIES_ITS_OWN_ACTIVATIONS}"
        );
        eprintln!("PK MUTANT {name}: caught by {caught_by:?}");
    }

    let (c, sp) = pk3_case();
    let want = [
        segment(&c.reference, 0, sp.q_rows),
        segment(&c.reference, sp.q_rows, sp.v_off),
        segment(&c.reference, sp.v_off, c.n),
    ];
    let mag = [
        segment(&c.magnitude, 0, sp.q_rows),
        segment(&c.magnitude, sp.q_rows, sp.v_off),
        segment(&c.magnitude, sp.v_off, c.n),
    ];
    for i in PK3_MUTANTS {
        let (name, from, to) = MUTANTS[i];
        let bad = mutate(&src, from, to);
        let caught = match run_pk3(ctx, &bad, &c, &sp) {
            Ok(got) => {
                let mut hit = false;
                for (s, (w, m)) in want.iter().zip(mag.iter()).enumerate() {
                    let r = report(&pk3_segment(&got, s), w, m);
                    if r.caught() {
                        eprintln!("PK3 MUTANT {name} caught in segment {s}: {r}");
                        hit = true;
                    }
                }
                hit
            }
            Err(_) => true,
        };
        assert!(
            caught,
            "pk3 mutant {name} was NOT caught by any of Q/K/V. The fused q/k/v split scatter is \
             exactly where an index or stride error hides. {WHY}"
        );
    }
}

#[test]
fn only_a_short_chunk_can_see_the_per_token_guard() {
    let ctx = ctx();
    let src = source();
    let (name, from, to) = MUTANTS[GUARD_MUTANT];
    assert_eq!(name, "token-count-guard-wired-open-to-the-full-unroll");
    let bad = mutate(&src, from, to);
    let mut short_caught = 0usize;
    let mut full_survived = false;
    for c in pk_cases() {
        let caught = match run_pk(ctx, &bad, &c, 0) {
            Ok(run) => {
                report(&run.rows, &c.reference, &c.magnitude).caught()
                    || run.tail.iter().any(|w| *w != TAIL_POISON)
            }
            Err(_) => true,
        };
        eprintln!("[{}] m={} guard mutant caught={caught}", c.label, c.m);
        if c.m == MK_ROWS {
            full_survived = !caught;
        } else if caught {
            short_caught += 1;
        }
    }
    assert!(
        short_caught > 0,
        "no chunk shorter than the unroll noticed the per-token guard being wired open, so the \
         corpus never reaches the state the unrolled token loop carries and every guard in the \
         shader is untested. {EVERY_TOKEN_CARRIES_ITS_OWN_ACTIVATIONS}"
    );
    assert!(
        full_survived,
        "the m = {MK_ROWS} case also caught a guard that can only differ below m = {MK_ROWS}, \
         which contradicts the unroll: at m = {MK_ROWS} the guard is true for every token anyway. \
         This suite is then measuring something other than what it names."
    );
}
