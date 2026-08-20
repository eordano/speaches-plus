#![cfg(feature = "wgpu")]

mod common;
use common::wgpu_allow_skip;
use half::bf16;
use nv_kernels::wgpu_backend::dequant::{bytes_to_words, NVFP4_BLOCK_SIZE};
use nv_kernels::wgpu_backend::device::{shared_or_reason, WgpuContext};
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::gemv_nvfp4::{self, GemvVariant};
use nv_quant::nvfp4::{decode_e2m1, decode_ue4m3, swizzle_scales};
use common::LcgShift32TwoSided as Lcg;
use common::ctx_or_skip_reasoned as ctx_or_skip;
use common::swizzled_scale_dst;

fn scale_byte(rng: &mut Lcg, lo_exp: u32, hi_exp: u32) -> u8 {
    let r = rng.next_u32();
    let exp = lo_exp + (r % (hi_exp - lo_exp + 1));
    let mant = (r >> 8) & 7;
    ((exp << 3) | mant) as u8
}

struct Case {
    w_words: Vec<u32>,
    ws_swizzled: Vec<u8>,
    ws_linear: Vec<u8>,
    x_words: Vec<u32>,
    xs: Vec<u8>,
    alpha: f32,
    n: usize,
    k: usize,
}

fn build_case(seed: u64, n: usize, k: usize, lo_exp: u32, hi_exp: u32) -> Case {
    let mut rng = Lcg(seed);
    let k_blocks = k / NVFP4_BLOCK_SIZE;
    let row_words = k / 8;
    let mut w_words = vec![0u32; n * row_words];
    for v in w_words.iter_mut() {
        *v = rng.next_u32();
    }
    let mut x_words = vec![0u32; row_words];
    for v in x_words.iter_mut() {
        *v = rng.next_u32();
    }
    let mut ws_linear = vec![0u8; n * k_blocks];
    for v in ws_linear.iter_mut() {
        *v = scale_byte(&mut rng, lo_exp, hi_exp);
    }
    let ws_swizzled = swizzle_scales(&ws_linear, n, k_blocks);
    let mut xs = vec![0u8; k_blocks];
    for v in xs.iter_mut() {
        *v = scale_byte(&mut rng, lo_exp, hi_exp);
    }
    Case {
        w_words,
        ws_swizzled,
        ws_linear,
        x_words,
        xs,
        alpha: 0.75,
        n,
        k,
    }
}

fn run_variant(ctx: &WgpuContext, c: &Case, variant: GemvVariant) -> Vec<u16> {
    let k_blocks = c.k / NVFP4_BLOCK_SIZE;
    let w_scale_words = bytes_to_words(&c.ws_swizzled);
    let x_scale_words = bytes_to_words(&c.xs);
    let w_buf = dispatch::storage_from_slice(ctx, "vcmp-w", &c.w_words);
    let ws_buf = dispatch::storage_from_slice(ctx, "vcmp-ws", &w_scale_words);
    let x_buf = dispatch::storage_from_slice(ctx, "vcmp-x", &c.x_words);
    let xs_buf = dispatch::storage_from_slice(ctx, "vcmp-xs", &x_scale_words);
    let y_buf = dispatch::storage_zeroed(ctx, "vcmp-y", (c.n * 4) as u64);

    let groups = dispatch::workgroup_count_1d(ctx, c.n as u64, variant.rows_per_group());
    let params = gemv_nvfp4::gemv_params(c.alpha, c.n, c.k, groups.0);
    assert_eq!(params.k_blocks as usize, k_blocks);
    let params_buf = dispatch::uniform_from(ctx, "vcmp-params", &params);

    dispatch::run(
        ctx,
        variant.label(),
        &variant.source(),
        variant.entry(),
        &[
            (0, &w_buf),
            (1, &ws_buf),
            (2, &x_buf),
            (3, &xs_buf),
            (4, &params_buf),
            (5, &y_buf),
        ],
        groups,
    )
    .unwrap_or_else(|e| panic!("{:?} dispatch n={} k={}: {e}", variant, c.n, c.k));

    let words: Vec<u32> = dispatch::read_back(ctx, &y_buf, c.n).expect("readback");
    words.iter().map(|w| (*w & 0xffff) as u16).collect()
}

fn nibble(words: &[u32], base_word: usize, idx: usize) -> u8 {
    let w = words[base_word + idx / 8];
    ((w >> (4 * (idx % 8))) & 0xf) as u8
}

fn oracle_row(c: &Case, row: usize) -> f64 {
    let k_blocks = c.k / NVFP4_BLOCK_SIZE;
    let row_words = c.k / 8;
    let mut acc = 0f64;
    for kb in 0..k_blocks {
        let ws = decode_ue4m3(c.ws_linear[row * k_blocks + kb]) as f64;
        let xs = decode_ue4m3(c.xs[kb]) as f64;
        let mut dot = 0f64;
        for i in 0..NVFP4_BLOCK_SIZE {
            let wn = nibble(&c.w_words, row * row_words, kb * NVFP4_BLOCK_SIZE + i);
            let xn = nibble(&c.x_words, 0, kb * NVFP4_BLOCK_SIZE + i);
            dot += decode_e2m1(wn) as f64 * decode_e2m1(xn) as f64;
        }
        acc += dot * ws * xs;
    }
    acc * c.alpha as f64
}

fn ulp(a: u16, b: u16) -> i32 {
    (a as i32 - b as i32).abs()
}

struct Report {
    rows: usize,
    diff_rows: usize,
    max_ulp: i32,
    max_rel: f64,
    tree_max_ulp_vs_oracle: i32,
    sg_max_ulp_vs_oracle: i32,
    sampled: usize,
}

fn compare(ctx: &WgpuContext, c: &Case, sample: usize) -> Report {
    let tree = run_variant(ctx, c, GemvVariant::Tree);
    let sg = run_variant(ctx, c, GemvVariant::Sg);
    assert_eq!(tree.len(), c.n);
    assert_eq!(sg.len(), c.n);

    let nonzero = tree.iter().filter(|b| **b & 0x7fff != 0).count();
    let distinct: std::collections::BTreeSet<u16> = tree.iter().copied().collect();
    assert!(
        nonzero * 4 >= c.n * 3,
        "degenerate case n={} k={}: only {nonzero}/{} rows are nonzero, the comparison would be vacuous",
        c.n,
        c.k,
        c.n
    );
    assert!(
        distinct.len() >= (c.n / 2).clamp(1, 8),
        "degenerate case n={} k={}: only {} distinct outputs across {} rows",
        c.n,
        c.k,
        distinct.len(),
        c.n
    );

    let mut diff_rows = 0usize;
    let mut max_ulp = 0i32;
    let mut max_rel = 0f64;
    for r in 0..c.n {
        if tree[r] != sg[r] {
            diff_rows += 1;
            max_ulp = max_ulp.max(ulp(tree[r], sg[r]));
            let a = bf16::from_bits(tree[r]).to_f32() as f64;
            let b = bf16::from_bits(sg[r]).to_f32() as f64;
            let denom = a.abs().max(b.abs()).max(1e-30);
            max_rel = max_rel.max((a - b).abs() / denom);
        }
    }

    let stride = (c.n / sample.max(1)).max(1);
    let mut sampled = 0usize;
    let mut tree_max = 0i32;
    let mut sg_max = 0i32;
    let mut r = 0usize;
    while r < c.n {
        let want = bf16::from_f32(oracle_row(c, r) as f32).to_bits();
        tree_max = tree_max.max(ulp(tree[r], want));
        sg_max = sg_max.max(ulp(sg[r], want));
        sampled += 1;
        r += stride;
    }

    Report {
        rows: c.n,
        diff_rows,
        max_ulp,
        max_rel,
        tree_max_ulp_vs_oracle: tree_max,
        sg_max_ulp_vs_oracle: sg_max,
        sampled,
    }
}

fn report(name: &str, c: &Case, r: &Report) {
    println!(
        "gemv_nvfp4 variant-cmp {name:<16} n={:<6} k={:<6} rows={} tree_vs_sg_diff_rows={} ({:.4}%) max_bf16_ulp={} max_rel={:.3e} | vs f64 oracle over {} sampled rows: tree_max_ulp={} sg_max_ulp={}",
        c.n,
        c.k,
        r.rows,
        r.diff_rows,
        100.0 * r.diff_rows as f64 / r.rows as f64,
        r.max_ulp,
        r.max_rel,
        r.sampled,
        r.tree_max_ulp_vs_oracle,
        r.sg_max_ulp_vs_oracle
    );
}

fn subgroup_or_skip(what: &str, ctx: &WgpuContext) -> bool {
    if !gemv_nvfp4::sg32_ok(ctx) {
        if !wgpu_allow_skip() {
            panic!(
                "{what}: adapter has no usable subgroups (subgroup={} min={} max={} probe={:?}), \
                 so the Sg variant is unreachable and this gate would report success having \
                 compared nothing. Set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose.",
                ctx.caps.subgroup,
                ctx.caps.subgroup_min_size,
                ctx.caps.subgroup_max_size,
                ctx.subgroup_width()
            );
        }
        eprintln!(
            "{what}: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) adapter has no usable subgroups (subgroup={} min={} max={} probe={:?})",
            ctx.caps.subgroup,
            ctx.caps.subgroup_min_size,
            ctx.caps.subgroup_max_size,
            ctx.subgroup_width()
        );
        return false;
    }
    true
}

#[test]
fn tree_is_the_selected_variant_only_without_subgroups() {
    let Some(ctx) = ctx_or_skip("tree_is_the_selected_variant_only_without_subgroups") else {
        return;
    };
    let picked = gemv_nvfp4::select_variant(ctx);
    println!(
        "select_variant={:?} subgroup_ok={} sg32_ok={} subgroup={} min={} max={} probe={:?}",
        picked,
        gemv_nvfp4::subgroup_ok(ctx),
        gemv_nvfp4::sg32_ok(ctx),
        ctx.caps.subgroup,
        ctx.caps.subgroup_min_size,
        ctx.caps.subgroup_max_size,
        ctx.subgroup_width(),
    );
    if gemv_nvfp4::sg32_ok(ctx) {
        assert_eq!(picked, GemvVariant::Sg);
    } else {
        assert_eq!(picked, GemvVariant::Tree);
    }
}

#[test]
fn tree_and_sg_agree_on_the_real_gemma4_31b_projection_shapes() {
    let Some(ctx) = ctx_or_skip("tree_and_sg_agree_on_the_real_gemma4_31b_projection_shapes")
    else {
        return;
    };
    if !subgroup_or_skip(
        "tree_and_sg_agree_on_the_real_gemma4_31b_projection_shapes",
        ctx,
    ) {
        return;
    }

    let shapes: &[(&str, usize, usize)] = &[
        ("qkv_q", 8192, 5376),
        ("qkv_kv", 4096, 5376),
        ("o_proj", 5376, 8192),
        ("gate_up", 8192, 5376),
        ("down", 5376, 21504),
    ];

    let mut worst_ulp = 0i32;
    let mut worst_rel = 0f64;
    let mut worst_oracle = 0i32;
    let mut ran = 0usize;
    for (i, (name, n, k)) in shapes.iter().enumerate() {
        let bytes = (*n as u64) * (*k as u64) / 2;
        if bytes > ctx.caps.max_storage_buffer_binding_size {
            eprintln!("skipping {name}: {bytes} B exceeds max_storage_buffer_binding_size");
            continue;
        }
        ran += 1;
        let c = build_case(0x5eed_0001 + i as u64 * 977, *n, *k, 5, 9);
        let r = compare(ctx, &c, 24);
        report(name, &c, &r);
        worst_ulp = worst_ulp.max(r.max_ulp);
        worst_rel = worst_rel.max(r.max_rel);
        worst_oracle = worst_oracle
            .max(r.tree_max_ulp_vs_oracle)
            .max(r.sg_max_ulp_vs_oracle);
    }

    assert_eq!(
        ran,
        shapes.len(),
        "only {ran} of {} Gemma4-31B shapes were compared; the rest fell past the storage \
         binding limit and the worst-case assertions below would be scored on nothing",
        shapes.len()
    );
    println!(
        "gemma4-31b shapes summary: worst_tree_vs_sg_ulp={worst_ulp} worst_rel={worst_rel:.3e} worst_ulp_vs_f64_oracle={worst_oracle}"
    );
    assert!(
        worst_ulp <= 1,
        "Tree and Sg disagree by {worst_ulp} bf16 ulp on a real Gemma4-31B projection shape"
    );
    assert!(
        worst_oracle <= 1,
        "a variant is {worst_oracle} bf16 ulp off the f64 oracle on a real Gemma4-31B projection shape"
    );
}

#[test]
fn tree_and_sg_agree_on_row_count_edge_cases() {
    let Some(ctx) = ctx_or_skip("tree_and_sg_agree_on_row_count_edge_cases") else {
        return;
    };
    if !subgroup_or_skip("tree_and_sg_agree_on_row_count_edge_cases", ctx) {
        return;
    }

    let mut worst_ulp = 0i32;
    let mut worst_oracle = 0i32;
    for (i, n) in [
        1usize, 2, 3, 4, 5, 7, 31, 32, 33, 127, 128, 129, 131, 255, 257,
    ]
    .iter()
    .enumerate()
    {
        let c = build_case(0x00ed_2000 + i as u64 * 131, *n, 5376, 5, 9);
        let r = compare(ctx, &c, *n);
        report(&format!("n={n}"), &c, &r);
        worst_ulp = worst_ulp.max(r.max_ulp);
        worst_oracle = worst_oracle
            .max(r.tree_max_ulp_vs_oracle)
            .max(r.sg_max_ulp_vs_oracle);
    }
    assert_eq!(
        worst_ulp, 0,
        "Tree and Sg must be bit-identical on the row-count edge cases"
    );
    assert!(
        worst_oracle <= 1,
        "variant is {worst_oracle} ulp off oracle"
    );
}

#[test]
fn tree_and_sg_agree_on_k_block_edge_cases() {
    let Some(ctx) = ctx_or_skip("tree_and_sg_agree_on_k_block_edge_cases") else {
        return;
    };
    if !subgroup_or_skip("tree_and_sg_agree_on_k_block_edge_cases", ctx) {
        return;
    }

    let mut worst_ulp = 0i32;
    let mut worst_oracle = 0i32;
    for (i, k) in [
        16usize,
        32,
        48,
        64,
        4080,
        4096,
        4112,
        21504 - 16,
        21504 + 16,
    ]
    .iter()
    .enumerate()
    {
        let c = build_case(0x00ed_3000 + i as u64 * 179, 37, *k, 5, 9);
        let r = compare(ctx, &c, 37);
        report(&format!("k={k}"), &c, &r);
        worst_ulp = worst_ulp.max(r.max_ulp);
        worst_oracle = worst_oracle
            .max(r.tree_max_ulp_vs_oracle)
            .max(r.sg_max_ulp_vs_oracle);
    }
    assert!(
        worst_ulp <= 1,
        "Tree and Sg disagree by {worst_ulp} bf16 ulp on a k edge case"
    );
    assert!(
        worst_oracle <= 1,
        "variant is {worst_oracle} ulp off oracle"
    );
}

#[test]
fn tree_and_sg_agree_under_wide_scale_dynamic_range() {
    let Some(ctx) = ctx_or_skip("tree_and_sg_agree_under_wide_scale_dynamic_range") else {
        return;
    };
    if !subgroup_or_skip("tree_and_sg_agree_under_wide_scale_dynamic_range", ctx) {
        return;
    }

    let mut worst_ulp = 0i32;
    for (i, (lo, hi)) in [(0u32, 15u32), (1, 4), (11, 15), (0, 0)].iter().enumerate() {
        let c = build_case(0x00ed_4000 + i as u64 * 313, 129, 5376, *lo, *hi);
        let tree = run_variant(ctx, &c, GemvVariant::Tree);
        let sg = run_variant(ctx, &c, GemvVariant::Sg);
        let mut diff = 0usize;
        let mut mu = 0i32;
        let mut nonfinite = 0usize;
        for r in 0..c.n {
            if !bf16::from_bits(tree[r]).to_f32().is_finite()
                || !bf16::from_bits(sg[r]).to_f32().is_finite()
            {
                nonfinite += 1;
            }
            if tree[r] != sg[r] {
                diff += 1;
                mu = mu.max(ulp(tree[r], sg[r]));
            }
        }
        println!(
            "gemv_nvfp4 scale-range exp=[{lo},{hi}] n={} k={}: diff_rows={diff} max_ulp={mu} nonfinite_rows={nonfinite}",
            c.n, c.k
        );
        worst_ulp = worst_ulp.max(mu);
    }
    assert!(
        worst_ulp <= 1,
        "Tree and Sg disagree by {worst_ulp} bf16 ulp under a wide scale range"
    );
}
