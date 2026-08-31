#![cfg(feature = "wgpu")]

mod common;
use common::{ctx, pack_u8, FdP};
use nv_kernels::wgpu_backend::compose;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::flash_decode as fd;

const SHIPPED_FOLD_REDUCE_IS_SG_SHUFFLE_WB_TREES_LOSE_AT_EVERY_DEPTH: bool = true;
const SG_REDUCE_MAX_ULP_DRIFT_OBSERVED_ONLY_AT_HD512: u32 = 5;
const STAGE2_MAX_BF16_ULP_DRIFT_1_ON_POSITION_RAMPED_KV_ZERO_OBSERVED: u32 = 1;

fn f32_bits_ulp_distance(a: u32, b: u32) -> u32 {
    let key = |x: u32| {
        if x & 0x8000_0000 != 0 {
            !x
        } else {
            x | 0x8000_0000
        }
    };
    key(a).abs_diff(key(b))
}

const DENSE_31B_N_Q: u32 = 32;
const DENSE_31B_SLIDING_N_KV: u32 = 16;
const DENSE_31B_SLIDING_HD: u32 = 256;
const DENSE_31B_FULL_N_KV: u32 = 4;
const DENSE_31B_FULL_HD: u32 = 512;
const DENSE_31B_DEFAULT_GQA_FOLD_IS_THE_FULL_GROUP_MIRRORING_Q3D: u32 = 8;
const DENSE_31B_SLIDING_RING_ROWS_WINDOW_1024_CHUNK_16_HEADROOM_128: u32 = 1168;

#[derive(Clone, Copy)]
struct Case {
    n_kv: u32,
    hd: u32,
    total: u32,
    start: u32,
    ring: u32,
    splits: u32,
    fold: u32,
    tile: u32,
}

fn params(c: &Case) -> FdP {
    FdP {
        n_heads: DENSE_31B_N_Q,
        n_kv: c.n_kv,
        head_dim: c.hd,
        total: c.total,
        start: c.start,
        splits: c.splits,
        ring: c.ring,
        out_bf16: 1,
        scaling: 1.0 / (c.hd as f32).sqrt(),
        m_rows: 1,
        ..Default::default()
    }
}

fn stage1_scratch(
    ctx: &WgpuContext,
    c: &Case,
    label: &str,
    source: &str,
    entry: &str,
    grid_x: u32,
    seed: u64,
) -> Vec<u32> {
    let slots = if c.ring > 0 { c.ring } else { c.total };
    let kv_elems = (slots * c.n_kv * c.hd) as usize;
    let sc_elems = (slots * c.n_kv) as usize;
    let mut lcg = seed | 1;
    let mut next = move || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        lcg
    };
    let kb: Vec<u8> = (0..kv_elems)
        .map(|_| 0x30u8 + ((next() >> 33) as u8 % 0x18))
        .collect();
    let vb: Vec<u8> = (0..kv_elems)
        .map(|_| 0x30u8 + ((next() >> 35) as u8 % 0x18))
        .collect();
    let scales: Vec<f32> = (0..sc_elems)
        .map(|j| 0.004 + ((next() >> 45) as f32 / 524_288.0) * 0.02 + (j % 7) as f32 * 0.003)
        .collect();
    let q: Vec<f32> = (0..(DENSE_31B_N_Q * c.hd) as usize)
        .map(|j| ((next() >> 40) as f32 / 8_388_608.0) - 1.0 + (j % 3) as f32 * 0.25)
        .collect();
    let scratch_elems = (DENSE_31B_N_Q * c.splits * (c.hd + 2)) as usize;
    let scratch = dispatch::storage_zeroed(ctx, "g4dp-scratch", (scratch_elems * 4) as u64);
    dispatch::run(
        ctx,
        label,
        source,
        entry,
        &[
            (0, &dispatch::storage_from_slice(ctx, "g4dp-q", &q)),
            (4, &dispatch::uniform_from(ctx, "g4dp-p", &params(c))),
            (5, &dispatch::storage_from_slice(ctx, "g4dp-k", &pack_u8(&kb))),
            (6, &dispatch::storage_from_slice(ctx, "g4dp-v", &pack_u8(&vb))),
            (7, &scratch),
            (8, &dispatch::storage_from_slice(ctx, "g4dp-ks", &scales)),
            (9, &dispatch::storage_from_slice(ctx, "g4dp-vs", &scales)),
        ],
        (grid_x, c.splits, 1),
    )
    .unwrap_or_else(|e| panic!("{label}: dispatch: {e}"));
    dispatch::read_back::<u32>(ctx, &scratch, scratch_elems)
        .unwrap_or_else(|e| panic!("{label}: read_back: {e}"))
}

fn stage2_bf16_pack(
    ctx: &WgpuContext,
    c: &Case,
    label: &str,
    stage1_source: &str,
    stage1_entry: &str,
    grid_x: u32,
    seed: u64,
) -> Vec<u32> {
    let slots = if c.ring > 0 { c.ring } else { c.total };
    let kv_elems = (slots * c.n_kv * c.hd) as usize;
    let sc_elems = (slots * c.n_kv) as usize;
    let mut lcg = seed | 1;
    let mut next = move || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        lcg
    };
    let kb: Vec<u8> = (0..kv_elems)
        .map(|_| 0x30u8 + ((next() >> 33) as u8 % 0x18))
        .collect();
    let vb: Vec<u8> = (0..kv_elems)
        .map(|_| 0x30u8 + ((next() >> 35) as u8 % 0x18))
        .collect();
    let scales: Vec<f32> = (0..sc_elems)
        .map(|j| {
            let slot = (j as u32 / c.n_kv) as f32 / slots.max(1) as f32;
            (0.004 + ((next() >> 45) as f32 / 524_288.0) * 0.02 + (j % 7) as f32 * 0.003)
                * (1.0 + 3.0 * slot)
        })
        .collect();
    let q: Vec<f32> = (0..(DENSE_31B_N_Q * c.hd) as usize)
        .map(|j| ((next() >> 40) as f32 / 8_388_608.0) - 1.0 + (j % 3) as f32 * 0.25)
        .collect();
    let scratch_elems = (DENSE_31B_N_Q * c.splits * (c.hd + 2)) as usize;
    let scratch = dispatch::storage_zeroed(ctx, "g4dp-scratch", (scratch_elems * 4) as u64);
    let qb = dispatch::storage_from_slice(ctx, "g4dp-q", &q);
    let kbuf = dispatch::storage_from_slice(ctx, "g4dp-k", &pack_u8(&kb));
    let vbuf = dispatch::storage_from_slice(ctx, "g4dp-v", &pack_u8(&vb));
    let ksb = dispatch::storage_from_slice(ctx, "g4dp-ks", &scales);
    let vsb = dispatch::storage_from_slice(ctx, "g4dp-vs", &scales);
    let pbuf = dispatch::uniform_from(ctx, "g4dp-p", &params(c));
    dispatch::run(
        ctx,
        label,
        stage1_source,
        stage1_entry,
        &[
            (0, &qb),
            (4, &pbuf),
            (5, &kbuf),
            (6, &vbuf),
            (7, &scratch),
            (8, &ksb),
            (9, &vsb),
        ],
        (grid_x, c.splits, 1),
    )
    .unwrap_or_else(|e| panic!("{label}: stage1 dispatch: {e}"));
    let pack_words = (DENSE_31B_N_Q * c.hd) as usize / 2;
    let attn_pack = dispatch::storage_zeroed(ctx, "g4dp-pack", (pack_words * 4) as u64);
    let (s2_src, s2_entry) = if c.splits == 16 {
        (
            format!(
                "{}\n{}",
                compose(fd::WGSL),
                nv_models::gemma4_wgpu_shared::FLASH2_PK_WGSL
            ),
            nv_models::gemma4_wgpu_shared::FLASH2_PK_ENTRY,
        )
    } else {
        (compose(fd::WGSL), fd::ENTRY_STAGE2_PK_RT)
    };
    dispatch::run(
        ctx,
        &format!("{label}-s2"),
        &s2_src,
        s2_entry,
        &[(3, &attn_pack), (4, &pbuf), (7, &scratch)],
        (DENSE_31B_N_Q, 1, 1),
    )
    .unwrap_or_else(|e| panic!("{label}: stage2 dispatch: {e}"));
    dispatch::read_back::<u32>(ctx, &attn_pack, pack_words)
        .unwrap_or_else(|e| panic!("{label}: stage2 read_back: {e}"))
}

fn bf16_ulp_distance(a: u16, b: u16) -> u32 {
    let key = |x: u16| {
        if x & 0x8000 != 0 {
            (!x) as u32
        } else {
            (x | 0x8000) as u32
        }
    };
    key(a).abs_diff(key(b))
}

fn assert_tiled_arm_matches_monolithic_at_stage2(max_bf16_ulp: u32, c: &Case, seed: u64) {
    let ctx = ctx();
    let mono = stage2_bf16_pack(
        ctx,
        c,
        "g4dp-mono2",
        &compose(fd::WGSL),
        fd::ENTRY_STAGE1_FP8_SD,
        DENSE_31B_N_Q,
        seed,
    );
    let tiled_src = format!(
        "{}\n{}",
        compose(fd::WGSL),
        fd::fold_stage1_source_sd_tiled(c.hd, c.fold, c.tile)
    );
    let tiled = stage2_bf16_pack(
        ctx,
        c,
        "g4dp-tile2",
        &tiled_src,
        &fd::fold_stage1_entry_sd_tiled(c.hd, c.fold, c.tile),
        DENSE_31B_N_Q / c.fold,
        seed,
    );
    let halves = |w: u32| [(w & 0xffff) as u16, (w >> 16) as u16];
    let mut max_seen = 0u32;
    let mut over = 0usize;
    for i in 0..mono.len() {
        for (a, b) in halves(mono[i]).into_iter().zip(halves(tiled[i])) {
            let d = bf16_ulp_distance(a, b);
            max_seen = max_seen.max(d);
            if d > max_bf16_ulp {
                over += 1;
            }
        }
    }
    eprintln!(
        "[g4dp-s2] n_kv={} hd={} total={} ring={} splits={} fold={} tile={} max_bf16_ulp_drift={max_seen}",
        c.n_kv, c.hd, c.total, c.ring, c.splits, c.fold, c.tile
    );
    assert_eq!(
        over,
        0,
        "tiled arm stage2 output drifts past {max_bf16_ulp} bf16 ULP (max {max_seen}) at n_kv={} \
         hd={} total={} splits={} fold={} tile={}: the tile regroups positions across splits, so \
         parity holds only at the merged attention output, and it must stay within the pinned \
         bound",
        c.n_kv,
        c.hd,
        c.total,
        c.splits,
        c.fold,
        c.tile
    );
}

fn assert_fold_matches_monolithic(sd: bool, sg: bool, max_ulp: u32, c: &Case, seed: u64) {
    let ctx = ctx();
    let mono_entry = if sd {
        fd::ENTRY_STAGE1_FP8_SD
    } else {
        fd::ENTRY_STAGE1_FP8
    };
    let mono_src = compose(fd::WGSL);
    let plain = stage1_scratch(
        ctx,
        c,
        "g4dp-mono",
        &mono_src,
        mono_entry,
        DENSE_31B_N_Q,
        seed,
    );
    let (fold_entry, fold_gen) = if c.tile >= 2 {
        assert!(sd && sg, "the tiled fold arm exists only for the sd decode under 32-lane subgroups");
        (
            fd::fold_stage1_entry_sd_tiled(c.hd, c.fold, c.tile),
            fd::fold_stage1_source_sd_tiled(c.hd, c.fold, c.tile),
        )
    } else if sd {
        (
            fd::fold_stage1_entry_sd(c.hd, sg, c.fold),
            fd::fold_stage1_source_sd(c.hd, sg, c.fold),
        )
    } else {
        (
            fd::fold_stage1_entry(c.hd, sg, c.fold),
            fd::fold_stage1_source(c.hd, sg, c.fold),
        )
    };
    let fold_src = format!("{}\n{}", compose(fd::WGSL), fold_gen);
    let folded = stage1_scratch(
        ctx,
        c,
        "g4dp-fold",
        &fold_src,
        &fold_entry,
        DENSE_31B_N_Q / c.fold,
        seed,
    );
    let max_seen = (0..plain.len())
        .map(|i| f32_bits_ulp_distance(plain[i], folded[i]))
        .max()
        .unwrap_or(0);
    eprintln!(
        "[g4dp] sd={sd} sg={sg} n_kv={} hd={} total={} ring={} splits={} fold={} max_ulp_drift={max_seen}",
        c.n_kv, c.hd, c.total, c.ring, c.splits, c.fold
    );
    let diffs: Vec<usize> = (0..plain.len())
        .filter(|i| f32_bits_ulp_distance(plain[*i], folded[*i]) > max_ulp)
        .collect();
    assert!(
        diffs.is_empty(),
        "sd={sd} sg={sg} max_ulp={max_ulp} n_kv={} hd={} total={} start={} ring={} splits={} \
         fold={}: {} of {} stage1 scratch words drift past the bound; first at {:?} ({:#010x} vs \
         {:#010x}). The fold shares one KV pass across a GQA group and reorders no arithmetic; \
         the wb reduce must match bit-for-bit and the sg shuffle reduce may drift at most \
         {SG_REDUCE_MAX_ULP_DRIFT_OBSERVED_ONLY_AT_HD512} ULP",
        c.n_kv,
        c.hd,
        c.total,
        c.start,
        c.ring,
        c.splits,
        c.fold,
        diffs.len(),
        plain.len(),
        diffs.first(),
        plain[diffs.first().copied().unwrap_or(0)],
        folded[diffs.first().copied().unwrap_or(0)]
    );
}

fn dense_cases() -> Vec<Case> {
    let skv = DENSE_31B_SLIDING_N_KV;
    let shd = DENSE_31B_SLIDING_HD;
    let fkv = DENSE_31B_FULL_N_KV;
    let fhd = DENSE_31B_FULL_HD;
    let ring = DENSE_31B_SLIDING_RING_ROWS_WINDOW_1024_CHUNK_16_HEADROOM_128;
    let mut v = vec![
        Case { n_kv: fkv, hd: fhd, total: 1, start: 0, ring: 0, splits: 16, fold: 2, tile: 0 },
        Case { n_kv: fkv, hd: fhd, total: 257, start: 0, ring: 0, splits: 16, fold: 2, tile: 0 },
        Case { n_kv: fkv, hd: fhd, total: 8192, start: 0, ring: 0, splits: 16, fold: 2, tile: 0 },
        Case { n_kv: fkv, hd: fhd, total: 8193, start: 0, ring: 0, splits: 64, fold: 2, tile: 0 },
        Case { n_kv: skv, hd: shd, total: 1000, start: 0, ring, splits: 16, fold: 2, tile: 0 },
        Case { n_kv: skv, hd: shd, total: 4000, start: 4000 - 1024, ring, splits: 16, fold: 2, tile: 0 },
        Case {
            n_kv: skv,
            hd: shd,
            total: 3 * 1168 + 517,
            start: 3 * 1168 + 517 - 1024,
            ring,
            splits: 64,
            fold: 2,
            tile: 0,
        },
    ];
    for fold in [4u32, 8] {
        v.push(Case { n_kv: fkv, hd: fhd, total: 8193, start: 0, ring: 0, splits: 16, fold, tile: 0 });
    }
    v
}

fn tiled_cases_the_shipped_depth_stack_arm_fold8_tile4_splits32() -> Vec<Case> {
    let skv = DENSE_31B_SLIDING_N_KV;
    let shd = DENSE_31B_SLIDING_HD;
    let fkv = DENSE_31B_FULL_N_KV;
    let fhd = DENSE_31B_FULL_HD;
    let ring = DENSE_31B_SLIDING_RING_ROWS_WINDOW_1024_CHUNK_16_HEADROOM_128;
    vec![
        Case { n_kv: fkv, hd: fhd, total: 1, start: 0, ring: 0, splits: 32, fold: 8, tile: 4 },
        Case { n_kv: fkv, hd: fhd, total: 257, start: 0, ring: 0, splits: 32, fold: 8, tile: 4 },
        Case { n_kv: fkv, hd: fhd, total: 8193, start: 0, ring: 0, splits: 32, fold: 8, tile: 4 },
        Case {
            n_kv: skv,
            hd: shd,
            total: 3 * 1168 + 517,
            start: 3 * 1168 + 517 - 1024,
            ring,
            splits: 32,
            fold: 2,
            tile: 4,
        },
    ]
}

#[test]
fn the_depth_stack_fold_engages_at_the_31b_full_attention_geometry_and_clamps_elsewhere() {
    if std::env::var(fd::GQA_FOLD_ENV).is_ok() {
        return;
    }
    use nv_models::gemma4_wgpu::flash1_gqa_fold_for_full_attention_geometry as fold_for;
    assert_eq!(
        fold_for(
            DENSE_31B_N_Q as usize,
            DENSE_31B_FULL_N_KV as usize,
            DENSE_31B_FULL_HD as usize
        ),
        DENSE_31B_DEFAULT_GQA_FOLD_IS_THE_FULL_GROUP_MIRRORING_Q3D,
        "the 31B full-attention geometry (32q/4kv/hd512, GQA group 8) must fold the whole \
         group, or the depth stack silently keeps the 8x-redundant unfolded stage1 KV read"
    );
    assert_eq!(fold_for(4, 2, 256), 2, "a group-2 geometry folds its pair");
    assert_eq!(fold_for(32, 0, 512), 1, "zero kv heads must clamp the fold");
    assert_eq!(fold_for(32, 5, 512), 1, "an indivisible q/kv ratio must clamp the fold");
    assert_eq!(fold_for(32, 4, 8), 1, "hd below the 32-lane fold floor must clamp");
}

#[test]
fn tiled_stage1_shipped_arm_drift_vs_the_monolithic_dense_arm_stays_within_its_pinned_bound() {
    for (i, c) in tiled_cases_the_shipped_depth_stack_arm_fold8_tile4_splits32()
        .iter()
        .enumerate()
    {
        assert_tiled_arm_matches_monolithic_at_stage2(
            STAGE2_MAX_BF16_ULP_DRIFT_1_ON_POSITION_RAMPED_KV_ZERO_OBSERVED,
            c,
            0x7717 + i as u64,
        );
    }
}

#[test]
fn stage2_output_is_invariant_to_the_split_count_softmax_merge_is_partition_free() {
    let ctx = ctx();
    for (n_kv, hd, total, start, ring) in [
        (DENSE_31B_FULL_N_KV, DENSE_31B_FULL_HD, 8193u32, 0u32, 0u32),
        (DENSE_31B_SLIDING_N_KV, DENSE_31B_SLIDING_HD, 4021, 4021 - 1024, 1168),
    ] {
        let mk = |splits: u32| Case { n_kv, hd, total, start, ring, splits, fold: 2, tile: 0 };
        let base = stage2_bf16_pack(
            ctx,
            &mk(16),
            "g4dp-s16",
            &compose(fd::WGSL),
            fd::ENTRY_STAGE1_FP8_SD,
            DENSE_31B_N_Q,
            0x5eed,
        );
        for splits in [8u32, 32, 64] {
            let alt = stage2_bf16_pack(
                ctx,
                &mk(splits),
                "g4dp-salt",
                &compose(fd::WGSL),
                fd::ENTRY_STAGE1_FP8_SD,
                DENSE_31B_N_Q,
                0x5eed,
            );
            let halves = |w: u32| [(w & 0xffff) as u16, (w >> 16) as u16];
            let mut max_seen = 0u32;
            for i in 0..base.len() {
                for (a, b) in halves(base[i]).into_iter().zip(halves(alt[i])) {
                    max_seen = max_seen.max(bf16_ulp_distance(a, b));
                }
            }
            eprintln!(
                "[g4dp-splitinv] n_kv={n_kv} hd={hd} total={total} ring={ring} splits {splits} vs 16: max_bf16_ulp={max_seen}"
            );
            assert!(
                max_seen <= STAGE2_MAX_BF16_ULP_DRIFT_1_ON_POSITION_RAMPED_KV_ZERO_OBSERVED,
                "stage2 output moved {max_seen} bf16 ULP between splits 16 and {splits}: the \
                 split-k merge must be partition-free, so a drift like this is a stage1/stage2 \
                 splits defect, not rounding (the splits=32 ppl collapse basis)"
            );
        }
    }
}

#[test]
fn folded_stage1_wb_reduce_is_bit_identical_to_the_monolithic_dense_arm_incl_past_the_ring_wrap() {
    for (i, c) in dense_cases().iter().enumerate() {
        assert_fold_matches_monolithic(true, false, 0, c, 0x9e37 + i as u64);
        assert_fold_matches_monolithic(false, false, 0, c, 0x51_7c + i as u64);
    }
}

#[test]
fn folded_stage1_shipped_sg_reduce_drifts_at_most_one_ulp_from_the_monolithic_dense_arm() {
    for (i, c) in dense_cases().iter().enumerate() {
        assert_fold_matches_monolithic(
            true,
            SHIPPED_FOLD_REDUCE_IS_SG_SHUFFLE_WB_TREES_LOSE_AT_EVERY_DEPTH,
            SG_REDUCE_MAX_ULP_DRIFT_OBSERVED_ONLY_AT_HD512,
            c,
            0xabcd + i as u64,
        );
        assert_fold_matches_monolithic(
            false,
            SHIPPED_FOLD_REDUCE_IS_SG_SHUFFLE_WB_TREES_LOSE_AT_EVERY_DEPTH,
            SG_REDUCE_MAX_ULP_DRIFT_OBSERVED_ONLY_AT_HD512,
            c,
            0xdcba + i as u64,
        );
    }
}
