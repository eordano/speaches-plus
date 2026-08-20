use half::bf16;

include!("support/w4a16_host_oracle.rs");

const FUZZ_SHAPES: &[(usize, usize)] = &[
    (1, 16),
    (1, 8192),
    (3, 48),
    (5, 96),
    (16, 128),
    (7, 4096),
    (2, 2048),
    (11, 176),
];

const GROUP_SIZES: &[usize] = &[16, 32, 64, 128];

#[test]
fn independent_decompositions_agree_over_the_fuzzed_shape_grid() {
    let mut cases = 0usize;
    for &(n, k) in FUZZ_SHAPES {
        for &gs in GROUP_SIZES {
            if k % gs != 0 || k % NIBBLES_PER_WORD != 0 {
                continue;
            }
            for seed in [1u64, 0x9e3779b9, 0xdeadbeef] {
                let (packed, scales, x) = gen_inputs(n, k, gs, seed);
                let a = ref_row_major(&packed, &scales, &x, n, k, gs);
                let b = ref_group_major_f64(&packed, &scales, &x, n, k, gs, PlantedBug::None);
                let d = max_rel_diff(&a, &b);
                assert!(
                    d < CROSS_DECOMPOSITION_REL_TOL_BOUNDS_F32_VS_F64_ACCUMULATION,
                    "row-major f32-product oracle and group-major f64 oracle diverged \
                     (n={n} k={k} gs={gs} seed={seed:#x}: rel {d:.3e}); one of the two \
                     decompositions no longer implements the documented dequant rule"
                );
                cases += 1;
            }
        }
    }
    assert!(cases >= 60, "shape grid shrank to {cases} cases; the fuzz lost coverage");
}

#[test]
fn every_planted_bug_is_caught_at_every_group_size() {
    for &bug in &[
        PlantedBug::ScaleIndexOffByOneGroup,
        PlantedBug::NibbleOrderReversed,
        PlantedBug::MissingSignOffset,
    ] {
        for &gs in GROUP_SIZES {
            let (n, k) = (5usize, 512usize);
            if k % gs != 0 {
                continue;
            }
            let mut caught = false;
            for seed in [1u64, 2, 3] {
                let (packed, scales, x) = gen_inputs(n, k, gs, seed);
                let good = ref_group_major_f64(&packed, &scales, &x, n, k, gs, PlantedBug::None);
                let bad = ref_group_major_f64(&packed, &scales, &x, n, k, gs, bug);
                if max_rel_diff(&good, &bad)
                    > 10.0 * CROSS_DECOMPOSITION_REL_TOL_BOUNDS_F32_VS_F64_ACCUMULATION
                {
                    caught = true;
                }
            }
            assert!(
                caught,
                "planted bug survived the gate at gs={gs}: a gate that cannot catch its own \
                 seeded mutations vouches for nothing (05.2 planted-bug protocol)"
            );
        }
    }
}

#[test]
fn scale_off_by_one_is_invisible_with_a_single_group_the_incident_shape() {
    let (n, k, gs) = (4usize, 128usize, 128usize);
    let (packed, scales, x) = gen_inputs(n, k, gs, 7);
    let good = ref_group_major_f64(&packed, &scales, &x, n, k, gs, PlantedBug::None);
    let bad = ref_group_major_f64(
        &packed,
        &scales,
        &x,
        n,
        k,
        gs,
        PlantedBug::ScaleIndexOffByOneGroup,
    );
    assert!(
        max_rel_diff(&good, &bad) < 1e-12,
        "with one group per row the off-by-one wraps to itself; this test pins WHY \
         single-group shapes can never gate scale indexing and multi-group shapes are \
         mandatory in every w4a16 suite (the recorded group-16 incident class)"
    );
}
