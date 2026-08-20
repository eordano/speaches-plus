#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::kernels::gemv_nvfp4_v2::{
    select, select_pk, select_pk_slots, select_slots, V2Kernel, FDEC_PK_ENTRY, FMLUT_PK_ENTRY,
    MROW_PK_ENTRY, WARP_PK_ENTRY, MROW_BEATS_FMLUT_ON_SINGLE_SLOT_DENSE_SHAPES_ON_THIS_LADDER,
};

const NVFP4_BLOCK: usize = 16;

struct Shape {
    label: &'static str,
    n: usize,
    k: usize,
    want: &'static str,
    want_kernel: V2Kernel,
}

const DENSE_9B: &[Shape] = &[
    Shape {
        label: "mlp.gate_proj",
        n: 12288,
        k: 4096,
        want: MROW_PK_ENTRY,
        want_kernel: V2Kernel::MRow,
    },
    Shape {
        label: "mlp.up_proj",
        n: 12288,
        k: 4096,
        want: MROW_PK_ENTRY,
        want_kernel: V2Kernel::MRow,
    },
    Shape {
        label: "mlp.down_proj",
        n: 4096,
        k: 12288,
        want: MROW_PK_ENTRY,
        want_kernel: V2Kernel::MRow,
    },
    Shape {
        label: "self_attn.q_proj (output-gated, 2x wide)",
        n: 8192,
        k: 4096,
        want: MROW_PK_ENTRY,
        want_kernel: V2Kernel::MRow,
    },
    Shape {
        label: "self_attn.k_proj",
        n: 1024,
        k: 4096,
        want: FDEC_PK_ENTRY,
        want_kernel: V2Kernel::FDec,
    },
    Shape {
        label: "self_attn.v_proj",
        n: 1024,
        k: 4096,
        want: FDEC_PK_ENTRY,
        want_kernel: V2Kernel::FDec,
    },
    Shape {
        label: "self_attn.o_proj",
        n: 4096,
        k: 4096,
        want: MROW_PK_ENTRY,
        want_kernel: V2Kernel::MRow,
    },
];

#[test]
fn every_nvfp4_matrix_in_qwen35_9b_takes_a_pair_packed_v2_route() {
    for s in DENSE_9B {
        let (kernel, cfg) = select(s.n, s.k);
        let (pk_kernel, pk_cfg, entry) = select_pk(s.n, s.k)
            .unwrap_or_else(|| panic!("{}: n={} k={} fell off the v2 path", s.label, s.n, s.k));
        eprintln!(
            "[route] {:<42} n={:<6} k={:<6} -> {entry} wg={} mr={} rows/group={}",
            s.label,
            s.n,
            s.k,
            pk_cfg.wg,
            pk_cfg.mr,
            pk_cfg.rows_per_group(pk_kernel)
        );
        assert_eq!(
            kernel,
            s.want_kernel,
            "{}: {}",
            s.label,
            MROW_BEATS_FMLUT_ON_SINGLE_SLOT_DENSE_SHAPES_ON_THIS_LADDER
        );
        assert_eq!(pk_kernel, s.want_kernel, "{}", s.label);
        assert_eq!(cfg, pk_cfg, "{}", s.label);
        assert_eq!(entry, s.want, "{}", s.label);
        assert!(kernel.shape_ok(s.k), "{}", s.label);
        assert!(
            pk_cfg.rows_per_group(pk_kernel).is_multiple_of(2),
            "{}",
            s.label
        );
    }
}

#[test]
fn the_dense_shapes_clear_the_extra_gates_the_model_route_applies() {
    for s in DENSE_9B {
        let k_blocks = s.k / NVFP4_BLOCK;
        assert!(
            k_blocks.is_multiple_of(4),
            "{}: k_blocks {k_blocks} must be a multiple of 4",
            s.label
        );
        assert!(
            (s.n * k_blocks).is_multiple_of(2),
            "{}: n*k_blocks must be even",
            s.label
        );
        assert!(s.n >= 2, "{}", s.label);
        assert!(
            s.k.is_multiple_of(NVFP4_BLOCK),
            "{}: k must be block aligned",
            s.label
        );
    }
}

#[test]
fn the_9b_shapes_are_distinct_from_the_already_pinned_qwen36_and_gemma4_shapes() {
    let pinned: &[(usize, usize)] = &[
        (43008, 5376),
        (5376, 21504),
        (8192, 2048),
        (2048, 4096),
        (512, 2048),
        (2048, 512),
    ];
    for s in DENSE_9B {
        assert!(
            !pinned.contains(&(s.n, s.k)),
            "{} ({}, {}) is already covered upstream; drop it from this suite",
            s.label,
            s.n,
            s.k
        );
    }
}

#[test]
fn the_narrow_kv_projection_is_the_only_shape_below_the_mrow_row_threshold() {
    let mut narrow = 0usize;
    for s in DENSE_9B {
        let (kernel, _) = select(s.n, s.k);
        if s.n < 2048 {
            narrow += 1;
            assert_eq!(kernel, V2Kernel::FDec, "{}", s.label);
        } else {
            assert_eq!(
                kernel,
                V2Kernel::MRow,
                "{}: {}",
                s.label,
                MROW_BEATS_FMLUT_ON_SINGLE_SLOT_DENSE_SHAPES_ON_THIS_LADDER
            );
        }
    }
    assert_eq!(narrow, 2, "k_proj and v_proj are the two narrow matrices");
}

#[test]
fn the_moe_expert_stacks_need_the_slot_count_to_route_correctly() {
    const SLOTS: usize = 9;
    let cases: &[(&str, usize, usize, &str, &str)] = &[
        (
            "moe experts gate/up, fused stack",
            1024,
            2048,
            FDEC_PK_ENTRY,
            FMLUT_PK_ENTRY,
        ),
        (
            "moe experts gate or up, unfused half",
            512,
            2048,
            FDEC_PK_ENTRY,
            FMLUT_PK_ENTRY,
        ),
        ("moe experts down", 2048, 512, WARP_PK_ENTRY, WARP_PK_ENTRY),
    ];
    for (label, n, k, blind, routed) in cases {
        let got_blind = select_pk(*n, *k).expect("pk route").2;
        let got_routed = select_pk_slots(*n, *k, SLOTS).expect("pk route").2;
        eprintln!(
            "[moe] {label:<38} n={n:<5} k={k:<5} slots={SLOTS} |              slot-blind {got_blind} -> slot-aware {got_routed}"
        );
        assert_eq!(got_blind, *blind, "{label}: slot-blind route moved");
        assert_eq!(got_routed, *routed, "{label}: slot-aware route");
    }

    assert_eq!(
        select_slots(512, 2048, SLOTS),
        select_slots(1024, 2048, SLOTS),
        "the gate/up fusion predicate compares route_of(inter) with route_of(2*inter)"
    );

    for s in DENSE_9B {
        assert_eq!(select_slots(s.n, s.k, 1), select(s.n, s.k), "{}", s.label);
    }
    for (n, k) in [
        (43008usize, 5376usize),
        (5376, 21504),
        (8192, 2048),
        (2048, 4096),
        (512, 2048),
        (2048, 512),
    ] {
        assert_eq!(select_pk_slots(n, k, 1), select_pk(n, k), "n={n} k={k}");
    }
}
