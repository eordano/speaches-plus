#![cfg(feature = "wgpu")]

mod common;
use common::pipeline;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;

pub const A_COOPERATIVE_MATRIX_ACCESS_TO_A_WORKGROUP_ARRAY_TRUNCATES_ITS_ADDRESS_TO_16_BYTES_ON_NAGA_30_0_0_WITH_NVIDIA_590_48_01: &str =
    "On naga 30.0.0 / wgpu 30.0.0 / NVIDIA 590.48.01 `coopLoadT` and `coopStoreT` against a \
     `var<workgroup>` array address the 16-byte-aligned word at or below the array, not the array. \
     The workgroup block places a variable at the total size of everything declared after it, so a \
     tile is correctly addressed exactly when that total is a multiple of 16 bytes, and otherwise \
     the access lands `total % 16` bytes low: a trailing `u32`, `atomic<u32>`, `array<u32, 1>` or \
     `array<f32, 1>` costs 4 bytes and reads two f16 early, `array<f16, 1>` costs 2 and reads one \
     early, `array<f32, 3>` costs 12 and reads six early, `array<f32, 5>` costs 20 and reads two \
     early because only the remainder matters, and any multiple of 16 costs nothing in either \
     declaration order. A cooperative store lands the same distance low. Nothing traps. The shader \
     compiles clean, no validation error fires, the tile stays finite and right-magnitude, and the \
     kernel returns a plausible answer that only a reference knowing the expected values can \
     reject. If this suite starts failing after a naga, wgpu or driver bump, the compiler is what \
     changed: re-measure before removing any declaration-order workaround, and update the versions \
     named here.";

pub const A_PLAIN_INDEXED_READ_OF_A_WORKGROUP_F16_ARRAY_IS_ADDRESSED_CORRECTLY_SO_ONLY_COOPERATIVE_MATRIX_KERNELS_ARE_EXPOSED: &str =
    "Ordinary WGSL indexing of a workgroup f16 array beside a scalar, an atomic, or both reads back \
     every element it wrote on this toolchain, and so does every 4-byte-element workgroup array. \
     The defect is confined to the cooperative-matrix pointer lowering, which is why a kernel that \
     pairs a scalar workgroup variable with an f32 or u32 workgroup array and never touches a \
     cooperative matrix needs no change. These arms are that exoneration: delete them and it is \
     gone with them.";

pub const REPLACING_A_WORKGROUP_SCALAR_WITH_A_ONE_ELEMENT_ARRAY_DOES_NOT_FIX_THE_COOPERATIVE_MATRIX_SHIFT: &str =
    "An `array<u32, 1>` occupies the same four bytes as the `u32` it replaced, so it shifts a \
     cooperative-matrix tile declared above it by the same two f16 and fixes nothing. What removes \
     the shift is making the total size of everything declared after the tile a multiple of 16 \
     bytes -- an `array<u32, 4>` in place of a scalar, or moving the small variables above the \
     tile, or having none. A kernel that reads correct after such a swap did so because the \
     replacement happened to be 16-byte sized, and swapping in a shorter one silently breaks it \
     again.";

const COOPERATIVE_MATRIX_WORKGROUP_ACCESSES_TRUNCATE_TO_THIS_BYTE_BOUNDARY: usize = 16;

pub const A_NON_CONSTANT_STRIDE_ARGUMENT_MOVES_THE_COOPERATIVE_MATRIX_LOAD_ONTO_A_CORRECTLY_ADDRESSED_PATH: &str =
    "The same load that reads the wrong base with a literal stride reads the right one when the \
     stride comes from a uniform, so a kernel whose stride is a runtime value is not evidence that \
     its neighbours are safe. Stride provenance is not a fix to reach for; it is a hazard when \
     comparing two kernels that look alike.";

pub const THE_PROBE_MUST_PANIC_RATHER_THAN_SKIP_BECAUSE_A_MISSING_ADAPTER_PRINTS_OK: &str =
    "A wgpu suite with no Vulkan loader reports success without running a single shader, so this \
     suite refuses to be skipped by any environment variable.";

const N: usize = 256;
const WG: u32 = 64;
const FILL: u32 = 3;
const GAIN: u32 = 16;
const EXPECT: u32 = FILL * GAIN;
const SCALAR_SLOT: usize = N;
const ATOMIC_SLOT: usize = N + 1;
const OUT_LEN: usize = N + 2;
const SCALAR_MARK: u32 = 0xa5a5_00ffu32;
const ATOMIC_LANES: u32 = WG;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Elem {
    F16,
    F32,
    U32,
}

impl Elem {
    fn ty(self) -> &'static str {
        match self {
            Elem::F16 => "f16",
            Elem::F32 => "f32",
            Elem::U32 => "u32",
        }
    }
    fn store(self) -> String {
        match self {
            Elem::F16 => format!("f16({FILL}.0)"),
            Elem::F32 => format!("{FILL}.0"),
            Elem::U32 => format!("{FILL}u"),
        }
    }
    fn load(self, ix: &str) -> String {
        match self {
            Elem::F16 => format!("u32(f32(pb_tile[{ix}]) * {GAIN}.0)"),
            Elem::F32 => format!("u32(pb_tile[{ix}] * {GAIN}.0)"),
            Elem::U32 => format!("pb_tile[{ix}] * {GAIN}u"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scalars {
    None,
    PlainAfter,
    PlainBefore,
    AtomicAndPlainAfter,
    OneElementArraysAfter,
}

#[derive(Clone, Copy, Debug)]
struct Arm {
    name: &'static str,
    elem: Elem,
    scalars: Scalars,
    companion_f32_tile: bool,
}

const ARMS: &[Arm] = &[
    Arm {
        name: "f16_tile_alone_control",
        elem: Elem::F16,
        scalars: Scalars::None,
        companion_f32_tile: false,
    },
    Arm {
        name: "f16_tile_plus_scalar_after",
        elem: Elem::F16,
        scalars: Scalars::PlainAfter,
        companion_f32_tile: false,
    },
    Arm {
        name: "f16_tile_plus_scalar_before",
        elem: Elem::F16,
        scalars: Scalars::PlainBefore,
        companion_f32_tile: false,
    },
    Arm {
        name: "f16_tile_plus_atomic_and_scalar",
        elem: Elem::F16,
        scalars: Scalars::AtomicAndPlainAfter,
        companion_f32_tile: false,
    },
    Arm {
        name: "f32_score_tile_then_f16_tile_plus_atomic_and_scalar",
        elem: Elem::F16,
        scalars: Scalars::AtomicAndPlainAfter,
        companion_f32_tile: true,
    },
    Arm {
        name: "f16_tile_plus_one_element_arrays",
        elem: Elem::F16,
        scalars: Scalars::OneElementArraysAfter,
        companion_f32_tile: false,
    },
    Arm {
        name: "f32_tile_plus_atomic_and_scalar",
        elem: Elem::F32,
        scalars: Scalars::AtomicAndPlainAfter,
        companion_f32_tile: false,
    },
    Arm {
        name: "f32_tile_plus_scalar_after",
        elem: Elem::F32,
        scalars: Scalars::PlainAfter,
        companion_f32_tile: false,
    },
    Arm {
        name: "u32_tile_plus_atomic_and_scalar",
        elem: Elem::U32,
        scalars: Scalars::AtomicAndPlainAfter,
        companion_f32_tile: false,
    },
];

fn decl_scalars(b: &mut String, scalars: Scalars) {
    match scalars {
        Scalars::None => {}
        Scalars::PlainAfter | Scalars::PlainBefore => {
            b.push_str("var<workgroup> pb_flag: u32;\n");
        }
        Scalars::AtomicAndPlainAfter => {
            b.push_str("var<workgroup> pb_cnt: atomic<u32>;\n");
            b.push_str("var<workgroup> pb_flag: u32;\n");
        }
        Scalars::OneElementArraysAfter => {
            b.push_str("var<workgroup> pb_cnt: array<atomic<u32>, 1>;\n");
            b.push_str("var<workgroup> pb_flag: array<u32, 1>;\n");
        }
    }
}

fn write_scalars(b: &mut String, scalars: Scalars, wg: u32) {
    use std::fmt::Write as _;
    let _ = wg;
    match scalars {
        Scalars::None => {}
        Scalars::PlainAfter | Scalars::PlainBefore => {
            b.push_str("    if (lidx == 0u) { pb_flag = 0u; }\n    workgroupBarrier();\n");
            writeln!(b, "    if (lidx == 0u) {{ pb_flag = {SCALAR_MARK}u; }}").unwrap();
            b.push_str("    workgroupBarrier();\n");
        }
        Scalars::AtomicAndPlainAfter => {
            b.push_str("    if (lidx == 0u) { atomicStore(&pb_cnt, 0u); pb_flag = 0u; }\n");
            b.push_str("    workgroupBarrier();\n    atomicAdd(&pb_cnt, 1u);\n");
            writeln!(b, "    if (lidx == 0u) {{ pb_flag = {SCALAR_MARK}u; }}").unwrap();
            b.push_str("    workgroupBarrier();\n");
        }
        Scalars::OneElementArraysAfter => {
            b.push_str("    if (lidx == 0u) { atomicStore(&pb_cnt[0], 0u); pb_flag[0] = 0u; }\n");
            b.push_str("    workgroupBarrier();\n    atomicAdd(&pb_cnt[0], 1u);\n");
            writeln!(b, "    if (lidx == 0u) {{ pb_flag[0] = {SCALAR_MARK}u; }}").unwrap();
            b.push_str("    workgroupBarrier();\n");
        }
    }
}

fn read_scalars(scalars: Scalars, flag_slot: usize, atomic_slot: usize, atomic_when_absent: u32) -> String {
    match scalars {
        Scalars::None => format!(
            "        pb_out[{flag_slot}u] = {SCALAR_MARK}u;\n        pb_out[{atomic_slot}u] = {atomic_when_absent}u;\n"
        ),
        Scalars::PlainAfter | Scalars::PlainBefore => format!(
            "        pb_out[{flag_slot}u] = pb_flag;\n        pb_out[{atomic_slot}u] = {atomic_when_absent}u;\n"
        ),
        Scalars::AtomicAndPlainAfter => format!(
            "        pb_out[{flag_slot}u] = pb_flag;\n        pb_out[{atomic_slot}u] = atomicLoad(&pb_cnt);\n"
        ),
        Scalars::OneElementArraysAfter => format!(
            "        pb_out[{flag_slot}u] = pb_flag[0];\n        pb_out[{atomic_slot}u] = atomicLoad(&pb_cnt[0]);\n"
        ),
    }
}

fn source(arm: &Arm) -> String {
    use std::fmt::Write as _;
    let mut b = String::new();
    b.push_str("enable f16;\n");
    b.push_str("@group(0) @binding(0) var<storage, read_write> pb_out: array<u32>;\n");
    if arm.scalars == Scalars::PlainBefore {
        decl_scalars(&mut b, arm.scalars);
    }
    if arm.companion_f32_tile {
        writeln!(b, "var<workgroup> pb_scores: array<f32, {N}>;").unwrap();
    }
    writeln!(b, "var<workgroup> pb_tile: array<{}, {N}>;", arm.elem.ty()).unwrap();
    if arm.scalars != Scalars::PlainBefore {
        decl_scalars(&mut b, arm.scalars);
    }
    writeln!(b, "\n@compute @workgroup_size({WG})").unwrap();
    b.push_str("fn pb_alias(@builtin(local_invocation_index) lidx: u32) {\n");
    if arm.companion_f32_tile {
        writeln!(
            b,
            "    for (var i = lidx; i < {N}u; i = i + {WG}u) {{ pb_scores[i] = f32(i); }}"
        )
        .unwrap();
    }
    writeln!(
        b,
        "    for (var i = lidx; i < {N}u; i = i + {WG}u) {{ pb_tile[i] = {}; }}",
        arm.elem.store()
    )
    .unwrap();
    b.push_str("    workgroupBarrier();\n");
    write_scalars(&mut b, arm.scalars, WG);
    writeln!(
        b,
        "    for (var i = lidx; i < {N}u; i = i + {WG}u) {{ pb_out[i] = {}; }}",
        arm.elem.load("i")
    )
    .unwrap();
    b.push_str("    if (lidx == 0u) {\n");
    b.push_str(&read_scalars(
        arm.scalars,
        SCALAR_SLOT,
        ATOMIC_SLOT,
        ATOMIC_LANES,
    ));
    b.push_str("    }\n}\n");
    b
}

fn ctx_or_panic(what: &str) -> &'static WgpuContext {
    let ctx = WgpuContext::shared().unwrap_or_else(|e| {
        panic!("{what}: no wgpu adapter: {e}. {THE_PROBE_MUST_PANIC_RATHER_THAN_SKIP_BECAUSE_A_MISSING_ADAPTER_PRINTS_OK}")
    });
    eprintln!("{what}: {}", ctx.summary());
    ctx
}

fn run_arm(ctx: &WgpuContext, arm: &Arm, zero_init: bool) -> Vec<u32> {
    let src = source(arm);
    let label = format!("{}_zi{}", arm.name, u8::from(zero_init));
    let pl = pipeline(ctx, &label, &src, "pb_alias", zero_init);
    let out = dispatch::storage_zeroed(ctx, &label, (OUT_LEN * 4) as u64);
    dispatch::dispatch(ctx, &pl, &[(0, &out)], (1, 1, 1))
        .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
    dispatch::read_back::<u32>(ctx, &out, OUT_LEN)
        .unwrap_or_else(|e| panic!("{label}: readback failed: {e}"))
}

fn bad_elems(v: &[u32]) -> Vec<usize> {
    (0..N).filter(|i| v[*i] != EXPECT).collect()
}

fn report(arm: &Arm, zero_init: bool, v: &[u32]) -> String {
    let bad = bad_elems(v);
    format!(
        "{:52} zi={} bad={:3}/{N} flag={:#010x} atomic={}",
        arm.name,
        u8::from(zero_init),
        bad.len(),
        v[SCALAR_SLOT],
        v[ATOMIC_SLOT]
    )
}

fn arm(name: &str) -> &'static Arm {
    ARMS.iter()
        .find(|a| a.name == name)
        .unwrap_or_else(|| panic!("no arm named {name}"))
}

const COOP_T: usize = 16;
const COOP_ELEMS: usize = COOP_T * COOP_T;
const COOP_WG: u32 = 32;
const COOP_SINK: usize = COOP_ELEMS;
const COOP_OUT_LEN: usize = COOP_ELEMS + 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Wg {
    Tile,
    Stage,
    ArrF16(usize),
    ArrF32(usize),
    ArrU32(usize),
    ScalarU32,
    ScalarAtomic,
    OneElemU32,
    OneElemAtomic,
}

impl Wg {
    fn bytes(self) -> usize {
        match self {
            Wg::Tile => COOP_ELEMS * 2,
            Wg::Stage => COOP_ELEMS * 4,
            Wg::ArrF16(n) => n * 2,
            Wg::ArrF32(n) | Wg::ArrU32(n) => n * 4,
            Wg::ScalarU32 | Wg::ScalarAtomic | Wg::OneElemU32 | Wg::OneElemAtomic => 4,
        }
    }
    fn decl(self, name: &str) -> String {
        match self {
            Wg::Tile => format!("var<workgroup> {name}: array<f16, {COOP_ELEMS}>;\n"),
            Wg::Stage => format!("var<workgroup> {name}: array<f32, {COOP_ELEMS}>;\n"),
            Wg::ArrF16(n) => format!("var<workgroup> {name}: array<f16, {n}>;\n"),
            Wg::ArrF32(n) => format!("var<workgroup> {name}: array<f32, {n}>;\n"),
            Wg::ArrU32(n) => format!("var<workgroup> {name}: array<u32, {n}>;\n"),
            Wg::ScalarU32 => format!("var<workgroup> {name}: u32;\n"),
            Wg::ScalarAtomic => format!("var<workgroup> {name}: atomic<u32>;\n"),
            Wg::OneElemU32 => format!("var<workgroup> {name}: array<u32, 1>;\n"),
            Wg::OneElemAtomic => format!("var<workgroup> {name}: array<atomic<u32>, 1>;\n"),
        }
    }
    fn fill(self, name: &str) -> String {
        match self {
            Wg::Tile => format!(
                "    for (var i = lidx; i < {COOP_ELEMS}u; i = i + {COOP_WG}u) {{ {name}[i] = f16(f32(i) + 1.0); }}\n"
            ),
            Wg::Stage => format!(
                "    for (var i = lidx; i < {COOP_ELEMS}u; i = i + {COOP_WG}u) {{ {name}[i] = -1.0; }}\n"
            ),
            Wg::ArrF16(n) => format!(
                "    for (var i = lidx; i < {n}u; i = i + {COOP_WG}u) {{ {name}[i] = f16(-1.0); }}\n"
            ),
            Wg::ArrF32(n) => format!(
                "    for (var i = lidx; i < {n}u; i = i + {COOP_WG}u) {{ {name}[i] = -1.0; }}\n"
            ),
            Wg::ArrU32(n) => format!(
                "    for (var i = lidx; i < {n}u; i = i + {COOP_WG}u) {{ {name}[i] = {SCALAR_MARK}u; }}\n"
            ),
            Wg::ScalarU32 => format!("    if (lidx == 0u) {{ {name} = {SCALAR_MARK}u; }}\n"),
            Wg::OneElemU32 => format!("    if (lidx == 0u) {{ {name}[0] = {SCALAR_MARK}u; }}\n"),
            Wg::ScalarAtomic => {
                format!("    if (lidx == 0u) {{ atomicStore(&{name}, 0u); }}\n    workgroupBarrier();\n    atomicAdd(&{name}, 1u);\n")
            }
            Wg::OneElemAtomic => {
                format!("    if (lidx == 0u) {{ atomicStore(&{name}[0], 0u); }}\n    workgroupBarrier();\n    atomicAdd(&{name}[0], 1u);\n")
            }
        }
    }
    fn sink(self, name: &str) -> String {
        match self {
            Wg::Tile | Wg::Stage => String::new(),
            Wg::ArrF16(_) => format!(" + f32({name}[0])"),
            Wg::ArrF32(_) => format!(" + {name}[0]"),
            Wg::ArrU32(_) => format!(" + f32({name}[0])"),
            Wg::ScalarU32 => format!(" + f32({name})"),
            Wg::OneElemU32 => format!(" + f32({name}[0])"),
            Wg::ScalarAtomic => format!(" + f32(atomicLoad(&{name}))"),
            Wg::OneElemAtomic => format!(" + f32(atomicLoad(&{name}[0]))"),
        }
    }
}

#[derive(Clone, Copy)]
struct Layout {
    name: &'static str,
    vars: &'static [Wg],
    dyn_stride: bool,
}

const TILE_MUST_BE_PRESENT_EXACTLY_ONCE_IN_EVERY_LAYOUT: &str =
    "every coop layout arm loads its A operand from the one Wg::Tile it declares";

const LAYOUTS: &[Layout] = &[
    Layout {
        name: "tile_is_the_only_workgroup_variable",
        vars: &[Wg::Tile],
        dyn_stride: false,
    },
    Layout {
        name: "tile_then_u32_scalar",
        vars: &[Wg::Tile, Wg::ScalarU32],
        dyn_stride: false,
    },
    Layout {
        name: "u32_scalar_then_tile",
        vars: &[Wg::ScalarU32, Wg::Tile],
        dyn_stride: false,
    },
    Layout {
        name: "tile_then_atomic_and_u32_scalar",
        vars: &[Wg::Tile, Wg::ScalarAtomic, Wg::ScalarU32],
        dyn_stride: false,
    },
    Layout {
        name: "tile_then_two_one_element_arrays",
        vars: &[Wg::Tile, Wg::OneElemAtomic, Wg::OneElemU32],
        dyn_stride: false,
    },
    Layout {
        name: "tile_then_f32_array",
        vars: &[Wg::Tile, Wg::ArrF32(64)],
        dyn_stride: false,
    },
    Layout {
        name: "f32_array_then_tile",
        vars: &[Wg::ArrF32(64), Wg::Tile],
        dyn_stride: false,
    },
    Layout {
        name: "f16_array_then_tile",
        vars: &[Wg::ArrF16(64), Wg::Tile],
        dyn_stride: false,
    },
    Layout {
        name: "shipping_coop_gemm_order_tile_first",
        vars: &[
            Wg::Tile,
            Wg::ArrF16(COOP_ELEMS),
            Wg::ArrF32(COOP_ELEMS),
            Wg::ArrF32(COOP_ELEMS),
        ],
        dyn_stride: false,
    },
    Layout {
        name: "shipping_coop_gemm_order_tile_second",
        vars: &[
            Wg::ArrF16(COOP_ELEMS),
            Wg::Tile,
            Wg::ArrF32(COOP_ELEMS),
            Wg::ArrF32(COOP_ELEMS),
        ],
        dyn_stride: false,
    },
    Layout {
        name: "tile_then_f32_array_of_three",
        vars: &[Wg::Tile, Wg::ArrF32(3)],
        dyn_stride: false,
    },
    Layout {
        name: "tile_then_two_f32_arrays_of_three",
        vars: &[Wg::Tile, Wg::ArrF32(3), Wg::ArrF32(3)],
        dyn_stride: false,
    },
    Layout {
        name: "tile_then_f32_array_of_three_and_f32_array_of_one",
        vars: &[Wg::Tile, Wg::ArrF32(3), Wg::ArrF32(1)],
        dyn_stride: false,
    },
    Layout {
        name: "tile_then_f32_array_of_five",
        vars: &[Wg::Tile, Wg::ArrF32(5)],
        dyn_stride: false,
    },
    Layout {
        name: "tile_then_f32_array_of_four",
        vars: &[Wg::Tile, Wg::ArrF32(4)],
        dyn_stride: false,
    },
    Layout {
        name: "tile_then_f32_array_of_eight",
        vars: &[Wg::Tile, Wg::ArrF32(8)],
        dyn_stride: false,
    },
    Layout {
        name: "tile_then_f32_array_of_sixteen",
        vars: &[Wg::Tile, Wg::ArrF32(16)],
        dyn_stride: false,
    },
    Layout {
        name: "coop_store_stage_then_tile_no_small_variable",
        vars: &[Wg::Stage, Wg::Tile],
        dyn_stride: false,
    },
    Layout {
        name: "coop_store_stage_then_u32_scalar_then_tile",
        vars: &[Wg::Stage, Wg::ScalarU32, Wg::Tile],
        dyn_stride: false,
    },
    Layout {
        name: "tile_then_u32_array_of_two",
        vars: &[Wg::Tile, Wg::ArrU32(2)],
        dyn_stride: false,
    },
    Layout {
        name: "tile_then_f32_array_of_one",
        vars: &[Wg::Tile, Wg::ArrF32(1)],
        dyn_stride: false,
    },
    Layout {
        name: "tile_then_f16_array_of_one",
        vars: &[Wg::Tile, Wg::ArrF16(1)],
        dyn_stride: false,
    },
    Layout {
        name: "f32_array_then_tile_then_u32_scalar",
        vars: &[Wg::ArrF32(64), Wg::Tile, Wg::ScalarU32],
        dyn_stride: false,
    },
    Layout {
        name: "tile_then_u32_scalar_then_f32_array",
        vars: &[Wg::Tile, Wg::ScalarU32, Wg::ArrF32(64)],
        dyn_stride: false,
    },
    Layout {
        name: "gemm_nvfp4_and_moe_grouped_gemm_declarations_tile_at_coop_b",
        vars: &[
            Wg::ArrF16(512),
            Wg::Tile,
            Wg::ArrF16(256),
            Wg::ArrF32(1024),
            Wg::ArrF32(256),
        ],
        dyn_stride: false,
    },
    Layout {
        name: "q3d_pf_flash_tiled_coop_declarations_tile_at_fdc_psl",
        vars: &[
            Wg::ArrF16(4096),
            Wg::ArrF16(4096),
            Wg::ArrF32(4096),
            Wg::Tile,
            Wg::ArrF16(256),
            Wg::ArrF32(16),
            Wg::ArrF32(16),
            Wg::ArrF32(256),
        ],
        dyn_stride: false,
    },
    Layout {
        name: "gemm_coop_f16_declarations_tile_at_xas",
        vars: &[Wg::ArrF32(1024), Wg::ArrF16(1024), Wg::Tile],
        dyn_stride: false,
    },
    Layout {
        name: "tile_then_atomic_and_u32_scalar_dynamic_stride",
        vars: &[Wg::Tile, Wg::ScalarAtomic, Wg::ScalarU32],
        dyn_stride: true,
    },
    Layout {
        name: "f32_array_then_tile_dynamic_stride",
        vars: &[Wg::ArrF32(64), Wg::Tile],
        dyn_stride: true,
    },
];

fn var_name(i: usize, v: Wg) -> String {
    if v == Wg::Tile {
        "pb_tile".to_string()
    } else {
        format!("pb_v{i}")
    }
}

fn coop_source(l: &Layout) -> String {
    use std::fmt::Write as _;
    let _ = TILE_MUST_BE_PRESENT_EXACTLY_ONCE_IN_EVERY_LAYOUT;
    let mut b = String::new();
    b.push_str("enable f16;\nenable wgpu_cooperative_matrix;\n");
    writeln!(b, "alias PBA = coop_mat{COOP_T}x{COOP_T}<f16, A>;").unwrap();
    writeln!(b, "alias PBB = coop_mat{COOP_T}x{COOP_T}<f16, B>;").unwrap();
    writeln!(b, "alias PBC = coop_mat{COOP_T}x{COOP_T}<f32, C>;\n").unwrap();
    b.push_str("struct PbP { stride: u32, pad0: u32, pad1: u32, pad2: u32 };\n");
    b.push_str("@group(0) @binding(0) var<storage, read_write> pb_out: array<f32>;\n");
    b.push_str("@group(0) @binding(1) var<storage, read> pb_zero: array<f32>;\n");
    b.push_str("@group(0) @binding(2) var<storage, read> pb_id: array<f16>;\n");
    b.push_str("@group(0) @binding(3) var<uniform> pb_p: PbP;\n\n");
    for (i, v) in l.vars.iter().enumerate() {
        b.push_str(&v.decl(&var_name(i, *v)));
    }
    writeln!(b, "\n@compute @workgroup_size({COOP_WG})").unwrap();
    b.push_str("fn pb_coop(@builtin(local_invocation_index) lidx: u32) {\n");
    for (i, v) in l.vars.iter().enumerate() {
        b.push_str(&v.fill(&var_name(i, *v)));
    }
    b.push_str("    workgroupBarrier();\n");
    let stride = if l.dyn_stride {
        "pb_p.stride".to_string()
    } else {
        format!("{COOP_T}u")
    };
    writeln!(b, "    let a = coopLoadT<PBA>(&pb_tile[0], {stride});").unwrap();
    writeln!(b, "    let bb = coopLoadT<PBB>(&pb_id[0], {COOP_T}u);").unwrap();
    writeln!(b, "    let z = coopLoadT<PBC>(&pb_zero[0], {COOP_T}u);").unwrap();
    b.push_str("    let c = coopMultiplyAdd(a, bb, z);\n");
    let staged = l.vars.iter().position(|v| *v == Wg::Stage);
    match staged {
        None => writeln!(b, "    coopStoreT(c, &pb_out[0], {COOP_T}u);").unwrap(),
        Some(i) => {
            let s = var_name(i, Wg::Stage);
            writeln!(b, "    coopStoreT(c, &{s}[0], {COOP_T}u);").unwrap();
            b.push_str("    workgroupBarrier();\n");
            writeln!(
                b,
                "    for (var i = lidx; i < {COOP_ELEMS}u; i = i + {COOP_WG}u) {{ pb_out[i] = {s}[i]; }}"
            )
            .unwrap();
        }
    }
    b.push_str("    workgroupBarrier();\n    if (lidx == 0u) {\n");
    let mut sink = String::from("0.0");
    for (i, v) in l.vars.iter().enumerate() {
        sink.push_str(&v.sink(&var_name(i, *v)));
    }
    writeln!(b, "        pb_out[{COOP_SINK}u] = {sink};").unwrap();
    b.push_str("    }\n}\n");
    b
}

fn run_coop(ctx: &WgpuContext, l: &Layout) -> Vec<f32> {
    let src = coop_source(l);
    let label = l.name;
    dispatch::coop_matrix_guard(ctx, label, &src, "pb_coop").unwrap_or_else(|e| {
        panic!(
            "{label}: this adapter does not advertise the {COOP_T}x{COOP_T} f16 cooperative-matrix \
             config the probe needs, so the arms that reproduce the defect cannot run: {e}"
        )
    });
    let pl = pipeline(ctx, label, &src, "pb_coop", true);
    let out = dispatch::storage_zeroed(ctx, label, (COOP_OUT_LEN * 4) as u64);
    let zero = dispatch::storage_from_slice(ctx, "pb_zero", &vec![0f32; COOP_ELEMS]);
    let mut ident = vec![0u16; COOP_ELEMS];
    for i in 0..COOP_T {
        ident[i * COOP_T + i] = half::f16::from_f32(1.0).to_bits();
    }
    let idb = dispatch::storage_from_slice(ctx, "pb_id", &ident);
    let up = dispatch::uniform_from(ctx, "pb_p", &[COOP_T as u32, 0u32, 0u32, 0u32]);
    let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![(0, &out), (1, &zero), (2, &idb)];
    if l.dyn_stride {
        binds.push((3, &up));
    }
    dispatch::dispatch(ctx, &pl, &binds, (1, 1, 1))
        .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
    dispatch::read_back::<f32>(ctx, &out, COOP_OUT_LEN)
        .unwrap_or_else(|e| panic!("{label}: readback failed: {e}"))
}

fn coop_bad(v: &[f32]) -> Vec<usize> {
    (0..COOP_ELEMS)
        .filter(|i| v[*i] != (*i as f32) + 1.0)
        .collect()
}

fn coop_shift(v: &[f32]) -> Option<isize> {
    for s in -32isize..=32 {
        let lo = s.max(0) as usize;
        let hi = COOP_ELEMS - (-s).max(0) as usize;
        if lo < hi && (lo..hi).all(|i| v[i] == (i as isize - s) as f32 + 1.0) {
            return Some(s);
        }
    }
    None
}

fn bytes_declared_after(l: &Layout, target: Wg) -> usize {
    let mut seen = false;
    let mut n = 0;
    for v in l.vars {
        if *v == target {
            seen = true;
            continue;
        }
        if seen {
            n += v.bytes();
        }
    }
    n
}

fn misalignment(l: &Layout, target: Wg) -> usize {
    bytes_declared_after(l, target) % COOPERATIVE_MATRIX_WORKGROUP_ACCESSES_TRUNCATE_TO_THIS_BYTE_BOUNDARY
}

fn expected_shift(l: &Layout) -> isize {
    if l.dyn_stride {
        return 0;
    }
    if l.vars.contains(&Wg::Stage) {
        return -((misalignment(l, Wg::Stage) / 4) as isize);
    }
    (misalignment(l, Wg::Tile) / 2) as isize
}

fn coop_report(l: &Layout, v: &[f32]) -> String {
    format!(
        "{:58} dynstride={} after_tile={:5}B misalign={:2}B bad={:3}/{COOP_ELEMS} shift={:?} want={} head={:?}",
        l.name,
        u8::from(l.dyn_stride),
        bytes_declared_after(l, if l.vars.contains(&Wg::Stage) { Wg::Stage } else { Wg::Tile }),
        misalignment(l, if l.vars.contains(&Wg::Stage) { Wg::Stage } else { Wg::Tile }),
        coop_bad(v).len(),
        coop_shift(v),
        expected_shift(l),
        &v[..4]
    )
}

fn layout(name: &str) -> &'static Layout {
    LAYOUTS
        .iter()
        .find(|l| l.name == name)
        .unwrap_or_else(|| panic!("no layout named {name}"))
}

#[test]
fn workgroup_alias_probe_matrix_prints_every_arm() {
    let ctx = ctx_or_panic("workgroup_alias_probe_matrix");
    for zi in [true, false] {
        for a in ARMS {
            let v = run_arm(ctx, a, zi);
            eprintln!("  {}", report(a, zi, &v));
        }
    }
}

#[test]
fn coop_alias_probe_matrix_prints_every_layout() {
    let ctx = ctx_or_panic("coop_alias_probe_matrix");
    for l in LAYOUTS {
        let v = run_coop(ctx, l);
        eprintln!("  {}", coop_report(l, &v));
    }
}

#[test]
fn an_f16_workgroup_array_with_no_other_workgroup_variable_is_clean() {
    let ctx = ctx_or_panic("f16_tile_alone_control");
    for zi in [true, false] {
        let a = arm("f16_tile_alone_control");
        let v = run_arm(ctx, a, zi);
        assert_eq!(
            bad_elems(&v),
            Vec::<usize>::new(),
            "{}: an f16 workgroup array read back wrong with nothing declared beside it, so every \
             other arm of this suite is measuring something other than the defect",
            report(a, zi, &v)
        );
    }
}

#[test]
fn plain_indexed_reads_of_a_workgroup_array_never_alias_whatever_is_declared_beside_them() {
    let ctx = ctx_or_panic("plain_indexed_reads");
    for a in ARMS {
        for zi in [true, false] {
            let v = run_arm(ctx, a, zi);
            assert_eq!(
                bad_elems(&v),
                Vec::<usize>::new(),
                "{}: {} {}",
                report(a, zi, &v),
                A_PLAIN_INDEXED_READ_OF_A_WORKGROUP_F16_ARRAY_IS_ADDRESSED_CORRECTLY_SO_ONLY_COOPERATIVE_MATRIX_KERNELS_ARE_EXPOSED,
                A_COOPERATIVE_MATRIX_ACCESS_TO_A_WORKGROUP_ARRAY_TRUNCATES_ITS_ADDRESS_TO_16_BYTES_ON_NAGA_30_0_0_WITH_NVIDIA_590_48_01
            );
            if a.scalars != Scalars::None {
                assert_eq!(
                    v[SCALAR_SLOT], SCALAR_MARK,
                    "{}: the flag did not read back its own store, so this arm proves nothing",
                    report(a, zi, &v)
                );
            }
        }
    }
}

#[test]
fn a_cooperative_matrix_access_lands_at_the_16_byte_boundary_below_its_workgroup_array() {
    let ctx = ctx_or_panic("coop_declaration_order");
    let mut ran = 0;
    let mut shifted = 0;
    for l in LAYOUTS {
        if l.dyn_stride {
            continue;
        }
        let v = run_coop(ctx, l);
        let line = coop_report(l, &v);
        eprintln!("  {line}");
        assert_eq!(
            coop_shift(&v),
            Some(expected_shift(l)),
            "{line}: {} {}",
            A_COOPERATIVE_MATRIX_ACCESS_TO_A_WORKGROUP_ARRAY_TRUNCATES_ITS_ADDRESS_TO_16_BYTES_ON_NAGA_30_0_0_WITH_NVIDIA_590_48_01,
            REPLACING_A_WORKGROUP_SCALAR_WITH_A_ONE_ELEMENT_ARRAY_DOES_NOT_FIX_THE_COOPERATIVE_MATRIX_SHIFT
        );
        ran += 1;
        if expected_shift(l) != 0 {
            shifted += 1;
        }
    }
    assert!(
        ran >= 20 && shifted >= 8,
        "the coop layout matrix ran {ran} arms of which {shifted} predicted a shift; with fewer \
         than 20 arms and 8 shifting arms this suite is not characterizing the defect"
    );
}

#[test]
fn the_shipping_cooperative_matrix_kernel_layouts_put_every_tile_on_a_16_byte_workgroup_offset() {
    let ctx = ctx_or_panic("shipping_coop_layouts");
    for name in [
        "gemm_nvfp4_and_moe_grouped_gemm_declarations_tile_at_coop_b",
        "q3d_pf_flash_tiled_coop_declarations_tile_at_fdc_psl",
        "gemm_coop_f16_declarations_tile_at_xas",
    ] {
        let l = layout(name);
        assert_eq!(
            misalignment(l, Wg::Tile),
            0,
            "{name}: this arm no longer mirrors a shipping kernel whose cooperative-matrix tile \
             sits at a 16-byte-aligned workgroup offset, so it cannot exonerate one"
        );
        let v = run_coop(ctx, l);
        let line = coop_report(l, &v);
        eprintln!("  {line}");
        assert_eq!(
            coop_bad(&v),
            Vec::<usize>::new(),
            "{line}: a workgroup layout shaped like a shipping cooperative-matrix kernel read the \
             wrong tile. {}",
            A_COOPERATIVE_MATRIX_ACCESS_TO_A_WORKGROUP_ARRAY_TRUNCATES_ITS_ADDRESS_TO_16_BYTES_ON_NAGA_30_0_0_WITH_NVIDIA_590_48_01
        );
    }
}

#[test]
fn a_uniform_stride_argument_hides_the_cooperative_matrix_declaration_offset_defect() {
    let ctx = ctx_or_panic("coop_dynamic_stride");
    for name in [
        "tile_then_atomic_and_u32_scalar_dynamic_stride",
        "f32_array_then_tile_dynamic_stride",
    ] {
        let l = layout(name);
        let v = run_coop(ctx, l);
        let line = coop_report(l, &v);
        eprintln!("  {line}");
        assert_eq!(
            coop_bad(&v),
            Vec::<usize>::new(),
            "{line}: {}",
            A_NON_CONSTANT_STRIDE_ARGUMENT_MOVES_THE_COOPERATIVE_MATRIX_LOAD_ONTO_A_CORRECTLY_ADDRESSED_PATH
        );
    }
}

pub const AN_ELEMENT_OFFSET_INTO_A_WORKGROUP_TILE_MUST_ALSO_LAND_ON_16_BYTES: &str =
    "The truncation is on the final byte address, not on the variable's base, so an offset inside \
     the array is subject to the same rule: every shipping kernel indexes its cooperative-matrix \
     tiles at whole 16x16 fragments, which are 512 bytes for f16 and 1024 for f32 and therefore \
     safe, but a tile taken at an odd element slides to the 16-byte boundary below it and reads \
     neighbouring rows.";

fn elem_offset_source(off: usize) -> String {
    use std::fmt::Write as _;
    let mut b = String::new();
    b.push_str("enable f16;\nenable wgpu_cooperative_matrix;\n");
    writeln!(b, "alias PBA = coop_mat{COOP_T}x{COOP_T}<f16, A>;").unwrap();
    writeln!(b, "alias PBB = coop_mat{COOP_T}x{COOP_T}<f16, B>;").unwrap();
    writeln!(b, "alias PBC = coop_mat{COOP_T}x{COOP_T}<f32, C>;\n").unwrap();
    b.push_str("@group(0) @binding(0) var<storage, read_write> pb_out: array<f32>;\n");
    b.push_str("@group(0) @binding(1) var<storage, read> pb_zero: array<f32>;\n");
    b.push_str("@group(0) @binding(2) var<storage, read> pb_id: array<f16>;\n\n");
    let n = COOP_ELEMS * 2;
    writeln!(b, "var<workgroup> pb_tile: array<f16, {n}>;").unwrap();
    writeln!(b, "\n@compute @workgroup_size({COOP_WG})").unwrap();
    b.push_str("fn pb_coop(@builtin(local_invocation_index) lidx: u32) {\n");
    writeln!(
        b,
        "    for (var i = lidx; i < {n}u; i = i + {COOP_WG}u) {{ pb_tile[i] = f16(f32(i) + 1.0); }}"
    )
    .unwrap();
    b.push_str("    workgroupBarrier();\n");
    writeln!(b, "    let a = coopLoadT<PBA>(&pb_tile[{off}], {COOP_T}u);").unwrap();
    writeln!(b, "    let bb = coopLoadT<PBB>(&pb_id[0], {COOP_T}u);").unwrap();
    writeln!(b, "    let z = coopLoadT<PBC>(&pb_zero[0], {COOP_T}u);").unwrap();
    b.push_str("    let c = coopMultiplyAdd(a, bb, z);\n");
    writeln!(b, "    coopStoreT(c, &pb_out[0], {COOP_T}u);").unwrap();
    b.push_str("}\n");
    b
}

#[test]
fn a_cooperative_matrix_element_offset_that_is_not_a_multiple_of_sixteen_bytes_slides_the_tile() {
    let ctx = ctx_or_panic("coop_element_offset");
    for off in [0usize, 1, 2, 8, 16, 24, 32, 256] {
        let src = elem_offset_source(off);
        let label = format!("coop_elem_offset_{off}");
        dispatch::coop_matrix_guard(ctx, &label, &src, "pb_coop")
            .unwrap_or_else(|e| panic!("{label}: no cooperative-matrix config: {e}"));
        let pl = pipeline(ctx, &label, &src, "pb_coop", true);
        let out = dispatch::storage_zeroed(ctx, &label, (COOP_ELEMS * 4) as u64);
        let zero = dispatch::storage_from_slice(ctx, "pb_zero", &vec![0f32; COOP_ELEMS]);
        let mut ident = vec![0u16; COOP_ELEMS];
        for i in 0..COOP_T {
            ident[i * COOP_T + i] = half::f16::from_f32(1.0).to_bits();
        }
        let idb = dispatch::storage_from_slice(ctx, "pb_id", &ident);
        dispatch::dispatch(ctx, &pl, &[(0, &out), (1, &zero), (2, &idb)], (1, 1, 1))
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let v = dispatch::read_back::<f32>(ctx, &out, COOP_ELEMS)
            .unwrap_or_else(|e| panic!("{label}: readback failed: {e}"));
        let base = off - (off % 8);
        let want: Vec<f32> = (0..COOP_ELEMS).map(|i| (base + i) as f32 + 1.0).collect();
        eprintln!(
            "  elem_offset={off:3} byte_offset={:4} truncated_to={:4} first={:?}",
            off * 2,
            base * 2,
            &v[..3]
        );
        assert_eq!(
            v, want,
            "{label}: a cooperative-matrix load at f16 element {off} did not read the tile \
             starting at element {base}. {} {}",
            AN_ELEMENT_OFFSET_INTO_A_WORKGROUP_TILE_MUST_ALSO_LAND_ON_16_BYTES,
            A_COOPERATIVE_MATRIX_ACCESS_TO_A_WORKGROUP_ARRAY_TRUNCATES_ITS_ADDRESS_TO_16_BYTES_ON_NAGA_30_0_0_WITH_NVIDIA_590_48_01
        );
    }
}
