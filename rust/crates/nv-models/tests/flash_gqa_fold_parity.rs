#![cfg(feature = "wgpu")]

mod common;
use common::flash_gqa_fold::{inputs, stage1_scratch, Shape};
use common::ctx;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::flash_decode as fd;
use nv_kernels::wgpu_backend::compose;
use nv_models::gemma4_e4b_wgpu::{flash1_e4b_entry, flash1_e4b_source, flash1_sg_supported};

fn parity_at(c: &WgpuContext, s: Shape, sg: bool, fold: u32, seed: u64) {
    let group = s.n_q / s.n_kv;
    assert!(
        group % fold == 0,
        "fold {fold} must divide the group {group} or the folded grid crosses a KV boundary"
    );
    let inp = inputs(c, &s, seed);
    let plain = stage1_scratch(
        c,
        &s,
        &inp,
        "fp-unfolded",
        &format!(
            "{}\n{}",
            compose(fd::WGSL),
            flash1_e4b_source(s.hd, sg)
        ),
        &flash1_e4b_entry(s.hd, sg),
        s.n_q,
    );
    let folded = stage1_scratch(
        c,
        &s,
        &inp,
        "fp-folded",
        &format!(
            "{}\n{}",
            compose(fd::WGSL),
            fd::fold_stage1_source(s.hd, sg, fold)
        ),
        &fd::fold_stage1_entry(s.hd, sg, fold),
        s.n_q / fold,
    );
    assert_eq!(
        plain.len(),
        folded.len(),
        "both arms write the same stage1 scratch geometry"
    );
    let diffs: Vec<usize> = (0..plain.len()).filter(|i| plain[*i] != folded[*i]).collect();
    assert!(
        diffs.is_empty(),
        "sg={sg} fold={fold} total={} start={}: {} of {} stage1 scratch words differ; first at \
         {:?} ({:#010x} vs {:#010x}). The fold reorders no arithmetic, so any difference at all \
         means a chain saw a different position, a different warp, or a different reduce",
        s.total,
        s.start,
        diffs.len(),
        plain.len(),
        diffs.first(),
        plain[diffs.first().copied().unwrap_or(0)],
        folded[diffs.first().copied().unwrap_or(0)]
    );
}

#[test]
fn folded_stage1_is_bit_identical_to_unfolded_at_e4b_geometry() {
    let c = ctx();
    let sg = flash1_sg_supported(c);
    for (total, start) in [(1u32, 0u32), (129, 0), (1000, 0), (4096, 3584), (8192, 0)] {
        for fold in [2u32, 4] {
            for reduce in [false, sg] {
                parity_at(
                    c,
                    Shape {
                        n_q: 8,
                        n_kv: 2,
                        hd: 256,
                        total,
                        start,
                    },
                    reduce,
                    fold,
                    0xbeef ^ (total as u64) << 8 ^ fold as u64,
                );
            }
        }
    }
}

#[test]
fn folded_stage1_is_bit_identical_at_a_wider_group() {
    let c = ctx();
    for fold in [2u32, 4, 8] {
        parity_at(
            c,
            Shape {
                n_q: 16,
                n_kv: 2,
                hd: 256,
                total: 4096,
                start: 0,
            },
            false,
            fold,
            0xf01d ^ fold as u64,
        );
    }
}
