#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::compose;
use nv_kernels::wgpu_backend::kernels::flash_decode as fd;

fn twins() -> [(&'static str, String); 3] {
    [
        (fd::ENTRY_STAGE1_FP8_MK_SD, fd::mk_stage1_source_sd()),
        (fd::ENTRY_STAGE1_FP8_MK_U_SD, fd::mk_u_stage1_source_sd()),
        (fd::ENTRY_SMV2_STAGE1_FP8_SD, fd::smv2_stage1_source_sd()),
    ]
}

#[test]
fn a_missed_anchor_silently_applies_exact_decode_magnitudes_against_2pow120_folded_scales() {
    for (entry_sd, src) in twins() {
        assert!(
            src.contains(&format!("fn {entry_sd}(")),
            "{entry_sd}: twin entry not renamed; the pipeline would compile the stock exact-decode \
             body under the shift-decode name"
        );
        assert_eq!(
            src.matches("fd_k_fp8(").count(),
            0,
            "{entry_sd}: a bare exact k-decode call survived; its output lacks the 2pow-120 carry \
             the folded k-scale expects, so scores come out 2pow120 too large"
        );
        assert_eq!(
            src.matches("fd_v_fp8(").count(),
            0,
            "{entry_sd}: a bare exact v-decode call survived; its output lacks the 2pow-120 carry \
             the folded v-scale expects, so the attention output comes out 2pow120 too large"
        );
        assert_eq!(
            src.matches("0x7B800000").count(),
            2,
            "{entry_sd}: expected exactly one 2pow120 fold on the k-scale and one on the v-scale; \
             any other count leaves shift-decoded values off by a factor of 2pow120 per missing or \
             doubled fold"
        );
        assert!(
            src.contains("fd_k_fp8_sd(") && src.contains("fd_v_fp8_sd("),
            "{entry_sd}: shift decoders absent, the twin would be an exact-decode copy with folded \
             scales and wrong magnitudes"
        );
    }
}

#[test]
fn twins_leave_stock_entries_untouched_and_validate_as_wgsl_appended_to_the_stock_module() {
    for (entry_sd, src) in twins() {
        let stock = entry_sd.strip_suffix("_sd").expect("twin names end in _sd");
        assert!(
            !src.contains(&format!("fn {stock}(")),
            "{entry_sd}: twin still declares the stock entry, appending it to the stock module \
             would redefine {stock} and fail every pipeline built from that module"
        );
        assert!(
            fd::WGSL.contains(&format!("fn {stock}(")),
            "{entry_sd}: stock entry {stock} vanished from flash_decode.wgsl; callers routing the \
             exact-decode path would fail pipeline creation"
        );
        let module = compose(&format!("{}\n{}", fd::WGSL, src));
        let parsed = naga::front::wgsl::parse_str(&module)
            .unwrap_or_else(|e| panic!("{entry_sd}: wgsl parse: {}", e.message()));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&parsed)
        .unwrap_or_else(|e| panic!("{entry_sd}: validate: {e}"));
    }
}
