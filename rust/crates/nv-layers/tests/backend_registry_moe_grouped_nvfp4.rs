use nv_layers::backend::{kind_supports, missing_on, supporting_backends, BackendKind, KernelId};

#[cfg(feature = "wgpu")]
const MOE_HIDDEN_SIZES_FROM_THE_CUDA_WGPU_DIVERGENCE_SWEEP: [usize; 2] = [2048, 2816];
#[cfg(feature = "wgpu")]
const MOE_INTERMEDIATE_SIZES_FROM_THE_CUDA_WGPU_DIVERGENCE_SWEEP: [usize; 5] =
    [512, 640, 704, 768, 896];

#[cfg(feature = "wgpu")]
fn moe_gemm_shapes() -> Vec<(usize, usize)> {
    let mut shapes = Vec::new();
    for hidden in MOE_HIDDEN_SIZES_FROM_THE_CUDA_WGPU_DIVERGENCE_SWEEP {
        for inter in MOE_INTERMEDIATE_SIZES_FROM_THE_CUDA_WGPU_DIVERGENCE_SWEEP {
            shapes.push((inter, hidden));
            shapes.push((hidden, inter));
        }
    }
    shapes
}

#[test]
fn wgpu_reports_the_grouped_nvfp4_moe_gemm_that_moe_wgpu_dispatches() {
    assert!(
        kind_supports(BackendKind::Wgpu, KernelId::MoeGroupedGemmNvfp4),
        "moe_wgpu::try_forward pushes wgpu_backend::kernels::moe_grouped_gemm for the gate, up \
         and down expert gemms, and wgpu_correct_moe_cuda_shape_sweep holds that path bit-exact \
         against forward_grouped on CUDA: the capability registry must not report it missing"
    );
    assert!(
        missing_on(BackendKind::Wgpu, &[KernelId::MoeGroupedGemmNvfp4]).is_empty(),
        "missing_on is the only thing wgpu_missing_kernels and cuda_only_fast_paths read"
    );
    assert_eq!(
        supporting_backends(KernelId::MoeGroupedGemmNvfp4),
        vec![BackendKind::Cuda, BackendKind::Wgpu]
    );
}

#[test]
fn marlin_w4a16_stays_missing_on_wgpu_because_no_wgsl_module_implements_it() {
    assert!(
        !kind_supports(BackendKind::Wgpu, KernelId::MarlinGemmW4a16),
        "wgpu_backend/kernels ships gemm_w4a16_small_m and gemv_w4a16, neither of which is the \
         marlin w4a16 kernel: this one is truthfully absent and must keep reporting missing"
    );
    let wgpu_missing = missing_on(BackendKind::Wgpu, &KernelId::ALL);
    assert_eq!(
        wgpu_missing,
        vec![KernelId::MarlinGemmW4a16],
        "wgpu missing set drifted; update kind_supports against src/wgpu_backend/kernels/"
    );
    eprintln!(
        "backend registry: {} kernels, wgpu missing = {:?}",
        KernelId::ALL.len(),
        wgpu_missing.iter().map(|k| k.name()).collect::<Vec<_>>()
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn every_moe_gemm_shape_in_the_divergence_sweep_selects_a_wgsl_entry_that_exists() {
    use nv_kernels::wgpu_backend::kernels::moe_grouped_gemm::{
        scalar_source, select_scalar_variant, ScalarVariant,
    };
    let source = scalar_source();
    let shapes = moe_gemm_shapes();
    assert_eq!(shapes.len(), 20, "sweep shape product changed");
    for (n, k) in shapes {
        assert!(
            k.is_multiple_of(nv_layers::moe_wgpu::BLOCK_SIZE),
            "k={k} is not a whole number of nvfp4 blocks; the grouped gemm computes k_blocks = k \
             / BLOCK_SIZE with no remainder"
        );
        let variant = select_scalar_variant(n, k);
        assert!(
            variant.supports(n, k),
            "select_scalar_variant returned {variant:?} for (n={n}, k={k}) which it does not \
             support"
        );
        assert!(
            ScalarVariant::Base.supports(n, k),
            "the base scalar entry is the unconditional fallback for (n={n}, k={k})"
        );
        assert!(
            source.contains(variant.entry()),
            "wgsl/moe_grouped_gemm.wgsl scalar section has no entry point {} for (n={n}, k={k})",
            variant.entry()
        );
    }
    eprintln!("moe grouped nvfp4 wgsl entries resolved for 20 (n, k) shapes");
}
