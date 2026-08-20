#![cfg(feature = "cuda")]

use candle_core::Device;
use cudarc::driver::{CudaContext, CudaStream};
use nv_models::gemma4_batch_graph::graph_teardown::CtorForkGuard;
use std::sync::Arc;

const NO_CUDA: &str = "no CUDA device 0: this soak is the only gate proving N failed graph-engine \
                       ctor rounds hold no device memory and must not report success having \
                       executed nothing";

const FAILED_CTOR_ROUNDS_40_A_96_MIB_PER_ROUND_LEAK_WOULD_STRAND_3800_MIB: usize = 40;

const FREE_TOLERANCE_MIB_256_ABSORBS_ALLOCATOR_POOL_AND_CUBLASLT_WARMUP: f64 = 256.0;

fn device_free_mib() -> f64 {
    let (free, _total) = cudarc::driver::result::mem_get_info().expect("cuMemGetInfo");
    free as f64 / (1024.0 * 1024.0)
}

fn failed_ctor_round_forks_installs_the_warmup_caches_then_errors(
    ctx: &Arc<CudaContext>,
) -> (usize, u64, anyhow::Result<()>) {
    let mut key = 0usize;
    let mut epoch_before = 0u64;
    let r = (|| -> anyhow::Result<()> {
        let mut ctor_guard = CtorForkGuard::new();
        let forked: Arc<CudaStream> = ctor_guard.fork(ctx)?;
        key = forked.cu_stream() as usize;
        epoch_before = nv_quant::stream_epoch(key);
        nv_quant::nvfp4::ensure_workspace_for_stream(&forked)?;
        let _ = nv_quant::matmul::TensorCoreGemm::new(forked.clone())?;
        anyhow::bail!(
            "planted ctor failure: the S1 scenario is a graph-engine ctor that forks its stream, \
             warms up (installing the 64 MiB nvfp4 workspace and a cublasLt handle keyed on the \
             forked CUstream), then errors before the engine's Drop exists"
        )
    })();
    (key, epoch_before, r)
}

#[test]
fn n_failed_graph_ctor_rounds_leave_free_memory_flat_and_release_every_forked_streams_caches() {
    let Ok(device) = Device::new_cuda(0) else {
        panic!("{NO_CUDA}");
    };
    let Device::Cuda(d) = &device else {
        unreachable!()
    };
    let ctx = d.cuda_stream().context().clone();

    let (warm_key, warm_epoch, warm) =
        failed_ctor_round_forks_installs_the_warmup_caches_then_errors(&ctx);
    assert!(warm.is_err(), "the planted ctor failure must propagate");
    assert!(
        nv_quant::stream_epoch(warm_key) > warm_epoch,
        "warmup round: the guard did not release the forked stream's nv_quant caches"
    );

    let baseline = device_free_mib();
    let mut min_free = baseline;
    for round in 1..=FAILED_CTOR_ROUNDS_40_A_96_MIB_PER_ROUND_LEAK_WOULD_STRAND_3800_MIB {
        let (key, epoch_before, r) =
            failed_ctor_round_forks_installs_the_warmup_caches_then_errors(&ctx);
        assert!(r.is_err(), "round {round}: the planted ctor failure must propagate");
        assert!(
            nv_quant::stream_epoch(key) > epoch_before,
            "round {round}: CtorForkGuard left the forked stream's per-stream caches installed \
             (key {key:#x} epoch stayed {epoch_before}); each such round strands ~96 MiB keyed \
             to a destroyed stream and the retrying laguna_serve loop repeats it every proposal \
             round"
        );
        let free = device_free_mib();
        min_free = min_free.min(free);
        if round % 10 == 0 || round == 1 {
            eprintln!(
                "[ctor-soak] round={round} free_mib={free:.1} baseline_mib={baseline:.1} \
                 held_mib={:.1}",
                baseline - free
            );
        }
    }
    let held = baseline - min_free;
    assert!(
        held <= FREE_TOLERANCE_MIB_256_ABSORBS_ALLOCATOR_POOL_AND_CUBLASLT_WARMUP,
        "{} failed ctor rounds drove free memory {held:.1} MiB below the post-warmup baseline \
         {baseline:.1} MiB (tolerance {} MiB) -- failed graph-engine construction is stranding \
         per-stream workspaces again",
        FAILED_CTOR_ROUNDS_40_A_96_MIB_PER_ROUND_LEAK_WOULD_STRAND_3800_MIB,
        FREE_TOLERANCE_MIB_256_ABSORBS_ALLOCATOR_POOL_AND_CUBLASLT_WARMUP
    );
    eprintln!(
        "[ctor-soak] VERDICT rounds={} baseline_mib={baseline:.1} min_free_mib={min_free:.1} \
         held_mib={held:.1} FLAT",
        FAILED_CTOR_ROUNDS_40_A_96_MIB_PER_ROUND_LEAK_WOULD_STRAND_3800_MIB
    );
}
