
#![cfg(feature = "wgpu")]
#![allow(dead_code)]

use nv_kernels::wgpu_backend::device::WgpuContext;

pub fn wgpu_allow_skip() -> bool {
    std::env::var("NV_KERNELS_WGPU_ALLOW_SKIP").as_deref() == Ok("1")
}

pub fn parity_allow_skip() -> bool {
    std::env::var("NV_KERNELS_PARITY_ALLOW_SKIP").as_deref() == Ok("1")
}

pub fn parity_require() -> bool {
    !parity_allow_skip()
}

pub fn wgpu_ctx_or_skip(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("{test}: {}", ctx.summary());
            let st = ctx.qualify();
            if !st.qualified {
                if wgpu_allow_skip() {
                    eprintln!(
                        "SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1), NOT PASSED: {test}: adapter not \
                         qualified: {:?}",
                        st.reason
                    );
                    return None;
                }
                panic!(
                    "{test}: wgpu adapter not qualified: {:?}. Set NV_KERNELS_WGPU_ALLOW_SKIP=1 \
                     to skip on purpose.",
                    st.reason
                );
            }
            Some(ctx)
        }
        Err(e) => {
            if wgpu_allow_skip() {
                eprintln!(
                    "SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1), NOT PASSED: {test}: no wgpu adapter: {e}"
                );
                return None;
            }
            panic!(
                "{test}: no wgpu adapter: {e}. nvk.sh wires VK_ICD_FILENAMES and the store's \
                 vulkan-loader, so a miss means that wiring regressed. Set \
                 NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
            );
        }
    }
}
