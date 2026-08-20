#![cfg(feature = "wgpu")]

use half::f16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::gemm_spirv as spv;

const PROBE_SPV: &[u8] = include_bytes!("fixtures/spirv/coopmat_16x8x16_probe.spv");

pub const SPIRV_PASSTHROUGH_IS_THE_ROUTE_TO_GLSL_CLASS_COOPMAT_GEMM_CODEGEN: &str =
    "llama.cpp's KHR-class prefill rate (current numbers: perf/runs.jsonl) rides \
     GLSL-compiled cooperative-matrix SPIR-V whose staging and unrolling naga's WGSL arm \
     does not reproduce, so matching it from this engine requires \
     Features::PASSTHROUGH_SHADERS on the Vulkan backend and a precompiled SPIR-V GEMM; \
     every blob stays on the square 16x16x16 f16xf16->f32 subgroup-scope fragment because \
     that is the only f16 shape RADV advertises (raw \
     vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR on RADV STRIX_HALO lists square 16 \
     only) and NVIDIA advertises it too -- an unadvertised fragment shape such as 16x8x16 \
     is invalid usage the driver silently compiles to a kernel that stores nothing";

pub const RDNA3_WMMA_ACCUMULATION_IS_NOT_IEEE_SO_THE_PROBE_SPLITS_BY_SIGN: &str =
    "v_wmma_f32_16x16x16_f16 on RADV STRIX_HALO loses low mantissa bits on dots that mix \
     product signs -- a one-step dot of clean f16 operands {+3, -1} returns 2 - 2^-23 in \
     both wave32 and wave64, with operands proven bit-clean in LDS and registers -- while \
     every all-same-sign dot observed is exact; the probe therefore computes \
     (A+ B+ + A- B-) - (A+ B- + A- B+) from sign-split non-negative operands, four \
     all-non-negative coopMatMulAdds combined with exact f32 adds, which is exact on \
     integer-valued inputs on both this unit and IEEE tensor cores";

#[test]
fn passthrough_shaders_grant_state_is_reported_for_the_spirv_gemm_route() {
    let c = match WgpuContext::shared() {
        Ok(c) => c,
        Err(e) => panic!("no wgpu adapter: {e}"),
    };
    let granted = c
        .device
        .features()
        .contains(wgpu::Features::PASSTHROUGH_SHADERS);
    let backend = format!("{:?}", c.adapter.get_info().backend);
    eprintln!(
        "[spirv-probe] backend={backend} PASSTHROUGH_SHADERS granted={granted} \
         (route: {})",
        if granted {
            "OPEN -- precompiled SPIR-V coopmat GEMM is loadable"
        } else {
            "CLOSED on this adapter -- rectangular coopmat unreachable from this engine"
        }
    );
    assert_eq!(
        granted,
        c.adapter
            .features()
            .contains(wgpu::Features::PASSTHROUGH_SHADERS),
        "the device must have been created with passthrough whenever the adapter offers \
         it; device.rs requests it unconditionally when available"
    );
}

fn pack_f16(v: &[f16]) -> Vec<u32> {
    let mut out = vec![0u32; v.len().div_ceil(2)];
    for (i, x) in v.iter().enumerate() {
        out[i / 2] |= (x.to_bits() as u32) << (16 * (i % 2));
    }
    out
}

#[test]
fn rectangular_coopmat_spirv_dispatches_and_matches_the_exact_integer_matmul() {
    let c = match WgpuContext::shared() {
        Ok(c) => c,
        Err(e) => panic!("no wgpu adapter: {e}"),
    };
    if !c
        .device
        .features()
        .contains(wgpu::Features::PASSTHROUGH_SHADERS)
    {
        panic!(
            "PASSTHROUGH_SHADERS not granted: the spirv gemm route this probe guards is \
             closed on this adapter; the grant-state test above reports the same"
        );
    }
    let (m, n, k) = (16usize, 8usize, 16usize);
    let a: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(((i * 7 + 3) % 9) as f32 - 4.0))
        .collect();
    let b: Vec<f16> = (0..k * n)
        .map(|i| f16::from_f32(((i * 5 + 1) % 7) as f32 - 3.0))
        .collect();
    let mut want = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for ki in 0..k {
                acc += a[mi * k + ki].to_f32() * b[ki * n + ni].to_f32();
            }
            want[mi * n + ni] = acc;
        }
    }

    let words: Vec<u32> = PROBE_SPV
        .chunks_exact(4)
        .map(|c4| u32::from_le_bytes([c4[0], c4[1], c4[2], c4[3]]))
        .collect();
    let validation = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = unsafe {
        c.device
            .create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some("coopmat-16x8x16-probe"),
                entry_points: std::borrow::Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint {
                    name: std::borrow::Cow::Borrowed("main"),
                    workgroup_size: (32, 1, 1),
                }]),
                spirv: Some(std::borrow::Cow::Borrowed(&words)),
                ..Default::default()
            })
    };
    if let Some(err) = pollster::block_on(validation.pop()) {
        panic!("spirv passthrough module: {err}");
    }
    let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..3)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: i < 2 },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let bgl = c
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("coopmat-probe-bgl"),
            entries: &bgl_entries,
        });
    let pl = c
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("coopmat-probe-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let pipeline = c
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("coopmat-probe"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let ab = dispatch::storage_from_slice(c, "probe-a", &pack_f16(&a));
    let bb = dispatch::storage_from_slice(c, "probe-b", &pack_f16(&b));
    let db = dispatch::storage_zeroed(c, "probe-d", (m * n * 4) as u64);
    let bind = dispatch::bind_group_offsets(
        c,
        &pipeline,
        &[(0, &ab, 0), (1, &bb, 0), (2, &db, 0)],
    );
    let mut enc = c.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    c.queue.submit([enc.finish()]);
    c.poll_blocking().expect("poll");
    let got: Vec<f32> = dispatch::read_back(c, &db, m * n)
        .expect("read back")
        .iter()
        .map(|w| f32::from_bits(*w))
        .collect();
    let ndiff = got
        .iter()
        .zip(want.iter())
        .filter(|(g, w)| *g != *w)
        .count();
    assert_eq!(
        ndiff, 0,
        "rectangular 16x8x16 coopmat product must be exact on integer-valued f16 inputs; \
         {ndiff} of {} differ (got[0..4]={:?} want[0..4]={:?})",
        m * n,
        &got[..4],
        &want[..4]
    );
    eprintln!(
        "[spirv-probe] rectangular coopmat 16x8x16 dispatched via passthrough: exact ({}/{})",
        m * n - ndiff,
        m * n
    );
}

#[test]
fn every_checked_in_mulmm_blob_builds_a_validation_clean_passthrough_pipeline() {
    let c = match WgpuContext::shared() {
        Ok(c) => c,
        Err(e) => panic!("no wgpu adapter: {e}"),
    };
    if let Some(why) = spv::preflight(c) {
        panic!("spirv mulmm route closed on this adapter: {why}");
    }
    let mut built = 0usize;
    for blob in spv::blobs() {
        let g = spv::pipeline(c, blob)
            .unwrap_or_else(|e| panic!("{} failed to build: {e}", blob.name));
        assert_eq!(g.blob.name, blob.name);
        built += 1;
    }
    assert!(
        built >= 4,
        "the blob table must at least cover w4/w8 x f32/y16; found {built}"
    );
    eprintln!("[spirv-probe] {built} mulmm blobs built passthrough pipelines validation-clean");
}
