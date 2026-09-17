use std::sync::{Arc, Mutex, OnceLock};

use anyhow::Result;
use nv_kernels::wgpu_backend::{na, WgpuContext};

use super::vision_attn_msl;

pub const CAUSAL_ENTRY: &str = "dg_steel_causal";

pub const PREFILL_STEEL_IS_THE_VISION_KERNEL_WITH_DO_CAUSAL_TRUE_AND_QL_OFF_CARRYING_THE_CHUNK_BASE: &str =
    "The decoder prefill reuses the vendored steel attention verbatim through the vision \
     driver's source builder with exactly two replaces: do_causal flips to true, so the \
     kernel skips every key block past the diagonal and masks the partial one, and the \
     host_name is renamed so both instantiations can live in one process. qL_off carries \
     the chunk's absolute position, which is how one recorded pipeline serves every chunk \
     of the width ladder at any base. align_Q is baked true because every ladder width is \
     a multiple of the 32-row query block; align_K is baked false because kL is base plus \
     chunk and the base is only known at dispatch, written into the params buffer by the \
     dg_steel_pp pre-pass on the GPU itself. GQA is native: Q rides in as twelve heads \
     with gqa_factor six, so the qpack replication pass of the GEMM-form arm disappears.";

fn causal_source() -> String {
    let base = vision_attn_msl::steel_source(true, false);
    let causal_from = "constant bool do_causal = false;";
    let entry_from = format!(
        "host_name(\"{}\")",
        vision_attn_msl::steel_entry(true, false)
    );
    assert!(
        base.contains(causal_from) && base.contains(&entry_from),
        "steel source lost the do_causal or host_name anchor"
    );
    base.replace(causal_from, "constant bool do_causal = true;")
        .replace(&entry_from, &format!("host_name(\"{CAUSAL_ENTRY}\")"))
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub struct CausalSteelPipelines {
    pub pipeline: Arc<wgpu::ComputePipeline>,
    pub layout: wgpu::BindGroupLayout,
}

type CacheRow = (
    wgpu::Device,
    std::result::Result<Arc<CausalSteelPipelines>, String>,
);

fn cache() -> &'static Mutex<Vec<CacheRow>> {
    static CACHE: OnceLock<Mutex<Vec<CacheRow>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn build(ctx: &WgpuContext) -> Result<Arc<CausalSteelPipelines>> {
    let _ =
        PREFILL_STEEL_IS_THE_VISION_KERNEL_WITH_DO_CAUSAL_TRUE_AND_QL_OFF_CARRYING_THE_CHUNK_BASE;
    let src: &'static str = Box::leak(causal_source().into_boxed_str());
    let module = na::msl_module(ctx, "dg-steel-causal", &[(CAUSAL_ENTRY, (128, 1, 1))], src)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let entries = [
        storage_entry(0, true),
        storage_entry(1, true),
        storage_entry(2, true),
        storage_entry(3, false),
        uniform_entry(4),
        uniform_entry(5),
        storage_entry(6, true),
        storage_entry(7, true),
    ];
    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dg-steel-causal"),
            entries: &entries,
        });
    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dg-steel-causal"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let pipeline = Arc::new(
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(CAUSAL_ENTRY),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(CAUSAL_ENTRY),
                compilation_options: Default::default(),
                cache: None,
            }),
    );
    if let Some(err) = pollster::block_on(scope.pop()) {
        anyhow::bail!("dg-steel-causal pipeline: {err}");
    }
    Ok(Arc::new(CausalSteelPipelines {
        pipeline,
        layout: bgl,
    }))
}

pub fn causal_pipelines(ctx: &'static WgpuContext) -> Result<Arc<CausalSteelPipelines>> {
    let mut guard = cache().lock().unwrap();
    for (dev, entry) in guard.iter() {
        if *dev == ctx.device {
            return entry.clone().map_err(|e| anyhow::anyhow!("{e}"));
        }
    }
    let built = build(ctx);
    let stored = match &built {
        Ok(p) => Ok(p.clone()),
        Err(e) => Err(format!("{e:#}")),
    };
    guard.push((ctx.device.clone(), stored));
    built
}
