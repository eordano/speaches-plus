use wgpu::util::DeviceExt;

use super::device::WgpuContext;
use super::qualify;
use super::{Result, WgpuError};

#[path = "buffer.rs"]
pub mod buffer;

pub use buffer::{GpuBind, GpuTensor, GpuUniform};

pub const DEFAULT_WORKGROUP_SIZE: u32 = 256;

pub fn storage_from_slice<T: bytemuck::Pod>(
    ctx: &WgpuContext,
    label: &str,
    data: &[T],
) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
}

pub fn storage_zeroed(ctx: &WgpuContext, label: &str, bytes: u64) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub fn uniform_from<T: bytemuck::Pod>(ctx: &WgpuContext, label: &str, value: &T) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(value),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
}

type PipelineCache =
    std::collections::HashMap<(u64, String, bool), std::sync::Arc<wgpu::ComputePipeline>>;

fn pipeline_cache() -> &'static std::sync::Mutex<PipelineCache> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<PipelineCache>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(PipelineCache::new()))
}

fn source_key(source: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut h);
    h.finish()
}

const NOZI_AUDITED_ENTRIES: &[&str] = &[
    "argmax_bf16_stage1",
    "argmax_bf16_stage2",
    "argmax_f32_rows_stage1",
    "argmax_f32_rows_stage2",
    "argmax_softcap_bf16_stage1",
    "flash_splitk_stage1_bf16kv",
    "flash_splitk_stage1_bf16kv_mk",
    "flash_splitk_stage1_bf16kv_mk_u",
    "flash_splitk_stage1_fp8kv",
    "flash_splitk_stage1_fp8kv_mk",
    "flash_splitk_stage1_fp8kv_mk_u",
    "g4m_argmax_bf16_stage1",
    "g4m_argmax_stage2",
    "g4m_attn_decode",
    "g4m_attn_norm_rope",
    "g4m_gemv_bf16",
    "g4m_gemv_w4",
    "g4m_norm",
    "g4m_norm_residual",
    "g4w_gemv_bf16_vec8_pk",
    "g4w_gemv_bf16_vec8_pk3",
    "g4w_gemv_fp8_pk",
    "g4w_gemv_fp8_pk3",
    "g4w_gemv_int8_pk",
    "g4w_gemv_int8_pk3",
    "g4w_gemv_legacy_pk",
    "g4w_gemv_legacy_pk3",
    "g4w_gemv_nvfp4_pk",
    "g4w_head_prep",
    "g4w_norm_add_norm",
    "g4w_norm_res_norm",
    "g4w_quant_row_pk",
    "gemv_bf16_normed",
    "gemv_bf16_scalar",
    "gemv_bf16_sg_v4_pk_wg128",
    "gemv_bf16_sg_v4_pk_wg256",
    "gemv_bf16_vec8",
    "gemv_bf16_vec8_v4",
    "gemv_i8_normed",
    "gemv_i8_normed_mk",
    "gemv_nvfp4_fdec_pk",
    "gemv_nvfp4_warp_pk",
    "gow_argmax_stage1",
    "gow_argmax_stage2",
    "gow_attn_decode",
    "gow_gemv_bf16",
    "gow_gemv_mx",
    "q3w_argmax_stage1",
    "q3w_argmax_stage2",
    "q3w_attn_decode",
    "q3w_attn_norm_rope",
    "q3w_attn_qk_norm_rope_qcast",
    "q3w_delta_head_fused",
    "q3w_delta_out",
    "q3w_delta_qkv",
    "q3w_delta_recurrent",
    "q3w_gemv_bf16",
    "q3w_gemv_dn_merged_fp8_qkv_fp8_z_bf16_ab",
    "q3w_gemv_nvfp4",
    "q3w_gemv_nvfp4_fdec",
    "q3w_gemv_nvfp4_warp",
    "q3w_quant_rows",
    "quantize_kv_fp8",
    "rmsnorm_bf16",
    "rmsnorm_f32",
    "rmsnorm_residual_bf16",
    "rmsnorm_residual_f32",
    "rowquant_i8",
];

const NOZI_PENDING_ENTRIES: &[&str] = &["gemv_nvfp4_bf16", "gemv_nvfp4_fmlut_pk"];

#[cfg(test)]
mod nozi_list_tests {
    #[test]
    fn audited_entry_list_is_sorted_and_unique() {
        assert!(super::NOZI_AUDITED_ENTRIES.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn pending_entry_list_is_sorted_and_unique() {
        assert!(super::NOZI_PENDING_ENTRIES.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn pending_entries_are_not_already_audited() {
        for e in super::NOZI_PENDING_ENTRIES {
            assert!(
                super::NOZI_AUDITED_ENTRIES.binary_search(e).is_err(),
                "{e} is in both the audited and the pending list"
            );
        }
    }

    #[test]
    fn pending_entries_are_inert_until_the_gate_opens() {
        for e in super::NOZI_PENDING_ENTRIES {
            assert!(!super::nozi_entry_listed(e, false));
            assert!(super::nozi_entry_listed(e, true));
        }
    }

    #[test]
    fn the_gate_does_not_reach_audited_or_unknown_entries() {
        for e in super::NOZI_AUDITED_ENTRIES {
            assert!(super::nozi_entry_listed(e, false));
            assert!(super::nozi_entry_listed(e, true));
        }
        assert!(!super::nozi_entry_listed("gemv_nvfp4_fmlut", true));
        assert!(!super::nozi_entry_listed("no_such_entry", true));
    }
}

#[cfg(test)]
mod coop_guard_tests {
    use super::coop_matrix_guard_against;
    use crate::wgpu_backend::kernels::{gemm_coop_f16, gemm_nvfp4};
    use crate::wgpu_backend::qualify::{CoopConfig, CoopScalar};

    fn strix_halo() -> Vec<CoopConfig> {
        vec![
            CoopConfig::new(16, 16, 16, CoopScalar::F16, CoopScalar::F16),
            CoopConfig::new(16, 16, 16, CoopScalar::F16, CoopScalar::F32),
        ]
    }

    fn probe_src(tile: u32, ab: &str, c: &str) -> String {
        format!(
            "enable f16;\nenable wgpu_cooperative_matrix;\n\
             alias CA = coop_mat{tile}x{tile}<{ab}, A>;\n\
             alias CB = coop_mat{tile}x{tile}<{ab}, B>;\n\
             alias CC = coop_mat{tile}x{tile}<{c}, C>;\n"
        )
    }

    #[test]
    fn the_f32_ab_probe_variant_is_refused_on_the_advertised_list() {
        let err = coop_matrix_guard_against(
            &strix_halo(),
            false,
            "probe",
            &probe_src(16, "f32", "f32"),
            "probe",
        )
        .expect_err("f32 A/B at 16x16x16 must not reach the driver");
        let msg = err.to_string();
        assert!(msg.contains("16x16x16 f32xf32->f32"), "{msg}");
        assert!(msg.contains("NV_KERNELS_WGPU_COOP_UNSAFE_SWEEP"), "{msg}");
    }

    #[test]
    fn the_advertised_variants_pass_and_the_unadvertised_tile_does_not() {
        let adv = strix_halo();
        for (ab, c) in [("f16", "f16"), ("f16", "f32")] {
            coop_matrix_guard_against(&adv, false, "probe", &probe_src(16, ab, c), "probe")
                .unwrap_or_else(|e| panic!("advertised {ab}->{c} was refused: {e}"));
        }
        assert!(
            coop_matrix_guard_against(&adv, false, "probe", &probe_src(8, "f16", "f32"), "probe")
                .is_err(),
            "8x8x8 is not on this adapter's list"
        );
    }

    #[test]
    fn the_unsafe_env_gate_lets_the_crashing_config_through() {
        coop_matrix_guard_against(
            &strix_halo(),
            true,
            "probe",
            &probe_src(16, "f32", "f32"),
            "probe",
        )
        .expect("the opt-in must restore the old ask-anyway behaviour");
    }

    #[test]
    fn sources_without_a_coop_matrix_type_are_untouched() {
        coop_matrix_guard_against(&[], false, "plain", "@compute fn main() {}", "main")
            .expect("a shader with no coop_mat type must not be judged");
        coop_matrix_guard_against(&[], false, "plain", &gemm_nvfp4::scalar_source(), "x")
            .expect("the scalar nvfp4 shader has no coop_mat type");
    }

    #[test]
    fn the_shipped_shaders_are_judged_against_the_list_they_need() {
        let coop16 = vec![CoopConfig::new(
            16,
            16,
            16,
            CoopScalar::F16,
            CoopScalar::F32,
        )];
        coop_matrix_guard_against(&coop16, false, "nvfp4", &gemm_nvfp4::coop_source(), "e")
            .expect("16x16x16 f16->f32 is exactly what the nvfp4 coop shader declares");
        assert!(
            coop_matrix_guard_against(
                &strix_halo(),
                false,
                "gemm8",
                &gemm_coop_f16::source(2, 8, 4, 1),
                "e"
            )
            .is_err(),
            "the 8x8 prefill GEMM must be refused where only 16x16x16 is advertised"
        );
        let coop8 = vec![CoopConfig::new(8, 8, 8, CoopScalar::F16, CoopScalar::F32)];
        coop_matrix_guard_against(
            &coop8,
            false,
            "gemm8",
            &gemm_coop_f16::source(2, 8, 4, 1),
            "e",
        )
        .expect("8x8x8 f16->f32 is what the prefill GEMM declares");
        assert!(
            coop_matrix_guard_against(&coop8, false, "nvfp4", &gemm_nvfp4::coop_source(), "e")
                .is_err(),
            "an 8x8x8-only adapter must not be handed the 16x16x16 nvfp4 shader"
        );
    }
}

pub fn nozi_audited_entries() -> &'static [&'static str] {
    NOZI_AUDITED_ENTRIES
}

pub fn nozi_env_enabled() -> bool {
    std::env::var("NV_WGPU_NOZI").ok().as_deref() != Some("0")
}

pub fn nozi_pending_enabled() -> bool {
    std::env::var("NV_WGPU_NOZI_NVFP4_V2").ok().as_deref() == Some("1")
}

pub fn nozi_entry_listed(entry: &str, pending: bool) -> bool {
    NOZI_AUDITED_ENTRIES.binary_search(&entry).is_ok()
        || (pending && NOZI_PENDING_ENTRIES.binary_search(&entry).is_ok())
}

fn nozi_for_entry(entry: &str) -> bool {
    nozi_env_enabled() && nozi_entry_listed(entry, nozi_pending_enabled())
}

fn pipeline_log(label: &str, entry: &str) {
    match std::env::var("NV_WGPU_PIPELINE_LOG") {
        Err(_) => {}
        Ok(v) if v == "0" || v.is_empty() => {}
        Ok(v) if v == "1" => eprintln!("[pipeline] {label}:{entry}"),
        Ok(path) => {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                let _ = writeln!(f, "[pipeline] {label}:{entry}");
            }
        }
    }
}

pub fn cached_compute_pipeline(
    ctx: &WgpuContext,
    label: &str,
    source: &str,
    entry: &str,
) -> Result<std::sync::Arc<wgpu::ComputePipeline>> {
    pipeline_log(label, entry);
    let nozi = nozi_for_entry(entry);
    let key = (source_key(source), entry.to_string(), nozi);
    if let Some(hit) = pipeline_cache().lock().unwrap().get(&key) {
        return Ok(hit.clone());
    }
    let built = std::sync::Arc::new(compute_pipeline_opts(ctx, label, source, entry, nozi)?);
    profile::name_pipeline(&built, &format!("{label}:{entry}"));
    pipeline_cache().lock().unwrap().insert(key, built.clone());
    Ok(built)
}

pub fn compute_pipeline(
    ctx: &WgpuContext,
    label: &str,
    source: &str,
    entry: &str,
) -> Result<wgpu::ComputePipeline> {
    compute_pipeline_opts(ctx, label, source, entry, false)
}

pub const UNCHECKED_SHADERS_ENV: &str = "NV_WGPU_UNCHECKED_SHADERS";

pub fn unchecked_shaders_opted_in() -> bool {
    std::env::var(UNCHECKED_SHADERS_ENV).ok().as_deref() == Some("1")
}

fn create_module_honoring_unchecked_optin(
    ctx: &WgpuContext,
    label: &str,
    source: &str,
) -> wgpu::ShaderModule {
    let desc = wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    };
    if unchecked_shaders_opted_in() {
        unsafe {
            ctx.device
                .create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked())
        }
    } else {
        ctx.device.create_shader_module(desc)
    }
}

pub fn coop_matrix_guard(ctx: &WgpuContext, label: &str, source: &str, entry: &str) -> Result<()> {
    coop_matrix_guard_against(
        &ctx.caps.coop_configs,
        qualify::coop_unsafe_sweep_enabled(),
        label,
        source,
        entry,
    )
}

pub fn coop_matrix_guard_against(
    advertised: &[qualify::CoopConfig],
    allow_unadvertised: bool,
    label: &str,
    source: &str,
    entry: &str,
) -> Result<()> {
    if !source.contains("coop_mat") {
        return Ok(());
    }
    for req in qualify::coop_requests_in_wgsl(source) {
        match qualify::coop_decide(&req, advertised, allow_unadvertised) {
            qualify::CoopDecision::Compile => {}
            qualify::CoopDecision::CompileUnadvertised(why) => eprintln!(
                "[coop] {label}::{entry}: {} is set, compiling anyway: {why}",
                qualify::COOP_UNSAFE_SWEEP_ENV
            ),
            qualify::CoopDecision::Skip(why) => {
                eprintln!(
                    "[coop] {label}::{entry}: SKIP unadvertised cooperative-matrix config: {why}"
                );
                return Err(WgpuError::Unsupported(format!(
                    "{label}::{entry}: refusing to compile an unadvertised cooperative-matrix \
                     configuration ({why}); set {}=1 to ask anyway",
                    qualify::COOP_UNSAFE_SWEEP_ENV
                )));
            }
        }
    }
    Ok(())
}

pub fn compute_pipeline_opts(
    ctx: &WgpuContext,
    label: &str,
    source: &str,
    entry: &str,
    nozi: bool,
) -> Result<wgpu::ComputePipeline> {
    coop_matrix_guard(ctx, label, source, entry)?;
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = create_module_honoring_unchecked_optin(ctx, label, source);
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[],
                zero_initialize_workgroup_memory: !nozi,
            },
            cache: None,
        });
    if let Some(err) = pollster::block_on(scope.pop()) {
        return Err(WgpuError::ShaderCompile(format!("{label}::{entry}: {err}")));
    }
    Ok(pipeline)
}

pub fn bind_group(
    ctx: &WgpuContext,
    pipeline: &wgpu::ComputePipeline,
    bindings: &[(u32, &wgpu::Buffer)],
) -> wgpu::BindGroup {
    let layout = pipeline.get_bind_group_layout(0);
    let entries: Vec<wgpu::BindGroupEntry> = bindings
        .iter()
        .map(|(binding, buf)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buf.as_entire_binding(),
        })
        .collect();
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &entries,
    })
}

pub fn bind_group_offsets(
    ctx: &WgpuContext,
    pipeline: &wgpu::ComputePipeline,
    bindings: &[(u32, &wgpu::Buffer, u64)],
) -> wgpu::BindGroup {
    let layout = pipeline.get_bind_group_layout(0);
    let entries: Vec<wgpu::BindGroupEntry> = bindings
        .iter()
        .map(|(binding, buf, offset)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: buf,
                offset: *offset,
                size: None,
            }),
        })
        .collect();
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &entries,
    })
}

pub fn dispatch(
    ctx: &WgpuContext,
    pipeline: &wgpu::ComputePipeline,
    bindings: &[(u32, &wgpu::Buffer)],
    workgroups: (u32, u32, u32),
) -> Result<()> {
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let group = bind_group(ctx, pipeline, bindings);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }
    ctx.queue.submit([enc.finish()]);
    if let Some(err) = pollster::block_on(scope.pop()) {
        return Err(WgpuError::ShaderCompile(format!("dispatch: {err}")));
    }
    Ok(())
}

pub fn run(
    ctx: &WgpuContext,
    label: &str,
    source: &str,
    entry: &str,
    bindings: &[(u32, &wgpu::Buffer)],
    workgroups: (u32, u32, u32),
) -> Result<()> {
    let pipeline = cached_compute_pipeline(ctx, label, source, entry)?;
    dispatch(ctx, &pipeline, bindings, workgroups)
}

pub fn run_resident(
    ctx: &WgpuContext,
    label: &str,
    source: &str,
    entry: &str,
    bindings: &[(u32, &dyn GpuBind)],
    workgroups: (u32, u32, u32),
) -> Result<()> {
    let raw: Vec<(u32, &wgpu::Buffer)> = bindings
        .iter()
        .map(|(slot, handle)| (*slot, handle.bind_buffer()))
        .collect();
    run(ctx, label, source, entry, &raw, workgroups)
}

type Pass = (
    std::sync::Arc<wgpu::ComputePipeline>,
    wgpu::BindGroup,
    (u32, u32, u32),
);

pub mod profile {
    use std::collections::{BTreeMap, HashMap};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};

    static ACC: OnceLock<Mutex<BTreeMap<String, (u64, f64)>>> = OnceLock::new();
    static ON: OnceLock<bool> = OnceLock::new();
    static NAMES: OnceLock<Mutex<HashMap<wgpu::ComputePipeline, String>>> = OnceLock::new();
    static PEND: OnceLock<Mutex<Vec<Parked>>> = OnceLock::new();
    static LOST: AtomicU64 = AtomicU64::new(0);
    static OVER: AtomicU64 = AtomicU64::new(0);

    const MAX_TRIES: u32 = 64;

    const AUTOFLUSH_AT: usize = 256;

    fn acc() -> &'static Mutex<BTreeMap<String, (u64, f64)>> {
        ACC.get_or_init(|| Mutex::new(BTreeMap::new()))
    }

    fn names() -> &'static Mutex<HashMap<wgpu::ComputePipeline, String>> {
        NAMES.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn pend() -> &'static Mutex<Vec<Parked>> {
        PEND.get_or_init(|| Mutex::new(Vec::new()))
    }

    pub fn name_pipeline(pipeline: &wgpu::ComputePipeline, name: &str) {
        if !enabled() {
            return;
        }
        if let Ok(mut m) = names().lock() {
            m.entry(pipeline.clone())
                .or_insert_with(|| name.to_string());
        }
    }

    pub fn pipeline_name(pipeline: &wgpu::ComputePipeline) -> Option<String> {
        names().lock().ok().and_then(|m| m.get(pipeline).cloned())
    }

    pub(super) struct Parked {
        pub(super) device: wgpu::Device,
        pub(super) queue: wgpu::Queue,
        pub(super) resolve: wgpu::Buffer,
        pub(super) sets: Vec<wgpu::QuerySet>,
        pub(super) labels: Vec<String>,
        pub(super) per_chunk: usize,
        pub(super) period: f64,
        pub(super) tries: u32,
    }

    pub(super) fn park(entry: Parked) {
        if let Ok(mut v) = pend().lock() {
            v.push(entry);
        }
    }

    pub(super) fn autoflush() {
        if pending() >= AUTOFLUSH_AT {
            flush();
        }
    }

    pub fn pending() -> usize {
        pend().lock().map(|v| v.len()).unwrap_or(0)
    }

    pub fn lost() -> u64 {
        LOST.load(Ordering::Relaxed)
    }

    pub fn over_ceiling() -> u64 {
        OVER.load(Ordering::Relaxed)
    }

    pub(super) fn record_over_ceiling(n: usize) {
        OVER.fetch_add(n as u64, Ordering::Relaxed);
    }

    pub fn flush() {
        let mut left = match pend().lock() {
            Ok(mut v) => std::mem::take(&mut *v),
            Err(_) => return,
        };
        if left.is_empty() {
            return;
        }
        let mut keep = Vec::new();

        while !left.is_empty() {
            let device = left[0].device.clone();
            let queue = left[0].queue.clone();
            let (batch, rest): (Vec<Parked>, Vec<Parked>) =
                left.into_iter().partition(|p| p.device == device);
            left = rest;

            let mut enc = device.create_command_encoder(&Default::default());
            let staging: Vec<wgpu::Buffer> = batch
                .iter()
                .map(|p| {
                    let bytes = p.resolve.size();
                    let s = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("nv-kernels-profile-drain"),
                        size: bytes,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    enc.copy_buffer_to_buffer(&p.resolve, 0, &s, 0, bytes);
                    s
                })
                .collect();
            queue.submit([enc.finish()]);
            let waits: Vec<_> = staging
                .iter()
                .map(|s| {
                    let (tx, rx) = std::sync::mpsc::channel();
                    s.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                        let _ = tx.send(r);
                    });
                    rx
                })
                .collect();
            let polled = device.poll(wgpu::PollType::wait_indefinitely()).is_ok();

            for (i, mut p) in batch.into_iter().enumerate() {
                let ts = if polled && matches!(waits[i].recv(), Ok(Ok(()))) {
                    let out = staging[i]
                        .slice(..)
                        .get_mapped_range()
                        .ok()
                        .map(|v| bytemuck::cast_slice::<u8, u64>(&v).to_vec());
                    staging[i].unmap();
                    out
                } else {
                    None
                };

                match ts.filter(|t| t.iter().any(|&x| x != 0)) {
                    Some(ts) => {
                        for (j, label) in p.labels.iter().enumerate() {
                            let w = (j / p.per_chunk) * super::PROFILE_CHUNK_QUERIES as usize
                                + (j % p.per_chunk) * 2;
                            let ns = match (ts.get(w), ts.get(w + 1)) {
                                (Some(a), Some(b)) => b.saturating_sub(*a) as f64 * p.period,
                                _ => 0.0,
                            };
                            record(label, ns);
                        }
                    }
                    None => {
                        p.tries += 1;
                        if p.tries >= MAX_TRIES {
                            LOST.fetch_add(p.labels.len() as u64, Ordering::Relaxed);
                        } else {
                            keep.push(p);
                        }
                    }
                }
            }
        }
        if let Ok(mut v) = pend().lock() {
            keep.append(&mut v);
            *v = keep;
        }
    }

    pub fn enabled() -> bool {
        *ON.get_or_init(|| std::env::var("NV_WGPU_PROFILE").as_deref() == Ok("1"))
    }

    pub fn record(label: &str, ns: f64) {
        if let Ok(mut m) = acc().lock() {
            let e = m.entry(label.to_string()).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ns;
        }
    }

    pub fn reset() {
        if let Ok(mut v) = pend().lock() {
            v.clear();
        }
        LOST.store(0, Ordering::Relaxed);
        OVER.store(0, Ordering::Relaxed);
        if let Ok(mut m) = acc().lock() {
            m.clear();
        }
    }

    pub fn report() -> Vec<(String, u64, f64)> {
        flush();
        let Ok(m) = acc().lock() else {
            return Vec::new();
        };
        let mut v: Vec<(String, u64, f64)> =
            m.iter().map(|(k, (c, n))| (k.clone(), *c, *n)).collect();
        v.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    pub fn total_ns() -> f64 {
        flush();
        acc()
            .lock()
            .map(|m| m.values().map(|(_, n)| *n).sum::<f64>() + 0.0)
            .unwrap_or(0.0)
    }

    pub fn dispatches() -> u64 {
        flush();
        acc()
            .lock()
            .map(|m| m.values().map(|(c, _)| *c).sum())
            .unwrap_or(0)
    }

    pub fn table(tag: &str, expect_per_step: usize) -> String {
        use std::fmt::Write;
        let rows = report();
        let n: u64 = rows.iter().map(|r| r.1).sum();
        let total: f64 = rows.iter().map(|r| r.2).sum();
        let mut s = String::new();
        let _ = writeln!(
            s,
            "[NV_WGPU_PROFILE] {tag}: {} labels, {n} dispatches, {:.3} ms GPU",
            rows.len(),
            total / 1e6
        );
        if expect_per_step == 0 {
            let _ = writeln!(s, "  (no expected passes/step supplied; gate skipped)");
        } else if n == 0 {
            let _ = writeln!(
                s,
                "  UNTRUSTWORTHY: 0 dispatches observed but the graph runs {expect_per_step}/step. \
                 Nothing on this path goes through encode_pass_list, submit_passes, Chain::submit \
                 or Recorded::replay, so the profiler is blind. Do not report per-kernel numbers."
            );
        } else if !n.is_multiple_of(expect_per_step as u64) {
            let _ = writeln!(
                s,
                "  UNTRUSTWORTHY: {n} dispatches is not a multiple of {expect_per_step}/step; \
                 part of the graph bypasses the instrument."
            );
        } else {
            let steps = n / expect_per_step as u64;
            let _ = writeln!(
                s,
                "  gate ok: {steps} steps x {expect_per_step} passes, {:.3} ms GPU/step \
                 (profiled: pass-split inflated, ratios only)",
                total / 1e6 / steps as f64
            );
        }
        let over = over_ceiling();
        if over > 0 {
            let _ = writeln!(
                s,
                "  UNTRUSTWORTHY: {over} dispatches went out uninstrumented because their command \
                 buffer held more than {} timestamped passes; split the step into several submits \
                 to profile it.",
                super::PROFILE_MAX_PASSES
            );
        }
        let lost = lost();
        if lost > 0 {
            let _ = writeln!(
                s,
                "  UNTRUSTWORTHY: {lost} dispatches were parked but never became readable; \
                 their command buffers were encoded and dropped unsubmitted."
            );
        }
        let parked = pending();
        if parked > 0 {
            let _ = writeln!(
                s,
                "  note: {parked} resolve(s) still parked (a pre-encoded command buffer awaiting \
                 submit); those dispatches are not in the totals above."
            );
        }
        for (label, count, ns) in &rows {
            let share = if total > 0.0 { ns / total * 100.0 } else { 0.0 };
            let each = ns / (*count).max(1) as f64 / 1e3;
            let _ = writeln!(
                s,
                "  {label:<40} n={count:<6} {:>10.3} ms {share:>6.2}%  {each:>9.1} us/disp",
                ns / 1e6
            );
        }
        s
    }
}

pub type PassRef<'a> = (
    &'a wgpu::ComputePipeline,
    &'a wgpu::BindGroup,
    (u32, u32, u32),
);

pub fn submit_passes(ctx: &WgpuContext, passes: &[PassRef<'_>], labels: &[&str]) -> Result<()> {
    if passes.is_empty() {
        return Ok(());
    }
    if profile::enabled() && ctx.caps.timestamp_query {
        return submit_profiled(ctx, passes, labels);
    }
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        for (pipeline, group, wg) in passes {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, *group, &[]);
            pass.dispatch_workgroups(wg.0, wg.1, wg.2);
        }
    }
    ctx.queue.submit([enc.finish()]);
    Ok(())
}

const PROFILE_CHUNK_QUERIES: u32 = wgpu::QUERY_SET_MAX_QUERIES;

const PROFILE_MAX_PASSES: usize = 1792;

pub fn encode_pass_list<'a, I>(ctx: &WgpuContext, passes: I) -> wgpu::CommandBuffer
where
    I: IntoIterator<Item = PassRef<'a>>,
    I::IntoIter: ExactSizeIterator,
{
    encode_pass_list_labeled(ctx, passes, &[])
}

pub fn encode_pass_list_labeled<'a, I>(
    ctx: &WgpuContext,
    passes: I,
    labels: &[&str],
) -> wgpu::CommandBuffer
where
    I: IntoIterator<Item = PassRef<'a>>,
    I::IntoIter: ExactSizeIterator,
{
    let passes = passes.into_iter();
    let n = passes.len();
    let over = n > PROFILE_MAX_PASSES && profile::enabled() && ctx.caps.timestamp_query;
    if over {
        profile::record_over_ceiling(n);
    }
    if n == 0 || over || !(profile::enabled() && ctx.caps.timestamp_query) {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for (pipeline, group, wg) in passes {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(wg.0, wg.1, wg.2);
            }
        }
        return enc.finish();
    }

    profile::autoflush();
    let per_chunk = (PROFILE_CHUNK_QUERIES / 2) as usize;
    let n_chunks = n.div_ceil(per_chunk);
    let chunk_bytes = PROFILE_CHUNK_QUERIES as u64 * 8;
    let sets: Vec<wgpu::QuerySet> = (0..n_chunks)
        .map(|c| {
            let count = ((n - c * per_chunk).min(per_chunk) * 2) as u32;
            ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("nv-kernels-profile"),
                ty: wgpu::QueryType::Timestamp,
                count,
            })
        })
        .collect();
    let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nv-kernels-profile-resolve"),
        size: n_chunks as u64 * chunk_bytes,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut names = Vec::with_capacity(n);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    for (i, (pipeline, group, wg)) in passes.enumerate() {
        let local = (i % per_chunk) as u32;
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &sets[i / per_chunk],
                    beginning_of_pass_write_index: Some(local * 2),
                    end_of_pass_write_index: Some(local * 2 + 1),
                }),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(wg.0, wg.1, wg.2);
        }
        names.push(match labels.get(i) {
            Some(l) => (*l).to_string(),
            None => profile::pipeline_name(pipeline).unwrap_or_else(|| format!("#{i:04}")),
        });
    }
    for (c, qs) in sets.iter().enumerate() {
        let count = ((n - c * per_chunk).min(per_chunk) * 2) as u32;
        enc.resolve_query_set(qs, 0..count, &resolve, c as u64 * chunk_bytes);
    }
    let cb = enc.finish();

    profile::park(profile::Parked {
        device: ctx.device.clone(),
        queue: ctx.queue.clone(),
        resolve,
        sets,
        labels: names,
        per_chunk,
        period: ctx.queue.get_timestamp_period() as f64,
        tries: 0,
    });
    cb
}

pub fn submit_pass_list<'a, I>(ctx: &WgpuContext, passes: I)
where
    I: IntoIterator<Item = PassRef<'a>>,
    I::IntoIter: ExactSizeIterator,
{
    ctx.queue.submit([encode_pass_list(ctx, passes)]);
}

fn as_refs<'a>(passes: &'a [Pass]) -> Vec<PassRef<'a>> {
    passes
        .iter()
        .map(|(p, g, wg)| (p.as_ref(), g, *wg))
        .collect()
}

fn submit_profiled(ctx: &WgpuContext, passes: &[PassRef<'_>], labels: &[&str]) -> Result<()> {
    if passes.len() > PROFILE_MAX_PASSES {
        for (c, chunk) in passes.chunks(PROFILE_MAX_PASSES).enumerate() {
            let base = c * PROFILE_MAX_PASSES;
            let sub: Vec<&str> = labels
                .iter()
                .skip(base)
                .take(chunk.len())
                .copied()
                .collect();
            submit_profiled_chunk(ctx, chunk, &sub, base)?;
        }
        return Ok(());
    }
    submit_profiled_chunk(ctx, passes, labels, 0)
}

pub fn submit_profiled_slices(
    ctx: &WgpuContext,
    passes: &[PassRef<'_>],
    labels: &[String],
) -> Result<()> {
    let borrowed: Vec<&str> = labels.iter().map(String::as_str).collect();
    submit_profiled(ctx, passes, &borrowed)
}

fn submit_profiled_chunk(
    ctx: &WgpuContext,
    passes: &[PassRef<'_>],
    labels: &[&str],
    base: usize,
) -> Result<()> {
    let n = passes.len();
    let queries = (n * 2) as u32;
    let bytes = (n * 2 * 8) as u64;
    let qs = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("nv-kernels-profile"),
        ty: wgpu::QueryType::Timestamp,
        count: queries,
    });
    let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nv-kernels-profile-resolve"),
        size: bytes,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nv-kernels-profile-staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut enc = ctx.device.create_command_encoder(&Default::default());
    for (i, (pipeline, group, wg)) in passes.iter().enumerate() {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &qs,
                beginning_of_pass_write_index: Some((i * 2) as u32),
                end_of_pass_write_index: Some((i * 2 + 1) as u32),
            }),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, *group, &[]);
        pass.dispatch_workgroups(wg.0, wg.1, wg.2);
    }
    enc.resolve_query_set(&qs, 0..queries, &resolve, 0);
    enc.copy_buffer_to_buffer(&resolve, 0, &staging, 0, bytes);
    ctx.queue.submit([enc.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.poll_blocking()?;
    rx.recv()
        .map_err(|e| WgpuError::Readback(format!("profile map callback: {e}")))?
        .map_err(|e| WgpuError::Readback(format!("profile map: {e}")))?;
    let view = slice
        .get_mapped_range()
        .map_err(|e| WgpuError::Readback(format!("profile mapped range: {e}")))?;
    let ts = bytemuck::cast_slice::<u8, u64>(&view);
    let period = ctx.queue.get_timestamp_period() as f64;
    for i in 0..n {
        let ns = ts[i * 2 + 1].saturating_sub(ts[i * 2]) as f64 * period;
        match labels.get(i) {
            Some(l) => profile::record(l, ns),
            None => profile::record(&format!("#{:04}", base + i), ns),
        }
    }
    drop(view);
    staging.unmap();
    Ok(())
}

pub struct Chain<'a> {
    ctx: &'a WgpuContext,
    passes: Vec<Pass>,
    labels: Vec<String>,
}

impl<'a> Chain<'a> {
    pub fn new(ctx: &'a WgpuContext) -> Self {
        Self {
            ctx,
            passes: Vec::new(),
            labels: Vec::new(),
        }
    }

    pub fn push(
        &mut self,
        label: &str,
        source: &str,
        entry: &str,
        bindings: &[(u32, &dyn GpuBind)],
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        let pipeline = cached_compute_pipeline(self.ctx, label, source, entry)?;
        let raw: Vec<(u32, &wgpu::Buffer)> = bindings
            .iter()
            .map(|(slot, handle)| (*slot, handle.bind_buffer()))
            .collect();
        let group = bind_group(self.ctx, &pipeline, &raw);
        self.passes.push((pipeline, group, workgroups));
        if profile::enabled() {
            self.labels.push(format!("{label}:{entry}"));
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.passes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    pub fn submit(&mut self) -> Result<()> {
        if self.passes.is_empty() {
            return Ok(());
        }
        let scope = self
            .ctx
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        if profile::enabled() && self.ctx.caps.timestamp_query {
            let labels: Vec<&str> = self.labels.iter().map(String::as_str).collect();
            submit_profiled(self.ctx, &as_refs(&self.passes), &labels)?;
        } else {
            let mut enc = self.ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for (pipeline, group, wg) in &self.passes {
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(wg.0, wg.1, wg.2);
                }
            }
            self.ctx.queue.submit([enc.finish()]);
        }
        self.passes.clear();
        self.labels.clear();
        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(WgpuError::ShaderCompile(format!("chain submit: {err}")));
        }
        Ok(())
    }
}

pub struct Recorded {
    passes: Vec<Pass>,
    labels: Vec<String>,
    validated: bool,
}

impl Recorded {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            labels: Vec::new(),
            validated: false,
        }
    }

    pub fn push(
        &mut self,
        ctx: &WgpuContext,
        label: &str,
        source: &str,
        entry: &str,
        bindings: &[(u32, &dyn GpuBind)],
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        let pipeline = cached_compute_pipeline(ctx, label, source, entry)?;
        let raw: Vec<(u32, &wgpu::Buffer)> = bindings
            .iter()
            .map(|(slot, handle)| (*slot, handle.bind_buffer()))
            .collect();
        let group = bind_group(ctx, &pipeline, &raw);
        self.passes.push((pipeline, group, workgroups));
        if profile::enabled() {
            self.labels.push(format!("{label}:{entry}"));
        }
        Ok(())
    }

    pub fn push_raw(
        &mut self,
        ctx: &WgpuContext,
        pipeline: std::sync::Arc<wgpu::ComputePipeline>,
        bindings: &[(u32, &wgpu::Buffer)],
        workgroups: (u32, u32, u32),
    ) {
        let group = bind_group(ctx, &pipeline, bindings);
        if profile::enabled() {
            let idx = self.passes.len();
            let name = profile::pipeline_name(&pipeline).unwrap_or_else(|| format!("#{idx:04}"));
            self.labels.push(name);
        }
        self.passes.push((pipeline, group, workgroups));
    }

    pub fn len(&self) -> usize {
        self.passes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    pub fn replay(&mut self, ctx: &WgpuContext) -> Result<()> {
        if self.passes.is_empty() {
            return Ok(());
        }
        let scope = if self.validated {
            None
        } else {
            Some(ctx.device.push_error_scope(wgpu::ErrorFilter::Validation))
        };
        if profile::enabled() && ctx.caps.timestamp_query {
            let labels: Vec<&str> = self.labels.iter().map(String::as_str).collect();
            submit_profiled(ctx, &as_refs(&self.passes), &labels)?;
        } else {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for (pipeline, group, wg) in &self.passes {
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(wg.0, wg.1, wg.2);
                }
            }
            ctx.queue.submit([enc.finish()]);
        }
        if let Some(scope) = scope {
            if let Some(err) = pollster::block_on(scope.pop()) {
                return Err(WgpuError::ShaderCompile(format!("recorded replay: {err}")));
            }
            self.validated = true;
        }
        Ok(())
    }

    pub fn replay_n(&mut self, ctx: &WgpuContext, n: usize) -> Result<()> {
        if self.passes.is_empty() || n == 0 {
            return Ok(());
        }
        if profile::enabled() && ctx.caps.timestamp_query {
            for _ in 0..n {
                self.replay(ctx)?;
            }
            return Ok(());
        }
        let scope = if self.validated {
            None
        } else {
            Some(ctx.device.push_error_scope(wgpu::ErrorFilter::Validation))
        };
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for _ in 0..n {
                for (pipeline, group, wg) in &self.passes {
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(wg.0, wg.1, wg.2);
                }
            }
        }
        ctx.queue.submit([enc.finish()]);
        if let Some(scope) = scope {
            if let Some(err) = pollster::block_on(scope.pop()) {
                return Err(WgpuError::ShaderCompile(format!(
                    "recorded replay_n: {err}"
                )));
            }
            self.validated = true;
        }
        Ok(())
    }
}

impl Default for Recorded {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Chain<'a> {
    pub fn into_recorded(self) -> Recorded {
        Recorded {
            passes: self.passes,
            labels: self.labels,
            validated: false,
        }
    }
}

pub fn read_back<T: bytemuck::Pod>(
    ctx: &WgpuContext,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Result<Vec<T>> {
    read_back_at(ctx, buffer, 0, len)
}

pub fn read_back_at<T: bytemuck::Pod>(
    ctx: &WgpuContext,
    buffer: &wgpu::Buffer,
    offset: usize,
    len: usize,
) -> Result<Vec<T>> {
    let elem = std::mem::size_of::<T>() as u64;
    let bytes = len as u64 * elem;
    if bytes == 0 {
        return Ok(Vec::new());
    }
    let start = offset as u64 * elem;
    if !start.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
        || !bytes.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
        || start + bytes > buffer.size()
    {
        return Err(WgpuError::Shape(format!(
            "read_back_at: range {start}..{} misaligned or outside buffer of {} bytes",
            start + bytes,
            buffer.size()
        )));
    }
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nv-kernels-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buffer, start, &staging, 0, bytes);
    ctx.queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.poll_blocking()?;
    rx.recv()
        .map_err(|e| WgpuError::Readback(format!("map callback: {e}")))?
        .map_err(|e| WgpuError::Readback(format!("map: {e}")))?;
    let view = slice
        .get_mapped_range()
        .map_err(|e| WgpuError::Readback(format!("mapped range: {e}")))?;
    let out = bytemuck::cast_slice::<u8, T>(&view).to_vec();
    drop(view);
    staging.unmap();
    Ok(out)
}

pub fn workgroup_count_1d(
    ctx: &WgpuContext,
    invocations: u64,
    workgroup_size: u32,
) -> (u32, u32, u32) {
    let wg = workgroup_size.max(1) as u64;
    let groups = invocations.div_ceil(wg).max(1);
    let limit = ctx.caps.max_compute_workgroups_per_dimension.max(1) as u64;
    if groups <= limit {
        return (groups as u32, 1, 1);
    }
    let y = groups.div_ceil(limit);
    (limit as u32, y as u32, 1)
}

pub fn require_workgroup(ctx: &WgpuContext, what: &str, wg: u32) -> Result<()> {
    if ctx.caps.max_compute_invocations_per_workgroup < wg
        || ctx.caps.max_compute_workgroup_size_x < wg
    {
        return Err(WgpuError::Unsupported(format!(
            "{what} needs a {wg}-invocation workgroup; device allows {} (x max {})",
            ctx.caps.max_compute_invocations_per_workgroup, ctx.caps.max_compute_workgroup_size_x
        )));
    }
    Ok(())
}

pub fn require_workgroup_and_scratch(
    ctx: &WgpuContext,
    what: &str,
    wg: u32,
    scratch_bytes: u32,
) -> Result<()> {
    require_workgroup(ctx, what, wg)?;
    if !ctx.caps.workgroup_storage_fits(scratch_bytes) {
        return Err(WgpuError::Unsupported(format!(
            "{what} scratch needs {scratch_bytes} bytes of workgroup storage; device allows {}",
            ctx.caps.max_compute_workgroup_storage_size
        )));
    }
    Ok(())
}

pub fn check_len(what: &str, got: usize, want: usize) -> Result<()> {
    if got != want {
        return Err(WgpuError::Shape(format!("{what}: got {got} want {want}")));
    }
    Ok(())
}
