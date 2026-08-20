use std::sync::OnceLock;

use super::qualify::{self, Capabilities, QualStatus};
use super::{Result, WgpuError};

pub struct WgpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub caps: Capabilities,
}

static SG_WIDTH: OnceLock<Option<u32>> = OnceLock::new();

const SG_PROBE_SRC: &str = "@group(0) @binding(0) var<storage, read_write> sgp: array<atomic<u32>, 2>;\n@compute @workgroup_size(256)\nfn probe_subgroup_size(@builtin(subgroup_size) s: u32) {\n    atomicMin(&sgp[0], s);\n    atomicMax(&sgp[1], s);\n}\n";

fn probe_subgroup_width(ctx: &WgpuContext) -> Option<u32> {
    if !ctx.caps.subgroup {
        return None;
    }
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nv-sg-probe"),
            source: wgpu::ShaderSource::Wgsl(SG_PROBE_SRC.into()),
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nv-sg-probe"),
            layout: None,
            module: &module,
            entry_point: Some("probe_subgroup_size"),
            compilation_options: Default::default(),
            cache: None,
        });
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nv-sg-probe"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue
        .write_buffer(&buf, 0, bytemuck::cast_slice(&[u32::MAX, 0u32]));
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nv-sg-probe-read"),
        size: 8,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buf.as_entire_binding(),
        }],
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(4, 1, 1);
    }
    enc.copy_buffer_to_buffer(&buf, 0, &staging, 0, 8);
    ctx.queue.submit([enc.finish()]);
    if pollster::block_on(scope.pop()).is_some() {
        return None;
    }
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    if ctx.poll_blocking().is_err() {
        return None;
    }
    match rx.recv() {
        Ok(Ok(())) => {}
        _ => return None,
    }
    let view = match slice.get_mapped_range() {
        Ok(v) => v,
        Err(_) => return None,
    };
    let words: [u32; 2] = [
        u32::from_le_bytes(view[0..4].try_into().unwrap()),
        u32::from_le_bytes(view[4..8].try_into().unwrap()),
    ];
    drop(view);
    staging.unmap();
    let (mn, mx) = (words[0], words[1]);
    if mn == mx && mn.is_power_of_two() && (4..=128).contains(&mn) {
        Some(mn)
    } else {
        None
    }
}

static SHARED: OnceLock<std::result::Result<WgpuContext, WgpuError>> = OnceLock::new();

fn adapter_score(info: &wgpu::AdapterInfo) -> i32 {
    match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 8,
        wgpu::DeviceType::IntegratedGpu => 4,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 1,
        wgpu::DeviceType::Cpu => 0,
    }
}

fn coop_matrix_opt_out() -> bool {
    matches!(
        std::env::var("NV_KERNELS_WGPU_COOP_MATRIX")
            .ok()
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}

fn requested_backends() -> wgpu::Backends {
    match std::env::var("NV_KERNELS_WGPU_BACKEND")
        .ok()
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("vulkan") | Some("vk") => wgpu::Backends::VULKAN,
        Some("metal") => wgpu::Backends::METAL,
        Some("dx12") | Some("d3d12") => wgpu::Backends::DX12,
        Some("gl") => wgpu::Backends::GL,
        _ => wgpu::Backends::all(),
    }
}

impl WgpuContext {
    pub fn new() -> Result<Self> {
        let backends = requested_backends();
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = backends;
        let instance = wgpu::Instance::new(desc);
        let adapters = pollster::block_on(instance.enumerate_adapters(backends));
        let mut best: Option<(i32, wgpu::Adapter)> = None;
        for a in adapters {
            let s = adapter_score(&a.get_info());
            if best.as_ref().map(|(bs, _)| s > *bs).unwrap_or(true) {
                best = Some((s, a));
            }
        }
        let adapter = match best {
            Some((_, a)) => a,
            None => {
                return Err(WgpuError::NoAdapter(format!(
                    "instance enumerated no adapter for backends {backends:?}; compiled backends {:?}; a Vulkan loader (libvulkan.so.1) must be on the library path",
                    wgpu::Instance::enabled_backend_features()
                )))
            }
        };
        let info = adapter.get_info();
        let available = adapter.features();
        let limits = adapter.limits();
        let downlevel = adapter.get_downlevel_capabilities();
        let coop_configs = qualify::coop_configs(&adapter.cooperative_matrix_properties());
        let mut caps = Capabilities::probe(&info, available, &limits, &downlevel);
        caps.coop_configs = coop_configs;

        if info.backend == wgpu::Backend::Vulkan
            && caps.subgroup
            && caps.subgroup_min_size < caps.subgroup_max_size
            && caps.subgroup_min_size <= qualify::COOP_SUBGROUP_SIZE
            && qualify::COOP_SUBGROUP_SIZE <= caps.subgroup_max_size
            && std::env::var_os("NV_WGPU_REQUIRED_SUBGROUP_SIZE").is_none()
        {
            std::env::set_var(
                "NV_WGPU_REQUIRED_SUBGROUP_SIZE",
                qualify::COOP_SUBGROUP_SIZE.to_string(),
            );
        }

        let mut base = wgpu::Features::empty();
        if caps.shader_f16 {
            base |= wgpu::Features::SHADER_F16;
        }
        if caps.subgroup {
            base |= wgpu::Features::SUBGROUP;
        }
        if caps.timestamp_query {
            base |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if available.contains(wgpu::Features::PASSTHROUGH_SHADERS) {
            base |= wgpu::Features::PASSTHROUGH_SHADERS;
        }
        let want_coop = caps.cooperative_matrix && !coop_matrix_opt_out();
        let coop_bit = wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;

        let experimental = unsafe { wgpu::ExperimentalFeatures::enabled() };
        let mut coop_error = None;
        let mut opened = None;
        if want_coop {
            match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("nv-kernels-wgpu"),
                required_features: base | coop_bit,
                required_limits: limits.clone(),
                experimental_features: experimental,
                ..Default::default()
            })) {
                Ok(pair) => opened = Some(pair),
                Err(e) => coop_error = Some(e.to_string()),
            }
        } else if caps.cooperative_matrix {
            coop_error = Some("NV_KERNELS_WGPU_COOP_MATRIX opted out".to_string());
        }
        let (device, queue) = match opened {
            Some(pair) => pair,
            None => pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("nv-kernels-wgpu"),
                required_features: base,
                required_limits: limits,
                ..Default::default()
            }))
            .map_err(|e| WgpuError::DeviceRequest(format!("{} : {e}", info.name)))?,
        };

        caps.timestamp_query = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        caps.cooperative_matrix = device.features().contains(coop_bit);
        if !caps.cooperative_matrix {
            caps.coop_configs.clear();
            caps.coop_note = coop_error;
        }

        let mut ctx = Self {
            instance,
            adapter,
            device,
            queue,
            info,
            caps,
        };
        ctx.caps.subgroup_runtime_width = ctx.subgroup_width();
        Ok(ctx)
    }

    pub fn subgroup_width(&self) -> Option<u32> {
        *SG_WIDTH.get_or_init(|| probe_subgroup_width(self))
    }

    pub fn shared() -> Result<&'static WgpuContext> {
        match SHARED.get_or_init(Self::new) {
            Ok(ctx) => Ok(ctx),
            Err(e) => Err(e.clone()),
        }
    }

    pub fn qualify(&self) -> QualStatus {
        qualify::qualify(&self.caps)
    }

    pub fn summary(&self) -> String {
        self.caps.summary()
    }

    pub fn poll_blocking(&self) -> Result<()> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|e| WgpuError::Readback(format!("device poll: {e}")))
    }
}

pub fn shared_or_reason() -> std::result::Result<&'static WgpuContext, String> {
    WgpuContext::shared().map_err(|e| e.to_string())
}
