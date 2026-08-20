use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{dispatch, Result, WgpuError};

pub const ENTRY: &str = "main";

pub const SUBGROUP: u32 = 32;

pub const ACC_TILE_M: u32 = 16;

pub const ACC_TILE_N: u32 = 16;

pub const SHARED_STRIDE_PAD_F16: u32 = 8;

pub const GLSLC_LINE_THE_BLOBS_WERE_BUILT_WITH_NO_DASH_O_MATCHES_THE_PROVEN_PROBE_BLOB: &str =
    "nix-shell -p shaderc --run 'glslc -fshader-stage=comp --target-env=vulkan1.3 \
     -D<defines> gemm_mulmm_coop.comp -o <name>.spv'; defines are WQ_FP8, OUT_BF16, and \
     BM/BN/BK/WM/WN as encoded in each blob filename; coopmat_16x8x16_probe.spv reproduces \
     byte-for-byte under exactly these flags and diverges under -O";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpvWq {
    Nvfp4Block16,
    Fp8RowscalePlain,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpvOut {
    F32,
    Bf16Alpha,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Blocking {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub wm: u32,
    pub wn: u32,
}

impl Blocking {
    pub const fn subgroups(&self) -> u32 {
        (self.bm / self.wm) * (self.bn / self.wn)
    }

    pub const fn workgroup_size(&self) -> u32 {
        self.subgroups() * SUBGROUP
    }

    pub const fn stage_bytes(&self, out: SpvOut) -> u32 {
        let f16 = (self.bm + self.bn) * (self.bk + SHARED_STRIDE_PAD_F16) * 2;
        match out {
            SpvOut::F32 => f16,
            SpvOut::Bf16Alpha => f16 + self.subgroups() * ACC_TILE_M * ACC_TILE_N * 4,
        }
    }

    pub const fn acc_lane_dwords(&self) -> u32 {
        (self.wm / ACC_TILE_M) * (self.wn / ACC_TILE_N) * ACC_TILE_M * ACC_TILE_N / SUBGROUP
    }
}

pub struct SpvBlob {
    pub wq: SpvWq,
    pub out: SpvOut,
    pub blocking: Blocking,
    pub name: &'static str,
    pub bytes: &'static [u8],
}

macro_rules! blob {
    ($wq:expr, $out:expr, $bm:literal, $bn:literal, $bk:literal, $wm:literal, $wn:literal, $name:literal) => {
        SpvBlob {
            wq: $wq,
            out: $out,
            blocking: Blocking {
                bm: $bm,
                bn: $bn,
                bk: $bk,
                wm: $wm,
                wn: $wn,
            },
            name: $name,
            bytes: include_bytes!(concat!("../../../tests/fixtures/spirv/", $name, ".spv")),
        }
    };
}

pub fn blobs() -> &'static [SpvBlob] {
    static TABLE: &[SpvBlob] = &[
        blob!(
            SpvWq::Nvfp4Block16,
            SpvOut::F32,
            64,
            64,
            32,
            32,
            32,
            "mulmm_w4a16_f32_bm64_bn64_bk32_wm32_wn32"
        ),
        blob!(
            SpvWq::Nvfp4Block16,
            SpvOut::F32,
            128,
            64,
            32,
            64,
            32,
            "mulmm_w4a16_f32_bm128_bn64_bk32_wm64_wn32"
        ),
        blob!(
            SpvWq::Nvfp4Block16,
            SpvOut::F32,
            64,
            128,
            32,
            32,
            64,
            "mulmm_w4a16_f32_bm64_bn128_bk32_wm32_wn64"
        ),
        blob!(
            SpvWq::Nvfp4Block16,
            SpvOut::F32,
            128,
            128,
            32,
            64,
            32,
            "mulmm_w4a16_f32_bm128_bn128_bk32_wm64_wn32"
        ),
        blob!(
            SpvWq::Nvfp4Block16,
            SpvOut::F32,
            128,
            128,
            32,
            64,
            64,
            "mulmm_w4a16_f32_bm128_bn128_bk32_wm64_wn64"
        ),
        blob!(
            SpvWq::Nvfp4Block16,
            SpvOut::F32,
            64,
            64,
            64,
            32,
            32,
            "mulmm_w4a16_f32_bm64_bn64_bk64_wm32_wn32"
        ),
        blob!(
            SpvWq::Nvfp4Block16,
            SpvOut::F32,
            256,
            64,
            32,
            64,
            32,
            "mulmm_w4a16_f32_bm256_bn64_bk32_wm64_wn32"
        ),
        blob!(
            SpvWq::Nvfp4Block16,
            SpvOut::F32,
            128,
            64,
            64,
            64,
            32,
            "mulmm_w4a16_f32_bm128_bn64_bk64_wm64_wn32"
        ),
        blob!(
            SpvWq::Nvfp4Block16,
            SpvOut::F32,
            128,
            64,
            32,
            32,
            32,
            "mulmm_w4a16_f32_bm128_bn64_bk32_wm32_wn32"
        ),
        blob!(
            SpvWq::Nvfp4Block16,
            SpvOut::F32,
            128,
            32,
            32,
            64,
            32,
            "mulmm_w4a16_f32_bm128_bn32_bk32_wm64_wn32"
        ),
        blob!(
            SpvWq::Nvfp4Block16,
            SpvOut::F32,
            64,
            64,
            32,
            64,
            32,
            "mulmm_w4a16_f32_bm64_bn64_bk32_wm64_wn32"
        ),
        blob!(
            SpvWq::Fp8RowscalePlain,
            SpvOut::F32,
            128,
            64,
            32,
            64,
            32,
            "mulmm_w8a16_f32_bm128_bn64_bk32_wm64_wn32"
        ),
        blob!(
            SpvWq::Nvfp4Block16,
            SpvOut::Bf16Alpha,
            128,
            64,
            32,
            64,
            32,
            "mulmm_w4a16_y16_bm128_bn64_bk32_wm64_wn32"
        ),
        blob!(
            SpvWq::Fp8RowscalePlain,
            SpvOut::Bf16Alpha,
            128,
            64,
            32,
            64,
            32,
            "mulmm_w8a16_y16_bm128_bn64_bk32_wm64_wn32"
        ),
        blob!(
            SpvWq::Nvfp4Block16,
            SpvOut::Bf16Alpha,
            64,
            64,
            32,
            32,
            32,
            "mulmm_w4a16_y16_bm64_bn64_bk32_wm32_wn32"
        ),
        blob!(
            SpvWq::Fp8RowscalePlain,
            SpvOut::F32,
            64,
            64,
            32,
            32,
            32,
            "mulmm_w8a16_f32_bm64_bn64_bk32_wm32_wn32"
        ),
        blob!(
            SpvWq::Fp8RowscalePlain,
            SpvOut::Bf16Alpha,
            64,
            64,
            32,
            32,
            32,
            "mulmm_w8a16_y16_bm64_bn64_bk32_wm32_wn32"
        ),
    ];
    TABLE
}

pub const DEFAULT_BLOCKING_WON_THE_QWEN_RATE_SWEEP: &str =
    "bm128 bn64 bk32 wm64 wn32 holds the best or within-noise cell on the gate_up shape at every \
     probed M in {128,256,512} for both weight formats and on every shape at M=512 \
     (spirv_mulmm_rate_at_qwen_shapes); the wide-K down shape at M<=256 and gate_up at M=64 \
     prefer bm64_bn64 wm32_wn32 because a 64-row block doubles the workgroup count on shapes \
     with few N blocks (current numbers: perf/runs.jsonl) -- consumers picking per-shape use \
     blockings()";

pub const DEFAULT_BLOCKING: Blocking = Blocking {
    bm: 128,
    bn: 64,
    bk: 32,
    wm: 64,
    wn: 32,
};

pub fn blob(wq: SpvWq, out: SpvOut, blocking: Blocking) -> Option<&'static SpvBlob> {
    blobs()
        .iter()
        .find(|b| b.wq == wq && b.out == out && b.blocking == blocking)
}

pub fn default_blob(wq: SpvWq, out: SpvOut) -> &'static SpvBlob {
    blob(wq, out, DEFAULT_BLOCKING).expect("every (wq, out) combination ships DEFAULT_BLOCKING")
}

pub fn blockings(wq: SpvWq, out: SpvOut) -> Vec<Blocking> {
    blobs()
        .iter()
        .filter(|b| b.wq == wq && b.out == out)
        .map(|b| b.blocking)
        .collect()
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpirvGemmParams {
    pub n_rows: u32,
    pub k_elems: u32,
    pub m_rows: u32,
    pub y_stride: u32,
    pub alpha: f32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

pub fn check_shape(m: u32, n: u32, k: u32, b: Blocking) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(WgpuError::Shape("zero extent".to_string()));
    }
    if !m.is_multiple_of(ACC_TILE_M) {
        return Err(WgpuError::Shape(format!(
            "M {m} must be a multiple of {ACC_TILE_M}: the epilogue stores whole \
             {ACC_TILE_M}x{ACC_TILE_N} accumulator tiles and skips any tile crossing row M, so a \
             ragged last row band would be left unwritten"
        )));
    }
    if !n.is_multiple_of(ACC_TILE_N) {
        return Err(WgpuError::Shape(format!(
            "N {n} must be a multiple of {ACC_TILE_N}: the epilogue stores whole \
             {ACC_TILE_M}x{ACC_TILE_N} accumulator tiles and skips any tile crossing column N, so \
             a ragged last column band would be left unwritten"
        )));
    }
    if !k.is_multiple_of(b.bk) {
        return Err(WgpuError::Shape(format!(
            "K {k} must be a multiple of the staged BK {}: the k loop has no ragged tail",
            b.bk
        )));
    }
    Ok(())
}

pub struct SpirvGemm {
    pub pipeline: wgpu::ComputePipeline,
    pub blob: &'static SpvBlob,
}

impl SpirvGemm {
    pub fn grid(&self, m: u32, n: u32) -> (u32, u32, u32) {
        (
            n.div_ceil(self.blob.blocking.bn),
            m.div_ceil(self.blob.blocking.bm),
            1,
        )
    }
}

pub fn preflight(ctx: &WgpuContext) -> Option<String> {
    if !ctx
        .device
        .features()
        .contains(wgpu::Features::PASSTHROUGH_SHADERS)
    {
        return Some("PASSTHROUGH_SHADERS not granted: precompiled SPIR-V is unloadable".into());
    }
    if !ctx.caps.cooperative_matrix {
        return Some("EXPERIMENTAL_COOPERATIVE_MATRIX not granted".into());
    }
    if !ctx.caps.shader_f16 {
        return Some("SHADER_F16 not available".into());
    }
    ctx.caps.subgroup32_reason()
}

fn cache() -> &'static Mutex<HashMap<&'static str, Arc<SpirvGemm>>> {
    static CACHE_KEYED_ON_BLOB_NAME_BECAUSE_THE_PROCESS_HOLDS_ONE_SHARED_DEVICE: OnceLock<
        Mutex<HashMap<&'static str, Arc<SpirvGemm>>>,
    > = OnceLock::new();
    CACHE_KEYED_ON_BLOB_NAME_BECAUSE_THE_PROCESS_HOLDS_ONE_SHARED_DEVICE
        .get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn pipeline(ctx: &WgpuContext, blob: &'static SpvBlob) -> Result<Arc<SpirvGemm>> {
    if let Some(hit) = cache().lock().unwrap().get(blob.name) {
        return Ok(hit.clone());
    }
    if let Some(why) = preflight(ctx) {
        return Err(WgpuError::Unsupported(why));
    }
    let words: Vec<u32> = blob
        .bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = unsafe {
        ctx.device
            .create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some(blob.name),
                entry_points: Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint {
                    name: Cow::Borrowed(ENTRY),
                    workgroup_size: (blob.blocking.workgroup_size(), 1, 1),
                }]),
                spirv: Some(Cow::Owned(words)),
                ..Default::default()
            })
    };
    let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..5)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if i == 3 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage { read_only: i != 2 }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(blob.name),
            entries: &bgl_entries,
        });
    let pl = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(blob.name),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let built = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(blob.name),
            layout: Some(&pl),
            module: &module,
            entry_point: Some(ENTRY),
            compilation_options: Default::default(),
            cache: None,
        });
    if let Some(err) = pollster::block_on(scope.pop()) {
        return Err(WgpuError::ShaderCompile(format!(
            "spirv passthrough {}: {err}",
            blob.name
        )));
    }
    let out = Arc::new(SpirvGemm {
        pipeline: built,
        blob,
    });
    cache().lock().unwrap().insert(blob.name, out.clone());
    Ok(out)
}

pub fn bind(
    ctx: &WgpuContext,
    g: &SpirvGemm,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    params: &wgpu::Buffer,
    sf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    dispatch::bind_group(
        ctx,
        &g.pipeline,
        &[(0, w), (1, x), (2, y), (3, params), (4, sf)],
    )
}

pub fn dispatch(
    ctx: &WgpuContext,
    g: &SpirvGemm,
    bindgroup: &wgpu::BindGroup,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    check_shape(m, n, k, g.blob.blocking)?;
    let grid = g.grid(m, n);
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&g.pipeline);
        pass.set_bind_group(0, bindgroup, &[]);
        pass.dispatch_workgroups(grid.0, grid.1, grid.2);
    }
    ctx.queue.submit([enc.finish()]);
    if let Some(err) = pollster::block_on(scope.pop()) {
        return Err(WgpuError::ShaderCompile(format!(
            "spirv gemm dispatch {}: {err}",
            g.blob.name
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPIRV_MAGIC: u32 = 0x0723_0203;

    #[test]
    fn every_checked_in_blob_is_word_aligned_spirv_with_a_consistent_name() {
        for b in blobs() {
            assert!(
                b.bytes.len() >= 20 && b.bytes.len().is_multiple_of(4),
                "{}: not a SPIR-V module",
                b.name
            );
            let magic = u32::from_le_bytes([b.bytes[0], b.bytes[1], b.bytes[2], b.bytes[3]]);
            assert_eq!(magic, SPIRV_MAGIC, "{}: bad SPIR-V magic", b.name);
            let wq = match b.wq {
                SpvWq::Nvfp4Block16 => "w4a16",
                SpvWq::Fp8RowscalePlain => "w8a16",
            };
            let out = match b.out {
                SpvOut::F32 => "f32",
                SpvOut::Bf16Alpha => "y16",
            };
            let bl = b.blocking;
            assert_eq!(
                b.name,
                format!(
                    "mulmm_{wq}_{out}_bm{}_bn{}_bk{}_wm{}_wn{}",
                    bl.bm, bl.bn, bl.bk, bl.wm, bl.wn
                ),
                "blob name must encode its own compile defines or the blob cannot be rebuilt"
            );
        }
    }

    #[test]
    fn every_blocking_fits_the_hardware_it_names() {
        for b in blobs() {
            let bl = b.blocking;
            assert!(bl.wm.is_multiple_of(ACC_TILE_M) && bl.wm > 0, "{}", b.name);
            assert!(bl.wn.is_multiple_of(ACC_TILE_N) && bl.wn > 0, "{}", b.name);
            assert!(bl.bm.is_multiple_of(bl.wm), "{}", b.name);
            assert!(bl.bn.is_multiple_of(bl.wn), "{}", b.name);
            assert!(bl.bk.is_multiple_of(ACC_TILE_M), "{}", b.name);
            assert!(
                bl.workgroup_size() <= 1024,
                "{}: workgroup over the API ceiling",
                b.name
            );
            assert!(
                bl.stage_bytes(b.out) <= 48 * 1024,
                "{}: staged tiles exceed the 48K workgroup shared budget",
                b.name
            );
            assert!(
                bl.acc_lane_dwords() <= 128,
                "{}: accumulator dwords per lane exceed the budget the WGSL path proved \
                 (ACC_LANE_DWORD_BUDGET)",
                b.name
            );
        }
    }

    #[test]
    fn the_default_blocking_ships_for_every_format_and_output_combination() {
        for wq in [SpvWq::Nvfp4Block16, SpvWq::Fp8RowscalePlain] {
            for out in [SpvOut::F32, SpvOut::Bf16Alpha] {
                assert!(
                    blob(wq, out, DEFAULT_BLOCKING).is_some(),
                    "{wq:?}/{out:?} missing DEFAULT_BLOCKING"
                );
            }
        }
        assert_eq!(DEFAULT_BLOCKING.subgroups(), 4);
        assert_eq!(DEFAULT_BLOCKING.workgroup_size(), 128);
    }

    #[test]
    fn params_layout_matches_the_std140_uniform_block() {
        assert_eq!(std::mem::size_of::<SpirvGemmParams>(), 32);
        assert_eq!(std::mem::offset_of!(SpirvGemmParams, n_rows), 0);
        assert_eq!(std::mem::offset_of!(SpirvGemmParams, k_elems), 4);
        assert_eq!(std::mem::offset_of!(SpirvGemmParams, m_rows), 8);
        assert_eq!(std::mem::offset_of!(SpirvGemmParams, y_stride), 12);
        assert_eq!(std::mem::offset_of!(SpirvGemmParams, alpha), 16);
    }

    #[test]
    fn a_ragged_extent_is_refused_rather_than_left_unwritten() {
        check_shape(32, 128, 256, DEFAULT_BLOCKING).expect("aligned shape");
        for (m, n, k) in [(24u32, 128u32, 256u32), (32, 124, 256), (32, 128, 250)] {
            assert!(
                check_shape(m, n, k, DEFAULT_BLOCKING).is_err(),
                "must refuse M={m} N={n} K={k}"
            );
        }
    }

    #[test]
    fn grid_covers_the_matrix_with_n_on_x_and_m_on_y() {
        let g = DEFAULT_BLOCKING;
        assert_eq!((17408u32.div_ceil(g.bn), 256u32.div_ceil(g.bm)), (272, 2));
    }
}
