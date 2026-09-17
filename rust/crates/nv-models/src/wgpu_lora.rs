use anyhow::Result;
use std::sync::Arc;

use nv_kernels::wgpu_backend::dispatch::{self, GpuTensor, GpuUniform};
use nv_kernels::wgpu_backend::kernels as wk;
use nv_kernels::wgpu_backend::{compose, WgpuContext};

use crate::gemma4_wgpu_shared::{err, pack_pairs};

#[derive(Clone)]
pub(crate) struct Pass {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bind: wgpu::BindGroup,
    pub(crate) grid: (u32, u32, u32),

    pub(crate) bound_bytes: u64,
    pub(crate) widest_bytes: u64,
}

pub(crate) fn bind_bytes<'a>(bufs: impl Iterator<Item = &'a wgpu::Buffer>) -> (u64, u64) {
    let mut sum = 0u64;
    let mut widest = 0u64;
    for b in bufs {
        sum += b.size();
        widest = widest.max(b.size());
    }
    (sum, widest)
}

const LORA_CVT_WGSL: &str = include_str!("../../nv-kernels/wgsl/e4b_lora_cvt.wgsl");

const LORA_WIDEN_ENTRY: &str = "e4b_lora_widen";
const LORA_REPACK_ENTRY: &str = "e4b_lora_repack";

const LORA_PK_TILES_WGSL: &str = include_str!("../../nv-kernels/wgsl/lora_pk_tiles.wgsl");
const LORA_PK_SHRINK_ENTRY: &str = "lora_pk_shrink_tiles";
const LORA_PK_EXPAND_ENTRY: &str = "lora_pk_expand_tiles";
const LORA_PK_N_CHUNK: usize = 1024;
const LORA_PK_K_TILE_512_IS_THE_CUDA_LORA_M1_PROVEN_GEOMETRY: usize = 512;
pub(crate) const LORA_PK_DEST_SLOTS_COVER_THE_WIDEST_Q3D_SITE: usize = 3;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct LoraCvtParams {
    m: u32,
    width: u32,
    pk_row_words: u32,
    wide_row_elems: u32,
    wide_col_off: u32,
    total: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct LoraFusedParams {
    m: u32,
    rank: u32,
    k: u32,
    a_slice_stride: u32,
    a_d0_stride: u32,
    y_row_stride: u32,
    win_off: u32,
    win_len: u32,
    scale: f32,
    off_counts: u32,
    off_start: u32,
    off_active: u32,
    off_slice_n: u32,
    off_slice_start: u32,
    off_b_off: u32,
    off_b_d0: u32,
}

struct LoraMetaLayout {
    data: Vec<i32>,
    off_counts: u32,
    off_start: u32,
    off_active: u32,
    off_slice_n: u32,
    off_slice_start: u32,
    off_b_off: u32,
    off_b_d0: u32,
}

fn build_lora_meta(meta: &wk::lora::LoraMeta, widths: &[usize], rank: usize) -> LoraMetaLayout {
    let mut data: Vec<i32> = Vec::new();
    data.extend_from_slice(&meta.token_indices_sorted);
    let off_counts = data.len() as u32;
    data.extend_from_slice(&meta.num_tokens_per_lora);
    let off_start = data.len() as u32;
    data.extend_from_slice(&meta.lora_token_start_loc);
    let off_active = data.len() as u32;
    data.extend_from_slice(&meta.active_lora_ids);
    let off_slice_n = data.len() as u32;
    for &w in widths {
        data.push(w as i32);
    }
    let off_slice_start = data.len() as u32;
    let mut acc = 0usize;
    for &w in widths {
        data.push(acc as i32);
        acc += w;
    }
    let off_b_off = data.len() as u32;
    let mut b_acc = 0usize;
    for &w in widths {
        data.push(b_acc as i32);
        b_acc += meta.max_loras * w * rank;
    }
    let off_b_d0 = data.len() as u32;
    for &w in widths {
        data.push((w * rank) as i32);
    }
    LoraMetaLayout {
        data,
        off_counts,
        off_start,
        off_active,
        off_slice_n,
        off_slice_start,
        off_b_off,
        off_b_d0,
    }
}

pub(crate) struct LoraPipelines {
    fused: Arc<wgpu::ComputePipeline>,
    widen: Arc<wgpu::ComputePipeline>,
    repack: Arc<wgpu::ComputePipeline>,
    pk_shrink: Arc<wgpu::ComputePipeline>,
    pk_expand: Arc<wgpu::ComputePipeline>,
    idle_dest_slots_stay_distinct_because_writable_bindings_must_not_alias:
        [GpuTensor<u32>; LORA_PK_DEST_SLOTS_COVER_THE_WIDEST_Q3D_SITE - 1],
}

pub(crate) fn build_lora_pipelines(ctx: &WgpuContext) -> Result<LoraPipelines> {
    let mk = |label: &str, source: &str, entry: &str| {
        dispatch::cached_compute_pipeline(ctx, label, source, entry)
            .map_err(|e| anyhow::anyhow!("pipeline {label}: {e}"))
    };
    let src_fused = compose(wk::lora::FUSED_WGSL);
    let src_cvt = compose(LORA_CVT_WGSL);
    let src_pk = compose(LORA_PK_TILES_WGSL);
    Ok(LoraPipelines {
        fused: mk("e4bw-lora-fused", &src_fused, wk::lora::FUSED_ENTRY)?,
        widen: mk("e4bw-lora-widen", &src_cvt, LORA_WIDEN_ENTRY)?,
        repack: mk("e4bw-lora-repack", &src_cvt, LORA_REPACK_ENTRY)?,
        pk_shrink: mk("q3d-lora-pk-shrink", &src_pk, LORA_PK_SHRINK_ENTRY)?,
        pk_expand: mk("q3d-lora-pk-expand", &src_pk, LORA_PK_EXPAND_ENTRY)?,
        idle_dest_slots_stay_distinct_because_writable_bindings_must_not_alias: [
            GpuTensor::<u32>::zeroed(ctx, "q3d-lora-pk-idle1", 2),
            GpuTensor::<u32>::zeroed(ctx, "q3d-lora-pk-idle2", 2),
        ],
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_lora_site(
    ctx: &WgpuContext,
    out: &mut Vec<Pass>,
    keep: &mut Vec<Box<dyn std::any::Any + Send>>,
    pl: &LoraPipelines,
    scratch: &wgpu::Buffer,
    m: usize,
    site: &LoraSiteGpu,
    x: &wgpu::Buffer,
    segs: &[&wgpu::Buffer],
) {
    assert_eq!(
        segs.len(),
        site.segs.len(),
        "lora site: {} destination buffers for {} segments",
        segs.len(),
        site.segs.len()
    );
    let total_w: usize = site.segs.iter().sum();
    fn cvt(
        ctx: &WgpuContext,
        out: &mut Vec<Pass>,
        keep: &mut Vec<Box<dyn std::any::Any + Send>>,
        pipeline: &Arc<wgpu::ComputePipeline>,
        scratch: &wgpu::Buffer,
        m: usize,
        total_w: usize,
        seg_widths: &[usize],
        segs: &[&wgpu::Buffer],
        widen: bool,
    ) {
        let mut off = 0usize;
        for (buf, w) in segs.iter().zip(seg_widths.iter()) {
            let total = if widen { m * w } else { m * w / 2 };
            let p = GpuUniform::new(
                ctx,
                "e4bw-lora-cvt-p",
                &LoraCvtParams {
                    m: m as u32,
                    width: *w as u32,
                    pk_row_words: (w / 2) as u32,
                    wide_row_elems: total_w as u32,
                    wide_col_off: off as u32,
                    total: total as u32,
                    ..Default::default()
                },
            );
            let grid = dispatch::workgroup_count_1d(ctx, total as u64, 256);
            let binds: [(u32, &wgpu::Buffer); 3] = [(0, buf), (1, scratch), (2, p.raw())];
            let bind = dispatch::bind_group(ctx, pipeline, &binds);
            let (bound_bytes, widest_bytes) = bind_bytes(binds.iter().map(|(_, b)| *b));
            out.push(Pass {
                pipeline: pipeline.clone(),
                bind,
                grid,
                bound_bytes,
                widest_bytes,
            });
            keep.push(Box::new(p));
            off += w;
        }
    }
    cvt(
        ctx, out, keep, &pl.widen, scratch, m, total_w, &site.segs, segs, true,
    );

    let meta = wk::lora::LoraMeta::prepare(&vec![0i32; m], 1);
    let ml = build_lora_meta(&meta, &site.widths, site.rank);
    let meta_buf = GpuTensor::<i32>::upload(ctx, "e4bw-lora-meta", &ml.data);
    let params = GpuUniform::new(
        ctx,
        "e4bw-lora-fused-p",
        &LoraFusedParams {
            m: m as u32,
            rank: site.rank as u32,
            k: site.k as u32,
            a_slice_stride: (site.rank * site.k) as u32,
            a_d0_stride: (site.rank * site.k) as u32,
            y_row_stride: total_w as u32,
            win_off: 0,
            win_len: total_w as u32,
            scale: 1.0,
            off_counts: ml.off_counts,
            off_start: ml.off_start,
            off_active: ml.off_active,
            off_slice_n: ml.off_slice_n,
            off_slice_start: ml.off_slice_start,
            off_b_off: ml.off_b_off,
            off_b_d0: ml.off_b_d0,
        },
    );
    let max_n = *site.widths.iter().max().expect("lora site has slices");
    let grid = (
        (m * max_n.div_ceil(wk::lora::FUSED_N_CHUNK as usize)) as u32,
        site.widths.len() as u32,
        meta.grid_loras() as u32,
    );
    let limit = ctx.caps.max_compute_workgroups_per_dimension;
    assert!(
        grid.0 <= limit && grid.1 <= limit && grid.2 <= limit,
        "lora fused grid {grid:?} exceeds device limit {limit}"
    );
    let binds: [(u32, &wgpu::Buffer); 6] = [
        (0, x),
        (1, site.a.raw()),
        (2, site.b.raw()),
        (3, scratch),
        (4, meta_buf.raw()),
        (5, params.raw()),
    ];
    let bind = dispatch::bind_group(ctx, &pl.fused, &binds);
    let (bound_bytes, widest_bytes) = bind_bytes(binds.iter().map(|(_, b)| *b));
    out.push(Pass {
        pipeline: pl.fused.clone(),
        bind,
        grid,
        bound_bytes,
        widest_bytes,
    });
    keep.push(Box::new(meta_buf));
    keep.push(Box::new(params));

    cvt(
        ctx, out, keep, &pl.repack, scratch, m, total_w, &site.segs, segs, false,
    );
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct LoraPkParams {
    m: u32,
    rank: u32,
    k: u32,
    a_slice_stride: u32,
    a_d0_stride: u32,
    scale: f32,
    k_tile: u32,
    n_tiles: u32,
    n_slices: u32,
    off_counts: u32,
    off_start: u32,
    off_active: u32,
    off_slice_n: u32,
    off_slice_start: u32,
    off_b_off: u32,
    off_b_d0: u32,
    off_dest_buf: u32,
    off_dest_col: u32,
    off_dest_words: u32,
    pad0: u32,
}

struct LoraPkRouting {
    dest_buf: Vec<i32>,
    dest_col: Vec<i32>,
    dest_words: Vec<i32>,
}

fn route_slices_to_seg_buffers(widths: &[usize], seg_widths: &[usize]) -> LoraPkRouting {
    let seg_starts: Vec<usize> = seg_widths
        .iter()
        .scan(0usize, |acc, w| {
            let s = *acc;
            *acc += w;
            Some(s)
        })
        .collect();
    let mut r = LoraPkRouting {
        dest_buf: Vec::with_capacity(widths.len()),
        dest_col: Vec::with_capacity(widths.len()),
        dest_words: Vec::with_capacity(widths.len()),
    };
    let mut start = 0usize;
    for &w in widths {
        let seg = seg_starts
            .iter()
            .zip(seg_widths.iter())
            .position(|(&s, &sw)| s <= start && start + w <= s + sw)
            .expect("every lora slice lies inside one destination segment");
        let col = start - seg_starts[seg];
        assert!(
            col.is_multiple_of(2),
            "slice starts on an odd packed column {col}; the pk kernel owns whole u32 words"
        );
        r.dest_buf.push(seg as i32);
        r.dest_col.push(col as i32);
        r.dest_words.push((seg_widths[seg] / 2) as i32);
        start += w;
    }
    r
}

pub(crate) fn emit_lora_site_pk(
    ctx: &WgpuContext,
    out: &mut Vec<Pass>,
    keep: &mut Vec<Box<dyn std::any::Any + Send>>,
    pl: &LoraPipelines,
    m: usize,
    site: &LoraSiteGpu,
    x: &wgpu::Buffer,
    segs: &[&wgpu::Buffer],
) {
    assert_eq!(
        segs.len(),
        site.segs.len(),
        "lora pk site: {} destination buffers for {} segments",
        segs.len(),
        site.segs.len()
    );
    assert!(
        segs.len() <= LORA_PK_DEST_SLOTS_COVER_THE_WIDEST_Q3D_SITE,
        "lora pk site carries {} segments; the kernel binds {}",
        segs.len(),
        LORA_PK_DEST_SLOTS_COVER_THE_WIDEST_Q3D_SITE
    );
    let meta = wk::lora::LoraMeta::prepare(&vec![0i32; m], 1);
    let ml = build_lora_meta(&meta, &site.widths, site.rank);
    let routing = route_slices_to_seg_buffers(&site.widths, &site.segs);
    let mut data = ml.data;
    let off_dest_buf = data.len() as u32;
    data.extend_from_slice(&routing.dest_buf);
    let off_dest_col = data.len() as u32;
    data.extend_from_slice(&routing.dest_col);
    let off_dest_words = data.len() as u32;
    data.extend_from_slice(&routing.dest_words);
    let meta_buf = GpuTensor::<i32>::upload(ctx, "q3d-lora-pk-meta", &data);
    let n_slices = site.widths.len();
    let n_tiles = site
        .k
        .div_ceil(LORA_PK_K_TILE_512_IS_THE_CUDA_LORA_M1_PROVEN_GEOMETRY);
    let grid_loras = meta.grid_loras();
    let params = GpuUniform::new(
        ctx,
        "q3d-lora-pk-p",
        &LoraPkParams {
            m: m as u32,
            rank: site.rank as u32,
            k: site.k as u32,
            a_slice_stride: (site.rank * site.k) as u32,
            a_d0_stride: (site.rank * site.k) as u32,
            scale: 1.0,
            k_tile: LORA_PK_K_TILE_512_IS_THE_CUDA_LORA_M1_PROVEN_GEOMETRY as u32,
            n_tiles: n_tiles as u32,
            n_slices: n_slices as u32,
            off_counts: ml.off_counts,
            off_start: ml.off_start,
            off_active: ml.off_active,
            off_slice_n: ml.off_slice_n,
            off_slice_start: ml.off_slice_start,
            off_b_off: ml.off_b_off,
            off_b_d0: ml.off_b_d0,
            off_dest_buf,
            off_dest_col,
            off_dest_words,
            pad0: 0,
        },
    );
    let partials = GpuTensor::<f32>::zeroed(
        ctx,
        "q3d-lora-pk-partials",
        grid_loras * m * n_slices * n_tiles * site.rank,
    );
    let max_n = *site.widths.iter().max().expect("lora site has slices");
    let shrink_grid = (n_tiles as u32, n_slices as u32, (grid_loras * m) as u32);
    let expand_grid = (
        (m * max_n.div_ceil(LORA_PK_N_CHUNK)) as u32,
        n_slices as u32,
        grid_loras as u32,
    );
    let limit = ctx.caps.max_compute_workgroups_per_dimension;
    for grid in [shrink_grid, expand_grid] {
        assert!(
            grid.0 <= limit && grid.1 <= limit && grid.2 <= limit,
            "lora pk grid {grid:?} exceeds device limit {limit}"
        );
    }
    let idle = &pl.idle_dest_slots_stay_distinct_because_writable_bindings_must_not_alias;
    let y1 = segs.get(1).copied().unwrap_or_else(|| idle[0].raw());
    let y2 = segs.get(2).copied().unwrap_or_else(|| idle[1].raw());
    let shrink_binds: [(u32, &wgpu::Buffer); 5] = [
        (0, x),
        (1, site.a.raw()),
        (6, meta_buf.raw()),
        (7, params.raw()),
        (8, partials.raw()),
    ];
    let expand_binds: [(u32, &wgpu::Buffer); 7] = [
        (2, site.b.raw()),
        (3, segs[0]),
        (4, y1),
        (5, y2),
        (6, meta_buf.raw()),
        (7, params.raw()),
        (8, partials.raw()),
    ];
    let shrink_bind = dispatch::bind_group(ctx, &pl.pk_shrink, &shrink_binds);
    let (sb, sw) = bind_bytes(shrink_binds.iter().map(|(_, b)| *b));
    out.push(Pass {
        pipeline: pl.pk_shrink.clone(),
        bind: shrink_bind,
        grid: shrink_grid,
        bound_bytes: sb,
        widest_bytes: sw,
    });
    let expand_bind = dispatch::bind_group(ctx, &pl.pk_expand, &expand_binds);
    let (eb, ew) = bind_bytes(expand_binds.iter().map(|(_, b)| *b));
    out.push(Pass {
        pipeline: pl.pk_expand.clone(),
        bind: expand_bind,
        grid: expand_grid,
        bound_bytes: eb,
        widest_bytes: ew,
    });
    keep.push(Box::new(meta_buf));
    keep.push(Box::new(params));
    keep.push(Box::new(partials));
}

pub(crate) struct LoraShrinkFold {
    pub(crate) partials: wgpu::Buffer,
    pub(crate) n_tiles: usize,
}

pub(crate) fn lora_pk_n_tiles(k: usize) -> usize {
    k.div_ceil(LORA_PK_K_TILE_512_IS_THE_CUDA_LORA_M1_PROVEN_GEOMETRY)
}

pub(crate) fn emit_lora_site_pk_shrink_only(
    ctx: &WgpuContext,
    out: &mut Vec<Pass>,
    keep: &mut Vec<Box<dyn std::any::Any + Send>>,
    pl: &LoraPipelines,
    m: usize,
    site: &LoraSiteGpu,
    x: &wgpu::Buffer,
) -> LoraShrinkFold {
    let meta = wk::lora::LoraMeta::prepare(&vec![0i32; m], 1);
    let ml = build_lora_meta(&meta, &site.widths, site.rank);
    let meta_buf = GpuTensor::<i32>::upload(ctx, "q3d-lora-pk-meta", &ml.data);
    let n_slices = site.widths.len();
    let n_tiles = lora_pk_n_tiles(site.k);
    let grid_loras = meta.grid_loras();
    let params = GpuUniform::new(
        ctx,
        "q3d-lora-pk-p",
        &LoraPkParams {
            m: m as u32,
            rank: site.rank as u32,
            k: site.k as u32,
            a_slice_stride: (site.rank * site.k) as u32,
            a_d0_stride: (site.rank * site.k) as u32,
            scale: 1.0,
            k_tile: LORA_PK_K_TILE_512_IS_THE_CUDA_LORA_M1_PROVEN_GEOMETRY as u32,
            n_tiles: n_tiles as u32,
            n_slices: n_slices as u32,
            off_counts: ml.off_counts,
            off_start: ml.off_start,
            off_active: ml.off_active,
            off_slice_n: ml.off_slice_n,
            off_slice_start: ml.off_slice_start,
            off_b_off: ml.off_b_off,
            off_b_d0: ml.off_b_d0,
            off_dest_buf: 0,
            off_dest_col: 0,
            off_dest_words: 0,
            pad0: 0,
        },
    );
    let partials = GpuTensor::<f32>::zeroed(
        ctx,
        "q3d-lora-pk-partials",
        grid_loras * m * n_slices * n_tiles * site.rank,
    );
    let shrink_grid = (n_tiles as u32, n_slices as u32, (grid_loras * m) as u32);
    let limit = ctx.caps.max_compute_workgroups_per_dimension;
    assert!(
        shrink_grid.0 <= limit && shrink_grid.1 <= limit && shrink_grid.2 <= limit,
        "lora pk shrink grid {shrink_grid:?} exceeds device limit {limit}"
    );
    let shrink_binds: [(u32, &wgpu::Buffer); 5] = [
        (0, x),
        (1, site.a.raw()),
        (6, meta_buf.raw()),
        (7, params.raw()),
        (8, partials.raw()),
    ];
    let shrink_bind = dispatch::bind_group(ctx, &pl.pk_shrink, &shrink_binds);
    let (sb, sw) = bind_bytes(shrink_binds.iter().map(|(_, b)| *b));
    out.push(Pass {
        pipeline: pl.pk_shrink.clone(),
        bind: shrink_bind,
        grid: shrink_grid,
        bound_bytes: sb,
        widest_bytes: sw,
    });
    let fold = LoraShrinkFold {
        partials: partials.raw().clone(),
        n_tiles,
    };
    keep.push(Box::new(meta_buf));
    keep.push(Box::new(params));
    keep.push(Box::new(partials));
    fold
}
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn lora_site_probe(
    x_bf16: &[u16],
    a_bf16: &[u16],
    b_bf16: &[u16],
    y_segments: &mut [Vec<u16>],
    widths: &[usize],
    segs: &[usize],
    m: usize,
    rank: usize,
    k: usize,
) -> Result<()> {
    let ctx = WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu context: {e}"))?;
    let pl = build_lora_pipelines(ctx)?;
    let host = LoraSite {
        a: a_bf16.to_vec(),
        b: b_bf16.to_vec(),
        widths: widths.to_vec(),
        segs: segs.to_vec(),
        k,
        rank,
    };
    let site = upload_lora_site(ctx, "e4bw-lora-probe", &host)?;
    anyhow::ensure!(
        x_bf16.len() == m * k,
        "lora probe x is {} elements, want {}",
        x_bf16.len(),
        m * k
    );
    anyhow::ensure!(y_segments.len() == segs.len(), "lora probe segment count");
    let x = GpuTensor::upload(ctx, "e4bw-lora-probe-x", &pack_pairs(x_bf16));
    let total_w: usize = segs.iter().sum();
    let scratch = GpuTensor::<u32>::zeroed(ctx, "e4bw-lora-probe-scratch", m * total_w);
    let mut ys = Vec::with_capacity(segs.len());
    for (i, w) in segs.iter().enumerate() {
        anyhow::ensure!(
            y_segments[i].len() == m * w,
            "lora probe segment {i} is {} elements, want {}",
            y_segments[i].len(),
            m * w
        );
        ys.push(GpuTensor::upload(
            ctx,
            "e4bw-lora-probe-y",
            &pack_pairs(&y_segments[i]),
        ));
    }
    let mut passes: Vec<Pass> = Vec::new();
    let mut keep: Vec<Box<dyn std::any::Any + Send>> = Vec::new();
    let seg_refs: Vec<&wgpu::Buffer> = ys.iter().map(|t| t.raw()).collect();
    emit_lora_site(
        ctx,
        &mut passes,
        &mut keep,
        &pl,
        scratch.raw(),
        m,
        &site,
        x.raw(),
        &seg_refs,
    );
    let refs: Vec<dispatch::PassRef<'_>> = passes
        .iter()
        .map(|p| (p.pipeline.as_ref(), &p.bind, p.grid))
        .collect();
    dispatch::submit_pass_list(ctx, refs);
    ctx.poll_blocking().map_err(err)?;
    for (i, w) in segs.iter().enumerate() {
        let words = ys[i].download(ctx).map_err(err)?;
        for j in 0..(m * w) {
            let word = words[j / 2];
            y_segments[i][j] = if j.is_multiple_of(2) {
                (word & 0xffff) as u16
            } else {
                (word >> 16) as u16
            };
        }
    }
    Ok(())
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn lora_site_probe_pk(
    x_bf16: &[u16],
    a_bf16: &[u16],
    b_bf16: &[u16],
    y_segments: &mut [Vec<u16>],
    widths: &[usize],
    segs: &[usize],
    m: usize,
    rank: usize,
    k: usize,
) -> Result<()> {
    let ctx = WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu context: {e}"))?;
    let pl = build_lora_pipelines(ctx)?;
    let host = LoraSite {
        a: a_bf16.to_vec(),
        b: b_bf16.to_vec(),
        widths: widths.to_vec(),
        segs: segs.to_vec(),
        k,
        rank,
    };
    let site = upload_lora_site(ctx, "q3d-lora-pk-probe", &host)?;
    anyhow::ensure!(x_bf16.len() == m * k, "pk probe x length");
    anyhow::ensure!(y_segments.len() == segs.len(), "pk probe segment count");
    let x = GpuTensor::upload(ctx, "q3d-lora-pk-probe-x", &pack_pairs(x_bf16));
    let mut ys = Vec::with_capacity(segs.len());
    for (i, w) in segs.iter().enumerate() {
        anyhow::ensure!(y_segments[i].len() == m * w, "pk probe segment {i} length");
        ys.push(GpuTensor::upload(
            ctx,
            "q3d-lora-pk-probe-y",
            &pack_pairs(&y_segments[i]),
        ));
    }
    let mut passes: Vec<Pass> = Vec::new();
    let mut keep: Vec<Box<dyn std::any::Any + Send>> = Vec::new();
    let seg_refs: Vec<&wgpu::Buffer> = ys.iter().map(|t| t.raw()).collect();
    emit_lora_site_pk(
        ctx,
        &mut passes,
        &mut keep,
        &pl,
        m,
        &site,
        x.raw(),
        &seg_refs,
    );
    let refs: Vec<dispatch::PassRef<'_>> = passes
        .iter()
        .map(|p| (p.pipeline.as_ref(), &p.bind, p.grid))
        .collect();
    dispatch::submit_pass_list(ctx, refs);
    ctx.poll_blocking().map_err(err)?;
    for (i, w) in segs.iter().enumerate() {
        let words = ys[i].download(ctx).map_err(err)?;
        for j in 0..(m * w) {
            let word = words[j / 2];
            y_segments[i][j] = if j.is_multiple_of(2) {
                (word & 0xffff) as u16
            } else {
                (word >> 16) as u16
            };
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct LoraSite {
    pub a: Vec<u16>,
    pub b: Vec<u16>,
    pub widths: Vec<usize>,
    pub segs: Vec<usize>,
    pub k: usize,
    pub rank: usize,
}

impl LoraSite {
    pub(crate) fn validate(&self, label: &str) -> Result<()> {
        anyhow::ensure!(!self.widths.is_empty(), "{label}: no lora slices");
        anyhow::ensure!(!self.segs.is_empty(), "{label}: no lora segments");
        anyhow::ensure!(
            self.segs.iter().sum::<usize>() == self.widths.iter().sum::<usize>(),
            "{label}: segment widths {:?} do not cover slice widths {:?}",
            self.segs,
            self.widths
        );
        anyhow::ensure!(self.rank > 0 && self.k > 0, "{label}: zero lora rank/k");
        anyhow::ensure!(
            self.rank <= wk::lora::FUSED_MAX_RANK,
            "{label}: rank {} exceeds kernel limit {}",
            self.rank,
            wk::lora::FUSED_MAX_RANK
        );
        anyhow::ensure!(
            self.a.len() == self.widths.len() * self.rank * self.k,
            "{label}: lora_a is {} elements, want {}",
            self.a.len(),
            self.widths.len() * self.rank * self.k
        );
        let want_b: usize = self.widths.iter().map(|w| w * self.rank).sum();
        anyhow::ensure!(
            self.b.len() == want_b,
            "{label}: lora_b is {} elements, want {want_b}",
            self.b.len()
        );
        anyhow::ensure!(self.k.is_multiple_of(2), "{label}: lora k must be even");
        for w in self.widths.iter().chain(self.segs.iter()) {
            anyhow::ensure!(
                w.is_multiple_of(2),
                "{label}: lora slice width must be even"
            );
        }
        Ok(())
    }
}
pub(crate) fn build_lora_site(
    adapter: &nv_weights::lora_adapter::LoraAdapter,
    rank: usize,
    k: usize,
    slices: &[(&str, usize)],
    segs: &[usize],
) -> Result<(Option<LoraSite>, usize)> {
    use candle_core::{DType, Device};
    use nv_layers::lora_slots::LoraSlotStack;

    let mut matched = 0usize;
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut widths = Vec::with_capacity(slices.len());
    for (name, width) in slices {
        let stack = LoraSlotStack::new(1, rank, k, *width, DType::BF16, &Device::Cpu)
            .map_err(|e| anyhow::anyhow!("lora slot stack for {name}: {e}"))?;
        if let Some(w) = adapter.loras.get(*name) {
            anyhow::ensure!(
                !w.is_embedding,
                "{name}: embedding lora is not supported by the wgpu graph"
            );
            stack
                .set_lora(0, &w.lora_a, &w.lora_b)
                .map_err(|e| anyhow::anyhow!("set lora for {name}: {e}"))?;
            matched += 1;
        }
        a.extend_from_slice(&bf16_bits_of_tensor(stack.lora_a_stacked())?);
        b.extend_from_slice(&bf16_bits_of_tensor(stack.lora_b_stacked())?);
        widths.push(*width);
    }
    if matched == 0 {
        return Ok((None, 0));
    }
    let site = LoraSite {
        a,
        b,
        widths,
        segs: segs.to_vec(),
        k,
        rank,
    };
    site.validate("lora site")?;
    Ok((Some(site), matched))
}

fn bf16_bits_of_tensor(t: &candle_core::Tensor) -> Result<Vec<u16>> {
    let v: Vec<half::bf16> = t.flatten_all()?.to_vec1()?;
    Ok(v.into_iter().map(|x| x.to_bits()).collect())
}

pub(crate) struct LoraSiteGpu {
    pub(crate) a: GpuTensor<u32>,
    pub(crate) b: GpuTensor<u32>,
    pub(crate) widths: Vec<usize>,
    pub(crate) segs: Vec<usize>,
    pub(crate) k: usize,
    pub(crate) rank: usize,
}

pub(crate) fn upload_lora_site(
    ctx: &WgpuContext,
    label: &str,
    site: &LoraSite,
) -> Result<LoraSiteGpu> {
    site.validate(label)?;
    Ok(LoraSiteGpu {
        a: GpuTensor::upload(ctx, label, &pack_pairs(&site.a)),
        b: GpuTensor::upload(ctx, label, &pack_pairs(&site.b)),
        widths: site.widths.clone(),
        segs: site.segs.clone(),
        k: site.k,
        rank: site.rank,
    })
}
