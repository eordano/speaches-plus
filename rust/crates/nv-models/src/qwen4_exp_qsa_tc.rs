use super::{cuda_ptr, qsa_row_attention, slice_ptr, slice_ptr_mut, stream_of, uninit, Qwen4ExpConfig};
use anyhow::{ensure, Result};
use candle_core::Tensor;
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use nv_kernels::cuda as nvk;
use std::ffi::c_void;
use std::sync::Arc;

struct Selection {
    idx: CudaSlice<i32>,
    count: CudaSlice<i32>,
    width: usize,
}

fn select(cfg: &Qwen4ExpConfig, iq: &Tensor, cmp_k: &Tensor, first_position: usize, rows: usize, stream: &Arc<CudaStream>) -> Result<Selection> {
    let r = cfg.indexer_compress_ratio;
    let block_topk = cfg.qsa_block_topk();
    let width = cfg.indexer_budget + r - 1;
    let positions: Vec<i32> = (0..rows).map(|m| (first_position + m) as i32).collect();
    let positions_dev: CudaSlice<i32> = stream.clone_htod(&positions)?;
    let nb: Vec<i32> = positions.iter().map(|&p| ((p + 1) as usize / r) as i32).collect();
    let nb_stride = *nb.iter().max().unwrap() as usize;
    let nb_dev: CudaSlice<i32> = stream.clone_htod(&nb)?;
    let mut scores: CudaSlice<f32> = unsafe { stream.alloc::<f32>(rows * nb_stride.max(1))? };
    let rc = unsafe {
        nvk::q4exp_qsa_scores(
            stream.cu_stream() as *mut c_void,
            cuda_ptr::<bf16>(iq, stream)? as *const u16,
            cuda_ptr::<bf16>(cmp_k, stream)? as *const u16,
            slice_ptr(&nb_dev, stream) as *const i32,
            rows as i32,
            cfg.indexer_n_heads as i32,
            cfg.indexer_head_dim as i32,
            nb_stride as i32,
            1.0 / (cfg.indexer_head_dim as f32).sqrt(),
            slice_ptr_mut(&mut scores, stream) as *mut f32,
        )
    };
    ensure!(rc == 0, "q4exp_qsa_scores rc={rc}");
    let mut blocks: CudaSlice<i32> = unsafe { stream.alloc::<i32>(rows * block_topk)? };
    let mut block_count: CudaSlice<i32> = unsafe { stream.alloc::<i32>(rows)? };
    let rc = unsafe {
        nvk::q4exp_qsa_topk(
            stream.cu_stream() as *mut c_void,
            slice_ptr(&scores, stream) as *const f32,
            slice_ptr(&nb_dev, stream) as *const i32,
            rows as i32,
            block_topk as i32,
            nb_stride as i32,
            slice_ptr_mut(&mut blocks, stream) as *mut i32,
            slice_ptr_mut(&mut block_count, stream) as *mut i32,
        )
    };
    ensure!(rc == 0, "q4exp_qsa_topk rc={rc}");
    let mut idx: CudaSlice<i32> = unsafe { stream.alloc::<i32>(rows * width)? };
    let mut count: CudaSlice<i32> = unsafe { stream.alloc::<i32>(rows)? };
    let rc = unsafe {
        nvk::q4exp_qsa_expand(
            stream.cu_stream() as *mut c_void,
            slice_ptr(&blocks, stream) as *const i32,
            slice_ptr(&block_count, stream) as *const i32,
            slice_ptr(&positions_dev, stream) as *const i32,
            rows as i32,
            r as i32,
            block_topk as i32,
            width as i32,
            slice_ptr_mut(&mut idx, stream) as *mut i32,
            slice_ptr_mut(&mut count, stream) as *mut i32,
        )
    };
    ensure!(rc == 0, "q4exp_qsa_expand rc={rc}");
    Ok(Selection { idx, count, width })
}

pub fn qsa_row_attention_tc(
    cfg: &Qwen4ExpConfig,
    q: &Tensor,
    gate: &Tensor,
    iq: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    cmp_k: &Tensor,
    first_position: usize,
) -> Result<Tensor> {
    let rows = q.dim(0)?;
    if first_position + rows <= cfg.dense_attention_context_limit() {
        return qsa_row_attention(cfg, q, gate, iq, k_cache, v_cache, cmp_k, first_position);
    }
    let nq = cfg.num_attention_heads;
    let hd = cfg.head_dim;
    let stream = stream_of(q.device())?;
    let sel = select(cfg, iq, cmp_k, first_position, rows, &stream)?;
    let out = uninit::<bf16>(q.device(), &[rows, nq * hd])?;
    let rc = unsafe {
        nvk::q4exp_qsa_tc_rows(
            stream.cu_stream() as *mut c_void,
            cuda_ptr::<bf16>(q, &stream)? as *const u16,
            cuda_ptr::<bf16>(gate, &stream)? as *const u16,
            cuda_ptr::<bf16>(k_cache, &stream)? as *const u16,
            cuda_ptr::<bf16>(v_cache, &stream)? as *const u16,
            slice_ptr(&sel.idx, &stream) as *const i32,
            slice_ptr(&sel.count, &stream) as *const i32,
            rows as i32,
            nq as i32,
            cfg.num_key_value_heads as i32,
            hd as i32,
            sel.width as i32,
            1.0 / (hd as f32).sqrt(),
            cuda_ptr::<bf16>(&out, &stream)? as *mut u16,
        )
    };
    ensure!(rc == 0, "q4exp_qsa_tc_rows rc={rc}");
    Ok(out)
}
