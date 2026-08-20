#![cfg(feature = "cuda")]

use std::ffi::c_void;

pub type CudaStreamRaw = *mut c_void;

#[repr(C)]
pub struct FlashFwdParams {
    pub q_ptr: *mut c_void,
    pub k_ptr: *mut c_void,
    pub v_ptr: *mut c_void,
    pub q_batch_stride: i64,
    pub k_batch_stride: i64,
    pub v_batch_stride: i64,
    pub q_row_stride: i64,
    pub k_row_stride: i64,
    pub v_row_stride: i64,
    pub q_head_stride: i64,
    pub k_head_stride: i64,
    pub v_head_stride: i64,
    pub h: i32,
    pub h_k: i32,
    pub h_h_k_ratio: i32,

    pub o_ptr: *mut c_void,
    pub oaccum_ptr: *mut c_void,
    pub o_batch_stride: i64,
    pub o_row_stride: i64,
    pub o_head_stride: i64,
    pub p_ptr: *mut c_void,
    pub softmax_lse_ptr: *mut c_void,
    pub softmax_lseaccum_ptr: *mut c_void,
    pub b: i32,
    pub seqlen_q: i32,
    pub seqlen_k: i32,
    pub seqlen_knew: i32,
    pub d: i32,
    pub seqlen_q_rounded: i32,
    pub seqlen_k_rounded: i32,
    pub d_rounded: i32,
    pub rotary_dim: i32,
    pub total_q: i32,
    pub scale_softmax: f32,
    pub scale_softmax_log2: f32,
    pub cu_seqlens_q: *mut i32,
    pub cu_seqlens_k: *mut i32,
    pub leftpad_k: *mut i32,
    pub seqused_k: *mut i32,
    pub blockmask: *mut i32,
    pub knew_ptr: *mut c_void,
    pub vnew_ptr: *mut c_void,
    pub knew_batch_stride: i64,
    pub vnew_batch_stride: i64,
    pub knew_row_stride: i64,
    pub vnew_row_stride: i64,
    pub knew_head_stride: i64,
    pub vnew_head_stride: i64,
    pub rotary_cos_ptr: *mut c_void,
    pub rotary_sin_ptr: *mut c_void,
    pub cache_batch_idx: *mut i32,
    pub block_table: *mut i32,
    pub block_table_batch_stride: i64,
    pub page_block_size: i32,
    pub p_dropout: f32,
    pub p_dropout_in_uint8_t: u8,
    pub rp_dropout: f32,
    pub scale_softmax_rp_dropout: f32,
    pub window_size_left: i32,
    pub window_size_right: i32,
    pub softcap: f32,
    pub rng_state: *mut u64,
    pub is_bf16: bool,
    pub is_causal: bool,
    pub is_seqlens_k_cumulative: bool,
    pub is_rotary_interleaved: bool,
    pub num_splits: i32,
    pub alibi_slopes_ptr: *mut c_void,
    pub alibi_slopes_batch_stride: i64,
    pub unpadded_lse: bool,
    pub seqlenq_ngroups_swapped: bool,
}

extern "C" {

    #[link_name = "_Z11run_mha_fwdR16Flash_fwd_paramsP11CUstream_st"]
    fn run_mha_fwd(params: *mut FlashFwdParams, stream: CudaStreamRaw);
}

fn round_multiple(x: usize, m: usize) -> usize {
    (x + m - 1) / m * m
}

pub struct VarlenArgs {
    pub q_ptr: u64,
    pub k_ptr: u64,
    pub v_ptr: u64,
    pub o_ptr: u64,
    pub lse_ptr: u64,
    pub cu_seqlens_q: u64,
    pub cu_seqlens_k: u64,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub h: usize,
    pub h_k: usize,
    pub d: usize,
    pub softmax_scale: f32,
    pub window_size_left: Option<usize>,
    pub window_size_right: Option<usize>,
}

pub unsafe fn varlen_fwd_bf16(stream: CudaStreamRaw, a: &VarlenArgs) -> anyhow::Result<()> {
    anyhow::ensure!(a.d % 8 == 0 && a.d <= 256, "fa2 shim: head dim {}", a.d);
    anyhow::ensure!(a.h % a.h_k == 0, "fa2 shim: h {} % h_k {}", a.h, a.h_k);

    let mut window_size_left = a
        .window_size_left
        .filter(|v| *v <= a.max_seqlen_k)
        .map(|v| v as i32)
        .unwrap_or(-1);
    let mut window_size_right = a
        .window_size_right
        .filter(|v| *v <= a.max_seqlen_k)
        .map(|v| v as i32)
        .unwrap_or(-1);
    let is_causal = window_size_left < 0 && window_size_right == 0;
    if window_size_left < 0 && window_size_right >= 0 {
        window_size_left = a.max_seqlen_k as i32;
    }
    if window_size_left >= 0 && window_size_right < 0 {
        window_size_right = a.max_seqlen_k as i32;
    }

    let head_size = round_multiple(a.d, 8);
    let head_size_rounded = round_multiple(head_size, 32);
    let seqlen_q_rounded = round_multiple(a.max_seqlen_q, 128);
    let seqlen_k_rounded = round_multiple(a.max_seqlen_k, 128);

    let mut p: FlashFwdParams = std::mem::zeroed();
    p.q_ptr = a.q_ptr as *mut c_void;
    p.k_ptr = a.k_ptr as *mut c_void;
    p.v_ptr = a.v_ptr as *mut c_void;
    p.o_ptr = a.o_ptr as *mut c_void;
    p.softmax_lse_ptr = a.lse_ptr as *mut c_void;
    p.q_batch_stride = 0;
    p.k_batch_stride = 0;
    p.v_batch_stride = 0;
    p.o_batch_stride = 0;
    p.q_row_stride = (a.h * a.d) as i64;
    p.k_row_stride = (a.h_k * a.d) as i64;
    p.v_row_stride = (a.h_k * a.d) as i64;
    p.o_row_stride = (a.h * a.d) as i64;
    p.q_head_stride = a.d as i64;
    p.k_head_stride = a.d as i64;
    p.v_head_stride = a.d as i64;
    p.o_head_stride = a.d as i64;
    p.b = 1;
    p.h = a.h as i32;
    p.h_k = a.h_k as i32;
    p.h_h_k_ratio = (a.h / a.h_k) as i32;
    p.seqlen_q = a.max_seqlen_q as i32;
    p.seqlen_k = a.max_seqlen_k as i32;
    p.seqlen_q_rounded = seqlen_q_rounded as i32;
    p.seqlen_k_rounded = seqlen_k_rounded as i32;
    p.d = head_size as i32;
    p.d_rounded = head_size_rounded as i32;
    p.scale_softmax = a.softmax_scale;
    p.scale_softmax_log2 = a.softmax_scale * std::f32::consts::LOG2_E;
    p.softcap = 0.0;
    p.p_dropout = 1.0;
    p.p_dropout_in_uint8_t = 255;
    p.rp_dropout = 1.0;
    p.scale_softmax_rp_dropout = p.scale_softmax;
    p.is_bf16 = true;
    p.cu_seqlens_q = a.cu_seqlens_q as *mut i32;
    p.cu_seqlens_k = a.cu_seqlens_k as *mut i32;
    p.is_causal = is_causal;
    p.window_size_left = window_size_left;
    p.window_size_right = window_size_right;
    p.is_seqlens_k_cumulative = true;
    p.num_splits = 1;
    p.unpadded_lse = true;

    run_mha_fwd(&mut p, stream);
    Ok(())
}

pub struct BatchDecodeArgs {
    pub q_ptr: u64,
    pub k_ptr: u64,
    pub v_ptr: u64,
    pub o_ptr: u64,
    pub lse_ptr: u64,
    pub seqused_k: u64,
    pub b: usize,
    pub q_batch_stride: usize,
    pub kv_batch_stride: usize,
    pub max_seqlen_k: usize,
    pub h: usize,
    pub h_k: usize,
    pub d: usize,
    pub softmax_scale: f32,
}

pub unsafe fn batch_decode_fwd_bf16(
    stream: CudaStreamRaw,
    a: &BatchDecodeArgs,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        a.d % 8 == 0 && a.d <= 256,
        "fa2 batch decode: head dim {}",
        a.d
    );
    anyhow::ensure!(
        a.h % a.h_k == 0,
        "fa2 batch decode: h {} % h_k {}",
        a.h,
        a.h_k
    );
    anyhow::ensure!(a.b > 0, "fa2 batch decode: b must be > 0");

    let head_size = round_multiple(a.d, 8);
    let head_size_rounded = round_multiple(head_size, 32);
    let seqlen_q_rounded = round_multiple(1, 128);
    let seqlen_k_rounded = round_multiple(a.max_seqlen_k, 128);

    let mut p: FlashFwdParams = std::mem::zeroed();
    p.q_ptr = a.q_ptr as *mut c_void;
    p.k_ptr = a.k_ptr as *mut c_void;
    p.v_ptr = a.v_ptr as *mut c_void;
    p.o_ptr = a.o_ptr as *mut c_void;
    p.softmax_lse_ptr = a.lse_ptr as *mut c_void;
    p.q_batch_stride = a.q_batch_stride as i64;
    p.k_batch_stride = a.kv_batch_stride as i64;
    p.v_batch_stride = a.kv_batch_stride as i64;
    p.o_batch_stride = a.q_batch_stride as i64;
    p.q_row_stride = (a.h * a.d) as i64;
    p.k_row_stride = (a.h_k * a.d) as i64;
    p.v_row_stride = (a.h_k * a.d) as i64;
    p.o_row_stride = (a.h * a.d) as i64;
    p.q_head_stride = a.d as i64;
    p.k_head_stride = a.d as i64;
    p.v_head_stride = a.d as i64;
    p.o_head_stride = a.d as i64;
    p.b = a.b as i32;
    p.h = a.h as i32;
    p.h_k = a.h_k as i32;
    p.h_h_k_ratio = (a.h / a.h_k) as i32;
    p.seqlen_q = 1;
    p.seqlen_k = a.max_seqlen_k as i32;
    p.seqlen_q_rounded = seqlen_q_rounded as i32;
    p.seqlen_k_rounded = seqlen_k_rounded as i32;
    p.d = head_size as i32;
    p.d_rounded = head_size_rounded as i32;
    p.scale_softmax = a.softmax_scale;
    p.scale_softmax_log2 = a.softmax_scale * std::f32::consts::LOG2_E;
    p.softcap = 0.0;
    p.p_dropout = 1.0;
    p.p_dropout_in_uint8_t = 255;
    p.rp_dropout = 1.0;
    p.scale_softmax_rp_dropout = p.scale_softmax;
    p.is_bf16 = true;
    p.cu_seqlens_q = std::ptr::null_mut();
    p.cu_seqlens_k = std::ptr::null_mut();
    p.seqused_k = a.seqused_k as *mut i32;
    p.is_causal = true;
    p.window_size_left = a.max_seqlen_k as i32;
    p.window_size_right = 0;
    p.is_seqlens_k_cumulative = true;
    p.num_splits = 1;
    p.unpadded_lse = false;

    run_mha_fwd(&mut p, stream);
    Ok(())
}
