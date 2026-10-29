use std::ffi::c_void;

pub const PAD_DRAFT_ROWS_MAX: usize = 8;

unsafe extern "C" {
    pub fn nv_kernels_voxtral_spec_begin(
        stream: *mut c_void,
        pos: *const i32,
        rows_pos: *mut i32,
        pos_end: *mut i32,
        m: i32,
    ) -> i32;
    pub fn nv_kernels_voxtral_spec_qkv_prep(
        stream: *mut c_void,
        qkv: *const u16,
        q_out: *mut f32,
        cache_k: *mut u16,
        cache_v: *mut u16,
        cos_tbl: *const f32,
        sin_tbl: *const f32,
        rows_pos: *const i32,
        m: i32,
        nh: i32,
        nkv: i32,
        hd: i32,
    ) -> i32;
    pub fn nv_kernels_voxtral_spec_silu_mul_halves(
        stream: *mut c_void,
        gate_up: *const u16,
        act: *mut u16,
        rows: i32,
        inter: i32,
    ) -> i32;
    pub fn nv_kernels_voxtral_spec_argmax_parts() -> i32;
    pub fn nv_kernels_voxtral_spec_accept(
        stream: *mut c_void,
        logits: *const u16,
        m: i32,
        n: i32,
        part_val: *mut f32,
        part_idx: *mut i32,
        pos: *mut i32,
        toks: *mut i32,
        ring: *mut u32,
        ring_mask: i32,
        pad: i32,
    ) -> i32;
}

pub fn argmax_parts() -> usize {
    (unsafe { nv_kernels_voxtral_spec_argmax_parts() }) as usize
}
