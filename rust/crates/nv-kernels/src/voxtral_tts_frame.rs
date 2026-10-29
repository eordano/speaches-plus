use std::ffi::c_void;

pub const FLOW_TOKENS_MAX: usize = 4;

unsafe extern "C" {
    pub fn nv_kernels_voxtral_tts_begin(
        stream: *mut c_void,
        pos: *const i32,
        rows_pos: *mut i32,
        pos_end: *mut i32,
        fi: *const i32,
        noise: *const u16,
        x: *mut u16,
        noise_rows: i32,
        n: i32,
    ) -> i32;
    pub fn nv_kernels_voxtral_tts_semantic_argmax(
        stream: *mut c_void,
        logits: *const u16,
        rows: i32,
        fi: *const i32,
        ring_slots: i32,
        nc: i32,
        ring: *mut u32,
        end_token: i32,
    ) -> i32;
    pub fn nv_kernels_voxtral_tts_flow_attn(
        stream: *mut c_void,
        qkv: *const u16,
        out: *mut u16,
        b: i32,
        s: i32,
        nh: i32,
        nkv: i32,
        hd: i32,
        scale: f32,
    ) -> i32;
    pub fn nv_kernels_voxtral_tts_fsq(
        stream: *mut c_void,
        x: *const u16,
        n: i32,
        levels: f32,
        fi: *const i32,
        ring_slots: i32,
        nc: i32,
        ring: *mut u32,
        end_token: i32,
        n_special: i32,
    ) -> i32;
    pub fn nv_kernels_voxtral_tts_embed_sum(
        stream: *mut c_void,
        table: *const u16,
        offsets: *const i32,
        ring: *const u32,
        teacher: *const i32,
        teacher_len: *const i32,
        fi: *const i32,
        ring_slots: i32,
        nc: i32,
        out: *mut u16,
        d: i32,
    ) -> i32;
    pub fn nv_kernels_voxtral_tts_advance(stream: *mut c_void, pos: *mut i32, fi: *mut i32) -> i32;
}
