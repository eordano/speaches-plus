use std::ffi::c_void;

pub const PAD_ZERO: i32 = 0;
pub const PAD_REPLICATE: i32 = 1;
pub const PAD_REFLECT: i32 = 2;

unsafe extern "C" {
    pub fn nv_kernels_voxtral_tts_codec_im2col(
        stream: *mut c_void,
        src: *const u16,
        hist: *const u16,
        ctl: *const i32,
        dst: *mut f32,
        rows: i32,
        channels: i32,
        taps: i32,
        mode: i32,
    ) -> i32;
    pub fn nv_kernels_voxtral_tts_codec_carry(
        stream: *mut c_void,
        cols: *const f32,
        hist: *mut u16,
        rows: i32,
        channels: i32,
        taps: i32,
    ) -> i32;
    pub fn nv_kernels_voxtral_tts_codec_cast_norm(
        stream: *mut c_void,
        src: *const f32,
        x: *mut u16,
        w: *const u16,
        h: *mut u16,
        rows: i32,
        eps: f32,
    ) -> i32;
    pub fn nv_kernels_voxtral_tts_codec_qk_norm(
        stream: *mut c_void,
        qkv: *const u16,
        qw: *const u16,
        kw: *const u16,
        qn: *mut u16,
        kn: *mut u16,
        rows: i32,
        eps: f32,
    ) -> i32;
    pub fn nv_kernels_voxtral_tts_codec_residual_norm(
        stream: *mut c_void,
        x: *mut u16,
        t: *const u16,
        scale: *const u16,
        w_next: *const u16,
        h: *mut u16,
        rows: i32,
        eps: f32,
    ) -> i32;
    pub fn nv_kernels_voxtral_tts_codec_attn(
        stream: *mut c_void,
        q: *const u16,
        k: *const u16,
        qkv: *const u16,
        kring: *const u16,
        vring: *const u16,
        ctl: *const i32,
        out: *mut u16,
        rows: i32,
        window: i32,
        rows_per_frame: i32,
        scale: f32,
    ) -> i32;
    pub fn nv_kernels_voxtral_tts_codec_kv_commit(
        stream: *mut c_void,
        k: *const u16,
        qkv: *const u16,
        kring: *mut u16,
        vring: *mut u16,
        ctl: *const i32,
        rows: i32,
        window: i32,
        rows_per_frame: i32,
    ) -> i32;
    pub fn nv_kernels_voxtral_tts_codec_silu_mul(
        stream: *mut c_void,
        gu: *const u16,
        act: *mut u16,
        rows: i32,
        inter: i32,
    ) -> i32;
    pub fn nv_kernels_voxtral_tts_codec_advance(stream: *mut c_void, ctl: *mut i32, frames: i32) -> i32;
}
