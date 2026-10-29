use std::ffi::c_void;

pub const GEMV_PLAIN: i32 = 0;
pub const GEMV_RMSNORM_PROLOGUE: i32 = 1;
pub const GEMV_RESIDUAL_RMSNORM_PROLOGUE_WRITES_THE_SUM: i32 = 2;
pub const GEMV_SILU_PROLOGUE_RESIDUAL_EPILOGUE: i32 = 3;

unsafe extern "C" {
    pub fn nv_kernels_voxtral_tts_backbone_gemv(
        stream: *mut c_void,
        mode: i32,
        w: *const c_void,
        row_scale: *const f32,
        x: *const u16,
        aux: *const u16,
        wn: *const u16,
        eps: f32,
        side: *mut u16,
        y: *mut u16,
        n: i32,
        k: i32,
    ) -> i32;
}
