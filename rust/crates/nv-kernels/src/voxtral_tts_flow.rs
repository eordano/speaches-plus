use std::ffi::c_void;

pub const FLOW_ROWS: usize = 6;
pub const FLOW_STEP_MAX_ACOUSTIC: usize = 64;

pub const EPILOGUE_STORE: i32 = 0;
pub const EPILOGUE_SILU_OF_PAIRED_GATE_AND_UP_ROWS: i32 = 1;
pub const EPILOGUE_ADD_RESIDUAL: i32 = 2;

unsafe extern "C" {
    pub fn nv_kernels_voxtral_tts_flow_gemm(
        stream: *mut c_void,
        epilogue: i32,
        fp8: i32,
        w: *const c_void,
        row_scale: *const f32,
        x: *const u16,
        epi_aux: *const u16,
        y: *mut u16,
        n: i32,
        k: i32,
    ) -> i32;
    pub fn nv_kernels_voxtral_tts_flow_step(
        stream: *mut c_void,
        x_in: *const u16,
        v: *const u16,
        uncond_off: i32,
        alpha: f32,
        one_minus_alpha: f32,
        dt: f32,
        x_out: *mut u16,
        w_in: *const u16,
        tp: *const u16,
        cond: *const u16,
        tokens: *mut u16,
        n: i32,
        d: i32,
    ) -> i32;
}
