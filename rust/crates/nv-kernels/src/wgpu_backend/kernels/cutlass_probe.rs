pub const CUTLASS_STATUS_SUCCESS: i32 = 0;
pub const E2M1_MAX_TIMES_32: i32 = 192;

pub fn cutlass_flashinfer_probe() -> (i32, i32) {
    (CUTLASS_STATUS_SUCCESS, E2M1_MAX_TIMES_32)
}
