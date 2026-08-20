struct HelloParams {
    n: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> hello_out: array<f32>;
@group(0) @binding(1) var<uniform> hello_params: HelloParams;

const HELLO_BLOCK: u32 = 256u;

@compute @workgroup_size(256)
fn hello_fill(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let block = wg.x + wg.y * nwg.x;
    let idx = block * HELLO_BLOCK + tid.x;
    if (idx < hello_params.n) {
        hello_out[idx] = f32(idx);
    }
}
