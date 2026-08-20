
struct UpkParams {
    src_off: u32,
    n_words: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> upk_src: array<u32>;
@group(0) @binding(1) var<storage, read_write> upk_dst: array<f32>;
@group(0) @binding(2) var<uniform> upk_p: UpkParams;

@compute @workgroup_size(256)
fn adw_unpack_hidden(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= upk_p.n_words) {
        return;
    }
    let w = upk_src[upk_p.src_off + i];
    upk_dst[2u * i] = bitcast<f32>((w & 0xffffu) << 16u);
    upk_dst[2u * i + 1u] = bitcast<f32>(w & 0xffff0000u);
}
