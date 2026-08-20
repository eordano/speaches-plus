struct MoePermuteParams {
    total: u32,
    k: u32,
    num_experts: u32,
    num_blocks: u32,
};

@group(0) @binding(0) var<storage, read> mp_topk_ids: array<i32>;
@group(0) @binding(1) var<storage, read_write> mp_counts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> mp_block_counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> mp_expert_offsets: array<i32>;
@group(0) @binding(4) var<storage, read_write> mp_perm: array<i32>;
@group(0) @binding(5) var<storage, read_write> mp_inv_perm: array<i32>;
@group(0) @binding(6) var<uniform> mp_params: MoePermuteParams;

const MOE_PERMUTE_BLOCK: u32 = 256u;

var<workgroup> mp_tile: array<i32, 256>;

fn mp_linear_group(wg: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wg.x + wg.y * nwg.x;
}

@compute @workgroup_size(256)
fn moe_permute_count(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let block = mp_linear_group(wg, nwg);
    let t = block * MOE_PERMUTE_BLOCK + lid.x;
    if (t >= mp_params.total) {
        return;
    }
    let e = mp_topk_ids[t];
    if (e < 0 || u32(e) >= mp_params.num_experts) {
        return;
    }
    let eu = u32(e);
    atomicAdd(&mp_counts[eu], 1u);
    atomicAdd(&mp_block_counts[block * mp_params.num_experts + eu], 1u);
}

@compute @workgroup_size(1)
fn moe_permute_scan() {
    var acc: u32 = 0u;
    mp_expert_offsets[0] = 0;
    for (var e: u32 = 0u; e < mp_params.num_experts; e = e + 1u) {
        acc = acc + atomicLoad(&mp_counts[e]);
        mp_expert_offsets[e + 1u] = i32(acc);
    }
}

@compute @workgroup_size(256)
fn moe_permute_block_scan(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let e = mp_linear_group(wg, nwg) * MOE_PERMUTE_BLOCK + lid.x;
    if (e >= mp_params.num_experts) {
        return;
    }
    var base = u32(mp_expert_offsets[e]);
    for (var b: u32 = 0u; b < mp_params.num_blocks; b = b + 1u) {
        let idx = b * mp_params.num_experts + e;
        let c = atomicLoad(&mp_block_counts[idx]);
        atomicStore(&mp_block_counts[idx], base);
        base = base + c;
    }
}

@compute @workgroup_size(256)
fn moe_permute_assign(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let block = mp_linear_group(wg, nwg);
    let i = lid.x;
    let t = block * MOE_PERMUTE_BLOCK + i;
    var e: i32 = -1;
    if (t < mp_params.total) {
        e = mp_topk_ids[t];
    }
    mp_tile[i] = e;
    workgroupBarrier();

    if (t >= mp_params.total) {
        return;
    }
    if (e < 0 || u32(e) >= mp_params.num_experts) {
        return;
    }
    var rank: u32 = 0u;
    for (var j: u32 = 0u; j < i; j = j + 1u) {
        if (mp_tile[j] == e) {
            rank = rank + 1u;
        }
    }
    let pos = atomicLoad(&mp_block_counts[block * mp_params.num_experts + u32(e)]) + rank;
    mp_perm[pos] = i32(t / mp_params.k);
    mp_inv_perm[t] = i32(pos);
}
