struct SamplerParams {
    vocab: u32,
    batch: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    flags: u32,
    u01_bits: u32,
    inv_t: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> smp_logits: array<f32>;
@group(0) @binding(1) var<storage, read> smp_seeds: array<u32>;
@group(0) @binding(2) var<storage, read_write> smp_probs: array<f32>;
@group(0) @binding(3) var<storage, read_write> smp_token: array<u32>;
@group(0) @binding(4) var<uniform> smp_params: SamplerParams;

const SMP_BLOCK: u32 = 256u;
const SMP_SENTINEL: u32 = 0xFFFFFFFFu;
const SMP_NEG_MAX: f32 = -3.4028235e38f;
const SMP_LOG2E: f32 = 1.442695041f;
const SMP_INV_2P24: f32 = 5.9604644775390625e-8f;

var<workgroup> smp_f: array<f32, 256>;
var<workgroup> smp_u: array<u32, 256>;
var<workgroup> smp_threshold: f32;
var<workgroup> smp_total: f32;
var<workgroup> smp_target: f32;
var<workgroup> smp_winner: u32;

fn smp_sync() {
    workgroupBarrier();
    storageBarrier();
}

fn smp_u64_xor(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

fn smp_u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    var carry = 0u;
    if (lo < a.x) {
        carry = 1u;
    }
    return vec2<u32>(lo, a.y + b.y + carry);
}

fn smp_u64_shr(a: vec2<u32>, n: u32) -> vec2<u32> {
    return vec2<u32>((a.x >> n) | (a.y << (32u - n)), a.y >> n);
}

fn smp_mul_wide(x: u32, y: u32) -> vec2<u32> {
    let x0 = x & 0xffffu;
    let x1 = x >> 16u;
    let y0 = y & 0xffffu;
    let y1 = y >> 16u;
    let p00 = x0 * y0;
    let p01 = x0 * y1;
    let p10 = x1 * y0;
    let p11 = x1 * y1;
    let mid = (p00 >> 16u) + (p10 & 0xffffu) + (p01 & 0xffffu);
    let lo = (p00 & 0xffffu) | (mid << 16u);
    let hi = p11 + (p10 >> 16u) + (p01 >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

fn smp_u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let m = smp_mul_wide(a.x, b.x);
    return vec2<u32>(m.x, m.y + a.x * b.y + a.y * b.x);
}

fn smp_splitmix64(seed: vec2<u32>) -> vec2<u32> {
    var z = seed;
    z = smp_u64_mul(smp_u64_xor(z, smp_u64_shr(z, 30u)), vec2<u32>(0x1ce4e5b9u, 0xbf58476du));
    z = smp_u64_mul(smp_u64_xor(z, smp_u64_shr(z, 27u)), vec2<u32>(0x133111ebu, 0x94d049bbu));
    return smp_u64_xor(z, smp_u64_shr(z, 31u));
}

fn smp_unit_float(r: vec2<u32>) -> f32 {
    let mant = r.y >> 8u;
    return f32(mant) * SMP_INV_2P24;
}

fn smp_recip(b: f32) -> f32 {
    let e = 1.0 / b;
    let e1 = fma(fma(-b, e, 1.0), e, e);
    let r = fma(-b, e1, 1.0);
    return fma(r, e1, e1);
}

fn smp_reduce_max(lid: u32, v: f32) -> f32 {
    smp_f[lid] = v;
    smp_sync();
    for (var stride = SMP_BLOCK / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            let other = smp_f[lid + stride];
            if (other > smp_f[lid]) {
                smp_f[lid] = other;
            }
        }
        smp_sync();
    }
    let result = smp_f[0];
    smp_sync();
    return result;
}

fn smp_reduce_sum(lid: u32, v: f32) -> f32 {
    smp_f[lid] = v;
    smp_sync();
    for (var stride = SMP_BLOCK / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            smp_f[lid] = smp_f[lid] + smp_f[lid + stride];
        }
        smp_sync();
    }
    let result = smp_f[0];
    smp_sync();
    return result;
}

fn smp_reduce_sum_u32(lid: u32, v: u32) -> u32 {
    smp_u[lid] = v;
    smp_sync();
    for (var stride = SMP_BLOCK / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            smp_u[lid] = smp_u[lid] + smp_u[lid + stride];
        }
        smp_sync();
    }
    let result = smp_u[0];
    smp_sync();
    return result;
}

@compute @workgroup_size(256)
fn sampler_topk_topp(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.x + wg.y * nwg.x;
    if (row >= smp_params.batch) {
        return;
    }
    let lid = tid.x;
    let vocab = smp_params.vocab;
    let base = row * vocab;

    var inv_t = 1.0e6f;
    if (smp_params.temperature > 0.0) {
        inv_t = smp_recip(smp_params.temperature);
    }

    var local_max = SMP_NEG_MAX;
    for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
        let v = smp_logits[base + i] * inv_t;
        if (v > local_max) {
            local_max = v;
        }
    }
    let row_max = smp_reduce_max(lid, local_max);

    var local_sum = 0.0;
    for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
        let v = smp_logits[base + i] * inv_t;
        let e = exp2((v - row_max) * SMP_LOG2E);
        smp_probs[base + i] = e;
        local_sum = local_sum + e;
    }
    let row_sum = smp_reduce_sum(lid, local_sum);
    var inv_sum = 0.0;
    if (row_sum > 0.0) {
        inv_sum = smp_recip(row_sum);
    }

    for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
        smp_probs[base + i] = smp_probs[base + i] * inv_sum;
    }
    smp_sync();

    if (smp_params.top_k > 0u && smp_params.top_k < vocab) {
        var lo = 0.0;
        var hi = 1.0f + 1e-6f;
        for (var iter = 0; iter < 40; iter = iter + 1) {
            let mid = 0.5 * (lo + hi);
            var local_count = 0u;
            for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
                if (smp_probs[base + i] >= mid) {
                    local_count = local_count + 1u;
                }
            }
            let total_count = smp_reduce_sum_u32(lid, local_count);
            if (total_count > smp_params.top_k) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        if (lid == 0u) {
            smp_threshold = hi;
        }
        smp_sync();
        let thr = smp_threshold;
        for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
            if (smp_probs[base + i] < thr) {
                smp_probs[base + i] = 0.0;
            }
        }
        smp_sync();

        var local_sum2 = 0.0;
        for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
            local_sum2 = local_sum2 + smp_probs[base + i];
        }
        let sum2 = smp_reduce_sum(lid, local_sum2);
        var inv2 = 0.0;
        if (sum2 > 0.0) {
            inv2 = smp_recip(sum2);
        }
        for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
            smp_probs[base + i] = smp_probs[base + i] * inv2;
        }
        smp_sync();
    }

    if (smp_params.top_p < 1.0 && smp_params.top_p > 0.0) {
        var lo = 0.0;
        var hi = 1.0;
        for (var iter = 0; iter < 40; iter = iter + 1) {
            let mid = 0.5 * (lo + hi);
            var local_mass = 0.0;
            for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
                let p = smp_probs[base + i];
                if (p >= mid) {
                    local_mass = local_mass + p;
                }
            }
            let mass = smp_reduce_sum(lid, local_mass);
            if (mass > smp_params.top_p) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        if (lid == 0u) {
            smp_threshold = lo;
        }
        smp_sync();
        let thr = smp_threshold;
        for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
            if (smp_probs[base + i] < thr) {
                smp_probs[base + i] = 0.0;
            }
        }
        smp_sync();

        var local_sum3 = 0.0;
        for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
            local_sum3 = local_sum3 + smp_probs[base + i];
        }
        let sum3 = smp_reduce_sum(lid, local_sum3);
        var inv3 = 0.0;
        if (sum3 > 0.0) {
            inv3 = smp_recip(sum3);
        }
        for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
            smp_probs[base + i] = smp_probs[base + i] * inv3;
        }
        smp_sync();
    }

    var local_total = 0.0;
    for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
        local_total = local_total + smp_probs[base + i];
    }
    let total = smp_reduce_sum(lid, local_total);

    if (lid == 0u) {
        let seed = vec2<u32>(smp_seeds[row * 2u], smp_seeds[row * 2u + 1u]);
        let golden = smp_u64_add(vec2<u32>(0x7f4a7c15u, 0x9e3779b9u), vec2<u32>(row, 0u));
        let mixed = smp_splitmix64(smp_u64_xor(seed, golden));
        var u = smp_unit_float(mixed);
        if (u >= 1.0) {
            u = 0.99999994;
        }
        smp_total = total;
        smp_target = u * total;
        smp_winner = SMP_SENTINEL;
    }
    smp_sync();

    let tgt = smp_target;

    var local_partial = 0.0;
    for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
        local_partial = local_partial + smp_probs[base + i];
    }
    smp_f[lid] = local_partial;
    smp_sync();

    if (lid == 0u) {
        var cum = 0.0;
        var found = -1;
        var prefix = 0.0;
        for (var t = 0u; t < SMP_BLOCK; t = t + 1u) {
            let seg = smp_f[t];
            if (cum + seg >= tgt) {
                found = i32(t);
                prefix = cum;
                break;
            }
            cum = cum + seg;
        }
        if (found < 0) {
            found = i32(SMP_BLOCK) - 1;
            prefix = smp_total;
        }
        smp_u[0] = u32(found);
        smp_f[0] = prefix;
    }
    smp_sync();

    let found_tid = smp_u[0];
    let prefix_before = smp_f[0];
    smp_sync();

    if (lid == found_tid) {
        var cum = prefix_before;
        var pick = SMP_SENTINEL;
        for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
            let p = smp_probs[base + i];
            cum = cum + p;
            if (cum >= tgt && p > 0.0) {
                pick = i;
                break;
            }
        }
        if (pick == SMP_SENTINEL) {
            var i = vocab;
            loop {
                if (i == 0u) {
                    break;
                }
                if (smp_probs[base + i - 1u] > 0.0) {
                    pick = i - 1u;
                    break;
                }
                i = i - 1u;
            }
        }
        smp_winner = pick;
    }
    smp_sync();

    if (lid == 0u) {
        smp_token[row] = smp_winner;
    }
}

const EX_KMAX: u32 = 256u;
const EX_SENTINEL: u32 = 0xFFFFFFFFu;
const EX_UMAX: f32 = 0.99999988f;
const EX_FLAG_HOST_U01: u32 = 1u;

var<workgroup> ex_idx: array<u32, 256>;
var<workgroup> ex_val: array<f32, 256>;
var<workgroup> ex_prob: array<f32, 256>;
var<workgroup> ex_ord: array<u32, 256>;
var<workgroup> ex_count: atomic<u32>;

fn ex_okey(x: f32) -> u32 {
    if (x == 0.0) {
        return 0x80000000u;
    }
    let b = bitcast<u32>(x);
    if ((b & 0x80000000u) != 0u) {
        return ~b;
    }
    return b | 0x80000000u;
}

fn ex_reduce_min_u32(lid: u32, v: u32) -> u32 {
    smp_u[lid] = v;
    smp_sync();
    for (var stride = SMP_BLOCK / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            let other = smp_u[lid + stride];
            if (other < smp_u[lid]) {
                smp_u[lid] = other;
            }
        }
        smp_sync();
    }
    let result = smp_u[0];
    smp_sync();
    return result;
}

@compute @workgroup_size(256)
fn sampler_exact_token(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.x + wg.y * nwg.x;
    if (row >= smp_params.batch) {
        return;
    }
    let lid = tid.x;
    let vocab = smp_params.vocab;
    let base = row * vocab;
    let k = smp_params.top_k;

    if (lid == 0u) {
        smp_token[row] = EX_SENTINEL;
        atomicStore(&ex_count, 0u);
    }
    smp_sync();

    if (smp_params.temperature <= 1e-6) {
        var local_max = SMP_NEG_MAX;
        for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
            let v = smp_logits[base + i];
            if (v > local_max) {
                local_max = v;
            }
        }
        let row_max = smp_reduce_max(lid, local_max);
        var local_i = EX_SENTINEL;
        for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
            if (smp_logits[base + i] == row_max && i < local_i) {
                local_i = i;
            }
        }
        let best = ex_reduce_min_u32(lid, local_i);
        if (lid == 0u) {
            smp_token[row] = best;
        }
        return;
    }

    if (k == 0u || k > EX_KMAX || k > vocab) {
        return;
    }

    let inv_t = smp_params.inv_t;

    var t = 0u;
    for (var b = 0u; b < 32u; b = b + 1u) {
        let bit = 31u - b;
        let cand = t | (1u << bit);
        var c = 0u;
        for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
            if (ex_okey(smp_logits[base + i] * inv_t) >= cand) {
                c = c + 1u;
            }
        }
        let tot = smp_reduce_sum_u32(lid, c);
        if (tot >= k) {
            t = cand;
        }
    }

    var cgt = 0u;
    for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
        if (ex_okey(smp_logits[base + i] * inv_t) > t) {
            cgt = cgt + 1u;
        }
    }
    let n_gt = smp_reduce_sum_u32(lid, cgt);
    let need = k - n_gt;

    var lo = 0u;
    var hi = vocab - 1u;
    for (var it = 0u; it < 32u; it = it + 1u) {
        let mid = lo + (hi - lo) / 2u;
        var c = 0u;
        for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
            if (i <= mid && ex_okey(smp_logits[base + i] * inv_t) == t) {
                c = c + 1u;
            }
        }
        let tot = smp_reduce_sum_u32(lid, c);
        if (tot >= need) {
            hi = mid;
        } else {
            lo = min(mid + 1u, hi);
        }
    }
    let ilim = lo;

    for (var i = lid; i < vocab; i = i + SMP_BLOCK) {
        let v = smp_logits[base + i] * inv_t;
        let key = ex_okey(v);
        if (key > t || (key == t && i <= ilim)) {
            let slot = atomicAdd(&ex_count, 1u);
            if (slot < EX_KMAX) {
                ex_idx[slot] = i;
                ex_val[slot] = v;
            }
        }
    }
    smp_sync();

    if (lid != 0u) {
        return;
    }
    let cnt = atomicLoad(&ex_count);
    if (cnt == 0u || cnt > EX_KMAX) {
        return;
    }

    for (var a = 1u; a < cnt; a = a + 1u) {
        let vi = ex_idx[a];
        let vv = ex_val[a];
        var j = a;
        loop {
            if (j == 0u) {
                break;
            }
            if (ex_idx[j - 1u] <= vi) {
                break;
            }
            ex_idx[j] = ex_idx[j - 1u];
            ex_val[j] = ex_val[j - 1u];
            j = j - 1u;
        }
        ex_idx[j] = vi;
        ex_val[j] = vv;
    }

    var mx = SMP_NEG_MAX;
    for (var a = 0u; a < cnt; a = a + 1u) {
        if (ex_val[a] > mx) {
            mx = ex_val[a];
        }
    }

    var sum = 0.0;
    for (var a = 0u; a < cnt; a = a + 1u) {
        let e = exp(ex_val[a] - mx);
        ex_prob[a] = e;
        sum = sum + e;
    }
    if (!(sum > 0.0)) {
        return;
    }
    for (var a = 0u; a < cnt; a = a + 1u) {
        ex_prob[a] = ex_prob[a] / sum;
    }

    let tp = smp_params.top_p;
    if (tp > 0.0 && tp < 1.0) {
        for (var a = 0u; a < cnt; a = a + 1u) {
            ex_ord[a] = a;
        }
        for (var a = 1u; a < cnt; a = a + 1u) {
            let cur = ex_ord[a];
            var j = a;
            loop {
                if (j == 0u) {
                    break;
                }
                let prev = ex_ord[j - 1u];
                let pp = ex_prob[prev];
                let pc = ex_prob[cur];
                if (pp > pc || (pp == pc && ex_idx[prev] <= ex_idx[cur])) {
                    break;
                }
                ex_ord[j] = prev;
                j = j - 1u;
            }
            ex_ord[j] = cur;
        }
        var cum = 0.0;
        var keep = cnt;
        for (var a = 0u; a < cnt; a = a + 1u) {
            cum = cum + ex_prob[ex_ord[a]];
            if (cum >= tp) {
                keep = a + 1u;
                break;
            }
        }
        for (var a = keep; a < cnt; a = a + 1u) {
            ex_prob[ex_ord[a]] = 0.0;
        }
    }

    let mp = smp_params.min_p;
    if (mp > 0.0) {
        var pmax = 0.0;
        for (var a = 0u; a < cnt; a = a + 1u) {
            if (ex_prob[a] > pmax) {
                pmax = ex_prob[a];
            }
        }
        let thr = mp * pmax;
        for (var a = 0u; a < cnt; a = a + 1u) {
            if (ex_prob[a] < thr) {
                ex_prob[a] = 0.0;
            }
        }
    }

    var renorm = 0.0;
    for (var a = 0u; a < cnt; a = a + 1u) {
        renorm = renorm + ex_prob[a];
    }
    if (!(renorm > 0.0)) {
        return;
    }
    for (var a = 0u; a < cnt; a = a + 1u) {
        ex_prob[a] = ex_prob[a] / renorm;
    }

    var u = bitcast<f32>(smp_params.u01_bits);
    if ((smp_params.flags & EX_FLAG_HOST_U01) == 0u) {
        let seed = vec2<u32>(smp_seeds[row * 2u], smp_seeds[row * 2u + 1u]);
        let golden = smp_u64_add(vec2<u32>(0x7f4a7c15u, 0x9e3779b9u), vec2<u32>(row, 0u));
        u = smp_unit_float(smp_splitmix64(smp_u64_xor(seed, golden)));
    }
    u = clamp(u, 0.0, EX_UMAX);

    var acc = 0.0;
    var picked = EX_SENTINEL;
    for (var a = 0u; a < cnt; a = a + 1u) {
        acc = acc + ex_prob[a];
        if (u < acc) {
            picked = ex_idx[a];
            break;
        }
    }
    if (picked == EX_SENTINEL) {
        var a = cnt;
        loop {
            if (a == 0u) {
                break;
            }
            if (ex_prob[a - 1u] > 0.0) {
                picked = ex_idx[a - 1u];
                break;
            }
            a = a - 1u;
        }
    }
    smp_token[row] = picked;
}
