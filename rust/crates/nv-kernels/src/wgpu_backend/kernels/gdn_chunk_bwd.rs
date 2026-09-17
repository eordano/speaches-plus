#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};

pub const HEAD_DIM: usize = 128;
pub const WORKGROUP_SIZE: u32 = 128;

pub const BROWSER_PORTABLE_BY_CONSTRUCTION_NO_SUBGROUPS_TEN_STORAGE_BINDINGS_F32_ONLY: &str =
    "the chunked-recompute backward is the q38 browser-training seam, so both entries stay \
     inside the measured Chrome/Metal caps: no subgroup ops (no enable-directive split needed), \
     at most 10 storage buffers per entry, f32 storage only, workgroup_size 128 with ~7 KB of \
     workgroup memory against the 32768-byte limit";

pub const THE_BACKWARD_RECOMPUTES_KV_MEM_FROM_THE_STORED_PRE_STEP_STATE_WITH_THE_FORWARD_EXPRESSIONS: &str =
    "gdn_chunk_fwd_states_f32 replays gdn_recurrent.wgsl's exact fma sequence from a chunk \
     checkpoint and stores every post-step state, so the backward's s_hat = state*ge and \
     kv_mem/delta recomputations see bit-identical operands to the forward that actually ran; \
     cross-column reductions (dq, dk, dbeta, dg) are workgroup-tree folds, so grads gate at \
     fp tolerance against a sequential host oracle, never bitwise";

pub const STATES_LAYOUT_IS_PAIR_MAJOR_WITH_SLOT_ZERO_HOLDING_THE_CHUNK_INPUT_CHECKPOINT: &str =
    "gc_states holds (seq+1) state slots of 128x128 f32 per (batch,head) pair; the host writes \
     the chunk's input checkpoint into slot 0 before dispatch and slot t+1 receives the state \
     after step t, so slot seq is the next chunk's checkpoint";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GdnChunkParams {
    batch: u32,
    seq: u32,
    heads: u32,
    pairs: u32,
}

pub const WGSL: &str = include_str!("../../../wgsl/gdn_chunk_bwd.wgsl");

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup(ctx, "gdn_chunk_bwd", WORKGROUP_SIZE)
}

fn slice_rows(
    full: &[f32],
    b: usize,
    t_total: usize,
    h: usize,
    width: usize,
    t0: usize,
    clen: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(b * clen * h * width);
    for bi in 0..b {
        let start = ((bi * t_total) + t0) * h * width;
        out.extend_from_slice(&full[start..start + clen * h * width]);
    }
    out
}

fn scatter_rows(
    full: &mut [f32],
    part: &[f32],
    b: usize,
    t_total: usize,
    h: usize,
    width: usize,
    t0: usize,
    clen: usize,
) {
    for bi in 0..b {
        let start = ((bi * t_total) + t0) * h * width;
        full[start..start + clen * h * width]
            .copy_from_slice(&part[bi * clen * h * width..(bi + 1) * clen * h * width]);
    }
}

pub struct GdnChunkedGrads {
    pub out: Vec<f32>,
    pub dq: Vec<f32>,
    pub dk: Vec<f32>,
    pub dv: Vec<f32>,
    pub dg: Vec<f32>,
    pub dbeta: Vec<f32>,
    pub final_state: Vec<f32>,
}

struct ChunkIo {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    gb: Vec<f32>,
    t0: usize,
    clen: usize,
}

fn chunk_io(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g_exp: &[f32],
    beta: &[f32],
    b: usize,
    t: usize,
    h: usize,
    t0: usize,
    clen: usize,
) -> ChunkIo {
    let gs = slice_rows(g_exp, b, t, h, 1, t0, clen);
    let bs = slice_rows(beta, b, t, h, 1, t0, clen);
    let gb: Vec<f32> = gs
        .iter()
        .zip(bs.iter())
        .flat_map(|(g, be)| [*g, *be])
        .collect();
    ChunkIo {
        q: slice_rows(q, b, t, h, HEAD_DIM, t0, clen),
        k: slice_rows(k, b, t, h, HEAD_DIM, t0, clen),
        v: slice_rows(v, b, t, h, HEAD_DIM, t0, clen),
        gb,
        t0,
        clen,
    }
}

fn run_fwd_states(
    ctx: &WgpuContext,
    io: &ChunkIo,
    checkpoint: &[f32],
    b: usize,
    h: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let pairs = b * h;
    let state_words = HEAD_DIM * HEAD_DIM;
    let slots = io.clen + 1;
    let mut states_init = vec![0f32; pairs * slots * state_words];
    for bh in 0..pairs {
        states_init[bh * slots * state_words..bh * slots * state_words + state_words]
            .copy_from_slice(&checkpoint[bh * state_words..(bh + 1) * state_words]);
    }
    let params = GdnChunkParams {
        batch: b as u32,
        seq: io.clen as u32,
        heads: h as u32,
        pairs: pairs as u32,
    };
    let qb = dispatch::storage_from_slice(ctx, "gdnck-q", &io.q);
    let kb = dispatch::storage_from_slice(ctx, "gdnck-k", &io.k);
    let vb = dispatch::storage_from_slice(ctx, "gdnck-v", &io.v);
    let gbb = dispatch::storage_from_slice(ctx, "gdnck-gb", &io.gb);
    let sb = dispatch::storage_from_slice(ctx, "gdnck-states", &states_init);
    let ob = dispatch::storage_zeroed(ctx, "gdnck-out", (io.q.len() * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "gdnck-params", &params);
    let groups = dispatch::workgroup_count_1d(ctx, pairs as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_gdn_chunk_fwd_states_f32",
        &compose(WGSL),
        "gdn_chunk_fwd_states_f32",
        &[
            (0, &qb),
            (1, &kb),
            (2, &vb),
            (3, &gbb),
            (4, &sb),
            (5, &ob),
            (10, &pb),
        ],
        groups,
    )?;
    let states: Vec<f32> = dispatch::read_back(ctx, &sb, pairs * slots * state_words)?;
    let out: Vec<f32> = dispatch::read_back(ctx, &ob, io.q.len())?;
    let mut final_state = vec![0f32; pairs * state_words];
    for bh in 0..pairs {
        let src = (bh * slots + io.clen) * state_words;
        final_state[bh * state_words..(bh + 1) * state_words]
            .copy_from_slice(&states[src..src + state_words]);
    }
    Ok((states, out, final_state))
}

fn run_bwd_chunk(
    ctx: &WgpuContext,
    io: &ChunkIo,
    dout_slice: &[f32],
    states: &[f32],
    dstate: &mut [f32],
    b: usize,
    h: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let pairs = b * h;
    let params = GdnChunkParams {
        batch: b as u32,
        seq: io.clen as u32,
        heads: h as u32,
        pairs: pairs as u32,
    };
    let plane = io.q.len();
    let qb = dispatch::storage_from_slice(ctx, "gdnck-q", &io.q);
    let kb = dispatch::storage_from_slice(ctx, "gdnck-k", &io.k);
    let vb = dispatch::storage_from_slice(ctx, "gdnck-v", &io.v);
    let gbb = dispatch::storage_from_slice(ctx, "gdnck-gb", &io.gb);
    let sb = dispatch::storage_from_slice(ctx, "gdnck-states", states);
    let dob = dispatch::storage_from_slice(ctx, "gdnck-dout", dout_slice);
    let dsb = dispatch::storage_from_slice(ctx, "gdnck-dstate", dstate);
    let dqkvb = dispatch::storage_zeroed(ctx, "gdnck-dqkv", (3 * plane * 4) as u64);
    let dgbb = dispatch::storage_zeroed(ctx, "gdnck-dgb", (io.gb.len() * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "gdnck-params", &params);
    let groups = dispatch::workgroup_count_1d(ctx, pairs as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_gdn_chunk_bwd_f32",
        &compose(WGSL),
        "gdn_chunk_bwd_f32",
        &[
            (0, &qb),
            (1, &kb),
            (2, &vb),
            (3, &gbb),
            (4, &sb),
            (6, &dob),
            (7, &dsb),
            (8, &dqkvb),
            (9, &dgbb),
            (10, &pb),
        ],
        groups,
    )?;
    let dqkv: Vec<f32> = dispatch::read_back(ctx, &dqkvb, 3 * plane)?;
    let dgb: Vec<f32> = dispatch::read_back(ctx, &dgbb, io.gb.len())?;
    let ds: Vec<f32> = dispatch::read_back(ctx, &dsb, dstate.len())?;
    dstate.copy_from_slice(&ds);
    let dq = dqkv[..plane].to_vec();
    let dk = dqkv[plane..2 * plane].to_vec();
    let dv = dqkv[2 * plane..].to_vec();
    let rows = io.gb.len() / 2;
    let mut dg = vec![0f32; rows];
    let mut dbeta = vec![0f32; rows];
    for r in 0..rows {
        dg[r] = dgb[2 * r];
        dbeta[r] = dgb[2 * r + 1];
    }
    Ok((dq, dk, dv, dg, dbeta))
}

pub fn gdn_chunked_backward_f32(
    ctx: &WgpuContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g_exp: &[f32],
    beta: &[f32],
    dout: &[f32],
    b: usize,
    h: usize,
    t: usize,
    chunk: usize,
) -> Result<GdnChunkedGrads> {
    let _ = BROWSER_PORTABLE_BY_CONSTRUCTION_NO_SUBGROUPS_TEN_STORAGE_BINDINGS_F32_ONLY;
    let _ =
        THE_BACKWARD_RECOMPUTES_KV_MEM_FROM_THE_STORED_PRE_STEP_STATE_WITH_THE_FORWARD_EXPRESSIONS;
    let _ = STATES_LAYOUT_IS_PAIR_MAJOR_WITH_SLOT_ZERO_HOLDING_THE_CHUNK_INPUT_CHECKPOINT;
    if chunk == 0 {
        return Err(WgpuError::Shape("chunk interval must be at least 1".into()));
    }
    let rows = b * t * h;
    let vecs = rows * HEAD_DIM;
    dispatch::check_len("gdn_chunk q", q.len(), vecs)?;
    dispatch::check_len("gdn_chunk k", k.len(), vecs)?;
    dispatch::check_len("gdn_chunk v", v.len(), vecs)?;
    dispatch::check_len("gdn_chunk g_exp", g_exp.len(), rows)?;
    dispatch::check_len("gdn_chunk beta", beta.len(), rows)?;
    dispatch::check_len("gdn_chunk dout", dout.len(), vecs)?;
    check_device(ctx)?;

    let pairs = b * h;
    let state_words = HEAD_DIM * HEAD_DIM;
    let n_chunks = t.div_ceil(chunk);

    let mut checkpoints: Vec<Vec<f32>> = Vec::with_capacity(n_chunks);
    let mut out = vec![0f32; vecs];
    let mut state = vec![0f32; pairs * state_words];
    for cj in 0..n_chunks {
        let t0 = cj * chunk;
        let clen = chunk.min(t - t0);
        checkpoints.push(state.clone());
        let io = chunk_io(q, k, v, g_exp, beta, b, t, h, t0, clen);
        let (_, chunk_out, final_state) = run_fwd_states(ctx, &io, &state, b, h)?;
        scatter_rows(&mut out, &chunk_out, b, t, h, HEAD_DIM, t0, clen);
        state = final_state;
    }
    let final_state = state;

    let mut dq = vec![0f32; vecs];
    let mut dk = vec![0f32; vecs];
    let mut dv = vec![0f32; vecs];
    let mut dg = vec![0f32; rows];
    let mut dbeta = vec![0f32; rows];
    let mut dstate = vec![0f32; pairs * state_words];
    for cj in (0..n_chunks).rev() {
        let t0 = cj * chunk;
        let clen = chunk.min(t - t0);
        let io = chunk_io(q, k, v, g_exp, beta, b, t, h, t0, clen);
        let (states, _, _) = run_fwd_states(ctx, &io, &checkpoints[cj], b, h)?;
        let dout_slice = slice_rows(dout, b, t, h, HEAD_DIM, t0, clen);
        let (cdq, cdk, cdv, cdg, cdb) =
            run_bwd_chunk(ctx, &io, &dout_slice, &states, &mut dstate, b, h)?;
        scatter_rows(&mut dq, &cdq, b, t, h, HEAD_DIM, t0, clen);
        scatter_rows(&mut dk, &cdk, b, t, h, HEAD_DIM, t0, clen);
        scatter_rows(&mut dv, &cdv, b, t, h, HEAD_DIM, t0, clen);
        scatter_rows(&mut dg, &cdg, b, t, h, 1, t0, clen);
        scatter_rows(&mut dbeta, &cdb, b, t, h, 1, t0, clen);
    }

    Ok(GdnChunkedGrads {
        out,
        dq,
        dk,
        dv,
        dg,
        dbeta,
        final_state,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_module_stays_inside_the_measured_browser_caps() {
        assert!(
            !WGSL.contains("subgroup"),
            "no subgroup ops, no directive split needed"
        );
        assert!(!WGSL.contains("enable "), "no enable directives");
        assert!(!WGSL.contains("f16"), "f32 only");
        assert_eq!(WGSL.matches("@workgroup_size(128)").count(), 2);
        let storage_bindings = WGSL.matches("var<storage").count();
        assert_eq!(
            storage_bindings, 10,
            "10 storage declarations total; each entry references at most 9, under the \
             measured maxStorageBuffersPerShaderStage=10"
        );
        let wg_bytes = 3 * 128 * 4 + 2 * 128 * 16 + 128 * 8;
        assert_eq!(WGSL.matches("var<workgroup>").count(), 6);
        assert!(
            wg_bytes <= 32768,
            "workgroup memory {wg_bytes} must fit the measured 32768-byte browser limit"
        );
    }

    #[test]
    fn both_entries_are_declared() {
        let s = compose(WGSL);
        assert!(s.contains("fn gdn_chunk_fwd_states_f32("));
        assert!(s.contains("fn gdn_chunk_bwd_f32("));
    }
}
