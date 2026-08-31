#![cfg(feature = "cuda")]

mod common;
use common::assert_u16_bits;
use common::dtoh_u16;
use common::htod_f32;
use common::htod_u16;
use common::lcg_unit_f32 as lcg;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;

const Q38_NK: usize = 16;
const Q38_NV: usize = 48;
const Q38_DK: usize = 128;
const Q38_DV: usize = 128;
const Q38_CONV_DIM: usize = 10240;
const Q38_CONV_K: usize = 4;
const TWO_BRANCH_T6_PARENTS: [i32; 6] = [-1, 0, 1, 2, 0, 4];
const TWO_BRANCH_A_PATH_ROWS: [usize; 4] = [0, 1, 2, 3];
const TWO_BRANCH_B_PATH_ROWS: [usize; 3] = [0, 4, 5];
const LINEAR_T_CASES: [usize; 3] = [2, 6, 16];

fn gated() -> bool {
    std::env::var("NV_GDN_CHUNK_TEST").as_deref() == Ok("1")
}

fn stream_or_skip(test: &str) -> Option<Arc<CudaStream>> {
    if !gated() {
        eprintln!("{test}: SKIP set NV_GDN_CHUNK_TEST=1 to run");
        return None;
    }
    match CudaContext::new(0) {
        Ok(c) => Some(c.default_stream()),
        Err(e) => panic!("{test}: NV_GDN_CHUNK_TEST=1 set but no CUDA device 0: {e}"),
    }
}

fn rand_bf16(seed: &mut u64, n: usize, lo: f32, hi: f32) -> Vec<u16> {
    (0..n)
        .map(|_| bf16::from_f32(lo + lcg(seed) * (hi - lo)).to_bits())
        .collect()
}

fn rand_f32_nonzero(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let mut v = (lcg(seed) * 2.0 - 1.0) * scale;
            if v == 0.0 {
                v = 0.03125 * scale;
            }
            v
        })
        .collect()
}

fn assert_f32_bits(name: &str, a: &[f32], b: &[f32]) {
    let diff = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    assert_eq!(diff, 0, "{name}: {diff}/{} f32 words differ", a.len());
}

fn dtoh_f32(stream: &Arc<CudaStream>, d: &CudaSlice<f32>) -> Vec<f32> {
    #[allow(deprecated)]
    let v = stream.memcpy_dtov(d).unwrap();
    v
}

fn htod_i32(stream: &Arc<CudaStream>, v: &[i32]) -> CudaSlice<i32> {
    #[allow(deprecated)]
    let d = stream.clone_htod(v).unwrap();
    d
}

struct ConvOut {
    y: Vec<u16>,
    ckpt: Vec<u16>,
}

#[allow(clippy::too_many_arguments)]
fn run_conv(
    stream: &Arc<CudaStream>,
    x_seq: &[u16],
    conv0: &[u16],
    w: &[u16],
    parents: Option<&[i32]>,
    conv_dim: usize,
    k: usize,
    t: usize,
) -> ConvOut {
    let raw = stream.cu_stream() as *mut c_void;
    let d_x = htod_u16(stream, x_seq);
    let d_conv = htod_u16(stream, conv0);
    let d_w = htod_u16(stream, w);
    let mut d_y: CudaSlice<u16> = stream.alloc_zeros::<u16>(t * conv_dim).unwrap();
    let mut d_ckpt: CudaSlice<u16> = stream.alloc_zeros::<u16>(t * conv_dim * (k - 1)).unwrap();
    let d_par = parents.map(|p| htod_i32(stream, p));
    let rc = {
        let (px, _1) = d_x.device_ptr(stream);
        let (pc, _2) = d_conv.device_ptr(stream);
        let (pw, _3) = d_w.device_ptr(stream);
        let (py, _4) = d_y.device_ptr_mut(stream);
        let (pk, _5) = d_ckpt.device_ptr_mut(stream);
        match &d_par {
            Some(dp) => {
                let (pp, _6) = dp.device_ptr(stream);
                unsafe {
                    cuda::gdn_conv_decode_chunk_silu_tree_bf16(
                        raw,
                        px as *const u16,
                        pc as *const u16,
                        pw as *const u16,
                        py as *mut u16,
                        pk as *mut u16,
                        pp as *const i32,
                        conv_dim as i32,
                        k as i32,
                        t as i32,
                    )
                }
            }
            None => unsafe {
                cuda::gdn_conv_decode_chunk_silu_bf16(
                    raw,
                    px as *const u16,
                    pc as *const u16,
                    pw as *const u16,
                    py as *mut u16,
                    pk as *mut u16,
                    conv_dim as i32,
                    k as i32,
                    t as i32,
                )
            },
        }
    };
    assert_eq!(rc, 0, "conv rc={rc} tree={}", parents.is_some());
    stream.synchronize().unwrap();
    ConvOut {
        y: dtoh_u16(stream, &d_y),
        ckpt: dtoh_u16(stream, &d_ckpt),
    }
}

struct SplitFixture {
    mixed: Vec<u16>,
    z: Vec<u16>,
    a: Vec<u16>,
    b: Vec<u16>,
    a_log: Vec<u16>,
    dt_bias: Vec<u16>,
    norm_w: Vec<u16>,
    state0: Vec<f32>,
}

fn split_fixture(t: usize) -> SplitFixture {
    let key_dim = Q38_NK * Q38_DK;
    let value_dim = Q38_NV * Q38_DV;
    let mixed_stride = 2 * key_dim + value_dim;
    let mut seed = 0x71ee_5eed_u64 ^ (t as u64);
    SplitFixture {
        mixed: rand_bf16(&mut seed, t * mixed_stride, -1.0, 1.0),
        z: rand_bf16(&mut seed, t * value_dim, -2.0, 2.0),
        a: rand_bf16(&mut seed, t * Q38_NV, -2.0, 2.0),
        b: rand_bf16(&mut seed, t * Q38_NV, -3.0, 3.0),
        a_log: rand_bf16(&mut seed, Q38_NV, -3.0, 0.5),
        dt_bias: rand_bf16(&mut seed, Q38_NV, -1.0, 1.0),
        norm_w: rand_bf16(&mut seed, Q38_DV, 0.5, 1.5),
        state0: rand_f32_nonzero(&mut seed, Q38_NV * Q38_DK * Q38_DV, 0.5),
    }
}

struct SplitOut {
    out: Vec<u16>,
    ckpt: Vec<f32>,
    live: Vec<f32>,
}

fn run_split(
    stream: &Arc<CudaStream>,
    fx: &SplitFixture,
    parents: Option<&[i32]>,
    t: usize,
) -> SplitOut {
    let raw = stream.cu_stream() as *mut c_void;
    let key_dim = Q38_NK * Q38_DK;
    let value_dim = Q38_NV * Q38_DV;
    let state_len = Q38_NV * Q38_DK * Q38_DV;
    let d_mixed = htod_u16(stream, &fx.mixed);
    let d_z = htod_u16(stream, &fx.z);
    let d_a = htod_u16(stream, &fx.a);
    let d_b = htod_u16(stream, &fx.b);
    let d_alog = htod_u16(stream, &fx.a_log);
    let d_dt = htod_u16(stream, &fx.dt_bias);
    let d_nw = htod_u16(stream, &fx.norm_w);
    let d_state = htod_f32(stream, &fx.state0);
    let mut d_ckpt: CudaSlice<f32> = stream.alloc_zeros::<f32>(t * state_len).unwrap();
    let mut d_live: CudaSlice<f32> = stream.alloc_zeros::<f32>(state_len).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(t * value_dim).unwrap();
    let mut d_qn: CudaSlice<f32> = stream.alloc_zeros::<f32>(t * key_dim).unwrap();
    let mut d_kn: CudaSlice<f32> = stream.alloc_zeros::<f32>(t * key_dim).unwrap();
    let mut d_ge: CudaSlice<f32> = stream.alloc_zeros::<f32>(t * Q38_NV).unwrap();
    let mut d_be: CudaSlice<f32> = stream.alloc_zeros::<f32>(t * Q38_NV).unwrap();
    let mut d_core: CudaSlice<u16> = stream.alloc_zeros::<u16>(t * value_dim).unwrap();
    let d_par = parents.map(|p| htod_i32(stream, p));
    let rc = {
        let (pm, _1) = d_mixed.device_ptr(stream);
        let (pz, _2) = d_z.device_ptr(stream);
        let (pa, _3) = d_a.device_ptr(stream);
        let (pb, _4) = d_b.device_ptr(stream);
        let (pl, _5) = d_alog.device_ptr(stream);
        let (pd, _6) = d_dt.device_ptr(stream);
        let (pn, _7) = d_nw.device_ptr(stream);
        let (ps, _8) = d_state.device_ptr(stream);
        let (pck, _9) = d_ckpt.device_ptr_mut(stream);
        let (plv, _10) = d_live.device_ptr_mut(stream);
        let (po, _11) = d_out.device_ptr_mut(stream);
        let (pq, _12) = d_qn.device_ptr_mut(stream);
        let (pk, _13) = d_kn.device_ptr_mut(stream);
        let (pg, _14) = d_ge.device_ptr_mut(stream);
        let (pbe, _15) = d_be.device_ptr_mut(stream);
        let (pc, _16) = d_core.device_ptr_mut(stream);
        match &d_par {
            Some(dp) => {
                let (pp, _17) = dp.device_ptr(stream);
                unsafe {
                    cuda::gdn_decode_chunk_split_tree_bf16(
                        raw,
                        pm as *const u16,
                        pz as *const u16,
                        pa as *const u16,
                        pb as *const u16,
                        pl as *const u16,
                        pd as *const u16,
                        pn as *const u16,
                        ps as *const f32,
                        pck as *mut f32,
                        plv as *mut f32,
                        po as *mut u16,
                        pq as *mut f32,
                        pk as *mut f32,
                        pg as *mut f32,
                        pbe as *mut f32,
                        pc as *mut u16,
                        pp as *const i32,
                        Q38_NK as i32,
                        Q38_NV as i32,
                        Q38_DK as i32,
                        Q38_DV as i32,
                        1e-6f32,
                        t as i32,
                        Q38_NV as i32,
                    )
                }
            }
            None => unsafe {
                cuda::gdn_decode_chunk_split_bf16(
                    raw,
                    pm as *const u16,
                    pz as *const u16,
                    pa as *const u16,
                    pb as *const u16,
                    pl as *const u16,
                    pd as *const u16,
                    pn as *const u16,
                    ps as *const f32,
                    pck as *mut f32,
                    plv as *mut f32,
                    po as *mut u16,
                    pq as *mut f32,
                    pk as *mut f32,
                    pg as *mut f32,
                    pbe as *mut f32,
                    pc as *mut u16,
                    Q38_NK as i32,
                    Q38_NV as i32,
                    Q38_DK as i32,
                    Q38_DV as i32,
                    1e-6f32,
                    t as i32,
                    Q38_NV as i32,
                )
            },
        }
    };
    assert_eq!(rc, 0, "split rc={rc} tree={}", parents.is_some());
    stream.synchronize().unwrap();
    SplitOut {
        out: dtoh_u16(stream, &d_out),
        ckpt: dtoh_f32(stream, &d_ckpt),
        live: dtoh_f32(stream, &d_live),
    }
}

fn gather_rows<T: Copy>(src: &[T], rows: &[usize], row_len: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(rows.len() * row_len);
    for &r in rows {
        out.extend_from_slice(&src[r * row_len..(r + 1) * row_len]);
    }
    out
}

fn split_fixture_path(fx: &SplitFixture, rows: &[usize]) -> SplitFixture {
    let key_dim = Q38_NK * Q38_DK;
    let value_dim = Q38_NV * Q38_DV;
    let mixed_stride = 2 * key_dim + value_dim;
    SplitFixture {
        mixed: gather_rows(&fx.mixed, rows, mixed_stride),
        z: gather_rows(&fx.z, rows, value_dim),
        a: gather_rows(&fx.a, rows, Q38_NV),
        b: gather_rows(&fx.b, rows, Q38_NV),
        a_log: fx.a_log.clone(),
        dt_bias: fx.dt_bias.clone(),
        norm_w: fx.norm_w.clone(),
        state0: fx.state0.clone(),
    }
}

#[test]
#[ignore]
fn gdn_tree_kernels_with_linear_parents_are_bitwise_the_chain_kernels() {
    let Some(stream) = stream_or_skip("gdn_tree_kernels_with_linear_parents") else {
        return;
    };
    for t in LINEAR_T_CASES {
        let parents: Vec<i32> = (0..t as i32).map(|i| i - 1).collect();
        let mut seed = 0xdead_0001_u64 ^ (t as u64);
        let x_seq = rand_bf16(&mut seed, t * Q38_CONV_DIM, -1.0, 1.0);
        let conv0 = rand_bf16(&mut seed, Q38_CONV_DIM * (Q38_CONV_K - 1), -1.0, 1.0);
        let w = rand_bf16(&mut seed, Q38_CONV_DIM * Q38_CONV_K, -0.5, 0.5);
        let chain = run_conv(&stream, &x_seq, &conv0, &w, None, Q38_CONV_DIM, Q38_CONV_K, t);
        let tree = run_conv(
            &stream,
            &x_seq,
            &conv0,
            &w,
            Some(&parents),
            Q38_CONV_DIM,
            Q38_CONV_K,
            t,
        );
        assert_u16_bits(&format!("conv y t={t}"), &tree.y, &chain.y);
        assert_u16_bits(&format!("conv ckpt t={t}"), &tree.ckpt, &chain.ckpt);

        let fx = split_fixture(t);
        let chain = run_split(&stream, &fx, None, t);
        let tree = run_split(&stream, &fx, Some(&parents), t);
        assert_u16_bits(&format!("split out t={t}"), &tree.out, &chain.out);
        assert_f32_bits(&format!("split ckpt t={t}"), &tree.ckpt, &chain.ckpt);
        assert_f32_bits(&format!("split live t={t}"), &tree.live, &chain.live);
        let nonzero = tree.out.iter().filter(|w| **w != 0).count();
        assert!(
            nonzero * 2 > tree.out.len(),
            "tree output mostly zero: {nonzero}/{}",
            tree.out.len()
        );
        eprintln!("GDN-TREE dead-branch t={t}: conv+split bitwise the chain kernels");
    }
}

#[test]
#[ignore]
fn gdn_tree_kernels_fork_each_branch_bitwise_from_the_parents_checkpoint() {
    let Some(stream) = stream_or_skip("gdn_tree_kernels_fork_each_branch") else {
        return;
    };
    let t = TWO_BRANCH_T6_PARENTS.len();
    let conv_row = Q38_CONV_DIM * (Q38_CONV_K - 1);
    let mut seed = 0xfeed_0002_u64;
    let x_seq = rand_bf16(&mut seed, t * Q38_CONV_DIM, -1.0, 1.0);
    let conv0 = rand_bf16(&mut seed, conv_row, -1.0, 1.0);
    let w = rand_bf16(&mut seed, Q38_CONV_DIM * Q38_CONV_K, -0.5, 0.5);
    let tree = run_conv(
        &stream,
        &x_seq,
        &conv0,
        &w,
        Some(&TWO_BRANCH_T6_PARENTS),
        Q38_CONV_DIM,
        Q38_CONV_K,
        t,
    );
    let fx = split_fixture(t);
    let tree_split = run_split(&stream, &fx, Some(&TWO_BRANCH_T6_PARENTS), t);

    let value_dim = Q38_NV * Q38_DV;
    let state_len = Q38_NV * Q38_DK * Q38_DV;
    for (branch, path) in [
        ("A", &TWO_BRANCH_A_PATH_ROWS[..]),
        ("B", &TWO_BRANCH_B_PATH_ROWS[..]),
    ] {
        let x_path = gather_rows(&x_seq, path, Q38_CONV_DIM);
        let conv_ref = run_conv(
            &stream,
            &x_path,
            &conv0,
            &w,
            None,
            Q38_CONV_DIM,
            Q38_CONV_K,
            path.len(),
        );
        let fx_path = split_fixture_path(&fx, path);
        let split_ref = run_split(&stream, &fx_path, None, path.len());
        for (j, &row) in path.iter().enumerate() {
            assert_u16_bits(
                &format!("branch {branch} conv y row {row}"),
                &tree.y[row * Q38_CONV_DIM..(row + 1) * Q38_CONV_DIM],
                &conv_ref.y[j * Q38_CONV_DIM..(j + 1) * Q38_CONV_DIM],
            );
            assert_u16_bits(
                &format!("branch {branch} conv ckpt row {row}"),
                &tree.ckpt[row * conv_row..(row + 1) * conv_row],
                &conv_ref.ckpt[j * conv_row..(j + 1) * conv_row],
            );
            assert_u16_bits(
                &format!("branch {branch} split out row {row}"),
                &tree_split.out[row * value_dim..(row + 1) * value_dim],
                &split_ref.out[j * value_dim..(j + 1) * value_dim],
            );
            assert_f32_bits(
                &format!("branch {branch} split ckpt row {row}"),
                &tree_split.ckpt[row * state_len..(row + 1) * state_len],
                &split_ref.ckpt[j * state_len..(j + 1) * state_len],
            );
        }
        eprintln!(
            "GDN-TREE fork branch {branch} (rows {path:?}): every row bitwise the re-chained \
             reference"
        );
    }
}
