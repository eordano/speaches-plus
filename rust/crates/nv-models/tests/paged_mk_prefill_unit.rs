#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use cudarc::driver::CudaSlice;
use half::bf16;
use nv_models::paged_fp8::{
    flash_scratch_elems_for, LayerKvGeometry, PagedKvFp8Pool, PagedPoolConfig,
};

const N_KV: usize = 4;
const N_Q: usize = 32;
const HEAD_DIM: usize = 256;
const BLOCK_SIZE: usize = 16;
const N_BLOCKS: usize = 8;
const N_TOKENS: usize = 40;

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn ring_cfg(ring_blocks: usize) -> PagedPoolConfig {
    PagedPoolConfig {
        num_blocks: N_BLOCKS,
        block_size: BLOCK_SIZE,
        layers: vec![LayerKvGeometry {
            n_kv: N_KV,
            head_dim: HEAD_DIM,
        }],
        layer_blocks: vec![ring_blocks],
        layer_sliding: vec![true],
        lanes: 1,
        sliding_ring_blocks: ring_blocks,
    }
}

fn cfg() -> PagedPoolConfig {
    PagedPoolConfig {
        num_blocks: N_BLOCKS,
        block_size: BLOCK_SIZE,
        layers: vec![LayerKvGeometry {
            n_kv: N_KV,
            head_dim: HEAD_DIM,
        }],
        layer_blocks: vec![N_BLOCKS],
        layer_sliding: vec![false],
        lanes: 0,
        sliding_ring_blocks: 0,
    }
}

fn rand_kv(rng: &mut Lcg, n: usize, device: &Device) -> Tensor {
    let data: Vec<f32> = (0..n * N_KV * HEAD_DIM).map(|_| rng.next_f32()).collect();
    Tensor::from_vec(data, (1usize, n, N_KV, HEAD_DIM), device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

fn dev_i32(stream: &std::sync::Arc<cudarc::driver::CudaStream>, v: &[i32]) -> CudaSlice<i32> {
    stream.memcpy_stod(v).expect("htod")
}

#[test]
#[ignore]
fn mk_prefill_with_one_row_equals_the_single_query_decode() {
    if std::env::var("NV_PAGED_MK_UNIT").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_MK_UNIT=1");
    }
    let Ok(device) = Device::new_cuda(0) else {
        panic!("PRECONDITION NOT MET: no CUDA device");
    };
    let candle_dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = nv_layers::cuda_stream::current_stream(&candle_dev);

    let mut pool = PagedKvFp8Pool::new(cfg(), &device).expect("pool");
    let table: Vec<i32> = (0..(N_BLOCKS as i32)).collect();
    let table_dev = dev_i32(&stream, &table);
    let start_dev = dev_i32(&stream, &[0]);
    let n_total_dev = dev_i32(&stream, &[N_TOKENS as i32]);

    let mut rng = Lcg(0x51ded_u64 | 1);
    let k = rand_kv(&mut rng, N_TOKENS, &device);
    let v = rand_kv(&mut rng, N_TOKENS, &device);
    pool.append_layer(0, &k, &v, N_TOKENS, &start_dev, &table_dev)
        .expect("append");

    let qdata: Vec<f32> = (0..N_Q * HEAD_DIM).map(|_| rng.next_f32()).collect();
    let q = Tensor::from_vec(qdata, (1usize, 1usize, N_Q, HEAD_DIM), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let q_c = q.contiguous().unwrap();
    let (q_storage, q_l) = q_c.storage_and_layout();
    let q_cuda = match &*q_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => unreachable!(),
    };
    let q_slice = q_cuda.as_cuda_slice::<bf16>().unwrap();

    let need_decode = flash_scratch_elems_for(N_Q, HEAD_DIM);
    let need_mk = nv_kernels::cuda::flash_splitk_scratch_elems_mk(N_Q as i32, HEAD_DIM as i32, 8);
    let mut scratch = stream
        .alloc_zeros::<f32>(need_decode.max(need_mk))
        .expect("scratch");
    let mut fan_in = stream.alloc_zeros::<u32>(N_Q).expect("fan_in");

    let mut out_decode = stream.alloc_zeros::<bf16>(N_Q * HEAD_DIM).expect("out d");
    pool.decode_attention_paged_into(
        0,
        q_slice,
        q_l.start_offset(),
        &mut out_decode,
        0,
        N_Q,
        &table_dev,
        BLOCK_SIZE,
        &n_total_dev,
        &mut scratch,
        &mut fan_in,
        None,
        1.0,
    )
    .expect("decode path");

    let mut out_mk = stream.alloc_zeros::<bf16>(N_Q * HEAD_DIM).expect("out mk");
    pool.prefill_attention_paged_into(
        0,
        q_slice,
        q_l.start_offset(),
        &mut out_mk,
        0,
        1,
        N_Q,
        &table_dev,
        BLOCK_SIZE,
        &n_total_dev,
        &mut scratch,
        &mut fan_in,
        None,
        1.0,
    )
    .expect("mk path");

    let d: Vec<bf16> = stream.clone_dtoh(&out_decode).expect("dtoh d");
    let m: Vec<bf16> = stream.clone_dtoh(&out_mk).expect("dtoh mk");
    let diffs: Vec<usize> = (0..d.len()).filter(|i| d[*i] != m[*i]).collect();
    let worst = (0..d.len())
        .map(|i| (d[i].to_f32() - m[i].to_f32()).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[mk-unit] {} of {} outputs differ, worst |decode - mk| = {worst:e}",
        diffs.len(),
        d.len()
    );
    assert!(
        worst < 1e-2,
        "the mk kernel at M=1 disagrees with the single-query kernel on the same state: {} of {} \
         elements differ, worst {worst:e}. delta is 0 and there is one tile, so the per-tile \
         bookkeeping is not involved -- this is the binding, the scratch, or the kernel itself",
        diffs.len(),
        d.len()
    );
}

#[test]
#[ignore]
fn mk_prefill_tiles_match_a_single_query_decode_at_every_row() {
    for window in [None, Some(24usize)] {
        tiles_match_at(window);
    }
}

fn tiles_match_at(window: Option<usize>) {
    if std::env::var("NV_PAGED_MK_UNIT").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_MK_UNIT=1");
    }
    const SEQ: usize = 20;
    let Ok(device) = Device::new_cuda(0) else {
        panic!("PRECONDITION NOT MET: no CUDA device");
    };
    let candle_dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = nv_layers::cuda_stream::current_stream(&candle_dev);

    let mut pool = PagedKvFp8Pool::new(cfg(), &device).expect("pool");
    let table: Vec<i32> = (0..(N_BLOCKS as i32)).collect();
    let table_dev = dev_i32(&stream, &table);
    let start_dev = dev_i32(&stream, &[0]);
    let n_total_dev = dev_i32(&stream, &[N_TOKENS as i32]);

    let mut rng = Lcg(0xa11ce_u64 | 1);
    let k = rand_kv(&mut rng, N_TOKENS, &device);
    let v = rand_kv(&mut rng, N_TOKENS, &device);
    pool.append_layer(0, &k, &v, N_TOKENS, &start_dev, &table_dev)
        .expect("append");

    let qdata: Vec<f32> = (0..SEQ * N_Q * HEAD_DIM).map(|_| rng.next_f32()).collect();
    let q = Tensor::from_vec(qdata, (1usize, SEQ, N_Q, HEAD_DIM), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let q_c = q.contiguous().unwrap();
    let (q_storage, q_l) = q_c.storage_and_layout();
    let q_cuda = match &*q_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => unreachable!(),
    };
    let q_slice = q_cuda.as_cuda_slice::<bf16>().unwrap();

    let need_decode = flash_scratch_elems_for(N_Q, HEAD_DIM);
    let need_mk = nv_kernels::cuda::flash_splitk_scratch_elems_mk(N_Q as i32, HEAD_DIM as i32, 8);
    let mut scratch = stream
        .alloc_zeros::<f32>(need_decode.max(need_mk))
        .expect("scratch");
    let mut fan_in = stream.alloc_zeros::<u32>(N_Q).expect("fan_in");

    let row = N_Q * HEAD_DIM;
    let mut out_mk = stream.alloc_zeros::<bf16>(SEQ * row).expect("out mk");
    pool.prefill_attention_paged_into(
        0,
        q_slice,
        q_l.start_offset(),
        &mut out_mk,
        0,
        SEQ,
        N_Q,
        &table_dev,
        BLOCK_SIZE,
        &n_total_dev,
        &mut scratch,
        &mut fan_in,
        window,
        1.0,
    )
    .expect("mk path");
    let got: Vec<bf16> = stream.clone_dtoh(&out_mk).expect("dtoh mk");

    let mut worst = 0.0f32;
    let mut worst_row = 0usize;
    for j in 0..SEQ {

        let n_total = (N_TOKENS - SEQ + j + 1) as i32;
        let nt = dev_i32(&stream, &[n_total]);
        let mut one = stream.alloc_zeros::<bf16>(row).expect("out one");
        pool.decode_attention_paged_into(
            0,
            q_slice,
            q_l.start_offset() + j * row,
            &mut one,
            0,
            N_Q,
            &table_dev,
            BLOCK_SIZE,
            &nt,
            &mut scratch,
            &mut fan_in,
            window,
            1.0,
        )
        .expect("decode path");
        let want: Vec<bf16> = stream.clone_dtoh(&one).expect("dtoh one");
        for i in 0..row {
            let d = (want[i].to_f32() - got[j * row + i].to_f32()).abs();
            if d > worst {
                worst = d;
                worst_row = j;
            }
        }
    }
    eprintln!("[mk-tiles] window {window:?}, SEQ {SEQ}, worst |decode - mk| = {worst:e} at row {worst_row}");
    assert!(
        worst < 1e-2,
        "tiled mk prefill disagrees with a per-row single-query decode at window {window:?}: \
         worst {worst:e} at row {worst_row} of {SEQ}"
    );
}

#[test]
#[ignore]
fn the_mk_launcher_refuses_a_head_dim_past_what_its_kernel_holds() {
    if std::env::var("NV_PAGED_MK_UNIT").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_MK_UNIT=1");
    }
    let Ok(device) = Device::new_cuda(0) else {
        panic!("PRECONDITION NOT MET: no CUDA device");
    };
    let candle_dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = nv_layers::cuda_stream::current_stream(&candle_dev);
    let over = 512i32;
    let mut scratch = stream.alloc_zeros::<f32>(1 << 16).expect("scratch");
    let mut fan_in = stream.alloc_zeros::<u32>(N_Q).expect("fan_in");
    let junk_u8 = stream.alloc_zeros::<u8>(1 << 12).expect("u8");
    let junk_f32 = stream.alloc_zeros::<f32>(1 << 10).expect("f32");
    let junk_u16 = stream.alloc_zeros::<bf16>(1 << 12).expect("u16");
    let nt = dev_i32(&stream, &[1]);
    let tbl = dev_i32(&stream, &[0]);
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let (q, _a) = junk_u16.device_ptr(&stream);
    let (k, _b) = junk_u8.device_ptr(&stream);
    let (sc, _c) = junk_f32.device_ptr(&stream);
    let (ntp, _d) = nt.device_ptr(&stream);
    let (tp, _e) = tbl.device_ptr(&stream);
    let mut out = stream.alloc_zeros::<bf16>(1 << 12).expect("out");
    let (op, _f) = out.device_ptr_mut(&stream);
    let (scr, _g) = scratch.device_ptr_mut(&stream);
    let (fi, _h) = fan_in.device_ptr_mut(&stream);
    let rc = unsafe {
        nv_kernels::cuda::flash_decode_fused_fp8kv_mk_paged(
            stream.cu_stream() as *mut std::ffi::c_void,
            q as *const u16,
            k as *const u8,
            k as *const u8,
            sc as *const f32,
            sc as *const f32,
            op as *mut u16,
            ntp as *const i32,
            0,
            1,
            scr as *mut f32,
            fi as *mut u32,
            N_Q as i32,
            N_KV as i32,
            over,
            0,
            0,
            1.0,
            tp as *const i32,
            BLOCK_SIZE as i32,
        )
    };
    assert_ne!(
        rc, 0,
        "the paged mk launcher accepted head_dim {over} when splitk_mk holds only \
         kMaxHDmk = 256, so it would overrun shared qsh and the acc array instead of \
         declining the launch"
    );
}

#[test]
#[ignore]
fn mk_prefill_matches_per_row_decode_on_a_wrapped_ring() {
    if std::env::var("NV_PAGED_MK_UNIT").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_MK_UNIT=1");
    }
    const RING_BLOCKS: usize = 2;
    const WINDOW: usize = 24;
    const SEQ: usize = 12;
    let ring_slots = RING_BLOCKS * BLOCK_SIZE;
    assert!(
        N_TOKENS > ring_slots,
        "{N_TOKENS} tokens must exceed the {ring_slots}-slot ring or nothing wraps"
    );
    let Ok(device) = Device::new_cuda(0) else {
        panic!("PRECONDITION NOT MET: no CUDA device");
    };
    let candle_dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = nv_layers::cuda_stream::current_stream(&candle_dev);

    let mut pool = PagedKvFp8Pool::new(ring_cfg(RING_BLOCKS), &device).expect("pool");

    let rows = N_TOKENS.div_ceil(BLOCK_SIZE);
    let table: Vec<i32> = (0..rows).map(|j| (j % RING_BLOCKS) as i32).collect();
    let table_dev = dev_i32(&stream, &table);
    let n_total_dev = dev_i32(&stream, &[N_TOKENS as i32]);

    let mut rng = Lcg(0x21b6_u64 | 1);
    for t in 0..N_TOKENS {
        let k = rand_kv(&mut rng, 1, &device);
        let v = rand_kv(&mut rng, 1, &device);
        let start = dev_i32(&stream, &[t as i32]);
        pool.append_layer(0, &k, &v, 1, &start, &table_dev)
            .expect("append");
    }

    let qdata: Vec<f32> = (0..SEQ * N_Q * HEAD_DIM).map(|_| rng.next_f32()).collect();
    let q = Tensor::from_vec(qdata, (1usize, SEQ, N_Q, HEAD_DIM), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let q_c = q.contiguous().unwrap();
    let (q_storage, q_l) = q_c.storage_and_layout();
    let q_cuda = match &*q_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => unreachable!(),
    };
    let q_slice = q_cuda.as_cuda_slice::<bf16>().unwrap();

    let need = nv_kernels::cuda::flash_splitk_scratch_elems_mk(N_Q as i32, HEAD_DIM as i32, 8)
        .max(flash_scratch_elems_for(N_Q, HEAD_DIM));
    let mut scratch = stream.alloc_zeros::<f32>(need).expect("scratch");
    let mut fan_in = stream.alloc_zeros::<u32>(N_Q).expect("fan_in");

    let row = N_Q * HEAD_DIM;
    let mut out_mk = stream.alloc_zeros::<bf16>(SEQ * row).expect("out mk");
    pool.prefill_attention_paged_into(
        0,
        q_slice,
        q_l.start_offset(),
        &mut out_mk,
        0,
        SEQ,
        N_Q,
        &table_dev,
        BLOCK_SIZE,
        &n_total_dev,
        &mut scratch,
        &mut fan_in,
        Some(WINDOW),
        1.0,
    )
    .expect("mk path");
    let got: Vec<bf16> = stream.clone_dtoh(&out_mk).expect("dtoh mk");

    let mut worst = 0.0f32;
    let mut worst_row = 0usize;
    for j in 0..SEQ {
        let nt = dev_i32(&stream, &[(N_TOKENS - SEQ + j + 1) as i32]);
        let mut one = stream.alloc_zeros::<bf16>(row).expect("out one");
        pool.decode_attention_paged_into(
            0,
            q_slice,
            q_l.start_offset() + j * row,
            &mut one,
            0,
            N_Q,
            &table_dev,
            BLOCK_SIZE,
            &nt,
            &mut scratch,
            &mut fan_in,
            Some(WINDOW),
            1.0,
        )
        .expect("decode path");
        let want: Vec<bf16> = stream.clone_dtoh(&one).expect("dtoh one");
        for i in 0..row {
            let d = (want[i].to_f32() - got[j * row + i].to_f32()).abs();
            if d > worst {
                worst = d;
                worst_row = j;
            }
        }
    }
    eprintln!("[mk-ring] ring {ring_slots} slots, window {WINDOW}, worst {worst:e} at row {worst_row}");
    assert!(
        worst < 1e-2,
        "on a WRAPPED ring the tiled mk prefill disagrees with a per-row decode: worst {worst:e} \
         at row {worst_row} of {SEQ}. The non-hybrid checks are bit-identical, so the wrap is \
         what they do not share"
    );
}
