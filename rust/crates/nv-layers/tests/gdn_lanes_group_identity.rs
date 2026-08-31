#![cfg(feature = "cuda")]

use candle_core::Device;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;

const WHY: &str = "NV_Q38_GDN_LANES_GROUP replaces the batch GDN's per-lane conv+step launch \
                   pair with one conv launch and one step launch whose grid.y is the lane. That \
                   is only allowed to ship as a launch-count change if lane j of the group call \
                   writes the SAME BITS as call j of the loop it replaces, in the gated output \
                   AND in the conv and recurrent state it leaves behind, because the recurrent \
                   state is carried into every future step and a one-ulp difference there \
                   compounds without bound. This suite is the certificate; the arm's constant \
                   cites it by name.";

const THE_QWEN38_27B_GDN_DECODE_SHAPE: GdnShape = GdnShape {
    n_k: 16,
    n_v: 48,
    d_k: 128,
    d_v: 128,
    k: 4,
};

#[derive(Clone, Copy)]
struct GdnShape {
    n_k: usize,
    n_v: usize,
    d_k: usize,
    d_v: usize,
    k: usize,
}

impl GdnShape {
    fn key_dim(&self) -> usize {
        self.n_k * self.d_k
    }
    fn value_dim(&self) -> usize {
        self.n_v * self.d_v
    }
    fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
    fn state_elems(&self) -> usize {
        self.n_v * self.d_k * self.d_v
    }
}

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 33) & 0x7f_ffff) as f32 / 8388608.0) * 2.0 - 1.0
    }
    fn bits(&mut self, n: usize, gain: f32) -> Vec<u16> {
        (0..n)
            .map(|_| bf16::from_f32(self.next_f32() * gain).to_bits())
            .collect()
    }
    fn f32s(&mut self, n: usize, gain: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * gain).collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn run_per_lane_loop(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    sh: &GdnShape,
    m: usize,
    qkv: &CudaSlice<u16>,
    z: &CudaSlice<u16>,
    ab: &CudaSlice<u16>,
    a_log: &CudaSlice<u16>,
    dt_bias: &CudaSlice<u16>,
    norm_w: &CudaSlice<u16>,
    conv_w: &CudaSlice<u16>,
    conv_states: &mut [CudaSlice<u16>],
    rec_states: &mut [CudaSlice<f32>],
    mixed: &mut CudaSlice<u16>,
    gated: &mut CudaSlice<u16>,
    rms_eps: f32,
) {
    let conv_dim = sh.conv_dim();
    let value_dim = sh.value_dim();
    let ab_stride = 2 * sh.n_v;
    for lane in 0..m {
        let x_lane = qkv.slice(lane * conv_dim..(lane + 1) * conv_dim);
        let mut y_lane = mixed.slice_mut(lane * conv_dim..(lane + 1) * conv_dim);
        let rc = {
            let (xp, _a) = x_lane.device_ptr(stream);
            let (sp, _b) = conv_states[lane].device_ptr_mut(stream);
            let (wp, _c) = conv_w.device_ptr(stream);
            let (yp, _d) = y_lane.device_ptr_mut(stream);
            unsafe {
                nv_kernels::cuda::gdn_conv_decode_silu_bf16(
                    stream.cu_stream() as *mut c_void,
                    xp as *const u16,
                    sp as *mut u16,
                    wp as *const u16,
                    yp as *mut u16,
                    conv_dim as i32,
                    sh.k as i32,
                )
            }
        };
        assert_eq!(rc, 0, "gdn_conv_decode_silu_bf16 rc={rc} lane={lane}");
    }
    for lane in 0..m {
        let mixed_lane = mixed.slice(lane * conv_dim..(lane + 1) * conv_dim);
        let z_lane = z.slice(lane * value_dim..(lane + 1) * value_dim);
        let a_lane = ab.slice(lane * ab_stride..lane * ab_stride + sh.n_v);
        let b_lane = ab.slice(lane * ab_stride + sh.n_v..(lane + 1) * ab_stride);
        let mut out_lane = gated.slice_mut(lane * value_dim..(lane + 1) * value_dim);
        let rc = {
            let (mp, _a) = mixed_lane.device_ptr(stream);
            let (zp, _b) = z_lane.device_ptr(stream);
            let (ap, _c) = a_lane.device_ptr(stream);
            let (bp, _d) = b_lane.device_ptr(stream);
            let (alp, _e) = a_log.device_ptr(stream);
            let (dtp, _f) = dt_bias.device_ptr(stream);
            let (nwp, _g) = norm_w.device_ptr(stream);
            let (stp, _h) = rec_states[lane].device_ptr_mut(stream);
            let (op, _i) = out_lane.device_ptr_mut(stream);
            unsafe {
                nv_kernels::cuda::gdn_decode_step_bf16(
                    stream.cu_stream() as *mut c_void,
                    mp as *const u16,
                    zp as *const u16,
                    ap as *const u16,
                    bp as *const u16,
                    alp as *const u16,
                    dtp as *const u16,
                    nwp as *const u16,
                    stp as *mut f32,
                    op as *mut u16,
                    sh.n_k as i32,
                    sh.n_v as i32,
                    sh.d_k as i32,
                    sh.d_v as i32,
                    rms_eps,
                )
            }
        };
        assert_eq!(rc, 0, "gdn_decode_step_bf16 rc={rc} lane={lane}");
    }
}

#[allow(clippy::too_many_arguments)]
fn run_group(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    sh: &GdnShape,
    m: usize,
    qkv: &CudaSlice<u16>,
    z: &CudaSlice<u16>,
    ab: &CudaSlice<u16>,
    a_log: &CudaSlice<u16>,
    dt_bias: &CudaSlice<u16>,
    norm_w: &CudaSlice<u16>,
    conv_w: &CudaSlice<u16>,
    conv_states: &mut [CudaSlice<u16>],
    rec_states: &mut [CudaSlice<f32>],
    mixed: &mut CudaSlice<u16>,
    gated: &mut CudaSlice<u16>,
    rms_eps: f32,
) {
    let conv_dim = sh.conv_dim();
    let value_dim = sh.value_dim();
    let ab_stride = 2 * sh.n_v;
    let mut conv_ptrs: Vec<*mut u16> = Vec::with_capacity(m);
    let mut conv_guards = Vec::with_capacity(m);
    for st in conv_states.iter_mut().take(m) {
        let (p, g) = st.device_ptr_mut(stream);
        conv_ptrs.push(p as *mut u16);
        conv_guards.push(g);
    }
    let rc = {
        let (xp, _a) = qkv.device_ptr(stream);
        let (wp, _b) = conv_w.device_ptr(stream);
        let (yp, _c) = mixed.device_ptr_mut(stream);
        unsafe {
            nv_kernels::cuda::gdn_conv_decode_silu_group_bf16(
                stream.cu_stream() as *mut c_void,
                xp as *const u16,
                conv_ptrs.as_ptr(),
                wp as *const u16,
                yp as *mut u16,
                conv_dim as i32,
                sh.k as i32,
                m as i32,
                conv_dim as i32,
                conv_dim as i32,
            )
        }
    };
    assert_eq!(rc, 0, "gdn_conv_decode_silu_group_bf16 rc={rc} m={m}");
    drop(conv_guards);

    let mut rec_ptrs: Vec<*mut f32> = Vec::with_capacity(m);
    let mut rec_guards = Vec::with_capacity(m);
    for st in rec_states.iter_mut().take(m) {
        let (p, g) = st.device_ptr_mut(stream);
        rec_ptrs.push(p as *mut f32);
        rec_guards.push(g);
    }
    let rc = {
        let (mp, _a) = mixed.device_ptr(stream);
        let (zp, _b) = z.device_ptr(stream);
        let (abp, _c) = ab.device_ptr(stream);
        let (alp, _d) = a_log.device_ptr(stream);
        let (dtp, _e) = dt_bias.device_ptr(stream);
        let (nwp, _f) = norm_w.device_ptr(stream);
        let (op, _g) = gated.device_ptr_mut(stream);
        unsafe {
            nv_kernels::cuda::gdn_decode_step_group_bf16(
                stream.cu_stream() as *mut c_void,
                mp as *const u16,
                zp as *const u16,
                abp as *const u16,
                (abp + (sh.n_v * 2) as u64) as *const u16,
                alp as *const u16,
                dtp as *const u16,
                nwp as *const u16,
                rec_ptrs.as_ptr(),
                op as *mut u16,
                sh.n_k as i32,
                sh.n_v as i32,
                sh.d_k as i32,
                sh.d_v as i32,
                rms_eps,
                m as i32,
                conv_dim as i32,
                value_dim as i32,
                ab_stride as i32,
                value_dim as i32,
            )
        }
    };
    assert_eq!(rc, 0, "gdn_decode_step_group_bf16 rc={rc} m={m}");
    drop(rec_guards);
}

#[allow(clippy::too_many_arguments)]
fn run_per_lane_split(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    sh: &GdnShape,
    m: usize,
    mixed: &CudaSlice<u16>,
    z: &CudaSlice<u16>,
    ab: &CudaSlice<u16>,
    a_log: &CudaSlice<u16>,
    dt_bias: &CudaSlice<u16>,
    norm_w: &CudaSlice<u16>,
    rec_states: &mut [CudaSlice<f32>],
    gated: &mut CudaSlice<u16>,
    rms_eps: f32,
) -> i32 {
    let conv_dim = sh.conv_dim();
    let value_dim = sh.value_dim();
    let ab_stride = 2 * sh.n_v;
    let mut qn = stream.alloc_zeros::<f32>(sh.key_dim()).unwrap();
    let mut kn = stream.alloc_zeros::<f32>(sh.key_dim()).unwrap();
    let mut ge = stream.alloc_zeros::<f32>(sh.n_v).unwrap();
    let mut be = stream.alloc_zeros::<f32>(sh.n_v).unwrap();
    let mut core = stream.alloc_zeros::<u16>(value_dim).unwrap();
    for lane in 0..m {
        let mixed_lane = mixed.slice(lane * conv_dim..(lane + 1) * conv_dim);
        let z_lane = z.slice(lane * value_dim..(lane + 1) * value_dim);
        let a_lane = ab.slice(lane * ab_stride..lane * ab_stride + sh.n_v);
        let b_lane = ab.slice(lane * ab_stride + sh.n_v..(lane + 1) * ab_stride);
        let mut out_lane = gated.slice_mut(lane * value_dim..(lane + 1) * value_dim);
        let rc = {
            let (mp, _a) = mixed_lane.device_ptr(stream);
            let (zp, _b) = z_lane.device_ptr(stream);
            let (ap, _c) = a_lane.device_ptr(stream);
            let (bp, _d) = b_lane.device_ptr(stream);
            let (alp, _e) = a_log.device_ptr(stream);
            let (dtp, _f) = dt_bias.device_ptr(stream);
            let (nwp, _g) = norm_w.device_ptr(stream);
            let (stp, _h) = rec_states[lane].device_ptr_mut(stream);
            let (op, _i) = out_lane.device_ptr_mut(stream);
            let (qnp, _j) = qn.device_ptr_mut(stream);
            let (knp, _k) = kn.device_ptr_mut(stream);
            let (gep, _l) = ge.device_ptr_mut(stream);
            let (bep, _n) = be.device_ptr_mut(stream);
            let (cop, _o) = core.device_ptr_mut(stream);
            unsafe {
                nv_kernels::cuda::gdn_decode_step_split_bf16(
                    stream.cu_stream() as *mut c_void,
                    mp as *const u16,
                    zp as *const u16,
                    ap as *const u16,
                    bp as *const u16,
                    alp as *const u16,
                    dtp as *const u16,
                    nwp as *const u16,
                    stp as *mut f32,
                    op as *mut u16,
                    qnp as *mut f32,
                    knp as *mut f32,
                    gep as *mut f32,
                    bep as *mut f32,
                    cop as *mut u16,
                    sh.n_k as i32,
                    sh.n_v as i32,
                    sh.d_k as i32,
                    sh.d_v as i32,
                    rms_eps,
                )
            }
        };
        if rc != 0 {
            return rc;
        }
    }
    0
}

#[allow(clippy::too_many_arguments)]
fn run_group_split(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    sh: &GdnShape,
    m: usize,
    mixed: &CudaSlice<u16>,
    z: &CudaSlice<u16>,
    ab: &CudaSlice<u16>,
    a_log: &CudaSlice<u16>,
    dt_bias: &CudaSlice<u16>,
    norm_w: &CudaSlice<u16>,
    rec_states: &mut [CudaSlice<f32>],
    gated: &mut CudaSlice<u16>,
    rms_eps: f32,
) -> i32 {
    let value_dim = sh.value_dim();
    let ab_stride = 2 * sh.n_v;
    let mut qn = stream.alloc_zeros::<f32>(m * sh.key_dim()).unwrap();
    let mut kn = stream.alloc_zeros::<f32>(m * sh.key_dim()).unwrap();
    let mut ge = stream.alloc_zeros::<f32>(m * sh.n_v).unwrap();
    let mut be = stream.alloc_zeros::<f32>(m * sh.n_v).unwrap();
    let mut core = stream.alloc_zeros::<u16>(m * value_dim).unwrap();
    let mut rec_ptrs: Vec<*mut f32> = Vec::with_capacity(m);
    let mut rec_guards = Vec::with_capacity(m);
    for st in rec_states.iter_mut().take(m) {
        let (p, g) = st.device_ptr_mut(stream);
        rec_ptrs.push(p as *mut f32);
        rec_guards.push(g);
    }
    let rc = {
        let (mp, _a) = mixed.device_ptr(stream);
        let (zp, _b) = z.device_ptr(stream);
        let (abp, _c) = ab.device_ptr(stream);
        let (alp, _d) = a_log.device_ptr(stream);
        let (dtp, _e) = dt_bias.device_ptr(stream);
        let (nwp, _f) = norm_w.device_ptr(stream);
        let (op, _g) = gated.device_ptr_mut(stream);
        let (qnp, _h) = qn.device_ptr_mut(stream);
        let (knp, _i) = kn.device_ptr_mut(stream);
        let (gep, _j) = ge.device_ptr_mut(stream);
        let (bep, _k) = be.device_ptr_mut(stream);
        let (cop, _l) = core.device_ptr_mut(stream);
        unsafe {
            nv_kernels::cuda::gdn_decode_step_split_group_bf16(
                stream.cu_stream() as *mut c_void,
                mp as *const u16,
                zp as *const u16,
                abp as *const u16,
                (abp + (sh.n_v * 2) as u64) as *const u16,
                alp as *const u16,
                dtp as *const u16,
                nwp as *const u16,
                rec_ptrs.as_ptr(),
                op as *mut u16,
                qnp as *mut f32,
                knp as *mut f32,
                gep as *mut f32,
                bep as *mut f32,
                cop as *mut u16,
                sh.n_k as i32,
                sh.n_v as i32,
                sh.d_k as i32,
                sh.d_v as i32,
                rms_eps,
                m as i32,
                ab_stride as i32,
            )
        }
    };
    drop(rec_guards);
    rc
}

#[test]
#[ignore = "needs CUDA device 0; set NV_GDN_LANES_GROUP_TEST=1 -- proves the group-spanning GDN \
            SPLIT step leaves bit-identical gated output and recurrent state to the per-lane \
            split step it replaces, so the group form of the split is a launch-count and \
            scratch-width change and not a second arithmetic"]
fn gdn_group_split_step_is_bit_identical_to_the_per_lane_split_step_in_output_and_state() {
    if std::env::var("NV_GDN_LANES_GROUP_TEST").as_deref() != Ok("1") {
        panic!("set NV_GDN_LANES_GROUP_TEST=1 to run this GPU test (it must never silently skip)");
    }
    let device = Device::new_cuda_with_stream(0).expect("cuda device 0");
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut rng = Lcg(0x71c3_5d09_a4e2_bb17);
    let rms_eps = 1e-6f32;
    let sh = THE_QWEN38_27B_GDN_DECODE_SHAPE;
    let conv_dim = sh.conv_dim();
    let value_dim = sh.value_dim();
    let ab_stride = 2 * sh.n_v;
    let lane_counts = [1usize, 2, 3, 4, 6, 8, 12, 16];
    let m_max = *lane_counts.iter().max().unwrap();

    let mixed_bits = rng.bits(m_max * conv_dim, 1.0);
    let z_bits = rng.bits(m_max * value_dim, 1.0);
    let ab_bits = rng.bits(m_max * ab_stride, 1.0);
    let a_log_bits = rng.bits(sh.n_v, 0.5);
    let dt_bits = rng.bits(sh.n_v, 0.5);
    let nw_bits = rng.bits(sh.d_v, 1.0);
    let rs_seed: Vec<Vec<f32>> = (0..m_max)
        .map(|_| rng.f32s(sh.state_elems(), 0.25))
        .collect();

    #[allow(deprecated)]
    let mixed: CudaSlice<u16> = stream.clone_htod(&mixed_bits).unwrap();
    #[allow(deprecated)]
    let z: CudaSlice<u16> = stream.clone_htod(&z_bits).unwrap();
    #[allow(deprecated)]
    let ab: CudaSlice<u16> = stream.clone_htod(&ab_bits).unwrap();
    #[allow(deprecated)]
    let a_log: CudaSlice<u16> = stream.clone_htod(&a_log_bits).unwrap();
    #[allow(deprecated)]
    let dt_bias: CudaSlice<u16> = stream.clone_htod(&dt_bits).unwrap();
    #[allow(deprecated)]
    let norm_w: CudaSlice<u16> = stream.clone_htod(&nw_bits).unwrap();

    let mut cells = 0usize;
    for &m in lane_counts.iter() {
        let mut mk = || -> Vec<CudaSlice<f32>> {
            (0..m)
                .map(|l| {
                    #[allow(deprecated)]
                    stream.clone_htod(&rs_seed[l]).unwrap()
                })
                .collect()
        };
        let mut rs_loop = mk();
        let mut rs_grp = mk();
        let mut gated_loop: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * value_dim).unwrap();
        let mut gated_grp: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * value_dim).unwrap();
        let rc = run_per_lane_split(
            &stream, &sh, m, &mixed, &z, &ab, &a_log, &dt_bias, &norm_w, &mut rs_loop,
            &mut gated_loop, rms_eps,
        );
        assert_eq!(rc, 0, "gdn_decode_step_split_bf16 rc={rc} m={m}");
        let rc = run_group_split(
            &stream, &sh, m, &mixed, &z, &ab, &a_log, &dt_bias, &norm_w, &mut rs_grp,
            &mut gated_grp, rms_eps,
        );
        assert_eq!(rc, 0, "gdn_decode_step_split_group_bf16 rc={rc} m={m}");
        stream.synchronize().unwrap();

        #[allow(deprecated)]
        let x = stream.memcpy_dtov(&gated_loop).unwrap();
        #[allow(deprecated)]
        let y = stream.memcpy_dtov(&gated_grp).unwrap();
        let d = x.iter().zip(y.iter()).filter(|(p, q)| p != q).count();
        assert_eq!(
            d,
            0,
            "m={m}: {d} of {} gated bf16 values differ between the group split and the per-lane \
             split. {WHY}",
            x.len()
        );
        for l in 0..m {
            #[allow(deprecated)]
            let a = stream.memcpy_dtov(&rs_loop[l]).unwrap();
            #[allow(deprecated)]
            let b = stream.memcpy_dtov(&rs_grp[l]).unwrap();
            let d = a
                .iter()
                .zip(b.iter())
                .filter(|(p, q)| p.to_bits() != q.to_bits())
                .count();
            assert_eq!(
                d,
                0,
                "m={m} lane={l}: {d} of {} recurrent-state f32 values differ between the group \
                 split and the per-lane split. {WHY}",
                a.len()
            );
        }
        cells += 1;
    }
    assert_eq!(cells, lane_counts.len());
    eprintln!(
        "[gdn-lanes-group] split form: {cells} lane counts {lane_counts:?} bit-identical in \
         gated output and recurrent state"
    );
}

#[test]
#[ignore = "needs CUDA device 0; set NV_GDN_LANES_GROUP_TEST=1 -- proves the group-spanning GDN \
            conv and step kernels leave bit-identical gated outputs, conv states and recurrent \
            states to the per-lane loop they replace, at the qwen3.8 GDN decode shape for every \
            lane count the batch route can form"]
fn gdn_group_conv_and_step_are_bit_identical_to_the_per_lane_loop_in_output_and_in_both_states() {
    if std::env::var("NV_GDN_LANES_GROUP_TEST").as_deref() != Ok("1") {
        panic!("set NV_GDN_LANES_GROUP_TEST=1 to run this GPU test (it must never silently skip)");
    }
    eprintln!("[gdn-lanes-group] {WHY}");
    let device = Device::new_cuda_with_stream(0).expect("cuda device 0");
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut rng = Lcg(0x2f6b_ba17_04d3_9e51);
    let rms_eps = 1e-6f32;

    let shapes = [
        THE_QWEN38_27B_GDN_DECODE_SHAPE,
        GdnShape {
            n_k: 8,
            n_v: 16,
            d_k: 128,
            d_v: 128,
            k: 4,
        },
        GdnShape {
            n_k: 4,
            n_v: 8,
            d_k: 64,
            d_v: 64,
            k: 3,
        },
    ];
    let lane_counts = [1usize, 2, 3, 4, 8, 12, 16];
    let mut cells = 0usize;
    let mut differing_total = 0usize;
    for sh in shapes {
        let conv_dim = sh.conv_dim();
        let value_dim = sh.value_dim();
        let ab_stride = 2 * sh.n_v;
        let m_max = *lane_counts.iter().max().unwrap();

        let qkv_bits = rng.bits(m_max * conv_dim, 1.0);
        let z_bits = rng.bits(m_max * value_dim, 1.0);
        let ab_bits = rng.bits(m_max * ab_stride, 1.0);
        let a_log_bits = rng.bits(sh.n_v, 0.5);
        let dt_bits = rng.bits(sh.n_v, 0.5);
        let nw_bits = rng.bits(sh.d_v, 1.0);
        let cw_bits = rng.bits(conv_dim * sh.k, 0.5);
        let cs_seed: Vec<Vec<u16>> = (0..m_max)
            .map(|_| rng.bits(conv_dim * (sh.k - 1), 1.0))
            .collect();
        let rs_seed: Vec<Vec<f32>> = (0..m_max)
            .map(|_| rng.f32s(sh.state_elems(), 0.25))
            .collect();

        #[allow(deprecated)]
        let qkv: CudaSlice<u16> = stream.clone_htod(&qkv_bits).unwrap();
        #[allow(deprecated)]
        let z: CudaSlice<u16> = stream.clone_htod(&z_bits).unwrap();
        #[allow(deprecated)]
        let ab: CudaSlice<u16> = stream.clone_htod(&ab_bits).unwrap();
        #[allow(deprecated)]
        let a_log: CudaSlice<u16> = stream.clone_htod(&a_log_bits).unwrap();
        #[allow(deprecated)]
        let dt_bias: CudaSlice<u16> = stream.clone_htod(&dt_bits).unwrap();
        #[allow(deprecated)]
        let norm_w: CudaSlice<u16> = stream.clone_htod(&nw_bits).unwrap();
        #[allow(deprecated)]
        let conv_w: CudaSlice<u16> = stream.clone_htod(&cw_bits).unwrap();

        for &m in lane_counts.iter() {
            let mut mk_states = || {
                let cs: Vec<CudaSlice<u16>> = (0..m)
                    .map(|l| {
                        #[allow(deprecated)]
                        stream.clone_htod(&cs_seed[l]).unwrap()
                    })
                    .collect();
                let rs: Vec<CudaSlice<f32>> = (0..m)
                    .map(|l| {
                        #[allow(deprecated)]
                        stream.clone_htod(&rs_seed[l]).unwrap()
                    })
                    .collect();
                (cs, rs)
            };
            let (mut cs_loop, mut rs_loop) = mk_states();
            let (mut cs_grp, mut rs_grp) = mk_states();
            let mut mixed_loop: CudaSlice<u16> =
                stream.alloc_zeros::<u16>(m * conv_dim).unwrap();
            let mut gated_loop: CudaSlice<u16> =
                stream.alloc_zeros::<u16>(m * value_dim).unwrap();
            let mut mixed_grp: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * conv_dim).unwrap();
            let mut gated_grp: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * value_dim).unwrap();

            run_per_lane_loop(
                &stream,
                &sh,
                m,
                &qkv,
                &z,
                &ab,
                &a_log,
                &dt_bias,
                &norm_w,
                &conv_w,
                &mut cs_loop,
                &mut rs_loop,
                &mut mixed_loop,
                &mut gated_loop,
                rms_eps,
            );
            run_group(
                &stream,
                &sh,
                m,
                &qkv,
                &z,
                &ab,
                &a_log,
                &dt_bias,
                &norm_w,
                &conv_w,
                &mut cs_grp,
                &mut rs_grp,
                &mut mixed_grp,
                &mut gated_grp,
                rms_eps,
            );
            stream.synchronize().unwrap();

            let mut cell_differing = 0usize;
            let mut compare_u16 = |name: &str, a: &CudaSlice<u16>, b: &CudaSlice<u16>| {
                #[allow(deprecated)]
                let x = stream.memcpy_dtov(a).unwrap();
                #[allow(deprecated)]
                let y = stream.memcpy_dtov(b).unwrap();
                let d = x.iter().zip(y.iter()).filter(|(p, q)| p != q).count();
                assert_eq!(
                    d,
                    0,
                    "n_k={} n_v={} d_k={} d_v={} k={} m={m}: {d} of {} {name} bf16 values differ \
                     between the group launch and the per-lane loop. {WHY}",
                    sh.n_k,
                    sh.n_v,
                    sh.d_k,
                    sh.d_v,
                    sh.k,
                    x.len()
                );
                cell_differing += d;
            };
            compare_u16("mixed", &mixed_loop, &mixed_grp);
            compare_u16("gated", &gated_loop, &gated_grp);
            for l in 0..m {
                compare_u16("conv_state", &cs_loop[l], &cs_grp[l]);
            }
            for l in 0..m {
                #[allow(deprecated)]
                let x = stream.memcpy_dtov(&rs_loop[l]).unwrap();
                #[allow(deprecated)]
                let y = stream.memcpy_dtov(&rs_grp[l]).unwrap();
                let d = x
                    .iter()
                    .zip(y.iter())
                    .filter(|(p, q)| p.to_bits() != q.to_bits())
                    .count();
                assert_eq!(
                    d, 0,
                    "n_v={} d_k={} d_v={} m={m} lane={l}: {d} of {} recurrent-state f32 values \
                     differ between the group launch and the per-lane loop. {WHY}",
                    sh.n_v,
                    sh.d_k,
                    sh.d_v,
                    x.len()
                );
                cell_differing += d;
            }
            differing_total += cell_differing;
            cells += 1;
        }
        eprintln!(
            "[gdn-lanes-group] n_k={} n_v={} d_k={} d_v={} k={}: lanes {lane_counts:?} all \
             bit-identical in mixed, gated, conv_state and recurrent_state",
            sh.n_k, sh.n_v, sh.d_k, sh.d_v, sh.k
        );
    }
    assert_eq!(
        cells,
        shapes.len() * lane_counts.len(),
        "the ladder must cover every lane count at every shape or it is not a certificate"
    );
    assert_eq!(differing_total, 0, "{WHY}");
    eprintln!(
        "[gdn-lanes-group] {cells} (shape, m) cells, 0 differing values across gated output, \
         conv state and recurrent state"
    );
}
