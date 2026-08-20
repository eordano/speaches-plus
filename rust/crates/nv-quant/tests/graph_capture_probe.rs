#![cfg(feature = "cuda")]

use cudarc::driver::CudaContext;
use half::bf16;
use nv_kernels::graph::CudaGraphRunner;
use nv_quant::matmul::TensorCoreGemm;

fn build(m: usize, k: usize, seed: usize) -> Vec<bf16> {
    (0..m * k)
        .map(|j| bf16::from_f32(((j + seed) as f32 * 0.0011).cos()))
        .collect()
}

#[test]
fn det_and_slice_gemm_survive_graph_capture() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let main = ctx.new_stream().unwrap();
    let gemm = TensorCoreGemm::new(main.clone()).unwrap();

    let k = 5376usize;
    let m = 4usize;
    let n_fused = 16384usize;
    let n_q2 = 16384usize;
    let n_k2 = 2048usize;

    let a_host = build(m, k, 7);
    let w1_host = build(n_fused, k, 11);
    let w2_host = build(n_q2 + n_k2, k, 13);
    #[allow(deprecated)]
    let a_dev = main.clone_htod(&a_host).unwrap();
    #[allow(deprecated)]
    let w1_dev = main.clone_htod(&w1_host).unwrap();
    #[allow(deprecated)]
    let w2_dev = main.clone_htod(&w2_host).unwrap();

    let mp = 40usize;
    let ap_host = build(mp, k, 3);
    #[allow(deprecated)]
    let ap_dev = main.clone_htod(&ap_host).unwrap();
    let mut cp = main.alloc_zeros::<bf16>(mp * n_k2).unwrap();
    gemm.bf16_matmul_row_major_bt_off(
        &main,
        &ap_dev,
        &w2_dev,
        n_q2 * k,
        &mut cp,
        mp as u64,
        n_k2 as u64,
        k as u64,
        1.0,
        0.0,
    )
    .unwrap();
    main.synchronize().unwrap();

    let mut c1_ref = main.alloc_zeros::<bf16>(m * n_fused).unwrap();
    gemm.bf16_matmul_row_major_bt_det(
        &main,
        &a_dev,
        &w1_dev,
        &mut c1_ref,
        m as u64,
        n_fused as u64,
        k as u64,
        1.0,
        0.0,
    )
    .unwrap();
    let mut c2_ref = main.alloc_zeros::<bf16>(m * n_k2).unwrap();
    gemm.bf16_matmul_row_major_bt_off(
        &main,
        &a_dev,
        &w2_dev,
        n_q2 * k,
        &mut c2_ref,
        m as u64,
        n_k2 as u64,
        k as u64,
        1.0,
        0.0,
    )
    .unwrap();
    main.synchronize().unwrap();
    #[allow(deprecated)]
    let c1_ref_host = main.memcpy_dtov(&c1_ref).unwrap();
    #[allow(deprecated)]
    let c2_ref_host = main.memcpy_dtov(&c2_ref).unwrap();

    unsafe { ctx.disable_event_tracking() };

    let forked = ctx.new_stream().unwrap();
    let mut c1_out = forked.alloc_zeros::<bf16>(m * n_fused).unwrap();
    let mut c2_out = forked.alloc_zeros::<bf16>(m * n_k2).unwrap();

    gemm.bf16_matmul_row_major_bt_det(
        &forked,
        &a_dev,
        &w1_dev,
        &mut c1_out,
        m as u64,
        n_fused as u64,
        k as u64,
        1.0,
        0.0,
    )
    .unwrap();
    gemm.bf16_matmul_row_major_bt_off(
        &forked,
        &a_dev,
        &w2_dev,
        n_q2 * k,
        &mut c2_out,
        m as u64,
        n_k2 as u64,
        k as u64,
        1.0,
        0.0,
    )
    .unwrap();
    forked.synchronize().unwrap();

    let mut runner = CudaGraphRunner::new(forked.clone());
    runner
        .run(42, |s| {
            gemm.bf16_matmul_row_major_bt_det(
                s,
                &a_dev,
                &w1_dev,
                &mut c1_out,
                m as u64,
                n_fused as u64,
                k as u64,
                1.0,
                0.0,
            )?;
            gemm.bf16_matmul_row_major_bt_off(
                s,
                &a_dev,
                &w2_dev,
                n_q2 * k,
                &mut c2_out,
                m as u64,
                n_k2 as u64,
                k as u64,
                1.0,
                0.0,
            )?;
            Ok(())
        })
        .unwrap();
    forked
        .synchronize()
        .expect("sync after capture+first launch");

    for round in 0..3 {
        runner.run(42, |_| unreachable!("should replay")).unwrap();
        forked
            .synchronize()
            .unwrap_or_else(|e| panic!("sync after replay {round}: {e:?}"));
        #[allow(deprecated)]
        let c1_host = forked.memcpy_dtov(&c1_out).unwrap();
        #[allow(deprecated)]
        let c2_host = forked.memcpy_dtov(&c2_out).unwrap();
        let mm1 = c1_host
            .iter()
            .zip(c1_ref_host.iter())
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let mm2 = c2_host
            .iter()
            .zip(c2_ref_host.iter())
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        eprintln!("replay {round}: det mismatches={mm1} slice mismatches={mm2}");
        assert_eq!(mm1, 0);
        assert_eq!(mm2, 0);
    }
}

struct ReplayRig {
    runner: CudaGraphRunner,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    a_dev: cudarc::driver::CudaSlice<bf16>,
    w_dev: std::sync::Arc<cudarc::driver::CudaSlice<bf16>>,
    c_out: cudarc::driver::CudaSlice<bf16>,
    m: usize,
    n: usize,
    k: usize,
}

impl ReplayRig {
    fn new(
        ctx: &std::sync::Arc<CudaContext>,
        w_dev: std::sync::Arc<cudarc::driver::CudaSlice<bf16>>,
        m: usize,
        n: usize,
        k: usize,
        seed: usize,
    ) -> Self {
        let stream = ctx.new_stream().unwrap();
        let a_host = build(m, k, seed);
        #[allow(deprecated)]
        let a_dev = stream.clone_htod(&a_host).unwrap();
        let c_out = stream.alloc_zeros::<bf16>(m * n).unwrap();
        let _ = TensorCoreGemm::new(stream.clone()).unwrap();
        stream.synchronize().unwrap();
        let runner = CudaGraphRunner::new(stream.clone());
        Self {
            runner,
            stream,
            a_dev,
            w_dev,
            c_out,
            m,
            n,
            k,
        }
    }

    fn capture(&mut self) {
        let gemm = TensorCoreGemm;
        let ReplayRig {
            runner,
            a_dev,
            w_dev,
            c_out,
            m,
            n,
            k,
            ..
        } = self;
        let (m, n, k) = (*m as u64, *n as u64, *k as u64);
        runner
            .run(7, |s| {
                gemm.bf16_matmul_row_major_bt_det(s, a_dev, w_dev, c_out, m, n, k, 1.0, 0.0)
            })
            .unwrap();
        self.stream.synchronize().unwrap();
    }

    fn replay(&mut self) {
        self.runner.run(7, |_| unreachable!("must replay")).unwrap();
        self.stream.synchronize().unwrap();
    }

    fn readback(&self) -> Vec<bf16> {
        #[allow(deprecated)]
        self.stream.memcpy_dtov(&self.c_out).unwrap()
    }

    fn reference(&self) -> Vec<bf16> {
        let gemm = TensorCoreGemm;
        let mut c_ref = self.stream.alloc_zeros::<bf16>(self.m * self.n).unwrap();
        gemm.bf16_matmul_row_major_bt_det(
            &self.stream,
            &self.a_dev,
            &self.w_dev,
            &mut c_ref,
            self.m as u64,
            self.n as u64,
            self.k as u64,
            1.0,
            0.0,
        )
        .unwrap();
        self.stream.synchronize().unwrap();
        #[allow(deprecated)]
        self.stream.memcpy_dtov(&c_ref).unwrap()
    }
}

fn mismatches(a: &[bf16], b: &[bf16]) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

#[test]
fn concurrent_replays_on_distinct_streams_are_bitwise_stable() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    unsafe { ctx.disable_event_tracking() };
    let (m, n, k) = (4usize, 4096usize, 2048usize);
    let w_host = build(n, k, 11);
    let boot = ctx.new_stream().unwrap();
    #[allow(deprecated)]
    let w_dev = std::sync::Arc::new(boot.clone_htod(&w_host).unwrap());
    boot.synchronize().unwrap();

    let mut rig1 = ReplayRig::new(&ctx, w_dev.clone(), m, n, k, 3);
    let mut rig2 = ReplayRig::new(&ctx, w_dev.clone(), m, n, k, 17);
    let ref1 = rig1.reference();
    let ref2 = rig2.reference();
    rig1.capture();
    rig2.capture();

    let h1 = std::thread::spawn(move || {
        for i in 0..300 {
            rig1.replay();
            if i % 25 == 0 {
                assert_eq!(mismatches(&rig1.readback(), &ref1), 0, "rig1 replay {i}");
            }
        }
        assert_eq!(mismatches(&rig1.readback(), &ref1), 0, "rig1 final");
    });
    let h2 = std::thread::spawn(move || {
        for i in 0..300 {
            rig2.replay();
            if i % 25 == 0 {
                assert_eq!(mismatches(&rig2.readback(), &ref2), 0, "rig2 replay {i}");
            }
        }
        assert_eq!(mismatches(&rig2.readback(), &ref2), 0, "rig2 final");
    });
    h1.join().expect("rig1 thread");
    h2.join().expect("rig2 thread");
    eprintln!("concurrent replays: 2 x 300 replays bitwise stable");
}

#[test]
fn capture_and_eager_work_concurrent_with_replays() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    unsafe { ctx.disable_event_tracking() };
    let (m, n, k) = (4usize, 4096usize, 2048usize);
    let w_host = build(n, k, 11);
    let boot = ctx.new_stream().unwrap();
    #[allow(deprecated)]
    let w_dev = std::sync::Arc::new(boot.clone_htod(&w_host).unwrap());
    boot.synchronize().unwrap();

    let mut replayer = ReplayRig::new(&ctx, w_dev.clone(), m, n, k, 3);
    let ref_r = replayer.reference();
    replayer.capture();

    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    let stop_r = stop.clone();
    let h_replay = std::thread::spawn(move || {
        let mut count = 0u64;
        while !stop_r.load(std::sync::atomic::Ordering::Relaxed) {
            replayer.replay();
            count += 1;
            if count % 50 == 0 {
                assert_eq!(
                    mismatches(&replayer.readback(), &ref_r),
                    0,
                    "replay {count}"
                );
            }
        }
        assert_eq!(
            mismatches(&replayer.readback(), &ref_r),
            0,
            "replayer final"
        );
        count
    });

    let stop_e = stop.clone();
    let w_e = w_dev.clone();
    let ctx_e = ctx.clone();
    let h_eager = std::thread::spawn(move || {
        let eager = ReplayRig::new(&ctx_e, w_e, m, n, k, 29);
        let ref_e = eager.reference();
        let mut count = 0u64;
        while !stop_e.load(std::sync::atomic::Ordering::Relaxed) {
            let got = eager.reference();
            assert_eq!(mismatches(&got, &ref_e), 0, "eager pass {count}");
            count += 1;
        }
        count
    });

    std::thread::sleep(std::time::Duration::from_millis(50));
    let mut captures = 0usize;
    for seed in 0..6usize {
        let mut fresh = ReplayRig::new(&ctx, w_dev.clone(), m, n, k, 100 + seed);
        let ref_f = fresh.reference();
        fresh.capture();
        for _ in 0..20 {
            fresh.replay();
        }
        assert_eq!(
            mismatches(&fresh.readback(), &ref_f),
            0,
            "fresh capture {seed}"
        );
        captures += 1;
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let replays = h_replay.join().expect("replay thread");
    let eagers = h_eager.join().expect("eager thread");
    assert!(replays > 0 && eagers > 0);
    eprintln!(
        "capture-vs-replay probe: {captures} captures concurrent with {replays} replays and {eagers} eager passes, all bitwise stable"
    );
}
