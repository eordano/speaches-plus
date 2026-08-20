#![cfg(feature = "wgpu")]

mod common;
use common::FdParams;
use nv_kernels::wgpu_backend::compose;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::flash_decode as fd;
use nv_kernels::wgpu_backend::kernels::kv_fp8;

fn ctx() -> &'static WgpuContext {
    let c = WgpuContext::shared().expect("wgpu adapter required for --features wgpu");
    assert!(
        c.qualify().qualified,
        "adapter not qualified: {:?}",
        c.qualify().reason
    );
    c
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / 8_388_608.0) - 1.0
    }
}

fn bf16(v: f32) -> u16 {
    half::bf16::from_f32(v).to_bits()
}

fn unbf16(w: u16) -> f32 {
    f32::from_bits((w as u32) << 16)
}

fn worst_rel(got: &[u16], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mag = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    got.iter()
        .zip(want)
        .map(|(g, w)| (unbf16(*g) as f64 - w).abs() / mag)
        .fold(0.0, f64::max)
}

struct Fp8Case {
    q: Vec<u16>,
    k: Vec<u8>,
    v: Vec<u8>,
    ks: Vec<f32>,
    vs: Vec<f32>,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    total: usize,
}

impl Fp8Case {
    fn new(n_heads: usize, n_kv: usize, head_dim: usize, total: usize, seed: u64) -> Self {
        let mut r = Lcg(seed);
        let q = (0..n_heads * head_dim)
            .map(|_| bf16(r.next() * 1.5))
            .collect();
        let per = n_kv * head_dim;
        let k = (0..total * per)
            .map(|_| kv_fp8::encode_e4m3(r.next() * 240.0))
            .collect();
        let v = (0..total * per)
            .map(|_| kv_fp8::encode_e4m3(r.next() * 240.0))
            .collect();

        let ks = (0..total * n_kv)
            .map(|i| 0.002 * (1.0 + (i % 37) as f32) * (1.0 + (i / 7 % 11) as f32))
            .collect();
        let vs = (0..total * n_kv)
            .map(|i| 0.003 * (1.0 + (i % 23) as f32) * (1.0 + (i / 5 % 13) as f32))
            .collect();
        Self {
            q,
            k,
            v,
            ks,
            vs,
            n_heads,
            n_kv,
            head_dim,
            total,
        }
    }

    fn reference(&self, scaling: f64) -> Vec<f64> {
        let (hd, nkv) = (self.head_dim, self.n_kv);
        let group = self.n_heads / nkv;
        let mut out = vec![0f64; self.n_heads * hd];
        for h in 0..self.n_heads {
            let kvh = h / group;
            let scores: Vec<f64> = (0..self.total)
                .map(|p| {
                    let base = (p * nkv + kvh) * hd;
                    let dot: f64 = (0..hd)
                        .map(|d| {
                            unbf16(self.q[h * hd + d]) as f64
                                * kv_fp8::decode_e4m3(self.k[base + d]) as f64
                        })
                        .sum();
                    dot * self.ks[p * nkv + kvh] as f64 * scaling
                })
                .collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let denom: f64 = w.iter().sum();
            for d in 0..hd {
                let acc: f64 = (0..self.total)
                    .map(|p| {
                        let base = (p * nkv + kvh) * hd;
                        w[p] * kv_fp8::decode_e4m3(self.v[base + d]) as f64
                            * self.vs[p * nkv + kvh] as f64
                    })
                    .sum();
                out[h * hd + d] = acc / denom;
            }
        }
        out
    }

    fn run(&self, scaling: f32, splits: usize) -> Vec<u16> {
        let c = ctx();
        let mut out = vec![0u16; self.n_heads * self.head_dim];
        let elems = fd::flash_splitk_scratch_elems(self.n_heads, self.head_dim, splits)
            .expect("scratch size");
        let mut scratch = vec![0f32; elems];
        fd::flash_decode_fused_fp8kv(
            c,
            &self.q,
            &self.k,
            &self.v,
            &self.ks,
            &self.vs,
            &mut out,
            &mut scratch,
            &[self.total as i32],
            self.n_heads,
            self.n_kv,
            self.head_dim,
            0,
            scaling,
            splits,
            0,
        )
        .expect("flash_decode_fused_fp8kv");
        out
    }
}

fn assert_live(got: &[u16], want: &[f64], label: &str) {
    assert!(
        want.iter().any(|v| *v != 0.0),
        "{label}: the f64 reference is all zero -- this case would pass on any output"
    );
    assert!(
        got.iter().any(|w| unbf16(*w) != 0.0),
        "{label}: every output word is zero -- the kernel wrote nothing and this case \
         compares nothing against nothing"
    );
}

#[test]
fn fp8_splitk_decode_matches_an_f64_host_reference() {
    for &(nh, nkv, hd, total, splits) in &[
        (8usize, 2usize, 64usize, 96usize, 4usize),
        (8, 8, 128, 256, 8),
        (16, 4, 128, 131, 8),
        (4, 1, 256, 40, 4),
    ] {
        let c = Fp8Case::new(
            nh,
            nkv,
            hd,
            total,
            0x51ee_0000 ^ (total as u64) << 8 ^ hd as u64,
        );
        let scaling = 1.0 / (hd as f32).sqrt();
        let got = c.run(scaling, splits);
        let want = c.reference(scaling as f64);
        let label = format!("nh={nh} nkv={nkv} hd={hd} total={total} splits={splits}");
        assert_live(&got, &want, &label);
        let e = worst_rel(&got, &want);
        assert!(
            e < 8e-3,
            "{label}: worst relative error {e:.3e} against an f64 host reference"
        );
    }
}

#[test]
fn the_softmax_scaling_reaches_the_fp8_kernel() {
    let (nh, nkv, hd, total) = (8usize, 2usize, 64usize, 128usize);
    let c = Fp8Case::new(nh, nkv, hd, total, 0xBEEF_5C41);
    let scaling = 1.0 / (hd as f32).sqrt();

    let scaled = c.run(scaling, 8);
    let unscaled = c.run(1.0, 8);
    let want_scaled = c.reference(scaling as f64);
    let want_unscaled = c.reference(1.0);
    assert_live(&scaled, &want_scaled, "scaled");
    assert_live(&unscaled, &want_unscaled, "unscaled");
    assert!(
        worst_rel(&scaled, &want_scaled) < 8e-3,
        "scaled arm disagrees with its f64 reference"
    );
    assert!(
        worst_rel(&unscaled, &want_unscaled) < 8e-3,
        "unscaled arm disagrees with its f64 reference"
    );

    let separation = worst_rel(&scaled, &want_unscaled);
    assert!(
        separation > 5e-2,
        "an unscaled kernel would sit only {separation:.3e} from the scaled reference on this \
         fixture, which is inside the {:.0e} tolerance -- the case cannot see a dropped scale",
        8e-3
    );
}

#[test]
fn per_slot_and_per_head_fp8_scales_are_indexed_correctly() {
    let (nh, nkv, hd, total) = (16usize, 4usize, 128usize, 64usize);
    let c = Fp8Case::new(nh, nkv, hd, total, 0x5CA1_E00D);
    let scaling = 1.0 / (hd as f32).sqrt();
    let got = c.run(scaling, 8);
    let want = c.reference(scaling as f64);
    assert_live(&got, &want, "per-slot scales");
    let e = worst_rel(&got, &want);
    assert!(
        e < 8e-3,
        "worst relative error {e:.3e}: the per-slot/per-head fp8 scales are not landing where \
         the reference puts them"
    );

    let mut flat = c;
    let mean = flat.vs.iter().sum::<f32>() / flat.vs.len() as f32;
    flat.vs = vec![mean; flat.vs.len()];
    let flat_got = flat.run(scaling, 8);
    let moved = got.iter().zip(&flat_got).filter(|(a, b)| a != b).count();
    assert!(
        moved > got.len() / 2,
        "only {moved}/{} outputs moved when every V scale was replaced by their mean",
        got.len()
    );
}

fn run_stage2(entry: &str, scratch: &[f32], p: &FdParams) -> Vec<u16> {
    let c = ctx();
    let src = compose(fd::WGSL);
    let pipeline = dispatch::cached_compute_pipeline(c, entry, &src, entry).expect("pipeline");
    let sb = dispatch::storage_from_slice(c, "fd2-scratch", scratch);
    let out_elems = (p.n_heads * p.head_dim) as usize;
    let ob = dispatch::storage_zeroed(c, "fd2-out", (out_elems * 4) as u64);
    let pb = dispatch::uniform_from(c, "fd2-params", p);
    let group = dispatch::bind_group(c, &pipeline, &[(3, &ob), (4, &pb), (7, &sb)]);
    let mut enc = c.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(p.n_heads, 1, 1);
    }
    c.queue.submit([enc.finish()]);
    let words: Vec<u32> = dispatch::read_back(c, &ob, out_elems).expect("stage2 read back");
    words.iter().map(|w| (*w & 0xffff) as u16).collect()
}

fn stage2_reference(scratch: &[f32], n_heads: usize, head_dim: usize, splits: usize) -> Vec<f64> {
    let stride = head_dim + 2;
    let mut out = vec![0f64; n_heads * head_dim];
    for h in 0..n_heads {
        let base = h * splits * stride;
        let m_glob = (0..splits)
            .map(|s| scratch[base + s * stride] as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let sc: Vec<f64> = (0..splits)
            .map(|s| {
                let p0 = scratch[base + s * stride] as f64;
                if p0 > f64::NEG_INFINITY {
                    (p0 - m_glob).exp()
                } else {
                    0.0
                }
            })
            .collect();
        let l: f64 = (0..splits)
            .map(|s| scratch[base + s * stride + 1] as f64 * sc[s])
            .sum();
        for d in 0..head_dim {
            let a: f64 = (0..splits)
                .map(|s| scratch[base + s * stride + 2 + d] as f64 * sc[s])
                .sum();
            out[h * head_dim + d] = if l > 0.0 { a / l } else { 0.0 };
        }
    }
    out
}

#[test]
fn stage2_combine_matches_an_f64_host_reference() {
    for &(n_heads, head_dim, splits) in &[(4usize, 64usize, 8usize), (8, 128, 16), (2, 256, 4)] {
        let stride = head_dim + 2;
        let mut r = Lcg(0x2AAE_0001 ^ (head_dim as u64) << 8 ^ splits as u64);
        let mut scratch = vec![0f32; n_heads * splits * stride];
        for h in 0..n_heads {
            for s in 0..splits {
                let b = (h * splits + s) * stride;

                scratch[b] = 4.0 * r.next();
                scratch[b + 1] = 0.5 + 2.0 * (r.next() + 1.0);
                for d in 0..head_dim {
                    scratch[b + 2 + d] = 3.0 * r.next();
                }
            }
        }
        let p = FdParams {
            n_heads: n_heads as u32,
            n_kv: n_heads as u32,
            head_dim: head_dim as u32,
            total: 1,
            start: 0,
            splits: splits as u32,
            ring: 0,
            out_bf16: 1,
            scaling: 1.0,
            m_rows: 1,
            ..Default::default()
        };
        let got = run_stage2(fd::ENTRY_STAGE2, &scratch, &p);
        let want = stage2_reference(&scratch, n_heads, head_dim, splits);
        let label = format!("stage2 n_heads={n_heads} hd={head_dim} splits={splits}");
        assert_live(&got, &want, &label);
        let e = worst_rel(&got, &want);
        assert!(
            e < 8e-3,
            "{label}: worst relative error {e:.3e} against an f64 host reference"
        );
    }
}

#[test]
fn stage2_normalises_by_the_combined_denominator() {
    let (n_heads, head_dim, splits) = (4usize, 64usize, 8usize);
    let stride = head_dim + 2;
    let mut r = Lcg(0x0D10_0BED);
    let mut a = vec![0f32; n_heads * splits * stride];
    for h in 0..n_heads {
        for s in 0..splits {
            let b = (h * splits + s) * stride;
            a[b] = 3.0 * r.next();
            a[b + 1] = 1.0 + 2.0 * (r.next() + 1.0);
            for d in 0..head_dim {
                a[b + 2 + d] = 2.0 * r.next();
            }
        }
    }
    let mut b_scratch = a.clone();
    for h in 0..n_heads {
        for s in 0..splits {
            b_scratch[(h * splits + s) * stride + 1] *= 2.0;
        }
    }
    let p = FdParams {
        n_heads: n_heads as u32,
        n_kv: n_heads as u32,
        head_dim: head_dim as u32,
        total: 1,
        start: 0,
        splits: splits as u32,
        ring: 0,
        out_bf16: 1,
        scaling: 1.0,
        m_rows: 1,
        ..Default::default()
    };
    let ga = run_stage2(fd::ENTRY_STAGE2, &a, &p);
    let gb = run_stage2(fd::ENTRY_STAGE2, &b_scratch, &p);
    let wa = stage2_reference(&a, n_heads, head_dim, splits);
    let wb = stage2_reference(&b_scratch, n_heads, head_dim, splits);
    assert_live(&ga, &wa, "denominator arm a");
    assert_live(&gb, &wb, "denominator arm b");
    assert!(worst_rel(&ga, &wa) < 8e-3, "arm a left the f64 reference");
    assert!(worst_rel(&gb, &wb) < 8e-3, "arm b left the f64 reference");
    let moved = ga.iter().zip(&gb).filter(|(x, y)| x != y).count();
    assert!(
        moved > ga.len() / 2,
        "only {moved}/{} outputs moved when every split's running length doubled, so stage2 \
         is not dividing by the combined denominator",
        ga.len()
    );
}
