#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::WgpuContext;
use nv_specdecode::wgpu_spec::{SpecDims, SpecWeights, WgpuChainSpec, WgpuSpecModel};
mod common;
use common::spec_dims_tiny as dims;

fn ctx(test: &str) -> Option<&'static WgpuContext> {
    let bail = |what: String| -> Option<&'static WgpuContext> {
        if std::env::var("NV_KERNELS_WGPU_ALLOW_SKIP").as_deref() == Ok("1") {
            eprintln!("SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1): {test}: {what}. Not a pass.");
            return None;
        }
        panic!(
            "{test}: {what}. This box has an adapter and nvk.sh wires the Vulkan loader \
             (VK_ICD_FILENAMES + libvulkan on LD_LIBRARY_PATH), so a miss here means the loader \
             wiring regressed -- it does NOT mean the wgpu spec-decode assertions may report a \
             pass having dispatched nothing. Set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
        );
    };
    match WgpuContext::shared() {
        Ok(c) if c.qualify().qualified => Some(c),
        Ok(c) => bail(format!(
            "wgpu adapter not qualified: {:?}",
            c.qualify().reason
        )),
        Err(e) => bail(format!("no wgpu adapter: {e}")),
    }
}

fn verifier_weights(d: &SpecDims) -> SpecWeights {
    SpecWeights::synthetic(d, 3)
}

fn perturbed_drafter_weights(d: &SpecDims) -> SpecWeights {
    let mut w = verifier_weights(d);
    for (i, v) in w.wlm.iter_mut().enumerate() {
        *v += 0.015 * ((i as f32) * 0.911).sin();
    }
    w
}

fn prompt() -> Vec<u32> {
    vec![1, 17, 42, 5, 90, 33, 7]
}

fn wgpu_greedy(c: &'static WgpuContext, d: SpecDims, w: &SpecWeights, n: usize) -> Vec<u32> {
    let mut m = WgpuSpecModel::new(c, d, w, 1).unwrap();
    let mut out = Vec::with_capacity(n);
    let mut cur = m.prefill(&prompt()).unwrap();
    out.push(cur);
    while out.len() < n {
        cur = m.decode1(cur).unwrap();
        out.push(cur);
    }
    out
}

#[test]
fn wgpu_spec_loop_matches_wgpu_greedy_stream() {
    let Some(c) = ctx("wgpu_spec_loop_matches_wgpu_greedy_stream") else {
        return;
    };
    let d = dims();
    let wv = verifier_weights(&d);
    let wd = perturbed_drafter_weights(&d);
    let n = 48;

    let mut spec = WgpuChainSpec::new(c, d, &wv, &wd, 4).unwrap();
    let passes_before = spec.verifier.pass_count();
    let stats = spec.generate(&prompt(), n).unwrap();
    assert_eq!(
        spec.verifier.pass_count(),
        passes_before,
        "recording churned"
    );
    assert!(spec.verifier.replays() > 0);

    assert!(stats.emitted.len() >= n);
    let greedy = wgpu_greedy(c, d, &wv, stats.emitted.len());
    assert_eq!(
        stats.emitted, greedy,
        "spec-decoded stream must equal the verifier's greedy stream"
    );

    assert!(stats.drafted > 0);
    let rate = stats.acceptance_rate();
    println!(
        "wgpu chain spec: rounds={} drafted={} accepted={} acceptance_rate={:.3}",
        stats.rounds, stats.drafted, stats.accepted_drafts, rate
    );
    assert!(
        rate > 0.0 && rate <= 1.0,
        "acceptance rate {rate} out of range"
    );
}

#[test]
fn wgpu_spec_identical_drafter_accepts_every_draft() {
    let Some(c) = ctx("wgpu_spec_identical_drafter_accepts_every_draft") else {
        return;
    };
    let d = dims();
    let wv = verifier_weights(&d);
    let mut spec = WgpuChainSpec::new(c, d, &wv, &wv, 4).unwrap();
    let stats = spec.generate(&prompt(), 32).unwrap();
    assert_eq!(
        stats.accepted_drafts, stats.drafted,
        "identical drafter must be accepted verbatim"
    );
    assert_eq!(stats.acceptance_rate(), 1.0);
    let greedy = wgpu_greedy(c, d, &wv, stats.emitted.len());
    assert_eq!(stats.emitted, greedy);
    println!(
        "wgpu chain spec (identical drafter): rounds={} drafted={} acceptance_rate={:.3}",
        stats.rounds,
        stats.drafted,
        stats.acceptance_rate()
    );
}

#[cfg(feature = "cuda")]
mod cuda_parity {
    use super::*;
    use anyhow::Result;
    use candle_core::{Device, Tensor, D};
    use nv_specdecode::chain::{accept_prefix_argmax, build_chain_batch};
    use nv_specdecode::wgpu_spec::rope_tables;

    struct RefModel {
        dev: Device,
        d: SpecDims,
        committed: usize,
        kc: Vec<f32>,
        vc: Vec<f32>,
        embed: Vec<f32>,
        ln1: Tensor,
        wq_t: Tensor,
        wk_t: Tensor,
        wv_t: Tensor,
        wo_t: Tensor,
        ln2: Tensor,
        wg_t: Tensor,
        wu_t: Tensor,
        wd_t: Tensor,
        lnf: Tensor,
        wlm_t: Tensor,
        cos: Vec<f32>,
        sin: Vec<f32>,
    }

    fn transpose_host(w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = w[r * cols + c];
            }
        }
        out
    }

    fn rms(x: &Tensor, w: &Tensor, h: usize, eps: f32) -> Result<Tensor> {
        let ss = (x * x)?
            .sum_keepdim(D::Minus1)?
            .affine(1.0 / h as f64, eps as f64)?;
        let rr = ss.sqrt()?.recip()?;
        Ok(x.broadcast_mul(&rr)?.broadcast_mul(w)?)
    }

    impl RefModel {
        fn new(dev: &Device, d: SpecDims, w: &SpecWeights) -> Result<Self> {
            let up = |data: Vec<f32>, shape: (usize, usize)| -> Result<Tensor> {
                Ok(Tensor::from_vec(data, shape, dev)?)
            };
            let (cos, sin) = rope_tables(&d);
            Ok(Self {
                dev: dev.clone(),
                d,
                committed: 0,
                kc: vec![0f32; d.max_seq * d.kvdim()],
                vc: vec![0f32; d.max_seq * d.kvdim()],
                embed: w.embed.clone(),
                ln1: up(w.ln1.clone(), (1, d.h))?,
                wq_t: up(transpose_host(&w.wq, d.qdim(), d.h), (d.h, d.qdim()))?,
                wk_t: up(transpose_host(&w.wk, d.kvdim(), d.h), (d.h, d.kvdim()))?,
                wv_t: up(transpose_host(&w.wv, d.kvdim(), d.h), (d.h, d.kvdim()))?,
                wo_t: up(transpose_host(&w.wo, d.h, d.qdim()), (d.qdim(), d.h))?,
                ln2: up(w.ln2.clone(), (1, d.h))?,
                wg_t: up(transpose_host(&w.wg, d.inter, d.h), (d.h, d.inter))?,
                wu_t: up(transpose_host(&w.wu, d.inter, d.h), (d.h, d.inter))?,
                wd_t: up(transpose_host(&w.wd, d.h, d.inter), (d.inter, d.h))?,
                lnf: up(w.lnf.clone(), (1, d.h))?,
                wlm_t: up(transpose_host(&w.wlm, d.vocab, d.h), (d.h, d.vocab))?,
                cos,
                sin,
            })
        }

        fn rope_split(&self, t3: &Tensor, heads: usize, k: usize) -> Result<Tensor> {
            let half = self.d.hd / 2;
            let mut ch = Vec::with_capacity(k * half);
            let mut sh = Vec::with_capacity(k * half);
            for i in 0..k {
                let pos = self.committed + i;
                ch.extend_from_slice(&self.cos[pos * half..(pos + 1) * half]);
                sh.extend_from_slice(&self.sin[pos * half..(pos + 1) * half]);
            }
            let cos_t = Tensor::from_vec(ch, (k, 1, half), &self.dev)?;
            let sin_t = Tensor::from_vec(sh, (k, 1, half), &self.dev)?;
            let a = t3.narrow(2, 0, half)?;
            let b = t3.narrow(2, half, half)?;
            let lo = (a.broadcast_mul(&cos_t)? - b.broadcast_mul(&sin_t)?)?;
            let hi = (a.broadcast_mul(&sin_t)? + b.broadcast_mul(&cos_t)?)?;
            Ok(Tensor::cat(&[lo, hi], 2)?.reshape((k, heads * self.d.hd))?)
        }

        fn forward(&mut self, tokens: &[u32]) -> Result<Vec<u32>> {
            let d = self.d;
            let k = tokens.len();
            assert!(self.committed + k <= d.max_seq);
            let mut xh = vec![0f32; k * d.h];
            for (i, &t) in tokens.iter().enumerate() {
                xh[i * d.h..(i + 1) * d.h]
                    .copy_from_slice(&self.embed[t as usize * d.h..(t as usize + 1) * d.h]);
            }
            let x = Tensor::from_vec(xh, (k, d.h), &self.dev)?;
            let xn = rms(&x, &self.ln1, d.h, d.eps)?;
            let q = xn.matmul(&self.wq_t)?.reshape((k, d.nh, d.hd))?;
            let kx = xn.matmul(&self.wk_t)?.reshape((k, d.nkv, d.hd))?;
            let vx = xn.matmul(&self.wv_t)?;
            let qr = self.rope_split(&q, d.nh, k)?;
            let kr = self.rope_split(&kx, d.nkv, k)?;

            let kr_h: Vec<f32> = kr.reshape((k * d.kvdim(),))?.to_vec1()?;
            let vx_h: Vec<f32> = vx.reshape((k * d.kvdim(),))?.to_vec1()?;
            for i in 0..k {
                let dst = (self.committed + i) * d.kvdim();
                self.kc[dst..dst + d.kvdim()]
                    .copy_from_slice(&kr_h[i * d.kvdim()..(i + 1) * d.kvdim()]);
                self.vc[dst..dst + d.kvdim()]
                    .copy_from_slice(&vx_h[i * d.kvdim()..(i + 1) * d.kvdim()]);
            }

            let s_total = self.committed + k;
            let grp = d.nh / d.nkv;
            let mut krep = vec![0f32; d.nh * s_total * d.hd];
            let mut vrep = vec![0f32; d.nh * s_total * d.hd];
            for head in 0..d.nh {
                let kvh = head / grp;
                for s in 0..s_total {
                    let src = s * d.kvdim() + kvh * d.hd;
                    let dst = head * s_total * d.hd + s * d.hd;
                    krep[dst..dst + d.hd].copy_from_slice(&self.kc[src..src + d.hd]);
                    vrep[dst..dst + d.hd].copy_from_slice(&self.vc[src..src + d.hd]);
                }
            }
            let k_t = Tensor::from_vec(krep, (d.nh, s_total, d.hd), &self.dev)?;
            let v_t = Tensor::from_vec(vrep, (d.nh, s_total, d.hd), &self.dev)?;
            let q_t = qr.reshape((k, d.nh, d.hd))?.transpose(0, 1)?.contiguous()?;
            let scores = q_t
                .matmul(&k_t.transpose(1, 2)?.contiguous()?)?
                .affine(d.scale() as f64, 0.0)?;
            let mut maskh = vec![0f32; k * s_total];
            for i in 0..k {
                for s in 0..s_total {
                    if s > self.committed + i {
                        maskh[i * s_total + s] = -1e30;
                    }
                }
            }
            let mask_t = Tensor::from_vec(maskh, (1, k, s_total), &self.dev)?;
            let probs = candle_nn::ops::softmax(&scores.broadcast_add(&mask_t)?, 2)?;
            let ao = probs
                .matmul(&v_t)?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((k, d.qdim()))?;
            let h1 = (ao.matmul(&self.wo_t)? + x)?;
            let tn = rms(&h1, &self.ln2, d.h, d.eps)?;
            let g = tn.matmul(&self.wg_t)?;
            let u = tn.matmul(&self.wu_t)?;
            let sig = g.neg()?.exp()?.affine(1.0, 1.0)?.recip()?;
            let act = ((&g * &sig)? * &u)?;
            let h2 = (act.matmul(&self.wd_t)? + h1)?;
            let xf = rms(&h2, &self.lnf, d.h, d.eps)?;
            let logits: Vec<Vec<f32>> = xf.matmul(&self.wlm_t)?.to_vec2()?;
            Ok(logits
                .iter()
                .map(|row| {
                    let mut bi = 0usize;
                    let mut best = row[0];
                    for (v, &val) in row.iter().enumerate().skip(1) {
                        if val > best {
                            best = val;
                            bi = v;
                        }
                    }
                    bi as u32
                })
                .collect())
        }

        fn decode1(&mut self, token: u32) -> Result<u32> {
            let out = self.forward(&[token])?;
            self.committed += 1;
            Ok(out[0])
        }

        fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
            let mut last = 0;
            for &t in tokens {
                last = self.decode1(t)?;
            }
            Ok(last)
        }

        fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>> {
            self.forward(batch)
        }
    }

    fn cuda_greedy(dev: &Device, d: SpecDims, w: &SpecWeights, n: usize) -> Vec<u32> {
        let mut m = RefModel::new(dev, d, w).unwrap();
        let mut out = Vec::with_capacity(n);
        let mut cur = m.prefill(&prompt()).unwrap();
        out.push(cur);
        while out.len() < n {
            cur = m.decode1(cur).unwrap();
            out.push(cur);
        }
        out
    }

    #[test]
    fn wgpu_spec_matches_cuda_chain_path() {
        let Some(c) = ctx("wgpu_spec_matches_cuda_chain_path") else {
            return;
        };
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("wgpu_spec_matches_cuda_chain_path: SKIP no CUDA device 0: {e}");
                return;
            }
        };
        let d = dims();
        let k = 4usize;
        let wv = verifier_weights(&d);
        let wd = perturbed_drafter_weights(&d);

        let mut wg_v = WgpuSpecModel::new(c, d, &wv, k).unwrap();
        let mut wg_d = WgpuSpecModel::new(c, d, &wd, 1).unwrap();
        let mut cu_v = RefModel::new(&dev, d, &wv).unwrap();
        let mut cu_d = RefModel::new(&dev, d, &wd).unwrap();

        let p = prompt();
        let b_w = wg_v.prefill(&p).unwrap();
        let b_c = cu_v.prefill(&p).unwrap();
        assert_eq!(
            b_w, b_c,
            "prefill bonus token diverges between wgpu and cuda"
        );
        wg_d.prefill(&p).unwrap();
        cu_d.prefill(&p).unwrap();

        let mut context = p.clone();
        let mut bonus = b_w;
        let mut emitted: Vec<u32> = Vec::new();
        let mut drafted = 0usize;
        let mut accepted = 0usize;
        let mut rounds = 0usize;

        for round in 0..24 {
            if wg_v.committed() + k > d.max_seq {
                break;
            }
            while wg_d.committed() < context.len() {
                let t = context[wg_d.committed()];
                wg_d.decode1(t).unwrap();
                cu_d.decode1(t).unwrap();
            }
            assert_eq!(wg_d.committed(), cu_d.committed);

            let base = context.len();
            let mut drafts_w = Vec::with_capacity(k - 1);
            let mut drafts_c = Vec::with_capacity(k - 1);
            let mut cur_w = bonus;
            let mut cur_c = bonus;
            for _ in 0..k - 1 {
                let tw = wg_d.decode1(cur_w).unwrap();
                let tc = cu_d.decode1(cur_c).unwrap();
                drafts_w.push(tw);
                drafts_c.push(tc);
                cur_w = tw;
                cur_c = tc;
            }
            assert_eq!(
                drafts_w, drafts_c,
                "round {round}: drafter tokens diverge between wgpu and cuda"
            );

            let batch = build_chain_batch(bonus, &drafts_w, k, true).unwrap();
            let amax_w = wg_v.verify_chain(&batch).unwrap();
            let amax_c = cu_v.verify_chain(&batch).unwrap();
            assert_eq!(
                amax_w, amax_c,
                "round {round}: verify argmax rows diverge between wgpu and cuda (batch={batch:?})"
            );

            let acc = accept_prefix_argmax(&batch, &amax_w).unwrap();
            wg_v.advance(acc.commit_len).unwrap();
            cu_v.committed += acc.commit_len;
            context.extend_from_slice(&batch[..acc.commit_len]);
            emitted.extend_from_slice(&batch[..acc.commit_len]);
            let target = base + acc.commit_len.min(k - 1);
            wg_d.rollback_to(target).unwrap();
            cu_d.committed = target;
            drafted += k - 1;
            accepted += acc.draft_accepted;
            rounds += 1;
            bonus = acc.next_bonus;
        }

        assert!(rounds > 0 && drafted > 0);
        let greedy = cuda_greedy(&dev, d, &wv, emitted.len());
        assert_eq!(
            emitted, greedy,
            "spec-decoded stream must equal the CUDA greedy stream"
        );

        let rate = accepted as f64 / drafted as f64;
        println!(
            "wgpu-vs-cuda chain spec: rounds={rounds} drafted={drafted} accepted={accepted} acceptance_rate={rate:.3}"
        );
        assert!(rate > 0.0 && rate <= 1.0);
    }
}
