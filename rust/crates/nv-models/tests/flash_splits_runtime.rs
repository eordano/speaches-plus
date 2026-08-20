#![cfg(feature = "wgpu")]

const WARPS: usize = 8;

const LOG2E: f32 = 1.442_695_f32;

fn fd_exp(x: f32) -> f32 {
    (x * LOG2E).exp2()
}

fn fd_recip(x: f32) -> f32 {
    let r = 1.0f32 / x;
    (-x).mul_add(r, 1.0).mul_add(r, r)
}

struct Partial {
    m: f32,
    l: f32,
    a0: f32,
    a1: f32,
}

fn stage1(scores: &[f32], v0: &[f32], v1: &[f32], splits: usize, split: usize) -> Partial {
    let stride = splits * WARPS;
    let mut wm = vec![f32::NEG_INFINITY; WARPS];
    let mut wl = vec![0.0f32; WARPS];
    let mut wa0 = vec![0.0f32; WARPS];
    let mut wa1 = vec![0.0f32; WARPS];
    for (w, item) in wm.iter_mut().enumerate() {
        let mut p = split * WARPS + w;
        while p < scores.len() {
            let s = scores[p];
            let m_new = item.max(s);
            let corr = fd_exp(*item - m_new);
            let weight = fd_exp(s - m_new);
            wl[w] = wl[w].mul_add(corr, weight);
            wa0[w] = weight.mul_add(v0[p], wa0[w] * corr);
            wa1[w] = weight.mul_add(v1[p], wa1[w] * corr);
            *item = m_new;
            p += stride;
        }
    }
    let m_blk = wm.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut l_blk = 0.0f32;
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    for w in 0..WARPS {
        if wm[w] > f32::NEG_INFINITY {
            let sc = fd_exp(wm[w] - m_blk);
            l_blk += wl[w] * sc;
            a0 += wa0[w] * sc;
            a1 += wa1[w] * sc;
        }
    }
    Partial {
        m: m_blk,
        l: l_blk,
        a0,
        a1,
    }
}

fn stage2_looped(parts: &[Partial]) -> (f32, f32) {
    let m_glob = parts.iter().map(|p| p.m).fold(f32::NEG_INFINITY, f32::max);
    let scale = |p: &Partial| {
        if p.m > f32::NEG_INFINITY {
            fd_exp(p.m - m_glob)
        } else {
            0.0
        }
    };
    let mut l_glob = 0.0f32;
    for p in parts {
        l_glob = p.l.mul_add(scale(p), l_glob);
    }
    let inv_l = if l_glob > 0.0 { fd_recip(l_glob) } else { 0.0 };
    let mut a0 = 0.0f32;
    for p in parts {
        a0 = p.a0.mul_add(scale(p), a0);
    }
    let mut a1 = 0.0f32;
    for p in parts {
        a1 = p.a1.mul_add(scale(p), a1);
    }
    (a0 * inv_l, a1 * inv_l)
}

fn stage2_unrolled16(parts: &[Partial]) -> (f32, f32) {
    assert_eq!(parts.len(), 16, "the shipping packed stage2 unrolls 16");
    let m_glob = parts.iter().map(|p| p.m).fold(f32::NEG_INFINITY, f32::max);
    let mut ssc = [0.0f32; 16];
    let mut l_glob = 0.0f32;
    for s in 0..16 {
        let sc = if parts[s].m > f32::NEG_INFINITY {
            fd_exp(parts[s].m - m_glob)
        } else {
            0.0
        };
        ssc[s] = sc;
        l_glob = parts[s].l.mul_add(sc, l_glob);
    }
    let inv_l = if l_glob > 0.0 { fd_recip(l_glob) } else { 0.0 };
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    for s in 0..16 {
        a0 = parts[s].a0.mul_add(ssc[s], a0);
    }
    for s in 0..16 {
        a1 = parts[s].a1.mul_add(ssc[s], a1);
    }
    (a0 * inv_l, a1 * inv_l)
}

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as f32 / (u32::MAX >> 1) as f32 - 1.0
    }
}

fn draw(total: usize, seed: u64, score_span: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = Lcg(seed);
    let scores = (0..total).map(|_| r.next_f32() * score_span).collect();
    let v0 = (0..total).map(|_| r.next_f32()).collect();
    let v1 = (0..total).map(|_| r.next_f32()).collect();
    (scores, v0, v1)
}

fn run(total: usize, splits: usize, seed: u64, span: f32, looped: bool) -> (f32, f32) {
    let (scores, v0, v1) = draw(total, seed, span);
    let parts: Vec<Partial> = (0..splits)
        .map(|s| stage1(&scores, &v0, &v1, splits, s))
        .collect();
    if looped {
        stage2_looped(&parts)
    } else {
        stage2_unrolled16(&parts)
    }
}

#[test]
fn looped_stage2_is_bit_identical_to_the_unrolled_one_at_the_default_split_count() {
    for (i, total) in [1usize, 7, 128, 1000, 8192, 48_000].iter().enumerate() {
        for span in [1.0f32, 8.0, 30.0] {
            let seed = 0x51ed ^ (i as u64) << 8 ^ span.to_bits() as u64;
            let looped = run(*total, 16, seed, span, true);
            let unrolled = run(*total, 16, seed, span, false);
            assert_eq!(
                looped.0.to_bits(),
                unrolled.0.to_bits(),
                "total {total} span {span}: the loop form reorders nothing at 16 partials, so \
                 lane 0 must match bit-for-bit"
            );
            assert_eq!(
                looped.1.to_bits(),
                unrolled.1.to_bits(),
                "total {total} span {span}: lane 1 must match bit-for-bit"
            );
        }
    }
}

const SPLIT_CHANGE_REL_TOL: f32 = 1.5e-6;

const TIGHTNESS: f32 = 8.0;

#[test]
fn changing_the_split_count_moves_the_answer_by_reassociation_only() {
    let mut worst = 0.0f32;
    for (i, total) in [128usize, 1000, 8192, 48_000, 96_000].iter().enumerate() {
        for splits in [8usize, 32, 64] {
            for span in [1.0f32, 8.0, 30.0] {
                let seed = 0xf1a5 ^ (i as u64) << 8 ^ (splits as u64) << 3 ^ span.to_bits() as u64;
                let base = run(*total, 16, seed, span, true);
                let alt = run(*total, splits, seed, span, true);
                for (b, a) in [(base.0, alt.0), (base.1, alt.1)] {
                    let rel = (b - a).abs() / b.abs().max(1e-3);
                    assert!(
                        rel <= SPLIT_CHANGE_REL_TOL,
                        "total {total} splits {splits} span {span}: relative move {rel:e} exceeds \
                         {SPLIT_CHANGE_REL_TOL:e}; a split count only regroups the same summands, \
                         so anything larger is a partition bug, not rounding"
                    );
                    worst = worst.max(rel);
                }
            }
        }
    }
    assert!(
        worst * TIGHTNESS >= SPLIT_CHANGE_REL_TOL,
        "the observed worst relative move {worst:e} is more than {TIGHTNESS}x under the declared \
         tolerance {SPLIT_CHANGE_REL_TOL:e}; a bound that loose would pass a genuinely broken \
         partition, so tighten it to what the reassociation actually costs"
    );
    assert!(
        worst > 0.0,
        "no split count moved the answer at all, which means the sweep never exercised a \
         different partition"
    );
}

#[test]
fn a_wider_split_count_still_covers_every_position_exactly_once() {
    for total in [1usize, 7, 128, 1000, 8192] {
        for splits in [8usize, 16, 32, 64] {
            let mut seen = vec![0u32; total];
            for split in 0..splits {
                for w in 0..WARPS {
                    let mut p = split * WARPS + w;
                    while p < total {
                        seen[p] += 1;
                        p += splits * WARPS;
                    }
                }
            }
            assert!(
                seen.iter().all(|c| *c == 1),
                "total {total} splits {splits}: the split/warp stride must tile the position axis \
                 exactly once, otherwise softmax mass is dropped or double counted"
            );
        }
    }
}
