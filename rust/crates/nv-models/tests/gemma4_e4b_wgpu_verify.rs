#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use common::TINY_E4B_CONFIG;
use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_e4b_wgpu::{E4bHostLayer, E4bHostWeights, Gemma4E4bWgpu, HostLin};
use common::LcgShift33Centered0p1 as Lcg;
use common::e4b_snapshot_dir;
use common::tiny_e4b_host_weights;

fn tiny_model(max_seq: usize) -> Gemma4E4bWgpu {
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let weights = tiny_e4b_host_weights(&config, 0x5eed);
    Gemma4E4bWgpu::new(config, &weights, max_seq).unwrap()
}

fn stepped_replay(m: &mut Gemma4E4bWgpu, batch: &[u32]) -> Vec<u32> {
    let pos0 = m.current_pos();
    let out: Vec<u32> = batch.iter().map(|&t| m.decode_step(t).unwrap()).collect();
    m.truncate_to(pos0).unwrap();
    out
}

#[test]
fn verify_chain_matches_stepped_decode_at_random_depths() {
    if ctx_or_skip().is_none() {
        return;
    }
    let mut m = tiny_model(64);
    let cap = m.verify_max_rows();
    assert!(
        (1..=9).contains(&cap),
        "verify_max_rows {cap} out of range with default prefill"
    );
    assert_eq!(cap, m.prefill_chunk_len().min(9));

    let vocab = m.config().vocab_size;
    let window = m.config().sliding_window;
    let prefix: Vec<u32> = (0..21u32).map(|i| (i * 37 + 5) % vocab as u32).collect();
    assert!(
        prefix.len() > window,
        "prefix must cross the sliding window"
    );
    for &t in &prefix {
        m.decode_step(t).unwrap();
    }

    let mut rng = Lcg(0xc0ffee);
    for round in 0..10 {
        let k = 1 + round % cap;
        let batch: Vec<u32> = (0..k).map(|_| rng.token(vocab)).collect();
        let pos0 = m.current_pos();
        let va = m.verify_chain(&batch).unwrap();
        assert_eq!(m.current_pos(), pos0, "verify_chain must not move pos");
        assert_eq!(va.len(), k);
        let sa = stepped_replay(&mut m, &batch);
        assert_eq!(va, sa, "round {round}: verify argmax != stepped argmax");
        let commit = 1 + (round * 7) % k;
        m.advance(commit).unwrap();
        assert_eq!(m.current_pos(), pos0 + commit);
    }
}

#[test]
fn verify_chain_matches_a_never_rolled_back_model() {
    if ctx_or_skip().is_none() {
        return;
    }
    let mut a = tiny_model(96);
    let mut b = tiny_model(96);
    let cap = a.verify_max_rows();
    assert!(cap >= 2);
    let vocab = a.config().vocab_size;
    let prefix: Vec<u32> = (0..13u32).map(|i| (i * 53 + 11) % vocab as u32).collect();
    for &t in &prefix {
        a.decode_step(t).unwrap();
        b.decode_step(t).unwrap();
    }

    let mut rng = Lcg(0xfeedbee5);
    for round in 0..8 {
        let k = 1 + rng.next_u32() as usize % cap;
        let batch: Vec<u32> = (0..k).map(|_| rng.token(vocab)).collect();
        let commit = 1 + rng.next_u32() as usize % k;
        let va = a.verify_chain(&batch).unwrap();
        let vb: Vec<u32> = batch[..commit]
            .iter()
            .map(|&t| b.decode_step(t).unwrap())
            .collect();
        assert_eq!(
            &va[..commit],
            &vb[..],
            "round {round}: rolled-back model diverged from straight-line model"
        );
        a.advance(commit).unwrap();
        assert_eq!(a.current_pos(), b.current_pos());
    }
}

struct ChainSim {
    bonus: u32,
    emitted: Vec<u32>,
}

impl ChainSim {
    fn new(seed: u32) -> Self {
        Self {
            bonus: seed,
            emitted: vec![seed],
        }
    }

    fn round(&mut self, m: &mut Gemma4E4bWgpu, draft: &[u32]) -> usize {
        let mut batch = Vec::with_capacity(draft.len() + 1);
        batch.push(self.bonus);
        batch.extend_from_slice(draft);
        let amax = m.verify_chain(&batch).unwrap();
        let mut commit = 1;
        while commit < batch.len() && batch[commit] == amax[commit - 1] {
            commit += 1;
        }
        m.advance(commit).unwrap();
        self.emitted.extend_from_slice(&batch[1..commit]);
        self.bonus = amax[commit - 1];
        self.emitted.push(self.bonus);
        commit
    }
}

#[test]
fn chain_rounds_reproduce_the_plain_greedy_stream() {
    if ctx_or_skip().is_none() {
        return;
    }
    let mut m = tiny_model(64);
    let cap = m.verify_max_rows();
    assert!(cap >= 2);
    let vocab = m.config().vocab_size;
    let prefix: Vec<u32> = (0..11u32).map(|i| (i * 29 + 3) % vocab as u32).collect();

    let mut seed = 0u32;
    for &t in &prefix {
        seed = m.decode_step(t).unwrap();
    }
    let n_gen = 20usize;
    let mut plain = vec![seed];
    let mut t = seed;
    for _ in 1..n_gen {
        t = m.decode_step(t).unwrap();
        plain.push(t);
    }

    m.reset();
    let mut seed2 = 0u32;
    for &t in &prefix {
        seed2 = m.decode_step(t).unwrap();
    }
    assert_eq!(seed2, seed, "refeed after reset must reproduce the seed");

    let mut rng = Lcg(0xdeadbeef);
    let mut sim = ChainSim::new(seed);
    let mut mode = 0usize;
    while sim.emitted.len() < n_gen {
        let done = sim.emitted.len();
        let left = n_gen - done;
        let k = (1 + rng.next_u32() as usize % (cap - 1)).min(left.max(1));
        let draft: Vec<u32> = match mode % 4 {
            0 => plain[done..(done + k).min(plain.len())].to_vec(),
            1 => (0..k).map(|_| rng.token(vocab)).collect(),
            2 => {
                let mut d = plain[done..(done + k).min(plain.len())].to_vec();
                if !d.is_empty() {
                    let flip = d.len() / 2;
                    d[flip] = (d[flip] + 1) % vocab as u32;
                }
                d
            }
            _ => Vec::new(),
        };
        mode += 1;
        if draft.is_empty() {
            let next = m.decode_step(sim.bonus).unwrap();
            sim.bonus = next;
            sim.emitted.push(next);
            continue;
        }
        sim.round(&mut m, &draft);
    }

    assert_eq!(
        &sim.emitted[..n_gen],
        &plain[..],
        "chain-loop emitted stream must equal the plain greedy stream"
    );
}

#[test]
fn verify_chain_rejects_bad_batches() {
    if ctx_or_skip().is_none() {
        return;
    }
    let mut m = tiny_model(64);
    let cap = m.verify_max_rows();
    m.decode_step(1).unwrap();
    assert!(m.verify_chain(&[]).is_err(), "empty batch must fail");
    let long: Vec<u32> = vec![1; cap + 1];
    assert!(m.verify_chain(&long).is_err(), "oversized batch must fail");
    let oov = vec![m.config().vocab_size as u32];
    assert!(
        m.verify_chain(&oov).is_err(),
        "out-of-vocab token must fail"
    );
    assert!(m.truncate_to(m.current_pos() + 1).is_err());
    let pos = m.current_pos();
    m.truncate_to(pos).unwrap();
    assert_eq!(m.current_pos(), pos);
}

#[test]
#[ignore]
fn real_e4b_verify_chain_parity_and_lossless_stream() {
    if std::env::var("NV_E4B_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping: set NV_E4B_WGPU_TEST=1 to run");
        return;
    }
    let Some(_ctx) = ctx_or_skip() else { return };
    let dir = e4b_snapshot_dir();
    eprintln!("E4B checkpoint: {}", dir.display());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let max_seq: usize = std::env::var("NV_E4B_WGPU_MAXSEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    let window = config.sliding_window;
    assert!(
        max_seq > window + 128,
        "max_seq {max_seq} too small to cross the sliding window {window}"
    );

    let t_build = std::time::Instant::now();
    let mut m = Gemma4E4bWgpu::from_loader(config.clone(), &loader, max_seq).unwrap();
    eprintln!("from_loader: {:.1}s", t_build.elapsed().as_secs_f64());
    let cap = m.verify_max_rows();
    assert!(cap >= 2, "verify epilogue disabled (cap {cap})");
    eprintln!("verify_max_rows: {cap}");

    let vocab = config.vocab_size;
    let prefix_len = window + 48;
    let mut rng = Lcg(0x5eed_cafe);
    let prefix: Vec<u32> = (0..prefix_len)
        .map(|_| 2 + rng.next_u32() % 30_000)
        .collect();

    let mut seed = 0u32;
    let done = m.prefill_tokens(&prefix[..prefix.len() - 1]).unwrap();
    for &t in &prefix[done..] {
        seed = m.decode_step(t).unwrap();
    }
    let n_gen = 48usize;
    let mut plain = vec![seed];
    let mut t = seed;
    for _ in 1..n_gen {
        t = m.decode_step(t).unwrap();
        plain.push(t);
    }
    eprintln!("plain greedy stream: {:?}", &plain[..12.min(plain.len())]);

    m.reset();
    let mut seed2 = 0u32;
    let done2 = m.prefill_tokens(&prefix[..prefix.len() - 1]).unwrap();
    assert_eq!(done2, done);
    for &t in &prefix[done2..] {
        seed2 = m.decode_step(t).unwrap();
    }
    assert_eq!(seed2, seed, "refeed after reset must reproduce the seed");

    let mut sim = ChainSim::new(seed);
    let mut mode = 0usize;
    let mut rounds = 0usize;
    let mut parity_checked = 0usize;
    while sim.emitted.len() < n_gen {
        let done = sim.emitted.len();
        let left = n_gen - done;
        let k = (1 + rng.next_u32() as usize % (cap - 1)).min(left.max(1));
        let draft: Vec<u32> = match mode % 4 {
            0 => plain[done..(done + k).min(plain.len())].to_vec(),
            1 => (0..k).map(|_| rng.token(vocab)).collect(),
            2 => {
                let mut d = plain[done..(done + k).min(plain.len())].to_vec();
                if !d.is_empty() {
                    let flip = d.len() / 2;
                    d[flip] = (d[flip] + 1) % vocab as u32;
                }
                d
            }
            _ => plain[done..(done + k).min(plain.len())].to_vec(),
        };
        mode += 1;
        if draft.is_empty() {
            let next = m.decode_step(sim.bonus).unwrap();
            sim.bonus = next;
            sim.emitted.push(next);
            continue;
        }
        if rounds.is_multiple_of(3) {
            let mut batch = vec![sim.bonus];
            batch.extend_from_slice(&draft);
            let pos0 = m.current_pos();
            let va = m.verify_chain(&batch).unwrap();
            assert_eq!(m.current_pos(), pos0);
            let sa = stepped_replay(&mut m, &batch);
            assert_eq!(
                va, sa,
                "round {rounds}: verify argmax != stepped argmax at pos {pos0}"
            );
            parity_checked += 1;
        }
        sim.round(&mut m, &draft);
        rounds += 1;
    }

    assert!(parity_checked >= 3, "too few parity-checked rounds");
    assert_eq!(
        &sim.emitted[..n_gen],
        &plain[..],
        "chain-loop emitted stream must equal the plain greedy stream"
    );
    eprintln!(
        "real E4B verify parity: {} rounds ({} with stepped replay), {} tokens lossless",
        rounds, parity_checked, n_gen
    );
}
