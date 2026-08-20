use candle_core::{DType, Device, Tensor};
use nv_models::dense_train::naive_sdpa;

const BATCH: usize = 2;
const SEQ: usize = 3;
const STORED: usize = 5;
const N_Q: usize = 4;
const N_KV: usize = 2;
const HEAD_DIM: usize = 8;

fn seeded(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn attention_in_f64(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    window: Option<usize>,
) -> Vec<f64> {
    let group = N_Q / N_KV;
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    let key_base = STORED - SEQ;
    let mut out = vec![0f64; SEQ * N_Q * HEAD_DIM];
    for h in 0..N_Q {
        let kvh = h / group;
        for i in 0..SEQ {
            let qpos = (key_base + i) as i64;
            let mut logits = vec![f64::NEG_INFINITY; STORED];
            for (j, logit) in logits.iter_mut().enumerate() {
                let kpos = j as i64;
                let visible = kpos <= qpos
                    && match window {
                        Some(w) => qpos - kpos < w as i64,
                        None => true,
                    };
                if !visible {
                    continue;
                }
                let mut dot = 0f64;
                for d in 0..HEAD_DIM {
                    dot += q[(i * N_Q + h) * HEAD_DIM + d] as f64
                        * k[(j * N_KV + kvh) * HEAD_DIM + d] as f64;
                }
                *logit = dot * scale;
            }
            let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> =
                logits.iter().map(|l| if l.is_finite() { (l - max).exp() } else { 0.0 }).collect();
            let denom: f64 = exps.iter().sum();
            assert!(denom > 0.0, "row {i} head {h} attends to nothing, so this row is vacuous");
            for d in 0..HEAD_DIM {
                let mut acc = 0f64;
                for (j, e) in exps.iter().enumerate() {
                    acc += (e / denom) * v[(j * N_KV + kvh) * HEAD_DIM + d] as f64;
                }
                out[(i * N_Q + h) * HEAD_DIM + d] = acc;
            }
        }
    }
    out
}

fn run(window: Option<usize>) -> Vec<(Vec<f32>, Vec<f64>)> {
    let dev = Device::Cpu;
    let seeds: [(u64, u64, u64); BATCH] = [(11, 22, 33), (44, 55, 66)];
    let mut qh_all = Vec::new();
    let mut kh_all = Vec::new();
    let mut vh_all = Vec::new();
    let mut per_row: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
    for (sq, sk, sv) in seeds {
        let qh = seeded(SEQ * N_Q * HEAD_DIM, sq);
        let kh = seeded(STORED * N_KV * HEAD_DIM, sk);
        let vh = seeded(STORED * N_KV * HEAD_DIM, sv);
        qh_all.extend_from_slice(&qh);
        kh_all.extend_from_slice(&kh);
        vh_all.extend_from_slice(&vh);
        per_row.push((qh, kh, vh));
    }
    let q = Tensor::from_vec(qh_all, (BATCH, SEQ, N_Q * HEAD_DIM), &dev).unwrap();
    let k = Tensor::from_vec(kh_all, (BATCH, STORED, N_KV * HEAD_DIM), &dev).unwrap();
    let v = Tensor::from_vec(vh_all, (BATCH, STORED, N_KV * HEAD_DIM), &dev).unwrap();

    let got = naive_sdpa(&q, &k, &v, BATCH, N_Q, N_KV, HEAD_DIM, SEQ, window).unwrap();
    assert_eq!(got.dims(), &[BATCH, SEQ, N_Q, HEAD_DIM]);
    let got_flat: Vec<f32> = got
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();
    let row_len = SEQ * N_Q * HEAD_DIM;
    per_row
        .iter()
        .enumerate()
        .map(|(bi, (qh, kh, vh))| {
            let got_row = got_flat[bi * row_len..(bi + 1) * row_len].to_vec();
            let want_row = attention_in_f64(qh, kh, vh, window);
            (got_row, want_row)
        })
        .collect()
}

fn assert_matches(got: &[f32], want: &[f64], label: &str) {
    let mut worst = 0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        assert!(g.is_finite(), "{label}: non-finite output");
        worst = worst.max((*g as f64 - w).abs() / w.abs().max(1e-3));
    }
    assert!(
        worst <= 1e-5,
        "{label}: the attention dense_train.rs trains through differs from an independent \
         f64 attention by {worst:e} relative. Every adapter is fitted through this op, so a \
         defect here is baked into the weights and never reported."
    );
}

#[test]
fn the_attention_the_trainer_trains_through_matches_an_independent_f64_attention() {
    for (bi, (got, want)) in run(None).iter().enumerate() {
        assert_matches(got, want, &format!("causal row {bi}"));
    }
}

#[test]
fn the_sliding_window_the_trainer_trains_through_matches_the_same_oracle() {
    for (bi, (got, want)) in run(Some(2)).iter().enumerate() {
        assert_matches(got, want, &format!("window=2 row {bi}"));
    }
}

#[test]
fn the_window_and_the_gqa_mapping_both_change_the_answer_which_is_what_makes_the_rows_above_gates()
{

    let open_rows = run(None);
    let windowed_rows = run(Some(2));
    for bi in 0..BATCH {
        let open = &open_rows[bi].0;
        let windowed = &windowed_rows[bi].0;
        assert!(
            open.iter().zip(windowed.iter()).any(|(a, b)| (a - b).abs() > 1e-4),
            "row {bi}: the window changes nothing on this fixture, so the windowed row proves nothing"
        );

        let head0 = &open[..HEAD_DIM];
        let head_last = &open[(N_Q - 1) * HEAD_DIM..N_Q * HEAD_DIM];
        assert!(
            head0.iter().zip(head_last.iter()).any(|(a, b)| (a - b).abs() > 1e-4),
            "row {bi}: all query heads produced the same output, so the GQA mapping is untested"
        );
    }
    assert_eq!(STORED - SEQ, 2, "key_base must be nonzero or the offset goes untested");

    let (row0, _) = &open_rows[0];
    let (row1, _) = &open_rows[1];
    assert!(
        row0.iter().zip(row1.iter()).any(|(a, b)| (a - b).abs() > 1e-4),
        "both batch rows produced the same output on different inputs, so the batch \
         dimension is copying row 0 instead of attending each row to its own keys"
    );
}

#[test]
fn a_key_run_longer_than_the_query_run_still_lets_every_query_see_its_own_position() {

    assert!(STORED > SEQ, "this row is about the stored > seq case and needs one");
    for (bi, (got, want)) in run(None).iter().enumerate() {
        assert_matches(got, want, &format!("stored > seq row {bi}"));
        assert!(
            got.iter().all(|g| g.is_finite()),
            "row {bi}: a query row attended to nothing, so its softmax ran over an all -inf row"
        );
    }
}
