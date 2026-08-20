use nv_ocr::ctc::{beam_decode, best_path, codes_to_unichars, CodeStep};
use nv_ocr::lstm::{lstm_forward, Buf, Tensor};
use nv_ocr::traineddata::Cursor;
use nv_ocr::vgsl::{LstmLayer, WeightMatrix, Weights};
use nv_ocr::{Logits, Recoder, Unicharset};

fn wm(rows: usize, cols: usize, w: Vec<f32>) -> WeightMatrix {
    assert_eq!(w.len(), rows * cols);
    WeightMatrix {
        rows,
        cols,
        weights: Weights::Float(w),
    }
}

fn f32_data(t: &Tensor) -> &[f32] {
    match &t.buf {
        Buf::F32(v) => v,
        _ => panic!("expected f32"),
    }
}

#[test]
fn lstm_cell_matches_hand_computation() {
    let layer = LstmLayer {
        ni: 1,
        ns: 1,
        na: 2,
        summarizing: false,
        gates: [
            wm(1, 3, vec![1.0, 0.5, 0.1]),
            wm(1, 3, vec![0.2, -0.3, 1.5]),
            wm(1, 3, vec![-0.4, 0.6, -0.5]),
            wm(1, 3, vec![0.7, 0.2, 0.8]),
        ],
    };
    let input = Tensor {
        h: 1,
        w: 3,
        d: 1,
        buf: Buf::F32(vec![0.5, -0.25, 1.0]),
    };
    let out = lstm_forward(&layer, &input).unwrap();
    let got = f32_data(&out);
    let expected = [0.3184584, 0.1361782, 0.5245237];
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!(
            (g - e).abs() < 1e-5,
            "got {:?} expected {:?}",
            got,
            expected
        );
    }
}

#[test]
fn summarizing_lstm_emits_one_step_per_row_and_resets_state() {
    let layer = LstmLayer {
        ni: 1,
        ns: 1,
        na: 2,
        summarizing: true,
        gates: [
            wm(1, 3, vec![1.0, 0.5, 0.1]),
            wm(1, 3, vec![0.2, -0.3, 1.5]),
            wm(1, 3, vec![-0.4, 0.6, -0.5]),
            wm(1, 3, vec![0.7, 0.2, 0.8]),
        ],
    };
    let input = Tensor {
        h: 2,
        w: 3,
        d: 1,
        buf: Buf::F32(vec![0.5, -0.25, 1.0, 0.5, -0.25, 1.0]),
    };
    let out = lstm_forward(&layer, &input).unwrap();
    assert_eq!((out.h, out.w, out.d), (2, 1, 1));
    let d = f32_data(&out);
    assert!((d[0] - 0.5245237).abs() < 1e-5);
    assert_eq!(d[0], d[1]);
}

fn logits(timesteps: usize, classes: usize, data: Vec<f32>) -> Logits {
    assert_eq!(data.len(), timesteps * classes);
    Logits {
        data,
        timesteps,
        classes,
    }
}

#[test]
fn best_path_collapses_repeats_and_blanks() {
    let l = logits(
        6,
        3,
        vec![
            0.9, 0.05, 0.05, 0.9, 0.05, 0.05, 0.05, 0.05, 0.9, 0.9, 0.05, 0.05, 0.05, 0.9, 0.05,
            0.05, 0.9, 0.05,
        ],
    );
    let steps = best_path(&l, 2).unwrap();
    let codes: Vec<usize> = steps.iter().map(|s| s.code).collect();
    assert_eq!(codes, vec![0, 0, 1]);
    assert_eq!(steps[0].t, 0);
    assert_eq!(steps[1].t, 3);
    assert_eq!(steps[2].t, 4);
}

#[test]
fn beam_beats_best_path_on_blank_mass_split() {
    let l = logits(2, 2, vec![0.4, 0.6, 0.4, 0.6]);
    assert!(best_path(&l, 1).unwrap().is_empty());
    let beam = beam_decode(&l, 1, 8).unwrap();
    let codes: Vec<usize> = beam.iter().map(|s| s.code).collect();
    assert_eq!(codes, vec![0]);
}

#[test]
fn beam_matches_best_path_on_confident_sequence() {
    let l = logits(
        5,
        3,
        vec![
            0.98, 0.01, 0.01, 0.01, 0.01, 0.98, 0.01, 0.98, 0.01, 0.01, 0.98, 0.01, 0.98, 0.01,
            0.01,
        ],
    );
    let bp: Vec<usize> = best_path(&l, 2).unwrap().iter().map(|s| s.code).collect();
    let bd: Vec<usize> = beam_decode(&l, 2, 8)
        .unwrap()
        .iter()
        .map(|s| s.code)
        .collect();
    assert_eq!(bp, vec![0, 1, 0]);
    assert_eq!(bd, bp);
}

#[test]
fn beam_requires_blank_between_repeats() {
    let l = logits(3, 2, vec![0.95, 0.05, 0.05, 0.95, 0.95, 0.05]);
    let codes: Vec<usize> = beam_decode(&l, 1, 8)
        .unwrap()
        .iter()
        .map(|s| s.code)
        .collect();
    assert_eq!(codes, vec![0, 0]);
}

fn synthetic_recoder(entries: &[(bool, &[i32])]) -> Recoder {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (sn, codes) in entries {
        bytes.push(*sn as u8);
        bytes.extend_from_slice(&(codes.len() as u32).to_le_bytes());
        for c in *codes {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
    }
    Recoder::deserialize(&mut Cursor::new(&bytes)).unwrap()
}

#[test]
fn recoder_decodes_single_and_multi_code_sequences() {
    let ucs =
        Unicharset::parse(b"4\nNULL 0 Common 0\na 3 Latin 1\nb 3 Latin 2\nd 3 Latin 3\n").unwrap();
    assert_eq!(ucs.len(), 4);
    assert_eq!(ucs.glyph(0), Some(" "));
    assert_eq!(ucs.glyph(1), Some("a"));
    let rec = synthetic_recoder(&[(true, &[0]), (true, &[1]), (true, &[2]), (true, &[3, 4])]);
    assert_eq!(rec.code_range(), 5);
    assert_eq!(rec.decode(&[1]), Some(1));
    assert_eq!(rec.decode(&[3, 4]), Some(3));
    assert_eq!(rec.decode(&[3]), None);
    assert!(rec.is_prefix(&[3]));
    assert!(!rec.is_prefix(&[1]));
    let steps: Vec<CodeStep> = [(1usize, 0usize), (3, 2), (4, 3), (2, 5), (0, 7)]
        .iter()
        .map(|&(code, t)| CodeStep { code, t, prob: 1.0 })
        .collect();
    let chars = codes_to_unichars(&steps, &rec, &ucs);
    let text: String = chars.iter().map(|c| c.text.as_str()).collect();
    assert_eq!(text, "adb ");
    assert_eq!(chars[1].t, 2);
    assert_eq!(chars[1].unichar_id, 3);
}

#[test]
fn unknown_code_sequences_are_dropped() {
    let ucs = Unicharset::parse(b"2\nNULL 0 Common 0\na 3 Latin 1\n").unwrap();
    let rec = synthetic_recoder(&[(true, &[0]), (true, &[1])]);
    let steps = vec![
        CodeStep {
            code: 9,
            t: 0,
            prob: 1.0,
        },
        CodeStep {
            code: 1,
            t: 1,
            prob: 1.0,
        },
    ];
    let chars = codes_to_unichars(&steps, &rec, &ucs);
    let text: String = chars.iter().map(|c| c.text.as_str()).collect();
    assert_eq!(text, "a");
}
