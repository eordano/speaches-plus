
use nv_ocr::traineddata::{Cursor, TD_LSTM, TD_LSTM_UNICHARSET, TD_VERSION};
use nv_ocr::Traineddata;

fn container(offsets: &[i64], payload: &[u8]) -> Vec<u8> {
    let mut v = (offsets.len() as u32).to_le_bytes().to_vec();
    for o in offsets {
        v.extend_from_slice(&o.to_le_bytes());
    }
    v.extend_from_slice(payload);
    v
}

fn full_offset_table(present: &[(usize, &[u8])], total_slots: usize) -> Vec<u8> {
    let base = 4 + 8 * total_slots;
    let mut offsets = vec![-1i64; total_slots];
    let mut payload = Vec::new();
    for (slot, bytes) in present {
        offsets[*slot] = (base + payload.len()) as i64;
        payload.extend_from_slice(bytes);
    }
    container(&offsets, &payload)
}

#[test]
fn a_component_extends_to_the_next_present_offset_skipping_absent_slots() {

    let data = full_offset_table(&[(TD_LSTM, b"lstm-bytes"), (TD_LSTM_UNICHARSET, b"ucs")], 24);
    let td = Traineddata::parse(&data).unwrap();
    assert_eq!(td.component(TD_LSTM).unwrap(), b"lstm-bytes");
    assert_eq!(td.component(TD_LSTM_UNICHARSET).unwrap(), b"ucs");
    assert!(td.component(TD_LSTM + 1).is_none(), "absent slot must read as None");
}

#[test]
fn the_last_component_runs_to_eof() {
    let data = full_offset_table(&[(TD_VERSION, b"5.3.0\n")], 24);
    let td = Traineddata::parse(&data).unwrap();
    assert_eq!(td.component(TD_VERSION).unwrap(), b"5.3.0\n");
}

#[test]
fn version_trims_the_trailing_nul_and_newline_stock_files_carry() {
    let data = full_offset_table(&[(TD_VERSION, b"4.00.00alpha:eng:synth\n\0")], 24);
    let td = Traineddata::parse(&data).unwrap();
    assert_eq!(td.version().unwrap(), "4.00.00alpha:eng:synth");
}

#[test]
fn into_components_moves_the_lstm_pieces_and_leaves_the_rest_behind() {
    let data = full_offset_table(&[(TD_LSTM, b"net"), (TD_LSTM_UNICHARSET, b"ucs")], 24);
    let c = Traineddata::parse(&data).unwrap().into_components();
    assert_eq!(c.lstm.as_deref().unwrap(), b"net");
    assert_eq!(c.lstm_unicharset.as_deref().unwrap(), b"ucs");
    assert!(c.lstm_recoder.is_none() && c.version.is_none());
}

#[test]
fn every_truncation_and_corruption_is_an_error_never_a_panic() {

    for (label, data) in [
        ("empty", vec![]),
        ("three header bytes", vec![2, 0, 0]),
        ("zero entries", container(&[], b"")),
        ("implausible count", 1000u32.to_le_bytes().to_vec()),
        ("table cut short", {
            let mut v = 4u32.to_le_bytes().to_vec();
            v.extend_from_slice(&8i64.to_le_bytes());
            v
        }),
        ("offset past eof", container(&[100], b"short")),
        ("offsets out of order", {
            let base = 4 + 8 * 2;
            container(&[(base + 4) as i64, base as i64], b"abcdefgh")
        }),
    ] {
        assert!(
            Traineddata::parse(&data).is_err(),
            "{label}: parse accepted corrupt input"
        );
    }
}

#[test]
fn an_all_absent_table_parses_and_answers_none_everywhere() {

    let td = Traineddata::parse(&full_offset_table(&[], 24)).unwrap();
    for slot in 0..24 {
        assert!(td.component(slot).is_none());
    }
    assert!(td.version().is_none());
}

#[test]
fn the_cursor_reports_eof_and_oversized_strings_instead_of_panicking() {
    let mut c = Cursor::new(b"ab");
    assert_eq!(c.bytes(2).unwrap(), b"ab");
    assert!(c.bytes(1).is_err(), "reading past eof must be an error");

    let mut c = Cursor::new(&[255, 255, 255, 255, b'x']);
    assert!(c.string().is_err(), "a 4 GB length prefix in a 5-byte buffer must not allocate");

    let mut c = Cursor::new(&[2, 0, 0, 0, 0xff, 0xfe]);
    assert_eq!(c.string().unwrap(), "\u{fffd}\u{fffd}");
}
