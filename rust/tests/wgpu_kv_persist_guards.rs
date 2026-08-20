#![cfg(feature = "wgpu")]

use speaches_plus::oapi::chat_engine_wgpu::persist::{restore_file, save_file, KvDisk, Meta};

const LAYERS: usize = 3;
const CHUNK_WORDS: [usize; 4] = [16, 16, 8, 8];

struct FakeDecoder {
    layers: usize,
    lens: [usize; 4],
    prefill_chunk: usize,
    sliding_window: usize,
    restored: Vec<usize>,
    restored_pos: Option<usize>,
}

impl FakeDecoder {
    fn new() -> Self {
        Self {
            layers: LAYERS,
            lens: CHUNK_WORDS,
            prefill_chunk: 64,
            sliding_window: 512,
            restored: Vec::new(),
            restored_pos: None,
        }
    }
}

impl KvDisk for FakeDecoder {
    fn kv_layer_count(&self) -> usize {
        self.layers
    }
    fn kv_layer_lens(&self, li: usize) -> Option<[usize; 4]> {
        (li < self.layers).then_some(self.lens)
    }
    fn kv_cache_snapshot(
        &self,
        li: usize,
    ) -> anyhow::Result<Option<(Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>)>> {
        if li >= self.layers {
            return Ok(None);
        }
        let seed = li as u32;
        Ok(Some((
            (0..self.lens[0] as u32).map(|i| i + seed).collect(),
            (0..self.lens[1] as u32).map(|i| i * 3 + seed).collect(),
            (0..self.lens[2]).map(|i| i as f32 + seed as f32).collect(),
            (0..self.lens[3]).map(|i| i as f32 * 0.5).collect(),
        )))
    }
    fn kv_cache_restore(
        &mut self,
        li: usize,
        _snap: &(Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>),
    ) -> anyhow::Result<bool> {
        self.restored.push(li);
        Ok(true)
    }
    fn restore_pos(&mut self, pos: usize) -> anyhow::Result<()> {
        self.restored_pos = Some(pos);
        Ok(())
    }
    fn prefill_chunk_len(&self) -> usize {
        self.prefill_chunk
    }
    fn sliding_window(&self) -> usize {
        self.sliding_window
    }
}

fn meta() -> Meta {
    Meta {
        model_id: "google/gemma-4-E4B-it".into(),
        kind: "gemma4-e4b (nv-models::gemma4_e4b_wgpu)".into(),
        adapter: None,
        max_seq: 4096,
    }
}

const TOKENS: [u32; 6] = [11, 22, 33, 44, 55, 66];
const FRONTIER: usize = 9;

fn written(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
    let path = dir.join(format!("{name}.nvkv"));
    save_file(&path, &meta(), &FakeDecoder::new(), &TOKENS, FRONTIER);
    assert!(
        path.exists(),
        "save_file wrote nothing, so every refusal below would pass vacuously"
    );
    path
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("nvkv-guards-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn refusal(path: &std::path::Path, m: &Meta, model: &mut FakeDecoder) -> String {
    match restore_file(path, m, model) {
        Err(e) => format!("{e:#}"),
        Ok(other) => panic!("expected a refusal, got {:?}", other.map(|(t, f)| (t.len(), f))),
    }
}

#[test]
fn a_snapshot_round_trips_through_every_layer_and_restores_its_frontier() {
    let d = tmp("roundtrip");
    let path = written(&d, "m");
    let mut model = FakeDecoder::new();
    let got = restore_file(&path, &meta(), &mut model)
        .expect("restore must not error on the snapshot just written")
        .expect("restore must find the snapshot just written");
    assert_eq!(got.0, TOKENS, "tokens must survive the round trip");
    assert_eq!(
        got.1, FRONTIER,
        "the frontier is prompt + generated and is what plan_prefix_reuse feeds \
         rewind_limits().target(); returning tokens.len() here is the inversion \
         badd695fc fixed"
    );
    assert_eq!(
        model.restored,
        (0..LAYERS).collect::<Vec<_>>(),
        "every live layer must be handed to kv_cache_restore"
    );
    assert_eq!(model.restored_pos, Some(FRONTIER));
}

#[test]
fn a_snapshot_from_a_different_model_is_refused() {
    let d = tmp("model");
    let path = written(&d, "m");
    let mut other = meta();
    other.model_id = "google/gemma-4-31B-it".into();
    let e = refusal(&path, &other, &mut FakeDecoder::new());
    assert!(e.contains("model"), "wrong reason: {e}");
}

#[test]
fn a_snapshot_from_a_different_kind_is_refused() {
    let d = tmp("kind");
    let path = written(&d, "m");
    let mut other = meta();
    other.kind = "qwen3_5-dense (nv-models::qwen3_5_dense_wgpu)".into();
    let e = refusal(&path, &other, &mut FakeDecoder::new());
    assert!(e.contains("kind"), "wrong reason: {e}");
}

#[test]
fn a_snapshot_taken_without_a_lora_adapter_is_refused_for_one_with() {
    let d = tmp("adapter");
    let path = written(&d, "m");
    let mut other = meta();
    other.adapter = Some("some-adapter".into());
    let e = refusal(&path, &other, &mut FakeDecoder::new());
    assert!(e.contains("adapter"), "wrong reason: {e}");
}

#[test]
fn a_snapshot_from_a_different_max_seq_is_refused() {
    let d = tmp("maxseq");
    let path = written(&d, "m");
    let mut other = meta();
    other.max_seq = 2048;
    let e = refusal(&path, &other, &mut FakeDecoder::new());
    assert!(e.contains("max_seq"), "wrong reason: {e}");
}

#[test]
fn a_snapshot_from_a_different_prefill_chunk_is_refused() {
    let d = tmp("chunk");
    let path = written(&d, "m");
    let mut model = FakeDecoder::new();
    model.prefill_chunk = 128;
    let e = refusal(&path, &meta(), &mut model);
    assert!(e.contains("prefill_chunk"), "wrong reason: {e}");
}

#[test]
fn a_snapshot_from_a_different_sliding_window_is_refused() {
    let d = tmp("window");
    let path = written(&d, "m");
    let mut model = FakeDecoder::new();
    model.sliding_window = 1024;
    let e = refusal(&path, &meta(), &mut model);
    assert!(e.contains("sliding_window"), "wrong reason: {e}");
}

#[test]
fn a_snapshot_whose_kv_geometry_does_not_match_the_decoder_is_refused() {
    let d = tmp("geom");
    let path = written(&d, "m");

    let mut fewer = FakeDecoder::new();
    fewer.layers = LAYERS - 1;
    let e = refusal(&path, &meta(), &mut fewer);
    assert!(e.contains("geometry"), "layer count, wrong reason: {e}");

    let mut wider = FakeDecoder::new();
    wider.lens = [32, 16, 8, 8];
    let e = refusal(&path, &meta(), &mut wider);
    assert!(e.contains("geometry"), "chunk width, wrong reason: {e}");
}

#[test]
fn a_truncated_or_corrupted_snapshot_is_refused_by_the_payload_hash() {
    let d = tmp("hash");
    let path = written(&d, "m");
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(&path, &bytes).unwrap();
    let e = refusal(&path, &meta(), &mut FakeDecoder::new());
    assert!(
        e.contains("payload hash"),
        "a flipped payload byte must reach the hash, not die earlier: {e}"
    );

    std::fs::write(&path, &bytes[..bytes.len() - 8]).unwrap();
    let e = refusal(&path, &meta(), &mut FakeDecoder::new());
    assert!(e.contains("file length"), "wrong reason: {e}");
}

#[test]
fn a_file_that_is_not_a_snapshot_is_refused_before_anything_is_read() {
    let d = tmp("magic");
    let path = d.join("m.nvkv");
    std::fs::write(&path, b"not a snapshot at all, but long enough to have a header").unwrap();
    let e = refusal(&path, &meta(), &mut FakeDecoder::new());
    assert!(e.contains("magic"), "wrong reason: {e}");

    std::fs::write(&path, b"short").unwrap();
    let e = refusal(&path, &meta(), &mut FakeDecoder::new());
    assert!(e.contains("shorter than magic"), "wrong reason: {e}");
}

#[test]
fn a_missing_snapshot_is_not_an_error() {
    let d = tmp("missing");
    let got = restore_file(&d.join("nothing.nvkv"), &meta(), &mut FakeDecoder::new())
        .expect("a cold start must not be an error");
    assert!(got.is_none(), "a cold start must restore nothing");
}
