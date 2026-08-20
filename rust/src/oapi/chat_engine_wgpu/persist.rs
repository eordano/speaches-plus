use std::path::PathBuf;
use std::time::{Duration, Instant};

use nv_models::gemma4_e4b_wgpu::{Gemma4E4bWgpu, KvCacheSnapshot};
use nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu;
use nv_models::gemma4_wgpu::Gemma4Wgpu;
use nv_models::gpt_oss_wgpu::GptOssWgpu;
use nv_models::laguna_wgpu::LagunaWgpu;
use nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseWgpu;
use nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu;
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::Xxh3;

pub trait KvDisk {
    fn kv_layer_count(&self) -> usize;
    fn kv_layer_lens(&self, li: usize) -> Option<[usize; 4]>;
    fn kv_cache_snapshot(&self, li: usize) -> anyhow::Result<Option<KvCacheSnapshot>>;
    fn kv_cache_restore(&mut self, li: usize, snap: &KvCacheSnapshot) -> anyhow::Result<bool>;
    fn restore_pos(&mut self, pos: usize) -> anyhow::Result<()>;
    fn prefill_chunk_len(&self) -> usize;
    fn sliding_window(&self) -> usize;
}

macro_rules! impl_kv_disk {
    ($t:ty) => {
        impl KvDisk for $t {
            fn kv_layer_count(&self) -> usize {
                <$t>::kv_layer_count(self)
            }
            fn kv_layer_lens(&self, li: usize) -> Option<[usize; 4]> {
                <$t>::kv_layer_lens(self, li)
            }
            fn kv_cache_snapshot(&self, li: usize) -> anyhow::Result<Option<KvCacheSnapshot>> {
                <$t>::kv_cache_snapshot(self, li)
            }
            fn kv_cache_restore(
                &mut self,
                li: usize,
                snap: &KvCacheSnapshot,
            ) -> anyhow::Result<bool> {
                <$t>::kv_cache_restore(self, li, snap)
            }
            fn restore_pos(&mut self, pos: usize) -> anyhow::Result<()> {
                <$t>::restore_pos(self, pos)
            }
            fn prefill_chunk_len(&self) -> usize {
                <$t>::prefill_chunk_len(self)
            }
            fn sliding_window(&self) -> usize {
                self.config().sliding_window
            }
        }
    };
}

impl_kv_disk!(Gemma4E4bWgpu);
impl_kv_disk!(Gemma4Wgpu);

macro_rules! impl_state_disk {
    ($t:ty) => {
        impl KvDisk for $t {
            fn kv_layer_count(&self) -> usize {
                self.state_blob_count()
            }
            fn kv_layer_lens(&self, li: usize) -> Option<[usize; 4]> {
                self.state_blob_words(li).map(|w| [w, 0, 0, 0])
            }
            fn kv_cache_snapshot(&self, li: usize) -> anyhow::Result<Option<KvCacheSnapshot>> {
                Ok(self
                    .state_blob_download(li)?
                    .map(|w| (w, Vec::new(), Vec::new(), Vec::new())))
            }
            fn kv_cache_restore(
                &mut self,
                li: usize,
                snap: &KvCacheSnapshot,
            ) -> anyhow::Result<bool> {
                anyhow::ensure!(
                    snap.1.is_empty() && snap.2.is_empty() && snap.3.is_empty(),
                    "a state-registry snapshot carries one blob per layer"
                );
                self.state_blob_restore(li, &snap.0)
            }
            fn restore_pos(&mut self, pos: usize) -> anyhow::Result<()> {
                <$t>::restore_pos(self, pos)
            }
            fn prefill_chunk_len(&self) -> usize {
                0
            }
            fn sliding_window(&self) -> usize {
                0
            }
        }
    };
}

impl_state_disk!(Qwen3MoeWgpu);
impl_state_disk!(Qwen3_5DenseWgpu);
impl_state_disk!(GptOssWgpu);
impl_state_disk!(LagunaWgpu);
impl_state_disk!(Gemma4MoeWgpu);

pub const KV_CACHE_DIR_ENV: &str = "NV_KV_CACHE_DIR";

pub const SNAPSHOT_MAGIC: &[u8; 8] = b"NVKVSNP2";

pub const SAVE_DEBOUNCE: Duration = Duration::from_secs(5);

pub const HEADER_MAX: usize = 4 << 20;

pub const RESPONSE_STORE_KEEPS_NEWEST: usize = 64;

pub fn cache_dir() -> Option<PathBuf> {
    std::env::var(KV_CACHE_DIR_ENV)
        .ok()
        .filter(|d| !d.trim().is_empty())
        .map(PathBuf::from)
}

pub fn response_snapshot_path(dir: &std::path::Path, id: &str) -> PathBuf {
    dir.join(format!("{}.nvkv", sanitize(id)))
}

pub fn gc_response_store(dir: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut snaps: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            let name = p.file_name()?.to_str()?;
            (name.starts_with("resp_") && name.ends_with(".nvkv"))
                .then(|| e.metadata().ok().and_then(|m| m.modified().ok()).map(|t| (t, p)))
                .flatten()
        })
        .collect();
    if snaps.len() <= RESPONSE_STORE_KEEPS_NEWEST {
        return;
    }
    snaps.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, p) in snaps.split_off(RESPONSE_STORE_KEEPS_NEWEST) {
        let _ = std::fs::remove_file(p.with_extension("json"));
        let _ = std::fs::remove_file(p);
    }
}

fn hash_tokens(tokens: &[u32], frontier: usize) -> u64 {
    let mut h = Xxh3::new();
    h.update(&(frontier as u64).to_le_bytes());
    h.update(bytemuck::cast_slice(tokens));
    h.digest()
}

fn snap_chunks(s: &KvCacheSnapshot) -> [&[u8]; 4] {
    [
        bytemuck::cast_slice(&s.0),
        bytemuck::cast_slice(&s.1),
        bytemuck::cast_slice(&s.2),
        bytemuck::cast_slice(&s.3),
    ]
}

fn nonzero_prefix_words(bytes: &[u8]) -> usize {
    let (head, mid, tail) = bytemuck::pod_align_to::<u8, u128>(bytes);
    let end = if let Some(i) = tail.iter().rposition(|&b| b != 0) {
        head.len() + mid.len() * 16 + i + 1
    } else if let Some(i) = mid.iter().rposition(|&w| w != 0) {
        let base = head.len() + i * 16;
        base + bytes[base..base + 16]
            .iter()
            .rposition(|&b| b != 0)
            .map_or(0, |j| j + 1)
    } else if let Some(i) = head.iter().rposition(|&b| b != 0) {
        i + 1
    } else {
        0
    };
    end.div_ceil(4)
}

#[derive(Serialize, Deserialize)]
struct LayerLens {
    li: usize,
    chunks: [(usize, usize); 4],
}

#[derive(Serialize, Deserialize)]
struct Header {
    model_id: String,
    kind: String,
    adapter: Option<String>,
    max_seq: usize,
    prefill_chunk: usize,
    sliding_window: usize,
    frontier: usize,
    tokens: Vec<u32>,
    token_hash: u64,
    layers: Vec<LayerLens>,
    payload_len: u64,
    payload_hash: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Meta {
    pub model_id: String,
    pub kind: String,
    pub adapter: Option<String>,
    pub max_seq: usize,
}

pub struct Session {
    path: PathBuf,
    meta: Meta,
    last_save: Option<(Instant, u64)>,
    writer: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
    }
}

fn sanitize(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

impl Session {
    pub fn from_env(meta: Meta) -> Option<Self> {
        let dir = std::env::var(KV_CACHE_DIR_ENV)
            .ok()
            .filter(|d| !d.trim().is_empty())?;
        let dir = PathBuf::from(dir);
        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::warn!(
                dir = %dir.display(),
                error = %e,
                "kv persistence disabled: cache dir not writable"
            );
            return None;
        }
        let path = dir.join(format!("{}.nvkv", sanitize(&meta.model_id)));
        Some(Self {
            path,
            meta,
            last_save: None,
            writer: None,
        })
    }

    pub fn restore(&self, model: &mut dyn KvDisk) -> Option<(Vec<u32>, usize)> {
        let t = Instant::now();
        match self.try_restore(model) {
            Ok(Some((tokens, frontier))) => {
                tracing::info!(
                    path = %self.path.display(),
                    frontier,
                    prefix_tokens = tokens.len(),
                    restore_ms = t.elapsed().as_millis() as u64,
                    "kv snapshot restored; prefix cache starts warm"
                );
                Some((tokens, frontier))
            }
            Ok(None) => None,
            Err(e) => {
                tracing::warn!(
                    path = %self.path.display(),
                    error = format!("{e:#}"),
                    "kv snapshot rejected; starting cold"
                );
                None
            }
        }
    }

    fn try_restore(&self, model: &mut dyn KvDisk) -> anyhow::Result<Option<(Vec<u32>, usize)>> {
        restore_file(&self.path, &self.meta, model)
    }
}

fn read_header(map: &memmap2::Mmap) -> anyhow::Result<(Header, usize)> {
    anyhow::ensure!(map.len() >= 12, "file shorter than magic + header length");
    anyhow::ensure!(&map[..8] == SNAPSHOT_MAGIC, "bad magic");
    let hlen = u32::from_le_bytes(map[8..12].try_into().unwrap()) as usize;
    anyhow::ensure!(hlen <= HEADER_MAX, "header of {hlen} bytes exceeds HEADER_MAX");
    anyhow::ensure!(map.len() >= 12 + hlen, "file shorter than its declared header");
    let h: Header = serde_json::from_slice(&map[12..12 + hlen])?;
    anyhow::ensure!(
        h.frontier > 0 && h.frontier <= h.max_seq,
        "frontier {} outside 1..={}",
        h.frontier,
        h.max_seq
    );
    anyhow::ensure!(
        !h.tokens.is_empty() && h.tokens.len() <= h.max_seq,
        "token prefix of {} outside the window",
        h.tokens.len()
    );
    anyhow::ensure!(
        h.tokens.len() <= h.frontier,
        "frontier {} is below the {}-token prompt it must already cover: the writer stores \
         tokens = the request's PROMPT and frontier = decoder.current_pos(), which is prompt plus \
         everything generated, so frontier < tokens.len() means the KV does not reach the end of \
         the prefix this snapshot claims. Asserting the opposite -- frontier <= tokens.len() -- \
         rejected every snapshot with a non-zero completion, which is every snapshot this engine \
         has ever written",
        h.frontier,
        h.tokens.len()
    );
    anyhow::ensure!(
        h.token_hash == hash_tokens(&h.tokens, h.frontier),
        "token prefix hash mismatch"
    );
    Ok((h, hlen))
}

pub fn peek_stream(path: &std::path::Path) -> Option<(Vec<u32>, usize)> {
    let f = std::fs::File::open(path).ok()?;
    let map = unsafe { memmap2::Mmap::map(&f).ok()? };
    let (h, _) = read_header(&map).ok()?;
    Some((h.tokens, h.frontier))
}

pub fn restore_file(
    path: &std::path::Path,
    meta: &Meta,
    model: &mut dyn KvDisk,
) -> anyhow::Result<Option<(Vec<u32>, usize)>> {
    {
        let f = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let map = unsafe { memmap2::Mmap::map(&f)? };
        let (h, hlen) = read_header(&map)?;
        anyhow::ensure!(
            h.model_id == meta.model_id,
            "model {} != {}",
            h.model_id,
            meta.model_id
        );
        anyhow::ensure!(h.kind == meta.kind, "kind {} != {}", h.kind, meta.kind);
        anyhow::ensure!(h.adapter == meta.adapter, "lora adapter mismatch");
        anyhow::ensure!(
            h.max_seq == meta.max_seq,
            "max_seq {} != {}",
            h.max_seq,
            meta.max_seq
        );
        anyhow::ensure!(
            h.prefill_chunk == model.prefill_chunk_len(),
            "prefill_chunk {} != {}",
            h.prefill_chunk,
            model.prefill_chunk_len()
        );
        anyhow::ensure!(
            h.sliding_window == model.sliding_window(),
            "sliding_window {} != {}",
            h.sliding_window,
            model.sliding_window()
        );
        let live: Vec<(usize, [usize; 4])> = (0..model.kv_layer_count())
            .filter_map(|li| model.kv_layer_lens(li).map(|l| (li, l)))
            .collect();
        anyhow::ensure!(
            live.len() == h.layers.len()
                && live.iter().zip(h.layers.iter()).all(|((li, l), hl)| {
                    *li == hl.li
                        && (0..4).all(|c| l[c] == hl.chunks[c].0 && hl.chunks[c].1 <= hl.chunks[c].0)
                }),
            "kv layer geometry mismatch"
        );
        let need: u64 = h
            .layers
            .iter()
            .map(|l| l.chunks.iter().map(|(_, s)| *s as u64 * 4).sum::<u64>())
            .sum();
        anyhow::ensure!(
            need == h.payload_len,
            "declared payload {} != geometry {need}",
            h.payload_len
        );
        anyhow::ensure!(
            map.len() as u64 == 12 + hlen as u64 + h.payload_len,
            "file length {} != declared {}",
            map.len(),
            12 + hlen as u64 + h.payload_len
        );
        let payload = &map[12 + hlen..];
        let mut hash = Xxh3::new();
        for part in payload.chunks(4 << 20) {
            hash.update(part);
        }
        anyhow::ensure!(hash.digest() == h.payload_hash, "payload hash mismatch");
        let mut off = 0usize;
        for l in &h.layers {
            let mut read_u32s = |words: usize, stored: usize| -> Vec<u32> {
                let mut v = vec![0u32; words];
                bytemuck::cast_slice_mut::<u32, u8>(&mut v)[..stored * 4]
                    .copy_from_slice(&payload[off..off + stored * 4]);
                off += stored * 4;
                v
            };
            let snap: KvCacheSnapshot = (
                read_u32s(l.chunks[0].0, l.chunks[0].1),
                read_u32s(l.chunks[1].0, l.chunks[1].1),
                bytemuck::cast_vec(read_u32s(l.chunks[2].0, l.chunks[2].1)),
                bytemuck::cast_vec(read_u32s(l.chunks[3].0, l.chunks[3].1)),
            );
            anyhow::ensure!(
                model.kv_cache_restore(l.li, &snap)?,
                "layer {} has no kv buffers",
                l.li
            );
        }
        model.restore_pos(h.frontier)?;
        Ok(Some((h.tokens, h.frontier)))
    }
}

impl Session {
    pub fn maybe_save(&mut self, model: &dyn KvDisk, tokens: &[u32], frontier: usize) {
        if let Some((at, _)) = self.last_save {
            if at.elapsed() < SAVE_DEBOUNCE {
                return;
            }
        }
        self.save(model, tokens, frontier, false);
    }

    pub fn save_now(&mut self, model: &dyn KvDisk, tokens: &[u32], frontier: usize) {
        self.save(model, tokens, frontier, true);
    }

    fn save(&mut self, model: &dyn KvDisk, tokens: &[u32], frontier: usize, sync: bool) {
        if tokens.is_empty() || frontier == 0 {
            return;
        }
        let fp = hash_tokens(tokens, frontier);
        if self.last_save.map(|(_, h)| h) == Some(fp) {
            return;
        }
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
        let t_download = Instant::now();
        let Some(snaps) = collect_snaps(&self.path, model) else {
            return;
        };
        let header = build_header(&self.meta, model, tokens, frontier);
        let download_ms = t_download.elapsed().as_millis() as u64;
        self.last_save = Some((Instant::now(), fp));
        if sync {
            write_snapshot(&self.path, header, snaps, download_ms);
        } else {
            let path = self.path.clone();
            self.writer = Some(std::thread::spawn(move || {
                write_snapshot(&path, header, snaps, download_ms)
            }));
        }
    }
}

fn collect_snaps(
    path_for_log: &std::path::Path,
    model: &dyn KvDisk,
) -> Option<Vec<(usize, KvCacheSnapshot)>> {
    let mut snaps: Vec<(usize, KvCacheSnapshot)> = Vec::new();
    for li in 0..model.kv_layer_count() {
        match model.kv_cache_snapshot(li) {
            Ok(Some(snap)) => snaps.push((li, snap)),
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(
                    path = %path_for_log.display(),
                    error = format!("{e:#}"),
                    "kv snapshot download failed"
                );
                return None;
            }
        }
    }
    if snaps.is_empty() {
        tracing::warn!(
            path = %path_for_log.display(),
            "kv snapshot save skipped: decoder exposes no kv layers"
        );
        return None;
    }
    Some(snaps)
}

fn build_header(meta: &Meta, model: &dyn KvDisk, tokens: &[u32], frontier: usize) -> Header {
    Header {
        model_id: meta.model_id.clone(),
        kind: meta.kind.clone(),
        adapter: meta.adapter.clone(),
        max_seq: meta.max_seq,
        prefill_chunk: model.prefill_chunk_len(),
        sliding_window: model.sliding_window(),
        frontier,
        tokens: tokens.to_vec(),
        token_hash: hash_tokens(tokens, frontier),
        layers: Vec::new(),
        payload_len: 0,
        payload_hash: 0,
    }
}

pub fn save_file(
    path: &std::path::Path,
    meta: &Meta,
    model: &dyn KvDisk,
    tokens: &[u32],
    frontier: usize,
) {
    if tokens.is_empty() || frontier == 0 {
        return;
    }
    let t_download = Instant::now();
    let Some(snaps) = collect_snaps(path, model) else {
        return;
    };
    let header = build_header(meta, model, tokens, frontier);
    write_snapshot(path, header, snaps, t_download.elapsed().as_millis() as u64);
}

fn write_snapshot(
    path: &std::path::Path,
    mut header: Header,
    snaps: Vec<(usize, KvCacheSnapshot)>,
    download_ms: u64,
) {
    let t_write = Instant::now();
    let mut len = 0u64;
    let mut hash = Xxh3::new();
    for (li, s) in &snaps {
        let mut chunks = [(0usize, 0usize); 4];
        for (c, bytes) in snap_chunks(s).into_iter().enumerate() {
            let stored = nonzero_prefix_words(bytes);
            chunks[c] = (bytes.len() / 4, stored);
            len += stored as u64 * 4;
            hash.update(&bytes[..stored * 4]);
        }
        header.layers.push(LayerLens { li: *li, chunks });
    }
    header.payload_len = len;
    header.payload_hash = hash.digest();
    match write_atomic(path, &header, &snaps) {
        Ok(bytes) => tracing::info!(
            path = %path.display(),
            frontier = header.frontier,
            bytes,
            download_ms,
            write_ms = t_write.elapsed().as_millis() as u64,
            "kv snapshot saved"
        ),
        Err(e) => tracing::warn!(
            path = %path.display(),
            error = format!("{e:#}"),
            "kv snapshot save failed; not retried until the prefix changes"
        ),
    }
}

fn write_atomic(
    path: &std::path::Path,
    header: &Header,
    snaps: &[(usize, KvCacheSnapshot)],
) -> anyhow::Result<u64> {
    use std::io::Write as _;
    let hjson = serde_json::to_vec(header)?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let tmp = path.with_extension(format!("nvkv.tmp.{}.{nonce:x}", std::process::id()));
    let res = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .and_then(|f| {
            let mut w = std::io::BufWriter::new(f);
            w.write_all(SNAPSHOT_MAGIC)?;
            w.write_all(&(hjson.len() as u32).to_le_bytes())?;
            w.write_all(&hjson)?;
            let mut n = 12 + hjson.len() as u64;
            for ((_, s), hl) in snaps.iter().zip(header.layers.iter()) {
                for (bytes, (_, stored)) in snap_chunks(s).into_iter().zip(hl.chunks.iter()) {
                    w.write_all(&bytes[..stored * 4])?;
                    n += *stored as u64 * 4;
                }
            }
            w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
            Ok(n)
        });
    match res {
        Ok(n) => {
            if let Err(e) = std::fs::rename(&tmp, path) {
                let _ = std::fs::remove_file(&tmp);
                return Err(e.into());
            }
            Ok(n)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e.into())
        }
    }
}

#[cfg(test)]
mod state_disk_macro_tests {
    use super::*;

    struct FakeStateModel {
        blobs: Vec<Vec<u32>>,
        restored: Vec<usize>,
        pos: Option<usize>,
    }

    impl FakeStateModel {
        fn new() -> Self {
            Self {
                blobs: vec![vec![1, 2, 3], vec![4, 5, 6, 7]],
                restored: Vec::new(),
                pos: None,
            }
        }
        fn state_blob_count(&self) -> usize {
            self.blobs.len()
        }
        fn state_blob_words(&self, i: usize) -> Option<usize> {
            self.blobs.get(i).map(|b| b.len())
        }
        fn state_blob_download(&self, i: usize) -> anyhow::Result<Option<Vec<u32>>> {
            Ok(self.blobs.get(i).cloned())
        }
        fn state_blob_restore(&mut self, i: usize, words: &[u32]) -> anyhow::Result<bool> {
            self.restored.push(i);
            self.blobs[i] = words.to_vec();
            Ok(true)
        }
        fn restore_pos(&mut self, pos: usize) -> anyhow::Result<()> {
            self.pos = Some(pos);
            Ok(())
        }
    }

    impl_state_disk!(FakeStateModel);

    #[test]
    fn a_state_registry_snapshot_carrying_more_than_one_blob_per_layer_is_refused() {
        let mut m = FakeStateModel::new();
        let one_blob: KvCacheSnapshot = (vec![9, 9, 9], Vec::new(), Vec::new(), Vec::new());
        assert!(
            m.kv_cache_restore(0, &one_blob).expect("the shape the macro writes"),
            "the one-blob shape must restore"
        );
        assert_eq!(m.restored, vec![0]);

        for (label, bad) in [
            ("k", (vec![9], vec![1u32], Vec::new(), Vec::new())),
            ("v-scale", (vec![9], Vec::new(), vec![1.0f32], Vec::new())),
            ("k-scale", (vec![9], Vec::new(), Vec::new(), vec![1.0f32])),
        ] {
            let e = m
                .kv_cache_restore(0, &bad)
                .expect_err("a snapshot with a populated per-position chunk must be refused");
            assert!(
                format!("{e:#}").contains("one blob per layer"),
                "{label}: wrong reason: {e:#}"
            );
        }
    }

    #[test]
    fn the_state_disk_family_reports_its_geometry_as_one_chunk_per_layer() {
        let m = FakeStateModel::new();
        assert_eq!(KvDisk::kv_layer_count(&m), 2);
        assert_eq!(m.kv_layer_lens(0), Some([3, 0, 0, 0]));
        assert_eq!(m.kv_layer_lens(1), Some([4, 0, 0, 0]));
        assert_eq!(m.kv_layer_lens(2), None, "past the end must be None");

        let snap = m.kv_cache_snapshot(1).unwrap().expect("layer 1 has a blob");
        assert_eq!(snap.0, vec![4, 5, 6, 7]);
        assert!(
            snap.1.is_empty() && snap.2.is_empty() && snap.3.is_empty(),
            "the writer must emit exactly the shape kv_cache_restore accepts, or the two \
             halves of this family disagree and every restore fails the ensure!"
        );

        assert_eq!(KvDisk::prefill_chunk_len(&m), 0);
        assert_eq!(KvDisk::sliding_window(&m), 0);
    }
}
