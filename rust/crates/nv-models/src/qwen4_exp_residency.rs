
use super::*;
use crate::session_lru::{Budget, Entry, Held, SessionLru, Tier, METRICS};
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

pub const NV_Q4EXP_SESSION_KV_MIB_DEFAULT_0_DEVICE_MIB_FOR_PARKED_LONG_SESSIONS_BEYOND_THE_WORKING_ONE: &str =
    "NV_Q4EXP_SESSION_KV_MIB";
pub const NV_Q4EXP_SESSION_KV_MIN_FREE_MIB_DEFAULT_4096_NEVER_PARK_ON_DEVICE_BELOW_THIS_FREE_VRAM: &str =
    "NV_Q4EXP_SESSION_KV_MIN_FREE_MIB";
pub const NV_Q4EXP_SESSION_SPILL_GIB_DEFAULT_32_DISK_FOR_SPILLED_SESSIONS_UNDER_NV_KV_CACHE_DIR: &str =
    "NV_Q4EXP_SESSION_SPILL_GIB";
pub const NV_Q4EXP_SESSION_SPILL_MAX_AGE_SECS_DEFAULT_28800_A_SPILLED_SESSION_IDLE_LONGER_IS_DELETED: &str =
    "NV_Q4EXP_SESSION_SPILL_MAX_AGE_SECS";
pub const KV_CACHE_DIR_ENV: &str = "NV_KV_CACHE_DIR";
pub const SPILL_SUBDIR: &str = "q4exp-sessions";
const SPILL_PREFIX: &str = "q4kv-";
const SPILL_FORMAT_VERSION: u32 = 1;
const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

#[derive(Clone, Debug, PartialEq)]
pub struct SessionResidency {
    pub device_budget_bytes: u64,
    pub min_free_bytes: u64,
    pub spill_dir: Option<PathBuf>,
    pub disk_budget_bytes: u64,
    pub spill_max_age: Duration,
}

impl Default for SessionResidency {
    fn default() -> Self {
        Self {
            device_budget_bytes: 0,
            min_free_bytes: 4096 * MIB,
            spill_dir: None,
            disk_budget_bytes: 32 * GIB,
            spill_max_age: Duration::from_secs(8 * 3600),
        }
    }
}

impl SessionResidency {
    pub fn from_env() -> Self {
        Self::from_vars(|k| std::env::var(k).ok())
    }

    pub fn from_vars(get: impl Fn(&str) -> Option<String>) -> Self {
        let num = |k: &str| get(k).and_then(|v| v.trim().parse::<u64>().ok());
        let d = Self::default();
        Self {
            device_budget_bytes: num(NV_Q4EXP_SESSION_KV_MIB_DEFAULT_0_DEVICE_MIB_FOR_PARKED_LONG_SESSIONS_BEYOND_THE_WORKING_ONE)
                .map_or(d.device_budget_bytes, |m| m * MIB),
            min_free_bytes: num(NV_Q4EXP_SESSION_KV_MIN_FREE_MIB_DEFAULT_4096_NEVER_PARK_ON_DEVICE_BELOW_THIS_FREE_VRAM)
                .map_or(d.min_free_bytes, |m| m * MIB),
            spill_dir: get(KV_CACHE_DIR_ENV).filter(|v| !v.trim().is_empty()).map(|v| PathBuf::from(v).join(SPILL_SUBDIR)),
            disk_budget_bytes: num(NV_Q4EXP_SESSION_SPILL_GIB_DEFAULT_32_DISK_FOR_SPILLED_SESSIONS_UNDER_NV_KV_CACHE_DIR)
                .map_or(d.disk_budget_bytes, |g| g * GIB),
            spill_max_age: num(NV_Q4EXP_SESSION_SPILL_MAX_AGE_SECS_DEFAULT_28800_A_SPILLED_SESSION_IDLE_LONGER_IS_DELETED)
                .map_or(d.spill_max_age, Duration::from_secs),
        }
    }
}

pub struct ParkedState {
    kv: Vec<Tensor>,
    gdn: Vec<(Tensor, Tensor)>,
    ple_conv: Tensor,
    bytes: u64,
}

pub enum Parked {
    InPlace,
    Device { state: ParkedState, spec_k: usize },
    Disk { path: PathBuf, bytes: u64, spec_k: usize, written: SystemTime },
}

impl Parked {
    fn spec_k(&self) -> Option<usize> {
        match self {
            Parked::InPlace => None,
            Parked::Device { spec_k, .. } | Parked::Disk { spec_k, .. } => Some(*spec_k),
        }
    }
}

impl Held for Parked {
    fn tier(&self) -> Tier {
        match self {
            Parked::InPlace => Tier::InPlace,
            Parked::Device { .. } => Tier::Device,
            Parked::Disk { .. } => Tier::Disk,
        }
    }

    fn bytes(&self) -> u64 {
        match self {
            Parked::InPlace => 0,
            Parked::Device { state, .. } => state.bytes,
            Parked::Disk { bytes, .. } => *bytes,
        }
    }
}

pub struct Residency {
    pub(super) lru: SessionLru<Parked>,
    pub(super) cfg: SessionResidency,
    pub(super) tag: u64,
}

impl Residency {
    pub(super) fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let unique = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ^ ((std::process::id() as u64) << 32)
            ^ SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
        let cfg = SessionResidency::default();
        Self {
            lru: SessionLru::new(Budget { device_bytes: cfg.device_budget_bytes, disk_bytes: 0 }),
            cfg,
            tag: unique,
        }
    }

    fn disk_budget(&self) -> u64 {
        if self.cfg.spill_dir.is_some() { self.cfg.disk_budget_bytes } else { 0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResidencyStats {
    pub in_place: usize,
    pub device: usize,
    pub disk: usize,
    pub device_bytes: u64,
    pub disk_bytes: u64,
}

pub fn weights_tag_for_dir(dir: &Path) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf()).hash(&mut h);
    std::fs::read(dir.join("config.json")).unwrap_or_default().hash(&mut h);
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "safetensors")).collect())
        .unwrap_or_default();
    files.sort();
    for f in files {
        f.file_name().hash(&mut h);
        if let Ok(m) = std::fs::metadata(&f) {
            m.len().hash(&mut h);
            m.modified().ok().and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok()).map(|d| d.as_nanos()).hash(&mut h);
        }
        if let Ok(mut file) = std::fs::File::open(&f) {
            use std::io::Read;
            let mut head = vec![0u8; 1 << 16];
            let n = file.read(&mut head).unwrap_or(0);
            head[..n].hash(&mut h);
        }
    }
    h.finish()
}

fn deep_copy_rows(t: &Tensor, rows: usize) -> Result<Tensor> {
    let mut dims = t.dims().to_vec();
    dims[0] = rows;
    let out = Tensor::zeros(dims, t.dtype(), t.device())?;
    if rows > 0 {
        out.slice_set(&t.narrow(0, 0, rows)?, 0, 0)?;
    }
    Ok(out)
}

fn tensor_bytes(t: &Tensor) -> u64 {
    (t.elem_count() * t.dtype().size_in_bytes()) as u64
}

fn free_vram() -> Option<u64> {
    cudarc::driver::result::mem_get_info().ok().map(|(free, _)| free as u64)
}

impl Qwen4Exp {

    pub fn configure_session_residency(&self, cfg: SessionResidency) -> Result<()> {
        if let Some(dir) = &cfg.spill_dir {
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
            }
        }
        let mut res = self.residency.lock().map_err(|e| anyhow::anyhow!("residency poisoned: {e}"))?;
        if let Some(dir) = &cfg.spill_dir {
            let indexed: std::collections::HashSet<PathBuf> = res
                .lru
                .entries()
                .iter()
                .filter_map(|e| match &e.value {
                    Parked::Disk { path, .. } => Some(path.clone()),
                    _ => None,
                })
                .collect();
            for e in std::fs::read_dir(dir)?.filter_map(|e| e.ok()) {
                if e.file_name().to_string_lossy().starts_with(SPILL_PREFIX) && !indexed.contains(&e.path()) {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
        res.cfg = cfg;
        let budget = Budget { device_bytes: res.cfg.device_budget_bytes, disk_bytes: res.disk_budget() };
        res.lru.set_budget(budget);
        self.enforce_residency(&mut res);
        eprintln!(
            "[qwen4_exp] long-session residency: working session + {} MiB parked on device (min free {} MiB), spill {}",
            res.cfg.device_budget_bytes / MIB,
            res.cfg.min_free_bytes / MIB,
            res.cfg.spill_dir.as_ref().map_or("off".to_string(), |d| format!("{} up to {} GiB, {} h", d.display(), res.cfg.disk_budget_bytes / GIB, res.cfg.spill_max_age.as_secs() / 3600))
        );
        Ok(())
    }

    pub fn set_weights_tag(&self, tag: u64) {
        if let Ok(mut res) = self.residency.lock() {
            res.tag = tag;
        }
    }

    pub fn residency_stats(&self) -> ResidencyStats {
        let Ok(res) = self.residency.lock() else { return ResidencyStats::default() };
        ResidencyStats {
            in_place: res.lru.count(Tier::InPlace),
            device: res.lru.count(Tier::Device),
            disk: res.lru.count(Tier::Disk),
            device_bytes: res.lru.bytes(Tier::Device),
            disk_bytes: res.lru.bytes(Tier::Disk),
        }
    }

    fn attn_slots<'a>(st: &'a Qwen4ExpState) -> impl Iterator<Item = &'a AttnSlot> {
        st.attn.iter().chain(st.mtp_attn.as_ref())
    }

    fn parked_bytes(&self, st: &Qwen4ExpState, pos: usize) -> u64 {
        let r = self.cfg.indexer_compress_ratio;
        let kv: u64 = Self::attn_slots(st)
            .map(|s| {
                let row = |t: &Tensor| tensor_bytes(t) / t.dim(0).unwrap_or(1).max(1) as u64;
                (row(&s.k) + row(&s.v) + row(&s.raw_idx_k)) * pos as u64 + row(&s.cmp_k) * (pos / r) as u64
            })
            .sum();
        let gdn: u64 = st.gdn.iter().map(|g| tensor_bytes(g.conv_state()) + tensor_bytes(g.recurrent_state())).sum();
        kv + gdn + tensor_bytes(&st.ple_conv)
    }

    fn snapshot(&self, s: &Qwen4ExpSession, pos: usize, host: bool) -> Result<ParkedState> {
        let ck = s.ckpts.iter().find(|c| c.pos == pos).ok_or_else(|| anyhow::anyhow!("no checkpoint at {pos}"))?;
        let r = self.cfg.indexer_compress_ratio;
        ensure!(pos % r == 0, "checkpoint {pos} is not a multiple of the indexer compress ratio {r}");
        let place = |t: Tensor| -> Result<Tensor> { Ok(if host { t.to_device(&Device::Cpu)? } else { t }) };
        let mut kv = Vec::new();
        for slot in Self::attn_slots(&s.state) {
            for (t, n) in [(&slot.k, pos), (&slot.v, pos), (&slot.raw_idx_k, pos), (&slot.cmp_k, pos / r)] {
                kv.push(place(deep_copy_rows(t, n)?)?);
            }
        }
        let gdn = ck
            .gdn
            .iter()
            .map(|g| Ok((place(g.conv_state().copy()?)?, place(g.recurrent_state().copy()?)?)))
            .collect::<Result<Vec<_>>>()?;
        let ple_conv = place(ck.ple_conv.copy()?)?;
        stream_of(&self.device)?.synchronize()?;
        let bytes = kv.iter().map(tensor_bytes).sum::<u64>()
            + gdn.iter().map(|(c, r)| tensor_bytes(c) + tensor_bytes(r)).sum::<u64>()
            + tensor_bytes(&ple_conv);
        Ok(ParkedState { kv, gdn, ple_conv, bytes })
    }

    fn spill_path(&self, res: &Residency, tokens: &[u32], pos: usize) -> Option<PathBuf> {
        use std::hash::{Hash, Hasher};
        let dir = res.cfg.spill_dir.as_ref()?;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        tokens[..pos.min(tokens.len())].hash(&mut h);
        Some(dir.join(format!("{SPILL_PREFIX}{:016x}-{:016x}-{pos}.safetensors", res.tag, h.finish())))
    }

    fn write_spill(&self, path: &Path, tag: u64, tokens: &[u32], pos: usize, spec_k: usize, st: &ParkedState) -> Result<u64> {
        let mut map: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        for (i, t) in st.kv.iter().enumerate() {
            map.insert(format!("kv.{i}"), t.to_device(&Device::Cpu)?);
        }
        for (i, (c, r)) in st.gdn.iter().enumerate() {
            map.insert(format!("gdn.{i}.conv"), c.to_device(&Device::Cpu)?);
            map.insert(format!("gdn.{i}.rec"), r.to_device(&Device::Cpu)?);
        }
        map.insert("ple_conv".into(), st.ple_conv.to_device(&Device::Cpu)?);
        map.insert("tokens".into(), Tensor::from_vec(tokens.to_vec(), tokens.len(), &Device::Cpu)?);
        let meta = vec![SPILL_FORMAT_VERSION, pos as u32, spec_k as u32, (tag >> 32) as u32, tag as u32, st.kv.len() as u32, st.gdn.len() as u32];
        map.insert("meta".into(), Tensor::from_vec(meta, 7, &Device::Cpu)?);
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        candle_core::safetensors::save(&map, &tmp)?;
        std::fs::rename(&tmp, path)?;
        Ok(std::fs::metadata(path)?.len())
    }

    fn read_spill(&self, e: &Entry<Parked>, tag: u64, live: &Qwen4ExpState) -> Result<ParkedState> {
        let Parked::Disk { path, spec_k, .. } = &e.value else { bail!("not a spilled entry") };
        let mut m = candle_core::safetensors::load(path, &Device::Cpu)?;
        let mut take = |k: &str| m.remove(k).ok_or_else(|| anyhow::anyhow!("spill {} lacks {k}", path.display()));
        let meta = take("meta")?.to_vec1::<u32>()?;
        let n_kv = 4 * Self::attn_slots(live).count();
        let n_gdn = live.gdn.len();
        let want = vec![SPILL_FORMAT_VERSION, e.pos as u32, *spec_k as u32, (tag >> 32) as u32, tag as u32, n_kv as u32, n_gdn as u32];
        ensure!(meta == want, "spill header {meta:?} does not match {want:?}");
        ensure!(take("tokens")?.to_vec1::<u32>()? == e.tokens, "spill token stream differs from its index entry");
        let r = self.cfg.indexer_compress_ratio;
        let mut kv = Vec::with_capacity(n_kv);
        for (j, slot) in Self::attn_slots(live).enumerate() {
            for (k, (t, n)) in [(&slot.k, e.pos), (&slot.v, e.pos), (&slot.raw_idx_k, e.pos), (&slot.cmp_k, e.pos / r)].into_iter().enumerate() {
                let h = take(&format!("kv.{}", 4 * j + k))?;
                let mut dims = t.dims().to_vec();
                dims[0] = n;
                ensure!(h.dims() == dims.as_slice() && h.dtype() == t.dtype(), "spilled kv.{} has {:?}, expected {dims:?}", 4 * j + k, h.dims());
                kv.push(h.to_device(&self.device)?);
            }
        }
        let mut gdn = Vec::with_capacity(n_gdn);
        for (i, g) in live.gdn.iter().enumerate() {
            let c = take(&format!("gdn.{i}.conv"))?;
            let rr = take(&format!("gdn.{i}.rec"))?;
            ensure!(c.dims() == g.conv_state().dims() && rr.dims() == g.recurrent_state().dims(), "spilled gdn.{i} geometry differs");
            gdn.push((c.to_device(&self.device)?, rr.to_device(&self.device)?));
        }
        let p = take("ple_conv")?;
        ensure!(p.dims() == live.ple_conv.dims(), "spilled ple_conv geometry differs");
        let ple_conv = p.to_device(&self.device)?;
        let bytes = kv.iter().map(tensor_bytes).sum::<u64>() + gdn.iter().map(|(c, r)| tensor_bytes(c) + tensor_bytes(r)).sum::<u64>() + tensor_bytes(&ple_conv);
        Ok(ParkedState { kv, gdn, ple_conv, bytes })
    }

    fn discard(&self, e: Entry<Parked>) {
        if let Parked::Disk { path, .. } = &e.value {
            let _ = std::fs::remove_file(path);
        }
    }

    fn spill_entry(&self, res: &Residency, e: Entry<Parked>, st: &ParkedState, spec_k: usize) -> Option<Entry<Parked>> {
        let path = self.spill_path(res, &e.tokens, e.pos)?;
        if !res.lru.fits_disk(st.bytes) {
            return None;
        }
        let t0 = std::time::Instant::now();
        match self.write_spill(&path, res.tag, &e.tokens, e.pos, spec_k, st) {
            Ok(bytes) => {
                METRICS.spilled.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                eprintln!("[qwen4_exp] spilled a {}-token session prefix ({:.1} MiB) in {:.2}s", e.pos, bytes as f64 / MIB as f64, t0.elapsed().as_secs_f64());
                Some(Entry { value: Parked::Disk { path, bytes, spec_k, written: SystemTime::now() }, ..e })
            }
            Err(err) => {
                let _ = std::fs::remove_file(&path);
                eprintln!("[qwen4_exp] session spill failed, dropping it: {err:#}");
                None
            }
        }
    }

    pub(super) fn enforce_residency(&self, res: &mut Residency) {
        let max_age = res.cfg.spill_max_age;
        let now = SystemTime::now();
        let mut i = 0;
        while i < res.lru.len() {
            let stale = matches!(&res.lru.get(i).value, Parked::Disk { written, .. } if now.duration_since(*written).unwrap_or_default() > max_age);
            if stale {
                let e = res.lru.remove(i);
                self.discard(e);
            } else {
                i += 1;
            }
        }
        let ev = res.lru.enforce();
        for e in ev.from_device {
            let Entry { model, tokens, pos, used, value } = e;
            let Parked::Device { state, spec_k } = value else { continue };
            let shell = Entry { model, tokens, pos, used, value: Parked::InPlace };
            match self.spill_entry(res, shell, &state, spec_k) {
                Some(spilled) => {
                    for gone in res.lru.insert(spilled) {
                        self.discard(gone);
                    }
                }
                None => {
                    METRICS.dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        for e in res.lru.enforce().from_disk.into_iter().chain(ev.from_disk) {
            self.discard(e);
        }
        METRICS.set_resident(&res.lru);
    }

    pub(super) fn park_working(&self, res: &mut Residency, s: &Qwen4ExpSession) {
        let Some(i) = res.lru.find_in_place(res.tag) else { return };
        let e = res.lru.remove(i);
        if e.tokens != s.prefix || !s.ckpts.iter().any(|c| c.pos == e.pos) {
            return;
        }
        let bytes = self.parked_bytes(&s.state, e.pos);
        let spec_k = s.spec_k();
        let device_ok = res.lru.fits_device(bytes) && free_vram().is_some_and(|f| f > bytes + res.cfg.min_free_bytes);
        if device_ok {
            match self.snapshot(s, e.pos, false) {
                Ok(state) => {
                    METRICS.parked_device.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    for gone in res.lru.insert(Entry { value: Parked::Device { state, spec_k }, ..e }) {
                        self.discard(gone);
                    }
                    return;
                }
                Err(err) => eprintln!("[qwen4_exp] parking a session on device failed: {err:#}"),
            }
        }
        if res.cfg.spill_dir.is_some() && res.lru.fits_disk(bytes) {
            match self.snapshot(s, e.pos, true) {
                Ok(st) => {
                    if let Some(spilled) = self.spill_entry(res, e, &st, spec_k) {
                        for gone in res.lru.insert(spilled) {
                            self.discard(gone);
                        }
                        return;
                    }
                }
                Err(err) => eprintln!("[qwen4_exp] snapshot for spill failed: {err:#}"),
            }
        }
        METRICS.dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub(super) fn install_parked(&self, res: &Residency, i: usize, sess: &mut Qwen4ExpSession) -> Result<()> {
        let e = res.lru.get(i);
        ensure!(e.pos < sess.state.max_len, "parked checkpoint {} past the {}-token session", e.pos, sess.state.max_len);
        let loaded;
        let st = match &e.value {
            Parked::Device { state, .. } => state,
            Parked::Disk { .. } => {
                loaded = self.read_spill(e, res.tag, &sess.state)?;
                &loaded
            }
            Parked::InPlace => bail!("the working session is not a parked state"),
        };
        sess.prefix.clear();
        sess.ckpts.clear();
        let slots: Vec<&AttnSlot> = Self::attn_slots(&sess.state).collect();
        ensure!(st.kv.len() == 4 * slots.len(), "parked state has {} kv tensors for {} attention slots", st.kv.len(), slots.len());
        for (slot, rows) in slots.into_iter().zip(st.kv.chunks(4)) {
            for (t, h) in [&slot.k, &slot.v, &slot.raw_idx_k, &slot.cmp_k].into_iter().zip(rows) {
                if h.dim(0)? > 0 {
                    t.slice_set(h, 0, 0)?;
                }
            }
        }
        ensure!(st.gdn.len() == sess.state.gdn.len(), "parked state has {} GDN layers, the session {}", st.gdn.len(), sess.state.gdn.len());
        let gdn = sess
            .state
            .gdn
            .iter()
            .zip(&st.gdn)
            .map(|(live, (c, r))| {
                let g = live.deep_clone()?;
                g.conv_state().slice_set(c, 0, 0)?;
                g.recurrent_state().slice_set(r, 0, 0)?;
                Ok(g)
            })
            .collect::<Result<Vec<_>>>()?;
        let ple_conv = st.ple_conv.copy()?;
        stream_of(&self.device)?.synchronize()?;
        sess.ckpts.push(PrefixCkpt { pos: e.pos, gdn, ple_conv });
        sess.prefix = e.tokens.clone();
        Ok(())
    }

    pub(super) fn best_parked(&self, res: &Residency, prompt: &[u32], live_best: usize, want_k: usize) -> Option<usize> {
        let mut cands: Vec<(usize, usize)> = res
            .lru
            .entries()
            .iter()
            .enumerate()
            .filter(|(_, e)| e.model == res.tag && e.value.spec_k() == Some(want_k))
            .map(|(i, e)| (i, e.resume_point(prompt)))
            .filter(|&(_, r)| r > live_best)
            .collect();
        cands.sort_by_key(|&(i, r)| (std::cmp::Reverse(r), res.lru.get(i).value.tier()));
        cands.first().map(|&(i, _)| i)
    }

    pub(super) fn register_working(&self, res: &mut Residency, s: &Qwen4ExpSession) {
        if let Some(i) = res.lru.find_in_place(res.tag) {
            res.lru.remove(i);
        }
        if let Some(ck) = s.ckpts.last().filter(|c| !s.prefix.is_empty() && c.pos < s.prefix.len()) {
            let used = res.lru.tick();
            let entry = Entry { model: res.tag, tokens: s.prefix.clone(), pos: ck.pos, used, value: Parked::InPlace };
            for gone in res.lru.insert(entry) {
                self.discard(gone);
            }
        }
        self.enforce_residency(res);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_env_knobs_parse_and_default_to_todays_single_working_session() {
        let d = SessionResidency::from_vars(|_| None);
        assert_eq!(d, SessionResidency::default());
        assert_eq!((d.device_budget_bytes, d.spill_dir), (0, None));
        let v = SessionResidency::from_vars(|k| match k {
            "NV_Q4EXP_SESSION_KV_MIB" => Some("2048".into()),
            "NV_Q4EXP_SESSION_KV_MIN_FREE_MIB" => Some("3000".into()),
            "NV_KV_CACHE_DIR" => Some("/var/lib/speaches/kv-cache".into()),
            "NV_Q4EXP_SESSION_SPILL_GIB" => Some("64".into()),
            "NV_Q4EXP_SESSION_SPILL_MAX_AGE_SECS" => Some("3600".into()),
            _ => None,
        });
        assert_eq!(v.device_budget_bytes, 2048 * MIB);
        assert_eq!(v.min_free_bytes, 3000 * MIB);
        assert_eq!(v.spill_dir.as_deref(), Some(Path::new("/var/lib/speaches/kv-cache/q4exp-sessions")));
        assert_eq!(v.disk_budget_bytes, 64 * GIB);
        assert_eq!(v.spill_max_age, Duration::from_secs(3600));
    }
}
