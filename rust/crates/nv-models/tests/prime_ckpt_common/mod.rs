#![cfg(feature = "cuda")]
#![allow(dead_code)]

use anyhow::{Context, Result};
use nv_models::qwen3_5_moe::{
    Qwen3MoeKvCache, PRIME_CKPT_CACHE_LAYOUT_VERSION_1_BUMP_WHEN_FP8_ROW_SCALE_OR_LIN_STATE_LAYOUT_CHANGES,
};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const PRIME_CKPT_DIR_ENV_NV_KV_CKPT_DIR: &str = "NV_KV_CKPT_DIR";
pub const PRIME_CKPT_DISK_TIGHT: &str = "PrimeCkptDiskTight";
const DISK_MARGIN_BYTES_2GIB_KEEPS_A_98PCT_HOME_OUT_OF_ENOSPC: u64 = 2 << 30;
const LIN_STATE_AND_HEADER_SLACK_BYTES_256MIB_MEASURED_154MIB_AT_QWEN38_L64_8K: u64 = 256 << 20;
const FP8_KV_BYTES_PER_ELEM_1_PLUS_F32_SCALE_PER_KV_HEAD_ROW: u64 = 1;

pub fn prime_ckpt_dir_env_off_by_default_so_the_ladder_defaults_never_change() -> Option<PathBuf> {
    std::env::var(PRIME_CKPT_DIR_ENV_NV_KV_CKPT_DIR)
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

pub fn expected_ckpt_file_bytes_fp8_kv_rows_scales_plus_lin_state_slack(
    full_attention_slots: usize,
    depth: usize,
    n_kv: usize,
    head_dim: usize,
) -> u64 {
    let kv = (full_attention_slots * 2 * depth * n_kv * head_dim) as u64
        * FP8_KV_BYTES_PER_ELEM_1_PLUS_F32_SCALE_PER_KV_HEAD_ROW;
    let scales = (full_attention_slots * 2 * depth * n_kv * 4) as u64;
    kv + scales + LIN_STATE_AND_HEADER_SLACK_BYTES_256MIB_MEASURED_154MIB_AT_QWEN38_L64_8K
}

fn free_bytes_on(dir: &Path) -> Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(dir.as_os_str().as_bytes())
        .context("ckpt dir path has a NUL byte")?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c.as_ptr(), &mut st) };
    anyhow::ensure!(
        rc == 0,
        "statvfs({}): {}",
        dir.display(),
        std::io::Error::last_os_error()
    );
    Ok(st.f_bavail as u64 * st.f_frsize as u64)
}

pub fn refuse_dump_unless_disk_headroom_covers_ckpt_plus_margin(
    dir: &Path,
    expected_file_bytes: u64,
) -> Result<()> {
    let free = free_bytes_on(dir)?;
    let needed = expected_file_bytes + DISK_MARGIN_BYTES_2GIB_KEEPS_A_98PCT_HOME_OUT_OF_ENOSPC;
    anyhow::ensure!(
        free >= needed,
        "{PRIME_CKPT_DISK_TIGHT}: {} has {:.2} GiB free but the checkpoint needs {:.2} GiB \
         ({:.2} GiB file + 2 GiB margin); free space or point {} at a roomier filesystem",
        dir.display(),
        free as f64 / (1u64 << 30) as f64,
        needed as f64 / (1u64 << 30) as f64,
        expected_file_bytes as f64 / (1u64 << 30) as f64,
        PRIME_CKPT_DIR_ENV_NV_KV_CKPT_DIR
    );
    Ok(())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn fingerprint_of_checkpoint_dims_depth_fillmode_and_cache_layout_version(
    config_json_raw: &str,
    num_layers: usize,
    n_kv: usize,
    head_dim: usize,
    depth: usize,
    fill_mode: &str,
) -> String {
    format!(
        "q38-cfg{:016x}-L{num_layers}-kv{n_kv}x{head_dim}-d{depth}-{fill_mode}-layout{}",
        fnv1a64(config_json_raw.as_bytes()),
        PRIME_CKPT_CACHE_LAYOUT_VERSION_1_BUMP_WHEN_FP8_ROW_SCALE_OR_LIN_STATE_LAYOUT_CHANGES
    )
}

pub struct PrimeCkptFlock {
    _held_until_drop: File,
}

pub fn flock_exclusive_blocking_so_concurrent_lanes_serialize_per_fingerprint(
    dir: &Path,
    fingerprint: &str,
) -> Result<PrimeCkptFlock> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create prime ckpt dir {}", dir.display()))?;
    let path = dir.join(format!("{fingerprint}.lock"));
    let f = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
        .with_context(|| format!("open lock file {}", path.display()))?;
    let rc = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) };
    anyhow::ensure!(
        rc == 0,
        "flock({}) failed: {}",
        path.display(),
        std::io::Error::last_os_error()
    );
    Ok(PrimeCkptFlock { _held_until_drop: f })
}

pub fn ckpt_file_path(dir: &Path, fingerprint: &str) -> PathBuf {
    dir.join(format!("{fingerprint}.bin"))
}

pub fn dump_cache_to_ckpt_file_tmp_then_rename_so_a_kill_never_leaves_a_torn_file(
    cache: &Qwen3MoeKvCache,
    dir: &Path,
    fingerprint: &str,
    expected_file_bytes: u64,
) -> Result<PathBuf> {
    refuse_dump_unless_disk_headroom_covers_ckpt_plus_margin(dir, expected_file_bytes)?;
    let final_path = ckpt_file_path(dir, fingerprint);
    let tmp_path = dir.join(format!("{fingerprint}.tmp.{}", std::process::id()));
    {
        let f = File::create(&tmp_path)
            .with_context(|| format!("create {}", tmp_path.display()))?;
        let mut w = BufWriter::new(f);
        cache.dump_primed_state_for_reuse(fingerprint, &mut w)?;
        use std::io::Write;
        w.flush()?;
    }
    std::fs::rename(&tmp_path, &final_path)
        .with_context(|| format!("rename into {}", final_path.display()))?;
    Ok(final_path)
}

pub fn restore_cache_from_ckpt_file_checked(
    cache: &mut Qwen3MoeKvCache,
    path: &Path,
    fingerprint: &str,
) -> Result<()> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut r = BufReader::new(f);
    cache.restore_primed_state_checked(fingerprint, &mut r)
}

pub enum PrimeSource {
    RestoredFromCkpt,
    PrimedFreshCkptWritten,
    PrimedFreshCkptDisabled,
    PrimedFreshCkptSkippedDiskTight,
}

impl PrimeSource {
    pub fn label(&self) -> &'static str {
        match self {
            PrimeSource::RestoredFromCkpt => "restored",
            PrimeSource::PrimedFreshCkptWritten => "primed_ckpt_written",
            PrimeSource::PrimedFreshCkptDisabled => "primed_no_ckpt",
            PrimeSource::PrimedFreshCkptSkippedDiskTight => "primed_disk_tight",
        }
    }
}

pub fn restore_or_prime_then_dump_holding_the_flock(
    cache: &mut Qwen3MoeKvCache,
    dir: Option<&Path>,
    fingerprint: &str,
    expected_file_bytes: u64,
    prime: &mut dyn FnMut(&mut Qwen3MoeKvCache) -> Result<()>,
) -> Result<(PrimeSource, f64)> {
    let Some(dir) = dir else {
        let t0 = Instant::now();
        prime(cache)?;
        return Ok((
            PrimeSource::PrimedFreshCkptDisabled,
            t0.elapsed().as_secs_f64(),
        ));
    };
    let _lock = flock_exclusive_blocking_so_concurrent_lanes_serialize_per_fingerprint(
        dir,
        fingerprint,
    )?;
    let path = ckpt_file_path(dir, fingerprint);
    if path.exists() {
        let t0 = Instant::now();
        match restore_cache_from_ckpt_file_checked(cache, &path, fingerprint) {
            Ok(()) => {
                return Ok((PrimeSource::RestoredFromCkpt, t0.elapsed().as_secs_f64()))
            }
            Err(e) => {
                eprintln!(
                    "PRIME-CKPT refuse-and-reprime: restore of {} failed ({e:#}); priming fresh and rewriting",
                    path.display()
                );
                cache.reset();
            }
        }
    }
    let t0 = Instant::now();
    prime(cache)?;
    let prime_s = t0.elapsed().as_secs_f64();
    match dump_cache_to_ckpt_file_tmp_then_rename_so_a_kill_never_leaves_a_torn_file(
        cache,
        dir,
        fingerprint,
        expected_file_bytes,
    ) {
        Ok(_) => Ok((PrimeSource::PrimedFreshCkptWritten, prime_s)),
        Err(e) if format!("{e:#}").contains(PRIME_CKPT_DISK_TIGHT) => {
            eprintln!("PRIME-CKPT dump skipped, prime kept: {e:#}");
            Ok((PrimeSource::PrimedFreshCkptSkippedDiskTight, prime_s))
        }
        Err(e) => Err(e),
    }
}
