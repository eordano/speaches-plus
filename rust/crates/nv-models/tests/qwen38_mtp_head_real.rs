#![cfg(feature = "cuda")]

use candle_core::{DType, Device};
use nv_models::qwen3_5_mtp::{MtpHead, QWEN38_27B_MTP_DIMS};

fn snapshot_mtp_path() -> Option<std::path::PathBuf> {
    if let Ok(d) = std::env::var("NV_QWEN38_DIR") {
        let p = std::path::PathBuf::from(d).join("model_mtp.safetensors");
        return p.is_file().then_some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots");
    let mut dirs: Vec<_> = std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs.into_iter()
        .map(|d| d.join("model_mtp.safetensors"))
        .find(|p| p.is_file())
}

#[test]
#[ignore]
fn qwen38_dense_mtp_head_loads_every_tensor_of_the_real_shard() {
    if std::env::var("NV_QWEN38_MTP_TEST").as_deref() != Ok("1") {
        eprintln!(
            "SKIP qwen38_dense_mtp_head_loads_every_tensor_of_the_real_shard: set \
             NV_QWEN38_MTP_TEST=1 (needs the unsloth/Qwen3.8-27B-NVFP4 snapshot's \
             model_mtp.safetensors, ~850MB, loads on CPU)"
        );
        return;
    }
    let path = snapshot_mtp_path().expect("model_mtp.safetensors present in the snapshot");
    let map = candle_core::safetensors::load(&path, &Device::Cpu).expect("load mtp shard");
    assert_eq!(
        map.len(),
        15,
        "the 27B MTP shard carries exactly 15 tensors; a different count means the \
         checkpoint layout changed and QWEN38_27B_MTP_DIMS needs re-verification"
    );
    let head = MtpHead::from_map_dense(&map, &QWEN38_27B_MTP_DIMS, DType::BF16)
        .expect("every tensor must resolve against the dense dims contract");
    let _ = head;
    eprintln!("[qwen38-mtp] dense head constructed from {} (15/15 tensors)", path.display());
}
