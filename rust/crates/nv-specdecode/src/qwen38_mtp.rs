use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const NV_DRAFTER_ENV: &str = "NV_DRAFTER";
pub const NV_MTP_DRAFT_DIR_ENV: &str = "NV_MTP_DRAFT_DIR";
pub const NV_MTP_K_ENV: &str = "NV_MTP_K";
pub const NV_Q38_DRAFT_FP8_ENV: &str = "NV_Q38_DRAFT_FP8";
pub const NV_Q38_MTP_REANCHOR_ENV: &str = "NV_Q38_MTP_REANCHOR";
pub const NV_Q38_DRAFT_FAST_ENV: &str = "NV_Q38_DRAFT_FAST";

pub fn draft_fast_device_chained_tokens_selected(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

pub fn draft_fast_device_chained_tokens_selected_from_env() -> bool {
    draft_fast_device_chained_tokens_selected(
        std::env::var(NV_Q38_DRAFT_FAST_ENV).ok().as_deref(),
    )
}

pub const TRUNK_OUTPUT_NORM_TENSOR_CANDIDATES: [&str; 2] =
    ["model.norm.weight", "model.language_model.norm.weight"];

pub fn mtp_reanchor_post_norm_selected(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

pub fn mtp_reanchor_post_norm_selected_from_env() -> bool {
    mtp_reanchor_post_norm_selected(std::env::var(NV_Q38_MTP_REANCHOR_ENV).ok().as_deref())
}

pub fn mtp_drafter_fp8_resident(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

pub fn mtp_drafter_fp8_resident_from_env() -> bool {
    mtp_drafter_fp8_resident(std::env::var(NV_Q38_DRAFT_FP8_ENV).ok().as_deref())
}

pub const NV_Q38_DRAFT_LMHEAD_NVFP4_ENV: &str = "NV_Q38_DRAFT_LMHEAD_NVFP4";

pub fn mtp_draft_lm_head_nvfp4_twin(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

pub fn mtp_draft_lm_head_nvfp4_twin_from_env() -> bool {
    mtp_draft_lm_head_nvfp4_twin(std::env::var(NV_Q38_DRAFT_LMHEAD_NVFP4_ENV).ok().as_deref())
}

pub const MTP_WEIGHTS_FILE_NAME: &str = "model_mtp.safetensors";
pub const MTP_TENSOR_PREFIX: &str = "mtp.";

pub const MTP_CHAIN_DEPTH_DEFAULT: usize = 3;
pub const MTP_CHAIN_DEPTH_MAX_SO_VERIFY_ROWS_STAY_UNDER_THE_F32_FAST_PATH_HARD_CAP: usize = 7;

pub const VERIFY_LM_HEAD_F32_FAST_PATH_HARD_ROW_CAP: usize = 8;
pub const VERIFY_LM_HEAD_BF16_PAIR_FALLBACK_HARD_ROW_CAP: usize = 16;

pub fn verify_lm_head_f32_fast_path_smem_bytes_per_row(hidden: usize) -> usize {
    (hidden >> 4) * 17 * core::mem::size_of::<f32>()
}

pub fn verify_lm_head_bf16_pair_fallback_smem_bytes_per_row(hidden: usize) -> usize {
    (hidden >> 4) * 9 * core::mem::size_of::<u32>()
}

pub fn verify_lm_head_f32_fast_path_rows_ceiling(
    hidden: usize,
    smem_optin_limit_bytes: usize,
) -> Result<usize> {
    ensure!(
        hidden > 0 && hidden % 16 == 0,
        "gemv_i8_normed_mk requires hidden % 16 == 0, got hidden={hidden}"
    );
    if smem_optin_limit_bytes == 0 {
        return Ok(VERIFY_LM_HEAD_F32_FAST_PATH_HARD_ROW_CAP);
    }
    let per_row = verify_lm_head_f32_fast_path_smem_bytes_per_row(hidden);
    Ok((smem_optin_limit_bytes / per_row).min(VERIFY_LM_HEAD_F32_FAST_PATH_HARD_ROW_CAP))
}

pub fn verify_lm_head_rows_per_call_ceiling(
    hidden: usize,
    smem_optin_limit_bytes: usize,
) -> Result<usize> {
    let mf = verify_lm_head_f32_fast_path_rows_ceiling(hidden, smem_optin_limit_bytes)?;
    if smem_optin_limit_bytes == 0 {
        return Ok(mf);
    }
    let h_row = verify_lm_head_bf16_pair_fallback_smem_bytes_per_row(hidden);
    let mh = (smem_optin_limit_bytes / h_row).min(VERIFY_LM_HEAD_BF16_PAIR_FALLBACK_HARD_ROW_CAP);
    Ok(mf.max(mh))
}

pub fn mtp_verify_rows_per_round(k: usize) -> usize {
    k + 1
}

pub fn assert_mtp_chain_depth_fits_verify_lm_head_rows_ceiling(
    hidden: usize,
    k: usize,
    smem_optin_limit_bytes: usize,
) -> Result<()> {
    ensure!(k >= 1, "mtp chain depth must be >= 1, got {k}");
    let rows = mtp_verify_rows_per_round(k);
    ensure!(
        rows <= VERIFY_LM_HEAD_F32_FAST_PATH_HARD_ROW_CAP,
        "mtp round verifies rows=k+1={rows} but the gemv_i8_normed_mk f32 fast path holds a hard \
         M <= {VERIFY_LM_HEAD_F32_FAST_PATH_HARD_ROW_CAP} (the #107 lesson, recomputed for \
         hidden={hidden})"
    );
    let ceiling = verify_lm_head_rows_per_call_ceiling(hidden, smem_optin_limit_bytes)?;
    ensure!(
        rows <= ceiling,
        "mtp round verifies rows=k+1={rows} > lm_head rows-per-call ceiling {ceiling} \
         (basis: hidden={hidden}, smem_optin_limit_bytes={smem_optin_limit_bytes}, \
         f32_row={} B, bf16_pair_row={} B)",
        verify_lm_head_f32_fast_path_smem_bytes_per_row(hidden),
        verify_lm_head_bf16_pair_fallback_smem_bytes_per_row(hidden)
    );
    Ok(())
}

pub fn mtp_drafter_selected(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("mtp"))
}

pub fn mtp_drafter_selected_from_env() -> bool {
    mtp_drafter_selected(std::env::var(NV_DRAFTER_ENV).ok().as_deref())
}

pub fn mtp_chain_depth(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&k| k >= 1)
        .unwrap_or(MTP_CHAIN_DEPTH_DEFAULT)
        .min(MTP_CHAIN_DEPTH_MAX_SO_VERIFY_ROWS_STAY_UNDER_THE_F32_FAST_PATH_HARD_CAP)
}

pub fn mtp_chain_depth_from_env() -> usize {
    mtp_chain_depth(std::env::var(NV_MTP_K_ENV).ok().as_deref())
}

pub fn mtp_draft_dir_override_from_env() -> Option<PathBuf> {
    std::env::var_os(NV_MTP_DRAFT_DIR_ENV)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

pub fn mtp_round_hidden_reanchor_index(draft_accepted: usize) -> usize {
    draft_accepted
}

pub const NV_Q38_MTP_VERIFY_REPLAY_ENV: &str = "NV_Q38_MTP_VERIFY_REPLAY";

pub fn mtp_verify_replay_selected(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

pub fn mtp_verify_replay_selected_from_env() -> bool {
    mtp_verify_replay_selected(std::env::var(NV_Q38_MTP_VERIFY_REPLAY_ENV).ok().as_deref())
}

pub const MTP_VERIFY_INC2_COMMITS_BATCHED_ROWS_AND_TOLERATES_MROW_VS_M1_DRIFT: &str =
    "increment 2 of the qwen3.8 wgpu MTP verify: the k+1-row verify_chain forward decides \
     acceptance AND supplies the committed state. A full accept advances in place; a partial \
     accept rolls back via advance(0), re-forwards only the accepted prefix through the same \
     M-row graph commit-only (state advancement without the per-row final-norm/lm_head tail or \
     any readback, because every prefix token is already known from the first forward), and \
     full-accept-advances that prefix -- so committed recurrent/conv state always comes from \
     batched kernels and the increment-1 M=1 replay (one full trunk forward per accepted token, \
     the 9x-decode round cost) never runs. Batched rows are NOT required to \
     be byte-identical to sequential M=1 decode; the contract is the acceptance A/B gate, the \
     teacher-forced ppl match, and the empirical logit-drift pin, all in \
     nv-specdecode/tests/qwen38_mtp_wgpu_inc2.rs. NV_Q38_MTP_VERIFY_REPLAY=1 restores the \
     increment-1 replay commit as the debugging escape.";

pub const Q38_WGPU_BATCHED_VERIFY_VS_M1_MAX_ABS_LOGIT_DRIFT_EMPIRICAL_PIN: f32 = 6.0;

pub trait MtpBatchedVerifyTarget {
    fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>>;
    fn verify_chain_commit_only(&mut self, batch: &[u32]) -> Result<()> {
        self.verify_chain(batch).map(|_| ())
    }
    fn advance(&mut self, n: usize) -> Result<()>;
}

#[derive(Debug)]
pub struct MtpBatchedVerifyRound {
    pub batch: Vec<u32>,
    pub accept: crate::chain::ChainAccept,
    pub emitted: Vec<u32>,
    pub prefix_reforwarded_batched: bool,
}

pub fn run_mtp_verify_round<T: MtpBatchedVerifyTarget + ?Sized>(
    target: &mut T,
    anchor: u32,
    drafts: &[u32],
    replay_commit: bool,
) -> Result<MtpBatchedVerifyRound> {
    let batch = crate::chain::build_chain_batch(anchor, drafts, drafts.len() + 1, true)?;
    let amax = target.verify_chain(&batch)?;
    let accept = crate::chain::accept_prefix_argmax(&batch, &amax)?;
    let full = accept.commit_len == batch.len();
    if replay_commit || full {
        target.advance(accept.commit_len)?;
    } else {
        target.advance(0)?;
        target.verify_chain_commit_only(&batch[..accept.commit_len])?;
        target.advance(accept.commit_len)?;
    }
    let mut emitted: Vec<u32> = drafts[..accept.draft_accepted].to_vec();
    emitted.push(accept.next_bonus);
    Ok(MtpBatchedVerifyRound {
        batch,
        accept,
        emitted,
        prefix_reforwarded_batched: !replay_commit && !full,
    })
}

#[cfg(feature = "wgpu")]
impl MtpBatchedVerifyTarget for nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseWgpu {
    fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>> {
        Qwen3_5DenseWgpu::verify_chain(self, batch)
    }

    fn verify_chain_commit_only(&mut self, batch: &[u32]) -> Result<()> {
        Qwen3_5DenseWgpu::verify_chain_commit_only(self, batch)
    }

    fn advance(&mut self, n: usize) -> Result<()> {
        Qwen3_5DenseWgpu::advance(self, n)
    }
}

#[cfg(feature = "wgpu")]
use nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseWgpu;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen38MtpGeometry {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub attn_output_gate: bool,
}

pub const QWEN38_27B_MTP_GEOMETRY: Qwen38MtpGeometry = Qwen38MtpGeometry {
    hidden: 5120,
    heads: 24,
    kv_heads: 4,
    head_dim: 256,
    intermediate: 17408,
    attn_output_gate: true,
};

impl Qwen38MtpGeometry {
    pub fn fc_in(&self) -> usize {
        2 * self.hidden
    }

    pub fn q_proj_out(&self) -> usize {
        let gate_doubles_q_rows = if self.attn_output_gate { 2 } else { 1 };
        self.heads * self.head_dim * gate_doubles_q_rows
    }

    pub fn kv_proj_out(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    pub fn o_proj_in(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn expected_tensor_shapes(&self) -> BTreeMap<String, Vec<usize>> {
        let h = self.hidden;
        let mut m = BTreeMap::new();
        let mut put = |name: &str, shape: Vec<usize>| {
            m.insert(name.to_string(), shape);
        };
        put("mtp.fc.weight", vec![h, self.fc_in()]);
        put("mtp.pre_fc_norm_embedding.weight", vec![h]);
        put("mtp.pre_fc_norm_hidden.weight", vec![h]);
        put("mtp.norm.weight", vec![h]);
        put("mtp.layers.0.input_layernorm.weight", vec![h]);
        put("mtp.layers.0.post_attention_layernorm.weight", vec![h]);
        put(
            "mtp.layers.0.self_attn.q_proj.weight",
            vec![self.q_proj_out(), h],
        );
        put(
            "mtp.layers.0.self_attn.k_proj.weight",
            vec![self.kv_proj_out(), h],
        );
        put(
            "mtp.layers.0.self_attn.v_proj.weight",
            vec![self.kv_proj_out(), h],
        );
        put(
            "mtp.layers.0.self_attn.o_proj.weight",
            vec![h, self.o_proj_in()],
        );
        put("mtp.layers.0.self_attn.q_norm.weight", vec![self.head_dim]);
        put("mtp.layers.0.self_attn.k_norm.weight", vec![self.head_dim]);
        put("mtp.layers.0.mlp.gate_proj.weight", vec![self.intermediate, h]);
        put("mtp.layers.0.mlp.up_proj.weight", vec![self.intermediate, h]);
        put("mtp.layers.0.mlp.down_proj.weight", vec![h, self.intermediate]);
        m
    }
}

pub fn validate_mtp_named_shapes(
    geometry: &Qwen38MtpGeometry,
    named: &BTreeMap<String, Vec<usize>>,
) -> Result<()> {
    let expected = geometry.expected_tensor_shapes();
    for (name, shape) in &expected {
        let got = named
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("mtp tensor missing from checkpoint: {name}"))?;
        ensure!(
            got == shape,
            "mtp tensor {name}: expected shape {shape:?}, checkpoint has {got:?}"
        );
    }
    for name in named.keys() {
        ensure!(
            expected.contains_key(name),
            "checkpoint ships unexpected mtp tensor {name}; the dense head map only knows \
             {} tensors and silently ignoring extras would hide a geometry change",
            expected.len()
        );
    }
    Ok(())
}

pub fn read_safetensors_header_shapes(path: &Path) -> Result<BTreeMap<String, Vec<usize>>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(bytes.len() >= 8, "{}: shorter than a header length", path.display());
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    ensure!(
        bytes.len() >= 8 + n,
        "{}: header claims {n} bytes but the file holds {}",
        path.display(),
        bytes.len()
    );
    let v: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n])
        .with_context(|| format!("parse safetensors header of {}", path.display()))?;
    let obj = v
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{}: header is not an object", path.display()))?;
    let mut out = BTreeMap::new();
    for (k, e) in obj {
        if k == "__metadata__" {
            continue;
        }
        let shape: Vec<usize> = e
            .get("shape")
            .and_then(|s| s.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
            .ok_or_else(|| anyhow::anyhow!("{}: tensor {k} has no shape", path.display()))?;
        out.insert(k.clone(), shape);
    }
    Ok(out)
}

pub fn resolve_mtp_weight_files(
    explicit_dir: Option<&Path>,
    main_checkpoint_dir: &Path,
) -> Result<Vec<PathBuf>> {
    let mut searched: Vec<String> = Vec::new();
    let dirs: Vec<&Path> = explicit_dir
        .into_iter()
        .chain(std::iter::once(main_checkpoint_dir))
        .collect();
    for dir in dirs {
        searched.push(dir.display().to_string());
        let dedicated = dir.join(MTP_WEIGHTS_FILE_NAME);
        if dedicated.is_file() {
            return Ok(vec![dedicated]);
        }
        let index = dir.join("model.safetensors.index.json");
        if index.is_file() {
            let raw = std::fs::read_to_string(&index)
                .with_context(|| format!("read {}", index.display()))?;
            let v: serde_json::Value = serde_json::from_str(&raw)
                .with_context(|| format!("parse {}", index.display()))?;
            let wm = v
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| anyhow::anyhow!("{}: no weight_map", index.display()))?;
            let mut files: Vec<PathBuf> = wm
                .iter()
                .filter(|(k, _)| k.starts_with(MTP_TENSOR_PREFIX))
                .filter_map(|(_, f)| f.as_str().map(|s| dir.join(s)))
                .collect();
            files.sort();
            files.dedup();
            if !files.is_empty() {
                return Ok(files);
            }
        }
        let single = dir.join("model.safetensors");
        if single.is_file()
            && read_safetensors_header_shapes(&single)?
                .keys()
                .any(|k| k.starts_with(MTP_TENSOR_PREFIX))
        {
            return Ok(vec![single]);
        }
    }
    anyhow::bail!(
        "no MTP weights found: searched {searched:?} for {MTP_WEIGHTS_FILE_NAME}, an index \
         mapping {MTP_TENSOR_PREFIX}* tensors, or a single-file checkpoint carrying them; \
         if this checkpoint truly ships no MTP head, NV_DRAFTER=mtp cannot serve it"
    )
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use candle_core::{DType, Device, Tensor, D};
    use nv_layers::attn::{flash_attn, AttnConfig};
    use nv_layers::linear::Linear;
    use nv_layers::mlp::Mlp;
    use nv_layers::norm::RmsNorm;
    use nv_models::graph_engine::GraphedQwen3Moe;
    use nv_models::qwen3_5_moe::decode_prof;
    use nv_models::qwen3_5_moe::{AttentionLayer, Qwen3Moe, Qwen3MoeKvCache};
    use nv_models::qwen3_5_mtp::MtpSpecStats;
    use std::collections::HashMap;

    pub const DRAFTER_FP8_CATCHUP_ROWS_PER_CALL_16_THE_GEMV_E4M3_MK_LEADING_CAP_BECAUSE_THIS_DEVICE_SERVES_NO_CUBLASLT_PER_ROW_FP8:
        usize = 16;

    pub struct MtpKvCache {
        k_bf16_rows_stay_exact_because_the_drafter_never_shares_the_trunk_fp8_slots: Tensor,
        v_rows: Tensor,
        len: usize,
        max_seq: usize,
    }

    impl MtpKvCache {
        pub fn new(
            kv_heads: usize,
            head_dim: usize,
            max_seq: usize,
            dtype: DType,
            device: &Device,
        ) -> Result<Self> {
            Ok(Self {
                k_bf16_rows_stay_exact_because_the_drafter_never_shares_the_trunk_fp8_slots:
                    Tensor::zeros((1usize, max_seq, kv_heads, head_dim), dtype, device)?,
                v_rows: Tensor::zeros((1usize, max_seq, kv_heads, head_dim), dtype, device)?,
                len: 0,
                max_seq,
            })
        }

        pub fn len(&self) -> usize {
            self.len
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        pub fn max_seq(&self) -> usize {
            self.max_seq
        }

        fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
            let d = k_new.dims();
            ensure!(
                d.len() == 4 && d[0] == 1,
                "MtpKvCache.append expects [1, t, kv, hd], got {d:?}"
            );
            let t = d[1];
            ensure!(
                self.len + t <= self.max_seq,
                "MtpKvCache overflow: len {} + t {t} > max_seq {}",
                self.len,
                self.max_seq
            );
            self.k_bf16_rows_stay_exact_because_the_drafter_never_shares_the_trunk_fp8_slots
                .slice_set(&k_new.contiguous()?, 1, self.len)?;
            self.v_rows.slice_set(&v_new.contiguous()?, 1, self.len)?;
            self.len += t;
            Ok(())
        }

        pub fn rewind_to(&mut self, len: usize) -> Result<()> {
            ensure!(
                len <= self.len,
                "MtpKvCache.rewind_to({len}) beyond len {}",
                self.len
            );
            self.len = len;
            Ok(())
        }

        fn views(&self) -> Result<(Tensor, Tensor)> {
            ensure!(self.len > 0, "MtpKvCache.views on an empty cache");
            Ok((
                self.k_bf16_rows_stay_exact_because_the_drafter_never_shares_the_trunk_fp8_slots
                    .narrow(1, 0, self.len)?,
                self.v_rows.narrow(1, 0, self.len)?,
            ))
        }

        pub fn k_rows_host_f32(&self) -> Result<Vec<f32>> {
            let (k, _) = self.views()?;
            Ok(k.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
        }

        pub fn v_rows_host_f32(&self) -> Result<Vec<f32>> {
            let (_, v) = self.views()?;
            Ok(v.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
        }
    }

    fn load_trunk_output_norm_plus_one(
        main_checkpoint_dir: &Path,
        hidden: usize,
        eps: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<RmsNorm> {
        let loader = nv_weights::WeightLoader::open_dir(main_checkpoint_dir, device)
            .with_context(|| {
                format!(
                    "{NV_Q38_MTP_REANCHOR_ENV}=1 needs the trunk output norm from {}",
                    main_checkpoint_dir.display()
                )
            })?;
        let mut raw = None;
        for name in TRUNK_OUTPUT_NORM_TENSOR_CANDIDATES {
            if let Ok(t) = loader.get(name, DType::F32) {
                raw = Some(t);
                break;
            }
        }
        let raw = raw.ok_or_else(|| {
            anyhow::anyhow!(
                "{NV_Q38_MTP_REANCHOR_ENV}=1: none of {TRUNK_OUTPUT_NORM_TENSOR_CANDIDATES:?} \
                 found in {}",
                main_checkpoint_dir.display()
            )
        })?;
        ensure!(
            raw.dims() == [hidden],
            "trunk output norm expected [{hidden}], got {:?}",
            raw.dims()
        );
        let w = raw.affine(1.0, 1.0)?.to_dtype(dtype)?;
        Ok(RmsNorm::new(w, eps))
    }

    pub struct Qwen38DenseMtpHead {
        pre_fc_norm_embedding: RmsNorm,
        pre_fc_norm_hidden: RmsNorm,
        fc: Linear,
        input_layernorm: RmsNorm,
        self_attn: AttentionLayer,
        post_attention_layernorm: RmsNorm,
        mlp: Mlp,
        norm: RmsNorm,
        trunk_output_norm_because_llama_cpp_feeds_the_mtp_post_norm_h_nextn: Option<RmsNorm>,
        draft_only_lm_head_nvfp4_w4a8_halves_twin_of_the_shared_fp8:
            Option<DraftLmHeadNvfp4W4a8DualHalves>,
        geometry: Qwen38MtpGeometry,
        dtype: DType,
    }

    pub struct DraftLmHeadNvfp4W4a8DualHalves {
        top: Linear,
        bottom: Linear,
    }

    impl DraftLmHeadNvfp4W4a8DualHalves {
        fn dual_gemv_logits_one_pass_over_both_row_halves(
            &self,
            x_q8: &cudarc::driver::CudaSlice<i8>,
            x_scale: &cudarc::driver::CudaSlice<f32>,
            device: &Device,
        ) -> Result<Tensor> {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            use half::bf16;
            use std::ffi::c_void;

            let dev = match device {
                Device::Cuda(d) => d.clone(),
                _ => anyhow::bail!("w4a8 draft lm_head needs a CUDA device"),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let n2 = self.top.out_features();
            let n = n2 + self.bottom.out_features();
            let k = self.top.in_features();
            let (tw, ts, t_alpha, t_ig) = self
                .top
                .nvfp4_parts()
                .ok_or_else(|| anyhow::anyhow!("w4a8 draft lm_head: top half lost nvfp4 parts"))?;
            let (bw, bs, b_alpha, b_ig) = self.bottom.nvfp4_parts().ok_or_else(|| {
                anyhow::anyhow!("w4a8 draft lm_head: bottom half lost nvfp4 parts")
            })?;
            let mut y: cudarc::driver::CudaSlice<bf16> =
                unsafe { stream.alloc::<bf16>(n).map_err(|e| anyhow::anyhow!(e))? };
            let rc = {
                let (twp, _g1) = tw.device_ptr(&stream);
                let (tsp, _g2) = ts.device_ptr(&stream);
                let (bwp, _g3) = bw.device_ptr(&stream);
                let (bsp, _g4) = bs.device_ptr(&stream);
                let (xqp, _g5) = x_q8.device_ptr(&stream);
                let (xsp, _g6) = x_scale.device_ptr(&stream);
                let (yp, _g7) = y.device_ptr_mut(&stream);
                unsafe {
                    nv_kernels::cuda::gemv_nvfp4_w4a8_dual_m1(
                        stream.cu_stream() as *mut c_void,
                        twp as *const u8,
                        tsp as *const u8,
                        bwp as *const u8,
                        bsp as *const u8,
                        xqp as *const i8,
                        xsp as *const f32,
                        yp as *mut u16,
                        (yp + (n2 * 2) as u64) as *mut u16,
                        t_alpha * t_ig,
                        b_alpha * b_ig,
                        n2 as i32,
                        k as i32,
                    )
                }
            };
            ensure!(rc == 0, "gemv_nvfp4_w4a8_dual_m1 draft lm_head rc={rc}");
            let storage = candle_core::CudaStorage::wrap_cuda_slice(y, dev);
            Ok(Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, n),
                candle_core::op::BackpropOp::none(),
                false,
            ))
        }
    }

    impl Qwen38DenseMtpHead {
        pub fn geometry_from_dense_base(base: &Qwen3Moe) -> Result<Qwen38MtpGeometry> {
            let cfg = base.config();
            let intermediate = base.dense_intermediate().ok_or_else(|| {
                anyhow::anyhow!(
                    "Qwen38DenseMtpHead needs a dense-hybrid base; this Qwen3Moe was loaded \
                     with routed MoE layers, use nv_models::qwen3_5_mtp::MtpHead for those"
                )
            })?;
            Ok(Qwen38MtpGeometry {
                hidden: cfg.hidden_size,
                heads: cfg.num_attention_heads,
                kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                intermediate,
                attn_output_gate: cfg.attn_output_gate,
            })
        }

        pub fn from_checkpoint(
            explicit_dir: Option<&Path>,
            main_checkpoint_dir: &Path,
            base: &Qwen3Moe,
            device: &Device,
        ) -> Result<Self> {
            let files = resolve_mtp_weight_files(explicit_dir, main_checkpoint_dir)?;
            let mut map: HashMap<String, Tensor> = HashMap::new();
            for f in &files {
                let loaded = candle_core::safetensors::load(f, device)
                    .with_context(|| format!("load MTP safetensors {}", f.display()))?;
                map.extend(
                    loaded
                        .into_iter()
                        .filter(|(k, _)| k.starts_with(MTP_TENSOR_PREFIX)),
                );
            }
            let mut head = Self::from_map(&map, base)?;
            if mtp_draft_lm_head_nvfp4_twin_from_env() {
                head.install_draft_only_nvfp4_lm_head_twin_leaving_the_shared_fp8_untouched(
                    base,
                )?;
            }
            if mtp_reanchor_post_norm_selected_from_env() {
                head.trunk_output_norm_because_llama_cpp_feeds_the_mtp_post_norm_h_nextn = Some(
                    load_trunk_output_norm_plus_one(
                        main_checkpoint_dir,
                        base.config().hidden_size,
                        base.config().rms_norm_eps,
                        base.dtype(),
                        device,
                    )?,
                );
            }
            Ok(head)
        }

        pub fn from_safetensors(
            path: impl AsRef<Path>,
            base: &Qwen3Moe,
            device: &Device,
        ) -> Result<Self> {
            let map = candle_core::safetensors::load(path.as_ref(), device)
                .with_context(|| format!("load MTP safetensors {}", path.as_ref().display()))?;
            Self::from_map(&map, base)
        }

        pub fn from_map(map: &HashMap<String, Tensor>, base: &Qwen3Moe) -> Result<Self> {
            let geometry = Self::geometry_from_dense_base(base)?;
            let cfg = base.config();
            Self::from_map_for_geometry(
                map,
                geometry,
                cfg.rms_norm_eps,
                cfg.rotary_dim(),
                base.dtype(),
            )
        }

        pub fn from_map_for_geometry(
            map: &HashMap<String, Tensor>,
            geometry: Qwen38MtpGeometry,
            rms_norm_eps: f64,
            rotary_dim: usize,
            dtype: DType,
        ) -> Result<Self> {
            let named: BTreeMap<String, Vec<usize>> = map
                .iter()
                .filter(|(k, _)| k.starts_with(MTP_TENSOR_PREFIX))
                .map(|(k, t)| (k.clone(), t.dims().to_vec()))
                .collect();
            validate_mtp_named_shapes(&geometry, &named)?;

            let eps = rms_norm_eps;
            let get = |name: &str| -> Result<Tensor> {
                map.get(name)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("MTP tensor missing: {name}"))
            };
            let fp8_runner = if mtp_drafter_fp8_resident_from_env() {
                let dev = map
                    .values()
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("{NV_Q38_DRAFT_FP8_ENV}=1: empty MTP tensor map"))?
                    .device()
                    .clone();
                let cuda = match &dev {
                    Device::Cuda(d) => d.clone(),
                    _ => anyhow::bail!(
                        "{NV_Q38_DRAFT_FP8_ENV}=1 quantizes the drafter at load through the resident \
                         e4m3 arm and needs the MTP tensors on a CUDA device"
                    ),
                };
                let stream = nv_layers::cuda_stream::current_stream(&cuda);
                Some(std::sync::Arc::new(std::sync::Mutex::new(
                    nv_quant::fp8::Fp8GemmRunner::new(stream)?,
                )))
            } else {
                None
            };
            let lin = |name: &str| -> Result<Linear> {
                let w = get(name)?.to_dtype(dtype)?.contiguous()?;
                match &fp8_runner {
                    Some(runner) => {
                        let out_f = w.dim(0)?;
                        let in_f = w.dim(1)?;
                        let host: Vec<half::bf16> =
                            w.to_dtype(DType::BF16)?.flatten_all()?.to_vec1()?;
                        let (bytes, rows) = nv_layers::linear::fp8_weight_payload(
                            &host,
                            out_f,
                            in_f,
                            None,
                            nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
                        )?;
                        let dev = match w.device() {
                            Device::Cuda(d) => d.clone(),
                            _ => anyhow::bail!(
                                "{NV_Q38_DRAFT_FP8_ENV}=1 drafter quantization needs CUDA tensors"
                            ),
                        };
                        let stream = nv_layers::cuda_stream::current_stream(&dev);
                        #[allow(deprecated)]
                        let weight_u8 = stream
                            .clone_htod(&bytes)
                            .map_err(|e| anyhow::anyhow!(e))?;
                        Linear::new_fp8_e4m3_row_scales_without_the_cublaslt_probe(
                            weight_u8,
                            rows,
                            in_f,
                            out_f,
                            None,
                            w.device(),
                            runner.clone(),
                            nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
                        )
                    }
                    None => Linear::new(w, None),
                }
            };
            let zero_centered_norm_stored_as_weight_minus_one = |name: &str| -> Result<RmsNorm> {
                let raw = get(name)?.to_dtype(DType::F32)?;
                let w = raw.affine(1.0, 1.0)?.to_dtype(dtype)?;
                Ok(RmsNorm::new(w, eps))
            };

            let self_attn = AttentionLayer::from_parts(
                lin("mtp.layers.0.self_attn.q_proj.weight")?,
                lin("mtp.layers.0.self_attn.k_proj.weight")?,
                lin("mtp.layers.0.self_attn.v_proj.weight")?,
                lin("mtp.layers.0.self_attn.o_proj.weight")?,
                zero_centered_norm_stored_as_weight_minus_one(
                    "mtp.layers.0.self_attn.q_norm.weight",
                )?,
                zero_centered_norm_stored_as_weight_minus_one(
                    "mtp.layers.0.self_attn.k_norm.weight",
                )?,
                geometry.heads,
                geometry.kv_heads,
                geometry.head_dim,
                geometry.attn_output_gate,
                rotary_dim,
            );
            let mlp = Mlp::new(
                lin("mtp.layers.0.mlp.gate_proj.weight")?,
                lin("mtp.layers.0.mlp.up_proj.weight")?,
                lin("mtp.layers.0.mlp.down_proj.weight")?,
            )?;

            Ok(Self {
                pre_fc_norm_embedding: zero_centered_norm_stored_as_weight_minus_one(
                    "mtp.pre_fc_norm_embedding.weight",
                )?,
                pre_fc_norm_hidden: zero_centered_norm_stored_as_weight_minus_one(
                    "mtp.pre_fc_norm_hidden.weight",
                )?,
                fc: lin("mtp.fc.weight")?,
                input_layernorm: zero_centered_norm_stored_as_weight_minus_one(
                    "mtp.layers.0.input_layernorm.weight",
                )?,
                self_attn,
                post_attention_layernorm: zero_centered_norm_stored_as_weight_minus_one(
                    "mtp.layers.0.post_attention_layernorm.weight",
                )?,
                mlp,
                norm: zero_centered_norm_stored_as_weight_minus_one("mtp.norm.weight")?,
                trunk_output_norm_because_llama_cpp_feeds_the_mtp_post_norm_h_nextn: None,
                draft_only_lm_head_nvfp4_w4a8_halves_twin_of_the_shared_fp8: None,
                geometry,
                dtype,
            })
        }

        pub fn install_draft_only_nvfp4_lm_head_twin_leaving_the_shared_fp8_untouched(
            &mut self,
            base: &Qwen3Moe,
        ) -> Result<()> {
            let dev = match base.device() {
                Device::Cuda(d) => d.clone(),
                _ => anyhow::bail!(
                    "{NV_Q38_DRAFT_LMHEAD_NVFP4_ENV}=1 quantizes a second resident lm_head copy \
                     and needs the base on a CUDA device"
                ),
            };
            let lm = base.lm_head();
            let n = lm.out_features();
            ensure!(
                n % 256 == 0,
                "{NV_Q38_DRAFT_LMHEAD_NVFP4_ENV}=1 serves the lm_head as two equal row halves \
                 on 128-row swizzle tiles, so vocab {n} must be a multiple of 256"
            );
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let runner = std::sync::Arc::new(std::sync::Mutex::new(
                nv_quant::nvfp4::Nvfp4GemmRunner::new(stream)?,
            ));
            let n2 = n / 2;
            let top = lm.nvfp4_draft_twin_rows_quantized_from_this_resident_copy_which_stays_untouched(
                runner.clone(),
                0,
                n2,
            )?;
            let bottom = lm
                .nvfp4_draft_twin_rows_quantized_from_this_resident_copy_which_stays_untouched(
                    runner, n2, n2,
                )?;
            self.draft_only_lm_head_nvfp4_w4a8_halves_twin_of_the_shared_fp8 =
                Some(DraftLmHeadNvfp4W4a8DualHalves { top, bottom });
            Ok(())
        }

        pub fn draft_lm_head_nvfp4_twin_active(&self) -> bool {
            self.draft_only_lm_head_nvfp4_w4a8_halves_twin_of_the_shared_fp8
                .is_some()
        }

        fn draft_final_add_norm_rowquant_i8_feeding_the_w4a8_lm_head(
            &self,
            mlp_out: &Tensor,
            residual: &Tensor,
        ) -> Result<(
            Tensor,
            cudarc::driver::CudaSlice<i8>,
            cudarc::driver::CudaSlice<f32>,
        )> {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            use half::bf16;
            use std::ffi::c_void;

            let hidden = self.geometry.hidden;
            let dev = match mlp_out.device() {
                Device::Cuda(d) => d.clone(),
                _ => anyhow::bail!("w4a8 draft final norm needs CUDA tensors"),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let x_c = mlp_out.contiguous()?;
            let r_c = residual.contiguous()?;
            let w_c = self.norm.weight_bf16().contiguous()?;
            let mut res_out: cudarc::driver::CudaSlice<bf16> = unsafe {
                stream.alloc::<bf16>(hidden).map_err(|e| anyhow::anyhow!(e))?
            };
            let mut x_q8: cudarc::driver::CudaSlice<i8> = unsafe {
                stream.alloc::<i8>(hidden).map_err(|e| anyhow::anyhow!(e))?
            };
            let mut x_scale: cudarc::driver::CudaSlice<f32> =
                unsafe { stream.alloc::<f32>(1).map_err(|e| anyhow::anyhow!(e))? };
            let rc = {
                let (xs, xl) = x_c.storage_and_layout();
                let (rs, rl) = r_c.storage_and_layout();
                let (ws, wl) = w_c.storage_and_layout();
                let x_cuda = match &*xs {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("w4a8 draft final norm: mlp_out storage not cuda"),
                };
                let r_cuda = match &*rs {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("w4a8 draft final norm: residual storage not cuda"),
                };
                let w_cuda = match &*ws {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("w4a8 draft final norm: norm weight storage not cuda"),
                };
                let x_view = x_cuda.as_cuda_slice::<bf16>()?.slice(xl.start_offset()..);
                let r_view = r_cuda.as_cuda_slice::<bf16>()?.slice(rl.start_offset()..);
                let w_view = w_cuda.as_cuda_slice::<bf16>()?.slice(wl.start_offset()..);
                let (px, _gx) = x_view.device_ptr(&stream);
                let (pr, _gr) = r_view.device_ptr(&stream);
                let (pw, _gw) = w_view.device_ptr(&stream);
                let (pro, _g1) = res_out.device_ptr_mut(&stream);
                let (pq, _g2) = x_q8.device_ptr_mut(&stream);
                let (ps, _g3) = x_scale.device_ptr_mut(&stream);
                unsafe {
                    nv_kernels::cuda::rmsnorm_residual_writeout_rowquant_i8_m1(
                        stream.cu_stream() as *mut c_void,
                        px as *const u16,
                        pr as *const u16,
                        pw as *const u16,
                        pro as *mut u16,
                        pq as *mut i8,
                        ps as *mut f32,
                        hidden as i32,
                        self.norm.eps() as f32,
                    )
                }
            };
            ensure!(rc == 0, "rmsnorm_residual_writeout_rowquant_i8_m1 draft rc={rc}");
            let h_storage = candle_core::CudaStorage::wrap_cuda_slice(res_out, dev);
            let h = Tensor::from_storage(
                candle_core::Storage::Cuda(h_storage),
                (1usize, 1usize, hidden),
                candle_core::op::BackpropOp::none(),
                false,
            );
            Ok((h, x_q8, x_scale))
        }

        pub fn reanchor_post_norm_active(&self) -> bool {
            self.trunk_output_norm_because_llama_cpp_feeds_the_mtp_post_norm_h_nextn
                .is_some()
        }

        pub fn trunk_hidden_rows_as_drafter_anchor_input(&self, rows: &Tensor) -> Result<Tensor> {
            match &self.trunk_output_norm_because_llama_cpp_feeds_the_mtp_post_norm_h_nextn {
                Some(n) => Ok(n.forward(&rows.to_dtype(self.dtype)?)?),
                None => Ok(rows.clone()),
            }
        }

        pub fn new_kv_cache(&self, max_seq: usize, device: &Device) -> Result<MtpKvCache> {
            MtpKvCache::new(
                self.geometry.kv_heads,
                self.geometry.head_dim,
                max_seq,
                self.dtype,
                device,
            )
        }

        fn layer_input_from_sampled_tokens_and_their_preceding_trunk_hidden(
            &self,
            base: &Qwen3Moe,
            tokens: &[u32],
            hiddens: &Tensor,
        ) -> Result<Tensor> {
            let device = hiddens.device().clone();
            let tok = Tensor::from_vec(tokens.to_vec(), tokens.len(), &device)?;
            self.layer_input_from_token_ids_already_on_device(base, &tok, hiddens)
        }

        fn layer_input_from_token_ids_already_on_device(
            &self,
            base: &Qwen3Moe,
            tok: &Tensor,
            hiddens: &Tensor,
        ) -> Result<Tensor> {
            let t = tok.dim(0)?;
            let h = self.geometry.hidden;
            ensure!(
                hiddens.dims() == [1, t, h],
                "MTP layer input: {t} tokens need trunk hiddens [1, {t}, {h}], got {:?}",
                hiddens.dims()
            );
            let emb = base
                .embed_weight()
                .index_select(tok, 0)?
                .reshape((1usize, t, h))?
                .to_dtype(self.dtype)?;
            let norm_emb = self.pre_fc_norm_embedding.forward(&emb)?;
            let norm_hid = self
                .pre_fc_norm_hidden
                .forward(&hiddens.to_dtype(self.dtype)?)?;
            let fused_in = Tensor::cat(&[&norm_emb, &norm_hid], D::Minus1)?.contiguous()?;
            decode_prof::lap("draft_embed_norms_cat");
            let out = self.fc.forward(&fused_in);
            decode_prof::lap("draft_fc");
            out
        }

        pub fn catch_up_kv_recomputing_and_discarding_q_which_v1_prices_over_a_kv_only_projection(
            &self,
            base: &Qwen3Moe,
            tokens: &[u32],
            hiddens: &Tensor,
            cache: &mut MtpKvCache,
        ) -> Result<()> {
            let t = tokens.len();
            ensure!(t > 0, "MTP catch-up with zero tokens");
            let rows_per_call = if self.fc.fp8_scale_parts().is_some() {
                DRAFTER_FP8_CATCHUP_ROWS_PER_CALL_16_THE_GEMV_E4M3_MK_LEADING_CAP_BECAUSE_THIS_DEVICE_SERVES_NO_CUBLASLT_PER_ROW_FP8
            } else {
                t
            };
            let mut off = 0usize;
            while off < t {
                let n = (t - off).min(rows_per_call);
                let start = cache.len();
                let fused = self.layer_input_from_sampled_tokens_and_their_preceding_trunk_hidden(
                    base,
                    &tokens[off..off + n],
                    &hiddens.narrow(1, off, n)?.contiguous()?,
                )?;
                let normed = self.input_layernorm.forward(&fused)?;
                let positions: Vec<i32> = (start as i32..(start + n) as i32).collect();
                let pos_t = Tensor::from_vec(positions, n, hiddens.device())?;
                let (_q_discarded, k, v, _gate_discarded) = self
                    .self_attn
                    .project_qkv_roped_for_a_drafter_owned_kv(&normed, base.rope(), &pos_t)?;
                cache.append(&k, &v)?;
                off += n;
            }
            Ok(())
        }

        pub fn prefill_prompt_kv_shifted_by_one_with_position_zero_on_a_zero_hidden(
            &self,
            base: &Qwen3Moe,
            prompt: &[u32],
            trunk_hidden: &Tensor,
            cache: &mut MtpKvCache,
        ) -> Result<()> {
            let seq = prompt.len();
            ensure!(seq > 0, "MTP prompt prefill with empty prompt");
            ensure!(
                cache.is_empty(),
                "MTP prompt prefill expects a fresh cache, len={}",
                cache.len()
            );
            let h = self.geometry.hidden;
            ensure!(
                trunk_hidden.dims() == [1, seq, h],
                "MTP prompt prefill: trunk hidden must be [1, {seq}, {h}], got {:?}",
                trunk_hidden.dims()
            );
            let zero_hidden_stands_in_for_h_minus_one_and_rmsnorms_to_zero =
                Tensor::zeros((1usize, 1usize, h), trunk_hidden.dtype(), trunk_hidden.device())?;
            let hiddens = Tensor::cat(
                &[
                    &zero_hidden_stands_in_for_h_minus_one_and_rmsnorms_to_zero,
                    &trunk_hidden.narrow(1, 0, seq - 1)?,
                ],
                1,
            )?
            .contiguous()?;
            self.catch_up_kv_recomputing_and_discarding_q_which_v1_prices_over_a_kv_only_projection(
                base, prompt, &hiddens, cache,
            )
        }

        pub fn forward_draft(
            &self,
            base: &Qwen3Moe,
            base_hidden: &Tensor,
            next_token_id: u32,
            cache: &mut MtpKvCache,
        ) -> Result<(Tensor, Tensor)> {
            let device = base_hidden.device().clone();
            let tok = Tensor::from_vec(vec![next_token_id], 1usize, &device)?;
            self.forward_draft_from_token_ids_already_on_device(base, base_hidden, &tok, cache)
        }

        pub fn forward_draft_from_token_ids_already_on_device(
            &self,
            base: &Qwen3Moe,
            base_hidden: &Tensor,
            next_token: &Tensor,
            cache: &mut MtpKvCache,
        ) -> Result<(Tensor, Tensor)> {
            let d = base_hidden.dims();
            let hidden = self.geometry.hidden;
            ensure!(
                d == [1, 1, hidden],
                "MTP base_hidden: expected [1, 1, {hidden}], got {d:?}"
            );
            let device = base_hidden.device().clone();
            let position = cache.len() as i32;

            let mut h = self.layer_input_from_token_ids_already_on_device(
                base,
                next_token,
                base_hidden,
            )?;

            let positions = Tensor::from_vec(vec![position], 1usize, &device)?;
            let residual = h.clone();
            let normed = self.input_layernorm.forward(&h)?;
            let (q, k, v, q_gate) = self
                .self_attn
                .project_qkv_roped_for_a_drafter_owned_kv(&normed, base.rope(), &positions)?;
            cache.append(&k, &v)?;
            decode_prof::lap("draft_kv_append");
            let (k_full, v_full) = cache.views()?;
            let attn_cfg = AttnConfig {
                num_heads: self.geometry.heads,
                num_kv_heads: self.geometry.kv_heads,
                head_dim: self.geometry.head_dim,
                softmax_scale: 1.0 / (self.geometry.head_dim as f32).sqrt(),
                causal: true,
            };
            let attn_out = flash_attn(
                &q.contiguous()?,
                &k_full.contiguous()?,
                &v_full.contiguous()?,
                &attn_cfg,
            )?;
            decode_prof::lap("draft_attn_flash");
            let attn = self.self_attn.finalize_attn_from_a_drafter_owned_kv(
                attn_out,
                q_gate,
                1,
                1,
                self.dtype,
            )?;
            h = residual.add(&attn)?;
            decode_prof::lap("draft_attn_finalize");

            let residual2 = h.clone();
            let normed2 = self.post_attention_layernorm.forward(&h)?;
            let mlp_out = self.mlp.forward(&normed2)?.to_dtype(self.dtype)?;

            if let Some(halves) = &self.draft_only_lm_head_nvfp4_w4a8_halves_twin_of_the_shared_fp8
            {
                decode_prof::lap("draft_mlp");
                let (h_new, x_q8, x_scale) = self
                    .draft_final_add_norm_rowquant_i8_feeding_the_w4a8_lm_head(
                        &mlp_out, &residual2,
                    )?;
                h = h_new;
                let chained_hidden = if self.reanchor_post_norm_active() {
                    self.norm.forward(&h)?
                } else {
                    h.clone()
                };
                decode_prof::lap("draft_final_norm");
                let logits = halves.dual_gemv_logits_one_pass_over_both_row_halves(
                    &x_q8,
                    &x_scale,
                    h.device(),
                )?;
                decode_prof::lap("draft_lm_head");
                return Ok((logits, chained_hidden));
            }

            h = residual2.add(&mlp_out)?;
            decode_prof::lap("draft_mlp");

            let normed_final = self.norm.forward(&h)?;
            let chained_hidden = if self.reanchor_post_norm_active() {
                normed_final.clone()
            } else {
                h.clone()
            };
            decode_prof::lap("draft_final_norm");
            let logits = base.lm_head().forward(&normed_final)?;
            let logits = logits.reshape((1usize, logits.dim(D::Minus1)?))?;
            decode_prof::lap("draft_lm_head");
            Ok((logits, chained_hidden))
        }

        pub fn forward_draft_tok(
            &self,
            base: &Qwen3Moe,
            base_hidden: &Tensor,
            next_token_id: u32,
            cache: &mut MtpKvCache,
        ) -> Result<(u32, Tensor)> {
            let (logits, mtp_hidden) = self.forward_draft(base, base_hidden, next_token_id, cache)?;
            let tok = logits
                .argmax(D::Minus1)?
                .flatten_all()?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?[0];
            decode_prof::lap("draft_argmax_readback");
            Ok((tok, mtp_hidden))
        }

        fn device_row_argmax_u32_matching_candle_lowest_index_tie_break(
            logits: &Tensor,
        ) -> Result<Tensor> {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let dev = match logits.device() {
                Device::Cuda(d) => d.clone(),
                _ => {
                    return Ok(logits
                        .argmax(D::Minus1)?
                        .flatten_all()?
                        .to_dtype(DType::U32)?)
                }
            };
            let vocab = logits.dim(D::Minus1)?;
            let lf = logits.to_dtype(DType::F32)?.contiguous()?;
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let parts = nv_kernels::cuda::dflash_accept_parts();
            let mut part_val = unsafe {
                stream
                    .alloc::<f32>(parts)
                    .map_err(|e| anyhow::anyhow!(e))?
            };
            let mut part_idx = unsafe {
                stream
                    .alloc::<i32>(parts)
                    .map_err(|e| anyhow::anyhow!(e))?
            };
            let mut row_argmax = unsafe {
                stream.alloc::<u32>(1).map_err(|e| anyhow::anyhow!(e))?
            };
            let mut out_buf = unsafe {
                stream.alloc::<u32>(2).map_err(|e| anyhow::anyhow!(e))?
            };
            let drafts_buf = stream
                .alloc_zeros::<u32>(1)
                .map_err(|e| anyhow::anyhow!(e))?;
            let rc = {
                let (l_storage, ll) = lf.storage_and_layout();
                let l_cuda = match &*l_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("draft argmax: expected cuda storage for logits"),
                };
                let l_slice = l_cuda.as_cuda_slice::<f32>()?;
                let l_view = l_slice.slice(ll.start_offset()..);
                let (lp, _gl) = l_view.device_ptr(&stream);
                let (dp, _gd) = drafts_buf.device_ptr(&stream);
                let (pvp, _gpv) = part_val.device_ptr_mut(&stream);
                let (pip, _gpi) = part_idx.device_ptr_mut(&stream);
                let (rp, _gr) = row_argmax.device_ptr_mut(&stream);
                let (op, _go) = out_buf.device_ptr_mut(&stream);
                unsafe {
                    nv_kernels::cuda::dflash_accept_f32(
                        stream.cu_stream() as *mut _,
                        lp as *const f32,
                        dp as *const u32,
                        rp as *mut u32,
                        op as *mut u32,
                        pvp as *mut f32,
                        pip as *mut i32,
                        1,
                        vocab as i32,
                    )
                }
            };
            ensure!(rc == 0, "dflash_accept_f32 m=1 draft argmax returned {rc}");
            let storage = candle_core::CudaStorage::wrap_cuda_slice(row_argmax, dev);
            Ok(Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                1usize,
                candle_core::op::BackpropOp::none(),
                false,
            ))
        }

        pub fn draft_chain_reading_tokens_back_once_because_per_step_readback_stalls_the_stream(
            &self,
            base: &Qwen3Moe,
            anchor: u32,
            base_hidden: &Tensor,
            k: usize,
            forced_drafts: Option<&[u32]>,
            cache: &mut MtpKvCache,
        ) -> Result<Vec<u32>> {
            ensure!(k >= 1, "draft chain: k must be >= 1");
            if let Some(f) = forced_drafts {
                ensure!(
                    f.len() == k,
                    "draft chain: forced drafts len {} != k {k}",
                    f.len()
                );
            }
            let device = base_hidden.device().clone();
            let mut h = base_hidden.clone();
            let mut tok = Tensor::from_vec(vec![anchor], 1usize, &device)?;
            let mut sampled: Vec<Tensor> = Vec::with_capacity(k);
            for j in 0..k {
                let (logits, dhidden) =
                    self.forward_draft_from_token_ids_already_on_device(base, &h, &tok, cache)?;
                tok = match forced_drafts {
                    Some(f) => Tensor::from_vec(vec![f[j]], 1usize, &device)?,
                    None => {
                        let am =
                            Self::device_row_argmax_u32_matching_candle_lowest_index_tie_break(
                                &logits,
                            )?;
                        sampled.push(am.clone());
                        am
                    }
                };
                h = dhidden;
            }
            match forced_drafts {
                Some(f) => Ok(f.to_vec()),
                None => Ok(Tensor::cat(&sampled, 0)?.to_vec1::<u32>()?),
            }
        }
    }

    pub struct Qwen38MtpDecodeSession<'a> {
        base: &'a Qwen3Moe,
        mtp: &'a Qwen38DenseMtpHead,
        k: usize,
        cache: Qwen3MoeKvCache,
        mtp_kv: MtpKvCache,
        anchor: u32,
        base_hidden: Tensor,
        max_seq: usize,
        pub stats: MtpSpecStats,
    }

    impl<'a> Qwen38MtpDecodeSession<'a> {
        pub fn start(
            base: &'a Qwen3Moe,
            mtp: &'a Qwen38DenseMtpHead,
            k: usize,
            prompt: &[u32],
            max_seq: usize,
        ) -> Result<Self> {
            ensure!(!prompt.is_empty(), "mtp session: empty prompt");
            ensure!(k >= 1, "mtp session: k must be >= 1");
            ensure!(
                prompt.len() + k + 1 <= max_seq,
                "mtp session: prompt {} + k+1 {} > max_seq {max_seq}",
                prompt.len(),
                k + 1
            );
            let hidden = base.config().hidden_size;
            assert_mtp_chain_depth_fits_verify_lm_head_rows_ceiling(hidden, k, 0)?;
            let device_reported_rows_ceiling =
                nv_kernels::cuda::gemv_i8_normed_mk_max_m(hidden as i32).max(0) as usize;
            ensure!(
                mtp_verify_rows_per_round(k) <= device_reported_rows_ceiling,
                "this device's lm_head rows-per-call ceiling is {device_reported_rows_ceiling} \
                 for hidden={hidden}; a k={k} mtp round verifies {} rows (the #107 dflash k=8 \
                 overflow is the cautionary tale)",
                mtp_verify_rows_per_round(k)
            );
            let device = base.device().clone();
            let mut cache = base.new_kv_cache(max_seq)?;
            let mut mtp_kv = mtp.new_kv_cache(max_seq, &device)?;

            let seq = prompt.len();
            let tokens = Tensor::from_vec(prompt.to_vec(), (1usize, seq), &device)?;
            let pos = Tensor::from_vec((0..seq as i32).collect::<Vec<i32>>(), seq, &device)?;
            let (logits, hidden) = base.forward_with_cache_dispatched_hidden_rows(
                &tokens,
                &pos,
                &mut cache,
                None,
                Some(1),
            )?;
            let anchor = argmax_last_row(&logits)?;
            ensure!(
                !mtp_reanchor_post_norm_selected_from_env() || mtp.reanchor_post_norm_active(),
                "{NV_Q38_MTP_REANCHOR_ENV}=1 but this head was not loaded through from_checkpoint \
                 so it carries no trunk output norm"
            );
            let hidden = mtp.trunk_hidden_rows_as_drafter_anchor_input(&hidden)?;
            let base_hidden = hidden.narrow(1, seq - 1, 1)?.contiguous()?;
            mtp.prefill_prompt_kv_shifted_by_one_with_position_zero_on_a_zero_hidden(
                base, prompt, &hidden, &mut mtp_kv,
            )?;
            drop(logits);
            drop(hidden);
            let _ = device.synchronize();

            Ok(Self {
                base,
                mtp,
                k,
                cache,
                mtp_kv,
                anchor,
                base_hidden,
                max_seq,
                stats: MtpSpecStats::default(),
            })
        }

        pub fn anchor(&self) -> u32 {
            self.anchor
        }

        pub fn committed_len(&self) -> usize {
            self.cache.current_len()
        }

        pub fn round_fits(&self) -> bool {
            self.cache.current_len() + self.k + 1 <= self.max_seq
        }

        pub fn round(&mut self) -> Result<Vec<u32>> {
            self.round_inner(None)
        }

        pub fn round_with_drafts_from_a_clairvoyant_test_oracle(
            &mut self,
            drafts: &[u32],
        ) -> Result<Vec<u32>> {
            ensure!(
                drafts.len() == self.k,
                "oracle round needs exactly k={} drafts, got {}",
                self.k,
                drafts.len()
            );
            self.round_inner(Some(drafts))
        }

        fn round_inner(&mut self, forced_drafts: Option<&[u32]>) -> Result<Vec<u32>> {
            let device = self.base.device().clone();
            let round_t0 = std::time::Instant::now();
            let l = self.cache.current_len();
            let k = self.k;
            ensure!(
                l + k + 1 <= self.max_seq,
                "mtp verify overflow: committed={l} + k+1={} > max_seq={}",
                k + 1,
                self.max_seq
            );
            ensure!(
                self.mtp_kv.len() == l,
                "mtp kv desync: drafter cache holds {} rows but the trunk committed {l}",
                self.mtp_kv.len()
            );

            let drafts: Vec<u32> = if draft_fast_device_chained_tokens_selected_from_env() {
                self.mtp
                    .draft_chain_reading_tokens_back_once_because_per_step_readback_stalls_the_stream(
                        self.base,
                        self.anchor,
                        &self.base_hidden,
                        k,
                        forced_drafts,
                        &mut self.mtp_kv,
                    )?
            } else {
                let mut drafts: Vec<u32> = Vec::with_capacity(k);
                let mut h = self.base_hidden.clone();
                let mut tok = self.anchor;
                for j in 0..k {
                    let (d, dhidden) =
                        self.mtp
                            .forward_draft_tok(self.base, &h, tok, &mut self.mtp_kv)?;
                    let d = match forced_drafts {
                        Some(f) => f[j],
                        None => d,
                    };
                    drafts.push(d);
                    tok = d;
                    h = dhidden;
                }
                drafts
            };
            let _ = device.synchronize();
            let draft_dt = round_t0.elapsed().as_secs_f64();

            let verify_t0 = std::time::Instant::now();
            let batch = crate::chain::build_chain_batch(self.anchor, &drafts, k + 1, true)?;
            let m = batch.len();
            let block_pos: Vec<i32> = (0..m as i32).map(|i| l as i32 + i).collect();
            let bt = Tensor::from_vec(batch.clone(), (1usize, m), &device)?;
            let bp = Tensor::from_vec(block_pos, m, &device)?;

            self.cache.set_capture_lin_ckpts(true);
            let (vlogits, vhidden) = self.base.forward_with_cache_dispatched_hidden(
                &bt,
                &bp,
                &mut self.cache,
                None,
            )?;
            let greedy: Vec<u32> = vlogits
                .argmax(D::Minus1)?
                .flatten_all()?
                .to_dtype(DType::U32)?
                .to_vec1()?;
            let _ = device.synchronize();
            let verify_dt = verify_t0.elapsed().as_secs_f64();

            let commit_t0 = std::time::Instant::now();
            let acc = crate::chain::accept_prefix_argmax(&batch, &greedy)?;
            let consumed = acc.commit_len;
            let accepted = acc.draft_accepted;
            let bonus = acc.next_bonus;

            let mut emitted: Vec<u32> = drafts[..accepted].to_vec();
            emitted.push(bonus);

            let vhidden_for_drafter = self
                .mtp
                .trunk_hidden_rows_as_drafter_anchor_input(&vhidden)?;
            self.base_hidden = vhidden_for_drafter
                .narrow(1, mtp_round_hidden_reanchor_index(accepted), 1)?
                .contiguous()?;

            self.mtp_kv
                .rewind_to(l + 1)?;
            if accepted >= 1 {
                self.mtp.catch_up_kv_recomputing_and_discarding_q_which_v1_prices_over_a_kv_only_projection(
                    self.base,
                    &batch[1..=accepted],
                    &vhidden_for_drafter.narrow(1, 0, accepted)?.contiguous()?,
                    &mut self.mtp_kv,
                )?;
            }
            ensure!(
                self.mtp_kv.len() == l + consumed,
                "mtp kv catch-up desync: drafter cache holds {} rows, trunk commits {}",
                self.mtp_kv.len(),
                l + consumed
            );

            self.cache.set_current_len(l + consumed);
            self.cache.rollback_lin_to(consumed)?;
            self.anchor = bonus;

            let _ = device.synchronize();
            let commit_dt = commit_t0.elapsed().as_secs_f64();
            self.stats
                .round_ms
                .push(1000.0 * round_t0.elapsed().as_secs_f64());
            self.stats.draft_ms += 1000.0 * draft_dt;
            self.stats.verify_ms += 1000.0 * verify_dt;
            self.stats.commit_ms += 1000.0 * commit_dt;
            self.stats.rounds += 1;
            self.stats.drafted += drafts.len();
            self.stats.accepted += accepted;
            self.stats.emitted += emitted.len();
            if accepted > 0 {
                self.stats.pos0_accepted += 1;
            }
            *self.stats.accept_len_hist.entry(accepted).or_insert(0) += 1;
            Ok(emitted)
        }
    }

    pub struct Qwen38MtpGraphedDecodeSession<'a> {
        engine: &'a mut GraphedQwen3Moe,
        mtp: &'a Qwen38DenseMtpHead,
        k: usize,
        mtp_kv: MtpKvCache,
        anchor: u32,
        base_hidden: Tensor,
        max_seq: usize,
        pub stats: MtpSpecStats,
    }

    impl<'a> Qwen38MtpGraphedDecodeSession<'a> {
        pub fn start(
            engine: &'a mut GraphedQwen3Moe,
            mtp: &'a Qwen38DenseMtpHead,
            k: usize,
            prompt: &[u32],
        ) -> Result<Self> {
            ensure!(!prompt.is_empty(), "graphed mtp session: empty prompt");
            ensure!(k >= 1, "graphed mtp session: k must be >= 1");
            let max_seq = engine.cache().max_seq_len();
            ensure!(
                prompt.len() + k + 1 <= max_seq,
                "graphed mtp session: prompt {} + k+1 {} > max_seq {max_seq}",
                prompt.len(),
                k + 1
            );
            let hidden = engine.underlying().config().hidden_size;
            assert_mtp_chain_depth_fits_verify_lm_head_rows_ceiling(hidden, k, 0)?;
            let device_reported_rows_ceiling =
                nv_kernels::cuda::gemv_i8_normed_mk_max_m(hidden as i32).max(0) as usize;
            ensure!(
                mtp_verify_rows_per_round(k) <= device_reported_rows_ceiling,
                "this device's lm_head rows-per-call ceiling is {device_reported_rows_ceiling} \
                 for hidden={hidden}; a k={k} mtp round verifies {} rows",
                mtp_verify_rows_per_round(k)
            );
            engine.reset()?;
            engine.install_grouped_moe()?;
            let device = engine.device().clone();
            let mut mtp_kv = mtp.new_kv_cache(max_seq, &device)?;
            let (last_row, hidden_t) = engine.prefill_hidden_serving_last_row_logits(prompt)?;
            engine.ensure_verify_lane(k + 1)?;
            let anchor = argmax_host_row(&last_row)?;
            ensure!(
                !mtp_reanchor_post_norm_selected_from_env() || mtp.reanchor_post_norm_active(),
                "{NV_Q38_MTP_REANCHOR_ENV}=1 but this head was not loaded through from_checkpoint \
                 so it carries no trunk output norm"
            );
            let hidden_t = mtp.trunk_hidden_rows_as_drafter_anchor_input(&hidden_t)?;
            let base_hidden = hidden_t.narrow(1, prompt.len() - 1, 1)?.contiguous()?;
            mtp.prefill_prompt_kv_shifted_by_one_with_position_zero_on_a_zero_hidden(
                engine.underlying(),
                prompt,
                &hidden_t,
                &mut mtp_kv,
            )?;
            let _ = device.synchronize();
            Ok(Self {
                engine,
                mtp,
                k,
                mtp_kv,
                anchor,
                base_hidden,
                max_seq,
                stats: MtpSpecStats::default(),
            })
        }

        pub fn anchor(&self) -> u32 {
            self.anchor
        }

        pub fn drafter_anchor_token_and_trunk_hidden_row(&self) -> (u32, Tensor) {
            (self.anchor, self.base_hidden.clone())
        }

        pub fn trunk_for_shadow_probes(&self) -> &Qwen3Moe {
            self.engine.underlying()
        }

        pub fn committed_len(&self) -> usize {
            self.engine.current_pos()
        }

        pub fn round_fits(&self) -> bool {
            self.engine.current_pos() + self.k + 1 <= self.max_seq
        }

        pub fn round(&mut self) -> Result<Vec<u32>> {
            self.round_inner(None)
        }

        pub fn round_with_drafts_from_a_clairvoyant_test_oracle(
            &mut self,
            drafts: &[u32],
        ) -> Result<Vec<u32>> {
            ensure!(
                drafts.len() == self.k,
                "oracle round needs exactly k={} drafts, got {}",
                self.k,
                drafts.len()
            );
            self.round_inner(Some(drafts))
        }

        pub fn profile_one_draft_chain_then_rewind_arming_nv_prof_decode_whose_every_lap_syncs(
            &mut self,
        ) -> Result<()> {
            let device = self.engine.device().clone();
            let l = self.mtp_kv.len();
            ensure!(
                l == self.engine.current_pos(),
                "draft-chain profile needs a settled session: drafter kv holds {} rows, trunk \
                 committed {}",
                l,
                self.engine.current_pos()
            );
            let _ = device.synchronize();
            decode_prof::begin_env_nv_prof_decode_refusing_mid_capture_because_every_lap_syncs(
                &device,
            );
            let mut h = self.base_hidden.clone();
            let mut tok = self.anchor;
            for _ in 0..self.k {
                let (d, dhidden) =
                    self.mtp
                        .forward_draft_tok(self.engine.underlying(), &h, tok, &mut self.mtp_kv)?;
                tok = d;
                h = dhidden;
            }
            decode_prof::report_and_end(l);
            self.mtp_kv.rewind_to(l)
        }

        fn round_inner(&mut self, forced_drafts: Option<&[u32]>) -> Result<Vec<u32>> {
            let device = self.engine.device().clone();
            let round_t0 = std::time::Instant::now();
            let l = self.engine.current_pos();
            let k = self.k;
            ensure!(
                l + k + 1 <= self.max_seq,
                "graphed mtp verify overflow: committed={l} + k+1={} > max_seq={}",
                k + 1,
                self.max_seq
            );
            ensure!(
                self.mtp_kv.len() == l,
                "graphed mtp kv desync: drafter cache holds {} rows but the trunk committed {l}",
                self.mtp_kv.len()
            );

            let drafts: Vec<u32> = if draft_fast_device_chained_tokens_selected_from_env() {
                self.mtp
                    .draft_chain_reading_tokens_back_once_because_per_step_readback_stalls_the_stream(
                        self.engine.underlying(),
                        self.anchor,
                        &self.base_hidden,
                        k,
                        forced_drafts,
                        &mut self.mtp_kv,
                    )?
            } else {
                let mut drafts: Vec<u32> = Vec::with_capacity(k);
                let mut h = self.base_hidden.clone();
                let mut tok = self.anchor;
                for j in 0..k {
                    let (d, dhidden) = self.mtp.forward_draft_tok(
                        self.engine.underlying(),
                        &h,
                        tok,
                        &mut self.mtp_kv,
                    )?;
                    let d = match forced_drafts {
                        Some(f) => f[j],
                        None => d,
                    };
                    drafts.push(d);
                    tok = d;
                    h = dhidden;
                }
                drafts
            };
            let _ = device.synchronize();
            let draft_dt = round_t0.elapsed().as_secs_f64();

            let verify_t0 = std::time::Instant::now();
            let batch = crate::chain::build_chain_batch(self.anchor, &drafts, k + 1, true)?;
            let m = batch.len();
            self.engine.forward_verify_chain(&batch)?;
            let flat = self.engine.verify_logits_host()?;
            let vocab = self.engine.vocab_size();
            let greedy = argmax_host_rows(&flat, m, vocab)?;
            let verify_dt = verify_t0.elapsed().as_secs_f64();

            let commit_t0 = std::time::Instant::now();
            let acc = crate::chain::accept_prefix_argmax(&batch, &greedy)?;
            let consumed = acc.commit_len;
            let accepted = acc.draft_accepted;
            let bonus = acc.next_bonus;

            let mut emitted: Vec<u32> = drafts[..accepted].to_vec();
            emitted.push(bonus);

            let vhidden = self
                .engine
                .verify_hidden_rows_tensor_valid_until_next_forward()?;
            let vhidden_for_drafter = self
                .mtp
                .trunk_hidden_rows_as_drafter_anchor_input(&vhidden)?;
            self.base_hidden = vhidden_for_drafter
                .narrow(1, mtp_round_hidden_reanchor_index(accepted), 1)?
                .copy()?;

            self.mtp_kv.rewind_to(l + 1)?;
            if accepted >= 1 {
                self.mtp.catch_up_kv_recomputing_and_discarding_q_which_v1_prices_over_a_kv_only_projection(
                    self.engine.underlying(),
                    &batch[1..=accepted],
                    &vhidden_for_drafter.narrow(1, 0, accepted)?.contiguous()?,
                    &mut self.mtp_kv,
                )?;
            }
            ensure!(
                self.mtp_kv.len() == l + consumed,
                "graphed mtp kv catch-up desync: drafter cache holds {} rows, trunk commits {}",
                self.mtp_kv.len(),
                l + consumed
            );

            self.engine.commit_verify_consumed(consumed)?;
            self.anchor = bonus;

            let _ = device.synchronize();
            let commit_dt = commit_t0.elapsed().as_secs_f64();
            self.stats
                .round_ms
                .push(1000.0 * round_t0.elapsed().as_secs_f64());
            self.stats.draft_ms += 1000.0 * draft_dt;
            self.stats.verify_ms += 1000.0 * verify_dt;
            self.stats.commit_ms += 1000.0 * commit_dt;
            self.stats.rounds += 1;
            self.stats.drafted += drafts.len();
            self.stats.accepted += accepted;
            self.stats.emitted += emitted.len();
            if accepted > 0 {
                self.stats.pos0_accepted += 1;
            }
            *self.stats.accept_len_hist.entry(accepted).or_insert(0) += 1;
            Ok(emitted)
        }
    }

    fn argmax_host_row(row: &[f32]) -> Result<u32> {
        nv_layers::sampler::argmax_host_row(row)
    }

    fn argmax_host_rows(flat: &[f32], rows: usize, vocab: usize) -> Result<Vec<u32>> {
        ensure!(
            flat.len() == rows * vocab,
            "argmax rows: {} logits != {rows} x {vocab}",
            flat.len()
        );
        (0..rows)
            .map(|r| argmax_host_row(&flat[r * vocab..(r + 1) * vocab]))
            .collect()
    }

    fn argmax_last_row(logits: &Tensor) -> Result<u32> {
        let seq = logits.dim(1)?;
        let amax = logits
            .narrow(1, seq - 1, 1)?
            .argmax(D::Minus1)?
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?;
        Ok(amax[0])
    }

    pub struct Qwen38MtpSelfSpecEngine<'a> {
        base: &'a Qwen3Moe,
        mtp: &'a Qwen38DenseMtpHead,
        k: usize,
        stop_ids: Vec<u32>,
    }

    impl<'a> Qwen38MtpSelfSpecEngine<'a> {
        pub fn new(base: &'a Qwen3Moe, mtp: &'a Qwen38DenseMtpHead, k: usize) -> Result<Self> {
            let hidden = base.config().hidden_size;
            assert_mtp_chain_depth_fits_verify_lm_head_rows_ceiling(hidden, k, 0)?;
            let device_reported_rows_ceiling =
                nv_kernels::cuda::gemv_i8_normed_mk_max_m(hidden as i32).max(0) as usize;
            ensure!(
                mtp_verify_rows_per_round(k) <= device_reported_rows_ceiling,
                "this device's lm_head rows-per-call ceiling is {device_reported_rows_ceiling} \
                 for hidden={hidden}; a k={k} mtp round verifies {} rows",
                mtp_verify_rows_per_round(k)
            );
            Ok(Self {
                base,
                mtp,
                k,
                stop_ids: Vec::new(),
            })
        }

        pub fn with_stop_ids(mut self, ids: Vec<u32>) -> Self {
            self.stop_ids = ids;
            self
        }

        pub fn k(&self) -> usize {
            self.k
        }

        pub fn generate_greedy(
            &self,
            prompt: &[u32],
            max_new: usize,
            max_seq: usize,
        ) -> Result<(Vec<u32>, MtpSpecStats)> {
            self.generate_inner(prompt, max_new, max_seq, true)
        }

        pub fn generate_reference(
            &self,
            prompt: &[u32],
            max_new: usize,
            max_seq: usize,
        ) -> Result<(Vec<u32>, MtpSpecStats)> {
            self.generate_inner(prompt, max_new, max_seq, false)
        }

        fn generate_inner(
            &self,
            prompt: &[u32],
            max_new: usize,
            max_seq: usize,
            use_draft: bool,
        ) -> Result<(Vec<u32>, MtpSpecStats)> {
            ensure!(!prompt.is_empty(), "generate: empty prompt");
            let device = self.base.device().clone();

            if use_draft {
                let mut session =
                    Qwen38MtpDecodeSession::start(self.base, self.mtp, self.k, prompt, max_seq)?;
                let anchor = session.anchor();
                let mut generated: Vec<u32> = vec![anchor];
                if self.stop_ids.contains(&anchor) {
                    session.stats.stop_token = Some(anchor);
                    return Ok((generated, session.stats));
                }
                'rounds: while generated.len() < max_new && session.round_fits() {
                    let emitted = session.round()?;
                    for &t in emitted.iter() {
                        generated.push(t);
                        if self.stop_ids.contains(&t) {
                            session.stats.stop_token = Some(t);
                            break 'rounds;
                        }
                        if generated.len() >= max_new {
                            break 'rounds;
                        }
                    }
                }
                return Ok((generated, session.stats));
            }

            let mut cache = self.base.new_kv_cache(max_seq)?;
            let mut stats = MtpSpecStats::default();
            let seq = prompt.len();
            let tokens = Tensor::from_vec(prompt.to_vec(), (1usize, seq), &device)?;
            let pos = Tensor::from_vec((0..seq as i32).collect::<Vec<i32>>(), seq, &device)?;
            let (logits, _hidden) =
                self.base
                    .forward_with_cache_dispatched_hidden(&tokens, &pos, &mut cache, None)?;
            let mut anchor = argmax_last_row(&logits)?;
            drop(logits);

            let mut generated: Vec<u32> = vec![anchor];
            if self.stop_ids.contains(&anchor) {
                stats.stop_token = Some(anchor);
                return Ok((generated, stats));
            }
            let _ = device.synchronize();

            while generated.len() < max_new {
                let round_t0 = std::time::Instant::now();
                let l = cache.current_len();
                ensure!(
                    l + 1 <= max_seq,
                    "reference decode overflow: committed={l} + 1 > max_seq={max_seq}"
                );
                let bt = Tensor::from_vec(vec![anchor], (1usize, 1usize), &device)?;
                let bp = Tensor::from_vec(vec![l as i32], 1usize, &device)?;
                let (vlogits, _vh) =
                    self.base
                        .forward_with_cache_dispatched_hidden(&bt, &bp, &mut cache, None)?;
                anchor = argmax_last_row(&vlogits)?;
                generated.push(anchor);
                stats.rounds += 1;
                stats.emitted += 1;
                stats
                    .round_ms
                    .push(1000.0 * round_t0.elapsed().as_secs_f64());
                if self.stop_ids.contains(&anchor) {
                    stats.stop_token = Some(anchor);
                    break;
                }
            }
            Ok((generated, stats))
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{
    MtpKvCache, Qwen38DenseMtpHead, Qwen38MtpDecodeSession, Qwen38MtpGraphedDecodeSession,
    Qwen38MtpSelfSpecEngine,
};
