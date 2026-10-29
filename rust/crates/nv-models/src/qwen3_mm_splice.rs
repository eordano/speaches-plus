use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor};
use nv_layers::rope::Rope;

pub const QWEN3_5_MROPE_SECTION_FROM_THE_RELEASE_CONFIG: [usize; 3] = [11, 11, 10];

pub fn mrope_section_from_hf_json_str(raw: &str) -> Result<[usize; 3]> {
    let v: serde_json::Value = serde_json::from_str(raw)?;
    let arr = v["text_config"]["rope_parameters"]["mrope_section"]
        .as_array()
        .ok_or_else(|| {
            anyhow::anyhow!("config.json: text_config.rope_parameters.mrope_section missing")
        })?;
    ensure!(arr.len() == 3, "mrope_section must have 3 entries");
    let mut out = [0usize; 3];
    for (i, x) in arr.iter().enumerate() {
        out[i] = x
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("mrope_section[{i}] not an integer"))?
            as usize;
    }
    let interleaved = v["text_config"]["rope_parameters"]["mrope_interleaved"]
        .as_bool()
        .unwrap_or(false);
    ensure!(
        interleaved,
        "this splice path implements only mrope_interleaved=true (the qwen3_5 release layout)"
    );
    Ok(out)
}

pub fn interleaved_axis_of_half_freq_matching_hf_overwrite_semantics(
    j: usize,
    section: [usize; 3],
) -> usize {
    let h_cap = 3 * section[1];
    let w_cap = 3 * section[2];
    match j % 3 {
        1 if j < h_cap => 1,
        2 if j < w_cap => 2,
        _ => 0,
    }
}

#[derive(Clone, Debug)]
pub struct Qwen3MropePositions {
    pub t: Vec<u32>,
    pub h: Vec<u32>,
    pub w: Vec<u32>,
    pub delta_added_to_token_index_for_every_position_after_this_prefill: i64,
}

impl Qwen3MropePositions {
    pub fn len(&self) -> usize {
        self.t.len()
    }

    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }

    pub fn is_text_degenerate(&self) -> bool {
        self.t == self.h && self.h == self.w
    }

    pub fn decode_position(&self, token_index: usize) -> i64 {
        token_index as i64 + self.delta_added_to_token_index_for_every_position_after_this_prefill
    }
}

pub fn build_mrope_positions_matching_hf_get_rope_index(
    tokens: &[u32],
    image_token_id: u32,
    llm_grids_thw: &[(usize, usize, usize)],
) -> Result<Qwen3MropePositions> {
    build_mrope_positions_for_vision_runs(tokens, &[image_token_id], llm_grids_thw)
}

pub fn build_mrope_positions_for_vision_runs(
    tokens: &[u32],
    vision_token_ids: &[u32],
    llm_grids_thw: &[(usize, usize, usize)],
) -> Result<Qwen3MropePositions> {
    let n = tokens.len();
    let mut t = Vec::with_capacity(n);
    let mut h = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    let mut current_pos = 0u32;
    let mut grid_iter = llm_grids_thw.iter();
    let mut i = 0usize;
    while i < n {
        if vision_token_ids.contains(&tokens[i]) {
            let run_start = i;
            while i < n && tokens[i] == tokens[run_start] {
                i += 1;
            }
            let run_len = i - run_start;
            let (gt, gh, gw) = *grid_iter.next().ok_or_else(|| {
                anyhow::anyhow!(
                    "token run of {run_len} image tokens at {run_start} has no llm grid; \
                     pass one (t,h,w) grid per image run"
                )
            })?;
            ensure!(
                gt * gh * gw == run_len,
                "image run at {run_start} has {run_len} tokens but grid ({gt},{gh},{gw}) \
                 covers {}",
                gt * gh * gw
            );
            for ti in 0..gt {
                let _ = ti;
                for hh in 0..gh {
                    for ww in 0..gw {
                        t.push(current_pos);
                        h.push(current_pos + hh as u32);
                        w.push(current_pos + ww as u32);
                    }
                }
            }
            current_pos += gh.max(gw) as u32;
        } else {
            t.push(current_pos);
            h.push(current_pos);
            w.push(current_pos);
            current_pos += 1;
            i += 1;
        }
    }
    if grid_iter.next().is_some() {
        anyhow::bail!(
            "more llm grids than image-token runs: {} grids for the token sequence",
            llm_grids_thw.len()
        );
    }
    let max_pos = t
        .iter()
        .chain(h.iter())
        .chain(w.iter())
        .copied()
        .max()
        .map(|v| v as i64)
        .unwrap_or(-1);
    Ok(Qwen3MropePositions {
        t,
        h,
        w,
        delta_added_to_token_index_for_every_position_after_this_prefill: max_pos + 1 - n as i64,
    })
}

pub fn mrope_cos_sin_rows(
    rope: &Rope,
    pos: &Qwen3MropePositions,
    section: [usize; 3],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = rope.config().head_dim / 2;
    ensure!(
        section.iter().sum::<usize>() == half,
        "mrope section {:?} must tile the {half} rotary half-frequencies",
        section
    );
    let tokens = pos.len();
    let table_rows = rope.config().max_seq_len;
    for (axis, v) in [("t", &pos.t), ("h", &pos.h), ("w", &pos.w)] {
        if let Some(mx) = v.iter().max() {
            ensure!(
                (*mx as usize) < table_rows,
                "mrope {axis} position {mx} exceeds the rope table ({table_rows} rows)"
            );
        }
    }
    let gather = |v: &Vec<u32>| -> Result<(Tensor, Tensor)> {
        let idx = Tensor::from_vec(v.clone(), tokens, device)?;
        Ok((
            rope.cos().index_select(&idx, 0)?.to_dtype(DType::F32)?,
            rope.sin().index_select(&idx, 0)?.to_dtype(DType::F32)?,
        ))
    };
    let (cos_t, sin_t) = gather(&pos.t)?;
    let (cos_h, sin_h) = gather(&pos.h)?;
    let (cos_w, sin_w) = gather(&pos.w)?;

    let mut mask = [vec![0f32; half], vec![0f32; half], vec![0f32; half]];
    for (j, m) in (0..half).map(|j| {
        (
            j,
            interleaved_axis_of_half_freq_matching_hf_overwrite_semantics(j, section),
        )
    }) {
        mask[m][j] = 1.0;
    }
    let mt = Tensor::from_vec(mask[0].clone(), (1, half), device)?;
    let mh = Tensor::from_vec(mask[1].clone(), (1, half), device)?;
    let mw = Tensor::from_vec(mask[2].clone(), (1, half), device)?;
    let cos = cos_t
        .broadcast_mul(&mt)?
        .add(&cos_h.broadcast_mul(&mh)?)?
        .add(&cos_w.broadcast_mul(&mw)?)?;
    let sin = sin_t
        .broadcast_mul(&mt)?
        .add(&sin_h.broadcast_mul(&mh)?)?
        .add(&sin_w.broadcast_mul(&mw)?)?;
    Ok((cos.contiguous()?, sin.contiguous()?))
}

pub fn mrope_state_table_head_and_tail_start(
    rope: &Rope,
    pos: &Qwen3MropePositions,
    section: [usize; 3],
    rows: usize,
) -> Result<(Tensor, Tensor, usize)> {
    let n = pos.len();
    let tail_start = (n as i64 + pos.delta_added_to_token_index_for_every_position_after_this_prefill) as usize;
    ensure!(n <= rows, "a {n}-token prompt does not fit a {rows}-row rope table");
    ensure!(
        tail_start + (rows - n) <= rope.config().max_seq_len,
        "continuation rows {tail_start}..{} exceed the {}-row base rope table",
        tail_start + rows - n,
        rope.config().max_seq_len
    );
    let (cos, sin) = mrope_cos_sin_rows(rope, pos, section, rope.cos().device())?;
    Ok((cos, sin, tail_start))
}

pub const VISION_PREFIX_KEY_IDS_SET_THE_HIGH_BIT_SO_NO_VOCABULARY_ID_EVER_EQUALS_ONE: u32 = 0x8000_0000;

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

pub fn prefix_key_with_vision_content(
    tokens: &[u32],
    vision_token_ids: &[u32],
    run_content_hashes: &[u64],
) -> Result<Vec<u32>> {
    let mut out = tokens.to_vec();
    let mut runs = run_content_hashes.iter();
    let mut i = 0usize;
    while i < out.len() {
        if !vision_token_ids.contains(&tokens[i]) {
            i += 1;
            continue;
        }
        let h = *runs
            .next()
            .ok_or_else(|| anyhow::anyhow!("vision run at {i} has no content hash"))?;
        let start = i;
        while i < out.len() && tokens[i] == tokens[start] {
            let j = (i - start) as u64;
            out[i] = VISION_PREFIX_KEY_IDS_SET_THE_HIGH_BIT_SO_NO_VOCABULARY_ID_EVER_EQUALS_ONE
                | (splitmix64(h ^ j.wrapping_mul(0x9e37_79b9_7f4a_7c15)) as u32 & 0x7fff_ffff);
            i += 1;
        }
    }
    ensure!(
        runs.next().is_none(),
        "{} content hashes for fewer vision runs",
        run_content_hashes.len()
    );
    Ok(out)
}

pub fn mrope_rope_one_row_per_token(
    rope: &Rope,
    pos: &Qwen3MropePositions,
    section: [usize; 3],
    device: &Device,
) -> Result<Rope> {
    let (cos, sin) = mrope_cos_sin_rows(rope, pos, section, device)?;
    let mut cfg = *rope.config();
    cfg.max_seq_len = pos.len();
    Rope::from_precomputed_tables_one_row_per_token_so_mrope_rides_the_standard_apply(cfg, cos, sin)
}

pub struct Qwen3ImageRowSplice {
    pub position: usize,
    pub rows: Tensor,
}

pub fn splice_image_rows_into_embedded(
    x: &Tensor,
    splices: &[Qwen3ImageRowSplice],
) -> Result<Tensor> {
    if splices.is_empty() {
        return Ok(x.clone());
    }
    let (b, seq, hidden) = x.dims3()?;
    ensure!(b == 1, "splice expects [1, seq, hidden], got batch {b}");
    let mut parts: Vec<Tensor> = Vec::new();
    let mut cursor = 0usize;
    let mut prev_end = 0usize;
    for s in splices {
        let (rows, width) = s.rows.dims2()?;
        ensure!(
            width == hidden,
            "splice at {} has width {width}, model hidden is {hidden}",
            s.position
        );
        ensure!(
            s.position >= prev_end,
            "splices must be sorted and non-overlapping: {} < {prev_end}",
            s.position
        );
        let end = s.position + rows;
        ensure!(
            end <= seq,
            "splice at {} with {rows} rows overruns seq {seq}",
            s.position
        );
        if s.position > cursor {
            parts.push(x.narrow(1, cursor, s.position - cursor)?);
        }
        parts.push(s.rows.to_dtype(x.dtype())?.unsqueeze(0)?);
        cursor = end;
        prev_end = end;
    }
    if cursor < seq {
        parts.push(x.narrow(1, cursor, seq - cursor)?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Ok(Tensor::cat(&refs, 1)?.contiguous()?)
}

#[cfg(test)]
mod vision_run_tests {
    use super::*;
    use nv_layers::rope::{RopeConfig, RopeKind};

    const IMG: u32 = 248056;
    const VID: u32 = 248057;

    fn rope(rows: usize) -> Rope {
        Rope::new(
            RopeConfig { head_dim: 64, max_seq_len: rows, base: 10_000_000.0, kind: RopeKind::Standard },
            &Device::Cpu,
        )
        .unwrap()
    }

    #[test]
    fn positions_follow_hf_get_rope_index_for_an_image_then_a_video_frame_group() {
        let mut toks = vec![1, 2, 3];
        toks.extend([IMG; 6]);
        toks.extend([4, 5]);
        toks.extend([VID; 4]);
        toks.push(6);
        let p = build_mrope_positions_for_vision_runs(&toks, &[IMG, VID], &[(1, 2, 3), (1, 2, 2)]).unwrap();
        assert_eq!(p.t, [0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7, 8, 8, 8, 8, 10]);
        assert_eq!(p.h, [0, 1, 2, 3, 3, 3, 4, 4, 4, 6, 7, 8, 8, 9, 9, 10]);
        assert_eq!(p.w, [0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 8, 9, 10]);
        assert_eq!(p.delta_added_to_token_index_for_every_position_after_this_prefill, 11 - 16);
        assert_eq!(p.decode_position(16), 11);
    }

    #[test]
    fn a_text_only_state_table_reproduces_the_base_rows_bit_for_bit() {
        let r = rope(64);
        let toks: Vec<u32> = (0..10).collect();
        let p = build_mrope_positions_for_vision_runs(&toks, &[IMG], &[]).unwrap();
        let (cos, sin, tail) = mrope_state_table_head_and_tail_start(&r, &p, [11, 11, 10], 32).unwrap();
        assert_eq!(tail, 10);
        let want_c: Vec<f32> = r.cos().narrow(0, 0, 10).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let want_s: Vec<f32> = r.sin().narrow(0, 0, 10).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(cos.flatten_all().unwrap().to_vec1::<f32>().unwrap(), want_c);
        assert_eq!(sin.flatten_all().unwrap().to_vec1::<f32>().unwrap(), want_s);
        image_rows_take_h_and_w_frequencies_interleaved_and_the_tail_restarts_at_max_plus_one();
    }

    fn image_rows_take_h_and_w_frequencies_interleaved_and_the_tail_restarts_at_max_plus_one() {
        let r = rope(64);
        let mut toks = vec![7, 8];
        toks.extend([IMG; 4]);
        toks.push(9);
        let p = build_mrope_positions_for_vision_runs(&toks, &[IMG], &[(1, 2, 2)]).unwrap();
        let (cos, _, tail) = mrope_state_table_head_and_tail_start(&r, &p, [11, 11, 10], 16).unwrap();
        assert_eq!(tail, 5, "positions 0,1 | 2..3 grid | 4 then the continuation starts at 5");
        let c: Vec<Vec<f32>> = cos.to_vec2().unwrap();
        let base: Vec<Vec<f32>> = r.cos().to_vec2().unwrap();
        let last_img = 5;
        for j in 0..32 {
            let want = match interleaved_axis_of_half_freq_matching_hf_overwrite_semantics(j, [11, 11, 10]) {
                0 => base[2][j],
                _ => base[3][j],
            };
            assert_eq!(c[last_img][j], want, "freq {j}");
        }
        assert_eq!(c[6], base[4]);
    }

    #[test]
    fn prefix_keys_change_only_vision_tokens_and_separate_different_images() {
        let mut toks = vec![10, 11];
        toks.extend([IMG; 3]);
        toks.push(12);
        let a = prefix_key_with_vision_content(&toks, &[IMG], &[1]).unwrap();
        let b = prefix_key_with_vision_content(&toks, &[IMG], &[2]).unwrap();
        let a2 = prefix_key_with_vision_content(&toks, &[IMG], &[1]).unwrap();
        assert_eq!(a, a2);
        assert_eq!((a[0], a[1], a[5]), (10, 11, 12));
        assert!(a[2..5].iter().all(|k| k & VISION_PREFIX_KEY_IDS_SET_THE_HIGH_BIT_SO_NO_VOCABULARY_ID_EVER_EQUALS_ONE != 0));
        assert!(a[2..5].iter().zip(&b[2..5]).all(|(x, y)| x != y));
        assert_ne!(a[2], a[3], "tokens inside one run carry distinct keys");
        assert!(prefix_key_with_vision_content(&toks, &[IMG], &[]).is_err());
        assert!(prefix_key_with_vision_content(&toks, &[IMG], &[1, 2]).is_err());
    }

    #[test]
    #[ignore]
    fn positions_and_rope_rows_match_the_hf_reference_dump() {
        let Ok(dir) = std::env::var("NV_Q4EXP_VISION_REF") else {
            panic!("set NV_Q4EXP_VISION_REF=<dir with meta.json and ref.safetensors from the HF reference script>");
        };
        let dir = std::path::PathBuf::from(dir);
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
        let lay = &meta["layout"];
        let ids: Vec<u32> = lay["input_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let refs = candle_core::safetensors::load(dir.join("ref.safetensors"), &Device::Cpu).unwrap();
        let grids: Vec<(usize, usize, usize)> = lay["images"]
            .as_array()
            .unwrap()
            .iter()
            .map(|k| {
                let g: Vec<i64> = refs[&format!("grid.{}", k.as_str().unwrap())].to_vec1().unwrap();
                (g[0] as usize, g[1] as usize / 2, g[2] as usize / 2)
            })
            .collect();
        let p = build_mrope_positions_for_vision_runs(&ids, &[IMG, VID], &grids).unwrap();
        let hf: Vec<Vec<u32>> = lay["position_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect())
            .collect();
        assert_eq!((p.t.clone(), p.h.clone(), p.w.clone()), (hf[0].clone(), hf[1].clone(), hf[2].clone()));
        assert_eq!(p.delta_added_to_token_index_for_every_position_after_this_prefill, lay["rope_delta"].as_i64().unwrap());
        let r = rope(4096);
        let (cos, sin, _) = mrope_state_table_head_and_tail_start(&r, &p, QWEN3_5_MROPE_SECTION_FROM_THE_RELEASE_CONFIG, ids.len()).unwrap();
        for (ours, name) in [(cos, "layout.cos"), (sin, "layout.sin")] {
            let theirs = refs[name].narrow(1, 0, 32).unwrap();
            let d = (ours - theirs).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            println!("{name}: {} tokens, {} vision runs, max |ours - hf| = {d:.3e}", ids.len(), grids.len());
            assert!(d < 1e-4, "{name} differs from the HF rotary embedding by {d}");
        }
    }
}
