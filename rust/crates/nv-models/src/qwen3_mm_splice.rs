use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use nv_layers::rope::Rope;

pub const QWEN3_5_IMAGE_TOKEN_ID_FROM_THE_RELEASE_CONFIG: u32 = 248056;

pub const QWEN3_5_MROPE_SECTION_FROM_THE_RELEASE_CONFIG: [usize; 3] = [11, 11, 10];

pub fn mrope_section_from_hf_json_str(raw: &str) -> Result<[usize; 3]> {
    let v: serde_json::Value = serde_json::from_str(raw)?;
    let arr = v["text_config"]["rope_parameters"]["mrope_section"]
        .as_array()
        .ok_or_else(|| {
            anyhow::anyhow!("config.json: text_config.rope_parameters.mrope_section missing")
        })?;
    anyhow::ensure!(arr.len() == 3, "mrope_section must have 3 entries");
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
    anyhow::ensure!(
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
    let n = tokens.len();
    let mut t = Vec::with_capacity(n);
    let mut h = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    let mut current_pos = 0u32;
    let mut grid_iter = llm_grids_thw.iter();
    let mut i = 0usize;
    while i < n {
        if tokens[i] == image_token_id {
            let run_start = i;
            while i < n && tokens[i] == image_token_id {
                i += 1;
            }
            let run_len = i - run_start;
            let (gt, gh, gw) = *grid_iter.next().ok_or_else(|| {
                anyhow::anyhow!(
                    "token run of {run_len} image tokens at {run_start} has no llm grid; \
                     pass one (t,h,w) grid per image run"
                )
            })?;
            anyhow::ensure!(
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
    anyhow::ensure!(
        section.iter().sum::<usize>() == half,
        "mrope section {:?} must tile the {half} rotary half-frequencies",
        section
    );
    let tokens = pos.len();
    let table_rows = rope.config().max_seq_len;
    for (axis, v) in [("t", &pos.t), ("h", &pos.h), ("w", &pos.w)] {
        if let Some(mx) = v.iter().max() {
            anyhow::ensure!(
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

pub fn mrope_rope_one_row_per_token(
    rope: &Rope,
    pos: &Qwen3MropePositions,
    section: [usize; 3],
    device: &Device,
) -> Result<Rope> {
    let (cos, sin) = mrope_cos_sin_rows(rope, pos, section, device)?;
    let mut cfg = *rope.config();
    cfg.max_seq_len = pos.len();
    Rope::from_precomputed_tables_one_row_per_token_so_mrope_rides_the_standard_apply(
        cfg, cos, sin,
    )
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
    let (b, seq, hidden) = x.dims3().map_err(|e| anyhow::anyhow!(e))?;
    anyhow::ensure!(b == 1, "splice expects [1, seq, hidden], got batch {b}");
    let mut parts: Vec<Tensor> = Vec::new();
    let mut cursor = 0usize;
    let mut prev_end = 0usize;
    for s in splices {
        let (rows, width) = s.rows.dims2().map_err(|e| anyhow::anyhow!(e))?;
        anyhow::ensure!(
            width == hidden,
            "splice at {} has width {width}, model hidden is {hidden}",
            s.position
        );
        anyhow::ensure!(
            s.position >= prev_end,
            "splices must be sorted and non-overlapping: {} < {prev_end}",
            s.position
        );
        let end = s.position + rows;
        anyhow::ensure!(
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
