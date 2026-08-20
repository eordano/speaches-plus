pub const EMBED_ROW_SPLICE_IS_DECODER_NEUTRAL_SO_THE_SERVING_SEAM_HOLDS_ONE_TYPE_FOR_EVERY_WGPU_KIND: &str = "every wgpu decoder that accepts multimodal rows takes the same payload: a run of bf16 hidden rows that REPLACES the gathered token rows starting at `position`, after the embedding scale and before the first layer. gemma4 calls them embed rows and qwen3 called them image rows; forking the struct per kind would force the chat_engine_wgpu seam to clone whole megabyte row runs to cross families, so both families re-export this one struct.";

pub struct EmbedRowSplice {
    pub position: usize,
    pub rows_bf16: Vec<u16>,
}

pub fn bf16_bits_round_nearest_even(x: f32) -> u16 {
    let bits = x.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return 0x7fc0;
    }
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

pub fn rows_to_bf16(rows: &[f32]) -> Vec<u16> {
    rows.iter()
        .copied()
        .map(bf16_bits_round_nearest_even)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounding_is_nearest_even_and_keeps_nan_quiet() {
        assert_eq!(bf16_bits_round_nearest_even(1.0), 0x3f80);
        assert_eq!(bf16_bits_round_nearest_even(-1.0), 0xbf80);
        assert_eq!(bf16_bits_round_nearest_even(0.0), 0x0000);
        assert_eq!(
            bf16_bits_round_nearest_even(f32::from_bits(0x3f80_8000)),
            0x3f80,
            "an exact tie rounds to the even mantissa"
        );
        assert_eq!(
            bf16_bits_round_nearest_even(f32::from_bits(0x3f81_8000)),
            0x3f82,
            "an exact tie rounds to the even mantissa"
        );
        assert_eq!(bf16_bits_round_nearest_even(f32::NAN), 0x7fc0);
        assert_eq!(rows_to_bf16(&[1.0, -1.0]), vec![0x3f80, 0xbf80]);
    }
}
