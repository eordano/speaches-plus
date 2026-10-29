use anyhow::{bail, Result};
use candle_core::DType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TalkerPrecision {
    F32,
    Bf16,
    Fp8E4m3,
}

impl TalkerPrecision {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "fp32" | "float32" => Ok(Self::F32),
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            "fp8" | "f8" | "e4m3" | "fp8_e4m3" => Ok(Self::Fp8E4m3),
            other => bail!("talker precision {other:?}: expected f32, bf16 or fp8"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
            Self::Fp8E4m3 => "fp8",
        }
    }

    pub fn activation_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Bf16 | Self::Fp8E4m3 => DType::BF16,
        }
    }

    pub fn needs_cuda(self) -> bool {
        self != Self::F32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_three_spellings_and_refuses_the_rest() {
        assert_eq!(TalkerPrecision::parse("F32").unwrap(), TalkerPrecision::F32);
        assert_eq!(TalkerPrecision::parse(" bf16 ").unwrap(), TalkerPrecision::Bf16);
        assert_eq!(TalkerPrecision::parse("fp8").unwrap(), TalkerPrecision::Fp8E4m3);
        assert_eq!(TalkerPrecision::parse("e4m3").unwrap(), TalkerPrecision::Fp8E4m3);
        assert!(TalkerPrecision::parse("int4").is_err());
        assert_eq!(TalkerPrecision::Fp8E4m3.activation_dtype(), DType::BF16);
        assert!(!TalkerPrecision::F32.needs_cuda());
    }
}
