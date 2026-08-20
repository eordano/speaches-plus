#![allow(dead_code)]

use std::fmt;

use serde::{Serialize, Serializer};

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct Millis(pub u64);

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct DurationMs(pub u64);

impl Millis {
    pub fn zero() -> Self {
        Self(0)
    }

    pub fn from_session_start(d: DurationMs) -> Self {
        Self(d.0)
    }

    pub fn raw(self) -> u64 {
        self.0
    }

    pub fn saturating_sub_dur(self, d: DurationMs) -> Self {
        Self(self.0.saturating_sub(d.0))
    }
}

impl DurationMs {
    pub fn zero() -> Self {
        Self(0)
    }

    pub fn raw(self) -> u64 {
        self.0
    }

    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }
}

impl fmt::Display for Millis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}ms@T", self.0)
    }
}

impl fmt::Display for DurationMs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}ms", self.0)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct Epoch(pub u64);

impl Epoch {
    pub fn zero() -> Self {
        Self(0)
    }

    pub fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }

    pub fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for Epoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epoch_{}", self.0)
    }
}

macro_rules! id_newtype {
    ($name:ident, $prefix:expr) => {
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        #[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
        pub struct $name(pub String);

        impl $name {
            pub fn new(s: impl Into<String>) -> Self {
                Self(s.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn into_string(self) -> String {
                self.0
            }

            pub fn prefix() -> &'static str {
                $prefix
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(&self.0)
            }
        }

        impl PartialEq<str> for $name {
            fn eq(&self, other: &str) -> bool {
                self.0 == other
            }
        }

        impl PartialEq<&str> for $name {
            fn eq(&self, other: &&str) -> bool {
                self.0 == *other
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_string())
            }
        }
    };
}

id_newtype!(SessionId, "sess_");
id_newtype!(ItemId, "item_");
id_newtype!(ResponseId, "resp_");
id_newtype!(EventId, "evt_");

macro_rules! audio_newtype_f32 {
    ($name:ident, $rate:expr) => {
        #[derive(Debug, Clone, Default)]
        pub struct $name(pub Vec<f32>);

        impl $name {
            pub const SAMPLE_RATE: u32 = $rate;

            pub fn new(samples: Vec<f32>) -> Self {
                Self(samples)
            }

            pub fn samples(&self) -> &[f32] {
                &self.0
            }

            pub fn into_vec(self) -> Vec<f32> {
                self.0
            }

            pub fn len(&self) -> usize {
                self.0.len()
            }

            pub fn is_empty(&self) -> bool {
                self.0.is_empty()
            }

            pub fn duration_ms(&self) -> DurationMs {
                DurationMs((self.0.len() as u64) * 1000 / Self::SAMPLE_RATE as u64)
            }
        }
    };
}

audio_newtype_f32!(MonoF32At16k, 16_000);
audio_newtype_f32!(MonoF32At24k, 24_000);
audio_newtype_f32!(MonoF32At48k, 48_000);

#[derive(Debug, Clone, Default)]
pub struct StereoS16At48k(pub Vec<i16>);

impl StereoS16At48k {
    pub const SAMPLE_RATE: u32 = 48_000;
    pub const CHANNELS: u32 = 2;

    pub fn new(samples: Vec<i16>) -> Self {
        Self(samples)
    }

    pub fn samples(&self) -> &[i16] {
        &self.0
    }

    pub fn into_vec(self) -> Vec<i16> {
        self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_distinct_at_compile_time() {
        let item: ItemId = ItemId::new("item_x");
        let resp: ResponseId = ResponseId::new("resp_y");
        assert_ne!(item.as_str(), resp.as_str());
    }

    #[test]
    fn epoch_next_monotonic() {
        let e = Epoch::zero();
        assert_eq!(e.next().raw(), 1);
        assert_eq!(e.next().next().raw(), 2);
    }

    #[test]
    fn millis_dur_sub() {
        let m = Millis(1000);
        let d = DurationMs(300);
        assert_eq!(m.saturating_sub_dur(d), Millis(700));
        assert_eq!(Millis(100).saturating_sub_dur(DurationMs(500)), Millis(0));
    }

    #[test]
    fn millis_from_session_start() {
        let d = DurationMs(123);
        let m = Millis::from_session_start(d);
        assert_eq!(m.raw(), 123);
    }

    #[test]
    fn audio_buffer_duration_matches_rate() {
        let b16 = MonoF32At16k::new(vec![0.0; 16_000]);
        assert_eq!(b16.duration_ms().raw(), 1000);
        let b24 = MonoF32At24k::new(vec![0.0; 12_000]);
        assert_eq!(b24.duration_ms().raw(), 500);
        let b48 = MonoF32At48k::new(vec![0.0; 48_000]);
        assert_eq!(b48.duration_ms().raw(), 1000);
    }

    #[test]
    fn audio_types_distinct() {
        let _b16 = MonoF32At16k::default();
        let _b24 = MonoF32At24k::default();
        let _b48 = MonoF32At48k::default();
        let _s = StereoS16At48k::default();
    }
}
