#[derive(Clone, Debug)]
pub struct IntegratedVerdict {
    pub p_eot: f32,
    pub p_eager_eot: f32,
    pub transcript_so_far: String,
}

pub trait IntegratedEouBackend: Send + Sync {
    fn step(&self, audio_ms_so_far: u64) -> Option<IntegratedVerdict>;
    fn reset(&self);
}

pub struct FakeIntegratedBackend {
    schedule: Vec<(u64, IntegratedVerdict)>,
    cursor: std::sync::Mutex<usize>,
}

impl FakeIntegratedBackend {
    pub fn new(schedule: Vec<(u64, IntegratedVerdict)>) -> Self {
        let mut s = schedule;
        s.sort_by_key(|(t, _)| *t);
        Self {
            schedule: s,
            cursor: std::sync::Mutex::new(0),
        }
    }

    pub fn smoke_default() -> Self {
        Self::new(vec![
            (
                500,
                IntegratedVerdict {
                    p_eot: 0.1,
                    p_eager_eot: 0.2,
                    transcript_so_far: "hi".into(),
                },
            ),
            (
                1500,
                IntegratedVerdict {
                    p_eot: 0.3,
                    p_eager_eot: 0.6,
                    transcript_so_far: "hi there".into(),
                },
            ),
            (
                2500,
                IntegratedVerdict {
                    p_eot: 0.85,
                    p_eager_eot: 0.9,
                    transcript_so_far: "hi there friend".into(),
                },
            ),
        ])
    }
}

impl IntegratedEouBackend for FakeIntegratedBackend {
    fn step(&self, audio_ms_so_far: u64) -> Option<IntegratedVerdict> {
        let mut cur = self.cursor.lock().expect("fake integrated cursor poisoned");
        let mut emit: Option<IntegratedVerdict> = None;
        while *cur < self.schedule.len() && self.schedule[*cur].0 <= audio_ms_so_far {
            emit = Some(self.schedule[*cur].1.clone());
            *cur += 1;
        }
        emit
    }

    fn reset(&self) {
        *self.cursor.lock().expect("fake integrated cursor poisoned") = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_integrated_emits_in_order() {
        let b = FakeIntegratedBackend::smoke_default();
        assert!(b.step(0).is_none());
        let v1 = b.step(800).expect("v1");
        assert!(v1.p_eot < 0.5);
        let v2 = b.step(2700).expect("v2");
        assert!(v2.p_eot >= 0.8);
        assert!(b.step(2700).is_none());
        b.reset();
        assert!(b.step(800).is_some());
    }
}
