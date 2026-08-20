use super::{pack_u8, FdP};
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;

pub const SPLITS: u32 = 16;

#[derive(Clone, Copy)]
pub struct Shape {
    pub n_q: u32,
    pub n_kv: u32,
    pub hd: u32,
    pub total: u32,
    pub start: u32,
}

impl Shape {
    pub fn params(&self) -> FdP {
        FdP {
            n_heads: self.n_q,
            n_kv: self.n_kv,
            head_dim: self.hd,
            total: self.total,
            start: self.start,
            splits: SPLITS,
            out_bf16: 1,
            scaling: 1.0 / (self.hd as f32).sqrt(),
            m_rows: 1,
            ..Default::default()
        }
    }

    pub fn scratch_elems(&self) -> usize {
        (self.n_q * SPLITS * (self.hd + 2)) as usize
    }
}

pub struct Inputs {
    pub q: wgpu::Buffer,
    pub k: wgpu::Buffer,
    pub v: wgpu::Buffer,
    pub ks: wgpu::Buffer,
    pub vs: wgpu::Buffer,
    pub p: wgpu::Buffer,
}

pub fn inputs(c: &WgpuContext, s: &Shape, seed: u64) -> Inputs {
    let kv_elems = (s.total.max(1) * s.n_kv * s.hd) as usize;
    let sc_elems = (s.total.max(1) * s.n_kv) as usize;
    let mut lcg = seed | 1;
    let mut next = move || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        lcg
    };
    let kb: Vec<u8> = (0..kv_elems)
        .map(|_| 0x30u8 + ((next() >> 33) as u8 % 0x18))
        .collect();
    let vb: Vec<u8> = (0..kv_elems)
        .map(|_| 0x30u8 + ((next() >> 33) as u8 % 0x18))
        .collect();
    let scales: Vec<f32> = (0..sc_elems)
        .map(|j| 0.004 + ((next() >> 45) as f32 / 524_288.0) * 0.02 + (j % 7) as f32 * 0.003)
        .collect();
    let q: Vec<f32> = (0..(s.n_q * s.hd) as usize)
        .map(|j| ((next() >> 40) as f32 / 8_388_608.0) - 1.0 + (j % 3) as f32 * 0.25)
        .collect();
    Inputs {
        q: dispatch::storage_from_slice(c, "fp-q", &q),
        k: dispatch::storage_from_slice(c, "fp-k", &pack_u8(&kb)),
        v: dispatch::storage_from_slice(c, "fp-v", &pack_u8(&vb)),
        ks: dispatch::storage_from_slice(c, "fp-ks", &scales),
        vs: dispatch::storage_from_slice(c, "fp-vs", &scales),
        p: dispatch::uniform_from(c, "fp-p", &s.params()),
    }
}

pub fn stage1_scratch(
    c: &WgpuContext,
    s: &Shape,
    inp: &Inputs,
    label: &str,
    source: &str,
    entry: &str,
    grid_x: u32,
) -> Vec<u32> {
    let scratch = dispatch::storage_zeroed(c, "fp-scratch", (s.scratch_elems() * 4) as u64);
    dispatch::run(
        c,
        label,
        source,
        entry,
        &[
            (0, &inp.q),
            (4, &inp.p),
            (5, &inp.k),
            (6, &inp.v),
            (7, &scratch),
            (8, &inp.ks),
            (9, &inp.vs),
        ],
        (grid_x, SPLITS, 1),
    )
    .unwrap_or_else(|e| panic!("{label}: dispatch: {e}"));
    dispatch::read_back::<u32>(c, &scratch, s.scratch_elems())
        .unwrap_or_else(|e| panic!("{label}: read_back: {e}"))
}
