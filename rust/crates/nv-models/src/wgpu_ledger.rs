use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::WgpuContext;
use std::env;
use wgpu::{Buffer, BufferUsages};

#[macro_export]
macro_rules! dispatch_pass {
    ($pass:ident, $p:ident) => {{
        $pass.set_pipeline(&$p.pipeline);
        $pass.set_bind_group(0, &$p.bind, &[]);
        $pass.dispatch_workgroups($p.grid.0, $p.grid.1, $p.grid.2);
    }};
}

#[macro_export]
macro_rules! submit_passes {
    ($ctx:expr, $passes:expr) => {{
        let mut enc = $ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for p in $passes {
                $crate::dispatch_pass!(pass, p);
            }
        }
        $ctx.queue.submit([enc.finish()]);
    }};
}

#[macro_export]
macro_rules! pod {
    ($($(#[$m:meta])* $v:vis struct $n:ident { $($fv:vis $f:ident: $t:ty),* $(,)? })*) => {$(
        $(#[$m])*
        #[repr(C)]
        #[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
        $v struct $n { $($fv $f: $t),* }
    )*};
}

#[macro_export]
macro_rules! wgpu_step_readback_api {
    () => {
        pub fn decode_step(&mut self, token: u32) -> ::anyhow::Result<u32> {
            self.step_inner(token, true)?;
            let t: Vec<u32> =
                ::nv_kernels::wgpu_backend::dispatch::read_back(self.ctx, &self.token_out, 1)
                    .map_err(|e| ::anyhow::anyhow!("{e}"))?;
            Ok(t[0])
        }

        pub fn decode_step_logits(&mut self, token: u32) -> ::anyhow::Result<(u32, Vec<f32>)> {
            self.step_inner(token, true)?;
            let t: Vec<u32> =
                ::nv_kernels::wgpu_backend::dispatch::read_back(self.ctx, &self.token_out, 1)
                    .map_err(|e| ::anyhow::anyhow!("{e}"))?;
            let l: Vec<f32> =
                ::nv_kernels::wgpu_backend::dispatch::read_back(self.ctx, &self.logits, self.vocab)
                    .map_err(|e| ::anyhow::anyhow!("{e}"))?;
            Ok((t[0], l))
        }

        pub fn prefill_step(&mut self, token: u32) -> ::anyhow::Result<()> {
            self.step_inner(token, false)
        }
    };
}

#[derive(Clone, Debug, Default)]
pub struct VramReport {
    pub buffers: usize,
    pub total_bytes: u64,
    pub by_class: Vec<(String, usize, u64)>,
}

impl VramReport {
    pub fn render(&self) -> String {
        let mut out = format!(
            "wgpu buffers: {} allocations, {:.3} GiB total\n",
            self.buffers,
            self.total_bytes as f64 / (1u64 << 30) as f64
        );
        for (class, count, bytes) in &self.by_class {
            out.push_str(&format!(
                "  {class:<22} {count:>7} bufs  {:>10.3} GiB\n",
                *bytes as f64 / (1u64 << 30) as f64
            ));
        }
        out
    }
}

pub fn vram_report_var_enabled(var: &str) -> bool {
    env_on(var)
}

pub fn env_on(name: &str) -> bool {
    matches!(
        env::var(name).ok().as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}

pub fn env_not_off(name: &str) -> bool {
    !matches!(
        env::var(name).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub fn env_flag(name: &str) -> Option<bool> {
    env::var(name).ok().map(|v| {
        let t = v.trim();
        !t.is_empty() && t != "0"
    })
}

pub fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    nv_kernels::wgpu_backend::pack::pack_u8_words_padded_to_multiple(bytes, 2)
}

pub struct VramLedger {
    pub ctx: &'static WgpuContext,
    pub buffers: Vec<Buffer>,
    pub alloc: std::collections::BTreeMap<String, (usize, u64)>,
    pub alloc_total: u64,
    pub since_flush: u64,
    pub class_prefix: &'static str,
    pub flush_bytes: u64,
}

impl VramLedger {
    pub fn new(
        ctx: &'static WgpuContext,
        class_prefix: &'static str,
        flush_bytes: u64,
    ) -> Self {
        Self {
            ctx,
            buffers: Vec::new(),
            alloc: std::collections::BTreeMap::new(),
            alloc_total: 0,
            since_flush: 0,
            class_prefix,
            flush_bytes,
        }
    }

    pub fn vram_class<'a>(&self, label: &'a str) -> &'a str {
        label.strip_prefix(self.class_prefix).unwrap_or(label)
    }

    pub fn record(&mut self, label: &str, bytes: u64) {
        let class = self.vram_class(label).to_string();
        let e = self.alloc.entry(class).or_insert((0, 0));
        e.0 += 1;
        e.1 += bytes;
        self.alloc_total += bytes;
        self.since_flush += bytes;
    }

    pub fn report(&self) -> VramReport {
        let mut by_class: Vec<(String, usize, u64)> = self
            .alloc
            .iter()
            .map(|(k, (c, b))| (k.clone(), *c, *b))
            .collect();
        by_class.sort_by_key(|e| std::cmp::Reverse(e.2));
        VramReport {
            buffers: self.buffers.len(),
            total_bytes: self.alloc_total,
            by_class,
        }
    }

    pub fn flush_staging(&mut self) {
        if self.since_flush == 0 {
            return;
        }
        self.since_flush = 0;
        self.ctx.queue.submit(std::iter::empty());
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
    }

    pub fn flush_staging_if_due(&mut self) {
        if self.since_flush >= self.flush_bytes {
            self.flush_staging();
        }
    }

    pub fn store(&mut self, label: &str, b: Buffer) -> Buffer {
        self.record(label, b.size());
        self.buffers.push(b.clone());
        b
    }

    pub fn zeros(&mut self, label: &str, bytes: u64) -> Buffer {
        let b = dispatch::storage_zeroed(self.ctx, label, bytes);
        self.store(label, b)
    }

    pub fn upload_u32(&mut self, label: &str, data: &[u32]) -> Buffer {
        let b = dispatch::storage_from_slice(self.ctx, label, data);
        let b = self.store(label, b);
        self.flush_staging_if_due();
        b
    }

    pub fn upload_f32(&mut self, label: &str, data: &[f32]) -> Buffer {
        let b = dispatch::storage_from_slice(self.ctx, label, data);
        let b = self.store(label, b);
        self.flush_staging_if_due();
        b
    }

    pub fn uni<T: bytemuck::Pod>(&mut self, label: &str, v: T) -> Buffer {
        let b = dispatch::uniform_from(self.ctx, label, &v);
        self.store(label, b)
    }

    pub fn grid1(&self, invocations: u64, wg: u32) -> (u32, u32, u32) {
        dispatch::workgroup_count_1d(self.ctx, invocations, wg)
    }
}

pub fn wgsl_binding_is_storage_read_write_a_read_only_binding_can_never_raise_a_hazard(
    source: &str,
    binding: u32,
) -> bool {
    let needle = format!("@binding({binding})");
    source
        .match_indices(&needle)
        .filter_map(|(i, _)| source[i..].split(';').next())
        .find(|decl| decl.contains("var<"))
        .is_none_or(|decl| decl.contains("read_write"))
}

pub fn hazard_tracked_storage_ids<'a>(
    registry: &mut Vec<Buffer>,
    source: &str,
    binds: impl Iterator<Item = (u32, &'a Buffer)>,
) -> Vec<(u32, bool)> {
    let mut ids: Vec<(u32, bool)> = binds
        .filter(|(_, buf)| buf.usage().contains(BufferUsages::STORAGE))
        .map(|(slot, buf)| {
            let id = match registry.iter().position(|b| b == buf) {
                Some(i) => i,
                None => {
                    registry.push(buf.clone());
                    registry.len() - 1
                }
            };
            (
                id as u32,
                wgsl_binding_is_storage_read_write_a_read_only_binding_can_never_raise_a_hazard(
                    source, slot,
                ),
            )
        })
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

pub struct ChunkRowSplice<'a> {
    pub rel_pos: usize,
    pub row_words: &'a [u32],
}

pub fn spliced_chunk<'a>(
    tokens: &[u32],
    splices: &[crate::embed_row_splice::EmbedRowSplice],
    packed: &'a [Vec<u32>],
    hidden: usize,
    m: usize,
    done: usize,
    live: usize,
) -> (Vec<u32>, Vec<ChunkRowSplice<'a>>) {
    let mut chunk: Vec<u32> = tokens[done..done + live].to_vec();
    let pad = *chunk.last().expect("non-empty chunk");
    chunk.resize(m, pad);
    let hidden_words = hidden / 2;
    let mut rows = Vec::new();
    for (si, sp) in splices.iter().enumerate() {
        let n_slots = sp.rows_bf16.len() / hidden;
        let lo = sp.position.max(done);
        let hi = (sp.position + n_slots).min(done + live);
        for abs in lo..hi {
            let w0 = (abs - sp.position) * hidden_words;
            rows.push(ChunkRowSplice {
                rel_pos: abs - done,
                row_words: &packed[si][w0..w0 + hidden_words],
            });
        }
    }
    (chunk, rows)
}
