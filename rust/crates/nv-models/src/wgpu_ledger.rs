use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::WgpuContext;

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
            let l: Vec<f32> = ::nv_kernels::wgpu_backend::dispatch::read_back(
                self.ctx,
                &self.logits,
                self.vocab,
            )
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
    matches!(
        std::env::var(var).ok().as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}

pub fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    nv_kernels::wgpu_backend::pack::pack_u8_words_padded_to_multiple(bytes, 2)
}

pub struct VramLedger {
    pub ctx: &'static WgpuContext,
    pub buffers: Vec<wgpu::Buffer>,
    pub alloc: std::collections::BTreeMap<String, (usize, u64)>,
    pub alloc_total: u64,
    pub since_flush: u64,
    pub class_prefix: &'static str,
    pub flush_enabled: fn() -> bool,
    pub flush_bytes: u64,
}

impl VramLedger {
    pub fn new(
        ctx: &'static WgpuContext,
        class_prefix: &'static str,
        flush_enabled: fn() -> bool,
        flush_bytes: u64,
    ) -> Self {
        Self {
            ctx,
            buffers: Vec::new(),
            alloc: std::collections::BTreeMap::new(),
            alloc_total: 0,
            since_flush: 0,
            class_prefix,
            flush_enabled,
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
        if (self.flush_enabled)() && self.since_flush >= self.flush_bytes {
            self.flush_staging();
        }
    }

    pub fn store(&mut self, label: &str, b: wgpu::Buffer) -> wgpu::Buffer {
        self.record(label, b.size());
        self.buffers.push(b.clone());
        b
    }

    pub fn zeros(&mut self, label: &str, bytes: u64) -> wgpu::Buffer {
        let b = dispatch::storage_zeroed(self.ctx, label, bytes);
        self.store(label, b)
    }

    pub fn upload_u32(&mut self, label: &str, data: &[u32]) -> wgpu::Buffer {
        let b = dispatch::storage_from_slice(self.ctx, label, data);
        let b = self.store(label, b);
        self.flush_staging_if_due();
        b
    }

    pub fn upload_f32(&mut self, label: &str, data: &[f32]) -> wgpu::Buffer {
        let b = dispatch::storage_from_slice(self.ctx, label, data);
        let b = self.store(label, b);
        self.flush_staging_if_due();
        b
    }

    pub fn uni<T: bytemuck::Pod>(&mut self, label: &str, v: T) -> wgpu::Buffer {
        let b = dispatch::uniform_from(self.ctx, label, &v);
        self.store(label, b)
    }

    pub fn grid1(&self, invocations: u64, wg: u32) -> (u32, u32, u32) {
        dispatch::workgroup_count_1d(self.ctx, invocations, wg)
    }
}
