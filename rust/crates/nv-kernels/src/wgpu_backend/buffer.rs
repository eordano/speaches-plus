use std::marker::PhantomData;

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dispatch;
use crate::wgpu_backend::{Result, WgpuError};

#[derive(Clone)]
pub struct GpuTensor<T: bytemuck::Pod> {
    buffer: wgpu::Buffer,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> GpuTensor<T> {
    pub fn upload(ctx: &WgpuContext, label: &str, data: &[T]) -> Self {
        Self {
            buffer: dispatch::storage_from_slice(ctx, label, data),
            len: data.len(),
            _marker: PhantomData,
        }
    }

    pub fn zeroed(ctx: &WgpuContext, label: &str, len: usize) -> Self {
        let bytes = (len * std::mem::size_of::<T>()) as u64;
        Self {
            buffer: dispatch::storage_zeroed(ctx, label, bytes),
            len,
            _marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn raw(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn write(&self, ctx: &WgpuContext, data: &[T]) -> Result<()> {
        if data.len() != self.len {
            return Err(WgpuError::Shape(format!(
                "GpuTensor::write: got {} elements, buffer holds {}",
                data.len(),
                self.len
            )));
        }
        if data.is_empty() {
            return Ok(());
        }
        ctx.queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
        Ok(())
    }

    pub fn download(&self, ctx: &WgpuContext) -> Result<Vec<T>> {
        dispatch::read_back(ctx, &self.buffer, self.len)
    }

    pub fn download_range(&self, ctx: &WgpuContext, offset: usize, len: usize) -> Result<Vec<T>> {
        if offset + len > self.len {
            return Err(WgpuError::Shape(format!(
                "GpuTensor::download_range: {offset}+{len} outside buffer of {} elements",
                self.len
            )));
        }
        dispatch::read_back_at(ctx, &self.buffer, offset, len)
    }

    pub fn download_into(&self, ctx: &WgpuContext, out: &mut [T]) -> Result<()> {
        if out.len() != self.len {
            return Err(WgpuError::Shape(format!(
                "GpuTensor::download_into: got {} elements, buffer holds {}",
                out.len(),
                self.len
            )));
        }
        let host = self.download(ctx)?;
        out.copy_from_slice(&host);
        Ok(())
    }
}

pub struct GpuUniform<T: bytemuck::Pod> {
    buffer: wgpu::Buffer,
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> GpuUniform<T> {
    pub fn new(ctx: &WgpuContext, label: &str, value: &T) -> Self {
        Self {
            buffer: dispatch::uniform_from(ctx, label, value),
            _marker: PhantomData,
        }
    }

    pub fn raw(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn write(&self, ctx: &WgpuContext, value: &T) {
        ctx.queue
            .write_buffer(&self.buffer, 0, bytemuck::bytes_of(value));
    }
}

pub trait GpuBind {
    fn bind_buffer(&self) -> &wgpu::Buffer;
}

impl<T: bytemuck::Pod> GpuBind for GpuTensor<T> {
    fn bind_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl<T: bytemuck::Pod> GpuBind for GpuUniform<T> {
    fn bind_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl GpuBind for wgpu::Buffer {
    fn bind_buffer(&self) -> &wgpu::Buffer {
        self
    }
}
