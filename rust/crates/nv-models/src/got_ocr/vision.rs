use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use nv_weights::{TensorSource, WeightLoader};

use crate::deepseek_ocr::compressor::layer_norm_2d;
use crate::deepseek_ocr::linear;
use crate::deepseek_ocr::sam::{SamConfig, SamEncoder};

pub fn sam_checkpoint_name(n: &str) -> String {
    let mapped = n
        .replacen("patch_embed.proj.", "patch_embed.projection.", 1)
        .replacen("blocks.", "layers.", 1)
        .replace(".norm1.", ".layer_norm1.")
        .replace(".norm2.", ".layer_norm2.");
    format!("vision_tower.{mapped}")
}

pub struct GotSamNames<'a>(pub &'a WeightLoader);

impl TensorSource for GotSamNames<'_> {
    fn get(&self, name: &str, dtype: DType) -> Result<Tensor> {
        self.0.get(&sam_checkpoint_name(name), dtype)
    }
    fn has(&self, name: &str) -> bool {
        self.0.has(&sam_checkpoint_name(name))
    }
}

pub struct GotLmNames<'a>(pub &'a WeightLoader);

impl TensorSource for GotLmNames<'_> {
    fn get(&self, name: &str, dtype: DType) -> Result<Tensor> {
        self.0.get(&format!("language_model.{name}"), dtype)
    }
    fn has(&self, name: &str) -> bool {
        self.0.has(&format!("language_model.{name}"))
    }
}

pub struct GotVision {
    sam: SamEncoder,
    neck_conv1: Tensor,
    neck_ln1_w: Tensor,
    neck_ln1_b: Tensor,
    neck_conv2: Tensor,
    neck_ln2_w: Tensor,
    neck_ln2_b: Tensor,
    up1: Tensor,
    up2: Tensor,
    proj_w: Tensor,
    proj_b: Tensor,
    ln_eps: f64,
}

impl GotVision {
    pub fn from_loader(weights: &WeightLoader, dtype: DType) -> Result<Self> {
        let cfg = SamConfig::vit_b();
        let ln_eps = cfg.ln_eps;
        let sam = SamEncoder::from_loader(&GotSamNames(weights), "", cfg, dtype)?;
        let get = |name: &str, want: &[usize]| -> Result<Tensor> {
            let t = weights
                .get(name, dtype)
                .with_context(|| format!("load {name}"))?;
            anyhow::ensure!(
                t.dims() == want,
                "{name}: expected shape {want:?}, got {:?}",
                t.dims()
            );
            Ok(t)
        };
        Ok(Self {
            neck_conv1: get("vision_tower.neck.conv1.weight", &[256, 768, 1, 1])?,
            neck_ln1_w: get("vision_tower.neck.layer_norm1.weight", &[256])?,
            neck_ln1_b: get("vision_tower.neck.layer_norm1.bias", &[256])?,
            neck_conv2: get("vision_tower.neck.conv2.weight", &[256, 256, 3, 3])?,
            neck_ln2_w: get("vision_tower.neck.layer_norm2.weight", &[256])?,
            neck_ln2_b: get("vision_tower.neck.layer_norm2.bias", &[256])?,
            up1: get(
                "multi_modal_projector.conv_upsampler1.weight",
                &[512, 256, 3, 3],
            )?,
            up2: get(
                "multi_modal_projector.conv_upsampler2.weight",
                &[1024, 512, 3, 3],
            )?,
            proj_w: get(
                "multi_modal_projector.multimodal_projector.weight",
                &[1024, 1024],
            )?,
            proj_b: get(
                "multi_modal_projector.multimodal_projector.bias",
                &[1024],
            )?,
            sam,
            ln_eps,
        })
    }

    pub fn forward(&self, pixels: &Tensor) -> Result<Tensor> {
        let x = self.sam.forward(pixels)?;
        let x = x.permute((0, 3, 1, 2))?.contiguous()?;
        let x = x.conv2d(&self.neck_conv1, 0, 1, 1, 1)?;
        let x = layer_norm_2d(&x, &self.neck_ln1_w, &self.neck_ln1_b, self.ln_eps)?;
        let x = x.conv2d(&self.neck_conv2, 1, 1, 1, 1)?;
        let x = layer_norm_2d(&x, &self.neck_ln2_w, &self.neck_ln2_b, self.ln_eps)?;
        let x = x.conv2d(&self.up1, 1, 2, 1, 1)?;
        let x = x.conv2d(&self.up2, 1, 2, 1, 1)?;
        let (b, c, h, w) = x.dims4()?;
        anyhow::ensure!(
            b == 1 && h * w == 256,
            "got vision projector produced [{b},{c},{h},{w}], expected b=1 h*w=256"
        );
        let x = x.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()?;
        let x = linear(&x, &self.proj_w, Some(&self.proj_b))?;
        Ok(x.reshape((h * w, c))?)
    }
}
