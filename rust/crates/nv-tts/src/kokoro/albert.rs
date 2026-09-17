use anyhow::Result;
use candle_core::{Module, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, Embedding, LayerNorm, LayerNormConfig, Linear, VarBuilder,
};

#[derive(Clone, Debug)]
pub struct AlbertConfig {
    pub vocab_size: usize,
    pub embedding_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub layer_norm_eps: f64,
}

impl Default for AlbertConfig {
    fn default() -> Self {
        Self {
            vocab_size: 178,
            embedding_size: 128,
            hidden_size: 768,
            num_attention_heads: 12,
            intermediate_size: 2048,
            max_position_embeddings: 512,
            num_hidden_layers: 12,
            layer_norm_eps: 1e-12,
        }
    }
}

pub struct Albert {
    cfg: AlbertConfig,
    word: Embedding,
    position: Embedding,
    token_type: Embedding,
    emb_ln: LayerNorm,
    mapping_in: Linear,
    query: Linear,
    key: Linear,
    value: Linear,
    dense: Linear,
    attn_ln: LayerNorm,
    ffn: Linear,
    ffn_out: Linear,
    full_ln: LayerNorm,
}

impl Albert {
    pub fn load(vb: VarBuilder, cfg: AlbertConfig) -> Result<Self> {
        let ln = LayerNormConfig {
            eps: cfg.layer_norm_eps,
            ..Default::default()
        };
        let e = vb.pp("embeddings");
        let l = vb.pp("encoder.albert_layer_groups.0.albert_layers.0");
        Ok(Self {
            word: embedding(cfg.vocab_size, cfg.embedding_size, e.pp("word_embeddings"))?,
            position: embedding(
                cfg.max_position_embeddings,
                cfg.embedding_size,
                e.pp("position_embeddings"),
            )?,
            token_type: embedding(2, cfg.embedding_size, e.pp("token_type_embeddings"))?,
            emb_ln: layer_norm(cfg.embedding_size, ln, e.pp("LayerNorm"))?,
            mapping_in: linear(
                cfg.embedding_size,
                cfg.hidden_size,
                vb.pp("encoder.embedding_hidden_mapping_in"),
            )?,
            query: linear(cfg.hidden_size, cfg.hidden_size, l.pp("attention.query"))?,
            key: linear(cfg.hidden_size, cfg.hidden_size, l.pp("attention.key"))?,
            value: linear(cfg.hidden_size, cfg.hidden_size, l.pp("attention.value"))?,
            dense: linear(cfg.hidden_size, cfg.hidden_size, l.pp("attention.dense"))?,
            attn_ln: layer_norm(cfg.hidden_size, ln, l.pp("attention.LayerNorm"))?,
            ffn: linear(cfg.hidden_size, cfg.intermediate_size, l.pp("ffn"))?,
            ffn_out: linear(cfg.intermediate_size, cfg.hidden_size, l.pp("ffn_output"))?,
            full_ln: layer_norm(cfg.hidden_size, ln, l.pp("full_layer_layer_norm"))?,
            cfg,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, t) = input_ids.dims2()?;
        let dev = input_ids.device();
        let pos = Tensor::arange(0u32, t as u32, dev)?
            .unsqueeze(0)?
            .repeat((b, 1))?;
        let types = Tensor::zeros((b, t), candle_core::DType::U32, dev)?;
        let x = (self.word.forward(input_ids)? + self.position.forward(&pos)?)?
            + self.token_type.forward(&types)?;
        let x = self.emb_ln.forward(&x?)?;
        let mut h = self.mapping_in.forward(&x)?;
        let heads = self.cfg.num_attention_heads;
        let dh = self.cfg.hidden_size / heads;
        let scale = 1.0 / (dh as f64).sqrt();
        for _ in 0..self.cfg.num_hidden_layers {
            let q = self
                .query
                .forward(&h)?
                .reshape((b, t, heads, dh))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = self
                .key
                .forward(&h)?
                .reshape((b, t, heads, dh))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = self
                .value
                .forward(&h)?
                .reshape((b, t, heads, dh))?
                .transpose(1, 2)?
                .contiguous()?;
            let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
            let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
            let ctx =
                probs
                    .matmul(&v)?
                    .transpose(1, 2)?
                    .contiguous()?
                    .reshape((b, t, heads * dh))?;
            let attn = self.attn_ln.forward(&(&h + self.dense.forward(&ctx)?)?)?;
            let f = self.ffn_out.forward(&self.ffn.forward(&attn)?.gelu()?)?;
            h = self.full_ln.forward(&(f + &attn)?)?;
        }
        Ok(h)
    }
}
