use crate::config::ModelConfig;
use candle_core::{Device, Result, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, ops::softmax_last_dim, Embedding, LayerNorm,
    Linear, Module, VarBuilder,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use sha2::{Digest, Sha256};

pub struct CausalSelfAttention {
    qkv: Linear,
    projection: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    fn new(config: &ModelConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            qkv: linear(config.d_model, 3 * config.d_model, vb.pp("qkv"))?,
            projection: linear(config.d_model, config.d_model, vb.pp("projection"))?,
            n_heads: config.n_heads,
            head_dim: config.d_model / config.n_heads,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, time, width) = xs.dims3()?;
        let qkv = self.qkv.forward(xs)?;
        let q = qkv.narrow(D::Minus1, 0, width)?;
        let k = qkv.narrow(D::Minus1, width, width)?;
        let v = qkv.narrow(D::Minus1, 2 * width, width)?;

        let split_heads = |x: Tensor| {
            x.reshape((batch, time, self.n_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split_heads(q)?;
        let k = split_heads(k)?;
        let v = split_heads(v)?;

        let scale = (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
        let mask =
            causal_mask(time, xs.device())?.broadcast_as((batch, self.n_heads, time, time))?;
        let minus_infinity = Tensor::full(f32::NEG_INFINITY, scores.shape(), xs.device())?;
        let scores = mask.where_cond(&scores, &minus_infinity)?;
        let weights = softmax_last_dim(&scores)?;
        let attended = weights.matmul(&v)?;
        let attended = attended
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, time, width))?;
        self.projection.forward(&attended)
    }
}

fn causal_mask(time: usize, device: &Device) -> Result<Tensor> {
    let values: Vec<u8> = (0..time)
        .flat_map(|row| (0..time).map(move |column| u8::from(column <= row)))
        .collect();
    Tensor::from_vec(values, (1, 1, time, time), device)
}

pub struct FeedForward {
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn new(config: &ModelConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            up: linear(config.d_model, config.d_ff, vb.pp("up"))?,
            down: linear(config.d_ff, config.d_model, vb.pp("down"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.down.forward(&self.up.forward(xs)?.gelu_erf()?)
    }
}

pub struct TransformerBlock {
    attention_norm: LayerNorm,
    attention: CausalSelfAttention,
    feed_forward_norm: LayerNorm,
    feed_forward: FeedForward,
}

impl TransformerBlock {
    fn new(config: &ModelConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            attention_norm: layer_norm(config.d_model, 1e-5, vb.pp("attention_norm"))?,
            attention: CausalSelfAttention::new(config, vb.pp("attention"))?,
            feed_forward_norm: layer_norm(config.d_model, 1e-5, vb.pp("feed_forward_norm"))?,
            feed_forward: FeedForward::new(config, vb.pp("feed_forward"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let attention_input = self.attention_norm.forward(xs)?;
        let xs = (xs + self.attention.forward(&attention_input)?)?;
        let feed_forward_input = self.feed_forward_norm.forward(&xs)?;
        xs + self.feed_forward.forward(&feed_forward_input)?
    }
}

pub struct Gpt {
    token_embedding: Embedding,
    position_embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    lm_head: Linear,
    context_length: usize,
    vocab_size: usize,
}

impl Gpt {
    pub fn new(config: &ModelConfig, vocab_size: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let blocks = (0..config.n_layers)
            .map(|index| TransformerBlock::new(config, vb.pp(format!("block-{index}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            token_embedding: embedding(vocab_size, config.d_model, vb.pp("token_embedding"))?,
            position_embedding: embedding(
                config.context_length,
                config.d_model,
                vb.pp("position_embedding"),
            )?,
            blocks,
            final_norm: layer_norm(config.d_model, 1e-5, vb.pp("final_norm"))?,
            lm_head: linear_no_bias(config.d_model, vocab_size, vb.pp("lm_head"))?,
            context_length: config.context_length,
            vocab_size,
        })
    }

    /// Returns next-token logits with shape [batch, time, vocabulary].
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let (_batch, time) = token_ids.dims2()?;
        if time > self.context_length {
            candle_core::bail!(
                "sequence length {time} exceeds context length {}",
                self.context_length
            );
        }
        let positions = Tensor::arange(0u32, time as u32, token_ids.device())?;
        let token_embeddings = self.token_embedding.forward(token_ids)?;
        let position_embeddings = self.position_embedding.forward(&positions)?;
        let mut xs = token_embeddings.broadcast_add(&position_embeddings)?;
        for block in &self.blocks {
            xs = block.forward(&xs)?;
        }
        self.lm_head.forward(&self.final_norm.forward(&xs)?)
    }

    pub fn loss(&self, token_ids: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let logits = self.forward(token_ids)?;
        let (batch, time, _) = logits.dims3()?;
        let logits = logits.reshape((batch * time, self.vocab_size))?;
        let targets = targets.flatten_all()?;
        candle_nn::loss::cross_entropy(&logits, &targets)
    }
}

pub fn parameter_count(variables: &candle_nn::VarMap) -> usize {
    variables
        .data()
        .lock()
        .expect("parameter map lock poisoned")
        .values()
        .map(|variable| variable.as_tensor().elem_count())
        .sum()
}

/// Deterministically initialize GPT parameters on CPU.
///
/// Candle's CPU backend cannot currently seed its default initializer. Deriving
/// a separate RNG seed from `(experiment_seed, parameter_name)` also makes the
/// result independent of HashMap iteration order.
pub fn initialize_variables(
    variables: &candle_nn::VarMap,
    seed: u64,
    device: &Device,
) -> Result<()> {
    let mut parameters = variables
        .data()
        .lock()
        .expect("parameter map lock poisoned");
    for (name, variable) in parameters.iter_mut() {
        let shape = variable.as_tensor().shape().clone();
        let count = shape.elem_count();
        let values = if name.ends_with("bias") {
            vec![0.0f32; count]
        } else if name.contains("norm") && name.ends_with("weight") {
            vec![1.0f32; count]
        } else {
            let digest = Sha256::digest(name.as_bytes());
            let name_seed = u64::from_le_bytes(digest[..8].try_into().expect("eight bytes"));
            let mut rng = StdRng::seed_from_u64(seed ^ name_seed);
            (0..count)
                .map(|_| {
                    let u1 = rng.random_range(f64::MIN_POSITIVE..1.0);
                    let u2 = rng.random_range(0.0..1.0);
                    let standard_normal =
                        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                    (standard_normal * 0.02) as f32
                })
                .collect()
        };
        variable.set(&Tensor::from_vec(values, shape, device)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, IndexOp};
    use candle_nn::{VarBuilder, VarMap};

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            context_length: 8,
            d_model: 16,
            n_heads: 4,
            n_layers: 2,
            d_ff: 32,
            expected_parameters: None,
        }
    }

    #[test]
    fn causal_mask_hides_future_positions() {
        let mask = causal_mask(3, &Device::Cpu)
            .unwrap()
            .reshape((3, 3))
            .unwrap()
            .to_vec2::<u8>()
            .unwrap();
        assert_eq!(mask, vec![vec![1, 0, 0], vec![1, 1, 0], vec![1, 1, 1]]);
    }

    #[test]
    fn model_returns_one_distribution_per_input_token() {
        let device = Device::Cpu;
        let variables = VarMap::new();
        let vb = VarBuilder::from_varmap(&variables, DType::F32, &device);
        let model = Gpt::new(&tiny_config(), 11, vb).unwrap();
        let input = Tensor::zeros((2, 5), DType::U32, &device).unwrap();
        assert_eq!(model.forward(&input).unwrap().dims(), &[2, 5, 11]);
    }

    #[test]
    fn future_tokens_do_not_change_earlier_logits() {
        let device = Device::Cpu;
        let variables = VarMap::new();
        let vb = VarBuilder::from_varmap(&variables, DType::F32, &device);
        let model = Gpt::new(&tiny_config(), 11, vb).unwrap();
        let a = Tensor::new(&[[1u32, 2, 3, 4]], &device).unwrap();
        let b = Tensor::new(&[[1u32, 2, 9, 8]], &device).unwrap();
        let a_prefix = model.forward(&a).unwrap().i((0, 0..2)).unwrap();
        let b_prefix = model.forward(&b).unwrap().i((0, 0..2)).unwrap();
        let difference = (&a_prefix - &b_prefix)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(difference < 1e-5, "future information leaked: {difference}");
    }

    #[test]
    fn saved_model_reload_produces_identical_logits() {
        let device = Device::Cpu;
        let first_variables = VarMap::new();
        let first_builder = VarBuilder::from_varmap(&first_variables, DType::F32, &device);
        let first_model = Gpt::new(&tiny_config(), 11, first_builder).unwrap();
        let input = Tensor::new(&[[1u32, 2, 3, 4]], &device).unwrap();
        let expected = first_model.forward(&input).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let weights = directory.path().join("model.safetensors");
        first_variables.save(&weights).unwrap();

        let mut second_variables = VarMap::new();
        let second_builder = VarBuilder::from_varmap(&second_variables, DType::F32, &device);
        let second_model = Gpt::new(&tiny_config(), 11, second_builder).unwrap();
        second_variables.load(&weights).unwrap();
        let actual = second_model.forward(&input).unwrap();
        let difference = (&expected - &actual)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(difference, 0.0);
    }

    #[test]
    fn named_initialization_is_reproducible_from_seed() {
        let device = Device::Cpu;
        let input = Tensor::new(&[[1u32, 2, 3, 4]], &device).unwrap();
        let build = |seed| {
            let variables = VarMap::new();
            let builder = VarBuilder::from_varmap(&variables, DType::F32, &device);
            let model = Gpt::new(&tiny_config(), 11, builder).unwrap();
            initialize_variables(&variables, seed, &device).unwrap();
            model.forward(&input).unwrap()
        };
        let first = build(42);
        let second = build(42);
        let third = build(43);
        let same = (&first - &second)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let different = (&first - &third)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(same, 0.0);
        assert!(different > 0.0);
    }
}
