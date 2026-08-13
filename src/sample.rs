use crate::{config::ExperimentConfig, data::CharTokenizer, model::Gpt};
use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::{fs, path::Path};

/// Load a completed run and greedily extend a prompt.
///
/// Greedy decoding is intentionally used first: it is deterministic and makes
/// the relationship between logits and the selected token easy to inspect.
pub fn run(run_dir: &Path, prompt: &str, new_tokens: usize) -> Result<String> {
    anyhow::ensure!(!prompt.is_empty(), "prompt must not be empty");
    let config = ExperimentConfig::load(run_dir.join("config.resolved.toml"))?;
    let tokenizer_bytes = fs::read(run_dir.join("tokenizer.json"))
        .with_context(|| format!("failed to read tokenizer from {}", run_dir.display()))?;
    let mut tokenizer: CharTokenizer = serde_json::from_slice(&tokenizer_bytes)?;
    tokenizer.rebuild_index();
    let device = Device::Cpu;
    let mut variables = VarMap::new();
    let vb = VarBuilder::from_varmap(&variables, DType::F32, &device);
    let model = Gpt::new(&config.model, tokenizer.vocab_size(), vb)?;
    variables
        .load(run_dir.join("model.safetensors"))
        .with_context(|| format!("failed to load model from {}", run_dir.display()))?;

    greedy_generate(
        &model,
        &tokenizer,
        prompt,
        new_tokens,
        config.model.context_length,
        &device,
    )
}

pub fn greedy_generate(
    model: &Gpt,
    tokenizer: &CharTokenizer,
    prompt: &str,
    new_tokens: usize,
    context_length: usize,
    device: &Device,
) -> Result<String> {
    anyhow::ensure!(!prompt.is_empty(), "prompt must not be empty");
    let mut token_ids = tokenizer.encode(prompt)?;
    for _ in 0..new_tokens {
        let start = token_ids.len().saturating_sub(context_length);
        let window = &token_ids[start..];
        let input = Tensor::from_vec(window.to_vec(), (1, window.len()), device)?;
        let logits = model.forward(&input)?;
        let last_logits = logits.i((0, window.len() - 1))?;
        let next = last_logits.argmax(0)?.to_scalar::<u32>()?;
        token_ids.push(next);
    }

    tokenizer.decode(&token_ids)
}
