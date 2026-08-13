use crate::{
    config::ExperimentConfig,
    data::{random_batch, CharTokenizer, TokenDataset},
    model::Gpt,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::{rngs::StdRng, SeedableRng};
use serde::Serialize;
use std::{fs, path::Path};

#[derive(Serialize)]
struct Metric {
    step: usize,
    train_nll: f32,
    validation_nll: f32,
    validation_perplexity: f32,
}

pub fn run(config: &ExperimentConfig) -> Result<()> {
    let text = fs::read_to_string(&config.data.path)
        .with_context(|| format!("failed to read dataset {}", config.data.path))?;
    let tokenizer = CharTokenizer::train(&text);
    let dataset = TokenDataset::from_text(&text, &tokenizer, config.data.train_fraction)?;

    let output_dir = Path::new(&config.output_dir);
    fs::create_dir_all(output_dir)?;
    fs::write(
        output_dir.join("config.resolved.toml"),
        toml::to_string_pretty(config)?,
    )?;
    fs::write(
        output_dir.join("tokenizer.json"),
        serde_json::to_vec_pretty(&tokenizer)?,
    )?;

    let device = Device::Cpu;
    let variables = VarMap::new();
    let vb = VarBuilder::from_varmap(&variables, DType::F32, &device);
    let model = Gpt::new(&config.model, tokenizer.vocab_size(), vb)?;
    let optimizer_config = ParamsAdamW {
        lr: config.training.learning_rate,
        weight_decay: config.training.weight_decay,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(variables.all_vars(), optimizer_config)?;
    let mut rng = StdRng::seed_from_u64(config.training.seed);
    let metrics_path = output_dir.join("metrics.jsonl");
    let mut metrics = String::new();

    println!(
        "vocabulary={} train_tokens={} validation_tokens={} device={:?}",
        tokenizer.vocab_size(),
        dataset.train.len(),
        dataset.validation.len(),
        device
    );

    for step in 0..=config.training.steps {
        if step % config.training.eval_interval == 0 || step == config.training.steps {
            let train_nll = evaluate(&model, &dataset.train, config, &mut rng, &device)?;
            let validation_nll = evaluate(&model, &dataset.validation, config, &mut rng, &device)?;
            let metric = Metric {
                step,
                train_nll,
                validation_nll,
                validation_perplexity: validation_nll.exp(),
            };
            println!(
                "step={step:>6} train_nll={train_nll:.4} validation_nll={validation_nll:.4} perplexity={:.2}",
                metric.validation_perplexity
            );
            metrics.push_str(&serde_json::to_string(&metric)?);
            metrics.push('\n');
            fs::write(&metrics_path, &metrics)?;
        }

        if step == config.training.steps {
            break;
        }
        let (inputs, targets) = random_batch(
            &dataset.train,
            config.training.batch_size,
            config.model.context_length,
            &mut rng,
            &device,
        )?;
        let loss = model.loss(&inputs, &targets)?;
        optimizer.backward_step(&loss)?;
    }

    variables.save(output_dir.join("model.safetensors"))?;
    Ok(())
}

fn evaluate(
    model: &Gpt,
    tokens: &[u32],
    config: &ExperimentConfig,
    rng: &mut StdRng,
    device: &Device,
) -> Result<f32> {
    let mut sum = 0.0;
    for _ in 0..config.training.eval_batches {
        let (inputs, targets) = random_batch(
            tokens,
            config.training.batch_size,
            config.model.context_length,
            rng,
            device,
        )?;
        sum += model.loss(&inputs, &targets)?.to_scalar::<f32>()?;
    }
    Ok(sum / config.training.eval_batches as f32)
}
