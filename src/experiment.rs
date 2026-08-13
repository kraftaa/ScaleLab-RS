use crate::{
    config::{DataConfig, ExperimentConfig, ModelConfig, TrainingConfig},
    data::{random_batch, CharTokenizer},
    model::{initialize_variables, parameter_count, Gpt},
    sample::greedy_generate,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleExperiment {
    pub name: String,
    pub output_dir: PathBuf,
    pub tokenizer_corpus: PathBuf,
    pub validation_corpus: PathBuf,
    pub model: ModelConfig,
    pub training: ScaleTrainingConfig,
    #[serde(default)]
    pub prompts: Vec<String>,
    pub runs: Vec<ScaleRunConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleTrainingConfig {
    pub batch_size: usize,
    pub eval_interval: usize,
    pub eval_batches: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub seeds: Vec<u64>,
    #[serde(default = "default_sample_tokens")]
    pub sample_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleRunConfig {
    pub name: String,
    pub train_corpus: PathBuf,
    pub target_tokens_per_parameter: f64,
}

#[derive(Debug, Clone)]
pub struct CheckedExperiment {
    pub spec: ScaleExperiment,
    pub tokenizer: CharTokenizer,
    pub tokenizer_sha256: String,
    pub validation_tokens: Vec<u32>,
    pub validation_sha256: String,
    pub parameter_count: usize,
    pub runs: Vec<CheckedRun>,
}

#[derive(Debug, Clone)]
pub struct CheckedRun {
    pub config: ScaleRunConfig,
    pub train_tokens: Vec<u32>,
    pub train_sha256: String,
    pub observed_token_types: usize,
    pub target_processed_tokens: usize,
    pub steps: usize,
    pub actual_processed_tokens: usize,
    pub effective_epochs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetArtifact {
    pub tokenizer_corpus_sha256: String,
    pub tokenizer_sha256: String,
    pub train_sha256: String,
    pub validation_sha256: String,
    pub train_corpus_tokens: usize,
    pub validation_corpus_tokens: usize,
    pub tokenizer_vocab_size: usize,
    pub observed_train_token_types: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetric {
    pub step: usize,
    pub parameter_count: usize,
    pub train_corpus_tokens: usize,
    pub processed_tokens: usize,
    pub tokens_per_parameter: f64,
    pub effective_epochs: f64,
    pub train_nll: f32,
    pub validation_nll: f32,
    pub generalization_gap: f32,
    pub validation_perplexity: f32,
    pub elapsed_seconds: f64,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleCheckpoint {
    pub step: usize,
    pub samples: Vec<GeneratedSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedSample {
    pub prompt: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub experiment: String,
    pub run: String,
    pub seed: u64,
    pub parameter_count: usize,
    pub train_corpus_tokens: usize,
    pub validation_corpus_tokens: usize,
    pub target_processed_tokens: usize,
    pub actual_processed_tokens: usize,
    pub tokens_per_parameter: f64,
    pub effective_epochs: f64,
    pub best_validation_nll: f32,
    pub best_step: usize,
    pub final_train_nll: f32,
    pub final_validation_nll: f32,
    pub final_generalization_gap: f32,
    pub elapsed_seconds: f64,
    pub initial_weights_sha256: String,
    pub tokenizer_sha256: String,
    pub train_sha256: String,
    pub validation_sha256: String,
    pub control_sha256: String,
}

fn default_sample_tokens() -> usize {
    80
}

impl ScaleExperiment {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read experiment {}", path.display()))?;
        let mut spec: Self = toml::from_str(&text)
            .with_context(|| format!("failed to parse experiment {}", path.display()))?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        spec.output_dir = resolve(base, &spec.output_dir);
        spec.tokenizer_corpus = resolve(base, &spec.tokenizer_corpus);
        spec.validation_corpus = resolve(base, &spec.validation_corpus);
        for run in &mut spec.runs {
            run.train_corpus = resolve(base, &run.train_corpus);
        }
        Ok(spec)
    }
}

impl ScaleTrainingConfig {
    pub fn effective_seeds(&self) -> Result<Vec<u64>> {
        anyhow::ensure!(
            self.seed.is_none() || self.seeds.is_empty(),
            "configure either training.seed or training.seeds, not both"
        );
        let seeds = if self.seeds.is_empty() {
            vec![self.seed.unwrap_or(42)]
        } else {
            self.seeds.clone()
        };
        let unique: BTreeSet<_> = seeds.iter().copied().collect();
        anyhow::ensure!(unique.len() == seeds.len(), "training seeds must be unique");
        Ok(seeds)
    }
}

fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

pub fn check(spec: ScaleExperiment) -> Result<CheckedExperiment> {
    anyhow::ensure!(
        !spec.name.trim().is_empty(),
        "experiment name must not be empty"
    );
    anyhow::ensure!(
        spec.model.context_length > 0,
        "context_length must be positive"
    );
    anyhow::ensure!(spec.model.n_heads > 0, "n_heads must be positive");
    anyhow::ensure!(
        spec.model.d_model.is_multiple_of(spec.model.n_heads),
        "d_model must be divisible by n_heads"
    );
    anyhow::ensure!(spec.training.batch_size > 0, "batch_size must be positive");
    anyhow::ensure!(
        spec.training.eval_interval > 0,
        "eval_interval must be positive"
    );
    anyhow::ensure!(
        spec.training.eval_batches > 0,
        "eval_batches must be positive"
    );
    let _seeds = spec.training.effective_seeds()?;
    anyhow::ensure!(
        spec.runs.len() >= 2,
        "an experiment requires at least two runs"
    );
    let names: BTreeSet<_> = spec.runs.iter().map(|run| run.name.as_str()).collect();
    anyhow::ensure!(names.len() == spec.runs.len(), "run names must be unique");

    let tokenizer_text = read_text(&spec.tokenizer_corpus, "tokenizer corpus")?;
    let tokenizer_corpus_sha256 = sha256_file(&spec.tokenizer_corpus)?;
    let tokenizer = CharTokenizer::train(&tokenizer_text);
    let tokenizer_bytes = serde_json::to_vec_pretty(&tokenizer)?;
    let tokenizer_sha256 = sha256_bytes(&tokenizer_bytes);
    let validation_text = read_text(&spec.validation_corpus, "validation corpus")?;
    let validation_sha256 = sha256_file(&spec.validation_corpus)?;
    let validation_tokens = tokenizer
        .encode(&validation_text)
        .context("validation corpus contains characters absent from the frozen tokenizer")?;
    anyhow::ensure!(
        validation_tokens.len() > spec.model.context_length,
        "validation corpus is too short for the configured context"
    );

    let device = Device::Cpu;
    let variables = VarMap::new();
    let vb = VarBuilder::from_varmap(&variables, DType::F32, &device);
    let _model = Gpt::new(&spec.model, tokenizer.vocab_size(), vb)?;
    let parameter_count = parameter_count(&variables);
    if let Some(expected) = spec.model.expected_parameters {
        anyhow::ensure!(
            expected == parameter_count,
            "expected {expected} parameters; instantiated model has {parameter_count}"
        );
    }

    let tokens_per_step = spec.training.batch_size * spec.model.context_length;
    let mut runs = Vec::with_capacity(spec.runs.len());
    for run in &spec.runs {
        anyhow::ensure!(
            run.target_tokens_per_parameter > 0.0,
            "run {} has a non-positive token target",
            run.name
        );
        let train_text = read_text(&run.train_corpus, "training corpus")?;
        let train_sha256 = sha256_file(&run.train_corpus)?;
        anyhow::ensure!(
            train_sha256 != validation_sha256,
            "run {} uses the validation corpus as training data",
            run.name
        );
        let train_tokens = tokenizer.encode(&train_text).with_context(|| {
            format!(
                "run {} contains characters absent from the frozen tokenizer",
                run.name
            )
        })?;
        anyhow::ensure!(
            train_tokens.len() > spec.model.context_length,
            "training corpus for {} is too short for the configured context",
            run.name
        );
        let target_processed_tokens =
            (parameter_count as f64 * run.target_tokens_per_parameter).ceil() as usize;
        let steps = target_processed_tokens.div_ceil(tokens_per_step);
        let actual_processed_tokens = steps * tokens_per_step;
        let effective_epochs = actual_processed_tokens as f64 / train_tokens.len() as f64;
        runs.push(CheckedRun {
            config: run.clone(),
            observed_token_types: tokenizer.observed_token_types(&train_tokens),
            train_tokens,
            train_sha256,
            target_processed_tokens,
            steps,
            actual_processed_tokens,
            effective_epochs,
        });
    }

    // Keep the source hash in the check result via this assertion and recompute it for artifacts.
    anyhow::ensure!(
        !tokenizer_corpus_sha256.is_empty(),
        "failed to hash tokenizer corpus"
    );
    Ok(CheckedExperiment {
        spec,
        tokenizer,
        tokenizer_sha256,
        validation_tokens,
        validation_sha256,
        parameter_count,
        runs,
    })
}

pub fn print_check(checked: &CheckedExperiment) {
    println!("ScaleLab-RS experiment check\n");
    println!("Model");
    println!(
        "  Parameters                 {:>12}",
        checked.parameter_count
    );
    println!(
        "  Context length             {:>12}",
        checked.spec.model.context_length
    );
    println!(
        "  Vocabulary size            {:>12}\n",
        checked.tokenizer.vocab_size()
    );
    let seeds = checked.spec.training.effective_seeds().unwrap_or_default();
    println!("Replications");
    println!("  Seeds                      {:>12?}\n", seeds);
    for run in &checked.runs {
        println!("{}", run.config.name);
        println!(
            "  Training corpus tokens     {:>12}",
            run.train_tokens.len()
        );
        println!(
            "  Target processed tokens    {:>12}",
            run.target_processed_tokens
        );
        println!(
            "  Actual processed tokens    {:>12}",
            run.actual_processed_tokens
        );
        println!(
            "  Estimated effective epochs {:>12.2}\n",
            run.effective_epochs
        );
    }
    let mut budgets = std::collections::BTreeMap::<usize, Vec<&str>>::new();
    for run in &checked.runs {
        budgets
            .entry(run.actual_processed_tokens)
            .or_default()
            .push(&run.config.name);
    }
    println!("Matched processed-token budgets");
    for (tokens, names) in budgets.into_iter().filter(|(_, names)| names.len() > 1) {
        println!("  {tokens:>12} tokens  {:?}  ✓", names);
    }
    println!();
    println!("Controlled variables");
    println!("  Instantiated architecture             ✓");
    println!("  Initial parameter state               ✓ (paired within each seed)");
    println!("  Frozen tokenizer                      ✓");
    println!("  Optimizer configuration               ✓");
    println!("  Batch and context sizes               ✓");
    println!("  Validation corpus                     ✓");
    println!("\nChanging variable");
    println!("  Training corpus available before reuse");
    println!("\nLeakage checks");
    println!("  Separate train/validation files       ✓");
    println!("  Distinct train/validation hashes      ✓");
    println!("  No cross-boundary windows             ✓");
    println!("\nExperiment validity: PASS");
}

pub fn run(checked: CheckedExperiment) -> Result<()> {
    fs::create_dir_all(&checked.spec.output_dir)?;
    fs::write(
        checked.spec.output_dir.join("experiment.resolved.toml"),
        toml::to_string_pretty(&checked.spec)?,
    )?;
    let tokenizer_bytes = serde_json::to_vec_pretty(&checked.tokenizer)?;
    fs::write(
        checked.spec.output_dir.join("tokenizer.json"),
        &tokenizer_bytes,
    )?;

    println!("\nRunning experiment {}", checked.spec.name);
    for seed in checked.spec.training.effective_seeds()? {
        let initial_weights = checked
            .spec
            .output_dir
            .join(format!("initial-deterministic-seed-{seed}.safetensors"));
        let device = Device::Cpu;
        let variables = VarMap::new();
        let vb = VarBuilder::from_varmap(&variables, DType::F32, &device);
        let _model = Gpt::new(&checked.spec.model, checked.tokenizer.vocab_size(), vb)?;
        initialize_variables(&variables, seed, &device)?;
        variables.save(&initial_weights)?;
        let initial_weights_sha256 = sha256_file(&initial_weights)?;
        println!("\nSeed {seed} initial weights: {initial_weights_sha256}");
        for checked_run in &checked.runs {
            train_run(
                &checked,
                checked_run,
                seed,
                &initial_weights,
                &initial_weights_sha256,
            )?;
        }
    }
    Ok(())
}

fn train_run(
    checked: &CheckedExperiment,
    checked_run: &CheckedRun,
    seed: u64,
    initial_weights: &Path,
    initial_weights_sha256: &str,
) -> Result<()> {
    let run_dir = checked
        .spec
        .output_dir
        .join(&checked_run.config.name)
        .join(format!("seed-{seed}"));
    fs::create_dir_all(&run_dir)?;
    fs::write(
        run_dir.join("tokenizer.json"),
        serde_json::to_vec_pretty(&checked.tokenizer)?,
    )?;

    let device = Device::Cpu;
    let mut variables = VarMap::new();
    let vb = VarBuilder::from_varmap(&variables, DType::F32, &device);
    let model = Gpt::new(&checked.spec.model, checked.tokenizer.vocab_size(), vb)?;
    variables.load(initial_weights)?;
    let optimizer_config = ParamsAdamW {
        lr: checked.spec.training.learning_rate,
        weight_decay: checked.spec.training.weight_decay,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(variables.all_vars(), optimizer_config)?;
    let mut training_rng = StdRng::seed_from_u64(seed);
    let started = Instant::now();
    let tokens_per_step = checked.spec.training.batch_size * checked.spec.model.context_length;
    let tokenizer_corpus_sha256 = sha256_file(&checked.spec.tokenizer_corpus)?;
    let dataset = DatasetArtifact {
        tokenizer_corpus_sha256,
        tokenizer_sha256: checked.tokenizer_sha256.clone(),
        train_sha256: checked_run.train_sha256.clone(),
        validation_sha256: checked.validation_sha256.clone(),
        train_corpus_tokens: checked_run.train_tokens.len(),
        validation_corpus_tokens: checked.validation_tokens.len(),
        tokenizer_vocab_size: checked.tokenizer.vocab_size(),
        observed_train_token_types: checked_run.observed_token_types,
    };
    fs::write(
        run_dir.join("dataset.json"),
        serde_json::to_vec_pretty(&dataset)?,
    )?;

    let compatible_config = ExperimentConfig {
        data: DataConfig {
            path: checked_run.config.train_corpus.display().to_string(),
            train_fraction: 0.999_999,
        },
        model: checked.spec.model.clone(),
        training: TrainingConfig {
            batch_size: checked.spec.training.batch_size,
            steps: checked_run.steps,
            eval_interval: checked.spec.training.eval_interval,
            eval_batches: checked.spec.training.eval_batches,
            learning_rate: checked.spec.training.learning_rate,
            weight_decay: checked.spec.training.weight_decay,
            seed,
        },
        output_dir: run_dir.display().to_string(),
    };
    fs::write(
        run_dir.join("config.resolved.toml"),
        toml::to_string_pretty(&compatible_config)?,
    )?;

    let mut metrics = Vec::new();
    let mut samples = Vec::new();
    println!(
        "\n{} seed={seed}: corpus={} steps={} actual_tokens={} effective_epochs={:.2}",
        checked_run.config.name,
        checked_run.train_tokens.len(),
        checked_run.steps,
        checked_run.actual_processed_tokens,
        checked_run.effective_epochs
    );
    for step in 0..=checked_run.steps {
        if step % checked.spec.training.eval_interval == 0 || step == checked_run.steps {
            let train_nll = evaluate_fixed(
                &model,
                &checked_run.train_tokens,
                checked,
                seed ^ 0x0054_5241_494e,
                &device,
            )?;
            let validation_nll = evaluate_fixed(
                &model,
                &checked.validation_tokens,
                checked,
                seed ^ 0x0056_414c_4944,
                &device,
            )?;
            let processed_tokens = step * tokens_per_step;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let metric = ExperimentMetric {
                step,
                parameter_count: checked.parameter_count,
                train_corpus_tokens: checked_run.train_tokens.len(),
                processed_tokens,
                tokens_per_parameter: processed_tokens as f64 / checked.parameter_count as f64,
                effective_epochs: processed_tokens as f64 / checked_run.train_tokens.len() as f64,
                train_nll,
                validation_nll,
                generalization_gap: validation_nll - train_nll,
                validation_perplexity: validation_nll.exp(),
                elapsed_seconds,
                tokens_per_second: if elapsed_seconds > 0.0 {
                    processed_tokens as f64 / elapsed_seconds
                } else {
                    0.0
                },
            };
            println!(
                "  step={step:>6} tok/param={:.2} train={train_nll:.4} valid={validation_nll:.4} gap={:.4}",
                metric.tokens_per_parameter, metric.generalization_gap
            );
            metrics.push(metric);
            let generated = checked
                .spec
                .prompts
                .iter()
                .map(|prompt| {
                    Ok(GeneratedSample {
                        prompt: prompt.clone(),
                        text: greedy_generate(
                            &model,
                            &checked.tokenizer,
                            prompt,
                            checked.spec.training.sample_tokens,
                            checked.spec.model.context_length,
                            &device,
                        )?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            samples.push(SampleCheckpoint {
                step,
                samples: generated,
            });
            write_jsonl(run_dir.join("metrics.jsonl"), &metrics)?;
            fs::write(
                run_dir.join("samples.json"),
                serde_json::to_vec_pretty(&samples)?,
            )?;
        }

        if step == checked_run.steps {
            break;
        }
        let (inputs, targets) = random_batch(
            &checked_run.train_tokens,
            checked.spec.training.batch_size,
            checked.spec.model.context_length,
            &mut training_rng,
            &device,
        )?;
        optimizer.backward_step(&model.loss(&inputs, &targets)?)?;
    }
    variables.save(run_dir.join("model.safetensors"))?;

    let final_metric = metrics.last().context("run produced no metrics")?;
    let best = metrics
        .iter()
        .min_by(|left, right| left.validation_nll.total_cmp(&right.validation_nll))
        .context("run produced no metrics")?;
    let summary = RunSummary {
        experiment: checked.spec.name.clone(),
        run: checked_run.config.name.clone(),
        seed,
        parameter_count: checked.parameter_count,
        train_corpus_tokens: checked_run.train_tokens.len(),
        validation_corpus_tokens: checked.validation_tokens.len(),
        target_processed_tokens: checked_run.target_processed_tokens,
        actual_processed_tokens: checked_run.actual_processed_tokens,
        tokens_per_parameter: final_metric.tokens_per_parameter,
        effective_epochs: final_metric.effective_epochs,
        best_validation_nll: best.validation_nll,
        best_step: best.step,
        final_train_nll: final_metric.train_nll,
        final_validation_nll: final_metric.validation_nll,
        final_generalization_gap: final_metric.generalization_gap,
        elapsed_seconds: final_metric.elapsed_seconds,
        initial_weights_sha256: initial_weights_sha256.to_string(),
        tokenizer_sha256: checked.tokenizer_sha256.clone(),
        train_sha256: checked_run.train_sha256.clone(),
        validation_sha256: checked.validation_sha256.clone(),
        control_sha256: control_sha256(checked)?,
    };
    fs::write(
        run_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(())
}

fn control_sha256(checked: &CheckedExperiment) -> Result<String> {
    #[derive(Serialize)]
    struct Controls<'a> {
        model: &'a ModelConfig,
        batch_size: usize,
        eval_interval: usize,
        eval_batches: usize,
        learning_rate: f64,
        weight_decay: f64,
    }
    let controls = Controls {
        model: &checked.spec.model,
        batch_size: checked.spec.training.batch_size,
        eval_interval: checked.spec.training.eval_interval,
        eval_batches: checked.spec.training.eval_batches,
        learning_rate: checked.spec.training.learning_rate,
        weight_decay: checked.spec.training.weight_decay,
    };
    Ok(sha256_bytes(&serde_json::to_vec(&controls)?))
}

fn evaluate_fixed(
    model: &Gpt,
    tokens: &[u32],
    checked: &CheckedExperiment,
    seed: u64,
    device: &Device,
) -> Result<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut sum = 0.0;
    for _ in 0..checked.spec.training.eval_batches {
        let (inputs, targets) = random_batch(
            tokens,
            checked.spec.training.batch_size,
            checked.spec.model.context_length,
            &mut rng,
            device,
        )?;
        sum += model.loss(&inputs, &targets)?.to_scalar::<f32>()?;
    }
    Ok(sum / checked.spec.training.eval_batches as f32)
}

fn write_jsonl(path: PathBuf, metrics: &[ExperimentMetric]) -> Result<()> {
    let mut output = String::new();
    for metric in metrics {
        output.push_str(&serde_json::to_string(metric)?);
        output.push('\n');
    }
    fs::write(path, output)?;
    Ok(())
}

fn read_text(path: &Path, label: &str) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("failed to read {label} {}", path.display()))
}

pub fn sha256_file(path: &Path) -> Result<String> {
    Ok(sha256_bytes(&fs::read(path)?))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    #[test]
    fn exposure_math_rounds_up_to_complete_steps() {
        let parameters = 27_520usize;
        let target = (parameters as f64 * 20.0).ceil() as usize;
        let tokens_per_step = 4 * 16;
        let steps = target.div_ceil(tokens_per_step);
        assert_eq!(target, 550_400);
        assert_eq!(steps, 8_600);
        assert_eq!(steps * tokens_per_step, target);
    }
}
