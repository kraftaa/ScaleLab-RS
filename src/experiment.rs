use crate::{
    config::{DataConfig, ExperimentConfig, ModelConfig, TrainingConfig},
    data::{batch_from_starts, random_batch, strided_starts, CharTokenizer},
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
    #[serde(default)]
    pub in_domain_validation_corpus: Option<PathBuf>,
    pub model: ModelConfig,
    pub training: ScaleTrainingConfig,
    #[serde(default)]
    pub prompts: Vec<String>,
    /// When true, distinct training corpora must form a prefix chain: each
    /// shorter corpus is an exact token prefix of each longer one.
    #[serde(default = "default_nested_training_corpora")]
    pub nested_training_corpora: bool,
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
    pub in_domain_validation_tokens: Option<Vec<u32>>,
    pub in_domain_validation_sha256: Option<String>,
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
    #[serde(default)]
    pub in_domain_validation_sha256: Option<String>,
    pub train_corpus_tokens: usize,
    pub validation_corpus_tokens: usize,
    #[serde(default)]
    pub in_domain_validation_corpus_tokens: Option<usize>,
    pub tokenizer_vocab_size: usize,
    pub observed_train_token_types: usize,
    pub train_eval_starts: Vec<usize>,
    pub validation_eval_starts: Vec<usize>,
    #[serde(default)]
    pub in_domain_eval_starts: Option<Vec<usize>>,
    pub eval_windows_sha256: String,
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
    #[serde(default)]
    pub in_domain_validation_nll: Option<f32>,
    #[serde(default)]
    pub in_domain_generalization_gap: Option<f32>,
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
    #[serde(default)]
    pub final_in_domain_validation_nll: Option<f32>,
    #[serde(default)]
    pub final_in_domain_generalization_gap: Option<f32>,
    pub elapsed_seconds: f64,
    pub initial_weights_sha256: String,
    pub tokenizer_sha256: String,
    pub train_sha256: String,
    pub validation_sha256: String,
    #[serde(default)]
    pub in_domain_validation_sha256: Option<String>,
    pub control_sha256: String,
}

fn default_sample_tokens() -> usize {
    80
}

fn default_nested_training_corpora() -> bool {
    true
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
        if let Some(path) = spec.in_domain_validation_corpus.as_mut() {
            *path = resolve(base, path);
        }
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

    let (in_domain_validation_tokens, in_domain_validation_sha256, in_domain_text) =
        if let Some(path) = &spec.in_domain_validation_corpus {
            let text = read_text(path, "in-domain validation corpus")?;
            let sha256 = sha256_file(path)?;
            anyhow::ensure!(
                sha256 != validation_sha256,
                "in-domain validation corpus must differ from the out-of-domain validation corpus"
            );
            let tokens = tokenizer.encode(&text).context(
                "in-domain validation corpus contains characters absent from the frozen tokenizer",
            )?;
            anyhow::ensure!(
                tokens.len() > spec.model.context_length,
                "in-domain validation corpus is too short for the configured context"
            );
            (Some(tokens), Some(sha256), Some(text))
        } else {
            (None, None, None)
        };

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
        if let Some(in_domain_sha256) = &in_domain_validation_sha256 {
            anyhow::ensure!(
                train_sha256 != *in_domain_sha256,
                "run {} uses the in-domain validation corpus as training data",
                run.name
            );
        }
        if let Some(in_domain_text) = &in_domain_text {
            anyhow::ensure!(
                !train_text.contains(in_domain_text.trim_end()),
                "run {} training corpus contains the in-domain validation text",
                run.name
            );
        }
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

    if spec.nested_training_corpora {
        ensure_nested_token_corpora(
            runs.iter()
                .map(|run| (run.config.name.as_str(), run.train_tokens.as_slice())),
        )?;
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
        in_domain_validation_tokens,
        in_domain_validation_sha256,
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
    if checked.in_domain_validation_tokens.is_some() {
        println!("  In-domain validation corpus            ✓");
    }
    println!("  Strided eval windows                  ✓");
    println!("\nChanging variable");
    println!("  Training corpus available before reuse");
    println!("\nLeakage checks");
    println!("  Separate train/validation files       ✓");
    println!("  Distinct train/validation hashes      ✓");
    if checked.in_domain_validation_sha256.is_some() {
        println!("  In-domain val held out of training     ✓");
    }
    println!("  No cross-boundary windows             ✓");
    if checked.spec.nested_training_corpora {
        println!("  Nested training corpora (prefix)      ✓");
    } else {
        println!("  Nested training corpora (prefix)      skipped");
    }
    println!("\nExperiment validity: PASS");
}

fn ensure_nested_token_corpora<'a>(
    named: impl IntoIterator<Item = (&'a str, &'a [u32])>,
) -> Result<()> {
    let mut unique: Vec<(&str, &[u32])> = Vec::new();
    for (name, tokens) in named {
        if unique.iter().any(|(_, existing)| *existing == tokens) {
            continue;
        }
        unique.push((name, tokens));
    }
    unique.sort_by_key(|(_, tokens)| tokens.len());
    for pair in unique.windows(2) {
        let (short_name, short) = pair[0];
        let (long_name, long) = pair[1];
        anyhow::ensure!(
            long.len() > short.len(),
            "training corpora {short_name} and {long_name} have the same length but different contents; \
             nested_training_corpora requires a prefix chain"
        );
        anyhow::ensure!(
            long.starts_with(short),
            "training corpus {short_name} is not a prefix of {long_name}; \
             nested_training_corpora requires each smaller corpus to be an exact prefix of each larger one"
        );
    }
    Ok(())
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
        if !initial_weights.exists() {
            let variables = VarMap::new();
            let vb = VarBuilder::from_varmap(&variables, DType::F32, &device);
            let _model = Gpt::new(&checked.spec.model, checked.tokenizer.vocab_size(), vb)?;
            initialize_variables(&variables, seed, &device)?;
            variables.save(&initial_weights)?;
        }
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
    if should_skip_completed_run(&run_dir, checked, checked_run, seed, initial_weights_sha256)? {
        println!(
            "Skipping {} seed={seed}: completed run matches current controls",
            checked_run.config.name
        );
        return Ok(());
    }
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
    let train_eval_starts = eval_starts_for(&checked_run.train_tokens, checked)?;
    let validation_eval_starts = eval_starts_for(&checked.validation_tokens, checked)?;
    let in_domain_eval_starts = checked
        .in_domain_validation_tokens
        .as_ref()
        .map(|tokens| eval_starts_for(tokens, checked))
        .transpose()?;
    let eval_windows_sha256 = sha256_bytes(&serde_json::to_vec(&(
        &train_eval_starts,
        &validation_eval_starts,
        &in_domain_eval_starts,
    ))?);
    let dataset = DatasetArtifact {
        tokenizer_corpus_sha256,
        tokenizer_sha256: checked.tokenizer_sha256.clone(),
        train_sha256: checked_run.train_sha256.clone(),
        validation_sha256: checked.validation_sha256.clone(),
        in_domain_validation_sha256: checked.in_domain_validation_sha256.clone(),
        train_corpus_tokens: checked_run.train_tokens.len(),
        validation_corpus_tokens: checked.validation_tokens.len(),
        in_domain_validation_corpus_tokens: checked
            .in_domain_validation_tokens
            .as_ref()
            .map(Vec::len),
        tokenizer_vocab_size: checked.tokenizer.vocab_size(),
        observed_train_token_types: checked_run.observed_token_types,
        train_eval_starts,
        validation_eval_starts,
        in_domain_eval_starts,
        eval_windows_sha256,
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
            let train_nll = evaluate_strided(&model, &checked_run.train_tokens, checked, &device)?;
            let validation_nll =
                evaluate_strided(&model, &checked.validation_tokens, checked, &device)?;
            let in_domain_validation_nll = checked
                .in_domain_validation_tokens
                .as_ref()
                .map(|tokens| evaluate_strided(&model, tokens, checked, &device))
                .transpose()?;
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
                in_domain_validation_nll,
                in_domain_generalization_gap: in_domain_validation_nll.map(|nll| nll - train_nll),
                elapsed_seconds,
                tokens_per_second: if elapsed_seconds > 0.0 {
                    processed_tokens as f64 / elapsed_seconds
                } else {
                    0.0
                },
            };
            match in_domain_validation_nll {
                Some(in_domain) => println!(
                    "  step={step:>6} tok/param={:.2} train={train_nll:.4} valid={validation_nll:.4} in-domain={in_domain:.4} gap={:.4}",
                    metric.tokens_per_parameter, metric.generalization_gap
                ),
                None => println!(
                    "  step={step:>6} tok/param={:.2} train={train_nll:.4} valid={validation_nll:.4} gap={:.4}",
                    metric.tokens_per_parameter, metric.generalization_gap
                ),
            }
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
        final_in_domain_validation_nll: final_metric.in_domain_validation_nll,
        final_in_domain_generalization_gap: final_metric.in_domain_generalization_gap,
        elapsed_seconds: final_metric.elapsed_seconds,
        initial_weights_sha256: initial_weights_sha256.to_string(),
        tokenizer_sha256: checked.tokenizer_sha256.clone(),
        train_sha256: checked_run.train_sha256.clone(),
        validation_sha256: checked.validation_sha256.clone(),
        in_domain_validation_sha256: checked.in_domain_validation_sha256.clone(),
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
        eval_protocol: &'static str,
    }
    let controls = Controls {
        model: &checked.spec.model,
        batch_size: checked.spec.training.batch_size,
        eval_interval: checked.spec.training.eval_interval,
        eval_batches: checked.spec.training.eval_batches,
        learning_rate: checked.spec.training.learning_rate,
        weight_decay: checked.spec.training.weight_decay,
        eval_protocol: "strided-windows",
    };
    Ok(sha256_bytes(&serde_json::to_vec(&controls)?))
}

fn should_skip_completed_run(
    run_dir: &Path,
    checked: &CheckedExperiment,
    checked_run: &CheckedRun,
    seed: u64,
    initial_weights_sha256: &str,
) -> Result<bool> {
    let summary_path = run_dir.join("summary.json");
    if !summary_path.exists()
        || !run_dir.join("model.safetensors").exists()
        || !run_dir.join("metrics.jsonl").exists()
        || !run_dir.join("samples.json").exists()
        || !run_dir.join("dataset.json").exists()
    {
        return Ok(false);
    }
    // A process can be interrupted while writing its final summary. Treat a
    // partial or unreadable summary as an incomplete run so the next invocation
    // retrains it instead of making the whole experiment permanently fail.
    let Ok(summary_bytes) = fs::read(summary_path) else {
        return Ok(false);
    };
    let Ok(summary) = serde_json::from_slice::<RunSummary>(&summary_bytes) else {
        return Ok(false);
    };
    Ok(summary.seed == seed
        && summary.run == checked_run.config.name
        && summary.parameter_count == checked.parameter_count
        && summary.actual_processed_tokens == checked_run.actual_processed_tokens
        && summary.tokenizer_sha256 == checked.tokenizer_sha256
        && summary.train_sha256 == checked_run.train_sha256
        && summary.validation_sha256 == checked.validation_sha256
        && summary.in_domain_validation_sha256 == checked.in_domain_validation_sha256
        && summary.initial_weights_sha256 == initial_weights_sha256
        && summary.control_sha256 == control_sha256(checked)?)
}

fn eval_starts_for(tokens: &[u32], checked: &CheckedExperiment) -> Result<Vec<usize>> {
    let window_count = checked.spec.training.eval_batches * checked.spec.training.batch_size;
    strided_starts(
        tokens.len(),
        checked.spec.model.context_length,
        window_count,
    )
}

fn evaluate_strided(
    model: &Gpt,
    tokens: &[u32],
    checked: &CheckedExperiment,
    device: &Device,
) -> Result<f32> {
    let starts = eval_starts_for(tokens, checked)?;
    let mut sum = 0.0;
    let mut batches = 0usize;
    for chunk in starts.chunks(checked.spec.training.batch_size) {
        let (inputs, targets) =
            batch_from_starts(tokens, chunk, checked.spec.model.context_length, device)?;
        sum += model.loss(&inputs, &targets)?.to_scalar::<f32>()?;
        batches += 1;
    }
    anyhow::ensure!(batches > 0, "eval produced no batches");
    Ok(sum / batches as f32)
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
    use super::*;
    use std::fs;

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

    #[test]
    fn nested_corpora_accept_prefix_chain_and_duplicates() {
        let small = [1u32, 2, 3];
        let broad = [1u32, 2, 3, 4, 5];
        ensure_nested_token_corpora([
            ("small-low", small.as_slice()),
            ("reused", small.as_slice()),
            ("broad", broad.as_slice()),
        ])
        .unwrap();
    }

    #[test]
    fn nested_corpora_reject_non_prefix() {
        let error = ensure_nested_token_corpora([
            ("small", [1u32, 2].as_slice()),
            ("broad", [9u32, 8, 7].as_slice()),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("not a prefix"));
    }

    #[test]
    fn smoke_fixture_passes_nested_corpus_check() {
        let spec = ScaleExperiment::load("experiments/smoke-mvp.toml").unwrap();
        check(spec).unwrap();
    }

    #[test]
    fn check_rejects_training_corpus_that_is_not_a_prefix() {
        let directory = tempfile::tempdir().unwrap();
        let tokenizer = directory.path().join("tokenizer.txt");
        let validation = directory.path().join("validation.txt");
        let small = directory.path().join("small.txt");
        let broad = directory.path().join("broad.txt");
        fs::write(&tokenizer, "abcdefghijklmnopqrstuvwxyz .").unwrap();
        fs::write(&validation, "held out validation text xx").unwrap();
        fs::write(&small, "aaaaaaaaaaaaaaaa").unwrap();
        fs::write(&broad, "bbbbbbbbbbbbbbbbbbbbbbbb").unwrap();
        let spec = ScaleExperiment {
            name: "non-nested".into(),
            output_dir: directory.path().join("runs"),
            tokenizer_corpus: tokenizer,
            validation_corpus: validation,
            in_domain_validation_corpus: None,
            model: ModelConfig {
                context_length: 8,
                d_model: 16,
                n_heads: 4,
                n_layers: 1,
                d_ff: 32,
                expected_parameters: None,
            },
            training: ScaleTrainingConfig {
                batch_size: 2,
                eval_interval: 10,
                eval_batches: 1,
                learning_rate: 0.001,
                weight_decay: 0.01,
                seed: None,
                seeds: vec![11],
                sample_tokens: 4,
            },
            prompts: Vec::new(),
            nested_training_corpora: true,
            runs: vec![
                ScaleRunConfig {
                    name: "small".into(),
                    train_corpus: small,
                    target_tokens_per_parameter: 0.5,
                },
                ScaleRunConfig {
                    name: "broad".into(),
                    train_corpus: broad,
                    target_tokens_per_parameter: 0.5,
                },
            ],
        };
        let error = check(spec).unwrap_err();
        assert!(error.to_string().contains("not a prefix"));
    }

    fn nested_spec(
        directory: &std::path::Path,
        in_domain: Option<&str>,
        broad: &str,
    ) -> ScaleExperiment {
        let tokenizer = directory.join("tokenizer.txt");
        let validation = directory.join("validation.txt");
        let small = directory.join("small.txt");
        let broad_path = directory.join("broad.txt");
        fs::write(&tokenizer, "abcdefghijklmnopqrstuvwxyz .!").unwrap();
        fs::write(&validation, "held out validation text xx").unwrap();
        fs::write(&small, &broad[..16]).unwrap();
        fs::write(&broad_path, broad).unwrap();
        let in_domain_validation_corpus = in_domain.map(|text| {
            let path = directory.join("in-domain.txt");
            fs::write(&path, text).unwrap();
            path
        });
        ScaleExperiment {
            name: "nested".into(),
            output_dir: directory.join("runs"),
            tokenizer_corpus: tokenizer,
            validation_corpus: validation,
            in_domain_validation_corpus,
            model: ModelConfig {
                context_length: 8,
                d_model: 16,
                n_heads: 4,
                n_layers: 1,
                d_ff: 32,
                expected_parameters: None,
            },
            training: ScaleTrainingConfig {
                batch_size: 2,
                eval_interval: 10,
                eval_batches: 1,
                learning_rate: 0.001,
                weight_decay: 0.01,
                seed: None,
                seeds: vec![11],
                sample_tokens: 4,
            },
            prompts: Vec::new(),
            nested_training_corpora: true,
            runs: vec![
                ScaleRunConfig {
                    name: "small".into(),
                    train_corpus: small,
                    target_tokens_per_parameter: 0.5,
                },
                ScaleRunConfig {
                    name: "broad".into(),
                    train_corpus: broad_path,
                    target_tokens_per_parameter: 0.5,
                },
            ],
        }
    }

    #[test]
    fn check_rejects_in_domain_text_leaked_into_training() {
        let directory = tempfile::tempdir().unwrap();
        let spec = nested_spec(
            directory.path(),
            Some("extra heldout window!!"),
            "abcdefghijklmnop extra heldout window!! more",
        );
        let error = check(spec).unwrap_err();
        assert!(error.to_string().contains("in-domain validation"));
    }

    #[test]
    fn matching_completed_run_is_skipped() {
        let directory = tempfile::tempdir().unwrap();
        let spec = nested_spec(
            directory.path(),
            Some("heldout eval window text!!"),
            "abcdefghijklmnop qrstuvwxyz more tokens here",
        );
        let checked = check(spec).unwrap();
        let run = &checked.runs[0];
        let run_dir = directory.path().join("completed");
        fs::create_dir_all(&run_dir).unwrap();
        fs::write(run_dir.join("model.safetensors"), b"weights").unwrap();
        fs::write(run_dir.join("metrics.jsonl"), "{}\n").unwrap();
        fs::write(run_dir.join("samples.json"), "[]").unwrap();
        fs::write(run_dir.join("dataset.json"), "{}").unwrap();
        let summary = RunSummary {
            experiment: checked.spec.name.clone(),
            run: run.config.name.clone(),
            seed: 11,
            parameter_count: checked.parameter_count,
            train_corpus_tokens: run.train_tokens.len(),
            validation_corpus_tokens: checked.validation_tokens.len(),
            target_processed_tokens: run.target_processed_tokens,
            actual_processed_tokens: run.actual_processed_tokens,
            tokens_per_parameter: 0.5,
            effective_epochs: run.effective_epochs,
            best_validation_nll: 1.0,
            best_step: 0,
            final_train_nll: 1.0,
            final_validation_nll: 1.0,
            final_generalization_gap: 0.0,
            final_in_domain_validation_nll: None,
            final_in_domain_generalization_gap: None,
            elapsed_seconds: 0.0,
            initial_weights_sha256: "init".into(),
            tokenizer_sha256: checked.tokenizer_sha256.clone(),
            train_sha256: run.train_sha256.clone(),
            validation_sha256: checked.validation_sha256.clone(),
            in_domain_validation_sha256: checked.in_domain_validation_sha256.clone(),
            control_sha256: control_sha256(&checked).unwrap(),
        };
        fs::write(
            run_dir.join("summary.json"),
            serde_json::to_vec_pretty(&summary).unwrap(),
        )
        .unwrap();
        assert!(should_skip_completed_run(&run_dir, &checked, run, 11, "init").unwrap());
        fs::remove_file(run_dir.join("samples.json")).unwrap();
        assert!(!should_skip_completed_run(&run_dir, &checked, run, 11, "init").unwrap());
        assert!(!should_skip_completed_run(&run_dir, &checked, run, 42, "init").unwrap());
    }

    #[test]
    fn incomplete_or_corrupt_run_is_retrained() {
        let directory = tempfile::tempdir().unwrap();
        let spec = nested_spec(
            directory.path(),
            Some("heldout eval window text!!"),
            "abcdefghijklmnop qrstuvwxyz more tokens here",
        );
        let checked = check(spec).unwrap();
        let run = &checked.runs[0];
        let run_dir = directory.path().join("incomplete");
        fs::create_dir_all(&run_dir).unwrap();
        fs::write(run_dir.join("model.safetensors"), b"weights").unwrap();
        fs::write(run_dir.join("metrics.jsonl"), "{}\n").unwrap();
        fs::write(run_dir.join("dataset.json"), "{}").unwrap();
        fs::write(run_dir.join("summary.json"), b"{not complete").unwrap();

        assert!(!should_skip_completed_run(&run_dir, &checked, run, 11, "init").unwrap());

        fs::write(run_dir.join("samples.json"), "[]").unwrap();
        assert!(!should_skip_completed_run(&run_dir, &checked, run, 11, "init").unwrap());
    }
}
