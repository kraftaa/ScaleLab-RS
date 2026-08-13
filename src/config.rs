use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub data: DataConfig,
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub output_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub path: String,
    #[serde(default = "default_train_fraction")]
    pub train_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub context_length: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    #[serde(default)]
    pub expected_parameters: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub steps: usize,
    pub eval_interval: usize,
    pub eval_batches: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_train_fraction() -> f64 {
    0.9
}

fn default_seed() -> u64 {
    42
}

impl ExperimentConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        let config: Self = toml::from_str(&text)
            .with_context(|| format!("failed to parse config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            (0.0..1.0).contains(&self.data.train_fraction),
            "data.train_fraction must be between 0 and 1"
        );
        anyhow::ensure!(
            self.model.context_length > 0,
            "context_length must be positive"
        );
        anyhow::ensure!(self.model.n_heads > 0, "n_heads must be positive");
        anyhow::ensure!(
            self.model.d_model.is_multiple_of(self.model.n_heads),
            "d_model must be divisible by n_heads"
        );
        anyhow::ensure!(self.training.batch_size > 0, "batch_size must be positive");
        anyhow::ensure!(self.training.steps > 0, "steps must be positive");
        anyhow::ensure!(
            self.training.eval_interval > 0,
            "eval_interval must be positive"
        );
        anyhow::ensure!(
            self.training.eval_batches > 0,
            "eval_batches must be positive"
        );
        Ok(())
    }
}
