use anyhow::Result;
use clap::{Parser, Subcommand};
use scalelab_rs::{config::ExperimentConfig, experiment, report, train};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "scalelab")]
#[command(about = "A readable, reproducible GPT learning lab built with Candle")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a model from a TOML experiment configuration.
    Train { config: PathBuf },
    /// Validate and explain a controlled multi-run experiment.
    Check { experiment: PathBuf },
    /// Parse and validate a legacy single-run configuration.
    CheckRun { config: PathBuf },
    /// Check and execute every run in a controlled experiment.
    Experiment { experiment: PathBuf },
    /// Generate a self-contained comparison report from completed runs.
    Report { experiment_dir: PathBuf },
    /// Normalize two downloaded Gutenberg books into controlled corpora.
    PrepareCorpus {
        #[arg(long)]
        train_raw: PathBuf,
        #[arg(long)]
        validation_raw: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long, default_value_t = 25_184)]
        small_tokens: usize,
    },
    /// Load a trained run and greedily generate text from a prompt.
    Sample {
        run_dir: PathBuf,
        prompt: String,
        #[arg(long, default_value_t = 100)]
        tokens: usize,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Train { config } => train::run(&ExperimentConfig::load(config)?),
        Command::Check { experiment: path } => {
            let checked = experiment::check(experiment::ScaleExperiment::load(path)?)?;
            experiment::print_check(&checked);
            Ok(())
        }
        Command::CheckRun { config } => {
            let config = ExperimentConfig::load(config)?;
            println!("{config:#?}");
            Ok(())
        }
        Command::Experiment { experiment: path } => {
            let checked = experiment::check(experiment::ScaleExperiment::load(path)?)?;
            experiment::print_check(&checked);
            experiment::run(checked)
        }
        Command::Report { experiment_dir } => report::generate(&experiment_dir),
        Command::PrepareCorpus {
            train_raw,
            validation_raw,
            output_dir,
            small_tokens,
        } => scalelab_rs::corpus::prepare_gutenberg(
            &train_raw,
            &validation_raw,
            &output_dir,
            small_tokens,
        ),
        Command::Sample {
            run_dir,
            prompt,
            tokens,
        } => {
            println!("{}", scalelab_rs::sample::run(&run_dir, &prompt, tokens)?);
            Ok(())
        }
    }
}
