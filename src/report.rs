use crate::experiment::{ExperimentMetric, RunSummary, SampleCheckpoint};
use anyhow::{Context, Result};
use std::{collections::BTreeMap, fs, path::Path};

struct RunReport {
    summary: RunSummary,
    metrics: Vec<ExperimentMetric>,
    samples: Vec<SampleCheckpoint>,
}

type ChartSeries = (String, bool, Vec<(f64, f64)>);

pub fn generate(experiment_dir: &Path) -> Result<()> {
    let mut runs = Vec::new();
    let mut completed = Vec::new();
    collect_completed_runs(experiment_dir, &mut completed)?;
    for path in completed {
        let summary: RunSummary = serde_json::from_slice(&fs::read(path.join("summary.json"))?)?;
        let metrics = read_metrics(&path.join("metrics.jsonl"))?;
        let samples = serde_json::from_slice(&fs::read(path.join("samples.json"))?)?;
        runs.push(RunReport {
            summary,
            metrics,
            samples,
        });
    }
    anyhow::ensure!(
        runs.len() >= 2,
        "report requires at least two completed runs"
    );
    runs.sort_by(|left, right| {
        left.summary
            .run
            .cmp(&right.summary.run)
            .then(left.summary.seed.cmp(&right.summary.seed))
    });
    validate_controls(&runs)?;

    let report_dir = experiment_dir.join("report");
    fs::create_dir_all(&report_dir)?;
    let loss_svg = loss_chart(&runs);
    let gap_svg = gap_chart(&runs);
    let headline_svg = aggregate_gap_chart(&runs);
    fs::write(report_dir.join("headline-comparison.svg"), &headline_svg)?;
    fs::write(report_dir.join("training-validation-nll.svg"), &loss_svg)?;
    fs::write(report_dir.join("generalization-gap.svg"), &gap_svg)?;
    fs::write(report_dir.join("comparison.csv"), comparison_csv(&runs))?;
    fs::write(
        report_dir.join("index.html"),
        html_report(&runs, &headline_svg, &loss_svg, &gap_svg),
    )?;
    println!(
        "Report written to {}",
        report_dir.join("index.html").display()
    );
    Ok(())
}

fn collect_completed_runs(directory: &Path, output: &mut Vec<std::path::PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)
        .with_context(|| format!("failed to read {}", directory.display()))?
    {
        let path = entry?.path();
        if !path.is_dir() || path.file_name().is_some_and(|name| name == "report") {
            continue;
        }
        if path.join("summary.json").exists() {
            output.push(path);
        } else {
            collect_completed_runs(&path, output)?;
        }
    }
    Ok(())
}

fn validate_controls(runs: &[RunReport]) -> Result<()> {
    let first = &runs[0].summary;
    for run in runs {
        anyhow::ensure!(
            run.summary.parameter_count == first.parameter_count,
            "run {} has a different parameter count",
            run.summary.run
        );
        anyhow::ensure!(
            run.summary.tokenizer_sha256 == first.tokenizer_sha256,
            "run {} used a different tokenizer",
            run.summary.run
        );
        anyhow::ensure!(
            run.summary.validation_sha256 == first.validation_sha256,
            "run {} used a different validation corpus",
            run.summary.run
        );
        anyhow::ensure!(
            run.summary.control_sha256 == first.control_sha256,
            "run {} used different model or training controls",
            run.summary.run
        );
    }
    let mut weights_by_seed = BTreeMap::<u64, &str>::new();
    let mut run_names_by_seed = BTreeMap::<u64, Vec<&str>>::new();
    for run in runs {
        match weights_by_seed.get(&run.summary.seed) {
            Some(hash) => anyhow::ensure!(
                *hash == run.summary.initial_weights_sha256,
                "run {} seed {} did not use paired initial weights",
                run.summary.run,
                run.summary.seed
            ),
            None => {
                weights_by_seed.insert(run.summary.seed, &run.summary.initial_weights_sha256);
            }
        }
        run_names_by_seed
            .entry(run.summary.seed)
            .or_default()
            .push(&run.summary.run);
    }
    let expected_names: std::collections::BTreeSet<_> =
        runs.iter().map(|run| run.summary.run.as_str()).collect();
    for (seed, names) in run_names_by_seed {
        let names: std::collections::BTreeSet<_> = names.into_iter().collect();
        anyhow::ensure!(
            names == expected_names,
            "seed {seed} does not contain every configured run"
        );
    }
    Ok(())
}

fn read_metrics(path: &Path) -> Result<Vec<ExperimentMetric>> {
    let text = fs::read_to_string(path)?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).map_err(Into::into))
        .collect()
}

fn comparison_csv(runs: &[RunReport]) -> String {
    let mut csv = String::from(
        "run,seed,parameters,train_corpus_tokens,processed_tokens,tokens_per_parameter,effective_epochs,best_validation_nll,final_train_nll,final_train_perplexity,final_validation_nll,final_validation_perplexity,generalization_gap,elapsed_seconds\n",
    );
    for run in runs {
        let s = &run.summary;
        csv.push_str(&format!(
            "{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3}\n",
            s.run,
            s.seed,
            s.parameter_count,
            s.train_corpus_tokens,
            s.actual_processed_tokens,
            s.tokens_per_parameter,
            s.effective_epochs,
            s.best_validation_nll,
            s.final_train_nll,
            perplexity(s.final_train_nll as f64),
            s.final_validation_nll,
            perplexity(s.final_validation_nll as f64),
            s.final_generalization_gap,
            s.elapsed_seconds
        ));
    }
    csv
}

fn html_report(runs: &[RunReport], headline_svg: &str, loss_svg: &str, gap_svg: &str) -> String {
    let grouped = grouped_runs(runs);
    let rows = grouped
        .iter()
        .map(|(name, replications)| {
            let s = &replications[0].summary;
            let train_mean = mean(
                replications
                    .iter()
                    .map(|run| run.summary.final_train_nll as f64),
            );
            let validation_values = replications
                .iter()
                .map(|run| run.summary.final_validation_nll as f64)
                .collect::<Vec<_>>();
            let validation_mean = mean(validation_values.iter().copied());
            let validation_sd = standard_deviation(&validation_values);
            let gap_mean = mean(
                replications
                    .iter()
                    .map(|run| run.summary.final_generalization_gap as f64),
            );
            format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.2}</td><td>{:.2}</td><td>{:.4}</td><td>{:.2}</td><td>{:.4} ± {:.4}</td><td>{:.2}</td><td>{:.4}</td></tr>",
                escape(name),
                replications.len(),
                s.parameter_count,
                s.train_corpus_tokens,
                s.actual_processed_tokens,
                s.tokens_per_parameter,
                s.effective_epochs,
                train_mean,
                perplexity(train_mean),
                validation_mean,
                validation_sd,
                perplexity(validation_mean),
                gap_mean
            )
        })
        .collect::<String>();
    let experiment = escape(&runs[0].summary.experiment);
    let sample_sections = runs
        .iter()
        .map(|run| {
            let final_checkpoint = run.samples.last();
            let samples = final_checkpoint
                .map(|checkpoint| {
                    checkpoint
                        .samples
                        .iter()
                        .map(|sample| {
                            format!(
                                "<h4>Prompt: <code>{}</code></h4><pre>{}</pre>",
                                escape(&sample.prompt),
                                escape(&sample.text)
                            )
                        })
                        .collect::<String>()
                })
                .unwrap_or_else(|| "<p>No samples recorded.</p>".to_string());
            format!(
                "<section><h3>{} — seed {}</h3>{samples}</section>",
                escape(&run.summary.run),
                run.summary.seed
            )
        })
        .collect::<String>();
    format!(
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScaleLab-RS — {experiment}</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe3ef;--paper:#fff;--wash:#f4f7fb;--accent:#2563eb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:48px 24px}} h1{{font-size:38px;margin:0 0 4px}} h2{{margin-top:42px}} .lede{{color:var(--muted);font-size:18px;max-width:780px}}
.card{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:22px;margin:18px 0;box-shadow:0 4px 18px #1720330a;overflow:auto}}
table{{border-collapse:collapse;width:100%;min-width:1100px}} th,td{{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line)}} th:first-child,td:first-child{{text-align:left}} th{{font-size:12px;text-transform:uppercase;color:var(--muted)}}
.valid{{display:inline-block;color:#166534;background:#dcfce7;border-radius:999px;padding:5px 10px;font-weight:700}} code{{background:#e8eef8;padding:2px 5px;border-radius:4px}}
pre{{white-space:pre-wrap;background:#0f172a;color:#e2e8f0;border-radius:8px;padding:14px;overflow:auto}} section+section{{border-top:1px solid var(--line);margin-top:24px;padding-top:12px}}
svg{{max-width:100%;height:auto}} footer{{color:var(--muted);margin-top:38px}}
</style></head><body><main>
<span class="valid">CONTROL CHECKS PASSED</span><h1>ScaleLab-RS</h1>
<p class="lede">{experiment}: equal processed-token budgets do not necessarily imply equal corpus exposure. Runs are paired within each seed and share architecture, initial weights, tokenizer, and validation data.</p>
<h2>Run comparison</h2><div class="card"><table><thead><tr><th>Run</th><th>Seeds</th><th>Parameters</th><th>Corpus tokens</th><th>Processed tokens</th><th>Tok/param</th><th>Effective epochs</th><th>Train NLL mean</th><th>Train PPL</th><th>Validation NLL mean ± SD</th><th>Validation PPL</th><th>Gap mean</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>Headline comparison</h2><p class="lede">Mean generalization gap across paired seeds; vertical bars show the observed minimum-to-maximum range.</p><div class="card">{headline_svg}</div>
<details><summary>Detailed per-seed charts</summary><h2>Training and validation NLL</h2><div class="card">{loss_svg}</div>
<h2>Generalization gap</h2><p class="lede">The gap is validation NLL minus training NLL. A growing positive gap is evidence that training performance is improving faster than held-out performance.</p><div class="card">{gap_svg}</div></details>
<h2>Final fixed-prompt samples</h2><div class="card">{sample_sections}</div>
<h2>Interpretation discipline</h2><div class="card"><p>This experiment measures <strong>corpus reuse</strong>, not semantic diversity. Effective epochs are exposure equivalents because training samples random context windows. Results at this scale do not prove or disprove large-model scaling laws.</p></div>
<footer>Generated locally by ScaleLab-RS. NLL is measured in nats per token; perplexity is exp(NLL).</footer>
</main></body></html>"#
    )
}

fn grouped_runs(runs: &[RunReport]) -> BTreeMap<&str, Vec<&RunReport>> {
    let mut grouped = BTreeMap::<&str, Vec<&RunReport>>::new();
    for run in runs {
        grouped.entry(&run.summary.run).or_default().push(run);
    }
    grouped
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let values = values.collect::<Vec<_>>();
    values.iter().sum::<f64>() / values.len() as f64
}

fn standard_deviation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let average = mean(values.iter().copied());
    let variance = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
}

fn perplexity(nll: f64) -> f64 {
    nll.exp()
}

fn aggregate_gap_chart(runs: &[RunReport]) -> String {
    const COLORS: [&str; 6] = [
        "#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2",
    ];
    let matched_budget = runs
        .iter()
        .map(|run| run.summary.actual_processed_tokens)
        .max()
        .unwrap_or(0);
    let headline_runs = runs
        .iter()
        .filter(|run| run.summary.actual_processed_tokens == matched_budget)
        .collect::<Vec<_>>();
    let mut grouped = BTreeMap::<&str, Vec<&RunReport>>::new();
    for run in headline_runs {
        grouped.entry(&run.summary.run).or_default().push(run);
    }
    let mut series = Vec::<(String, Vec<(f64, f64, f64, f64)>)>::new();
    for (name, replications) in grouped {
        let point_count = replications
            .iter()
            .map(|run| run.metrics.len())
            .min()
            .unwrap_or(0);
        let mut points = Vec::with_capacity(point_count);
        for index in 0..point_count {
            let x = replications[0].metrics[index].processed_tokens as f64;
            let values = replications
                .iter()
                .map(|run| run.metrics[index].generalization_gap as f64)
                .collect::<Vec<_>>();
            let average = mean(values.iter().copied());
            let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
            let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            points.push((x, average, minimum, maximum));
        }
        series.push((name.to_string(), points));
    }
    let all = series
        .iter()
        .flat_map(|(_, points)| points.iter())
        .collect::<Vec<_>>();
    if all.is_empty() {
        return "<svg></svg>".to_string();
    }
    let width = 940.0;
    let height = 440.0;
    let left = 70.0;
    let right = 24.0;
    let top = 50.0;
    let bottom = 64.0;
    let plot_w = width - left - right;
    let plot_h = height - top - bottom;
    let x_max = all.iter().map(|(x, _, _, _)| *x).fold(1.0f64, f64::max);
    let raw_min = all
        .iter()
        .map(|(_, _, minimum, _)| *minimum)
        .fold(f64::INFINITY, f64::min);
    let raw_max = all
        .iter()
        .map(|(_, _, _, maximum)| *maximum)
        .fold(f64::NEG_INFINITY, f64::max);
    let padding = ((raw_max - raw_min) * 0.12).max(0.05);
    let y_min = raw_min - padding;
    let y_max = raw_max + padding;
    let sx = |x: f64| left + x / x_max * plot_w;
    let sy = |y: f64| top + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h;
    let mut svg = format!("<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Mean generalization gap across seeds\"><rect width=\"100%\" height=\"100%\" fill=\"white\"/><text x=\"{left}\" y=\"22\" fill=\"#172033\" font-size=\"16\" font-weight=\"700\">Same processed tokens, different corpus exposure</text>");
    for tick in 0..=4 {
        let fraction = tick as f64 / 4.0;
        let y = top + fraction * plot_h;
        let value = y_max - fraction * (y_max - y_min);
        svg.push_str(&format!("<line x1=\"{left}\" y1=\"{y}\" x2=\"{}\" y2=\"{y}\" stroke=\"#e2e8f0\"/><text x=\"{}\" y=\"{}\" text-anchor=\"end\" fill=\"#64748b\" font-size=\"12\">{value:.2}</text>", width-right, left-8.0, y+4.0));
    }
    if y_min <= 0.0 && y_max >= 0.0 {
        let zero = sy(0.0);
        svg.push_str(&format!("<line x1=\"{left}\" y1=\"{zero}\" x2=\"{}\" y2=\"{zero}\" stroke=\"#94a3b8\" stroke-width=\"1.5\"/>", width-right));
    }
    for tick in 0..=4 {
        let fraction = tick as f64 / 4.0;
        let x = left + fraction * plot_w;
        svg.push_str(&format!("<text x=\"{x}\" y=\"{}\" text-anchor=\"middle\" fill=\"#64748b\" font-size=\"12\">{}</text>", height-34.0, compact_number(fraction*x_max)));
    }
    for (index, (name, points)) in series.iter().enumerate() {
        let color = COLORS[index % COLORS.len()];
        let coordinates = points
            .iter()
            .map(|(x, average, _, _)| format!("{:.2},{:.2}", sx(*x), sy(*average)))
            .collect::<Vec<_>>()
            .join(" ");
        for (x, average, minimum, maximum) in points {
            svg.push_str(&format!("<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{color}\" stroke-width=\"5\" opacity=\"0.18\"/><circle cx=\"{}\" cy=\"{}\" r=\"2.5\" fill=\"{color}\"/>", sx(*x), sy(*minimum), sx(*x), sy(*maximum), sx(*x), sy(*average)));
        }
        svg.push_str(&format!("<polyline points=\"{coordinates}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"3\"/><line x1=\"{}\" y1=\"38\" x2=\"{}\" y2=\"38\" stroke=\"{color}\" stroke-width=\"3\"/><text x=\"{}\" y=\"42\" fill=\"#334155\" font-size=\"12\">{}</text>", left+index as f64*270.0, left+25.0+index as f64*270.0, left+32.0+index as f64*270.0, escape(name)));
    }
    svg.push_str(&format!("<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" fill=\"#334155\" font-size=\"13\">Processed tokens</text><text transform=\"translate(17 {}) rotate(-90)\" text-anchor=\"middle\" fill=\"#334155\" font-size=\"13\">Validation NLL − train NLL</text></svg>", left+plot_w/2.0, height-8.0, top+plot_h/2.0));
    svg
}

fn loss_chart(runs: &[RunReport]) -> String {
    let mut series = Vec::new();
    for run in runs {
        series.push((
            format!("{} s{} train", run.summary.run, run.summary.seed),
            false,
            run.metrics
                .iter()
                .map(|m| (m.processed_tokens as f64, m.train_nll as f64))
                .collect(),
        ));
        series.push((
            format!("{} s{} validation", run.summary.run, run.summary.seed),
            true,
            run.metrics
                .iter()
                .map(|m| (m.processed_tokens as f64, m.validation_nll as f64))
                .collect(),
        ));
    }
    svg_chart("NLL (nats/token)", &series)
}

fn gap_chart(runs: &[RunReport]) -> String {
    let series = runs
        .iter()
        .map(|run| {
            (
                format!("{} s{}", run.summary.run, run.summary.seed),
                false,
                run.metrics
                    .iter()
                    .map(|m| (m.processed_tokens as f64, m.generalization_gap as f64))
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    svg_chart("Validation NLL − train NLL", &series)
}

fn svg_chart(y_label: &str, series: &[ChartSeries]) -> String {
    const COLORS: [&str; 6] = [
        "#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2",
    ];
    let width = 940.0;
    let height = 420.0;
    let left = 70.0;
    let right = 24.0;
    let legend_rows = series.len().div_ceil(3);
    let top = 26.0 + legend_rows as f64 * 20.0;
    let bottom = 64.0;
    let all = series
        .iter()
        .flat_map(|(_, _, points)| points.iter())
        .collect::<Vec<_>>();
    if all.is_empty() {
        return "<svg></svg>".to_string();
    }
    let x_max = all.iter().map(|(x, _)| *x).fold(1.0f64, f64::max);
    let y_min_raw = all.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let y_max_raw = all
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);
    let padding = ((y_max_raw - y_min_raw) * 0.1).max(0.05);
    let y_min = y_min_raw - padding;
    let y_max = y_max_raw + padding;
    let plot_w = width - left - right;
    let plot_h = height - top - bottom;
    let sx = |x: f64| left + (x / x_max) * plot_w;
    let sy = |y: f64| top + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h;
    let mut svg = format!(
        "<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"{}\"><rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
        escape(y_label)
    );
    for (index, (name, dashed, _)) in series.iter().enumerate() {
        let color = COLORS[(index / if series.len() > 3 { 2 } else { 1 }) % COLORS.len()];
        let legend_x = left + (index % 3) as f64 * 270.0;
        let legend_y = 18.0 + (index / 3) as f64 * 20.0;
        let dash = if *dashed {
            " stroke-dasharray=\"7 5\""
        } else {
            ""
        };
        svg.push_str(&format!("<line x1=\"{legend_x}\" y1=\"{legend_y}\" x2=\"{}\" y2=\"{legend_y}\" stroke=\"{color}\" stroke-width=\"2.5\"{dash}/><text x=\"{}\" y=\"{}\" fill=\"#334155\" font-size=\"12\">{}</text>", legend_x+25.0, legend_x+32.0, legend_y+4.0, escape(name)));
    }
    for tick in 0..=4 {
        let fraction = tick as f64 / 4.0;
        let y = top + fraction * plot_h;
        let value = y_max - fraction * (y_max - y_min);
        svg.push_str(&format!("<line x1=\"{left}\" y1=\"{y}\" x2=\"{}\" y2=\"{y}\" stroke=\"#e2e8f0\"/><text x=\"{}\" y=\"{}\" text-anchor=\"end\" fill=\"#64748b\" font-size=\"12\">{value:.2}</text>", width-right, left-8.0, y+4.0));
    }
    for tick in 0..=4 {
        let fraction = tick as f64 / 4.0;
        let x = left + fraction * plot_w;
        let value = fraction * x_max;
        svg.push_str(&format!("<text x=\"{x}\" y=\"{}\" text-anchor=\"middle\" fill=\"#64748b\" font-size=\"12\">{}</text>", height-34.0, compact_number(value)));
    }
    svg.push_str(&format!("<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" fill=\"#334155\" font-size=\"13\">Processed tokens</text><text transform=\"translate(17 {}) rotate(-90)\" text-anchor=\"middle\" fill=\"#334155\" font-size=\"13\">{}</text>", left+plot_w/2.0, height-8.0, top+plot_h/2.0, escape(y_label)));
    for (index, (_name, dashed, points)) in series.iter().enumerate() {
        let color = COLORS[(index / if series.len() > 3 { 2 } else { 1 }) % COLORS.len()];
        let coordinates = points
            .iter()
            .map(|(x, y)| format!("{:.2},{:.2}", sx(*x), sy(*y)))
            .collect::<Vec<_>>()
            .join(" ");
        let dash = if *dashed {
            " stroke-dasharray=\"7 5\""
        } else {
            ""
        };
        svg.push_str(&format!("<polyline points=\"{coordinates}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2.5\"{dash}/>"));
    }
    svg.push_str("</svg>");
    svg
}

fn compact_number(value: f64) -> String {
    if value >= 1_000_000.0 {
        format!("{:.1}M", value / 1_000_000.0)
    } else if value >= 1_000.0 {
        format!("{:.0}K", value / 1_000.0)
    } else {
        format!("{value:.0}")
    }
}

fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::perplexity;

    #[test]
    fn perplexity_is_the_exponential_of_nll() {
        assert_eq!(perplexity(0.0), 1.0);
        assert!((perplexity(10.0_f64.ln()) - 10.0).abs() < 1e-12);
    }
}
