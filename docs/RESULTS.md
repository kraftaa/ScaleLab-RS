# Three-seed corpus-reuse result

## Result

The controlled experiment supports the MVP hypothesis at this scale:
processing the same token budget from a broader corpus generalized better than
replaying a short corpus.

The primary evaluation is now in-domain. The final six chapters of *Pride and
Prejudice* were removed before constructing either training corpus and used
only for evaluation. A second evaluation uses *Alice's Adventures in
Wonderland* to check whether the result also holds across books.

| Condition | Corpus chars | Processed chars | Effective epochs | Train NLL | In-domain NLL | In-domain PPL | In-domain gap | Out-of-domain NLL | Out-of-domain PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Broad 20× | 651,530 | 550,656 | 0.85 | 3.1880 | **3.2038** | **24.63** | **0.0158** | **3.1791** | **24.02** |
| Repeated 20× | 25,184 | 550,656 | 21.87 | **3.1713** | 3.2298 | 25.27 | 0.0585 | 3.2027 | 24.60 |

The entries are means over paired seeds 11, 42, and 73. Both conditions used
the same 27,520-parameter model and processed 20.009 character tokens per
parameter. Aggregate perplexity is `exp(mean NLL)`, the geometric mean of the
per-seed perplexities.

The broad condition's final held-out NLL was lower in every seed on both
evaluations:

| Seed | Broad in-domain NLL | Repeated in-domain NLL | Repeated − broad | Broad out-of-domain NLL | Repeated out-of-domain NLL | Repeated − broad |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 3.2618 | 3.2870 | +0.0252 | 3.2409 | 3.2651 | +0.0243 |
| 42 | 3.2101 | 3.2372 | +0.0271 | 3.1794 | 3.2031 | +0.0237 |
| 73 | 3.1395 | 3.1651 | +0.0257 | 3.1169 | 3.1399 | +0.0230 |

The mean paired difference was +0.0260 NLL in-domain and +0.0236 NLL
out-of-domain. The repeated condition's in-domain train/validation gap was
0.0585 versus 0.0158 for the broad condition—approximately 3.7 times larger.
The repeated condition achieved the lower training NLL while producing the
higher held-out NLL, which is the expected signature of increased corpus reuse
rather than increased information exposure.

## What was controlled

Within each seed, the two 20× conditions shared the exact initial SafeTensor,
architecture, tokenizer, optimizer, learning rate, batch size, context length,
processed-token budget, evaluation positions, and validation text. Official
evaluation uses 160 fixed, evenly strided context windows per corpus instead of
resampling random batches. The preflight and report reject mismatched controls.

The 25,184-character repeated corpus is the exact prefix of the
651,530-character broad corpus. Both are drawn from *Pride and Prejudice* after
the final six chapters have been held out. Source, normalized-corpus,
tokenizer, validation, evaluation-window, and initial-weight hashes are stored
with the artifacts.

## Limits

- These are character tokens, not BPE tokens, and this is a tiny CPU model.
- Corpus size is not the same as semantic diversity.
- The short corpus is one prefix of one book; another segment could differ.
- Train NLL is evaluated on each condition's own training corpus, so the
  held-out NLL comparisons are stronger evidence than comparing gaps alone.
- Three paired seeds establish repeatability for this demo, not a universal
  scaling-law result.
- The experiment illustrates what a processed-tokens-per-parameter number can
  hide. It neither proves nor disproves Chinchilla-style scaling laws.

## Reproduce

```sh
./scripts/prepare-mvp-data.sh
cargo run -- check experiments/mvp.toml
cargo run --release -- experiment experiments/mvp.toml
cargo run --release -- report runs/mvp
```

Open `runs/mvp/report/index.html` for the full report. The LinkedIn-ready chart
is `runs/mvp/report/headline-comparison.svg`, and the raw tables are
`runs/mvp/report/comparison.csv` and `runs/mvp/report/paired-comparison.csv`.
The measured tables used by the public site are also checked in as
[`comparison.csv`](assets/comparison.csv) and
[`paired-comparison.csv`](assets/paired-comparison.csv).
