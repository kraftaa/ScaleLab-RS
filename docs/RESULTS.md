# Three-seed corpus-reuse result

## Result

The controlled experiment supports the MVP hypothesis at this scale:
processing the same token budget from a broader corpus generalized better than
replaying a short corpus.

| Condition | Corpus characters | Processed characters | Effective epochs | Train NLL | Train PPL | Validation NLL | Validation PPL | Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Broad 20x | 719,847 | 550,656 | 0.76 | 3.1758 | 23.95 | 3.1952 | 24.41 | 0.0193 |
| Repeated 20x | 25,184 | 550,656 | 21.87 | 3.1698 | 23.80 | 3.2165 | 24.94 | 0.0468 |

The entries are means over paired seeds 11, 42, and 73. Both conditions used
the same 27,520-parameter model and processed 20.009 character tokens per
parameter. Aggregate perplexity is `exp(mean NLL)`, the geometric mean of the
per-seed perplexities. The broad condition's final validation NLL was lower in
every seed:

| Seed | Broad validation NLL | Repeated validation NLL | Repeated minus broad |
|---:|---:|---:|---:|
| 11 | 3.2607 | 3.2816 | +0.0208 |
| 42 | 3.1940 | 3.2168 | +0.0228 |
| 73 | 3.1307 | 3.1512 | +0.0205 |

The mean paired difference was +0.0214 NLL. More importantly for the headline
chart, the repeated condition's final train/validation gap was 0.0468 versus
0.0193 for the broad condition. Its training NLL was slightly lower even while
its validation NLL was higher, which is the expected signature of increased
reuse rather than increased information.

## What was controlled

Within each seed, both 20x conditions shared the exact initial SafeTensor,
architecture, tokenizer, optimizer, learning rate, batch size, context length,
processed-token budget, fixed validation batches, and validation text. The
preflight and report both reject mismatched controls. Source, normalized-corpus,
tokenizer, validation, and initial-weight hashes are stored with the artifacts.

The 25,184-character repeated corpus is the exact prefix of the 719,847-character
broad corpus. Training uses *Pride and Prejudice*; validation uses the separate
*Alice's Adventures in Wonderland* corpus. This makes available corpus
positions before reuse the intended independent variable.

## Limits

- These are character tokens, not BPE tokens, and this is a tiny CPU model.
- Corpus size is not the same as semantic diversity.
- The validation book differs stylistically from the training book; that shift
  is shared by both conditions but affects the absolute NLL.
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
is `runs/mvp/report/headline-comparison.svg`, and the raw comparison table is
`runs/mvp/report/comparison.csv`.
