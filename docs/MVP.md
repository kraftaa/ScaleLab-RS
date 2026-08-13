# ScaleLab-RS MVP

## Question

When two identical GPT-style Transformers process the same number of training
tokens, how does repeatedly reusing a small corpus compare with drawing those
tokens from a broader corpus?

This measures corpus reuse. It does not formally measure semantic diversity and
does not prove or disprove the Chinchilla scaling laws.

## Quantities

```text
train_corpus_tokens
    Token positions available in the training corpus before sampling.

processed_tokens
    Tokens consumed by optimizer steps: steps × batch size × context length.

tokens_per_parameter
    processed_tokens / exact instantiated parameter count.

effective_epochs
    processed_tokens / train_corpus_tokens.

observed_train_token_types
    Number of frozen-tokenizer IDs that appear in a training corpus.
```

Effective epochs are exposure equivalents. Random context sampling does not
guarantee that every corpus position is visited exactly the same number of
times.

## Controls

Every paired run uses the same:

- Instantiated model architecture and exact parameter count
- Initial SafeTensor weights
- Frozen tokenizer
- Optimizer configuration
- Batch size and context length
- Validation corpus and fixed validation batches
- Processed-token target

Only the training corpus available before reuse changes.

The recommended manifest uses three seeds. Each seed gets an independently
initialized model, then that exact initial SafeTensor is shared by all compared
runs for the seed. The report shows the mean and observed range so one lucky
initialization cannot carry the conclusion.

Training and validation are separate files and are encoded independently, so a
context window cannot cross their boundary. Exact file hashes are recorded and
the checker rejects a training file whose hash equals the validation file hash.
For a substantive experiment, prepare the files from distinct source documents;
hash inequality alone cannot detect duplicated passages.

## Commands

```sh
cargo run -- check experiments/smoke-mvp.toml
cargo run --release -- experiment experiments/smoke-mvp.toml
cargo run -- report runs/smoke-mvp-seeded
```

`check` is a scientific preflight, not just TOML validation. If
`expected_parameters` is configured, a mismatch with the instantiated model is
fatal.

## Artifacts

```text
runs/<experiment>/
├── experiment.resolved.toml
├── initial-deterministic-seed-<seed>.safetensors
├── tokenizer.json
├── <run>/
│   └── seed-<seed>/
│       ├── config.resolved.toml
│       ├── dataset.json
│       ├── metrics.jsonl
│       ├── model.safetensors
│       ├── samples.json
│       ├── summary.json
│       └── tokenizer.json
└── report/
    ├── comparison.csv
    ├── generalization-gap.svg
    ├── headline-comparison.svg
    ├── index.html
    └── training-validation-nll.svg
```

The report refuses to combine runs whose parameter count, tokenizer hash, or
validation-corpus hash differs. Within each seed, it also requires every run to
share the same initial-weight hash.

## Interpreting the charts

The first chart displays training and validation NLL against processed tokens.
The second shows:

```text
generalization_gap = validation_nll - train_nll
```

A growing positive gap means training performance is improving faster than
held-out performance. The value also reflects differences in corpus difficulty,
so compare trends and matched corpora rather than treating it as a standalone
measure of memorization.

The fixture corpora only prove that the pipeline works. A real experiment
should derive its small training corpus as a document-level subset of the broad
training corpus and reserve separate documents for validation.
