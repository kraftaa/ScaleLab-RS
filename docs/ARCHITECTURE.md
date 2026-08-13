# Explaining the Model

This walkthrough follows one batch through the implementation. Use these names
when reading `src/model.rs`:

- `B`: number of sequences in a batch
- `T`: tokens in each sequence
- `C`: embedding width (`d_model`)
- `H`: attention heads
- `D`: width of one head (`C / H`)
- `V`: vocabulary size

## 1. The learning problem

Given a sequence such as `rust`, training creates two aligned sequences:

```text
input:   r u s
target:  u s t
```

The model sees each input prefix and predicts the token one position to its
right. A random batch therefore has input and target shapes `[B, T]`.

## 2. Embeddings

Token IDs index a learned table `[V, C]`; position IDs index a learned table
`[context_length, C]`. Adding both produces `[B, T, C]`.

The token embedding answers "which symbol is this?" and the position embedding
answers "where is it?". Attention itself has no built-in concept of order.

## 3. Multi-head causal self-attention

One linear layer projects `[B, T, C]` to the concatenated queries, keys, and
values `[B, T, 3C]`. Each is reshaped to `[B, H, T, D]`.

For each head, the model computes:

```text
scores  = Q × transpose(K) / sqrt(D)       [B, H, T, T]
weights = softmax(causal_mask(scores))     [B, H, T, T]
result  = weights × V                      [B, H, T, D]
```

Each row of `weights` says how strongly one token reads every earlier token.
The triangular causal mask replaces future-token scores with negative infinity,
so their softmax probability becomes zero.

Heads are joined back into `[B, T, C]` and passed through an output projection.

## 4. Transformer block

Every block contains two residual updates:

```text
x = x + attention(layer_norm(x))
x = x + mlp(layer_norm(x))
```

Layer normalization stabilizes the values entering each transformation. The
residual path lets information and gradients travel through many blocks without
requiring every block to reconstruct the entire representation.

The MLP expands each token independently from `C` to `d_ff`, applies GELU, and
projects back to `C`. Attention mixes information between token positions; the
MLP transforms the information at each position.

## 5. Logits and loss

After the final normalization, `lm_head` maps every `[C]` vector to `V` raw
scores, producing logits `[B, T, V]`. Cross-entropy rewards a high score for the
actual next token at all `B × T` positions.

The reported negative log-likelihood uses natural logarithms and therefore has
units of nats per token. Perplexity is `exp(nll)`.

## 6. Why the tests matter

- `tokenizer_round_trips_unicode`: encoding does not lose text.
- `batches_are_shifted_by_one_token`: the labels represent the intended task.
- `causal_mask_hides_future_positions`: the mask has the right orientation.
- `model_returns_one_distribution_per_input_token`: the output contract holds.
- `future_tokens_do_not_change_earlier_logits`: the whole network—not merely the
  mask matrix—obeys causality.

That last test catches one of the most damaging silent errors in a language
model: accidentally letting training targets leak into their own predictions.
