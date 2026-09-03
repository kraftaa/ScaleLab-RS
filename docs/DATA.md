# MVP data

The real experiment uses two document-separated public-domain books from
Project Gutenberg:

- Training source: Jane Austen, *Pride and Prejudice*, ebook 1342 —
  https://www.gutenberg.org/ebooks/1342
- Held-out validation source: Lewis Carroll, *Alice's Adventures in
  Wonderland*, ebook 11 — https://www.gutenberg.org/ebooks/11

Project Gutenberg marks both ebooks public domain in the USA. Users outside the
USA should check local copyright law. The raw downloads retain the Gutenberg
license wrapper; generated corpora strip the wrapper but record source URLs and
SHA-256 hashes in `data/mvp/provenance.json`.

Prepare the data with:

```sh
./scripts/prepare-mvp-data.sh
```

Normalization lowercases text, normalizes typographic quotes and dashes, and
keeps ASCII letters, digits, selected punctuation, spaces, and paragraph
breaks. `train-small.txt` is exactly the first 25,184 character tokens of
`train-broad.txt`. The last six chapters of *Pride and Prejudice* are held out
as `validation-in-domain.txt` and are not part of either training file.
Out-of-domain validation still comes from *Alice's Adventures in Wonderland*
and is never used to construct the tokenizer vocabulary.

Rerunning `prepare-corpus` produces that in-domain holdout. The published
[`RESULTS.md`](RESULTS.md) numbers used the earlier layout that trained on the
full Pride and Prejudice text.

`prepare-corpus` pins the SHA-256 hashes of the Gutenberg downloads used for
the published result. A reissued ebook fails the command unless you pass
`--allow-source-drift` after inspecting the new text.
