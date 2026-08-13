#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
raw_dir="$project_dir/data/raw"
output_dir="$project_dir/data/mvp"
mkdir -p "$raw_dir"

curl --fail --location --retry 3 \
  https://www.gutenberg.org/cache/epub/1342/pg1342.txt \
  --output "$raw_dir/pride-and-prejudice.txt"

curl --fail --location --retry 3 \
  https://www.gutenberg.org/cache/epub/11/pg11.txt \
  --output "$raw_dir/alice-in-wonderland.txt"

cargo run --release --manifest-path "$project_dir/Cargo.toml" -- prepare-corpus \
  --train-raw "$raw_dir/pride-and-prejudice.txt" \
  --validation-raw "$raw_dir/alice-in-wonderland.txt" \
  --output-dir "$output_dir" \
  --small-tokens 25184
