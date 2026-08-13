use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

/// A deliberately simple tokenizer: every Unicode scalar value is a token.
/// It keeps the first experiment transparent; a BPE tokenizer can be swapped in later.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharTokenizer {
    chars: Vec<char>,
    #[serde(skip)]
    ids: HashMap<char, u32>,
}

impl CharTokenizer {
    pub fn train(text: &str) -> Self {
        let chars: Vec<char> = text.chars().collect::<BTreeSet<_>>().into_iter().collect();
        let ids = chars
            .iter()
            .enumerate()
            .map(|(id, ch)| (*ch, id as u32))
            .collect();
        Self { chars, ids }
    }

    pub fn vocab_size(&self) -> usize {
        self.chars.len()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        text.chars()
            .map(|ch| {
                self.ids
                    .get(&ch)
                    .copied()
                    .with_context(|| format!("character {ch:?} is not in the vocabulary"))
            })
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        ids.iter()
            .map(|id| {
                self.chars
                    .get(*id as usize)
                    .copied()
                    .with_context(|| format!("token id {id} is outside the vocabulary"))
            })
            .collect()
    }

    pub fn rebuild_index(&mut self) {
        self.ids = self
            .chars
            .iter()
            .enumerate()
            .map(|(id, ch)| (*ch, id as u32))
            .collect();
    }

    pub fn observed_token_types(&self, tokens: &[u32]) -> usize {
        tokens.iter().copied().collect::<BTreeSet<_>>().len()
    }
}

pub struct TokenDataset {
    pub train: Vec<u32>,
    pub validation: Vec<u32>,
}

impl TokenDataset {
    pub fn from_text(text: &str, tokenizer: &CharTokenizer, train_fraction: f64) -> Result<Self> {
        let tokens = tokenizer.encode(text)?;
        let split = (tokens.len() as f64 * train_fraction) as usize;
        Ok(Self {
            train: tokens[..split].to_vec(),
            validation: tokens[split..].to_vec(),
        })
    }
}

pub fn random_batch(
    tokens: &[u32],
    batch_size: usize,
    context_length: usize,
    rng: &mut impl Rng,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    anyhow::ensure!(
        tokens.len() > context_length,
        "split has {} tokens but a batch needs at least {}",
        tokens.len(),
        context_length + 1
    );
    let mut inputs = Vec::with_capacity(batch_size * context_length);
    let mut targets = Vec::with_capacity(batch_size * context_length);
    let last_start = tokens.len() - context_length - 1;
    for _ in 0..batch_size {
        let start = rng.random_range(0..=last_start);
        inputs.extend_from_slice(&tokens[start..start + context_length]);
        targets.extend_from_slice(&tokens[start + 1..start + context_length + 1]);
    }
    let shape = (batch_size, context_length);
    Ok((
        Tensor::from_vec(inputs, shape, device)?,
        Tensor::from_vec(targets, shape, device)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_round_trips_unicode() {
        let text = "hello, λ!";
        let tokenizer = CharTokenizer::train(text);
        let ids = tokenizer.encode(text).unwrap();
        assert_eq!(tokenizer.decode(&ids).unwrap(), text);
    }

    #[test]
    fn batches_are_shifted_by_one_token() {
        let device = Device::Cpu;
        let tokens: Vec<u32> = (0..20).collect();
        let mut rng = rand::rng();
        let (x, y) = random_batch(&tokens, 3, 4, &mut rng, &device).unwrap();
        let x = x.to_vec2::<u32>().unwrap();
        let y = y.to_vec2::<u32>().unwrap();
        for (input, target) in x.iter().zip(y.iter()) {
            assert_eq!(&input[1..], &target[..target.len() - 1]);
        }
    }
}
