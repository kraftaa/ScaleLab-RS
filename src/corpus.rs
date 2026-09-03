use crate::experiment::sha256_file;
use anyhow::{Context, Result};
use serde::Serialize;
use std::{fs, path::Path};

const TRAIN_URL: &str = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt";
const VALIDATION_URL: &str = "https://www.gutenberg.org/cache/epub/11/pg11.txt";
const EXPECTED_TRAIN_RAW_SHA256: &str =
    "74f2665d6e6925fc2c17dec644bec9e87df478a0f1836822125e8acbb3777806";
const EXPECTED_VALIDATION_RAW_SHA256: &str =
    "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e";
const IN_DOMAIN_HELD_OUT_CHAPTERS: usize = 6;

#[derive(Serialize)]
struct CorpusProvenance {
    training_source: SourceProvenance,
    validation_source: SourceProvenance,
    normalization: &'static str,
    small_is_prefix_of_broad: bool,
    in_domain_held_out_chapters: usize,
    train_small_tokens: usize,
    train_broad_tokens: usize,
    validation_tokens: usize,
    in_domain_validation_tokens: usize,
    train_small_sha256: String,
    train_broad_sha256: String,
    validation_sha256: String,
    in_domain_validation_sha256: String,
}

#[derive(Serialize)]
struct SourceProvenance {
    title: &'static str,
    author: &'static str,
    project_gutenberg_ebook: u32,
    url: &'static str,
    raw_sha256: String,
    license_note: &'static str,
}

pub fn prepare_gutenberg(
    train_raw: &Path,
    validation_raw: &Path,
    output_dir: &Path,
    small_tokens: usize,
    allow_source_drift: bool,
) -> Result<()> {
    verify_pinned_source_hash(
        train_raw,
        EXPECTED_TRAIN_RAW_SHA256,
        "Pride and Prejudice raw download",
        allow_source_drift,
    )?;
    verify_pinned_source_hash(
        validation_raw,
        EXPECTED_VALIDATION_RAW_SHA256,
        "Alice's Adventures in Wonderland raw download",
        allow_source_drift,
    )?;
    let train_raw_text = fs::read_to_string(train_raw)
        .with_context(|| format!("failed to read {}", train_raw.display()))?;
    let validation_raw_text = fs::read_to_string(validation_raw)
        .with_context(|| format!("failed to read {}", validation_raw.display()))?;
    let full_train_book = normalize(&strip_gutenberg(&train_raw_text)?);
    let validation = normalize(&strip_gutenberg(&validation_raw_text)?);
    let (broad, in_domain) =
        hold_out_final_chapters(&full_train_book, IN_DOMAIN_HELD_OUT_CHAPTERS)?;
    anyhow::ensure!(
        broad.chars().count() >= small_tokens,
        "normalized training corpus has fewer than {small_tokens} characters after the in-domain holdout"
    );
    let small: String = broad.chars().take(small_tokens).collect();
    anyhow::ensure!(
        broad.starts_with(&small),
        "small corpus must be a prefix of broad corpus"
    );
    anyhow::ensure!(
        broad != validation,
        "training and out-of-domain validation corpora must differ"
    );
    anyhow::ensure!(
        !broad.contains(in_domain.trim_end()),
        "in-domain validation chapters leaked into the training corpus"
    );

    fs::create_dir_all(output_dir)?;
    let small_path = output_dir.join("train-small.txt");
    let broad_path = output_dir.join("train-broad.txt");
    let validation_path = output_dir.join("validation.txt");
    let in_domain_path = output_dir.join("validation-in-domain.txt");
    // The tokenizer sees training data only. The checker will reject unseen
    // validation characters rather than silently learning its alphabet from eval data.
    let tokenizer_path = output_dir.join("tokenizer-source.txt");
    fs::write(&small_path, &small)?;
    fs::write(&broad_path, &broad)?;
    fs::write(&validation_path, &validation)?;
    fs::write(&in_domain_path, &in_domain)?;
    fs::write(&tokenizer_path, &broad)?;

    let provenance = CorpusProvenance {
        training_source: SourceProvenance {
            title: "Pride and Prejudice",
            author: "Jane Austen",
            project_gutenberg_ebook: 1342,
            url: TRAIN_URL,
            raw_sha256: sha256_file(train_raw)?,
            license_note: "Project Gutenberg marks this ebook public domain in the USA.",
        },
        validation_source: SourceProvenance {
            title: "Alice's Adventures in Wonderland",
            author: "Lewis Carroll",
            project_gutenberg_ebook: 11,
            url: VALIDATION_URL,
            raw_sha256: sha256_file(validation_raw)?,
            license_note: "Project Gutenberg marks this ebook public domain in the USA.",
        },
        normalization: "strip Gutenberg wrapper; lowercase; normalize quotes/dashes; retain ASCII letters, digits, selected punctuation, spaces, and paragraph breaks; hold out the final six Pride and Prejudice chapters as in-domain validation",
        small_is_prefix_of_broad: true,
        in_domain_held_out_chapters: IN_DOMAIN_HELD_OUT_CHAPTERS,
        train_small_tokens: small.chars().count(),
        train_broad_tokens: broad.chars().count(),
        validation_tokens: validation.chars().count(),
        in_domain_validation_tokens: in_domain.chars().count(),
        train_small_sha256: sha256_file(&small_path)?,
        train_broad_sha256: sha256_file(&broad_path)?,
        validation_sha256: sha256_file(&validation_path)?,
        in_domain_validation_sha256: sha256_file(&in_domain_path)?,
    };
    fs::write(
        output_dir.join("provenance.json"),
        serde_json::to_vec_pretty(&provenance)?,
    )?;
    println!("Prepared corpus in {}", output_dir.display());
    println!(
        "  small training tokens  {:>10}",
        provenance.train_small_tokens
    );
    println!(
        "  broad training tokens  {:>10}",
        provenance.train_broad_tokens
    );
    println!(
        "  validation tokens      {:>10}",
        provenance.validation_tokens
    );
    println!(
        "  in-domain val tokens   {:>10}",
        provenance.in_domain_validation_tokens
    );
    Ok(())
}

fn hold_out_final_chapters(text: &str, chapters: usize) -> Result<(String, String)> {
    let starts = chapter_starts(text);
    anyhow::ensure!(
        starts.len() > chapters,
        "normalized training book has {} chapters; need more than {chapters} to hold out",
        starts.len()
    );
    let split_at = starts[starts.len() - chapters];
    anyhow::ensure!(split_at > 0, "cannot hold out the entire training book");
    let train = text[..split_at].trim_end().to_string() + "\n";
    let held = text[split_at..].to_string();
    anyhow::ensure!(
        !held.trim().is_empty(),
        "held-out in-domain chapters are empty"
    );
    Ok((train, held))
}

fn chapter_starts(text: &str) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut offset = 0usize;
    for line in text.split_inclusive('\n') {
        if line.starts_with("chapter ") {
            starts.push(offset);
        }
        offset += line.len();
    }
    starts
}

fn verify_pinned_source_hash(
    path: &Path,
    expected: &str,
    label: &str,
    allow_source_drift: bool,
) -> Result<String> {
    let actual = sha256_file(path)?;
    if actual == expected {
        return Ok(actual);
    }
    if allow_source_drift {
        eprintln!(
            "warning: {label} SHA-256 {actual} does not match pinned {expected}; continuing because --allow-source-drift was set"
        );
        return Ok(actual);
    }
    anyhow::bail!(
        "{label} SHA-256 is {actual}, expected pinned {expected}. \
         Project Gutenberg may have reissued the ebook. Inspect the download, \
         then update the pinned hashes in src/corpus.rs if the new text is acceptable, \
         or pass --allow-source-drift to prepare-corpus."
    );
}

fn strip_gutenberg(text: &str) -> Result<String> {
    let start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK";
    let end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK";
    let start = text
        .find(start_marker)
        .context("Project Gutenberg start marker not found")?;
    let body_start = text[start..]
        .find('\n')
        .map(|offset| start + offset + 1)
        .context("Project Gutenberg start marker has no following line")?;
    let end = text[body_start..]
        .find(end_marker)
        .map(|offset| body_start + offset)
        .context("Project Gutenberg end marker not found")?;
    Ok(text[body_start..end].to_string())
}

fn normalize(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut previous_space = false;
    let mut consecutive_newlines = 0usize;
    for original in text.chars() {
        let replacement: &[char] = match original {
            '‘' | '’' | '‚' => &['\''],
            '“' | '”' | '„' => &['"'],
            '—' | '–' => &['-'],
            '…' => &['.', '.', '.'],
            '\u{00a0}' | '\t' | '\r' => &[' '],
            _ => &[],
        };
        let characters: Vec<char> = if replacement.is_empty() {
            original.to_lowercase().collect()
        } else {
            replacement.to_vec()
        };
        for character in characters {
            if character == '\n' {
                while output.ends_with(' ') {
                    output.pop();
                }
                if consecutive_newlines < 2 {
                    output.push('\n');
                }
                consecutive_newlines += 1;
                previous_space = false;
            } else if character.is_whitespace() {
                if !previous_space && !output.ends_with('\n') {
                    output.push(' ');
                }
                previous_space = true;
                consecutive_newlines = 0;
            } else if is_allowed(character) {
                output.push(character);
                previous_space = false;
                consecutive_newlines = 0;
            }
        }
    }
    output.trim().to_string() + "\n"
}

fn is_allowed(character: char) -> bool {
    character.is_ascii_lowercase()
        || character.is_ascii_digit()
        || matches!(
            character,
            '.' | ',' | ';' | ':' | '!' | '?' | '\'' | '"' | '-' | '(' | ')'
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn normalization_is_lowercase_and_stable() {
        assert_eq!(
            normalize("  Hello—WORLD!\r\n\r\n“Hi”  "),
            "hello-world!\n\n\"hi\"\n"
        );
    }

    #[test]
    fn final_chapters_are_held_out_of_training() {
        let text = "preface\nchapter i.\none\nchapter ii.\ntwo\nchapter iii.\nthree\n";
        let (train, held) = hold_out_final_chapters(text, 1).unwrap();
        assert_eq!(train, "preface\nchapter i.\none\nchapter ii.\ntwo\n");
        assert_eq!(held, "chapter iii.\nthree\n");
        assert!(!train.contains(held.trim_end()));
    }

    #[test]
    fn wrapper_is_removed() {
        let text = "header\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nBody\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nlicense";
        assert_eq!(strip_gutenberg(text).unwrap(), "Body\n");
    }

    #[test]
    fn unexpected_raw_source_hash_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("unexpected.txt");
        fs::write(&path, "not a pinned gutenberg ebook").unwrap();
        let error = verify_pinned_source_hash(
            &path,
            EXPECTED_TRAIN_RAW_SHA256,
            "Pride and Prejudice raw download",
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("expected pinned"));
    }

    #[test]
    fn source_hash_drift_can_be_allowed() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("unexpected.txt");
        fs::write(&path, "not a pinned gutenberg ebook").unwrap();
        let actual = verify_pinned_source_hash(
            &path,
            EXPECTED_TRAIN_RAW_SHA256,
            "Pride and Prejudice raw download",
            true,
        )
        .unwrap();
        assert_ne!(actual, EXPECTED_TRAIN_RAW_SHA256);
    }
}
