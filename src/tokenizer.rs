use crate::bundle::{END_ID, START_ID};
use anyhow::{Context, Result, bail};
use indicatif::ProgressBar;
use std::collections::{BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::iter::Peekable;
use std::path::{Path, PathBuf};
use std::str::Chars;

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric()
}

fn is_word_joiner(c: char) -> bool {
    matches!(c, '\'' | '’' | '-')
}

fn is_emittable_punctuation(c: char) -> bool {
    matches!(
        c,
        '.' | ',' | '!' | '?' | ';' | ':' | '(' | ')' | '"' | '„' | '”' | '«' | '»' | '…' | '-'
    )
}

fn flush_word(tokens: &mut Vec<String>, word: &mut String) {
    if !word.is_empty() {
        tokens.push(std::mem::take(word));
    }
}

fn next_is_word_char(iter: &mut Peekable<Chars<'_>>) -> bool {
    iter.peek().copied().map(is_word_char).unwrap_or(false)
}

pub fn tokenize_line(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut word = String::new();
    let mut iter = line.chars().peekable();

    while let Some(c) = iter.next() {
        if is_word_char(c) {
            word.push(c);
            continue;
        }

        if is_word_joiner(c) && !word.is_empty() && next_is_word_char(&mut iter) {
            word.push(c);
            continue;
        }

        flush_word(&mut tokens, &mut word);

        if c.is_whitespace() {
            continue;
        }

        if is_emittable_punctuation(c) {
            tokens.push(c.to_string());
        }
    }

    flush_word(&mut tokens, &mut word);
    tokens
}

pub fn scan_vocab_from_files(paths: &[PathBuf]) -> Result<Vec<String>> {
    let dummy = ProgressBar::hidden();
    scan_vocab_from_files_with_progress(paths, &dummy)
}

pub fn scan_vocab_from_files_with_progress(
    paths: &[PathBuf],
    pb: &ProgressBar,
) -> Result<Vec<String>> {
    let mut seen = BTreeSet::new();

    let total_bytes: u64 = paths
        .iter()
        .map(|p| {
            fs::metadata(p)
                .with_context(|| format!("stat {}", p.display()))
                .map(|m| m.len())
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .sum();

    pb.set_length(total_bytes);
    pb.set_position(0);

    let mut processed_bytes: u64 = 0;

    for path in paths {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line.with_context(|| format!("reading {}", path.display()))?;
            processed_bytes = processed_bytes
                .saturating_add(line.len() as u64)
                .saturating_add(1);
            pb.set_position(processed_bytes.min(total_bytes));

            for token in tokenize_line(&line) {
                if token == "<s>" || token == "</s>" {
                    bail!("reserved token appeared in input: {}", token);
                }
                seen.insert(token);
            }
        }
    }

    pb.set_position(total_bytes);

    let mut vocab = Vec::with_capacity(seen.len() + 2);
    vocab.push("<s>".to_string());
    vocab.push("</s>".to_string());
    vocab.extend(seen);

    Ok(vocab)
}

pub fn line_to_token_ids(line: &str, token_to_id: &HashMap<String, u32>) -> Result<Vec<u32>> {
    let tokens = tokenize_line(line);
    let mut ids = Vec::with_capacity(tokens.len());

    for token in tokens {
        let id = token_to_id
            .get(token.as_str())
            .copied()
            .with_context(|| format!("token not found in vocabulary: {}", token))?;
        ids.push(id);
    }

    Ok(ids)
}

pub fn line_to_sequence_ids(line: &str, token_to_id: &HashMap<String, u32>) -> Result<Vec<u32>> {
    let token_ids = line_to_token_ids(line, token_to_id)?;
    if token_ids.is_empty() {
        return Ok(Vec::new());
    }

    let mut ids = Vec::with_capacity(token_ids.len() + 2);
    ids.push(START_ID);
    ids.extend(token_ids);
    ids.push(END_ID);
    Ok(ids)
}

pub fn open_reader(path: &Path) -> Result<BufReader<File>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    Ok(BufReader::new(file))
}
