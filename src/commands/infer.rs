use crate::bundle::{
    END_ID, START_ID, bytes_to_u32s, bytes_to_u64s, read_bundle_table, read_section_data,
};
use anyhow::{Context, Result, bail};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputMode {
    Tokens,
    Sentences,
}

#[derive(Debug)]
struct LoadedModel {
    context_keys: Vec<u64>,
    offsets: Vec<u32>,
    next_ids: Vec<u32>,
    counts: Vec<u32>,
    start_ids: Vec<u32>,
    start_counts: Vec<u32>,
}

#[derive(Debug)]
struct BundleData {
    vocab: Vec<String>,
    models: BTreeMap<String, LoadedModel>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ActiveModel {
    Primary,
    Secondary,
}

struct Generator<'a> {
    vocab: &'a [String],
    primary: &'a LoadedModel,
    secondary: &'a LoadedModel,
    alpha: f64,
    beta: f64,
    rng: StdRng,
    active: ActiveModel,
}

pub fn run(
    input: &Path,
    primary: &str,
    secondary: &str,
    alpha: f64,
    beta: f64,
    mode: OutputMode,
    size: usize,
    seed: u64,
) -> Result<()> {
    if !(0.0..=1.0).contains(&alpha) {
        bail!("--alpha must be in [0, 1]");
    }
    if !(0.0..=1.0).contains(&beta) {
        bail!("--beta must be in [0, 1]");
    }
    if size == 0 {
        bail!("--size must be > 0");
    }
    if primary == secondary {
        bail!("--primary and --secondary must be different model names");
    }

    let bundle = load_bundle(input)?;

    let primary_model = bundle
        .models
        .get(primary)
        .with_context(|| format!("primary model '{}' not found in bundle", primary))?;
    let secondary_model = bundle
        .models
        .get(secondary)
        .with_context(|| format!("secondary model '{}' not found in bundle", secondary))?;

    let mut generator = Generator {
        vocab: &bundle.vocab,
        primary: primary_model,
        secondary: secondary_model,
        alpha,
        beta,
        rng: StdRng::seed_from_u64(seed),
        active: ActiveModel::Primary,
    };

    let output = match mode {
        OutputMode::Tokens => generator.generate_tokens(size),
        OutputMode::Sentences => generator.generate_sentences(size),
    };

    println!("{output}");
    Ok(())
}

fn load_bundle(input: &Path) -> Result<BundleData> {
    let (_header, sections) = read_bundle_table(input)?;
    let mut file = File::open(input).with_context(|| format!("opening {}", input.display()))?;

    let mut raw_sections: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for section in &sections {
        let data = read_section_data(&mut file, section)
            .with_context(|| format!("reading section {}", section.name))?;
        raw_sections.insert(section.name.clone(), data);
    }

    let vocab_idx = raw_sections
        .get("shared.vocab_idx")
        .with_context(|| "missing shared.vocab_idx section".to_string())?;
    let vocab_bin = raw_sections
        .get("shared.vocab_bin")
        .with_context(|| "missing shared.vocab_bin section".to_string())?;
    let vocab = decode_vocab(vocab_idx, vocab_bin)?;

    let mut model_names = BTreeSet::new();
    for name in raw_sections.keys() {
        if let Some((prefix, _)) = name.split_once('.') {
            if prefix != "shared" {
                model_names.insert(prefix.to_string());
            }
        }
    }

    let mut models = BTreeMap::new();
    for model_name in model_names {
        let model = load_model_from_sections(&raw_sections, &model_name)
            .with_context(|| format!("loading model {}", model_name))?;
        models.insert(model_name, model);
    }

    Ok(BundleData { vocab, models })
}

fn decode_vocab(vocab_idx_bytes: &[u8], vocab_bin: &[u8]) -> Result<Vec<String>> {
    let idx = bytes_to_u64s(vocab_idx_bytes)?;
    if idx.is_empty() {
        bail!("shared.vocab_idx is empty");
    }
    if idx[0] != 0 {
        bail!("shared.vocab_idx[0] must be 0");
    }
    if *idx.last().unwrap() != vocab_bin.len() as u64 {
        bail!(
            "shared.vocab_idx last offset {} != shared.vocab_bin size {}",
            idx.last().unwrap(),
            vocab_bin.len()
        );
    }

    let mut vocab = Vec::with_capacity(idx.len().saturating_sub(1));
    for i in 0..idx.len() - 1 {
        let start = idx[i] as usize;
        let end = idx[i + 1] as usize;
        if end < start || end > vocab_bin.len() {
            bail!("invalid vocab offset range {}..{}", start, end);
        }
        let token = std::str::from_utf8(&vocab_bin[start..end])
            .with_context(|| format!("shared vocab token {} is not valid UTF-8", i))?
            .to_string();
        vocab.push(token);
    }

    Ok(vocab)
}

fn load_model_from_sections(
    sections: &BTreeMap<String, Vec<u8>>,
    model_name: &str,
) -> Result<LoadedModel> {
    let context_keys = bytes_to_u64s(section_bytes(
        sections,
        &format!("{}.context_keys", model_name),
    )?)?;
    let offsets = bytes_to_u32s(section_bytes(sections, &format!("{}.offsets", model_name))?)?;
    let next_ids = bytes_to_u32s(section_bytes(
        sections,
        &format!("{}.next_ids", model_name),
    )?)?;
    let counts = bytes_to_u32s(section_bytes(sections, &format!("{}.counts", model_name))?)?;
    let start_ids = bytes_to_u32s(section_bytes(
        sections,
        &format!("{}.start_ids", model_name),
    )?)?;
    let start_counts = bytes_to_u32s(section_bytes(
        sections,
        &format!("{}.start_counts", model_name),
    )?)?;

    if offsets.len() != context_keys.len() + 1 {
        bail!(
            "model {}: offsets length {} != context_keys+1 {}",
            model_name,
            offsets.len(),
            context_keys.len() + 1
        );
    }

    if next_ids.len() != counts.len() {
        bail!(
            "model {}: next_ids length {} != counts length {}",
            model_name,
            next_ids.len(),
            counts.len()
        );
    }

    if start_ids.len() != start_counts.len() {
        bail!(
            "model {}: start_ids length {} != start_counts length {}",
            model_name,
            start_ids.len(),
            start_counts.len()
        );
    }

    if offsets.last().copied().unwrap_or(0) as usize != next_ids.len() {
        bail!(
            "model {}: offsets last {} != num_edges {}",
            model_name,
            offsets.last().copied().unwrap_or(0),
            next_ids.len()
        );
    }

    Ok(LoadedModel {
        context_keys,
        offsets,
        next_ids,
        counts,
        start_ids,
        start_counts,
    })
}

fn section_bytes<'a>(sections: &'a BTreeMap<String, Vec<u8>>, name: &str) -> Result<&'a [u8]> {
    sections
        .get(name)
        .map(|v| v.as_slice())
        .with_context(|| format!("missing section {}", name))
}

impl LoadedModel {
    fn sample_start(&self, rng: &mut StdRng) -> Option<u32> {
        sample_weighted(&self.start_ids, &self.start_counts, rng)
    }

    fn sample_next(&self, w1: u32, w2: u32, rng: &mut StdRng) -> Option<u32> {
        let key = ((w1 as u64) << 32) | (w2 as u64);
        let row = self.context_keys.binary_search(&key).ok()?;
        let lo = self.offsets[row] as usize;
        let hi = self.offsets[row + 1] as usize;

        if hi <= lo || hi > self.next_ids.len() || hi > self.counts.len() {
            return None;
        }

        sample_weighted(&self.next_ids[lo..hi], &self.counts[lo..hi], rng)
    }
}

impl<'a> Generator<'a> {
    fn maybe_switch(&mut self) {
        match self.active {
            ActiveModel::Primary => {
                if self.rng.gen_bool(self.alpha) {
                    self.active = ActiveModel::Secondary;
                }
            }
            ActiveModel::Secondary => {
                if self.rng.gen_bool(self.beta) {
                    self.active = ActiveModel::Primary;
                }
            }
        }
    }

    fn sample_first_token(&mut self) -> u32 {
        self.maybe_switch();

        match self.active {
            ActiveModel::Primary => self
                .primary
                .sample_start(&mut self.rng)
                .or_else(|| self.secondary.sample_start(&mut self.rng))
                .unwrap_or(END_ID),
            ActiveModel::Secondary => self
                .secondary
                .sample_start(&mut self.rng)
                .or_else(|| self.primary.sample_start(&mut self.rng))
                .unwrap_or(END_ID),
        }
    }

    fn sample_next_token(&mut self, w1: u32, w2: u32) -> u32 {
        self.maybe_switch();

        match self.active {
            ActiveModel::Primary => self
                .primary
                .sample_next(w1, w2, &mut self.rng)
                .or_else(|| self.secondary.sample_next(w1, w2, &mut self.rng))
                .unwrap_or(END_ID),
            ActiveModel::Secondary => self
                .secondary
                .sample_next(w1, w2, &mut self.rng)
                .or_else(|| self.primary.sample_next(w1, w2, &mut self.rng))
                .unwrap_or(END_ID),
        }
    }

    fn generate_tokens(&mut self, size: usize) -> String {
        let mut out = String::new();
        let mut prev_plain: Option<String> = None;
        let mut produced = 0usize;
        let mut sentence_start = true;
        let mut w1 = START_ID;
        let mut w2 = END_ID;

        while produced < size {
            if sentence_start {
                let first = self.sample_first_token();
                if first == END_ID {
                    break;
                }

                let token = &self.vocab[first as usize];
                append_token(&mut out, &mut prev_plain, token, true);

                produced += 1;
                sentence_start = false;
                w1 = START_ID;
                w2 = first;
                continue;
            }

            let next = self.sample_next_token(w1, w2);
            if next == END_ID {
                sentence_start = true;
                continue;
            }

            let token = &self.vocab[next as usize];
            append_token(&mut out, &mut prev_plain, token, false);

            produced += 1;
            w1 = w2;
            w2 = next;
        }

        out
    }

    fn generate_sentences(&mut self, size: usize) -> String {
        let mut sentences = Vec::with_capacity(size);

        for _ in 0..size {
            let sentence = self.generate_one_sentence();
            if sentence.is_empty() {
                break;
            }
            sentences.push(sentence);
        }

        sentences.join("\n")
    }

    fn generate_one_sentence(&mut self) -> String {
        let mut out = String::new();
        let mut prev_plain: Option<String> = None;

        let first = self.sample_first_token();
        if first == END_ID {
            return out;
        }

        let mut w1 = START_ID;
        let mut w2 = first;
        append_token(&mut out, &mut prev_plain, &self.vocab[first as usize], true);

        loop {
            let next = self.sample_next_token(w1, w2);
            if next == END_ID {
                break;
            }

            let token = &self.vocab[next as usize];
            append_token(&mut out, &mut prev_plain, token, false);

            w1 = w2;
            w2 = next;
        }

        out
    }
}

fn sample_weighted(ids: &[u32], weights: &[u32], rng: &mut StdRng) -> Option<u32> {
    if ids.is_empty() || ids.len() != weights.len() {
        return None;
    }

    let total: u64 = weights.iter().map(|&w| w as u64).sum();
    if total == 0 {
        return None;
    }

    let mut r = rng.gen_range(0..total);

    for (&id, &weight) in ids.iter().zip(weights.iter()) {
        let w = weight as u64;
        if r < w {
            return Some(id);
        }
        r -= w;
    }

    ids.last().copied()
}

fn append_token(
    out: &mut String,
    prev_plain: &mut Option<String>,
    token: &str,
    sentence_start: bool,
) {
    if out.is_empty() {
        out.push_str(token);
        *prev_plain = Some(token.to_string());
        return;
    }

    if sentence_start {
        out.push(' ');
        out.push_str(token);
        *prev_plain = Some(token.to_string());
        return;
    }

    let no_space_before = matches!(
        token,
        "." | "," | "!" | "?" | ";" | ":" | ")" | "]" | "}" | "»" | "”"
    );

    let no_space_after = prev_plain
        .as_deref()
        .map(|prev| matches!(prev, "(" | "[" | "{" | "«" | "„"))
        .unwrap_or(false);

    if no_space_before || no_space_after {
        out.push_str(token);
    } else {
        out.push(' ');
        out.push_str(token);
    }

    *prev_plain = Some(token.to_string());
}
