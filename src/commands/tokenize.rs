use crate::bundle::{
    SectionBlob, SectionKind, build_vocab_files, compute_start_distribution, flatten_model,
    u32s_to_bytes, u64s_to_bytes, validate_vocab_files, write_bundle,
};
use crate::tokenizer::{line_to_sequence_ids, open_reader, scan_vocab_from_files_with_progress};
use anyhow::{Context, Result, bail};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};

pub fn run(inputs: &[PathBuf], models: &[String], output: &Path) -> Result<()> {
    if output.exists() {
        bail!("refusing to overwrite existing file: {}", output.display());
    }

    if inputs.len() != models.len() {
        bail!(
            "--input count {} must equal --model count {}",
            inputs.len(),
            models.len()
        );
    }

    let mp = MultiProgress::new();

    let vocab_pb = mp.add(new_bytes_pb("scan vocab"));
    let vocab = scan_vocab_from_files_with_progress(inputs, &vocab_pb)?;
    vocab_pb.finish_with_message("scan vocab: done");

    if vocab.len() > u32::MAX as usize {
        bail!("vocabulary size {} exceeds u32::MAX", vocab.len());
    }

    let pack_pb = mp.add(new_spinner("pack shared vocab"));
    let (vocab_idx, vocab_bin) = build_vocab_files(&vocab);
    validate_vocab_files(&vocab_idx, &vocab_bin)?;
    pack_pb.finish_with_message("pack shared vocab: done");

    let token_to_id: HashMap<String, u32> = vocab
        .iter()
        .enumerate()
        .map(|(i, token)| (token.clone(), i as u32))
        .collect();

    let mut sections: Vec<SectionBlob> = Vec::new();
    sections.push(SectionBlob {
        name: "shared.vocab_idx".to_string(),
        kind: SectionKind::SharedVocabIdx,
        elem_size: 8,
        data: vocab_idx,
    });
    sections.push(SectionBlob {
        name: "shared.vocab_bin".to_string(),
        kind: SectionKind::SharedVocabBin,
        elem_size: 1,
        data: vocab_bin,
    });

    for (input_path, model_name) in inputs.iter().zip(models.iter()) {
        let model_pb = mp.add(new_bytes_pb(&format!("build model {}", model_name)));
        let model_map = build_model_from_text(input_path, &token_to_id, &model_pb)
            .with_context(|| format!("building model {}", model_name))?;
        model_pb.finish_with_message(format!("build model {}: done", model_name));

        let flatten_pb = mp.add(new_spinner(&format!("flatten {}", model_name)));
        let (context_keys, offsets, next_ids, counts) = flatten_model(&model_map)
            .with_context(|| format!("flattening model {}", model_name))?;

        let (start_ids, start_counts) =
            compute_start_distribution(&context_keys, &offsets, &counts).with_context(|| {
                format!("computing start distribution for model {}", model_name)
            })?;
        flatten_pb.finish_with_message(format!("flatten {}: done", model_name));

        sections.push(SectionBlob {
            name: format!("{}.context_keys", model_name),
            kind: SectionKind::ContextKeys,
            elem_size: 8,
            data: u64s_to_bytes(&context_keys),
        });
        sections.push(SectionBlob {
            name: format!("{}.offsets", model_name),
            kind: SectionKind::Offsets,
            elem_size: 4,
            data: u32s_to_bytes(&offsets),
        });
        sections.push(SectionBlob {
            name: format!("{}.next_ids", model_name),
            kind: SectionKind::NextIds,
            elem_size: 4,
            data: u32s_to_bytes(&next_ids),
        });
        sections.push(SectionBlob {
            name: format!("{}.counts", model_name),
            kind: SectionKind::Counts,
            elem_size: 4,
            data: u32s_to_bytes(&counts),
        });
        sections.push(SectionBlob {
            name: format!("{}.start_ids", model_name),
            kind: SectionKind::StartIds,
            elem_size: 4,
            data: u32s_to_bytes(&start_ids),
        });
        sections.push(SectionBlob {
            name: format!("{}.start_counts", model_name),
            kind: SectionKind::StartCounts,
            elem_size: 4,
            data: u32s_to_bytes(&start_counts),
        });
    }

    let write_pb = mp.add(new_spinner("write bundle"));
    write_bundle(output, &sections)?;
    write_pb.finish_with_message(format!("write bundle: done ({})", output.display()));

    println!("wrote {}", output.display());
    Ok(())
}

fn build_model_from_text(
    input_path: &Path,
    token_to_id: &HashMap<String, u32>,
    pb: &ProgressBar,
) -> Result<BTreeMap<u64, BTreeMap<u32, u32>>> {
    let mut model: BTreeMap<u64, BTreeMap<u32, u32>> = BTreeMap::new();

    let total_bytes = fs::metadata(input_path)
        .with_context(|| format!("stat {}", input_path.display()))?
        .len();
    pb.set_length(total_bytes);
    pb.set_position(0);

    let reader = open_reader(input_path)?;
    let mut processed_bytes: u64 = 0;

    for line in reader.lines() {
        let line = line.with_context(|| format!("reading {}", input_path.display()))?;
        processed_bytes = processed_bytes
            .saturating_add(line.len() as u64)
            .saturating_add(1);
        pb.set_position(processed_bytes.min(total_bytes));

        let ids = line_to_sequence_ids(&line, token_to_id)?;

        if ids.is_empty() || ids.len() < 3 {
            continue;
        }

        for i in 2..ids.len() {
            let w1 = ids[i - 2];
            let w2 = ids[i - 1];
            let w3 = ids[i];

            let context_key = ((w1 as u64) << 32) | (w2 as u64);

            let next_map = model.entry(context_key).or_default();
            let count = next_map.entry(w3).or_insert(0);

            if *count == u32::MAX {
                bail!("count overflow for context ({}, {}) -> {}", w1, w2, w3);
            }

            *count += 1;
        }
    }

    pb.set_position(total_bytes);
    Ok(model)
}

fn new_bytes_pb(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new(0);
    let style = ProgressStyle::with_template(
        "{msg:20} [{bar:40.cyan/blue}] {bytes:>10}/{total_bytes:<10} {percent:>3}% {elapsed_precise}",
    )
    .unwrap()
    .progress_chars("##-");
    pb.set_style(style);
    pb.set_message(msg.to_string());
    pb
}

fn new_spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    let style = ProgressStyle::with_template("{msg:20} {spinner} {elapsed_precise}").unwrap();
    pb.set_style(style);
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(120));
    pb
}
