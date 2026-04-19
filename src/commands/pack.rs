use crate::bundle::{
    SectionBlob, SectionKind, bytes_to_u32s, bytes_to_u64s, read_file, validate_vocab_files,
    write_bundle,
};
use anyhow::{Context, Result, bail};
use std::path::Path;

pub fn run(input_root: &Path, output: &Path, models: &[String]) -> Result<()> {
    if output.exists() {
        bail!("refusing to overwrite existing file: {}", output.display());
    }

    let mut sections: Vec<SectionBlob> = Vec::new();

    let vocab_idx = read_file(&input_root.join("vocab.idx"))
        .with_context(|| format!("reading {}", input_root.join("vocab.idx").display()))?;
    let vocab_bin = read_file(&input_root.join("vocab.bin"))
        .with_context(|| format!("reading {}", input_root.join("vocab.bin").display()))?;

    validate_vocab_files(&vocab_idx, &vocab_bin)?;

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

    for model_name in models {
        let model_dir = input_root.join(model_name);
        if !model_dir.is_dir() {
            bail!("missing model directory: {}", model_dir.display());
        }

        let context_keys_bytes = read_file(&model_dir.join("context_keys.bin"))
            .with_context(|| format!("reading {}", model_dir.join("context_keys.bin").display()))?;
        let offsets_bytes = read_file(&model_dir.join("offsets.bin"))
            .with_context(|| format!("reading {}", model_dir.join("offsets.bin").display()))?;
        let next_ids_bytes = read_file(&model_dir.join("next_ids.bin"))
            .with_context(|| format!("reading {}", model_dir.join("next_ids.bin").display()))?;
        let counts_bytes = read_file(&model_dir.join("counts.bin"))
            .with_context(|| format!("reading {}", model_dir.join("counts.bin").display()))?;
        let start_ids_bytes = read_file(&model_dir.join("start_ids.bin"))
            .with_context(|| format!("reading {}", model_dir.join("start_ids.bin").display()))?;
        let start_counts_bytes = read_file(&model_dir.join("start_counts.bin"))
            .with_context(|| format!("reading {}", model_dir.join("start_counts.bin").display()))?;

        let context_keys = bytes_to_u64s(&context_keys_bytes)?;
        let offsets = bytes_to_u32s(&offsets_bytes)?;
        let next_ids = bytes_to_u32s(&next_ids_bytes)?;
        let counts = bytes_to_u32s(&counts_bytes)?;
        let start_ids = bytes_to_u32s(&start_ids_bytes)?;
        let start_counts = bytes_to_u32s(&start_counts_bytes)?;

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

        sections.push(SectionBlob {
            name: format!("{}.context_keys", model_name),
            kind: SectionKind::ContextKeys,
            elem_size: 8,
            data: context_keys_bytes,
        });
        sections.push(SectionBlob {
            name: format!("{}.offsets", model_name),
            kind: SectionKind::Offsets,
            elem_size: 4,
            data: offsets_bytes,
        });
        sections.push(SectionBlob {
            name: format!("{}.next_ids", model_name),
            kind: SectionKind::NextIds,
            elem_size: 4,
            data: next_ids_bytes,
        });
        sections.push(SectionBlob {
            name: format!("{}.counts", model_name),
            kind: SectionKind::Counts,
            elem_size: 4,
            data: counts_bytes,
        });
        sections.push(SectionBlob {
            name: format!("{}.start_ids", model_name),
            kind: SectionKind::StartIds,
            elem_size: 4,
            data: start_ids_bytes,
        });
        sections.push(SectionBlob {
            name: format!("{}.start_counts", model_name),
            kind: SectionKind::StartCounts,
            elem_size: 4,
            data: start_counts_bytes,
        });
    }

    write_bundle(output, &sections)?;
    println!("wrote {}", output.display());
    Ok(())
}
