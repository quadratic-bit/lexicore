use crate::bundle::{MAGIC, SectionEntry, read_bundle_table, split_section_name};
use anyhow::Result;
use std::collections::BTreeMap;
use std::path::Path;

pub fn run(input: &Path) -> Result<()> {
    let (header, sections) = read_bundle_table(input)?;

    println!("file: {}", input.display());
    println!("magic: {}", String::from_utf8_lossy(MAGIC));
    println!("version: {}", header.version);
    println!("section_count: {}", header.section_count);
    println!("directory_offset: {}", header.directory_offset);
    println!("data_offset: {}", header.data_offset);
    println!();

    println!(
        "{:<32} {:<12} {:>10} {:>12} {:>12} {:>12}",
        "name", "kind", "elem_size", "offset", "length", "count"
    );

    for s in &sections {
        println!(
            "{:<32} {:<12} {:>10} {:>12} {:>12} {:>12}",
            s.name,
            s.kind.label(),
            s.elem_size,
            s.offset,
            s.length,
            s.count
        );
    }

    println!();
    print_model_summary(&sections);

    Ok(())
}

fn print_model_summary(sections: &[SectionEntry]) {
    let mut map: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();

    for s in sections {
        if let Some((prefix, suffix)) = split_section_name(&s.name) {
            if prefix != "shared" {
                map.entry(prefix.to_string())
                    .or_default()
                    .insert(suffix.to_string(), s.count);
            }
        }
    }

    println!("models:");
    for (name, items) in map {
        let contexts = items.get("context_keys").copied().unwrap_or(0);
        let offsets = items.get("offsets").copied().unwrap_or(0);
        let edges = items.get("next_ids").copied().unwrap_or(0);
        let counts = items.get("counts").copied().unwrap_or(0);
        let starts = items.get("start_ids").copied().unwrap_or(0);

        println!("  {}", name);
        println!("    contexts: {}", contexts);
        println!("    offsets: {}", offsets);
        println!("    edges(next_ids): {}", edges);
        println!("    counts: {}", counts);
        println!("    start_ids: {}", starts);
    }
}
