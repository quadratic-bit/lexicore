use crate::bundle::{dump_path_for_section, read_bundle_table, read_section_data};
use anyhow::{Context, Result, bail};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

pub fn run(input: &Path, output_root: &Path) -> Result<()> {
    if output_root.exists() {
        bail!(
            "refusing to overwrite existing output directory: {}",
            output_root.display()
        );
    }
    fs::create_dir_all(output_root)
        .with_context(|| format!("creating {}", output_root.display()))?;

    let (_header, sections) = read_bundle_table(input)?;
    let mut f = File::open(input).with_context(|| format!("opening {}", input.display()))?;

    for s in &sections {
        let data = read_section_data(&mut f, s)?;
        let out_path = dump_path_for_section(output_root, s)?;
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let mut out =
            File::create(&out_path).with_context(|| format!("creating {}", out_path.display()))?;
        out.write_all(&data)
            .with_context(|| format!("writing {}", out_path.display()))?;
    }

    println!("dumped to {}", output_root.display());
    Ok(())
}
