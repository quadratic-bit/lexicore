use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const MAGIC: &[u8; 8] = b"TGRMBNDL";
pub const VERSION: u32 = 1;
pub const HEADER_SIZE: u64 = 32;
pub const SECTION_ENTRY_SIZE: u64 = 64;
pub const SECTION_NAME_LEN: usize = 32;
pub const START_ID: u32 = 0;
pub const END_ID: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SectionKind {
    SharedVocabIdx = 1,
    SharedVocabBin = 2,
    ContextKeys = 3,
    Offsets = 4,
    NextIds = 5,
    Counts = 6,
    StartIds = 7,
    StartCounts = 8,
}

impl SectionKind {
    pub fn from_u32(v: u32) -> Result<Self> {
        match v {
            1 => Ok(Self::SharedVocabIdx),
            2 => Ok(Self::SharedVocabBin),
            3 => Ok(Self::ContextKeys),
            4 => Ok(Self::Offsets),
            5 => Ok(Self::NextIds),
            6 => Ok(Self::Counts),
            7 => Ok(Self::StartIds),
            8 => Ok(Self::StartCounts),
            _ => bail!("unknown section kind {}", v),
        }
    }

    pub fn as_u32(self) -> u32 {
        self as u32
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::SharedVocabIdx => "shared.vocab_idx",
            Self::SharedVocabBin => "shared.vocab_bin",
            Self::ContextKeys => "context_keys",
            Self::Offsets => "offsets",
            Self::NextIds => "next_ids",
            Self::Counts => "counts",
            Self::StartIds => "start_ids",
            Self::StartCounts => "start_counts",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Header {
    pub version: u32,
    pub section_count: u32,
    pub directory_offset: u64,
    pub data_offset: u64,
}

#[derive(Debug, Clone)]
pub struct SectionEntry {
    pub name: String,
    pub kind: SectionKind,
    pub elem_size: u32,
    pub offset: u64,
    pub length: u64,
    pub count: u64,
}

#[derive(Debug, Clone)]
pub struct SectionBlob {
    pub name: String,
    pub kind: SectionKind,
    pub elem_size: u32,
    pub data: Vec<u8>,
}

pub fn build_vocab_files(tokens: &[String]) -> (Vec<u8>, Vec<u8>) {
    let mut idx: Vec<u64> = Vec::with_capacity(tokens.len() + 1);
    let mut bin: Vec<u8> = Vec::new();
    idx.push(0);

    for t in tokens {
        bin.extend_from_slice(t.as_bytes());
        idx.push(bin.len() as u64);
    }

    (u64s_to_bytes(&idx), bin)
}

pub fn validate_vocab_files(vocab_idx: &[u8], vocab_bin: &[u8]) -> Result<()> {
    let idx = bytes_to_u64s(vocab_idx)?;
    if idx.is_empty() {
        bail!("vocab.idx is empty");
    }
    if idx[0] != 0 {
        bail!("vocab.idx[0] must be 0");
    }
    validate_non_decreasing_u64(&idx, "vocab.idx")?;
    let last = *idx.last().unwrap();
    if last != vocab_bin.len() as u64 {
        bail!(
            "vocab.idx last offset {} != vocab.bin size {}",
            last,
            vocab_bin.len()
        );
    }
    Ok(())
}

pub fn compute_start_distribution(
    context_keys: &[u64],
    offsets: &[u32],
    counts: &[u32],
) -> Result<(Vec<u32>, Vec<u32>)> {
    if offsets.len() != context_keys.len() + 1 {
        bail!(
            "offsets length {} != context_keys+1 {}",
            offsets.len(),
            context_keys.len() + 1
        );
    }

    let mut start_ids = Vec::new();
    let mut start_counts = Vec::new();

    for i in 0..context_keys.len() {
        let key = context_keys[i];
        let w1 = (key >> 32) as u32;
        if w1 != START_ID {
            break;
        }

        let w2 = (key & 0xffff_ffff) as u32;
        if w2 == START_ID || w2 == END_ID {
            continue;
        }

        let lo = offsets[i] as usize;
        let hi = offsets[i + 1] as usize;
        if hi < lo || hi > counts.len() {
            bail!("bad offset slice for start distribution at row {}", i);
        }

        let total: u64 = counts[lo..hi].iter().map(|&x| x as u64).sum();
        if total > u32::MAX as u64 {
            bail!("start count overflow for token id {}", w2);
        }

        if total > 0 {
            start_ids.push(w2);
            start_counts.push(total as u32);
        }
    }

    Ok((start_ids, start_counts))
}

pub fn flatten_model(
    model: &BTreeMap<u64, BTreeMap<u32, u32>>,
) -> Result<(Vec<u64>, Vec<u32>, Vec<u32>, Vec<u32>)> {
    let mut context_keys = Vec::with_capacity(model.len());
    let mut offsets = Vec::with_capacity(model.len() + 1);
    let mut next_ids = Vec::new();
    let mut counts = Vec::new();

    offsets.push(0);

    for (&context_key, next_map) in model {
        context_keys.push(context_key);

        for (&next_id, &count) in next_map {
            if count == 0 {
                continue;
            }
            next_ids.push(next_id);
            counts.push(count);
        }

        if next_ids.len() > u32::MAX as usize {
            bail!(
                "number of edges {} exceeds u32::MAX; cannot store offsets as u32",
                next_ids.len()
            );
        }

        offsets.push(next_ids.len() as u32);
    }

    Ok((context_keys, offsets, next_ids, counts))
}

pub fn bytes_to_u32s(bytes: &[u8]) -> Result<Vec<u32>> {
    if bytes.len() % 4 != 0 {
        bail!("byte length {} not divisible by 4", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

pub fn bytes_to_u64s(bytes: &[u8]) -> Result<Vec<u64>> {
    if bytes.len() % 8 != 0 {
        bail!("byte length {} not divisible by 8", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        out.push(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

pub fn u32s_to_bytes(values: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn u64s_to_bytes(values: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn validate_non_decreasing_u64(values: &[u64], label: &str) -> Result<()> {
    for i in 1..values.len() {
        if values[i - 1] > values[i] {
            bail!(
                "{} is decreasing at index {}: {} > {}",
                label,
                i,
                values[i - 1],
                values[i]
            );
        }
    }
    Ok(())
}

pub fn read_file(path: &Path) -> Result<Vec<u8>> {
    let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut data = Vec::new();
    f.read_to_end(&mut data)
        .with_context(|| format!("reading {}", path.display()))?;
    Ok(data)
}

pub fn write_bundle(output: &Path, sections: &[SectionBlob]) -> Result<()> {
    if sections.is_empty() {
        bail!("no sections to write");
    }
    if sections.len() > u32::MAX as usize {
        bail!("too many sections");
    }

    for s in sections {
        if s.name.len() > SECTION_NAME_LEN {
            bail!(
                "section name '{}' too long; max is {} bytes",
                s.name,
                SECTION_NAME_LEN
            );
        }
    }

    let section_count = sections.len() as u32;
    let directory_offset = HEADER_SIZE;
    let data_offset = align8(directory_offset + (section_count as u64) * SECTION_ENTRY_SIZE);

    let mut entries: Vec<SectionEntry> = Vec::with_capacity(sections.len());
    let mut cursor = data_offset;

    for s in sections {
        let length = s.data.len() as u64;
        let count = if s.elem_size == 0 {
            0
        } else {
            length / s.elem_size as u64
        };

        entries.push(SectionEntry {
            name: s.name.clone(),
            kind: s.kind,
            elem_size: s.elem_size,
            offset: cursor,
            length,
            count,
        });

        cursor = align8(cursor + length);
    }

    let header = Header {
        version: VERSION,
        section_count,
        directory_offset,
        data_offset,
    };

    let mut f = File::create(output).with_context(|| format!("creating {}", output.display()))?;
    write_header(&mut f, &header)?;

    for e in &entries {
        write_section_entry(&mut f, e)?;
    }

    let pos = f.stream_position()?;
    if pos < data_offset {
        write_padding(&mut f, (data_offset - pos) as usize)?;
    }

    for (blob, entry) in sections.iter().zip(entries.iter()) {
        let pos = f.stream_position()?;
        if pos < entry.offset {
            write_padding(&mut f, (entry.offset - pos) as usize)?;
        }
        f.write_all(&blob.data)?;
        let pos2 = f.stream_position()?;
        let aligned = align8(pos2);
        if aligned > pos2 {
            write_padding(&mut f, (aligned - pos2) as usize)?;
        }
    }

    Ok(())
}

pub fn read_bundle_table(path: &Path) -> Result<(Header, Vec<SectionEntry>)> {
    let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let header = read_header(&mut f)?;

    if header.version != VERSION {
        bail!("unsupported version {}", header.version);
    }

    f.seek(SeekFrom::Start(header.directory_offset))?;
    let mut sections = Vec::with_capacity(header.section_count as usize);
    for _ in 0..header.section_count {
        sections.push(read_section_entry(&mut f)?);
    }

    Ok((header, sections))
}

pub fn read_section_data(f: &mut File, entry: &SectionEntry) -> Result<Vec<u8>> {
    f.seek(SeekFrom::Start(entry.offset))?;
    let mut buf = vec![0u8; entry.length as usize];
    f.read_exact(&mut buf)?;
    Ok(buf)
}

pub fn split_section_name(name: &str) -> Option<(&str, &str)> {
    name.split_once('.')
}

pub fn dump_path_for_section(output_root: &Path, s: &SectionEntry) -> Result<PathBuf> {
    let (prefix, suffix) =
        split_section_name(&s.name).with_context(|| format!("bad section name '{}'", s.name))?;

    if prefix == "shared" {
        let file_name = match s.kind {
            SectionKind::SharedVocabIdx => "vocab.idx",
            SectionKind::SharedVocabBin => "vocab.bin",
            _ => bail!("unexpected shared section kind for {}", s.name),
        };
        return Ok(output_root.join(file_name));
    }

    let file_name = match s.kind {
        SectionKind::ContextKeys => "context_keys.bin",
        SectionKind::Offsets => "offsets.bin",
        SectionKind::NextIds => "next_ids.bin",
        SectionKind::Counts => "counts.bin",
        SectionKind::StartIds => "start_ids.bin",
        SectionKind::StartCounts => "start_counts.bin",
        _ => bail!("unexpected model section kind for {}", s.name),
    };

    if suffix.is_empty() {
        bail!("bad model section suffix in '{}'", s.name);
    }

    Ok(output_root.join(prefix).join(file_name))
}

fn align8(x: u64) -> u64 {
    (x + 7) & !7
}

fn write_header(f: &mut File, h: &Header) -> Result<()> {
    f.write_all(MAGIC)?;
    f.write_all(&h.version.to_le_bytes())?;
    f.write_all(&h.section_count.to_le_bytes())?;
    f.write_all(&h.directory_offset.to_le_bytes())?;
    f.write_all(&h.data_offset.to_le_bytes())?;
    Ok(())
}

fn read_header(f: &mut File) -> Result<Header> {
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("bad magic: {:?}", magic);
    }

    let version = read_u32(f)?;
    let section_count = read_u32(f)?;
    let directory_offset = read_u64(f)?;
    let data_offset = read_u64(f)?;

    Ok(Header {
        version,
        section_count,
        directory_offset,
        data_offset,
    })
}

fn write_section_entry(f: &mut File, s: &SectionEntry) -> Result<()> {
    let mut name_buf = [0u8; SECTION_NAME_LEN];
    let name_bytes = s.name.as_bytes();
    if name_bytes.len() > SECTION_NAME_LEN {
        bail!("section name too long: {}", s.name);
    }
    name_buf[..name_bytes.len()].copy_from_slice(name_bytes);

    f.write_all(&name_buf)?;
    f.write_all(&s.kind.as_u32().to_le_bytes())?;
    f.write_all(&s.elem_size.to_le_bytes())?;
    f.write_all(&s.offset.to_le_bytes())?;
    f.write_all(&s.length.to_le_bytes())?;
    f.write_all(&s.count.to_le_bytes())?;
    Ok(())
}

fn read_section_entry(f: &mut File) -> Result<SectionEntry> {
    let mut name_buf = [0u8; SECTION_NAME_LEN];
    f.read_exact(&mut name_buf)?;
    let kind = SectionKind::from_u32(read_u32(f)?)?;
    let elem_size = read_u32(f)?;
    let offset = read_u64(f)?;
    let length = read_u64(f)?;
    let count = read_u64(f)?;

    let end = name_buf
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(SECTION_NAME_LEN);
    let name = std::str::from_utf8(&name_buf[..end])
        .context("section name is not valid UTF-8")?
        .to_string();

    Ok(SectionEntry {
        name,
        kind,
        elem_size,
        offset,
        length,
        count,
    })
}

fn read_u32(f: &mut File) -> Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(f: &mut File) -> Result<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn write_padding(f: &mut File, n: usize) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let zeros = vec![0u8; n];
    f.write_all(&zeros)?;
    Ok(())
}
