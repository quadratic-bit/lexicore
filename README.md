lexicore
--------

`lexicore` is a CLI for building, inspecting, dumping, repacking, and inferencing compact trigram-model bundles.

The project features a single-file bundle format `.lexi`, alongside a CLI tool
for interacting with such files.

### Build

```
cargo build --release
```

### CLI overview

```
lexicore <command> [args]
```

Commands:

- `tokenize` — read text files, tokenize them, build models, and write one `.lexi` bundle
- `parse` — inspect a `.lexi` bundle and print its section table
- `dump` — dump a `.lexi` bundle into split binary files on disk
- `pack` — repack a dumped split layout back into a `.lexi` bundle
- `infer` — load a `.lexi` bundle and generate text using two named models

> [!NOTE]
> Currently, only inference of two models is supported, and is to be generalized in the future.

#### Build a bundle from text files

For two named models:

```
cargo run -- tokenize \
  --input ./corpus_a.txt --model alpha \
  --input ./corpus_b.txt --model beta \
  --output ./bundle.lexi
```

#### Inspect a bundle

```
cargo run -- parse --input ./bundle.lexi
```

#### Dump a bundle into split files

```
cargo run -- dump \
  --input ./bundle.lexi \
  --output-root ./dumped
```

Resulting shape:

```text
dumped/
├── vocab.bin
├── vocab.idx
├── alpha/
│   ├── context_keys.bin
│   ├── offsets.bin
│   ├── next_ids.bin
│   ├── counts.bin
│   ├── start_ids.bin
│   └── start_counts.bin
└── beta/
    ├── context_keys.bin
    ├── offsets.bin
    ├── next_ids.bin
    ├── counts.bin
    ├── start_ids.bin
    └── start_counts.bin
```

#### Repack a dumped layout

```
cargo run -- pack \
  --input-root ./dumped \
  --output ./bundle2.lexi \
  --model alpha \
  --model beta
```

#### Run inference

Generate a fixed number of tokens:

```
cargo run -- infer \
  --input ./bundle.lexi \
  --primary alpha \
  --secondary beta \
  --alpha 0.03 \
  --beta 0.25 \
  --mode tokens \
  --size 200 \
  --seed 42
```

Generate a fixed number of sentences:

```
cargo run -- infer \
  --input ./bundle.lexi \
  --primary alpha \
  --secondary beta \
  --alpha 0.03 \
  --beta 0.25 \
  --mode sentences \
  --size 20 \
  --seed 42
```

##### Inference parameters

- `--primary <name>` — first model
- `--secondary <name>` — second model
- `--alpha <p>` — probability of switching from primary to secondary before sampling a token
- `--beta <p>` — probability of switching from secondary back to primary before sampling a token
- `--mode tokens|sentences`
- `--size <n>`
- `--seed <u64>`

##### Inference behavior

The generator keeps a current active model:

- if active is `primary`, switch to `secondary` with probability `alpha`
- if active is `secondary`, switch back with probability `beta`

At each step:

1. maybe switch model
2. sample from the active model
3. if that context is missing, try the other model
4. if both fail, emit `</s>`

### Tokenization rules

The tokenizer is a manual single-pass Unicode scanner.
- a word starts with a Unicode letter or digit, and continues as such
- internal joiners are allowed only when already inside a word and followed by another word character:
  - `'`
  - `’`
  - `-`
- whitespace separates tokens
- selected punctuation is emitted as standalone tokens:
  - `. , ! ? ; : ( ) " „ ” « » … -`
- anything else is ignored


### File format

Read `FORMAT.md` for the binary layout of bundle files.
