`.lexi` bundle format
---------------------

This document specifies the `lexicore` single-file binary bundle format.

Format name:
- `.lexi`

Magic:
- `TGRMBNDL` (stands for "three-gram bundle")

Endianness:
- little-endian throughout

Alignment:
- section payloads are 8-byte aligned

Version in this document:
- `1`

### High-level structure

A `.lexi` file contains:
1. fixed-size file header
2. fixed-size section directory
3. section payloads

Section payloads store:
- shared vocabulary
- one or more named trigram models

Each trigram model stores:
- packed trigram context keys
- candidate list offsets
- next-token ids
- counts
- explicit sentence-start distribution

### Header

The file header is 32 bytes.

| Offset | Size | Type      | Meaning                   |
|-------:|-----:|-----------|---------------------------|
|      0 |    8 | `[u8; 8]` | magic = `TGRMBNDL`        |
|      8 |    4 | `u32`     | format version            |
|     12 |    4 | `u32`     | section count             |
|     16 |    8 | `u64`     | section directory offset  |
|     24 |    8 | `u64`     | first data section offset |

For version `1`:
- `magic = b"TGRMBNDL"`
- `version = 1`
- `section directory offset = 32`

### Section directory

Each section directory entry is 64 bytes.

| Offset | Size | Type       | Meaning                         |
|-------:|-----:|------------|---------------------------------|
|      0 |   32 | `[u8; 32]` | UTF-8 section name, zero-padded |
|     32 |    4 | `u32`      | section kind                    |
|     36 |    4 | `u32`      | element size in bytes           |
|     40 |    8 | `u64`      | byte offset of section payload  |
|     48 |    8 | `u64`      | byte length of section payload  |
|     56 |    8 | `u64`      | element count                   |

Notes:
- section names are stored as UTF-8 bytes followed by zero padding
- section payloads are aligned to 8 bytes

### Section kinds

| Kind value | Meaning                | Element type |
|-----------:|------------------------|--------------|
|          1 | `shared.vocab_idx`     | `u64[]`      |
|          2 | `shared.vocab_bin`     | raw bytes    |
|          3 | `<model>.context_keys` | `u64[]`      |
|          4 | `<model>.offsets`      | `u32[]`      |
|          5 | `<model>.next_ids`     | `u32[]`      |
|          6 | `<model>.counts`       | `u32[]`      |
|          7 | `<model>.start_ids`    | `u32[]`      |
|          8 | `<model>.start_counts` | `u32[]`      |

### Shared vocabulary sections

#### `shared.vocab_idx`

Type:
- `u64[]`

Length:
- `vocab_size + 1`

Meaning:
- byte offsets into `shared.vocab_bin`

For token id `i`:
- start byte = `vocab_idx[i]`
- end byte = `vocab_idx[i + 1]`

Constraints:
- `vocab_idx[0] = 0`
- non-decreasing
- `vocab_idx[last] = len(shared.vocab_bin)`

#### `shared.vocab_bin`

Type:
- raw concatenated UTF-8 bytes

Meaning:
- token strings concatenated without separators

Token `i` is the UTF-8 slice:

```text
shared.vocab_bin[vocab_idx[i] .. vocab_idx[i+1]]
```

### Per-model sections

For each model name `m`, the bundle contains:

- `m.context_keys`
- `m.offsets`
- `m.next_ids`
- `m.counts`
- `m.start_ids`
- `m.start_counts`

All token ids refer to the shared vocabulary.

### Packed context key

A trigram context `(w1, w2)` is packed into one `u64`:

```text
key = ((w1 as u64) << 32) | (w2 as u64)
```

High 32 bits:
- `w1`

Low 32 bits:
- `w2`

Because contexts are sorted lexicographically by `(w1, w2)`, sorting packed `u64` keys preserves the same order.

#### `m.context_keys`

Type:
- `u64[]`

Length:
- `num_contexts`

Meaning:
- sorted packed trigram context keys

Constraints:
- strictly increasing
- each key decodes to valid token ids

### Candidate edge arrays

#### `m.offsets`

Type:
- `u32[]`

Length:
- `num_contexts + 1`

Meaning:
- slice boundaries into `m.next_ids` and `m.counts`

For context row `i`:

- candidate start index = `offsets[i]`
- candidate end index = `offsets[i + 1]`

Constraints:

- `offsets[0] = 0`
- non-decreasing
- `offsets[last] = len(m.next_ids) = len(m.counts)`

Because offsets are `u32`, total edge count must satisfy:

```text
num_edges <= u32::MAX
```

#### `m.next_ids`

Type:
- `u32[]`

Length:
- `num_edges`

Meaning:
- flattened next-token ids

For context row `i`, valid candidates are:

```text
next_ids[offsets[i] .. offsets[i+1]]
```

#### `m.counts`

Type:
- `u32[]`

Length:
- `num_edges`

Meaning:
- raw counts parallel to `m.next_ids`

For candidate index `j`:
- `next_ids[j]` is the next token id
- `counts[j]` is the raw count for that trigram transition

Constraints:
- `counts[j] >= 1`
- `len(counts) = len(next_ids)`

### Start-token distribution

The training convention is:

```text
<s> token1 token2 ... tokenN </s>
```

That means there is no explicit context:

```text
(<s>, <s>) -> token1
```

Instead, sentence start is represented indirectly through contexts:

```text
(<s>, token1) -> token2
```

To avoid reconstructing sentence-start distribution at load time, each model stores it explicitly.

#### `m.start_ids`

Type:
- `u32[]`

Length:
- `num_starts`

Meaning:
- possible first token ids after `<s>`

#### `m.start_counts`

Type:
- `u32[]`

Length:
- `num_starts`

Meaning:
- total outgoing count mass for the corresponding start token

For index `i`:

- `start_ids[i]` is a possible first token
- `start_counts[i]` is the total count mass of context `(<s>, start_ids[i])`

Constraints:

- `len(start_ids) = len(start_counts)`
- all counts are positive

### Sampling semantics

#### Sentence start

To start a sequence:

1. sample from `m.start_ids` weighted by `m.start_counts`
2. returned token becomes `token1`
3. active context becomes `(<s>, token1)`

#### Regular next token

Given active context `(w1, w2)`:

1. pack it into a `u64` key
2. binary-search `m.context_keys`
3. if found at row `i`, read slice:
   - `lo = offsets[i]`
   - `hi = offsets[i+1]`
4. sample one `next_ids[lo..hi]` weighted by `counts[lo..hi]`

### Required ordering

#### Context ordering
`m.context_keys` must be strictly increasing.

#### Candidate ordering
The current builder emits candidates in sorted order because it flattens ordered maps. Consumers should not require candidate order for correctness, but deterministic ordering is expected.

#### Start ordering
The current builder emits `start_ids` in the same order as start contexts appear in `context_keys`, which means ascending token id order within the `<s>` prefix range.

### Example logical model

Suppose shared vocab is:

| Token id | Token  |
|---------:|--------|
|        0 | `<s>`  |
|        1 | `</s>` |
|        2 | `a`    |
|        3 | `b`    |
|        4 | `x`    |

Suppose one model has these contexts:
- `(<s>, a) -> b : 2, x : 1`
- `(a, b) -> </s> : 2`
- `(a, x) -> </s> : 1`

Then:

#### `context_keys`
- pack `(<s>, a)` as `((0 << 32) | 2)`
- pack `(a, b)` as `((2 << 32) | 3)`
- pack `(a, x)` as `((2 << 32) | 4)`

#### `offsets`
```text
[0, 2, 3, 4]
```

#### `next_ids`
```text
[b, x, </s>, </s>]
```

in ids:

```text
[3, 4, 1, 1]
```

#### `counts`
```text
[2, 1, 2, 1]
```

#### `start_ids`
```text
[a]
```

in ids:

```text
[2]
```

#### `start_counts`
Total outgoing mass from context `(<s>, a)` is `2 + 1 = 3`, so:

```text
[3]
```
