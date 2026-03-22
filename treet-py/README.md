# treet

A CLI utility for scanning and analyzing directory trees. It recursively walks a directory, collects file metadata, optionally identifies file types via ML, and outputs results in multiple formats.

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```bash
uv run treet <root_directory> [options]
```

### Options

| Flag | Description |
|------|-------------|
| `--stat` | Collect `os.stat` metadata (size, timestamps, permissions, ownership) |
| `--magika` | Identify file types using ML ([Magika](https://github.com/google/magika)) |
| `--pretty` | Convert raw values to human-readable formats (e.g. `4.2 MB`, `rwxr-xr-x`) |
| `--output <file>` | Write results to a file (format determined by extension) |
| `--tree <depth>` | Render a visual directory tree to the console at the given depth |

### Output Formats

Output format is determined by the file extension passed to `--output`:

| Extension | Format |
|-----------|--------|
| `.json` | JSON array |
| `.jsonl` | JSON Lines (one record per line) |
| `.yaml` | YAML |
| `.csv` | CSV |
| `.xlsx` | Excel spreadsheet |

### Examples

```bash
# Scan a directory and print a tree (depth 2)
uv run treet /path/to/dir --tree 2

# Collect stat metadata and write to JSON
uv run treet /path/to/dir --stat --output results.json

# Full scan with ML file type detection, pretty output to Excel
uv run treet /path/to/dir --stat --magika --pretty --output results.xlsx
```

## Output Fields

Each record includes:

- `path` — file path (relative to root by default)
- `depth` — directory depth from root

With `--stat`:
- `st_size`, `st_mtime`, `st_ctime`, `st_atime`, `st_birthtime` (macOS), `st_mode`, `st_uid`, `st_gid`
- With `--pretty`: `pretty_size`, `pretty_mtime`, `pretty_ctime`, `pretty_atime`, `pretty_birthtime`, `pretty_mode`, `pretty_uid`, `pretty_gid`

With `--magika`:
- `magika_group`, `magika_label`, `magika_mime_type`

## Build

```bash
uv build
```
