# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**treet** is a file tree scanning and analysis CLI utility. It recursively scans directories, collects file metadata (size, timestamps, permissions, ownership), optionally identifies file types via ML (Magika), and outputs results in JSON, JSONL, YAML, CSV, or XLSX formats. It can also render a visual directory tree.

## Commands

This project uses [uv](https://docs.astral.sh/uv/) as the package manager.

```bash
# Install dependencies
uv sync

# Run the CLI
uv run treet <root_directory> [options]

# Build the package
uv build
```

CLI flags: `--stat` (collect os.stat metadata), `--magika` (ML file type detection), `--pretty` (human-readable formatting), `--output <file>` (write to file), `--tree <depth>` (render tree visualization).

There is no test suite or linting configuration at this time.

## Architecture

All core logic lives in `treet-py/src/treet/__init__.py`. The console script entry point is `treet:app` (Typer `app`); `main` is the Typer command that runs the scan pipeline.

Key classes:
- `Config` — dataclass holding the root path and all feature flags
- `FileInfo` — central data model for a single file/directory entry; uses `threading.Lock` for concurrent access; tracks per-function errors
- `FileInfoOutputOptions` — controls serialization (relative paths, sort order, pretty printing)
- `SimpleProgress` — custom Rich-based progress bar

Key functions:
- `scan_path()` / `do_scan()` — recursive directory walker
- `do_magika()` — concurrent file type identification via `concurrent_map()`
- `concurrent_map()` — thin wrapper around `ThreadPoolExecutor`
- `do_output()` — dispatches to format-specific writers (JSON, JSONL, YAML, CSV, XLSX via pandas/openpyxl)
- `do_tree()` — renders Rich tree to console
- `app` / `main()` — Typer application and command entry

Platform notes: `FileInfo` handles macOS/Windows-specific stat fields (`st_birthtime`, `st_ctime` semantics differ by OS).
