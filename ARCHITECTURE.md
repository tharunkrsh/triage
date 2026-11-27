# triage — System Architecture

## 1. Tech Stack
- **Language & Packaging:** Python 3.12+, `uv`
- **CLI:** `typer` + `rich` (terminal tables/progress)
- **Validation:** `pydantic` v2
- **Analytics Engine:** `duckdb` (in-memory / ephemeral `.duckdb` file)
- **API Client:** `anthropic` official Python SDK (Batch API)
- **Reporting:** `jinja2` (self-contained HTML + Markdown templates)
- **Config:** `pyyaml` (`pricing.yaml`)
- **Quality:** `pytest`, `ruff`

## 2. Directory Layout
```text
triage/
├── pyproject.toml
├── pricing.yaml                 # Single source of truth for Anthropic pricing
├── SPEC.md
├── ARCHITECTURE.md
├── CLAUDE.md
├── src/
│   └── triage/
│       ├── __init__.py
│       ├── cli.py               # Typer entrypoint (`ingest`, `audit`, `simulate`)
│       ├── models.py            # Canonical Pydantic schemas
│       ├── pricing.py           # Loads pricing.yaml + cost calculation logic
│       ├── ingest/
│       │   ├── __init__.py      # Format detection and `load` (file, folder or glob)
│       │   ├── base.py          # Adapter interface, result models and shared helpers
│       │   ├── paths.py         # Resolves a path argument into a set of input files
│       │   ├── jsonl.py         # Canonical JSONL reader + replay-task mapping
│       │   ├── anthropic_csv.py # Anthropic Console export adapters (CSV and JSON)
│       │   ├── claude_code.py   # Claude Code session logs, deduplicated across files
│       │   └── replay.py        # Loads replay tasks for `simulate`
│       ├── audit/
│       │   ├── bleed.py         # Runs the bleed detectors
│       │   ├── sql.py           # DuckDB SQL for Mode 1 bleed detection
│       │   └── models.py        # Audit result schemas
│       ├── simulate/
│       │   ├── estimate.py      # Batch API spend estimate (shown before any paid call)
│       │   ├── batch.py         # Anthropic Batch API submission & polling
│       │   ├── eval_det.py      # Deterministic JSON/schema/tool-call checks
│       │   ├── eval_judge.py    # LLM-as-a-judge scorer
│       │   ├── stats.py         # Wilson score 95% confidence intervals
│       │   ├── summarize.py     # DuckDB aggregates per feature, prompt version and model
│       │   ├── charts.py        # Inline SVG charts for the report
│       │   ├── cli_helpers.py   # Sampling, terminal tables, report writing
│       │   └── models.py        # Replay task and outcome schemas
│       └── report/
│           ├── builder.py       # Jinja2 renderer and formatting helpers
│           └── templates/
│               ├── report.html.j2
│               ├── report.md.j2
│               ├── simulate.html.j2
│               └── simulate.md.j2
└── tests/
    ├── fixtures/                # Sample JSONL + Console CSV files
    └── ...
```
