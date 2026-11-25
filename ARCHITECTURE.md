ARCHITECTURE.md
# tokenize — System Architecture

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
tokenize/
├── pyproject.toml
├── pricing.yaml                 # Single source of truth for Anthropic pricing
├── SPEC.md
├── ARCHITECTURE.md
├── CLAUDE.md
├── src/
│   └── tokenize_cli/            # Named tokenize_cli to avoid stdlib `tokenize` clash
│       ├── __init__.py
│       ├── cli.py               # Typer entrypoint (`audit`, `simulate`)
│       ├── models.py            # Canonical Pydantic schemas
│       ├── pricing.py           # Loads pricing.yaml + cost calculation logic
│       ├── ingest/
│       │   ├── base.py          # Adapter protocol
│       │   ├── jsonl.py         # Canonical JSONL reader
│       │   └── anthropic_csv.py # Anthropic Console export adapter
│       ├── audit/
│       │   └── bleed.py         # DuckDB SQL queries for Mode 1 bleed detection
│       ├── simulate/
│       │   ├── batch.py         # Anthropic Batch API submission & polling
│       │   ├── eval_det.py      # Deterministic JSON/schema/tool-call checks
│       │   ├── eval_judge.py    # LLM-as-a-judge scorer
│       │   └── stats.py         # Wilson score 95% confidence intervals
│       └── report/
│           ├── builder.py       # DuckDB aggregations + Jinja2 renderer
│           └── templates/
│               ├── report.html.j2
│               └── report.md.j2
└── tests/
    ├── fixtures/                # Sample JSONL + Console CSV files
    └── ...