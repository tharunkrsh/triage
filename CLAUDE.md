---

## 3. `CLAUDE.md`

```markdown
# CLAUDE.md — Project Rules for `tokenize`

## Stack & Commands
- Package manager: `uv` (never use bare `pip`).
- Run CLI: `uv run tokenize --help`
- Run tests: `uv run pytest`
- Lint & format: `uv run ruff check . && uv run ruff format --check .`

## Architecture Rules
- **Package name:** Always import from `tokenize_cli` (never name a directory `tokenize/` inside `src/` to avoid shadowing Python's stdlib `tokenize`).
- **No hardcoded pricing:** All model costs, cliffs, cache multipliers, and batch discounts must be read from `pricing.yaml` via `pricing.py`.
- **Single-turn only:** Do not build multi-turn conversation loops or live tool execution in v1.
- **Zero-spend safety:** `tokenize simulate` must support `--dry-run` and display estimated Batch API spend before making paid API calls.

## Coding Conventions
- Use `pydantic` v2 models for all data boundaries; avoid passing raw untyped dicts between modules.
- Put analytical aggregations and bleed detection in DuckDB SQL rather than slow Python `for` loops.
- Keep modules focused and under ~250 lines where practical.
- Every new module requires unit tests in `tests/`. Mock all Anthropic API calls in tests—tests must run offline with zero API keys.

## Guardrails
- Never commit `.env`, API keys, or real customer logs.
- Before editing code on a new task, inspect the relevant files and state your plan first.
- After completing any task, run `uv run pytest` and `uv run ruff check .` before considering it done.