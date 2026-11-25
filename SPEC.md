SPEC.md
# tokenize — Product Specification (v1)

## 1. Overview
`tokenize` is a local-first Python CLI for auditing Anthropic Claude API spend and simulating model downgrades offline. It runs entirely on historical usage exports or anonymised logs with zero production traffic risk.

## 2. Core Modes (v1)

### Mode 1: Metadata Bleed Audit (`tokenize audit`)
Runs locally with **zero API calls** and **zero prompt text required**.
- **Inputs:** Canonical JSONL log file OR Anthropic Console Usage/Cost export (CSV/JSON), including Claude Code usage broken down by API key/user.
- **Bleed Detectors:**
  1. **Uncached Prompt Bleed:** Flags high-volume requests with `input_tokens >= 1024` and `cache_read_input_tokens == 0`. If `system_prompt_hash` (or raw text) is present, groups exact repeats to calculate exact 5-minute cache payback; otherwise flags as a metadata heuristic.
  2. **Haiku 100k Cliff:** Flags Haiku requests crossing the 100,000-token threshold where input/output pricing jumps 5x.
  3. **Bloated `max_tokens`:** Flags requests where `max_tokens` is >4x actual `output_tokens` (risking unnecessary rate-limit reservation).
  4. **Spend Concentration / Claude Code Burn:** Breaks down spend by `model`, `feature`, and `api_key_id` (highlighting top Claude Code / API spenders).
- **Output:** Terminal summary table + standalone HTML/Markdown report.

### Mode 2: Offline Downgrade Simulator (`tokenize simulate`)
Replays historical single-turn prompts across cheaper Claude tiers to measure **cost-per-successful-task**.
- **Inputs:** Canonical JSONL containing prompt messages, baseline model/response, and optional expected schema or tool definitions.
- **Execution:** Submits replays via the **Anthropic Message Batches API** (50% cost discount) against target candidate models (e.g., `haiku-5-5`, `sonnet-5-5`).
- **Evaluation Pipeline:**
  1. **Deterministic Checks:** Valid JSON parsing, JSON Schema validation, and single-step tool-call name/argument match (no live tool execution).
  2. **LLM-as-a-Judge:** Semantic parity grading against the baseline response, reporting agreement rate between deterministic checks and judge scores.
- **Statistical Rigor:** Reports pass rates with **95% Wilson score confidence intervals** (never raw percentages without sample size `n` and error bars).
- **Output:** DuckDB-backed Pareto Frontier HTML/Markdown report plotting Cost vs. Accuracy per `feature` / `prompt_version`.

## 3. Explicit Non-Goals (Out of Scope for v1)
- **No live proxy or production middleware.**
- **No multi-turn agentic replay or live tool execution** (single-turn and single-step tool-call validation only).
- **No Langfuse, Helicone, LangSmith, or OpenTelemetry adapters** (v2).
- **No GitHub Action CI release gate** (v2).
- **No hosted web UI, database server, or SaaS backend.**

## 4. v1 Definition of Done
- `tokenize audit <file>` runs on both canonical JSONL and Anthropic Console exports and generates a valid HTML + Markdown report.
- `tokenize simulate <file> --target haiku-5-5` runs end-to-end via Anthropic Batch API (with a `--dry-run` flag that estimates replay cost before spending money).
- Full test suite passes with mocked Anthropic API responses (`pytest`).