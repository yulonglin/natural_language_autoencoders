# GPT-OSS-20B Actor Proxy Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable offline-first harness that scores cached held-out GPT-OSS-20B actor verbalizations on all six requested axes, preserving separate round-trip-cosine and LLM-grounding faithfulness components.

**Architecture:** `evaluate.proxy_metrics_models` owns schemas/settings, `evaluate.proxy_metrics_judge` owns the versioned blind prompt and API adapter, and `evaluate.proxy_metrics` owns strict parquet ingestion, deterministic diagnostics, serialization, aggregation, and the CLI. The entrypoint re-exports the public judge types. Tests inject a fake Responses API client, so all implementation verification is network/GPU/checkpoint free.

**Tech Stack:** Python 3.10+, argparse, concurrent.futures, hashlib/json/logging/pathlib/statistics, PyArrow, Pydantic, pydantic-settings, optional OpenAI Python SDK at live-call time, pytest, ruff.

## Global Constraints

- New implementation code lives under `nla_gptoss/evaluate/`; tests live under `nla_gptoss/tests/`.
- Do not modify `monitorability-evals/` or `natural_language_autoencoders/`.
- Do not blend cached round-trip cosine with LLM grounding.
- Do not penalize novelty merely because it is surprising or independently unverified.
- Do not run a live 3-5 item integration in this environment.
- Do not push or bump the package version.

---

### Task 1: Input, deterministic metrics, and aggregation

**Files:**
- Create: `evaluate/__init__.py`
- Create: `evaluate/proxy_metrics.py`
- Create: `evaluate/proxy_metrics_models.py`
- Create: `evaluate/proxy_metrics_judge.py`
- Create: `tests/test_proxy_metrics.py`

**Interfaces:**
- Consumes: parquet path and configurable `content`, `explanation`, `cosine_similarity`, and optional parse-status column names.
- Produces: `load_input_items(...) -> list[InputItem]`, `degeneracy_stats(text, parse_ok) -> DegeneracyStats`, `quality_cosine(raw) -> float`, and `summarize_records(records) -> dict[str, Any]`.

- [x] **Step 1: Write failing strict-input and deterministic-metric tests**

```python
def test_load_input_items_reuses_roundtrip_columns(tmp_path: Path) -> None:
    path = tmp_path / "roundtrip.parquet"
    pq.write_table(pa.table({
        "content": ["alpha", "beta"],
        "explanation": ["specific alpha", "specific beta"],
        "cosine_similarity": [0.75, -0.2],
    }), path)
    items = load_input_items(path, limit=1, offset=1)
    assert items[0].source_row == 1
    assert items[0].cosine_similarity == pytest.approx(-0.2)

def test_degeneracy_stats_and_cosine_gate_direction() -> None:
    stats = degeneracy_stats("loop loop loop loop", parse_ok=None)
    assert stats.word_count == 4
    assert stats.parse_ok is None
    assert quality_cosine(-0.2) == 0.0
    assert quality_cosine(0.75) == 0.75
```

- [x] **Step 2: Run tests and confirm RED**

Run: `uv run --no-sync pytest tests/test_proxy_metrics.py -q`

Expected: collection/import failure because `evaluate.proxy_metrics` does not exist.

- [x] **Step 3: Implement strict models, ingestion, diagnostics, and aggregation**

Implement typed Pydantic/dataclass records; validate text types and finite cosine in `[-1,1]`; tail-preserve source row indices; compute word count, maximum trigram repetition, non-ASCII, CJK, non-word, and distinct-word ratios; clip raw cosine to `[0,1]`; aggregate each quality score as `{count, mean, standard_error}`.

- [x] **Step 4: Run tests and confirm GREEN**

Run: `uv run --no-sync pytest tests/test_proxy_metrics.py -q`

Expected: deterministic/input tests pass.

### Task 2: Novelty-safe structured judge

**Files:**
- Modify: `evaluate/proxy_metrics.py`
- Modify: `tests/test_proxy_metrics.py`

**Interfaces:**
- Consumes: `InputItem`, `JudgeSettings`, and an injectable OpenAI-compatible client.
- Produces: `JudgeVerdict`, `render_judge_input(...) -> str`, `OpenAIJudge.score(...) -> JudgeCall`, and `build_item_record(...) -> dict[str, Any]`.

- [x] **Step 1: Write failing prompt/schema/API tests**

```python
def test_prompt_protects_plausible_unverified_novelty() -> None:
    prompt = render_judge_input(InputItem(
        source_row=0, content="eval question", explanation="possible eval awareness",
        cosine_similarity=0.8, parse_ok=None,
    ), context_chars=100)
    assert "plausible but unverified" in JUDGE_INSTRUCTIONS.lower()
    assert "HALLUCINATION" in JUDGE_INSTRUCTIONS
    assert "GROUNDING" in JUDGE_INSTRUCTIONS
    assert "0.8" not in prompt

def test_item_record_keeps_faithfulness_components_separate(fake_judge_call: JudgeCall) -> None:
    record = build_item_record(INPUT, fake_judge_call)
    assert record["metrics"]["roundtrip_cosine"] == 0.8
    assert record["metrics"]["llm_grounding"] == 0.75
    assert "faithfulness" not in record["metrics"]
    assert record["subrubrics"]["grounding"]["verifiability"] == "plausible_but_unverified"
```

- [x] **Step 2: Run focused tests and confirm RED**

Run: `uv run --no-sync pytest tests/test_proxy_metrics.py -q`

Expected: failures for missing judge schema, prompt, and item-record functions.

- [x] **Step 3: Implement prompt, Pydantic schemas, and Responses API adapter**

Use separate `NoveltySensitiveRating` values for hallucination and grounding with `quality_score`, `plausibility_score`, `verifiability`, and rationale. Use `FalseFactsRating` claim/contradiction counts for the actual false-facts rate and `AxisRating` for coherence, relevance, and information gain. Invoke `client.responses.parse(..., text_format=JudgeVerdict, reasoning={"effort": ...}, store=False)`; fail on incomplete/refused/unparsed output; serialize the raw response while omitting API keys.

- [x] **Step 4: Run focused tests and confirm GREEN**

Run: `uv run --no-sync pytest tests/test_proxy_metrics.py -q`

Expected: prompt/schema/API tests pass.

### Task 3: CLI, JSONL, provenance, and documentation

**Files:**
- Modify: `evaluate/proxy_metrics.py`
- Modify: `tests/test_proxy_metrics.py`
- Modify: `pyproject.toml`
- Verify: `evaluate/proxy_metrics_design.md`

**Interfaces:**
- Consumes: CLI flags plus `NLA_PROXY_*` settings.
- Produces: ordered per-item JSONL and aggregate summary JSON with input/prompt SHA-256, UTC timing, non-secret settings, all metric summaries, and the deferred live-run command.

- [x] **Step 1: Write failing CLI test with a fake judge factory**

```python
def test_run_pipeline_writes_jsonl_and_complete_summary(tmp_path: Path) -> None:
    records, summary = run_pipeline(
        input_path=ROUNDTRIP_PARQUET,
        output_jsonl=tmp_path / "items.jsonl",
        summary_json=tmp_path / "summary.json",
        settings=TEST_SETTINGS,
        judge=FakeJudge(),
        limit=2,
    )
    assert len(records) == 2
    assert set(summary["metrics"]) == {
        "hallucination", "false_facts", "coherence", "relevance_to_context",
        "roundtrip_cosine", "llm_grounding", "information_gain",
    }
    assert summary["metrics"]["coherence"]["count"] == 2
    assert summary["input"]["sha256"]
```

- [x] **Step 2: Run focused tests and confirm RED**

Run: `uv run --no-sync pytest tests/test_proxy_metrics.py -q`

Expected: failure for missing pipeline/CLI serialization.

- [x] **Step 3: Implement concurrency, ordered writes, CLI, settings, and dependency declaration**

Add `pydantic-settings` to dependencies without changing the package version. Execute independent judge calls through `ThreadPoolExecutor(max_workers=settings.concurrency)`, preserve input order when serializing, create parent directories, write JSONL and indented summary JSON, log paths and all non-secret configuration, and expose column overrides plus `--limit`/`--offset`.

- [x] **Step 4: Run full offline verification**

Run:

```bash
uv run --no-sync pytest tests/test_proxy_metrics.py -q
uv run --no-sync ruff check evaluate/proxy_metrics.py tests/test_proxy_metrics.py
uv run --no-sync python -m evaluate.proxy_metrics --help
```

Expected: all tests pass, ruff reports no errors, and CLI help lists required parquet/output paths and `NLA_PROXY_*` configuration.

- [x] **Step 5: Inspect the final diff and acceptance criteria**

Run: `git diff --check && git diff --stat && git status --short`

Expected: no whitespace errors; only the planned `nla_gptoss/` files are changed. Do not commit or push; hand the verified diff to the user.
