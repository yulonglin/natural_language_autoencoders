"""Proxy metrics for cached GPT-OSS-20B actor verbalizations.

The input MUST be an output parquet from ``evaluate/roundtrip_from_text.py``
with ``content``, ``explanation``, and ``cosine_similarity`` columns. This
module never loads a checkpoint or regenerates actor text.

Live judging requires the optional ``openai`` SDK plus ``OPENAI_API_KEY`` (or
``NLA_PROXY_OPENAI_API_KEY``). All other settings use the ``NLA_PROXY_``
prefix; CLI flags override non-secret judge settings.

Example once the cache and API key are reachable on RunPod or Silico::

    python -m evaluate.proxy_metrics \
        --input-parquet /workspace/data/gptoss20b_sft_roundtrip.parquet \
        --output-jsonl /workspace/results/gptoss20b_proxy_items.jsonl \
        --summary-json /workspace/results/gptoss20b_proxy_summary.json \
        --limit 5

The cached raw cosine and LLM grounding score are reported separately; they
are never blended into a single faithfulness number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from evaluate.proxy_metrics_judge import JUDGE_INSTRUCTIONS, PROMPT_VERSION, OpenAIJudge, render_judge_input
from evaluate.proxy_metrics_models import (
    AxisRating,
    DegeneracyStats,
    FalseFactsRating,
    InputItem,
    JudgeCall,
    JudgeSettings,
    JudgeVerdict,
    NoveltySensitiveRating,
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_MIN_CONFIDENT_SAMPLE_COUNT = 4
LOGGER = logging.getLogger(__name__)

__all__ = [
    "AxisRating",
    "FalseFactsRating",
    "InputItem",
    "JUDGE_INSTRUCTIONS",
    "JudgeCall",
    "JudgeSettings",
    "JudgeVerdict",
    "NoveltySensitiveRating",
    "OpenAIJudge",
    "PROMPT_VERSION",
    "build_item_record",
    "degeneracy_stats",
    "load_input_items",
    "main",
    "parse_args",
    "quality_cosine",
    "render_judge_input",
    "run_pipeline",
    "summarize_values",
]


def _validate_cosine(value: object, *, source_row: int | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        location = f" at source row {source_row}" if source_row is not None else ""
        raise ValueError(f"cosine_similarity must be numeric{location}")
    cosine = float(value)
    if not math.isfinite(cosine) or not -1.0 <= cosine <= 1.0:
        location = f" at source row {source_row}" if source_row is not None else ""
        raise ValueError(f"cosine_similarity must be finite and in [-1, 1]{location}: {cosine!r}")
    return cosine


def quality_cosine(cosine_similarity: float) -> float:
    """Return the higher-is-better [0, 1] gate score while preserving raw cosine elsewhere."""
    cosine = _validate_cosine(cosine_similarity)
    return max(0.0, cosine)


def _validate_text(value: object, *, column: str, source_row: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{column} must be a string at source row {source_row}")
    return value


def load_input_items(
    path: Path,
    *,
    content_column: str = "content",
    explanation_column: str = "explanation",
    cosine_column: str = "cosine_similarity",
    parse_ok_column: str = "explanation_parse_ok",
    limit: int | None = None,
    offset: int = 0,
) -> list[InputItem]:
    """Load cached round-trip rows without regenerating actor verbalizations."""
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")

    parquet_path = Path(path)
    schema_names = set(pq.read_schema(parquet_path).names)
    required = {content_column, explanation_column, cosine_column}
    missing = sorted(required - schema_names)
    if missing:
        raise ValueError(f"input parquet is missing required columns: {', '.join(missing)}")

    columns = [content_column, explanation_column, cosine_column]
    has_parse_ok = parse_ok_column in schema_names
    if has_parse_ok:
        columns.append(parse_ok_column)
    rows = pq.read_table(parquet_path, columns=columns).to_pylist()
    stop = None if limit is None else offset + limit
    selected = rows[offset:stop]

    items: list[InputItem] = []
    for selected_index, row in enumerate(selected):
        source_row = offset + selected_index
        parse_ok = row.get(parse_ok_column) if has_parse_ok else None
        if parse_ok is not None and not isinstance(parse_ok, bool):
            raise ValueError(f"{parse_ok_column} must be boolean or null at source row {source_row}")
        items.append(
            InputItem(
                source_row=source_row,
                content=_validate_text(row[content_column], column=content_column, source_row=source_row),
                explanation=_validate_text(row[explanation_column], column=explanation_column, source_row=source_row),
                cosine_similarity=_validate_cosine(row[cosine_column], source_row=source_row),
                parse_ok=parse_ok,
            )
        )
    return items


def _max_trigram_repeat_fraction(words: list[str]) -> float:
    if len(words) < 3:
        return 0.0
    trigrams = [tuple(words[index : index + 3]) for index in range(len(words) - 2)]
    return max(Counter(trigrams).values()) / len(trigrams)


def _character_fraction(text: str, predicate: Any) -> float:
    if not text:
        return 0.0
    return sum(1 for character in text if predicate(character)) / len(text)


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )


def degeneracy_stats(explanation: str, parse_ok: bool | None) -> DegeneracyStats:
    """Compute no-LLM diagnostics without interpreting semantic quality."""
    words = _WORD_RE.findall(explanation.lower())
    non_whitespace = [character for character in explanation if not character.isspace()]
    nonword_fraction = (
        sum(1 for character in non_whitespace if not (character.isalnum() or character == "_")) / len(non_whitespace)
        if non_whitespace
        else 0.0
    )
    return DegeneracyStats(
        parse_ok=parse_ok,
        empty=not bool(explanation.strip()),
        word_count=len(words),
        max_trigram_repeat_fraction=_max_trigram_repeat_fraction(words),
        non_ascii_fraction=_character_fraction(explanation, lambda character: ord(character) > 127),
        cjk_fraction=_character_fraction(explanation, _is_cjk),
        nonword_fraction=nonword_fraction,
        distinct_word_ratio=len(set(words)) / len(words) if words else 0.0,
    )


def summarize_values(values: list[float]) -> dict[str, int | float | None]:
    """Return the required count/mean plus standard error for one metric."""
    if not values:
        return {"count": 0, "mean": None, "standard_error": None}
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"cannot summarize a non-finite value: {value!r}")
    standard_error = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "standard_error": standard_error,
    }


def _normalize_ordinal(score: int) -> float:
    if not 0 <= score <= 4:
        raise ValueError(f"ordinal score must be in [0, 4], got {score}")
    return score / 4.0


def build_item_record(item: InputItem, judge_call: JudgeCall) -> dict[str, Any]:
    """Join cached/deterministic evidence with the blind semantic verdict."""
    verdict = judge_call.verdict
    false_facts_rate = verdict.false_facts.rate()
    false_facts_quality = 1.0 - false_facts_rate
    metrics = {
        "hallucination": _normalize_ordinal(verdict.hallucination.quality_score),
        "false_facts": false_facts_quality,
        "coherence": _normalize_ordinal(verdict.coherence.score),
        "relevance_to_context": _normalize_ordinal(verdict.relevance_to_context.score),
        "roundtrip_cosine": quality_cosine(item.cosine_similarity),
        "llm_grounding": _normalize_ordinal(verdict.llm_grounding.quality_score),
        "information_gain": _normalize_ordinal(verdict.information_gain.score),
    }
    deterministic = degeneracy_stats(item.explanation, item.parse_ok)
    return {
        "schema_version": "nla_proxy_metrics.v1",
        "item_id": f"row_{item.source_row:08d}",
        "source_row": item.source_row,
        "input": {
            "content": item.content,
            "explanation": item.explanation,
            "cosine_similarity": item.cosine_similarity,
        },
        "metrics": metrics,
        "subrubrics": {
            "hallucination": {
                "plausibility": _normalize_ordinal(verdict.hallucination.plausibility_score),
                "verifiability": verdict.hallucination.verifiability,
                "calibration": verdict.hallucination.calibration,
            },
            "grounding": {
                "plausibility": _normalize_ordinal(verdict.llm_grounding.plausibility_score),
                "verifiability": verdict.llm_grounding.verifiability,
                "calibration": verdict.llm_grounding.calibration,
            },
        },
        "diagnostics": {
            **deterministic.to_dict(),
            "cosine_similarity_raw": item.cosine_similarity,
            "unit_vector_mse_equivalent": 2.0 * (1.0 - item.cosine_similarity),
            "false_facts_rate": false_facts_rate,
            "checkable_claim_count": verdict.false_facts.checkable_claim_count,
            "contradicted_claim_count": verdict.false_facts.contradicted_claim_count,
        },
        "judge": {
            "prompt_version": PROMPT_VERSION,
            "theme": verdict.theme,
            "ordinal_ratings": verdict.model_dump(mode="json"),
            "request": judge_call.request,
            "response": {
                "id": judge_call.response_id,
                "model": judge_call.response_model,
                "status": judge_call.response_status,
                "usage": judge_call.usage,
                "raw": judge_call.raw_response,
            },
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = list(records[0]["metrics"])
    metrics = {name: summarize_values([float(record["metrics"][name]) for record in records]) for name in metric_names}
    for metric_summary in metrics.values():
        mean = metric_summary["mean"]
        metric_summary["passes_0_5_gate"] = mean is not None and mean >= 0.5
        metric_summary["low_confidence"] = metric_summary["count"] < _MIN_CONFIDENT_SAMPLE_COUNT
    numeric_diagnostics = (
        "word_count",
        "max_trigram_repeat_fraction",
        "non_ascii_fraction",
        "cjk_fraction",
        "nonword_fraction",
        "distinct_word_ratio",
        "cosine_similarity_raw",
        "unit_vector_mse_equivalent",
        "false_facts_rate",
        "checkable_claim_count",
        "contradicted_claim_count",
    )
    diagnostics = {
        name: summarize_values([float(record["diagnostics"][name]) for record in records])
        for name in numeric_diagnostics
    }
    diagnostics["empty_rate"] = summarize_values([float(record["diagnostics"]["empty"]) for record in records])
    known_parse = [
        bool(record["diagnostics"]["parse_ok"]) for record in records if record["diagnostics"]["parse_ok"] is not None
    ]
    diagnostics["parse_ok_rate"] = {
        "known_count": len(known_parse),
        "mean": statistics.fmean(known_parse) if known_parse else None,
    }
    subrubrics = {
        "hallucination_plausibility": summarize_values(
            [float(record["subrubrics"]["hallucination"]["plausibility"]) for record in records]
        ),
        "grounding_plausibility": summarize_values(
            [float(record["subrubrics"]["grounding"]["plausibility"]) for record in records]
        ),
        "hallucination_verifiability": dict(
            sorted(Counter(record["subrubrics"]["hallucination"]["verifiability"] for record in records).items())
        ),
        "grounding_verifiability": dict(
            sorted(Counter(record["subrubrics"]["grounding"]["verifiability"] for record in records).items())
        ),
        "hallucination_calibration": dict(
            sorted(Counter(record["subrubrics"]["hallucination"]["calibration"] for record in records).items())
        ),
        "grounding_calibration": dict(
            sorted(Counter(record["subrubrics"]["grounding"]["calibration"] for record in records).items())
        ),
    }
    return {"metrics": metrics, "subrubrics": subrubrics, "diagnostics": diagnostics}


def run_pipeline(
    *,
    input_path: Path,
    output_jsonl: Path,
    summary_json: Path,
    settings: JudgeSettings,
    judge: Any | None = None,
    limit: int | None = None,
    offset: int = 0,
    content_column: str = "content",
    explanation_column: str = "explanation",
    cosine_column: str = "cosine_similarity",
    parse_ok_column: str = "explanation_parse_ok",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score cached rows concurrently, then write ordered JSONL and aggregate JSON."""
    started_at = _utc_now()
    parquet_path = Path(input_path)
    items = load_input_items(
        parquet_path,
        content_column=content_column,
        explanation_column=explanation_column,
        cosine_column=cosine_column,
        parse_ok_column=parse_ok_column,
        limit=limit,
        offset=offset,
    )
    if not items:
        raise ValueError("selected input row range is empty")
    active_judge = judge if judge is not None else OpenAIJudge(settings)
    LOGGER.info(
        "Scoring %d cached verbalizations with model=%s concurrency=%d",
        len(items),
        settings.judge_model,
        settings.concurrency,
    )
    with ThreadPoolExecutor(max_workers=settings.concurrency) as executor:
        judge_calls = list(executor.map(active_judge.score, items))
    records = [build_item_record(item, call) for item, call in zip(items, judge_calls, strict=True)]

    output_path = Path(output_jsonl)
    summary_path = Path(summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    aggregates = _aggregate_records(records)
    parquet_metadata = pq.ParquetFile(parquet_path).metadata
    summary: dict[str, Any] = {
        "schema_version": "nla_proxy_metrics_summary.v1",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "input": {
            "path": str(parquet_path.resolve()),
            "sha256": _sha256_file(parquet_path),
            "file_size_bytes": parquet_path.stat().st_size,
            "total_rows": parquet_metadata.num_rows,
            "selected_rows": len(items),
            "offset": offset,
            "limit": limit,
            "columns": {
                "content": content_column,
                "explanation": explanation_column,
                "cosine_similarity": cosine_column,
                "parse_ok": parse_ok_column,
            },
        },
        "configuration": settings.public_dict(),
        "prompt": {
            "version": PROMPT_VERSION,
            "sha256": _sha256_text(JUDGE_INSTRUCTIONS),
            "instructions": JUDGE_INSTRUCTIONS,
        },
        **aggregates,
        "redundancy_flags": [
            "roundtrip_cosine overlaps with FVE: unit-vector MSE = 2 * (1 - raw cosine)",
            "hallucination and llm_grounding are related but remain separately reported",
            "false_facts and hallucination overlap but use contradiction versus plausibility criteria",
        ],
        "outputs": {
            "items_jsonl": str(output_path.resolve()),
            "summary_json": str(summary_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s and %s", output_path, summary_path)
    return records, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-parquet",
        type=Path,
        required=True,
        help="cached output parquet from evaluate/roundtrip_from_text.py",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True, help="per-item scored JSONL")
    parser.add_argument("--summary-json", type=Path, required=True, help="aggregate/provenance JSON")
    parser.add_argument("--limit", type=int, default=None, help="maximum selected rows")
    parser.add_argument("--offset", type=int, default=0, help="zero-based first source row")
    parser.add_argument("--content-column", default="content")
    parser.add_argument("--explanation-column", default="explanation")
    parser.add_argument("--cosine-column", default="cosine_similarity")
    parser.add_argument("--parse-ok-column", default="explanation_parse_ok")
    parser.add_argument("--judge-model", default=None, help="overrides NLA_PROXY_JUDGE_MODEL")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default=None,
        help="overrides NLA_PROXY_REASONING_EFFORT",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="overrides NLA_PROXY_MAX_OUTPUT_TOKENS",
    )
    parser.add_argument("--concurrency", type=int, default=None, help="overrides NLA_PROXY_CONCURRENCY")
    parser.add_argument(
        "--context-chars",
        type=int,
        default=None,
        help="source-tail character cap; overrides NLA_PROXY_CONTEXT_CHARS",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    overrides = {
        field: value
        for field, value in (
            ("judge_model", args.judge_model),
            ("reasoning_effort", args.reasoning_effort),
            ("max_output_tokens", args.max_output_tokens),
            ("concurrency", args.concurrency),
            ("context_chars", args.context_chars),
        )
        if value is not None
    }
    settings = JudgeSettings(**overrides)
    run_pipeline(
        input_path=args.input_parquet,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        settings=settings,
        limit=args.limit,
        offset=args.offset,
        content_column=args.content_column,
        explanation_column=args.explanation_column,
        cosine_column=args.cosine_column,
        parse_ok_column=args.parse_ok_column,
    )


if __name__ == "__main__":
    main()
