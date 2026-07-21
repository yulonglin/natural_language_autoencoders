from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from evaluate import proxy_metrics


def _write_roundtrip(path: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "content": ["alpha source", "beta source"],
                "explanation": ["specific alpha", "specific beta"],
                "cosine_similarity": [0.75, -0.2],
            }
        ),
        path,
    )


def test_load_input_items_reuses_roundtrip_columns(tmp_path: Path) -> None:
    path = tmp_path / "roundtrip.parquet"
    _write_roundtrip(path)

    items = proxy_metrics.load_input_items(path, limit=1, offset=1)

    assert len(items) == 1
    assert items[0].source_row == 1
    assert items[0].content == "beta source"
    assert items[0].explanation == "specific beta"
    assert items[0].cosine_similarity == pytest.approx(-0.2)
    assert items[0].parse_ok is None


def test_load_input_items_fails_on_missing_required_column(tmp_path: Path) -> None:
    path = tmp_path / "missing.parquet"
    pq.write_table(pa.table({"content": ["alpha"], "explanation": ["beta"]}), path)

    with pytest.raises(ValueError, match="cosine_similarity"):
        proxy_metrics.load_input_items(path)


def test_degeneracy_stats_and_cosine_gate_direction() -> None:
    stats = proxy_metrics.degeneracy_stats("loop loop loop loop 漢!", parse_ok=None)

    assert stats.word_count == 5
    assert stats.parse_ok is None
    assert stats.max_trigram_repeat_fraction == pytest.approx(2 / 3)
    assert stats.cjk_fraction > 0
    assert stats.non_ascii_fraction > 0
    assert stats.nonword_fraction > 0
    assert stats.distinct_word_ratio == pytest.approx(2 / 5)
    assert proxy_metrics.quality_cosine(-0.2) == 0.0
    assert proxy_metrics.quality_cosine(0.75) == 0.75
    assert proxy_metrics.quality_cosine(1.0) == 1.0


def test_quality_cosine_rejects_non_finite_or_out_of_range() -> None:
    for value in (float("nan"), float("inf"), -1.01, 1.01):
        with pytest.raises(ValueError):
            proxy_metrics.quality_cosine(value)


def test_summarize_values_includes_count_mean_and_standard_error() -> None:
    summary = proxy_metrics.summarize_values([0.25, 0.75])

    assert summary == {
        "count": 2,
        "mean": 0.5,
        "standard_error": pytest.approx(0.25),
    }


def _verdict() -> proxy_metrics.JudgeVerdict:
    return proxy_metrics.JudgeVerdict(
        hallucination=proxy_metrics.NoveltySensitiveRating(
            quality_score=4,
            plausibility_score=4,
            verifiability="plausible_but_unverified",
            calibration="appropriately_calibrated",
            rationale="Novel but plausible.",
        ),
        false_facts=proxy_metrics.FalseFactsRating(
            checkable_claim_count=4,
            contradicted_claim_count=1,
            rationale="One of four checkable propositions is contradicted.",
        ),
        coherence=proxy_metrics.AxisRating(score=4, rationale="Clear prose."),
        relevance_to_context=proxy_metrics.AxisRating(score=3, rationale="Specific to the source."),
        llm_grounding=proxy_metrics.NoveltySensitiveRating(
            quality_score=3,
            plausibility_score=4,
            verifiability="plausible_but_unverified",
            calibration="appropriately_calibrated",
            rationale="A plausible latent property.",
        ),
        information_gain=proxy_metrics.AxisRating(score=2, rationale="Some discriminating detail."),
        theme="evaluation awareness",
    )


def _input_item() -> proxy_metrics.InputItem:
    return proxy_metrics.InputItem(
        source_row=7,
        content="The evaluator presents a held-out question.",
        explanation="The model may recognize an evaluation setting.",
        cosine_similarity=0.8,
        parse_ok=True,
    )


def test_judge_schema_bounds_ordinal_scores() -> None:
    with pytest.raises(ValidationError):
        proxy_metrics.AxisRating(score=5, rationale="out of range")


def test_false_facts_rejects_more_contradictions_than_claims() -> None:
    with pytest.raises(ValidationError):
        proxy_metrics.FalseFactsRating(
            checkable_claim_count=1,
            contradicted_claim_count=2,
            rationale="impossible counts",
        )


def test_prompt_protects_plausible_unverified_novelty_and_is_blind() -> None:
    prompt = proxy_metrics.render_judge_input(_input_item(), context_chars=100)
    instructions = proxy_metrics.JUDGE_INSTRUCTIONS.lower()

    assert proxy_metrics.PROMPT_VERSION == "gptoss_actor_proxy_v2"
    assert "hallucination rubric" in instructions
    assert "grounding rubric" in instructions
    assert instructions.count("plausible but unverified") >= 2
    assert "do not lower" in instructions
    assert "negligent falsehood" in instructions
    assert "appropriately calibrated" in instructions
    assert "cosine" not in prompt.lower()
    assert "0.8" not in prompt
    assert "gpt-oss" not in prompt.lower()


def test_render_judge_input_retains_source_tail() -> None:
    item = proxy_metrics.InputItem(
        source_row=0,
        content="discard-this-prefix|TAIL",
        explanation="description",
        cosine_similarity=0.5,
        parse_ok=None,
    )

    prompt = proxy_metrics.render_judge_input(item, context_chars=4)

    assert "TAIL" in prompt
    assert "discard-this-prefix" not in prompt


class _FakeResponse:
    def __init__(
        self,
        verdict: object,
        *,
        status: str = "completed",
        refusal: str | None = None,
    ) -> None:
        self.id = "resp_test"
        self.model = "gpt-5.5-test"
        self.status = status
        self.usage = SimpleNamespace(model_dump=lambda mode: {"input_tokens": 10, "output_tokens": 20})
        content = SimpleNamespace(type="output_text", parsed=verdict, refusal=refusal)
        self.output = [SimpleNamespace(type="message", content=[content])]

    def model_dump(self, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {"id": self.id, "model": self.model, "status": self.status}


class _FakeResponses:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> _FakeResponse:
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.responses = _FakeResponses(response)


def _settings() -> proxy_metrics.JudgeSettings:
    return proxy_metrics.JudgeSettings(
        openai_api_key="test-key",
        judge_model="gpt-5.5-test",
        reasoning_effort="low",
        max_output_tokens=1234,
        concurrency=2,
        context_chars=100,
    )


def test_openai_judge_uses_structured_responses_contract() -> None:
    fake_client = _FakeClient(_FakeResponse(_verdict()))
    judge = proxy_metrics.OpenAIJudge(_settings(), client=fake_client)

    call = judge.score(_input_item())

    assert call.verdict == _verdict()
    assert call.response_id == "resp_test"
    request = fake_client.responses.calls[0]
    assert request["model"] == "gpt-5.5-test"
    assert request["text_format"] is proxy_metrics.JudgeVerdict
    assert request["reasoning"] == {"effort": "low"}
    assert request["max_output_tokens"] == 1234
    assert request["store"] is False


def test_openai_judge_fails_on_refusal() -> None:
    fake_client = _FakeClient(_FakeResponse(None, refusal="policy refusal"))
    judge = proxy_metrics.OpenAIJudge(_settings(), client=fake_client)

    with pytest.raises(RuntimeError, match="judge refused the request: policy refusal"):
        judge.score(_input_item())


def test_openai_judge_fails_when_parsed_output_is_absent() -> None:
    fake_client = _FakeClient(_FakeResponse(None))
    judge = proxy_metrics.OpenAIJudge(_settings(), client=fake_client)

    with pytest.raises(RuntimeError, match="no parsed JudgeVerdict"):
        judge.score(_input_item())


def test_openai_judge_fails_on_incomplete_response() -> None:
    fake_client = _FakeClient(_FakeResponse(object(), status="incomplete"))
    judge = proxy_metrics.OpenAIJudge(_settings(), client=fake_client)

    with pytest.raises(RuntimeError, match="did not complete.*incomplete"):
        judge.score(_input_item())


def test_item_record_keeps_faithfulness_components_separate() -> None:
    call = proxy_metrics.JudgeCall(
        verdict=_verdict(),
        response_id="resp_test",
        response_model="gpt-5.5-test",
        response_status="completed",
        usage={"input_tokens": 10, "output_tokens": 20},
        request={"model": "gpt-5.5-test"},
        raw_response={"id": "resp_test"},
    )

    record = proxy_metrics.build_item_record(_input_item(), call)

    assert record["metrics"]["roundtrip_cosine"] == 0.8
    assert record["metrics"]["llm_grounding"] == 0.75
    assert "faithfulness" not in record["metrics"]
    assert record["diagnostics"]["false_facts_rate"] == 0.25
    assert record["diagnostics"]["checkable_claim_count"] == 4
    assert record["diagnostics"]["contradicted_claim_count"] == 1
    assert record["subrubrics"]["grounding"]["plausibility"] == 1.0
    assert record["subrubrics"]["grounding"]["verifiability"] == "plausible_but_unverified"
    assert record["subrubrics"]["grounding"]["calibration"] == "appropriately_calibrated"


def test_zero_checkable_claims_remain_perfect_but_are_auditable() -> None:
    verdict = _verdict().model_copy(
        update={
            "false_facts": proxy_metrics.FalseFactsRating(
                checkable_claim_count=0,
                contradicted_claim_count=0,
                rationale="No checkable propositions.",
            )
        }
    )
    call = proxy_metrics.JudgeCall(
        verdict=verdict,
        response_id="resp_test",
        response_model="gpt-5.5-test",
        response_status="completed",
        usage={},
        request={},
        raw_response={},
    )

    record = proxy_metrics.build_item_record(_input_item(), call)

    assert record["metrics"]["false_facts"] == 1.0
    assert record["diagnostics"]["false_facts_rate"] == 0.0
    assert record["diagnostics"]["checkable_claim_count"] == 0
    assert record["diagnostics"]["contradicted_claim_count"] == 0


def test_aggregate_gate_flags_counts_below_four_as_low_confidence() -> None:
    call = proxy_metrics.JudgeCall(
        verdict=_verdict(),
        response_id="resp_test",
        response_model="gpt-5.5-test",
        response_status="completed",
        usage={},
        request={},
        raw_response={},
    )
    record = proxy_metrics.build_item_record(_input_item(), call)

    low_n = proxy_metrics._aggregate_records([record] * 3)
    adequate_n = proxy_metrics._aggregate_records([record] * 4)

    assert low_n["metrics"]["coherence"]["passes_0_5_gate"] is True
    assert low_n["metrics"]["coherence"]["low_confidence"] is True
    assert adequate_n["metrics"]["coherence"]["low_confidence"] is False


class _FakeJudge:
    def score(self, item: proxy_metrics.InputItem) -> proxy_metrics.JudgeCall:
        return proxy_metrics.JudgeCall(
            verdict=_verdict(),
            response_id=f"resp_{item.source_row}",
            response_model="gpt-5.5-test",
            response_status="completed",
            usage={"input_tokens": 10, "output_tokens": 20},
            request={"model": "gpt-5.5-test", "input": f"row {item.source_row}"},
            raw_response={"id": f"resp_{item.source_row}"},
        )


def test_run_pipeline_writes_jsonl_and_complete_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "roundtrip.parquet"
    output_jsonl = tmp_path / "out" / "items.jsonl"
    summary_json = tmp_path / "out" / "summary.json"
    _write_roundtrip(input_path)

    records, summary = proxy_metrics.run_pipeline(
        input_path=input_path,
        output_jsonl=output_jsonl,
        summary_json=summary_json,
        settings=_settings(),
        judge=_FakeJudge(),
        limit=2,
    )

    assert len(records) == 2
    assert [record["source_row"] for record in records] == [0, 1]
    assert set(summary["metrics"]) == {
        "hallucination",
        "false_facts",
        "coherence",
        "relevance_to_context",
        "roundtrip_cosine",
        "llm_grounding",
        "information_gain",
    }
    assert summary["metrics"]["coherence"]["count"] == 2
    assert summary["metrics"]["coherence"]["passes_0_5_gate"] is True
    assert summary["metrics"]["coherence"]["low_confidence"] is True
    assert summary["input"]["sha256"]
    assert summary["prompt"]["sha256"]
    assert summary["configuration"]["judge_model"] == "gpt-5.5-test"
    assert "openai_api_key" not in summary["configuration"]
    assert output_jsonl.exists()
    assert summary_json.exists()
    assert len(output_jsonl.read_text().splitlines()) == 2
    assert json.loads(summary_json.read_text())["metrics"] == summary["metrics"]
    assert "test-key" not in output_jsonl.read_text()
    assert "test-key" not in summary_json.read_text()


def test_judge_settings_accepts_standard_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NLA_PROXY_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "standard-key")

    settings = proxy_metrics.JudgeSettings()

    assert settings.openai_api_key.get_secret_value() == "standard-key"


def test_prefixed_api_key_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NLA_PROXY_OPENAI_API_KEY", "prefixed-key")
    monkeypatch.setenv("OPENAI_API_KEY", "standard-key")

    settings = proxy_metrics.JudgeSettings()

    assert settings.openai_api_key.get_secret_value() == "prefixed-key"


def test_parse_args_requires_explicit_artifact_paths() -> None:
    args = proxy_metrics.parse_args(
        [
            "--input-parquet",
            "roundtrip.parquet",
            "--output-jsonl",
            "items.jsonl",
            "--summary-json",
            "summary.json",
            "--limit",
            "3",
        ]
    )

    assert args.input_parquet == Path("roundtrip.parquet")
    assert args.limit == 3
