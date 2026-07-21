"""Typed records and settings for actor-verbalization proxy metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Verifiability = Literal[
    "confirmed_by_context",
    "plausible_but_unverified",
    "not_assessable",
    "contradicted_by_context",
]
Calibration = Literal[
    "appropriately_calibrated",
    "overstated",
    "not_assessable",
]


class AxisRating(BaseModel):
    """One ordinary 0-4 semantic rating."""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=4)
    rationale: str = Field(min_length=1, max_length=240)


class NoveltySensitiveRating(BaseModel):
    """A quality rating that keeps plausibility, verifiability, and calibration separate."""

    model_config = ConfigDict(extra="forbid")

    quality_score: int = Field(ge=0, le=4)
    plausibility_score: int = Field(ge=0, le=4)
    verifiability: Verifiability
    calibration: Calibration
    rationale: str = Field(min_length=1, max_length=240)


class FalseFactsRating(BaseModel):
    """Claim counts used to compute an actual contradicted-claim rate."""

    model_config = ConfigDict(extra="forbid")

    checkable_claim_count: int = Field(ge=0)
    contradicted_claim_count: int = Field(ge=0)
    rationale: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def contradictions_cannot_exceed_claims(self) -> FalseFactsRating:
        if self.contradicted_claim_count > self.checkable_claim_count:
            raise ValueError("contradicted_claim_count cannot exceed checkable_claim_count")
        return self

    def rate(self) -> float:
        if self.checkable_claim_count == 0:
            return 0.0
        return self.contradicted_claim_count / self.checkable_claim_count


class JudgeVerdict(BaseModel):
    """Structured blind-judge response for the six requested semantic axes."""

    model_config = ConfigDict(extra="forbid")

    hallucination: NoveltySensitiveRating
    false_facts: FalseFactsRating
    coherence: AxisRating
    relevance_to_context: AxisRating
    llm_grounding: NoveltySensitiveRating
    information_gain: AxisRating
    theme: str = Field(min_length=1, max_length=80)


class JudgeSettings(BaseSettings):
    """Live judge configuration loaded from NLA_PROXY_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="NLA_PROXY_", extra="ignore", populate_by_name=True)

    openai_api_key: SecretStr = Field(
        validation_alias=AliasChoices(
            "NLA_PROXY_OPENAI_API_KEY",
            "OPENAI_API_KEY",
        )
    )
    judge_model: str = "gpt-5.5"
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "low"
    max_output_tokens: int = Field(default=3000, gt=0)
    concurrency: int = Field(default=4, gt=0)
    context_chars: int = Field(default=12000, gt=0)

    def public_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude={"openai_api_key"}, mode="json")


@dataclass(frozen=True)
class JudgeCall:
    """Parsed verdict plus request/response provenance for one live judge call."""

    verdict: JudgeVerdict
    response_id: str
    response_model: str
    response_status: str
    usage: dict[str, Any]
    request: dict[str, Any]
    raw_response: dict[str, Any]


@dataclass(frozen=True)
class InputItem:
    """One row from the cached round-trip parquet."""

    source_row: int
    content: str
    explanation: str
    cosine_similarity: float
    parse_ok: bool | None


@dataclass(frozen=True)
class DegeneracyStats:
    """Deterministic string diagnostics for one verbalization."""

    parse_ok: bool | None
    empty: bool
    word_count: int
    max_trigram_repeat_fraction: float
    non_ascii_fraction: float
    cjk_fraction: float
    nonword_fraction: float
    distinct_word_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
