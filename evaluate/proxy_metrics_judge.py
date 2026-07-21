"""Novelty-safe prompt and structured OpenAI judge adapter."""

from __future__ import annotations

from typing import Any

from evaluate.proxy_metrics_models import InputItem, JudgeCall, JudgeSettings, JudgeVerdict

PROMPT_VERSION = "gptoss_actor_proxy_v2"

JUDGE_INSTRUCTIONS = """You are a blind evaluator of activation verbalizations. A verbalization is
a model's attempt to describe information represented by one internal activation extracted at the
FINAL token of a supplied source context. You are not told the checkpoint, condition, or any
round-trip reconstruction score. Rate only the supplied source and verbalization.

Return all ratings on the integer 0-4 scale. Keep each rationale to one sentence.

CALIBRATION AND NEGLIGENT-FALSEHOOD RULE
- Treat the verbalization's own confidence, hedging, and caveats as evidence. For each novelty-sensitive
  rating, set calibration to appropriately_calibrated when the wording matches the available evidence,
  overstated when it claims more certainty than the evidence warrants, or not_assessable when there is
  no substantive claim to calibrate.
- Apply the Truthful AI negligent falsehood test: penalize a claim as false or implausible only when it
  is both unacceptably likely to be false and feasible for the speaker to have recognized that from the
  available evidence. A sufficiently hedged or caveated plausible-but-unverified claim is appropriately
  calibrated and must not be penalized merely because independent verification is unavailable.
- Calibration is a separate, non-gating diagnostic. Overstatement alone does not prove contradiction;
  only the negligent-falsehood conditions or other implausibility/contradiction can lower quality.

HALLUCINATION RUBRIC
- quality_score: 4 means no implausible fabrication; 0 means the text is dominated by content that
  is implausible or contradicted for this activation/context.
- plausibility_score: separately rate whether the claimed content could reasonably be encoded in or
  inferred from this kind of activation and context.
- verifiability: separately classify the content as confirmed_by_context,
  plausible_but_unverified, not_assessable, or contradicted_by_context.
- calibration: separately classify whether the verbalization's phrasing is appropriately calibrated,
  overstated, or not assessable.
- Novel or surprising content is valuable here. A claim may reveal a latent property such as
  evaluation awareness. If it is plausible but unverified, DO NOT lower hallucination quality merely
  because the source does not independently confirm it. Penalize implausibility or contradiction,
  not novelty.

FALSE-FACTS RUBRIC
- Count specific checkable propositions, then count how many are contradicted by the source or a
  well-established fact. Generic interpretations that make no checkable claim count as zero claims.
- Do not call a plausible but unverified proposition false. Lack of independent confirmation is not
  contradiction.

COHERENCE RUBRIC
- Rate intelligibility and internal consistency. 0 is empty, looping, word salad, or character soup;
  4 is clear, specific, internally consistent prose. Do not use truth or source fit for this rating.

RELEVANCE-TO-CONTEXT RUBRIC
- Rate whether the text identifies features specific to this source, especially its final portion
  near the activation position, rather than generic boilerplate.

GROUNDING RUBRIC
- quality_score: rate whether the verbalization is a plausible positive description of information
  this activation could encode. This is not a demand for lexical entailment by the source.
- plausibility_score: separately rate whether the content could reasonably live in this activation.
- verifiability: separately classify independent support as confirmed_by_context,
  plausible_but_unverified, not_assessable, or contradicted_by_context.
- calibration: separately classify whether the verbalization's phrasing is appropriately calibrated,
  overstated, or not assessable.
- Novel latent content is an intended success case. If it is plausible but unverified, DO NOT lower
  grounding quality merely because it was not already known or independently confirmed. Penalize
  implausibility or contradiction, not novelty.

INFORMATION-GAIN RUBRIC
- Rate whether the text contributes concrete, discriminating content beyond a generic actor
  template. Plausible novel latent content can have high information gain even when unverified.
"""


def render_judge_input(item: InputItem, *, context_chars: int) -> str:
    """Render the blind user input, retaining the activation-adjacent source tail."""
    if context_chars <= 0:
        raise ValueError("context_chars must be positive")
    source_tail = item.content[-context_chars:]
    return (
        "SOURCE CONTEXT (activation extracted at its final token):\n"
        f"<source>\n{source_tail}\n</source>\n\n"
        "ACTIVATION VERBALIZATION:\n"
        f"<verbalization>\n{item.explanation}\n</verbalization>"
    )


def _extract_parsed_verdict(response: Any) -> JudgeVerdict:
    for output_item in getattr(response, "output", []) or []:
        if getattr(output_item, "type", None) != "message":
            continue
        for content_item in getattr(output_item, "content", []) or []:
            refusal = getattr(content_item, "refusal", None)
            if refusal:
                raise RuntimeError(f"judge refused the request: {refusal}")
            parsed = getattr(content_item, "parsed", None)
            if parsed is not None:
                return parsed if isinstance(parsed, JudgeVerdict) else JudgeVerdict.model_validate(parsed)
    raise RuntimeError("judge response contained no parsed JudgeVerdict")


class OpenAIJudge:
    """Structured OpenAI Responses API adapter with an injectable test client."""

    def __init__(self, settings: JudgeSettings, *, client: Any | None = None) -> None:
        self.settings = settings
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        self.client: Any = client

    def score(self, item: InputItem) -> JudgeCall:
        user_input = render_judge_input(item, context_chars=self.settings.context_chars)
        response = self.client.responses.parse(
            model=self.settings.judge_model,
            instructions=JUDGE_INSTRUCTIONS,
            input=user_input,
            text_format=JudgeVerdict,
            reasoning={"effort": self.settings.reasoning_effort},
            max_output_tokens=self.settings.max_output_tokens,
            store=False,
        )
        status = str(getattr(response, "status", ""))
        if status != "completed":
            raise RuntimeError(f"judge response did not complete: status={status!r}")
        verdict = _extract_parsed_verdict(response)
        usage_obj = getattr(response, "usage", None)
        usage = usage_obj.model_dump(mode="json") if usage_obj is not None else {}
        raw_response = response.model_dump(mode="json")
        serialized_request: dict[str, Any] = {
            "model": self.settings.judge_model,
            "instructions": JUDGE_INSTRUCTIONS,
            "input": user_input,
            "text_format": JudgeVerdict.__name__,
            "reasoning": {"effort": self.settings.reasoning_effort},
            "max_output_tokens": self.settings.max_output_tokens,
            "store": False,
        }
        return JudgeCall(
            verdict=verdict,
            response_id=str(getattr(response, "id", "")),
            response_model=str(getattr(response, "model", self.settings.judge_model)),
            response_status=status,
            usage=usage,
            request=serialized_request,
            raw_response=raw_response,
        )
