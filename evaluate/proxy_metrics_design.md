# GPT-OSS-20B Actor Verbalization Proxy Metrics

## Overview

This evaluator scores held-out GPT-OSS-20B actor verbalizations before RL. Its input is the parquet written by `evaluate/roundtrip_from_text.py`; that file is the reusable cache and already contains `content`, `explanation`, and `cosine_similarity`. The evaluator does not regenerate activations or verbalizations and does not load the actor or critic checkpoints.

The judge is blind to checkpoint identity, experimental condition, and round-trip cosine. It sees only the source context and the actor verbalization. One structured judge call returns the semantic ratings, while deterministic string diagnostics and cached round-trip cosine are computed without an API call. Every quality metric is reported in a higher-is-better 0-1 direction for the `>=0.5` gate. Raw ordinal judgments, raw cosine, sub-rubrics, rationales, API response metadata, prompt version, and input provenance remain available for audit.

## Requirements

### Operational definitions

| Axis | Operational definition for an activation verbalization | Scorer | Reported value |
|---|---|---|---|
| Hallucination | Whether the text avoids **implausible fabrication** for an activation extracted at the final source token. Novel, surprising, or independently unverified content is not a hallucination when it is plausible for this activation/context. | Blind LLM judge. A semantic plausibility judgment cannot be replaced by lexical overlap. | `hallucination` quality in `[0,1]` (1 = no implausible fabrication), plus separately reported `plausibility`, `verifiability`, and non-gating `calibration` sub-rubrics. |
| False-facts rate | The share of specific checkable propositions contradicted by the supplied context or well-established facts. A merely unverified but plausible claim is not false. | Blind LLM judge because claim extraction and contradiction require semantic interpretation. | The judge returns checkable-claim and contradicted-claim counts. `false_facts_rate = contradicted / checkable` (0 when there are no checkable claims); `false_facts = 1 - false_facts_rate` is the positive `[0,1]` gate value. Both counts are surfaced in per-item diagnostics so `0/0` is distinguishable from a perfect score supported by checkable claims. |
| Coherence | Whether the verbalization forms an intelligible, internally consistent description rather than empty text, word salad, character soup, or looping. Truth and activation fit are scored elsewhere. | Blind LLM judge, supported by deterministic repetition, script, word-count, and lexical-diversity diagnostics. | `coherence` in `[0,1]`. |
| Relevance to context | Whether the verbalization identifies features specific to this source and especially its tail, where the activation was extracted, rather than emitting broadly applicable boilerplate. | Blind LLM judge because topical and next-token relevance are semantic. | `relevance_to_context` in `[0,1]`. |
| Faithfulness | Two non-blended components: (a) whether AV text lets the frozen AR reconstruct the original activation direction; and (b) whether the text is a plausible description of features this activation could encode, including novel latent properties such as eval-awareness. | (a) deterministic cached cosine; (b) blind LLM grounding judge with novelty-safe plausibility plus separate verifiability and calibration. | `roundtrip_cosine` in `[0,1]` for gating, raw `cosine_similarity` in `[-1,1]`, and separate `llm_grounding` in `[0,1]`. They are never averaged. |
| Information gain | Whether the text adds concrete, discriminating information that helps distinguish this activation/context from generic actor templates. Specific plausible latent content can score highly even when independently unverified. | Blind LLM judge because semantic specificity is not captured by word count or distinct-word ratio. | `information_gain` in `[0,1]`. |

Except for the count-derived false-facts rate, the judge emits integer ratings from 0 through 4, which the harness divides by 4. Ordinal ratings are more reproducible than unconstrained decimal probabilities while retaining the required midpoint.

### Novelty-safe grounding and hallucination

The project wants verbalizations to surface previously unknown but plausible information. Therefore the hallucination and LLM-grounding rubrics each contain four independent outputs:

1. **Quality**: the axis score used by the gate.
2. **Plausibility**: whether the content could reasonably be encoded in or inferred from this kind of activation and context.
3. **Verifiability**: one of `confirmed_by_context`, `plausible_but_unverified`, `not_assessable`, or `contradicted_by_context`.
4. **Calibration**: one of `appropriately_calibrated`, `overstated`, or `not_assessable`, based on whether the verbalization's own confidence, hedging, and caveats match the available evidence. This diagnostic is not folded into the gate score.

`plausible_but_unverified` must not lower quality by itself. Only contradiction or implausibility is negative. The judge prompt states this rule separately in both the hallucination and grounding rubric blocks. Verifiability is reported but never folded into either quality score.

[Truthful AI (Evans, Cotton-Barratt, et al., 2021, arXiv:2110.06674)](https://arxiv.org/abs/2110.06674) sharpens this policy with its negligent-falsehood test: penalization requires both an unacceptably high likelihood of falsehood and evidence from which the speaker could feasibly have recognized that risk. A sufficiently hedged or caveated plausible-but-unverified claim is therefore not penalized merely for lacking independent verification. Verifiability alone cannot record whether the actor asserted or hedged that claim appropriately, so `calibration` is retained as a distinct non-gating audit signal; overstatement alone does not establish contradiction or mechanically reduce quality.

### Redundancy flags for human review

- **Round-trip cosine overlaps with FVE and is not an independent semantic axis.** For unit-normalized vectors, round-trip MSE is `2 * (1 - cosine_similarity)`. The gate score clips the cached raw cosine to `[0,1]`, while preserving raw cosine so this identity remains auditable. Report it because it directly probes activation reconstruction, but do not count it as independent corroboration of FVE.
- **Hallucination and LLM grounding are closely related but not identical.** Hallucination asks whether the text adds implausible content; grounding asks whether the text positively fits features this activation could encode. A terse omission can be non-hallucinatory yet weakly grounded.
- **False facts and hallucination overlap.** False facts require identifiable contradiction; hallucination also covers implausible fabrication for which a strict truth value is unavailable. Neither axis penalizes plausible novelty.
- **Relevance and information gain overlap.** Generic on-topic text can be relevant but add little discriminating information.

The judge's plausibility decisions for `not_assessable` and `plausible_but_unverified` self-reports are currently uncalibrated against a baseline predictor. [Looking Inward (Binder et al., 2024, arXiv:2410.13787)](https://arxiv.org/abs/2410.13787) motivates a future held-out calibration study: compare this same verbalization-and-judge process with a baseline predictor on a subset whose claims can be independently verified, then use any demonstrated advantage only as limited evidence when interpreting unverifiable reports. This would not validate fully unverifiable claims, which the paper itself leaves unresolved.

### Inputs and configuration

Required parquet columns, configurable by CLI flags:

- `content`: the held-out source text passed through the base model.
- `explanation`: the cached actor verbalization.
- `cosine_similarity`: the cached AV-to-AR reconstruction cosine.

Optional `explanation_parse_ok` is used when a future round-trip cache preserves tag-parse status. Its absence is recorded as unknown rather than silently treated as a parse failure.

The CLI takes `--input-parquet`, `--output-jsonl`, `--summary-json`, `--limit`, `--offset`, and column-name overrides. Judge configuration is provided by `pydantic-settings` with the `NLA_PROXY_` prefix and may be overridden by non-secret CLI arguments:

- `NLA_PROXY_OPENAI_API_KEY` or standard `OPENAI_API_KEY` (required for live judging; the prefixed value takes precedence)
- `NLA_PROXY_JUDGE_MODEL` (default `gpt-5.5`)
- `NLA_PROXY_REASONING_EFFORT` (default `low`)
- `NLA_PROXY_MAX_OUTPUT_TOKENS` (default `3000`)
- `NLA_PROXY_CONCURRENCY` (default `4`)
- `NLA_PROXY_CONTEXT_CHARS` (default `12000`, retaining the source tail)

The API key is never serialized. Run metadata records all non-secret settings, UTC timestamps, source path and SHA-256, selected row range, prompt version and SHA-256, request parameters, model-returned identity, response ID, token usage, parsed ratings, and the raw API response.

Once the existing cache and API key are reachable on RunPod or Silico, the deliberately bounded review run is:

```bash
export OPENAI_API_KEY='...'
uv run --no-sync python -m evaluate.proxy_metrics \
  --input-parquet /workspace/data/gptoss20b_sft_roundtrip.parquet \
  --output-jsonl /workspace/results/gptoss20b_proxy_items.jsonl \
  --summary-json /workspace/results/gptoss20b_proxy_summary.json \
  --limit 5
```

The input parquet path may differ, but it must be the cached output of `evaluate/roundtrip_from_text.py` for the held-out SFT checkpoint and must contain the three required columns above. The environment must also have the repo's optional `openai` SDK installed for the live judge call.

## Architecture and data flow

The implementation is split at stable responsibility boundaries while preserving `python -m evaluate.proxy_metrics` as the public entrypoint:

- `evaluate/proxy_metrics_models.py`: Pydantic judge schemas, pydantic-settings configuration, and typed item/call records.
- `evaluate/proxy_metrics_judge.py`: the versioned novelty-safe prompt, blind input rendering, and structured OpenAI adapter.
- `evaluate/proxy_metrics.py`: parquet ingestion, deterministic diagnostics, metric normalization, aggregation, JSONL/provenance output, and CLI orchestration. It re-exports the public schemas and judge symbols for callers.

The data-flow boundaries are:

1. Strictly load and validate cached parquet rows.
2. Compute deterministic degeneracy diagnostics and the nonnegative gate form of cached cosine.
3. Render a blind, source-tail-aware prompt and issue structured OpenAI Responses API calls concurrently.
4. Write one auditable JSON object per item.
5. Write an aggregate JSON summary containing count, mean, standard error, the unchanged `mean >= 0.5` gate result, and a non-gating `low_confidence` flag for every reported quality metric. `low_confidence` is true when `n < 4`, making n=1-3 review runs explicit without pretending the threshold establishes adequate statistical power.

The harness fails loudly on missing columns, non-string text, non-finite/out-of-range cosine, absent parsed judge output, refusal, or incomplete API response. It does not convert judge failures into zero scores.

### Relationship to the R1 real-vs-random precedent

This v1 keeps the precedent's deterministic diagnostics and blind semantic judging, but does not report real-vs-random AUC. The reusable `roundtrip_from_text.py` cache contains held-out real-activation verbalizations and no matched random-vector verbalizations or condition column. Computing AUC without that control would require new GPU generation or invented labels, neither of which is allowed here. If a future cache adds matched random controls, discrimination should be added as a separate analysis over the same blind per-item scores rather than changing the six scoring rubrics.

## Acceptance Criteria

- All six requested axes are present; faithfulness reports cached round-trip cosine and LLM grounding separately, with no blended score.
- Hallucination and grounding each expose quality, plausibility, verifiability, and non-gating calibration; plausible-but-unverified novelty is not penalized by definition.
- Every gate-facing score is per-item and aggregated on `[0,1]`; aggregates contain count, mean, standard error, the unchanged point-estimate gate result, and `low_confidence = (n < 4)`.
- Deterministic diagnostics cover empty/word count, maximum trigram repetition, non-ASCII ratio, CJK ratio, non-word ratio, and distinct-word ratio.
- The judge is blind to cosine, checkpoint, and condition labels.
- JSONL and summary output contain enough non-secret request, response, prompt-version, and input-provenance data to audit the run without repeating it.
- Unit and CLI tests run without GPU, checkpoint, network, or API key by injecting a fake judge client.
- No live 3-5 item integration run is attempted in this environment. The documented command is ready to use once the cached parquet and API key are reachable on RunPod or Silico.
