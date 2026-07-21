# NLA — instructions for Claude / AI assistants

## Constraints

- **This is an open-source repo.** Only standard libs: `pathlib.Path`,
  `pyarrow`, `transformers`, `datasets`, `httpx`, `pyyaml`, `numpy`, `orjson`,
  `safetensors`, the public `anthropic` and `openai` SDKs, and whatever
  Miles/SGLang pull in. No private/internal dependencies. (`openai` added
  2026-07-21 for the `evaluate/proxy_metrics` judge — structured Responses API
  output via `client.responses.parse`.)
- **Miles is upstream, not ours.** Don't edit the installed `miles` package —
  extend via subclassing (`NLAFSDPActor`) and the `--*-path` function-pointer
  args. The upstream patches we depend on live as `.patch` files in
  `nla/miles_patches/` (`0001`/`0002` for `--custom-actor-cls-path` +
  `--force-use-critic`, `0003` for the `harmony` loss-mask branch (gpt-oss
  forced_final), + `UPSTREAM_PIN` version pin), applied to the installed
  package — not vendored here. They're documented in `docs/design.md` §2.
- Miles uses argparse; match that for CLIs in `nla/`.
- **Two training backends.** FSDP (`nla/train_actor.py`, primary) and Megatron
  (`nla/megatron/train_actor.py`). Both carry the same injection hook and
  `cp_size == 1` invariant; keep changes to the actor in sync across both.
- Storage and completion-provider backends are pluggable via import-path
  strings (`--storage-cls`, `--provider-cls`). The shipped implementations are
  `LocalStorage` and `AnthropicProvider`. Cloud storage / other LLM APIs are
  bring-your-own — don't hardcode bucket paths or vendor SDKs into `nla/`.

## Key invariants (do not break these)

- **Data-gen NEVER normalizes** — all parquets store raw vectors
  (`norm="none"`). `stage3_build` asserts input `norm == "none"`. Normalization
  happens at injection time (`injection_scale`) and at loss time (`mse_scale`),
  both read from the sidecar.
- **Stage-1 split is DOCUMENT-level** — partition by unique `doc_id`, all rows
  from the same doc go to the same bucket. Never split positions from one doc
  across `av_sft` / `ar_sft` / `rl`.
- **Stage-0 `_MIN_POSITION = 50`** — need enough left-context for the
  activation to be meaningful. Earlier positions decode to noise.
- **Critic extraction is suffix-anchored** — no scan, no marker token. The
  critic prompt template ends with `... <summary>`; training extracts at
  `tokens[-1]`. `critic_suffix_ids` in the sidecar is for sanity-checking only.
- **Per-doc keyed RNG** — same `(seed, doc_id)` → same sampled positions
  regardless of chunk boundaries, slice ordering, or process count. This is
  what makes multi-GPU stage-0 sharding bit-reproducible.
- **Injection hook scans for the token ID inside the hook** (`inputs[0]`), not
  from precomputed positions. Miles reorders samples twice before the forward
  pass; any precomputed index is wrong by construction.
- **`cp_size == 1` only.** Context-parallel splits each sample across ranks
  and breaks the neighbor check. NLA sequences are short; CP buys nothing.
- **Sidecar is the contract.** Token IDs, prompt templates, `injection_scale`,
  `mse_scale`, `d_model` — all loaded from `nla_meta.yaml` and asserted
  against the live tokenizer at startup. Never hardcode them.
- **Per-model constants live in `model_presets.py`.** `num_layers`, `d_model`,
  `default_layer` (the 2/3-depth rule), and `extractor_kwargs` (batch size,
  `device_map`) come from the `MODELS` dict via `resolve()` — one source of
  truth, set by `model: <key>` in a datagen yaml. Don't re-derive layer/d_model
  or scatter them across configs. Adding a model = a new `ModelPreset` entry
  (qwen7b / gemma12b / gemma27b / llama70b / gpt_oss_20b today; the last is the
  MoE/MXFP4 case — MXFP4 packing is on the expert MLPs only).
- **Multimodal unwrapping goes through `arch_adapters.py`.** Wrapped HF
  checkpoints (Gemma-3 `language_model`/`text_config`) get unwrapped to their
  text side there. Extend `_WRAPPER_MODEL_ATTRS`/`_WRAPPER_CONFIG_ATTRS` —
  don't duck-type new `getattr` fallbacks at callsites.

## Debugging

If injection silently fails the actor sees the literal CJK marker char and
free-associates Chinese. Grep generated text for CJK — that's the loudest
smoke test for the entire injection path. See `docs/inference.md`
§ "Debugging: injection-failure smell" for the cause checklist.

- **gpt-oss trains with `--attn-implementation eager` ONLY** (asserted in
  `NLAFSDPActor.__init__`). The sink-attention FA2 hub kernels are NON-CAUSAL
  under miles' packed convention (`attention_mask=None` + per-sample
  `position_ids`): tokens attend to their own future and SFT loss collapses
  to ~0 by copy-forward while learning nothing (caught by the Phase-2 smoke,
  2026-06-12). Diagnostic signature: train loss ≪ teacher-forced NLL of the
  saved checkpoint on the SAME rows in a single-sequence forward. Confirm by
  comparing packed vs standalone per-sample NLL — pack position 0 differing
  from standalone is impossible under causal attention.

## Where to look

- `docs/design.md` — architecture, Miles integration, the two upstream
  patches (§2), data transport, and the `nla/` package map (§6). Note §6's map
  predates `arch_adapters.py`, `model_presets.py`, and `nla/megatron/`.
- `docs/inference.md` — injection at inference time + debugging.
- `docs/setup.md` — installing Miles and applying the patches.
