"""NLACriticModel: truncated transformer + vector value head.

Architecture per docs/design.md §4:
  - First K transformer layers only (K = extraction layer, set via config override)
  - No final layernorm — raw residual stream goes to head
  - Linear(d_model, d_model) head, no bias
  - Forward returns .values at every position; extract at PM-token position for MSE

Layer truncation: set config.num_hidden_layers BEFORE from_pretrained so the
weight loader only reads K layers. Do NOT slice nn.ModuleList post-hoc — breaks
FSDP sharding assumptions.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from nla.arch_adapters import resolve_text_config, resolve_text_model


def apply_gptoss_packed_mask_patch() -> None:
    """Work around transformers 4.57.1 cross-sequence attention leakage on gpt-oss.

    GptOssModel.forward omits position_ids from its create_causal_mask /
    create_sliding_window_causal_mask kwargs, so masking_utils' packed-sequence
    detection never fires: with thd packing (attention_mask=None, position_ids
    reset at boundaries) later segments attend into earlier ones. Measured on
    the SFT critic: packed preds deviate up to 53% rel-L2 from single-sequence,
    growing with pack position. Fixed upstream in transformers 5.x.

    Fix: when input looks packed, pre-build the mask dict WITH position_ids
    (both mask creators accept it and compose packed_sequence_mask_function
    via and_masks); GptOssModel.forward accepts a prepared dict as
    attention_mask and skips its own (broken) mask creation.
    """
    try:
        from transformers.models.gpt_oss import modeling_gpt_oss as m
    except ImportError:
        return
    if getattr(m.GptOssModel.forward, "_nla_packed_mask_patch", False):
        return
    orig_forward = m.GptOssModel.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, cache_position=None, **kw):
        if (
            attention_mask is None
            and position_ids is not None
            and past_key_values is None
            and position_ids.shape[-1] > 1
            and (torch.diff(position_ids, dim=-1) != 1).any()
        ):
            ref = input_ids if input_ids is not None else inputs_embeds
            B, T, device = ref.shape[0], ref.shape[1], ref.device
            dtype = inputs_embeds.dtype if inputs_embeds is not None else self.embed_tokens.weight.dtype
            if cache_position is None:
                cache_position = torch.arange(T, device=device)
            # input_embeds is used only for batch/length/dtype/device — a
            # zero-width dummy avoids materializing real embeddings twice.
            dummy = torch.empty((B, T, 0), dtype=dtype, device=device)
            mask_kwargs = dict(
                config=self.config,
                input_embeds=dummy,
                attention_mask=None,
                cache_position=cache_position,
                past_key_values=None,
                position_ids=position_ids,
            )
            attention_mask = {
                "full_attention": m.create_causal_mask(**mask_kwargs),
                "sliding_attention": m.create_sliding_window_causal_mask(**mask_kwargs),
            }
        return orig_forward(
            self, input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, cache_position=cache_position, **kw,
        )

    forward._nla_packed_mask_patch = True
    m.GptOssModel.forward = forward


apply_gptoss_packed_mask_patch()


# Embedding weight key suffixes across architectures.
# Llama/Qwen/Mistral/Gemma: model.embed_tokens.weight
# GPT-2: transformer.wte.weight
# Falcon: transformer.word_embeddings.weight
_EMBED_KEY_SUFFIXES = ("embed_tokens.weight", "wte.weight", "word_embeddings.weight")


def mxfp4_dequantize_kwargs(
    pretrained_model_name_or_path: str, trust_remote_code: bool = True
) -> dict:
    """from_pretrained kwargs that make MXFP4 checkpoints (gpt-oss) trainable.

    Phase −1.0 finding (gpt-oss-20b, user-confirmed decision 2026-06-11):
    under a default load the MXFP4-packed expert MLPs come up as
    triton_kernels custom tensors — present in NEITHER named_parameters nor
    named_buffers — and the packed MoE forward has no autograd backward, so
    experts AND router receive zero gradient (only 1.8B of 20.9B params are
    optimizer-visible). The only path that trains the MoE is
    Mxfp4Config(dequantize=True): experts load as bf16 nn.Parameters
    (~42 GB weights; redo memory planning before 8-GPU runs).

    Returns {} for non-MXFP4 checkpoints — safe to call unconditionally at
    every TRAINING load site. Datagen extraction (stage0) deliberately does
    NOT use this: forward-only, packed weights are fine and 4× smaller.
    """
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code
    )
    qc = getattr(config, "quantization_config", None)
    method = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
    if method != "mxfp4":
        return {}
    from transformers import Mxfp4Config  # transformers>=4.55; import here so

    # non-MoE models never need the symbol.
    print(
        f"[NLA] {pretrained_model_name_or_path}: MXFP4 quantization_config "
        f"detected → loading with Mxfp4Config(dequantize=True) so the expert "
        f"MLPs are trainable bf16 parameters (Phase −1.0 decision)."
    )
    return {"quantization_config": Mxfp4Config(dequantize=True)}


@dataclass
class NLACriticOutput:
    values: torch.Tensor  # [B, T, d_model] — value-head output at every position
    backbone_last_hidden: torch.Tensor  # [B, T, d_model] — pre-value-head (for norm tracking)


def _truncate_config_layers(config, num_layers: int) -> None:
    """Set num_hidden_layers AND truncate per-layer arrays to match.

    transformers >=4.50 validates len(layer_types) == num_hidden_layers at
    config init (configuration_utils.py:layer_type_validation). Qwen2/Llama3
    configs carry per-layer arrays that must be sliced consistently.
    """
    config.num_hidden_layers = num_layers
    for attr in ("layer_types", "sliding_window_pattern", "no_rope_layers"):
        v = getattr(config, attr, None)
        if isinstance(v, (list, tuple)) and len(v) > num_layers:
            setattr(config, attr, type(v)(v[:num_layers]))


def _inner_transformer(backbone: PreTrainedModel) -> nn.Module:
    """Get the inner transformer module (the part with .layers + .norm).

    Qwen/Llama/Mistral/Gemma (post-resolve_text_model → CausalLM wrapper): backbone.model
    GPT-2/Falcon: backbone.transformer
    """
    if hasattr(backbone, "model"):
        return backbone.model
    if hasattr(backbone, "transformer"):
        return backbone.transformer
    raise AssertionError(
        f"{type(backbone).__name__} has neither .model nor .transformer — "
        f"add the attribute name here if supporting a new arch"
    )


class NLACriticModel(PreTrainedModel):
    """Wraps an HF causal LM backbone with layer truncation + vector value head.

    Delegates everything structural (save/load/fsdp-wrapping) to the backbone's
    PreTrainedModel machinery. Only the forward path + head are NLA-specific.
    """

    def __init__(self, config, backbone: PreTrainedModel):
        super().__init__(config)
        self.backbone = backbone
        self.value_head = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # FSDP's apply_fsdp2 reads this to decide which modules to wrap.
        # Instance attr (not class attr) — two NLACriticModels with different
        # backbones in one process would clobber each other on class attr.
        self._no_split_modules = backbone._no_split_modules

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, nla_num_layers: int | None = None, **kwargs):
        """Load an NLACriticModel from an HF checkpoint.

        Normal case: checkpoint was produced by prepare_critic_checkpoint.py or
        a previous NLA training run → config.json already has the truncated
        num_hidden_layers. Just load.

        Bootstrapping case (prepare_critic_checkpoint.py only): pass
        nla_num_layers = the datagen extraction layer_index. Truncation keeps
        blocks 0..layer_index INCLUSIVE — we need the output OF block K, which
        means we need block K to exist, so num_hidden_layers = K+1.

        Indexing convention (matches datagen/extractors.py):
          - datagen `layer_index=K` hooks `model.model.layers[K]` →
            captures its output (= HF's `hidden_states[K+1]`, post-residual-add,
            the stream entering block K+1).
          - critic `last_hidden_state` (with final-LN → Identity) is the output
            of the LAST block.
          - So last block must be block K → num_hidden_layers = K+1.
        """
        # Miles' fsdp_utils/actor.py:96 passes trust_remote_code + attn_implementation
        # via kwargs. Default to True if caller doesn't specify.
        kwargs.setdefault("trust_remote_code", True)
        # gpt-oss: critic is trained too — its truncated checkpoint keeps the
        # MXFP4 quantization_config (prepare_critic_checkpoint copies
        # config.json), so the experts must dequantize here as well. Explicit
        # caller-passed quantization_config wins over the detection.
        for k, v in mxfp4_dequantize_kwargs(
            pretrained_model_name_or_path, kwargs["trust_remote_code"]
        ).items():
            kwargs.setdefault(k, v)
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=kwargs["trust_remote_code"]
        )
        # For multimodal wrappers (Gemma-3), truncate the nested text config —
        # the wrapper config references it by identity so the truncation is
        # visible to from_pretrained. For plain text models, text_config IS
        # config (see arch_adapters).
        text_config = resolve_text_config(config)
        if nla_num_layers is not None:
            needed = nla_num_layers + 1
            assert needed <= text_config.num_hidden_layers, (
                f"nla_num_layers={nla_num_layers} needs blocks 0..{nla_num_layers} "
                f"inclusive (num_hidden_layers={needed}), but base model has "
                f"only {text_config.num_hidden_layers}."
            )
            _truncate_config_layers(text_config, needed)

        backbone = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            **kwargs,
        )
        # Discard the vision tower / projection for multimodal wrappers — keep
        # only the text transformer. No-op for plain causal LMs.
        backbone = resolve_text_model(backbone)
        # Strip lm_head — critic never produces logits. Frees memory and
        # prevents FSDP from sharding a weight matrix we never use.
        if hasattr(backbone, "lm_head"):
            backbone.lm_head = nn.Identity()

        # Strip final layernorm — arch doc §4: raw residual-stream → value head.
        inner = _inner_transformer(backbone)
        for attr in ("norm", "final_layernorm", "ln_f"):
            if hasattr(inner, attr):
                setattr(inner, attr, nn.Identity())
                break
        else:
            raise AssertionError(
                f"could not find final layernorm on {type(inner).__name__} — "
                f"add the attribute name to the list above"
            )

        model = cls(text_config, backbone)

        # If an HF-format critic checkpoint exists (from save_pretrained), load
        # the value head from it. Fresh init otherwise. Skip under
        # init_empty_weights (rank≠0 in fsdp_utils/actor.py:202 when
        # tie_word_embeddings=False): nn.Linear was registered on meta;
        # load_state_dict(cpu_tensors) would crash. Rank 0 loads,
        # _fsdp2_load_full_state_dict broadcasts.
        head_path = Path(pretrained_model_name_or_path) / "value_head.safetensors"
        if head_path.exists() and not model.value_head.weight.is_meta:
            model.value_head.load_state_dict(load_file(str(head_path)))

        # nn.Linear in __init__ is CPU/fp32; load_state_dict upcasts the bf16
        # file tensors into that. Backbone got device+dtype via kwargs (incl.
        # device_map="auto" → accelerate sharding), but value_head is outside
        # the safetensors index so accelerate doesn't see it. Align to the last
        # layer's placement so forward()'s `h` → value_head doesn't bounce
        # device/dtype. Skip on meta (FSDP rank≠0 — broadcast handles it).
        if not model.value_head.weight.is_meta:
            last = next(inner.layers[-1].parameters())
            model.value_head.to(device=last.device, dtype=last.dtype)

        return model

    def forward(self, input_ids=None, position_ids=None, attention_mask=None, **kwargs):
        out = _inner_transformer(self.backbone)(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        # out.last_hidden_state: [B, T, d_model] — post last-kept transformer block,
        # pre-final-LN (norm was replaced with Identity in from_pretrained).
        h = out.last_hidden_state
        return NLACriticOutput(values=self.value_head(h), backbone_last_hidden=h)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def save_pretrained(self, save_directory, state_dict=None, **kwargs):
        if state_dict is None:
            state_dict = self.state_dict()
        backbone_sd = {k.removeprefix("backbone."): v for k, v in state_dict.items() if k.startswith("backbone.")}
        head_sd = {k.removeprefix("value_head."): v for k, v in state_dict.items() if k.startswith("value_head.")}

        self.backbone.save_pretrained(save_directory, state_dict=backbone_sd, **kwargs)
        # Guard against silent value_head corruption (seen 2026-07: NaN-riddled
        # safetensors on disk while the DCP ckpt of the same weights was clean).
        # Force plain contiguous CPU tensors, refuse to write non-finite
        # weights, and verify the bytes that actually landed on disk.
        head_sd = {k: v.detach().to("cpu").contiguous() for k, v in head_sd.items()}
        for k, v in head_sd.items():
            bad = (~torch.isfinite(v.float())).sum().item()
            if bad:
                raise RuntimeError(
                    f"value_head export: {bad} non-finite elements in '{k}' BEFORE write "
                    f"(shape={tuple(v.shape)}, dtype={v.dtype}) — gathered state dict is corrupt"
                )
        head_path = Path(save_directory) / "value_head.safetensors"
        save_file(head_sd, str(head_path))
        reread = load_file(str(head_path))
        for k, v in head_sd.items():
            bad = (~torch.isfinite(reread[k].float())).sum().item()
            if bad or not torch.equal(reread[k], v):
                raise RuntimeError(
                    f"value_head export: readback mismatch for '{k}' "
                    f"(non_finite={bad}) — save_file wrote corrupt bytes to {head_path}"
                )
        # config.json written by backbone.save_pretrained includes the truncated num_hidden_layers

    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()


# Path the actor dumps a fresh embedding to after each SGLang weight sync.
# nla_generate reloads from here so rollout embeddings track the trained
# model (not the stale initial HF checkpoint). Default goes under args.save
# (disk); set NLA_EMBED_DUMP_DIR=/dev/shm/nla to use tmpfs — the 1GB reload
# drops from ~1.8s to ~0.2s per rollout.
ROLLOUT_EMBED_DUMP = "_nla_rollout_embed.pt"


def embed_dump_path(save_dir: str | None) -> Path | None:
    """Resolve the embedding dump path. NLA_EMBED_DUMP_DIR overrides save_dir."""
    override = os.environ.get("NLA_EMBED_DUMP_DIR")
    base = override or save_dir
    if base is None:
        return None
    return Path(base) / ROLLOUT_EMBED_DUMP


def _find_embed_key(keys: list[str], where: str) -> str:
    matches = [k for k in keys if k.endswith(_EMBED_KEY_SUFFIXES)]
    assert len(matches) == 1, (
        f"expected exactly one input-embedding key in {where} "
        f"(suffixes {_EMBED_KEY_SUFFIXES}), got {matches!r}"
    )
    return matches[0]


def load_embedding_only(hf_checkpoint: str, dtype: torch.dtype = torch.float32) -> nn.Embedding:
    """Load ONLY the input embedding layer from an HF checkpoint's safetensors.

    Returns a plain `nn.Embedding` wrapping the weight tensor. This is the RAW
    lookup — if the model's embedding forward applies a scale (Gemma ×√d, T5),
    the CALLER must multiply. See `arch_adapters.resolve_embed_scale()` and
    how `nla_generate.py` uses `_EMBED_SCALE` explicitly.

    Avoids materializing the full model — just reads the one weight tensor
    via safe_open's lazy loading. Handles HF Hub names via snapshot_download.
    """
    root = Path(hf_checkpoint)
    if not root.exists():
        root = Path(snapshot_download(hf_checkpoint))

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        key = _find_embed_key(list(weight_map), str(index_path))
        shard = root / weight_map[key]
    else:
        shard = root / "model.safetensors"
        assert shard.exists(), (
            f"no model.safetensors or .index.json at {root!r}"
        )
        with safe_open(str(shard), framework="pt") as f:
            key = _find_embed_key(list(f.keys()), str(shard))

    with safe_open(str(shard), framework="pt") as f:
        weight = f.get_tensor(key).to(dtype)

    vocab, d_model = weight.shape
    embed = nn.Embedding(vocab, d_model, _weight=weight)
    embed.requires_grad_(False)
    embed.eval()
    return embed
