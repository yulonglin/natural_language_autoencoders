"""Patch a reasoning model's chat template to drop the force-inserted <think> block.

DeepSeek-R1-Distill models ship a chat template whose generation prompt ends
with `<｜Assistant｜><think>\\n` — every assistant turn is forced to open a
think block. NLA actors never think (SFT targets are plain <explanation>…
payloads; RL has a 150-token response budget), so the forced opener would
(a) pollute teacher-forced SFT targets with an unmatched <think> and
(b) waste RL response tokens.

This script materializes a local copy of the checkpoint with the forced
opener removed from the chat template, leaving everything else byte-identical.
Point INSTRUCT_MODEL / --hf-checkpoint at the output directory.

    python -m nla.scripts.patch_chat_template \\
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
        --output /vol/models/r1distill-1.5b-nothink
"""

import argparse
from pathlib import Path

# Forced-opener spellings seen in the wild. The Jinja source stores the literal
# either with a real newline or an escaped one depending on serialization.
_FORCED_OPENERS = [
    "<｜Assistant｜><think>\\n",
    "<｜Assistant｜><think>\n",
    "<｜Assistant｜><think>",
]
_REPLACEMENT = "<｜Assistant｜>"


def strip_forced_think(template: str) -> tuple[str, bool]:
    """Remove a forced `<think>` opener after the assistant tag. Pure string
    transform — unit-testable without downloading a tokenizer."""
    for opener in _FORCED_OPENERS:
        if opener in template:
            return template.replace(opener, _REPLACEMENT), True
    return template, False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model name or local checkpoint path")
    p.add_argument("--output", required=True, help="output dir for the patched checkpoint copy")
    p.add_argument("--allow-noop", action="store_true",
                   help="exit 0 even if no forced <think> opener was found (e.g. Qwen3, "
                        "whose template gates thinking on enable_thinking instead)")
    args = p.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    out = Path(args.output)
    if Path(args.model).exists():
        import shutil
        shutil.copytree(args.model, out, dirs_exist_ok=True)
    else:
        snapshot_download(args.model, local_dir=str(out))

    tok = AutoTokenizer.from_pretrained(str(out))
    template = tok.chat_template or ""
    patched, changed = strip_forced_think(template)
    if not changed:
        msg = f"no forced <think> opener found in {args.model}'s chat template"
        if not args.allow_noop:
            raise SystemExit(f"{msg} — pass --allow-noop if this is expected. "
                             f"Template head: {template[:300]!r}")
        print(f"{msg} (noop allowed)")
    else:
        tok.chat_template = patched
        tok.save_pretrained(str(out))
        print(f"stripped forced <think> opener; patched checkpoint → {out}")

    # Loud verification: render a one-turn prompt and check no <think> leaks in.
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "ping"}], tokenize=False, add_generation_prompt=True
    )
    assert "<think>" not in rendered, (
        f"patched template still force-inserts <think>: {rendered!r}"
    )
    print(f"verified: generation prompt ends {rendered[-40:]!r}")


if __name__ == "__main__":
    main()
