"""Tests for the zero-valid-rollouts skip path in nla/train_actor.py.

These tests run without a GPU — dist.all_reduce and torch.cuda.current_device
are monkeypatched so the collective logic can run on a single CPU process.
The Miles stack is NOT needed; only pure nla/ helpers are exercised.
"""

import pytest
import torch
import torch.distributed as dist

import nla.train_actor as ta
from nla.schema import MM_CRITIC_TOKENS_KEY
from nla.models import MM_ACTIVATION_KEY


@pytest.fixture(autouse=True)
def _reset_globals(monkeypatch):
    """Clean ping-flag and counter before each test; stub GPU collectives."""
    monkeypatch.setattr(ta, "_EMPTY_ROLLOUT_PINGED", False, raising=False)
    monkeypatch.setattr(ta, "_CONSECUTIVE_EMPTY_STEPS", 0, raising=False)
    # Stub collective ops: all_reduce is a no-op (tensor keeps its local value),
    # get_world_size returns 1.  current_device returns a CPU device so torch.tensor
    # calls in _truncate_to_cross_rank_min don't error.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(dist, "all_reduce", lambda *a, **kw: None)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 1)


def _mm(with_critic: bool):
    """Build a minimal multimodal_train_inputs entry."""
    d = {MM_ACTIVATION_KEY: torch.zeros(1, 8)}
    if with_critic:
        d[MM_CRITIC_TOKENS_KEY] = torch.tensor([1, 2, 3])
    return d


# ---------------------------------------------------------------------------
# _swap_rollout_to_critic_tokens
# ---------------------------------------------------------------------------

def test_swap_filters_missing_critic_tokens():
    rollout = {"multimodal_train_inputs": [_mm(True), _mm(False), _mm(True)]}
    out = ta._swap_rollout_to_critic_tokens(rollout, torch.device("cpu"))
    assert len(out["tokens"]) == 2  # the no-key sample is dropped


def test_swap_all_missing_yields_empty():
    rollout = {"multimodal_train_inputs": [_mm(False), _mm(False)]}
    out = ta._swap_rollout_to_critic_tokens(rollout, torch.device("cpu"))
    assert len(out["tokens"]) == 0


# ---------------------------------------------------------------------------
# _truncate_to_cross_rank_min
# ---------------------------------------------------------------------------

def test_truncate_returns_none_when_empty():
    rollout = {
        "tokens": [], "total_lengths": [], "response_lengths": [],
        "loss_masks": [], "multimodal_train_inputs": [],
    }
    result = ta._truncate_to_cross_rank_min(rollout, dp_group=None, micro_batch_size=2)
    assert result is None


def test_truncate_returns_dict_when_nonempty():
    t = torch.tensor([1, 2, 3])
    rollout = {
        "tokens": [t, t],
        "total_lengths": [3, 3],
        "response_lengths": [0, 0],
        "loss_masks": [t, t],
        "multimodal_train_inputs": [_mm(True), _mm(True)],
    }
    out = ta._truncate_to_cross_rank_min(rollout, dp_group=None, micro_batch_size=2)
    assert out is not None
    assert len(out["tokens"]) == 2
    # world_size stub returns 1, n_min=2 → dynamic_global_batch_size=2
    assert out["dynamic_global_batch_size"] == 2


def test_truncate_micro_batch_rounding():
    """n_min=3 with micro_batch_size=2 → rounds to 2."""
    t = torch.tensor([1])
    rollout = {
        "tokens": [t, t, t],
        "total_lengths": [1, 1, 1],
        "response_lengths": [0, 0, 0],
        "loss_masks": [t, t, t],
        "multimodal_train_inputs": [_mm(True)] * 3,
    }
    out = ta._truncate_to_cross_rank_min(rollout, dp_group=None, micro_batch_size=2)
    assert out is not None
    assert len(out["tokens"]) == 2


# ---------------------------------------------------------------------------
# _note_empty_rollout / _reset_empty_rollout_counter
# ---------------------------------------------------------------------------

def test_page_fires_once(monkeypatch):
    calls = []
    monkeypatch.setenv("NLA_PINGME_CMD", "/bin/true")
    monkeypatch.delenv("NLA_MAX_EMPTY_STEPS", raising=False)
    monkeypatch.setattr(ta.subprocess, "Popen", lambda *a, **kw: calls.append(a))
    ta._note_empty_rollout("test")
    ta._note_empty_rollout("test")
    assert len(calls) == 1  # fire-once regardless of repeated calls


def test_no_page_without_cmd(monkeypatch):
    monkeypatch.delenv("NLA_PINGME_CMD", raising=False)
    monkeypatch.delenv("NLA_MAX_EMPTY_STEPS", raising=False)
    called = []
    monkeypatch.setattr(ta.subprocess, "Popen", lambda *a, **kw: called.append(a))
    ta._note_empty_rollout("test")
    assert len(called) == 0  # no crash, no page


def test_max_empty_steps_raises(monkeypatch):
    monkeypatch.setenv("NLA_MAX_EMPTY_STEPS", "2")
    monkeypatch.delenv("NLA_PINGME_CMD", raising=False)
    ta._note_empty_rollout("test")  # 1st: skip
    with pytest.raises(RuntimeError, match="consecutive"):
        ta._note_empty_rollout("test")  # 2nd: crash (>= 2)


def test_reset_counter_prevents_spurious_crash(monkeypatch):
    monkeypatch.setenv("NLA_MAX_EMPTY_STEPS", "2")
    monkeypatch.delenv("NLA_PINGME_CMD", raising=False)
    ta._note_empty_rollout("test")   # consecutive=1
    ta._reset_empty_rollout_counter()  # reset: consecutive=0
    ta._note_empty_rollout("test")   # consecutive=1 again — no crash
    # if reset didn't work this would raise on the second _note call
