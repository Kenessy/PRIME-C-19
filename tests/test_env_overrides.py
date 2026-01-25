import importlib
import sys

import pytest

from prime_c19.settings import load_settings


def test_load_settings_respects_env_overrides(monkeypatch):
    monkeypatch.setenv("TP6_RING_LEN", "123")
    monkeypatch.setenv("TP6_SLOT_DIM", "99")
    cfg = load_settings()
    assert cfg.ring_len == 123
    assert cfg.slot_dim == 99


def test_tournament_phase6_reads_direct_env_overrides(monkeypatch):
    # TP6_EXPERT_HEADS is read directly in tournament_phase6, not via load_settings.
    monkeypatch.setenv("TP6_EXPERT_HEADS", "7")

    # Ensure a fresh import so the module-level constant is computed from env.
    sys.modules.pop("tournament_phase6", None)
    import tournament_phase6 as tp6

    assert tp6.EXPERT_HEADS == 7

    # Cleanup: exercise reload path as well.
    monkeypatch.setenv("TP6_EXPERT_HEADS", "3")
    tp6 = importlib.reload(tp6)
    assert tp6.EXPERT_HEADS == 3
