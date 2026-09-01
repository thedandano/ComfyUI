import math

import torch

import comfy.model_base
import comfy.model_management
import comfy.sampler_helpers


class _StubModel:
    """Minimal stand-in for BaseModel: just the attributes memory_required reads."""
    memory_usage_factor_conds = ()
    memory_usage_shape_process = {}
    memory_usage_factor = 2.0
    memory_required = comfy.model_base.BaseModel.memory_required

    def get_dtype_inference(self):
        return torch.bfloat16


INPUT_SHAPE = (1, 16, 1, 180, 320)
AREA = INPUT_SHAPE[0] * math.prod(INPUT_SHAPE[2:])
DTYPE_SIZE = 2  # bf16
EFFICIENT = AREA * DTYPE_SIZE * 0.01 * _StubModel.memory_usage_factor * (1024 * 1024)
CONSERVATIVE = AREA * 0.15 * _StubModel.memory_usage_factor * (1024 * 1024)


def _patch_attention(monkeypatch, xformers=False, pytorch_flash=False, flash=False):
    monkeypatch.setattr(comfy.model_management, "xformers_enabled", lambda: xformers)
    monkeypatch.setattr(comfy.model_management, "pytorch_attention_flash_attention", lambda: pytorch_flash)
    monkeypatch.setattr(comfy.model_management, "flash_attention_enabled", lambda: flash)


def _estimate(model_options={}):
    return comfy.model_base.BaseModel.memory_required(_StubModel(), INPUT_SHAPE, model_options=model_options)


def test_no_efficient_attention_uses_conservative_estimate(monkeypatch):
    _patch_attention(monkeypatch)
    assert _estimate() == CONSERVATIVE


def test_pytorch_flash_attention_uses_efficient_estimate(monkeypatch):
    _patch_attention(monkeypatch, pytorch_flash=True)
    assert _estimate() == EFFICIENT


def test_xformers_uses_efficient_estimate(monkeypatch):
    _patch_attention(monkeypatch, xformers=True)
    assert _estimate() == EFFICIENT


def test_flash_attention_flag_uses_efficient_estimate(monkeypatch):
    # --use-flash-attention must select the efficient estimate even when
    # pytorch attention was not auto enabled (e.g. torch builds without
    # working aotriton), otherwise the estimate is 7.5x too large.
    _patch_attention(monkeypatch, flash=True)
    assert _estimate() == EFFICIENT


def test_attention_override_uses_conservative_estimate(monkeypatch):
    # A ModelAttentionBackend override replaces the attention path outright,
    # so the estimate can't trust the global flash/xformers flags anymore.
    _patch_attention(monkeypatch, flash=True)
    model_options = {"transformer_options": {"optimized_attention_override": lambda *a, **k: None}}
    assert _estimate(model_options) == CONSERVATIVE


def test_attention_override_set_to_none_still_uses_conservative_estimate(monkeypatch):
    # wrap_attn (attention.py) branches on key presence, not truthiness.
    _patch_attention(monkeypatch, flash=True)
    model_options = {"transformer_options": {"optimized_attention_override": None}}
    assert _estimate(model_options) == CONSERVATIVE


def test_transformer_options_set_to_none_does_not_crash(monkeypatch):
    # transformer_options itself being None (not absent) shouldn't crash the .get() fallback.
    _patch_attention(monkeypatch, flash=True)
    model_options = {"transformer_options": None}
    assert _estimate(model_options) == EFFICIENT


def test_estimate_memory_prefers_active_run_options_over_patcher_options(monkeypatch):
    # estimate_memory() must use the model_options it's given (the CFGGuider's active,
    # possibly hook-cloned options), not the ModelPatcher's own model_options, or a
    # run-level override goes unseen.
    _patch_attention(monkeypatch, flash=True)

    class _StubPatcherModel:
        def __init__(self):
            self.model = _StubModel()
            self.model_options = {}  # no override on the patcher itself

    run_options = {"transformer_options": {"optimized_attention_override": lambda *a, **k: None}}
    _, minimum_memory_required = comfy.sampler_helpers.estimate_memory(
        _StubPatcherModel(), INPUT_SHAPE, conds={}, model_options=run_options
    )
    assert minimum_memory_required == CONSERVATIVE


def test_estimate_memory_handles_explicit_none_model_options(monkeypatch):
    # _prepare_sampling()'s own model_options defaults to None and passes it through
    # explicitly, so estimate_memory() has to tolerate that, not just an omitted arg.
    _patch_attention(monkeypatch, flash=True)

    class _StubPatcherModel:
        def __init__(self):
            self.model = _StubModel()
            self.model_options = {}

    _, minimum_memory_required = comfy.sampler_helpers.estimate_memory(
        _StubPatcherModel(), INPUT_SHAPE, conds={}, model_options=None
    )
    assert minimum_memory_required == EFFICIENT


def test_conservative_estimate_is_7_5x_efficient_at_bf16(monkeypatch):
    # Documents the size of the gap between the two formulas: at bf16 the
    # conservative path asks for 7.5x more working memory than the efficient
    # one for the same shapes. If either formula is retuned, this ratio (and
    # the impact of picking the wrong branch) changes; update it consciously.
    _patch_attention(monkeypatch, flash=True)
    efficient = _estimate()
    _patch_attention(monkeypatch)
    conservative = _estimate()
    assert conservative == 7.5 * efficient
