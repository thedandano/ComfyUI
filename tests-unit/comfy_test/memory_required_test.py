import math

import torch

import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.sampler_helpers


class _StubModel:
    """Minimal stand-in for BaseModel: just the attributes memory_required reads."""
    memory_usage_factor_conds = ()
    memory_usage_shape_process = {}
    memory_usage_factor = 2.0
    memory_required = comfy.model_base.BaseModel.memory_required

    def get_dtype_inference(self):
        return torch.bfloat16


class _LegacyStubModel:
    """Mimics an out-of-tree BaseModel subclass that hasn't been updated for model_options."""
    def memory_required(self, input_shape, cond_shapes={}):
        return "legacy estimate"


class _StubPatcher:
    """Minimal stand-in for ModelPatcher: just the attributes memory_required reads."""
    memory_required = comfy.model_patcher.ModelPatcher.memory_required

    def __init__(self, model, model_options={}):
        self.model = model
        self.model_options = model_options


INPUT_SHAPE = (1, 16, 1, 180, 320)
AREA = INPUT_SHAPE[0] * math.prod(INPUT_SHAPE[2:])
DTYPE_SIZE = 2  # bf16
EFFICIENT = AREA * DTYPE_SIZE * 0.01 * _StubModel.memory_usage_factor * (1024 * 1024)
CONSERVATIVE = AREA * 0.15 * _StubModel.memory_usage_factor * (1024 * 1024)


def _patch_attention(monkeypatch, xformers=False, pytorch_flash=False, flash=False, amd=True):
    monkeypatch.setattr(comfy.model_management, "xformers_enabled", lambda: xformers)
    monkeypatch.setattr(comfy.model_management, "pytorch_attention_flash_attention", lambda: pytorch_flash)
    monkeypatch.setattr(comfy.model_management, "flash_attention_enabled", lambda: flash)
    monkeypatch.setattr(comfy.model_management, "is_amd", lambda: amd)


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
    _patch_attention(monkeypatch, flash=True, amd=True)
    model_options = {"transformer_options": {"optimized_attention_override": lambda *a, **k: None}}
    assert _estimate(model_options) == CONSERVATIVE


def test_attention_override_set_to_none_still_uses_conservative_estimate(monkeypatch):
    # wrap_attn (attention.py) branches on key presence, not truthiness.
    _patch_attention(monkeypatch, flash=True, amd=True)
    model_options = {"transformer_options": {"optimized_attention_override": None}}
    assert _estimate(model_options) == CONSERVATIVE


def test_transformer_options_set_to_none_does_not_crash(monkeypatch):
    # transformer_options itself being None (not absent) shouldn't crash the .get() fallback.
    _patch_attention(monkeypatch, flash=True, amd=True)
    model_options = {"transformer_options": None}
    assert _estimate(model_options) == EFFICIENT


def test_attention_override_ignored_on_non_amd(monkeypatch):
    # The aotriton-kernel-image gap this distrust protects against is AMD-only. On other
    # platforms an override (e.g. ModelAttentionBackend's "pytorch attention" on CUDA, which
    # is already the efficient SDPA path) shouldn't be penalized with the conservative estimate.
    _patch_attention(monkeypatch, flash=True, amd=False)
    model_options = {"transformer_options": {"optimized_attention_override": lambda *a, **k: None}}
    assert _estimate(model_options) == EFFICIENT


def test_model_patcher_falls_back_for_legacy_memory_required_signature(monkeypatch):
    # An out-of-tree BaseModel subclass that hasn't been updated for model_options must not
    # crash - it should keep getting its own (pre-fix) estimate, not a TypeError.
    _patch_attention(monkeypatch, flash=True, amd=True)
    patcher = _StubPatcher(_LegacyStubModel(), model_options={"transformer_options": {"optimized_attention_override": lambda *a, **k: None}})
    assert patcher.memory_required(INPUT_SHAPE) == "legacy estimate"


def test_model_patcher_passes_model_options_for_updated_models(monkeypatch):
    # A model whose memory_required accepts model_options gets the real override-aware estimate.
    _patch_attention(monkeypatch, flash=True, amd=True)
    patcher = _StubPatcher(_StubModel(), model_options={"transformer_options": {"optimized_attention_override": lambda *a, **k: None}})
    assert patcher.memory_required(INPUT_SHAPE) == CONSERVATIVE


def test_estimate_memory_prefers_active_run_options_over_patcher_options(monkeypatch):
    # estimate_memory() must use the model_options it's given (the CFGGuider's active,
    # possibly hook-cloned options), not the ModelPatcher's own model_options, or a
    # run-level override goes unseen.
    _patch_attention(monkeypatch, flash=True, amd=True)

    patcher = _StubPatcher(_StubModel())  # no override on the patcher itself
    run_options = {"transformer_options": {"optimized_attention_override": lambda *a, **k: None}}
    _, minimum_memory_required = comfy.sampler_helpers.estimate_memory(
        patcher, INPUT_SHAPE, conds={}, model_options=run_options
    )
    assert minimum_memory_required == CONSERVATIVE


def test_estimate_memory_handles_explicit_none_model_options(monkeypatch):
    # _prepare_sampling()'s own model_options defaults to None and passes it through
    # explicitly, so estimate_memory() has to tolerate that, not just an omitted arg.
    _patch_attention(monkeypatch, flash=True, amd=True)

    patcher = _StubPatcher(_StubModel())
    _, minimum_memory_required = comfy.sampler_helpers.estimate_memory(
        patcher, INPUT_SHAPE, conds={}, model_options=None
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
