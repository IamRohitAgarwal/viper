"""Phase 6 — verify real-model modules import WITHOUT the heavy GPU/API deps.

The offline suite must stay importable on a machine with no anthropic/torch/
transformers. These modules use lazy imports, so importing them (and finding
the classes) must succeed; only *instantiating* pulls the heavy deps.
"""

from src.vlm.interface import VLMInterface


def test_anthropic_module_imports():
    from src.vlm.anthropic_vlm import AnthropicVLM

    assert issubclass(AnthropicVLM, VLMInterface)


def test_molmo_module_imports():
    from src.vlm.molmo_vlm import MolmoVLM

    assert issubclass(MolmoVLM, VLMInterface)


def test_kaggle_run_module_imports():
    import importlib

    mod = importlib.import_module("scripts.kaggle_run")
    assert hasattr(mod, "main")


def test_vlm_grounding_requires_locate_points():
    # vlm mode with a VLM lacking locate_points should raise a clear error.
    import pytest

    from src.grounding.locate import ground_locations
    from src.vlm.mock_vlm import MockVLM

    from .conftest import make_cfg

    cfg = make_cfg(GROUNDING_MODE="vlm")
    with pytest.raises(ValueError):
        ground_locations(None, "a", "b", cfg, vlm=MockVLM())
