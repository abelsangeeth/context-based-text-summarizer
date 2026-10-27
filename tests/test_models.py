import pytest
import torch
from transformers import PegasusConfig, PegasusForConditionalGeneration

from summarizer.models import restore_sinusoidal_positions
from summarizer.postprocessing import postprocess_summary


def _tiny_pegasus():
    cfg = PegasusConfig(vocab_size=64, d_model=16, encoder_layers=1, decoder_layers=1,
                        encoder_attention_heads=2, decoder_attention_heads=2,
                        encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=32)
    return PegasusForConditionalGeneration(cfg)


def _position_tables(model):
    return [m for m in model.modules() if hasattr(m, "create_weight")]


def test_random_position_tables_are_rebuilt():
    # What some transformers 5.x releases did on load: the tables are not in the
    # checkpoint and were left randomly initialised (PEGASUS then emits gibberish).
    model = _tiny_pegasus()
    tables = _position_tables(model)
    assert len(tables) == 2
    for t in tables:
        torch.nn.init.normal_(t.weight)

    restore_sinusoidal_positions(model)

    for t in tables:
        assert torch.equal(t.weight, t.create_weight())


def test_intact_position_tables_are_left_alone(monkeypatch):
    model = _tiny_pegasus()
    for t in _position_tables(model):
        with torch.no_grad():
            t.weight.copy_(t.create_weight())
    before = [t.weight.clone() for t in _position_tables(model)]

    calls = []
    cls = type(_position_tables(model)[0])
    original = cls.create_weight
    monkeypatch.setattr(cls, "create_weight", lambda self: calls.append(1) or original(self))
    restore_sinusoidal_positions(model)

    assert calls == []   # rebuilding PEGASUS's tables costs ~5 s; skip when not needed
    assert all(torch.equal(a, t.weight) for a, t in zip(before, _position_tables(model)))


@pytest.mark.parametrize("raw, expected", [
    ("Rates rose.<n>Markets fell.", "Rates rose. Markets fell."),
    ("the the budget passed", "The budget passed."),
    ("Same point. Same point. New point.", "Same point. New point."),
])
def test_postprocess_summary(raw, expected):
    assert postprocess_summary(raw) == expected
