import pytest

from evaluation.rouge_scorer import compute_rouge
from evaluation.bleu_scorer import compute_bleu
from evaluation.aggregator import WEIGHTS, composite_score, abstractiveness_score, compression_ratio

REFERENCE = ("The city council approved a new budget on Monday that raises "
             "spending on public transport and cuts funding for road expansion.")


def test_rouge_against_reference_is_f1_not_precision():
    # Copying three words of the reference is perfectly "precise" but a poor summary.
    pred = "council approved budget"
    ref_mode = compute_rouge([pred], [REFERENCE])
    src_mode = compute_rouge([pred], [REFERENCE], against_source=True)
    assert src_mode["ROUGE-1"] == 1.0          # precision
    assert ref_mode["ROUGE-1"] < 0.35          # F1 punishes the missing content


def test_rouge_identical_text_scores_one_in_both_modes():
    for against_source in (False, True):
        scores = compute_rouge([REFERENCE], [REFERENCE], against_source=against_source)
        assert scores == {"ROUGE-1": 1.0, "ROUGE-2": 1.0, "ROUGE-L": 1.0}


def test_bleu_against_reference_is_bleu4():
    assert compute_bleu([REFERENCE], [REFERENCE])["BLEU"] == 1.0
    # Same words in a scrambled order: unigram precision stays 1.0, BLEU-4 drops.
    scrambled = " ".join(reversed(REFERENCE.split()))
    assert compute_bleu([scrambled], [REFERENCE], against_source=True)["BLEU"] == 1.0
    assert compute_bleu([scrambled], [REFERENCE])["BLEU"] < 0.2


def test_bleu_source_mode_has_no_brevity_penalty():
    # A short, fully supported summary of a long source should not score ~0.
    source = REFERENCE * 10
    assert compute_bleu(["council approved a new budget"], [source], against_source=True)["BLEU"] == 1.0
    assert compute_bleu(["council approved a new budget"], [source])["BLEU"] < 0.05


def test_bleu_empty_prediction_scores_zero():
    assert compute_bleu([""], [REFERENCE])["BLEU"] == 0.0


def test_composite_weights_come_from_config_and_sum_to_one():
    assert set(WEIGHTS) == {"ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore_F1", "METEOR"}
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    perfect = {k: 1.0 for k in WEIGHTS}
    assert composite_score(perfect) == pytest.approx(1.0)


def test_novelty_metrics():
    assert abstractiveness_score("budget approved", "budget approved today") == 0.0
    assert abstractiveness_score("money passed", "budget approved today") == 1.0
    assert compression_ratio("one two", "one two three four") == 0.5
    assert compression_ratio("anything", "") == 0.0
