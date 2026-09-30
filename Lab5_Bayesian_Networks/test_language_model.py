"""
Unit and Property Tests for Bayesian Language Models
=====================================================
Validates probabilistic invariants, transition counts, determinism,
boundary token enforcement, and model comparisons.
"""

import pytest
import random
from language_model import (
    START_TOKEN,
    END_TOKEN,
    prepare_dataset,
    FirstOrderLanguageModel,
    SecondOrderLanguageModel,
)


@pytest.fixture
def dataset():
    return prepare_dataset()


@pytest.fixture
def first_order_model(dataset):
    model = FirstOrderLanguageModel()
    model.fit(dataset)
    return model


@pytest.fixture
def second_order_model(dataset):
    model = SecondOrderLanguageModel()
    model.fit(dataset)
    return model


def test_tokenization_boundaries(dataset):
    """Ensure every tokenized sentence begins with <START> and ends with <END>."""
    for sent in dataset:
        assert sent[0] == START_TOKEN
        assert sent[-1] == END_TOKEN
        # Ensure lowercase
        for token in sent:
            if token not in (START_TOKEN, END_TOKEN):
                assert token == token.lower()


def test_first_order_normalization_invariant(first_order_model):
    """
    Test Kolmogorov Axiom: for every context w, sum_{v} P(v | w) must equal 1.0.
    """
    norm_results = first_order_model.check_normalization(tolerance=1e-7)
    assert len(norm_results) > 0
    for word, total in norm_results.items():
        assert total == pytest.approx(1.0, rel=1e-6), f"Word '{word}' fails sum=1.0: got {total}"


def test_first_order_exact_probabilities(first_order_model):
    """
    Verify exact conditional probabilities derived from counting frequencies in the dataset.
    """
    # From <START>: only 'the' appears (6 times out of 6)
    assert first_order_model.probabilities[START_TOKEN]["the"] == pytest.approx(1.0)

    # From 'the': appears 12 times (3 cat, 3 dog, 2 mat, 2 rug, 2 park)
    the_dist = first_order_model.probabilities["the"]
    assert the_dist["cat"] == pytest.approx(3 / 12)
    assert the_dist["dog"] == pytest.approx(3 / 12)
    assert the_dist["mat"] == pytest.approx(2 / 12)
    assert the_dist["rug"] == pytest.approx(2 / 12)
    assert the_dist["park"] == pytest.approx(2 / 12)

    # From 'cat': appears 3 times (2 sat, 1 ran)
    cat_dist = first_order_model.probabilities["cat"]
    assert cat_dist["sat"] == pytest.approx(2 / 3)
    assert cat_dist["ran"] == pytest.approx(1 / 3)

    # From 'dog': appears 3 times (2 sat, 1 ran)
    dog_dist = first_order_model.probabilities["dog"]
    assert dog_dist["sat"] == pytest.approx(2 / 3)
    assert dog_dist["ran"] == pytest.approx(1 / 3)

    # Deterministic transitions
    assert first_order_model.probabilities["sat"]["on"] == pytest.approx(1.0)
    assert first_order_model.probabilities["ran"]["to"] == pytest.approx(1.0)
    assert first_order_model.probabilities["on"]["the"] == pytest.approx(1.0)
    assert first_order_model.probabilities["to"]["the"] == pytest.approx(1.0)
    assert first_order_model.probabilities["mat"][END_TOKEN] == pytest.approx(1.0)
    assert first_order_model.probabilities["rug"][END_TOKEN] == pytest.approx(1.0)
    assert first_order_model.probabilities["park"][END_TOKEN] == pytest.approx(1.0)


def test_zero_probability_transitions(first_order_model):
    """Verify that unobserved transitions have 0 probability in the distribution."""
    # 'cat' never follows 'cat'
    cat_dist = first_order_model.get_distribution("cat")
    assert "cat" not in cat_dist
    assert cat_dist.get("cat", 0.0) == 0.0

    # 'sat' never follows 'the' directly
    the_dist = first_order_model.get_distribution("the")
    assert "sat" not in the_dist
    assert the_dist.get("sat", 0.0) == 0.0


def test_greedy_generation_determinism(first_order_model, second_order_model):
    """Greedy generation should produce the exact same sequence every time."""
    m1_sent1 = first_order_model.generate(mode="greedy")
    m1_sent2 = first_order_model.generate(mode="greedy")
    assert m1_sent1 == m1_sent2

    m2_sent1 = second_order_model.generate(mode="greedy")
    m2_sent2 = second_order_model.generate(mode="greedy")
    assert m2_sent1 == m2_sent2


def test_sampling_generation_diversity(first_order_model):
    """Probabilistic sampling should produce variation across multiple seeds."""
    samples = set()
    for seed in range(50):
        rng = random.Random(seed)
        tokens = first_order_model.generate(mode="sample", rng=rng)
        samples.add(" ".join(tokens))
    assert len(samples) > 1, "Sampling must produce diverse outputs"


def test_second_order_normalization_invariant(second_order_model):
    """
    Test Kolmogorov Axiom for second order model:
    sum_{v} P(v | X_{t-2}, X_{t-1}) == 1.0 for every observed bigram context.
    """
    norm_results = second_order_model.check_normalization(tolerance=1e-7)
    assert len(norm_results) > 0
    for ctx, total in norm_results.items():
        assert total == pytest.approx(1.0, rel=1e-6), f"Context {ctx} fails sum=1.0: got {total}"


def test_second_order_disambiguation(second_order_model):
    """
    Verify that second-order context disambiguates positions of 'the':
    - ('<START>', 'the') only predicts subject nouns ('cat', 'dog').
    - ('on', 'the') only predicts resting objects ('mat', 'rug').
    - ('to', 'the') only predicts destinations ('park').
    """
    # Start position: ('<START>', 'the')
    start_the = second_order_model.probabilities[(START_TOKEN, "the")]
    assert start_the["cat"] == pytest.approx(0.5)
    assert start_the["dog"] == pytest.approx(0.5)
    assert "mat" not in start_the
    assert "rug" not in start_the
    assert "park" not in start_the

    # Preposition 'on': ('on', 'the')
    on_the = second_order_model.probabilities[("on", "the")]
    assert on_the["mat"] == pytest.approx(0.5)
    assert on_the["rug"] == pytest.approx(0.5)
    assert "cat" not in on_the
    assert "dog" not in on_the
    assert "park" not in on_the

    # Preposition 'to': ('to', 'the')
    to_the = second_order_model.probabilities[("to", "the")]
    assert to_the["park"] == pytest.approx(1.0)
    assert "mat" not in to_the
    assert "rug" not in to_the
    assert "cat" not in to_the
