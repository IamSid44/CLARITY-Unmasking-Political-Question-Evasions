"""Tests for the C1 factorized head.

The numerical properties here are the ones that fail SILENTLY if wrong: an
underflowing product shows up as an untrainable rare class, not as an exception.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from higrec.data.attributes import ATTRIBUTE_CODE, ATTRIBUTES, attribute_index_matrix
from higrec.data.labels import EVASION_LABELS, N_EVASION
from higrec.models.factorized import (
    ATTRIBUTE_SIZES,
    FactorizedConfig,
    FactorizedHead,
    attribute_sum_loss,
    factorized_leaf_logits,
    random_code_index,
)


def test_head_sizes_match_the_code_table():
    assert ATTRIBUTE_SIZES == (3, 3, 2, 2, 3)


def test_output_shape_is_drop_in_compatible_with_a_flat_head():
    """P4.1 requires the ONLY difference from the baseline be the parameterization."""
    head = FactorizedHead(FactorizedConfig(hidden_size=32))
    out = head(torch.randn(7, 32))
    assert out.shape == (7, N_EVASION)


def test_posterior_is_normalized_over_the_nine_valid_codes():
    head = FactorizedHead(FactorizedConfig(hidden_size=16))
    log_posterior = head(torch.randn(5, 16))
    totals = log_posterior.exp().sum(dim=-1)
    assert torch.allclose(totals, torch.ones(5), atol=1e-6)


def test_one_hot_attribute_heads_recover_the_correct_leaf_with_probability_one():
    """The specification test from P4.1.

    If every attribute head is certain of the true leaf's code, the renormalized
    leaf posterior must place essentially all mass on that leaf.
    """
    code_index = torch.from_numpy(attribute_index_matrix()).long()

    for leaf_i, leaf in enumerate(EVASION_LABELS):
        logits = []
        for attr_j, attr in enumerate(ATTRIBUTES):
            size = ATTRIBUTE_SIZES[attr_j]
            confident = torch.full((1, size), -50.0)
            confident[0, code_index[leaf_i, attr_j]] = 50.0
            logits.append(confident)

        posterior = torch.log_softmax(
            factorized_leaf_logits(logits, code_index), dim=-1
        ).exp()

        assert posterior[0, leaf_i] == pytest.approx(1.0, abs=1e-5), leaf
        assert posterior.argmax().item() == leaf_i


def test_invalid_attribute_combinations_receive_exactly_zero_mass():
    """Only 9 of 3*3*2*2*3 = 108 combinations are valid codes.

    The invalid 99 must be unreachable by construction, not merely improbable.
    """
    head = FactorizedHead(FactorizedConfig(hidden_size=8))
    posterior = head(torch.randn(3, 8)).exp()

    assert posterior.shape[1] == N_EVASION
    assert torch.allclose(posterior.sum(dim=-1), torch.ones(3), atol=1e-6)
    assert (posterior >= 0).all()


def test_computation_is_numerically_stable_under_extreme_logits():
    """Five multiplied probabilities underflow float32 readily.

    Log-space arithmetic must survive logits that would zero out a naive product.
    """
    head = FactorizedHead(FactorizedConfig(hidden_size=8))
    with torch.no_grad():
        for layer in head.heads:
            layer.weight.mul_(1000.0)
            layer.bias.mul_(1000.0)

    log_posterior = head(torch.randn(4, 8) * 100)

    assert torch.isfinite(log_posterior).all(), "log-space arithmetic overflowed"
    assert torch.allclose(log_posterior.exp().sum(dim=-1), torch.ones(4), atol=1e-5)


def test_gradients_flow_to_every_attribute_head():
    """A head receiving no gradient would be silently dead weight."""
    head = FactorizedHead(FactorizedConfig(hidden_size=16))
    target = torch.randint(0, N_EVASION, (6,))
    head.loss(torch.randn(6, 16), target).backward()

    for j, layer in enumerate(head.heads):
        assert layer.weight.grad is not None, f"head {ATTRIBUTES[j]} got no gradient"
        assert torch.any(layer.weight.grad != 0), f"head {ATTRIBUTES[j]} gradient is all zero"


# --- Objectives ------------------------------------------------------------


def test_marginal_objective_is_cross_entropy_on_the_leaf_posterior():
    head = FactorizedHead(FactorizedConfig(hidden_size=16, objective="marginal"))
    hidden = torch.randn(5, 16)
    target = torch.randint(0, N_EVASION, (5,))

    expected = torch.nn.functional.nll_loss(head(hidden), target)
    assert head.loss(hidden, target).item() == pytest.approx(expected.item(), abs=1e-6)


def test_attribute_sum_objective_supervises_each_head_separately():
    head = FactorizedHead(FactorizedConfig(hidden_size=16, objective="attribute_sum"))
    hidden = torch.randn(5, 16)
    target = torch.randint(0, N_EVASION, (5,))

    expected = attribute_sum_loss(head.attribute_logits(hidden), target, head.code_index)
    assert head.loss(hidden, target).item() == pytest.approx(expected.item(), abs=1e-6)


def test_both_objective_requires_a_positive_auxiliary_weight():
    with pytest.raises(ValueError, match="aux_weight"):
        FactorizedConfig(hidden_size=8, objective="both", aux_weight=0.0)


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="unknown objective"):
        FactorizedConfig(hidden_size=8, objective="whatever_works")


def test_attribute_sum_loss_is_zero_for_a_perfect_prediction():
    code_index = torch.from_numpy(attribute_index_matrix()).long()
    target = torch.tensor([0])
    logits = []
    for attr_j in range(len(ATTRIBUTES)):
        confident = torch.full((1, ATTRIBUTE_SIZES[attr_j]), -50.0)
        confident[0, code_index[0, attr_j]] = 50.0
        logits.append(confident)

    assert attribute_sum_loss(logits, target, code_index).item() == pytest.approx(0.0, abs=1e-4)


# --- The random-code control ----------------------------------------------


def test_random_code_preserves_uniqueness_and_attribute_balance():
    """The P4.2 control must differ ONLY in semantics.

    A freshly sampled random table would also change per-attribute value
    distributions, confounding the control with a change in attribute balance.
    Permuting rows of the real table preserves both exactly.
    """
    real = attribute_index_matrix()
    shuffled = random_code_index(seed=0).numpy()

    assert len({tuple(r) for r in shuffled}) == N_EVASION, "codes must stay unique"
    for attr_j in range(len(ATTRIBUTES)):
        assert sorted(shuffled[:, attr_j]) == sorted(real[:, attr_j]), (
            "per-attribute value distribution changed; the control is confounded"
        )


def test_random_code_actually_differs_from_the_semantic_one():
    real = attribute_index_matrix()
    for seed in range(5):
        if not np.array_equal(random_code_index(seed=seed).numpy(), real):
            return
    pytest.fail("random code identical to the semantic code across all seeds tried")


def test_random_code_is_deterministic_given_a_seed():
    assert torch.equal(random_code_index(seed=7), random_code_index(seed=7))


# --- Parameter accounting --------------------------------------------------


def test_trainable_parameter_count_is_reported_for_the_matched_control():
    """P3.2's capacity control needs this number to match LoRA rank against."""
    head = FactorizedHead(FactorizedConfig(hidden_size=64))
    expected = sum(64 * size + size for size in ATTRIBUTE_SIZES)
    assert head.trainable_parameter_count() == expected


def test_code_table_travels_with_the_checkpoint():
    """The code table is a persistent buffer, so a checkpoint records its structure."""
    head = FactorizedHead(FactorizedConfig(hidden_size=8))
    assert "code_index" in head.state_dict()
    assert np.array_equal(head.state_dict()["code_index"].numpy(), attribute_index_matrix())


def test_code_version_is_recorded_in_the_config():
    config = FactorizedConfig(hidden_size=8)
    assert config.code_version == "c1-attribute-code-v1"
