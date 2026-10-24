"""Test business cost calculations for late deliveries."""

import pytest

from latedeliveries.evaluation import choose_threshold, order_cost


# Verify the total cost using a small example calculated by hand.
def test_order_cost_small_example():
    y_true = [1, 0, 0, 1, 0]
    flagged = [True, True, False, False, False]

    # Two vouchers cost $8; one missed late delivery costs $15.
    assert order_cost(y_true, flagged) == 23


# Verify the costs of flagging nobody and flagging everybody.
def test_order_cost_extreme_policies():
    y_true = [1, 0, 0, 1, 0]

    # No vouchers, but two missed late deliveries cost $30.
    assert order_cost(y_true, [False] * 5) == 30

    # Five vouchers cost $20; no late deliveries are missed.
    assert order_cost(y_true, [True] * 5) == 20


# Reject warning decisions that do not match the number of orders.
def test_order_cost_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        order_cost([1, 0, 1], [True, False])


# Verify that flagging nobody is cheapest when all orders are on time.
def test_threshold_can_flag_nobody():
    y_true = [0, 0, 0]
    probabilities = [0.10, 0.20, 0.30]

    threshold, cost = choose_threshold(y_true, probabilities)

    # No vouchers are needed because every delivery is on time.
    assert threshold == pytest.approx(0.35)
    assert cost == 0


# Verify that flagging everybody is cheapest when all orders are late.
def test_threshold_can_flag_everybody():
    y_true = [1, 1, 1]
    probabilities = [0.20, 0.50, 0.90]

    threshold, cost = choose_threshold(y_true, probabilities)

    # Three vouchers cost $12, less than missing late deliveries.
    assert threshold == pytest.approx(0.05)
    assert cost == 12
