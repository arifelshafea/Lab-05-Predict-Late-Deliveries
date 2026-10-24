"""Shared fixtures for Lab 05 tests."""

import pytest

from latedeliveries.data import make_orders


# Provide the same generated dataset to tests that need it.
@pytest.fixture
def orders():
    """Return the reproducible dataset of 3,000 orders."""
    return make_orders()
