from pathlib import Path

import pytest

from tokenize_cli.pricing import PricingTable

FIXTURES = Path(__file__).parent / "fixtures"
TEST_PRICING = FIXTURES / "pricing_test.yaml"


@pytest.fixture
def table() -> PricingTable:
    return PricingTable.load(TEST_PRICING)


@pytest.fixture
def real_table() -> PricingTable:
    return PricingTable.load()
