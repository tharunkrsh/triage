from pathlib import Path

import pytest

from triage.pricing import PricingTable

FIXTURES = Path(__file__).parent / "fixtures"
TEST_PRICING = FIXTURES / "pricing_test.yaml"


@pytest.fixture
def table() -> PricingTable:
    return PricingTable.load(TEST_PRICING)


@pytest.fixture
def real_table() -> PricingTable:
    return PricingTable.load()


# Milestone 3: pricing fixture with fast-mode, service-tier and audit-threshold config.
M3_PRICING = FIXTURES / "pricing_m3.yaml"
AUDIT_FIXTURES = FIXTURES / "audit"


@pytest.fixture
def m3_table() -> PricingTable:
    return PricingTable.load(M3_PRICING)
