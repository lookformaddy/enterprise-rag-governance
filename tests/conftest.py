import pytest

from enterprise_rag_governance.demo import ALICE, BOB, CAROL, DAVE, build_demo_system


@pytest.fixture()
def system():
    return build_demo_system()


@pytest.fixture()
def alice():
    return ALICE


@pytest.fixture()
def bob():
    return BOB


@pytest.fixture()
def carol():
    return CAROL


@pytest.fixture()
def dave():
    return DAVE
