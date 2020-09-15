
import pytest
from nintorch.utils import seed


def test_seed():
    SEED: int = 2020
    seed(SEED)

