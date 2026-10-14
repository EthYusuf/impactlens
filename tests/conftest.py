from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from impactlens.simulate import ScenarioConfig, generate_panel  # noqa: E402


@pytest.fixture(scope="session")
def panel():
    """A standard synthetic panel: 8 controls, +12% effect, confounding shock."""
    return generate_panel(ScenarioConfig(seed=7))


@pytest.fixture(scope="session")
def ds(panel):
    return panel.to_dataset()


@pytest.fixture(scope="session")
def null_panel():
    """Same data generating process with no effect inserted."""
    return generate_panel(ScenarioConfig(true_effect=0.0, seed=11))


@pytest.fixture(scope="session")
def null_ds(null_panel):
    return null_panel.to_dataset()


@pytest.fixture(scope="session")
def no_control_ds():
    """A panel with only the treated unit, for the ITS path."""
    return generate_panel(ScenarioConfig(n_controls=0, seed=3)).to_dataset()
