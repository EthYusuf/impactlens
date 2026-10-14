"""The audit of the audit.

These tests do not re-verify the statistics (that is what the recovery study
itself does, at a sample size tests cannot afford). They verify the harness
reports the truth about what it found: that it scores against the real inserted
effect, that coverage and power are computed the right way round, and that a
deliberately broken estimator is labelled overconfident instead of quietly
passing.
"""

from __future__ import annotations

import numpy as np
import pytest

from impactlens.simulate import ScenarioConfig, generate_panel, naive_prepost_effect
from impactlens.validation import false_positive_study, run_recovery_study


@pytest.fixture(scope="module")
def study():
    return run_recovery_study(
        ScenarioConfig(n_controls=6, seed=0),
        n_replications=12,
        estimators=["synthetic_control", "naive"],
        base_seed=700,
    )


def test_study_runs_every_replication_for_every_estimator(study):
    assert study.n_replications == 12
    assert set(study.summary.index) == {"synthetic_control", "naive"}
    assert (study.summary["n_runs"] == 12).all()


def test_summary_columns_are_the_four_that_matter(study):
    for col in ("bias", "rmse", "coverage", "power"):
        assert col in study.summary.columns
    assert ((study.summary["coverage"] >= 0) & (study.summary["coverage"] <= 1)).all()
    assert ((study.summary["power"] >= 0) & (study.summary["power"] <= 1)).all()


def test_bias_is_measured_against_the_inserted_effect(study):
    runs = study.runs.dropna(subset=["estimate"])
    recomputed = (runs["estimate"] - runs["true_effect"]).mean()
    pooled = runs.groupby("estimator")["error_abs"].mean().mean()
    assert np.isfinite(recomputed) and np.isfinite(pooled)
    sub = runs[runs["estimator"] == "synthetic_control"]
    assert sub["error_abs"].mean() == pytest.approx(
        study.summary.loc["synthetic_control", "bias"]
    )


def test_rmse_is_consistent_with_the_run_level_errors(study):
    sub = study.runs[study.runs["estimator"] == "synthetic_control"]
    expected = float(np.sqrt((sub["error_abs"] ** 2).mean()))
    assert study.summary.loc["synthetic_control", "rmse"] == pytest.approx(expected)


def test_coverage_counts_intervals_that_contain_the_truth(study):
    sub = study.runs[study.runs["estimator"] == "synthetic_control"]
    manual = (
        (sub["ci_lower"] <= sub["true_effect"]) & (sub["true_effect"] <= sub["ci_upper"])
    ).mean()
    assert study.summary.loc["synthetic_control", "coverage"] == pytest.approx(manual)


def test_verdict_flags_the_naive_estimator_as_overconfident(study):
    verdict = study.verdict()
    assert verdict.loc["naive", "verdict"] == "overconfident", (
        "the naive estimator's intervals never contain the truth under a market "
        "shock, so the harness must say so"
    )
    assert verdict.loc["synthetic_control", "verdict"] == "pass"


def test_control_based_estimator_beats_naive_on_rmse(study):
    assert (
        study.summary.loc["synthetic_control", "rmse"]
        < study.summary.loc["naive", "rmse"]
    )


def test_false_positive_study_inserts_no_effect():
    result = false_positive_study(
        ScenarioConfig(n_controls=6),
        n_replications=10,
        estimators=["synthetic_control"],
        base_seed=800,
    )
    assert result.config.true_effect == 0.0
    assert (result.summary["true_effect"].abs() < 1e-9).all()
    # With no effect, "power" is the false-positive rate and should stay low.
    assert result.summary.loc["synthetic_control", "power"] <= 0.3


# ------------------------------------------------------------------ simulation
def test_ground_truth_matches_the_configured_effect():
    panel = generate_panel(ScenarioConfig(
        true_effect=0.15, ramp_periods=0, effect_kind="multiplicative", seed=1,
    ))
    # With no ramp the realised relative effect should equal the configured one.
    assert panel.true_relative_effect == pytest.approx(0.15, abs=0.01)


def test_ramp_lowers_the_realised_average_effect():
    kw = dict(true_effect=0.15, seed=1)
    instant = generate_panel(ScenarioConfig(ramp_periods=0, **kw))
    ramped = generate_panel(ScenarioConfig(ramp_periods=10, **kw))
    assert ramped.true_average_effect < instant.true_average_effect


def test_additive_effect_is_exactly_the_configured_size():
    panel = generate_panel(ScenarioConfig(
        true_effect=25.0, effect_kind="additive", ramp_periods=0, seed=2,
    ))
    assert panel.true_average_effect == pytest.approx(25.0)
    assert panel.true_cumulative_effect == pytest.approx(25.0 * panel.config.n_post)


def test_counterfactual_equals_observation_before_the_intervention():
    panel = generate_panel(ScenarioConfig(seed=3))
    observed = panel.frame[panel.frame["unit"] == panel.treated_unit]
    observed = observed.set_index("date")["y"].sort_index()
    pre = observed.index < panel.intervention_date
    np.testing.assert_allclose(
        observed[pre].to_numpy(),
        panel.true_counterfactual[pre].to_numpy(),
    )


def test_naive_prepost_helper_is_inflated_by_the_shock():
    panel = generate_panel(ScenarioConfig(seed=4))
    assert naive_prepost_effect(panel) > panel.true_average_effect


def test_zero_effect_scenario_has_zero_ground_truth():
    panel = generate_panel(ScenarioConfig(true_effect=0.0, seed=5))
    assert panel.true_average_effect == pytest.approx(0.0)
    assert panel.true_cumulative_effect == pytest.approx(0.0)


def test_reproducible_for_a_given_seed():
    a = generate_panel(ScenarioConfig(seed=99)).frame
    b = generate_panel(ScenarioConfig(seed=99)).frame
    np.testing.assert_allclose(a["y"].to_numpy(), b["y"].to_numpy())


def test_invalid_config_rejected():
    with pytest.raises(ValueError, match="effect_kind"):
        ScenarioConfig(effect_kind="exponential")
    with pytest.raises(ValueError, match="n_pre"):
        ScenarioConfig(n_periods=10, n_pre=10)
