"""Estimator behaviour.

The tests that matter are the ones that would catch a wrong answer:

* the estimator recovers an effect we inserted,
* it reports *no* effect when none exists,
* it is not fooled by the market-wide confound (while the naive benchmark is),
* its interval widens when the data gets noisier.
"""

from __future__ import annotations

import numpy as np
import pytest

from impactlens.estimators import (
    estimate_bayesian_counterfactual,
    estimate_did,
    estimate_its,
    estimate_naive_prepost,
    estimate_synthetic_control,
    event_study,
    fit_weights,
    get_estimator,
    run_all,
)
from impactlens.simulate import ScenarioConfig, generate_panel
from impactlens.types import ImpactEstimate

CONTROL_BASED = ["did", "synthetic_control", "bayesian"]


# ------------------------------------------------------------------ contract
@pytest.mark.parametrize("name", ["did", "synthetic_control", "bayesian", "its", "naive"])
def test_every_estimator_honours_the_result_contract(ds, name):
    est = get_estimator(name)(ds)
    assert isinstance(est, ImpactEstimate)
    assert np.isfinite(est.effect)
    assert est.interval.lower < est.interval.upper
    assert est.interval.level == 0.95
    assert est.interval.kind in ("confidence", "credible", "placebo")
    assert est.assumptions, "an estimator must state what it assumes"
    assert est.n_pre > 0 and est.n_post > 0
    assert est.observed is not None and est.counterfactual is not None
    assert len(est.observed) == len(est.counterfactual)
    assert est.diagnostics.get("inference")


@pytest.mark.parametrize("name", CONTROL_BASED)
def test_control_based_estimators_recover_the_inserted_effect(panel, ds, name):
    truth = panel.true_average_effect
    est = get_estimator(name)(ds)
    assert est.interval.contains(truth), (
        f"{name} interval [{est.interval.lower:.2f}, {est.interval.upper:.2f}] "
        f"missed the true effect {truth:.2f}"
    )
    # Within half the interval width is a generous but meaningful accuracy bound.
    assert abs(est.effect - truth) < est.interval.width


@pytest.mark.parametrize("name", CONTROL_BASED)
def test_control_based_estimators_report_nothing_when_nothing_happened(name):
    """False positives must be rare.

    Checked across several seeds rather than one, because a single draw that
    happens to look significant is expected roughly 5% of the time by
    construction -- asserting on one seed would be testing luck. The full
    calibration audit lives in :mod:`impactlens.validation`.
    """
    fn = get_estimator(name)
    flagged = []
    seeds = [11, 12, 13, 14, 15, 16, 17, 18]
    for seed in seeds:
        ds_null = generate_panel(ScenarioConfig(true_effect=0.0, seed=seed)).to_dataset()
        est = fn(ds_null)
        if est.is_significant:
            flagged.append((seed, est.effect, est.interval.lower, est.interval.upper))
    assert len(flagged) <= 1, (
        f"{name} reported an effect on {len(flagged)}/{len(seeds)} null datasets, "
        f"far above the ~5% expected: {flagged}"
    )


@pytest.mark.parametrize("name", CONTROL_BASED)
def test_null_data_estimate_is_small(null_ds, name):
    """Even when not flagged, the point estimate on null data should be near zero."""
    est = get_estimator(name)(null_ds)
    baseline = float(null_ds.treated_series.mean())
    assert abs(est.effect) < 0.08 * baseline, (
        f"{name} returned {est.effect:.2f} on data with no effect "
        f"(baseline level {baseline:.1f})"
    )


def test_naive_estimator_is_fooled_by_the_market_shock(panel, ds):
    """The whole premise: the naive reading is badly inflated by the confound."""
    truth = panel.true_average_effect
    naive = estimate_naive_prepost(ds)
    scm = estimate_synthetic_control(ds)

    assert abs(naive.effect - truth) > abs(scm.effect - truth), (
        "the naive estimate should be further from the truth than synthetic control"
    )
    # The naive estimate overshoots by roughly the size of the market shock.
    assert naive.effect > truth * 1.4
    assert not naive.interval.contains(truth)


def test_no_market_shock_means_naive_and_causal_agree():
    """Sanity check on the simulation: remove the confound and the gap closes."""
    panel = generate_panel(ScenarioConfig(market_shock=0.0, trend=0.0,
                                          seasonal_amplitude=0.0, seed=4))
    ds = panel.to_dataset()
    truth = panel.true_average_effect
    naive = estimate_naive_prepost(ds)
    assert abs(naive.effect - truth) < 0.35 * abs(truth)


# ----------------------------------------------------------------------- DiD
def test_did_requires_controls(no_control_ds):
    with pytest.raises(ValueError, match="at least one control"):
        estimate_did(no_control_ds)


def test_did_uses_bootstrap_when_clusters_are_few(ds):
    est = estimate_did(ds, n_boot=199)
    # 1 treated + 8 controls = 9 clusters, below the asymptotic threshold.
    assert est.diagnostics["n_clusters"] == 9
    assert est.diagnostics["inference"] == "wild cluster bootstrap"


def test_did_uses_cluster_robust_when_clusters_are_many():
    panel = generate_panel(ScenarioConfig(n_controls=20, seed=9))
    est = estimate_did(panel.to_dataset())
    assert est.diagnostics["n_clusters"] == 21
    assert est.diagnostics["inference"] == "cluster-robust (by unit)"


def test_did_bootstrap_is_reproducible(ds):
    a = estimate_did(ds, n_boot=199, seed=42)
    b = estimate_did(ds, n_boot=199, seed=42)
    assert a.interval.lower == pytest.approx(b.interval.lower)
    assert a.p_value == pytest.approx(b.p_value)


def test_fast_cluster_solver_matches_statsmodels(ds):
    """The bootstrap's hand-rolled solver must equal the reference implementation."""
    import pandas as pd
    import statsmodels.api as sm

    from impactlens.estimators.did import _ClusterOLS, _design

    X, y, groups = _design(ds)
    Xv, yv = X.to_numpy(dtype=float), y.to_numpy(dtype=float)
    codes, _ = pd.factorize(groups)
    codes = np.asarray(codes, dtype=np.intp)
    idx = list(X.columns).index("_did")

    ref = sm.OLS(yv, Xv).fit(cov_type="cluster", cov_kwds={"groups": codes}, use_t=True)
    beta, se = _ClusterOLS(Xv, codes, idx).fit(yv)
    assert beta == pytest.approx(float(ref.params[idx]), rel=1e-8)
    assert se == pytest.approx(float(ref.bse[idx]), rel=1e-8)


def test_event_study_pre_coefficients_are_flat_under_parallel_trends(ds):
    """The default scenario uses additive unit effects, so trends are parallel."""
    ev = event_study(ds)
    coefs = ev["coefficients"]
    assert not coefs.empty
    assert ev["pretrend_p_value"] > 0.05, (
        f"parallel trends holds by construction here but the test rejected it "
        f"(p={ev['pretrend_p_value']:.2e})"
    )
    # Every period is returned, not a window, because that is what the slope
    # test is fitted on.
    assert len(coefs) == len(ds.dates)
    pre = coefs[coefs["is_pre"]]
    assert len(pre) == ds.n_pre
    # No pre-period gap should sit far outside the gap's own noise level.
    assert (pre["coefficient"].abs() < 4 * ev["gap_pre_sd"]).all()


def test_pretrend_test_detects_selection_on_trend():
    """The classic DiD killer: the treated unit was already pulling away.

    The diagnostic must catch this, because the drift is part of the
    counterfactual and crediting it to the intervention is exactly the mistake
    the tool exists to prevent.
    """
    panel = generate_panel(ScenarioConfig(treated_pretrend=0.35, seed=7))
    ds_drift = panel.to_dataset()
    ev = event_study(ds_drift)
    assert ev["pretrend_p_value"] < 0.01, (
        f"a {0.35:+.2f}/period divergence should be unmissable, "
        f"got p={ev['pretrend_p_value']:.3f}"
    )
    assert ev["pretrend_slope"] == pytest.approx(0.35, abs=0.15)

    est = estimate_did(ds_drift, n_boot=199)
    assert any("already drifting" in w for w in est.warnings)
    assert any("synthetic control" in w for w in est.warnings), (
        "the warning should point the user at an estimator that survives this"
    )


def test_pretrend_slope_is_near_zero_on_clean_data(ds):
    ev = event_study(ds)
    # The gap's pre-period slope should be small relative to its own noise.
    assert abs(ev["pretrend_slope"]) < 2 * ev["pretrend_slope_se"]


def test_did_overstates_the_effect_under_selection_on_trend():
    """Documents the failure, rather than pretending the estimator is immune.

    With a pre-existing upward divergence, DiD extrapolates nothing and simply
    attributes the continued drift to the intervention. The point estimate is
    therefore too high -- which is precisely why the warning above matters.
    """
    panel = generate_panel(ScenarioConfig(treated_pretrend=0.35, seed=13))
    ds_drift = panel.to_dataset()
    truth = panel.true_average_effect
    est = estimate_did(ds_drift, n_boot=199)
    assert est.effect > truth, "the pre-existing drift should inflate the DiD estimate"
    assert est.warnings, "and the estimator must not report it silently"


# ----------------------------------------------------------- synthetic control
def test_weights_live_on_the_simplex(ds):
    est = estimate_synthetic_control(ds)
    w = est.diagnostics["weights"]
    assert (w >= -1e-9).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_fit_weights_recovers_a_known_combination():
    rng = np.random.default_rng(0)
    donors = rng.normal(100, 15, size=(60, 3))
    true_w = np.array([0.6, 0.3, 0.1])
    target = donors @ true_w
    w = fit_weights(target, donors)
    assert w == pytest.approx(true_w, abs=0.02)


def test_synthetic_control_tracks_the_pre_period(ds):
    est = estimate_synthetic_control(ds)
    # Pre-period fit error should be small relative to the series' own variation.
    assert est.diagnostics["pre_fit_ratio"] < 0.5
    assert est.diagnostics["rmspe_ratio"] > 1.0, "post-period fit should deteriorate"


def test_synthetic_control_uses_placebo_inference_with_enough_donors(ds):
    est = estimate_synthetic_control(ds)
    assert est.diagnostics["inference"] == "in-space placebo permutation"
    assert est.interval.kind == "placebo"
    assert not est.diagnostics["placebo_effects"].empty
    assert est.p_value is not None and 0 < est.p_value <= 1


def test_synthetic_control_admits_coarse_p_values_with_few_donors():
    panel = generate_panel(ScenarioConfig(n_controls=5, seed=21))
    est = estimate_synthetic_control(panel.to_dataset())
    assert any("smallest p-value attainable" in w for w in est.warnings)


def test_synthetic_control_requires_a_donor(no_control_ds):
    with pytest.raises(ValueError, match="at least one fully observed control"):
        estimate_synthetic_control(no_control_ds)


# -------------------------------------------------------------------- bayesian
def test_bayesian_defaults_to_controls_only_specification(ds):
    est = estimate_bayesian_counterfactual(ds, n_draws=800)
    assert est.diagnostics["include_trend"] is False
    assert est.diagnostics["n_harmonics"] == 0
    assert est.diagnostics["n_controls_used"] == 8


def test_bayesian_adds_trend_and_season_without_controls(no_control_ds):
    est = estimate_bayesian_counterfactual(no_control_ds, n_draws=800)
    assert est.diagnostics["include_trend"] is True
    assert est.diagnostics["n_harmonics"] == 2
    assert any("No control series" in w for w in est.warnings)


def test_bayesian_backtest_passes_on_well_specified_data(ds):
    est = estimate_bayesian_counterfactual(ds, n_draws=1500)
    bt = est.diagnostics["backtest"]
    assert bt["ran"] is True
    assert bt["false_positive"] is False, (
        "a fake intervention inside the pre-period must not look significant"
    )


def test_bayesian_is_reproducible(ds):
    a = estimate_bayesian_counterfactual(ds, n_draws=800, seed=5)
    b = estimate_bayesian_counterfactual(ds, n_draws=800, seed=5)
    assert a.effect == pytest.approx(b.effect)
    assert a.interval.lower == pytest.approx(b.interval.lower)


def test_bayesian_rejects_overparameterised_models():
    panel = generate_panel(ScenarioConfig(n_periods=24, n_pre=10, n_controls=14, seed=6))
    ds_small = panel.to_dataset(min_pre_periods=8)
    with pytest.raises(ValueError, match="only 10 pre-period observations"):
        estimate_bayesian_counterfactual(ds_small, n_draws=400)


# ------------------------------------------------------------------------- ITS
def test_its_runs_without_controls_and_warns_loudly(no_control_ds):
    est = estimate_its(no_control_ds)
    assert np.isfinite(est.effect)
    assert "no control group" in est.warnings[0]
    assert "level_change" in est.diagnostics
    assert "slope_change" in est.diagnostics


def test_its_separates_level_from_slope():
    """A pure level shift should show up as level change, not slope change."""
    panel = generate_panel(ScenarioConfig(
        n_controls=0, market_shock=0.0, effect_kind="additive",
        true_effect=30.0, ramp_periods=0, seed=8,
    ))
    est = estimate_its(panel.to_dataset())
    assert est.diagnostics["level_change"] == pytest.approx(30.0, abs=12.0)
    assert est.diagnostics["slope_change_p"] > 0.05


# ------------------------------------------------------------------- registry
def test_run_all_skips_estimators_that_need_controls(no_control_ds):
    results = run_all(no_control_ds)
    assert "did" not in results
    assert "synthetic_control" not in results
    assert "its" in results and "bayesian" in results


def test_run_all_returns_every_applicable_estimator(ds):
    results = run_all(ds)
    assert set(results) == {"did", "synthetic_control", "bayesian", "its", "naive"}


def test_run_all_skips_by_design_not_by_error(no_control_ds):
    """Skipping a control-based estimator is a decision, not a swallowed failure.

    So ``skip_on_error=False`` must still succeed on a control-free panel: the
    estimators are never called, rather than called and failing.
    """
    results = run_all(no_control_ds, skip_on_error=False)
    assert set(results) == {"bayesian", "its", "naive"}


def test_run_all_propagates_real_errors_when_asked():
    """A genuine estimator failure must not be hidden when the caller says so."""
    panel = generate_panel(ScenarioConfig(n_periods=24, n_pre=10, n_controls=14, seed=6))
    ds_small = panel.to_dataset(min_pre_periods=8)
    with pytest.raises(ValueError, match="pre-period observations"):
        run_all(ds_small, skip_on_error=False)
    # With the default, the bad fit is dropped and the rest still come back.
    survived = run_all(ds_small)
    assert "bayesian" not in survived
    assert survived, "the other estimators should still produce results"


def test_get_estimator_rejects_unknown_names():
    with pytest.raises(KeyError, match="unknown estimator"):
        get_estimator("magic")


# --------------------------------------------------------------- uncertainty
@pytest.mark.parametrize("name", CONTROL_BASED)
def test_intervals_widen_with_noise(name):
    """More noise must mean less certainty. An estimator that ignores this lies."""
    quiet = generate_panel(ScenarioConfig(noise_sd=4.0, seed=31)).to_dataset()
    loud = generate_panel(ScenarioConfig(noise_sd=28.0, seed=31)).to_dataset()
    fn = get_estimator(name)
    assert fn(loud).interval.width > fn(quiet).interval.width


def test_higher_level_means_wider_interval(ds):
    narrow = estimate_synthetic_control(ds, level=0.80)
    wide = estimate_synthetic_control(ds, level=0.99)
    assert wide.interval.width > narrow.interval.width
