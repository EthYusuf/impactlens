"""Reporting must never overstate the finding.

The riskiest thing this package does is put a sentence in front of a
decision-maker. These tests pin the wording down: an interval that crosses zero
is never described as an effect, and the assumptions always travel with the
number.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from impactlens.estimators import (
    estimate_naive_prepost,
    estimate_synthetic_control,
    event_study,
    run_all,
)
from impactlens.reporting import narrative
from impactlens.reporting.plots import (
    plot_counterfactual,
    plot_cumulative_effect,
    plot_event_study,
    plot_method_comparison,
    plot_panel_overview,
    plot_placebo_distribution,
    plot_pointwise_effect,
    plot_recovery_summary,
    plot_weights,
)
from impactlens.reporting.report import build_pdf_report
from impactlens.simulate import ScenarioConfig, generate_panel
from impactlens.types import ImpactEstimate, Interval, relative_from_counterfactual


@pytest.fixture(scope="module")
def est(ds):
    return estimate_synthetic_control(ds)


# ------------------------------------------------------------------ narrative
def test_headline_reports_an_effect_when_the_interval_excludes_zero(est):
    assert est.is_significant
    text = narrative.headline(est, outcome_name="appointments")
    assert "appointments" in text
    assert "excludes zero" in text
    assert "increased" in text


def test_headline_refuses_to_claim_a_direction_when_the_interval_crosses_zero(null_ds):
    est_null = estimate_synthetic_control(null_ds)
    if est_null.is_significant:
        pytest.skip("this seed produced a false positive; calibration is tested elsewhere")
    text = narrative.headline(est_null, outcome_name="appointments")
    assert "No detectable effect" in text
    assert "includes zero" in text
    # The crucial property: no directional verb sneaks in.
    assert "increased" not in text and "decreased" not in text


def test_executive_summary_always_carries_caveats_or_says_there_were_none(est, ds):
    paras = narrative.executive_summary(est, ds, outcome_name="appointments")
    assert len(paras) >= 3
    joined = " ".join(paras)
    assert "Method." in joined
    assert "Data." in joined
    assert "Caveats." in joined or "No data-quality" in joined


def test_assumptions_are_never_empty(est):
    assert len(narrative.assumption_block(est)) >= 3


def test_comparison_note_calls_out_the_naive_inflation(ds):
    results = run_all(ds)
    note = narrative.comparison_note(results)
    assert note
    assert "naive before/after" in note


def test_comparison_note_counts_only_control_based_methods(ds):
    """ITS has no control group, so it must not be described as control-based.

    Letting it vote on whether the confound was handled would be exactly the
    overstatement this project exists to prevent.
    """
    results = run_all(ds)
    assert "its" in results, "the fixture should produce an ITS estimate"
    n_controlled = sum(
        1 for e in results.values() if e.diagnostics.get("n_controls_used", 0) > 0
    )
    assert n_controlled == 3, "did, synthetic_control and bayesian use controls"
    note = narrative.comparison_note(results)
    assert "All 3 control-based methods" in note
    assert "All 4 control-based" not in note
    # And ITS is named as a no-control method sitting above the consensus.
    assert "Interrupted Time Series" in note
    assert "no control group" in note


def test_comparison_note_spread_ignores_the_no_control_outlier(ds):
    """The agreement claim must be about the control-based spread alone."""
    results = run_all(ds)
    controlled = [
        e.effect for e in results.values() if e.diagnostics.get("n_controls_used", 0) > 0
    ]
    all_effects = [e.effect for e in results.values()]
    controlled_spread = max(controlled) - min(controlled)
    overall_spread = max(all_effects) - min(all_effects)
    assert overall_spread > 3 * controlled_spread, (
        "the no-control estimators should be far outside the control-based cluster"
    )
    note = narrative.comparison_note(results)
    assert f"({controlled_spread:,.1f})" in note


def test_comparison_note_flags_disagreement_between_methods(est):
    """Build two control-based estimates pointing opposite ways."""
    up = est
    down = ImpactEstimate(
        method="Fabricated downward estimate",
        effect=-50.0,
        interval=Interval(-80.0, -20.0),
        n_pre=est.n_pre,
        n_post=est.n_post,
        diagnostics={"n_controls_used": 5},
    )
    note = narrative.comparison_note({"a": up, "b": down})
    assert "disagree" in note
    assert "red flag" in note


def test_comparison_note_is_silent_with_a_single_method(est):
    assert narrative.comparison_note({"synthetic_control": est}) == ""


def test_comparison_note_is_silent_without_control_based_methods(no_control_ds):
    """Nothing to compare when no estimate saw a control unit."""
    results = run_all(no_control_ds)
    assert all(e.diagnostics.get("n_controls_used", 0) == 0 for e in results.values())
    assert narrative.comparison_note(results) == ""


# ---------------------------------------------------------------------- types
def test_interval_rejects_impossible_levels():
    with pytest.raises(ValueError, match="level must be in"):
        Interval(1.0, 2.0, level=1.5)


def test_interval_containment_and_width():
    iv = Interval(-1.0, 3.0)
    assert iv.contains(0.0) and iv.contains(3.0)
    assert not iv.contains(3.5)
    assert iv.width == pytest.approx(4.0)


def test_significance_follows_the_interval_not_the_point_estimate():
    crosses_zero = ImpactEstimate(
        method="x", effect=10.0, interval=Interval(-2.0, 22.0), n_pre=10, n_post=5
    )
    assert not crosses_zero.is_significant
    assert crosses_zero.direction == "inconclusive"

    clears_zero = ImpactEstimate(
        method="x", effect=10.0, interval=Interval(2.0, 18.0), n_pre=10, n_post=5
    )
    assert clears_zero.is_significant
    assert clears_zero.direction == "increase"


def test_relative_effect_refuses_a_non_positive_base():
    rel, iv = relative_from_counterfactual(5.0, Interval(1.0, 9.0), [0.0, -1.0, 1.0])
    assert rel is None and iv is None


def test_relative_effect_divides_by_the_counterfactual_mean():
    rel, iv = relative_from_counterfactual(10.0, Interval(5.0, 15.0), [100.0, 100.0])
    assert rel == pytest.approx(0.10)
    assert iv.lower == pytest.approx(0.05)


def test_summary_and_to_dict_round_trip(est):
    text = est.summary()
    assert est.method in text
    assert "Uncertainty interval" in text
    d = est.to_dict()
    assert d["effect"] == pytest.approx(est.effect)
    assert d["significant"] is est.is_significant


def test_pointwise_effect_is_observed_minus_counterfactual(est):
    gap = est.pointwise_effect
    assert gap is not None
    pd.testing.assert_series_equal(gap, est.observed - est.counterfactual)


# ---------------------------------------------------------------------- plots
@pytest.mark.parametrize(
    "fn",
    [plot_counterfactual, plot_pointwise_effect, plot_cumulative_effect],
)
def test_series_plots_render(est, ds, fn):
    fig = fn(est, intervention_date=ds.intervention_date)
    assert isinstance(fig, Figure)
    assert fig.axes
    plt.close(fig)


def test_placebo_and_weight_plots_render(est):
    for fig in (plot_placebo_distribution(est), plot_weights(est)):
        assert isinstance(fig, Figure)
        plt.close(fig)


def test_event_study_plot_renders_and_degrades_gracefully(ds, no_control_ds):
    fig = plot_event_study(event_study(ds))
    assert fig.axes
    plt.close(fig)
    # With no controls there is nothing to plot; it must not raise.
    fig2 = plot_event_study(event_study(no_control_ds))
    assert isinstance(fig2, Figure)
    plt.close(fig2)


def test_panel_overview_renders(ds):
    fig = plot_panel_overview(ds)
    assert fig.axes
    plt.close(fig)


def test_method_comparison_renders_with_and_without_truth(ds, panel):
    results = run_all(ds)
    for truth in (None, panel.true_average_effect):
        fig = plot_method_comparison(results, truth=truth)
        assert fig.axes
        plt.close(fig)
    fig = plot_method_comparison({})
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_recovery_summary_plot_renders():
    summary = pd.DataFrame(
        {
            "label": ["Synthetic Control", "Naive"],
            "coverage": [0.94, 0.02],
            "rmse": [2.5, 17.0],
        },
        index=["synthetic_control", "naive"],
    )
    fig = plot_recovery_summary(summary)
    assert len(fig.axes) == 2
    plt.close(fig)
    fig = plot_recovery_summary(pd.DataFrame())
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plots_reject_an_estimate_with_no_series():
    bare = ImpactEstimate(
        method="x", effect=1.0, interval=Interval(0.5, 1.5), n_pre=5, n_post=2
    )
    with pytest.raises(ValueError, match="no series"):
        plot_counterfactual(bare)


# ------------------------------------------------------------------------ pdf
def test_pdf_report_is_written_and_non_trivial(tmp_path, ds, est):
    out = build_pdf_report(
        tmp_path / "report.pdf",
        est,
        ds,
        project_name="Test clinic",
        outcome_name="appointments",
    )
    assert out.exists()
    assert out.stat().st_size > 30_000, "a multi-page report should not be tiny"
    assert out.read_bytes().startswith(b"%PDF")


def test_pdf_report_includes_comparison_and_validation_pages(tmp_path, ds):
    results = run_all(ds)
    summary = pd.DataFrame(
        {
            "label": ["Synthetic Control"],
            "n_runs": [10],
            "true_effect": [24.0],
            "mean_estimate": [24.3],
            "bias": [0.3],
            "rmse": [2.6],
            "coverage": [0.94],
            "power": [0.8],
        },
        index=["synthetic_control"],
    )
    slim = build_pdf_report(tmp_path / "slim.pdf", results["synthetic_control"], ds)
    full = build_pdf_report(
        tmp_path / "full.pdf",
        results["synthetic_control"],
        ds,
        all_estimates=results,
        validation_summary=summary,
    )
    assert full.stat().st_size > slim.stat().st_size, (
        "adding comparison and validation pages must add content"
    )


def test_pdf_report_works_without_controls(tmp_path, no_control_ds):
    est_its = estimate_naive_prepost(no_control_ds)
    out = build_pdf_report(tmp_path / "its.pdf", est_its, no_control_ds)
    assert out.exists() and out.stat().st_size > 20_000


def test_pdf_report_creates_missing_directories(tmp_path, ds, est):
    out = build_pdf_report(tmp_path / "nested" / "deeper" / "r.pdf", est, ds)
    assert out.exists()


def test_pdf_report_handles_a_null_result(tmp_path, null_ds):
    """A "no effect" report must still render every page."""
    panel = generate_panel(ScenarioConfig(true_effect=0.0, seed=11))
    assert panel.true_average_effect == 0.0
    est_null = estimate_synthetic_control(null_ds)
    out = build_pdf_report(tmp_path / "null.pdf", est_null, null_ds)
    assert out.exists() and out.stat().st_size > 20_000
