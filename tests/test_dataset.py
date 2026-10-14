"""Schema validation must fail loudly, with a message that says what to fix."""

from __future__ import annotations

import pandas as pd
import pytest

from impactlens.dataset import PanelDataset, SchemaError
from impactlens.simulate import ScenarioConfig, generate_panel


def _frame(**kwargs) -> pd.DataFrame:
    return generate_panel(ScenarioConfig(seed=5, **kwargs)).frame


def test_valid_panel_loads(ds):
    assert ds.treated_unit == "flagship_clinic"
    assert ds.n_controls == 8
    assert ds.n_pre == 78
    assert ds.n_post == 26
    assert ds.has_controls
    assert len(ds.dates) == 104


def test_missing_column_names_the_column():
    frame = _frame().rename(columns={"y": "value"})
    with pytest.raises(SchemaError, match="missing required column"):
        PanelDataset.from_frame(frame, treated_unit="flagship_clinic",
                                intervention_date="2025-06-30")


def test_unknown_treated_unit_lists_available_units():
    with pytest.raises(SchemaError, match="not present in data"):
        PanelDataset.from_frame(_frame(), treated_unit="no_such_clinic",
                                intervention_date="2025-06-30")


def test_duplicate_unit_date_rows_rejected():
    frame = _frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicated"):
        PanelDataset.from_frame(frame, treated_unit="flagship_clinic",
                                intervention_date="2025-06-30")


def test_intervention_before_data_start_rejected():
    with pytest.raises(SchemaError, match="no pre-period"):
        PanelDataset.from_frame(_frame(), treated_unit="flagship_clinic",
                                intervention_date="2020-01-01")


def test_intervention_after_data_end_rejected():
    with pytest.raises(SchemaError, match="no post-period"):
        PanelDataset.from_frame(_frame(), treated_unit="flagship_clinic",
                                intervention_date="2099-01-01")


def test_too_few_pre_periods_rejected():
    frame = _frame()
    early = frame[frame["date"] <= frame["date"].min() + pd.Timedelta(weeks=12)]
    with pytest.raises(SchemaError, match="pre-intervention period"):
        PanelDataset.from_frame(
            early,
            treated_unit="flagship_clinic",
            intervention_date=early["date"].min() + pd.Timedelta(weeks=3),
            min_pre_periods=8,
        )


def test_unparseable_date_rejected():
    frame = _frame()
    frame.loc[0, "date"] = "not-a-date"
    with pytest.raises(SchemaError, match="unparseable"):
        PanelDataset.from_frame(frame, treated_unit="flagship_clinic",
                                intervention_date="2025-06-30")


def test_missing_outcomes_dropped_and_reported():
    frame = _frame()
    frame.loc[0:4, "y"] = None
    ds = PanelDataset.from_frame(frame, treated_unit="flagship_clinic",
                                 intervention_date="2025-06-30")
    assert ds.quality_report()["dropped_rows"] == 5
    assert any("dropped" in w for w in ds.warnings())


def test_short_pre_period_triggers_preliminary_warning():
    panel = generate_panel(ScenarioConfig(n_periods=24, n_pre=10, seed=2))
    ds = panel.to_dataset(min_pre_periods=8)
    assert any("preliminary" in w for w in ds.warnings())


def test_no_controls_warns_and_reports_its_only(no_control_ds):
    assert not no_control_ds.has_controls
    assert any("No control units" in w for w in no_control_ds.warnings())


def test_wide_and_series_views_agree(ds):
    wide = ds.wide()
    assert wide.shape == (104, 9)
    pd.testing.assert_series_equal(
        ds.treated_series.rename(None),
        wide[ds.treated_unit].rename(None),
        check_freq=False,
    )


def test_control_matrix_excludes_units_with_gaps():
    frame = _frame()
    frame = frame[~((frame["unit"] == "control_01") & (frame.index % 7 == 0))]
    ds = PanelDataset.from_frame(frame, treated_unit="flagship_clinic",
                                 intervention_date="2025-06-30")
    assert "control_01" not in ds.control_matrix().columns
    assert any("Unbalanced panel" in w for w in ds.warnings())


def test_frequency_inferred_for_weekly_data(ds):
    assert ds.frequency is not None
    assert "W" in ds.frequency
