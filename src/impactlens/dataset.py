"""Panel dataset wrapper and validation.

The estimators all need the same thing: a balanced panel of one treated unit and
one or more control units, split by an intervention date. :class:`PanelDataset`
does that validation once, loudly, so the estimators can assume clean input.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["PanelDataset", "SchemaError", "load_panel"]


class SchemaError(ValueError):
    """Raised when the input frame cannot be used for impact estimation."""


@dataclass
class PanelDataset:
    """A validated panel ready for causal estimation.

    Parameters
    ----------
    frame:
        Long-format data with one row per (unit, date).
    unit_col, date_col, outcome_col:
        Column names.
    treated_unit:
        The unit that received the intervention.
    intervention_date:
        First period in which the intervention is active. This period counts as
        post-intervention.
    """

    frame: pd.DataFrame
    unit_col: str
    date_col: str
    outcome_col: str
    treated_unit: str
    intervention_date: pd.Timestamp

    # ---------------------------------------------------------------- builders
    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        treated_unit: str,
        intervention_date: str | pd.Timestamp,
        unit_col: str = "unit",
        date_col: str = "date",
        outcome_col: str = "y",
        min_pre_periods: int = 8,
        min_post_periods: int = 2,
    ) -> PanelDataset:
        missing = [c for c in (unit_col, date_col, outcome_col) if c not in frame.columns]
        if missing:
            raise SchemaError(
                f"missing required column(s): {', '.join(missing)}. "
                f"Found columns: {', '.join(map(str, frame.columns))}"
            )

        df = frame.loc[:, [unit_col, date_col, outcome_col]].copy()
        df[unit_col] = df[unit_col].astype(str)
        # Coercion is deliberate: unparseable dates are counted and reported as a
        # SchemaError below, which is a better message than pandas' warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")

        bad_dates = int(df[date_col].isna().sum())
        if bad_dates:
            raise SchemaError(f"{bad_dates} row(s) have an unparseable {date_col!r} value")

        n_before = len(df)
        df = df.dropna(subset=[outcome_col])
        dropped = n_before - len(df)

        if df.empty:
            raise SchemaError("no usable rows after dropping missing outcomes")

        treated_unit = str(treated_unit)
        units = sorted(df[unit_col].unique())
        if treated_unit not in units:
            raise SchemaError(
                f"treated unit {treated_unit!r} not present in data. Available: {', '.join(units)}"
            )

        dupes = df.duplicated(subset=[unit_col, date_col]).sum()
        if dupes:
            raise SchemaError(
                f"{dupes} duplicated (unit, date) row(s); aggregate the data before loading"
            )

        intervention_date = pd.Timestamp(intervention_date)
        if pd.isna(intervention_date):
            raise SchemaError("intervention_date could not be parsed")

        df = df.sort_values([unit_col, date_col]).reset_index(drop=True)
        ds = cls(
            frame=df,
            unit_col=unit_col,
            date_col=date_col,
            outcome_col=outcome_col,
            treated_unit=treated_unit,
            intervention_date=intervention_date,
        )

        dates = ds.dates
        if intervention_date <= dates.min():
            raise SchemaError(
                f"intervention_date {intervention_date.date()} is at or before the first "
                f"observation ({dates.min().date()}); there is no pre-period to learn from"
            )
        if intervention_date > dates.max():
            raise SchemaError(
                f"intervention_date {intervention_date.date()} is after the last observation "
                f"({dates.max().date()}); there is no post-period to evaluate"
            )
        if ds.n_pre < min_pre_periods:
            raise SchemaError(
                f"only {ds.n_pre} pre-intervention period(s); at least {min_pre_periods} are "
                "needed to characterise the baseline"
            )
        if ds.n_post < min_post_periods:
            raise SchemaError(
                f"only {ds.n_post} post-intervention period(s); at least {min_post_periods} "
                "are needed to measure an effect"
            )

        if dropped:
            ds._dropped_rows = dropped  # noqa: SLF001 - informational only
        return ds

    # ------------------------------------------------------------- properties
    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(self.frame[self.date_col].unique()))

    @property
    def pre_dates(self) -> pd.DatetimeIndex:
        return self.dates[self.dates < self.intervention_date]

    @property
    def post_dates(self) -> pd.DatetimeIndex:
        return self.dates[self.dates >= self.intervention_date]

    @property
    def n_pre(self) -> int:
        return len(self.pre_dates)

    @property
    def n_post(self) -> int:
        return len(self.post_dates)

    @property
    def units(self) -> list[str]:
        return sorted(self.frame[self.unit_col].unique())

    @property
    def control_units(self) -> list[str]:
        return [u for u in self.units if u != self.treated_unit]

    @property
    def n_controls(self) -> int:
        return len(self.control_units)

    @property
    def has_controls(self) -> bool:
        return self.n_controls > 0

    @property
    def frequency(self) -> str | None:
        """Best-guess pandas frequency string for the date index."""
        try:
            return pd.infer_freq(self.dates)
        except (ValueError, TypeError):
            return None

    # ----------------------------------------------------------------- views
    def wide(self) -> pd.DataFrame:
        """Pivot to a date x unit matrix."""
        return self.frame.pivot(
            index=self.date_col, columns=self.unit_col, values=self.outcome_col
        ).sort_index()

    def series(self, unit: str) -> pd.Series:
        """Outcome series for one unit, indexed by date."""
        sub = self.frame[self.frame[self.unit_col] == str(unit)]
        if sub.empty:
            raise KeyError(f"unit {unit!r} not in dataset")
        return sub.set_index(self.date_col)[self.outcome_col].sort_index()

    @property
    def treated_series(self) -> pd.Series:
        return self.series(self.treated_unit)

    def control_matrix(self) -> pd.DataFrame:
        """Date x control-unit matrix with fully observed columns only."""
        wide = self.wide()
        controls = [c for c in self.control_units if c in wide.columns]
        if not controls:
            return pd.DataFrame(index=wide.index)
        mat = wide.loc[:, controls]
        complete = mat.columns[mat.notna().all(axis=0)]
        return mat.loc[:, list(complete)]

    def post_mask(self, index: pd.DatetimeIndex | None = None) -> np.ndarray:
        idx = self.dates if index is None else pd.DatetimeIndex(index)
        return np.asarray(idx >= self.intervention_date, dtype=bool)

    # ------------------------------------------------------------ diagnostics
    def quality_report(self) -> dict[str, object]:
        """Data-quality facts the dashboard and report surface to the user."""
        wide = self.wide()
        gaps = int(wide.isna().sum().sum())
        unbalanced = [str(c) for c in wide.columns[wide.isna().any(axis=0)]]
        treated = self.treated_series
        pre = treated.loc[treated.index < self.intervention_date]
        report: dict[str, object] = {
            "n_units": len(self.units),
            "n_controls": self.n_controls,
            "n_periods": len(self.dates),
            "n_pre": self.n_pre,
            "n_post": self.n_post,
            "frequency": self.frequency or "irregular",
            "first_date": self.dates.min(),
            "last_date": self.dates.max(),
            "missing_cells": gaps,
            "unbalanced_units": unbalanced,
            "treated_pre_mean": float(pre.mean()),
            "treated_pre_std": float(pre.std(ddof=1)) if len(pre) > 1 else float("nan"),
            "dropped_rows": int(getattr(self, "_dropped_rows", 0)),
        }
        return report

    def warnings(self) -> list[str]:
        """Conditions that should temper how strongly results are read."""
        out: list[str] = []
        if self.n_pre < 12:
            out.append(
                f"Only {self.n_pre} pre-intervention periods. The baseline is thinly "
                "estimated; treat results as preliminary rather than conclusive."
            )
        if self.n_post < 4:
            out.append(
                f"Only {self.n_post} post-intervention periods. Short post windows cannot "
                "separate a durable effect from a one-off spike."
            )
        if not self.has_controls:
            out.append(
                "No control units supplied. Only interrupted time series is available, which "
                "assumes nothing else changed at the intervention date."
            )
        elif self.n_controls < 3:
            out.append(
                f"Only {self.n_controls} control unit(s). Synthetic control and placebo "
                "inference need more donors to be credible."
            )
        dropped = int(getattr(self, "_dropped_rows", 0))
        if dropped:
            out.append(f"{dropped} row(s) dropped for missing outcome values.")
        unbalanced = self.quality_report()["unbalanced_units"]
        if unbalanced:
            out.append(
                "Unbalanced panel: "
                + ", ".join(map(str, unbalanced[:5]))
                + (" ..." if len(unbalanced) > 5 else "")
                + ". Units with gaps are excluded from control-based estimators."
            )
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PanelDataset(treated={self.treated_unit!r}, controls={self.n_controls}, "
            f"pre={self.n_pre}, post={self.n_post}, "
            f"intervention={self.intervention_date.date()})"
        )


def load_panel(
    path: str,
    *,
    treated_unit: str,
    intervention_date: str | pd.Timestamp,
    unit_col: str = "unit",
    date_col: str = "date",
    outcome_col: str = "y",
) -> PanelDataset:
    """Read a CSV and validate it as a :class:`PanelDataset`."""
    frame = pd.read_csv(path)
    return PanelDataset.from_frame(
        frame,
        treated_unit=treated_unit,
        intervention_date=intervention_date,
        unit_col=unit_col,
        date_col=date_col,
        outcome_col=outcome_col,
    )
