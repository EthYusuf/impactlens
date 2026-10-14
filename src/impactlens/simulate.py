"""Synthetic data with a known ground-truth effect.

This module exists so the tool can be audited. Every estimator is first pointed
at data where the true effect was *inserted by us*, which means the correct
answer is known and recovery error is measurable. Real client data never offers
that luxury.

The generator deliberately builds in the confound that motivates the whole
project: a market-wide seasonal surge that starts near the intervention date.
A naive before/after comparison attributes that surge to the intervention. A
control-based estimator should not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dataset import PanelDataset

__all__ = ["ScenarioConfig", "SyntheticPanel", "generate_panel", "naive_prepost_effect"]


@dataclass
class ScenarioConfig:
    """Knobs for the data generating process.

    Parameters
    ----------
    n_periods:
        Total number of periods (weeks by default).
    n_pre:
        Number of pre-intervention periods. Must be < ``n_periods``.
    n_controls:
        Number of untreated donor units.
    true_effect:
        The effect inserted after the intervention date. Interpreted as an
        absolute level shift when ``effect_kind="additive"``, or as a fraction
        of the counterfactual level when ``effect_kind="multiplicative"``.
    effect_kind:
        ``"additive"`` or ``"multiplicative"``.
    ramp_periods:
        Periods over which the effect ramps linearly from 0 to full size.
        0 means the effect switches on instantly.
    base_level:
        Mean outcome level before the intervention.
    trend:
        Per-period common drift.
    seasonal_amplitude:
        Amplitude of the shared annual seasonal cycle, in outcome units.
    seasonal_period:
        Length of the seasonal cycle in periods (52 for weekly data).
    market_shock:
        Size of a market-wide surge affecting *all* units, starting at
        ``shock_start``. This is the confound a naive pre/post comparison
        mistakes for the intervention effect.
    shock_start:
        Period index where the market shock begins. Defaults to the
        intervention date, which is the hardest and most realistic case.
    noise_sd:
        Standard deviation of idiosyncratic noise.
    unit_dispersion:
        Spread of unit-level fixed effects, as a fraction of ``base_level``.
        These are **additive** level shifts, so every unit still shares the same
        trend and seasonal amplitude. Parallel trends therefore holds exactly,
        which is what makes the default scenario a fair test of the estimators.
    proportional_heterogeneity:
        Spread of **multiplicative** unit scales. Non-zero values make each unit
        respond proportionally to the shared trend and season, so a high-volume
        unit grows faster in absolute terms than a low-volume one. That breaks
        parallel trends *in levels* even though no unit does anything unusual --
        a realistic misspecification, and the one the event-study diagnostic is
        designed to catch. Defaults to 0 so the baseline scenario is clean.
    heterogeneous_trends:
        When True, control units get their own small linear trends. Averaged over
        a reasonable donor pool these largely cancel, so this mostly adds noise
        rather than bias -- use ``treated_pretrend`` for a real violation.
    treated_pretrend:
        Extra per-period drift applied to the **treated unit only**, across the
        whole window. This is the classic difference-in-differences killer:
        selection on trend, where the unit that adopted the intervention was
        already pulling away from its peers beforehand. It is part of the
        counterfactual, not of the effect, so a correct estimator must not
        credit it to the intervention. Defaults to 0.
    seed:
        RNG seed.
    """

    n_periods: int = 104
    n_pre: int = 78
    n_controls: int = 8
    true_effect: float = 0.12
    effect_kind: str = "multiplicative"
    ramp_periods: int = 4
    base_level: float = 180.0
    trend: float = 0.25
    seasonal_amplitude: float = 22.0
    seasonal_period: int = 52
    market_shock: float = 26.0
    shock_start: int | None = None
    noise_sd: float = 9.0
    unit_dispersion: float = 0.22
    proportional_heterogeneity: float = 0.0
    heterogeneous_trends: bool = False
    treated_pretrend: float = 0.0
    start_date: str = "2024-01-01"
    freq: str = "W-MON"
    treated_name: str = "flagship_clinic"
    seed: int = 7

    def __post_init__(self) -> None:
        if self.effect_kind not in ("additive", "multiplicative"):
            raise ValueError("effect_kind must be 'additive' or 'multiplicative'")
        if not 0 < self.n_pre < self.n_periods:
            raise ValueError("n_pre must satisfy 0 < n_pre < n_periods")
        if self.n_controls < 0:
            raise ValueError("n_controls cannot be negative")
        if self.ramp_periods < 0:
            raise ValueError("ramp_periods cannot be negative")

    @property
    def n_post(self) -> int:
        return self.n_periods - self.n_pre


@dataclass
class SyntheticPanel:
    """Generated panel plus the ground truth used to build it."""

    frame: pd.DataFrame
    config: ScenarioConfig
    intervention_date: pd.Timestamp
    treated_unit: str
    #: Counterfactual series for the treated unit: what it would have been.
    true_counterfactual: pd.Series
    #: Period-by-period true effect over the post window.
    true_effect_path: pd.Series
    meta: dict[str, float] = field(default_factory=dict)

    @property
    def true_average_effect(self) -> float:
        """Ground-truth average effect per post period, in outcome units."""
        return float(self.true_effect_path.mean())

    @property
    def true_cumulative_effect(self) -> float:
        return float(self.true_effect_path.sum())

    @property
    def true_relative_effect(self) -> float:
        """Ground-truth average effect as a fraction of the counterfactual."""
        post = self.true_counterfactual.loc[self.true_effect_path.index]
        return float(self.true_effect_path.mean() / post.mean())

    def to_dataset(self, **kwargs) -> PanelDataset:
        """Validate into a :class:`~impactlens.dataset.PanelDataset`."""
        return PanelDataset.from_frame(
            self.frame,
            treated_unit=self.treated_unit,
            intervention_date=self.intervention_date,
            **kwargs,
        )


def _ramped_effect(n_post: int, ramp: int) -> np.ndarray:
    """Multiplier in [0, 1] describing how the effect phases in."""
    if ramp <= 0:
        return np.ones(n_post)
    ramp = min(ramp, n_post)
    phase = np.linspace(1.0 / ramp, 1.0, ramp)
    return np.concatenate([phase, np.ones(n_post - ramp)])


def generate_panel(config: ScenarioConfig | None = None, **overrides) -> SyntheticPanel:
    """Build a synthetic panel with a known effect.

    Notes
    -----
    With ``effect_kind="multiplicative"`` and a non-zero ``ramp_periods``, the
    realised average relative effect is slightly below ``true_effect`` because
    the first few post periods are still ramping. Always compare estimates
    against :attr:`SyntheticPanel.true_average_effect` rather than against the
    configured ``true_effect``.
    """
    cfg = config or ScenarioConfig()
    if overrides:
        cfg = ScenarioConfig(**{**cfg.__dict__, **overrides})

    rng = np.random.default_rng(cfg.seed)
    t = np.arange(cfg.n_periods)
    dates = pd.date_range(cfg.start_date, periods=cfg.n_periods, freq=cfg.freq)
    intervention_date = dates[cfg.n_pre]

    shock_start = cfg.n_pre if cfg.shock_start is None else cfg.shock_start
    market = np.where(t >= shock_start, cfg.market_shock, 0.0)
    # Smooth the shock onset so it looks like a season turning, not a step edge.
    if cfg.market_shock:
        onset = np.clip((t - shock_start) / 6.0, 0.0, 1.0)
        market = cfg.market_shock * onset

    seasonal = cfg.seasonal_amplitude * np.sin(2 * np.pi * t / cfg.seasonal_period)
    common = cfg.base_level + cfg.trend * t + seasonal + market

    unit_names = [cfg.treated_name] + [f"control_{i:02d}" for i in range(1, cfg.n_controls + 1)]
    records: list[pd.DataFrame] = []
    true_counterfactual: pd.Series | None = None

    for i, unit in enumerate(unit_names):
        # Additive unit fixed effect: shifts the level, leaves the trend and the
        # seasonal amplitude identical across units, so parallel trends holds.
        level_shift = (
            0.0 if i == 0
            else float(rng.normal(0.0, cfg.unit_dispersion * cfg.base_level))
        )
        # Multiplicative scale: only non-zero when the caller asks for it. This
        # is what breaks parallel trends in levels.
        scale = (
            1.0 if i == 0 or cfg.proportional_heterogeneity <= 0
            else float(np.exp(rng.normal(0.0, cfg.proportional_heterogeneity)))
        )
        unit_trend = 0.0
        if cfg.heterogeneous_trends and i > 0:
            unit_trend = float(rng.normal(0.0, 0.06))
        if i == 0:
            # Pre-existing divergence of the treated unit. Deliberately applied
            # to the whole window, so it is part of the counterfactual rather
            # than part of the effect.
            unit_trend += cfg.treated_pretrend
        baseline = scale * common + level_shift + unit_trend * t
        noise = rng.normal(0.0, cfg.noise_sd * max(scale, 0.4), size=cfg.n_periods)
        y = baseline + noise

        if i == 0:
            counterfactual = y.copy()
            effect_path = np.zeros(cfg.n_periods)
            ramp = _ramped_effect(cfg.n_post, cfg.ramp_periods)
            if cfg.effect_kind == "multiplicative":
                effect_path[cfg.n_pre :] = cfg.true_effect * ramp * counterfactual[cfg.n_pre :]
            else:
                effect_path[cfg.n_pre :] = cfg.true_effect * ramp
            y = counterfactual + effect_path
            true_counterfactual = pd.Series(counterfactual, index=dates, name="counterfactual")
            true_effect_path = pd.Series(
                effect_path[cfg.n_pre :], index=dates[cfg.n_pre :], name="true_effect"
            )

        records.append(pd.DataFrame({"unit": unit, "date": dates, "y": y}))

    frame = pd.concat(records, ignore_index=True)
    assert true_counterfactual is not None  # treated unit is always index 0

    return SyntheticPanel(
        frame=frame,
        config=cfg,
        intervention_date=intervention_date,
        treated_unit=cfg.treated_name,
        true_counterfactual=true_counterfactual,
        true_effect_path=true_effect_path,
        meta={
            "market_shock": cfg.market_shock,
            "shock_start_index": float(shock_start),
        },
    )


def naive_prepost_effect(panel: SyntheticPanel) -> float:
    """The number a dashboard would report: post mean minus pre mean.

    Kept in the library because the gap between this and the ground truth is the
    clearest possible demonstration of why the project exists.
    """
    treated = panel.frame[panel.frame["unit"] == panel.treated_unit]
    treated = treated.set_index("date")["y"].sort_index()
    pre = treated.loc[treated.index < panel.intervention_date].mean()
    post = treated.loc[treated.index >= panel.intervention_date].mean()
    return float(post - pre)
