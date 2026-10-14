"""Shared result types.

Every estimator in :mod:`impactlens.estimators` returns an :class:`ImpactEstimate`
so that the reporting layer, the dashboard and the validation harness can treat
them interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Interval:
    """An uncertainty interval.

    ``kind`` records what the interval actually means, because a frequentist
    confidence interval and a Bayesian credible interval are not the same claim
    and the report should not pretend otherwise.
    """

    lower: float
    upper: float
    level: float = 0.95
    kind: str = "confidence"  # "confidence" | "credible" | "placebo"

    def __post_init__(self) -> None:
        if not 0 < self.level < 1:
            raise ValueError(f"level must be in (0, 1), got {self.level}")

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)

    def contains(self, value: float) -> bool:
        return bool(self.lower <= value <= self.upper)

    def __str__(self) -> str:
        pct = int(round(self.level * 100))
        return f"[{self.lower:,.3g}, {self.upper:,.3g}] ({pct}% {self.kind})"


@dataclass
class ImpactEstimate:
    """The answer to "did this intervention work, and how sure are we?".

    Attributes
    ----------
    method:
        Human-readable estimator name, e.g. ``"Difference-in-Differences"``.
    effect:
        Average effect on the treated unit per post-intervention period, in the
        units of the outcome variable.
    interval:
        Uncertainty interval around ``effect``.
    relative_effect:
        ``effect`` divided by the mean counterfactual level, i.e. the lift
        expressed as a fraction. ``None`` when no counterfactual is defined.
    relative_interval:
        Uncertainty interval for ``relative_effect``.
    cumulative_effect:
        Total effect summed over all post-intervention periods.
    p_value:
        Two-sided p-value for the null hypothesis of zero effect. ``None`` for
        estimators where no p-value is defined.
    counterfactual:
        Predicted outcome had the intervention not happened, indexed by date.
    observed:
        Realised outcome for the treated unit, indexed by date.
    counterfactual_lower / counterfactual_upper:
        Pointwise interval bounds for the counterfactual series.
    n_pre / n_post:
        Number of pre- and post-intervention periods used.
    assumptions:
        Short statements the estimate relies on. Shown verbatim in the report.
    warnings:
        Conditions detected in the data that weaken the estimate.
    diagnostics:
        Estimator-specific extras (weights, fit statistics, placebo draws...).
    """

    method: str
    effect: float
    interval: Interval
    n_pre: int
    n_post: int
    relative_effect: float | None = None
    relative_interval: Interval | None = None
    cumulative_effect: float | None = None
    p_value: float | None = None
    observed: pd.Series | None = None
    counterfactual: pd.Series | None = None
    counterfactual_lower: pd.Series | None = None
    counterfactual_upper: pd.Series | None = None
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_significant(self) -> bool:
        """True when the uncertainty interval excludes zero."""
        return not self.interval.contains(0.0)

    @property
    def direction(self) -> str:
        if not self.is_significant:
            return "inconclusive"
        return "increase" if self.effect > 0 else "decrease"

    @property
    def pointwise_effect(self) -> pd.Series | None:
        """Observed minus counterfactual, period by period."""
        if self.observed is None or self.counterfactual is None:
            return None
        return self.observed - self.counterfactual

    def to_dict(self) -> dict[str, Any]:
        """Flat summary suitable for a results table or JSON export."""
        out: dict[str, Any] = {
            "method": self.method,
            "effect": self.effect,
            "ci_lower": self.interval.lower,
            "ci_upper": self.interval.upper,
            "ci_level": self.interval.level,
            "ci_kind": self.interval.kind,
            "relative_effect": self.relative_effect,
            "cumulative_effect": self.cumulative_effect,
            "p_value": self.p_value,
            "n_pre": self.n_pre,
            "n_post": self.n_post,
            "significant": self.is_significant,
            "n_warnings": len(self.warnings),
        }
        if self.relative_interval is not None:
            out["relative_ci_lower"] = self.relative_interval.lower
            out["relative_ci_upper"] = self.relative_interval.upper
        return out

    def summary(self) -> str:
        """One-paragraph plain-text summary."""
        lines = [f"{self.method}", "-" * len(self.method)]
        lines.append(f"Average effect per period : {self.effect:,.4g}")
        lines.append(f"Uncertainty interval      : {self.interval}")
        if self.relative_effect is not None:
            rel = f"{self.relative_effect * 100:+.1f}%"
            if self.relative_interval is not None:
                rel += (
                    f"  [{self.relative_interval.lower * 100:+.1f}%, "
                    f"{self.relative_interval.upper * 100:+.1f}%]"
                )
            lines.append(f"Relative lift             : {rel}")
        if self.cumulative_effect is not None:
            lines.append(f"Cumulative effect         : {self.cumulative_effect:,.4g}")
        if self.p_value is not None:
            lines.append(f"p-value                   : {self.p_value:.4f}")
        lines.append(f"Periods (pre / post)      : {self.n_pre} / {self.n_post}")
        lines.append(f"Verdict                   : {self.direction}")
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  ! {w}" for w in self.warnings)
        return "\n".join(lines)


def relative_from_counterfactual(
    effect: float,
    interval: Interval,
    counterfactual: pd.Series | np.ndarray,
) -> tuple[float | None, Interval | None]:
    """Convert an absolute effect into a relative one using the counterfactual base.

    The base is the mean counterfactual level over the post period. When that
    base is at or below zero a percentage lift is meaningless, so ``(None, None)``
    is returned rather than a misleading number.
    """
    base = float(np.mean(np.asarray(counterfactual, dtype=float)))
    if not np.isfinite(base) or base <= 0:
        return None, None
    rel_interval = Interval(
        lower=interval.lower / base,
        upper=interval.upper / base,
        level=interval.level,
        kind=interval.kind,
    )
    return effect / base, rel_interval
