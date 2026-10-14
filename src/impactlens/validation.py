"""Does the tool recover an effect it was never told about?

A causal estimate on real data cannot be checked, because the counterfactual is
unobservable. On synthetic data it can: we insert the effect ourselves, so the
right answer is known. This module runs that audit as a Monte Carlo study and
reports four things per estimator:

bias
    Mean estimate minus true effect. Should be near zero.
rmse
    Root mean squared error around the truth. Lower is better.
coverage
    Share of runs where the 95% interval contained the true effect. Should be
    close to 0.95. Materially below that means the intervals are too narrow and
    the tool is overconfident -- a worse failure than a noisy point estimate.
power
    Share of runs where the interval excluded zero. With a true effect this is
    statistical power; with ``true_effect=0`` it is the false-positive rate and
    should be close to 0.05.

Any estimator that fails these checks has no business producing a number for a
real decision, and the README reports the table rather than hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .estimators import ESTIMATOR_LABELS, get_estimator, requires_controls
from .simulate import ScenarioConfig, generate_panel

__all__ = ["RecoveryResult", "run_recovery_study", "false_positive_study"]


@dataclass
class RecoveryResult:
    """Outcome of a recovery study."""

    #: One row per (estimator, replication).
    runs: pd.DataFrame
    #: One row per estimator with bias, rmse, coverage, power.
    summary: pd.DataFrame
    config: ScenarioConfig
    n_replications: int
    level: float

    def verdict(self, *, coverage_tolerance: float = 0.10) -> pd.DataFrame:
        """Pass/fail judgement per estimator.

        An estimator passes when its interval coverage is within
        ``coverage_tolerance`` of the nominal level. Point-estimate bias is
        reported but not used as a pass criterion, because a small bias with
        honest intervals is far less dangerous than an unbiased estimate with
        intervals that are too narrow.
        """
        out = self.summary.copy()
        target = self.level
        out["coverage_gap"] = out["coverage"] - target
        out["intervals_honest"] = out["coverage"] >= target - coverage_tolerance
        out["verdict"] = np.where(out["intervals_honest"], "pass", "overconfident")
        return out[["bias", "rmse", "coverage", "coverage_gap", "power", "verdict"]]

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.summary.to_string(float_format=lambda v: f"{v:,.4g}")


def _run_one(
    seed: int,
    cfg: ScenarioConfig,
    estimator_names: list[str],
    level: float,
    estimator_kwargs: dict,
) -> list[dict]:
    """One replication: generate data, run each estimator, score against truth."""
    panel = generate_panel(cfg, seed=seed)
    truth = panel.true_average_effect
    try:
        ds = panel.to_dataset(min_pre_periods=4, min_post_periods=2)
    except Exception as exc:  # pragma: no cover - config guard
        return [{"seed": seed, "estimator": n, "error": str(exc)} for n in estimator_names]

    rows: list[dict] = []
    for name in estimator_names:
        if requires_controls(name) and not ds.has_controls:
            continue
        fn = get_estimator(name)
        kwargs = dict(estimator_kwargs.get(name, {}))
        try:
            est = fn(ds, level=level, **kwargs)
        except Exception as exc:
            rows.append({"seed": seed, "estimator": name, "error": str(exc)})
            continue
        rows.append(
            {
                "seed": seed,
                "estimator": name,
                "true_effect": truth,
                "estimate": est.effect,
                "error_abs": est.effect - truth,
                "ci_lower": est.interval.lower,
                "ci_upper": est.interval.upper,
                "ci_width": est.interval.width,
                "covered": est.interval.contains(truth),
                "detected": est.is_significant,
                "p_value": est.p_value,
                "error": None,
            }
        )
    return rows


def run_recovery_study(
    config: ScenarioConfig | None = None,
    *,
    n_replications: int = 100,
    estimators: list[str] | None = None,
    level: float = 0.95,
    base_seed: int = 1000,
    estimator_kwargs: dict | None = None,
    progress: bool = False,
) -> RecoveryResult:
    """Run the Monte Carlo recovery audit.

    Parameters
    ----------
    config:
        Scenario to simulate. The default scenario includes a market-wide shock
        starting at the intervention date, which is the case that separates
        control-based estimators from naive ones.
    n_replications:
        Number of simulated datasets. 100 gives coverage estimates with a
        standard error of roughly 2 percentage points; use 500+ for a headline
        claim.
    estimators:
        Short names to audit. Defaults to the control-based estimators plus the
        naive benchmark, so the contrast is visible.
    level:
        Nominal interval level, used as the coverage target.
    progress:
        Print a progress line every 10 replications. Off by default so library
        use stays quiet.
    """
    cfg = config or ScenarioConfig()
    names = estimators or ["did", "synthetic_control", "bayesian", "its", "naive"]
    # Lighter settings per replication. Across 100+ replications the Monte
    # Carlo error between replications dominates the simulation error inside
    # any single one, so full posterior draws and 999 bootstrap reps buy
    # nothing here -- they just make the audit too slow to run routinely.
    kwargs = {
        "bayesian": {"n_draws": 1500},
        "did": {"n_boot": 299},
    }
    if estimator_kwargs:
        for k, v in estimator_kwargs.items():
            kwargs.setdefault(k, {}).update(v)

    rows: list[dict] = []
    for i in range(n_replications):
        rows.extend(_run_one(base_seed + i, cfg, names, level, kwargs))
        if progress and (i + 1) % 10 == 0:
            print(f"  recovery study: {i + 1}/{n_replications} replications")

    runs = pd.DataFrame(rows)
    ok = runs[runs["error"].isna()] if "error" in runs.columns else runs

    records = []
    for name in names:
        sub = ok[ok["estimator"] == name] if not ok.empty else ok
        if sub.empty:
            continue
        records.append(
            {
                "estimator": name,
                "label": ESTIMATOR_LABELS.get(name, name),
                "n_runs": int(len(sub)),
                "true_effect": float(sub["true_effect"].mean()),
                "mean_estimate": float(sub["estimate"].mean()),
                "bias": float(sub["error_abs"].mean()),
                "mae": float(sub["error_abs"].abs().mean()),
                "rmse": float(np.sqrt((sub["error_abs"] ** 2).mean())),
                "coverage": float(sub["covered"].mean()),
                "power": float(sub["detected"].mean()),
                "mean_ci_width": float(sub["ci_width"].mean()),
            }
        )
    summary = pd.DataFrame(records).set_index("estimator") if records else pd.DataFrame()

    return RecoveryResult(
        runs=runs,
        summary=summary,
        config=cfg,
        n_replications=n_replications,
        level=level,
    )


def false_positive_study(
    config: ScenarioConfig | None = None,
    *,
    n_replications: int = 100,
    estimators: list[str] | None = None,
    level: float = 0.95,
    base_seed: int = 5000,
) -> RecoveryResult:
    """The null test: no effect inserted, but the market shock still happens.

    Every estimator should report "no detectable effect" roughly 95% of the time.
    An estimator whose ``power`` column here is far above 0.05 is manufacturing
    effects out of the seasonal confound, which is precisely the failure mode
    this tool exists to prevent.
    """
    cfg = config or ScenarioConfig()
    cfg = ScenarioConfig(**{**cfg.__dict__, "true_effect": 0.0})
    return run_recovery_study(
        cfg,
        n_replications=n_replications,
        estimators=estimators,
        level=level,
        base_seed=base_seed,
    )
