"""Synthetic control method.

Instead of assuming the treated unit moves parallel to the control average, the
synthetic control method *builds* a weighted combination of control units that
tracks the treated unit during the pre-period, then extrapolates that
combination forward as the counterfactual.

Weights are constrained to be non-negative and sum to one. That restriction is
the point: it prevents extrapolation outside the range of the donors and makes
the comparison unit interpretable ("62% control_03 + 38% control_07").

Inference follows Abadie, Diamond & Hainmueller (2010): re-run the whole
procedure pretending each control unit was treated, compare the treated unit's
post/pre fit-deterioration ratio against that placebo distribution, and read off
a rank-based p-value. With few donors this p-value is coarse by construction --
with 8 donors the smallest attainable p-value is 1/9 -- and the estimator says
so rather than inventing precision it does not have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..dataset import PanelDataset
from ..types import ImpactEstimate, Interval, relative_from_counterfactual

__all__ = ["estimate_synthetic_control", "fit_weights"]

_ASSUMPTIONS = [
    "Good pre-period fit: the weighted donors must track the treated unit closely "
    "before the intervention. A poor pre-fit makes the counterfactual worthless.",
    "No interference: donor units were not themselves affected by the intervention.",
    "Convex hull: the treated unit's behaviour must be reachable by a weighted "
    "average of the donors. An outlier treated unit cannot be matched.",
    "Stable relationship: the pre-period relationship between treated unit and "
    "donors continues to hold through the post period.",
]


def fit_weights(
    treated_pre: np.ndarray,
    donors_pre: np.ndarray,
    *,
    ridge: float = 0.0,
    max_iter: int = 400,
) -> np.ndarray:
    """Solve for simplex-constrained donor weights minimising pre-period MSE.

    Parameters
    ----------
    treated_pre:
        Pre-period outcomes of the treated unit, shape ``(T_pre,)``.
    donors_pre:
        Pre-period outcomes of the donors, shape ``(T_pre, J)``.
    ridge:
        Optional penalty on the squared L2 norm of the weights. A small value
        breaks ties toward diversified weights when donors are collinear.

    Returns
    -------
    Weights of shape ``(J,)``, non-negative and summing to one.
    """
    treated_pre = np.asarray(treated_pre, dtype=float).ravel()
    donors_pre = np.asarray(donors_pre, dtype=float)
    if donors_pre.ndim == 1:
        donors_pre = donors_pre.reshape(-1, 1)
    n_donors = donors_pre.shape[1]
    if n_donors == 1:
        return np.ones(1)

    def loss(w: np.ndarray) -> float:
        resid = treated_pre - donors_pre @ w
        return float(resid @ resid / len(resid) + ridge * (w @ w))

    def grad(w: np.ndarray) -> np.ndarray:
        resid = treated_pre - donors_pre @ w
        return -2.0 * donors_pre.T @ resid / len(resid) + 2.0 * ridge * w

    x0 = np.full(n_donors, 1.0 / n_donors)
    result = minimize(
        loss,
        x0,
        jac=grad,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_donors,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"maxiter": max_iter, "ftol": 1e-10},
    )
    w = np.clip(result.x, 0.0, None)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:  # pragma: no cover - degenerate donors
        return x0
    return w / total


def _rmspe(resid: np.ndarray) -> float:
    resid = np.asarray(resid, dtype=float)
    if resid.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(resid**2)))


def _fit_one(
    target: pd.Series,
    donors: pd.DataFrame,
    pre_mask: np.ndarray,
    *,
    ridge: float,
) -> dict[str, object]:
    """Fit a synthetic control for one target series against one donor pool."""
    y = target.to_numpy(dtype=float)
    D = donors.to_numpy(dtype=float)
    w = fit_weights(y[pre_mask], D[pre_mask], ridge=ridge)
    synth = D @ w
    resid = y - synth
    pre_rmspe = _rmspe(resid[pre_mask])
    post_rmspe = _rmspe(resid[~pre_mask])
    ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else float("inf")
    return {
        "weights": w,
        "synthetic": synth,
        "residual": resid,
        "pre_rmspe": pre_rmspe,
        "post_rmspe": post_rmspe,
        "ratio": ratio,
        "effect": float(np.mean(resid[~pre_mask])),
    }


def estimate_synthetic_control(
    ds: PanelDataset,
    *,
    level: float = 0.95,
    ridge: float = 1e-6,
    run_placebos: bool = True,
) -> ImpactEstimate:
    """Estimate the treatment effect by synthetic control.

    The point estimate is the mean post-period gap between the treated unit and
    its synthetic counterpart. The interval is derived from the in-space placebo
    distribution of placebo gaps, which is a permutation interval rather than a
    standard-error interval -- it is labelled ``kind="placebo"`` for that reason.
    """
    donors = ds.control_matrix()
    if donors.shape[1] == 0:
        raise ValueError(
            "synthetic control needs at least one fully observed control unit; "
            "use interrupted time series instead"
        )

    treated = ds.treated_series
    common_index = treated.index.intersection(donors.index)
    treated = treated.loc[common_index].astype(float)
    donors = donors.loc[common_index]
    pre_mask = np.asarray(common_index < ds.intervention_date, dtype=bool)
    n_pre = int(pre_mask.sum())
    n_post = int((~pre_mask).sum())

    fit = _fit_one(treated, donors, pre_mask, ridge=ridge)
    w = np.asarray(fit["weights"], dtype=float)
    synth = pd.Series(np.asarray(fit["synthetic"], dtype=float), index=common_index)
    synth.name = "counterfactual"
    effect = float(fit["effect"])
    pre_rmspe = float(fit["pre_rmspe"])

    warn_list: list[str] = list(ds.warnings())

    # ------------------------------------------------------------- placebos
    placebo_effects: list[float] = []
    placebo_ratios: list[float] = []
    placebo_units: list[str] = []
    p_value: float | None = None

    if run_placebos and donors.shape[1] >= 2:
        for unit in donors.columns:
            others = donors.drop(columns=[unit])
            if others.shape[1] == 0:
                continue
            pl = _fit_one(donors[unit], others, pre_mask, ridge=ridge)
            # Discard placebos whose own pre-fit is hopeless: their post-period
            # gap reflects a failed match, not a shock, and would inflate the
            # reference distribution.
            if float(pl["pre_rmspe"]) > 5.0 * pre_rmspe:
                continue
            placebo_effects.append(float(pl["effect"]))
            placebo_ratios.append(float(pl["ratio"]))
            placebo_units.append(str(unit))

    if placebo_ratios:
        ratios = np.array(placebo_ratios + [float(fit["ratio"])])
        rank = float(np.sum(ratios >= float(fit["ratio"])))
        p_value = rank / len(ratios)

    if placebo_effects and len(placebo_effects) >= 4:
        arr = np.array(placebo_effects, dtype=float)
        alpha = 1.0 - level
        lo_q, hi_q = np.quantile(arr, [alpha / 2, 1 - alpha / 2])
        # Centre the placebo spread on the point estimate: under the null of no
        # effect, the treated gap should look like a typical placebo gap.
        interval = Interval(
            lower=float(effect - hi_q),
            upper=float(effect - lo_q),
            level=level,
            kind="placebo",
        )
        if len(placebo_effects) < 10:
            warn_list.append(
                f"Placebo interval built from only {len(placebo_effects)} donor placebos. "
                "The smallest p-value attainable with this donor pool is "
                f"{1 / (len(placebo_ratios) + 1):.3f}; finer significance claims are not "
                "supported by the data."
            )
    else:
        # Not enough donors for permutation inference. Fall back to the spread of
        # pre-period residuals, which assumes post-period noise resembles
        # pre-period noise -- weaker, and flagged as such.
        resid = np.asarray(fit["residual"], dtype=float)[pre_mask]
        se = float(np.std(resid, ddof=1) / np.sqrt(max(n_post, 1))) if len(resid) > 1 else np.nan
        from scipy import stats as _stats

        z = float(_stats.norm.ppf(0.5 + level / 2))
        interval = Interval(
            lower=effect - z * se,
            upper=effect + z * se,
            level=level,
            kind="confidence",
        )
        warn_list.append(
            "Too few donor units for placebo inference. The interval assumes post-period "
            "noise matches pre-period noise, which is optimistic. Add control units."
        )

    # ------------------------------------------------------- fit diagnostics
    treated_pre_sd = float(np.std(treated.to_numpy(dtype=float)[pre_mask], ddof=1))
    fit_quality = pre_rmspe / treated_pre_sd if treated_pre_sd > 0 else float("nan")
    if np.isfinite(fit_quality) and fit_quality > 0.5:
        warn_list.append(
            f"Weak pre-period fit: the synthetic control's pre-period error is "
            f"{fit_quality:.0%} of the treated unit's own variation. The counterfactual "
            "is not tracking the treated unit well, so the estimated effect is unreliable."
        )

    effective_donors = int(np.sum(w > 0.01))
    if effective_donors == 1 and donors.shape[1] > 1:
        warn_list.append(
            "The synthetic control collapsed onto a single donor. The result now hinges "
            "entirely on that one unit."
        )

    half = (interval.upper - interval.lower) / 2.0
    post_sel = common_index >= ds.intervention_date
    cf_lower = synth.copy()
    cf_upper = synth.copy()
    cf_lower.loc[post_sel] = synth.loc[post_sel] - half
    cf_upper.loc[post_sel] = synth.loc[post_sel] + half

    rel, rel_interval = relative_from_counterfactual(
        effect, interval, synth.loc[post_sel].to_numpy()
    )

    weights = pd.Series(w, index=donors.columns, name="weight").sort_values(ascending=False)

    return ImpactEstimate(
        method="Synthetic Control",
        effect=effect,
        interval=interval,
        relative_effect=rel,
        relative_interval=rel_interval,
        cumulative_effect=effect * n_post,
        p_value=p_value,
        n_pre=n_pre,
        n_post=n_post,
        observed=treated,
        counterfactual=synth,
        counterfactual_lower=cf_lower,
        counterfactual_upper=cf_upper,
        assumptions=list(_ASSUMPTIONS),
        warnings=warn_list,
        diagnostics={
            "weights": weights,
            "pre_rmspe": pre_rmspe,
            "post_rmspe": float(fit["post_rmspe"]),
            "rmspe_ratio": float(fit["ratio"]),
            "pre_fit_ratio": fit_quality,
            "n_donors": int(donors.shape[1]),
            "n_controls_used": int(donors.shape[1]),
            "n_effective_donors": effective_donors,
            "placebo_effects": pd.Series(placebo_effects, index=placebo_units, name="effect")
            if placebo_units
            else pd.Series(dtype=float),
            "placebo_ratios": pd.Series(placebo_ratios, index=placebo_units, name="ratio")
            if placebo_units
            else pd.Series(dtype=float),
            "inference": "in-space placebo permutation"
            if placebo_effects
            else "pre-period residual spread",
        },
    )
