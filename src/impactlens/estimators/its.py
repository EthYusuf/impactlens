"""Interrupted time series (segmented regression).

The fallback when there is no control group at all. Fits

``y_t = b0 + b1*t + b2*post_t + b3*(t - T0)*post_t + seasonality + e_t``

so it separates a *level* jump at the intervention (``b2``) from a *slope*
change afterwards (``b3``). Reporting only the level jump when the slope also
changed is a classic way to overstate or understate an effect, so both are
returned.

This is the weakest estimator in the package and the module says so loudly: with
no control group, anything else that changed at the same date -- a competitor
closing, a holiday, a price change, a news cycle -- is absorbed into the
estimated effect. Standard errors use a Newey-West HAC covariance because
time-series residuals are autocorrelated.
"""

from __future__ import annotations

import warnings as _warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..dataset import PanelDataset
from ..types import ImpactEstimate, Interval, relative_from_counterfactual
from .bayesian import _infer_season_length, _seasonal_basis

__all__ = ["estimate_its"]

_ASSUMPTIONS = [
    "Nothing else changed at the intervention date. With no control group this "
    "cannot be tested, only argued -- it is the load-bearing assumption.",
    "The pre-intervention trend would have continued unchanged.",
    "Seasonality is captured by the fitted Fourier terms.",
    "The intervention date is known exactly and the effect is not anticipated.",
]


def estimate_its(
    ds: PanelDataset,
    *,
    level: float = 0.95,
    n_harmonics: int = 2,
    hac_lags: int | None = None,
) -> ImpactEstimate:
    """Estimate the effect by segmented regression on the treated series alone.

    Parameters
    ----------
    level:
        Confidence level.
    n_harmonics:
        Fourier harmonics for seasonality. 0 disables seasonal terms.
    hac_lags:
        Newey-West lag truncation. Defaults to ``floor(4*(n/100)^(2/9))``.
    """
    y_series = ds.treated_series.astype(float)
    index = y_series.index
    t = np.arange(len(index), dtype=float)
    post = np.asarray(index >= ds.intervention_date, dtype=float)
    t0 = float(np.flatnonzero(post)[0])

    cols = {
        "time": t,
        "post": post,
        "time_since": np.maximum(t - t0, 0.0) * post,
    }
    X = pd.DataFrame(cols, index=index)

    season_length = _infer_season_length(pd.DatetimeIndex(index))
    n_pre = int((~post.astype(bool)).sum())
    if n_harmonics > 0 and season_length and season_length >= 4 and n_pre > 4 * n_harmonics + 4:
        basis = _seasonal_basis(t, season_length, n_harmonics)
        for k in range(1, n_harmonics + 1):
            X[f"sin_{k}"] = basis[:, 2 * (k - 1)]
            X[f"cos_{k}"] = basis[:, 2 * (k - 1) + 1]

    X = sm.add_constant(X, has_constant="add")
    n = len(index)
    if hac_lags is None:
        hac_lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    hac_lags = max(1, min(hac_lags, max(1, n // 4)))

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        model = sm.OLS(y_series.to_numpy(), X.to_numpy(dtype=float)).fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_lags}, use_t=True
        )

    names = list(X.columns)
    i_post = names.index("post")
    i_slope = names.index("time_since")

    beta = np.asarray(model.params, dtype=float)
    # Counterfactual = prediction with the intervention terms switched off.
    X_cf = X.copy()
    X_cf["post"] = 0.0
    X_cf["time_since"] = 0.0
    cf = pd.Series(X_cf.to_numpy(dtype=float) @ beta, index=index, name="counterfactual")

    post_sel = post.astype(bool)
    n_post = int(post_sel.sum())
    # Average effect over the post window: level shift plus the accumulated
    # slope change. Taking only the level shift would understate a growing effect.
    periods_since = np.maximum(t - t0, 0.0)[post_sel]
    contrib = np.zeros(len(beta))
    contrib[i_post] = 1.0
    contrib[i_slope] = float(periods_since.mean())
    effect = float(contrib @ beta)

    cov = np.asarray(model.cov_params(), dtype=float)
    se = float(np.sqrt(max(contrib @ cov @ contrib, 0.0)))
    from scipy import stats as _stats

    dof = int(model.df_resid)
    crit = float(_stats.t.ppf(0.5 + level / 2, dof)) if dof > 0 else 1.96
    interval = Interval(
        lower=effect - crit * se, upper=effect + crit * se, level=level, kind="confidence"
    )
    t_stat = effect / se if se > 0 else np.nan
    p_value = (
        float(2 * (1 - _stats.t.cdf(abs(t_stat), dof))) if np.isfinite(t_stat) and dof > 0 else None
    )

    half = crit * se
    cf_lower = cf.copy()
    cf_upper = cf.copy()
    cf_lower.loc[post_sel] = cf.loc[post_sel] - half
    cf_upper.loc[post_sel] = cf.loc[post_sel] + half

    rel, rel_interval = relative_from_counterfactual(
        effect, interval, cf.to_numpy()[post_sel]
    )

    resid = np.asarray(model.resid, dtype=float)
    ss_res = float(resid @ resid)
    dw = float(np.sum(np.diff(resid) ** 2) / ss_res) if ss_res > 0 else float("nan")

    warn_list: list[str] = list(ds.warnings())
    warn_list.insert(
        0,
        "Interrupted time series has no control group. Any concurrent change -- season, "
        "competitor, pricing, news -- is attributed to the intervention. Read this as the "
        "weakest available evidence, not as a causal estimate.",
    )
    slope_p = float(model.pvalues[i_slope])
    if slope_p < 0.05:
        warn_list.append(
            f"The post-intervention slope also changed (p={slope_p:.3f}). A single average "
            "effect hides that: the effect is "
            + ("growing" if beta[i_slope] > 0 else "decaying")
            + " over time."
        )
    if np.isfinite(dw) and (dw < 1.0 or dw > 3.0):
        warn_list.append(
            f"Residual autocorrelation remains after HAC correction (Durbin-Watson={dw:.2f}). "
            "Consider modelling the series directly (ARIMA) before trusting the interval."
        )

    return ImpactEstimate(
        method="Interrupted Time Series",
        effect=effect,
        interval=interval,
        relative_effect=rel,
        relative_interval=rel_interval,
        cumulative_effect=effect * n_post,
        p_value=p_value,
        n_pre=n_pre,
        n_post=n_post,
        observed=y_series,
        counterfactual=cf,
        counterfactual_lower=cf_lower,
        counterfactual_upper=cf_upper,
        assumptions=list(_ASSUMPTIONS),
        warnings=warn_list,
        diagnostics={
            "level_change": float(beta[i_post]),
            "level_change_p": float(model.pvalues[i_post]),
            "slope_change": float(beta[i_slope]),
            "slope_change_p": slope_p,
            "pre_trend": float(beta[names.index("time")]),
            "r_squared": float(model.rsquared),
            "durbin_watson": dw,
            "hac_lags": int(hac_lags),
            "std_error": se,
            "inference": f"Newey-West HAC (lags={hac_lags})",
        },
    )
