"""The naive before/after comparison.

This is not a serious estimator. It is included because it is what almost every
business dashboard actually reports, and because showing its answer next to a
control-based answer is the fastest way to demonstrate the bias that motivates
this package.

Its interval is a two-sample t interval on the difference in means, which treats
the pre and post windows as independent samples from the same process. That is
false whenever there is a trend or a season -- which is almost always.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..dataset import PanelDataset
from ..types import ImpactEstimate, Interval, relative_from_counterfactual

__all__ = ["estimate_naive_prepost"]


def estimate_naive_prepost(ds: PanelDataset, *, level: float = 0.95) -> ImpactEstimate:
    """Post-period mean minus pre-period mean for the treated unit."""
    series = ds.treated_series.astype(float)
    post_sel = np.asarray(series.index >= ds.intervention_date, dtype=bool)
    pre = series.to_numpy()[~post_sel]
    post = series.to_numpy()[post_sel]

    effect = float(post.mean() - pre.mean())
    # Welch's t interval: unequal variances, independent samples.
    se = float(np.sqrt(pre.var(ddof=1) / len(pre) + post.var(ddof=1) / len(post)))
    dof_num = (pre.var(ddof=1) / len(pre) + post.var(ddof=1) / len(post)) ** 2
    dof_den = (pre.var(ddof=1) / len(pre)) ** 2 / (len(pre) - 1) + (
        post.var(ddof=1) / len(post)
    ) ** 2 / max(len(post) - 1, 1)
    dof = float(dof_num / dof_den) if dof_den > 0 else float(len(pre) + len(post) - 2)
    crit = float(stats.t.ppf(0.5 + level / 2, max(dof, 1.0)))
    interval = Interval(
        lower=effect - crit * se, upper=effect + crit * se, level=level, kind="confidence"
    )
    t_stat = effect / se if se > 0 else np.nan
    p_value = (
        float(2 * (1 - stats.t.cdf(abs(t_stat), max(dof, 1.0))))
        if np.isfinite(t_stat)
        else None
    )

    # The implied counterfactual is a flat line at the pre-period mean.
    counterfactual = series.copy()
    counterfactual.loc[:] = float(pre.mean())
    counterfactual.name = "counterfactual"
    cf_lower = counterfactual - crit * se
    cf_upper = counterfactual + crit * se

    rel, rel_interval = relative_from_counterfactual(
        effect, interval, counterfactual.to_numpy()[post_sel]
    )

    return ImpactEstimate(
        method="Naive pre/post comparison",
        effect=effect,
        interval=interval,
        relative_effect=rel,
        relative_interval=rel_interval,
        cumulative_effect=effect * int(post_sel.sum()),
        p_value=p_value,
        n_pre=int((~post_sel).sum()),
        n_post=int(post_sel.sum()),
        observed=series,
        counterfactual=counterfactual,
        counterfactual_lower=cf_lower,
        counterfactual_upper=cf_upper,
        assumptions=[
            "Nothing but the intervention differs between the two windows. No trend, "
            "no seasonality, no market movement. This is almost never true.",
            "Observations are independent within each window, which time series are not.",
        ],
        warnings=[
            "This is a benchmark, not an estimate. It is reported so you can see how far "
            "a naive before/after reading drifts from a control-based one. Do not quote it.",
        ],
        diagnostics={
            "pre_mean": float(pre.mean()),
            "post_mean": float(post.mean()),
            "n_controls_used": 0,
            "std_error": se,
            "inference": "Welch two-sample t",
        },
    )
