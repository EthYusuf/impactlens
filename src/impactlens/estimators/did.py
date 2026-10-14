"""Difference-in-Differences with two-way fixed effects.

Model
-----
``y_it = alpha_i + gamma_t + tau * (treated_i * post_t) + e_it``

``tau`` is the average treatment effect on the treated. Unit fixed effects
absorb permanent level differences between units; time fixed effects absorb
anything that moved every unit at once -- the seasonal surge, a holiday, a
market-wide price change. That is exactly the confound a naive before/after
comparison cannot handle.

Inference
---------
Standard errors are clustered by unit, because outcomes within a unit are
serially correlated and treating each (unit, period) as independent understates
the variance badly. Cluster-robust inference is itself asymptotic in the number
of clusters, so with few units the estimator falls back to a wild cluster
bootstrap and says so in ``warnings``.
"""

from __future__ import annotations

import warnings as _warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..dataset import PanelDataset
from ..types import ImpactEstimate, Interval, relative_from_counterfactual

__all__ = ["estimate_did", "event_study"]

_ASSUMPTIONS = [
    "Parallel trends: without the intervention, the treated unit would have moved "
    "parallel to the controls. This is the assumption that can break the result, "
    "and it is testable only in the pre-period.",
    "No spillover: the intervention did not affect the control units.",
    "No anticipation: behaviour did not change before the intervention date.",
    "Stable composition: the units are the same entities throughout the window.",
]

_MIN_CLUSTERS_FOR_ASYMPTOTICS = 10


def _design(ds: PanelDataset) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build the two-way fixed effects design matrix."""
    df = ds.frame.copy()
    df["_treated"] = (df[ds.unit_col] == ds.treated_unit).astype(float)
    df["_post"] = (df[ds.date_col] >= ds.intervention_date).astype(float)
    df["_did"] = df["_treated"] * df["_post"]

    unit_d = pd.get_dummies(df[ds.unit_col], prefix="u", drop_first=True, dtype=float)
    time_d = pd.get_dummies(
        df[ds.date_col].dt.strftime("%Y-%m-%d"), prefix="t", drop_first=True, dtype=float
    )
    X = pd.concat([df[["_did"]], unit_d, time_d], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = df[ds.outcome_col].astype(float)
    groups = df[ds.unit_col]
    return X, y, groups


class _ClusterOLS:
    """Fixed-design OLS with cluster-robust variance, reusable across refits.

    The wild bootstrap refits the *same* design matrix thousands of times with
    only the response vector changing. Calling ``sm.OLS(...).fit()`` each time
    re-factorises a 900x110 design every iteration, which made the recovery
    study take hours. Factorising once and reusing the projection matrix is the
    same arithmetic, two orders of magnitude faster.

    Uses the same finite-sample correction as statsmodels'
    ``cov_type="cluster"``: ``G/(G-1) * (N-1)/(N-K)``.
    """

    def __init__(self, X: np.ndarray, codes: np.ndarray, target_idx: int) -> None:
        self.X = X
        self.codes = codes
        self.target_idx = target_idx
        self.n, self.k = X.shape
        self.n_clusters = int(codes.max()) + 1
        xtx = X.T @ X
        self.bread = np.linalg.pinv(xtx)
        self.proj = self.bread @ X.T  # beta = proj @ y
        self.correction = (
            self.n_clusters / max(self.n_clusters - 1, 1) * (self.n - 1) / max(self.n - self.k, 1)
        )
        # Group row indices once so the variance "meat" is a cheap loop.
        order = np.argsort(codes, kind="stable")
        self._order = order
        boundaries = np.searchsorted(codes[order], np.arange(self.n_clusters + 1))
        self._slices = [
            (boundaries[g], boundaries[g + 1]) for g in range(self.n_clusters)
        ]

    def fit(self, y: np.ndarray) -> tuple[float, float]:
        """Return ``(coefficient, cluster-robust std error)`` for the target column."""
        beta = self.proj @ y
        resid = y - self.X @ beta
        Xo = self.X[self._order]
        uo = resid[self._order]
        meat = np.zeros((self.k, self.k))
        for lo, hi in self._slices:
            if hi <= lo:
                continue
            sg = Xo[lo:hi].T @ uo[lo:hi]
            meat += np.outer(sg, sg)
        cov = self.bread @ meat @ self.bread * self.correction
        var = cov[self.target_idx, self.target_idx]
        se = float(np.sqrt(var)) if var > 0 else float("nan")
        return float(beta[self.target_idx]), se


def _wild_cluster_bootstrap(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    n_boot: int,
    level: float,
    seed: int,
) -> tuple[float, float, float]:
    """Rademacher wild cluster bootstrap-t for the ``_did`` coefficient.

    Imposes the null (tau = 0) when resampling, which is the variant with the
    best small-sample behaviour (Cameron, Gelbach & Miller 2008). Returns
    ``(ci_lower, ci_upper, p_value)``: a bootstrap-t interval obtained by
    inverting the simulated t distribution around the point estimate, and a
    p-value from the share of bootstrap t statistics at least as extreme as the
    observed one.
    """
    rng = np.random.default_rng(seed)
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    did_idx = list(X.columns).index("_did")
    codes, uniques = pd.factorize(groups)
    codes = np.asarray(codes, dtype=np.intp)
    n_clusters = len(uniques)

    solver = _ClusterOLS(Xv, codes, did_idx)
    tau_hat, se_hat = solver.fit(yv)
    if not np.isfinite(se_hat) or se_hat <= 0:
        return float("nan"), float("nan"), float("nan")

    # Restricted fit: drop the DiD column to impose the null.
    keep = [i for i in range(Xv.shape[1]) if i != did_idx]
    X_r = Xv[:, keep]
    beta_r = np.linalg.pinv(X_r.T @ X_r) @ (X_r.T @ yv)
    y_null = X_r @ beta_r
    resid_null = yv - y_null

    t_stats = np.empty(n_boot)
    for b in range(n_boot):
        signs = rng.choice((-1.0, 1.0), size=n_clusters)
        y_star = y_null + resid_null * signs[codes]
        tau_b, se_b = solver.fit(y_star)
        t_stats[b] = tau_b / se_b if np.isfinite(se_b) and se_b > 0 else np.nan

    t_stats = t_stats[np.isfinite(t_stats)]
    if t_stats.size < 20:
        return float("nan"), float("nan"), float("nan")

    t_obs = tau_hat / se_hat
    p_value = float(np.mean(np.abs(t_stats) >= abs(t_obs)))
    alpha = 1.0 - level
    lo_q, hi_q = np.quantile(t_stats, [alpha / 2, 1 - alpha / 2])
    # Invert the bootstrap-t distribution around the point estimate. The
    # standard error comes from the same solver as the t statistics, so the
    # interval is internally consistent.
    return float(tau_hat - hi_q * se_hat), float(tau_hat - lo_q * se_hat), p_value


def estimate_did(
    ds: PanelDataset,
    *,
    level: float = 0.95,
    bootstrap: bool | None = None,
    n_boot: int = 999,
    seed: int = 0,
) -> ImpactEstimate:
    """Estimate the average treatment effect on the treated by DiD.

    Parameters
    ----------
    ds:
        Validated panel. Must contain at least one control unit.
    level:
        Confidence level for the interval.
    bootstrap:
        Force the wild cluster bootstrap on or off. ``None`` (default) turns it
        on automatically when there are too few clusters for cluster-robust
        asymptotics to be trusted.
    n_boot:
        Bootstrap replications when the bootstrap is used.
    seed:
        Bootstrap RNG seed.
    """
    if not ds.has_controls:
        raise ValueError(
            "difference-in-differences needs at least one control unit; "
            "use interrupted time series instead"
        )

    X, y, groups = _design(ds)
    n_clusters = groups.nunique()
    codes, _ = pd.factorize(groups)

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        model = sm.OLS(y.to_numpy(dtype=float), X.to_numpy(dtype=float)).fit(
            cov_type="cluster", cov_kwds={"groups": codes}, use_t=True
        )

    did_idx = list(X.columns).index("_did")
    tau = float(model.params[did_idx])
    se = float(model.bse[did_idx])

    use_boot = (n_clusters < _MIN_CLUSTERS_FOR_ASYMPTOTICS) if bootstrap is None else bootstrap
    warn_list: list[str] = list(ds.warnings())
    p_value: float | None

    if use_boot:
        lo, hi, p_value = _wild_cluster_bootstrap(
            X, y, groups, n_boot=n_boot, level=level, seed=seed
        )
        if not np.isfinite(lo):
            use_boot = False
            warn_list.append(
                "Wild cluster bootstrap failed to produce a usable interval; "
                "reverting to cluster-robust asymptotics, which are optimistic here."
            )
        else:
            interval = Interval(lo, hi, level=level, kind="confidence")
            warn_list.append(
                f"Only {n_clusters} clusters (units). The interval comes from a wild cluster "
                f"bootstrap ({n_boot} reps) because cluster-robust standard errors are "
                "unreliable with few clusters."
            )

    if not use_boot:
        ci = model.conf_int(alpha=1 - level)
        lo = float(ci[did_idx][0])
        hi = float(ci[did_idx][1])
        interval = Interval(lo, hi, level=level, kind="confidence")
        p_value = float(model.pvalues[did_idx])

    # Counterfactual for the treated unit: observed minus the estimated effect
    # over the post period. Pre-period counterfactual equals the observation.
    observed = ds.treated_series
    post_mask = observed.index >= ds.intervention_date
    counterfactual = observed.copy().astype(float)
    counterfactual.loc[post_mask] = counterfactual.loc[post_mask] - tau
    counterfactual.name = "counterfactual"

    half = (interval.upper - interval.lower) / 2.0
    cf_lower = counterfactual.copy()
    cf_upper = counterfactual.copy()
    cf_lower.loc[post_mask] = counterfactual.loc[post_mask] - half
    cf_upper.loc[post_mask] = counterfactual.loc[post_mask] + half

    rel, rel_interval = relative_from_counterfactual(
        tau, interval, counterfactual.loc[post_mask].to_numpy()
    )

    pre_test = event_study(ds, level=level)
    pretrend_p = pre_test.get("pretrend_p_value")
    if pretrend_p is not None and np.isfinite(pretrend_p) and pretrend_p < 0.05:
        slope = pre_test.get("pretrend_slope")
        slope_txt = f" at {slope:+,.3g} per period" if slope is not None else ""
        warn_list.append(
            f"The treated-minus-control gap was already drifting{slope_txt} before the "
            f"intervention (slope test p={pretrend_p:.4f}). Parallel trends -- the "
            "assumption this estimate depends on -- looks violated, so the DiD number is "
            "likely biased. Prefer synthetic control, which fits the pre-period instead "
            "of assuming it away."
        )

    return ImpactEstimate(
        method="Difference-in-Differences (two-way FE)",
        effect=tau,
        interval=interval,
        relative_effect=rel,
        relative_interval=rel_interval,
        cumulative_effect=tau * ds.n_post,
        p_value=p_value,
        n_pre=ds.n_pre,
        n_post=ds.n_post,
        observed=observed,
        counterfactual=counterfactual,
        counterfactual_lower=cf_lower,
        counterfactual_upper=cf_upper,
        assumptions=list(_ASSUMPTIONS),
        warnings=warn_list,
        diagnostics={
            "std_error": se,
            "n_clusters": int(n_clusters),
            "n_obs": int(len(y)),
            "r_squared": float(model.rsquared),
            "inference": "wild cluster bootstrap" if use_boot else "cluster-robust (by unit)",
            "event_study": pre_test,
        },
    )


def event_study(ds: PanelDataset, *, level: float = 0.95) -> dict[str, object]:
    """Event-study path and a pre-trend test, built on the treated-control gap.

    Why not the textbook saturated regression
    -----------------------------------------
    The usual event study interacts a treated indicator with a dummy per
    relative period inside a two-way fixed effects model. With a *single* treated
    unit -- the normal case for "did our new booking system work?" -- each of
    those interaction dummies is identified by exactly one observation, so each
    coefficient is just that cell's residual. A joint F-test across several of
    them, with a cluster-robust covariance whose rank is capped at the number of
    units, then rejects far too often: on the packaged simulation it returned
    p = 5e-4 for data where parallel trends holds by construction.

    What is tested instead
    ----------------------
    With one treated unit and a balanced panel, two-way FE DiD is algebraically
    the change in the treated unit minus the change in the control mean. So the
    object that has to be flat before the intervention is the **gap**

    ``gap_t = y_treated,t - mean_j y_control_j,t``

    This function regresses that gap on time over the pre-period with
    Newey-West standard errors and tests whether its slope is zero. That is a
    well-posed, correctly sized test with a single treated unit, and it is the
    assumption DiD actually needs: not that the gap is zero, but that it is not
    drifting.

    Returns
    -------
    dict with keys ``coefficients`` (DataFrame indexed by relative period, for
    plotting), ``pretrend_p_value``, ``pretrend_slope``, ``pretrend_slope_se``,
    ``gap``, and ``reference_period``.
    """
    empty = {
        "coefficients": pd.DataFrame(),
        "pretrend_p_value": None,
        "pretrend_slope": None,
        "reference_period": -1,
    }
    if not ds.has_controls:
        return empty

    controls = ds.control_matrix()
    if controls.shape[1] == 0:
        return empty

    treated = ds.treated_series.astype(float)
    index = treated.index.intersection(controls.index)
    gap = treated.loc[index] - controls.loc[index].mean(axis=1)
    gap.name = "gap"

    pre_sel = np.asarray(index < ds.intervention_date, dtype=bool)
    n_pre = int(pre_sel.sum())
    if n_pre < 6:
        return {**empty, "gap": gap}

    # ------------------------------------------------- pre-trend slope test
    t_all = np.arange(len(index), dtype=float)
    t_pre = t_all[pre_sel]
    g_pre = gap.to_numpy(dtype=float)[pre_sel]

    X_pre = sm.add_constant(t_pre.reshape(-1, 1), has_constant="add")
    hac = max(1, min(int(np.floor(4 * (n_pre / 100.0) ** (2.0 / 9.0))), n_pre // 4))
    pretrend_p: float | None = None
    slope = slope_se = float("nan")
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            trend_fit = sm.OLS(g_pre, X_pre).fit(
                cov_type="HAC", cov_kwds={"maxlags": hac}, use_t=True
            )
        slope = float(trend_fit.params[1])
        slope_se = float(trend_fit.bse[1])
        pretrend_p = float(trend_fit.pvalues[1])
    except Exception:  # pragma: no cover - degenerate pre-periods
        pass

    # --------------------------------------------- event path for plotting
    # Each point is the gap in that period, centred on the pre-period mean, so
    # a flat scatter at zero before the intervention is the "nothing was already
    # happening" picture. The band is the pre-period noise level, which is the
    # relevant yardstick for judging whether a point is unusual.
    #
    # Every period is returned, not a window around the intervention. The slope
    # test uses the whole pre-period, and a plot showing only the last few
    # periods would hide the very drift the test is reporting: a 0.38/period
    # divergence is 2.7 units over 7 periods but 30 units over 78.
    gap_pre_mean = float(np.mean(g_pre))
    gap_pre_sd = float(np.std(g_pre, ddof=1)) if n_pre > 1 else float("nan")
    crit = float(stats.norm.ppf(0.5 + level / 2))
    half = crit * gap_pre_sd if np.isfinite(gap_pre_sd) else 0.0

    rel = np.arange(len(index)) - n_pre
    centred = gap.to_numpy(dtype=float) - gap_pre_mean
    coefficients = pd.DataFrame(
        {
            "relative_period": rel.astype(int),
            "coefficient": centred,
            "std_error": gap_pre_sd,
            "ci_lower": centred - half,
            "ci_upper": centred + half,
            "is_pre": rel < 0,
        }
    ).set_index("relative_period")

    return {
        "coefficients": coefficients,
        "pretrend_p_value": pretrend_p,
        "pretrend_slope": slope,
        "pretrend_slope_se": slope_se,
        "pretrend_hac_lags": hac,
        "gap": gap,
        "gap_pre_sd": gap_pre_sd,
        "n_pre_tested": n_pre,
        "reference_period": -1,
        "test": "Newey-West slope test on the treated-minus-control-mean gap",
    }
