"""Bayesian counterfactual regression (CausalImpact-style).

Google's CausalImpact fits a Bayesian structural time series model on the
pre-period, using control series as regressors, then forecasts the post-period
counterfactual and reads the effect off the gap. This module implements the same
idea with a conjugate Bayesian linear model rather than a full BSTS state-space
model.

What is kept
------------
* Control series as contemporaneous regressors.
* A deterministic trend and Fourier seasonality basis in place of BSTS's
  stochastic local-level and seasonal components.
* Full posterior predictive simulation, so the credible interval propagates both
  coefficient uncertainty and observation noise.
* Cumulative-effect inference from the same posterior draws.

What is given up
----------------
A deterministic trend cannot drift the way a stochastic local level can. On a
series whose level wanders without the controls wandering with it, this model
will be over-confident. That limitation is stated in ``assumptions`` and is the
reason the pre-period back-test below is run on every fit.

Model
-----
``y_t = x_t' beta + e_t``, ``e_t ~ N(0, sigma^2)`` with a conjugate
Normal-Inverse-Gamma prior ``beta | sigma^2 ~ N(0, sigma^2 / lambda * I)``,
``sigma^2 ~ InvGamma(a0, b0)``. The posterior is analytic, so no MCMC is needed
and results are exactly reproducible given a seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..dataset import PanelDataset
from ..types import ImpactEstimate, Interval, relative_from_counterfactual

__all__ = ["estimate_bayesian_counterfactual"]

_ASSUMPTIONS = [
    "The pre-period relationship between the treated series and the control "
    "series continues to hold through the post period.",
    "Control series were not themselves affected by the intervention.",
    "The level of the series is explained by the controls, the trend and the "
    "seasonal basis. An unmodelled drift in the treated series alone will be "
    "misread as an effect.",
    "Residuals are approximately normal and not strongly autocorrelated. The "
    "reported Durbin-Watson statistic is the check on this.",
]


def _seasonal_basis(t: np.ndarray, period: float, n_harmonics: int) -> np.ndarray:
    """Fourier terms for a seasonal cycle of the given period."""
    cols = []
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * k * t / period))
        cols.append(np.cos(2 * np.pi * k * t / period))
    if not cols:
        return np.empty((len(t), 0))
    return np.column_stack(cols)


def _infer_season_length(index: pd.DatetimeIndex) -> float | None:
    """Guess the annual cycle length in periods from the index frequency."""
    if len(index) < 3:
        return None
    deltas = np.diff(index.to_numpy()).astype("timedelta64[D]").astype(float)
    median_days = float(np.median(deltas))
    if median_days <= 0:
        return None
    if median_days <= 1.5:
        return 7.0  # daily data: weekly cycle
    if median_days <= 10:
        return 365.25 / median_days  # weekly data: annual cycle
    if median_days <= 45:
        return 12.0  # monthly data
    return None


def estimate_bayesian_counterfactual(
    ds: PanelDataset,
    *,
    level: float = 0.95,
    n_draws: int = 4000,
    prior_precision: float = 1.0,
    prior_shape: float = 2.0,
    prior_scale: float = 1.0,
    n_harmonics: int | None = None,
    include_trend: bool | None = None,
    seed: int = 0,
) -> ImpactEstimate:
    """Fit the pre-period, forecast the counterfactual, measure the gap.

    Parameters
    ----------
    ds:
        Validated panel. Works with or without control units; without controls
        it reduces to a trend-plus-seasonality forecast, which is much weaker.
    level:
        Credible-interval level.
    n_draws:
        Posterior predictive draws. More draws means smoother interval edges.
    prior_precision:
        ``lambda`` in the prior. Higher values shrink coefficients toward zero,
        which stabilises fits when control series are collinear.
    n_harmonics, include_trend:
        Deterministic structure to add alongside the controls. ``None`` (the
        default) resolves them automatically: **off** when control series are
        present, **on** when they are not.

        That default is empirical, not cosmetic. Over 120 replications of the
        packaged simulation with 8 controls, adding a deterministic trend and two
        Fourier harmonics on top of the donors raised the bias from +0.58 to
        +12.25 and dropped 95% interval coverage from 92.5% to 49.2%. The donors
        already carry the trend and the season, so the duplicated terms are
        near-collinear with them and the fitted seasonal amplitude
        mis-extrapolates past the intervention date. See
        ``scripts/measure_bayesian_spec.py`` for the full table. Pass the
        arguments explicitly to override, and re-run
        :func:`impactlens.run_recovery_study` if you do.
    seed:
        RNG seed for the posterior draws.
    """
    treated = ds.treated_series.astype(float)
    donors = ds.control_matrix()
    if donors.shape[1] > 0:
        index = treated.index.intersection(donors.index)
        donors = donors.loc[index]
    else:
        index = treated.index
    treated = treated.loc[index]

    has_controls = donors.shape[1] > 0
    if include_trend is None:
        include_trend = not has_controls
    if n_harmonics is None:
        n_harmonics = 0 if has_controls else 2

    pre_mask = np.asarray(index < ds.intervention_date, dtype=bool)
    n_pre = int(pre_mask.sum())
    n_post = int((~pre_mask).sum())

    t = np.arange(len(index), dtype=float)
    blocks: list[np.ndarray] = [np.ones((len(index), 1))]
    names: list[str] = ["intercept"]

    if include_trend:
        # Centre and scale the trend so the prior means the same thing
        # regardless of series length.
        blocks.append(((t - t.mean()) / max(t.std(), 1.0)).reshape(-1, 1))
        names.append("trend")

    season_length = _infer_season_length(pd.DatetimeIndex(index))
    if n_harmonics > 0 and season_length and season_length >= 4 and n_pre >= 2 * season_length / 4:
        basis = _seasonal_basis(t, season_length, n_harmonics)
        if basis.shape[1]:
            blocks.append(basis)
            for k in range(1, n_harmonics + 1):
                names += [f"sin_{k}", f"cos_{k}"]

    if has_controls:
        # Standardise donors on the pre-period so the shared prior precision
        # means the same thing for every column regardless of its scale.
        D = donors.to_numpy(dtype=float)
        donor_means = D[pre_mask].mean(axis=0)
        donor_sds = D[pre_mask].std(axis=0, ddof=1)
        donor_sds = np.where(donor_sds > 0, donor_sds, 1.0)
        blocks.append((D - donor_means) / donor_sds)
        names += [str(c) for c in donors.columns]

    X = np.column_stack(blocks)
    y = treated.to_numpy(dtype=float)
    X_pre, y_pre = X[pre_mask], y[pre_mask]

    if n_pre <= X.shape[1]:
        raise ValueError(
            f"model has {X.shape[1]} parameters but only {n_pre} pre-period observations. "
            "Reduce n_harmonics, drop control units, or supply a longer pre-period."
        )

    # ------------------------------------------------- conjugate NIG posterior
    # Intercept is left unpenalised; everything else is shrunk toward zero.
    penalty = np.full(X.shape[1], float(prior_precision))
    penalty[0] = 1e-8
    A = X_pre.T @ X_pre + np.diag(penalty)
    V_n = np.linalg.inv(A)
    beta_n = V_n @ (X_pre.T @ y_pre)

    a_n = prior_shape + n_pre / 2.0
    b_n = prior_scale + 0.5 * float(y_pre @ y_pre - beta_n @ A @ beta_n)
    b_n = max(b_n, 1e-12)

    rng = np.random.default_rng(seed)
    sigma2 = stats.invgamma.rvs(a=a_n, scale=b_n, size=n_draws, random_state=rng)
    sigma = np.sqrt(sigma2)

    # beta | sigma^2 ~ N(beta_n, sigma^2 * V_n)
    L = np.linalg.cholesky((V_n + V_n.T) / 2.0 + 1e-12 * np.eye(len(beta_n)))
    z = rng.standard_normal((n_draws, len(beta_n)))
    beta_draws = beta_n[None, :] + (z @ L.T) * sigma[:, None]

    # Posterior predictive counterfactual for every period.
    mu_draws = beta_draws @ X.T  # (n_draws, T)
    pred_draws = mu_draws + rng.standard_normal(mu_draws.shape) * sigma[:, None]

    alpha = 1.0 - level
    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    cf_mean = pd.Series(mu_draws.mean(axis=0), index=index, name="counterfactual")
    cf_lower = pd.Series(np.quantile(pred_draws, lo_q, axis=0), index=index)
    cf_upper = pd.Series(np.quantile(pred_draws, hi_q, axis=0), index=index)

    # Effect posterior: observed minus counterfactual draws, averaged over post.
    post_idx = np.flatnonzero(~pre_mask)
    effect_draws = (y[post_idx][None, :] - pred_draws[:, post_idx]).mean(axis=1)
    effect = float(effect_draws.mean())
    interval = Interval(
        lower=float(np.quantile(effect_draws, lo_q)),
        upper=float(np.quantile(effect_draws, hi_q)),
        level=level,
        kind="credible",
    )

    cumulative_draws = (y[post_idx][None, :] - pred_draws[:, post_idx]).sum(axis=1)
    # Two-sided posterior tail probability of no effect.
    tail = float(min((effect_draws <= 0).mean(), (effect_draws >= 0).mean()))
    p_value = min(1.0, 2.0 * tail)

    rel, rel_interval = relative_from_counterfactual(
        effect, interval, cf_mean.to_numpy()[post_idx]
    )

    # -------------------------------------------------------- fit diagnostics
    resid_pre = y_pre - X_pre @ beta_n
    ss_res = float(resid_pre @ resid_pre)
    ss_tot = float(((y_pre - y_pre.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dw = (
        float(np.sum(np.diff(resid_pre) ** 2) / ss_res) if ss_res > 0 else float("nan")
    )
    mape_pre = (
        float(np.mean(np.abs(resid_pre / np.where(y_pre != 0, y_pre, np.nan))) * 100)
        if n_pre
        else float("nan")
    )

    warn_list: list[str] = list(ds.warnings())
    if not has_controls:
        warn_list.append(
            "No control series available. The counterfactual rests entirely on extrapolating "
            "the treated unit's own trend and seasonality, so any market-wide movement that "
            "began near the intervention date will be reported as an effect -- confidently, "
            "and wrongly. On the packaged simulation this specification attributes essentially "
            "the whole market shock to the intervention. Add control units before quoting this."
        )
    if np.isfinite(dw) and (dw < 1.0 or dw > 3.0):
        warn_list.append(
            f"Residual autocorrelation detected (Durbin-Watson={dw:.2f}, expected near 2). "
            "The credible interval is too narrow; treat it as a lower bound on uncertainty."
        )
    if np.isfinite(r2) and r2 < 0.3:
        warn_list.append(
            f"Weak pre-period fit (R^2={r2:.2f}). The model does not explain the treated "
            "series well, so its forecast of the counterfactual is not trustworthy."
        )

    # Pre-period back-test: hold out the last 20% of the pre-period, refit, and
    # see whether the model would have correctly found no effect there. This is
    # the honest check that the machinery is not manufacturing effects.
    backtest = _pre_period_backtest(
        X, y, pre_mask, penalty, level=level, prior_shape=prior_shape,
        prior_scale=prior_scale, seed=seed,
    )
    if backtest.get("false_positive"):
        warn_list.append(
            "Placebo back-test failed: applying the same model to a fake intervention "
            "inside the pre-period produced a 'significant' effect. The model is finding "
            "signal where there is none, so the headline result is not credible."
        )

    return ImpactEstimate(
        method="Bayesian Counterfactual (CausalImpact-style)",
        effect=effect,
        interval=interval,
        relative_effect=rel,
        relative_interval=rel_interval,
        cumulative_effect=float(cumulative_draws.mean()),
        p_value=p_value,
        n_pre=n_pre,
        n_post=n_post,
        observed=treated,
        counterfactual=cf_mean,
        counterfactual_lower=cf_lower,
        counterfactual_upper=cf_upper,
        assumptions=list(_ASSUMPTIONS),
        warnings=warn_list,
        diagnostics={
            "r_squared_pre": r2,
            "durbin_watson": dw,
            "mape_pre": mape_pre,
            "sigma_posterior_mean": float(sigma.mean()),
            "n_regressors": int(X.shape[1]),
            "regressor_names": names,
            "season_length": season_length,
            "n_draws": int(n_draws),
            "n_controls_used": int(donors.shape[1]),
            "include_trend": bool(include_trend),
            "n_harmonics": int(n_harmonics),
            "prior_precision": float(prior_precision),
            "cumulative_ci": (
                float(np.quantile(cumulative_draws, lo_q)),
                float(np.quantile(cumulative_draws, hi_q)),
            ),
            "coefficients": pd.Series(beta_n, index=names, name="posterior_mean"),
            "backtest": backtest,
            "inference": "posterior predictive simulation",
        },
    )


def _pre_period_backtest(
    X: np.ndarray,
    y: np.ndarray,
    pre_mask: np.ndarray,
    penalty: np.ndarray,
    *,
    level: float,
    prior_shape: float,
    prior_scale: float,
    seed: int,
    holdout_frac: float = 0.2,
) -> dict[str, object]:
    """Refit on an earlier slice and test for an effect that cannot exist.

    Splits the pre-period, fits on the first part, and evaluates the second part
    as if the intervention had happened there. A credible interval that excludes
    zero is a false positive and means the model is mis-specified.
    """
    pre_idx = np.flatnonzero(pre_mask)
    n_pre = len(pre_idx)
    n_hold = max(2, int(round(n_pre * holdout_frac)))
    n_train = n_pre - n_hold
    if n_train <= X.shape[1] + 1:
        return {"ran": False, "reason": "pre-period too short to back-test"}

    train_idx = pre_idx[:n_train]
    hold_idx = pre_idx[n_train:]

    X_tr, y_tr = X[train_idx], y[train_idx]
    A = X_tr.T @ X_tr + np.diag(penalty)
    try:
        V = np.linalg.inv(A)
    except np.linalg.LinAlgError:  # pragma: no cover
        return {"ran": False, "reason": "singular design"}
    beta = V @ (X_tr.T @ y_tr)
    a_n = prior_shape + n_train / 2.0
    b_n = max(prior_scale + 0.5 * float(y_tr @ y_tr - beta @ A @ beta), 1e-12)

    rng = np.random.default_rng(seed + 991)
    n_draws = 2000
    sigma = np.sqrt(stats.invgamma.rvs(a=a_n, scale=b_n, size=n_draws, random_state=rng))
    L = np.linalg.cholesky((V + V.T) / 2.0 + 1e-12 * np.eye(len(beta)))
    z = rng.standard_normal((n_draws, len(beta)))
    beta_draws = beta[None, :] + (z @ L.T) * sigma[:, None]

    X_h = X[hold_idx]
    pred = beta_draws @ X_h.T + rng.standard_normal((n_draws, len(hold_idx))) * sigma[:, None]
    eff = (y[hold_idx][None, :] - pred).mean(axis=1)
    alpha = 1.0 - level
    lo = float(np.quantile(eff, alpha / 2))
    hi = float(np.quantile(eff, 1 - alpha / 2))
    return {
        "ran": True,
        "n_train": int(n_train),
        "n_holdout": int(n_hold),
        "placebo_effect": float(eff.mean()),
        "ci_lower": lo,
        "ci_upper": hi,
        "false_positive": bool(lo > 0 or hi < 0),
    }
