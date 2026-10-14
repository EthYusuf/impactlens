"""Figures for the report and the dashboard.

Every function takes an estimate (or a dataset) and returns a matplotlib
``Figure``, so the same code renders PNGs for the README, panels in Streamlit
and pages in the PDF.

Design rules, applied consistently:

* The counterfactual is always dashed and muted; the observation is solid and
  saturated. The reader should never confuse a model output with a measurement.
* Uncertainty is always drawn as a band, never omitted. A point estimate with no
  band is the thing this project is arguing against.
* Zero lines are drawn on every effect plot, because "does the band cross zero"
  is the actual decision.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - must precede pyplot import

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ..dataset import PanelDataset  # noqa: E402
from ..types import ImpactEstimate  # noqa: E402

__all__ = [
    "COLORS",
    "plot_counterfactual",
    "plot_pointwise_effect",
    "plot_cumulative_effect",
    "plot_event_study",
    "plot_placebo_distribution",
    "plot_weights",
    "plot_panel_overview",
    "plot_method_comparison",
    "plot_recovery_summary",
    "apply_style",
]

COLORS = {
    "observed": "#1b3a5c",
    "counterfactual": "#c2653b",
    "band": "#e8a87c",
    "effect": "#2b7a6f",
    "negative": "#a63d40",
    "grid": "#d8d8d8",
    "muted": "#8a8a8a",
    "control": "#b9c4cf",
    "intervention": "#4a4a4a",
}


def apply_style() -> None:
    """Set the shared matplotlib style. Idempotent."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "figure.dpi": 110,
        }
    )


def _mark_intervention(ax, when, label: str = "intervention") -> None:
    ax.axvline(when, color=COLORS["intervention"], linestyle=":", linewidth=1.4, zorder=1)
    ax.annotate(
        label,
        xy=(when, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(5, -12),
        textcoords="offset points",
        fontsize=8,
        color=COLORS["intervention"],
        ha="left",
        va="top",
    )


def _format_dates(ax, index) -> None:
    span_days = (pd.Timestamp(max(index)) - pd.Timestamp(min(index))).days
    if span_days > 500:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    elif span_days > 180:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_ha("center")


def plot_counterfactual(
    est: ImpactEstimate,
    *,
    intervention_date: pd.Timestamp | None = None,
    title: str | None = None,
    outcome_label: str = "Outcome",
    figsize: tuple[float, float] = (9.5, 4.6),
) -> Figure:
    """Observed series against the estimated counterfactual with its band."""
    apply_style()
    if est.observed is None or est.counterfactual is None:
        raise ValueError("estimate carries no series to plot")

    when = intervention_date or est.counterfactual.index[est.n_pre]
    fig, ax = plt.subplots(figsize=figsize)

    if est.counterfactual_lower is not None and est.counterfactual_upper is not None:
        ax.fill_between(
            est.counterfactual.index,
            est.counterfactual_lower,
            est.counterfactual_upper,
            color=COLORS["band"],
            alpha=0.35,
            linewidth=0,
            label=f"{int(est.interval.level * 100)}% {est.interval.kind} band",
        )
    ax.plot(
        est.counterfactual.index,
        est.counterfactual,
        color=COLORS["counterfactual"],
        linestyle="--",
        linewidth=1.8,
        label="Counterfactual (modelled)",
    )
    ax.plot(
        est.observed.index,
        est.observed,
        color=COLORS["observed"],
        linewidth=2.0,
        label="Observed (measured)",
    )
    _mark_intervention(ax, when)
    _format_dates(ax, est.observed.index)

    ax.set_ylabel(outcome_label)
    ax.set_title(title or f"{est.method}: observed vs counterfactual")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    return fig


def plot_pointwise_effect(
    est: ImpactEstimate,
    *,
    intervention_date: pd.Timestamp | None = None,
    figsize: tuple[float, float] = (9.5, 3.4),
) -> Figure:
    """Period-by-period gap between observation and counterfactual."""
    apply_style()
    gap = est.pointwise_effect
    if gap is None:
        raise ValueError("estimate carries no series to plot")
    when = intervention_date or gap.index[est.n_pre]

    fig, ax = plt.subplots(figsize=figsize)
    if est.counterfactual_lower is not None and est.counterfactual_upper is not None:
        lo = est.observed - est.counterfactual_upper
        hi = est.observed - est.counterfactual_lower
        ax.fill_between(
            gap.index, lo, hi, color=COLORS["band"], alpha=0.35, linewidth=0,
            label=f"{int(est.interval.level * 100)}% band",
        )
    ax.axhline(0, color=COLORS["muted"], linewidth=1.0)
    ax.plot(gap.index, gap, color=COLORS["effect"], linewidth=1.8, label="Effect per period")
    _mark_intervention(ax, when)
    _format_dates(ax, gap.index)
    ax.set_ylabel("Observed - counterfactual")
    ax.set_title("Pointwise effect")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    return fig


def plot_cumulative_effect(
    est: ImpactEstimate,
    *,
    intervention_date: pd.Timestamp | None = None,
    figsize: tuple[float, float] = (9.5, 3.4),
) -> Figure:
    """Running total of the effect across the post period."""
    apply_style()
    gap = est.pointwise_effect
    if gap is None:
        raise ValueError("estimate carries no series to plot")
    when = intervention_date or gap.index[est.n_pre]

    post = gap.loc[gap.index >= when]
    cumulative = post.cumsum()
    # The band widens with the square root of elapsed periods: independent
    # per-period errors accumulate that way, not linearly.
    half = (est.interval.upper - est.interval.lower) / 2.0
    steps = np.arange(1, len(post) + 1)
    spread = half * np.sqrt(steps)

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(
        cumulative.index,
        cumulative - spread,
        cumulative + spread,
        color=COLORS["band"],
        alpha=0.35,
        linewidth=0,
        label=f"{int(est.interval.level * 100)}% band",
    )
    ax.axhline(0, color=COLORS["muted"], linewidth=1.0)
    ax.plot(
        cumulative.index, cumulative, color=COLORS["effect"], linewidth=2.0,
        label="Cumulative effect",
    )
    _format_dates(ax, cumulative.index)
    ax.set_ylabel("Cumulative effect")
    ax.set_title("Cumulative effect since the intervention")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    return fig


def plot_event_study(
    event: dict,
    *,
    figsize: tuple[float, float] = (8.5, 3.8),
) -> Figure:
    """The parallel-trends check: was the gap already drifting?

    Each point is the treated unit minus the control mean in that period,
    centred on the pre-period average. Points left of the dotted line should
    scatter flat around zero. A visible slope there means the treated unit was
    already diverging before the intervention, and difference-in-differences is
    not identified. The band is the pre-period noise level.
    """
    apply_style()
    coefs: pd.DataFrame = event.get("coefficients", pd.DataFrame())
    fig, ax = plt.subplots(figsize=figsize)
    if coefs.empty:
        ax.text(0.5, 0.5, "Pre-trend check unavailable\n(no control units)",
                ha="center", va="center", transform=ax.transAxes, color=COLORS["muted"])
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    pre = coefs[coefs["is_pre"]]
    post = coefs[~coefs["is_pre"]]
    if not pre.empty:
        half = float((pre["ci_upper"] - pre["coefficient"]).mean())
        ax.axhspan(-half, half, color=COLORS["band"], alpha=0.25, linewidth=0,
                   label="pre-period noise band")

    # Marker size scales down as the window grows, so a 100-period panel does
    # not become a solid blob.
    msize = 5.5 if len(coefs) <= 40 else (4.0 if len(coefs) <= 90 else 3.0)
    for part, color, label in (
        (pre, COLORS["muted"], "before intervention (should be flat)"),
        (post, COLORS["effect"], "after intervention"),
    ):
        if part.empty:
            continue
        ax.plot(part.index, part["coefficient"], "o", color=color, markersize=msize,
                label=label)

    # The fitted pre-period slope: exactly what the test measures, drawn across
    # the whole pre-period it was fitted on.
    slope = event.get("pretrend_slope")
    if slope is not None and np.isfinite(slope) and not pre.empty:
        xs = np.array([float(pre.index.min()), -1.0])
        centre = float(pre.index.to_numpy().mean())
        ax.plot(xs, slope * (xs - centre),
                color=COLORS["negative"], linestyle="--", linewidth=2.0,
                label=f"fitted pre-trend ({slope:+,.2g}/period)")

    ax.axhline(0, color="#222222", linewidth=1.0)
    ax.axvline(-0.5, color=COLORS["intervention"], linestyle=":", linewidth=1.4)
    ax.set_xlabel("Periods relative to the intervention")
    ax.set_ylabel("Treated - control mean\n(centred on pre-period)")

    p = event.get("pretrend_p_value")
    verdict = ""
    if p is not None and np.isfinite(p):
        verdict = (
            f"  |  pre-trend slope test p={p:.3f} "
            f"({'FAILS' if p < 0.05 else 'passes'})"
        )
    ax.set_title(f"Parallel-trends check{verdict}")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_placebo_distribution(
    est: ImpactEstimate,
    *,
    figsize: tuple[float, float] = (8.5, 3.8),
) -> Figure:
    """Where the treated unit's effect sits among placebo effects.

    Each placebo is a control unit analysed as if it had been treated. If the
    treated unit's bar is unremarkable within that spread, the effect is noise.
    """
    apply_style()
    placebos: pd.Series = est.diagnostics.get("placebo_effects", pd.Series(dtype=float))
    fig, ax = plt.subplots(figsize=figsize)
    if placebos.empty:
        ax.text(0.5, 0.5, "Placebo inference unavailable\n(too few donor units)",
                ha="center", va="center", transform=ax.transAxes, color=COLORS["muted"])
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    vals = placebos.to_numpy(dtype=float)
    ax.hist(vals, bins=max(6, min(16, len(vals))), color=COLORS["control"],
            edgecolor="white", label=f"placebo units (n={len(vals)})")
    ax.axvline(
        est.effect,
        color=COLORS["negative"] if est.effect < 0 else COLORS["effect"],
        linewidth=2.4,
        label=f"treated unit ({est.effect:+,.1f})",
    )
    ax.axvline(0, color=COLORS["muted"], linewidth=1.0)
    p = est.p_value
    title = "Placebo inference: treated effect vs donor placebos"
    if p is not None and np.isfinite(p):
        title += f"  |  rank p={p:.3f}"
    ax.set_title(title)
    ax.set_xlabel("Average post-period gap")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_weights(
    est: ImpactEstimate,
    *,
    top_n: int = 10,
    figsize: tuple[float, float] = (7.0, 3.6),
) -> Figure:
    """Donor weights behind the synthetic control."""
    apply_style()
    weights: pd.Series = est.diagnostics.get("weights", pd.Series(dtype=float))
    fig, ax = plt.subplots(figsize=figsize)
    if weights.empty:
        ax.text(0.5, 0.5, "No donor weights for this method",
                ha="center", va="center", transform=ax.transAxes, color=COLORS["muted"])
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    top = weights.sort_values(ascending=True).tail(top_n)
    ax.barh(top.index.astype(str), top.to_numpy(dtype=float), color=COLORS["observed"])
    for name, value in top.items():
        ax.annotate(
            f"{value:.0%}",
            xy=(value, str(name)),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, max(float(top.max()) * 1.2, 0.1))
    ax.set_xlabel("Weight in the synthetic control")
    ax.set_title("Which control units build the counterfactual")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_panel_overview(
    ds: PanelDataset,
    *,
    outcome_label: str = "Outcome",
    figsize: tuple[float, float] = (9.5, 4.4),
) -> Figure:
    """All units at once: treated unit highlighted against the donor pool."""
    apply_style()
    wide = ds.wide()
    fig, ax = plt.subplots(figsize=figsize)
    for col in wide.columns:
        if str(col) == ds.treated_unit:
            continue
        ax.plot(wide.index, wide[col], color=COLORS["control"], linewidth=0.9, alpha=0.9, zorder=1)
    ax.plot([], [], color=COLORS["control"], linewidth=1.2,
            label=f"control units (n={ds.n_controls})")
    ax.plot(
        wide.index,
        wide[ds.treated_unit],
        color=COLORS["observed"],
        linewidth=2.2,
        zorder=3,
        label=f"treated: {ds.treated_unit}",
    )
    _mark_intervention(ax, ds.intervention_date)
    _format_dates(ax, wide.index)
    ax.set_ylabel(outcome_label)
    ax.set_title("Panel overview")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    return fig


def plot_method_comparison(
    estimates: dict[str, ImpactEstimate],
    *,
    truth: float | None = None,
    relative: bool = False,
    figsize: tuple[float, float] = (8.5, 4.0),
) -> Figure:
    """Forest plot of every method's estimate and interval side by side.

    Disagreement between methods is information, not an embarrassment: it tells
    the reader how much of the headline number is driven by the modelling choice
    rather than by the data.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    if not estimates:
        ax.text(0.5, 0.5, "No estimates to compare", ha="center", va="center",
                transform=ax.transAxes, color=COLORS["muted"])
        ax.set_axis_off()
        return fig

    items = list(estimates.items())
    labels, centers, los, his = [], [], [], []
    for _, est in items:
        if relative:
            if est.relative_effect is None or est.relative_interval is None:
                continue
            centers.append(est.relative_effect * 100)
            los.append(est.relative_interval.lower * 100)
            his.append(est.relative_interval.upper * 100)
        else:
            centers.append(est.effect)
            los.append(est.interval.lower)
            his.append(est.interval.upper)
        labels.append(est.method.replace(" (two-way FE)", "").replace(
            " (CausalImpact-style)", ""))

    y = np.arange(len(labels))[::-1]
    for yi, c, lo, hi, (name, _) in zip(y, centers, los, his, items, strict=False):
        is_naive = name == "naive"
        color = COLORS["muted"] if is_naive else COLORS["observed"]
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=2.0,
                alpha=0.55 if is_naive else 1.0)
        ax.plot([c], [yi], "o", color=color, markersize=7,
                alpha=0.55 if is_naive else 1.0)

    ax.axvline(0, color="#222222", linewidth=1.0, label="no effect")
    if truth is not None:
        tv = truth * 100 if relative else truth
        ax.axvline(tv, color=COLORS["effect"], linestyle="--", linewidth=1.6,
                   label=f"ground truth ({tv:+,.3g})")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Relative effect (%)" if relative else "Effect per period")
    ax.set_title("Method comparison with uncertainty intervals")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_recovery_summary(
    summary: pd.DataFrame,
    *,
    level: float = 0.95,
    figsize: tuple[float, float] = (9.5, 4.0),
) -> Figure:
    """Validation scoreboard: interval coverage and RMSE per estimator.

    The left panel is the one that matters. A bar short of the dashed line means
    that estimator's intervals are too narrow, i.e. it claims more certainty
    than it has earned.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    if summary.empty:
        for ax in axes:
            ax.set_axis_off()
        axes[0].text(0.5, 0.5, "No validation results", ha="center", va="center",
                     transform=axes[0].transAxes, color=COLORS["muted"])
        return fig

    labels = [str(v) for v in summary.get("label", summary.index)]
    short = [
        lbl.replace("Difference-in-Differences", "DiD")
        .replace("Synthetic Control", "SCM")
        .replace("Bayesian Counterfactual", "Bayesian")
        .replace("Interrupted Time Series", "ITS")
        .replace("Naive pre/post (benchmark)", "Naive")
        for lbl in labels
    ]
    x = np.arange(len(short))

    cov = summary["coverage"].to_numpy(dtype=float)
    colors = [COLORS["effect"] if c >= level - 0.10 else COLORS["negative"] for c in cov]
    axes[0].bar(x, cov, color=colors, width=0.62)
    axes[0].axhline(level, color="#222222", linestyle="--", linewidth=1.4,
                    label=f"nominal {level:.0%}")
    for xi, c in zip(x, cov, strict=False):
        # Label inside any bar tall enough to hold one, above the rest. Placing
        # a near-nominal label above its bar would collide with the nominal
        # level line, which is the one annotation that must stay readable.
        inside = c >= 0.15
        axes[0].annotate(
            f"{c:.0%}",
            xy=(xi, c),
            xytext=(0, -14) if inside else (0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="white" if inside else "#222222",
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short, rotation=20, ha="right")
    axes[0].set_ylim(0, 1.18)
    axes[0].set_ylabel("Interval coverage")
    axes[0].set_title("Do the intervals keep their promise?")
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="x", visible=False)

    rmse = summary["rmse"].to_numpy(dtype=float)
    axes[1].bar(x, rmse, color=COLORS["observed"], width=0.62)
    for xi, r in zip(x, rmse, strict=False):
        axes[1].annotate(f"{r:,.3g}", xy=(xi, r), xytext=(0, 4),
                         textcoords="offset points", ha="center", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short, rotation=20, ha="right")
    axes[1].set_ylabel("RMSE vs ground truth")
    axes[1].set_title("How far off is the point estimate?")
    axes[1].grid(axis="x", visible=False)

    fig.tight_layout()
    return fig
