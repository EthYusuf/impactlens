"""Turning an estimate into sentences a manager can act on.

The point of the project is that a decision-maker reads a number and acts on it.
So the summary has to say three things and never skip the third:

1. What the estimate is, in the units the business cares about.
2. How uncertain it is, as a range rather than a point.
3. What would have to be true for the number to be valid -- and whether any of
   those conditions look broken.

Phrasing is deliberately hedged where the statistics are hedged. An interval
that crosses zero is reported as "no detectable effect", never as a small
positive effect, because the data does not support a direction.
"""

from __future__ import annotations

import numpy as np

from ..dataset import PanelDataset
from ..types import ImpactEstimate

__all__ = ["headline", "executive_summary", "assumption_block", "comparison_note"]


def _fmt(value: float, unit: str = "") -> str:
    suffix = f" {unit}" if unit else ""
    return f"{value:,.1f}{suffix}"


def _pct(value: float) -> str:
    return f"{value * 100:+.1f}%"


def headline(est: ImpactEstimate, *, outcome_name: str = "the outcome") -> str:
    """One sentence, suitable for the top of a report or a Slack message."""
    if not est.is_significant:
        band = f"{_fmt(est.interval.lower)} to {_fmt(est.interval.upper)}"
        return (
            f"No detectable effect on {outcome_name}. The estimate is "
            f"{_fmt(est.effect)} per period, but the {int(est.interval.level * 100)}% "
            f"interval spans {band}, which includes zero. The data cannot distinguish "
            f"this intervention from no change."
        )

    verb = "increased" if est.effect > 0 else "decreased"
    rel = ""
    if est.relative_effect is not None and est.relative_interval is not None:
        rel = (
            f" ({_pct(est.relative_effect)}, interval "
            f"{_pct(est.relative_interval.lower)} to "
            f"{_pct(est.relative_interval.upper)})"
        )
    return (
        f"The intervention {verb} {outcome_name} by about {_fmt(abs(est.effect))} "
        f"per period{rel}. The {int(est.interval.level * 100)}% "
        f"{est.interval.kind} interval runs from {_fmt(est.interval.lower)} to "
        f"{_fmt(est.interval.upper)} and excludes zero."
    )


def executive_summary(
    est: ImpactEstimate,
    ds: PanelDataset | None = None,
    *,
    outcome_name: str = "the outcome",
    unit_label: str = "",
) -> list[str]:
    """Several short paragraphs: finding, scale, method, caveats."""
    paras: list[str] = [headline(est, outcome_name=outcome_name)]

    if est.cumulative_effect is not None and est.is_significant:
        paras.append(
            f"Over the {est.n_post} periods since the intervention, the cumulative "
            f"difference is approximately {_fmt(est.cumulative_effect, unit_label)}. "
            f"That total inherits all the uncertainty of the per-period estimate, so "
            f"it should be read as an order of magnitude rather than an exact figure."
        )

    method_note = {
        "Difference-in-Differences (two-way FE)": (
            "The estimate compares the change in the treated unit against the change in "
            "the control units over the same periods. Anything that moved every unit at "
            "once -- season, market, holidays -- is absorbed by the time fixed effects "
            "and therefore does not contaminate the estimate."
        ),
        "Synthetic Control": (
            "The counterfactual is a weighted blend of control units chosen to track the "
            "treated unit before the intervention. The reported interval comes from "
            "re-running the analysis on each control unit as a placebo, so it answers "
            "'how unusual is this gap among units that received nothing?'"
        ),
        "Bayesian Counterfactual (CausalImpact-style)": (
            "A Bayesian model was fitted to the pre-intervention period using the control "
            "series, then used to forecast what would have happened without the "
            "intervention. The interval is a credible interval from the posterior "
            "predictive distribution, so it carries both parameter and noise uncertainty."
        ),
        "Interrupted Time Series": (
            "With no control group, the counterfactual is the pre-intervention trend "
            "extended forward. This isolates a break at the intervention date but cannot "
            "rule out that something else broke at the same moment."
        ),
        "Naive pre/post comparison": (
            "This simply differences the two window means. It is included as a benchmark "
            "to show how much the seasonal and market movement would have inflated or "
            "deflated a naive reading."
        ),
    }.get(est.method)
    if method_note:
        paras.append(f"Method. {method_note}")

    if ds is not None:
        q = ds.quality_report()
        paras.append(
            f"Data. {q['n_periods']} periods of {q['frequency']} data from "
            f"{q['first_date']:%d %b %Y} to {q['last_date']:%d %b %Y}: "
            f"{q['n_pre']} before the intervention and {q['n_post']} after, with "
            f"{q['n_controls']} control unit(s) in the donor pool."
        )

    if est.warnings:
        paras.append(
            "Caveats. "
            + " ".join(w if w.endswith(".") else w + "." for w in est.warnings[:4])
        )
    else:
        paras.append(
            "No data-quality or assumption warnings were raised, which means the checks "
            "that could be run passed. It does not mean the untestable assumptions below "
            "are satisfied."
        )
    return paras


def assumption_block(est: ImpactEstimate) -> list[str]:
    """The assumptions the estimate rests on, verbatim from the estimator."""
    return list(est.assumptions)


def comparison_note(estimates: dict[str, ImpactEstimate]) -> str:
    """How much of the headline depends on the modelling choice.

    Compares the credible/confidence estimates against each other, and separately
    against the naive benchmark, which is the number a business would otherwise
    have reported.
    """
    serious = {k: v for k, v in estimates.items() if k != "naive"}
    if len(serious) < 2:
        return ""

    effects = np.array([e.effect for e in serious.values()], dtype=float)
    spread = float(effects.max() - effects.min())
    widest = max(e.interval.width for e in serious.values())
    agree = len({e.direction for e in serious.values()}) == 1

    lines: list[str] = []
    if agree and spread <= 0.5 * widest:
        lines.append(
            f"All {len(serious)} control-based methods agree in direction, and the spread "
            f"between their point estimates ({_fmt(spread)}) is small relative to their own "
            f"uncertainty. The conclusion is not an artefact of one modelling choice."
        )
    elif agree:
        lines.append(
            f"The methods agree in direction but their point estimates differ by "
            f"{_fmt(spread)}, which is comparable to their uncertainty intervals. Treat "
            f"the direction as the finding and the magnitude as approximate."
        )
    else:
        lines.append(
            "The methods disagree on direction. That is a red flag: the result depends on "
            "the modelling assumptions rather than on the data. Do not report a single "
            "headline number until the disagreement is understood."
        )

    naive = estimates.get("naive")
    if naive is not None:
        best = next(iter(serious.values()))
        gap = naive.effect - best.effect
        if abs(best.effect) > 1e-9:
            factor = naive.effect / best.effect
            if np.isfinite(factor) and factor > 1.15:
                lines.append(
                    f"The naive before/after comparison reports {_fmt(naive.effect)} per "
                    f"period -- {factor:.1f}x the control-based estimate of "
                    f"{_fmt(best.effect)}. The extra {_fmt(gap)} is movement the control "
                    f"units experienced too, so it was never attributable to the "
                    f"intervention."
                )
            elif np.isfinite(factor) and 0 < factor < 0.85:
                lines.append(
                    f"The naive before/after comparison understates the effect, reporting "
                    f"{_fmt(naive.effect)} against the control-based "
                    f"{_fmt(best.effect)}. A declining market masked part of the gain."
                )
    return " ".join(lines)
