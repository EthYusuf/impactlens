"""Multi-page PDF report.

Built on matplotlib's ``PdfPages`` rather than an HTML-to-PDF toolchain, so the
only dependency is matplotlib and the report renders identically on any machine
with no system packages to install.

Page order is deliberate: the finding and its uncertainty come first, the
diagnostics that could invalidate it come second, and the assumptions come last
as a standing reminder that the number is conditional.
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from ..dataset import PanelDataset  # noqa: E402
from ..types import ImpactEstimate  # noqa: E402
from . import narrative  # noqa: E402
from .plots import (  # noqa: E402
    COLORS,
    apply_style,
    plot_counterfactual,
    plot_cumulative_effect,
    plot_event_study,
    plot_method_comparison,
    plot_panel_overview,
    plot_placebo_distribution,
    plot_pointwise_effect,
    plot_recovery_summary,
    plot_weights,
)

__all__ = ["build_pdf_report"]

PAGE = (8.27, 11.69)  # A4 portrait, inches


def _text_page(
    pdf: PdfPages,
    title: str,
    blocks: list[tuple[str, list[str]]],
    *,
    footer: str = "",
) -> None:
    """Render a page of headed prose."""
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    y = 0.95
    fig.text(0.08, y, title, fontsize=17, fontweight="bold", color=COLORS["observed"])
    y -= 0.012
    fig.add_artist(
        plt.Line2D([0.08, 0.92], [y, y], color=COLORS["counterfactual"], linewidth=2.0)
    )
    y -= 0.035

    for heading, paragraphs in blocks:
        if y < 0.10:
            break
        if heading:
            fig.text(0.08, y, heading, fontsize=11.5, fontweight="bold", color="#222222")
            y -= 0.026
        for para in paragraphs:
            wrapped = textwrap.fill(para, width=96)
            n_lines = wrapped.count("\n") + 1
            fig.text(0.08, y, wrapped, fontsize=9.3, va="top", color="#333333",
                     linespacing=1.55)
            y -= 0.0165 * n_lines + 0.014
            if y < 0.08:
                break
        y -= 0.012

    if footer:
        fig.text(0.08, 0.035, footer, fontsize=7.5, color=COLORS["muted"])
    pdf.savefig(fig)
    plt.close(fig)


def _figure_page(pdf: PdfPages, figures: list, title: str, footer: str = "") -> None:
    """Place one to three figures on an A4 page by re-rendering them as images."""
    import io

    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, title, fontsize=15, fontweight="bold", color=COLORS["observed"])
    fig.add_artist(
        plt.Line2D([0.08, 0.92], [0.945, 0.945], color=COLORS["counterfactual"], linewidth=2.0)
    )

    n = len(figures)
    top, bottom = 0.915, 0.06
    gap = 0.018
    height = (top - bottom - gap * (n - 1)) / n
    for i, src in enumerate(figures):
        buf = io.BytesIO()
        src.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(src)
        buf.seek(0)
        img = plt.imread(buf)
        y0 = top - (i + 1) * height - i * gap
        ax = fig.add_axes([0.06, y0, 0.88, height])
        ax.imshow(img)
        ax.set_axis_off()

    if footer:
        fig.text(0.08, 0.025, footer, fontsize=7.5, color=COLORS["muted"])
    pdf.savefig(fig)
    plt.close(fig)


def _table_page(
    pdf: PdfPages,
    title: str,
    frame: pd.DataFrame,
    *,
    caption: str = "",
    footer: str = "",
    float_fmt: str = "{:,.4g}",
) -> None:
    """Render a DataFrame as a clean table page."""
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, title, fontsize=15, fontweight="bold", color=COLORS["observed"])
    fig.add_artist(
        plt.Line2D([0.08, 0.92], [0.945, 0.945], color=COLORS["counterfactual"], linewidth=2.0)
    )
    y = 0.915
    if caption:
        wrapped = textwrap.fill(caption, width=100)
        fig.text(0.08, y, wrapped, fontsize=9.2, va="top", color="#333333", linespacing=1.5)
        y -= 0.02 * (wrapped.count("\n") + 1) + 0.025

    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(
                lambda v: "" if pd.isna(v) else float_fmt.format(v)
            )
        else:
            display[col] = display[col].astype(str)

    ax = fig.add_axes([0.06, max(y - 0.5, 0.08), 0.88, min(0.5, y - 0.08)])
    ax.set_axis_off()
    table = ax.table(
        cellText=display.to_numpy(),
        colLabels=[str(c) for c in display.columns],
        rowLabels=[str(i) for i in display.index],
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#e0e0e0")
        if row == 0:
            cell.set_facecolor(COLORS["observed"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f8fa")

    if footer:
        fig.text(0.08, 0.025, footer, fontsize=7.5, color=COLORS["muted"])
    pdf.savefig(fig)
    plt.close(fig)


def _kpi_page(
    pdf: PdfPages,
    est: ImpactEstimate,
    ds: PanelDataset,
    *,
    project: str,
    outcome_name: str,
    footer: str,
) -> None:
    """Cover page: the finding, the range, and the verdict."""
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")

    fig.text(0.08, 0.95, "I M P A C T L E N S", fontsize=10.5,
             color=COLORS["counterfactual"], fontweight="bold")
    fig.text(0.08, 0.905, "Impact Evaluation Report", fontsize=25, fontweight="bold",
             color=COLORS["observed"])
    fig.text(0.08, 0.875, project, fontsize=12.5, color="#444444")
    fig.add_artist(
        plt.Line2D([0.08, 0.92], [0.855, 0.855], color=COLORS["counterfactual"], linewidth=2.5)
    )

    # The three numbers that matter.
    verdict = "EFFECT DETECTED" if est.is_significant else "NO DETECTABLE EFFECT"
    verdict_color = (
        COLORS["effect"] if est.is_significant and est.effect > 0
        else COLORS["negative"] if est.is_significant
        else COLORS["muted"]
    )
    fig.text(0.08, 0.80, verdict, fontsize=13, fontweight="bold", color=verdict_color)

    rel_txt = "n/a"
    if est.relative_effect is not None:
        rel_txt = f"{est.relative_effect * 100:+.1f}%"
    cards = [
        ("Effect per period", f"{est.effect:+,.1f}", f"{int(est.interval.level*100)}% "
         f"interval {est.interval.lower:+,.1f} to {est.interval.upper:+,.1f}"),
        ("Relative lift", rel_txt, (
            f"{est.relative_interval.lower*100:+.1f}% to "
            f"{est.relative_interval.upper*100:+.1f}%"
            if est.relative_interval is not None else "no counterfactual base")),
        ("Cumulative effect", f"{est.cumulative_effect:+,.0f}"
         if est.cumulative_effect is not None else "n/a",
         f"over {est.n_post} post-intervention periods"),
    ]
    x0 = 0.08
    for i, (label, value, sub) in enumerate(cards):
        y0 = 0.715 - i * 0.075
        fig.text(x0, y0 + 0.028, " ".join(label.upper()), fontsize=7.6,
                 color=COLORS["muted"])
        fig.text(x0, y0 - 0.004, value, fontsize=21, fontweight="bold",
                 color=COLORS["observed"])
        fig.text(x0 + 0.30, y0 + 0.004, sub, fontsize=9, color="#555555", va="center")
        fig.add_artist(
            plt.Line2D([x0, 0.92], [y0 - 0.022, y0 - 0.022], color="#e4e4e4", linewidth=1.0)
        )

    meta = [
        ("Method", est.method),
        ("Inference", str(est.diagnostics.get("inference", "n/a"))),
        ("p-value", f"{est.p_value:.4f}" if est.p_value is not None else "n/a"),
        ("Treated unit", ds.treated_unit),
        ("Intervention date", f"{ds.intervention_date:%d %B %Y}"),
        ("Periods (pre / post)", f"{est.n_pre} / {est.n_post}"),
        ("Control units", str(ds.n_controls)),
        ("Outcome", outcome_name),
    ]
    fig.text(0.08, 0.455, "Analysis setup", fontsize=11.5, fontweight="bold", color="#222222")
    for i, (k, v) in enumerate(meta):
        y0 = 0.425 - i * 0.0235
        fig.text(0.08, y0, f"{k}", fontsize=9, color=COLORS["muted"])
        fig.text(0.34, y0, f"{v}", fontsize=9, color="#222222")

    n_warn = len(est.warnings)
    warn_txt = (
        f"{n_warn} caveat(s) were raised during the analysis and are listed in full on the "
        "next page. Read them before acting on the number above."
        if n_warn
        else "No automated caveats were raised. The untestable assumptions still apply."
    )
    fig.text(
        0.08, 0.195, textwrap.fill(warn_txt, width=92), fontsize=9.2, va="top",
        color=COLORS["negative"] if n_warn else "#333333", linespacing=1.5,
    )

    fig.text(0.08, 0.065, footer, fontsize=7.8, color=COLORS["muted"])
    fig.text(
        0.08, 0.035,
        "Causal estimates are conditional on their assumptions. This report states them "
        "rather than hiding them.",
        fontsize=7.8, color=COLORS["muted"], style="italic",
    )
    pdf.savefig(fig)
    plt.close(fig)


def build_pdf_report(
    path: str | Path,
    est: ImpactEstimate,
    ds: PanelDataset,
    *,
    project_name: str = "Untitled analysis",
    outcome_name: str = "the outcome",
    unit_label: str = "",
    all_estimates: dict[str, ImpactEstimate] | None = None,
    validation_summary: pd.DataFrame | None = None,
    author: str = "",
) -> Path:
    """Write the full report to ``path`` and return the path.

    Parameters
    ----------
    est:
        The headline estimate the report is built around.
    ds:
        The dataset it was computed from.
    all_estimates:
        Optional dict of every method that was run, for the comparison page.
    validation_summary:
        Optional recovery-study summary table, for the validation page. Including
        it is what lets a reader judge whether the machinery works at all.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    apply_style()

    stamp = datetime.now().strftime("%d %B %Y %H:%M")
    footer = f"ImpactLens - generated {stamp}" + (f" - {author}" if author else "")

    with PdfPages(path) as pdf:
        _kpi_page(pdf, est, ds, project=project_name, outcome_name=outcome_name, footer=footer)

        # ---------------------------------------------------- summary prose
        blocks: list[tuple[str, list[str]]] = [
            ("What we found", narrative.executive_summary(
                est, ds, outcome_name=outcome_name, unit_label=unit_label)),
        ]
        if all_estimates:
            note = narrative.comparison_note(all_estimates)
            if note:
                blocks.append(("Do the methods agree?", [note]))
        if est.warnings:
            blocks.append(("Caveats raised by the analysis", list(est.warnings)))
        _text_page(pdf, "Executive summary", blocks, footer=footer)

        # ------------------------------------------------------------ charts
        figs = [
            plot_counterfactual(
                est, intervention_date=ds.intervention_date, outcome_label=outcome_name
            ),
            plot_pointwise_effect(est, intervention_date=ds.intervention_date),
        ]
        _figure_page(pdf, figs, "The effect over time", footer=footer)

        figs2 = [
            plot_cumulative_effect(est, intervention_date=ds.intervention_date),
            plot_panel_overview(ds, outcome_label=outcome_name),
        ]
        _figure_page(pdf, figs2, "Accumulation and panel context", footer=footer)

        # ------------------------------------------------------- diagnostics
        diag_figs = []
        event = est.diagnostics.get("event_study")
        if event is None and ds.has_controls:
            from ..estimators import event_study as _event_study

            try:
                event = _event_study(ds)
            except Exception:  # pragma: no cover
                event = None
        if event is not None:
            diag_figs.append(plot_event_study(event))
        if not est.diagnostics.get("placebo_effects", pd.Series(dtype=float)).empty:
            diag_figs.append(plot_placebo_distribution(est))
        if not est.diagnostics.get("weights", pd.Series(dtype=float)).empty:
            diag_figs.append(plot_weights(est))
        if diag_figs:
            _figure_page(pdf, diag_figs[:3], "Diagnostics: can this result be trusted?",
                         footer=footer)

        # -------------------------------------------------- method comparison
        if all_estimates and len(all_estimates) > 1:
            _figure_page(
                pdf,
                [plot_method_comparison(all_estimates)],
                "Method comparison",
                footer=footer,
            )
            rows = {
                k: v.to_dict() for k, v in all_estimates.items()
            }
            table = pd.DataFrame(rows).T
            keep = [
                "effect", "ci_lower", "ci_upper", "relative_effect",
                "p_value", "significant", "n_warnings",
            ]
            table = table.loc[:, [c for c in keep if c in table.columns]]
            table.index = [all_estimates[k].method for k in table.index]
            _table_page(
                pdf,
                "Every method side by side",
                table,
                caption=(
                    "Where the methods agree, the finding is robust to the modelling "
                    "choice. Where they disagree, the headline number is an artefact of "
                    "assumptions and should not be quoted without explanation. The naive "
                    "row is a benchmark showing what a before/after dashboard would have "
                    "reported."
                ),
                footer=footer,
            )

        # ---------------------------------------------------------- validation
        if validation_summary is not None and not validation_summary.empty:
            _figure_page(
                pdf,
                [plot_recovery_summary(validation_summary, level=est.interval.level)],
                "Validation: does the tool recover a known effect?",
                footer=footer,
            )
            cols = [c for c in ("label", "n_runs", "true_effect", "mean_estimate",
                                "bias", "rmse", "coverage", "power")
                    if c in validation_summary.columns]
            _table_page(
                pdf,
                "Recovery study results",
                validation_summary.loc[:, cols],
                caption=(
                    "Each estimator was run on simulated data where the true effect was "
                    "inserted by us, so the correct answer is known. 'coverage' is the "
                    "share of runs whose interval contained the truth and should be close "
                    "to the nominal level; materially lower means the intervals are too "
                    "narrow. 'power' is the share of runs that detected the effect."
                ),
                footer=footer,
            )

        # --------------------------------------------------------- assumptions
        quality = ds.quality_report()
        data_lines = [
            f"{k.replace('_', ' ').capitalize()}: {v}"
            for k, v in quality.items()
            if not isinstance(v, list) or v
        ]
        _text_page(
            pdf,
            "Assumptions and limits",
            [
                (
                    "What must be true for this number to be valid",
                    narrative.assumption_block(est),
                ),
                (
                    "When this tool gives the wrong answer",
                    [
                        "The control units were affected by the intervention too. Spillover "
                        "shrinks the measured gap and the estimate understates the effect.",
                        "Something else changed at the same date and affected only the "
                        "treated unit. That change is indistinguishable from the "
                        "intervention and will be credited to it.",
                        "The pre-period is short. The baseline is then poorly characterised "
                        "and the counterfactual is close to a guess.",
                        "The treated unit was selected because it was already improving. "
                        "Selection on the outcome biases every method here upward.",
                    ],
                ),
                ("Dataset facts", data_lines),
            ],
            footer=footer,
        )

        meta = pdf.infodict()
        meta["Title"] = f"ImpactLens report - {project_name}"
        meta["Author"] = author or "ImpactLens"
        meta["Subject"] = f"Causal impact evaluation of {outcome_name}"
        meta["Keywords"] = "causal inference, impact evaluation, " + est.method
        meta["CreationDate"] = datetime.now()

    return path
