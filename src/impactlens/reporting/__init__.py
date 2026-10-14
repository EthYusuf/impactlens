"""Figures, prose and PDF output."""

from __future__ import annotations

from .narrative import assumption_block, comparison_note, executive_summary, headline
from .plots import (
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
from .report import build_pdf_report

__all__ = [
    "COLORS",
    "apply_style",
    "assumption_block",
    "build_pdf_report",
    "comparison_note",
    "executive_summary",
    "headline",
    "plot_counterfactual",
    "plot_cumulative_effect",
    "plot_event_study",
    "plot_method_comparison",
    "plot_panel_overview",
    "plot_placebo_distribution",
    "plot_pointwise_effect",
    "plot_recovery_summary",
    "plot_weights",
]
