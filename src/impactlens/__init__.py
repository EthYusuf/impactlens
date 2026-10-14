"""ImpactLens -- did the intervention actually work?

A business installs a booking system, a chatbot, a new website. Appointments go
up. The dashboard shows the increase and everyone celebrates. But the appointment
count would have gone up anyway -- it was spring, a competitor closed, a holiday
fell in the window. The dashboard cannot tell the difference. ImpactLens can.

Quick start
-----------
>>> from impactlens import generate_panel, estimate_synthetic_control
>>> panel = generate_panel(true_effect=0.12, seed=7)
>>> ds = panel.to_dataset()
>>> est = estimate_synthetic_control(ds)
>>> est.is_significant in (True, False)
True

Everything returns an :class:`~impactlens.types.ImpactEstimate`, which always
carries an uncertainty interval, the assumptions it rests on, and any warnings
raised while computing it.
"""

from __future__ import annotations

from .dataset import PanelDataset, SchemaError, load_panel
from .estimators import (
    ESTIMATOR_LABELS,
    ESTIMATORS,
    estimate_bayesian_counterfactual,
    estimate_did,
    estimate_its,
    estimate_naive_prepost,
    estimate_synthetic_control,
    event_study,
    get_estimator,
    run_all,
)
from .reporting import build_pdf_report, executive_summary, headline
from .simulate import ScenarioConfig, SyntheticPanel, generate_panel
from .types import ImpactEstimate, Interval
from .validation import RecoveryResult, false_positive_study, run_recovery_study

__version__ = "0.1.0"

__all__ = [
    "ESTIMATORS",
    "ESTIMATOR_LABELS",
    "ImpactEstimate",
    "Interval",
    "PanelDataset",
    "RecoveryResult",
    "ScenarioConfig",
    "SchemaError",
    "SyntheticPanel",
    "__version__",
    "build_pdf_report",
    "estimate_bayesian_counterfactual",
    "estimate_did",
    "estimate_its",
    "estimate_naive_prepost",
    "estimate_synthetic_control",
    "event_study",
    "executive_summary",
    "false_positive_study",
    "generate_panel",
    "get_estimator",
    "headline",
    "load_panel",
    "run_all",
    "run_recovery_study",
]
