"""Causal estimators, plus a registry so callers can select one by name."""

from __future__ import annotations

from collections.abc import Callable

from ..dataset import PanelDataset
from ..types import ImpactEstimate
from .bayesian import estimate_bayesian_counterfactual
from .did import estimate_did, event_study
from .its import estimate_its
from .naive import estimate_naive_prepost
from .synthetic_control import estimate_synthetic_control, fit_weights

__all__ = [
    "ESTIMATORS",
    "ESTIMATOR_LABELS",
    "estimate_bayesian_counterfactual",
    "estimate_did",
    "estimate_its",
    "estimate_naive_prepost",
    "estimate_synthetic_control",
    "event_study",
    "fit_weights",
    "get_estimator",
    "requires_controls",
    "run_all",
]

#: Estimator registry, keyed by the short name used in the CLI and dashboard.
ESTIMATORS: dict[str, Callable[..., ImpactEstimate]] = {
    "did": estimate_did,
    "synthetic_control": estimate_synthetic_control,
    "bayesian": estimate_bayesian_counterfactual,
    "its": estimate_its,
    "naive": estimate_naive_prepost,
}

ESTIMATOR_LABELS: dict[str, str] = {
    "did": "Difference-in-Differences",
    "synthetic_control": "Synthetic Control",
    "bayesian": "Bayesian Counterfactual",
    "its": "Interrupted Time Series",
    "naive": "Naive pre/post (benchmark)",
}

#: Estimators that cannot run without at least one control unit.
_NEEDS_CONTROLS = {"did", "synthetic_control"}


def requires_controls(name: str) -> bool:
    return name in _NEEDS_CONTROLS


def get_estimator(name: str) -> Callable[..., ImpactEstimate]:
    """Look up an estimator by short name."""
    try:
        return ESTIMATORS[name]
    except KeyError:
        raise KeyError(
            f"unknown estimator {name!r}; available: {', '.join(sorted(ESTIMATORS))}"
        ) from None


def run_all(
    ds: PanelDataset,
    *,
    level: float = 0.95,
    include_naive: bool = True,
    skip_on_error: bool = True,
    **kwargs,
) -> dict[str, ImpactEstimate]:
    """Run every applicable estimator on a dataset.

    Estimators that need control units are skipped when none exist. When
    ``skip_on_error`` is True, an estimator that raises is omitted rather than
    aborting the whole run -- useful in the dashboard, where a single bad fit
    should not blank the page.
    """
    results: dict[str, ImpactEstimate] = {}
    for name, fn in ESTIMATORS.items():
        if name == "naive" and not include_naive:
            continue
        if requires_controls(name) and not ds.has_controls:
            continue
        try:
            results[name] = fn(ds, level=level, **_filter_kwargs(fn, kwargs))
        except Exception:
            if not skip_on_error:
                raise
    return results


def _filter_kwargs(fn: Callable[..., ImpactEstimate], kwargs: dict) -> dict:
    """Pass through only the keyword arguments this estimator accepts."""
    import inspect

    allowed = set(inspect.signature(fn).parameters)
    return {k: v for k, v in kwargs.items() if k in allowed}
