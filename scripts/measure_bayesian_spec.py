"""Measure why the Bayesian estimator drops deterministic trend and seasonality.

The default in :func:`impactlens.estimators.estimate_bayesian_counterfactual`
turns the trend and Fourier terms **off** when control series are present. That
is an empirical decision, and the table this script prints is the evidence for
it -- quoted in ``docs/methodology.md``. Re-run it whenever the simulation or
the estimator changes, so the documented numbers cannot drift from the code.

    python scripts/measure_bayesian_spec.py --replications 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from impactlens.estimators.bayesian import estimate_bayesian_counterfactual  # noqa: E402
from impactlens.simulate import ScenarioConfig, generate_panel  # noqa: E402

SPECS: dict[str, dict] = {
    "controls only (the default)": dict(include_trend=False, n_harmonics=0),
    "+ deterministic trend": dict(include_trend=True, n_harmonics=0),
    "+ trend + 2 harmonics": dict(include_trend=True, n_harmonics=2),
    "+ 2 harmonics only": dict(include_trend=False, n_harmonics=2),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=120)
    parser.add_argument("--n-draws", type=int, default=1500)
    parser.add_argument("--base-seed", type=int, default=3000)
    parser.add_argument("--csv-out", default=str(ROOT / "docs" / "bayesian_spec.csv"))
    args = parser.parse_args()

    # Generate the panels once and reuse them across specifications, so the
    # comparison is not contaminated by different random draws.
    panels = [
        generate_panel(ScenarioConfig(seed=args.base_seed + i))
        for i in range(args.replications)
    ]
    datasets = [(p.to_dataset(), p.true_average_effect) for p in panels]

    rows = []
    for label, kw in SPECS.items():
        errors, covered, widths = [], [], []
        for ds, truth in datasets:
            est = estimate_bayesian_counterfactual(ds, n_draws=args.n_draws, **kw)
            errors.append(est.effect - truth)
            covered.append(est.interval.contains(truth))
            widths.append(est.interval.width)
        errors = np.asarray(errors)
        rows.append(
            {
                "specification": label,
                "bias": float(errors.mean()),
                "rmse": float(np.sqrt((errors**2).mean())),
                "coverage": float(np.mean(covered)),
                "mean_ci_width": float(np.mean(widths)),
            }
        )
        print(f"  measured: {label}")

    table = pd.DataFrame(rows).set_index("specification")
    print()
    print(
        f"{args.replications} replications, 8 control units, nominal 95% intervals"
    )
    print(table.to_string(float_format=lambda v: f"{v:,.3f}"))
    table.to_csv(args.csv_out)
    print(f"\nwrote {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
