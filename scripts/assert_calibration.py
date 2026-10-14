"""Fail the build if the estimators stop being honest.

The unit tests check that each estimator runs and behaves sensibly on a handful
of cases. They cannot check *calibration* — whether a 95% interval really
contains the truth 95% of the time — because that needs a sample of simulated
datasets, not a single one.

This script reads the summary written by ``impactlens validate --csv-out`` and
asserts the two properties the README claims:

1. **Coverage.** The control-based estimators' intervals must keep their
   promise. A regression that narrows them would pass every unit test and still
   make the tool lie, so this is the binding check.
2. **Bias.** The point estimate must stay close to the inserted effect, scored
   relative to the size of that effect.

Usage::

    impactlens validate -n 60 --csv-out validation.csv
    python scripts/assert_calibration.py validation.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

# Estimators that are supposed to be calibrated. ITS and the naive benchmark are
# deliberately excluded: under a market-wide confound they are *expected* to
# fail, and the README reports that failure as a finding.
CALIBRATED = ("did", "synthetic_control", "bayesian")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", help="CSV written by `impactlens validate --csv-out`")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.85,
        help="coverage floor. Below the 0.95 nominal level to allow for Monte "
        "Carlo error at small replication counts, while still catching a real "
        "calibration regression (default: 0.85)",
    )
    parser.add_argument(
        "--max-relative-bias",
        type=float,
        default=0.25,
        help="maximum |bias| as a fraction of the true effect (default: 0.25)",
    )
    args = parser.parse_args(argv)

    summary = pd.read_csv(args.summary, index_col=0)
    print(summary.to_string(float_format=lambda v: f"{v:,.4g}"))
    print()

    failures: list[str] = []
    for name in CALIBRATED:
        if name not in summary.index:
            failures.append(f"{name}: did not run")
            continue
        row = summary.loc[name]
        if row["coverage"] < args.min_coverage:
            failures.append(
                f"{name}: coverage {row['coverage']:.3f} below "
                f"{args.min_coverage:.2f} - the intervals have become overconfident"
            )
        true_effect = abs(float(row["true_effect"]))
        if true_effect > 0 and abs(row["bias"]) > args.max_relative_bias * true_effect:
            failures.append(
                f"{name}: bias {row['bias']:+.3f} exceeds "
                f"{args.max_relative_bias:.0%} of the true effect {true_effect:.3f}"
            )

    # The naive benchmark must stay *bad*. If it ever looks calibrated, the
    # simulation has lost its confound and the whole comparison is meaningless.
    if "naive" in summary.index and summary.loc["naive", "coverage"] > 0.5:
        failures.append(
            f"naive: coverage {summary.loc['naive', 'coverage']:.3f} is suspiciously "
            "high - the simulation's confound may have stopped working, which would "
            "invalidate the comparison the project is built on"
        )

    if failures:
        print("CALIBRATION FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("All control-based estimators remain calibrated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
