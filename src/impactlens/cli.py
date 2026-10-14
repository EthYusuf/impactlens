"""Command line interface.

Three subcommands:

``analyse``
    Run the estimators on a CSV and print the results; optionally write a PDF.
``validate``
    Run the Monte Carlo recovery study and print the scoreboard.
``demo``
    Generate a synthetic case, analyse it, and write report plus figures. This
    is what produces the artefacts shown in the README.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .dataset import PanelDataset, SchemaError, load_panel
from .estimators import ESTIMATOR_LABELS, get_estimator, requires_controls, run_all
from .reporting import build_pdf_report, narrative
from .simulate import ScenarioConfig, generate_panel
from .validation import run_recovery_study

_RULE = "=" * 78


def _print_estimate(est, *, outcome_name: str) -> None:
    print()
    print(est.summary())
    print()
    print("Reading:")
    for line in narrative.headline(est, outcome_name=outcome_name).split(". "):
        if line.strip():
            print(f"  {line.strip().rstrip('.')}.")


def _cmd_analyse(args: argparse.Namespace) -> int:
    try:
        ds = load_panel(
            args.csv,
            treated_unit=args.treated,
            intervention_date=args.date,
            unit_col=args.unit_col,
            date_col=args.date_col,
            outcome_col=args.outcome_col,
        )
    except SchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError:
        print(f"error: no such file: {args.csv}", file=sys.stderr)
        return 2

    print(_RULE)
    print(f"ImpactLens {__version__}  |  {Path(args.csv).name}")
    print(_RULE)
    q = ds.quality_report()
    print(
        f"treated={ds.treated_unit}  controls={q['n_controls']}  "
        f"periods={q['n_periods']} ({q['n_pre']} pre / {q['n_post']} post)  "
        f"freq={q['frequency']}"
    )
    for w in ds.warnings():
        print(f"  ! {w}")

    if args.method == "all":
        results = run_all(ds, level=args.level)
        if not results:
            print("error: no estimator could run on this data", file=sys.stderr)
            return 1
        for name, est in results.items():
            print()
            print(f"--- {ESTIMATOR_LABELS.get(name, name)} ---")
            _print_estimate(est, outcome_name=args.outcome_name)
        note = narrative.comparison_note(results)
        if note:
            print()
            print("Agreement across methods:")
            print(f"  {note}")
        headline_name = next(
            (n for n in ("synthetic_control", "bayesian", "did", "its") if n in results),
            next(iter(results)),
        )
        headline_est = results[headline_name]
    else:
        if requires_controls(args.method) and not ds.has_controls:
            print(
                f"error: {args.method} requires at least one control unit; "
                "try --method its",
                file=sys.stderr,
            )
            return 2
        headline_est = get_estimator(args.method)(ds, level=args.level)
        results = {args.method: headline_est}
        _print_estimate(headline_est, outcome_name=args.outcome_name)

    if args.pdf:
        out = build_pdf_report(
            args.pdf,
            headline_est,
            ds,
            project_name=args.project or Path(args.csv).stem,
            outcome_name=args.outcome_name,
            all_estimates=results if len(results) > 1 else None,
            author=args.author,
        )
        print()
        print(f"report written: {out}")

    if args.csv_out:
        table = pd.DataFrame({k: v.to_dict() for k, v in results.items()}).T
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv_out)
        print(f"results written: {args.csv_out}")

    print()
    print(_RULE)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    cfg = ScenarioConfig(
        true_effect=args.true_effect,
        n_controls=args.n_controls,
        market_shock=args.market_shock,
    )
    print(_RULE)
    print(
        f"Recovery study: {args.n} replications, true relative effect "
        f"{args.true_effect:+.0%}, {args.n_controls} controls, "
        f"market shock {args.market_shock:g}"
    )
    print(_RULE)
    result = run_recovery_study(cfg, n_replications=args.n, level=args.level, progress=True)
    if result.summary.empty:
        print("error: no estimator completed any replication", file=sys.stderr)
        return 1

    cols = ["label", "n_runs", "true_effect", "mean_estimate", "bias", "rmse",
            "coverage", "power"]
    print()
    print(result.summary.loc[:, [c for c in cols if c in result.summary.columns]].to_string(
        float_format=lambda v: f"{v:,.4g}"
    ))
    print()
    print("Verdict (is the interval honest?):")
    print(result.verdict().to_string(float_format=lambda v: f"{v:,.4g}"))
    print()
    print(
        f"coverage should be near {args.level:.0%}. Materially below means the "
        "intervals are too narrow and that estimator is overconfident."
    )

    if args.csv_out:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        result.summary.to_csv(args.csv_out)
        print(f"summary written: {args.csv_out}")
    print(_RULE)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    panel = generate_panel(
        true_effect=args.true_effect,
        n_controls=args.n_controls,
        seed=args.seed,
    )
    ds: PanelDataset = panel.to_dataset()
    csv_path = outdir / "demo_panel.csv"
    panel.frame.to_csv(csv_path, index=False)

    print(_RULE)
    print(f"ImpactLens demo  |  ground-truth effect inserted: {panel.true_average_effect:+,.2f} "
          f"per period ({panel.true_relative_effect:+.1%})")
    print(_RULE)

    results = run_all(ds, level=args.level)
    for name, est in results.items():
        err = est.effect - panel.true_average_effect
        covered = est.interval.contains(panel.true_average_effect)
        print(
            f"{ESTIMATOR_LABELS.get(name, name):<30} "
            f"{est.effect:+8.2f}  "
            f"[{est.interval.lower:+8.2f}, {est.interval.upper:+8.2f}]  "
            f"err={err:+7.2f}  truth in interval: {'yes' if covered else 'NO'}"
        )

    headline_name = next(
        (n for n in ("synthetic_control", "bayesian", "did") if n in results),
        next(iter(results)),
    )
    headline_est = results[headline_name]

    validation_summary = None
    if args.validate:
        print()
        print(f"running recovery study ({args.validate} replications)...")
        validation_summary = run_recovery_study(
            ScenarioConfig(true_effect=args.true_effect, n_controls=args.n_controls),
            n_replications=args.validate,
            level=args.level,
            progress=True,
        ).summary

    pdf_path = build_pdf_report(
        outdir / "impact_report.pdf",
        headline_est,
        ds,
        project_name=args.project,
        outcome_name=args.outcome_name,
        all_estimates=results,
        validation_summary=validation_summary,
        author=args.author,
    )
    print()
    print(f"panel written : {csv_path}")
    print(f"report written: {pdf_path}")
    print(_RULE)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="impactlens",
        description="Measure whether an intervention actually worked, with honest "
        "uncertainty intervals.",
    )
    parser.add_argument("--version", action="version", version=f"impactlens {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # --------------------------------------------------------------- analyse
    p = sub.add_parser("analyse", aliases=["analyze"], help="analyse a CSV panel")
    p.add_argument("csv", help="long-format CSV with unit, date and outcome columns")
    p.add_argument("--treated", required=True, help="name of the treated unit")
    p.add_argument("--date", required=True, help="intervention date (YYYY-MM-DD)")
    p.add_argument(
        "--method",
        default="all",
        choices=["all", "did", "synthetic_control", "bayesian", "its", "naive"],
        help="estimator to use (default: all applicable)",
    )
    p.add_argument("--level", type=float, default=0.95, help="interval level (default 0.95)")
    p.add_argument("--unit-col", default="unit")
    p.add_argument("--date-col", default="date")
    p.add_argument("--outcome-col", default="y")
    p.add_argument("--outcome-name", default="the outcome",
                   help="human-readable outcome name for the report, e.g. 'appointments'")
    p.add_argument("--project", default="", help="project name shown on the report cover")
    p.add_argument("--author", default="", help="author name shown in the report footer")
    p.add_argument("--pdf", default=None, help="write a PDF report to this path")
    p.add_argument("--csv-out", default=None, help="write the results table to this path")
    p.set_defaults(func=_cmd_analyse)

    # -------------------------------------------------------------- validate
    p = sub.add_parser("validate", help="audit the estimators against a known effect")
    p.add_argument("-n", type=int, default=100, help="replications (default 100)")
    p.add_argument("--true-effect", type=float, default=0.12,
                   help="relative effect to insert (default 0.12)")
    p.add_argument("--n-controls", type=int, default=8)
    p.add_argument("--market-shock", type=float, default=26.0,
                   help="size of the confounding market-wide surge (default 26)")
    p.add_argument("--level", type=float, default=0.95)
    p.add_argument("--csv-out", default=None)
    p.set_defaults(func=_cmd_validate)

    # ------------------------------------------------------------------ demo
    p = sub.add_parser("demo", help="generate a synthetic case and report on it")
    p.add_argument("--outdir", default="outputs", help="output directory (default outputs/)")
    p.add_argument("--true-effect", type=float, default=0.12)
    p.add_argument("--n-controls", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--level", type=float, default=0.95)
    p.add_argument("--validate", type=int, default=0,
                   help="also run N recovery replications and include them in the report")
    p.add_argument("--project", default="Flagship clinic booking system")
    p.add_argument("--outcome-name", default="weekly appointments")
    p.add_argument("--author", default="")
    p.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
