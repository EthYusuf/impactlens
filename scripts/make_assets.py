"""Regenerate every figure and table used in the README and the docs.

Run from the repository root::

    python scripts/make_assets.py                 # figures only (fast)
    python scripts/make_assets.py --replications 200   # also refresh validation

Writing the assets from a script rather than by hand means the numbers in the
README cannot drift away from the code that produced them. The recovery-study
summary is cached to ``docs/validation_summary.csv`` so the slow part only runs
when asked for.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from impactlens.dataset import PanelDataset  # noqa: E402
from impactlens.estimators import (  # noqa: E402
    estimate_naive_prepost,
    estimate_synthetic_control,
    event_study,
    run_all,
)
from impactlens.reporting.plots import (  # noqa: E402
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
from impactlens.reporting.report import build_pdf_report  # noqa: E402
from impactlens.simulate import ScenarioConfig, generate_panel  # noqa: E402
from impactlens.validation import run_recovery_study  # noqa: E402

OUT = ROOT / "docs" / "screenshots"
DOCS = ROOT / "docs"
DPI = 150


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def figure_the_problem(panel, ds) -> None:
    """The figure that states the problem: the same data, two answers.

    Left: what a dashboard shows -- appointments up, celebrate. Right: the
    control units moved too, and once that is removed most of the rise was
    never the intervention's doing.
    """
    apply_style()
    naive = estimate_naive_prepost(ds)
    scm = estimate_synthetic_control(ds)
    truth = panel.true_average_effect

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    # ---- left: the dashboard view
    ax = axes[0]
    obs = ds.treated_series
    ax.plot(obs.index, obs, color=COLORS["observed"], linewidth=2.0, label="appointments")
    pre_mean = float(obs[obs.index < ds.intervention_date].mean())
    post_mean = float(obs[obs.index >= ds.intervention_date].mean())
    ax.hlines(pre_mean, obs.index.min(), ds.intervention_date,
              color=COLORS["muted"], linestyle="--", linewidth=1.6)
    ax.hlines(post_mean, ds.intervention_date, obs.index.max(),
              color=COLORS["negative"], linestyle="--", linewidth=1.6)
    ax.axvline(ds.intervention_date, color=COLORS["intervention"], linestyle=":",
               linewidth=1.4)
    ax.annotate(
        f"+{naive.effect:,.0f} per week\n(+{naive.relative_effect*100:.0f}%)",
        xy=(0.62, 0.12), xycoords="axes fraction", fontsize=11, fontweight="bold",
        color=COLORS["negative"],
    )
    ax.set_title("What the dashboard reports")
    ax.set_ylabel("Weekly appointments")
    ax.legend(loc="upper left")

    # ---- right: the control units did the same thing
    ax = axes[1]
    wide = ds.wide()
    for col in wide.columns:
        if str(col) == ds.treated_unit:
            continue
        ax.plot(wide.index, wide[col], color=COLORS["control"], linewidth=0.9, zorder=1)
    ax.plot([], [], color=COLORS["control"], linewidth=1.2,
            label=f"{ds.n_controls} comparable clinics (no new system)")
    ax.plot(obs.index, obs, color=COLORS["observed"], linewidth=2.0, zorder=3,
            label="the clinic that installed it")
    ax.plot(scm.counterfactual.index, scm.counterfactual, color=COLORS["counterfactual"],
            linestyle="--", linewidth=1.8, zorder=2, label="counterfactual")
    ax.axvline(ds.intervention_date, color=COLORS["intervention"], linestyle=":",
               linewidth=1.4)
    ax.annotate(
        f"real effect: +{scm.effect:,.0f} per week\n"
        f"(ground truth +{truth:,.0f})",
        xy=(0.52, 0.12), xycoords="axes fraction", fontsize=11, fontweight="bold",
        color=COLORS["effect"],
    )
    ax.set_title("What actually happened")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"The same data. The dashboard says +{naive.effect:,.0f}/week; "
        f"the truth is +{truth:,.0f}/week.",
        fontsize=13, fontweight="bold", color=COLORS["observed"], y=1.02,
    )
    fig.tight_layout()
    _save(fig, "01_the_problem.png")


def figure_shock_sensitivity() -> None:
    """How each estimator responds as the confound grows.

    The naive line climbs with the market shock because it cannot tell the two
    apart. The control-based lines stay flat on the truth. This is the single
    most informative picture in the project.
    """
    apply_style()
    shocks = [0, 10, 20, 30, 40, 50]
    rows = []
    for shock in shocks:
        for seed in range(6):
            panel = generate_panel(ScenarioConfig(market_shock=float(shock), seed=500 + seed))
            ds_s = panel.to_dataset()
            truth = panel.true_average_effect
            results = run_all(ds_s)
            for name, est in results.items():
                rows.append({
                    "shock": shock, "estimator": name,
                    "error": est.effect - truth,
                })
    df = pd.DataFrame(rows)
    agg = df.groupby(["shock", "estimator"])["error"].mean().unstack()

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    styles = {
        "synthetic_control": (COLORS["effect"], "-", "Synthetic Control", 2.2),
        "did": (COLORS["observed"], "-", "Difference-in-Differences", 2.2),
        "bayesian": ("#6b5b95", "-", "Bayesian Counterfactual", 2.2),
        "its": ("#c2653b", "--", "Interrupted Time Series (no control)", 1.6),
        "naive": (COLORS["negative"], "--", "Naive pre/post", 2.0),
    }
    for name, (color, ls, label, lw) in styles.items():
        if name not in agg.columns:
            continue
        ax.plot(agg.index, agg[name], ls, color=color, linewidth=lw, marker="o",
                markersize=5, label=label)
    ax.axhline(0, color="#222222", linewidth=1.2)
    # Below the zero line on the right, where no series passes.
    ax.annotate("correct answer", xy=(shocks[-1], 0), xytext=(-82, -15),
                textcoords="offset points", fontsize=8.5, color="#222222")
    ax.set_xlabel("Size of the market-wide shock starting at the intervention date")
    ax.set_ylabel("Estimation error\n(estimate - true effect)")
    ax.set_title("Only the control-based estimators survive a confound")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    _save(fig, "02_confound_sensitivity.png")


def figure_pretrend_violation() -> None:
    """Side by side: the diagnostic passing, and the diagnostic catching a real problem."""
    apply_style()
    clean = generate_panel(ScenarioConfig(seed=7)).to_dataset()
    drift = generate_panel(ScenarioConfig(treated_pretrend=0.35, seed=7)).to_dataset()

    figs = []
    for ds_x in (clean, drift):
        figs.append(plot_event_study(event_study(ds_x)))

    # Stitch the two renders into one image so the contrast is immediate.
    import io

    fig = plt.figure(figsize=(12.6, 4.0))
    for i, src in enumerate(figs):
        buf = io.BytesIO()
        src.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(src)
        buf.seek(0)
        ax = fig.add_axes([0.005 + i * 0.5, 0.0, 0.49, 1.0])
        ax.imshow(plt.imread(buf))
        ax.set_axis_off()
        ax.set_title(
            "Clean data: assumption holds" if i == 0
            else "Treated unit was already diverging: assumption fails",
            fontsize=10.5, fontweight="bold",
            color=COLORS["effect"] if i == 0 else COLORS["negative"],
        )
    _save(fig, "05_pretrend_check.png")


def figure_naive_bias_distribution() -> None:
    """Distribution of estimation error across many simulated datasets."""
    apply_style()
    rows = []
    for seed in range(60):
        panel = generate_panel(ScenarioConfig(seed=900 + seed))
        ds_s = panel.to_dataset()
        truth = panel.true_average_effect
        for name, est in run_all(ds_s).items():
            rows.append({"estimator": name, "error": est.effect - truth})
    df = pd.DataFrame(rows)

    order = ["synthetic_control", "did", "bayesian", "its", "naive"]
    labels = ["Synthetic\nControl", "DiD", "Bayesian", "ITS\n(no control)", "Naive\npre/post"]
    data = [df.loc[df["estimator"] == n, "error"].to_numpy() for n in order]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": "white"})
    face = [COLORS["effect"], COLORS["observed"], "#6b5b95", "#c2653b", COLORS["negative"]]
    for patch, c in zip(bp["boxes"], face, strict=False):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    ax.axhline(0, color="#222222", linewidth=1.2, label="correct answer")
    ax.set_xticklabels(labels)
    ax.set_ylabel("Estimation error\n(estimate - true effect)")
    ax.set_title("Estimation error across 60 simulated datasets")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "06_error_distribution.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=0,
                        help="refresh the validation summary with N replications")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    print("generating the demo case...")
    panel = generate_panel(ScenarioConfig(seed=args.seed))
    ds: PanelDataset = panel.to_dataset()
    panel.frame.to_csv(ROOT / "examples" / "clinic_appointments.csv", index=False)
    print(f"  ground truth: {panel.true_average_effect:+.2f}/week "
          f"({panel.true_relative_effect:+.1%})")

    results = run_all(ds)
    scm = results["synthetic_control"]

    print("rendering figures...")
    figure_the_problem(panel, ds)
    figure_shock_sensitivity()

    _save(
        plot_counterfactual(scm, intervention_date=ds.intervention_date,
                            outcome_label="Weekly appointments"),
        "03_counterfactual.png",
    )
    _save(plot_method_comparison(results, truth=panel.true_average_effect),
          "04_method_comparison.png")
    figure_pretrend_violation()
    figure_naive_bias_distribution()
    _save(plot_placebo_distribution(scm), "07_placebo_inference.png")
    _save(plot_weights(scm), "08_donor_weights.png")
    _save(plot_pointwise_effect(scm, intervention_date=ds.intervention_date),
          "09_pointwise_effect.png")
    _save(plot_cumulative_effect(scm, intervention_date=ds.intervention_date),
          "10_cumulative_effect.png")
    _save(plot_panel_overview(ds, outcome_label="Weekly appointments"),
          "11_panel_overview.png")

    # ----------------------------------------------------------- validation
    summary_path = DOCS / "validation_summary.csv"
    summary = None
    if args.replications > 0:
        print(f"running the recovery study ({args.replications} replications)...")
        study = run_recovery_study(
            ScenarioConfig(seed=0), n_replications=args.replications, progress=True
        )
        summary = study.summary
        summary.to_csv(summary_path)
        print(f"  wrote {summary_path.relative_to(ROOT)}")
    elif summary_path.exists():
        summary = pd.read_csv(summary_path, index_col=0)
        print(f"using cached {summary_path.relative_to(ROOT)}")

    if summary is not None and not summary.empty:
        _save(plot_recovery_summary(summary), "12_validation_scoreboard.png")

    # ---------------------------------------------------------------- report
    print("building the sample PDF report...")
    pdf = build_pdf_report(
        DOCS / "sample_report.pdf",
        scm,
        ds,
        project_name="Flagship clinic booking system",
        outcome_name="weekly appointments",
        all_estimates=results,
        validation_summary=summary,
    )
    print(f"  wrote {pdf.relative_to(ROOT)} ({pdf.stat().st_size / 1024:,.0f} KB)")

    # Headline numbers, printed so the README can be checked against them.
    print("\nheadline numbers for the README:")
    print(f"  ground truth      : {panel.true_average_effect:+.2f}/week")
    for name, est in results.items():
        print(
            f"  {name:<18}: {est.effect:+7.2f}  "
            f"[{est.interval.lower:+7.2f}, {est.interval.upper:+7.2f}]  "
            f"err={est.effect - panel.true_average_effect:+6.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
