"""Five-minute tour of the library.

    python examples/quickstart.py

Walks through the whole argument: generate a case with a known answer, show the
naive reading failing, show the control-based estimators succeeding, interrogate
the diagnostics, and check that nothing is reported when nothing happened.
"""

from __future__ import annotations

from impactlens import (
    ScenarioConfig,
    estimate_did,
    estimate_naive_prepost,
    estimate_synthetic_control,
    event_study,
    generate_panel,
    headline,
    run_all,
)

RULE = "=" * 76


def main() -> None:
    # ------------------------------------------------------------------ 1
    print(RULE)
    print("1. A case where we know the right answer")
    print(RULE)
    panel = generate_panel(ScenarioConfig(true_effect=0.12, n_controls=8, seed=7))
    ds = panel.to_dataset()
    print(ds)
    print(f"Inserted effect : {panel.true_average_effect:+.2f} per week "
          f"({panel.true_relative_effect:+.1%})")
    print("The data also contains a market-wide surge starting on the same date.")
    print("That surge is the confound: it is not the intervention's doing.")

    # ------------------------------------------------------------------ 2
    print()
    print(RULE)
    print("2. What a dashboard would report")
    print(RULE)
    naive = estimate_naive_prepost(ds)
    print(f"Post mean - pre mean : {naive.effect:+.2f} per week")
    print(f"True effect          : {panel.true_average_effect:+.2f} per week")
    print(f"Overstated by        : {naive.effect - panel.true_average_effect:+.2f} "
          f"({naive.effect / panel.true_average_effect:.1f}x too large)")
    print(f"Its own warning      : {naive.warnings[0]}")

    # ------------------------------------------------------------------ 3
    print()
    print(RULE)
    print("3. What the control-based estimators find")
    print(RULE)
    for est in run_all(ds).values():
        covered = "yes" if est.interval.contains(panel.true_average_effect) else "NO"
        print(
            f"{est.method:<46} {est.effect:+7.2f}  "
            f"[{est.interval.lower:+7.2f}, {est.interval.upper:+7.2f}]  "
            f"truth inside: {covered}"
        )

    # ------------------------------------------------------------------ 4
    print()
    print(RULE)
    print("4. The sentence you would actually put in front of a client")
    print(RULE)
    scm = estimate_synthetic_control(ds)
    print(headline(scm, outcome_name="weekly appointments"))
    weights = scm.diagnostics["weights"]
    top = weights[weights > 0.01]
    print()
    print("The counterfactual is built from:")
    for unit, w in top.items():
        print(f"  {w:5.1%}  {unit}")

    # ------------------------------------------------------------------ 5
    print()
    print(RULE)
    print("5. Checking the assumption before believing the number")
    print(RULE)
    ev = event_study(ds)
    p = ev["pretrend_p_value"]
    print(f"Pre-trend slope : {ev['pretrend_slope']:+.4f} per week "
          f"(SE {ev['pretrend_slope_se']:.4f})")
    print(f"Slope test      : p = {p:.4f} -> "
          f"{'FAILS, DiD is biased here' if p < 0.05 else 'passes'}")

    print()
    print("Now break it on purpose: make the treated unit diverge beforehand.")
    drift = generate_panel(ScenarioConfig(treated_pretrend=0.35, seed=7)).to_dataset()
    ev_bad = event_study(drift)
    print(f"Pre-trend slope : {ev_bad['pretrend_slope']:+.4f} per week")
    print(f"Slope test      : p = {ev_bad['pretrend_p_value']:.4g} -> FAILS, as it should")
    bad = estimate_did(drift, n_boot=199)
    print()
    print("And the estimator says so rather than reporting a clean number:")
    for w in bad.warnings:
        if "drifting" in w:
            print(f"  ! {w}")

    # ------------------------------------------------------------------ 6
    print()
    print(RULE)
    print("6. The test that matters most: nothing happened")
    print(RULE)
    null_panel = generate_panel(ScenarioConfig(true_effect=0.0, seed=11))
    null_ds = null_panel.to_dataset()
    print("Same market-wide surge, but no intervention effect at all.")
    print()
    for est in run_all(null_ds).values():
        verdict = "claims an effect" if est.is_significant else "correctly finds nothing"
        print(f"{est.method:<46} {est.effect:+7.2f}  {verdict}")
    print()
    print("The two without a control group report the surge as an effect.")
    print("That is not a bug in the code -- it is what having no control group costs.")

    print()
    print(RULE)
    print("Next: `impactlens validate -n 250` audits all of this over 250 datasets.")
    print(RULE)


if __name__ == "__main__":
    main()
