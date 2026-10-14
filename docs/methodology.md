# Methodology

Four estimators, one question: **what would have happened without the intervention?**

Everything here is an attempt to construct that counterfactual. They differ in
what they are willing to assume in order to build it, and the right choice is
the one whose assumptions your data can actually support.

---

## 1. Difference-in-Differences (two-way fixed effects)

```
y_it = α_i + γ_t + τ·(treated_i × post_t) + ε_it
```

`τ` is the average treatment effect on the treated.

- **Unit fixed effects (`α_i`)** absorb permanent level differences. A clinic
  that always sees 300 patients a week and one that always sees 90 can be
  compared, because only their *changes* enter the estimate.
- **Time fixed effects (`γ_t`)** absorb anything that moved every unit in the
  same period — the season turning, a public holiday, a market-wide price
  change. This is the mechanism that defeats the confound a before/after
  comparison falls for.

**Identifying assumption: parallel trends.** Without the intervention, the
treated unit's outcome would have moved parallel to the controls'. Not *equal
to* — parallel.

**Inference.** Standard errors are clustered by unit, because observations within
a unit are serially correlated and treating each (unit, period) cell as
independent understates the variance severely. Cluster-robust inference is
asymptotic in the *number of clusters*, so with few units (the usual case) the
estimator switches to a **wild cluster bootstrap** with the null imposed
(Cameron, Gelbach & Miller, 2008) and records which method it used in
`diagnostics["inference"]`.

The bootstrap refits a fixed design thousands of times, so the design matrix is
factorised once and reused (`_ClusterOLS`). This is identical arithmetic to
`statsmodels` — verified to 1e-8 in
[`tests/test_estimators.py`](../tests/test_estimators.py) — and roughly 70x
faster, which is what makes the recovery study runnable.

**Validated behaviour** (250 replications, see [below](#validation)): bias
+0.17, RMSE 2.35 — the lowest of any estimator here — but *conservative*.
Coverage reaches 100% against a nominal 95%, and the mean interval is 33.1 wide
against synthetic control's 24.7. The wild cluster bootstrap with 9 clusters
deliberately errs wide.

Power is still 98% in that study, but that is because the inserted effect (≈24)
is large relative to the noise. The wide intervals are a real cost: on a smaller
effect, DiD is the first estimator here to lose the ability to detect it. The
trade is honest — it will rarely claim an effect that is not there.

---

## 2. Synthetic Control

Rather than assume the treated unit moves parallel to the controls, **build** a
comparison unit that tracks it:

```
minimise  Σ_pre (y_treated,t − Σ_j w_j · y_j,t)²
subject to  w_j ≥ 0,  Σ_j w_j = 1
```

The simplex constraint is the point. Non-negative weights summing to one prevent
extrapolation beyond the range of the donors, and the result is interpretable:
"62% control_03 + 38% control_07" is a comparison unit you can describe to a
client.

**Identifying assumptions.** A close pre-period fit (reported as
`pre_fit_ratio` — the fit error as a share of the treated unit's own variation,
with a warning above 0.5), no interference with the donors, and that the treated
unit lies within the donors' convex hull.

**Inference: in-space placebos** (Abadie, Diamond & Hainmueller, 2010). Re-run
the entire procedure pretending each control unit was treated. If the treated
unit's post/pre RMSPE ratio is unremarkable within that placebo distribution,
the effect is noise. The p-value is a rank statistic, so with `J` usable donors
the smallest attainable p-value is `1/(J+1)` — with 8 donors that is 0.111, and
the estimator **says so in a warning** rather than reporting a precision it
cannot deliver. Placebos whose own pre-fit is more than 5x worse than the
treated unit's are discarded, since their post-period gap reflects a failed
match rather than a shock.

The interval is labelled `kind="placebo"` because it is a permutation interval,
not a standard-error interval. Conflating the two would be dishonest.

**Validated behaviour:** the best overall balance — near-zero bias, lowest RMSE,
coverage near nominal, and enough power to be useful.

---

## 3. Bayesian Counterfactual (CausalImpact-style)

Google's CausalImpact fits a Bayesian structural time series model on the
pre-period using control series as regressors, then forecasts the
counterfactual. This implements the same idea with a **conjugate Bayesian linear
model** instead of a full BSTS state-space model.

```
y_t = x_t'β + ε_t,     ε_t ~ N(0, σ²)
β | σ² ~ N(0, σ²/λ · I),   σ² ~ InvGamma(a₀, b₀)
```

The posterior is analytic, so there is no MCMC, no convergence diagnostics, and
results are exactly reproducible from a seed. Posterior predictive draws
propagate both coefficient and observation uncertainty into the interval, and
the cumulative effect comes from the same draws.

### What is kept, and what is given up

| CausalImpact (BSTS) | This implementation |
|---|---|
| Control series as regressors | ✅ same |
| Stochastic local level (drifting) | ❌ deterministic trend, or none |
| Stochastic seasonal component | ❌ Fourier basis, or none |
| Posterior predictive intervals | ✅ same |
| MCMC sampling | Analytic posterior (faster, exactly reproducible) |

A deterministic trend cannot wander the way a stochastic local level can. On a
series whose level drifts *without the controls drifting with it*, this model
will be overconfident. That is stated in `assumptions`, and it is why a
**pre-period back-test** runs on every fit.

### The specification default is empirical, not cosmetic

`include_trend` and `n_harmonics` default to `None`, which resolves to **off
when controls are present** and **on when they are not**. That is not a style
choice. Measured over 120 replications of the packaged simulation with 8
controls and nominal 95% intervals
([`scripts/measure_bayesian_spec.py`](../scripts/measure_bayesian_spec.py)):

| Specification | bias | RMSE | coverage | mean CI width |
|---|---:|---:|---:|---:|
| **controls only (the default)** | **+0.58** | **2.60** | **92.5%** | 9.4 |
| + deterministic trend | −0.02 | 3.50 | 90.8% | 12.5 |
| + 2 harmonics only | +7.03 | 9.29 | 72.5% | 21.8 |
| + trend + 2 harmonics | +12.25 | 13.74 | **49.2%** | 24.3 |

The donors already carry the trend and the season. Adding deterministic terms on
top makes them near-collinear with the donors, and the fitted seasonal amplitude
then mis-extrapolates past the intervention date. **Coverage collapsing to 49%
means the intervals stop being intervals** — a nominal 95% range that contains
the truth half the time is worse than no range at all.

Adding the trend alone is nearly unbiased but noisier (RMSE 3.50 vs 2.60) and
slightly worse-covered, so the default drops it too. The seasonal harmonics are
the real culprit. Re-run the script if the simulation or the estimator changes,
so the numbers in this table cannot drift from the code.

### Pre-period back-test

Every fit holds out the last 20% of the pre-period, refits on the earlier part,
and evaluates the holdout *as if the intervention had happened there*. Nothing
happened there, so a credible interval excluding zero is a **false positive** and
proof the model is mis-specified. When that happens the estimate carries a
warning saying the headline number is not credible.

---

## 4. Interrupted Time Series

The fallback when there is no control group at all. Segmented regression:

```
y_t = β₀ + β₁·t + β₂·post_t + β₃·(t − T₀)·post_t + seasonality + ε_t
```

`β₂` is the level jump at the intervention; `β₃` is the change in slope
afterwards. Both are reported, because quoting only the level jump when the
slope also changed is a standard way to misstate an effect. The average effect
over the post window combines them.

Standard errors use a **Newey-West HAC** covariance, since time-series residuals
are autocorrelated.

**This is the weakest estimator in the package and it says so in its first
warning.** With no control group, anything else that changed at the same date —
a competitor closing, a holiday, a price change, a news cycle — is absorbed into
the "effect". The validation table below quantifies exactly how badly: under a
market-wide shock, ITS reports the shock as the intervention's effect, with
0% interval coverage.

---

## The naive benchmark

`estimate_naive_prepost` differences the post-period mean against the
pre-period mean. It is **not a serious estimator** and its own warning says not
to quote it. It ships because showing it next to a control-based estimate is the
fastest way to demonstrate the bias that motivates the whole project — and
because it is what most business dashboards actually report.

---

## Diagnostics

### Parallel-trends check

The textbook event study interacts a treated indicator with a dummy per relative
period inside a two-way FE model. **With a single treated unit — the normal case
for "did our new booking system work?" — that test is not well posed.** Each
interaction dummy is identified by exactly one observation, so each coefficient
is just that cell's residual, and a joint F-test with a cluster-robust covariance
whose rank is capped at the number of units rejects far too often. On data where
parallel trends holds *by construction*, that formulation returned p = 5×10⁻⁴.

So `event_study()` tests the object that actually has to be flat. With one
treated unit and a balanced panel, two-way FE DiD is algebraically the change in
the treated unit minus the change in the control mean, so the relevant series is
the **gap**:

```
gap_t = y_treated,t − mean_j y_control_j,t
```

The gap is regressed on time over the pre-period with Newey-West standard
errors, and the slope is tested against zero. This is correctly sized with one
treated unit, and it tests the assumption DiD needs: not that the gap is zero,
but that it is not **drifting**.

A significant pre-trend slope triggers a warning that names the problem
(selection on trend) and points at synthetic control, which fits the pre-period
rather than assuming it away.

### Other checks

| Check | Fires when | Means |
|---|---|---|
| Pre-fit quality (SCM) | pre-period error > 50% of treated variation | counterfactual is not tracking; estimate unreliable |
| Donor collapse (SCM) | one donor holds all the weight | result hinges on a single unit |
| Coarse p-value (SCM) | fewer than 10 usable placebos | finer significance claims unsupported |
| Durbin–Watson | outside [1, 3] | residual autocorrelation; interval too narrow |
| Pre-period R² | below 0.3 | model does not explain the series |
| Back-test (Bayesian) | placebo interval excludes zero | model manufactures effects |
| Cluster count (DiD) | fewer than 10 units | switches to wild cluster bootstrap |
| Slope change (ITS) | p < 0.05 | a single average effect hides a growing or decaying one |

---

## Validation

A causal estimate on real data **cannot be checked**, because the counterfactual
is unobservable. On synthetic data it can: the effect is inserted deliberately,
so the right answer is known.

`run_recovery_study` simulates many datasets with a known effect, runs every
estimator, and scores four things:

- **bias** — mean estimate minus truth. Should be near zero.
- **RMSE** — root mean squared error around the truth.
- **coverage** — share of runs whose 95% interval contained the truth. **This is
  the column that matters.** Materially below 0.95 means the intervals are too
  narrow and the estimator is overconfident, which is a worse failure than a
  noisy point estimate.
- **power** — share of runs that detected the effect. With `true_effect=0` this
  becomes the false-positive rate and should sit near 0.05
  (`false_positive_study`).

Results are in the [README](../README.md#does-the-tool-actually-work) and
reproducible with:

```bash
impactlens validate -n 250
```

### The simulation

`ScenarioConfig` generates a panel where the data generating process is known.
By default it includes a **market-wide shock starting at the intervention date** —
the hardest and most realistic confound, and the one a naive reading cannot
survive.

Unit heterogeneity is **additive** by default (level shifts), so every unit
shares the same trend and seasonal amplitude and parallel trends holds exactly.
That makes the baseline scenario a fair test. Each assumption has its own knob
for breaking it deliberately:

| Knob | Breaks |
|---|---|
| `market_shock` | the naive before/after comparison |
| `treated_pretrend` | parallel trends (selection on trend) |
| `proportional_heterogeneity` | parallel trends in levels |
| `heterogeneous_trends` | parallel trends via the donors |
| `noise_sd` | precision (intervals must widen) |
| `n_controls=0` | every control-based method |

---

## References

- Abadie, A., Diamond, A., & Hainmueller, J. (2010). *Synthetic Control Methods
  for Comparative Case Studies.* JASA 105(490).
- Brodersen, K. et al. (2015). *Inferring causal impact using Bayesian
  structural time-series models.* Annals of Applied Statistics 9(1).
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). *Bootstrap-Based
  Improvements for Inference with Clustered Errors.* Review of Economics and
  Statistics 90(3).
- Bertrand, M., Duflo, E., & Mullainathan, S. (2004). *How Much Should We Trust
  Differences-in-Differences Estimates?* QJE 119(1).
- Angrist, J. & Pischke, J.-S. (2009). *Mostly Harmless Econometrics.*
