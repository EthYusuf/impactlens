# When this tool gives the wrong answer

Causal inference rests on assumptions. A tool that hides them produces numbers
that look authoritative and are sometimes badly wrong. This page is the list of
ways ImpactLens can mislead you, written down on purpose.

Read it before you quote a number to anyone.

---

## The failures the tool detects for you

These raise warnings automatically. They appear in `estimate.warnings`, on the
dashboard, and in the PDF report.

| Problem | How it is caught | What to do |
|---|---|---|
| Treated unit was already diverging | Pre-trend slope test on the treated-minus-control gap | Use synthetic control; DiD is biased here |
| Synthetic control does not track the pre-period | `pre_fit_ratio > 0.5` | Add better-matched donors, or drop the method |
| Weight collapsed onto one donor | `n_effective_donors == 1` | Treat the result as a two-unit comparison |
| Too few donors for real inference | fewer than 10 usable placebos | Report the p-value floor, not a precise figure |
| Too few clusters for asymptotic SEs | fewer than 10 units | Handled automatically via wild cluster bootstrap |
| Residual autocorrelation | Durbin–Watson outside [1, 3] | Interval is too narrow; treat as a lower bound |
| Model manufactures effects | Pre-period placebo back-test fails | Do not use the estimate |
| Short pre-period | fewer than 12 pre periods | Report as a preliminary finding |
| Short post-period | fewer than 4 post periods | Cannot distinguish a durable effect from a spike |
| No control group | `n_controls == 0` | Only ITS available; evidence is weak |
| Unbalanced panel | units with gaps | Those units are excluded from control-based methods |

---

## The failures no tool can detect

These are the dangerous ones. The software cannot see them in the data, so they
are your job.

### 1. The treated unit was chosen *because* it was improving

A clinic installs a booking system precisely in the quarter it is already
turning around. Every method here will credit the turnaround to the system.

If `treated_pretrend` is detectable, the pre-trend test catches it. If the
divergence begins *at* the intervention date for an unrelated reason, nothing
catches it.

**Mitigation:** know why this unit and not another. If the answer involves the
outcome, your estimate is biased upward and no amount of modelling fixes it.

### 2. Spillover onto the controls

Your new website attracts customers *away* from the comparison clinics. Their
outcomes fall, the gap widens, and the measured effect is too large. Or a
shared marketing campaign lifts everyone and the gap narrows, making the effect
look smaller than it was.

**Mitigation:** choose donors that cannot be affected — different city, different
catchment, different customer base. Document why.

### 3. Something else changed on the same date

You launched the booking system the same week you raised prices, hired two
staff, or appeared in the local press. Those are indistinguishable from the
intervention, and all of their effect lands in the estimate.

**Mitigation:** write down everything that changed in that window before you
look at the result. If the list has more than one item, the estimate is an
estimate of the bundle.

### 4. Anticipation

Customers who heard the system was coming delayed their booking. The pre-period
is depressed, so the measured jump is inflated.

**Mitigation:** exclude the periods immediately before the launch, or set the
intervention date to when the announcement happened rather than when the system
went live.

### 5. The post-period is too short to mean anything

A two-week window after launch mostly measures novelty. People try the new
thing, then revert.

**Mitigation:** the tool warns below four post periods, but "statistically
detectable" and "durable" are different claims. Check whether the pointwise
effect plot is flat or decaying.

### 6. The outcome was redefined

The booking system counts appointments differently from the old paper diary —
it records no-shows, or counts a rescheduled slot twice. The level shift is a
measurement artefact.

**Mitigation:** confirm the outcome is measured identically on both sides of the
intervention date. This is the most common real-world explanation for a suspiciously
clean level jump.

### 7. The counterfactual period is out of sample

Synthetic control and the Bayesian model both extrapolate a pre-period
relationship forward. If the post period is unlike anything in the pre period —
a pandemic, a new competitor, a regulatory change — the relationship may simply
no longer hold.

**Mitigation:** check that the donors' post-period values are within their
pre-period range. If not, every counterfactual here is a guess.

---

## Reporting standards

Follow these and the number survives scrutiny.

**Always report the interval, never the point estimate alone.** "+12%" is not a
finding. "+12% (95% interval: +4% to +20%)" is.

**When the interval crosses zero, say "no detectable effect".** Do not say "a
small positive effect". The data does not support a direction, and the narrative
layer refuses to write one.

**Name the method and its inference.** "Synthetic control, placebo p = 0.111 with
8 donors" tells a reader what they need. "Statistically significant" does not.

**Quote the strongest *applicable* method, not the most favourable one.** Picking
the method that gives the biggest number is specification shopping. If the
methods disagree, that is the finding — report the disagreement.

**Call it a preliminary finding when the data is thin.** With fewer than ~12
pre-periods or fewer than ~5 donors, you have an indication, not a result.

---

## Data handling

**Never commit client data to this repository.** The `.gitignore` blocks
`data/raw/`, `data/private/` and `*.xlsx`, but that is a safety net, not a
policy.

Before any real data touches the tool:

1. **Get written permission** to analyse it, and separately to publish anything
   derived from it.
2. **Anonymise unit names.** `control_03`, not `Dr Yılmaz Clinic`.
3. **Aggregate.** The tool needs one row per unit per period. It never needs an
   individual-level record, so do not give it one.
4. **Strip everything else.** Only `unit`, `date` and the outcome are used.
5. **Keep the raw extract outside the repository.**

If you cannot do all five, use the synthetic generator. The validation story is
just as strong, and the methodology is what the project is demonstrating.
