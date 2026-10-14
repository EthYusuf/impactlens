# Contributing

## Setup

```bash
git clone https://github.com/EthYusuf/impactlens.git
cd impactlens
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,app]"
pytest
```

## The bar for a change

This project makes quantitative claims about its own accuracy. A change is only
finished when those claims still hold.

**Before opening a pull request:**

```bash
ruff check .
pytest
impactlens validate -n 60 --csv-out validation.csv
python scripts/assert_calibration.py validation.csv
```

All four are what the CI workflow runs. Note that the workflow is currently set
to manual trigger only (GitHub Actions is unavailable on this account for
billing reasons), so **run them locally before every commit** — they are the
source of truth, not a formality. The calibration gate is the important one; see
below.

## Touching an estimator

Unit tests check that an estimator runs and behaves sensibly on a few cases.
They **cannot** check calibration: whether a 95% interval really contains the
truth 95% of the time needs a sample of simulated datasets, not one.

So if you change anything that affects an estimate or its interval:

1. Run the full study: `impactlens validate -n 250`
2. Check that coverage for `did`, `synthetic_control` and `bayesian` has not
   regressed.
3. If you changed a default, **measure the alternative** and put the table in
   the docstring. `scripts/measure_bayesian_spec.py` is the pattern: it exists
   purely to justify one default with numbers, and
   [docs/methodology.md](docs/methodology.md) quotes its output.
4. Refresh the published figures: `python scripts/make_assets.py --replications 250`

A change that narrows the intervals will pass every unit test and still make the
tool overconfident. That is the failure mode this project exists to prevent, so
please do not route around the gate.

## Adding an estimator

1. New module in `src/impactlens/estimators/`.
2. Return an `ImpactEstimate`. It must carry a real `Interval`, a populated
   `assumptions` list, and a `diagnostics["inference"]` string naming how the
   interval was derived.
3. Register it in `estimators/__init__.py` (`ESTIMATORS`, `ESTIMATOR_LABELS`,
   and `_NEEDS_CONTROLS` if it requires controls).
4. Add it to the parametrised contract tests in `tests/test_estimators.py`.
5. Run the recovery study and report its coverage in the PR description. An
   estimator that cannot be validated does not go in.

Label the interval type honestly: `"confidence"`, `"credible"` and `"placebo"`
are different claims, and the report distinguishes them.

## Style

- Ruff, 100 columns. `ruff check .` must be clean.
- Docstrings explain **why**, especially where the statistics are subtle. The
  `event_study` docstring is the model: it documents the formulation that was
  rejected and the measurement that justified the replacement.
- Warnings that reach a user should name the problem, quantify it, and say what
  to do instead. `"Weak pre-period fit"` is not enough; the actual message
  reports the ratio and tells the reader the counterfactual is unreliable.
- Never let the narrative layer claim a direction when the interval crosses
  zero. There is a test for this.

## Never commit

Client data. `.gitignore` blocks `data/raw/`, `data/private/` and `*.xlsx`, but
that is a safety net. See [docs/assumptions.md](docs/assumptions.md#data-handling).

## Reporting a statistical bug

Most valuable contribution there is. Please include:

- A `ScenarioConfig` that reproduces it, or a minimal CSV.
- What you expected and what you got.
- Whether coverage is affected — a biased point estimate with honest intervals
  is a smaller problem than a correct point estimate with intervals that lie.
