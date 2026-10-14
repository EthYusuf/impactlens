"""ImpactLens dashboard.

Run with::

    streamlit run src/impactlens/app/streamlit_app.py

The tab order mirrors how the question should actually be answered: look at the
data, declare the intervention, read the estimate, then interrogate the
diagnostics before believing it. The validation tab is separate because it
audits the tool, not the dataset.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow `streamlit run path/to/streamlit_app.py` without installing the package.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:  # pragma: no cover - import side effect
    sys.path.insert(0, str(_SRC))

from impactlens import __version__  # noqa: E402
from impactlens.dataset import PanelDataset, SchemaError  # noqa: E402
from impactlens.estimators import ESTIMATOR_LABELS, event_study, run_all  # noqa: E402
from impactlens.reporting import narrative  # noqa: E402
from impactlens.reporting.plots import (  # noqa: E402
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

st.set_page_config(
    page_title="ImpactLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1400px; }
      h1, h2, h3 { color: #1b3a5c; }
      .il-tag {
          display:inline-block; padding:2px 10px; border-radius:10px;
          background:#eef2f7; color:#1b3a5c; font-size:0.75rem;
          font-weight:600; letter-spacing:0.08em; text-transform:uppercase;
      }
      .il-verdict-yes { color:#2b7a6f; font-weight:700; font-size:1.15rem; }
      .il-verdict-no  { color:#8a8a8a; font-weight:700; font-size:1.15rem; }
      .il-note {
          background:#f7f8fa; border-left:3px solid #c2653b;
          padding:0.7rem 0.9rem; border-radius:3px; font-size:0.9rem;
      }
      div[data-testid="stMetricValue"] { color:#1b3a5c; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------- cache
@st.cache_data(show_spinner=False)
def _make_demo(
    true_effect: float, n_controls: int, market_shock: float, seed: int
) -> tuple[pd.DataFrame, str, pd.Timestamp, float, float]:
    panel = generate_panel(
        ScenarioConfig(
            true_effect=true_effect,
            n_controls=n_controls,
            market_shock=market_shock,
            seed=seed,
        )
    )
    return (
        panel.frame,
        panel.treated_unit,
        panel.intervention_date,
        panel.true_average_effect,
        panel.true_relative_effect,
    )


@st.cache_data(show_spinner=False)
def _run_estimators(
    frame: pd.DataFrame,
    treated: str,
    date: pd.Timestamp,
    unit_col: str,
    date_col: str,
    outcome_col: str,
    level: float,
) -> tuple[dict, dict]:
    ds = PanelDataset.from_frame(
        frame,
        treated_unit=treated,
        intervention_date=date,
        unit_col=unit_col,
        date_col=date_col,
        outcome_col=outcome_col,
    )
    results = run_all(ds, level=level)
    return results, {"warnings": ds.warnings(), "quality": ds.quality_report()}


@st.cache_data(show_spinner=False)
def _run_validation(n_reps: int, true_effect: float, n_controls: int) -> pd.DataFrame:
    return run_recovery_study(
        ScenarioConfig(true_effect=true_effect, n_controls=n_controls),
        n_replications=n_reps,
    ).summary


def _build_dataset(
    frame: pd.DataFrame, treated: str, date, unit_col: str, date_col: str, outcome_col: str
) -> PanelDataset:
    return PanelDataset.from_frame(
        frame,
        treated_unit=treated,
        intervention_date=date,
        unit_col=unit_col,
        date_col=date_col,
        outcome_col=outcome_col,
    )


# ------------------------------------------------------------------- sidebar
st.sidebar.markdown(f"### 🔍 ImpactLens `{__version__}`")
st.sidebar.caption("Did the intervention actually work?")
st.sidebar.divider()

source = st.sidebar.radio(
    "Data source",
    ["Demo (known ground truth)", "Upload CSV"],
    help="The demo inserts an effect we choose, so you can check whether the tool "
    "recovers the right answer. Real data never lets you do that.",
)

frame: pd.DataFrame | None = None
truth: float | None = None
truth_rel: float | None = None
default_treated = ""
default_date = None

if source.startswith("Demo"):
    st.sidebar.markdown("**Simulation settings**")
    # Held as whole percentage points so the slider label reads "+12%" rather
    # than formatting the fraction 0.12 as "+0%".
    true_effect_pct = st.sidebar.slider(
        "True effect to insert", -20, 40, 12, 1,
        format="%+d%%",
        help="The real lift applied to the treated unit after the intervention. "
        "Set it to 0 to check that the estimators report nothing.",
    )
    true_effect = true_effect_pct / 100.0
    n_controls = st.sidebar.slider("Control units", 0, 15, 8)
    market_shock = st.sidebar.slider(
        "Market-wide shock", 0.0, 60.0, 26.0, 2.0,
        help="A surge that hits every unit starting at the intervention date. This is "
        "the confound. Raise it and watch the naive estimate break while the "
        "control-based ones hold.",
    )
    seed = st.sidebar.number_input("Random seed", 0, 9999, 7, 1)
    frame, default_treated, default_date, truth, truth_rel = _make_demo(
        true_effect, n_controls, market_shock, int(seed)
    )
    unit_col, date_col, outcome_col = "unit", "date", "y"
    outcome_name = "weekly appointments"
    project_name = "Flagship clinic booking system"
else:
    upload = st.sidebar.file_uploader("Long-format CSV", type=["csv"])
    st.sidebar.markdown(
        '<div class="il-note">One row per unit and date. Never upload identifiable '
        "client data: anonymise unit names and aggregate before you do.</div>",
        unsafe_allow_html=True,
    )
    if upload is not None:
        frame = pd.read_csv(upload)
        cols = list(frame.columns)
        unit_col = st.sidebar.selectbox("Unit column", cols, index=0)
        date_col = st.sidebar.selectbox("Date column", cols, index=min(1, len(cols) - 1))
        outcome_col = st.sidebar.selectbox("Outcome column", cols, index=len(cols) - 1)
        outcome_name = st.sidebar.text_input("Outcome name", "the outcome")
        project_name = st.sidebar.text_input("Project name", "Untitled analysis")
    else:
        unit_col = date_col = outcome_col = ""
        outcome_name = "the outcome"
        project_name = "Untitled analysis"

st.sidebar.divider()
level = st.sidebar.select_slider(
    "Interval level", options=[0.80, 0.90, 0.95, 0.99], value=0.95,
    format_func=lambda v: f"{v:.0%}",
)

# --------------------------------------------------------------------- header
st.markdown('<span class="il-tag">Causal impact measurement</span>', unsafe_allow_html=True)
st.title("Did the intervention actually work?")
st.markdown(
    "A business installs a booking system and appointments rise. The dashboard shows "
    "the rise; it cannot tell you whether the system caused it or the season did. "
    "**ImpactLens separates the two and reports the answer as a range, not a point.**"
)

if frame is None:
    st.info("Upload a CSV in the sidebar, or switch to the demo to see the tool work.")
    st.stop()

# ------------------------------------------------------------- setup controls
units = sorted(frame[unit_col].astype(str).unique())
dates = pd.to_datetime(frame[date_col], errors="coerce").dropna()

c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    treated = st.selectbox(
        "Treated unit",
        units,
        index=units.index(default_treated) if default_treated in units else 0,
        help="The unit that received the intervention.",
    )
with c2:
    dmin, dmax = dates.min().date(), dates.max().date()
    fallback = default_date.date() if default_date is not None else (
        dates.quantile(0.7).date()
    )
    intervention = st.date_input(
        "Intervention date",
        value=fallback,
        min_value=dmin,
        max_value=dmax,
        help="First period in which the intervention was live. Counts as post.",
    )
with c3:
    st.markdown("&nbsp;", unsafe_allow_html=True)

try:
    ds = _build_dataset(frame, treated, intervention, unit_col, date_col, outcome_col)
except SchemaError as exc:
    st.error(f"**Cannot analyse this data.** {exc}")
    st.stop()

try:
    results, meta = _run_estimators(
        frame, treated, pd.Timestamp(intervention), unit_col, date_col, outcome_col, level
    )
except Exception as exc:  # pragma: no cover - surfaced in the UI
    st.error(f"Estimation failed: {exc}")
    st.stop()

if not results:
    st.error("No estimator could run on this data.")
    st.stop()

available = [k for k in results if k != "naive"]
default_method = next(
    (m for m in ("synthetic_control", "bayesian", "did", "its") if m in results),
    next(iter(results)),
)

tabs = st.tabs([
    "📊 Result",
    "🔬 Diagnostics",
    "⚖️ Method comparison",
    "✅ Tool validation",
    "📄 Report",
    "📁 Data",
])

# ================================================================== RESULT ==
with tabs[0]:
    method = st.radio(
        "Estimator",
        list(results.keys()),
        index=list(results.keys()).index(default_method),
        format_func=lambda k: ESTIMATOR_LABELS.get(k, k),
        horizontal=True,
    )
    est = results[method]

    verdict_class = "il-verdict-yes" if est.is_significant else "il-verdict-no"
    verdict_text = (
        f"Effect detected ({est.direction})" if est.is_significant
        else "No detectable effect"
    )
    st.markdown(
        f'<span class="{verdict_class}">{verdict_text}</span>', unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Effect per period",
        f"{est.effect:+,.1f}",
        help=f"{int(level*100)}% {est.interval.kind} interval: "
        f"{est.interval.lower:+,.1f} to {est.interval.upper:+,.1f}",
    )
    m2.metric(
        "Relative lift",
        f"{est.relative_effect*100:+.1f}%" if est.relative_effect is not None else "n/a",
        help=(
            f"{est.relative_interval.lower*100:+.1f}% to "
            f"{est.relative_interval.upper*100:+.1f}%"
            if est.relative_interval is not None else "no positive counterfactual base"
        ),
    )
    m3.metric(
        "Cumulative effect",
        f"{est.cumulative_effect:+,.0f}" if est.cumulative_effect is not None else "n/a",
        help=f"over {est.n_post} post-intervention periods",
    )
    m4.metric(
        "p-value",
        f"{est.p_value:.4f}" if est.p_value is not None else "n/a",
        help=str(est.diagnostics.get("inference", "")),
    )

    st.caption(
        f"**{int(level*100)}% {est.interval.kind} interval:** "
        f"{est.interval.lower:+,.2f} to {est.interval.upper:+,.2f} "
        f"· inference: {est.diagnostics.get('inference', 'n/a')}"
    )

    if truth is not None:
        covered = est.interval.contains(truth)
        st.markdown(
            f'<div class="il-note"><b>Ground truth check.</b> The simulation inserted '
            f"an effect of <b>{truth:+,.2f}</b> per period ({truth_rel:+.1%}). This "
            f"estimator returned <b>{est.effect:+,.2f}</b> "
            f"(error {est.effect - truth:+,.2f}), and its interval "
            f"<b>{'contains' if covered else 'MISSES'}</b> the truth.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### What this means")
    for para in narrative.executive_summary(est, ds, outcome_name=outcome_name):
        st.write(para)

    st.pyplot(
        plot_counterfactual(
            est, intervention_date=ds.intervention_date, outcome_label=outcome_name
        ),
        width="stretch",
    )
    g1, g2 = st.columns(2)
    with g1:
        st.pyplot(
            plot_pointwise_effect(est, intervention_date=ds.intervention_date),
            width="stretch",
        )
    with g2:
        st.pyplot(
            plot_cumulative_effect(est, intervention_date=ds.intervention_date),
            width="stretch",
        )

    if est.warnings:
        with st.expander(f"⚠️ {len(est.warnings)} caveat(s) raised", expanded=True):
            for w in est.warnings:
                st.warning(w)
    with st.expander("What must be true for this number to be valid"):
        for a in est.assumptions:
            st.markdown(f"- {a}")

# ============================================================= DIAGNOSTICS ==
with tabs[1]:
    st.markdown("#### Can this result be trusted?")
    st.caption(
        "These checks cannot prove the estimate is right. They can show it is wrong."
    )
    est = results[default_method]

    if ds.has_controls:
        ev = est.diagnostics.get("event_study") or event_study(ds, level=level)
        st.pyplot(plot_event_study(ev), width="stretch")
        p = ev.get("pretrend_p_value")
        if p is not None:
            if p < 0.05:
                st.error(
                    f"**Parallel-trends test fails** (joint p={p:.4f}). The treated unit "
                    "was already moving differently from the controls before the "
                    "intervention, so difference-in-differences is biased here. Prefer "
                    "synthetic control, which fits the pre-period instead of assuming it."
                )
            else:
                st.success(
                    f"**Parallel-trends test passes** (joint p={p:.4f}). Pre-intervention "
                    "coefficients are jointly indistinguishable from zero. This is "
                    "necessary, not sufficient."
                )
    else:
        st.warning(
            "No control units, so there is no parallel-trends test to run and no placebo "
            "pool to compare against. Everything below is unavailable and the estimate "
            "rests on untestable assumptions."
        )

    sc = results.get("synthetic_control")
    if sc is not None:
        d1, d2 = st.columns(2)
        with d1:
            st.pyplot(plot_placebo_distribution(sc), width="stretch")
        with d2:
            st.pyplot(plot_weights(sc), width="stretch")
        f1, f2, f3 = st.columns(3)
        f1.metric("Pre-period RMSPE", f"{sc.diagnostics['pre_rmspe']:,.2f}",
                  help="Fit error of the synthetic control before the intervention. "
                       "Lower is better.")
        f2.metric("Post/pre RMSPE ratio", f"{sc.diagnostics['rmspe_ratio']:,.2f}",
                  help="How much worse the fit gets after the intervention. A large "
                       "ratio is the signature of a real effect.")
        f3.metric("Effective donors", f"{sc.diagnostics['n_effective_donors']}",
                  help="Donors with weight above 1%.")

    bayes = results.get("bayesian")
    if bayes is not None:
        with st.expander("Bayesian model fit and placebo back-test", expanded=False):
            b1, b2, b3 = st.columns(3)
            b1.metric("Pre-period R²", f"{bayes.diagnostics['r_squared_pre']:.3f}")
            b2.metric("Durbin-Watson", f"{bayes.diagnostics['durbin_watson']:.2f}",
                      help="Near 2 means no residual autocorrelation.")
            b3.metric("Pre-period MAPE", f"{bayes.diagnostics['mape_pre']:.2f}%")
            bt = bayes.diagnostics.get("backtest", {})
            if bt.get("ran"):
                if bt["false_positive"]:
                    st.error(
                        "**Back-test failed.** A fake intervention placed inside the "
                        "pre-period produced a significant effect "
                        f"({bt['placebo_effect']:+,.2f}, interval {bt['ci_lower']:+,.2f} "
                        f"to {bt['ci_upper']:+,.2f}). The model manufactures effects, so "
                        "the headline number is not credible."
                    )
                else:
                    st.success(
                        "**Back-test passed.** A fake intervention inside the pre-period "
                        f"produced no significant effect ({bt['placebo_effect']:+,.2f}, "
                        f"interval {bt['ci_lower']:+,.2f} to {bt['ci_upper']:+,.2f}), "
                        "which is the correct answer there."
                    )
            else:
                st.info(f"Back-test not run: {bt.get('reason', 'unknown')}")

    with st.expander("Data quality"):
        q = meta["quality"]
        st.dataframe(
            pd.DataFrame({"value": {k: str(v) for k, v in q.items()}}),
            width="stretch",
        )
        for w in meta["warnings"]:
            st.warning(w)

# ========================================================== METHOD COMPARE ==
with tabs[2]:
    st.markdown("#### Does the answer survive a change of method?")
    st.caption(
        "Agreement across methods means the finding comes from the data. Disagreement "
        "means it comes from the assumptions."
    )
    st.pyplot(
        plot_method_comparison(results, truth=truth), width="stretch"
    )
    note = narrative.comparison_note(results)
    if note:
        st.markdown(f'<div class="il-note">{note}</div>', unsafe_allow_html=True)

    table = pd.DataFrame({ESTIMATOR_LABELS.get(k, k): v.to_dict()
                          for k, v in results.items()}).T
    show = [c for c in ("effect", "ci_lower", "ci_upper", "relative_effect",
                        "cumulative_effect", "p_value", "significant", "n_warnings")
            if c in table.columns]
    if truth is not None:
        table["error_vs_truth"] = table["effect"] - truth
        table["truth_in_interval"] = [
            results[k].interval.contains(truth) for k in results
        ]
        show += ["error_vs_truth", "truth_in_interval"]
    st.dataframe(
        table.loc[:, show].style.format(
            {c: "{:,.3f}" for c in show
             if c not in ("significant", "n_warnings", "truth_in_interval")}
        ),
        width="stretch",
    )
    st.caption(
        "The naive row is what a before/after dashboard would report. The gap between "
        "it and the control-based rows is movement that was never caused by the "
        "intervention."
    )

# ============================================================== VALIDATION ==
with tabs[3]:
    st.markdown("#### Does the tool recover an effect it was never told about?")
    st.markdown(
        "On real data a causal estimate cannot be checked, because the counterfactual "
        "is unobservable. On simulated data it can: the effect is inserted deliberately, "
        "so the right answer is known. Each estimator is scored on **bias**, **RMSE**, "
        "**interval coverage** and **power** across many simulated datasets."
    )
    st.markdown(
        '<div class="il-note"><b>Coverage is the column that matters.</b> It is the share '
        "of runs whose interval actually contained the true effect. It should land near "
        "the nominal level. Materially below means the intervals are too narrow and the "
        "estimator is overconfident — a worse failure than a noisy point estimate.</div>",
        unsafe_allow_html=True,
    )
    v1, v2 = st.columns([1, 3])
    with v1:
        n_reps = st.select_slider("Replications", options=[20, 50, 100, 200], value=50)
        run_it = st.button("Run recovery study", type="primary")
    if run_it:
        with st.spinner(f"Simulating {n_reps} datasets and re-estimating each..."):
            summary = _run_validation(
                int(n_reps),
                true_effect=0.12,
                n_controls=max(ds.n_controls, 4),
            )
        st.session_state["validation_summary"] = summary

    summary = st.session_state.get("validation_summary")
    if summary is not None and not summary.empty:
        st.pyplot(plot_recovery_summary(summary, level=level), width="stretch")
        cols = [c for c in ("label", "n_runs", "true_effect", "mean_estimate", "bias",
                            "rmse", "coverage", "power", "mean_ci_width")
                if c in summary.columns]
        st.dataframe(
            summary.loc[:, cols].style.format(
                {c: "{:,.3f}" for c in cols if c != "label" and c != "n_runs"}
            ),
            width="stretch",
        )
    else:
        st.info("Press **Run recovery study** to audit the estimators.")

# ================================================================== REPORT ==
with tabs[4]:
    st.markdown("#### Download the report")
    st.caption(
        "A multi-page PDF: the finding, the uncertainty, the diagnostics that could "
        "invalidate it, and the assumptions it rests on."
    )
    r1, r2 = st.columns(2)
    with r1:
        report_method = st.selectbox(
            "Headline method",
            list(results.keys()),
            index=list(results.keys()).index(default_method),
            format_func=lambda k: ESTIMATOR_LABELS.get(k, k),
        )
        author = st.text_input("Author (optional)", "")
    with r2:
        include_validation = st.checkbox(
            "Include a recovery study (slower)", value=False,
            help="Runs 50 simulated replications and appends the validation pages.",
        )
        proj = st.text_input("Project name", project_name)

    if st.button("Generate PDF", type="primary"):
        with st.spinner("Rendering report..."):
            val = None
            if include_validation:
                val = _run_validation(50, true_effect=0.12, n_controls=max(ds.n_controls, 4))
            buf_path = Path(st.session_state.get("_tmpdir", ".")) / "impactlens_report.pdf"
            try:
                import tempfile

                buf_path = Path(tempfile.gettempdir()) / "impactlens_report.pdf"
                build_pdf_report(
                    buf_path,
                    results[report_method],
                    ds,
                    project_name=proj,
                    outcome_name=outcome_name,
                    all_estimates=results,
                    validation_summary=val,
                    author=author,
                )
                data = buf_path.read_bytes()
            except Exception as exc:  # pragma: no cover - surfaced in the UI
                st.error(f"Report generation failed: {exc}")
                data = None
        if data:
            st.success(f"Report ready ({len(data) / 1024:,.0f} KB).")
            st.download_button(
                "⬇️ Download PDF",
                data=data,
                file_name=f"impactlens_{proj.lower().replace(' ', '_')}.pdf",
                mime="application/pdf",
            )

    st.divider()
    st.markdown("##### Results as CSV")
    csv_table = pd.DataFrame({k: v.to_dict() for k, v in results.items()}).T
    buf = io.StringIO()
    csv_table.to_csv(buf)
    st.download_button(
        "⬇️ Download results CSV",
        data=buf.getvalue(),
        file_name="impactlens_results.csv",
        mime="text/csv",
    )

# ==================================================================== DATA ==
with tabs[5]:
    st.markdown("#### The panel")
    st.pyplot(
        plot_panel_overview(ds, outcome_label=outcome_name), width="stretch"
    )
    q = meta["quality"]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Units", q["n_units"])
    d2.metric("Periods", q["n_periods"])
    d3.metric("Pre / post", f"{q['n_pre']} / {q['n_post']}")
    d4.metric("Frequency", str(q["frequency"]))
    st.dataframe(ds.wide(), width="stretch", height=320)
    csv_buf = io.StringIO()
    frame.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download this panel as CSV",
        data=csv_buf.getvalue(),
        file_name="impactlens_panel.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "ImpactLens reports causal estimates conditional on assumptions it states openly. "
    "A number from this tool is evidence, not proof — and with a short pre-period or a "
    "weak donor pool it is a preliminary finding, not a result."
)
