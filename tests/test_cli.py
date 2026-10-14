"""CLI behaviour, including the exit codes a CI pipeline would rely on."""

from __future__ import annotations

import pytest

from impactlens.cli import main
from impactlens.simulate import ScenarioConfig, generate_panel


@pytest.fixture(scope="module")
def panel_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("cli") / "panel.csv"
    panel = generate_panel(ScenarioConfig(n_controls=5, seed=17))
    panel.frame.to_csv(path, index=False)
    return path, panel


def test_analyse_runs_every_method_and_exits_clean(panel_csv, capsys):
    path, panel = panel_csv
    code = main([
        "analyse", str(path),
        "--treated", panel.treated_unit,
        "--date", str(panel.intervention_date.date()),
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "Synthetic Control" in out
    assert "Difference-in-Differences" in out
    assert "Agreement across methods" in out


def test_analyse_single_method(panel_csv, capsys):
    path, panel = panel_csv
    code = main([
        "analyse", str(path),
        "--treated", panel.treated_unit,
        "--date", str(panel.intervention_date.date()),
        "--method", "synthetic_control",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "Synthetic Control" in out
    assert "Difference-in-Differences" not in out


def test_analyse_writes_pdf_and_csv(panel_csv, tmp_path):
    path, panel = panel_csv
    pdf = tmp_path / "out" / "report.pdf"
    csv = tmp_path / "out" / "results.csv"
    code = main([
        "analyse", str(path),
        "--treated", panel.treated_unit,
        "--date", str(panel.intervention_date.date()),
        "--pdf", str(pdf), "--csv-out", str(csv),
        "--outcome-name", "appointments",
        "--project", "CLI test",
    ])
    assert code == 0
    assert pdf.exists() and pdf.stat().st_size > 20_000
    assert csv.exists()
    assert "effect" in csv.read_text()


def test_analyse_rejects_a_missing_file(capsys):
    code = main(["analyse", "no_such_file.csv", "--treated", "x", "--date", "2025-01-01"])
    assert code == 2
    assert "no such file" in capsys.readouterr().err


def test_analyse_reports_a_schema_error_without_a_traceback(panel_csv, capsys):
    path, panel = panel_csv
    code = main([
        "analyse", str(path),
        "--treated", "not_a_real_unit",
        "--date", str(panel.intervention_date.date()),
    ])
    assert code == 2
    err = capsys.readouterr().err
    assert "not present in data" in err
    assert "Traceback" not in err


def test_analyse_refuses_did_without_controls(tmp_path, capsys):
    panel = generate_panel(ScenarioConfig(n_controls=0, seed=18))
    path = tmp_path / "solo.csv"
    panel.frame.to_csv(path, index=False)
    code = main([
        "analyse", str(path),
        "--treated", panel.treated_unit,
        "--date", str(panel.intervention_date.date()),
        "--method", "did",
    ])
    assert code == 2
    assert "requires at least one control unit" in capsys.readouterr().err


def test_validate_prints_the_scoreboard(capsys, tmp_path):
    csv = tmp_path / "summary.csv"
    code = main(["validate", "-n", "6", "--n-controls", "5", "--csv-out", str(csv)])
    out = capsys.readouterr().out
    assert code == 0
    assert "coverage" in out
    assert "Verdict" in out
    assert csv.exists()


def test_demo_writes_panel_and_report(tmp_path, capsys):
    code = main(["demo", "--outdir", str(tmp_path), "--n-controls", "5"])
    out = capsys.readouterr().out
    assert code == 0
    assert "ground-truth effect inserted" in out
    assert "truth in interval" in out
    assert (tmp_path / "demo_panel.csv").exists()
    assert (tmp_path / "impact_report.pdf").exists()


def test_version_flag_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_unknown_subcommand_is_rejected():
    with pytest.raises(SystemExit) as exc:
        main(["frobnicate"])
    assert exc.value.code == 2
