"""Capture screenshots of the running Streamlit dashboard.

Starts the app on a free port, drives it with Playwright, and saves one PNG per
tab into ``docs/screenshots/``. Run from the repository root::

    pip install playwright && playwright install chromium
    python scripts/capture_screenshots.py

The screenshots in the README are produced by this script, so they always show
the app as it actually renders rather than a mock-up.
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "impactlens" / "app" / "streamlit_app.py"
OUT = ROOT / "docs" / "screenshots"

# Tab label -> output file. Labels must match the app's tab captions.
TABS = [
    ("Result", "ui_01_result.png"),
    ("Diagnostics", "ui_02_diagnostics.png"),
    ("Method comparison", "ui_03_method_comparison.png"),
    ("Tool validation", "ui_04_validation.png"),
    ("Report", "ui_05_report.png"),
    ("Data", "ui_06_data.png"),
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_server(port: int, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2.0):
                return True
        except OSError:
            time.sleep(1.0)
    return False


def _settle(page, extra_ms: int = 1200) -> None:
    """Wait until Streamlit stops rendering.

    The status widget only exists *while* a rerun is in flight, so polling for
    its absence returns immediately on a page that has not started rendering
    yet. Callers must therefore wait for real content first (see
    ``wait_for_selector`` on the tab list below) -- this function only handles
    the settle after an interaction.
    """
    page.wait_for_load_state("networkidle")
    for _ in range(120):
        running = page.locator('[data-testid="stStatusWidget"]')
        if running.count() == 0:
            break
        try:
            if not running.first.is_visible():
                break
        except Exception:
            break
        page.wait_for_timeout(500)
    page.wait_for_timeout(extra_ms)


def _visible_figures(page) -> int:
    """Count matplotlib figures in the *active* tab.

    Streamlit keeps every tab's content mounted in the DOM and hides the
    inactive panels, so a plain ``img`` count is the same on every tab and tells
    you nothing. Restricting to visible elements is what makes it a usable
    signal for "this tab has rendered its chart".
    """
    return page.locator('[data-testid="stImage"] img:visible').count()


def _run_validation_in_app(page, *, timeout_s: int) -> None:
    """Click through the recovery study so the screenshot shows real numbers."""
    btn = page.get_by_role("button").filter(has_text="Run recovery study")
    if btn.count() == 0:
        print("  ! 'Run recovery study' button not found")
        return

    before = _visible_figures(page)
    print(f"  running the recovery study in-app (visible figures now: {before})...")
    btn.first.click()

    start = time.time()
    last_report = 0.0
    while time.time() - start < timeout_s:
        page.wait_for_timeout(2000)
        now = _visible_figures(page)
        if now > before:
            print(f"  scoreboard rendered after {time.time() - start:.0f}s")
            _settle(page, extra_ms=2500)
            return
        elapsed = time.time() - start
        if elapsed - last_report >= 30:
            last_report = elapsed
            print(f"    still running ({elapsed:.0f}s elapsed)")
    print(f"  ! the study did not finish within {timeout_s}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1100)
    parser.add_argument("--scale", type=int, default=2, help="device scale factor")
    parser.add_argument("--validation-timeout", type=int, default=900,
                        help="seconds to wait for the in-app recovery study")
    parser.add_argument("--keep-open", action="store_true",
                        help="leave the server running after capture")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is not installed. Run:\n"
            "  pip install playwright\n  playwright install chromium",
            file=sys.stderr,
        )
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    print(f"starting streamlit on port {port}...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(APP),
            "--server.port", str(port),
            "--server.headless", "true",
            "--server.fileWatcherType", "none",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        if not _wait_for_server(port):
            print("streamlit did not start in time", file=sys.stderr)
            if proc.stdout:
                print(proc.stdout.read()[-4000:], file=sys.stderr)
            return 1
        print("server up; launching browser...")

        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(
                viewport={"width": args.width, "height": args.height},
                device_scale_factor=args.scale,
            )
            page.goto(f"http://127.0.0.1:{port}", wait_until="domcontentloaded")
            # The first render runs every estimator -- bootstrap, placebos and
            # posterior draws -- so it can take tens of seconds. Wait for the
            # tab list to exist rather than guessing at a timeout.
            print("waiting for the first render (runs all estimators)...")
            page.wait_for_selector("[role='tab']", timeout=180_000)
            _settle(page, extra_ms=3000)
            found = page.get_by_role("tab").count()
            print(f"  app rendered with {found} tabs")

            # The sidebar stays expanded on purpose: the controls are part of
            # what these screenshots document.
            for label, filename in TABS:
                tab = page.get_by_role("tab").filter(has_text=label)
                if tab.count() == 0:
                    print(f"  ! tab {label!r} not found, skipping")
                    continue
                tab.first.click()
                _settle(page)

                if label == "Tool validation":
                    _run_validation_in_app(page, timeout_s=args.validation_timeout)

                path = OUT / filename
                page.screenshot(path=str(path), full_page=True)
                size_kb = path.stat().st_size / 1024
                print(f"  wrote {path.relative_to(ROOT)} ({size_kb:,.0f} KB)")

            browser.close()
    finally:
        if not args.keep_open:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                proc.kill()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
