"""Demonstration: prints a complete Analysis for a rich synthetic scenario.

Run with `pytest tests/analytics/test_demo.py -s` to read the output.
"""
from __future__ import annotations

from app.analytics.build import build_analysis
from tests.analytics.factories import snapshot
from tests.analytics.render import render
from tests.analytics.scenarios import bullish_large_cap


def test_demo_full_analysis(capsys) -> None:
    inputs = bullish_large_cap()
    inputs.previous = snapshot(28, sentinel=58, score=0.12, label="bullish", price=191.0,
                               narratives=["Acme stock jumps 6% after record data-center sales"])
    analysis = build_analysis(inputs)
    text = render(analysis)
    with capsys.disabled():
        print("\n" + "=" * 110 + "\n" + text + "\n" + "=" * 110)
    for section in ("WHY", "COMPONENTS", "INSIGHTS", "NARRATIVES", "BRIEF", "CATALYSTS", "DELTA"):
        assert section in text
