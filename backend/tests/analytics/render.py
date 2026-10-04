"""Plain-text rendering of an `Analysis`, for the demo test and debugging."""
from __future__ import annotations

from app.schemas import Analysis


def render(a: Analysis) -> str:
    v = a.verdict
    out = [f"{a.ticker} — SentiNET {v.score}/100 · {v.label} · {v.confidence} confidence ({v.confidence_value:.2f})",
           f"  {v.headline}", "", "WHY"]
    out += [f"  [{r.polarity:>7}] {r.weight:.2f}  {r.text}" for r in v.reasons]
    out += ["", "COMPONENTS"]
    for c in v.components:
        score = f"{c.score:5.1f}" if c.score is not None else "  n/a"
        out.append(f"  {c.label:<11} {score}  w={c.weight:.2f} conf={c.confidence:.2f}  {c.detail}")
    out += ["", f"SENTIMENT  all {a.sentiment.score:+.3f} (n={a.sentiment.n}) · news {a.news.score:+.3f} "
                f"(n={a.news.n}) · social {a.social.score:+.3f} (n={a.social.n})", "", "INSIGHTS"]
    out += [f"  [{i.severity:>5}|{i.polarity:>7}] {i.kind}: {i.title} — {i.detail}" for i in a.insights]
    out += ["", "NARRATIVES"]
    for n in a.narratives:
        new = " NEW" if n.is_new else ""
        out.append(f"  impact {n.impact:.2f} · {n.count} items · {len(n.publishers)} outlets · tone {n.score:+.2f} · "
                   f"24h {n.velocity_24h}{new} · {', '.join(n.events) or '-'}\n      {n.headline}")
    out += ["", "BRIEF", f"  {a.brief.summary}"]
    out += [f"  + {p}" for p in a.brief.bull_points] + [f"  - {p}" for p in a.brief.bear_points]
    out += [f"  ? {p}" for p in a.brief.watch]
    out += ["", "CATALYSTS"]
    out += [f"  {c.date:%Y-%m-%d} {'↑' if c.upcoming else ' '} {c.kind:<8} {c.polarity:<7} {c.title}"
            + (f" — {c.detail}" if c.detail else "") for c in a.catalysts]
    if a.delta.previous_at:
        out += ["", f"DELTA  {a.delta.note}"]
    if a.crowd:
        out += ["", f"CROWD  {a.crowd.model_dump(exclude_none=True)}"]
    if a.attention:
        out += [f"ATTENTION  {a.attention.model_dump(exclude_none=True)}"]
    out += ["", "THEMES  " + ", ".join(f"{t.label} {t.count} ({t.score:+.2f})" for t in a.themes[:8])]
    out += ["SOURCES  " + " · ".join(f"{s.key} {s.status} {s.fetched}->{s.kept}" for s in a.sources)]
    return "\n".join(out)
