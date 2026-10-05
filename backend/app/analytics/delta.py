"""What changed since the previous stored snapshot of this ticker.

The note never restates the previous time (the UI shows it in the viewer's
time zone). When the previous verdict's components are known, the score move
is attributed to its top one or two components — each component's points are
its share of the confidence-weighted nominal weight × its distance from 50,
the composite's own formula — and a move led by a component that became
available or unavailable (GDELT tone loading on a second look) is reported as
a coverage change, not as sentiment improving or deteriorating.
"""
from __future__ import annotations

from app.analytics.narratives import Story
from app.analytics.util import count, join_and, pct, signed
from app.schemas import Component, DeltaView, Quote, SentimentStat, Snapshot, Verdict

SHARP_CHANGE = 12
NOTABLE_CHANGE = 5
MOVER_POINTS = 1.0  # a component moving the score by less is not named
MAX_MOVERS = 2


def points(components: list[Component]) -> dict[str, float]:
    """Each available component's signed contribution to the score (the composite's weighting,
    before its small-coverage pull)."""
    avail = [c for c in components if c.available and c.score is not None]
    eff = {c.key: c.weight * (0.35 + 0.65 * c.confidence) for c in avail}
    total = sum(eff.values())
    if total <= 0:
        return {}
    return {c.key: eff[c.key] / total * ((c.score or 50.0) - 50.0) for c in avail}


def movers(previous: list[Component], current: list[Component]) -> list[tuple[str, float, str | None]]:
    """(label, points moved, 'now available' / 'no longer available' / None), largest first."""
    before, after = points(previous), points(current)
    labels = {c.key: c.label for c in [*previous, *current]}
    was = {c.key: c.available for c in previous}
    out = []
    for key in set(before) | set(after):
        moved = after.get(key, 0.0) - before.get(key, 0.0)
        if abs(moved) < MOVER_POINTS:
            continue
        status = None
        if key in after and not was.get(key, False):
            status = "now available"
        elif key in before and key not in after:
            status = "no longer available"
        out.append((labels.get(key, key), moved, status))
    out.sort(key=lambda m: -abs(m[1]))
    return out[:MAX_MOVERS]


def build_delta(previous: Snapshot | None, verdict: Verdict, sentiment: SentimentStat, quote: Quote | None,
                stories: list[Story], previous_components: list[Component] | None = None) -> DeltaView:
    if previous is None:
        return DeltaView()
    change = verdict.score - previous.sentinel_score
    price_change = None
    if quote is not None and quote.price and previous.price:
        price_change = round((quote.price / previous.price - 1) * 100, 2)
    new = [s.narrative.headline for s in stories if s.narrative.is_new]
    moved = movers(previous_components, verdict.components) if previous_components else []
    coverage = bool(moved) and moved[0][2] is not None  # the main mover appeared or disappeared

    if coverage and abs(change) >= NOTABLE_CHANGE:
        lead = f"Data coverage changed ({signed(change, 'd')}: {previous.sentinel_score} → {verdict.score})"
    elif abs(change) >= SHARP_CHANGE:
        lead = f"Sentiment {'improved' if change > 0 else 'deteriorated'} sharply ({signed(change, 'd')}: " \
               f"{previous.sentinel_score} → {verdict.score})"
    elif abs(change) >= NOTABLE_CHANGE:
        lead = f"Sentiment {'improved' if change > 0 else 'softened'} ({signed(change, 'd')}: " \
               f"{previous.sentinel_score} → {verdict.score})"
    else:
        lead = f"Little changed ({signed(change, 'd')})"
    if previous.label != verdict.stance:
        lead += f", now {verdict.stance} (was {previous.label})"
    bits = [f"{lead} since the last look"]
    if moved and abs(change) >= NOTABLE_CHANGE:
        bits.append(join_and([f"{label.lower()} {signed(pts, '.1f')} pts" + (f" ({status})" if status else "")
                              for label, pts, status in moved]))
    if price_change is not None:
        bits.append(f"price {pct(price_change)}")
    if new:
        bits.append(f"{count(len(new), 'new story', 'new stories')}")
    return DeltaView(
        previous_at=previous.at, score_change=round(sentiment.score - previous.score, 3),
        sentinel_change=change, price_change_pct=price_change, new_narratives=new, note="; ".join(bits),
    )
