"""What changed since the previous stored snapshot of this ticker."""
from __future__ import annotations

from app.analytics.narratives import Story
from app.analytics.util import count, pct, short_date, signed
from app.schemas import DeltaView, Quote, SentimentStat, Snapshot, Verdict

SHARP_CHANGE = 12
NOTABLE_CHANGE = 5


def build_delta(previous: Snapshot | None, verdict: Verdict, sentiment: SentimentStat, quote: Quote | None,
                stories: list[Story]) -> DeltaView:
    if previous is None:
        return DeltaView()
    change = verdict.score - previous.sentinel_score
    price_change = None
    if quote is not None and quote.price and previous.price:
        price_change = round((quote.price / previous.price - 1) * 100, 2)
    new = [s.narrative.headline for s in stories if s.narrative.is_new]

    if abs(change) >= SHARP_CHANGE:
        lead = f"Sentiment {'improved' if change > 0 else 'deteriorated'} sharply ({signed(change, 'd')}: " \
               f"{previous.sentinel_score} → {verdict.score})"
    elif abs(change) >= NOTABLE_CHANGE:
        lead = f"Sentiment {'improved' if change > 0 else 'softened'} ({signed(change, 'd')}: " \
               f"{previous.sentinel_score} → {verdict.score})"
    else:
        lead = f"Little changed ({signed(change, 'd')})"
    if previous.label != verdict.stance:
        lead += f", now {verdict.stance} (was {previous.label})"
    bits = [f"{lead} since {short_date(previous.at)} {previous.at:%H:%M} UTC"]
    if price_change is not None:
        bits.append(f"price {pct(price_change)}")
    if new:
        bits.append(f"{count(len(new), 'new story', 'new stories')}")
    return DeltaView(
        previous_at=previous.at, score_change=round(sentiment.score - previous.score, 3),
        sentinel_change=change, price_change_pct=price_change, new_narratives=new, note="; ".join(bits),
    )
