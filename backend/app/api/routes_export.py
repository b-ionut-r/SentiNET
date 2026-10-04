"""Export the signals behind the latest analysis as CSV or JSON."""
from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, Response

from app.schemas import Analysis, Signal
from app.services import analyzer
from app.services.errors import NotFound

router = APIRouter(tags=["export"])

CSV_COLUMNS = (
    "id", "timestamp", "source", "kind", "publisher", "author", "title", "url", "score", "label",
    "confidence", "relevance", "weight", "engagement", "duplicates", "themes", "events", "user_label",
    "narrative_id", "drivers",
)
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _safe_cell(value: str) -> str:
    """Neutralize spreadsheet formula injection in text pulled from the internet."""
    return f"'{value}" if value.startswith(_FORMULA_PREFIXES) else value


def _text(value: str | None) -> str:
    return _safe_cell(value) if value else ""


def _row(s: Signal) -> list[object]:
    return [
        s.id, s.timestamp.isoformat() if s.timestamp else "", s.source, s.kind, _text(s.publisher),
        _text(s.author), _text(s.title), s.url or "", f"{s.score:.4f}", s.label, f"{s.confidence:.3f}",
        f"{s.relevance:.3f}", f"{s.weight:.4f}", s.engagement, s.duplicates, ";".join(s.themes),
        ";".join(s.events), s.user_label or "", s.narrative_id or "",
        _text("; ".join(f"{d.term} {d.impact:+.2f}" for d in s.drivers)),
    ]


def _filename(a: Analysis, ext: str) -> str:
    return f"sentinet_{a.ticker}_{a.generated_at.strftime('%Y%m%d-%H%M')}.{ext}"


@router.get("/export/{filename}", responses={200: {"content": {"text/csv": {}, "application/json": {}}}})
async def export_signals(filename: str) -> Response:
    """`/api/export/NVDA.csv` or `/api/export/NVDA.json` — every kept signal with its scoring trail.

    Uses the latest analysis computed by this server (what the user is looking
    at); runs one if none exists yet.
    """
    stem, dot, ext = filename.rpartition(".")
    ext = ext.lower()
    if not dot or ext not in ("csv", "json"):
        raise NotFound("Export as /api/export/{ticker}.csv or /api/export/{ticker}.json")
    symbol = analyzer.normalize(stem)
    analysis = analyzer.latest_analysis(symbol) or await analyzer.analyze(symbol)
    disposition = {"Content-Disposition": f'attachment; filename="{_filename(analysis, ext)}"'}

    if ext == "json":
        v = analysis.verdict
        payload = {
            "ticker": analysis.ticker,
            "generated_at": analysis.generated_at.isoformat(),
            "engine": analysis.engine,
            "verdict": {"score": v.score, "label": v.label, "confidence": v.confidence, "headline": v.headline},
            "count": len(analysis.signals),
            "signals": [s.model_dump(mode="json") for s in analysis.signals],
        }
        return Response(json.dumps(payload, ensure_ascii=False, indent=2), media_type="application/json",
                        headers=disposition)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_COLUMNS)
    writer.writerows(_row(s) for s in analysis.signals)
    return Response(buf.getvalue(), media_type="text/csv; charset=utf-8", headers=disposition)
