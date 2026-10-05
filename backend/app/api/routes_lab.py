"""Sentiment lab: score your own texts with the production engine."""
from __future__ import annotations

from fastapi import APIRouter

from app.schemas import ScoreRequest, ScoreResponse
from app.services import lab

router = APIRouter(tags=["lab"])


@router.post("/lab/score", response_model=ScoreResponse)
async def score(req: ScoreRequest) -> ScoreResponse:
    """Score 1-500 texts (≤ 10,000 chars each, ≤ 250,000 in all): label, confidence, drivers, themes,
    events and, with `ticker`, relevance to that company (404 when no provider knows the symbol). One
    request is scored at a time; when two more are already waiting the lab answers 503 with `Retry-After`."""
    return await lab.score_texts(req)
