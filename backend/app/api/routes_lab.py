"""Sentiment lab: score your own texts with the production engine."""
from __future__ import annotations

from fastapi import APIRouter

from app.schemas import ScoreRequest, ScoreResponse
from app.services import lab

router = APIRouter(tags=["lab"])


@router.post("/lab/score", response_model=ScoreResponse)
async def score(req: ScoreRequest) -> ScoreResponse:
    """Score 1-500 texts (≤ 10,000 chars each): label, confidence, drivers, themes, events and,
    with `ticker`, relevance to that company."""
    return await lab.score_texts(req)
