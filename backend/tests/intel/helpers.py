"""Load recorded intel fixtures back into the exact shapes yfinance returns."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "intel"
NY = "America/New_York"


def fixture_path(rel: str) -> Path:
    return FIX / rel


def load_json(rel: str) -> Any:
    return json.loads((FIX / rel).read_text(encoding="utf-8"))


def load_text(rel: str) -> str:
    return (FIX / rel).read_text(encoding="utf-8")


def info(sym: str) -> dict[str, Any]:
    return load_json(f"yahoo/{sym}_info.json")


def bars(sym: str, kind: str = "daily") -> pd.DataFrame:
    """OHLCV frame with a tz-aware index (exchange tz for equities, UTC for crypto)."""
    df = pd.read_csv(FIX / f"yahoo/{sym}_{kind}.csv", index_col=0)
    idx = pd.to_datetime(df.index, utc=True)
    df.index = idx.tz_convert("UTC" if sym.endswith("-USD") else NY)
    return df


def recommendations(sym: str) -> pd.DataFrame:
    return pd.read_csv(FIX / f"yahoo/{sym}_recommendations.csv")


def upgrades(sym: str) -> pd.DataFrame:
    df = pd.read_csv(FIX / f"yahoo/{sym}_upgrades.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "GradeDate"
    return df.fillna({"FromGrade": "", "ToGrade": ""})


def calendar(sym: str) -> dict[str, Any]:
    raw = load_json(f"yahoo/{sym}_calendar.json")

    def conv(v: Any) -> Any:
        if isinstance(v, list):
            return [conv(x) for x in v]
        if isinstance(v, str) and len(v) == 10 and v[4] == "-":
            return date.fromisoformat(v)
        return v

    return {k: conv(v) for k, v in raw.items()}


def earnings_dates(sym: str) -> pd.DataFrame:
    df = pd.read_csv(FIX / f"yahoo/{sym}_earnings_dates.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(NY)
    df.index.name = "Earnings Date"
    return df


def insiders(sym: str) -> pd.DataFrame:
    df = pd.read_csv(FIX / f"yahoo/{sym}_insiders.csv")
    df["Start Date"] = pd.to_datetime(df["Start Date"])
    for col in ("Text", "URL", "Transaction", "Position", "Ownership", "Insider"):
        if col in df:
            df[col] = df[col].fillna("")
    return df


def indices() -> pd.DataFrame:
    df = pd.read_csv(FIX / "yahoo/indices.csv", header=[0, 1], index_col=0)
    df.index = pd.to_datetime(df.index)
    return df
