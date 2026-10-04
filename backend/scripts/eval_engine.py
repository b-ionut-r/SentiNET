"""Benchmark the sentiment engines on real, labeled financial text.

    python -m scripts.eval_engine                  # VADER vs Sentinel on the held-out sets
    python -m scripts.eval_engine --write          # ... and save tests/nlp/data/engine_eval_results.json
    python -m scripts.eval_engine --errors 40      # print misclassified *train* examples (tuning)

Datasets (downloaded once into $SENTINET_CACHE or ~/.cache/sentinet - never committed):

* zeroshot/twitter-financial-news-sentiment (MIT): finance-news tweets,
  0=bearish 1=bullish 2=neutral; ``train`` (used for tuning) and ``valid`` (held out).
* Financial PhraseBank v1.0, Sentences_AllAgree (Malo et al. 2014, CC BY-NC-SA 3.0):
  sentences from company releases with unanimous annotator labels (held out).

Protocol: lexicon/rules/blend were tuned on Twitter **train** only; Twitter
**valid** and PhraseBank AllAgree are reported as held-out. All texts are scored
with the "news" register. FinBERT (via the HF Inference API) is included when
HF_TOKEN is set.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

from app.nlp.engine import SentinelEngine, VaderEngine
from app.nlp.types import SentimentEngine

LABELS = ("bullish", "bearish", "neutral")
CACHE = Path(os.environ.get("SENTINET_CACHE", Path.home() / ".cache" / "sentinet"))
RESULTS_PATH = Path(__file__).resolve().parent.parent / "tests" / "nlp" / "data" / "engine_eval_results.json"

TWITTER_URL = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/main/sent_{split}.csv"
PHRASEBANK_URL = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
TWITTER_LABELS = {"0": "bearish", "1": "bullish", "2": "neutral"}
PHRASEBANK_LABELS = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}

Dataset = list[tuple[str, str]]  # (text, gold label)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url} -> {dest}", file=sys.stderr)
        req = urllib.request.Request(url, headers={"User-Agent": "SentiNET-eval/2.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed https URLs
            dest.write_bytes(resp.read())
    return dest


def load_twitter(split: str) -> Dataset:
    """``split`` is "train" or "valid"."""
    path = _download(TWITTER_URL.format(split=split), CACHE / f"twitter_{split}.csv")
    with path.open(encoding="utf-8") as f:
        return [(row["text"], TWITTER_LABELS[row["label"]]) for row in csv.DictReader(f)]


def load_phrasebank(subset: str = "AllAgree") -> Dataset:
    path = _download(PHRASEBANK_URL, CACHE / "FinancialPhraseBank-v1.0.zip")
    with zipfile.ZipFile(path) as zf:
        raw = zf.read(f"FinancialPhraseBank-v1.0/Sentences_{subset}.txt").decode("latin-1")
    out: Dataset = []
    for line in io.StringIO(raw):
        line = line.strip()
        if "@" in line:
            text, label = line.rsplit("@", 1)
            out.append((text.strip(), PHRASEBANK_LABELS[label.strip()]))
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(gold: list[str], pred: list[str]) -> dict[str, object]:
    """Accuracy, macro-F1, per-class precision/recall/F1 and the confusion matrix."""
    per: dict[str, dict[str, float]] = {}
    for c in LABELS:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": tp + fn}
    confusion = {g: {p: sum(1 for a, b in zip(gold, pred) if a == g and b == p) for p in LABELS} for g in LABELS}
    return {
        "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(gold), 4),
        "macro_f1": round(sum(v["f1"] for v in per.values()) / len(LABELS), 4),
        "per_class": per,
        "confusion": confusion,
    }


def evaluate(engine: SentimentEngine, data: Dataset, batch: int = 256) -> tuple[dict[str, object], list[str]]:
    texts = [t for t, _ in data]
    preds: list[str] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i: i + batch]
        preds.extend(a.label for a in engine.score(chunk, ["news"] * len(chunk)))
    return metrics([g for _, g in data], preds), preds


def throughput(engine: SentimentEngine, texts: list[str], repeats: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        engine.score(texts, ["news"] * len(texts))
    return len(texts) * repeats / (time.perf_counter() - start)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _engines(names: Iterable[str]) -> dict[str, SentimentEngine]:
    out: dict[str, SentimentEngine] = {}
    for name in names:
        if name == "vader":
            out[name] = VaderEngine()
        elif name == "sentinel":
            out[name] = SentinelEngine()
        elif name == "finbert-api":
            from app.config import settings
            from app.nlp.transformer import FinBertApiEngine

            token = settings.hf_token or os.environ.get("HF_TOKEN", "")
            if token:
                out[name] = FinBertApiEngine(SentinelEngine(), token=token)
            else:
                print("finbert-api skipped: HF_TOKEN not set", file=sys.stderr)
    return out


def print_errors(engine: SentinelEngine, data: Dataset, preds: list[str], n: int, seed: int = 0) -> None:
    """Show misclassified examples with their evidence (use on the TRAIN split only)."""
    wrong = [(t, g, p) for (t, g), p in zip(data, preds) if g != p]
    buckets = Counter((g, p) for _, g, p in wrong)
    print(f"\n{len(wrong)} errors; buckets (gold->pred): " +
          ", ".join(f"{g}->{p}: {c}" for (g, p), c in buckets.most_common()))
    random.Random(seed).shuffle(wrong)
    for text, gold, pred in wrong[:n]:
        a = engine.analyze(text, "news")
        drivers = ", ".join(f"{t} {v:+.2f}" for t, v in a.drivers)
        print(f"[{gold[:4]}->{pred[:4]} {a.score:+.2f}] {' '.join(text.split())[:150]}\n      {drivers}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", default="vader,sentinel,finbert-api")
    ap.add_argument("--errors", type=int, default=0, help="print N misclassified Twitter-train examples")
    ap.add_argument("--train", action="store_true", help="also report the Twitter train split (tuning set)")
    ap.add_argument("--write", action="store_true", help=f"write {RESULTS_PATH.name}")
    args = ap.parse_args(argv)

    engines = _engines(n.strip() for n in args.engines.split(",") if n.strip())
    datasets: dict[str, Dataset] = {"twitter_valid": load_twitter("valid"), "phrasebank_allagree": load_phrasebank()}
    if args.train or args.errors:
        datasets = {"twitter_train": load_twitter("train"), **datasets}

    results: dict[str, dict[str, object]] = {}
    for dname, data in datasets.items():
        results[dname] = {}
        dist = Counter(g for _, g in data)
        print(f"\n== {dname} (n={len(data)}; " + ", ".join(f"{k} {v}" for k, v in sorted(dist.items())) + ")")
        for ename, engine in engines.items():
            m, preds = evaluate(engine, data)
            results[dname][ename] = m
            pc = m["per_class"]  # type: ignore[index]
            print(f"  {ename:12s} acc {m['accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}   " +
                  "  ".join(f"{c[:4]} F1 {pc[c]['f1']:.2f}" for c in LABELS))  # type: ignore[index]
            if dname == "twitter_train" and args.errors and isinstance(engine, SentinelEngine):
                print_errors(engine, data, preds, args.errors)

    speed_texts = [t for t, _ in datasets["twitter_valid"]]
    speeds = {name: round(throughput(e, speed_texts)) for name, e in engines.items() if name != "finbert-api"}
    print("\nthroughput (texts/s, single thread): " + ", ".join(f"{k} {v:,}" for k, v in speeds.items()))

    if args.write:
        payload = {
            "generated": date.today().isoformat(),
            "protocol": ("Lexicon, rules and blend weights tuned on twitter-financial-news-sentiment TRAIN only; "
                         "twitter_valid and phrasebank_allagree are held out. All texts scored with the 'news' "
                         "register; labels via NEUTRAL_BAND=0.05. VADER uses its standard +-0.05 thresholds."),
            "datasets": {
                "twitter_valid": {"source": "huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment "
                                            "(sent_valid.csv)", "license": "MIT",
                                  "n": len(datasets["twitter_valid"])},
                "phrasebank_allagree": {"source": "Financial PhraseBank v1.0 Sentences_AllAgree (Malo et al. 2014; "
                                                  "huggingface.co/datasets/takala/financial_phrasebank)",
                                        "license": "CC BY-NC-SA 3.0", "n": len(datasets["phrasebank_allagree"])},
            },
            "results": {k: v for k, v in results.items() if k != "twitter_train"},
            "tuning_split": results.get("twitter_train"),
            "throughput_texts_per_s": speeds,
        }
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
