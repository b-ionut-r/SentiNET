"""Benchmark the sentiment engines on real, labeled financial text.

    python -m scripts.eval_engine                  # VADER vs Sentinel on every held-out set
    python -m scripts.eval_engine --write          # ... and save tests/nlp/data/engine_eval_results.json
    python -m scripts.eval_engine --errors 40      # print misclassified *train* examples (tuning)
    python -m scripts.eval_engine --collect-stocktwits   # (re)collect the StockTwits sample

Datasets (downloaded once into $SENTINET_CACHE or ~/.cache/sentinet - never committed):

* zeroshot/twitter-financial-news-sentiment (MIT): finance-news tweets,
  0=bearish 1=bullish 2=neutral; ``train`` (the ONLY news tuning set) and ``valid`` (held out).
* Financial PhraseBank v1.0, Sentences_AllAgree (Malo et al. 2014, CC BY-NC-SA 3.0):
  company-release sentences with unanimous annotator labels (held out).
* FiQA 2018 task 1 (TheFinAI/fiqa-sentiment-classification, MIT): headlines and
  microblog posts with continuous scores; labels pre-registered with the FinGPT
  convention (score >= 0.1 bullish, < -0.1 bearish, else neutral). Headlines are
  scored with the news register, posts with the social register. Never inspected
  during tuning: the cleanest held-out set here.
* StockTwits author tags: public stream messages whose *authors* tagged them
  Bullish/Bearish (collected with ``--collect-stocktwits``). Even ids tuned the
  social register; odd ids are held out. Binary task: we report coverage (share
  given a non-neutral label) and accuracy / balanced accuracy on that share.

Protocol: lexicon, rules and blend weights were tuned on Twitter **train** only
(news) and StockTwits even ids (social). Everything else is reported as held out.
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
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import NamedTuple

from app.nlp import lexicon as lx
from app.nlp.engine import SentinelEngine, VaderEngine
from app.nlp.types import SentimentEngine, TextAnalysis

LABELS = ("bullish", "bearish", "neutral")
CACHE = Path(os.environ.get("SENTINET_CACHE", Path.home() / ".cache" / "sentinet"))
RESULTS_PATH = Path(__file__).resolve().parent.parent / "tests" / "nlp" / "data" / "engine_eval_results.json"
USER_AGENT = "SentiNET-eval/2.0"

TWITTER_URL = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/main/sent_{split}.csv"
PHRASEBANK_URL = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
FIQA_ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset=TheFinAI/fiqa-sentiment-classification"
                 "&config=default&split={split}&offset={offset}&length=100")
STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
TWITTER_LABELS = {"0": "bearish", "1": "bullish", "2": "neutral"}
PHRASEBANK_LABELS = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}
FIQA_THRESHOLD = 0.1  # FinGPT/PIXIU convention, fixed before looking at any result


class Example(NamedTuple):
    text: str
    gold: str  # bullish | bearish | neutral
    kind: str = "news"  # register the engine is told


Dataset = list[Example]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _get(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URLs
        return resp.read()


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url} -> {dest}", file=sys.stderr)
        dest.write_bytes(_get(url))
    return dest


def load_twitter(split: str) -> Dataset:
    """``split`` is "train" or "valid"."""
    path = _download(TWITTER_URL.format(split=split), CACHE / f"twitter_{split}.csv")
    with path.open(encoding="utf-8") as f:
        return [Example(row["text"], TWITTER_LABELS[row["label"]]) for row in csv.DictReader(f)]


def load_phrasebank(subset: str = "AllAgree") -> Dataset:
    path = _download(PHRASEBANK_URL, CACHE / "FinancialPhraseBank-v1.0.zip")
    with zipfile.ZipFile(path) as zf:
        raw = zf.read(f"FinancialPhraseBank-v1.0/Sentences_{subset}.txt").decode("latin-1")
    out: Dataset = []
    for line in io.StringIO(raw):
        line = line.strip()
        if "@" in line:
            text, label = line.rsplit("@", 1)
            out.append(Example(text.strip(), PHRASEBANK_LABELS[label.strip()]))
    return out


def _fiqa_rows() -> list[dict[str, object]]:
    """All FiQA task-1 rows (train+valid+test: none are used for tuning), cached as JSON."""
    path = CACHE / "fiqa_rows.json"
    if not path.exists():
        rows: list[dict[str, object]] = []
        for split in ("train", "valid", "test"):
            offset = 0
            while True:
                page = json.loads(_get(FIQA_ROWS_URL.format(split=split, offset=offset)))
                rows.extend({**r["row"], "split": split} for r in page["rows"])
                offset += 100
                if offset >= page["num_rows_total"]:
                    break
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def _fiqa_label(score: float) -> str:
    if score >= FIQA_THRESHOLD:
        return "bullish"
    return "bearish" if score < -FIQA_THRESHOLD else "neutral"


def load_fiqa(kind: str) -> Dataset:
    """FiQA ``kind`` = "headline" | "post". Sentences labelled differently for different
    aspect targets are dropped (sentence-level scoring cannot be right for both)."""
    labels: dict[str, set[str]] = {}
    for r in _fiqa_rows():
        if r["type"] == kind:
            labels.setdefault(str(r["sentence"]).strip(), set()).add(_fiqa_label(float(r["score"])))  # type: ignore[arg-type]
    register = "news" if kind == "headline" else "social"
    return [Example(text, next(iter(ls)), register) for text, ls in labels.items() if len(ls) == 1]


STOCKTWITS_SYMBOLS = """AAPL NVDA TSLA AMD MSFT AMZN META GOOGL NFLX PLTR SOFI GME AMC SPY QQQ IWM BTC.X ETH.X DOGE.X
SOL.X COIN HOOD RIVN LCID NIO BABA INTC MU SMCI ARM AVGO TSM CRWD SNOW SHOP PYPL SQ UBER DIS BA F GM XOM CVX JPM BAC
WFC C GS MARA RIOT MSTR UPST AFRM DKNG RBLX U NET DDOG ZM ROKU PINS SNAP LULU NKE SBUX WMT TGT COST MRNA PFE LLY NVO
ABBV BMY CVS UNH TLRY SNDL ACB CGC PLUG FCEL ENPH SPCE CCL AAL DAL NCLH OPEN SOUN IONQ RKLB HIMS CELH BBAI AI
PATH""".split()


def collect_stocktwits(pages: int = 2, pause: float = 1.1) -> Path:
    """Append author-tagged public StockTwits messages to the cache (polite: ~1 request/s)."""
    out = CACHE / "stocktwits_sample.json"
    rows: list[dict[str, object]] = json.loads(out.read_text())["messages"] if out.exists() else []
    seen = {r["id"] for r in rows}
    for symbol in STOCKTWITS_SYMBOLS:
        url = STOCKTWITS_URL.format(symbol=symbol)
        for _ in range(pages):
            try:
                msgs = json.loads(_get(url, timeout=15)).get("messages", [])
            except (urllib.error.URLError, ValueError) as exc:
                print(f"{symbol}: {exc}", file=sys.stderr)
                break
            for m in msgs:
                if m["id"] not in seen:
                    seen.add(m["id"])
                    tag = (((m.get("entities") or {}).get("sentiment") or {}).get("basic") or "").lower() or None
                    rows.append({"id": m["id"], "symbol": symbol, "created_at": m.get("created_at"),
                                 "body": m.get("body", ""), "tag": tag})
            if not msgs:
                break
            url = f"{STOCKTWITS_URL.format(symbol=symbol)}?max={msgs[-1]['id']}"
            time.sleep(pause)
        time.sleep(pause)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fetched": date.today().isoformat(), "source": "api.stocktwits.com streams/symbol",
                               "messages": rows}), encoding="utf-8")
    return out


def load_stocktwits(half: str = "odd") -> tuple[Dataset, str | None]:
    """Author-tagged messages; ``half`` "odd" (held out) or "even" (social tuning). Empty if not collected."""
    path = CACHE / "stocktwits_sample.json"
    if not path.exists():
        return [], None
    payload = json.loads(path.read_text(encoding="utf-8"))
    parity = 1 if half == "odd" else 0
    data = [Example(str(r["body"]), str(r["tag"]), "social") for r in payload["messages"]
            if r.get("tag") in ("bullish", "bearish") and int(r["id"]) % 2 == parity]
    return data, payload.get("fetched")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(gold: list[str], pred: list[str]) -> dict[str, object]:
    """Accuracy, macro-F1 (over the classes present in gold), per-class P/R/F1 and confusion."""
    pairs = list(zip(gold, pred, strict=True))
    present = [c for c in LABELS if c in gold]
    per: dict[str, dict[str, float]] = {}
    for c in LABELS:
        tp = sum(g == c and p == c for g, p in pairs)
        fp = sum(g != c and p == c for g, p in pairs)
        fn = sum(g == c and p != c for g, p in pairs)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": tp + fn}
    confusion = {g: {p: sum(1 for a, b in pairs if a == g and b == p) for p in LABELS} for g in present}
    return {
        "n": len(pairs),
        "accuracy": round(sum(g == p for g, p in pairs) / len(pairs), 4),
        "macro_f1": round(sum(per[c]["f1"] for c in present) / len(present), 4),
        "per_class": {c: per[c] for c in present},
        "confusion": confusion,
    }


def directional_metrics(gold: list[str], pred: list[str]) -> dict[str, object]:
    """For binary (bull/bear) gold: how often the engine commits, and how right it is when it does."""
    pairs = list(zip(gold, pred, strict=True))
    covered = [(g, p) for g, p in pairs if p != "neutral"]
    recall = {c: round(sum(g == c and p == c for g, p in covered) / max(1, sum(g == c for g, _ in covered)), 4)
              for c in ("bullish", "bearish")}
    return {
        "n": len(pairs),
        "gold": dict(Counter(gold)),
        "coverage": round(len(covered) / len(pairs), 4) if pairs else 0.0,
        "accuracy_covered": round(sum(g == p for g, p in covered) / max(1, len(covered)), 4),
        "balanced_accuracy_covered": round((recall["bullish"] + recall["bearish"]) / 2, 4),
        "recall_covered": recall,
        "flipped": round(sum(p not in (g, "neutral") for g, p in pairs) / max(1, len(pairs)), 4),
    }


def analyze(engine: SentimentEngine, data: Dataset, batch: int = 256) -> list[TextAnalysis]:
    out: list[TextAnalysis] = []
    for i in range(0, len(data), batch):
        chunk = data[i: i + batch]
        out.extend(engine.score([e.text for e in chunk], [e.kind for e in chunk]))
    return out


def predict(engine: SentimentEngine, data: Dataset, batch: int = 256) -> list[str]:
    return [a.label for a in analyze(engine, data, batch)]


def reliability(gold: list[str], analyses: list[TextAnalysis], bins: int = 10) -> dict[str, dict[str, float]]:
    """Does confidence match accuracy? Per predicted kind (polar / neutral): mean confidence, accuracy
    and expected calibration error (|accuracy - confidence| over equal-width bins, weighted by size)."""
    out: dict[str, dict[str, float]] = {}
    for kind, keep in (("polar", lambda lab: lab != "neutral"), ("neutral", lambda lab: lab == "neutral")):
        rows = [(a.confidence, a.label == g) for g, a in zip(gold, analyses, strict=True) if keep(a.label)]
        if not rows:
            continue
        groups: dict[int, list[tuple[float, bool]]] = {}
        for c, ok in rows:
            groups.setdefault(min(bins - 1, int(c * bins)), []).append((c, ok))
        ece = sum(abs(sum(ok for _, ok in g) / len(g) - sum(c for c, _ in g) / len(g)) * len(g)
                  for g in groups.values()) / len(rows)
        out[kind] = {"n": len(rows), "mean_confidence": round(sum(c for c, _ in rows) / len(rows), 4),
                     "accuracy": round(sum(ok for _, ok in rows) / len(rows), 4), "ece": round(ece, 4)}
    return out


def throughput(engine: SentimentEngine, texts: list[str], kind: str = "news", repeats: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        engine.score(texts, [kind] * len(texts))
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
    """Show misclassified examples with their drivers (use on the TRAIN split only)."""
    wrong = [(e, p) for e, p in zip(data, preds, strict=True) if e.gold != p]
    buckets = Counter((e.gold, p) for e, p in wrong)
    print(f"\n{len(wrong)} errors; buckets (gold->pred): " +
          ", ".join(f"{g}->{p}: {c}" for (g, p), c in buckets.most_common()))
    random.Random(seed).shuffle(wrong)
    for e, pred in wrong[:n]:
        a = engine.analyze(e.text, e.kind)
        drivers = ", ".join(f"{t} {v:+.2f}" for t, v in a.drivers)
        print(f"[{e.gold[:4]}->{pred[:4]} {a.score:+.2f}] {' '.join(e.text.split())[:150]}\n      {drivers}")


DATASET_INFO: dict[str, dict[str, str]] = {
    "twitter_train": {"source": "huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment (sent_train.csv)",
                      "license": "MIT", "register": "news", "role": "tuning"},
    "twitter_valid": {"source": "huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment (sent_valid.csv)",
                      "license": "MIT", "register": "news"},
    "phrasebank_allagree": {"source": "Financial PhraseBank v1.0 Sentences_AllAgree (Malo et al. 2014; "
                                      "huggingface.co/datasets/takala/financial_phrasebank)",
                            "license": "CC BY-NC-SA 3.0", "register": "news"},
    "fiqa_headlines": {"source": "FiQA 2018 task 1, headlines (huggingface.co/datasets/TheFinAI/"
                                 "fiqa-sentiment-classification, all splits)", "license": "MIT", "register": "news",
                       "labels": f"score >= {FIQA_THRESHOLD} bullish, < -{FIQA_THRESHOLD} bearish, else neutral"},
    "fiqa_posts": {"source": "FiQA 2018 task 1, microblog posts (same dataset, all splits)", "license": "MIT",
                   "register": "social",
                   "labels": f"score >= {FIQA_THRESHOLD} bullish, < -{FIQA_THRESHOLD} bearish, else neutral"},
    "stocktwits_even": {"source": "api.stocktwits.com public symbol streams; author-tagged; even message ids",
                        "register": "social", "role": "tuning"},
    "stocktwits_odd": {"source": "api.stocktwits.com public symbol streams; author-tagged Bullish/Bearish; odd "
                                 "message ids (even ids tuned the social register)", "register": "social"},
}
TUNING_SETS = ("twitter_train", "stocktwits_even")


def _summary(results: dict[str, dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    """Headline numbers per dataset and engine (the full metrics follow in the payload)."""
    keys = ("accuracy", "macro_f1", "coverage", "accuracy_covered", "balanced_accuracy_covered")
    out: dict[str, dict[str, dict[str, float]]] = {}
    for d, per in results.items():
        out[d] = {}
        for e, m in per.items():
            row = {k: m[k] for k in keys if k in m}  # type: ignore[operator,index]
            polar = m.get("polar") if isinstance(m, dict) else None
            if polar:
                row["polar_coverage"] = polar["coverage"]
                row["polar_accuracy_committed"] = polar["accuracy_covered"]
            out[d][e] = row
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", default="vader,sentinel,finbert-api")
    ap.add_argument("--errors", type=int, default=0, help="print N misclassified Twitter-train examples")
    ap.add_argument("--train", action="store_true", help="also report the tuning sets")
    ap.add_argument("--write", action="store_true", help=f"write {RESULTS_PATH.name}")
    ap.add_argument("--collect-stocktwits", action="store_true", help="(re)collect the StockTwits sample first")
    args = ap.parse_args(argv)

    if args.collect_stocktwits:
        print(f"collected -> {collect_stocktwits()}", file=sys.stderr)
    engines = _engines(n.strip() for n in args.engines.split(",") if n.strip())
    three_way: dict[str, Dataset] = {
        "twitter_valid": load_twitter("valid"),
        "phrasebank_allagree": load_phrasebank(),
        "fiqa_headlines": load_fiqa("headline"),
        "fiqa_posts": load_fiqa("post"),
    }
    if args.train or args.errors:
        three_way = {"twitter_train": load_twitter("train"), **three_way}
    binary: dict[str, Dataset] = {}
    stocktwits, fetched = load_stocktwits("odd")
    if stocktwits:
        binary["stocktwits_odd"] = stocktwits
        DATASET_INFO["stocktwits_odd"]["collected"] = fetched or "?"
        if args.train:
            binary = {"stocktwits_even": load_stocktwits("even")[0], **binary}
    else:
        print("stocktwits sample not collected (use --collect-stocktwits); skipping", file=sys.stderr)

    results: dict[str, dict[str, object]] = {}
    sizes: dict[str, int] = {}
    for dname, data in three_way.items():
        results[dname], sizes[dname] = {}, len(data)
        dist = Counter(e.gold for e in data)
        print(f"\n== {dname} (n={len(data)}; " + ", ".join(f"{k} {v}" for k, v in sorted(dist.items())) + ")")
        for ename, engine in engines.items():
            analyses = analyze(engine, data)
            preds = [a.label for a in analyses]
            m = metrics([e.gold for e in data], preds)
            if ename != "vader":  # VADER's "confidence" is just |compound|
                m["calibration"] = reliability([e.gold for e in data], analyses)
            # on the polar (bullish/bearish) gold items: how often it commits, how right when it does
            polar = [(e.gold, p) for e, p in zip(data, preds, strict=True) if e.gold != "neutral"]
            d = directional_metrics([g for g, _ in polar], [p for _, p in polar])
            m["polar"] = {k: d[k] for k in ("n", "coverage", "accuracy_covered", "flipped")}
            results[dname][ename] = m
            pc = m["per_class"]  # type: ignore[index]
            print(f"  {ename:12s} acc {m['accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}   " +
                  "  ".join(f"{c[:4]} F1 {pc[c]['f1']:.2f}" for c in pc) +  # type: ignore[index]
                  f"   | polar: commits {d['coverage']:.2f}, right when committed {d['accuracy_covered']:.3f}")
            cal = m.get("calibration")
            if cal:
                print("               calibration: " + "; ".join(
                    f"{k} conf {v['mean_confidence']:.2f} vs acc {v['accuracy']:.2f} (ECE {v['ece']:.3f})"
                    for k, v in cal.items()))  # type: ignore[union-attr]
            if dname == "twitter_train" and args.errors and isinstance(engine, SentinelEngine):
                print_errors(engine, data, preds, args.errors)
    for dname, data in binary.items():
        results[dname], sizes[dname] = {}, len(data)
        print(f"\n== {dname} (n={len(data)}; " + ", ".join(f"{k} {v}" for k, v in Counter(e.gold for e in data).items())
              + ")")
        for ename, engine in engines.items():
            m = directional_metrics([e.gold for e in data], predict(engine, data))
            results[dname][ename] = m
            print(f"  {ename:12s} coverage {m['coverage']:.2f}  acc(committed) {m['accuracy_covered']:.3f}  "
                  f"balanced {m['balanced_accuracy_covered']:.3f}  flipped {m['flipped']:.3f}")

    speed_texts = [e.text for e in three_way["twitter_valid"]]
    speeds = {name: round(throughput(e, speed_texts)) for name, e in engines.items() if name != "finbert-api"}
    print("\nthroughput (texts/s, single thread, twitter_valid): " + ", ".join(f"{k} {v:,}" for k, v in speeds.items()))

    if args.write:
        payload = {
            "generated": date.today().isoformat(),
            "protocol": (
                "News lexicon/rules/blend tuned on twitter-financial-news-sentiment TRAIN only; social register on "
                "StockTwits even ids. Everything under 'held_out' was not used for tuning. Labels via "
                "engine.label_for (NEUTRAL_BAND=0.05); VADER uses its standard +-0.05 compound thresholds. "
                "Caveat: the first build session inspected some PhraseBank AllAgree and Twitter-valid sentences "
                "(a few appeared verbatim in unit tests; replaced), so those two numbers may be mildly optimistic. "
                "FiQA texts and errors were never inspected (only label counts and aggregate scores at a few "
                "checkpoints; no setting was chosen on them), so it is the cleanest held-out estimate. FiQA is "
                "~88% polar, which penalizes an engine calibrated to abstain (neutral) on weak evidence: the news "
                "dead zone (0.28) was fitted to Twitter's neutral-heavy labels, so on FiQA headlines Sentinel commits "
                "on only ~half the polar items (see 'polar': share of polar items it commits on, and its accuracy "
                "when it does). Review fixes (sign flips on guidance metrics, negated approvals, size-cap compounds, "
                "the retailer Target, bare 'record', signed percents, 'stock up/down', forum prose, limited upside, "
                "Form-4 plan trades, enforcement actions, 'wiped out', rating infinitives, 13F bot headlines, "
                "purpose and relative clauses; final review: distance below a high, valuation calls, rating "
                "templates, bad-news quantities such as recalls or withdrawal queues, 'top analyst' column "
                "titles) were selected on the Twitter train split and hand-written regression cases only; "
                "held-out sets were re-run for reporting, not selection. Subject attribution (the optional "
                "per-text target: peers' and the market's clauses count as context) is not exercised here, since "
                "no benchmark names an analysed company; it is covered by unit tests on live headlines. "
                "'calibration': "
                "news polar confidence is remapped by an isotonic fit on Twitter train (ECE on train/valid shows "
                "the fit holds); neutral confidence is deliberately prior-agnostic (0.5 when nothing was found, "
                "higher for explicit 'in line'/'unchanged' cues), because whether 'no evidence' means neutral "
                "depends on the stream's base rate (about 90% of neutral calls are right on Twitter, under 20% on "
                "the ~88%-polar FiQA sets), so neutral ECE is large on every set by construction."),
            "datasets": {k: {**DATASET_INFO[k], "n": sizes[k]} for k in results},
            "summary": _summary(results),
            "held_out": {k: v for k, v in results.items() if k not in TUNING_SETS},
            "tuning": {k: v for k, v in results.items() if k in TUNING_SETS} or None,
            "throughput_texts_per_s": speeds,
            "lexicon": lx.lexicon_stats(),
        }
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
