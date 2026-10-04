"""Re-capture the intel test fixtures from live providers (run manually).

    cd backend && python -m tests.intel.capture_fixtures [--only yahoo,sec,market,gdelt,wiki]

yfinance DataFrames are stored pickle-free as CSV (index included) or JSON so the
pure transforms in `app.intel.transforms` can be tested offline on real shapes.
Payloads are trimmed to keep the repo light. Rate-limited providers (GDELT,
Wikipedia) are skipped with a message when they refuse.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "intel"
UA_CONTACT = "SentiNET/2.0 (sentinet@example.com)"
UA_BROWSER = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
YAHOO_SYMBOLS = ["NVDA", "AAPL", "SOFI", "SPY", "BTC-USD", "JPM"]
SEC_SYMBOLS = {"NVDA": "0001045810", "SOFI": "0001818874", "JPM": "0000019617"}


def _write(rel: str, data: Any) -> None:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (dict, list)):
        path.write_text(json.dumps(data, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
    else:
        path.write_text(str(data), encoding="utf-8")
    print(f"  wrote {rel} ({path.stat().st_size // 1024} KB)")


def capture_yahoo() -> None:
    import yfinance as yf

    for sym in YAHOO_SYMBOLS:
        print(sym)
        t = yf.Ticker(sym)
        info = {k: v for k, v in t.info.items() if k not in {"companyOfficers"}}
        _write(f"yahoo/{sym}_info.json", info)
        daily = t.history(period="2y", interval="1d", auto_adjust=False, actions=False)
        (OUT / "yahoo").mkdir(parents=True, exist_ok=True)
        daily.tail(420).to_csv(OUT / f"yahoo/{sym}_daily.csv", float_format="%.4f")
        if sym in {"NVDA", "BTC-USD"}:
            t.history(period="1d", interval="5m", auto_adjust=False,
                      actions=False).to_csv(OUT / f"yahoo/{sym}_5m.csv", float_format="%.4f")
        if sym in {"SPY", "BTC-USD"}:
            time.sleep(0.5)
            continue
        recs = t.recommendations
        if recs is not None and not recs.empty:
            recs.to_csv(OUT / f"yahoo/{sym}_recommendations.csv", index=False)
        ud = t.upgrades_downgrades
        if ud is not None and not ud.empty:
            ud.head(80).to_csv(OUT / f"yahoo/{sym}_upgrades.csv")
        cal = t.calendar
        if cal:
            _write(f"yahoo/{sym}_calendar.json", cal)
        ed = t.get_earnings_dates(limit=16)
        if ed is not None and not ed.empty:
            ed.to_csv(OUT / f"yahoo/{sym}_earnings_dates.csv")
        ins = t.insider_transactions
        if ins is not None and not ins.empty:
            ins.head(120).to_csv(OUT / f"yahoo/{sym}_insiders.csv", index=False)
        time.sleep(0.5)
    df = yf.download(["SPY", "QQQ", "DIA", "IWM", "^VIX", "^TNX", "GC=F", "BTC-USD"], period="3mo",
                     interval="1d", auto_adjust=False, progress=False, group_by="column", threads=True)
    df.to_csv(OUT / "yahoo/indices.csv", float_format="%.4f")
    for q in ["apple", "nvidia", "btc"]:
        _write(f"yahoo/search_{q}.json", yf.Search(q, max_results=16, news_count=0, lists_count=0).quotes)


def _trim_submissions(sub: dict[str, Any], keep: int = 600) -> dict[str, Any]:
    fields = ["accessionNumber", "filingDate", "reportDate", "form", "items", "primaryDocument", "primaryDocDescription"]
    recent = sub["filings"]["recent"]
    sub["filings"] = {"recent": {f: recent[f][:keep] for f in fields}, "files": []}
    return {k: sub[k] for k in ("cik", "name", "tickers", "exchanges", "sicDescription", "filings")}


def capture_sec() -> None:
    c = httpx.Client(timeout=30, headers={"User-Agent": UA_CONTACT})
    tickers = c.get("https://www.sec.gov/files/company_tickers.json").json()
    wanted = {"NVDA", "AAPL", "SOFI", "JPM", "BRK-B", "GOOGL", "GOOG", "SPY", "META", "TGT", "XYZ", "AMZN", "MSFT"}
    sample = {k: v for i, (k, v) in enumerate(tickers.items()) if i < 60 or v["ticker"] in wanted}
    _write("sec/company_tickers_sample.json", sample)
    for sym, cik in SEC_SYMBOLS.items():
        time.sleep(0.3)
        sub = c.get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
        _write(f"sec/submissions_{sym}.json", _trim_submissions(sub))
        if sym == "SOFI":
            recent = sub["filings"]["recent"]
            n = 0
            for i, form in enumerate(recent["form"]):
                if form != "4":
                    continue
                acc = recent["accessionNumber"][i].replace("-", "")
                doc = recent["primaryDocument"][i].split("/")[-1]
                time.sleep(0.3)
                xml = c.get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}").text
                _write(f"sec/form4_{sym}_{n}.xml", xml)
                n += 1
                if n == 3:
                    break


def capture_market() -> None:
    c = httpx.Client(timeout=30, headers={"User-Agent": UA_BROWSER}, follow_redirects=True)
    cnn = c.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                headers={"Referer": "https://www.cnn.com/", "Origin": "https://www.cnn.com"}).json()
    for key, block in cnn.items():
        if key != "fear_and_greed_historical" and isinstance(block, dict) and "data" in block:
            block["data"] = block["data"][-10:]
    _write("market/cnn_fear_greed.json", cnn)
    _write("market/crypto_fng.json", c.get("https://api.alternative.me/fng/", params={"limit": 400}).json())
    ape = c.get("https://apewisdom.io/api/v1.0/filter/all-stocks/page/1").json()
    ape["results"] = ape["results"][:40]
    _write("market/apewisdom.json", ape)
    st = c.get("https://api.stocktwits.com/api/2/trending/symbols.json").json()
    for s in st.get("symbols", []):
        s.pop("fundamentals", None)
    _write("market/stocktwits_trending.json", st)
    from app.intel.market import FEEDS

    for key, url, _pub, _filt in FEEDS:
        r = c.get(url)
        if r.status_code == 200:
            text = r.text
            if key == "google_news":  # keep ~40 items
                head, _, rest = text.partition("<item>")
                items = rest.split("<item>")[:40]
                text = head + "<item>" + "<item>".join(items)
                if not text.rstrip().endswith("</rss>"):
                    text = text[: text.rfind("</item>") + 7] + "</channel></rss>"
            _write(f"market/feed_{key}.xml", text)
        else:
            print(f"  skip {key}: HTTP {r.status_code}")


def capture_gdelt() -> None:
    c = httpx.Client(timeout=40, headers={"User-Agent": UA_CONTACT})
    for mode in ("timelinetone", "timelinevolraw"):
        r = c.get("https://api.gdeltproject.org/api/v2/doc/doc",
                  params={"query": '"Nvidia" sourcelang:english', "mode": mode, "timespan": "90d", "format": "json"})
        if r.text.lstrip().startswith("{"):
            _write(f"gdelt/nvidia_{mode}.json", r.json())
        else:
            print(f"  GDELT {mode} refused: {r.status_code} {r.text[:80]!r}")
        time.sleep(7)


def capture_wiki() -> None:
    c = httpx.Client(timeout=30, headers={"User-Agent": UA_CONTACT})
    r = c.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query", "format": "json", "formatversion": "2", "redirects": "1", "generator": "search",
        "gsrsearch": 'Target Corporation hastemplate:"Infobox company"', "gsrlimit": "5", "gsrnamespace": "0",
        "prop": "pageviews|description", "pvipdays": "60"})
    if r.status_code == 200:
        _write("wiki/target_search.json", r.json())
    else:
        print(f"  Wikipedia refused: {r.status_code}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="yahoo,sec,market,gdelt,wiki")
    for part in parser.parse_args().only.split(","):
        print(f"== {part}")
        globals()[f"capture_{part.strip()}"]()


if __name__ == "__main__":
    main()
