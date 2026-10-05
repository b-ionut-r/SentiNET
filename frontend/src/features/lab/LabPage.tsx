/**
 * Sentiment lab: paste or upload text (one item per line, or a CSV column),
 * optionally scope to a ticker for relevance, and inspect the engine's
 * per-item scores, drivers, themes and events. Export results as CSV.
 */
import { Beaker, Download, FileUp, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { useLabScore } from "../../api/hooks";
import type { ScoreResponse } from "../../api/types";
import { SegmentLegend, StackedBar } from "../../components/charts/Bars";
import { Chip, Mark, PolarityChip, ScoreChip } from "../../components/ui/Badges";
import { DriverText } from "../../components/ui/DriverText";
import { MetaGroup } from "../../components/ui/MetaGroup";
import { Empty, ErrorState } from "../../components/ui/Misc";
import { Panel, SubHead } from "../../components/ui/Panel";
import { cx } from "../../lib/cx";
import { plural, signed } from "../../lib/format";
import { divergingFill, polarityOf, textTone } from "../../lib/sentiment";
import { eventLabel, themeLabel } from "../intel/themes";

const MAX_ITEMS = 500;

const EXAMPLE = [
  "Morgan Stanley raises Nvidia price target to $250 from $220, keeps Overweight",
  "Apple shares fall 3% after Bank of America flags AI-agent threat to App Store revenue",
  "Tesla recalls 120,000 vehicles over faulty seat belts",
  "Microsoft beats on revenue but guidance disappoints; stock slips after hours",
  "$SOFI diamond hands, this is going to the moon 🚀",
  "Is Apple a buy before the Oct 13 event?",
].join("\n");

/** Minimal CSV parser (quotes, escaped quotes, commas, CRLF). */
function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else cell += ch;
  }
  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }
  return rows.filter((r) => r.some((c) => c.trim()));
}

/** Pick the text column of a CSV: a header named text/title/headline/body, else the longest column. */
function textsFromCsv(raw: string): string[] {
  const rows = parseCsv(raw);
  if (!rows.length) return [];
  const header = rows[0].map((h) => h.trim().toLowerCase());
  let col = header.findIndex((h) => ["text", "title", "headline", "body", "message", "content"].includes(h));
  let body = rows.slice(1);
  if (col < 0) {
    const widths = rows[0].map((_, c) => rows.reduce((s, r) => s + (r[c]?.length ?? 0), 0));
    col = widths.indexOf(Math.max(...widths));
    body = rows;
  }
  return body.map((r) => (r[col] ?? "").trim()).filter(Boolean);
}

const csvCell = (v: string | number | null) => {
  const s = v == null ? "" : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

function exportCsv(res: ScoreResponse) {
  const lines = [["text", "score", "label", "confidence", "relevance", "drivers", "themes", "events"].join(",")];
  for (const r of res.results) {
    lines.push(
      [r.text, r.score.toFixed(3), r.label, r.confidence.toFixed(2), r.relevance == null ? "" : r.relevance.toFixed(2), r.drivers.map((d) => `${d.term}:${d.impact.toFixed(2)}`).join(" | "), r.themes.join(" | "), r.events.join(" | ")]
        .map(csvCell)
        .join(","),
    );
  }
  const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = "sentinet-lab.csv";
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function LabPage() {
  const [text, setText] = useState(EXAMPLE);
  const [ticker, setTicker] = useState("");
  const [fileNote, setFileNote] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const score = useLabScore();
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const over = lines.length > MAX_ITEMS;

  useEffect(() => {
    document.title = "Sentiment lab — SentiNET";
  }, []);

  const run = () => {
    if (!lines.length || over) return;
    score.mutate({ texts: lines, ticker: ticker.trim() ? ticker.trim().toUpperCase() : null });
  };

  const onFile = async (f: File) => {
    const raw = await f.text();
    const items = /\.csv$/i.test(f.name) ? textsFromCsv(raw) : raw.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    setText(items.join("\n"));
    setFileNote(`${f.name}: ${plural(items.length, "item")}${items.length > MAX_ITEMS ? ` (only the first ${MAX_ITEMS} can be scored)` : ""}`);
  };

  return (
    <div className="space-y-4">
      <div>
        <h1 className="flex items-center gap-2 text-[22px] font-semibold tracking-[-0.01em] text-ink">
          <Beaker className="size-5 text-muted" /> Sentiment lab
        </h1>
        <p className="mt-0.5 text-sm text-muted">Score any text with the same engine SentiNET uses — finance lexicon, rules, negation and hedging, explained term by term.</p>
      </div>
      <div className="grid gap-4 lg:grid-cols-12">
        <Panel title="Input" subtitle="One item per line, or upload a .txt / .csv" className="lg:col-span-5">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={12}
            spellCheck={false}
            className="field h-auto w-full resize-y py-2 font-mono text-xs leading-5"
            aria-label="Texts to score, one per line"
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
            }}
          />
          <div className="mt-1 flex justify-between text-2xs">
            <span className={cx(over ? "text-critical" : "text-muted")}>
              {plural(lines.length, "item")}
              {over && ` — max ${MAX_ITEMS}`}
            </span>
            {fileNote && <span className="truncate text-muted">{fileNote}</span>}
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <input value={ticker} onChange={(e) => setTicker(e.target.value)} placeholder="Ticker (optional)" className="field w-36 uppercase placeholder:normal-case" aria-label="Optional ticker for relevance scoring" />
            <input ref={fileRef} type="file" accept=".txt,.csv,text/plain,text/csv" className="hidden" onChange={(e) => e.target.files?.[0] && void onFile(e.target.files[0])} />
            <button className="btn" onClick={() => fileRef.current?.click()}>
              <FileUp className="size-3.5" /> Upload
            </button>
            <button className="btn" onClick={() => (setText(EXAMPLE), setFileNote(null))}>
              <Sparkles className="size-3.5" /> Example
            </button>
            <button className="btn btn-primary ml-auto" onClick={run} disabled={!lines.length || over || score.isPending}>
              {score.isPending ? "Scoring…" : "Score"}
            </button>
          </div>
          <p className="mt-2 text-2xs text-muted">⌘/Ctrl + Enter to score. Add a ticker to see how clearly each item is about it.</p>
        </Panel>
        <div className="space-y-4 lg:col-span-7">
          {score.error ? (
            <Panel title="Results">
              <ErrorState title="Scoring failed" message={score.error.message} onRetry={run} />
            </Panel>
          ) : !score.data ? (
            <Panel title="Results">
              <Empty icon={<Beaker />} title="Nothing scored yet">
                Press Score to run the engine. Every result shows the terms that moved it.
              </Empty>
            </Panel>
          ) : (
            <Results res={score.data} stale={score.isPending} />
          )}
        </div>
      </div>
    </div>
  );
}

function Results({ res, stale }: { res: ScoreResponse; stale: boolean }) {
  const s = res.summary;
  const weightedByRelevance = res.results.some((r) => r.relevance != null);
  const segs = [
    { key: "bull", label: "Bullish", value: s.bullish, color: "rgb(var(--bull))" },
    { key: "neu", label: "Neutral", value: s.neutral, color: "rgb(var(--mid))" },
    { key: "bear", label: "Bearish", value: s.bearish, color: "rgb(var(--bear))" },
  ];
  return (
    <div className={cx("space-y-4", stale && "is-refetching")}>
      <Panel
        title="Summary"
        subtitle={`engine ${res.engine} · ${plural(s.n, "item")}`}
        actions={
          <button className="btn h-7 text-xs" onClick={() => exportCsv(res)}>
            <Download className="size-3.5" /> CSV
          </button>
        }
      >
        <div className="grid gap-5 sm:grid-cols-2">
          <div>
            <div className="flex items-center gap-3">
              <span className={cx("text-[28px] font-semibold leading-none tracking-[-0.02em]", textTone[polarityOf(s.score)])}>{signed(s.score)}</span>
              <PolarityChip p={polarityOf(s.score)} />
            </div>
            <p className="mt-1.5 text-2xs text-muted">
              {weightedByRelevance ? "confidence- and relevance-weighted mean" : "confidence-weighted mean"} (−1 bearish … +1 bullish) · confidence {Math.round(s.confidence * 100)}%
            </p>
            <StackedBar segments={segs} height={8} className="mt-3" />
            <SegmentLegend segments={segs} className="mt-2" />
          </div>
          <div>
            <SubHead>Themes</SubHead>
            {res.themes.length === 0 ? (
              <p className="text-xs text-muted">No themes detected.</p>
            ) : (
              <ul className="space-y-1.5">
                {res.themes.slice(0, 6).map((t) => (
                  <li key={t.theme} className="grid grid-cols-[132px_minmax(0,1fr)_44px] items-center gap-2 text-xs">
                    <span className="truncate text-ink-2">{t.label}</span>
                    <div className="h-1.5 rounded-r-[3px]" style={{ width: `${Math.max(4, t.share * 100)}%`, background: divergingFill(t.score / 0.4) }} />
                    <span className={cx("text-right font-medium num", textTone[polarityOf(t.score)])}>{signed(t.score)}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </Panel>
      <Panel title="Items" subtitle="Driver terms underlined: teal pushes up, red pushes down" flush>
        <ul className="divide-hair hairline-t">
          {res.results.map((r, i) => (
            <li key={i} className="grid grid-cols-[64px_minmax(0,1fr)] gap-3 px-4 py-2.5">
              <div className="pt-px">
                <ScoreChip score={r.score} />
              </div>
              <div className="min-w-0">
                <p className="text-[13px] leading-[19px] text-ink"><DriverText text={r.text} drivers={r.drivers} /></p>
                <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-2xs text-muted">
                  <MetaGroup
                    items={[
                      <span key="c">confidence {Math.round(r.confidence * 100)}%</span>,
                      r.relevance != null ? (
                        <span key="r" className={cx(r.relevance < 0.35 && "font-medium text-ink-2")} title="How clearly the text is about the ticker; it scales the item's weight in the summary">
                          relevance {Math.round(r.relevance * 100)}%{r.relevance < 0.35 ? " — barely about this ticker" : ""}
                        </span>
                      ) : null,
                    ]}
                  />
                  <MetaGroup
                    items={r.drivers.slice(0, 4).map((d) => (
                      <span key={d.term} className="inline-flex max-w-[240px] items-center gap-1 text-ink-2" title={`${d.term} ${signed(d.impact)}`}>
                        <Mark p={d.impact > 0 ? "bull" : d.impact < 0 ? "bear" : "neutral"} className="text-[7px]" />
                        <span className="truncate">{d.term}</span>
                        <span className="num text-muted">{signed(d.impact)}</span>
                      </span>
                    ))}
                  />
                  {r.events.map((e) => (
                    <Chip key={e}>{eventLabel(e)}</Chip>
                  ))}
                  {r.themes.map((t) => (
                    <Chip key={t} tone="muted">
                      {themeLabel(t)}
                    </Chip>
                  ))}
                </div>
              </div>
            </li>
          ))}
        </ul>
      </Panel>
    </div>
  );
}
