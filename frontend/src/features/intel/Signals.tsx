/**
 * Signal explorer: every kept item with its score, provenance and weight,
 * driver terms highlighted inline. One filter row scopes the whole list.
 */
import { ExternalLink, ListFilter, Search } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";

import type { Analysis, Driver, Signal } from "../../api/types";
import { Chip, ScoreChip } from "../../components/ui/Badges";
import { Empty, Segmented } from "../../components/ui/Misc";
import { Panel } from "../../components/ui/Panel";
import { cx } from "../../lib/cx";
import { plural, signed, timeAgo } from "../../lib/format";
import { polarityOf } from "../../lib/sentiment";
import { eventLabel, themeLabel } from "./themes";

type Kind = "all" | "news" | "social";
type Tone = "all" | "bull" | "bear" | "neutral";
type Sort = "weight" | "recent" | "bull" | "bear";

const PAGE = 25;

export function SignalExplorer({ a }: { a: Analysis }) {
  const [q, setQ] = useState("");
  const [kind, setKind] = useState<Kind>("all");
  const [tone, setTone] = useState<Tone>("all");
  const [source, setSource] = useState("all");
  const [theme, setTheme] = useState("all");
  const [sort, setSort] = useState<Sort>("weight");
  const [limit, setLimit] = useState(PAGE);

  const sources = useMemo(() => [...new Map(a.signals.map((s) => [s.source, s.source_label])).entries()], [a.signals]);
  const themes = useMemo(() => [...new Set(a.signals.flatMap((s) => s.themes))].sort(), [a.signals]);

  const rows = useMemo(() => {
    const term = q.trim().toLowerCase();
    const out = a.signals.filter(
      (s) =>
        (kind === "all" || s.kind === kind) &&
        (tone === "all" || polarityOf(s.score) === tone) &&
        (source === "all" || s.source === source) &&
        (theme === "all" || s.themes.includes(theme)) &&
        (!term || `${s.title} ${s.body ?? ""} ${s.publisher ?? ""} ${s.author ?? ""}`.toLowerCase().includes(term)),
    );
    const ts = (s: Signal) => (s.timestamp ? Date.parse(s.timestamp) : 0);
    out.sort((x, y) => (sort === "weight" ? y.weight - x.weight : sort === "recent" ? ts(y) - ts(x) : sort === "bull" ? y.score - x.score : x.score - y.score));
    return out;
  }, [a.signals, q, kind, tone, source, theme, sort]);

  const reset = () => setLimit(PAGE);

  return (
    <Panel id="signals" title="Signal explorer" icon={<ListFilter />} subtitle={`${plural(a.signals.length, "item")} kept after relevance filtering and de-duplication`} flush>
      <div className="flex flex-wrap items-center gap-2 px-4 pb-3">
        <label className="relative min-w-[200px] flex-1 sm:max-w-xs">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted" aria-hidden />
          <input
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              reset();
            }}
            placeholder="Filter text, outlet, author…"
            className="field w-full pl-8"
            aria-label="Filter signals"
          />
        </label>
        <Segmented<Kind>
          label="Kind"
          size="xs"
          value={kind}
          onChange={(v) => {
            setKind(v);
            reset();
          }}
          options={[
            { value: "all", label: "All" },
            { value: "news", label: "News" },
            { value: "social", label: "Social" },
          ]}
        />
        <Segmented<Tone>
          label="Sentiment"
          size="xs"
          value={tone}
          onChange={(v) => {
            setTone(v);
            reset();
          }}
          options={[
            { value: "all", label: "Any" },
            { value: "bull", label: <span><span className="text-[8px] text-bull">▲</span> Bull</span> },
            { value: "bear", label: <span><span className="text-[8px] text-bear">▼</span> Bear</span> },
            { value: "neutral", label: "Neutral" },
          ]}
        />
        <select className="field h-7 pr-7 text-xs" value={source} onChange={(e) => (setSource(e.target.value), reset())} aria-label="Source">
          <option value="all">All sources</option>
          {sources.map(([k, l]) => (
            <option key={k} value={k}>
              {l}
            </option>
          ))}
        </select>
        <select className="field h-7 pr-7 text-xs" value={theme} onChange={(e) => (setTheme(e.target.value), reset())} aria-label="Theme">
          <option value="all">All themes</option>
          {themes.map((t) => (
            <option key={t} value={t}>
              {themeLabel(t)}
            </option>
          ))}
        </select>
        <select className="field h-7 pr-7 text-xs sm:ml-auto" value={sort} onChange={(e) => setSort(e.target.value as Sort)} aria-label="Sort">
          <option value="weight">Sort: weight</option>
          <option value="recent">Sort: newest</option>
          <option value="bull">Sort: most bullish</option>
          <option value="bear">Sort: most bearish</option>
        </select>
      </div>
      {rows.length === 0 ? (
        <Empty title="No signals match these filters" className="hairline-t" />
      ) : (
        <ul className="divide-hair hairline-t">
          {rows.slice(0, limit).map((s) => (
            <SignalRow key={s.id} s={s} />
          ))}
        </ul>
      )}
      <div className="flex items-center justify-between px-4 py-2.5 text-xs text-muted hairline-t">
        <span>
          Showing {Math.min(limit, rows.length)} of {rows.length}
        </span>
        {rows.length > limit && (
          <button className="font-medium text-ink-2 hover:text-ink" onClick={() => setLimit((l) => l + PAGE * 2)}>
            Show more
          </button>
        )}
      </div>
    </Panel>
  );
}

function escapeRe(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Wrap driver terms in the text with a polarity underline. */
function highlight(text: string, drivers: Driver[]): ReactNode {
  const terms = drivers.filter((d) => d.term.trim().length > 1);
  if (!terms.length) return text;
  const map = new Map(terms.map((d) => [d.term.toLowerCase(), d.impact]));
  const re = new RegExp(`(${terms.map((d) => escapeRe(d.term)).sort((x, y) => y.length - x.length).join("|")})`, "gi");
  return text.split(re).map((part, i) => {
    const impact = map.get(part.toLowerCase());
    if (impact == null) return part;
    return (
      <mark
        key={i}
        title={`driver ${signed(impact)}`}
        className={cx("bg-transparent text-ink underline decoration-2 underline-offset-[3px]", impact > 0 ? "decoration-bull/80" : impact < 0 ? "decoration-bear/80" : "decoration-neu/70")}
      >
        {part}
      </mark>
    );
  });
}

function SignalRow({ s }: { s: Signal }) {
  const meta: ReactNode[] = [
    <span key="pub" className="font-medium text-ink-2">
      {s.publisher ?? (s.author ? `@${s.author}` : s.source_label)}
    </span>,
    s.publisher || s.author ? <span key="src">{s.source_label}</span> : null,
    <span key="t">{timeAgo(s.timestamp)}</span>,
    <span key="rel" title="How clearly the item is about this ticker">
      relevance {Math.round(s.relevance * 100)}%
    </span>,
    <span key="w" title="Aggregation weight: source trust × recency × engagement × relevance × confidence">
      weight {s.weight.toFixed(2)}
    </span>,
    s.duplicates > 0 ? (
      <span key="dup" className="text-ink-2">
        ×{s.duplicates + 1} syndicated
      </span>
    ) : null,
    s.engagement > 0 ? <span key="eng">{s.engagement} engagement</span> : null,
  ].filter(Boolean);
  return (
    <li className="grid grid-cols-[64px_minmax(0,1fr)] gap-3 px-4 py-2.5">
      <div className="pt-px">
        <ScoreChip score={s.score} />
      </div>
      <div className="min-w-0">
        <div className="flex items-start gap-2">
          <p className={cx("min-w-0 flex-1 text-[13px] leading-[19px]", s.kind === "news" ? "text-ink" : "text-ink-2")}>
            {s.url ? (
              <a href={s.url} target="_blank" rel="noreferrer" className="hover:text-ink">
                {highlight(s.title, s.drivers)}
              </a>
            ) : (
              highlight(s.title, s.drivers)
            )}
          </p>
          {s.url && (
            <a href={s.url} target="_blank" rel="noreferrer" className="mt-0.5 text-faint hover:text-ink-2" aria-label="Open source">
              <ExternalLink className="size-3.5" />
            </a>
          )}
        </div>
        <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-2xs text-muted">
          {meta.map((m, i) => (
            <span key={i} className="inline-flex items-center gap-2">
              {i > 0 && <span className="text-faint">·</span>}
              {m}
            </span>
          ))}
          {s.user_label && (
            <Chip tone={s.user_label === "bullish" ? "bull" : s.user_label === "bearish" ? "bear" : "neutral"}>
              author tag: {s.user_label}
            </Chip>
          )}
          {s.events.slice(0, 2).map((e) => (
            <Chip key={e} tone="neutral">
              {eventLabel(e)}
            </Chip>
          ))}
          {s.themes.slice(0, 2).map((t) => (
            <Chip key={t} tone="muted">
              {themeLabel(t)}
            </Chip>
          ))}
        </div>
      </div>
    </li>
  );
}
