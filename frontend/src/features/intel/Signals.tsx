/**
 * Signal explorer: every kept item with its score, provenance and weight,
 * driver terms highlighted inline. One filter row scopes the whole list.
 */
import { ExternalLink, ListFilter, Search } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";

import type { Analysis, Signal } from "../../api/types";
import { Chip, ScoreChip } from "../../components/ui/Badges";
import { DriverText } from "../../components/ui/DriverText";
import { MetaGroup } from "../../components/ui/MetaGroup";
import { Empty, Segmented } from "../../components/ui/Misc";
import { Panel } from "../../components/ui/Panel";
import { cx } from "../../lib/cx";
import { compact, plural, timeAgo } from "../../lib/format";
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

  // Nothing collected at all: filters would only imply hidden data, so explain instead.
  if (a.signals.length === 0) {
    return (
      <Panel id="signals" title="Signal explorer" icon={<ListFilter />} subtitle="Every scored news and social item behind the verdict">
        <Empty title="No news or social items were collected this run">
          Every text source came back empty or failed — see{" "}
          <a href="#sources" className="link">
            Source health
          </a>{" "}
          for what each one returned.
        </Empty>
      </Panel>
    );
  }

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
            { value: "bull", label: <span><span className="text-[8px] text-bull-ink">▲</span> Bull</span> },
            { value: "bear", label: <span><span className="text-[8px] text-bear-ink">▼</span> Bear</span> },
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

function SignalRow({ s }: { s: Signal }) {
  // Social posts are about who said it; news is about which outlet ran it.
  const who = s.kind === "social" && s.author ? `@${s.author.replace(/^@/, "")}` : s.publisher ?? (s.author ? `@${s.author.replace(/^@/, "")}` : null);
  const sameAsSource = !who || who.toLowerCase().replace(/[^a-z0-9]/g, "") === s.source_label.toLowerCase().replace(/[^a-z0-9]/g, "");
  const provenance: ReactNode[] = [
    <span key="pub" className="font-medium text-ink-2">
      {who ?? s.source_label}
    </span>,
    sameAsSource ? null : <span key="src">{s.source_label}</span>,
    <span key="t" title={s.timestamp ?? undefined}>
      {timeAgo(s.timestamp)}
    </span>,
  ];
  const metrics: ReactNode[] = [
    <span key="rel" title="How clearly the item is about this ticker">
      relevance {Math.round(s.relevance * 100)}%
    </span>,
    <span key="w" title="Aggregation weight: source trust × recency × engagement × relevance × confidence">
      weight {s.weight.toFixed(2)}
    </span>,
  ];
  const reach: ReactNode[] = [
    s.duplicates > 0 ? (
      <span key="dup" className="text-ink-2">
        ×{s.duplicates + 1} syndicated
      </span>
    ) : null,
    s.engagement > 0 ? <span key="eng">{compact(s.engagement)} engagement</span> : null,
  ];
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
                <DriverText text={s.title} drivers={s.drivers} />
              </a>
            ) : (
              <DriverText text={s.title} drivers={s.drivers} />
            )}
          </p>
          {s.url && (
            <a href={s.url} target="_blank" rel="noreferrer" className="mt-0.5 text-muted hover:text-ink-2" aria-label="Open source">
              <ExternalLink className="size-3.5" />
            </a>
          )}
        </div>
        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-2xs text-muted">
          <MetaGroup items={provenance} />
          <MetaGroup items={metrics} />
          <MetaGroup items={reach} />
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
