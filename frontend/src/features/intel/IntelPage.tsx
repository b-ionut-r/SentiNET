/**
 * Intel page: one ticker, fully synthesized. Hierarchy is deliberate —
 * verdict → insights → narratives → the case → price × tone → smart money →
 * crowd → themes → raw signals → filings & source health.
 */
import { RefreshCw, Star } from "lucide-react";
import { lazy, Suspense, useEffect, useMemo } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ApiError } from "../../api/client";
import { useAnalysis, useSearch, useWatchlist, useWatchToggle } from "../../api/hooks";
import type { Analysis } from "../../api/types";
import { useCommands, usePageCommands } from "../../components/layout/commands";
import { ErrorState, Skeleton, TickerLogo, TopProgress } from "../../components/ui/Misc";
import { Panel } from "../../components/ui/Panel";
import { cx } from "../../lib/cx";
import { useHotkeys } from "../../lib/hotkeys";
import { getRecent, pushRecent } from "../../lib/storage";
import { AnalystsPanel, EarningsPanel, InsidersPanel } from "./SmartMoney";
import { CasePanel, WatchNext } from "./Case";
import { CrowdPanel } from "./Crowd";
import { FilingsPanel, SourcesPanel } from "./Health";
import { HeaderStrip } from "./HeaderStrip";
import { InsightsRail } from "./Insights";
import { Narratives } from "./Narratives";
import { ScanPanel } from "./ScanPanel";
import { SectionNav } from "./SectionNav";
import { SignalExplorer } from "./Signals";
import { ThemesPanel } from "./Themes";
import { VerdictHero } from "./VerdictHero";

const PriceTone = lazy(() => import("./PriceTone"));

export default function IntelPage() {
  const { ticker: raw = "" } = useParams();
  const ticker = raw.trim().toUpperCase(); // already URL-decoded by the router
  const navigate = useNavigate();
  const q = useAnalysis(ticker);
  const watchlist = useWatchlist();
  const toggleWatch = useWatchToggle();
  const watched = !!watchlist.data?.some((w) => w.ticker === ticker);
  const data = q.data;

  useEffect(() => {
    if (data) pushRecent(data.ticker);
  }, [data]);

  useEffect(() => {
    const v = data?.verdict;
    document.title = v ? `${ticker} ${v.score} · ${v.label} — SentiNET` : `${ticker} — SentiNET`;
    return () => {
      document.title = "SentiNET — Market Intelligence";
    };
  }, [ticker, data]);

  const onWatch = () => toggleWatch.mutate({ ticker, watched });

  useHotkeys({ r: () => !q.isFetching && q.refresh(), w: onWatch });
  usePageCommands([
    { id: "refresh", title: `Refresh ${ticker}`, hint: "r", icon: <RefreshCw className="size-4" />, run: q.refresh },
    { id: "watch", title: watched ? `Remove ${ticker} from watchlist` : `Add ${ticker} to watchlist`, hint: "w", icon: <Star className="size-4" />, run: onWatch },
    { id: "compare", title: `Compare ${ticker} with…`, run: () => navigate(`/compare?t=${encodeURIComponent(ticker)}`) },
  ]);

  if (!data) {
    if (q.error) {
      const e = q.error;
      const notFound = e instanceof ApiError && (e.status === 400 || e.status === 404);
      return notFound ? <TickerNotFound ticker={ticker} message={e.message} /> : <ErrorState title="Analysis failed" message={e.message} onRetry={() => void q.refetch()} />;
    }
    return <ScanPanel ticker={ticker} progress={q.progress} />;
  }

  return (
    <>
      <TopProgress active={q.isFetching} />
      <IntelView a={data} refetching={q.isFetching} watched={watched} onWatch={onWatch} onRefresh={q.refresh} progressCount={q.progress.length} />
    </>
  );
}

function IntelView({
  a,
  refetching,
  watched,
  onWatch,
  onRefresh,
  progressCount,
}: {
  a: Analysis;
  refetching: boolean;
  watched: boolean;
  onWatch: () => void;
  onRefresh: () => void;
  progressCount: number;
}) {
  const narrativeSignals = useMemo(() => {
    const map = new Map<string, Analysis["signals"]>();
    for (const s of a.signals) {
      if (!s.narrative_id) continue;
      const list = map.get(s.narrative_id) ?? [];
      list.push(s);
      map.set(s.narrative_id, list);
    }
    return map;
  }, [a.signals]);

  return (
    <div className={cx("space-y-4 transition-opacity", refetching && "is-refreshing")}>
      <HeaderStrip a={a} watched={watched} onWatch={onWatch} onRefresh={onRefresh} refreshing={refetching} progressCount={progressCount} />
      <SectionNav a={a} />
      <VerdictHero a={a} />

      {/* Two independent columns on desktop (story + case | insights + catalysts) so a short
          column never forces gaps into the other; on phones the wrappers dissolve
          (display: contents) and `order` restores reading order: insights, stories, case, catalysts. */}
      <div className="flex flex-col gap-4 lg:grid lg:grid-cols-12 lg:items-start">
        <div className="contents lg:col-span-8 lg:flex lg:flex-col lg:gap-4">
          <Narratives a={a} membersOf={narrativeSignals} className="order-2 lg:order-none" />
          <CasePanel a={a} className="order-3 lg:order-none" />
        </div>
        <div className="contents lg:col-span-4 lg:flex lg:flex-col lg:gap-4">
          <InsightsRail insights={a.insights} evidence={a.sentiment.n} className="order-1 lg:order-none" />
          <WatchNext a={a} className="order-4 lg:order-none" />
        </div>
      </div>

      <Suspense fallback={<Skeleton className="h-[460px] w-full rounded-xl" />}>
        <PriceTone a={a} />
      </Suspense>

      <section id="smart-money" className="grid scroll-mt-36 md:scroll-mt-28 gap-4 lg:grid-cols-3">
        <AnalystsPanel a={a} />
        <InsidersPanel a={a} />
        <EarningsPanel a={a} />
      </section>

      <div className="grid gap-4 lg:grid-cols-12">
        <CrowdPanel a={a} className="lg:col-span-7" />
        <ThemesPanel a={a} className="lg:col-span-5" />
      </div>

      <SignalExplorer a={a} />

      <div className="grid gap-4 lg:grid-cols-12">
        <FilingsPanel a={a} className="lg:col-span-5" />
        <SourcesPanel a={a} className="lg:col-span-7" />
      </div>
    </div>
  );
}

/** Unknown/invalid ticker: say why, then offer the closest symbols and recent ones. */
function TickerNotFound({ ticker, message }: { ticker: string; message: string }) {
  const search = useSearch(ticker.replace(/[^A-Za-z0-9 .-]/g, ""));
  const { setPaletteOpen } = useCommands();
  const matches = (search.data ?? []).filter((m) => m.symbol.toUpperCase() !== ticker).slice(0, 6);
  const recent = getRecent().filter((t) => t !== ticker).slice(0, 6);
  return (
    <div className="mx-auto max-w-xl pt-6">
      <Panel>
        <div className="pt-2 text-center">
          <p className="eyebrow">No analysis</p>
          <h1 className="mt-1.5 text-xl font-semibold text-ink">Couldn't analyze “{ticker}”</h1>
          <p className="mt-1.5 text-sm text-ink-2">{message}</p>
        </div>
        {matches.length > 0 && (
          <div className="mt-5">
            <p className="eyebrow mb-2">Did you mean</p>
            <ul className="grid gap-1.5 sm:grid-cols-2">
              {matches.map((m) => (
                <li key={m.symbol}>
                  <Link to={`/t/${encodeURIComponent(m.symbol)}`} className="flex items-center gap-2.5 rounded-lg bg-sunken px-2.5 py-2 hover:bg-raised">
                    <TickerLogo symbol={m.symbol} url={m.logo_url} size={24} />
                    <span className="min-w-0">
                      <span className="block font-mono text-xs font-semibold text-ink">{m.symbol}</span>
                      <span className="block truncate text-2xs text-muted">{m.name}</span>
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        )}
        {recent.length > 0 && (
          <div className="mt-4">
            <p className="eyebrow mb-2">Recent</p>
            <div className="flex flex-wrap gap-1.5">
              {recent.map((t) => (
                <Link key={t} to={`/t/${encodeURIComponent(t)}`} className="btn h-7 font-mono text-xs">
                  {t}
                </Link>
              ))}
            </div>
          </div>
        )}
        <div className="mt-5 flex justify-center">
          <button className="btn" onClick={() => setPaletteOpen(true)}>
            Search tickers <kbd className="kbd">/</kbd>
          </button>
        </div>
      </Panel>
    </div>
  );
}
