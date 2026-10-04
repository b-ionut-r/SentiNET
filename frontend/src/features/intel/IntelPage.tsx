/**
 * Intel page: one ticker, fully synthesized. Hierarchy is deliberate —
 * verdict → insights → narratives → the case → price × tone → smart money →
 * crowd → themes → raw signals → filings & source health.
 */
import { RefreshCw, Star } from "lucide-react";
import { lazy, Suspense, useEffect, useMemo } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { ApiError } from "../../api/client";
import { useAnalysis, useWatchlist, useWatchToggle } from "../../api/hooks";
import type { Analysis } from "../../api/types";
import { usePageCommands } from "../../components/layout/commands";
import { ErrorState, Skeleton, TopProgress } from "../../components/ui/Misc";
import { cx } from "../../lib/cx";
import { useHotkeys } from "../../lib/hotkeys";
import { pushRecent } from "../../lib/storage";
import { AnalystsPanel, EarningsPanel, InsidersPanel } from "./SmartMoney";
import { CaseAndCatalysts } from "./Case";
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
  const ticker = decodeURIComponent(raw).toUpperCase();
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
      return (
        <ErrorState
          title={notFound ? `Couldn't analyze “${ticker}”` : "Analysis failed"}
          message={e.message}
          onRetry={notFound ? undefined : () => void q.refetch()}
        />
      );
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
    <div className={cx("space-y-4 transition-opacity", refetching && "is-refetching")}>
      <HeaderStrip a={a} watched={watched} onWatch={onWatch} onRefresh={onRefresh} refreshing={refetching} />
      <SectionNav a={a} />
      <VerdictHero a={a} />

      <div className="grid gap-4 lg:grid-cols-12">
        <InsightsRail insights={a.insights} className="lg:col-span-4 lg:col-start-9 lg:row-start-1" />
        <Narratives a={a} membersOf={narrativeSignals} className="lg:col-span-8 lg:col-start-1 lg:row-start-1" />
      </div>

      <CaseAndCatalysts a={a} />

      <Suspense fallback={<Skeleton className="h-[460px] w-full rounded-xl" />}>
        <PriceTone a={a} />
      </Suspense>

      <section id="smart-money" className="grid scroll-mt-28 gap-4 lg:grid-cols-3">
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
