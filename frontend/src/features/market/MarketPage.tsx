/**
 * Market overview: regime call, Fear & Greed (stocks + crypto), index tape,
 * Reddit trending, market-wide narratives and the user's watchlist at a glance.
 */
import { ArrowRight, Flame, Newspaper, Plus, Star, TrendingUp } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useMarket, useWatchlist, useWatchToggle } from "../../api/hooks";
import type { FearGreed, IndexQuote, MarketOverview, Narrative, TrendingTicker, WatchItem } from "../../api/types";
import { DivergingBar } from "../../components/charts/Bars";
import { Dial, FEAR_GREED_DIAL } from "../../components/charts/Dial";
import { LineChart } from "../../components/charts/LineChart";
import { Sparkline } from "../../components/charts/Sparkline";
import { Delta, Mark, ScoreChip } from "../../components/ui/Badges";
import { CountUp, Empty, ErrorState, Segmented, Skeleton, TickerLogo } from "../../components/ui/Misc";
import { Panel, SubHead } from "../../components/ui/Panel";
import { cx } from "../../lib/cx";
import { dayTime, int, pct, plural, price, timeAgo } from "../../lib/format";
import { fearGreedBand, polarityOf, polarityOf100, textTone, toneVar } from "../../lib/sentiment";

export default function MarketPage() {
  const q = useMarket();
  useEffect(() => {
    document.title = "Market — SentiNET";
  }, []);
  if (q.isPending) return <MarketSkeleton />;
  if (q.error || !q.data) {
    // The watchlist is served independently — keep it useful when the overview fails.
    return (
      <div className="space-y-4">
        <ErrorState title="Market overview unavailable" message={q.error?.message} onRetry={() => void q.refetch()} />
        <WatchSummary />
      </div>
    );
  }
  const m = q.data;
  return (
    <div className={cx("space-y-4", q.isFetching && !q.isPending && "transition-opacity")}>
      <RegimeBanner m={m} />
      <IndexTape indices={m.indices} />
      <div className="grid gap-4 lg:grid-cols-12">
        <FearGreedPanel fg={m.fear_greed} className="lg:col-span-8" />
        <div className="space-y-4 lg:col-span-4">
          <CryptoPanel fg={m.crypto_fear_greed} />
          <HeadlinesPanel m={m} />
        </div>
      </div>
      <div className="grid gap-4 lg:grid-cols-12">
        <MarketNarratives narratives={m.narratives} className="lg:col-span-7" />
        <TrendingPanel trending={m.trending} className="lg:col-span-5" />
      </div>
      <WatchSummary />
      <p className="text-2xs text-faint">
        Generated {timeAgo(m.generated_at)} ·{" "}
        {Object.entries(m.status)
          .map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`)
          .join(" · ")}
      </p>
    </div>
  );
}

function MarketSkeleton() {
  return (
    <div className="space-y-4" aria-busy="true">
      <Skeleton className="h-28 w-full rounded-xl" />
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-8">
        {Array.from({ length: 8 }, (_, i) => (
          <Skeleton key={i} className="h-24 rounded-xl" />
        ))}
      </div>
      <div className="grid gap-4 lg:grid-cols-12">
        <Skeleton className="h-96 rounded-xl lg:col-span-8" />
        <Skeleton className="h-96 rounded-xl lg:col-span-4" />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------------- */

function RegimeBanner({ m }: { m: MarketOverview }) {
  const fg = m.fear_greed;
  const p = fg ? fearGreedBand(fg.score).polarity : "neutral";
  const vix = m.indices.find((i) => i.symbol === "^VIX");
  const tnx = m.indices.find((i) => i.symbol === "^TNX");
  const spy = m.indices.find((i) => i.symbol === "SPY");
  const spy1m = spy && spy.spark.length > 1 ? (spy.spark[spy.spark.length - 1] / spy.spark[0] - 1) * 100 : null;
  return (
    <section className="panel relative overflow-hidden shadow-hero">
      <div className="pointer-events-none absolute inset-0" style={{ background: `radial-gradient(600px 220px at 0% 0%, ${toneVar(p, 0.1)}, transparent 70%)` }} aria-hidden />
      <div className="relative flex flex-col gap-4 p-5 lg:flex-row lg:items-center lg:justify-between">
        <div className="min-w-0 max-w-3xl">
          <p className="eyebrow">Market regime</p>
          <h1 className="mt-1.5 flex items-center gap-2.5 text-[26px] font-semibold leading-8 tracking-[-0.02em] text-ink">
            <Mark p={p} className="text-[12px]" />
            {m.regime}
          </h1>
          <p className="mt-2 text-sm leading-[21px] text-ink-2">{m.regime_detail}</p>
        </div>
        <dl className="grid shrink-0 grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-4">
          <Fact label="Fear & Greed" value={fg ? <><CountUp value={fg.score} /> <span className="text-sm font-medium text-ink-2">{fearGreedBand(fg.score).label}</span></> : "—"} sub={fg?.month_ago != null ? <Delta value={fg.score - fg.month_ago} suffix=" vs 1m" /> : null} />
          <Fact label="VIX" value={vix?.price != null ? vix.price.toFixed(2) : "—"} sub={vix?.change_pct != null ? <Delta value={vix.change_pct} digits={1} suffix="%" invert /> : null} />
          <Fact label="10Y yield" value={tnx?.price != null ? `${tnx.price.toFixed(2)}%` : "—"} sub={tnx?.change_pct != null ? <Delta value={tnx.change_pct} digits={1} suffix="%" invert /> : null} />
          <Fact label="S&P 500 · 1M" value={spy1m != null ? pct(spy1m) : "—"} sub={spy?.change_pct != null ? <span className="text-muted">today {pct(spy.change_pct)}</span> : null} />
        </dl>
      </div>
    </section>
  );
}

function Fact({ label, value, sub }: { label: string; value: React.ReactNode; sub: React.ReactNode }) {
  return (
    <div>
      <dt className="text-2xs text-muted">{label}</dt>
      <dd className="mt-1 text-xl font-semibold leading-6 text-ink">{value}</dd>
      <dd className="mt-0.5 text-2xs">{sub}</dd>
    </div>
  );
}

/* ------------------------------------------------------------------------- */

function IndexTape({ indices }: { indices: IndexQuote[] }) {
  if (!indices.length) return null;
  return (
    <section aria-label="Indices" className="grid grid-cols-2 gap-3 sm:grid-cols-4 xl:grid-cols-8">
      {indices.map((i) => {
        const p = polarityOf(i.change_pct ?? 0, 0.005);
        const inverse = i.symbol === "^VIX" || i.symbol === "^TNX";
        const first = i.spark[0];
        const last = i.spark[i.spark.length - 1];
        return (
          <Link key={i.symbol} to={i.symbol.startsWith("^") ? "#" : `/t/${encodeURIComponent(i.symbol)}`} className="panel block p-3 transition-colors hover:bg-raised" onClick={(e) => i.symbol.startsWith("^") && e.preventDefault()}>
            <div className="flex items-baseline justify-between gap-2">
              <span className="truncate text-xs font-medium text-ink-2">{i.name}</span>
              <span className="font-mono text-2xs text-faint">{i.symbol.replace("^", "")}</span>
            </div>
            <div className="mt-1.5 flex items-baseline justify-between gap-2">
              <span className="text-base font-semibold text-ink">{i.symbol === "^TNX" ? `${i.price?.toFixed(2)}%` : i.symbol === "^VIX" ? i.price?.toFixed(2) : price(i.price)}</span>
              <span className={cx("text-xs font-medium", textTone[inverse && p !== "neutral" ? (p === "bull" ? "bear" : "bull") : p])}>{pct(i.change_pct, 2)}</span>
            </div>
            <Sparkline values={i.spark} height={26} className="mt-1.5" color="rgb(var(--ink-2))" reference={first} area ariaLabel={`${i.name}, one month${first != null && last != null ? `, ${pct((last / first - 1) * 100)}` : ""}`} />
            {first != null && last != null && <div className="mt-1 text-right text-2xs text-muted">1M {pct((last / first - 1) * 100)}</div>}
          </Link>
        );
      })}
    </section>
  );
}

/* ------------------------------------------------------------------------- */

function FearGreedPanel({ fg, className }: { fg: FearGreed | null; className?: string }) {
  const series = useMemo(() => (fg ? [{ key: "fg", label: "Fear & Greed", color: "rgb(var(--ink-2))", points: fg.history.map((h) => ({ x: Date.parse(h.t), y: h.v })) }] : []), [fg]);
  if (!fg) {
    return (
      <Panel title="CNN Fear & Greed" className={className}>
        <Empty title="Fear & Greed unavailable">CNN's index couldn't be fetched right now.</Empty>
      </Panel>
    );
  }
  const band = fearGreedBand(fg.score);
  const comps: Array<[string, number | null]> = [
    ["Previous close", fg.previous_close],
    ["1 week ago", fg.week_ago],
    ["1 month ago", fg.month_ago],
    ["1 year ago", fg.year_ago],
  ];
  return (
    <Panel title="CNN Fear & Greed" subtitle="Seven market indicators, 0 = extreme fear · 100 = extreme greed" className={className}>
      <div className="grid gap-6 md:grid-cols-[200px_minmax(0,1fr)]">
        <div>
          <Dial value={fg.score} bands={FEAR_GREED_DIAL} size={196} thickness={10} ariaLabel={`Fear and Greed ${Math.round(fg.score)}, ${band.label}`}>
            <span className="text-[48px] font-semibold leading-none tracking-[-0.04em] text-ink">
              <CountUp value={fg.score} />
            </span>
            <span className={cx("mt-1.5 flex items-center gap-1.5 text-sm font-semibold", textTone[band.polarity])}>
              <Mark p={band.polarity} className="text-[10px]" />
              {band.label}
            </span>
          </Dial>
          <dl className="mt-1 space-y-1">
            {comps.map(([l, v]) => (
              <div key={l} className="flex items-center justify-between text-xs">
                <dt className="text-muted">{l}</dt>
                <dd className="flex items-center gap-2">
                  <span className="font-medium text-ink-2 num">{v != null ? Math.round(v) : "—"}</span>
                  <span className="w-20 text-right text-2xs text-muted">{v != null ? fearGreedBand(v).label : ""}</span>
                </dd>
              </div>
            ))}
          </dl>
        </div>
        <div className="min-w-0">
          <SubHead right="fear ← 50 → greed">Components</SubHead>
          <ul className="space-y-2.5">
            {fg.components.map((c) => (
              <li key={c.key} className="grid grid-cols-[minmax(0,140px)_minmax(0,1fr)_112px] items-center gap-3 text-xs">
                <span className="truncate text-ink-2">{c.label}</span>
                <DivergingBar value={c.score} deadZone={5} />
                <span className="flex items-center justify-end gap-1.5">
                  <span className="font-semibold text-ink num">{c.score != null ? Math.round(c.score) : "—"}</span>
                  <span className="w-[78px] truncate text-right text-2xs text-muted">{c.rating ?? ""}</span>
                </span>
              </li>
            ))}
          </ul>
          {fg.history.length > 1 && (
            <div className="mt-5">
              <SubHead right={`${fg.history.length} sessions`}>One year</SubHead>
              <LineChart
                series={series}
                height={150}
                yDomain={[0, 100]}
                yTicks={[0, 25, 50, 75, 100]}
                baseline={50}
                zones={[
                  { from: 0, to: 25, label: "Extreme fear", color: toneVar("bear", 0.09) },
                  { from: 75, to: 100, label: "Extreme greed", color: toneVar("bull", 0.09) },
                ]}
                valueFormat={(v) => `${Math.round(v)} · ${fearGreedBand(v).label}`}
                xFormat={(ms) => new Date(ms).toLocaleDateString("en-US", { month: "short" })}
                ariaLabel="Fear and Greed index over the last year"
              />
            </div>
          )}
        </div>
      </div>
    </Panel>
  );
}

function CryptoPanel({ fg }: { fg: FearGreed | null }) {
  if (!fg) {
    return (
      <Panel title="Crypto Fear & Greed">
        <Empty title="Unavailable" />
      </Panel>
    );
  }
  const band = fearGreedBand(fg.score);
  const spark = fg.history.slice(-60).map((h) => h.v);
  return (
    <Panel title="Crypto Fear & Greed" subtitle="alternative.me · daily">
      <div className="flex items-end justify-between gap-4">
        <div>
          <div className="flex items-baseline gap-2">
            <span className="text-[32px] font-semibold leading-none tracking-[-0.03em] text-ink">{Math.round(fg.score)}</span>
            <span className={cx("text-sm font-semibold", textTone[band.polarity])}>
              <Mark p={band.polarity} /> {fg.rating || band.label}
            </span>
          </div>
          <div className="mt-2 flex gap-3 text-2xs text-muted">
            {fg.previous_close != null && <span>yday {Math.round(fg.previous_close)}</span>}
            {fg.week_ago != null && <span>1w {Math.round(fg.week_ago)}</span>}
            {fg.month_ago != null && <span>1m {Math.round(fg.month_ago)}</span>}
          </div>
        </div>
        <div className="w-36">
          <Sparkline values={spark} height={40} domain={[0, 100]} reference={50} color="rgb(var(--ink-2))" ariaLabel="Crypto Fear and Greed, 60 days" />
          <div className="mt-0.5 text-right text-2xs text-faint">60 days · line at 50</div>
        </div>
      </div>
    </Panel>
  );
}

function HeadlinesPanel({ m }: { m: MarketOverview }) {
  const h = m.headlines;
  const tot = Math.max(1, h.bullish + h.bearish + h.neutral);
  return (
    <Panel title="Headline tone" icon={<Newspaper />} subtitle="Market-wide news, scored by the Sentinel engine">
      <div className="flex items-center justify-between gap-3">
        <ScoreChip score={h.score} label />
        <span className="text-xs text-muted">{plural(h.n, "headline")}</span>
      </div>
      <div className="mt-3 flex h-2 gap-[2px]" role="img" aria-label={`${h.bullish} bullish, ${h.neutral} neutral, ${h.bearish} bearish`}>
        <div className="rounded-l-[3px] bg-bull" style={{ width: `${(h.bullish / tot) * 100}%` }} />
        <div className="bg-[rgb(var(--mid))]" style={{ width: `${(h.neutral / tot) * 100}%` }} />
        <div className="rounded-r-[3px] bg-bear" style={{ width: `${(h.bearish / tot) * 100}%` }} />
      </div>
      <div className="mt-1.5 flex justify-between text-2xs text-muted">
        <span>
          <span className="text-bull">▲</span> {h.bullish} bullish
        </span>
        <span>{h.neutral} neutral</span>
        <span>
          <span className="text-bear">▼</span> {h.bearish} bearish
        </span>
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------------- */

function MarketNarratives({ narratives, className }: { narratives: Narrative[]; className?: string }) {
  return (
    <Panel title="What's moving markets" icon={<TrendingUp />} subtitle="Market-wide stories clustered from CNBC, MarketWatch, Google & Bing News" className={className} flush>
      {narratives.length === 0 ? (
        <Empty title="No market narratives" className="hairline-t" />
      ) : (
        <ol className="divide-hair hairline-t">
          {narratives.slice(0, 8).map((n, i) => (
            <li key={n.id} className="flex gap-3 px-4 py-3">
              <span className="mt-0.5 w-5 shrink-0 font-mono text-xs text-faint">{String(i + 1).padStart(2, "0")}</span>
              <div className="min-w-0 flex-1">
                <div className="flex items-start gap-2">
                  <p className="min-w-0 flex-1 text-[14px] font-medium leading-5 text-ink">
                    {n.url ? (
                      <a href={n.url} target="_blank" rel="noreferrer" className="hover:underline">
                        {n.headline}
                      </a>
                    ) : (
                      n.headline
                    )}
                    {n.is_new && <span className="ml-2 inline-block translate-y-[-1px] rounded bg-accent/15 px-1 py-px align-middle text-2xs font-semibold uppercase tracking-wider text-accent">New</span>}
                  </p>
                  <ScoreChip score={n.score} className="mt-px shrink-0" />
                </div>
                <p className="mt-1 text-xs text-muted">
                  <span className="font-medium text-ink-2">{plural(n.count, "item")}</span> · {n.publishers.slice(0, 3).join(", ")}
                  {n.publishers.length > 3 ? ` +${n.publishers.length - 3}` : ""}
                  {n.last_seen ? ` · latest ${timeAgo(n.last_seen)}` : ""}
                </p>
              </div>
            </li>
          ))}
        </ol>
      )}
    </Panel>
  );
}

type TrendSort = "mentions" | "risers";

function TrendingPanel({ trending, className }: { trending: TrendingTicker[]; className?: string }) {
  const [sort, setSort] = useState<TrendSort>("mentions");
  const rows = useMemo(() => {
    const list = [...trending];
    if (sort === "risers") list.sort((a, b) => (b.change_pct ?? -Infinity) - (a.change_pct ?? -Infinity) || (b.mentions ?? 0) - (a.mentions ?? 0));
    else list.sort((a, b) => (a.rank ?? 1e9) - (b.rank ?? 1e9));
    return list.slice(0, 15);
  }, [trending, sort]);
  const maxM = Math.max(1, ...trending.map((t) => t.mentions ?? 0));
  return (
    <Panel
      title="Trending on Reddit"
      icon={<Flame />}
      subtitle="ApeWisdom mentions, last 24h · WSB tone where available"
      className={className}
      flush
      actions={
        <Segmented<TrendSort>
          size="xs"
          label="Sort trending"
          value={sort}
          onChange={setSort}
          options={[
            { value: "mentions", label: "Top" },
            { value: "risers", label: "Risers" },
          ]}
        />
      }
    >
      {rows.length === 0 ? (
        <Empty title="Nothing trending" className="hairline-t">ApeWisdom returned no rankings.</Empty>
      ) : (
        <table className="w-full text-xs">
          <thead>
            <tr className="text-2xs text-muted hairline-t hairline-b">
              <th className="py-1.5 pl-4 text-left font-normal">#</th>
              <th className="py-1.5 text-left font-normal">Ticker</th>
              <th className="py-1.5 text-left font-normal">Mentions</th>
              <th className="py-1.5 text-right font-normal">24h</th>
              <th className="py-1.5 pr-4 text-right font-normal">WSB</th>
            </tr>
          </thead>
          <tbody className="divide-hair">
            {rows.map((t) => {
              const rankChg = t.rank != null && t.rank_prev != null ? t.rank_prev - t.rank : null;
              return (
                <tr key={`${t.source}-${t.symbol}`} className="group hover:bg-raised/60">
                  <td className="py-1.5 pl-4 text-muted num">
                    {t.rank ?? "—"}
                    {rankChg != null && rankChg !== 0 && <Delta value={rankChg} className="ml-1 text-2xs" />}
                  </td>
                  <td className="max-w-[150px] py-1.5">
                    <Link to={`/t/${encodeURIComponent(t.symbol)}`} className="flex min-w-0 items-baseline gap-1.5">
                      <span className="font-mono font-semibold text-ink group-hover:underline">{t.symbol}</span>
                      <span className="truncate text-2xs text-muted">{t.name}</span>
                    </Link>
                  </td>
                  <td className="w-[28%] py-1.5">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 flex-1 rounded-full bg-[rgb(var(--grid))]">
                        <div className="h-full rounded-full bg-[rgb(var(--ink-2))]" style={{ width: `${((t.mentions ?? 0) / maxM) * 100}%` }} />
                      </div>
                      <span className="w-6 text-right text-ink-2 num">{int(t.mentions)}</span>
                    </div>
                  </td>
                  <td className={cx("py-1.5 text-right font-medium num", t.change_pct == null ? "text-faint" : t.change_pct > 0 ? "text-ink" : "text-muted")}>{t.change_pct != null ? pct(t.change_pct, 0) : "new"}</td>
                  <td className="py-1.5 pr-4 text-right">{t.sentiment != null ? <ScoreChip score={t.sentiment} /> : <span className="text-faint">—</span>}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

/* ------------------------------------------------------------------------- */

function WatchSummary() {
  const wl = useWatchlist();
  const add = useWatchToggle();
  const navigate = useNavigate();
  const [t, setT] = useState("");
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const sym = t.trim().toUpperCase().replace(/^\$/, "");
    if (!sym) return;
    add.mutate({ ticker: sym, watched: false });
    setT("");
  };
  return (
    <Panel
      title="Your watchlist"
      icon={<Star />}
      subtitle="Latest stored SentiNET scores · refreshed by the background monitor"
      flush
      actions={
        <>
          <form onSubmit={submit} className="flex items-center gap-1.5">
            <input value={t} onChange={(e) => setT(e.target.value)} placeholder="Add ticker" className="field h-7 w-28 text-xs uppercase placeholder:normal-case" aria-label="Add ticker to watchlist" />
            <button className="btn h-7 px-2" type="submit" aria-label="Add" disabled={add.isPending}>
              <Plus className="size-3.5" />
            </button>
          </form>
          <Link to="/watchlist" className="btn btn-ghost hidden h-7 text-xs sm:inline-flex">
            Manage <ArrowRight className="size-3" />
          </Link>
        </>
      }
    >
      {wl.isPending ? (
        <div className="space-y-2 p-4">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      ) : wl.error ? (
        <Empty title="Watchlist unavailable" className="hairline-t">{wl.error.message}</Empty>
      ) : !wl.data?.length ? (
        <Empty title="Nothing watched yet" className="hairline-t">Add a ticker above, or press w on any ticker page.</Empty>
      ) : (
        <ul className="grid divide-y divide-[var(--hairline)] hairline-t sm:grid-cols-2 sm:divide-y-0 xl:grid-cols-5">
          {wl.data.map((w) => (
            <WatchTile key={w.ticker} w={w} onOpen={() => navigate(`/t/${encodeURIComponent(w.ticker)}`)} />
          ))}
        </ul>
      )}
    </Panel>
  );
}

function WatchTile({ w, onOpen }: { w: WatchItem; onOpen: () => void }) {
  const score = w.last?.sentinel_score ?? null;
  const prev = w.previous?.sentinel_score ?? null;
  const p = polarityOf100(score);
  return (
    <li className="sm:[&:not(:nth-child(2n+1))]:border-l xl:[&:not(:nth-child(5n+1))]:border-l border-[var(--hairline)]">
      <button onClick={onOpen} className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-raised/60">
        <TickerLogo symbol={w.ticker} size={28} url={`https://logos.stocktwits-cdn.com/${w.ticker}.png`} />
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-sm font-semibold text-ink">{w.ticker}</span>
            {score != null && <span className={cx("text-sm font-semibold", textTone[p])}>{score}</span>}
            {score != null && prev != null && <Delta value={score - prev} className="text-2xs" />}
          </div>
          <div className="truncate text-2xs text-muted">{w.last ? `${w.name ?? ""} · ${dayTime(w.last.at)}` : "not analyzed yet"}</div>
        </div>
        <div className="w-16">{w.spark.length > 1 && <Sparkline values={w.spark} height={24} domain={[0, 100]} reference={50} color={toneVar(p)} />}</div>
      </button>
    </li>
  );
}

