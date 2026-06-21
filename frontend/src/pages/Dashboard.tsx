import { useState } from "react";

import { useAnalysis, usePrice } from "../hooks/useAnalysis";
import { TickerSearch } from "../components/TickerSearch";
import { SentimentGauge } from "../components/SentimentGauge";
import { StatCard } from "../components/StatCard";
import { SourceBreakdown } from "../components/SourceBreakdown";
import { NewsFeed } from "../components/NewsFeed";
import { PriceChart } from "../components/PriceChart";
import { SentimentTimeline } from "../components/SentimentTimeline";
import { TrendingMentions } from "../components/TrendingMentions";
import { SourceStatus } from "../components/SourceStatus";
import { labelColor, labelText } from "../lib/sentiment";

export function Dashboard() {
  const [ticker, setTicker] = useState<string | null>(null);
  const [range, setRange] = useState("1M");

  const analysis = useAnalysis(ticker);
  const price = usePrice(ticker, range);
  const data = analysis.data;

  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      {/* Header */}
      <header className="mb-6 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-extrabold tracking-tight text-slate-100">
            <span className="text-accent">◆</span> Sentinel
          </h1>
          <p className="text-sm text-slate-500">
            Real-time market sentiment from free social &amp; news sources
          </p>
        </div>
        <div className="md:w-[440px]">
          <TickerSearch onSearch={setTicker} initial={ticker ?? ""} />
        </div>
      </header>

      {!ticker && <EmptyState />}

      {ticker && analysis.isLoading && <LoadingState />}

      {ticker && analysis.isError && (
        <div className="card border-bear/40 text-bear">
          Couldn’t analyze “{ticker}”: {(analysis.error as Error).message}
        </div>
      )}

      {ticker && data && (
        <div className="space-y-4">
          {/* Title row */}
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="text-xl font-bold text-slate-100">
              {data.company ?? data.ticker}
            </h2>
            <span className="font-mono text-sm text-slate-500">{data.ticker}</span>
            <span
              className="chip"
              style={{
                background: `${labelColor(data.overall_label)}22`,
                color: labelColor(data.overall_label),
              }}
            >
              {labelText(data.overall_label)}
            </span>
            {analysis.isFetching && (
              <span className="text-xs text-slate-500">refreshing…</span>
            )}
          </div>

          {/* Top grid: gauge + stats + price */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
            <div className="card lg:col-span-4">
              <div className="card-title">Overall Sentiment</div>
              <SentimentGauge score={data.overall_score} totalSignals={data.total_signals} />
            </div>

            <div className="grid grid-cols-2 gap-4 lg:col-span-4">
              <StatCard
                label="Price"
                value={
                  price.data?.current_price != null
                    ? `$${price.data.current_price.toFixed(2)}`
                    : "—"
                }
                sub={
                  price.data?.change_pct != null
                    ? `${price.data.change_pct >= 0 ? "+" : ""}${price.data.change_pct.toFixed(2)}% today`
                    : undefined
                }
                accent={
                  price.data?.change != null
                    ? price.data.change >= 0
                      ? "#22c55e"
                      : "#ef4444"
                    : undefined
                }
                trend={
                  price.data?.change != null
                    ? price.data.change >= 0
                      ? "up"
                      : "down"
                    : null
                }
              />
              <StatCard label="Signals" value={`${data.total_signals}`} sub="analyzed now" />
              <StatCard
                label="Active Sources"
                value={`${data.active_sources}`}
                sub={`of ${data.sources.length} connected`}
              />
              <StatCard
                label="Bull / Bear"
                value={`${data.bullish_pct}% / ${data.bearish_pct}%`}
                sub={`${data.neutral_pct}% neutral`}
                accent="#38bdf8"
              />
            </div>

            <div className="lg:col-span-4">
              <PriceChart data={price.data} loading={price.isLoading} range={range} onRange={setRange} />
            </div>
          </div>

          {/* Middle grid: breakdown + timeline + trending */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
            <div className="lg:col-span-5">
              <SourceBreakdown sources={data.sources} />
            </div>
            <div className="space-y-4 lg:col-span-7">
              <SentimentTimeline timeline={data.timeline} />
              <TrendingMentions trending={data.trending} />
            </div>
          </div>

          {/* Feed + status */}
          <NewsFeed signals={data.signals} />
          <SourceStatus
            sources={data.sources}
            generatedAt={data.generated_at}
            cached={data.cached}
          />
        </div>
      )}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="card flex flex-col items-center justify-center py-20 text-center">
      <div className="mb-3 text-4xl">📈</div>
      <h3 className="text-lg font-semibold text-slate-200">
        Search a ticker to see the real-time mood
      </h3>
      <p className="mt-1 max-w-md text-sm text-slate-500">
        Sentinel aggregates genuine signals from Reddit, StockTwits, Yahoo
        Finance, Google News, Hacker News and more — then scores the sentiment so
        you can read the room at a glance.
      </p>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="skeleton h-64 lg:col-span-4" />
        <div className="skeleton h-64 lg:col-span-4" />
        <div className="skeleton h-64 lg:col-span-4" />
      </div>
      <div className="skeleton h-80 w-full" />
    </div>
  );
}
