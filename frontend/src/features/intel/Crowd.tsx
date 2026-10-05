/** Retail positioning (StockTwits, Reddit, WSB, Bluesky) and the attention gauge. */
import { Flame, Users } from "lucide-react";
import type { ReactNode } from "react";

import type { Analysis, AttentionView, CrowdView } from "../../api/types";
import { Meter } from "../../components/charts/Bars";
import { Delta, ScoreChip } from "../../components/ui/Badges";
import { Empty } from "../../components/ui/Misc";
import { Panel, SubHead } from "../../components/ui/Panel";
import { Tip } from "../../components/ui/Tooltip";
import { cx } from "../../lib/cx";
import { compact, int, pct, plural, signed } from "../../lib/format";

/** StockTwits skews bullish structurally; this is the typical bull share. */
const ST_BASELINE = 0.62;

export function CrowdPanel({ a, className }: { a: Analysis; className?: string }) {
  const c = a.crowd;
  const att = a.attention;
  return (
    <Panel id="crowd" title="Crowd & attention" icon={<Users />} subtitle="Retail positioning and how loudly the market is talking" className={className}>
      <div className={cx("grid gap-5", (c || att) && "md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]")}>
        <div className="space-y-4">{c ? <Retail c={c} socialScore={a.social.n ? a.social.score : null} socialN={a.social.n} /> : <Empty title="No crowd data">Retail sources returned nothing for this ticker.</Empty>}</div>
        <div>
          {att ? (
            <Attention att={att} />
          ) : (
            <>
              <SubHead>Attention</SubHead>
              <p className="text-xs text-muted">No attention reading this run — it needs GDELT article volume, Wikipedia pageviews or Reddit mention history, and none came back usable.</p>
            </>
          )}
        </div>
      </div>
    </Panel>
  );
}

function Retail({ c, socialScore, socialN }: { c: CrowdView; socialScore: number | null; socialN: number }) {
  const tagged = (c.stocktwits_bullish ?? 0) + (c.stocktwits_bearish ?? 0);
  const ratio = c.stocktwits_bull_ratio;
  const rankDelta = c.reddit_rank != null && c.reddit_rank_prev != null ? c.reddit_rank_prev - c.reddit_rank : null;
  const mentionChg = c.reddit_mentions != null && c.reddit_mentions_prev ? (c.reddit_mentions / c.reddit_mentions_prev - 1) * 100 : null;
  return (
    <>
      <div>
        <SubHead right={c.stocktwits_watchers != null ? `${compact(c.stocktwits_watchers)} watchers` : undefined}>StockTwits</SubHead>
        {ratio != null && tagged > 0 ? (
          <>
            <div className="flex items-baseline gap-2">
              <span className="text-xl font-semibold leading-none text-ink">{Math.round(ratio * 100)}%</span>
              <span className="text-xs text-ink-2">bullish of {plural(tagged, "tagged post")}</span>
              {tagged < 15 && <span className="rounded bg-raised px-1 text-2xs text-ink-2">small sample</span>}
            </div>
            <div className="relative mt-2.5">
              <div className="flex h-2.5 gap-[2px]" role="img" aria-label={`${Math.round(ratio * 100)}% bullish, ${Math.round((1 - ratio) * 100)}% bearish`}>
                <div className="h-full rounded-l-[3px] bg-bull" style={{ width: `${ratio * 100}%` }} />
                <div className="h-full flex-1 rounded-r-[3px] bg-bear" />
              </div>
              <div className="absolute -top-1 -bottom-1 w-0.5 rounded-full bg-[rgb(var(--ink))]" style={{ left: `calc(${ST_BASELINE * 100}% - 1px)`, boxShadow: "0 0 0 2px rgb(var(--panel))" }} />
            </div>
            <div className="mt-1.5 flex justify-between text-2xs text-muted">
              <span>
                <span className="text-bull-ink">▲</span> {c.stocktwits_bullish} bull · <span className="text-bear-ink">▼</span> {c.stocktwits_bearish} bear
              </span>
              <span>│ typical {Math.round(ST_BASELINE * 100)}%</span>
            </div>
            <p className="mt-1 text-2xs text-muted">
              {ratio - ST_BASELINE >= 0.15
                ? "Well above the usual bullish skew — crowded long."
                : ratio - ST_BASELINE <= -0.15
                  ? "Well below the usual bullish skew — retail is souring."
                  : "Near StockTwits' structural bullish skew — no edge either way."}
            </p>
          </>
        ) : (
          <p className="text-xs text-muted">{c.stocktwits_messages ? `${c.stocktwits_messages} recent messages, none tagged bullish/bearish.` : "No StockTwits stream for this symbol."}</p>
        )}
      </div>

      <div className="grid grid-cols-2 gap-2">
        <Tile
          label="Reddit rank"
          value={c.reddit_rank != null ? `#${c.reddit_rank}` : "—"}
          sub={rankDelta != null ? <><Delta value={rankDelta} /> <span className="text-muted">vs #{c.reddit_rank_prev} yesterday</span></> : "not in ApeWisdom's list"}
        />
        <Tile
          label="Reddit mentions · 24h"
          value={c.reddit_mentions != null ? int(c.reddit_mentions) : "—"}
          sub={mentionChg != null ? <><span className={cx("font-medium", mentionChg > 0 ? "text-ink" : "text-ink-2")}>{pct(mentionChg, 0)}</span> <span className="text-muted">vs {int(c.reddit_mentions_prev)} prior</span></> : c.reddit_upvotes != null ? `${int(c.reddit_upvotes)} upvotes` : "—"}
        />
        <Tile
          label="WallStreetBets"
          value={c.wsb_sentiment != null ? <ScoreChip score={c.wsb_sentiment} label /> : "—"}
          sub={c.wsb_comments != null ? `${plural(c.wsb_comments, "comment")} in the top-50 feed` : "not in WSB's top 50"}
        />
        <Tile
          label="Social text tone"
          value={socialScore != null ? <ScoreChip score={socialScore} label /> : "—"}
          sub={socialN ? `${plural(socialN, "post")} scored${c.bluesky_posts != null ? ` · ${c.bluesky_posts} on Bluesky` : ""}` : "no posts kept"}
        />
      </div>
    </>
  );
}

function Tile({ label, value, sub }: { label: string; value: ReactNode; sub: ReactNode }) {
  return (
    <div className="rounded-md bg-sunken px-2.5 py-2">
      <div className="text-2xs text-muted">{label}</div>
      <div className="mt-1 text-base font-semibold leading-6 text-ink">{value}</div>
      <div className="truncate text-2xs text-ink-2">{sub}</div>
    </div>
  );
}

function Attention({ att }: { att: AttentionView }) {
  const rows: Array<{ label: string; value: string; note: string; z: number | null }> = [
    { label: "News volume", value: att.news_volume_z != null ? `${signed(att.news_volume_z, 1)}σ` : "—", note: "GDELT articles vs 30-day baseline", z: att.news_volume_z },
    { label: "Wikipedia views", value: att.wiki_views_7d != null ? `${compact(att.wiki_views_7d)}/day` : "—", note: att.wiki_views_z != null ? `${signed(att.wiki_views_z, 1)}σ vs 90 days` : "pageviews unavailable", z: att.wiki_views_z },
    { label: "Reddit mentions", value: att.reddit_change_pct != null ? pct(att.reddit_change_pct, 0) : "—", note: "vs the prior 24 hours", z: att.reddit_change_pct != null ? att.reddit_change_pct / 50 : null },
    { label: "Signals · 24h", value: int(att.signals_24h), note: "items timestamped in the last day", z: null },
  ];
  return (
    <div>
      <SubHead>Attention</SubHead>
      <div className="flex items-end gap-3">
        <Flame className="mb-0.5 size-5 text-heat" aria-hidden />
        <span className="text-[28px] font-semibold leading-none tracking-[-0.02em] text-ink">{att.heat}</span>
        <span className="pb-0.5 text-sm font-semibold text-ink-2">{att.label}</span>
      </div>
      <Meter value={att.heat / 100} color="rgb(var(--heat))" track="rgb(var(--heat) / 0.16)" height={8} className="mt-3" />
      <div className="mt-1 flex justify-between text-2xs text-muted">
        <span>Quiet</span>
        <span>Normal</span>
        <span>Elevated</span>
        <span>Spiking</span>
      </div>
      <ul className="mt-3 divide-hair">
        {rows.map((r) => (
          <li key={r.label} className="flex items-center justify-between gap-3 py-1.5 text-xs">
            <Tip content={r.note}>
              <span className="text-ink-2">{r.label}</span>
            </Tip>
            <span className="inline-flex items-center gap-1.5 font-semibold text-ink num">
              {r.z != null && r.z >= 2 && <span className="size-1.5 rounded-full bg-heat" title="unusually high (≥ 2σ)" />}
              {r.value}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
