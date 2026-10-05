/**
 * The verdict: score dial, one-line read, evidence-backed reasons, the six
 * score components, and what changed since the last look. Built to be read
 * in five seconds.
 */
import { ArrowUpRight, ClockArrowLeft } from "lucide-react";
import type { ReactNode } from "react";

import type { Analysis, Component, Reason } from "../../api/types";
import { useSnapshots } from "../../api/hooks";
import { Dial, VERDICT_DIAL } from "../../components/charts/Dial";
import { Sparkline } from "../../components/charts/Sparkline";
import { DivergingBar } from "../../components/charts/Bars";
import { Delta, Mark } from "../../components/ui/Badges";
import { CountUp } from "../../components/ui/Misc";
import { Tip } from "../../components/ui/Tooltip";
import { cx } from "../../lib/cx";
import { dayTime, pct, signed, timeAgo } from "../../lib/format";
import { polarityOf100, textTone, toneVar } from "../../lib/sentiment";

export function VerdictHero({ a }: { a: Analysis }) {
  const v = a.verdict;
  const p = polarityOf100(v.score);
  return (
    <section id="verdict" className="panel relative scroll-mt-36 md:scroll-mt-28 overflow-hidden shadow-hero" aria-label="SentiNET verdict">
      <div
        className="pointer-events-none absolute inset-0"
        style={{ background: `radial-gradient(520px 260px at 140px 120px, ${toneVar(p, 0.09)}, transparent 70%)` }}
        aria-hidden
      />
      <div className="relative grid gap-6 p-5 lg:grid-cols-[220px_minmax(0,1fr)] xl:grid-cols-[220px_minmax(0,1fr)_380px] xl:gap-8">
        <ScoreBlock a={a} />
        <ReadBlock a={a} />
        <ComponentsBlock components={v.components} className="lg:col-span-2 xl:col-span-1" />
      </div>
      <DeltaStrip a={a} />
    </section>
  );
}

function ScoreBlock({ a }: { a: Analysis }) {
  const v = a.verdict;
  const p = polarityOf100(v.score);
  const pips = v.confidence === "high" ? 3 : v.confidence === "medium" ? 2 : 1;
  return (
    <div className="flex flex-col items-center">
      <Dial value={v.score} bands={VERDICT_DIAL} size={208} thickness={11} ariaLabel={`SentiNET score ${v.score} of 100, ${v.label}`}>
        <span className="text-[56px] font-semibold leading-none tracking-[-0.04em] text-ink">
          <CountUp value={v.score} />
        </span>
        <span className={cx("mt-2 flex items-center gap-1.5 text-sm font-semibold", textTone[p])}>
          <Mark p={p} className="text-[10px]" />
          {v.label}
        </span>
      </Dial>
      <div className="-mt-1 flex flex-col items-center gap-1.5">
        <Tip content={`Confidence ${Math.round(v.confidence_value * 100)}% — data volume, source count, component agreement and freshness`}>
          <span className="inline-flex items-center gap-1.5 text-xs text-ink-2">
            <span className="flex gap-0.5" aria-hidden>
              {[1, 2, 3].map((i) => (
                <span key={i} className={cx("h-2.5 w-1 rounded-full", i <= pips ? "bg-[rgb(var(--ink-2))]" : "bg-[rgb(var(--grid))]")} />
              ))}
            </span>
            <span className="capitalize">{v.confidence}</span> confidence
          </span>
        </Tip>
        <ScoreHistory ticker={a.ticker} current={v.score} />
        {a.delta.sentinel_change != null && a.delta.previous_at && (
          <Tip content={`SentiNET ${a.verdict.score - a.delta.sentinel_change} → ${a.verdict.score} since ${dayTime(a.delta.previous_at)}`}>
            <span className="inline-flex items-center gap-1 whitespace-nowrap rounded-md bg-raised px-1.5 py-0.5 text-xs">
              <Delta value={a.delta.sentinel_change} />
              <span className="text-muted">since {dayTime(a.delta.previous_at)}</span>
            </span>
          </Tip>
        )}
      </div>
    </div>
  );
}

/** Stored SentiNET scores for this ticker (one per analysis), oldest → newest. */
function ScoreHistory({ ticker, current }: { ticker: string; current: number }) {
  const snaps = useSnapshots(ticker);
  const list = [...(snaps.data ?? [])].sort((x, y) => x.at.localeCompare(y.at));
  if (list.length < 2) return null;
  const values = list.map((s) => s.sentinel_score);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  return (
    <Tip content={`${list.length} stored looks since ${dayTime(list[0].at)} · range ${lo}–${hi} · now ${current}`}>
      <div className="flex w-[150px] items-center gap-2" tabIndex={0}>
        <Sparkline values={values} height={20} reference={50} color="rgb(var(--ink-2))" className="flex-1" />
        <span className="text-2xs text-muted">{list.length} looks</span>
      </div>
    </Tip>
  );
}

function ReadBlock({ a }: { a: Analysis }) {
  const v = a.verdict;
  return (
    <div className="min-w-0">
      <div className="flex items-center gap-2">
        <p className="eyebrow">The read</p>
        <span className="text-2xs text-faint">·</span>
        <p className="text-2xs text-muted">
          {a.sentiment.n} signals · {a.narratives.length} stories · {a.sources.filter((s) => s.status === "ok").length} sources
        </p>
      </div>
      <h2 className="mt-2 text-balance text-[21px] font-semibold leading-[29px] tracking-[-0.01em] text-ink sm:text-[23px] sm:leading-[31px]">{v.headline}</h2>
      {v.reasons.length > 0 ? (
        <ul className="mt-4 space-y-2.5">
          {v.reasons.slice(0, 5).map((r, i) => (
            <ReasonLine key={i} r={r} narratives={a.narratives.map((n) => n.id)} />
          ))}
        </ul>
      ) : (
        <p className="mt-4 text-sm text-muted">No reason cleared the evidence bar — the inputs are thin or balanced.</p>
      )}
    </div>
  );
}

function ReasonLine({ r, narratives }: { r: Reason; narratives: string[] }) {
  const target = r.ref && narratives.includes(r.ref) ? `narr-${r.ref}` : r.ref ? refTarget[r.ref] : undefined;
  const body: ReactNode = (
    <>
      <span className="mt-[5px] flex w-3 shrink-0 justify-center">
        <Mark p={r.polarity} className="text-[10px]" />
      </span>
      <span className="text-sm leading-[21px] text-ink-2 group-hover:text-ink">{r.text}</span>
      {target && <ArrowUpRight className="mt-[3px] size-3.5 shrink-0 text-faint opacity-0 transition-opacity group-hover:opacity-100" aria-hidden />}
    </>
  );
  return (
    <li>
      {target ? (
        <a
          href={`#${target}`}
          className="group flex gap-2.5"
          onClick={(e) => {
            const el = document.getElementById(target);
            if (!el) return;
            e.preventDefault();
            el.scrollIntoView({ behavior: "smooth", block: "center" });
            el.dispatchEvent(new CustomEvent("sentinet:focus"));
          }}
        >
          {body}
        </a>
      ) : (
        <div className="flex gap-2.5">{body}</div>
      )}
    </li>
  );
}

/** Where component-keyed reasons should jump to. */
const refTarget: Record<string, string> = {
  news: "narratives",
  social: "crowd",
  analysts: "smart-money",
  insiders: "smart-money",
  earnings: "smart-money",
  momentum: "price",
  technicals: "price",
};

function ComponentsBlock({ components, className }: { components: Component[]; className?: string }) {
  const totalW = components.filter((c) => c.available).reduce((s, c) => s + c.weight, 0) || 1;
  return (
    <div className={cx("min-w-0", className)}>
      <div className="mb-2.5 flex items-baseline justify-between">
        <p className="eyebrow">Score components</p>
        <p className="text-2xs text-muted">bear ← 50 → bull · weight</p>
      </div>
      {components.length === 0 && <p className="text-xs text-muted">No component scores were produced for this run.</p>}
      <ul className="space-y-2.5">
        {components.map((c) => {
          const p = polarityOf100(c.score);
          const effW = c.available ? c.weight / totalW : 0;
          return (
            <li key={c.key}>
              <Tip
                className="block"
                content={
                  <div className="max-w-[260px] space-y-1">
                    <div className="font-semibold text-ink">
                      {c.label} · {c.available && c.score != null ? Math.round(c.score) : "n/a"}
                    </div>
                    <div>{c.detail}</div>
                    <div className="text-muted">
                      Weight {Math.round(c.weight * 100)}% nominal{c.available ? `, ${Math.round(effW * 100)}% effective` : " — excluded"} · confidence {Math.round(c.confidence * 100)}%
                    </div>
                  </div>
                }
              >
                <div className="grid grid-cols-[84px_minmax(0,1fr)_30px_30px] items-center gap-2.5" tabIndex={0}>
                  <span className={cx("truncate text-xs font-medium", c.available ? "text-ink-2" : "text-muted")}>{c.label}</span>
                  {c.available ? <DivergingBar value={c.score} height={6} /> : <div className="h-1.5 rounded-full bg-[rgb(var(--grid))] opacity-60" />}
                  <span className={cx("text-right text-xs font-semibold num", c.available ? textTone[p] : "text-muted")}>{c.available && c.score != null ? Math.round(c.score) : "n/a"}</span>
                  <span className="text-right text-2xs text-muted num">{Math.round(c.weight * 100)}%</span>
                </div>
                <p className={cx("mt-0.5 line-clamp-2 pl-[94px] text-2xs", "text-muted")}>{c.detail}</p>
              </Tip>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function DeltaStrip({ a }: { a: Analysis }) {
  const d = a.delta;
  if (!d.previous_at) {
    return (
      <div className="relative flex items-center gap-2 px-5 py-2.5 text-xs text-muted hairline-t">
        <ClockArrowLeft className="size-3.5 shrink-0" aria-hidden />
        First look at {a.ticker} — what changed will show here on your next visit.
      </div>
    );
  }
  return (
    <div className="relative flex flex-wrap items-center gap-x-4 gap-y-1.5 px-5 py-2.5 text-xs hairline-t">
      <span className="inline-flex items-center gap-1.5 text-ink-2">
        <ClockArrowLeft className="size-3.5 shrink-0 text-muted" aria-hidden />
        <span className="font-medium">Since {dayTime(d.previous_at)}</span>
        <span className="text-muted">({timeAgo(d.previous_at)})</span>
      </span>
      {d.note && <span className="text-ink">{d.note}</span>}
      {d.sentinel_change != null && (
        <span className="inline-flex items-center gap-1 text-muted">
          SentiNET <Delta value={d.sentinel_change} />
        </span>
      )}
      {d.score_change != null && (
        <span className="inline-flex items-center gap-1 text-muted">
          tone <span className={cx("font-medium num", textTone[d.score_change > 0.02 ? "bull" : d.score_change < -0.02 ? "bear" : "neutral"])}>{signed(d.score_change)}</span>
        </span>
      )}
      {d.price_change_pct != null && (
        <span className="inline-flex items-center gap-1 text-muted">
          price <span className={cx("font-medium num", textTone[d.price_change_pct > 0 ? "bull" : d.price_change_pct < 0 ? "bear" : "neutral"])}>{pct(d.price_change_pct)}</span>
        </span>
      )}
      {d.new_narratives.length > 0 && (
        <span className="inline-flex min-w-0 items-center gap-1.5 text-muted">
          <span className="rounded bg-accent/15 px-1 py-px text-2xs font-semibold uppercase tracking-wider text-accent">New</span>
          <span className="truncate text-ink-2">“{d.new_narratives[0]}”</span>
          {d.new_narratives.length > 1 && <span className="shrink-0">+{d.new_narratives.length - 1} more</span>}
        </span>
      )}
    </div>
  );
}
