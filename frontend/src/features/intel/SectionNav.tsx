/**
 * Sticky in-page navigation with scrollspy. Once the verdict scrolls away it
 * also carries a compact identity (ticker, price, score) so context is never lost.
 */
import { useEffect, useState } from "react";

import type { Analysis } from "../../api/types";
import { TickerLogo } from "../../components/ui/Misc";
import { cx } from "../../lib/cx";
import { pct, price } from "../../lib/format";
import { glyph, polarityOf, polarityOf100, textTone } from "../../lib/sentiment";

const SECTIONS = [
  { id: "verdict", label: "Verdict" },
  { id: "narratives", label: "Narratives" },
  { id: "insights", label: "Insights" },
  { id: "case", label: "Bull vs bear" },
  { id: "price", label: "Price × tone" },
  { id: "smart-money", label: "Smart money" },
  { id: "crowd", label: "Crowd" },
  { id: "themes", label: "Themes" },
  { id: "signals", label: "Signals" },
  { id: "sources", label: "Sources" },
];

export function SectionNav({ a }: { a: Analysis }) {
  const [active, setActive] = useState("verdict");
  const [compact, setCompact] = useState(false);

  useEffect(() => {
    const els = SECTIONS.map((s) => document.getElementById(s.id)).filter((e): e is HTMLElement => !!e);
    const io = new IntersectionObserver(
      (entries) => {
        const vis = entries.filter((e) => e.isIntersecting).sort((x, y) => x.boundingClientRect.top - y.boundingClientRect.top);
        if (vis[0]) setActive(vis[0].target.id);
      },
      { rootMargin: "-110px 0px -55% 0px" },
    );
    els.forEach((e) => io.observe(e));
    const hero = document.getElementById("verdict");
    const io2 = new IntersectionObserver(([e]) => setCompact(!e.isIntersecting && e.boundingClientRect.top < 0), { rootMargin: "-100px 0px 0px 0px" });
    if (hero) io2.observe(hero);
    return () => {
      io.disconnect();
      io2.disconnect();
    };
  }, [a.ticker]);

  const chg = polarityOf(a.quote?.change_pct ?? 0, 0.005);
  const sp = polarityOf100(a.verdict.score);

  return (
    <nav className="sticky top-[52px] z-30 -mx-6 hidden bg-page/85 px-6 backdrop-blur-md md:block hairline-b" aria-label="Sections">
      <div className="flex h-10 items-center gap-4">
        <div className={cx("flex shrink-0 items-center gap-2 overflow-hidden transition-all duration-200", compact ? "max-w-[420px] opacity-100" : "max-w-0 opacity-0")}>
          <TickerLogo symbol={a.ticker} url={a.profile?.logo_url} size={20} className="rounded-md" />
          <span className="font-mono text-xs font-semibold text-ink">{a.ticker}</span>
          <span className="text-xs text-ink-2 num">{price(a.quote?.price, a.quote?.currency)}</span>
          {a.quote?.change_pct != null && (
            <span className={cx("text-xs font-medium num", textTone[chg])}>
              <span className="text-[8px]">{glyph(chg)}</span> {pct(a.quote.change_pct, 2)}
            </span>
          )}
          <span className="h-4 w-px bg-[var(--hairline-strong)]" />
          <span className={cx("text-xs font-semibold", textTone[sp])}>
            {a.verdict.score} <span className="font-medium">{a.verdict.label}</span>
          </span>
        </div>
        <div className="no-scrollbar flex min-w-0 flex-1 items-center gap-0.5 overflow-x-auto">
          {SECTIONS.map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              onClick={(e) => {
                e.preventDefault();
                document.getElementById(s.id)?.scrollIntoView({ behavior: "smooth", block: "start" });
              }}
              className={cx(
                "shrink-0 rounded-md px-2 py-1 text-xs font-medium transition-colors",
                active === s.id ? "bg-raised text-ink" : "text-muted hover:text-ink-2",
              )}
            >
              {s.label}
            </a>
          ))}
        </div>
      </div>
    </nav>
  );
}
