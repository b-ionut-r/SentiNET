/**
 * Sticky in-page navigation with scrollspy. Once the verdict scrolls away it
 * also carries a compact identity (ticker, price, score) so context is never lost.
 */
import { useEffect, useRef, useState } from "react";

import type { Analysis } from "../../api/types";
import { TickerLogo } from "../../components/ui/Misc";
import { cx } from "../../lib/cx";
import { pct, price } from "../../lib/format";
import { glyph, polarityOf, polarityOf100, textTone } from "../../lib/sentiment";

const SECTIONS = [
  { id: "verdict", label: "Verdict" },
  { id: "insights", label: "Insights" },
  { id: "narratives", label: "Narratives" },
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
      { rootMargin: "-140px 0px -55% 0px" },
    );
    els.forEach((e) => io.observe(e));
    const hero = document.getElementById("verdict");
    const io2 = new IntersectionObserver(([e]) => setCompact(!e.isIntersecting && e.boundingClientRect.top < 0), { rootMargin: "-100px 0px 0px 0px" });
    if (hero) io2.observe(hero);
    return () => {
      io.disconnect();
      io2.disconnect();
    };
  }, [a.ticker, a.generated_at]); // a re-run can add or drop sections

  // Keep the active tab visible in the horizontally scrolling strip (phones).
  const strip = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const box = strip.current;
    const link = box?.querySelector<HTMLElement>(`[data-section="${active}"]`);
    if (!box || !link) return;
    const left = link.offsetLeft; // the strip is the offsetParent (relative)
    if (left < box.scrollLeft || left + link.offsetWidth > box.scrollLeft + box.clientWidth) {
      box.scrollTo({ left: Math.max(0, left - 24), behavior: "smooth" });
    }
  }, [active]);

  const chg = polarityOf(a.quote?.change_pct ?? 0, 0.005);
  const sp = polarityOf100(a.verdict.score);

  return (
    <nav className="sticky top-[88px] z-30 -mx-4 bg-page/85 px-4 backdrop-blur-md hairline-b sm:-mx-6 sm:px-6 md:top-[52px]" aria-label="Sections">
      <div className="flex h-10 items-center gap-3 sm:gap-4">
        <div className={cx("flex shrink-0 items-center gap-2 overflow-hidden transition-all duration-200", compact ? "max-w-[96px] opacity-100 sm:max-w-[420px]" : "max-w-0 opacity-0")}>
          <TickerLogo symbol={a.ticker} url={a.profile?.logo_url} size={20} className="rounded-md" />
          <span className="font-mono text-xs font-semibold text-ink">{a.ticker}</span>
          <span className="hidden text-xs text-ink-2 num sm:inline">{price(a.quote?.price, a.quote?.currency)}</span>
          {a.quote?.change_pct != null && (
            <span className={cx("hidden text-xs font-medium num sm:inline", textTone[chg])}>
              <span className="text-[8px]">{glyph(chg)}</span> {pct(a.quote.change_pct, 2)}
            </span>
          )}
          <span className="hidden h-4 w-px bg-[var(--hairline-strong)] sm:block" />
          <span className={cx("text-xs font-semibold", textTone[sp])}>
            {a.verdict.score} <span className="hidden font-medium sm:inline">{a.verdict.label}</span>
          </span>
        </div>
        <div ref={strip} className="no-scrollbar relative flex min-w-0 flex-1 items-center gap-0.5 overflow-x-auto">
          {SECTIONS.map((s) => (
            <a
              key={s.id}
              data-section={s.id}
              aria-current={active === s.id ? "location" : undefined}
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
