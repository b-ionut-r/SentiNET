/** Themes (share of coverage × tone) and sentiment-tinted keywords. */
import { Hash } from "lucide-react";

import type { Analysis } from "../../api/types";
import { Empty } from "../../components/ui/Misc";
import { Panel, SubHead } from "../../components/ui/Panel";
import { Tip } from "../../components/ui/Tooltip";
import { cx } from "../../lib/cx";
import { plural, signed } from "../../lib/format";
import { divergingFill, glyph, polarityOf, textTone, tintFor } from "../../lib/sentiment";

export function ThemesPanel({ a, className }: { a: Analysis; className?: string }) {
  const themes = a.themes.slice(0, 9);
  const maxShare = Math.max(0.01, ...themes.map((t) => t.share));
  return (
    <Panel id="themes" title="Themes & keywords" icon={<Hash />} subtitle="What the coverage is about, and how it feels" className={className}>
      {themes.length === 0 ? (
        <Empty title="No themes detected" />
      ) : (
        <ul className="space-y-2">
          {themes.map((t) => {
            const p = polarityOf(t.score);
            return (
              <li key={t.theme}>
                <Tip
                  className="block"
                  content={
                    <span>
                      <b className="text-ink">{t.label}</b> · {plural(t.count, "item")} · {Math.round(t.share * 100)}% of coverage · tone {signed(t.score)}
                    </span>
                  }
                >
                  <div className="grid grid-cols-[112px_minmax(0,1fr)_52px] items-center gap-2.5 text-xs" tabIndex={0}>
                    <span className="truncate text-ink-2">{t.label}</span>
                    <div className="flex items-center gap-2">
                      <div
                        className="h-2 rounded-r-[3px]"
                        style={{ width: `${Math.max(3, (t.share / maxShare) * 100)}%`, background: divergingFill(Math.max(-1, Math.min(1, t.score / 0.4))) }}
                      />
                      <span className="shrink-0 text-2xs text-muted num">{Math.round(t.share * 100)}%</span>
                    </div>
                    <span className={cx("text-right font-semibold num", textTone[p])}>
                      <span className="mr-0.5 text-[8px]">{glyph(p)}</span>
                      {signed(t.score)}
                    </span>
                  </div>
                </Tip>
              </li>
            );
          })}
        </ul>
      )}
      <div className="mt-2 flex justify-between text-2xs text-muted">
        <span>bar = share of coverage · color = tone</span>
        <span>
          <span className="text-bear-ink">▼</span> bear · <span className="text-bull-ink">▲</span> bull
        </span>
      </div>

      <div className="mt-4 pt-3.5 hairline-t">
        <SubHead right="tint = mean tone of items using it">Keywords</SubHead>
        {a.keywords.length === 0 ? (
          <p className="text-xs text-muted">No recurring phrases.</p>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {a.keywords.slice(0, 18).map((k) => {
              const p = polarityOf(k.score);
              return (
                <Tip key={k.term} content={`“${k.term}” · ${plural(k.count, "mention")} · tone ${signed(k.score)}`}>
                  <span className="inline-flex h-6 items-center gap-1.5 rounded-md px-2 text-xs text-ink" style={{ background: tintFor(k.score, 1.2) }}>
                    <span className={cx("text-[8px]", textTone[p])}>{glyph(p)}</span>
                    {k.term}
                    <span className="text-2xs text-muted num">{k.count}</span>
                  </span>
                </Tip>
              );
            })}
          </div>
        )}
      </div>
    </Panel>
  );
}
