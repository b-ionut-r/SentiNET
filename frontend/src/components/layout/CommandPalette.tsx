/**
 * ⌘K / "/" command palette: debounced symbol search, recent tickers, global
 * navigation and the current page's actions — fully keyboard driven.
 */
import { ArrowRight, ChartColumn, Beaker, Clock, Columns3, CornerDownLeft, Keyboard, Moon, Search, Star } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";

import { useSearch } from "../../api/hooks";
import { cx } from "../../lib/cx";
import { getRecent } from "../../lib/storage";
import { useTheme } from "../../lib/theme";
import { Kbd } from "../ui/Badges";
import { TickerLogo } from "../ui/Misc";
import { useCommands } from "./commands";

interface Item {
  id: string;
  group: string;
  title: ReactNode;
  subtitle?: ReactNode;
  icon: ReactNode;
  hint?: string;
  run: () => void;
}

const TICKER_RE = /^\$?[A-Za-z0-9^][A-Za-z0-9.\-=^]{0,14}$/;

function useDebounced<T>(value: T, delay: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setV(value), delay);
    return () => window.clearTimeout(id);
  }, [value, delay]);
  return v;
}

export function CommandPalette() {
  const { paletteOpen, setPaletteOpen, commands, setHelpOpen } = useCommands();
  if (!paletteOpen) return null;
  return <PaletteDialog onClose={() => setPaletteOpen(false)} pageCommands={commands} openHelp={() => setHelpOpen(true)} />;
}

function PaletteDialog({ onClose, pageCommands, openHelp }: { onClose: () => void; pageCommands: ReturnType<typeof useCommands>["commands"]; openHelp: () => void }) {
  const navigate = useNavigate();
  const { toggle } = useTheme();
  const [q, setQ] = useState("");
  const [active, setActive] = useState(0);
  const debounced = useDebounced(q, 180);
  const search = useSearch(debounced);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => inputRef.current?.focus(), []);

  const go = (path: string) => {
    onClose();
    navigate(path);
  };

  // Rebuilt every render: cheap, and always sees the latest results and handlers.
  const items: Item[] = (() => {
    const term = q.trim();
    const lower = term.toLowerCase();
    const out: Item[] = [];
    const sym = term.replace(/^\$/, "").toUpperCase();

    // "Analyze X" leads when X is an exact symbol match or reads like a ticker;
    // for a company-name query ("apple") the symbol matches lead instead.
    const exact = search.data?.find((m) => m.symbol.toUpperCase() === sym);
    const analyze: Item | null =
      term && TICKER_RE.test(term)
        ? {
            id: `go-${sym}`,
            group: "Analyze",
            title: (
              <>
                Analyze <span className="font-semibold text-ink">{sym}</span>
              </>
            ),
            subtitle: exact ? `${exact.name}${exact.exchange ? ` · ${exact.exchange}` : ""}` : "Fuse news, crowd, analysts, insiders and filings",
            icon: exact ? <TickerLogo symbol={exact.symbol} url={exact.logo_url} size={20} /> : <ArrowRight className="size-4" />,
            run: () => go(`/t/${encodeURIComponent(sym)}`),
          }
        : null;
    const tickerLike = !!exact || /^\$/.test(term) || /[0-9.\-=^]/.test(term) || sym.length <= 4;
    const symbols: Item[] = term
      ? (search.data ?? [])
          .filter((m) => m !== exact)
          .slice(0, 8)
          .map((m) => ({
            id: `sym-${m.symbol}`,
            group: "Symbols",
            title: (
              <>
                <span className="font-semibold text-ink">{m.symbol}</span>
                <span className="ml-2 text-ink-2">{m.name}</span>
              </>
            ),
            subtitle: [m.exchange, m.type].filter(Boolean).join(" · ") || undefined,
            icon: <TickerLogo symbol={m.symbol} url={m.logo_url} size={20} />,
            run: () => go(`/t/${encodeURIComponent(m.symbol)}`),
          }))
      : [];
    if (analyze && tickerLike) out.push(analyze, ...symbols);
    else out.push(...symbols, ...(analyze ? [analyze] : []));

    if (!term) {
      for (const t of getRecent()) {
        out.push({ id: `recent-${t}`, group: "Recent", title: <span className="font-semibold text-ink">{t}</span>, icon: <Clock className="size-4" />, run: () => go(`/t/${encodeURIComponent(t)}`) });
      }
    }

    const actions: Item[] = [
      ...pageCommands.map((c) => ({ id: `page-${c.id}`, group: "This page", title: c.title, icon: c.icon ?? <ArrowRight className="size-4" />, hint: c.hint, run: () => (onClose(), c.run()) })),
      { id: "nav-market", group: "Go to", title: "Market overview", icon: <ChartColumn className="size-4" />, hint: "g m", run: () => go("/") },
      { id: "nav-watch", group: "Go to", title: "Watchlist & alerts", icon: <Star className="size-4" />, hint: "g w", run: () => go("/watchlist") },
      { id: "nav-compare", group: "Go to", title: "Compare tickers", icon: <Columns3 className="size-4" />, hint: "g c", run: () => go("/compare") },
      { id: "nav-lab", group: "Go to", title: "Sentiment lab", icon: <Beaker className="size-4" />, hint: "g l", run: () => go("/lab") },
      { id: "theme", group: "Settings", title: "Toggle light / dark theme", icon: <Moon className="size-4" />, hint: "t", run: () => (onClose(), toggle()) },
      { id: "help", group: "Settings", title: "Keyboard shortcuts", icon: <Keyboard className="size-4" />, hint: "?", run: () => (onClose(), openHelp()) },
    ];
    const textOf = (i: Item) => (typeof i.title === "string" ? i.title : "") + " " + i.group;
    out.push(...(lower ? actions.filter((a) => textOf(a).toLowerCase().includes(lower)) : actions));
    return out;
  })();

  useEffect(() => setActive(0), [q]);
  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>(`[data-idx="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [active]);

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => Math.min(items.length - 1, a + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(0, a - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      items[active]?.run();
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  let lastGroup = "";
  return (
    <div className="fixed inset-0 z-[90] flex items-start justify-center bg-black/50 px-4 pt-[12vh] backdrop-blur-[2px] animate-fade-in" onMouseDown={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        className="w-full max-w-[620px] overflow-hidden rounded-xl bg-panel shadow-pop"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2.5 px-4 hairline-b">
          <Search className="size-4 shrink-0 text-muted" aria-hidden />
          <input
            ref={inputRef}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={onKey}
            placeholder="Ticker, company, or command…"
            className="h-12 w-full bg-transparent text-base text-ink placeholder:text-muted outline-none"
            aria-label="Search tickers and commands"
            role="combobox"
            aria-expanded="true"
            aria-controls="palette-list"
            aria-activedescendant={items[active] ? `pal-${items[active].id}` : undefined}
            spellCheck={false}
            autoComplete="off"
          />
          {search.isFetching && <span className="size-1.5 shrink-0 animate-pulse-soft rounded-full bg-accent" aria-label="Searching" />}
          <Kbd>esc</Kbd>
        </div>
        <div ref={listRef} id="palette-list" role="listbox" className="scroll-thin max-h-[min(440px,60vh)] overflow-y-auto p-1.5">
          {items.length === 0 && <p className="px-3 py-8 text-center text-sm text-muted">No matches. Type a ticker like NVDA or BTC-USD.</p>}
          {items.map((it, i) => {
            const header = it.group !== lastGroup ? it.group : null;
            lastGroup = it.group;
            return (
              <div key={it.id}>
                {header && <div className="eyebrow px-2.5 pb-1 pt-2.5">{header}</div>}
                <button
                  id={`pal-${it.id}`}
                  data-idx={i}
                  role="option"
                  aria-selected={i === active}
                  onMouseMove={() => setActive(i)}
                  onClick={() => it.run()}
                  className={cx("flex w-full items-center gap-3 rounded-lg px-2.5 py-2 text-left text-sm", i === active ? "bg-raised text-ink" : "text-ink-2")}
                >
                  <span className="flex size-5 shrink-0 items-center justify-center text-muted">{it.icon}</span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate">{it.title}</span>
                    {it.subtitle && <span className="block truncate text-xs text-muted">{it.subtitle}</span>}
                  </span>
                  {it.hint && <span className="flex gap-1">{it.hint.split(" ").map((k) => <Kbd key={k}>{k}</Kbd>)}</span>}
                  {i === active && !it.hint && <CornerDownLeft className="size-3.5 text-muted" aria-hidden />}
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
