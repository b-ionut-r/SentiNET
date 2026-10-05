/** Global chrome: sticky top bar, nav, palette, help overlay, shortcuts. */
import { Keyboard, Moon, Search, Sun, X } from "lucide-react";
import { useRef, type ReactNode } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";

import { cx } from "../../lib/cx";
import { modKey, useHotkeys } from "../../lib/hotkeys";
import { useTheme } from "../../lib/theme";
import { useRestoreFocus } from "../../lib/useRestoreFocus";
import { Kbd } from "../ui/Badges";
import { CommandPalette } from "./CommandPalette";
import { useCommands } from "./commands";
import { Wordmark } from "./Logo";

const NAV = [
  { to: "/", label: "Market", end: true },
  { to: "/watchlist", label: "Watchlist" },
  { to: "/compare", label: "Compare" },
  { to: "/lab", label: "Lab" },
];

export function AppShell({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const { theme, toggle } = useTheme();
  const { setPaletteOpen, paletteOpen, helpOpen, setHelpOpen } = useCommands();

  useHotkeys({
    "mod+k": () => setPaletteOpen(!paletteOpen),
    "/": () => setPaletteOpen(true),
    "?": () => setHelpOpen(!helpOpen),
    t: toggle,
    "g m": () => navigate("/"),
    "g w": () => navigate("/watchlist"),
    "g l": () => navigate("/lab"),
    "g c": () => navigate("/compare"),
    Escape: () => {
      setPaletteOpen(false);
      setHelpOpen(false);
    },
  });

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 bg-page/85 backdrop-blur-md hairline-b">
        <div className="mx-auto flex h-[52px] max-w-[1440px] items-center gap-3 px-4 sm:gap-5 sm:px-6">
          <Link to="/" className="shrink-0 rounded-md" aria-label="SentiNET home">
            <Wordmark />
          </Link>
          <nav className="hidden items-center gap-0.5 md:flex" aria-label="Primary">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end={n.end}
                className={({ isActive }) =>
                  cx("rounded-md px-2.5 py-1.5 text-sm font-medium transition-colors", isActive ? "bg-raised text-ink" : "text-muted hover:text-ink-2")
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
          <div className="flex-1" />
          <button
            onClick={() => setPaletteOpen(true)}
            className="group flex h-8 min-w-0 items-center gap-2 rounded-lg bg-sunken px-2.5 text-sm text-muted transition-colors hover:text-ink-2 sm:w-[300px]"
            style={{ boxShadow: "0 0 0 1px var(--hairline)" }}
            aria-label="Search tickers and commands"
          >
            <Search className="size-3.5 shrink-0" aria-hidden />
            <span className="hidden flex-1 truncate text-left sm:block">Search ticker or command…</span>
            <span className="hidden items-center gap-0.5 sm:flex">
              <Kbd>{modKey}</Kbd>
              <Kbd>K</Kbd>
            </span>
          </button>
          <button className="btn btn-ghost size-8 px-0" onClick={toggle} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} title="Toggle theme (t)">
            {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </button>
          <button className="btn btn-ghost hidden size-8 px-0 sm:inline-flex" onClick={() => setHelpOpen(true)} aria-label="Keyboard shortcuts" title="Keyboard shortcuts (?)">
            <Keyboard className="size-4" />
          </button>
        </div>
        <nav className="no-scrollbar flex gap-1 overflow-x-auto px-3 pb-2 md:hidden" aria-label="Primary">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) =>
                cx("shrink-0 rounded-md px-3 py-1 text-sm font-medium", isActive ? "bg-raised text-ink" : "text-muted")
              }
            >
              {n.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="mx-auto w-full max-w-[1440px] flex-1 px-4 pb-16 pt-4 sm:px-6 sm:pt-5">{children}</main>
      <footer className="mx-auto w-full max-w-[1440px] px-4 pb-6 text-2xs text-muted sm:px-6">
        SentiNET fuses public news, social, analyst, insider, filing and GDELT data. Not investment advice. Missing data is shown as missing — never filled in.
      </footer>
      <CommandPalette />
      {helpOpen && <HelpOverlay onClose={() => setHelpOpen(false)} />}
    </div>
  );
}

const SHORTCUTS: Array<[string[], string]> = [
  [[modKey, "K"], "Command palette"],
  [["/"], "Search tickers"],
  [["g", "m"], "Market overview"],
  [["g", "w"], "Watchlist & alerts"],
  [["g", "c"], "Compare"],
  [["g", "l"], "Sentiment lab"],
  [["r"], "Refresh analysis (ticker page)"],
  [["w"], "Watch / unwatch (ticker page)"],
  [["t"], "Toggle theme"],
  [["?"], "This help"],
];

function HelpOverlay({ onClose }: { onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);
  useRestoreFocus(closeRef);
  return (
    <div className="fixed inset-0 z-[95] flex items-center justify-center bg-black/50 px-4 backdrop-blur-[2px] animate-fade-in" onMouseDown={onClose}>
      <div role="dialog" aria-modal="true" aria-label="Keyboard shortcuts" className="w-full max-w-sm rounded-xl bg-panel p-4 shadow-pop" onMouseDown={(e) => e.stopPropagation()}>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold">Keyboard shortcuts</h2>
          <button ref={closeRef} className="btn btn-ghost size-7 px-0" onClick={onClose} aria-label="Close">
            <X className="size-4" />
          </button>
        </div>
        <dl className="divide-hair">
          {SHORTCUTS.map(([keys, label]) => (
            <div key={label} className="flex items-center justify-between py-2 text-sm">
              <dt className="text-ink-2">{label}</dt>
              <dd className="flex gap-1">
                {keys.map((k) => (
                  <Kbd key={k}>{k}</Kbd>
                ))}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  );
}
