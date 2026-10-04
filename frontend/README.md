# SentiNET frontend

React 18 + TypeScript (strict) + Vite 5 + Tailwind 3, TanStack Query 5, react-router 6,
lightweight-charts 4.2 (price) and hand-built SVG for everything else.

```bash
npm install
npm run dev        # http://localhost:5173, /api proxied to SENTINET_API (default http://127.0.0.1:8000)
npm run build      # tsc -b (zero errors) + vite build → dist/ (served by FastAPI in production)
```

## Layout

| Path | What |
|---|---|
| `src/api/types.ts` | TS mirror of `backend/app/schemas.py` (contract — owned by the lead) |
| `src/api/client.ts` | typed fetchers, `ApiError`, SSE `streamAnalysis` with fallback to plain GET |
| `src/api/hooks.ts` | one TanStack Query hook per resource |
| `src/features/*` | route pages: `market` (/), `intel` (/t/:ticker), `compare`, `lab`, `watchlist` — code-split |
| `src/components/charts` | Dial, diverging/stacked/range bars, Columns, LineChart, Sparkline, Pulse |
| `src/components/ui` | Panel, chips/badges (sentiment is never color-only), tooltip, misc states |
| `src/components/layout` | app shell, ⌘K / `/` command palette, page command registry |
| `src/lib` | formatting (real minus signs, calendar-safe dates), sentiment semantics, theme, hotkeys |

Design tokens live in `src/styles/index.css` (dark default, light via `data-theme`); the
sentiment pair (bull/bear) and status colors are reserved and always carry a glyph or label.

## Verification (dev/test only — fixtures are never imported by the app)

`e2e/fixtures/*.json` are schema-valid sample responses; `*.LIVE.json` is verbatim output
of a real backend run. `e2e/mock.mjs` serves the built app with `vite preview` and answers
`/api/**` from them. Chromium comes from `/opt/pw-browsers` (`PLAYWRIGHT_BROWSERS_PATH`).

```bash
npm run build && npm run screens   # every page at 1440 + 390 px, dark + light → e2e/screens/
                                   # fails on page/console errors and horizontal overflow
npm run smoke                      # 26 interaction checks: shortcuts, palette, filters, alerts, export…
SENTINET_API=http://127.0.0.1:8000 npm run live -- AAPL   # real backend, no mocks → e2e/screens/live/
```

`node e2e/screens.mjs intel --w=375 --t=dark` limits a run to matching pages, widths and themes.
