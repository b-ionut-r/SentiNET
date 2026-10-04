import { lazy, Suspense } from "react";
import { Link, Route, Routes } from "react-router-dom";

import { AppShell } from "./components/layout/AppShell";
import { Empty, Skeleton } from "./components/ui/Misc";

// Route-level code splitting: each page (and its chart libraries) loads on demand.
const MarketPage = lazy(() => import("./features/market/MarketPage"));
const IntelPage = lazy(() => import("./features/intel/IntelPage"));
const ComparePage = lazy(() => import("./features/compare/ComparePage"));
const LabPage = lazy(() => import("./features/lab/LabPage"));
const WatchlistPage = lazy(() => import("./features/watchlist/WatchlistPage"));

function PageFallback() {
  return (
    <div className="space-y-4" aria-busy="true">
      <Skeleton className="h-10 w-72" />
      <Skeleton className="h-56 w-full rounded-xl" />
      <div className="grid gap-4 md:grid-cols-3">
        <Skeleton className="h-40 rounded-xl" />
        <Skeleton className="h-40 rounded-xl" />
        <Skeleton className="h-40 rounded-xl" />
      </div>
    </div>
  );
}

function NotFound() {
  return (
    <Empty title="Page not found">
      Try the <Link className="link" to="/">market overview</Link> or press / to search a ticker.
    </Empty>
  );
}

export default function App() {
  return (
    <AppShell>
      <Suspense fallback={<PageFallback />}>
        <Routes>
          <Route path="/" element={<MarketPage />} />
          <Route path="/t/:ticker" element={<IntelPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/lab" element={<LabPage />} />
          <Route path="/watchlist" element={<WatchlistPage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
    </AppShell>
  );
}
