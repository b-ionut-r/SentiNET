/**
 * Each score component's actual share of the SentiNET score (pure, unit-tested in
 * e2e/unit.mjs). The nominal weights (30/15/20/10/10/15) are not what the score uses:
 * the backend (analytics/composite.py `compose`) scales each available component by
 * its confidence — w × (0.35 + 0.65·confidence) — and renormalizes over the available
 * ones, so an n/a component has no share and the rest absorb its weight. Showing those
 * shares lets the components reproduce the score's lean.
 */
import type { Component } from "../../api/types";

export function effectiveShares(components: ReadonlyArray<Pick<Component, "key" | "weight" | "available" | "confidence">>): Map<string, number> {
  const raw = components.filter((c) => c.available).map((c) => [c.key, c.weight * (0.35 + 0.65 * Math.max(0, Math.min(1, c.confidence)))] as const);
  const total = raw.reduce((s, [, w]) => s + w, 0);
  return new Map(total > 0 ? raw.map(([k, w]) => [k, w / total]) : []);
}

/** Integer percentages that add up to exactly 100 (largest-remainder rounding). */
export function wholePercents(shares: Map<string, number>): Map<string, number> {
  const entries = [...shares.entries()].map(([k, v]) => ({ k, exact: v * 100, floor: Math.floor(v * 100) }));
  let left = entries.length ? 100 - entries.reduce((s, e) => s + e.floor, 0) : 0;
  const out = new Map(entries.map((e) => [e.k, e.floor]));
  for (const e of [...entries].sort((x, y) => y.exact - y.floor - (x.exact - x.floor))) {
    if (left <= 0) break;
    out.set(e.k, (out.get(e.k) ?? 0) + 1);
    left -= 1;
  }
  return out;
}
