/** Price move between two stored looks (pure, unit-tested in e2e/unit.mjs). */
import type { Snapshot } from "../../api/types";

type Look = Pick<Snapshot, "price" | "currency"> | null | undefined;

/**
 * Percent change from the previous look's price to the latest, or null when either
 * price is missing or the two are quoted in different units (GBp vs GBP would read
 * as a 99% crash). A look stored before snapshots carried a currency counts as the
 * same unit.
 */
export function priceChange(last: Look, previous: Look): number | null {
  if (last?.price == null || !previous?.price) return null;
  if (last.currency && previous.currency && last.currency !== previous.currency) return null;
  return (last.price / previous.price - 1) * 100;
}
