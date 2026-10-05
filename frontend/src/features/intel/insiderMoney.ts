/**
 * Insider money (pure, unit-tested in e2e/unit.mjs).
 *
 * Insider trade values are US dollars from every source (backend
 * analytics/util.INSIDER_CURRENCY): Form 4 is filed in USD and Yahoo converts non-US
 * filings — VOD.L's "Sold at price 1.70 per share" is $1.70, i.e. 126.8p, not £1.70.
 * So they are never formatted in the quote's currency, and never divided by a market
 * cap quoted in another one.
 */
import { majorCurrency } from "../../lib/format";

export const INSIDER_CURRENCY = "USD";

/** The listing quotes in something other than US dollars (pence, euros…), or says nothing. */
export function insiderValuesForeign(quoteCurrency: string | null | undefined): boolean {
  return majorCurrency(quoteCurrency) !== INSIDER_CURRENCY;
}

/**
 * Total insider sales as a share of market cap, as the backend words it ("0.06%",
 * "<0.01%"); null unless the market cap is in US dollars too (a GBp/EUR cap would
 * mix currencies, and no FX rate is guessed).
 */
export function sellShareOfCap(sellValue: number, marketCap: number | null | undefined, quoteCurrency: string | null | undefined): string | null {
  if (!marketCap || marketCap <= 0 || !(sellValue > 0) || insiderValuesForeign(quoteCurrency)) return null;
  const bps = (sellValue / marketCap) * 1e4;
  return bps < 1 ? "<0.01%" : `${(bps / 100).toFixed(2)}%`;
}
