/**
 * Sentiment semantics shared by every view: polarity from scores, glyphs that
 * carry direction without color, and diverging color mixes for marks.
 */
import type { Polarity, SentimentLabel } from "../api/types";

/** Engine neutral band for -1..1 sentiment scores. */
export const NEUTRAL_BAND = 0.05;

/** Verdict label bands on the 0..100 SentiNET scale (mirrors the backend). */
export const VERDICT_BANDS: ReadonlyArray<{ min: number; max: number; label: string; level: number }> = [
  { min: 0, max: 25, label: "Strongly Bearish", level: -3 },
  { min: 26, max: 38, label: "Bearish", level: -2 },
  { min: 39, max: 45, label: "Leaning Bearish", level: -1 },
  { min: 46, max: 54, label: "Neutral", level: 0 },
  { min: 55, max: 61, label: "Leaning Bullish", level: 1 },
  { min: 62, max: 74, label: "Bullish", level: 2 },
  { min: 75, max: 100, label: "Strongly Bullish", level: 3 },
];

export function polarityOf(score: number | null | undefined, band = NEUTRAL_BAND): Polarity {
  if (score == null || !Number.isFinite(score)) return "neutral";
  if (score > band) return "bull";
  if (score < -band) return "bear";
  return "neutral";
}

/** Polarity of a 0..100 score (neutral band 46–54). */
export function polarityOf100(score: number | null | undefined): Polarity {
  if (score == null || !Number.isFinite(score)) return "neutral";
  if (score >= 55) return "bull";
  if (score <= 45) return "bear";
  return "neutral";
}

export function polarityOfLabel(label: SentimentLabel | null | undefined): Polarity {
  return label === "bullish" ? "bull" : label === "bearish" ? "bear" : "neutral";
}

export function labelOf(p: Polarity): string {
  return p === "bull" ? "Bullish" : p === "bear" ? "Bearish" : "Neutral";
}

/** Direction glyph: ▲ bull, ▼ bear, ● neutral. */
export function glyph(p: Polarity): string {
  return p === "bull" ? "▲" : p === "bear" ? "▼" : "●";
}

export const textTone: Record<Polarity, string> = {
  bull: "text-bull",
  bear: "text-bear",
  neutral: "text-neu",
};

export const bgTone: Record<Polarity, string> = {
  bull: "bg-bull",
  bear: "bg-bear",
  neutral: "bg-neu",
};

export function toneVar(p: Polarity, alpha = 1): string {
  const v = p === "bull" ? "--bull" : p === "bear" ? "--bear" : "--neu";
  return alpha >= 1 ? `rgb(var(${v}))` : `rgb(var(${v}) / ${alpha})`;
}

/**
 * Diverging fill for t in [-1, 1]: bear pole ↔ gray midpoint ↔ bull pole.
 * Mixing in OKLab keeps the ramp perceptually even.
 */
export function divergingFill(t: number): string {
  const x = Math.max(-1, Math.min(1, t));
  if (Math.abs(x) < 0.02) return "rgb(var(--mid))";
  const pole = x > 0 ? "--bull" : "--bear";
  const pctPole = Math.round(30 + 70 * Math.abs(x));
  return `color-mix(in oklab, rgb(var(${pole})) ${pctPole}%, rgb(var(--mid)))`;
}

/** Soft background tint for chips/cells; strength 0..1. */
export function tintFor(score: number, strength = 1, max = 0.5): string {
  const p = polarityOf(score);
  if (p === "neutral") return "rgb(var(--neu) / 0.12)";
  const a = Math.min(0.22, 0.07 + (Math.min(Math.abs(score), max) / max) * 0.15) * strength;
  return toneVar(p, a);
}

export function verdictBand(score: number) {
  return VERDICT_BANDS.find((b) => score >= b.min && score <= b.max) ?? VERDICT_BANDS[3];
}

/** Fear & Greed bands (CNN convention). */
export function fearGreedBand(score: number): { label: string; polarity: Polarity } {
  if (score < 25) return { label: "Extreme Fear", polarity: "bear" };
  if (score < 45) return { label: "Fear", polarity: "bear" };
  if (score <= 55) return { label: "Neutral", polarity: "neutral" };
  if (score <= 75) return { label: "Greed", polarity: "bull" };
  return { label: "Extreme Greed", polarity: "bull" };
}
