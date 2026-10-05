/**
 * What one lab run can score (pure, unit-tested in e2e/unit.mjs).
 *
 * Mirrors the backend's bounds (backend/app/services/lab.py, ScoreRequest): at most
 * `MAX_ITEMS` texts, each at most `MAX_TEXT_CHARS` characters, `MAX_TOTAL_CHARS` in
 * all. Characters are Unicode code points, as Python's `len` counts them (an emoji is
 * one character, not two UTF-16 units). Instead of letting the server reject the whole
 * batch, the page sends what fits and says exactly what was cut or left out.
 */
import { int, plural } from "../../lib/format";

export const MAX_ITEMS = 500;
export const MAX_TEXT_CHARS = 10_000;
export const MAX_TOTAL_CHARS = 250_000;

/** Characters (Unicode code points) in `s`; a surrogate pair counts once. */
export function charCount(s: string): number {
  let n = s.length;
  for (let i = 0; i < s.length - 1; i++) {
    const c = s.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff) {
      const d = s.charCodeAt(i + 1);
      if (d >= 0xdc00 && d <= 0xdfff) {
        n--;
        i++;
      }
    }
  }
  return n;
}

/** UTF-16 index just past the first `max` code points of `s` (never splits a surrogate pair). */
function indexAfter(s: string, max: number): number {
  let i = 0;
  for (let seen = 0; seen < max && i < s.length; seen++) {
    const c = s.charCodeAt(i);
    const pair = c >= 0xd800 && c <= 0xdbff && i + 1 < s.length && s.charCodeAt(i + 1) >= 0xdc00 && s.charCodeAt(i + 1) <= 0xdfff;
    i += pair ? 2 : 1;
  }
  return i;
}

/** How far back a cut may move to land on a word boundary instead of mid-word. */
const WORD_SLACK = 200;

/** `s` cut to at most `max` characters — at the last space within reach, so no half-word is scored. */
export function clip(s: string, max = MAX_TEXT_CHARS): string {
  if (charCount(s) <= max) return s;
  const end = indexAfter(s, max);
  const head = s.slice(0, end);
  if (!/\s/.test(s[end] ?? " ")) {
    const space = head.search(/\s\S*$/);
    if (space > 0 && end - space <= WORD_SLACK) return head.slice(0, space).trimEnd();
  }
  return head.trimEnd();
}

export interface BatchPlan {
  /** What will be sent, in input order. */
  texts: string[];
  /** Non-blank input items. */
  items: number;
  /** Characters in the input items. */
  inputChars: number;
  /** Characters actually sent. */
  chars: number;
  /** Sent texts that were longer than `MAX_TEXT_CHARS` and were cut. */
  cut: number;
  /** Input items not sent at all (past the item limit or the character budget). */
  left: number;
  /** Why items were left out: the item limit, or the character budget (which binds first). */
  leftBy: "items" | "budget" | null;
}

/** Fit one-item-per-entry input (already trimmed, blanks dropped) to one run's limits. */
export function planBatch(input: readonly string[]): BatchPlan {
  const items = input.filter((t) => t.trim());
  const texts: string[] = [];
  let inputChars = 0;
  let chars = 0;
  let cut = 0;
  let leftBy: BatchPlan["leftBy"] = null;
  for (const raw of items) {
    const n = charCount(raw);
    inputChars += n;
    if (leftBy) continue;
    if (texts.length >= MAX_ITEMS) {
      leftBy = "items";
      continue;
    }
    const text = n > MAX_TEXT_CHARS ? clip(raw) : raw;
    const size = n > MAX_TEXT_CHARS ? charCount(text) : n;
    if (chars + size > MAX_TOTAL_CHARS) {
      leftBy = "budget";
      continue;
    }
    if (text !== raw) cut++;
    texts.push(text);
    chars += size;
  }
  return { texts, items: items.length, inputChars, chars, cut, left: items.length - texts.length, leftBy };
}

/** Plain-language notes on what the run will cut or leave out (empty when everything fits). */
export function planNotes(p: BatchPlan): string[] {
  const notes: string[] = [];
  if (p.cut) {
    notes.push(
      `${p.cut === 1 ? "1 text is" : `${int(p.cut)} texts are`} longer than ${int(MAX_TEXT_CHARS)} characters and will be cut to ${p.cut === 1 ? "its" : "their"} first ${int(MAX_TEXT_CHARS)}.`,
    );
  }
  if (p.leftBy === "budget") {
    notes.push(`Over the ${int(MAX_TOTAL_CHARS)}-character budget of one run: the first ${int(p.texts.length)} of ${plural(p.items, "item")} will be scored, ${int(p.left)} left out — score the rest in another run.`);
  } else if (p.leftBy === "items") {
    notes.push(`One run scores at most ${int(MAX_ITEMS)} items: the first ${int(MAX_ITEMS)} of ${int(p.items)} will be scored, ${int(p.left)} left out — score the rest in another run.`);
  }
  return notes;
}
