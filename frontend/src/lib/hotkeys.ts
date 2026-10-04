import { useEffect, useRef, type MutableRefObject } from "react";

/**
 * Keyboard shortcuts through one global dispatcher, so sequences ("g w") and
 * page-level single keys ("w") never both fire. Binding syntax:
 *   "r", "?", "/"   single key (ignored while typing in a field)
 *   "g m"           sequence (second key within 900 ms)
 *   "mod+k"         ⌘ on macOS, Ctrl elsewhere (fires even in fields)
 *   "Escape"        fires even in fields
 * Later-registered bindings (the current page) win over earlier ones (the shell).
 */
export type Bindings = Record<string, (e: KeyboardEvent) => void>;

const SEQ_TIMEOUT = 900;
const registry: Array<MutableRefObject<Bindings>> = [];
let prefix: { key: string; at: number } | null = null;

function isEditable(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  const tag = el.tagName;
  return el.isContentEditable || tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

function find(name: string): ((e: KeyboardEvent) => void) | undefined {
  for (let i = registry.length - 1; i >= 0; i--) {
    const fn = registry[i].current[name];
    if (fn) return fn;
  }
  return undefined;
}

function hasPrefix(key: string): boolean {
  return registry.some((r) => Object.keys(r.current).some((k) => k.startsWith(`${key} `)));
}

function dispatch(e: KeyboardEvent): void {
  if (e.defaultPrevented) return;
  const mod = e.metaKey || e.ctrlKey;
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;

  if (mod) {
    const fn = find(`mod+${key}`);
    if (fn) {
      e.preventDefault();
      fn(e);
    }
    return;
  }
  if (key === "Escape") {
    find("Escape")?.(e);
    return;
  }
  if (e.altKey || isEditable(e.target)) return;

  const now = Date.now();
  if (prefix && now - prefix.at < SEQ_TIMEOUT) {
    const fn = find(`${prefix.key} ${key}`);
    prefix = null;
    if (fn) {
      e.preventDefault();
      fn(e);
    }
    return; // a key right after a prefix never falls through to a single binding
  }
  prefix = null;
  if (hasPrefix(key)) {
    prefix = { key, at: now };
    return;
  }
  const fn = find(e.key === "?" ? "?" : key);
  if (fn) {
    e.preventDefault();
    fn(e);
  }
}

export function useHotkeys(bindings: Bindings, enabled = true): void {
  const ref = useRef(bindings);
  ref.current = bindings;

  useEffect(() => {
    if (!enabled) return;
    if (registry.length === 0) window.addEventListener("keydown", dispatch);
    registry.push(ref);
    return () => {
      const i = registry.indexOf(ref);
      if (i >= 0) registry.splice(i, 1);
      if (registry.length === 0) window.removeEventListener("keydown", dispatch);
    };
  }, [enabled]);
}

export const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);
export const modKey = isMac ? "⌘" : "Ctrl";
