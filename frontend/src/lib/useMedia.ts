import { useCallback, useSyncExternalStore } from "react";

/** Live result of a CSS media query (false during SSR / when matchMedia is missing). */
export function useMedia(query: string): boolean {
  const subscribe = useCallback(
    (notify: () => void) => {
      const mql = window.matchMedia?.(query);
      mql?.addEventListener("change", notify);
      return () => mql?.removeEventListener("change", notify);
    },
    [query],
  );
  return useSyncExternalStore(subscribe, () => window.matchMedia?.(query).matches ?? false, () => false);
}
