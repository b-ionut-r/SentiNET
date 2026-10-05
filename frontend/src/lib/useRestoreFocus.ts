import { useEffect, useRef, type RefObject } from "react";

/**
 * Dialog focus contract: on mount, move focus into the dialog (`target`), and
 * on unmount hand it back to whatever opened it — never strand it on <body>.
 */
export function useRestoreFocus<T extends HTMLElement>(target: RefObject<T>): void {
  const opener = useRef<Element | null>(null);
  useEffect(() => {
    opener.current = document.activeElement;
    target.current?.focus();
    return () => {
      const el = opener.current;
      // Only restore when focus would otherwise be lost (a navigation may have moved it on purpose).
      if (el instanceof HTMLElement && el.isConnected && (document.activeElement === document.body || document.activeElement == null)) {
        el.focus();
      }
    };
  }, [target]);
}
