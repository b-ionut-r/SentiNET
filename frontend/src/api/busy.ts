/**
 * "Busy, retry shortly" handling (pure, unit-tested in e2e/unit.mjs). The backend
 * answers 503 with a `Retry-After` header when a bounded work queue is full (the
 * sentiment lab scores one batch at a time, two waiting). That is not a failure: wait
 * the time the server asked for and try again, a bounded number of times, cancellable.
 * A 503 without `Retry-After` is a real outage and is never retried.
 */

/** Automatic retries after a busy answer (so at most 1 + BUSY_RETRIES requests). */
export const BUSY_RETRIES = 3;
/** Longest wait honoured from a `Retry-After` header (seconds). */
export const MAX_RETRY_AFTER_S = 60;

/** `Retry-After` (delta-seconds or an HTTP date) → whole seconds in [1, MAX_RETRY_AFTER_S], or null. */
export function parseRetryAfter(header: string | null | undefined, now = Date.now()): number | null {
  const h = header?.trim();
  if (!h) return null;
  let s: number;
  if (/^\d+$/.test(h)) s = Number(h);
  else {
    const at = Date.parse(h);
    if (Number.isNaN(at)) return null;
    s = Math.ceil((at - now) / 1000);
  }
  return Math.min(MAX_RETRY_AFTER_S, Math.max(1, s));
}

/** Seconds to wait before retrying, if `err` is a busy answer (503 + Retry-After); else null. */
export function busyRetryAfter(err: unknown): number | null {
  if (!err || typeof err !== "object") return null;
  const { status, retryAfter } = err as { status?: unknown; retryAfter?: unknown };
  return status === 503 && typeof retryAfter === "number" && retryAfter > 0 ? retryAfter : null;
}

/** A pending retry after a busy answer. */
export interface BusyWait {
  /** Retry number about to happen (1 … retries). */
  retry: number;
  retries: number;
  seconds: number;
  /** Epoch ms when the retry fires. */
  until: number;
  message: string;
}

export interface BusyOptions {
  signal?: AbortSignal;
  retries?: number;
  /** Called with the pending retry before each wait, and with null once a request is under way again. */
  onWait?: (w: BusyWait | null) => void;
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
  now?: () => number;
}

const aborted = () => new DOMException("Aborted", "AbortError");

/** Resolve after `ms`, or reject with an AbortError as soon as `signal` aborts. */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(aborted());
    const onAbort = () => {
      clearTimeout(timer);
      reject(aborted());
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/**
 * Run `fn`; while it fails with a busy answer, wait the server's `Retry-After` and run
 * it again, at most `retries` more times. Any other error, the last busy answer, or an
 * abort (during a request or a wait) rejects.
 */
export async function retryWhenBusy<T>(fn: (signal?: AbortSignal) => Promise<T>, opts: BusyOptions = {}): Promise<T> {
  const { signal, retries = BUSY_RETRIES, onWait, sleep: wait = sleep, now = Date.now } = opts;
  for (let retry = 1; ; retry++) {
    try {
      return await fn(signal);
    } catch (err) {
      const seconds = busyRetryAfter(err);
      if (seconds == null || retry > retries || signal?.aborted) throw err;
      onWait?.({ retry, retries, seconds, until: now() + seconds * 1000, message: err instanceof Error ? err.message : String(err) });
      await wait(seconds * 1000, signal);
      onWait?.(null);
    }
  }
}
