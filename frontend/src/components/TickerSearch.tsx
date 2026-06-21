import { FormEvent, useState } from "react";

interface Props {
  onSearch: (ticker: string) => void;
  initial?: string;
}

const POPULAR = ["AAPL", "TSLA", "NVDA", "GME", "AMD", "MSFT"];

export function TickerSearch({ onSearch, initial = "" }: Props) {
  const [value, setValue] = useState(initial);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const t = value.trim().toUpperCase();
    if (t) onSearch(t);
  };

  return (
    <div className="w-full">
      <form onSubmit={submit} className="flex gap-2">
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Search a ticker — e.g. AAPL, TSLA, NVDA"
          className="flex-1 rounded-xl border border-edge bg-panel px-4 py-3 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-accent"
          autoFocus
        />
        <button
          type="submit"
          className="rounded-xl bg-accent px-5 py-3 text-sm font-semibold text-slate-900 transition hover:bg-sky-300"
        >
          Analyze
        </button>
      </form>
      <div className="mt-2 flex flex-wrap gap-2">
        {POPULAR.map((t) => (
          <button
            key={t}
            onClick={() => {
              setValue(t);
              onSearch(t);
            }}
            className="rounded-full border border-edge bg-panel-2 px-3 py-1 text-xs text-slate-400 transition hover:border-accent hover:text-accent"
          >
            {t}
          </button>
        ))}
      </div>
    </div>
  );
}
