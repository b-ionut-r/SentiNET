/**
 * Command registry: pages contribute context actions (e.g. "Refresh AAPL")
 * that the command palette lists alongside global navigation.
 */
import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

export interface Command {
  id: string;
  title: string;
  hint?: string;
  icon?: ReactNode;
  run: () => void;
}

interface Registry {
  commands: Command[];
  setPageCommands: (cmds: Command[]) => void;
  paletteOpen: boolean;
  setPaletteOpen: (open: boolean) => void;
  helpOpen: boolean;
  setHelpOpen: (open: boolean) => void;
}

const Ctx = createContext<Registry | null>(null);

export function CommandProvider({ children }: { children: ReactNode }) {
  const [commands, setPageCommands] = useState<Command[]>([]);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const value = useMemo(
    () => ({ commands, setPageCommands, paletteOpen, setPaletteOpen, helpOpen, setHelpOpen }),
    [commands, paletteOpen, helpOpen],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useCommands(): Registry {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useCommands outside CommandProvider");
  return ctx;
}

/**
 * Register page-level commands for as long as the page is mounted. Runs are
 * dispatched through a ref, so handlers always see the latest page state.
 */
export function usePageCommands(cmds: Command[]): void {
  const { setPageCommands } = useCommands();
  const latest = useRef(cmds);
  latest.current = cmds;
  const sig = cmds.map((c) => `${c.id}:${c.title}:${c.hint ?? ""}`).join("|");
  useEffect(() => {
    setPageCommands(
      latest.current.map((c) => ({ ...c, run: () => latest.current.find((x) => x.id === c.id)?.run() })),
    );
    return () => setPageCommands([]);
  }, [sig, setPageCommands]);
}
