import type { ReactNode } from "react";

import { cx } from "../../lib/cx";

interface PanelProps {
  title?: ReactNode;
  subtitle?: ReactNode;
  icon?: ReactNode;
  actions?: ReactNode;
  footer?: ReactNode;
  id?: string;
  className?: string;
  bodyClassName?: string;
  /** Remove body padding (lists that run edge to edge). */
  flush?: boolean;
  children: ReactNode;
}

/** The standard card: hairline ring, quiet header, optional actions and footer. */
export function Panel({ title, subtitle, icon, actions, footer, id, className, bodyClassName, flush, children }: PanelProps) {
  return (
    <section id={id} className={cx("panel flex flex-col min-w-0 scroll-mt-36 md:scroll-mt-28", className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start gap-x-3 gap-y-2 px-4 pt-3.5 pb-2.5">
          <div className="min-w-[min(100%,200px)] flex-1">
            <h2 className="flex items-center gap-2 text-sm font-semibold text-ink">
              {icon && <span className="text-muted [&>svg]:size-4">{icon}</span>}
              {title}
            </h2>
            {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
        </header>
      )}
      <div className={cx("min-w-0 flex-1", !flush && "px-4 pb-4", bodyClassName)}>{children}</div>
      {footer && <footer className="hairline-t px-4 py-2.5 text-xs text-muted">{footer}</footer>}
    </section>
  );
}

/** A small labelled block inside a panel. */
export function SubHead({ children, right, className }: { children: ReactNode; right?: ReactNode; className?: string }) {
  return (
    <div className={cx("mb-2 flex items-baseline justify-between gap-2", className)}>
      <h3 className="eyebrow">{children}</h3>
      {right && <div className="text-xs text-muted">{right}</div>}
    </div>
  );
}
