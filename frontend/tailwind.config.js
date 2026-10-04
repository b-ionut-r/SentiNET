/** @type {import('tailwindcss').Config} */
const rgb = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        page: rgb("page"),
        panel: rgb("panel"),
        raised: rgb("raised"),
        sunken: rgb("sunken"),
        ink: rgb("ink"),
        "ink-2": rgb("ink-2"),
        muted: rgb("muted"),
        faint: rgb("faint"),
        accent: rgb("accent"),
        bull: rgb("bull"),
        bear: rgb("bear"),
        neu: rgb("neu"),
        good: rgb("good"),
        warn: rgb("warn"),
        serious: rgb("serious"),
        critical: rgb("critical"),
        heat: rgb("heat"),
        line: "var(--hairline)",
        "line-strong": "var(--hairline-strong)",
      },
      fontFamily: {
        sans: ['"Inter Variable"', "Inter", "system-ui", "-apple-system", '"Segoe UI"', "sans-serif"],
        mono: ['"JetBrains Mono Variable"', '"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      fontSize: {
        "2xs": ["10.5px", { lineHeight: "14px" }],
        xs: ["11.5px", { lineHeight: "16px" }],
        sm: ["13px", { lineHeight: "19px" }],
        base: ["14px", { lineHeight: "21px" }],
      },
      borderRadius: {
        DEFAULT: "6px",
        lg: "8px",
        xl: "12px",
      },
      boxShadow: {
        pop: "0 12px 40px -8px rgb(0 0 0 / 0.45), 0 0 0 1px var(--hairline-strong)",
        hero: "inset 0 1px 0 0 var(--glow-top), 0 0 0 1px var(--hairline)",
      },
      keyframes: {
        "fade-in": { from: { opacity: "0", transform: "translateY(2px)" }, to: { opacity: "1", transform: "none" } },
        "scan-sweep": { "0%": { transform: "translateX(-100%)" }, "100%": { transform: "translateX(400%)" } },
        "pulse-soft": { "0%,100%": { opacity: "1" }, "50%": { opacity: "0.35" } },
      },
      animation: {
        "fade-in": "fade-in 180ms ease-out both",
        "scan-sweep": "scan-sweep 1.6s cubic-bezier(.4,0,.2,1) infinite",
        "pulse-soft": "pulse-soft 1.4s ease-in-out infinite",
      },
      transitionDuration: { DEFAULT: "150ms" },
      opacity: {
        6: "0.06",
        8: "0.08",
        12: "0.12",
        14: "0.14",
        16: "0.16",
        18: "0.18",
        22: "0.22",
        35: "0.35",
        45: "0.45",
        55: "0.55",
        65: "0.65",
        85: "0.85",
      },
    },
  },
  plugins: [],
};
