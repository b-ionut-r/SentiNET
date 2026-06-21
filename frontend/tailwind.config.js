/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0a0e17",
        panel: "#111726",
        "panel-2": "#161d2e",
        edge: "#1f2937",
        bull: "#22c55e",
        bear: "#ef4444",
        neutral: "#eab308",
        accent: "#38bdf8",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
