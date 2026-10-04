import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// /api is proxied to the FastAPI backend (SSE included). Override the target
// with SENTINET_API, e.g. SENTINET_API=http://127.0.0.1:8765 npm run dev
const API = process.env.SENTINET_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": { target: API, changeOrigin: true } },
  },
  preview: {
    proxy: { "/api": { target: API, changeOrigin: true } },
  },
  build: {
    target: "es2020",
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom", "react-router-dom"],
          query: ["@tanstack/react-query"],
        },
      },
    },
  },
});
