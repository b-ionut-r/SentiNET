import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During dev, proxy /api to the FastAPI backend on :8000.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
