import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During `npm run dev`, the Vite dev server (http://localhost:5173) proxies
// /api/* to the FastAPI backend (http://localhost:8000), so the app is
// same-origin in development. `npm run build` emits static files into dist/,
// which FastAPI serves directly in production.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
