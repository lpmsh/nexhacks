// Configuration for the COSMOS client
// For Bun, environment variables are available at build time via Bun.env
// or can be hardcoded for simplicity

export const config = {
  // API base URL - change this to your backend URL
  // Leave empty string if frontend is served by the same server as the API
  // Set to "http://localhost:8000" if running frontend separately from backend
  apiBase: "http://localhost:8000",
} as const;

export type Config = typeof config;
