// Thin wrapper around fetch. Paths are relative ("/api/...") so the same code
// works behind the Vite dev proxy and when served by FastAPI in production.
export async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

export const escapeText = (s) => String(s ?? "");
export const shorten = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
export const fmtScore = (s) => (typeof s === "number" ? s.toFixed(2) : "n/a");
