// Common API client. Loaded by every page.

export const api = {
  async get(path, params = {}) {
    const qs = new URLSearchParams(
      Object.entries(params).filter(([_, v]) => v !== undefined && v !== null && v !== "")
    ).toString();
    const url = qs ? `${path}?${qs}` : path;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  },
  async post(path, params = {}) {
    const qs = new URLSearchParams(
      Object.entries(params).filter(([_, v]) => v !== undefined && v !== null && v !== "")
    ).toString();
    const url = qs ? `${path}?${qs}` : path;
    const r = await fetch(url, { method: "POST" });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  },
};

export function showToast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

// Backend ensemble.PROB_FLOOR — anything at or below this is the "data minim"
// floor and should not be presented as evidence of safety.
export const PROB_FLOOR = 1e-6;
// "Very low" probabilities still warrant a clear "data minim / belum cukup data"
// hint on the UI so users do not misread tiny values as guarantees.
export const PROB_VERY_LOW = 1e-4;

export function isProbAtFloor(p) {
  return p != null && p <= PROB_FLOOR * 1.01;
}

export function isProbVeryLow(p) {
  return p != null && p <= PROB_VERY_LOW;
}

export function probColor(p) {
  if (p == null) return "#475569";
  if (p < 0.005) return "#10b981";   // < 0.5% — Hijau (rendah)
  if (p < 0.01)  return "#84cc16";   // 0.5–1% — Hijau-kuning (rendah)
  if (p < 0.03)  return "#f59e0b";   // 1–3%   — Kuning/amber (sedang)
  if (p < 0.06)  return "#f97316";   // 3–6%   — Oranye (tinggi)
  return "#ef4444";                   // > 6%   — Merah (sangat tinggi)
}

export function formatPct(p) {
  if (p == null) return "—";
  if (isProbAtFloor(p)) return "data minim";
  const pct = p * 100;
  if (pct < 0.01) return "< 0.01%";
  if (pct < 1) return `${pct.toFixed(2)}%`;
  return `${pct.toFixed(1)}%`;
}

export function formatTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString("id-ID", { dateStyle: "short", timeStyle: "short" });
}

// Human "X lalu" relative time, falls back to absolute for old timestamps.
export function formatRelative(iso) {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const diffMs = Date.now() - then;
  const sec = Math.round(diffMs / 1000);
  if (sec < 0) return "baru saja";
  if (sec < 60) return "baru saja";
  const min = Math.round(sec / 60);
  if (min < 60) return `${min} menit lalu`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr} jam lalu`;
  const day = Math.round(hr / 24);
  if (day < 7) return `${day} hari lalu`;
  return formatTime(iso);
}

export function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[c]));
}
