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

export function probColor(p) {
  if (p == null) return "#475569";
  if (p < 0.005) return "#10b981";   // < 0.5% — Hijau (aman)
  if (p < 0.01)  return "#84cc16";   // 0.5–1% — Hijau-kuning (rendah)
  if (p < 0.03)  return "#f59e0b";   // 1–3%   — Kuning/amber (sedang)
  if (p < 0.06)  return "#f97316";   // 3–6%   — Oranye (tinggi)
  return "#ef4444";                   // > 6%   — Merah (sangat tinggi)
}

export function formatPct(p) {
  if (p == null) return "—";
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

export function activateNav(name) {
  document.querySelectorAll(".navbar .nav-links a").forEach((a) => {
    if (a.dataset.page === name) a.classList.add("active");
  });
}
