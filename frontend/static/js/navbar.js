// Inject shared navbar into <header id="navbar"></header>
export function renderNavbar(activePage) {
  const html = `
    <nav class="navbar">
      <div class="brand">seismic<span class="accent">id</span></div>
      <button class="navbar-toggle" id="navbar-toggle" aria-label="Toggle navigation" aria-expanded="false">
        <span class="hamburger-icon"></span>
      </button>
      <ul class="nav-links">
        <li><a href="/" data-page="map">map</a></li>
        <li><a href="/cells.html" data-page="cells">cells</a></li>
        <li><a href="/events.html" data-page="events">events</a></li>
        <li><a href="/performance.html" data-page="performance">model</a></li>
        <li><a href="/backtest.html" data-page="backtest">backtest</a></li>
        <li><a href="/changelog.html" data-page="changelog">changelog</a></li>
        <li><a href="/scheduler.html" data-page="scheduler">scheduler</a></li>
        <li><a href="/about.html" data-page="about">about</a></li>
      </ul>
      ${activePage === "map" ? `
      <div class="nav-search">
        <span class="search-icon">/</span>
        <input type="text" id="search-input" class="search-box-input" placeholder="cari daerah… (palu, aceh)" aria-label="Cari daerah">
        <ul class="search-results" id="search-results"></ul>
      </div>` : ""}
      <span class="meta" id="footer-meta">v0.1 // id</span>
    </nav>
  `;
  const el = document.getElementById("navbar");
  if (el) el.outerHTML = html;
  // After replace, mark active link and setup hamburger
  setTimeout(() => {
    document.querySelectorAll(".navbar .nav-links a").forEach((a) => {
      if (a.dataset.page === activePage) a.classList.add("active");
    });

    // Hamburger menu toggle
    const toggle = document.getElementById("navbar-toggle");
    const navLinks = document.querySelector(".navbar .nav-links");
    if (toggle && navLinks) {
      toggle.addEventListener("click", () => {
        const isOpen = navLinks.classList.toggle("is-open");
        toggle.setAttribute("aria-expanded", isOpen);
      });
      // Close menu when a link is clicked
      navLinks.querySelectorAll("a").forEach((link) => {
        link.addEventListener("click", () => {
          navLinks.classList.remove("is-open");
          toggle.setAttribute("aria-expanded", "false");
        });
      });
      const closeMenu = () => {
        navLinks.classList.remove("is-open");
        toggle.setAttribute("aria-expanded", "false");
      };

      // Close when tapping the dark backdrop/pseudo-overlay area.
      document.querySelector(".navbar")?.addEventListener("click", (e) => {
        if (
          navLinks.classList.contains("is-open") &&
          !e.target.closest(".nav-links") &&
          !e.target.closest(".navbar-toggle")
        ) {
          closeMenu();
        }
      });

      // Close menu on outside click
      document.addEventListener("click", (e) => {
        if (!e.target.closest(".navbar")) closeMenu();
      });

      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeMenu();
      });
    }

    // First-visit disclaimer modal
    showDisclaimerIfNeeded();
  }, 0);
}

// Show first-visit disclaimer popup. Closes via X, backdrop click, ESC, or "mengerti".
// Persists via localStorage so it appears only once per browser.
const DISCLAIMER_KEY = "seismicid_disclaimer_seen_v1";

function showDisclaimerIfNeeded() {
  try {
    if (localStorage.getItem(DISCLAIMER_KEY) === "1") return;
  } catch (_) {
    // localStorage may be blocked (private mode) — show once per session anyway
  }

  const overlay = document.createElement("div");
  overlay.className = "disclaimer-modal";
  overlay.setAttribute("role", "presentation");
  overlay.innerHTML = `
    <div class="disclaimer-modal-content" role="dialog" aria-modal="true" aria-labelledby="disclaimer-modal-title" aria-describedby="disclaimer-modal-text">
      <button type="button" class="disclaimer-modal-close" aria-label="Tutup">&times;</button>
      <h2 id="disclaimer-modal-title" class="disclaimer-modal-title">eksperimental — bukan sistem peringatan dini</h2>
      <p id="disclaimer-modal-text" class="disclaimer-modal-text">
        gunakan bmkg untuk informasi resmi. probabilitas yang ditampilkan adalah ranking risiko relatif berbasis model riset, bukan prediksi deterministik kapan atau di mana gempa akan terjadi.
      </p>
      <div class="disclaimer-modal-actions">
        <button type="button" class="btn disclaimer-modal-ok">mengerti</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
  document.body.classList.add("modal-open");

  const close = () => {
    try { localStorage.setItem(DISCLAIMER_KEY, "1"); } catch (_) {}
    overlay.remove();
    document.body.classList.remove("modal-open");
    document.removeEventListener("keydown", onKey);
  };
  const onKey = (e) => { if (e.key === "Escape") close(); };

  // Backdrop click closes (only when clicking the overlay itself, not the dialog).
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
  overlay.querySelector(".disclaimer-modal-close").addEventListener("click", close);
  overlay.querySelector(".disclaimer-modal-ok").addEventListener("click", close);
  document.addEventListener("keydown", onKey);

  // Move keyboard focus to the dialog for accessibility.
  const okBtn = overlay.querySelector(".disclaimer-modal-ok");
  if (okBtn) okBtn.focus();
}
