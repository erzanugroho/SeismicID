// Inject shared navbar into <header id="navbar"></header>
export function renderNavbar(activePage) {
  const html = `
    <nav class="navbar">
      <div class="brand"><span class="accent">Seismic</span>ID</div>
      <button class="navbar-toggle" id="navbar-toggle" aria-label="Toggle navigation" aria-expanded="false">
        <span class="hamburger-icon"></span>
      </button>
      <ul class="nav-links">
        <li><a href="/" data-page="map">Peta</a></li>
        <li><a href="/cells.html" data-page="cells">Daftar Cell</a></li>
        <li><a href="/events.html" data-page="events">Gempa Terkini</a></li>
        <li><a href="/performance.html" data-page="performance">Performa Model</a></li>
        <li><a href="/scheduler.html" data-page="scheduler">Scheduler</a></li>
        <li><a href="/about.html" data-page="about">Tentang</a></li>
      </ul>
      <span class="meta" id="footer-meta">v0.1 · ID</span>
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
  }, 0);
}
