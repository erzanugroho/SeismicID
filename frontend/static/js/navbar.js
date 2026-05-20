// Inject shared navbar into <header id="navbar"></header>
export function renderNavbar(activePage) {
  const html = `
    <nav class="navbar">
      <div class="brand"><span class="accent">Gempa</span> Forecast</div>
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
  // After replace, mark active link
  setTimeout(() => {
    document.querySelectorAll(".navbar .nav-links a").forEach((a) => {
      if (a.dataset.page === activePage) a.classList.add("active");
    });
  }, 0);
}
