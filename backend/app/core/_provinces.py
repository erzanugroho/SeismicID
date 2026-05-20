"""Hardcoded Indonesian province bounding boxes and anchor cities.

Used as a fallback geocoder when GADM shapefiles aren't downloaded yet.
Coordinates are rough approximations sufficient for grid-cell labelling.
For higher-accuracy production labels, use shapefile-based geocoder
(see geocode.py:_shapefile_geocoder).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorCity:
    name: str
    lat: float
    lon: float


@dataclass(frozen=True)
class ProvinceBox:
    name: str
    macro: str  # Sumatera|Jawa|BaliNusa|Kalimantan|Sulawesi|MalukuPapua
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    anchors: tuple[AnchorCity, ...]

    def contains(self, lat: float, lon: float) -> bool:
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max

    @property
    def area(self) -> float:
        return (self.lat_max - self.lat_min) * (self.lon_max - self.lon_min)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.lat_min + self.lat_max) / 2.0, (self.lon_min + self.lon_max) / 2.0)


PROVINCES: tuple[ProvinceBox, ...] = (
    # === Sumatera ===
    ProvinceBox("Aceh", "Sumatera", 2.0, 6.0, 95.0, 98.5, (
        AnchorCity("Banda Aceh", 5.55, 95.32),
        AnchorCity("Lhokseumawe", 5.18, 97.14),
        AnchorCity("Meulaboh", 4.14, 96.13),
    )),
    ProvinceBox("Sumatera Utara", "Sumatera", 1.0, 4.3, 97.5, 100.5, (
        AnchorCity("Medan", 3.59, 98.67),
        AnchorCity("Pematangsiantar", 2.96, 99.06),
        AnchorCity("Sibolga", 1.74, 98.78),
    )),
    ProvinceBox("Sumatera Barat", "Sumatera", -3.5, 0.5, 99.0, 101.7, (
        AnchorCity("Padang", -0.95, 100.35),
        AnchorCity("Bukittinggi", -0.30, 100.37),
        AnchorCity("Mentawai", -2.50, 99.50),
    )),
    ProvinceBox("Riau", "Sumatera", -1.0, 2.5, 100.0, 103.5, (
        AnchorCity("Pekanbaru", 0.51, 101.45),
        AnchorCity("Dumai", 1.67, 101.45),
    )),
    ProvinceBox("Kepulauan Riau", "Sumatera", -1.0, 4.0, 103.0, 109.0, (
        AnchorCity("Tanjung Pinang", 0.92, 104.45),
        AnchorCity("Batam", 1.13, 104.05),
    )),
    ProvinceBox("Jambi", "Sumatera", -3.0, -0.5, 101.0, 104.5, (
        AnchorCity("Jambi", -1.61, 103.61),
    )),
    ProvinceBox("Bengkulu", "Sumatera", -5.5, -2.0, 101.5, 104.0, (
        AnchorCity("Bengkulu", -3.79, 102.26),
    )),
    ProvinceBox("Sumatera Selatan", "Sumatera", -5.0, -1.5, 102.0, 106.0, (
        AnchorCity("Palembang", -2.99, 104.76),
        AnchorCity("Lubuklinggau", -3.30, 102.86),
    )),
    ProvinceBox("Bangka Belitung", "Sumatera", -3.5, -1.0, 105.0, 108.3, (
        AnchorCity("Pangkal Pinang", -2.13, 106.11),
    )),
    ProvinceBox("Lampung", "Sumatera", -6.0, -3.5, 103.5, 106.0, (
        AnchorCity("Bandar Lampung", -5.43, 105.27),
    )),
    # === Jawa ===
    ProvinceBox("Banten", "Jawa", -7.0, -5.7, 105.0, 106.8, (
        AnchorCity("Serang", -6.12, 106.15),
        AnchorCity("Tangerang", -6.18, 106.63),
    )),
    ProvinceBox("DKI Jakarta", "Jawa", -6.4, -5.9, 106.5, 107.0, (
        AnchorCity("Jakarta", -6.20, 106.85),
    )),
    ProvinceBox("Jawa Barat", "Jawa", -7.8, -5.9, 106.3, 108.85, (
        AnchorCity("Bandung", -6.91, 107.61),
        AnchorCity("Bogor", -6.60, 106.80),
        AnchorCity("Sukabumi", -6.92, 106.93),
    )),
    ProvinceBox("Jawa Tengah", "Jawa", -8.2, -6.0, 108.5, 111.7, (
        AnchorCity("Semarang", -6.97, 110.42),
        AnchorCity("Yogyakarta", -7.80, 110.37),
        AnchorCity("Solo", -7.57, 110.83),
    )),
    ProvinceBox("DI Yogyakarta", "Jawa", -8.2, -7.5, 110.0, 110.85, (
        AnchorCity("Yogyakarta", -7.80, 110.37),
    )),
    ProvinceBox("Jawa Timur", "Jawa", -8.8, -6.5, 110.8, 114.6, (
        AnchorCity("Surabaya", -7.25, 112.75),
        AnchorCity("Malang", -7.98, 112.62),
        AnchorCity("Banyuwangi", -8.22, 114.37),
    )),
    # === Bali & Nusa Tenggara ===
    ProvinceBox("Bali", "BaliNusa", -8.9, -8.0, 114.4, 115.7, (
        AnchorCity("Denpasar", -8.65, 115.22),
    )),
    ProvinceBox("Nusa Tenggara Barat", "BaliNusa", -9.5, -8.0, 115.7, 119.2, (
        AnchorCity("Mataram", -8.58, 116.12),
        AnchorCity("Bima", -8.46, 118.73),
    )),
    ProvinceBox("Nusa Tenggara Timur", "BaliNusa", -11.0, -8.0, 118.7, 125.5, (
        AnchorCity("Kupang", -10.18, 123.61),
        AnchorCity("Maumere", -8.62, 122.21),
        AnchorCity("Ende", -8.85, 121.66),
    )),
    # === Kalimantan ===
    ProvinceBox("Kalimantan Barat", "Kalimantan", -3.0, 2.5, 108.5, 114.5, (
        AnchorCity("Pontianak", -0.02, 109.34),
        AnchorCity("Singkawang", 0.91, 108.98),
    )),
    ProvinceBox("Kalimantan Tengah", "Kalimantan", -4.0, 1.0, 110.5, 115.5, (
        AnchorCity("Palangkaraya", -2.21, 113.92),
    )),
    ProvinceBox("Kalimantan Selatan", "Kalimantan", -4.5, -1.2, 114.0, 117.0, (
        AnchorCity("Banjarmasin", -3.32, 114.59),
        AnchorCity("Banjarbaru", -3.44, 114.83),
    )),
    ProvinceBox("Kalimantan Timur", "Kalimantan", -2.5, 4.0, 113.5, 119.0, (
        AnchorCity("Samarinda", -0.50, 117.15),
        AnchorCity("Balikpapan", -1.27, 116.83),
    )),
    ProvinceBox("Kalimantan Utara", "Kalimantan", 1.0, 4.5, 114.0, 118.0, (
        AnchorCity("Tarakan", 3.30, 117.63),
        AnchorCity("Nunukan", 4.14, 117.66),
    )),
    # === Sulawesi ===
    ProvinceBox("Sulawesi Utara", "Sulawesi", 0.0, 5.0, 121.0, 127.0, (
        AnchorCity("Manado", 1.49, 124.84),
        AnchorCity("Bitung", 1.45, 125.18),
    )),
    ProvinceBox("Gorontalo", "Sulawesi", 0.0, 1.5, 121.0, 124.0, (
        AnchorCity("Gorontalo", 0.54, 123.06),
    )),
    ProvinceBox("Sulawesi Tengah", "Sulawesi", -3.5, 1.5, 119.0, 123.5, (
        AnchorCity("Palu", -0.90, 119.87),
        AnchorCity("Poso", -1.40, 120.75),
        AnchorCity("Luwuk", -0.95, 122.79),
    )),
    ProvinceBox("Sulawesi Barat", "Sulawesi", -4.0, -1.0, 118.5, 120.0, (
        AnchorCity("Mamuju", -2.68, 118.89),
    )),
    ProvinceBox("Sulawesi Selatan", "Sulawesi", -7.5, -2.0, 118.5, 122.0, (
        AnchorCity("Makassar", -5.13, 119.41),
        AnchorCity("Parepare", -4.02, 119.62),
        AnchorCity("Palopo", -3.00, 120.20),
    )),
    ProvinceBox("Sulawesi Tenggara", "Sulawesi", -7.0, -2.0, 120.0, 124.0, (
        AnchorCity("Kendari", -3.99, 122.51),
        AnchorCity("Bau-Bau", -5.47, 122.59),
    )),
    # === Maluku & Papua ===
    ProvinceBox("Maluku Utara", "MalukuPapua", -2.5, 3.0, 124.0, 130.0, (
        AnchorCity("Ternate", 0.79, 127.37),
        AnchorCity("Tidore", 0.69, 127.43),
    )),
    ProvinceBox("Maluku", "MalukuPapua", -8.5, -2.0, 125.0, 135.0, (
        AnchorCity("Ambon", -3.65, 128.18),
        AnchorCity("Tual", -5.64, 132.74),
    )),
    ProvinceBox("Papua Barat", "MalukuPapua", -4.5, 0.0, 130.0, 134.5, (
        AnchorCity("Manokwari", -0.86, 134.06),
        AnchorCity("Sorong", -0.88, 131.25),
    )),
    ProvinceBox("Papua", "MalukuPapua", -9.5, -1.0, 134.0, 141.0, (
        AnchorCity("Jayapura", -2.53, 140.72),
        AnchorCity("Wamena", -4.10, 138.95),
        AnchorCity("Merauke", -8.49, 140.40),
        AnchorCity("Timika", -4.53, 136.88),
    )),
)
