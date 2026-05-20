"""Download geographic assets used by the forecast system.

Assets:
- GADM Indonesia level 1 (provinsi) shapefile     — administrative boundaries
- GADM Indonesia level 2 (kabupaten) shapefile    — sub-region boundaries
- USGS Slab2.0 grid (Sunda + Banda + Philippines) — subduction depth
- Optional: PUSGEN 2017 fault DB (or GEM Active Faults substitute)

Usage:
    python -m scripts.download_geo_assets
    python -m scripts.download_geo_assets --skip-faults

Outputs to ``data/geo/``. Files are skipped if already present.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

# Ensure project root on path so this script can be run as `python scripts/...`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.config import get_settings  # noqa: E402
from backend.app.core.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

# Source URLs (subject to occasional upstream changes; keep this list as the single source of truth)
GADM_LEVEL1_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_IDN_shp.zip"
SLAB2_BASE = "https://www.sciencebase.gov/catalog/file/get/"
SLAB_FILES = {
    # Slab2.0 grids (depth, .grd, geographic). Source DOI: https://doi.org/10.5066/F7PV6JNV
    "sun_slab2_dep.grd": "5b1b4f60e4b092d9651fbd7c?f=__disk__c4%2Fda%2F4d%2Fc4da4d8f3eb83055ee79df0d2b89c80d4ec6cd0d",  # placeholder
}


def download(url: str, target: Path, *, force: bool = False) -> Path:
    """Stream a URL to a local file. No-op if target exists (unless force)."""
    import requests
    from tenacity import retry, stop_after_attempt, wait_exponential

    if target.exists() and not force:
        logger.info("download_skip_exists", target=str(target))
        return target

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _do() -> None:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)

    logger.info("download_start", url=url, target=str(target))
    _do()
    logger.info("download_done", size_bytes=target.stat().st_size)
    return target


def unzip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    logger.info("unzip_done", source=str(zip_path), dest=str(dest))


def fetch_gadm(geo_dir: Path, *, force: bool) -> None:
    zip_path = geo_dir / "gadm41_IDN_shp.zip"
    download(GADM_LEVEL1_URL, zip_path, force=force)
    unzip(zip_path, geo_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    parser.add_argument("--skip-gadm", action="store_true")
    parser.add_argument("--skip-slab", action="store_true")
    parser.add_argument("--skip-faults", action="store_true")
    args = parser.parse_args()

    configure_logging("INFO")
    settings = get_settings()
    settings.ensure_dirs()
    geo_dir = settings.geo_path

    if not args.skip_gadm:
        try:
            fetch_gadm(geo_dir, force=args.force)
        except Exception as e:  # noqa: BLE001
            logger.error("gadm_download_failed", error=str(e))
    if not args.skip_slab:
        logger.warning(
            "slab_manual_download_required",
            url="https://www.sciencebase.gov/catalog/item/5aa1b00ee4b0b1c392e86467",
            note="Download Sunda + Banda + Philippines slab .grd files manually into data/geo/",
        )
    if not args.skip_faults:
        logger.warning(
            "fault_manual_download_required",
            note="PUSGEN 2017 not freely downloadable. Substitute: GEM Global Active Faults (https://github.com/GEMScienceTools/gem-global-active-faults)",
        )

    logger.info("download_geo_assets_complete", geo_dir=str(geo_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
