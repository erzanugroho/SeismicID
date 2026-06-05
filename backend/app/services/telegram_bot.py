"""Telegram bot command + inline keyboard flow for user forecast areas."""

from __future__ import annotations

import html
import csv
import io
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from typing import Any

from backend.app.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.db.sqlite import get_connection, migrate

logger = get_logger(__name__)
PAGE_SIZE = 8


def _api(method: str, payload: dict[str, Any]) -> bool:
    settings = get_settings()
    if not settings.telegram_bot_token:
        logger.info("telegram_bot_skip", reason="not_configured", method=method)
        return False
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/{method}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            ok = 200 <= resp.status < 300
            logger.info("telegram_bot_api", method=method, ok=ok, status=resp.status)
            return ok
    except urllib.error.HTTPError as exc:
        logger.warning("telegram_bot_api_failed", method=method, status=exc.code)
    except Exception as exc:  # noqa: BLE001
        logger.warning("telegram_bot_api_failed", method=method, error=str(exc))
    return False


def _send(
    chat_id: int | str,
    text: str,
    keyboard: list[list[dict[str, str]]] | None = None,
    reply_markup: dict[str, Any] | None = None,
) -> bool:
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    elif keyboard:
        payload["reply_markup"] = {"inline_keyboard": keyboard}
    return _api("sendMessage", payload)


def _edit(chat_id: int | str, message_id: int, text: str, keyboard: list[list[dict[str, str]]] | None = None) -> bool:
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if keyboard:
        payload["reply_markup"] = {"inline_keyboard": keyboard}
    return _api("editMessageText", payload)


def _answer_callback(callback_id: str, text: str = "") -> None:
    payload: dict[str, Any] = {"callback_query_id": callback_id}
    if text:
        payload["text"] = text
    _api("answerCallbackQuery", payload)


def ensure_bot_commands() -> None:
    migrate()
    with get_connection() as conn:
        version = conn.execute("SELECT value FROM app_metadata WHERE key = 'telegram_bot_commands_version'").fetchone()
        if version and version["value"] == "menu-v1":
            return
        ok = _api(
            "setMyCommands",
            {
                "commands": [
                    {"command": "start", "description": "Mulai dan tampilkan menu"},
                    {"command": "menu", "description": "Tampilkan tombol menu"},
                    {"command": "setlokasi", "description": "Atur area laporan"},
                    {"command": "lokasi", "description": "Lihat area tersimpan"},
                    {"command": "laporan", "description": "Cek risiko area kamu"},
                    {"command": "hapuslokasi", "description": "Hapus data lokasi"},
                    {"command": "stopbot", "description": "Berhenti menerima bot/alert"},
                    {"command": "help", "description": "Bantuan"},
                    {"command": "admin", "description": "Panel admin bot"},
                ]
            },
        )
        if ok:
            conn.execute(
                """INSERT INTO app_metadata(key, value, updated_at)
                   VALUES ('telegram_bot_commands_version', 'menu-v1', datetime('now'))
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at"""
            )


def _admin_ids() -> set[str]:
    settings = get_settings()
    return {x.strip() for x in settings.telegram_bot_admin_chat_ids.split(",") if x.strip()}


def _is_admin(chat_id: int | str) -> bool:
    return str(chat_id) in _admin_ids()


def _main_menu(chat_id: int | str) -> dict[str, Any]:
    rows = [
        [{"text": "📍 Atur lokasi"}, {"text": "📊 Laporan"}],
        [{"text": "🗺 Lokasi saya"}, {"text": "❓ Help"}],
        [{"text": "🛑 Stop bot"}],
    ]
    if _is_admin(chat_id):
        rows.append([{"text": "👑 Admin"}])
    return {"keyboard": rows, "resize_keyboard": True, "is_persistent": True}


def _e(value: Any) -> str:
    return html.escape("—" if value is None else str(value), quote=False)


def _risk_level(prob: float | None) -> str:
    if prob is None:
        return "belum ada data"
    if prob < 0.001:
        return "rendah"
    if prob < 0.005:
        return "rendah-sedang"
    if prob < 0.01:
        return "sedang"
    if prob < 0.02:
        return "meningkat"
    return "tinggi relatif"


def _pct(prob: float | None) -> str:
    return "—" if prob is None else f"{prob * 100:.2f}%"


def _cell_detail_url(cell_id: str) -> str:
    settings = get_settings()
    base_url = "https://seismicid.erzanugroho.xyz"
    cors_origins = (settings.cors_allow_origins or "").split(",")
    for origin in cors_origins:
        origin = origin.strip().rstrip("/")
        if origin.startswith("https://") and "localhost" not in origin and "127.0.0.1" not in origin:
            base_url = origin
            break
    return f"{base_url}/area.html?cell={urllib.parse.quote(str(cell_id))}"


def _fetch_csv(url: str) -> list[list[str]]:
    with urllib.request.urlopen(url, timeout=25) as resp:
        text = resp.read().decode("utf-8-sig")
    return [row for row in csv.reader(io.StringIO(text)) if row]


def _nearest_cell(conn, lat: float, lon: float) -> dict[str, Any] | None:  # noqa: ANN001
    rows = conn.execute("SELECT cell_id, full_label, lat, lon FROM area_labels").fetchall()
    best: dict[str, Any] | None = None
    best_dist = float("inf")
    for row in rows:
        d = (float(row["lat"]) - lat) ** 2 + (float(row["lon"]) - lon) ** 2
        if d < best_dist:
            best_dist = d
            best = dict(row)
    return best


def _geocode_region(query: str) -> tuple[float | None, float | None]:
    """Geocode district centroid with Nominatim; return approximate lat/lon."""
    params = urllib.parse.urlencode({"q": query, "format": "jsonv2", "limit": "1"})
    req = urllib.request.Request(
        f"https://nominatim.openstreetmap.org/search?{params}",
        headers={"User-Agent": "SeismicID/1.0 (telegram region setup)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not data:
            return None, None
        return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("telegram_region_geocode_failed", query=query, error=str(exc))
        return None, None


def ensure_region_catalog() -> None:
    """Seed selectable regions from official Indonesia admin-code CSV.

    Hierarchy is province -> kab/kota -> kecamatan. Kecamatan coordinates are
    resolved lazily when selected, then matched to nearest forecast cell.
    """
    migrate()
    with get_connection() as conn:
        version = conn.execute(
            "SELECT value FROM app_metadata WHERE key = 'telegram_region_catalog_version'"
        ).fetchone()
        existing = conn.execute("SELECT COUNT(*) AS n FROM telegram_regions").fetchone()["n"]
        if existing > 0 and version and version["value"] == "admin-v1":
            return
        conn.execute("DELETE FROM telegram_regions")
        provinces = _fetch_csv(
            "https://raw.githubusercontent.com/edwardsamuel/Wilayah-Administratif-Indonesia/master/csv/provinces.csv"
        )
        regencies = _fetch_csv(
            "https://raw.githubusercontent.com/edwardsamuel/Wilayah-Administratif-Indonesia/master/csv/regencies.csv"
        )
        districts = _fetch_csv(
            "https://raw.githubusercontent.com/edwardsamuel/Wilayah-Administratif-Indonesia/master/csv/districts.csv"
        )
        province_ids: dict[str, int] = {}
        for code, name in provinces:
            cur = conn.execute(
                """INSERT INTO telegram_regions(code, name, level, parent_id)
                   VALUES (?, ?, 'province', NULL)""",
                (code, name.title()),
            )
            assert cur.lastrowid is not None
            province_ids[code] = int(cur.lastrowid)
        regency_ids: dict[str, int] = {}
        for code, province_code, name in regencies:
            parent_id = province_ids.get(province_code)
            if not parent_id:
                continue
            cur = conn.execute(
                """INSERT INTO telegram_regions(code, name, level, parent_id)
                   VALUES (?, ?, 'regency', ?)""",
                (code, name.title(), parent_id),
            )
            assert cur.lastrowid is not None
            regency_ids[code] = int(cur.lastrowid)
        for code, regency_code, name in districts:
            parent_id = regency_ids.get(regency_code)
            if not parent_id:
                continue
            conn.execute(
                """INSERT INTO telegram_regions(code, name, level, parent_id)
                   VALUES (?, ?, 'district', ?)""",
                (code, name.title(), parent_id),
            )
        conn.execute(
            """INSERT INTO app_metadata(key, value, updated_at)
               VALUES ('telegram_region_catalog_version', 'admin-v1', datetime('now'))
               ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at"""
        )
        logger.info(
            "telegram_region_catalog_seeded",
            provinces=len(provinces),
            regencies=len(regencies),
            districts=len(districts),
        )


def _children(parent_id: int | None, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    ensure_region_catalog()
    with get_connection() as conn:
        if parent_id is None:
            total = conn.execute(
                "SELECT COUNT(*) AS n FROM telegram_regions WHERE parent_id IS NULL"
            ).fetchone()["n"]
            rows = conn.execute(
                """SELECT * FROM telegram_regions WHERE parent_id IS NULL
                   ORDER BY name LIMIT ? OFFSET ?""",
                (PAGE_SIZE, offset),
            ).fetchall()
        else:
            total = conn.execute(
                "SELECT COUNT(*) AS n FROM telegram_regions WHERE parent_id = ?",
                (parent_id,),
            ).fetchone()["n"]
            rows = conn.execute(
                """SELECT * FROM telegram_regions WHERE parent_id = ?
                   ORDER BY name LIMIT ? OFFSET ?""",
                (parent_id, PAGE_SIZE, offset),
            ).fetchall()
    return [dict(r) for r in rows], int(total)


def _region(region_id: int) -> dict[str, Any] | None:
    ensure_region_catalog()
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM telegram_regions WHERE id = ?", (region_id,)).fetchone()
    return dict(row) if row else None


def _path(region_id: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cur_id: int | None = region_id
    with get_connection() as conn:
        while cur_id:
            row = conn.execute("SELECT * FROM telegram_regions WHERE id = ?", (cur_id,)).fetchone()
            if not row:
                break
            item = dict(row)
            out.append(item)
            cur_id = item.get("parent_id")
    return list(reversed(out))


def _keyboard(parent_id: int | None, offset: int = 0) -> list[list[dict[str, str]]]:
    rows, total = _children(parent_id, offset)
    keyboard: list[list[dict[str, str]]] = []
    for row in rows:
        keyboard.append([{"text": row["name"][:48], "callback_data": f"loc:sel:{row['id']}"}])
    nav: list[dict[str, str]] = []
    if offset > 0:
        nav.append({"text": "< Kembali", "callback_data": f"loc:page:{parent_id or 0}:{max(0, offset - PAGE_SIZE)}"})
    if offset + PAGE_SIZE < total:
        nav.append({"text": "Berikutnya >", "callback_data": f"loc:page:{parent_id or 0}:{offset + PAGE_SIZE}"})
    if nav:
        keyboard.append(nav)
    if parent_id:
        parent = _region(parent_id)
        back_id = int(parent["parent_id"] or 0) if parent else 0
        keyboard.append([{"text": "Naik satu level", "callback_data": f"loc:page:{back_id}:0"}])
    keyboard.append([{"text": "Batal", "callback_data": "loc:cancel"}])
    return keyboard


def _level_title(parent_id: int | None) -> str:
    if parent_id is None:
        return "Pilih provinsi:"
    parent = _region(parent_id)
    if not parent:
        return "Pilih wilayah:"
    if parent["level"] == "province":
        return f"Pilih kab/kota di {_e(parent['name'])}:"
    return f"Pilih kecamatan di {_e(parent['name'])}:"


def _show_picker(chat_id: int | str, *, parent_id: int | None = None, offset: int = 0, message_id: int | None = None) -> bool:
    text = (
        "📍 <b>Atur Area Laporan SeismicID</b>\n\n"
        f"{_level_title(parent_id)}\n\n"
        "Catatan privasi: pilihan wilayah dipakai hanya untuk mencari cell forecast terdekat."
    )
    keyboard = _keyboard(parent_id, offset)
    if message_id:
        return _edit(chat_id, message_id, text, keyboard)
    return _send(chat_id, text, keyboard)


def _save_location(chat_id: int | str, user: dict[str, Any], region_id: int) -> bool:
    region = _region(region_id)
    if not region or region.get("level") != "district":
        return False
    path = _path(region_id)
    province = next((p["name"] for p in path if p["level"] == "province"), None)
    regency = next((p["name"] for p in path if p["level"] == "regency"), None)
    district = region["name"]
    now = datetime.now(UTC).isoformat()
    lat = float(region["lat"]) if region.get("lat") is not None else None
    lon = float(region["lon"]) if region.get("lon") is not None else None
    query = f"{district}, {regency}, {province}, Indonesia"
    with get_connection() as conn:
        if lat is None or lon is None:
            lat, lon = _geocode_region(query)
            if lat is not None and lon is not None:
                conn.execute(
                    "UPDATE telegram_regions SET lat = ?, lon = ?, updated_at = datetime('now') WHERE id = ?",
                    (lat, lon, region_id),
                )
        if lat is None or lon is None:
            return False
        cell = _nearest_cell(conn, lat, lon)
        if not cell:
            return False
        conn.execute("DELETE FROM telegram_bot_opt_outs WHERE chat_id = ?", (str(chat_id),))
        conn.execute(
            """
            INSERT INTO telegram_user_locations(
                chat_id, username, first_name, province, regency, district,
                lat_rounded, lon_rounded, nearest_cell_id, area_label, radius_km,
                created_at, updated_at, stopped_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 50, ?, ?, NULL)
            ON CONFLICT(chat_id) DO UPDATE SET
                username = excluded.username,
                first_name = excluded.first_name,
                province = excluded.province,
                regency = excluded.regency,
                district = excluded.district,
                lat_rounded = excluded.lat_rounded,
                lon_rounded = excluded.lon_rounded,
                nearest_cell_id = excluded.nearest_cell_id,
                area_label = excluded.area_label,
                stopped_at = NULL,
                updated_at = excluded.updated_at
            """,
            (
                str(chat_id),
                user.get("username"),
                user.get("first_name"),
                province,
                regency,
                district,
                round(lat, 1),
                round(lon, 1),
                cell["cell_id"],
                district,
                now,
                now,
            ),
        )
    return True


def _get_location(chat_id: int | str) -> dict[str, Any] | None:
    migrate()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM telegram_user_locations WHERE chat_id = ?", (str(chat_id),)
        ).fetchone()
    return dict(row) if row else None


def _delete_location(chat_id: int | str) -> None:
    migrate()
    with get_connection() as conn:
        conn.execute("DELETE FROM telegram_user_locations WHERE chat_id = ?", (str(chat_id),))


def _stop_bot(chat_id: int | str) -> None:
    migrate()
    now = datetime.now(UTC).isoformat()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO telegram_bot_opt_outs(chat_id, stopped_at)
               VALUES (?, ?)
               ON CONFLICT(chat_id) DO UPDATE SET stopped_at = excluded.stopped_at""",
            (str(chat_id), now),
        )
        conn.execute(
            """UPDATE telegram_user_locations
               SET stopped_at = ?, updated_at = ?
               WHERE chat_id = ?""",
            (now, now, str(chat_id)),
        )


def _is_stopped(chat_id: int | str) -> bool:
    migrate()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT stopped_at FROM telegram_bot_opt_outs WHERE chat_id = ?", (str(chat_id),)
        ).fetchone()
        if row:
            return True
    loc = _get_location(chat_id)
    return bool(loc and loc.get("stopped_at"))


def _location_text(loc: dict[str, Any]) -> str:
    return (
        "📍 <b>Area laporan tersimpan</b>\n\n"
        f"Provinsi: {_e(loc.get('province'))}\n"
        f"Kab/Kota/Subregion: {_e(loc.get('regency'))}\n"
        f"Area: {_e(loc.get('district'))}\n"
        f"Cell: <code>{_e(loc.get('nearest_cell_id'))}</code>\n"
        f"Radius pantau: {_e(loc.get('radius_km'))} km\n\n"
        "Ketik /laporan untuk cek risiko terbaru.\n"
        "Ketik /hapuslokasi untuk hapus data lokasi."
    )


def _report(chat_id: int | str) -> str:
    loc = _get_location(chat_id)
    if not loc:
        return (
            "📍 Area kamu belum diatur.\n\n"
            "Ketik /setlokasi untuk memilih provinsi, kab/kota, lalu area forecast."
        )
    cell_id = loc["nearest_cell_id"]
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT horizon_days, mag_threshold, probability, computed_at, model_version
            FROM current_forecasts
            WHERE cell_id = ? AND mag_threshold = 5.0
            ORDER BY horizon_days
            """,
            (cell_id,),
        ).fetchall()
    forecasts = [dict(r) for r in rows]
    p30 = next((float(r["probability"]) for r in forecasts if int(r["horizon_days"]) == 30), None)
    lines = [
        "📍 <b>Laporan Risiko Area Kamu</b>",
        f"Wilayah: {_e(loc.get('area_label'))}",
        f"Cell: <code>{_e(cell_id)}</code>",
        f"Detail cell: {_e(_cell_detail_url(cell_id))}",
        f"Update: {_e(forecasts[0]['computed_at'] if forecasts else None)}",
        "",
        "Probabilitas M ≥ 5.0:",
    ]
    if forecasts:
        for row in forecasts:
            lines.append(f"• {int(row['horizon_days'])} hari: {_pct(float(row['probability']))}")
    else:
        lines.append("• belum ada forecast aktif")
    lines.extend(
        [
            "",
            f"Status risiko: {_risk_level(p30)}",
            "",
            "Catatan: ini model probabilistik, bukan prediksi pasti. Untuk peringatan resmi, ikuti BMKG.",
        ]
    )
    return "\n".join(lines)


def _admin_text() -> str:
    if not _admin_ids():
        return "Admin bot belum dikonfigurasi."
    migrate()
    with get_connection() as conn:
        users = conn.execute("SELECT COUNT(*) AS n FROM telegram_user_locations").fetchone()["n"]
        stopped = conn.execute("SELECT COUNT(*) AS n FROM telegram_bot_opt_outs").fetchone()["n"]
        rows = conn.execute(
            """
            SELECT day, COUNT(*) AS dau, SUM(hits) AS hits
            FROM daily_active_users
            GROUP BY day
            ORDER BY day DESC
            LIMIT 3
            """
        ).fetchall()
    lines = [
        "👑 <b>Admin Bot SeismicID</b>",
        "",
        f"User lokasi tersimpan: {_e(users)}",
        f"User stopbot: {_e(stopped)}",
        "",
        "DAU terbaru:",
    ]
    if rows:
        for row in rows:
            lines.append(f"• {_e(row['day'])}: {_e(row['dau'])} user · {_e(row['hits'])} hits")
    else:
        lines.append("• belum ada data")
    lines.extend(["", "Command admin:", "/admin — panel admin", "/botusers — jumlah user bot", "/dau — ringkas DAU"])
    return "\n".join(lines)


def _start_text() -> str:
    return (
        "🌏 <b>SeismicID Bot</b>\n\n"
        "Bot ini memberi laporan probabilitas gempa berbasis area pilihan kamu.\n\n"
        "Command:\n"
        "/setlokasi — atur area laporan\n"
        "/lokasi — lihat area tersimpan\n"
        "/laporan — cek risiko area kamu\n"
        "/hapuslokasi — hapus data lokasi\n"
        "/stopbot — berhenti menerima bot/alert area\n"
        "/menu — tampilkan tombol menu\n"
        "/help atau /bantuan — daftar command\n"
        "/admin — panel admin bot (admin saja)\n\n"
        "Output bukan peringatan resmi. Info keselamatan tetap BMKG."
    )


def _handle_message(message: dict[str, Any]) -> bool:
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return False
    text = (message.get("text") or "").strip()
    button_map = {
        "📍 atur lokasi": "/setlokasi",
        "📊 laporan": "/laporan",
        "🗺 lokasi saya": "/lokasi",
        "❓ help": "/help",
        "🛑 stop bot": "/stopbot",
        "👑 admin": "/admin",
    }
    normalized = text.lower()
    cmd = button_map.get(normalized) or (text.split()[0].split("@")[0].lower() if text else "")
    if cmd in {"/start", "/bantuan", "/help", "/menu"}:
        return _send(chat_id, _start_text(), reply_markup=_main_menu(chat_id))
    if cmd in {"/admin", "/botusers", "/dau"}:
        if not _is_admin(chat_id):
            return _send(chat_id, "Command admin hanya untuk admin bot.", reply_markup=_main_menu(chat_id))
        return _send(chat_id, _admin_text(), reply_markup=_main_menu(chat_id))
    if cmd == "/stopbot":
        _stop_bot(chat_id)
        return _send(
            chat_id,
            "🛑 Bot dihentikan untuk chat ini.\n\n"
            "Laporan/alert area tidak akan dikirim.\n"
            "Ketik /setlokasi untuk mengaktifkan lagi.",
            reply_markup={"remove_keyboard": True},
        )
    if cmd == "/setlokasi":
        return _show_picker(chat_id)
    if _is_stopped(chat_id):
        return _send(chat_id, "🛑 Bot sedang dihentikan. Ketik /setlokasi untuk aktifkan lagi.")
    if cmd == "/lokasi":
        loc = _get_location(chat_id)
        return _send(chat_id, _location_text(loc) if loc else "📍 Area belum diatur. Ketik /setlokasi.", reply_markup=_main_menu(chat_id))
    if cmd == "/hapuslokasi":
        _delete_location(chat_id)
        return _send(chat_id, "✅ Data lokasi kamu sudah dihapus.", reply_markup=_main_menu(chat_id))
    if cmd == "/laporan":
        return _send(chat_id, _report(chat_id), reply_markup=_main_menu(chat_id))
    return _send(chat_id, "Command belum dikenali. Ketik /help.", reply_markup=_main_menu(chat_id))


def _handle_callback(callback: dict[str, Any]) -> bool:
    callback_id = callback.get("id")
    data = callback.get("data") or ""
    message = callback.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    message_id = message.get("message_id")
    user = callback.get("from") or {}
    if callback_id:
        _answer_callback(callback_id)
    if not chat_id or not message_id:
        return False
    if data == "loc:cancel":
        return _edit(chat_id, message_id, "Setup lokasi dibatalkan.")
    parts = data.split(":")
    if len(parts) >= 4 and parts[:2] == ["loc", "page"]:
        parent_raw = int(parts[2])
        parent_id = parent_raw or None
        offset = int(parts[3])
        return _show_picker(chat_id, parent_id=parent_id, offset=offset, message_id=message_id)
    if len(parts) == 3 and parts[:2] == ["loc", "sel"]:
        region_id = int(parts[2])
        region = _region(region_id)
        if not region:
            return _edit(chat_id, message_id, "Wilayah tidak ditemukan. Ketik /setlokasi untuk ulang.")
        if region["level"] != "district":
            return _show_picker(chat_id, parent_id=region_id, offset=0, message_id=message_id)
        if not _save_location(chat_id, user, region_id):
            return _edit(chat_id, message_id, "Gagal menyimpan area. Ketik /setlokasi untuk ulang.")
        loc = _get_location(chat_id)
        if not loc:
            return _edit(chat_id, message_id, "Area tersimpan, tapi gagal dibaca ulang. Ketik /lokasi.")
        return _edit(chat_id, message_id, "✅ Area laporan disimpan.\n\n" + _location_text(loc))
    return _edit(chat_id, message_id, "Aksi tidak dikenali. Ketik /setlokasi untuk ulang.")


def handle_update(update: dict[str, Any]) -> bool:
    """Handle one Telegram webhook update."""
    ensure_region_catalog()
    ensure_bot_commands()
    if "message" in update:
        return _handle_message(update["message"])
    if "callback_query" in update:
        return _handle_callback(update["callback_query"])
    logger.info("telegram_update_ignored", keys=list(update.keys()))
    return False
