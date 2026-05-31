"""Telegram bot command + inline keyboard flow for user forecast areas."""

from __future__ import annotations

import html
import json
import urllib.error
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


def _send(chat_id: int | str, text: str, keyboard: list[list[dict[str, str]]] | None = None) -> bool:
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if keyboard:
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


def ensure_region_catalog() -> None:
    """Seed selectable Telegram regions from area_labels.

    Source hierarchy:
    province -> subregion -> forecast cell/full_label.
    This keeps setup private and uses existing forecast coverage. Full kecamatan
    gazetteer can replace/extend this table later without changing bot flow.
    """
    migrate()
    with get_connection() as conn:
        existing = conn.execute("SELECT COUNT(*) AS n FROM telegram_regions").fetchone()["n"]
        area_count = conn.execute("SELECT COUNT(*) AS n FROM area_labels").fetchone()["n"]
        if existing > 0 or area_count == 0:
            return
        provinces = conn.execute(
            """
            SELECT COALESCE(NULLIF(province, ''), NULLIF(region_macro, ''), 'Indonesia') AS name,
                   AVG(lat) AS lat, AVG(lon) AS lon
            FROM area_labels
            GROUP BY name
            ORDER BY name
            """
        ).fetchall()
        province_ids: dict[str, int] = {}
        for p in provinces:
            cur = conn.execute(
                """INSERT INTO telegram_regions(code, name, level, parent_id, lat, lon)
                   VALUES (?, ?, 'province', NULL, ?, ?)""",
                (f"prov:{p['name']}", p["name"], p["lat"], p["lon"]),
            )
            assert cur.lastrowid is not None
            province_ids[p["name"]] = int(cur.lastrowid)

        subregions = conn.execute(
            """
            SELECT COALESCE(NULLIF(province, ''), NULLIF(region_macro, ''), 'Indonesia') AS province_name,
                   COALESCE(NULLIF(subregion, ''), full_label) AS name,
                   AVG(lat) AS lat, AVG(lon) AS lon
            FROM area_labels
            GROUP BY province_name, name
            ORDER BY province_name, name
            """
        ).fetchall()
        subregion_ids: dict[tuple[str, str], int] = {}
        for r in subregions:
            parent_id = province_ids.get(r["province_name"])
            if not parent_id:
                continue
            cur = conn.execute(
                """INSERT INTO telegram_regions(code, name, level, parent_id, lat, lon)
                   VALUES (?, ?, 'regency', ?, ?, ?)""",
                (f"reg:{r['province_name']}:{r['name']}", r["name"], parent_id, r["lat"], r["lon"]),
            )
            assert cur.lastrowid is not None
            subregion_ids[(r["province_name"], r["name"])] = int(cur.lastrowid)

        cells = conn.execute(
            """
            SELECT cell_id, full_label, lat, lon,
                   COALESCE(NULLIF(province, ''), NULLIF(region_macro, ''), 'Indonesia') AS province_name,
                   COALESCE(NULLIF(subregion, ''), full_label) AS subregion_name
            FROM area_labels
            ORDER BY province_name, subregion_name, full_label
            """
        ).fetchall()
        for c in cells:
            parent_id = subregion_ids.get((c["province_name"], c["subregion_name"]))
            if not parent_id:
                continue
            conn.execute(
                """INSERT INTO telegram_regions(code, name, level, parent_id, lat, lon, cell_id)
                   VALUES (?, ?, 'district', ?, ?, ?, ?)""",
                (f"cell:{c['cell_id']}", c["full_label"], parent_id, c["lat"], c["lon"], c["cell_id"]),
            )
        logger.info("telegram_region_catalog_seeded", provinces=len(provinces), cells=len(cells))


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
        return f"Pilih kab/kota/subregion di {_e(parent['name'])}:"
    return f"Pilih area forecast di {_e(parent['name'])}:"


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
    if not region or region.get("level") != "district" or not region.get("cell_id"):
        return False
    path = _path(region_id)
    province = next((p["name"] for p in path if p["level"] == "province"), None)
    regency = next((p["name"] for p in path if p["level"] == "regency"), None)
    district = region["name"]
    now = datetime.now(UTC).isoformat()
    lat = round(float(region["lat"]), 1) if region.get("lat") is not None else None
    lon = round(float(region["lon"]), 1) if region.get("lon") is not None else None
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO telegram_user_locations(
                chat_id, username, first_name, province, regency, district,
                lat_rounded, lon_rounded, nearest_cell_id, area_label, radius_km,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 50, ?, ?)
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
                updated_at = excluded.updated_at
            """,
            (
                str(chat_id),
                user.get("username"),
                user.get("first_name"),
                province,
                regency,
                district,
                lat,
                lon,
                region["cell_id"],
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


def _start_text() -> str:
    return (
        "🌏 <b>SeismicID Bot</b>\n\n"
        "Bot ini memberi laporan probabilitas gempa berbasis area pilihan kamu.\n\n"
        "Command:\n"
        "/setlokasi — atur area laporan\n"
        "/lokasi — lihat area tersimpan\n"
        "/laporan — cek risiko area kamu\n"
        "/hapuslokasi — hapus data lokasi\n"
        "/bantuan — daftar command\n\n"
        "Output bukan peringatan resmi. Info keselamatan tetap BMKG."
    )


def _handle_message(message: dict[str, Any]) -> bool:
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return False
    text = (message.get("text") or "").strip()
    cmd = text.split()[0].split("@")[0].lower() if text else ""
    if cmd in {"/start", "/bantuan", "/help"}:
        return _send(chat_id, _start_text())
    if cmd == "/setlokasi":
        return _show_picker(chat_id)
    if cmd == "/lokasi":
        loc = _get_location(chat_id)
        return _send(chat_id, _location_text(loc) if loc else "📍 Area belum diatur. Ketik /setlokasi.")
    if cmd == "/hapuslokasi":
        _delete_location(chat_id)
        return _send(chat_id, "✅ Data lokasi kamu sudah dihapus.")
    if cmd == "/laporan":
        return _send(chat_id, _report(chat_id))
    return _send(chat_id, "Command belum dikenali. Ketik /bantuan.")


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
    if "message" in update:
        return _handle_message(update["message"])
    if "callback_query" in update:
        return _handle_callback(update["callback_query"])
    logger.info("telegram_update_ignored", keys=list(update.keys()))
    return False
