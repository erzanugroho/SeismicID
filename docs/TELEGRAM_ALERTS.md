# Telegram Alerts

SeismicID memiliki bot Telegram untuk laporan risiko gempa. Bot ini dirancang agar informatif, bukan berisik.

## Tujuan

- Memberi ringkasan risiko harian.
- Memberi alert cepat hanya saat risiko berubah signifikan.
- Menghindari pesan berulang tiap 10–30 menit saat probabilitas tidak berubah.

## Jadwal Laporan Harian

Default:

```text
07:00 WIB
```

Konfigurasi:

```env
TELEGRAM_DAILY_REPORT_HOUR_UTC=0
```

Catatan:

```text
00:00 UTC = 07:00 WIB
```

Isi laporan:

- top 5 area risiko tertinggi
- horizon 30 hari
- threshold M ≥ 5.0
- disclaimer eksperimental

## Alert Perubahan Signifikan

Alert dikirim jika salah satu kondisi terjadi:

1. Top #1 cell berubah.
2. Probabilitas berubah minimal 0.5 percentage point.
3. Perubahan relatif minimal 25%.
4. Risiko melewati ambang alert dari bawah ke atas.

Konfigurasi:

```env
TELEGRAM_ALERT_MIN_PROBABILITY=0.03
TELEGRAM_SIGNIFICANT_ABS_DELTA=0.005
TELEGRAM_SIGNIFICANT_REL_DELTA=0.25
```

Arti nilai:

```text
0.03  = 3% minimum top risk
0.005 = 0.5 percentage point
0.25  = 25% perubahan relatif
```

## Anti-spam

Bot tidak mengirim alert setiap forecast recompute.

Alur:

```text
forecast recompute
  → ambil top 5 terbaru
  → bandingkan dengan snapshot forecast sebelumnya
  → jika signifikan, kirim Telegram
  → jika tidak signifikan, diam
```

First run setelah deploy hanya membuat baseline snapshot. Tidak langsung mengirim alert perubahan palsu.

## Metadata yang Disimpan

Di SQLite metadata table:

```text
telegram_last_forecast_snapshot
telegram_last_alert_snapshot
telegram_last_alert_at
telegram_last_daily_report_date
telegram_last_daily_report_at
```

## Manual Trigger

Lewat admin scheduler UI atau endpoint:

```http
POST /api/scheduler/trigger/telegram_daily_report
Authorization: Bearer <ADMIN_TOKEN>
```

## Disclaimer

Pesan Telegram bukan peringatan dini resmi. Gunakan BMKG/otoritas terkait untuk keselamatan.
