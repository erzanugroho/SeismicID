# Dokumentasi SeismicID

Folder ini berisi dokumentasi teknis dan operasional SeismicID.

## Peta Dokumen

| Dokumen | Tujuan |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Arsitektur sistem, data flow, komponen backend/frontend |
| [`API.md`](API.md) | Reference endpoint REST API |
| [`DATA.md`](DATA.md) | Struktur data SQLite, Parquet, archive, dan runtime DB |
| [`MAINTENANCE.md`](MAINTENANCE.md) | Panduan operasi, testing, deploy, troubleshooting |
| [`PROBABILITY_GUIDE.md`](PROBABILITY_GUIDE.md) | Cara membaca probabilitas risiko |
| [`TELEGRAM_ALERTS.md`](TELEGRAM_ALERTS.md) | Kebijakan laporan Telegram dan alert signifikan |
| [`UI_GUIDE.md`](UI_GUIDE.md) | Panduan UI desktop/mobile dan fitur peta |
| [`runbooks/`](runbooks/) | Runbook khusus pipeline/model |
| [`plans/`](plans/) | Catatan rencana dan follow-up ilmiah |

## Untuk Pembaca Baru

Mulai dari urutan ini:

1. [`../README.md`](../README.md)
2. [`PROBABILITY_GUIDE.md`](PROBABILITY_GUIDE.md)
3. [`UI_GUIDE.md`](UI_GUIDE.md)
4. [`ARCHITECTURE.md`](ARCHITECTURE.md)
5. [`MAINTENANCE.md`](MAINTENANCE.md)

## Prinsip Dokumentasi

- README menjelaskan produk dan quickstart.
- `docs/` menjelaskan detail teknis.
- `.env.example` menjadi sumber template environment variable.
- Jangan commit secret/token/chat id asli.
