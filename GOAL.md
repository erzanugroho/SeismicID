# Goal Akhir: Gempa Forecast System

## Pernyataan Tujuan

Membangun sistem yang mampu memberikan **probabilitas gempa bumi di Indonesia secara otomatis, akurat, dan dapat diakses publik melalui browser** — bukan prediksi deterministik, melainkan *relative risk ranking* berbasis data ilmiah.

> **Contoh output:** *"Sulawesi Tengah - Palu, 12.4% probabilitas M≥5.0 dalam 30 hari"*

---

## Pertanyaan yang Harus Dijawab Sistem

> "Daerah mana di Indonesia yang paling berisiko gempa dalam 7, 14, 30, atau 60 hari ke depan?"

---

## Tiga Pilar Utama

| Pilar | Goal |
|---|---|
| **Ilmiah** | Ensemble ML (XGBoost + LightGBM + ETAS) + physics-informed features yang defensible secara seismologi |
| **Operasional** | Auto-update tiap 15 menit, forecast tiap 1 jam, retrain mingguan — berjalan tanpa intervensi manual |
| **Aksesibel** | 5-halaman browser UI yang dapat digunakan siapapun, bukan hanya peneliti |

---

## Kriteria Keberhasilan

- [ ] Sistem menghasilkan 16 prediksi independen per grid cell (4 horizon × 4 threshold magnitudo)
- [ ] Coverage seluruh Indonesia: ~3.000 grid cells 0.5°×0.5° dengan label provinsi
- [ ] Model terkalibrasi dan terevaluasi (Brier score, ROC, BSS)
- [ ] UI selalu memiliki data (3-tier fallback: ML ensemble → ETAS → demo seed)
- [ ] Scheduler berjalan otomatis tanpa operator
- [ ] 76+ unit tests passing, coverage ≥80%

---

## Batasan (Bukan Goal)

- ❌ Prediksi deterministik kapan/di mana persisnya gempa terjadi
- ❌ Sistem peringatan dini real-time (domain BMKG/TEWS)
- ✅ **Probabilistic hazard ranking** untuk perencanaan dan kewaspadaan

---

## Roadmap Milestone

### M1 — Foundation
- Grid Indonesia + label provinsi
- Ingestion USGS + BMKG
- Database schema + API dasar

### M2 — ML Core
- Feature engineering (~25 fitur per cell)
- Training XGBoost + LightGBM
- ETAS baseline + ensemble blending
- Kalibrasi model

### M3 — Operasional
- Scheduler auto-update + retrain
- 3-tier fallback forecast service
- Evaluasi & monitoring endpoint

### M4 — UI & Polish
- 5 halaman browser (Map, Detail, Events, Performa, Tentang)
- Docker deployment
- Dokumentasi lengkap

---

*Diperbarui: Mei 2026*
