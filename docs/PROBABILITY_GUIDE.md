# Probability Guide: Cara Membaca Angka SeismicID

Panduan ini menjelaskan arti angka probabilitas yang ditampilkan SeismicID, apa yang bisa dan tidak bisa disimpulkan, serta batasan ilmiah sistem.

## Apa yang dihitung?

SeismicID menghitung **probabilitas setidaknya satu gempa ≥ magnitudo M dalam horizon H hari** untuk setiap grid cell 0.5° × 0.5° di Indonesia.

Contoh: *"Sulawesi Tengah - Palu, 8.3% probabilitas M≥5.0 dalam 30 hari"* artinya:

> Dari data historis dan fitur geologi saat ini, model memperkirakan peluang ~8.3% bahwa setidaknya satu gempa M≥5.0 akan terjadi di dalam grid cell Palu dalam 30 hari ke depan.

## Yang bisa disimpulkan

| ✅ BISA | Penjelasan |
|---|---|
| **Ranking relatif** | Cell A (8.3%) lebih berisiko dari Cell B (2.1%) — untuk horizon dan threshold yang SAMA. |
| **Pola spasial** | Area dekat zona subduksi/trench cenderung lebih tinggi — ini ekspektasi seismologi yang valid. |
| **Tren waktu** | Jika probabilitas naik dari 1% ke 5% di cell yang sama → aktivitas seismik terdeteksi meningkat. |
| **Perbandingan antar threshold** | M≥4.5 selalu ≥ M≥5.0 ≥ M≥5.5 ≥ M≥6.0 (dijamin oleh monotonicity enforcement). |
| **Perbandingan antar horizon** | 60 hari selalu ≥ 30 hari ≥ 14 hari ≥ 7 hari (dijamin oleh monotonicity enforcement). |

## Yang TIDAK bisa disimpulkan

| ❌ TIDAK BISA | Penjelasan |
|---|---|
| **"Aman" dari angka rendah** | Probabilitas 0.001% bukan berarti aman. Gempa besar bisa terjadi di mana saja kapan saja. Floor 1e-6 adalah batas numerik, bukan jaminan. |
| **"Pasti terjadi" dari angka tinggi** | Probabilitas 12% BUKAN prediksi deterministik. Artinya: dari 100 situasi serupa secara statistik, ~12 mengalami gempa. 88 tidak. |
| **Waktu pasti** | Model tidak memprediksi jam/tanggal spesifik. Hanya horizon (7/14/30/60 hari). |
| **Magnitudo pasti** | Model tidak memprediksi M=5.3 vs M=5.7. Hanya threshold (M≥4.5/5.0/5.5/6.0). |
| **Lokasi persis** | Grid cell 0.5°×0.5° ≈ 55km×55km. Gempa bisa terjadi di mana saja dalam cell itu. |
| **Membandingkan antar horizon/threshold berbeda** | 8% M≥5.0/30h vs 5% M≥6.0/7h → tidak bisa dibandingkan langsung. Pilih horizon + threshold yang sama. |

## Arti warna di peta

| Warna | Range | Arti |
|---|---|---|
| 🟢 Hijau | < 0.5% | Rendah — tapi **bukan berarti aman** |
| 🟡 Hijau-kuning | 0.5% – 1% | Rendah-menengah |
| 🟠 Kuning/amber | 1% – 3% | Sedang |
| 🟠 Oranye | 3% – 6% | Tinggi |
| 🔴 Merah | 6% – 12% | Sangat tinggi |
| 🔴 Merah tua | > 12% | Ekstrem — **bukan berarti pasti terjadi** |

**CATATAN**: Skala warna adalah **absolut**, bukan relatif terhadap distribusi. Sangat sensitif terhadap pilihan horizon dan threshold. Bandingkan ranking relatif antar area untuk horizon/threshold yang sama — jangan baca warna sebagai jaminan.

## Arti "data minim"

Jika cell menampilkan **"data minim"**:

- Cell tersebut sangat jarang/sama sekali tidak memiliki gempa historis M≥4.5.
- Probabilitas yang ditampilkan adalah floor numerik (1e-6), bukan estimasi statistik yang informatif.
- Ranking untuk cell ini tidak bisa diandalkan — pada dasarnya model "tidak tahu."

## Memilih horizon dan threshold

### Horizon (7 / 14 / 30 / 60 hari)
- **7 hari**: Sangat jangka pendek. Hanya area dengan aktivitas swarm/aftershock aktif yang signifikan.
- **14 hari**: Menengah pendek. Baik untuk monitoring pasca-gempa besar.
- **30 hari**: Seimbang. Default yang direkomendasikan untuk eksplorasi umum.
- **60 hari**: Jangka panjang. Lebih banyak noise, tapi menangkap sinyal seismisitas latar.

### Threshold (M≥4.5 / 5.0 / 5.5 / 6.0)
- **M≥4.5**: Gempa kecil-menengah. Sangat sering (~ribuan/tahun di Indonesia). Probabilitas akan tinggi di mana-mana.
- **M≥5.0**: Gempa menengah. Cukup sering. Baik untuk melihat variasi spasial.
- **M≥5.5**: Gempa menengah-besar. Lebih jarang, lebih informatif secara spasial.
- **M≥6.0**: Gempa besar. Sangat jarang (~puluhan/tahun di Indonesia). Probabilitas rendah, tapi high-impact.

## Batasan ilmiah fundamental

1. **Earthquake predictability is low**: Gempa besar pada dasarnya sulit diprediksi. Model statistik/ML hanya menangkap pola spasial-temporal dari katalog historis — bukan prekursor fisik.

2. **Small probabilities**: Probabilitas per cell kecil (biasanya < 5%). Ini normal — Indonesia luas, gempa besar jarang. Jangan misinterpretasi angka kecil sebagai "pasti aman."

3. **Retrospective ≠ prospective**: Metrik evaluasi pada data training cenderung overestimate performa. Evaluasi sesungguhnya harus prospektif (forecast archive vs kejadian aktual yang belum terjadi saat forecast dibuat).

4. **Catalog completeness**: Katalog USGS/BMKG tidak lengkap untuk magnitudo kecil (M < 4.5) dan periode historis tertentu. Ini mempengaruhi baseline rate.

5. **Fault/slab data quality**: Saat ini menggunakan approximation, bukan PUSGEN/Slab2.0 shapefile asli. Fitur geologi akan lebih akurat dengan data resmi.

6. **Demo seed mode**: Jika sistem berjalan dalam mode demo seed, probabilitas adalah placeholder berbasis fisika (jarak patahan, kedalaman slab) — **bukan output ML nyata**.

## Rekomendasi penggunaan

- ✅ Eksplorasi pola risiko spasial untuk edukasi dan penelitian.
- ✅ Monitoring internal untuk melihat tren aktivitas seismik.
- ✅ Sebagai salah satu input dalam perencanaan — bersama data BMKG, PUSGEN, dan sumber resmi lainnya.
- ❌ **JANGAN** digunakan untuk keputusan evakuasi atau peringatan dini.
- ❌ **JANGAN** digunakan sebagai satu-satunya dasar keputusan keselamatan.
- ❌ **JANGAN** menggantikan informasi dari BMKG atau otoritas resmi.

---

*Dokumen ini harus dibaca bersama **MODEL_CARD.md** untuk informasi teknis model.*