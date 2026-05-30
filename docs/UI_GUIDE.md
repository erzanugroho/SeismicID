# UI Guide

Dokumen ini menjelaskan fitur UI SeismicID untuk desktop dan mobile.

## Halaman Utama: Map

Path:

```text
/
```

Fungsi:

- menampilkan grid risiko gempa Indonesia
- memilih horizon waktu
- memilih threshold magnitudo
- melihat popup cell
- membuka top 10 risiko
- menyimpan Wilayah Saya
- menjalankan animasi risiko

## Kontrol Forecast

Filter utama:

```text
horizon: 7 / 14 / 30 / 60 hari
magnitude: M ≥ 4.5 / 5.0 / 5.5 / 6.0
```

Perubahan filter membaca cache forecast terbaru dari API.

## Popup Cell

Popup cell berisi:

```text
nama area
probabilitas + horizon + threshold
cell id + badge kualitas data
link detail
```

Contoh:

```text
Lepas Pantai DKI Jakarta - dekat Jakarta
0.18% probabilitas · 30 hari · M ≥ 4.5
Cell: Cm48_p1072 · data sedang
Lihat detail →
```

## Badge Kualitas Data

Badge membantu user membaca tingkat dukungan data/fitur untuk cell:

| Badge | Arti |
|---|---|
| data kuat | fitur geologi/tektonik lebih lengkap |
| data sedang | sebagian fitur geologi tersedia |
| data terbatas | fitur geologi terbatas, pakai ranking relatif |
| data minim | forecast belum kuat atau berada di probability floor |

## Wilayah Saya

Wilayah Saya menyimpan cell default di browser user.

Sumber default:

1. GPS browser saat pertama buka web, jika user mengizinkan.
2. Pilihan manual dari popup/map/search.
3. Input cari wilayah di panel Wilayah Saya.

Data disimpan di localStorage browser, bukan akun server.

## Mobile UI

Mobile dibuat map-first:

- map full screen
- FAB kiri bawah: `☰ kontrol peta`
- legend tetap di bawah
- panel Wilayah Saya tampil di atas FAB dan legend
- menu mobile berisi:
  - filter
  - top 10
  - wilayah saya
  - animasi risiko
  - refresh

## Animasi Risiko

Tombol `animasi risiko` menjalankan rotasi horizon:

```text
7d → 14d → 30d → 60d
```

Di mobile, timeline besar disembunyikan dan diganti chip kecil:

```text
animasi: 7d
```

## Top 10 Risiko

Top 10 bisa dilihat sebagai:

- cell
- cluster

Mode cluster menampilkan agregasi per subregion dan highlight semua cell anggota cluster di peta.

## Scheduler Admin

Path:

```text
/scheduler.html
```

Akses dilindungi password via `ADMIN_TOKEN`.

Fungsi:

- melihat scheduler runs
- trigger job manual
- trigger forecast recompute
- trigger Telegram daily report
- logout admin

## Disclaimer UI

UI selalu menampilkan bahwa SeismicID eksperimental dan bukan sistem peringatan dini resmi. User harus merujuk BMKG untuk informasi keselamatan.
