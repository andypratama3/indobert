# Alur Lengkap Pipeline IndoBERT ABSA

Dokumen ini merangkum seluruh alur kerja aplikasi, dari pengumpulan data sampai
RCA, lengkap dengan skrip yang dijalankan, file input/output, dan **pemetaan
ke bab skripsi** agar bisa dicross-check.

Disusun dari kode yang benar-benar ada di repository, bukan dari ingatan.

---

## 1. Ringkasan satu kalimat

> Scraper mengambil komentar TikTok → dibersihkan menjadi korpus 1.031 komentar
> → 800 komentar dianotasi manual menjadi 8 kelas → IndoBERT di-*fine-tune* dengan
> **10-Fold Cross Validation** → seluruh 1.031 komentar diprediksi **out-of-fold**
> (tanpa *data leakage*) → distribusinya dianalisis, 950 komentar negatif
> dikelompokkan menjadi kategori masalah per aspek lewat *content analysis*,
> lalu akar masalahnya ditelusuri dengan RCA 5-Whys.

---

## 2. Arsitektur tingkat tinggi

```
┌─────────────────────────────────────────────────────────────────────┐
│  SUMBER DATA                                                        │
├─────────────────────────────────────────────────────────────────────┤
│  TikTok scrape (Selenium)                                           │
│      data/raw/tiktok_comments_mobil_dinas_kaltim.csv    1.081 baris │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┴─────────────────┐
              ▼                                  ▼
┌──────────────────────────┐      ┌───────────────────────────────────┐
│  STAGE 1  KORPUS         │      │  STAGE 2 + 4  DATA LATIH          │
│  preprocessing.preprocess│      │  anotasi manual (3 lewat)         │
│                          │      │  data/annotation/dataset_train.csv│
│  → comments_clean.csv    │      │  → 800 baris, 8 kelas             │
│    1.031 baris           │      └────────────────┬──────────────────┘
│                          │                       │
│  kolom:                  │      ┌────────────────▼──────────────────┐
│   clean_text  ← model    │      │  STAGE 3+5  FINE-TUNING           │
│   content_text ← keputus │      │  preprocessing.preprocess_absa    │
└──────────┬───────────────┘      │  indobert.train_cv                │
           │                      │                                   │
           │                      │  StratifiedKFold(10, shuffle, 42) │
           │                      │  → models/.../fold_1 .. fold_10   │
           │                      │  → data/results/cv/*              │
           │                      └────────────────┬──────────────────┘
           │                                       │
           └───────────────────┬───────────────────┘
                               ▼
             ┌─────────────────────────────────────┐
             │  STAGE 6  PREDIKSI OUT-OF-FOLD      │
             │  indobert.predict_absa_oof          │
             │                                     │
             │  673 baris label   → oof_fold_N      │
             │  358 baris tanpa   → ensemble vote  │
             │  ─────────────────────────────────  │
             │  1.031 baris  indobert_absa_result_ │
             │              oof.csv   ← SUMBER    │
             │                         TUNGGAL    │
             └─────────────────┬───────────────────┘
                               │
        ┌──────────┬───────────┼───────────┬──────────────┐
        ▼          ▼           ▼           ▼              ▼
   STAGE 7-8  STAGE 9-12   STAGE 13   STAGE 14-15   STAGE 16-18
   Tematik     Identifikasi RCA 5-    Audit        Visualisasi
               masalah (4      Whys   provenance   (3 keluarga
               skrip, per      +      + imbalance   gambar)
               aspek)          rekom.
```

---

## 3. Peta 19 tahap

`python run_all.py` menjalankan tahap ini secara berurutan. Tahap 5
(*training*) **tidak** dijalankan secara default karena butuh berjam-jam dan
~80 GB.

| # | Kunci | Nama | Skrip | Input utama | Output utama |
|---|---|---|---|---|---|
| 0 | `env` | Cek lingkungan | (manual) | `requirements.txt` | — |
| 1 | `corpus` | Bangun korpus | `preprocessing.preprocess` | raw scrape 1.081 | `processed/comments_clean.csv` 1.031 |
| 2 | `trainset` | Bersihkan data latih | `preprocessing.preprocess_absa` | `dataset_train.csv` 800 | `processed/dataset_train_clean.csv` 800 |
| 3 | `verify_train` | Verifikasi integritas | (manual) | dataset_train_clean | **gerbang keras** — hentikan run |
| 4 | `annotate` | Anotasi manual | (manual, manusia) | komentar sampel | `annotation/dataset_train.csv` |
| 5 | `train` | *Fine-tune* 10-fold | `indobert.train_cv` | dataset_train_clean | `models/.../fold_1..10`, `results/cv/*` |
| 6 | `predict` | Prediksi OOF | `indobert.predict_absa_oof` | komentar_clean + 10 model | **`indobert_absa_result_oof.csv`** |
| 7 | `thematic` | *Thematic coding* | `thematic_coding` | hasil prediksi OOF | `thematic_coding_result.csv`, `thematic_summary.csv` |
| 8 | `thematic_plot` | Grafik tematik | `plot_thematic_distribution` | `thematic_summary.csv` | `gambar_5_10_distribusi_tema.png` |
| 9 | `issue_akuntabilitas` | Identifikasi masalah | `identifikasi_masalah_akuntabilitas` | hasil prediksi OOF | `bukti_*` + `rekap_masalah_akuntabilitas.csv` |
| 10 | `issue_efektivitas` | Identifikasi masalah | `identifikasi_masalah_efektivitas_efisiensi` | hasil prediksi OOF | `rekap_masalah_efektivitas_efisiensi.csv` |
| 11 | `issue_responsivitas` | Identifikasi masalah | `identifikasi_masalah_responsivitas` | hasil prediksi OOF | `rekap_masalah_responsivitas.csv` |
| 12 | `issue_transparansi` | Identifikasi masalah | `identifikasi_masalah_transparansi` | hasil prediksi OOF | `rekap_masalah_transparansi.csv` |
| 13 | `rca` | RCA + rekomendasi | `analysis.analysis_pipeline` | hasil prediksi OOF | `identifikasi_permasalahan/*` |
| 14 | `provenance` | Audit provenance & bias | `analysis.provenance_audit` | dataset_train + raw | `provenance_*.csv` |
| 15 | `imbalance` | Metrik per kelas | `analysis.class_imbalance_report` | `cv_per_fold_metrics` + `confusion_matrix_fold_1` | `per_class_metrics.csv`, `fold_class_coverage.csv` |
| 16 | `viz` | Gambar dari **prediksi** | `visualization.visualize_absa` | hasil prediksi OOF | `distribusi_aspek.png`, `distribusi_sentimen.png`, `aspect_sentiment_distribution.png` |
| 17 | `viz_annotation` | Gambar dari **annotasi** | `visualization.visualize` | `dataset_train.csv` | `*_annotasi.png`, `jumlah_data.png`, `distribusi_final_label.png` |
| 18 | `viz_cm` | *Confusion matrix* | `visualization.plot_confusion_matrix` | `data/results/cv` | `confusion_matrix_fold_1..10.png` |

> **Tahap 3 adalah gerbang keras.** Kalau `dataset_train_clean.csv` tidak lagi
> mereproduksi data latih secara *byte-for-byte*, run dihentikan supaya model
> tidak menerima teks asing. Ini pembalik dari BUG-05 / BUG-12.

---

## 4. Tahap-tahap yang menjelaskan

### Fase A — Persiapan data

**Tahap 1 · Korpus** (`preprocessing.preprocess`)

Baca `data/raw/tiktok_comments_mobil_dinas_kaltim.csv` (1.081 baris).

1. Pangkas *stamp* tanggal TikTok (`4-25`) — harus **sebelum** filter panjang,
   supaya komentar pendek bermakna ("setuju", "parah") tidak terbuang (BUG-02).
2. Filter `is_analysable()` — membuang 50 baris (emoji saja / *stamp* saja).
3. Deduplikasi berdasar **`(username, clean_text)`**, bukan `clean_text` saja.
   Versi lama menggabungkan pengguna berbeda dan membuang 9 opini sah (BUG-01).
4. `normalize()` menghasilkan `clean_text`:

   - huruf kecil
   - hapus URL, mention, hashtag
   - `8,5` / `8.5` → `8 5` (rupiah tetap terbaca)
   - hapus tanda baca (kecuali di dalam angka)
   - normalisasi spasi

5. `normalize_for_gate()` menghasilkan `content_text` (keputusan penyertaan).

> **PENTING — `expand_slang` harus `False`.** Model di-*fine-tune* pada teks
> **tanpa** tahap perluasan slang. Mengaktifkannya saat inferensi menciptakan
> *train/inference mismatch* (BUG-03). `clean_text` harus persis byte-identik
> dengan kolom yang dipakai saat pelatihan.

**Tahap 2 · Data latih** (`preprocessing.preprocess_absa`)
Idempoten; memastikan `normalize(original_text) == clean_text` untuk 800/800 baris.

**Tahap 3 · Gerbang verifikasi** — memeriksa 800/800. Bila gagal, run berhenti.

**Tahap 4 · Anotasi (manusia)**
Tiga lewat per komentar sesuai `aspect/aspect_guideline.md`:

| Kelompok kolom | Arti |
|---|---|
| `annotator_aspect`, `annotator_sentiment` | pemberi label 1 |
| `ai_aspect`, `ai_sentiment` | lewat bantu-ai, mengikuti struktur data anotasi |
| `researcher_aspect`, `researcher_sentiment` | adjudikasi peneliti |
| `final_aspect`, `final_sentiment`, `final_label` | hasil rekonsiliasi |

**4 aspek + kategori "Lainnya"** untuk di luar cakupan:
Transparansi · Akuntabilitas · Efektivitas dan Efisiensi · Responsivitas.

---

### Fase B — Model

**Tahap 5 · Fine-tuning 10-fold** (`indobert.train_cv`) — *opsional, lambat*

| Parameter | Nilai |
|---|---|
| Model dasar | `indobenchmark/indobert-base-p1` |
| Pembagian | `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` |
| Epoch | 7 |
| *Learning rate* | `2e-5` |
| *Batch size* | 8 (train & eval) |
| *Weight decay* | 0,01 |
| *Eval/save strategy* | per epoch |
| `load_best_model_at_end` | `True`, dengan `metric_for_best_model="f1"` |
| `save_total_limit` | 1 |

Menghasilkan `models/indobert_aspect_sentiment_cv/fold_1..fold_10` (~80 GB,
*gitignored*) dan metrik di `data/results/cv/`.

**Tahap 6 · Prediksi out-of-fold** (`indobert.predict_absa_oof`) — *tahap kunci*

```
Setiap baris korpus mendapat model yang TIDAK PERNAH melatih baris itu.

1. Reproduksi split yang sama: StratifiedKFold(10, shuffle, random_state=42)
   ├── 673 komentar punya label  → diprediksi oleh model fold yang menahan
   │                                baris itu (holdout) → prediction_source = oof_fold_N
   └── 358 komentar tanpa label  → diprediksi oleh MAJORITY VOTE dari
                                    ke-10 model → prediction_source = ensemble_10fold_unseen
```

Kolom `prediction_source` dan `votes` membuat setiap baris bisa diaudit.

> **Jangan pakai `indobert/predict_absa.py`.** Skrip lama itu memprediksi seluruh
> korpus dengan satu model *fold* yang sudah menghafal dua pertiga korpus
> (BUG-01). Skrip itu kini berupa stub keras yang berhenti dengan *exit 2*;
> implementasi aslinya masih ada di riwayat git (`876dc00`).

---

### Fase C — Analisis

**Tahap 7–8 · Koding tematik** (`thematic_coding`, `plot_thematic_distribution`)
Kamus kata kunci per aspek, hanya pada komentar negatif.

**Tahap 9–12 · Identifikasi masalah per aspek** (4 skrip independen, boleh paralel)
Untuk **tiap aspek**:

- `rekap_masalah_<aspek>.csv` — jumlah & persentase kategori masalah
- `bukti_komentar_<aspek>.csv` — bukti per baris

Logika: kamus kata kunci (`CATEGORY_KEYWORDS`) hasil *open coding*, dicocokkan
berurutan dengan *regular expression*. Bila tidak ada yang cocok, komentar jatuh
ke `FALLBACK_LABEL` ("tanpa keluhan spesifik"), sehingga cakupan selalu 100%.
Persentase dihitung dengan metode *largest remainder* agar total tepat 100,0%.

**Tahap 13 · RCA** (`analysis.analysis_pipeline`)
Distribusi aspek×sentimen, sampel kualitatif komentar negatif, tabel 5-Whys, dan
rekomendasi.

> `RCA_DATA` di dalam `analysis/analysis_pipeline.py` **ditulis oleh peneliti**,
> bukan diturunkan dari data. Bila distribusi berubah, kamus itu harus disesuaikan.

**Tahap 14 · Audit provenance** (`analysis.provenance_audit`)
Mengkuantifikasi **116 baris anotasi** yang tidak ada di korpus mentah (BUG-07).

**Tahap 15 · Ketidakseimbangan & metrik per kelas** (`analysis.class_imbalance_report`)
`per_class_metrics.csv` — di sinilah terlihat **tiga kelas positif F1 = 0,000**
(BUG-08).

**Tahap 16–18 · Visualisasi** — tiga keluarga gambar, **jangan tertukar**:

| Keluarga | Skrip | Sumber | Keluaran |
|---|---|---|---|
| **Prediksi** (hasil penelitian) | `visualize_absa` | `indobert_absa_result_oof.csv` | `distribusi_aspek.png`, `distribusi_sentimen.png`, `aspect_sentiment_distribution.png` |
| **Annotasi** (deskripsi data latih) | `visualize` | `dataset_train.csv` | `distribusi_aspek_annotasi.png`, `distribusi_sentimen_annotasi.png`, `distribusi_final_label.png`, `jumlah_data.png` |
| **Confusion matrix** | `plot_confusion_matrix` | `data/results/cv` | `confusion_matrix_fold_1..10.png` (+ `confusion_matrix_overall.png`, lihat catatan di bawah) |

> Akhiran `_annotasi` dibuat sengaja. Sebelumnya kedua skrip menulis
> `distribusi_aspek.png`, sehingga gambar anotasi bisa **menimpa** gambar prediksi
> tergantung urutan tahap.

> **Catatan `confusion_matrix_overall.png`.** Skrip `plot_confusion_matrix.py`
> hanya meng-*loop* fold 1–10 — ia **tidak** menghasilkan PNG agregat. File ini
> dibuat terpisah (langkah tunggal, di luar `run_all.py`) langsung dari
> `data/results/cv/confusion_matrix_overall.csv` memakai gaya yang sama
> (`figsize=(10,8)`, `cmap="Blues"`, `dpi=300`, `bbox_inches="tight"`).
> Angkanya konsisten: diagonal 621 dari 800 = **77,62%**, cocok dengan rata-rata
> akurasi 10-fold (77,63%). Untuk mereproduksi jalankan ulang langkah tersebut —
> tahap 18 saja tidak akan memulihkannya.

---

## 5. Peta silang: pipeline ↔ skripsi

Pemetaan ini disusun dengan membaca urutan *heading* + caption di dalam
`_REVISI.docx`, bukan dari asumsi penomoran.

### Bab 4 — Pengolahan Data dan Implementasi Metode

| Subbab | Tahap pipeline | Objek di skripsi |
|---|---|---|
| 4.1 Pengumpulan Data | 0, 1 | Tabel 4.1, Tabel 4.2, Algoritme 4.1 |
| 4.2 Pelabelan Data | 4 | Tabel 4.3, Tabel 4.4, Tabel 4.5 |
| 4.3 Text Preprocessing | 1, 2 | Tabel 4.6, Algoritme 4.2 |
| 4.4 Fine-Tuning IndoBERT | 5 | Tabel 4.7, Algoritme 4.3 |
| 4.5 10-Fold Cross Validation | 5 | Tabel 4.8, Algoritme 4.4 |
| 4.6 Root Cause Analysis (RCA) | 13 | Tabel 4.9, Algoritme 4.5 |

> **Subbab "Prediksi Seluruh Komentar" dan "Identifikasi Permasalahan" sudah
> dihapus dari Bab 4** (ronde 4, permintaan penulis). Bab 4 kini 6 subbab, dan
> penomoran otomatis menggeser RCA dari 4.8 menjadi 4.6.
>
> - **Metode** tetap utuh di **Bab 3.5.3** (prediksi *out-of-fold*, inti koreksi
>   *leakage*) dan **Bab 3.5.4** (identifikasi dua tahap + *open coding*).
> - **Hasil** tetap di **Bab 5.6** dan **Bab 5.8**.
> - Akibatnya 4.5 ditutup dengan kalimat penghubung yang menunjuk ke 3.5.3/3.5.4,
>   dan rujukan silang lama dialihkan: `subbab 4.6 → 3.5.3`, `4.7 → 3.5.4` (metode)
>   atau `→ 5.8` / `5.7` (hasil), `Algoritme 4.6 → 4.5`.
> - Caption Algoritme berupa **teks biasa, bukan field `SEQ`**, jadi penomoran
>   ulangnya dilakukan manual di dalam tabel.

### Bab 5 — Hasil Penelitian dan Pembahasan

| Subbab | Tahap pipeline | Objek di skripsi |
|---|---|---|
| 5.1 Hasil Pengumpulan Data | 1 | Tabel 5.1, Gambar 5.1 |
| 5.2 Hasil Pelabelan Data | 4 | Tabel 5.2–5.4, Gambar 5.2–5.4 |
| 5.3 Hasil Text Preprocessing | 1 | Tabel 5.5 |
| 5.4 Hasil Fine-Tuning IndoBERT | 5 | Tabel 5.6, Tabel 5.7 |
| 5.5 Hasil 10-Fold Cross Validation | 5, 15, 18 | Tabel 5.8, Tabel 5.9, Gambar 5.5 |
| 5.6 Hasil Prediksi Seluruh Komentar | 6 | — |
| 5.7.1 Distribusi Aspek Hasil Prediksi | 6, 16 | Tabel 5.10, Gambar 5.6 |
| 5.7.2 Distribusi Sentimen Hasil Prediksi | 6, 16 | Tabel 5.11, Gambar 5.7 |
| 5.7.3 Distribusi Aspek dan Sentimen | 6, 16 | Tabel 5.12, Gambar 5.8 |
| 5.8.1 Distribusi Sentimen Negatif per Aspek | 9–12 | Tabel 5.13, Gambar 5.9 |
| 5.8.2 Telaah Kualitatif Komentar Negatif | 13 | Tabel 5.14 |
| 5.8.3 Analisis Isi (Content Analysis) | 9–12 | Tabel 5.15–5.18 |
| 5.9.1–5.9.4 Hasil RCA per aspek | 13 | Tabel 5.19–5.22 |
| 5.9.5 Kesimpulan Hasil RCA | 13 | — |
| Bab 3 — keterbatasan metodologi | 14, 15 | `provenance_*.csv`, `per_class_metrics.csv` |

### Tahap yang TIDAK dikutip di skripsi

| Tahap | Output | Status |
|---|---|---|
| 7, 8 — koding tematik | `thematic_coding_result.csv`, `thematic_summary.csv`, `gambar_5_10_distribusi_tema.png` | ❌ **Tidak muncul di skripsi.** Tidak ada subbab tematik, tidak ada caption `Gambar 5.10`. Sebagai gantinya skripsi memakai *content analysis* (tahap 9–12) di 5.8.3. |
| 17 — gambar anotasi | `*_annotasi.png` | ⚠️ Hanya dipakai bila membahas distribusi **data latih** — hati-hati jangan disebut sebagai hasil prediksi. |

---

## 6. Angka kunci untuk cross-check

### 6.1 Aliran data

```
1.081   raw scrape (TikTok)
  -50   terlalu pendek / emoji saja
─────
1.031   korpus bersih            → comments_clean.csv
                                    ▲
  800   dianotasi manual          │ dataset_train.csv
                                    │
1.031   hasil prediksi            → indobert_absa_result_oof.csv
        ├ 673  oof_fold_*          (punya label)
        └ 358  ensemble_*          (tanpa label, majority vote 10 fold)
```

> **Catatan penting:** 684 baris anotasi dapat ditemukan di korpus, 116 tidak
> dapat dipulihkan. Dari 684 itu, **673 masuk hitungan OOF** (6 baris runtuh saat
> *join*). Yang boleh dikutip: **673 / 358**, bukan 684.

### 6.2 8 kelas distribusi hasil prediksi (1.031 komentar)

| Aspek | Negatif | Positif | Total |
|---|---:|---:|---:|
| Akuntabilitas | 559 (87,89%) | 77 (12,11%) | 636 (61,69%) |
| Efektivitas dan Efisiensi | 295 (98,66%) | 4 (1,34%) | 299 (29,00%) |
| Responsivitas | 64 (100,00%) | 0 (0,00%) | 64 (6,21%) |
| Transparansi | 32 (100,00%) | 0 (0,00%) | 32 (3,10%) |
| **Total** | **950 (92,14%)** | **81 (7,86%)** | **1.031 (100%)** |

Distribusi sentimen: **Negatif 950 (92,14%) · Positif 81 (7,86%)**

### 6.3 8 kelas distribusi data latih (800 komentar)

| Label final | Jumlah |
|---|---:|
| Akuntabilitas_Negatif | 305 |
| Efektivitas & Efisiensi_Negatif | 251 |
| Responsivitas_Negatif | 92 |
| Transparansi_Negatif | 69 |
| Akuntabilitas_Positif | 58 |
| Efektivitas & Efisiensi_Positif | 12 |
| Responsivitas_Positif | 7 |
| Transparansi_Positif | 6 |
| **Total** | **800** |

### 6.4 Evaluasi 10-Fold Cross Validation

**Rata-rata 10 fold** (`cv_summary_metrics.csv`) → Tabel 5.9

| Metrik | Rata-rata (%) |
|---|---:|
| Accuracy | 77,63 |
| Precision | 76,36 |
| Recall | 77,63 |
| F1-Score | 76,09 |

**Per fold** (`cv_per_fold_metrics.csv`) → Tabel 5.8

| Fold | Accuracy | Precision | Recall | F1-Score |
|---:|---:|---:|---:|---:|
| 1 | 77,50 | 79,11 | 77,50 | 77,48 |
| 2 | 82,50 | 82,78 | 82,50 | **81,04** |
| 3 | 77,50 | 79,15 | 77,50 | 76,78 |
| 4 | 77,50 | 74,81 | 77,50 | 75,81 |
| 5 | 78,75 | 76,56 | 78,75 | 77,28 |
| 6 | 82,50 | 81,38 | 82,50 | 79,84 |
| 7 | 72,50 | 71,29 | 72,50 | 70,94 |
| 8 | 78,75 | 74,94 | 78,75 | 76,54 |
| 9 | 78,75 | 76,81 | 78,75 | 77,03 |
| 10 | 70,00 | 66,76 | 70,00 | 68,11 |

> **F1 tertinggi ada di Fold 2 (81,04%), bukan Fold 6.** Fold 2 dan Fold 6 sama-sama
> memegang *Accuracy* tertinggi (82,50%) — tidak ada satu *fold* yang unggul
> konsisten di semua metrik (Fold 6 unggul di *accuracy* + *precision*, Fold 2 di F1).
> Metrik "model terbaik" sudah diganti menjadi **out-of-fold** karena memilih satu
> *fold* lalu memakainya untuk prediksi akan menimbulkan *data leakage*.

---

## 7. Berkas yang boleh dan tidak boleh dikutip

### ✅ Sumber kebenaran

| Berkas | Isi |
|---|---|
| `data/results/indobert_absa_result_oof.csv` | **Satu-satunya** hasil prediksi. 1.031 baris, bebas *leakage*. |
| `data/results/cv/cv_per_fold_metrics.csv` | Metrik per fold |
| `data/results/cv/cv_summary_metrics.csv` | Rata-rata 10 fold |
| `data/results/cv/per_class_metrics.csv` | Metrik **per kelas** — baca ini, bukan rata-rata tertimbang |
| `data/results/cv/confusion_matrix_overall.csv` | Jumlah matriks ke-10 fold |
| `data/results/rekap_masalah_*.csv` | Kategori masalah per aspek |
| `data/results/provenance_audit.csv` | Audit 116 baris tak terlacak |

### ❌ Jangan dikutip

| Berkas | Alasan |
|---|---|
| `data/results/indobert_absa_result.csv` | Bocor *leakage*. Audit saja. |
| `data/results/indobert_absa_result_leaky.csv` | Salinan cadangan dari yang di atas |
| `data/results/confusion_matrix.csv` | **Yatim** — tak ditulis/dibaca tahap mana pun |
| `data/results/indobert_aspect_sentiment_evaluation.csv` | **Yatim** — sisa skrip pelatihan lama |

Dua berkas yatim itu dihasilkan oleh `indobert/train_aspect.py` dan
`train_indobert.py`, yang membutuhkan `aspect_annotation_final_clean.csv` yang
tidak ada di repository (BUG-11).

---

## 8. Keterbatasan yang wajib terungkap

Bagian ini harus muncul sebagai keterbatasan di **bab metodologi** dan
diulang di **caveats bab hasil**.

| # | Keterbatasan | Angka pendukung |
|---|---|---|
| 1 | **Tiga dari delapan kelas punya F1 = 0,000 dan TP = 0** di metrik per-fold: `efektivitas_efisiensi_positif`, `responsivitas_positif`, `transparansi_positif`. Dari 10 fold, Responsivitas-positif zéro di 3 fold dan Transparansi-positif zéro di 4 fold. | `per_class_metrics.csv` |
| 1b | **Hasil prediksi sangat timpang.** Dari 81 komentar positif: 77 Akuntabilitas + 4 EfE + 0 Responsivitas + 0 Transparansi. Artinya klasifikasi positif praktis hanya bekerja untuk satu aspek. | Tabel 5.12 |
| 2 | **Ketidakseimbangan 92,14% negatif.** | Tabel distribusi sentimen |
| 3 | **116 baris anotasi tak terlacak** dan tidak acak secara topikal. | χ² = 248,73, df = 3, p ≈ 1,2×10⁻⁵³, Cramér's V = 0,558 |
| 4 | **Bias 2,34×.** Responsivitas + Transparansi = 21,75% set latih vs 9,31% korpus. | `annotated_not_in_corpus.csv` |
| 5 | **RCA aspek kecil lemah.** Transparansi: kategori terbesar justru *fallback* "tanpa keluhan spesifik" = 14/32 (**43,7%**) — hampir separuh komentar aspek ini tidak punya keluhan terkategorikan. Responsivitas: 64 komentar dipecah jadi 7 kategori, dan 12 (18,7%) *fallback* + 5 (7,8%) noise = **26,5% tak informatif**. | `rekap_masalah_*.csv` |
| 5b | ***Content analysis* menyumbang 28,0% *fallback*.** Secara global "tanpa keluhan spesifik" = 266/950 komentar negatif (Akuntabilitas 229/559 = 41,0%; Transparansi 14/32 = 43,7%; Responsivitas 12/64 = 18,7%; EfE 11/295 = 3,7%). Angka TOTAL di tiap rekap **termasuk** blok ini, jadi persentase setiap kategori nyata ditekan oleh *fallback* — di Akuntabilitas, kategori terbesar yang sesungguhnya (114) cuma tampak 20,4%. | `rekap_masalah_*.csv` |
| 6 | **Selektor scraper TikTok tidak pernah divalidasi langsung.** | BUG-09 |
| 7 | **Dua batch scrape berbeda.** Korpus dan sampel anotasi berasal dari pengambilan yang tidak sama. | `data/annotation/README.md` |

### Catatan khusus: angka "100% negatif" adalah artefak model

Responsivitas (64/64) dan Transparansi (32/32) terlihat 100% negatif, tetapi
karena model **tidak pernah memprediksi satu pun kelas positif** untuk kedua
aspek itu, angka itu **bukan** bukti keseragaman pendapat publik. Ini harus
ditulis eksplisit supaya pembaca tidak salah tafsir.

---

## 9. Cara menjalankan

```powershell
cd D:\skripsi_indobert_sentiment

# Semua tahap kecuali training (~2,5 menit)
.\run_all.bat

# Kendali
python run_all.py --list              # daftar tahap
python run_all.py --include-train     # ikut training (jam, ~80 GB)
python run_all.py --only 6            # satu tahap
python run_all.py --from 6            # lanjut dari tahap 6
python run_all.py --skip 15 16        # lewati tahap
python run_all.py --keep-going        # jangan berhenti di kegagalan pertama
```

Satu per satu:

```powershell
.\.venv\Scripts\python.exe -m preprocessing.preprocess
.\.venv\Scripts\python.exe -m preprocessing.preprocess_absa
.\.venv\Scripts\python.exe -m indobert.train_cv                 # opsional
.\.venv\Scripts\python.exe -m indobert.predict_absa_oof
.\.venv\Scripts\python.exe -m thematic_coding
.\.venv\Scripts\python.exe -m plot_thematic_distribution
.\.venv\Scripts\python.exe -m identifikasi_masalah_akuntabilitas
.\.venv\Scripts\python.exe -m identifikasi_masalah_efektivitas_efisiensi
.\.venv\Scripts\python.exe -m identifikasi_masalah_responsivitas
.\.venv\Scripts\python.exe -m identifikasi_masalah_transparansi
.\.venv\Scripts\python.exe -m analysis.analysis_pipeline
.\.venv\Scripts\python.exe -m analysis.provenance_audit
.\.venv\Scripts\python.exe -m analysis.class_imbalance_report
.\.venv\Scripts\python.exe -m visualization.visualize_absa
.\.venv\Scripts\python.exe -m visualization.plot_confusion_matrix
```

---

## 10. Yang tidak bisa direproduksi dari repository ini

| Tahap | Status |
|---|---|
| Scrape mentah | **Tidak ada scraper TikTok yang di-commit.** `scraper/scraper.py` menargetkan Twitter/X (BUG-09). |
| Anotasi | Kerja manusia. Berkasnya ada, prosesnya tidak otomatis. |
| *Checkpoint* model | ~80 GB, *gitignore*. Tahap 5 harus dijalankan ulang oleh siapa pun yang meng-*clone*. |

---

## 11. Daftar isu yang perlu disesuaikan di skripsi

| # | Isu | Status |
|---|---|---|
| 1 | **Gambar 5.5 kontradiktif.** Caption "Confusion Matrix Fold 6" padahal paragraf bilang agregat 10 fold, dan `confusion_matrix_overall.png` tidak pernah ada. | ✅ **diperbaiki** — PNG agregat dibuat dari `cv/confusion_matrix_overall.csv` (621/800 = 77,62%), gambar tertanam diganti, caption → "Confusion Matrix Agregat 10 Fold", DAFTAR GAMBAR ikut ter-update. |
| 2 | **Tabel 4.9 belum 8 kelas** + desimal titik (`12.11%`). | ✅ **diperbaiki lalu dihapus** — dirombak jadi 10×5 (8 kelas + Total, desimal koma), tetapi pada ronde 3 tabel ini beserta Tabel 4.10 dihapus dari Bab 4 saat pemisahan *hasil*/*proses*. Setara isinya kini di Tabel 5.10–5.14. |
| 3 | **Subbab 4.7** — saran Pak Bayu: hapus, tapi dia sekaligus memberi perbaikan detail *untuk* isi 4.7. | ✅ **DIHAPUS** (ronde 4, permintaan penulis, bersama 4.6) — Bab 4 kini 6 subbab dan RCA jadi 4.6. Metode tetap di Bab 3.5.3/3.5.4, hasil tetap di Bab 5.6/5.8; seluruh rujukan silang dialihkan dan lolos audit silang (subbab 15, Tabel 59, Algoritme 9, Gambar 14, Bab 23 — nol putus). |
| 4 | **Paragraf imbalance** perlu ditopang literatur. Referensi tak ada di daftar pustaka. | ⚠️ **paragraf sudah ditulis, sitasi BELUM** — pakai angka sendiri (950/1.031 = 92,14%; kelas 54,22% s/d 0,00%) tanpa mengarang sitasi. Kalau Pak Bayu tetap minta rujukan, **kasih saya referensi aslinya**. |
| 5 | ~~Kalimat terpotong di "Tahap kedua dilakukan…"~~ | ❌ **bukan isu** — kalimatnya lengkap; info di laporan sebelumnya keliru. |
| 6 | **Keterangan kolom `AI Aspect` / `AI Sentiment`** belum dijelaskan. | ✅ **diperbaiki** — dijelaskan sebagai label dari AI sebagai salah satu dari tiga pemberi label, diperiksa terpisah sebelum *majority voting*. Typo `pustaka  / di jelaskan` ikut diperbaiki. |
| 7 | ~~5.8 menyebut "sampel" padahal semua 950 dikodekan~~ | ❌ **bukan isu** — 5.8.2 (telaah kualitatif) memang pakai sampel acak untuk Tabel 5.14; 5.8.3 memang pakai seluruh 950. Keduanya sudah benar dan konsisten. |
| 8 | **Subbab 4.6 tanpa objek pendukung.** Satu-satunya subbab metode tanpa tabel/algoritme, padahal di situ inti koreksi *leakage*-nya. | ✅ **selesai lewat penghapusan** — 4.6 dan 4.7 dihapus, jadi isu ini kedaluwarsa. Bab 4.6 baru (RCA) punya Tabel 4.9 + Algoritme 4.5. Inti koreksi *leakage* tetap dijelaskan di Bab 3.5.3 dan dilaporkan di 5.6. Tidak perlu Algoritme baru dari `predict_absa_oof.py`. |
| 9 | **Coding manual hanya 4 dari 8 kelas** (hanya negatif). | ✅ **diperjelas** — pengodean eksplisit hanya pada komentar **bersentimen negatif** per aspek (Bab 3.5.4 dan 5.8.3); tabel distribusi 8 kelas kini sepenuhnya di Bab 5 (Tabel 5.10–5.14), sebab Bab 4 tidak lagi memuat tabel hasil. |
| 10 | **Tahap 7–8 (koding tematik) tidak dipakai** di skripsi. | ℹ️ info — tidak masalah, asal laporan konsisten menyebut metode yang benar-benar dilaporkan. |
| 11 | **12 dari 43 referensi (28%) tidak pernah disitasi** di seluruh badan teks — cacat yang lazim diperiksa penguji. | ✅ **diperbaiki (ronde 8)** — kesemuanya disitakan di titik paling cocok (Bab 1.1, Bab 2 ×4, RCA ×2), semua sumber memenuhi syarat penulis: dari Indonesia, terbit ≥ 2021. Kini **43/43 disitasi**; arah sebaliknya sudah bersih, dan `(Sinuraya, 2004)` dipastikan kutipan sekunder yang memang tidak boleh didaftar. |
| 12 | **Tabel RCA kosong 9 baris di ujung Bab 5** — label `Isu Kritis` / `Why 1–5` / `Akar Masalah` / `Rekomendasi` ada, tetapi kolom "Hasil Analisis" **seluruhnya kosong**, berborder `single` (terlihat), tanpa caption, diapit 20 paragraf kosong, tepat sebelum heading BAB 6. Ada juga di `work/original_backup.docx` (sisa *scaffold* penulis). | ✅ **dihapus (ronde 9)** — 21 node dibuang (tabel + caption `Keterangan` kosong + 20 paragraf kosong). Heading BAB 6 diberi **`pageBreakBefore` pada paragrafnya langsung**, bukan pada style: `Judul`/`Title` memang tidak punya `pageBreakBefore` dan tidak ada satu pun *page break* manual di dokumen, jadi sebelumnya Bab 6 hanya *"kebetulan"* mulai halaman baru karena 15 paragraf kosong itu. Gaya `Title` tidak disentuh → DAFTAR ISI aman. |
| 13 | **Spasi nyasar di heading + kapitalisasi `Tiktok`.** 6 heading berspasi **ekor** (`2.2`, `2.3`, `2.4`, `2.5`, `2.6` dan heading BAB 4), 1 heading berspasi **depan** (`Teknik Analisis Data` — satu-satunya dari 41+ heading), judul `2.8 … Komentar Tiktok` menulis `Tiktok` padahal 82 kemunculan lain sudah `TikTok`, plus 5 paragraf prosa/halaman judul ikut berspasi. | ✅ **diperbaiki (ronde 10)** — 13 paragraf disentuh, seluruhnya murni whitespace + kapitalisasi. Penomoran 4.1–4.6 aman karena `numPr` hidup di **style**, dan entri DAFTAR ISI `2.8` ikut ter-regenerate otomatis. **Pemeriksanya** (`postchk.py`) ikut dikoreksi: ia mengecualikan caption dalam *text box* sehingga DAFTAR GAMBAR 10 dianggap ber-caption 9. |
| 14 | **Dua cacat mikro teks.** (a) `Persamaan 2.4..` — titik ganda di akhir kalimat F1-score (Bab 2); (b) label `Responsivitas _Negatif` — satu-satunya dari 12 label `*_Negatif/*_Positif` yang berspasi sebelum `_`. | ✅ **diperbaiki (ronde 11)** — titik-ganda → satu titik; label → `Responsivitas_Negatif` konsisten dengan saudaranya. Kata 25.892 → 25.891 karena dua token bergabung setelah spasi dibuang. Sapuan luas ikut membuktikan **bersih**: tanpa artefak teknis, tanpa `w:ins`/`w:del`/komentar, tanpa kalimat kembar, `saya/kami` hanya muncul di kutipan komentar & judul berita (data/sumber, bukan suara penulis). |
| 15 | **Label `BAB 6 PENUTUP` di Sistematika Pembahasan (1.7)** — stempel lama; heading asli bab 6 kini `KESIMPULAN DAN SARAN` (p1690, gaya `Judul`), entri DAFTAR ISI pun `BAB 6 KESIMPULAN DAN SARAN`. Ini satu-satunya `PENUTUP` tersisa di seluruh dokumen; label bab 1–5 di bagian itu konsisten dengan heading aslinya. | ✅ **diperbaiki (ronde 12)** — run bold `PENUTUP` → `KESIMPULAN DAN SARAN` (run `BAB 6` + dua `w:tab` tak disentuh). Kata 25.891 → **25.893** (+2 kata). 99 halaman, 46 tabel, rasio field 3.00, xref 0 putus, sitasi 43/43, DAFTAR TABEL/GAMBAR 32=32 / 10=10. |

### Koreksi atas laporan saya sebelumnya ❌

Dua "isu" yang saya catat ternyata **salah** dan sengaja tidak diubah:

1. **"Kalimat terpotong"** — kalimat "Tahap kedua dilakukan…" lengkap dari awal.
   (Indeks paragrafnya sudah berganti-ganti tiap DAFTAR ISI diregenerasi, jadi
   rujukan isi lebih andal daripada nomor `pNN`.)
2. **"5.8 salah bilang sampel"** — justru benar; ada pemisahan 5.8.2 (sampel) vs 5.8.3 (seluruh populasi).

Satu "temuan" lain juga ternyata salah: **Tabel 5.7 (76,25%) bersumber sah** dari
`data/results/indobert_aspect_sentiment_evaluation.csv` (kolom `eval_accuracy` = `0.7625`).
RegEx pencarian saya sebelumnya mencari `76.25`, padahal isinya `0.7625`. Tabel itu juga
sudah diberi disclaimer benar di p481 ("satu proses pelatihan… belum dapat dijadikan acuan utama").

### Ronde 4 — subbab 4.6 dan 4.7 dihapus ✅

- Bab 4: 8 subbab → **6 subbab**; RCA naik dari 4.8 menjadi **4.6** (penomoran
  otomatis lewat style `Judul2`/`numId=5`, tidak ada yang diketik manual).
- Tabel Algoritme *Identifikasi* dihapus, Algoritme RCA **4.6 → 4.5** (teks biasa).
- Kalimat penghubung ditambahkan di **akhir 4.5** supaya alur 4.5 → 4.6 tidak
  melompat: prediksi + identifikasi dirujuk ke subbab 3.5.3 dan 3.5.4.
- Rujukan silang dialihkan — `4.6 → 3.5.3`, `4.7 → 3.5.4` (untuk *metode*) dan
  `→ 5.8` / `5.7` (untuk *hasil*); "yang diperoleh pada tahap sebelumnya" di dua
  paragraf RCA diganti karena tahap sebelumnya kini adalah 10-Fold CV.
- Dirapikan: "selanjutnya" kembar di akhir 4.5, pointer "subbab 5.8" yang diulang
  di lead-in Tabel 4.9.
- **Sistematika Pembahasan (Bab 1)** — deskripsi Bab 4 masih menyebut "analisis
  sentimen berbasis aspek" dan "identifikasi permasalahan" sebagai isi Bab 4.
  Diganti: `…, hingga penerapan Root Cause Analysis (RCA) untuk menghasilkan
  rekomendasi perbaikan tata kelola.` (italik *text preprocessing*,
  *10-Fold Cross Validation*, *Root Cause Analysis* tetap di run terpisah.)
- **Cacat yang saya buat sendiri di ronde awal**: paragraf "Catatan keterbatasan
  hasil RCA" (733 karakter) salah memakai gaya `Title` sehingga (a) tampil
  16pt bold rata tengah, (b) **ikut masuk DAFTAR ISI** sebagai entri palsu
  antara `BAB 6 KESIMPULAN DAN SARAN` dan `6.1 Kesimpulan`. Gaya diganti
  `Body Text`, typo `ber rests pada` → `dapat ditelusuri berkat`, dan
  `pada subbab ini` → `pada subbab 5.9` (paragraf ini ada di Bab 6).
  DAFTAR ISI bersih, tanpa entri terlalu panjang.
- **Bab 4 dibersihkan dari angka hasil** (pedoman Pak Bayu: bab 4 = proses,
  bab 5 = hasil). Paragraf 4.2 "Hasil pelabelan menghasilkan 800 komentar…"
  adalah satu-satunya paragraf Bab IV yang melaporkan angka hasil, dan angkanya
  tersalin persis di Bab V (`717`/`83` → p453; `305`/`6` → p456). Dipangkas
  menjadi kualitatif + penunjuk `subbab 5.2`, sementara persentase
  `89,6%`/`10,4%` **dipindah ke p453 (Bab 5)** sehingga tidak ada informasi
  yang hilang. Italik *fine-tuning* tetap di run terpisah. Verifikasi:
  `717`, `83 komentar`, `305` masing-masing turun tepat 1×, `89,6`/`10,4`
  pindah lokasi ke p453, seluruh angka kunci lain identik.
- **DAFTAR ISI tidak berjudul** — cacat bawaan dokumen (ada juga di
  `work/original_backup.docx`, jadi bukan akibat suntingan ronde sebelumnya).
  Paragraf tepat sebelum SDT yang memuat field TOC memakai style
  `DefaultHeading` (bold + caps + 16pt + rata tengah + `pageBreakBefore` +
  `outlineLvl=0`, identik dengan style heading `DAFTAR TABEL`) tetapi teksnya
  hanya satu spasi. Diisi `DAFTAR ISI`.
- **Skripsi tidak punya Abstrak sama sekali** — diverifikasi pada ketiga
  versi (non-REVISI, REVISI, `work/original_backup.docx`): literal
  `abstrak`/`kata kunci:` = 0, dan antara halaman judul dengan heading
  `DAFTAR ISI` hanya berisi paragraf kosong. Dituliskan **ABSTRAK (204
  kata) + Kata Kunci**, disisipkan tepat sebelum `DAFTAR ISI` (urutan baku
  judul → abstrak → daftar isi) memakai style `DefaultHeading`, sehingga
  mendapat halaman sendiri dan otomatis tercantum di daftar isi.
  **Semua angka diambil dari tabel, bukan dari ingatan**: Tabel 5.9
  (Accuracy 77,63 / Precision 76,36 / Recall 77,63 / F1 76,09), Tabel 5.10
  (Akuntabilitas 636 = 61,69%), Tabel 5.11 (950 = 92,14% negatif), serta
  1.081 → 800 → 1.031; seluruhnya diverifikasi aritmetika (636+299+64+32
  = 1.031; 950+81 = 1.031). **Sengaja TIDAK membuat** halaman Pengesahan /
  Pernyataan Keaslian — keduanya butuh nama dan tanda tangan pembimbing
  yang nyata.
- Hasil: **99 halaman** (98 + 1 halaman abstrak), 0 error field, 0 rujukan
  putus (subbab 16, Tabel 59, Algoritme 9, Gambar 14, Bab 24). Inventaris
  field XML = 194 `instrText` / 582 `fldChar` (rasio 3.00; +1 PAGEREF dari
  entri `ABSTRAK` yang baru). Awal daftar isi kini `ABSTRAK 2`,
  `DAFTAR ISI 3`, `DAFTAR TABEL 6`, `DAFTAR GAMBAR 8`, `BAB 1 PENDAHULUAN 10`.
  **Catatan alat**: counter `Fields.Count` dari Word COM (pernah melapor
  380 lalu 378) **tidak stabil** karena bergantung pada apakah TOC sudah
  diregenerasi ketika dokumen dibuka — jangan dipakai sebagai penanda
  regresi; inventaris `instrText`/`fldChar` di XML yang dipakai sebagai
  rujukan.
- Bug yang ditemukan dan diperbaiki saat pengerjaan: penghapusan tabel sempat
  salah sasaran karena caption RCA sudah lebih dulu diganti 4.6 → 4.5 sehingga
  ada dua tabel "Algoritme 4.5" dan loop mengambil kecocokan terakhir. Dipulihkan
  dari backup `work/pre_hapus_4647.docx`, seleksi diganti berdasar *judul* caption.

### Ronde 12 — label Sistematika Pembahasan Bab 6 disinkronkan ✅

- **Pemeriksaan konsistensi Bab 1 ↔ Bab 6** (permintaan "terus perbaiki sampai
  benar-benar baik") menemukan satu cacat nyata: pada bagian **Sistematika
  Pembahasan (1.7)** dokumen mendaftarkan bab 6 sebagai **`BAB 6 PENUTUP`**,
  padahal heading asli bab 6 adalah **`BAB 6 KESIMPULAN DAN SARAN`** (p1690,
  gaya `Judul` + `pageBreakBefore`), entri DAFTAR ISI pun `BAB 6 KESIMPULAN DAN
  SARAN`. Penelusuran `PENUTUP` ke seluruh dokumen: **hanya tersisa 1** — label
  di Sistematika ini. Label Sistematika bab 1–5 (dengan pemisah dua `w:tab`)
  sudah cocok dengan heading `Judul1` aslinya (`PENDAHULUAN`, `LANDASAN
  KEPUSTAKAAN`, `METODOLOGI PENELITIAN`, `PENGOLAHAN DATA DAN IMPLEMENTASI
  METODE`, `HASIL PENELITIAN DAN PEMBAHASAN`).
- **Catatan struktur heading bab**: bab 1–5 memakai gaya `Judul1` berisi judul
  saja (mis. `PENDAHULUAN`); awalan "BAB n" dirender oleh penomoran
  style/numbering — bukan teks. Bab 6 yang unik memakai gaya `Judul` dengan teks
  lengkap `BAB 6 KESIMPULAN DAN SARAN`. Di layar dan di DAFTAR ISI hasilnya
  konsisten, jadi mekanisme berbeda itu **bukan cacat** dan tidak disentuh.
- **Perbaikan**: hanya run bold `PENUTUP` → `KESIMPULAN DAN SARAN` pada paragraf
  Sistematika (run `BAB 6` dan dua `w:tab` tidak tersentuh). Deskripsi bab 6
  ("Bab ini berisi kesimpulan…saran…") memang sudah benar.
- Hasil: **99 halaman, 25.893 kata** (+2 kata), 1.750 paragraf, 46 tabel, 0 error
  field, inventaris **194 `instrText` / 582 `fldChar` (rasio 3.00)**, rujukan
  silang **0 putus**, sitasi **43/43**, DAFTAR TABEL/GAMBAR **32=32 / 10=10**,
  italic & `EE0000` tampak identik, selisih teks vs `HEAD` **persis 1 paragraf**.

### Ronde 11 — dua cacat mikro teks (titik ganda & label berspasi) ✅

- **Temuan lewat sapuan luas ronde 11** (pola `fold#`, sisa template, frasa kaku,
  singkatan alay `yg/dgn/utk/krn`, `Rp d(=d.`, kata ganti orang pertama, referensi
  `Persamaan`, `w:ins`/`w:del`, kalimat kembar, titik-koma/koma ganda, label
  `*_Negatif/*_Positif`). Hampir semuanya **nihil** atau *false positive* sah:
  - kutipan komentar mentah sering "tidak beraturan" (titik/emoji/`yg`) — itu **data**,
    bukan teks penulis; tabel aspek memang berisi label berulang karena itu hasil
    annotator/AI/researcher yang **setuju** (Tabel 4.3–4.4);
  - `dst`/`dll` di dua kalimat Bab 5 = singkatan lazim yang sah;
  - `saya`/`kami` hanya pada kutipan komentar dan sebuah judul berita di referensi —
    tidak ada suara orang pertama penulis di prosa;
  - `diatas/dibawah/diantara` tidak ada; `Persamaan 2.1–2.4` semuanya dirujuk;
  - **`w:ins=0`, `w:del=0`, komentar Word=0** — dokumen bersih dari jejak revisi.
- **Dua cacat nyata yang diperbaiki:**
  1. **`Persamaan 2.4..`** (Bab 2, definisi F1-score) — titik ganda di akhir kalimat;
     diedit hanya run polos terakhir (`' dihitung menggunakan Persamaan 2.4..'` → `'.'`),
     italic `F1-score`/`precision`/`recall` tidak tersentuh.
  2. **`Responsivitas _Negatif`** (Tabel 4.4, kolom Final Label) — satu-satunya dari 12
     label `*_Negatif/*_Positif` yang berspasi sebelum `_`; diedit menjadi
     `Responsivitas_Negatif` agar seragam dengan `Akuntabilitas_Negatif` dll.
- Hasil: **99 halaman, 25.891 kata** (−1 = dua token `Responsivitas _Negatif` melebur
  menjadi satu saat spasi dibuang), 1.750 paragraf, 46 tabel, 0 error field, inventaris
  **194 `instrText` / 582 `fldChar` (rasio 3.00)**, rujukan silang **0 putus**, sitasi
  **43/43**, 47 caption **0 yatim** (DAFTAR TABEL 32=32, DAFTAR GAMBAR 10=10),
  1.228 run italic tampak identik, 2 penanda `EE0000` identik, selisih teks vs `HEAD`
  **persis 2 paragraf**, urutan 1.595 teks lain identik.

### Ronde 10 — spasi nyasar di heading, kapitalisasi TikTok, pemeriksa caption diperbaiki ✅

- **13 paragraf dibersihkan — seluruhnya murni whitespace + kapitalisasi,**
  tidak ada satu pun perubahan substansi:

  | # | Sasaran | Sebelum → sesudah |
  |---|---|---|
  | 1–6 | heading spasi **ekor** (`2.2` NLP, `2.3` ABSA, `2.4` IndoBERT, `2.5` Fine-Tuning, `2.6` Hybrid Annotation, heading **BAB 4**) | `'…Sentimen '` → `'…Sentimen'` dll. |
  | 7 | heading `Teknik Analisis Data` — spasi **depan**, satu-satunya dari 41+ heading | `'  Teknik…'` → `'Teknik…'` |
  | 8 | judul `2.8` | `'…Komentar Tiktok'` → `'…Komentar TikTok'` |
  | 9–13 | `DEPARTEMEN SISTEM INFORMASI ` + 4 paragraf isi (flowchart, Algoritme 4.2, gambaran permasalahan, Tabel 5.15) | spasi ekor (dan depan bila ada) dibuang |

- **Mengapa aman.** Spasinya terbukti berupa run `w:t` literal, **bukan `w:tab`**,
  jadi menghapus run-nya tidak mengubah struktur paragraf. `pPr` keenam heading
  hanya berisi `pStyle` (tiga di antaranya plus `ind`), dan **`numPr` hidup di
  style** `Judul2`/`numId=5` — bukan di paragraf — sehingga penomoran 4.1–4.6
  sama sekali tidak tersentuh. Karena itu pula suntingan teks heading tidak
  berpengaruh ke penomoran.
- **DAFTAR ISI ikut ter-regenerate.** Satu-satunya perubahan isi di luar 13 edit
  adalah entri TOC `2.8` yang otomatis mengikuti (`Tiktok` → `TikTok`, nomor halaman
  30 tetap). Pada tingkat field, **hanya nama bookmark tersembunyi `_Toc*` yang
  di-*rebuild* Word** — setelah dinormalkan (`_Toc\d+` → `_TocN`), 194 `instrText`
  **identik**; jenis & jumlah tak berubah (`TOC` 6 / `PAGEREF` 145 / `SEQ` 43),
  rasio `fldChar` tetap **3.00**. Semua **145 `PAGEREF` punya pasangan
  `bookmarkStart`** (0 yatim) dan pasangan ID `bookmarkStart`/`bookmarkEnd`
  seimbang **185/185**.
- **Pemeriksanya yang dikoreksi, bukan dokumennya.** `postchk.py` semula
  mengecualikan paragraf di dalam *text box*, sehingga caption `Gambar 3.1`
  (memang sengaja di dalam kotak melayang — layout asli penulis) tak terhitung dan
  DAFTAR GAMBAR (10 entri) gagal disandingkan dengan caption 9. Kini caption dalam
  kotak ikut dihitung → **47 caption, 0 tak pernah dirujuk, 0 rujukan yatim**,
  DAFTAR TABEL **32 = 32**, DAFTAR GAMBAR **10 = 10**. Perbaikan kedua: pemeriksa
  teks semula ikut mengumpulkan kode field `w:instrText` sehingga melaporkan 19
  blok perubahan; setelah dibatasi ke `w:t` terlihat, hasilnya **14 blok = 13 edit
  + 1 entri TOC**, teks lain identik urutannya.
- **Yang sengaja tidak disentuh:** padding spasi URL pada 10 entri referensi
  (identik dengan `work/original_backup.docx` = format bawaan penulis), caption
  `Gambar 3.1` di dalam kotak, serta seluruh isi substantif dan angka.
- Hasil: **99 halaman, 25.892 kata** (tak berubah — spasi bukan kata), 1.750
  paragraf (di luar kotak teks), 46 tabel, 0 error field, inventaris **194
  `instrText` / 582 `fldChar` (rasio 3.00)**, rujukan silang **0 putus** (subbab 16,
  Tabel 59, Algoritme 9, Gambar 14, Bab 24), sitasi **43/43**, 5 Algoritme, Bab IV
  6 subbab. Run italic tampak identik (1.228 berhuruf — satu run yang hilang hanya
  berisi spasi murni), 2 penanda revisi `EE0000` identik, 10 entri References
  ber-padding identik dengan `HEAD`.

### Ronde 9 — audit caption↔rujukan, aritmetika tabel, tabel kosong dihapus ✅

- **Audit caption ↔ rujukan (dua arah) — bersih.** Dari 46 caption sejati
  (32 Tabel + 9 Gambar + 5 Algoritme, gaya `Keterangan`/`SourceCode`), **0 yang
  tak pernah dirujuk** di prosa. Sebaliknya, satu-satunya "rujukan tanpa caption"
  adalah `Gambar 3.1` — **bukan cacat**: captionnya memang sengaja ada *di dalam*
  kotak teks melayang (`Kotak Teks 1`), bukan paragraf biasa. Diverifikasi against
  `work/original_backup.docx`: strukturnya **identik** (11 `a:blip`, caption-in-textbox
  juga ada di versi asli), DAFTAR GAMBAR tetap memuatnya, dan prosa merujuknya 2×.
  Menyentuhnya justru berisiko tinggi, jadi dibiarkan.
  **Catatan alat**: caption memakai field `SEQ`, jadi teksnya `Tabel 4. 1`
  (ada spasi) sementara rujukan prosa ditulis `Tabel 4.1` — RegEx pencocokan
  wajib memakai `\d+\.\s*\d+`. Deteksi DAFTAR TABEL/GAMBAR juga harus dibatasi
  *next heading* apa pun (`Judul1/2/3`, `DefaultHeading`, `Title`), kalau tidak
  `DAFTAR GAMBAR` akan menelan sampai DAFTAR REFERENSI (1.413 "entri" palsu).
- **Audit aritmetika seluruh tabel distribusi — semua lolos.** Pembulatan memakai
  `ROUND_HALF_UP` seperti dokumen (rata-rata fold 77,625 → **77,63**; `round()`
  Python memakai *banker's rounding* sehingga menghasilkan 77,62 dan sempat
  dianggap selisih — itu bug pemeriksa, bukan dokumen):

  | Pemeriksaan | Hasil |
  |---|---|
  | Tabel 5.2 / 5.3 / 5.4 (data latih) | 363+263+99+75 = 717+83 = 305+251+92+69+58+12+7+6 = **800** ✓ |
  | Tabel 5.8 rata-rata 10 fold | **77,63 / 76,36 / 77,63 / 76,09** ✓ (identik Tabel 5.9) |
  | Tabel 5.10 aspek prediksi | 636+299+64+32 = **1.031**; 61,69+29,00+6,21+3,10 = **100,00%** ✓ |
  | Tabel 5.11 sentimen prediksi | 950+81 = **1.031**; 92,14+7,86 = **100,00%** ✓ |
  | Tabel 5.12 silang | kolom (950, 81, 1.031) ✓ dan tiap baris neg+pos = total ✓ |
  | Tabel 5.13 negatif per aspek | Σ 1.031 komentar, Σ 950 negatif; 559/636=87,89 · 295/299=98,66 · 64/64=100 · 32/32=100 ✓ |
  | Tabel 5.15–5.18 kategori masalah | Σ = 559 / 295 / 64 / 32 ✓, Σ persen = 100% tiap tabel, tiap baris cocok hitung ulang ✓ |
  | RCA vs kategori | 114/559 = 20,4% · 69/559 = 12,3% · 170/295 = 57,6% · 95/295 = 32,2% ✓ |

  Silang aspek↔sentimen (Tabel 5.12) juga konsisten: 559+295+64+32 = **950**
  negatif, 77+4+0+0 = **81** positif. Distribusi label latih Tabel 5.4
  (305+251+92+69 = **717** negatif; 58+12+7+6 = **83** positif) cocok persis
  dengan Tabel 5.3 (717/83) dan Tabel 5.2 per aspek (363/263/99/75 = 800).
- **Satu temuan nyata: tabel RCA kosong 9 baris.** Berborder `single` (terlihat),
  tanpa caption (`Keterangan`-nya pun kosong), diapit paragraf kosong, tepat sebelum
  BAB 6 — dan **ada juga di `work/original_backup.docx`**, jadi memang sisa
  *scaffold* penulis, bukan kerusakan suntingan. Dihapus bersama 20 paragraf
  kosong pengisi (21 node).
- **Paginasi Bab 6 dibuat deterministik.** Seluruh dokumen **tidak punya satu pun
  *page break* manual** (`w:br type="page"` = 0), dan `Judul`/`Title` **tidak**
  punya `pageBreakBefore` (hanya `Judul1` yang punya). Jadi Bab 6 selama ini cuma
  kebetulan di halaman 91 karena 15 paragraf kosongnya. Kini `pageBreakBefore`
  dipasang **langsung di paragraf heading** — mengubah *format langsung*, bukan
  style, sehingga penomoran/DAFTAR ISI tidak tersentuh.
- **Sitasi diverifikasi ulang secara independen**: `refcheck6.py` → **43/43
  disitasi, 0 tak pernah disitasi, 0 sitasi tanpa entri** (`Sinuraya 2004` tetap
  kutipan sekunder yang sah).
- Hasil: **99 halaman, 25.892 kata** (−18 = persis jumlah kata 10 label tabel
  yang dihapus), paragraf 1.790 → **1.752**, tabel 47 → **46**, 0 error field,
  inventaris field **194 `instrText` / 582 `fldChar` (rasio 3.00 — tak berubah)**
  (`TOC` 6 / `PAGEREF` 145 / `SEQ` 43), rujukan silang **0 putus** (subbab 16,
  Tabel 59, Algoritme 9, Gambar 14, Bab 24), DAFTAR ISI 108 entri 0 terlalu
  panjang, DAFTAR TABEL 32, DAFTAR GAMBAR 10, Bab IV 6 subbab, 5 Algoritme,
  1.229 run italic identik, 2 penanda revisi `EE0000` identik.
  Bandingan presisi ke `HEAD` memastikan **hanya** blok 10 label tabel
  (`Tahap`, `Hasil Analisis`, `Isu Kritis`, `Why 1–5`, `Akar Masalah`,
  `Rekomendasi`) yang hilang dari teks; 1.596 teks non-kosong lain identik
  urutannya.

### Ronde 8 — 12 referensi tak pernah disitasi ✅

- **Temuan lewat pemeriksaan dua arah** (sitasi in-text ↔ Daftar Referensi):
  dari 43 entri, **12 (28%) tidak pernah muncul sama sekali di badan teks**.
  Angka ini diambil dengan pencocokan ternormalisasi (hapus spasi/bacaan tanda
  baca) supaya "detik Kalimantan" vs "detikKalimantan" tidak dihitung lolos.
- **Arah sebaliknya bersih** — tidak ada sitasi yang tanpa entri. Satu-satunya
  kandidat `(Sinuraya, 2004)` ternyata **kutipan sekunder yang sah**:
  "Menurut Nurhidayat (2023), yang mengutip Simamungsong dan Sinuraya (2004)".
  Menurut APA hanya sumber primer (Nurhidayat) yang didaftarkan, jadi daftar
  memang tidak boleh memuat Simamungsong & Sinuraya.
- **Syarat penulis: sumber dari Indonesia, penelitian minimal tahun 2021.**
  Diverifikasi untuk keempat makalah: G-Tech Univ. Brawijaya (2024,
  `ejournal.uniramalang.ac.id`), Gudang Jurnal Multidisiplin Ilmu (2024, DOI
  `gjmi`), JINTEKS (2023), JAIC Polibatam (2024, `jurnal.polibatam.ac.id`) —
  semuanya jurnal Indonesia. Delapan sisanya media Indonesia terbit 2026. Pemeriksaan
  menyeluruh: **seluruh 43 referensi dokumen bertahun ≥ 2021**, tidak ada satu
  pun di bawahnya.
- **Titik sisipan** — semuanya menempelkan sumber yang **sudah ada di daftar**
  ke kalimat yang **sudah ada**; tidak ada satu pun sumber yang dikarang:

  | Sumber | Titik sisipan | Alasan kecocokan |
  |---|---|---|
  | Antara News, Kaltim Prov, Kaltimtoday, Katakaltim | Bab 1.1 "menjadi perhatian masyarakat" | kalimat ini sebelumnya **tidak bersumber sama sekali** |
  | Irawan dkk. 2024; Nugroho & Amrullah 2023 | Bab 2 Cross Validation | keduanya memakai *cross validation* untuk evaluasi model sentimen / K-NN |
  | Jabar dkk. 2024 | Bab 2 *Akuntabilitas* | judulnya persis "Akuntabilitas dan transparansi…" |
  | Sejati dkk. 2024 | Bab 2 *ABSA* | ABSA pada komentar akun Kemenkeu |
  | detikKalimantan; Sorot Mata | RCA Why 4 (Permendagri) | judulnya persis soal Permendagri sebagai dasar pengadaan |
  | BeritaSatu; Otomotif Sindonews | RCA Why 2 (harga) | keduanya memuat angka Rp8,5 miliar |

- **Teknik sisipan: run-level**, bukan menulis ulang paragraf — target ditunjuk
  per indeks run. Pada Bab 2 (p345) sisipan pertama diletakkan *di dalam* run
  polos sebelum run italic `machine learning`, dan sisipan kedua disisipkan di
  depan run `.` — sehingga **1.171 run italic tetap identik** dan sitasi tidak
  ikut miring. Penanda revisi `color=EE0000` identik (2), jumlah paragraf tetap
  1.790 (hanya run yang bertambah), seluruh angka kunci identik.
- Hasil: **43/43 referensi kini disitasi**, 99 halaman, 25.910 kata (+38 dari
  12 sitasi), 0 error field, 0 rujukan putus (subbab 16, Tabel 59, Algoritme 9,
  Gambar 14, Bab 24), inventaris field 194 `instrText` / 582 `fldChar`
  (rasio 3.00 — tak berubah karena memang tidak ada heading/field baru).

### Yang sudah diperbaiki di ronde sebelumnya ✅

- Semua "model terbaik / Fold 6" → **out-of-fold** (11 bagian, Bab 1–6).
- `998` → `1.031` (10×); distribusi Tabel 5.10–5.13, 5.15–5.18 diperbarui dari CSV.
- Batasan disampaikan jujur — termasuk bahwa *fallback* "tanpa keluhan spesifik"
  jadi **kategori terbesar di Transparansi (14/32 = 43,7%)**; globalnya 266/950 = 28,0%.
- DAFTAR TABEL hyperlink diperbaiki; *out-of-fold* selalu miring.
- `1081` → `1.081` di p393 (konsistensi pemisah ribuan).

---

*Terakhir disesuaikan pada ronde 12 (label Sistematika Pembahasan bab 6
`BAB 6 PENUTUP` → `BAB 6 KESIMPULAN DAN SARAN`, konsisten dgn heading & DAFTAR
ISI; struktur heading bab yang beda-mekanisme antara `Judul1` dan `Judul` dicatat
sebagai bukan cacat) dengan berkas `Mulai Revisi RCA BARU_..._REVISI.docx`
(99 halaman, 25.893 kata, 6 subbab di Bab 4, ABSTRAK 204 kata).*
*Commit pipeline tetap `703d9c4` — kode IndoBERT tidak diubah.*
