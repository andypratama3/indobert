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
  DAFTAR ISI bersih: 104 entri, tanpa entri terlalu panjang.
- Hasil: **98 halaman** (dari 99), 380 field, 0 error field, 0 rujukan putus.
- Bug yang ditemukan dan diperbaiki saat pengerjaan: penghapusan tabel sempat
  salah sasaran karena caption RCA sudah lebih dulu diganti 4.6 → 4.5 sehingga
  ada dua tabel "Algoritme 4.5" dan loop mengambil kecocokan terakhir. Dipulihkan
  dari backup `work/pre_hapus_4647.docx`, seleksi diganti berdasar *judul* caption.

### Yang sudah diperbaiki di ronde sebelumnya ✅

- Semua "model terbaik / Fold 6" → **out-of-fold** (11 bagian, Bab 1–6).
- `998` → `1.031` (10×); distribusi Tabel 5.10–5.13, 5.15–5.18 diperbarui dari CSV.
- Batasan disampaikan jujur — termasuk bahwa *fallback* "tanpa keluhan spesifik"
  jadi **kategori terbesar di Transparansi (14/32 = 43,7%)**; globalnya 266/950 = 28,0%.
- DAFTAR TABEL hyperlink diperbaiki; *out-of-fold* selalu miring.
- `1081` → `1.081` di p393 (konsistensi pemisah ribuan).

---

*Terakhir disesuaikan pada ronde 4 (penghapusan subbab 4.6/4.7) dengan berkas
`Mulai Revisi RCA BARU_..._REVISI.docx` (98 halaman, 6 subbab di Bab 4).*
*Commit pipeline tetap `703d9c4` — kode IndoBERT tidak diubah.*
