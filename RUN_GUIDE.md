# Run Guide — IndoBERT ABSA Pipeline

Step-by-step, in execution order. Every command is run from the repository root.

All paths are relative, so `cd` to the project root first:

```powershell
cd D:\skripsi_indobert_sentiment
```

---

## Fastest path — one click

Double-click **`run_all.bat`**, or:

```powershell
.\run_all.bat
```

That runs all 18 stages except retraining and finishes in about **2.5 minutes**.
It rebuilds every file in `data/processed/` and `data/results/`, verifies each
stage actually produced its declared outputs, and stops with a non-zero exit code
if anything fails.

```powershell
python run_all.py --list           # show the stages
python run_all.py --include-train  # also retrain the 10 folds (hours, ~80 GB)
python run_all.py --only predict   # run one stage
python run_all.py --from 6         # resume from a stage
python run_all.py --skip 15 16     # skip stages
python run_all.py --keep-going     # do not stop on the first failure
python run_all.py --cleanup        # delete stale checkpoints first (destructive)
```

Stage 3 is a **hard gate**. If the training data no longer reproduces byte-for-byte
— the failure mode behind BUG-05 and BUG-12 — the run aborts rather than feeding
the models unfamiliar text.

The steps below explain what each stage does.

---

## Where the outputs land

Everything the pipeline produces is under `data/`. Nothing is written anywhere
else.

### `data/processed/` — intermediate datasets

| File | Rows | Meaning |
|---|---:|---|
| `comments_clean.csv` | 1031 | analysable corpus; `clean_text` is what the model receives |
| `dataset_train_clean.csv` | 800 | the fine-tuning data; must reproduce 800/800 byte-for-byte |

### `data/results/` — all analysis results

**Predictions**

| File | Meaning |
|---|---|
| `indobert_absa_result_oof.csv` | **the one to use.** 1031 rows, leakage-free, with `prediction_source` per row |
| `indobert_absa_result.csv` | contaminated original. Audit only, never cite |
| `indobert_absa_result_leaky.csv` | preserved copy of the above |

**Per-aspect findings** (`rekap_*` = category counts, `bukti_*` = row-level evidence)

`rekap_masalah_akuntabilitas.csv` · `rekap_masalah_efektivitas_efisiensi.csv` ·
`rekap_masalah_responsivitas.csv` · `rekap_masalah_transparansi.csv` and the four
matching `bukti_komentar_*.csv`

**Thematic analysis**

`thematic_coding_result.csv` (row level) · `thematic_summary.csv` (frequency per aspect)

**Model evaluation**

| File | Contents |
|---|---|
| `cv/cv_summary_metrics.csv` | mean and std across the 10 folds |
| `cv/per_class_metrics.csv` | **per-class precision/recall/F1 — read this, not the weighted average** |
| `cv/fold_class_coverage.csv` | which folds contain which classes |
| `cv/confusion_matrix_fold_*.csv` | per-fold confusion matrices |
| `cv/confusion_matrix_overall.csv` | all folds summed |

**Audits**

`provenance_audit.csv` · `provenance_missing_rows.csv`

### `data/results/identifikasi_permasalahan/` — RCA, for Chapter 5

| File | Use |
|---|---|
| `rca_rekomendasi.csv` | 5-Whys RCA and recommendations per aspect |
| `laporan_lengkap.csv` | the same, open in Excel |
| `sampel_komentar_negatif.csv` | qualitative sample, 8 comments per aspect |
| `distribusi_aspek_sentimen.csv` | aspect × sentiment counts |
| `distribusi_negatif_per_aspek.png` | summary figure |

### `data/results/visualization/` — figures

Two families, distinguishable by name. Do not mix them up.

**From model predictions** (`indobert_absa_result_oof.csv`) — these are the
results:

`distribusi_aspek.png` · `distribusi_sentimen.png` ·
`aspect_sentiment_distribution.png` · `gambar_5_10_distribusi_tema.png` ·
`confusion_matrix_fold_1..10.png`

**From the annotated sample** (`dataset_train.csv`) — these describe the training
data, not what the model found:

`distribusi_aspek_annotasi.png` · `distribusi_sentimen_annotasi.png` ·
`distribusi_final_label.png` · `jumlah_data.png`

The `_annotasi` suffix is deliberate. Both scripts previously wrote
`distribusi_aspek.png` and `distribusi_sentimen.png`, so the annotation figures
silently overwrote the prediction ones depending on stage order.

### Two files that are NOT regenerated

`data/results/confusion_matrix.csv` and
`data/results/indobert_aspect_sentiment_evaluation.csv` are leftovers from
superseded training scripts (`indobert/train_aspect.py`, `indobert/train_indobert.py`)
whose input file `data/processed/aspect_annotation_final_clean.csv` is not in the
repository. No current stage writes or reads them. They predate the leakage fix,
so do not cite them.

---

Python 3.13 is what this was developed against.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`accelerate` is required by `transformers.Trainer` (used by
`indobert/train_indobert.py`, `train_sentiment.py`, `train_aspect.py`, and
`train_cv.py`) and is now listed in `requirements.txt`, so the single
`pip install -r requirements.txt` above is enough.

All versions in `requirements.txt` are pinned to the ones this repo was
developed and measured with. On Linux with an NVIDIA GPU, install the CUDA
build of torch first so pip does not overwrite it:

```bash
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Verify:

```powershell
.\.venv\Scripts\python.exe -c "import torch, transformers, sklearn, pandas; print(torch.__version__, transformers.__version__)"
```

Models download from HuggingFace on first run (`indobenchmark/indobert-base-p1`,
~440 MB) and are cached. Subsequent runs are offline.

---

## Step 1 — Prepare the corpus

**Input** `data/raw/tiktok_comments_mobil_dinas_kaltim.csv` (1081 rows)
**Output** `data/processed/comments_clean.csv` (1031 rows)

```powershell
.\.venv\Scripts\python.exe -m preprocessing.preprocess
```

Expected:

```
Input data          : 1081
Dibuang (terlalu pendek) : 50
Dibuang (duplikat user)  : 0
Output data         : 1031
```

The 50 dropped rows are emoji-only reactions or a bare date stamp. The output
carries two text columns:

- `clean_text` — fed to the model. Byte-identical in format to the fine-tuning data.
- `content_text` — the inclusion decision, with the TikTok date stamp removed.

---

## Step 2 — Annotate (manual, not automated)

**Input** comments sampled from the corpus
**Output** `data/annotation/dataset_train.csv` (800 rows)

This stage is human work and cannot be scripted. Three passes per comment, per
`aspect/aspect_guideline.md`:

| Column group | Meaning |
|---|---|
| `annotator_aspect`, `annotator_sentiment` | human annotator 1 |
| `ai_aspect`, `ai_sentiment` | model-assisted pass |
| `researcher_aspect`, `researcher_sentiment` | adjudicated final |
| `final_aspect`, `final_sentiment`, `final_label` | reconciled result |

Four aspects, plus "Lainnya" for out-of-scope comments. Adjudication resolves
disagreement between the two human passes.

Helpers:

```powershell
.\.venv\Scripts\python.exe -m aspect.create_annotation_template
.\.venv\Scripts\python.exe -m preprocessing.create_additional_annotation
```

If this stage is skipped, steps 4 and 5 cannot run — the models require the
`final_label` column.

---

## Step 3 — Clean the training data

**Input** `data/annotation/dataset_train.csv`
**Output** `data/processed/dataset_train_clean.csv` (800 rows)

```powershell
.\.venv\Scripts\python.exe -m preprocessing.preprocess_absa
```

Sanity check that the cleaner matches what the models were trained on:

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; from preprocessing.text_cleaning import normalize; d=pd.read_csv('data/processed/dataset_train_clean.csv'); print((d['original_text'].map(normalize)==d['clean_text']).sum(), '/', len(d))"
```

Must print `800 / 800`. Anything less means the cleaner has drifted from the
training data and the models will receive unfamiliar input — see BUG-05.

---

## Step 4 — Fine-tune with 10-fold cross-validation

**Input** `data/processed/dataset_train_clean.csv`
**Output** `models/indobert_aspect_sentiment_cv/fold_1 … fold_10`, `data/results/cv/*`

```powershell
.\.venv\Scripts\python.exe -m indobert.train_cv
```

This is the long step — hours on a GPU. It writes ~80 GB to `models/`, which is
why that directory is gitignored.

Expected label distribution:

```
akuntabilitas_negatif            305
efektivitas_efisiensi_negatif    251
responsivitas_negatif             92
transparansi_negatif             69
akuntabilitas_positif             58
efektivitas_efisiensi_positif     12
responsivitas_positif              7
transparansi_positif               6
```

A `UserWarning` about classes smaller than `n_splits` is expected and is
discussed in BUG-08. It is not fatal.

Results land in `data/results/cv/cv_summary_metrics.csv`.

**Only needed if the annotation set changed.** If `dataset_train_clean.csv` is
unchanged, skip straight to step 5 and reuse the existing checkpoints.

---

## Step 5 — Predict (leakage-free)

**Input** `data/processed/comments_clean.csv` + the ten fold models
**Output** `data/results/indobert_absa_result_oof.csv` (1031 rows)

```powershell
.\.venv\Scripts\python.exe -m indobert.predict_absa_oof
```

Expected:

```
Total baris               : 1031
Dari OOF holdout         : 673
Dari ensemble unseen     : 358
```

Roughly 5–15 minutes. Loads each fold model once.

**Use this script, not `indobert/predict_absa.py`.** The older script scores the
whole corpus with a single fold model, which memorised two thirds of it. See
BUG-01.

Two provenance columns make the result auditable:

- `prediction_source` — `oof_fold_N` or `ensemble_10fold_unseen`
- `votes` — how many of the ten folds agreed

---

## Step 6 — Thematic coding

**Input** `data/results/indobert_absa_result_oof.csv`
**Output** `data/results/thematic_coding_result.csv`, `thematic_summary.csv`

```powershell
.\.venv\Scripts\python.exe -m thematic_coding
```

Keyword dictionary per aspect, applied to negative comments only.

---

## Step 7 — Thematic distribution figure

```powershell
.\.venv\Scripts\python.exe -m plot_thematic_distribution
```

**Output** `data/results/visualization/gambar_5_10_distribusi_tema.png`

---

## Step 8 — Problem identification per aspect

Four scripts, one per aspect. Independent of each other; run all four.

```powershell
.\.venv\Scripts\python.exe -m identifikasi_masalah_akuntabilitas
.\.venv\Scripts\python.exe -m identifikasi_masalah_efektivitas_efisiensi
.\.venv\Scripts\python.exe -m identifikasi_masalah_responsivitas
.\.venv\Scripts\python.exe -m identifikasi_masalah_transparansi
```

Each writes `data/results/bukti_komentar_<aspek>.csv` (row-level evidence) and
`data/results/rekap_masalah_<aspek>.csv` (category counts and percentages).

---

## Step 9 — RCA and recommendations

**Input** `data/results/indobert_absa_result_oof.csv`
**Output** `data/results/identifikasi_permasalahan/`

```powershell
.\.venv\Scripts\python.exe -m analysis.analysis_pipeline
```

Produces the aspect/sentiment distribution, the qualitative sample of negative
comments, the 5-Whys RCA table and the summary figure.

Note the `RCA_DATA` dictionary in `analysis/analysis_pipeline.py` is written by
the researcher, not derived from the data. Update it if the corrected
distribution shifts the interpretation.

---

## Step 10 — Figures

```powershell
.\.venv\Scripts\python.exe -m visualization.visualize_absa
.\.venv\Scripts\python.exe -m visualization.plot_confusion_matrix
```

---

## Full clean run

```powershell
.\run_all.bat
```

Or stage by stage:

```powershell
.\.venv\Scripts\python.exe -m preprocessing.preprocess
.\.venv\Scripts\python.exe -m preprocessing.preprocess_absa
.\.venv\Scripts\python.exe -m indobert.train_cv
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

Skipping step 2 assumes `data/annotation/dataset_train.csv` already exists.

---

## What is not reproducible from this repository

| Stage | Status |
|---|---|
| Raw scrape | **No TikTok scraper committed.** `scraper/scraper.py` targets Twitter/X. See BUG-09. |
| Annotation | Human work. Files are committed, the process is not automated. |
| Model checkpoints | ~80 GB, gitignored. Step 4 must be re-run by anyone cloning. |

---

## Troubleshooting

**`FileNotFoundError: data/processed/aspect_annotation_final_clean.csv`**
Only `indobert/train_aspect.py` needs this file, and it is not produced by any
current stage. It exists only under `backup/`, which is gitignored. That script
is superseded by `indobert/train_cv.py`; see BUG-11.

**`Repository not found` on push**
Confirm credentials with
`"protocol=https`nhost=github.com`n`n" | git credential-manager get`, then
`git push -u origin main`.

**Low disk before step 4**
`models/` reaches ~80 GB. Delete the `checkpoint-*` directories afterwards; each
fold retains its final `model.safetensors`. See BUG-10.

**Predictions look implausibly accurate (>90%)**
Almost certainly the wrong script. `indobert/predict_absa.py` reproduces the
leakage bug. Use `indobert.predict_absa_oof`. A true figure is around 78% exact
8-class agreement.