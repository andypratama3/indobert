# Run Guide — IndoBERT ABSA Pipeline

Step-by-step, in execution order. Every command is run from the repository root.

All paths are relative, so `cd` to the project root first:

```powershell
cd D:\skripsi_indobert_sentiment
```

---

## Step 0 — Environment (once)

Python 3.13 is what this was developed against.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`accelerate` is needed by `transformers.Trainer` but is not listed in
`requirements.txt`. Install it explicitly:

```powershell
pip install accelerate
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