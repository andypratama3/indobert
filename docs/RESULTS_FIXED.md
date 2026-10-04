# Fixed Results — IndoBERT ABSA

Generated 2026-10-04 by re-running the complete pipeline on leakage-free
predictions. Supersedes every table and figure previously produced from
`data/results/indobert_absa_result_leaky.csv`.

**Read `docs/BUG_TRACKER.md` first.** The previous numbers were inflated by
training-data leakage and should not be quoted anywhere in the thesis.

---

## 1. What changed

| | Before | After |
|---|---|---|
| Inference method | single fold (`fold_6`) over the whole corpus | out-of-fold + 10-fold ensemble |
| Rows predicted by a model that trained on them | **673 (67.4%)** | **0 (0%)** |
| Mean confidence | 0.9632 | 0.9263 |
| Apparent exact-label accuracy | 97.19% | **78.37%** |

The honest accuracy is the one already reported by the untouched 10-fold
cross-validation: **77.62% accuracy, 76.09% weighted F1**.

---

## 2. Data lineage

```
tiktok_comments_mobil_dinas_kaltim.csv          1081   raw scrape
        │
        │  preprocessing.preprocess   (date-stamp strip, user-aware dedup)
        ▼
comments_clean.csv                              1031   analysable corpus
        │
        ├── dataset_train.csv                       800   manually annotated
        │           │  preprocessing.preprocess_absa
        │           ▼
        │   dataset_train_clean.csv                 800   model training input
        │           │  indobert.train_cv  (10-fold, seed 42)
        │           ▼
        │   models/indobert_aspect_sentiment_cv/fold_1..10
        │           │  indobert.predict_absa_oof
        │           ▼
        └── indobert_absa_result_oof.csv           1031   leakage-free predictions
                    │  thematic_coding → identifikasi_masalah_* → analysis_pipeline
                    ▼
                data/results/identifikasi_permasalahan/
```

Corpus accounting: 1081 raw → 1031 analysable. The 50 dropped rows are
emoji-only reactions (`😂`, `🥰👍👍`) or a bare date stamp (`4-27`) with no text.

The previous pipeline kept 998 rows and silently discarded 24 meaningful short
comments. See BUG-02.

---

## 3. Model performance (unaffected by the leakage)

From `data/results/cv/cv_summary_metrics.csv`, 10-fold stratified CV, seed 42.
These metrics were always computed on held-out folds and remain valid.

| Metric | Mean | Std |
|---|---|---|
| Accuracy | 0.7762 | 0.0388 |
| Weighted precision | 0.7636 | 0.0476 |
| Weighted recall | 0.7762 | 0.0388 |
| Weighted F1 | 0.7609 | 0.0386 |

Independent confirmation from the leakage-free run: exact 8-class agreement with
human labels was **78.37%** on 675 rows that could be checked, against a
cross-validated accuracy of 77.62%. The two agree to within one point.

Component accuracy, leakage-free:

| Level | Accuracy |
|---|---|
| Aspect (4-way) | 84.59% |
| Sentiment (2-way) | 91.41% |

Caveat: see BUG-08. `transparansi_positif` has 6 training examples against 10
folds, so the reported mean carries real instability.

---

## 4. Aspect and sentiment distribution (n = 1031)

| Aspect | Negative | Positive | Total | % negative |
|---|---:|---:|---:|---:|
| Akuntabilitas | 559 | 77 | 636 | 87.9% |
| Efektivitas dan Efisiensi | 295 | 4 | 299 | 98.7% |
| Responsivitas | 64 | 0 | 64 | 100.0% |
| Transparansi | 32 | 0 | 32 | 100.0% |
| **Total** | **950** | **81** | **1031** | **92.1%** |

Headline shift against the contaminated run:

| | Contaminated | Fixed |
|---|---|---|
| n | 998 | 1031 |
| Negative | 895 (89.7%) | 950 (92.1%) |
| Positive | 103 (10.3%) | 81 (7.9%) |

Removing the optimistic bias moved the negative share **up** by 2.4 points, not
down. This is the opposite of the usual direction and worth understanding before
writing it up: the leaked run was not merely flattering, it was also *distorting
the class balance*. Memorised rows were assigned their training label with high
confidence, and the training set is 72% negative, so leakage disproportionately
manufactured confident negative predictions on rows the model had merely got
wrong. The corrected figure is higher because those mislabelled rows have now
been honestly classified.

### Reading the distribution

`Akuntabilitas` dominates at 61.7% of the corpus, and `Efektivitas dan
Efisiensi` at 29.0%. Together they are 90.7%.

`Responsivitas` (64 rows, 6.2%) and `Transparansi` (32 rows, 3.1%) are small, and
both receive **zero** positive predictions. Do not read that as unanimous public
approval. Given only 6 and 7 positive training examples respectively, the model
has effectively learned to emit the negative class for these aspects. Report
these two aspects with an explicit small-sample caveat, or report per-class
recall so the limitation is visible rather than implied.

---

## 5. Negative-comment thematic breakdown

From `data/results/identifikasi_permasalahan/` and the four
`rekap_masalah_*.csv` files.

**Akuntabilitas — 559 negative comments**

| Category | Count | % |
|---|---:|---:|
| No specific complaint | 229 | 41.0 |
| Luxury lifestyle gap, officials vs public | 114 | 20.4 |
| Abuse of office / corruption, demands for investigation | 69 | 12.3 |
| Demands for political accountability, decisive action | 30 | 5.4 |
| Personal cynicism about the governor | 21 | 3.7 |
| Poor leadership, perceived self-interest | 21 | 3.8 |
| Residents' embarrassment, damage to the region's name | 12 | 2.1 |
| Nepotism / political dynasty | 11 | 2.0 |
| General disappointment with national governance (off-topic) | 5 | 0.9 |
| Comments defending the governor | 2 | 0.4 |
| Off-topic noise | 45 | 8.0 |

**Efektivitas dan Efisiensi — 295 negative comments**

| Category | Count | % |
|---|---:|---:|
| Wrong budget priority: roads neglected for official cars | 170 | 57.6 |
| Price seen as excessive, cheaper alternatives suggested | 95 | 32.2 |
| Budget wasted on unrelated items | 5 | 1.7 |
| Off-topic noise | 5 | 1.7 |
| Use of public tax money for non-essentials | 7 | 2.4 |
| Suspected misuse of official assets after tenure | 2 | 0.7 |
| No specific complaint | 11 | 3.7 |

**Responsivitas — 64 negative comments**

| Category | Count | % |
|---|---:|---:|
| Officials judged to only insinuate, no real action | 23 | 35.9 |
| Lack of public voices from/local responses in East Kalimantan | 14 | 21.9 |
| Personal cynicism about speaking style | 8 | 12.5 |
| No specific complaint | 12 | 18.7 |
| Off-topic noise | 5 | 7.8 |
| Demands for official clarification | 1 | 1.6 |
| Criticism of slow government response to other issues | 1 | 1.6 |

**Transparansi — 32 negative comments**

| Category | Count | % |
|---|---:|---:|
| No specific complaint | 14 | 43.7 |
| Public disbelieves the car or money was actually returned | 6 | 18.8 |
| Unclear or inconsistent budget figures | 4 | 12.5 |
| Suspicion over licence plate change as concealment | 3 | 9.4 |
| Off-topic noise | 3 | 9.4 |
| Transparency demands, lack of official coverage | 2 | 6.2 |

---

## 6. Fixed conclusion

> Public discourse on the East Kalimantan gubernatorial official-car procurement
> is overwhelmingly critical. Of 1031 analysable comments, **92.1% were predicted
> negative**. Criticism concentrates on **accountability (61.7% of the corpus)**
> and **effectiveness and efficiency (29.0%)**, which together account for 90.7%
> of all comments.
>
> Within accountability, the dominant substantive concern is the **gap between
> officials' luxury and public conditions (20.4%)**, followed by suspected abuse
> of office and demands for legal investigation (12.3%).
>
> Within effectiveness and efficiency, criticism is concentrated on **wrong budget
> priority — roads neglected in favour of official cars (57.6%)** and the
> **perceived excess of the purchase price (32.2%)**. Together these two account
> for 89.8% of negative comments in this aspect.
>
> Model performance is reported from 10-fold cross-validation as **77.62%
> accuracy and 76.09% weighted F1**, confirmed by leakage-free out-of-fold
> agreement of 78.37%. Aspect classification reaches 84.59% and sentiment
> classification 91.41%.
>
> **Caveats that must accompany this conclusion.** First, the 673 annotated
> comments and 358 unannotated comments are not statistically equivalent; the
> annotated subset was selected for labelling, and if that selection was not
> random the corpus distribution inherits that bias. Second, `Responsivitas` (64
> comments) and `Transparansi` (32 comments) rest on very small samples and
> receive zero positive predictions, reflecting 7 and 6 positive training examples
> respectively rather than genuine consensus. Third, 116 of the 800 annotated
> comments could not be located in the raw scrape, so the corpus and the
> annotation sample derive from two collection batches (BUG-07). Fourth, the raw
> TikTok scrape cannot be regenerated from this repository, since the committed
> scraper targets Twitter/X (BUG-09).

---

## 7. Files

**New**

| File | Purpose |
|---|---|
| `config.py` | single source of truth for all paths |
| `preprocessing/text_cleaning.py` | canonical cleaner, reproduces training data exactly |
| `indobert/predict_absa_oof.py` | leakage-free out-of-fold inference |
| `docs/BUG_TRACKER.md` | full audit |
| `docs/RESULTS_FIXED.md` | this document |

**Rewritten outputs**

`comments_clean.csv` (1031 rows) · `indobert_absa_result_oof.csv` ·
`thematic_coding_result.csv` · `thematic_summary.csv` · all four
`rekap_masalah_*.csv` and `bukti_komentar_*.csv` ·
`identifikasi_permasalahan/*` · `visualization/*`

**Audit only, do not cite**

`data/results/indobert_absa_result_leaky.csv`

---

## 8. Reproducing these numbers

See `RUN_GUIDE.md`. Steps 1–3 are sufficient; step 4 is only needed if the
annotation set changes, and takes hours on a GPU.