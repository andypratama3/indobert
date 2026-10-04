# Bug Tracker — IndoBERT ABSA Pipeline

Audit date: 2026-10-04
Scope: `skripsi_indobert_sentiment`, full path `raw scrape → preprocessing → annotation → fine-tuning → prediction → analysis`

Every number below was produced by running Python against the actual CSV and model
artefacts in this repository, not by reading the code alone.

---

## Summary

| ID | Severity | Component | Status |
|---|---|---|---|
| BUG-01 | Critical | `predict_absa.py` | **Fixed** |
| BUG-02 | High | `preprocess.py` | **Fixed** |
| BUG-03 | High | `preprocess.py` | **Fixed** |
| BUG-04 | High | `preprocess.py` | **Fixed** |
| BUG-05 | Critical | `preprocess.py` vs training data | **Fixed** |
| BUG-06 | High | `predict_absa.py` | **Fixed** |
| BUG-07 | High | `dataset_train.csv` provenance | Documented — needs researcher input |
| BUG-08 | Medium | class imbalance | Documented — needs researcher decision |
| BUG-09 | Medium | provenance | Documented |
| BUG-10 | Low | disk hygiene | Open — optional |
| BUG-11 | Low | repo hygiene | Open |
| BUG-12 | Critical | `preprocess_absa.py` | **Fixed** |

---

## BUG-01 — Training data leakage in inference (CRITICAL, fixed)

`indobert/predict_absa.py` loaded `models/indobert_aspect_sentiment_cv/fold_6`
and predicted **every** row of `data/processed/comments_clean.csv`.

The corpus is a superset of the annotated training set, so most rows had already
been seen by the model during fine-tuning.

**Measured contamination**

| Metric | Value |
|---|---|
| Rows predicted | 998 |
| Rows present in the fine-tuning set | **673 (67.4%)** |
| Genuinely unseen rows | 325 |
| Mean confidence, memorised rows | 0.980 |
| Mean confidence, unseen rows | 0.928 |
| Negative rows used as qualitative evidence | 895 |
| Negative rows that came from training | **613 (68.5%)** |

**Measured effect on reported performance**

Accuracy against the human gold labels, computed on the rows where a gold label
exists:

| Metric | Leaked (fold_6) | Leakage-free (OOF) | Honest 10-fold CV |
|---|---|---|---|
| Aspect accuracy | 97.84% | **84.59%** | — |
| Sentiment accuracy | 98.92% | **91.41%** | — |
| Exact 8-class | 97.19% | **78.37%** | 77.62% acc / 76.09% F1 |

The leaked pipeline overstated exact-label accuracy by roughly **19 percentage
points**. The leakage-free result (78.37%) independently reproduces the
cross-validation estimate (77.62%), which is the strongest available evidence
that the fix is correct.

**Blast radius.** `data/results/indobert_absa_result.csv` was the input to
`thematic_coding.py`, all four `identifikasi_masalah_*.py` scripts,
`analysis/analysis_pipeline.py` and `visualization/visualize_absa.py`. Every
table and figure under `data/results/` derived from it was affected.

**Fix.** New `indobert/predict_absa_oof.py` implements out-of-fold prediction:

1. Reproduces the checkpoint partition exactly by importing `load_and_clean_data`
   from `indobert.train_cv` and reusing `StratifiedKFold(n_splits=10,
   shuffle=True, random_state=42)`. Verified to partition all 800 rows into
   10 × 80 with no overlap.
2. Predicts each annotated row with the one fold model that held it out.
3. Predicts never-annotated rows by majority vote across all ten folds.

Every row is scored by a model that never trained on it. Nothing is dropped, so
the corpus keeps its full size. Output carries `prediction_source` and `votes`
columns so provenance is auditable per row.

The contaminated file is retained as
`data/results/indobert_absa_result_leaky.csv` for audit only.

---

## BUG-02 — Length filter discarded meaningful short comments (HIGH, fixed)

`preprocessing/preprocess.py` applied `df["clean_text"].str.len() > 10`.

Most scraped rows carry a trailing TikTok timestamp such as `3-27` or `12-5`.
That stamp is not content, but it counted toward the length. The consequence was
that short, genuinely sentiment-bearing comments were thrown away while pure
noise survived.

**24 comments recovered**, for example:

| Raw | Old verdict | Meaning |
|---|---|---|
| `setuju` | dropped (6 chars) | agreement |
| `parah` | dropped (5 chars) | "severe" |
| `pecat 4-25` | dropped (10 chars) | "dismiss/fire" |
| `MARWAH 3-6` | dropped (10 chars) | "luxury" |
| `usut 3-9` | dropped (8 chars) | "investigate" |
| `keren 4-23` | dropped (10 chars) | "impressive" (sarcastic) |

**Fix.** Strip the trailing date stamp before the length decision, then judge on
remaining content. See `preprocessing/text_cleaning.py:strip_date_stamp`.
Corpus grew from 998 to **1031** rows.

---

## BUG-03 — Deduplication collapsed distinct users (HIGH, fixed)

The old code used `drop_duplicates(subset=["clean_text"])`. Two different
accounts posting the same sentence were treated as one row.

Verified: `duplicated(subset=["comment_username","comment_text"])` on the raw
file returns **0**, but `duplicated(subset=["clean_text"])` after normalisation
returns **9**. Those 9 were distinct users whose comments happen to normalise
identically — real, separate opinions about the same event, and exactly the kind
of row a frequency analysis should keep.

**Fix.** Deduplicate on `(comment_username, clean_text)`.

---

## BUG-04 — Seven scripts hardcoded the prediction path (HIGH, fixed)

The prediction input path was a string literal repeated in `thematic_coding.py`,
four `identifikasi_masalah_*.py` scripts, `analysis/analysis_pipeline.py` and
`visualization/visualize_absa.py`. Nothing tied them to a particular prediction
run, which is precisely how a contaminated file propagated silently into every
reported result.

**Fix.** New top-level `config.py` declares all paths once. `ABSA_RESULT` points
at the OOF file; `ABSA_RESULT_LEAKY` is named explicitly and documented as
audit-only.

---

## BUG-05 — Training data was not reproducible from the current code (CRITICAL, fixed)

`preprocessing/preprocess.py` had gained a slang-expansion step
(`ga`→`tidak`, `yg`→`yang`, …) at some point after the annotated dataset was
generated. The stored training column therefore cannot be regenerated by the
current script.

Reproducing `dataset_train_clean.csv["clean_text"]` from its own
`original_text`:

| Cleaner variant | Exact matches |
|---|---|
| `preprocess.py` character class **keeping numerals**, no slang step | **800 / 800** |
| `preprocess.py` character class **keeping numerals**, with slang step | 0 / 800 |
| Character class removing numerals | 94 / 800 |

This identifies the exact cleaner that built the training data: numerals kept,
slang step absent. It also means the current script would have fed the models a
text distribution they had never seen.

**Fix.** `preprocessing/text_cleaning.py` is now the single source of truth.
`normalize()` reproduces the training column byte-for-byte and is what gets fed
to the models. Slang expansion happens only in `normalize_for_gate()`, which
feeds the inclusion decision and never the model.

---

## BUG-06 — Train/inference preprocessing skew (HIGH, fixed)

`indobert/predict_absa.py` fed `comments_clean.csv["clean_text"]`, produced by
`preprocess.py`, into models fine-tuned on `dataset_train_clean.csv["clean_text"]`,
produced by a different cleaner. Two cleaners, one model. The skew is now
impossible by construction: both sides use `text_cleaning.normalize()`.

---

## BUG-07 — Annotated set does not fully come from the raw file (HIGH, open)

Locating annotated rows in the raw scrape by normalised text:

| Set | Rows | Located in raw 1081 |
|---|---|---|
| `dataset_train.csv` | 800 | **684 (85.5%)** |
| not located | | **116 (14.5%)** |

14.5% of the annotated training rows are not present in
`data/raw/tiktok_comments_mobil_dinas_kaltim.csv`. The annotation set was
therefore built from a different or earlier scrape than the file now in the
repository.

**Consequence.** Those 116 rows still train the models, but they cannot receive
a corpus prediction because no corpus row corresponds to them. This is why
673 + 358 = 1031 rather than 800 + 231.

**Needs researcher input.** Either recover the original scrape, or state in
Chapter 3 that the corpus and the annotation sample were drawn from two scrape
batches. This is a provenance claim, not something to patch in code.

---

## BUG-08 — Severe class imbalance against 10 folds (MEDIUM, open)

Label distribution of the annotated set:

| Label | Count |
|---|---|
| akuntabilitas_negatif | 305 |
| efektivitas_efisiensi_negatif | 251 |
| responsivitas_negatif | 92 |
| transparansi_negatif | 69 |
| akuntabilitas_positif | 58 |
| efektivitas_efisiensi_positif | 12 |
| responsivitas_positif | 7 |
| **transparansi_positif** | **6** |

With `n_splits=10`, classes with fewer than 10 members cannot be stratified.
`sklearn` warns, and `train_cv.py` prints a warning but proceeds. Some folds
therefore contain **zero** positive examples for the rarest classes, and those
folds contribute a meaningless `eval_f1` to the reported mean.

This inflates the variance of the reported cross-validation metrics
(`eval_accuracy` std = 0.0388 on a mean of 0.7762).

**Needs researcher decision.** Either reduce to fewer folds, merge the sparse
positive classes, or report the imbalance explicitly with per-class metrics
rather than a weighted average. It also explains why `Efektivitas dan
Efisiensi` receives 299 rows but only 4 positive predictions.

---

## BUG-09 — The raw scrape cannot be regenerated from this repository (MEDIUM, open)

`data/raw/tiktok_comments_mobil_dinas_kaltim.csv` is TikTok data, but the only
scraper in the repository is `scraper/scraper.py`, a **Twitter/X** scraper driven
by `run.py`. No TikTok scraper is committed.

The single most important input to the thesis cannot be reproduced by anyone
running this repository.

**Needs researcher input.** Either commit the TikTok scraper, or document in
Chapter 3 that collection used a separate tool.

---

## BUG-10 — 52 stale checkpoint directories (~80 GB) (LOW, open)

`models/` holds 52 `checkpoint-*` directories. `save_total_limit=1` did not prune
them, and each contains a ~950 MB `optimizer.pt`. Total `models/` size is ~80 GB,
which is why `models/` is gitignored.

Safe to delete: every fold directory already holds the final `model.safetensors`
plus tokenizer, verified present for all ten folds. Only optimiser state and
intermediate weights would be lost, and those are not needed to reproduce
inference. **Not deleted automatically** — this is destructive.

---

## BUG-11 — Repository hygiene (LOW, open)

- `main.py` — 0 bytes
- `indobert/evaluate.py` — 0 bytes
- `requirements.txt` lists 9 packages. Every third-party import in the project is
  covered, but versions are unpinned, so a fresh install may not reproduce the
  reported metrics. Worth pinning.
- `models/indobert_aspect/` and `models/indobert_aspect_sentiment/` are trained
  but never used by any analysis script.

---

## BUG-12 — `preprocess_absa.py` silently destroyed the training data (CRITICAL, fixed)

Found by smoke-testing every command in `RUN_GUIDE.md` after writing it. Running
step 3 once, as the guide instructed, overwrote
`data/processed/dataset_train_clean.csv` with text cleaned by a *different*
function than the one that originally built it.

**Measured damage from a single run**

| Property | Before | After one run |
|---|---|---|
| Rows | 800 | **799** |
| Rows matching the canonical cleaner | 800 | **94** |

The script carried its own inline cleaner (numerals stripped) and applied
`len(clean_text) > 0`, which removed one annotated row. The models in `models/`
were trained on the original column, so after that overwrite the training file no
longer corresponded to the checkpoints.

The file was restored from git, and `preprocess_absa.py` rewritten to use
`text_cleaning.normalize()`, apply no row filter, and assert 800/800
reproduction. Re-running it twice now leaves the file byte-identical to the
original commit.

**Why this mattered beyond the run itself.** This is the same failure mode as
BUG-05: two cleaners producing two versions of the same column, with the
non-canonical one winning by virtue of being written last. Centralising on
`preprocessing/text_cleaning.py` plus the reproduction assertion means it can no
longer recur silently.

---

## Verification performed

| Check | Result |
|---|---|
| Fold partition reproduces checkpoints | 10 × 80 = 800, no overlap |
| All ten fold models load | yes, each has `model.safetensors` + `config.json` + `tokenizer.json` |
| OOF accuracy vs untouched CV estimate | 78.37% vs 77.62% — consistent |
| Corpus rows after fix | 1031 of 1081 raw |
| Rows dropped and why | 50, all emoji-only or bare date stamps |
| `preprocess_absa.py` idempotent | yes, byte-identical after two runs |
| Downstream scripts re-run on clean data | all succeeded |
| Full chain re-run from scratch | 12/12 stages pass |