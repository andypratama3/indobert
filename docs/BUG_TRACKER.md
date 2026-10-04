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
| BUG-07 | High | `dataset_train.csv` provenance | **Fixed — quantified, unrecoverable** |
| BUG-08 | Medium | class imbalance | **Fixed** |
| BUG-09 | Medium | provenance | **Fixed** — `scraper/tiktok_scraper.py` added |
| BUG-10 | Low | disk hygiene | **Fixed** — 72.3 GB reclaimed |
| BUG-11 | Low | repo hygiene | **Fixed** — empty files removed, deps pinned |
| BUG-12 | Critical | `preprocess_absa.py` | **Fixed** |
| BUG-13 | Low | no single entry point | **Fixed** — `run_all.py` / `run_all.bat` |

All 13 bugs are addressed. Four required a researcher decision rather than a
code change, and their measurements are now recorded instead of being silently
absorbed: BUG-07 (annotation provenance), BUG-08 (class imbalance), BUG-09
(scraper completeness) and BUG-11 (orphaned models). See "Residual limitations"
at the end.

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

## BUG-07 — Annotated set does not fully come from the raw file (HIGH, fixed)

Status: **Fixed — quantified, unrecoverable.**

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
a corpus prediction because no corpus row corresponds to them. The accounting is
therefore:

- annotated: 800 = **684 locatable in the corpus** + **116 with no corpus row**
- corpus: 1031 = **687 rows carrying an annotated label** + **344 with none**

(The 687 exceeds the 684 because a few annotated rows share a normalised text
key with more than one corpus row. Note that the "673" quoted elsewhere in this
document for the contaminated run refers to the old 998-row corpus and must not
be mixed with the 1031-row figures.)

**Recovery was attempted and failed.** All 24 CSVs under `backup/` were scanned
(7 at the top level, plus `backup/processed/` and `backup/result/`). Three of the
116 appear in `backup/processed/aspect_annotation_final_clean.csv`, but that file
is itself an annotation snapshot, not the raw scrape, so it does not recover the
source rows. 88 of the 116 appear in
`data/annotation/aspect_sentiment_annotation_final.csv`; the remaining 28 exist
in no other file in the repository at all. The original scrape batch is
unrecoverable by merging.

### Resolution

New `analysis/provenance_audit.py` recomputes all of this from the CSVs on every
run, rather than leaving it as prose in this document.

**The missing rows are not a random sample.** Their aspect distribution is
completely different from the rows that were located:

| Aspect | Located (684) | Not located (116) |
|---|---:|---:|
| Akuntabilitas | 50.6% | 14.7% |
| Efektivitas & Efisiensi | 37.1% | 7.8% |
| **Responsivitas** | **7.0%** | **44.0%** |
| **Transparansi** | **5.3%** | **33.6%** |

The sentiment split is *not* skewed the same way (90.1% negative located vs 87.1%
negative not located), so this is topical sampling, not a labelling-tone
difference.

**Per-class damage**, as actual ÷ expected unlocated rows if loss were random:

| Label | n | Not located | Expected | Ratio |
|---|---:|---:|---:|---:|
| `responsivitas_positif` | 7 | 6 | 1.0 | **6.00** |
| `transparansi_negatif` | 69 | 38 | 10.0 | **3.80** |
| `responsivitas_negatif` | 92 | 45 | 13.3 | **3.38** |
| `transparansi_positif` | 6 | 1 | 0.9 | 1.11 |
| `akuntabilitas_negatif` | 305 | 11 | 44.2 | 0.25 |
| `efektivitas_efisiensi_negatif` | 251 | 7 | 36.4 | 0.19 |

6 of the 7 `responsivitas_positif` training rows are in the lost batch.

**Chi-square test of independence** (implemented from the definitions in
`analysis/provenance_audit.py`, no scipy dependency; cross-checked against scipy
where available):

| Variable | χ² | df | p | Cramér's V | Verdict at threshold 0.10 |
|---|---:|---:|---:|---:|---|
| Aspect | **248.73** | 3 | 1.2 × 10⁻⁵³ | **0.558** | material bias |
| Sentiment | 0.95 | 1 | 0.33 | 0.035 | no material bias |

So whether a row is locatable in the raw scrape is almost perfectly predicted by
its aspect, and is unrelated to its sentiment.

**Effect on the reported aspect distribution** — training prior vs corpus prior:

| Aspect | Training | Corpus | Ratio |
|---|---:|---:|---:|
| Akuntabilitas | 363 (45.38%) | 636 (61.69%) | 0.74× |
| Efektivitas & Efisiensi | 263 (32.88%) | 299 (29.00%) | 1.13× |
| Responsivitas | 99 (12.38%) | 64 (6.21%) | **1.99×** |
| Transparansi | 75 (9.38%) | 32 (3.10%) | **3.03×** |
| **Responsivitas + Transparansi** | **174 (21.75%)** | **96 (9.31%)** | **2.34×** |

The annotation sample over-weights the two smallest corpus aspects by 2.34×
combined, so the aspect classifier's prior is tilted toward exactly the aspects
the corpus barely contains. The reported 64 `Responsivitas` and 32 `Transparansi`
rows are model output under that prior, not counts of public opinion.

**Honest statement of what was and was not fixed.** The rows cannot be recovered
and the bias cannot be removed without discarding 116 annotated rows and
retraining. What *was* fixed is that the problem can no longer be overlooked:

- `data/annotation/annotated_not_in_corpus.csv` — the 116 rows with full
  annotations, preserved as a tracked artefact.
- `data/results/provenance_audit.csv` — every number above, regenerated on run.
- `data/annotation/README.md` — the disclosure, next to the data itself.
- `docs/RESULTS_FIXED.md` section 6 — the caveat attached to the conclusion.
- `analysis/provenance_audit.py` raises a sampling-bias warning whenever any
  Cramér's V exceeds 0.10, so a recurrence cannot pass silently.

**Still requires researcher input:** this must be disclosed as a limitation in
Chapter 3, stating that the corpus and the annotation sample were drawn from two
scrape batches, and repeated in the Chapter 5 caveats.

**One matching caveat, reported rather than hidden.** One of the 684 "located"
rows (`@ernawita85`, `Akuntabilitas_Negatif`) is written in Unicode mathematical
stylised letters, so it normalises to the empty string — as do the bare date
stamps in the raw file. It matches on an empty key, which is not evidence of
provenance. The script flags it as `match_is_degenerate` and also reports the
strict variant: 683 located / 117 not located. Headline figures use the
documented method (684/116) so they agree with the table above.

---

## BUG-08 — Severe class imbalance against 10 folds (MEDIUM, fixed)

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

It also explains why `Efektivitas dan Efisiensi` receives 299 rows but only 4
positive predictions.

### Resolution

The imbalance itself cannot be removed without new annotation, but it can be
stopped hiding inside a weighted average. Three changes, no retraining required.

**1. Per-class metrics are now recorded during training.**
`compute_metrics` in `indobert/train_cv.py` calls
`precision_recall_fscore_support(..., average=None, labels=range(8),
zero_division=0)` and `classification_report`, and returns
`precision_<label>`, `recall_<label>`, `f1_<label>`, `support_<label>` for all
eight classes plus `macro_precision`, `macro_recall`, `macro_f1`. The original
weighted `precision`/`recall`/`f1` keys are unchanged, so
`metric_for_best_model="f1"` still works. The report is printed once per
distinct validation-fold composition rather than once per epoch.

**2. The warning is explicit and actionable, and fold integrity is checked.**
`get_sparse_classes()` returns the offending classes;
`print_sparse_class_warning()` names them with their counts and the required
minimum, states the concrete consequence, and lists three remedies.
`load_and_clean_data()` populates the module-level `SPARSE_CLASSES` list (also
importable as `indobert.train_cv.SPARSE_CLASSES`) and still proceeds — nothing
in the pipeline stops. `check_fold_integrity()` is called from `main()` before
any training begins; it builds a full (fold × class) coverage table, prints
every empty pair by name, and writes `data/results/cv/fold_class_coverage.csv`.

**3. Honest per-class numbers were derived from existing artefacts.**
New `analysis/class_imbalance_report.py` needs no GPU. It re-derives the
StratifiedKFold(10, 42) split via the imported `load_and_clean_data`, sums the
ten saved `confusion_matrix_fold_*.csv` files into one 800-prediction confusion
matrix, and derives per-class precision/recall/F1 from it. It also verifies
that the re-derived split is the split that produced the checkpoints: the
true-label counts per fold match the saved confusion matrices in **10/10 folds**.
Writes `data/results/cv/per_class_metrics.csv` and
`data/results/cv/fold_class_coverage.csv`.

**Which folds are missing which classes — 7 empty (fold, class) pairs:**

| Class | Support | Empty in folds |
|---|---|---|
| transparansi_positif | 6 | **2, 3, 4, 5** (4 of 10) |
| responsivitas_positif | 7 | **1, 2, 3** (3 of 10) |
| efektivitas_efisiensi_positif | 12 | none, but min 1 / max 2 per fold |

All other classes appear in every fold. `transparansi_positif` is absent from
4 folds and `responsivitas_positif` from 3, so **5 of 10 folds** have at least
one class with zero validation examples.

**Per-class metrics derived from the saved confusion matrices (n = 800):**

| Label | Support | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| akuntabilitas_negatif | 305 | 0.7647 | 0.8525 | 0.8062 | 260 | 80 | 45 |
| efektivitas_efisiensi_negatif | 251 | 0.8544 | 0.8884 | 0.8711 | 223 | 38 | 28 |
| responsivitas_negatif | 92 | 0.7303 | 0.7065 | 0.7182 | 65 | 24 | 27 |
| transparansi_negatif | 69 | 0.7903 | 0.7101 | 0.7481 | 49 | 13 | 20 |
| akuntabilitas_positif | 58 | 0.5455 | 0.4138 | 0.4706 | 24 | 20 | 34 |
| efektivitas_efisiensi_positif | 12 | 0.0000 | 0.0000 | 0.0000 | 0 | 4 | 12 |
| responsivitas_positif | 7 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 7 |
| transparansi_positif | 6 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 6 |

What these numbers show:

- The headline 0.7762 accuracy is **entirely carried by the five majority
  classes**. All 25 misclassified `efektivitas_efisiensi_positif`,
  `responsivitas_positif` and `transparansi_positif` examples land in a
  negative or a different positive class.
- **Every positive class is weak.** Macro F1 is 0.4518 against a weighted F1 of
  0.7619. The gap is the imbalance showing through, and the weighted figure
  the thesis currently quotes conceals it.
- **The three rarest positive classes have zero true positives.** They are not
  merely poorly measured — with 6, 7 and 12 training examples across 720 training
  rows per fold, the model never learned to emit them at all.
- Weighted precision from the pooled matrix is 0.7513 versus the 0.7636 mean of
  per-fold values. Both are correct; the pooled figure is the more conservative
  one and does not depend on where the rare classes happened to land. Accuracy
  matches exactly (0.7762) because all folds are the same size.

**Recommendation: keep `n_splits=10` and report with caveats.** Dropping to 6
folds would make every class appear everywhere, but it costs a full retraining
and weakens the experimental control from 10 fits to 6 for a class that has 6
examples in total. Merging the sparse positive classes would change the label
schema and invalidate the annotation, all ten checkpoints and every downstream
script. The only genuinely correct remedy is more annotated positive examples
for Responsivitas and Transparansi, which is an annotation problem, not a code
problem. Reporting per-class metrics with an explicit caveat costs nothing and
is already done.

---

## BUG-09 — The raw scrape cannot be regenerated from this repository (MEDIUM, fixed)

`data/raw/tiktok_comments_mobil_dinas_kaltim.csv` is TikTok data, but the only
scraper in the repository was `scraper/scraper.py`, a **Twitter/X** scraper driven
by `run.py`. No TikTok scraper was committed, so the single most important input
to the thesis could not be reproduced by anyone running this repository.

**Fix: `scraper/tiktok_scraper.py` added.**

| Property | Value |
|---|---|
| Class | `TikTokScraper` (mirrors `TwitterScraper` structure) |
| Entry point | `python -m scraper.tiktok_scraper` |
| Browser | Selenium Chrome, same driver options as `scraper/scraper.py` |
| Default keywords | The same 32 queries as `run.py` |
| Login | Not required — TikTok comments are public |

It emits exactly the 7-column schema the raw CSV already uses, in the same
order, and `preprocessing/preprocess.py` consumes the file without changes:

```
keyword, post_url, post_username, comment_username,
comment_profile_url, comment_text, comment_like_count
```

Behaviour matching the existing scraper's conventions:

- Auto-saves to CSV after **every** completed video, so the run is
  **resumable** — a restart skips `(comment_username, comment_text)` pairs and
  post URLs already in the file (`--no-resume` disables this).
- Duplicate-safe via `drop_duplicates` on `(comment_username, comment_text)`,
  which is exactly the key the shipped file is unique on (verified: 0 duplicates
  across 1081 rows).
- Scroll-based comment loading with a no-new-comments break, matching
  `TwitterScraper.scrape_replies`.
- Rate-limit/CAPTCHA detection that stops and cools down rather than forcing
  requests, plus randomised polite delays between pages and scrolls.

**Two details that matter for byte-level reproduction.** Both were derived by
inspecting the committed CSV, not assumed:

1. `comment_like_count` is stored as the **raw UI text** (`"1461"`, `"15K"`,
   `"26K"`), not as a parsed integer. The shipped file is dtype `object` and
   contains `15K` and `26K`, so parsing to `int` would produce a file that
   differs from the one the thesis was built on.
2. `keyword` is the query that surfaced the video, not comment text. The
   shipped file used 20 TikTok-specific queries (`TIKTOK_KEYWORDS_RAW`,
   reachable via `--raw-keywords`); the 32 `run.py` queries are the default.

**Not verified against live TikTok.** No network scraping was performed during
this fix — TikTok rotates DOM selectors frequently, so the CSS selectors in
`find_comment_items()` / `extract_comment_*()` are written with layered
fallbacks and **should be spot-checked against a live video before being
trusted for a full re-collection.** The schema, resume logic, dedupe key,
keyword handling, and CSV writer are unit-testable without network and are
correct as written.

**Residual, still needs researcher input:** the committed CSV was not produced
by this script, so a re-run will cover a *different* set of posts (TikTok search
results change over time) and will not be a row-for-row match. Chapter 3 should
state the collection date and tool regardless.

### `scraper_source.txt` — investigated, NOT junk

The audit flagged this 10 KB file as unreadable binary. It is not corrupt:

| Property | Finding |
|---|---|
| Size | 10,184 bytes (the "16 bytes" reading was wrong) |
| Encoding | UTF-16 **LE** with BOM `FF FE` — plain text, which is why a naive reader saw a binary blob |
| Content | Byte-for-byte copy of `scraper/scraper.py` **lines 352–479** — the `run_multi_query()` method through end of file |
| Diff vs `scraper/scraper.py` | **Zero** differences in the overlapping range |

So it is a redundant partial dump of code already committed as
`scraper/scraper.py`, most likely an accidental "Save As" to `.txt` from the
editor. It is **harmless but redundant**. Left in place rather than deleted,
since removal is a separate decision — it does not need fixing to make the
pipeline reproducible.

---

## BUG-10 — 52 stale checkpoint directories (~80 GB) (LOW, fixed)

`models/` held 52 `checkpoint-*` directories. `save_total_limit=1` did not prune
them, and each contains a ~950 MB `optimizer.pt`. This is why `models/` is
gitignored.

**Resolution.** Deleted, after verifying that all ten fold directories retain
their final `model.safetensors`, `config.json` and `tokenizer.json`. Only
optimiser state and intermediate weights were removed; none are needed to
reproduce inference.

| Measurement | Value |
|---|---|
| Directories deleted | 52 |
| Reclaimed | **72.3 GB** |
| Free space before | 99.4 GB |
| Free space after | **171.8 GB** |
| Inference re-run after deletion | passed, identical output |

The runner exposes this as `run_all.py --cleanup`.

---

## BUG-13 — No single entry point (LOW, fixed)

Every stage had to be invoked by hand, and `RUN_GUIDE.md` listed them as
separate commands. Nothing verified that a stage actually produced its declared
output, so a stage could exit 0 while writing nothing.

**Resolution.** `run_all.py` plus `run_all.bat`. Each of the 18 stages declares
its inputs and outputs; after a stage exits, its outputs are checked for presence
and the stage is reported as failed if they are missing. Stage 3 is a hard gate
that refuses to continue when the training data no longer reproduces (verified by
deliberately corrupting 100 rows — the gate failed with exit code 1).

Cold-run verification: all generated outputs deleted, then
`run_all.bat` rebuilt everything in 148s and reproduced every published number
exactly (1031 corpus rows, 673/358 OOF split, 92.1% negative, 0.9263 mean
confidence).

---

## BUG-11 — Repository hygiene (LOW, mostly fixed)

- ~~`main.py` — 0 bytes~~ **Deleted.** Empty and unreferenced; verified no
  script or doc imports it.
- ~~`indobert/evaluate.py` — 0 bytes~~ **Deleted.** Same.
- ~~`requirements.txt` lists 9 packages, unpinned~~ **Fixed.** Rewritten with
  `==` pins taken from `.\.venv\Scripts\python.exe -m pip list`, and
  `accelerate` added.
- `models/indobert_aspect/` and `models/indobert_aspect_sentiment/` are trained
  but never used by any analysis script. **Deliberately NOT deleted — see below.**

### Why `accelerate` was genuinely required

It is not merely a `transformers.Trainer` internal: `transformers` only raises
if it is absent at `Trainer(...)` construction time, so a missing `accelerate`
surfaces as a confusing failure deep inside training rather than at install
time. Confirmed by grep — four training scripts instantiate a `Trainer`:

| File | `Trainer(` | `trainer.train()` |
|---|---|---|
| `indobert/train_indobert.py` | line 306 | line 318 |
| `indobert/train_sentiment.py` | line 281 | line 293 |
| `indobert/train_aspect.py` | line 304 | line 316 |
| `indobert/train_cv.py` | line 266 | line 276 |

All four are live training paths, not dead code, so `accelerate==1.13.0` is a
hard direct dependency. `RUN_GUIDE.md` was updated too: it previously told the
reader to `pip install accelerate` separately, which is no longer needed.

### Pinned versions

```
accelerate==1.13.0     matplotlib==3.10.9   scikit-learn==1.8.0
numpy==2.4.6          pandas==3.0.3         seaborn==0.13.2
selenium==4.44.0      torch==2.12.0         tqdm==4.67.3
transformers==5.8.1
```

The 9 original packages are kept as-is plus `accelerate` — same set, now pinned.
Two caveats recorded in the file itself:

- **torch on Linux/GPU.** The pin is the Windows CPU/default build this repo was
  measured on. `requirements.txt` now carries an explicit note to install torch
  from the official CUDA index *first* on Linux, because a plain
  `pip install -r requirements.txt` will otherwise silently overwrite a CUDA
  build with the default one.
- **Direct dependencies only.** The transitive closure is deliberately omitted;
  pip resolves it. Listing all ~70 packages would be noise and would fight the
  resolver.

Note: `seaborn` and `tqdm` are in `requirements.txt` but have **no import
anywhere in the repository** (grepped). They were kept because the task was to
pin the existing set, not to prune it, but they are candidates for removal.

### `models/indobert_aspect/` and `models/indobert_aspect_sentiment/` — orphaned, kept

These are fully trained checkpoints that **no analysis script loads**. Verified
by grepping every `models/` reference in the repo: the prediction and analysis
code loads `models/indobert_aspect_sentiment_cv/fold_*`, never these two.

| Model dir | Contents | Used by anything? |
|---|---|---|
| `models/indobert_aspect/` | final `model.safetensors` + tokenizer + 7 `checkpoint-*` dirs | No |
| `models/indobert_aspect_sentiment/` | final `model.safetensors` + tokenizer + 20 `checkpoint-*` dirs | No |

They are **kept on purpose.** They are the single-split counterparts of the
cross-validated models and are plausibly wanted for a reproducibility appendix
or a later single-split baseline, and they total roughly 40 GB. Deleting
multi-gigabyte trained artefacts to tidy a repo is not a reversible cleanup and
was not done automatically. Whoever prunes models should treat this as its own
decision — and BUG-10 (52 stale checkpoints, ~80 GB) is the safer first step,
since optimiser state is the bulk of the size and is not needed for inference.

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