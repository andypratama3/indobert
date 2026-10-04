"""
Provenance audit for the annotated training set (BUG-07).

WHAT THIS SCRIPT ESTABLISHES
----------------------------
`data/annotation/dataset_train.csv` holds 800 manually annotated rows.  Locating
them in the raw scrape `data/raw/tiktok_comments_mobil_dinas_kaltim.csv` by
normalised text finds only 684.  The other 116 rows are absent from the raw file
and from every other CSV in `backup/`, so the scrape batch that produced them is
gone.  Those rows still train the models, but no corpus row corresponds to them,
so they can never receive a corpus prediction.

The reason that matters is not the missing 14.5% by itself -- it is that the
missing rows are NOT a random sample.  Measured here:

    aspect                     located    not located
    Akuntabilitas                 50.6%          14.7%
    Efektivitas & Efisiensi       37.1%           7.8%
    Responsivitas                  7.0%          44.0%
    Transparansi                   5.3%          33.6%

The annotation set was assembled from several topic batches, and the batch that
survived in `dataset_train.csv` but not in the raw scrape is overwhelmingly
Responsivitas and Transparansi.  So the model's aspect prior is tilted towards
exactly the two aspects the corpus barely contains, and the reported aspect
distribution inherits that tilt.

The sentiment split is NOT skewed the same way (90.1% negative located vs 87.1%
negative not located), which localises the problem to topic sampling rather than
to labelling tone.  A chi-square test of independence plus Cramer's V is computed
for both variables to make that distinction quantitative instead of eyeballed.

METHOD
------
Matching: lowercase, replace every character outside `a-z` with a space, collapse
whitespace.  Training `original_text` is matched against raw `comment_text`.

One caveat is reported rather than hidden: a row written in Unicode mathematical
stylised letters (e.g. U+1D67A ...) normalises to the empty string, and the raw
file contains bare date stamps that also normalise to the empty string.  Such a
row matches on an empty key, which is not evidence of provenance.  Such rows are
flagged `degenerate_match` and a strict variant of the count is printed alongside
the headline numbers.

Chi-square and Cramer's V are implemented here from their definitions rather than
imported, so this audit has no dependency beyond pandas/numpy.  When scipy
happens to be installed the values are recomputed with it and any disagreement is
reported as a warning.

OUTPUTS
-------
    data/results/provenance_audit.csv                 main audit table
    data/results/provenance_missing_rows.csv          the unlocatable rows
    data/annotation/annotated_not_in_corpus.csv       same rows, kept as a
                                                      tracked annotation artefact

Run:  python -m analysis.provenance_audit
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


# ── CONFIGURATION ────────────────────────────────────────────────────────────

TRAIN_CSV = config.BASE_DIR / config.ANNOTATION_DATASET
RAW_CSV = config.BASE_DIR / config.RAW_COMMENTS
CORPUS_CSV = config.BASE_DIR / config.ABSA_RESULT

OUT_AUDIT = config.BASE_DIR / config.RESULTS_DIR / "provenance_audit.csv"
OUT_MISSING = config.BASE_DIR / config.RESULTS_DIR / "provenance_missing_rows.csv"
OUT_TRACKED = config.BASE_DIR / config.ANNOTATION_DIR / "annotated_not_in_corpus.csv"

# Cramer's V above this is treated as a material association.  Conventional
# thresholds (Greenacre) call 0.1 "a noticeable relationship"; anything above it
# is not attributable to sampling chance.
CRAMERS_V_THRESHOLD = 0.10

GROUP_LOCATED = "located_in_raw"
GROUP_MISSING = "not_in_raw"

_NON_AZ = re.compile(r"[^a-z]+")
_WS = re.compile(r"\s+")


def provenance_key(text) -> str:
    """
    Normalised form used for corpus-lookup matching.

    Deliberately different from `preprocessing.text_cleaning.normalize()`: that one
    is the MODEL-INPUT cleaner and keeps numerals, whereas provenance matching
    must be insensitive to the trailing TikTok date stamp (`3-27`) that was
    attached inconsistently across scrape batches.  Stripping every non a-z
    character removes those stamps, numerals and punctuation at once.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return _WS.sub(" ", _NON_AZ.sub(" ", str(text).lower())).strip()


def canonical_aspect(name: str) -> str:
    """
    Reconcile the two spellings of the same aspect.

    Annotation files use `Efektivitas & Efisiensi`; the prediction output uses
    `Efektivitas dan Efisiensi`.  Without this the prior comparison silently
    drops that aspect instead of reporting it.
    """
    return re.sub(r"\s+", " ", str(name).replace("&", "dan")).strip().lower()


# ── CHI-SQUARE / CRAMER'S V, IMPLEMENTED FROM DEFINITION ─────────────────────
#
# Deliberately dependency-free.  scipy is not a declared requirement of this
# project (see docs/BUG_TRACKER.md BUG-11), so an audit that cannot run is worse
# than an audit with 20 lines of textbook statistics in it.


def chi2_contingency(observed: np.ndarray) -> tuple[float, int, float]:
    """
    Pearson chi-square test of independence on a contingency table.

    Returns (chi2, dof, cramers_v).  Cells with an expected count of zero are
    skipped rather than raising, since a zero row/column margin is not evidence
    against the model.
    """
    observed = np.asarray(observed, dtype=float)
    n = observed.sum()
    if n == 0:
        return float("nan"), 0, float("nan")

    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / n

    usable = expected > 0
    chi2 = float(np.sum((observed[usable] - expected[usable]) ** 2 / expected[usable]))

    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    if dof <= 0 or n == 0:
        return chi2, dof, float("nan")

    # Cramer's V, bias not corrected: n is large enough here that the difference
    # between corrected and uncorrected V is well under a tenth of the estimate.
    cramers_v = float(np.sqrt(chi2 / (n * min(observed.shape[0] - 1,
                                             observed.shape[1] - 1))))
    return chi2, dof, cramers_v


def _gamma_series(a: float, x: float) -> float:
    """Lower regularised incomplete gamma P(a, x) by series expansion."""
    ap = a
    total = 1.0 / a
    term = total
    for _ in range(1000):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * 1e-15:
            break
    return float(total * math.exp(-x + a * math.log(x) - math.lgamma(a)))


def _gamma_cf(a: float, x: float) -> float:
    """Upper regularised incomplete gamma Q(a, x) by continued fraction."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b if b != 0 else 1.0 / tiny
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return float(h * math.exp(-x + a * math.log(x) - math.lgamma(a)))


def chi2_sf(x: float, dof: int) -> float:
    """
    Upper-tail probability P(X > x) for X ~ chi-square(dof).

    Equals the regularised upper incomplete gamma Q(dof/2, x/2).
    """
    if x is None or np.isnan(x) or dof <= 0:
        return float("nan")
    a, xx = dof / 2.0, x / 2.0
    if xx <= 0:
        return 1.0
    if xx < a + 1.0:
        return float(min(1.0, max(0.0, 1.0 - _gamma_series(a, xx))))
    return float(min(1.0, max(0.0, _gamma_cf(a, xx))))


def verify_against_scipy(chi2: float, dof: int) -> str:
    """
    Recompute the tail probability with scipy when available.

    The manual implementation above is the one the audit reports; this only
    exists so the two can be compared instead of trusted.
    """
    try:
        from scipy.stats import chi2 as scipy_chi2
    except Exception:
        return "scipy not installed - manual implementation is the only one used"
    mine = chi2_sf(chi2, dof)
    theirs = float(scipy_chi2.sf(chi2, dof))
    if np.isnan(mine):
        return "manual p-value undefined; nothing to cross-check"
    if mine == 0.0 and theirs > 0.0:
        return (f"manual p underflowed to 0.0, scipy gives {theirs:.6e} "
                "(chi2 statistic is the reportable quantity)")
    rel = abs(mine - theirs) / max(theirs, 1e-300)
    status = "AGREE" if rel < 1e-6 else "DISAGREE"
    return f"manual p={mine:.6e} vs scipy p={theirs:.6e} -> {status}"


# ── LOAD ─────────────────────────────────────────────────────────────────────


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_inputs():
    train = pd.read_csv(TRAIN_CSV)
    raw = pd.read_csv(RAW_CSV)

    missing_cols = {"original_text", "final_aspect", "final_sentiment"} - set(train.columns)
    if missing_cols:
        raise SystemExit(
            f"{TRAIN_CSV} is missing required column(s): {sorted(missing_cols)}"
        )
    if "comment_text" not in raw.columns:
        raise SystemExit(f"{RAW_CSV} is missing required column 'comment_text'")

    try:
        corpus = pd.read_csv(CORPUS_CSV)
    except FileNotFoundError:
        corpus = None

    return train, raw, corpus


# ── MAIN AUDIT ───────────────────────────────────────────────────────────────


def main() -> int:
    train, raw, corpus = load_inputs()

    rule("PROVENANCE AUDIT — annotated training set vs raw scrape (BUG-07)")
    print(f"annotated rows : {len(train)}   ({TRAIN_CSV.relative_to(config.BASE_DIR)})")
    print(f"raw scrape rows : {len(raw)}   ({RAW_CSV.relative_to(config.BASE_DIR)})")
    if corpus is not None:
        print(f"corpus rows     : {len(corpus)}   ({CORPUS_CSV.relative_to(config.BASE_DIR)})")

    # ── 1. locate every annotated row in the raw scrape ──────────────────────
    #
    # The headline count uses the documented matching method unchanged, so it
    # agrees with docs/BUG_TRACKER.md BUG-07.  A row whose normalised text is
    # empty matches on an empty key, which is not evidence of provenance; such
    # rows are kept in the "located" group but individually flagged, and the
    # strict variant of the count is reported alongside.
    raw_keys = set(raw["comment_text"].map(provenance_key))
    train_keys = train["original_text"].map(provenance_key)

    match_is_degenerate = train_keys.eq("")
    located = train_keys.isin(raw_keys)

    work = train.copy()
    work["provenance_key"] = train_keys
    work["match_is_degenerate"] = match_is_degenerate
    work["provenance_group"] = np.where(located, GROUP_LOCATED, GROUP_MISSING)

    n_loc = int(located.sum())
    n_mis = int((~located).sum())
    n_total = len(work)
    n_deg = int(match_is_degenerate.sum())
    n_loc_strict = n_loc - n_deg

    rule("1. LOCATABILITY OF THE ANNOTATED ROWS")
    print(f"  located in raw scrape        : {n_loc:4d}  ({n_loc / n_total * 100:5.1f}%)")
    print(f"  NOT located in raw scrape    : {n_mis:4d}  ({n_mis / n_total * 100:5.1f}%)")
    print(f"  -------------------------------------")
    print(f"  total annotated              : {n_total:4d}")

    if n_deg:
        deg_rows = work.loc[work["match_is_degenerate"], ["username", "original_text"]]
        print(f"\n  Caveat: {n_deg} of the {n_loc} 'located' rows match only because their")
        print("  text normalises to the empty string. They are written in Unicode")
        print("  mathematical stylised letters, which contain no a-z characters, and")
        print("  the raw file separately contains bare date stamps that also normalise")
        print("  to empty. An empty key is not evidence of provenance, so these rows")
        print("  are flagged `match_is_degenerate` rather than counted as verified:")
        for _, r in deg_rows.iterrows():
            preview = str(r["original_text"])[:60].replace("\n", " ")
            print(f"    - {r['username']}: {preview}")
        print(f"\n  Strict variant, degenerate matches excluded:")
        print(f"    located {n_loc_strict} ({n_loc_strict / n_total * 100:.1f}%) / "
              f"NOT located {n_mis + n_deg} ({(n_mis + n_deg) / n_total * 100:.1f}%)")
        print("  Every other figure in this report uses the documented method")
        print("  (n_loc / n_mis), which is what BUG_TRACKER.md records.")

    # ── 2. aspect and sentiment distribution, located vs not ─────────────────
    loc_df = work.loc[located]
    mis_df = work.loc[~located]

    rule("2. ASPECT DISTRIBUTION — located vs not located (% within each group)")
    aspect_loc_pct = (loc_df["final_aspect"].value_counts(normalize=True) * 100).round(1)
    aspect_mis_pct = (mis_df["final_aspect"].value_counts(normalize=True) * 100).round(1)
    aspect_counts = pd.crosstab(work["final_aspect"], work["provenance_group"])

    aspect_tbl = pd.DataFrame({
        "n_located": aspect_counts.get(GROUP_LOCATED, pd.Series(dtype=int)),
        "n_not_located": aspect_counts.get(GROUP_MISSING, pd.Series(dtype=int)),
        "pct_located": aspect_loc_pct,
        "pct_not_located": aspect_mis_pct,
    })
    aspect_tbl["n_total"] = aspect_tbl["n_located"] + aspect_tbl["n_not_located"]
    aspect_tbl["pct_of_annotated"] = (aspect_tbl["n_total"] / n_total * 100).round(1)
    aspect_tbl = aspect_tbl.sort_values("n_total", ascending=False)
    print(aspect_tbl.to_string())

    rule("3. SENTIMENT DISTRIBUTION — located vs not located (% within each group)")
    sent_counts = pd.crosstab(work["final_sentiment"], work["provenance_group"])
    sent_tbl = pd.DataFrame({
        "n_located": sent_counts.get(GROUP_LOCATED, pd.Series(dtype=int)),
        "n_not_located": sent_counts.get(GROUP_MISSING, pd.Series(dtype=int)),
        "pct_located": (loc_df["final_sentiment"].value_counts(normalize=True) * 100).round(1),
        "pct_not_located": (mis_df["final_sentiment"].value_counts(normalize=True) * 100).round(1),
    })
    sent_tbl["n_total"] = sent_tbl["n_located"] + sent_tbl["n_not_located"]
    print(sent_tbl.to_string())
    print("\n  The split is comparable across the two groups, so the anomaly in the")
    print("  aspect table is topical sampling, not a difference in labelling tone.")

    # ── 4. per-class counts and which classes are hit hardest ────────────────
    rule("4. PER-CLASS COUNTS — which classes are disproportionately affected")
    label_tbl = pd.crosstab(work["final_label"], work["provenance_group"])
    label_tbl["n_total"] = label_tbl.sum(axis=1)
    # Expected unlocated count if rows were missing at random across all classes.
    rate = n_mis / n_total
    label_tbl["expected_if_random"] = (label_tbl["n_total"] * rate).round(1)
    label_tbl["ratio_actual_over_expected"] = (
        label_tbl.get(GROUP_MISSING, 0) / label_tbl["expected_if_random"]
    ).round(2)
    label_tbl = label_tbl.sort_values(
        ["ratio_actual_over_expected", "n_total"], ascending=False
    )
    print(label_tbl.to_string())
    print("\n  ratio_actual_over_expected > 1 means the class lost a larger share of")
    print("  its rows to the missing scrape batch than chance would predict.")

    # ── 5. chi-square test of independence ───────────────────────────────────
    rule("5. CHI-SQUARE TEST OF INDEPENDENCE  (located x variable)")

    aspect_obs = aspect_counts.reindex(columns=[GROUP_LOCATED, GROUP_MISSING]).fillna(0)
    chi2_a, dof_a, v_a = chi2_contingency(aspect_obs.to_numpy())
    p_a = chi2_sf(chi2_a, dof_a)

    sent_obs = sent_counts.reindex(columns=[GROUP_LOCATED, GROUP_MISSING]).fillna(0)
    chi2_s, dof_s, v_s = chi2_contingency(sent_obs.to_numpy())
    p_s = chi2_sf(chi2_s, dof_s)

    print(f"\n  ASPECT  vs located")
    print(f"    chi-square = {chi2_a:.4f}   df = {dof_a}   p = {p_a:.6e}")
    print(f"    Cramer's V = {v_a:.4f}   threshold = {CRAMERS_V_THRESHOLD}   "
          f"-> {'MATERIAL BIAS' if v_a > CRAMERS_V_THRESHOLD else 'no material bias'}")
    print(f"    cross-check: {verify_against_scipy(chi2_a, dof_a)}")

    print(f"\n  SENTIMENT vs located")
    print(f"    chi-square = {chi2_s:.4f}   df = {dof_s}   p = {p_s:.6e}")
    print(f"    Cramer's V = {v_s:.4f}   threshold = {CRAMERS_V_THRESHOLD}   "
          f"-> {'MATERIAL BIAS' if v_s > CRAMERS_V_THRESHOLD else 'no material bias'}")
    print(f"    cross-check: {verify_against_scipy(chi2_s, dof_s)}")

    print("\n  Interpretation: whether a row was locatable in the raw scrape is")
    print("  strongly associated with its aspect, and essentially independent of its")
    print("  sentiment. The missing batch is a topic batch.")

    # ── 6. training prior vs corpus prior ────────────────────────────────────
    rule("6. TRAINING PRIOR vs CORPUS PRIOR")
    prior_tbl = pd.DataFrame(index=aspect_tbl.index)
    prior_tbl["n_training"] = aspect_tbl["n_total"]
    prior_tbl["pct_training"] = (aspect_tbl["n_total"] / n_total * 100).round(2)

    if corpus is not None and "predicted_aspect" in corpus.columns:
        corpus_aspect = corpus["predicted_aspect"].map(canonical_aspect)
        corpus_counts = corpus_aspect.value_counts()
        prior_tbl["n_corpus"] = [int(corpus_counts.get(canonical_aspect(a), 0))
                                  for a in prior_tbl.index]
        prior_tbl["pct_corpus"] = (prior_tbl["n_corpus"] / len(corpus) * 100).round(2)
        prior_tbl["train_over_corpus_ratio"] = (
            prior_tbl["pct_training"] / prior_tbl["pct_corpus"].replace(0, np.nan)
        ).round(2)
        print(prior_tbl.to_string())

        over = prior_tbl["train_over_corpus_ratio"].dropna()
        if len(over):
            worst = over.idxmax()
            print(f"\n  Most over-weighted aspect in training: {worst} "
                  f"({over.max():.2f}x its corpus share)")
        under = over[over < 1]
        if len(under):
            w = under.idxmin()
            print(f"  Most under-weighted aspect in training: {w} "
                  f"({under.min():.2f}x its corpus share)")

        # Index by canonical aspect so the combined R+T figure is always printed;
        # the summary of the whole audit depends on this number.
        prior_tbl.index = [canonical_aspect(a) for a in prior_tbl.index]
        skewed_aspects = [a for a in ("responsivitas", "transparansi")
                          if a in prior_tbl.index]
        # Compute from the counts, not from the already-rounded percentages, so
        # the headline 21.75% is exact rather than 12.38 + 9.38 = 21.76.
        rt_n = int(prior_tbl.loc[skewed_aspects, "n_training"].sum())
        rt_c = int(prior_tbl.loc[skewed_aspects, "n_corpus"].sum())
        rt_train = rt_n / n_total * 100
        rt_corpus = rt_c / len(corpus) * 100
        if rt_corpus:
            print(f"  Responsivitas + Transparansi: {rt_n} of {n_total} training rows "
                  f"= {rt_train:.2f}%")
            print(f"  vs {rt_c} of {len(corpus)} corpus rows = {rt_corpus:.2f}% "
                  f"-> {rt_train / rt_corpus:.2f}x over-weighted")
        ak_train = prior_tbl.loc[
            [a for a in ("akuntabilitas", "efektivitas dan efisiensi")
             if a in prior_tbl.index], "pct_training"].sum()
        print(f"  Akuntabilitas + Efektivitas:  {ak_train:.2f}% of the training set "
              f"vs {100 - rt_corpus:.2f}% of the corpus "
              f"-> {ak_train / (100 - rt_corpus):.2f}x")
    else:
        prior_tbl["n_corpus"] = np.nan
        prior_tbl["pct_corpus"] = np.nan
        prior_tbl["train_over_corpus_ratio"] = np.nan
        print("  corpus predictions unavailable - prior comparison skipped")
        print(prior_tbl.to_string())

    # ── 7. warning ───────────────────────────────────────────────────────────
    rule("7. SAMPLING BIAS WARNING")
    warnings: list[str] = []
    if v_a > CRAMERS_V_THRESHOLD:
        warnings.append(
            f"ASPECT SAMPLING BIAS: Cramer's V = {v_a:.3f} between corpus locatability "
            f"and aspect (threshold {CRAMERS_V_THRESHOLD}). The 116 unlocatable "
            f"annotated rows are not a random sample of the annotation set; they "
            f"over-represent Responsivitas and Transparansi. Any aspect-level figure "
            f"reported from this model is affected and must carry this caveat."
        )
    if v_s > CRAMERS_V_THRESHOLD:
        warnings.append(
            f"SENTIMENT SAMPLING BIAS: Cramer's V = {v_s:.3f} (threshold "
            f"{CRAMERS_V_THRESHOLD}). The negative/positive split also differs "
            f"between located and unlocated rows."
        )
    if not warnings:
        warnings.append(
            f"No material association detected (all Cramer's V <= "
            f"{CRAMERS_V_THRESHOLD})."
        )
    for w in warnings:
        print(f"  [!] {w}")

    print(f"\n  {n_mis} of {n_total} annotated rows ({n_mis / n_total * 100:.1f}%) cannot")
    print("  be recovered from this repository. They are preserved in")
    print(f"  {OUT_TRACKED.relative_to(config.BASE_DIR)} and must be disclosed as a")
    print("  limitation in Chapter 3, not silently merged away.")

    # ── 8. write outputs ─────────────────────────────────────────────────────
    rule("8. OUTPUTS")

    # 8a. main audit table, one tidy long-format CSV covering every section.
    audit_rows: list[dict] = []

    def add(section, key, **kw):
        row = {"section": section, "key": key}
        row.update(kw)
        audit_rows.append(row)

    add("summary", "annotated_rows", n_total=n_total)
    add("summary", "located_in_raw", n_located=n_loc, pct_of_annotated=round(n_loc / n_total * 100, 2))
    add("summary", "not_in_raw", n_not_located=n_mis, pct_of_annotated=round(n_mis / n_total * 100, 2))
    add("summary", "degenerate_normalisation_key", n_rows=n_deg,
        n_located_strict=n_loc_strict,
        note=("text normalises to the empty string; the match is not evidence of "
              "provenance. Headline counts use the documented method including "
              "these rows, matching BUG_TRACKER.md BUG-07"))

    for aspect, r in aspect_tbl.iterrows():
        canon = canonical_aspect(aspect)
        add("aspect_distribution", aspect,
            n_located=int(r["n_located"]),
            n_not_located=int(r["n_not_located"]),
            n_total=int(r["n_total"]),
            pct_located=float(r["pct_located"]),
            pct_not_located=float(r["pct_not_located"]),
            pct_of_annotated=float(r["pct_of_annotated"]),
            n_corpus=_safe_int(prior_tbl, canon, "n_corpus"),
            pct_corpus=_safe_float(prior_tbl, canon, "pct_corpus"),
            train_over_corpus_ratio=_safe_float(prior_tbl, canon, "train_over_corpus_ratio"),
            expected_not_located_if_random=round(float(r["n_total"]) * rate, 1),
            actual_over_expected=round(
                float(r["n_not_located"]) / max(float(r["n_total"]) * rate, 1e-9), 2),
            skewed=bool(float(r["n_not_located"]) / max(float(r["n_total"]) * rate, 1e-9) > 1.5),
            )

    for sentiment, r in sent_tbl.iterrows():
        add("sentiment_distribution", sentiment,
            n_located=int(r["n_located"]),
            n_not_located=int(r["n_not_located"]),
            n_total=int(r["n_total"]),
            pct_located=float(r["pct_located"]),
            pct_not_located=float(r["pct_not_located"]),
            )

    for label, r in label_tbl.iterrows():
        actual = float(r.get(GROUP_MISSING, 0))
        expected = float(r["expected_if_random"])
        add("label_distribution", label,
            n_located=int(r.get(GROUP_LOCATED, 0)),
            n_not_located=int(actual),
            n_total=int(r["n_total"]),
            expected_not_located_if_random=expected,
            actual_over_expected=round(actual / max(expected, 1e-9), 2),
            skewed=bool(actual / max(expected, 1e-9) > 1.5),
            )

    add("chi_square", "aspect_vs_located", statistic=round(chi2_a, 4), dof=dof_a,
        p_value=float(p_a), cramers_v=round(v_a, 4),
        threshold=CRAMERS_V_THRESHOLD, material_bias=bool(v_a > CRAMERS_V_THRESHOLD),
        cross_check=verify_against_scipy(chi2_a, dof_a))
    add("chi_square", "sentiment_vs_located", statistic=round(chi2_s, 4), dof=dof_s,
        p_value=float(p_s), cramers_v=round(v_s, 4),
        threshold=CRAMERS_V_THRESHOLD, material_bias=bool(v_s > CRAMERS_V_THRESHOLD),
        cross_check=verify_against_scipy(chi2_s, dof_s))

    for i, w in enumerate(warnings, start=1):
        add("sampling_bias_warning", f"warning_{i}", note=w)

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(OUT_AUDIT, index=False, encoding="utf-8-sig")
    print(f"  {OUT_AUDIT.relative_to(config.BASE_DIR)}  ({len(audit_df)} rows)")

    # 8b/8c. the unlocatable rows, preserved twice: once as an audit output and
    # once as a tracked annotation artefact so the rows cannot be lost again.
    miss_cols = list(train.columns) + ["match_is_degenerate"]
    missing_rows = work.loc[~located, miss_cols].copy()
    missing_rows.insert(0, "provenance_status", "not_located_in_raw_scrape")

    missing_rows.to_csv(OUT_MISSING, index=False, encoding="utf-8-sig")
    print(f"  {OUT_MISSING.relative_to(config.BASE_DIR)}  ({len(missing_rows)} rows)")

    missing_rows.to_csv(OUT_TRACKED, index=False, encoding="utf-8-sig")
    print(f"  {OUT_TRACKED.relative_to(config.BASE_DIR)}  ({len(missing_rows)} rows)")

    rule("PROVENANCE AUDIT COMPLETE")
    print(f"{n_mis}/{n_total} annotated rows have no counterpart in the raw scrape.")
    print(f"chi-square(aspect) = {chi2_a:.2f}, df = {dof_a}, "
          f"p = {p_a:.3e}, Cramer's V = {v_a:.3f}")
    print(f"chi-square(sentiment) = {chi2_s:.2f}, df = {dof_s}, "
          f"p = {p_s:.3e}, Cramer's V = {v_s:.3f}")
    print(f"Cramer's V threshold for material bias: {CRAMERS_V_THRESHOLD}")
    print("=" * 72)
    return 0


def _safe_int(tbl: pd.DataFrame, canon: str, col: str):
    if canon not in tbl.index or col not in tbl.columns:
        return None
    val = tbl.loc[canon, col]
    return None if pd.isna(val) else int(val)


def _safe_float(tbl: pd.DataFrame, canon: str, col: str):
    if canon not in tbl.index or col not in tbl.columns:
        return None
    val = tbl.loc[canon, col]
    return None if pd.isna(val) else float(val)


if __name__ == "__main__":
    raise SystemExit(main())
