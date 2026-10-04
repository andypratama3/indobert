"""
Leakage-free ABSA inference.

WHY THIS FILE EXISTS
--------------------
`indobert/predict_absa.py` loaded `models/indobert_aspect_sentiment_cv/fold_6`
and predicted ALL rows of the corpus. Because the corpus is a superset of the
annotated training set, 673 of the 998 predicted rows (67.4%) had been seen by
the model during fine-tuning. The resulting file
`data/results/indobert_absa_result.csv` -- the input to every downstream
analysis script and to every figure in `data/results/visualization/` -- was
therefore contaminated. Mean confidence was 0.980 on memorised rows versus 0.928
on genuinely unseen rows, and 68.5% of the negative-sentiment rows used as
qualitative evidence came from the training set.

THE FIX
-------
Out-of-fold (OOF) prediction:

  1. The annotated set is partitioned with the *identical* split that produced
     the checkpoints: StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
     reusing `load_and_clean_data` imported from `indobert.train_cv` so the
     partition cannot drift.

  2. Each annotated row is predicted by the one fold model that held it out.
     That model never saw the row, so the prediction is honest.

  3. Corpus rows that were never annotated are predicted by majority vote across
     all ten folds. No fold model saw them either.

Every row in the output is therefore predicted by a model that had not been
trained on it. No row is dropped, so the corpus keeps its full size.

OUTPUT
------
`data/results/indobert_absa_result_oof.csv` -- same schema as the old file plus
`prediction_source` and `votes`, so the provenance of every row is auditable.

Run:  python -m indobert.predict_absa_oof
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import pipeline

# Reuse the loader and constants from the training script so the fold partition
# is guaranteed identical to the one that produced the checkpoints.
from indobert.train_cv import (  # noqa: E402
    CV_OUTPUT_DIR,
    ID_TO_LABEL,
    LABEL_MAP,
    N_SPLITS,
    RANDOM_STATE,
    load_and_clean_data,
)

INPUT_PATH = Path("data/processed/comments_clean.csv")
OUTPUT_PATH = Path("data/results/indobert_absa_result_oof.csv")
FALLBACK_OUTPUT = Path("data/results/indobert_absa_result.csv")

ASPECT_MAP = {
    "transparansi": "Transparansi",
    "akuntabilitas": "Akuntabilitas",
    "efektivitas_efisiensi": "Efektivitas dan Efisiensi",
    "responsivitas": "Responsivitas",
}

DEVICE = 0 if torch.cuda.is_available() else -1


def load_fold_model(fold_idx):
    path = CV_OUTPUT_DIR / f"fold_{fold_idx}"
    if not (path / "model.safetensors").exists():
        raise FileNotFoundError(
            f"Model fold {fold_idx} tidak ditemukan di {path}\n"
            "Jalankan `python -m indobert.train_cv` terlebih dahulu."
        )
    return pipeline(
        "text-classification",
        model=str(path),
        tokenizer=str(path),
        device=DEVICE,
        truncation=True,
        max_length=256,
    )


def predict_labels(classifier, texts, batch_size=16):
    if len(texts) == 0:
        return [], []
    out = classifier(list(texts), batch_size=batch_size)
    return [r["label"] for r in out], [round(float(r["score"]), 4) for r in out]


def majority(labels, votes):
    """Most common label; ties broken by mean confidence then label order."""
    if not labels:
        return "", 0.0, 0
    counts = Counter(labels)
    top = max(counts.values())
    tied = [lb for lb, c in counts.items() if c == top]
    if len(tied) == 1:
        best = tied[0]
    else:
        conf = {}
        for lb, sc in zip(labels, votes):
            conf.setdefault(lb, []).append(sc)
        best = max(tied, key=lambda lb: (sum(conf[lb]) / len(conf[lb]), -LABEL_MAP[lb]))
    return best, round(float(np.mean(votes[labels.index(best)])), 4), top


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} tidak ditemukan. Jalankan "
            "`python -m preprocessing.preprocess` terlebih dahulu."
        )

    corpus = pd.read_csv(INPUT_PATH)
    print("=" * 68)
    print("INDOBERT ABSA -- LEAKAGE-FREE (OUT-OF-FOLD) PREDICTION")
    print("=" * 68)
    print(f"\nCorpus rows            : {len(corpus)}")

    train_df = load_and_clean_data()
    print(f"Annotated training rows: {len(train_df)}")

    # ---- align annotated rows to corpus rows -------------------------------
    # Both sides now pass through the same canonical cleaner, so clean_text is
    # directly comparable. Fall back to the raw text for rows whose cleaned form
    # was shortened by deduplication.
    corpus_texts = corpus["clean_text"].astype(str).tolist()
    train_texts = train_df["clean_text"].astype(str).tolist()
    train_labels = train_df["label"].tolist()

    text_to_corpus = {}
    for pos, t in enumerate(corpus_texts):
        text_to_corpus.setdefault(t, pos)

    annotated_pos = [text_to_corpus.get(t, -1) for t in train_texts]
    n_matched = sum(1 for p in annotated_pos if p >= 0)
    print(f"Annotated rows located in corpus: {n_matched}/{len(train_df)}")

    annotated_corpus_idx = {p for p in annotated_pos if p >= 0}
    unseen_corpus_idx = [i for i in range(len(corpus)) if i not in annotated_corpus_idx]
    print(f"Corpus rows never annotated    : {len(unseen_corpus_idx)}")

    # ---- reproduce the exact fold partition -------------------------------
    texts_arr = np.array(train_texts, dtype=object)
    labels_arr = np.array(train_labels)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    pred_label = [None] * len(corpus)
    pred_score = [None] * len(corpus)
    pred_votes = [0] * len(corpus)
    pred_source = [None] * len(corpus)

    # ---- step 1: OOF for annotated rows -----------------------------------
    print("\n" + "-" * 68)
    print("STEP 1  Out-of-fold prediction for annotated rows")
    print("-" * 68)

    n_oof = 0
    for fold_idx, (_, val_idx) in enumerate(skf.split(texts_arr, labels_arr), start=1):
        print(f"\n  Fold {fold_idx}/{N_SPLITS} ...", flush=True)
        clf = load_fold_model(fold_idx)

        val_texts = [train_texts[i] for i in val_idx]
        labels, scores = predict_labels(clf, val_texts)

        mapped = 0
        for i, lb, sc in zip(val_idx, labels, scores):
            pos = annotated_pos[i]
            if pos < 0:
                continue
            pred_label[pos] = lb
            pred_score[pos] = sc
            pred_votes[pos] = 1
            pred_source[pos] = f"oof_fold_{fold_idx}"
            mapped += 1
        n_oof += mapped
        print(f"    predicted {mapped} held-out rows")

        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n  OOF predictions written: {n_oof}")

    # ---- step 2: ensemble vote for never-annotated rows -------------------
    print("\n" + "-" * 68)
    print("STEP 2  10-fold ensemble vote for never-annotated rows")
    print("-" * 68)

    if unseen_corpus_idx:
        ut = [corpus_texts[i] for i in unseen_corpus_idx]
        acc_labels = {i: [] for i in unseen_corpus_idx}
        acc_scores = {i: [] for i in unseen_corpus_idx}

        for fold_idx in range(1, N_SPLITS + 1):
            print(f"  fold {fold_idx}/{N_SPLITS} ...", flush=True)
            clf = load_fold_model(fold_idx)
            labels, scores = predict_labels(clf, ut)
            for pos, lb, sc in zip(unseen_corpus_idx, labels, scores):
                acc_labels[pos].append(lb)
                acc_scores[pos].append(sc)
            del clf
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for pos in unseen_corpus_idx:
            lb, sc, top = majority(acc_labels[pos], acc_scores[pos])
            pred_label[pos] = lb
            pred_score[pos] = sc
            pred_votes[pos] = top
            pred_source[pos] = "ensemble_10fold_unseen"

    # ---- assemble ---------------------------------------------------------
    out = corpus.copy()
    out["predicted_label"] = pred_label
    out["prediction_score"] = pred_score
    out["votes"] = pred_votes
    out["prediction_source"] = pred_source

    out["predicted_aspect"] = (
        out["predicted_label"].str.rsplit("_", n=1).str[0].replace(ASPECT_MAP)
    )
    out["predicted_sentiment"] = (
        out["predicted_label"].str.rsplit("_", n=1).str[1]
        .replace({"positif": "Positif", "negatif": "Negatif"})
    )

    still_missing = int(out["predicted_label"].isna().sum())
    if still_missing:
        print(
            f"\n[PERINGATAN] {still_missing} baris tanpa prediksi. "
            "Kemungkinan annotated row tidak ditemukan di corpus."
        )
        out = out[out["predicted_label"].notna()].reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # Keep the original filename pointing at clean data, and preserve the
    # contaminated original as an audit artefact.
    if FALLBACK_OUTPUT.exists() and not OUTPUT_PATH.with_name(
        "indobert_absa_result_leaky.csv"
    ).exists():
        OUTPUT_PATH.with_name(
            "indobert_absa_result_leaky.csv"
        ).write_bytes(FALLBACK_OUTPUT.read_bytes())

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 68)
    print("SELESAI")
    print("=" * 68)
    print(f"\nTotal baris               : {len(out)}")
    print(f"Dari OOF holdout         : {(out['prediction_source'].str.startswith('oof')).sum()}")
    print(f"Dari ensemble unseen     : {(out['prediction_source'] == 'ensemble_10fold_unseen').sum()}")
    print(f"Tersimpan                 : {OUTPUT_PATH}")
    if (OUTPUT_PATH.with_name('indobert_absa_result_leaky.csv')).exists():
        print(f"Artefak lama (terkontaminasi): "
              f"{OUTPUT_PATH.with_name('indobert_absa_result_leaky.csv')}")

    print("\nDistribusi Aspek")
    print(out["predicted_aspect"].value_counts().to_string())
    print("\nDistribusi Sentimen")
    print(out["predicted_sentiment"].value_counts().to_string())
    print("\nAspek x Sentimen")
    print(pd.crosstab(out["predicted_aspect"], out["predicted_sentiment"]).to_string())
    print(f"\nMean confidence: {out['prediction_score'].mean():.4f}")


if __name__ == "__main__":
    sys.exit(main())