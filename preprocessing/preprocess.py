"""
Stage 1 -- Corpus preparation.

Reads the raw scrape and produces `data/processed/comments_clean.csv`, the full
analysable corpus that every later stage reads from.

Changes from the previous version (see docs/BUG_TRACKER.md):

  BUG-01  Deduplication collapsed different users. The old code used
          `drop_duplicates(subset=["clean_text"])`, so two different accounts
          posting the same sentence were treated as one row and 9 legitimately
          distinct opinions were discarded. Now keyed on
          (username, clean_text).

  BUG-02  The length filter destroyed short but meaningful comments. The old
          threshold was `len(clean_text) > 10`, but a trailing TikTok date stamp
          counts towards the length, so "pecat 4-25" scored 11 and "setuju"
          scored 6 and was thrown away. 24 comments carrying real sentiment
          ("setuju", "parah", "pecat", "marwah", "usut", "keren") were lost.
          The stamp is now stripped before the length decision.

  BUG-03  Slang expansion changed the text handed to the model. The models in
          `models/` were fine-tuned on text produced WITHOUT the slang step.
          Expanding slang here created a train/inference mismatch. The
          model-input column is now byte-identical to the training column.

Run:  python -m preprocessing.preprocess
"""

import pandas as pd
from pathlib import Path

from preprocessing.text_cleaning import (
    MIN_LENGTH,
    is_analysable,
    normalize,
    normalize_for_gate,
    strip_date_stamp,
)

RAW_PATH = Path("data/raw/tiktok_comments_mobil_dinas_kaltim.csv")
OUTPUT_PATH = Path("data/processed/comments_clean.csv")

# Text-column candidates, tried in order.
TEXT_COLUMNS = [
    "comment_text",
    "text",
    "tweet",
    "content",
    "full_text",
    "reply_text",
    "komentar",
]

USER_COLUMNS = ["comment_username", "username", "user", "author"]


def find_column(df, candidates, label):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Kolom {label} tidak ditemukan. "
        f"Kolom tersedia: {list(df.columns)}"
    )


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"File raw tidak ditemukan: {RAW_PATH}\n"
            "Jalankan scraper terlebih dahulu."
        )

    df = pd.read_csv(RAW_PATH)
    total_input = len(df)
    print(f"Input data          : {total_input}")

    text_col = find_column(df, TEXT_COLUMNS, "teks")
    user_col = find_column(df, USER_COLUMNS, "user")

    df["original_text"] = df[text_col].astype(str)

    # Model-input form: byte-identical to the fine-tuning data. BUG-03.
    df["clean_text"] = df["original_text"].map(normalize)

    # Content used for the inclusion decision, with the TikTok date stamp
    # removed first. BUG-02.
    df["content_text"] = df["original_text"].map(normalize_for_gate)

    df["has_date_stamp"] = df["original_text"].map(
        lambda t: str(t) != strip_date_stamp(t)
    )

    after_len = len(df)
    df = df[df["content_text"].str.len() >= MIN_LENGTH].copy()
    dropped_len = after_len - len(df)

    # Deduplicate on (user, text) so distinct accounts stay distinct. BUG-01.
    after_dedup = len(df)
    df = df.drop_duplicates(subset=[user_col, "clean_text"]).reset_index(drop=True)
    dropped_dup = after_dedup - len(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Dibuang (terlalu pendek) : {dropped_len}")
    print(f"Dibuang (duplikat user)  : {dropped_dup}")
    print(f"Output data         : {len(df)}")
    print(f"Tersimpan di        : {OUTPUT_PATH}")
    print(
        f"\nKolom output       : {list(df.columns)}\n"
        f"Catatan: 'clean_text' = input model (sinkron dengan data latih).\n"
        f"        'content_text' = dasar keputusan kelayakan komentar."
    )


if __name__ == "__main__":
    main()