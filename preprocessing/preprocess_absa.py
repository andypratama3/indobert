"""
Stage 3 -- Training data preparation.

Reads the annotated dataset and produces `data/processed/dataset_train_clean.csv`,
the file the fine-tuned models were trained on.

BUG-05 / BUG-12. The previous version of this script used its own inline cleaner
and silently overwrote the canonical training file with text formatted
differently from what the models in `models/` had seen. Re-running it destroyed
the training column: 800 rows became 799 and byte-exact reproduction of the
stored column dropped to 94/800.

This version now:

  * uses `preprocessing.text_cleaning.normalize()`, which reproduces the stored
    training column byte-for-byte (verified 800/800), so the script is idempotent
    and safe to re-run;

  * performs NO row filtering, because the annotated set is the curated sample and
    silently dropping a row changes the training distribution. The previous
    `len(clean_text) > 0` filter removed one row.

Run:  python -m preprocessing.preprocess_absa

Verify afterwards -- this must print 800 / 800:

    python -c "import pandas as pd; from preprocessing.text_cleaning import \
normalize; d=pd.read_csv('data/processed/dataset_train_clean.csv'); \
print((d['original_text'].map(normalize)==d['clean_text']).sum(), '/', len(d))"
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.text_cleaning import normalize  # noqa: E402

INPUT_PATH = Path("data/annotation/dataset_train.csv")
OUTPUT_PATH = Path("data/processed/dataset_train_clean.csv")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} tidak ditemukan. "
            "Anotasi manual harus selesai terlebih dahulu."
        )

    df = pd.read_csv(INPUT_PATH)
    print("PREPROCESSING DATA TRAINING")
    print(f"Jumlah data awal : {len(df)}")

    df["clean_text"] = df["original_text"].fillna("").astype(str).map(normalize)

    # No filtering. See module docstring.
    df = df.reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)

    reproduced = int(
        (df["original_text"].astype(str).map(normalize) == df["clean_text"]).sum()
    )
    print(f"Reproduksi kolom latih : {reproduced}/{len(df)}")
    if reproduced != len(df):
        raise AssertionError(
            "Kolom clean_text tidak reproduces data latih. "
            "Membersihkan ulang akan merusak model -- jangan diabaikan."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Jumlah data akhir : {len(df)}")
    print(f"Hasil disimpan    : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()