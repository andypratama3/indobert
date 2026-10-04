import re
import pandas as pd
from pathlib import Path

INPUT_PATH = Path(
    "data/annotation/dataset_train.csv"
)

OUTPUT_PATH = Path(
    "data/processed/dataset_train_clean.csv"
)


def clean_text(text):

    text = str(text)

    # Case folding
    text = text.lower()

    # Hapus URL
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Hapus mention
    text = re.sub(r"@\w+", " ", text)

    # Hapus hashtag (#)
    text = re.sub(r"#\w+", " ", text)

    # Hapus angka
    text = re.sub(r"\d+", " ", text)

    # Hapus karakter khusus
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)

    # Hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():

    print("PREPROCESSING DATA ABSA")

    df = pd.read_csv(INPUT_PATH)

    print(
        f"Jumlah data awal: {len(df)}"
    )

    df["clean_text"] = (
        df["original_text"]
        .fillna("")
        .astype(str)
        .apply(clean_text)
    )

    df = df[
        df["clean_text"].str.len() > 0
    ].copy()

    df = df.reset_index(
        drop=True
    )

    df["id"] = range(
        1,
        len(df) + 1
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"Jumlah data akhir: {len(df)}"
    )

    print(
        f"Hasil disimpan: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()