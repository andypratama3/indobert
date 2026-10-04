import pandas as pd
from pathlib import Path


INPUT_PATH = Path("data/processed/comments_clean.csv")
OUTPUT_PATH = Path("data/processed/comments_relevant.csv")
IRRELEVANT_OUTPUT_PATH = Path("data/processed/comments_irrelevant.csv")


RELEVANT_KEYWORDS = [
    "mobil",
    "dinas",
    "mobil dinas",
    "pengadaan",
    "gubernur",
    "kaltim",
    "kalimantan timur",
    "anggaran",
    "dana",
    "miliar",
    "milyar",
    "milyaran",
    "8 5",
    "8,5",
    "delapan",
    "pejabat",
    "fasilitas",
    "pemerintah",
    "kebijakan",
    "uang rakyat",
    "boros",
    "mahal",
    "mewah",
    "prioritas",
]


IRRELEVANT_KEYWORDS = [
    "dandanan",
    "dandanannya",
    "ketoprak",
    "ondel",
    "cantik",
    "glamor",
    "style",
    "taste",
    "class",
    "spamzilla",
    "expireddomains",
]


def is_relevant(text):
    text = str(text).lower()

    has_relevant = any(keyword in text for keyword in RELEVANT_KEYWORDS)
    has_irrelevant = any(keyword in text for keyword in IRRELEVANT_KEYWORDS)

    if has_irrelevant and not any(
        keyword in text for keyword in ["mobil", "dinas", "pengadaan", "gubernur", "kaltim"]
    ):
        return False

    return has_relevant


def main():
    df = pd.read_csv(INPUT_PATH)

    if "clean_text" not in df.columns:
        raise ValueError("Kolom 'clean_text' tidak ditemukan. Jalankan preprocess.py terlebih dahulu.")

    df["is_relevant"] = df["clean_text"].apply(is_relevant)

    relevant_df = df[df["is_relevant"] == True].copy()
    irrelevant_df = df[df["is_relevant"] == False].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    relevant_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    irrelevant_df.to_csv(IRRELEVANT_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Filter relevansi selesai.")
    print(f"Total data awal: {len(df)}")
    print(f"Data relevan: {len(relevant_df)}")
    print(f"Data tidak relevan: {len(irrelevant_df)}")
    print(f"File relevan disimpan di: {OUTPUT_PATH}")
    print(f"File tidak relevan disimpan di: {IRRELEVANT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()