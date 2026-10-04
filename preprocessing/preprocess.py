import re
import pandas as pd
from pathlib import Path


RAW_PATH = Path("data/raw/tiktok_comments_mobil_dinas_kaltim.csv")
OUTPUT_PATH = Path("data/processed/comments_clean.csv")


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Hapus URL dan mention
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)

    # Hashtag: hapus tanda #, tapi pertahankan katanya
    text = re.sub(r"#", " ", text)

    # Ubah angka penting agar konteks tetap terbaca
    text = text.replace("8,5", "8 5")
    text = text.replace("8.5", "8 5")

    # Hapus karakter selain huruf, angka, dan spasi
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)

    # Rapikan spasi
    text = re.sub(r"\s+", " ", text).strip()

    slang = {
        "ga": "tidak",
        "gak": "tidak",
        "gk": "tidak",
        "nggak": "tidak",
        "tdk": "tidak",
        "bgt": "banget",
        "yg": "yang",
        "dgn": "dengan",
        "utk": "untuk",
        "krn": "karena",
        "dr": "dari",
        "sm": "sama",
        "aja": "saja",
        "dlm": "dalam",
        "td": "tadi",
        "tp": "tapi",
        "klo": "kalau",
        "kl": "kalau",
        "skrg": "sekarang",
        "krna": "karena",
    }

    words = text.split()
    words = [slang.get(word, word) for word in words]

    return " ".join(words)


def find_text_column(df):
    candidates = [
        "comment_text",
        "text",
        "tweet",
        "content",
        "full_text",
        "reply_text",
        "komentar",
        "clean_text",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(f"Kolom teks tidak ditemukan. Kolom tersedia: {list(df.columns)}")


def main():
    df = pd.read_csv(RAW_PATH)
    total_input = len(df)

    text_col = find_text_column(df)

    df["original_text"] = df[text_col].astype(str)
    df["clean_text"] = df["original_text"].apply(clean_text)

    df = df[df["clean_text"].str.len() > 10]
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Preprocessing selesai.")
    print(f"Input data: {total_input}")
    print(f"Output data: {len(df)}")
    print(f"File tersimpan di: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()