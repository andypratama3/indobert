import pandas as pd
from pathlib import Path

# ============================================
# INPUT DATA TIKTOK
# ============================================

INPUT_PATH = Path(
    "data/raw/tiktok_comments_mobil_dinas_kaltim.csv"
)

# ============================================
# OUTPUT SAMPLE
# ============================================

OUTPUT_PATH = Path(
    "data/annotation/sample_500_tiktok.csv"
)


def main():

    print("=" * 50)
    print("MEMULAI SAMPLING DATA TIKTOK")
    print("=" * 50)

    # ============================================
    # LOAD DATA
    # ============================================

    df = pd.read_csv(INPUT_PATH)

    print(f"Jumlah data awal: {len(df)}")

    # ============================================
    # DETEKSI KOLOM KOMENTAR
    # ============================================

    if "comment" in df.columns:
        text_col = "comment"

    elif "comment_text" in df.columns:
        text_col = "comment_text"

    elif "text" in df.columns:
        text_col = "text"

    elif "content" in df.columns:
        text_col = "content"

    else:
        print(df.columns.tolist())
        raise ValueError(
            "Kolom komentar tidak ditemukan"
        )

    print(f"Kolom komentar digunakan: {text_col}")

    # ============================================
    # HAPUS DATA KOSONG
    # ============================================

    df = df.dropna(
        subset=[text_col]
    )

    # ============================================
    # HAPUS DUPLIKAT
    # ============================================

    df = df.drop_duplicates(
        subset=[text_col]
    )

    print(
        f"Jumlah data setelah cleaning: "
        f"{len(df)}"
    )

    # ============================================
    # AMBIL SAMPLE 500
    # ============================================

    sample_size = min(
        500,
        len(df)
    )

    sample_df = df.sample(
        n=sample_size,
        random_state=42
    )

    print(
        f"Jumlah sample diambil: "
        f"{sample_size}"
    )

    # ============================================
    # SIMPAN
    # ============================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sample_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSAMPLING BERHASIL")

    print(
        f"File output:\n{OUTPUT_PATH}"
    )

    print(
        "\n500 sample komentar TikTok "
        "siap untuk proses anotasi."
    )


if __name__ == "__main__":
    main()