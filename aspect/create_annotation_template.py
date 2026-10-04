import pandas as pd
from pathlib import Path

# ============================================
# INPUT HASIL SAMPLING TIKTOK
# ============================================

INPUT_PATH = Path(
    "data/annotation/sample_500_tiktok.csv"
)

# ============================================
# OUTPUT TEMPLATE ANOTASI
# ============================================

OUTPUT_PATH = Path(
    "data/annotation/aspect_sentiment_annotation_template.csv"
)

ASPECT_LABELS = [
    "transparansi",
    "akuntabilitas",
    "efektivitas_efisiensi",
    "responsivitas",
]

SENTIMENT_LABELS = [
    "positif",
    "negatif",
]


def main():

    print(
        "MEMBUAT TEMPLATE ANOTASI ASPEK & SENTIMEN"
    )

    # ============================================
    # LOAD DATA
    # ============================================

    df = pd.read_csv(INPUT_PATH)

    print(f"Jumlah data ditemukan: {len(df)}")

    # ============================================
    # TEMPLATE OUTPUT
    # ============================================

    output = pd.DataFrame()

    output["id"] = range(
        1,
        len(df) + 1
    )

    # Username komentator
    output["username"] = df["comment_username"]

    # Isi komentar
    output["original_text"] = (
        df["comment_text"]
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.slice(0, 500)
    )

    # Link profil komentator
    output["reply_url"] = df[
        "comment_profile_url"
    ]

    # Link video TikTok sumber komentar
    output["source_tiktok"] = df[
        "post_url"
    ]

    # ============================================
    # KOLOM ANOTASI
    # ============================================

    output["annotator_aspect"] = ""
    output["ai_aspect"] = ""
    output["researcher_aspect"] = ""
    output["final_aspect"] = ""

    output["annotator_sentiment"] = ""
    output["ai_sentiment"] = ""
    output["researcher_sentiment"] = ""
    output["final_sentiment"] = ""

    output["final_label"] = ""

    output["notes"] = ""

    # ============================================
    # URUTAN KOLOM
    # ============================================

    output = output[
        [
            "id",
            "username",
            "original_text",
            "reply_url",
            "source_tiktok",

            "annotator_aspect",
            "ai_aspect",
            "researcher_aspect",
            "final_aspect",

            "annotator_sentiment",
            "ai_sentiment",
            "researcher_sentiment",
            "final_sentiment",

            "final_label",
            "notes"
        ]
    ]

    # ============================================
    # SIMPAN
    # ============================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\nTemplate anotasi berhasil dibuat"
    )

    print(
        f"Jumlah data: {len(output)}"
    )

    print(
        f"Lokasi file:\n{OUTPUT_PATH}"
    )

    print("\nLabel aspek:")

    for label in ASPECT_LABELS:
        print(f"- {label}")

    print("\nLabel sentimen:")

    for label in SENTIMENT_LABELS:
        print(f"- {label}")

    print(
        "\nSelesai membuat template anotasi hybrid."
    )


if __name__ == "__main__":
    main()