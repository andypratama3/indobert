import pandas as pd
import torch
from pathlib import Path
from transformers import pipeline

# ============================
# PATH
# ============================

INPUT_PATH = Path(
    "data/processed/comments_clean.csv"
)

OUTPUT_PATH = Path(
    "data/results/indobert_absa_result.csv"
)

MODEL_PATH = (
    "models/indobert_aspect_sentiment_cv/fold_6"
)

# ============================
# MAIN
# ============================

def main():

    print("=" * 60)
    print("INDOBERT ABSA PREDICTION")
    print("=" * 60)

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_PATH)

    texts = (
        df["clean_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    device = (
        0 if torch.cuda.is_available()
        else -1
    )

    print("Loading model Fold 6...")

    classifier = pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        device=device,
        truncation=True,
        max_length=256,
    )

    print("Predicting...")

    results = classifier(
        texts,
        batch_size=16,
    )

    df["predicted_label"] = [
        r["label"]
        for r in results
    ]

    df["prediction_score"] = [
        round(r["score"],4)
        for r in results
    ]

    # ============================
    # Pisahkan aspek & sentimen
    # ============================

    df["predicted_aspect"] = (
        df["predicted_label"]
        .str.rsplit("_",n=1)
        .str[0]
    )

    df["predicted_sentiment"] = (
        df["predicted_label"]
        .str.rsplit("_",n=1)
        .str[1]
    )

    aspect_map = {

        "transparansi":
            "Transparansi",

        "akuntabilitas":
            "Akuntabilitas",

        "efektivitas_efisiensi":
            "Efektivitas dan Efisiensi",

        "responsivitas":
            "Responsivitas",

    }

    sentiment_map = {

        "positif":
            "Positif",

        "negatif":
            "Negatif",

    }

    df["predicted_aspect"] = (
        df["predicted_aspect"]
        .replace(aspect_map)
    )

    df["predicted_sentiment"] = (
        df["predicted_sentiment"]
        .replace(sentiment_map)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===================================")
    print("PREDIKSI SELESAI")
    print("===================================")

    print(
        f"\nTotal komentar : {len(df)}"
    )

    print(
        f"\nHasil disimpan :\n{OUTPUT_PATH}"
    )

    print("\nDistribusi Aspek")

    print(
        df["predicted_aspect"]
        .value_counts()
    )

    print("\nDistribusi Sentimen")

    print(
        df["predicted_sentiment"]
        .value_counts()
    )

    print("\nDistribusi Aspek x Sentimen")

    print(
        pd.crosstab(
            df["predicted_aspect"],
            df["predicted_sentiment"]
        )
    )


if __name__ == "__main__":
    main()