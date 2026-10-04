import pandas as pd
import torch
from pathlib import Path
from transformers import pipeline


INPUT_PATH = Path("data/processed/comments_relevant.csv")
OUTPUT_PATH = Path("data/results/indobert_sentiment_result.csv")

MODEL_PATH = "models/indobert_sentiment"


def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    texts = df["clean_text"].fillna("").astype(str).tolist()

    print("Loading fine-tuned IndoBERT model...")

    device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        device=device,
        truncation=True,
        max_length=512
    )

    print("Predicting sentiment...")

    results = classifier(
        texts,
        batch_size=16
    )

    df["sentiment_label"] = [r["label"].lower() for r in results]
    df["sentiment_score"] = [r["score"] for r in results]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nPrediksi selesai.")
    print(f"Hasil disimpan di: {OUTPUT_PATH}")

    print("\nDistribusi sentimen:")
    print(df["sentiment_label"].value_counts())


if __name__ == "__main__":
    main()