"""
Figures describing the ANNOTATED sample, not the model predictions.

Use `visualization.visualize_absa` for anything derived from
`indobert_absa_result_oof.csv`.

Every output name here carries an `_annotasi` suffix. The two scripts used to
write `distribusi_aspek.png` and `distribusi_sentimen.png` to the same paths with
different contents, so whichever ran last silently overwrote the other and the
figure on disk depended on stage order. Distinct filenames make that impossible.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import config

# ==========================
# PATH
# ==========================
INPUT_PATH = config.ANNOTATION_DATASET
RAW_PATH = config.RAW_COMMENTS
OUTPUT_DIR = config.VISUALIZATION_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================
# Gambar 5.1
# Jumlah Data Penelitian
# ==========================
def plot_dataset_summary():

    labels = [
        "Hasil Scraping",
        "Dataset Penelitian"
    ]

    # Counted from the files, never hardcoded. These previously read 1081 and
    # 800 as literals, so the figure would have kept showing the old numbers
    # after either dataset changed.
    values = [
        len(pd.read_csv(RAW_PATH)),
        len(pd.read_csv(INPUT_PATH)),
    ]

    plt.figure(figsize=(7,5))

    bars = plt.bar(labels, values)

    plt.title("Jumlah Data Penelitian")
    plt.ylabel("Jumlah Komentar")

    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height()+10,
            int(bar.get_height()),
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "jumlah_data.png",
        dpi=300
    )

    plt.close()


# ==========================
# Gambar 5.2
# Distribusi Aspect
# ==========================
def plot_aspect(df):

    counts = df["final_aspect"].value_counts()

    plt.figure(figsize=(8,5))

    counts.plot(kind="bar")

    plt.title("Distribusi Final Aspect")

    plt.xlabel("Aspect")

    plt.ylabel("Jumlah Komentar")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "distribusi_aspek_annotasi.png",
        dpi=300
    )

    plt.close()


# ==========================
# Gambar 5.3
# Distribusi Sentiment
# ==========================
def plot_sentiment(df):

    counts = df["final_sentiment"].value_counts()

    plt.figure(figsize=(7,5))

    counts.plot(kind="bar")

    plt.title("Distribusi Final Sentiment")

    plt.xlabel("Sentiment")

    plt.ylabel("Jumlah Komentar")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "distribusi_sentimen_annotasi.png",
        dpi=300
    )

    plt.close()


# ==========================
# Gambar 5.4
# Distribusi Final Label
# ==========================
def plot_final_label(df):

    counts = df["final_label"].value_counts()

    plt.figure(figsize=(10,6))

    counts.plot(kind="bar")

    plt.title("Distribusi Final Label")

    plt.xlabel("Final Label")

    plt.ylabel("Jumlah Komentar")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "distribusi_final_label.png",
        dpi=300
    )

    plt.close()


def main():

    df = pd.read_csv(INPUT_PATH)

    plot_dataset_summary()

    plot_aspect(df)

    plot_sentiment(df)

    plot_final_label(df)

    print("="*50)
    print("Visualisasi berhasil dibuat.")
    print("="*50)
    print(f"Folder output : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()